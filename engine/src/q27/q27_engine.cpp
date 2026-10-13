// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B GPU engine with layer placement — see hive/q27/q27_engine.h. The forward follows the CPU engine (q27_cpu_engine.cpp) step by
//   step: the same rounding points, the same row / segment bookkeeping for prefill, decode and verify, and the same MTP entry rules.
#include "hive/q27/q27_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <thread>

#include "hive/clock.h"
#include "hive/common.h"
#include "hive/cublas_ops.h"
#include "hive/model_kernels.h"
#include "hive/q27/q27_dflash.h"
#include "hive/q27/q27_kernels.h"
#include "hive/q27/q27_lowp.h"
#include "hive/safetensors.h"
#include "nlohmann/json.hpp"

namespace hive::q27 {

using k::GMat;
using k::WFmt;

namespace {
constexpr size_t kMiB = 1ull << 20;
constexpr int kMaxRows = 32;   // decode / verify rows per forward (hived max batch 16; MTP verify rows of every part together)
constexpr int kMaxParts = 16;  // verify parts (sequences) per forward: one recurrent-state snapshot slot each
inline float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
inline uint16_t f2b(float x) {  // round to nearest even
  uint32_t u; memcpy(&u, &x, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)(u >> 16);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
size_t env_mb(const char* n, size_t dflt) { const char* v = getenv(n); return v && atof(v) > 0 ? (size_t)(atof(v) * kMiB) : dflt; }
const std::string P = "model.language_model.";

struct GpuLayer {
  bool on = false, attn = false;
  DevBuf mem;                         // every matrix and vector of the layer, one allocation
  GMat qkvz, ba, lin_out, qkv, o, gate, up, down;
  const bf16 *ln1 = nullptr, *ln2 = nullptr, *lin_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
  const float *conv_w = nullptr, *A_log = nullptr, *dt_bias = nullptr;
};

// Bytes a list of tensors needs in VRAM (FP8 scales become fp32, BF16 vectors stay bf16, conv / A_log / dt_bias become fp32)
struct Planner {
  Checkpoint& ck;
  size_t mat(const std::string& b) {
    if (ck.has(b + ".weight_packed")) return ck.get(b + ".weight_packed").nbytes + ck.get(b + ".weight_scale").nbytes + 256;
    const TensorInfo& t = ck.get(b + ".weight");
    return t.nbytes + (t.dtype == "F8_E4M3" ? (size_t)t.shape[0] * 4 : 0) + 256;
  }
  size_t vec(const std::string& n, int esize = 2) { return (size_t)ck.get(n).numel() * esize + 256; }
  // placement only: the device bytes a matrix brings beyond its own upload — the cuBLASLt-tiled scale copy of an NVFP4 matrix
  //   (HIVE_Q27_PREFILL_QUANT / _DECODE_QUANT) and the NVFP4 decode copy of an FP8 matrix (HIVE_Q27_W4, + its tiled scales with
  //   PREFILL_QUANT). Measured 2026-10-12, 32 GB cap (ctx 262144, DFlash2, pq + W4): the plan counted the checkpoint bytes
  //   only, placed all 64 layers in a 32 GB cap and hived measured 33330 MiB after the load ("the weights alone do not fit") — the W4
  //   copies are 6.3 GB and the tiled NVFP4 scales ~0.9 GB.
  bool w4 = false, pq = false, dq = false;
  size_t extra(const std::string& b, bool decode_copy = true) {
    if (ck.has(b + ".weight_packed")) {
      const TensorInfo& w = ck.get(b + ".weight_packed");
      return (pq || dq) ? lowp::swz_bytes((int)w.shape[0], (int)w.shape[1] * 2) + 256 : 0;
    }
    const TensorInfo& t = ck.get(b + ".weight");
    if (!w4 || !decode_copy || t.dtype != "F8_E4M3" || t.shape.size() != 2) return 0;
    const size_t N = (size_t)t.shape[0], K = (size_t)t.shape[1];
    return N * K / 2 + N * K / 16 + 512 + (pq ? lowp::swz_bytes((int)N, (int)K) + 256 : 0);
  }
};
}  // namespace

namespace { int64_t s_plan_ctx = 0; }   // Q27Engine::set_plan_ctx (KV plan)
void Q27Engine::set_plan_ctx(int64_t max_ctx) { s_plan_ctx = max_ctx; }

struct Q27Engine::Impl {
  Q27Config c;
  std::string dir;
  std::unique_ptr<cpu::Q27CpuEngine> cpu;
  Placement pl;
  std::vector<GpuLayer> L;
  std::vector<int> lin_index, attn_index;
  GpuLayer mtpL;
  DevBuf head_mem, mtp_mem;
  GMat lm_head, mtp_fc;
  // HIVE_Q27_DRAFT_VOCAB=FILE (little-endian int32 token ids, tools/build_draft_vocab.py; unset / "" / "0" = whole vocabulary): the MTP
  //   draft head scores only these lm_head rows (a copy) — verify scores every draft against the full head, so outputs never change
  DevBuf draft_mem;
  GMat draft_head;
  std::vector<int32_t> draft_ids;
  DevBuf draft_ids_d;
  const bf16 *final_norm = nullptr, *mtp_en = nullptr, *mtp_hn = nullptr, *mtp_norm = nullptr;
  int mtp_k = 0, chunk = 512;
  cudaStream_t st = nullptr;
  std::unique_ptr<Blas> blas;
  cublasHandle_t cbh = nullptr;   // bf16-output GEMMs of the prefill (own handle on the engine stream)
  // HIVE_Q27_PREFILL_QUANT=1: W8A8 / W4A4 prefill GEMMs (hive/q27/q27_lowp.h) — lossy, off by default
  bool pq = false, pq_warned = false;
  bool dq = false;                    // HIVE_Q27_DECODE_QUANT=1: W8A8 decode / verify GEMMs for the FP8 matrices (lossy, off by default)
  DevBuf dq_x, dq_s, dq_xs4;
  std::unique_ptr<lowp::Lt> lt;
  DevBuf pq_x, pq_s, pq_swz, pq_dyn;
  std::vector<DevBuf> swz_store, w4_store;
  void prep_nvfp4(Checkpoint& ck, const std::string& base, GMat& g);
  DevBuf rope_inv, rope_axis, rope_tab, attn_part;
  // work buffers (rows ≤ chunk)
  DevBuf h, x, y, proj, gy, qk, gb, go, gout, q, attf, ao, gu, acc, deq, logits_d, coords_d, rows_d, segs_d, cat, fh;
  size_t deq_bytes = 0;
  std::vector<uint16_t> hb;          // host bf16 rows (MTP entry embeddings; not used across a layer yield)
  std::vector<cpu::Q27CpuEngine::EmbedOverride> emb_over;
  int64_t emb_base = 0;
  GpuStats st_;
  // verify
  bool verify = false;
  bool host_logits = true;   // set_host_logits (q27_engine.h)
  long captures = 0;         // CUDA graphs captured so far (graph_captures: hived's gate drops step samples that included one)
  // HIVE_Q27_VSNAP=1: the old verify (state snapshot per part, verify writes the state, rollback replays from the snapshot). Default: the
  //   verify only reads the recurrent state and the rollback always replays the kept rows in place — per stream and step 3 state passes
  //   instead of 5 (read · read + write, vs read + snapshot + write · read + write), and no 16 × 144 MiB snapshot slots (2026-10-10, nsys
  //   S4 k7: gdn_rec 1.08 + gdn_replay 2.56 ms per round). The same gdn_rows arithmetic → bit-identical states and tokens.
  const bool vsnap = getenv("HIVE_Q27_VSNAP") && strcmp(getenv("HIVE_Q27_VSNAP"), "1") == 0;
  struct VPart { Q27Seq* s; int64_t pos0; int M, r0; };
  std::vector<VPart> vparts;
  std::vector<int32_t> verify_ids;
  std::vector<DevBuf> snap_conv, snap_state;   // conv [row slot · n_lin_gpu + j] · state before the verify [part · n_lin_gpu + j]
  DevBuf vq_y, vq_qk, vq_gb, rp_items;
  std::vector<int> gpu_lin;                    // linear-layer indices on the GPU (snapshot order)
  DevBuf snap_ptrs;                            // device pointer table per layer: [n_lin][conv 8 | state 8]
  std::function<void(int, const float*, int)>* layer_hook = nullptr;
  std::function<void(int, float*, int)>* layer_inject = nullptr;
  std::function<void(int)>* layer_boundary = nullptr;
  std::function<bool()>* layer_boundary_due = nullptr;
  int depth = 0;

  void load(bool want_mtp);
  void load_layer(Checkpoint& ck, const std::string& p, bool attn, GpuLayer& G);
  GMat upload_mat(Checkpoint& ck, const std::vector<std::string>& bases, uint8_t*& cur);
  // pieces
  void lin(const GMat& W, const bf16* xin, int ldx, int M, bf16* yb, int ldy, float* yf = nullptr);
  // device row tables of one forward: per attention layer index R AttnRow, per linear layer index kSegs GdnSeg, then R AttnRow for the
  //   MTP layer; built on the host (KV grown first) and uploaded in one copy (tab_pin → tab_d)
  static constexpr int kSegs = 64;
  DevBuf tab_d;
  uint8_t* tab_pin = nullptr;
  size_t tab_bytes = 0;
  int tab_rows = 0;
  size_t attn_tab_off(int ai) const { return (size_t)ai * tab_rows * sizeof(k::AttnRow); }
  size_t seg_tab_off(int li) const { return (size_t)c.n_attn() * tab_rows * sizeof(k::AttnRow) + (size_t)li * kSegs * sizeof(k::GdnSeg); }
  size_t mtp_tab_off() const { return seg_tab_off(c.n_lin()); }
  int build_tables(std::vector<Q27Seq*>& seqs, const std::vector<int64_t>& pos, std::vector<uint8_t>& tab);   // returns segment count
  void upload_tables(const std::vector<uint8_t>& tab);
  // pend_in: hd still lacks the previous layer's MLP output (in y) — added by this layer's first norm; pend_out: leave this layer's MLP
  //   output in y instead of adding it (the next add_norm adds it). Same values as the unfused order.
  void layer_gpu(int l, const GpuLayer& G, const k::AttnRow* rows_dev, const k::GdnSeg* segs_dev, int nsegs, bf16* hd, int M,
                 bool pend_in = false, bool pend_out = false, const int32_t* coords = nullptr);
  DevBuf ids_d;                      // token ids of a forward / of MTP entries (device embedding gathers)
  DevBuf emb_d;                      // the token embedding table in VRAM (when it fits): device-side MTP draft chain
  // the table the device gathers read: emb_d, or (under a cap, when it does not fit next to the KV plan) the CPU engine's host table mapped into
  //   the device address space (zero-copy reads of a few rows per step; same values, the drafter keeps its CUDA graphs) — null = host gathers
  const bf16* emb_ptr = nullptr;
  void gpu_step(int M, int nsegs, bf16* hd, int r0, int nr, const bf16* norm_w, bool fh_copy, float* dst);   // every layer + head into dst (pinned host)
  std::map<const void*, int> host_dst;   // registered (pinned) caller logits buffers → graph-key index
  // CUDA graphs of whole decode / verify steps (every layer on the GPU, no hooks), per (rows, segments, logits rows)
  std::map<int64_t, cudaGraphExec_t> graphs;
  bool use_graphs = true;
  // prefill of one sequence (rows = consecutive positions from pf_pos0): the tensor-core attention reads that sequence's caches
  bool pf_on = false;
  int pf_pos0 = 0;
  Q27Seq* pf_seq = nullptr;
  DevBuf dr_buf;                      // MTP draft rows: current stream · embedding · block output
  DevBuf small_d, me_emb, me_hm, me_cd, eh_buf;  // argmax result · MTP entry work rows (8 rows) · their coordinates
  float* small_pin = nullptr;
  float* logits_pin = nullptr;
  void forward(std::vector<Q27Seq*> seqs, std::vector<int64_t> pos, const int32_t* ids, bool prefill, float* logits, bool all_logits);
  void set_rope(std::vector<Q27Seq*>& seqs, std::vector<int64_t>& pos, bool prefill, std::vector<int32_t>& coords);
  void embed(const int32_t* ids, int M, bool prefill, bf16* out);
  void head(const bf16* hd, int M, const bf16* norm_w, float* logits_host);
  void grow_kv(Q27Seq::Kv& kv, int64_t tokens);
  std::function<bool()>* kv_reclaim = nullptr;
  // KV buffers released by growth, kept for the next growth instead of cudaFree + cudaMalloc (measured 2026-10-11 nsys, cold 8K prefill in
  //   2048-row chunks: every chunk regrew 16 layers' K / V — cudaMalloc 31 + cudaFree 21 ms of GPU idle per request, 17 ms on a cold 1K).
  //   Same stream for every use, so a recycled buffer is only written after the copy out of it. Bounded (kv_free_max); emptied first when VRAM runs out.
  std::vector<DevBuf> kv_free;
  size_t kv_free_bytes = 0;
  // cap: 1/16 of the VRAM left after the weights, at most 4 GB, none below 8 GB left (the VRAM-budget profiles of 8-24 GB keep nothing idle)
  size_t kv_free_max() const {
    const size_t left = pl.cap > pl.weights ? pl.cap - pl.weights : 0;
    return left < ((size_t)8 << 30) ? 0 : std::min<size_t>((size_t)4 << 30, left / 16);
  }
  bool kv_take(DevBuf& b, size_t n) {   // smallest cached buffer ≥ n and ≤ 2n
    int best = -1;
    for (int i = 0; i < (int)kv_free.size(); ++i)
      if (kv_free[i].n >= n && kv_free[i].n <= 2 * n && (best < 0 || kv_free[i].n < kv_free[best].n)) best = i;
    if (best < 0) return false;
    kv_free_bytes -= kv_free[best].n;
    b = std::move(kv_free[best]);
    kv_free.erase(kv_free.begin() + best);
    return true;
  }
  void kv_give(DevBuf&& b) {
    if (!b.p) return;
    const size_t mx = kv_free_max();
    while (!kv_free.empty() && kv_free_bytes + b.n > mx) { kv_free_bytes -= kv_free.front().n; kv_free.erase(kv_free.begin()); }
    if (b.n > mx) { b = DevBuf(); return; }
    kv_free_bytes += b.n;
    kv_free.push_back(std::move(b));
  }
  void alloc_serving(DevBuf& b, size_t n, const char* what);
  // MTP
  // MTP hidden input definition (A/B): HIVE_Q27_MTP_INPUT=post feeds the final-normed main stream, HIVE_Q27_MTP_CHAIN=post feeds a chained
  //   draft the mtp.norm-ed block output; default pre for both (the oracle contract)
  bool mtp_post_main = false, mtp_post_chain = false;
  void mtp_input(const bf16* emb, const bf16* hrows, int n, bf16* out, const bf16* prenorm = nullptr);
  void mtp_entries(std::vector<Q27Seq*>& seqs, std::vector<int64_t>& pos, const bf16* hrows, const int32_t* next_ids, const std::vector<int64_t>& ov_rows);
  void mtp_after_forward(std::vector<Q27Seq*>& seqs, std::vector<int64_t>& pos, const int32_t* ids, bool prefill);
  int mtp_draft(Q27Seq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  // DFlash2 drafter (HIVE_Q27_DFLASH=<draft checkpoint dir>; hive/q27/q27_dflash.h) in place of the MTP block: same draft / verify /
  //   rollback interface. Context = the target's hidden of 5 layers, captured in every forward (gpu_step / layer loop) and turned into the
  //   drafter's context K / V right after it (df_ingest, every row; rejected verify rows are overwritten later — ring slots by position).
  struct DfLayer { GMat qkv, o, gu, down, kpa, kpm; const bf16 *ln1 = nullptr, *ln2 = nullptr, *qn = nullptr, *kn = nullptr; const float *base_a = nullptr, *base_m = nullptr; };
  bool dflash = false;
  bool df_cpu_capture = false;
  bool kv_tier = false, w4_dropped = false;   // HIVE_Q27_KV_TIER (see the placement)
  bool w4_partial = false;                    // KV tier: the plan dropped the W4 copies, the load adds those that fit (see the W4 copies)
  size_t kv_budget = 0;                       // KV tier: VRAM bytes for KV buffers
  DevBuf kv_stage;                            // KV tier: a host-tier layer's K | V copied here for the prefill attention
  const bool kv_host_grow = !(getenv("HIVE_Q27_KV_HOST_GROW") && strcmp(getenv("HIVE_Q27_KV_HOST_GROW"), "0") == 0);
  // HIVE_Q27_KVLOG=1 (diagnostics): KV tier allocations and the prefill's KV costs per prompt on stderr
  const bool kvlog = getenv("HIVE_Q27_KVLOG") && strcmp(getenv("HIVE_Q27_KVLOG"), "1") == 0;
  struct KvLog { long host_allocs = 0, dev_allocs = 0, stage_copies = 0, stage_miss = 0, grows = 0; double host_ms = 0, dev_ms = 0, grow_ms = 0;
                 size_t host_b = 0, grow_copy_b = 0, stage_b = 0; } kl, kl0;
  double kl_t0 = 0;
  void alloc_kv(KvBuf& b, size_t bytes, size_t freed_dev);      // drafter on while some layers run on the CPU: their hidden rows are captured from the host stream
  bool stage_alloc(size_t rb, int ai);   // KV tier: the prefill staging copy (see the definition)
  const Q27Seq* stage_seq = nullptr;     // the sequence whose prefill holds kv_stage (freed when it decodes: its prompt is done)
  bool kv_pressure = false;              // alloc_kv is reclaiming for the VRAM KV budget (release_kv then frees every VRAM KV buffer)
  // HIVE_Q27_W4_PARTIAL=1: the partial W4 copies (w4_part_store) give their VRAM to the KV / the staging copy when it runs out (w4_yield)
  std::vector<DevBuf> w4_part_store;
  bool w4_yield();
  // KV tier, prefill: the staging copy of the next host-tier layer's earlier rows runs on its own stream while the layers in front of it compute
  //   (HIVE_Q27_STAGE_PREFETCH=0: the copy in line on the engine stream). stage_ai = the attention layer whose rows [0, pf_pos0) are in flight /
  //   in kv_stage for the current chunk (-1: none); its rows of this chunk follow on the engine stream after attn_prep wrote them.
  const bool stage_pf = !(getenv("HIVE_Q27_STAGE_PREFETCH") && strcmp(getenv("HIVE_Q27_STAGE_PREFETCH"), "0") == 0);
  cudaStream_t cst = nullptr;
  cudaEvent_t ev_ready = nullptr, ev_used = nullptr;
  int stage_ai = -1;
  void stage_prefetch(int from_ai, int passed_ai, int M);
  bool dsp = false;                 // DSpark drafter (DFlash backbone + Markov bigram bias + confidence head), else DFlash2
  int df_B = df::kBlock, df_out0 = 1, df_window = df::kWindow;   // rows per block, first output row, context window (0 = whole context)
  float df_mscale = 1.f;            // YaRN attention factor (DSpark)
  GMat df_confh;                    // DSpark confidence head, hidden part [1, 5120]
  const bf16* df_confw = nullptr;   // its Markov-latent part [256]
  float df_confb = 0.f;
  DevBuf dch;                       // confidence head hidden dot per output row
  void df_ensure_ring(Q27Seq& s, int64_t need_pos);
  std::string df_dir;
  // target layer → capture slot (−1 = none); −1 everywhere until the drafter is loaded — HIVE_Q27_DFLASH set while the drafter did not fit
  //   (pl.mtp_gpu false) left it uninitialized and the forward captured into an unallocated buffer (illegal address, 32 GB cap, 2026-10-12)
  int df_slot[64] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  std::vector<DevBuf> df_store;
  GMat df_fc, df_kvall, df_hp;
  const bf16 *df_hn = nullptr, *df_norm = nullptr, *df_knorm = nullptr, *df_pred = nullptr, *df_succ = nullptr;
  DfLayer dfl[df::kLayers];
  DevBuf dtk, dedge, df_inv, df_cap, df_f, df_f2, df_kv, df_rows, df_ids, dx, dxn, dy, dz, dproj, dq_, dbk, dbv, dattn, ddyn, dgu, dcidx, dcval, dhp, dout;
  df::CtxRow* df_rows_pin = nullptr;
  int32_t* df_out_pin = nullptr;
  int mask_id = 0;
  float df_eps = 1e-6f;
  void df_load(size_t& bytes_out, bool load);
  void df_ingest(std::vector<Q27Seq*>& seqs, std::vector<int64_t>& pos, const bf16* capb);
  int df_draft(Q27Seq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  static constexpr int kDfSeqs = 16;   // sequences per stacked draft pass
  DevBuf df_seqs, df_logits;
  std::map<int, cudaGraphExec_t> df_graphs;   // draft pass per stacked sequence count
  df::BlkSeq* df_seq_pin = nullptr;
  int32_t* df_ids_pin = nullptr;
  void df_draft_batch(Q27Seq* const* seqs, const int32_t* toks, int S, int k, int32_t* drafts, float* conf, int* n_out);   // drafts / conf [S, 7]
};

// ---- loading --------------------------------------------------------------------------------------------------------------------------------
GMat Q27Engine::Impl::upload_mat(Checkpoint& ck, const std::vector<std::string>& bases, uint8_t*& cur) {
  auto align = [&](size_t a) { cur = (uint8_t*)(((uintptr_t)cur + 255) & ~(uintptr_t)255); (void)a; };
  GMat g;
  align(0);
  if (ck.has(bases[0] + ".weight_packed")) {
    HIVE_CHECK(bases.size() == 1, "q27: NVFP4 matrices are not stacked");
    const TensorInfo& w = ck.get(bases[0] + ".weight_packed");
    const TensorInfo& s = ck.get(bases[0] + ".weight_scale");
    const TensorInfo& gs = ck.get(bases[0] + ".weight_global_scale");
    float gv; memcpy(&gv, gs.data, 4);
    g.fmt = WFmt::NVFP4; g.N = (int)w.shape[0]; g.K = (int)w.shape[1] * 2; g.g = 1.f / gv;
    CUDA_CHECK(cudaMemcpy(cur, w.data, w.nbytes, cudaMemcpyHostToDevice)); g.w = cur; cur += w.nbytes; align(0);
    CUDA_CHECK(cudaMemcpy(cur, s.data, s.nbytes, cudaMemcpyHostToDevice)); g.s = cur; cur += s.nbytes;
    return g;
  }
  int N = 0, K = 0;
  std::string fmt;
  uint8_t* w0 = cur;
  for (const auto& b : bases) {
    const TensorInfo& t = ck.get(b + ".weight");
    if (fmt.empty()) fmt = t.dtype;
    HIVE_CHECK(t.dtype == fmt && t.shape.size() == 2, "q27: stacked matrices must share one format: " + b);
    K = (int)t.shape[1]; N += (int)t.shape[0];
    CUDA_CHECK(cudaMemcpy(cur, t.data, t.nbytes, cudaMemcpyHostToDevice));
    cur += t.nbytes;
  }
  g.N = N; g.K = K; g.w = w0;
  if (fmt == "BF16") { g.fmt = WFmt::BF16; return g; }
  HIVE_CHECK(fmt == "F8_E4M3", "q27: unexpected weight format " + fmt);
  g.fmt = WFmt::FP8ROW;
  align(0);
  std::vector<float> sc;
  for (const auto& b : bases) {
    const TensorInfo& s = ck.get(b + ".weight_scale");
    const uint16_t* p = (const uint16_t*)s.data;
    for (int64_t i = 0; i < s.numel(); ++i) sc.push_back(b2f(p[i]));
  }
  CUDA_CHECK(cudaMemcpy(cur, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
  g.s = cur; cur += sc.size() * 4;
  return g;
}

void Q27Engine::Impl::prep_nvfp4(Checkpoint& ck, const std::string& base, GMat& g) {
  DevBuf b(lowp::swz_bytes(g.N, g.K));
  lowp::swizzle_scales(reinterpret_cast<const uint8_t*>(g.s), g.N, g.K, b.as<uint8_t>(), st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  g.s_swz = b.as<uint8_t>();
  swz_store.push_back(std::move(b));
  const TensorInfo& t = ck.get(base + ".input_global_scale");
  memcpy(&g.in_gs, t.data, 4);
}

void Q27Engine::Impl::load_layer(Checkpoint& ck, const std::string& p, bool attn, GpuLayer& G) {
  Planner pn{ck};
  std::vector<std::string> mats, bvec, fvec;   // matrices, bf16 vectors, fp32 vectors
  const std::string a = p + (attn ? "self_attn." : "linear_attn.");
  size_t bytes = pn.mat(p + "mlp.gate_proj") + pn.mat(p + "mlp.up_proj") + pn.mat(p + "mlp.down_proj") + pn.vec(p + "input_layernorm.weight") +
                 pn.vec(p + "post_attention_layernorm.weight");
  if (attn) bytes += pn.mat(a + "q_proj") + pn.mat(a + "k_proj") + pn.mat(a + "v_proj") + pn.mat(a + "o_proj") + pn.vec(a + "q_norm.weight") + pn.vec(a + "k_norm.weight");
  else bytes += pn.mat(a + "in_proj_qkv") + pn.mat(a + "in_proj_z") + pn.mat(a + "in_proj_b") + pn.mat(a + "in_proj_a") + pn.mat(a + "out_proj") +
                pn.vec(a + "conv1d.weight", 4) + pn.vec(a + "A_log", 4) + pn.vec(a + "dt_bias", 4) + pn.vec(a + "norm.weight");
  G.mem.alloc(bytes + 4096);
  uint8_t* cur = G.mem.as<uint8_t>();
  auto bvec_up = [&](const std::string& n) -> const bf16* {
    cur = (uint8_t*)(((uintptr_t)cur + 255) & ~(uintptr_t)255);
    const TensorInfo& t = ck.get(n);
    HIVE_CHECK(t.dtype == "BF16", "q27: expected a BF16 vector: " + n);
    CUDA_CHECK(cudaMemcpy(cur, t.data, t.nbytes, cudaMemcpyHostToDevice));
    const bf16* r = (const bf16*)cur; cur += t.nbytes; return r;
  };
  auto fvec_up = [&](const std::string& n) -> const float* {
    cur = (uint8_t*)(((uintptr_t)cur + 255) & ~(uintptr_t)255);
    const TensorInfo& t = ck.get(n);
    std::vector<float> v((size_t)t.numel());
    if (t.dtype == "BF16") for (size_t i = 0; i < v.size(); ++i) v[i] = b2f(((const uint16_t*)t.data)[i]);
    else { HIVE_CHECK(t.dtype == "F32", "q27: vector dtype " + n); memcpy(v.data(), t.data, v.size() * 4); }
    CUDA_CHECK(cudaMemcpy(cur, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    const float* r = (const float*)cur; cur += v.size() * 4; return r;
  };
  G.on = true; G.attn = attn;
  G.ln1 = bvec_up(p + "input_layernorm.weight"); G.ln2 = bvec_up(p + "post_attention_layernorm.weight");
  if (attn) {
    G.qkv = upload_mat(ck, {a + "q_proj", a + "k_proj", a + "v_proj"}, cur);
    G.o = upload_mat(ck, {a + "o_proj"}, cur);
    G.q_norm = bvec_up(a + "q_norm.weight"); G.k_norm = bvec_up(a + "k_norm.weight");
  } else {
    G.qkvz = upload_mat(ck, {a + "in_proj_qkv", a + "in_proj_z"}, cur);
    G.ba = upload_mat(ck, {a + "in_proj_b", a + "in_proj_a"}, cur);
    G.lin_out = upload_mat(ck, {a + "out_proj"}, cur);
    G.conv_w = fvec_up(a + "conv1d.weight"); G.A_log = fvec_up(a + "A_log"); G.dt_bias = fvec_up(a + "dt_bias");
    G.lin_norm = bvec_up(a + "norm.weight");
  }
  G.gate = upload_mat(ck, {p + "mlp.gate_proj"}, cur);
  G.up = upload_mat(ck, {p + "mlp.up_proj"}, cur);
  G.down = upload_mat(ck, {p + "mlp.down_proj"}, cur);
  HIVE_CHECK((size_t)(cur - G.mem.as<uint8_t>()) <= G.mem.n, "q27: layer arena overflow");
  if ((pq || dq) && G.gate.fmt == WFmt::NVFP4) { prep_nvfp4(ck, p + "mlp.gate_proj", G.gate); prep_nvfp4(ck, p + "mlp.up_proj", G.up); prep_nvfp4(ck, p + "mlp.down_proj", G.down); }
}

void Q27Engine::Impl::load(bool want_mtp) {
  const double t0 = hive::mono_ms();
  c = Q27Config::load(dir);
  if (const char* v = getenv("HIVE_Q27_CHUNK"); v && atoi(v) > 0) chunk = std::max(16, atoi(v));
  pq = getenv("HIVE_Q27_PREFILL_QUANT") && strcmp(getenv("HIVE_Q27_PREFILL_QUANT"), "1") == 0;
  dq = getenv("HIVE_Q27_DECODE_QUANT") && strcmp(getenv("HIVE_Q27_DECODE_QUANT"), "1") == 0;
  Checkpoint ck(dir);
  Planner pn{ck};
  lin_index.assign(c.n_layers, -1); attn_index.assign(c.n_layers, -1);
  int nl = 0, na = 0;
  for (int l = 0; l < c.n_layers; ++l) (c.is_attn[l] ? attn_index[l] = na++ : lin_index[l] = nl++);
  // ---- placement
  pn.w4 = getenv("HIVE_Q27_W4") && strcmp(getenv("HIVE_Q27_W4"), "1") == 0; pn.pq = pq; pn.dq = dq;
  // KV plan (default on; HIVE_Q27_KV_PLAN=0 = the old plan): every GPU attention layer also reserves the KV of one
  //   sequence at the plan's context (Q27Engine::set_plan_ctx = hived's --max-ctx, else config max_position), so a prompt up to that context
  //   never runs out of VRAM for KV at any budget; the layers that do not fit with their KV run on the CPU (weights and KV in host memory).
  //   Measured 2026-10-12: at a 24 GB cap the 27B placed every layer and its 32K / 64K prompts failed ("out of GPU memory for KV").
  // KV tier (default on under a VRAM cap, HIVE_Q27_KV_TIER=0 = the KV plan with whole layers on the CPU; replaces the KV plan there): the layers, the lm_head and the drafter are placed as if KV
  //   took no VRAM; KV buffers live in VRAM while the leftover budget lasts (kv_budget) and in pinned device-mapped host memory beyond it, so
  //   every context up to max_ctx fits and short chats keep the all-GPU speed
  const char* kvt = getenv("HIVE_Q27_KV_TIER");
  kv_tier = env_mb("HIVE_Q27_VRAM_MB", env_mb("HIVE_VRAM_LIMIT_MB", 0)) > 0 && !(kvt && strcmp(kvt, "0") == 0);   // default on under a cap
  const char* kvp = getenv("HIVE_Q27_KV_PLAN");
  const bool kv_plan = !kv_tier && !(kvp && strcmp(kvp, "0") == 0);
  const int64_t plan_ctx = s_plan_ctx > 0 ? s_plan_ctx : c.max_position;
  const size_t kv_layer = kv_plan ? (size_t)plan_ctx * 2 * c.kv_width() * 2 : 0;
  std::vector<size_t> lb(c.n_layers), w4x(c.n_layers, 0);
  for (int l = 0; l < c.n_layers; ++l) {
    const std::string p = P + "layers." + std::to_string(l) + ".";
    const std::string a = p + (c.is_attn[l] ? "self_attn." : "linear_attn.");
    size_t b = pn.mat(p + "mlp.gate_proj") + pn.mat(p + "mlp.up_proj") + pn.mat(p + "mlp.down_proj") + 4 * c.hidden;
    if (c.is_attn[l]) b += pn.mat(a + "q_proj") + pn.mat(a + "k_proj") + pn.mat(a + "v_proj") + pn.mat(a + "o_proj");
    else b += pn.mat(a + "in_proj_qkv") + pn.mat(a + "in_proj_z") + pn.mat(a + "in_proj_b") + pn.mat(a + "in_proj_a") + pn.mat(a + "out_proj") + 200000;
    std::vector<std::string> ms = {p + "mlp.gate_proj", p + "mlp.up_proj", p + "mlp.down_proj"};
    if (c.is_attn[l]) for (const char* n : {"q_proj", "k_proj", "v_proj", "o_proj"}) ms.push_back(a + n);
    else for (const char* n : {"in_proj_qkv", "in_proj_z", "out_proj"}) ms.push_back(a + n);   // (in_proj_b / _a are BF16)
    for (const auto& m : ms) b += pn.extra(m);
    if (pn.w4) {   // the W4-copy part of the layer (dropped under the KV tier when it keeps the drafter off, see below)
      pn.w4 = false;
      size_t b0 = 0;
      for (const auto& m : ms) b0 += pn.extra(m);
      pn.w4 = true;
      size_t bw = 0;
      for (const auto& m : ms) bw += pn.extra(m);
      w4x[l] = bw - b0;
    }
    if (c.is_attn[l]) b += kv_layer;
    lb[l] = b;
  }
  const size_t head_b = pn.mat("lm_head") + 2 * c.hidden;
  size_t head_plan = head_b + pn.extra("lm_head", !(getenv("HIVE_Q27_W4_HEAD") && strcmp(getenv("HIVE_Q27_W4_HEAD"), "0") == 0));
  if (const char* dv = getenv("HIVE_Q27_DFLASH"); want_mtp && dv && *dv && strcmp(dv, "0") != 0) { df_dir = dv; dflash = true; }
  const bool mtp_ck = dflash || (want_mtp && c.n_mtp > 0 && ck.has("mtp.fc.weight"));
  size_t mtp_b = 0;
  if (dflash) {   // drafter weights (as stored) + its work buffers (context capture rows) + a context ring per planned session
    size_t wb = 0;
    df_load(wb, false);
    mtp_b = wb + (size_t)std::max(chunk, kMaxRows) * df::kLayers * c.hidden * 2 + 3 * (size_t)std::max(chunk, kMaxRows) * 2 * c.hidden * 2 + 64 * kMiB;
    if (kv_plan && dsp) mtp_b += (size_t)plan_ctx * 2 * df::kLayers * df::kKvW * 2;   // DSpark keeps a context ring of every position
  } else if (mtp_ck) {
    mtp_b = pn.mat("mtp.fc") + 8 * c.hidden * 2;
    for (const char* n : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"})
      mtp_b += pn.mat(std::string("mtp.layers.0.") + n);
    mtp_b += kv_layer;   // the MTP layer's own KV cache
  }
  size_t free_b = 0, total_b = 0;
  size_t cap = env_mb("HIVE_Q27_VRAM_MB", env_mb("HIVE_VRAM_LIMIT_MB", 0));
  const bool plan_only = getenv("HIVE_Q27_PLAN_ONLY") && strcmp(getenv("HIVE_Q27_PLAN_ONLY"), "1") == 0;
  if (!(plan_only && cap > 0)) CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));   // (a capped plan-only run needs no GPU)
  const size_t free_start = free_b;   // before any upload (KV tier: the measured load)
  const bool memlog_on = getenv("HIVE_Q27_MEMLOG") && strcmp(getenv("HIVE_Q27_MEMLOG"), "1") == 0;   // tests: device bytes per load phase
  auto memlog = [&](const char* what) {
    if (!memlog_on) return;
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    fprintf(stderr, "[q27-mem] %-28s %8.0f MiB allocated since the plan\n", what, (double)(free_start > f ? free_start - f : 0) / 1048576.0);
  };

  size_t ctx = env_mb("HIVE_Q27_CTX_MB", 1000 * kMiB);
  const bool capped = cap > 0;
  if (!capped) cap = free_b + ctx;   // the context already exists: free memory is what is left after it
  // sequences whose recurrent states the reserve holds: HIVE_Q27_SESSIONS, default 2 (4 under a VRAM cap: the warm-up and a pooled session
  //   next to the running one — measured 2026-10-12, 32 GB cap: 1.3 GB left after the load and the daemon aborted on a 3 MiB allocation)
  //   — under the KV tier 16: every sequence of a full batch (hived --max-batch 16) has its recurrent state in VRAM (the KV of the batch may
  //   spill to host memory, the states cannot)
  const int sessions = getenv("HIVE_Q27_SESSIONS") ? std::max(1, atoi(getenv("HIVE_Q27_SESSIONS"))) : capped ? (kv_tier ? 16 : 4) : 2;
  const size_t per_row = 360000;                                        // work buffers per chunk row (sum of the allocations below, rounded up)
  const size_t deq_b = (size_t)c.inter * c.hidden * 2;                   // largest matrix as bf16 (prefill dequantization scratch)
  const size_t per_seq_lin = ((size_t)3 * c.lin_conv_ch() + (size_t)c.lin_v_heads * c.lin_k_dim * c.lin_v_dim) * 4;
  const size_t kv_allow = kv_tier ? 0 : (size_t)c.n_attn() * 2 * c.kv_width() * 2 * 4096;   // 4K positions of KV per sequence (grows on demand; the
  //   KV tier needs no allowance: KV that finds no VRAM goes to host memory)
  // KV tier: one layer's K and V up to the plan context in VRAM — a prefill's attention reads a host-tier layer from this copy (the kernel
  //   reads every key once per 64 query rows; over PCIe that would dominate)
  //   — allocated on the first prefill chunk that needs it (a long prompt whose KV spilled), not reserved: it then competes with the KV
  //   budget, and a failed allocation reads the host rows directly (slower, same values)
  const size_t stage_b = 0;
  size_t reserve = env_mb("HIVE_Q27_RESERVE_MB", 0);
  // (KV tier: the sequences' recurrent states are counted per GPU layer instead — lb below — so a small budget does not reserve the states of
  //   layers that run on the CPU; measured 2026-10-12, 16 GB: 2.4 GB of states for 48 layers reserved, 16 layers placed)
  if (!reserve) reserve = 300 * kMiB + per_row * chunk + deq_b + (size_t)sessions * ((kv_tier ? 0 : per_seq_lin * c.n_lin()) + kv_allow) + stage_b;
  if (kv_tier) for (int l = 0; l < c.n_layers; ++l) if (!c.is_attn[l]) lb[l] += (size_t)sessions * per_seq_lin;
  // under a cap: CUDA graphs, cuBLAS / cuBLASLt workspaces and the drafter's per-sequence rings also come out of the budget (the uncapped
  //   plan is unchanged)
  if (capped && !env_mb("HIVE_Q27_RESERVE_MB", 0)) reserve += 768 * kMiB;
  // KV tier: the load allocates more than the plan's estimates (measured 2026-10-12, benchmark settings + DFlash2, the plan's "used" vs the measured load:
  //   48 GB 28754 / 31932, 32 GB 22421 / 25372, 24 GB 18089 / 21890 MiB — 3.0-3.8 GB, of which the reserve's work-buffer estimate covers ~1.9);
  //   without the slack the 24 GB start had 2.1 GB left for the runtime's 4.3 and failed on a state allocation. HIVE_Q27_LOAD_SLACK_MB.
  if (kv_tier && !env_mb("HIVE_Q27_RESERVE_MB", 0)) reserve += env_mb("HIVE_Q27_LOAD_SLACK_MB", 2048 * kMiB);
  mtp_b += (vsnap ? (size_t)kMaxParts * per_seq_lin * c.n_lin() : 0) + (size_t)kMaxRows * c.n_lin() * (3 * c.lin_conv_ch() * 4 + 2 * c.lin_conv_ch() + 8 * c.lin_k_width());   // verify snapshots + replay inputs
  size_t room = cap > ctx + reserve ? cap - ctx - reserve : 0;
  const int sessions_df = sessions;
  const size_t ring = dflash ? (size_t)2 * df::kLayers * df::kRing * df::kKvW * 2 * sessions_df : 0;
  // HIVE_Q27_DRAFTER_FIRST (DFlash2 / DSpark under a VRAM cap): the drafter, its rings and its draft-vocabulary head are placed before the
  //   layers, so speculation stays on when some layers run on the CPU (their hidden states are captured from the host rows); 0 = the drafter
  //   only when every layer and the lm_head are on the GPU (the old rule)
  size_t dvocab_b = 0;
  if (const char* dv = getenv("HIVE_Q27_DRAFT_VOCAB"); dv && *dv && strcmp(dv, "0") != 0)
    if (FILE* f = fopen(dv, "rb")) { fseek(f, 0, SEEK_END); dvocab_b = (size_t)ftell(f) / 4 * ((size_t)c.hidden + 4 + 4) + 512; fclose(f); }
  const char* dfe = getenv("HIVE_Q27_DRAFTER_FIRST");
  const bool dfirst_on = dflash && capped && dvocab_b > 0 && !(dfe && strcmp(dfe, "0") == 0);
  // HIVE_Q27_GPU_LAYERS=N (tests): the first N layers on the GPU; unset / empty / "0" = placement by the cap
  const char* gl_env = getenv("HIVE_Q27_GPU_LAYERS");
  const bool gl_forced = gl_env && *gl_env && strcmp(gl_env, "0") != 0;
  size_t used = 0;
  int g = 0;
  bool pre_ok = false;
  // pass 0: layers, lm_head, then the drafter / MTP block when everything is on the GPU (the plan of every budget where all of it fits);
  //   pass 1 (only when pass 0 leaves layers on the CPU or the drafter off, and DRAFTER_FIRST is on): the drafter first, then the layers
  // KV tier: attempt 0 with the W4 decode copies; attempt 1 without them (the decode then reads the checkpoint's FP8 matrices — the lossless
  //   path; values as with HIVE_Q27_W4=0) when attempt 0 does not put every layer, the lm_head and the drafter on the GPU with at least
  //   HIVE_Q27_KV_TIER_MIN_MB (default 4096) left for KV — speculation is worth ~2-3x at c1, W4 ~10 % (measured 2026-10-12, 32 GB: with the W4
  //   copies 62 layers fit, 477 MiB were left for KV and the start failed on a 3 MiB state allocation)
  // HIVE_Q27_HEAD_FIRST=1 (under a cap): the lm_head before the layers (tests: a CPU lm_head reads ~1.3 GB per token on the CPU)
  bool head_first = capped && getenv("HIVE_Q27_HEAD_FIRST") && strcmp(getenv("HIVE_Q27_HEAD_FIRST"), "1") == 0;
  auto place_passes = [&]() {
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1 && (!dfirst_on || gl_forced || (g == c.n_layers && pl.head_gpu && pl.mtp_gpu))) break;
    const size_t pre = pass == 1 ? mtp_b + ring + dvocab_b : 0;
    if (pass == 1 && pre > room) break;
    used = pre; g = 0;
    if (head_first) { pl.head_gpu = used + head_plan <= room; if (pl.head_gpu) used += head_plan; }
    if (gl_forced) g = std::max(0, std::min(c.n_layers, atoi(gl_env)));
    else while (g < c.n_layers && used + lb[g] <= room) used += lb[g++];
    if (gl_forced) for (int l = 0; l < g; ++l) used += lb[l];
    pl.gpu_layers = g;
    if (!head_first) {
      pl.head_gpu = used + head_plan <= room || (g == c.n_layers && !capped);
      if (pl.head_gpu) used += head_plan;
    }
    pre_ok = pass == 1;
    if (pre_ok) {
      pl.mtp_gpu = true;   // drafter (+ rings + draft head) already counted in used
    } else {
      pl.mtp_gpu = mtp_ck && g == c.n_layers && pl.head_gpu && used + mtp_b <= room;
      if (pl.mtp_gpu) used += mtp_b;
      if (dflash && pl.mtp_gpu && used + ring <= room) used += ring;
    }
  }
  };
  const size_t kv_min = env_mb("HIVE_Q27_KV_TIER_MIN_MB", 4096 * kMiB);
  for (int attempt = 0; attempt < 2; ++attempt) {
  if (attempt == 1) {
    if (!kv_tier || !pn.w4 || gl_forced ||
        (g == c.n_layers && pl.head_gpu && (!mtp_ck || pl.mtp_gpu) && room >= used + kv_min)) break;
    const size_t hw = pn.extra("lm_head", !(getenv("HIVE_Q27_W4_HEAD") && strcmp(getenv("HIVE_Q27_W4_HEAD"), "0") == 0));
    pn.w4 = false;
    size_t w4all = hw - pn.extra("lm_head");
    head_plan -= w4all;
    for (int l = 0; l < c.n_layers; ++l) { lb[l] -= w4x[l]; w4all += w4x[l]; }
    w4_dropped = true;
    fprintf(stderr, "[q27] KV tier: the W4 decode copies (%.0f MiB) are dropped for the drafter / the VRAM KV budget (decode reads the FP8 matrices)\n",
            w4all / 1048576.0);
  }
  place_passes();
  }
  // KV tier, the W4 copies dropped by attempt 1 while every layer, the lm_head and the drafter are on the GPU (32 GB): the load converts as many
  //   copies as the measured VRAM allows after the drafter and HIVE_Q27_W4_KV_KEEP_MB of KV (the W4 copies below). Opt-in, HIVE_Q27_W4_PARTIAL=1:
  //   measured 2026-10-13, 32 GB: keep 1024 / 512 MiB → c1 169.8 / 170.9 tok/s (all copies dropped: 158.9) but the KV budget falls to
  //   0.97 / 0.43 GB and 32K / 64K prompts went from 7.2 / 21 s to > 600 s (staging and host tier starved) — rejected as the default
  //   (long context first). Budgets with layers on the CPU keep the rule above (their VRAM goes to layers first).
  w4_partial = w4_dropped && !gl_forced && g == c.n_layers && pl.head_gpu && (!mtp_ck || pl.mtp_gpu) &&
               getenv("HIVE_Q27_W4_PARTIAL") && strcmp(getenv("HIVE_Q27_W4_PARTIAL"), "1") == 0;
  // KV tier, when the plan above leaves layers on the CPU (so the ≥ 32 GB plans, where everything fits, are not touched): the reserve from
  //   measured peaks instead of estimates plus a slack. Measured 2026-10-12, benchmark settings + DFlash2, 16 GB, chunk 2048, batch 16: the load's work
  //   buffers took 2012 MiB (≈ 300 + 753 KiB per chunk row + the dequantization scratch), and after the load the runtime (recurrent states of up
  //   to 16 sequences, drafter rings, snapshots, CUDA graphs, cuBLAS workspaces) took 2157 MiB at its peak (c1 / c4 / c8 / 8K prompt; peak device
  //   use 15636 of 16384 MiB). The states and the rings are counted in the layers / the drafter; the rest is 1280 MiB.
  if (kv_tier && capped && !gl_forced && !env_mb("HIVE_Q27_RESERVE_MB", 0) && !(g == c.n_layers && pl.head_gpu && (!mtp_ck || pl.mtp_gpu))) {
    const size_t reserve2 = 300 * kMiB + (size_t)753 * 1024 * chunk + deq_b + 1280 * kMiB;
    if (reserve2 < reserve) {
      reserve = reserve2;
      // the CUDA context as hived measures it (555 MiB on the RTX PRO 6000 + margin) instead of the 1000 MiB estimate
      if (!env_mb("HIVE_Q27_CTX_MB", 0)) ctx = std::min(ctx, (size_t)640 * kMiB);
      // the lm_head before the layers: measured 2026-10-12, 16 GB: c1 15.7 tok/s (23 GPU layers + GPU lm_head) vs 15.0 (27 layers + CPU
      //   lm_head); HIVE_Q27_HEAD_FIRST=0 = layers first
      if (!(getenv("HIVE_Q27_HEAD_FIRST") && strcmp(getenv("HIVE_Q27_HEAD_FIRST"), "0") == 0)) head_first = true;
      room = cap > ctx + reserve ? cap - ctx - reserve : 0;
      place_passes();
    }
  }
  size_t w4_plan = 0;   // the W4 copies the plan counted (GPU layers + head)
  for (int l = 0; l < g; ++l) w4_plan += pn.w4 ? w4x[l] : 0;
  if (pn.w4 && pl.head_gpu) { pn.w4 = false; const size_t h0 = pn.extra("lm_head"); pn.w4 = true; w4_plan += pn.extra("lm_head") - h0; }
  // KV tier: what the runtime allocates after the load (per-sequence recurrent states of a full batch, the drafter's rings, snapshot slots,
  //   CUDA graphs and library workspaces) — the measured KV budget at the end of the load keeps it free
  int n_lin_gpu = 0;
  for (int l = 0; l < g; ++l) n_lin_gpu += !c.is_attn[l];
  const size_t tier_after = kv_tier ? (size_t)sessions * per_seq_lin * n_lin_gpu + (pl.mtp_gpu && dflash ? ring : 0) + 768 * kMiB + 512 * kMiB : 0;
  const size_t tier_work = kv_tier ? per_row * chunk + deq_b + stage_b + 300 * kMiB : 0;
  if (kv_tier) {
    kv_budget = room > used ? room - used : 0;
    if (const char* kb = getenv("HIVE_Q27_KV_BUDGET_MB"); kb && *kb && atof(kb) >= 0) kv_budget = std::min(kv_budget, (size_t)(atof(kb) * kMiB));   // tests: force the host tier
    fprintf(stderr, "[q27] KV tier: %.0f MiB of VRAM for KV (≈ %lld positions of one sequence), the rest in pinned host memory up to %lld positions\n",
            kv_budget / 1048576.0, (long long)(kv_budget / std::max<size_t>(1, (size_t)c.n_attn() * 2 * c.kv_width() * 2)), (long long)plan_ctx);
  }
  df_cpu_capture = pl.mtp_gpu && dflash && g < c.n_layers;
  if (g < c.n_layers && !gl_forced) {   // the Eke Hive rule: what does not fit the VRAM budget runs on the CPU (never a refusal)
    size_t all = 0;
    for (int l = 0; l < c.n_layers; ++l) all += lb[l];
    fprintf(stderr, "[q27] VRAM budget %.0f MiB: layers %d..%d (%d of %d) run on the CPU — the layers need %.0f MiB, %.0f MiB is left after the CUDA "
            "context (%.0f), the runtime reserve (%.0f)%s%s\n", cap / 1048576.0, g, c.n_layers - 1, c.n_layers - g, c.n_layers, all / 1048576.0,
            room / 1048576.0, ctx / 1048576.0, reserve / 1048576.0, pre_ok ? " and the drafter placed first (HIVE_Q27_DRAFTER_FIRST)" : "",
            kv_plan ? " · KV plan: the KV of one sequence at the plan context counted per GPU attention layer (HIVE_Q27_KV_PLAN=0 = off)" : "");
  }
  pl.cap = cap; pl.weights = used; pl.reserve = reserve;
  fprintf(stderr, "[q27] placement: cap %.0f MiB (%s) · reserve %.0f MiB · GPU layers 0..%d (%d of %d, %.0f MiB) · lm_head %s · MTP %s\n", cap / 1048576.0,
          env_mb("HIVE_Q27_VRAM_MB", 0) ? "HIVE_Q27_VRAM_MB" : env_mb("HIVE_VRAM_LIMIT_MB", 0) ? "HIVE_VRAM_LIMIT_MB" : "free VRAM", reserve / 1048576.0, g - 1, g,
          c.n_layers, used / 1048576.0, pl.head_gpu ? "GPU" : "CPU", pl.mtp_gpu ? (dflash ? "GPU (DFlash2 drafter)" : "GPU") : mtp_ck ? "off (does not fit)" : "off");
  if (plan_only) {   // tests: print the plan, load nothing
    fprintf(stderr, "[q27] plan only: cap %.0f MiB · reserve %.0f · room %.0f · GPU layers %d of %d (%.0f MiB) · lm_head %s · MTP/drafter %s (%.0f MiB) · KV plan %s "
            "(%lld positions, %.0f MiB per attention layer)\n", cap / 1048576.0, reserve / 1048576.0, room / 1048576.0, g, c.n_layers, used / 1048576.0,
            pl.head_gpu ? "GPU" : "CPU", pl.mtp_gpu ? "GPU" : "off", mtp_b / 1048576.0, kv_plan ? "on" : "off", (long long)plan_ctx, kv_layer / 1048576.0);
    exit(0);
  }
  // ---- CPU engine: the remaining layers, embeddings (always), lm_head when not on the GPU, the vision encoder (lazily)
  cpu::CpuOptions co;
  co.load_mtp = false; co.head = true; co.lm_head = !pl.head_gpu;
  co.layers.assign(c.n_layers, false);
  for (int l = g; l < c.n_layers; ++l) co.layers[l] = true;
  cpu = std::make_unique<cpu::Q27CpuEngine>(dir, co);
  // ---- GPU weights
  memlog("CPU engine");
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  blas = std::make_unique<Blas>(st);
  L.resize(c.n_layers);
  for (int l = 0; l < g; ++l) {
    load_layer(ck, P + "layers." + std::to_string(l) + ".", c.is_attn[l], L[l]);
    if ((l + 1) % 16 == 0) fprintf(stderr, "[q27] GPU layers 0..%d loaded (%.1f s)\n", l, (hive::mono_ms() - t0) / 1000);
  }
  for (int l = 0; l < g; ++l) if (!c.is_attn[l]) gpu_lin.push_back(lin_index[l]);
  memlog("GPU layers");
  {
    head_mem.alloc((pl.head_gpu ? head_b : 2 * (size_t)c.hidden + 256) + 4096);   // (lm_head on the CPU: only the final norm — the full
    //   head-size buffer was allocated anyway: 1214 MiB of VRAM unused at 16 GB, measured 2026-10-12)
    uint8_t* cur = head_mem.as<uint8_t>();
    const TensorInfo& n = ck.get(P + "norm.weight");
    CUDA_CHECK(cudaMemcpy(cur, n.data, n.nbytes, cudaMemcpyHostToDevice));
    final_norm = (const bf16*)cur; cur += 2 * c.hidden + 256;
    if (pl.head_gpu) lm_head = upload_mat(ck, {"lm_head"}, cur);
  }
  memlog("lm_head");
  if (pl.mtp_gpu && dflash) {
    size_t wb = 0;
    df_load(wb, true);
    mtp_k = getenv("HIVE_Q27_MTP_K") ? std::max(0, std::min(df::kBlock - 1, atoi(getenv("HIVE_Q27_MTP_K")))) : df::kBlock - 1;   // 7 drafts (both drafters)
  }
  if (pl.mtp_gpu && !dflash) {
    mtp_mem.alloc(pn.mat("mtp.fc") + 4 * c.hidden * 2 + 4096);
    uint8_t* cur = mtp_mem.as<uint8_t>();
    mtp_fc = upload_mat(ck, {"mtp.fc"}, cur);
    auto vec = [&](const std::string& nm) {
      cur = (uint8_t*)(((uintptr_t)cur + 255) & ~(uintptr_t)255);
      const TensorInfo& t = ck.get(nm);
      CUDA_CHECK(cudaMemcpy(cur, t.data, t.nbytes, cudaMemcpyHostToDevice));
      const bf16* r = (const bf16*)cur; cur += t.nbytes; return r;
    };
    mtp_en = vec("mtp.pre_fc_norm_embedding.weight"); mtp_hn = vec("mtp.pre_fc_norm_hidden.weight"); mtp_norm = vec("mtp.norm.weight");
    load_layer(ck, "mtp.layers.0.", true, mtpL);
    mtp_k = getenv("HIVE_Q27_MTP_K") ? std::max(0, std::min(7, atoi(getenv("HIVE_Q27_MTP_K")))) : 3;
    mtp_post_main = getenv("HIVE_Q27_MTP_INPUT") && strcmp(getenv("HIVE_Q27_MTP_INPUT"), "post") == 0;
    mtp_post_chain = getenv("HIVE_Q27_MTP_CHAIN") && strcmp(getenv("HIVE_Q27_MTP_CHAIN"), "post") == 0;
  }
  if (pl.mtp_gpu) {
    const bool host_head = !pl.head_gpu && ck.has("lm_head.weight") && ck.get("lm_head.weight").dtype == "F8_E4M3";   // lm_head on the CPU
    if (const char* dv = getenv("HIVE_Q27_DRAFT_VOCAB"); dv && *dv && strcmp(dv, "0") != 0 && ((pl.head_gpu && lm_head.fmt == WFmt::FP8ROW) || host_head)) {
      FILE* f = fopen(dv, "rb");
      if (f) {
        int32_t id;
        while (fread(&id, 4, 1, f) == 1) if (id >= 0 && id < c.vocab) draft_ids.push_back(id);
        fclose(f);
      }
      if (draft_ids.empty()) fprintf(stderr, "[q27] HIVE_Q27_DRAFT_VOCAB=%s: no usable token ids — drafts score the whole vocabulary\n", dv);
      else {
        const int n = (int)draft_ids.size(), K = host_head ? (int)ck.get("lm_head.weight").shape[1] : lm_head.K;
        draft_mem.alloc((size_t)n * K + (size_t)n * 4 + 512);
        uint8_t* dw = draft_mem.as<uint8_t>();
        float* ds = reinterpret_cast<float*>(dw + (((size_t)n * K + 255) & ~(size_t)255));
        if (host_head) {   // lm_head on the CPU: the rows and their per-row scales straight from the checkpoint
          const TensorInfo& hw = ck.get("lm_head.weight");
          const TensorInfo& hs = ck.get("lm_head.weight_scale");
          std::vector<uint8_t> rows((size_t)n * K);
          std::vector<float> sc(n);
          for (int i = 0; i < n; ++i) {
            memcpy(rows.data() + (size_t)i * K, hw.data + (size_t)draft_ids[i] * K, K);
            sc[i] = b2f(reinterpret_cast<const uint16_t*>(hs.data)[draft_ids[i]]);
          }
          CUDA_CHECK(cudaMemcpy(dw, rows.data(), rows.size(), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(ds, sc.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
          draft_head.fmt = WFmt::FP8ROW; draft_head.K = K;
        } else {
          for (int i = 0; i < n; ++i) {   // device-to-device row copies (rows and scales of the FP8 lm_head)
            CUDA_CHECK(cudaMemcpyAsync(dw + (size_t)i * K, lm_head.w + (size_t)draft_ids[i] * K, K, cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(ds + i, reinterpret_cast<const float*>(lm_head.s) + draft_ids[i], 4, cudaMemcpyDeviceToDevice, st));
          }
          CUDA_CHECK(cudaStreamSynchronize(st));
          draft_head = lm_head;
        }
        draft_head.w = dw; draft_head.s = ds; draft_head.N = n;
        draft_ids_d.alloc((size_t)n * 4);
        CUDA_CHECK(cudaMemcpy(draft_ids_d.p, draft_ids.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        fprintf(stderr, "[q27] draft vocabulary %s: %d tokens (%.0f MiB draft head)\n", dv, n, (double)n * K / 1048576.0);
      }
    }
  }
  if (pl.mtp_gpu && dflash && !pl.head_gpu && draft_ids.empty()) {   // the drafter scores its candidates on a GPU head
    pl.mtp_gpu = false; df_cpu_capture = false;
    fprintf(stderr, "[q27] drafter off: the lm_head runs on the CPU and there is no draft-vocabulary head (HIVE_Q27_DRAFT_VOCAB)\n");
  }
  memlog("drafter + draft head");
  ck.release_all();
  // the embedding table in VRAM when the whole model and the MTP block are there and 2× its size stays free (MTP draft chain on the device)
  if (pl.mtp_gpu && !(getenv("HIVE_Q27_EMB_GPU") && strcmp(getenv("HIVE_Q27_EMB_GPU"), "0") == 0)) {
    const size_t eb = (size_t)c.vocab * c.hidden * 2;
    size_t fr = 0, tt = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tt));
    // (used already holds the drafter / MTP block; under the KV tier the leftover belongs to the KV budget — the table is mapped instead)
    if (fr > 2 * eb + reserve && (!capped || (!kv_tier && used + eb + reserve + ctx <= cap))) {
      emb_d.alloc(eb);
      CUDA_CHECK(cudaMemcpy(emb_d.p, cpu->embed_table(), eb, cudaMemcpyHostToDevice));
      emb_ptr = emb_d.as<bf16>();
      fprintf(stderr, "[q27] embedding table in VRAM (%.0f MiB): device-side MTP draft chain\n", eb / 1048576.0);
    } else if (capped && !(getenv("HIVE_Q27_EMB_MAPPED") && strcmp(getenv("HIVE_Q27_EMB_MAPPED"), "0") == 0)) {
      // no room next to the KV plan (48 GB cap, benchmark settings: 44.5 of 44.6 GB planned): the host table (lives as long as the engine, never freed) mapped
      //   for the device gathers instead of host gathers with a stream sync — the drafter keeps its device path and CUDA graphs
      void* hp = const_cast<uint16_t*>(cpu->embed_table());
      void* dp = nullptr;
      cudaError_t e = cudaHostRegister(hp, eb, cudaHostRegisterMapped);
      if (e == cudaSuccess) e = cudaHostGetDevicePointer(&dp, hp, 0);
      if (e == cudaSuccess) {
        emb_ptr = static_cast<const bf16*>(dp);
        fprintf(stderr, "[q27] embedding table: host memory mapped for the device gathers (%.0f MiB do not fit next to the KV plan)\n", eb / 1048576.0);
      } else {
        cudaGetLastError();
        fprintf(stderr, "[q27] embedding table: host gathers (no VRAM next to the KV plan; mapping refused: %s)\n", cudaGetErrorString(e));
      }
    }
  }
  // HIVE_Q27_W4=1 (lossy, off by default): NVFP4 copies of the FP8 matrices for decode / verify (prefill keeps the FP8 originals)
  // (the W4A4 prefill of the copies — PREFILL_QUANT without HIVE_Q27_W4_PREFILL=0 — needs their scales tiled for cuBLASLt; the shipped q27.env
  //   sets W4_PREFILL=0, where the tiled copy was allocated and never read: ~0.7 GB of VRAM at every size, 2026-10-12)
  const bool w4_swz = pq && !(getenv("HIVE_Q27_W4_PREFILL") && strcmp(getenv("HIVE_Q27_W4_PREFILL"), "0") == 0);
  const bool w4_on = getenv("HIVE_Q27_W4") && strcmp(getenv("HIVE_Q27_W4"), "1") == 0;
  // the NVFP4 decode copies of the FP8 matrices of the GPU layers and the lm_head, up to `limit` bytes (SIZE_MAX = all)
  auto w4_convert = [&](size_t limit) {
    DevBuf am(4);
    size_t bytes = 0;
    int n = 0, skipped = 0;
    auto conv = [&](GMat& W) {
      if (W.fmt != WFmt::FP8ROW || W.w4) return;
      const size_t cb = ((size_t)W.N * W.K / 2 + (size_t)W.N * W.K / 16 + 256) + (w4_swz ? lowp::swz_bytes(W.N, W.K) : 0);
      if (limit != SIZE_MAX && bytes + cb > limit) { ++skipped; return; }   // partial copies: this matrix decodes from FP8
      DevBuf b((size_t)W.N * W.K / 2 + (size_t)W.N * W.K / 16 + 256);
      uint8_t* w4 = b.as<uint8_t>();
      uint8_t* s4 = w4 + (((size_t)W.N * W.K / 2 + 255) & ~(size_t)255);
      W.g4 = k::requant_nvfp4(W, w4, s4, am.as<float>(), st);
      W.w4 = w4; W.s4 = s4;
      if (w4_swz) {   // tiled scales for the W4A4 prefill of the copy
        DevBuf sz(lowp::swz_bytes(W.N, W.K));
        lowp::swizzle_scales(s4, W.N, W.K, sz.as<uint8_t>(), st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        W.s4_swz = sz.as<uint8_t>();
        bytes += sz.n;
        (limit != SIZE_MAX ? w4_part_store : w4_store).push_back(std::move(sz));
      }
      bytes += b.n; ++n;
      (limit != SIZE_MAX ? w4_part_store : w4_store).push_back(std::move(b));
    };
    for (int l = 0; l < g; ++l) for (GMat* m : {&L[l].qkvz, &L[l].lin_out, &L[l].qkv, &L[l].o, &L[l].gate, &L[l].up, &L[l].down}) conv(*m);
    if (!(getenv("HIVE_Q27_W4_HEAD") && strcmp(getenv("HIVE_Q27_W4_HEAD"), "0") == 0) && pl.head_gpu) conv(lm_head);
    CUDA_CHECK(cudaStreamSynchronize(st));
    fprintf(stderr, "[q27] HIVE_Q27_W4: %d FP8 matrices get an NVFP4 decode copy (%.0f MiB)%s\n", n, bytes / 1048576.0,
            skipped ? (" · " + std::to_string(skipped) + " decode from FP8 (partial copies under the KV tier)").c_str() : "");
  };
  // KV tier, the copies do not all fit (32 GB: the plan dropped them for the drafter, or the measured load leaves less than KV_TIER_MIN):
  //   partial copies, converted at the end of the load from the VRAM really left after the work buffers, minus the runtime and
  //   HIVE_Q27_W4_KV_KEEP_MB of VRAM KV (measured 2026-10-12, 32 GB: a limit taken before the work buffers left 0 MiB for KV — the load
  //   allocated 2.3 GB after that point vs 1.2 GB estimated — and a 32K prompt then took > 600 s)
  bool w4_defer = false;
  if (kv_tier && (!w4_dropped || w4_partial) && w4_on) {   // KV tier: measured, not planned
    size_t fr = 0, tt = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tt));
    const size_t eng = free_start > fr ? free_start - fr : 0;   // what the load allocated so far
    if (w4_dropped || eng + w4_plan + tier_work + tier_after + kv_min + ctx > cap) { w4_defer = true; w4_dropped = true; }
  }
  if (w4_on && !w4_dropped) w4_convert(SIZE_MAX);
  // HIVE_Q27_DFLASH_FP4: the draft head (FP8 rows of the draft vocabulary) gets an NVFP4 decode copy as well (drafts only)
  if (getenv("HIVE_Q27_DFLASH_FP4") && strcmp(getenv("HIVE_Q27_DFLASH_FP4"), "1") == 0 && draft_head.fmt == WFmt::FP8ROW && !draft_head.w4 &&
      draft_head.K % 16 == 0) {
    DevBuf am(4), b((size_t)draft_head.N * draft_head.K / 2 + (size_t)draft_head.N * draft_head.K / 16 + 256);
    uint8_t* w4 = b.as<uint8_t>();
    uint8_t* s4 = w4 + (((size_t)draft_head.N * draft_head.K / 2 + 255) & ~(size_t)255);
    draft_head.g4 = k::requant_nvfp4(draft_head, w4, s4, am.as<float>(), st);
    draft_head.w4 = w4; draft_head.s4 = s4;
    w4_store.push_back(std::move(b));
  }
  // decode / verify activation policy per use (GMat.act; NInfer-style, lossy — quality gate): HIVE_Q27_A4_MLP=1 → NVFP4 activations (W4A4) for
  //   the MLP gate / up / down of every layer (the FP8 ones through their HIVE_Q27_W4 NVFP4 copy), HIVE_Q27_A4_PROJ=1 → the same for the
  //   attention / GDN projections; the lm_head and the drafter stay bf16
  {
    // HIVE_Q27_A4_MLP: 1 = gate, up and down · 2 = gate and up · 3 = down only (per-use bisection of the quality gate)
    const int a4mv = getenv("HIVE_Q27_A4_MLP") ? atoi(getenv("HIVE_Q27_A4_MLP")) : 0;
    const bool a4m = a4mv >= 1 && a4mv <= 3;
    const bool a4p = getenv("HIVE_Q27_A4_PROJ") && atoi(getenv("HIVE_Q27_A4_PROJ")) == 1;
    // act == 4 needs the codes / scales repacked in mma fragment order (k::a4_repack; a copy beside the matrix the decode GEMVs read)
    auto mark = [&](GMat& W) {
      if (!(W.fmt == WFmt::NVFP4 || W.w4)) return;
      const GMat v = k::decode_view(W);
      if (v.fmt != WFmt::NVFP4 || v.K % 64) return;
      DevBuf b(k::a4_repack(v, nullptr, st));
      k::a4_repack(v, b.as<uint8_t>(), st);
      const size_t cb = (size_t)((v.N + 15) / 16) * 16 * v.K / 2;
      W.a4w = b.as<uint8_t>(); W.a4s = b.as<uint8_t>() + cb; W.act = 4;
      w4_store.push_back(std::move(b));
    };
    int nm = 0, np = 0;
    for (int l = 0; l < g; ++l) {
      if (a4m) for (GMat* m : {&L[l].gate, &L[l].up, &L[l].down}) {
        if ((a4mv == 2 && m == &L[l].down) || (a4mv == 3 && m != &L[l].down)) continue;
        mark(*m); nm += m->act == 4;
      }
      if (a4p) for (GMat* m : {&L[l].qkvz, &L[l].lin_out, &L[l].qkv, &L[l].o}) { mark(*m); np += m->act == 4; }
    }
    if (a4m || a4p) { CUDA_CHECK(cudaStreamSynchronize(st)); fprintf(stderr, "[q27] W4A4 decode / verify: %d MLP + %d projection matrices take NVFP4 activations\n", nm, np); }
  }
  memlog("embedding + W4 copies");
  // ---- rope tables
  const int half = c.rotary_dim / 2;
  std::vector<double> inv(half);
  std::vector<int32_t> ax(half, 0);
  for (int i = 0; i < half; ++i) {
    inv[i] = std::pow((double)c.rope_theta, -2.0 * i / c.rotary_dim);
    if (c.mrope_interleaved) ax[i] = (i % 3 == 1 && i < 3 * c.mrope_section[1]) ? 1 : (i % 3 == 2 && i < 3 * c.mrope_section[2]) ? 2 : 0;
    else ax[i] = i < c.mrope_section[0] ? 0 : i < c.mrope_section[0] + c.mrope_section[1] ? 1 : 2;
  }
  rope_inv.alloc(half * 8); rope_axis.alloc(half * 4);
  CUDA_CHECK(cudaMemcpy(rope_inv.p, inv.data(), half * 8, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rope_axis.p, ax.data(), half * 4, cudaMemcpyHostToDevice));
  {   // cos / sin table for positions < HIVE_Q27_ROPE_TABLE (default 262144): attn_prep reads it instead of fp64 sincos per row
    const int n = getenv("HIVE_Q27_ROPE_TABLE") ? std::max(0, atoi(getenv("HIVE_Q27_ROPE_TABLE"))) : 262144;
    if (n > 0) {
      std::vector<float> t((size_t)n * half * 2);
      const int nt = 16;
      std::vector<std::thread> th;
      for (int w = 0; w < nt; ++w)
        th.emplace_back([&, w] {
          for (int p = w; p < n; p += nt)
            for (int i = 0; i < half; ++i) { const double a = (double)p * inv[i]; t[((size_t)p * half + i) * 2] = (float)std::cos(a); t[((size_t)p * half + i) * 2 + 1] = (float)std::sin(a); }
        });
      for (auto& x : th) x.join();
      rope_tab.alloc(t.size() * 4);
      CUDA_CHECK(cudaMemcpy(rope_tab.p, t.data(), t.size() * 4, cudaMemcpyHostToDevice));
      k::set_rope_table(reinterpret_cast<const float2*>(rope_tab.p), n);
    }
  }
  memlog("rope table");
  // ---- work buffers
  const int R = std::max(chunk, kMaxRows), H = c.hidden, C = c.lin_conv_ch(), VW = c.lin_v_width(), KW = c.lin_k_width(), VH = c.lin_v_heads;
  const int PWL = C + VW + 2 * VH, PWA = 2 * c.q_width() + 2 * c.kv_width();
  h.alloc((size_t)R * H * 2); x.alloc((size_t)R * H * 2); y.alloc((size_t)R * H * 2);
  proj.alloc((size_t)R * std::max(PWL, PWA) * 2); gy.alloc((size_t)R * C * 2); qk.alloc((size_t)R * 2 * KW * 4); gb.alloc((size_t)R * 2 * VH * 4);
  go.alloc((size_t)R * VW * 4); gout.alloc((size_t)R * VW * 2); q.alloc((size_t)R * c.q_width() * 2); attf.alloc((size_t)R * c.q_width() * 4);
  ao.alloc((size_t)R * c.q_width() * 2); gu.alloc((size_t)R * 2 * c.inter * 2); acc.alloc((size_t)R * std::max(c.inter, PWL) * 4);
  deq_bytes = deq_b; deq.alloc(deq_b);
  if (kv_tier) {   // the allocation hook of session image loads (the host-tier prefill staging copy is allocated on demand)
    g_kv_alloc = [this](KvBuf& b, size_t n) { alloc_kv(b, n, 0); };
  }
  if (dq) {
    dq_x.alloc((size_t)kMaxRows * std::max(c.inter, 2 * c.hidden)); dq_s.alloc((size_t)kMaxRows * 4);
    dq_xs4.alloc((size_t)kMaxRows * std::max(c.inter, 2 * c.hidden) / 16);
    fprintf(stderr, "[q27] decode / verify GEMMs of the FP8 matrices: W8A8 (HIVE_Q27_DECODE_QUANT=1)\n");
  }
  if (pq) {
    const int KM = std::max(c.inter, c.lin_v_width());
    pq_x.alloc((size_t)R * KM); pq_s.alloc((size_t)R * 4); pq_swz.alloc(lowp::swz_bytes(R, KM)); pq_dyn.alloc(64);
    lt = std::make_unique<lowp::Lt>(st);
    fprintf(stderr, "[q27] prefill GEMMs: W8A8 (FP8 layers) / W4A4 (NVFP4 MLP) through cuBLASLt (HIVE_Q27_PREFILL_QUANT=1)\n");
  }
  attn_part.alloc(k::attention_scratch_bytes(R, c.n_kv_heads));   // split-key decode attention partials (rows ≤ chunk: also inner forwards)
  k::set_attention_scratch(attn_part.as<float>(), R);
  logits_d.alloc((size_t)kMaxRows * c.vocab * 4); coords_d.alloc((size_t)R * 12); rows_d.alloc((size_t)R * sizeof(k::AttnRow)); segs_d.alloc(64 * sizeof(k::GdnSeg));
  cat.alloc((size_t)R * 2 * H * 2); fh.alloc((size_t)R * H * 2);
  tab_rows = R;
  ids_d.alloc((size_t)(R + 64) * 4);
  tab_bytes = mtp_tab_off() + (size_t)R * sizeof(k::AttnRow);
  tab_d.alloc(tab_bytes);
  CUDA_CHECK(cudaMallocHost(&tab_pin, tab_bytes));
  CUDA_CHECK(cudaMallocHost(&logits_pin, (size_t)kMaxRows * c.vocab * 4));
  CUDA_CHECK(cudaMallocHost(&small_pin, 64));
  small_d.alloc(16384); me_emb.alloc((size_t)R * H * 2); me_hm.alloc((size_t)R * H * 2); me_cd.alloc((size_t)R * 12); eh_buf.alloc((size_t)(R + 16) * H * 2);
  use_graphs = !(getenv("HIVE_Q27_GRAPHS") && strcmp(getenv("HIVE_Q27_GRAPHS"), "0") == 0);
  k::gemv_prepare();   // split-N GEMV scratch (HIVE_Q27_SW=20), before the warm-up captures graphs
  k::gdn_reserve(R, c.lin_k_heads, VH);   // the chunked GDN prefill workspaces (counted in the measured load: KV tier budget, partial W4)
  if (pl.mtp_gpu) {
    snap_conv.resize((size_t)(kMaxRows - 1) * gpu_lin.size()); snap_state.resize(vsnap ? (size_t)kMaxParts * gpu_lin.size() : 0);
    for (size_t i = 0; i < snap_conv.size(); ++i) snap_conv[i].alloc((size_t)3 * C * 4);
    for (size_t i = 0; i < snap_state.size(); ++i) snap_state[i].alloc((size_t)VH * 128 * 128 * 4);
    // the GDN inputs of a verify per linear layer (rollback replay): v rows (bf16 [rows, C]), q / k (fp32 [rows, 2·KW]), gate / beta
    vq_y.alloc((size_t)c.n_lin() * kMaxRows * C * 2); vq_qk.alloc((size_t)c.n_lin() * kMaxRows * 2 * KW * 4); vq_gb.alloc((size_t)c.n_lin() * kMaxRows * 2 * VH * 4);
    rp_items.alloc((size_t)kMaxParts * c.n_lin() * (sizeof(k::GdnReplay) + sizeof(k::CopyItem)));
    snap_ptrs.alloc((size_t)c.n_lin() * 2 * kMaxRows * sizeof(float*));
    std::vector<float*> tab((size_t)c.n_lin() * 2 * kMaxRows, nullptr);
    for (size_t j = 0; j < gpu_lin.size(); ++j) {
      for (int sl = 0; sl < kMaxRows - 1; ++sl) tab[(size_t)gpu_lin[j] * 2 * kMaxRows + sl] = snap_conv[(size_t)sl * gpu_lin.size() + j].as<float>();
      for (int sl = 0; vsnap && sl < kMaxParts; ++sl) tab[(size_t)gpu_lin[j] * 2 * kMaxRows + kMaxRows + sl] = snap_state[(size_t)sl * gpu_lin.size() + j].as<float>();
    }
    CUDA_CHECK(cudaMemcpy(snap_ptrs.p, tab.data(), tab.size() * sizeof(float*), cudaMemcpyHostToDevice));
  }
  if (w4_defer) {   // partial W4 copies (see w4_defer)
    size_t fr = 0, tt = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tt));
    const size_t eng = free_start > fr ? free_start - fr : 0;
    const size_t keep = env_mb("HIVE_Q27_W4_KV_KEEP_MB", 1024 * kMiB);
    const size_t need = eng + tier_after + keep + ctx;
    const size_t limit = cap > need ? cap - need : 0;
    if (limit >= 64 * kMiB) {
      fprintf(stderr, "[q27] KV tier: W4 decode copies up to %.0f MiB (loaded %.0f MiB, %.0f MiB kept for KV; the other matrices decode from FP8)\n",
              limit / 1048576.0, eng / 1048576.0, keep / 1048576.0);
      w4_convert(limit);
      w4_dropped = false;
    } else {
      fprintf(stderr, "[q27] KV tier: W4 decode copies skipped — loaded %.0f MiB, less than %.0f MiB would stay for KV (decode reads the FP8 matrices)\n",
              eng / 1048576.0, keep / 1048576.0);
    }
  }
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  memlog("work buffers (end of load)");
  if (kv_tier) {   // the VRAM KV budget from the measured load (the plan is an estimate; a short budget only moves KV to host memory earlier)
    size_t fr = 0, tt = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tt));
    const size_t eng = free_start > fr ? free_start - fr : 0;
    const size_t avail = cap > ctx + eng + tier_after ? cap - ctx - eng - tier_after : 0;
    if (avail < kv_budget) kv_budget = avail;
    fprintf(stderr, "[q27] KV tier (measured): the load took %.0f MiB · %.0f MiB of VRAM for KV (≈ %lld positions of one sequence) after %.0f MiB kept for "
            "the runtime (states of %d sequences, drafter rings, snapshots, graphs)\n", eng / 1048576.0, kv_budget / 1048576.0,
            (long long)(kv_budget / std::max<size_t>(1, (size_t)c.n_attn() * 2 * c.kv_width() * 2)), tier_after / 1048576.0, sessions);
  }
  if (kv_tier && kv_host_grow) {   // the host tier's pinned ranges, ahead of use (KvHostSlot): a background thread registers them after the load
    // (only where a long prompt spills: a VRAM KV budget under 4 GiB = 64K positions — 32 GB; the 48 / 72 GB budgets hold 200K+ positions)
    const size_t pre_total = kv_budget < ((size_t)4 << 30) ? env_mb("HIVE_Q27_KV_HOST_PREPIN_MB", 4096 * kMiB) : 0, slot_pre = env_mb("HIVE_Q27_KV_HOST_SLOT_MB", 128 * kMiB);   // 128 MiB per range = 64K positions (tests: smaller)
    g_kv_host_pool_cap = std::max(pre_total, env_mb("HIVE_Q27_KV_HOST_POOL_MB", 8192 * kMiB));
    const size_t rsv = ((size_t)(s_plan_ctx > 0 ? s_plan_ctx : (int64_t)c.max_position) * c.kv_width() * 2 + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    const int n = (int)(pre_total / slot_pre);
    int dev = 0;
    cudaGetDevice(&dev);
    if (n > 0) std::thread([n, rsv, slot_pre, dev] {
      cudaSetDevice(dev);
      const double t = hive::mono_ms();
      int ok = 0;
      for (int i = 0; i < n; ++i) {
        KvHostSlot* s = KvHostSlot::make(rsv);
        if (!s) break;
        if (!s->extend_to(std::min(slot_pre, rsv))) { s->destroy(); break; }
        kv_host_give(s); ++ok;
      }
      fprintf(stderr, "[q27] KV tier: %d host ranges pinned ahead of use (%zu MiB, %.1f s, background)\n", ok, (size_t)ok * (slot_pre >> 20), (hive::mono_ms() - t) / 1000);
    }).detach();
  }
  fprintf(stderr, "[q27] loaded in %.1f s · VRAM free %.0f MiB of %.0f · CPU layers %.2f GiB · chunk %d · MTP k %d\n", (hive::mono_ms() - t0) / 1000,
          free_b / 1048576.0, total_b / 1048576.0, cpu->weight_bytes() / 1073741824.0, chunk, mtp_k);
}

// ---- pieces ---------------------------------------------------------------------------------------------------------------------------------
void Q27Engine::Impl::lin(const GMat& W, const bf16* xin, int ldx, int M, bf16* yb, int ldy, float* yf) {
  if (M <= 0) return;
  if (M <= kMaxRows) {
    if (W.w4) { k::gemv(k::decode_view(W), xin, ldx, M, yb, yf, ldy, st); return; }   // HIVE_Q27_W4 decode copy
    if (dq && M <= 16 && W.fmt == WFmt::FP8ROW) {   // (the W8A8 / W4A4 decode kernels take ≤ 16 rows)
      lowp::quant_fp8_rows(xin, ldx, M, W.K, dq_x.as<uint8_t>(), dq_s.as<float>(), st);
      lowp::fp8_gemm_rows(dq_x.as<uint8_t>(), W.K, dq_s.as<float>(), M, W.w, reinterpret_cast<const float*>(W.s), W.N, W.K, yb, yf, ldy, st);
      return;
    }
    if (dq && M <= 16 && W.fmt == WFmt::NVFP4 && W.in_gs > 0.f) {
      lowp::quant_nvfp4_rm(xin, ldx, M, W.K, W.in_gs, dq_x.as<uint8_t>(), dq_xs4.as<uint8_t>(), st);
      lowp::nvfp4_gemm_rows(dq_x.as<uint8_t>(), dq_xs4.as<uint8_t>(), M, W.w, reinterpret_cast<const uint8_t*>(W.s), W.N, W.K, W.g / W.in_gs, yb, yf,
                            ldy, st);
      return;
    }
    k::gemv(W, xin, ldx, M, yb, yf, ldy, st);   // decode / verify rows (tensor cores above 8 rows)
    return;
  }   // decode / verify rows (tensor cores above 8 rows)
  // HIVE_Q27_W4 + HIVE_Q27_PREFILL_QUANT: FP8 matrices with an NVFP4 copy run W4A4 prefill GEMMs on it (dynamic activation global scale,
  //   bf16 straight from the GEMM); HIVE_Q27_W4_PREFILL=0 keeps their W8A8 path
  static const bool w4pf = !(getenv("HIVE_Q27_W4_PREFILL") && strcmp(getenv("HIVE_Q27_W4_PREFILL"), "0") == 0);
  if (pq && yb && w4pf && W.w4 && W.s4_swz) {
    bool ok = true;
    for (int m0 = 0; m0 < M && ok; m0 += chunk) {
      const int mm = std::min(chunk, M - m0);
      lowp::quant_nvfp4_rows_dyn(xin + (size_t)m0 * ldx, ldx, mm, W.K, W.g4, pq_dyn.as<float>(), pq_x.as<uint8_t>(), pq_swz.as<uint8_t>(), st);
      ok = lt->nvfp4_bf16_dev(pq_x.as<uint8_t>(), pq_swz.as<uint8_t>(), W.w4, W.s4_swz, mm, W.N, W.K, pq_dyn.as<float>() + 2, yb + (size_t)m0 * ldy, ldy);
    }
    if (ok) return;
  }
  if (pq && yb && (W.fmt == WFmt::FP8ROW || (W.fmt == WFmt::NVFP4 && W.s_swz))) {
    bool ok = true;
    for (int m0 = 0; m0 < M && ok; m0 += chunk) {
      const int mm = std::min(chunk, M - m0);
      const bf16* xr = xin + (size_t)m0 * ldx;
      static const bool direct = !(getenv("HIVE_Q27_LT_DIRECT") && strcmp(getenv("HIVE_Q27_LT_DIRECT"), "0") == 0);   // bf16 straight from the GEMM
      if (W.fmt == WFmt::FP8ROW) {
        lowp::quant_fp8_rows(xr, ldx, mm, W.K, pq_x.as<uint8_t>(), pq_s.as<float>(), st);
        if (direct && lt->fp8_bf16(pq_x.as<uint8_t>(), pq_s.as<float>(), W.w, reinterpret_cast<const float*>(W.s), mm, W.N, W.K, yb + (size_t)m0 * ldy, ldy)) continue;
        ok = lt->fp8(pq_x.as<uint8_t>(), W.w, mm, W.N, W.K, acc.as<float>());
        if (ok) lowp::scale_out(acc.as<float>(), mm, W.N, 1.f, pq_s.as<float>(), reinterpret_cast<const float*>(W.s), yb + (size_t)m0 * ldy, ldy, st);
      } else {
        lowp::quant_nvfp4_rows(xr, ldx, mm, W.K, W.in_gs, pq_x.as<uint8_t>(), pq_swz.as<uint8_t>(), st);
        if (direct && lt->nvfp4_bf16(pq_x.as<uint8_t>(), pq_swz.as<uint8_t>(), W.w, W.s_swz, mm, W.N, W.K, W.g / W.in_gs, yb + (size_t)m0 * ldy, ldy)) continue;
        ok = lt->nvfp4(pq_x.as<uint8_t>(), pq_swz.as<uint8_t>(), W.w, W.s_swz, mm, W.N, W.K, W.g / W.in_gs, acc.as<float>());
        if (ok) lowp::scale_out(acc.as<float>(), mm, W.N, 1.f, nullptr, nullptr, yb + (size_t)m0 * ldy, ldy, st);
      }
    }
    if (ok) return;
    if (!pq_warned) { fprintf(stderr, "[q27] cuBLASLt has no low-precision algorithm for this GEMM — the bf16 path is used\n"); pq_warned = true; }
  }
  const bf16* B = (const bf16*)W.w;
  if (W.fmt != WFmt::BF16) {
    HIVE_CHECK((size_t)W.N * W.K * 2 <= deq_bytes, "q27: dequantization scratch too small");
    k::dequant_pre(W, deq.as<bf16>(), st);
    B = deq.as<bf16>();
  }
  // NVFP4 / BF16 weights have one scale for the whole matrix: cuBLAS writes bf16 = round(acc · g) itself (the same single rounding as the
  //   epilogue; skips the fp32 round trip). FP8 keeps the per-row scale epilogue.
  static const bool direct_bf16 = !(getenv("HIVE_Q27_BF16_DIRECT") && strcmp(getenv("HIVE_Q27_BF16_DIRECT"), "0") == 0);
  if (direct_bf16 && yb && !yf && W.fmt != WFmt::FP8ROW) {
    if (!cbh) { CUBLAS_CHECK(cublasCreate(&cbh)); CUBLAS_CHECK(cublasSetStream(cbh, st)); }
    const float alpha = W.fmt == WFmt::NVFP4 ? W.g : 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(cbh, CUBLAS_OP_T, CUBLAS_OP_N, W.N, M, W.K, &alpha, B, CUDA_R_16BF, W.K, xin, CUDA_R_16BF, ldx, &beta, yb, CUDA_R_16BF,
                              ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return;
  }
  for (int m0 = 0; m0 < M; m0 += chunk) {   // acc holds chunk rows
    const int mm = std::min(chunk, M - m0);
    blas->gemm_bf16_f32out_ld(xin + (size_t)m0 * ldx, ldx, B, W.K, acc.as<float>(), W.N, mm, W.N, W.K);
    k::epilogue(W, acc.as<float>(), W.N, mm, yb ? yb + (size_t)m0 * ldy : nullptr, yf ? yf + (size_t)m0 * ldy : nullptr, ldy, st);
  }
}

// Device allocations of the serving path (KV growth, context rings): on failure hived's reclaim hook evicts the least recently used idle
//   session (image to host, sequence trimmed — its KV freed) and the allocation is retried; when nothing is left to evict the request fails
//   with an exception instead of aborting the daemon. (Measured 2026-10-10, perf-core 600 W: 16 pooled 32K sessions kept ~51 GB of KV and
//   the 64K prefill's KV growth aborted hived in DevBuf::alloc.)
void Q27Engine::Impl::alloc_serving(DevBuf& b, size_t n, const char* what) {
  int other = 0;
  for (;;) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, n);
    if (e == cudaSuccess) { b.free(); b.p = p; b.n = n; return; }
    cudaGetLastError();
    if (e != cudaErrorMemoryAllocation && other++ < 2) {   // an earlier, non-sticky error reported here — not a memory shortage: retry
      fprintf(stderr, "[q27] cudaMalloc(%zu MiB) for %s returned %s — retrying\n", n >> 20, what, cudaGetErrorString(e));
      continue;
    }
    if (!kv_free.empty()) { CUDA_CHECK(cudaStreamSynchronize(st)); kv_free.clear(); kv_free_bytes = 0; continue; }   // cached KV first
    if (kv_reclaim && *kv_reclaim) {   // (KV tier: the evicted session gives back all of its VRAM KV — kv_pressure; measured 2026-10-13, 32 GB:
      //   reclaims for a new session's recurrent state kept the short caches of 9 trimmed sessions, 779 MiB of a 970 MiB KV budget)
      kv_pressure = kv_tier;
      const bool ev = (*kv_reclaim)();
      kv_pressure = false;
      if (ev) continue;
    }
    size_t fr = 0, tt = 0;
    cudaMemGetInfo(&fr, &tt);
    throw std::runtime_error(std::string("q27: out of GPU memory for ") + what + " (" + std::to_string(n >> 20) + " MiB wanted, " +
                             std::to_string(fr >> 20) + " MiB free, nothing left to evict; " + cudaGetErrorString(e) + ")");
  }
}

// a KV buffer: VRAM (recycled or allocated through the evicting serving path), or — KV tier, when the VRAM KV budget would be exceeded
//   (freed_dev: VRAM bytes the caller releases right after) — pinned host memory mapped for the device
void Q27Engine::Impl::alloc_kv(KvBuf& b, size_t bytes, size_t freed_dev) {
  // KV tier, within the budget: VRAM if it is really there (the budget is a plan; runtime buffers — snapshot slots, drafter rings, sessions'
  //   states — take some of it: measured 2026-10-12, 32 GB, a 32K prompt found 58 MiB free with 1.3 GB of budget left), else the host tier
  // over the budget: the KV of idle pooled sessions goes first (hived's reclaim: the least recently used idle session to a host image, its KV
  //   freed) — measured 2026-10-12, 32 GB batch 16: after c4 / c8 runs the pool held the budget and a 32K prompt decoded at 18 tok/s from host KV
  // (kv_pressure: the evicted session gives back all of its VRAM KV — release_kv — its image is on the host anyway; measured 2026-10-12, 32 GB
  //   partial W4, KV budget 968 MiB: after c1/c4/c8 the pooled sessions held 811 MiB of short caches that the reclaim kept, and every row of
  //   the next 32K prompt went to the host tier)
  kv_pressure = kv_tier;
  for (int tries = 0; kv_tier && tries < 16 && g_kv_dev_bytes.load() + bytes > kv_budget + freed_dev && kv_reclaim && *kv_reclaim; ++tries)
    if (!(*kv_reclaim)()) break;
  kv_pressure = false;
  if (kv_tier && g_kv_dev_bytes.load() + bytes > kv_budget + freed_dev) w4_yield();
  if (kv_tier && g_kv_dev_bytes.load() + bytes <= kv_budget + freed_dev) {
    DevBuf d;
    if (kv_take(d, bytes)) { b.set_dev(std::move(d)); return; }
    for (int tries = 0; tries < 64; ++tries) {   // VRAM first; idle pooled sessions are evicted (hived's reclaim) before spilling to host
      void* p = nullptr;
      if (cudaMalloc(&p, bytes) == cudaSuccess) { d.p = p; d.n = bytes; b.set_dev(std::move(d)); return; }
      cudaGetLastError();
      if (!kv_free.empty()) { CUDA_CHECK(cudaStreamSynchronize(st)); kv_free.clear(); kv_free_bytes = 0; continue; }
      if (!(kv_reclaim && *kv_reclaim && (*kv_reclaim)())) break;
    }
  }
  if (kv_tier) {
    void* hp = nullptr;
    const double th = hive::mono_ms();
    // growable host tier (HIVE_Q27_KV_HOST_GROW=0: a new pinned buffer per growth step): the address range of the plan context reserved, the rows
    //   registered as the cache grows (grow_kv extends in place) — see the measurement at grow_kv
    if (kv_host_grow) {
      const size_t rsv = std::max(bytes, (size_t)(s_plan_ctx > 0 ? s_plan_ctx : (int64_t)c.max_position) * c.kv_width() * 2);
      if (b.set_host_growable(bytes, (rsv + (2u << 20) - 1) / (2u << 20) * (2u << 20))) {
        if (kvlog) {
          ++kl.host_allocs; kl.host_b += bytes; kl.host_ms += hive::mono_ms() - th;
          if (kl.host_allocs <= 8 || kl.host_allocs % 32 == 0)
            fprintf(stderr, "[q27-kv] host range #%ld %zu MiB (%.1f ms) · dev KV %zu MiB / budget %zu (+freed %zu)\n", kl.host_allocs, bytes >> 20,
                    hive::mono_ms() - th, g_kv_dev_bytes.load() >> 20, kv_budget >> 20, freed_dev >> 20);
        }
        return;
      }
    }
    if (cudaHostAlloc(&hp, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
      void* dp = nullptr;
      if (cudaHostGetDevicePointer(&dp, hp, 0) == cudaSuccess && dp == hp) {
        b.set_host(hp, bytes);
        if (kvlog) {
          size_t fr = 0, tt = 0;
          cudaMemGetInfo(&fr, &tt);
          ++kl.host_allocs; kl.host_b += bytes; kl.host_ms += hive::mono_ms() - th;
          if (kl.host_allocs <= 8 || kl.host_allocs % 32 == 0)
            fprintf(stderr, "[q27-kv] host alloc #%ld %zu MiB (%.1f ms) · dev KV %zu MiB / budget %zu (+freed %zu) · VRAM free %zu MiB\n", kl.host_allocs,
                    bytes >> 20, hive::mono_ms() - th, g_kv_dev_bytes.load() >> 20, kv_budget >> 20, freed_dev >> 20, fr >> 20);
        }
        return;
      }
      cudaFreeHost(hp);
    }
    cudaGetLastError();
    static int warned = 0;
    if (warned++ < 4) fprintf(stderr, "[q27] KV tier: pinned host memory refused for %zu MiB — VRAM is used\n", bytes >> 20);
  }
  DevBuf d;
  if (!kv_take(d, bytes)) alloc_serving(d, bytes, "KV");
  b.set_dev(std::move(d));
}

// KV tier, prefill of a sequence with host-tier layers: the staging copy (K | V rows [0, rb) of one layer) in VRAM. It is counted with the VRAM
//   KV (g_kv_dev_bytes, kv_budget) and, when the VRAM is not there, a layer of the same sequence that this chunk has already passed (index
//   < ai: its table rows are not read again in this chunk) moves to the host tier to make room — that layer is then staged as well, which
//   costs one PCIe read of its rows per chunk instead of the attention reading host rows directly (once per 64 query rows). Measured
//   2026-10-12, 32 GB, KV budget 2963 MiB, a 34.8K-token prompt: the staging allocation found 58-74 MiB free, the last chunk read host rows
//   directly and took 10.9 s (TTFT 15.6 s vs 4.2 s at 48 GB). Sizes in 16 MiB steps per half (8192 positions; the old ×1.5 growth left up
//   to a third unused). A refused allocation still leaves the host rows to the kernel (slower, same values).
// the partial W4 copies dropped for the KV tier (HIVE_Q27_W4_PARTIAL=1): the matrices decode from FP8 from now on, the CUDA graphs that read the
//   copies are destroyed (captured again on use), the KV budget grows by the freed bytes. false = nothing to give.
bool Q27Engine::Impl::w4_yield() {
  if (w4_part_store.empty()) return false;
  CUDA_CHECK(cudaStreamSynchronize(st));
  if (cst) CUDA_CHECK(cudaStreamSynchronize(cst));
  size_t b = 0;
  for (auto& d : w4_part_store) b += d.n;
  auto drop = [](GMat& W) { if (W.w4) { W.w4 = nullptr; W.s4 = nullptr; W.s4_swz = nullptr; } };
  for (auto& G : L) for (GMat* m : {&G.qkvz, &G.lin_out, &G.qkv, &G.o, &G.gate, &G.up, &G.down}) drop(*m);
  drop(lm_head);
  for (auto& [k, ge] : graphs) cudaGraphExecDestroy(ge);
  graphs.clear();
  w4_part_store.clear();
  kv_budget += b;
  fprintf(stderr, "[q27] KV tier: the partial W4 decode copies (%zu MiB) gave their VRAM to KV (decode reads the FP8 matrices from now on)\n", b >> 20);
  return true;
}

bool Q27Engine::Impl::stage_alloc(size_t rb, int ai) {
  const size_t step = (size_t)16 << 20;
  const size_t want = 2 * ((rb + step - 1) / step * step);
  CUDA_CHECK(cudaStreamSynchronize(st));
  if (cst) CUDA_CHECK(cudaStreamSynchronize(cst));
  stage_ai = -1;
  if (kv_stage.p) { g_kv_dev_bytes -= kv_stage.n; kv_stage.free(); }
  // order: within the KV budget, or beyond it while 256 MiB of VRAM stay free after it (the margin keeps the runtime's own allocations — states
  //   of new sessions, CUDA graphs — possible) · the VRAM KV of idle sessions (hived's reclaim, all of their KV: kv_pressure) · a passed layer
  //   of this sequence to the host tier · last, any VRAM that is there. Measured 2026-10-13, 32 GB: with every layer of a 32K prompt in the
  //   host tier (nothing to move) and the margin refused, the chunks read host rows directly — 88.9 s for one 8K-position stretch.
  for (int tries = 0; tries <= 64; ++tries) {
    void* p = nullptr;
    size_t fr = 0, tt = 0;
    if (g_kv_dev_bytes.load() + want > kv_budget) { cudaMemGetInfo(&fr, &tt); if (fr < want + ((size_t)256 << 20)) fr = 0; } else fr = 1;
    if (fr && cudaMalloc(&p, want) == cudaSuccess) {
      kv_stage.p = p; kv_stage.n = want; g_kv_dev_bytes += want; stage_seq = pf_seq;
      return true;
    }
    cudaGetLastError();
    if (kv_reclaim && *kv_reclaim) {
      const size_t b0 = g_kv_dev_bytes.load();
      kv_pressure = true;
      const bool ev = (*kv_reclaim)();
      kv_pressure = false;
      if (ev) {
        if (kvlog) fprintf(stderr, "[q27-kv] stage: idle session reclaimed (VRAM KV %zu → %zu MiB)\n", b0 >> 20, g_kv_dev_bytes.load() >> 20);
        continue;
      }
    }
    if (w4_yield()) continue;
    int j = -1;   // the highest already-passed VRAM layer of this sequence
    for (int a = ai - 1; a >= 0 && j < 0; --a) if (pf_seq->kv[a].k.p && !pf_seq->kv[a].k.host() && !pf_seq->kv[a].v.host()) j = a;
    if (j < 0) {
      if (cudaMalloc(&p, want) == cudaSuccess) { kv_stage.p = p; kv_stage.n = want; g_kv_dev_bytes += want; stage_seq = pf_seq; return true; }
      cudaGetLastError();
      return false;
    }
    Q27Seq::Kv& kv = pf_seq->kv[j];
    KvBuf k2, v2;
    const bool g1 = kv_host_grow ? k2.set_host_growable(kv.k.n, (size_t)(s_plan_ctx > 0 ? s_plan_ctx : (int64_t)c.max_position) * c.kv_width() * 2) : false;
    const bool g2 = g1 && v2.set_host_growable(kv.v.n, (size_t)(s_plan_ctx > 0 ? s_plan_ctx : (int64_t)c.max_position) * c.kv_width() * 2);
    if (!g2) return false;
    CUDA_CHECK(kv_memcpy_async(k2.p, kv.k.p, kv.k.n, st));
    CUDA_CHECK(kv_memcpy_async(v2.p, kv.v.p, kv.v.n, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    kv.k = std::move(k2); kv.v = std::move(v2);   // the VRAM buffers freed (KvBuf::reset via the move)
    if (kvlog) fprintf(stderr, "[q27-kv] stage: attention layer %d moved to the host tier for the staging copy (%zu MiB)\n", j, (kv.k.n + kv.v.n) >> 20);
  }
  return false;
}

// the rows [0, pf_pos0) of the first host-tier attention layer at index ≥ from_ai into kv_stage on the copy stream (after the previous use of
//   the buffer); passed_ai = the last attention layer this chunk has passed (stage_alloc may move layers below it to the host tier)
void Q27Engine::Impl::stage_prefetch(int from_ai, int passed_ai, int M) {
  stage_ai = -1;
  if (!stage_pf || !pf_on || depth != 1 || pf_pos0 <= 0) return;
  int a = from_ai;
  while (a < (int)pf_seq->kv.size() && !(pf_seq->kv[a].k.host() || pf_seq->kv[a].v.host())) ++a;
  if (a >= (int)pf_seq->kv.size()) return;
  // (the attention layer must be on the GPU: GPU layers form a prefix, every attention index of pf_seq->kv belongs to a GPU layer)
  const size_t row = (size_t)c.kv_width() * 2, rb = (size_t)(pf_pos0 + M) * row;
  if (kv_stage.n < 2 * rb && !stage_alloc(rb, passed_ai + 1)) return;
  if (!cst) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_used, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev_used, st));
  }
  const Q27Seq::Kv& kv = pf_seq->kv[a];
  CUDA_CHECK(cudaStreamWaitEvent(cst, ev_used, 0));
  CUDA_CHECK(kv_memcpy_async(kv_stage.p, kv.k.p, (size_t)pf_pos0 * row, cst));
  CUDA_CHECK(kv_memcpy_async(kv_stage.as<uint8_t>() + kv_stage.n / 2, kv.v.p, (size_t)pf_pos0 * row, cst));
  CUDA_CHECK(cudaEventRecord(ev_ready, cst));
  stage_ai = a;
}

void Q27Engine::Impl::grow_kv(Q27Seq::Kv& kv, int64_t tokens) {
  if (kv.tokens >= tokens) return;
  // growth: ×1.5 for short caches, at most +8192 positions at a time for long ones (old and new buffer coexist during the copy)
  int64_t nt = std::max<int64_t>(tokens, std::max<int64_t>(256, std::min<int64_t>(kv.tokens * 3 / 2, kv.tokens + 8192)));
  nt = (nt + 255) / 256 * 256;
  const size_t row = (size_t)c.kv_width() * 2;
  // growable host tier: the cache grows in place (registered pages appended), it does not move back to VRAM. Measured 2026-10-12 (32 GB, KV
  //   budget 968 / 2014 MiB, HIVE_Q27_KVLOG): the old growth allocated a new pinned buffer and copied the rows on every step — 72.8K-token prompt
  //   67 pinned allocations = 6.2 s (cudaHostAlloc ~1.3 GB/s) + 5.6 GB copied host → host; letting a host-tier cache return to VRAM when the
  //   budget had room ping-ponged (the budget had room, cudaMalloc then failed: VRAM really short) — 159 allocations, 14.9 GB copied
  if (kv.k.growable() && kv.v.growable()) {
    const double tg = kvlog ? hive::mono_ms() : 0;
    if (kv.k.extend((size_t)nt * row) && kv.v.extend((size_t)nt * row)) {
      if (kvlog) { ++kl.grows; kl.grow_ms += hive::mono_ms() - tg; }
      kv.tokens = nt;
      return;
    }
  }
  KvBuf k2, v2;
  const size_t freed = (kv.k.p && !kv.k.host() ? kv.k.n : 0) + (kv.v.p && !kv.v.host() ? kv.v.n : 0);
  const double tg = kvlog ? hive::mono_ms() : 0;
  alloc_kv(k2, nt * row, freed);
  alloc_kv(v2, nt * row, freed);
  if (kvlog) { ++kl.grows; kl.grow_ms += hive::mono_ms() - tg; if (kv.tokens && (k2.host() || kv.k.host())) kl.grow_copy_b += 2 * kv.tokens * row; }
  nt = (int64_t)(std::min(k2.n, v2.n) / row);   // a recycled buffer may be larger
  if (kv.tokens) {   // (cudaMemcpyDefault: either side may be host-tier memory)
    CUDA_CHECK(kv_memcpy_async(k2.p, kv.k.p, kv.tokens * row, st));
    CUDA_CHECK(kv_memcpy_async(v2.p, kv.v.p, kv.tokens * row, st));
  }
  for (KvBuf* o : {&kv.k, &kv.v}) {   // stream-ordered reuse of VRAM buffers: no sync needed; host-tier buffers are freed after a sync
    if (o->host()) o->reset();
    else if (o->p) kv_give(o->take_dev());
  }
  kv.k = std::move(k2); kv.v = std::move(v2); kv.tokens = nt;
}

int Q27Engine::Impl::build_tables(std::vector<Q27Seq*>& seqs, const std::vector<int64_t>& pos, std::vector<uint8_t>& tab) {
  const int M = (int)seqs.size();
  HIVE_CHECK(M <= tab_rows, "q27: rows exceed the row tables");
  tab.assign(mtp_tab_off(), 0);
  for (int l = 0; l < pl.gpu_layers; ++l) {
    if (c.is_attn[l]) {
      k::AttnRow* ar = reinterpret_cast<k::AttnRow*>(tab.data() + attn_tab_off(attn_index[l]));
      // grow every row's cache first (a regrowth moves the buffer: pointers are taken only after the last one)
      for (int r = 0; r < M; ++r) grow_kv(seqs[r]->kv[attn_index[l]], pos[r] + 1);
      for (int r = 0; r < M; ++r) {
        Q27Seq::Kv& kv = seqs[r]->kv[attn_index[l]];
        ar[r] = k::AttnRow{kv.k.as<bf16>(), kv.v.as<bf16>(), (int)pos[r], (int)pos[r] + 1};
      }
    }
  }
  int ns = 0;
  for (int l = 0; l < pl.gpu_layers; ++l) {
    if (c.is_attn[l]) continue;
    const int li = lin_index[l];
    k::GdnSeg* sg = reinterpret_cast<k::GdnSeg*>(tab.data() + seg_tab_off(li));
    float* const* snaps = snap_ptrs.p ? snap_ptrs.as<float*>() + (size_t)li * 2 * kMaxRows : nullptr;
    ns = 0;
    for (int r = 0; r < M;) {
      int n = 1;
      while (r + n < M && seqs[r + n] == seqs[r] && pos[r + n] == pos[r] + n) ++n;
      int part = -1;
      if (verify) for (size_t p = 0; p < vparts.size(); ++p) if (vparts[p].r0 == r) part = (int)p;
      HIVE_CHECK(ns < kSegs, "q27: too many segments");
      sg[ns] = k::GdnSeg{r, n, seqs[r]->conv[li].as<float>(), seqs[r]->state[li].as<float>(), nullptr, nullptr};
      if (part >= 0 && snaps) {   // conv: slot = row r + t · state: snapshot slot of the part (HIVE_Q27_VSNAP) or read-only
        sg[ns].snap_conv = snaps + r;
        if (vsnap) sg[ns].snap_state = snaps + kMaxRows + part; else sg[ns].keep_state = 1;
      }
      ++ns;
      r += n;
    }
  }
  if (ns == 0) for (int r = 0; r < M;) { int n = 1; while (r + n < M && seqs[r + n] == seqs[r] && pos[r + n] == pos[r] + n) ++n; ++ns; r += n; }
  return ns;
}

void Q27Engine::Impl::upload_tables(const std::vector<uint8_t>& tab) {
  CUDA_CHECK(cudaStreamSynchronize(st));   // the pinned staging buffer may still feed an earlier copy
  memcpy(tab_pin, tab.data(), tab.size());
  CUDA_CHECK(cudaMemcpyAsync(tab_d.p, tab_pin, tab.size(), cudaMemcpyHostToDevice, st));
}

void Q27Engine::Impl::layer_gpu(int l, const GpuLayer& G, const k::AttnRow* rows_dev, const k::GdnSeg* segs_dev, int nsegs, bf16* hd, int M,
                                bool pend_in, bool pend_out, const int32_t* coords) {
  if (!coords) coords = coords_d.as<int32_t>();
  const int H = c.hidden;
  bf16* xb = x.as<bf16>();
  bf16* yb = y.as<bf16>();
  if (pend_in) k::add_norm(hd, yb, G.ln1, M, H, c.rms_eps, xb, st);
  else k::rmsnorm1p(hd, H, G.ln1, M, H, H, c.rms_eps, xb, H, st);
  if (G.attn) {
    const int NH = c.n_heads, NKV = c.n_kv_heads, D = c.head_dim, PW = 2 * c.q_width() + 2 * c.kv_width();
    lin(G.qkv, xb, H, M, proj.as<bf16>(), PW);
    k::attn_prep(rows_dev, M, proj.as<bf16>(), PW, G.q_norm, G.k_norm, c.rms_eps, coords, rope_inv.as<double>(),
                 rope_axis.as<int32_t>(), c.rotary_dim, NH, NKV, D, q.as<bf16>(), false, st);
    if (pf_on && l >= 0) {
      const Q27Seq::Kv& kv = pf_seq->kv[attn_index[l]];
      const bf16* kk = kv.k.as<bf16>();
      const bf16* vv = kv.v.as<bf16>();
      bool staged = false;
      if (kv.k.host() || kv.v.host()) {   // KV tier: the rows [0, pos0 + M) of a host-tier layer copied once to VRAM for this chunk's attention
        const size_t rb = (size_t)(pf_pos0 + M) * c.kv_width() * 2;
        const int ai = attn_index[l];
        const bool pre = stage_ai == ai && depth == 1 && kv_stage.n >= 2 * rb;   // rows [0, pf_pos0) prefetched (stage_prefetch)
        if (!pre) {
          if (cst) CUDA_CHECK(cudaStreamWaitEvent(st, ev_ready, 0));   // a prefetch in flight must not land on top of this copy
          stage_ai = -1;
          if (kv_stage.n < 2 * rb) stage_alloc(rb, ai);
        }
        if (kvlog) { if (kv_stage.n >= 2 * rb) { ++kl.stage_copies; kl.stage_b += 2 * rb; } else ++kl.stage_miss; }
        if (kv_stage.n >= 2 * rb) {
          const size_t off = pre ? (size_t)pf_pos0 * c.kv_width() * 2 : 0;   // prefetched: only this chunk's rows (written by attn_prep above)
          if (pre) CUDA_CHECK(cudaStreamWaitEvent(st, ev_ready, 0));
          CUDA_CHECK(kv_memcpy_async(kv_stage.as<uint8_t>() + off, kv.k.as<uint8_t>() + off, rb - off, st));
          CUDA_CHECK(kv_memcpy_async(kv_stage.as<uint8_t>() + kv_stage.n / 2 + off, kv.v.as<uint8_t>() + off, rb - off, st));
          kk = kv_stage.as<bf16>();
          vv = reinterpret_cast<const bf16*>(kv_stage.as<uint8_t>() + kv_stage.n / 2);
          staged = true;
        }
      }
      k::attention_prefill(q.as<bf16>(), M, pf_pos0, kk, vv, NH, NKV, D, 1.f / std::sqrt((float)D), attf.as<float>(), st);
      if (staged && cst) {   // the buffer is free again after this attention: the next host-tier layer's earlier rows start copying
        CUDA_CHECK(cudaEventRecord(ev_used, st));
        stage_prefetch(attn_index[l] + 1, attn_index[l], M);
      }
    } else {
      k::attention(rows_dev, M, q.as<bf16>(), NH, NKV, D, 1.f / std::sqrt((float)D), attf.as<float>(), st);
    }
    k::attn_gate(attf.as<float>(), proj.as<bf16>(), PW, M, NH, D, ao.as<bf16>(), st);
    lin(G.o, ao.as<bf16>(), c.q_width(), M, yb, H);
  } else {
    const int C = c.lin_conv_ch(), VW = c.lin_v_width(), VH = c.lin_v_heads, PW = C + VW + 2 * VH;
    lin(G.qkvz, xb, H, M, proj.as<bf16>(), PW);
    lin(G.ba, xb, H, M, proj.as<bf16>() + C + VW, PW);
    k::GdnW gw{G.conv_w, G.A_log, G.dt_bias, G.lin_norm, c.rms_eps};
    const int li = l >= 0 ? lin_index[l] : 0;
    const bool vsave = verify && vq_y.p && l >= 0;   // verify: this layer's GDN inputs stay for the rollback replay
    bf16* gyl = vsave ? vq_y.as<bf16>() + (size_t)li * kMaxRows * C : gy.as<bf16>();
    float* qkl = vsave ? vq_qk.as<float>() + (size_t)li * kMaxRows * 2 * c.lin_k_width() : qk.as<float>();
    float* gbl = vsave ? vq_gb.as<float>() + (size_t)li * kMaxRows * 2 * VH : gb.as<float>();
    k::gdn(gw, segs_dev, nsegs, M, proj.as<bf16>(), PW, c.lin_k_heads, VH, gyl, qkl, gbl, go.as<float>(), gout.as<bf16>(), st);
    lin(G.lin_out, gout.as<bf16>(), VW, M, yb, H);
  }
  k::add_norm(hd, yb, G.ln2, M, H, c.rms_eps, xb, st);
  const int I = c.inter;
  if (M <= kMaxRows && (G.gate.w4 || M > 16 || !(dq && G.gate.fmt != WFmt::BF16))) k::gemv_gu(k::decode_view(G.gate), k::decode_view(G.up), xb, H, M, gu.as<bf16>(), 2 * I, st);
  else {
    lin(G.gate, xb, H, M, gu.as<bf16>(), 2 * I);
    lin(G.up, xb, H, M, gu.as<bf16>() + I, 2 * I);
    k::silu_mul(gu.as<bf16>(), 2 * I, gu.as<bf16>() + I, 2 * I, M, I, st);
  }
  lin(G.down, gu.as<bf16>(), 2 * I, M, yb, H);
  if (!pend_out) k::add_round(hd, yb, (size_t)M * H, st);
}

// every layer (all on the GPU) and the head of rows [r0, r0 + nr) into logits_pin — one stream sequence without host syncs (graph body)
void Q27Engine::Impl::gpu_step(int M, int nsegs, bf16* hd, int r0, int nr, const bf16* norm_w, bool fh_copy, float* dst) {
  const int H = c.hidden, V = c.vocab;
  for (int l = 0; l < c.n_layers; ++l) {
    layer_gpu(l, L[l], tab_d.as<k::AttnRow>() + (c.is_attn[l] ? attn_tab_off(attn_index[l]) / sizeof(k::AttnRow) : 0),
              reinterpret_cast<const k::GdnSeg*>(tab_d.as<uint8_t>() + (c.is_attn[l] ? 0 : seg_tab_off(lin_index[l]))), nsegs, hd, M, l > 0, true);
    if (dflash && pl.mtp_gpu && df_slot[l] >= 0) df::capture(hd, y.as<bf16>(), M, H, df_cap.as<bf16>() + (size_t)df_slot[l] * H, df::kLayers * H, st);
  }
  k::add_norm(hd, y.as<bf16>(), norm_w, M, H, c.rms_eps, x.as<bf16>(), st);   // last residual add + final norm of every row
  if (fh_copy) CUDA_CHECK(cudaMemcpyAsync(fh.p, hd, (size_t)M * H * 2, cudaMemcpyDeviceToDevice, st));
  lin(lm_head, x.as<bf16>() + (size_t)r0 * H, H, nr, nullptr, V, logits_d.as<float>());
  if (dst) CUDA_CHECK(cudaMemcpyAsync(dst, logits_d.p, (size_t)nr * V * 4, cudaMemcpyDeviceToHost, st));   // null: logits stay on the device
}

void Q27Engine::Impl::set_rope(std::vector<Q27Seq*>& seqs, std::vector<int64_t>& pos, bool prefill, std::vector<int32_t>& coords) {
  const int M = (int)seqs.size();
  coords.resize((size_t)M * 3);
  for (int r = 0; r < M; ++r) {
    Q27Seq& s = *seqs[r];
    const int64_t p = pos[r], gi = emb_base + r;
    const cpu::Q27CpuEngine::EmbedOverride* sp = nullptr;
    if (prefill) for (const auto& e : emb_over) if (e.gw > 0 && gi >= e.row && gi < e.row + e.n) { sp = &e; break; }
    int32_t* o = coords.data() + (size_t)r * 3;
    if (!sp) { o[0] = o[1] = o[2] = (int32_t)(p + s.rope_delta); }
    else {
      const int64_t kk = gi - sp->row, st0 = (p - kk) + s.rope_delta;
      o[0] = (int32_t)st0; o[1] = (int32_t)(st0 + kk / sp->gw); o[2] = (int32_t)(st0 + kk % sp->gw);
      if (kk == sp->n - 1) s.rope_delta += std::max(sp->gh, sp->gw) - sp->n;
    }
    if ((int64_t)s.rope.size() < (p + 1) * 3) s.rope.resize((size_t)(p + 1) * 3);
    memcpy(s.rope.data() + (size_t)p * 3, o, 12);
  }
}

void Q27Engine::Impl::embed(const int32_t* ids, int M, bool prefill, bf16* out) {
  const int H = c.hidden;
  if (emb_ptr && !(prefill && !emb_over.empty())) {   // table in VRAM: gather on the device, no host round trip
    CUDA_CHECK(cudaMemcpyAsync(ids_d.p, ids, (size_t)M * 4, cudaMemcpyHostToDevice, st));
    k::embed_rows(emb_ptr, ids_d.as<int32_t>(), M, H, out, st);
    return;
  }
  const uint16_t* E = cpu->embed_table();
  std::vector<uint16_t> hb((size_t)M * H);
  for (int r = 0; r < M; ++r) memcpy(hb.data() + (size_t)r * H, E + (size_t)ids[r] * H, (size_t)H * 2);
  if (prefill)
    for (const auto& o : emb_over) {
      const int64_t a = std::max<int64_t>(o.row, emb_base), b = std::min<int64_t>(o.row + o.n, emb_base + M);
      for (int64_t g = a; g < b; ++g)
        for (int d = 0; d < H; ++d) hb[(size_t)(g - emb_base) * H + d] = f2b(o.emb[(size_t)(g - o.row) * H + d]);
    }
  CUDA_CHECK(cudaMemcpyAsync(out, hb.data(), (size_t)M * H * 2, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaStreamSynchronize(st));   // hb is reused
}

void Q27Engine::Impl::head(const bf16* hd, int M, const bf16* norm_w, float* logits_host) {
  const int H = c.hidden, V = c.vocab;
  if (pl.head_gpu) {
    k::rmsnorm1p(hd, H, norm_w, M, H, H, c.rms_eps, x.as<bf16>(), H, st);
    for (int m0 = 0; m0 < M; m0 += 8) {
      const int mm = std::min(8, M - m0);
      lin(lm_head, x.as<bf16>() + (size_t)m0 * H, H, mm, nullptr, V, logits_d.as<float>());
      CUDA_CHECK(cudaMemcpyAsync(logits_host + (size_t)m0 * V, logits_d.p, (size_t)mm * V * 4, cudaMemcpyDeviceToHost, st));
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    return;
  }
  std::vector<uint16_t> t((size_t)M * H);
  std::vector<float> f((size_t)M * H);
  CUDA_CHECK(cudaMemcpyAsync(t.data(), hd, t.size() * 2, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  for (size_t i = 0; i < f.size(); ++i) f[i] = b2f(t[i]);
  cpu->head_logits(f.data(), M, logits_host);
}

void Q27Engine::Impl::forward(std::vector<Q27Seq*> seqs, std::vector<int64_t> pos, const int32_t* ids, bool prefill, float* logits, bool all_logits) {
  const int M = (int)seqs.size(), H = c.hidden;
  HIVE_CHECK(M >= 1 && M <= std::max(chunk, 8), "q27: forward rows");
  const double t0 = hive::mono_ms();
  for (int r = 0; r < M; ++r) if (pos[r] + 1 > seqs[r]->cap) throw std::runtime_error("q27: sequence capacity exceeded");
  ++depth;
  struct Depth { int& d; ~Depth() { --d; } } dg{depth};
  // inner forwards (layer yields) use their own stream copy
  DevBuf hown;
  bf16* hd = h.as<bf16>();
  if (depth > 1) { hown.alloc((size_t)M * H * 2); hd = hown.as<bf16>(); }
  if (!prefill && kv_stage.p && std::find(seqs.begin(), seqs.end(), stage_seq) != seqs.end()) {   // its prompt is done: the VRAM back to KV
    CUDA_CHECK(cudaStreamSynchronize(st));
    if (cst) CUDA_CHECK(cudaStreamSynchronize(cst));
    g_kv_dev_bytes -= kv_stage.n; kv_stage.free(); stage_seq = nullptr; stage_ai = -1;
  }
  std::vector<int32_t> coords;
  set_rope(seqs, pos, prefill, coords);
  CUDA_CHECK(cudaMemcpyAsync(coords_d.p, coords.data(), coords.size() * 4, cudaMemcpyHostToDevice, st));
  embed(ids, M, prefill, hd);
  std::vector<uint8_t> tab;
  const int nsegs = build_tables(seqs, pos, tab);
  // one sequence, consecutive positions, more than 8 rows: tensor-core prefill attention (HIVE_Q27_FA=0 keeps the per-row kernel)
  static const bool fa_on = !(getenv("HIVE_Q27_FA") && strcmp(getenv("HIVE_Q27_FA"), "0") == 0);
  bool one = M > 8 && fa_on;
  for (int r = 1; one && r < M; ++r) one = seqs[r] == seqs[0] && pos[r] == pos[0] + r;
  const bool pf_save = pf_on; Q27Seq* const pf_seq_save = pf_seq; const int pf_pos_save = pf_pos0;
  pf_on = one; pf_seq = seqs[0]; pf_pos0 = (int)pos[0];
  struct PfRestore { Impl& I; bool on; Q27Seq* s; int p; ~PfRestore() { I.pf_on = on; I.pf_seq = s; I.pf_pos0 = p; } } pf_restore{*this, pf_save, pf_seq_save, pf_pos_save};
  upload_tables(tab);
  if (pf_on && kv_tier && depth == 1) stage_prefetch(0, -1, M);   // KV tier: the first host-tier layer's earlier rows (stage_prefetch)
  const bool hooks = layer_hook && *layer_hook;
  const int r0 = all_logits ? 0 : M - 1, nr = all_logits ? M : 1;
  // whole-step CUDA graph: decode / verify with every layer and the head on the GPU, no validation hooks
  if (use_graphs && depth == 1 && !prefill && logits && nr <= kMaxRows && pl.gpu_layers == c.n_layers && pl.head_gpu && !hooks) {
    // the logits go straight to a registered caller buffer (no host copy) or to logits_pin
    auto hd_it = host_dst.find(logits);
    float* dst = !host_logits ? nullptr : hd_it != host_dst.end() ? logits : logits_pin;
    const int64_t key = (((((int64_t)M * 128 + nsegs) * 64 + nr) * 2 + (verify ? 1 : 0)) * 8 + (hd_it != host_dst.end() ? 1 + hd_it->second : 0)) * 2
                        + (host_logits ? 0 : 1);   // verify graphs bind the per-layer replay buffers · device-only logits
    auto it = graphs.find(key);
    if (it == graphs.end()) {
      cudaGraph_t g;
      CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
      gpu_step(M, nsegs, h.as<bf16>(), r0, nr, final_norm, pl.mtp_gpu && !dflash, dst);
      CUDA_CHECK(cudaStreamEndCapture(st, &g));
      cudaGraphExec_t ge;
      CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
      ++captures;
      CUDA_CHECK(cudaGraphDestroy(g));
      it = graphs.emplace(key, ge).first;
    }
    const double tg = hive::mono_ms();
    CUDA_CHECK(cudaGraphLaunch(it->second, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    if (dst && dst != logits) memcpy(logits, logits_pin, (size_t)nr * c.vocab * 4);   // the runtime reads logits_host() in place
    st_.ms_gpu_layers += hive::mono_ms() - tg;
    const double tm = hive::mono_ms();
    if (pl.mtp_gpu && mtp_k > 0) { if (dflash) df_ingest(seqs, pos, df_cap.as<bf16>()); else mtp_after_forward(seqs, pos, ids, prefill); }
    st_.ms_mtp += hive::mono_ms() - tm;
    CUDA_CHECK(cudaStreamSynchronize(st));
    for (int r = 0; r < M; ++r) { seqs[r]->pos = pos[r] + 1; seqs[r]->tokens.push_back(ids[r]); }
    ++st_.steps;
    st_.ms_total += hive::mono_ms() - t0;
    return;
  }
  bf16* capb = df_cap.as<bf16>();   // an inner forward (layer yield) captures into its own buffer
  DevBuf capown;
  if (dflash && depth > 1) { capown.alloc((size_t)M * df::kLayers * H * 2); capb = capown.as<bf16>(); }
  std::vector<cpu::Q27CpuEngine::Row> crow(M);
  for (int r = 0; r < M; ++r) crow[r] = {&seqs[r]->cs, pos[r]};
  bool on_host = false;   // the stream currently lives in hf (CPU layers)
  std::vector<float> hf;   // host rows: local, an inner forward (layer yield) has its own
  std::vector<uint16_t> hb;
  auto to_host = [&]() {
    const double tx = hive::mono_ms();
    hb.resize((size_t)M * H); hf.resize((size_t)M * H);
    CUDA_CHECK(cudaMemcpyAsync(hb.data(), hd, (size_t)M * H * 2, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    for (size_t i = 0; i < hf.size(); ++i) hf[i] = b2f(hb[i]);
    st_.ms_xfer += hive::mono_ms() - tx;
  };
  auto to_dev = [&]() {
    const double tx = hive::mono_ms();
    hb.resize((size_t)M * H);
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = f2b(hf[i]);
    CUDA_CHECK(cudaMemcpyAsync(hd, hb.data(), (size_t)M * H * 2, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    st_.ms_xfer += hive::mono_ms() - tx;
  };
  for (int l = 0; l < c.n_layers; ++l) {
    if (prefill && l > 0 && layer_inject && *layer_inject) {
      if (!on_host) to_host();
      (*layer_inject)(l, hf.data(), M);
      if (L[l].on) to_dev(); else on_host = true;
    }
    const double tl = hive::mono_ms();
    if (L[l].on) {
      if (on_host) { to_dev(); on_host = false; }
      layer_gpu(l, L[l], tab_d.as<k::AttnRow>() + (c.is_attn[l] ? attn_tab_off(attn_index[l]) / sizeof(k::AttnRow) : 0),
                reinterpret_cast<const k::GdnSeg*>(tab_d.as<uint8_t>() + (c.is_attn[l] ? 0 : seg_tab_off(lin_index[l]))), nsegs, hd, M);
      if (dflash && pl.mtp_gpu && df_slot[l] >= 0) df::capture(hd, nullptr, M, H, capb + (size_t)df_slot[l] * H, df::kLayers * H, st);
      if (hooks) { to_host(); on_host = false; }
      st_.ms_gpu_layers += hive::mono_ms() - tl;
    } else {
      if (!on_host) { to_host(); on_host = true; }
      cpu->run_layer(l, crow.data(), M, coords.data(), hf.data());
      if (df_cpu_capture && df_slot[l] >= 0) {   // drafter context from a CPU layer: the host stream rows (bf16 values) into the capture slot
        std::vector<uint16_t> cb((size_t)M * H);
        for (size_t i = 0; i < cb.size(); ++i) cb[i] = f2b(hf[i]);
        CUDA_CHECK(cudaMemcpy2DAsync(capb + (size_t)df_slot[l] * H, (size_t)df::kLayers * H * 2, cb.data(), (size_t)H * 2, (size_t)H * 2, M,
                                     cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
      }
      st_.ms_cpu_layers += hive::mono_ms() - tl;
    }
    if (hooks) (*layer_hook)(l, hf.data(), M);
    if (prefill && depth == 1 && layer_boundary && *layer_boundary && l + 1 < c.n_layers &&
        (!layer_boundary_due || !*layer_boundary_due || (*layer_boundary_due)())) {
      if (!on_host) CUDA_CHECK(cudaStreamSynchronize(st));
      auto parked = std::move(emb_over);
      const int64_t base = emb_base;
      emb_over.clear(); emb_base = 0;
      (*layer_boundary)(l);
      emb_over = std::move(parked); emb_base = base;
      CUDA_CHECK(cudaMemcpyAsync(coords_d.p, coords.data(), coords.size() * 4, cudaMemcpyHostToDevice, st));   // an inner forward used it
      upload_tables(tab);
    }
  }
  if (on_host) to_dev();
  if (pl.mtp_gpu) CUDA_CHECK(cudaMemcpyAsync(fh.p, hd, (size_t)M * H * 2, cudaMemcpyDeviceToDevice, st));
  const double th = hive::mono_ms();
  if (logits) head(hd + (size_t)r0 * H, nr, final_norm, logits);
  st_.ms_head += hive::mono_ms() - th;
  const double tm = hive::mono_ms();
  if (pl.mtp_gpu && mtp_k > 0) { if (dflash) df_ingest(seqs, pos, capb); else mtp_after_forward(seqs, pos, ids, prefill); }
  st_.ms_mtp += hive::mono_ms() - tm;
  CUDA_CHECK(cudaStreamSynchronize(st));
  for (int r = 0; r < M; ++r) { seqs[r]->pos = pos[r] + 1; seqs[r]->tokens.push_back(ids[r]); }
  if (!prefill) ++st_.steps;
  st_.ms_total += hive::mono_ms() - t0;
}

// ---- MTP ------------------------------------------------------------------------------------------------------------------------------------
void Q27Engine::Impl::mtp_input(const bf16* emb, const bf16* hrows, int n, bf16* out, const bf16* prenorm) {
  const int H = c.hidden;
  if (prenorm) {   // the stream through its producer's final norm first (y is free here: n ≤ 8 rows)
    k::rmsnorm1p(hrows, H, prenorm, n, H, H, c.rms_eps, y.as<bf16>(), H, st);
    hrows = y.as<bf16>();
  }
  k::rmsnorm1p(emb, H, mtp_en, n, H, H, c.rms_eps, cat.as<bf16>(), 2 * H, st);
  k::rmsnorm1p(hrows, H, mtp_hn, n, H, H, c.rms_eps, cat.as<bf16>() + H, 2 * H, st);
  lin(mtp_fc, cat.as<bf16>(), 2 * H, n, out, H);
}

// K / V writes of MTP entries (sequence, entry position) from streams hrows [n, H] and the next tokens (embedding rows; override rows of an
//   image in the current prefill: ov_rows[i] ≥ 0 = that row of emb_over's range)
void Q27Engine::Impl::mtp_entries(std::vector<Q27Seq*>& seqs, std::vector<int64_t>& pos, const bf16* hrows, const int32_t* next_ids,
                                  const std::vector<int64_t>& ov_rows) {
  const int n = (int)seqs.size(), H = c.hidden;
  if (n == 0) return;
  for (int i0 = 0; i0 < n; i0 += tab_rows) {   // entries in chunks of the work-buffer rows (a prompt's entries in one pass)
    const int nn = std::min(tab_rows, n - i0);
    // embeddings of the next tokens
    hb.resize((size_t)nn * H);
    const uint16_t* E = cpu->embed_table();
    for (int i = 0; i < nn; ++i) {
      memcpy(hb.data() + (size_t)i * H, E + (size_t)next_ids[i0 + i] * H, (size_t)H * 2);
      if (!ov_rows.empty() && ov_rows[i0 + i] >= 0)
        for (const auto& o : emb_over)
          if (ov_rows[i0 + i] >= o.row && ov_rows[i0 + i] < o.row + o.n)
            for (int d = 0; d < H; ++d) hb[(size_t)i * H + d] = f2b(o.emb[(size_t)(ov_rows[i0 + i] - o.row) * H + d]);
    }
    bf16* emb = me_emb.as<bf16>();
    bf16* hm = me_hm.as<bf16>();
    bool ov_any = false;
    for (int i = 0; i < nn && !ov_rows.empty(); ++i) ov_any = ov_any || ov_rows[i0 + i] >= 0;
    if (emb_ptr && !ov_any) {
      CUDA_CHECK(cudaMemcpyAsync(ids_d.as<int32_t>() + tab_rows, next_ids + i0, (size_t)nn * 4, cudaMemcpyHostToDevice, st));
      k::embed_rows(emb_ptr, ids_d.as<int32_t>() + tab_rows, nn, H, emb, st);
    } else {
      CUDA_CHECK(cudaMemcpyAsync(emb, hb.data(), (size_t)nn * H * 2, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaStreamSynchronize(st));
    }
    mtp_input(emb, hrows + (size_t)i0 * H, nn, hm, mtp_post_main ? final_norm : nullptr);
    k::rmsnorm1p(hm, H, mtpL.ln1, nn, H, H, c.rms_eps, x.as<bf16>(), H, st);
    const int PW = 2 * c.q_width() + 2 * c.kv_width();
    lin(mtpL.qkv, x.as<bf16>(), H, nn, proj.as<bf16>(), PW);
    std::vector<k::AttnRow> ar(nn);
    std::vector<int32_t> cd((size_t)nn * 3);
    for (int i = 0; i < nn; ++i) grow_kv(seqs[i0 + i]->mtp, pos[i0 + i] + 1 + 8);   // all growth before any pointer is taken
    for (int i = 0; i < nn; ++i) {
      Q27Seq& s = *seqs[i0 + i];
      const int64_t j = pos[i0 + i];
      ar[i] = k::AttnRow{s.mtp.k.as<bf16>(), s.mtp.v.as<bf16>(), (int)j, (int)j + 1};
      if ((int64_t)s.rope.size() >= (j + 1) * 3) memcpy(cd.data() + (size_t)i * 3, s.rope.data() + (size_t)j * 3, 12);
      else cd[i * 3] = cd[i * 3 + 1] = cd[i * 3 + 2] = (int32_t)(j + s.rope_delta);
      s.mtp_written = std::max(s.mtp_written, j + 1);
    }
    CUDA_CHECK(cudaMemcpyAsync(rows_d.p, ar.data(), nn * sizeof(k::AttnRow), cudaMemcpyHostToDevice, st));
    int32_t* cdd = me_cd.as<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(cdd, cd.data(), (size_t)nn * 12, cudaMemcpyHostToDevice, st));
    k::attn_prep(rows_d.as<k::AttnRow>(), nn, proj.as<bf16>(), PW, mtpL.q_norm, mtpL.k_norm, c.rms_eps, cdd, rope_inv.as<double>(),
                 rope_axis.as<int32_t>(), c.rotary_dim, c.n_heads, c.n_kv_heads, c.head_dim, q.as<bf16>(), true, st);
  }
}

void Q27Engine::Impl::mtp_after_forward(std::vector<Q27Seq*>& seqs, std::vector<int64_t>& pos, const int32_t* ids, bool prefill) {
  const int M = (int)seqs.size(), H = c.hidden;
  std::vector<Q27Seq*> es;
  std::vector<int64_t> ep, ov;
  std::vector<int32_t> nx;
  HIVE_CHECK((size_t)(M + 16) * H * 2 <= eh_buf.n, "q27: MTP entry rows");
  DevBuf& eh = eh_buf;
  int ne = 0;
  auto add = [&](Q27Seq* s, int64_t j, const bf16* hsrc, int r_next) {
    CUDA_CHECK(cudaMemcpyAsync(eh.as<bf16>() + (size_t)ne * H, hsrc, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
    es.push_back(s); ep.push_back(j); nx.push_back(ids[r_next]); ov.push_back(prefill ? emb_base + r_next : -1);
    ++ne;
  };
  for (int r = 0; r < M;) {
    Q27Seq& s = *seqs[r];
    int n = 1;
    while (r + n < M && seqs[r + n] == &s && pos[r + n] == pos[r] + n) ++n;
    if (s.h_last_valid && pos[r] >= 1) add(&s, pos[r] - 1, s.h_last.as<bf16>(), r);
    if (!verify) {
      for (int t = 0; t + 1 < n; ++t) add(&s, pos[r] + t, fh.as<bf16>() + (size_t)(r + t) * H, r + t + 1);
      if (!s.h_last.p) s.h_last.alloc((size_t)H * 2);
      CUDA_CHECK(cudaMemcpyAsync(s.h_last.p, fh.as<bf16>() + (size_t)(r + n - 1) * H, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
      s.h_last_valid = true;
    }
    r += n;
  }
  mtp_entries(es, ep, eh.as<bf16>(), nx.data(), ov);
}

int Q27Engine::Impl::mtp_draft(Q27Seq& s, int32_t tok, int kk, int32_t* drafts, float* conf) {
  if (dflash) return pl.mtp_gpu && kk > 0 ? df_draft(s, tok, kk, drafts, conf) : 0;
  if (!pl.mtp_gpu || kk <= 0 || !s.h_last_valid || s.pos < 1) return 0;
  kk = std::min(kk, 7);
  const int H = c.hidden, V = c.vocab;
  if (!dr_buf.p) dr_buf.alloc((size_t)3 * H * 2);
  bf16* hcur = dr_buf.as<bf16>();
  bf16* emb = hcur + H;
  bf16* hm = emb + H;
  CUDA_CHECK(cudaMemcpyAsync(hcur, s.h_last.p, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
  const int64_t j0 = s.pos - 1;
  // coordinates and attention rows of the k entries j0 … j0 + k − 1 (cache grown first: pointers taken after the last growth)
  grow_kv(s.mtp, j0 + kk + 8);
  std::vector<int32_t> cd((size_t)kk * 3);
  std::vector<k::AttnRow> ar(kk);
  for (int i = 0; i < kk; ++i) {
    const int64_t j = j0 + i;
    if ((int64_t)s.rope.size() >= (j + 1) * 3) memcpy(cd.data() + (size_t)i * 3, s.rope.data() + (size_t)j * 3, 12);
    else cd[i * 3] = cd[i * 3 + 1] = cd[i * 3 + 2] = (int32_t)(j + s.rope_delta);
    ar[i] = k::AttnRow{s.mtp.k.as<bf16>(), s.mtp.v.as<bf16>(), (int)j, (int)j + 1};
  }
  int32_t* cdd = me_cd.as<int32_t>();
  k::AttnRow* ard = reinterpret_cast<k::AttnRow*>(tab_d.as<uint8_t>() + mtp_tab_off());
  CUDA_CHECK(cudaMemcpyAsync(cdd, cd.data(), cd.size() * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(ard, ar.data(), ar.size() * sizeof(k::AttnRow), cudaMemcpyHostToDevice, st));
  const bool dev_chain = emb_ptr && pl.head_gpu;
  float* res = small_d.as<float>();           // per draft i: res[i·64 + 0] = id bits, res[i·64 + 1] = p (+ the argmax work area)
  if (dev_chain) {
    // draft 0 takes tok: written as the "previous result" slot −1 (res[(kk)·64]) so every step reads its token from the device
    int32_t* tok_slot = reinterpret_cast<int32_t*>(res + (size_t)kk * 200);
    CUDA_CHECK(cudaMemcpyAsync(tok_slot, &tok, 4, cudaMemcpyHostToDevice, st));
    for (int i = 0; i < kk; ++i) {
      const int32_t* idp = i == 0 ? tok_slot : reinterpret_cast<const int32_t*>(res + (size_t)(i - 1) * 200);
      const GMat& dh = draft_ids.empty() ? lm_head : draft_head;
      k::embed_by_id(emb_ptr, idp, H, emb, st);
      mtp_input(emb, hcur, 1, hm, i == 0 ? (mtp_post_main ? final_norm : nullptr) : (mtp_post_chain ? mtp_norm : nullptr));
      layer_gpu(-1, mtpL, ard + i, nullptr, 0, hm, 1, false, false, cdd + 3 * i);
      k::rmsnorm1p(hm, H, mtp_norm, 1, H, H, c.rms_eps, x.as<bf16>(), H, st);
      lin(dh, x.as<bf16>(), H, 1, nullptr, dh.N, logits_d.as<float>());
      k::argmax_prob(logits_d.as<float>(), dh.N, res + (size_t)i * 200, st);
      if (!draft_ids.empty()) k::map_ids(reinterpret_cast<int32_t*>(res + (size_t)i * 200), draft_ids_d.as<int32_t>(), st);
      CUDA_CHECK(cudaMemcpyAsync(hcur, hm, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
    }
    std::vector<float> out((size_t)kk * 200);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), res, out.size() * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    for (int i = 0; i < kk; ++i) { memcpy(&drafts[i], &out[(size_t)i * 200], 4); if (conf) conf[i] = out[(size_t)i * 200 + 1]; }
    return kk;
  }
  std::vector<float> lg(V);
  int32_t t = tok;
  for (int i = 0; i < kk; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(emb, cpu->embed_table() + (size_t)t * H, (size_t)H * 2, cudaMemcpyHostToDevice, st));
    mtp_input(emb, hcur, 1, hm, i == 0 ? (mtp_post_main ? final_norm : nullptr) : (mtp_post_chain ? mtp_norm : nullptr));
    layer_gpu(-1, mtpL, ard + i, nullptr, 0, hm, 1, false, false, cdd + 3 * i);
    int best = 0;
    float pbest = 0.f;
    head(hm, 1, mtp_norm, lg.data());
    for (int v = 1; v < V; ++v) if (lg[v] > lg[best]) best = v;
    double z = 0;
    for (int v = 0; v < V; ++v) z += std::exp((double)lg[v] - lg[best]);
    pbest = (float)(1.0 / z);
    drafts[i] = best;
    if (conf) conf[i] = pbest;
    CUDA_CHECK(cudaMemcpyAsync(hcur, hm, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
    t = best;
  }
  return kk;
}

// ---- DFlash2 drafter (hive/q27/q27_dflash.h; math after z-lab/dflash, MIT License, Copyright (c) 2026 Z Lab) ----------------------------
void Q27Engine::Impl::df_load(size_t& bytes_out, bool load) {
  std::ifstream cf(df_dir + "/config.json");
  HIVE_CHECK((bool)cf, "q27: HIVE_Q27_DFLASH: no config.json in " + df_dir);
  const nlohmann::json j = nlohmann::json::parse(cf);
  const auto& d = j.at("dflash_config");
  dsp = j.contains("markov_rank") || d.contains("markov_rank");
  const bool common = j.at("num_hidden_layers").get<int>() == df::kLayers && j.at("head_dim").get<int>() == df::kHeadDim &&
                      j.at("num_attention_heads").get<int>() == df::kHeads && j.at("num_key_value_heads").get<int>() == df::kKvHeads &&
                      j.at("hidden_size").get<int>() == c.hidden && j.at("intermediate_size").get<int>() == c.inter &&
                      d.at("target_layer_ids").size() == df::kLayers;
  if (dsp) {   // DSpark: block of 7 noise rows (anchor + 6 masks), all 7 rows predict, full-context attention, YaRN rope, rank-256 Markov head
    HIVE_CHECK(common && j.value("block_size", 0) == 7 && j.value("markov_rank", 0) == df::kRank && j.value("markov_head_type", "") == "vanilla" &&
                   j.value("enable_confidence_head", false) && j.value("confidence_head_with_markov", false),
               "q27: HIVE_Q27_DFLASH: unsupported DSpark drafter shape");
    df_B = 7; df_out0 = 0; df_window = 0;
  } else {
    HIVE_CHECK(common && d.at("block_size").get<int>() == df::kBlock && d.at("selector_top_k").get<int>() == df::kTopK &&
                   d.at("selector_rank").get<int>() == df::kRank && d.at("conv_group_size").get<int>() == df::kGroup &&
                   d.at("conv_kernel_size").get<int>() == 2 && j.value("sliding_window", 0) == df::kWindow,
               "q27: HIVE_Q27_DFLASH: unsupported drafter shape (expected the Qwen3.8-27B DFlash2 or DSpark drafter)");
    df_B = df::kBlock; df_out0 = 1; df_window = df::kWindow;
  }
  SafetensorsFile sf(df_dir + "/model.safetensors");
  if (!load) {   // device bytes of the drafter as it will be stored: BF16, or FP8 per row (HIVE_Q27_DFLASH_FP8: matrices except fc, + fp32 row
    //   scales, + NVFP4 copies with HIVE_Q27_DFLASH_FP4) — the file size overstated the FP8 drafter ~1.4x in the placement (2026-10-12)
    const bool f8 = (getenv("HIVE_Q27_DFLASH_FP8") && strcmp(getenv("HIVE_Q27_DFLASH_FP8"), "1") == 0) ||
                    (getenv("HIVE_Q27_DFLASH_FP4") && strcmp(getenv("HIVE_Q27_DFLASH_FP4"), "1") == 0);
    const bool f4 = getenv("HIVE_Q27_DFLASH_FP4") && strcmp(getenv("HIVE_Q27_DFLASH_FP4"), "1") == 0;
    size_t b = 0;
    for (const auto& [name, t] : sf.tensors()) {
      const bool mat = t.shape.size() == 2 && t.shape[0] > 1 && t.shape[1] > 1 && name != "fc.weight" && name.find("codebook") == std::string::npos &&
                       name.find("markov_head") == std::string::npos && name.find("confidence_head") == std::string::npos;
      if (f8 && mat) b += (size_t)t.numel() + (size_t)t.shape[0] * 4 + 512 + (f4 ? (size_t)t.numel() / 2 + (size_t)t.numel() / 16 + 512 : 0);
      else b += t.nbytes + 256;
    }
    for (int l = 0; l < df::kLayers; ++l)   // the stacked context K / V projection (df_kvall) is a second copy of every layer's k_proj / v_proj
      for (const char* n : {"self_attn.k_proj.weight", "self_attn.v_proj.weight"})
        if (const TensorInfo* t = sf.find("layers." + std::to_string(l) + "." + n)) b += f8 ? (size_t)t->numel() + (size_t)t->shape[0] * 4 : t->nbytes;
    bytes_out = b;
    return;
  }
  for (int l = 0; l < 64; ++l) df_slot[l] = -1;
  for (int i = 0; i < df::kLayers; ++i) {
    const int t = d.at("target_layer_ids")[i].get<int>();
    HIVE_CHECK(t >= 0 && t < c.n_layers && t < 64, "q27: DFlash target layer");
    df_slot[t] = i;
  }
  mask_id = d.at("mask_token_id").get<int>();
  auto T = [&](const std::string& n) -> const TensorInfo& {
    const TensorInfo* t = sf.find(n);
    HIVE_CHECK(t && t->dtype == "BF16", "q27: DFlash tensor missing or not BF16: " + n);
    return *t;
  };
  auto up = [&](const std::vector<std::string>& names) -> const uint8_t* {   // bf16 tensors stacked along dim 0
    size_t n = 0;
    for (auto& nm : names) n += T(nm).nbytes;
    DevBuf b(n);
    size_t off = 0;
    for (auto& nm : names) { const TensorInfo& t = T(nm); CUDA_CHECK(cudaMemcpy(b.as<uint8_t>() + off, t.data, t.nbytes, cudaMemcpyHostToDevice)); off += t.nbytes; }
    const uint8_t* p = b.as<uint8_t>();
    df_store.push_back(std::move(b));
    return p;
  };
  bool fp8 = getenv("HIVE_Q27_DFLASH_FP8") && strcmp(getenv("HIVE_Q27_DFLASH_FP8"), "1") == 0;
  // HIVE_Q27_DFLASH_FP4=1 (implies FP8): the drafter's decode GEMVs read an NVFP4 copy of each FP8 matrix (half the bytes — NInfer stores its
  //   DFlash2 drafter in 4/5-bit groups); the FP8 matrix stays for the context projection's prefill-sized GEMMs. Drafts only: every token is
  //   still verified exactly by the target.
  const bool fp4 = getenv("HIVE_Q27_DFLASH_FP4") && strcmp(getenv("HIVE_Q27_DFLASH_FP4"), "1") == 0;
  if (fp4) fp8 = true;
  DevBuf am4(4);
  auto add_w4 = [&](GMat& g) {
    if (g.fmt != WFmt::FP8ROW || g.w4 || g.K % 16) return;
    DevBuf b((size_t)g.N * g.K / 2 + (size_t)g.N * g.K / 16 + 256);
    uint8_t* w4 = b.as<uint8_t>();
    uint8_t* s4 = w4 + (((size_t)g.N * g.K / 2 + 255) & ~(size_t)255);
    g.g4 = k::requant_nvfp4(g, w4, s4, am4.as<float>(), st);
    g.w4 = w4; g.s4 = s4;
    df_store.push_back(std::move(b));
  };
  auto mat = [&](const std::vector<std::string>& names) {
    GMat g; g.fmt = WFmt::BF16; g.K = (int)T(names[0]).shape[1];
    for (auto& nm : names) { HIVE_CHECK((int)T(nm).shape[1] == g.K, "q27: DFlash stacked K"); g.N += (int)T(nm).shape[0]; }
    g.w = up(names);
    if (fp8) {   // HIVE_Q27_DFLASH_FP8=1: per-row FP8 copy, the bf16 upload freed
      DevBuf q((size_t)g.N * g.K + (size_t)g.N * 4 + 256);
      float* sc = reinterpret_cast<float*>(q.as<uint8_t>() + (((size_t)g.N * g.K + 255) & ~(size_t)255));
      df::to_fp8_rows(reinterpret_cast<const bf16*>(g.w), g.N, g.K, q.as<uint8_t>(), sc, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      df_store.pop_back();
      g.fmt = WFmt::FP8ROW; g.w = q.as<uint8_t>(); g.s = sc;
      df_store.push_back(std::move(q));
      if (fp4) add_w4(g);
    }
    return g;
  };
  auto vec = [&](const std::string& n) { return reinterpret_cast<const bf16*>(up({n})); };
  auto fvec = [&](const std::string& n) {   // bf16 → fp32
    const TensorInfo& t = T(n);
    std::vector<float> v((size_t)t.numel());
    for (size_t i = 0; i < v.size(); ++i) v[i] = b2f(((const uint16_t*)t.data)[i]);
    DevBuf b(v.size() * 4);
    CUDA_CHECK(cudaMemcpy(b.p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    const float* p = b.as<float>();
    df_store.push_back(std::move(b));
    return p;
  };
  {   // the context projection stays bf16 (prefill rows go through cuBLAS)
    const bool f = fp8;
    fp8 = false;
    df_fc = mat({"fc.weight"});
    fp8 = f;
  }
  df_hn = vec("hidden_norm.weight"); df_norm = vec("norm.weight");
  if (dsp) {   // Markov bigram head: bias(prev → c) = w2[c] · w1[prev] (the selector's edge term with no hidden factor)
    df_pred = vec("markov_head.markov_w1.weight"); df_succ = vec("markov_head.markov_w2.weight");
    const TensorInfo& cw = T("confidence_head.proj.weight");
    HIVE_CHECK(cw.numel() == c.hidden + df::kRank, "q27: DSpark confidence head shape");
    DevBuf hb((size_t)c.hidden * 2), mb((size_t)df::kRank * 2);
    CUDA_CHECK(cudaMemcpy(hb.p, cw.data, (size_t)c.hidden * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mb.p, (const uint8_t*)cw.data + (size_t)c.hidden * 2, (size_t)df::kRank * 2, cudaMemcpyHostToDevice));
    df_confh.fmt = WFmt::BF16; df_confh.N = 1; df_confh.K = c.hidden; df_confh.w = hb.as<uint8_t>();
    df_confw = mb.as<bf16>();
    df_store.push_back(std::move(hb)); df_store.push_back(std::move(mb));
    df_confb = b2f(*(const uint16_t*)T("confidence_head.proj.bias").data);
  } else {
    df_hp = mat({"candidate_selector.hidden_projection.weight"});
    df_pred = vec("candidate_selector.predecessor_codebook"); df_succ = vec("candidate_selector.successor_codebook");
  }
  std::vector<std::string> kv, kn;
  for (int l = 0; l < df::kLayers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    DfLayer& D = dfl[l];
    D.qkv = mat({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
    D.o = mat({p + "self_attn.o_proj.weight"});
    D.gu = mat({p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"});
    D.down = mat({p + "mlp.down_proj.weight"});
    if (!dsp) {
      D.kpa = mat({p + "attention_conv.kernel_projection.weight"});
      D.kpm = mat({p + "mlp_conv.kernel_projection.weight"});
      D.base_a = fvec(p + "attention_conv.base_kernel"); D.base_m = fvec(p + "mlp_conv.base_kernel");
    }
    D.ln1 = vec(p + "input_layernorm.weight"); D.ln2 = vec(p + "post_attention_layernorm.weight");
    D.qn = vec(p + "self_attn.q_norm.weight"); D.kn = vec(p + "self_attn.k_norm.weight");
    kv.push_back(p + "self_attn.k_proj.weight"); kv.push_back(p + "self_attn.v_proj.weight");
    kn.push_back(p + "self_attn.k_norm.weight");
  }
  df_kvall = mat(kv);   // context K / V of every layer in one pass: per layer [k 1024 | v 1024]
  df_knorm = reinterpret_cast<const bf16*>(up(kn));
  const double theta = j.contains("rope_parameters") ? j["rope_parameters"].value("rope_theta", 1e7) : j.value("rope_theta", 1e7);
  df_eps = j.value("rms_norm_eps", 1e-6);
  std::vector<float> inv(df::kHeadDim / 2);
  for (int i = 0; i < df::kHeadDim / 2; ++i) inv[i] = (float)std::pow(theta, -2.0 * i / df::kHeadDim);
  const nlohmann::json rp = j.contains("rope_parameters") ? j["rope_parameters"] : j.value("rope_scaling", nlohmann::json::object());
  if (rp.is_object() && rp.value("rope_type", "") == "yarn") {   // YaRN (the published rule): interpolate low frequencies by `factor`, keep high ones,
    //   linear ramp between the correction dims of beta_fast / beta_slow; attention factor 0.1 · ln(factor) + 1 on cos / sin
    const double factor = rp.value("factor", 1.0), orig = rp.value("original_max_position_embeddings", 8192), bf = rp.value("beta_fast", 32.0),
                 bs = rp.value("beta_slow", 1.0), dim = df::kHeadDim;
    auto corr = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log(theta)); };
    const double low = std::max(0.0, std::floor(corr(bf))), high = std::min(dim - 1, std::ceil(corr(bs)));
    for (int i = 0; i < df::kHeadDim / 2; ++i) {
      const double ramp = std::min(1.0, std::max(0.0, (i - low) / std::max(high - low, 1e-3)));
      const double extrap = 1.0 - ramp;   // 1 = keep the original frequency (high-frequency dims)
      const double f = std::pow(theta, -2.0 * i / df::kHeadDim);
      inv[i] = (float)(f / factor * (1.0 - extrap) + f * extrap);
    }
    df_mscale = (float)(0.1 * std::log(factor) + 1.0);
  }
  df_inv.alloc(inv.size() * 4);
  CUDA_CHECK(cudaMemcpy(df_inv.p, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
  const int R = std::max(chunk, kMaxRows), H = c.hidden, B = df::kBlock * kDfSeqs;
  df_cap.alloc((size_t)R * df::kLayers * H * 2); df_f.alloc((size_t)R * H * 2); df_f2.alloc((size_t)R * H * 2);
  df_kv.alloc((size_t)R * 2 * df::kLayers * df::kKvW * 2); df_rows.alloc((size_t)R * sizeof(df::CtxRow)); df_ids.alloc((size_t)B * 4);
  df_seqs.alloc(kDfSeqs * sizeof(df::BlkSeq));
  CUDA_CHECK(cudaMallocHost(&df_seq_pin, kDfSeqs * sizeof(df::BlkSeq)));
  CUDA_CHECK(cudaMallocHost(&df_ids_pin, (size_t)B * 4));
  dx.alloc((size_t)B * H * 2); dxn.alloc((size_t)B * H * 2); dy.alloc((size_t)B * H * 2); dz.alloc((size_t)B * H * 2);
  dproj.alloc((size_t)B * (df::kHeads + 2 * df::kKvHeads) * df::kHeadDim * 2); dq_.alloc((size_t)B * df::kHeads * df::kHeadDim * 2);
  dbk.alloc((size_t)B * df::kKvW * 2); dbv.alloc((size_t)B * df::kKvW * 2); dattn.alloc((size_t)B * df::kHeads * df::kHeadDim * 2);
  ddyn.alloc((size_t)B * 4 * (H / df::kGroup) * 4); dgu.alloc((size_t)B * 2 * c.inter * 2);
  dch.alloc((size_t)B * 4);
  df::attention_prepare();
  dcidx.alloc((size_t)B * df::kTopK * 4); dtk.alloc((size_t)B * 32 * df::kTopK * 8); dedge.alloc((size_t)kDfSeqs * 7 * 16 * 16 * 4); dcval.alloc((size_t)B * df::kTopK * 4); dhp.alloc((size_t)B * df::kRank * 4); dout.alloc((size_t)2 * B * 4);
  CUDA_CHECK(cudaMallocHost(&df_out_pin, (size_t)2 * B * 4));
  CUDA_CHECK(cudaMallocHost(&df_rows_pin, (size_t)R * sizeof(df::CtxRow)));
  fprintf(stderr, "[q27] %s drafter %s: %d layers, block %d, target layers %d/%d/%d/%d/%d, mask %d (%.0f MiB)\n", dsp ? "DSpark" : "DFlash2", df_dir.c_str(), df::kLayers, df_B,
          d.at("target_layer_ids")[0].get<int>(), d.at("target_layer_ids")[1].get<int>(), d.at("target_layer_ids")[2].get<int>(),
          d.at("target_layer_ids")[3].get<int>(), d.at("target_layer_ids")[4].get<int>(), mask_id, sf.size() / 1048576.0);
}

// the context ring: DFlash2 keeps kRing slots (2048-position window); DSpark keeps every position (slots grown ×1.5 past need_pos + one block,
//   contents of positions < pos copied; nothing ever wraps)
void Q27Engine::Impl::df_ensure_ring(Q27Seq& s, int64_t need_pos) {
  if (!dsp) {
    if (s.df_k.p) return;
    const size_t ring = (size_t)df::kLayers * df::kRing * df::kKvW * 2;
    alloc_serving(s.df_k, ring, "DFlash ring"); alloc_serving(s.df_v, ring, "DFlash ring");
    s.df_slots = df::kRing;
    return;
  }
  const int64_t need = need_pos + 64;
  if (s.df_k.p && s.df_slots >= need) return;
  const int64_t ns = std::max<int64_t>(need, std::max<int64_t>(2048, (int64_t)s.df_slots * 3 / 2)) / 256 * 256 + 256;
  DevBuf k2, v2;
  alloc_serving(k2, (size_t)df::kLayers * ns * df::kKvW * 2, "DSpark context");
  alloc_serving(v2, (size_t)df::kLayers * ns * df::kKvW * 2, "DSpark context");
  if (s.df_k.p && s.pos > 0) {
    const size_t used = (size_t)std::min<int64_t>(s.pos, s.df_slots) * df::kKvW * 2;
    for (int l = 0; l < df::kLayers; ++l) {
      CUDA_CHECK(cudaMemcpyAsync(k2.as<uint8_t>() + (size_t)l * ns * df::kKvW * 2, s.df_k.as<uint8_t>() + (size_t)l * s.df_slots * df::kKvW * 2, used, cudaMemcpyDeviceToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(v2.as<uint8_t>() + (size_t)l * ns * df::kKvW * 2, s.df_v.as<uint8_t>() + (size_t)l * s.df_slots * df::kKvW * 2, used, cudaMemcpyDeviceToDevice, st));
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
  }
  s.df_k = std::move(k2); s.df_v = std::move(v2); s.df_slots = (int)ns;
}

void Q27Engine::Impl::df_ingest(std::vector<Q27Seq*>& seqs, std::vector<int64_t>& pos, const bf16* capb) {
  const int M = (int)seqs.size(), H = c.hidden;
  for (int r = 0; r < M; ++r) df_ensure_ring(*seqs[r], pos[r] + 1 + df_B);   // grown before any pointer is taken
  // df_rows_pin is free: every forward ends with a stream synchronization
  for (int r = 0; r < M; ++r) df_rows_pin[r] = df::CtxRow{seqs[r]->df_k.as<bf16>(), seqs[r]->df_v.as<bf16>(), (int)pos[r], seqs[r]->df_slots};
  CUDA_CHECK(cudaMemcpyAsync(df_rows.p, df_rows_pin, (size_t)M * sizeof(df::CtxRow), cudaMemcpyHostToDevice, st));
  lin(df_fc, capb, df::kLayers * H, M, df_f.as<bf16>(), H);
  df::rmsnorm(df_f.as<bf16>(), H, df_hn, M, H, df_eps, df_f2.as<bf16>(), H, st);
  lin(df_kvall, df_f2.as<bf16>(), H, M, df_kv.as<bf16>(), df_kvall.N);
  df::ctx_kv(df_kv.as<bf16>(), df_knorm, df_rows.as<df::CtxRow>(), M, df_eps, df_inv.as<float>(), df_mscale, st);
  static const bool dbg = getenv("HIVE_Q27_DF_DEBUG") && strcmp(getenv("HIVE_Q27_DF_DEBUG"), "0") != 0;
  if (dbg) {
    DevBuf cnt(16);
    auto nf16 = [&](const void* p, size_t n) { CUDA_CHECK(cudaMemsetAsync(cnt.p, 0, 4, st)); df::count_nonfinite_bf16((const bf16*)p, n, cnt.as<int>(), st); int h = 0; CUDA_CHECK(cudaMemcpy(&h, cnt.p, 4, cudaMemcpyDeviceToHost)); return h; };
    fprintf(stderr, "[q27-df-debug] ingest M %d pos0 %lld verify %d · cap nonfinite %d · f %d · f2 %d · kv %d\n", M, (long long)pos[0], (int)verify,
            nf16(capb, (size_t)M * df::kLayers * H), nf16(df_f.p, (size_t)M * H), nf16(df_f2.p, (size_t)M * H), nf16(df_kv.p, (size_t)M * df_kvall.N));
  }
  for (int r = 0; r < M; ++r) seqs[r]->h_last_valid = true;
}

int Q27Engine::Impl::df_draft(Q27Seq& s, int32_t tok, int kk, int32_t* drafts, float* conf) {
  Q27Seq* sp = &s;
  int n = 0;
  df_draft_batch(&sp, &tok, 1, kk, drafts, conf, &n);
  return n;
}

// S sequences' blocks stacked (rows [8s, 8s + 8)); sequences without a context get n_out = 0
void Q27Engine::Impl::df_draft_batch(Q27Seq* const* seqs_in, const int32_t* toks, int S_in, int kk, int32_t* drafts, float* conf, int* n_out) {
  const int B = df_B, H = c.hidden, I = c.inter, P = df::kBlock - 1;   // 7 drafts per block for both drafters
  kk = std::min(kk, P);
  // quantized drafter matrices (HIVE_Q27_DFLASH_FP8) take more than 16 rows as 16-row decode GEMVs (the prefill path would dequantize them)
  auto dlin = [&](const GMat& W, const bf16* xin, int ldx, int M, bf16* yb, int ldy, float* yf = nullptr) {
    if (W.fmt == WFmt::BF16 || M <= kMaxRows) { lin(W, xin, ldx, M, yb, ldy, yf); return; }
    if (k::gemv_draft(W.w4 ? k::decode_view(W) : W, xin, ldx, M, yb, yf, ldy, st)) return;   // one pass for 33..64 drafter rows
    for (int r0 = 0; r0 < M; r0 += kMaxRows)
      lin(W, xin + (size_t)r0 * ldx, ldx, std::min(kMaxRows, M - r0), yb ? yb + (size_t)r0 * ldy : nullptr, ldy, yf ? yf + (size_t)r0 * ldy : nullptr);
  };
  std::vector<int> at;
  for (int i = 0; i < S_in; ++i) {
    n_out[i] = 0;
    if (toks[i] < 0 || toks[i] >= c.vocab) {   // an anchor outside the vocabulary would index the embedding / Markov tables out of range
      static int warned = 0;
      if (warned++ < 8) fprintf(stderr, "[q27] draft skipped: anchor token %d outside the vocabulary (pos %lld)\n", toks[i], (long long)seqs_in[i]->pos);
      continue;
    }
    if (seqs_in[i]->pos >= 1 && seqs_in[i]->df_k.p && seqs_in[i]->h_last_valid) at.push_back(i);
  }
  if (at.empty() || kk <= 0) return;
  for (size_t a0 = 0; a0 < at.size(); a0 += kDfSeqs) {
    const int S = (int)std::min<size_t>(kDfSeqs, at.size() - a0), R = S * B;
    for (int j = 0; j < S; ++j) {
      Q27Seq& q = *seqs_in[at[a0 + j]];
      df_seq_pin[j] = df::BlkSeq{q.df_k.as<bf16>(), q.df_v.as<bf16>(), (int)q.pos, toks[at[a0 + j]], q.df_slots};
      df_ids_pin[j * B] = toks[at[a0 + j]];
      for (int i = 1; i < B; ++i) df_ids_pin[j * B + i] = mask_id;
    }
    CUDA_CHECK(cudaMemcpyAsync(df_seqs.p, df_seq_pin, (size_t)S * sizeof(df::BlkSeq), cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(df_ids.p, df_ids_pin, (size_t)R * 4, cudaMemcpyHostToDevice, st));
    const df::BlkSeq* sq = df_seqs.as<df::BlkSeq>();
    bf16* X = dx.as<bf16>();
    bf16* XN = dxn.as<bf16>();
    bf16* Y = dy.as<bf16>();
    bf16* Z = dz.as<bf16>();
    float* DY = ddyn.as<float>();
    if (!emb_ptr) {
      CUDA_CHECK(cudaStreamSynchronize(st));
      std::vector<uint16_t> e((size_t)R * H);
      for (int i = 0; i < R; ++i) memcpy(e.data() + (size_t)i * H, cpu->embed_table() + (size_t)df_ids_pin[i] * H, (size_t)H * 2);
      CUDA_CHECK(cudaMemcpyAsync(X, e.data(), e.size() * 2, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaStreamSynchronize(st));
    }
    {
      const GMat& dh0 = draft_ids.empty() ? lm_head : draft_head;
      if (df_logits.n < (size_t)kDfSeqs * P * dh0.N * 4) df_logits.alloc((size_t)kDfSeqs * P * dh0.N * 4);
    }
    auto body = [&]() {
    if (emb_ptr) k::embed_rows(emb_ptr, df_ids.as<int32_t>(), R, H, X, st);
    const int ND = 4 * (H / df::kGroup), PW = (df::kHeads + 2 * df::kKvHeads) * df::kHeadDim, QW = df::kHeads * df::kHeadDim;
    for (int l = 0; l < df::kLayers; ++l) {
      const DfLayer& D = dfl[l];
      df::rmsnorm(X, H, D.ln1, R, H, df_eps, XN, H, st);
      if (!dsp) { dlin(D.kpa, XN, H, R, nullptr, ND, DY); df::conv(XN, H, DY, 0, D.base_a, R, B, H, Y, false, st); }
      dlin(D.qkv, dsp ? XN : Y, H, R, dproj.as<bf16>(), PW);
      df::blk_prep(dproj.as<bf16>(), D.qn, D.kn, sq, S, B, df_eps, df_inv.as<float>(), df_mscale, dq_.as<bf16>(), dbk.as<bf16>(), dbv.as<bf16>(), st);
      df::attention(dq_.as<bf16>(), dbk.as<bf16>(), dbv.as<bf16>(), sq, l, S, B, df_window, 1.f / std::sqrt((float)df::kHeadDim), dattn.as<bf16>(), st);
      dlin(D.o, dattn.as<bf16>(), QW, R, Z, H);
      if (dsp) k::add_round(X, Z, (size_t)R * H, st);
      else df::conv(Z, H, DY, 1, D.base_a + 2 * H, R, B, H, X, true, st);
      df::rmsnorm(X, H, D.ln2, R, H, df_eps, XN, H, st);
      if (!dsp) { dlin(D.kpm, XN, H, R, nullptr, ND, DY); df::conv(XN, H, DY, 0, D.base_m, R, B, H, Y, false, st); }
      dlin(D.gu, dsp ? XN : Y, H, R, dgu.as<bf16>(), 2 * I);
      k::silu_mul(dgu.as<bf16>(), 2 * I, dgu.as<bf16>() + I, 2 * I, R, I, st);
      dlin(D.down, dgu.as<bf16>(), 2 * I, R, Z, H);
      if (dsp) k::add_round(X, Z, (size_t)R * H, st);
      else df::conv(Z, H, DY, 1, D.base_m + 2 * H, R, B, H, X, true, st);
    }
    // the predicting rows of every block (DFlash2: rows 1 … 7, the anchor row has no output · DSpark: rows 0 … 6) → final norm, gathered
    for (int j = 0; j < S; ++j) df::rmsnorm(X + ((size_t)j * B + df_out0) * H, H, df_norm, P, H, df_eps, XN + (size_t)j * P * H, H, st);
    const GMat& dh = draft_ids.empty() ? lm_head : draft_head;
    const int PR = P * S;
    for (int r0 = 0; r0 < PR; r0 += kMaxRows) {   // lm_head as decode GEMVs (the prefill path would dequantize the whole head)
      const int mm = std::min(kMaxRows, PR - r0);
      lin(dh, XN + (size_t)r0 * H, H, mm, nullptr, dh.N, df_logits.as<float>() + (size_t)r0 * dh.N);
    }
    if (dh.N <= 32 * 4096) df::top16(df_logits.as<float>(), PR, dh.N, dtk.as<float>(), dcidx.as<int32_t>(), dcval.as<float>(), st);
    else {
      hive::k::topk_select_rows(df_logits.as<float>(), PR, dh.N, df::kTopK, dh.N, dcidx.as<int32_t>(), df::kTopK, st);
      hive::k::gather_vals(df_logits.as<float>(), dcidx.as<int32_t>(), PR, df::kTopK, dh.N, dcval.as<float>(), st);
    }
    if (!dsp) dlin(df_hp, XN, H, PR, nullptr, df::kRank, dhp.as<float>());
    else lin(df_confh, XN, H, PR, nullptr, 1, dch.as<float>());   // confidence head, hidden part
    int32_t* oid = dout.as<int32_t>();
    float* ocf = reinterpret_cast<float*>(oid + kDfSeqs * P);
    df::select(dcidx.as<int32_t>(), dcval.as<float>(), dsp ? nullptr : dhp.as<float>(), df_pred, df_succ, draft_ids.empty() ? nullptr : draft_ids_d.as<int32_t>(),
               sq, S, dedge.as<float>(), oid, ocf, st, dsp ? dch.as<float>() : nullptr, df_confw, df_confb);
    CUDA_CHECK(cudaMemcpyAsync(df_out_pin, oid, (size_t)2 * kDfSeqs * P * 4, cudaMemcpyDeviceToHost, st));
    };
    // the draft pass as one CUDA graph per sequence count (inputs come from df_seqs / df_ids; only decode GEMVs inside)
    const bool gemv_only = R <= kMaxRows || dfl[0].qkv.fmt != WFmt::BF16;
    if (use_graphs && emb_ptr && gemv_only) {
      auto it = df_graphs.find(S);
      if (it == df_graphs.end()) {
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        body();
        CUDA_CHECK(cudaStreamEndCapture(st, &g));
        cudaGraphExec_t ge;
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        ++captures;
        CUDA_CHECK(cudaGraphDestroy(g));
        it = df_graphs.emplace(S, ge).first;
      }
      CUDA_CHECK(cudaGraphLaunch(it->second, st));
    } else {
      body();
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    static const bool dbg = getenv("HIVE_Q27_DF_DEBUG") && strcmp(getenv("HIVE_Q27_DF_DEBUG"), "0") != 0;
    if (dbg) {   // non-finite values in the block's hidden rows, the draft logits and the context ring of each sequence
      DevBuf cnt(16);
      auto nf16 = [&](const void* p, size_t n) { CUDA_CHECK(cudaMemsetAsync(cnt.p, 0, 4, st)); df::count_nonfinite_bf16((const bf16*)p, n, cnt.as<int>(), st); int h = 0; CUDA_CHECK(cudaMemcpy(&h, cnt.p, 4, cudaMemcpyDeviceToHost)); return h; };
      auto nf32 = [&](const void* p, size_t n) { CUDA_CHECK(cudaMemsetAsync(cnt.p, 0, 4, st)); df::count_nonfinite_f32((const float*)p, n, cnt.as<int>(), st); int h = 0; CUDA_CHECK(cudaMemcpy(&h, cnt.p, 4, cudaMemcpyDeviceToHost)); return h; };
      const GMat& dh = draft_ids.empty() ? lm_head : draft_head;
      fprintf(stderr, "[q27-df-debug] S %d · X nonfinite %d · logits nonfinite %d", S, nf16(dx.p, (size_t)R * H), nf32(df_logits.p, (size_t)S * P * dh.N));
      for (int j = 0; j < S; ++j) {
        Q27Seq& q = *seqs_in[at[a0 + j]];
        const size_t used = (size_t)std::min<int64_t>(q.pos, q.df_slots) * df::kKvW;
        fprintf(stderr, " · seq pos %lld slots %d ring k nonfinite %d v %d", (long long)q.pos, q.df_slots, nf16(q.df_k.p, used), nf16(q.df_v.p, used));
      }
      fprintf(stderr, "\n");
    }
    for (int j = 0; j < S; ++j) {
      const int i = at[a0 + j];
      for (int d = 0; d < kk; ++d) {
        drafts[(size_t)i * P + d] = df_out_pin[j * P + d];
        if (conf) memcpy(&conf[(size_t)i * P + d], df_out_pin + kDfSeqs * P + j * P + d, 4);
      }
      n_out[i] = kk;
    }
  }
}

// ---- public API ------------------------------------------------------------------------------------------------------------------------
Q27Engine::Q27Engine(const std::string& ckpt_dir, bool load_mtp) : p_(std::make_unique<Impl>()) {
  p_->dir = ckpt_dir;
  p_->layer_hook = &layer_hook; p_->layer_inject = &layer_inject; p_->layer_boundary = &layer_boundary; p_->layer_boundary_due = &layer_boundary_due; p_->kv_reclaim = &kv_reclaim;
  p_->load(load_mtp);
}
Q27Engine::~Q27Engine() { if (p_ && p_->st) { cudaStreamSynchronize(p_->st); } }
const Q27Config& Q27Engine::cfg() const { return p_->c; }
const Placement& Q27Engine::placement() const { return p_->pl; }
GpuStats& Q27Engine::stats() { return p_->st_; }
const float* Q27Engine::logits_dev() const { return p_->logits_d.as<float>(); }
float* Q27Engine::logits_host() { return p_->logits_pin; }
void Q27Engine::set_host_logits(bool on) { p_->host_logits = on; }
long Q27Engine::graph_captures() const { return p_->captures; }
bool Q27Engine::register_host_logits(void* p, size_t bytes) {
  Impl& I = *p_;
  if (I.host_dst.count(p)) return true;
  if (I.host_dst.size() >= 7) return false;
  if (cudaHostRegister(p, bytes, cudaHostRegisterPortable) != cudaSuccess) { cudaGetLastError(); return false; }
  const int id = (int)I.host_dst.size();
  I.host_dst[p] = id;
  return true;
}
cudaStream_t Q27Engine::stream() const { return p_->st; }
cpu::CpuStats& Q27Engine::cpu_stats() { return p_->cpu->stats(); }
bool Q27Engine::mtp_on() const { return p_->pl.mtp_gpu && p_->mtp_k > 0; }
int Q27Engine::mtp_k() const { return p_->mtp_k; }
void Q27Engine::set_embed_override(std::vector<EmbedOverride> v) { p_->emb_over = std::move(v); }
bool Q27Engine::has_vision() const { return p_->cpu->has_vision(); }
void Q27Engine::encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out) { p_->cpu->encode_image(patches, gh, gw, out); }
int Q27Engine::mtp_draft(Q27Seq& s, int32_t tok, int k, int32_t* drafts, float* conf) { return p_->mtp_draft(s, tok, k, drafts, conf); }
void Q27Engine::mtp_draft_batch(Q27Seq* const* seqs, const int32_t* toks, int S, int k, int32_t* drafts, float* conf, int* n_out) {
  Impl& I = *p_;
  if (I.dflash && I.pl.mtp_gpu) { I.df_draft_batch(seqs, toks, S, k, drafts, conf, n_out); return; }
  for (int s = 0; s < S; ++s) n_out[s] = I.mtp_draft(*seqs[s], toks[s], k, drafts + (size_t)s * 7, conf ? conf + (size_t)s * 7 : nullptr);
}

// startup warm-up: captures the CUDA graphs of decode steps (1 … max_batch sequences) and single-sequence verifies (2 … max_k + 1 rows)
//   and runs one draft, so the first real steps (and hived's step-cost table) do not include graph captures and lazy allocations
void Q27Engine::warmup(int max_batch, int max_k) {
  Impl& I = *p_;
  if (I.pl.gpu_layers != I.c.n_layers || !I.pl.head_gpu) return;
  const double t0 = hive::mono_ms();
  max_batch = std::max(1, std::min(max_batch, 16));
  // under a VRAM cap the warm-up keeps to the sequences the reserve holds (larger batches capture their graphs on first use)
  if (env_mb("HIVE_Q27_VRAM_MB", env_mb("HIVE_VRAM_LIMIT_MB", 0)) > 0)
    max_batch = std::min(max_batch, getenv("HIVE_Q27_SESSIONS") ? std::max(1, atoi(getenv("HIVE_Q27_SESSIONS"))) : 4);
  std::vector<std::unique_ptr<Q27Seq>> sq(max_batch);
  std::vector<Q27Seq*> sp(max_batch);
  std::vector<int32_t> ids(32, 1000);
  float* lg = logits_host();
  for (int i = 0; i < max_batch; ++i) {
    sq[i] = std::make_unique<Q27Seq>();
    init_seq(*sq[i], 256);
    sp[i] = sq[i].get();
    std::vector<float> l1(I.c.vocab);
    prefill(*sp[i], ids.data(), 8, l1.data());
  }
  for (int M = 1; M <= max_batch; ++M) decode(sp.data(), ids.data(), M, lg);
  if (mtp_on()) {
    for (int hl = 1; hl >= 0; --hl) {   // verify graphs with host logits and device-only (greedy, set_host_logits(false))
      I.host_logits = hl;
      for (int k = 1; k <= std::min(max_k, kMaxRows - 1); ++k) {
        const int Mv = k + 1, keep = 1;
        verify_batch(sp.data(), &Mv, 1, ids.data(), lg);
        rollback_batch(&keep, 1);
      }
    }
    I.host_logits = true;
    int32_t d[8]; float cf[8];
    mtp_draft(*sp[0], 1000, mtp_k(), d, cf);
  }
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  fprintf(stderr, "[q27] warm-up: %zu CUDA graphs (decode 1..%d · verify 2..%d rows) in %.1f s\n", I.graphs.size(), max_batch, std::min(max_k, kMaxRows - 1) + 1,
          (hive::mono_ms() - t0) / 1000);
}

// trim (pool eviction / reclaim): KV buffers of more than 8192 positions are freed (16 pooled 32K sessions held ~51 GB, 2026-10-10); smaller ones
//   stay with the pooled sequence for its next session — a free + malloc pair per layer synchronises the device and stalled every arrival's
//   admission (bench c1: TTFT +30 ms from the 17th session on, when the 16-session pool starts evicting)
void Q27Engine::release_kv(Q27Seq& s) {
  const int64_t kKeep = p_->kv_pressure ? 0 : 8192;
  bool any = false;
  for (auto& q : s.kv) any |= q.tokens > kKeep || q.k.host();
  any |= s.mtp.tokens > kKeep || s.mtp.k.host();
  if (!any) return;
  CUDA_CHECK(cudaStreamSynchronize(p_->st));
  for (auto& q : s.kv) if (q.tokens > kKeep || q.k.host()) { q.k.reset(); q.v.reset(); q.tokens = 0; }
  if (s.mtp.tokens > kKeep || s.mtp.k.host()) { s.mtp.k.reset(); s.mtp.v.reset(); s.mtp.tokens = 0; }
}

void Q27Engine::init_seq(Q27Seq& s, int64_t cap) {
  Impl& I = *p_;
  s.cap = cap;
  s.conv.clear(); s.state.clear();
  s.conv.resize(I.c.n_lin()); s.state.resize(I.c.n_lin());
  for (int l = 0; l < I.pl.gpu_layers; ++l)
    if (!I.c.is_attn[l]) {   // serving allocations: on failure an idle session is evicted and the allocation retried (no abort)
      I.alloc_serving(s.conv[I.lin_index[l]], (size_t)3 * I.c.lin_conv_ch() * 4, "recurrent state");
      I.alloc_serving(s.state[I.lin_index[l]], (size_t)I.c.lin_v_heads * 128 * 128 * 4, "recurrent state");
    }
  s.kv.clear(); s.kv.resize(I.c.n_attn());
  I.cpu->init_seq(s.cs, cap);
  reset_seq(s);
}

void Q27Engine::reset_seq(Q27Seq& s) {
  Impl& I = *p_;
  for (auto& b : s.conv) if (b.p) CUDA_CHECK(cudaMemsetAsync(b.p, 0, b.n, I.st));
  for (auto& b : s.state) if (b.p) CUDA_CHECK(cudaMemsetAsync(b.p, 0, b.n, I.st));
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  I.cpu->reset_seq(s.cs);
  s.pos = 0; s.tokens.clear(); s.rope_delta = 0; s.rope.clear(); s.broken = false;
  s.h_last_valid = false; s.mtp_written = 0;
}

void Q27Engine::prefill(Q27Seq& s, const int32_t* ids, int T, float* logits) {
  Impl& I = *p_;
  if (T <= 0) return;
  // (hived prefills a prompt in max-chunk pieces: the diagnostics count from the piece at position 0)
  if (I.kvlog && s.pos == 0) { I.kl_t0 = hive::mono_ms(); I.kl0 = I.kl; }
  const double tp0 = I.kl_t0;
  const Impl::KvLog& kl0 = I.kl0;
  try {
    for (int off = 0; off < T; off += I.chunk) {
      const int n = std::min(I.chunk, T - off);
      I.emb_base = off;
      std::vector<Q27Seq*> seqs(n, &s);
      std::vector<int64_t> pos(n);
      for (int i = 0; i < n; ++i) pos[i] = s.pos + i;
      const double tc = hive::mono_ms();
      I.forward(seqs, pos, ids + off, true, off + n == T ? logits : nullptr, false);
      if (I.kvlog && s.pos >= 16384 && (s.pos / I.chunk) % 4 == 0) {
        int nh = 0;
        for (auto& q : s.kv) nh += q.k.host();
        fprintf(stderr, "[q27-kv] prefill chunk to %d: %.0f ms · host-tier layers %d of %zu · dev KV %zu MiB\n", (int)s.pos, hive::mono_ms() - tc, nh, s.kv.size(),
                g_kv_dev_bytes.load() >> 20);
      }
    }
  } catch (...) { s.broken = true; I.emb_base = 0; I.emb_over.clear(); throw; }
  if (I.kvlog && s.pos >= 4096 && T != I.chunk)
    fprintf(stderr, "[q27-kv] prefill %d tokens %.0f ms · grows %ld (%.0f ms in allocation) · host allocs %ld (%.0f ms, %zu MiB) · copies through host %zu MiB · "
            "stage copies %ld (%zu MiB) · stage misses %ld\n", (int)s.pos, hive::mono_ms() - tp0, I.kl.grows - kl0.grows, I.kl.grow_ms - kl0.grow_ms,
            I.kl.host_allocs - kl0.host_allocs, I.kl.host_ms - kl0.host_ms, (I.kl.host_b - kl0.host_b) >> 20, (I.kl.grow_copy_b - kl0.grow_copy_b) >> 20,
            I.kl.stage_copies - kl0.stage_copies, (I.kl.stage_b - kl0.stage_b) >> 20, I.kl.stage_miss - kl0.stage_miss);
  I.emb_base = 0;
  I.emb_over.clear();
}

void Q27Engine::decode(Q27Seq* const* seqs, const int32_t* ids, int M, float* logits) {
  std::vector<Q27Seq*> sv(seqs, seqs + M);
  std::vector<int64_t> pos(M);
  for (int m = 0; m < M; ++m) pos[m] = seqs[m]->pos;
  try { p_->forward(sv, pos, ids, false, logits, true); } catch (...) { for (int m = 0; m < M; ++m) seqs[m]->broken = true; throw; }
}

void Q27Engine::verify_batch(Q27Seq* const* seqs, const int* Ms, int S, const int32_t* ids, float* logits) {
  Impl& I = *p_;
  I.vparts.clear();
  std::vector<Q27Seq*> sv;
  std::vector<int64_t> pos, pos0(S);
  std::vector<cpu::CpuSeq*> cs(S);
  int R = 0;
  for (int p = 0; p < S; ++p) {
    if (Ms[p] < 1 || R + Ms[p] > kMaxRows || S > kMaxParts) throw std::runtime_error("q27 verify rows");
    I.vparts.push_back({seqs[p], seqs[p]->pos, Ms[p], R});
    for (int r = 0; r < Ms[p]; ++r) { sv.push_back(seqs[p]); pos.push_back(seqs[p]->pos + r); }
    pos0[p] = seqs[p]->pos; cs[p] = &seqs[p]->cs;
    R += Ms[p];
  }
  I.verify_ids.assign(ids, ids + R);
  I.cpu->begin_verify(cs.data(), pos0.data(), Ms, S);
  I.verify = true;
  try { I.forward(sv, pos, ids, false, logits, true); } catch (...) { I.verify = false; I.cpu->end_verify(); for (int p = 0; p < S; ++p) seqs[p]->broken = true; throw; }
  I.verify = false;
  I.cpu->end_verify();
}

void Q27Engine::rollback_batch(const int* n_keep, int S) {
  Impl& I = *p_;
  if (S != (int)I.vparts.size()) throw std::runtime_error("q27 rollback without a matching verify");
  const int H = I.c.hidden;
  std::vector<cpu::CpuSeq*> cs(S);
  std::vector<int> Ms(S);
  for (int p = 0; p < S; ++p) { cs[p] = &I.vparts[p].s->cs; Ms[p] = I.vparts[p].M; }
  I.cpu->rollback_layers(cs.data(), Ms.data(), n_keep, S);
  std::vector<k::GdnReplay> rp;
  std::vector<k::CopyItem> cp;   // conv-state restores, one launch
  for (int p = 0; p < S; ++p) {
    const Impl::VPart& vp = I.vparts[p];
    Q27Seq& s = *vp.s;
    const int M = vp.M, nk = n_keep[p];
    if (nk < 1 || nk > M || s.pos != vp.pos0 + M) throw std::runtime_error("q27 rollback without a matching verify");
    // conv: the snapshot after the last kept row (nk < M) · recurrent state: replay the kept rows — from the verify's snapshot (HIVE_Q27_VSNAP,
    //   only when rows were rejected) or in place from the state the verify left untouched (always, also when every row is kept)
    if (nk < M || !I.vsnap) {
      const int slot = vp.r0 + nk - 1;
      for (size_t j = 0; j < I.gpu_lin.size(); ++j) {
        const int li = I.gpu_lin[j];
        if (nk < M) cp.push_back(k::CopyItem{I.snap_conv[(size_t)slot * I.gpu_lin.size() + j].as<float>(), s.conv[li].as<float>(), (int)(s.conv[li].n / 4)});
        float* init = I.vsnap ? I.snap_state[(size_t)p * I.gpu_lin.size() + j].as<float>() : s.state[li].as<float>();
        rp.push_back(k::GdnReplay{I.vq_qk.as<float>() + (size_t)li * kMaxRows * 2 * I.c.lin_k_width(), I.vq_gb.as<float>() + (size_t)li * kMaxRows * 2 * I.c.lin_v_heads,
                                  I.vq_y.as<bf16>() + (size_t)li * kMaxRows * I.c.lin_conv_ch(), vp.r0, nk, init, s.state[li].as<float>()});
      }
    }
    s.pos = vp.pos0 + nk;
    s.tokens.resize(s.tokens.size() - (size_t)(M - nk));
    if (I.dflash) s.h_last_valid = I.pl.mtp_gpu && I.mtp_k > 0;   // the drafter's context ring holds every position < pos (df_ingest)
    else if (I.pl.mtp_gpu && I.mtp_k > 0) {
      if (nk >= 2) {
        std::vector<Q27Seq*> es(nk - 1, &s);
        std::vector<int64_t> ep(nk - 1);
        for (int i = 0; i < nk - 1; ++i) ep[i] = vp.pos0 + i;
        I.mtp_entries(es, ep, I.fh.as<bf16>() + (size_t)vp.r0 * H, I.verify_ids.data() + vp.r0 + 1, {});
      }
      if (!s.h_last.p) s.h_last.alloc((size_t)H * 2);
      CUDA_CHECK(cudaMemcpyAsync(s.h_last.p, I.fh.as<bf16>() + (size_t)(vp.r0 + nk - 1) * H, (size_t)H * 2, cudaMemcpyDeviceToDevice, I.st));
      s.h_last_valid = true;
    }
  }
  if (!rp.empty()) {   // one upload: replay items, then the conv copy items
    const size_t rb = rp.size() * sizeof(k::GdnReplay), cb = cp.size() * sizeof(k::CopyItem);
    std::vector<uint8_t> up(rb + cb);
    memcpy(up.data(), rp.data(), rb);
    memcpy(up.data() + rb, cp.data(), cb);
    CUDA_CHECK(cudaMemcpyAsync(I.rp_items.p, up.data(), up.size(), cudaMemcpyHostToDevice, I.st));
    k::gdn_replay(I.rp_items.as<k::GdnReplay>(), (int)rp.size(), I.c.lin_k_heads, I.c.lin_v_heads, I.c.lin_conv_ch(), I.st);
    if (!cp.empty()) k::multi_copy(reinterpret_cast<const k::CopyItem*>(I.rp_items.as<uint8_t>() + rb), (int)cp.size(), I.st);
  }
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  I.vparts.clear();
}

// ---- session images: GPU state copied to host bytes, then the CPU layers' CpuSeq ------------------------------------------------------------
// layout: header · tokens · rope · per linear layer (conv, state) · per attention layer (K, V rows < pos) · MTP K, V · h_last ·
//   DFlash ring per layer (K, V: slots of positions < min(pos, ring)) · CpuSeq pieces
void Q27Seq::save_plan(std::vector<Piece>& out, std::vector<std::vector<uint8_t>>& hold) const {
  out.clear(); hold.clear();
  hold.reserve(4);
  const int64_t hdr[8] = {pos, cap, rope_delta, h_last_valid ? 1 : 0, mtp_written, broken ? 1 : 0, (int64_t)conv.size(), (int64_t)kv.size()};
  hold.emplace_back((const uint8_t*)hdr, (const uint8_t*)hdr + sizeof hdr);
  out.push_back({hold.back().data(), nullptr, sizeof hdr, false});
  out.push_back({tokens.data(), nullptr, tokens.size() * 4, true});
  out.push_back({rope.data(), nullptr, std::min(rope.size(), (size_t)pos * 3) * 4, true});
  // (b: DevBuf or KvBuf — both expose p / n)
  auto dv = [&](const auto& b, size_t n, bool app, size_t off = 0) { out.push_back({nullptr, b.p ? (const uint8_t*)b.p + off : nullptr, b.p ? std::min(n, b.n - off) : 0, app}); };
  for (size_t i = 0; i < conv.size(); ++i) { dv(conv[i], conv[i].n, false); dv(state[i], state[i].n, false); }
  const size_t row = 1024 * 2;
  for (const Kv& k : kv) { dv(k.k, (size_t)pos * row, true); dv(k.v, (size_t)pos * row, true); }
  dv(mtp.k, (size_t)mtp_written * row, true); dv(mtp.v, (size_t)mtp_written * row, true);
  dv(h_last, h_last.n, false);
  hold.emplace_back(8, 0);   // the ring's slot count
  { const int64_t sl = df_k.p ? df_slots : 0; memcpy(hold.back().data(), &sl, 8); }
  out.push_back({hold.back().data(), nullptr, 8, false});
  const int slots = df_k.p ? df_slots : 1;
  const size_t lay = (size_t)slots * df::kKvW * 2, used = (size_t)std::min<int64_t>(pos, slots) * df::kKvW * 2;
  for (int l = 0; l < df::kLayers; ++l) { dv(df_k, df_k.p ? used : 0, pos <= slots, l * lay); dv(df_v, df_v.p ? used : 0, pos <= slots, l * lay); }
  for (auto& b : cs.save_bytes()) { hold.push_back(std::move(b)); out.push_back({hold.back().data(), nullptr, hold.back().size(), false}); }
}

void Q27Seq::load_views(const std::vector<std::pair<const uint8_t*, size_t>>& b, cudaStream_t st,
                        const std::function<void(void*, const uint8_t*, size_t)>& h2d) {
  auto cp = [&](void* d, const uint8_t* h, size_t n) { if (h2d) h2d(d, h, n); else CUDA_CHECK(kv_memcpy_async(d, h, n, st)); };
  if (b.size() < 3 || b[0].second != 64) throw std::runtime_error("q27: bad state image");
  int64_t hdr[8];
  memcpy(hdr, b[0].first, 64);
  const size_t nl = (size_t)hdr[6], na = (size_t)hdr[7];
  if (nl != conv.size() || na != kv.size()) throw std::runtime_error("q27: state image does not match the model");
  pos = hdr[0]; cap = hdr[1]; rope_delta = hdr[2]; h_last_valid = hdr[3] != 0; mtp_written = hdr[4]; broken = hdr[5] != 0;
  size_t k = 1;
  tokens.assign((const int32_t*)b[k].first, (const int32_t*)(b[k].first + b[k].second)); ++k;
  rope.assign((const int32_t*)b[k].first, (const int32_t*)(b[k].first + b[k].second)); ++k;
  auto ul = [&](DevBuf& d, const std::pair<const uint8_t*, size_t>& v, size_t off = 0, size_t total = 0) {
    if (!v.second) return;
    const size_t need = std::max(total, off + v.second);
    if (d.n < need) d.alloc(need);
    cp((uint8_t*)d.p + off, v.first, v.second);
  };
  for (size_t i = 0; i < nl; ++i) { ul(conv[i], b[k++]); ul(state[i], b[k++]); }
  const size_t row = 1024 * 2;
  auto ulkv = [&](Kv& q) {
    const auto& kb = b[k++];
    const auto& vb = b[k++];
    if (!kb.second) { q.k.reset(); q.v.reset(); q.tokens = 0; return; }
    const int64_t t = (int64_t)(kb.second / row);
    if (q.tokens < t || !q.k.p) {   // reused when large enough; allocated through the engine (KV tier: VRAM or host)
      const int64_t nt = t + 256;
      if (g_kv_alloc) { q.k.reset(); q.v.reset(); g_kv_alloc(q.k, (size_t)nt * row); g_kv_alloc(q.v, (size_t)nt * row); }
      else { q.k.set_dev(DevBuf((size_t)nt * row)); q.v.set_dev(DevBuf((size_t)nt * row)); }
      q.tokens = nt;
    }
    cp(q.k.p, kb.first, kb.second);
    cp(q.v.p, vb.first, vb.second);
  };
  for (Kv& q : kv) ulkv(q);
  ulkv(mtp);
  ul(h_last, b[k++]);
  int64_t sl = 0;
  if (b[k].second == 8) memcpy(&sl, b[k].first, 8);
  ++k;
  if (sl > 0 && sl != df_slots) { df_k = DevBuf(); df_v = DevBuf(); }   // another ring geometry: reallocated below
  if (sl > 0) df_slots = (int)sl;
  const size_t lay = (size_t)std::max<int64_t>(sl, 1) * df::kKvW * 2;
  for (int l = 0; l < df::kLayers; ++l) { ul(df_k, b[k++], l * lay, df::kLayers * lay); ul(df_v, b[k++], l * lay, df::kLayers * lay); }
  std::vector<std::vector<uint8_t>> rest;
  for (; k < b.size(); ++k) rest.emplace_back(b[k].first, b[k].first + b[k].second);
  CUDA_CHECK(cudaStreamSynchronize(st));
  cs.load_bytes(rest);
}

std::vector<std::vector<uint8_t>> Q27Seq::save_bytes() const {
  std::vector<Piece> pc;
  std::vector<std::vector<uint8_t>> hold, out;
  save_plan(pc, hold);
  for (const Piece& p : pc) {
    std::vector<uint8_t> v(p.n);
    if (p.n) { if (p.dev) { CUDA_CHECK(kv_memcpy_async(v.data(), p.dev, p.n, 0)); CUDA_CHECK(cudaStreamSynchronize(0)); } else memcpy(v.data(), p.host, p.n); }
    out.push_back(std::move(v));
  }
  return out;
}

void Q27Seq::load_bytes(const std::vector<std::vector<uint8_t>>& b) {
  std::vector<std::pair<const uint8_t*, size_t>> v;
  for (const auto& x : b) v.push_back({x.data(), x.size()});
  load_views(v, nullptr);
}

}  // namespace hive::q27
