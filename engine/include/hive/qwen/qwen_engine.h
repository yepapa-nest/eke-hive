// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next forward pass: prefill of one sequence in chunks (or layer-major over several chunks), batched decode (one token per
//   sequence), MTP drafting / verify / rollback. Built after GlmEngine (hive/glm/glm_engine.h) — the same contract towards fam::Runtime and
//   hived, the same routed-expert store (GlmExperts: VRAM slot cache + pinned RAM + CPU misses), plus:
//   · Device routing (HIVE_QWEN_DEVICE_ROUTE, default on): while every routed expert is resident in VRAM (the "VRAM 100 %" configuration —
//     all 24,576 records fit the slot cache), decode and verify route on the device and run the experts from a device table of record
//     pointers: no host round trip per MoE layer (GlmExperts::decode_layer synchronizes the stream once per layer to read the routing).
//     As soon as one expert is not resident (the cache shrank for a long context, or a VRAM budget is set) the host path is used.
//   · VRAM budget (HIVE_QWEN_VRAM_MB, startup only; unset = the whole card): the most device memory this process may use. Everything that
//     sizes itself from free VRAM — the expert cache at startup, KV growth, the vision encoder — counts against min(free VRAM, budget − used).
//   · Rope coordinates per token (M-RoPE: images take (t, h, w) coordinates and shift the text after them — QwenSeq::rope_delta).
#pragma once
#include <functional>
#include <map>
#include <string>
#include <memory>
#include <vector>

#include "hive/cublas_ops.h"
#include "hive/grow_buf.h"
#include "hive/glm/glm_experts.h"
#include "hive/qwen/gdn.h"
#include "hive/qwen/ple.h"
#include "hive/qwen/qsa.h"
#include "hive/qwen/qwen_cpu.h"
#include "hive/qwen/qwen_model.h"

namespace hive::qwen {

// decode / verify rows per forward (batched sequences × their verify rows): the GEMV kernels take up to 64 activation rows in one weight pass
constexpr int kDraftRows = 16;   // sequences per batched MTP draft (mtp_draft_batch)
constexpr int kDecRows = 64;


using glm::GlmExperts;
using glm::GlmCacheStats;

// One full-attention cache (main layer or MTP): address range for cap positions, memory mapped as the sequence grows (kv_reserve).
struct AttnCache {
  GrowBuf k, v, ik, gk;
  QsaCache view(const int32_t* rope) const { return QsaCache{k.as<bf16>(), v.as<bf16>(), ik.as<bf16>(), gk.as<bf16>(), rope}; }
};

struct QwenSeq {
  int64_t pos = 0, cap = 0;
  std::vector<int32_t> tokens;
  int64_t rope_delta = 0;             // text token at position p has rope coordinates p + rope_delta (≤ 0 after images)
  std::vector<DevBuf> conv, state;    // per linear layer: fp32 [3·10240], fp32 [48·128·128]
  std::vector<AttnCache> attn;        // per full-attention layer
  GrowBuf rope;                       // int32 [cap, 3]
  DevBuf ple_state;                   // fp32 [9, hc·hidden] (PLE conv inputs t−9 … t−1)
  bool broken = false;
  size_t device_bytes = 0;
  int64_t kv_tokens = 0;              // positions backed by memory in every growing buffer
  // MTP: its own attention cache; entry j = f(main stream h_j, token x_{j+1}) at position j. Entries < mtp_written are exact.
  AttnCache mtp;
  DevBuf h_last;                      // bf16 [hc·hidden]: final hidden stream (before the mixer) of the last processed position
  bool h_last_valid = false;
  int64_t mtp_written = 0;
  std::unique_ptr<cpu::CpuSeq> tail;  // hybrid placement: the state of the CPU-tail layers (host memory)
};

struct QwenForwardStats {
  double ms_total = 0, ms_moe = 0, ms_tail = 0;   // ms_tail: the CPU-tail layers (hybrid placement)
  int steps = 0;
  long device_route_steps = 0, host_route_steps = 0, hybrid_route_steps = 0;   // (hybrid steps also count as device-routed)
};

class QwenEngine {
 public:
  QwenEngine(QwenModel& m, GlmExperts& x, int max_chunk);
  ~QwenEngine();
  void init_seq(QwenSeq& s, int64_t cap);
  // called by kv_reserve while VRAM is short, before the expert cache lends slots: free one idle sequence's KV, false when none is left
  std::function<bool()> kv_reclaim_;
  void reset_seq(QwenSeq& s);
  const float* logits_dev() const { return logits_dev_.as<float>(); }
  const int32_t* route_ids_dev() const { return rids_.as<int32_t>(); }   // the last MoE layer's top-k ids [rows, k] (validation)
  // KV / rope memory follows use (cap only reserves addresses); growth takes VRAM above HIVE_CACHE_RESERVE_MB (within the budget), giving
  //   back expert-cache tail slots when needed. Throws if even that is not enough.
  void kv_reserve(QwenSeq& s, int64_t tokens, bool reclaim = true);   // reclaim: may call kv_reclaim_ (not from init_seq)
  static constexpr int64_t kKvSlack = 16;
  // Vision input of the next prefill: rows [row, row + n) of the ids passed to prefill() take their embedding from dev (bf16 [n, hidden]);
  //   images also carry their merged grid (gh × gw tokens) for the rope coordinates. The caller clears it after the prefill.
  struct EmbedOverride { int64_t row; int n; const bf16* dev; int gh, gw; };
  void set_embed_override(std::vector<EmbedOverride> v) { emb_over_ = std::move(v); }
  // Device memory for `bytes` within the budget and the reserve (expert-cache tail slots are given back first); give_back_room() regrows.
  void make_room(size_t bytes);
  void give_back_room();
  size_t vram_avail() const;          // min(free VRAM, budget − used) — what the engine may still allocate
  void prefill(QwenSeq& s, const int32_t* ids, int T, float* logits_last);
  void decode(QwenSeq* const* seqs, const int32_t* ids, int M, float* logits);
  // MTP (hived contract, as GlmEngine)
  bool mtp_on() const { return mtp_k_ > 0 && m_.mtp() != nullptr; }
  int mtp_k() const { return mtp_k_; }
  int mtp_draft(QwenSeq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  // Drafts for S sequences at once (S ≤ kDraftRows): the k-step MTP chain runs with S rows (one per sequence); drafts / conf [S · k],
  //   n_out[s] = drafts made for sequence s (0 when it has no hidden state). Same results as S calls of mtp_draft.
  void mtp_draft_batch(QwenSeq* const* seqs, const int32_t* toks, int S, int k, int32_t* drafts, float* conf, int* n_out);
  void verify(QwenSeq& s, const int32_t* ids, int M, float* logits_host) { QwenSeq* sp[1] = {&s}; verify_batch(sp, &M, 1, ids, logits_host); }
  void rollback(QwenSeq& s, int n_keep) {
    HIVE_CHECK(vparts_.size() == 1 && vparts_[0].s == &s, "qwen rollback without a matching verify");
    rollback_batch(&n_keep, 1);
  }
  void verify_batch(QwenSeq* const* seqs, const int* M, int S, const int32_t* ids, float* logits_host);
  void rollback_batch(const int* n_keep, int S);
  void enable_batch_verify(int parts);
  int verify_parts_cap() const { return snap_parts_; }
  int prefill_tiles() const { return tiles_; }
  QwenForwardStats& stats() { return st_; }
  cudaStream_t stream() const { return stream_; }
  bool device_route_ready() const;    // every routed expert resident and device routing on
  void clear_graphs();                // drop the captured decode / verify graphs (buffers moved: sleep / wake)
  // Hybrid placement (QwenModel::gpu_layers() < n_layers): layers from gpu_layers() on run in this CPU engine; the hidden streams cross
  //   PCIe once per forward in each direction (bf16 values, exact). nullptr = everything on the GPU.
  cpu::QwenCpuEngine* tail() { return tail_.get(); }
  int gpu_layers() const { return m_.gpu_layers(); }
  // Validation hook: after every layer with the hidden stream h [rows, hc·hidden] (device, bf16, stream synchronized).
  std::function<void(int layer, const bf16* h_dev, int rows)> layer_hook;
  // Validation hook: before every layer l ≥ 1 of a forward (prefill or decode), may overwrite h [rows, hc·hidden] (device, bf16, stream synchronized) — per-layer
  //   injection of a reference's layer inputs.
  std::function<void(int layer, bf16* h_dev, int rows)> layer_inject;
  // Layer yield (see GlmEngine): called between layers of a prefill; the callback may run other forwards inside yield_enter / yield_exit.
  std::function<void(int layer)> layer_boundary;
  double pf_yield_body_ms = 0;
  struct YieldSave { bf16* hcur = nullptr; std::vector<EmbedOverride> emb; int64_t emb_base = 0; bool on = false; };
  bool yield_enter(YieldSave& s);
  void yield_exit(YieldSave& s);
  int yield_rows_cap() const { return hcur_ ? max_chunk_ : hy_rows_; }
  void alloc_yield_rows(int rows);

 private:
  // rows = consecutive tokens of one sequence (prefill / verify part) or one token per sequence (decode)
  struct RowInfo { QwenSeq* s; int64_t pos; };
  void forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, float* logits_host, bool all_logits);
  void prefill_layer_major(QwenSeq& s, const int32_t* ids, int T, float* logits_last);
  // the layer up to the MoE input: PLE (its layer), attention-side mix → linear / full attention → combine, MLP-side mix → x_ (dl_ keeps
  //   the MLP side's inject logits for the combine after the MoE)
  void layer_front(int l, const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, bf16* h);
  void embed(const std::vector<RowInfo>& rows, const int32_t* ids, int64_t emb_base, bf16* h);
  void set_rope(const std::vector<RowInfo>& rows, int64_t emb_base, bool prefill);
  void hc_mix(const HcW& w, const bf16* h, int rows, bf16* x, bool normed = false);
  void combine_mlp(int l, bf16* h, int rows, bool may_fuse);
  static bool hc_fuse();
  bool pre_normed_ = false;   // leaves hn_ / dl_ (inject logits) for the combine
  void hc_combine_(const HcW& w, bf16* h, const bf16* y, int rows, const bf16* dl);
  void ple_block(const std::vector<RowInfo>& rows, const int32_t* ids, bf16* h, bool prefill);
  void lin_layer(int l, const std::vector<RowInfo>& rows, bool prefill, const bf16* x, bf16* out);
  void attn_layer(int l, const std::vector<RowInfo>& rows, const bf16* x, bf16* out, bool mtp_layer = false);
  void moe_layer(int l, int rows, bool prefill, const bf16* x, bf16* out);
  void moe_shared(const MoeW& w, const bf16* x, int rows, bf16* shared_out);
  void lin(const QMat& W, const bf16* A, int lda, int rows, bf16* C, int ldc);
  void head(const bf16* h, int rows, float* logits_host);
  void router_logits(const QMat& W, const bf16* x, int rows, float* lg, int ld);         // final mixer → lm_head over the rows' hidden streams
  // MTP (qwen_mtp.cpp)
  // ov_row: index (in the current prefill call's ids) of next_ids[0], so pad tokens of an image span take the vision rows as their
  //   embedding like the main model did (−1 = no image rows in this call)
  void mtp_kv(QwenSeq& s, int64_t pos0, int n, const bf16* h_rows, const int32_t* next_ids, int64_t ov_row = -1);
  // mtp_kv for several sequences in one pass (the rollback of a batched verify): same rows, same per-row arithmetic, one stream sync
  struct MtpKvPart { QwenSeq* s; int64_t pos0; int n; const bf16* h_rows; const int32_t* next_ids; };
  void mtp_kv_multi(const MtpKvPart* parts, int P);
  DevBuf mtp_hin_;   // mtp_kv_multi: the parts' hidden rows gathered
  void mtp_input(const bf16* emb, const bf16* h_rows, int n, bf16* hc_out);
  void mtp_step(QwenSeq& s, int step, int32_t tok, const bf16* h_in, int64_t pos, bf16* h_out);
  void mtp_after_forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill);
  void mtp_moe(const MoeW& w, const bf16* x, int rows, bf16* out);
  void alloc_snapshots(int parts);
  size_t kv_bytes(const QwenSeq& s, int64_t tokens) const;
  void kv_trim(QwenSeq& s, int64_t tokens);
  void apply_embed_override(bf16* x_dev, int64_t base, int rows);
  QwenModel& m_;
  GlmExperts& ex_;
  const QwenConfig& c_;
  cudaStream_t stream_ = nullptr;
  Blas blas_;
  int max_chunk_;
  int n_lin_ = 0, n_attn_ = 0;
  std::vector<int> lin_index_, attn_index_;
  int64_t kv_base_ = 0, kv_step_ = 4096;
  size_t kv_headroom_ = 0;
  size_t budget_ = 0, free0_ = 0;     // HIVE_QWEN_VRAM_MB (0 = none) · free VRAM before the model was loaded (see vram_avail)
  // hidden stream rows: the work buffer h_, a block of the layer-major buffer, or hy_ (inner forwards of a layer yield)
  bf16* hcur_ = nullptr;
  bf16* hb() const { return hcur_ ? hcur_ : h_.as<bf16>(); }
  DevBuf hy_; int hy_rows_ = 0;
  int tiles_ = 1;
  std::vector<EmbedOverride> emb_over_;
  int64_t emb_base_ = 0;
  // work buffers (rows ≤ max_chunk)
  DevBuf h_, x_, y_, hn_, dl_, hs_, hu_, proj_, q_, iq_, idx_, attn_out_, gdn_out_, mlp_out_f32_, shared_out_, sh_gu_, sh_y_, router_, rids_, rw_;
  DevBuf ws_, logits_dev_, xfin_, gemv_ws_;
  size_t ws_bytes_ = 0;
  // PLE
  uint8_t* ple_rows_pin_ = nullptr;   // pinned [max_chunk, 2560] FP8 rows of the lookup (read by ple_dequant through UVA)
  std::vector<int64_t> ple_ids_;
  std::unique_ptr<PleTable> ple_;
  PleHash ple_hash_;
  float ple_scale_ = 1.f;
  // attention row tables: a ring in pinned memory (a call's table stays untouched until the stream passed it — qrows_alloc)
  QsaRow* qrows_pin_ = nullptr;
  int qrows_cap_ = 0;
  size_t qrows_off_ = 0;
  QsaRow* qrows_alloc(int n);
  DevBuf rope_inv_freq_, rope_axis_;
  int32_t* rope_pin_ = nullptr;       // pinned [max_chunk, 3] rope coordinates of the rows
  int32_t** rope_dst_pin_ = nullptr;  // pinned [max_chunk]: where each row's coordinates go (its sequence's rope buffer)
  float** gdn_ptr_pin_ = nullptr;     // pinned [n_lin · 16]: 8 conv then 8 state pointers per linear layer (decode / verify rows)
  float** ple_ptr_pin_ = nullptr;     // pinned [8]: PLE conv state per decode row
  bf16* emb_pin_ = nullptr;           // pinned [max_chunk, hidden] embedding rows
  std::vector<int32_t> h_ids_; std::vector<float> h_w_;
  int32_t* rids_h_ = nullptr; float* rw_h_ = nullptr;   // mapped host router outputs (≤ 64 rows)
  // device routing (every expert resident): record pointer per (MoE layer, expert), rebuilt when the residency changed
  DevBuf rec_tab_, pairs_, moe_ws_;
  size_t moe_ws_bytes_ = 0;
  bool device_route_ = true, dr_now_ = false, fn_moe_ = true, fuse_ = true;
  bool fn_pf_ = false, pf_dev_ = false;   // FN grouped prefill MoE (HIVE_QWEN_FN_PREFILL) · this forward is a prefill with every expert resident
  int pf_rows_ = 0;                       // its sub-chunk (HIVE_QWEN_MOE_PF_ROWS)
  DevBuf pf_ws_;
  int qsa_split_ = 0;
  DevBuf qsa_split_ws_;
  DevBuf qrows_dev_;
  DevBuf ptr_dev_;                        // device mirror of gdn_ptr_pin_ | ple_ptr_pin_ | snap_ptr_pin_ (upload_ptr_tables)
  size_t ptr_gdn_n_ = 0, ptr_snap_n_ = 0;
  void upload_ptr_tables();
  float** dev_ptr(float** pin) const;
  bool no_regrow_ = false;
  bool snap_replay_ = false;               // HIVE_QWEN_SNAP_REPLAY: verify parts ≥ 1 keep one state snapshot and replay on rollback
  bool replay_part(int p) const { return snap_replay_ && p >= 1; }
  std::vector<DevBuf> snap_conv0_;         // per (part, linear layer): conv window before the verify (replay parts)
  DevBuf proj_save_, rb_dev_;              // projections of the verify rows per linear layer · replay pointer table
  float** rb_pin_ = nullptr;
  int rb_parts_ = 0;   // parts rb_pin_ / rb_dev_ hold a section for
  DevBuf dq_;                             // bf16 expansion of an 8-bit-only matrix for a prefill GEMM (HIVE_QWEN_DENSE8)
  std::vector<DevBuf> rs_;            // fused decode: per layer [router 513 | shared gate 640 | shared up 640] rows (bf16)
  DevBuf mtp_rs_;                     // the same for the MTP layer
  glm::Fp8BMat* mtp_tab_ = nullptr;   // device table of the MTP layer's FP8 experts (fn_moe_fp8_routed)
  int mtp_slot_ = 0;                  // draft step: the attention row-table slot in graph phases
  std::vector<const uint8_t*> rec_host_, rec_up_;   // the table built for this step · the one on the device
  // allow_host: experts that are not resident get the address of their record in pinned host memory (mapped device routing) ·
  //   allow_null: they get nullptr (hybrid route: the CPU computes them)
  bool refresh_rec_table(bool allow_host = false, bool allow_null = false);
  bool mapped_route_ = false;           // HIVE_QWEN_MAPPED_ROUTE: decode / verify route on the device with a partial expert cache
  double mapped_min_ = 0.6;             // … while the cache holds at least this share of the experts (HIVE_QWEN_MAPPED_MIN)
  int mapped_nonres_ = -1, nonres_now_ = 0;
  bool map_now_ = false;                // this decode / verify forward reads some experts from host memory
  int32_t* route_h_ = nullptr;          // its routing, per MoE layer [kDecRows · K] (pinned, D2H nodes of the forward)
  int route_rows_ = 0;
  void route_note(int li, int rows);
  void route_bookkeeping();
  // Hybrid route (HIVE_QWEN_HYBRID_ROUTE, default on): decode / verify with a partial expert cache below the mapped threshold. The step
  //   stays on the device (CUDA graph): per MoE layer the routing (mapped) and the MoE input (D2H node) go to pinned host memory, a host
  //   node on a side stream runs the non-resident experts on the CPU pool (GlmExperts::cpu_rows) while the resident ones run on the GPU,
  //   and the reduce adds the CPU share (fn_moe_decode_routed's add / join). Replaces the host route's 48+ stream syncs and eager launches.
  bool hybrid_route_ = false, hyb_now_ = false;
  cudaStream_t hyb_st_ = nullptr;
  std::vector<cudaEvent_t> hyb_ev_;      // [2 · layers]: fork (MoE input copied), join (CPU share written)
  struct HybArg { QwenEngine* e; int li; };
  std::vector<HybArg> hyb_arg_;
  bf16* hyb_x_ = nullptr;                // pinned [kDecRows, hidden]: the MoE input rows
  float* hyb_y_ = nullptr;               // pinned [kDecRows, hidden]: the CPU share (read by the reduce through UVA)
  std::vector<float> hyb_xf_;
  std::vector<int32_t> hyb_ids_;
  int hyb_rows_ = 0;                     // rows of the running forward (the host nodes read it)
  long hyb_cpu_pairs_ = 0;               // pairs the CPU computed since the last bookkeeping
  int hyb_max_rows_ = 0;                 // HIVE_QWEN_HYB_ROWS: steps with more rows take the host route (0 = no limit)
  int hyb_pcie_div_ = 0;                 // HIVE_QWEN_HYB_PCIE: every n-th non-resident expert of a layer is read over PCIe by the GPU (0 = CPU only)
  DevBuf host_tab_;                      // pinned host record per key (device array; the PCIe share)
  bool hyb_diag_nocpu_ = false;          // HIVE_QWEN_HYB_DIAG=nocpu (measurement only, wrong outputs): the host nodes skip the CPU experts
  std::vector<int> hyb_mrank_;
  static void CUDART_CB hyb_host(void* arg);
  void hyb_cpu(int li);
  void hyb_fork(int li, int rows, const bf16* x);
 public:
  bool mapped_route() const { return mapped_route_; }
 private:
  void moe_device(int li, int rows, const bf16* x, float* out);
  // verify
  bool verify_ = false;
  struct VPart { QwenSeq* s; int64_t pos0; int M, r0; };
  std::vector<VPart> vparts_;
  std::vector<int32_t> verify_ids_;
  int mtp_k_ = 0, snap_parts_ = 1, snap_live_ = 0, snap_idle_ = 0;
  static constexpr int kSnapIdle = 256;   // single-part verify steps before parts 1.. are freed
  void ensure_parts(int n);
  void release_parts(int keep);
  std::vector<DevBuf> snap_state_, snap_conv_, snap_ple_;   // [(part · kmax + row) · n_lin + li] · ple [part · kmax + row]
  float** snap_ptr_pin_ = nullptr;    // [part][n_lin][conv 8 | state 8], then [part][ple 8]
  DevBuf fh_;                          // [rows, hc·hidden] final hidden streams of the last forward (MTP inputs)
  // MTP work
  DevBuf mtp_emb_, mtp_en_, mtp_h_, mtp_hn_, mtp_fc_, mtp_hout_, mtp_ids_d_, mtp_conf_d_, one_f_, draft_rows_, draft_idx_;
  DevBuf mtpb_h_, mtpb_ids_d_, mtpb_conf_d_;   // batched drafts: hidden rows [3][S][HC] (input, ping, pong), ids / conf [step][S]
  void mtp_step_rows(QwenSeq* const* seqs, int S, int step, const int32_t* toks, const int64_t* pos, const bf16* h_in, bf16* h_out);
  DevBuf draft_q_, draft_s_, draft_lg_, draft_part_;   // FP8 draft head (HIVE_QWEN_DRAFT_FP8), its bf16 logits, fn_draft_pick partials
  bf16* mtp_emb_pin_ = nullptr;
  int draft_n_ = 0;
  QwenForwardStats st_;
  // hybrid placement
  std::unique_ptr<cpu::QwenCpuEngine> tail_;
  bf16* tail_pin_ = nullptr;          // pinned bf16 [max_chunk, hc·hidden]: the streams on their way to / from the CPU tail
  std::vector<float> tail_h_;
  void run_tail(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, const int32_t* coords, bf16* h);
  // CUDA graphs of decode / verify forwards
  int gphase_ = 0;                    // 0 = eager, 1 = host part only (before a replay), 2 = device part only (capture)
  int graph_ctx_ = 0;
  std::map<std::string, cudaGraphExec_t> graphs_;
  std::map<std::string, int> graph_seen_;
  // HIVE_QWEN_GRAPH_CAP (default: 32 while the expert cache cannot hold every expert, else no cap): at most this many captured graphs,
  //   least recently launched dropped first. Graph executables take device memory, and on a VRAM budget every MiB of it is taken from the
  //   expert cache (KV growth and make_room lend slots while vram_avail is short). Measured (32 GB, hybrid route, short pass c4/c8 2026-10-13):
  //   the cache fell 2797 → 1354 → 331 slots over the c4 / c8 / 64K steps against 2797 → 1181 on the host route (no graphs) — the batched
  //   verify keys (rows, part layout, context bucket) and per-sequence draft keys accumulate without bound. Cap 128 → 32 at 48 GB:
  //   c1 / c4 / c8 129.8 / 181.9 / 230.0 → 131.4 / 186.5 / 248.4 tok/s; 64K TTFT at 32 GB 20.2 s (no cap) → 15.0 (host route 14.8).
  std::map<std::string, uint64_t> graph_tick_;
  uint64_t graph_clock_ = 0;
  size_t graph_cap_ = 0;
  bool graph_cap_set_ = false;
  std::map<std::string, cudaGraphExec_t>::iterator graph_put(const std::string& key, cudaGraphExec_t ex);
  void graph_used(const std::string& key) { if (graph_cap_) graph_tick_[key] = ++graph_clock_; }
 public:
  size_t n_graphs() const { return graphs_.size(); }
 private:
  bool graph_forward(const std::vector<RowInfo>& rows, const int32_t* ids, float* logits_host, bool all_logits);
  void graph_body(const std::vector<RowInfo>& rows, const int32_t* ids, float* logits_host, bool all_logits);
};

}  // namespace hive::qwen
