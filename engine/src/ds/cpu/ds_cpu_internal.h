// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Internal declarations of the DeepSeek-V4.1-Flash CPU engine (hive/ds/ds_cpu.h) shared by its source files.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hive/cpu/cpu_backend.h"
#include "hive/ds/ds_cpu.h"
#include "hive/engram_hash.h"
#include "hive/safetensors.h"

namespace hive::ds::cpu {

using hive::cpu::Mat;
using hive::cpu::MatJob;
using hive::cpu::MatView;
using hive::cpu::bf16r;

// ---- numerics (ds_cpu_ops.cpp) — every function mirrors one reference operation; the comments name it -------------------------------------
void par_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f);
inline float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }
inline uint16_t f2b(float x) { x = bf16r(x); uint32_t u; __builtin_memcpy(&u, &x, 4); return (uint16_t)(u >> 16); }
void round_rows(float* y, int M, int N, int ld);                                   // y = bf16(y)
// act_quant(x, 32, ue8m0) of the reference (kernel.py), returned dequantized: per 32-block s = 2^ceil(log2(max(amax,1e-4)·(1/448))),
//   y = e4m3_rne(clamp(x/s, ±448))·s. In place allowed (out == x).
void act_quant_fp8(const float* x, int M, int K, int ldx, float* out, int ldo);
// fp4_act_quant(x, block, e8m0 / e4m3 scales) (kernel.py), dequantized: e8m0 s = 2^ceil(log2(max(amax, 6·2^-126)·(1/6))),
//   e4m3 s = e4m3(max(amax, 6·2^-9)/6); y = e2m1_rne(clamp(x/s, ±6))·s.
void fp4_quant_row(float* x, int K, int block, bool e4m3_scale);
// rmsnorm(x, w, eps) of the oracle: bf16(w · x · rsqrt(mean(x²) + eps)) — x fp32 holding bf16 values, w fp32 (bf16 values)
void rmsnorm_rows(const float* x, int ldx, const float* w, int M, int K, float eps, float* out, int ldo);
// RoPE tables of one kind (reference precompute_freqs_cis): freqs[rd/2]; cos/sin of position p computed on demand in fp32
struct Rope {
  std::vector<float> freq;
  void init(int rd, int64_t orig, float base, float factor, float beta_fast, float beta_slow);
  // apply_rotary_emb on the last rd dims of a [heads, hd] row at position pos (pairs (2i, 2i+1)), result rounded to bf16; inverse = conjugate
  void apply(float* x, int heads, int hd, int rd, int64_t pos, bool inverse) const;
};
// hc_split_sinkhorn of one row: mixes[(2+hc)·hc] → pre[hc], post[hc], comb[hc·hc] (comb[j·hc + i])
void sinkhorn_row(const float* mixes, const float* scale, const float* base, int hc, int iters, float eps, float* pre, float* post, float* comb);
// e4m3 RNE / e2m1 RNE value (exact reference casts)
float e4m3_round(float v);
// sparse attention of one query row (reference sparse_attn): q [H, D] (bf16 values), keys K[n, D] fp32 (bf16 values), sink[H] →
//   o [H, D] = bf16(Σ_j bf16(exp(s_j − max)) · k_j / (Σ_j exp(s_j − max) + exp(sink − max))), s_j = (q · k_j) · scale. ws ≥ H·n floats.
void sparse_attn_row(const float* q, int H, int D, const float* K, int n, const float* sink, float scale, float* o, float* ws);
// indexer scores of one query row (reference: bf16 einsum → relu → × bf16 weights → Σ over heads in fp32 → bf16): q [Hi, Di], keys [T, Di]
//   at rows key_idx[0..n) (nullptr = 0..n−1) → score[n] (bf16 values)
void index_scores_row(const float* q, const float* w, int Hi, int Di, const float* keys, const int32_t* key_idx, int n, float* score);
// top-k positions of the scores above −inf (ties → smaller index), ascending → sel; returns the count
int topk_sorted(const float* score, const int32_t* pos, int n, int k, int32_t* sel, std::vector<int32_t>& scratch);

// ---- weights -----------------------------------------------------------------------------------------------------------------------------
struct AttnW {
  Mat wq_a, wkv, wq_b, wo_a, wo_b;            // FP8 (32×32 e8m0 blocks)
  std::vector<float> q_norm, kv_norm, sink;
  Mat comp_wkv, comp_wgate;                    // BF16 [head_dim, dim] (wgate: ratio > 1)
  std::vector<float> comp_norm;
  Mat idx_wq_b;                                // FP8 [Hi·Di, q_lora_rank]
  Mat idx_wproj;                               // BF16 [Hi, dim]
  Mat idx_wk;                                  // BF16 [Di, head_dim] (kv-source index layers)
  std::vector<float> idx_k_norm;
};
// one routed expert: gate / up rows and the down projection (two halves in the NUMA-split layout, one view in the mmap layout)
struct ExpertW { MatView g{nullptr, 0, 0}, u{nullptr, 0, 0}, d0{nullptr, 0, 0}, d1{nullptr, 0, 0}; };
struct MoeW {
  int E = 0, k = 0;
  Mat gate;                                    // BF16 [E, dim] (the oracle's fp32 copy holds the same values)
  std::vector<float> bias, bias_vl;
  Mat sh1, sh3, sh2;                           // FP8 shared expert
  std::vector<Mat> own;                        // expert storage (stacked, owned) — or aliases of the checkpoint mapping (alias = true)
  bool alias = false;
  std::vector<ExpertW> ex;
};
struct EngramTab {                             // rows [N, hd] e4m3 + [N, hd/32] e8m0, split into two halves (NUMA) or one mapping
  uint64_t rows = 0, split = 0;
  const uint8_t* v[2] = {nullptr, nullptr};
  const uint8_t* s[2] = {nullptr, nullptr};
  bool owned = false;
  size_t bytes[2] = {0, 0};
  const uint8_t* val(uint64_t r, int hd) const { return r < split ? v[0] + r * hd : v[1] + (r - split) * hd; }
  const uint8_t* scl(uint64_t r, int ns) const { return r < split ? s[0] + r * ns : s[1] + (r - split) * ns; }
};
struct EngramW { Mat wkv; std::vector<float> qk; int hash_index = 0; EngramTab tab; };
struct LayerW {
  int id = 0, ratio = 0;
  bool kv_src = false, idx_src = false, mtp = false;
  int comp_ord = -1;                           // kv-source ordinal whose compressed KV / indexer keys this layer reads (ratio > 0)
  int own_ord = -1;                            // this layer's own kv-source ordinal (kv_src)
  AttnW at;
  MoeW moe;
  std::vector<float> attn_norm, ffn_norm;
  Mat hc_attn_fn, hc_ffn_fn;                   // F32 [(2+hc)·hc, hc·dim]
  float hc_attn_scale[3] = {0, 0, 0}, hc_ffn_scale[3] = {0, 0, 0};
  std::vector<float> hc_attn_base, hc_ffn_base;
  std::unique_ptr<EngramW> engram;
};
struct MtpStage {
  LayerW L;
  Mat main_proj;                               // FP8 [dim, n_targets·dim] (stage 0)
  std::vector<float> main_norm, norm;
  Mat markov_head;                             // BF16 [V, rank] (last stage)
  std::vector<uint16_t> markov_embed;          // bf16 [V, rank]
  std::vector<float> conf;                     // [dim + rank]
};
struct VisionW {                               // ViT + aligner (ds_cpu_vision.cpp), BF16 matrices, fp32 vectors
  struct Block { Mat qkv, wo, w1, w2; std::vector<float> n1, n2, bqkv, bo; };
  Mat patch, a1, a2;
  std::vector<float> patch_b, a1_b, a2_b, norm;
  std::vector<Block> blocks;
  ~VisionW() {
    for (Mat* m : {&patch, &a1, &a2}) ::hive::cpu::free_mat(*m);
    for (Block& b : blocks) for (Mat* m : {&b.qkv, &b.wo, &b.w1, &b.w2}) ::hive::cpu::free_mat(*m);
  }
};

// ---- one forward -------------------------------------------------------------------------------------------------------------------------
struct Part { CpuSeq* s; int64_t pos0; int r0, n; };
struct Fwd {
  std::vector<Part> parts;
  int M = 0;
  std::vector<int64_t> pos;
  std::vector<int> part_of;
  std::vector<int32_t> ids;
  std::vector<int8_t> img;
  std::vector<float> h, premix;                // [M, hc·dim], [M, hc]
  std::vector<int32_t> topk;                   // [M, index_topk] compressed positions of the last index-source layer (ascending)
  std::vector<int> topk_n;
  std::vector<int32_t> cand;                   // [M, cand_topk_blocks] candidate blocks of the candidate-source layer
  std::vector<int> cand_n;
  bool have_cand = false, have_topk = false;
  std::vector<int64_t> hashes;                 // engram [M, n_engram_layers, cols]
  int hash_layers = 0;                         // n_engram_layers of hashes
  std::function<void(int, int, uint8_t*, uint8_t*)> engram_rows;   // block mode: engram rows from the caller (else the engine's tables)
  int64_t win_min = 0;                         // window keys before this position are not seen (decoder tail rows)
  bool verify = false, prefill = false;
  std::vector<float> mh;                       // [M, n_targets·dim] target-layer hc means (bf16 values)
  std::vector<float> comp_in;                  // verify: ratio>1 compressor inputs [n_src2][M][2·head_dim] for rollback
};

struct DsCpuEngine::Impl {
  DsConfig c;
  EngineOptions opt;
  std::string dir;
  std::unique_ptr<Checkpoint> ck;
  std::vector<LayerW> layers;
  std::vector<std::unique_ptr<MtpStage>> mtp;
  std::vector<uint16_t> embed;                 // bf16 [V, dim]
  Mat head;                                    // BF16 [V, dim]
  std::vector<float> norm;
  bool head_ok = false;
  Rope rope_win, rope_comp;
  std::vector<int> kv_src;                     // kv-source layer ids (ordinal = index)
  std::vector<int> src2;                       // ordinals with ratio > 1
  int tail_layer = -1;
  const hive::EngramHash* eh = nullptr;
  std::unique_ptr<VisionW> vision;
  std::vector<float> image_start, image_end, image_newline;
  CpuStats st;
  int legacy = 0;   // A/B measurement only (set_ab_legacy): 1 = attention in row blocks, 2 = compressor pooling row by row
  DsCpuEngine* pub = nullptr;
  // pending verify (until rollback_batch)
  struct VPart { CpuSeq* s; int64_t pos0; int M, r0; size_t hist; std::vector<uint16_t> ring_snap; std::vector<float> st_snap; };
  std::vector<VPart> vparts;
  std::vector<float> v_mh, v_comp_in;
  int v_rows = 0;
  ~Impl();

  // load (ds_cpu_load.cpp)
  void load();
  void load_layer(LayerW& L, const std::string& p, int id, bool backbone, int E, int k);
  void load_mtp();
  // forward (ds_cpu_engine.cpp)
  void run(Fwd& f, const std::vector<DsCpuEngine::Image>* images, float* logits, bool all_logits);
  void layer_forward(LayerW& L, Fwd& f, bool skip_compress);
  void hc_front(LayerW& L, Fwd& f, bool ffn, const float* pre_in, float* pre, float* post, float* comb, float* xn);
  void hc_post(Fwd& f, const float* y, const float* post, const float* comb);
  void engram(LayerW& L, Fwd& f);
  void attention(LayerW& L, Fwd& f, const float* xn, float* out, bool skip_compress, bool mtp_block, int mtp_stage);
  void compress(LayerW& L, Fwd& f, const float* ckv, const float* csc);
  void indexer(LayerW& L, Fwd& f, float* iq, const float* iw);
  void moe(LayerW& L, Fwd& f, const float* xn, float* out);
  void head_rows(const float* h, const float* premix, int M, const std::vector<float>& nw, float* logits, float* x_out);
  void tail_prepare(LayerW& L, Fwd& f, int tail);
  void dump(const char* what, int l, const Fwd& f, const float* data, int cols);
  const Rope& rope_of(const LayerW& L) const { return L.ratio ? rope_comp : rope_win; }
  // DSpark (ds_cpu_mtp.cpp)
  void mtp_sync(CpuSeq& s, const float* mh, int n, int64_t pos0);
  int mtp_draft(CpuSeq& s, int32_t tok, int32_t* drafts, float* conf);
  // vision (ds_cpu_vision.cpp)
  void ensure_vision();
  void encode_image(const uint16_t* patches, int nh, int nw, std::vector<float>& out, const std::string& tag);
};

// checkpoint helpers (ds_cpu_load.cpp)
std::vector<float> load_vec(Checkpoint& ck, const std::string& name);
Mat load_mat(Checkpoint& ck, const std::string& name);   // FP8 (+ .scale) / BF16 / F32 by dtype

}  // namespace hive::ds::cpu
