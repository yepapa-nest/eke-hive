// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Internal declarations of the GLM-5.3-Flash CPU engine (hive/glm/glm_cpu.h) shared by its source files.
#pragma once
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hive/cpu/cpu_backend.h"
#include "hive/glm/glm_cpu.h"
#include "hive/safetensors.h"

namespace hive::glm::cpu {

using hive::cpu::Mat;
using hive::cpu::MatJob;
using hive::cpu::MatView;
using hive::cpu::bf16r;

inline float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline uint16_t f2b(float x) { const float r = bf16r(x); uint32_t u; std::memcpy(&u, &r, 4); return (uint16_t)(u >> 16); }
inline float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }

// Parallel loop over [0, n) in blocks of `grain` (serial when there is one block).
void par_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f);
// Y = X·Wᵀ, rounded to bf16 when `round` (the GPU path's bf16 GEMM outputs)
void lin(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy, bool round = true);
inline void lin(const Mat& W, const float* X, int M, int ldx, float* Y, int ldy, bool round = true) { lin(MatView{&W, 0, W.N}, X, M, ldx, Y, ldy, round); }
void round_rows(float* Y, int M, int ldy, int N);
// out = bf16(w · (x · rsqrt(mean(x²) + eps))) per row (hive rmsnorm: one rounding)
void rmsnorm_rows(const float* x, int ldx, const float* w, int M, int K, float eps, float* out, int ldo);

struct HcW { Mat fn; std::vector<float> base, scale; };   // fn fp32 [24, 4·4096]
struct KdaW {
  Mat in;                                                // [q 8192 | k 8192 | v 8192 | f_a 128 | g_a 128 | b 64] × 4096, BF16
  Mat f_b, g_b, o;
  std::vector<float> conv_w, A_log, dt_bias, o_norm;     // [24576·4] (tap 3 = current token), [64], [8192], [128]
  int fa_dim = 128, ga_dim = 128;
};
struct DsaW {
  Mat q_a, kv_a, ix, q_b, wq_b, ukt, kv_b, o;            // ix = [wk 128 | pool gate 128 | weights_proj 32] × 4096 BF16; ukt = per head W_UKᵀ [64·512, 256]
  std::vector<float> q_a_norm, kv_a_norm, k_norm_w, k_norm_b, ape;  // ape [4, 128]
};
struct DenseW { Mat g, u, d; };                          // NVFP4 (global scale inside each Mat)
struct MoeW {
  Mat router;                                            // BF16 [288, 4096], fp32 logits
  std::vector<float> bias;
  Mat sh_g, sh_u, sh_d;                                  // shared expert (BF16 or Z.ai FP8)
  Mat gu, d;                                             // routed experts stacked: gu = [gate rows of every expert | up rows], d = [rows 0..H/2 of every
                                                         //   expert's down | rows H/2..H] — each half on its own NUMA node
  std::vector<float> g_gate, g_up, g_down;               // NVFP4 global scale per expert (1 for FP8)
};
struct LayerW { bool kda = false, moe = false, loaded = false; std::vector<float> in_norm, post_norm; HcW ha, hf; KdaW kw; DsaW dw; DenseW dense; MoeW mw; };
struct MtpW { std::vector<float> enorm, hnorm, head_norm, in_norm, post_norm; Mat eh; DsaW dw; MoeW mw; };

struct VisionW {
  struct Block { Mat qkv, proj, gate, up, down; std::vector<float> n1, n2, qkv_b, q_norm, k_norm, proj_b, gate_b, up_b, down_b; };
  Mat patch, down, m_proj, m_gate, m_up, m_down;
  std::vector<float> patch_b, post_norm, down_b, m_ln_w, m_ln_b;
  std::vector<Block> blocks;
  int dim = 1024, heads = 16, inter = 4096, out = 4096, pinter = 10240;
  float eps = 1e-5f, limit = 10.f;
};

struct RowInfo { CpuSeq* s; int64_t pos; };

struct GlmCpuEngine::Impl {
  CpuConfig c;
  std::string ckpt_dir;
  std::unique_ptr<Checkpoint> ck, f8;
  std::vector<LayerW> layers;
  std::vector<int> kda_index, dsa_index;
  int n_kda = 0, n_dsa = 0, first = 0, last = 0;
  std::unique_ptr<MtpW> mtp;
  int mtp_k = 0;
  Mat lm_head, draft_head;                 // draft_head: HIVE_GLM_DRAFT_VOCAB rows of lm_head (copied), draft_rows their ids
  std::vector<int32_t> draft_rows;
  std::vector<float> final_norm;
  std::vector<uint16_t> embed;             // [vocab, 4096] bf16
  int max_chunk = 2048;
  std::vector<EmbedOverride> emb_over;
  int64_t emb_base = 0;
  CpuStats st;
  std::unique_ptr<VisionW> vision;
  int depth = 0;
  std::vector<std::vector<float>> h_stack;
  // verify
  bool verify = false;
  struct VPart { CpuSeq* s; int64_t pos0; int M, r0; };
  std::vector<VPart> vparts;
  std::vector<int32_t> verify_ids;
  std::vector<float> fh;                   // final normed hidden of every row of the last forward [rows, 4096] (MTP on)
  struct Snap { std::vector<std::vector<float>> conv, state; };
  std::vector<std::vector<Snap>> snaps;    // [part][row < M−1]
  std::function<void(int, const float*, int)>* layer_hook = nullptr;
  std::function<void(int)>* layer_boundary = nullptr;
  std::function<void(int, float*, int)>* layer_inject = nullptr;
  int cur_layer = -1;               // layer of the running moe() (−1 = NextN) for the routing dump
  FILE* route_dump = nullptr;       // HIVE_GLM_CPU_ROUTE_DUMP: per MoE call "layer row margin ids…" (margin = 8th − 9th biased score)
  ~Impl();

  void load(bool load_mtp);
  void load_dsa(Checkpoint& ck, const std::string& a, DsaW& w);
  void load_moe(const std::string& p, MoeW& w, bool mtp_layer);
  Mat mat_fp8ok(const std::string& name);  // Z.ai FP8 when HIVE_GLM_FP8_DIR has it, else BF16 from the checkpoint

  void forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, float* logits, bool all_logits);
  void run_range(const std::vector<RowInfo>& rows, float* h, bool prefill);   // layers first..last on the hc streams h
  bool hybrid = false;
  std::function<void(int, int, const int32_t*, const float*, const float*, float*)> ext_experts;
  void prepare_snaps(CpuSeq* const* seqs, const int* Ms, int S);
  const MoeW& mtp_moe() const;
  // hyper-connections: pre (mix, Sinkhorn, collapse, RMS norm) → xn [M, H]; post/comb kept per row for hc_post
  void hc_pre(const HcW& w, const std::vector<float>& norm, const float* h, int M, float* xn, float* post, float* comb);
  void hc_post(float* h, const float* y, int M, const float* post, const float* comb);
  void kda_layer(const KdaW& w, int ki, const std::vector<RowInfo>& rows, const float* xn, float* out);
  // DSA attention; mtp_layer = the NextN cache; kv_only = write the cache entries only (MTP entries)
  void dsa_layer(const DsaW& w, int di, bool mtp_layer, const std::vector<RowInfo>& rows, const float* xn, float* out, bool kv_only = false);
  void dense_mlp(const DenseW& w, int M, const float* xn, float* out);
  void moe(const MoeW& w, int M, const float* xn, float* out);
  void final_hidden(const float* h, int M, float* out);   // bf16(rmsnorm(mean of the streams))
  // MTP (glm_cpu_mtp.cpp)
  void mtp_input(const float* emb, const float* h_rows, int n, float* x0);
  void mtp_kv(CpuSeq& s, int64_t pos0, int n, const float* h_rows, const int32_t* next_ids);
  void mtp_after_forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill);
  int mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf, float* logits0);
  // vision (glm_cpu_vision.cpp)
  void ensure_vision();
  void encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out);
};

Mat load_bf16_mat(Checkpoint& ck, const std::vector<std::string>& names);
std::vector<float> load_vec(Checkpoint& ck, const std::string& name);

}  // namespace hive::glm::cpu
