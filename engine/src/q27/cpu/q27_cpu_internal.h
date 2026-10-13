// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Internal declarations of the Qwen3.8-27B CPU engine (hive/q27/q27_cpu.h) shared by its source files.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hive/cpu/cpu_backend.h"
#include "hive/q27/q27_cpu.h"
#include "hive/safetensors.h"
#include "q27_cpu_simd.h"

namespace hive::q27::cpu {

using hive::cpu::Mat;
using hive::cpu::MatJob;
using hive::cpu::MatView;
using hive::cpu::bf16r;

inline float sigm(float x) { return simd::sigm_s(x); }
inline float siluf(float x) { return simd::silu_s(x); }

// Parallel loop over [0, n) in blocks of `grain` (serial when the work is small).
void par_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f);

// Gated DeltaNet: qkvz = [in_proj_qkv 10240 | in_proj_z 6144] (FP8), ba = [in_proj_b 48 | in_proj_a 48] (BF16)
struct LinW { Mat qkvz, ba, out; std::vector<float> conv_w, A_log, dt_bias, norm; };
// full attention: qkv = [q_proj 12288 (per head q 256 | gate 256) | k_proj 1024 | v_proj 1024]
struct AttW { Mat qkv, out; std::vector<float> q_norm, k_norm; };
struct MlpW { Mat gate, up, down; };
struct LayerW { bool attn = false, on = false; LinW lin; AttW att; MlpW mlp; std::vector<float> ln1, ln2; };
struct MtpW { Mat fc; std::vector<float> pre_emb_norm, pre_hidden_norm, norm; LayerW layer; };

// vision encoder weights (q27_cpu_vision.cpp)
struct VisionW {
  struct Block { Mat qkv, proj, fc1, fc2; std::vector<float> n1w, n1b, n2w, n2b, qkv_b, proj_b, fc1_b, fc2_b; };
  Mat patch, m1, m2;
  std::vector<float> patch_b, pos, mn_w, mn_b, m1_b, m2_b;
  std::vector<Block> blocks;
  int dim = 1152, heads = 16, inter = 4304, out = 5120, side = 48;
  float eps = 1e-6f;
};

using RowInfo = Q27CpuEngine::Row;

struct Q27CpuEngine::Impl {
  Q27Config c;
  CpuOptions opt;
  std::unique_ptr<Checkpoint> ck;
  std::vector<LayerW> layers;
  std::vector<int> lin_index, attn_index;   // per layer: index among linear / attention layers (−1 otherwise)
  std::unique_ptr<MtpW> mtp;
  Mat lm_head;
  std::vector<float> final_norm;
  std::vector<uint16_t> embed;      // [vocab, hidden] bf16 (CPU-only daemon)
  std::vector<double> rope_inv;     // [rot/2]
  std::vector<int> rope_axis;       // [rot/2]
  int max_chunk = 2048, mtp_k = 0;
  std::vector<EmbedOverride> emb_over;
  int64_t emb_base = 0;
  CpuStats st;
  std::unique_ptr<VisionW> vision;
  std::string ckpt_dir;
  size_t wbytes = 0;
  bool vis_avail = false;           // the checkpoint has a vision encoder
  int depth = 0;                    // nesting of forwards (a decode inside a prefill's layer boundary)
  std::vector<std::vector<float>> h_stack;
  // verify
  bool verify = false;
  struct VPart { CpuSeq* s; int64_t pos0; int M, r0; };
  std::vector<VPart> vparts;
  std::vector<int32_t> verify_ids;
  std::vector<float> draft_logits;  // first draft of the last mtp_draft (validation)
  std::vector<float> fh;            // residual streams of the last forward [rows, 5120] (MTP inputs)
  // snapshots per verify row r < M−1 of part p: [p][r] → per linear layer conv + state (owned layers only)
  struct Snap { std::vector<std::vector<float>> conv, state; };
  std::vector<std::vector<Snap>> snaps;
  std::function<void(int, const float*, int)>* layer_hook = nullptr;
  std::function<void(int)>* layer_boundary = nullptr;
  std::function<void(int, float*, int)>* layer_inject = nullptr;
  ~Impl();

  void load();
  void alloc_snaps(CpuSeq* const* seqs, const int* M, int S);
  // forward pieces
  void forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, float* logits, bool all_logits);
  void set_rope(const std::vector<RowInfo>& rows, bool prefill, std::vector<int32_t>& coords);
  void layer(int l, const LayerW& L, bool mtp_layer, const std::vector<RowInfo>& rows, const int32_t* coords, float* h);
  void lin_layer(const LinW& w, int li, const std::vector<RowInfo>& rows, const float* x, float* out);
  void attn_layer(const AttW& w, int ai, bool mtp_layer, const std::vector<RowInfo>& rows, const int32_t* coords, const float* x, float* out,
                  bool kv_only = false);
  void mlp(const MlpW& w, int M, const float* x, float* out);
  void head(const float* h, int M, const std::vector<float>& norm_w, float* logits);
  void rope_row(float* v, int n_heads, int head_dim, const int32_t* coord) const;
  void embed_rows(const int32_t* ids, int n, float* out) const;
  // MTP (q27_cpu_mtp.cpp)
  bool mtp_post = false;           // HIVE_Q27_MTP_INPUT=post: the MTP hidden input is the normed stream (A/B of the input definition)
  void mtp_input(const float* emb, const float* h_rows, int n, float* h, const std::vector<float>* prenorm = nullptr);
  void mtp_kv(CpuSeq& s, int64_t pos0, int n, const float* h_rows, const int32_t* next_ids);
  void mtp_after_forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill);
  int mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  void ensure_vision();
  void encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out);
};

// linear: Y = bf16(X·Wᵀ), Y [M, ldy]
void lin(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy, bool round = true);
inline void lin(const Mat& W, const float* X, int M, int ldx, float* Y, int ldy, bool round = true) { lin(MatView{&W, 0, W.N}, X, M, ldx, Y, ldy, round); }
// (1 + w) RMS norm over groups of `group` elements, one bf16 rounding
void norm1p(const float* x, int ldx, const float* w, int M, int K, int group, float eps, float* out, int ldo);
void round_rows(float* Y, int M, int N, int ldy);

Mat load_mat(Checkpoint& ck, const std::vector<std::string>& bases);   // one or several stacked linears of one format (BF16 / FP8 per row)
Mat load_nvfp4(Checkpoint& ck, const std::string& base);
std::vector<float> load_vec(Checkpoint& ck, const std::string& name);

}  // namespace hive::q27::cpu
