// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Internal declarations of the Qwen3.8-Flash-Next CPU engine (hive/qwen/qwen_cpu.h) shared by its source files.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hive/cpu/cpu_backend.h"
#include "hive/qwen/ple.h"
#include "hive/qwen/qwen_cpu.h"
#include "hive/safetensors.h"
#include "qwen_cpu_simd.h"

namespace hive::qwen::cpu {

using hive::cpu::Mat;
using hive::cpu::MatJob;
using hive::cpu::MatView;
using hive::cpu::bf16r;

// sigmoid / SiLU share the engine's exp (qwen_cpu_simd.h) so scalar and AVX2 paths agree bit for bit
inline float sigm(float x) { return simd::sigm_s(x); }
inline float siluf(float x) { return simd::silu_s(x); }

// Parallel loop over [0, n) in blocks of `grain` (serial when the work is small).
void par_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f);

struct HcW { Mat down_inj, up; std::vector<float> norm; bool inject = false; };
struct LinW { Mat in, out; std::vector<float> conv_w, A_log, dt_bias, norm; };
struct AttW { Mat in, out; std::vector<float> q_norm, k_norm, iq_norm, ik_norm; };
// MoE of one layer. front = [router 512 | shared gate logit 1 | shared gate 640 | shared up 640] rows; routed experts stacked so each
//   expert's work is split over the NUMA halves: gu = [gate rows of every expert | up rows of every expert], d = [rows 0..H/2 of every expert's
//   down | rows H/2..H of every expert's down]. NVFP4 global scales (one per expert and projection) are applied to the outputs.
struct MoeW {
  Mat front, sh_down, gu, d;
  std::vector<float> g_gate, g_up, g_down;  // NVFP4 global scale per expert (1 for FP8 experts)
  bool fp8 = false;
};
struct PleW { Mat key, value; std::vector<float> norm_key, norm_query, norm_conv, conv_w; };
struct LayerW { bool attn = false, ple = false; HcW ha, hm; LinW lin; AttW att; MoeW moe; PleW plew; };
struct MtpW { Mat fc_emb, fc_hidden; std::vector<float> pre_emb_norm, pre_hidden_norm; LayerW layer; HcW mixer; };

// vision encoder weights (qwen_cpu_vision.cpp)
struct VisionW {
  struct Block { Mat qkv, proj, fc1, fc2; std::vector<float> n1w, n1b, n2w, n2b, qkv_b, proj_b, fc1_b, fc2_b; };
  Mat patch, m1, m2;
  std::vector<float> patch_b, pos, mn_w, mn_b, m1_b, m2_b;  // pos [side², dim] (bf16 values)
  std::vector<Block> blocks;
  int dim = 1152, heads = 16, inter = 4304, out = 2560, side = 48;
  float eps = 1e-6f;
};

struct RowInfo { CpuSeq* s; int64_t pos; };

struct QwenCpuEngine::Impl {
  QwenConfig c;
  std::unique_ptr<Checkpoint> ck;
  std::vector<LayerW> layers;
  std::unique_ptr<MtpW> mtp;
  HcW mixer;
  Mat lm_head;
  std::vector<uint16_t> embed;      // [vocab, hidden] bf16
  int ple_layer = -1;
  std::unique_ptr<PleTable> ple;
  PleHash ple_hash;
  float ple_scale = 1.f;
  std::vector<double> rope_inv;     // [rot/2]
  std::vector<int> rope_axis;       // [rot/2]
  int max_chunk = 2048, mtp_k = 0;
  std::vector<EmbedOverride> emb_over;
  int64_t emb_base = 0;
  CpuStats st;
  std::unique_ptr<VisionW> vision;
  std::string ckpt_dir;
  int depth = 0;                    // nesting of forwards (a decode inside a prefill's layer boundary)
  std::vector<std::vector<float>> h_stack;  // hidden stream rows per nesting depth
  // verify
  bool verify = false;
  struct VPart { CpuSeq* s; int64_t pos0; int M, r0; };
  std::vector<VPart> vparts;
  std::vector<int32_t> verify_ids;
  std::vector<float> fh;            // final hidden streams of the last forward [rows, 10240]
  // snapshots per verify row r < M−1 of part p: [p][r] → per linear layer conv + state, PLE state
  struct Snap { std::vector<std::vector<float>> conv, state; std::vector<float> ple; };
  std::vector<std::vector<Snap>> snaps;
  std::function<void(int, const float*, int)>* layer_hook = nullptr;
  std::function<void(int)>* layer_boundary = nullptr;
  std::function<void(int, float*, int)>* layer_inject = nullptr;
  int legacy = 0;                   // A/B measurement, previous schedule per bit (identical results): 1 GEMM dispatch, 2 MoE per-row-count
                                    //   dispatch, 4 serial loops, 8 GDN two-pass one task per head, 16 scalar QSA scoring
  void ppar(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f);  // par_for, serial under legacy bit 4
  std::vector<std::pair<Mat*, Mat>> i8_alt;   // HIVE_QWEN_CPU_DENSE_I8_AB: (dense matrix, its I8 twin), swapped by set_dense_i8
  bool i8_on = false;
  void build_i8_alt(const std::string& cats);
  int cur_layer = -1;               // layer of the running moe_layer (−1 = MTP) for the routing dump
  FILE* route_dump = nullptr;       // HIVE_QWEN_CPU_ROUTE_DUMP: "layer rows" then one line of expert ids per row, every MoE call
  ~Impl();

  // load (qwen_cpu_engine.cpp)
  void load(bool load_mtp);
  // forward pieces
  void forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, float* logits, bool all_logits);
  void run_range(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, const std::vector<int32_t>& coords, float* h, int l0);
  int first_layer = 0;              // hybrid placement: this engine runs layers [first_layer, n_layers)
  bool with_head = true;            //   and, without the head, no embedding / final mixer / lm_head
  void set_rope(const std::vector<RowInfo>& rows, bool prefill, std::vector<int32_t>& coords);
  void hc_mix(const HcW& w, const float* h, int M, float* x, std::vector<float>& hn, std::vector<float>& dl);
  void hc_combine(const HcW& w, float* h, const float* y, int M, const std::vector<float>& dl);
  void ple_block(const std::vector<RowInfo>& rows, const int32_t* ids, float* h, bool prefill);
  void lin_layer(const LinW& w, int li, const std::vector<RowInfo>& rows, bool prefill, const float* x, float* out);
  void attn_layer(const AttW& w, int ai, bool mtp_layer, const std::vector<RowInfo>& rows, const std::vector<int32_t>& coords, const float* x,
                  float* out, bool kv_only = false);
  void moe_layer(const MoeW& w, int M, const float* x, float* out);
  void head(const float* h, int M, float* logits);
  void rope_row(float* v, int n_heads, int head_dim, const int32_t* coord) const;
  // MTP (qwen_cpu_mtp.cpp)
  void mtp_input(const float* emb, const float* h_rows, int n, float* hc);
  void mtp_kv(CpuSeq& s, int64_t pos0, int n, const float* h_rows, const int32_t* next_ids);
  void mtp_after_forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill);
  int mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  void ensure_vision();
  void encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out);
};

// linear: Y = bf16(X·Wᵀ) (the reference's bf16 GEMM output), Y [M, ldy]
void lin(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy, bool round = true);
inline void lin(const Mat& W, const float* X, int M, int ldx, float* Y, int ldy, bool round = true) { lin(MatView{&W, 0, W.N}, X, M, ldx, Y, ldy, round); }
// (1 + w) RMS norm over groups of `group` elements, one bf16 rounding (rmsnorm1p of the oracle)
void norm1p(const float* x, int ldx, const float* w, int M, int K, int group, float eps, float* out, int ldo);

Mat load_bf16_mat(Checkpoint& ck, const std::vector<std::string>& names, const char* cat = nullptr);  // cat: opt-in I8 category
std::vector<float> load_vec(Checkpoint& ck, const std::string& name);

}  // namespace hive::qwen::cpu
