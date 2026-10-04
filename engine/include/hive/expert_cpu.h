// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU expert kernels — compute routed experts that are not in VRAM in place from pinned RAM (Zen2 AVX2+FMA+F16C).
// Numeric contract (reference MoE.Expert.forward + fp4_gemm):
//   gate = bf16(w1·a)  up = bf16(w3·a)  (a = e4m3 activations + e8m0/32 scales; products accumulate per block in fp32 × sa × sw)
//   up ∈ [-limit, limit] · gate ≤ limit → y = silu(gate)·up·route_w (fp32) → bf16 → act_quant(fp8/32) → out = bf16(w2·ŷ)
//   Returns those bf16 values as fp32 (the caller accumulates in fp32).
// The expert format lives in the descriptor — only e2m1/32 today, but int3/int2 (requantized cold experts) would be selected here later.
#pragma once
#include <cstdint>

namespace hive::cpu {

enum class ExpertFormat : uint8_t { E2M1_B32 = 0, NVFP4_B16 = 1 };

struct ExpertDesc {
  ExpertFormat fmt = ExpertFormat::E2M1_B32;
  int dim = 5120;        // K of w1/w3, N of w2
  int inter = 2304;      // N of w1/w3, K of w2
  const uint8_t* w1;     // [inter, dim/2] nibbles
  const uint8_t* s1;     // [inter, dim/32] e8m0
  const uint8_t* w2;     // [dim, inter/2]
  const uint8_t* s2;     // [dim, inter/32]
  const uint8_t* w3;     // [inter, dim/2]
  const uint8_t* s3;     // [inter, dim/32]
};

// Expert forward for one activation (e4m3 [dim] + e8m0 [dim/32]). out[dim] = result including route_w (fp32, overwritten).
// tmp is a caller-provided thread-local workspace (floats ≥ 3*inter + dim + (inter+dim)/32 + 64).
void expert_forward(const ExpertDesc& e, const uint8_t* a_q, const uint8_t* a_s, float route_w, float swiglu_limit,
                    float* out, float* tmp);

// Multi-row: unpack each weight row once for R activations (a_f[r], sa[r]) and produce R results y[r][n] (batched decode where R
// tokens hit the same expert).
void gemv_e2m1_rows_multi(const uint8_t* W, const uint8_t* sw, int N, int K, const float* const* a_f32, const float* const* sa_f32, int R,
                          int n0, int n1, float* const* y);
// Internal primitive: y[n] = Σ_b dot32(a_f32 block b, W[n] block b) · sa[b] · sw[n,b]   (W = e2m1 nibbles [N, K/2], sw = [N, K/32])
//   a_f32: activation values (unscaled) fp32 [K]; sa_f32: block scales fp32 [K/32]. Row range [n0, n1).
void gemv_e2m1_rows(const uint8_t* W, const uint8_t* sw, int N, int K, const float* a_f32, const float* sa_f32, int n0,
                    int n1, float* y);

// HIVE_CPU_GEMV2: the two functions above are switch-selected entry points — off (unset · "" · "0") = _ref (baseline kernel) ·
//   on = _v2 (faster version with the same accumulation order, bit-identical output).
//   Calling _ref/_v2 directly is for tests and benchmarks (engine/tests/bench_expert_cpu.cpp). gemv2_enabled() = the switch value the
//   process read.
void gemv_e2m1_rows_ref(const uint8_t* W, const uint8_t* sw, int N, int K, const float* a_f32, const float* sa_f32, int n0, int n1, float* y);
void gemv_e2m1_rows_v2(const uint8_t* W, const uint8_t* sw, int N, int K, const float* a_f32, const float* sa_f32, int n0, int n1, float* y);
void gemv_e2m1_rows_multi_ref(const uint8_t* W, const uint8_t* sw, int N, int K, const float* const* a_f32, const float* const* sa_f32, int R,
                              int n0, int n1, float* const* y);
void gemv_e2m1_rows_multi_v2(const uint8_t* W, const uint8_t* sw, int N, int K, const float* const* a_f32, const float* const* sa_f32, int R,
                             int n0, int n1, float* const* y);
bool gemv2_enabled();
// HIVE_CPU_MULTIROW2: K-tiled version for multi-row (R ≥ 2) — output bits = _v2 = _ref (same accumulation order; only the loop order
//   changes to row groups × K tiles). R = 1 goes to _v2.
//   The gemv_e2m1_rows_multi entry point uses it when multirow2_enabled() (unset · "" · "0" = off) and R ≥ multirow2_min_rows()
//   (default 5 when on · values 2..8).
void gemv_e2m1_rows_multi_v3(const uint8_t* W, const uint8_t* sw, int N, int K, const float* const* a_f32, const float* const* sa_f32, int R,
                             int n0, int n1, float* const* y);
bool multirow2_enabled();
int multirow2_min_rows();  // 0 = off
// HIVE_CPU_UNPACK2: unpack one activation row — vector version bit-identical to a_f[d] = e4m3_to_f32(q[d]) · a_s[b] = e8m0_to_f32(s[b])
//   (b < dim/32) / the scalar formula (_ref).
//   runtime.cpp moe_decode_experts uses it for rows sent to the CPU when unpack2_enabled() (unset · "" · "0" = off).
void unpack_act_row(const uint8_t* q, const uint8_t* s, int dim, float* a_f, float* a_s);
void unpack_act_row_ref(const uint8_t* q, const uint8_t* s, int dim, float* a_f, float* a_s);
bool unpack2_enabled();

// ---- NVFP4 (GLM-5.3-Flash checkpoints quantized with ModelOpt) ----------------------------------------------------------------------------------
// Weights: e2m1 nibbles [N, K/2] (low nibble = even column, the same code table as E2M1_B32), one e4m3 scale per 16 columns [N, K/16]
//   and one fp32 global scale per matrix: w = e2m1 · e4m3(scale) · global. Activations are bf16 values given as fp32 (W4A16 — the
//   activations are not quantized: a public run of W4A4 on this model changed 5.9 % of repeated greedy answers, W4A16 none).
// y[n] = global · Σ_blocks16 (Σ_i a[i]·w[n,i]) · scale[n,block]   — fp32 accumulation, row range [n0, n1).
void gemv_nvfp4_rows(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* a_f32, int n0, int n1, float* y);
// R activations (R ≤ 8) per weight row: the row is unpacked once (batched decode rows that hit the same expert).
void gemv_nvfp4_rows_multi(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* const* a_f32, int R, int n0, int n1,
                           float* const* y);
// Scalar reference of gemv_nvfp4_rows (tests).
void gemv_nvfp4_rows_scalar(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* a_f32, int n0, int n1, float* y);

struct ExpertDescNvfp4 {
  int dim = 4096, inter = 2048;
  const uint8_t *w1, *s1, *w3, *s3, *w2, *s2;  // gate [inter,dim/2]+[inter,dim/16] · up (same) · down [dim,inter/2]+[dim,inter/16]
  float g1 = 1.f, g3 = 1.f, g2 = 1.f;          // global scales (weight_scale_2)
};
// GLM expert for one activation a[dim] (bf16 values as fp32): gate = bf16(w1·a) · up = bf16(w3·a) → clamp (up ∈ [-limit, limit],
//   gate ≤ limit) → y = bf16(silu(gate)·up) → out = bf16(w2·y) · route_w (fp32). tmp ≥ 3·inter floats.
// W4A8 variant (HIVE_GLM_CPU_Q8): activations as int8 per 32-column block in the nvfp4 nibble order (quantize_q8_nvfp4), integer dot.
void quantize_q8_nvfp4(const float* a, int K, int8_t* q, float* s);  // q [K], s [K/32]
void gemv_nvfp4_q8_rows_multi(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const int8_t* const* aq, const float* const* as,
                              int R, int n0, int n1, float* const* y);
void expert_forward_nvfp4(const ExpertDescNvfp4& e, const float* a, float route_w, float swiglu_limit, float* out, float* tmp);

}  // namespace hive::cpu
