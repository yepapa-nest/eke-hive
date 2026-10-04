// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_GEMV2 — bandwidth-oriented versions of the decode (M <= 8) dense weight GEMVs, router and hc mix (gemv_decode.cu).
//   Covered (share of the ~178 MB of weights per layer that these read): wq_a/wq_b (fused_q_proj), wo_a (gemv_bf16_fp8_grouped),
//   wo_b (fused_gemv_quantin), shared expert w1||w3 and w2 (fused_shared_experts), router gate_w (fused_router), hc_attn_fn/hc_ffn_fn (hc_mix_pre_norm).
//   Numeric contract: **bit-identical** to the original functions (test_decode_gemv). Every output element uses the same lane->block
//   assignment, the same fmaf chain, the same xor reduction and the same epilogue formula.
//   Only the loading changes — a warp loads all weights of its columns (the lane's whole share) into registers at once **before** input
//   quantization (16 B vectors, no L1 allocation); the router loads the lane's share of gate_w in groups of 32 and does top-k with a warp
//   argmax (larger value wins, ties -> smaller index — the same choice as the sequential lane-0 scan); the hc mix loads the thread's share in
//   groups of 16 (in-thread k order and warp/block reductions unchanged).
//   Signatures match the original functions — call sites pick with `(k::decode_gemv2_on() ? k::gemv2_x : k::x)(...)`. If a shape is outside
//   this version's templates, the function calls the original one internally (absorbed — never rejected).
// HIVE_DECODE_PDL (env_on, default off): launches the kernels above with cudaLaunchKernelEx + programmaticStreamSerialization. After
//   preloading weights a kernel passes griddepcontrol.wait (previous kernel complete, memory visible) before reading activations or writing
//   outputs — only the read-only weight load overlaps the previous kernel's tail, so values and ordering do not depend on PDL. A kernel only
//   finishes after every CTA has passed the wait (keeps stream order for subsequent ordinary launches). No effect when GEMV2 is off.
#pragma once
#include <cstdint>

#include "hive/common.h"

namespace hive::k {

bool decode_gemv2_on();  // env_on("HIVE_DECODE_GEMV2") — read once
bool decode_pdl_on();    // env_on("HIVE_DECODE_PDL") — read once (tests override with gemv2_force_pdl)
void gemv2_force_pdl(int mode);  // for tests: -1 = environment, 0 = off, 1 = on

// Same signatures and same (bit-identical) outputs as the original fused.cu / gemm.cu / model_kernels.cu functions
void gemv2_q_proj(const bf16* xn, int K, const uint8_t* wqa, const uint8_t* sqa, int Nqa, const bf16* q_norm, float eps, const uint8_t* wqb,
                  const uint8_t* sqb, int H, int D, int rd, const float2* freqs, const int32_t* pos, int M, bf16* qr, bf16* qrn, bf16* q, uint8_t* qrq,
                  uint8_t* qrs, int* counter, cudaStream_t st);
void gemv2_quantin(const bf16* A, int K, const uint8_t* B, const uint8_t* sb, int M, int N, bf16* C, cudaStream_t st);
void gemv2_shared_experts(const bf16* xn, int K, const uint8_t* w1, const uint8_t* s1, const uint8_t* w3, const uint8_t* s3, const uint8_t* w2,
                          const uint8_t* s2, int M, int I, float limit, bf16* y, uint8_t* xq, uint8_t* xs, float* acc, cudaStream_t st);
void gemv2_router(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                  float route_scale, float* scores, int32_t* ids, float* w, int* counter, cudaStream_t st);
void gemv2_bf16_fp8_grouped(const bf16* A, int lda, const uint8_t* B, const uint8_t* sb, int M, int G, int R, int sub, bf16* C, cudaStream_t st);
void hc_mix_pre_norm2(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                      const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, cudaStream_t st);
// HIVE_HC_DECODE_FUSED: hc_mix_pre_norm2 + sinkhorn tail (same contract as hc_mix_pre_norm_sk in model_kernels.h; PDL launch rules as above)
void hc_mix_pre_norm2_sk(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                         const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, const float* scale, const float* base, int iters, float hc_eps, float* pre_out,
                         float* post, float* comb, int* cnt, cudaStream_t st);
void hc_mix_pre_norm2_sk_preload();

}  // namespace hive::k
