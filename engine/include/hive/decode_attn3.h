// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Decode-side (M <= 8) attention: new versions of the four remaining kernels (decode_attn3.cu). One switch each
// (env_on — unset / "" / "0" = off, read once):
//   HIVE_DECODE_ROUTER3  router (replaces attn2_router) — block = expert. For M=1, xn is also staged in smem via cp.async
//                        (the row dot-product chain does not wait for L2 round trips). For M >= 2, xn uses a 3-stage
//                        register ring. Tail: softplus/sqrt over 256 threads; top-k in warp registers (no smem round
//                        trip, no lane-0 serialization).
//   HIVE_DECODE_SPARSE3  sparse attention (replaces D2 dec_attn_kernel) — block = (row, G heads, chunk). Each 16-key tile is
//                        dequantized (fp4 -> bf16) once per block into smem and shared by the G heads (previously it
//                        was re-dequantized per head and per pair). The index chunk is loaded to smem once (removes the
//                        key-address dependency chain); tiles are prefetched one ahead.
//   HIVE_DECODE_QKV3     q_a||kv (replaces D2 dec_qkv_a_kernel) — the 3 weight blocks are loaded into registers *before*
//                        quantization; input quantization uses 256 threads; the row table, window indices and pos/ring
//                        pointers (mapped pinned) are read at the *head* of the kernel; the tail (q rmsnorm, kv rmsnorm
//                        -> RoPE -> fp8 round trip -> ring) runs in parallel with one warp per row.
//   HIVE_DECODE_HCMIX3   hc_mix_pre_norm (HIVE_HC_FUSE2 path) — same grid, same reduction; each thread issues 40 loads
//                        at once (previously ~4 at a time -> 20 round trips become 2).
//   HIVE_DECODE_SPARSE3_G  (number, optional) heads per block for SPARSE3: 1/2/4/8 — unset / 0 = auto (smallest G with
//                        M*H*S/G <= number of SMs).
// Numeric contract: all four are **bit-identical** to the current production kernels (attn2_router, dec_attn_kernel,
//   dec_qkv_a_kernel, hc_mix_pre_norm_kernel) — every output element uses the same expression, the same fmaf chain and
//   the same reduction tree (xor 16..1, warp partial sums summed in order); only the thread/block mapping, load timing
//   and smem placement change.
//   The router's only divergent case (fewer candidates than k — NaN activations) matches the A1 version (id -1, weight 0).
//   Shapes outside a version's range fall back to the previous function inside the call (absorbed — never rejected).
//   test_decode_attn3 compares and profiles them on the GPU.
#pragma once
#include <cstdint>

#include "hive/common.h"

namespace hive::k {

struct DecAttnArgs;

bool decode_router3_on();   // env_on("HIVE_DECODE_ROUTER3")
bool decode_sparse3_on();   // env_on("HIVE_DECODE_SPARSE3")
bool decode_qkv3_on();      // env_on("HIVE_DECODE_QKV3")
bool decode_hcmix3_on();    // env_on("HIVE_DECODE_HCMIX3")
int decode_sparse3_g();     // HIVE_DECODE_SPARSE3_G (1/2/4/8; anything else = 0, auto)

// Same signature and bit-identical output as attn2_router. Out-of-range shapes go to attn2_router.
void attn3_router(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                  float route_scale, float* scores, int32_t* ids, float* w, int* counter, cudaStream_t st);
// Cache prior (HIVE_CACHE_PRIOR=λ, decode): residency-aware routing after Skliar et al., TMLR 2025 (arXiv 2412.00099, eq. 9-10; idea only) —
//   the selection scores of VRAM-resident experts (mask[e] != 0, E bytes readable by the device) and of the row's top_j get + λ·Δavg
//   (Δavg = running average in *range_avg of max − min of the selection scores); weights stay from the unmodified scores. mask null / λ ≤ 0 = off.
//   Applied only when this version handles the shape (the ATTN2 fallback ignores it).
// Expert deferral (HIVE_DECODE_DEFER): h[m, c, :] = bf16(h[m, c, :] + post[m, c] · Σ_{r: rows[r] = m} y[r, :]) for the n deferred job rows y (fp32,
//   mapped host memory read through UVA) of M sequence rows; copy_f32 = an SM copy for small device buffers (does not queue on the copy engine)
void hc_inject_rows(bf16* h, const float* post, const float* y, const int32_t* rows, int n, int M, int hc, int dim, cudaStream_t st);
void copy_f32(float* dst, const float* src, size_t n, cudaStream_t st);
struct CachePriorArgs { const uint8_t* mask = nullptr; float lambda = 0.f; float* range_avg = nullptr; int top_j = 2; };
void attn3_router_cp(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                     float route_scale, float* scores, int32_t* ids, float* w, int* counter, const CachePriorArgs& cp, cudaStream_t st);
// For tests (cost breakdown): phase 0 = full; 1 = load gate_w rows (+xn) only; 2 = load + dot product (no tail — scores only). mode: -1 auto, 0 register xn, 1 smem xn
void attn3_router_phase(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                        float route_scale, float* scores, int32_t* ids, float* w, int* counter, int phase, int mode, cudaStream_t st);
// Same signature and bit-identical output as hc_mix_pre_norm (model_kernels.cu). Out-of-range shapes go to hc_mix_pre_norm.
void hc_mix_pre_norm3(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                      const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, cudaStream_t st);
// Stage replacements for the D2 front end (dec_attention_front). false = shape not handled by this version (caller launches the previous kernel unchanged).
bool attn3_qkv_a(const DecAttnArgs& a, bool xf_on, cudaStream_t st);
bool attn3_sparse(const DecAttnArgs& a, int G, cudaStream_t st);  // G = 0 -> auto
int attn3_sparse_auto_g(int M, int H, int S);                       // auto rule (for test output)

}  // namespace hive::k
