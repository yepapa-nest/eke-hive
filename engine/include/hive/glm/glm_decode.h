// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash decode-path kernels (M ≤ 8 activation rows): fused hyper-connection pre/post, multi-segment BF16 GEMV, router + top-k,
//   dense NVFP4 MLP (layers 0-2) and the FP8 shared expert, each in one or two launches. Every launch uses programmatic dependent launch
//   (PDL): weights that do not depend on the previous kernel are prefetched before griddepcontrol.wait, activations are read after it.
//   Without a PDL-capable predecessor the wait returns immediately, so the functions are safe on any stream and in graph capture.
// Determinism: no atomics in any reduction; every output element has a fixed summation order independent of the grid size. The hc pre
//   kernel uses device counters (sync) for completion only — never for summing values.
// Validation and timings: engine/tests/test_glm_decode.cu.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "hive/glm/fp8b.h"
#include "hive/glm/nvfp4.h"

namespace hive::glm {

// Device synchronization words for the last-block kernel (hc pre): allocate kGlmDecodeSyncInts ints per stream, zero them ONCE (cudaMemset); the
//   kernels leave them at zero again. Do not share one sync buffer between kernels that may run concurrently on different streams.
constexpr int kGlmDecodeSyncInts = 8;

// ---- 1. hc pre-stage (+ optional fused hc_post of the previous sub-layer) -----------------------------------------------------------------
// Computes exactly what glm_engine.cpp hc_pre_norm() computes for rows ≤ 8 (k::hc_mix → k::hc_split_sinkhorn → k::hc_pre_norm):
//   mixes[m, j] = Σ_i f32(h[m, i])·fn[j, i] (j < 24, i < 4·dim) · rsq[m] = rsqrt(mean_i h[m, i]² + rms_eps)
//   pre/post/comb = Sinkhorn(mixes, rsq, scale, base, iters, hc_eps)        (fp32 [M,4] [M,4] [M,4,4])
//   x[m, d] = bf16(Σ_c pre[m, c]·h[m, c, d]) · xn = bf16(norm_w · x·rsqrt(mean x² + rms_eps))   (bf16 [M, dim])
//   h: bf16 [M, 4, dim] (row stride 4·dim), fn fp32 [24, 4·dim], scale fp32 [3], base fp32 [24], norm_w bf16 [dim]. dim = 4096, M ≤ 8.
//   post_x != nullptr: first applies hc_post(post_x, post, comb) to h IN PLACE (post_x bf16 [M, dim]; post/comb hold the PREVIOUS
//   stage's values on entry and the new ones on exit) — i.e. hc_post + hc_pre_norm of the next stage in one launch.
// Numeric contract: BIT-IDENTICAL to the three-kernel path (and, with post_x, to k::hc_post followed by it): same thread → element
//   mapping and fp32 operation order in every dot product and reduction, the same 16-lane Sinkhorn device code (hc_decode_fused.cuh).
// Structure: one launch of M·25 mix blocks + M norm blocks. The last block to finish solves Sinkhorn and raises a flag; the norm blocks
//   (preloaded h) spin on that flag. All M·26 ≤ 208 blocks are co-resident on the target GPU; the mix blocks never wait.
//   mixes [M,24] and rsq [M] are written as well (scratch the caller may read). sync: kGlmDecodeSyncInts ints (see above).
void glm_hc_pre_decode(__nv_bfloat16* h, int M, int dim, const float* fn, const float* scale, const float* base, const __nv_bfloat16* norm_w,
                       float rms_eps, float hc_eps, int iters, float* mixes, float* rsq, float* pre, float* post, float* comb, __nv_bfloat16* x,
                       __nv_bfloat16* xn, int* sync, cudaStream_t st, const __nv_bfloat16* post_x = nullptr);

// ---- 2. hc post (standalone) --------------------------------------------------------------------------------------------------------------
// h[m, c, d] = bf16(post[m, c]·x[m, d] + Σ_j comb[m, j, c]·h[m, j, d]) in place — bit-identical to k::hc_post (same expression per
//   element), 8 consecutive d per thread with 16-byte loads/stores. x bf16 [M, dim], h bf16 [M, 4, dim]; dim % 8 == 0, M ≤ 8.
//   Use this only where no hc pre-stage follows (the last layer); elsewhere pass post_x to glm_hc_pre_decode.
void glm_hc_post_decode(const __nv_bfloat16* x, const float* post, const float* comb, int M, int dim, __nv_bfloat16* h, cudaStream_t st);

// ---- 3. BF16 GEMV over a segment table ------------------------------------------------------------------------------------------------------
// Segment s: y_s[m, n] = bf16( Σ_k x_s[m, k]·W_s[n, k] ) for m < M, n < N_s, fp32 accumulation, round-to-nearest-even to bf16 once.
//   W_s bf16 [N_s, K_s] row-major (the stored [out, in] layout); x_s rows at stride ldx_s, y_s rows at stride ldy_s (e.g. q|k|v interleaved
//   into one [M, 3W] buffer = three segments with ldy 3W and y offset 0, W, 2W). Segments may have different K and inputs (f_b reads
//   f_a's output — but not in the same launch). One launch for all segments: bf16 tensor cores (mma.m16n8k16, fp32 accumulate), block =
//   16-row tiles with K split over up to 8 warps, parts added in fixed order. Requirements: K_s % 32 == 0, 16-byte aligned W and x rows
//   (ldx % 8 == 0), nseg ≤ kGemvMaxSeg, 1 ≤ M ≤ 8.
// Numeric contract: an fp32 sum in a different order than cuBLAS, rounded once. Measured (test): every output within the fp32-order error
//   budget ulp_bf16 + K·2^-24·Σ|x·w| of the fp64 result (max ratio 0.34) and of cuBLAS's fp32-output GEMM. NOTE: the bf16-OUTPUT cuBLAS GEMM
//   the engine calls today is not a single rounding of an fp32 sum at these shapes (test: up to 444 bf16 steps / 17 % of outputs outside
//   that budget at M = 8), so it is not used as the reference. Deterministic and independent of the grid size.
constexpr int kGemvMaxSeg = 8;
struct GemvSeg {
  const __nv_bfloat16* W = nullptr;
  const __nv_bfloat16* x = nullptr;
  __nv_bfloat16* y = nullptr;
  int N = 0, K = 0, ldx = 0, ldy = 0;
};
void glm_gemv_bf16_multi(const GemvSeg* segs, int nseg, int M, cudaStream_t st);

// ---- 4. Router -----------------------------------------------------------------------------------------------------------------------------
// logits[m, e] = Σ_k x[m, k]·gate_w[e, k] (fp32, bf16 inputs, tensor cores), then exactly glm_kernels.cu router_topk: s = 1/(1+expf(-logit)),
//   choose k times the maximum of s + bias (strict '>' scan → on equal values the SMALLER expert id wins; NaN never chosen),
//   weights = s[id] / (Σ chosen s + 1e-20) · scale (sum in selection order). x bf16 [M, K] (stride ldx), gate_w bf16 [E, K], bias fp32 [E];
//   E ≤ 512, k ≤ 16, K % 512 == 0, M ≤ 8. Two launches: E/16 · K/512 single-warp blocks write fp32 partial logits; a PDL tail kernel adds
//   them in fixed order and runs the top-k (warp m = row m).
//   logits: fp32 scratch of glm_router_logits_floats(M, E, K) floats; on return its first M·E floats hold the logits [M, E].
//   ids int32 [M, k], w fp32 [M, k]; ids_host / w_host (optional): device-accessible pointers into cudaHostAllocMapped memory (with UVA the
//   host pointer itself) — the kernel stores the same values there, so the caller only needs a stream sync instead of two D2H copies.
// Numeric contract: the logits differ from the cuBLAS fp32-output GEMM only by fp32 summation order; given the same logits the top-k is
//   bit-identical to router_topk (tested by feeding this kernel's logits to router_topk). Rows with fewer than k selectable experts (only
//   with NaN logits) get id -1 / weight 0 (router_topk reads out of bounds there).
size_t glm_router_logits_floats(int M, int E, int K);
// Cache prior (HIVE_CACHE_PRIOR=λ, decode only): residency-aware routing after Skliar et al., "Mixture of Cache-Conditional Experts for
//   Efficient Mobile Device Inference" (TMLR 2025, arXiv 2412.00099, eq. 9-10; idea only, no code): the selection scores of experts that are in
//   the VRAM cache (mask[e] != 0, E bytes readable by the device) and of the row's top_j experts get + λ·Δavg, Δavg being this layer's running
//   average (in *range_avg, device float, 0 = not started) of max − min of the selection scores. The weights of the chosen experts still come
//   from the unmodified scores. mask == nullptr or λ ≤ 0 = off (the default top-k, bit-identical).
struct GlmCachePrior { const uint8_t* mask = nullptr; float lambda = 0.f; float* range_avg = nullptr; int top_j = 2; };
void glm_router_decode(const __nv_bfloat16* x, int ldx, int M, const __nv_bfloat16* gate_w, const float* bias, int E, int K, int k, float scale,
                       float* logits, int32_t* ids, float* w, int32_t* ids_host, float* w_host, cudaStream_t st, const GlmCachePrior* cache_prior = nullptr);

// ---- 5. Dense NVFP4 MLP (layers 0-2) ---------------------------------------------------------------------------------------------------------
// out[m] = bf16( down · bf16(swiglu(bf16(gate·x[m]·g_gate), bf16(up·x[m]·g_up), limit)) · g_down ) — the composition
//   nvfp4_gemv → f32_to_bf16 → swiglu_rows → nvfp4_gemv → f32_to_bf16 of glm_engine.cpp mlp_layer. gate/up [I, H], down [H, I] NVFP4;
//   x bf16 [M, H] (stride ldx), y_ws bf16 [M, I] scratch, out bf16 [M, H] (stride ldo). H % 1024 == 0, I % 1024 == 0, M ≤ 8.
// Two launches (gate/up+SwiGLU: block = 8 rows of I; down with PDL: block = 16 rows of H, 16 warps splitting K). Dot products on the
//   tensor cores as in glm_moe.cu (e2m1·e4m3 decoded exactly to f16, bf16 activations converted to f16, fp32 accumulation, split-K parts
//   added in fixed order); an item whose f16 operands overflow (|activation| ≥ 65520) is recomputed exactly on the CUDA cores.
//   Numeric contract: fp32 summation-order differences, plus the f16 rounding of activations below 2^-14 in magnitude (≤ 2^-25 absolute per
//   element). Test: y and out within that error budget of an fp64 reference (max ratio 0.37 / 0.21; the old path scores the same).
//   (The old composition aborts for M > 4: nvfp4_gemv stages M·12288 bf16 of the down input in ≤ 96 KB shared memory.)
size_t glm_dense_nvfp4_ws_bytes(int M, int I);
void glm_dense_nvfp4_decode(const Nvfp4Mat& gate, const Nvfp4Mat& up, const Nvfp4Mat& down, const __nv_bfloat16* x, int ldx, int M, float limit,
                            __nv_bfloat16* y_ws, __nv_bfloat16* out, int ldo, cudaStream_t st);

// ---- 6. Shared expert, Z.ai FP8 128×128 block scales ---------------------------------------------------------------------------------------
// out[m] = fp8b_gemv(down, swiglu_rows(fp8b_gemv(gate, x) | fp8b_gemv(up, x), limit)) — glm_engine.cpp mlp_layer's shared expert via lin()
//   for rows ≤ 8. gate/up [I, H], down [H, I]; x bf16 [M, H] (stride ldx, % 8), y_ws bf16 [M, I], out bf16 [M, H] (stride ldo).
//   Instantiated for H 4096 · I 2048 (the GLM shared expert), M ≤ 8. Two launches (warp = gate+up row with SwiGLU; warp = down row, PDL).
// Numeric contract: BIT-IDENTICAL to the fp8b_gemv / swiglu_rows composition (same lane → 16-column chunk mapping, same fma order, same
//   block-scale application and butterfly, same SwiGLU expression).
void glm_shared_fp8_decode(const Fp8BMat& gate, const Fp8BMat& up, const Fp8BMat& down, const __nv_bfloat16* x, int ldx, int M, float limit,
                           __nv_bfloat16* y_ws, __nv_bfloat16* out, int ldo, cudaStream_t st);

}  // namespace hive::glm
