// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Small Qwen3.8-Flash-Next kernels: (1 + w) RMS norm, the gated-residual hyper-connection pieces, the softmax router, the MoE output
//   combine and a few row utilities. The linear-attention core is hive/qwen/gdn.h, attention + indexer hive/qwen/qsa.h, PLE hive/qwen/ple.h.
// Rounding: every intermediate the reference keeps in bf16 is rounded to bf16 here at the same point (GEMM outputs, the norm output, the
//   mixed hidden); the arithmetic in between is fp32.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "hive/devbuf.h"

namespace hive::qwen {

// A (1 + w) norm weight: bf16, or fp32 when the checkpoint stores it so (NVIDIA FP4-Local keeps the hyper-connection and PLE norms in fp32 —
//   rounding them to bf16 moved the QAD-trained values by up to 2^-9 relative); the kernels compute 1 + w in fp32 either way.
struct NormW {
  DevBuf buf;
  bool f32 = false;
};

// out[m, i] = bf16( x · rsqrt(mean_{group of i}(x²) + eps) · (1 + w[i]) )  — fp32 math, one rounding.
//   x bf16 [M, K] (row stride ldx), w bf16 [K], group | K (group = K for a whole-row norm), out bf16 [M, K] (row stride ldo).
void gemma_rmsnorm(const __nv_bfloat16* x, int ldx, const __nv_bfloat16* w, int M, int K, int group, float eps, __nv_bfloat16* out, int ldo,
                   cudaStream_t st);
void gemma_rmsnorm(const __nv_bfloat16* x, int ldx, const float* w, int M, int K, int group, float eps, __nv_bfloat16* out, int ldo, cudaStream_t st);
void gemma_rmsnorm(const __nv_bfloat16* x, int ldx, const NormW& w, int M, int K, int group, float eps, __nv_bfloat16* out, int ldo, cudaStream_t st);

// ---- Hyper-connections (gated residual) --------------------------------------------------------------------------------------------------
// The engine runs a mix as: hn = gemma_rmsnorm(h, group = hidden) → dl = GEMM(hn, down_inj) [M, rank (+ hc)] → hc_mix_act(dl) → s [M, rank]
//   → u = GEMM(s, up) [M, hc·hidden] → hc_mix_out(u, hn) → x [M, hidden]; and later the combine hc_combine(h, y, dl) in place.
// s[m, j] = bf16( silu( dl[m, j] / hc ) )  for j < rank (dl row stride ld)
void hc_mix_act(const __nv_bfloat16* dl, int ld, int M, int rank, int hc, __nv_bfloat16* s, cudaStream_t st);
// x[m, d] = bf16( (1/hc) · Σ_c bf16( bf16(sigmoid(u[m, c, d])) · hn[m, c, d] ) )  (u, hn bf16 [M, hc·dim])
void hc_mix_out(const __nv_bfloat16* u, const __nv_bfloat16* hn, int M, int hc, int dim, __nv_bfloat16* x, cudaStream_t st);
// h[m, c, d] = bf16( h[m, c, d] + y[m, d] · 2·sigmoid(dl[m, rank + c] / hc) )  in place (h bf16 [M, hc·dim], y bf16 [M, dim] row stride ldy)
void hc_combine(__nv_bfloat16* h, const __nv_bfloat16* y, int ldy, const __nv_bfloat16* dl, int ld, int rank, int M, int hc, int dim, cudaStream_t st);
// hc_combine, then gemma_rmsnorm(h, group = dim) with weight w into hn [M, hc·dim] — one launch, bit-identical to the two
void hc_combine_norm(__nv_bfloat16* h, const __nv_bfloat16* y, int ldy, const __nv_bfloat16* dl, int ld, int rank, int M, int hc, int dim,
                     const __nv_bfloat16* w, float eps, __nv_bfloat16* hn, cudaStream_t st);
void hc_combine_norm(__nv_bfloat16* h, const __nv_bfloat16* y, int ldy, const __nv_bfloat16* dl, int ld, int rank, int M, int hc, int dim,
                     const NormW& w, float eps, __nv_bfloat16* hn, cudaStream_t st);
// h[m, c, :] = x[m, :] for every copy c (embedding → hidden stream). x may be pinned host memory (UVA).
void hc_expand(const __nv_bfloat16* x, int M, int hc, int dim, __nv_bfloat16* h, cudaStream_t st);
// answer-span token penalty (hive/token_mask.h): logits[r, ids[j]] −= pen[r] for every row r with pen[r] != 0 (∞ → −inf; pen: device-readable,
//   e.g. mapped pinned memory written by the host before the launch — so a captured graph reads the current values)
void mask_logits(float* logits, int rows, int V, const float* pen, const int32_t* ids, int n, cudaStream_t st);

// ---- MoE ---------------------------------------------------------------------------------------------------------------------------------
// Router: logits fp32 [M, ld] (columns 0..E-1 = router; the GEMM's fp32 output, rounded to bf16 here like the reference's bf16 linear),
//   p = softmax(logits) (fp32), the k largest (ties: smaller id first), weights p / Σ_k p (norm_topk) or p. ids int32 [M, k], w fp32 [M, k].
//   ids_host / w_host (optional): mapped host copies written by the same kernel (UVA). E ≤ 1024, k ≤ 16.
void router_softmax_topk(const float* logits, int ld, int M, int E, int k, bool renorm, int32_t* ids, float* w, int32_t* ids_host,
                         float* w_host, cudaStream_t st);
// out[m, d] = bf16( bf16(routed[m, d]) + bf16( sigmoid(bf16(gate_logit[m])) · shared[m, d] ) )
//   routed fp32 [M, dim] (sum of the weighted routed experts), shared bf16 [M, dim], gate_logit = logits[m·ld + col] (fp32).
void moe_combine(const float* routed, const __nv_bfloat16* shared, const float* logits, int ld, int col, int M, int dim, __nv_bfloat16* out,
                 cudaStream_t st);

// Device routing (every expert resident): pairs[m·K + k] = {m, rec_tab[li·E + ids[m, k]], w[m, k]} — the input of glm::moe_decode, built on
//   the device so no host round trip is needed. rec_tab: device array [n_moe · E] of record pointers (nullptr = not resident: that pair's
//   weight is set to 0 and its record to the first non-null one of the row — the caller only uses this path when every entry is set).
struct MoePairDev { int row; const uint8_t* rec; float w; };   // layout of glm::MoePair
void build_moe_pairs(const int32_t* ids, const float* w, int M, int K, const uint8_t* const* rec_tab, int li, int E, MoePairDev* pairs,
                     cudaStream_t st);

// h[m, c, d] = bf16(h[m, c, d] + x[m, d]) for every copy c (the MTP input: embedding projection added to each projected copy)
void hc_add_rows(__nv_bfloat16* h, const __nv_bfloat16* x, int M, int hc, int dim, cudaStream_t st);
// dst[r][0..2] = src[r·3 .. r·3+2] — rope coordinates of n rows into each row's sequence buffer (src and dst may be pinned host memory,
//   read through UVA: no copy-engine transfer that could queue behind expert copies)
void write_rope(const int32_t* src, int32_t* const* dst, int n, cudaStream_t st);

// ---- Rows ----------------------------------------------------------------------------------------------------------------------------------
// out[r, :] = table[ids[r], :] for bf16 rows of width dim; table may be pinned host memory (UVA), ids device or mapped host.
void embed_rows(const __nv_bfloat16* table, const int32_t* ids, int M, int dim, __nv_bfloat16* out, cudaStream_t st);
// dst rows (stride ldd) = src rows (stride lds), bf16, n rows of width dim
void copy_rows(const __nv_bfloat16* src, int lds, int n, int dim, __nv_bfloat16* dst, int ldd, cudaStream_t st);

}  // namespace hive::qwen
