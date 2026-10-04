// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Small GLM-specific kernels (everything else is shared with the DeepSeek path: rmsnorm, hyper-connection mixing/Sinkhorn).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::glm {
// LayerNorm with weight and bias over the last dim (fp32 math): out = (x − mean)/sqrt(var + eps)·w + b
void layernorm(const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b, float eps, int M, int K, __nv_bfloat16* out, cudaStream_t st);
// Router: logits fp32 [M, E] → s = sigmoid(logits); choose top-k of s + bias; weights = s / Σs · scale. ids/w [M, k].
void router_topk(const float* logits, const float* bias, int M, int E, int k, float scale, int32_t* ids, float* w, cudaStream_t st);
// Same selection, parallel; optionally a second layer in the same launch: logits row = [layer A E | layer B E] (row stride ld), bias1 == nullptr
//   = layer A only. Layer A → ids0/w0, layer B → ids1/w1.
void router_topk2(const float* logits, int ld, const float* bias0, const float* bias1, int M, int E, int k, float scale, int32_t* ids0, float* w0,
                  int32_t* ids1, float* w1, cudaStream_t st);
// Final head input: x[m] = bf16(mean over hc streams of h[m, c, :])
void hc_stream_mean(const __nv_bfloat16* h, int M, int hc, int dim, __nv_bfloat16* x, cudaStream_t st);
// fp32 → bf16
void f32_to_bf16(const float* a, size_t n, __nv_bfloat16* out, cudaStream_t st);
// x *= s (fp32)
void scale_f32(float* x, size_t n, float s, cudaStream_t st);
// out = bf16(a + b) for bf16 inputs
void add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n, __nv_bfloat16* out, cudaStream_t st);
// device-side copy by SM threads (16-byte aligned; avoids the copy engines, which queue behind bulk H2D transfers)
void dcopy(void* dst, const void* src, size_t bytes, cudaStream_t st);
// out[r] = [a[r] | b[r]] (bf16 rows of H each → 2H)
void cat_rows(const __nv_bfloat16* a, const __nv_bfloat16* b, int n, int H, __nv_bfloat16* out, cudaStream_t st);
// acc[i] += w · x[i]
void accum_scaled(float* acc, const __nv_bfloat16* x, float w, size_t n, cudaStream_t st);

// out[idx[r]] += src[r] (fp32 rows of H; idx rows distinct; src/idx may be pinned host memory read via UVA)
void scatter_add_f32_rows(const float* src, const int32_t* idx, int n, int H, float* out, cudaStream_t st);

// out[0:H] = table[*id_dev] — the row index is read on the device (table may be pinned host memory, UVA)
void embed_gather_dev(const __nv_bfloat16* table, const int32_t* id_dev, int H, __nv_bfloat16* out, cudaStream_t st);

// Expert deferral: h[m, c, :] = bf16(h[m, c, :] + post[m, c] · y[m, :]) for M rows of hc streams of width dim; y fp32 [M, dim] (may be pinned
//   host memory, read through UVA), post fp32 [M, hc] — the contribution hc_post would have added for a MoE output y, added later
void hc_inject(__nv_bfloat16* h, const float* post, const float* y, int M, int hc, int dim, cudaStream_t st);

}  // namespace hive::glm
