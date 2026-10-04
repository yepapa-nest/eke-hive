// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM routed-expert compute on the GPU. An expert record (NVFP4) is one contiguous block:
//   [w1 gate e2m1 [I, H/2]][s1 e4m3 [I, H/16]][w3 up][s3][w2 down e2m1 [H, I/2]][s2 e4m3 [H, I/16]][g1 g3 g2 fp32] (4096-aligned)
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::glm {

struct ExpertRecLayout {
  size_t w1, s1, w3, s3, w2, s2, g, total;
  static ExpertRecLayout make(int H, int I) {
    ExpertRecLayout L{};
    const size_t wgu = (size_t)I * H / 2, sgu = (size_t)I * H / 16, wd = (size_t)H * I / 2, sd = (size_t)H * I / 16;
    L.w1 = 0; L.s1 = wgu; L.w3 = L.s1 + sgu; L.s3 = L.w3 + wgu; L.w2 = L.s3 + sgu; L.s2 = L.w2 + wd; L.g = L.s2 + sd;
    L.total = (L.g + 16 + 4095) / 4096 * 4096;
    return L;
  }
};

// One (row, expert) pair of a decode step: x row index, device record pointer, routing weight.
struct MoePair { int row; const uint8_t* rec; float w; };

// Decode (rows M ≤ 64): out[row] += Σ_pairs w · expert(x[row]) for the given pairs (device array of n pairs).
//   ws: device workspace ≥ moe_decode_ws_bytes(n). Deterministic: per-pair partials are reduced in pair order.
size_t moe_decode_ws_bytes(int n_pairs, int H, int I);
void moe_decode(const ExpertRecLayout& L, const MoePair* pairs, int n_pairs, const __nv_bfloat16* x, int H, int I, float limit,
                float* out, void* ws, cudaStream_t st);
// Prefill helper: dequantize one expert record to bf16 matrices gu [2I, H] (gate rows then up rows) and d [H, I].
void expert_dequant(const ExpertRecLayout& L, const uint8_t* rec, int H, int I, __nv_bfloat16* gu, __nv_bfloat16* d, cudaStream_t st);
// y[r, i] = bf16(silu(clamp(gate)) · clamp(up)) from gu_out [R, 2I] (gate cols then up cols)
void swiglu_rows(const __nv_bfloat16* gu_out, int R, int I, float limit, __nv_bfloat16* y, cudaStream_t st);
// out[rows[r]] += w[r] · e[r]  (fp32 accumulate; rows unique within one call)
void scatter_add_rows(const __nv_bfloat16* e, const int32_t* rows, const float* w, int R, int H, float* out, cudaStream_t st);
// out[r] = x[rows[r]] (bf16 rows gather)
void gather_rows(const __nv_bfloat16* x, const int32_t* rows, int R, int H, __nv_bfloat16* out, cudaStream_t st);
// a[i] += b[i]
void add_f32(float* a, const float* b, size_t n, cudaStream_t st);

}  // namespace hive::glm
