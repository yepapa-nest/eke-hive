// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_MTP_VERIFY2 / HIVE_MTP_BATCH — three kernels that run verification rows through the decode path (numeric contract in the header comment of hive/verify_decode.h).
#include <cfloat>

#include "hive/verify_decode.h"
#include "hive/verify_rows.h"  // window column formula (shared with the host CPU test)

namespace hive::k {

namespace {
inline int vgrid(size_t n, int block = 256) { return (int)((n + block - 1) / block); }

// Same formula as window_idxs_kernel (prefill, start_pos = first position of the group, min_src 0) and window_idxs_rows_kernel (one group of rows)
__global__ void window_idxs_verify_kernel(int M, int win, const int32_t* __restrict__ pos, const int32_t* __restrict__ grp, int32_t* __restrict__ out,
                                          int out_stride) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * win) return;
  const int m = (int)(i / win), j = (int)(i % win);
  const int g = grp[m];
  const int32_t v = vrows::window_col(pos[m], pos[g], g, win, j);  // before the group = ring slot; inside the group = chunk row; negative = -1 (verify_rows.h)
  out[(size_t)m * out_stride + j] = v;
}

// Per row the same formula as compressor_step_rows_kernel (write slot -> at the end of a group max, expf, sum, weighted sum — same order). Thread = d; rows in order.
__global__ void compressor_step_seq_kernel(const float* __restrict__ kv, const float* __restrict__ score, int M, int D, int ratio,
                                           const int32_t* __restrict__ pos, float* const* __restrict__ state_kv,
                                           float* const* __restrict__ state_score, float* __restrict__ out, uint8_t* __restrict__ valid) {
  const int d = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  for (int m = 0; m < M; ++m) {
    const int p = pos[m];
    const int slot = p % ratio;
    const bool should = ((p + 1) % ratio) == 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) valid[m] = should ? 1 : 0;
    if (d >= D) continue;
    float* skv = state_kv[m];
    float* ssc = state_score[m];
    skv[slot * D + d] = kv[(size_t)m * D + d];
    ssc[slot * D + d] = score[(size_t)m * D + d];
    if (should) {
      float mx = -FLT_MAX;
      for (int r = 0; r < ratio; ++r) mx = fmaxf(mx, ssc[r * D + d]);
      float sum = 0.f, acc = 0.f, e[8];
      for (int r = 0; r < ratio; ++r) { e[r] = expf(ssc[r * D + d] - mx); sum += e[r]; }
      for (int r = 0; r < ratio; ++r) acc += skv[r * D + d] * (e[r] / sum);
      out[(size_t)m * D + d] = acc;
    }
  }
}

__global__ void compressor_apply_rows_kernel(const float* __restrict__ kv, const float* __restrict__ score, int n, int D, int ratio, int64_t pos0,
                                             float* __restrict__ state_kv, float* __restrict__ state_score) {
  const int d = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (d >= D) return;
  for (int i = 0; i < n; ++i) {
    const int slot = (int)((pos0 + i) % ratio);
    state_kv[slot * D + d] = kv[(size_t)i * D + d];
    state_score[slot * D + d] = score[(size_t)i * D + d];
  }
}
__global__ void accum_f32_rows_ordered_kernel(const float* __restrict__ src, const int32_t* __restrict__ rows, int R, int dim, float* __restrict__ acc) {
  const int d = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (d >= dim) return;
  for (int r = 0; r < R; ++r) acc[(size_t)rows[r] * dim + d] += src[(size_t)r * dim + d];
}
}  // namespace

void accum_f32_rows_ordered(const float* src, const int32_t* rows, int R, int dim, float* acc, cudaStream_t st) {
  if (R <= 0) return;
  accum_f32_rows_ordered_kernel<<<vgrid((size_t)dim), 256, 0, st>>>(src, rows, R, dim, acc);
}
void window_idxs_verify(int M, int win, const int32_t* pos, const int32_t* grp, int32_t* out, int out_stride, cudaStream_t st) {
  window_idxs_verify_kernel<<<vgrid((size_t)M * win), 256, 0, st>>>(M, win, pos, grp, out, out_stride);
}
void compressor_step_seq(const float* kv, const float* score, int M, int D, int ratio, const int32_t* pos, float* const* state_kv,
                         float* const* state_score, float* out, uint8_t* valid, cudaStream_t st) {
  compressor_step_seq_kernel<<<vgrid((size_t)D), 256, 0, st>>>(kv, score, M, D, ratio, pos, state_kv, state_score, out, valid);
}
void compressor_apply_rows(const float* kv, const float* score, int n, int D, int ratio, int64_t pos0, float* state_kv, float* state_score, cudaStream_t st) {
  if (n <= 0) return;
  compressor_apply_rows_kernel<<<vgrid((size_t)D), 256, 0, st>>>(kv, score, n, D, ratio, pos0, state_kv, state_score);
}

}  // namespace hive::k
