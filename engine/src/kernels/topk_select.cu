// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Per-row top-k v2 — single-block radix select (fp32 keys). Avoids a full CUB sort for decode (M=1, large T), and can be put into a
// CUDA graph when the kernel arguments come from mapped memory. Result = top-k indices in ascending order (padded with -1 when a row has
// fewer than k valid candidates). Ties go to the smaller index.
//   (1) map keys to sortable uint32 (sign flip) (2) four 8-bit digit histograms determine the digits of the k-th value
//   (3) collect everything above the threshold (values equal to the threshold from the front) (4) in-block sort (bitonic, k <= 2048)
#include "hive/model_kernels.h"

namespace hive::k {

namespace {
constexpr int TPK_THREADS = 1024;

__device__ inline uint32_t f2ord(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // so that larger values become larger integers
}

constexpr int CAND_CAP = 8192;  // compaction buffer for candidates after pass 0 (same top 8 bits)

__global__ void __launch_bounds__(TPK_THREADS) topk_select_kernel(const float* __restrict__ keys, const int32_t* __restrict__ T_rows, int T_in,
                                                                  int k, int row_stride, int32_t* __restrict__ out, int out_stride) {
  const int T = T_rows ? min(T_rows[blockIdx.x], T_in) : T_in;
  const int keff = min(k, T);  // rows with fewer than k candidates emit only what exists (rest -1)
  __shared__ int hist[32][256];   // per-warp histograms (32x less atomic contention)
  __shared__ uint32_t prefix;
  __shared__ int need;
  __shared__ int cursor;
  __shared__ int ncand;           // number of candidates (-1 if above CAND_CAP = fall back to a full scan)
  extern __shared__ int32_t dyn[];  // [CAND_CAP] candidate indices + [K2] selection
  int32_t* cand = dyn;
  int32_t* sel = dyn + CAND_CAP;
  const int row = blockIdx.x;
  const float* kr = keys + (size_t)row * row_stride;
  const int tid = threadIdx.x, warp = tid >> 5;
  if (tid == 0) { prefix = 0; need = keff - 1; cursor = 0; ncand = -1; }
  __syncthreads();
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - pass * 8;
    const uint32_t mask = pass == 0 ? 0u : (0xFFFFFFFFu << (32 - pass * 8));
    for (int i = tid; i < 32 * 256; i += TPK_THREADS) (&hist[0][0])[i] = 0;
    __syncthreads();
    if (ncand >= 0) {  // candidates only
      for (int c = tid; c < ncand; c += TPK_THREADS) {
        uint32_t u = f2ord(kr[cand[c]]);
        if ((u & mask) == prefix) atomicAdd(&hist[warp][(u >> shift) & 0xFF], 1);
      }
    } else {
      for (int t = tid; t < T; t += TPK_THREADS) {
        uint32_t u = f2ord(kr[t]);
        if ((u & mask) == prefix) atomicAdd(&hist[warp][(u >> shift) & 0xFF], 1);
      }
    }
    __syncthreads();
    for (int b = tid; b < 256; b += TPK_THREADS) {
      int sum = 0;
      for (int w = 0; w < 32; ++w) sum += hist[w][b];
      hist[0][b] = sum;
    }
    __syncthreads();
    if (tid == 0) {
      int acc = 0, bin = 0;
      for (int b = 255; b >= 0; --b) {
        if (acc + hist[0][b] > need) { bin = b; break; }
        acc += hist[0][b];
      }
      need -= acc;
      prefix |= ((uint32_t)bin) << shift;
      cursor = 0;
    }
    __syncthreads();
    if (pass == 0) {  // top 8 bits are fixed -> compact the elements with that prefix into candidates
      const uint32_t m8 = 0xFF000000u;
      for (int t = tid; t < T; t += TPK_THREADS) {
        uint32_t u = f2ord(kr[t]);
        if ((u & m8) == prefix) { int p = atomicAdd(&cursor, 1); if (p < CAND_CAP) cand[p] = t; }
      }
      __syncthreads();
      if (tid == 0) { ncand = cursor <= CAND_CAP ? cursor : -1; cursor = 0; }
      __syncthreads();
    }
  }
  const uint32_t thr = prefix;
  // Collect: everything greater than the threshold (one full scan) + (k - count_gt) elements equal to it (in index order)
  for (int t = tid; t < T; t += TPK_THREADS) {
    uint32_t u = f2ord(kr[t]);
    if (u > thr) { int p = atomicAdd(&cursor, 1); if (p < keff) sel[p] = t; }
  }
  __syncthreads();
  const int count_gt = cursor;
  __syncthreads();
  // Gather the elements equal to the threshold in parallel (capped at EQ_CAP) and pick the want smallest indices — a sequential scan by one thread took 5 ms at T=200K
  __shared__ int eq_cursor;
  __shared__ int eqbuf[1024];
  if (tid == 0) eq_cursor = 0;
  __syncthreads();
  for (int t = tid; t < T; t += TPK_THREADS) {
    if (f2ord(kr[t]) == thr) { int p = atomicAdd(&eq_cursor, 1); if (p < 1024) eqbuf[p] = t; }
  }
  __syncthreads();
  if (tid == 0) {
    int want = keff - count_gt;
    int p = count_gt;
    const int neq = eq_cursor <= 1024 ? eq_cursor : -1;
    if (neq >= 0) {
      for (int rep = 0; rep < want; ++rep) {
        int best = 0x7FFFFFFF, bi = -1;
        for (int c = 0; c < neq; ++c) if (eqbuf[c] >= 0 && eqbuf[c] < best) { best = eqbuf[c]; bi = c; }
        if (bi < 0) break;
        sel[p++] = best;
        eqbuf[bi] = -1;
      }
    } else {
      for (int t = 0; t < T && want > 0; ++t) if (f2ord(kr[t]) == thr) { sel[p++] = t; --want; }
    }
    cursor = p;
  }
  __syncthreads();
  const int n = cursor;
  int K2 = 1;
  while (K2 < k) K2 <<= 1;
  int32_t* arr = sel;
  for (int i = tid; i < K2; i += TPK_THREADS) {
    if (i >= n) arr[i] = 0x7FFFFFFF;
    else if (kr[arr[i]] == -INFINITY) arr[i] = 0x7FFFFFFF;
  }
  __syncthreads();
  for (int size = 2; size <= K2; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = tid; i < K2; i += TPK_THREADS) {
        int j = i ^ stride;
        if (j > i) {
          bool up = ((i & size) == 0);
          int32_t a = arr[i], b2 = arr[j];
          if ((a > b2) == up) { arr[i] = b2; arr[j] = a; }
        }
      }
      __syncthreads();
    }
  }
  for (int i = tid; i < k; i += TPK_THREADS) out[(size_t)row * out_stride + i] = arr[i] == 0x7FFFFFFF ? -1 : arr[i];
}
}  // namespace

void topk_select_rows_t(const float* keys, int M, const int32_t* T_rows, int T, int k, int row_stride, int32_t* out, int out_stride,
                        cudaStream_t st) {
  HIVE_CHECK(k >= 1 && k <= 2048, "topk_select k");
  int K2 = 1;
  while (K2 < k) K2 <<= 1;
  size_t smem = (size_t)(CAND_CAP + K2) * 4 + 64;
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(topk_select_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)((CAND_CAP + 2048) * 4 + 64)));
    configured = true;
  }
  topk_select_kernel<<<M, TPK_THREADS, smem, st>>>(keys, T_rows, T, k, row_stride, out, out_stride);
}

void topk_select_rows(const float* keys, int M, int T, int k, int row_stride, int32_t* out, int out_stride, cudaStream_t st) {
  HIVE_CHECK(k >= 1 && k <= 2048 && k <= T, "topk_select k");
  int K2 = 1;
  while (K2 < k) K2 <<= 1;
  size_t smem = (size_t)(CAND_CAP + K2) * 4 + 64;
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(topk_select_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)((CAND_CAP + 2048) * 4 + 64)));
    configured = true;
  }
  topk_select_kernel<<<M, TPK_THREADS, smem, st>>>(keys, nullptr, T, k, row_stride, out, out_stride);
}

}  // namespace hive::k
