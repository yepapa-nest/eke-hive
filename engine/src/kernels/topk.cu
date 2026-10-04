// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include <cub/cub.cuh>

#include "hive/model_kernels.h"
#include "hive/topk.h"

namespace hive {

namespace {
__global__ void take_first_k_kernel(const int32_t* __restrict__ sorted_idx, int M, int T, int k, int32_t* __restrict__ out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * k) return;
  int m = i / k, j = i % k;
  out[i] = sorted_idx[(size_t)m * T + j];
}
__global__ void seg_offsets_kernel(int32_t* off, int M, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= M) off[i] = i * stride;
}
}  // namespace

void TopK::reserve(int M, int T) {
  if (M <= capM_ && T <= capT_) return;
  capM_ = std::max(capM_, M);
  capT_ = std::max(capT_, T);
  idx_in_.alloc((size_t)capM_ * capT_ * 4);
  idx_out_.alloc((size_t)capM_ * capT_ * 4);
  keys_out_.alloc((size_t)capM_ * capT_ * 4);
  offsets_.alloc((size_t)(capM_ + 1) * 4);
  sel_keys_.alloc((size_t)capM_ * 4096 * 4);
  sel_vals_.alloc((size_t)capM_ * 4096 * 4);
  sel_out_.alloc((size_t)capM_ * 4096 * 4);
  // Query CUB temporary storage size
  size_t bytes1 = 0, bytes2 = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending<float, int32_t, const int32_t*, const int32_t*>(
      nullptr, bytes1, (const float*)nullptr, (float*)nullptr, (const int32_t*)nullptr, (int32_t*)nullptr, capM_ * capT_, capM_,
      (const int32_t*)nullptr, (const int32_t*)nullptr, 0, 32, 0);
  cub::DeviceSegmentedRadixSort::SortKeys<int32_t, const int32_t*, const int32_t*>(nullptr, bytes2, (const int32_t*)nullptr,
                                                                                    (int32_t*)nullptr, capM_ * 4096, capM_,
                                                                                    (const int32_t*)nullptr, (const int32_t*)nullptr, 0,
                                                                                    32, 0);
  tmp_bytes_ = std::max(bytes1, bytes2) + 1024;
  tmp_.alloc(tmp_bytes_);
}

void TopK::topk_rows(const float* keys, int M, int T, int k, int32_t* out, cudaStream_t st) {
  HIVE_CHECK(M <= capM_ && T <= capT_ && k <= 4096 && k <= T, "topk capacity");
  k::iota_rows(idx_in_.as<int32_t>(), M, T, st);
  seg_offsets_kernel<<<(M + 256) / 256, 256, 0, st>>>(offsets_.as<int32_t>(), M, T);
  size_t bytes = tmp_bytes_;
  CUDA_CHECK((cudaError_t)cub::DeviceSegmentedRadixSort::SortPairsDescending(
      tmp_.p, bytes, keys, keys_out_.as<float>(), idx_in_.as<const int32_t>(), idx_out_.as<int32_t>(), M * T, M,
      offsets_.as<const int32_t>(), offsets_.as<const int32_t>() + 1, 0, 32, st));
  take_first_k_kernel<<<(int)(((size_t)M * k + 255) / 256), 256, 0, st>>>(idx_out_.as<int32_t>(), M, T, k, sel_keys_.as<int32_t>());
  // Sort the k selected entries in ascending order
  seg_offsets_kernel<<<(M + 256) / 256, 256, 0, st>>>(offsets_.as<int32_t>(), M, k);
  bytes = tmp_bytes_;
  CUDA_CHECK((cudaError_t)cub::DeviceSegmentedRadixSort::SortKeys(tmp_.p, bytes, sel_keys_.as<const int32_t>(), out, M * k, M,
                                                                  offsets_.as<const int32_t>(), offsets_.as<const int32_t>() + 1, 0,
                                                                  32, st));
}

}  // namespace hive
