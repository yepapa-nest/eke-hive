// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Warp/block reduction and indexing helpers for the CUDA kernels.
//
// All reductions use the butterfly order 16, 8, 4, 2, 1 and combine as op(own, partner), so a call
// produces exactly the bits of the hand-unrolled `v = v OP shfl_xor(v, o)` ladder it replaces.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace hive::cu {

constexpr unsigned kAllLanes = 0xffffffffu;

struct Add {
  template <class T> __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};
struct Max {
  __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
  __device__ __forceinline__ int operator()(int a, int b) const { return a > b ? a : b; }
};
struct Min {
  __device__ __forceinline__ float operator()(float a, float b) const { return fminf(a, b); }
  __device__ __forceinline__ int operator()(int a, int b) const { return a < b ? a : b; }
};

// Every lane of the warp ends with the reduction of all 32 lanes.
template <class Op = Add, class T>
__device__ __forceinline__ T warp_allreduce(T v, Op op = Op{}) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask /= 2) v = op(v, __shfl_xor_sync(kAllLanes, v, mask));
  return v;
}

// Inclusive prefix sum across the warp (lane i receives lanes 0..i).
template <class T>
__device__ __forceinline__ T warp_inclusive_sum(T v) {
  const int lane = (int)(threadIdx.x % 32u);
#pragma unroll
  for (int dist = 1; dist <= 16; dist *= 2) {
    const T up = __shfl_up_sync(kAllLanes, v, dist);
    v = lane >= dist ? v + up : v;
  }
  return v;
}

// Block-wide all-reduce for blocks of NWARPS full warps. `slots` is __shared__ T[NWARPS].
// Stage 1 reduces each warp, stage 2 lets warp 0 fold the per-warp values (missing warps contribute
// `identity`), then thread 0 publishes the result in slots[0] which every thread reads back.
// Ends with a barrier so `slots` may be reused immediately.
template <int NWARPS, class Op = Add, class T>
__device__ __forceinline__ T block_allreduce(T v, T* slots, T identity, Op op = Op{}) {
  const unsigned lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
  v = warp_allreduce(v, op);
  if (lane == 0) slots[warp] = v;
  __syncthreads();
  if (warp == 0) {
    T w = lane < (unsigned)NWARPS ? slots[lane] : identity;
    w = warp_allreduce(w, op);
    if (lane == 0) slots[0] = w;
  }
  __syncthreads();
  const T out = slots[0];
  __syncthreads();
  return out;
}

// Flat global thread id and total thread count of a 1-D launch (for grid-stride loops).
__device__ __forceinline__ size_t global_tid() { return (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x; }
__device__ __forceinline__ size_t grid_size() { return (size_t)gridDim.x * (size_t)blockDim.x; }

}  // namespace hive::cu
