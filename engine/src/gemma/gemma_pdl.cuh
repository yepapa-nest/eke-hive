// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Programmatic dependent launch for the Gemma decode chain (internal to src/gemma): every kernel of a forward is launched with
//   programmatic stream serialization and starts with pdl_trigger(), so the next kernel's blocks are scheduled while this one runs; a
//   kernel touches nothing a predecessor writes (and writes nothing a predecessor reads) before pdl_wait(), which returns once the previous
//   kernel — and so, transitively, every earlier one — has completed. Weight reads are independent of the chain: the GEMVs issue an L2 bulk
//   prefetch of their weights before pdl_wait(), so the weight stream does not stop at kernel boundaries.
//   HIVE_GEMMA_NO_PDL=1 launches plainly (A/B switch; the kernels are correct either way — the two instructions are no-ops then).
#pragma once
#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

#include "hive/common.h"

namespace hive::gemma::k {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// L2 prefetch of [p, p + bytes) (p and bytes multiples of 16)
__device__ __forceinline__ void prefetch_l2(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

inline bool pdl_off() {
  static const bool off = [] { const char* v = getenv("HIVE_GEMMA_NO_PDL"); return v && *v && *v != '0'; }();
  return off;
}

template <class... KArgs, class... Args>
void launch_pdl(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl_off() ? 0 : 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, std::forward<Args>(args)...));
}

}  // namespace hive::gemma::k
