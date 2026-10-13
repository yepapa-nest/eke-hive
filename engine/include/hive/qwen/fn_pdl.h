// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Programmatic dependent launch for FN's decode kernels (HIVE_QWEN_PDL=1; off by default): a kernel launched with launch_pdl may start while the
//   kernel before it in the stream is still running — its blocks become resident, issue the loads that do not depend on that kernel (weights)
//   and then block in pdl_wait() until the previous grid has finished and its writes are visible. The decode / verify step is a chain of
//   ~1,000 small dependent kernels (nsys 2026-10-09: 13.9 ms of kernels per MTP step); without it every edge costs the launch of the next
//   grid after the last block of the previous one retired.
// Contract for a kernel launched with launch_pdl: no read of data written by earlier kernels and no write of anything an earlier kernel may
//   still read before pdl_wait(). (pdl_wait() is a no-op in a kernel launched without the attribute.)
#pragma once
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <utility>

#include "hive/common.h"

namespace hive::qwen {

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
// let the next grid in the stream start launching now (its blocks still wait in pdl_wait() for this grid to finish)
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

inline bool pdl_on() {
  static const bool on = getenv("HIVE_QWEN_PDL") && atoi(getenv("HIVE_QWEN_PDL")) > 0;   // off by default: no gain measured (int8 MTP step 14.05 on / 14.02 off ms, 2026-10-10)
  return on;
}

template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl_on() ? 1 : 0;
  cfg.attrs = at; cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

}  // namespace hive::qwen
