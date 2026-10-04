// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// RAII device buffer shared by every model family (moved out of model.h so family headers need not include the DeepSeek model).
#pragma once
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "hive/devmem.h"

namespace hive {

// RAII device memory
struct DevBuf {
  void* p = nullptr;
  size_t n = 0;
  DevBuf() = default;
  explicit DevBuf(size_t bytes) { alloc(bytes); }
  ~DevBuf() { free(); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    free();
    p = o.p; n = o.n; o.p = nullptr; o.n = 0;
    return *this;
  }
  // With HIVE_SLEEP_VMM (or hived --start-asleep), VMM memory that can be restored at the same VA (hive/devmem.h — sleep level 3). Off = plain cudaMalloc.
  void alloc(size_t bytes) {
    free();
    if (bytes) {
      cudaError_t e = cudaSuccess;
      if (devmem::on()) { p = devmem::alloc(bytes); if (!p) e = cudaErrorMemoryAllocation; }
      else e = cudaMalloc(&p, bytes);
      if (e != cudaSuccess) {
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        fprintf(stderr, "cudaMalloc(%.1f MiB) failed: %s (free %.1f MiB / %.1f MiB)\n", bytes / 1048576.0, cudaGetErrorString(e),
                fr / 1048576.0, tot / 1048576.0);
        abort();
      }
    }
    n = bytes;
  }
  void free() {
    if (p && !devmem::free_if_owned(p)) cudaFree(p);
    p = nullptr; n = 0;
  }
  template <class T> T* as() const { return reinterpret_cast<T*>(p); }
};

}  // namespace hive
