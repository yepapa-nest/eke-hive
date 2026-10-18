// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// RAII device buffer shared by every model family (moved out of model.h so family headers need not include the DeepSeek model).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "hive/devmem.h"
#include "hive/vram_reserve.h"

namespace hive {

// A device allocation the driver refused. Thrown instead of abort() (2026-10-18: one 2 MiB buffer refused at 26 MiB free took the whole
//   daemon and every open conversation down); the request paths catch std::exception and fail only that request.
struct DeviceOutOfMemory : std::runtime_error { using std::runtime_error::runtime_error; };

// RAII device memory
struct DevBuf {
  void* p = nullptr;
  size_t n = 0;
  DevBuf() = default;
  explicit DevBuf(size_t bytes) { alloc(bytes); }
  ~DevBuf() { free(); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n), counted_(o.counted_) { o.p = nullptr; o.n = 0; o.counted_ = false; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    free();
    p = o.p; n = o.n; counted_ = o.counted_; o.p = nullptr; o.n = 0; o.counted_ = false;
    return *this;
  }
  // With HIVE_SLEEP_VMM (or hived --start-asleep), VMM memory that can be restored at the same VA (hive/devmem.h — sleep level 3). Off = plain cudaMalloc.
  void alloc(size_t bytes) {
    free();
    if (bytes) {
      // HIVE_VRAM_LIMIT_MB while serving: the room comes out of the process's own reserve, and past it the allocation is refused
      const bool within = vram_reserve::before_alloc(bytes, &counted_);
      cudaError_t e = within ? try_alloc(bytes) : cudaErrorMemoryAllocation;
      // Serving only: the reserve is ours, so use it before failing (during startup a refusal must stay a refusal — the limit decides what fits).
      if (within && e != cudaSuccess && vram_reserve::cap_on() && vram_reserve::give(bytes + vram_reserve::kChunk) > 0) e = try_alloc(bytes);
      if (e != cudaSuccess) {
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        char msg[200];
        snprintf(msg, sizeof msg, "cudaMalloc(%.1f MiB) failed: %s (free %.1f MiB / %.1f MiB%s)", bytes / 1048576.0,
                 within ? cudaGetErrorString(e) : "over the VRAM limit", fr / 1048576.0, tot / 1048576.0,
                 within ? "" : (" · reserve " + std::to_string(vram_reserve::held() >> 20) + " MiB").c_str());
        fprintf(stderr, "%s\n", msg);
        p = nullptr;
        counted_ = false;
        if (within) vram_reserve::alloc_refused(bytes);  // the room given for it goes back to the reserve (cap on, up to its target), not to the neighbour
        throw DeviceOutOfMemory(msg);
      }
    }
    n = bytes;
  }
  void free() {
    // Only a real free refills the reserve: an aliased view (p into another buffer) makes cudaFree fail quietly and must not count as room.
    if (p) {
      bool freed = devmem::free_if_owned(p);
      if (!freed) { freed = cudaFree(p) == cudaSuccess; if (!freed) (void)cudaGetLastError(); }
      if (freed) vram_reserve::after_free(n, counted_);
    }
    p = nullptr; n = 0; counted_ = false;
  }
  template <class T> T* as() const { return reinterpret_cast<T*>(p); }

 private:
  bool counted_ = false;  // the room came out of the reserve while serving (vram_reserve::before_alloc) — its free gives it back even asleep
  cudaError_t try_alloc(size_t bytes) {
    p = nullptr;
    if (devmem::on()) { p = devmem::alloc(bytes); return p ? cudaSuccess : cudaErrorMemoryAllocation; }
    const cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) { (void)cudaGetLastError(); p = nullptr; }
    return e;
  }
};

}  // namespace hive
