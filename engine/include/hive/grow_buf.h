// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Growable device buffer: one virtual address range reserved for the largest size, physical memory mapped only for the part in use
//   (CUDA VMM: cuMemAddressReserve once, then cuMemCreate + cuMemMap per growth step; trim unmaps whole steps from the end).
//   The address never changes, so kernels, pointer tables and views made once (DsaSeqCache) stay valid while the buffer grows or shrinks.
// Use: per-sequence KV caches that must allow a long context without holding memory for it in every short conversation.
// Without VMM (driver entry points unavailable) the whole size is allocated once with cudaMalloc — the behaviour before this buffer.
// ⚠️ Callers must make sure no stream reads the part being trimmed (synchronize first).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "hive/common.h"
#include "hive/cache_reserve.h"
#include "hive/devmem.h"

namespace hive {

inline devmem::Backend* grow_backend() {
  static devmem::Backend* be = [] {
    auto* b = new devmem::Backend();
    if (!b->ok) { fprintf(stderr, "[grow] ⚠️CUDA VMM unavailable — growable buffers are allocated at full size\n"); delete b; return (devmem::Backend*)nullptr; }
    return b;
  }();
  return be;
}

struct GrowBuf {
  void* p = nullptr;
  size_t n = 0;          // reserved size (the largest the buffer can hold)
  size_t mapped = 0;     // bytes backed by physical memory, from p
  GrowBuf() = default;
  GrowBuf(const GrowBuf&) = delete;
  GrowBuf& operator=(const GrowBuf&) = delete;
  GrowBuf(GrowBuf&& o) noexcept { *this = std::move(o); }
  GrowBuf& operator=(GrowBuf&& o) noexcept {
    if (this != &o) {
      free();
      p = o.p; n = o.n; mapped = o.mapped; vmm_ = o.vmm_; steps_ = std::move(o.steps_);
      o.p = nullptr; o.n = 0; o.mapped = 0; o.vmm_ = false;
    }
    return *this;
  }
  ~GrowBuf() { free(); }

  // Reserve max_bytes of address space (VMM) or allocate them (fallback). Nothing is mapped yet with VMM.
  void reserve(size_t max_bytes) {
    free();
    if (!max_bytes) return;
    devmem::Backend* be = grow_backend();
    if (be) {
      uintptr_t va = 0;
      const size_t sz = devmem::round_up(max_bytes, be->gran);
      HIVE_CHECK(be->reserve(&va, sz), "GrowBuf: address reservation failed");
      p = (void*)va; n = sz; vmm_ = true;
      return;
    }
    CUDA_CHECK(cudaMalloc(&p, max_bytes));
    n = mapped = max_bytes;
  }
  // Physical memory needed to back the first `bytes` (0 if already mapped).
  size_t grow_cost(size_t bytes) const {
    if (!vmm_ || bytes <= mapped) return 0;
    return devmem::round_up(std::min(bytes, n), grow_backend()->gran) - mapped;
  }
  // Back the first `bytes` with physical memory. false = the driver could not provide it (nothing changed).
  bool ensure(size_t bytes) {
    if (!vmm_ || bytes <= mapped) return bytes <= n;
    devmem::Backend* be = grow_backend();
    const size_t to = devmem::round_up(std::min(bytes, n), be->gran), sz = to - mapped;
    devmem::Handle h{};
    if (!be->create(&h, sz)) { (void)cudaGetLastError(); return false; }
    if (!be->map((uintptr_t)p + mapped, sz, h)) { be->release(h); (void)cudaGetLastError(); return false; }
    steps_.push_back({h, sz});
    mapped = to;
    return bytes <= n;
  }
  // Unmap whole growth steps above `bytes`. Returns the bytes given back.
  size_t trim(size_t bytes) {
    if (!vmm_) return 0;
    devmem::Backend* be = grow_backend();
    size_t freed = 0;
    while (!steps_.empty() && mapped - steps_.back().size >= bytes) {
      const Step s = steps_.back();
      steps_.pop_back();
      mapped -= s.size;
      be->unmap((uintptr_t)p + mapped, s.size);
      be->release(s.h);
      freed += s.size;
    }
    return freed;
  }
  void free() {
    if (!p) return;
    if (vmm_) {
      trim(0);
      grow_backend()->unreserve((uintptr_t)p, n);
    } else {
      cudaFree(p);
    }
    p = nullptr; n = mapped = 0; vmm_ = false;
  }
  bool vmm() const { return vmm_; }
  template <class T> T* as() const { return reinterpret_cast<T*>(p); }

 private:
  struct Step { devmem::Handle h; size_t size; };
  bool vmm_ = false;
  std::vector<Step> steps_;
};

}  // namespace hive
