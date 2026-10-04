// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Sleep level 3 — device memory that is restored at the same virtual address (CUDA VMM: cuMemAddressReserve/cuMemCreate/cuMemMap).
//
// Why: sleep levels 1 and 2 release expert slots, staging, the elastic range and session KV. What remains (dense weights ~10 GB, the small
//   Work buffers, the cuBLAS workspace, runtime leftovers) is referenced by the decode CUDA graphs and by stored pointers (weight addresses are
//   captured as arguments), so cudaFree followed by a fresh allocation (a different address) would invalidate every graph and pointer.
//   VMM can detach and reattach physical memory while keeping the virtual address (VA) → contents are moved to a pinned host shadow (D2H),
//   then remapped at the same VA and copied back (H2D). Graphs, pointers and the cuBLAS workspace address all stay valid, so nothing has to be
//   rebuilt (the same principle as SGLang's memory saver).
// Enable: HIVE_SLEEP_VMM (env_on — unset/""/"0" = off) or hived --start-asleep (force_on). Decided at startup — when off, DevBuf uses plain
//   cudaMalloc/cudaFree (no path in this header is taken = default behaviour unchanged). If the driver entry points cannot be obtained, this is
//   absorbed as "off" (one log line).
// Allocation: small requests (<= 2 MiB) are packed at 256 B alignment into 32 MiB chunks (one VA + one physical allocation), so that hundreds of
//   small buffers such as weight norms and biases do not each waste a 2 MiB allocation granule; large requests get their own region (rounded
//   up to the granularity). A chunk is returned as a whole once its live allocation count drops to 0.
// release_all (sleep level 3): copy the used range of every mapped region into a pinned host shadow (shadows are reused by the next sleep — the
//   same choice as torch_memory_saver's TMS_RETAIN_CPU_BACKUP) → cuMemUnmap + cuMemRelease. The VA stays reserved.
//   ⚠️ Precondition: the device is idle (the caller runs cudaDeviceSynchronize).
// restore_all (wake): cuMemCreate + cuMemMap + access rights at the same VA → shadow H2D. If anything fails, whatever was restored in this call
//   is detached again and the failure is returned (left released).
// free while asleep: there is no mapping, so only the VA is returned. alloc while asleep: the new region is mapped as usual (restore skips it).
// First-sleep speedup (prepare_shadows): the first level-3 sleep took 6,828 ms, the second 573 ms (same 10,164 MiB). Since the second one is
//   fast (17 GB/s) and the wake H2D takes 407 ms (25 GB/s), the shadows were pinned from the start — this is not a pageable copy. The ~6.25 s
//   difference is the `ext > r.shadow_n` branch of release_all, i.e. the cudaHostAlloc that only runs on the first sleep (allocating and pinning
//   9,862 MiB of shadows at ~1.6 GB/s). Therefore shadows for long-lived regions (keep — dense weights, MTP stage, the Runtime constructor's
//   Work, cuBLAS and sampler buffers, which hived wraps in keep_begin/keep_end at startup) are allocated in advance during startup (background
//   thread, before serving). Sleeping then reuses the existing shadows (the same path as a second sleep). Total shadow memory is the same as
//   after a first sleep without this (only the timing moves earlier).
//   Non-keep regions (slots, staging, session pool — released earlier by levels 1/2 — and buffers allocated late at runtime) still get their
//   shadow when going to sleep (Report.shadow_new is that amount).
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "hive/clock.h"

#ifndef HIVE_FAKE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace hive {
namespace devmem {

constexpr size_t kChunk = 32ull << 20, kSmall = 2ull << 20, kAlign = 256;

#ifdef HIVE_FAKE_CUDA
// ---- Fake backend (CPU tests): VA = aligned_alloc, detach = overwrite with 0xCD (simulates content loss), attach = no-op. FAKE_DEV_MALLOC_MAX makes attach fail.
using Handle = unsigned long long;
struct Backend {
  bool ok = true;
  size_t gran = 2ull << 20;
  bool reserve(uintptr_t* va, size_t n) { void* p = aligned_alloc(4096, n); *va = (uintptr_t)p; return p != nullptr; }
  void unreserve(uintptr_t va, size_t) { ::free((void*)va); }
  bool create(Handle* h, size_t n) {
    if (const char* lim = getenv("FAKE_DEV_MALLOC_MAX"); lim && *lim && n > (size_t)strtoull(lim, nullptr, 10)) return false;
    static Handle next = 0;
    *h = ++next;
    return true;
  }
  bool map(uintptr_t, size_t, Handle) { return true; }
  void unmap(uintptr_t va, size_t n) { memset((void*)va, 0xCD, n); }
  void release(Handle) {}
};
#else
// ---- Real backend: driver functions are obtained with cudaGetDriverEntryPoint (libcuda is not linked separately — build files unchanged).
using Handle = CUmemGenericAllocationHandle;
struct Backend {
  bool ok = false;
  size_t gran = 2ull << 20;
  int dev = 0;
  using FnGran = CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
  using FnReserve = CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
  using FnAddrFree = CUresult (*)(CUdeviceptr, size_t);
  using FnCreate = CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
  using FnRelease = CUresult (*)(CUmemGenericAllocationHandle);
  using FnMap = CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
  using FnUnmap = CUresult (*)(CUdeviceptr, size_t);
  using FnAccess = CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
  FnGran f_gran = nullptr; FnReserve f_reserve = nullptr; FnAddrFree f_addr_free = nullptr; FnCreate f_create = nullptr; FnRelease f_release = nullptr;
  FnMap f_map = nullptr; FnUnmap f_unmap = nullptr; FnAccess f_access = nullptr;
  CUmemAllocationProp prop{};
  template <class F> bool sym(const char* name, F& f) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || !p) { (void)cudaGetLastError(); return false; }
    f = reinterpret_cast<F>(p);
    return true;
  }
  Backend() {
    if (cudaFree(nullptr) != cudaSuccess || cudaGetDevice(&dev) != cudaSuccess) { (void)cudaGetLastError(); return; }  // make the primary context current (precondition for driver calls)
    ok = sym("cuMemGetAllocationGranularity", f_gran) && sym("cuMemAddressReserve", f_reserve) && sym("cuMemAddressFree", f_addr_free) &&
         sym("cuMemCreate", f_create) && sym("cuMemRelease", f_release) && sym("cuMemMap", f_map) && sym("cuMemUnmap", f_unmap) &&
         sym("cuMemSetAccess", f_access);
    if (!ok) return;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    size_t g = 0;
    if (f_gran(&g, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS || g == 0) { ok = false; return; }
    gran = g;
  }
  bool reserve(uintptr_t* va, size_t n) { CUdeviceptr p = 0; if (f_reserve(&p, n, gran, 0, 0) != CUDA_SUCCESS) return false; *va = (uintptr_t)p; return true; }
  void unreserve(uintptr_t va, size_t n) { f_addr_free((CUdeviceptr)va, n); }
  bool create(Handle* h, size_t n) { return f_create(h, n, &prop, 0) == CUDA_SUCCESS; }
  bool map(uintptr_t va, size_t n, Handle h) {
    if (f_map((CUdeviceptr)va, n, 0, h, 0) != CUDA_SUCCESS) return false;
    CUmemAccessDesc a{};
    a.location = prop.location;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (f_access((CUdeviceptr)va, n, &a, 1) != CUDA_SUCCESS) { f_unmap((CUdeviceptr)va, n); return false; }
    return true;
  }
  void unmap(uintptr_t va, size_t n) { f_unmap((CUdeviceptr)va, n); }
  void release(Handle h) { f_release(h); }
};
#endif

struct Region {
  size_t size = 0;      // VA/physical size (multiple of the granularity)
  size_t used = 0;      // chunk: end of the packed range (shadow extent); large region: requested size
  int live = 0;         // live allocations in the chunk
  bool chunk = false, mapped = false;
  Handle h{};
  uint8_t* shadow = nullptr;  // host shadow (pinned; plain memory if pinning fails)
  size_t shadow_n = 0;
  bool shadow_pinned = false;
  bool keep = false;          // long-lived region (created inside keep_begin..keep_end, or a chunk that received a small allocation there) — prepare_shadows target
};
struct State {
  std::mutex mu;
  bool forced = false;
  Backend* be = nullptr;
  std::map<uintptr_t, Region> regions;     // VA start -> region
  std::map<uintptr_t, uintptr_t> owner;    // allocation address -> region VA
  uintptr_t open = 0;                      // chunk currently being packed (0 = none)
  bool released = false;                   // after release_all, before restore_all
  size_t mapped_bytes = 0;
  int keep_depth = 0;                      // keep_begin depth (> 0 marks new regions as keep)
};
inline State& st() { static State s; return s; }
inline bool env_flag() { const char* v = getenv("HIVE_SLEEP_VMM"); return v && *v && strcmp(v, "0") != 0; }
inline void force_on() { std::lock_guard<std::mutex> lk(st().mu); st().forced = true; }
// Enabled? (the first call creates the backend — on failure this is absorbed as permanently off)
inline bool on() {
  static std::atomic<int> state{-1};  // -1 undecided, 0 off, 1 on
  const int v = state.load(std::memory_order_acquire);
  if (v >= 0) return v == 1;
  State& s = st();
  std::lock_guard<std::mutex> lk(s.mu);
  if (state.load() >= 0) return state.load() == 1;
  if (!env_flag() && !s.forced) { state.store(0); return false; }
  s.be = new Backend();
  if (!s.be->ok) {
    fprintf(stderr, "[devmem] ⚠️CUDA VMM unavailable (driver entry points) — plain cudaMalloc, sleep level 3 off\n");
    state.store(0);
    return false;
  }
  fprintf(stderr, "[devmem] VMM device memory on (granularity %zu KiB · small allocations in %zu MiB chunks) — sleep level 3 can release dense weights\n",
          s.be->gran >> 10, kChunk >> 20);
  state.store(1, std::memory_order_release);
  return true;
}
inline size_t round_up(size_t n, size_t g) { return (n + g - 1) / g * g; }
// Regions allocated in between are long-lived (they survive until sleep level 3) — no-op when off
inline void keep_begin() { if (!on()) return; std::lock_guard<std::mutex> lk(st().mu); ++st().keep_depth; }
inline void keep_end() { if (!on()) return; std::lock_guard<std::mutex> lk(st().mu); if (st().keep_depth > 0) --st().keep_depth; }
inline bool new_region(State& s, size_t size, bool chunk, uintptr_t* va) {
  if (!s.be->reserve(va, size)) return false;
  Region r;
  r.size = size; r.chunk = chunk;
  if (!s.be->create(&r.h, size)) { s.be->unreserve(*va, size); return false; }
  if (!s.be->map(*va, size, r.h)) { s.be->release(r.h); s.be->unreserve(*va, size); return false; }
  r.mapped = true;
  s.mapped_bytes += size;
  s.regions[*va] = r;
  return true;
}
inline void drop_region(State& s, uintptr_t va) {
  Region& r = s.regions.at(va);
  if (r.mapped) { s.be->unmap(va, r.size); s.be->release(r.h); s.mapped_bytes -= r.size; }
  s.be->unreserve(va, r.size);
  if (r.shadow) { if (r.shadow_pinned) cudaFreeHost(r.shadow); else ::free(r.shadow); }
  if (s.open == va) s.open = 0;
  s.regions.erase(va);
}
// nullptr = failure (the calling DevBuf reports and stops as before)
inline void* alloc(size_t n) {
  State& s = st();
  std::lock_guard<std::mutex> lk(s.mu);
  if (n == 0) return nullptr;
  if (n <= kSmall) {
    const size_t need = round_up(n, kAlign);
    if (s.open) {
      Region& r = s.regions.at(s.open);
      if (r.mapped && r.used + need <= r.size) {
        const uintptr_t p = s.open + r.used;
        r.used += need; ++r.live;
        r.keep = r.keep || s.keep_depth > 0;
        s.owner[p] = s.open;
        return (void*)p;
      }
    }
    uintptr_t va = 0;
    if (!new_region(s, round_up(kChunk, s.be->gran), true, &va)) return nullptr;
    Region& r = s.regions.at(va);
    r.used = need; r.live = 1;
    r.keep = s.keep_depth > 0;
    s.open = va;
    s.owner[va] = va;
    return (void*)va;
  }
  uintptr_t va = 0;
  if (!new_region(s, round_up(n, s.be->gran), false, &va)) return nullptr;
  s.regions.at(va).used = n;
  s.regions.at(va).keep = s.keep_depth > 0;
  s.owner[va] = va;
  return (void*)va;
}
// If this module handed out the address, return it and give true (otherwise false — the caller uses cudaFree). Immediately false when off.
inline bool free_if_owned(void* p) {
  if (!on()) return false;
  State& s = st();
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.owner.find((uintptr_t)p);
  if (it == s.owner.end()) return false;
  const uintptr_t va = it->second;
  s.owner.erase(it);
  Region& r = s.regions.at(va);
  if (r.chunk && --r.live > 0) return true;
  drop_region(s, va);
  return true;
}
struct Report {
  int regions = 0; size_t bytes = 0, copied = 0; double ms = 0; bool ok = true; std::string error;
  size_t shadow_new = 0; double shadow_ms = 0;  // release_all: shadows newly allocated this time (bytes, time — the part prepare_shadows did not cover); prepare_shadows: shadows allocated
  int shadow_failed = 0;                        // prepare_shadows: regions whose shadow could not be allocated (absorbed — those get one at sleep time as before)
};
// First-sleep speedup: allocate a pinned shadow for every keep region in advance (a chunk gets its full size — covering small allocations appended
//   later; a large region gets its requested size). The lock is held only while taking the list and while attaching (the allocation itself runs
//   outside the lock — other threads' alloc/free are not blocked meanwhile). On attach the shadow is discarded unless the region is unchanged
//   (keep, mapped, no shadow, size sufficient) — the shadow of a sleeping region (which holds contents) is never swapped. Failures are absorbed
//   (the region keeps the old path). Not called while asleep (hived finishes it during startup, before ready).
inline Report prepare_shadows() {
  Report rep;
  if (!on()) return rep;
  State& s = st();
  const auto t0 = hive::SteadyClock::now();
#ifndef HIVE_FAKE_CUDA
  if (cudaSetDevice(s.be->dev) != cudaSuccess) { (void)cudaGetLastError(); rep.ok = false; rep.error = "cudaSetDevice"; return rep; }  // background thread: primary context of the same device
#endif
  std::vector<std::pair<uintptr_t, size_t>> todo;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.released) return rep;
    for (auto& [va, r] : s.regions) {
      if (!r.keep || !r.mapped || r.shadow) continue;
      todo.push_back({va, r.chunk ? r.size : std::min(r.used, r.size)});
    }
  }
  for (auto [va, want] : todo) {
    if (!want) continue;
    void* h = nullptr;
    if (cudaHostAlloc(&h, want, cudaHostAllocDefault) != cudaSuccess || !h) { (void)cudaGetLastError(); ++rep.shadow_failed; continue; }
    bool used = false;
    {
      std::lock_guard<std::mutex> lk(s.mu);
      auto it = s.regions.find(va);
      if (it != s.regions.end() && !s.released) {
        Region& r = it->second;
        if (r.keep && r.mapped && !r.shadow && want >= std::min(r.used, r.size)) {
          r.shadow = (uint8_t*)h; r.shadow_n = want; r.shadow_pinned = true;
          used = true;
        }
      }
    }
    if (used) { rep.shadow_new += want; ++rep.regions; }
    else cudaFreeHost(h);
  }
  rep.shadow_ms = rep.ms = hive::ms_since(t0);
  return rep;
}
inline Report release_all() {
  Report rep;
  if (!on()) { rep.ok = false; rep.error = "VMM off (start hived with HIVE_SLEEP_VMM=1)"; return rep; }
  State& s = st();
  std::lock_guard<std::mutex> lk(s.mu);
  const auto t0 = hive::SteadyClock::now();
  for (auto& [va, r] : s.regions) {
    if (!r.mapped) continue;
    const size_t ext = std::min(r.used, r.size);
    if (ext > r.shadow_n) {
      const auto ts = hive::SteadyClock::now();
      if (r.shadow) { if (r.shadow_pinned) cudaFreeHost(r.shadow); else ::free(r.shadow); }
      r.shadow = nullptr; r.shadow_n = 0;
      void* h = nullptr;
      if (cudaHostAlloc(&h, ext, cudaHostAllocDefault) == cudaSuccess && h) r.shadow_pinned = true;
      else { (void)cudaGetLastError(); h = malloc(ext); r.shadow_pinned = false; }  // a pinning failure is absorbed with plain memory (only the copy is slower)
      if (!h) { rep.ok = false; rep.error = "host shadow allocation failed"; break; }
      r.shadow = (uint8_t*)h; r.shadow_n = ext;
      rep.shadow_new += ext;
      rep.shadow_ms += hive::ms_since(ts);
    }
    if (ext && cudaMemcpy(r.shadow, (void*)va, ext, cudaMemcpyDeviceToHost) != cudaSuccess) {
      (void)cudaGetLastError(); rep.ok = false; rep.error = "D2H copy failed"; break;
    }
    rep.copied += ext;
  }
  if (rep.ok) {
    for (auto& [va, r] : s.regions) {
      if (!r.mapped) continue;
      s.be->unmap(va, r.size);
      s.be->release(r.h);
      r.mapped = false;
      s.mapped_bytes -= r.size;
      rep.bytes += r.size; ++rep.regions;
    }
    s.released = true;
  }
  rep.ms = hive::ms_since(t0);
  return rep;
}
inline Report restore_all() {
  Report rep;
  if (!on()) return rep;
  State& s = st();
  std::lock_guard<std::mutex> lk(s.mu);
  const auto t0 = hive::SteadyClock::now();
  std::vector<uintptr_t> done;
  for (auto& [va, r] : s.regions) {
    if (r.mapped) continue;
    if (!s.be->create(&r.h, r.size) || !s.be->map(va, r.size, r.h)) {
      rep.ok = false; rep.error = "not enough free VRAM to map " + std::to_string(r.size >> 20) + " MiB back at its address";
      break;
    }
    r.mapped = true;
    s.mapped_bytes += r.size;
    done.push_back(va);
    const size_t ext = std::min(std::min(r.used, r.size), r.shadow_n);
    if (ext && cudaMemcpy((void*)va, r.shadow, ext, cudaMemcpyHostToDevice) != cudaSuccess) {
      (void)cudaGetLastError(); rep.ok = false; rep.error = "H2D copy failed"; break;
    }
    rep.copied += ext; rep.bytes += r.size; ++rep.regions;
  }
  if (!rep.ok) {  // detach whatever was restored this time (left released — the next wake starts from scratch)
    for (uintptr_t va : done) { Region& r = s.regions.at(va); s.be->unmap(va, r.size); s.be->release(r.h); r.mapped = false; s.mapped_bytes -= r.size; }
    rep.bytes = 0; rep.regions = 0; rep.copied = 0;
  } else s.released = false;
  rep.ms = hive::ms_since(t0);
  return rep;
}
inline bool released() { if (!on()) return false; std::lock_guard<std::mutex> lk(st().mu); return st().released; }
inline size_t mapped_bytes() { if (!on()) return 0; std::lock_guard<std::mutex> lk(st().mu); return st().mapped_bytes; }
inline size_t shadow_bytes() {
  if (!on()) return 0;
  std::lock_guard<std::mutex> lk(st().mu);
  size_t n = 0;
  for (auto& [va, r] : st().regions) n += r.shadow_n;
  return n;
}

}  // namespace devmem
}  // namespace hive
