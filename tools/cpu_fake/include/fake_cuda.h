// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU-only fake CUDA runtime for host-logic tests. NO GPU is touched.
//
// Device memory is host memory. Streams have two modes:
//   * default (FAKE_CUDA_ASYNC unset/0): every stream op runs immediately in the caller.
//   * FAKE_CUDA_ASYNC=1: every non-null stream is a real worker thread that executes its
//     queue in order; events/streams synchronise ONLY through the calls the production code
//     makes (cudaEventRecord/Query/Synchronize, cudaStreamWaitEvent/Synchronize). Under TSAN
//     a host access that is not ordered by such a fence races with the worker's memcpy and is
//     reported. FAKE_CUDA_DELAY_US=N sleeps N us before each async memcpy to widen windows.
// The null stream is executed synchronously in the caller and does NOT implicitly
// synchronise other streams (the production streams are cudaStreamNonBlocking).
#pragma once
#define HIVE_FAKE_CUDA 1  // makes hive/devmem.h select the fake VMM backend
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

typedef int cudaError_t;
enum : int { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2, cudaErrorNotReady = 600 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
#define cudaEventDisableTiming 0x02
#define cudaEventDefault 0x00
#define cudaStreamNonBlocking 0x01
#define cudaStreamDefault 0x00
#define cudaHostRegisterDefault 0x00
#define cudaHostAllocDefault 0x00
#define cudaHostAllocMapped 0x02
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
struct float2 { float x, y; };
inline float2 make_float2(float x, float y) { return float2{x, y}; }
struct CUgraphExec_st; typedef CUgraphExec_st* cudaGraphExec_t;
struct CUgraph_st; typedef CUgraph_st* cudaGraph_t;

namespace fakecuda {
inline bool async_mode() { static const bool v = [] { const char* s = getenv("FAKE_CUDA_ASYNC"); return s && *s && strcmp(s, "0") != 0; }(); return v; }
inline int delay_us() { static const int v = [] { const char* s = getenv("FAKE_CUDA_DELAY_US"); return s ? atoi(s) : 0; }(); return v; }
struct Counters {
  std::atomic<long> host_alloc{0}, host_free{0}, dev_alloc{0}, dev_free{0}, d2h{0}, h2d{0}, d2d{0}, events_live{0}, streams{0};
};
inline Counters& counters() { static Counters c; return c; }

struct Event {
  std::mutex mu; std::condition_variable cv;
  uint64_t recorded = 0, completed = 0;
  void complete(uint64_t t) { std::lock_guard<std::mutex> l(mu); if (t > completed) completed = t; cv.notify_all(); }
  void wait_for(uint64_t t) { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return completed >= t; }); }
};
struct EventHandle { std::shared_ptr<Event> e = std::make_shared<Event>(); };

struct Stream {
  std::mutex mu; std::condition_variable cv, idle;
  std::deque<std::function<void()>> q; bool stop = false, busy = false;
  std::thread th;
  Stream() { if (async_mode()) th = std::thread([this] { run(); }); ++counters().streams; }
  ~Stream() { { std::lock_guard<std::mutex> l(mu); stop = true; } cv.notify_all(); if (th.joinable()) th.join(); }
  void run() {
    for (;;) {
      std::function<void()> f;
      { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return stop || !q.empty(); });
        if (q.empty()) return; f = std::move(q.front()); q.pop_front(); busy = true; }
      f();
      { std::lock_guard<std::mutex> l(mu); busy = false; if (q.empty()) idle.notify_all(); }
    }
  }
  void push(std::function<void()> f) {
    if (!th.joinable()) { f(); return; }
    { std::lock_guard<std::mutex> l(mu); q.push_back(std::move(f)); } cv.notify_all();
  }
  void sync() { if (!th.joinable()) return; std::unique_lock<std::mutex> l(mu); idle.wait(l, [&] { return q.empty() && !busy; }); }
};
inline void enqueue(Stream* s, std::function<void()> f) { if (!s) f(); else s->push(std::move(f)); }
inline void copy(void* d, const void* s, size_t n, int kind, bool is_async) {
  if (is_async && delay_us() > 0) std::this_thread::sleep_for(std::chrono::microseconds(delay_us()));
  if (n) memcpy(d, s, n);
  if (kind == cudaMemcpyDeviceToHost) ++counters().d2h; else if (kind == cudaMemcpyHostToDevice) ++counters().h2d; else if (kind == cudaMemcpyDeviceToDevice) ++counters().d2d;
}
}  // namespace fakecuda

typedef fakecuda::Stream* cudaStream_t;
typedef fakecuda::EventHandle* cudaEvent_t;

inline const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "fake-success" : e == cudaErrorNotReady ? "fake-not-ready" : "fake-error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaMalloc(void** p, size_t n) {
  // FAKE_DEV_MALLOC_MAX=bytes — any single allocation larger than this fails (simulates VRAM exhaustion; read on every call — tests change it midway). Unset or empty = no limit.
  if (const char* lim = getenv("FAKE_DEV_MALLOC_MAX"); lim && *lim && n > (size_t)strtoull(lim, nullptr, 10)) { *p = nullptr; return cudaErrorMemoryAllocation; }
  *p = aligned_alloc(256, (n + 255) / 256 * 256); if (!*p) return cudaErrorMemoryAllocation;
  memset(*p, 0xCD, n); ++fakecuda::counters().dev_alloc; return cudaSuccess;
}
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
inline cudaError_t cudaFree(void* p) { if (p) { free(p); ++fakecuda::counters().dev_free; } return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* fr, size_t* tot) { *fr = 64ull << 30; *tot = 96ull << 30; return cudaSuccess; }
struct cudaDeviceProp { char name[256]; int major, minor; };  // hived start-up fitness check (reports a Blackwell-class device)
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { *p = cudaDeviceProp{}; const char* n = "fake"; for (int i = 0; n[i]; ++i) p->name[i] = n[i]; p->major = 12; p->minor = 0; return cudaSuccess; }
inline cudaError_t cudaHostAlloc(void** p, size_t n, unsigned) { *p = malloc(n ? n : 1); if (!*p) return cudaErrorMemoryAllocation; ++fakecuda::counters().host_alloc; return cudaSuccess; }
inline cudaError_t cudaMallocHost(void** p, size_t n) { return cudaHostAlloc(p, n, 0); }
inline cudaError_t cudaFreeHost(void* p) { if (p) { free(p); ++fakecuda::counters().host_free; } return cudaSuccess; }
inline cudaError_t cudaHostRegister(void*, size_t, unsigned) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { if (d) *d = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }
inline cudaError_t cudaHostGetDevicePointer(void** d, void* h, unsigned) { *d = h; return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = new fakecuda::Stream; return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { return cudaStreamCreateWithFlags(s, 0); }
// HIVE_DECODE_COPY_PRIO: priorities mean nothing on fake streams (ordering comes only from streams and events — the production code does not rely on priorities either)
inline cudaError_t cudaDeviceGetStreamPriorityRange(int* least, int* greatest) { if (least) *least = 0; if (greatest) *greatest = -5; return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned f, int) { return cudaStreamCreateWithFlags(s, f); }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) { delete s; return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) { if (s) s->sync(); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = new fakecuda::EventHandle; ++fakecuda::counters().events_live; return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { return cudaEventCreateWithFlags(e, 0); }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { if (e) { delete e; --fakecuda::counters().events_live; } return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t h, cudaStream_t s = nullptr) {
  std::shared_ptr<fakecuda::Event> e = h->e; uint64_t t;
  { std::lock_guard<std::mutex> l(e->mu); t = ++e->recorded; }
  fakecuda::enqueue(s, [e, t] { e->complete(t); });
  return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t h) { std::lock_guard<std::mutex> l(h->e->mu); return h->e->completed >= h->e->recorded ? cudaSuccess : cudaErrorNotReady; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t h) { uint64_t t; { std::lock_guard<std::mutex> l(h->e->mu); t = h->e->recorded; } h->e->wait_for(t); return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0.f; return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t h, unsigned = 0) {
  std::shared_ptr<fakecuda::Event> e = h->e; uint64_t t;
  { std::lock_guard<std::mutex> l(e->mu); t = e->recorded; }
  fakecuda::enqueue(s, [e, t] { e->wait_for(t); });
  return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, int kind) { fakecuda::copy(d, s, n, kind, false); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, int kind, cudaStream_t st = nullptr) {
  const bool a = st != nullptr && fakecuda::async_mode();
  fakecuda::enqueue(st, [d, s, n, kind, a] { fakecuda::copy(d, s, n, kind, a); });
  return cudaSuccess;
}
// HIVE_STAGE_COPY2D: a 2D copy = one row copy per height row (same stream order)
inline cudaError_t cudaMemcpy2DAsync(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, int kind, cudaStream_t st = nullptr) {
  const bool a = st != nullptr && fakecuda::async_mode();
  fakecuda::enqueue(st, [d, dp, s, sp, w, h, kind, a] { for (size_t i = 0; i < h; ++i) fakecuda::copy((uint8_t*)d + i * dp, (const uint8_t*)s + i * sp, w, kind, a); });
  return cudaSuccess;
}
inline cudaError_t cudaMemset(void* d, int v, size_t n) { memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t st = nullptr) { fakecuda::enqueue(st, [d, v, n] { memset(d, v, n); }); return cudaSuccess; }
