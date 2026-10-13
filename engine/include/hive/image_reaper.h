// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Deferred release of session-image memory (all families). Dropping a long session's image on the engine thread ran munmap / delete[] of
//   GBs of populated pages there — measured 2026-10-11 on hived_q27 (the bench's 20 prefill64k prompts): once the images passed the host
//   budget every prompt checkpoint evicted the oldest image and the engine stood still ~400 ms between the prefill and the first decode step
//   (nsys: GPU idle, the main thread in no CUDA / traced libc call — glibc's internal munmap). Large pieces are released on one background
//   thread instead.
// Safety rules for callers: hand memory over only through the deleter of the LAST reference (an async copy job must hold a reference —
//   a shared_ptr — to every piece it writes until the copy has completed); never for cudaHostRegister'ed ranges. Bounded: more than 8 GiB
//   waiting → the caller frees inline (logged), so a burst cannot hold tens of GB of released RAM against the host budget. The thread drains
//   and joins at exit (atexit); the object is never destroyed, so a release during static destruction is still safe. Header-only (one
//   instance per process: function-local static of an inline function); no CUDA dependency (pinned memory is not handled here).
#pragma once
#include <sys/mman.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace hive {

struct ImageReaper {
  enum Kind : int { kDelete = 0, kMunmap = 1 };
  static constexpr size_t kMaxQueued = (size_t)8 << 30;
  static constexpr size_t kMinDeferred = (size_t)8 << 20;   // smaller pieces are freed at once (cheap)
  struct Item { void* p; size_t n; int kind; };
  std::mutex mu;
  std::condition_variable cv;
  std::vector<Item> q;
  size_t queued = 0;
  bool stop = false;
  std::thread th;
  static void release(const Item& it) {
    if (it.kind == kMunmap) munmap(it.p, it.n);
    else delete[] static_cast<uint8_t*>(it.p);
  }
  ImageReaper() {
    th = std::thread([this] {
      for (;;) {
        std::vector<Item> v;
        { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return stop || !q.empty(); }); if (q.empty() && stop) return; v.swap(q); }
        size_t freed = 0;
        for (const Item& it : v) { release(it); freed += it.n; }
        std::lock_guard<std::mutex> lk(mu);
        queued -= freed;
      }
    });
  }
  void drain() { { std::lock_guard<std::mutex> lk(mu); stop = true; } cv.notify_one(); if (th.joinable()) th.join(); }
  void put(void* p, size_t n, int kind) {
    if (n >= kMinDeferred) {
      std::lock_guard<std::mutex> lk(mu);
      if (!stop && queued + n <= kMaxQueued) { q.push_back({p, n, kind}); queued += n; cv.notify_one(); return; }
    }
    if (n >= kMinDeferred) {
      static std::atomic<int> warned{0};
      if (warned++ < 4) fprintf(stderr, "[hive] image reaper: %zu MiB already waiting — freeing %zu MiB on the caller\n", queued >> 20, n >> 20);
    }
    release({p, n, kind});
  }
};
inline ImageReaper*& image_reaper_ptr() { static ImageReaper* r = nullptr; return r; }
inline ImageReaper& image_reaper() {   // never destroyed; drained and joined at exit
  static ImageReaper* r = [] { image_reaper_ptr() = new ImageReaper; std::atexit([] { image_reaper_ptr()->drain(); }); return image_reaper_ptr(); }();
  return *r;
}
// n bytes of image memory (new[]) whose release goes through the reaper
inline std::shared_ptr<uint8_t> reaped_bytes(size_t n) {
  uint8_t* p = new uint8_t[n ? n : 1];
  if (n < ImageReaper::kMinDeferred) return std::shared_ptr<uint8_t>(p, std::default_delete<uint8_t[]>());
  return std::shared_ptr<uint8_t>(p, [n](uint8_t* q) { image_reaper().put(q, n, ImageReaper::kDelete); });
}

}  // namespace hive
