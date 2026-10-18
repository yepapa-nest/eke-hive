// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_VRAM_LIMIT_MB for the whole life of the process, not only at startup.
//
// Why: the limit is emulated by a ballast that holds the rest of the card. Startup sizing (VRAM plan, session pool, cache fit) ran
//   under it, but runtime allocations (DevBuf, GrowBuf growth) still came from the device-wide free memory. With another process on
//   the card that free memory is that process's: on 2026-10-18 06:25 hived (72 GB limit, ballast released) took a 2 MiB buffer with
//   26.3 MiB left on the card beside a 25.6 GB neighbour and aborted (cudaMalloc out of memory → DevBuf abort).
// How: the startup ballast is held as kChunk pieces; at ready it is swapped for pieces of the headroom left inside the limit (rebase), so
//   the rest of the card goes back to other processes. Once serving (arm: cap on), a runtime allocation first frees as many of the process's own
//   pieces as it needs and a runtime free refills them, so the process keeps one constant footprint and never takes memory a
//   neighbour freed. The refill is bounded by a target (the reserve when armed, minus runtime allocations net of frees): a first cut
//   refilled by a running credit, and a refused 1 TiB request credited 1 TiB and took every free byte of the card (GPU test, 10-18).
//   Sleep turns the cap off (what sleep frees belongs to the media engines, and wake takes it back from the card) and a successful
//   wake turns it back on. Without a limit, or after HIVE_VRAM_BALLAST_RELEASE, there are no pieces and nothing changes.
#pragma once
#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

#ifndef HIVE_FAKE_CUDA
#include <cuda_runtime.h>
#endif

namespace hive {
namespace vram_reserve {

constexpr size_t kChunk = 32ull << 20;

struct State {
  std::mutex mu;
  std::vector<void*> pieces;
  long long target = 0;  // bytes the reserve should hold: its size when armed, minus what runtime allocations have taken since (net of frees)
  std::atomic<bool> cap{false};
};
inline State& st() { static State s; return s; }

inline size_t held() { std::lock_guard<std::mutex> g(st().mu); return st().pieces.size() * kChunk; }
inline bool cap_on() { return st().cap.load(std::memory_order_relaxed); }
inline void set_cap(bool on) { st().cap.store(on, std::memory_order_relaxed); }

#ifndef HIVE_FAKE_CUDA
inline bool piece_alloc(void** p) {
  if (cudaMalloc(p, kChunk) == cudaSuccess && *p) return true;
  (void)cudaGetLastError();
  return false;
}
inline void piece_free(void* p) { cudaFree(p); }
#else
inline bool piece_alloc(void**) { return false; }
inline void piece_free(void*) {}
#endif

// Unlocked: free pieces until at most `limit` bytes are held. Returns the bytes freed.
inline size_t shrink_to_locked(long long limit) {
  size_t freed = 0;
  while (!st().pieces.empty() && (long long)(st().pieces.size() * kChunk) > limit) {
    piece_free(st().pieces.back());
    st().pieces.pop_back();
    freed += kChunk;
  }
  return freed;
}
// Unlocked: take pieces while one more still fits under `limit` (stops at the first refusal). Returns the bytes added.
inline size_t grow_to_locked(long long limit) {
  size_t got = 0;
  while ((long long)((st().pieces.size() + 1) * kChunk) <= limit) {
    void* p = nullptr;
    if (!piece_alloc(&p)) break;
    st().pieces.push_back(p);
    got += kChunk;
  }
  return got;
}

// Startup ballast: take up to `want` bytes in pieces. Returns the bytes added.
inline size_t fill(size_t want) {
  std::lock_guard<std::mutex> g(st().mu);
  return grow_to_locked((long long)(st().pieces.size() * kChunk + want));
}
// Free pieces until at least `want` bytes are given back (or none are left) — the target is unchanged, so later frees refill them.
inline size_t give(size_t want) {
  std::lock_guard<std::mutex> g(st().mu);
  const long long h = (long long)(st().pieces.size() * kChunk);
  return shrink_to_locked(h - (long long)want);
}
// Ready: swap the startup ballast (the rest of the card, outside the limit) for the process's headroom inside the limit (`headroom` = what is
//   free while the ballast is held). The new pieces are taken first and the ballast freed after, so nothing can take the headroom in between.
//   Holding the ballast for the whole run left a neighbour no room to grow after this process started (a co-located server that starts one
//   more 644 MiB worker on its first request would have been refused). Returns the bytes now held.
inline size_t rebase(size_t headroom) {
  std::lock_guard<std::mutex> g(st().mu);
  const size_t n_old = st().pieces.size();
  grow_to_locked((long long)(n_old * kChunk + headroom));
  for (size_t i = 0; i < n_old; ++i) piece_free(st().pieces[i]);
  st().pieces.erase(st().pieces.begin(), st().pieces.begin() + (long long)n_old);
  return st().pieces.size() * kChunk;
}
// Serving starts (or resumes after wake): the reserve as it is now is the process's share, and runtime allocations come out of it.
inline void arm() {
  std::lock_guard<std::mutex> g(st().mu);
  st().target = (long long)(st().pieces.size() * kChunk);
  st().cap.store(!st().pieces.empty(), std::memory_order_relaxed);
}
inline size_t release_all() {
  std::lock_guard<std::mutex> g(st().mu);
  st().cap.store(false, std::memory_order_relaxed);
  st().target = 0;
  return shrink_to_locked(0);
}

// Runtime allocation of `bytes` is about to happen (cap on): its room comes out of the reserve. False = it would take the process past
//   its limit (more than the reserve's share) — the caller refuses it without asking the device, whose free memory belongs to other processes.
inline bool before_alloc(size_t bytes) {
  if (!cap_on() || !bytes) return true;
  std::lock_guard<std::mutex> g(st().mu);
  if (st().target < (long long)bytes) return false;
  st().target -= (long long)bytes;
  shrink_to_locked(st().target);
  return true;
}
// The runtime allocation of `bytes` announced by before_alloc did not happen: the reserve takes its room back (never past the target).
inline void alloc_refused(size_t bytes) {
  if (!cap_on() || !bytes) return;
  std::lock_guard<std::mutex> g(st().mu);
  st().target += (long long)bytes;
  grow_to_locked(st().target - (long long)kChunk);
}
// A runtime free of `bytes` happened (cap on): the reserve refills up to its target, so a neighbour's later allocation is not ours to take.
//   The target bounds it — the reserve never grows past the share it was armed with, whatever the call sequence. Refills stop one piece
//   short of the target: small buffers then come and go inside that slack instead of freeing and re-taking a piece every time.
inline void after_free(size_t bytes) {
  if (!cap_on() || !bytes) return;
  std::lock_guard<std::mutex> g(st().mu);
  st().target += (long long)bytes;
  grow_to_locked(st().target - (long long)kChunk);
}

}  // namespace vram_reserve

}  // namespace hive
