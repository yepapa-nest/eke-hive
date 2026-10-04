// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_EARLY_ROUTE — hands a decode layer's routing to the host *before* the shared expert runs
//   (runtime.cpp decode_layer · moe_router_shared).
//   Pure helper that compiles without CUDA: the CPU test (engine/tests/test_early_route_cpu.cpp — fake GPU thread + TSAN) and the
//   GPU test (engine/tests/test_early_route.cu) call **the same code as production**.
//
// Default path (decode_layer): graph A = [previous layer tail · engram · attention · hc · router · shared expert · route_to_host
//   (routing + activations)] → the host spin-syncs until A ends → classify · issue DMA · unpack CPU activations · start the pool ·
//   H2D the tables · launch GPU experts. The GPU idles in between ([decode-host] sync + prep + launch — M=1 sample: 0.83 ms/step of
//   the 1.44 ms/step gpu-idle once the 0.61 ms tail is excluded; batch sample: prep 3.14 + launch 1.09 ms/step).
// This path: er_route_post (decode_handshake.cu — copies routing to mapped host memory + posts a sequence number) is inserted right
//   after the router; only the activations are still copied at the end of A.
//   The host wakes on the posted number alone (a mapped u32) and, while the shared expert runs, finishes classification, DMA issue,
//   the CPU work table, the table H2D and the GPU expert launches — those launches are ordered after A on the stream (same kernels,
//   same inputs, same order → bit-identical). Only the CPU unpack and pool start, which need the activations, wait for the end of A
//   (event).
//
// (1) Gate — wait for the posted number. On every post the GPU increments a device counter and writes the value to the mapped u32
//     (routing writes → __threadfence_system → number).
//     Host seq = the last number received. Expected value for this launch = seq + 1.
//       · number == expected → the routing is on the host (kOk).
//       · the front graph has finished (front_done — non-blocking query) but number != expected → mismatch. A post precedes the end of
//         the front graph in stream order, so the number seen then is the last post: number > expected → kResync (an earlier launch
//         went out of step), number < expected → kMissing (this launch posted nothing — structurally impossible).
//         The caller *absorbs* both (waits for the end of the front graph, copies the routing again and sets seq to the number — the
//         same values as the default path).
//     begin() checks whether the previous launch was abandoned without being waited on (e.g. an exception mid-layer): if so it returns
//     false — the number is not trusted afterwards and the caller uses end-of-front + re-copy.
//     There is no timeout: end-of-front (front_done) always arrives (on error cudaEventQuery returns the error and the caller raises it
//     via CUDA_CHECK — same as a plain synchronize).
//
// (2) run_layer — the host order for one layer (exactly the body below). Invariants:
//       · launch(l) (classify · tables · GPU expert launch) comes after wait_route(l) (routing arrived)
//       · start_cpu(l) (activation unpack + pool start) comes after wait_front(l) (activations arrive at the end of A)
//       · prep_next(l) (writes the next layer's mapped row table — HOST_FAST) comes after wait_front(l) (front graph A(l), which reads
//         that table, has finished — the precondition HOST_FAST relies on)
//       · finish(l) (CPU wait · accumulate) comes after start_cpu(l) · the next layer's front(l+1) comes after finish(l) (caller loop)
//     ⚠️ Reordering is caught by the CPU test (the fake GPU thread checks markers at execution time) and its mutation negative controls
//     (tools/test_early_route_cpu.py).
#pragma once
#include <atomic>
#include <cstdint>

namespace hive {
namespace er {

enum Wait : int { kOk = 0, kResync = 1, kMissing = 2 };

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}
inline uint32_t load_acq(const uint32_t* p) { return std::atomic_ref<uint32_t>(*const_cast<uint32_t*>(p)).load(std::memory_order_acquire); }

struct Gate {
  uint32_t seq = 0;    // last posted number received (or resynced to)
  bool open = false;   // launched but not yet closed by wait/resync
  long n_ok = 0, n_resync = 0, n_missing = 0, n_untrusted = 0;
  // Called just before a launch. false = the previous launch was never closed (exception path) → do not trust the number for this
  // launch; use end-of-front + re-copy (untrusted()).
  bool begin() {
    const bool ok = !open;
    open = true;
    if (!ok) ++n_untrusted;
    return ok;
  }
  // front_done(): has the front graph finished (non-blocking query)? Asked once every poll_every spins.
  template <class Done>
  int wait(const uint32_t* flag, Done&& front_done, unsigned poll_every = 256) {
    const uint32_t want = seq + 1u;
    for (unsigned it = 1;; ++it) {
      if (load_acq(flag) == want) { seq = want; open = false; ++n_ok; return kOk; }
      if (it % poll_every == 0 && front_done()) {
        const uint32_t g = load_acq(flag);  // after end-of-front every post up to this launch is visible
        if (g == want) { seq = want; open = false; ++n_ok; return kOk; }
        const int r = (int32_t)(g - want) > 0 ? kResync : kMissing;
        seq = g; open = false;
        ++(r == kResync ? n_resync : n_missing);
        return r;
      }
      cpu_relax();
    }
  }
  // After the caller has waited for end-of-front, accept the number as is — for untrusted launches and after absorbing kResync/kMissing
  void resync(const uint32_t* flag) { seq = load_acq(flag); open = false; }
};

// Ops requirements (all called on the host — asynchronous work is enqueued on st_ and the call returns):
//   prep(l)       host tables for layer l (engram · attention row tables — mapped pinned) · skipped when prepped (HOST_FAST filled
//                 them during the previous layer)
//   front(l)      launch graph A (tail of layer l−1 + front of layer l, including the post after the router) + record the end-of-front event
//   wait_route(l) wait for the posted number (Gate) — routing arrived
//   launch(l)     classify · issue DMA · CPU work table (activation unpack deferred) · table H2D · GPU expert launch (after A in stream order)
//   wait_front(l) wait for the end-of-front event — activations arrived · layer l's mapped row table is free
//   start_cpu(l)  deferred activation unpack + CPU pool start (no-op when the CPU has no share)
//   prep_next(l)  HOST_FAST: host tables for layer l+1 (no-op when off)
//   finish(l)     wait for CPU completion · launch accumulation of the CPU results
template <class Ops>
void run_layer(int l, bool prepped, Ops& o) {
  if (!prepped) o.prep(l);
  o.front(l);
  o.wait_route(l);
  o.launch(l);
  o.wait_front(l);
  o.start_cpu(l);
  o.prep_next(l);
  o.finish(l);
}

}  // namespace er
}  // namespace hive
