// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>

#include <pthread.h>  // O1 engine-thread pinning (pthread_setaffinity_np)
#include <sched.h>

#include "hive/attn_decode_fused.h"  // D2 HIVE_DECODE_ATTN_FUSED
#include "hive/decode_overlap.h"  // O1 HIVE_DECODE_HOST_FAST, HIVE_DECODE_UBATCH (batching helpers, half-pipeline order)
#include "hive/clock.h"
#include "hive/batch_miss.h"    // B2 batched decode misses (CPU_FIRST, STAGE_HIT, PREFETCH — host-only logic)
#include "hive/decode_split.h"  // C1 HIVE_DECODE_SPLIT=balance cost model for sharing decode misses (host-only)
#include "hive/decode_dispatch.h"  // G1 HIVE_DECODE_STEP_GRAPH (dispatcher — includes decode_handshake.h)
#include "hive/engram_ssd.h"  // HIVE_ENGRAM_SSD (engram table SSD offload, host-only), HIVE_ENGRAM_DIGEST
#include "hive/pregate.h"      // P1 HIVE_DECODE_PREGATE (pre-gate, owned prefetch — host rules)
#include "hive/early_route.h"  // S1 HIVE_DECODE_EARLY_ROUTE (published-sequence wait, per-layer host order — host-only)
#include "hive/decode_attn2.h"  // A1 HIVE_DECODE_ATTN2: bandwidth-oriented router, q_b, wo_b, shared expert (bit-identical) — attention_decode_dev, moe_router_shared
#include "hive/decode_attn3.h"  // A2 HIVE_DECODE_ROUTER3 (moe_router_shared), HIVE_DECODE_HCMIX3 (hc_attn_pre/hc_ffn_pre FUSE2); QKV3/SPARSE3 live inside the D2 front
#include "hive/gemv_decode.h"  // D2b HIVE_DECODE_GEMV2 (+HIVE_DECODE_PDL): bandwidth-oriented decode dense GEMV, router, hc mix — bit-identical to the base functions
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/moe_grouped_plan.h"  // H1 HIVE_GROUPED_PREFILL group plan (host-only)
#include "hive/prefill_split.h"  // G2 HIVE_PREFILL_SPLIT=balance cost model (host-only)
#include "hive/short_prefill.h"  // R2 short-prefill (TTFT) decisions — short tail path, pre-copy conversion, short-chunk DMA kind (host-only)
#include "hive/verify_rows.h"    // R1 verify-row table formulas (shared by host, device and CPU tests)
#include "hive/verify_decode.h"  // R1 HIVE_MTP_VERIFY2, HIVE_MTP_BATCH: window indices, sequential compressor and state restore of the verify-row decode path

namespace hive {
namespace {
// Uniform switch parsing: on when set and the value is neither "0" nor empty (HENV forwards HIVE_* values verbatim, so =0 must not mean on)
bool env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }


// E3 HIVE_HC_DECODE_FUSED (env_on — read once by k::hc_decode_fused_on): the decode (M ≤ 8) hc mix kernel also runs sinkhorn. hc ≠ 4 falls back to the base path.
bool hc_dfused_on(int hc) { return hc == 4 && k::hc_decode_fused_on(); }
}  // namespace

// F2 HIVE_DECODE_PREDICT_EVAL (runtime.h comment): next-layer routing prediction, measurement only. Writes go to its own buffers on its own stream;
//   the main stream only gets an event recorded on it, so outputs are unchanged apart from the extra GPU work.
//   Measured (DS-V4.1, c1 + c4 chat): recall top-6 72.5 % / top-12 86.4 %; the first non-resident candidate was routed 63.1 % of the time.
//   Using it for B2 pre-copies was built and rejected (docs/performance.md 'Rejected'): every variant lost c1 10-27 % — waiting for the prediction breaks the
//   early-route overlap, computing it in the front graph adds a router to the critical path, and the copies (~0.67 ms a record vs a ~0.45 ms layer) only
//   arrived in time for 25 % of 18K issued because demand DMA and promotions already fill the link.
struct Runtime::PredEval {
  cudaStream_t st = nullptr;
  cudaEvent_t ready = nullptr, done = nullptr;
  float *scores = nullptr, *rw = nullptr;
  int32_t *ids = nullptr, *ids_h = nullptr;
  int* counter = nullptr;
  int pend_l = -1, pend_M = 0, pend_k = 0;
  long long layers = 0, real = 0, hit_k = 0, hit_2k = 0, miss = 0, miss_k = 0, miss_2k = 0;
  long long pf_n[3] = {0, 0, 0}, pf_used[3] = {0, 0, 0};  // prefetch candidates (first 1/2/4 non-resident predictions, rank-interleaved over rows) and how many were routed
};


namespace {
template <class T>
T* pinned(size_t n) {
  T* p;
  CUDA_CHECK(cudaHostAlloc((void**)&p, n * sizeof(T), cudaHostAllocDefault));
  return p;
}
// Mapped pinned: kernels read host memory directly over PCIe — keeps small per-layer inputs from queueing behind the copy engine (shared with promotion DMA)
template <class T>
T* pinned_mapped(size_t n, T** dev) {
  T* p;
  CUDA_CHECK(cudaHostAlloc((void**)&p, n * sizeof(T), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)dev, p, 0));
  return p;
}
inline double now_ms() { return hive::mono_ms(); }
// e4m3 → fp32 table (256): lookup instead of bit manipulation when unpacking CPU-expert input activations per token
const float* e4m3_lut() {
  static const std::vector<float> t = [] { std::vector<float> v(256); for (int i = 0; i < 256; ++i) v[i] = e4m3_to_f32((uint8_t)i); return v; }();
  return t.data();
}
inline int next_pow2(int v) { int p = 1; while (p < v) p <<= 1; return p; }
// ---- Host helpers BEGIN — host-only (no CUDA). tools/test_snapshot_cpu.py slices from e4m3_lut to END and
//   tests it on the CPU via engine/tests/test_prefill_host_cpu.cpp (sequential == parallel bytes, capacity, ring-plan invariants).
// Split [0,n) into at most `threads` contiguous pieces and call f(i0,i1); piece 0 runs on the calling thread. With a single piece
//   (threads ≤ 1 or n < 2·min_per) f(0,n) runs once on the calling thread = the sequential path.
//   Pieces write disjoint rows (caller contract). Exceptions: all threads are joined, then the first one is rethrown.
inline void par_rows(int n, int threads, int min_per, const std::function<void(int, int)>& f) {
  const int t = std::max(1, std::min(threads, n / std::max(1, min_per)));
  if (t <= 1) { if (n > 0) f(0, n); return; }
  const int per = (n + t - 1) / t;
  std::exception_ptr err;
  std::mutex mu;
  auto run = [&](int a, int b) { try { f(a, b); } catch (...) { std::lock_guard<std::mutex> lk(mu); if (!err) err = std::current_exception(); } };
  std::vector<std::thread> th;
  th.reserve((size_t)t - 1);
  for (int k = 1; k < t; ++k) { const int a = k * per, b = std::min(n, a + per); if (a < b) th.emplace_back(run, a, b); }
  run(0, std::min(n, per));
  for (auto& x : th) x.join();
  if (err) std::rethrow_exception(err);
}
// Minimum rows per parallel piece, so that piece work stays larger than thread creation (tens of µs) — decode (M ≤ max_batch) is always sequential. Piece size not measured (performance only, does not affect values).
constexpr int kParMinRows = 64;
// L2 engram row lookup: hashed row for row m, column col → copy vals/scales (bytes as is). threads ≤ 1 = the plain sequential loop (range check during the copy).
//   Parallel: check all ranges first (same message), then copy row pieces — each row has its own destination, so the bytes are identical regardless of order.
inline void engram_gather(const int64_t* hashes, int M, int nL, int hidx, int cols, int hd, int64_t n_rows, const uint8_t* vals, const uint8_t* scales,
                          uint8_t* v_out, uint8_t* s_out, int threads) {
  auto body = [&](int m0, int m1) {
    for (int m = m0; m < m1; ++m)
      for (int col = 0; col < cols; ++col) {
        int64_t row = hashes[((size_t)m * nL + hidx) * cols + col];
        HIVE_CHECK(row >= 0 && row < n_rows, "engram row range");
        memcpy(v_out + ((size_t)m * cols + col) * hd, vals + (size_t)row * hd, hd);
        memcpy(s_out + ((size_t)m * cols + col) * (hd / 32), scales + (size_t)row * (hd / 32), hd / 32);
      }
  };
  if (threads <= 1) { body(0, M); return; }
  for (int m = 0; m < M; ++m)
    for (int col = 0; col < cols; ++col) {
      const int64_t row = hashes[((size_t)m * nL + hidx) * cols + col];
      HIVE_CHECK(row >= 0 && row < n_rows, "engram row range");
    }
  par_rows(M, threads, kParMinRows, body);
}
// Unpack CPU-expert input activations (fp8 → fp32 table lookup, e8m0 scales) for rows [i0,i1). Rows are independent — identical bytes when parallel (L1).
inline void unpack_acts(const uint8_t* xq, const uint8_t* xs, int i0, int i1, int dim, float* a_f, float* a_s) {
  const float* lut = e4m3_lut();
  for (int i = i0; i < i1; ++i) {
    for (int d = 0; d < dim; ++d) a_f[(size_t)i * dim + d] = lut[xq[(size_t)i * dim + d]];
    for (int b = 0; b < dim / 32; ++b) a_s[(size_t)i * (dim / 32) + b] = e8m0_to_f32(xs[(size_t)i * (dim / 32) + b]);
  }
}
// L3 indexer sub-chunk row count (HIVE_IDX_MSUB_ACTUAL): the constructor sizes Msub = budget / max_ctx (worst-case T). With this call's actual T,
//   process more rows at once within the same scratch capacity — iscore/iscore_f (rows × columns, columns ≤ max(T, ⌈T/bs⌉·bs), including the
//   candidate-pool path), bmax (rows × blocks), topk_pos (rows × selection width).
//   Capacity = the constructor allocation (Msub·(Tcap+8), Msub·nblocks_cap, Msub·topk_w). Kernel grid.y = row count, hence the 65535 cap. The result is
//   never smaller than Msub (columns, blocks and width are all ≤ worst case) → only the batch size of row-independent kernels changes; values are the same.
inline int idx_msub_for(int Msub, int Tcap, int nblocks_cap, int topk_w, int M, int T, int bs, int CB, int index_topk) {
  const int nblocks = (T + bs - 1) / bs;
  const long long cols = std::max<long long>(T, (long long)nblocks * bs);
  const long long used_w = std::max(std::min(index_topk, T), std::min(CB, nblocks));
  long long r = (long long)Msub * (Tcap + 8) / std::max(1LL, cols);
  r = std::min(r, (long long)Msub * nblocks_cap / std::max(1, nblocks));
  r = std::min(r, (long long)Msub * topk_w / std::max(1LL, used_w));
  r = std::min<long long>(r, std::min(M, 65535));
  return (int)std::max<long long>(Msub, r);
}
// L1 early-issue plan (HIVE_EARLY_STREAM): in the exact order the expert loop will consume (order), give ring slots to the leading streamed experts.
//   Slot = next++ % S — the same slot the loop would assign to that expert (the loop skips early-issued ones and continues with the rest). Stops:
//   (1) after one ring lap (S slots) — the previous occupant of the next slot has not yet had its release event recorded by this layer's loop (we would wait on a stale record)
//   (2) on reaching busy_slot (a pre-copy slot this layer's loop consumes first) — same reason.
inline void plan_early_stream(const std::vector<int>& order, const std::function<bool(int)>& streamed, const std::vector<char>& busy_slot, int S,
                              uint32_t& next, std::vector<std::pair<int, int>>& out) {
  out.clear();
  for (int e : order) {
    if ((int)out.size() >= S) break;
    if (!streamed(e)) continue;
    const int si = (int)(next % (uint32_t)S);
    if (busy_slot[(size_t)si]) break;
    ++next;
    out.push_back({e, si});
  }
}
// C2/H2/H3 tile plan (chunk of M rows → sub-chunk sizes, each ≤ Wm): {M} when M ≤ Wm. Otherwise the C2 rule — split into nearly equal sizes while making sure
//   the last sub-chunk is not smaller than the tail-mode minimum (need = max(decoder_tail+1, prefill_threshold)). Size checks (1 ≤ n ≤ Wm) are the caller's job.
inline void tile_plan(int M, int Wm, int need, std::vector<int>& out) {
  out.clear();
  if (M <= Wm) { out.push_back(M); return; }
  const int T = (M + Wm - 1) / Wm;
  int last = M - (T - 1) * ((M + T - 1) / T);
  if (last < need) last = std::min(Wm, need);
  const int rest = M - last, Ms = T > 1 ? (rest + T - 2) / (T - 1) : 0;
  for (int s = 0, off = 0; s < T; ++s) {
    const int n = s < T - 1 ? std::min(Ms, rest - off) : last;
    out.push_back(n); off += n;
  }
}
// H2 host tile count. on = HIVE_PREFILL_HOST_TILES (env_on), mb = HIVE_PREFILL_HOST_MB (host pinned budget — unset/empty = room for 2 tiles,
//   non-numeric or ≤ 0 = 0), h_bytes = h of one sub-chunk (max_chunk × hc × dim × 2), tile_ok = condition under which tiles can be on (tail-mode sub-chunk).
//   Upper bound = number of sub-chunks for a chunk not exceeding max_ctx (no larger chunk can arrive).
inline int host_tile_count(bool on, const char* mb, size_t h_bytes, bool tile_ok, int resident, int Wm, int64_t max_ctx) {
  if (!on || !tile_ok || h_bytes == 0 || Wm <= 0) return 0;
  int n = 2;
  if (mb && *mb) {
    const double v = atof(mb);
    n = std::isfinite(v) && v > 0 ? (int)std::min(1e6, v * 1048576.0 / (double)h_bytes) : 0;
  }
  const int64_t max_slots = (max_ctx + Wm - 1) / Wm;
  return (int)std::max<int64_t>(0, std::min<int64_t>(n, max_slots - resident));
}
// H3/H2 multi-unit plan: split each part's chunk Ms[p] with tile_plan and list units (sub-chunks) in part order, then sub-chunk order.
//   Slot = unit index (0 = Work, 1..resident−1 = resident tiles, then host tiles). A part with several sub-chunks needs a tail-mode last sub-chunk (tail_ok).
//   Returns false when slots run out or sizes do not fit (the caller treats it as a contract violation).
struct MultiUnitPlan { int part = 0, M = 0, off = 0, slot = 0; bool last = false; };
inline bool plan_units(const std::vector<int>& Ms, int Wm, int need, int slots, const std::function<bool(int)>& tail_ok, std::vector<MultiUnitPlan>& out) {
  out.clear();
  std::vector<int> t;
  for (int p = 0; p < (int)Ms.size(); ++p) {
    if (Ms[p] < 1) return false;
    tile_plan(Ms[p], Wm, need, t);
    if (t.size() > 1 && !(tail_ok(t.back()) && tail_ok(Ms[p]))) return false;
    for (size_t s = 0, off = 0; s < t.size(); off += (size_t)t[s], ++s) {
      if (t[s] < 1 || t[s] > Wm || (int)out.size() >= slots) return false;
      out.push_back(MultiUnitPlan{p, t[s], (int)off, (int)out.size(), s + 1 == t.size()});
    }
  }
  return true;
}
// ---- Host helpers END
}  // namespace

// ---- Host tile staging BEGIN — H2: tools/cpu_fake/test_host_tiles.cpp slices this section and tests it with fake CUDA (async streams) and TSAN.
//   Host tile k keeps its h in pinned host_[k]. It is uploaded into a staging buffer (n_stage of them, device) only for the duration of a visit
//   (tail of layer l−1 + front of layer l). Ordering is by events only: upload (up_ H2D) ← [end of the previous user's download of that buffer (free_),
//   end of this tile's last download (stored_)]; compute (st_) ← end of upload (loaded_); download (down_ D2H, opposite direction to expert-streaming H2D)
//   ← end of compute (done_). The host thread never reads or writes host_.
//   run_pass: within one pass (unit order) upload right before a visit and download right after; whenever a buffer frees up, upload the **next** host
//   unit early (overlapping the previous unit's compute).
class HostStager {
 public:
  struct Item { int k = -1; size_t load = 0, store = 0; };  // k = host tile (-1 = resident unit); load/store = bytes to upload/download (0 = none: h about to be overwritten, or dead h)
  HostStager(cudaStream_t compute, int n_host, size_t bytes, int n_stage) : st_(compute), bytes_(bytes) {
    for (int k = 0; k < n_host; ++k) {  // pinned failure is absorbed: use the tiles allocated so far (the caller re-reads n_host())
      uint8_t* p = nullptr;
      if (cudaHostAlloc(reinterpret_cast<void**>(&p), bytes, cudaHostAllocDefault) != cudaSuccess || !p) { (void)cudaGetLastError(); break; }
      host_.push_back(p);
    }
    const int ns = host_.empty() ? 0 : std::max(1, std::min(n_stage, (int)host_.size()));
    for (int b = 0; b < ns; ++b) { buf_.emplace_back(bytes); dev_.push_back(buf_.back().p); }
    for (auto* v : {&loaded_, &done_, &free_}) { v->resize((size_t)ns); for (auto& e : *v) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); }
    stored_.resize(host_.size());
    for (auto& e : stored_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    free_rec_.assign((size_t)ns, 0); owner_.assign((size_t)ns, -1);
    stored_rec_.assign(host_.size(), 0); stage_of_.assign(host_.size(), -1);
    CUDA_CHECK(cudaStreamCreateWithFlags(&up_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&down_, cudaStreamNonBlocking));
  }
  ~HostStager() {
    sync();
    for (size_t b = 0; b < ext_.size(); ++b) if (ext_[b]) { buf_[b].p = nullptr; buf_[b].n = 0; }  // Q1: buffers inside the elastic region are not owned (the store frees them)
    for (auto* v : {&loaded_, &done_, &free_, &stored_}) for (auto e : *v) cudaEventDestroy(e);
    for (auto s : {up_, down_}) if (s) cudaStreamDestroy(s);
    for (auto p : host_) cudaFreeHost(p);
  }
  HostStager(const HostStager&) = delete;
  HostStager& operator=(const HostStager&) = delete;
  int n_host() const { return (int)host_.size(); }
  int n_stage() const { return (int)buf_.size(); }
  size_t bytes() const { return bytes_; }
  bool staged(int k) const { return stage_of_.at((size_t)k) >= 0; }
  const uint8_t* host(int k) const { return host_.at((size_t)k); }  // for tests (read only after a sync)
  // New forward: every host copy is dead (the first visit of this forward does not upload)
  void begin() { HIVE_CHECK(std::all_of(owner_.begin(), owner_.end(), [](int o) { return o < 0; }), "host stager busy"); stored_rec_.assign(host_.size(), 0); }
  // Upload k into a free staging buffer (load bytes, 0 = reserve only). Returns false when no buffer is free.
  bool stage_in(int k, size_t load) {
    HIVE_CHECK(!staged(k) && load <= bytes_, "host tile stage_in");
    int b = -1;
    for (size_t i = 0; i < owner_.size(); ++i) if (owner_[i] < 0) { b = (int)i; break; }
    if (b < 0) return false;
    if (free_rec_[(size_t)b]) CUDA_CHECK(cudaStreamWaitEvent(up_, free_[(size_t)b], 0));
    if (load) {
      HIVE_CHECK(stored_rec_[(size_t)k], "host tile h loaded before it was stored");
      CUDA_CHECK(cudaStreamWaitEvent(up_, stored_[(size_t)k], 0));
      CUDA_CHECK(cudaMemcpyAsync(dev_[(size_t)b], host_[(size_t)k], load, cudaMemcpyHostToDevice, up_));
    }
    CUDA_CHECK(cudaEventRecord(loaded_[(size_t)b], up_));
    owner_[(size_t)b] = k; stage_of_[(size_t)k] = b;
    return true;
  }
  // Before the compute stream uses k's buffer: make it wait for the end of the upload and hand out the buffer (DevBuf — the caller swaps it with TileSlot.h and returns it)
  DevBuf& use(int k) {
    const int b = stage_of_.at((size_t)k);
    HIVE_CHECK(b >= 0, "host tile not staged");
    CUDA_CHECK(cudaStreamWaitEvent(st_, loaded_[(size_t)b], 0));
    return buf_[(size_t)b];
  }
  DevBuf& buf_of(int k) { return buf_.at((size_t)stage_of_.at((size_t)k)); }
  // After the work issued so far on the compute stream: download store bytes (0 = discard — this h is never read again). Returns the buffer.
  void stage_out(int k, size_t store) {
    const int b = stage_of_.at((size_t)k);
    HIVE_CHECK(b >= 0 && buf_[(size_t)b].p == dev_[(size_t)b] && store <= bytes_, "host tile stage_out (buffer not returned)");
    CUDA_CHECK(cudaEventRecord(done_[(size_t)b], st_));
    if (store) {
      CUDA_CHECK(cudaStreamWaitEvent(down_, done_[(size_t)b], 0));
      CUDA_CHECK(cudaMemcpyAsync(host_[(size_t)k], dev_[(size_t)b], store, cudaMemcpyDeviceToHost, down_));
      CUDA_CHECK(cudaEventRecord(stored_[(size_t)k], down_));
      CUDA_CHECK(cudaEventRecord(free_[(size_t)b], down_));
      stored_rec_[(size_t)k] = 1;
    } else {
      CUDA_CHECK(cudaEventRecord(free_[(size_t)b], st_));
      stored_rec_[(size_t)k] = 0;
    }
    free_rec_[(size_t)b] = 1; owner_[(size_t)b] = -1; stage_of_[(size_t)k] = -1;
  }
  // After an exception: drop what is staged (released behind the compute stream) and mark all copies dead. The caller returns the buffers first.
  void release_all() {
    for (size_t k = 0; k < stage_of_.size(); ++k) if (stage_of_[k] >= 0 && buf_[(size_t)stage_of_[k]].p == dev_[(size_t)stage_of_[k]]) stage_out((int)k, 0);
    stored_rec_.assign(host_.size(), 0);
  }
  void sync() { for (auto s : {up_, down_}) if (s) CUDA_CHECK(cudaStreamSynchronize(s)); }
  // Q1 HIVE_CACHE_ELASTIC: rebind staging buffer b to p (inside the elastic region — not owned; nullptr = not yet available). The first call frees its own allocation.
  //   Called between forwards only (buffers returned, up_/down_ synchronized — runtime ElasticScope calls it only then).
  void rebind_stage(int b, void* p) {
    if (ext_.size() != buf_.size()) ext_.assign(buf_.size(), 0);
    if (!ext_[(size_t)b]) buf_[(size_t)b].free();
    buf_[(size_t)b].p = p; buf_[(size_t)b].n = p ? bytes_ : 0; dev_[(size_t)b] = p; ext_[(size_t)b] = 1;
  }
  // One pass: visit(i, staging buffer or nullptr) in items order. A host unit must be staged right before its visit (early or now) and is downloaded after it.
  //   Early uploads follow item order only — so the earliest staged item is always the next visit (never stuck holding every buffer).
  void run_pass(const std::vector<Item>& items, const std::function<void(size_t, DevBuf*)>& visit) {
    size_t next = 0;
    auto prefetch = [&] {
      for (; next < items.size(); ++next) {
        const Item& it = items[next];
        if (it.k < 0 || staged(it.k)) continue;
        if (!stage_in(it.k, it.load)) break;
      }
    };
    prefetch();
    for (size_t i = 0; i < items.size(); ++i) {
      const Item& it = items[i];
      if (it.k < 0) { visit(i, nullptr); continue; }
      if (!staged(it.k)) HIVE_CHECK(stage_in(it.k, it.load), "host tile stage_in (no free buffer)");
      DevBuf& b = use(it.k);
      visit(i, &b);
      stage_out(it.k, it.store);
      next = std::max(next, i + 1);
      prefetch();
    }
  }

 private:
  cudaStream_t st_ = nullptr, up_ = nullptr, down_ = nullptr;
  size_t bytes_ = 0;
  std::vector<uint8_t*> host_;
  std::vector<DevBuf> buf_;
  std::vector<void*> dev_;
  std::vector<char> ext_;  // Q1: buffer b is not owned (rebind_stage)
  std::vector<cudaEvent_t> loaded_, done_, free_, stored_;
  std::vector<char> free_rec_, stored_rec_;
  std::vector<int> owner_, stage_of_;
};
// ---- Host tile staging END

struct Runtime::Work {
  int M = 0;
  DevBuf h, hflat, mixes, rsq, pre_a, post_a, comb_a, pre_f, post_f, comb_f, pre_mix;
  DevBuf x, xn, xf, xq, xs, qr, qrn, qrq, qrs, q, kv, ckv, cscore, cout, latent, ik, iq, iw, iqp;
  DevBuf pos, gpos, visible, idx, o, og, ogq, ogs, attn_out;
  DevBuf scores, ids, rw, gate, up, y, yq, ys, eout, acc, ffn_out;
  DevBuf g_rows, g_xq, g_xs, g_rw, cpu_out, cpu_rows;
  DevBuf eg_v, eg_s, eg_vals, eg_q, eg_sc, eg_kv;
  bool eg_alias = false;  // R5 HIVE_ENGRAM_ALIAS: eg_kv points into q, eg_vals/eg_q/eg_sc into o (borrowed, not owned — detached before destruction to prevent a double free)
  ~Work() { if (eg_alias) for (DevBuf* b : {&eg_vals, &eg_q, &eg_sc, &eg_kv}) { b->p = nullptr; b->n = 0; } }
  DevBuf iscore, iscore_f, bmax, topk_pos, cand;
  DevBuf woa;  // P3 prefill wo_a bf16 scratch [o_groups·o_lora_rank, H·D/o_groups] (only when max_chunk > 8). cand [M, cand_topk_blocks] int32 = layer-20 candidate blocks (P1)
  DevBuf xlast, logits, next;
  DevBuf attn_S, attn_mx, attn_sum;  // tensor-core attention work buffers [M,H,win+topk] fp32, [M,H]
  DevBuf attn_pacc, attn_pm, attn_ps;  // decode split-K partial sums [Mb,H,S,D] f32, [Mb,H,S]
  DevBuf dec_tab, dec_cnt;  // D2 HIVE_DECODE_ATTN_FUSED: device row table [Mb] k::DecRow + k::DecSoA; split-merge counters [Mb·H] int (0 — the kernel resets them to 0)
  // Batched-decode row tables (mapped pinned): row = sequence
  k::KvRow* kvrows_h = nullptr; k::KvRow* kvrows_d = nullptr;
  const uint8_t** kptrs_h = nullptr; const uint8_t** kptrs_d = nullptr;  // per-row packed indexer key cache
  int32_t *trows_h = nullptr, *trows_d = nullptr, *dsti_h = nullptr, *dsti_d = nullptr;
  bf16 **ringp_h = nullptr, **ringp_d = nullptr;
  uint8_t **dstp_h = nullptr, **dstp_d = nullptr, **dstk_h = nullptr, **dstk_d = nullptr;  // per-row packed compressed KV / indexer key cache (write)
  float **skv_h = nullptr, **skv_d = nullptr, **ssc_h = nullptr, **ssc_d = nullptr;
  DevBuf valid;  // u8 [Mb] (written by the GPU)
  DevBuf logits_b;  // [Mb, V]
  DevBuf next_b;    // [Mb]
  int32_t* next_b_h = nullptr;
  DevBuf is_image, img_types, img_row_of, img_rows, emb_rows, iota;
  bf16* emb_h = nullptr;  // images: [M] i8, span type i8, row map i32, aligner rows [max_img_tokens, dim]
  int8_t* is_image_h = nullptr;
  int32_t *pos_d = nullptr, *g_rows_d = nullptr, *cpu_rows_d = nullptr, *gpos_d = nullptr, *visible_d = nullptr;
  float *g_rw_d = nullptr, *cpu_out_d = nullptr;
  bf16* emb_d = nullptr;
  int8_t* is_image_d = nullptr;
  uint8_t *eg_v_d = nullptr, *eg_s_d = nullptr;
  int32_t *route_ids_d = nullptr; float* rw_d = nullptr; uint8_t *xq_hd = nullptr, *xs_hd = nullptr;  // device pointers of mapped pinned memory
  int g_rows_cap = 0;  // row capacity of g_rows_h/g_rw_h (= sum of tile sub-chunks × k)
  int32_t *ids_h = nullptr, *route_ids_h = nullptr, *pos_h = nullptr, *gpos_h = nullptr, *visible_h = nullptr, *g_rows_h = nullptr, *cpu_rows_h = nullptr,
          *next_h = nullptr;
  float *rw_h = nullptr, *g_rw_h = nullptr, *cpu_out_h = nullptr, *a_f_h = nullptr, *a_s_h = nullptr, *scratch_h = nullptr;
  uint8_t *xq_h = nullptr, *xs_h = nullptr, *eg_v_h = nullptr, *eg_s_h = nullptr;
  k::GroupDesc* gdesc_h = nullptr; k::GroupDesc* gdesc_d = nullptr;  // decode grouped-expert table (mapped pinned) [min(M, threshold)·k]
  DevBuf gdesc_dev;  // device copy read by the kernel — reading mapped pinned memory over PCIe per block leaked 0.5–1.4 ms per layer (measured)
  uint8_t* tbl_h = nullptr; DevBuf tbl_dev;  // pinned scratch that sends [gdesc][rows][rw] in one copy + its device copy
  std::vector<int> rows_by_e;  // per-expert (m·k+j) sort buffer
  std::vector<ExpertStore::Job> cpu_jobs;  // this layer's CPU jobs (must live from start_jobs to wait_jobs, hence owned by Work)
  std::unique_ptr<k::GroupedMoe> gmoe;  // H1 HIVE_GROUPED_PREFILL group executor (table ring — built at the first prefill layer when on)
  int Msub = 0, Tcap = 0, nblocks_cap = 0;
  // ---- DSpark ----
  DevBuf mh, mhq, mhs, mx;       // target-layer hidden capture [mh_rows, ntgt·dim] bf16, fp8, scales; main_x [mh_rows, dim]
  int mh_rows = 0;               // capture row cap = max(win, Mb, verify rows)
  int32_t* mpos_h = nullptr; int32_t* mpos_d = nullptr;   // capture row positions [mh_rows]
  std::vector<bf16**> mringp_h, mringp_d;                 // per-stage row→ring pointers [Mb]
  bf16** mhid_h = nullptr; bf16** mhid_d = nullptr;       // row→seq.mtp_hidden [Mb]
  DevBuf dids, dconf, membed, dx;   // draft ids [B+1] i32, confidences [B] f32, markov embeddings [B, rank] bf16, head input x [B, dim] bf16
  int32_t* dids_h = nullptr; float* dconf_h = nullptr;
  DevBuf ring_snap;               // verify ring snapshot [L, Mv, D] bf16
  bf16** lringp_h = nullptr; bf16** lringp_d = nullptr;   // layer→ring pointers [L] (verify sequence)
  std::vector<DevBuf> vxn;        // attention input xn per ratio>1 kv-source layer [Mv, dim] bf16 (for rollback recomputation)
  std::vector<DevBuf> cstate_snap;  // per layer [2, ratio, D] f32 (kv and score state)
  DevBuf counters;                // last-block counters of fused kernels (int[16], zero-initialized — kernels reset them to 0); +3/+4 = E3 hc sinkhorn tail (before attention / before ffn)
  bf16** ringp1_h = nullptr; bf16** ringp1_d = nullptr;  // row→ring pointers [8] for the single-sequence path (eager execution, draft ring sync)
  // R1 verify decode path (HIVE_MTP_VERIFY2, HIVE_MTP_BATCH — allocated only when on)
  int32_t* vgrp_h = nullptr; int32_t* vgrp_d = nullptr;   // row → first row of its part (mapped pinned [Mb] — window_idxs_verify)
  bf16** vlringp_h = nullptr; bf16** vlringp_d = nullptr;  // layer→ring pointers of part p [kVParts·L] (read by snapshot/restore kernels after launch — one set per part)
  std::vector<DevBuf> vcsnap;     // [kVParts·|v_src2_|] per-part ratio>1 compressor state snapshots [2, ratio, D] f32
  std::vector<DevBuf> vckv;       // [|v_src2_|] compressor inputs of verify rows (kv [Mv, D] ‖ score [Mv, D]) f32 — rollback rewrites accepted rows from them as is
  // Q1 HIVE_CACHE_ELASTIC (off = on false, lists empty — never read): see the comment above Runtime::ElasticScope
  struct Elastic {
    bool on = false, big = false, bound = false, lendable = false;
    int S = 0, depth = 0;            // S = small-layout row count (forwards with rows ≤ S use the small layout); depth = nesting of big-layout forwards (forward → forward_multi)
    size_t bytes = 0;                // bytes placed in the elastic region (big layout + buffers used only by big forwards)
    uint8_t* base = nullptr;         // elastic region (tail of the store's slot area — own when it cannot be obtained)
    DevBuf own;                      // fallback: a separately allocated region when the store could not provide elastic slots (never lent = same VRAM as without the switch)
    struct Ent { DevBuf* b; size_t off, n; void* alt; size_t alt_n; };  // Work member — swapped with the idle layout (alt) (the small layout is owned by the member)
    std::vector<Ent> work;
    struct Fix { DevBuf* b; size_t off, n; };  // buffers only big forwards use (tile slots, H3 mbufs) — always inside the elastic region (not owned)
    std::vector<Fix> fixed;
    std::vector<std::pair<int, size_t>> stage;  // H2 host tile staging buffers (index, offset)
    uint64_t reclaims = 0, lends = 0;
    double reclaim_ms = 0;
  } el;
};

// ---- Q1 HIVE_CACHE_ELASTIC — lend prefill-only VRAM to the expert cache as slots while no prefill is running ------------------------------------
// What: Work's row-proportional buffers (h, q, o, x, xf, acc, gate/up/y … ≈ 6–7 GB at max_chunk 16K rows), the device part of C2/H2 tile slots, H2 staging
//   and H3 mbufs are used at full size only by prefill forwards (rows > S). Decode, verify, draft, MTP sync and short forwards with rows ≤ S use only the
//   first S rows. Hence two layouts:
//     small layout = separately allocated buffers sized for S rows (owned by the members) — always active outside big forwards (these are the addresses the
//                    decode CUDA graphs and the G1 step graph capture).
//     big layout   = addresses inside the elastic region (not owned) — the bytes occupied by n_el = ⌈bytes/record⌉ elastic slots at the tail of the store's
//                    slot area (expert_store.h elastic_reserve).
//   Head of a big forward (ElasticScope construction): if lent, synchronize st_ and side_ → store.elastic_reclaim() (evict residents of elastic slots =
//   table update; promotion copies headed there are only waited for and dropped) → switch Work to the big layout. End (destruction): back to the small
//   layout, and lend unless hived is in the middle of a multi-chunk prefill (prefill_pending_) (synchronize st_, side_ and staging → elastic_lend — the
//   elastic slots come back as free slots that promote/warm_cache fill). During a multi-chunk prefill, lending happens afterwards at the head of the
//   first forward with rows ≤ S, forward_batch or warm_cache (lend_if_idle — synchronizes only when lending conditions hold, which is rare), so that decode
//   steps between chunks do not pour promotions into slots that are about to be reclaimed. (promote_after_step has no hook because the CPU fake runtime
//   slices its source text — the head of forward_batch is the same point.)
// The layout depends only on the forward's row count (M ≤ S → small; M > S and forward_multi → big) — graphs for the same M always capture the same
//   addresses (M is part of the run_graph key).
// Lossless: same kernels, same inputs, same order (only buffer addresses differ); only the cache's resident set changes (every expert has the same weights and
//   computation — GPU slot when resident, otherwise CPU/DMA; that path difference is the same as an ordinary cache hit/miss — fp32 accumulation order).
//   The initial contents of big-layout buffers were undefined cudaMalloc memory before as well (never read before written).
// S = integer value of HIVE_CACHE_ELASTIC (≥ 2); "1" or non-numeric = default. Default = the most rows Work uses outside big forwards =
//   max(window (MTP sync mh_rows cap), max_batch, dspark_block + 1). Off when S ≥ prefill_threshold (the small layout must never take the prefill path
//   moe_experts_multi, which assumes capacity w.M).
struct Runtime::ElasticScope {
  Runtime& r;
  bool on = false;
  int ex = 0;
  ElasticScope(Runtime& rt, int M, bool force) : r(rt), ex(std::uncaught_exceptions()) {
    Work::Elastic& E = r.w_->el;
    if (!E.on) return;
    if (!force && M <= E.S && E.depth == 0) { lend_if_idle(r); return; }  // small forward: lend here once a multi-chunk prefill has finished (promotions fill it afterwards)
    on = true;
    if (E.depth++ > 0) return;
    reclaim(r);
    bind(r);
    view(r, true);
  }
  ~ElasticScope() {
    if (!on) return;
    Work::Elastic& E = r.w_->el;
    if (--E.depth > 0) return;
    view(r, false);
    if (std::uncaught_exceptions() > ex) return;  // exception: work may remain on the streams — the next idle point (head of a forward with rows ≤ S, forward_batch, warm_cache) lends after a sync
    lend_if_idle(r);
  }
  static void eg_alias_rows(Work& w, const Config& c, int rows) {  // same placement formula as the constructor's HIVE_ENGRAM_ALIAS (only the row count is the layout's)
    if (!w.eg_alias) return;
    const int cols = c.n_hash_cols(), hd = c.engram_head_dim;
    const size_t b_vals = align_up((size_t)rows * cols * hd * 2, 256), b_q = align_up((size_t)rows * cols * hd, 256);
    uint8_t* o = w.o.as<uint8_t>();
    w.eg_kv.p = w.q.p; w.eg_kv.n = (size_t)rows * c.dim * (c.hc + 1) * 2;
    w.eg_vals.p = o; w.eg_vals.n = (size_t)rows * cols * hd * 2;
    w.eg_q.p = o + b_vals; w.eg_q.n = (size_t)rows * cols * hd;
    w.eg_sc.p = o + b_vals + b_q; w.eg_sc.n = (size_t)rows * cols * hd / 32;
  }
  // End of the constructor (after all allocations): read the switch and split the layouts. Free big-layout buffers, allocate the small layout, then ask the store for elastic slots.
  static void setup(Runtime& r) {
    if (!env_on("HIVE_CACHE_ELASTIC")) return;
    Work& w = *r.w_;
    Work::Elastic& E = w.el;
    const Config& c = r.model_.cfg();
    const int M = w.M, k = c.n_act, thr = r.opt_.prefill_threshold;
    const int Sdec = std::max({c.window, std::max(1, std::min(r.opt_.max_batch, M)), c.dspark_block + 1});
    int S = Sdec;
    {
      const char* v = getenv("HIVE_CACHE_ELASTIC");
      char* end = nullptr;
      const long x = strtol(v, &end, 10);
      if (end != v && end && *end == 0 && x >= 2) S = (int)std::min<long>(x, 1 << 20);
      S = std::max(S, Sdec);
    }
    if (S >= thr || S >= M) {
      fprintf(stderr, "[runtime] ⚠️elastic cache off: small rows %d ≥ min(prefill_threshold %d, max_chunk %d)\n", S, thr, M);
      return;
    }
    const size_t Mrows = (size_t)std::max(M, std::min(std::max(r.opt_.max_batch, thr), M) * k);  // rows of the constructor's expert intermediate buffers (same formula)
    const size_t Srows = (size_t)std::max(S, std::min(std::max(r.opt_.max_batch, thr), S) * k);
    std::vector<std::pair<DevBuf*, int>> list;  // (member, 0 = proportional to M, 1 = proportional to Mrows)
    for (DevBuf* b : {&w.h, &w.x, &w.xn, &w.attn_out, &w.ffn_out, &w.xf, &w.xq, &w.xs, &w.qr, &w.qrn, &w.qrq, &w.kv, &w.latent, &w.ckv, &w.cscore,
                      &w.cout, &w.ik, &w.iq, &w.iw, &w.idx, &w.attn_mx, &w.attn_sum, &w.og, &w.ogq, &w.ogs, &w.scores, &w.acc, &w.cand})
      list.push_back({b, 0});
    // q, o: with the engram alias the small layout must also hold the alias (it does, the formula is proportional — if not, q and o stay as a single big layout)
    bool qo = true;
    if (w.eg_alias) {
      const int cols = c.n_hash_cols(), hd = c.engram_head_dim;
      const size_t kv_s = (size_t)S * c.dim * (c.hc + 1) * 2, o_s = align_up((size_t)S * cols * hd * 2, 256) + align_up((size_t)S * cols * hd, 256) +
                          (size_t)S * cols * hd / 32;
      qo = kv_s <= w.q.n / (size_t)M * S && o_s <= w.o.n / (size_t)M * S;
    }
    // Alias off: the four engram work buffers also get two layouts (only addresses move — the only reader is still engram_dev). Dead eg_v/eg_s are left alone (R5 contract: allocation lines kept, zero uses)
    if (!w.eg_alias) for (DevBuf* b : {&w.eg_vals, &w.eg_q, &w.eg_sc, &w.eg_kv}) list.push_back({b, 0});
    if (qo) { list.push_back({&w.q, 0}); list.push_back({&w.o, 0}); }
    for (DevBuf* b : {&w.gate, &w.up, &w.y, &w.yq, &w.ys, &w.eout, &w.g_xq, &w.g_xs}) list.push_back({b, 1});
    size_t off = 0, small = 0;
    auto place = [&](size_t n) { off = align_up(off, 256); const size_t o = off; off += n; return o; };
    for (auto& [b, kind] : list) {
      const size_t R = kind ? Mrows : (size_t)M, Rs = kind ? Srows : (size_t)S;
      if (!b->p || b->n == 0 || b->n % R != 0) continue;  // sizes that are not proportional (configuration-specific formulas) are left alone
      const size_t n = b->n, ns = n / R * Rs;
      E.work.push_back({b, place(n), n, nullptr, 0});
      b->free();
      b->alloc(ns);
      small += ns;
    }
    eg_alias_rows(w, c, S);
    for (TileSlot& t : r.tiles_)
      for (DevBuf* b : {&t.h, &t.pre_mix, &t.idx, &t.xq, &t.xs, &t.acc, &t.pre_f, &t.post_f, &t.comb_f})
        if (b->p && b->n) { E.fixed.push_back({b, place(b->n), b->n}); b->free(); }
    for (MemberBufs& m : r.mbufs_)
      for (DevBuf* b : {&m.cand, &m.mh})
        if (b->p && b->n) { E.fixed.push_back({b, place(b->n), b->n}); b->free(); }
    if (r.stager_)
      for (int b = 0; b < r.stager_->n_stage(); ++b) { E.stage.push_back({b, place(r.stager_->bytes())}); r.stager_->rebind_stage(b, nullptr); }
    E.bytes = align_up(off, 256);
    E.S = S;
    E.on = true;
    r.work_addr_[0] = w.h.p; r.work_addr_[1] = w.acc.p;  // the small layout is now the "original address" (TileGuard/MultiGuard invariant)
    const int n_el = r.store_.elastic_reserve(E.bytes);
    fprintf(stderr, "[runtime] elastic cache on: small rows %d (%.0f MiB kept) · prefill-only %.0f MiB (%zu work + %zu tile/stage buffers) → %d elastic slots%s\n", S,
            small / 1048576.0, E.bytes / 1048576.0, E.work.size(), E.fixed.size() + E.stage.size(), n_el,
            r.store_.cache_deferred() ? " (placed when the deferred cache is sized)" : "");
    if (!r.store_.cache_deferred()) bind(r);  // store already allocated: bind now (if no elastic slots were granted, the store's separate region — the VRAM peak stays at startup)
  }
  static void bind(Runtime& r) {
    Work::Elastic& E = r.w_->el;
    if (!E.on || E.bound) return;
    E.base = r.store_.elastic_base();  // elastic slot region; if the store gave up on elasticity, a same-size region it allocated separately (never lent)
    E.lendable = E.base != nullptr && r.store_.elastic_slots() > 0 && (size_t)r.store_.elastic_slots() * r.store_.layout().total >= E.bytes;
    if (!E.base) {  // deferred store not allocated yet (big forward before alloc_cache — does not happen in hived's order): allocate separately (fallback, never lent)
      fprintf(stderr, "[runtime] ⚠️elastic cache: store region not ready — prefill work buffers allocated separately (%.0f MiB, no lending)\n", E.bytes / 1048576.0);
      E.own.alloc(E.bytes);
      E.base = E.own.as<uint8_t>();
      E.lendable = false;
    }
    for (auto& e : E.work) { e.alt = E.base + e.off; e.alt_n = e.n; }
    for (auto& f : E.fixed) { f.b->p = E.base + f.off; f.b->n = f.n; }
    for (auto& [b, o] : E.stage) r.stager_->rebind_stage(b, E.base + o);
    E.bound = true;
  }
  static void view(Runtime& r, bool big) {
    Work& w = *r.w_;
    Work::Elastic& E = w.el;
    if (E.big == big) return;
    for (auto& e : E.work) { std::swap(e.b->p, e.alt); std::swap(e.b->n, e.alt_n); }
    eg_alias_rows(w, r.model_.cfg(), big ? w.M : E.S);
    r.work_addr_[0] = w.h.p; r.work_addr_[1] = w.acc.p;
    E.big = big;
  }
  static void reclaim(Runtime& r) {
    Work::Elastic& E = r.w_->el;
    if (!r.store_.elastic_lent() || (E.bound && !E.lendable)) return;
    const double t0 = now_ms();
    CUDA_CHECK(cudaStreamSynchronize(r.st_));   // kernels that read elastic slots (st_)
    CUDA_CHECK(cudaStreamSynchronize(r.side_)); // copies issued by place() and demand DMA (side_)
    const int ev = r.store_.elastic_reclaim();
    ++E.reclaims;
    const double ms = now_ms() - t0;
    E.reclaim_ms += ms;
    const auto& st = r.store_.elastic_stats();
    fprintf(stderr, "[runtime] elastic reclaim #%llu: evicted %d · dropped promotions %llu (total) · %.2f ms · slots %d → %d\n", (unsigned long long)E.reclaims, ev,
            (unsigned long long)st.dropped, ms, r.store_.base_slots() + r.store_.elastic_slots(), r.store_.n_slots());
  }
  static void lend_if_idle(Runtime& r) {
    Work::Elastic& E = r.w_->el;
    // T11: never lend inside a layer-boundary yield (ly_.depth > 0) — the outer prefill's big layout (inter-layer state) lives in the elastic region (it was only switched to the small layout temporarily)
    if (!E.on || E.depth > 0 || E.big || r.prefill_pending_ || r.ly_.depth > 0 || r.store_.elastic_slots() == 0 || r.store_.elastic_lent() || (E.bound && !E.lendable)) return;
    CUDA_CHECK(cudaStreamSynchronize(r.st_));    // kernels that wrote to the big layout
    CUDA_CHECK(cudaStreamSynchronize(r.side_));
    if (r.stager_) r.stager_->sync();            // H2 staging uploads/downloads
    r.store_.elastic_lend();
    ++E.lends;
  }
  static void teardown(Runtime& r) {  // ~Runtime: back to the small layout (members free their own allocations) and detach non-owned buffers (the store frees the elastic region)
    Work::Elastic& E = r.w_->el;
    if (!E.on) return;
    view(r, false);
    for (auto& f : E.fixed) { f.b->p = nullptr; f.b->n = 0; }
  }
};


// ---- O1 decode overlap and instrumentation state -----------------------------------------------------------------------------------------------
// Goal: reduce decode GPU idle time (nsys kernel occupancy 58 %). Three parts — all off by default; off = the same calls in the same order as the base path.
//   A) Instrumentation (HIVE_PROFILE sample steps only — same decision as pmark): per layer, 7 events on st_ (before front launch, end of front, host wake-up,
//      right before the expert-table H2D, end of GPU experts, right before CPU accumulation, end of layer) plus host-clock intervals (prepare, launch,
//      follow-up, next layer's host tables), printed as one [decode-host] line at the end of the step (after st_ sync).
//      Events are created once at the first sample step and reused (not per step like pmark). The measure is GPU-clock gaps — an event marks when the
//      stream reached that point, so the gap "end of previous work → time the host issued the next work" = time st_ sat idle (when the stream was empty).
//      ⚠️With UBATCH the other half's work is on the same stream, so that gap is not idle time (the line is marked "ubatch" — read it as an upper bound).
//      The record calls themselves add a little host time to sample steps (7 per layer).
//   B) HIVE_DECODE_HOST_FAST: move work that does not affect the result off the critical path of moe_decode_experts(launch) (routing arrival → GPU expert
//      launch) — observe, xtrace and promotion slices (promo_pump) after the launch; only used experts instead of scanning all E (dov::group_routes_sparse —
//      same cnt and rows_by_e); unpack CPU activations only for rows going to the CPU; ship CPU row numbers with the expert-table H2D (one copy after the
//      CPU wait → only the accumulation kernel); prepare the next layer's host tables (engram_host, attention_decode_host) while this layer's experts run
//      (before the wait); pin the engine thread to CPUs of the GPU's NUMA node (HIVE_ENGINE_CPU=n pins to that single core).
//      Same experts, same devices (same resident/DMA/CPU classification inputs), same launch order, same accumulation order → bit-identical. LRU time
//      (use_total_) becomes smaller by R because observe moves after touch, but the order between slots is unchanged (victim selection only looks at order —
//      only the < comparison in expert_store.cpp).
//   C) HIVE_DECODE_UBATCH (M ≥ 2, forward_batch layer loop): split the batch into halves A=[0,⌈M/2⌉) and B=[⌈M/2⌉,M) in dov::run_ubatch order — while the
//      CPU pool works on A's misses, the GPU runs B's front (attention, router) and both halves' GPU experts. A half moves Work's row-based buffer pointers to
//      its row range (ub_view) and calls the base functions unchanged with M = half's row count (each half has its own graph key). Per-row math is the same;
//      what changes = expert grouping (an expert called by both halves forms two groups — GPU row computation is row-independent so values are equal,
//      test_decode_ubatch compares bits); DMA/CPU classification of missed experts and its order are per half (half's row and miss counts) → rows with
//      misses may compute the same expert on a different device (CPU fp32 LUT/GEMV vs GPU mma — low bits may differ; the same sequence's values already vary
//      like this with batch composition); if front kernels pick a different reduction order depending on M (e.g. cuBLAS gemm_f32/bf16 algorithm
//      selection), that layer's low bits may differ too (not verified on GPU). Outside the conditions (M < 2, dump/inject, expert tracing, verify forward)
//      the base layer loop runs. MTP verify goes forward_verify → forward (single-sequence path), so it never takes this loop (clean fallback); batched
//      decode MTP capture works on half rows as is (w.mh moves with the rows too).
struct Runtime::MoePend {  // state between moe_decode_experts(launch) → (start_cpu) → wait → accum
  std::vector<ExpertStore::Job>* jobs = nullptr;
  int job_rows = 0, kind = 0, slot = -1;  // slot = [decode-host] event slot (-1 outside sample steps)
  double t_cpu0 = 0;
  bool time_dma = false, started = false;
  const int32_t* rows_dev = nullptr;  // HOST_FAST: CPU row numbers shipped with the table H2D (device); nullptr = base path (copy cpu_rows_h → w.cpu_rows)
  const int32_t* cpu_rows_h = nullptr; const float* cpu_out_d = nullptr; float* acc = nullptr;  // pointers at launch time (UBATCH accumulation happens outside the half view)
  bool ds = false, ds_sample = false;  // C1 HIVE_DECODE_SPLIT: whether the model handled this layer (CPU table update), whether sample events were set (event at the accumulation point)
  int ds_cls = 0, ds_jobs = 0;
  // S1 HIVE_DECODE_EARLY_ROUTE: defer_unpack (input) = defer activation unpacking (activations arrive at the end of the front graph) — moe_decode_experts only
  //   leaves row markers in DecodeOverlap::need and sets unpack_pending. decode_layer unpacks the same way after the front ends and starts the pool (called with start_cpu = false).
  bool defer_unpack = false, unpack_pending = false;
  int unpack_M = 0;
  // T2 (S1 batched regression): er_tbl = issue the expert-table H2D on D.er_tbl_st instead of st_ (does not wait for front A) — only decode_layer's early path sets it (see the table-copy comment in moe_decode_experts)
  bool er_tbl = false;
  bool rowind = false;  // R1 HIVE_MTP_VERIFY_ROWIND: accumulate CPU results in fixed job-row order (moe_decode_accum)
};
struct Runtime::DecodeOverlap {
  enum { kPre, kFront, kWake, kExp, kGEnd, kAcc, kEnd, kPts };
  // ---- A) instrumentation
  bool on = false, ub = false;
  int nL = 0;
  std::vector<cudaEvent_t> ev;          // [slot = 2·l + half (+ 2·nL = head)][kPts]
  std::vector<uint8_t> used;            // [slot] bits of recorded points
  std::vector<double> h_prep, h_launch, h_post, cpu_ms;  // [slot] host ms
  std::vector<char> jobs_on;            // [slot] whether there was a CPU share
  double h_hprep = 0;                   // sum of per-layer host tables (engram_host + attention_decode_host)
  void begin(bool prof, int layers, bool ubatch) {
    on = prof; ub = ubatch; nL = layers; h_hprep = 0;
    if (!on) return;
    const size_t slots = (size_t)2 * layers + 2;
    if (ev.size() < slots * kPts) {
      const size_t old = ev.size();
      ev.resize(slots * kPts, nullptr);
      for (size_t i = old; i < ev.size(); ++i) CUDA_CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDefault));
    }
    used.assign(slots, 0); h_prep.assign(slots, 0); h_launch.assign(slots, 0); h_post.assign(slots, 0); cpu_ms.assign(slots, 0); jobs_on.assign(slots, 0);
  }
  int slot(int l, int h) const { return on ? 2 * l + h : -1; }
  void mark(int s, int pt, cudaStream_t st) {
    if (s < 0 || !on) return;
    CUDA_CHECK(cudaEventRecord(ev[(size_t)s * kPts + pt], st));
    used[s] |= (uint8_t)(1u << pt);
  }
  bool has(int s, int pt) const { return s >= 0 && (size_t)s < used.size() && (used[s] >> pt & 1); }
  float el(int s0, int p0, int s1, int p1) const {  // only when both were recorded (0 otherwise)
    if (!has(s0, p0) || !has(s1, p1)) return 0.f;
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev[(size_t)s0 * kPts + p0], ev[(size_t)s1 * kPts + p1]));
    return ms;
  }
  // ---- B) HOST_FAST: whether the next layer's host tables were prefilled
  int pre_l = -1, pre_Tmax = 0;
  std::vector<int> tmp;                 // group_routes_sparse sort scratch (R)
  std::vector<uint8_t> need;            // markers for rows going to the CPU (M)
  // ---- C) UBATCH
  cudaEvent_t front_done[2] = {nullptr, nullptr};
  std::vector<ExpertStore::Job> jobs[2];
  MoePend pend[2];
  std::vector<int> topk_rows[2];        // per-half shared_topk_rows_
  std::vector<int64_t> hashes[2];       // per-half seq_hashes_
  bool have_cand[2] = {false, false};   // per-half have_candidates_
  int Tb[2] = {0, 0};
  int view = -1;                        // half currently mapped into Work (-1 = original)
  bool dma_armed = false;               // a launch already armed the dma_e0_/e1_ sample (so the two halves do not overwrite the same events — always false on the default path)
  // ---- D) S1 HIVE_DECODE_EARLY_ROUTE (see the comment at the top of hive/early_route.h) — created at the first layer when on (outside capture; graphs capture the pointers by value)
  bool er_on = false;                   // switch (env_on)
  bool er_arm = false;                  // only while decode_layer calls the front body — moe_router_shared publishes after the router, activations only at the end
  uint32_t* er_flag_h = nullptr; uint32_t* er_flag_d = nullptr;  // published sequence number (mapped pinned)
  uint32_t* er_ctr_d = nullptr;         // device counter (4 B)
  cudaEvent_t er_front = nullptr;       // end of the front graph (spin wait — same method as cudaStreamSynchronize)
  er::Gate er_gate;
  long er_layers = 0, er_early = 0, er_fix = 0;  // [early-route] totals: layers, layers where the host issued GPU experts before the front ended, fallbacks (re-copies)
  cudaStream_t er_tbl_st = nullptr;     // T2: stream dedicated to the expert-table H2D (unordered with the front graph — see the table-copy comment in moe_decode_experts)
  cudaEvent_t er_tbl = nullptr;         // T2: end of that copy (st_ waits for it before the expert groups)
  long er_tbl_n = 0;                    // T2: [early-route] total — layers whose table went through the dedicated stream
  // ---- F) P1 HIVE_DECODE_PREGATE (see the comment at the top of hive/pregate.h) — early-route path only; created on first use (outside capture)
  bool pg_arm = false;                  // only while decode_layer calls the front graph — decode_front/hc_ffn_pre insert the prediction launch and waits
  cudaStream_t pg_st = nullptr;         // stream dedicated to the prediction (overlaps attention)
  cudaEvent_t pg_read = nullptr, pg_done = nullptr;  // prediction finished reading h (waited before hc_post); whole prediction done (waited at the end of the front — capture join)
  bf16* pg_x = nullptr; bf16* pg_xn = nullptr;       // [8·dim] (M ≤ 8) prediction inputs (hc_pre, ffn_norm)
  float* pg_scores = nullptr;           // [8·512]
  int32_t* pg_ids_d = nullptr; float* pg_rw_d = nullptr;  // [8·kMaxK] router output (device)
  int32_t* pg_ids_h = nullptr; int32_t* pg_ids_hd = nullptr; float* pg_rw_h = nullptr; float* pg_rw_hd = nullptr;  // mapped pinned (host, device alias)
  uint32_t* pg_flag_h = nullptr; uint32_t* pg_flag_d = nullptr; uint32_t* pg_ctr_d = nullptr;
  int* pg_counter = nullptr;            // last-block counter of the router kernel (separate from the main router — they run concurrently)
  er::Gate pg_gate;
  bool pg_resync = false;               // layer whose published number could not be trusted — reconciled after the front ends (wait_front)
  int pg_pf[pg::kMaxPf]; int pg_npf = 0, pg_l = -1;  // prefetch experts handed over for this layer (for comparison)
  pg::Acc pg_acc;
  // ---- E) F1 prefill isolation (enabled only for the upper layers of HIVE_PREFILL_MULTI_TAIL_SHORT — see the up_experts comment in forward_multi): while on,
  //   moe_decode_experts and moe_experts_multi skip usage observation (observe — scores, D4 policy) and expert tracing (xtrace) — the caller already observed
  //   all of that layer's sub-chunks with the same formula and order as a single moe_experts_multi(subs) call. Off (default, every other call) = false →
  //   both functions observe as usual.
  bool obs_off = false;
  ~DecodeOverlap() {
    for (cudaEvent_t e : ev) if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : front_done) if (e) cudaEventDestroy(e);
    if (er_front) cudaEventDestroy(er_front);
    if (er_tbl) cudaEventDestroy(er_tbl);
    if (er_tbl_st) cudaStreamDestroy(er_tbl_st);
    if (pg_read) cudaEventDestroy(pg_read);
    if (pg_done) cudaEventDestroy(pg_done);
    if (pg_st) cudaStreamDestroy(pg_st);
    for (void* q : {(void*)pg_x, (void*)pg_xn, (void*)pg_scores, (void*)pg_ids_d, (void*)pg_rw_d, (void*)pg_ctr_d, (void*)pg_counter}) if (q) cudaFree(q);
    for (void* q : {(void*)pg_ids_h, (void*)pg_rw_h, (void*)pg_flag_h}) if (q) cudaFreeHost(q);
    if (er_flag_h) cudaFreeHost(er_flag_h);
    if (er_ctr_d) cudaFree(er_ctr_d);
  }
};
// ---- O1 END

// ---- C1 HIVE_DECODE_SPLIT=balance state (model and formulas: see the comment at the top of hive/decode_split.h) ------------------------------------------------------
//   Sample = 5 GPU events for one layer: g0 (st_, right before the expert-table H2D — O1 kExp point), r1 (st_, end of resident groups), c1 (side_, end of
//   this layer's last demand copy), g1 (st_, end of DMA groups = O1 kGEnd point), acc (st_, CPU result accumulation launch = O1 kAcc point). Only one is armed
//   at a time (collected with cudaEventQuery at the next layer's decision — decode synchronizes on routing every layer, so it has usually finished already);
//   on HIVE_PROFILE sample steps every layer with misses, otherwise one of every kSampleEvery miss layers (the 5 event-record calls add host time on the
//   critical path — value not tuned). The CPU table is updated every layer regardless of sampling (host clock — no events needed).
struct Runtime::DecodeSplit {
  static constexpr unsigned kSampleEvery = 4;
  dsplit::Model m[dsplit::kCls];
  dsplit::Acc prof, win;  // prof = samples armed on sample steps (sum for that step — same step as [decode-host]); win = everything since the last line
  dsplit::Input in;       // reused per layer (no allocation)
  bool pending = false, open = false, armed_prof = false, has_cpu = false;
  int cls = 0, base_n = 0;
  unsigned tick = 0;
  dsplit::Decision dec;
  dsplit::Obs obs;
  cudaEvent_t g0 = nullptr, r1 = nullptr, c1 = nullptr, g1 = nullptr, acc = nullptr;
  void make_events() {
    if (g0) return;
    for (cudaEvent_t* e : {&g0, &r1, &c1, &g1, &acc}) CUDA_CHECK(cudaEventCreateWithFlags(e, cudaEventDefault));
  }
  ~DecodeSplit() { for (cudaEvent_t e : {g0, r1, c1, g1, acc}) if (e) cudaEventDestroy(e); }
};
// ---- B2 batched decode miss state (switches and rules: see the comment at the top of hive/batch_miss.h) --------------------------------------------------------------
struct Runtime::BatchMiss {
  bool cpu_first = false, stage_hit = false;  // HIVE_DECODE_CPU_FIRST, HIVE_DECODE_STAGE_HIT (PREFETCH turns it on too — pre-copies are consumed through this lookup)
  int prefetch = 0;                           // HIVE_DECODE_PREFETCH records per layer (0 = off)
  bmiss::StageTrack track;                    // (layer, expert) → ring position
  bmiss::Pred pred;                           // per layer, the CPU misses of the previous batched step
  bmiss::Acc acc;                             // [decode-miss] totals
  cudaEvent_t probe[2] = {nullptr, nullptr};  // whether the link is idle before a pre-copy (recorded at the end of side_/promo_ → cudaEventQuery)
  ~BatchMiss() { for (cudaEvent_t e : probe) if (e) cudaEventDestroy(e); }
};

namespace {
// C1 switch: on only when env_on (unset, empty or "0" = off) and the name is "balance" (dsplit::parse_mode, case-insensitive) — unknown names are off too (fall back to the base rule). Same form as HIVE_PREFILL_SPLIT.
bool decode_split_on() {
  static const bool decode_split = env_on("HIVE_DECODE_SPLIT") && dsplit::parse_mode(getenv("HIVE_DECODE_SPLIT")) == dsplit::kBalance;
  return decode_split;
}
}  // namespace

Runtime::Runtime(Model& model, ExpertStore& store, const EngramHash* eh, const RuntimeOptions& opt)
    : model_(model), store_(store), eh_(eh), opt_(opt) {
  if (getenv("HIVE_MTP_CACHE")) opt_.mtp_cache = env_on("HIVE_MTP_CACHE");
  if (getenv("HIVE_PHASE_SCORE")) opt_.phase_score = env_on("HIVE_PHASE_SCORE");
  if (getenv("HIVE_PROMOTE_SCORE")) opt_.promote_score_victims = env_on("HIVE_PROMOTE_SCORE");
  if (const char* v = getenv("HIVE_PROMOTE_BYTES")) opt_.promote_byte_budget = (size_t)std::max(0ll, atoll(v));
  if(env_on("HIVE_CKPT_ASYNC") && getenv("HIVE_CKPT_PINNED_POOL_MB")) {
    const double mb=std::max(0.0,atof(getenv("HIVE_CKPT_PINNED_POOL_MB")));
    const size_t budget=std::isfinite(mb) ? (size_t)(std::min(mb,(double)(SIZE_MAX/1048576))*1048576.0) : 0;
    if(budget) snapshot_pool_=std::make_shared<HostImagePool>(budget,
      // CKPT_ASYNC pinned failure must not abort: report it as bad_alloc so save_image falls back to pageable memory
      [](size_t n){uint8_t* p=nullptr;if(cudaHostAlloc(reinterpret_cast<void**>(&p),n,cudaHostAllocDefault)!=cudaSuccess||!p){(void)cudaGetLastError();throw std::bad_alloc();}return p;},
      [](uint8_t* p){cudaFreeHost(p);});
  }
  // Round-2 opt-ins (prefill investigation — all off by default; switch rule unset/""/"0" = off). Off = same as the base path.
  prefill_pause_promote_ = env_on("HIVE_PREFILL_PAUSE_PROMOTE");  // M7
  early_stream_ = env_on("HIVE_EARLY_STREAM");                    // L1
  // HIVE_LAYER_YIELD_SMALL (optional, default off): layer-boundary yields also inside HIVE_PREFILL_SMALL forwards (prompt chunks below the
  //   prefill threshold). Measured (interactive traffic on the reference machine): the prefills that stalled running decoders had a median of 420 new tokens
  //   (follow-up turns of long conversations), i.e. one short forward of 0.5–1.5 s during which no decode step ran — per decoder the longest
  //   stall was 1.35 s median, 2.0 s p90. Only the eligibility changes; the park/unpark path is the same as for long forwards.
  //   Verdict (4 decoders + follow-up turns, bench_turns): with the synchronous post-prefill warm still in place no gain (the warm
  //   stall dominated); on top of HIVE_WARM_DEFER the decoders' longest gap 1.05–1.07 → 0.59–0.60 s at −1.5 % decode (21.5 → 21.2 tok/s) —
  //   adopted (docs/performance.md step 20).
  ly_small_ = env_on("HIVE_LAYER_YIELD_SMALL");
  engram_par_ = env_on("HIVE_ENGRAM_PAR");                        // L2
  engram_digest_ = engram_digest_on();                            // HIVE_ENGRAM_DIGEST (verification log)
  idx_msub_actual_ = env_on("HIVE_IDX_MSUB_ACTUAL");              // L3
  host_par_threads_ = std::max(1, store_.cpu_threads());          // L1/L2: the CPU expert pool is idle in that section (after the previous layer's wait_jobs, before this layer's start_jobs)
  if (const char* v = getenv("HIVE_VIT_CACHE_MB"); v && *v) {     // M3: MB (fractions allowed). 0, negative or non-numeric = off
    const double mb = atof(v);
    if (std::isfinite(mb) && mb > 0) vit_cache_ = std::make_unique<VitCache>((size_t)(std::min(mb, (double)(SIZE_MAX / 1048576)) * 1048576.0));
  }
  // Round 3 (switch rule unset/""/"0" = off; off = same as the base path)
  batch_prefill_ = env_on("HIVE_BATCH_PREFILL");  // H3: run several waiting prefills as one forward_multi (hived batches them)
  step_graph_ = env_on("HIVE_DECODE_STEP_GRAPH");  // G1: a forward_batch step = one graph (no per-layer host sync — forward_batch_step)
  host_fast_ = env_on("HIVE_DECODE_HOST_FAST");  // O1 B: reduce host work on the decode layer's critical path (same experts, devices and order — bit-identical)
  // P1 HIVE_DECODE_PREGATE=K: value as is (pg::parse_k — 0, "" or unset = off; "1" or non-numeric = 8; above 16 = 16). The store enables owned mode with the same value
  //   (ExpertStore constructor) — if they disagree (store off), prefetch requests are dropped by the store and only prediction/publishing runs (no effect on results).
  pregate_k_ = pg::parse_k(getenv("HIVE_DECODE_PREGATE"));
  ubatch_ = env_on("HIVE_DECODE_UBATCH");        // O1 C: run batched decode (M ≥ 2) as a two-half layer pipeline (dov::run_ubatch)
  verify2_ = env_on("HIVE_MTP_VERIFY2");        // R1: run forward_verify on the decode path (verify_decode — usability is v2_ok_ at the end of the constructor)
  mtp_batch_ = env_on("HIVE_MTP_BATCH");        // R1: multi-sequence verify (forward_verify_batch — hived speculates with ≥ 2 active)
  verify_rowind_ = env_on("HIVE_MTP_VERIFY_ROWIND");  // R1: make verify rows' expert sums row-independent (see the comment at the top of moe_decode_experts)
  // H2: host tile count (requested) — actual allocation (pinned, VRAM) happens after the tile slots and is reduced when short. Tile condition = a max_chunk chunk is in tail mode.
  host_slots_ = host_tile_count(env_on("HIVE_PREFILL_HOST_TILES"), getenv("HIVE_PREFILL_HOST_MB"), (size_t)opt_.max_chunk * model_.cfg().hc * model_.cfg().dim * 2,
                                tail_mode_for(opt_.max_chunk), std::max(1, opt_.prefill_tile), opt_.max_chunk, opt_.max_ctx);
  if (const char* xp = getenv("HIVE_EXPERT_TRACE"); xp && *xp && strcmp(xp, "0") != 0) {  // unset/""/"0" = off (switch rule — so that "0" never becomes a file name)
    xtrace_ = fopen(xp, "ab");
    if (xtrace_) { setvbuf(xtrace_, nullptr, _IOFBF, 1 << 20); fprintf(stderr, "[hive] expert trace → %s\n", xp); }
    else fprintf(stderr, "[hive] ⚠️cannot open expert trace file: %s\n", xp);
  }
  CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
  dov_ = std::make_shared<DecodeOverlap>();  // O1 state (events are created on use)
  dov_->er_on = env_on("HIVE_DECODE_EARLY_ROUTE");  // S1: publish routing right after the router (host classification and GPU expert launch during the shared expert — hive/early_route.h)
  if (decode_split_on()) dsplit_ = std::make_shared<DecodeSplit>();  // C1 (not created when off — events at the first sample)
  {  // B2 (hive/batch_miss.h): all three off (unset, "", "0") → bmiss_ = nullptr → moe_decode_experts takes the base path
    const bool cpu_first = env_on("HIVE_DECODE_CPU_FIRST"), stage_hit = env_on("HIVE_DECODE_STAGE_HIT");
    const int prefetch = env_on("HIVE_DECODE_PREFETCH") ? bmiss::parse_prefetch(getenv("HIVE_DECODE_PREFETCH")) : 0;  // on = env_on; value = records per layer
    if (cpu_first || stage_hit || prefetch > 0) {
      bmiss_ = std::make_shared<BatchMiss>();
      bmiss_->cpu_first = cpu_first; bmiss_->stage_hit = stage_hit || prefetch > 0; bmiss_->prefetch = prefetch;
      if (prefetch > 0) for (cudaEvent_t& e : bmiss_->probe) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
  }
  if (env_on("HIVE_DECODE_PREDICT_EVAL")) pe_ = std::make_shared<PredEval>();  // F2 (measurement only)
  if (const char* cp = getenv("HIVE_CACHE_PRIOR")) cache_prior_ = std::max(0.f, (float)atof(cp));
  if (const char* cj = getenv("HIVE_CACHE_PRIOR_TOPJ")) cache_prior_j_ = std::min(4, std::max(0, atoi(cj)));
  dec_defer_ = env_on("HIVE_DECODE_DEFER");
  if (const char* sk = getenv("HIVE_DECODE_SKIP_MISS")) dec_skip_ = std::max(0.f, (float)atof(sk));
  if (cache_prior_ > 0.f) {
    const Config& cc = model_.cfg();
    CUDA_CHECK(cudaMallocHost(&cache_mask_h_, (size_t)cc.n_layers * cc.n_routed));
    std::memset(cache_mask_h_, 0, (size_t)cc.n_layers * cc.n_routed);
    prior_range_.alloc((size_t)cc.n_layers * sizeof(float));
    CUDA_CHECK(cudaMemset(prior_range_.p, 0, (size_t)cc.n_layers * sizeof(float)));
    fprintf(stderr, "[runtime] cache prior on: λ %.3f · top-%d always kept (decode routing biased toward VRAM-resident experts; HIVE_DECODE_ROUTER3 path)\n",
            cache_prior_, cache_prior_j_);
  }
  // E4 HIVE_DECODE_COPY_PRIO: demand DMA (side_) highest, promotion (promo_) lowest priority + promotion H2D pacing (promo_pump). Priorities are kernel
  //   scheduling hints with no guarantee the copy engines follow them — order is set by promo_pump's explicit gate (after demand copies) and the per-layer byte budget (priority is only an extra).
  copy_prio_ = env_on("HIVE_DECODE_COPY_PRIO");
  if (copy_prio_) {
    int least = 0, greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&side_, cudaStreamNonBlocking, greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&promo_, cudaStreamNonBlocking, least));
    CUDA_CHECK(cudaEventCreateWithFlags(&promo_gate_, cudaEventDisableTiming));
  } else {
  CUDA_CHECK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&promo_, cudaStreamNonBlocking));
  }
  store_.set_promo_pacing(copy_prio_);
  CUDA_CHECK(cudaStreamCreateWithFlags(&snapshot_st_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&hc_side_, cudaStreamNonBlocking));
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaEventCreateWithFlags(&hc_fork_[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&hc_join_[i], cudaEventDisableTiming));
  }
  blas_ = std::make_unique<Blas>(st_);
  profile_ = getenv("HIVE_PROFILE") && atoi(getenv("HIVE_PROFILE")) > 0;
  if (profile_) profile_every_ = std::max(1, atoi(getenv("HIVE_PROFILE")));
  CUDA_CHECK(cudaEventCreateWithFlags(&promo_evt_, cudaEventDisableTiming));
  stage_copied_.resize(store_.staging_slots()); stage_freed_.resize(store_.staging_slots()); stage_used_.assign(store_.staging_slots(), 0);
  for (int i = 0; i < store_.staging_slots(); ++i) {
    CUDA_CHECK(cudaEventCreateWithFlags(&stage_copied_[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage_freed_[i], cudaEventDisableTiming));
  }
  const Config& c = model_.cfg();
  const int M = opt_.max_chunk;
  auto w = std::make_unique<Work>();
  w->M = M;
  const int dim = c.dim, hc = c.hc, HD = c.n_heads * c.head_dim, D = c.head_dim, I = c.moe_inter, k = c.n_act;
  const int Hi = c.index_n_heads, Di = c.index_head_dim;
  const int mix_hc = (2 + hc) * hc;
  w->h.alloc((size_t)M * hc * dim * 2);
  // hflat: full size only for the alternative hc mix path (HIVE_HC_CUBLAS, fp32 [M, hc·dim]) — by default only the decoder-tail row-move temporary (bf16 [tail, hc·dim]) (saves 1.2 GB)
  w->hflat.alloc(env_on("HIVE_HC_CUBLAS") ? (size_t)M * hc * dim * 4 : (size_t)std::min(M, std::max(opt_.decoder_tail, 1)) * hc * dim * 2);
  w->mixes.alloc((size_t)M * mix_hc * 4);
  w->rsq.alloc((size_t)M * 4);
  for (DevBuf* b : {&w->pre_a, &w->post_a, &w->pre_f, &w->post_f, &w->pre_mix}) b->alloc((size_t)M * hc * 4);
  for (DevBuf* b : {&w->comb_a, &w->comb_f}) b->alloc((size_t)M * hc * hc * 4);
  for (DevBuf* b : {&w->x, &w->xn, &w->attn_out, &w->eout, &w->ffn_out}) b->alloc((size_t)M * dim * 2);
  w->xf.alloc((size_t)M * dim * 4);
  w->xq.alloc((size_t)M * dim); w->xs.alloc((size_t)M * dim / 32);
  w->qr.alloc((size_t)M * c.q_lora_rank * 2); w->qrn.alloc((size_t)M * c.q_lora_rank * 2);
  w->qrq.alloc((size_t)M * c.q_lora_rank); w->qrs.alloc((size_t)M * c.q_lora_rank / 32);
  w->q.alloc((size_t)M * HD * 2); w->o.alloc((size_t)M * HD * 2);
  w->kv.alloc((size_t)M * D * 2); w->latent.alloc((size_t)M * D * 2);
  w->ckv.alloc((size_t)M * D * 4); w->cscore.alloc((size_t)M * D * 4); w->cout.alloc((size_t)M * D * 4);
  w->ik.alloc((size_t)M * Di * 2); w->iq.alloc((size_t)M * Hi * Di * 2); w->iw.alloc((size_t)M * Hi * 2);
  if (getenv("HIVE_IDX_TC") && atoi(getenv("HIVE_IDX_TC")) != 0) w->iqp.alloc((size_t)M * Hi * kvp::IDX_ROW);  // D-2 packed queries for the tensor-core indexer (only when on)
  w->pos.alloc((size_t)M * 4); w->gpos.alloc((size_t)M * 4); w->visible.alloc((size_t)M * 4);
  w->idx.alloc((size_t)M * (c.window + c.index_topk) * 4);
  // P2: prefill attention defaults to the flash variant that does not write S to global memory (M ≥ 16), so the v2 scratch only covers small M
  //   (2.7 GB at a 16K chunk → a few MB (max_batch 8: 2.6 MB, 64: 10 MB)). Full size with HIVE_ATTN_V2=1 (for comparison).
  //   The batched-decode comparison kernel (HIVE_ATTN_TC, sparse_attn_tc_rows) has rows = batch size, so keep at least max_batch rows (otherwise out-of-range writes at --max-batch 17+).
  const int attn_rows = env_on("HIVE_ATTN_V2") ? M : std::max(std::min(M, 16), std::min(M, opt_.max_batch));
  w->attn_S.alloc((size_t)attn_rows * c.n_heads * (c.window + c.index_topk) * 4);
  if (M > 8) w->woa.alloc((size_t)c.o_groups * c.o_lora_rank * (c.n_heads * c.head_dim / c.o_groups) * 2);
  w->attn_mx.alloc((size_t)M * c.n_heads * 4); w->attn_sum.alloc((size_t)M * c.n_heads * 4);
  w->og.alloc((size_t)M * c.o_groups * c.o_lora_rank * 2);
  w->ogq.alloc((size_t)M * c.o_groups * c.o_lora_rank); w->ogs.alloc((size_t)M * c.o_groups * c.o_lora_rank / 32);
  w->scores.alloc((size_t)M * c.n_routed * 4);
  w->ids.alloc((size_t)M * k * 4); w->rw.alloc((size_t)M * k * 4);
  // Expert intermediate buffers: prefill uses M rows, the decode group path uses gathered rows (≤ Mb·k) — whichever is larger
  const int Mrows = std::max(M, std::min(std::max(opt_.max_batch, opt_.prefill_threshold), M) * k);  // rows gathered by the group path ≤ min(M, threshold)·k
  for (DevBuf* b : {&w->gate, &w->up, &w->y}) b->alloc((size_t)Mrows * I * 2);
  w->yq.alloc((size_t)Mrows * I); w->ys.alloc((size_t)Mrows * I / 32);
  w->eout.alloc((size_t)Mrows * dim * 2);
  w->acc.alloc((size_t)M * dim * 4);
  // Concurrent-use audit: g_xq/g_xs only need gathered rows ≤ Mrows (one prefill expert or CPU share ≤ M, decode groups ≤ threshold·k) (M·k would be 495 MB);
  //   a device cpu_out (M·k·dim·4 = 1.9 GB) is not needed (only the mapped pinned cpu_out_d is used).
  w->g_rows.alloc((size_t)M * k * 4); w->g_xq.alloc((size_t)Mrows * dim); w->g_xs.alloc((size_t)Mrows * dim / 32); w->g_rw.alloc((size_t)M * k * 4);
  w->cpu_rows.alloc((size_t)M * k * 4);
  if (eh_) {
    const int cols = c.n_hash_cols(), hd = c.engram_head_dim;
    // R5 HIVE_ENGRAM_ALIAS (optional; unset, "" or "0" = off = the allocation below): borrow engram work buffers inside the attention buffers — saves 1,190 MiB
    //   at a 16K chunk (with hived HIVE_CACHE_FIT that becomes ~66 expert slots). (1) device eg_v/eg_s (99 MiB) have no reader (kernels read the mapped pinned
    //   eg_v_d/eg_s_d — no use of w.eg_v/w.eg_s besides the allocation lines) → not allocated when on. (2) eg_kv (M·dim·(hc+1) bf16) → q; eg_vals/eg_q/eg_sc → o
    //   (contiguous, 256 B aligned). Lifetime: the four buffers are used only in engram_dev (dequant → act_quant → gemm_bs → engram_gate, all in st_ order), and
    //   engram runs at the head of the layer (before attention), when q and o hold dead values — q/o are used and finished within attention,
    //   attention_decode_dev and mtp_attention (they never cross layers; no other stream touches them). Once engram_gate has written h the four buffers are
    //   never read again. Same kernels, same inputs, same order → bit-identical values (only addresses differ).
    //   Configurations where the sizes do not fit (larger than q/o) fall back to the separate allocation.
    const size_t b_kv = (size_t)M * dim * (hc + 1) * 2, b_vals = align_up((size_t)M * cols * hd * 2, 256), b_q = align_up((size_t)M * cols * hd, 256),
                 b_sc = (size_t)M * cols * hd / 32;
    w->eg_alias = env_on("HIVE_ENGRAM_ALIAS") && b_kv <= w->q.n && b_vals + b_q + b_sc <= w->o.n;
    if (w->eg_alias) {
      uint8_t* o = w->o.as<uint8_t>();
      w->eg_kv.p = w->q.p; w->eg_kv.n = b_kv;
      w->eg_vals.p = o; w->eg_vals.n = (size_t)M * cols * hd * 2;
      w->eg_q.p = o + b_vals; w->eg_q.n = (size_t)M * cols * hd;
      w->eg_sc.p = o + b_vals + b_q; w->eg_sc.n = b_sc;
      fprintf(stderr, "[runtime] engram alias on: %.0f MiB of engram work buffers live inside q/o\n",
              ((size_t)M * cols * hd + (size_t)M * cols * hd / 32 + (size_t)M * cols * hd * 2 + (size_t)M * cols * hd + b_sc + b_kv) / 1048576.0);
    } else {
    w->eg_v.alloc((size_t)M * cols * hd); w->eg_s.alloc((size_t)M * cols * hd / 32);
    w->eg_vals.alloc((size_t)M * cols * hd * 2);
    w->eg_q.alloc((size_t)M * cols * hd); w->eg_sc.alloc((size_t)M * cols * hd / 32);
    w->eg_kv.alloc((size_t)M * dim * (hc + 1) * 2);
    }
    w->eg_v_h = pinned_mapped<uint8_t>((size_t)M * cols * hd, &w->eg_v_d);
    w->eg_s_h = pinned_mapped<uint8_t>((size_t)M * cols * hd / 32, &w->eg_s_d);
  }
  // Indexer score buffers: Tcap = max_ctx (ratio 1 basis), Msub = budget / Tcap
  w->Tcap = (int)opt_.max_ctx;
  w->Msub = std::max(1, std::min(M, (int)(opt_.index_budget / std::max(1, w->Tcap))));
  w->nblocks_cap = (w->Tcap + c.cand_block - 1) / c.cand_block;
  // Column count of the candidate-pool path = ⌈T/8⌉·8 ≤ Tcap + 7 (overflows by 7 columns when max_ctx % 8 ≠ 0) → 8 spare columns
  w->iscore.alloc((size_t)w->Msub * (w->Tcap + 8) * 2);
  w->iscore_f.alloc((size_t)w->Msub * (w->Tcap + 8) * 4);
  w->bmax.alloc((size_t)w->Msub * w->nblocks_cap * 4);
  w->topk_pos.alloc((size_t)w->Msub * std::max(c.index_topk, c.cand_topk_blocks) * 4);
  if (c.cand_source_layer >= 0) w->cand.alloc((size_t)M * c.cand_topk_blocks * 4);  // 128 MB at a 16K chunk (a per-context keep buffer would be 512 MB at 262K context)
  // (CUB TopK replaced by the radix-select kernel — nothing reserved)
  w->xlast.alloc((size_t)dim * 4);
  w->logits.alloc((size_t)c.vocab * 4);
  w->next.alloc(4);
  w->ids_h = pinned<int32_t>(M); w->route_ids_h = pinned_mapped<int32_t>((size_t)M * k, &w->route_ids_d);
  w->pos_h = pinned_mapped<int32_t>(M, &w->pos_d); w->gpos_h = pinned_mapped<int32_t>(M, &w->gpos_d); w->visible_h = pinned_mapped<int32_t>(M, &w->visible_d);
  w->g_rows_cap = M * k * std::max(1, opt_.prefill_tile + host_slots_);  // C2: all sub-chunks of a tile share the table within one layer (H2 host tiles included — off = the base value)
  w->g_rows_h = pinned_mapped<int32_t>((size_t)w->g_rows_cap, &w->g_rows_d); w->g_rw_h = pinned_mapped<float>((size_t)w->g_rows_cap, &w->g_rw_d);
  w->rw_h = pinned_mapped<float>((size_t)M * k, &w->rw_d);
  w->cpu_rows_h = pinned_mapped<int32_t>((size_t)M * k, &w->cpu_rows_d);
  w->cpu_out_h = pinned_mapped<float>((size_t)Mrows * dim, &w->cpu_out_d);  // CPU job rows ≤ Mrows (prefill share ≤ M, decode ≤ threshold·k)
  if (dec_defer_) {  // deferred decode rows: at most max_batch-sized decode × k (decode kernel M ≤ 8)
    const size_t drows = (size_t)8 * k;
    def_out_h_ = pinned_mapped<float>(drows * dim, &def_out_d_);
    def_rows_h_ = pinned_mapped<int32_t>(drows, &def_rows_d_);
    def_scratch_h_ = pinned<float>(drows * ExpertStore::job_scratch_floats(c.moe_inter));
    post_def_.alloc((size_t)8 * c.hc * 4);
    fprintf(stderr, "[runtime] expert deferral on: CPU misses ranked 3rd or lower are added one layer later (early-route decode)\n");
  }
  w->xq_h = pinned_mapped<uint8_t>((size_t)M * dim, &w->xq_hd); w->xs_h = pinned_mapped<uint8_t>((size_t)M * dim / 32, &w->xs_hd);
  w->a_f_h = (float*)aligned_alloc(64, (size_t)M * dim * 4); w->a_s_h = (float*)aligned_alloc(64, (size_t)M * (dim / 32) * 4);
  w->scratch_h = (float*)aligned_alloc(64, (size_t)Mrows * ExpertStore::job_scratch_floats(I) * 4);
  w->next_h = pinned<int32_t>(1);
  {
    const int Mb = std::max(1, std::min(opt_.max_batch, M));
    w->kvrows_h = pinned_mapped<k::KvRow>(Mb, &w->kvrows_d);
    w->kptrs_h = pinned_mapped<const uint8_t*>(Mb, &w->kptrs_d);
    w->trows_h = pinned_mapped<int32_t>(Mb, &w->trows_d);
    w->dsti_h = pinned_mapped<int32_t>(Mb, &w->dsti_d);
    w->ringp_h = pinned_mapped<bf16*>(Mb, &w->ringp_d);
    w->dstp_h = pinned_mapped<uint8_t*>(Mb, &w->dstp_d);
    w->dstk_h = pinned_mapped<uint8_t*>(Mb, &w->dstk_d);
    w->skv_h = pinned_mapped<float*>(Mb, &w->skv_d);
    w->ssc_h = pinned_mapped<float*>(Mb, &w->ssc_d);
    w->valid.alloc((size_t)Mb);
    w->attn_pacc.alloc((size_t)Mb * c.n_heads * 16 * c.head_dim * 4);
    w->attn_pm.alloc((size_t)Mb * c.n_heads * 16 * 4); w->attn_ps.alloc((size_t)Mb * c.n_heads * 16 * 4);
    if (env_on("HIVE_DECODE_ATTN_FUSED")) {  // D2: only when on (a few KB)
      w->dec_tab.alloc((size_t)Mb * sizeof(k::DecRow) + sizeof(k::DecSoA)); w->dec_cnt.alloc((size_t)Mb * c.n_heads * 4);  // row table + column-array copy
      CUDA_CHECK(cudaMemset(w->dec_cnt.p, 0, w->dec_cnt.n));
    }
    const int Mlog = std::max(Mb, c.dspark_block + 1);  // draft block heads and per-verify-row logits use this buffer too
    w->logits_b.alloc((size_t)Mlog * c.vocab * 4);
    w->next_b.alloc((size_t)Mlog * 4);
    w->next_b_h = pinned<int32_t>(Mlog);
    w->gdesc_h = pinned_mapped<k::GroupDesc>((size_t)std::min(std::max(Mb, opt_.prefill_threshold), M) * k, &w->gdesc_d);
    w->gdesc_dev.alloc(sizeof(k::GroupDesc) * (size_t)std::min(std::max(Mb, opt_.prefill_threshold), M) * k);
    const size_t tbl_bytes = (sizeof(k::GroupDesc) + 8) * (size_t)std::min(std::max(Mb, opt_.prefill_threshold), M) * k;
    w->tbl_h = pinned<uint8_t>(tbl_bytes);
    w->tbl_dev.alloc(tbl_bytes);
  }
  // Graphs: off in dump mode (in-layer device dumps are incompatible with capture). HIVE_GRAPH_DUMP=1 keeps graphs on and only keeps host dumps (routing)
  // and final-state dumps (h_final, logits) — for comparing the graph path with the eager path.
  attn_splits_ = getenv("HIVE_ATTN_SPLITS") ? atoi(getenv("HIVE_ATTN_SPLITS")) : 4;
  fuse_ = getenv("HIVE_FUSE") ? atoi(getenv("HIVE_FUSE")) != 0 : true;  // decode fused kernels (fused.cu) — launches per layer ~40 → ~22
  if (getenv("HIVE_DMA_FRAC")) dma_frac_fixed_ = (float)atof(getenv("HIVE_DMA_FRAC"));
  // H4 (opt-in): per-layer DMA cap for short prefills. Unset/empty = 8 (default). Below 1 or non-numeric falls back to 8 (never rejected). Clipped again by the staging slot count.
  if (const char* v = getenv("HIVE_SHORT_DMA_CAP"); v && *v) { const int n = atoi(v); short_dma_cap_ = n >= 1 ? n : 8; }
  // R2 short-prefill (TTFT) opt-ins — all follow the switch rule (unset, "" or "0" = off = base path). Meanings: runtime.h ShortPrefillOpts.
  spo_.tail_short = env_on("HIVE_PREFILL_TAIL_SHORT");
  spo_.prefetch_scale = env_on("HIVE_PREFETCH_SCALE");
  spo_.prof = env_on("HIVE_PREFILL_PROF");
  //   HIVE_PREFILL_SHORT_ADAPT: on = N (rows, integer ≥ 2); "1" or non-numeric = 4096 (upper end of the 1K–4K target range — a range definition, not a measured value); off = 0 (hive/short_prefill.h)
  spo_.adapt_rows = env_on("HIVE_PREFILL_SHORT_ADAPT") ? sp::parse_short_adapt(getenv("HIVE_PREFILL_SHORT_ADAPT")) : 0;
  // Q3 (same rule): see the top of hive/short_prefill.h and the Q3 comments in forward_multi / forward
  spo_.multi_tail_short = env_on("HIVE_PREFILL_MULTI_TAIL_SHORT");
  //   HIVE_PREFILL_SMALL: on = row floor max(N or none, max_batch+1, 9); off = 0 (hive/short_prefill.h parse_small)
  spo_.small_rows = env_on("HIVE_PREFILL_SMALL") ? sp::parse_small(getenv("HIVE_PREFILL_SMALL"), opt_.max_batch) : 0;
  CUDA_CHECK(cudaEventCreateWithFlags(&dma_e0_, cudaEventDefault));
  CUDA_CHECK(cudaEventCreateWithFlags(&prefetch_begin_, cudaEventDefault));
  CUDA_CHECK(cudaEventCreateWithFlags(&prefetch_end_, cudaEventDefault));
  CUDA_CHECK(cudaEventCreateWithFlags(&dma_e1_, cudaEventDefault));
  w->counters.alloc(16 * 4);
  CUDA_CHECK(cudaMemset(w->counters.p, 0, w->counters.n));
  if (k::hc_decode_fused_on() || k::hc_sinkhorn_par_on()) k::hc_decode_fused_preload();  // E3: lazy module loading outside capture and step-graph spin waits (same reason as hs_preload)
  w->ringp1_h = pinned_mapped<bf16*>(8, &w->ringp1_d);
  if (opt_.sampler_cands > 0) {
    const int Mlog = std::max(std::max(1, std::min(opt_.max_batch, M)), c.dspark_block + 1), NC = opt_.sampler_cands;
    cand_it_h_ = pinned_mapped<float>(Mlog, &cand_it_d_);
    for (int i = 0; i < Mlog; ++i) cand_it_h_[i] = 1.f;
    cand_idx_h_ = pinned<int32_t>((size_t)Mlog * NC); cand_val_h_ = pinned<float>((size_t)Mlog * NC);
    cand_max_h_ = pinned<float>(Mlog); cand_sum_h_ = pinned<float>(Mlog);
    cand_idx_.alloc((size_t)Mlog * NC * 4); cand_val_.alloc((size_t)Mlog * NC * 4); cand_max_.alloc((size_t)Mlog * 4); cand_sum_.alloc((size_t)Mlog * 4);
  }
  graphs_ = !env_on("HIVE_NO_GRAPH") && (opt_.dump_dir.empty() || env_on("HIVE_GRAPH_DUMP"));
  w->is_image_h = pinned_mapped<int8_t>(M, &w->is_image_d);
  w->emb_h = pinned_mapped<bf16>((size_t)M * dim, &w->emb_d);
  { std::vector<int32_t> io(M); for (int i = 0; i < M; ++i) io[i] = i; w->iota.alloc((size_t)M * 4); CUDA_CHECK(cudaMemcpy(w->iota.p, io.data(), (size_t)M * 4, cudaMemcpyHostToDevice)); }
  w->img_types.alloc((size_t)c.vision_max_tokens + 64); w->img_row_of.alloc(((size_t)c.vision_max_tokens + 64) * 4);
  w->img_rows.alloc((size_t)c.vision_max_tokens * dim * 2);
  // ids are needed on the device too (embedding gather): ids_h is uploaded together with pos → the ids buffer is separate rather than reusing g_rows
  // DSpark drafts: only when the model carries mtp stages and the trunk and head are complete
  mtp_on_ = opt_.mtp && model_.n_mtp() > 0 && model_.has_head() && model_.n_loaded_layers() == c.n_layers && !c.dspark_targets.empty();
  if (mtp_on_) {
    ntgt_ = (int)c.dspark_targets.size();
    HIVE_CHECK(store_.n_layers() >= c.n_layers + model_.n_mtp(), "expert store lacks the dspark stages");
    HIVE_CHECK(c.dspark_block <= 8 && c.dspark_block + 1 <= M, "dspark block must fit the decode kernels (≤8 rows)");
    const int Mb = std::max(1, std::min(opt_.max_batch, M));
    const int B = c.dspark_block, rank = c.dspark_markov_rank, K = ntgt_ * dim;
    w->mh_rows = std::max({c.window, Mb, B + 1});
    w->mh.alloc((size_t)w->mh_rows * K * 2); w->mhq.alloc((size_t)w->mh_rows * K); w->mhs.alloc((size_t)w->mh_rows * K / 32);
    w->mx.alloc((size_t)w->mh_rows * dim * 2);
    w->mpos_h = pinned_mapped<int32_t>(w->mh_rows, &w->mpos_d);
    w->mringp_h.resize(model_.n_mtp()); w->mringp_d.resize(model_.n_mtp());
    for (int s = 0; s < model_.n_mtp(); ++s) w->mringp_h[s] = pinned_mapped<bf16*>(Mb, &w->mringp_d[s]);
    w->mhid_h = pinned_mapped<bf16*>(Mb, &w->mhid_d);
    w->dids.alloc((size_t)(B + 1) * 4); w->dconf.alloc((size_t)B * 4); w->membed.alloc((size_t)B * rank * 2); w->dx.alloc((size_t)B * dim * 2);
    w->dids_h = pinned<int32_t>(B + 1); w->dconf_h = pinned<float>(B);
    // R1: only when the verify decode path (HIVE_MTP_VERIFY2, HIVE_MTP_BATCH) is on, size snapshot rows for several sequences (≤ 8, ≤ Mb) — off keeps the base size
    const bool v2_any = verify2_ || mtp_batch_;
    const int Vrows = v2_any ? std::max(B + 1, std::min(8, Mb)) : B + 1;
    w->ring_snap.alloc((size_t)c.n_layers * Vrows * c.head_dim * 2);
    w->lringp_h = pinned_mapped<bf16*>(c.n_layers, &w->lringp_d);
    for (int l : c.kv_source_layers) if (c.ratio(l) > 1) v_src2_.push_back(l);
    for (size_t i = 0; i < v_src2_.size(); ++i) {
      w->vxn.emplace_back((size_t)(B + 1) * dim * 2);
      w->cstate_snap.emplace_back((size_t)2 * c.ratio(v_src2_[i]) * c.head_dim * 4);
    }
    if (v2_any) {
      w->vgrp_h = pinned_mapped<int32_t>(Mb, &w->vgrp_d);
      w->vlringp_h = pinned_mapped<bf16*>((size_t)kVParts * c.n_layers, &w->vlringp_d);
      for (int p = 0; p < kVParts; ++p)
        for (size_t i = 0; i < v_src2_.size(); ++i) w->vcsnap.emplace_back((size_t)2 * c.ratio(v_src2_[i]) * c.head_dim * 4);
      for (size_t i = 0; i < v_src2_.size(); ++i) w->vckv.emplace_back((size_t)2 * Vrows * c.head_dim * 4);
      // Usable when: decode fused path (fuse_ — attention_verify_dev exists only in that variant) and no dump/inject (layer dumps and golden injection use the base verify path)
      v2_rows_ = std::min({8, Mb, w->Msub, M, Vrows});
      v2_ok_ = fuse_ && opt_.dump_dir.empty() && opt_.inject_dir.empty() && v2_rows_ >= 2;
      fprintf(stderr, "[runtime] mtp verify decode path %s: verify2 %d · batch %d · rows ≤ %d\n", v2_ok_ ? "on" : "off (fuse/dump/inject/rows)", verify2_ ? 1 : 0,
              mtp_batch_ ? 1 : 0, v2_rows_);
    }
    fprintf(stderr, "[runtime] dspark on: stages %d · block %d · targets %d · capture rows %d\n", model_.n_mtp(), B, ntgt_, w->mh_rows);
  }
  // HIVE_IDX_F32 and HIVE_IDX_TC are mutually exclusive — fail at startup rather than on the first request
  HIVE_CHECK(!(env_on("HIVE_IDX_F32") && env_on("HIVE_IDX_TC")), "HIVE_IDX_F32 and HIVE_IDX_TC are exclusive");
  if (opt_.prefill_tile > 1 && !(opt_.decoder_tail > 0 && M >= std::max(opt_.decoder_tail + 1, opt_.prefill_threshold))) {
    // Tiles require tail mode after the encoder layers (layers past 20 run only on the last sub-chunk's tail) — impossible when the tail is off or the chunk is smaller than the tail
    fprintf(stderr, "[runtime] ⚠️prefill tile %d off: decoder_tail %d · max_chunk %d (tiling needs decoder_tail > 0 and max_chunk > decoder_tail)\n", opt_.prefill_tile,
            opt_.decoder_tail, M);
    opt_.prefill_tile = 1;
  }
  if (opt_.prefill_tile > 1) {  // C2 tile slots (sub-chunks 2..T): ≈1.14 GB per slot at a 16K chunk
    tiles_.resize((size_t)opt_.prefill_tile - 1);
    for (TileSlot& t : tiles_) {
      t.h.alloc((size_t)M * hc * dim * 2); t.pre_mix.alloc((size_t)M * hc * 4); t.idx.alloc((size_t)M * (c.window + c.index_topk) * 4);
      t.xq.alloc((size_t)M * dim); t.xs.alloc((size_t)M * dim / 32); t.acc.alloc((size_t)M * dim * 4);
      t.pre_f.alloc((size_t)M * hc * 4); t.post_f.alloc((size_t)M * hc * 4); t.comb_f.alloc((size_t)M * hc * hc * 4);
      t.route_ids_h = pinned_mapped<int32_t>((size_t)M * k, &t.route_ids_d); t.rw_h = pinned_mapped<float>((size_t)M * k, &t.rw_d);
      t.pos_h = pinned_mapped<int32_t>(M, &t.pos_d); t.visible_h = pinned_mapped<int32_t>(M, &t.visible_d); t.gpos_h = pinned_mapped<int32_t>(M, &t.gpos_d);
      t.is_image_h = pinned_mapped<int8_t>(M, &t.is_image_d); t.ids_h = pinned<int32_t>(M);
    }
    fprintf(stderr, "[runtime] prefill tile %d × %d rows (slots %zu)\n", opt_.prefill_tile, M, tiles_.size());
  }
  // H2 host tiles: the slot's device part (everything but h — pre_mix, idx, xq/xs, acc, pre_f/post_f/comb_f, ≈ 444 MiB at 16K rows) + h staging buffers
  //   (≤ 2, 640 MiB each at 16K rows) + pinned h (640 MiB per tile). If VRAM is short the tile count is reduced (absorbed), likewise on pinned failure. The free value in the log is before the session pool allocation.
  if (host_slots_ > 0 && tail_mode_for(M)) {
    const size_t hb = (size_t)M * hc * dim * 2;
    const size_t part = (size_t)M * hc * 4 * 3 + (size_t)M * (c.window + c.index_topk) * 4 + (size_t)M * dim + (size_t)M * dim / 32 + (size_t)M * dim * 4 +
                        (size_t)M * hc * hc * 4;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    int n = host_slots_;
    while (n > 0 && part * (size_t)n + hb * (size_t)std::min(2, n) > fr) --n;
    if (n < host_slots_) fprintf(stderr, "[runtime] ⚠️host tiles %d → %d: VRAM free %.0f MiB\n", host_slots_, n, fr / 1048576.0);
    if (n > 0) stager_ = std::make_unique<HostStager>(st_, n, hb, std::min(2, n));
    if (stager_ && stager_->n_host() < n) fprintf(stderr, "[runtime] ⚠️host tiles %d → %d: pinned host allocation failed\n", n, stager_->n_host());
    n = stager_ ? stager_->n_host() : 0;
    if (n == 0) stager_.reset();
    const size_t base = tiles_.size();
    tiles_.resize(base + (size_t)n);
    for (size_t i = base; i < tiles_.size(); ++i) {
      TileSlot& t = tiles_[i];  // h stays empty — a staging buffer is swapped in during visits
      t.pre_mix.alloc((size_t)M * hc * 4); t.idx.alloc((size_t)M * (c.window + c.index_topk) * 4);
      t.xq.alloc((size_t)M * dim); t.xs.alloc((size_t)M * dim / 32); t.acc.alloc((size_t)M * dim * 4);
      t.pre_f.alloc((size_t)M * hc * 4); t.post_f.alloc((size_t)M * hc * 4); t.comb_f.alloc((size_t)M * hc * hc * 4);
      t.route_ids_h = pinned_mapped<int32_t>((size_t)M * k, &t.route_ids_d); t.rw_h = pinned_mapped<float>((size_t)M * k, &t.rw_d);
      t.pos_h = pinned_mapped<int32_t>(M, &t.pos_d); t.visible_h = pinned_mapped<int32_t>(M, &t.visible_d); t.gpos_h = pinned_mapped<int32_t>(M, &t.gpos_d);
      t.is_image_h = pinned_mapped<int8_t>(M, &t.is_image_d); t.ids_h = pinned<int32_t>(M);
    }
    host_slots_ = n;
    if (n > 0) {
      cudaMemGetInfo(&fr, &tot);
      fprintf(stderr, "[runtime] prefill host tiles %d × %d rows (h %.0f MiB pinned each · %d staging) · slots %d · VRAM free %.0f MiB\n", n, M, hb / 1048576.0,
              stager_->n_stage(), prefill_slots(), fr / 1048576.0);
    }
  } else host_slots_ = 0;
  // H3: candidate blocks (layer 20 → 21+) and MTP capture for the second and later sequences running the upper layers — upper-layer rows ≤ max(tail, threshold−1)
  if (batch_prefill_ && prefill_slots() > 1) {
    const int rows_up = std::min(M, std::max(std::max(opt_.decoder_tail, 0), opt_.prefill_threshold - 1));
    mbufs_.resize((size_t)prefill_slots() - 1);
    for (MemberBufs& b : mbufs_) {
      if (c.cand_source_layer >= 0) b.cand.alloc((size_t)rows_up * c.cand_topk_blocks * 4);
      if (mtp_on_) b.mh.alloc(w->mh.n);
    }
    fprintf(stderr, "[runtime] batch prefill on: up to %d sequences per forward\n", prefill_slots());
  }
  work_addr_[0] = w->h.p; work_addr_[1] = w->acc.p; work_addr_[2] = (void*)w->route_ids_d; work_addr_[3] = (void*)w->pos_d;
  w_ = std::move(w);
  ElasticScope::setup(*this);  // Q1 HIVE_CACHE_ELASTIC (off = does nothing)
  // T11 HIVE_LAYER_YIELD: lower bound of handed-over rows = the most rows decode, verify, draft and MTP sync use in Work (same formula as the elastic small layout S); upper bound = Work row capacity.
  //   hived enables it (set_layer_yield) — before that ly_.on() is false and the yield points below do not even read the clock (same as without the switch).
  ly_.cap = w_->M;
  ly_.floor = std::max({c.window, std::max(1, std::min(opt_.max_batch, w_->M)), c.dspark_block + 1});
  if (model_.ckpt().has("vision.patch_embed.proj.weight") && opt_.vision) {
    vision_can_ = true;
    vision_max_patches_ = opt_.vision_max_patches > 0 ? opt_.vision_max_patches : c.vision_max_tokens * c.vision_downsample * c.vision_downsample;
    // Lazy loading: the encoder (0.8 GB of weights + ~0.9 GB of work buffers at 9216 patches) is loaded on the first image request and released after a
    //   period of disuse — that VRAM goes to expert slots (~85). The operating headroom (VRAM hived leaves free) must be large enough for this encoder.
    if (!opt_.vision_lazy) ensure_vision();
    else fprintf(stderr, "[runtime] vision encoder: lazy (max %d patches · loads on first image, released after %.0fs idle)\n", vision_max_patches_, opt_.vision_idle_ms / 1000.0);
  }
  { size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot); fprintf(stderr, "[runtime] max_chunk %d · max_ctx %lld · index Msub %d · VRAM free %.0f MiB\n", M, (long long)opt_.max_ctx, w_->Msub, fr / 1048576.0); }
}

void Runtime::ensure_vision() {
  if (vision_ || !vision_can_) return;
  const double t0 = now_ms();
  vision_ = std::make_unique<Vision>(model_, *blas_, st_, vision_max_patches_);
  size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
  fprintf(stderr, "[runtime] vision encoder loaded (max %d patches) in %.0f ms · VRAM free %.0f MiB\n", vision_max_patches_, now_ms() - t0, fr / 1048576.0);
  vision_last_use_ = now_ms();
}

void Runtime::release_vision_if_idle() {
  if (!vision_ || !opt_.vision_lazy) return;
  if (now_ms() - vision_last_use_ < opt_.vision_idle_ms) return;
  CUDA_CHECK(cudaStreamSynchronize(st_));
  vision_.reset();
  fprintf(stderr, "[runtime] vision encoder released (idle)\n");
}

// ---- expert usage tracing (HIVE_EXPERT_TRACE; record format in runtime.h) ----
void Runtime::xtrace_step(int kind, int M, Seq* const* seqs, int n) {
  if (store_.cache_policy()) {  // D4 HIVE_CACHE_POLICY=seq: rows of this forward → sequences (every forward head passes here — the same point as trace recording)
    uint32_t u[64];
    const int m = std::min(n, 64);  // forward_batch rows ≤ max_batch ≤ 64 — any extra rows are treated as unowned
    for (int i = 0; i < m; ++i) u[i] = seqs[i]->uid;
    store_.set_row_owners(u, m, kind == 1);
  }
  if (!xtrace_) return;
  const double t = hive::Millis(std::chrono::system_clock::now().time_since_epoch()).count();
  const uint8_t h[2] = {'S', (uint8_t)kind};
  // With tiled prefill (--prefill-tile ≥ 4) M ≥ 65536 would saturate the u16 M at 65535 and lose the real M. The format stays the same; only when
  //   M ≥ 65535, M = 0xFFFF (marker) and one extra tag is appended: last tag {uid 0xFFFFFFFF, pos = real M (u32)}, with n incremented by one.
  //   Record length is still 14 + 8·n, so older analyzers skip it as before (analyzers read per-row tags this way only for kind 0, and kind 0 has M ≤ max_batch, so it never carries the marker).
  const bool big_m = M >= 65535;
  const uint16_t mn[2] = {(uint16_t)std::min(M, 65535), (uint16_t)(n + (big_m ? 1 : 0))};
  fwrite(h, 1, 2, xtrace_); fwrite(mn, 2, 2, xtrace_); fwrite(&t, 8, 1, xtrace_);
  for (int i = 0; i < n; ++i) { const uint32_t tag[2] = {seqs[i]->uid, (uint32_t)seqs[i]->pos}; fwrite(tag, 4, 2, xtrace_); }
  if (big_m) { const uint32_t ext[2] = {0xFFFFFFFFu, (uint32_t)M}; fwrite(ext, 4, 2, xtrace_); }
  if (t - xtrace_flush_ms_ > 1000.0) { fflush(xtrace_); xtrace_flush_ms_ = t; }
}
void Runtime::xtrace_layer(int l, int M, int k, const int32_t* ids) {
  if (route_cap_) {  // R1 test-only routing capture (set_route_capture — off by default): [layer, M, k, ids(M·k)]
    std::vector<int32_t> v{l, M, k};
    v.insert(v.end(), ids, ids + (size_t)M * k);
    route_log_.push_back(std::move(v));
  }
  if (!xtrace_) return;
  const uint8_t h[2] = {'L', (uint8_t)l};
  const uint16_t km[2] = {(uint16_t)k, (uint16_t)M};
  fwrite(h, 1, 2, xtrace_); fwrite(km, 2, 2, xtrace_);
  uint16_t buf[512];
  for (int i0 = 0; i0 < M * k; i0 += 512) {
    const int n = std::min(512, M * k - i0);
    for (int i = 0; i < n; ++i) buf[i] = (uint16_t)ids[i0 + i];
    fwrite(buf, 2, n, xtrace_);
  }
}
void Runtime::xtrace_hist(int l, int M, int k, int E, const int32_t* ids) {
  if (!xtrace_) return;
  std::vector<uint16_t> cnt(E, 0);
  for (int i = 0; i < M * k; ++i) { uint16_t& v = cnt[ids[i]]; if (v < 65535) ++v; }
  const uint8_t h[2] = {'H', (uint8_t)l};
  const uint16_t e16 = (uint16_t)E;
  const uint32_t m32 = (uint32_t)M;
  fwrite(h, 1, 2, xtrace_); fwrite(&e16, 2, 1, xtrace_); fwrite(&m32, 4, 1, xtrace_); fwrite(cnt.data(), 2, E, xtrace_);
}

void Runtime::pmark(const char* name) {
  if (!profile_ || capturing_ || (profile_every_ > 1 && step_ % profile_every_ != 0)) return;  // no event creation during capture (in-graph sections are timed as a whole)
  cudaEvent_t e;
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(e, st_));
  prof_.ev.push_back(e);
  prof_.names.push_back(name);
}
void Runtime::preport(int M) {
  if (!profile_ || prof_.ev.size() < 2) return;
  CUDA_CHECK(cudaEventSynchronize(prof_.ev.back()));
  std::map<std::string, float> acc;
  std::vector<std::string> order;
  for (size_t i = 1; i < prof_.ev.size(); ++i) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, prof_.ev[i - 1], prof_.ev[i]));
    if (!acc.count(prof_.names[i])) order.push_back(prof_.names[i]);
    acc[prof_.names[i]] += ms;
  }
  float total = 0;
  for (auto& [k, v] : acc) total += v;
  fprintf(stderr, "[profile M=%d] total %.2f ms:", M, total);
  for (auto& k : order) fprintf(stderr, " %s %.2f", k.c_str(), acc[k]);
  fprintf(stderr, "%s\n", cw_line_.c_str());  // E4 demand-copy and promotion bytes (only when measured this step — cw_collect)
  for (auto e : prof_.ev) cudaEventDestroy(e);
  prof_.ev.clear();
  prof_.names.clear();
}

Runtime::~Runtime() {
  if (EngramSsd* ssd = store_.engram_ssd()) ssd->quiesce();  // HIVE_ENGRAM_SSD: read-ahead plans read eh_ (about to go away), so finish the remaining work
  sg_.reset();  // G1 step graph (dispatcher thread, graph, pinned) — before the streams and Work
  dov_.reset();  // O1 events
  if (snapshot_st_) { cudaStreamSynchronize(snapshot_st_); cudaStreamDestroy(snapshot_st_); }
  if (xtrace_) { fclose(xtrace_); xtrace_ = nullptr; }
  if (w_) tile_select(0);  // restore Work before freeing any pinned memory (if left swapped, slot buffers would be freed twice)
  if (w_) ElasticScope::teardown(*this);  // Q1: back to the small layout, detach non-owned buffers inside the elastic region (off = does nothing)
  if (w_) {
    for (void* p : {(void*)w_->ids_h, (void*)w_->route_ids_h, (void*)w_->pos_h, (void*)w_->gpos_h, (void*)w_->visible_h, (void*)w_->g_rows_h, (void*)w_->g_rw_h,
                    (void*)w_->rw_h, (void*)w_->cpu_rows_h, (void*)w_->cpu_out_h, (void*)w_->xq_h, (void*)w_->xs_h, (void*)w_->next_h,
                    (void*)w_->eg_v_h, (void*)w_->eg_s_h, (void*)w_->is_image_h, (void*)w_->emb_h, (void*)w_->kvrows_h,
                    (void*)w_->kptrs_h, (void*)w_->trows_h, (void*)w_->dsti_h, (void*)w_->ringp_h, (void*)w_->dstp_h, (void*)w_->dstk_h, (void*)w_->skv_h,
                    (void*)w_->ssc_h, (void*)w_->next_b_h, (void*)w_->gdesc_h, (void*)w_->tbl_h, (void*)w_->mpos_h, (void*)w_->mhid_h,
                    (void*)w_->dids_h, (void*)w_->dconf_h, (void*)w_->lringp_h, (void*)w_->ringp1_h})
      if (p) cudaFreeHost(p);
    for (void* p : w_->mringp_h) if (p) cudaFreeHost(p);
  }
  for (TileSlot& t : tiles_)
    for (void* p : {(void*)t.route_ids_h, (void*)t.rw_h, (void*)t.pos_h, (void*)t.visible_h, (void*)t.gpos_h, (void*)t.is_image_h, (void*)t.ids_h})
      if (p) cudaFreeHost(p);
  for (void* p : {(void*)cand_it_h_, (void*)cand_idx_h_, (void*)cand_val_h_, (void*)cand_max_h_, (void*)cand_sum_h_}) if (p) cudaFreeHost(p);
  if (w_) { free(w_->a_f_h); free(w_->a_s_h); free(w_->scratch_h); }
  for (auto& [key, g] : graphs_map_) if (g.exec) cudaGraphExecDestroy(g.exec);
  if (promo_evt_) cudaEventDestroy(promo_evt_);
  if (promo_gate_) cudaEventDestroy(promo_gate_);  // E4
  for (auto e : cw_ev_) cudaEventDestroy(e);
  if (dma_e0_) cudaEventDestroy(dma_e0_);
  if (prefetch_begin_) cudaEventDestroy(prefetch_begin_);
  if (prefetch_end_) cudaEventDestroy(prefetch_end_);
  if (dma_e1_) cudaEventDestroy(dma_e1_);
  for (cudaEvent_t e : {split_g0_, split_g1_, split_c0_, split_c1_}) if (e) cudaEventDestroy(e);  // G2 (created only when on)
  for (auto& pr : split_d_) for (cudaEvent_t e : pr) if (e) cudaEventDestroy(e);
  for (auto e : stage_copied_) cudaEventDestroy(e);
  for (auto e : stage_freed_) cudaEventDestroy(e);
  if (st_) cudaStreamDestroy(st_);
  if (side_) cudaStreamDestroy(side_);
  if (promo_) cudaStreamDestroy(promo_);
  if (hc_side_) cudaStreamDestroy(hc_side_);
  for (int i = 0; i < 2; ++i) { if (hc_fork_[i]) cudaEventDestroy(hc_fork_[i]); if (hc_join_[i]) cudaEventDestroy(hc_join_[i]); }
}

std::unique_ptr<Seq> Runtime::new_seq() const {
  const Config& c = model_.cfg();
  auto s = std::make_unique<Seq>();
  s->cap = opt_.max_ctx;
  s->ring.resize(model_.n_loaded_layers());
  for (int l = 0; l < model_.n_loaded_layers(); ++l) {
    s->ring[l].alloc((size_t)c.window * c.head_dim * 2);
    CUDA_CHECK(cudaMemset(s->ring[l].p, 0, s->ring[l].n));
    if (c.is_kv_source(l)) {
      int r = c.ratio(l);
      size_t T = (size_t)(s->cap / r + 2);
      // Packed fp4 (kv_pack.h): compressed KV 288 B/row (1,024 unpacked), indexer keys 80 B/row (256 unpacked) — session KV 820 → ~250 MiB (262K)
      HIVE_CHECK(c.head_dim == kvp::COMP_D && c.index_head_dim == kvp::IDX_D, "kv_pack layout assumes D=512, Di=128");
      s->comp_cache[l].alloc(T * kvp::COMP_ROW);
      s->idx_k_cache[l].alloc(T * kvp::IDX_ROW);
      if (r > 1) {
        s->comp_state_kv[l].alloc((size_t)r * c.head_dim * 4);
        s->comp_state_score[l].alloc((size_t)r * c.head_dim * 4);
        CUDA_CHECK(cudaMemset(s->comp_state_kv[l].p, 0, s->comp_state_kv[l].n));
        std::vector<float> ninf((size_t)r * c.head_dim, -INFINITY);
        CUDA_CHECK(cudaMemcpy(s->comp_state_score[l].p, ninf.data(), ninf.size() * 4, cudaMemcpyHostToDevice));
      }
    }
  }
  if (mtp_on_) {
    s->mtp_ring.resize(model_.n_mtp());
    for (auto& r : s->mtp_ring) { r.alloc((size_t)c.window * c.head_dim * 2); CUDA_CHECK(cudaMemset(r.p, 0, r.n)); }
    s->mtp_hidden.alloc((size_t)ntgt_ * c.dim * 2);
  }
  return s;
}

void Runtime::reset_seq(Seq& s) const {
  if (s.snapshot_fence) { s.snapshot_fence->wait(); s.snapshot_fence.reset(); }
  s.uid = Seq::next_uid(); s.generation = s.uid;
  const Config& c = model_.cfg();
  s.pos = 0;
  s.broken = false;
  s.tokens.clear();
  s.engram_history.clear();
  for (auto& r : s.ring) CUDA_CHECK(cudaMemsetAsync(r.p, 0, r.n, st_));
  for (auto& [l, b] : s.comp_state_kv) CUDA_CHECK(cudaMemsetAsync(b.p, 0, b.n, st_));
  for (auto& [l, b] : s.comp_state_score) {
    std::vector<float> ninf(b.n / 4, -INFINITY);
    CUDA_CHECK(cudaMemcpyAsync(b.p, ninf.data(), b.n, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));  // host vector lifetime
  }
  for (auto& r : s.mtp_ring) CUDA_CHECK(cudaMemsetAsync(r.p, 0, r.n, st_));
  s.mtp_hidden_valid = false;
  s.mtp_pos = -1;
  (void)c;
  CUDA_CHECK(cudaStreamSynchronize(st_));
}

// ---- Sequence images (conversation KV reuse, kept in RAM) ----
namespace {
int64_t comp_rows_used(const Config& c, int l, int64_t pos) {
  const int r = c.ratio(l);
  return r > 1 ? pos / r : pos;  // compress_source: only completed groups [start/r, end/r) are written (every row when r ≤ 1)
}
void h2d(void* dst, const HostImageBuffer& src, size_t cap) {
  HIVE_CHECK(src.size() <= cap, "seq image larger than the target buffer");
  size_t off = 0;
  for (auto& segment : src.segments) {
    CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(dst) + off, segment.data.get(), segment.n, cudaMemcpyHostToDevice));
    off += segment.n;
  }
}
}  // namespace

void Runtime::quiesce_after_host_error() {
  store_.wait_jobs();
  store_.flush_promotions();  // E4: issue the remaining slices of decided promotions (batch events before issue are not recorded yet) so the sync and commit below cover them
  CUDA_CHECK(cudaStreamSynchronize(st_));
  CUDA_CHECK(cudaStreamSynchronize(side_));
  CUDA_CHECK(cudaStreamSynchronize(promo_));
  CUDA_CHECK(cudaStreamSynchronize(snapshot_st_));
  store_.commit_pending();
  if (stager_) stager_->sync();  // H2 host tile upload/download streams
  verify_=false; verify_logits_out_=nullptr; v_M_=0;
  step_miss_.clear(); draft_miss_.clear(); pf_.clear(); pf_l_=-1;
  have_candidates_=false; dma_evt_pending_=false;
}

void Runtime::save_image(const Seq& s, SeqImage& img, const SeqImage* base) const {
  const Config& c = model_.cfg();
  const bool async = env_on("HIVE_CKPT_ASYNC");
  if (!env_on("HIVE_CKPT_DELTA") || !base || base == &img || base->generation != s.generation || base->pos > s.pos ||
      base->tokens.size() > s.tokens.size() || !std::equal(base->tokens.begin(), base->tokens.end(), s.tokens.begin())) base = nullptr;
  if (s.snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, s.snapshot_fence->event, 0));
  if (!async) CUDA_CHECK(cudaStreamSynchronize(st_));
  cudaEvent_t fork = nullptr;
  if (async) {
    CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(fork, st_));
    CUDA_CHECK(cudaStreamWaitEvent(snapshot_st_, fork, 0));
    CUDA_CHECK(cudaEventDestroy(fork));
  }
  // Default member assignment releases buffers before the trailing fence member.
  // A caller may reuse an image whose D2H is still in flight.
  if (img.fence) img.fence->wait();
  img = SeqImage{};
  if (async) img.fence = std::make_shared<SnapshotFence>();
  struct RecordSnapshot {
    std::shared_ptr<SnapshotFence> fence; cudaStream_t stream;
    ~RecordSnapshot() { if (fence) CUDA_CHECK(cudaEventRecord(fence->event, stream)); }
  } record{img.fence, snapshot_st_};
  if (async) s.snapshot_fence = img.fence;
  size_t transferred = 0;
  // CKPT_ASYNC pinned failure must not abort: calling cudaHostAlloc under CUDA_CHECK for every async snapshot would let a single pinned/memlock exhaustion
  //   abort the whole daemon over state that only serves reuse. Fallback: once a pinned allocation fails in this snapshot, the remaining pieces use
  //   pageable memory, copied on snapshot_st_ and then waited for (synchronous — pageable memory cannot leave lifetime and ordering to events). Logged once
  //   per process. The default (synchronous) path is unchanged. Async copies of pinned pieces already issued are covered by the fence at the end.
  bool pageable = !async;
  auto host_alloc = [&](size_t bytes) -> std::shared_ptr<uint8_t> {
    if (!pageable) {
      try {
        if (snapshot_pool_) return snapshot_pool_->acquire(bytes);
        uint8_t* p = nullptr;
        if (cudaHostAlloc(reinterpret_cast<void**>(&p), bytes, cudaHostAllocDefault) == cudaSuccess && p)
          return std::shared_ptr<uint8_t>(p, [](uint8_t* q) { cudaFreeHost(q); });
      } catch (const std::bad_alloc&) {}
      (void)cudaGetLastError();  // clear the failed call's error so a later CUDA_CHECK(cudaGetLastError()) does not abort on it
      pageable = true;
      static std::atomic<bool> logged{false};
      if (!logged.exchange(true)) fprintf(stderr, "[hive] ⚠️snapshot pinned allocation failed (%zu B) — absorbed with pageable synchronous copies from this snapshot on (further warnings suppressed)\n", bytes);
    }
    return std::shared_ptr<uint8_t>(new uint8_t[bytes], std::default_delete<uint8_t[]>());
  };
  auto d2h = [&](HostImageBuffer& dst, const void* src, size_t n, const HostImageBuffer* prev = nullptr) {
    const size_t prefix = dst.extend(prev, n, host_alloc);
    if (n == prefix) return;
    auto* out = dst.segments.back().data.get();
    const auto* in = static_cast<const uint8_t*>(src) + prefix;
    if (!async) CUDA_CHECK(cudaMemcpy(out, in, n - prefix, cudaMemcpyDeviceToHost));
    else {
      CUDA_CHECK(cudaMemcpyAsync(out, in, n - prefix, cudaMemcpyDeviceToHost, snapshot_st_));  // snapshot_st_ is ordered after st_ by the fork above
      if (pageable) CUDA_CHECK(cudaStreamSynchronize(snapshot_st_));  // pageable pieces must be filled before returning
    }
    transferred += n - prefix;
  };
  img.pos = s.pos;
  img.generation = s.generation;
  img.tokens = s.tokens;
  img.engram_history = s.engram_history;
  img.ring.resize(s.ring.size());
  for (size_t l = 0; l < s.ring.size(); ++l) d2h(img.ring[l], s.ring[l].p, s.ring[l].n);
  for (auto& [l, b] : s.comp_cache) {
    const size_t n = std::min<size_t>(b.n, (size_t)comp_rows_used(c, l, s.pos) * kvp::COMP_ROW);
    d2h(img.comp[l], b.p, n, base && base->comp.count(l) ? &base->comp.at(l) : nullptr);
  }
  for (auto& [l, b] : s.idx_k_cache) {
    const size_t n = std::min<size_t>(b.n, (size_t)comp_rows_used(c, l, s.pos) * kvp::IDX_ROW);
    d2h(img.idx[l], b.p, n, base && base->idx.count(l) ? &base->idx.at(l) : nullptr);
  }
  for (auto& [l, b] : s.comp_state_kv) d2h(img.st_kv[l], b.p, b.n);
  for (auto& [l, b] : s.comp_state_score) d2h(img.st_score[l], b.p, b.n);
  img.mtp_ring.resize(s.mtp_ring.size());
  for (size_t i = 0; i < s.mtp_ring.size(); ++i) d2h(img.mtp_ring[i], s.mtp_ring[i].p, s.mtp_ring[i].n);
  if (s.mtp_hidden.p) d2h(img.mtp_hidden, s.mtp_hidden.p, s.mtp_hidden.n);
  img.mtp_hidden_valid = s.mtp_hidden_valid;
  img.mtp_pos = s.mtp_pos;
  if (env_on("HIVE_TRACE_CACHE")) fprintf(stderr, "[snapshot] pos %lld d2h_bytes %zu host_bytes %zu async %d delta %d\n",
      (long long)s.pos, transferred, img.bytes(), async, base != nullptr);
}

void Runtime::load_image(Seq& s, const SeqImage& img) const {
  if (s.snapshot_fence) { s.snapshot_fence->wait(); s.snapshot_fence.reset(); }
  if (img.fence) img.fence->wait();
  s.broken = false;  // restored wholesale from the saved image
  CUDA_CHECK(cudaStreamSynchronize(st_));
  HIVE_CHECK(img.ring.size() == s.ring.size() && img.mtp_ring.size() == s.mtp_ring.size() && img.pos <= s.cap, "seq image shape mismatch");
  for (size_t l = 0; l < s.ring.size(); ++l) h2d(s.ring[l].p, img.ring[l], s.ring[l].n);
  auto put = [](std::map<int, DevBuf>& dst, const std::map<int, HostImageBuffer>& src, const char* what) {
    HIVE_CHECK(dst.size() == src.size(), what);
    for (auto& [l, v] : src) { auto it = dst.find(l); HIVE_CHECK(it != dst.end(), what); h2d(it->second.p, v, it->second.n); }
  };
  put(s.comp_cache, img.comp, "seq image comp layers");
  put(s.idx_k_cache, img.idx, "seq image idx layers");
  put(s.comp_state_kv, img.st_kv, "seq image compressor layers");
  put(s.comp_state_score, img.st_score, "seq image compressor layers");
  for (size_t i = 0; i < s.mtp_ring.size(); ++i) h2d(s.mtp_ring[i].p, img.mtp_ring[i], s.mtp_ring[i].n);
  if (s.mtp_hidden.p && !img.mtp_hidden.empty()) h2d(s.mtp_hidden.p, img.mtp_hidden, s.mtp_hidden.n);
  s.pos = img.pos;
  s.generation = img.generation;
  s.tokens = img.tokens;
  s.engram_history = img.engram_history;
  s.mtp_hidden_valid = img.mtp_hidden_valid && !img.mtp_hidden.empty();
  s.mtp_pos = img.mtp_pos;
}

void Runtime::dump(const std::string& name, const void* dev, size_t bytes) {
  if (opt_.dump_dir.empty() || (graphs_ && !dump_final_)) return;  // in graph mode only the final dumps
  std::vector<uint8_t> buf(bytes);
  CUDA_CHECK(cudaStreamSynchronize(st_));
  CUDA_CHECK(cudaMemcpy(buf.data(), dev, bytes, cudaMemcpyDeviceToHost));
  std::ofstream f(opt_.dump_dir + "/s" + std::to_string(step_) + "_" + name, std::ios::binary);
  f.write((const char*)buf.data(), bytes);
}
void Runtime::dump_host(const std::string& name, const void* host, size_t bytes) {
  if (opt_.dump_dir.empty()) return;
  std::ofstream f(opt_.dump_dir + "/s" + std::to_string(step_) + "_" + name, std::ios::binary);
  f.write((const char*)host, bytes);
}

// Per-layer injection (validation): replace layer l's input h[M,hc,dim] and pre_mix[M,hc] with the golden output of layer l-1 — measures this layer's error alone.
void Runtime::inject_layer_input(int l, int M) {
  if (opt_.inject_dir.empty() || l == 0) return;
  HIVE_CHECK(!graphs_, "inject requires eager mode (dump)");
  const Config& c = model_.cfg();
  Work& w = *w_;
  const std::string pre = opt_.inject_dir + "/s" + std::to_string(step_) + "_";
  std::ifstream fh(pre + "h_L" + std::to_string(l - 1) + ".f32", std::ios::binary);
  std::ifstream fp(pre + "premix_L" + std::to_string(l - 1) + ".f32", std::ios::binary);
  if (!fh || !fp) return;  // skip when there is no golden for this step
  const size_t n = (size_t)M * c.hc * c.dim;
  std::vector<float> hf(n), pm((size_t)M * c.hc);
  fh.read((char*)hf.data(), n * 4);
  fp.read((char*)pm.data(), pm.size() * 4);
  HIVE_CHECK((size_t)fh.gcount() == n * 4 && (size_t)fp.gcount() == pm.size() * 4, "inject file size");
  std::vector<bf16> hb(n);
  for (size_t i = 0; i < n; ++i) hb[i] = f2bf(hf[i]);
  CUDA_CHECK(cudaStreamSynchronize(st_));
  CUDA_CHECK(cudaMemcpy(w.h.p, hb.data(), n * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(w.pre_mix.p, pm.data(), pm.size() * 4, cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------------------------------------------------
// R2 HIVE_PREFILL_PROF (env_on; off = pprof_ is nullptr and each hook is a single pointer check): time breakdown of one prefill forward (single sequence, not verify).
//   Per layer: 4 host-clock points (layer start, end of router launch = right before the sync, end of sync, end of experts) + 3 st_ events (layer start,
//   end of router, end of layer) + ForwardStats deltas.
//   GPU intervals: front = layer start → end of router; moe = end of router → end of layer (experts + CPU accumulation + tail — includes GPU idle while the
//   host prepares the CPU share after the sync, and DMA waits); gap = end of previous layer → start of this layer (GPU idle between layers = time the host
//   issued late — except that the gap of the first tail-group layer includes the GPU work of the preceding prepare_decoder_tail (hc, compressor, indexer keys
//   and row moves for all rows of layer 20)). GPU detail inside moe (gpu_experts/cpu_experts) comes from the [profile] line of HIVE_PROFILE=1 (enable both).
//   Output: per chunk a [prefill-prof] header line + one line per layer group (enc = original rows, tail = tail rows).
//   Events are reused from a pool (~3·layers per chunk — created only once). Samples are read after the sync at the end of the forward, so instrumentation
//   never blocks the GPU (host cost = 3 event records + 4 now_ms per layer).
//   (Placement: must stay outside the tail slice of harness.runtime_source (tail_mode_for → forward) — the fake-runtime test has no now_ms)
struct Runtime::PrefillProf {
  bool on = false;
  std::vector<cudaEvent_t> pool;
  size_t used = 0;
  struct Layer {
    int l = 0, M = 0; bool short_moe = false;
    double h0 = 0, h_router = 0, h_synced = 0, h_moe = 0, h_end = 0;
    cudaEvent_t e0 = nullptr, e_router = nullptr, e_end = nullptr;
    int routed = 0, hit = 0, cpu = 0, streamed = 0, dma_rows = 0; double cpu_wait = 0, cpu_span = 0; uint64_t pf = 0;
  };
  std::vector<Layer> layers;
  int M = 0; int64_t pos = 0; double t0 = 0, t_embed = 0, t_layers = 0;
  cudaEvent_t e_start = nullptr, e_embed = nullptr, e_layers = nullptr;
  ForwardStats base{};  // totals at layer start (stay 0 without stats)
  ~PrefillProf() { for (cudaEvent_t e : pool) cudaEventDestroy(e); }
  cudaEvent_t ev(cudaStream_t st) {
    if (used == pool.size()) { cudaEvent_t e = nullptr; CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDefault)); pool.push_back(e); }
    cudaEvent_t e = pool[used++];
    CUDA_CHECK(cudaEventRecord(e, st));
    return e;
  }
  void begin(int m, int64_t p, cudaStream_t st) { on = true; used = 0; layers.clear(); M = m; pos = p; t0 = now_ms(); e_start = ev(st); }
  void embed(cudaStream_t st) { if (on) { t_embed = now_ms(); e_embed = ev(st); } }
  void layer_begin(int l, int m, bool short_moe, const ForwardStats* s, uint64_t pf, cudaStream_t st) {
    if (!on) return;
    Layer x; x.l = l; x.M = m; x.short_moe = short_moe; x.h0 = now_ms(); x.e0 = ev(st);
    x.h_router = x.h_synced = x.h_moe = x.h0;
    x.pf = pf;
    if (s) base = *s;
    layers.push_back(x);
  }
  void router(cudaStream_t st) { if (on && !layers.empty()) { layers.back().h_router = now_ms(); layers.back().e_router = ev(st); } }
  void synced() { if (on && !layers.empty()) layers.back().h_synced = now_ms(); }
  void moe_done() { if (on && !layers.empty()) layers.back().h_moe = now_ms(); }
  void layer_end(const ForwardStats* s, uint64_t pf, cudaStream_t st) {
    if (!on || layers.empty()) return;
    Layer& x = layers.back();
    x.h_end = now_ms(); x.e_end = ev(st);
    if (!x.e_router) { x.e_router = x.e_end; x.h_router = x.h_synced = x.h_moe = x.h_end; }  // layer without a router mark (defensive — zero-length intervals)
    if (s) {
      x.routed = s->n_routed - base.n_routed; x.hit = s->n_hit - base.n_hit; x.cpu = s->n_cpu - base.n_cpu; x.streamed = s->n_streamed - base.n_streamed;
      x.dma_rows = s->n_dma_rows - base.n_dma_rows; x.cpu_wait = s->ms_cpu_wait - base.ms_cpu_wait; x.cpu_span = s->ms_cpu_span - base.ms_cpu_span;
    }
    x.pf = pf - x.pf;
  }
  void layers_done(cudaStream_t st) { if (on) { t_layers = now_ms(); e_layers = ev(st); } }
  // End of forward (after the sync): sum into two groups (enc = M rows, tail = fewer rows) and print. dfrac = DMA share at the end (short prefill, short streaming kinds)
  void report(int tail_rows, double t_end, float dfrac_short, float dfrac_sstream) {
    if (!on) return;
    on = false;
    auto el = [](cudaEvent_t a, cudaEvent_t b) { float ms = 0.f; if (a && b) CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); return (double)ms; };
    struct G { int n = 0, n_short = 0, rows = 0; double hf = 0, hs = 0, hm = 0, ht = 0, gf = 0, gm = 0, gap = 0, cw = 0, cs = 0;
               long routed = 0, hit = 0, cpu = 0, str = 0, dmar = 0; uint64_t pf = 0; } g[2];
    cudaEvent_t prev = e_embed ? e_embed : e_start;
    for (const Layer& x : layers) {
      G& a = g[x.M < M ? 1 : 0];
      ++a.n; a.n_short += x.short_moe; a.rows = x.M;
      a.hf += x.h_router - x.h0; a.hs += x.h_synced - x.h_router; a.hm += x.h_moe - x.h_synced; a.ht += x.h_end - x.h_moe;
      a.gf += el(x.e0, x.e_router); a.gm += el(x.e_router, x.e_end); a.gap += el(prev, x.e0);
      a.cw += x.cpu_wait; a.cs += x.cpu_span; a.routed += x.routed; a.hit += x.hit; a.cpu += x.cpu; a.str += x.streamed; a.dmar += x.dma_rows; a.pf += x.pf;
      prev = x.e_end;
    }
    const double gpu_all = el(e_start, e_layers);
    fprintf(stderr, "[prefill-prof M=%d pos=%lld tail=%d] wall %.1f ms · embed %.1f · layers %.1f · head+sync %.1f · gpu(start→layers end) %.1f · dma frac short %.2f sstream %.2f\n",
            M, (long long)pos, tail_rows, t_end - t0, t_embed - t0, t_layers - t_embed, t_end - t_layers, gpu_all, dfrac_short, dfrac_sstream);
    for (int i = 0; i < 2; ++i) {
      const G& a = g[i];
      if (!a.n) continue;
      fprintf(stderr, "[prefill-prof M=%d] %s layers %d (rows %d, short-moe %d): host front %.1f · sync wait %.1f · moe host %.1f (cpu wait %.1f · cpu span %.1f) · tail %.1f"
              " | gpu front %.1f · gpu moe %.1f · gpu gap %.1f | rows routed %ld hit %ld cpu %ld dma %ld · streamed %ld · prefetch %llu · per layer %.2f ms\n",
              M, i ? "tail" : "enc", a.n, a.rows, a.n_short, a.hf, a.hs, a.hm, a.cw, a.cs, a.ht, a.gf, a.gm, a.gap, a.routed, a.hit, a.cpu, a.dmar, a.str,
              (unsigned long long)a.pf, (a.hf + a.hs + a.hm + a.ht) / a.n);
    }
  }
};

bool Runtime::tail_mode_for(int M) const {
  const Config& c = model_.cfg();
  const int tail_layer = c.kv_source_layers.empty() ? -1 : *std::max_element(c.kv_source_layers.begin(), c.kv_source_layers.end());
  const int tail_len = opt_.decoder_tail > 0 ? opt_.decoder_tail : 0;
  return tail_len > 0 && M >= opt_.prefill_threshold && M > tail_len && tail_layer >= 0 && tail_layer < model_.n_loaded_layers();
}
// H2/H3: number of sub-chunks for a chunk of M rows = T of tile_plan (1 when M ≤ max_chunk)
int Runtime::slots_for(int M) const { return M <= opt_.max_chunk ? 1 : (M + opt_.max_chunk - 1) / opt_.max_chunk; }

int32_t Runtime::forward(Seq& seq, const int32_t* ids, int M, const std::vector<ImageInput>* images, std::vector<float>* logits_out,
                         ForwardStats* stats, bool upper_needed) {
  deferred_flush();  // HIVE_DECODE_DEFER: a pending deferred batch reads the activation table this forward rewrites
  ElasticScope elastic_scope(*this, M, false);  // Q1 HIVE_CACHE_ELASTIC: rows > S reclaim the elastic slots and switch to the big layout (off or rows ≤ S = does nothing)
  if (seq.snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, seq.snapshot_fence->event, 0));
  const int tiles = std::max(1, opt_.prefill_tile);
  if (host_slots_ > 0 && M > w_->M * tiles && (!images || images->empty()) && !verify_) {  // H2: a chunk larger than the resident tiles = one forward including host tiles
    std::vector<PrefillPart> parts(1);
    parts[0].seq = &seq; parts[0].ids = ids; parts[0].M = M; parts[0].logits_out = logits_out; parts[0].upper_needed = upper_needed;
    forward_multi(parts, stats);
    if (stats) stats->tail_rows = parts[0].tail_rows;
    return parts[0].next;
  }
  HIVE_CHECK(M >= 1 && M <= w_->M * tiles, "chunk size");
  const bool tiled = M > w_->M;  // C2: a chunk larger than max_chunk = tiled (several sub-chunks, encoder layers first)
  if (M == 1 && (!images || images->empty())) {  // a single token = batched decode path (one sequence)
    std::vector<Seq*> seqs{&seq};
    std::vector<int32_t> next;
    std::vector<std::vector<float>> lg;
    forward_batch(seqs, ids, next, logits_out ? &lg : nullptr, stats);
    if (logits_out && !lg.empty()) *logits_out = std::move(lg[0]);
    return next[0];
  }
  { Seq* sp = &seq; xtrace_step(verify_ ? 2 : 1, M, &sp, 1); }
  struct BrokenGuard { Seq* q; int ex; ~BrokenGuard() { if (std::uncaught_exceptions() > ex) q->broken = true; } } broken_guard{&seq, std::uncaught_exceptions()};
  HIVE_CHECK(seq.pos + M <= seq.cap, "context overflow");
  const Config& c = model_.cfg();
  Work& w = *w_;
  int64_t start_pos = seq.pos;
  const double t0 = now_ms();
  // Q3 HIVE_PREFILL_SMALL (spo_.small_rows; 0 = off → false → the code below is the base path): chunks below prefill_threshold otherwise run all 40 layers
  //   over all rows on the decode-policy expert path (moe_decode_experts, kDmaShort, per-layer DMA ≤ HIVE_SHORT_DMA_CAP and staging slots — slots cannot be
  //   reused within a layer — and every other miss on the CPU). Evidence (service A/B, same build and cache): 906-token TTFT 7.31–7.50 s while 4,213 tokens
  //   took 4.19–4.21 s and 15K 5.5 s — a cliff where the shorter prompt is slower. With 1K rows × top-6, hundreds of misses per layer nearly all go to the
  //   CPU while PCIe carries only 8 per layer (fixed slots, not a ring).
  //   On = treat this chunk as prefill (in_prefill_): experts = moe_experts (staging **ring** DMA, pre-copy, misses with fewest rows go to the CPU first — the
  //   same path as long prefills); promotion slices are all issued at the head (flush below — this path never calls promo_pump). Tail mode (tail_mode_for —
  //   rows ≥ prefill_threshold) is **not** changed: every layer keeps all rows, so decoder replay (P6 — changes the output) is not extended below it.
  //   Post-step promotion (promote_after_step — M_orig < threshold) stays as is (moe_experts_multi fills the miss list; see the Q3 comment). Math: same
  //   experts, same weights, same activation quantization (act_quant_fp8(xn), 32-block y quantization) — the only possible difference is fp32 accumulation
  //   order: (a) GPU expert kernels (prefill gemm_bs/grouped GEMM vs the decode mx_grouped/F2/F3 chain), (b) which misses go to the CPU (CPU and GPU kernels
  //   accumulate in different orders), (c) expert output → acc accumulation order (accum_bf16_rows vs accum_bf16_rows_seq / atomic f32 accumulation).
  //   Attention, compressor, indexer, hc and MTP inputs are the same. Outside the conditions (verify, batched decode, tiles, below the floor, rows ≥ threshold) the base path runs.
  const bool small_fwd = sp::small_forward(spo_.small_rows, verify_, batch_ != nullptr, tiled, M, opt_.prefill_threshold);
  if (M >= opt_.prefill_threshold || small_fwd) store_.flush_promotions();  // E4: prefill chunks do not take per-layer pacing (promo_pump) — issue all remaining promotion slices at the head (Q3 SMALL too)
  store_.commit_pending();
  pmark("start");
  // R2 HIVE_PREFILL_PROF: start the breakdown for this forward (multi-row chunk, not verify) (off = no pprof_ — every hook below is just a pointer check)
  if (spo_.prof && !pprof_) pprof_ = std::make_shared<PrefillProf>();
  if (pprof_) { pprof_->on = false; if (spo_.prof && !verify_) pprof_->begin(M, start_pos, st_); }
  mh_n_ = 0;
  if (verify_) {  // rollback snapshot: ring slots [pos, pos+M) of every layer + ratio>1 compressor state. History length was recorded by forward_verify.
    for (int l = 0; l < c.n_layers; ++l) w.lringp_h[l] = seq.ring[l].as<bf16>();
    k::ring_gather_layers(w.lringp_d, c.n_layers, c.window, start_pos, M, c.head_dim, w.ring_snap.as<bf16>(), st_);
    for (size_t i = 0; i < v_src2_.size(); ++i) {
      const int l = v_src2_[i];
      const size_t half = w.cstate_snap[i].n / 2;
      CUDA_CHECK(cudaMemcpyAsync(w.cstate_snap[i].p, seq.comp_state_kv[l].p, half, cudaMemcpyDeviceToDevice, st_));
      CUDA_CHECK(cudaMemcpyAsync(w.cstate_snap[i].as<uint8_t>() + half, seq.comp_state_score[l].p, half, cudaMemcpyDeviceToDevice, st_));
    }
  }
  // Tile plan: T sub-chunks (nearly equal sizes — the last sub-chunk never smaller than the tail row count (2,688)), each ≤ w.M
  std::vector<int> tM, tStart;
  struct TileGuard {
    Runtime* r;
    ~TileGuard() {
      r->tile_select(0);
      // Invariant: the Work addresses the decode graphs captured by value must be unchanged — otherwise decode silently goes wrong
      if (!r->tiles_.empty() && (r->w_->h.p != r->work_addr_[0] || r->w_->acc.p != r->work_addr_[1] || (void*)r->w_->route_ids_d != r->work_addr_[2] ||
                                 (void*)r->w_->pos_d != r->work_addr_[3]))
        { fprintf(stderr, "[runtime] FATAL: Work buffers not restored after tiled prefill\n"); abort(); }
    }
  } tile_guard{this};  // restore Work even when leaving through an exception
  if (tiled) {
    HIVE_CHECK(!images || images->empty(), "tiled prefill takes no images");
    HIVE_CHECK(!verify_ && opt_.inject_dir.empty(), "tiled prefill: no verify/inject");  // per-layer dumps keep only the last sub-chunk's (logits dumps are fine)
    // **Allocate** sizes so the last sub-chunk is in tail mode (rows > decoder_tail, ≥ prefill_threshold); merely checking would make requests fail
    //   when max_chunk < ~5.4K. The constructor guarantees w.M ≥ need (otherwise tiles are turned off).
    tile_plan(M, w.M, std::max(opt_.decoder_tail + 1, opt_.prefill_threshold), tM);  // formula lives in tile_plan (shared with H2/H3, same values)
    for (int s = 0, off = 0; s < (int)tM.size(); off += tM[s], ++s) tStart.push_back((int)(start_pos + off));
    for (int n : tM) HIVE_CHECK(n >= 1 && n <= w.M, "tile plan sub-chunk size");
    HIVE_CHECK((int)tM.size() <= tiles && tail_mode_for(tM.back()) && tail_mode_for(M), "tile plan");
  }
  if (!tiled) {
    prologue_rows(ids, M, start_pos);
    // image-token markers (engram blocking, router VL bias) + span embedding merge
    if (images && !images->empty()) {
      for (const ImageInput& im : *images)
        for (size_t p = 0; p < im.types.size(); ++p) { HIVE_CHECK(im.start + (int)p < M, "image span outside chunk"); w.is_image_h[im.start + p] = 1; }
      merge_images(M, *images);
    }
  } else {
    for (size_t s = 0, off = 0; s < tM.size(); off += tM[s], ++s) {
      tile_select((int)s);
      prologue_rows(ids + off, tM[s], tStart[s]);
      CUDA_CHECK(cudaStreamSynchronize(st_));  // emb_h (mapped pinned) is shared by slots — embed_expand must finish reading it before the next slot overwrites it
    }
    tile_select(0);
  }
  engram_ssd_hint(ids, tiled ? nullptr : w.is_image_h, M, nullptr, &seq);  // HIVE_ENGRAM_SSD read-ahead (off = no-op) — overlaps layer-0 compute
  pmark("embed");
  if (pprof_) pprof_->embed(st_);
  // engram hashes (host) — looked up on layer entry
  shared_comp_kv_ = nullptr; shared_index_k_ = nullptr; shared_topk_cols_ = 0; have_candidates_ = false;
  // Decoder tail (prefill): from the last kv-source layer (20) on, compressed KV comes only from that layer's input, so that layer's compressor and indexer keys
  // are built for all rows, then only the last tail rows continue through the remaining layers. With tail = 128×(40−20+1) the last 128 rows, rings and caches are exact (contamination from earlier rows spreads 128 rows per layer).
  const int M_orig = M;
  const int64_t start_orig = start_pos;
  // Tail mode replaces M with the tail row count — with P6 (tail 128) that is < prefill_threshold, which would make the tail layers take the decode policy
  //   (CPU misses, DMA 8) and even post-step promotion. "Is this forward a prefill" is decided by the original chunk size.
  in_prefill_ = (M_orig >= opt_.prefill_threshold || small_fwd) && !verify_;  // Q3 SMALL (off = small_fwd false → base formula)
  small_fwd_ = small_fwd;
  score_prefill_ = !verify_;
  struct ScoreFlag { bool& f; ~ScoreFlag() { f=false; } } score_flag{score_prefill_};
  struct PrefillFlag { bool* f; ~PrefillFlag() { *f = false; } } prefill_flag{&in_prefill_}, small_flag{&small_fwd_};
  // T11 HIVE_LAYER_YIELD: only the outer prefill forward (original rows ≥ prefill_threshold; not verify, batched decode, SMALL, dump/inject) yields at layer boundaries.
  //   Forwards called inside a yield have ly_.depth > 0 and are not eligible (no recursion). Off (not installed) = ly_ok false = base path.
  const bool ly_ok = ly_.on() && ly_.depth == 0 && in_prefill_ && (!small_fwd || ly_small_) && !batch_ && (M_orig >= opt_.prefill_threshold || small_fwd) &&
                     opt_.dump_dir.empty() && opt_.inject_dir.empty();
  if (ly_ok) ly_begin({{&seq, M_orig}});
  const int tail_layer = c.kv_source_layers.empty() ? -1 : *std::max_element(c.kv_source_layers.begin(), c.kv_source_layers.end());
  const int tail_len = opt_.decoder_tail > 0 ? opt_.decoder_tail : 0;
  const bool tail_mode = tail_mode_for(M);
  win_min_pos_ = 0;
  // C1: skip the tail layers only when this chunk is in tail mode (prepare_decoder_tail writes the tail layer's compressed KV for all rows) and it is not verify or an image.
  const bool skip_upper = !upper_needed && tail_mode && !verify_ && !(images && !images->empty());
  bool upper_skipped = false;
  int l_begin = 0;
  if (tiled) {
    // C2 layer-first tiling: for each encoder layer l (front of every sub-chunk → experts once → tail of every sub-chunk). Layer l's ring, compressed cache
    //   and indexer keys are appended in sub-chunk order, so values equal sequential chunk processing (only expert fp32 accumulation order differs). Non-resident experts are transferred once per tile (instead of once per chunk).
    const int T = (int)tM.size();
    for (int l = 0; l < tail_layer; ++l) {
      if (pprof_) pprof_->layer_begin(l, M, false, stats, prefetch_issued_, st_);  // R2 (a tiled layer = all T sub-chunks as one layer)
      prefetch_layer_experts(l, M);  // pre-copy during the fronts of the T sub-chunks (R2: rows = all tile rows)
      std::vector<SubChunk> subs;
      for (int s = 0; s < T; ++s) {
        tile_select(s);
        layer_front(seq, l, tM[s], tStart[s]);
        subs.push_back(SubChunk{tM[s], w.route_ids_h, w.rw_h, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), w.acc.as<float>()});
      }
      tile_select(0);
      if (pprof_) { pprof_->router(st_); pprof_->synced(); }  // each sub-chunk front synchronized by itself (layer_front) — end of router = end of the last front
      moe_experts_multi(model_.layer(l), l, subs, stats);
      pmark("moe");
      if (pprof_) pprof_->moe_done();
      for (int s = 0; s < T; ++s) { tile_select(s); layer_tail(l, tM[s]); }
      if (pprof_) pprof_->layer_end(stats, prefetch_issued_, st_);
      if (ly_ok) layer_yield_point();  // T11 (after the layer tails are issued — the yield point switches the slot to 0 and back)
    }
    // Layer 20 (last kv source): compressed KV and indexer keys of all sub-chunks (all rows) → keep the last sub-chunk in Work and continue with the tail loop below (TileGuard resets to 0 at the end)
    for (int s = 0; s < T - 1; ++s) { tile_select(s); prepare_decoder_tail(seq, tail_layer, tM[s], tStart[s], 0); }
    tile_select(T - 1);
    M = tM[T - 1];
    start_pos = tStart[T - 1];
    l_begin = tail_layer;
  }
  for (int l = l_begin; l < model_.n_loaded_layers(); ++l) {
    bool skip = false;
    if (tail_mode && l == tail_layer) {
      HIVE_CHECK(c.ratio(l) == 1, "decoder tail expects ratio-1 source");
      prepare_decoder_tail(seq, l, M, start_pos, tail_len);
      if (skip_upper) { upper_skipped = true; if (stats) stats->tail_rows = 0; break; }
      const int r0 = M - tail_len;
      M = tail_len;
      start_pos += r0;
      skip = true;
      if (stats) stats->tail_rows = tail_len;
      // P6 Decoder SWA Bounded Replay (model report §3.2.2, deployment recipe): tail rows do not see ring cells before the replay range in the window.
      //   With tail_len == window the ring after replay = the 128 replayed rows = the same state as deployment. Not bit-identical to the reference computation (post-training was fit to this approximation).
      if (opt_.decoder_replay) win_min_pos_ = start_pos;
    }
    layer_forward(seq, l, M, start_pos, stats, skip);
    if (ly_ok && l + 1 < model_.n_loaded_layers()) layer_yield_point();  // T11 (after the layer tail; no yield after the last layer, only the head remains)
  }
  win_min_pos_ = 0;
  if (pprof_) pprof_->layers_done(st_);
  seq.tokens.insert(seq.tokens.end(), ids, ids + M_orig);
  seq.pos += M_orig;
  (void)start_orig;
  int32_t next = -1;
  if (upper_skipped) {
    // Chunk that skipped the tail layers — no logits or MTP hidden (the next chunk produces them). mh_n_ accumulates only in tail layers (target layers 37–39), so it is 0.
    if (logits_out) logits_out->clear();
  } else if (verify_ && model_.has_head() && model_.n_loaded_layers() == c.n_layers) {
    // Speculative verify: logits of every row (to the host) — row i's logits judge draft d_{i+1}
    k::hc_pre(w.h.as<bf16>(), w.pre_mix.as<float>(), M, c.hc, c.dim, w.x.as<bf16>(), st_);
    k::rmsnorm(w.x.as<bf16>(), model_.norm(), c.norm_eps, M, c.dim, w.xn.as<bf16>(), st_);
    k::bf16_to_f32(w.xn.as<bf16>(), M * c.dim, w.xf.as<float>(), st_);
    k::head_logits(w.xf.as<float>(), model_.head(), M, c.vocab, c.dim, w.logits_b.as<float>(), st_);
    k::argmax_rows(w.logits_b.as<float>(), M, c.vocab, w.next_b.as<int32_t>(), st_);
    CUDA_CHECK(cudaMemcpyAsync(w.next_b_h, w.next_b.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st_));
    sampler_cands_dev(M);
    if (verify_logits_out_) {
      verify_logits_out_->resize((size_t)M * c.vocab);
      CUDA_CHECK(cudaMemcpyAsync(verify_logits_out_->data(), w.logits_b.p, (size_t)M * c.vocab * 4, cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    next = w.next_b_h[M - 1];
  } else if (model_.has_head() && model_.n_loaded_layers() == c.n_layers) {
    next = head_last_row(M, logits_out);
  }
  pmark("head");
  // DSpark ring/hidden update (during verify, rollback does it with the accepted rows only)
  if (mtp_on_ && !verify_ && mh_n_ > 0) { mtp_sync_seq(seq, mh_pos0_, mh_n_); pmark("mtp.sync"); }
  CUDA_CHECK(cudaStreamSynchronize(st_));
  preport(M);
  if (pprof_) pprof_->report(tail_mode && !upper_skipped ? tail_len : 0, now_ms(), dma_frac_[kDmaShort], dma_frac_[kDmaStreamShort]);  // R2 (after the sync — all events have completed)
  // Adaptive promotion (decode): after score decay, move hot non-resident experts in on the side stream
  if (M_orig < opt_.prefill_threshold && (opt_.promote_per_token > 0 || opt_.promote_misses > 0)) {
    int n = promote_after_step();
    if (stats) stats->n_promoted += n;
  } else step_miss_.clear();
  if (stats) stats->ms_total += now_ms() - t0;
  ++step_;
  return next;
}

// Chunk head (n rows): tokens/positions → embedding (through mapped pinned emb_h) → h; identity pre_mix; image markers 0. emb_h is shared by slots, so st_ must finish reading it before the next call.
void Runtime::prologue_rows(const int32_t* rows, int n, int64_t start) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  for (int m = 0; m < n; ++m) { w.ids_h[m] = rows[m]; w.pos_h[m] = (int32_t)(start + m); }
  for (int m = 0; m < n; ++m) memcpy(w.emb_h + (size_t)m * c.dim, model_.embed_host() + (size_t)rows[m] * c.dim, (size_t)c.dim * 2);
  k::embed_expand(w.emb_d, w.iota.as<int32_t>(), n, c.dim, c.hc, w.h.as<bf16>(), st_);
  k::identity_pre_mix(w.pre_mix.as<float>(), n, c.hc, st_);
  memset(w.is_image_h, 0, (size_t)n);
}

// Last token: hc_pre(h, pre_mix) → norm → head (synchronous). Returns the argmax
int32_t Runtime::head_last_row(int M, std::vector<float>* logits_out) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  k::hc_pre(w.h.as<bf16>() + (size_t)(M - 1) * c.hc * c.dim, w.pre_mix.as<float>() + (size_t)(M - 1) * c.hc, 1, c.hc, c.dim,
            w.x.as<bf16>(), st_);
  k::rmsnorm(w.x.as<bf16>(), model_.norm(), c.norm_eps, 1, c.dim, w.xn.as<bf16>(), st_);
  k::bf16_to_f32(w.xn.as<bf16>(), c.dim, w.xlast.as<float>(), st_);
  k::head_logits(w.xlast.as<float>(), model_.head(), 1, c.vocab, c.dim, w.logits.as<float>(), st_);
  k::argmax_rows(w.logits.as<float>(), 1, c.vocab, w.next.as<int32_t>(), st_);
  CUDA_CHECK(cudaMemcpyAsync(w.next_h, w.next.p, 4, cudaMemcpyDeviceToHost, st_));
  if (logits_out) {
    logits_out->resize(c.vocab);
    CUDA_CHECK(cudaMemcpyAsync(logits_out->data(), w.logits.p, (size_t)c.vocab * 4, cudaMemcpyDeviceToHost, st_));
  }
  CUDA_CHECK(cudaStreamSynchronize(st_));
  if (!opt_.dump_dir.empty()) dump("logits.f32", w.logits.p, (size_t)c.vocab * 4);
  return *w.next_h;
}

// H3/H2 unit switch: slot (tile_select) + inter-layer globals that differ per sequence (compressed KV and indexer key cache pointers, candidate presence,
//   P6 window floor, MTP capture rows) + candidate-block and MTP capture buffers of the second and later sequences. On leaving, the globals are saved into
//   the unit; on entering they are restored (for single-sequence C2 tiles every sub-chunk is the same sequence, so globals were enough).
void Runtime::unit_select(int u) {
  if (u == unit_cur_) return;
  Work& w = *w_;
  if (unit_cur_ >= 0) {
    UnitCtx& x = units_[(size_t)unit_cur_];
    x.comp_kv = shared_comp_kv_; x.index_k = shared_index_k_; x.have_cand = have_candidates_; x.win_min = win_min_pos_; x.mh_n = mh_n_; x.mh_pos0 = mh_pos0_;
    if (x.mbuf >= 0) { std::swap(w.cand, mbufs_[(size_t)x.mbuf].cand); std::swap(w.mh, mbufs_[(size_t)x.mbuf].mh); }
  }
  tile_select(u >= 0 ? units_[(size_t)u].slot : 0);
  unit_cur_ = u;
  if (u >= 0) {
    const UnitCtx& x = units_[(size_t)u];
    shared_comp_kv_ = x.comp_kv; shared_index_k_ = x.index_k; have_candidates_ = x.have_cand; win_min_pos_ = x.win_min; mh_n_ = x.mh_n; mh_pos0_ = x.mh_pos0;
    if (x.mbuf >= 0) { std::swap(w.cand, mbufs_[(size_t)x.mbuf].cand); std::swap(w.mh, mbufs_[(size_t)x.mbuf].mh); }
  } else {
    shared_comp_kv_ = nullptr; shared_index_k_ = nullptr; have_candidates_ = false; win_min_pos_ = 0; mh_n_ = 0; mh_pos0_ = 0;
  }
}

// H3 (several sequences) and H2 (host tiles) layer-first prefill. Computation follows the C2 tile path (the tiled branch of forward) with the same kernels in
//   the same order, per unit (sub-chunk) — layer l: fronts of all units → experts once (moe_experts_multi) → tails. Only two things change:
//   (1) unit u's tail (l−1) runs right before u's front (l) (a single visit). Tails are row-independent and a front depends only on earlier units of the same
//       sequence and its own tail, so every kernel's input is bit-identical to the original order. Thanks to this reordering a host tile's h is uploaded
//       once and downloaded once per layer.
//   (2) with several sequences, layers 20+ (tail layers) also run layer-first over each sequence's last unit (upper-layer experts once as well).
//   Where values may differ (lossless — same experts, same weights, same activation quantization): a layer's expert batch spans several sequences and more
//   rows, so (a) how per-expert GEMMs group rows, (b) the CPU/DMA split of missed experts (depends on row count and dma_frac — CPU and GPU kernels
//   accumulate fp32 in different orders), (c) chunk boundaries (H2 turns 2 chunks into 1) change only fp32 accumulation order. Attention, compressor,
//   indexer and MTP see the same inputs per sequence and position.
void Runtime::forward_multi(std::vector<PrefillPart>& parts, ForwardStats* stats) {
  deferred_flush();  // HIVE_DECODE_DEFER: a pending deferred batch reads the activation table this forward rewrites
  ElasticScope elastic_scope(*this, 0, true);  // Q1 HIVE_CACHE_ELASTIC: forward_multi always uses the big layout (tiles, staging, mbufs) (off = does nothing)
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int P = (int)parts.size();
  HIVE_CHECK(P >= 1 && !verify_ && opt_.inject_dir.empty() && !batch_, "forward_multi: prefill only");
  for (auto& p : parts) {
    HIVE_CHECK(p.seq && p.ids && p.M >= 1 && p.seq->pos + p.M <= p.seq->cap, "forward_multi part");
    for (auto& q : parts) HIVE_CHECK(&q == &p || q.seq != p.seq, "forward_multi: one part per sequence");
    if (p.seq->snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, p.seq->snapshot_fence->event, 0));
    Seq* sp = p.seq;
    xtrace_step(1, p.M, &sp, 1);
    p.next = -1; p.tail_rows = 0;
    engram_ssd_hint(p.ids, nullptr, p.M, nullptr, p.seq);  // HIVE_ENGRAM_SSD read-ahead (off = no-op; no image markers — only a hint, no effect on values)
  }
  HIVE_CHECK(P == 1 || (int)mbufs_.size() >= P - 1, "forward_multi: batch prefill buffers (HIVE_BATCH_PREFILL)");
  struct BrokenGuard { std::vector<PrefillPart>* ps; int ex; ~BrokenGuard() { if (std::uncaught_exceptions() > ex) for (auto& p : *ps) p.seq->broken = true; } }
      broken_guard{&parts, std::uncaught_exceptions()};
  const int nl = model_.n_loaded_layers();
  const int tail_layer = c.kv_source_layers.empty() ? -1 : *std::max_element(c.kv_source_layers.begin(), c.kv_source_layers.end());
  const int tail_len = opt_.decoder_tail > 0 ? opt_.decoder_tail : 0;
  const int L = tail_layer >= 0 && tail_layer < nl ? tail_layer : nl;  // encoder layers [0, L) — all units
  const int need = std::max(opt_.decoder_tail + 1, opt_.prefill_threshold);
  std::vector<int> Ms;
  for (auto& p : parts) Ms.push_back(p.M);
  std::vector<MultiUnitPlan> plan;
  HIVE_CHECK(plan_units(Ms, w.M, need, prefill_slots(), [&](int m) { return tail_mode_for(m); }, plan), "forward_multi: unit plan exceeds prefill slots");
  const int resident = std::max(1, opt_.prefill_tile);
  const size_t rowb = (size_t)c.hc * c.dim * 2;  // one row of h
  struct MultiGuard {
    Runtime* r; int ex;
    ~MultiGuard() {
      r->unit_select(-1);
      if (r->stager_) {  // when an exception left mid-visit, a staging buffer is still in a slot — give it back and discard it
        for (int k = 0; k < r->stager_->n_host(); ++k) {
          TileSlot& t = r->tiles_.at((size_t)(std::max(1, r->opt_.prefill_tile) - 1 + k));
          if (r->stager_->staged(k) && t.h.p) std::swap(t.h, r->stager_->buf_of(k));
        }
        if (std::uncaught_exceptions() > ex) r->stager_->release_all();
      }
      r->units_.clear();
      if (r->w_->h.p != r->work_addr_[0] || r->w_->acc.p != r->work_addr_[1] || (void*)r->w_->route_ids_d != r->work_addr_[2] ||
          (void*)r->w_->pos_d != r->work_addr_[3])
        { fprintf(stderr, "[runtime] FATAL: Work buffers not restored after multi prefill\n"); abort(); }
    }
  } multi_guard{this, std::uncaught_exceptions()};
  units_.clear();
  unit_cur_ = -1;
  for (const MultiUnitPlan& q : plan) {
    UnitCtx x;
    x.part = q.part; x.slot = q.slot; x.M = q.M; x.off = q.off; x.last = q.last;
    x.start = parts[(size_t)q.part].seq->pos + q.off;
    x.tail = q.last && tail_mode_for(parts[(size_t)q.part].M) && L < nl;
    x.mbuf = q.part > 0 ? q.part - 1 : -1;
    HIVE_CHECK(q.slot < resident || stager_, "forward_multi: host slot without stager");
    units_.push_back(x);
  }
  if (stager_) stager_->begin();
  const double t0 = now_ms();
  store_.flush_promotions();  // E4: forward_multi (prefill) does not take per-layer pacing — same as the prefill head of forward
  store_.commit_pending();
  pmark("start");
  shared_comp_kv_ = nullptr; shared_index_k_ = nullptr; shared_topk_cols_ = 0; have_candidates_ = false; win_min_pos_ = 0; mh_n_ = 0;
  in_prefill_ = true;
  score_prefill_ = true;
  struct ScoreFlag { bool& f; ~ScoreFlag() { f = false; } } score_flag{score_prefill_};
  struct PrefillFlag { bool* f; ~PrefillFlag() { *f = false; } } prefill_flag{&in_prefill_};
  // T11 HIVE_LAYER_YIELD: forward_multi is always a prefill (injection was rejected at the head) — yield at layer boundaries (after the expert stage, where every unit has left via unit_select(-1)).
  const bool ly_ok = ly_.on() && ly_.depth == 0 && opt_.dump_dir.empty();
  if (ly_ok) {
    std::vector<std::pair<Seq*, int>> own;
    for (auto& p : parts) own.push_back({p.seq, p.M});
    ly_begin(std::move(own));
  }
  auto host_k = [&](int u) { return units_[(size_t)u].slot >= resident ? units_[(size_t)u].slot - resident : -1; };
  // One pass: visit the units in us in order. load/store = rows to upload before / download after the visit (meaningful only for host units, 0 = none). body(u) runs with the unit selected.
  std::vector<SubChunk> subs;
  auto pass = [&](const std::vector<int>& us, const std::function<size_t(int)>& load_rows, const std::function<size_t(int)>& store_rows,
                  const std::function<void(int)>& body) {
    std::vector<HostStager::Item> items(us.size());
    for (size_t i = 0; i < us.size(); ++i) {
      const int u = us[i];
      items[i].k = host_k(u);
      items[i].load = items[i].k >= 0 ? load_rows(u) * rowb : 0;
    }
    auto visit = [&](size_t i, DevBuf* stage) {
      const int u = us[i];
      if (stage) std::swap(tiles_.at((size_t)units_[(size_t)u].slot - 1).h, *stage);
      unit_select(u);
      body(u);
      unit_select(-1);
      if (stage) std::swap(tiles_.at((size_t)units_[(size_t)u].slot - 1).h, *stage);
      if (items[i].k >= 0) items[i].store = store_rows(u) * rowb;  // row count after the visit (shrinks at the tail layer)
    };
    if (stager_) stager_->run_pass(items, visit);
    else for (size_t i = 0; i < items.size(); ++i) visit(i, nullptr);
  };
  std::vector<int> sub_u;  // Q3: unit index of subs[i] (push_sub appends both; cleared together with subs)
  auto push_sub = [&](int u) { subs.push_back(SubChunk{units_[(size_t)u].M, w.route_ids_h, w.rw_h, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), w.acc.as<float>()}); sub_u.push_back(u); };
  // Q3 HIVE_PREFILL_MULTI_TAIL_SHORT (spo_.multi_tail_short; off = mshort always false → up_experts is the single moe_experts_multi(subs) call):
  //   R2 TAIL_SHORT (forward's tail layers → moe_decode_experts) carried over to the upper layers (l ≥ L) of forward_multi (H2 host tiles, H3 multiple
  //   sequences). Without it, upper-layer units (P6 tail of 128 rows) also take moe_experts_multi streaming (the HIVE_DMA_FRAC_PREFILL share of misses as
  //   whole-record H2D — 18.8 MB for 1–3 rows per expert). On = for every unit with rows < threshold (in unit order): select its slot as Work → move
  //   activations (xq/xs) to mapped host memory (route_to_host with 0 routes — layer_front already moved the routing) → sync → run the short-prefill path
  //   (moe_decode_experts: resident groups, kDmaShort DMA ≤ HIVE_SHORT_DMA_CAP, the rest on the CPU) **one unit at a time**. The host activation copy
  //   (xq_hd), group tables and staging slots are shared through Work, so the sync between units ensures the previous unit's GPU work is done before the
  //   next unit overwrites them (same rule as between decode layers). Units with rows ≥ threshold (the non-tail last unit) still go through
  //   moe_experts_multi at once. Encoder layers [0, L) are untouched. If every upper-layer unit is short, no pre-copy is issued either (that path does not
  //   use pre-copies — same as R2). Math: same experts, same weights, same activation quantization (act_quant_fp8(xn) of layer_front) — the only possible
  //   difference is fp32 accumulation order: (a) GPU expert kernels (decode mx_grouped/F2/F3 chain vs prefill grouped GEMM), (b) which misses go to the CPU,
  //   (c) acc accumulation order, (d) with several sequences expert batches split per sequence (otherwise sequences are batched into one GEMM). Attention,
  //   compressor, indexer and MTP inputs are the same. Row markers of cache observation (observe) are those of the short path (row numbers).
  auto mshort = [&](int rows) { return sp::multi_short(spo_.multi_tail_short, opt_.cpu_for_misses, rows, opt_.prefill_threshold); };
  auto up_experts = [&](int l) {
    size_t n_short = 0;
    for (int u : sub_u) n_short += mshort(units_[(size_t)u].M) ? 1 : 0;
    if (n_short == 0) { moe_experts_multi(model_.layer(l), l, subs, stats); return; }  // off = always here (the plain call)
    // F1 prefill isolation — usage observation (observe: score_, use_count_, D4 policy) is done once here with **the same formula and order** as the single
    //   moe_experts_multi(subs) call. If each unit's moe_decode_experts observed instead, (1) the weight would be 1/(unit rows) instead of 1/(sum of the
    //   layer's sub-chunk rows) — with H3 multiple sequences (P units) upper-layer observation becomes P times heavier (breaking the prompt weighting of
    //   HIVE_PHASE_SCORE), so after prefill warm_cache and step promotion (promote — score_/prio_ order) would favour the prompt's upper-layer experts over
    //   decode-resident experts; and (2) the row argument i/k would resolve against the D4 policy row-owner table (overwritten per sequence by xtrace_step at
    //   the head of forward_multi = only the **last** sequence), so every sequence's upper-layer use would count as request evidence of the last sequence
    //   (the multi call uses -1 = unowned with two or more sub-chunks). Expert tracing (xtrace) keeps the same 'H' records (so tools/cache_replay replays
    //   identically). Computation (which experts, devices and order) is unchanged — observation has no effect on results (only on cache residency).
    {
      const LayerWeights& Lw = model_.layer(l);
      const int kk = Lw.n_act ? Lw.n_act : c.n_act, EE = Lw.n_routed ? Lw.n_routed : c.n_routed;
      sp::observe_like_multi(subs, kk, opt_.phase_score, [&](int e, float wt, int row) { store_.observe(l, e, wt, row); },
                             [&](const SubChunk& sc) { xtrace_hist(l, sc.M, kk, EE, sc.route_ids_h); });  // hive/short_prefill.h (same formula and order as moe_experts_multi's observation)
    }
    struct ObsOff { DecodeOverlap& D; ~ObsOff() { D.obs_off = false; } } obs_off{*dov_};
    dov_->obs_off = true;  // the two calls below skip observation and tracing (done above)
    if (n_short < subs.size()) {
      std::vector<SubChunk> rest;
      for (size_t i = 0; i < subs.size(); ++i) if (!mshort(units_[(size_t)sub_u[i]].M)) rest.push_back(subs[i]);
      moe_experts_multi(model_.layer(l), l, rest, stats);
    }
    for (int u : sub_u) {
      const int Mu = units_[(size_t)u].M;
      if (!mshort(Mu)) continue;
      unit_select(u);  // xq/xs, acc, routing tables = this unit's slot (h is not used — may be empty for a host unit)
      k::route_to_host(w.ids.as<int32_t>(), w.rw.as<float>(), w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), 0, Mu * c.dim, w.route_ids_d, w.rw_d, w.xq_hd, w.xs_hd, st_);
      CUDA_CHECK(cudaStreamSynchronize(st_));  // activation copy arrived + previous unit's GPU expert work done (before overwriting shared tables and slots)
      moe_decode_experts(model_.layer(l), l, Mu, stats);
      unit_select(-1);
    }
  };
  auto tail_prev = [&](int l, int M) {
    if (M <= 8) CUDA_CHECK(cudaStreamWaitEvent(st_, hc_join_[1], 0));  // fused path ffn sinkhorn branch (same join as layer_forward)
    layer_tail(l, M);
  };
  std::vector<int> all(units_.size());
  for (size_t u = 0; u < all.size(); ++u) all[u] = (int)u;
  auto rows_of = [&](int u) -> size_t { return (size_t)units_[(size_t)u].M; };
  auto none = [](int) -> size_t { return 0; };
  // ---- Encoder layers [0, L): all units
  for (int l = 0; l < L; ++l) {
    prefetch_layer_experts(l);  // pre-copy during the fronts (same point as C2 — tail l−1 follows experts l−1 in st_ order, unrelated to the pre-copy)
    subs.clear(); sub_u.clear();
    pass(all, l == 0 ? std::function<size_t(int)>(none) : std::function<size_t(int)>(rows_of), rows_of, [&](int u) {
      UnitCtx& x = units_[(size_t)u];
      PrefillPart& p = parts[(size_t)x.part];
      if (l == 0) { prologue_rows(p.ids + x.off, x.M, x.start); CUDA_CHECK(cudaStreamSynchronize(st_)); }  // emb_h is shared by slots
      else tail_prev(l - 1, x.M);
      layer_front(*p.seq, l, x.M, x.start);
      push_sub(u);
    });
    moe_experts_multi(model_.layer(l), l, subs, stats);
    pmark("moe");
    if (ly_ok) layer_yield_point();  // T11 (tail l is done by the next pass's visit — acc, post_f, comb_f and h are the inter-layer state to park)
  }
  // ---- Tail layer L (last kv source): earlier units only produce compressed KV and indexer keys (all rows); the last unit shrinks to the tail rows (ends here with C1) or continues with all rows
  std::vector<int> up;
  if (L < nl) {
    bool any_up = false;
    bool up_long = false;  // Q3: whether any unit heading to the upper layers is not on the short path (would use pre-copies) — off = same as any_up
    for (const UnitCtx& x : units_) if (x.last && (!x.tail || parts[(size_t)x.part].upper_needed)) { any_up = true; if (!mshort(x.tail ? tail_len : x.M)) up_long = true; }
    if (any_up && up_long) prefetch_layer_experts(L);
    subs.clear(); sub_u.clear();
    pass(all, L == 0 ? std::function<size_t(int)>(none) : std::function<size_t(int)>(rows_of),
         [&](int u) -> size_t { return units_[(size_t)u].up ? (size_t)units_[(size_t)u].M : 0; }, [&](int u) {
      UnitCtx& x = units_[(size_t)u];
      PrefillPart& p = parts[(size_t)x.part];
      if (L == 0) { prologue_rows(p.ids + x.off, x.M, x.start); CUDA_CHECK(cudaStreamSynchronize(st_)); }  // model whose tail layer is layer 0 (no encoder layers)
      else tail_prev(L - 1, x.M);
      if (!x.last) { prepare_decoder_tail(*p.seq, L, x.M, x.start, 0); return; }
      if (x.tail) {
        HIVE_CHECK(c.ratio(L) == 1, "decoder tail expects ratio-1 source");
        prepare_decoder_tail(*p.seq, L, x.M, x.start, tail_len);
        if (!p.upper_needed) return;  // C1: skip the tail layers (same condition as forward's skip_upper — tail and !upper_needed, no images)
        const int r0 = x.M - tail_len;
        x.M = tail_len;
        x.start += r0;
        p.tail_rows = tail_len;
        if (opt_.decoder_replay) win_min_pos_ = x.start;  // P6 (same as forward)
        HIVE_CHECK(x.mbuf < 0 || !mbufs_[(size_t)x.mbuf].cand.p || (size_t)x.M * c.cand_topk_blocks * 4 <= mbufs_[(size_t)x.mbuf].cand.n, "batch prefill cand rows");
        x.up = true;
        layer_front(*p.seq, L, x.M, x.start, true);
      } else {
        HIVE_CHECK(x.mbuf < 0 || !mbufs_[(size_t)x.mbuf].cand.p || (size_t)x.M * c.cand_topk_blocks * 4 <= mbufs_[(size_t)x.mbuf].cand.n, "batch prefill cand rows");
        x.up = true;
        layer_front(*p.seq, L, x.M, x.start, false);
      }
      push_sub(u);
    });
    for (size_t u = 0; u < units_.size(); ++u) if (units_[u].up) up.push_back((int)u);
    if (!subs.empty()) { up_experts(L); pmark("moe"); }  // Q3 (off = plain moe_experts_multi(L, subs))
    if (ly_ok && !subs.empty()) layer_yield_point();  // T11
    // ---- upper layers (L, nl): the last units that continue
    for (int l = L + 1; l < nl; ++l) {
      if (up.empty()) break;
      bool long_l = false;  // Q3 (off = always true)
      for (int u : up) if (!mshort(units_[(size_t)u].M)) long_l = true;
      if (long_l) prefetch_layer_experts(l);
      subs.clear(); sub_u.clear();
      pass(up, rows_of, rows_of, [&](int u) {
        UnitCtx& x = units_[(size_t)u];
        tail_prev(l - 1, x.M);
        layer_front(*parts[(size_t)x.part].seq, l, x.M, x.start);
        push_sub(u);
      });
      up_experts(l);  // Q3 (off = plain moe_experts_multi(l, subs))
      pmark("moe");
      if (ly_ok) layer_yield_point();  // T11 (upper layers — 128 tail rows make layers short: until the period is reached it only reads the clock)
    }
  } else {
    for (size_t u = 0; u < units_.size(); ++u) if (units_[u].last) { units_[u].up = true; up.push_back((int)u); }
  }
// ---- Last layer tail → head (logits) → MTP ring/hidden (each sequence from its own capture)
  const bool head = model_.has_head() && nl == c.n_layers;
  pass(up, rows_of, none, [&](int u) {
    UnitCtx& x = units_[(size_t)u];
    PrefillPart& p = parts[(size_t)x.part];
    tail_prev(nl - 1, x.M);
    if (head) p.next = head_last_row(x.M, p.logits_out);
    if (mtp_on_ && mh_n_ > 0) mtp_sync_seq(*p.seq, mh_pos0_, mh_n_);
  });
  pmark("head");
  for (const UnitCtx& x : units_) if (x.last && !x.up && parts[(size_t)x.part].logits_out) parts[(size_t)x.part].logits_out->clear();  // C1: no logits
  unit_select(-1);
  for (auto& p : parts) { p.seq->tokens.insert(p.seq->tokens.end(), p.ids, p.ids + p.M); p.seq->pos += p.M; }
  CUDA_CHECK(cudaStreamSynchronize(st_));
  int total = 0;
  for (auto& p : parts) total += p.M;
  preport(total);
  step_miss_.clear();  // prefill (same as forward's M ≥ threshold branch — no post-step promotion)
  if (stats) stats->ms_total += now_ms() - t0;
  ++step_;
}

// ---- T11 HIVE_LAYER_YIELD — layer-boundary yield inside long prefill forwards ---------------------------------------------------------------------------------
// What (see the top of hive/layer_yield.h and T11 in hived.cpp): when an outer prefill forward (forward's tiled/single layer loop, forward_multi's encoder,
//   tail and upper-layer loops) finishes one layer's expert stage, the period has elapsed and hived has work (in-flight decode, short requests to admit), park
//   the outer prefill's inter-layer state, run hived's yield body, then restore. Evidence (measured): short-request TTFT 15.5 s during a 100K prefill; an
//   in-flight decode stalled 14 s during a long prefill; in service, 5 stalls of up to 4.9 s within 70 minutes (27K and 7.5K-row prefills). The existing yields
//   (T3 HIVE_PREFILL_YIELD, decode_between) exist only between forwards, so a prompt that is a single forward (98,304 rows ≈ 14 s) had no boundary at all.
// Parked = "what must live between layers" (the list defined in the comment above tile_select — exactly the state C2 tiles keep across other sub-chunks'
//   fronts), taken from slot 0 (Work): the first R rows of device h, pre_mix, idx, xq/xs, acc, pre_f/post_f/comb_f + the first R rows of cand (layer-20
//   candidate blocks — read by upper layers) + mh (MTP target-layer capture — whole, small) → pinned stash (D2H, st_); the first R rows of the mapped pinned
//   tables route_ids/rw/pos/ids/is_image/visible/gpos → host copies (memcpy); globals shared_comp_kv_/shared_index_k_/shared_topk_cols_/have_candidates_/
//   win_min_pos_/mh_n_/mh_pos0_/seq_hashes_/shared_topk_rows_, in_prefill_/score_prefill_/small_fwd_; the pre-copy prediction table last_rows_ (if an inner
//   short prefill overwrote it, the outer upper-layer pre-copies would follow that request — same values, only cost); profiling (prof_, pprof_ — the inner
//   forward writes its own and discards it).
//   R = hive/layer_yield.h park_rows(want, floor, cap): the row bound of inner work in Work — the decode floor (same formula as the elastic small layout S;
//   typically 128) or, for admitting a short prefill, its row count (hived only admits ≤ R). Estimate: ≈ 70 KB per row (h 40 KB + acc 20 KB + idx 2.5 KB +
//   xq 5 KB …) → decode only 128 rows ≈ 9 MB, 1023 rows ≈ 71 MB. The pinned stash is sized on first use (only grows) — no VRAM is used (CACHE_FIT keeps the
//   headroom tight; a reserve of 1800 MiB caused an image OOM). Pinned allocation failure = skip this yield only (fallback — the outer prefill runs to the end as usual).
// Not parked: in-layer scratch (rewritten by that layer's front — a property the C2 tile path already relies on), host tile h (pinned host; HostStager
//   buffers are empty between passes), resident tile slots (tiles_ — inner work never uses tiles: rows ≤ R ≤ max_chunk means forward's non-tiled path),
//   the staging ring (stage_next_; inner work follows the stage_freed_ event rule too), DMA share / split EMAs (performance state).
// Order (layer_yield.h yield_point): sync the CPU pool, st_, side_, hc_side_ (the outer expert stage's GPU and CPU work must be done before mapped tables and
//   slots are handed over) → slot 0 (tile_select — in upper layers the last sub-chunk may be swapped in; restore it: invariant on the Work addresses captured
//   by decode CUDA graphs) → park → elastic: depth 0, small layout (addresses captured by decode graphs; an inner forward with rows > S switches to the big
//   layout through its own ElasticScope — the first R rows of the big layout are parked), lending blocked (ly_.depth in lend_if_idle), promotion paused
//   (prefill_pending_ — same meaning as the M7 HIVE_PREFILL_PAUSE_PROMOTE marker: keep decode promotions from evicting residents the outer prefill needs in
//   the next layer) → run → sync → restore elastic state → restore (H2D st_, memcpy) → globals → original slot → clear pre-copy markers (pf_ — the outer next
//   layer issues them again) → rewrite the xtrace head and D4 row owners (so analyzers attribute the following 'H' records to the outer prefill; row owners for cache policy seq).
// Math: the outer prefill's kernels, inputs and order are the same (the same forward is merely paused between layers). What may differ = the cache resident set
//   (inner decode demand DMA and CPU misses do not change residency, but an inner short prefill's observation and promotions after the pause ends can) → the
//   CPU/DMA split of the outer next layer's misses → fp32 accumulation order — the same property as decode_between between chunks (token comparison is a GPU validation command).
struct Runtime::LyPark {
  uint8_t* stash = nullptr;  // pinned (D2H/H2D) — device rows
  size_t cap = 0;
  std::vector<uint8_t> host;  // copies of the mapped pinned tables
  std::vector<ly::RowBuf> dev, hst;
  bool warned = false;
  const uint8_t* comp_kv = nullptr; const uint8_t* index_k = nullptr;
  int topk_cols = 0, mh_n = 0, tile_cur = 0, el_depth = 0;
  bool have_cand = false, in_prefill = false, score_prefill = false, small_fwd = false, prefill_pending = false, el_big = false;
  int64_t win_min = 0, mh_pos0 = 0;
  std::vector<int64_t> hashes;
  std::vector<int> topk_rows;
  std::vector<std::vector<int>> last_rows;
  Prof prof;
  std::shared_ptr<PrefillProf> pprof;
  ~LyPark() { if (stash) cudaFreeHost(stash); }
  bool reserve(size_t n) {
    if (n <= cap) return true;
    if (stash) { cudaFreeHost(stash); stash = nullptr; cap = 0; }
    void* p = nullptr;
    if (cudaHostAlloc(&p, n, cudaHostAllocDefault) != cudaSuccess || !p) { (void)cudaGetLastError(); return false; }
    stash = static_cast<uint8_t*>(p);
    cap = n;
    return true;
  }
};

void Runtime::ly_begin(std::vector<std::pair<Seq*, int>> owners) {
  if (ly_.depth > 0) return;
  ly_.last = now_ms();
  ly_owner_ = std::move(owners);
}

void Runtime::layer_yield_point() {
  if (!ly_.on() || ly_.depth > 0 || capturing_) return;
  if (!lypark_) lypark_ = std::make_shared<LyPark>();
  LyPark& P = *lypark_;
  Work& w = *w_;
  const Config& c = model_.cfg();
  auto drain = [&] {
    store_.wait_jobs();
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaStreamSynchronize(side_));
    CUDA_CHECK(cudaStreamSynchronize(hc_side_));
  };
  auto park = [&](int R) -> bool {
    drain();
    P.tile_cur = tile_cur_;
    tile_select(0);
    // slot 0's inter-layer state (same set as tile_xchg's swap list + cand and mh — tools/test_layer_yield_cpu.py compares the two lists textually)
    P.dev.clear(); P.hst.clear();
    const size_t hc = (size_t)c.hc, dim = (size_t)c.dim, kk = (size_t)c.n_act;
    auto dv = [&](DevBuf& b, size_t rb) { if (b.p && b.n >= rb && rb) P.dev.push_back({b.p, rb, b.n / rb}); };
    dv(w.h, hc * dim * 2); dv(w.pre_mix, hc * 4); dv(w.idx, (size_t)(c.window + c.index_topk) * 4); dv(w.xq, dim); dv(w.xs, dim / 32); dv(w.acc, dim * 4);
    dv(w.pre_f, hc * 4); dv(w.post_f, hc * 4); dv(w.comb_f, hc * hc * 4);
    if (c.cand_source_layer >= 0) dv(w.cand, (size_t)c.cand_topk_blocks * 4);
    if (w.mh.p && w.mh.n) P.dev.push_back({w.mh.p, w.mh.n, 1});  // row 1 = whole (mh_rows rows — all of them when R ≥ 1)
    auto hb = [&](void* p, size_t rb) { if (p) P.hst.push_back({p, rb, (size_t)w.M}); };
    hb(w.route_ids_h, kk * 4); hb(w.rw_h, kk * 4); hb(w.pos_h, 4); hb(w.ids_h, 4); hb(w.is_image_h, 1); hb(w.visible_h, 4); hb(w.gpos_h, 4);
    if (!P.reserve(ly::park_bytes(P.dev, R))) {
      if (!P.warned) fprintf(stderr, "[runtime] ⚠️layer yield: pinned park buffer (%.0f MiB) unavailable — yield skipped (prefill continues)\n",
                             ly::park_bytes(P.dev, R) / 1048576.0);
      P.warned = true;
      tile_select(P.tile_cur);
      return false;
    }
    ly::park(P.dev, R, P.stash, [&](void* d, const void* src, size_t n) { CUDA_CHECK(cudaMemcpyAsync(d, src, n, cudaMemcpyDeviceToHost, st_)); });
    P.host.resize(ly::park_bytes(P.hst, R));
    ly::park(P.hst, R, P.host.data(), [](void* d, const void* src, size_t n) { memcpy(d, src, n); });
    P.comp_kv = shared_comp_kv_; P.index_k = shared_index_k_; P.topk_cols = shared_topk_cols_; P.have_cand = have_candidates_; P.win_min = win_min_pos_;
    P.mh_n = mh_n_; P.mh_pos0 = mh_pos0_; P.hashes = seq_hashes_; P.topk_rows = shared_topk_rows_; P.last_rows = last_rows_;
    P.in_prefill = in_prefill_; P.score_prefill = score_prefill_; P.small_fwd = small_fwd_; P.prefill_pending = prefill_pending_;
    std::swap(P.prof, prof_);  // the inner forward starts from an empty table (the outer one is kept)
    P.pprof = std::move(pprof_);
    P.el_depth = w.el.depth; P.el_big = w.el.big;
    if (w.el.on) { w.el.depth = 0; ElasticScope::view(*this, false); }  // the inner forward picks its own layout (decode = small layout = the addresses graphs captured)
    in_prefill_ = false; score_prefill_ = false; small_fwd_ = false;
    prefill_pending_ = true;  // pauses promotion (with HIVE_PREFILL_PAUSE_PROMOTE), blocks elastic lending
    CUDA_CHECK(cudaStreamSynchronize(st_));  // D2H done — before inner work (including other streams) overwrites those rows
    return true;
  };
  auto unpark = [&](int R) {
    drain();
    if (w.el.on) { ElasticScope::view(*this, P.el_big); w.el.depth = P.el_depth; }  // back to the layout at park time (restore address = parked address)
    ly::unpark(P.dev, R, P.stash, [&](void* d, const void* src, size_t n) { CUDA_CHECK(cudaMemcpyAsync(d, src, n, cudaMemcpyHostToDevice, st_)); });
    ly::unpark(P.hst, R, P.host.data(), [](void* d, const void* src, size_t n) { memcpy(d, src, n); });
    shared_comp_kv_ = P.comp_kv; shared_index_k_ = P.index_k; shared_topk_cols_ = P.topk_cols; have_candidates_ = P.have_cand; win_min_pos_ = P.win_min;
    mh_n_ = P.mh_n; mh_pos0_ = P.mh_pos0; seq_hashes_ = std::move(P.hashes); shared_topk_rows_ = std::move(P.topk_rows); last_rows_ = std::move(P.last_rows);
    in_prefill_ = P.in_prefill; score_prefill_ = P.score_prefill; small_fwd_ = P.small_fwd; prefill_pending_ = P.prefill_pending;
    for (auto e : prof_.ev) cudaEventDestroy(e);  // table left by the inner forward (cut off before its report)
    prof_ = std::move(P.prof); P.prof = Prof{};
    pprof_ = std::move(P.pprof);
    pf_.clear(); pf_l_ = -1;  // the outer pre-copies were consumed (this point is after the expert stage) — drop the inner work's pre-copy markers too (the next layer head issues them again)
    tile_select(P.tile_cur);
    for (auto& [sq, m] : ly_owner_) { Seq* sp = sq; xtrace_step(1, m, &sp, 1); }
    CUDA_CHECK(cudaStreamSynchronize(st_));
  };
  ly::yield_point(ly_, park, unpark, [] { return now_ms(); });
}

// E4 HIVE_DECODE_COPY_PRIO — decode PCIe ordering: demand DMA first, promotions in the gaps after it.
//   Evidence (nsys, service build CLI, warm cache, last 128 decode steps, 32.3 ms per step on average): 8.9 ms/step of H2D during decode (238 MB — DMA-share
//   misses + promotions). Without pacing, promote_after_step issues all promotions (--promote 8 = 8 records) on promo_ at once at the end of the step, and
//   the demand DMA of the next step's first layers (side_ — that layer's expert kernels wait for it) shares the same PCIe link with those copies. Streams are
//   unordered, so demand copies can queue behind promotions.
//   This variant (only when on): ExpertStore **decides** promotions at the same time with the same rules (what, which victim slot, pending marker, stats) but
//   queues the H2D as slices (12 per record = same bytes in the same order as copy_rec_async); here, per layer, **after that layer's demand copies are issued
//   on side_**, promo_ is chained behind them (promo_gate_) and only a budget is issued. Budget = ⌈remaining bytes / remaining layers⌉ — so everything goes out
//   by the step's last layer (the last layer's budget = everything left). The amount of earlier-layer promotion slices that can sit ahead of the next layer's
//   demand copies is bounded by one layer's budget (+ one slice). Layer numbers ≥ the layer count (DSpark draft layers) issue everything left.
//   On the step-graph (G1) path the dispatcher calls the same function on every publish (after that layer's DMA issue and signal — decode_dispatch.h after_dma).
//   Residency time: a paced batch, once all slices are issued, is committed at the first commit_pending (step head) after **waiting** for those copies
//   (ExpertStore::commit_pending). A decision at the end of step t goes out during step t+1 (or the next promote, prefill head or commit_all issues what is
//   left) and becomes resident at the head of step t+2 — fixed by call order, hence deterministic. Unpaced residency is a query and depends on copy speed:
//   150 MB (8 records) ≈ 6 ms is longer than the gap between steps, so usually the head of t+2 (same), a small batch finishing within the gap gives t+1 (pacing
//   is one step later), and a congested PCIe that cannot finish within one step gives t+3 or later (pacing is earlier). With equal residency times, later
//   promotion and victim decisions are equal as well (decision inputs = scores, slot table, pending).
//   Math: expert computation does not change (same bytes in the same slot — with equal residency times, the same device and kernel per row). Off = this function returns immediately.
void Runtime::promo_pump(int l, int nL) {
  if (!copy_prio_ || !store_.promo_backlog()) return;
  const size_t left = store_.promo_backlog_bytes();
  const size_t layers_left = (size_t)std::max(1, nL - l);
  CUDA_CHECK(cudaEventRecord(promo_gate_, side_));  // after this layer's demand copies (just issued on side_)
  CUDA_CHECK(cudaStreamWaitEvent(promo_, promo_gate_, 0));
  store_.issue_promotions((left + layers_left - 1) / layers_left);
}

// Demand-copy instrumentation (E4 — HIVE_PROFILE sample steps, HIVE_TRACE_CACHE): for layer l, [a = before the side_ demand copy, b = after, s = right before st_ waits for that copy].
//   span = Σ(b − a) = time demand copies took on side_ (grows when sharing the link with promotions); wait = Σ max(0, b − s) = time expert kernels waited for
//   copies (per-layer path only — in the step graph the wait kernel is inside the graph, so there is no s: "wait -"). promo = promotion H2D bytes actually
//   issued since the same point of the previous step (off = the batch issued at the end of a step shows on the next step's line; on = slices paced in this
//   step + slices left over from the end of the previous step — both are "promotion copies that overlapped this step").
void Runtime::cw_begin(bool on) {
  cw_on_ = false;
  cw_line_.clear();
  if (!on) return;
  const size_t n = (size_t)model_.cfg().n_layers + 8;  // trunk + draft layers (layer numbers beyond this are not measured)
  if (cw_ev_.empty()) {
    cw_ev_.resize(3 * n, nullptr);
    for (auto& e : cw_ev_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDefault));
  }
  cw_used_.assign(n, 0);
  cw_on_ = true;
}
void Runtime::cw_collect() {
  const uint64_t pb = store_.promo_h2d_bytes();
  const double promo_mb = (double)(pb - cw_promo_last_) / 1048576.0;
  cw_promo_last_ = pb;
  if (!cw_on_) return;
  cw_on_ = false;
  double span = 0, wait = 0;
  int n_span = 0, n_wait = 0;
  for (size_t l = 0; l < cw_used_.size(); ++l) {
    if (!(cw_used_[l] & 1)) continue;
    cudaEvent_t a = cw_ev_[3 * l], b = cw_ev_[3 * l + 1], s = cw_ev_[3 * l + 2];
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    span += ms; ++n_span;
    if (cw_used_[l] & 2) { CUDA_CHECK(cudaEventElapsedTime(&ms, s, b)); wait += std::max(0.f, ms); ++n_wait; }
  }
  char buf[160];
  if (n_wait > 0) snprintf(buf, sizeof buf, " · dma.span %.2f dma.wait %.2f ms (%d layers) · promo %.1f MiB%s", span, wait, n_span, promo_mb, copy_prio_ ? " paced" : "");
  else snprintf(buf, sizeof buf, " · dma.span %.2f dma.wait - ms (%d layers) · promo %.1f MiB%s", span, n_span, promo_mb, copy_prio_ ? " paced" : "");
  cw_line_ = buf;
}

// End-of-step promotion on the decode path: with promote_misses > 0, this step's misses (by decayed score) → LRU slots; otherwise score-based promotion
// ⚠️Precondition: called after an st_ sync (forward and forward_batch do so) — kernels reading a victim slot's old record must have finished before promo_ copies overwrite it (streams are unordered).
int Runtime::promote_after_step() {
  store_.decay_scores(opt_.score_decay);
  int n = 0;
  // REUSE_STAGE victim-slot marker for the dedicated D2D stream: recorded on st_ here, where the precondition above (after the st_ sync) holds (unused when the option is off)
  cudaEvent_t victim_ready = nullptr;
  if (store_.reuse_staging() && promo_evt_) { CUDA_CHECK(cudaEventRecord(promo_evt_, st_)); victim_ready = promo_evt_; }
  // M7 (opt-in HIVE_PREFILL_PAUSE_PROMOTE): while hived reports a multi-chunk prefill in progress (set_prefill_pending), steps between chunks issue no
  //   promotions — the upcoming chunk streams every expert per layer anyway, so promotions in between would share PCIe and churn the LRU. Score decay (above)
  //   and clearing the miss list (below) still happen, so the state after the prefill = the unpaused path minus "issued promotions" (promotions do not change scores).
  if (prefill_pause_promote_ && prefill_pending_) {
  } else if (opt_.promote_misses > 0) {
    if (opt_.mtp_cache) step_miss_.insert(step_miss_.end(), draft_miss_.begin(), draft_miss_.end());
    draft_miss_.clear();
    std::sort(step_miss_.begin(), step_miss_.end());
    step_miss_.erase(std::unique(step_miss_.begin(), step_miss_.end()), step_miss_.end());
    std::sort(step_miss_.begin(), step_miss_.end(), [&](int a, int b) { return store_.score_of(a) > store_.score_of(b); });
    int cap = opt_.promote_misses;
    if (opt_.promote_miss_ratio > 0.f) cap = std::min(cap, (int)std::ceil((double)step_miss_.size() * opt_.promote_miss_ratio));
    if (opt_.promote_byte_budget) cap = std::min<size_t>(cap, opt_.promote_byte_budget / store_.layout().total);
    n = cap > 0 ? store_.promote_keys(step_miss_, cap, promo_, opt_.promote_misses, opt_.promote_score_victims, victim_ready) : 0;
  } else if (opt_.promote_per_token > 0 || warm_left_ > 0) {
    int cap = opt_.promote_per_token;
    float min_score = opt_.promote_min_score;
    // Deferred warm (warm_defer): while a quota is left, this step's promotion runs with the warm's cap and threshold (score >= 1, as warm_cache) — the
    //   same promote() call, so the E4 pacing of the step's promotions is unchanged (a second call would flush the first batch unpaced).
    if (warm_left_ > 0) { cap = std::max(cap, std::min(warm_left_, warm_step_)); min_score = std::min(min_score, 1.f); }
    if (opt_.promote_byte_budget) cap = std::min<size_t>(cap, opt_.promote_byte_budget / store_.layout().total);
    // promote() issues nothing while 2 x its cap copies are still in flight — such a step keeps the quota (it is not "nothing left to warm")
    const bool backlog = store_.n_pending() >= 2 * std::min(cap, std::max(1, store_.n_slots() / 4));
    n = store_.promote(cap, min_score, promo_, victim_ready);
    if (warm_left_ > 0 && cap > 0 && !backlog) {
      ++warm_steps_;
      if (n > 0) { warm_left_ -= n; warm_done_ += n; }
      else warm_left_ = 0;  // no non-resident expert over the threshold beats its victim: the warm is complete (warm_cache stops on the same condition)
      if (warm_left_ <= 0) { warm_left_ = 0; fprintf(stderr, "[runtime] deferred warm: %d experts over %d steps\n", warm_done_, warm_steps_); }
    }
  }
  step_miss_.clear();
  draft_miss_.clear();
  return n;
}

// Prefill expert pre-copy: while layer l's front (attention, indexer, hc, router) computes, PCIe is idle (estimated 30–90 ms per layer). Stream the
//   non-resident experts that received the most rows at this layer in the previous prefill into the staging ring ahead of time (at 16K+ rows nearly every
//   expert is used — traces: 0–3 unused experts per layer). moe_experts_multi removes pre-copied experts from the CPU share and consumes them **first**
//   (ring reuse order = copy order). Unused pre-copies just free their slots.
void Runtime::prefetch_layer_experts(int l, int rows_now) {
  if(prefetch_timing_pending_) {
    const auto status=cudaEventQuery(prefetch_end_);
    if(status==cudaSuccess) {
      float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,prefetch_begin_,prefetch_end_));prefetch_copy_ms_+=ms;prefetch_timing_pending_=false;
    } else if(status!=cudaErrorNotReady) CUDA_CHECK(status);
  }
  pf_.clear();
  pf_l_ = -1;
  static const int depth_env = getenv("HIVE_PREFETCH") ? atoi(getenv("HIVE_PREFETCH")) : 0;
  int depth = std::min(store_.staging_slots(), std::max(0, depth_env));
  if (const char* bytes = getenv("HIVE_PREFETCH_BYTES")) {
    const long long budget = std::max(0ll, atoll(bytes));
    depth = (int)std::min<size_t>(depth, (size_t)budget / store_.layout().total);
  }
  if (depth <= 0 || l >= (int)last_rows_.size() || last_rows_[l].empty()) return;
  const std::vector<int>& lr = last_rows_[l];
  // Experts with few rows are handled cheaply by the CPU share (pre-copying pushes them onto PCIe, and a wrong prediction blocks real copies) → only experts with enough rows
  static const int min_rows = getenv("HIVE_PREFETCH_MIN_ROWS") ? atoi(getenv("HIVE_PREFETCH_MIN_ROWS")) : 32;
  // R2 HIVE_PREFETCH_SCALE (spo_.prefetch_scale): lr holds the **previous chunk's** row counts. A 1K chunk after a 16K chunk (a short next turn in a
  //   conversation) has 1/16 the rows per expert, but the plain rule (lr ≥ min_rows) judges by the previous counts and pre-copies up to depth records per layer
  //   (e.g. 96 × ~0.67 ms) of experts that now get only two or three rows (cheap on the CPU).
  //   On = judge by predicted rows = lr × (rows now / previous rows) (previous rows = Σ lr / top-k — top-k per row). The ranking (rows descending) scales
  //   proportionally and stays the same — only the number of candidates over the threshold changes (equal or larger chunks give the same or more candidates →
  //   the top depth equal the plain set). Pre-copies do not affect computation order or results (only which misses go to the GPU — CPU/GPU fp32 accumulation
  //   order). rows_now = 0 (unknown) uses the plain rule.
  int64_t pf_num = 1, pf_den = 1;  // off = 1/1 → sp::prefetch_pass is the plain lr ≥ max(1, min_rows) (hive/short_prefill.h)
  sp::prefetch_ratio(spo_.prefetch_scale, rows_now, model_.layer(l).n_act ? model_.layer(l).n_act : model_.cfg().n_act, lr, pf_num, pf_den);
  std::vector<std::pair<int, int>> cand;  // (-row count, e)
  for (int e = 0; e < (int)lr.size(); ++e) if (sp::prefetch_pass(lr[e], pf_num, pf_den, min_rows) && store_.slot_of(l, e) < 0) cand.push_back({-lr[e], e});
  if (cand.empty()) return;
  const size_t n = std::min<size_t>((size_t)depth, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + n, cand.end());
  const bool measure=!prefetch_timing_pending_;
  if(measure) CUDA_CHECK(cudaEventRecord(prefetch_begin_,side_));
  for (size_t i = 0; i < n; ++i) {
    const int e = cand[i].second;
    const int si = stage_next_++ % store_.staging_slots();
    if (stage_used_[si]) CUDA_CHECK(cudaStreamWaitEvent(side_, stage_freed_[si], 0));
    store_.copy_to_staging(si, l, e, side_);
    ++prefetch_issued_;
    CUDA_CHECK(cudaEventRecord(stage_copied_[si], side_));
    stage_used_[si] = 1;
    pf_.push_back({e, si});
  }
  pf_l_ = l;
  if(measure) { CUDA_CHECK(cudaEventRecord(prefetch_end_,side_));prefetch_timing_pending_=true; }
}

// HIVE_WARM_DEFER (hived warm_after_prefill): the post-prefill warm as paced promotions instead of one synchronous upload. warm_cache() copies up to warm_cap
//   top-score experts and waits — measured (chat shape, 4 decoders): 738–1,065 experts in 464–693 ms per prefill, during which
//   every running decoder stalled; capping that warm while others decode (HIVE_WARM_BUSY_CAP=1) kept the decoders moving but left the cache stale (decoder
//   throughput 17.0 → 10.3 tok/s on the same shape). Here the quota (warm_cap) is handed to promote_after_step: the following decode steps promote with the
//   warm's cap (per_step) and threshold (score >= 1) in their own promote() call, so the upload rides on the side stream next to the steps (E4 pacing included)
//   until the quota is spent or no candidate is left. Scores are read at each step, so a quota left over from a short request stays valid for the next steps.
//   Lossless: promotions only decide which slot serves an expert (the same bytes); what changes is when the uploads go out.
int Runtime::image_patch_bytes() const { const auto& c = model_.cfg(); return 3 * c.vision_patch * c.vision_patch * 2; }

void Runtime::warm_defer(int per_step) {
  ElasticScope::lend_if_idle(*this);  // Q1: as warm_cache — lend the elastic slots once the prefill has finished, the promotions then fill them
  if (warm_left_ > 0 && warm_steps_ > 0)  // the previous quota is still open: report what it did (the completion line below would otherwise never appear)
    fprintf(stderr, "[runtime] deferred warm: %d experts over %d steps (cut short by the next prefill)\n", warm_done_, warm_steps_);
  warm_left_ = opt_.warm_cap; warm_step_ = std::max(1, per_step); warm_steps_ = 0; warm_done_ = 0;
}

int Runtime::warm_cache() {
  // Fill free/low-score slots from the current top scores without score decay (right after a prefill). Synchronous. At most opt.warm_cap per call (so a request never spends seconds here).
  ElasticScope::lend_if_idle(*this);  // Q1: once prefill has finished (no multi-chunk prefill in progress), lend the elastic slots first and then fill (off or already lent = does nothing)
  int total = 0;
  cudaEvent_t victim_ready = nullptr;  // REUSE_STAGE dedicated D2D stream: after the work issued on st_ so far (same marker as promote_after_step)
  if (store_.reuse_staging() && promo_evt_) { CUDA_CHECK(cudaEventRecord(promo_evt_, st_)); victim_ready = promo_evt_; }
  for (int round = 0; round < 64 && total < opt_.warm_cap; ++round) {
    int n = store_.promote(std::min(std::max(1, store_.n_slots() / 4), opt_.warm_cap - total), 1.f, promo_, victim_ready);
    if (n == 0) break;
    store_.commit_all();
    total += n;
  }
  return total;
}

// After the head: per-row top-NC candidates (radix select, ascending index) + values + max and Σexp (temperature applied) → pinned host. Callable inside a graph (fixed pointers).
void Runtime::sampler_cands_dev(int M) {
  if (opt_.sampler_cands <= 0) return;
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int NC = opt_.sampler_cands, V = c.vocab;
  k::row_max_sumexp(w.logits_b.as<float>(), M, V, cand_it_d_, cand_max_.as<float>(), cand_sum_.as<float>(), st_);
  k::topk_select_rows(w.logits_b.as<float>(), M, V, NC, V, cand_idx_.as<int32_t>(), NC, st_);
  k::gather_vals(w.logits_b.as<float>(), cand_idx_.as<int32_t>(), M, NC, V, cand_val_.as<float>(), st_);
  CUDA_CHECK(cudaMemcpyAsync(cand_idx_h_, cand_idx_.p, (size_t)M * NC * 4, cudaMemcpyDeviceToHost, st_));
  CUDA_CHECK(cudaMemcpyAsync(cand_val_h_, cand_val_.p, (size_t)M * NC * 4, cudaMemcpyDeviceToHost, st_));
  CUDA_CHECK(cudaMemcpyAsync(cand_max_h_, cand_max_.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st_));
  CUDA_CHECK(cudaMemcpyAsync(cand_sum_h_, cand_sum_.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st_));
}
int32_t Runtime::row_argmax(int m) const { return w_->next_b_h[m]; }

int Runtime::warm_from_keys(const std::vector<int32_t>& keys) {
  if (keys.empty()) return 0;
  store_.seed_scores(keys, 2.f + (float)keys.size() * 1e-3f, 1e-3f);  // list order becomes the score (earlier = higher), all ≥ min_score 1
  int total = 0;
  cudaEvent_t victim_ready = nullptr;  // REUSE_STAGE dedicated D2D stream (same as warm_cache)
  if (store_.reuse_staging() && promo_evt_) { CUDA_CHECK(cudaEventRecord(promo_evt_, st_)); victim_ready = promo_evt_; }
  for (int round = 0; round < 256; ++round) {
    int n = store_.promote(std::max(1, store_.n_slots() / 4), 1.f, promo_, victim_ready);
    if (n == 0) break;
    store_.commit_all();
    total += n;
    if (total >= (int)keys.size()) break;
  }
  return total;
}

// ---- Sleep / wake (Z1, hived {"op":"sleep"} / {"op":"wake"}) ------------------------------------------------------------------------------
// Purpose: hand hive's VRAM to another process using the GPU (weights and conversations stay) and come back afterwards — similar to SGLang's memory saver.
// Released (VRAM): the whole expert slot area (base + elastic — with HIVE_CACHE_ELASTIC the big layout of prefill work buffers, C2 tiles, H2 staging and H3
//   mbufs live inside it), the staging ring, the regions allocated separately when elasticity was abandoned (store elastic_own_, Work el.own), and the vision
//   encoder (if loaded — reloaded on the first image after wake).
// Kept: dense weights (Model, MTP stages), the session pool (Seq KV rings and compressed caches — conversations resume across sleep through prefix reuse), the
//   Work small layout (decode, verify, draft and MTP sync buffers — addresses captured by decode layer graphs), the cuBLAS workspace (64 MiB, captured by
//   graphs), sampler candidates, and all pinned host memory (expert records, engram, H2 h).
//   ⚠️With HIVE_CACHE_ELASTIC off, Work's big layout (≈ 8.8 GiB at max_chunk 16K) and tiles stay owned by Work and are not released (layer graphs capture
//   those buffers, so the addresses cannot change) — the VRAM kept while asleep is that much larger (enable ELASTIC when using sleep).
// Graphs: run_graph's layer and head graphs capture only the small layout, weights, cuBLAS workspace and session row tables (mapped pinned) (expert slot and
//   staging addresses are passed per layer call through tables — moe_decode_experts, F2/F3, D1 host tables, moe_experts_multi) → kept (no recapture after
//   wake; same graph = same kernels and arguments). The G1 step graph (sg_) stores slot and staging base addresses in its plan, so it is dropped (rebuilt at
//   the next step if on — the first use takes the base path).
// Lossless: after wake the weights, kernels and session state are the same — only the cache resident set differs (the same difference as an ordinary cache
//   hit/miss: fp32 accumulation order on the CPU/DMA paths). B2 staging tracking (bmiss track) and prefill pre-copies (pf_) are cleared because the staging
//   contents are gone (the next miss copies again — same bytes).
size_t Runtime::sleep_release() {
  if (asleep_) return 0;
  deferred_flush();
  Work& w = *w_;
  HIVE_CHECK(!w.el.big && w.el.depth == 0 && tile_cur_ == 0 && unit_cur_ < 0 && v_M_ == 0 && vparts_.empty(), "sleep: a forward is in progress");
  store_.wait_jobs();
  store_.flush_promotions();
  CUDA_CHECK(cudaDeviceSynchronize());  // all streams (st_, side_, promo_, snapshot, hc branch, H2 upload/download, REUSE D2D, early-route table copy)
  store_.commit_all();
  if (vision_) { vision_.reset(); fprintf(stderr, "[runtime] sleep: vision encoder released (reloads on the next image)\n"); }
  Work::Elastic& E = w.el;
  if (E.on && E.bound) {  // detach non-owned buffers inside the elastic region (the store frees the region; the small layout owned by members stays)
    for (auto& e : E.work) { e.alt = nullptr; e.alt_n = 0; }
    for (auto& f : E.fixed) { f.b->p = nullptr; f.b->n = 0; }
    for (auto& [b, o] : E.stage) stager_->rebind_stage(b, nullptr);
    E.own.free();
    E.base = nullptr;
    E.bound = false;
  }
  sg_.reset();
  if (bmiss_) bmiss_->track = bmiss::StageTrack{};
  pf_.clear(); pf_l_ = -1;
  warm_left_ = 0; warm_steps_ = 0; warm_done_ = 0;  // an open deferred-warm quota ends here (wake warms the cache itself)
  const size_t freed = store_.sleep_release();
  asleep_ = true;
  return freed;
}

int Runtime::wake_restore(size_t reserve_bytes, std::string* why) {
  if (!asleep_) return (int)store_.n_slots();
  const int n = store_.wake_alloc(reserve_bytes);
  if (n < 0) {
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    if (why) *why = "not enough free VRAM to restore the expert cache / prefill work region (free " + std::to_string((long long)(fr >> 20)) + " MiB)";
    return -1;
  }
  asleep_ = false;
  Work::Elastic& E = w_->el;
  if (E.on) ElasticScope::bind(*this);  // rebind to the new elastic region addresses (big layout alt, tiles, mbufs, H2 staging) — the small layout (graph addresses) stays
  return n;
}

int Runtime::wake_warm(const std::vector<int32_t>& keys, const std::function<void(int, int)>& progress) {
  if (asleep_) return 0;
  return store_.restore_keys(keys, promo_, progress);
}

void Runtime::merge_images(int M, const std::vector<ImageInput>& images) {
  ensure_vision();
  HIVE_CHECK(vision_ != nullptr, "vision encoder not loaded");
  vision_last_use_ = now_ms();
  const Config& c = model_.cfg();
  Work& w = *w_;
  const VisionWeights& V = vision_->w();
  for (const ImageInput& im : images) {
    const int span = (int)im.types.size();
    HIVE_CHECK(span <= c.vision_max_tokens + 64, "image span too long");
    std::vector<int32_t> row_of(span, 0);
    int rows = 0;
    for (int p = 0; p < span; ++p) { row_of[p] = rows; if (im.types[p] == 1) ++rows; }
    HIVE_CHECK(rows <= c.vision_max_tokens, "too many image rows");
    // M3 (opt-in HIVE_VIT_CACHE_MB): for the same picture (patch bytes, types and grid all identical) upload the stored encoder output and skip encode.
    //   Patches live only on the device, so they are fetched to the host once (≤ tens of MB, much cheaper than encode) for the key and full comparison. Output =
    //   exactly the region encode writes as aligner rows [oh·ow, dim] (within img_rows capacity). Off = same as before (the single line below).
    if (vit_cache_) {
      const size_t pbytes = (size_t)im.n_vit_h * im.n_vit_w * 3 * c.vision_patch * c.vision_patch * 2;
      const int r = c.vision_downsample, oh = (im.n_vit_h + r - 1) / r, ow = (im.n_vit_w + r - 1) / r;
      const size_t obytes = std::min((size_t)oh * ow * c.dim * 2, w.img_rows.n);
      std::vector<uint8_t> host(pbytes);
      CUDA_CHECK(cudaMemcpyAsync(host.data(), im.patches_dev, pbytes, cudaMemcpyDeviceToHost, st_));
      CUDA_CHECK(cudaStreamSynchronize(st_));
      const uint64_t h = VitCache::hash_of(host.data(), pbytes, im.n_vit_h, im.n_vit_w, im.types.data(), im.types.size());
      if (const VitCache::Entry* e = vit_cache_->find(h, host.data(), pbytes, im.n_vit_h, im.n_vit_w, im.types.data(), im.types.size());
          e && e->rows.size() == obytes) {
        CUDA_CHECK(cudaMemcpyAsync(w.img_rows.p, e->rows.data(), obytes, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));  // the source is a cache entry (the next insert may evict it) — release only after the upload
      } else {
        vision_->encode(im.patches_dev, im.n_vit_h, im.n_vit_w, w.img_rows.as<bf16>());
        VitCache::Entry ne;
        ne.hash = h; ne.n_vit_h = im.n_vit_h; ne.n_vit_w = im.n_vit_w; ne.types = im.types; ne.patches = std::move(host);
        ne.rows.resize(obytes);
        CUDA_CHECK(cudaMemcpyAsync(ne.rows.data(), w.img_rows.p, obytes, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        vit_cache_->insert(std::move(ne));
      }
    } else
    vision_->encode(im.patches_dev, im.n_vit_h, im.n_vit_w, w.img_rows.as<bf16>());
    CUDA_CHECK(cudaMemcpyAsync(w.img_types.p, im.types.data(), (size_t)span, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(w.img_row_of.p, row_of.data(), (size_t)span * 4, cudaMemcpyHostToDevice, st_));
    k::merge_image(w.h.as<bf16>(), c.hc, c.dim, im.start, span, w.img_types.as<int8_t>(), w.img_row_of.as<int32_t>(), w.img_rows.as<bf16>(),
                   V.image_start.as<bf16>(), V.image_newline.as<bf16>(), V.image_end.as<bf16>(), st_);
    if (!opt_.dump_dir.empty()) dump("image_emb_" + std::to_string(im.start) + ".bf16", w.img_rows.p, (size_t)rows * c.dim * 2);
    CUDA_CHECK(cudaStreamSynchronize(st_));  // complete before the types/row_of host buffers are reused
  }
}

void Runtime::hc_attn_pre(const LayerWeights& L, int M) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int dim = c.dim, hc = c.hc, mix_hc = (2 + hc) * hc;
  if (M <= 8) {  // decode: mix coefficients and rsqrt → [branch: sinkhorn (overlaps attention)] + [main: pre+norm]. Joined before hc_post in hc_ffn_pre.
    static const bool fuse2 = getenv("HIVE_HC_FUSE2") && atoi(getenv("HIVE_HC_FUSE2")) != 0;  // hc_mix + hc_pre_norm in one launch (bit-identical; off by default until measured)
    if (hc_dfused_on(hc)) {  // E3 HIVE_HC_DECODE_FUSED: mix + pre+norm + sinkhorn (last mix block) in one launch — no branch or fork (bit-identical, test_hc_decode)
      (k::decode_gemv2_on() ? k::hc_mix_pre_norm2_sk : k::hc_mix_pre_norm_sk)(w.h.as<bf16>(), L.hc_attn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(),
          w.rsq.as<float>(), w.pre_mix.as<float>(), hc, dim, L.attn_norm.as<bf16>(), c.norm_eps, w.x.as<bf16>(), w.xn.as<bf16>(), L.hc_attn_scale.as<float>(),
          L.hc_attn_base.as<float>(), c.sinkhorn_iters, c.hc_eps, w.pre_a.as<float>(), w.post_a.as<float>(), w.comb_a.as<float>(), w.counters.as<int>() + 3, st_);
      CUDA_CHECK(cudaEventRecord(hc_join_[0], st_));  // the existing join wait (hc_ffn_pre) still holds — same stream, no extra edge
      pmark("hc.fused");
      return;
    }
    if (fuse2) (k::decode_hcmix3_on() ? k::hc_mix_pre_norm3 : k::decode_gemv2_on() ? k::hc_mix_pre_norm2 : k::hc_mix_pre_norm)(w.h.as<bf16>(), L.hc_attn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(), w.rsq.as<float>(),
                                  w.pre_mix.as<float>(), hc, dim, L.attn_norm.as<bf16>(), c.norm_eps, w.x.as<bf16>(), w.xn.as<bf16>(), st_);
    else k::hc_mix(w.h.as<bf16>(), L.hc_attn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(), w.rsq.as<float>(), st_);
    CUDA_CHECK(cudaEventRecord(hc_fork_[0], st_));
    CUDA_CHECK(cudaStreamWaitEvent(hc_side_, hc_fork_[0], 0));
    (k::hc_sinkhorn_par_on() ? k::hc_split_sinkhorn_par : k::hc_split_sinkhorn)(w.mixes.as<float>(), w.rsq.as<float>(), L.hc_attn_scale.as<float>(), L.hc_attn_base.as<float>(), M, hc, c.sinkhorn_iters,
                                                                                   c.hc_eps, w.pre_a.as<float>(), w.post_a.as<float>(), w.comb_a.as<float>(), hc_side_);  // E3 HIVE_HC_SINKHORN_PAR: 16-lane variant (bit-identical)
    CUDA_CHECK(cudaEventRecord(hc_join_[0], hc_side_));
    if (!fuse2) k::hc_pre_norm(w.h.as<bf16>(), w.pre_mix.as<float>(), M, hc, dim, L.attn_norm.as<bf16>(), c.norm_eps, w.x.as<bf16>(), w.xn.as<bf16>(), st_);
    pmark("hc.fused");
    return;
  }
  if (env_on("HIVE_HC_CUBLAS")) {  // for comparison (alternative path)
    k::hc_flatten_f32(w.h.as<bf16>(), M, hc * dim, w.hflat.as<float>(), st_);
    blas_->gemm_f32(w.hflat.as<float>(), L.hc_attn_fn.as<float>(), w.mixes.as<float>(), M, mix_hc, hc * dim);
    k::hc_row_rsqrt(w.h.as<bf16>(), M, hc * dim, c.norm_eps, w.rsq.as<float>(), st_);
  } else {
    k::hc_mix_rows(w.h.as<bf16>(), L.hc_attn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(), w.rsq.as<float>(), st_);  // P5
  }
  pmark("hc.flat+gemm");
  k::hc_split_sinkhorn(w.mixes.as<float>(), w.rsq.as<float>(), L.hc_attn_scale.as<float>(), L.hc_attn_base.as<float>(), M, hc,
                       c.sinkhorn_iters, c.hc_eps, w.pre_a.as<float>(), w.post_a.as<float>(), w.comb_a.as<float>(), st_);
  pmark("hc.sinkhorn");
  k::hc_pre(w.h.as<bf16>(), w.pre_mix.as<float>(), M, hc, dim, w.x.as<bf16>(), st_);
  k::rmsnorm(w.x.as<bf16>(), L.attn_norm.as<bf16>(), c.norm_eps, M, dim, w.xn.as<bf16>(), st_);
  pmark("hc.pre+norm");
}

void Runtime::hc_ffn_pre(const LayerWeights& L, int M) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int dim = c.dim, hc = c.hc, mix_hc = (2 + hc) * hc;
  if (M <= 8) CUDA_CHECK(cudaStreamWaitEvent(st_, hc_join_[0], 0));  // join attention sinkhorn (post_a, comb_a, pre_a)
  if (dov_ && dov_->pg_arm) CUDA_CHECK(cudaStreamWaitEvent(st_, dov_->pg_read, 0));  // P1: hc_post below overwrites h — wait until the prediction has read h
  k::hc_post(w.attn_out.as<bf16>(), w.post_a.as<float>(), w.comb_a.as<float>(), M, hc, dim, w.h.as<bf16>(), st_);
  if (M <= 8) {
    if (hc_dfused_on(hc)) {  // E3 HIVE_HC_DECODE_FUSED (same variant as hc_attn_pre) — pre_f, post_f and comb_f on the main stream too
      (k::decode_gemv2_on() ? k::hc_mix_pre_norm2_sk : k::hc_mix_pre_norm_sk)(w.h.as<bf16>(), L.hc_ffn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(),
          w.rsq.as<float>(), w.pre_a.as<float>(), hc, dim, L.ffn_norm.as<bf16>(), c.norm_eps, w.x.as<bf16>(), w.xn.as<bf16>(), L.hc_ffn_scale.as<float>(),
          L.hc_ffn_base.as<float>(), c.sinkhorn_iters, c.hc_eps, w.pre_f.as<float>(), w.post_f.as<float>(), w.comb_f.as<float>(), w.counters.as<int>() + 4, st_);
      CUDA_CHECK(cudaEventRecord(hc_join_[1], st_));  // the existing join waits (layer_forward, tail_prev, decode_front, mtp) still hold — same stream
      pmark("hc.fused");
      return;
    }
    static const bool fuse2 = getenv("HIVE_HC_FUSE2") && atoi(getenv("HIVE_HC_FUSE2")) != 0;
    if (fuse2) (k::decode_hcmix3_on() ? k::hc_mix_pre_norm3 : k::decode_gemv2_on() ? k::hc_mix_pre_norm2 : k::hc_mix_pre_norm)(w.h.as<bf16>(), L.hc_ffn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(), w.rsq.as<float>(),
                                  w.pre_a.as<float>(), hc, dim, L.ffn_norm.as<bf16>(), c.norm_eps, w.x.as<bf16>(), w.xn.as<bf16>(), st_);
    else k::hc_mix(w.h.as<bf16>(), L.hc_ffn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(), w.rsq.as<float>(), st_);
    CUDA_CHECK(cudaEventRecord(hc_fork_[1], st_));
    CUDA_CHECK(cudaStreamWaitEvent(hc_side_, hc_fork_[1], 0));
    (k::hc_sinkhorn_par_on() ? k::hc_split_sinkhorn_par : k::hc_split_sinkhorn)(w.mixes.as<float>(), w.rsq.as<float>(), L.hc_ffn_scale.as<float>(), L.hc_ffn_base.as<float>(), M, hc, c.sinkhorn_iters,
                                                                                   c.hc_eps, w.pre_f.as<float>(), w.post_f.as<float>(), w.comb_f.as<float>(), hc_side_);  // E3 HIVE_HC_SINKHORN_PAR
    CUDA_CHECK(cudaEventRecord(hc_join_[1], hc_side_));
    if (!fuse2) k::hc_pre_norm(w.h.as<bf16>(), w.pre_a.as<float>(), M, hc, dim, L.ffn_norm.as<bf16>(), c.norm_eps, w.x.as<bf16>(), w.xn.as<bf16>(), st_);
    pmark("hc.fused");
    return;
  }
  if (env_on("HIVE_HC_CUBLAS")) {
    k::hc_flatten_f32(w.h.as<bf16>(), M, hc * dim, w.hflat.as<float>(), st_);
    blas_->gemm_f32(w.hflat.as<float>(), L.hc_ffn_fn.as<float>(), w.mixes.as<float>(), M, mix_hc, hc * dim);
    k::hc_row_rsqrt(w.h.as<bf16>(), M, hc * dim, c.norm_eps, w.rsq.as<float>(), st_);
  } else {
    k::hc_mix_rows(w.h.as<bf16>(), L.hc_ffn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(), w.rsq.as<float>(), st_);  // P5
  }
  pmark("hc.flat+gemm");
  k::hc_split_sinkhorn(w.mixes.as<float>(), w.rsq.as<float>(), L.hc_ffn_scale.as<float>(), L.hc_ffn_base.as<float>(), M, hc,
                       c.sinkhorn_iters, c.hc_eps, w.pre_f.as<float>(), w.post_f.as<float>(), w.comb_f.as<float>(), st_);
  pmark("hc.sinkhorn");
  k::hc_pre(w.h.as<bf16>(), w.pre_a.as<float>(), M, hc, dim, w.x.as<bf16>(), st_);
  k::rmsnorm(w.x.as<bf16>(), L.ffn_norm.as<bf16>(), c.norm_eps, M, dim, w.xn.as<bf16>(), st_);
  pmark("hc.pre+norm");
}

// Layer tail: expert sum (fp32) → bf16 → hc_post → pre_mix ← ffn_pre (a copy — pointers must not change so the step can be put into a graph)
void Runtime::layer_tail(int l, int M) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int dim = c.dim, hc = c.hc;
  if (fuse_) {
    k::hc_post_tail(w.acc.as<float>(), w.post_f.as<float>(), w.comb_f.as<float>(), M, hc, dim, w.h.as<bf16>(), w.pre_f.as<float>(), w.pre_mix.as<float>(), st_);
  } else {
    k::f32_to_bf16(w.acc.as<float>(), M * dim, w.ffn_out.as<bf16>(), st_);
    k::hc_post(w.ffn_out.as<bf16>(), w.post_f.as<float>(), w.comb_f.as<float>(), M, hc, dim, w.h.as<bf16>(), st_);
    CUDA_CHECK(cudaMemcpyAsync(w.pre_mix.p, w.pre_f.p, (size_t)M * hc * 4, cudaMemcpyDeviceToDevice, st_));
  }
  if (!opt_.dump_dir.empty()) {
    dump("h_L" + std::to_string(l) + ".bf16", w.h.p, (size_t)M * hc * dim * 2);
    dump("premix_L" + std::to_string(l) + ".f32", w.pre_mix.p, (size_t)M * hc * 4);
  }
}

void Runtime::layer_forward(Seq& seq, int l, int M, int64_t start_pos, ForwardStats* stats, bool skip_compress) {
  const Config& c = model_.cfg();
  const LayerWeights& L = model_.layer(l);
  Work& w = *w_;
  const int dim = c.dim, hc = c.hc;
  if (batch_) { decode_layer(*batch_, l, M, stats); return; }
  const bool short_moe = prefill_short_moe(M);  // R2 HIVE_PREFILL_TAIL_SHORT: no pre-copy when this layer's experts take the short-prefill path (that path does not use pre-copies)
  if (pprof_) pprof_->layer_begin(l, M, short_moe, stats, prefetch_issued_, st_);
  if ((M >= opt_.prefill_threshold || in_prefill_) && !verify_ && !short_moe) prefetch_layer_experts(l, M);  // expert pre-copy during the front computation
  inject_layer_input(l, M);
  if (L.engram) {
    engram(seq, L, M, w.ids_h);
    if (!opt_.dump_dir.empty()) dump("engram_L" + std::to_string(l) + ".bf16", w.h.p, (size_t)M * hc * dim * 2);
  }
  if (mtp_on_) mtp_capture(l, M, start_pos);
  hc_attn_pre(L, M);
  pmark("hc_attn");
  attention(seq, L, l, M, start_pos, skip_compress);
  pmark("attention");
  hc_ffn_pre(L, M);
  pmark("hc_ffn");
  moe(seq, L, l, M, stats);
  pmark("moe");
  if (M <= 8) CUDA_CHECK(cudaStreamWaitEvent(st_, hc_join_[1], 0));  // join the fused path's ffn sinkhorn branch (post_f, comb_f) — moe's sync only waits for st_
  layer_tail(l, M);
  if (pprof_) pprof_->layer_end(stats, prefetch_issued_, st_);
}

// C2: the layer's front (engram, attention, hc, router, shared expert, sync) — the same as layer_forward's prefill path without experts and tail
void Runtime::layer_front(Seq& seq, int l, int M, int64_t start_pos, bool skip_compress) {
  const LayerWeights& L = model_.layer(l);
  Work& w = *w_;
  if (L.engram) engram(seq, L, M, w.ids_h);
  if (mtp_on_) mtp_capture(l, M, start_pos);
  hc_attn_pre(L, M);
  pmark("hc_attn");
  attention(seq, L, l, M, start_pos, skip_compress);  // H3: true only for the tail layer (prepare_decoder_tail already compressed) — plain calls default to false
  pmark("attention");
  hc_ffn_pre(L, M);
  pmark("hc_ffn");
  moe_router_shared(L, M, false);
  pmark("moe.router+shared");
  CUDA_CHECK(cudaStreamSynchronize(st_));
  pmark("moe.sync");
}

// Put sub-chunk s's buffers into Work (s=0 is Work itself). tile_select is idempotent and remembers the current slot — when a forward ends (including by
//   exception, TileGuard) it must be reset to 0: decode CUDA graphs capture Work buffer addresses by value, so if Work held other buffers the graphs would read
//   the wrong memory. Only what must live between layers is swapped:
//   h, pre_mix (residual), idx (written by the index source layer, read by reuse layers), xq/xs and acc (expert input, accumulation), pre_f/post_f/comb_f
//   (made by hc_ffn_pre, used by layer_tail), routing tables, positions, image markers, visible/gpos (mapped pinned), shared topk column count, engram hashes (made by layer 1, used by 14)
void Runtime::tile_select(int s) {
  if (s == tile_cur_) return;
  if (tile_cur_ != 0) { tile_xchg(tile_cur_); tile_cur_ = 0; }  // first back to the original (Work = sub-chunk 0)
  if (s != 0) tile_xchg(s);
  tile_cur_ = s;
}
void Runtime::tile_xchg(int s) {
  Work& w = *w_;
  TileSlot& t = tiles_.at((size_t)s - 1);
  std::swap(w.h, t.h); std::swap(w.pre_mix, t.pre_mix); std::swap(w.idx, t.idx);
  std::swap(w.xq, t.xq); std::swap(w.xs, t.xs); std::swap(w.acc, t.acc);
  std::swap(w.pre_f, t.pre_f); std::swap(w.post_f, t.post_f); std::swap(w.comb_f, t.comb_f);
  std::swap(w.route_ids_h, t.route_ids_h); std::swap(w.route_ids_d, t.route_ids_d); std::swap(w.rw_h, t.rw_h); std::swap(w.rw_d, t.rw_d);
  std::swap(w.pos_h, t.pos_h); std::swap(w.pos_d, t.pos_d); std::swap(w.ids_h, t.ids_h);
  std::swap(w.is_image_h, t.is_image_h); std::swap(w.is_image_d, t.is_image_d);
  std::swap(w.visible_h, t.visible_h); std::swap(w.visible_d, t.visible_d); std::swap(w.gpos_h, t.gpos_h); std::swap(w.gpos_d, t.gpos_d);
  std::swap(shared_topk_cols_, t.topk_cols); std::swap(seq_hashes_, t.hashes);
}

// ---------------------------------------------------------------------------------------------------------------------
// CUDA graphs: first use runs eagerly (so lazy init and cudaFuncSetAttribute finish outside capture), the second use captures and instantiates, later uses replay.
void Runtime::run_graph(int l, int M, int Tb, const std::function<void()>& body) {
  if (!graphs_) { body(); return; }
  GraphEntry& g = graphs_map_[std::make_tuple(l, vdec_ ? M + kVerifyGraphKey : M, Tb)];  // R1: the verify front has a different body (verify_decode.h steps 1–3), hence its own key
  if (g.exec) { CUDA_CHECK(cudaGraphLaunch(g.exec, st_)); return; }
  if (g.uses++ < 1) { ++graph_eager_; body(); return; }  // MB1: first use = eager (the batched MTP gate filters out this step's cost sample)
  cudaGraph_t graph = nullptr;
  capturing_ = true;
  ++graph_captures_;  // R1 (HIVE_MTP_GATE3: the gate discards cost samples of steps that included a capture)
  CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
  try {
    body();
  } catch (...) {  // leaving with the capture still open would turn every later st_ launch into a capture error — close it and discard
    cudaGraph_t bad = nullptr;
    cudaStreamEndCapture(st_, &bad);
    if (bad) cudaGraphDestroy(bad);
    cudaGetLastError();
    capturing_ = false;
    throw;
  }
  CUDA_CHECK(cudaStreamEndCapture(st_, &graph));
  capturing_ = false;
  CUDA_CHECK(cudaGraphInstantiate(&g.exec, graph, 0));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaGraphLaunch(g.exec, st_));
}

// D1 HIVE_DECODE_BLOCKING_SYNC: wait for the decode layer's routing sync (end of graph A → host reads route_ids_h) with a blocking wait instead of spinning.
//   Measured (service build, perf over 40 decode steps): of the engine thread's 100 %, 59 % was cuStreamSynchronize in decode_layer (default schedule = spin).
//   On: record a cudaEventBlockingSync event at the end of st_ and cudaEventSynchronize (OS wait) — same completion condition as a stream sync (all earlier
//   st_ work, including in-graph branches). HIVE_DECODE_SYNC_SPIN_US=N (default 0) first polls cudaEventQuery for N µs, then blocks (to compare wake-up
//   latency of short waits). Off (default) = plain cudaStreamSynchronize. On HIVE_PROFILE steps both modes count host wait time and print one [decode-sync]
//   line at the last layer (it must not start with [profile — analyze_hived_log.py splits steps by the order of [profile lines).
//   One event per thread (record and wait happen in turn on the same thread — never mixed with another engine thread).
namespace {
struct DecodeSyncProf { double ms = 0; long n = 0; };
thread_local DecodeSyncProf t_dsync_prof;
thread_local cudaEvent_t t_dsync_ev = nullptr;
void decode_stream_wait(cudaStream_t st, bool prof) {
  static const bool blocking = env_on("HIVE_DECODE_BLOCKING_SYNC");
  static const int spin_us = getenv("HIVE_DECODE_SYNC_SPIN_US") ? std::max(0, atoi(getenv("HIVE_DECODE_SYNC_SPIN_US"))) : 0;
  const double t0 = prof ? now_ms() : 0.0;
  if (!blocking) {
    CUDA_CHECK(cudaStreamSynchronize(st));
  } else {
    if (!t_dsync_ev) CUDA_CHECK(cudaEventCreateWithFlags(&t_dsync_ev, cudaEventBlockingSync | cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(t_dsync_ev, st));
    bool done = false;
    if (spin_us > 0) {
      const double t_end = now_ms() + spin_us / 1000.0;
      for (;;) {
        const cudaError_t e = cudaEventQuery(t_dsync_ev);
        if (e == cudaSuccess) { done = true; break; }
        if (e != cudaErrorNotReady) CUDA_CHECK(e);
        if (now_ms() >= t_end) break;
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
      }
    }
    if (!done) CUDA_CHECK(cudaEventSynchronize(t_dsync_ev));
  }
  if (prof) { t_dsync_prof.ms += now_ms() - t0; ++t_dsync_prof.n; }
}
void decode_sync_report(int M) {
  if (t_dsync_prof.n == 0) return;
  fprintf(stderr, "[decode-sync M=%d] layer waits %ld · host wait %.2f ms (%s)\n", M, t_dsync_prof.n, t_dsync_prof.ms,
          env_on("HIVE_DECODE_BLOCKING_SYNC") ? "blocking" : "spin");
  t_dsync_prof = DecodeSyncProf{};
}
// O1 C: UBATCH waits for the half's front-end event rather than the whole stream (the other half's front is queued behind it on st_). Same switch and
//   instrumentation — with HIVE_DECODE_BLOCKING_SYNC the event is created with cudaEventBlockingSync for an OS wait, otherwise a default event (spin wait like a stream sync).
bool decode_blocking_sync() { static const bool b = env_on("HIVE_DECODE_BLOCKING_SYNC"); return b; }
void decode_event_wait(cudaEvent_t e, bool prof) {
  const double t0 = prof ? now_ms() : 0.0;
  CUDA_CHECK(cudaEventSynchronize(e));
  if (prof) { t_dsync_prof.ms += now_ms() - t0; ++t_dsync_prof.n; }
}
// O1 B (HIVE_DECODE_HOST_FAST): pin the decode engine thread (the thread calling forward_batch) to the CPUs of the GPU's NUMA node — once per thread.
//   HIVE_ENGINE_CPU=n (integer ≥ 0) pins to that single core. Node from /sys/bus/pci/devices/<GPU PCI>/numa_node, CPU list from /sys/devices/system/node/nodeN/cpulist.
//   If unreadable (virtualization, node -1) or odd, do nothing (fallback — pinning is only a performance hint, no effect on values). CPU expert pool threads
//   are also bound to all node CPUs (expert_store.cpp worker) and share cores with the engine thread — the benefit is judged by A/B (goal: avoid cross-node
//   migration and remote cache traffic).
void pin_engine_thread_once() {
  thread_local bool done = false;
  if (done) return;
  done = true;
  cpu_set_t set;
  CPU_ZERO(&set);
  int n_set = 0;
  std::string how;
  if (const char* v = getenv("HIVE_ENGINE_CPU"); v && *v) {
    char* end = nullptr;
    const long cpu = strtol(v, &end, 10);
    if (end && *end == 0 && cpu >= 0 && cpu < CPU_SETSIZE) { CPU_SET((int)cpu, &set); n_set = 1; how = "HIVE_ENGINE_CPU=" + std::string(v); }
  }
  if (n_set == 0) {
    int dev = 0;
    char bus[64] = {0};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetPCIBusId(bus, (int)sizeof bus, dev) != cudaSuccess) { (void)cudaGetLastError(); return; }
    for (char* q = bus; *q; ++q) *q = (char)tolower((unsigned char)*q);
    int node = -1;
    { std::ifstream f(std::string("/sys/bus/pci/devices/") + bus + "/numa_node"); if (!(f >> node)) node = -1; }
    if (node < 0) return;
    std::string list;
    { std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist"); if (!std::getline(f, list)) return; }
    // "0-15,32-47" format
    size_t i = 0;
    while (i < list.size()) {
      size_t j = list.find(',', i);
      if (j == std::string::npos) j = list.size();
      const std::string part = list.substr(i, j - i);
      const size_t dash = part.find('-');
      const int a = atoi(part.c_str()), b = dash == std::string::npos ? a : atoi(part.c_str() + dash + 1);
      for (int x = a; x <= b && x >= 0 && x < CPU_SETSIZE; ++x) { CPU_SET(x, &set); ++n_set; }
      i = j + 1;
    }
    how = "GPU " + std::string(bus) + " node " + std::to_string(node) + " cpus " + list;
  }
  if (n_set == 0) return;
  const int rc = pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  fprintf(stderr, "[runtime] decode engine thread pinned (%s)%s\n", how.c_str(), rc == 0 ? "" : " — failed, left unpinned");
}
}  // namespace

// Graph A body of a batched decode layer (previous layer tail + engram + attention + hc + router + shared expert + D2H). G1: moved verbatim out of decode_layer's
//   lambda — the step graph (HIVE_DECODE_STEP_GRAPH) calls the same body per layer. copy_acts = also copy activations (xq/xs) to mapped host memory with the routing (CPU miss path).
void Runtime::decode_front(int l, int M, int Tb, bool copy_acts) {
  const Config& c = model_.cfg();
  const LayerWeights& L = model_.layer(l);
  Work& w = *w_;
  if (l == 0) {
    k::embed_expand(w.emb_d, w.iota.as<int32_t>(), M, c.dim, c.hc, w.h.as<bf16>(), st_);
    k::identity_pre_mix(w.pre_mix.as<float>(), M, c.hc, st_);
  } else {
    layer_tail(l - 1, M);
    inject_layer_input(l, M);  // eager mode only (a no-op check under graphs)
  }
  if (L.engram) {
    engram_dev(L, M);
    if (!opt_.dump_dir.empty()) dump("engram_L" + std::to_string(l) + ".bf16", w.h.p, (size_t)M * c.hc * c.dim * 2);
  }
  if (mtp_on_) mtp_capture(l, M, -1);  // batched: row positions are w.pos_h (used by mtp_sync_batch after capture)
  hc_attn_pre(L, M);
  pmark("hc_attn");
  if (dov_ && dov_->pg_arm) pregate_dev(L, M);  // P1: prediction (separate stream — overlaps attention); off = nothing issued
  if (vdec_) attention_verify_dev(L, l, M, Tb);  // R1 verify rows (consecutive positions of the same sequence) — only window, compressor and ring order differ (verify_decode.h)
  else attention_decode_dev(L, l, M, Tb);
  hc_ffn_pre(L, M);
  pmark("hc_ffn");
  moe_router_shared(L, M, copy_acts);
  pmark("moe.router+shared");
  if (M <= 8) CUDA_CHECK(cudaStreamWaitEvent(st_, hc_join_[1], 0));  // join the ffn sinkhorn branch inside the same graph (otherwise EndCapture reports "unjoined work") — the branch only exists for M ≤ 8
  if (dov_ && dov_->pg_arm) CUDA_CHECK(cudaStreamWaitEvent(st_, dov_->pg_done, 0));  // P1 join of the prediction stream (shorter than attention, already done)
}

// Host tables of a batched decode layer: engram row lookup + attention row tables (mapped pinned). Returns the indexer Tmax. O1: split out of decode_layer
//   (HOST_FAST calls it for the next layer while this layer's experts run — the tables are read only by the front graph, and that graph (this layer's) has already finished).
int Runtime::decode_host_prep(std::vector<Seq*>& seqs, int l, int M, ForwardStats* stats) {
  const LayerWeights& L = model_.layer(l);
  Work& w = *w_;
  const double th0 = now_ms();
  if (L.engram) engram_host(L, M, w.ids_h, &seqs, nullptr);
  const int Tmax = vdec_ ? attention_verify_host(L, l, M) : attention_decode_host(seqs, L, l, M);  // R1 (engram_host unchanged — seqs repeats the same sequence per row, so history accumulates in row order)
  const double dt = now_ms() - th0;
  if (stats) stats->ms_host += dt;
  if (dov_->on) dov_->h_hprep += dt;
  return Tmax;
}

// P1 HIVE_DECODE_PREGATE (see the comment at the top of hive/pregate.h): predict layer l's routing — inside the front graph, right after hc_attn_pre.
//   Approximates the FFN input from pre_a (= the coefficients the FFN uses to fold h, attn_pre of the reference Block.forward — already computed from the
//   pre-attention h) and the h **before** attention, and runs the real router (same weights and bias, top K) on it. Writes go only to prediction buffers —
//   the main path's h, pre_a and routing are only read (no effect on values). hc_post in hc_ffn_pre overwrites h, so st_ waits for pg_read before it and
//   for pg_done at the end of the front (decode_front, hc_ffn_pre). Publishing = the same kernel as er_route_post (sequence number + mapped host copy).
void Runtime::pregate_dev(const LayerWeights& L, int M) {
  const Config& c = model_.cfg();
  const MoeWeights& F = L.ffn;
  Work& w = *w_;
  DecodeOverlap& D = *dov_;
  const int dim = c.dim, hc = c.hc, E = L.n_routed ? L.n_routed : c.n_routed, K = pregate_k_;
  CUDA_CHECK(cudaStreamWaitEvent(D.pg_st, hc_join_[0], 0));  // pre_a (attention sinkhorn branch or fused variant), h (after layer_tail and engram)
  k::hc_pre_norm(w.h.as<bf16>(), w.pre_a.as<float>(), M, hc, dim, L.ffn_norm.as<bf16>(), c.norm_eps, D.pg_x, D.pg_xn, D.pg_st);
  CUDA_CHECK(cudaEventRecord(D.pg_read, D.pg_st));
  k::fused_router(D.pg_xn, dim, F.gate_w.as<float>(), M, E, F.gate_bias.as<float>(), F.gate_bias_vl.p ? F.gate_bias_vl.as<float>() : nullptr, w.is_image_d, K,
                  c.route_scale, D.pg_scores, D.pg_ids_d, D.pg_rw_d, D.pg_counter, D.pg_st);
  k::er_route_post(D.pg_ids_d, D.pg_rw_d, M * K, D.pg_ids_hd, D.pg_rw_hd, D.pg_ctr_d, D.pg_flag_d, D.pg_st);
  CUDA_CHECK(cudaEventRecord(D.pg_done, D.pg_st));
}

// Right after the front launch (engine thread — this publish arrives before the next item, the routing publish wait). Sends predicted non-resident experts to the store's prefetch.
//   If the publish is not received (front finished first, previous launch not closed) the layer goes without prefetch (fallback — no effect on results).
void Runtime::pregate_host(int l, int M) {
  DecodeOverlap& D = *dov_;
  D.pg_npf = 0; D.pg_l = l;
  if (!D.pg_gate.begin()) { D.pg_resync = true; ++D.pg_acc.untrusted; return; }
  const int res = D.pg_gate.wait(D.pg_flag_h, [&] {
    const cudaError_t e = cudaEventQuery(D.er_front);
    if (e != cudaErrorNotReady) CUDA_CHECK(e);
    return e == cudaSuccess;
  });
  if (res != er::kOk) { ++D.pg_acc.late; return; }
  ++D.pg_acc.posted;
  const LayerWeights& L = model_.layer(l);
  const int E = L.n_routed ? L.n_routed : model_.cfg().n_routed;
  D.pg_npf = pg::plan(D.pg_ids_h, M, pregate_k_, E, [&](int e) { return store_.slot_of(l, e) >= 0; }, D.pg_pf, pg::kMaxPf);
  if (D.pg_npf > 0) store_.prefetch_experts(l, D.pg_pf, D.pg_npf);
}

// Batched decode layer: host prep → graph A → sync → experts. The previous layer's tail is attached to the head of this layer's graph (it must follow the expert sum).
void Runtime::decode_layer(std::vector<Seq*>& seqs, int l, int M, ForwardStats* stats) {
  const Config& c = model_.cfg();
  const LayerWeights& L = model_.layer(l);
  // HIVE_CACHE_PRIOR: this layer's residency → its mask row, read by the router of the front about to be launched (graphs capture the row's
  //   address — the content is refreshed here every step). Draft layers (l ≥ n_layers) are not biased.
  struct CpReset { int& v; ~CpReset() { v = -1; } } cp_reset{cp_layer_};
  if (cache_prior_ > 0.f && l < c.n_layers) {
    uint8_t* row = cache_mask_h_ + (size_t)l * c.n_routed;
    for (int e = 0; e < c.n_routed; ++e) row[e] = store_.slot_of(l, e) >= 0;
    cp_layer_ = l;
  }
  Work& w = *w_;
  DecodeOverlap& D = *dov_;
  const int slot = D.slot(l, 0);
  // 1) host: write mapped buffers — after the stream sync of the previous layer's moe, so overwriting tables earlier kernels read is safe (O1 HOST_FAST: reuse them if the previous layer prefilled them)
  int Tmax;
  if (host_fast_ && D.pre_l == l) { Tmax = D.pre_Tmax; D.pre_l = -1; }
  else Tmax = decode_host_prep(seqs, l, M, stats);
  int Tb = 0;
  if (c.ratio(l) && c.is_index_source(l)) Tb = graphs_ ? std::min(w.Tcap, std::max(1024, next_pow2(Tmax))) : Tmax;
  // S1 HIVE_DECODE_EARLY_ROUTE (see the comment at the top of hive/early_route.h): receive routing as a publish right after the router and, while the shared
  //   expert runs, do classification, DMA, tables and the GPU expert launch (in stream order after the front — same kernels, inputs and order = bit-identical).
  //   CPU unpacking, which needs activations, and the pool start come after the front ends. Conditions = switch, fused router path (fuse_ && M ≤ 8 —
  //   moe_router_shared issues the router and shared expert separately), no dump/inject (layer-boundary dumps read the synchronized routing). The conditions
  //   are fixed for the process lifetime and M is part of the graph key — graphs with the same key (l, M, Tb) always have the same body (publish included). The base path below is unchanged.
  if (def_pending_ && !(D.er_on && fuse_ && M <= 8 && opt_.dump_dir.empty() && opt_.inject_dir.empty())) deferred_flush();  // the next path unpacks in place
  if (D.er_on && fuse_ && M <= 8 && opt_.dump_dir.empty() && opt_.inject_dir.empty()) {
    if (!D.er_front) {  // first use (outside capture — graphs capture these pointers by value); sequence number and counter start at 0
      D.er_flag_h = pinned_mapped<uint32_t>(1, &D.er_flag_d);
      *D.er_flag_h = 0;
      CUDA_CHECK(cudaMalloc((void**)&D.er_ctr_d, sizeof(uint32_t)));
      CUDA_CHECK(cudaMemsetAsync(D.er_ctr_d, 0, sizeof(uint32_t), st_));
      CUDA_CHECK(cudaStreamSynchronize(st_));
      CUDA_CHECK(cudaEventCreateWithFlags(&D.er_front, cudaEventDisableTiming));  // default schedule (spin) — same wake-up as cudaStreamSynchronize
      {  // T2: stream dedicated to the expert-table H2D (see the table-copy comment in moe_decode_experts) — with E4 on, the same highest priority as demand DMA (side_)
        int least = 0, greatest = 0;
        if (copy_prio_) CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&D.er_tbl_st, cudaStreamNonBlocking, copy_prio_ ? greatest : 0));
        CUDA_CHECK(cudaEventCreateWithFlags(&D.er_tbl, cudaEventDisableTiming));
      }
      D.er_gate = er::Gate{};
      fprintf(stderr, "[runtime] decode early route on (routing posted after the router · CPU unpack after the front graph)\n");
    }
    if (pregate_k_ > 0 && !D.pg_st) {  // P1 first use (outside capture — graphs capture these pointers by value); sequence number and counter start at 0
      constexpr int kPgM = 8;          // same as this path's condition M ≤ 8
      CUDA_CHECK(cudaStreamCreateWithFlags(&D.pg_st, cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreateWithFlags(&D.pg_read, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&D.pg_done, cudaEventDisableTiming));
      CUDA_CHECK(cudaMalloc((void**)&D.pg_x, (size_t)kPgM * c.dim * sizeof(bf16)));
      CUDA_CHECK(cudaMalloc((void**)&D.pg_xn, (size_t)kPgM * c.dim * sizeof(bf16)));
      CUDA_CHECK(cudaMalloc((void**)&D.pg_scores, (size_t)kPgM * 512 * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void**)&D.pg_ids_d, (size_t)kPgM * pg::kMaxK * sizeof(int32_t)));
      CUDA_CHECK(cudaMalloc((void**)&D.pg_rw_d, (size_t)kPgM * pg::kMaxK * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void**)&D.pg_ctr_d, sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc((void**)&D.pg_counter, sizeof(int)));
      CUDA_CHECK(cudaMemsetAsync(D.pg_ctr_d, 0, sizeof(uint32_t), st_));
      CUDA_CHECK(cudaMemsetAsync(D.pg_counter, 0, sizeof(int), st_));
      D.pg_ids_h = pinned_mapped<int32_t>((size_t)kPgM * pg::kMaxK, &D.pg_ids_hd);
      D.pg_rw_h = pinned_mapped<float>((size_t)kPgM * pg::kMaxK, &D.pg_rw_hd);
      D.pg_flag_h = pinned_mapped<uint32_t>(1, &D.pg_flag_d);
      *D.pg_flag_h = 0;
      CUDA_CHECK(cudaStreamSynchronize(st_));
      D.pg_gate = er::Gate{};
      fprintf(stderr, "[runtime] decode pregate on (top-%d before attention · CPU pool prefetch of predicted misses · owned items %s)\n", pregate_k_,
              store_.owned_mode() ? "on" : "OFF — store did not enable it");
    }
    const bool prof = profile_ && !(profile_every_ > 1 && step_ % profile_every_ != 0);  // same sample-step decision as pmark
    struct Ops {
      Runtime& r; DecodeOverlap& D; Work& w; const LayerWeights& L; std::vector<Seq*>& seqs; int M, Tb, slot; ForwardStats* s; bool prof;
      MoePend q{};
      bool trust = true;
      void prep(int) {}  // done above (prefilled by HOST_FAST at the previous layer, or just now by decode_host_prep)
      void front(int l) {
        struct Arm { DecodeOverlap& D; ~Arm() { D.er_arm = false; } } arm{D};  // even on an exception, never leak into moe_router_shared of other paths (verify, prefill)
        trust = D.er_gate.begin();
        D.mark(slot, DecodeOverlap::kPre, r.st_);
        D.er_arm = true;
        {
          struct PgArm { DecodeOverlap& D; ~PgArm() { D.pg_arm = false; } } pg_arm{D};  // P1 (even on an exception, never leak into decode_front of other paths)
          D.pg_arm = r.pregate_k_ > 0;
          r.run_graph(l, M, Tb, [&] { r.decode_front(l, M, Tb, r.opt_.cpu_for_misses); });
        }
        D.er_arm = false;
        D.mark(slot, DecodeOverlap::kFront, r.st_);
        CUDA_CHECK(cudaEventRecord(D.er_front, r.st_));
        if (r.pregate_k_ > 0) r.pregate_host(l, M);  // P1: prediction publish (front head) → prefetch request — the pool reads while attention runs
        const Config& c = r.model_.cfg();
        if (l == c.cand_source_layer && c.ratio(l) && c.is_index_source(l) && Tb > 0) r.have_candidates_ = true;  // same restoration as the base path (see the comment below)
      }
      void wait_route(int l) {
        const double t0 = prof ? now_ms() : 0.0;
        const int res = trust ? D.er_gate.wait(D.er_flag_h, [&] {
          const cudaError_t e = cudaEventQuery(D.er_front);
          if (e != cudaErrorNotReady) CUDA_CHECK(e);
          return e == cudaSuccess;
        }) : (int)er::kMissing;
        if (res != er::kOk) {  // fallback (see Gate comment): re-copy the routing after the front ends (same values as the routing half of route_to_host) and resync the sequence number
          CUDA_CHECK(cudaEventSynchronize(D.er_front));
          const int kk = L.n_act ? L.n_act : r.model_.cfg().n_act;
          k::route_to_host(w.ids.as<int32_t>(), w.rw.as<float>(), nullptr, nullptr, M * kk, 0, w.route_ids_d, w.rw_d, nullptr, nullptr, r.st_);
          CUDA_CHECK(cudaStreamSynchronize(r.st_));
          D.er_gate.resync(D.er_flag_h);
          if (D.er_fix++ < 4) fprintf(stderr, "[early-route] ⚠️layer %d: post %s — routing re-copied after the front graph (absorbed)\n", l,
                                      !trust ? "untrusted (previous launch not closed)" : res == er::kResync ? "ahead" : "missing");
        }
        if (prof) { t_dsync_prof.ms += now_ms() - t0; ++t_dsync_prof.n; }
        D.mark(slot, DecodeOverlap::kWake, r.st_);
        const Config& c = r.model_.cfg();
        r.pmark(L.engram ? "segA.engram" : c.is_index_source(l) ? "segA.index" : c.is_kv_source(l) ? "segA.kvsrc" : c.ratio(l) ? "segA.comp" : "segA.win");
      }
      void launch(int l) {
        q = MoePend{}; q.jobs = &w.cpu_jobs; q.slot = slot; q.defer_unpack = true; q.er_tbl = true;  // T2 er_tbl: table H2D on a stream unrelated to the front
        r.moe_decode_experts(L, l, M, s, q, /*start_cpu=*/false, true, true);
        ++D.er_layers;
        if (r.pregate_k_ > 0) {  // P1 comparison: this layer's actual CPU job experts (deduplicated) vs the prefetch handed over
          int ce[pg::kMaxPf]; int nc = 0;
          for (const auto& jb : w.cpu_jobs) {
            if (jb.layer != l) continue;
            bool dup = false;
            for (int i = 0; i < nc; ++i) dup |= ce[i] == jb.e;
            if (!dup && nc < pg::kMaxPf) ce[nc++] = jb.e;
          }
          D.pg_acc.add_layer(D.pg_pf, D.pg_l == l ? D.pg_npf : 0, ce, nc);
        }
        const cudaError_t e = cudaEventQuery(D.er_front);  // instrumentation: were GPU experts issued before the front ended (= no GPU idle after the front)
        if (e == cudaErrorNotReady) ++D.er_early; else CUDA_CHECK(e);
      }
      void wait_front(int) {  // activations arrived, layer mapped row tables released
        decode_event_wait(D.er_front, prof);
        if (D.pg_resync) { D.pg_gate.resync(D.pg_flag_h); D.pg_resync = false; }  // P1: untrusted launch — after the front ends that publish is visible
      }
      void start_cpu(int) {
        if (r.def_pending_) r.deferred_flush();  // before unpacking: the deferred jobs of the previous layer read the activation table this overwrites
        if (q.unpack_pending) {  // same formula and rows as moe_decode_experts' unpack_row (markers D.need) — tools/test_early_route_cpu.py compares the text
          const int dim = r.model_.cfg().dim;
          const float* lut = e4m3_lut();
          static const bool unpack2 = cpu::unpack2_enabled();
          auto unpack_row = [&](int m) {
      if (unpack2) { cpu::unpack_act_row(w.xq_h + (size_t)m * dim, w.xs_h + (size_t)m * (dim / 32), dim, w.a_f_h + (size_t)m * dim, w.a_s_h + (size_t)m * (dim / 32)); return; }
      for (int d = 0; d < dim; ++d) w.a_f_h[(size_t)m * dim + d] = lut[w.xq_h[(size_t)m * dim + d]];
      for (int b = 0; b < dim / 32; ++b) w.a_s_h[(size_t)m * (dim / 32) + b] = e8m0_to_f32(w.xs_h[(size_t)m * (dim / 32) + b]);
          };
          for (int m = 0; m < q.unpack_M; ++m) if (D.need[(size_t)m]) unpack_row(m);
          q.unpack_pending = false;
        }
        r.moe_decode_start_cpu(q);
      }
      void prep_next(int l) {  // O1 B HOST_FAST: after front (l) has ended (wait_front) — same precondition as the base path
        if (r.host_fast_ && l + 1 < r.model_.n_loaded_layers()) { D.pre_Tmax = r.decode_host_prep(seqs, l + 1, M, s); D.pre_l = l + 1; }
      }
      void finish(int l) {
        r.moe_decode_finish(q, s);
        if (r.dec_defer_ && !r.djobs_.empty()) {  // expert deferral: the immediate batch is done — start the deferred one and go on
          k::copy_f32(r.post_def_.as<float>(), w.post_f.as<float>(), (size_t)M * r.model_.cfg().hc, r.st_);  // post_f of this layer (the next front overwrites it)
          r.store_.start_jobs(r.djobs_, false);
          r.def_pending_ = true; r.def_M_ = M;
          ++r.def_layers_; r.def_rows_total_ += r.def_rows_n_;
          if (r.profile_ && r.def_layers_ % 4000 == 0)  // diagnostics only with HIVE_PROFILE (a line every ~100 decode steps otherwise)
            fprintf(stderr, "[defer] layers %ld · deferred rows %ld (%.2f per layer) · skipped rows %ld\n", r.def_layers_, r.def_rows_total_,
                    (double)r.def_rows_total_ / r.def_layers_, r.skip_rows_total_);
        }
        r.pmark("moe");
        if (prof && l == r.model_.n_loaded_layers() - 1) decode_sync_report(M);  // after counting the two waits per layer (publish, end of front)
        if (prof && r.pregate_k_ > 0 && l == r.model_.n_loaded_layers() - 1) {  // P1 totals (every sample step — same cadence as [early-route])
          const pg::Acc& a = D.pg_acc;
          const auto ps = r.store_.pf_stats();
          fprintf(stderr, "[pregate M=%d] layers %ld · posted %ld late %ld untrusted %ld · prefetch experts %ld · cpu experts %ld · covered %ld (recall %.2f · precision %.2f) · "
                          "pool req %ld experts %ld read %.1f MiB · finished %ld aborted %ld\n", M, a.layers, a.posted, a.late, a.untrusted, a.pred, a.actual, a.covered,
                  a.recall(), a.precision(), ps.req, ps.experts, ps.kib / 1024.0, ps.finished, ps.aborted);
        }
      }
    } ops{*this, D, w, L, seqs, M, Tb, slot, stats, prof};
    er::run_layer(l, /*prepped=*/true, ops);
    return;
  }
  // 2) graph A
  D.mark(slot, DecodeOverlap::kPre, st_);
  run_graph(l, M, Tb, [&] { decode_front(l, M, Tb, opt_.cpu_for_misses); });
  D.mark(slot, DecodeOverlap::kFront, st_);
  // Host metadata must be restored on graph replay, not only while capturing.
  // Graph replay and have_candidates_: the body (attention_decode_dev) runs with argument Tb — the Tmax of the indexer block `if (Tmax > 0)` is exactly this
  //   Tb, and its cand_src branch writes the candidate table and sets have_candidates_. In graph mode Tb is the bucket max(1024, next_pow2(Tmax)), so the
  //   captured graph produces candidates even when the real Tmax = 0. Conditioning on Tmax > 0 would leave the flag unset on that replay, and the
  //   HIVE_CHECK("candidates missing") of a candidate-consuming layer not yet captured (eager) would depend on the replay/eager combination. Use the body's
  //   condition (Tb > 0) (in eager mode Tb == Tmax, so behaviour is identical).
  if (l == c.cand_source_layer && c.ratio(l) && c.is_index_source(l) && Tb > 0) have_candidates_ = true;
  const bool prof_step = profile_ && !(profile_every_ > 1 && step_ % profile_every_ != 0);  // same sample-step decision as pmark
  decode_stream_wait(st_, prof_step);  // D1: default = plain cudaStreamSynchronize(st_); HIVE_DECODE_BLOCKING_SYNC=1 = blocking wait
  D.mark(slot, DecodeOverlap::kWake, st_);
  if (prof_step && l == model_.n_loaded_layers() - 1) decode_sync_report(M);
  pmark(L.engram ? "segA.engram" : c.is_index_source(l) ? "segA.index" : c.is_kv_source(l) ? "segA.kvsrc" : c.ratio(l) ? "segA.comp" : "segA.win");
  if (!opt_.dump_dir.empty()) {
    dump_host("route_ids_L" + std::to_string(l) + ".i32", w.route_ids_h, (size_t)M * (L.n_act ? L.n_act : c.n_act) * 4);
    dump_host("route_w_L" + std::to_string(l) + ".f32", w.rw_h, (size_t)M * (L.n_act ? L.n_act : c.n_act) * 4);
  }
  // 3) experts (eager — they depend on which experts are resident, so they cannot be in a graph). O1: launch + finish = the same calls and order as the base moe_decode_experts.
  MoePend p;
  p.jobs = &w.cpu_jobs;
  p.slot = slot;
  moe_decode_experts(L, l, M, stats, p, true, true, true);
  // O1 B HOST_FAST: the next layer's host tables while GPU/CPU experts run (before the wait) — front graph (l) has finished per the sync above and the next front is not issued yet
  if (host_fast_ && l + 1 < model_.n_loaded_layers()) { D.pre_Tmax = decode_host_prep(seqs, l + 1, M, stats); D.pre_l = l + 1; }
  moe_decode_finish(p, stats);
  pmark("moe");
}

// ---- O1 C HIVE_DECODE_UBATCH ------------------------------------------------------------------------------------------------------
namespace {
template <class T> inline void row_shift(T*& p, ptrdiff_t rows, size_t stride) {  // move a row-based buffer pointer by rows rows (negative = back). nullptr stays
  if (p) p = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(p) + (uintptr_t)(rows * (ptrdiff_t)stride));
}
constexpr int kUbGraphKey = 1 << 16;  // half graph key = row count + (half+1)·2^16 (never collides with plain keys M < 2^16 — each half captured different pointers)
}  // namespace

// Map half h's row range [r0, r0 + half rows) into Work — shift by r0 rows the pointers of row-based buffers that live across layers and host syncs
//   (residual, hc coefficients, indices, candidates, expert inputs/outputs, mapped row tables, CPU job area, expert table area), and swap batch-level host
//   state (engram hashes, per-row topk, candidate flag) for the half's. on=false restores exactly (the inverse shift, swap twice). Scratch used only within
//   one stream (xn, q, kv, y, yq, eout, attention partial sums …) is not moved — half launches are all in st_ order and never overlap. The expert table and
//   CPU areas get R = rows·k entries per half (total M·k ≤ allocated capacity — constructor formula).
void Runtime::ub_view(int h, int M, bool on) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  DecodeOverlap& D = *dov_;
  const ptrdiff_t r = on ? dov::half_row0(M, h) : -dov::half_row0(M, h);
  const size_t hc = c.hc, dim = c.dim, kk = c.n_act, cols = c.n_hash_cols(), hd = c.engram_head_dim;
  // inter-layer state
  row_shift(w.h.p, r, hc * dim * 2); row_shift(w.pre_mix.p, r, hc * 4);
  row_shift(w.pre_f.p, r, hc * 4); row_shift(w.post_f.p, r, hc * 4); row_shift(w.comb_f.p, r, hc * hc * 4);
  row_shift(w.idx.p, r, (size_t)(c.window + c.index_topk) * 4); row_shift(w.cand.p, r, (size_t)c.cand_topk_blocks * 4);
  row_shift(w.xq.p, r, dim); row_shift(w.xs.p, r, dim / 32); row_shift(w.acc.p, r, dim * 4);
  if (mtp_on_) row_shift(w.mh.p, r, (size_t)ntgt_ * dim * 2);
  // mapped pinned row tables (host/device pairs)
  row_shift(w.route_ids_h, r, kk * 4); row_shift(w.route_ids_d, r, kk * 4); row_shift(w.rw_h, r, kk * 4); row_shift(w.rw_d, r, kk * 4);
  row_shift(w.xq_h, r, dim); row_shift(w.xq_hd, r, dim); row_shift(w.xs_h, r, dim / 32); row_shift(w.xs_hd, r, dim / 32);
  row_shift(w.pos_h, r, 4); row_shift(w.pos_d, r, 4); row_shift(w.gpos_h, r, 4); row_shift(w.gpos_d, r, 4);
  row_shift(w.visible_h, r, 4); row_shift(w.visible_d, r, 4); row_shift(w.is_image_h, r, 1); row_shift(w.is_image_d, r, 1);
  row_shift(w.ids_h, r, 4); row_shift(w.emb_h, r, dim * 2); row_shift(w.emb_d, r, dim * 2);
  row_shift(w.eg_v_h, r, cols * hd); row_shift(w.eg_v_d, r, cols * hd); row_shift(w.eg_s_h, r, cols * hd / 32); row_shift(w.eg_s_d, r, cols * hd / 32);
  row_shift(w.kvrows_h, r, sizeof(k::KvRow)); row_shift(w.kvrows_d, r, sizeof(k::KvRow));
  row_shift(w.kptrs_h, r, sizeof(void*)); row_shift(w.kptrs_d, r, sizeof(void*)); row_shift(w.trows_h, r, 4); row_shift(w.trows_d, r, 4);
  row_shift(w.dsti_h, r, 4); row_shift(w.dsti_d, r, 4); row_shift(w.ringp_h, r, sizeof(void*)); row_shift(w.ringp_d, r, sizeof(void*));
  row_shift(w.dstp_h, r, sizeof(void*)); row_shift(w.dstp_d, r, sizeof(void*)); row_shift(w.dstk_h, r, sizeof(void*)); row_shift(w.dstk_d, r, sizeof(void*));
  row_shift(w.skv_h, r, sizeof(void*)); row_shift(w.skv_d, r, sizeof(void*)); row_shift(w.ssc_h, r, sizeof(void*)); row_shift(w.ssc_d, r, sizeof(void*));
  // expert host areas (k entries per row per half): CPU activations, outputs, scratch, row numbers; expert table (pinned source + device copy)
  row_shift(w.a_f_h, r, dim * 4); row_shift(w.a_s_h, r, dim / 32 * 4);
  row_shift(w.cpu_rows_h, r, kk * 4); row_shift(w.cpu_rows_d, r, kk * 4); row_shift(w.cpu_out_h, r, kk * dim * 4); row_shift(w.cpu_out_d, r, kk * dim * 4);
  row_shift(w.scratch_h, r, kk * ExpertStore::job_scratch_floats(c.moe_inter) * 4);
  row_shift(w.tbl_h, r, kk * (sizeof(k::GroupDesc) + 8)); row_shift(w.tbl_dev.p, r, kk * (sizeof(k::GroupDesc) + 8));
  // batch-level host state
  std::swap(seq_hashes_, D.hashes[h]); std::swap(shared_topk_rows_, D.topk_rows[h]); std::swap(have_candidates_, D.have_cand[h]);
  dec_row0_ = on ? dov::half_row0(M, h) : 0;
  D.view = on ? h : -1;
}

// All layers as a half pipeline (order = dov::run_ubatch — see the comment at the top of decode_overlap.h). The head (head graph) is run by the caller with the full M.
void Runtime::decode_layers_ubatch(std::vector<Seq*>& seqs, int M, ForwardStats* stats) {
  const Config& c = model_.cfg();
  DecodeOverlap& D = *dov_;
  const int nL = model_.n_loaded_layers();
  const bool prof_step = profile_ && !(profile_every_ > 1 && step_ % profile_every_ != 0);  // same sample-step decision as pmark
  std::vector<Seq*> hs[2];
  int Mh[2];
  for (int h = 0; h < 2; h++) {
    Mh[h] = dov::half_rows(M, h);
    const int r0 = dov::half_row0(M, h);
    hs[h].assign(seqs.begin() + r0, seqs.begin() + r0 + Mh[h]);
    D.topk_rows[h].assign((size_t)Mh[h], 0); D.hashes[h].clear(); D.have_cand[h] = false; D.Tb[h] = 0;
    D.jobs[h].clear(); D.pend[h] = MoePend{}; D.pend[h].jobs = &D.jobs[h];
    if (!D.front_done[h]) CUDA_CHECK(cudaEventCreateWithFlags(&D.front_done[h], cudaEventDisableTiming | (decode_blocking_sync() ? cudaEventBlockingSync : 0)));
  }
  if (stats) { stats->row_hit.resize(M); stats->row_cpu.resize(M); stats->row_dma.resize(M); }  // the resize(M) of the base moe_decode_experts — half launches do not shrink it
  struct Guard {  // even on an exception restore Work and host state (so DevBuf never frees shifted pointers) and wait for running CPU batches (job tables must stay alive)
    Runtime* r; int M; int ex;
    ~Guard() {
      if (std::uncaught_exceptions() > ex) r->store_.wait_jobs();
      if (r->dov_->view >= 0) r->ub_view(r->dov_->view, M, false);
      r->ub_active_ = false; r->dec_row0_ = 0;
    }
  } guard{this, M, std::uncaught_exceptions()};
  ub_active_ = true;
  struct Ops {
    Runtime& r; DecodeOverlap& D; const Config& c; std::vector<Seq*>* hs; const int* Mh; int M; ForwardStats* stats; bool prof;
    void view(int h) { if (D.view == h) return; if (D.view >= 0) r.ub_view(D.view, M, false); r.ub_view(h, M, true); }
    void prep(int h, int l) {
      view(h);
      const int Tmax = r.decode_host_prep(hs[h], l, Mh[h], stats);
      D.Tb[h] = c.ratio(l) && c.is_index_source(l) ? (r.graphs_ ? std::min(r.w_->Tcap, std::max(1024, next_pow2(Tmax))) : Tmax) : 0;
    }
    void front(int h, int l) {
      view(h);
      const int slot = D.slot(l, h), Tb = D.Tb[h], m = Mh[h];
      D.mark(slot, DecodeOverlap::kPre, r.st_);
      r.run_graph(l, m + (h + 1) * kUbGraphKey, Tb, [&] { r.decode_front(l, m, Tb, r.opt_.cpu_for_misses); });
      D.mark(slot, DecodeOverlap::kFront, r.st_);
      CUDA_CHECK(cudaEventRecord(D.front_done[h], r.st_));
      if (l == c.cand_source_layer && c.ratio(l) && c.is_index_source(l) && Tb > 0) r.have_candidates_ = true;  // flag of the half view (same condition as decode_layer)
    }
    void wait_front(int h, int l) {
      decode_event_wait(D.front_done[h], prof);
      D.mark(D.slot(l, h), DecodeOverlap::kWake, r.st_);
      const LayerWeights& L = r.model_.layer(l);
      r.pmark(L.engram ? "segA.engram" : c.is_index_source(l) ? "segA.index" : c.is_kv_source(l) ? "segA.kvsrc" : c.ratio(l) ? "segA.comp" : "segA.win");
    }
    void launch(int h, int l, bool go) {
      view(h);
      MoePend& p = D.pend[h];
      p = MoePend{}; p.jobs = &D.jobs[h]; p.slot = D.slot(l, h);
      r.moe_decode_experts(r.model_.layer(l), l, Mh[h], stats, p, go, /*pump=*/h == 1, /*cw_ok=*/h == 0);
    }
    bool cpu_idle(int h, int) const { return D.jobs[h].empty(); }
    void start_cpu(int h, int) { r.moe_decode_start_cpu(D.pend[h]); }
    void wait_cpu(int h, int) { r.moe_decode_wait(D.pend[h], stats); }
    void accum(int h, int) { r.moe_decode_accum(D.pend[h]); r.pmark("moe"); }
  } ops{*this, D, c, hs, Mh, M, stats, prof_step};
  dov::run_ubatch(nL, ops);
  if (D.view >= 0) ub_view(D.view, M, false);
  if (prof_step) decode_sync_report(M);
}

// One [decode-host] line (O1 A — HIVE_PROFILE sample steps, after the st_ sync at the end of forward_batch: all events have completed). Items (ms, step sums):
//   sync   = end of front → first record after the host woke up (spin/blocking wait latency + record submission)          [GPU clock]
//   prep   = wake-up → host work before the expert-table H2D (classification, observation, staging issue, CPU prep, pool start, promotion slices) [host clock]
//   launch = table H2D + GPU expert launch calls                                                                       [host clock]
//   cpu W vs gpu V = sum of CPU pool intervals (start → completion stamped by the pool) vs sum of GPU expert intervals (before table H2D → end of GPU experts)
//   cpu-bound layers = layers with a CPU share where GPU experts finished first and st_ waited for CPU results (gap > 5 µs — only a diagnostic threshold, no effect on behaviour) / measured layers
//   tail   = end of GPU experts → launch of CPU result accumulation (time st_ waited for the CPU)                       [GPU clock]
//   next   = end of layer → right before the next layer's front launch (next layer host tables + launch calls; the head for the last layer) [GPU clock]
//   front  = front graph (including launch latency); post = work HOST_FAST moved after the launch; hprep = host sum of per-layer host tables (engram, attention row tables)
//   gpu-idle = sync + (wake-up → before table H2D) + tail + next — time st_ sat empty because of this path (side-stream copies run separately) / of = first front → before the head launch
void Runtime::dh_report(int M) {
  DecodeOverlap& D = *dov_;
  if (!D.on) return;
  ds_report(M);  // C1 [decode-split] (only when on — same sample step)
  if (D.er_layers > 0) {  // S1 [early-route] (when on; totals since the last line): ahead = the host issued GPU experts before the front ended (no GPU idle after the front in that layer)
    const er::Gate& g = D.er_gate;
    fprintf(stderr, "[early-route M=%d] layers %ld · host ahead of front end %ld · table on own stream %ld · absorbed %ld (resync %ld · missing %ld · untrusted %ld)\n", M,
            D.er_layers, D.er_early, D.er_tbl_n, D.er_fix, g.n_resync, g.n_missing, g.n_untrusted);
    D.er_layers = D.er_early = D.er_tbl_n = 0;
  }
  D.on = false;
  using O = DecodeOverlap;
  const int nL = D.nL, head = 2 * nL;
  double sync = 0, gap = 0, gpu = 0, tail = 0, next = 0, front = 0, prep = 0, launch = 0, post = 0, cpu = 0;
  int layers = 0, cpu_bound = 0, cpu_layers = 0;
  for (int l = 0; l < nL; ++l)
    for (int h = 0; h < 2; ++h) {
      const int s = 2 * l + h;
      if (!D.has(s, O::kExp)) continue;
      ++layers;
      sync += D.el(s, O::kFront, s, O::kWake); front += D.el(s, O::kPre, s, O::kFront);
      gap += D.el(s, O::kWake, s, O::kExp); gpu += D.el(s, O::kExp, s, O::kGEnd);
      if (D.jobs_on[s] && D.has(s, O::kAcc)) { const float t = D.el(s, O::kGEnd, s, O::kAcc); tail += t; ++cpu_layers; if (t > 0.005f) ++cpu_bound; }
      next += D.el(s, O::kEnd, l + 1 < nL ? s + 2 : head, O::kPre);
      prep += D.h_prep[s]; launch += D.h_launch[s]; post += D.h_post[s]; cpu += D.cpu_ms[s];
    }
  if (layers == 0) return;  // steps that did not take the per-layer path (e.g. the G1 step graph)
  const double span = D.el(0, O::kPre, head, O::kPre);
  fprintf(stderr, "[decode-host M=%d%s] sync %.2f · prep %.2f · launch %.2f · cpu %.2f vs gpu %.2f (cpu-bound layers %d/%d) · tail %.2f ms · next %.2f · front %.2f · "
                  "post %.2f · hprep %.2f · gpu-idle %.2f of %.2f ms (cpu layers %d)\n",
          M, D.ub ? " ubatch" : vdec_ ? " verify" : "", sync, prep, launch, cpu, gpu, cpu_bound, layers, tail, next, front, post, D.h_hprep, sync + gap + tail + next, span, cpu_layers);  // R1 verify rows (verify_decode) are marked " verify"
}
// ---- O1 C END

// ---- C1 HIVE_DECODE_SPLIT=balance: collecting samples, the [decode-split] line ------------------------------------------------------------------
void Runtime::ds_consume() {
  if (!dsplit_ || !dsplit_->pending) return;
  DecodeSplit& S = *dsplit_;
  if (S.open) { S.pending = S.open = false; return; }  // sample that never reached accumulation after launch (exception recovery path) — dropped (the next layer arms a new one)
  dsplit::Obs& o = S.obs;
  for (cudaEvent_t e : {S.r1, S.g1}) if (cudaEventQuery(e) != cudaSuccess) return;
  if (o.n_str > 0 && cudaEventQuery(S.c1) != cudaSuccess) return;
  if (S.has_cpu && cudaEventQuery(S.acc) != cudaSuccess) return;
  float x = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&x, S.g0, S.r1)); o.res_ms = x;
  CUDA_CHECK(cudaEventElapsedTime(&x, S.g0, S.g1)); o.gpu_ms = x;
  if (o.n_str > 0) { CUDA_CHECK(cudaEventElapsedTime(&x, S.g0, S.c1)); o.cp_ms = x; }
  if (S.has_cpu) { CUDA_CHECK(cudaEventElapsedTime(&x, S.g0, S.acc)); o.end_ms = x; }
  if (!S.has_cpu) o.cpu_ms = -1;
  dsplit::observe(S.m[S.cls], o);
  S.win.add(S.dec, S.base_n, o);
  if (S.armed_prof) S.prof.add(S.dec, S.base_n, o);
  S.pending = false;
}

// [decode-split M=m] (HIVE_PROFILE sample steps — at the head of dh_report, after the st_ sync, so the last layer's sample has finished too). Items (ms):
//   step = this step's samples (on sample steps every layer with misses — the miss-layer partial sums of the same intervals as [decode-host] cpu/gpu/tail):
//     layers (model = chosen by the model, cold = empty table → base rule, unsure = failed validation → base rule), dma = sum of chosen DMA counts (base = what the base rule would have chosen)
//     cpu/gpu/end pred vs act = predicted vs actual for the chosen split (CPU pool interval, GPU expert interval g0→g1, layer end g0→accumulation launch); tail = Σ max(0, end − GPU)
//   win = every sample since the last line: mean |pred − actual| per layer (cpu, gpu, end); model state (this M class): err (validation error EMA), n (validation samples), skew, a few table cells
void Runtime::ds_report(int M) {
  if (!dsplit_) return;
  ds_consume();
  DecodeSplit& S = *dsplit_;
  const dsplit::Acc& a = S.prof;
  const dsplit::Acc& w = S.win;
  const dsplit::Model& m = S.m[dsplit::cls_of(M)];
  auto t = [](const dsplit::Tab& tb, int n) { return n < dsplit::kN && tb.has[n] ? tb.v[n] : -1.0; };
  const double wn = std::max(1, w.n_pred);
  fprintf(stderr, "[decode-split M=%d] step: layers %d (model %d · cold %d · unsure %d) · dma %d (base %d) · cpu pred %.2f act %.2f · gpu pred %.2f act %.2f · "
                  "end pred %.2f act %.2f · tail %.2f ms | win %d layers: |err|/layer cpu %.3f gpu %.3f end %.3f ms · model err_c %.2f err_g %.2f n %d/%d skew %.3f · "
                  "tab cpu 1:%.2f 2:%.2f 3:%.2f cp 1:%.2f 2:%.2f 4:%.2f dg 1:%.2f 2:%.2f\n",
          M, a.layers, a.why[dsplit::kModel], a.why[dsplit::kCold], a.why[dsplit::kUnsure], a.dma, a.base, a.pc, a.ac, a.pg, a.ag, a.pe, a.ae, a.tail, w.layers,
          w.abs_c / wn, w.abs_g / wn, w.abs_e / wn, m.err_c, m.err_g, m.n_val_c, m.n_val_g, m.has_skew ? m.skew : 0.0, t(m.cpu, 1), t(m.cpu, 2), t(m.cpu, 3), t(m.cp, 1),
          t(m.cp, 2), t(m.cp, 4), t(m.dg, 1), t(m.dg, 2));
  S.prof = dsplit::Acc{};
  S.win = dsplit::Acc{};
}

// ---------------------------------------------------------------------------------------------------------------------
// Decoder tail preparation: build compressor and indexer keys from layer l's input (all M rows), then move only the last tail rows to the front.
void Runtime::prepare_decoder_tail(Seq& seq, int l, int M, int64_t start_pos, int tail) {
  const Config& c = model_.cfg();
  const LayerWeights& L = model_.layer(l);
  Work& w = *w_;
  const int dim = c.dim, hc = c.hc, mix_hc = (2 + hc) * hc;
  // this layer's attention input xn (all rows) — same as the front of layer_forward
  if (env_on("HIVE_HC_CUBLAS")) {  // for comparison (alternative path) — same choice as hc_attn_pre
    k::hc_flatten_f32(w.h.as<bf16>(), M, hc * dim, w.hflat.as<float>(), st_);
    blas_->gemm_f32(w.hflat.as<float>(), L.hc_attn_fn.as<float>(), w.mixes.as<float>(), M, mix_hc, hc * dim);
    k::hc_row_rsqrt(w.h.as<bf16>(), M, hc * dim, c.norm_eps, w.rsq.as<float>(), st_);
  } else {
    k::hc_mix_rows(w.h.as<bf16>(), L.hc_attn_fn.as<float>(), M, hc * dim, mix_hc, c.norm_eps, w.mixes.as<float>(), w.rsq.as<float>(), st_);  // P5
  }
  k::hc_split_sinkhorn(w.mixes.as<float>(), w.rsq.as<float>(), L.hc_attn_scale.as<float>(), L.hc_attn_base.as<float>(), M, hc,
                       c.sinkhorn_iters, c.hc_eps, w.pre_a.as<float>(), w.post_a.as<float>(), w.comb_a.as<float>(), st_);
  k::hc_pre(w.h.as<bf16>(), w.pre_mix.as<float>(), M, hc, dim, w.x.as<bf16>(), st_);
  k::rmsnorm(w.x.as<bf16>(), L.attn_norm.as<bf16>(), c.norm_eps, M, dim, w.xn.as<bf16>(), st_);
  compress_source(seq, L, l, M, start_pos);
  if (tail <= 0) return;  // C2: earlier sub-chunks of a tile only produce compressed KV and indexer keys (tail rows come from the last sub-chunk)
  HIVE_CHECK(tail <= M, "decoder tail longer than rows");
  // row move: h and pre_mix on the device (via a temporary buffer), pos and is_image on the host (mapped)
  const int r0 = M - tail;
  CUDA_CHECK(cudaMemcpyAsync(w.hflat.p, w.h.as<bf16>() + (size_t)r0 * hc * dim, (size_t)tail * hc * dim * 2, cudaMemcpyDeviceToDevice, st_));
  CUDA_CHECK(cudaMemcpyAsync(w.h.p, w.hflat.p, (size_t)tail * hc * dim * 2, cudaMemcpyDeviceToDevice, st_));
  CUDA_CHECK(cudaMemcpyAsync(w.mixes.p, w.pre_mix.as<float>() + (size_t)r0 * hc, (size_t)tail * hc * 4, cudaMemcpyDeviceToDevice, st_));
  CUDA_CHECK(cudaMemcpyAsync(w.pre_mix.p, w.mixes.p, (size_t)tail * hc * 4, cudaMemcpyDeviceToDevice, st_));
  CUDA_CHECK(cudaStreamSynchronize(st_));  // kernels must have finished reading pos/is_image before the host arrays are moved below
  for (int i = 0; i < tail; ++i) { w.pos_h[i] = w.pos_h[r0 + i]; w.is_image_h[i] = w.is_image_h[r0 + i]; }
}

// Compressor and indexer keys of a kv-source layer → caches (all M rows). In decoder-tail mode this is called with all rows before the rows are reduced.
void Runtime::compress_source(Seq& seq, const LayerWeights& L, int l, int M, int64_t start_pos) {
  const Config& c = model_.cfg();
  const AttnWeights& A = L.attn;
  Work& w = *w_;
  const int dim = c.dim, D = c.head_dim, rd = c.rd;
  const int ratio = c.ratio(l);
  const float2* freqs = model_.rope_compress();
  const int64_t end = start_pos + M;
  const int64_t g0 = start_pos / ratio;
  const int G = (int)(end / ratio - g0);
    if (ratio > 1) {
      if (verify_) {  // for rollback recomputation: this layer's attention input rows (after normalization)
        for (size_t i = 0; i < v_src2_.size(); ++i)
          if (v_src2_[i] == l) CUDA_CHECK(cudaMemcpyAsync(w.vxn[i].p, w.xn.p, (size_t)M * dim * 2, cudaMemcpyDeviceToDevice, st_));
      }
      k::bf16_to_f32(w.xn.as<bf16>(), M * dim, w.xf.as<float>(), st_);
      blas_->gemm_f32(w.xf.as<float>(), A.comp_wkv.as<float>(), w.ckv.as<float>(), M, D, dim);
      blas_->gemm_f32(w.xf.as<float>(), A.comp_wgate.as<float>(), w.cscore.as<float>(), M, D, dim);
      k::compressor_pool(w.ckv.as<float>(), w.cscore.as<float>(), M, D, ratio, start_pos, seq.comp_state_kv[l].as<float>(),
                         seq.comp_state_score[l].as<float>(), w.cout.as<float>(), st_);
      if (G > 0) {
        k::f32_to_bf16(w.cout.as<float>(), G * D, w.latent.as<bf16>(), st_);
        k::rmsnorm(w.latent.as<bf16>(), A.comp_norm.as<bf16>(), c.norm_eps, G, D, w.latent.as<bf16>(), st_);
      }
    } else {
      blas_->gemm_bf16(w.xn.as<bf16>(), A.comp_wkv.as<bf16>(), w.latent.as<bf16>(), M, D, dim);
      k::rmsnorm(w.latent.as<bf16>(), A.comp_norm.as<bf16>(), c.norm_eps, M, D, w.latent.as<bf16>(), st_);
    }
    shared_comp_kv_ = seq.comp_cache[l].as<uint8_t>();
    // group positions (position of each group's first token)
    for (int g = 0; g < G; ++g) w.gpos_h[g] = (int32_t)((g0 + g) * ratio);
    (void)0;
    if (c.is_index_source(l)) {
      // indexer keys (from the latent before RoPE) → cache
      if (G > 0) {
        blas_->gemm_bf16(w.latent.as<bf16>(), A.idx_wk.as<bf16>(), w.ik.as<bf16>(), G, c.index_head_dim, D);
        k::rmsnorm(w.ik.as<bf16>(), A.idx_k_norm.as<bf16>(), c.norm_eps, G, c.index_head_dim, w.ik.as<bf16>(), st_);
        k::rope_last(w.ik.as<bf16>(), G, 1, c.index_head_dim, rd, freqs, w.gpos_d, false, st_);
        k::fp4_pack(w.ik.as<bf16>(), G, c.index_head_dim, 32, false, seq.idx_k_cache[l].as<uint8_t>() + (size_t)g0 * kvp::IDX_ROW, kvp::IDX_ROW, st_);
      }
      shared_index_k_ = seq.idx_k_cache[l].as<uint8_t>();
    }
    if (G > 0) {
      k::rope_last(w.latent.as<bf16>(), G, 1, D, rd, freqs, w.gpos_d, false, st_);
      k::fp4_pack(w.latent.as<bf16>(), G, D, 16, true, seq.comp_cache[l].as<uint8_t>() + (size_t)g0 * kvp::COMP_ROW, kvp::COMP_ROW, st_);
    }
  
}

void Runtime::attention(Seq& seq, const LayerWeights& L, int l, int M, int64_t start_pos, bool skip_compress) {
  const Config& c = model_.cfg();
  const AttnWeights& A = L.attn;
  Work& w = *w_;
  const int dim = c.dim, H = c.n_heads, D = c.head_dim, rd = c.rd, win = c.window;
  const int ratio = c.ratio(l);
  const float2* freqs = ratio ? model_.rope_compress() : model_.rope_window();
  // ⛔Measured defect: the fused kv kernel writes this chunk's rows' KV into the ring (pos % win) **before** attention. That is fine for batched decode
  //   (each row is a different sequence = a different ring), but here (one sequence, a chunk of M rows) when row m reads the oldest cell of its window, that
  //   cell has already been overwritten by the KV of a later row m' (> m) (position start+m'−win lies inside row m's window) → future leak. DSpark verify
  //   rows (2–6) "saw" the drafts and agreed, giving ~91 % acceptance even on random tokens; standalone greedy dropped closing brackets and the model wrote
  //   "I made a typo" (HumanEval+ 0.665). Short chunks of ≤ 8 tokens (short turns after the cache) leak the same way.
  //   The unfused path calls ring_write after attention. Fix: fusion is used for all M ≤ 8, but the kernel does not write the ring (ring=nullptr — same
  //   approach as mtp_attention) → ring_write **after** attention. Speculative verify (rows 2–6) takes the fused path (~22 launches) instead of the chain
  //   (~40). Fused and chained values are bit-identical (test_fused).
  const bool fused = fuse_ && M <= 8;
  if (fused) {  // fused q/kv chain (fused.cu)
    (k::decode_gemv2_on() ? k::gemv2_q_proj : k::fused_q_proj)(w.xn.as<bf16>(), dim, A.wq_a.w.as<uint8_t>(), A.wq_a.s.as<uint8_t>(), A.wq_a.N, A.q_norm.as<bf16>(), c.norm_eps, A.wq_b.w.as<uint8_t>(),
                    A.wq_b.s.as<uint8_t>(), H, D, rd, freqs, w.pos_d, M, w.qr.as<bf16>(), w.qrn.as<bf16>(), w.q.as<bf16>(), w.qrq.as<uint8_t>(),
                    w.qrs.as<uint8_t>(), w.counters.as<int>() + 0, st_);
    pmark("attn.q");
    k::fused_kv_proj(w.xn.as<bf16>(), dim, A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), D, A.kv_norm.as<bf16>(), c.norm_eps, rd, freqs, w.pos_d, M,
                     w.kv.as<bf16>(), nullptr, win, nullptr, 0, w.counters.as<int>() + 1, st_);
  } else {
  // q
  k::act_quant_fp8(w.xn.as<bf16>(), M, dim, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), st_);
  k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), A.wq_a.w.as<uint8_t>(), A.wq_a.s.as<uint8_t>(), false, M, A.wq_a.N, dim,
             w.qr.as<bf16>(), nullptr, st_);
  k::rmsnorm(w.qr.as<bf16>(), A.q_norm.as<bf16>(), c.norm_eps, M, c.q_lora_rank, w.qrn.as<bf16>(), st_);
  k::act_quant_fp8(w.qrn.as<bf16>(), M, c.q_lora_rank, w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), st_);
  k::gemm_bs(w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), A.wq_b.w.as<uint8_t>(), A.wq_b.s.as<uint8_t>(), false, M, A.wq_b.N,
             c.q_lora_rank, w.q.as<bf16>(), nullptr, st_);
  k::rope_last(w.q.as<bf16>(), M, H, D, rd, freqs, w.pos_d, false, st_);
  pmark("attn.q");
  // window kv
  k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), false, M, D, dim, w.kv.as<bf16>(),
             nullptr, st_);
  k::rmsnorm(w.kv.as<bf16>(), A.kv_norm.as<bf16>(), c.norm_eps, M, D, w.kv.as<bf16>(), st_);
  k::rope_last(w.kv.as<bf16>(), M, 1, D, rd, freqs, w.pos_d, false, st_);
  k::act_quant_fp8_roundtrip(w.kv.as<bf16>(), M, D, st_);
  }
  const int idx_stride = win + c.index_topk;
  k::window_idxs(M, win, start_pos, w.idx.as<int32_t>(), idx_stride, st_, win_min_pos_);  // P6: block keys before the replay range (normally 0)
  pmark("attn.kv");
  int topk_total = win;
  k::KvSources kvs{seq.ring[l].as<bf16>(), win, w.kv.as<bf16>(), M, nullptr, 0};
  if (ratio) {
    const int64_t end = start_pos + M;
    const int64_t g0 = start_pos / ratio;
    const int G = (int)(end / ratio - g0);  // number of groups completed by this call
    const int compress_len = (int)(end / ratio);
    if (c.is_kv_source(l) && !skip_compress) compress_source(seq, L, l, M, start_pos);
    HIVE_CHECK(shared_comp_kv_ != nullptr, "compressed kv source missing");
    if (c.is_index_source(l)) {
      if (compress_len > 0) indexer(seq, L, l, M, start_pos, G, g0);
      shared_topk_cols_ = std::min(c.index_topk, compress_len);
    } else {
      HIVE_CHECK(shared_topk_cols_ >= 0, "topk idxs missing");
      // the shared topk is already in w.idx's compressed columns (written by the index source layer). Only the window columns were rewritten for this layer, so this is fine.
    }
    topk_total = win + shared_topk_cols_;
    kvs.comp = shared_comp_kv_;
    kvs.comp_len = compress_len;
  }
  pmark("attn.compress");
  static const bool attn_v1 = env_on("HIVE_ATTN_V1"), attn_v2 = env_on("HIVE_ATTN_V2");
  if (attn_v1)
    k::sparse_attn(w.q.as<bf16>(), M, H, D, kvs, w.idx.as<int32_t>(), idx_stride, topk_total, A.attn_sink.as<float>(),
                   1.0f / sqrtf((float)D), w.o.as<bf16>(), st_);
  else if (M >= 16 && !attn_v2)  // P2: flash variant for prefill (small M uses v2 — one query per block, so below M<16 v2 fills the SMs better)
    k::sparse_attn_flash(w.q.as<bf16>(), M, H, D, kvs, w.idx.as<int32_t>(), idx_stride, topk_total, A.attn_sink.as<float>(),
                         1.0f / sqrtf((float)D), w.o.as<bf16>(), st_);
  else
    k::sparse_attn_tc(w.q.as<bf16>(), M, H, D, kvs, w.idx.as<int32_t>(), idx_stride, topk_total, A.attn_sink.as<float>(),
                      1.0f / sqrtf((float)D), w.attn_S.as<float>(), w.attn_mx.as<float>(), w.attn_sum.as<float>(), w.o.as<bf16>(), st_);
  k::rope_last(w.o.as<bf16>(), M, H, D, rd, freqs, w.pos_d, true, st_);
  pmark("attn.sparse");
  // wo_a: per group og[:, g, :] = o[:, g, :] · wo_a[g]ᵀ  (A row stride = HD, batch stride = sub; B batch stride = R·sub; C row stride = G·R)
  {  // wo_a: for each group g, og[:, g·R + r] = o[:, g·sub + k] · wo_a[g·R + r, k] — bf16 activations × fp8 original (no bf16 copy, same values)
    const int G = c.o_groups, sub = H * D / G, R = c.o_lora_rank;
    // Measured (speculative verify M=6): 8 tensor-core tile GEMM launches cost 2.1 ms per layer (attn.wo 85 ms / 40 layers) — rows ≤ 8 use the same grouped GEMV as decode
    if (M <= 8) (k::decode_gemv2_on() ? k::gemv2_bf16_fp8_grouped : k::gemv_bf16_fp8_grouped)(w.o.as<bf16>(), H * D, A.wo_a_q.w.as<uint8_t>(), A.wo_a_q.s.as<uint8_t>(), M, G, R, sub, w.og.as<bf16>(), st_);
    else if (w.woa.p && !env_on("HIVE_WOA_TC")) {
      // P3: per layer, unpack fp8 wo_a into a bf16 scratch (67 MB) and run a cuBLAS batched GEMM — the custom bf16×fp8 tensor-core kernel reached only 13 TFLOPS at M=2048.
      //   Values are the same (e4m3×2^k is exact in bf16); only accumulation order differs. VRAM cost is a single scratch (a full bf16 copy would be 2.7 GB).
      k::dequant_fp8_rows_to_bf16(A.wo_a_q.w.as<uint8_t>(), A.wo_a_q.s.as<uint8_t>(), G * R, sub, w.woa.as<bf16>(), st_);
      blas_->gemm_bf16_batched_ld(w.o.as<bf16>(), H * D, sub, w.woa.as<bf16>(), sub, (long)R * sub, w.og.as<bf16>(), G * R, R, M, R, sub, G);
    } else
    for (int g = 0; g < G; ++g)
      k::gemm_bf16a_fp8b_tc(w.o.as<bf16>() + (size_t)g * sub, H * D, A.wo_a_q.w.as<uint8_t>() + (size_t)g * R * sub,
                            A.wo_a_q.s.as<uint8_t>() + (size_t)g * R * (sub / 32), M, R, sub, w.og.as<bf16>() + (size_t)g * R, G * R, st_);
  }
  if (fused) {
    (k::decode_gemv2_on() ? k::gemv2_quantin : k::fused_gemv_quantin)(w.og.as<bf16>(), c.o_groups * c.o_lora_rank, A.wo_b.w.as<uint8_t>(), A.wo_b.s.as<uint8_t>(), M, dim, w.attn_out.as<bf16>(), st_);
    k::ring_write(w.kv.as<bf16>(), M, D, win, start_pos, seq.ring[l].as<bf16>(), st_);  // after attention — so chunk rows never see later rows' KV
  } else {
    k::act_quant_fp8(w.og.as<bf16>(), M, c.o_groups * c.o_lora_rank, w.ogq.as<uint8_t>(), w.ogs.as<uint8_t>(), st_);
    k::gemm_bs(w.ogq.as<uint8_t>(), w.ogs.as<uint8_t>(), A.wo_b.w.as<uint8_t>(), A.wo_b.s.as<uint8_t>(), false, M, dim,
               c.o_groups * c.o_lora_rank, w.attn_out.as<bf16>(), nullptr, st_);
    // ring update (this chunk's kv)
    k::ring_write(w.kv.as<bf16>(), M, D, win, start_pos, seq.ring[l].as<bf16>(), st_);
  }
  pmark("attn.wo");
  if (!opt_.dump_dir.empty()) {
    dump("attn_idx_L" + std::to_string(l) + ".i32", w.idx.p, (size_t)M * idx_stride * 4);
    dump("attn_out_L" + std::to_string(l) + ".bf16", w.attn_out.p, (size_t)M * dim * 2);
  }
}

// Indexer: M queries (sub-chunk) × T = compress_len keys. Results go into w.idx's compressed columns (win.. win+topk) as global indices (win+M offset).
void Runtime::indexer(Seq& seq, const LayerWeights& L, int l, int M, int64_t start_pos, int G, int64_t g0) {
  const Config& c = model_.cfg();
  const AttnWeights& A = L.attn;
  Work& w = *w_;
  const int ratio = c.ratio(l), Hi = c.index_n_heads, Di = c.index_head_dim, rd = c.rd, win = c.window;
  const int64_t end = start_pos + M;
  const int T = (int)(end / ratio);
  const int idx_stride = win + c.index_topk;
  const float2* freqs = model_.rope_compress();
  HIVE_CHECK(T <= w.Tcap, "index T cap");
  // q: wq_b(qr) → rope → fp4
  k::gemm_bs(w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), A.idx_wq_b.w.as<uint8_t>(), A.idx_wq_b.s.as<uint8_t>(), false, M, Hi * Di,
             c.q_lora_rank, w.iq.as<bf16>(), nullptr, st_);
  k::rope_last(w.iq.as<bf16>(), M, Hi, Di, rd, freqs, w.pos_d, false, st_);
  // D-2: with HIVE_IDX_TC=1 scores use e2m1×e2m1 tensor cores — queries are packed into the same IDX_ROW as keys from the bf16 before the roundtrip
  //   (fp4_pack uses the same scale and rounding as fp4_roundtrip_kernel → unpacked value = roundtrip value). Off by default (not yet compared on GPU).
  static const bool idx_tc = getenv("HIVE_IDX_TC") && atoi(getenv("HIVE_IDX_TC")) != 0;
  if (idx_tc) {
    HIVE_CHECK(Di == kvp::IDX_D, "idx tc Di");
    k::fp4_pack(w.iq.as<bf16>(), M * Hi, Di, 32, false, w.iqp.as<uint8_t>(), kvp::IDX_ROW, st_);
  }
  k::fp4_quant_roundtrip(w.iq.as<bf16>(), M, Hi * Di, 32, false, st_);
  // weights = bf16(bf16(proj(x)) · (Di^-0.5 · Hi^-0.5))
  blas_->gemm_bf16(w.xn.as<bf16>(), A.idx_weights_proj.as<bf16>(), w.iw.as<bf16>(), M, Hi, c.dim);
  k::scale_bf16(w.iw.as<bf16>(), M * Hi, (1.0f / sqrtf((float)Di)) * (1.0f / sqrtf((float)Hi)), st_);
  for (int m = 0; m < M; ++m) w.visible_h[m] = (int32_t)((start_pos + m + 1) / ratio);
  (void)0;
  const bool cand_src = (l == c.cand_source_layer);
  const bool uses_cand = (c.cand_source_layer >= 0 && c.cand_source_layer < l);
  const int bs = c.cand_block, nblocks = (T + bs - 1) / bs;
  const int topk = std::min(c.index_topk, T);
  const int CB = c.cand_topk_blocks;  // row stride of the candidate list
  // HIVE_IDX_F32=1: fp32 scores (no per-head bf16 rounding, written directly to iscore_f) — the production-kernel (DeepGEMM) style. Exclusive with the tensor-core variant
  static const bool idx_f32 = getenv("HIVE_IDX_F32") && atoi(getenv("HIVE_IDX_F32")) != 0;
  HIVE_CHECK(!(idx_f32 && idx_tc), "HIVE_IDX_F32 and HIVE_IDX_TC are exclusive");
  // L3 (opt-in HIVE_IDX_MSUB_ACTUAL): sub-chunk row count from the actual T (idx_msub_for — same scratch capacity, never fewer rows). Dump mode keeps the base row count (dump shape unchanged).
  const int Msub = idx_msub_actual_ && opt_.dump_dir.empty()
                       ? idx_msub_for(w.Msub, w.Tcap, w.nblocks_cap, std::max(c.index_topk, c.cand_topk_blocks), M, T, bs, CB, c.index_topk)
                       : w.Msub;
  for (int m0 = 0; m0 < M; m0 += Msub) {
    const int Ms = std::min(Msub, M - m0);
    if (uses_cand) {
      // P1: Reindex layers score only inside layer 20's candidate blocks (min(cand_topk_blocks, ⌈T/8⌉) per row) — full-T scores, keep masks and full-T top-k all disappear
      // The candidate block count is recomputed here with the same formula as layer 20 (passing it as state could drift after graph replay or an exception) — these are ratio-1 layers, so T is the same
      const int kbc = std::min(CB, nblocks);
      HIVE_CHECK(have_candidates_ && kbc > 0, "candidates missing");
      const int ncand = kbc * bs;
      if (idx_f32)
        k::indexer_scores_cand_f32(w.iq.as<bf16>() + (size_t)m0 * Hi * Di, shared_index_k_, nullptr, nullptr, w.iw.as<bf16>() + (size_t)m0 * Hi, Ms, Hi, Di,
                                   w.cand.as<int32_t>() + (size_t)m0 * CB, CB, kbc, bs, w.visible_d + m0, w.iscore_f.as<float>(), st_);
      else if (idx_tc)  // candidate-pool tensor-core variant (HIVE_IDX_TC=1 — packed queries iqp are built above)
        k::indexer_scores_cand_tc(w.iqp.as<uint8_t>() + (size_t)m0 * Hi * kvp::IDX_ROW, shared_index_k_, nullptr, nullptr, w.iw.as<bf16>() + (size_t)m0 * Hi, Ms, Hi,
                                  w.cand.as<int32_t>() + (size_t)m0 * CB, CB, kbc, bs, w.visible_d + m0, w.iscore.as<bf16>(), st_);
      else
        k::indexer_scores_cand(w.iq.as<bf16>() + (size_t)m0 * Hi * Di, shared_index_k_, nullptr, nullptr, w.iw.as<bf16>() + (size_t)m0 * Hi, Ms, Hi, Di,
                               w.cand.as<int32_t>() + (size_t)m0 * CB, CB, kbc, bs, w.visible_d + m0, w.iscore.as<bf16>(), st_);
      if (!idx_f32) k::bf16_rows_to_f32(w.iscore.as<bf16>(), Ms * ncand, w.iscore_f.as<float>(), st_);
      k::topk_select_rows(w.iscore_f.as<float>(), Ms, ncand, topk, ncand, w.topk_pos.as<int32_t>(), topk, st_);
      k::cand_to_pos(w.topk_pos.as<int32_t>(), Ms, topk, w.cand.as<int32_t>() + (size_t)m0 * CB, CB, bs, st_);
      k::offset_idxs(w.topk_pos.as<int32_t>(), Ms, topk, w.visible_d + m0, win + M,
                     w.idx.as<int32_t>() + (size_t)m0 * idx_stride + win, idx_stride, st_);
      continue;
    }
    if (idx_f32)
      k::indexer_scores_f32(w.iq.as<bf16>() + (size_t)m0 * Hi * Di, shared_index_k_, w.iw.as<bf16>() + (size_t)m0 * Hi, Ms, Hi, Di, T,
                            w.visible_d + m0, w.iscore_f.as<float>(), st_);
    else if (idx_tc)
      k::indexer_scores_tc(w.iqp.as<uint8_t>() + (size_t)m0 * Hi * kvp::IDX_ROW, shared_index_k_, w.iw.as<bf16>() + (size_t)m0 * Hi, Ms, Hi, T,
                           w.visible_d + m0, w.iscore.as<bf16>(), st_);
    else
      k::indexer_scores(w.iq.as<bf16>() + (size_t)m0 * Hi * Di, shared_index_k_, w.iw.as<bf16>() + (size_t)m0 * Hi, Ms, Hi, Di, T,
                        w.visible_d + m0, w.iscore.as<bf16>(), st_);
    if (cand_src) {
      if (idx_f32) k::block_max_f32(w.iscore_f.as<float>(), Ms, T, bs, w.visible_d + m0, w.bmax.as<float>(), st_);
      else k::block_max(w.iscore.as<bf16>(), Ms, T, bs, w.visible_d + m0, w.bmax.as<float>(), st_);
      const int kb = std::min(c.cand_topk_blocks, nblocks);
      k::topk_select_rows(w.bmax.as<float>(), Ms, nblocks, kb, nblocks, w.topk_pos.as<int32_t>(), kb, st_);
      // keep the candidate block list (ascending, -1 when short) per row — blocks with bmax = -inf (positions not yet visible) become -inf at the scoring step, so they are not filtered separately
      CUDA_CHECK(cudaMemcpy2DAsync(w.cand.as<int32_t>() + (size_t)m0 * CB, (size_t)CB * 4, w.topk_pos.p, (size_t)kb * 4, (size_t)kb * 4, Ms,
                                   cudaMemcpyDeviceToDevice, st_));
      have_candidates_ = true;
    }
    if (!idx_f32) k::bf16_rows_to_f32(w.iscore.as<bf16>(), Ms * T, w.iscore_f.as<float>(), st_);
    k::topk_select_rows(w.iscore_f.as<float>(), Ms, T, topk, T, w.topk_pos.as<int32_t>(), topk, st_);
    k::offset_idxs(w.topk_pos.as<int32_t>(), Ms, topk, w.visible_d + m0, win + M,
                   w.idx.as<int32_t>() + (size_t)m0 * idx_stride + win, idx_stride, st_);
  }
  if (!opt_.dump_dir.empty() && !idx_f32) {  // (the fp32 variant differs from the golden format and is not dumped)
    if (uses_cand) {  // golden comparison tools expect the full-T shape (-inf outside candidates) — only when dumping, rebuild the first sub-batch in that shape
      const int Ms = std::min(M, w.Msub);
      k::indexer_scores(w.iq.as<bf16>(), shared_index_k_, w.iw.as<bf16>(), Ms, Hi, Di, T, w.visible_d, w.iscore.as<bf16>(), st_);
      k::mask_by_cand(w.iscore.as<bf16>(), Ms, T, bs, w.cand.as<int32_t>(), CB, std::min(CB, nblocks), st_);
    }
    dump("index_score_L" + std::to_string(l) + ".bf16", w.iscore.p, (size_t)std::min(M, w.Msub) * T * 2);
  }
  (void)G; (void)g0; (void)seq;
}

// ---------------------------------------------------------------------------------------------------------------------
void Runtime::engram_host(const LayerWeights& L, int M, const int32_t* ids_host, std::vector<Seq*>* seqs, Seq* seq) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const EngramWeights& E = *L.engram;
  const EngramTable* T = store_.engram(E.hash_index);
  HIVE_CHECK(T != nullptr && eh_ != nullptr, "engram table/hash missing");
  const int cols = c.n_hash_cols(), hd = c.engram_head_dim;
  // Hashes: history is extended only at the first engram layer (same tokens for every layer). Assumes hash_index 0 comes first.
  std::vector<int64_t> hashes;
  if (E.hash_index == 0) {
    if (seqs) {  // each row with its own sequence's history
      hashes.clear();
      for (int m = 0; m < M; ++m) {
        std::vector<int64_t> hm;
        eh_->compute(ids_host + m, w.is_image_h + m, 1, (*seqs)[m]->engram_history, hm);
        hashes.insert(hashes.end(), hm.begin(), hm.end());
      }
    } else {
      eh_->compute(ids_host, w.is_image_h, M, seq->engram_history, hashes);
    }
    seq_hashes_ = hashes;
  } else {
    hashes = seq_hashes_;
  }
  const int nL = eh_->n_layers();
  // Row lookup → mapped pinned (read directly by the kernel). L2 (opt-in HIVE_ENGRAM_PAR): split a prefill chunk (M rows × 24 columns × 256 B random reads)
  //   over the pool thread count — the CPU expert pool is idle at this point (the previous layer's moe ended with wait_jobs). Each row has its own destination, so bytes are identical. Off = the sequential loop (engram_gather threads 1).
  if (EngramSsd* ssd = store_.engram_ssd()) {
    // HIVE_ENGRAM_SSD: from the file with the same hashes and the same output layout (cache + read-ahead). The first engram layer (layer 1) starts reads of
    //   the later table's (layer 14) rows — overlapping layers 2–13 (if engram_ssd_hint at the forward head already started them, they are assumed cached and only protection is refreshed). Bookkeeping and waits run in parallel (the pool is idle — same reasoning as L2).
    if (E.hash_index == 0 && nL > 1) ssd->prefetch_hashes(hashes.data(), M, nL, cols, 1);
    const EngramSsd::GatherInfo g = ssd->gather(hashes.data(), M, nL, E.hash_index, cols, hd, T->rows, w.eg_v_h, w.eg_s_h, host_par_threads_);
    // where and how long we waited: large lookups (prefill chunks — rows ≥ 256) get one line each, the rest (decode) a summary line every 30 s
    if (M >= 256)
      fprintf(stderr, "[engram-ssd] layer %d rows %d: lookups %llu · cached %llu · in-flight %llu · read now %llu · uncached %llu · waited %.1f ms\n", L.id,
              M, (unsigned long long)g.rows, (unsigned long long)g.hit, (unsigned long long)g.inflight, (unsigned long long)g.sync, (unsigned long long)g.bypass,
              g.wait_ms);
    const double now = now_ms();
    if (essd_log_ms_ == 0) essd_log_ms_ = now;
    if (now - essd_log_ms_ >= 30000) {
      const double sec = (now - essd_log_ms_) / 1000.0;
      essd_log_ms_ = now;
      const EngramSsd::Stats st = ssd->stats(true);
      const double lk = std::max<double>(1, (double)st.rows);
      fprintf(stderr, "[engram-ssd] %.0fs: gathers %llu · lookups %llu (cached %.1f%% · in-flight %.1f%% · read now %.1f%% · uncached %.1f%%) · wait %.1f ms total "
              "(max %.2f ms) · prefetch %llu calls %llu rows (issued %llu · present %llu · dropped %llu) · reads %llu (%.0f/s · %.1f MB) · retries %llu · unreadable %llu\n",
              sec, (unsigned long long)st.gathers, (unsigned long long)st.rows, 100.0 * st.hit / lk, 100.0 * st.inflight / lk, 100.0 * st.sync / lk,
              100.0 * st.bypass / lk, st.wait_ms, st.wait_max_ms, (unsigned long long)st.pf_calls, (unsigned long long)st.pf_rows, (unsigned long long)st.pf_issued,
              (unsigned long long)st.pf_present, (unsigned long long)st.pf_dropped, (unsigned long long)st.reads, st.reads / std::max(sec, 1e-9), st.read_bytes / 1e6,
              (unsigned long long)st.retries, (unsigned long long)st.failed);
    }
  } else {
    engram_gather(hashes.data(), M, nL, E.hash_index, cols, hd, T->rows, T->vals.base, T->scales.base, w.eg_v_h, w.eg_s_h, engram_par_ ? host_par_threads_ : 1);
  }
  if (engram_digest_) {  // HIVE_ENGRAM_DIGEST: two logs of the same request run in RAM and SSD mode must match in line order and values (bit-identity check)
    const uint64_t dv = engram_digest(w.eg_v_h, (size_t)M * cols * hd), ds = engram_digest(w.eg_s_h, (size_t)M * cols * (hd / 32));
    fprintf(stderr, "[engram-digest] layer %d rows %d vals %016llx scales %016llx\n", L.id, M, (unsigned long long)dv, (unsigned long long)ds);
  }
}

void Runtime::engram_ssd_hint(const int32_t* ids, const int8_t* is_image, int M, Seq* const* row_seqs, Seq* seq) {
  EngramSsd* ssd = store_.engram_ssd();
  if (!ssd || !eh_ || M <= 0 || !ssd->opts().prefetch) return;
  // Row groups (runs of the same sequence) = (starting history length, up to max_ngram−1 history tail ids, tokens). When a sequence reappears after a gap, the
  //   tokens of its earlier run are appended (same history as engram_host's row-order compute — verify batches repeat the same sequence per row).
  struct Seg { int m0, n; int64_t start; std::vector<int64_t> tail; };
  const int K = eh_->max_ngram() - 1;
  std::vector<Seg> segs;
  std::vector<std::pair<Seq*, std::vector<int64_t>>> added;  // compressed ids already appended per sequence in this forward
  for (int m = 0; m < M;) {
    Seq* s = row_seqs ? row_seqs[m] : seq;
    int n = 1;
    while (m + n < M && (row_seqs ? row_seqs[m + n] : seq) == s) ++n;
    std::vector<int64_t>* add = nullptr;
    for (auto& a : added) if (a.first == s) add = &a.second;
    if (!add) { added.push_back({s, {}}); add = &added.back().second; }
    const std::vector<int64_t>& hist = s->engram_history;
    const int64_t start = (int64_t)hist.size() + (int64_t)add->size();
    Seg g{m, n, start, {}};
    for (int64_t q = std::max<int64_t>(0, start - K); q < start; ++q)
      g.tail.push_back(q < (int64_t)hist.size() ? hist[(size_t)q] : (*add)[(size_t)(q - (int64_t)hist.size())]);
    for (int i = 0; i < n; ++i) add->push_back(eh_->compressed(ids[m + i], is_image && is_image[m + i]));
    segs.push_back(std::move(g));
    m += n;
  }
  std::vector<int32_t> id(ids, ids + M);
  std::vector<int8_t> img(M, 0);
  if (is_image) for (int m = 0; m < M; ++m) img[m] = is_image[m];
  const EngramHash* eh = eh_;
  const int nL = eh->n_layers(), C = eh->n_cols();
  ssd->prefetch([eh, segs = std::move(segs), id = std::move(id), img = std::move(img), M, nL, C](std::vector<int64_t>& out) {
    out.assign((size_t)M * nL * C, 0);
    std::vector<int64_t> part;
    for (const Seg& g : segs) {
      eh->peek(id.data() + g.m0, img.data() + g.m0, g.n, g.tail.data(), (int)g.tail.size(), g.start, part);
      std::copy(part.begin(), part.end(), out.begin() + (size_t)g.m0 * nL * C);
    }
  }, M, nL, C, 0);
}

void Runtime::engram(Seq& seq, const LayerWeights& L, int M, const int32_t* ids_host) {
  engram_host(L, M, ids_host, nullptr, &seq);
  engram_dev(L, M);
}

void Runtime::engram_dev(const LayerWeights& L, int M) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const EngramWeights& E = *L.engram;
  const int cols = c.n_hash_cols(), hd = c.engram_head_dim;
  k::engram_dequant(w.eg_v_d, w.eg_s_d, M, cols, hd, w.eg_vals.as<bf16>(), st_);
  k::act_quant_fp8(w.eg_vals.as<bf16>(), M, cols * hd, w.eg_q.as<uint8_t>(), w.eg_sc.as<uint8_t>(), st_);
  k::gemm_bs(w.eg_q.as<uint8_t>(), w.eg_sc.as<uint8_t>(), E.wkv.w.as<uint8_t>(), E.wkv.s.as<uint8_t>(), false, M, E.wkv.N, cols * hd,
             w.eg_kv.as<bf16>(), nullptr, st_);
  k::engram_gate(w.eg_kv.as<bf16>(), E.qk.as<float>(), w.is_image_d, M, c.hc, c.dim, c.norm_eps, w.h.as<bf16>(), st_);
}

// ---------------------------------------------------------------------------------------------------------------------
// R2 HIVE_PREFILL_TAIL_SHORT: inside a prefill forward (in_prefill_), layers with fewer rows than prefill_threshold = decoder tail layers (P6 deployment recipe, 128 rows).
//   Otherwise these layers take the streaming path too (moe_experts — the HIVE_DMA_FRAC_PREFILL share of misses (e.g. 0.7) as whole-record H2D). With
//   128 rows × top-6 an expert gets 1–3 rows, so a record (18.8 MB, measured ~0.67 ms each — see the bm_prefetch comment) is moved to compute one or two rows.
//   This variant sends those layers down the path short prefills (chunks with M < threshold) already take (moe_decode_experts, kind kDmaShort, per-layer
//   DMA ≤ HIVE_SHORT_DMA_CAP, the rest as CPU multi-row jobs).
//   Math: same experts, same weights, same activation quantization (act_quant_fp8(xn), 32-block y quantization). The only possible difference is fp32
//   accumulation order — (a) GPU expert kernels (grouped GEMV/mx_grouped chain vs prefill grouped GEMM), (b) which misses go to the CPU (CPU and GPU
//   kernels accumulate in different orders), (c) expert output → acc accumulation order (accum_bf16_rows_seq/accum_f32_rows vs fixed-order accumulation). Off = unchanged.
//   Outside the conditions (verify, CPU misses off, rows ≥ threshold, not a prefill) the base path runs. The upper layers of forward_multi (H2/H3) call moe_experts_multi directly and never reach this.
bool Runtime::prefill_short_moe(int M) const {
  if (small_fwd_) return false;  // Q3 HIVE_PREFILL_SMALL chunks: layers with rows < threshold are full-row layers, not the tail — keep the streaming path (off = small_fwd_ always false)
  return sp::short_moe(spo_.tail_short, in_prefill_, verify_, batch_ != nullptr, opt_.cpu_for_misses, M, opt_.prefill_threshold);
}

void Runtime::moe(Seq& seq, const LayerWeights& L, int l, int M, ForwardStats* stats) {
  const bool short_tail = prefill_short_moe(M);  // R2 (off = false → the code below is the base formula)
  const bool prefill = (M >= opt_.prefill_threshold || in_prefill_) && !short_tail;
  moe_router_shared(L, M, !prefill && opt_.cpu_for_misses);
  pmark("moe.router+shared");
  if (pprof_) pprof_->router(st_);  // R2 HIVE_PREFILL_PROF (only when on — GPU event at the end of the router, time right before the sync)
  if (prefill || short_tail) CUDA_CHECK(cudaStreamSynchronize(st_));  // R2: the short tail path uses the same sync as prefill (keeps decode wait instrumentation and reports out of prefill)
  else {  // D1: the decode-policy path (single sequence, MTP verify) uses the same wait as decode_layer (default = plain cudaStreamSynchronize)
    const bool prof_step = profile_ && !(profile_every_ > 1 && step_ % profile_every_ != 0);
    decode_stream_wait(st_, prof_step);
    if (prof_step && l == model_.n_loaded_layers() - 1) decode_sync_report(M);
  }
  pmark("moe.sync");
  if (!opt_.dump_dir.empty()) {
    dump_host("route_ids_L" + std::to_string(l) + ".i32", w_->route_ids_h, (size_t)M * model_.cfg().n_act * 4);
    dump_host("route_w_L" + std::to_string(l) + ".f32", w_->rw_h, (size_t)M * model_.cfg().n_act * 4);
  }
  if (pprof_) pprof_->synced();
  if (prefill) moe_experts(seq, L, l, M, stats);
  else moe_decode_experts(L, l, M, stats);
  if (pprof_) pprof_->moe_done();
}

void Runtime::moe_router_shared(const LayerWeights& L, int M, bool copy_acts) {
  const Config& c = model_.cfg();
  const MoeWeights& F = L.ffn;
  Work& w = *w_;
  const int dim = c.dim, I = c.moe_inter, E = L.n_routed ? L.n_routed : c.n_routed, k = L.n_act ? L.n_act : c.n_act;
  const bool er_post = dov_ && dov_->er_arm && fuse_ && M <= 8;  // S1 (only while decode_layer arms it — off = the code below is unchanged)
  if (fuse_ && M <= 8) {
    // router (fp32 GEMV + top-k in one kernel), shared expert (w1‖w3+swiglu+quantization, w2+acc write — two kernels). xq/xs are produced by block 0 of the w13 kernel.
    if (cache_prior_ > 0.f && cp_layer_ >= 0 && cp_layer_ < c.n_layers && k::decode_router3_on() && E == c.n_routed) {
      const k::CachePriorArgs cpa{cache_mask_h_ + (size_t)cp_layer_ * c.n_routed, cache_prior_, prior_range_.as<float>() + cp_layer_, cache_prior_j_};
      k::attn3_router_cp(w.xn.as<bf16>(), dim, F.gate_w.as<float>(), M, E, F.gate_bias.as<float>(), F.gate_bias_vl.p ? F.gate_bias_vl.as<float>() : nullptr,
                         w.is_image_d, k, c.route_scale, w.scores.as<float>(), w.ids.as<int32_t>(), w.rw.as<float>(), w.counters.as<int>() + 2, cpa, st_);
    } else
    (k::decode_router3_on() ? k::attn3_router : k::decode_attn2_on() ? k::attn2_router : k::decode_gemv2_on() ? k::gemv2_router : k::fused_router)(w.xn.as<bf16>(), dim, F.gate_w.as<float>(), M, E, F.gate_bias.as<float>(), F.gate_bias_vl.p ? F.gate_bias_vl.as<float>() : nullptr,
                    w.is_image_d, k, c.route_scale, w.scores.as<float>(), w.ids.as<int32_t>(), w.rw.as<float>(), w.counters.as<int>() + 2, st_);
    pmark("moe.router");
    // S1 HIVE_DECODE_EARLY_ROUTE: move only the routing to the host now + publish a sequence number (no input overlap with the shared expert — both read only xn, neither reads the router output)
    if (er_post) k::er_route_post(w.ids.as<int32_t>(), w.rw.as<float>(), M * k, w.route_ids_d, w.rw_d, dov_->er_ctr_d, dov_->er_flag_d, st_);
    (k::decode_attn2_on() ? k::attn2_shared_experts : k::decode_gemv2_on() ? k::gemv2_shared_experts : k::fused_shared_experts)(w.xn.as<bf16>(), dim, F.sh_w1.w.as<uint8_t>(), F.sh_w1.s.as<uint8_t>(), F.sh_w3.w.as<uint8_t>(), F.sh_w3.s.as<uint8_t>(),
                            F.sh_w2.w.as<uint8_t>(), F.sh_w2.s.as<uint8_t>(), M, I, c.swiglu_limit, w.y.as<bf16>(), w.xq.as<uint8_t>(), w.xs.as<uint8_t>(),
                            w.acc.as<float>(), st_);
    pmark("moe.shared");
  } else {
  // router
  k::bf16_to_f32(w.xn.as<bf16>(), M * dim, w.xf.as<float>(), st_);
  blas_->gemm_f32(w.xf.as<float>(), F.gate_w.as<float>(), w.scores.as<float>(), M, E, dim);
  k::router_topk(w.scores.as<float>(), F.gate_bias.as<float>(), F.gate_bias_vl.p ? F.gate_bias_vl.as<float>() : nullptr,
                 w.is_image_d, M, E, k, c.route_scale, w.ids.as<int32_t>(), w.rw.as<float>(), st_);
  pmark("moe.router");
  k::act_quant_fp8(w.xn.as<bf16>(), M, dim, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), st_);
  CUDA_CHECK(cudaMemsetAsync(w.acc.p, 0, (size_t)M * dim * 4, st_));
  // shared expert
  k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), F.sh_w1.w.as<uint8_t>(), F.sh_w1.s.as<uint8_t>(), false, M, I, dim, w.gate.as<bf16>(),
             nullptr, st_);
  k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), F.sh_w3.w.as<uint8_t>(), F.sh_w3.s.as<uint8_t>(), false, M, I, dim, w.up.as<bf16>(),
             nullptr, st_);
  k::swiglu_route(w.gate.as<bf16>(), w.up.as<bf16>(), nullptr, M, I, c.swiglu_limit, w.y.as<bf16>(), st_);
  k::act_quant_fp8(w.y.as<bf16>(), M, I, w.yq.as<uint8_t>(), w.ys.as<uint8_t>(), st_);
  k::gemm_bs(w.yq.as<uint8_t>(), w.ys.as<uint8_t>(), F.sh_w2.w.as<uint8_t>(), F.sh_w2.s.as<uint8_t>(), false, M, dim, I, w.eout.as<bf16>(),
             nullptr, st_);
  k::accum_bf16_rows(w.eout.as<bf16>(), nullptr, M, dim, w.acc.as<float>(), st_);
  pmark("moe.shared");
  }
  // routing result (+ activations for CPU experts) → mapped pinned host buffers (written directly by the kernel). The caller synchronizes. S1: routing was already moved above — only activations (no launch if none)
  if (er_post) { if (copy_acts) k::route_to_host(w.ids.as<int32_t>(), w.rw.as<float>(), w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), 0, M * dim, w.route_ids_d, w.rw_d, w.xq_hd, w.xs_hd, st_); }
  else
  k::route_to_host(w.ids.as<int32_t>(), w.rw.as<float>(), w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), M * k, copy_acts ? M * dim : 0,
                   w.route_ids_d, w.rw_d, w.xq_hd, w.xs_hd, st_);
  pmark("moe.d2h");
}

void Runtime::predict_eval(const LayerWeights& L, int l, int M) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  PredEval& P = *pe_;
  constexpr int kMaxM = 8, kMaxK2 = 16;
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(st_, &cs));
  if (cs != cudaStreamCaptureStatusNone || ub_active_ || M > kMaxM || l >= c.n_layers) { P.pend_l = -1; return; }
  const int k = L.n_act ? L.n_act : c.n_act;
  if (!P.st) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&P.st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&P.ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&P.done, cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc((void**)&P.scores, (size_t)kMaxM * 512 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&P.rw, (size_t)kMaxM * kMaxK2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&P.ids, (size_t)kMaxM * kMaxK2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void**)&P.counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(P.counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&P.ids_h, (size_t)kMaxM * kMaxK2 * sizeof(int32_t)));
    fprintf(stderr, "[runtime] decode predict eval on (layer l+1's router on layer l's FFN input — measurement only)\n");
  }
  // 1) compare the prediction made at layer l-1 with this layer's real routing (already on the host)
  if (P.pend_l == l && P.pend_M == M) {
    CUDA_CHECK(cudaEventSynchronize(P.done));
    const int K2 = P.pend_k;
    for (int m = 0; m < M; ++m) {
      const int32_t* pr = P.ids_h + (size_t)m * K2;
      for (int j = 0; j < k; ++j) {
        const int e = w.route_ids_h[m * k + j];
        int at = -1;
        for (int q = 0; q < K2; ++q) if (pr[q] == e) { at = q; break; }
        const bool res = store_.slot_of(l, e) >= 0;
        ++P.real; P.hit_k += at >= 0 && at < k; P.hit_2k += at >= 0;
        if (!res) { ++P.miss; P.miss_k += at >= 0 && at < k; P.miss_2k += at >= 0; }
      }
    }
    // prefetch precision: the first 1/2/4 predicted experts that are not resident now, taken rank by rank over the rows (rank 0 of every row, then rank 1, …)
    {
      int cand[4], nc = 0;
      for (int q = 0; q < K2 && nc < 4; ++q)
        for (int m = 0; m < M && nc < 4; ++m) {
          const int e = P.ids_h[(size_t)m * K2 + q];
          if (store_.slot_of(l, e) >= 0) continue;
          bool dup = false; for (int i = 0; i < nc; ++i) dup |= cand[i] == e;
          if (!dup) cand[nc++] = e;
        }
      static const int lim[3] = {1, 2, 4};
      for (int t = 0; t < 3; ++t)
        for (int i = 0; i < std::min(nc, lim[t]); ++i) {
          ++P.pf_n[t];
          for (int r = 0; r < M * k; ++r) if (w.route_ids_h[r] == cand[i]) { ++P.pf_used[t]; break; }
        }
    }
    if (++P.layers % 2000 == 0)
      fprintf(stderr, "[predict-eval] layers %lld · recall top-%d %.1f %% · top-%d %.1f %% · non-resident %lld (%.1f %% of picks): predicted top-%d %.1f %% · top-%d %.1f %%\n",
              P.layers, k, 100.0 * P.hit_k / P.real, 2 * k, 100.0 * P.hit_2k / P.real, P.miss, 100.0 * P.miss / P.real, k,
              P.miss ? 100.0 * P.miss_k / P.miss : 0.0, 2 * k, P.miss ? 100.0 * P.miss_2k / P.miss : 0.0);
    if (P.layers % 2000 == 0)
      fprintf(stderr, "[predict-eval] prefetch precision (non-resident, first 1/2/4): %.1f %% (%lld) · %.1f %% (%lld) · %.1f %% (%lld) per layer %.2f/%.2f/%.2f\n",
              P.pf_n[0] ? 100.0 * P.pf_used[0] / P.pf_n[0] : 0.0, P.pf_n[0], P.pf_n[1] ? 100.0 * P.pf_used[1] / P.pf_n[1] : 0.0, P.pf_n[1],
              P.pf_n[2] ? 100.0 * P.pf_used[2] / P.pf_n[2] : 0.0, P.pf_n[2], (double)P.pf_n[0] / P.layers, (double)P.pf_n[1] / P.layers, (double)P.pf_n[2] / P.layers);
  }
  P.pend_l = -1;
  // 2) predict layer l+1 from this layer's FFN input (w.xn — in stream order, the input the router just used)
  if (l + 1 >= c.n_layers) return;
  const LayerWeights& N = model_.layer(l + 1);
  const MoeWeights& F = N.ffn;
  const int E = N.n_routed ? N.n_routed : c.n_routed, K2 = std::min(kMaxK2, 2 * (N.n_act ? N.n_act : c.n_act));
  if (!F.gate_w.p || E > 512) return;
  CUDA_CHECK(cudaEventRecord(P.ready, st_));
  CUDA_CHECK(cudaStreamWaitEvent(P.st, P.ready, 0));
  k::fused_router(w.xn.as<bf16>(), c.dim, F.gate_w.as<float>(), M, E, F.gate_bias.as<float>(), F.gate_bias_vl.p ? F.gate_bias_vl.as<float>() : nullptr,
                  w.is_image_d, K2, c.route_scale, P.scores, P.ids, P.rw, P.counter, P.st);
  CUDA_CHECK(cudaMemcpyAsync(P.ids_h, P.ids, (size_t)M * K2 * sizeof(int32_t), cudaMemcpyDeviceToHost, P.st));
  CUDA_CHECK(cudaEventRecord(P.done, P.st));
  P.pend_l = l + 1; P.pend_M = M; P.pend_k = K2;
}

// Decode experts: resident (group kernels right away), DMA share (group kernels after place), CPU (multi-row jobs). Launches per layer ≈ 2 (gather) + 4 (resident) + 4 (DMA) + 1 (CPU sum).
// O1: steps 1–3 of moe_decode_experts (classification → CPU start → DMA → table H2D → GPU experts). With start_cpu=false the CPU batch is only built and
//   moe_decode_start_cpu starts it (UBATCH half B — the pool runs one batch at a time). pump=false skips promotion slices (promo_pump) (UBATCH does it once
//   per layer — in half B); cw_ok=false skips E4 demand-copy instrumentation (one slot per layer, so half A only). Default path = (true, true, true) → the same calls and order as without overlap.
void Runtime::moe_decode_experts(const LayerWeights& L, int l, int M, ForwardStats* stats, MoePend& p, bool start_cpu, bool pump, bool cw_ok) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  DecodeOverlap& D = *dov_;
  const int dim = c.dim, I = c.moe_inter, E = L.n_routed ? L.n_routed : c.n_routed, k = L.n_act ? L.n_act : c.n_act;
  const ExpertLayout& lay = store_.layout();
  const bool fast = host_fast_;  // O1 B
  const int grow0 = dec_row0_;   // O1 C: stats rows and observe rows = half's first row + local row (default 0)
  const double th0 = now_ms();
  // group (m,j) per expert — counting sort (no allocation of E vectors). rows_by_e[off[e]..off[e+1]) = m*k+j
  const int R = M * k;
  if (pe_) predict_eval(L, l, M);  // F2 (measurement only)
  if (stats) {
    if (!ub_active_) { stats->row_hit.resize(M); stats->row_cpu.resize(M); stats->row_dma.resize(M); }
    else for (auto* v : {&stats->row_hit, &stats->row_cpu, &stats->row_dma}) if ((int)v->size() < grow0 + M) v->resize((size_t)grow0 + M);  // UBATCH: only grow (never erase the other half's rows)
  }
  HIVE_CHECK(E <= 512 && M < std::max(opt_.prefill_threshold, opt_.max_batch + 1), "decode grouping bounds");
  int cnt[512 + 1] = {0};
  const float score_weight = opt_.phase_score && score_prefill_ ? 1.f / std::max(1, M) : 1.f;
  std::vector<int>& rows_by_e = w.rows_by_e;  // [M·k], reused
  rows_by_e.resize(R);
  // Usage observation (scores, D4 policy rows) — default = before grouping; O1 B HOST_FAST = after the GPU launch (same order and arguments). O1 C UBATCH: row number = half's first row + local row —
  //   wg views the routing table by global row (w.route_ids_h − grow0·k), so i runs over global numbers [grow0·k, grow0·k + R) while the values read are this half's rows (default grow0 = 0 → the plain call).
  struct { const int32_t* route_ids_h; } wg{w.route_ids_h - (ptrdiff_t)grow0 * k};
  auto observe_rows = [&] {
    if (D.obs_off) return;  // F1: the caller (forward_multi up_experts) already observed with the same formula as the single call (off = unchanged)
    auto& w = wg;
    for (int i = grow0 * k; i < grow0 * k + R; ++i) store_.observe(l, w.route_ids_h[i], score_weight, i / k);  // D4 row i/k (used only by HIVE_CACHE_POLICY=seq)
  };
  int used_e[512];  // O1 B: used experts in ascending order (fast only)
  int n_used = 0;
  if (!fast) {
    for (int i = 0; i < R; ++i) ++cnt[w.route_ids_h[i] + 1];
    observe_rows();
    if (!D.obs_off) xtrace_layer(l, M, k, w.route_ids_h);  // F1 (off = unchanged)
    for (int e = 0; e < E; ++e) cnt[e + 1] += cnt[e];
    int fill[512];
    std::memcpy(fill, cnt, sizeof(int) * E);
    for (int i = 0; i < R; ++i) rows_by_e[fill[w.route_ids_h[i]]++] = i;
  } else {  // same cnt[e], cnt[e+1] (for used e) and rows_by_e — observe and xtrace happen after the launch (step 5 below)
    int fill[512];
    if ((int)D.tmp.size() < R) D.tmp.resize(R);
    n_used = dov::group_routes_sparse(w.route_ids_h, R, cnt, rows_by_e.data(), used_e, D.tmp.data(), fill);
  }
  auto n_of = [&](int e) { return cnt[e + 1] - cnt[e]; };
  // 1) classification: resident → groups [0, ng_hit); DMA share → groups [ng_hit, ng_hit+ng_str); the rest CPU
  int hit_e[512], str_e[512], cpu_e[512];
  int n_hit_e = 0, n_str_e = 0, n_cpu_e = 0;
  // DMA share only with 3 or more misses: staging DMA costs ~0.7 ms per expert (PCIe) while the CPU takes 0.33/0.6/0.8 ms for 1/2/3 jobs (measured) —
  // with 1–2 misses the CPU is faster; from 3 on, moving one to DMA wins with max(2 CPU jobs, DMA).
  int n_miss = 0;
  if (!fast) { for (int e = 0; e < E; ++e) if (n_of(e) > 0 && store_.slot_of(l, e) < 0) ++n_miss; }
  else for (int u = 0; u < n_used; ++u) if (store_.slot_of(l, used_e[u]) < 0) ++n_miss;
  // Adaptive split: update frac from the previous layer's CPU wait vs DMA group completion time (each sample against its own kind's limit — dma_frac_consume)
  const int kind = sp::decode_path_short_kind(M, opt_.max_batch, in_prefill_) ? kDmaShort : kDmaDecode;  // F1: prefill layers (TAIL_SHORT, MULTI_TAIL_SHORT) use the short-prefill state regardless of row count (hive/short_prefill.h)
  // R1 HIVE_MTP_VERIFY_ROWIND (env_on, off by default; only on the verify decode path vdec_): make row m's expert sum **independent of other rows**.
  //   Measured: the new verify's future-independence check fails with CPU misses on at cut 2/4/5 with rel 0.09–0.12 (with routing flips), and is exact
  //   (rel 0) with --no-cpu --promote 0 — what changes with the row set is the device of missed experts (DMA share = round(frac·layer misses), sorted by row
  //   count descending) and the CPU result accumulation order.
  //   On: all misses go to the CPU (device = residency only — constant within a step), CPU job order = ascending expert id, CPU result accumulation = fixed
  //   job-row order (accum_f32_rows_ordered — no atomic adds), and this layer leaves DMA-share adaptation (dma_frac_relax) and C1 model samples alone (so
  //   verify does not alter ordinary decode state). The math is the same (same experts, same CPU kernels) — only the share of misses computed on the CPU instead of the GPU (DMA) and the accumulation order change.
  const bool vrowind = vdec_ && verify_rowind_ && opt_.cpu_for_misses;
  dma_frac_consume();
  const float frac = dma_frac_fixed_ >= 0.f ? dma_frac_fixed_ : dma_frac_[kind];
  int gpu_share_left = 0;
  // Per-layer DMA cap: decode is fixed at 8 (HIVE_STAGING is for prefill pre-copies); short prefill (kind 1) uses H4 HIVE_SHORT_DMA_CAP (unset = 8, same as decode)
  const int dma_cap = kind == kDmaShort ? short_dma_cap_ : 8;
  if (!opt_.cpu_for_misses) gpu_share_left = E;
  else if (n_miss >= 3) gpu_share_left = std::min(std::min(dma_cap, store_.staging_slots()), std::max(opt_.decode_gpu_share, (int)(frac * n_miss + 0.5f)));
  // missed experts by row count, descending — the ones with most rows (slowest on the CPU) go to DMA first
  int miss_e[512];
  int n_miss_e = 0;
  auto classify = [&](int e) {
    if (stats) stats->n_routed += n_of(e);
    if (store_.slot_of(l, e) >= 0) hit_e[n_hit_e++] = e;
    else { miss_e[n_miss_e++] = e; if (opt_.promote_misses > 0) step_miss_.push_back(l * c.n_routed + e);  /* ExpertStore key = l·E_ (trunk 384) + e — draft layers (E 128) follow the same rule */ }
  };
  if (!fast) { for (int e = 0; e < E; ++e) if (n_of(e) != 0) classify(e); }
  else for (int u = 0; u < n_used; ++u) classify(used_e[u]);  // ascending — same order as the full E scan above
  std::sort(miss_e, miss_e + n_miss_e, [&](int a, int b) { return n_of(a) > n_of(b); });
  // C1 HIVE_DECODE_SPLIT=balance (decode_split_on — when off this whole block is skipped = base rule): choose the DMA count with the cost model (hive/decode_split.h).
  //   Decode kind (M ≤ max_batch) only; UBATCH halves use the base rule (the CPU overlaps the other half, so the interval model does not fit). Miss order
  //   (rows descending → front ones go to DMA) is unchanged — only how many go to DMA changes. On cold start or failed validation dec.n_dma = the base rule's
  //   count (gpu_share_left clipped to the miss count — same result as the loop below). Cap = min(8, staging slots): reusing a staging slot within a layer
  //   would overwrite an earlier copy of the same layer (same as dma_cap 8 and decode_handshake.h kMaxDma).
  bool ds_model = false, ds_sample = false;
  DecodeSplit* S = nullptr;
  if (dsplit_ && opt_.cpu_for_misses && kind == kDmaDecode && !ub_active_ && n_miss_e > 0 && !vrowind) {  // R1 ROWIND layers stay outside the model
    S = dsplit_.get();
    ds_consume();
    dsplit::Input& in = S->in;
    in.rows.resize((size_t)n_miss_e);
    for (int i = 0; i < n_miss_e; ++i) in.rows[(size_t)i] = n_of(miss_e[i]);
    in.ng_hit = 0;
    for (int ei = 0; ei < n_hit_e; ++ei) in.ng_hit += (n_of(hit_e[ei]) + dsplit::kGroupRows - 1) / dsplit::kGroupRows;
    in.cap = std::min(8, store_.staging_slots());
    in.base_n = std::min(gpu_share_left, n_miss_e);
    in.max_rows = ExpertStore::kMaxRows;
    const int cls = dsplit::cls_of(M);
    const dsplit::Decision dec = dsplit::decide(S->m[cls], in);
    gpu_share_left = dec.n_dma;
    ds_model = dec.why == dsplit::kModel;
    ds_sample = !S->pending && (D.on || S->tick++ % DecodeSplit::kSampleEvery == 0);
    p.ds = true; p.ds_cls = cls;
    if (ds_sample) {
      S->make_events();
      S->pending = true; S->open = true; S->armed_prof = D.on; S->has_cpu = false;
      S->cls = cls; S->base_n = in.base_n; S->dec = dec;
      S->obs = dsplit::Obs{};
      S->obs.ng_hit = in.ng_hit; S->obs.has_pred = dec.has_pred; S->obs.pred = dec.pred;
      p.ds_sample = true;
    }
  }
  // B2 (see the comment at the top of hive/batch_miss.h): B only on batched decode layers (forward_batch, M ≥ 2, decode kind, not a UBATCH half, C1 model off, CPU misses on) —
  //   otherwise nullptr and everything below is unchanged (str_si is not read, copies happen at the usual place, the plain classification loop).
  if (vrowind) { gpu_share_left = 0; std::sort(miss_e, miss_e + n_miss_e); }  // R1 ROWIND: every miss on the CPU, in expert-id order (not row-count order)
  p.rowind = vrowind;
  BatchMiss* B = bmiss_ && batch_ && M >= 2 && kind == kDmaDecode && !ub_active_ && !dsplit_ && opt_.cpu_for_misses ? bmiss_.get() : nullptr;
  const bool sh = B && B->stage_hit;          // HIVE_DECODE_STAGE_HIT (or PREFETCH)
  int str_si[512];                             // sh: existing staging slot of str_e[i] (-1 = new copy); not read when off
  int stg_e[512], stg_si[512], n_stg = 0;      // sh: misses intact in the ring with finished copies — to the GPU after the resident groups, without copy or CPU
  int n_str_dma = 0;                           // DMA share count (copies + reused in-flight copies) — without B = n_str_e (frac sample, rollback decision)
  if (!sh) {
  for (int i = 0; i < n_miss_e; ++i) {
    if (gpu_share_left > 0) { --gpu_share_left; str_e[n_str_e++] = miss_e[i]; }
    else cpu_e[n_cpu_e++] = miss_e[i];
  }
  n_str_dma = n_str_e;
  } else {
    // ring lookup (slots this layer can still take after the decision = demand DMA cap + pre-copy cap — within that range a looked-up slot is never reused)
    const int SS = store_.staging_slots(), margin = bmiss::kLayerAlloc + B->prefetch;  // (S is the C1 sample pointer above)
    bmiss::MissIn mi[512];
    uint8_t wh[512];
    int msi[512];
    for (int i = 0; i < n_miss_e; ++i) {
      const int e = miss_e[i];
      const int si = B->track.find((size_t)l * c.n_routed + e, stage_next_, SS, margin);
      msi[i] = si;
      mi[i] = {e, n_of(e), si < 0 ? -1 : cudaEventQuery(stage_copied_[si]) == cudaSuccess ? 1 : 0};
    }
    // same DMA-share formula as gpu_share_left above (only the miss count becomes "not in the ring + in flight")
    auto share = [&](int n) { return n >= 3 ? std::min(std::min(dma_cap, SS), std::max(opt_.decode_gpu_share, (int)(frac * n + 0.5f))) : 0; };
    bmiss::assign(mi, n_miss_e, share, wh);
    // prediction measurement (regardless of pre-copy): of the previous step's CPU share P for this layer, how many actually missed now (routed while non-resident)
    if (const std::vector<bmiss::Ent>* P = B->pred.at(l)) {
      B->acc.pred += (long)P->size();
      for (const bmiss::Ent& x : *P) if (x.e < E && n_of(x.e) > 0 && store_.slot_of(l, x.e) < 0) ++B->acc.pred_used;
    }
    int cpu_rows[512];
    n_miss = 0;
    for (int i = 0; i < n_miss_e; ++i) {
      const int e = mi[i].e;
      if (wh[i] != bmiss::kStaged) ++n_miss;
      if (wh[i] == bmiss::kCpu) { cpu_rows[n_cpu_e] = mi[i].rows; cpu_e[n_cpu_e++] = e; continue; }
      if (wh[i] != bmiss::kDmaCopy && B->track.take_pf((size_t)l * c.n_routed + e)) ++B->acc.pf_used;
      if (wh[i] == bmiss::kStaged) { stg_e[n_stg] = e; stg_si[n_stg++] = msi[i]; continue; }
      str_si[n_str_e] = wh[i] == bmiss::kDmaReuse ? msi[i] : -1;
      str_e[n_str_e++] = e;
      ++n_str_dma;
      if (wh[i] == bmiss::kDmaReuse) ++B->acc.reuse; else ++B->acc.copy;
    }
    B->pred.update(l, cpu_e, cpu_rows, n_cpu_e);
    ++B->acc.layers; B->acc.miss += n_miss_e; B->acc.staged += n_stg; B->acc.cpu += n_cpu_e;
  }
  // B5: after splitting by frac, if one side is empty (all DMA or all CPU) no comparison sample arises and frac would freeze at a limit — take one step toward the default (dma_frac_relax)
  //   C1: layers chosen by the model did not use frac (no relaxation either, like G2 — layers chosen by the base rule behave as usual)
  //   B2 STAGE_HIT: n_miss = misses minus those served straight from the ring; DMA share = n_str_dma (off = same as n_str_e)
  if (opt_.cpu_for_misses && n_miss >= 3 && (n_str_dma == 0 || n_cpu_e == 0) && !ds_model && !vrowind) dma_frac_relax(kind);  // R1 ROWIND layers stay outside adaptation
  int goff = 0, ng = 0;
  auto add_group = [&](int e, const uint8_t* rec) {  // group = expert × rows ≤ 8 (kernel limit) — more rows chain further groups
    for (int i0 = cnt[e]; i0 < cnt[e + 1]; i0 += 8) {
      k::GroupDesc& d = w.gdesc_h[ng++];
      d.w1 = rec + lay.w1; d.s1 = rec + lay.s1; d.w3 = rec + lay.w3; d.s3 = rec + lay.s3; d.w2 = rec + lay.w2; d.s2 = rec + lay.s2;
      d.row0 = goff; d.n = std::min(8, cnt[e + 1] - i0);
      for (int i = i0; i < i0 + d.n; ++i) { const int mj = rows_by_e[i]; w.g_rows_h[goff] = mj / k; w.g_rw_h[goff] = w.rw_h[mj]; ++goff; }
    }
  };
  for (int ei = 0; ei < n_hit_e; ++ei) {
    const int e = hit_e[ei];
    int slot = store_.slot_of(l, e); store_.touch(slot); add_group(e, store_.dev_rec(slot));
    if (stats) stats->n_hit += n_of(e);
    if (stats) for (int i = cnt[e]; i < cnt[e + 1]; ++i) ++stats->row_hit[grow0 + rows_by_e[i] / k];
  }
  for (int ei = 0; ei < n_stg; ++ei) {  // B2 STAGE_HIT: ring records in the same range as resident groups (no copy wait — st_ only waits for copy events finished before launch)
    const int e = stg_e[ei];
    add_group(e, store_.staging_rec(stg_si[ei]));
    if (stats) stats->n_dma_rows += n_of(e);
    if (stats) for (int i = cnt[e]; i < cnt[e + 1]; ++i) ++stats->row_dma[grow0 + rows_by_e[i] / k];
  }
  const int ng_hit = ng, R_hit = goff;
  // The DMA share goes through the staging ring (as in prefill): it never evicts cache slots via LRU — cache entry/exit is done only by score-based promote()
  int str_stage[512];  // O1: a fixed array instead of a per-layer std::vector — n_str_e ≤ misses ≤ E ≤ 512
  int n_stage = 0;
  const bool cw = cw_ok && cw_on_ && n_str_e > 0 && (size_t)l < cw_used_.size();  // E4 demand-copy instrumentation (see the cw_begin comment)
  // B2 HIVE_DECODE_CPU_FIRST: decide only slots and group tables here (host); issue the copies after the CPU start and resident launch (the "deferred demand copies" below) — same slots, same order
  const bool defer_copy = B && B->cpu_first;
  int pc_e[512], pc_si[512], n_pc = 0;
  uint32_t pc_pos[512];
  auto issue_copy = [&](int e, int si, uint32_t pos) {  // one plain copy (wait → copy → event); with sh also recorded in the ring tracker
    if (stage_used_[si]) CUDA_CHECK(cudaStreamWaitEvent(side_, stage_freed_[si], 0));
    store_.copy_to_staging(si, l, e, side_);
    CUDA_CHECK(cudaEventRecord(stage_copied_[si], side_));
    if (sh) B->track.put((size_t)l * c.n_routed + e, pos, false);
  };
  if (cw && !defer_copy) CUDA_CHECK(cudaEventRecord(cw_ev_[3 * l], side_));
  for (int ei = 0; ei < n_str_e; ++ei) {  // issue side-stream H2D first — copies overlap resident expert computation
    const int e = str_e[ei];
    const int pre = sh ? str_si[ei] : -1;  // B2 STAGE_HIT: reuse the in-flight ring copy (no new copy)
    int slot = pre;
    if (pre < 0) {
      const uint32_t pos = stage_next_;  // ring position (STAGE_HIT tracking) — advanced by the formula below
      const int si = stage_next_++ % store_.staging_slots();
      slot = si;
      if (defer_copy) { pc_e[n_pc] = e; pc_si[n_pc] = si; pc_pos[n_pc++] = pos; }
      else issue_copy(e, si, pos);
      if (stats) stats->n_streamed++;
    }
    str_stage[n_stage++] = slot;
    add_group(e, store_.staging_rec(slot));
    if (stats) stats->n_dma_rows += n_of(e);
    if (stats) for (int i = cnt[e]; i < cnt[e + 1]; ++i) ++stats->row_dma[grow0 + rows_by_e[i] / k];
  }
  if (cw && !defer_copy) { CUDA_CHECK(cudaEventRecord(cw_ev_[3 * l + 1], side_)); cw_used_[l] |= 1; }
  const int ng_str = ng - ng_hit, R_str = goff - R_hit;
  if (ds_sample) {  // C1 sample: end of this layer's last demand copy (side_ — promotion slices are issued after it by promo_pump below)
    S->obs.n_str = n_str_e; S->obs.ng_str = ng_str;
    if (n_str_e > 0) CUDA_CHECK(cudaEventRecord(S->c1, side_));
  }
  // 2) prepare CPU jobs (fp8 activations → fp32 table lookup) → wake the pool right away (starting only after all GPU table copies and launches left the CPU idle for tens of µs per layer)
  std::vector<ExpertStore::Job>& jobs = *p.jobs;
  jobs.clear();
  int job_rows = 0;
  p.started = false; p.t_cpu0 = 0; p.rows_dev = nullptr;
  if (n_cpu_e > 0) {
    const float* lut = e4m3_lut();
    // B3 HIVE_CPU_UNPACK2 (cpu::unpack2_enabled — off = the table lookup below): same values (bit-compared over all 256 codes — bench_expert_cpu), 8 at a time with vectors.
    //   In batched decode every row going to the CPU costs 5120 + 160 scalar lookups on the prep critical path.
    static const bool unpack2 = cpu::unpack2_enabled();
    auto unpack_row = [&](int m) {
      if (unpack2) { cpu::unpack_act_row(w.xq_h + (size_t)m * dim, w.xs_h + (size_t)m * (dim / 32), dim, w.a_f_h + (size_t)m * dim, w.a_s_h + (size_t)m * (dim / 32)); return; }
      for (int d = 0; d < dim; ++d) w.a_f_h[(size_t)m * dim + d] = lut[w.xq_h[(size_t)m * dim + d]];
      for (int b = 0; b < dim / 32; ++b) w.a_s_h[(size_t)m * (dim / 32) + b] = e8m0_to_f32(w.xs_h[(size_t)m * (dim / 32) + b]);
    };
    if (p.defer_unpack) {  // S1 HIVE_DECODE_EARLY_ROUTE: activations are not in mapped host memory yet — only row markers for the pool (same rows as the two branches below); decode_layer (Ops::start_cpu) unpacks the same way after the front ends
      std::vector<uint8_t>& need = D.need;
      need.assign((size_t)M, fast ? 0 : 1);
      if (fast) for (int ei = 0; ei < n_cpu_e; ++ei) for (int i = cnt[cpu_e[ei]]; i < cnt[cpu_e[ei] + 1]; ++i) need[rows_by_e[i] / k] = 1;
      p.unpack_pending = true; p.unpack_M = M;
    } else
    if (!fast) { for (int m = 0; m < M; ++m) unpack_row(m); }
    else {  // O1 B: only rows going to the CPU (same table lookup — the values used are identical; rows not unpacked are never read by CPU jobs)
      std::vector<uint8_t>& need = D.need;
      need.assign((size_t)M, 0);
      for (int ei = 0; ei < n_cpu_e; ++ei) for (int i = cnt[cpu_e[ei]]; i < cnt[cpu_e[ei] + 1]; ++i) need[rows_by_e[i] / k] = 1;
      for (int m = 0; m < M; ++m) if (need[m]) unpack_row(m);
    }
    const size_t scratch_n = ExpertStore::job_scratch_floats(I);
    // HIVE_DECODE_DEFER / HIVE_DECODE_SKIP_MISS (runtime.h): rows ranked 3rd or lower (mj % k ≥ 2) are deferred / skipped — early-route decode of
    //   backbone layers only, never the last layer (nothing would add it), never verify row-independent accumulation
    const bool split_ok = p.defer_unpack && l < c.n_layers - 1 && !vrowind && !ub_active_;
    const bool defer_now = dec_defer_ && split_ok;
    const float skip_f = split_ok ? dec_skip_ : 0.f;
    // The previous layer's deferred batch may still be running on the pool: it reads djobs_ and def_rows_h_, which the lines below
    //   rewrite (without this the first decode of a DS run with HIVE_DECODE_DEFER=1 never finished) — finish and add it first
    if (def_pending_) deferred_flush();
    if (defer_now) { djobs_.clear(); def_rows_n_ = 0; }
    float rowsum[8] = {};
    if (skip_f > 0.f) for (int m = 0; m < M && m < 8; ++m) for (int j = 0; j < k; ++j) rowsum[m] += w.rw_h[m * k + j];
    for (int ei = 0; ei < n_cpu_e; ++ei) {
      const int e = cpu_e[ei];
      ExpertStore::Job db{};
      db.layer = l; db.e = e; db.R = 0;
      for (int i0 = cnt[e]; i0 < cnt[e + 1]; i0 += ExpertStore::kMaxRows) {
        ExpertStore::Job jb{};
        jb.layer = l; jb.e = e; jb.R = 0;
        jb.scratch = w.scratch_h + (size_t)job_rows * scratch_n;
        for (int i = i0; i < cnt[e + 1] && jb.R < ExpertStore::kMaxRows; ++i) {
          const int mj = rows_by_e[i], m = mj / k;
          if ((defer_now || skip_f > 0.f) && mj % k >= 2) {
            if (skip_f > 0.f && m < 8 && w.rw_h[mj] < skip_f * rowsum[m]) { ++skip_rows_total_; continue; }
            if (defer_now) {
              if (db.R == ExpertStore::kMaxRows) { djobs_.push_back(db); db = ExpertStore::Job{}; db.layer = l; db.e = e; db.R = 0; }
              if (db.R == 0) db.scratch = def_scratch_h_ + (size_t)def_rows_n_ * scratch_n;
              db.a_f[db.R] = w.a_f_h + (size_t)m * dim;
              db.a_s[db.R] = w.a_s_h + (size_t)m * (dim / 32);
              db.route_w[db.R] = w.rw_h[mj];
              db.out[db.R] = def_out_h_ + (size_t)def_rows_n_ * dim;
              def_rows_h_[def_rows_n_++] = m;
              ++db.R;
              continue;
            }
          }
          jb.a_f[jb.R] = w.a_f_h + (size_t)m * dim;
          jb.a_s[jb.R] = w.a_s_h + (size_t)m * (dim / 32);
          jb.route_w[jb.R] = w.rw_h[mj];
          jb.out[jb.R] = w.cpu_out_h + (size_t)job_rows * dim;
          w.cpu_rows_h[job_rows++] = m;
          ++jb.R;
        }
        if (jb.R > 0) jobs.push_back(jb);
      }
      if (db.R > 0) djobs_.push_back(db);
      if (stats) stats->n_cpu += n_of(e);
      if (stats) for (int i = cnt[e]; i < cnt[e + 1]; ++i) ++stats->row_cpu[grow0 + rows_by_e[i] / k];
    }
    if (start_cpu) { p.t_cpu0 = now_ms(); store_.start_jobs(jobs, /*owned=*/pregate_k_ > 0); p.started = true; }  // P1: owned batches on decode layers only
  }
  p.ds_jobs = (int)jobs.size();
  if (ds_sample) { S->obs.jobs = p.ds_jobs; S->has_cpu = !jobs.empty(); }
  // E4 HIVE_DECODE_COPY_PRIO: promotion slices after this layer's demand copies (issued on side_ above) — after waking the CPU pool, so the issue calls do not delay CPU experts
  //   (off = returns immediately). Nothing else is issued on side_ in between, so the gate sits right after this layer's demand copies. O1 B HOST_FAST moves it after the GPU launch (5).
  if (!fast && pump && !defer_copy) promo_pump(l, model_.n_loaded_layers());  // B2 CPU_FIRST: must follow the demand copies, so it runs after the deferred copies (below)
  if (stats) stats->ms_host += now_ms() - th0;
  const double tl0 = now_ms();
  if (p.slot >= 0) D.h_prep[p.slot] += tl0 - th0;
  D.mark(p.slot, DecodeOverlap::kExp, st_);
  if (ds_sample) CUDA_CHECK(cudaEventRecord(S->g0, st_));  // C1 sample origin (same point as O1 kExp)
  // 3) GPU: copy the small tables (groups, rows, weights) to the device once → gather activations → resident groups → (copy wait) DMA groups
  static const bool use_mx = getenv("HIVE_MX") ? atoi(getenv("HIVE_MX")) != 0 : true;  // tensor-core block-scale variant (HIVE_MX=0 = CUDA cores)
  const int32_t* g_rows_dev = w.g_rows.as<int32_t>();
  const float* g_rw_dev = w.g_rw.as<float>();
  const k::GroupDesc* gdesc_dev = w.gdesc_dev.as<k::GroupDesc>();
  (void)g_rows_dev; (void)g_rw_dev; (void)gdesc_dev;
  const bool ship_cpu_rows = fast && job_rows > 0;  // O1 B: ship CPU row numbers in this copy too (removes one copy after the CPU wait)
  if (goff > 0 || ship_cpu_rows) {
    // three tables in one copy (copy-engine queue latency ~10 µs each): [gdesc ng][rows goff][rw goff] concatenated in pinned scratch to the device (+ HOST_FAST: [cpu_rows job_rows])
    const size_t off_rows = sizeof(k::GroupDesc) * (size_t)ng, off_rw = off_rows + (size_t)goff * 4, off_cpu = off_rw + (size_t)goff * 4;
    const size_t total = off_cpu + (ship_cpu_rows ? (size_t)job_rows * 4 : 0);  // ≤ (GroupDesc+8)·R — goff + job_rows = R, ng ≤ goff
    std::memcpy(w.tbl_h, w.gdesc_h, off_rows);
    std::memcpy(w.tbl_h + off_rows, w.g_rows_h, (size_t)goff * 4);
    std::memcpy(w.tbl_h + off_rw, w.g_rw_h, (size_t)goff * 4);
    if (ship_cpu_rows) std::memcpy(w.tbl_h + off_cpu, w.cpu_rows_h, (size_t)job_rows * 4);
    // T2 (S1 HIVE_DECODE_EARLY_ROUTE batched regression — service A/B c4 93.4→60.9, c8 110.7→68.5, c1 unchanged): on the early path (p.er_tbl — only
    //   decode_layer sets it) this table copy sits on st_ **behind front graph A** (the host issues it during the shared expert). But B2 CPU_FIRST's demand DMA
    //   (side_, 18.8 MB per expert) is issued below in an immediately runnable state → the copy engine picks up the demand DMA (+ E4 promotion slices) first,
    //   and the few-KB table copy, ready only after A ends, queues behind it → resident groups and the DMA sample origin (dma_e0_) are pushed to the end of
    //   the demand DMA. The base path issues the table first after the A sync (empty st_ — runs at once) and CPU_FIRST defers demand DMA behind it, so it has no
    //   such problem. M=1 uses DMA only with ≥ 3 misses, so it is rarely hit (consistent with c1 unchanged). Second effect: with dma_e0_ after the copy, DMA
    //   samples (gpu_ms) exclude copy time and dma_frac_consume pushes frac to its ceiling (0.8) (a fixed HIVE_DMA_FRAC 0.9 cost c4/c8 −28–41 % — a loss of the same size).
    //   Fix: put the table on a dedicated stream unordered with the front (D.er_tbl_st) — with CPU_FIRST (defer_copy) it is issued **before** the demand DMA and
    //   runnable at once → st_ only waits for its end event (with CPU_FIRST off the demand DMA is issued first in the loop above — same order as the base path
    //   and the GPU test test_early_route, never worse).
    //   Safety: the previous layer's (l−1) expert groups and CPU accumulation that read tbl_dev/tbl_h sit before A(l) on st_, and the host only gets here after
    //   seeing the published number inside A(l) (or the sync of the fallback path) → those reads are done. Table values, kernels and order are the same (bit-identical) — only the stream of the table copy changes.
    if (p.er_tbl && D.er_tbl_st) {
      CUDA_CHECK(cudaMemcpyAsync(w.tbl_dev.p, w.tbl_h, total, cudaMemcpyHostToDevice, D.er_tbl_st));
      CUDA_CHECK(cudaEventRecord(D.er_tbl, D.er_tbl_st));
      CUDA_CHECK(cudaStreamWaitEvent(st_, D.er_tbl, 0));
      ++D.er_tbl_n;
    } else
    CUDA_CHECK(cudaMemcpyAsync(w.tbl_dev.p, w.tbl_h, total, cudaMemcpyHostToDevice, st_));
    gdesc_dev = reinterpret_cast<const k::GroupDesc*>(w.tbl_dev.as<uint8_t>());
    g_rows_dev = reinterpret_cast<const int32_t*>(w.tbl_dev.as<uint8_t>() + off_rows);
    g_rw_dev = reinterpret_cast<const float*>(w.tbl_dev.as<uint8_t>() + off_rw);
    if (ship_cpu_rows) p.rows_dev = reinterpret_cast<const int32_t*>(w.tbl_dev.as<uint8_t>() + off_cpu);
    if (goff > 0 && !(fuse_ && use_mx)) {  // the fused (tensor-core) variant reads xq/xs directly through the row map — saves 2 gather launches
      k::gather_rows_u8(w.xq.as<uint8_t>(), g_rows_dev, goff, dim, w.g_xq.as<uint8_t>(), st_);
      k::gather_rows_u8(w.xs.as<uint8_t>(), g_rows_dev, goff, dim / 32, w.g_xs.as<uint8_t>(), st_);
    }
  }
  // D1 HIVE_DECODE_FUSED: replace the fuse_ && use_mx chain below (w13+quantization → w2 → accum_bf16_rows_seq, 3 launches) with one launch (moe_decode.cu).
  //   Same formula, same mma order and same accumulation order, so acc is bit-identical (test_decode_moe). Outside the conditions (M > 8 short prefill, shape, record alignment) the chain runs.
  static const bool decode_fused = env_on("HIVE_DECODE_FUSED");
  bool use_fused = decode_fused && fuse_ && use_mx && ng > 0 && k::moe_decode_fused_ok(dim, I, M, ng);
  for (int gi = 0; use_fused && gi < ng; ++gi) use_fused = k::moe_decode_desc_aligned(w.gdesc_h[gi]);
  auto run_groups = [&](int g0, int n_groups, int r0, int n_rows) {
    if (n_groups <= 0) return;
    if (use_fused) {
      k::moe_decode_fused(gdesc_dev + g0, n_groups, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), g_rows_dev, g_rw_dev, r0, n_rows, M, dim, I, c.swiglu_limit,
                          w.y.as<bf16>(), w.yq.as<uint8_t>(), w.ys.as<uint8_t>(), w.eout.as<bf16>(), w.acc.as<float>(), st_);
      return;
    }
    if (fuse_ && use_mx) {  // w13 + swiglu + 32-block quantization in one kernel (no act_quant)
      k::mx_grouped_w13(gdesc_dev + g0, n_groups, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), I, dim, g_rw_dev, c.swiglu_limit, w.y.as<bf16>(), st_, g_rows_dev,
                        w.yq.as<uint8_t>(), w.ys.as<uint8_t>());
    } else {
    if (use_mx) k::mx_grouped_w13(gdesc_dev + g0, n_groups, w.g_xq.as<uint8_t>(), w.g_xs.as<uint8_t>(), I, dim, g_rw_dev, c.swiglu_limit, w.y.as<bf16>(), st_);
    else k::gemv_grouped_w13(gdesc_dev + g0, n_groups, w.g_xq.as<uint8_t>(), w.g_xs.as<uint8_t>(), I, dim, g_rw_dev, c.swiglu_limit, w.y.as<bf16>(), st_);
    k::act_quant_fp8(w.y.as<bf16>() + (size_t)r0 * I, n_rows, I, w.yq.as<uint8_t>() + (size_t)r0 * I, w.ys.as<uint8_t>() + (size_t)r0 * (I / 32), st_);
    }
    if (use_mx) k::mx_grouped_w2(gdesc_dev + g0, n_groups, w.yq.as<uint8_t>(), w.ys.as<uint8_t>(), dim, I, w.eout.as<bf16>(), st_);
    else k::gemv_grouped_w2(gdesc_dev + g0, n_groups, w.yq.as<uint8_t>(), w.ys.as<uint8_t>(), dim, I, w.eout.as<bf16>(), st_);
    k::accum_bf16_rows_seq(w.eout.as<bf16>() + (size_t)r0 * dim, g_rows_dev + r0, n_rows, M, dim, w.acc.as<float>(), st_);
  };
  const bool time_dma = !dma_evt_pending_ && !D.dma_armed && n_str_dma > 0 && !jobs.empty();  // comparison sample only when both exist (O1: so the two halves never overwrite the same events)
  if (time_dma) { CUDA_CHECK(cudaEventRecord(dma_e0_, st_)); D.dma_armed = true; }
  for (int j = 0; j < n_stg; ++j) CUDA_CHECK(cudaStreamWaitEvent(st_, stage_copied_[stg_si[j]], 0));  // B2 STAGE_HIT (finished copies — ordering made explicit)
  run_groups(0, ng_hit, 0, R_hit);
  if (ds_sample) CUDA_CHECK(cudaEventRecord(S->r1, st_));  // C1 end of resident groups
  if (cw) { CUDA_CHECK(cudaEventRecord(cw_ev_[3 * l + 2], st_)); cw_used_[l] |= 2; }  // E4 time st_ reaches the DMA groups (wait start)
  if (defer_copy) {  // B2 CPU_FIRST: the deferred demand copies (same slots, order and stream as without deferral) → then promotion slices (gate behind demand)
    if (cw) CUDA_CHECK(cudaEventRecord(cw_ev_[3 * l], side_));
    for (int j = 0; j < n_pc; ++j) issue_copy(pc_e[j], pc_si[j], pc_pos[j]);
    if (cw) { CUDA_CHECK(cudaEventRecord(cw_ev_[3 * l + 1], side_)); cw_used_[l] |= 1; }
    if (!fast && pump) promo_pump(l, model_.n_loaded_layers());
  }
  for (int j = 0; j < n_stage; ++j) CUDA_CHECK(cudaStreamWaitEvent(st_, stage_copied_[str_stage[j]], 0));
  run_groups(ng_hit, ng_str, R_hit, R_str);
  for (int j = 0; j < n_stage; ++j) { CUDA_CHECK(cudaEventRecord(stage_freed_[str_stage[j]], st_)); stage_used_[str_stage[j]] = 1; }
  for (int j = 0; j < n_stg; ++j) { CUDA_CHECK(cudaEventRecord(stage_freed_[stg_si[j]], st_)); stage_used_[stg_si[j]] = 1; }
  if (time_dma) CUDA_CHECK(cudaEventRecord(dma_e1_, st_));
  if (ds_sample) CUDA_CHECK(cudaEventRecord(S->g1, st_));  // C1 end of GPU experts (same point as O1 kGEnd)
  pmark("moe.gpu_experts");
  D.mark(p.slot, DecodeOverlap::kGEnd, st_);
  if (p.slot >= 0) { D.h_launch[p.slot] += now_ms() - tl0; D.jobs_on[p.slot] = !jobs.empty(); }
  // 4) deferred work (O1 B HOST_FAST — host bookkeeping that does not affect results moved after the GPU launch): promotion slices, usage observation (scores, policy), expert tracing
  if (fast) {
    const double tp0 = now_ms();
    if (pump) promo_pump(l, model_.n_loaded_layers());
    observe_rows();
    if (!D.obs_off) xtrace_layer(l, M, k, w.route_ids_h);  // F1 (off = unchanged)
    const double dt = now_ms() - tp0;
    if (stats) stats->ms_host += dt;
    if (p.slot >= 0) D.h_post[p.slot] += dt;
  }
  if (B) {  // B2: [decode-miss] CPU job count; HIVE_DECODE_PREFETCH predicted pre-copy for the next layer (after launch — off the critical path, after promotion slices)
    B->acc.cpu_jobs += (long)jobs.size();
    if (B->prefetch > 0 && l + 1 < model_.n_loaded_layers()) bm_prefetch(l + 1);
  }
  p.job_rows = job_rows; p.kind = kind; p.time_dma = time_dma;
  p.cpu_rows_h = w.cpu_rows_h; p.cpu_out_d = w.cpu_out_d; p.acc = w.acc.as<float>();
}

// Plain call (single-sequence path moe(), per-layer path): launch + finish back to back.
void Runtime::moe_decode_experts(const LayerWeights& L, int l, int M, ForwardStats* stats) {
  MoePend p;
  p.jobs = &w_->cpu_jobs;
  moe_decode_experts(L, l, M, stats, p, true, true, true);
  moe_decode_finish(p, stats);
}

void Runtime::moe_decode_start_cpu(MoePend& p) {
  if (p.started || p.jobs->empty()) return;
  p.t_cpu0 = now_ms();
  store_.start_jobs(*p.jobs, /*owned=*/pregate_k_ > 0);  // P1: owned batches on decode layers only
  p.started = true;
}

// 4) CPU experts were already woken in step 2) while the GPU runs the work above — here we only wait for completion
void Runtime::deferred_flush() {
  if (!def_pending_) return;
  store_.wait_jobs();
  const Config& c = model_.cfg();
  k::hc_inject_rows(w_->h.as<bf16>(), post_def_.as<float>(), def_out_d_, def_rows_d_, def_rows_n_, def_M_, c.hc, c.dim, st_);
  def_pending_ = false;
  djobs_.clear();
}

void Runtime::moe_decode_wait(MoePend& p, ForwardStats* stats) {
  if (p.jobs->empty()) return;
  moe_decode_start_cpu(p);  // already started on the default path (no-op) — so a deferred batch is never waited on without being started
  const double twait = now_ms();
  store_.wait_jobs();
  if (stats) stats->ms_cpu_wait += now_ms() - twait;
  const double cpu_ms = std::max(0.0, store_.jobs_done_ms() - p.t_cpu0);  // from start (wake-up) to the completion time stamped by the pool
  if (p.time_dma) { dma_evt_pending_ = true; dma_evt_kind_ = p.kind; dma_evt_cpu_ms_ = cpu_ms; dov_->dma_armed = false; }
  if (p.ds && dsplit_) {  // C1: the CPU table (job count → pool interval, host clock) every layer; on sample layers also into the sample
    dsplit::add_cpu(dsplit_->m[p.ds_cls], p.ds_jobs, cpu_ms);
    if (p.ds_sample) dsplit_->obs.cpu_ms = cpu_ms;
  }
  if (stats) stats->ms_cpu_span += cpu_ms;
  if (p.slot >= 0) dov_->cpu_ms[p.slot] += cpu_ms;
}
void Runtime::moe_decode_accum(MoePend& p) {
  Work& w = *w_;
  const int dim = model_.cfg().dim;
  if (p.ds_sample && dsplit_) {  // C1 sample: accumulation launch point (O1 kAcc) — not recorded without a CPU share (layer end = g1); closes the sample (it can now be collected)
    if (!p.jobs->empty()) CUDA_CHECK(cudaEventRecord(dsplit_->acc, st_));
    dsplit_->open = false;
  }
  if (!p.jobs->empty()) {
    dov_->mark(p.slot, DecodeOverlap::kAcc, st_);
    // R1 HIVE_MTP_VERIFY_ROWIND: fixed job-row-order accumulation (even with several CPU results for row m the order = expert-id order — independent of other rows and runs)
    auto accum = [&](const int32_t* rows) {
      if (p.rowind) k::accum_f32_rows_ordered(p.cpu_out_d, rows, p.job_rows, dim, p.acc, st_);
      else k::accum_f32_rows(p.cpu_out_d, rows, p.job_rows, dim, p.acc, st_);
    };
    if (p.rows_dev) accum(p.rows_dev);  // O1 B: row numbers are already on the device with the expert table
    else {
      CUDA_CHECK(cudaMemcpyAsync(w.cpu_rows.p, p.cpu_rows_h, (size_t)p.job_rows * 4, cudaMemcpyHostToDevice, st_));
      accum(w.cpu_rows.as<int32_t>());
    }
  }
  dov_->mark(p.slot, DecodeOverlap::kEnd, st_);
  pmark("moe.cpu_experts");
}
void Runtime::moe_decode_finish(MoePend& p, ForwardStats* stats) {
  moe_decode_wait(p, stats);
  moe_decode_accum(p);
}

// B2 HIVE_DECODE_PREFETCH: of layer l2's predicted misses (the same layer's CPU share in the previous batched step), copy up to prefetch experts that are neither
//   resident nor intact in the ring into the staging ring on promo_, **only while the link is idle** (no pending promotion slices; events recorded at the end of
//   side_ (demand DMA) and promo_ (promotions) complete immediately). Evidence: in batched measurements extra DMA was a loss (HIVE_DMA_FRAC 0.9 = c4/c8
//   −28–41 %, C1 balance −3/−5 %, --promote-misses on −12–14 %) — so nothing is issued while demand or promotion traffic uses the link. On promo_, later
//   promotion slices queue behind these copies (≤ prefetch records × ~0.67 ms per layer).
//   Cache slots are untouched (no eviction). Slot safety: the ring position comes from stage_next_ (same rule as other writers) and stage_freed_ is recorded
//   after the copy so the next writer waits for it; consumption happens through the STAGE_HIT lookup (StageTrack::find).
void Runtime::bm_prefetch(int l2) {
  BatchMiss& B = *bmiss_;
  const int S = store_.staging_slots(), margin = bmiss::kLayerAlloc + B.prefetch;
  if (S <= margin) return;  // with a ring smaller than one layer's allocation, pre-copied records could never pass the consumption check
  const size_t nr = (size_t)model_.cfg().n_routed;
  const int E2 = model_.layer(l2).n_routed ? model_.layer(l2).n_routed : (int)nr;
  int cand[bmiss::kMaxPrefetch];
  auto skip = [&](int e) { return e < 0 || e >= E2 || store_.slot_of(l2, e) >= 0 || B.track.find((size_t)l2 * nr + e, stage_next_, S, margin) >= 0; };
  const int n = B.pred.pick(l2, B.prefetch, skip, cand);
  if (n == 0) return;
  if (store_.promo_backlog()) { ++B.acc.pf_busy; return; }
  CUDA_CHECK(cudaEventRecord(B.probe[0], side_));
  CUDA_CHECK(cudaEventRecord(B.probe[1], promo_));
  if (cudaEventQuery(B.probe[0]) != cudaSuccess || cudaEventQuery(B.probe[1]) != cudaSuccess) { ++B.acc.pf_busy; return; }
  for (int i = 0; i < n; ++i) {
    const int e = cand[i];
    const uint32_t pos = stage_next_++;
    const int si = (int)(pos % (uint32_t)S);
    if (stage_used_[si]) CUDA_CHECK(cudaStreamWaitEvent(promo_, stage_freed_[si], 0));
    store_.copy_to_staging(si, l2, e, promo_);
    CUDA_CHECK(cudaEventRecord(stage_copied_[si], promo_));
    CUDA_CHECK(cudaEventRecord(stage_freed_[si], promo_));  // even if unused and pushed out, the next writer waits for the end of this copy
    stage_used_[si] = 1;
    B.track.put((size_t)l2 * nr + e, pos, true);
    ++B.acc.pf_issued;
  }
}

// [decode-miss M=m] (HIVE_PROFILE sample steps, when any B2 switch is on) — sum over all batched decode layers since the last line:
//   layers; miss = non-resident missed experts; staged = to the GPU from a finished ring record (no copy, no CPU); reuse = in-flight ring copy used as DMA share; copy = new demand DMA;
//   cpu = experts sent to the CPU (jobs = CPU jobs); pred/used = total size of the previous step's same-layer CPU share (prediction) / of those, actual misses this step → precision = used/pred,
//   recall = used/miss; prefetch issued/used/busy = pre-copies issued, of those consumed, layers skipped because the link was busy. With STAGE_HIT off (CPU_FIRST only) the later items are 0.
void Runtime::bm_report(int M) {
  if (!bmiss_ || !profile_ || (profile_every_ > 1 && step_ % profile_every_ != 0)) return;  // same sample-step decision as pmark
  bmiss::Acc& a = bmiss_->acc;
  auto ratio = [](long x, long y) { return y > 0 ? (double)x / (double)y : 0.0; };
  fprintf(stderr, "[decode-miss M=%d] layers %ld · miss %ld · staged %ld · reuse %ld · copy %ld · cpu %ld (jobs %ld) · pred %ld used %ld (precision %.2f · recall %.2f) · "
                  "prefetch issued %ld used %ld busy %ld%s%s\n",
          M, a.layers, a.miss, a.staged, a.reuse, a.copy, a.cpu, a.cpu_jobs, a.pred, a.pred_used, ratio(a.pred_used, a.pred), ratio(a.pred_used, a.miss),
          a.pf_issued, a.pf_used, a.pf_busy, bmiss_->cpu_first ? " · cpu-first" : "", bmiss_->stage_hit ? " · stage-hit" : "");
  a = bmiss::Acc{};
}

// Prefill expert computation — several sub-chunks (C2 layer-first tiles) at once: each expert's record is streamed **only once** (for both streaming and
//   resident experts) with one GEMM per sub-chunk. With a single sub-chunk this equals moe_experts. Missed experts = staging DMA streaming + CPU share
//   (misses with the fewest rows go to the CPU first).
//   Measured (M=2048 chunk, 7.3 s): moe.gpu_experts 6.0 s = ~235 misses per layer × 0.64 ms (PCIe 4.0 27.9 GB/s) — 82 % of prefill was DMA wait.
//   The CPU pool handles jobs of ≤ 8 rows in ~0.35 ms (both nodes ~110 GB/s), so peeling off the misses with fewest rows to the CPU runs in parallel with DMA.
//   The share uses the same adaptive rule as decode (kind 1, dma_frac_[1]): ±0.05 from the previous sample's CPU completion time vs GPU (streaming + compute) time.
// Correctness: in prefill, route_to_host does not copy activations (xq/xs) to the host (moe(): copy_acts = !prefill), so the CPU share must not read xq_h —
//   it would be stale (the CPU share takes the experts with the fewest rows, so a stale read affected well under 1 % of rows and slipped past e2e tests).
//   Instead only the rows going to the CPU are gathered on the GPU, copied to the host and unpacked to fp32 — copy volume = CPU rows × 5 KB.
// ---- adaptive DMA share (dma_frac_) — kinds kDmaDecode (batched decode), kDmaShort (short prefill: decode-policy path, M > max_batch), kDmaStream (streaming prefill)
// B4/B5: limits and defaults per kind are the established values (decode-policy path [0.1, 0.8], streaming [0.2, 0.95], defaults 0.25/0.5/0.5 — existing
//   values, not newly measured). Rules: a sample is clipped to the limits of **the kind that produced it** (not the consuming path's), short and streaming
//   prefill keep separate state, and on layers where no comparison sample can arise (one side of the split is empty) frac takes one step back toward the default.
//   The relax step = the same 0.05 as the sample step (not a new constant). It stops at the default, so long one-sided stretches never diverge, and once samples return the adaptive rule wins.
namespace {
constexpr float kDmaFracLo[4] = {0.1f, 0.1f, 0.2f, 0.1f};    // floor: at 0 the DMA samples would stop and it could never rise again
constexpr float kDmaFracHi[4] = {0.8f, 0.8f, 0.95f, 0.95f};  // R2 kDmaStreamShort: floor = 0.1 as for short prefill (kind 1), ceiling = 0.95 as for streaming (kind 2)
constexpr float kDmaFracStep = 0.05f;
}  // namespace
void Runtime::dma_frac_consume() {
  if (!dma_evt_pending_ || cudaEventQuery(dma_e1_) != cudaSuccess) return;
  float gpu_ms = 0.f;
  cudaEventElapsedTime(&gpu_ms, dma_e0_, dma_e1_);
  const int kd = dma_evt_kind_;
  float& f = dma_frac_[kd];
  if (dma_evt_cpu_ms_ > gpu_ms * 1.2) f = std::min(kDmaFracHi[kd], f + kDmaFracStep);
  else if (gpu_ms > dma_evt_cpu_ms_ * 1.2) f = std::max(kDmaFracLo[kd], f - kDmaFracStep);
  dma_evt_pending_ = false;
}
void Runtime::dma_frac_relax(int kind) {
  if (dma_frac_fixed_ >= 0.f || dma_evt_pending_) return;  // with a fixed value frac is unused; with a pending sample that sample decides
  float& f = dma_frac_[kind];
  const float d = kDmaFracDefault[kind];
  f = f > d ? std::max(d, f - kDmaFracStep) : std::min(d, f + kDmaFracStep);
}

// G2: armed GPU sample → per-layer-kind cost EMAs (pure copy time per record, resident compute time per row) + predicted vs actual GPU window. If events are not done yet, try later (never blocks).
void Runtime::split_consume(ForwardStats* stats) {
  if (!split_pending_) return;
  if (cudaEventQuery(split_g1_) != cudaSuccess) return;
  for (int i = 0; i < split_nd_; ++i) if (cudaEventQuery(split_d_[i][1]) != cudaSuccess) return;
  if (split_comp_rows_ > 0 && cudaEventQuery(split_c1_) != cudaSuccess) return;
  const int t = split_type_;
  float g = 0.f, x = 0.f;
  cudaEventElapsedTime(&g, split_g0_, split_g1_);
  if (split_nd_ > 0) {
    double sum = 0;
    for (int i = 0; i < split_nd_; ++i) { cudaEventElapsedTime(&x, split_d_[i][0], split_d_[i][1]); sum += x; }
    ps::ema(split_rec_ms_[t], split_has_rec_[t], sum / split_nd_);
  }
  if (split_comp_rows_ > 0) { cudaEventElapsedTime(&x, split_c0_, split_c1_); ps::ema(split_row_ms_[t], split_has_row_[t], (double)x / split_comp_rows_); }
  if (stats && split_pred_ok_) { stats->split_pred_gpu_ms += split_pred_gpu_; stats->split_act_gpu_ms += g; ++stats->split_gpu_n; }
  split_pending_ = false;
}

void Runtime::moe_experts(Seq& seq, const LayerWeights& L, int l, int M, ForwardStats* stats) {
  Work& w = *w_;
  std::vector<SubChunk> subs{SubChunk{M, w.route_ids_h, w.rw_h, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), w.acc.as<float>()}};
  moe_experts_multi(L, l, subs, stats);
  (void)seq;
}

void Runtime::moe_experts_multi(const LayerWeights& L, int l, const std::vector<SubChunk>& subs, ForwardStats* stats) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int dim = c.dim, I = c.moe_inter, E = L.n_routed ? L.n_routed : c.n_routed, k = L.n_act ? L.n_act : c.n_act;
  const ExpertLayout& lay = store_.layout();
  const int S = (int)subs.size();
  int total_rows = 0;
  for (const auto& sc : subs) total_rows += sc.M;
  const float score_weight = opt_.phase_score ? 1.f / std::max(1, total_rows) : 1.f;
  // group (sub-chunk, row, j) per expert — by s, then by row
  const bool obs = !dov_->obs_off;  // F1: false when forward_multi up_experts already observed this layer (off = always true → unchanged)
  struct Ref { int s, m, j; };
  std::vector<std::vector<Ref>> by_e(E);
  for (int s = 0; s < S; ++s) {
    const SubChunk& sc = subs[s];
    for (int m = 0; m < sc.M; ++m)
      for (int j = 0; j < k; ++j) { const int e = sc.route_ids_h[m * k + j]; by_e[e].push_back({s, m, j}); if (obs) store_.observe(l, e, score_weight, S == 1 ? 0 : -1); }  // D4: one sub-chunk = that sequence (unknown with several)
    if (obs) xtrace_hist(l, sc.M, k, E, sc.route_ids_h);
  }
  // for pre-copy prediction: per-expert row counts of this layer (used by the next prefill at the same layer)
  // Q3 HIVE_PREFILL_SMALL chunks (small_fwd_) do not record: chunks below threshold otherwise never take this function, so last_rows_ always holds the previous
  //   **long** chunk's rows — keeps long-prefill pre-copy prediction after short turns as before (HIVE_PREFETCH_SCALE's "previous rows = Σlr/top-k" also stays based on that long chunk). Off = unchanged.
  if (!small_fwd_) {
  if ((int)last_rows_.size() <= l) last_rows_.resize(l + 1);
  last_rows_[l].assign(E, 0);
  for (int e = 0; e < E; ++e) last_rows_[l][e] = (int)by_e[e].size();
  }
  // Q3 SMALL: make the input of post-step miss promotion (promote_after_step at the end of forward — called as usual for M_orig < threshold) equal to the
  //   decode-policy path — like moe_decode_experts' classify, every used non-resident expert (regardless of CPU/DMA share or pre-copy; same key l·E_trunk + e).
  //   The slot table does not change during a forward (promotions are committed only by commit_pending at the head). Cache policy only (no effect on computation). Off = this line does nothing.
  if (small_fwd_ && opt_.promote_misses > 0) for (int e = 0; e < E; ++e) if (!by_e[e].empty() && store_.slot_of(l, e) < 0) step_miss_.push_back(l * c.n_routed + e);
  std::vector<int> pf_slot(E, -1);  // pre-copied expert → staging slot
  if (pf_l_ == l) for (auto [e, si] : pf_) if (e < E) {
    pf_slot[e] = si;
    if (by_e[e].empty()) ++prefetch_wasted_; else ++prefetch_used_;
  }
  // 1) choose the CPU share: misses with fewest rows first (CPU time grows with rows, DMA does not). Total rows ≤ Work M (size of the host activation copy and a_f_h).
  std::vector<uint8_t> to_cpu_e(E, 0);
  int cpu_rows_total = 0;
  // G2 HIVE_PREFILL_SPLIT: on only when env_on (unset, empty or "0" = off) and the name is "balance" (ps::parse_mode, case-insensitive) — unknown names are off too (fall back to the base path).
  //   Off = none of the split_* below is used (base path). Same form as HIVE_CACHE_POLICY (on = env_on + name parsing).
  static const bool split_balance = env_on("HIVE_PREFILL_SPLIT") && ps::parse_mode(getenv("HIVE_PREFILL_SPLIT")) == ps::kBalance;
  ps::Decision split_dec;
  bool split_on_layer = false;
  int split_type = 0;  // layer kind: 0 = encoder layer, 1 = decoder tail (from the last kv-source layer — same formula as forward's tail_layer)
  if (split_balance) {
    const int tail_layer = c.kv_source_layers.empty() ? -1 : *std::max_element(c.kv_source_layers.begin(), c.kv_source_layers.end());
    split_type = tail_layer >= 0 && l >= tail_layer ? 1 : 0;
  }
  // R2 HIVE_PREFILL_SHORT_ADAPT=N (spo_.adapt_rows; 0 = off): when the layer's rows (sum over sub-chunks) ≤ N, use the own adaptive state kDmaStreamShort
  //   (the usual adaptive rule: CPU completion vs GPU expert interval ±0.05, [0.1, 0.95]) instead of the fixed value (HIVE_DMA_FRAC_PREFILL — 0.7, chosen at
  //   42K and 80K). Reason: at the same share, short chunks have few rows per expert (1K chunk × top-6 / 384 ≈ 16 rows), so the CPU share is cheap while DMA
  //   moves whole records regardless of rows (~0.67 ms each) — a share chosen on long chunks may not fit here (the numbers are decided by A/B — this variant
  //   adds no new rule, it only restores the adaptive rule for short chunks). Only how many go to the CPU changes — the CPU share selection rule (fewest rows
  //   first, cumulative rows ≤ w.M) and G2 are unchanged. With HIVE_DMA_FRAC (global fixed) that value is used as before.
  const int dma_kind = sp::short_stream_kind(spo_.adapt_rows, total_rows) ? kDmaStreamShort : kDmaStream;
  if (opt_.cpu_for_misses) {
    const int kind = dma_kind;  // B4: state separated from short prefill (kind 1, per-layer DMA ≤ 8) — sharing dma_frac_[1] made them contaminate each other; R2 = kDmaStream (off)
    dma_frac_consume();  // consume pending decode/short-prefill samples too — clipped to their own kind's limits (not to this path's [0.2,0.95])
    // H1 HIVE_DMA_FRAC_PREFILL=F (0 < F ≤ 1): fix the DMA share for streaming prefill (this kind) only — decode and short prefill are unchanged (adaptive or HIVE_DMA_FRAC).
    //   Measured: applying HIVE_DMA_FRAC=0.9 to every kind gives long prefill +4–9 % but decode c4/c8 −28–41 % — the global switch changed decode as well.
    //   Unset, empty, "0", out of range or non-numeric = unset (adaptive rule or HIVE_DMA_FRAC) — absorbed, never rejected. When both are set this one wins for this kind.
    static const float dma_frac_prefill = [] {
      const char* v = getenv("HIVE_DMA_FRAC_PREFILL");
      if (!v || !*v) return -1.f;
      const double f = atof(v);
      return (std::isfinite(f) && f > 0.0 && f <= 1.0) ? (float)f : -1.f;
    }();
    const bool fixed_prefill = dma_frac_prefill > 0.f && kind == kDmaStream;  // R2: the fixed prefill share is not applied to the short streaming kind (off = kind is always kDmaStream — base formula)
    const float frac = fixed_prefill ? dma_frac_prefill : dma_frac_fixed_ >= 0.f ? dma_frac_fixed_ : dma_frac_[kind];
    std::vector<std::pair<int, int>> miss;  // (row count, e)
    for (int e = 0; e < E; ++e) if (!by_e[e].empty() && store_.slot_of(l, e) < 0 && pf_slot[e] < 0) miss.push_back({(int)by_e[e].size(), e});  // pre-copied experts go to the GPU
    std::sort(miss.begin(), miss.end());
    int n_dma = (int)(frac * (float)miss.size() + 0.5f);
    if (split_balance) {
      // G2 HIVE_PREFILL_SPLIT=balance: choose the share with the cost model (hive/prefill_split.h — predicted CPU completion ≈ predicted GPU (streaming + compute) completion). Takes precedence over the fixed values and the adaptive rule above.
      //   The CPU share rule (fewest rows first, cumulative rows ≤ w.M) is unchanged — only how many go to the CPU changes.
      split_consume(stats);  // the previous layer's GPU sample (usually done already — the host got this layer's router result, so earlier st_ work has finished)
      std::vector<ps::Miss> pm(miss.size());
      for (size_t i = 0; i < miss.size(); ++i) {
        int jobs_e = 0;  // same count as the job creation below: kMaxRows rows per (expert, sub-chunk) — refs are in sub-chunk order
        const auto& refs = by_e[miss[i].second];
        for (size_t a = 0; a < refs.size();) { size_t b = a; while (b < refs.size() && refs[b].s == refs[a].s) ++b; jobs_e += (int)((b - a + ExpertStore::kMaxRows - 1) / ExpertStore::kMaxRows); a = b; }
        pm[i] = ps::Miss{miss[i].first, jobs_e};
      }
      int gpu_fixed = 0;
      for (int e = 0; e < E; ++e) if (!by_e[e].empty() && (store_.slot_of(l, e) >= 0 || pf_slot[e] >= 0)) gpu_fixed += (int)by_e[e].size();
      const int t = split_type;
      const ps::Costs cst{split_job_ms_[t], split_rec_ms_[t], split_row_ms_[t], split_has_job_[t], split_has_rec_[t], split_has_row_[t]};
      split_dec = ps::decide(cst, pm, gpu_fixed, w.M);
      n_dma = (int)miss.size() - split_dec.n_cpu;  // the loop below re-applies the same row cap and yields the same count (capped_cpu = that loop)
      split_on_layer = true;
      if (stats) { ++stats->split_layers; stats->split_share_sum += split_dec.share; if (split_dec.cold) ++stats->split_cold; }
    }
    const int n_cpu = std::max(0, (int)miss.size() - n_dma);
    for (int i = 0; i < n_cpu; ++i) {
      if (cpu_rows_total + miss[i].first > w.M) break;
      to_cpu_e[miss[i].second] = 1;
      cpu_rows_total += miss[i].first;
    }
    // B5: with zero CPU jobs no sample (time_dma) arises — e.g. at frac 0.95 with ≤ 10 misses rounding sends everything to DMA and frac would freeze at 0.95
    if (!miss.empty() && cpu_rows_total == 0 && !fixed_prefill && !split_balance) dma_frac_relax(kind);  // H1: a fixed share does not use frac (like a fixed HIVE_DMA_FRAC); G2 does not use frac either
  }
  // G2: whether to take a GPU sample at this layer (pure copy time, resident compute time per row, GPU window) — only when no sample is pending (off = always false)
  const bool split_time = split_on_layer && !split_pending_;
  int split_nd = 0, split_comp_rows = 0;
  if (split_time && !split_g0_) {
    for (cudaEvent_t* e : {&split_g0_, &split_g1_, &split_c0_, &split_c1_}) CUDA_CHECK(cudaEventCreateWithFlags(e, cudaEventDefault));
    for (auto& pr : split_d_) for (cudaEvent_t& e : pr) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDefault));
  }
  // One staging copy (the same chain as the three call sites — L1 early issue, H1 group kCopy, plain loop: release wait → copy → record copy completion). For a G2
  //   sample only the copy is wrapped in events (recorded after the release wait, so ring back-pressure waits are excluded and only the copy itself is timed).
  auto stage_copy = [&](int si, int e) {
    if (stage_used_[si]) CUDA_CHECK(cudaStreamWaitEvent(side_, stage_freed_[si], 0));
    const bool tm = split_time && split_nd < kSplitDmaSamples;
    if (tm) CUDA_CHECK(cudaEventRecord(split_d_[split_nd][0], side_));
    store_.copy_to_staging(si, l, e, side_);
    if (tm) CUDA_CHECK(cudaEventRecord(split_d_[split_nd++][1], side_));
    CUDA_CHECK(cudaEventRecord(stage_copied_[si], side_));
  };
  // Order: pre-copied experts (in copy order) → the rest (by e). Pre-copy slots must be released first so that when later copies lap the ring and overwrite those
  //   slots, the release events they wait on are the right ones. (L1: with early issue on, this is built before the CPU share preparation — release records of
  //   unused pre-copies must be on side_ before the early-issued copies. Off = the usual place.)
  std::vector<int> order;
  std::vector<char> pf_busy(store_.staging_slots(), 0);  // pre-copy slots this layer's loop will consume (early issue must not overwrite them)
  auto build_order = [&] {
    order.reserve(E);
    if (pf_l_ == l)
      for (auto [e, si] : pf_) {
        if (e < E && !by_e[e].empty()) { order.push_back(e); pf_busy[si] = 1; }
        else CUDA_CHECK(cudaEventRecord(stage_freed_[si], side_));  // unused pre-copy — the slot can be reused once the copy ends
      }
    for (int e = 0; e < E; ++e) if (!(pf_l_ == l && pf_slot[e] >= 0)) order.push_back(e);
    pf_.clear(); pf_l_ = -1;
  };
  // L1 (opt-in HIVE_EARLY_STREAM): PCIe is idle while the host runs the CPU share preparation below (gather → D2H → sync → LUT unpack → jobs).
  //   Issue the H2D of non-resident, non-CPU experts for the first staging ring lap (up to the pre-copy slots) first, in exactly the order the loop consumes —
  //   same experts, same slots, same ring order (plan_early_stream: slot = stage_next_++ % S; the loop skips early-issued ones and continues). Computation and
  //   accumulation order (the loop's order) are unchanged, so values are the same.
  std::vector<int> early_si;  // expert → early-issue slot (-1 = none)
  if (early_stream_) {
    build_order();
    std::vector<std::pair<int, int>> plan;
    plan_early_stream(order, [&](int e) { return !by_e[e].empty() && !to_cpu_e[e] && pf_slot[e] < 0 && store_.slot_of(l, e) < 0; }, pf_busy,
                      store_.staging_slots(), stage_next_, plan);
    early_si.assign(E, -1);
    for (auto [e, si] : plan) {
      stage_copy(si, e);
      early_si[e] = si;
    }
  }
  // 2) activations of CPU rows: gather per sub-chunk on the GPU into one contiguous buffer → host copy → sync → fp32 unpack → create jobs → wake the pool (in parallel with GPU launches)
  std::vector<ExpertStore::Job>& jobs = w.cpu_jobs;
  jobs.clear();
  std::vector<int> job_rows_begin(S + 1, 0);  // gathered row range of sub-chunk s: [begin[s], begin[s+1])
  const size_t scratch_n = ExpertStore::job_scratch_floats(I);
  double t_cpu0 = now_ms();
  if (cpu_rows_total > 0) {
    int g = 0;  // gathered row index = row of cpu_out_h/cpu_rows_h/a_f_h
    for (int s = 0; s < S; ++s) {
      job_rows_begin[s] = g;
      const int g0 = g;
      for (int e = 0; e < E; ++e) { if (!to_cpu_e[e]) continue; for (const Ref& r : by_e[e]) if (r.s == s) w.cpu_rows_h[g++] = r.m; }
      if (g > g0) {
        k::gather_rows_u8(subs[s].xq_d, w.cpu_rows_d + g0, g - g0, dim, w.g_xq.as<uint8_t>() + (size_t)g0 * dim, st_);
        k::gather_rows_u8(subs[s].xs_d, w.cpu_rows_d + g0, g - g0, dim / 32, w.g_xs.as<uint8_t>() + (size_t)g0 * (dim / 32), st_);
      }
    }
    job_rows_begin[S] = g;
    HIVE_CHECK(g == cpu_rows_total && g <= w.M, "cpu rows");
    CUDA_CHECK(cudaMemcpyAsync(w.xq_h, w.g_xq.p, (size_t)g * dim, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(w.xs_h, w.g_xs.p, (size_t)g * (dim / 32), cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));  // g_xq/g_xs are rewritten by the GPU loop below — only after the copy has finished
    t_cpu0 = now_ms();  // the adaptive rule's CPU time starts here (activations arrived) — including GPU gather and copy would overestimate the CPU share
    // L1: row-independent table lookup — with early issue on, split over the pool thread count (same bytes; the pool is idle here). Off = sequential.
    par_rows(g, early_stream_ ? host_par_threads_ : 1, kParMinRows, [&](int i0, int i1) { unpack_acts(w.xq_h, w.xs_h, i0, i1, dim, w.a_f_h, w.a_s_h); });
    g = 0;  // walk again in the same order to build jobs (one expert × one sub-chunk = one job, ≤ kMaxRows rows each)
    for (int s = 0; s < S; ++s)
      for (int e = 0; e < E; ++e) {
        if (!to_cpu_e[e]) continue;
        ExpertStore::Job jb{};
        jb.layer = l; jb.e = e; jb.R = 0; jb.scratch = w.scratch_h + (size_t)g * scratch_n;
        int rows_here = 0;
        for (const Ref& r : by_e[e]) {
          if (r.s != s) continue;
          if (jb.R == ExpertStore::kMaxRows) { jobs.push_back(jb); jb = ExpertStore::Job{}; jb.layer = l; jb.e = e; jb.R = 0; jb.scratch = w.scratch_h + (size_t)g * scratch_n; }
          jb.a_f[jb.R] = w.a_f_h + (size_t)g * dim;
          jb.a_s[jb.R] = w.a_s_h + (size_t)g * (dim / 32);
          jb.route_w[jb.R] = subs[s].rw_h[r.m * k + r.j];
          jb.out[jb.R] = w.cpu_out_h + (size_t)g * dim;
          ++jb.R; ++g; ++rows_here;
        }
        if (jb.R > 0) jobs.push_back(jb);
        if (stats) { stats->n_routed += rows_here; stats->n_cpu += rows_here; }
      }
    store_.start_jobs(jobs);
  }
  const bool time_dma = !split_on_layer && !dma_evt_pending_ && !jobs.empty();  // if G2 decided this layer, no adaptive-rule sample is taken (its frac is unused)
  if (time_dma) CUDA_CHECK(cudaEventRecord(dma_e0_, st_));
  if (split_time) CUDA_CHECK(cudaEventRecord(split_g0_, st_));
  // 3) GPU: resident (compute right away); streaming = staging ring (8): copy (side) → wait for completion → compute → release event. Per expert one GEMM per sub-chunk in order — the record moves once.
  //   g_rows_h/g_rw_h are written by the host and read asynchronously by kernels, so they are not reused within a layer (cumulative cursor hoff, capacity g_rows_cap).
  //   g_xq/g_xs (device) restart from 0 per expert and sub-chunk — same stream, so they are overwritten only after the previous GEMM has read them.
  int hoff = 0;
  auto run_gpu_expert = [&](const uint8_t* rec, int s, int off, int n) {
    const int32_t* rows_d = w.g_rows_d + off;
    uint8_t* gxq = w.g_xq.as<uint8_t>();
    uint8_t* gxs = w.g_xs.as<uint8_t>();
    k::gather_rows_u8(subs[s].xq_d, rows_d, n, dim, gxq, st_);
    k::gather_rows_u8(subs[s].xs_d, rows_d, n, dim / 32, gxs, st_);
    k::gemm_bs(gxq, gxs, rec + lay.w1, rec + lay.s1, true, n, I, dim, w.gate.as<bf16>(), nullptr, st_);
    k::gemm_bs(gxq, gxs, rec + lay.w3, rec + lay.s3, true, n, I, dim, w.up.as<bf16>(), nullptr, st_);
    k::swiglu_route(w.gate.as<bf16>(), w.up.as<bf16>(), w.g_rw_d + off, n, I, c.swiglu_limit, w.y.as<bf16>(), st_);
    k::act_quant_fp8(w.y.as<bf16>(), n, I, w.yq.as<uint8_t>(), w.ys.as<uint8_t>(), st_);
    k::gemm_bs(w.yq.as<uint8_t>(), w.ys.as<uint8_t>(), rec + lay.w2, rec + lay.s2, true, n, dim, I, w.eout.as<bf16>(), nullptr, st_);
    k::accum_bf16_rows(w.eout.as<bf16>(), rows_d, n, dim, subs[s].acc_d, st_);
  };
  static const bool tile_group_gemm = env_on("HIVE_TILE_GROUP_GEMM");
  auto run_joined_expert = [&](const uint8_t* rec, const std::vector<Ref>& refs) {
    // Bounded by the existing scratch row capacity: joining subchunks does not
    // increase startup VRAM. Gather/compute/scatter stay ordered on st_.
    struct Part { int s, off, begin, n; };
    for (size_t i=0;i<refs.size();) {
      const int begin_off=hoff;
      int n=0;
      std::vector<Part> parts;
      while(i<refs.size() && n<w.M) {
        const int s=refs[i].s, off=hoff, begin=n;
        while(i<refs.size() && refs[i].s==s && n<w.M) {
          const Ref& r=refs[i++];w.g_rows_h[hoff]=r.m;w.g_rw_h[hoff]=subs[s].rw_h[r.m*k+r.j];++hoff;++n;
        }
        const int count=n-begin;
        k::gather_rows_u8(subs[s].xq_d,w.g_rows_d+off,count,dim,w.g_xq.as<uint8_t>()+(size_t)begin*dim,st_);
        k::gather_rows_u8(subs[s].xs_d,w.g_rows_d+off,count,dim/32,w.g_xs.as<uint8_t>()+(size_t)begin*(dim/32),st_);
        parts.push_back({s,off,begin,count});
      }
      k::gemm_bs(w.g_xq.as<uint8_t>(),w.g_xs.as<uint8_t>(),rec+lay.w1,rec+lay.s1,true,n,I,dim,w.gate.as<bf16>(),nullptr,st_);
      k::gemm_bs(w.g_xq.as<uint8_t>(),w.g_xs.as<uint8_t>(),rec+lay.w3,rec+lay.s3,true,n,I,dim,w.up.as<bf16>(),nullptr,st_);
      k::swiglu_route(w.gate.as<bf16>(),w.up.as<bf16>(),w.g_rw_d+begin_off,n,I,c.swiglu_limit,w.y.as<bf16>(),st_);
      k::act_quant_fp8(w.y.as<bf16>(),n,I,w.yq.as<uint8_t>(),w.ys.as<uint8_t>(),st_);
      k::gemm_bs(w.yq.as<uint8_t>(),w.ys.as<uint8_t>(),rec+lay.w2,rec+lay.s2,true,n,dim,I,w.eout.as<bf16>(),nullptr,st_);
      for(const auto& p:parts) k::accum_bf16_rows(w.eout.as<bf16>()+(size_t)p.begin*dim,w.g_rows_d+p.off,p.n,dim,subs[p.s].acc_d,st_);
    }
  };
  if (!early_stream_) build_order();  // the usual place (with early issue it was built above)
  // B6: the resident hits of one prefill layer form one batch — calling touch() (LRU clock +1) per hit would spread slot times in expert-id order and drag
  //   victim order along with ids. Take one time per layer (lru_stamp) and mark all hits with it (expert_store.h touch_at — protection against being picked as
  //   a place() victim within the same layer is kept: that time is greater than every earlier time).
  const uint64_t lru_stamp = store_.lru_stamp();
  // H1 HIVE_GROUPED_PREFILL (opt-in, env_on): run the per-expert loop below as batches — one launch per stage (grouped GEMM w1‖w3, swiglu+quantization,
  //   grouped GEMM w2, fixed-order accumulation). Same experts, same weights, same activation quantization, and every seg (one GEMM call of the loop) has the
  //   same row count, so the kernel choice gemm_bs would make (n ≤ 8 gemv, otherwise mx tile/WMMA — gemm_bs_route) is kept. Accumulation adds once per
  //   destination row in the loop's expert order (order) → **bit-identical to the per-expert loop** (test_grouped_prefill). Possible differences: none (only
  //   GPU kernel execution order and overlap differ).
  //   Streaming: slot numbers and copy order equal the loop's (gp::plan_batches). A batch = compute resident segs first → wait for copies of that batch's
  //   streaming slots → streaming segs → accumulate → release events. If a slot held by the batch needs a new copy, the batch runs first (the "release record
  //   before the next copy" invariant); streaming per batch ≤ half the ring. Paths where the row-map encoding does not fit (sub-chunks < 128, rows < 2^24) or
  //   no grouped variant exists (e.g. only HIVE_MX_PREFILL on) run the per-expert loop for that layer (fallback — never rejected).
  static const bool grouped_prefill = env_on("HIVE_GROUPED_PREFILL");
  bool grouped = grouped_prefill && S < 128 && k::GroupedMoe::supported(dim, I);
  if (grouped) for (const auto& sc : subs) if (sc.M >= (1 << 24)) grouped = false;
  if (grouped) {
    const bool joined = tile_group_gemm && S > 1;  // same segs as run_joined_expert (refs in chunks of w.M); otherwise contiguous ranges per sub-chunk (run_gpu_expert)
    std::vector<gp::Expert> gex;
    std::vector<int> gex_e, seg_rows, seg_i0;
    for (int e : order) {
      const auto& refs = by_e[e];
      if (refs.empty() || to_cpu_e[e]) continue;
      gp::Expert X{};
      X.seg_begin = (int)seg_rows.size();
      if (joined) for (size_t i = 0; i < refs.size(); i += (size_t)w.M) { seg_i0.push_back((int)i); seg_rows.push_back((int)std::min(refs.size() - i, (size_t)w.M)); }
      else for (size_t i = 0; i < refs.size();) {
        size_t j = i;
        while (j < refs.size() && refs[j].s == refs[i].s) ++j;
        seg_i0.push_back((int)i); seg_rows.push_back((int)(j - i)); i = j;
      }
      X.seg_end = (int)seg_rows.size();
      if (pf_slot[e] >= 0) { X.kind = gp::kPrecopied; X.si = pf_slot[e]; }
      else if (!early_si.empty() && early_si[e] >= 0) { X.kind = gp::kPrecopied; X.si = early_si[e]; }
      else X.kind = store_.slot_of(l, e) >= 0 ? gp::kResident : gp::kStream;
      gex.push_back(X); gex_e.push_back(e);
    }
    const int SS = store_.staging_slots();
    std::vector<gp::Action> acts;
    gp::plan_batches(gex, seg_rows, w.M, std::max(1, SS / 2), SS, stage_next_, acts);  // stage_next_ advances by 1 per streaming expert (as in the per-expert loop)
    if (!w.gmoe || w.gmoe->max_subs() < S) w.gmoe = std::make_unique<k::GroupedMoe>(w.M, std::max(S, std::max(1, prefill_slots())));
    std::vector<const uint8_t*> xq_sub(S), xs_sub(S);
    std::vector<float*> acc_sub(S);
    std::vector<int> sub_rows(S);
    for (int s = 0; s < S; ++s) { xq_sub[s] = subs[s].xq_d; xs_sub[s] = subs[s].xs_d; acc_sub[s] = subs[s].acc_d; sub_rows[s] = subs[s].M; }
    const k::GroupedBufs gb{w.gate.as<bf16>(), w.up.as<bf16>(), w.y.as<bf16>(), w.eout.as<bf16>(), w.yq.as<uint8_t>(), w.ys.as<uint8_t>(),
                            w.g_xq.as<uint8_t>(), w.g_xs.as<uint8_t>()};
    std::vector<k::GroupedSeg> bsegs;
    std::vector<k::GroupedRef> brefs;
    std::vector<int> wait_si, free_si, x_si(gex.size(), -1);
    int res_rows = 0;  // G2: resident (non-late) rows of the batch being built
    std::vector<const uint8_t*> x_rec(gex.size(), nullptr);
    for (const gp::Action& a : acts) {
      if (a.type == gp::kCopy) {  // same chain as the loop's streaming branch (side stream)
        const int si = a.si;
        stage_copy(si, gex_e[a.x]);
        x_si[a.x] = si;
      } else if (a.type == gp::kSeg) {
        const gp::Expert& X = gex[a.x];
        const int e = gex_e[a.x];
        const auto& refs = by_e[e];
        if (a.seg == X.seg_begin) {  // first seg of an expert: stats, LRU marking, record (same as the expert head in the per-expert loop)
          if (stats) stats->n_routed += (int)refs.size();
          if (X.kind == gp::kResident) {
            const int slot = store_.slot_of(l, e);
            if (stats) stats->n_hit += (int)refs.size();
            store_.touch_at(slot, lru_stamp);
            x_rec[a.x] = store_.dev_rec(slot);
          } else {
            if (stats) { stats->n_streamed++; stats->n_dma_rows += refs.size(); }
            if (X.kind == gp::kPrecopied) x_si[a.x] = X.si;
            x_rec[a.x] = store_.staging_rec(x_si[a.x]);
          }
        }
        const uint8_t* rec = x_rec[a.x];
        const bool late = X.kind != gp::kResident;
        const int i0 = seg_i0[a.seg], n = seg_rows[a.seg];
        bsegs.push_back(k::GroupedSeg{rec + lay.w1, rec + lay.s1, rec + lay.w3, rec + lay.s3, rec + lay.w2, rec + lay.s2, (int)brefs.size(), n, late});
        for (int t = i0; t < i0 + n; ++t) { const Ref& r = refs[t]; brefs.push_back(k::GroupedRef{r.s, r.m, subs[r.s].rw_h[r.m * k + r.j]}); }
        if (late) { wait_si.push_back(x_si[a.x]); if (a.last) free_si.push_back(x_si[a.x]); }
        else res_rows += n;
      } else {
        // G2 compute sample: wrap only the front part (resident segs — before the copy wait) of the first batch with resident rows. Without a back part, record after run (including the fixed-order accumulation)
        const bool tc = split_time && split_comp_rows == 0 && res_rows > 0;
        bool c1_done = false;
        if (tc) CUDA_CHECK(cudaEventRecord(split_c0_, st_));
        w.gmoe->run(bsegs.data(), (int)bsegs.size(), brefs.data(), xq_sub.data(), xs_sub.data(), acc_sub.data(), S, sub_rows.data(), dim, I, c.swiglu_limit, gb,
                    st_, [&] {
                      if (tc) { CUDA_CHECK(cudaEventRecord(split_c1_, st_)); c1_done = true; }
                      for (int si : wait_si) CUDA_CHECK(cudaStreamWaitEvent(st_, stage_copied_[si], 0));
                    });
        if (tc) { if (!c1_done) CUDA_CHECK(cudaEventRecord(split_c1_, st_)); split_comp_rows = res_rows; }
        for (int si : free_si) { CUDA_CHECK(cudaEventRecord(stage_freed_[si], st_)); stage_used_[si] = 1; }
        bsegs.clear(); brefs.clear(); wait_si.clear(); free_si.clear(); res_rows = 0;
      }
    }
  } else
  for (int e : order) {
    const auto& refs = by_e[e];
    if (refs.empty() || to_cpu_e[e]) continue;
    if (stats) stats->n_routed += (int)refs.size();
    HIVE_CHECK(hoff + (int)refs.size() <= w.g_rows_cap, "gather table capacity");
    const int slot = store_.slot_of(l, e);
    const uint8_t* rec = nullptr;
    int si = -1;
    if (pf_slot[e] >= 0) {  // pre-copied (was not resident — promotion after a pre-copy cannot happen during prefill)
      if (stats) { stats->n_streamed++; stats->n_dma_rows += refs.size(); }
      si = pf_slot[e];
      CUDA_CHECK(cudaStreamWaitEvent(st_, stage_copied_[si], 0));
      rec = store_.staging_rec(si);
    } else if (!early_si.empty() && early_si[e] >= 0) {  // L1 early-issued (the copy was issued above — only wait for completion here)
      if (stats) { stats->n_streamed++; stats->n_dma_rows += refs.size(); }
      si = early_si[e];
      CUDA_CHECK(cudaStreamWaitEvent(st_, stage_copied_[si], 0));
      rec = store_.staging_rec(si);
    } else if (slot >= 0) {
      if (stats) stats->n_hit += (int)refs.size();
      store_.touch_at(slot, lru_stamp);
      rec = store_.dev_rec(slot);
    } else {
      if (stats) { stats->n_streamed++; stats->n_dma_rows += refs.size(); }
      si = stage_next_++ % store_.staging_slots();
      stage_copy(si, e);
      CUDA_CHECK(cudaStreamWaitEvent(st_, stage_copied_[si], 0));
      rec = store_.staging_rec(si);
    }
    const bool tc = split_time && split_comp_rows == 0 && si < 0;  // G2 compute sample: the first resident expert (no copy wait)
    if (tc) CUDA_CHECK(cudaEventRecord(split_c0_, st_));
    if (tile_group_gemm && S>1) run_joined_expert(rec,refs);
    else for (size_t i = 0; i < refs.size();) {
      const int s = refs[i].s, off = hoff;
      int n = 0;
      while (i < refs.size() && refs[i].s == s) { w.g_rows_h[hoff] = refs[i].m; w.g_rw_h[hoff] = subs[s].rw_h[refs[i].m * k + refs[i].j]; ++hoff; ++n; ++i; }
      run_gpu_expert(rec, s, off, n);
    }
    if (tc) { CUDA_CHECK(cudaEventRecord(split_c1_, st_)); split_comp_rows = (int)refs.size(); }
    if (si >= 0) { CUDA_CHECK(cudaEventRecord(stage_freed_[si], st_)); stage_used_[si] = 1; }
  }
  if (time_dma) CUDA_CHECK(cudaEventRecord(dma_e1_, st_));
  if (split_time) {
    CUDA_CHECK(cudaEventRecord(split_g1_, st_));
    split_pending_ = true; split_type_ = split_type; split_nd_ = split_nd; split_comp_rows_ = split_comp_rows;
    split_pred_ok_ = !split_dec.cold; split_pred_gpu_ = split_dec.pred_gpu_ms;
  }
  pmark("moe.gpu_experts");
  // 4) wait for CPU experts → sum per sub-chunk
  if (!jobs.empty()) {
    const double twait = now_ms();
    store_.wait_jobs();
    if (stats) stats->ms_cpu_wait += now_ms() - twait;
    // Completion time is the one stamped by the pool: the host reaches this point only once the GPU is nearly done, because the launch loop above (thousands per
    //   layer) blocks on the CUDA queue — measuring with now_ms() gives cpu_ms ≈ gpu_ms and the adaptive rule stalls at 0.5 (measured: GPU 3.6 s vs CPU 0.5–0.9 s on a 2048 chunk, yet the share did not move).
    const double cpu_ms = std::max(0.0, store_.jobs_done_ms() - t_cpu0);
    if (time_dma) { dma_evt_pending_ = true; dma_evt_kind_ = dma_kind; dma_evt_cpu_ms_ = cpu_ms; }  // R2 dma_kind (off = kDmaStream)
    if (stats) stats->ms_cpu_span += cpu_ms;
    if (split_on_layer) {  // G2 CPU sample: wall-clock share of one job (pool completion / job count); predicted vs actual on layers that had a prediction
      ps::ema(split_job_ms_[split_type], split_has_job_[split_type], cpu_ms / (double)jobs.size());
      if (stats && !split_dec.cold) { stats->split_pred_cpu_ms += split_dec.pred_cpu_ms; stats->split_act_cpu_ms += cpu_ms; ++stats->split_cpu_n; }
    }
    for (int s = 0; s < S; ++s) {
      const int b = job_rows_begin[s], n = job_rows_begin[s + 1] - b;
      if (n > 0) k::accum_f32_rows(w.cpu_out_d + (size_t)b * dim, w.cpu_rows_d + b, n, dim, subs[s].acc_d, st_);
    }
  }
  pmark("moe.cpu_experts");
}

// ---------------------------------------------------------------------------------------------------------------------
// Batched decode
// ---------------------------------------------------------------------------------------------------------------------
// G1 HIVE_DECODE_STEP_GRAPH — one forward_batch step as a single CUDA graph (see the comment at the top of hive/decode_handshake.h).
//   Per-layer path: per layer [host tables] → graph A → cudaStreamSynchronize → host moe_decode_experts (classification, observe, CPU jobs, group table H2D, eager launch).
//   This variant: (1) at the step head, write the host tables of every layer (engram_host, attention_decode_host) into per-layer mapped slots (StepGraph::Tabs)
//          ahead of time — the per-layer path overwrites one slot each layer, which is why it needs a sync per layer. Host-side inputs (sequence positions,
//          buffers, tokens) do not change during the step, so values equal the per-layer calls.
//          (2) graph = [layer l: decode_front (graph A body) → k::hs_layer_experts (plan → resident groups → DMA wait → DMA groups → CPU wait → CPU accumulation)] × nL + head.
//          (3) after a single wait at the end of the step (blocking event), replay observe, xtrace, touch, miss lists, stats and DMA-share adaptation in layer order.
//   Math: with the same inputs (slot table, DMA share) each row computes the same expert on the same device with the same kernel — classification follows the
//     same rules (hs::plan_host and the device hs_plan agree; miss sorting uses the same algorithm as libstdc++ std::sort — test_decode_handshake_cpu).
//     Resident and DMA groups use the same kernels and row order as the fuse_/use_mx chain (mx_grouped_w13 (row map + quantization) → mx_grouped_w2 →
//     accum_bf16_rows_seq) — HIVE_DECODE_FUSED (D1) is not used on this path (test_decode_moe confirms D1 is bit-identical to that chain). One difference:
//     CPU result accumulation uses fixed job-row-order addition instead of accum_f32_rows (atomic adds — with two or more CPU results per row the order
//     varied between runs) (one of those possible orders — deterministic).
//   Cache decisions: nothing inside a step reads the results of observe/touch (scores, LRU) (promotion and eviction happen only in promote_after_step and the
//     step-head commit_pending) — the replay makes the same calls in the same order, so the state promote_after_step sees is the same. Residency times are the
//     same too (the slot table changes only between steps).
//   DMA-share adaptation (dma_frac_): the per-layer path folds in the previous layer's sample, changing the share of the next layer within the same step. This
//     variant uses the step-head frac for the whole step (host values cannot change inside a graph) and replays the same rule (sample → relax → sample) in
//     layer order after the step — samples are execution times, so even the per-layer path is not reproducible here; with a fixed HIVE_DMA_FRAC both paths use the same share.
//   First use (step key = M and per-layer Tb) runs the per-layer path (same reason as run_graph — kernel lazy init outside capture). The second use captures, later uses replay.
//   Exceptions and timeouts: the GPU wait has a cap (HIVE_DECODE_STEP_TIMEOUT_MS, default 10000 — only a safety cap so a stuck CPU pool never holds the GPU
//     forever, not a measured value); beyond it a device error flag is set → after the step the host raises an exception (BatchGuard marks the sequences broken — same handling as other host errors).
struct Runtime::StepGraph {
  int Mb = 0, nL = 0, k = 0, Gh = 0, Gd = 0, Rcap = 0;
  hs::BoxLayout bl{};
  uint8_t* box_h = nullptr; uint8_t* box_d = nullptr;   // mailbox (mapped pinned)
  hs::Params* prm_h = nullptr;                           // pinned (H2D every step)
  int32_t* slot_pin = nullptr;                           // slot table H2D staging (pinned)
  std::vector<int32_t> slot_shadow;
  bool slot_valid = false;
  DevBuf prm_d, misc_d, cnt_d, gd_d, rows_d, rw_d, cpu_rows_d, slot_d;
  // per-layer mapped tables (swapped into Work's pointers of the same names per layer)
  struct Tabs {
    k::KvRow* kvrows_h = nullptr; k::KvRow* kvrows_d = nullptr;
    const uint8_t** kptrs_h = nullptr; const uint8_t** kptrs_d = nullptr;
    int32_t *trows_h = nullptr, *trows_d = nullptr, *dsti_h = nullptr, *dsti_d = nullptr, *visible_h = nullptr, *visible_d = nullptr, *gpos_h = nullptr,
            *gpos_d = nullptr;
    bf16 **ringp_h = nullptr, **ringp_d = nullptr;
    uint8_t **dstp_h = nullptr, **dstp_d = nullptr, **dstk_h = nullptr, **dstk_d = nullptr;
    float **skv_h = nullptr, **skv_d = nullptr, **ssc_h = nullptr, **ssc_d = nullptr;
    uint8_t *eg_v_h = nullptr, *eg_v_d = nullptr, *eg_s_h = nullptr, *eg_s_d = nullptr;  // engram layers only
  };
  std::vector<Tabs> tabs;
  Tabs saved{};
  bool bound = false;
  std::vector<void*> blocks;  // pinned blocks to free
  std::map<std::pair<int, std::vector<int>>, GraphEntry> graphs;  // (M, per-layer Tb) → step graph
  std::unique_ptr<hs::Dispatcher> disp;
  cudaEvent_t done_evt = nullptr;  // blocking wait at the end of the step (cudaEventBlockingSync — the whole step takes tens of ms, so wake-up latency is small)
  uint32_t seq_base = 0;
  hs::HostPlan hp;
  int warn_left = 4;
  uint64_t steps = 0;
  cudaEvent_t prof_e0 = nullptr, prof_e1 = nullptr;  // S1 instrumentation ([step-graph] line — recorded only on HIVE_PROFILE sample steps; no effect on behaviour)

  template <class T>
  T* tab(size_t n, T** dev) {
    T* p = pinned_mapped<T>(n, dev);
    blocks.push_back((void*)p);
    return p;
  }
  static void swap_in(Work& w, Tabs& t) {
    std::swap(w.kvrows_h, t.kvrows_h); std::swap(w.kvrows_d, t.kvrows_d); std::swap(w.kptrs_h, t.kptrs_h); std::swap(w.kptrs_d, t.kptrs_d);
    std::swap(w.trows_h, t.trows_h); std::swap(w.trows_d, t.trows_d); std::swap(w.dsti_h, t.dsti_h); std::swap(w.dsti_d, t.dsti_d);
    std::swap(w.visible_h, t.visible_h); std::swap(w.visible_d, t.visible_d); std::swap(w.gpos_h, t.gpos_h); std::swap(w.gpos_d, t.gpos_d);
    std::swap(w.ringp_h, t.ringp_h); std::swap(w.ringp_d, t.ringp_d); std::swap(w.dstp_h, t.dstp_h); std::swap(w.dstp_d, t.dstp_d);
    std::swap(w.dstk_h, t.dstk_h); std::swap(w.dstk_d, t.dstk_d); std::swap(w.skv_h, t.skv_h); std::swap(w.skv_d, t.skv_d);
    std::swap(w.ssc_h, t.ssc_h); std::swap(w.ssc_d, t.ssc_d);
    std::swap(w.eg_v_h, t.eg_v_h); std::swap(w.eg_v_d, t.eg_v_d); std::swap(w.eg_s_h, t.eg_s_h); std::swap(w.eg_s_d, t.eg_s_d);
  }
  // Put layer l's tables into Work (not idempotent — bind/unbind pairs). Layers without engram slots keep Work's own engram buffers (those layers do not use engram).
  void bind(Work& w, int l) {
    unbind(w);
    Tabs t = tabs[(size_t)l];
    if (!t.eg_v_h) { t.eg_v_h = w.eg_v_h; t.eg_v_d = w.eg_v_d; t.eg_s_h = w.eg_s_h; t.eg_s_d = w.eg_s_d; }
    saved = t;
    swap_in(w, saved);  // saved ← Work's original values
    bound = true;
  }
  void unbind(Work& w) {
    if (!bound) return;
    swap_in(w, saved);
    bound = false;
  }
  ~StepGraph() {
    disp.reset();
    for (auto& [key, g] : graphs) if (g.exec) cudaGraphExecDestroy(g.exec);
    for (void* p : blocks) cudaFreeHost(p);
    for (void* p : {(void*)box_h, (void*)prm_h, (void*)slot_pin}) if (p) cudaFreeHost(p);
    if (done_evt) cudaEventDestroy(done_evt);
    for (cudaEvent_t e : {prof_e0, prof_e1}) if (e) cudaEventDestroy(e);
  }
};

namespace {
constexpr int kStepTimeoutMsDefault = 10000;  // safety cap (see the header comment above) — the default is not a tuned value
}  // namespace

bool Runtime::forward_batch_step(std::vector<Seq*>& seqs, int M, const std::function<void()>& head_body, ForwardStats* stats) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  static const bool use_mx = getenv("HIVE_MX") ? atoi(getenv("HIVE_MX")) != 0 : true;  // same formula as moe_decode_experts
  const int nL = model_.n_loaded_layers(), k = c.n_act;
  const int Mb = std::max(1, std::min(opt_.max_batch, w.M));
  // Outside the conditions take the per-layer path (layer loop): graphs off, not the fused/block-scale variant (the base chain differs), CPU misses off (DMA share = E,
  //   exceeds the staging ring), dump/inject, no staging, or layer shapes beyond the plan kernel limits
  if (!graphs_ || !fuse_ || !use_mx || !opt_.cpu_for_misses || !opt_.dump_dir.empty() || !opt_.inject_dir.empty() || verify_ || M > Mb ||
      store_.staging_slots() < 1 || M * k > hs::kMaxR)
    return false;
  for (int l = 0; l < nL; ++l) {
    const LayerWeights& L = model_.layer(l);
    const int E = L.n_routed ? L.n_routed : c.n_routed, kl = L.n_act ? L.n_act : c.n_act;
    if (E > hs::kMaxE || kl != k || (size_t)(l + 1) * store_.E() > store_.slot_table_size()) return false;
  }
  // per-layer Tb (graph key) — same formula as decode_layer (Tmax of attention_decode_host = max_m (pos+1)/ratio, index source layers only)
  std::vector<int> Tb(nL, 0);
  for (int l = 0; l < nL; ++l)
    if (c.ratio(l) && c.is_index_source(l)) {
      int Tmax = 0;
      for (int m = 0; m < M; ++m) Tmax = std::max(Tmax, (int)((seqs[m]->pos + 1) / c.ratio(l)));
      Tb[l] = std::min(w.Tcap, std::max(1024, next_pow2(Tmax)));
    }
  if (!sg_) {
    try {
      auto g = std::make_shared<StepGraph>();
      g->Mb = Mb; g->nL = nL; g->k = k;
      g->Gh = Mb * k; g->Gd = hs::kMaxDma * ((Mb + hs::kGroupRows - 1) / hs::kGroupRows); g->Rcap = Mb * k;
      g->bl = hs::BoxLayout::make(Mb, k, c.dim, nL);
      g->box_h = pinned_mapped<uint8_t>(g->bl.total, &g->box_d);
      memset(g->box_h, 0, g->bl.total);
      g->prm_h = pinned<hs::Params>(1);
      memset((void*)g->prm_h, 0, sizeof(hs::Params));
      g->slot_pin = pinned<int32_t>(store_.slot_table_size());
      g->slot_d.alloc(store_.slot_table_size() * 4);
      g->prm_d.alloc(sizeof(hs::Params));
      g->misc_d.alloc(sizeof(hs::DevMisc));
      CUDA_CHECK(cudaMemset(g->misc_d.p, 0, sizeof(hs::DevMisc)));
      g->cnt_d.alloc(sizeof(hs::LayerCounts));
      g->gd_d.alloc(sizeof(k::GroupDesc) * (size_t)(g->Gh + g->Gd));
      g->rows_d.alloc((size_t)g->Rcap * 4); g->rw_d.alloc((size_t)g->Rcap * 4); g->cpu_rows_d.alloc((size_t)g->Rcap * 4);
      g->tabs.resize(nL);
      const int cols = eh_ ? c.n_hash_cols() : 0, hd = c.engram_head_dim;
      for (int l = 0; l < nL; ++l) {
        StepGraph::Tabs& t = g->tabs[l];
        t.kvrows_h = g->tab<k::KvRow>(Mb, &t.kvrows_d); t.kptrs_h = g->tab<const uint8_t*>(Mb, &t.kptrs_d);
        t.trows_h = g->tab<int32_t>(Mb, &t.trows_d); t.dsti_h = g->tab<int32_t>(Mb, &t.dsti_d);
        t.visible_h = g->tab<int32_t>(Mb, &t.visible_d); t.gpos_h = g->tab<int32_t>(Mb, &t.gpos_d);
        t.ringp_h = g->tab<bf16*>(Mb, &t.ringp_d); t.dstp_h = g->tab<uint8_t*>(Mb, &t.dstp_d); t.dstk_h = g->tab<uint8_t*>(Mb, &t.dstk_d);
        t.skv_h = g->tab<float*>(Mb, &t.skv_d); t.ssc_h = g->tab<float*>(Mb, &t.ssc_d);
        if (model_.layer(l).engram && cols > 0) {
          t.eg_v_h = g->tab<uint8_t>((size_t)Mb * cols * hd, &t.eg_v_d); t.eg_s_h = g->tab<uint8_t>((size_t)Mb * cols * hd / 32, &t.eg_s_d);
        }
      }
      CUDA_CHECK(cudaEventCreateWithFlags(&g->done_evt, cudaEventBlockingSync | cudaEventDisableTiming));
      hs::Dispatcher::Io io;
      io.store = &store_;
      io.ctrl = reinterpret_cast<hs::Ctrl*>(g->box_h + g->bl.ctrl);
      io.hdr = reinterpret_cast<const hs::PostHdr*>(g->box_h + g->bl.hdr);
      io.jobs = reinterpret_cast<const hs::JobItem*>(g->box_h + g->bl.jobs);
      io.xq = g->box_h + g->bl.xq; io.xs = g->box_h + g->bl.xs;
      io.a_f = w.a_f_h; io.a_s = w.a_s_h; io.scratch = w.scratch_h; io.cpu_out = w.cpu_out_h;  // the engine thread does not use these buffers during the step
      io.dim = c.dim; io.inter = c.moe_inter; io.Mb = Mb; io.Jcap = g->bl.Jcap; io.nL = nL;
      int dev = 0;
      CUDA_CHECK(cudaGetDevice(&dev));
      io.thread_init = [dev] { cudaSetDevice(dev); };
      io.dma_copy = [this](int l, int e, int si) {
        if (cw_on_ && (size_t)l < cw_used_.size() && !(cw_used_[l] & 5)) { CUDA_CHECK(cudaEventRecord(cw_ev_[3 * l], side_)); cw_used_[l] |= 4; }  // E4 instrumentation (before the layer's first copy; bit2 = only the front mark recorded)
        store_.copy_to_staging(si, l, e, side_);
      };
      uint32_t* flag = &reinterpret_cast<hs::DevMisc*>(g->misc_d.p)->dma_flag;
      StepGraph* gp = g.get();
      io.dma_signal = [this, flag, gp](uint32_t seq) {
        const int l = (int)(seq - gp->seq_base) - 1;  // published number = seq_base + l + 1 (decode_handshake.h Params)
        if (cw_on_ && l >= 0 && (size_t)l < cw_used_.size() && (cw_used_[l] & 4)) { CUDA_CHECK(cudaEventRecord(cw_ev_[3 * l + 1], side_)); cw_used_[l] = 1; }
        k::hs_signal(flag, seq, side_);
      };
      const int nL_pump = nL;
      io.after_dma = [this, nL_pump](int l) { promo_pump(l, nL_pump); };  // E4 HIVE_DECODE_COPY_PRIO (off = returns immediately)
      // Polling: default = spin while armed (replaces the engine thread's per-layer spin sync). With HIVE_DECODE_STEP_POLL_US=N (>0), after HIVE_DECODE_STEP_SPIN_US
      //   (default 0) µs without a publish, poll while sleeping N µs at a time (power ↔ CPU expert start latency — A/B after startup). Non-numeric or negative = 0 (off).
      if (const char* v = getenv("HIVE_DECODE_STEP_POLL_US"); v && *v) io.poll_us = std::max(0, atoi(v));
      if (io.poll_us > 0) { const char* v = getenv("HIVE_DECODE_STEP_SPIN_US"); io.spin_us = v && *v ? std::max(0, atoi(v)) : 0; }
      k::hs_preload(side_, st_);  // measured fix (see the hs_preload comment in decode_handshake.h): first-step lazy module loading deadlock (0x2) — load kernels before the spin wait
      g->disp = std::make_unique<hs::Dispatcher>(std::move(io));
      sg_ = std::move(g);
      fprintf(stderr, "[runtime] decode step graph on: Mb %d · layers %d · mailbox %.1f KiB · poll %s\n", Mb, nL, sg_->bl.total / 1024.0,
              getenv("HIVE_DECODE_STEP_POLL_US") && atoi(getenv("HIVE_DECODE_STEP_POLL_US")) > 0 ? "sleep" : "spin");
    } catch (const std::exception& ex) {  // setup failure (pinned, VRAM) = turn off only this feature and take the per-layer path (fallback)
      (void)cudaGetLastError();
      fprintf(stderr, "[runtime] ⚠️decode step graph off: %s\n", ex.what());
      step_graph_ = false;
      return false;
    }
  }
  StepGraph& g = *sg_;
  GraphEntry& ge = g.graphs[std::make_pair(M, Tb)];
  if (!ge.exec && ge.uses++ < 1) return false;  // first use = per-layer path (kernel lazy init outside capture — same reason as run_graph)
  const double th0 = now_ms();
  hs::Ctrl* ctrl = reinterpret_cast<hs::Ctrl*>(g.box_h + g.bl.ctrl);
  // (1) host tables of every layer (per-layer slots) — same functions and layer order as step 1) of decode_layer (engram history and shared_topk_rows_ advance in the same order)
  struct BindGuard { StepGraph& g; Work& w; ~BindGuard() { g.unbind(w); } } bind_guard{g, w};
  for (int l = 0; l < nL; ++l) {
    const LayerWeights& L = model_.layer(l);
    g.bind(w, l);
    if (L.engram) engram_host(L, M, w.ids_h, &seqs, nullptr);
    const int Tmax = attention_decode_host(seqs, L, l, M);
    HIVE_CHECK(Tb[l] == ((c.ratio(l) && c.is_index_source(l)) ? std::min(w.Tcap, std::max(1024, next_pow2(Tmax))) : 0), "step graph Tb");
  }
  g.unbind(w);
  const double th1 = now_ms();  // S1 instrumentation: end of the step-head host tables (40 layers) — the GPU idles after the previous step meanwhile (the per-layer path overlaps them with the GPU)
  // (2) step parameters: DMA share table (same formula, step-head frac), published sequence numbers, staging slots, time cap, slot table (if changed)
  dma_frac_consume();  // fold in any sample left by a per-layer-path step first
  const float frac = dma_frac_fixed_ >= 0.f ? dma_frac_fixed_ : dma_frac_[kDmaDecode];
  const int dma_cap = 8;  // per-layer DMA cap of decode (same as the per-layer path)
  hs::Params& P = *g.prm_h;
  for (int n = 0; n <= hs::kMaxE; ++n) {
    const int n_miss = n;
    P.share[n] = n_miss >= 3 ? std::min(std::min(dma_cap, store_.staging_slots()), std::max(opt_.decode_gpu_share, (int)(frac * n_miss + 0.5f))) : 0;
  }
  g.seq_base += (uint32_t)nL + 1u;
  P.seq_base = g.seq_base;
  P.n_staging = store_.staging_slots();
  P.stage_base = (int)(stage_next_ % (uint32_t)P.n_staging);
  {
    static const long tmo = [] { const char* v = getenv("HIVE_DECODE_STEP_TIMEOUT_MS"); const long x = v && *v ? atol(v) : 0; return x > 0 ? x : (long)kStepTimeoutMsDefault; }();
    P.timeout_ns = (unsigned long long)tmo * 1000000ull;
  }
  const size_t nkeys = store_.slot_table_size();
  if (!g.slot_valid || memcmp(g.slot_shadow.data(), store_.slot_table(), nkeys * 4) != 0) {
    g.slot_shadow.assign(store_.slot_table(), store_.slot_table() + nkeys);
    memcpy(g.slot_pin, g.slot_shadow.data(), nkeys * 4);
    CUDA_CHECK(cudaMemcpyAsync(g.slot_d.p, g.slot_pin, nkeys * 4, cudaMemcpyHostToDevice, st_));
    g.slot_valid = true;
  }
  CUDA_CHECK(cudaMemcpyAsync(g.prm_d.p, g.prm_h, sizeof(hs::Params), cudaMemcpyHostToDevice, st_));
  ctrl->err = 0; ctrl->err_seq = 0; ctrl->cpu_err = 0;  // the GPU has not run this step yet (the previous step was waited for to the end)
  for (int si = 0; si < store_.staging_slots(); ++si)
    if (stage_used_[si]) CUDA_CHECK(cudaStreamWaitEvent(side_, stage_freed_[si], 0));  // prerequisite for staging copies the dispatcher issues on side_ (the per-layer path does the same wait per copy)
  const double th2 = now_ms();  // S1 instrumentation: end of parameters, slot table and staging waits
  // (3) the graph (captured once at the second use) — the dispatcher is woken before the launch
  const ExpertLayout& lay = store_.layout();
  const hs::RecOff rec{lay.w1, lay.s1, lay.w3, lay.s3, lay.w2, lay.s2, lay.total};
  // D1 HIVE_DECODE_FUSED (same switch formula as moe_decode_experts): when on, groups run in the fused kernel (device-count variant). The per-layer conditions
  //   (shape, alignment of every group record) are structural here — record = slot/staging base + slot index × total (multiple of 4096) + matrix offset, so checking the bases and offsets decides every group alike.
  static const bool decode_fused = env_on("HIVE_DECODE_FUSED");
  auto al = [](uintptr_t v, uintptr_t a) { return (v & (a - 1)) == 0; };
  const bool fused = decode_fused && k::moe_decode_fused_ok(c.dim, c.moe_inter, M, g.Gh) && k::moe_decode_fused_ok(c.dim, c.moe_inter, M, g.Gd) &&
                     al((uintptr_t)store_.dev_rec(0), 16) && al((uintptr_t)store_.staging_rec(0), 16) && al(lay.total, 16) && al(lay.w1, 16) && al(lay.w3, 16) &&
                     al(lay.w2, 16) && al(lay.s1, 4) && al(lay.s3, 4) && al(lay.s2, 4);
  auto layer_args = [&](int l) {
    const LayerWeights& L = model_.layer(l);
    k::HsLayerArgs a{};
    k::HsPlanArgs& p = a.plan;
    p.ids = w.ids.as<int32_t>(); p.rw = w.rw.as<float>(); p.M = M; p.k = k; p.E = L.n_routed ? L.n_routed : c.n_routed; p.l = l;
    p.slot_row = g.slot_d.as<int32_t>() + (size_t)l * store_.E();
    p.slots_base = store_.dev_rec(0); p.staging_base = store_.staging_rec(0); p.rec = rec;
    p.xq = w.xq.as<uint8_t>(); p.xs = w.xs.as<uint8_t>(); p.dim = c.dim;
    p.prm = reinterpret_cast<const hs::Params*>(g.prm_d.p); p.misc = reinterpret_cast<hs::DevMisc*>(g.misc_d.p);
    p.cnt = reinterpret_cast<hs::LayerCounts*>(g.cnt_d.p); p.gd = g.gd_d.as<k::GroupDesc>(); p.Gh = g.Gh; p.Gd = g.Gd;
    p.g_rows = g.rows_d.as<int32_t>(); p.g_rw = g.rw_d.as<float>(); p.cpu_rows = g.cpu_rows_d.as<int32_t>();
    p.ctrl = reinterpret_cast<hs::Ctrl*>(g.box_d + g.bl.ctrl); p.hdr = reinterpret_cast<hs::PostHdr*>(g.box_d + g.bl.hdr);
    p.jobs = reinterpret_cast<hs::JobItem*>(g.box_d + g.bl.jobs); p.bxq = g.box_d + g.bl.xq; p.bxs = g.box_d + g.bl.xs;
    p.route_log_l = reinterpret_cast<int32_t*>(g.box_d + g.bl.route_log) + (size_t)l * Mb * k;
    p.counts_log_l = reinterpret_cast<hs::LayerCounts*>(g.box_d + g.bl.counts_log) + l;
    p.time_log_l = reinterpret_cast<unsigned long long*>(g.box_d + g.bl.time_log) + 2 * (size_t)l;
    a.I = c.moe_inter; a.limit = c.swiglu_limit;
    a.y = w.y.as<bf16>(); a.yq = w.yq.as<uint8_t>(); a.ys = w.ys.as<uint8_t>(); a.eout = w.eout.as<bf16>(); a.acc = w.acc.as<float>();
    a.cpu_out_d = w.cpu_out_d;
    a.fused = fused;
    return a;
  };
  auto body = [&] {
    k::hs_step_reset(reinterpret_cast<hs::DevMisc*>(g.misc_d.p), st_);
    for (int l = 0; l < nL; ++l) {
      g.bind(w, l);
      decode_front(l, M, Tb[l], false);  // hs_plan moves activations to the mailbox only when there are CPU jobs
      k::hs_layer_experts(layer_args(l), st_);
    }
    g.unbind(w);
    head_body();
  };
  struct ArmGuard {
    StepGraph& g; cudaStream_t st; bool on = false;
    ~ArmGuard() {
      if (!on) return;
      cudaStreamSynchronize(st);  // even when leaving by exception, wait until the launched step ends (the dispatcher keeps answering) — then let it rest
      (void)cudaGetLastError();
      g.disp->disarm();
    }
  } arm_guard{g, st_};
  g.disp->arm();
  arm_guard.on = true;
  if (!ge.exec) {
    cudaGraph_t graph = nullptr;
    capturing_ = true;
    CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
    try {
      body();
    } catch (...) {  // same as run_graph: close the capture and discard it
      cudaGraph_t bad = nullptr;
      cudaStreamEndCapture(st_, &bad);
      if (bad) cudaGraphDestroy(bad);
      cudaGetLastError();
      capturing_ = false;
      throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(st_, &graph));
    capturing_ = false;
    CUDA_CHECK(cudaGraphInstantiate(&ge.exec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
  }
  // S1 instrumentation ([step-graph] line — to tell why G1 was slow in service: serial host time at step boundaries vs GPU span inside the graph). Sample steps only.
  const bool sg_prof = profile_ && !(profile_every_ > 1 && step_ % profile_every_ != 0);
  if (sg_prof) {
    if (!g.prof_e0) { CUDA_CHECK(cudaEventCreateWithFlags(&g.prof_e0, cudaEventDefault)); CUDA_CHECK(cudaEventCreateWithFlags(&g.prof_e1, cudaEventDefault)); }
    CUDA_CHECK(cudaEventRecord(g.prof_e0, st_));
  }
  const double tl0 = now_ms();
  CUDA_CHECK(cudaGraphLaunch(ge.exec, st_));
  const double tl1 = now_ms();
  if (sg_prof) CUDA_CHECK(cudaEventRecord(g.prof_e1, st_));
  if (stats) stats->ms_host += now_ms() - th0;
  pmark("step_graph");
  CUDA_CHECK(cudaEventRecord(g.done_evt, st_));
  CUDA_CHECK(cudaEventSynchronize(g.done_evt));  // once per step (OS wait)
  const double tw1 = now_ms();
  arm_guard.on = false;
  g.disp->disarm();
  ++g.steps;
  if (ctrl->err != 0 || ctrl->cpu_err != 0) {
    cudaStreamSynchronize(side_);
    throw std::runtime_error(std::string("decode step graph: ") + (ctrl->cpu_err ? "CPU expert dispatcher error" : "") +
                             (ctrl->err & 1u ? " CPU expert wait timed out" : "") + (ctrl->err & 2u ? " DMA wait timed out" : "") + " (post " +
                             std::to_string(ctrl->err_seq) + ")");
  }
  // (4) replay: the host side effects of moe_decode_experts in layer order — observe (scores, policy, 'U' trace) → xtrace → miss list → DMA-share adaptation → touch (LRU) → stats
  const int32_t* route_log = reinterpret_cast<const int32_t*>(g.box_h + g.bl.route_log);
  const hs::LayerCounts* counts_log = reinterpret_cast<const hs::LayerCounts*>(g.box_h + g.bl.counts_log);
  const unsigned long long* time_log = reinterpret_cast<const unsigned long long*>(g.box_h + g.bl.time_log);
  const auto& dlog = g.disp->log();
  const float score_weight = opt_.phase_score && score_prefill_ ? 1.f / std::max(1, M) : 1.f;
  if (stats) { stats->row_hit.resize(M); stats->row_cpu.resize(M); stats->row_dma.resize(M); }
  auto frac_apply = [&](double cpu_ms, double gpu_ms) {  // the update formula of dma_frac_consume (kind = decode)
    float& f = dma_frac_[kDmaDecode];
    if (cpu_ms > gpu_ms * 1.2) f = std::min(kDmaFracHi[kDmaDecode], f + kDmaFracStep);
    else if (gpu_ms > cpu_ms * 1.2) f = std::max(kDmaFracLo[kDmaDecode], f - kDmaFracStep);
  };
  bool pend = false;
  double pend_cpu = 0, pend_gpu = 0;
  uint32_t n_dma_total = 0;
  hs::HostPlan& hp = g.hp;
  for (int l = 0; l < nL; ++l) {
    const LayerWeights& L = model_.layer(l);
    const int E = L.n_routed ? L.n_routed : c.n_routed, R = M * k;
    const int32_t* ids = route_log + (size_t)l * Mb * k;
    for (int i = 0; i < R; ++i) store_.observe(l, ids[i], score_weight, i / k);
    xtrace_layer(l, M, k, ids);
    hs::plan_host(ids, nullptr, M, k, E, store_.slot_table() + (size_t)l * store_.E(), P.share, hp);  // same build_tables as the device hs_plan
    if (memcmp(&hp.counts, &counts_log[l], sizeof(hs::LayerCounts)) != 0 && g.warn_left > 0) {  // device plan ≠ host replay (e.g. stale slot-table copy) — warning only
      --g.warn_left;
      const hs::LayerCounts& d = counts_log[l];
      fprintf(stderr, "[step-graph] ⚠️layer %d plan mismatch: dev hit %d/%d dma %d/%d jobs %d/%d · host hit %d/%d dma %d/%d jobs %d/%d\n", l, d.ng_hit, d.R_hit,
              d.ng_str, d.R_str, d.n_jobs, d.job_rows, hp.counts.ng_hit, hp.counts.R_hit, hp.counts.ng_str, hp.counts.R_str, hp.counts.n_jobs,
              hp.counts.job_rows);
    }
    if (stats) stats->n_routed += R;
    if (opt_.promote_misses > 0) for (int x = 0; x < hp.n_miss_e; ++x) step_miss_.push_back(l * c.n_routed + hp.miss_e[x]);
    if (pend) { frac_apply(pend_cpu, pend_gpu); pend = false; }
    if (hp.n_miss >= 3 && (hp.n_str_e == 0 || hp.n_cpu_e == 0)) dma_frac_relax(kDmaDecode);
    const hs::Dispatcher::LayerLog& lg = dlog[(size_t)l];
    if (hp.n_str_e > 0 && hp.counts.n_jobs > 0 && lg.layer == l && lg.cpu_ms >= 0 && time_log[2 * l + 1] >= time_log[2 * l]) {
      pend = true; pend_cpu = lg.cpu_ms; pend_gpu = (double)(time_log[2 * l + 1] - time_log[2 * l]) / 1e6;
    }
    for (int x = 0; x < hp.n_hit_e; ++x) {
      const int e = hp.hit_e[x];
      store_.touch(store_.slot_of(l, e));
      if (stats) { stats->n_hit += hp.cnt[e]; for (int i = hp.off[e]; i < hp.off[e + 1]; ++i) ++stats->row_hit[hp.rows_by_e[i] / k]; }
    }
    for (int x = 0; x < hp.n_str_e; ++x) {
      const int e = hp.str_e()[x];
      ++n_dma_total;
      if (stats) { stats->n_streamed++; stats->n_dma_rows += hp.cnt[e]; for (int i = hp.off[e]; i < hp.off[e + 1]; ++i) ++stats->row_dma[hp.rows_by_e[i] / k]; }
    }
    if (stats) for (int x = 0; x < hp.n_cpu_e; ++x) {
      const int e = hp.cpu_e()[x];
      stats->n_cpu += hp.cnt[e];
      for (int i = hp.off[e]; i < hp.off[e + 1]; ++i) ++stats->row_cpu[hp.rows_by_e[i] / k];
    }
    if (stats && hp.counts.n_jobs > 0 && lg.layer == l && lg.cpu_ms >= 0) stats->ms_cpu_span += lg.cpu_ms;
    if (l == c.cand_source_layer && c.ratio(l) && c.is_index_source(l) && Tb[l] > 0) have_candidates_ = true;  // same restoration as decode_layer
  }
  if (pend) frac_apply(pend_cpu, pend_gpu);
  // staging ring: release and copy events of the used slots (the per-layer path records them per layer — here only later, after the step), next slot
  for (uint32_t j = 0; j < n_dma_total; ++j) {
    const int si = (int)((stage_next_ + j) % (uint32_t)store_.staging_slots());
    CUDA_CHECK(cudaEventRecord(stage_copied_[si], side_));
    CUDA_CHECK(cudaEventRecord(stage_freed_[si], st_));
    stage_used_[si] = 1;
  }
  stage_next_ += n_dma_total;
  if (sg_prof) {  // host: tables = (1), params = (2), launch = cudaGraphLaunch call, wait = after launch → woken from the blocking wait, replay = (4) | gpu = graph span
    float span = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&span, g.prof_e0, g.prof_e1));
    fprintf(stderr, "[step-graph M=%d] host tables %.2f · params %.2f · launch %.2f · wait %.2f · replay %.2f ms (host before launch %.2f) | gpu span %.2f ms · posts %llu\n",
            M, th1 - th0, th2 - th1, tl1 - tl0, tw1 - tl1, now_ms() - tw1, tl0 - th0, span, (unsigned long long)g.disp->posts());
  }
  return true;
}

void Runtime::forward_batch(std::vector<Seq*>& seqs, const int32_t* ids, std::vector<int32_t>& next,
                            std::vector<std::vector<float>>* logits_out, ForwardStats* stats) {
  ElasticScope::lend_if_idle(*this);  // Q1 HIVE_CACHE_ELASTIC: lend the elastic slots once a multi-chunk prefill has finished (off, already lent or prefill in progress = does nothing)
  for (auto* seq : seqs) if (seq->snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, seq->snapshot_fence->event, 0));
  const int M = (int)seqs.size();
  HIVE_CHECK(M >= 1 && M <= opt_.max_batch && M <= w_->M, "batch size");
  xtrace_step(0, M, seqs.data(), M);
  const Config& c = model_.cfg();
  Work& w = *w_;
  const double t0 = now_ms();
  store_.commit_pending();
  static const bool cw_trace = env_on("HIVE_TRACE_CACHE");  // E4 demand-copy instrumentation: same switch as the [cache] line below, or HIVE_PROFILE sample steps (same decision as pmark)
  cw_begin(cw_trace || (profile_ && !(profile_every_ > 1 && step_ % profile_every_ != 0)));
  pmark("start");
  for (int m = 0; m < M; ++m) {
    HIVE_CHECK(seqs[m]->pos + 1 <= seqs[m]->cap, "context overflow");
    w.ids_h[m] = ids[m];
    w.pos_h[m] = (int32_t)seqs[m]->pos;
    w.is_image_h[m] = 0;
    memcpy(w.emb_h + (size_t)m * c.dim, model_.embed_host() + (size_t)ids[m] * c.dim, (size_t)c.dim * 2);
  }
  engram_ssd_hint(ids, nullptr, M, seqs.data(), nullptr);  // HIVE_ENGRAM_SSD read-ahead (off = no-op)
  pmark("embed");
  shared_comp_kv_ = nullptr; shared_index_k_ = nullptr; shared_topk_cols_ = 0; have_candidates_ = false;
  shared_topk_rows_.assign(M, 0);
  batch_ = &seqs;
  struct BatchGuard { Runtime* r; std::vector<Seq*>* ss; int ex; ~BatchGuard() {
    r->batch_ = nullptr;
    if (std::uncaught_exceptions() > ex) for (Seq* q : *ss) q->broken = true;  // an exception mid-layer leaves compressor state and engram history half advanced
  } } bguard{this, &seqs, std::uncaught_exceptions()};
  const int nL = model_.n_loaded_layers();
  const bool head = model_.has_head() && nL == c.n_layers;
  // O1: [decode-host] instrumentation (sample steps), HOST_FAST state, UBATCH conditions (M ≥ 2, no dump/inject/expert tracing — otherwise the base layer loop)
  const bool ub = ubatch_ && M >= 2 && opt_.dump_dir.empty() && opt_.inject_dir.empty() && !xtrace_;
  dov_->pre_l = -1; dov_->dma_armed = false;
  dov_->begin(profile_ && !(profile_every_ > 1 && step_ % profile_every_ != 0), nL, ub);
  if (host_fast_) pin_engine_thread_once();
  auto head_body = [&] {  // last layer tail + head
    layer_tail(nL - 1, M);
    if (!head) return;
    k::hc_pre(w.h.as<bf16>(), w.pre_mix.as<float>(), M, c.hc, c.dim, w.x.as<bf16>(), st_);
    k::rmsnorm(w.x.as<bf16>(), model_.norm(), c.norm_eps, M, c.dim, w.xn.as<bf16>(), st_);
    k::bf16_to_f32(w.xn.as<bf16>(), M * c.dim, w.xf.as<float>(), st_);
    k::head_logits(w.xf.as<float>(), model_.head(), M, c.vocab, c.dim, w.logits_b.as<float>(), st_);
    k::argmax_rows(w.logits_b.as<float>(), M, c.vocab, w.next_b.as<int32_t>(), st_);
    CUDA_CHECK(cudaMemcpyAsync(w.next_b_h, w.next_b.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st_));
    sampler_cands_dev(M);
  };
  // G1 HIVE_DECODE_STEP_GRAPH: when on and the conditions hold, all layers + head run as one step graph (forward_batch_step). Otherwise the layer loop + head graph.
  const bool stepped = step_graph_ && forward_batch_step(seqs, M, head_body, stats);
  if (!stepped && ub) decode_layers_ubatch(seqs, M, stats);  // O1 C
  else if (!stepped) for (int l = 0; l < nL; ++l) decode_layer(seqs, l, M, stats);  // the embedding goes into layer 0's graph, each layer tail into the next layer's graph
  batch_ = nullptr;
  for (int m = 0; m < M; ++m) { seqs[m]->tokens.push_back(ids[m]); seqs[m]->pos += 1; }
  next.assign(M, -1);
  if (!stepped) { dov_->mark(dov_->slot(nL, 0), DecodeOverlap::kPre, st_); run_graph(nL, M, 0, head_body); }  // O1 A: before the head launch (the last layer's next)
  if (!head && logits_out) logits_out->assign(M, std::vector<float>());
  if (head) {
    if (logits_out) {
      logits_out->assign(M, std::vector<float>(c.vocab));
      for (int m = 0; m < M; ++m)
        CUDA_CHECK(cudaMemcpyAsync((*logits_out)[m].data(), w.logits_b.as<float>() + (size_t)m * c.vocab, (size_t)c.vocab * 4,
                                   cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    for (int m = 0; m < M; ++m) next[m] = w.next_b_h[m];
    if (!opt_.dump_dir.empty()) dump("logits.f32", w.logits_b.p, (size_t)M * c.vocab * 4);
  }
  pmark("head");
  if (mtp_on_ && head) { mtp_sync_batch(seqs, M); pmark("mtp.sync"); }
  CUDA_CHECK(cudaStreamSynchronize(st_));
  if (!opt_.dump_dir.empty()) {  // final state (for comparing the graph path with eager execution — written in both modes)
    dump_final_ = true;
    dump("h_final.bf16", w.h.p, (size_t)M * c.hc * c.dim * 2);
    if (head && graphs_) dump("logits.f32", w.logits_b.p, (size_t)M * c.vocab * 4);
    dump_final_ = false;
  }
  cw_collect();  // E4 (after the st_ sync, before promotion decisions — only copies that overlapped this step)
  preport(M);
  dh_report(M);  // O1 A (sample steps only — after the st_ sync)
  bm_report(M);  // B2 [decode-miss] (when on — same sample step)
  if (opt_.promote_per_token > 0 || opt_.promote_misses > 0) {
    int n = promote_after_step();
    if (stats) stats->n_promoted += n;
  }
  static const bool trace = env_on("HIVE_TRACE_CACHE");
  if (trace && stats) {
    size_t vfree = 0, vtot = 0;
    if ((step_ & 63) == 0) cudaMemGetInfo(&vfree, &vtot);  // VRAM headroom every 64 steps (tracks growth from graph instances and fragmentation)
    fprintf(stderr, "[cache] step %d M=%d routed %d hit %d cpu %d streamed %d · resident %d pending %d / %d%s%s · wall %.2f ms\n", step_, M, stats->n_routed,
            stats->n_hit, stats->n_cpu, stats->n_streamed, store_.n_resident(), store_.n_pending(), store_.n_slots(),
            vtot ? (" · vram free " + std::to_string(vfree >> 20) + " MiB").c_str() : "", cw_line_.c_str(), now_ms() - t0);  // wall time of this step (monitor RE_STEP_WALL)
  }
  cw_line_.clear();  // E4: this step's summary belongs only to this step's line (never appended to a following prefill preport)
  if (stats) stats->ms_total += now_ms() - t0;
  ++step_;
}

// Row = sequence (one token). Applies the reference decode path (start_pos>0, seqlen=1) per row.
// Host half: writes only the row tables (mapped pinned). Returns the maximum indexer key count (Tmax; 0 unless an index source layer).
int Runtime::attention_decode_host(std::vector<Seq*>& seqs, const LayerWeights& L, int l, int M) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int ratio = c.ratio(l);
  int src = -1;
  for (int x : c.kv_source_layers) if (x <= l) src = std::max(src, x);
  for (int m = 0; m < M; ++m) {
    w.kvrows_h[m].ring = seqs[m]->ring[l].as<bf16>();
    w.kvrows_h[m].comp = nullptr;
    w.kvrows_h[m].comp_len = 0;
    w.kvrows_h[m].topk = 0;
    w.ringp_h[m] = seqs[m]->ring[l].as<bf16>();
  }
  int Tmax = 0;
  if (!ratio) return 0;
  HIVE_CHECK(src >= 0, "compressed kv source missing");
  for (int m = 0; m < M; ++m) {
    const int64_t p = seqs[m]->pos;
    w.kvrows_h[m].comp = seqs[m]->comp_cache[src].as<uint8_t>();
    w.kvrows_h[m].comp_len = (int)((p + 1) / ratio);
    w.kptrs_h[m] = seqs[m]->idx_k_cache[src].as<uint8_t>();
    w.trows_h[m] = w.kvrows_h[m].comp_len;
    w.visible_h[m] = (int32_t)((p + 1) / ratio);
  }
  if (c.is_kv_source(l)) {
    for (int m = 0; m < M; ++m) {
      const int64_t p = seqs[m]->pos;
      w.gpos_h[m] = (int32_t)(p + 1 - ratio);           // first position of the group (reference freqs_cis[start_pos + 1 - ratio])
      w.dsti_h[m] = (int32_t)(p / ratio);                // cache index (start_pos // ratio)
      w.dstp_h[m] = seqs[m]->comp_cache[l].as<uint8_t>();
      if (ratio > 1) { w.skv_h[m] = seqs[m]->comp_state_kv[l].as<float>(); w.ssc_h[m] = seqs[m]->comp_state_score[l].as<float>(); }
      if (c.is_index_source(l)) w.dstk_h[m] = seqs[m]->idx_k_cache[l].as<uint8_t>();
    }
  }
  if (c.is_index_source(l)) {
    for (int m = 0; m < M; ++m) Tmax = std::max(Tmax, w.trows_h[m]);
    HIVE_CHECK(Tmax <= w.Tcap && M <= w.Msub, "index T cap");
    for (int m = 0; m < M; ++m) shared_topk_rows_[m] = std::min(c.index_topk, w.trows_h[m]);
  }
  for (int m = 0; m < M; ++m) w.kvrows_h[m].topk = shared_topk_rows_[m];
  return Tmax;
}

// Device half: kernel arguments depend only on (layer, M, Tmax) → capturable as a graph. Tmax may be a bucket (≥ the real value) —
// columns beyond per-row T (trows) and visible become -inf/−1, and the attention column count is always win+index_topk (read clipped by per-row topk).
void Runtime::attention_decode_dev(const LayerWeights& L, int l, int M, int Tmax) {
  const Config& c = model_.cfg();
  const AttnWeights& A = L.attn;
  Work& w = *w_;
  const int dim = c.dim, H = c.n_heads, D = c.head_dim, rd = c.rd, win = c.window;
  const int ratio = c.ratio(l);
  const float2* freqs = ratio ? model_.rope_compress() : model_.rope_window();
  // D2 HIVE_DECODE_ATTN_FUSED (env_on): run the front below (q/kv → compressor → indexer → sparse attention) with the fewer launches of attn_decode_fused.cu —
  //   bit-identical to the fuse_ variant (test_decode_attn). Launches per layer at M=1: compressed-consumer layers 5 → 3, candidate-pool index layers 15 → 7,
  //   kv + index source layers 26 → 12 (wo_a and wo_b are shared). The first kernel copies the mapped pinned row table to the device so later kernels do not read PCIe again.
  //   HIVE_IDX_TC (typically on) is followed as is — candidate-pool layers use the same packed queries + indexer_scores_cand_tc. Switch parsing uses the same formula as the lines below.
  //   Outside the conditions (fuse_ off, M > 8, HIVE_ATTN_TC, HIVE_IDX_F32, indexer shape other than Hi 32 / Di 128) the path below runs unchanged.
  static const bool dec_attn_fused = env_on("HIVE_DECODE_ATTN_FUSED");
  static const bool dec_attn_tc = env_on("HIVE_ATTN_TC");
  static const bool dec_idx_tc = getenv("HIVE_IDX_TC") && atoi(getenv("HIVE_IDX_TC")) != 0;
  static const bool dec_idx_f32 = getenv("HIVE_IDX_F32") && atoi(getenv("HIVE_IDX_F32")) != 0;
  bool front_done = false;
  if (dec_attn_fused && fuse_ && M <= 8 && !dec_attn_tc && !dec_idx_f32 && w.dec_tab.p && (!dec_idx_tc || w.iqp.p) &&
      (!ratio || !c.is_index_source(l) || (c.index_n_heads == 32 && c.index_head_dim == 128))) {
    k::DecAttnArgs a;
    a.M = M; a.dim = dim; a.H = H; a.D = D; a.rd = rd; a.win = win; a.q_lora = c.q_lora_rank; a.Hi = c.index_n_heads; a.Di = c.index_head_dim;
    a.index_topk = c.index_topk; a.ratio = ratio; a.layer = l; a.cand_source_layer = c.cand_source_layer; a.cand_block = c.cand_block;
    a.cand_topk_blocks = c.cand_topk_blocks; a.kv_source = c.is_kv_source(l); a.index_source = c.is_index_source(l); a.have_candidates = have_candidates_;
    a.Tmax = Tmax; a.eps = c.norm_eps; a.idx_mode = dec_idx_tc ? 1 : 0; a.iqp = w.iqp.p ? w.iqp.as<uint8_t>() : nullptr;
    a.splits = (win + (ratio ? c.index_topk : 0)) > 256 ? attn_splits_ : 1;  // same rule as the sparse_attn_decode_rows call
    a.wqa = A.wq_a.w.as<uint8_t>(); a.sqa = A.wq_a.s.as<uint8_t>(); a.wqb = A.wq_b.w.as<uint8_t>(); a.sqb = A.wq_b.s.as<uint8_t>();
    a.wkv = A.wkv.w.as<uint8_t>(); a.skv = A.wkv.s.as<uint8_t>(); a.q_norm = A.q_norm.as<bf16>(); a.kv_norm = A.kv_norm.as<bf16>(); a.sink = A.attn_sink.as<float>();
    a.comp_wkv = A.comp_wkv.p; a.comp_wgate = A.comp_wgate.p ? A.comp_wgate.as<float>() : nullptr; a.comp_norm = A.comp_norm.p ? A.comp_norm.as<bf16>() : nullptr;
    a.idx_wqb = A.idx_wq_b.w.p ? A.idx_wq_b.w.as<uint8_t>() : nullptr; a.idx_sqb = A.idx_wq_b.s.p ? A.idx_wq_b.s.as<uint8_t>() : nullptr;
    a.idx_wproj = A.idx_weights_proj.p ? A.idx_weights_proj.as<bf16>() : nullptr; a.idx_wk = A.idx_wk.p ? A.idx_wk.as<bf16>() : nullptr;
    a.idx_k_norm = A.idx_k_norm.p ? A.idx_k_norm.as<bf16>() : nullptr;
    a.freqs = freqs; a.freqs_idx = model_.rope_compress();
    a.xn = w.xn.as<bf16>(); a.xf = w.xf.as<float>(); a.qr = w.qr.as<bf16>(); a.qrn = w.qrn.as<bf16>(); a.q = w.q.as<bf16>(); a.kv = w.kv.as<bf16>();
    a.qrq = w.qrq.as<uint8_t>(); a.qrs = w.qrs.as<uint8_t>(); a.ckv = w.ckv.as<float>(); a.cscore = w.cscore.as<float>(); a.cout = w.cout.as<float>();
    a.latent = w.latent.as<bf16>(); a.valid = w.valid.as<uint8_t>(); a.ik = w.ik.as<bf16>(); a.iq = w.iq.as<bf16>(); a.iw = w.iw.as<bf16>();
    a.iscore = w.iscore.as<bf16>(); a.bmax = w.bmax.as<float>(); a.cand = w.cand.p ? w.cand.as<int32_t>() : nullptr; a.idx = w.idx.as<int32_t>();
    a.idx_stride = win + c.index_topk; a.o = w.o.as<bf16>(); a.pacc = w.attn_pacc.as<float>(); a.pm = w.attn_pm.as<float>(); a.ps = w.attn_ps.as<float>();
    a.counters = w.counters.as<int>() + 12; a.attn_cnt = w.dec_cnt.as<int>(); a.tab = w.dec_tab.as<k::DecRow>();
    a.soa = reinterpret_cast<k::DecSoA*>(w.dec_tab.as<uint8_t>() + (size_t)std::max(1, std::min(opt_.max_batch, w.M)) * sizeof(k::DecRow));
    a.src = k::DecRowSrc{w.kvrows_d, w.kptrs_d, w.dstp_d, w.dstk_d, w.skv_d, w.ssc_d, w.pos_d, w.gpos_d, w.visible_d, w.trows_d, w.dsti_d};
    a.ring_ptrs = w.ringp_d;
    // R3 HIVE_DECODE_TOPK2: union buffer of the split top-k = iscore_f (the fused path uses bf16 scores, so this buffer is free — with HIVE_IDX_F32 this branch is not taken)
    a.scratch = w.iscore_f.p; a.scratch_bytes = w.iscore_f.n;
    if (k::dec_attention_front(a, blas_.get(), st_)) have_candidates_ = true;  // set at the same point as the cand_src branch of the path below
    front_done = true;
    pmark("attn.sparse");
  }
  if (!front_done) {
  if (fuse_) {  // fused q/kv chain + ring write (per-row ring pointers and positions)
    (k::decode_attn2_on() ? k::attn2_q_proj : k::decode_gemv2_on() ? k::gemv2_q_proj : k::fused_q_proj)(w.xn.as<bf16>(), dim, A.wq_a.w.as<uint8_t>(), A.wq_a.s.as<uint8_t>(), A.wq_a.N, A.q_norm.as<bf16>(), c.norm_eps, A.wq_b.w.as<uint8_t>(),
                    A.wq_b.s.as<uint8_t>(), H, D, rd, freqs, w.pos_d, M, w.qr.as<bf16>(), w.qrn.as<bf16>(), w.q.as<bf16>(), w.qrq.as<uint8_t>(),
                    w.qrs.as<uint8_t>(), w.counters.as<int>() + 0, st_);
    pmark("attn.q");
    k::fused_kv_proj(w.xn.as<bf16>(), dim, A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), D, A.kv_norm.as<bf16>(), c.norm_eps, rd, freqs, w.pos_d, M,
                     w.kv.as<bf16>(), w.ringp_d, win, w.idx.as<int32_t>(), win + c.index_topk, w.counters.as<int>() + 1, st_);
  } else {
  // q
  k::act_quant_fp8(w.xn.as<bf16>(), M, dim, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), st_);
  k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), A.wq_a.w.as<uint8_t>(), A.wq_a.s.as<uint8_t>(), false, M, A.wq_a.N, dim,
             w.qr.as<bf16>(), nullptr, st_);
  k::rmsnorm(w.qr.as<bf16>(), A.q_norm.as<bf16>(), c.norm_eps, M, c.q_lora_rank, w.qrn.as<bf16>(), st_);
  k::act_quant_fp8(w.qrn.as<bf16>(), M, c.q_lora_rank, w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), st_);
  k::gemm_bs(w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), A.wq_b.w.as<uint8_t>(), A.wq_b.s.as<uint8_t>(), false, M, A.wq_b.N,
             c.q_lora_rank, w.q.as<bf16>(), nullptr, st_);
  k::rope_last(w.q.as<bf16>(), M, H, D, rd, freqs, w.pos_d, false, st_);
  pmark("attn.q");
  // kv
  k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), false, M, D, dim, w.kv.as<bf16>(),
             nullptr, st_);
  k::rmsnorm(w.kv.as<bf16>(), A.kv_norm.as<bf16>(), c.norm_eps, M, D, w.kv.as<bf16>(), st_);
  k::rope_last(w.kv.as<bf16>(), M, 1, D, rd, freqs, w.pos_d, false, st_);
  k::act_quant_fp8_roundtrip(w.kv.as<bf16>(), M, D, st_);
  }
  const int idx_stride = win + c.index_topk;
  if (!fuse_) k::window_idxs_rows(M, win, w.pos_d, w.idx.as<int32_t>(), idx_stride, st_);
  pmark("attn.kv");
  const int topk_max_cols = ratio ? c.index_topk : 0;
  if (ratio) {
    if (c.is_kv_source(l)) {
      // one compressor step (per-row state) → latent (valid rows) → indexer keys → cache write
      if (ratio > 1) {
        k::bf16_to_f32(w.xn.as<bf16>(), M * dim, w.xf.as<float>(), st_);
        blas_->gemm_f32(w.xf.as<float>(), A.comp_wkv.as<float>(), w.ckv.as<float>(), M, D, dim);
        blas_->gemm_f32(w.xf.as<float>(), A.comp_wgate.as<float>(), w.cscore.as<float>(), M, D, dim);
        k::compressor_step_rows(w.ckv.as<float>(), w.cscore.as<float>(), M, D, ratio, w.pos_d, w.skv_d, w.ssc_d, w.cout.as<float>(),
                                w.valid.as<uint8_t>(), st_);
        k::f32_to_bf16(w.cout.as<float>(), M * D, w.latent.as<bf16>(), st_);
      } else {
        blas_->gemm_bf16(w.xn.as<bf16>(), A.comp_wkv.as<bf16>(), w.latent.as<bf16>(), M, D, dim);
        CUDA_CHECK(cudaMemsetAsync(w.valid.p, 1, (size_t)M, st_));
      }
      k::rmsnorm(w.latent.as<bf16>(), A.comp_norm.as<bf16>(), c.norm_eps, M, D, w.latent.as<bf16>(), st_);
      if (c.is_index_source(l)) {
        blas_->gemm_bf16(w.latent.as<bf16>(), A.idx_wk.as<bf16>(), w.ik.as<bf16>(), M, c.index_head_dim, D);
        k::rmsnorm(w.ik.as<bf16>(), A.idx_k_norm.as<bf16>(), c.norm_eps, M, c.index_head_dim, w.ik.as<bf16>(), st_);
        k::rope_last(w.ik.as<bf16>(), M, 1, c.index_head_dim, rd, freqs, w.gpos_d, false, st_);
        k::fp4_pack_rows(w.ik.as<bf16>(), M, c.index_head_dim, 32, false, w.dstk_d, w.dsti_d, w.valid.as<uint8_t>(), kvp::IDX_ROW, st_);
      }
      k::rope_last(w.latent.as<bf16>(), M, 1, D, rd, freqs, w.gpos_d, false, st_);
      k::fp4_pack_rows(w.latent.as<bf16>(), M, D, 16, true, w.dstp_d, w.dsti_d, w.valid.as<uint8_t>(), kvp::COMP_ROW, st_);
    }
    if (c.is_index_source(l)) {
      // indexer (per-row T ≤ Tmax)
      const int Hi = c.index_n_heads, Di = c.index_head_dim;
      if (Tmax > 0) {
        k::gemm_bs(w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), A.idx_wq_b.w.as<uint8_t>(), A.idx_wq_b.s.as<uint8_t>(), false, M, Hi * Di,
                   c.q_lora_rank, w.iq.as<bf16>(), nullptr, st_);
        k::rope_last(w.iq.as<bf16>(), M, Hi, Di, rd, model_.rope_compress(), w.pos_d, false, st_);
        static const bool idx_tc_dec = getenv("HIVE_IDX_TC") && atoi(getenv("HIVE_IDX_TC")) != 0;
        static const bool idx_f32_dec = getenv("HIVE_IDX_F32") && atoi(getenv("HIVE_IDX_F32")) != 0;  // fp32 scores (same switch as prefill)
        HIVE_CHECK(!(idx_f32_dec && idx_tc_dec), "HIVE_IDX_F32 and HIVE_IDX_TC are exclusive");
        const bool pack_dec = idx_tc_dec && c.cand_source_layer >= 0 && c.cand_source_layer < l;  // used only by candidate-pool layers (24, 28, 32, 36)
        if (pack_dec) { HIVE_CHECK(Di == kvp::IDX_D, "idx tc Di"); k::fp4_pack(w.iq.as<bf16>(), M * Hi, Di, 32, false, w.iqp.as<uint8_t>(), kvp::IDX_ROW, st_); }  // pack before the roundtrip
        k::fp4_quant_roundtrip(w.iq.as<bf16>(), M, Hi * Di, 32, false, st_);
        blas_->gemm_bf16(w.xn.as<bf16>(), A.idx_weights_proj.as<bf16>(), w.iw.as<bf16>(), M, Hi, dim);
        k::scale_bf16(w.iw.as<bf16>(), M * Hi, (1.0f / sqrtf((float)Di)) * (1.0f / sqrtf((float)Hi)), st_);
        const bool cand_src = (l == c.cand_source_layer);
        const bool uses_cand = (c.cand_source_layer >= 0 && c.cand_source_layer < l);
        const int bs = c.cand_block, nblocks = (Tmax + bs - 1) / bs, CB = c.cand_topk_blocks;
        const int topk = std::min(c.index_topk, Tmax);
        if (uses_cand) {
          // P1: only inside the candidate pool (each row with its own key cache and own T). kb is decided from the Tmax bucket and fixed together with the graph key (Tb)
          const int kbc = std::min(CB, nblocks);  // same formula at the same Tmax as layer 20 (= graph key Tb) — the captured value always matches
          HIVE_CHECK(have_candidates_ && kbc > 0, "candidates missing");
          const int ncand = kbc * bs;
          if (idx_f32_dec)
            k::indexer_scores_cand_f32(w.iq.as<bf16>(), nullptr, w.kptrs_d, w.trows_d, w.iw.as<bf16>(), M, Hi, Di, w.cand.as<int32_t>(), CB, kbc, bs,
                                       w.visible_d, w.iscore_f.as<float>(), st_);
          else if (pack_dec)
            k::indexer_scores_cand_tc(w.iqp.as<uint8_t>(), nullptr, w.kptrs_d, w.trows_d, w.iw.as<bf16>(), M, Hi, w.cand.as<int32_t>(), CB, kbc, bs,
                                      w.visible_d, w.iscore.as<bf16>(), st_);
          else
            k::indexer_scores_cand(w.iq.as<bf16>(), nullptr, w.kptrs_d, w.trows_d, w.iw.as<bf16>(), M, Hi, Di, w.cand.as<int32_t>(), CB, kbc, bs,
                                 w.visible_d, w.iscore.as<bf16>(), st_);
          if (!idx_f32_dec) k::bf16_rows_to_f32(w.iscore.as<bf16>(), M * ncand, w.iscore_f.as<float>(), st_);
          k::topk_select_rows(w.iscore_f.as<float>(), M, ncand, topk, ncand, w.topk_pos.as<int32_t>(), topk, st_);
          k::cand_to_pos(w.topk_pos.as<int32_t>(), M, topk, w.cand.as<int32_t>(), CB, bs, st_);
        } else {
          if (idx_f32_dec) k::indexer_scores_rows_f32(w.iq.as<bf16>(), w.kptrs_d, w.trows_d, Tmax, w.iw.as<bf16>(), M, Hi, Di, w.visible_d, w.iscore_f.as<float>(), st_);
          else k::indexer_scores_rows(w.iq.as<bf16>(), w.kptrs_d, w.trows_d, Tmax, w.iw.as<bf16>(), M, Hi, Di, w.visible_d, w.iscore.as<bf16>(), st_);
          if (cand_src) {
            if (idx_f32_dec) k::block_max_f32(w.iscore_f.as<float>(), M, Tmax, bs, w.visible_d, w.bmax.as<float>(), st_);
            else k::block_max(w.iscore.as<bf16>(), M, Tmax, bs, w.visible_d, w.bmax.as<float>(), st_);
            const int kb = std::min(c.cand_topk_blocks, nblocks);
            k::topk_select_rows(w.bmax.as<float>(), M, nblocks, kb, nblocks, w.topk_pos.as<int32_t>(), kb, st_);
            CUDA_CHECK(cudaMemcpy2DAsync(w.cand.p, (size_t)CB * 4, w.topk_pos.p, (size_t)kb * 4, (size_t)kb * 4, M, cudaMemcpyDeviceToDevice, st_));
                  have_candidates_ = true;
          }
          if (!idx_f32_dec) k::bf16_rows_to_f32(w.iscore.as<bf16>(), M * Tmax, w.iscore_f.as<float>(), st_);
          k::topk_select_rows_t(w.iscore_f.as<float>(), M, w.trows_d, Tmax, topk, Tmax, w.topk_pos.as<int32_t>(), topk, st_);
        }
        k::offset_idxs(w.topk_pos.as<int32_t>(), M, topk, w.visible_d, win + M, w.idx.as<int32_t>() + win, idx_stride, st_);
      }
    }
  }
  pmark("attn.compress");
  static const bool attn_tc = env_on("HIVE_ATTN_TC");  // for comparison: 3-kernel tensor-core variant
  if (attn_tc)
    k::sparse_attn_tc_rows(w.q.as<bf16>(), M, H, D, w.kvrows_d, w.kv.as<bf16>(), M, win, w.idx.as<int32_t>(), idx_stride, win + topk_max_cols,
                           A.attn_sink.as<float>(), 1.0f / sqrtf((float)D), w.attn_S.as<float>(), w.attn_mx.as<float>(), w.attn_sum.as<float>(),
                           w.o.as<bf16>(), st_);
  else
    k::sparse_attn_decode_rows(w.q.as<bf16>(), M, H, D, w.kvrows_d, w.kv.as<bf16>(), M, win, w.idx.as<int32_t>(), idx_stride,
                               win + topk_max_cols, A.attn_sink.as<float>(), 1.0f / sqrtf((float)D), w.o.as<bf16>(),
                               (win + topk_max_cols) > 256 ? attn_splits_ : 1, w.attn_pacc.as<float>(), w.attn_pm.as<float>(), w.attn_ps.as<float>(), st_,
                               (fuse_ && !attn_tc) ? freqs : nullptr, w.pos_d, rd);  // fused: the inverse RoPE is applied when storing the output
  if (!(fuse_ && !attn_tc)) k::rope_last(w.o.as<bf16>(), M, H, D, rd, freqs, w.pos_d, true, st_);
  pmark("attn.sparse");
  }  // !front_done (D2)
  {
    const int G = c.o_groups, sub = H * D / G, R = c.o_lora_rank;
    // decode: bf16 activations × original fp8 wo_a (same values, half the bytes). Prefill (M>8) uses cuBLAS bf16.
    if (M <= 8) (k::decode_gemv2_on() ? k::gemv2_bf16_fp8_grouped : k::gemv_bf16_fp8_grouped)(w.o.as<bf16>(), H * D, A.wo_a_q.w.as<uint8_t>(), A.wo_a_q.s.as<uint8_t>(), M, G, R, sub, w.og.as<bf16>(), st_);
    else
      for (int g = 0; g < G; ++g)
        k::gemm_bf16a_fp8b_tc(w.o.as<bf16>() + (size_t)g * sub, H * D, A.wo_a_q.w.as<uint8_t>() + (size_t)g * R * sub,
                              A.wo_a_q.s.as<uint8_t>() + (size_t)g * R * (sub / 32), M, R, sub, w.og.as<bf16>() + (size_t)g * R, G * R, st_);
  }
  if (fuse_) {
    (k::decode_attn2_on() ? k::attn2_gemv_quantin : k::decode_gemv2_on() ? k::gemv2_quantin : k::fused_gemv_quantin)(w.og.as<bf16>(), c.o_groups * c.o_lora_rank, A.wo_b.w.as<uint8_t>(), A.wo_b.s.as<uint8_t>(), M, dim, w.attn_out.as<bf16>(), st_);
  } else {
    k::act_quant_fp8(w.og.as<bf16>(), M, c.o_groups * c.o_lora_rank, w.ogq.as<uint8_t>(), w.ogs.as<uint8_t>(), st_);
    k::gemm_bs(w.ogq.as<uint8_t>(), w.ogs.as<uint8_t>(), A.wo_b.w.as<uint8_t>(), A.wo_b.s.as<uint8_t>(), false, M, dim,
               c.o_groups * c.o_lora_rank, w.attn_out.as<bf16>(), nullptr, st_);
    k::ring_write_rows(w.kv.as<bf16>(), M, D, win, w.pos_d, w.ringp_d, st_);
  }
  pmark("attn.wo");
  if (!opt_.dump_dir.empty()) {
    dump("attn_idx_L" + std::to_string(l) + ".i32", w.idx.p, (size_t)M * (win + c.index_topk) * 4);  // D2: idx_stride lives inside the non-fused front block
    dump("attn_out_L" + std::to_string(l) + ".bf16", w.attn_out.p, (size_t)M * dim * 2);
  }
}

// =====================================================================================================================
// DSpark (MTP) — reference DSparkBlock/forward_spec. A draft stage has the same skeleton as a trunk layer (hc → attention → hc → MoE (128/top3) → tail); differences:
//   (1) the attention window KV does not come from its own tokens but from the hc mean of the trunk target layers' (37, 38, 39) attention input → main_proj →
//       main_norm → per-stage wkv (mtp_sync writes it into the ring for every processed position; prefill = the last win positions, decode = that token)
//   (2) the B block rows ([tok, noise×(B−1)], positions pos..pos+B−1) see the whole ring + all block rows (non-causal) — get_dspark_topk_idxs
//   (3) head: last stage norm → trunk head → markov bigram bias on row i's logits (embedding of the previous output token × markov head) → argmax is the next draft
//       (drafts are greedy: verification compares against samples from the target distribution, so the draft distribution does not affect losslessness); confidence = conf_proj·[x, markov embedding]
int Runtime::target_index(int l) const {
  const auto& t = model_.cfg().dspark_targets;
  for (size_t i = 0; i < t.size(); ++i) if (t[i] == l) return (int)i;
  return -1;
}
int Runtime::mtp_block() const { return model_.cfg().dspark_block; }

void Runtime::mtp_capture(int l, int M, int64_t start_pos) {
  const int t = target_index(l);
  if (t < 0) return;
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int n = std::min(M, w.mh_rows), row0 = M - n;
  k::hc_mean_rows(w.h.as<bf16>(), c.hc, c.dim, row0, n, w.mh.as<bf16>(), ntgt_ * c.dim, t * c.dim, st_);
  if (start_pos >= 0) { mh_n_ = n; mh_pos0_ = start_pos + row0; }
}

// rows [0,n) of w.mh = target-layer hidden states of positions pos0.. → main_x → per-stage main_kv → ring (pos % win); the last row's hidden → seq.mtp_hidden
void Runtime::mtp_sync_seq(Seq& seq, int64_t pos0, int n, int row0) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  HIVE_CHECK(n >= 1 && row0 >= 0 && row0 + n <= w.mh_rows && !seq.mtp_ring.empty(), "mtp_sync rows");  // R1 row0 (default 0 = the plain condition n ≤ mh_rows)
  const int dim = c.dim, D = c.head_dim, rd = c.rd, win = c.window, K = ntgt_ * dim;
  const MtpWeights& M0 = model_.mtp(0);
  const bf16* mh = w.mh.as<bf16>() + (size_t)row0 * K;  // R1: partial rows of a multi-sequence verify (rollback_batch) — default row0 = 0 is the plain address
  k::act_quant_fp8(mh, n, K, w.mhq.as<uint8_t>(), w.mhs.as<uint8_t>(), st_);
  k::gemm_bs(w.mhq.as<uint8_t>(), w.mhs.as<uint8_t>(), M0.main_proj.w.as<uint8_t>(), M0.main_proj.s.as<uint8_t>(), false, n, dim, K,
             w.mx.as<bf16>(), nullptr, st_);
  k::rmsnorm(w.mx.as<bf16>(), M0.main_norm.as<bf16>(), c.norm_eps, n, dim, w.mx.as<bf16>(), st_);
  k::act_quant_fp8(w.mx.as<bf16>(), n, dim, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), st_);
  for (int i = 0; i < n; ++i) w.mpos_h[i] = (int32_t)(pos0 + i);
  for (int s = 0; s < model_.n_mtp(); ++s) {
    const AttnWeights& A = model_.mtp(s).L.attn;
    k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), false, n, D, dim, w.kv.as<bf16>(), nullptr, st_);
    k::rmsnorm(w.kv.as<bf16>(), A.kv_norm.as<bf16>(), c.norm_eps, n, D, w.kv.as<bf16>(), st_);
    k::rope_last(w.kv.as<bf16>(), n, 1, D, rd, model_.rope_window(), w.mpos_d, false, st_);
    k::act_quant_fp8_roundtrip(w.kv.as<bf16>(), n, D, st_);
    k::ring_write(w.kv.as<bf16>(), n, D, win, pos0, seq.mtp_ring[s].as<bf16>(), st_);
  }
  CUDA_CHECK(cudaMemcpyAsync(seq.mtp_hidden.p, mh + (size_t)(n - 1) * K, (size_t)K * 2, cudaMemcpyDeviceToDevice, st_));
  if (!opt_.dump_dir.empty()) { dump("mtp_main_x.bf16", w.mx.p, (size_t)n * dim * 2); dump("mtp_main_hidden.bf16", mh, (size_t)n * K * 2); }
  seq.mtp_hidden_valid = true;
  seq.mtp_pos = pos0 + n - 1;
}

// batched decode: row m = this token of seqs[m] (position w.pos_h[m])
void Runtime::mtp_sync_batch(std::vector<Seq*>& seqs, int M) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int dim = c.dim, D = c.head_dim, rd = c.rd, win = c.window, K = ntgt_ * dim;
  const MtpWeights& M0 = model_.mtp(0);
  for (int m = 0; m < M; ++m) HIVE_CHECK(!seqs[m]->mtp_ring.empty(), "seq without mtp rings");
  k::act_quant_fp8(w.mh.as<bf16>(), M, K, w.mhq.as<uint8_t>(), w.mhs.as<uint8_t>(), st_);
  k::gemm_bs(w.mhq.as<uint8_t>(), w.mhs.as<uint8_t>(), M0.main_proj.w.as<uint8_t>(), M0.main_proj.s.as<uint8_t>(), false, M, dim, K,
             w.mx.as<bf16>(), nullptr, st_);
  k::rmsnorm(w.mx.as<bf16>(), M0.main_norm.as<bf16>(), c.norm_eps, M, dim, w.mx.as<bf16>(), st_);
  k::act_quant_fp8(w.mx.as<bf16>(), M, dim, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), st_);
  for (int s = 0; s < model_.n_mtp(); ++s) {
    const AttnWeights& A = model_.mtp(s).L.attn;
    for (int m = 0; m < M; ++m) w.mringp_h[s][m] = seqs[m]->mtp_ring[s].as<bf16>();
    k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), false, M, D, dim, w.kv.as<bf16>(), nullptr, st_);
    k::rmsnorm(w.kv.as<bf16>(), A.kv_norm.as<bf16>(), c.norm_eps, M, D, w.kv.as<bf16>(), st_);
    k::rope_last(w.kv.as<bf16>(), M, 1, D, rd, model_.rope_window(), w.pos_d, false, st_);
    k::act_quant_fp8_roundtrip(w.kv.as<bf16>(), M, D, st_);
    k::ring_write_rows(w.kv.as<bf16>(), M, D, win, w.pos_d, w.mringp_d[s], st_);
  }
  for (int m = 0; m < M; ++m) w.mhid_h[m] = seqs[m]->mtp_hidden.as<bf16>();
  k::scatter_rows_ptrs(w.mh.as<bf16>(), M, K, w.mhid_d, st_);
  for (int m = 0; m < M; ++m) { seqs[m]->mtp_hidden_valid = true; seqs[m]->mtp_pos = w.pos_h[m]; }
}

// Draft stage attention (reference DSparkAttention decode): q/kv from the block rows (positions pos0..pos0+B−1), keys = ring slots [0,filled) + all block rows. Never writes the ring.
void Runtime::mtp_attention(Seq& seq, int s, int B, int64_t pos0, int filled) {
  const Config& c = model_.cfg();
  const AttnWeights& A = model_.mtp(s).L.attn;
  Work& w = *w_;
  const int dim = c.dim, H = c.n_heads, D = c.head_dim, rd = c.rd, win = c.window;
  const float2* freqs = model_.rope_window();
  (void)pos0;
  if (fuse_) {
    (k::decode_gemv2_on() ? k::gemv2_q_proj : k::fused_q_proj)(w.xn.as<bf16>(), dim, A.wq_a.w.as<uint8_t>(), A.wq_a.s.as<uint8_t>(), A.wq_a.N, A.q_norm.as<bf16>(), c.norm_eps, A.wq_b.w.as<uint8_t>(),
                    A.wq_b.s.as<uint8_t>(), H, D, rd, freqs, w.pos_d, B, w.qr.as<bf16>(), w.qrn.as<bf16>(), w.q.as<bf16>(), nullptr, nullptr,
                    w.counters.as<int>() + 0, st_);
    k::fused_kv_proj(w.xn.as<bf16>(), dim, A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), D, A.kv_norm.as<bf16>(), c.norm_eps, rd, freqs, w.pos_d, B,
                     w.kv.as<bf16>(), nullptr, win, nullptr, 0, w.counters.as<int>() + 1, st_);
  } else {
  k::act_quant_fp8(w.xn.as<bf16>(), B, dim, w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), st_);
  k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), A.wq_a.w.as<uint8_t>(), A.wq_a.s.as<uint8_t>(), false, B, A.wq_a.N, dim, w.qr.as<bf16>(), nullptr, st_);
  k::rmsnorm(w.qr.as<bf16>(), A.q_norm.as<bf16>(), c.norm_eps, B, c.q_lora_rank, w.qrn.as<bf16>(), st_);
  k::act_quant_fp8(w.qrn.as<bf16>(), B, c.q_lora_rank, w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), st_);
  k::gemm_bs(w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), A.wq_b.w.as<uint8_t>(), A.wq_b.s.as<uint8_t>(), false, B, A.wq_b.N, c.q_lora_rank, w.q.as<bf16>(),
             nullptr, st_);
  k::rope_last(w.q.as<bf16>(), B, H, D, rd, freqs, w.pos_d, false, st_);
  k::gemm_bs(w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), false, B, D, dim, w.kv.as<bf16>(), nullptr, st_);
  k::rmsnorm(w.kv.as<bf16>(), A.kv_norm.as<bf16>(), c.norm_eps, B, D, w.kv.as<bf16>(), st_);
  k::rope_last(w.kv.as<bf16>(), B, 1, D, rd, freqs, w.pos_d, false, st_);
  k::act_quant_fp8_roundtrip(w.kv.as<bf16>(), B, D, st_);
  }
  const int idx_stride = win + c.index_topk;
  k::dspark_idxs(B, filled, win, w.idx.as<int32_t>(), idx_stride, st_);
  // row tables (mapped pinned): the previous attention kernel has finished through that stage's router sync (or the sync of the previous forward)
  for (int m = 0; m < B; ++m) { w.kvrows_h[m].ring = seq.mtp_ring[s].as<bf16>(); w.kvrows_h[m].comp = nullptr; w.kvrows_h[m].comp_len = 0; w.kvrows_h[m].topk = B; }
  k::sparse_attn_decode_rows(w.q.as<bf16>(), B, H, D, w.kvrows_d, w.kv.as<bf16>(), B, win, w.idx.as<int32_t>(), idx_stride, win + B,
                             A.attn_sink.as<float>(), 1.0f / sqrtf((float)D), w.o.as<bf16>(), 1, w.attn_pacc.as<float>(), w.attn_pm.as<float>(),
                             w.attn_ps.as<float>(), st_, fuse_ ? freqs : nullptr, w.pos_d, rd);
  if (!fuse_) k::rope_last(w.o.as<bf16>(), B, H, D, rd, freqs, w.pos_d, true, st_);
  const int G = c.o_groups, sub = H * D / G, R = c.o_lora_rank;
  (k::decode_gemv2_on() ? k::gemv2_bf16_fp8_grouped : k::gemv_bf16_fp8_grouped)(w.o.as<bf16>(), H * D, A.wo_a_q.w.as<uint8_t>(), A.wo_a_q.s.as<uint8_t>(), B, G, R, sub, w.og.as<bf16>(), st_);
  if (fuse_) {
    (k::decode_gemv2_on() ? k::gemv2_quantin : k::fused_gemv_quantin)(w.og.as<bf16>(), G * R, A.wo_b.w.as<uint8_t>(), A.wo_b.s.as<uint8_t>(), B, dim, w.attn_out.as<bf16>(), st_);
  } else {
  k::act_quant_fp8(w.og.as<bf16>(), B, G * R, w.ogq.as<uint8_t>(), w.ogs.as<uint8_t>(), st_);
  k::gemm_bs(w.ogq.as<uint8_t>(), w.ogs.as<uint8_t>(), A.wo_b.w.as<uint8_t>(), A.wo_b.s.as<uint8_t>(), false, B, dim, G * R, w.attn_out.as<bf16>(), nullptr, st_);
  }
  if (!opt_.dump_dir.empty()) {
    dump("mtp_attn_idx_S" + std::to_string(s) + ".i32", w.idx.p, (size_t)B * idx_stride * 4);
    dump("mtp_attn_out_S" + std::to_string(s) + ".bf16", w.attn_out.p, (size_t)B * dim * 2);
  }
}

void Runtime::mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats* stats) {
  if (seq.snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, seq.snapshot_fence->event, 0));
  struct MissClear {
    std::vector<int>* v; std::vector<int>* draft; bool keep; int ex;
    ~MissClear() {
      if (keep && std::uncaught_exceptions() == ex) *draft = std::move(*v);
      v->clear();
    }
  } miss_clear{&step_miss_, &draft_miss_, opt_.mtp_cache, std::uncaught_exceptions()};
  const Config& c = model_.cfg();
  Work& w = *w_;
  HIVE_CHECK(mtp_on_ && seq.mtp_hidden_valid && seq.mtp_pos == seq.pos - 1, "mtp_draft needs the hidden of pos-1");
  { Seq* sp = &seq; xtrace_step(3, model_.cfg().dspark_block, &sp, 1); }
  const int B = c.dspark_block, dim = c.dim, hc = c.hc, V = c.vocab, rank = c.dspark_markov_rank, win = c.window;
  const int64_t pos0 = seq.pos;
  const int filled = (int)std::min<int64_t>(win, pos0);
  const double t0 = now_ms();
  store_.commit_pending();
  pmark("start");
  // draft input [tok, noise × (B−1)] at positions pos0.. (reference forward_embed)
  for (int i = 0; i < B; ++i) {
    w.ids_h[i] = i == 0 ? tok : c.dspark_noise_token;
    w.pos_h[i] = (int32_t)(pos0 + i);
    w.is_image_h[i] = 0;
    memcpy(w.emb_h + (size_t)i * dim, model_.embed_host() + (size_t)w.ids_h[i] * dim, (size_t)dim * 2);
  }
  k::embed_expand(w.emb_d, w.iota.as<int32_t>(), B, dim, hc, w.h.as<bf16>(), st_);
  k::identity_pre_mix(w.pre_mix.as<float>(), B, hc, st_);
  pmark("mtp.embed");
  for (int s = 0; s < model_.n_mtp(); ++s) {
    const LayerWeights& L = model_.mtp(s).L;
    const int l = c.n_layers + s;
    hc_attn_pre(L, B);
    mtp_attention(seq, s, B, pos0, filled);
    pmark("mtp.attn");
    hc_ffn_pre(L, B);
    moe_router_shared(L, B, opt_.cpu_for_misses);
    pmark("mtp.router");
    CUDA_CHECK(cudaStreamSynchronize(st_));
    if (!opt_.dump_dir.empty()) {
      dump_host("route_ids_L" + std::to_string(l) + ".i32", w.route_ids_h, (size_t)B * L.n_act * 4);
      dump_host("route_w_L" + std::to_string(l) + ".f32", w.rw_h, (size_t)B * L.n_act * 4);
    }
    moe_decode_experts(L, l, B, stats);
    pmark("mtp.experts");
    CUDA_CHECK(cudaStreamWaitEvent(st_, hc_join_[1], 0));
    layer_tail(l, B);
  }
  // head: hc_pre → last stage norm → trunk head → markov chain → confidence
  const MtpWeights& Wl = model_.mtp(model_.n_mtp() - 1);
  k::hc_pre(w.h.as<bf16>(), w.pre_mix.as<float>(), B, hc, dim, w.dx.as<bf16>(), st_);
  k::rmsnorm(w.dx.as<bf16>(), Wl.norm.as<bf16>(), c.norm_eps, B, dim, w.xn.as<bf16>(), st_);
  k::bf16_to_f32(w.xn.as<bf16>(), B * dim, w.xf.as<float>(), st_);
  k::head_logits(w.xf.as<float>(), model_.head(), B, V, dim, w.logits_b.as<float>(), st_);
  w.dids_h[0] = tok;
  CUDA_CHECK(cudaMemcpyAsync(w.dids.p, w.dids_h, 4, cudaMemcpyHostToDevice, st_));
  for (int i = 0; i < B; ++i) {
    k::markov_gather(Wl.markov_embed.as<bf16>(), w.dids.as<int32_t>() + i, rank, w.membed.as<bf16>() + (size_t)i * rank, st_);
    k::markov_add_bias(w.logits_b.as<float>() + (size_t)i * V, Wl.markov_head.as<bf16>(), w.membed.as<bf16>() + (size_t)i * rank, V, rank, st_);
    k::argmax_rows(w.logits_b.as<float>() + (size_t)i * V, 1, V, w.dids.as<int32_t>() + i + 1, st_);
  }
  k::conf_dot(w.dx.as<bf16>(), dim, w.membed.as<bf16>(), rank, Wl.conf_proj.as<float>(), B, w.dconf.as<float>(), st_);
  CUDA_CHECK(cudaMemcpyAsync(w.dids_h, w.dids.p, (size_t)(B + 1) * 4, cudaMemcpyDeviceToHost, st_));
  CUDA_CHECK(cudaMemcpyAsync(w.dconf_h, w.dconf.p, (size_t)B * 4, cudaMemcpyDeviceToHost, st_));
  pmark("mtp.head");
  if (!opt_.dump_dir.empty()) {
    dump("mtp_logits.f32", w.logits_b.p, (size_t)B * V * 4);
    dump("mtp_x.bf16", w.dx.p, (size_t)B * dim * 2);
    dump("mtp_markov.bf16", w.membed.p, (size_t)B * rank * 2);
  }
  CUDA_CHECK(cudaStreamSynchronize(st_));
  drafts.assign(w.dids_h + 1, w.dids_h + 1 + B);
  conf.assign(w.dconf_h, w.dconf_h + B);
  if (!opt_.dump_dir.empty()) { dump_host("mtp_ids.i32", w.dids_h, (size_t)(B + 1) * 4); dump_host("mtp_conf.f32", w.dconf_h, (size_t)B * 4); }
  preport(B);
  if (stats) stats->ms_total += now_ms() - t0;
  ++step_;
}

// =====================================================================================================================
// R1 HIVE_MTP_VERIFY2, HIVE_MTP_BATCH — verify rows on the decode path (verify_decode).
//   The base verify (forward → layer_forward, eager) runs prefill attention per layer (attention: chained q/kv, compress_source, prefill indexer,
//   sparse_attn_tc) and cannot use layer graphs, A1 q_b/wo_b, HOST_FAST or EARLY_ROUTE (measured: verify steps are so expensive that the gate rarely fires).
//   Here rows = verify rows (part = consecutive positions [p0, p0+M_p) of one sequence) and the same layer loop as forward_batch runs (decode_layer: host
//   tables → front graph → sync → moe_decode_experts). Only the three places where the decode path assumes "a different sequence per row" use verify
//   variants (verify_decode.h steps 1–3) — attention_verify_dev, attention_verify_host.
//   Numerics (vs the base verify — same weights, same experts, same activation quantization; only fp32 reduction order differs):
//   · same kernels: engram (engram_host history order = row order), hc mix (M ≤ 8 variant), kv projection (fused_kv_proj), wo_a, router and shared expert
//     (moe_router_shared), experts (moe_decode_experts — the base verify uses it too), tail, head (hc_pre, rmsnorm, head_logits, argmax, sampler candidates).
//     Window index values = window_idxs.
//   · q_a/q_b and wo_b: the A1 (HIVE_DECODE_ATTN2) variant when on (bit-identical to fused/gemv2 — test_decode_attn2).
//   · fp32 orders that may differ: (a) sparse attention = sparse_attn_decode_rows (+ fused inverse RoPE) vs sparse_attn_tc + rope_last (score, softmax sum
//     and P·V reduction order); (b) compressed latents and indexer keys of kv-source layers (2, 8, 14 with ratio 2; 20 with ratio 1): the decode variant uses
//     an M-row cuBLAS (gemm_bf16 idx_wk) and per-row rmsnorm/RoPE/fp4 pack, the base path a G-row cuBLAS over completed groups — low bits if cuBLAS picks a
//     different algorithm by row count; (c) indexer scores: indexer_scores_rows (per-row T) vs indexer_scores (full T, same key arithmetic idx_score_one) —
//     equal values, top-k via topk_select_rows_t vs topk_select_rows (same selection rule); (d) CPU/GPU placement of experts depends on the cache state at
//     that moment (same for the base verify — CPU fp32 LUT vs GPU mma low bits). → greedy accepted tokens are equal when per-row argmax is equal; flips
//     occur only when the logit margin is smaller than these differences (GPU comparison: hive --verify2-test).
//   · vs sequential decode (M=1 forward_batch) there is no (a) or (c) (same decode kernels) — what remains is the cuBLAS row count (compressor gemm_f32,
//     idx_weights_proj, idx_wk) and (d), plus the fuse_ variant instead of the D2 (HIVE_DECODE_ATTN_FUSED) / A2 (QKV3, SPARSE3) fronts (both bit-identical to the fuse_ variant — test_decode_attn/test_decode_attn3).
//   State: ring after attention; compressor in row order (rows of the same sequence carry the same state forward); rollback snapshots per part (ring slots
//     [p0, p0+M_p), ratio>1 state); compressor inputs (kv, score) are kept per layer so rollback re-applies the accepted rows as is (no re-projection); MTP
//     capture is done by decode_front on rows [0, M); draft rings and hidden states are updated by rollback_batch per part (accepted rows only) via mtp_sync_seq(row0).
//   Graph key = (layer, M + kVerifyGraphKey, Tb) — captured separately from ordinary decode (adds graphs for row counts 2..8 × Tb buckets).
int Runtime::attention_verify_host(const LayerWeights& L, int l, int M) {
  (void)L;
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int ratio = c.ratio(l);
  int src = -1;
  for (int x : c.kv_source_layers) if (x <= l) src = std::max(src, x);
  for (int m = 0; m < M; ++m) {
    Seq& s = *vseqs_[m];
    w.kvrows_h[m].ring = s.ring[l].as<bf16>();
    w.kvrows_h[m].comp = nullptr;
    w.kvrows_h[m].comp_len = 0;
    w.kvrows_h[m].topk = 0;
    w.ringp_h[m] = s.ring[l].as<bf16>();
  }
  int Tmax = 0;
  if (!ratio) return 0;
  HIVE_CHECK(src >= 0, "compressed kv source missing");
  for (int m = 0; m < M; ++m) {  // same formula as attention_decode_host (hive/verify_rows.h tab — shared with the CPU test) — only the position is this row's (w.pos_h[m]) instead of seq->pos
    Seq& s = *vseqs_[m];
    const vrows::Tab t = vrows::tab(w.pos_h[m], ratio);
    w.kvrows_h[m].comp = s.comp_cache[src].as<uint8_t>();
    w.kvrows_h[m].comp_len = t.comp_len;
    w.kptrs_h[m] = s.idx_k_cache[src].as<uint8_t>();
    w.trows_h[m] = t.trows;
    w.visible_h[m] = t.visible;
  }
  if (c.is_kv_source(l)) {
    for (int m = 0; m < M; ++m) {
      Seq& s = *vseqs_[m];
      const vrows::Tab t = vrows::tab(w.pos_h[m], ratio);
      w.gpos_h[m] = t.gpos;
      w.dsti_h[m] = t.dsti;
      w.dstp_h[m] = s.comp_cache[l].as<uint8_t>();
      if (ratio > 1) { w.skv_h[m] = s.comp_state_kv[l].as<float>(); w.ssc_h[m] = s.comp_state_score[l].as<float>(); }
      if (c.is_index_source(l)) w.dstk_h[m] = s.idx_k_cache[l].as<uint8_t>();
    }
  }
  if (c.is_index_source(l)) {
    for (int m = 0; m < M; ++m) Tmax = std::max(Tmax, w.trows_h[m]);
    HIVE_CHECK(Tmax <= w.Tcap && M <= w.Msub, "index T cap");
    for (int m = 0; m < M; ++m) shared_topk_rows_[m] = std::min(c.index_topk, w.trows_h[m]);
  }
  for (int m = 0; m < M; ++m) w.kvrows_h[m].topk = shared_topk_rows_[m];
  return Tmax;
}

// The fuse_ variant of attention_decode_dev (not the D2 front) moved verbatim, with verify variants in three places only: the kv projection writes neither the
//   ring nor window indices → window_idxs_verify; the ratio>1 compressor = compressor_step_seq (+ input kept); the ring is written after attention with
//   ring_write_rows. Switch parsing (HIVE_ATTN_TC, HIVE_IDX_TC, HIVE_IDX_F32, A1/gemv2) uses the same formulas as attention_decode_dev. (If R3 changes the
//   decode indexer/attention calls, keep this in step — the numeric contract is "the same kernels as ordinary decode".)
void Runtime::attention_verify_dev(const LayerWeights& L, int l, int M, int Tmax) {
  const Config& c = model_.cfg();
  const AttnWeights& A = L.attn;
  Work& w = *w_;
  const int dim = c.dim, H = c.n_heads, D = c.head_dim, rd = c.rd, win = c.window;
  const int ratio = c.ratio(l);
  const float2* freqs = ratio ? model_.rope_compress() : model_.rope_window();
  // Q2 HIVE_MTP_VERIFY2_FUSED (env_on — meaningful only when HIVE_MTP_VERIFY2/BATCH route here): instead of the R1 front below (the fuse_ chain), use the D2
  //   fused front (k::dec_attention_front — it also follows the HIVE_DECODE_QKV3/SPARSE3 and R3 IDXSCORE2/TOPK2 switches) in verify mode (DecAttnArgs::vgrp).
  //   Verify mode changes the same three places as R1: (1) window indices = the window_idxs_verify formula (inside the tail of the q_a‖kv kernel); (2) ratio>1
  //   compressor = compressor_step_seq (same kernel as R1, the kept input vckv at the same point); (3) ring = ring_write_rows after sparse attention (R1 does
  //   it after wo — wo touches neither ring nor kv, so the result is the same).
  //   Numerics (vs the R1 verify front): the D2, A2 and R3 variants are each bit-identical to the fuse_ variant (test_decode_attn, test_decode_attn3,
  //   test_decode_longctx), so bit identity is the contract — only launch counts and row-table placement differ (the compressed tail not writing latents of
  //   !valid rows is the same as D2 — consumers discard them by valid).
  //   GPU comparison: test_verify_decode step 5 (R1 chain vs verify-mode front, real shapes, partial batches, v3 bits 0..3). Outside the conditions the R1 chain runs (fallback — never rejected).
  //   Conditions = the same formula as attention_decode_dev's D2 condition (this switch instead of the D2 switch; the row table is allocated only when D2 is on, so w.dec_tab.p means D2 on).
  static const bool v2_fused = env_on("HIVE_MTP_VERIFY2_FUSED");
  static const bool dec_attn_tc = env_on("HIVE_ATTN_TC");
  static const bool dec_idx_tc = getenv("HIVE_IDX_TC") && atoi(getenv("HIVE_IDX_TC")) != 0;
  static const bool dec_idx_f32 = getenv("HIVE_IDX_F32") && atoi(getenv("HIVE_IDX_F32")) != 0;
  bool front_done = false;
  if (v2_fused && fuse_ && M <= 8 && !dec_attn_tc && !dec_idx_f32 && w.dec_tab.p && (!dec_idx_tc || w.iqp.p) &&
      (!ratio || !c.is_index_source(l) || (c.index_n_heads == 32 && c.index_head_dim == 128))) {
    k::DecAttnArgs a;
    a.M = M; a.dim = dim; a.H = H; a.D = D; a.rd = rd; a.win = win; a.q_lora = c.q_lora_rank; a.Hi = c.index_n_heads; a.Di = c.index_head_dim;
    a.index_topk = c.index_topk; a.ratio = ratio; a.layer = l; a.cand_source_layer = c.cand_source_layer; a.cand_block = c.cand_block;
    a.cand_topk_blocks = c.cand_topk_blocks; a.kv_source = c.is_kv_source(l); a.index_source = c.is_index_source(l); a.have_candidates = have_candidates_;
    a.Tmax = Tmax; a.eps = c.norm_eps; a.idx_mode = dec_idx_tc ? 1 : 0; a.iqp = w.iqp.p ? w.iqp.as<uint8_t>() : nullptr;
    a.splits = (win + (ratio ? c.index_topk : 0)) > 256 ? attn_splits_ : 1;  // same rule as the sparse_attn_decode_rows call
    a.wqa = A.wq_a.w.as<uint8_t>(); a.sqa = A.wq_a.s.as<uint8_t>(); a.wqb = A.wq_b.w.as<uint8_t>(); a.sqb = A.wq_b.s.as<uint8_t>();
    a.wkv = A.wkv.w.as<uint8_t>(); a.skv = A.wkv.s.as<uint8_t>(); a.q_norm = A.q_norm.as<bf16>(); a.kv_norm = A.kv_norm.as<bf16>(); a.sink = A.attn_sink.as<float>();
    a.comp_wkv = A.comp_wkv.p; a.comp_wgate = A.comp_wgate.p ? A.comp_wgate.as<float>() : nullptr; a.comp_norm = A.comp_norm.p ? A.comp_norm.as<bf16>() : nullptr;
    a.idx_wqb = A.idx_wq_b.w.p ? A.idx_wq_b.w.as<uint8_t>() : nullptr; a.idx_sqb = A.idx_wq_b.s.p ? A.idx_wq_b.s.as<uint8_t>() : nullptr;
    a.idx_wproj = A.idx_weights_proj.p ? A.idx_weights_proj.as<bf16>() : nullptr; a.idx_wk = A.idx_wk.p ? A.idx_wk.as<bf16>() : nullptr;
    a.idx_k_norm = A.idx_k_norm.p ? A.idx_k_norm.as<bf16>() : nullptr;
    a.freqs = freqs; a.freqs_idx = model_.rope_compress();
    a.xn = w.xn.as<bf16>(); a.xf = w.xf.as<float>(); a.qr = w.qr.as<bf16>(); a.qrn = w.qrn.as<bf16>(); a.q = w.q.as<bf16>(); a.kv = w.kv.as<bf16>();
    a.qrq = w.qrq.as<uint8_t>(); a.qrs = w.qrs.as<uint8_t>(); a.ckv = w.ckv.as<float>(); a.cscore = w.cscore.as<float>(); a.cout = w.cout.as<float>();
    a.latent = w.latent.as<bf16>(); a.valid = w.valid.as<uint8_t>(); a.ik = w.ik.as<bf16>(); a.iq = w.iq.as<bf16>(); a.iw = w.iw.as<bf16>();
    a.iscore = w.iscore.as<bf16>(); a.bmax = w.bmax.as<float>(); a.cand = w.cand.p ? w.cand.as<int32_t>() : nullptr; a.idx = w.idx.as<int32_t>();
    a.idx_stride = win + c.index_topk; a.o = w.o.as<bf16>(); a.pacc = w.attn_pacc.as<float>(); a.pm = w.attn_pm.as<float>(); a.ps = w.attn_ps.as<float>();
    a.counters = w.counters.as<int>() + 12; a.attn_cnt = w.dec_cnt.as<int>(); a.tab = w.dec_tab.as<k::DecRow>();
    a.soa = reinterpret_cast<k::DecSoA*>(w.dec_tab.as<uint8_t>() + (size_t)std::max(1, std::min(opt_.max_batch, w.M)) * sizeof(k::DecRow));
    a.src = k::DecRowSrc{w.kvrows_d, w.kptrs_d, w.dstp_d, w.dstk_d, w.skv_d, w.ssc_d, w.pos_d, w.gpos_d, w.visible_d, w.trows_d, w.dsti_d};
    a.ring_ptrs = w.ringp_d;
    // R3 HIVE_DECODE_TOPK2: union buffer of the split top-k = iscore_f (the fused path uses bf16 scores, so this buffer is free — with HIVE_IDX_F32 this branch is not taken)
    a.scratch = w.iscore_f.p; a.scratch_bytes = w.iscore_f.n;
    // ↑ identical text to the D2 argument fill in attention_decode_dev (compared by tools/test_verify_fused_cpu.py); ↓ verify mode only
    a.vgrp = w.vgrp_d;
    for (size_t i = 0; i < v_src2_.size(); ++i)
      if (v_src2_[i] == l) { a.vsave = w.vckv[i].as<float>(); a.vsave_rows = w.vckv[i].n / ((size_t)2 * D * 4); }  // keep rollback inputs as in R1
    if (k::dec_attention_front(a, blas_.get(), st_)) have_candidates_ = true;
    front_done = true;
    pmark("attn.sparse");
  }
  if (!front_done) {
  (k::decode_attn2_on() ? k::attn2_q_proj : k::decode_gemv2_on() ? k::gemv2_q_proj : k::fused_q_proj)(w.xn.as<bf16>(), dim, A.wq_a.w.as<uint8_t>(), A.wq_a.s.as<uint8_t>(), A.wq_a.N, A.q_norm.as<bf16>(), c.norm_eps, A.wq_b.w.as<uint8_t>(),
                  A.wq_b.s.as<uint8_t>(), H, D, rd, freqs, w.pos_d, M, w.qr.as<bf16>(), w.qrn.as<bf16>(), w.q.as<bf16>(), w.qrq.as<uint8_t>(),
                  w.qrs.as<uint8_t>(), w.counters.as<int>() + 0, st_);
  pmark("attn.q");
  // (1) kv: without ring or window indices (same call as the verify variant in attention()) → verify window indices
  k::fused_kv_proj(w.xn.as<bf16>(), dim, A.wkv.w.as<uint8_t>(), A.wkv.s.as<uint8_t>(), D, A.kv_norm.as<bf16>(), c.norm_eps, rd, freqs, w.pos_d, M,
                   w.kv.as<bf16>(), nullptr, win, nullptr, 0, w.counters.as<int>() + 1, st_);
  const int idx_stride = win + c.index_topk;
  k::window_idxs_verify(M, win, w.pos_d, w.vgrp_d, w.idx.as<int32_t>(), idx_stride, st_);
  pmark("attn.kv");
  const int topk_max_cols = ratio ? c.index_topk : 0;
  if (ratio) {
    if (c.is_kv_source(l)) {
      if (ratio > 1) {
        k::bf16_to_f32(w.xn.as<bf16>(), M * dim, w.xf.as<float>(), st_);
        blas_->gemm_f32(w.xf.as<float>(), A.comp_wkv.as<float>(), w.ckv.as<float>(), M, D, dim);
        blas_->gemm_f32(w.xf.as<float>(), A.comp_wgate.as<float>(), w.cscore.as<float>(), M, D, dim);
        for (size_t i = 0; i < v_src2_.size(); ++i)
          if (v_src2_[i] == l) {  // keep inputs for rollback (kv ‖ score, row capacity Vrows)
            const size_t vrows = w.vckv[i].n / ((size_t)2 * D * 4);
            CUDA_CHECK(cudaMemcpyAsync(w.vckv[i].p, w.ckv.p, (size_t)M * D * 4, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(w.vckv[i].as<float>() + vrows * D, w.cscore.p, (size_t)M * D * 4, cudaMemcpyDeviceToDevice, st_));
          }
        // (2) row-order compressor (rows of the same sequence carry the same state forward)
        k::compressor_step_seq(w.ckv.as<float>(), w.cscore.as<float>(), M, D, ratio, w.pos_d, w.skv_d, w.ssc_d, w.cout.as<float>(), w.valid.as<uint8_t>(), st_);
        k::f32_to_bf16(w.cout.as<float>(), M * D, w.latent.as<bf16>(), st_);
      } else {
        blas_->gemm_bf16(w.xn.as<bf16>(), A.comp_wkv.as<bf16>(), w.latent.as<bf16>(), M, D, dim);
        CUDA_CHECK(cudaMemsetAsync(w.valid.p, 1, (size_t)M, st_));
      }
      k::rmsnorm(w.latent.as<bf16>(), A.comp_norm.as<bf16>(), c.norm_eps, M, D, w.latent.as<bf16>(), st_);
      if (c.is_index_source(l)) {
        blas_->gemm_bf16(w.latent.as<bf16>(), A.idx_wk.as<bf16>(), w.ik.as<bf16>(), M, c.index_head_dim, D);
        k::rmsnorm(w.ik.as<bf16>(), A.idx_k_norm.as<bf16>(), c.norm_eps, M, c.index_head_dim, w.ik.as<bf16>(), st_);
        k::rope_last(w.ik.as<bf16>(), M, 1, c.index_head_dim, rd, freqs, w.gpos_d, false, st_);
        k::fp4_pack_rows(w.ik.as<bf16>(), M, c.index_head_dim, 32, false, w.dstk_d, w.dsti_d, w.valid.as<uint8_t>(), kvp::IDX_ROW, st_);
      }
      k::rope_last(w.latent.as<bf16>(), M, 1, D, rd, freqs, w.gpos_d, false, st_);
      k::fp4_pack_rows(w.latent.as<bf16>(), M, D, 16, true, w.dstp_d, w.dsti_d, w.valid.as<uint8_t>(), kvp::COMP_ROW, st_);
    }
    if (c.is_index_source(l)) {
      const int Hi = c.index_n_heads, Di = c.index_head_dim;
      if (Tmax > 0) {
        k::gemm_bs(w.qrq.as<uint8_t>(), w.qrs.as<uint8_t>(), A.idx_wq_b.w.as<uint8_t>(), A.idx_wq_b.s.as<uint8_t>(), false, M, Hi * Di,
                   c.q_lora_rank, w.iq.as<bf16>(), nullptr, st_);
        k::rope_last(w.iq.as<bf16>(), M, Hi, Di, rd, model_.rope_compress(), w.pos_d, false, st_);
        static const bool idx_tc_dec = getenv("HIVE_IDX_TC") && atoi(getenv("HIVE_IDX_TC")) != 0;  // same formula as attention_decode_dev
        static const bool idx_f32_dec = getenv("HIVE_IDX_F32") && atoi(getenv("HIVE_IDX_F32")) != 0;
        HIVE_CHECK(!(idx_f32_dec && idx_tc_dec), "HIVE_IDX_F32 and HIVE_IDX_TC are exclusive");
        const bool pack_dec = idx_tc_dec && c.cand_source_layer >= 0 && c.cand_source_layer < l;
        if (pack_dec) { HIVE_CHECK(Di == kvp::IDX_D, "idx tc Di"); k::fp4_pack(w.iq.as<bf16>(), M * Hi, Di, 32, false, w.iqp.as<uint8_t>(), kvp::IDX_ROW, st_); }
        k::fp4_quant_roundtrip(w.iq.as<bf16>(), M, Hi * Di, 32, false, st_);
        blas_->gemm_bf16(w.xn.as<bf16>(), A.idx_weights_proj.as<bf16>(), w.iw.as<bf16>(), M, Hi, dim);
        k::scale_bf16(w.iw.as<bf16>(), M * Hi, (1.0f / sqrtf((float)Di)) * (1.0f / sqrtf((float)Hi)), st_);
        const bool cand_src = (l == c.cand_source_layer);
        const bool uses_cand = (c.cand_source_layer >= 0 && c.cand_source_layer < l);
        const int bs = c.cand_block, nblocks = (Tmax + bs - 1) / bs, CB = c.cand_topk_blocks;
        const int topk = std::min(c.index_topk, Tmax);
        if (uses_cand) {
          const int kbc = std::min(CB, nblocks);
          HIVE_CHECK(have_candidates_ && kbc > 0, "candidates missing");
          const int ncand = kbc * bs;
          if (idx_f32_dec)
            k::indexer_scores_cand_f32(w.iq.as<bf16>(), nullptr, w.kptrs_d, w.trows_d, w.iw.as<bf16>(), M, Hi, Di, w.cand.as<int32_t>(), CB, kbc, bs,
                                       w.visible_d, w.iscore_f.as<float>(), st_);
          else if (pack_dec)
            k::indexer_scores_cand_tc(w.iqp.as<uint8_t>(), nullptr, w.kptrs_d, w.trows_d, w.iw.as<bf16>(), M, Hi, w.cand.as<int32_t>(), CB, kbc, bs,
                                      w.visible_d, w.iscore.as<bf16>(), st_);
          else
            k::indexer_scores_cand(w.iq.as<bf16>(), nullptr, w.kptrs_d, w.trows_d, w.iw.as<bf16>(), M, Hi, Di, w.cand.as<int32_t>(), CB, kbc, bs,
                                   w.visible_d, w.iscore.as<bf16>(), st_);
          if (!idx_f32_dec) k::bf16_rows_to_f32(w.iscore.as<bf16>(), M * ncand, w.iscore_f.as<float>(), st_);
          k::topk_select_rows(w.iscore_f.as<float>(), M, ncand, topk, ncand, w.topk_pos.as<int32_t>(), topk, st_);
          k::cand_to_pos(w.topk_pos.as<int32_t>(), M, topk, w.cand.as<int32_t>(), CB, bs, st_);
        } else {
          if (idx_f32_dec) k::indexer_scores_rows_f32(w.iq.as<bf16>(), w.kptrs_d, w.trows_d, Tmax, w.iw.as<bf16>(), M, Hi, Di, w.visible_d, w.iscore_f.as<float>(), st_);
          else k::indexer_scores_rows(w.iq.as<bf16>(), w.kptrs_d, w.trows_d, Tmax, w.iw.as<bf16>(), M, Hi, Di, w.visible_d, w.iscore.as<bf16>(), st_);
          if (cand_src) {
            if (idx_f32_dec) k::block_max_f32(w.iscore_f.as<float>(), M, Tmax, bs, w.visible_d, w.bmax.as<float>(), st_);
            else k::block_max(w.iscore.as<bf16>(), M, Tmax, bs, w.visible_d, w.bmax.as<float>(), st_);
            const int kb = std::min(c.cand_topk_blocks, nblocks);
            k::topk_select_rows(w.bmax.as<float>(), M, nblocks, kb, nblocks, w.topk_pos.as<int32_t>(), kb, st_);
            CUDA_CHECK(cudaMemcpy2DAsync(w.cand.p, (size_t)CB * 4, w.topk_pos.p, (size_t)kb * 4, (size_t)kb * 4, M, cudaMemcpyDeviceToDevice, st_));
            have_candidates_ = true;
          }
          if (!idx_f32_dec) k::bf16_rows_to_f32(w.iscore.as<bf16>(), M * Tmax, w.iscore_f.as<float>(), st_);
          k::topk_select_rows_t(w.iscore_f.as<float>(), M, w.trows_d, Tmax, topk, Tmax, w.topk_pos.as<int32_t>(), topk, st_);
        }
        k::offset_idxs(w.topk_pos.as<int32_t>(), M, topk, w.visible_d, win + M, w.idx.as<int32_t>() + win, idx_stride, st_);
      }
    }
  }
  pmark("attn.compress");
  static const bool attn_tc = env_on("HIVE_ATTN_TC");  // same comparison switch as attention_decode_dev
  if (attn_tc)
    k::sparse_attn_tc_rows(w.q.as<bf16>(), M, H, D, w.kvrows_d, w.kv.as<bf16>(), M, win, w.idx.as<int32_t>(), idx_stride, win + topk_max_cols,
                           A.attn_sink.as<float>(), 1.0f / sqrtf((float)D), w.attn_S.as<float>(), w.attn_mx.as<float>(), w.attn_sum.as<float>(),
                           w.o.as<bf16>(), st_);
  else
    k::sparse_attn_decode_rows(w.q.as<bf16>(), M, H, D, w.kvrows_d, w.kv.as<bf16>(), M, win, w.idx.as<int32_t>(), idx_stride,
                               win + topk_max_cols, A.attn_sink.as<float>(), 1.0f / sqrtf((float)D), w.o.as<bf16>(),
                               (win + topk_max_cols) > 256 ? attn_splits_ : 1, w.attn_pacc.as<float>(), w.attn_pm.as<float>(), w.attn_ps.as<float>(), st_,
                               attn_tc ? nullptr : freqs, w.pos_d, rd);
  if (attn_tc) k::rope_last(w.o.as<bf16>(), M, H, D, rd, freqs, w.pos_d, true, st_);
  pmark("attn.sparse");
  }  // !front_done (Q2)
  {
    const int G = c.o_groups, sub = H * D / G, R = c.o_lora_rank;
    (k::decode_gemv2_on() ? k::gemv2_bf16_fp8_grouped : k::gemv_bf16_fp8_grouped)(w.o.as<bf16>(), H * D, A.wo_a_q.w.as<uint8_t>(), A.wo_a_q.s.as<uint8_t>(), M, G, R, sub, w.og.as<bf16>(), st_);
  }
  (k::decode_attn2_on() ? k::attn2_gemv_quantin : k::decode_gemv2_on() ? k::gemv2_quantin : k::fused_gemv_quantin)(w.og.as<bf16>(), c.o_groups * c.o_lora_rank, A.wo_b.w.as<uint8_t>(), A.wo_b.s.as<uint8_t>(), M, dim, w.attn_out.as<bf16>(), st_);
  // (3) the ring is written after attention (each row into its own sequence's ring and position) — every row has read before a later row of the same sequence overwrites the oldest cell of an earlier row's window
  //   (the Q2 fused front already wrote it after attention with the same kernel — never written twice)
  if (!front_done) k::ring_write_rows(w.kv.as<bf16>(), M, D, win, w.pos_d, w.ringp_d, st_);
  pmark("attn.wo");
}

void Runtime::verify_decode(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  const int P = (int)parts.size();
  int M = 0;
  for (const VerifyPart& p : parts) M += p.M;
  HIVE_CHECK(mtp_on_ && v2_ok_ && P >= 1 && P <= kVParts && M >= 1 && M <= v2_rows_, "verify2 rows");
  HIVE_CHECK(v_M_ == 0 && vparts_.empty(), "rollback pending");
  for (const VerifyPart& p : parts) {
    HIVE_CHECK(p.seq && p.ids && p.M >= 1 && p.seq->pos + p.M <= p.seq->cap, "context overflow");  // same formula as forward (per part)
    if (p.seq->snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, p.seq->snapshot_fence->event, 0));
  }
  const double t0 = now_ms();
  vparts_.clear(); vseqs_.clear();
  int r = 0;
  for (const VerifyPart& p : parts) {  // row tables (mapped pinned — the previous forward/draft synchronized st_)
    Seq& s = *p.seq;
    vparts_.push_back(VPart{&s, s.pos, p.M, r, s.engram_history.size()});
    for (int i = 0; i < p.M; ++i, ++r) {
      w.ids_h[r] = p.ids[i];
      w.pos_h[r] = (int32_t)(s.pos + i);
      w.is_image_h[r] = 0;
      w.vgrp_h[r] = vparts_.back().row0;
      memcpy(w.emb_h + (size_t)r * c.dim, model_.embed_host() + (size_t)p.ids[i] * c.dim, (size_t)c.dim * 2);
      vseqs_.push_back(&s);
    }
  }
  engram_ssd_hint(w.ids_h, nullptr, M, vseqs_.data(), nullptr);  // HIVE_ENGRAM_SSD read-ahead (off = no-op) — including draft tokens
  v_M_ = M; v_pos0_ = vparts_[0].pos0; v_hist_ = vparts_[0].hist;
  verify_ = true; vdec_ = true;
  struct Guard {  // exception: release verify state (so the next forward is not caught by it) and mark the part sequences broken (mid-layer — compressor and history half advanced, same rule as forward_batch)
    Runtime* r; int ex;
    ~Guard() {
      r->verify_ = false; r->vdec_ = false;
      if (std::uncaught_exceptions() > ex) { for (VPart& p : r->vparts_) p.seq->broken = true; r->vparts_.clear(); r->vseqs_.clear(); r->v_M_ = 0; }
    }
  } guard{this, std::uncaught_exceptions()};
  xtrace_step(2, M, vseqs_.data(), M);
  store_.commit_pending();
  static const bool cw_trace = env_on("HIVE_TRACE_CACHE");
  const bool prof = profile_ && !(profile_every_ > 1 && step_ % profile_every_ != 0);
  cw_begin(cw_trace || prof);
  pmark("start");
  // rollback snapshots (same content as the verify head of forward() — each part with its own table and slots)
  const int L = c.n_layers, D = c.head_dim, win = c.window;
  for (int pi = 0; pi < P; ++pi) {
    const VPart& vp = vparts_[(size_t)pi];
    bf16** lr = w.vlringp_h + (size_t)pi * L;
    for (int l = 0; l < L; ++l) lr[l] = vp.seq->ring[l].as<bf16>();
    k::ring_gather_layers(w.vlringp_d + (size_t)pi * L, L, win, vp.pos0, vp.M, D, w.ring_snap.as<bf16>() + (size_t)L * vp.row0 * D, st_);
    for (size_t i = 0; i < v_src2_.size(); ++i) {
      const int l = v_src2_[i];
      DevBuf& sn = w.vcsnap[(size_t)pi * v_src2_.size() + i];
      const size_t half = sn.n / 2;
      CUDA_CHECK(cudaMemcpyAsync(sn.p, vp.seq->comp_state_kv[l].p, half, cudaMemcpyDeviceToDevice, st_));
      CUDA_CHECK(cudaMemcpyAsync(sn.as<uint8_t>() + half, vp.seq->comp_state_score[l].p, half, cudaMemcpyDeviceToDevice, st_));
    }
  }
  pmark("embed");
  shared_comp_kv_ = nullptr; shared_index_k_ = nullptr; shared_topk_cols_ = 0; have_candidates_ = false;
  shared_topk_rows_.assign(M, 0);
  const int nL = model_.n_loaded_layers();
  dov_->pre_l = -1; dov_->dma_armed = false;
  dov_->begin(prof, nL, false);
  if (host_fast_) pin_engine_thread_once();
  for (int l = 0; l < nL; ++l) decode_layer(vseqs_, l, M, stats);  // batch_ stays null (the B2 batched-miss policy is not used, as in the base verify)
  auto head_body = [&] {  // same as forward_batch's head body (last layer tail + head + sampler candidates)
    layer_tail(nL - 1, M);
    k::hc_pre(w.h.as<bf16>(), w.pre_mix.as<float>(), M, c.hc, c.dim, w.x.as<bf16>(), st_);
    k::rmsnorm(w.x.as<bf16>(), model_.norm(), c.norm_eps, M, c.dim, w.xn.as<bf16>(), st_);
    k::bf16_to_f32(w.xn.as<bf16>(), M * c.dim, w.xf.as<float>(), st_);
    k::head_logits(w.xf.as<float>(), model_.head(), M, c.vocab, c.dim, w.logits_b.as<float>(), st_);
    k::argmax_rows(w.logits_b.as<float>(), M, c.vocab, w.next_b.as<int32_t>(), st_);
    CUDA_CHECK(cudaMemcpyAsync(w.next_b_h, w.next_b.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st_));
    sampler_cands_dev(M);
  };
  dov_->mark(dov_->slot(nL, 0), DecodeOverlap::kPre, st_);
  run_graph(nL, M, 0, head_body);
  logits_rows.resize((size_t)M * c.vocab);
  CUDA_CHECK(cudaMemcpyAsync(logits_rows.data(), w.logits_b.p, (size_t)M * c.vocab * 4, cudaMemcpyDeviceToHost, st_));
  for (const VPart& vp : vparts_) {
    vp.seq->tokens.insert(vp.seq->tokens.end(), w.ids_h + vp.row0, w.ids_h + vp.row0 + vp.M);
    vp.seq->pos += vp.M;
  }
  CUDA_CHECK(cudaStreamSynchronize(st_));
  pmark("head");
  cw_collect();
  preport(M);
  dh_report(M);
  if (opt_.promote_per_token > 0 || opt_.promote_misses > 0) {  // post-step promotion as in forward_batch
    int n = promote_after_step();
    if (stats) stats->n_promoted += n;
  }
  cw_line_.clear();
  if (stats) stats->ms_total += now_ms() - t0;
  ++step_;
}

void Runtime::forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats) {
  verify_decode(parts, logits_rows, stats);
}

// Keep only the first n_keep[p] rows per part (same content as rollback): pos, tokens, engram history, ring slots, ratio>1 compressor state (snapshot restore +
//   re-applying the kept inputs of accepted rows), draft rings and hidden states (mtp_sync_seq — part rows row0). Entries of rejected rows in the compressed KV and indexer key caches are not reverted (same reason as in rollback).
void Runtime::rollback_batch(const std::vector<int>& n_keep) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  HIVE_CHECK(!vparts_.empty() && n_keep.size() == vparts_.size(), "rollback parts");
  for (size_t p = 0; p < vparts_.size(); ++p) HIVE_CHECK(n_keep[p] >= 1 && n_keep[p] <= vparts_[p].M, "rollback range");
  struct Clear { Runtime* r; ~Clear() { r->vparts_.clear(); r->vseqs_.clear(); r->v_M_ = 0; } } clr{this};
  const int L = c.n_layers, D = c.head_dim;
  for (size_t p = 0; p < vparts_.size(); ++p) {
    const VPart& vp = vparts_[p];
    Seq& seq = *vp.seq;
    const int nk = n_keep[p];
    if (nk < vp.M) {
      seq.pos = vp.pos0 + nk;
      seq.tokens.resize(seq.tokens.size() - (size_t)(vp.M - nk));
      if (eh_) seq.engram_history.resize(vp.hist + (size_t)nk);
      k::ring_scatter_layers(w.ring_snap.as<bf16>() + (size_t)L * vp.row0 * D, w.vlringp_d + p * L, L, c.window, vp.pos0, nk, vp.M, D, st_);
      for (size_t i = 0; i < v_src2_.size(); ++i) {
        const int l = v_src2_[i];
        const DevBuf& sn = w.vcsnap[p * v_src2_.size() + i];
        const size_t half = sn.n / 2;
        CUDA_CHECK(cudaMemcpyAsync(seq.comp_state_kv[l].p, sn.p, half, cudaMemcpyDeviceToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(seq.comp_state_score[l].p, sn.as<uint8_t>() + half, half, cudaMemcpyDeviceToDevice, st_));
        const size_t vrows = w.vckv[i].n / ((size_t)2 * D * 4);
        k::compressor_apply_rows(w.vckv[i].as<float>() + (size_t)vp.row0 * D, w.vckv[i].as<float>() + (vrows + vp.row0) * D, nk, D, c.ratio(l), vp.pos0,
                                 seq.comp_state_kv[l].as<float>(), seq.comp_state_score[l].as<float>(), st_);
      }
    }
    // mtp_sync_seq passes row positions through a mapped table (mpos_h) — overwrite it only after the previous part's kernels have read it
    if (p > 0) CUDA_CHECK(cudaStreamSynchronize(st_));
    mtp_sync_seq(seq, vp.pos0, nk, vp.row0);
  }
  CUDA_CHECK(cudaStreamSynchronize(st_));
}

void Runtime::forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats) {
  const Config& c = model_.cfg();
  HIVE_CHECK(mtp_on_ && M >= 2 && M <= c.dspark_block + 1 && M <= 8, "verify rows");
  HIVE_CHECK(v_M_ == 0, "rollback pending");
  if (verify2_ && v2_ok_ && M <= v2_rows_) {  // R1 HIVE_MTP_VERIFY2: decode path (outside the conditions fall back to the path below)
    std::vector<VerifyPart> parts{VerifyPart{&seq, ids, M}};
    verify_decode(parts, logits_rows, stats);
    return;
  }
  verify_ = true;
  verify_logits_out_ = &logits_rows;
  v_pos0_ = seq.pos; v_M_ = M; v_hist_ = seq.engram_history.size();
  // even when leaving by exception, reset the verify state (if left set, the next prefill would write past ring_snap and copy logits into a missing vector)
  struct VerifyGuard { Runtime* r; ~VerifyGuard() { r->verify_ = false; r->verify_logits_out_ = nullptr; if (std::uncaught_exceptions()) r->v_M_ = 0; } } vg{this};
  forward(seq, ids, M, nullptr, nullptr, stats);
}

// Keep only the first n_keep rows: pos, tokens, engram history, ring slots (restore the positions before win that rejected rows overwrote), ratio>1 compressor state (snapshot restore + re-applying accepted rows).
// Entries of rejected rows in the compressed KV and indexer key caches are not reverted — the next tokens rewrite them before they are read (before compress_len reaches them).
void Runtime::rollback(Seq& seq, int n_keep) {
  const Config& c = model_.cfg();
  Work& w = *w_;
  HIVE_CHECK(v_M_ > 0 && n_keep >= 1 && n_keep <= v_M_, "rollback range");
  if (!vparts_.empty()) {  // verify that ran through R1 HIVE_MTP_VERIFY2 (one part) — the same rollback via the part variant
    HIVE_CHECK(vparts_.size() == 1 && vparts_[0].seq == &seq, "rollback part");
    rollback_batch(std::vector<int>{n_keep});
    return;
  }
  const int dim = c.dim, D = c.head_dim;
  if (n_keep < v_M_) {
    seq.pos = v_pos0_ + n_keep;
    seq.tokens.resize(seq.tokens.size() - (size_t)(v_M_ - n_keep));
    if (eh_) seq.engram_history.resize(v_hist_ + (size_t)n_keep);
    k::ring_scatter_layers(w.ring_snap.as<bf16>(), w.lringp_d, c.n_layers, c.window, v_pos0_, n_keep, v_M_, D, st_);
    for (size_t i = 0; i < v_src2_.size(); ++i) {
      const int l = v_src2_[i];
      const size_t half = w.cstate_snap[i].n / 2;
      CUDA_CHECK(cudaMemcpyAsync(seq.comp_state_kv[l].p, w.cstate_snap[i].p, half, cudaMemcpyDeviceToDevice, st_));
      CUDA_CHECK(cudaMemcpyAsync(seq.comp_state_score[l].p, w.cstate_snap[i].as<uint8_t>() + half, half, cudaMemcpyDeviceToDevice, st_));
      const AttnWeights& A = model_.layer(l).attn;
      k::bf16_to_f32(w.vxn[i].as<bf16>(), n_keep * dim, w.xf.as<float>(), st_);
      blas_->gemm_f32(w.xf.as<float>(), A.comp_wkv.as<float>(), w.ckv.as<float>(), n_keep, D, dim);
      blas_->gemm_f32(w.xf.as<float>(), A.comp_wgate.as<float>(), w.cscore.as<float>(), n_keep, D, dim);
      k::compressor_pool(w.ckv.as<float>(), w.cscore.as<float>(), n_keep, D, c.ratio(l), v_pos0_, seq.comp_state_kv[l].as<float>(),
                         seq.comp_state_score[l].as<float>(), w.cout.as<float>(), st_);
    }
  }
  mtp_sync_seq(seq, v_pos0_, n_keep);
  CUDA_CHECK(cudaStreamSynchronize(st_));
  v_M_ = 0;
}

}  // namespace hive
