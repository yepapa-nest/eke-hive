// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/expert_store.h"
#include "hive/bulk_load.h"
#include "hive/clock.h"
#include "hive/engram_ssd.h"
#include "hive/pregate.h"

#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <immintrin.h>

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <functional>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace hive {

// ---------------------------------------------------------------------------------------------------------------------
// HIVE_LOAD_PREFAULT (see the HostPrefault header comment in expert_store.h)
namespace {
struct PrefaultRegion { uint8_t* base = nullptr; size_t bytes = 0; int node = -1; bool claimed = false; };
struct PrefaultPool {
  std::mutex mu;
  std::vector<PrefaultRegion> r;
  std::vector<std::pair<uint32_t, uint32_t>> items;  // (region, 2 MiB block)
  std::atomic<size_t> next{0};
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> touched{0};
  std::vector<std::thread> th;
  hive::SteadyClock::time_point t0;
  std::atomic<double> busy_s{0};
  bool active = false;
};
PrefaultPool& prefault_pool() { static PrefaultPool p; return p; }
uint8_t* arena_map(size_t bytes, int node, bool huge) {
  uint8_t* base = nullptr;
  if (numa_available() >= 0 && node >= 0) base = (uint8_t*)numa_alloc_onnode(bytes, node);
  else { base = (uint8_t*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); if (base == MAP_FAILED) base = nullptr; }
  if (base && huge) madvise(base, bytes, MADV_HUGEPAGE);
  return base;
}
void arena_unmap(uint8_t* base, size_t bytes, int node) {
  if (numa_available() >= 0 && node >= 0) numa_free(base, bytes);
  else munmap(base, bytes);
}
void prefault_stop_join(PrefaultPool& P) {
  P.stop = true;
  for (auto& t : P.th) if (t.joinable()) t.join();
  P.th.clear();
}
}  // namespace

std::vector<std::pair<size_t, int>> HostPrefault::predict(const std::string& dir, int max_layer, bool no_mtp) {
  std::vector<std::pair<size_t, int>> out;
  try {
    std::ifstream cf(dir + "/config.json");
    if (!cf) return {};
    nlohmann::json j = nlohmann::json::parse(cf);
    const nlohmann::json& t = j.contains("text_config") ? j["text_config"] : j;
    const int dim = t.at("hidden_size").get<int>(), inter = t.at("moe_intermediate_size").get<int>(), E = t.at("n_routed_experts").get<int>();
    const int n_layers = t.at("num_hidden_layers").get<int>();
    const int dspark_E = t.contains("dspark_n_routed_experts") ? t["dspark_n_routed_experts"].get<int>() : Config{}.dspark_experts;
    std::ifstream xf(dir + "/model.safetensors.index.json");
    if (!xf) return {};
    nlohmann::json idx = nlohmann::json::parse(xf);
    const nlohmann::json& wm = idx.at("weight_map");
    const int nl = max_layer < 0 ? n_layers : std::min(max_layer + 1, n_layers);
    int n_mtp = 0;  // Model::load_mtp condition (all layers + head); stage count = number of mtp.{s}.attn_norm.weight entries in the index
    if (!no_mtp && nl == n_layers && wm.contains("head.weight") && wm.contains("norm.weight"))
      while (wm.contains("mtp." + std::to_string(n_mtp) + ".attn_norm.weight")) ++n_mtp;
    const size_t per_node = ((size_t)nl * E + (size_t)n_mtp * dspark_E) * HalfLayout::make(dim, inter).total;
    const bool two_nodes = numa_available() >= 0 && numa_max_node() >= 1;
    out.push_back({per_node, two_nodes ? 0 : -1});
    out.push_back({per_node, two_nodes ? 1 : -1});
    if (t.contains("engram_layer_ids")) {
      int i = 0;
      for (auto& v : t["engram_layer_ids"]) {
        const int el = v.get<int>(), hi = i++;
        if (el >= nl) continue;
        const std::string w = "layers." + std::to_string(el) + ".engram.embed.weight", sc = "layers." + std::to_string(el) + ".engram.embed.scale";
        if (!wm.contains(w)) continue;
        const int node = two_nodes ? (hi & 1) : -1;
        const int ssd = engram_ssd_mode();  // HIVE_ENGRAM_SSD: 1 = value table is not held in RAM, 2 = scales are not either
        // Read each tensor's size from the header of its own shard — reading only the value table's shard header would miss the
        //   region when the scale lives in another shard (CPU test with a split-shard fixture: 5 regions instead of the expected 6;
        //   real checkpoints keep both in one shard). The header is read once per shard.
        std::map<std::string, nlohmann::json> heads;
        for (const std::string& name : {w, sc}) {
          if ((ssd >= 1 && name == w) || (ssd == 2 && name == sc) || !wm.contains(name)) continue;
          const std::string file = wm[name].get<std::string>();
          if (!heads.count(file)) {
            const int fd = ::open((dir + "/" + file).c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) continue;
            uint64_t hl = 0;
            std::string h;
            if (pread(fd, &hl, 8, 0) == 8 && hl < (64ull << 20)) { h.resize(hl); if (pread(fd, h.data(), hl, 8) != (ssize_t)hl) h.clear(); }
            ::close(fd);
            if (h.empty()) continue;
            heads[file] = nlohmann::json::parse(h);
          }
          const nlohmann::json& hd = heads[file];
          if (hd.contains(name)) out.push_back({hd[name]["data_offsets"][1].get<size_t>() - hd[name]["data_offsets"][0].get<size_t>(), node});
        }
      }
    }
  } catch (...) {
    return {};
  }
  return out;
}

void HostPrefault::start(const std::vector<std::pair<size_t, int>>& regions, int threads) {
  PrefaultPool& P = prefault_pool();
  std::lock_guard<std::mutex> lk(P.mu);
  if (P.active || regions.empty()) return;
  P.t0 = hive::SteadyClock::now();
  for (const auto& [n, node] : regions) {
    const size_t bytes = align_up(n, 2u << 20);
    uint8_t* base = arena_map(bytes, node, true);
    if (!base) continue;  // on failure only this region is dropped (the regular allocation later sees the same failure)
    P.r.push_back({base, bytes, node, false});
  }
  // Block order: alternate between the two expert arenas (loading fills the half records of both nodes together, in layer order), then the rest (engram) in sequence
  const size_t blk = 2u << 20;
  size_t nmax = 0;
  for (size_t i = 0; i < P.r.size() && i < 2; ++i) nmax = std::max(nmax, P.r[i].bytes / blk);
  for (size_t b = 0; b < nmax; ++b)
    for (uint32_t i = 0; i < P.r.size() && i < 2; ++i) if (b < P.r[i].bytes / blk) P.items.push_back({i, (uint32_t)b});
  for (uint32_t i = 2; i < P.r.size(); ++i)
    for (size_t b = 0; b < P.r[i].bytes / blk; ++b) P.items.push_back({i, (uint32_t)b});
  P.active = true;
  for (int t = 0; t < std::max(1, threads); ++t)
    P.th.emplace_back([&P] {
      const auto w0 = hive::SteadyClock::now();
      for (size_t k; !P.stop.load(std::memory_order_relaxed) && (k = P.next.fetch_add(1)) < P.items.size();) {
        const PrefaultRegion& R = P.r[P.items[k].first];
        uint8_t* p = R.base + (size_t)P.items[k].second * (2u << 20);
        // Value-preserving write touch: bytes the loader has already written into a claimed region stay intact (atomic RMW — nothing can interleave)
        for (size_t off = 0; off < (2u << 20); off += 4096) __atomic_fetch_or(reinterpret_cast<uint64_t*>(p + off), (uint64_t)0, __ATOMIC_RELAXED);
        P.touched += 2u << 20;
      }
      const double s = hive::sec_since(w0);
      double cur = P.busy_s.load();
      while (s > cur && !P.busy_s.compare_exchange_weak(cur, s)) {}
    });
  size_t total = 0;
  for (auto& R : P.r) total += R.bytes;
  fprintf(stderr, "[store] host prefault: %zu regions %.1f GB · %d threads (overlapping the dense load)\n", P.r.size(), total / 1e9, std::max(1, threads));
}

uint8_t* HostPrefault::take(size_t aligned_bytes, int node, bool huge) {
  PrefaultPool& P = prefault_pool();
  std::lock_guard<std::mutex> lk(P.mu);
  if (!P.active || !huge) return nullptr;
  for (auto& R : P.r)
    if (!R.claimed && R.bytes == aligned_bytes && R.node == node) { R.claimed = true; return R.base; }
  // Prediction missed — release the remaining regions now so memory is not held twice (claimed regions are kept)
  bool any = false;
  for (auto& R : P.r) any = any || !R.claimed;
  if (any) {
    fprintf(stderr, "[store] ⚠️host prefault: no region for %.1f GB on node %d — releasing the unclaimed regions (allocating normally)\n", aligned_bytes / 1e9, node);
    prefault_stop_join(P);
    for (auto& R : P.r) if (!R.claimed && R.base) { arena_unmap(R.base, R.bytes, R.node); R.base = nullptr; R.claimed = true; }
  }
  return nullptr;
}

std::pair<size_t, size_t> HostPrefault::finish() {
  PrefaultPool& P = prefault_pool();
  std::lock_guard<std::mutex> lk(P.mu);
  if (!P.active) return {0, 0};
  const bool done = P.next.load() >= P.items.size();
  prefault_stop_join(P);
  size_t freed = 0, claimed = 0;
  for (auto& R : P.r) {
    if (!R.claimed && R.base) { arena_unmap(R.base, R.bytes, R.node); freed += R.bytes; R.base = nullptr; }
    else if (R.base) claimed += R.bytes;
  }
  fprintf(stderr, "[store] host prefault: touched %.1f GB (%s, longest thread %.1fs) · claimed %.1f GB · released unclaimed %.1f GB\n", P.touched.load() / 1e9,
          done ? "complete" : "stopped early — the loader touched the rest", P.busy_s.load(), claimed / 1e9, freed / 1e9);
  P.r.clear();
  P.items.clear();
  P.next = 0;
  P.stop = false;
  P.touched = 0;
  P.busy_s = 0;
  P.active = false;
  return {claimed, freed};
}

void PinnedArena::alloc(size_t n, int numa_node, bool huge) {
  bytes = align_up(n, 2u << 20);
  node = numa_node;
  if (uint8_t* p = HostPrefault::take(bytes, numa_node, huge)) { base = p; return; }  // HIVE_LOAD_PREFAULT: the same region, reserved and touched in advance (same allocation function)
  if (numa_available() >= 0 && numa_node >= 0) {
    base = (uint8_t*)numa_alloc_onnode(bytes, numa_node);
  } else {
    base = (uint8_t*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) base = nullptr;
  }
  HIVE_CHECK(base != nullptr, "arena alloc failed");
  if (huge) madvise(base, bytes, MADV_HUGEPAGE);
}
void PinnedArena::pin(size_t unit) {
  // Chunk boundaries are placed on multiples of the record size (unit) — a cudaMemcpyAsync spanning two separately registered ranges fails with invalid argument.
  // (Observed failure: 4 GiB chunks split an 18.8 MB record and broke staging DMA for the full 40-layer model — invisible in the 1-layer test with a 3.4 GB arena.)
  size_t chunk = 4ull << 30;
  if (unit > 0) chunk = std::max<size_t>(unit, (chunk / unit) * unit);
  chunk_ = chunk;
  for (size_t off = 0; off < bytes; off += chunk) {
    size_t n = std::min(chunk, bytes - off);
    CUDA_CHECK(cudaHostRegister(base + off, n, cudaHostRegisterDefault));
  }
}
PinnedArena::~PinnedArena() {
  if (!base) return;
  const size_t chunk = chunk_ ? chunk_ : (4ull << 30);
  for (size_t off = 0; off < bytes; off += chunk) cudaHostUnregister(base + off);
  if (numa_available() >= 0 && node >= 0) numa_free(base, bytes);
  else munmap(base, bytes);
}

namespace {
// D8: switch rule identical to runtime.cpp env_on (unset / "" / "0" = off). The runtime.cpp one lives in an anonymous namespace, so this is a separate copy
//   (named differently because exporting hive::env_on from a header would make calls to runtime.cpp's anonymous env_on ambiguous).
inline bool store_env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }
inline double now_ms_() { return hive::mono_ms(); }
}  // namespace

// P1 HIVE_DECODE_PREGATE prefetch request (see the header comment in hive/pregate.h). Seqlock: the engine thread bumps seq to odd, writes, then to even (release).
//   The worker reads an even s1 (acquire), copies, and uses the copy only if seq is unchanged — elements are relaxed atomics, so overlapping reads are not a race (TSAN). If it changed, the copy is dropped (another request will come).
struct ExpertStore::PfReq {
  std::atomic<uint32_t> seq{0};
  std::atomic<int> l{0}, n{0};
  std::atomic<int> e[pg::kMaxPf];
  std::atomic<long> req{0}, experts{0}, kib{0}, aborted{0}, finished{0};
  PfReq() { for (auto& x : e) x.store(0, std::memory_order_relaxed); }
};

// ---------------------------------------------------------------------------------------------------------------------
struct ExpertStore::Pool {
  std::vector<std::thread> threads;
  std::mutex mu;
  std::condition_variable cv;
  std::vector<Job>* jobs = nullptr;
  // half = half-record node (0/1); r0/r1 = full row numbers (a range within the half); mat (B2 HIVE_CPU_SPLIT13) = phase-1 matrix: 0 = both w1 and w3 (default), 1 = w1 only, 3 = w3 only
  // B3 HIVE_CPU_FINE: parts = number of pieces of the phase-1 block (P1 rows) this item belongs to (1 = single item, SPLIT13 2, FINE 4 = w1/w3 x two row halves);
  //   [q0, q1) = that block (the range the last piece applies swiglu and quantization to — not read when parts == 1, [r0, r1) is used instead)
  struct Item { int job; int phase; int half; int r0, r1; int mat = 0; int parts = 1; int q0 = 0, q1 = 0; int own = -1; };  // own = P1 owning worker in owned mode (-1 = whoever picks it)
  std::vector<Item> items;                 // all phase-1 items, then all phase-2 items
  // D2: workers read this value instead of items.size() (otherwise straggling/stealing workers read size() while start_jobs rebuilds items —
  //   a race TSAN reproduced on every run). 0 = not published. Published with a release store after items is complete.
  std::atomic<int> n_items{0};
  std::atomic<int>* busy = nullptr;        // (shared by the store, common to both pools) number of workers touching next/items — start_jobs waits for 0 before rebuilding
  std::atomic<int>* p1_left = nullptr;     // (shared by the store) remaining phase-1 chunks per job — counts the chunks of both nodes
  std::atomic<int>* p13 = nullptr;         // B2 SPLIT13: (shared by the store) finished w1/w3 items per [job][node][block] — the second one quantizes; nblk = blocks per node
  int nblk = 0, p1_rows = 32;               // phase-1 blocks per node, rows per block (= HIVE_CPU_P1)
  bool p2_prefetch = false;                // B2 HIVE_CPU_P2_PREFETCH: while a phase-2 item waits for phase 1, prefetch its own w2 rows into L2
  std::atomic<uint64_t> pf_sink{0};        // P1 prefetch read values (discarded — keeps the compiler from eliding the reads)
  std::atomic<bool> owned_batch{false};    // P1: this batch is in owned mode (only the worker in item.own computes that item, no stealing) — written by start_jobs before publishing
  std::atomic<int> next{0};
  std::atomic<int> done{0};
  std::atomic<double> t_done{0};   // time (ms) the last item of this pool finished — gives the exact CPU completion time even when the caller waits late because it is blocked on the launch queue (sample for the adaptive prefill split)
  // D7: set by the last finisher **after** it writes t_done — wait_jobs waits on this, not on done
  //   (incrementing done before writing t_done made jobs_done_ms stale right after wait_jobs: 3–6 times in 3000 batches in the CPU test).
  std::atomic<bool> fin{false};
  std::atomic<uint64_t> epoch{0};
  std::atomic<bool> stop{false};  // read without the lock by the spin wait
  const ExpertStore* store = nullptr;
  Pool* other = nullptr;  // pool of the other node — steal its items when our queue is empty (a cross-NUMA read beats an idle core)
  int node = 0;
  int half_idx = 0;  // P1: half record handled by this pool (= pool number n — the item half in start_jobs)
  float limit = 10.f;
  int dim = 5120, inter = 2304;
  int spin_us = 0;        // time to spin for a new epoch after the items run out, before sleeping on the cv (decode batches arrive ~1 ms apart per layer, so this saves wake-up latency)

  void run_item(const Item& it) {
    Job& j = (*jobs)[it.job];
    const HalfLayout& hl = store->half_layout();
    const uint8_t* h = store->host_half(j.layer, j.e, it.half);
    const size_t per = ExpertStore::job_scratch_floats(inter);
    float* gate[ExpertStore::kMaxRows]; float* up[ExpertStore::kMaxRows]; float* yq[ExpertStore::kMaxRows]; float* sy[ExpertStore::kMaxRows];
    for (int r = 0; r < j.R; ++r) {
      float* base = j.scratch + per * r;
      gate[r] = base; up[r] = base + inter; yq[r] = base + 2 * inter; sy[r] = base + 3 * inter;
    }
    if (it.phase == 1) {
      // Rows of a half record are local indices [0, I/2) — shift the output pointers to full-row coordinates so y[local row] lands at gate[full row]
      const int base = it.half * (int)hl.w13_rows;
      float* g_out[ExpertStore::kMaxRows]; float* u_out[ExpertStore::kMaxRows];
      for (int r = 0; r < j.R; ++r) { g_out[r] = gate[r] + base; u_out[r] = up[r] + base; }
      if (it.mat != 3) cpu::gemv_e2m1_rows_multi(h + hl.w1, h + hl.s1, (int)hl.w13_rows, dim, j.a_f, j.a_s, j.R, it.r0 - base, it.r1 - base, g_out);
      if (it.mat != 1) cpu::gemv_e2m1_rows_multi(h + hl.w3, h + hl.s3, (int)hl.w13_rows, dim, j.a_f, j.a_s, j.R, it.r0 - base, it.r1 - base, u_out);
      // B2 SPLIT13: of the pair (the w1 item and the w3 item of the same job, node and block) only the one that finishes second proceeds to the swiglu/quantization below (acq_rel — sees the first one's gate/up writes).
      //   Values are unchanged: the GEMVs cover the same row range with the same calls (independent accumulation per row), and the code below runs once on the same [r0, r1).
      //   B3 FINE: same rule with 4 pieces (w1/w3 x row halves) — the piece that finishes fourth processes the whole block [q0, q1) (same formula, same range, once). Piece GEMVs
      //   accumulate each row independently, so halving the row range leaves every row value unchanged (the 2-row unroll of _v2 also uses the same chain per row — odd boundaries checked by bench_expert_cpu).
      const bool quant = it.parts == 1 || p13[((size_t)it.job * 2 + it.half) * nblk + (it.q0 - base) / p1_rows].fetch_add(1, std::memory_order_acq_rel) == it.parts - 1;
      const int q0 = it.parts == 1 ? it.r0 : it.q0, q1 = it.parts == 1 ? it.r1 : it.q1;
      for (int r = 0; quant && r < j.R; ++r) {
        for (int i = q0; i < q1; ++i) {
          float g = bf2f(f2bf(gate[r][i])), u = bf2f(f2bf(up[r][i]));
          if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
          float v = g / (1.f + expf(-g)) * u * j.route_w[r];
          yq[r][i] = bf2f(f2bf(v));
        }
        for (int b0 = q0; b0 < q1; b0 += 32) {
          float amax = 0.f;
          for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(yq[r][b0 + i]));
          amax = fmaxf(amax, 1e-4f);
          const float sc = e8m0_to_f32(f32_ceil_pow2_e8m0(amax * (1.0f / 448.0f)));
          sy[r][b0 / 32] = sc;
          for (int i = 0; i < 32; ++i) {
            float t = fminf(fmaxf(yq[r][b0 + i] / sc, -448.f), 448.f);
            yq[r][b0 + i] = e4m3_to_f32(f32_to_e4m3_host(t));
          }
        }
      }
      p1_left[it.job].fetch_sub(1, std::memory_order_acq_rel);
    } else {
      if (p2_prefetch && p1_left[it.job].load(std::memory_order_acquire) > 0) {
        // B2 HIVE_CPU_P2_PREFETCH: if phase 1 is not done yet (we are about to wait), pull the w2 rows and scales this item will read into L2 — moves DRAM reads into the otherwise idle barrier wait.
        //   Prefetch is only a load hint and does not affect values. 32 rows x (inter/2 + inter/32) B ≈ 39 KB (fits in a 512 KB Zen2 L2).
        const size_t wrow = (size_t)inter / 2, srow = (size_t)inter / 32;
        const int lr0 = it.r0 - it.half * (int)hl.w2_rows, lr1 = it.r1 - it.half * (int)hl.w2_rows;
        const uint8_t* w = h + hl.w2 + (size_t)lr0 * wrow; const uint8_t* we = h + hl.w2 + (size_t)lr1 * wrow;
        for (; w < we; w += 64) _mm_prefetch(reinterpret_cast<const char*>(w), _MM_HINT_T1);
        const uint8_t* sp = h + hl.s2 + (size_t)lr0 * srow; const uint8_t* se = h + hl.s2 + (size_t)lr1 * srow;
        for (; sp < se; sp += 64) _mm_prefetch(reinterpret_cast<const char*>(sp), _MM_HINT_T1);
      }
      while (p1_left[it.job].load(std::memory_order_acquire) > 0) std::this_thread::yield();
      const float* yqc[ExpertStore::kMaxRows]; const float* syc[ExpertStore::kMaxRows];
      float* o_out[ExpertStore::kMaxRows];
      const int base = it.half * (int)hl.w2_rows;
      for (int r = 0; r < j.R; ++r) { yqc[r] = yq[r]; syc[r] = sy[r]; o_out[r] = j.out[r] + base; }
      cpu::gemv_e2m1_rows_multi(h + hl.w2, h + hl.s2, (int)hl.w2_rows, inter, yqc, syc, j.R, it.r0 - base, it.r1 - base, o_out);
      for (int r = 0; r < j.R; ++r)
        for (int k = it.r0; k < it.r1; ++k) j.out[r][k] = bf2f(f2bf(j.out[r][k]));
    }
  }

  // P1: pin worker tid to the tid-th physical core of the node (first of its SMT siblings) — prefetched rows stay in that core's L2 / CCX L3 until the same worker computes them.
  //   If the core list cannot be read or is too short, keep the plain node binding (absorbed).
  bool pin_core(int tid) {
    if (numa_available() < 0) return false;
    struct bitmask* bm = numa_allocate_cpumask();
    numa_node_to_cpus(node, bm);
    std::vector<int> cores;
    for (int c = 0; c < (int)bm->size; ++c) {
      if (!numa_bitmask_isbitset(bm, c)) continue;
      std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
      int first = -1;
      if (!(f >> first)) first = c;  // if unreadable, treat the cpu itself as the first sibling
      if (first == c) cores.push_back(c);
    }
    numa_free_cpumask(bm);
    if (tid >= (int)cores.size()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cores[(size_t)tid], &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
  }
  // P1 prefetch: reads the rows this worker will own for the request (layer, experts) — same formula as the item layout in start_jobs (pg::owned_spans). Stops when a work batch (epoch) or a new request arrives.
  //   Returns the request number handled (a dropped request also counts as handled — the same request is not read again).
  uint32_t prefetch(int tid, uint32_t s1, uint64_t seen) {
    ExpertStore::PfReq& R = *store->pf_;
    const int l = R.l.load(std::memory_order_relaxed);
    const int n = std::min(R.n.load(std::memory_order_relaxed), pg::kMaxPf);
    int es[pg::kMaxPf];
    for (int i = 0; i < n; ++i) es[i] = R.e[i].load(std::memory_order_acquire);  // acquire: the seq re-check below cannot move ahead of these reads (no fence needed — TSAN)
    if (R.seq.load(std::memory_order_acquire) != s1) return s1;  // changed while reading — drop it (a torn read only mis-targets the prefetch, never the computed values)
    const HalfLayout& hl = store->half_layout();
    const int I2 = (int)hl.w13_rows, D2 = (int)hl.w2_rows, T = (int)threads.size();
    static const int P1 = getenv("HIVE_CPU_P1") ? atoi(getenv("HIVE_CPU_P1")) : 32;  // same value as start_jobs (same environment variable, same default)
    static const int P2 = getenv("HIVE_CPU_P2") ? atoi(getenv("HIVE_CPU_P2")) : 32;
    const size_t r13 = (size_t)store->cfg_.dim / 2, s13 = (size_t)store->cfg_.dim / 32;      // bytes per w1/w3 row and per scale row
    const size_t r2 = (size_t)store->cfg_.moe_inter / 2, s2r = (size_t)store->cfg_.moe_inter / 32;  // bytes per w2 row and per scale row
    uint64_t sink = 0;
    size_t bytes = 0;
    bool aborted = false;
    auto touch = [&](const uint8_t* a, size_t len) {
      for (size_t o = 0; o < len; o += 64) sink ^= *reinterpret_cast<const volatile uint64_t*>(a + o);
      bytes += len;
    };
    for (int i = 0; i < n && !aborted; ++i) {
      const uint8_t* h = store->host_half(l, es[i], half_idx);  // half record = pool number (same as item it.half — even with a single NUMA node)
      pg::owned_spans(es[i], tid, T, I2, D2, P1, P2, [&](const pg::Span& sp) {
        if (aborted) return;
        if (epoch.load(std::memory_order_relaxed) != seen || R.seq.load(std::memory_order_relaxed) != s1) { aborted = true; return; }
        const size_t nr = (size_t)(sp.r1 - sp.r0);
        if (sp.phase == 1) {
          touch(h + hl.w1 + (size_t)sp.r0 * r13, nr * r13); touch(h + hl.s1 + (size_t)sp.r0 * s13, nr * s13);
          touch(h + hl.w3 + (size_t)sp.r0 * r13, nr * r13); touch(h + hl.s3 + (size_t)sp.r0 * s13, nr * s13);
        } else {
          touch(h + hl.w2 + (size_t)sp.r0 * r2, nr * r2); touch(h + hl.s2 + (size_t)sp.r0 * s2r, nr * s2r);
        }
      });
    }
    pf_sink.fetch_xor(sink, std::memory_order_relaxed);  // keeps the reads from being elided (the value is unused)
    R.kib.fetch_add((long)(bytes >> 10), std::memory_order_relaxed);
    (aborted ? R.aborted : R.finished).fetch_add(1, std::memory_order_relaxed);
    return s1;
  }
  bool pf_ready(uint32_t pf_seen) const {  // is there a new (even = fully written) request
    if (!store->pf_) return false;
    const uint32_t s = store->pf_->seq.load(std::memory_order_acquire);
    return (s & 1u) == 0 && s != pf_seen;
  }

  void worker(int tid) {
    if (numa_available() >= 0) {
      struct bitmask* bm = numa_allocate_cpumask();
      numa_node_to_cpus(node, bm);
      cpu_set_t set;
      CPU_ZERO(&set);
      for (int c = 0; c < (int)bm->size; ++c) if (numa_bitmask_isbitset(bm, c)) CPU_SET(c, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      numa_free_cpumask(bm);
    }
    if (store->owned_) pin_core(tid);  // P1 (off = keep the node binding above)
    uint64_t seen = 0;
    uint32_t pf_seen = 0;
    for (;;) {
      {
        // First wait lock-free for a new epoch for spin_us (a cv wake-up = 32 threads taking the mutex in turn, tens of µs) — then sleep
        // P1: a prefetch request is also a wake-up condition (off = store->pf_ is nullptr -> pf_ready is always false — original condition)
        bool got = false;
        if (spin_us > 0) {
          const auto t0 = hive::SteadyClock::now();
          for (;;) {
            if (stop || epoch.load(std::memory_order_acquire) != seen || pf_ready(pf_seen)) { got = true; break; }
            __builtin_ia32_pause();
            if (std::chrono::duration<double, std::micro>(hive::SteadyClock::now() - t0).count() > spin_us) break;
          }
        }
        if (!got) {
          std::unique_lock lk(mu);
          cv.wait(lk, [&] { return stop || epoch.load(std::memory_order_acquire) != seen || pf_ready(pf_seen); });
        }
        if (stop) return;
      }
      if (epoch.load(std::memory_order_acquire) == seen) {
        // P1 prefetch (arrived before the work batch — the attention phase). Afterwards spin briefly for the batch (sleeping would add wake-up latency to this worker's owned items).
        const uint32_t s1 = store->pf_->seq.load(std::memory_order_acquire);
        pf_seen = prefetch(tid, s1, seen);
        if (store->pf_spin_us_ > 0) {
          const auto t0 = hive::SteadyClock::now();
          while (!stop && epoch.load(std::memory_order_acquire) == seen && !pf_ready(pf_seen) &&
                 std::chrono::duration<double, std::micro>(hive::SteadyClock::now() - t0).count() < store->pf_spin_us_)
            __builtin_ia32_pause();
        }
        continue;
      }
      seen = epoch.load(std::memory_order_acquire);
      if (store->pf_) pf_seen = store->pf_->seq.load(std::memory_order_acquire) & ~1u;  // P1: requests before this batch belong to the previous layer — do not read them again
      // D2 ordering: raise busy first (seq_cst) and only then touch next/n_items/items. start_jobs writes next=1<<30 and **then** waits for busy==0,
      //   so a worker that did fetch_add on next before that is guaranteed to be visible in busy (no rebuild until it finishes), and any later fetch_add
      //   gets a value >= 1<<30 or one after publication (next=0, release) — no path picks up a new batch item before publication or runs it twice.
      busy->fetch_add(1);
      if (owned_batch.load(std::memory_order_acquire)) run_owned(tid);  // P1 (no stealing — the other pool's workers handle it)
      else {
        drain(this);
        if (other) drain(other);
      }
      busy->fetch_sub(1);
    }
  }
  // P1 owned batch: only our own items, in list order (all phase-1, then all phase-2 — since the list is in that order, a worker waiting at the phase-2 barrier
  //   has already finished its own phase-1; the other workers also finish their phase-1 before the barrier, so there is no deadlock). Completion counting, the
  //   last finisher's timestamp and fin are the same as in drain. Called only inside busy.
  //   Does nothing before publication (next >= 1<<30) — that batch does not exist yet (this worker wakes again on the next epoch).
  void run_owned(int tid) {
    if (next.load() >= (1 << 30)) return;
    const int n = n_items.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
      const Item& it = items[(size_t)i];
      if (it.own != tid) continue;
      run_item(it);
      if (done.fetch_add(1) + 1 == n) {
        t_done.store(now_ms_(), std::memory_order_release);
        fin.store(true, std::memory_order_release);
      }
    }
  }
  // Picks and runs items of q until none remain (q = our own pool or the other pool to steal from). Called only inside busy.
  static void drain(Pool* q) {
    for (;;) {
      const int i = q->next.fetch_add(1);
      const int n = q->n_items.load(std::memory_order_acquire);
      if (i >= n) break;
      q->run_item(q->items[i]);
      if (q->done.fetch_add(1) + 1 == n) {  // last item: write the time first, then publish via fin (D7)
        q->t_done.store(now_ms_(), std::memory_order_release);
        q->fin.store(true, std::memory_order_release);
      }
    }
  }
};

ExpertStore::ExpertStore(const Config& cfg, int n_layers, size_t vram_cache_bytes, int cpu_threads_per_node, int n_mtp_layers, bool defer_cache)
    : cfg_(cfg), lay_(ExpertLayout::make(cfg.dim, cfg.moe_inter)), hlay_(HalfLayout::make(cfg.dim, cfg.moe_inter)), n_layers_(n_layers + n_mtp_layers),
      E_(cfg.n_routed), n_backbone_(n_layers), threads_per_node_(cpu_threads_per_node) {
  loaded_.assign(n_layers_, 0);
  HIVE_CHECK(cfg.dspark_experts <= E_, "mtp experts ≤ backbone");
  HIVE_CHECK(2 * hlay_.total == lay_.total, "half records must tile the full record");  // 4096 alignment must also hold for half records so both arenas have the same size
  // Per-node arenas: E_of(l) half records per layer (every expert has half on each node)
  rec_off_.assign(n_layers_ + 1, 0);
  for (int l = 0; l < n_layers_; ++l) rec_off_[l + 1] = rec_off_[l] + (size_t)E_of(l);
  const size_t per_node = rec_off_[n_layers_] * hlay_.total;
  const bool two_nodes = numa_available() >= 0 && numa_max_node() >= 1;
  for (int n = 0; n < 2; ++n) {
    fprintf(stderr, "[store] arena node %d: %.1f GiB\n", n, per_node / 1073741824.0);
    arena_[n].alloc(per_node, two_nodes ? n : -1);
  }
  // VRAM cache
  defer_cache_ = defer_cache;  // R5 HIVE_CACHE_FIT: slots are allocated by alloc_cache (the vectors below and the 'B' trace line are redone with the slot count then)
  n_slots_ = defer_cache_ ? 0 : (int)(vram_cache_bytes / lay_.total);
  if (n_slots_ > 0) slots_.alloc((size_t)n_slots_ * lay_.total);
  if (const char* sv = getenv("HIVE_STAGING")) n_staging_ = std::max(8, std::min(256, atoi(sv)));  // prefill pre-copy depth (default 8, 18.8 MB per slot)
  staging_.alloc((size_t)n_staging_ * lay_.total);
  reuse_staging_ = store_env_on("HIVE_CACHE_REUSE_STAGE");  // D8: an atoi check would treat "true" as off
  essd_mode_ = engram_ssd_mode();  // HIVE_ENGRAM_SSD (0 = off = default)
  if (reuse_staging_) {
    staging_key_.assign(n_staging_, -1);
    staging_ready_.resize(n_staging_); staging_reused_.resize(n_staging_);
    staging_reuse_pending_.assign(n_staging_, 0);
    for (int i = 0; i < n_staging_; ++i) {
      CUDA_CHECK(cudaEventCreateWithFlags(&staging_ready_[i], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&staging_reused_[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&reuse_st_, cudaStreamNonBlocking));  // dedicated D2D stream (copy_for_promotion)
    CUDA_CHECK(cudaEventCreateWithFlags(&reuse_done_, cudaEventDisableTiming));
  }
  slot_of_.assign((size_t)n_layers_ * E_, -1);
  expert_of_slot_.assign(n_slots_, -1);
  use_count_.assign((size_t)n_layers_ * E_, 0);
  score_.assign((size_t)n_layers_ * E_, 0.f);
  policy_ = store_env_on("HIVE_CACHE_POLICY") ? parse_cache_policy(getenv("HIVE_CACHE_POLICY")) : 0;  // D4 (unset / "" / "0" = 0 = default; a name = seq)
  if (policy_) { prio_.assign(score_.size(), 0.f); g_raw_.assign(score_.size(), 0.f); g_layer_raw_.assign(n_layers_, 0.f); owners_.resize(16); }
  pending_.assign(n_slots_, -1);
  pending_slot_.assign(slot_of_.size(), -1);
  slot_last_use_.assign(n_slots_, 0);
  if (store_env_on("HIVE_CACHE_EVENTS")) {  // D8: "0" and "" = off (an atoi-free check, so "0" does not create a file named "0" and enable tracing)
    const char* path=getenv("HIVE_CACHE_EVENTS");
    event_trace_=fopen(path,"ab");
    // B7: ~2M 'U' lines per prefill chunk — a 16 MiB buffer reduces write calls, and trace_event assembles lines without printf
    if(event_trace_) { setvbuf(event_trace_,nullptr,_IOFBF,16<<20); if(!defer_cache_) trace_event('B',n_slots_,E_); }  // R5: when deferred, alloc_cache writes it
    else fprintf(stderr,"[store] cache event trace unavailable: %s\n",path);
  }
  if (defer_cache_) fprintf(stderr, "[store] vram cache deferred (HIVE_CACHE_FIT: sized after the session pool) · record %.1f MiB · staging %d\n",
                           lay_.total / 1048576.0, n_staging_);
  else
  fprintf(stderr, "[store] vram cache %d slots × %.1f MiB = %.1f GiB · staging %d\n", n_slots_, lay_.total / 1048576.0,
          (double)n_slots_ * lay_.total / 1073741824.0, n_staging_);
  const int spin_us = getenv("HIVE_CPU_SPIN_US") ? atoi(getenv("HIVE_CPU_SPIN_US")) : 0;
  // B2 CPU pool item layout (both unset / "" / "0" = off = default item list; output bit-identical — test_pool_cpu compares bitwise against expert_forward)
  split13_ = store_env_on("HIVE_CPU_SPLIT13");          // split phase 1 into w1 items and w3 items (the second of the pair quantizes)
  p2_prefetch_ = store_env_on("HIVE_CPU_P2_PREFETCH");  // phase-2 items prefetch their own w2 rows while waiting at the barrier
  copy2d_ = store_env_on("HIVE_STAGE_COPY2D");          // B3: copy_rec_async as 8 calls including 2D copies (same bytes)
  //   The stride identity lay_.s3 - lay_.s1 = lay_.w3 - lay_.w1 (also for hlay_) follows directly from both layout formulas (ExpertLayout/HalfLayout::make) — used without a check.
  // P1 HIVE_DECODE_PREGATE (hive/pregate.h): value = number of top predicted experts (0 / "" / unset = off — pg::parse_k). When off, pf_ is nullptr and the pool is unchanged.
  owned_ = pg::parse_k(getenv("HIVE_DECODE_PREGATE")) > 0;
  if (owned_) {
    pf_ = new PfReq;
    // Upper bound of the post-prefetch spin while waiting for the batch: the attention phase (per-layer front) measured 0.25–0.42 ms ([decode-host] front 10.1–16.9 ms / 40 layers); roughly twice that = 1000 µs.
    //   Override with HIVE_DECODE_PREGATE_SPIN_US=N (µs, 1..100000); unset / "" / "0" / non-numeric / negative = default (option convention — 0 = default path). Use 1 to disable spinning.
    pf_spin_us_ = [] { const char* v = getenv("HIVE_DECODE_PREGATE_SPIN_US"); char* end = nullptr; const long x = v && *v ? strtol(v, &end, 10) : 0; return v && *v && end && *end == 0 && x > 0 ? (int)std::min<long>(x, 100000) : 1000; }();
  }
  fine_ = store_env_on("HIVE_CPU_FINE");                // B3: finer items for multi-row jobs (R >= 2) (start_jobs parts_of)
  for (int n = 0; n < 2; ++n) {
    pool_[n] = new Pool;
    pool_[n]->store = this;
    pool_[n]->node = two_nodes ? n : 0;
    pool_[n]->half_idx = n;
    pool_[n]->limit = cfg_.swiglu_limit;
    pool_[n]->spin_us = spin_us;
    pool_[n]->busy = &pool_busy_;
    for (int t = 0; t < threads_per_node_; ++t) pool_[n]->threads.emplace_back([p = pool_[n], t] { p->worker(t); });
    if (n == 1) { pool_[0]->other = pool_[1]; pool_[1]->other = pool_[0]; }
  }
}

ExpertStore::~ExpertStore() {
  delete essd_;  // HIVE_ENGRAM_SSD: stop the read workers first (the RAM copy of the scales lives in the engram_ arena, freed later)
  essd_ = nullptr;
  if(event_trace_) fclose(event_trace_);
  if (reuse_st_) { cudaStreamSynchronize(reuse_st_); cudaStreamDestroy(reuse_st_); }
  if (reuse_done_) cudaEventDestroy(reuse_done_);
  for (auto evt : staging_ready_) cudaEventDestroy(evt);
  for (auto evt : staging_reused_) cudaEventDestroy(evt);
  for (auto evt : evt_pool_) cudaEventDestroy(evt);
  for (auto& batch : promo_q_) cudaEventDestroy(batch.evt);
  // D6: stop and join the threads of **both** pools before deleting either — pool 1 workers read pool 0 when stealing, so deleting pool 0 first is a use-after-free.
  for (int n = 0; n < 2; ++n) {
    if (!pool_[n]) continue;
    { std::lock_guard<std::mutex> lk(pool_[n]->mu); pool_[n]->stop = true; }
    pool_[n]->cv.notify_all();
  }
  for (int n = 0; n < 2; ++n) if (pool_[n]) for (auto& t : pool_[n]->threads) t.join();
  for (int n = 0; n < 2; ++n) { delete pool_[n]; pool_[n] = nullptr; }
  delete pf_; pf_ = nullptr;
}

// R5 HIVE_CACHE_FIT: allocates the deferred slot region as one block (after hived has allocated the session pool — free VRAM at that point minus the operating margin).
//   Lossless: only the slot count changes (which experts are on the GPU — every expert has the same weights and the same computation). The slot region stays one
//   contiguous block, so the dev_rec(0) + slot*total assumption (G1 step graph slots_base) still holds. Tables sized by the slot count (expert_of_slot_, pending_,
//   slot_last_use_) get the same initial values as in the constructor (they are empty before this call and nothing is resident or promoted — hived defers the warm start until after it).
//   cudaMalloc failure is absorbed: retry with 1/64 fewer slots (at least 1) — startup never stops (0 = no cache, CPU/DMA only, same as --vram-cache-mb 0).
int ExpertStore::alloc_cache(int n) {
  if (!defer_cache_ || cache_allocated_) return n_slots_;
  cache_allocated_ = true;
  n = std::max(0, n);
  // Q1 HIVE_CACHE_ELASTIC: the requested elastic slots (the prefill work buffer space) go at the tail of this block too — n was measured from the remaining VRAM, so it
  //   already includes that buffer (runtime did not allocate it separately). Allocate at least that many (the work buffer is needed anyway). If shrinking goes below it, elasticity is dropped (runtime allocates the buffer itself).
  if (elastic_req_ > 0) n = std::max(n, elastic_req_);
  void* p = nullptr;
  while (n > 0) {
    if (cudaMalloc(&p, (size_t)n * lay_.total) == cudaSuccess && p) break;
    (void)cudaGetLastError();
    p = nullptr;
    n -= std::max(1, n / 64);
  }
  slots_.p = n > 0 ? p : nullptr; slots_.n = n > 0 ? (size_t)n * lay_.total : 0;
  n_slots_ = n;
  expert_of_slot_.assign(n_slots_, -1);
  pending_.assign(n_slots_, -1);
  slot_last_use_.assign(n_slots_, 0);
  trace_event('B', n_slots_, E_);
  fprintf(stderr, "[store] vram cache %d slots × %.1f MiB = %.1f GiB (HIVE_CACHE_FIT)\n", n_slots_, lay_.total / 1048576.0,
          (double)n_slots_ * lay_.total / 1073741824.0);
  if (elastic_req_ > 0) {  // Q1: the last n_el slots are elastic (initially lent out — the whole region is cache)
    if (n_slots_ >= elastic_req_) {
      n_elastic_ = elastic_req_; n_base_ = n_slots_ - n_elastic_;
      fprintf(stderr, "[store] elastic cache: %d of %d slots lent from prefill work buffers (base %d while a prefill runs)\n", n_elastic_, n_slots_, n_base_);
    } else {
      fprintf(stderr, "[store] ⚠️elastic cache off: %d slots < %d elastic (allocation shrank) — prefill work buffers allocated separately\n", n_slots_, elastic_req_);
      elastic_own_.alloc(elastic_bytes_);
    }
    elastic_req_ = 0;
  }
  return n_slots_;
}

// ---- Q1 HIVE_CACHE_ELASTIC (elastic slots — see the elastic_reserve comment in the header) ---------------------------------------------------------
// Lossless: only which experts are on the GPU changes (every expert has the same weights and computation — resident = GPU slot, otherwise CPU/DMA). Reclaiming only
//   updates tables (no copies) — the only wait is for promotion copies into elastic slots (if any, via the batch event). Slot numbers and the address formula are
//   unchanged, so users of slot numbers as values (G1 device slot table, D1 host table) see the new table at the head of the next step (G1 compares the table with
//   memcmp and does an H2D if it changed); while reclaimed, elastic slot numbers do not appear in the table.
void ExpertStore::elastic_size_tables(int n_phys) {
  expert_of_slot_.resize((size_t)n_phys, -1);
  pending_.resize((size_t)n_phys, -1);
  slot_last_use_.resize((size_t)n_phys, 0);
}

int ExpertStore::elastic_reserve(size_t bytes) {
  if (n_elastic_ > 0) return n_elastic_;
  if (elastic_req_ > 0) return elastic_req_;
  if (bytes == 0 || lay_.total == 0) return 0;
  const size_t n_el_z = (bytes + lay_.total - 1) / lay_.total;
  if (n_el_z > ((size_t)1 << 30)) return 0;
  const int n_el = (int)n_el_z;
  elastic_bytes_ = bytes;
  if (cache_deferred()) { elastic_req_ = n_el; return n_el; }  // HIVE_CACHE_FIT: alloc_cache allocates it as one block
  // Slot region already allocated: replace it with a new block of (n_slots + n_el) only when there is nothing to move (during startup — end of the runtime constructor; no copy). Otherwise absorb (0 — the caller allocates separately).
  if (n_resident() > 0 || n_pending() > 0 || !promo_q_.empty() || pace_unissued_ > 0) {
    fprintf(stderr, "[store] ⚠️elastic cache off: slots already in use (resident %d · pending %d)\n", n_resident(), n_pending());
    elastic_own_.alloc(bytes);
    return 0;
  }
  const int n_old = n_slots_, n_phys = n_old + n_el;
  slots_.free();
  void* p = nullptr;
  if (cudaMalloc(&p, (size_t)n_phys * lay_.total) != cudaSuccess || !p) {
    (void)cudaGetLastError();
    if (n_old > 0) slots_.alloc((size_t)n_old * lay_.total);  // restore the previous size (on failure DevBuf aborts — same condition as a normal startup)
    fprintf(stderr, "[store] ⚠️elastic cache off: cudaMalloc(%d slots) failed — prefill work buffers allocated separately\n", n_phys);
    elastic_own_.alloc(bytes);
    return 0;
  }
  slots_.p = p; slots_.n = (size_t)n_phys * lay_.total;
  n_slots_ = n_phys; n_base_ = n_old; n_elastic_ = n_el;
  elastic_size_tables(n_phys);
  trace_event('B', n_slots_, E_);  // the slot count changed (nothing resident — trace replay can restart here with the same result)
  fprintf(stderr, "[store] elastic cache: %d + %d slots (the last %d are lent from prefill work buffers, %.1f GiB)\n", n_base_, n_elastic_, n_elastic_,
          (double)n_elastic_ * lay_.total / 1073741824.0);
  return n_el;
}

int ExpertStore::elastic_reclaim() {
  if (!elastic_lent()) return 0;
  const auto t0 = hive::SteadyClock::now();
  flush_promotions();  // E4: a coordinated batch has an event only after all its pieces are issued (no-op when off)
  for (PromoBatch& b : promo_q_) {
    bool hit = false;
    for (int s : b.slots) hit = hit || s >= n_base_;
    if (!hit) continue;
    // This batch's copies (H2D, REUSE_STAGE D2D — join_reuse joins them ahead of the batch event) must finish before that range can be handed to the work buffer
    CUDA_CHECK(cudaEventSynchronize(b.evt));
    auto keep = std::remove_if(b.slots.begin(), b.slots.end(), [&](int s) {
      if (s < n_base_) return false;
      const int key = pending_[s];
      if (key >= 0) {
        if (pending_slot_[key] == s) pending_slot_[key] = -1;
        pending_[s] = -1;
        ++el_stats_.dropped;
        trace_event('D', key, s);
      }
      return true;  // remove from the batch — so this (finished) batch does not confirm a new promotion into the same slot after it is lent out again
    });
    b.slots.erase(keep, b.slots.end());
  }
  int ev = 0;
  for (int s = n_base_; s < n_base_ + n_elastic_; ++s) {
    const int key = expert_of_slot_[s];
    if (key >= 0 && slot_of_[key] == s) { slot_of_[key] = -1; ++evictions_; ++ev; trace_event('E', key, s); }
    expert_of_slot_[s] = -1;
    pending_[s] = -1;
    slot_last_use_[s] = 0;
  }
  n_slots_ = n_base_;
  ++el_stats_.reclaims;
  el_stats_.evicted += (uint64_t)ev;
  el_stats_.wait_ms += hive::ms_since(t0);
  return ev;
}

void ExpertStore::elastic_lend() {
  if (n_elastic_ == 0 || elastic_lent()) return;
  for (int s = n_base_; s < n_base_ + n_elastic_; ++s) { expert_of_slot_[s] = -1; pending_[s] = -1; slot_last_use_[s] = 0; }
  n_slots_ = n_base_ + n_elastic_;
  ++el_stats_.lends;
}

// ---- Sleep / wake (Z1 — see the sleep_release comment in the header) --------------------------------------------------------------------------
// Lossless: after sleep and wake only which experts sit in GPU slots (the resident set) may differ — every expert has the same weights (re-uploaded with the same
//   bytes from the pinned host records) and the same computation. Resident = GPU slot, otherwise CPU/DMA: the difference is the same as an ordinary cache hit vs miss
//   (fp32 accumulation order). Slot numbers and the address formula (dev_rec(0) + slot*total, one contiguous block; elastic = the last n_elastic_ slots) are the same
//   after waking — only the base address changes (runtime discards and rebuilds the G1 step graph; D1/F2/F3 tables and moe_experts_multi are built from dev_rec on every call).
// Cache event trace (HIVE_CACHE_EVENTS): 'B' 0 on sleep (all residents gone — the replayer clears its table), 'B' <physical slot count> on wake (same marker as alloc_cache).
size_t ExpertStore::sleep_release() {
  if (asleep_) return 0;
  commit_all();  // E4: issue all pieces and wait for and confirm every promotion batch (empties promo_q_ — so no event points at a released slot after sleeping)
  if (reuse_st_) CUDA_CHECK(cudaStreamSynchronize(reuse_st_));
  sleep_phys_ = (int)expert_of_slot_.size();  // physical slot count (with elasticity it stays base + elastic even while reclaimed)
  sleep_lent_ = elastic_lent();
  std::fill(slot_of_.begin(), slot_of_.end(), -1);
  std::fill(pending_slot_.begin(), pending_slot_.end(), -1);
  std::fill(expert_of_slot_.begin(), expert_of_slot_.end(), -1);
  std::fill(pending_.begin(), pending_.end(), -1);
  std::fill(slot_last_use_.begin(), slot_last_use_.end(), 0);
  if (reuse_staging_) {  // staging contents disappear — clear the reuse table (so D2D reuse after waking does not point at stale contents)
    std::fill(staging_key_.begin(), staging_key_.end(), -1);
    std::fill(staging_reuse_pending_.begin(), staging_reuse_pending_.end(), 0);
  }
  const size_t freed = slots_.n + staging_.n + elastic_own_.n;
  sleep_own_bytes_ = elastic_own_.n;
  slots_.free();
  staging_.free();
  elastic_own_.free();
  n_slots_ = 0;
  asleep_ = true;
  trace_event('B', 0, E_);
  fprintf(stderr, "[store] sleep: released %d slots + staging %d (%.1f GiB device)\n", sleep_phys_, n_staging_, freed / 1073741824.0);
  return freed;
}

int ExpertStore::wake_alloc(size_t reserve_bytes) {
  if (!asleep_) return (int)expert_of_slot_.size();
  auto raw = [](size_t n) -> void* {  // DevBuf::alloc aborts on failure — waking returns the failure instead (stays asleep, the caller retries)
    void* p = nullptr;
    if (n == 0) return nullptr;
    if (cudaMalloc(&p, n) != cudaSuccess || !p) { (void)cudaGetLastError(); return nullptr; }
    return p;
  };
  const size_t stage_bytes = (size_t)n_staging_ * lay_.total;
  void* sp = raw(stage_bytes);
  if (!sp) { fprintf(stderr, "[store] ⚠️wake: staging ring (%.0f MiB) does not fit — still asleep\n", stage_bytes / 1048576.0); return -1; }
  void* op = nullptr;
  if (sleep_own_bytes_ && !(op = raw(sleep_own_bytes_))) {
    cudaFree(sp);
    fprintf(stderr, "[store] ⚠️wake: prefill work region (%.0f MiB) does not fit — still asleep\n", sleep_own_bytes_ / 1048576.0);
    return -1;
  }
  int n = sleep_phys_;
  if (reserve_bytes > 0) {  // HIVE_CACHE_FIT: same formula as at startup (free memory then minus the operating margin) — if other GPU users freed less memory, fewer slots (absorbed)
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    n = std::min(n, fit_slots(fr, reserve_bytes, lay_.total));
  }
  n = std::max(n, n_elastic_);  // elastic slots = prefill work buffer space (required)
  void* p = nullptr;
  while (n > 0) {
    if ((p = raw((size_t)n * lay_.total))) break;
    if (n <= n_elastic_) { n = -1; break; }
    n = std::max(n_elastic_, n - std::max(1, n / 64));
  }
  if (n < 0) {
    cudaFree(sp);
    if (op) cudaFree(op);
    fprintf(stderr, "[store] ⚠️wake: %d elastic slots (prefill work buffers, %.0f MiB) do not fit — still asleep\n", n_elastic_,
            (double)n_elastic_ * lay_.total / 1048576.0);
    return -1;
  }
  slots_.p = n > 0 ? p : nullptr; slots_.n = n > 0 ? (size_t)n * lay_.total : 0;
  staging_.p = sp; staging_.n = stage_bytes;
  elastic_own_.p = op; elastic_own_.n = op ? sleep_own_bytes_ : 0;
  if (n_elastic_ > 0) { n_base_ = n - n_elastic_; n_slots_ = sleep_lent_ ? n : n_base_; }
  else n_slots_ = n;
  expert_of_slot_.assign((size_t)n, -1);
  pending_.assign((size_t)n, -1);
  slot_last_use_.assign((size_t)n, 0);
  asleep_ = false;
  trace_event('B', n, E_);
  fprintf(stderr, "[store] wake: %d slots (before sleep %d%s) × %.1f MiB · staging %d%s\n", n, sleep_phys_,
          n_elastic_ ? (sleep_lent_ ? " · elastic lent" : " · elastic reclaimed") : "", lay_.total / 1048576.0, n_staging_,
          n < sleep_phys_ ? " — ⚠️fewer slots than before (less free VRAM): lossless, lower hit rate until the next sleep/wake" : "");
  return n;
}

int ExpertStore::restore_keys(const std::vector<int32_t>& keys, cudaStream_t side, const std::function<void(int, int)>& progress) {
  if (asleep_ || n_slots_ == 0) return 0;
  std::vector<int> want;
  want.reserve(std::min(keys.size(), (size_t)n_slots_));
  std::vector<char> seen(slot_of_.size(), 0);
  for (int32_t k : keys) {  // normalize at the entry: skip out-of-range keys, nonexistent experts and duplicates (same rule as the warm-start file)
    if ((int)want.size() >= n_slots_) break;
    if (k < 0 || (size_t)k >= slot_of_.size() || k % E_ >= E_of(k / E_) || seen[(size_t)k] || slot_of_[(size_t)k] >= 0) continue;
    seen[(size_t)k] = 1;
    want.push_back(k);
  }
  const int target = (int)want.size();
  int placed = 0;
  size_t i = 0;
  if (progress) progress(0, target);
  while (i < want.size()) {
    const int room = n_slots_ - n_resident() - n_pending();
    const int batch = std::min({std::max(1, n_slots_ / 4), room, (int)(want.size() - i)});  // within promote_keys' per-call cap (1/4 of the slots) and the free slots
    if (batch <= 0) break;
    std::vector<int> chunk(want.begin() + (long)i, want.begin() + (long)(i + (size_t)batch));
    const int n = promote_keys(chunk, batch, side, batch);  // victims = free slots first (batch <= free slots, so no resident expert is evicted)
    commit_all();
    i += (size_t)batch;
    placed += n;
    if (progress) progress(placed, target);
    if (n == 0) break;
  }
  return placed;
}

// B7: the format is "%c,%d,%d,%.3f\n" (read by tools/check_cache_events.py) — assembled by hand and written with fwrite_unlocked instead of fprintf format parsing and %.3f conversion.
//   The third decimal of ms comes from rounding to integer µs (llround) — it can differ from %.3f in the last digit only exactly at a .0005 boundary (the checker reads floats and uses differences only).
void ExpertStore::trace_event(char event,int key,int slot) const {
  if(!event_trace_) return;
  const double ms=hive::mono_ms();
  char buf[80]; char* p=buf;
  auto put_int=[&p](long long v) {
    char t[24]; int n=0; const bool neg=v<0; unsigned long long u=neg?0ull-(unsigned long long)v:(unsigned long long)v;
    do { t[n++]=(char)('0'+u%10); u/=10; } while(u);
    if(neg) *p++='-';
    while(n) *p++=t[--n];
  };
  *p++=event; *p++=','; put_int(key); *p++=','; put_int(slot); *p++=',';
  long long us=llround(ms*1000.0);
  if(us<0) { *p++='-'; us=-us; }
  put_int(us/1000); *p++='.';
  const int f=(int)(us%1000); *p++=(char)('0'+f/100); *p++=(char)('0'+f/10%10); *p++=(char)('0'+f%10);
  *p++='\n';
  fwrite_unlocked(buf,1,(size_t)(p-buf),event_trace_);
}

const uint8_t* ExpertStore::host_half(int l, int e, int node) const {
  return arena_[node].base + (rec_off_[l] + (size_t)e) * hlay_.total;
}

// GPU record = [w1 all rows][s1][w3][s3][w2][s2]; each matrix piece of half record n goes unchanged into the n-th half byte range of that matrix.
// E4: the piece list lives in one place — copy_rec_async (issued immediately) and defer_rec (HIVE_DECODE_COPY_PRIO coordination — issued later) write the same 12 pieces in the same order.
template <class F>
void ExpertStore::for_each_rec_piece(uint8_t* dst, int l, int e, F&& f) const {
  const size_t w13 = hlay_.s1 - hlay_.w1, s13 = hlay_.w3 - hlay_.s1, w2 = hlay_.s2 - hlay_.w2, s2 = hlay_.w2_rows * cfg_.moe_inter / 32;
  for (int n = 0; n < 2; ++n) {
    const uint8_t* h = host_half(l, e, n);
    f(dst + lay_.w1 + n * w13, h + hlay_.w1, w13);
    f(dst + lay_.s1 + n * s13, h + hlay_.s1, s13);
    f(dst + lay_.w3 + n * w13, h + hlay_.w3, w13);
    f(dst + lay_.s3 + n * s13, h + hlay_.s3, s13);
    f(dst + lay_.w2 + n * w2, h + hlay_.w2, w2);
    f(dst + lay_.s2 + n * s2, h + hlay_.s2, s2);
  }
}

void ExpertStore::copy_rec_async(uint8_t* dst, int l, int e, cudaStream_t st) const {
  ++h2d_records_;
  if (copy2d_) {
    // B3 HIVE_STAGE_COPY2D (off = the 12 pieces below): the same 12 pieces' bytes in 8 copy calls — the (w1, w3) pieces of half record n are two equal-size rows spaced
    //   w13+s13 apart on the host and lay.w3-lay.w1 (= 2*w13 + 2*s13) apart in the GPU record, so they form one 2D copy (width w13, height 2); likewise (s1, s3).
    //   Rationale: a large share of batched-decode preparation (prep, 3.14 ms/step in [decode-host] batch samples) is ~16 CUDA API calls per demand-DMA record
    //   (12 copies + wait/record) — this reduces the call count (per-call host time must be measured on the target machine: [decode-host] prep/launch A/B).
    //   Bytes arriving are identical (tools/test_pool_cpu.py compares the records of both paths byte by byte with fake CUDA); only the piece order within the stream changes (the caller records the completion event afterwards).
    const size_t w13 = hlay_.s1 - hlay_.w1, s13 = hlay_.w3 - hlay_.s1, w2 = hlay_.s2 - hlay_.w2, s2 = hlay_.w2_rows * cfg_.moe_inter / 32;
    const size_t dp = lay_.w3 - lay_.w1, hp = hlay_.w3 - hlay_.w1;  // = lay_.s3 - lay_.s1, hlay_.s3 - hlay_.s1
    for (int n = 0; n < 2; ++n) {
      const uint8_t* h = host_half(l, e, n);
      CUDA_CHECK(cudaMemcpy2DAsync(dst + lay_.w1 + n * w13, dp, h + hlay_.w1, hp, w13, 2, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaMemcpy2DAsync(dst + lay_.s1 + n * s13, dp, h + hlay_.s1, hp, s13, 2, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(dst + lay_.w2 + n * w2, h + hlay_.w2, w2, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(dst + lay_.s2 + n * s2, h + hlay_.s2, s2, cudaMemcpyHostToDevice, st));
    }
    return;
  }
  for_each_rec_piece(dst, l, e, [&](uint8_t* d, const uint8_t* h, size_t n) { CUDA_CHECK(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, st)); });
}

void ExpertStore::defer_rec(uint8_t* dst, int l, int e) {
  ++h2d_records_;  // counted at decision time (same place as copy_rec_async)
  for_each_rec_piece(dst, l, e, [&](uint8_t* d, const uint8_t* h, size_t n) {
    pace_pieces_.push_back({d, h, n});
    pace_bytes_ += n;
    ++pace_batch_pieces_;
  });
}

void ExpertStore::copy_to_staging(int slot, int l, int e, cudaStream_t st) {
  // HIVE_LAYER_YIELD_INTRA staging hold — data-corruption guard (fail-closed on purpose; the only abort on this path). While an intra-layer
  //   yield runs decode steps, the paused prefill's pre-copies (runtime pf_) sit in staging slots whose stage_freed_ event is recorded only
  //   when the prefill consumes them, so a writer advancing stage_next_ would wait on an older event and overwrite a record the prefill then
  //   multiplies with — silently wrong weights, no error anywhere. Every decode writer checks staging_held() first (runtime.cpp: moe_decode_experts,
  //   bm_prefetch, forward_batch_step); reaching this line while held means a writer was missed, and continuing would corrupt the paused
  //   prompt. Evidence: derived from the code (the hold exists only on the new intra path; not measured on a GPU) — reproduced on CPU
  //   2026-10-09 by tools/test_layer_yield_intra_cpu.py (real ExpertStore, fake CUDA: copy_to_staging while held → this abort; mutant without
  //   the check → the slot is overwritten). Not an input-validation guard: no request, model or runtime value can set the hold.
  if (staging_held()) {
    fprintf(stderr, "[store] FATAL: staging slot %d written while held by an intra-layer yield (layer %d expert %d) — a decode staging writer ignored the hold\n", slot, l, e);
    abort();
  }
  if (reuse_staging_ && staging_reuse_pending_[slot]) {
    CUDA_CHECK(cudaStreamWaitEvent(st, staging_reused_[slot], 0));
    staging_reuse_pending_[slot] = 0;
  }
  copy_rec_async(staging_rec(slot), l, e, st);
  if (reuse_staging_) {
    staging_key_[slot] = l * E_ + e;
    CUDA_CHECK(cudaEventRecord(staging_ready_[slot], st));
  }
}

void ExpertStore::copy_for_promotion(uint8_t* dst, int l, int e, cudaStream_t st, cudaEvent_t victim_ready) {
  // (HIVE_CACHE_REUSE_STAGE only) dedicated D2D stream reuse_st_: if the D2D and staging_reused_ were queued on st (the promotion stream) behind earlier H2D copies, the
  //   next staging overwrite waiting on that event (prefill streaming, decode DMA) would be pushed behind every promotion.
  //   Since the D2D leaves st's ordering, the dependencies st provided are expressed as events:
  //   (1) the source (staging slot i) copy is complete = staging_ready_[i]
  //   (2) kernels reading the victim slot's old record have finished = victim_ready (recorded on st_ by the caller where that precondition holds — promote_after_step does it after syncing st_).
  //      st (the promotion stream) is not ordered with st_ either; this precondition comes from a host synchronization, which still holds here, and the event makes it explicit.
  //   (3) batch completion (commit) must include the D2D — promote/promote_keys join reuse_done_ into st via join_reuse(st) right before recording the batch event.
  //   staging_reused_[i] is recorded on reuse_st_, so staging overwrites wait only for the D2D (not for promotion H2D).
  if (reuse_staging_) for (int i = 0; i < n_staging_; ++i) if (staging_key_[i] == l * E_ + e) {
    if (victim_ready) CUDA_CHECK(cudaStreamWaitEvent(reuse_st_, victim_ready, 0));
    CUDA_CHECK(cudaStreamWaitEvent(reuse_st_, staging_ready_[i], 0));
    CUDA_CHECK(cudaMemcpyAsync(dst, staging_rec(i), lay_.total, cudaMemcpyDeviceToDevice, reuse_st_));
    CUDA_CHECK(cudaEventRecord(staging_reused_[i], reuse_st_));
    staging_reuse_pending_[i] = 1;
    reuse_join_pending_ = true;
    ++d2d_records_;
    return;
  }
  (void)victim_ready;
  if (pace_) { defer_rec(dst, l, e); return; }  // E4 HIVE_DECODE_COPY_PRIO: H2D is issued later by issue_promotions (the REUSE_STAGE D2D above is still immediate)
  copy_rec_async(dst, l, e, st);
  for_each_rec_piece(dst, l, e, [&](uint8_t*, const uint8_t*, size_t n) { promo_h2d_bytes_ += n; });
}

void ExpertStore::join_reuse(cudaStream_t side) {
  if (!reuse_join_pending_) return;
  CUDA_CHECK(cudaEventRecord(reuse_done_, reuse_st_));
  CUDA_CHECK(cudaStreamWaitEvent(side, reuse_done_, 0));
  reuse_join_pending_ = false;
}

void ExpertStore::load_layer_experts(Checkpoint& ck, int l, int copy_threads) {
  auto t0 = hive::SteadyClock::now();
  // Layers >= n_backbone are DSpark draft stages (mtp.{s}.ffn.experts.*) — same record format as the backbone
  const std::string p = (l < n_backbone_ ? "layers." + std::to_string(l) : "mtp." + std::to_string(l - n_backbone_)) + ".ffn.experts.";
  const int E_l = E_of(l);
  // Collect the tensor pointers first (shard mmap)
  struct Src { const uint8_t* w1; const uint8_t* s1; const uint8_t* w2; const uint8_t* s2; const uint8_t* w3; const uint8_t* s3; };
  std::vector<Src> src(E_l);
  const size_t wsz = (size_t)cfg_.moe_inter * cfg_.dim / 2, ssz = (size_t)cfg_.moe_inter * cfg_.dim / 32;
  for (int e = 0; e < E_l; ++e) {
    auto get = [&](const char* n, const char* k, size_t want) {
      const TensorInfo& t = ck.get(p + std::to_string(e) + "." + n + "." + k);
      HIVE_CHECK(t.nbytes == want, "expert tensor size " + t.name);
      return t.data;
    };
    src[e] = {get("w1", "weight", wsz), get("w1", "scale", ssz), get("w2", "weight", wsz), get("w2", "scale", ssz),
              get("w3", "weight", wsz), get("w3", "scale", ssz)};
  }
  ck.shard_for(p + "0.w1.weight").advise_sequential();
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int t = 0; t < copy_threads; ++t)
    th.emplace_back([&] {
      for (;;) {
        int e = next.fetch_add(1);
        if (e >= E_l) break;
        const size_t w13 = hlay_.s1 - hlay_.w1, s13 = hlay_.w3 - hlay_.s1, w2 = hlay_.s2 - hlay_.w2, s2 = hlay_.w2_rows * cfg_.moe_inter / 32;
        for (int n = 0; n < 2; ++n) {  // half record n = n-th row half of each matrix
          uint8_t* h = const_cast<uint8_t*>(host_half(l, e, n));
          memcpy(h + hlay_.w1, src[e].w1 + n * w13, w13); memcpy(h + hlay_.s1, src[e].s1 + n * s13, s13);
          memcpy(h + hlay_.w3, src[e].w3 + n * w13, w13); memcpy(h + hlay_.s3, src[e].s3 + n * s13, s13);
          memcpy(h + hlay_.w2, src[e].w2 + n * w2, w2); memcpy(h + hlay_.s2, src[e].s2 + n * s2, s2);
        }
      }
    });
  for (auto& t : th) t.join();
  loaded_[l] = 1;
  double s = hive::sec_since(t0);
  fprintf(stderr, "[store] layer %d experts(%d) → RAM %.1fs (%.1f GB/s)\n", l, E_l, s, (double)E_l * lay_.total / s / 1e9);
}

void ExpertStore::alloc_engram(Checkpoint& ck, int layer, int hash_index) {
  const std::string p = "layers." + std::to_string(layer) + ".engram.embed.";
  const TensorInfo& w = ck.get(p + "weight");
  const TensorInfo& s = ck.get(p + "scale");
  HIVE_CHECK(w.dtype == "F8_E4M3" && s.dtype == "F8_E8M0", "engram dtype");
  if ((int)engram_.size() <= hash_index) engram_.resize(hash_index + 1);
  EngramTable& T = engram_[hash_index];
  if (essd_mode_) {  // HIVE_ENGRAM_SSD: the value table (and the scales in mode 2) is not held in RAM; the file location is registered instead
    if (T.ssd && T.layer == layer) return;  // called again after a LOAD_PAR failure
    T.layer = layer; T.rows = w.shape[0]; T.hd = (int)w.shape[1]; T.ssd = true;
    HIVE_CHECK(s.shape[0] == w.shape[0] && (int64_t)s.shape[1] * 32 == w.shape[1], "engram scale shape");
    const bool two_nodes = numa_available() >= 0 && numa_max_node() >= 1;
    if (essd_mode_ == 1) T.scales.alloc(s.nbytes, two_nodes ? (hash_index & 1) : -1);
    if (!essd_) essd_ = new EngramSsd(EngramSsdOpts::from_env());
    essd_->add_table(hash_index, ck.shard_for(p + "weight").path(), w.offset, ck.shard_for(p + "scale").path(), s.offset, T.rows, T.hd,
                     essd_mode_ == 1 ? T.scales.base : nullptr);
    return;
  }
  if (T.vals.base && T.layer == layer) return;  // reloading via the regular path after a HIVE_LOAD_PAR failure: read into the already allocated table
  T.layer = layer;
  T.rows = w.shape[0];
  T.hd = (int)w.shape[1];
  const bool two_nodes = numa_available() >= 0 && numa_max_node() >= 1;
  int node = two_nodes ? (hash_index & 1) : -1;
  T.vals.alloc(w.nbytes, node);
  T.scales.alloc(s.nbytes, node);
}

void ExpertStore::load_engram(Checkpoint& ck, int layer, int hash_index) {
  const std::string p = "layers." + std::to_string(layer) + ".engram.embed.";
  const TensorInfo& w = ck.get(p + "weight");
  const TensorInfo& s = ck.get(p + "scale");
  auto t0 = hive::SteadyClock::now();
  alloc_engram(ck, layer, hash_index);
  EngramTable& T = engram_[hash_index];
  // Large tables use pread with 16 threads x 64 MB chunks (NVMe sequential read speed) instead of mmap+memcpy (stalls on every 4 KB page fault — 185 s for 94 GB, 0.5 GB/s)
  // Each tensor is read from its own shard — reading the scales from the value table's shard (fd) would load the wrong bytes for checkpoints whose scales live in another shard
  //   (CPU test test_engram_ssd_cpu with a split-shard fixture: 190/192 scale bytes of table 1 mismatched; the real DeepSeek-V4.1-Flash keeps values and scales in one shard).
  auto par_read = [&](int fd, uint8_t* dst, size_t file_off, size_t n) {
    const int nt = 16;
    std::atomic<size_t> next{0};
    const size_t chunk = 64ull << 20;
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
      th.emplace_back([&] {
        for (;;) {
          size_t off = next.fetch_add(chunk);
          if (off >= n) return;
          size_t len = std::min(chunk, n - off), done = 0;
          while (done < len) {
            ssize_t r = pread(fd, dst + off + done, len - done, (off_t)(file_off + off + done));
            HIVE_CHECK(r > 0, "engram pread");
            done += (size_t)r;
          }
        }
      });
    for (auto& t : th) t.join();
  };
  if (T.ssd) {  // HIVE_ENGRAM_SSD: only the scales (mode 1) go to RAM — read from their own file, since they may be in another shard
    if (T.scales.base) {
      par_read(ck.shard_for(p + "scale").fd(), T.scales.base, s.offset, s.nbytes);
      ck.shard_for(p + "scale").advise_dontneed();
    }
    const double sec = hive::sec_since(t0);
    fprintf(stderr, "[store] engram layer %d (%lld rows) → SSD (vals %.1f GB stay on disk%s) · scales %s %.2f GB in %.1fs\n", layer, (long long)T.rows,
            w.nbytes / 1e9, essd_->direct(hash_index) ? ", O_DIRECT" : ", buffered", T.scales.base ? "RAM" : "SSD", s.nbytes / 1e9, sec);
    return;  // row cache and startup log come after all tables are registered: engram_ssd_finish (hived, hive CLI)
  }
  par_read(ck.shard_for(p + "weight").fd(), T.vals.base, w.offset, w.nbytes);
  par_read(ck.shard_for(p + "scale").fd(), T.scales.base, s.offset, s.nbytes);
  double sec = hive::sec_since(t0);
  fprintf(stderr, "[store] engram layer %d (%lld rows) → RAM %.1fs (%.1f GB/s)\n", layer, (long long)T.rows, sec, (w.nbytes + s.nbytes) / sec / 1e9);
  ck.shard_for(p + "weight").advise_dontneed();
  if (&ck.shard_for(p + "scale") != &ck.shard_for(p + "weight")) ck.shard_for(p + "scale").advise_dontneed();
}

size_t ExpertStore::engram_ssd_cache_host_bytes() const { return essd_ ? essd_->cache_host_bytes() : 0; }

void ExpertStore::engram_ssd_finish(Checkpoint& ck) {
  if (!essd_ || !essd_->alloc_cache()) return;  // off, or second call
  // Savings = bytes of values (+ scales in mode 2) that would otherwise be loaded into RAM, minus the row cache
  size_t disk = 0;
  for (const auto& t : engram_)
    if (t.ssd) {
      const std::string p = "layers." + std::to_string(t.layer) + ".engram.embed.";
      disk += ck.get(p + "weight").nbytes + (essd_mode_ == 2 ? ck.get(p + "scale").nbytes : 0);
    }
  const EngramSsdOpts& o = essd_->opts();
  fprintf(stderr, "[engram-ssd] mode %d (%s on SSD) · row cache %.0f MiB (%zu rows) · read threads %d · prefetch %s · RAM saved %.1f GB vs RAM tables\n",
          essd_mode_, essd_mode_ == 2 ? "values + scales" : "values", essd_->cache_host_bytes() / 1048576.0, essd_->cache_rows(), o.threads,
          o.prefetch ? "on" : "off", ((double)disk - (double)essd_->cache_host_bytes()) / 1e9);
}

void ExpertStore::pin_all() {
  auto t0 = hive::SteadyClock::now();
  for (int n = 0; n < 2; ++n) arena_[n].pin(hlay_.total);
  fprintf(stderr, "[store] pinned %.1f GiB in %.1fs\n", 2.0 * arena_[0].bytes / 1073741824.0,
          hive::sec_since(t0));
}

std::vector<uint64_t> ExpertStore::host_checksums(int threads) const {
  std::vector<std::pair<const uint8_t*, size_t>> regions = {{arena_[0].base, arena_[0].bytes}, {arena_[1].base, arena_[1].bytes}};
  for (const auto& t : engram_) { regions.push_back({t.vals.base, t.vals.bytes}); regions.push_back({t.scales.base, t.scales.bytes}); }
  const size_t blk = 64ull << 20;
  std::vector<std::pair<size_t, size_t>> items;  // (region, block)
  std::vector<std::vector<uint64_t>> h(regions.size());
  for (size_t r = 0; r < regions.size(); ++r) {
    h[r].assign(regions[r].first ? (regions[r].second + blk - 1) / blk : 0, 0);
    for (size_t b = 0; b < h[r].size(); ++b) items.push_back({r, b});
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> th;
  for (int t = 0; t < std::max(1, threads); ++t)
    th.emplace_back([&] {
      for (size_t k; (k = next.fetch_add(1)) < items.size();) {
        const auto [r, b] = items[k];
        const uint8_t* p = regions[r].first + b * blk;
        const size_t n = std::min(blk, regions[r].second - b * blk);
        uint64_t x = 0xcbf29ce484222325ull ^ n;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); x = (x ^ w) * 0x100000001b3ull; x ^= x >> 29; }
        for (; i < n; ++i) x = (x ^ p[i]) * 0x100000001b3ull;
        h[r][b] = x;
      }
    });
  for (auto& t : th) t.join();
  std::vector<uint64_t> out;
  for (size_t r = 0; r < regions.size(); ++r) {
    uint64_t x = 0x84222325cbf29ce4ull ^ regions[r].second;
    for (size_t b = 0; b < h[r].size(); ++b) x = (x ^ h[r][b] ^ (b * 0x9E3779B97F4A7C15ull)) * 0x100000001b3ull;
    out.push_back(x);
  }
  return out;
}

// Cold load (HIVE_LOAD_PAR): the 12 memcpys of load_layer_experts (each matrix of half record n = the n-th half bytes of the source tensor) and the two tables of
//   load_engram are described as the same (file offset -> destination) pieces and handed to bulk_load (parallel O_DIRECT). The half-record tail (4 KiB alignment padding) is written by neither path.
bool ExpertStore::load_bulk(Checkpoint& ck, int n_layers_total, const std::vector<std::pair<int, int>>& engram_tables, const BulkLoadOpts& opt, bool pin,
                            BulkLoadStats* stats) {
  const auto t0 = hive::SteadyClock::now();
  std::vector<std::string> paths;
  std::map<std::string, int> file_idx;
  auto file_of = [&](const std::string& tensor) {
    const std::string& path = ck.shard_for(tensor).path();
    auto it = file_idx.find(path);
    if (it != file_idx.end()) return it->second;
    paths.push_back(path);
    return file_idx[path] = (int)paths.size() - 1;
  };
  std::vector<LoadSeg> segs;
  const size_t wsz = (size_t)cfg_.moe_inter * cfg_.dim / 2, ssz = (size_t)cfg_.moe_inter * cfg_.dim / 32;
  const size_t w13 = hlay_.s1 - hlay_.w1, s13 = hlay_.w3 - hlay_.s1, w2 = hlay_.s2 - hlay_.w2, s2 = hlay_.w2_rows * cfg_.moe_inter / 32;
  // Tensor order = memcpy order of load_layer_experts (w1 s1 w3 s3 w2 s2), half sizes, positions within the half record
  const char* names[6][2] = {{"w1", "weight"}, {"w1", "scale"}, {"w3", "weight"}, {"w3", "scale"}, {"w2", "weight"}, {"w2", "scale"}};
  const size_t want[6] = {wsz, ssz, wsz, ssz, wsz, ssz};
  const size_t half[6] = {w13, s13, w13, s13, w2, s2};
  const size_t at[6] = {hlay_.w1, hlay_.s1, hlay_.w3, hlay_.s3, hlay_.w2, hlay_.s2};
  size_t expert_bytes = 0, engram_bytes = 0;
  const int n_total = std::min(n_layers_total, n_layers_);
  for (int l = 0; l < n_total; ++l) {
    const std::string p = (l < n_backbone_ ? "layers." + std::to_string(l) : "mtp." + std::to_string(l - n_backbone_)) + ".ffn.experts.";
    const int E_l = E_of(l);
    for (int e = 0; e < E_l; ++e)
      for (int k = 0; k < 6; ++k) {
        const std::string name = p + std::to_string(e) + "." + names[k][0] + "." + names[k][1];
        const TensorInfo& t = ck.get(name);
        HIVE_CHECK(t.nbytes == want[k], "expert tensor size " + t.name);
        const int f = file_of(name);
        for (int n = 0; n < 2; ++n) {
          segs.push_back({f, (uint64_t)(t.offset + n * half[k]), (uint64_t)half[k], const_cast<uint8_t*>(host_half(l, e, n)) + at[k]});
          expert_bytes += half[k];
        }
      }
  }
  for (const auto& [layer, hi] : engram_tables) {
    alloc_engram(ck, layer, hi);
    const std::string p = "layers." + std::to_string(layer) + ".engram.embed.";
    const TensorInfo& w = ck.get(p + "weight");
    const TensorInfo& s = ck.get(p + "scale");
    EngramTable& T = engram_[hi];
    if (T.vals.base) { segs.push_back({file_of(p + "weight"), (uint64_t)w.offset, (uint64_t)w.nbytes, T.vals.base}); engram_bytes += w.nbytes; }  // SSD mode: none
    if (T.scales.base) { segs.push_back({file_of(p + "scale"), (uint64_t)s.offset, (uint64_t)s.nbytes, T.scales.base}); engram_bytes += s.nbytes; }
  }
  double pin_s = 0;
  std::thread pin_th;
  int dev = 0;
  if (pin) CUDA_CHECK(cudaGetDevice(&dev));
  if (pin) pin_th = std::thread([&, dev] {  // overlap registration (page pinning) with reading — registration only pins the arena pages, it never writes their contents
    CUDA_CHECK(cudaSetDevice(dev));  // the new thread's current device = the calling thread's device (do not rely on the default 0)
    const auto p0 = hive::SteadyClock::now();
    pin_all();
    pin_s = hive::sec_since(p0);
  });
  BulkLoadStats st;
  const bool ok = bulk_load(paths, std::move(segs), opt, &st);
  if (pin_th.joinable()) pin_th.join();
  const double sec = hive::sec_since(t0);
  if (ok) {
    for (int l = 0; l < n_total; ++l) loaded_[l] = 1;
    for (const auto& [layer, hi] : engram_tables)
      if (engram_[hi].ssd)
        fprintf(stderr, "[store] engram layer %d (%lld rows) → SSD (values stay on disk%s) · scales %s\n", layer, (long long)engram_[hi].rows,
                essd_->direct(hi) ? ", O_DIRECT" : ", buffered", engram_[hi].scales.base ? "RAM (bulk)" : "SSD");
    engram_ssd_finish(ck);
    char pinbuf[64] = "";
    if (pin) snprintf(pinbuf, sizeof pinbuf, " · pin overlapped %.1fs", pin_s);
    fprintf(stderr, "[store] bulk load: experts %d layers %.1f GB + engram %zu tables %.1f GB → RAM %.1fs (read %.1fs · %.2f GB/s · %s %d/%d files · %d threads × %zu MiB · %d jobs)%s\n",
            n_total, expert_bytes / 1e9, engram_tables.size(), engram_bytes / 1e9, sec, st.sec, st.read_bytes / std::max(st.sec, 1e-9) / 1e9,
            opt.direct ? "O_DIRECT" : "buffered", st.direct_files, st.files, st.threads, opt.chunk >> 20, st.jobs, pinbuf);
  } else {
    fprintf(stderr, "[store] ⚠️bulk load failed after %.1fs (%s) — reloading with the per-layer path\n", sec, st.err.c_str());
    for (int l = 0; l < n_total; ++l) load_layer_experts(ck, l);
    for (const auto& [layer, hi] : engram_tables) load_engram(ck, layer, hi);
  }
  if (stats) *stats = st;
  return ok;
}

int ExpertStore::place(int l, int e, cudaStream_t st) {
  int key = l * E_ + e;
  // A real cache-state dump contained 1,434 duplicate slots; CPU delayed-event
  // replay reproduced a duplicate promote followed by an orphaned resident mapping.
  // Absorb an in-flight request instead of placing the same expert a second time.
  if (pending_slot_[key] >= 0) commit_all();
  int s = slot_of_[key];
  if (s >= 0) { slot_last_use_[s] = ++use_total_; return s; }
  // Free slot -> otherwise the LRU victim (busy slots being promoted are excluded)
  int victim = -1;
  for (int i = 0; i < n_slots_; ++i) if (expert_of_slot_[i] < 0 && pending_[i] < 0) { victim = i; break; }
  if (victim < 0) {
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < n_slots_; ++i) if (pending_[i] < 0 && slot_last_use_[i] < best) { best = slot_last_use_[i]; victim = i; }
    if (victim < 0) {  // if all slots are being promoted, wait for and confirm those copies, then pick again
      HIVE_CHECK(!promo_q_.empty(), "no evictable slot");
      commit_all();
      for (int i = 0; i < n_slots_; ++i) if (pending_[i] < 0 && slot_last_use_[i] < best) { best = slot_last_use_[i]; victim = i; }
      HIVE_CHECK(victim >= 0, "no evictable slot");
    }
    const int old = expert_of_slot_[victim];
    if (old >= 0 && slot_of_[old] == victim) slot_of_[old] = -1;
    if (old >= 0) { ++evictions_; trace_event('E',old,victim); }
  }
  copy_rec_async(const_cast<uint8_t*>(dev_rec(victim)), l, e, st);
  expert_of_slot_[victim] = key;
  slot_of_[key] = victim;
  slot_last_use_[victim] = ++use_total_;
  trace_event('I',key,victim);
  return victim;
}

int ExpertStore::promote(int max_n, float min_score, cudaStream_t side, cudaEvent_t victim_ready) {
  if (pace_) flush_promotions();  // E4: issue the remaining pieces of the previous decision first (so a new decision never arrives with pieces outstanding — see the header comment)
  pace_batch_pieces_ = 0;
  if (n_slots_ == 0 || max_n <= 0) return 0;
  max_n = std::min(max_n, std::max(1, n_slots_ / 4));  // never promote more than 1/4 of the slots at once
  if (n_pending() >= 2 * max_n) return 0;               // do not pile up when the copy engine cannot keep up
  // Candidates: non-resident, score >= min_score, descending score
  std::vector<std::pair<float, int>> cand;
  // D4: with a policy, the priority is prio_ (otherwise score_). prio_ is measured in "expected uses per step", so the policy threshold p_min_ replaces the caller's
  //   score threshold (decode 2, warm 1 — score_ units) (0.025 = about once per 40 steps — the value at which a trace-replay sweep keeps promotions at +6.5% over the default; see the D4 block comment below)
  const std::vector<float>& sv = policy_ ? prio_ : score_;
  if (policy_) min_score = p_min_;
  for (size_t k = 0; k < sv.size(); ++k)
    if (slot_of_[k] < 0 && sv[k] >= min_score) {
      if (pending_slot_[k] >= 0) { ++duplicate_skips_; continue; }
      cand.push_back({sv[k], (int)k});
    }
  if (cand.empty()) return 0;
  std::partial_sort(cand.begin(), cand.begin() + std::min<size_t>(max_n, cand.size()), cand.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
  PromoBatch batch;
  for (size_t i = 0; i < cand.size() && (int)batch.slots.size() < max_n; ++i) {
    // Victim: not busy, lowest score (free slots first). Stop if it scores higher than the candidate.
    int victim = -1;
    float vscore = 1e30f;
    for (int s = 0; s < n_slots_; ++s) {
      if (pending_[s] >= 0) continue;
      int key = expert_of_slot_[s];
      float sc = key < 0 ? -1.f : sv[key];
      if (sc < vscore) { vscore = sc; victim = s; }
    }
    if (victim < 0 || vscore >= cand[i].first) break;
    const int old = expert_of_slot_[victim];
    if (old >= 0 && slot_of_[old] == victim) slot_of_[old] = -1;
    if (old >= 0) { ++evictions_; trace_event('E',old,victim); }
    expert_of_slot_[victim] = -1;
    int key = cand[i].second;
    int l = key / E_, e = key % E_;
    copy_for_promotion(const_cast<uint8_t*>(dev_rec(victim)), l, e, side, victim_ready);
    pending_[victim] = key;
    pending_slot_[key] = victim;
    trace_event('P',key,victim);
    batch.slots.push_back(victim);
  }
  if (batch.slots.empty()) return 0;
  join_reuse(side);  // REUSE_STAGE: so the batch event also covers this batch's D2D (dedicated stream)
  if (evt_pool_.empty()) CUDA_CHECK(cudaEventCreateWithFlags(&batch.evt, cudaEventDisableTiming));
  else { batch.evt = evt_pool_.back(); evt_pool_.pop_back(); }
  if (!pace_) CUDA_CHECK(cudaEventRecord(batch.evt, side));
  else { batch.paced = true; batch.issued = false; batch.pieces_left = pace_batch_pieces_; batch.st = side; ++pace_unissued_; }  // E4: the event is recorded after the last piece (issue_promotions)
  const int issued = (int)batch.slots.size();
  promotions_ += issued;
  promo_q_.push_back(std::move(batch));
  return issued;
}

int ExpertStore::promote_keys(const std::vector<int>& keys, int max_n, cudaStream_t side, int backlog_n, bool score_victims, cudaEvent_t victim_ready) {
  if (pace_) flush_promotions();  // E4: same as promote
  pace_batch_pieces_ = 0;
  if (n_slots_ == 0 || max_n <= 0 || keys.empty()) return 0;
  max_n = std::min(max_n, std::max(1, n_slots_ / 4));  // same cap as promote
  // Do not pile up when the copy engine cannot keep up — measured against the configured cap (backlog_n); using auto mode's per-step share would unfairly block steps with few misses
  if (n_pending() >= 2 * std::max(max_n, backlog_n)) return 0;
  // Victim candidates: slots not being promoted, ascending last use (free slots first)
  std::vector<int> order;
  order.reserve(n_slots_);
  for (int s = 0; s < n_slots_; ++s) if (pending_[s] < 0) order.push_back(s);
  auto older = [&](int a, int b) {
    const bool ea = expert_of_slot_[a] < 0, eb = expert_of_slot_[b] < 0;
    if (ea != eb) return ea;
    if (score_victims && !ea && score_[expert_of_slot_[a]] != score_[expert_of_slot_[b]])
      return score_[expert_of_slot_[a]] < score_[expert_of_slot_[b]];
    return slot_last_use_[a] < slot_last_use_[b];
  };
  const size_t need = std::min(order.size(), (size_t)max_n);  // at most max_n victims are needed — partial sort instead of a full sort (host time per step)
  std::partial_sort(order.begin(), order.begin() + need, order.end(), older);
  order.resize(need);
  PromoBatch batch;
  size_t vi = 0;
  for (int key : keys) {
    if ((int)batch.slots.size() >= max_n || vi >= order.size()) break;
    if (key < 0 || (size_t)key >= slot_of_.size() || key % E_ >= E_of(key / E_) || slot_of_[key] >= 0) continue;
    if (pending_slot_[key] >= 0) { ++duplicate_skips_; continue; }
    const int victim = order[vi];
    if (score_victims && expert_of_slot_[victim] >= 0 && score_[expert_of_slot_[victim]] >= score_[key]) continue;
    ++vi;
    const int old = expert_of_slot_[victim];
    if (old >= 0 && slot_of_[old] == victim) slot_of_[old] = -1;
    if (old >= 0) { ++evictions_; trace_event('E',old,victim); }
    expert_of_slot_[victim] = -1;
    copy_for_promotion(const_cast<uint8_t*>(dev_rec(victim)), key / E_, key % E_, side, victim_ready);
    pending_[victim] = key;
    pending_slot_[key] = victim;
    trace_event('P',key,victim);
    batch.slots.push_back(victim);
  }
  if (batch.slots.empty()) return 0;
  join_reuse(side);  // REUSE_STAGE: so the batch event also covers this batch's D2D (dedicated stream)
  if (evt_pool_.empty()) CUDA_CHECK(cudaEventCreateWithFlags(&batch.evt, cudaEventDisableTiming));
  else { batch.evt = evt_pool_.back(); evt_pool_.pop_back(); }
  if (!pace_) CUDA_CHECK(cudaEventRecord(batch.evt, side));
  else { batch.paced = true; batch.issued = false; batch.pieces_left = pace_batch_pieces_; batch.st = side; ++pace_unissued_; }  // E4: the event is recorded after the last piece (issue_promotions)
  const int issued = (int)batch.slots.size();
  promotions_ += issued;
  promo_q_.push_back(std::move(batch));
  return issued;
}

void ExpertStore::commit_pending() {
  while (!promo_q_.empty()) {
    PromoBatch& b = promo_q_.front();
    // E4 coordinated batch: there is no event until all its pieces are issued (querying an unrecorded event reports "done" — we must stop here). Once issued, wait instead of querying —
    //   this pins the residency point to "the first commit_pending after issuing finished" (step head); with a query the step would depend on copy speed — not deterministic.
    if (!b.issued) return;
    if (b.paced) {
      const auto w0 = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaEventSynchronize(b.evt));
      commit_wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    }
    const auto status = cudaEventQuery(b.evt);
    if (status == cudaErrorNotReady) return;
    CUDA_CHECK(status);  // A failed transfer is not an indefinitely pending transfer.
    for (int s : b.slots) {
      int key = pending_[s];
      if (key < 0) continue;
      // Normalize legacy duplicates at a completed-copy boundary. Never clear the
      // surviving mapping when an obsolete copy finishes later.
      if (slot_of_[key] >= 0 && slot_of_[key] != s) {
        pending_[s] = -1;
        if (pending_slot_[key] == s) pending_slot_[key] = -1;
        ++duplicate_skips_;
        trace_event('D',key,s);
        continue;
      }
      expert_of_slot_[s] = key;
      slot_of_[key] = s;
      slot_last_use_[s] = ++use_total_;
      pending_[s] = -1;
      pending_slot_[key] = -1;
      ++commits_;
      trace_event('C',key,s);
    }
    evt_pool_.push_back(b.evt);
    promo_q_.pop_front();
  }
}

// E4 HIVE_DECODE_COPY_PRIO: issues queued pieces in queue order (up to the byte budget — at least one piece if there is a backlog). After a batch's last piece (a batch with
//   zero pieces as soon as its turn comes), its batch event is recorded on the promotion stream chosen at decision time = issue complete. Only stream order is used (pieces and
//   event all on batch.st) — any wait the caller has placed on that stream (runtime promo_pump's gate after the demand DMA) precedes every piece issued here.
size_t ExpertStore::issue_promotions(size_t budget) {
  size_t done = 0;
  while (pace_unissued_ > 0) {
    PromoBatch& b = promo_q_[promo_q_.size() - (size_t)pace_unissued_];
    if (b.pieces_left == 0) {
      CUDA_CHECK(cudaEventRecord(b.evt, b.st));
      b.issued = true;
      --pace_unissued_;
      continue;
    }
    if (done >= budget && done > 0) break;
    const PromoPiece pc = pace_pieces_.front();
    CUDA_CHECK(cudaMemcpyAsync(pc.dst, pc.src, pc.n, cudaMemcpyHostToDevice, b.st));
    pace_pieces_.pop_front();
    pace_bytes_ -= pc.n;
    promo_h2d_bytes_ += pc.n;
    done += pc.n;
    --b.pieces_left;
  }
  return done;
}

std::vector<int32_t> ExpertStore::resident_keys_by_score() const {
  std::vector<std::pair<float, int32_t>> v;
  for (int s = 0; s < n_slots_; ++s) {
    const int k = expert_of_slot_[s];
    if (k >= 0 && slot_of_[k] == s) v.push_back({prio_of(k), k});
  }
  std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
  std::vector<int32_t> out;
  out.reserve(v.size());
  for (auto& x : v) out.push_back(x.second);
  return out;
}
void ExpertStore::seed_scores(const std::vector<int32_t>& keys, float top, float step) {
  std::vector<char> seen(score_.size(), 0);
  size_t rank = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    const int32_t k = keys[i];
    if (k < 0 || (size_t)k >= score_.size()) continue;
    if (k % E_ >= E_of(k / E_) || seen[k]) continue;
    seen[k] = 1;
    score_[k] = std::max(score_[k], top - step * (float)rank);
    if (policy_) prio_[k] = std::max(prio_[k], top - step * (float)rank);  // D4: the warm-start order also seeds the policy priority (promote uses prio_; from the first decode step the policy recomputes it)
    ++rank;
  }
}

ExpertStore::CacheStats ExpertStore::cache_stats() const {
  CacheStats out;
  std::vector<char> seen(slot_of_.size(), 0);
  for (int s = 0; s < n_slots_; ++s) {
    out.pending += pending_[s] >= 0;
    const int k = expert_of_slot_[s];
    if (k < 0) continue;
    ++out.resident;
    if (seen[k]) ++out.duplicates;
    else { seen[k] = 1; ++out.unique; }
    if (slot_of_[k] != s) ++out.invalid_mappings;
  }
  for (size_t k = 0; k < slot_of_.size(); ++k)
    if (slot_of_[k] >= 0 && expert_of_slot_[slot_of_[k]] != (int)k) ++out.invalid_mappings;
  out.promotions = promotions_; out.commits = commits_; out.evictions = evictions_;
  out.duplicate_skips = duplicate_skips_; out.h2d_records = h2d_records_; out.d2d_records = d2d_records_;
  return out;
}

std::vector<std::pair<int, double>> ExpertStore::coverage(const std::vector<int>& ns) const {
  std::vector<uint64_t> u(use_count_.begin(), use_count_.end());
  std::sort(u.begin(), u.end(), std::greater<uint64_t>());
  double total = 0;
  for (auto v : u) total += (double)v;
  std::vector<std::pair<int, double>> out;
  for (int n : ns) {
    double c = 0;
    for (int i = 0; i < n && i < (int)u.size(); ++i) c += (double)u[i];
    out.push_back({n, total > 0 ? c / total : 0.0});
  }
  return out;
}

// ---- D4 cache policy BEGIN (HIVE_CACHE_POLICY — tools/cache_replay.py builds this file unmodified to replay traces)
// seq = request-aware expected use count. Evidence (tools/cache_replay.py, 9 production traces spanning ~25 h, 3792 slots, --promote 8, HIVE_PHASE_SCORE=1):
//   · Victims of the default score (sum with 0.97 decay) are mostly dead slots with score ≈ 0, and wasted promotions (evicted without a single hit) are 7–11% — the bottleneck is predicting what to promote.
//     A future-aware budget bound (same 8/step, ordered by use count over the next 200 steps) reaches 93% decode hits (natural-language trace; default 80.1%), so prediction is where the headroom is.
//   · Per request: in a natural-language trace (296 requests, median response 255 tokens), the best predictor of a request's remaining use was its use so far + a global prior
//     (static coverage of 95 slots per layer: global only 57%, + request 25 tokens 71%, + 50 tokens 75%). Co-occurrence expansion gave +0.6pt — no effect, not included.
//   · Simple variants (decay 0.98–0.998, per-step presence bonus, hysteresis, threshold 0.5–4) all stayed within ±0.5pt — not included.
//   seq (α 8, pre 8, min 0.025, gd 0.999, od 0.99, idle 8) vs default, decode (batched decode, verify, draft) row hits (evaluated on the last 90% / last 50%):
//     natural language 80.1→81.7% (+1.6pt), 4.2 h battery 75.7→77.0% (+1.3pt), seven 10-minute benchmarks −0.3..+0.4pt / −0.6..+0.6pt;
//     decode-miss work (CPU/DMA experts) summed over 9 traces −5.0% / −5.6%; promotions +6.5%; policy compute ≈ 20 µs/step (measured in the replayer).
//   Sequences whose request has ended (not seen for idle steps) drop out — the default score held on to a finished request's experts for ~30 steps.
// Lossless: this only changes which experts stay in VRAM (every expert has the same weights and computation — resident = GPU, otherwise CPU/DMA). Output tokens do not depend on the policy.
int ExpertStore::parse_cache_policy(const char* v) {
  if (!v || !*v || strcmp(v, "0") == 0) return 0;
  const std::string s(v), name = s.substr(0, s.find(':'));
  if (name != "seq") { fprintf(stderr, "[store] HIVE_CACHE_POLICY=%s: unknown policy (known: seq) — using the default policy\n", v); return 0; }
  // Tuning (A/B and replay sweeps only): seq:alpha=8:pre=8:min=0.025:gd=0.999:od=0.99:idle=8 — unknown keys, negative values and non-numbers are ignored (defaults kept)
  for (size_t i = s.find(':'); i != std::string::npos;) {
    const size_t j = s.find(':', i + 1);
    const std::string kv = s.substr(i + 1, j == std::string::npos ? std::string::npos : j - i - 1);
    const size_t eq = kv.find('=');
    char* end = nullptr;
    const double x = eq == std::string::npos ? -1.0 : strtod(kv.c_str() + eq + 1, &end);
    const bool ok = eq != std::string::npos && end && *end == '\0' && std::isfinite(x) && x >= 0.0;
    const std::string k = kv.substr(0, eq);
    if (ok && k == "alpha") p_alpha_ = (float)x;
    else if (ok && k == "pre") p_pre_ = (float)x;
    else if (ok && k == "min") p_min_ = (float)x;
    else if (ok && k == "gd" && x <= 1.0) p_gdecay_ = (float)x;
    else if (ok && k == "od" && x <= 1.0) p_odecay_ = (float)x;
    else if (ok && k == "idle" && x >= 1.0) p_idle_ = (int)x;
    else if (!kv.empty()) fprintf(stderr, "[store] HIVE_CACHE_POLICY: ignored '%s'\n", kv.c_str());
    i = j;
  }
  fprintf(stderr, "[store] cache policy seq (alpha %.3g · pre %.3g · min %.3g · gd %.4g · od %.4g · idle %d)\n", p_alpha_, p_pre_, p_min_, p_gdecay_, p_odecay_, p_idle_);
  return 1;
}
void ExpertStore::set_row_owners_policy(const uint32_t* uids, int n, bool prompt) {
  row_owner_.assign(std::max(0, n), -1);
  for (int i = 0; i < n; ++i) {
    int o = -1, free_o = -1, old_o = -1;
    for (int j = 0; j < (int)owners_.size(); ++j) {
      if (owners_[j].live && owners_[j].uid == uids[i]) { o = j; break; }
      if (!owners_[j].live && free_o < 0) free_o = j;
      if (owners_[j].live && (old_o < 0 || owners_[j].last < owners_[old_o].last)) old_o = j;
    }
    bool fresh = false;
    if (o < 0) { o = free_o >= 0 ? free_o : old_o; fresh = true; }  // when all entries are taken, evict the sequence not seen for the longest time
    Owner& w = owners_[o];
    if (fresh || (prompt && !w.prompt)) {  // new sequence, or a prompt after decode (= a new request in the session): clear this request's share
      if (w.raw.empty()) w.raw.assign(prio_.size(), 0.f);
      for (int32_t k : w.touched) w.raw[k] = 0.f;
      w.touched.clear(); w.n = 0.f; w.scale = 1.f;
    }
    w.uid = uids[i]; w.live = true; w.prompt = prompt; w.last = pstep_;
    row_owner_[i] = o;
  }
}
void ExpertStore::observe_policy(size_t k, float w, int row) {
  // Prompt (prefill, short chunks — caller weight 1/rows < 1): request evidence capped at p_pre_ tokens' worth (prompt distribution != decode distribution — measured JSD 0.227)
  if (w < 1.f && p_pre_ > 0.f) w = std::min(1.f, w * p_pre_);
  const int l = (int)(k / E_);
  const int o = row < 0 || row_owner_.empty() ? -1 : row_owner_.size() == 1 ? row_owner_[0] : row < (int)row_owner_.size() ? row_owner_[row] : -1;
  if (o >= 0) {
    Owner& ow = owners_[o];
    if (ow.raw[k] == 0.f) ow.touched.push_back((int32_t)k);
    ow.raw[k] += w / ow.scale;
    if (l < n_backbone_) ow.n += w / (float)n_act_of(l) / (float)n_backbone_;  // token count (averaged over backbone layers)
  }
  if (w >= 1.f) { g_raw_[k] += 1.f / g_scale_; g_layer_raw_[l] += 1.f / g_scale_; }  // the global prior uses decode rows only (prompts have a different distribution)
}
void ExpertStore::decay_policy() {
  ++pstep_;
  g_scale_ *= p_gdecay_;
  if (g_scale_ < 1e-20f) { for (auto& v : g_raw_) v *= g_scale_; for (auto& v : g_layer_raw_) v *= g_scale_; g_scale_ = 1.f; }
  float A = 0.f;  // sum_o α/(n_o+α) — total prior share
  for (Owner& o : owners_) {
    if (!o.live) continue;
    if (pstep_ - o.last > (uint32_t)p_idle_) { o.live = false; for (int32_t k : o.touched) o.raw[k] = 0.f; o.touched.clear(); continue; }  // finished request
    A += p_alpha_ / (o.n + p_alpha_);
  }
  for (int l = 0; l < n_layers_; ++l) {  // prior share A·p_k — p_k = g_k / sum over layer g x n_act (per-token use probability). One coefficient per layer (no per-key division)
    const float c = g_layer_raw_[l] > 0.f ? A * (float)n_act_of(l) / g_layer_raw_[l] : 0.f;
    float* P = prio_.data() + (size_t)l * E_;
    const float* G = g_raw_.data() + (size_t)l * E_;
    for (int e = 0; e < E_; ++e) P[e] = c * G[e];
  }
  for (Owner& o : owners_) {
    if (!o.live) continue;
    const float b = o.scale / (o.n + p_alpha_);
    for (int32_t k : o.touched) prio_[k] += b * o.raw[k];
    o.scale *= p_odecay_; o.n *= p_odecay_;
    if (o.scale < 1e-20f) { for (int32_t k : o.touched) o.raw[k] *= o.scale; o.scale = 1.f; }
  }
}
// ---- D4 cache policy END

void ExpertStore::commit_all() {
  flush_promotions();  // E4: issue any pending pieces first (so we never wait on a batch event that has not been recorded)
  if (!promo_q_.empty()) CUDA_CHECK(cudaEventSynchronize(promo_q_.back().evt));
  commit_pending();
  HIVE_CHECK(promo_q_.empty(), "commit_all left batches");
}

void ExpertStore::run_jobs(std::vector<Job>& jobs) {
  start_jobs(jobs);
  wait_jobs();
}

double ExpertStore::jobs_done_ms() const {
  double t = 0;
  for (int n = 0; n < 2; ++n) if (pool_[n] && pool_[n]->n_items.load(std::memory_order_acquire) > 0) t = std::max(t, pool_[n]->t_done.load(std::memory_order_acquire));
  return t;
}

void ExpertStore::wait_jobs() {
  if (!jobs_active_) return;
  for (int n = 0; n < 2; ++n) {
    Pool* p = pool_[n];
    if (p->n_items.load(std::memory_order_acquire) == 0) continue;
    while (!p->fin.load(std::memory_order_acquire)) std::this_thread::yield();  // D7: until t_done has been published
  }
  jobs_active_ = false;
}

void ExpertStore::start_jobs(std::vector<Job>& jobs, bool owned) {
  HIVE_CHECK(!jobs_active_, "start_jobs while a batch is active");
  if (jobs.empty()) return;
  // Phase-1 chunks (inter rows, multiple of 32), phase-2 chunks (dim rows). Measured (oracle off, 16 threads/node + stealing): 96/160 -> 1 job 0.43 ms, 6 jobs 1.64 ms/layer;
  // 32/64 -> 0.35, 1.38 (adopted); 64/160 -> 0.36, 1.41; 128/256 -> 0.51, 1.47. Re-measure with HIVE_CPU_P1/P2.
  // WARNING: P1 must be a multiple of 32: phase 1's fp8 32-block quantization must close within a chunk (with 48 a block straddled two items and outputs blew up to 1e4).
  // Row split: every job emits items on both nodes — the pool of node n handles the row range of half record n (full row numbers [n*I/2, (n+1)*I/2) / [n*dim/2, ...)).
  static const int P1 = getenv("HIVE_CPU_P1") ? atoi(getenv("HIVE_CPU_P1")) : 32;
  static const int P2 = getenv("HIVE_CPU_P2") ? atoi(getenv("HIVE_CPU_P2")) : 32;
  HIVE_CHECK(P1 > 0 && P1 % 32 == 0 && P2 > 0, "HIVE_CPU_P1 must be a multiple of 32");
  const int I2 = (int)hlay_.w13_rows, D2 = (int)hlay_.w2_rows;
  HIVE_CHECK(I2 % 32 == 0, "half rows must close the fp8 32-block quantization");
  // D2: before rebuilding, close both pools (n_items=0, next=1<<30 — seq_cst) and wait until every straggling/stealing worker of the previous batch has left (busy==0).
  //   After this no worker reads items/n_items/jobs/p1_left until publication (next=0 below). If an exception occurs, the pools stay closed (n_items=0).
  for (int n = 0; n < 2; ++n) { pool_[n]->n_items.store(0); pool_[n]->next.store(1 << 30); }
  while (pool_busy_.load() != 0) std::this_thread::yield();
  if (p1_left_.size() < jobs.size()) { std::vector<std::atomic<int>> v(jobs.size()); p1_left_.swap(v); }
  const int nblk = (I2 + P1 - 1) / P1;  // phase-1 blocks per node
  // B2 HIVE_CPU_SPLIT13: two items per block, w1 and w3 (off = one). The pair counters p13_done_[job][node][block] are zeroed here.
  //   Rationale (bench-cpu on the target workstation, C1 GEMV2 on): 1 job 0.32 ms vs 6 jobs 1.59 ms (0.265 per job) — a single job is ~20% more expensive. 16 threads per node with 36
  //   phase-1 items = 2.25 waves (only 4 threads work in the third). Halving the items gives 72 = 4.5 waves. Whether this estimate holds is decided with bench_pool_cpu (A/B via the switch).
  // B3 HIVE_CPU_FINE (fine_ — off = default): only for multi-row jobs (R >= 2), split each phase-1 block into 4 pieces (w1/w3 x row halves — same pair-counter rule as SPLIT13, the fourth quantizes)
  //   and use P2/2 rows per phase-2 item. Rationale: with 16 threads per node, one R-row job = 36 phase-1 items = 2.25 waves (in the third wave only 4 threads work while 12 hold
  //   phase-2 items idling at the barrier) — an R=8 item costs ~2.5x an R=1 item, so that idle wave is long. 4 pieces -> 144 = 9 waves. R = 1 jobs are unchanged (or SPLIT13).
  auto parts_of = [&](const Job& jb) { return fine_ && jb.R >= 2 ? 4 : split13_ ? 2 : 1; };
  bool any_parts = false;
  for (const Job& jb : jobs) any_parts |= parts_of(jb) > 1;
  if (any_parts) {
    const size_t need = jobs.size() * 2 * (size_t)nblk;
    if (p13_done_.size() < need) { std::vector<std::atomic<int>> v(need); p13_done_.swap(v); }
    for (size_t i = 0; i != need; ++i) p13_done_[i].store(0, std::memory_order_relaxed);
  }
  for (size_t i = 0, nj = jobs.size(); i < nj; ++i) p1_left_[i].store(2 * parts_of(jobs[i]) * nblk);  // sum over both nodes (default = 2*nblk, SPLIT13 4*nblk)
  for (int n = 0; n < 2; ++n) {  // build the items of both pools first (stealing means the other pool is visible too), then wake them all at once
    Pool* p = pool_[n];
    p->done.store(0);
    p->t_done.store(0);
    p->fin.store(false);
    p->items.clear();
    p->p1_left = p1_left_.data();
    p->p13 = p13_done_.data(); p->nblk = nblk; p->p1_rows = P1; p->p2_prefetch = p2_prefetch_;
    const int r1_0 = n * I2, r2_0 = n * D2;
    for (size_t ji = 0; ji < jobs.size(); ++ji) {
      const int parts = parts_of(jobs[ji]);
      for (int r = 0; r < I2; r += P1) {
        const int b0 = r1_0 + r, b1 = r1_0 + std::min(r + P1, I2);
        if (parts == 1) { p->items.push_back({(int)ji, 1, n, b0, b1}); continue; }
        if (parts == 2) {  // SPLIT13: keep the pair adjacent (the second one quantizes)
          p->items.push_back({(int)ji, 1, n, b0, b1, 1, 2, b0, b1});
          p->items.push_back({(int)ji, 1, n, b0, b1, 3, 2, b0, b1});
          continue;
        }
        const int h = b0 + (b1 - b0 + 1) / 2;  // FINE: two row halves x w1/w3 (16 + 16 for a 32-row block)
        for (int m : {1, 3}) { p->items.push_back({(int)ji, 1, n, b0, h, m, 4, b0, b1}); p->items.push_back({(int)ji, 1, n, h, b1, m, 4, b0, b1}); }
      }
    }
    for (size_t ji = 0; ji < jobs.size(); ++ji) {
      const int p2 = fine_ && jobs[ji].R >= 2 ? std::max(1, P2 / 2) : P2;  // B3 FINE: phase-2 items of multi-row jobs = P2/2 rows (value-neutral — each row accumulates independently)
      for (int r = 0; r < D2; r += p2) p->items.push_back({(int)ji, 2, n, r2_0 + r, r2_0 + std::min(r + p2, D2)});
    }
    // P1 HIVE_DECODE_PREGATE owned batch: each item gets an owning worker (pg::owner — same numbering as the prefetch pg::owned_spans: phase-1 block b -> b, phase-2 block r -> nblk + r).
    //   Batches with pieces (SPLIT13, FINE) keep pick-as-you-go (the pair-counter rule does not assume a pick order, but it does not match the prefetch numbering).
    const bool own = owned && owned_ && !any_parts;
    if (own) {
      const int T = (int)p->threads.size();
      for (auto& it : p->items) {
        const int idx = it.phase == 1 ? (it.r0 - r1_0) / P1 : nblk + (it.r0 - r2_0) / P2;
        it.own = pg::owner(jobs[(size_t)it.job].e, idx, T);
      }
    }
    p->owned_batch.store(own, std::memory_order_relaxed);  // carried by the publication below (n_items release, next seq_cst, epoch acq_rel)
    p->jobs = &jobs;
    p->dim = cfg_.dim; p->inter = cfg_.moe_inter;
  }
  // Allocation/type exceptions above have not published work to either pool.
  // Mark active only once both task lists are complete, or wait_jobs could wait
  // forever on an unstarted partial list during host-error recovery.
  jobs_active_ = true;
  for (int n = 0; n < 2; ++n) pool_[n]->n_items.store((int)pool_[n]->items.size(), std::memory_order_release);
  for (int n = 0; n < 2; ++n) pool_[n]->next.store(0);  // publish (seq_cst ⊇ release) — a worker that reads this value via fetch_add sees items/n_items above
  for (int n = 0; n < 2; ++n) {
    Pool* p = pool_[n];
    { std::lock_guard<std::mutex> lk(p->mu); p->epoch.fetch_add(1, std::memory_order_acq_rel); }
    p->cv.notify_all();
  }
}


// P1 prefetch request (engine thread) — written via the seqlock, then both pools are woken. Does nothing outside owned mode. Non-resident selection and order are up to the caller (pg::plan).
void ExpertStore::prefetch_experts(int l, const int* e, int n) {
  if (!pf_ || n <= 0) return;
  n = std::min(n, pg::kMaxPf);
  PfReq& R = *pf_;
  const uint32_t s = R.seq.load(std::memory_order_relaxed);
  R.seq.exchange(s + 1, std::memory_order_acq_rel);  // odd = writing; acquire: the element writes below cannot move ahead of this (no fence needed — TSAN)
  R.l.store(l, std::memory_order_relaxed);
  R.n.store(n, std::memory_order_relaxed);
  for (int i = 0; i < n; ++i) R.e[i].store(e[i], std::memory_order_relaxed);
  R.seq.store(s + 2, std::memory_order_release);
  R.req.fetch_add(1, std::memory_order_relaxed);
  R.experts.fetch_add(n, std::memory_order_relaxed);
  for (int k = 0; k < 2; ++k) {
    if (!pool_[k]) continue;
    { std::lock_guard<std::mutex> lk(pool_[k]->mu); }  // so we do not race a sleeping worker's predicate check (wake after taking the lock — same sequence as start_jobs)
    pool_[k]->cv.notify_all();
  }
}

ExpertStore::PfStats ExpertStore::pf_stats() const {
  PfStats o;
  if (!pf_) return o;
  o.req = pf_->req.load(); o.experts = pf_->experts.load(); o.kib = pf_->kib.load(); o.aborted = pf_->aborted.load(); o.finished = pf_->finished.load();
  return o;
}
}  // namespace hive
