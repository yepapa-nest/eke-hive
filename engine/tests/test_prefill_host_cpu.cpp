// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU-only checks of the host-side logic behind the round-2 prefill options. The helpers are the PRODUCTION text
// of engine/src/runtime.cpp (the anonymous-namespace block of round-2 host helpers, ending at its "END" marker, sliced into prefill_helpers_gen.h by
// tools/cpu_fake/harness.py) and hive::VitCache (include/hive/vision.h). Built/run by tools/test_snapshot_cpu.py. No GPU.
//
//  1. par_rows: every row covered exactly once, at most `threads` chunks, one chunk = caller thread, exceptions rethrown after join.
//  2. L2 engram_gather (HIVE_ENGRAM_PAR): parallel output bytes == sequential (threads 1 = the pre-switch loop) for many M.
//  3. L1 unpack_acts via par_rows (HIVE_EARLY_STREAM LUT unpack): parallel == sequential == e4m3/e8m0 reference, bitwise.
//  4. L3 idx_msub_for (HIVE_IDX_MSUB_ACTUAL): never below the constructor Msub, never above max(Msub, min(M, 65535)), and the
//     four indexer scratch buffers stay within their constructor capacity; a row-independent model of the indexer loop gives the
//     same idx table for both sub-chunk sizes.
//  5. L1 plan_early_stream: simulated staging ring (prefetch slots, residents, CPU share) — the early-issue schedule gives every
//     streamed expert the same slot as the pre-switch loop, issues side_ copies in the same order, and never overwrites a slot
//     whose previous occupant has not been released (the stage_freed_ event it would wait on is recorded).
//  6. M3 VitCache (HIVE_VIT_CACHE_MB): exact-byte hits only (same hash + different bytes/types/grid = miss), LRU order, byte cap,
//     oversized entry absorbed.
// GPU-only (NOT covered here): the CUDA issue itself (copy_to_staging/event records on side_/st_ in moe_experts_multi, the D2H/H2D
// of merge_images, indexer kernels with a larger Ms), and every numerical/performance effect.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hive/vision.h"
#include "prefill_helpers_gen.h"

using namespace hive;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static void t_par_rows() {
  const std::thread::id me = std::this_thread::get_id();
  for (int n : {0, 1, 5, 63, 64, 127, 128, 1000, 4097})
    for (int threads : {1, 2, 3, 8, 32})
      for (int min_per : {1, 64}) {
        std::vector<std::atomic<int>> hit(n);
        std::atomic<int> chunks{0}, on_caller{0};
        par_rows(n, threads, min_per, [&](int a, int b) {
          ++chunks;
          if (std::this_thread::get_id() == me) ++on_caller;
          for (int i = a; i < b; ++i) ++hit[i];
        });
        int bad = 0;
        for (auto& h : hit) bad += h.load() != 1;
        EXPECT(bad == 0, "par_rows n %d threads %d: %d rows not covered exactly once", n, threads, bad);
        EXPECT(chunks.load() <= std::max(1, threads) && (n == 0 || on_caller.load() == 1), "par_rows chunks %d caller %d", chunks.load(), on_caller.load());
        if (threads <= 1 || n < 2 * min_per) EXPECT(chunks.load() == (n > 0 ? 1 : 0), "sequential path split into %d chunks", chunks.load());
      }
  bool threw = false;
  try { par_rows(1000, 4, 1, [](int a, int) { if (a > 0) throw std::runtime_error("x"); }); } catch (const std::runtime_error&) { threw = true; }
  EXPECT(threw, "par_rows lost a worker exception");
  printf("prefill host CPU: par_rows coverage/chunking/exception OK\n");
}

static void t_engram() {
  std::mt19937 rng(7);
  const int hd = 256, cols = 24, nL = 2;
  const int64_t rows = 5000;
  std::vector<uint8_t> vals((size_t)rows * hd), scales((size_t)rows * hd / 32);
  for (auto& v : vals) v = (uint8_t)rng();
  for (auto& v : scales) v = (uint8_t)rng();
  int cases = 0;
  for (int M : {1, 7, 63, 64, 129, 300, 1000, 2049})
    for (int hidx : {0, 1}) {
      std::vector<int64_t> h((size_t)M * nL * cols);
      for (auto& x : h) x = (int64_t)(rng() % rows);
      std::vector<uint8_t> v1((size_t)M * cols * hd, 0xAA), s1((size_t)M * cols * hd / 32, 0xAA);
      engram_gather(h.data(), M, nL, hidx, cols, hd, rows, vals.data(), scales.data(), v1.data(), s1.data(), 1);
      for (int threads : {2, 3, 8, 64}) {
        std::vector<uint8_t> v2(v1.size(), 0x55), s2(s1.size(), 0x55);
        engram_gather(h.data(), M, nL, hidx, cols, hd, rows, vals.data(), scales.data(), v2.data(), s2.data(), threads);
        EXPECT(v1 == v2 && s1 == s2, "engram_gather M %d hidx %d threads %d: parallel bytes differ", M, hidx, threads);
        ++cases;
      }
      // spot reference: row m, col c = table row h[(m·nL + hidx)·cols + c]
      const int m = M - 1, c = cols - 1;
      const int64_t r = h[((size_t)m * nL + hidx) * cols + c];
      EXPECT(!memcmp(v1.data() + ((size_t)m * cols + c) * hd, vals.data() + (size_t)r * hd, hd), "engram_gather reference row");
    }
  printf("prefill host CPU: L2 engram_gather parallel == sequential (%d cases)\n", cases);
}

static void t_unpack() {
  std::mt19937 rng(9);
  const int dim = 512;
  int cases = 0;
  for (int g : {1, 63, 64, 200, 1500}) {
    std::vector<uint8_t> xq((size_t)g * dim), xs((size_t)g * dim / 32);
    for (auto& v : xq) v = (uint8_t)rng();
    for (auto& v : xs) v = (uint8_t)(100 + rng() % 40);
    std::vector<float> af1((size_t)g * dim), as1((size_t)g * dim / 32);
    par_rows(g, 1, kParMinRows, [&](int a, int b) { unpack_acts(xq.data(), xs.data(), a, b, dim, af1.data(), as1.data()); });
    for (int threads : {2, 8, 32}) {
      std::vector<float> af2(af1.size(), -1.f), as2(as1.size(), -1.f);
      par_rows(g, threads, kParMinRows, [&](int a, int b) { unpack_acts(xq.data(), xs.data(), a, b, dim, af2.data(), as2.data()); });
      EXPECT(!memcmp(af1.data(), af2.data(), af1.size() * 4) && !memcmp(as1.data(), as2.data(), as1.size() * 4), "unpack g %d threads %d differ", g, threads);
      ++cases;
    }
    for (size_t i = 0; i < xq.size(); i += 97) {
      const float ref = e4m3_to_f32(xq[i]);
      EXPECT(!memcmp(&af1[i], &ref, 4), "unpack value vs e4m3_to_f32");
    }
    for (size_t i = 0; i < xs.size(); i += 7) {
      const float ref = e8m0_to_f32(xs[i]);
      EXPECT(!memcmp(&as1[i], &ref, 4), "unpack scale vs e8m0_to_f32");
    }
  }
  printf("prefill host CPU: L1 LUT unpack parallel == sequential == reference (%d cases)\n", cases);
}

static void t_idx_msub() {
  std::mt19937 rng(3);
  const long long budget = 32ll << 20;
  const int bs = 8, CB = 2048, topk_idx = 512, topk_w = std::max(topk_idx, CB);
  int cases = 0, grew = 0;
  for (int64_t max_ctx : {4096ll, 65536ll, 262144ll, 1ll << 20})
    for (int WM : {64, 2048, 16384}) {
      // constructor formula (runtime.cpp Runtime::Runtime)
      const int Tcap = (int)max_ctx, Msub = std::max(1, std::min(WM, (int)(budget / std::max(1, Tcap))));
      const int nblocks_cap = (Tcap + bs - 1) / bs;
      for (int it = 0; it < 400; ++it) {
        const int M = 1 + (int)(rng() % WM);
        const int T = 1 + (int)(rng() % Tcap);
        const int r = idx_msub_for(Msub, Tcap, nblocks_cap, topk_w, M, T, bs, CB, topk_idx);
        ++cases;
        EXPECT(r >= Msub, "msub %d below constructor %d", r, Msub);
        EXPECT(r <= std::max(Msub, std::min(M, 65535)), "msub %d above bound (M %d)", r, M);
        if (r > Msub) {
          ++grew;
          const int nblocks = (T + bs - 1) / bs;
          const long long cols = std::max<long long>(T, (long long)nblocks * bs);
          const long long used_w = std::max(std::min(topk_idx, T), std::min(CB, nblocks));
          EXPECT((long long)r * cols <= (long long)Msub * (Tcap + 8), "iscore overflow: r %d cols %lld cap %lld", r, cols, (long long)Msub * (Tcap + 8));
          EXPECT((long long)r * nblocks <= (long long)Msub * nblocks_cap, "bmax overflow");
          EXPECT((long long)r * used_w <= (long long)Msub * topk_w, "topk_pos overflow");
          EXPECT(r <= 65535, "grid.y overflow");
        }
      }
    }
  // row-independent model of the indexer loop: idx row m = f(m) written at m0 offsets; both sub-chunk sizes -> same table
  for (int it = 0; it < 200; ++it) {
    const int M = 1 + (int)(rng() % 3000), a = 1 + (int)(rng() % 200), b = a + (int)(rng() % 3000);
    auto table = [&](int Ms0) {
      std::vector<int> idx((size_t)M * 3, -1);
      for (int m0 = 0; m0 < M; m0 += Ms0) {
        const int Ms = std::min(Ms0, M - m0);
        for (int i = 0; i < Ms; ++i) for (int j = 0; j < 3; ++j) idx[(size_t)(m0 + i) * 3 + j] = (m0 + i) * 7 + j;  // per-row function of the row only
      }
      return idx;
    };
    EXPECT(table(a) == table(b), "sub-chunk size changed the row-independent result");
  }
  printf("prefill host CPU: L3 idx_msub_for bounds/capacity OK (%d cases, %d larger than the constructor Msub)\n", cases, grew);
}

// Staging-ring simulation. A slot's content is "live" from its copy until its consumer records stage_freed_ (host order).
// A copy into slot si waits the freed record that exists at issue time; it is safe iff the previous occupant is released.
struct RingSim {
  int S;
  uint32_t next;
  std::vector<int> occupant;       // expert whose copy is the latest in slot (-1 none)
  std::vector<char> released;      // occupant released (freed recorded) — or no occupant
  std::vector<std::pair<int, int>> copies;  // side_ copy order (expert, slot)
  std::map<int, int> slot_of;      // expert -> slot it was copied into (this layer)
  int unsafe = 0;
  RingSim(int S_, uint32_t n0) : S(S_), next(n0), occupant(S_, -1), released(S_, 1) {}
  int copy(int e, int si) {
    if (!released[si]) ++unsafe;
    occupant[si] = e; released[si] = 0; copies.push_back({e, si}); slot_of[e] = si;
    return si;
  }
  void release(int si) { released[si] = 1; }
};

static void t_early_plan() {
  std::mt19937 rng(21);
  int scenarios = 0, early_total = 0, stopped_busy = 0;
  for (int it = 0; it < 4000; ++it) {
    const int S = 8 + (int)(rng() % 3) * 8, E = 16 + (int)(rng() % 48);
    const uint32_t q0 = (uint32_t)(rng() % 1000);
    std::vector<char> routed(E), resident(E), cpu(E);
    for (int e = 0; e < E; ++e) { routed[e] = rng() % 10 != 0; resident[e] = rng() % 3 == 0; cpu[e] = !resident[e] && rng() % 4 == 0; }
    // prefetch: p experts (non-resident) copied first, ring positions q0.., some later unused (routed = 0)
    const int p = (int)(rng() % (S + 1));
    std::vector<int> pf_e;
    for (int e = 0; e < E && (int)pf_e.size() < p; ++e) if (!resident[e] && rng() % 2) pf_e.push_back(e);
    auto run = [&](bool early) {
      RingSim sim(S, q0);
      std::vector<int> pf_slot(E, -1);
      std::vector<std::pair<int, int>> pf;
      for (int e : pf_e) { const int si = (int)(sim.next++ % (uint32_t)S); sim.copy(e, si); pf.push_back({e, si}); pf_slot[e] = si; }
      for (int e : pf_e) cpu[e] = 0;  // production: prefetched experts are never CPU share
      // build_order (production order rule)
      std::vector<int> order;
      std::vector<char> busy(S, 0);
      for (auto [e, si] : pf) { if (routed[e]) { order.push_back(e); busy[si] = 1; } else sim.release(si); }
      for (int e = 0; e < E; ++e) if (pf_slot[e] < 0) order.push_back(e);
      std::vector<int> early_si(E, -1);
      if (early) {
        std::vector<std::pair<int, int>> plan;
        plan_early_stream(order, [&](int e) { return routed[e] && !cpu[e] && pf_slot[e] < 0 && !resident[e]; }, busy, S, sim.next, plan);
        for (auto [e, si] : plan) { sim.copy(e, si); early_si[e] = si; }
        early_total += (int)plan.size();
        if ((int)plan.size() < S) {
          int streamed = 0;
          for (int e : order) streamed += routed[e] && !cpu[e] && pf_slot[e] < 0 && !resident[e];
          if ((int)plan.size() < streamed) ++stopped_busy;
        }
      }
      // expert loop (production rule): consume in order; streamed experts copy on demand unless early-issued
      std::vector<int> consumed;
      for (int e : order) {
        if (!routed[e] || cpu[e]) continue;
        int si = -1;
        if (pf_slot[e] >= 0) si = pf_slot[e];
        else if (early_si[e] >= 0) si = early_si[e];
        else if (resident[e]) { consumed.push_back(e); continue; }
        else si = sim.copy(e, (int)(sim.next++ % (uint32_t)S));
        if (sim.occupant[si] != e) ++sim.unsafe;  // the slot must still hold this expert when it is consumed
        consumed.push_back(e);
        sim.release(si);
      }
      return std::make_tuple(sim.copies, sim.slot_of, consumed, sim.unsafe, sim.next);
    };
    const auto a = run(false), b = run(true);
    ++scenarios;
    EXPECT(std::get<3>(a) == 0, "pre-switch loop unsafe in scenario %d (simulation bug)", it);
    EXPECT(std::get<3>(b) == 0, "early schedule overwrote an unreleased slot or lost a copy (scenario %d)", it);
    EXPECT(std::get<0>(a) == std::get<0>(b), "side_ copy order differs (scenario %d)", it);
    EXPECT(std::get<1>(a) == std::get<1>(b), "expert -> slot mapping differs (scenario %d)", it);
    EXPECT(std::get<2>(a) == std::get<2>(b), "consumption order differs (scenario %d)", it);
    EXPECT(std::get<4>(a) == std::get<4>(b), "stage_next_ differs after the layer (scenario %d)", it);
    if (fails > 5) break;
  }
  printf("prefill host CPU: L1 early-stream ring plan == pre-switch loop (%d scenarios, %d early copies, %d stopped at a prefetch slot/ring)\n",
         scenarios, early_total, stopped_busy);
}

static void t_vit_cache() {
  auto entry = [](uint8_t fill, size_t np, size_t nr, int h, int w, std::vector<int8_t> types) {
    VitCache::Entry e;
    e.patches.assign(np, fill); e.rows.assign(nr, (uint8_t)(fill ^ 0xFF)); e.n_vit_h = h; e.n_vit_w = w; e.types = types;
    e.hash = VitCache::hash_of(e.patches.data(), np, h, w, types.data(), types.size());
    return e;
  };
  const std::vector<int8_t> ty{0, 1, 1, 2, 1, 1, 3};
  {
    VitCache c(1 << 20);
    auto e = entry(7, 4096, 2048, 4, 8, ty);
    const auto p = e.patches;
    c.insert(std::move(e));
    const uint64_t h = VitCache::hash_of(p.data(), p.size(), 4, 8, ty.data(), ty.size());
    const VitCache::Entry* hit = c.find(h, p.data(), p.size(), 4, 8, ty.data(), ty.size());
    EXPECT(hit && hit->rows.size() == 2048 && hit->rows[0] == (uint8_t)(7 ^ 0xFF), "exact hit");
    auto q = p; q[4095] ^= 1;  // one byte different, SAME hash passed (forced collision) -> must miss
    EXPECT(!c.find(h, q.data(), q.size(), 4, 8, ty.data(), ty.size()), "hash collision returned a different image");
    EXPECT(!c.find(h, p.data(), p.size(), 8, 4, ty.data(), ty.size()), "different grid hit");
    auto ty2 = ty; ty2[3] = 1;
    EXPECT(!c.find(h, p.data(), p.size(), 4, 8, ty2.data(), ty2.size()), "different types hit");
    EXPECT(!c.find(h, p.data(), p.size() - 1, 4, 8, ty.data(), ty.size()), "different length hit");
    EXPECT(VitCache::hash_of(q.data(), q.size(), 4, 8, ty.data(), ty.size()) != h, "hash_of ignores a byte");
    EXPECT(c.hits() == 1 && c.misses() == 4, "hit/miss counters %llu/%llu", (unsigned long long)c.hits(), (unsigned long long)c.misses());
  }
  {
    const size_t one = entry(0, 1000, 1000, 1, 1, ty).bytes();
    VitCache c(3 * one + one / 2);  // room for 3
    for (uint8_t f = 1; f <= 3; ++f) c.insert(entry(f, 1000, 1000, 1, 1, ty));
    auto touch = [&](uint8_t f) {
      std::vector<uint8_t> p(1000, f);
      return c.find(VitCache::hash_of(p.data(), 1000, 1, 1, ty.data(), ty.size()), p.data(), 1000, 1, 1, ty.data(), ty.size()) != nullptr;
    };
    EXPECT(touch(1), "entry 1 present");  // 1 becomes most recent -> LRU victim is 2
    c.insert(entry(4, 1000, 1000, 1, 1, ty));
    EXPECT(c.size() == 3 && c.bytes() <= c.cap() && c.evicted() == 1, "cap/evict: size %zu bytes %zu cap %zu", c.size(), c.bytes(), c.cap());
    EXPECT(touch(1) && !touch(2) && touch(3) && touch(4), "LRU evicted the wrong entry");
    c.insert(entry(9, 4 * one, 10, 1, 1, ty));  // larger than the whole cap: absorbed, nothing evicted
    EXPECT(c.skipped() == 1 && c.size() == 3 && c.evicted() == 1, "oversized entry not absorbed");
  }
  printf("prefill host CPU: M3 VitCache exact-byte hits only, LRU, byte cap, oversize absorbed\n");
}

int main() {
  t_par_rows();
  t_engram();
  t_unpack();
  t_idx_msub();
  t_early_plan();
  t_vit_cache();
  return fails ? 1 : 0;
}
