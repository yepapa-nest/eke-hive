// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_ENGRAM_SSD — CPU test (fake CUDA/NUMA, no GPU). Built and run by tools/test_engram_ssd_cpu.py (argument = a work directory that supports O_DIRECT).
//
// Reference (RAM mode) = the **real** engram_gather from runtime.cpp (prefill_helpers_gen.h, cut out by the harness) — over a copy of the tables read fully into RAM.
//  1. peek    : EngramHash::peek (history unchanged) == EngramHash::compute (history advances) — random token sequences, image markers, short histories (0·1·2), several pieces.
//  2. gather  : EngramSsd::gather == engram_gather bytes (values, scales) — modes 1/2 × cache rows 0·1·7·64·5000·all × read threads 1·3·64 × prefetch on/off ×
//               lookup threads 1·8 × consecutive decode-like (M 1–8) and prefill-like (M 200·1500) steps (layer 1 → layer 14 order · forward-head prefetch = same formula as Runtime::engram_ssd_hint).
//               Checks that the prefetch reads the right rows: with a large cache, looking up after the prefetch queue drains must hit the cache for every row (wrong row addresses → hits < rows).
//  3. concurrency: 4 sequences prefetch and look up concurrently on 4 threads (small 300-row cache — eviction, protection and wait contention) → all equal the reference.
//  4. failures: transient EIO (first attempt only) → same bytes · retries > 0 · failed 0 / one EINVAL (O_DIRECT alignment) → absorbed with 4 KiB alignment / persistent EIO →
//               only that lookup throws std::runtime_error("engram ssd read failed …") · other lookups on the same store are fine · after clearing the fault the same lookup equals the reference
//               (failed rows are not left in the cache) · also with cache 0 (direct reads) / failure during prefetch → the lookup reads again. Out-of-range row = the same "engram row range" exception.
//  5. store   : real ExpertStore + real safetensors shards (data start not 4 KiB-aligned · table 1 has its scales in another shard): with HIVE_ENGRAM_SSD=1/2, load_engram and
//               load_bulk (LOAD_PAR path) — gather of the store == engram_gather over the tables of a RAM store · value tables use 0 B of RAM · mode 2 scales 0 B of RAM · engram_host_bytes ≤
//               cache + scales · HostPrefault::predict excludes SSD tables (mode 1: 4 regions · mode 2: 2 regions · off: 6).
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hive/bulk_load.h"
#include "hive/engram_hash.h"
#include "hive/engram_ssd.h"
#include "hive/expert_store.h"
#include "hive/safetensors.h"
#include "nlohmann/json.hpp"
#include "prefill_helpers_gen.h"  // engram_gather from runtime.cpp (the real one)

using namespace hive;
namespace fs = std::filesystem;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { if (fails < 40) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } ++fails; } } while (0)

namespace {

constexpr int NL = 2, NG = 4, NH = 8, COLS = (NG - 1) * NH, HD = 256, SB = 8, VOCAB = 3000;

// ---- Fake hash config (structure = the real engram_hash.json: 2 layers · 4-gram · 8 heads) ----
struct FakeHash {
  std::string json, tmap;
  int64_t rows[NL] = {0, 0};
  void build(const std::string& dir, std::mt19937_64& rng) {
    nlohmann::json j;
    j["layer_ids"] = {1, 14};
    j["max_ngram"] = NG; j["n_heads"] = NH; j["pad_id"] = 2;
    std::vector<std::vector<std::vector<int64_t>>> primes(NL);
    std::vector<std::vector<int64_t>> offs(NL), mult(NL);
    int64_t p = 1009;
    auto next_prime = [&](int64_t x) { for (;; ++x) { bool ok = x > 1; for (int64_t d = 2; d * d <= x && ok; ++d) ok = x % d != 0; if (ok) return x; } };
    for (int l = 0; l < NL; ++l) {
      int64_t off = 0;
      primes[l].resize(NG - 1);
      for (int i = 0; i < NG - 1; ++i)
        for (int h = 0; h < NH; ++h) {
          p = next_prime(p + 1 + (int64_t)(rng() % 40));
          primes[l][i].push_back(p);
          offs[l].push_back(off);
          off += p;
        }
      rows[l] = off;
      for (int i = 0; i < NG; ++i) mult[l].push_back((int64_t)(rng() >> 17) | 1);  // large multipliers (int64 wraparound · negative remainder path)
    }
    j["primes"] = primes; j["offsets"] = offs; j["multipliers"] = mult;
    json = dir + "/engram_hash.json"; tmap = dir + "/token_map.bin";
    std::ofstream(json) << j.dump();
    std::vector<int32_t> tm(VOCAB);
    for (auto& t : tm) t = (int32_t)(rng() % 700);  // compressed vocab (several ids map to one compressed id — as in the real one)
    std::ofstream(tmap, std::ios::binary).write((const char*)tm.data(), (std::streamsize)tm.size() * 4);
  }
};

std::vector<uint8_t> rand_bytes(std::mt19937_64& rng, size_t n) { std::vector<uint8_t> v(n); for (auto& b : v) b = (uint8_t)rng(); return v; }

// ---- Fake table files: table 0 = one file with [664 B header][values][scales] (same layout as the real model-00047) · table 1 = separate value and scale files ----
struct Tables {
  std::string vpath[NL], spath[NL];
  uint64_t voff[NL], soff[NL];
  std::vector<uint8_t> vals[NL], scales[NL];  // RAM reference copies
  void build(const std::string& dir, const FakeHash& fh, std::mt19937_64& rng) {
    for (int l = 0; l < NL; ++l) {
      vals[l] = rand_bytes(rng, (size_t)fh.rows[l] * HD);
      scales[l] = rand_bytes(rng, (size_t)fh.rows[l] * SB);
    }
    vpath[0] = spath[0] = dir + "/table0.bin";
    voff[0] = 664; soff[0] = 664 + vals[0].size();
    {
      std::ofstream f(vpath[0], std::ios::binary);
      std::vector<uint8_t> head = rand_bytes(rng, 664);
      f.write((const char*)head.data(), 664);
      f.write((const char*)vals[0].data(), (std::streamsize)vals[0].size());
      f.write((const char*)scales[0].data(), (std::streamsize)scales[0].size());
      std::vector<uint8_t> tail = rand_bytes(rng, 37);  // file end is not a multiple of 512
      f.write((const char*)tail.data(), 37);
    }
    vpath[1] = dir + "/table1_vals.bin"; spath[1] = dir + "/table1_scales.bin";
    voff[1] = 1000; soff[1] = 8;
    {
      std::ofstream f(vpath[1], std::ios::binary);
      std::vector<uint8_t> head = rand_bytes(rng, 1000);
      f.write((const char*)head.data(), 1000);
      f.write((const char*)vals[1].data(), (std::streamsize)vals[1].size());  // file end = end of the last row (not a multiple of 512 → short-read path)
    }
    {
      std::ofstream f(spath[1], std::ios::binary);
      const uint64_t z = 0;
      f.write((const char*)&z, 8);
      f.write((const char*)scales[1].data(), (std::streamsize)scales[1].size());
    }
  }
  void attach(EngramSsd& s, int mode) const {
    for (int l = 0; l < NL; ++l)
      s.add_table(l, vpath[l], voff[l], spath[l], soff[l], (int64_t)(vals[l].size() / HD), HD, mode == 1 ? scales[l].data() : nullptr);
  }
};

// Reference: engram_gather from runtime.cpp (threads 1 = sequential loop)
void ref_gather(const Tables& T, const std::vector<int64_t>& h, int M, int hidx, std::vector<uint8_t>& v, std::vector<uint8_t>& s) {
  v.assign((size_t)M * COLS * HD, 0xA5); s.assign((size_t)M * COLS * SB, 0x5A);
  engram_gather(h.data(), M, NL, hidx, COLS, HD, (int64_t)(T.vals[hidx].size() / HD), T.vals[hidx].data(), T.scales[hidx].data(), v.data(), s.data(), 1);
}

// Same formula as Runtime::engram_ssd_hint (single sequence): last ≤ NG−1 history tokens + start position → the worker peeks
void hint(EngramSsd& ssd, const EngramHash& eh, const std::vector<int64_t>& hist, const std::vector<int32_t>& ids, const std::vector<int8_t>& img) {
  const int64_t start = (int64_t)hist.size();
  std::vector<int64_t> tail(hist.begin() + std::max<int64_t>(0, start - (NG - 1)), hist.end());
  const int M = (int)ids.size();
  ssd.prefetch([&eh, tail, start, ids, img, M](std::vector<int64_t>& out) { eh.peek(ids.data(), img.data(), M, tail.data(), (int)tail.size(), start, out); },
               M, NL, COLS, 0);
}

// ---- 1. peek == compute ----
void t_peek(const EngramHash& eh, std::mt19937_64& rng) {
  int bad = 0, n = 0;
  for (int round = 0; round < 200; ++round) {
    std::vector<int64_t> hist;
    const int pieces = 1 + (int)(rng() % 6);
    for (int k = 0; k < pieces; ++k) {
      const int M = 1 + (int)(rng() % (k == 0 && round % 3 == 0 ? 2 : 60));
      std::vector<int32_t> ids(M); std::vector<int8_t> img(M);
      for (int m = 0; m < M; ++m) { ids[m] = (int32_t)(rng() % VOCAB); img[m] = (rng() % 9 == 0); }
      const int64_t start = (int64_t)hist.size();
      std::vector<int64_t> tail(hist.begin() + std::max<int64_t>(0, start - (NG - 1)), hist.end());
      std::vector<int64_t> a, b;
      eh.peek(ids.data(), img.data(), M, tail.data(), (int)tail.size(), start, a);
      const size_t before = hist.size();
      eh.compute(ids.data(), img.data(), M, hist, b);
      bad += a != b;
      bad += hist.size() != before + (size_t)M;  // only compute extends the history
      ++n;
    }
  }
  EXPECT(bad == 0, "peek != compute in %d of %d pieces", bad, n);
  printf("engram ssd CPU: peek == compute (%d pieces, images, short histories)\n", n);
}

// ---- 2. gather == engram_gather ----
struct Variant { int mode; size_t cache_rows; int threads; bool prefetch; int gthreads; };

size_t cache_bytes_for(int mode, size_t rows) { return rows * (256 + (mode == 2 ? 8 : 0) + 2 * sizeof(EngramRowCache::Entry) + 16); }

int run_steps(EngramSsd& ssd, const EngramHash& eh, const Tables& T, std::mt19937_64& rng, int steps, bool prefetch, int gthreads, bool check_prefetch_hits,
              const char* what) {
  std::vector<int64_t> hist;
  int bad = 0;
  for (int st = 0; st < steps; ++st) {
    const int M = st % 5 == 0 ? (st % 10 == 0 ? 1500 : 200) : 1 + (int)(rng() % 8);
    std::vector<int32_t> ids(M); std::vector<int8_t> img(M);
    for (int m = 0; m < M; ++m) { ids[m] = (int32_t)((rng() % 4 == 0) ? (rng() % 40) : rng() % VOCAB); img[m] = (rng() % 50 == 0); }  // mix in repeated n-grams
    if (prefetch) hint(ssd, eh, hist, ids, img);
    if (check_prefetch_hits) ssd.quiesce();
    std::vector<int64_t> h;
    eh.compute(ids.data(), img.data(), M, hist, h);
    for (int hidx = 0; hidx < NL; ++hidx) {
      if (hidx == 0 && prefetch) ssd.prefetch_hashes(h.data(), M, NL, COLS, 1);  // same as runtime engram_host layer 1
      std::vector<uint8_t> rv, rs, v((size_t)M * COLS * HD, 0x11), s((size_t)M * COLS * SB, 0x22);
      ref_gather(T, h, M, hidx, rv, rs);
      const EngramSsd::GatherInfo g = ssd.gather(h.data(), M, NL, hidx, COLS, HD, (int64_t)(T.vals[hidx].size() / HD), v.data(), s.data(), gthreads);
      const bool same = v == rv && s == rs;
      bad += !same;
      if (!same && bad < 3) fprintf(stderr, "  %s: step %d M %d hidx %d differs\n", what, st, M, hidx);
      if (check_prefetch_hits) EXPECT(g.hit == g.rows, "%s: prefetch missed rows (step %d hidx %d: %llu/%llu cached)", what, st, hidx, (unsigned long long)g.hit,
                                      (unsigned long long)g.rows);
    }
  }
  return bad;
}

// HIVE_TEST_QUICK=1 (CI): a reduced matrix — threads 1/3 (no 64), cache rows 0/7/5000/all, fewer steps. The full matrix runs on maintainer
//   machines (tools/test_cpu.sh without the variable); on a 2-vCPU runner with the sanitizer the full matrix exceeded the suite's time budget.
bool quick_mode() { const char* v = getenv("HIVE_TEST_QUICK"); return v && *v && strcmp(v, "0") != 0; }

void t_gather(const EngramHash& eh, const Tables& T, std::mt19937_64& rng) {
  const size_t all = (size_t)(T.vals[0].size() + T.vals[1].size()) / HD + 10;
  const bool quick = quick_mode();
  std::vector<Variant> vs;
  const std::vector<size_t> cache_rows = quick ? std::vector<size_t>{0, 7, 5000, all} : std::vector<size_t>{0, 1, 7, 64, 5000, all};
  const std::vector<int> threads = quick ? std::vector<int>{1, 3} : std::vector<int>{1, 3, 64};
  for (int mode : {1, 2})
    for (size_t cr : cache_rows)
      for (int th : threads)
        for (bool pf : {true, false}) vs.push_back({mode, cr, th, pf, (th + (int)cr) % 2 ? 8 : 1});
  int bad = 0, nv = 0;
  for (const Variant& v : vs) {
    EngramSsdOpts o; o.mode = v.mode; o.cache_bytes = cache_bytes_for(v.mode, v.cache_rows); o.threads = v.threads; o.prefetch = v.prefetch;
    EngramSsd ssd(o);
    T.attach(ssd, v.mode);
    ssd.alloc_cache();
    char what[160];
    snprintf(what, sizeof what, "mode %d cache %zu rows threads %d prefetch %d gthreads %d", v.mode, ssd.cache_rows(), v.threads, v.prefetch, v.gthreads);
    const int b = run_steps(ssd, eh, T, rng, quick ? 8 : 12, v.prefetch, v.gthreads, false, what);
    EXPECT(b == 0, "%s: %d gather(s) differ from engram_gather", what, b);
    const EngramSsd::Stats st = ssd.stats();
    EXPECT(st.failed == 0 && st.retries == 0, "%s: failed %llu retries %llu", what, (unsigned long long)st.failed, (unsigned long long)st.retries);
    if (v.cache_rows == 0) EXPECT(st.bypass == st.rows && st.hit == 0, "%s: cache 0 must read every row directly", what);
    bad += b; ++nv;
  }
  // the prefetch reads exactly those rows (large cache · lookup after the queue drains = all hits)
  for (int mode : {1, 2}) {
    EngramSsdOpts o; o.mode = mode; o.cache_bytes = cache_bytes_for(mode, all); o.threads = 16;
    EngramSsd ssd(o);
    T.attach(ssd, mode);
    ssd.alloc_cache();
    char what[64];
    snprintf(what, sizeof what, "prefetch-exact mode %d", mode);
    EXPECT(run_steps(ssd, eh, T, rng, quick ? 10 : 15, true, 1, true, what) == 0, "%s: bytes differ", what);
  }
  printf("engram ssd CPU: gather == engram_gather in %d variants (mode 1/2 · cache 0..all · threads %s · prefetch on/off%s) · prefetch reads exactly the needed rows\n", nv, quick ? "1/3" : "1/3/64", quick ? " · quick matrix" : "");
}

// ---- 3. concurrency ----
void t_concurrent(const EngramHash& eh, const Tables& T) {
  for (int mode : {1, 2}) {
    EngramSsdOpts o; o.mode = mode; o.cache_bytes = cache_bytes_for(mode, 300); o.threads = 8;
    EngramSsd ssd(o);
    T.attach(ssd, mode);
    ssd.alloc_cache();
    std::atomic<int> bad{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t)
      th.emplace_back([&, t] {
        std::mt19937_64 r(1000 + t);
        char what[48];
        snprintf(what, sizeof what, "concurrent seq %d", t);
        bad += run_steps(ssd, eh, T, r, 20, true, t % 2 ? 4 : 1, false, what);
      });
    for (auto& x : th) x.join();
    EXPECT(bad.load() == 0, "concurrent mode %d: %d gather(s) differ", mode, bad.load());
  }
  printf("engram ssd CPU: 4 concurrent sequences on a 300-row cache == engram_gather\n");
}

// ---- 4. failure absorption ----
void t_faults(const EngramHash& eh, const Tables& T, std::mt19937_64& rng, bool direct_dir) {
  auto make_h = [&](int M, std::vector<int64_t>& h) {
    std::vector<int64_t> hist;
    std::vector<int32_t> ids(M); std::vector<int8_t> img(M, 0);
    for (auto& x : ids) x = (int32_t)(rng() % VOCAB);
    eh.compute(ids.data(), img.data(), M, hist, h);
  };
  for (size_t cache_rows : {(size_t)0, (size_t)2000}) {
    for (int mode : {1, 2}) {
      EngramSsdOpts o; o.mode = mode; o.cache_bytes = cache_bytes_for(mode, cache_rows); o.threads = 4; o.retries = 3;
      EngramSsd ssd(o);
      T.attach(ssd, mode);
      ssd.alloc_cache();
      // (a) transient EIO: only the first attempt of rows with row %5 == 0 fails → retry gives the same bytes
      ssd.fault = [](int, int64_t row, int attempt) { return (row % 5 == 0 && attempt == 0) ? EIO : 0; };
      std::vector<int64_t> h;
      make_h(40, h);
      for (int hidx = 0; hidx < NL; ++hidx) {
        std::vector<uint8_t> rv, rs, v((size_t)40 * COLS * HD), s((size_t)40 * COLS * SB);
        ref_gather(T, h, 40, hidx, rv, rs);
        ssd.gather(h.data(), 40, NL, hidx, COLS, HD, (int64_t)(T.vals[hidx].size() / HD), v.data(), s.data(), 1);
        EXPECT(v == rv && s == rs, "transient EIO (cache %zu mode %d hidx %d): bytes differ", cache_rows, mode, hidx);
      }
      EngramSsd::Stats st = ssd.stats(true);
      EXPECT(st.retries > 0 && st.failed == 0, "transient EIO: retries %llu failed %llu", (unsigned long long)st.retries, (unsigned long long)st.failed);
      // (b) persistent EIO: row X of table 0 (a row in this lookup) — only that lookup throws, other lookups on the same store (without X) are fine, after clearing the same lookup == reference
      std::vector<int64_t> h2;
      make_h(6, h2);
      const int64_t X = h2[(size_t)(3 * NL + 0) * COLS + 5];
      std::vector<int64_t> h3;
      for (;;) {  // another lookup that does not contain X
        make_h(6, h3);
        bool has = false;
        for (int m = 0; m < 6; ++m) for (int c = 0; c < COLS; ++c) has |= h3[(size_t)(m * NL + 0) * COLS + c] == X;
        if (!has) break;
      }
      std::atomic<int> attempts{0};
      ssd.fault = [X, &attempts](int kind, int64_t row, int) { if (kind == 0 && row == X) { attempts++; return EIO; } return 0; };
      std::vector<uint8_t> v((size_t)6 * COLS * HD), s((size_t)6 * COLS * SB), rv, rs;
      bool threw = false;
      std::string msg;
      try { ssd.gather(h2.data(), 6, NL, 0, COLS, HD, (int64_t)(T.vals[0].size() / HD), v.data(), s.data(), 1); }
      catch (const std::runtime_error& e) { threw = true; msg = e.what(); }
      EXPECT(threw && msg.find("engram ssd read failed") != std::string::npos, "persistent EIO (cache %zu mode %d): threw %d msg '%s'", cache_rows, mode, threw,
             msg.c_str());
      EXPECT(attempts.load() >= 4, "persistent EIO: %d attempts (want ≥ 1 + retries 3)", attempts.load());
      ref_gather(T, h3, 6, 0, rv, rs);
      ssd.gather(h3.data(), 6, NL, 0, COLS, HD, (int64_t)(T.vals[0].size() / HD), v.data(), s.data(), 1);
      EXPECT(v == rv && s == rs, "persistent EIO: an unrelated gather on the same store differs");
      ssd.fault = nullptr;
      ref_gather(T, h2, 6, 0, rv, rs);
      ssd.gather(h2.data(), 6, NL, 0, COLS, HD, (int64_t)(T.vals[0].size() / HD), v.data(), s.data(), 1);
      EXPECT(v == rv && s == rs, "persistent EIO: the same gather after the fault cleared differs (failed row cached?)");
      // (c) persistent failure during prefetch → not left in the cache, the lookup reads again
      if (cache_rows) {
        std::vector<int64_t> h4;
        make_h(5, h4);
        const int64_t Y = h4[(size_t)(2 * NL + 1) * COLS + 3];
        ssd.fault = [Y](int kind, int64_t row, int) { return (kind == 0 && row == Y) ? EIO : 0; };
        ssd.prefetch_hashes(h4.data(), 5, NL, COLS, 1);
        ssd.quiesce();
        ssd.fault = nullptr;
        ref_gather(T, h4, 5, 1, rv, rs);
        std::vector<uint8_t> v4((size_t)5 * COLS * HD), s4((size_t)5 * COLS * SB);
        ssd.gather(h4.data(), 5, NL, 1, COLS, HD, (int64_t)(T.vals[1].size() / HD), v4.data(), s4.data(), 1);
        EXPECT(v4 == rv && s4 == rs, "prefetch failure: later gather differs");
      }
      // (d) one EINVAL (imitating an O_DIRECT 512-alignment rejection) → absorbed with 4 KiB alignment
      (void)ssd.stats(true);  // clear the failure counts of (b)(c)
      std::atomic<int> once{0};
      ssd.fault = [&once](int, int64_t, int) { return once.fetch_add(1) == 0 ? EINVAL : 0; };
      std::vector<int64_t> h5;
      make_h(7, h5);
      ref_gather(T, h5, 7, 0, rv, rs);
      std::vector<uint8_t> v5((size_t)7 * COLS * HD), s5((size_t)7 * COLS * SB);
      ssd.gather(h5.data(), 7, NL, 0, COLS, HD, (int64_t)(T.vals[0].size() / HD), v5.data(), s5.data(), 1);
      EXPECT(v5 == rv && s5 == rs, "EINVAL absorb: bytes differ");
      st = ssd.stats(true);
      if (direct_dir) EXPECT(st.align_switch == 1 && st.failed == 0, "EINVAL absorb: align switch %llu failed %llu", (unsigned long long)st.align_switch,
                             (unsigned long long)st.failed);
      ssd.fault = nullptr;
      // (e) out-of-range row = the same exception as the RAM path
      std::vector<int64_t> hb(h5);
      hb[(size_t)(1 * NL + 0) * COLS + 2] = (int64_t)(T.vals[0].size() / HD);
      // The RAM-side engram_gather range check is a HIVE_CHECK (common.h — fprintf + abort), so it **terminates the process**
      //   rather than throwing — calling ref_gather inside try here would abort the test process ("hive check failed at
      //   prefill_helpers_gen.h:43: engram row range"). The hash takes the remainder by the table row count, so this is a
      //   defensive check that is never reached in practice — only check that the SSD side throws an exception with the same
      //   wording (only that request fails · the engine survives).
      std::string b;
      try { ssd.gather(hb.data(), 7, NL, 0, COLS, HD, (int64_t)(T.vals[0].size() / HD), v5.data(), s5.data(), 1); } catch (const std::exception& e) { b = e.what(); }
      EXPECT(b.find("engram row range") != std::string::npos, "range: ssd '%s'", b.c_str());
    }
  }
  printf("engram ssd CPU: faults absorbed (transient EIO retried · EINVAL → 4 KiB alignment · persistent EIO fails only that gather, not cached · range check)\n");
}

// ---- 5. real ExpertStore + safetensors ----
struct FakeTensor { std::string name, dtype; std::vector<int64_t> shape; std::vector<uint8_t> bytes; };
void write_shard(const std::string& path, const std::vector<FakeTensor>& ts, int pad) {
  nlohmann::json h = nlohmann::json::object();
  size_t off = 0;
  for (const auto& t : ts) { h[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {off, off + t.bytes.size()}}}; off += t.bytes.size(); }
  h["__metadata__"] = {{"pad", std::string((size_t)pad, 'x')}};
  const std::string hs = h.dump();
  std::ofstream f(path, std::ios::binary);
  const uint64_t n = hs.size();
  f.write((const char*)&n, 8);
  f.write(hs.data(), (std::streamsize)n);
  for (const auto& t : ts) f.write((const char*)t.bytes.data(), (std::streamsize)t.bytes.size());
}

void t_store(const std::string& dir, const EngramHash& eh, const Tables& T, std::mt19937_64& rng) {
  constexpr int DIM = 256, INTER = 256, E = 6, L = 3;
  const std::string cd = dir + "/ckpt";
  fs::create_directories(cd);
  nlohmann::json wm = nlohmann::json::object();
  std::map<std::string, std::vector<FakeTensor>> shards;
  auto put = [&](const std::string& sh, FakeTensor t) { wm[t.name] = sh; shards[sh].push_back(std::move(t)); };
  const int layers[NL] = {1, 2};
  for (int hi = 0; hi < NL; ++hi) {
    const std::string p = "layers." + std::to_string(layers[hi]) + ".engram.";
    const int64_t rows = (int64_t)(T.vals[hi].size() / HD);
    const std::string sh = "e" + std::to_string(hi) + ".safetensors", sh2 = hi == 1 ? "e1s.safetensors" : sh;
    std::vector<uint8_t> q = rand_bytes(rng, 1 + rng() % 5000);  // odd-sized dense tensor first (so the value table start is misaligned)
    put(sh, {p + "q_weight", "U8", {(int64_t)q.size()}, q});
    put(sh, {p + "embed.weight", "F8_E4M3", {rows, HD}, T.vals[hi]});
    put(sh2, {p + "embed.scale", "F8_E8M0", {rows, SB}, T.scales[hi]});
  }
  int pad = 3;
  for (auto& [name, ts] : shards) write_shard(cd + "/" + name, ts, pad += 41);
  std::ofstream(cd + "/model.safetensors.index.json") << nlohmann::json{{"weight_map", wm}}.dump();
  std::ofstream(cd + "/config.json") << nlohmann::json{{"text_config", {{"hidden_size", DIM}, {"moe_intermediate_size", INTER}, {"n_routed_experts", E},
      {"num_hidden_layers", L}, {"engram_layer_ids", {1, 2}}}}}.dump();
  Config cfg;
  cfg.dim = DIM; cfg.moe_inter = INTER; cfg.n_routed = E; cfg.dspark_experts = 4; cfg.swiglu_limit = 10.f;
  unsetenv("HIVE_ENGRAM_SSD");
  Checkpoint ckA(cd);
  ExpertStore A(cfg, L, 0, 1, 0);
  for (int hi = 0; hi < NL; ++hi) A.load_engram(ckA, layers[hi], hi);
  EXPECT(A.engram_ssd() == nullptr, "RAM store has an SSD reader");
  EXPECT(HostPrefault::predict(cd, -1, true).size() == 6, "predict (off): %zu regions (want 2 arenas + 2 × 2)", HostPrefault::predict(cd, -1, true).size());
  std::vector<std::pair<int, int>> tabs = {{1, 0}, {2, 1}};
  for (int mode : {1, 2})
    for (int bulk : {0, 1}) {
      setenv("HIVE_ENGRAM_SSD", mode == 2 ? "2" : "1", 1);
      setenv("HIVE_ENGRAM_SSD_CACHE_MB", "1", 1);
      setenv("HIVE_ENGRAM_SSD_THREADS", "5", 1);
      const size_t want_regions = mode == 1 ? 4 : 2;
      EXPECT(HostPrefault::predict(cd, -1, true).size() == want_regions, "predict (mode %d): %zu regions (want %zu)", mode, HostPrefault::predict(cd, -1, true).size(),
             want_regions);
      Checkpoint ck(cd);
      ExpertStore B(cfg, L, 0, 1, 0);
      if (bulk) {
        BulkLoadOpts o; o.threads = 3; o.chunk = 8192;
        const bool ok = B.load_bulk(ck, 0, tabs, o, false);
        EXPECT(ok, "store mode %d: load_bulk failed", mode);
      } else {
        for (int hi = 0; hi < NL; ++hi) B.load_engram(ck, layers[hi], hi);
      }
      B.engram_ssd_finish(ck);
      EngramSsd* ssd = B.engram_ssd();
      EXPECT(ssd != nullptr, "store mode %d bulk %d: no SSD reader", mode, bulk);
      if (!ssd) continue;
      EXPECT(ssd->opts().threads == 5 && ssd->cache_rows() > 0, "store: options not taken from env (threads %d rows %zu)", ssd->opts().threads, ssd->cache_rows());
      size_t scale_ram = 0;
      for (int hi = 0; hi < NL; ++hi) {
        const EngramTable* t = B.engram(hi);
        EXPECT(t && t->ssd && t->vals.bytes == 0 && t->vals.base == nullptr, "store mode %d: values table in RAM", mode);
        // mode 1 scale RAM = the same arena as the RAM store (PinnedArena allocates in 2 MiB units — compare against that allocation size, not the tensor bytes)
        EXPECT(t && (mode == 2 ? t->scales.bytes == 0 : (t->scales.bytes == A.engram(hi)->scales.bytes && t->scales.bytes >= T.scales[hi].size())),
               "store mode %d: scale RAM %zu (RAM store %zu · tensor %zu)", mode, t ? t->scales.bytes : 0, A.engram(hi)->scales.bytes, T.scales[hi].size());
        if (t) scale_ram += t->scales.bytes;
        EXPECT(t && t->rows == A.engram(hi)->rows && t->hd == A.engram(hi)->hd && t->layer == A.engram(hi)->layer, "store: table meta differs");
      }
      EXPECT(B.engram_host_bytes() == scale_ram + ssd->cache_host_bytes() && B.engram_host_bytes() <= scale_ram + (1u << 20),
             "store mode %d: engram_host_bytes %zu (scales %zu + cache %zu)", mode, B.engram_host_bytes(), scale_ram, ssd->cache_host_bytes());
      int bad = 0;
      std::vector<int64_t> hist;
      for (int st = 0; st < 8; ++st) {
        const int M = st == 3 ? 900 : 1 + st;
        std::vector<int32_t> ids(M); std::vector<int8_t> img(M, 0);
        for (auto& x : ids) x = (int32_t)(rng() % VOCAB);
        std::vector<int64_t> h;
        eh.compute(ids.data(), img.data(), M, hist, h);
        for (int hi = 0; hi < NL; ++hi) {
          const EngramTable* a = A.engram(hi);
          std::vector<uint8_t> rv((size_t)M * COLS * HD), rs((size_t)M * COLS * SB), v(rv.size()), s(rs.size());
          engram_gather(h.data(), M, NL, hi, COLS, HD, a->rows, a->vals.base, a->scales.base, rv.data(), rs.data(), 1);  // tables of the RAM store
          ssd->gather(h.data(), M, NL, hi, COLS, HD, B.engram(hi)->rows, v.data(), s.data(), 4);
          bad += !(v == rv && s == rs);
        }
      }
      EXPECT(bad == 0, "store mode %d bulk %d: %d gather(s) differ from the RAM store", mode, bulk, bad);
    }
  unsetenv("HIVE_ENGRAM_SSD"); unsetenv("HIVE_ENGRAM_SSD_CACHE_MB"); unsetenv("HIVE_ENGRAM_SSD_THREADS");
  printf("engram ssd CPU: real ExpertStore (load_engram · load_bulk) SSD tables == RAM tables · values 0 B in RAM · predict skips SSD tables\n");
}

void t_options() {
  struct C { const char* name; const char* v; double want; };
  auto get = [](const char* name) -> double {
    const std::string n = name;
    if (n == "HIVE_ENGRAM_SSD") return engram_ssd_mode();
    if (n == "HIVE_ENGRAM_SSD_CACHE_MB") return (double)(engram_ssd_cache_bytes() >> 20);
    if (n == "HIVE_ENGRAM_SSD_THREADS") return engram_ssd_threads();
    if (n == "HIVE_ENGRAM_SSD_NO_PREFETCH") return engram_ssd_no_prefetch();
    return engram_digest_on();
  };
  const C cs[] = {{"HIVE_ENGRAM_SSD", nullptr, 0}, {"HIVE_ENGRAM_SSD", "", 0}, {"HIVE_ENGRAM_SSD", "0", 0}, {"HIVE_ENGRAM_SSD", "1", 1}, {"HIVE_ENGRAM_SSD", "on", 1},
                  {"HIVE_ENGRAM_SSD", "2", 2}, {"HIVE_ENGRAM_SSD", "all", 2}, {"HIVE_ENGRAM_SSD_CACHE_MB", nullptr, 2048}, {"HIVE_ENGRAM_SSD_CACHE_MB", "", 2048},
                  {"HIVE_ENGRAM_SSD_CACHE_MB", "x", 2048}, {"HIVE_ENGRAM_SSD_CACHE_MB", "0", 0}, {"HIVE_ENGRAM_SSD_CACHE_MB", "512", 512},
                  {"HIVE_ENGRAM_SSD_CACHE_MB", "-3", 2048}, {"HIVE_ENGRAM_SSD_THREADS", nullptr, 64}, {"HIVE_ENGRAM_SSD_THREADS", "0", 64},
                  {"HIVE_ENGRAM_SSD_THREADS", "128", 128}, {"HIVE_ENGRAM_SSD_THREADS", "999", 256}, {"HIVE_ENGRAM_SSD_NO_PREFETCH", nullptr, 0},
                  {"HIVE_ENGRAM_SSD_NO_PREFETCH", "0", 0}, {"HIVE_ENGRAM_SSD_NO_PREFETCH", "1", 1}, {"HIVE_ENGRAM_DIGEST", nullptr, 0}, {"HIVE_ENGRAM_DIGEST", "1", 1}};
  for (const C& c : cs) {
    if (c.v) setenv(c.name, c.v, 1); else unsetenv(c.name);
    EXPECT(get(c.name) == c.want, "%s=%s → %g (want %g)", c.name, c.v ? c.v : "(unset)", get(c.name), c.want);
    unsetenv(c.name);
  }
  printf("engram ssd CPU: switch parsing (unset/\"\"/0 = off · values as given)\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir(argc < 2 ? "/tmp" : argv[1]);
  const bool direct_dir = getenv("HIVE_TEST_EXPECT_DIRECT") && strcmp(getenv("HIVE_TEST_EXPECT_DIRECT"), "1") == 0;
  std::mt19937_64 rng(20261001);
  FakeHash fh;
  fh.build(dir, rng);
  EngramHash eh;
  eh.load(fh.json, fh.tmap);
  Tables T;
  T.build(dir, fh, rng);
  {
    EngramSsdOpts o; o.mode = 1; o.cache_bytes = 0; o.threads = 1;
    EngramSsd probe(o);
    T.attach(probe, 1);
    EXPECT(!direct_dir || probe.direct(0), "work dir supports O_DIRECT but the reader opened buffered");
  }
  t_options();
  t_peek(eh, rng);
  t_gather(eh, T, rng);
  t_concurrent(eh, T);
  t_faults(eh, T, rng, direct_dir);
  t_store(dir, eh, T, rng);
  if (fails) { fprintf(stderr, "test_engram_ssd_cpu: %d failure(s)\n", fails); return 1; }
  printf("test_engram_ssd_cpu passed\n");
  return 0;
}
