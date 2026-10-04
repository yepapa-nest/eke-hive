// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU test (no GPU; tools/test_decode_ubatch_cpu.py builds and runs it on fake CUDA — TSAN variant included).
//   (1) dov::group_routes_sparse (grouping for HIVE_DECODE_HOST_FAST) == the original E-slot counting sort of runtime.cpp moe_decode_experts (ported verbatim into dense_ref below):
//       cnt[e]·cnt[e+1] for every used expert · all of rows_by_e · the list of used experts (= e with n(e) > 0, ascending). Random routing (with and without duplicates within a row).
//   (2) runs dov::run_ubatch (the HIVE_DECODE_UBATCH layer pipeline — the same header as production) on a fake GPU (fake CUDA streams — real threads with FAKE_CUDA_ASYNC=1)
//       + one real CPU pool thread. Data model (one uint64 state per half — exact integer arithmetic):
//         prep(h,l)   host writes an (h,l) tag into the "mapped row table" tab[h]          → front(h,l) checks the tag **when it executes** on the GPU thread
//         front(h,l)  GPU: acc = mix(acc, l) · route[h] (mapped) = hash(acc, l)  → the host reads route after wait_front
//         launch      host: read route → (h,l,route) into the expert table tbl[h] · CPU input cin[h] · GPU expert launch (checks the tbl tag at execution · acc += g(route))
//         CPU pool    cout[h] = c(cin[h]) (one batch at a time — starting while busy = FAIL: same contract as the HIVE_CHECK in production ExpertStore::start_jobs)
//         accum       GPU: acc += cout[h]
//       Reference = sequential computation per half (acc = mix → g → c order). PASS if the final state matches the reference, all tags match, and each operation ran exactly once per (half, layer).
//       Random delays in GPU operations and CPU jobs shake up the interleavings. Ordering violations (mutants) show up as tag mismatches, pool collisions or final-state mismatches — checked by the .py.
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "cuda_runtime.h"
#include "hive/decode_overlap.h"

using namespace hive;
using Micros = std::chrono::microseconds;

static std::atomic<int> g_fail{0};
#define EXPECT(cond, ...) do { if (!(cond)) { if (g_fail.fetch_add(1) < 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

// ---- (1) original formula (runtime.cpp moe_decode_experts, before the sparse grouping) ----
static void dense_ref(const int32_t* ids, int R, int E, int* cnt, int* rows_by_e) {
  for (int e = 0; e <= E; ++e) cnt[e] = 0;
  for (int i = 0; i < R; ++i) ++cnt[ids[i] + 1];
  for (int e = 0; e < E; ++e) cnt[e + 1] += cnt[e];
  int fill[512];
  std::memcpy(fill, cnt, sizeof(int) * E);
  for (int i = 0; i < R; ++i) rows_by_e[fill[ids[i]]++] = i;
}
static void test_grouping(int iters) {
  std::mt19937 rng(20260930);
  std::vector<int32_t> ids;
  std::vector<int> cref(513), rref, cnew(513), rnew, used, tmp;
  int fill[512];
  long checked = 0;
  for (int it = 0; it < iters; ++it) {
    const int E = (it % 3 == 0) ? 128 : 384;
    const int k = (it % 4 == 0) ? 3 : 6;
    const int M = 1 + (int)(rng() % (it % 5 == 0 ? 64 : 8));
    const bool distinct = it % 2 == 0;  // router top-k (no duplicates within a row) · general input
    const int R = M * k;
    ids.assign(R, 0);
    std::uniform_int_distribution<int> pick(0, (it % 7 == 0) ? std::min(E - 1, 11) : E - 1);  // narrow pool = heavy overlap between rows
    for (int m = 0; m < M; ++m)
      for (int j = 0; j < k; ++j) {
        int e;
        bool dup;
        do { e = pick(rng); dup = false; if (distinct) for (int q = 0; q < j; ++q) dup |= ids[m * k + q] == e; } while (dup && distinct && k <= 12);
        ids[m * k + j] = e;
      }
    rref.assign(R, -1); rnew.assign(R, -2); used.assign(R, -1); tmp.assign(R, 0);
    std::fill(cnew.begin(), cnew.end(), -777);  // unused slots must not be touched — nor are they read (only used slots are compared below)
    dense_ref(ids.data(), R, E, cref.data(), rref.data());
    const int nu = dov::group_routes_sparse(ids.data(), R, cnew.data(), rnew.data(), used.data(), tmp.data(), fill);
    int nu_ref = 0;
    for (int e = 0; e < E; ++e) {
      if (cref[e + 1] - cref[e] == 0) continue;
      EXPECT(nu_ref < nu && used[nu_ref] == e, "it %d: used[%d] = %d, want %d", it, nu_ref, nu_ref < nu ? used[nu_ref] : -1, e);
      EXPECT(cnew[e] == cref[e] && cnew[e + 1] == cref[e + 1], "it %d e %d: cnt %d/%d want %d/%d", it, e, cnew[e], cnew[e + 1], cref[e], cref[e + 1]);
      ++nu_ref;
    }
    EXPECT(nu == nu_ref, "it %d: nu %d want %d", it, nu, nu_ref);
    for (int i = 0; i < R; ++i) EXPECT(rnew[i] == rref[i], "it %d: rows_by_e[%d] = %d want %d", it, i, rnew[i], rref[i]);
    ++checked;
  }
  printf("ubatch CPU: group_routes_sparse == dense counting sort on %ld random routings\n", checked);
}

// ---- (2) pipeline ----
static inline uint64_t mix(uint64_t a, int l) { a ^= (uint64_t)(l + 1) * 0x9E3779B97F4A7C15ull; a *= 0xBF58476D1CE4E5B9ull; return a ^ (a >> 31); }
static inline uint64_t hsh(uint64_t a, int l) { a += (uint64_t)l * 0x94D049BB133111EBull; a ^= a >> 29; a *= 0xD6E8FEB86659FD93ull; return a ^ (a >> 32); }
static inline uint64_t gpu_part(uint64_t r) { return (r * 3) ^ 0x5555; }
static inline uint64_t cpu_part(uint64_t r) { return (r >> 7) + 17; }
static inline uint64_t tag(int h, int l) { return ((uint64_t)(h + 1) << 32) | (uint64_t)(l + 1); }

struct Pool {  // CPU expert pool emulation — one batch at a time (duplicate start = FAIL)
  std::mutex mu; std::condition_variable cv;
  bool has = false, busy = false, stop = false;
  std::function<void()> job;
  std::thread th;
  Pool() { th = std::thread([this] { run(); }); }
  ~Pool() { { std::lock_guard<std::mutex> l(mu); stop = true; } cv.notify_all(); th.join(); }
  void run() {
    for (;;) {
      std::function<void()> f;
      { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return stop || has; }); if (!has) return; f = job; has = false; }
      f();
      { std::lock_guard<std::mutex> l(mu); busy = false; } cv.notify_all();
    }
  }
  void start(std::function<void()> f) {
    std::lock_guard<std::mutex> l(mu);
    EXPECT(!busy, "pool: start while a batch is active (ExpertStore::start_jobs contract)");
    busy = true; has = true; job = std::move(f);
    cv.notify_all();
  }
  void wait() { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return !busy; }); }
};

struct Sim {
  cudaStream_t st = nullptr;
  cudaEvent_t done[2] = {nullptr, nullptr};
  Pool pool;
  std::mt19937 rng;
  int nL = 0, delay_us = 0;
  // per-half state: dev = "GPU memory" (touched only by the stream thread) · map_route = mapped memory (written by the GPU, read by the host after the event)
  uint64_t acc[2] = {11, 23}, map_route[2] = {0, 0};
  uint64_t tab[2] = {0, 0};                 // mapped row table (written by the host in prep — read by front at execution)
  uint64_t tbl_tag[2] = {0, 0}, tbl_route[2] = {0, 0};  // expert table (host pinned source — read by the expert op at execution)
  uint64_t cin[2] = {0, 0}, cout[2] = {0, 0};           // CPU input (host) · output (mapped — read by accum)
  bool has_cpu[2] = {false, false}, started[2] = {false, false};
  std::vector<int> cnt_ops;                 // [op][h][l]
  std::vector<uint8_t> cpu_plan;            // [h·nL + l]: does (half, layer) have a CPU share (random)
  enum { P, F, WF, LA, SC, WC, AC, NOPS };
  void count(int op, int h, int l) { ++cnt_ops[((size_t)op * 2 + h) * nL + l]; }
  void jitter() { if (delay_us > 0) std::this_thread::sleep_for(Micros(rng() % (unsigned)delay_us)); }
  // ---- Ops (same role as production runtime.cpp decode_layers_ubatch::Ops) ----
  void prep(int h, int l) { count(P, h, l); tab[h] = tag(h, l); }
  void front(int h, int l) {
    count(F, h, l);
    const int d = (int)(rng() % (unsigned)(delay_us + 1));
    fakecuda::enqueue(st, [this, h, l, d] {
      if (d) std::this_thread::sleep_for(std::chrono::microseconds(d));
      EXPECT(tab[h] == tag(h, l), "front(%d,%d) read row table of (%llx) — prep overwrote it early", h, l, (unsigned long long)tab[h]);
      acc[h] = mix(acc[h], l);
      map_route[h] = hsh(acc[h], l);
    });
    cudaEventRecord(done[h], st);
  }
  void wait_front(int h, int l) { count(WF, h, l); cudaEventSynchronize(done[h]); }
  void launch(int h, int l, bool go) {
    count(LA, h, l);
    const uint64_t r = map_route[h];  // host read of mapped memory (after the event)
    tbl_tag[h] = tag(h, l); tbl_route[h] = r;
    has_cpu[h] = cpu_plan[(size_t)h * nL + l] != 0;
    started[h] = false;
    if (has_cpu[h]) cin[h] = r;
    const int d = (int)(rng() % (unsigned)(delay_us + 1));
    fakecuda::enqueue(st, [this, h, l, d] {  // GPU expert op (the table is read at execution)
      if (d) std::this_thread::sleep_for(std::chrono::microseconds(d));
      EXPECT(tbl_tag[h] == tag(h, l), "experts(%d,%d) read expert table of (%llx) — overwritten before the copy ran", h, l, (unsigned long long)tbl_tag[h]);
      acc[h] += gpu_part(tbl_route[h]);
    });
    if (go) start_cpu(h, l);
  }
  bool cpu_idle(int h, int) const { return !has_cpu[h]; }
  void start_cpu(int h, int l) {
    if (!has_cpu[h] || started[h]) return;
    count(SC, h, l);
    started[h] = true;
    const int d = (int)(rng() % (unsigned)(delay_us + 1));
    pool.start([this, h, d] { if (d) std::this_thread::sleep_for(std::chrono::microseconds(d)); cout[h] = cpu_part(cin[h]); });
  }
  void wait_cpu(int h, int l) {
    count(WC, h, l);
    if (!has_cpu[h]) return;
    EXPECT(started[h], "wait_cpu(%d,%d) before start", h, l);
    pool.wait();
  }
  void accum(int h, int l) {
    count(AC, h, l);
    const bool c = has_cpu[h];
    fakecuda::enqueue(st, [this, h, c] { if (c) acc[h] += cout[h]; });
  }
};

static bool run_pipeline(int nL, int reps, int delay_us, unsigned seed) {
  for (int rep = 0; rep < reps; ++rep) {
    Sim s;
    s.nL = nL; s.delay_us = delay_us; s.rng.seed(seed + 977u * (unsigned)rep);
    s.cnt_ops.assign((size_t)Sim::NOPS * 2 * nL, 0);
    s.cpu_plan.resize((size_t)2 * nL);
    for (auto& v : s.cpu_plan) v = (s.rng() % 4) != 0;  // CPU share with probability 3/4
    cudaStreamCreateWithFlags(&s.st, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&s.done[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&s.done[1], cudaEventDisableTiming);
    dov::run_ubatch(nL, s);
    cudaStreamSynchronize(s.st);
    // reference: sequential per half
    uint64_t ref[2] = {11, 23};
    for (int h = 0; h < 2; ++h)
      for (int l = 0; l < nL; ++l) {
        ref[h] = mix(ref[h], l);
        const uint64_t r = hsh(ref[h], l);
        ref[h] += gpu_part(r);
        if (s.cpu_plan[(size_t)h * nL + l]) ref[h] += cpu_part(r);
      }
    for (int h = 0; h < 2; ++h) EXPECT(s.acc[h] == ref[h], "rep %d half %d: final state %llx != sequential %llx", rep, h, (unsigned long long)s.acc[h], (unsigned long long)ref[h]);
    for (int op = 0; op < Sim::NOPS; ++op)
      for (int h = 0; h < 2; ++h)
        for (int l = 0; l < nL; ++l) {
          const int n = s.cnt_ops[((size_t)op * 2 + h) * nL + l];
          const int want = op == Sim::SC ? (s.cpu_plan[(size_t)h * nL + l] ? 1 : 0) : 1;
          EXPECT(n == want, "rep %d: op %d (h %d, l %d) ran %d times, want %d", rep, op, h, l, n, want);
        }
    cudaEventDestroy(s.done[0]); cudaEventDestroy(s.done[1]);
    cudaStreamDestroy(s.st);
    if (g_fail.load()) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 4000;
  const int reps = argc > 2 ? atoi(argv[2]) : 60;
  const int delay = argc > 3 ? atoi(argv[3]) : 60;
  test_grouping(iters);
  bool ok = true;
  for (int nL : {1, 2, 3, 43}) ok = run_pipeline(nL, nL == 43 ? reps : std::max(4, reps / 4), delay, 1000u + (unsigned)nL) && ok;
  printf("ubatch CPU: pipeline (nL 1/2/3/43, %d reps, delay ≤ %d µs, async %d): %s\n", reps, delay, fakecuda::async_mode() ? 1 : 0, ok && !g_fail.load() ? "state == sequential, tags OK, ops once" : "FAIL");
  if (g_fail.load()) { printf("FAIL: %d violations\n", g_fail.load()); return 1; }
  printf("ubatch CPU suite OK\n");
  return 0;
}
