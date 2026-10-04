// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU-only stress of the REAL ExpertStore CPU worker pool (engine/src/expert_store.cpp) with
// fake CUDA/NUMA (tools/cpu_fake). Built and run by tools/test_pool_cpu.py; no GPU.
//
// Checks
//  1. every batch's output == cpu::expert_forward on the unsplit expert (bitwise), for random
//     job counts, R rows per job, layers/experts, over many iterations, with work stealing
//     between the two (fake) NUMA pools;
//  2. start_jobs/wait_jobs/run_jobs alternate, empty batches, wait without a batch;
//  3. allocation failure injected at every allocation inside start_jobs: the exception leaves
//     no published work, wait_jobs returns, and the next batch is still exact;
//  4. pool shutdown (ExpertStore destructor) joins, also with HIVE_CPU_SPIN_US spinning;
//  5. jobs_done_ms() after wait_jobs is >= the batch start (wait_jobs must not return
//     before the last finisher has published t_done). HIVE_POOL_STRICT_TDONE=1: exit 3 on any stale value.
//  HIVE_POOL_INJECT_RACE=1: TSAN negative control (the caller writes an input while the pool reads it).
//  argv[3] == "cache": staging-reuse D2D (HIVE_CACHE_REUSE_STAGE, own stream, overwrite-after-reuse and
//     batch-event ordering; run with FAKE_CUDA_ASYNC=1), HIVE_CACHE_EVENTS trace, touch_at LRU (B6).
//     Dedicated D2D stream — with the promotion stream HELD, a staging overwrite that follows a staged-key
//     promotion still completes (it waits only for the D2D), a held victim fence blocks the D2D (and so the overwrite),
//     and commit_all covers the D2D (promoted slot bytes exact).
// Under HIVE_TEST_SANITIZER=thread every item/steal/publication is race-checked.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hive/expert_cpu.h"
#include "hive/expert_store.h"
#include "hive/safetensors.h"
#include "hive/clock.h"

namespace hive { void fake_ckpt_put(const std::string&, const uint8_t*, size_t, const std::string& = "U8"); }
using namespace hive;
using Micros = std::chrono::microseconds;

// Per-thread allocation failure injection (only the submitting thread is armed).
thread_local long g_fail_after = -1;
void* operator new(size_t n) {
  if (g_fail_after >= 0 && g_fail_after-- == 0) throw std::bad_alloc();
  void* p = malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }

static double now_ms() { return hive::mono_ms(); }

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

constexpr int L = 2, E = 8, DIM = 512, INTER = 512, NA = 16;

struct World {
  Config cfg;
  std::vector<std::vector<uint8_t>> blobs;  // owned tensor storage for the fake checkpoint
  std::vector<uint8_t> aq[NA], as[NA];
  std::vector<float> af[NA], saf[NA];
  float rw[NA];
  std::vector<float> ref[L][E][NA];
  World() {
    cfg.dim = DIM; cfg.moe_inter = INTER; cfg.n_routed = E; cfg.dspark_experts = 4; cfg.swiglu_limit = 10.f;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> byte(0, 255);
    auto e4m3 = [&] { uint8_t v; do { v = (uint8_t)byte(rng); } while ((v & 0x7F) == 0x7F || (v & 0x78) > 0x50); return v; };
    const size_t wsz = (size_t)INTER * DIM / 2, ssz = (size_t)INTER * DIM / 32;
    blobs.reserve(L * E * 6);
    for (int l = 0; l < L; ++l)
      for (int e = 0; e < E; ++e)
        for (const char* m : {"w1", "w2", "w3"}) {
          blobs.emplace_back(wsz); for (auto& v : blobs.back()) v = (uint8_t)byte(rng);
          const std::string p = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + "." + m;
          fake_ckpt_put(p + ".weight", blobs.back().data(), wsz);
          blobs.emplace_back(ssz); for (auto& v : blobs.back()) v = (uint8_t)(127 - 10 + byte(rng) % 6);
          fake_ckpt_put(p + ".scale", blobs.back().data(), ssz);
        }
    for (int a = 0; a < NA; ++a) {
      aq[a].resize(DIM); as[a].resize(DIM / 32); af[a].resize(DIM); saf[a].resize(DIM / 32);
      for (auto& v : aq[a]) v = e4m3();
      for (auto& v : as[a]) v = (uint8_t)(127 - 6 + byte(rng) % 4);
      for (int k = 0; k < DIM; ++k) af[a][k] = e4m3_to_f32(aq[a][k]);
      for (int b = 0; b < DIM / 32; ++b) saf[a][b] = e8m0_to_f32(as[a][b]);
      rw[a] = 0.1f + 0.05f * a;
    }
    std::vector<float> tmp(4 * INTER + 2 * DIM + 64);
    for (int l = 0; l < L; ++l)
      for (int e = 0; e < E; ++e) {
        const size_t base = (size_t)(l * E + e) * 6;  // w1 w1s w2 w2s w3 w3s
        cpu::ExpertDesc d{cpu::ExpertFormat::E2M1_B32, DIM, INTER, blobs[base].data(), blobs[base + 1].data(), blobs[base + 2].data(),
                          blobs[base + 3].data(), blobs[base + 4].data(), blobs[base + 5].data()};
        for (int a = 0; a < NA; ++a) { ref[l][e][a].resize(DIM); cpu::expert_forward(d, aq[a].data(), as[a].data(), rw[a], cfg.swiglu_limit, ref[l][e][a].data(), tmp.data()); }
      }
  }
};

struct Batch {
  std::vector<ExpertStore::Job> jobs;
  std::vector<std::vector<float>> outs, scratch;
  std::vector<std::vector<int>> acts;
  void make(World& w, std::mt19937& rng, int nj) {
    jobs.assign(nj, {}); outs.clear(); scratch.clear(); acts.assign(nj, {});
    outs.reserve(nj * ExpertStore::kMaxRows); scratch.reserve(nj);
    for (int j = 0; j < nj; ++j) {
      auto& J = jobs[j];
      J.layer = (int)(rng() % L); J.e = (int)(rng() % E); J.R = 1 + (int)(rng() % ExpertStore::kMaxRows);
      scratch.emplace_back(ExpertStore::job_scratch_floats(INTER) * J.R, NAN);
      J.scratch = scratch.back().data();
      for (int r = 0; r < J.R; ++r) {
        const int a = (int)(rng() % NA); acts[j].push_back(a);
        J.a_f[r] = w.af[a].data(); J.a_s[r] = w.saf[a].data(); J.route_w[r] = w.rw[a];
        outs.emplace_back(DIM, NAN); J.out[r] = outs.back().data();
      }
    }
  }
  int check(World& w) const {
    int bad = 0;
    for (size_t j = 0; j < jobs.size(); ++j)
      for (int r = 0; r < jobs[j].R; ++r)
        if (memcmp(jobs[j].out[r], w.ref[jobs[j].layer][jobs[j].e][acts[j][r]].data(), DIM * sizeof(float)) != 0) ++bad;
    return bad;
  }
};

static std::unique_ptr<ExpertStore> make_store(World& w, Checkpoint& ck, int threads) {
  auto s = std::make_unique<ExpertStore>(w.cfg, L, 0, threads, 0);
  for (int l = 0; l < L; ++l) s->load_layer_experts(ck, l, 4);
  return s;
}

// Cache-side checks (no pool work). Expected d2d count comes from CACHE_EXPECT_D2D (python sets it per switch value).
static int cache_main(World& w, Checkpoint& ck) {
  const size_t total = ExpertLayout::make(DIM, INTER).total;
  std::vector<uint8_t> ref1(total), ref2(total);
  {
    auto st = std::make_unique<ExpertStore>(w.cfg, L, 4 * total, 1, 0);
    for (int l = 0; l < L; ++l) st->load_layer_experts(ck, l, 2);
    st->copy_rec_async(ref1.data(), 0, 1, nullptr); st->copy_rec_async(ref2.data(), 0, 2, nullptr);  // null stream = synchronous
    EXPECT(memcmp(ref1.data(), ref2.data(), total) != 0, "fixture experts identical");
    {  // the record from copy_rec_async (12 pieces by default, 8 calls with HIVE_STAGE_COPY2D) = checkpoint tensor bytes [w1][s1][w3][s3][w2][s2] (independent reference — every layer and expert)
      const ExpertLayout lay = ExpertLayout::make(DIM, INTER);
      const size_t wsz = (size_t)INTER * DIM / 2, ssz = (size_t)INTER * DIM / 32;
      std::vector<uint8_t> rec(total);
      int bad = 0;
      for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) {
          std::fill(rec.begin(), rec.end(), 0xA5);
          st->copy_rec_async(rec.data(), l, e, nullptr);
          const size_t b = (size_t)(l * E + e) * 6;  // w1 w1s w2 w2s w3 w3s
          bad += memcmp(rec.data() + lay.w1, w.blobs[b].data(), wsz) != 0;
          bad += memcmp(rec.data() + lay.s1, w.blobs[b + 1].data(), ssz) != 0;
          bad += memcmp(rec.data() + lay.w3, w.blobs[b + 4].data(), wsz) != 0;
          bad += memcmp(rec.data() + lay.s3, w.blobs[b + 5].data(), ssz) != 0;
          bad += memcmp(rec.data() + lay.w2, w.blobs[b + 2].data(), wsz) != 0;
          bad += memcmp(rec.data() + lay.s2, w.blobs[b + 3].data(), ssz) != 0;
        }
      EXPECT(bad == 0, "copy_rec_async record != checkpoint tensors (%d matrices)", bad);
      const char* c2 = getenv("HIVE_STAGE_COPY2D");
      printf("pool CPU: copy_rec_async records == checkpoint tensors (%d layers x %d experts, HIVE_STAGE_COPY2D=%s)\n", L, E, c2 && *c2 ? c2 : "off");
    }
    cudaStream_t side, promo; cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&promo, cudaStreamNonBlocking);
    st->observe(0, 1);
    st->copy_to_staging(0, 0, 1, side);
    EXPECT(st->promote_keys({0 * E + 1}, 1, promo) == 1, "promotion not issued");
    st->copy_to_staging(0, 0, 2, side);  // overwrite of the staging slot must wait for the D2D reuse
    st->commit_all();                    // the batch event must cover the D2D (host reads the slot below)
    const int slot = st->slot_of(0, 1);
    EXPECT(slot >= 0, "promoted expert not resident");
    if (slot >= 0) EXPECT(memcmp(st->dev_rec(slot), ref1.data(), total) == 0, "promoted slot content != expert record");
    cudaStreamSynchronize(side);
    EXPECT(memcmp(st->staging_rec(0), ref2.data(), total) == 0, "staging overwrite lost");
    const long want = getenv("CACHE_EXPECT_D2D") ? atol(getenv("CACHE_EXPECT_D2D")) : 1;
    EXPECT((long)st->cache_stats().d2d_records == want, "d2d_records %llu, expected %ld", (unsigned long long)st->cache_stats().d2d_records, want);
    st->observe(0, 1);
    st->place(0, 3, promo); st->observe(0, 3);
    cudaStreamSynchronize(promo);
    printf("pool CPU: cache staging reuse d2d=%llu, promoted slot exact\n", (unsigned long long)st->cache_stats().d2d_records);
    st.reset();  // flushes HIVE_CACHE_EVENTS before the streams it never owned go away
    cudaStreamDestroy(side); cudaStreamDestroy(promo);
  }
  const long want_d2d = getenv("CACHE_EXPECT_D2D") ? atol(getenv("CACHE_EXPECT_D2D")) : 1;
  if (want_d2d == 1 && fakecuda::async_mode()) {  // REUSE_STAGE dedicated D2D stream
    std::vector<uint8_t> ref4(total);
    auto st = std::make_unique<ExpertStore>(w.cfg, L, 4 * total, 1, 0);
    for (int l = 0; l < L; ++l) st->load_layer_experts(ck, l, 2);
    st->copy_rec_async(ref4.data(), 0, 4, nullptr);
    cudaStream_t side, promo, vst; cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&promo, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&vst, cudaStreamNonBlocking);
    auto sync_within = [](cudaStream_t s, int ms) {  // true if the stream drains within ms (the waiter keeps running otherwise)
      auto f = std::async(std::launch::async, [s] { cudaStreamSynchronize(s); });
      const bool ok = f.wait_for(std::chrono::milliseconds(ms)) == std::future_status::ready;
      return std::make_pair(ok, std::move(f));
    };
    {  // (a) promotion stream held: H2D promotion queued behind it, then a staged-key D2D promotion, then a staging overwrite
      std::promise<void> gate; std::shared_future<void> open = gate.get_future().share();
      fakecuda::enqueue(promo, [open] { open.wait(); });
      st->observe(0, 4); st->observe(0, 1);
      EXPECT(st->promote_keys({0 * E + 4}, 1, promo) == 1, "H2D promotion not issued");
      st->copy_to_staging(0, 0, 1, side);
      EXPECT(st->promote_keys({0 * E + 1}, 1, promo) == 1, "staged promotion not issued");
      st->copy_to_staging(0, 0, 2, side);
      auto [ok, f] = sync_within(side, 5000);
      EXPECT(ok, "staging overwrite waited behind the held promotion stream (D2D not on its own stream)");
      gate.set_value(); f.wait();
      st->commit_all();  // batch events must cover the D2D (joined into the promotion stream)
      const int s1 = st->slot_of(0, 1), s4 = st->slot_of(0, 4);
      EXPECT(s1 >= 0 && s4 >= 0, "promotions not committed");
      if (s1 >= 0) EXPECT(memcmp(st->dev_rec(s1), ref1.data(), total) == 0, "D2D-promoted slot != expert record");
      if (s4 >= 0) EXPECT(memcmp(st->dev_rec(s4), ref4.data(), total) == 0, "H2D-promoted slot != expert record");
      EXPECT(memcmp(st->staging_rec(0), ref2.data(), total) == 0, "staging overwrite lost");
    }
    {  // (b) victim fence held: the D2D (and so the next staging overwrite) must wait for it
      std::promise<void> gate; std::shared_future<void> open = gate.get_future().share();
      cudaEvent_t vf; cudaEventCreateWithFlags(&vf, cudaEventDisableTiming);
      fakecuda::enqueue(vst, [open] { open.wait(); });
      cudaEventRecord(vf, vst);
      st->observe(0, 3);
      st->copy_to_staging(1, 0, 3, side);
      EXPECT(st->promote_keys({0 * E + 3}, 1, promo, 0, false, vf) == 1, "fenced promotion not issued");
      st->copy_to_staging(1, 0, 2, side);
      auto [early, f] = sync_within(side, 300);
      EXPECT(!early, "D2D did not wait for the victim fence");
      gate.set_value(); f.wait();
      st->commit_all();
      const int s3 = st->slot_of(0, 3);
      std::vector<uint8_t> ref3(total); st->copy_rec_async(ref3.data(), 0, 3, nullptr);
      EXPECT(s3 >= 0 && memcmp(st->dev_rec(s3), ref3.data(), total) == 0, "fenced D2D-promoted slot != expert record");
      cudaEventDestroy(vf);
    }
    printf("pool CPU: REUSE_STAGE dedicated D2D stream: overwrite not behind held promotions, victim fence honoured, commit covers D2D\n");
    st.reset();
    cudaStreamDestroy(side); cudaStreamDestroy(promo); cudaStreamDestroy(vst);
  }
  {  // B6: touch_at gives every slot of one batch the same LRU time and never moves a slot back in time
    setenv("HIVE_CACHE_EVENTS", "", 1);
    auto st = std::make_unique<ExpertStore>(w.cfg, L, 4 * total, 1, 0);
    for (int l = 0; l < L; ++l) st->load_layer_experts(ck, l, 2);
    int s[4]; for (int e = 0; e < 4; ++e) s[e] = st->place(0, e, nullptr);
    const uint64_t stamp = st->lru_stamp();
    st->touch_at(s[1], stamp); st->touch_at(s[0], stamp);  // reverse order on purpose: same time either way
    EXPECT(st->place(0, 4, nullptr) == s[2], "LRU victim after touch_at is not the oldest untouched slot");
    st->touch_at(s[3], 1);  // older stamp: must not refresh nor regress
    EXPECT(st->place(0, 5, nullptr) == s[3], "touch_at with an older stamp changed recency");
    const int v = st->place(0, 6, nullptr);
    EXPECT(v == s[0] || v == s[1], "batch-stamped slots evicted out of order (victim %d)", v);
  }
  printf("pool CPU: touch_at LRU batch stamp checks done\n");
  return fails ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc > 3 && !strcmp(argv[3], "cache")) { World w; Checkpoint ck("fake"); return cache_main(w, ck); }
  const int iters = argc > 1 ? atoi(argv[1]) : 400;
  const int threads = argc > 2 ? atoi(argv[2]) : 4;
  World w;
  Checkpoint ck("fake");
  std::mt19937 rng(1234);
  long tdone_bad = 0, batches = 0, rows = 0;
  {
    auto store = make_store(w, ck, threads);
    store->wait_jobs();  // no batch: must return
    std::vector<ExpertStore::Job> none;
    store->start_jobs(none); store->wait_jobs();
    Batch b;
    for (int it = 0; it < iters; ++it) {
      b.make(w, rng, 1 + (int)(rng() % 12));
      // HIVE_DECODE_PREGATE: in owned mode, issue prefetch requests before a batch (batch experts + arbitrary experts, sometimes twice in a row, sometimes a pause before the batch) —
      //   covers the paths where a batch arrives mid-prefetch and cuts it short, where everything is read and then waits, and where requests overlap and are dropped. The bit comparison below decides.
      if (store->owned_mode() && !b.jobs.empty()) {
        int pe[16], np = 0;
        for (const auto& jb : b.jobs) if (np < 12 && jb.layer == b.jobs[0].layer) pe[np++] = jb.e;
        pe[np++] = (int)(rng() % (unsigned)w.cfg.n_routed);
        store->prefetch_experts(b.jobs[0].layer, pe, np);
        if (it % 4 == 1) store->prefetch_experts(b.jobs[0].layer, pe, np);
        if (it % 7 == 2) std::this_thread::sleep_for(Micros(rng() % 300));
      }
      const double t0 = now_ms();
      if (it % 3 == 0) store->run_jobs(b.jobs);
      else {
        store->start_jobs(b.jobs, /*owned=*/store->owned_mode() && it % 4 != 3);  // owned batches, occasionally mixed with legacy pick-up batches (run_jobs is legacy too)
        if (it % 5 == 1) std::this_thread::sleep_for(Micros(rng() % 200));  // caller busy with "GPU launches"
        // TSAN negative control: the caller mutates an activation the pool is reading (a bug in the TEST, not in hive).
        if (getenv("HIVE_POOL_INJECT_RACE")) for (int k = 0; k < DIM; ++k) w.af[b.acts[0][0]][k] += 0.f;
        store->wait_jobs();
      }
      const double td = store->jobs_done_ms();
      if (!(td >= t0)) ++tdone_bad;
      const int bad = b.check(w);
      EXPECT(bad == 0, "iteration %d: %d/%zu rows differ from expert_forward", it, bad, b.outs.size());
      ++batches; rows += (long)b.outs.size();
      if (fails > 5) break;
    }
    if (store->owned_mode()) {  // the requests actually reached the workers (fully read or cut short)
      const auto ps = store->pf_stats();
      printf("pool CPU: pregate owned mode — prefetch requests %ld · experts %ld · %ld KiB read · finished %ld · aborted %ld\n", ps.req, ps.experts, ps.kib, ps.finished, ps.aborted);
      EXPECT(ps.req > 0 && ps.finished + ps.aborted > 0, "prefetch requests never reached a worker");
    }
  }  // destructor joins the pool threads
  printf("pool CPU: %ld batches, %ld expert rows exact vs expert_forward (threads/node %d, spin_us %s)\n", batches, rows, threads,
         getenv("HIVE_CPU_SPIN_US") ? getenv("HIVE_CPU_SPIN_US") : "0");

  // Allocation failure at each allocation inside start_jobs.
  int injected = 0;
  for (long k = 0; k < 64; ++k) {
    auto store = make_store(w, ck, threads);
    Batch small, big, after;
    small.make(w, rng, 1); store->run_jobs(small.jobs);   // pool vectors sized for one job
    big.make(w, rng, 12); after.make(w, rng, 12);
    bool threw = false;
    g_fail_after = k;
    try { store->start_jobs(big.jobs); } catch (const std::bad_alloc&) { threw = true; }
    g_fail_after = -1;
    const double t0 = now_ms();
    store->wait_jobs();  // must not wait for an unpublished partial list
    EXPECT(now_ms() - t0 < 2000, "wait_jobs after injected failure %ld took %.0f ms", k, now_ms() - t0);
    if (!threw) { EXPECT(big.check(w) == 0, "uninjected big batch wrong"); break; }
    ++injected;
    store->run_jobs(after.jobs);
    EXPECT(after.check(w) == 0, "batch after injected failure %ld is wrong", k);
  }
  printf("pool CPU: %d allocation-failure points inside start_jobs recovered (wait returns, next batch exact)\n", injected);
  EXPECT(injected > 0, "no allocation happened inside start_jobs: injection did not exercise anything");
  printf("pool CPU: jobs_done_ms() < batch start after wait_jobs in %ld/%ld batches\n", tdone_bad, batches);
  if (getenv("HIVE_POOL_STRICT_TDONE") && tdone_bad) { fprintf(stderr, "FAIL jobs_done_ms stale after wait_jobs\n"); return 3; }
  return fails ? 1 : 0;
}
