// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Sleep level 3 — CPU test for hive/devmem.h (restoring VMM allocations at the same address) (fake CUDA: VA = host memory · unmap = overwrite with 0xCD). Built and run by tools/test_sleep_cpu.py.
//   on  : HIVE_SLEEP_VMM=1 — DevBuf goes through devmem · small allocations are appended to chunks · 0 mappings after release_all · contents lost (0xCD) · free/alloc while asleep ·
//         after restore_all the contents are byte-identical at the same addresses · a mapping failure (FAKE_DEV_MALLOC_MAX) rolls back to released and the next restore succeeds · chunk return · multi-threaded alloc/free.
//   off : HIVE_SLEEP_VMM unset — devmem::on() false · DevBuf uses plain cudaMalloc (fake counter) · 0 regions (default behaviour unchanged).
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hive/model.h"

using namespace hive;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static uint8_t pat(size_t i, size_t k) { return (uint8_t)(i * 131 + k * 7 + 1); }
static void fill(DevBuf& b, size_t i) { auto* p = b.as<uint8_t>(); for (size_t k = 0; k < b.n; ++k) p[k] = pat(i, k); }
static bool same(const DevBuf& b, size_t i) { auto* p = b.as<uint8_t>(); for (size_t k = 0; k < b.n; ++k) if (p[k] != pat(i, k)) return false; return true; }

static int run_on() {
  EXPECT(devmem::on(), "devmem should be on with HIVE_SLEEP_VMM=1");
  std::mt19937 rng(3);
  std::vector<std::unique_ptr<DevBuf>> bufs;
  std::vector<void*> addr;
  for (int i = 0; i < 300; ++i) { bufs.push_back(std::make_unique<DevBuf>((size_t)(1 + rng() % (1 << 20)))); }
  for (size_t sz : {3ull << 20, 9ull << 20, 40ull << 20}) bufs.push_back(std::make_unique<DevBuf>(sz));
  for (size_t i = 0; i < bufs.size(); ++i) { fill(*bufs[i], i); addr.push_back(bufs[i]->p); EXPECT(((uintptr_t)bufs[i]->p & 255) == 0, "alignment"); }
  const size_t mapped0 = devmem::mapped_bytes();
  size_t total = 0;
  for (auto& b : bufs) total += b->n;
  EXPECT(mapped0 >= total && mapped0 < total + 40 * (32ull << 20), "mapped %zu vs requested %zu (small allocations must share chunks)", mapped0, total);
  for (int cycle = 0; cycle < 3 && !fails; ++cycle) {
    const auto r = devmem::release_all();
    EXPECT(r.ok && devmem::released() && devmem::mapped_bytes() == 0, "release cycle %d: ok %d mapped %zu", cycle, r.ok, devmem::mapped_bytes());
    int lost = 0;
    for (auto& b : bufs) lost += b && b->p && b->as<uint8_t>()[0] == 0xCD;
    EXPECT(lost > (int)bufs.size() / 2, "release did not drop the contents (fake unmap poisons) — %d", lost);
    if (cycle == 0) {  // free and allocate anew while asleep
      bufs[5].reset(); bufs[300].reset();
      DevBuf fresh(1000);
      EXPECT(fresh.p != nullptr && devmem::mapped_bytes() > 0, "alloc while released");
    }
    if (cycle == 1) {  // mapping failure: stays released
      setenv("FAKE_DEV_MALLOC_MAX", "1048576", 1);
      const auto f = devmem::restore_all();
      EXPECT(!f.ok && devmem::released() && devmem::mapped_bytes() == 0, "failed restore must leave everything released");
      unsetenv("FAKE_DEV_MALLOC_MAX");
    }
    const auto w = devmem::restore_all();
    EXPECT(w.ok && !devmem::released(), "restore cycle %d: %s", cycle, w.error.c_str());
    for (size_t i = 0; i < bufs.size(); ++i) {
      if (!bufs[i]) continue;
      EXPECT(bufs[i]->p == addr[i], "address moved for buffer %zu", i);
      if (!same(*bufs[i], i)) { EXPECT(false, "cycle %d: buffer %zu (%zu B) not restored byte-identical", cycle, i, bufs[i]->n); break; }
    }
  }
  // chunk return: after freeing all small ones the mappings shrink to the large regions (2 remaining)
  for (size_t i = 0; i < 300; ++i) bufs[i].reset();
  EXPECT(devmem::mapped_bytes() == (10ull << 20) + (40ull << 20), "after freeing small buffers mapped %zu (9 MiB → 10 MiB granularity + 40 MiB)", devmem::mapped_bytes());
  bufs.clear();
  EXPECT(devmem::mapped_bytes() == 0, "all freed: mapped %zu", devmem::mapped_bytes());
  // multiple threads
  std::vector<std::thread> th;
  for (int t = 0; t < 4; ++t)
    th.emplace_back([t] {
      std::mt19937 r(t);
      std::vector<std::unique_ptr<DevBuf>> v;
      for (int i = 0; i < 400; ++i) {
        if (!v.empty() && r() % 3 == 0) v.erase(v.begin() + (long)(r() % v.size()));
        v.push_back(std::make_unique<DevBuf>((size_t)(1 + r() % 4096)));
        v.back()->as<uint8_t>()[0] = (uint8_t)i;
      }
    });
  for (auto& x : th) x.join();
  EXPECT(devmem::mapped_bytes() == 0, "threads leaked %zu", devmem::mapped_bytes());
  if (fails) return 1;
  printf("devmem CPU (on): shared chunks · release/restore x3 at the same addresses, contents byte-identical · free/alloc while released · failed map stays released · chunk return · threads OK\n");
  return 0;
}

static int run_off() {
  EXPECT(!devmem::on(), "devmem must be off without HIVE_SLEEP_VMM");
  const long a0 = fakecuda::counters().dev_alloc.load();
  { DevBuf b(1000); EXPECT(b.p != nullptr, "alloc"); }
  EXPECT(fakecuda::counters().dev_alloc.load() == a0 + 1, "default path must use cudaMalloc");
  EXPECT(devmem::mapped_bytes() == 0 && !devmem::release_all().ok, "off: no regions, release refuses");
  if (fails) return 1;
  printf("devmem CPU (off): DevBuf uses cudaMalloc as before\n");
  return 0;
}

// first-sleep acceleration (prepare_shadows): shadows are pre-allocated only for keep regions · at sleep time only non-keep regions get new shadows · calling it while asleep does not
//   swap the content shadows · byte-identical after wake · the second sleep allocates 0 new shadows · safe while overlapping alloc/free on other threads (TSAN).
static int run_prepin() {
  EXPECT(devmem::on(), "devmem should be on with HIVE_SLEEP_VMM=1");
  auto& C = fakecuda::counters();
  std::mt19937 rng(11);
  std::vector<std::unique_ptr<DevBuf>> bufs;
  devmem::keep_begin();
  for (int i = 0; i < 200; ++i) bufs.push_back(std::make_unique<DevBuf>((size_t)(1 + rng() % (1 << 20))));  // small → keep chunk
  bufs.push_back(std::make_unique<DevBuf>((size_t)(9ull << 20)));
  bufs.push_back(std::make_unique<DevBuf>((size_t)(40ull << 20) + 123));
  devmem::keep_end();
  const size_t nkeep_big = 2;
  // two large non-keep regions (slot and session pool areas) — one freed before sleep (mimics what levels 1 and 2 free), one kept until sleep (a late buffer)
  auto transient = std::make_unique<DevBuf>((size_t)(64ull << 20));
  bufs.push_back(std::make_unique<DevBuf>((size_t)(5ull << 20) + 7));
  const size_t late_n = (5ull << 20) + 7;
  // while shadows are allocated on a background thread, this thread keeps allocating and freeing (overlaps hived's session pool and slot allocation)
  devmem::Report pr;
  const long h0 = C.host_alloc.load();
  std::thread prep([&] { pr = devmem::prepare_shadows(); });
  std::vector<std::unique_ptr<DevBuf>> churn;
  for (int i = 0; i < 300; ++i) {
    churn.push_back(std::make_unique<DevBuf>((size_t)(devmem::kSmall + 1 + rng() % (4 << 20))));  // large regions only (small ones would close the open keep chunk and blur the append test below)
    if (i % 3 == 0) churn.erase(churn.begin() + (long)(rng() % churn.size()));
  }
  prep.join();
  churn.clear();
  const long prep_allocs = C.host_alloc.load() - h0;
  EXPECT(pr.ok && pr.regions >= 1 + (int)nkeep_big && pr.shadow_failed == 0, "prepare: ok %d regions %d failed %d", pr.ok, pr.regions, pr.shadow_failed);
  EXPECT(prep_allocs == pr.regions, "prepare: %ld host allocations for %d regions (only keep regions)", prep_allocs, pr.regions);
  EXPECT(pr.shadow_new >= (9ull << 20) + (40ull << 20) + 123 && pr.shadow_new < (9ull << 20) + (40ull << 20) + 123 + 8 * (32ull << 20),
         "prepare pinned %zu B — keep big regions + keep chunks only (the transient 64 MiB must not get a shadow)", pr.shadow_new);
  // small allocation appended to the keep chunk after pre-allocation (covered by the whole-chunk shadow)
  for (int i = 0; i < 20; ++i) bufs.push_back(std::make_unique<DevBuf>((size_t)(1 + rng() % 4096)));
  for (size_t i = 0; i < bufs.size(); ++i) fill(*bufs[i], i);
  std::vector<void*> addr;
  for (auto& b : bufs) addr.push_back(b->p);
  transient.reset();  // what levels 1 and 2 free first
  for (int cycle = 0; cycle < 3 && !fails; ++cycle) {
    const long h1 = C.host_alloc.load();
    const auto r = devmem::release_all();
    const long sleep_allocs = C.host_alloc.load() - h1;
    EXPECT(r.ok && devmem::released(), "release cycle %d", cycle);
    if (cycle == 0) {
      // new shadows at first sleep = only the one late non-keep region (everything else reuses the pre-allocated ones)
      EXPECT(r.shadow_new == late_n && sleep_allocs == 1, "first sleep allocated %zu B in %ld host allocations (want only the late non-keep region %zu B)",
             r.shadow_new, sleep_allocs, late_n);
      // prepare_shadows while asleep: swaps nothing (the shadows hold contents)
      const long h2 = C.host_alloc.load();
      const auto again = devmem::prepare_shadows();
      EXPECT(again.shadow_new == 0 && C.host_alloc.load() == h2, "prepare while released must not touch the shadows (%zu B)", again.shadow_new);
    } else {
      EXPECT(r.shadow_new == 0 && sleep_allocs == 0, "sleep %d allocated %zu B (want 0 — shadows reused)", cycle, r.shadow_new);
    }
    const auto w = devmem::restore_all();
    EXPECT(w.ok && !devmem::released(), "restore cycle %d: %s", cycle, w.error.c_str());
    for (size_t i = 0; i < bufs.size(); ++i) {
      EXPECT(bufs[i]->p == addr[i], "address moved for buffer %zu", i);
      if (!same(*bufs[i], i)) { EXPECT(false, "cycle %d: buffer %zu (%zu B) not restored byte-identical", cycle, i, bufs[i]->n); break; }
    }
    if (cycle == 0) {  // calling it again while awake leaves regions that already have a shadow untouched
      const long h3 = C.host_alloc.load();
      const auto again = devmem::prepare_shadows();
      EXPECT(again.shadow_new == 0 && C.host_alloc.load() == h3, "prepare after wake re-pinned %zu B", again.shadow_new);
    }
  }
  bufs.clear();
  EXPECT(devmem::mapped_bytes() == 0, "all freed: mapped %zu", devmem::mapped_bytes());
  if (fails) return 1;
  printf("devmem CPU (prepin): keep-only shadows pinned ahead (%d regions, %.1f MiB) while other threads allocate · first sleep allocates only the late "
         "non-keep region · later sleeps 0 · prepare while asleep is a no-op · contents byte-identical x3\n", pr.regions, pr.shadow_new / 1048576.0);
  return 0;
}

static int run_prepin_off() {  // off: keep_* · prepare_shadows do nothing
  EXPECT(!devmem::on(), "devmem must be off without HIVE_SLEEP_VMM");
  devmem::keep_begin();
  { DevBuf b(1000); }
  devmem::keep_end();
  const auto r = devmem::prepare_shadows();
  EXPECT(r.shadow_new == 0 && r.regions == 0 && fakecuda::counters().host_alloc.load() == 0, "off: prepare must do nothing");
  if (fails) return 1;
  printf("devmem CPU (prepin off): keep scope and prepare_shadows are no-ops\n");
  return 0;
}

int main(int argc, char** argv) {
  const std::string m = argc > 1 ? argv[1] : "on";
  if (m == "off") return run_off() || run_prepin_off();
  if (m == "prepin") return run_prepin();
  return run_on();
}
