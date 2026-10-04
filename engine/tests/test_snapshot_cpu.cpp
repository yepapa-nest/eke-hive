// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU-only stress of the PRODUCTION snapshot path: Runtime::save_image / load_image / reset_seq
// (text sliced from engine/src/runtime.cpp by tools/cpu_fake/harness.py), HostImagePool
// (include/hive/host_image.h) and SeqImage/Seq fences (include/hive/runtime.h), on the fake CUDA
// layer. With FAKE_CUDA_ASYNC=1 the snapshot D2H copies run on a real worker thread and are
// ordered only by the production fences, so under TSAN any missing wait is reported, and the
// fake state oracle (tools/cpu_fake/fake_runtime.inc) catches a snapshot that captured the wrong
// bytes. Built/run by tools/test_snapshot_cpu.py. No GPU.
//
// Part 1: HostImagePool concurrent acquire/release/trim from many threads + pool destroyed while
//         blocks are outstanding (the shared owner keeps it alive).
// Part 2: random single-engine-thread schedule over 3 Seqs: forward chunks, save (with/without
//         delta base, into fresh or reused images), load into another Seq, reset, trim pool, drop
//         images on the engine thread or on a second thread; every forward's token is predicted.
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "hive/host_image.h"
#include "hive/kv_pack.h"
#include "hive/runtime.h"

using namespace hive;
namespace hive { extern std::atomic<long> fake_host_alloc_calls, fake_host_alloc_injected; }  // tools/cpu_fake/fake_runtime.inc
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static void pool_stress(int threads, int iters) {
  std::atomic<long> allocs{0}, frees{0}, live{0};
  const size_t limit = 64 * 1024;
  auto pool = std::make_shared<HostImagePool>(limit,
      [&](size_t n) { ++allocs; ++live; auto* p = new uint8_t[n]; p[0] = 1; return p; },
      [&](uint8_t* p) { ++frees; --live; delete[] p; });
  std::vector<std::thread> th;
  std::atomic<bool> bad{false};
  std::vector<std::shared_ptr<uint8_t>> survivors(threads);
  for (int t = 0; t < threads; ++t)
    th.emplace_back([&, t] {
      std::mt19937 rng(t);
      std::vector<std::shared_ptr<uint8_t>> held;
      for (int i = 0; i < iters; ++i) {
        const size_t n = 1024u << (rng() % 5);  // 1..16 KiB exact-size bins
        auto b = pool->acquire(n);
        b.get()[n - 1] = (uint8_t)t;  // exclusive: another thread holding the same block would race here
        if (rng() % 3) held.push_back(std::move(b));
        if (held.size() > 6 || rng() % 4 == 0) { if (!held.empty()) held.erase(held.begin() + rng() % held.size()); }
        if (i % 97 == 0) pool->trim();
        if (pool->cached_bytes() > limit) bad = true;
      }
      if (!held.empty()) survivors[t] = held.back();  // outlives the pool below
    });
  for (auto& x : th) x.join();
  pool.reset();  // blocks still held by `survivors` keep the pool alive through their deleter
  EXPECT(live.load() > 0 || survivors.empty(), "pool freed blocks that are still referenced");
  survivors.clear();
  EXPECT(!bad, "cached bytes exceeded the pool limit");
  EXPECT(allocs.load() == frees.load() && live.load() == 0, "pool alloc/free mismatch: %ld allocs %ld frees", allocs.load(), frees.load());
  printf("snapshot CPU: HostImagePool %d threads x %d acquire/release/trim, limit held, %ld allocs == frees\n", threads, iters, allocs.load());
}

namespace {
constexpr uint64_t kB = 1469598103934665603ull, kP = 1099511628211ull;
int32_t expect_next(const std::vector<uint64_t>& truth) {
  uint64_t h = kB;
  for (uint64_t c : truth) for (int i = 0; i < 8; ++i) { h ^= (uint8_t)(c >> (8 * i)); h *= kP; }
  return (int32_t)(10 + h % 4000);
}
struct Image { std::shared_ptr<SeqImage> img; std::vector<uint64_t> truth; int from = -1; };
// Drops images on another thread (SeqImage destructor waits its fence, pool blocks return there).
struct Dropper {
  std::mutex mu; std::condition_variable cv; std::deque<std::shared_ptr<SeqImage>> q; bool stop = false; long dropped = 0;
  std::thread th{[this] { run(); }};
  void run() {
    for (;;) {
      std::shared_ptr<SeqImage> x;
      { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return stop || !q.empty(); }); if (q.empty()) return; x = std::move(q.front()); q.pop_front(); }
      x.reset(); ++dropped;
    }
  }
  void push(std::shared_ptr<SeqImage> x) { { std::lock_guard<std::mutex> l(mu); q.push_back(std::move(x)); } cv.notify_one(); }
  ~Dropper() { { std::lock_guard<std::mutex> l(mu); stop = true; } cv.notify_one(); th.join(); }
};
}  // namespace

static void snapshot_stress(int steps, unsigned seed) {
  Model model("fake");
  ExpertStore store(model.cfg(), model.n_loaded_layers(), 0, 1, 0);
  for (int l = 0; l < model.n_loaded_layers(); ++l) store.load_layer_experts(model.ckpt(), l, 2);
  RuntimeOptions opt; opt.max_ctx = 1536; opt.max_batch = 4;
  const long host0 = fakecuda::counters().host_alloc - fakecuda::counters().host_free;
  long saves = 0, loads = 0, forwards = 0, delta_saves = 0, reused = 0, shared = 0; size_t pool_peak = 0;
  {
    Runtime rt(model, store, nullptr, opt);
    std::vector<std::unique_ptr<Seq>> seqs;
    std::vector<std::vector<uint64_t>> truth(3);
    for (int i = 0; i < 3; ++i) seqs.push_back(rt.new_seq());
    std::vector<Image> images;
    Dropper dropper;
    std::mt19937 rng(seed);
    std::vector<float> logits;
    auto forward = [&](int i, int M) {
      if (seqs[i]->pos + M > seqs[i]->cap) { rt.reset_seq(*seqs[i]); truth[i].clear(); }
      std::vector<int32_t> ids(M);
      for (auto& v : ids) { v = 20 + (int)(rng() % 3000); truth[i].push_back((uint32_t)v); }
      const int32_t got = rt.forward(*seqs[i], ids.data(), M, nullptr, &logits, nullptr);
      ++forwards;
      EXPECT(got == expect_next(truth[i]), "seq %d pos %lld: state oracle mismatch", i, (long long)seqs[i]->pos);
    };
    for (int step = 0; step < steps && fails < 5; ++step) {
      const int i = (int)(rng() % 3), op = (int)(rng() % 10);
      if (op <= 3) forward(i, 1 + (int)(rng() % 48));
      else if (op <= 5) {  // save, sometimes delta against an earlier image of this seq, sometimes into a reused image
        const Image* base = nullptr;
        for (auto& im : images) if (im.from == i && rng() % 2) base = &im;
        Image out; out.truth = truth[i]; out.from = i;
        if (!images.empty() && rng() % 4 == 0) {  // overwrite an existing image object in place (in-flight fence path)
          const size_t k = rng() % images.size();
          if (&images[k] != base && images[k].img.use_count() == 1) {
            rt.save_image(*seqs[i], *images[k].img, base ? base->img.get() : nullptr);
            images[k].truth = truth[i]; images[k].from = i; ++reused; ++saves; if (base) ++delta_saves;
            continue;
          }
        }
        out.img = std::make_shared<SeqImage>();
        rt.save_image(*seqs[i], *out.img, base ? base->img.get() : nullptr);
        if (base) ++delta_saves;
        ++saves;
        if (!out.img->comp.empty() && out.img->comp.begin()->second.segments.size() > 1) ++shared;  // prefix segments shared with base
        images.push_back(std::move(out));
        forward(i, 1 + (int)(rng() % 8));  // keep writing the source right after the async snapshot
      } else if (op == 6 && !images.empty()) {  // restore into another seq and continue there
        const size_t k = rng() % images.size();
        const int j = (int)(rng() % 3);
        rt.load_image(*seqs[j], *images[k].img);
        truth[j] = images[k].truth;
        ++loads;
        forward(j, 1 + (int)(rng() % 4));
      } else if (op == 7) { rt.reset_seq(*seqs[i]); truth[i].clear(); }
      else if (op == 8 && !images.empty()) {
        const size_t k = rng() % images.size();
        if (rng() % 2) dropper.push(std::move(images[k].img)); else images[k].img.reset();
        images.erase(images.begin() + k);
      } else if (op == 9) rt.trim_snapshot_pool();
      pool_peak = std::max(pool_peak, rt.snapshot_pool_cached_bytes());
    }
    for (auto& im : images) dropper.push(std::move(im.img));
    images.clear();
    for (auto& s : seqs) rt.reset_seq(*s);
    rt.trim_snapshot_pool();
  }
  const long host_live = fakecuda::counters().host_alloc - fakecuda::counters().host_free - host0;
  EXPECT(host_live == 0, "pinned host buffers leaked or double-freed: live %ld", host_live);
  const char* d = getenv("HIVE_CKPT_DELTA");
  if (d && atoi(d)) EXPECT(shared > 0, "HIVE_CKPT_DELTA=1 but no snapshot shared a prefix segment");
  else EXPECT(shared == 0, "prefix segments shared with delta off");
  const char* a = getenv("HIVE_CKPT_ASYNC"); const char* pm = getenv("HIVE_CKPT_PINNED_POOL_MB");
  const char* inj = getenv("FAKE_HOST_ALLOC_FAIL");
  const bool fail_all = inj && atoi(inj) == 1;
  if (a && atoi(a) && inj && atoi(inj) > 0) EXPECT(fake_host_alloc_injected.load() > 0, "FAKE_HOST_ALLOC_FAIL set but no pinned allocation failed");
  if (a && atoi(a) && pm && atoi(pm) && fail_all) EXPECT(pool_peak == 0, "every pinned allocation failed but the pool cached a block");
  else if (a && atoi(a) && pm && atoi(pm)) EXPECT(pool_peak > 0, "pinned pool configured but never cached a block");
  else EXPECT(pool_peak == 0, "pinned pool active although async or pool size is off");
  printf("snapshot CPU: %ld forwards exact, %ld saves (%ld with base, %ld sharing prefix, %ld into reused images), %ld loads, pool peak %zu B; async %s pool %s delta %s; pinned live after teardown %ld; injected pinned failures %ld/%ld\n",
         forwards, saves, delta_saves, shared, reused, loads, pool_peak, getenv("HIVE_CKPT_ASYNC") ? getenv("HIVE_CKPT_ASYNC") : "-",
         getenv("HIVE_CKPT_PINNED_POOL_MB") ? getenv("HIVE_CKPT_PINNED_POOL_MB") : "-", getenv("HIVE_CKPT_DELTA") ? getenv("HIVE_CKPT_DELTA") : "-", host_live,
         fake_host_alloc_injected.load(), fake_host_alloc_calls.load());
}

// Part 3: byte-exact image == device state (what the sync path copies) and restore == source, in every
// option combination — with FAKE_HOST_ALLOC_FAIL this is the pinned-failure fallback (no abort, pageable
// segments mixed with in-flight pinned ones under one fence).
static std::vector<uint8_t> flat(const HostImageBuffer& b) {
  std::vector<uint8_t> v;
  for (auto& s : b.segments) v.insert(v.end(), s.data.get(), s.data.get() + s.n);
  return v;
}
static bool same_dev(const void* dev, const std::vector<uint8_t>& host) { return memcmp(dev, host.data(), host.size()) == 0; }
static void image_exact(unsigned seed) {
  Model model("fake");
  ExpertStore store(model.cfg(), model.n_loaded_layers(), 0, 1, 0);
  for (int l = 0; l < model.n_loaded_layers(); ++l) store.load_layer_experts(model.ckpt(), l, 2);
  RuntimeOptions opt; opt.max_ctx = 1536; opt.max_batch = 4;
  const long inj0 = fake_host_alloc_injected.load();
  long checked = 0;
  {
    Runtime rt(model, store, nullptr, opt);
    auto src = rt.new_seq(), dst = rt.new_seq();
    std::mt19937 rng(seed);
    std::vector<float> logits;
    std::shared_ptr<SeqImage> prev;
    for (int round = 0; round < 6 && fails < 5; ++round) {
      std::vector<int32_t> ids(1 + (int)(rng() % 40));
      for (auto& v : ids) v = 20 + (int)(rng() % 3000);
      rt.forward(*src, ids.data(), (int)ids.size(), nullptr, &logits, nullptr);
      auto img = std::make_shared<SeqImage>();
      rt.save_image(*src, *img, prev.get());  // delta against the previous image when HIVE_CKPT_DELTA=1
      if (img->fence) img->fence->wait();
      EXPECT(img->pos == src->pos && img->ring.size() == src->ring.size(), "image shape");
      for (size_t l = 0; l < src->ring.size(); ++l) {
        const auto r = flat(img->ring[l]);
        EXPECT(r.size() == src->ring[l].n && same_dev(src->ring[l].p, r), "round %d layer %zu: ring image != device", round, l);
        const auto c = flat(img->comp.at((int)l));
        EXPECT(c.size() == (size_t)src->pos * kvp::COMP_ROW && same_dev(src->comp_cache.at((int)l).p, c), "round %d layer %zu: comp image != device", round, l);
      }
      rt.load_image(*dst, *img);
      for (size_t l = 0; l < src->ring.size(); ++l) {
        EXPECT(memcmp(dst->ring[l].p, src->ring[l].p, src->ring[l].n) == 0, "round %d layer %zu: restored ring differs", round, l);
        EXPECT(memcmp(dst->comp_cache.at((int)l).p, src->comp_cache.at((int)l).p, (size_t)src->pos * kvp::COMP_ROW) == 0, "round %d: restored comp differs", round);
      }
      const int32_t t = 20 + (int)(rng() % 3000);
      const int32_t a = rt.forward(*src, &t, 1, nullptr, &logits, nullptr), b = rt.forward(*dst, &t, 1, nullptr, &logits, nullptr);
      EXPECT(a == b, "round %d: restored seq predicts %d, source %d", round, b, a);
      prev = img; ++checked;
    }
    prev.reset();
    rt.reset_seq(*src); rt.reset_seq(*dst);
    rt.trim_snapshot_pool();
  }
  printf("snapshot CPU: %ld images byte-exact vs device and restored exactly (injected pinned failures here %ld)\n", checked, fake_host_alloc_injected.load() - inj0);
}

int main(int argc, char** argv) {
  const int steps = argc > 1 ? atoi(argv[1]) : 600;
  pool_stress(8, argc > 2 ? atoi(argv[2]) : 2000);
  snapshot_stress(steps, 7);
  image_exact(11);
  return fails ? 1 : 0;
}
