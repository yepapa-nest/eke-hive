// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_EARLY_ROUTE CPU test (no GPU; tools/test_early_route_cpu.py builds and runs it — TSAN variant + mutant negative controls).
//   Runs er::Gate and er::run_layer from the production header (hive/early_route.h) with a fake GPU stream thread (in-order execution, random delays) + a CPU pool thread.
//   (1) Gate: normal publish · earlier (stale) publish = kResync · no publish = kMissing · unclosed launch = untrusted — in every case kOk only for this launch's routing.
//   (2) run_layer: data model (uint64 hashes — exact integer arithmetic, reads and writes at the same places as the operations in production decode_layer)
//         prep(l)      host writes tag l into the mapped row table tab                   → the fake front end checks the tag **at start and end** (assumes it may be read anywhere in graph A)
//         front(l)     GPU: x = mix(acc, l) · route = H(x,1) → publish (route_h + sequence number, release) · shared expert (delay) · acts = H(x,2) → acts_h · front-end end (release)
//         launch(l)    host: read route_h (check tag l) → expert table tbl · CPU job table (without activations) · GPU expert launch (checks the tbl tag at execution · acc += G(route))
//         start_cpu(l) host: read acts_h (check tag l — the "deferred activation unpack") → job to the pool (cout = C(acts, route))
//         finish(l)    wait for the pool → GPU accumulation (acc += cout) — a job whose pool start never happened is started here as in production moe_decode_wait (without unpack → stale activations → mismatch)
//       Reference = sequential computation per layer. PASS if the final acc matches the reference and all tags match. Ordering violations (mutants) show up as tag mismatches, final-state mismatches or TSAN reports.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "hive/early_route.h"

using namespace hive;
using Micros = std::chrono::microseconds;

static std::atomic<int> g_fail{0};
#define EXPECT(cond, ...) do { if (!(cond)) { if (g_fail.fetch_add(1) < 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static uint64_t mix(uint64_t a, uint64_t b) { uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2)); x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; return x; }
static uint32_t relaxed_load(const uint32_t* p) { return std::atomic_ref<uint32_t>(*const_cast<uint32_t*>(p)).load(std::memory_order_relaxed); }
static void release_store(uint32_t* p, uint32_t v) { std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release); }

// Fake GPU stream: executes in insertion order on one thread (stream order) · random delay per operation
struct Stream {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::function<void()>> q;
  bool stop = false;
  std::thread th;
  explicit Stream(uint32_t seed) : th([this, seed] { run(seed); }) {}
  ~Stream() { { std::scoped_lock lk(mu); stop = true; } cv.notify_all(); th.join(); }
  void push(std::function<void()> f) { { std::lock_guard<std::mutex> lk(mu); q.push_back(std::move(f)); } cv.notify_all(); }
  void sync() {
    std::atomic<bool> done{false};
    push([&] { done.store(true, std::memory_order_release); });
    while (!done.load(std::memory_order_acquire)) std::this_thread::yield();
  }
  void run(uint32_t seed) {
    std::mt19937 rng(seed);
    for (;;) {
      std::function<void()> f;
      {
        std::unique_lock lk(mu);
        cv.wait(lk, [&] { return stop || !q.empty(); });
        if (q.empty()) return;
        f = std::move(q.front());
        q.pop_front();
      }
      if (rng() % 4 == 0) std::this_thread::sleep_for(Micros(rng() % 60));
      f();
    }
  }
};

// ---- (1) Gate ----
static void test_gate(int iters) {
  uint32_t flag = 0;
  uint64_t data = 0;           // "mapped routing" (written by the GPU, published with a sequence number)
  std::atomic<uint64_t> front_done{0};  // number of the last finished launch (from 1)
  Stream gpu(7);
  er::Gate g;
  std::mt19937 rng(11);
  uint32_t ctr = 0;            // device counter (GPU thread only)
  long n_ok = 0, n_res = 0, n_miss = 0, n_untr = 0;
  for (int it = 1; it <= iters; ++it) {
    const int kind = rng() % 16 == 0 ? 1 : rng() % 23 == 0 ? 2 : rng() % 29 == 0 ? 3 : 0;  // 0 normal · 1 publish with a number ahead · 2 no publish · 3 unclosed launch (a stale publish arrives first)
    const uint64_t payload = mix((uint64_t)it, 99);
    if (kind == 3) {  // as if the previous launch was skipped without waiting: one extra begin
      (void)g.begin();
      gpu.push([&, it] { data = mix((uint64_t)it, 5); release_store(&flag, ++ctr); });  // that launch's publish (stale value)
    }
    const bool trust = g.begin();
    EXPECT(trust == (kind != 3), "it %d: begin trust %d kind %d", it, (int)trust, kind);
    if (kind == 1) gpu.push([&, payload] { data = payload; ctr += 2; release_store(&flag, ctr); });  // counter skew gives a number ahead (skips the expected value)
    else if (kind != 2) gpu.push([&, payload] { data = payload; release_store(&flag, ++ctr); });
    gpu.push([&, it] { front_done.store((uint64_t)it, std::memory_order_release); });
    auto done = [&] { return front_done.load(std::memory_order_acquire) >= (uint64_t)it; };
    int r = er::kMissing;
    if (trust) r = g.wait(&flag, done, 8);
    if (r == er::kOk) {
      EXPECT(data == payload, "it %d kind %d: kOk with stale routing", it, kind);
      ++n_ok;
    } else {
      // absorbed: after the front-end end (caller) — if this launch published, that value is the last one
      while (!done()) std::this_thread::yield();
      g.resync(&flag);
      EXPECT(g.seq == relaxed_load(&flag), "it %d: resync seq %u flag %u", it, g.seq, relaxed_load(&flag));
      if (!trust) ++n_untr; else if (r == er::kResync) ++n_res; else ++n_miss;
      EXPECT(kind != 0, "it %d: normal post not accepted (r=%d)", it, r);
      if (trust) EXPECT((kind == 1 && r == er::kResync) || (kind == 2 && r == er::kMissing), "it %d: kind %d gave %d", it, kind, r);
    }
    EXPECT(!g.open, "it %d: gate left open", it);
  }
  gpu.sync();
  printf("early-route CPU: Gate %d launches · ok %ld · resync %ld · missing %ld · untrusted %ld\n", iters, n_ok, n_res, n_miss, n_untr);
  EXPECT(n_ok > 0 && n_res > 0 && n_miss > 0 && n_untr > 0, "gate coverage");
}

// ---- (2) run_layer ----
struct Pool {  // one batch at a time (same contract as the HIVE_CHECK in production ExpertStore::start_jobs)
  std::mutex mu;
  std::condition_variable cv;
  bool has = false, done = true, stop = false;
  uint64_t in_acts = 0, in_route = 0, out = 0;
  int layer = -1;
  std::thread th;
  explicit Pool(uint32_t seed) : th([this, seed] { run(seed); }) {}
  ~Pool() { { std::scoped_lock lk(mu); stop = true; } cv.notify_all(); th.join(); }
  void start(int l, uint64_t acts, uint64_t route) {
    std::lock_guard<std::mutex> lk(mu);
    EXPECT(done && !has, "pool: second batch while busy (layer %d)", l);
    has = true; done = false; in_acts = acts; in_route = route; layer = l;
    cv.notify_all();
  }
  uint64_t wait() { std::unique_lock lk(mu); cv.wait(lk, [&] { return done; }); return out; }
  void run(uint32_t seed) {
    std::mt19937 rng(seed);
    std::unique_lock lk(mu);
    for (;;) {
      cv.wait(lk, [this] { return stop || has; });
      if (!has) return;
      const uint64_t a = in_acts, r = in_route;
      lk.unlock();
      std::this_thread::sleep_for(Micros(rng() % 80));
      const uint64_t o = mix(mix(a, r), 3);
      lk.lock();
      out = o; has = false; done = true;
      cv.notify_all();
    }
  }
};

struct Model {
  // "mapped host" buffers — tag = layer that wrote them
  struct Tagged { int tag = -1; uint64_t v = 0; };
  Tagged tab, route_h, acts_h, tbl;
  uint32_t flag = 0;             // publish sequence number (mapped)
  std::atomic<long> front_seq{0};  // number of finished front ends (event)
  // GPU side (stream thread only)
  uint64_t acc = 1, x = 0, route_dev = 0, acts_dev = 0;
  uint32_t ctr = 0;
};

struct LayerOps {
  Model& md; Stream& gpu; Pool& pool; er::Gate& gate; int nL; long launch_no = 0;
  // host-side layer state (where MoePend sits in production)
  uint64_t job_route = 0, job_acts = 0;
  bool has_cpu = false, started = false, unpacked = false;
  void prep(int l) { md.tab = {l, mix((uint64_t)l, 17)}; }
  void front(int l) {
    (void)gate.begin();
    const long my = ++launch_no;
    gpu.push([this, l] {  // front-end start: reads the mapped row table
      EXPECT(md.tab.tag == l, "front(%d) start: table tag %d", l, md.tab.tag);
      md.x = mix(md.acc, (uint64_t)l ^ md.tab.v);
      md.route_dev = mix(md.x, 1);
    });
    gpu.push([this, l] { md.route_h = {l, md.route_dev}; release_store(&md.flag, ++md.ctr); });  // er_route_post
    gpu.push([this] { md.acc = md.x; md.acts_dev = mix(md.x, 2); });  // shared expert (acc init · produces xq)
    gpu.push([this, l, my] {  // route_to_host (activations only) · front-end end — the table may be read anywhere in the front end (checked again at the end)
      md.acts_h = {l, md.acts_dev};
      EXPECT(md.tab.tag == l, "front(%d) end: table tag %d (rewritten while the front graph ran)", l, md.tab.tag);
      md.front_seq.store(my, std::memory_order_release);
    });
  }
  bool front_done() const { return md.front_seq.load(std::memory_order_acquire) >= launch_no; }
  void wait_route(int l) {
    const int r = gate.wait(&md.flag, [&] { return front_done(); }, 16);
    EXPECT(r == er::kOk, "layer %d: gate %d", l, r);
  }
  void launch(int l) {
    EXPECT(md.route_h.tag == l, "launch(%d): routing tag %d", l, md.route_h.tag);
    const uint64_t route = md.route_h.v;
    md.tbl = {l, route};
    gpu.push([this, l] {  // GPU experts (after the table H2D — table checked at execution)
      EXPECT(md.tbl.tag == l, "experts(%d): table tag %d", l, md.tbl.tag);
      md.acc += mix(md.tbl.v, 4);
    });
    has_cpu = (route & 3) != 0;  // about 3/4 of the layers have a CPU share
    started = false; unpacked = false;
    job_route = route; job_acts = 0xdeadull;  // the activation unpack is deferred
  }
  void wait_front(int) { while (!front_done()) std::this_thread::yield(); }
  void start_cpu(int l) {
    if (!has_cpu) return;
    EXPECT(md.acts_h.tag == l, "start_cpu(%d): activation tag %d", l, md.acts_h.tag);
    job_acts = md.acts_h.v; unpacked = true;
    pool.start(l, job_acts, job_route); started = true;
  }
  void prep_next(int l) { if (l + 1 < nL) prep(l + 1); }
  void finish(int l) {
    if (!has_cpu) return;
    if (!started) { pool.start(l, job_acts, job_route); started = true; }  // start without unpack, as in production moe_decode_wait
    const uint64_t out = pool.wait();
    gpu.push([this, out] { md.acc += out; });
  }
};

static uint64_t reference(int nL, int layers_seed) {
  uint64_t acc = 1;
  for (int l = 0; l < nL; ++l) {
    const uint64_t x = mix(acc, (uint64_t)l ^ mix((uint64_t)l, 17));
    const uint64_t route = mix(x, 1), acts = mix(x, 2);
    acc = x + mix(route, 4);
    if ((route & 3) != 0) acc += mix(mix(acts, route), 3);
  }
  (void)layers_seed;
  return acc;
}

static void test_layers(int steps, int nL) {
  long bad = 0;
  for (int s = 0; s < steps; ++s) {
    Model md;
    md.acc = 1;
    Stream gpu(1000u + (uint32_t)s);
    Pool pool(2000u + (uint32_t)s);
    er::Gate gate;
    LayerOps o{md, gpu, pool, gate, nL};
    o.prep(0);  // table of the first layer (production: decode_host_prep at the top of decode_layer)
    for (int l = 0; l < nL; ++l) er::run_layer(l, /*prepped=*/true, o);
    gpu.sync();
    const uint64_t want = reference(nL, s);
    if (md.acc != want) { ++bad; EXPECT(false, "step %d: final state %llx want %llx", s, (unsigned long long)md.acc, (unsigned long long)want); }
    EXPECT(gate.n_ok == nL && gate.n_resync == 0 && gate.n_missing == 0, "step %d: gate ok %ld", s, gate.n_ok);
  }
  printf("early-route CPU: run_layer %d steps × %d layers · final state == sequential reference: %s\n", steps, nL, bad == 0 ? "yes" : "NO");
}

int main(int argc, char** argv) {
  const int gate_iters = argc > 1 ? atoi(argv[1]) : 3000;
  const int steps = argc > 2 ? atoi(argv[2]) : 40;
  const int nL = argc > 3 ? atoi(argv[3]) : 40;
  test_gate(gate_iters);
  test_layers(steps, nL);
  if (g_fail.load()) { printf("FAIL (%d)\n", g_fail.load()); return 1; }
  printf("early-route CPU: PASS\n");
  return 0;
}
