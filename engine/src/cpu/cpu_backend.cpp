// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU backend: thread pool, NUMA memory, weight matrices and the small helpers — see hive/cpu/cpu_backend.h.
//   The matmul kernels are in cpu_matmul.cpp, attention in cpu_attention.cpp.
//
// Pool. Workers are pinned to their node's CPUs (all hardware threads of the node; the worker count defaults to the physical cores) and
//   wait for a dispatch by spinning HIVE_CPU_POOL_SPIN_US (default 200 µs: a decode step issues its next dispatch within microseconds)
//   before sleeping on a condition variable. A dispatch publishes per-queue task words [generation 24 | count 20 | next 20]; a worker
//   claims an index with a CAS on that word, so a worker still finishing an older dispatch can never take a task of a newer one (the
//   generation tag differs). run() uses one queue shared by every worker, run_nodes() one queue per node, served only by that node's
//   workers. The caller spins until every task has finished. One dispatcher at a time: the pool is not re-entrant and must not be called
//   from two threads at once (the engines dispatch from their single engine thread).
#include "hive/cpu/cpu_backend.h"

#include <immintrin.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

#include "hive/common.h"

namespace hive::cpu {

namespace {
bool numa_on() {
  static const bool on = numa_available() >= 0 && numa_max_node() >= 1 && !(getenv("HIVE_CPU_NUMA") && strcmp(getenv("HIVE_CPU_NUMA"), "0") == 0);
  return on;
}
// physical cores: distinct (package, core id) pairs of the online CPUs
int physical_cores() {
  std::set<std::pair<int, int>> ids;
  const long n = sysconf(_SC_NPROCESSORS_ONLN);
  for (long c = 0; c < n; ++c) {
    int pkg = -1, core = -1;
    std::ifstream a("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/physical_package_id");
    std::ifstream b("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/core_id");
    if (!(a >> pkg) || !(b >> core)) return (int)std::max(1L, n);
    ids.insert({pkg, core});
  }
  return std::max(1, (int)ids.size());
}
constexpr uint64_t kIdxMask = (1ull << 20) - 1, kGenMask = (1ull << 24) - 1;
}  // namespace

struct Pool::Impl {
  int n = 1, nodes = 1, spin_us = 200;
  // SMT helpers (HIVE_CPU_POOL_SMT=1): one extra worker per remaining hardware thread of the node, pinned like the others, that only joins
  //   dispatches flagged compute-bound (the GEMM). They sleep on their own condition variable between such dispatches, so a decode step
  //   (memory-bound GEMV, attention) never wakes them and their spinning never takes issue slots from the sibling core.
  int n_smt = 0, smt_spin_us = 50;
  std::atomic<uint64_t> smt_gen{0}, smt_tag{0};
  std::condition_variable cv_smt;
  std::vector<int> node_of;
  std::vector<std::thread> th;
  std::atomic<uint64_t> gen{0};
  std::atomic<uint64_t> next[2];
  std::atomic<long> pending{0};
  bool shared = false;
  const std::function<void(int)>* f1 = nullptr;
  const std::function<void(int, int)>* f2 = nullptr;
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> stop{false};
  void worker(int id) {
    const int node = node_of[id];
    if (nodes > 1) {
      bitmask* bm = numa_allocate_cpumask();
      if (numa_node_to_cpus(node, bm) == 0) {
        cpu_set_t cs; CPU_ZERO(&cs);
        for (unsigned i = 0; i < bm->size && i < CPU_SETSIZE; ++i) if (numa_bitmask_isbitset(bm, i)) CPU_SET(i, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
      }
      numa_free_cpumask(bm);
    }
    uint64_t seen = 0;
    for (;;) {
      uint64_t g = gen.load(std::memory_order_acquire);
      if (g == seen) {
        const auto t0 = std::chrono::steady_clock::now();
        while ((g = gen.load(std::memory_order_acquire)) == seen && !stop.load(std::memory_order_relaxed) &&
               std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < spin_us)
          _mm_pause();
        if (g == seen) {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&] { return stop.load() || gen.load() != seen; });
          if (stop.load()) return;
          g = gen.load();
        }
      }
      if (stop.load()) return;
      seen = g;
      const int q = shared ? 0 : node;
      const uint64_t tag = g & kGenMask;
      for (;;) {
        uint64_t v = next[q].load(std::memory_order_acquire);
        if ((v >> 40) != tag) break;
        const uint64_t i = v & kIdxMask, c = (v >> 20) & kIdxMask;
        if (i >= c) break;
        if (!next[q].compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) continue;
        if (shared) (*f1)((int)i);
        else (*f2)(q, (int)i);
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
    }
  }
  void smt_worker(int node) {
    if (nodes > 1) {
      bitmask* bm = numa_allocate_cpumask();
      if (numa_node_to_cpus(node, bm) == 0) {
        cpu_set_t cs; CPU_ZERO(&cs);
        for (unsigned i = 0; i < bm->size && i < CPU_SETSIZE; ++i) if (numa_bitmask_isbitset(bm, i)) CPU_SET(i, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
      }
      numa_free_cpumask(bm);
    }
    uint64_t seen = 0;
    for (;;) {
      uint64_t g = smt_gen.load(std::memory_order_acquire);
      if (g == seen) {
        const auto t0 = std::chrono::steady_clock::now();
        while ((g = smt_gen.load(std::memory_order_acquire)) == seen && !stop.load(std::memory_order_relaxed) &&
               std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < smt_spin_us)
          _mm_pause();
        if (g == seen) {
          std::unique_lock<std::mutex> lk(mu);
          cv_smt.wait(lk, [&] { return stop.load() || smt_gen.load() != seen; });
          if (stop.load()) return;
          g = smt_gen.load();
        }
      }
      if (stop.load()) return;
      seen = g;
      const uint64_t tag = smt_tag.load(std::memory_order_acquire);
      const int q = shared ? 0 : node;
      for (;;) {
        uint64_t v = next[q].load(std::memory_order_acquire);
        if ((v >> 40) != tag) break;
        const uint64_t i = v & kIdxMask, c = (v >> 20) & kIdxMask;
        if (i >= c) break;
        if (!next[q].compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) continue;
        if (shared) (*f1)((int)i);
        else (*f2)(q, (int)i);
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
    }
  }
  void dispatch(uint64_t c0, uint64_t c1, bool sh, bool smt = false) {
    HIVE_CHECK(c0 <= kIdxMask && c1 <= kIdxMask, "cpu pool: too many tasks in one dispatch");
    const uint64_t ng = gen.load() + 1, tag = ng & kGenMask;
    shared = sh;
    pending.store((long)(c0 + c1), std::memory_order_release);
    next[0].store((tag << 40) | (c0 << 20), std::memory_order_release);
    next[1].store((tag << 40) | (c1 << 20), std::memory_order_release);
    { std::lock_guard<std::mutex> lk(mu); gen.store(ng, std::memory_order_release); if (smt && n_smt) { smt_tag.store(tag); smt_gen.fetch_add(1); } }
    cv.notify_all();
    if (smt && n_smt) cv_smt.notify_all();
    while (pending.load(std::memory_order_acquire) > 0) _mm_pause();
  }
};

Pool& Pool::get() { static Pool p; return p; }

Pool::Pool() : impl_(std::make_unique<Impl>()) {
  Impl& I = *impl_;
  nodes_ = numa_on() ? 2 : 1;
  const char* tv = getenv("HIVE_CPU_POOL_THREADS");
  n_ = tv && atoi(tv) > 0 ? atoi(tv) : physical_cores();
  if (nodes_ > 1 && n_ < 2) n_ = 2;
  if (const char* s = getenv("HIVE_CPU_POOL_SPIN_US")) I.spin_us = std::max(0, atoi(s));
  I.n = n_; I.nodes = nodes_;
  per_node_.assign(nodes_, 0);
  for (int t = 0; t < n_; ++t) {
    const int node = nodes_ > 1 ? (t < (n_ + 1) / 2 ? 0 : 1) : 0;
    I.node_of.push_back(node);
    ++per_node_[node];
  }
  I.next[0] = 0; I.next[1] = 0;
  for (int t = 0; t < n_; ++t) I.th.emplace_back([this, t] { impl_->worker(t); });
  if (const char* e = getenv("HIVE_CPU_POOL_SMT"); e && atoi(e) > 0) {
    const int hw = (int)sysconf(_SC_NPROCESSORS_ONLN);
    I.n_smt = std::max(0, hw - n_);
    if (const char* sp = getenv("HIVE_CPU_POOL_SMT_SPIN_US")) I.smt_spin_us = std::max(0, atoi(sp));
    for (int t = 0; t < I.n_smt; ++t) {
      const int node = nodes_ > 1 ? (t < (I.n_smt + 1) / 2 ? 0 : 1) : 0;
      I.th.emplace_back([this, node] { impl_->smt_worker(node); });
    }
  }
}

Pool::~Pool() {
  { std::lock_guard<std::mutex> lk(impl_->mu); impl_->stop = true; impl_->gen.fetch_add(1); }
  impl_->cv.notify_all();
  impl_->cv_smt.notify_all();
  for (auto& t : impl_->th) t.join();
}

void Pool::run(int n, const std::function<void(int)>& f, bool compute_bound) {
  if (n <= 0) return;
  if (n == 1) { f(0); return; }
  impl_->f1 = &f;
  impl_->dispatch((uint64_t)n, 0, true, compute_bound);
}

int Pool::smt_threads() const { return impl_->n_smt; }

void Pool::run_nodes(const std::vector<int>& n_per_node, const std::function<void(int, int)>& f, bool compute_bound) {
  const int a = n_per_node.size() > 0 ? n_per_node[0] : 0, b = n_per_node.size() > 1 ? n_per_node[1] : 0;
  if (a + b <= 0) return;
  if (nodes_ == 1) {  // one node: every task goes to the shared queue, node index kept for the caller
    const std::function<void(int)> g = [&](int i) { if (i < a) f(0, i); else f(1, i - a); };
    run(a + b, g, compute_bound);
    return;
  }
  impl_->f2 = &f;
  impl_->dispatch((uint64_t)a, (uint64_t)b, false, compute_bound);
}

// ---- memory -------------------------------------------------------------------------------------------------------------------------------
void* numa_alloc_node(size_t bytes, int node) {
  if (bytes == 0) return nullptr;
  void* p = numa_on() ? numa_alloc_onnode(bytes, node) : nullptr;
  if (!p) { p = aligned_alloc(64, (bytes + 63) / 64 * 64); }
  HIVE_CHECK(p != nullptr, "cpu: host allocation failed");
  return p;
}
void numa_free_node(void* p, size_t bytes) {
  if (!p) return;
  if (numa_on()) numa_free(p, bytes); else free(p);
}

// ---- matrices --------------------------------------------------------------------------------------------------------------------------------
size_t Mat::row_bytes() const {
  switch (t) {
    case WType::BF16: return (size_t)K * 2;
    case WType::F32: return (size_t)K * 4;
    case WType::FP8: return (size_t)K;
    case WType::NVFP4: case WType::MXFP4: return (size_t)K / 2;
    case WType::I8: return (size_t)K;
  }
  return 0;
}
size_t Mat::scale_row_bytes() const {
  switch (t) {
    case WType::NVFP4: return (size_t)K / 16;
    case WType::MXFP4: return (size_t)K / 32;
    case WType::FP8: return (size_t)((K + bk - 1) / bk) * 4;   // per block row
    case WType::I8: return (size_t)((K + bk - 1) / bk) * 4;    // per row
    default: return 0;
  }
}

namespace {
float scale_to_f32(const void* s, size_t i, SType t) {
  switch (t) {
    case SType::F32: { float v; std::memcpy(&v, (const uint8_t*)s + i * 4, 4); return v; }
    case SType::BF16: { uint16_t b; std::memcpy(&b, (const uint8_t*)s + i * 2, 2); return bf16_to_f32(b); }
    case SType::E8M0: return e8m0_to_f32(((const uint8_t*)s)[i]);
    case SType::E4M3: return hive::e4m3_to_f32(((const uint8_t*)s)[i]);
  }
  return 0.f;
}
}  // namespace

Mat make_mat(WType t, int N, int K, const void* w, const void* s, SType sdt, float gscale, int bn, int bk) {
  Mat m;
  m.t = t; m.N = N; m.K = K; m.gscale = gscale; m.bn = bn; m.bk = bk;
  HIVE_CHECK(N > 0 && K > 0 && K % 8 == 0, "cpu mat: K must be a positive multiple of 8");
  if (t == WType::NVFP4 || t == WType::MXFP4) HIVE_CHECK(K % 32 == 0, "cpu mat: fp4 K must be a multiple of 32");
  if (t == WType::FP8) HIVE_CHECK(bk % 8 == 0 && bn > 0 && s, "cpu mat: fp8 block shape / scales");
  if (t == WType::NVFP4 || t == WType::MXFP4) HIVE_CHECK(s != nullptr, "cpu mat: fp4 scales missing");
  if (t == WType::I8) HIVE_CHECK(s != nullptr && sdt == SType::F32 && bk % 8 == 0, "cpu mat: i8 needs fp32 group scales, group % 8 == 0");
  Pool& P = Pool::get();
  m.nodes = P.nodes();
  const int gran = t == WType::FP8 ? bn : 16;
  int split = m.nodes > 1 ? (N / 2 + gran - 1) / gran * gran : N;
  split = std::min(split, N);
  m.row0[0] = 0; m.row0[1] = split; m.row0[2] = N;
  if (m.nodes == 1) m.row0[1] = N;
  const size_t rb = m.row_bytes(), sb = m.scale_row_bytes();
  const int nkb = t == WType::FP8 ? (K + bk - 1) / bk : 0;
  // fp8 scales: fp32 table of every block, then each node's block rows
  std::vector<float> fp8s;
  if (t == WType::FP8) {
    const size_t nb = (size_t)((N + bn - 1) / bn) * nkb;
    fp8s.resize(nb);
    for (size_t i = 0; i < nb; ++i) fp8s[i] = scale_to_f32(s, i, sdt);
  }
  for (int k = 0; k < m.nodes; ++k) {
    const int r0 = m.row0[k], r1 = m.row0[k + 1];
    if (r1 <= r0) continue;
    const size_t wbytes = (size_t)(r1 - r0) * rb;
    m.w[k] = (uint8_t*)numa_alloc_node(wbytes, k);
    m.bytes += wbytes;
    size_t sbytes = 0;
    if (t == WType::FP8) sbytes = (size_t)((r1 - r0 + bn - 1) / bn) * sb;
    else if (sb) sbytes = (size_t)(r1 - r0) * sb;
    if (sbytes) { m.s[k] = (uint8_t*)numa_alloc_node(sbytes, k); m.bytes += sbytes; }
  }
  // parallel copy: row chunks, each copied by a worker of the destination node
  const int chunk = 256;
  std::vector<int> nt(2, 0);
  for (int k = 0; k < m.nodes; ++k) nt[k] = (m.row0[k + 1] - m.row0[k] + chunk - 1) / chunk;
  P.run_nodes(nt, [&](int k, int c) {
    const int r0 = m.row0[k] + c * chunk, r1 = std::min(m.row0[k + 1], r0 + chunk);
    std::memcpy(m.w[k] + (size_t)(r0 - m.row0[k]) * rb, (const uint8_t*)w + (size_t)r0 * rb, (size_t)(r1 - r0) * rb);
    if ((t == WType::NVFP4 || t == WType::MXFP4 || t == WType::I8) && sb)
      std::memcpy(m.s[k] + (size_t)(r0 - m.row0[k]) * sb, (const uint8_t*)s + (size_t)r0 * sb, (size_t)(r1 - r0) * sb);
  });
  if (t == WType::FP8)
    for (int k = 0; k < m.nodes; ++k) {
      const int r0 = m.row0[k], r1 = m.row0[k + 1];
      if (r1 <= r0) continue;
      const int b0 = r0 / bn, b1 = (r1 + bn - 1) / bn;
      std::memcpy(m.s[k], fp8s.data() + (size_t)b0 * nkb, (size_t)(b1 - b0) * nkb * 4);
    }
  return m;
}

Mat make_mat_i8(int N, int K, const uint16_t* w, int group) {
  HIVE_CHECK(group > 0 && group % 8 == 0 && K % 8 == 0, "cpu mat: i8 group");
  const int ng = (K + group - 1) / group;
  std::vector<int8_t> q((size_t)N * K);
  std::vector<float> sc((size_t)N * ng);
  Pool::get().run((N + 63) / 64, [&](int c) {
    for (int r = c * 64; r < std::min(N, c * 64 + 64); ++r)
      for (int g = 0; g < ng; ++g) {
        const int k0 = g * group, k1 = std::min(K, k0 + group);
        float amax = 0.f;
        for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(bf16_to_f32(w[(size_t)r * K + k])));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        sc[(size_t)r * ng + g] = d;
        for (int k = k0; k < k1; ++k) q[(size_t)r * K + k] = (int8_t)std::lrint(std::max(-127.f, std::min(127.f, bf16_to_f32(w[(size_t)r * K + k]) * id)));
      }
  });
  return make_mat(WType::I8, N, K, q.data(), sc.data(), SType::F32, 1.f, 128, group);
}

void free_mat(Mat& m) {
  for (int k = 0; k < 2; ++k) {
    const int r0 = m.row0[k], r1 = m.row0[k + 1];
    const size_t n = r1 > r0 ? (size_t)(r1 - r0) : 0;
    if (m.w[k]) numa_free_node(m.w[k], n * m.row_bytes());
    if (m.s[k]) numa_free_node(m.s[k], m.t == WType::FP8 ? (size_t)((n + m.bn - 1) / m.bn) * m.scale_row_bytes() : n * m.scale_row_bytes());
    m.w[k] = m.s[k] = nullptr;
  }
  m.bytes = 0;
}

// ---- helpers -----------------------------------------------------------------------------------------------------------------------------
float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
float e4m3_to_f32(uint8_t v) { return hive::e4m3_to_f32(v); }

void bf16_to_f32_n(const uint16_t* src, size_t n, float* dst) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i v = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(src + i))), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(v));
  }
  for (; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
}

void round_bf16(float* x, size_t n) {
  auto body = [x](size_t a, size_t b) { for (size_t i = a; i < b; ++i) x[i] = bf16r(x[i]); };
  if (n < (1u << 16)) { body(0, n); return; }
  const int tasks = (int)std::min<size_t>(256, n / 8192);
  Pool::get().run(tasks, [&](int t) { body(n * t / tasks, n * (t + 1) / tasks); });
}

}  // namespace hive::cpu
