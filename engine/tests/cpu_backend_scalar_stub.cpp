// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// TEST-ONLY stand-in for engine/src/cpu/cpu_backend.cpp (hive/cpu/cpu_backend.h), used to build qwen_cpu_check while the real backend is
//   written: plain std::thread pool (no NUMA placement, one node), row-parallel AVX2 dot products for every weight type, a direct attention
//   loop. Same results contract (fp32 accumulation, fixed order per output element); NOT the performance of the real backend.
#include "hive/cpu/cpu_backend.h"

#include <immintrin.h>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace hive::cpu {

// ---- pool ----------------------------------------------------------------------------------------------------------------------------
struct Pool::Impl {
  std::vector<std::thread> th;
  std::mutex mu; std::condition_variable cv, done;
  const std::function<void(int)>* f = nullptr;
  int n = 0; std::atomic<int> next{0}; int pending = 0; uint64_t gen = 0; bool stop = false;
};
Pool& Pool::get() { static Pool p; return p; }
Pool::Pool() : impl_(new Impl) {
  const char* v = getenv("HIVE_CPU_POOL_THREADS");
  n_ = v && atoi(v) > 0 ? atoi(v) : std::max(1u, std::thread::hardware_concurrency() / 2);
  nodes_ = 1; per_node_ = {n_};
  Impl* I = impl_.get();
  for (int t = 0; t < n_ - 1; ++t)
    I->th.emplace_back([I] {
      uint64_t seen = 0;
      for (;;) {
        std::unique_lock<std::mutex> lk(I->mu);
        I->cv.wait(lk, [&] { return I->stop || I->gen != seen; });
        if (I->stop) return;
        seen = I->gen;
        const auto* f = I->f; const int n = I->n;
        lk.unlock();
        for (int i; (i = I->next.fetch_add(1)) < n;) (*f)(i);
        lk.lock();
        if (--I->pending == 0) I->done.notify_all();
      }
    });
}
Pool::~Pool() { { std::lock_guard<std::mutex> lk(impl_->mu); impl_->stop = true; } impl_->cv.notify_all(); for (auto& t : impl_->th) t.join(); }
void Pool::run(int n, const std::function<void(int)>& f) {
  if (n <= 0) return;
  if (n == 1 || n_ == 1) { for (int i = 0; i < n; ++i) f(i); return; }
  Impl* I = impl_.get();
  { std::lock_guard<std::mutex> lk(I->mu); I->f = &f; I->n = n; I->next = 0; I->pending = (int)I->th.size(); ++I->gen; }
  I->cv.notify_all();
  for (int i; (i = I->next.fetch_add(1)) < n;) f(i);
  std::unique_lock<std::mutex> lk(I->mu);
  I->done.wait(lk, [&] { return I->pending == 0; });
}
void Pool::run_nodes(const std::vector<int>& n_per_node, const std::function<void(int, int)>& f) {
  std::vector<std::pair<int, int>> t;
  for (size_t k = 0; k < n_per_node.size(); ++k) for (int i = 0; i < n_per_node[k]; ++i) t.push_back({(int)k, i});
  run((int)t.size(), [&](int i) { f(t[i].first, t[i].second); });
}

void* numa_alloc_node(size_t bytes, int) { void* p = aligned_alloc(64, (bytes + 63) / 64 * 64); if (!p) throw std::bad_alloc(); return p; }
void numa_free_node(void* p, size_t) { free(p); }

float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
void bf16_to_f32_n(const uint16_t* s, size_t n, float* d) { for (size_t i = 0; i < n; ++i) d[i] = bf16_to_f32(s[i]); }
float e4m3_to_f32(uint8_t v) {
  const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
  float r;
  if (e == 0) r = std::ldexp((float)m, -9);
  else if (e == 15 && m == 7) r = NAN;
  else r = std::ldexp(1.f + m / 8.f, e - 7);
  return s ? -r : r;
}
void round_bf16(float* x, size_t n) { for (size_t i = 0; i < n; ++i) x[i] = bf16r(x[i]); }

// ---- matrices ------------------------------------------------------------------------------------------------------------------------
size_t Mat::row_bytes() const {
  switch (t) {
    case WType::BF16: return (size_t)K * 2;
    case WType::F32: return (size_t)K * 4;
    case WType::FP8: return (size_t)K;
    case WType::NVFP4: case WType::MXFP4: return (size_t)K / 2;
  }
  return 0;
}
size_t Mat::scale_row_bytes() const {
  switch (t) {
    case WType::NVFP4: return (size_t)K / 16;
    case WType::MXFP4: return (size_t)K / 32;
    case WType::FP8: return (size_t)((K + bk - 1) / bk) * 4;
    default: return 0;
  }
}
static float e8m0(uint8_t v) { return std::ldexp(1.f, (int)v - 127); }
Mat make_mat(WType t, int N, int K, const void* w, const void* s, SType sdt, float gscale, int bn, int bk) {
  Mat m; m.t = t; m.N = N; m.K = K; m.gscale = gscale; m.bn = bn; m.bk = bk; m.nodes = 1; m.row0[0] = 0; m.row0[1] = N; m.row0[2] = N;
  const size_t wb = m.row_bytes() * N;
  m.w[0] = (uint8_t*)numa_alloc_node(wb, 0); memcpy(m.w[0], w, wb); m.bytes = wb;
  if (t == WType::NVFP4 || t == WType::MXFP4) {
    const size_t sb = m.scale_row_bytes() * N;
    m.s[0] = (uint8_t*)numa_alloc_node(sb, 0); memcpy(m.s[0], s, sb); m.bytes += sb;
  } else if (t == WType::FP8) {
    const size_t nb = (N + bn - 1) / bn, kb = (K + bk - 1) / bk;
    float* f = (float*)numa_alloc_node(nb * kb * 4, 0);
    for (size_t i = 0; i < nb * kb; ++i) {
      switch (sdt) {
        case SType::F32: f[i] = ((const float*)s)[i]; break;
        case SType::BF16: f[i] = bf16_to_f32(((const uint16_t*)s)[i]); break;
        case SType::E8M0: f[i] = e8m0(((const uint8_t*)s)[i]); break;
        case SType::E4M3: f[i] = e4m3_to_f32(((const uint8_t*)s)[i]); break;
      }
    }
    m.s[0] = (uint8_t*)f; m.bytes += nb * kb * 4;
  }
  return m;
}
void free_mat(Mat& m) { for (int k = 0; k < 2; ++k) { if (m.w[k]) numa_free_node(m.w[k], 0); if (m.s[k]) numa_free_node(m.s[k], 0); m.w[k] = m.s[k] = nullptr; } }

static const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
static float g_e4m3[256];
static const bool g_e4m3_ready = [] { for (int i = 0; i < 256; ++i) g_e4m3[i] = e4m3_to_f32((uint8_t)i); return true; }();

static inline float hsum(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_hadd_ps(a, a); a = _mm_hadd_ps(a, a); return _mm_cvtss_f32(a);
}
// one weight row (row index r of m) → fp32 [K] (scales applied)
static void row_f32(const Mat& m, int r, float* out) {
  const uint8_t* w = m.w[0] + (size_t)r * m.row_bytes();
  switch (m.t) {
    case WType::BF16: { const uint16_t* p = (const uint16_t*)w;
      int k = 0;
      for (; k + 8 <= m.K; k += 8) {
        __m256i x = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(p + k)));
        _mm256_storeu_ps(out + k, _mm256_castsi256_ps(_mm256_slli_epi32(x, 16)));
      }
      for (; k < m.K; ++k) out[k] = bf16_to_f32(p[k]);
      break; }
    case WType::F32: memcpy(out, w, (size_t)m.K * 4); break;
    case WType::FP8: { const float* st = (const float*)m.s[0] + (size_t)(r / m.bn) * ((m.K + m.bk - 1) / m.bk);
      for (int k = 0; k < m.K; ++k) out[k] = g_e4m3[w[k]] * st[k / m.bk];
      break; }
    case WType::NVFP4: { const uint8_t* s = m.s[0] + (size_t)r * m.scale_row_bytes();
      for (int k = 0; k < m.K; k += 2) { const float sc = g_e4m3[s[k / 16]] * m.gscale; out[k] = kE2M1[w[k / 2] & 15] * sc; out[k + 1] = kE2M1[w[k / 2] >> 4] * sc; }
      break; }
    case WType::MXFP4: { const uint8_t* s = m.s[0] + (size_t)r * m.scale_row_bytes();
      for (int k = 0; k < m.K; k += 2) { const float sc = e8m0(s[k / 32]); out[k] = kE2M1[w[k / 2] & 15] * sc; out[k + 1] = kE2M1[w[k / 2] >> 4] * sc; }
      break; }
  }
}
static inline float dot(const float* a, const float* b, int K) {
  __m256 acc = _mm256_setzero_ps();
  int k = 0;
  for (; k + 8 <= K; k += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
  float s = hsum(acc);
  for (; k < K; ++k) s += a[k] * b[k];
  return s;
}

void matmul_multi(const MatJob* jobs, int n_jobs, const float* X, int M, int ldx) {
  // tasks = (job, block of 16 rows)
  std::vector<std::pair<int, int>> tasks;
  for (int j = 0; j < n_jobs; ++j) for (int r = jobs[j].W.r0; r < jobs[j].W.r1; r += 16) tasks.push_back({j, r});
  Pool::get().run((int)tasks.size(), [&](int i) {
    const MatJob& J = jobs[tasks[i].first];
    const Mat& m = *J.W.m;
    thread_local std::vector<float> row;
    row.resize(m.K);
    const int r1 = std::min(J.W.r1, tasks[i].second + 16);
    for (int r = tasks[i].second; r < r1; ++r) {
      row_f32(m, r, row.data());
      for (int mm = 0; mm < M; ++mm) J.Y[(size_t)mm * J.ldy + (r - J.W.r0)] = dot(row.data(), X + (size_t)mm * ldx, m.K);
    }
  });
}
void matmul(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy) {
  const MatJob j{W, Y, ldy};
  matmul_multi(&j, 1, X, M, ldx);
}

void attention(const float* q, int M, int Hq, int Hkv, int D, const uint16_t* K, const uint16_t* V, const int32_t* idx, int idx_stride,
               const int32_t* causal_end, float scale, float* out) {
  const int rep = Hq / Hkv;
  Pool::get().run(M * Hq, [&](int t) {
    const int m = t / Hq, h = t % Hq, g = h / rep;
    const float* qv = q + ((size_t)m * Hq + h) * D;
    thread_local std::vector<float> sc, kv;
    kv.resize(D);
    int n = idx ? idx_stride : causal_end[m];
    sc.assign(n, -INFINITY);
    float mx = -INFINITY;
    for (int j = 0; j < n; ++j) {
      const int p = idx ? idx[(size_t)m * idx_stride + j] : j;
      if (p < 0) continue;
      bf16_to_f32_n(K + ((size_t)p * Hkv + g) * D, D, kv.data());
      sc[j] = dot(qv, kv.data(), D) * scale;
      mx = std::max(mx, sc[j]);
    }
    float* o = out + ((size_t)m * Hq + h) * D;
    for (int d = 0; d < D; ++d) o[d] = 0.f;
    if (mx == -INFINITY) return;
    float sum = 0.f;
    for (int j = 0; j < n; ++j) if (sc[j] != -INFINITY) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
    for (int j = 0; j < n; ++j) {
      if (sc[j] == -INFINITY) continue;
      const int p = idx ? idx[(size_t)m * idx_stride + j] : j;
      const float pr = sc[j] / sum;
      const uint16_t* vr = V + ((size_t)p * Hkv + g) * D;
      for (int d = 0; d < D; ++d) o[d] += pr * bf16_to_f32(vr[d]);
    }
  });
}

}  // namespace hive::cpu
