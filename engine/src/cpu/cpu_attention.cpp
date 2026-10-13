// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU attention (GQA, bf16 K/V cache) — see hive/cpu/cpu_backend.h.
//   One task per (query row, kv head): the query heads of that kv head share every K / V row, which is converted to fp32 once per tile
//   position. Positions are processed in tiles of 32: scores of the tile for every head, one rescale of the running output per tile
//   (online softmax), then P·V. fp32 throughout; the result depends only on the position order, not on the thread count.
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "hive/common.h"
#include "hive/cpu/cpu_backend.h"

namespace hive::cpu {

namespace {
inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}
inline void bf16_row(const uint16_t* src, int D, float* dst) {
  int d = 0;
  for (; d + 8 <= D; d += 8)
    _mm256_storeu_ps(dst + d, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(src + d))), 16)));
  for (; d < D; ++d) dst[d] = bf16_to_f32(src[d]);
}
inline float dot(const float* a, const float* b, int D) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  int d = 0;
  for (; d + 16 <= D; d += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 8), _mm256_loadu_ps(b + d + 8), s1);
  }
  float s = hsum(_mm256_add_ps(s0, s1));
  for (; d < D; ++d) s += a[d] * b[d];
  return s;
}
inline void axpy(float a, const float* x, float* y, int D) {
  const __m256 av = _mm256_set1_ps(a);
  int d = 0;
  for (; d + 8 <= D; d += 8) _mm256_storeu_ps(y + d, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + d), _mm256_loadu_ps(y + d)));
  for (; d < D; ++d) y[d] += a * x[d];
}
inline void scale_row(float a, float* y, int D) {
  const __m256 av = _mm256_set1_ps(a);
  int d = 0;
  for (; d + 8 <= D; d += 8) _mm256_storeu_ps(y + d, _mm256_mul_ps(av, _mm256_loadu_ps(y + d)));
  for (; d < D; ++d) y[d] *= a;
}
constexpr int kTile = 32;
}  // namespace

void attention(const float* q, int M, int Hq, int Hkv, int D, const uint16_t* K, const uint16_t* V, const int32_t* idx, int idx_stride,
               const int32_t* causal_end, float scale, float* out) {
  if (M <= 0) return;
  HIVE_CHECK(Hkv > 0 && Hq % Hkv == 0 && (idx || causal_end), "cpu attention: shape / positions");
  const int G = Hq / Hkv;
  const size_t row = (size_t)Hkv * D;
  Pool::get().run(M * Hkv, [&](int task) {
    const int m = task / Hkv, g = task % Hkv;
    thread_local std::vector<float> buf;
    buf.resize((size_t)kTile * D * 2 + (size_t)G * kTile + 3 * (size_t)G);
    float* kt = buf.data();
    float* vt = kt + (size_t)kTile * D;
    float* sc = vt + (size_t)kTile * D;
    float* mx = sc + (size_t)G * kTile;
    float* lsum = mx + G;
    const float* qh = q + ((size_t)m * Hq + (size_t)g * G) * D;
    float* oh = out + ((size_t)m * Hq + (size_t)g * G) * D;
    for (int h = 0; h < G; ++h) { mx[h] = -INFINITY; lsum[h] = 0.f; std::memset(oh + (size_t)h * D, 0, (size_t)D * 4); }
    const int n = idx ? idx_stride : causal_end[m];
    const int32_t* ir = idx ? idx + (size_t)m * idx_stride : nullptr;
    int pos[kTile];
    for (int t0 = 0; t0 < n;) {
      int c = 0;
      while (c < kTile && t0 < n) {
        const int p = ir ? ir[t0] : t0;
        ++t0;
        if (p < 0) continue;
        pos[c] = p;
        bf16_row(K + (size_t)p * row + (size_t)g * D, D, kt + (size_t)c * D);
        bf16_row(V + (size_t)p * row + (size_t)g * D, D, vt + (size_t)c * D);
        ++c;
      }
      if (c == 0) continue;
      for (int h = 0; h < G; ++h) {
        float tm = -INFINITY;
        for (int j = 0; j < c; ++j) { sc[h * kTile + j] = dot(qh + (size_t)h * D, kt + (size_t)j * D, D) * scale; tm = std::max(tm, sc[h * kTile + j]); }
        const float nm = std::max(mx[h], tm);
        const float corr = mx[h] == -INFINITY ? 0.f : expf(mx[h] - nm);
        float* o = oh + (size_t)h * D;
        if (corr != 1.f) scale_row(corr, o, D);
        float s = lsum[h] * corr;
        for (int j = 0; j < c; ++j) {
          const float p = expf(sc[h * kTile + j] - nm);
          s += p;
          axpy(p, vt + (size_t)j * D, o, D);
        }
        lsum[h] = s; mx[h] = nm;
      }
      (void)pos;
    }
    for (int h = 0; h < G; ++h) scale_row(lsum[h] > 0.f ? 1.f / lsum[h] : 0.f, oh + (size_t)h * D, D);
  });
}

}  // namespace hive::cpu
