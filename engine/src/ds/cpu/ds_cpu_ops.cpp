// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash CPU engine — element-wise numerics, RoPE, Sinkhorn and the attention / indexer row kernels (ds_cpu_internal.h).
//   Each function mirrors one operation of oracle/dsv41_oracle.py (the definitions of the reference kernel.py / model.py); vector bodies and
//   scalar tails perform the same IEEE operations, so the result does not depend on the row length or on the thread count.
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ds_cpu_internal.h"
#include "hive/common.h"

namespace hive::ds::cpu {

using ::hive::cpu::Pool;

void par_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f) {
  if (n == 0) return;
  const size_t nb = (n + grain - 1) / grain;
  if (nb <= 1) { f(0, n); return; }
  Pool::get().run((int)nb, [&](int b) { const size_t a = (size_t)b * grain; f(a, std::min(n, a + grain)); });
}

namespace {
inline __m256 bf16r8(__m256 x) {
  const __m256i u = _mm256_castps_si256(x);
  const __m256i ex = _mm256_set1_epi32(0x7f800000);
  const __m256i special = _mm256_cmpeq_epi32(_mm256_and_si256(u, ex), ex);
  const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
  const __m256i r = _mm256_and_si256(_mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), lsb), _mm256_set1_epi32((int)0xffff0000u));
  return _mm256_castsi256_ps(_mm256_blendv_epi8(r, u, special));
}
inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}
// exp: Cody–Waite reduction + degree-5 polynomial (relative error ~1e-7, below the bf16 rounding of P and comparable to fp32 rounding of the
//   denominator). Scalar and vector forms are the same operations.
constexpr float kLog2e = 1.44269504088896341f, kLn2Hi = 0.693359375f, kLn2Lo = -2.12194440e-4f;
constexpr float kP0 = 1.9875691500e-4f, kP1 = 1.3981999507e-3f, kP2 = 8.3334519073e-3f, kP3 = 4.1665795894e-2f, kP4 = 1.6666665459e-1f,
                kP5 = 5.0000001201e-1f;
inline __m256 exp8(__m256 x) {
  const __m256 lo = _mm256_cmp_ps(x, _mm256_set1_ps(-87.f), _CMP_LT_OQ);
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.f)), _mm256_set1_ps(88.f));
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kLog2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fmadd_ps(n, _mm256_set1_ps(-kLn2Hi), x);
  r = _mm256_fmadd_ps(n, _mm256_set1_ps(-kLn2Lo), r);
  __m256 p = _mm256_set1_ps(kP0);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP1)); p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP2)); p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP3));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP4)); p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP5));
  __m256 y = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.f));
  const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_andnot_ps(lo, _mm256_mul_ps(y, _mm256_castsi256_ps(e)));   // below −87: 0 (−inf scores)
}
inline float exp1(float x) {
  alignas(32) float t[8] = {x, x, x, x, x, x, x, x};
  _mm256_store_ps(t, exp8(_mm256_load_ps(t)));
  return t[0];
}
// 2^ceil(log2(x)) (reference fast_round_scale via frexp: unchanged for powers of two)
inline float pow2ceil(float x) {
  int e;
  const float m = std::frexp(x, &e);
  if (m == 0.5f) e -= 1;
  return std::ldexp(1.f, e);
}
inline __m256 e4m3_round8(__m256 v) {
  const __m256 sign = _mm256_and_ps(v, _mm256_set1_ps(-0.f));
  const __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.f), v);
  const __m256 sub = _mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(a, _mm256_set1_ps(512.f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
                                   _mm256_set1_ps(1.f / 512.f));
  const __m256i u = _mm256_castps_si256(a);
  const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 20), _mm256_set1_epi32(1));
  const __m256i r = _mm256_and_si256(_mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7ffff)), lsb), _mm256_set1_epi32((int)0xfff00000u));
  const __m256 isub = _mm256_cmp_ps(a, _mm256_set1_ps(0.015625f), _CMP_LT_OQ);
  return _mm256_or_ps(_mm256_blendv_ps(_mm256_castsi256_ps(r), sub, isub), sign);
}
}  // namespace

float e4m3_round(float v) {
  const float a = std::fabs(v);
  float r;
  if (a < 0.015625f) r = std::nearbyint(a * 512.f) * (1.f / 512.f);
  else {
    uint32_t u; std::memcpy(&u, &a, 4);
    u = (u + 0x7ffffu + ((u >> 20) & 1u)) & 0xfff00000u;
    std::memcpy(&r, &u, 4);
  }
  return std::copysign(r, v);
}

void round_rows(float* y, int M, int N, int ld) {
  par_for((size_t)M, std::max<size_t>(1, 262144 / std::max(1, N)), [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      float* r = y + m * ld;
      int i = 0;
      for (; i + 8 <= N; i += 8) _mm256_storeu_ps(r + i, bf16r8(_mm256_loadu_ps(r + i)));
      for (; i < N; ++i) r[i] = bf16r(r[i]);
    }
  });
}

void act_quant_fp8(const float* x, int M, int K, int ldx, float* out, int ldo) {
  const float inv448 = (float)(1.0 / 448.0);
  par_for((size_t)M, std::max<size_t>(1, 65536 / std::max(1, K)), [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const float* xr = x + m * ldx;
      float* o = out + m * ldo;
      for (int k0 = 0; k0 < K; k0 += 32) {
        __m256 mx = _mm256_setzero_ps();
        for (int i = 0; i < 32; i += 8) mx = _mm256_max_ps(mx, _mm256_andnot_ps(_mm256_set1_ps(-0.f), _mm256_loadu_ps(xr + k0 + i)));
        alignas(32) float t[8];
        _mm256_store_ps(t, mx);
        float amax = t[0];
        for (int i = 1; i < 8; ++i) amax = std::max(amax, t[i]);
        amax = std::max(amax, 1e-4f);
        const float s = pow2ceil(amax * inv448);
        const __m256 sv = _mm256_set1_ps(s), lim = _mm256_set1_ps(448.f), nlim = _mm256_set1_ps(-448.f);
        for (int i = 0; i < 32; i += 8) {
          __m256 v = _mm256_div_ps(_mm256_loadu_ps(xr + k0 + i), sv);
          v = _mm256_min_ps(_mm256_max_ps(v, nlim), lim);
          _mm256_storeu_ps(o + k0 + i, _mm256_mul_ps(e4m3_round8(v), sv));
        }
      }
    }
  });
}

void fp4_quant_row(float* x, int K, int block, bool e4m3_scale) {
  static const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float inv6 = (float)(1.0 / 6.0);
  for (int k0 = 0; k0 < K; k0 += block) {
    float amax = 0.f;
    for (int i = 0; i < block; ++i) amax = std::max(amax, std::fabs(x[k0 + i]));
    float s;
    if (e4m3_scale) {
      amax = std::max(amax, 6.f * std::ldexp(1.f, -9));
      s = ::hive::e4m3_to_f32(::hive::f32_to_e4m3_host(amax / 6.0f));
    } else {
      amax = std::max(amax, (float)(6.0 * std::ldexp(1.0, -126)));
      s = pow2ceil(amax * inv6);
    }
    for (int i = 0; i < block; ++i) {
      float v = x[k0 + i] / s;
      v = std::min(std::max(v, -6.f), 6.f);
      const uint8_t c = ::hive::f32_to_e2m1(v);
      const float g = grid[c & 7];
      x[k0 + i] = bf16r((v < 0 ? -g : (v > 0 ? g : 0.f)) * s);
    }
  }
}

void rmsnorm_rows(const float* x, int ldx, const float* w, int M, int K, float eps, float* out, int ldo) {
  par_for((size_t)M, std::max<size_t>(1, 65536 / std::max(1, K)), [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const float* xr = x + m * ldx;
      float* o = out + m * ldo;
      __m256 acc = _mm256_setzero_ps();
      int i = 0;
      for (; i + 8 <= K; i += 8) { const __m256 v = _mm256_loadu_ps(xr + i); acc = _mm256_fmadd_ps(v, v, acc); }
      float ss = hsum(acc);
      for (; i < K; ++i) ss = std::fma(xr[i], xr[i], ss);
      const float r = 1.f / std::sqrt(ss / (float)K + eps);
      const __m256 rv = _mm256_set1_ps(r);
      i = 0;
      for (; i + 8 <= K; i += 8)
        _mm256_storeu_ps(o + i, bf16r8(_mm256_mul_ps(_mm256_loadu_ps(w + i), _mm256_mul_ps(_mm256_loadu_ps(xr + i), rv))));
      for (; i < K; ++i) o[i] = bf16r(w[i] * (xr[i] * r));
    }
  });
}

void Rope::init(int rd, int64_t orig, float base, float factor, float beta_fast, float beta_slow) {
  freq.resize(rd / 2);
  for (int i = 0; i < rd / 2; ++i) freq[i] = 1.0f / std::pow(base, (float)(2 * i) / (float)rd);
  if (orig > 0) {
    auto corrected = [&](double rot) { return rd * std::log((double)orig / (rot * 2 * M_PI)) / (2 * std::log((double)base)); };
    const double low = std::max(std::floor(corrected(beta_fast)), 0.0), high = std::min(std::ceil(corrected(beta_slow)), (double)rd - 1);
    const float den = (float)std::max(high - low, 1e-3);
    for (int i = 0; i < rd / 2; ++i) {
      const float ramp = std::min(std::max(((float)i - (float)low) / den, 0.f), 1.f);
      const float smooth = 1.f - ramp;
      freq[i] = freq[i] / factor * (1.f - smooth) + freq[i] * smooth;
    }
  }
}

void Rope::apply(float* x, int heads, int hd, int rd, int64_t pos, bool inverse) const {
  float cs[64], sn[64];
  const int np = rd / 2;
  for (int i = 0; i < np; ++i) {
    const float ang = (float)pos * freq[i];
    cs[i] = (float)std::cos((double)ang);
    sn[i] = (float)std::sin((double)ang);
    if (inverse) sn[i] = -sn[i];
  }
  for (int h = 0; h < heads; ++h) {
    float* v = x + (size_t)h * hd + (hd - rd);
    for (int i = 0; i < np; ++i) {
      const float a = v[2 * i], b = v[2 * i + 1];
      v[2 * i] = bf16r(a * cs[i] - b * sn[i]);
      v[2 * i + 1] = bf16r(a * sn[i] + b * cs[i]);
    }
  }
}

void sinkhorn_row(const float* mixes, const float* scale, const float* base, int hc, int iters, float eps, float* pre, float* post, float* comb) {
  auto sig = [](float x) { return 1.f / (1.f + std::exp(-x)); };
  for (int i = 0; i < hc; ++i) {
    pre[i] = sig(mixes[i] * scale[0] + base[i]) + eps;
    post[i] = 2.f * sig(mixes[hc + i] * scale[1] + base[hc + i]);
  }
  float c[64];
  for (int j = 0; j < hc; ++j) {
    float mx = -INFINITY;
    for (int i = 0; i < hc; ++i) { c[j * hc + i] = mixes[2 * hc + j * hc + i] * scale[2] + base[2 * hc + j * hc + i]; mx = std::max(mx, c[j * hc + i]); }
    float s = 0.f;
    for (int i = 0; i < hc; ++i) { c[j * hc + i] = std::exp(c[j * hc + i] - mx); s += c[j * hc + i]; }
    for (int i = 0; i < hc; ++i) c[j * hc + i] = c[j * hc + i] / s + eps;
  }
  auto col_norm = [&]() {   // comb / (comb.sum(-2) + eps): over j for each i
    for (int i = 0; i < hc; ++i) {
      float s = 0.f;
      for (int j = 0; j < hc; ++j) s += c[j * hc + i];
      s += eps;
      for (int j = 0; j < hc; ++j) c[j * hc + i] /= s;
    }
  };
  auto row_norm = [&]() {   // comb / (comb.sum(-1) + eps): over i for each j
    for (int j = 0; j < hc; ++j) {
      float s = 0.f;
      for (int i = 0; i < hc; ++i) s += c[j * hc + i];
      s += eps;
      for (int i = 0; i < hc; ++i) c[j * hc + i] /= s;
    }
  };
  col_norm();
  for (int t = 0; t < iters - 1; ++t) { row_norm(); col_norm(); }
  for (int i = 0; i < hc * hc; ++i) comb[i] = c[i];
}

// ---- sparse attention of one query row ------------------------------------------------------------------------------------------------
void sparse_attn_row(const float* q, int H, int D, const float* K, int n, const float* sink, float scale, float* o, float* ws) {
  float* S = ws;  // [H, n]
  const int D8 = D / 8;
  // scores: 2 heads × 4 keys per register block, the 4 key rows stay in L1 while every head pair passes
  int j = 0;
  for (; j + 4 <= n; j += 4) {
    const float* k0 = K + (size_t)j * D;
    for (int h = 0; h < H; h += 2) {
      const float* q0 = q + (size_t)h * D;
      const float* q1 = q0 + D;
      __m256 a[2][4];
      for (int x = 0; x < 2; ++x) for (int y = 0; y < 4; ++y) a[x][y] = _mm256_setzero_ps();
      for (int d = 0; d < D8; ++d) {
        const __m256 qa = _mm256_loadu_ps(q0 + 8 * d), qb = _mm256_loadu_ps(q1 + 8 * d);
        for (int y = 0; y < 4; ++y) {
          const __m256 kv = _mm256_loadu_ps(k0 + (size_t)y * D + 8 * d);
          a[0][y] = _mm256_fmadd_ps(qa, kv, a[0][y]);
          a[1][y] = _mm256_fmadd_ps(qb, kv, a[1][y]);
        }
      }
      for (int y = 0; y < 4; ++y) { S[(size_t)h * n + j + y] = hsum(a[0][y]) * scale; S[(size_t)(h + 1) * n + j + y] = hsum(a[1][y]) * scale; }
    }
  }
  for (; j < n; ++j) {
    const float* kr = K + (size_t)j * D;
    for (int h = 0; h < H; ++h) {
      const float* qh = q + (size_t)h * D;
      __m256 a = _mm256_setzero_ps();
      for (int d = 0; d < D8; ++d) a = _mm256_fmadd_ps(_mm256_loadu_ps(qh + 8 * d), _mm256_loadu_ps(kr + 8 * d), a);
      S[(size_t)h * n + j] = hsum(a) * scale;
    }
  }
  // softmax with sink: p = exp(s − max) (fp32 sum), P rounded to bf16 for P·V
  float den[128];
  for (int h = 0; h < H; ++h) {
    float* s = S + (size_t)h * n;
    float mx = -INFINITY;
    for (int t = 0; t < n; ++t) mx = std::max(mx, s[t]);
    mx = std::max(mx, -1e30f);
    const __m256 mv = _mm256_set1_ps(mx);
    __m256 acc = _mm256_setzero_ps();
    int t = 0;
    for (; t + 8 <= n; t += 8) {
      const __m256 p = exp8(_mm256_sub_ps(_mm256_loadu_ps(s + t), mv));
      acc = _mm256_add_ps(acc, p);
      _mm256_storeu_ps(s + t, bf16r8(p));
    }
    float sum = hsum(acc);
    for (; t < n; ++t) { const float p = exp1(s[t] - mx); sum += p; s[t] = bf16r(p); }
    den[h] = sum + exp1(sink[h] - mx);
  }
  // P·V: 4 heads × 16 dims per register block
  for (int d0 = 0; d0 < D; d0 += 16) {
    for (int h = 0; h < H; h += 4) {
      __m256 a[4][2];
      for (int x = 0; x < 4; ++x) { a[x][0] = _mm256_setzero_ps(); a[x][1] = _mm256_setzero_ps(); }
      const float* p0 = S + (size_t)h * n;
      for (int t = 0; t < n; ++t) {
        const float* kr = K + (size_t)t * D + d0;
        const __m256 k0v = _mm256_loadu_ps(kr), k1v = _mm256_loadu_ps(kr + 8);
        for (int x = 0; x < 4; ++x) {
          const __m256 pv = _mm256_broadcast_ss(p0 + (size_t)x * n + t);
          a[x][0] = _mm256_fmadd_ps(pv, k0v, a[x][0]);
          a[x][1] = _mm256_fmadd_ps(pv, k1v, a[x][1]);
        }
      }
      for (int x = 0; x < 4; ++x) {
        const __m256 dv = _mm256_set1_ps(den[h + x]);
        float* oh = o + (size_t)(h + x) * D + d0;
        _mm256_storeu_ps(oh, bf16r8(_mm256_div_ps(a[x][0], dv)));
        _mm256_storeu_ps(oh + 8, bf16r8(_mm256_div_ps(a[x][1], dv)));
      }
    }
  }
}

// ---- indexer scores of one query row ---------------------------------------------------------------------------------------------------
void index_scores_row(const float* q, const float* w, int Hi, int Di, const float* keys, const int32_t* key_idx, int n, float* score) {
  const int D8 = Di / 8;
  int j = 0;
  auto key = [&](int t) { return keys + (size_t)(key_idx ? key_idx[t] : t) * Di; };
  for (; j + 4 <= n; j += 4) {
    const float* kp[4] = {key(j), key(j + 1), key(j + 2), key(j + 3)};
    float tot[4] = {0, 0, 0, 0};
    for (int h = 0; h < Hi; h += 2) {
      const float* q0 = q + (size_t)h * Di;
      const float* q1 = q0 + Di;
      __m256 a[2][4];
      for (int x = 0; x < 2; ++x) for (int y = 0; y < 4; ++y) a[x][y] = _mm256_setzero_ps();
      for (int d = 0; d < D8; ++d) {
        const __m256 qa = _mm256_loadu_ps(q0 + 8 * d), qb = _mm256_loadu_ps(q1 + 8 * d);
        for (int y = 0; y < 4; ++y) {
          const __m256 kv = _mm256_loadu_ps(kp[y] + 8 * d);
          a[0][y] = _mm256_fmadd_ps(qa, kv, a[0][y]);
          a[1][y] = _mm256_fmadd_ps(qb, kv, a[1][y]);
        }
      }
      for (int y = 0; y < 4; ++y) {
        tot[y] += bf16r(std::max(bf16r(hsum(a[0][y])), 0.f) * w[h]);
        tot[y] += bf16r(std::max(bf16r(hsum(a[1][y])), 0.f) * w[h + 1]);
      }
    }
    for (int y = 0; y < 4; ++y) score[j + y] = bf16r(tot[y]);
  }
  for (; j < n; ++j) {
    const float* kr = key(j);
    float tot = 0.f;
    for (int h = 0; h < Hi; ++h) {
      __m256 a = _mm256_setzero_ps();
      for (int d = 0; d < D8; ++d) a = _mm256_fmadd_ps(_mm256_loadu_ps(q + (size_t)h * Di + 8 * d), _mm256_loadu_ps(kr + 8 * d), a);
      tot += bf16r(std::max(bf16r(hsum(a)), 0.f) * w[h]);
    }
    score[j] = bf16r(tot);
  }
}

int topk_sorted(const float* score, const int32_t* pos, int n, int k, int32_t* sel, std::vector<int32_t>& scratch) {
  scratch.clear();
  for (int i = 0; i < n; ++i) if (score[i] > -INFINITY) scratch.push_back(i);   // −inf = masked (the pinned last block is +inf)
  const int m = (int)scratch.size();
  const int kk = std::min(k, m);
  auto P = [&](int i) { return pos ? pos[i] : i; };
  if (kk < m)
    std::nth_element(scratch.begin(), scratch.begin() + kk, scratch.end(),
                     [&](int a, int b) { return score[a] > score[b] || (score[a] == score[b] && P(a) < P(b)); });
  for (int i = 0; i < kk; ++i) sel[i] = P(scratch[i]);
  std::sort(sel, sel + kk);
  return kk;
}

}  // namespace hive::ds::cpu
