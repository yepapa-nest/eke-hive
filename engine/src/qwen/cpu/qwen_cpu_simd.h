// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// AVX2 element-wise kernels of the Qwen CPU engine, each with a scalar twin that performs the same IEEE operations in the same order, so
//   vector bodies and scalar tails (and engine/tests/test_qwen_cpu_simd.cpp) agree bit for bit. exp is a Cody–Waite reduction with a degree-5
//   polynomial (relative error ~1e-7, below the bf16 roundings that follow every use), shared by both forms.
#pragma once
#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hive::qwen::cpu::simd {

// ---- scalar ---------------------------------------------------------------------------------------------------------------------------------
inline float bf16r_s(float x) {
  uint32_t u; std::memcpy(&u, &x, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  float r; std::memcpy(&r, &u, 4); return r;
}
constexpr float kExpHi = 88.0f, kExpLo = -87.0f, kLog2e = 1.44269504088896341f, kLn2Hi = 0.693359375f, kLn2Lo = -2.12194440e-4f;
constexpr float kP0 = 1.9875691500e-4f, kP1 = 1.3981999507e-3f, kP2 = 8.3334519073e-3f, kP3 = 4.1665795894e-2f, kP4 = 1.6666665459e-1f,
                kP5 = 5.0000001201e-1f;
inline float exp_s(float x) {
  x = std::fmin(std::fmax(x, kExpLo), kExpHi);
  const float n = std::nearbyint(x * kLog2e);
  float r = std::fma(n, -kLn2Hi, x);
  r = std::fma(n, -kLn2Lo, r);
  float p = kP0;
  p = std::fma(p, r, kP1); p = std::fma(p, r, kP2); p = std::fma(p, r, kP3); p = std::fma(p, r, kP4); p = std::fma(p, r, kP5);
  const float rr = r * r;
  float y = std::fma(p, rr, r);
  y = y + 1.f;
  const uint32_t e = (uint32_t)((int32_t)n + 127) << 23;
  float s; std::memcpy(&s, &e, 4);
  return y * s;
}
inline float sigm_s(float x) { return 1.f / (1.f + exp_s(-x)); }
inline float silu_s(float x) { return x / (1.f + exp_s(-x)); }

// ---- AVX2 -----------------------------------------------------------------------------------------------------------------------------------
inline __m256 bf16r8(__m256 x) {
  const __m256i u = _mm256_castps_si256(x);
  const __m256i ex = _mm256_set1_epi32(0x7f800000);
  const __m256i special = _mm256_cmpeq_epi32(_mm256_and_si256(u, ex), ex);
  const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
  const __m256i r = _mm256_and_si256(_mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), lsb), _mm256_set1_epi32((int)0xffff0000u));
  return _mm256_castsi256_ps(_mm256_blendv_epi8(r, u, special));
}
inline __m256 exp8(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(kExpLo)), _mm256_set1_ps(kExpHi));
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kLog2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fmadd_ps(n, _mm256_set1_ps(-kLn2Hi), x);
  r = _mm256_fmadd_ps(n, _mm256_set1_ps(-kLn2Lo), r);
  __m256 p = _mm256_set1_ps(kP0);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP1)); p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP2)); p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP3));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP4)); p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP5));
  const __m256 rr = _mm256_mul_ps(r, r);
  __m256 y = _mm256_fmadd_ps(p, rr, r);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.f));
  const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}
inline __m256 neg8(__m256 x) { return _mm256_xor_ps(x, _mm256_set1_ps(-0.f)); }
inline __m256 sigm8(__m256 x) { return _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_set1_ps(1.f), exp8(neg8(x)))); }
inline __m256 silu8(__m256 x) { return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.f), exp8(neg8(x)))); }

// ---- array kernels (vector body + identical scalar tail); the *_ref forms are all-scalar ------------------------------------------------
// Σ x² with 8 lane partial sums (lane i takes k ≡ i mod 8 of the 8-aligned part), lanes added in order, then the tail
inline float sumsq_ref(const float* x, int n) {
  float l[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const int n8 = n & ~7;
  for (int k = 0; k < n8; ++k) l[k & 7] = std::fma(x[k], x[k], l[k & 7]);
  float s = l[0];
  for (int i = 1; i < 8; ++i) s += l[i];
  for (int k = n8; k < n; ++k) s = std::fma(x[k], x[k], s);
  return s;
}
inline float sumsq(const float* x, int n) {
  __m256 a = _mm256_setzero_ps();
  const int n8 = n & ~7;
  for (int k = 0; k < n8; k += 8) { const __m256 v = _mm256_loadu_ps(x + k); a = _mm256_fmadd_ps(v, v, a); }
  alignas(32) float l[8];
  _mm256_store_ps(l, a);
  float s = l[0];
  for (int i = 1; i < 8; ++i) s += l[i];
  for (int k = n8; k < n; ++k) s = std::fma(x[k], x[k], s);
  return s;
}
inline void round_n_ref(float* x, size_t n) { for (size_t i = 0; i < n; ++i) x[i] = bf16r_s(x[i]); }
inline void round_n(float* x, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, bf16r8(_mm256_loadu_ps(x + i)));
  for (; i < n; ++i) x[i] = bf16r_s(x[i]);
}
// out = bf16(x · r · (1 + w)), r = 1/sqrt(Σx²/n + eps)
#define HQ_NORM_BODY(SUMSQ)                                                       \
  const float r = 1.f / std::sqrt(SUMSQ(x, n) / (float)n + eps);
inline void norm1p_ref(const float* x, const float* w, int n, float eps, float* out) {
  HQ_NORM_BODY(sumsq_ref)
  for (int k = 0; k < n; ++k) out[k] = bf16r_s(x[k] * r * (1.f + w[k]));
}
inline void norm1p_v(const float* x, const float* w, int n, float eps, float* out) {
  HQ_NORM_BODY(sumsq)
  const __m256 rv = _mm256_set1_ps(r), one = _mm256_set1_ps(1.f);
  int k = 0;
  for (; k + 8 <= n; k += 8)
    _mm256_storeu_ps(out + k, bf16r8(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + k), rv), _mm256_add_ps(one, _mm256_loadu_ps(w + k)))));
  for (; k < n; ++k) out[k] = bf16r_s(x[k] * r * (1.f + w[k]));
}
#undef HQ_NORM_BODY
// hyper-connection mix of one row: x[d] = bf16(Σ_c bf16(bf16(sigmoid(u[c·H + d])) · hn[c·H + d]) / hc)
inline void hc_mix_ref(const float* u, const float* hn, int H, int hc, float* x) {
  for (int d = 0; d < H; ++d) {
    float acc = 0.f;
    for (int c = 0; c < hc; ++c) acc += bf16r_s(bf16r_s(sigm_s(u[(size_t)c * H + d])) * hn[(size_t)c * H + d]);
    x[d] = bf16r_s(acc / (float)hc);
  }
}
inline void hc_mix_v(const float* u, const float* hn, int H, int hc, float* x) {
  const __m256 hcv = _mm256_set1_ps((float)hc);
  int d = 0;
  for (; d + 8 <= H; d += 8) {
    __m256 acc = _mm256_setzero_ps();
    for (int c = 0; c < hc; ++c) {
      const __m256 s = bf16r8(sigm8(_mm256_loadu_ps(u + (size_t)c * H + d)));
      acc = _mm256_add_ps(acc, bf16r8(_mm256_mul_ps(s, _mm256_loadu_ps(hn + (size_t)c * H + d))));
    }
    _mm256_storeu_ps(x + d, bf16r8(_mm256_div_ps(acc, hcv)));
  }
  for (; d < H; ++d) {
    float acc = 0.f;
    for (int c = 0; c < hc; ++c) acc += bf16r_s(bf16r_s(sigm_s(u[(size_t)c * H + d])) * hn[(size_t)c * H + d]);
    x[d] = bf16r_s(acc / (float)hc);
  }
}
// h[c·H + d] = bf16(h + bf16(y[d] · g[c]))
inline void hc_combine_ref(float* h, const float* y, const float* g, int H, int hc) {
  for (int c = 0; c < hc; ++c) for (int d = 0; d < H; ++d) { float& v = h[(size_t)c * H + d]; v = bf16r_s(v + bf16r_s(y[d] * g[c])); }
}
inline void hc_combine_v(float* h, const float* y, const float* g, int H, int hc) {
  for (int c = 0; c < hc; ++c) {
    float* hr = h + (size_t)c * H;
    const __m256 gv = _mm256_set1_ps(g[c]);
    int d = 0;
    for (; d + 8 <= H; d += 8)
      _mm256_storeu_ps(hr + d, bf16r8(_mm256_add_ps(_mm256_loadu_ps(hr + d), bf16r8(_mm256_mul_ps(_mm256_loadu_ps(y + d), gv)))));
    for (; d < H; ++d) hr[d] = bf16r_s(hr[d] + bf16r_s(y[d] * g[c]));
  }
}
// out = bf16(silu(bf16(a · sa)) · bf16(b · sb))   (expert / shared-expert activation; sa = sb = 1 on bf16 inputs is the identity)
inline void silu_mul_ref(const float* a, const float* b, int n, float sa, float sb, float* out) {
  for (int i = 0; i < n; ++i) out[i] = bf16r_s(silu_s(bf16r_s(a[i] * sa)) * bf16r_s(b[i] * sb));
}
inline void silu_mul_v(const float* a, const float* b, int n, float sa, float sb, float* out) {
  const __m256 av = _mm256_set1_ps(sa), bv = _mm256_set1_ps(sb);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 g = bf16r8(_mm256_mul_ps(_mm256_loadu_ps(a + i), av)), u = bf16r8(_mm256_mul_ps(_mm256_loadu_ps(b + i), bv));
    _mm256_storeu_ps(out + i, bf16r8(_mm256_mul_ps(silu8(g), u)));
  }
  for (; i < n; ++i) out[i] = bf16r_s(silu_s(bf16r_s(a[i] * sa)) * bf16r_s(b[i] * sb));
}
// attention output gate: o = bf16(bf16(att) · sigmoid(gate))
inline void gate_ref(const float* att, const float* gate, int n, float* o) { for (int i = 0; i < n; ++i) o[i] = bf16r_s(bf16r_s(att[i]) * sigm_s(gate[i])); }
inline void gate_v(const float* att, const float* gate, int n, float* o) {
  int i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(o + i, bf16r8(_mm256_mul_ps(bf16r8(_mm256_loadu_ps(att + i)), sigm8(_mm256_loadu_ps(gate + i)))));
  for (; i < n; ++i) o[i] = bf16r_s(bf16r_s(att[i]) * sigm_s(gate[i]));
}
// gated RMS norm of one value head in place: o = bf16(o · rs · w · sigmoid(z)), rs = 1/sqrt(Σo²/n + eps)
inline void gated_norm_ref(float* o, const float* z, const float* w, int n, float eps) {
  const float rs = 1.f / std::sqrt(sumsq_ref(o, n) / (float)n + eps);
  for (int j = 0; j < n; ++j) o[j] = bf16r_s(o[j] * rs * w[j] * sigm_s(z[j]));
}
inline void gated_norm_v(float* o, const float* z, const float* w, int n, float eps) {
  const float rs = 1.f / std::sqrt(sumsq(o, n) / (float)n + eps);
  const __m256 rv = _mm256_set1_ps(rs);
  int j = 0;
  for (; j + 8 <= n; j += 8)
    _mm256_storeu_ps(o + j, bf16r8(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(o + j), rv), _mm256_loadu_ps(w + j)),
                                                 sigm8(_mm256_loadu_ps(z + j)))));
  for (; j < n; ++j) o[j] = bf16r_s(o[j] * rs * w[j] * sigm_s(z[j]));
}

}  // namespace hive::qwen::cpu::simd
