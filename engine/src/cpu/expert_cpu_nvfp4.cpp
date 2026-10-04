// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// NVFP4 CPU expert kernels (AVX2 + FMA + F16C) — see hive/expert_cpu.h. Kept apart from the E2M1_B32 kernels so that the DeepSeek
//   path is untouched.
#include <immintrin.h>

#include <cmath>
#include <cstring>

#include "hive/common.h"
#include "hive/expert_cpu.h"

namespace hive::cpu {

namespace {
alignas(32) const uint8_t kNibF16Hi[32] = {0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46, 0x80, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6,
                                            0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46, 0x80, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6};
const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

struct E4M3Lut {
  float v[256];
  E4M3Lut() { for (int i = 0; i < 256; ++i) v[i] = e4m3_to_f32((uint8_t)i); }
};
const E4M3Lut& e4m3_lut() { static const E4M3Lut L; return L; }

inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
// 16 bytes = 32 nibbles → fp32 in element order [0-7] [16-23] [8-15] [24-31] (same trick as the E2M1_B32 kernel)
inline void nib_to_f32x4(const uint8_t* wp, const __m256i& lut, __m256& w0, __m256& w1, __m256& w2, __m256& w3) {
  __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wp));
  __m256i raw2 = _mm256_cvtepu8_epi16(raw);
  __m256i lo = _mm256_and_si256(raw2, _mm256_set1_epi16(0x000F));
  __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw2, 4), _mm256_set1_epi16(0x000F));
  __m256i codes = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 8));
  __m256i hib = _mm256_shuffle_epi8(lut, codes);
  __m256i z = _mm256_setzero_si256();
  __m256i f16a = _mm256_unpacklo_epi8(z, hib), f16b = _mm256_unpackhi_epi8(z, hib);
  w0 = _mm256_cvtph_ps(_mm256_castsi256_si128(f16a));
  w1 = _mm256_cvtph_ps(_mm256_extracti128_si256(f16a, 1));
  w2 = _mm256_cvtph_ps(_mm256_castsi256_si128(f16b));
  w3 = _mm256_cvtph_ps(_mm256_extracti128_si256(f16b, 1));
}
inline float bf16_round(float x) { return bf2f(f2bf(x)); }
inline float silu(float x) { return x / (1.f + expf(-x)); }
}  // namespace

void gemv_nvfp4_rows_scalar(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* a, int n0, int n1, float* y) {
  const float* L = e4m3_lut().v;
  for (int n = n0; n < n1; ++n) {
    const uint8_t* wr = W + (size_t)n * (K / 2);
    const uint8_t* sr = sw + (size_t)n * (K / 16);
    float acc = 0.f;
    for (int b = 0; b < K / 16; ++b) {
      float d = 0.f;
      for (int i = 0; i < 16; ++i) {
        const int k = b * 16 + i;
        const uint8_t byte = wr[k / 2];
        d += a[k] * kE2M1[(k & 1) ? (byte >> 4) : (byte & 15)];
      }
      acc += d * L[sr[b]];
    }
    y[n] = acc * gscale;
  }
}

// 32-column blocks: elements 0-15 use scale[2b], 16-31 use scale[2b+1]. Partial sums stay in 8 lanes and are reduced at the row end.
void gemv_nvfp4_rows(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* a, int n0, int n1, float* y) {
  HIVE_CHECK(K % 32 == 0, "nvfp4 gemv needs K % 32 == 0");
  const float* L = e4m3_lut().v;
  const __m256i lut = _mm256_load_si256(reinterpret_cast<const __m256i*>(kNibF16Hi));
  const int nb = K / 32;
  for (int n = n0; n < n1; ++n) {
    const uint8_t* wr = W + (size_t)n * (K / 2);
    const uint8_t* sr = sw + (size_t)n * (K / 16);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      const float* ap = a + b * 32;
      __m256 w0, w1, w2, w3;
      nib_to_f32x4(wr + b * 16, lut, w0, w1, w2, w3);
      __m256 lo = _mm256_mul_ps(w0, _mm256_loadu_ps(ap));            // elements 0-7
      lo = _mm256_fmadd_ps(w2, _mm256_loadu_ps(ap + 8), lo);        // 8-15
      __m256 hi = _mm256_mul_ps(w1, _mm256_loadu_ps(ap + 16));       // 16-23
      hi = _mm256_fmadd_ps(w3, _mm256_loadu_ps(ap + 24), hi);       // 24-31
      acc = _mm256_fmadd_ps(lo, _mm256_set1_ps(L[sr[2 * b]]), acc);
      acc = _mm256_fmadd_ps(hi, _mm256_set1_ps(L[sr[2 * b + 1]]), acc);
    }
    y[n] = hsum256(acc) * gscale;
  }
}

void gemv_nvfp4_rows_multi(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* const* a, int R, int n0, int n1,
                           float* const* y) {
  HIVE_CHECK(K % 32 == 0 && R >= 1 && R <= 8, "nvfp4 multi gemv");
  if (R == 1) { gemv_nvfp4_rows(W, sw, gscale, N, K, a[0], n0, n1, y[0]); return; }
  const float* L = e4m3_lut().v;
  const __m256i lut = _mm256_load_si256(reinterpret_cast<const __m256i*>(kNibF16Hi));
  const int nb = K / 32;
  for (int n = n0; n < n1; ++n) {
    const uint8_t* wr = W + (size_t)n * (K / 2);
    const uint8_t* sr = sw + (size_t)n * (K / 16);
    __m256 acc[8];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      __m256 w0, w1, w2, w3;
      nib_to_f32x4(wr + b * 16, lut, w0, w1, w2, w3);
      const __m256 sl = _mm256_set1_ps(L[sr[2 * b]]), sh = _mm256_set1_ps(L[sr[2 * b + 1]]);
      for (int r = 0; r < R; ++r) {
        const float* ap = a[r] + b * 32;
        __m256 lo = _mm256_mul_ps(w0, _mm256_loadu_ps(ap));
        lo = _mm256_fmadd_ps(w2, _mm256_loadu_ps(ap + 8), lo);
        __m256 hi = _mm256_mul_ps(w1, _mm256_loadu_ps(ap + 16));
        hi = _mm256_fmadd_ps(w3, _mm256_loadu_ps(ap + 24), hi);
        acc[r] = _mm256_fmadd_ps(lo, sl, acc[r]);
        acc[r] = _mm256_fmadd_ps(hi, sh, acc[r]);
      }
    }
    for (int r = 0; r < R; ++r) y[r][n] = hsum256(acc[r]) * gscale;
  }
}

void expert_forward_nvfp4(const ExpertDescNvfp4& e, const float* a, float route_w, float swiglu_limit, float* out, float* tmp) {
  const int K = e.dim, I = e.inter;
  float* gate = tmp; float* up = gate + I; float* y = up + I;
  gemv_nvfp4_rows(e.w1, e.s1, e.g1, I, K, a, 0, I, gate);
  gemv_nvfp4_rows(e.w3, e.s3, e.g3, I, K, a, 0, I, up);
  for (int i = 0; i < I; ++i) {
    float g = bf16_round(gate[i]), u = bf16_round(up[i]);
    if (swiglu_limit > 0.f) { u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit); g = fminf(g, swiglu_limit); }
    y[i] = bf16_round(silu(g) * u);
  }
  gemv_nvfp4_rows(e.w2, e.s2, e.g2, K, I, y, 0, K, out);
  for (int k = 0; k < K; ++k) out[k] = bf16_round(out[k]) * route_w;
}


// ---- W4A8 path (HIVE_GLM_CPU_Q8) — the llama.cpp nvfp4·q8_0 AVX2 technique (ggml-org/llama.cpp PR #23961, MIT, © The ggml authors —
//   THIRD_PARTY_NOTICES item 4): weights via a pshufb table
//   of doubled e2m1 values (int8), activations quantized to int8 per 32-column block (fp32 scale), integer dot (maddubs + madd), block
//   scales applied in fp32. Activation layout per 32-column block: [even columns 0,2,..,30 | odd columns 1,3,..,31] — the order in which
//   the low / high nibbles of the 16 weight bytes come out of pshufb (byte j holds columns 2j (low) and 2j+1 (high)).
//   Measured (tests/bench_nvfp4_cpu.cpp, GLM gate shape [2048, 4096], node-local, 3975WX): relative error vs fp64 7e-3 (fp32-activation
//   kernel 3e-7); 1 row: 1 thread 5.4 → 7.1 GB/s but 12 threads 49.4 → 44.8 GB/s (the node's memory bandwidth is the limit, not the ALUs);
//   4 rows: 12 threads 31.0 → 39.4 GB/s. Not wired into the decode path (no gain at 1 row, accuracy cost) — kept for the multi-row case.
void quantize_q8_nvfp4(const float* a, int K, int8_t* q, float* s) {
  for (int b = 0; b < K / 32; ++b) {
    const float* ab = a + b * 32;
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(ab[i]));
    const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
    s[b] = d;
    int8_t* qb = q + b * 32;
    for (int i = 0; i < 16; ++i) {
      qb[i] = (int8_t)lrintf(ab[2 * i] * id);
      qb[16 + i] = (int8_t)lrintf(ab[2 * i + 1] * id);
    }
  }
}

void gemv_nvfp4_q8_rows_multi(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const int8_t* const* aq, const float* const* as,
                              int R, int n0, int n1, float* const* y) {
  HIVE_CHECK(K % 32 == 0 && R >= 1 && R <= 8, "nvfp4 q8 gemv");
  (void)N;
  const float* L = e4m3_lut().v;
  const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const __m128i m4 = _mm_set1_epi8(0x0f);
  const __m256i ones = _mm256_set1_epi16(1);
  const int nb = K / 32;
  const float post = gscale * 0.5f;  // table values are doubled
  for (int n = n0; n < n1; ++n) {
    const uint8_t* wr = W + (size_t)n * (K / 2);
    const uint8_t* sr = sw + (size_t)n * (K / 16);
    __m256 acc[8];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wr + b * 16));
      const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, m4));
      const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));
      const __m256i w = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
      const __m256i aw = _mm256_sign_epi8(w, w);
      const float s0 = L[sr[2 * b]], s1 = L[sr[2 * b + 1]];
      const __m256 sv = _mm256_setr_ps(s0, s0, s1, s1, s0, s0, s1, s1);
      for (int r = 0; r < R; ++r) {
        const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq[r] + b * 32));
        const __m256i p16 = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(av, w));
        const __m256i p32 = _mm256_madd_epi16(p16, ones);
        acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_mul_ps(sv, _mm256_set1_ps(as[r][b])), acc[r]);
      }
    }
    for (int r = 0; r < R; ++r) y[r][n] = hsum256(acc[r]) * post;
  }
}

}  // namespace hive::cpu
