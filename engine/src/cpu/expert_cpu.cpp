// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/expert_cpu.h"

#include <immintrin.h>

#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "hive/common.h"

namespace hive::cpu {

namespace {

// e2m1 code (nibble) → fp16 high byte. 0,.5,1,1.5,2,3,4,6 = 0x00,0x38,0x3C,0x3E,0x40,0x42,0x44,0x46 (negative |0x80). The low byte is 0.
alignas(32) const uint8_t kNibF16Hi[32] = {0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46, 0x80, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6,
                                            0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46, 0x80, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6};

inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
// e8m0 → fp32 (bit assembly instead of libm ldexpf; assumes 255 = NaN never occurs)
inline float e8m0_fast(uint8_t b) {
  uint32_t bits = b ? ((uint32_t)b << 23) : 0x00400000u;
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}
inline float bf16_round(float x) { return bf2f(f2bf(x)); }
inline float silu(float x) { return x / (1.f + expf(-x)); }

// 16 nibble bytes → 4 fp32 vectors (element order [0-7],[16-23],[8-15],[24-31] — order does not matter for the sum, so activations are multiplied in the same order)
inline void nib_to_f32x4(const uint8_t* wp, const __m256i& lut, __m256& w0, __m256& w1, __m256& w2, __m256& w3) {
  __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wp));
  __m256i raw2 = _mm256_cvtepu8_epi16(raw);
  __m256i lo = _mm256_and_si256(raw2, _mm256_set1_epi16(0x000F));
  __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw2, 4), _mm256_set1_epi16(0x000F));
  __m256i codes = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 8));        // 32 byte codes, element order
  __m256i hib = _mm256_shuffle_epi8(lut, codes);                          // fp16 high bytes
  __m256i z = _mm256_setzero_si256();
  __m256i f16a = _mm256_unpacklo_epi8(z, hib);  // elements 0-7 | 16-23
  __m256i f16b = _mm256_unpackhi_epi8(z, hib);  // elements 8-15 | 24-31
  w0 = _mm256_cvtph_ps(_mm256_castsi256_si128(f16a));
  w1 = _mm256_cvtph_ps(_mm256_extracti128_si256(f16a, 1));
  w2 = _mm256_cvtph_ps(_mm256_castsi256_si128(f16b));
  w3 = _mm256_cvtph_ps(_mm256_extracti128_si256(f16b, 1));
}

}  // namespace

// Reference version (_ref — used when HIVE_CPU_GEMV2 is off). Two rows at a time sharing the activation loads. 32 MACs per block → 8-lane partial sums × (sa·sw) accumulated → horizontal sum at the end of the row.
void gemv_e2m1_rows_ref(const uint8_t* W, const uint8_t* sw, int N, int K, const float* a_f32, const float* sa_f32, int n0,
                    int n1, float* y) {
  const int nb = K / 32;
  const __m256i lut = _mm256_load_si256(reinterpret_cast<const __m256i*>(kNibF16Hi));
  int n = n0;
  for (; n + 1 < n1; n += 2) {
    const uint8_t* wr0 = W + (size_t)n * (K / 2);
    const uint8_t* wr1 = wr0 + (K / 2);
    const uint8_t* sr0 = sw + (size_t)n * nb;
    const uint8_t* sr1 = sr0 + nb;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      const float* ap = a_f32 + b * 32;
      __m256 a0 = _mm256_loadu_ps(ap), a1 = _mm256_loadu_ps(ap + 16), a2 = _mm256_loadu_ps(ap + 8), a3 = _mm256_loadu_ps(ap + 24);
      __m256 w0, w1, w2, w3;
      nib_to_f32x4(wr0 + b * 16, lut, w0, w1, w2, w3);
      __m256 d0 = _mm256_mul_ps(w0, a0);
      d0 = _mm256_fmadd_ps(w1, a1, d0);
      d0 = _mm256_fmadd_ps(w2, a2, d0);
      d0 = _mm256_fmadd_ps(w3, a3, d0);
      nib_to_f32x4(wr1 + b * 16, lut, w0, w1, w2, w3);
      __m256 d1 = _mm256_mul_ps(w0, a0);
      d1 = _mm256_fmadd_ps(w1, a1, d1);
      d1 = _mm256_fmadd_ps(w2, a2, d1);
      d1 = _mm256_fmadd_ps(w3, a3, d1);
      const float sa = sa_f32[b];
      acc0 = _mm256_fmadd_ps(d0, _mm256_set1_ps(sa * e8m0_fast(sr0[b])), acc0);
      acc1 = _mm256_fmadd_ps(d1, _mm256_set1_ps(sa * e8m0_fast(sr1[b])), acc1);
    }
    y[n] = hsum256(acc0);
    y[n + 1] = hsum256(acc1);
  }
  for (; n < n1; ++n) {
    const uint8_t* wr0 = W + (size_t)n * (K / 2);
    const uint8_t* sr0 = sw + (size_t)n * nb;
    __m256 acc0 = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      const float* ap = a_f32 + b * 32;
      __m256 w0, w1, w2, w3;
      nib_to_f32x4(wr0 + b * 16, lut, w0, w1, w2, w3);
      __m256 d0 = _mm256_mul_ps(w0, _mm256_loadu_ps(ap));
      d0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(ap + 16), d0);
      d0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(ap + 8), d0);
      d0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(ap + 24), d0);
      acc0 = _mm256_fmadd_ps(d0, _mm256_set1_ps(sa_f32[b] * e8m0_fast(sr0[b])), acc0);
    }
    y[n] = hsum256(acc0);
  }
}

// R rows: each block's 32 weights are unpacked once and multiplied with R activations. R ≤ 8.
void gemv_e2m1_rows_multi_ref(const uint8_t* W, const uint8_t* sw, int N, int K, const float* const* a_f32, const float* const* sa_f32, int R,
                          int n0, int n1, float* const* y) {
  if (R == 1) { gemv_e2m1_rows_ref(W, sw, N, K, a_f32[0], sa_f32[0], n0, n1, y[0]); return; }  // keep the two-row unrolled path
  const int nb = K / 32;
  const __m256i lut = _mm256_load_si256(reinterpret_cast<const __m256i*>(kNibF16Hi));
  for (int n = n0; n < n1; ++n) {
    const uint8_t* wr = W + (size_t)n * (K / 2);
    const uint8_t* sr = sw + (size_t)n * nb;
    __m256 acc[8];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      __m256 w0, w1, w2, w3;
      nib_to_f32x4(wr + b * 16, lut, w0, w1, w2, w3);
      const float swb = e8m0_fast(sr[b]);
      for (int r = 0; r < R; ++r) {
        const float* ap = a_f32[r] + b * 32;
        __m256 d = _mm256_mul_ps(w0, _mm256_loadu_ps(ap));
        d = _mm256_fmadd_ps(w1, _mm256_loadu_ps(ap + 16), d);
        d = _mm256_fmadd_ps(w2, _mm256_loadu_ps(ap + 8), d);
        d = _mm256_fmadd_ps(w3, _mm256_loadu_ps(ap + 24), d);
        acc[r] = _mm256_fmadd_ps(d, _mm256_set1_ps(sa_f32[r][b] * swb), acc[r]);
      }
    }
    for (int r = 0; r < R; ++r) y[r][n] = hsum256(acc[r]);
  }
}

// ---- HIVE_CPU_GEMV2: faster version with the same accumulation order (_v2 below) --------------------------------------------------------------------------
// Rationale (reasoned for Zen 2 3975WX, not measured there): per block (32 weights, 16 B) the _ref unpack = vpmovzxbw (cross-lane) + and/srl/and/sll/or + pshufb + 2 unpacks +
//   2 extracts + 4 ymm vcvtph2ps (2 µops each on Zen 2) ≈ 20 µops; the scale = scalar e8m0 assembly (movzx, shl, branch/cmov, movd) + mulss + broadcast ≈ 6 µops.
//   Two rows × (unpack 20 + FMA 5 + scale 6) on 4 FP pipes ≈ 15 cycles per block — one core reads only ~2 B/cycle.
//   _v2: (a) nibble → fp32 directly, without fp16 conversion: the fp32 bits of the 8 e2m1 values are nonzero only in the top 2 bytes (low 16 bits = 0) → broadcast the raw 16 B to both lanes →
//   gather even/odd nibbles with pshufb into the byte order of **the same lane layout as _ref** (w0 = elements 0-7, w1 = 16-23, w2 = 8-15, w3 = 24-31) → two byte tables (pshufb) →
//   assemble fp32 bits with 2 8-bit + 4 16-bit unpacks (13 µops, all integer shuffles/logic — no converter). Values are bit-identical to the cvtph path (±0 included — all 16 codes checked).
//   (b) the scales sa·sw are built 8 blocks at a time as a vector (cvtepu8_epi32, sll 23, b==0 → 0x00400000 blend, mulps), kept on the stack and broadcast from memory per block (load µops only).
//   The product sa·sw is the same fp32 multiply (IEEE multiplication is commutative); the e8m0 bit assembly is the same formula as e8m0_fast.
//   Numerics are **bit-identical** to _ref: every output lane keeps the same operation chain — d = w0·a0 → fma(w1,a1) → fma(w2,a2) → fma(w3,a3), acc = fma(d, sa·sw, acc) in block order,
//   and the same final hsum256. Only **how** the w vectors and scale values are produced changes, not their values (engine/tests/bench_expert_cpu.cpp compares all output bits on real shapes).
//   (c) software prefetch per row stream (v2_prefetch — does not affect values).
//   The gain shrinks where memory bandwidth is the limit (many jobs, saturated node bandwidth) — decode (1–3 jobs per layer) is bound by per-core compute.
namespace {
alignas(32) const uint8_t kV2IdxLo[32] = {0x00, 0x80, 0x01, 0x80, 0x08, 0x80, 0x09, 0x80, 0x04, 0x80, 0x05, 0x80, 0x0C, 0x80, 0x0D, 0x80,
                                          0x02, 0x80, 0x03, 0x80, 0x0A, 0x80, 0x0B, 0x80, 0x06, 0x80, 0x07, 0x80, 0x0E, 0x80, 0x0F, 0x80};
alignas(32) const uint8_t kV2IdxHi[32] = {0x80, 0x00, 0x80, 0x01, 0x80, 0x08, 0x80, 0x09, 0x80, 0x04, 0x80, 0x05, 0x80, 0x0C, 0x80, 0x0D,
                                          0x80, 0x02, 0x80, 0x03, 0x80, 0x0A, 0x80, 0x0B, 0x80, 0x06, 0x80, 0x07, 0x80, 0x0E, 0x80, 0x0F};
// e2m1 code → fp32 bits 24-31 / 16-23 (0, .5, 1, 1.5, 2, 3, 4, 6; negative = sign bit). The same 16 entries in both halves.
alignas(32) const uint8_t kV2B3[32] = {0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0,
                                       0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};
alignas(32) const uint8_t kV2B2[32] = {0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
                                       0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
struct V2Tabs {
  __m256i lo, hi, b3, b2;
  V2Tabs() : lo(_mm256_load_si256(reinterpret_cast<const __m256i*>(kV2IdxLo))), hi(_mm256_load_si256(reinterpret_cast<const __m256i*>(kV2IdxHi))),
             b3(_mm256_load_si256(reinterpret_cast<const __m256i*>(kV2B3))), b2(_mm256_load_si256(reinterpret_cast<const __m256i*>(kV2B2))) {}
};
// 16 nibble bytes → 4 fp32 vectors — same element layout and same values as nib_to_f32x4
inline void nib_to_f32x4_v2(const uint8_t* wp, const V2Tabs& t, __m256& w0, __m256& w1, __m256& w2, __m256& w3) {
  const __m256i raw = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wp)));
  const __m256i m = _mm256_set1_epi8(0x0F);
  const __m256i lo = _mm256_and_si256(raw, m), hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m);
  const __m256i x = _mm256_or_si256(_mm256_shuffle_epi8(lo, t.lo), _mm256_shuffle_epi8(hi, t.hi));  // half 0 = [0-3,16-19,8-11,24-27] · half 1 = [4-7,20-23,12-15,28-31]
  const __m256i b3 = _mm256_shuffle_epi8(t.b3, x), b2 = _mm256_shuffle_epi8(t.b2, x);
  const __m256i wl = _mm256_unpacklo_epi8(b2, b3), wh = _mm256_unpackhi_epi8(b2, b3);  // bf16 bits (= top 16 bits of fp32)
  const __m256i z = _mm256_setzero_si256();
  w0 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(z, wl));  // 0-3 | 4-7
  w1 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(z, wl));  // 16-19 | 20-23
  w2 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(z, wh));  // 8-11 | 12-15
  w3 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(z, wh));  // 24-27 | 28-31
}
// 8 e8m0 values → fp32 (same bits as e8m0_fast: b<<23, b==0 → 0x00400000)
inline __m256 e8m0x8(const uint8_t* sr) {
  const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(sr)));
  const __m256i bits = _mm256_blendv_epi8(_mm256_slli_epi32(b, 23), _mm256_set1_epi32(0x00400000), _mm256_cmpeq_epi32(b, _mm256_setzero_si256()));
  return _mm256_castsi256_ps(bits);
}
// Software prefetch: every 8 blocks (128 B), the 128 B (two cache lines) kV2Pf ahead. Per row stream — past the end of a row it pulls in the next row (the next row pair of the same job).
//   Does not affect values (prefetch is a load hint). 2048 B = best measured single-thread DRAM value on a Ryzen 9 9955HX (of 256/512/1024/2048, bench_expert_cpu) — not measured on Zen 2.
constexpr int kV2Pf = 2048;
inline void v2_prefetch(const uint8_t* p) {
  _mm_prefetch(reinterpret_cast<const char*>(p + kV2Pf), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char*>(p + kV2Pf + 64), _MM_HINT_T0);
}
// One block: the _ref formula unchanged (d chain → acc fma)
inline __m256 blk_dot(const __m256& w0, const __m256& w1, const __m256& w2, const __m256& w3, const float* ap) {
  __m256 d = _mm256_mul_ps(w0, _mm256_loadu_ps(ap));
  d = _mm256_fmadd_ps(w1, _mm256_loadu_ps(ap + 16), d);
  d = _mm256_fmadd_ps(w2, _mm256_loadu_ps(ap + 8), d);
  d = _mm256_fmadd_ps(w3, _mm256_loadu_ps(ap + 24), d);
  return d;
}
}  // namespace

void gemv_e2m1_rows_v2(const uint8_t* W, const uint8_t* sw, int N, int K, const float* a_f32, const float* sa_f32, int n0, int n1, float* y) {
  (void)N;
  const int nb = K / 32;
  const V2Tabs t;
  int n = n0;
  for (; n + 1 < n1; n += 2) {
    const uint8_t* wr0 = W + (size_t)n * (K / 2);
    const uint8_t* wr1 = wr0 + (K / 2);
    const uint8_t* sr0 = sw + (size_t)n * nb;
    const uint8_t* sr1 = sr0 + nb;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int b = 0;
    for (; b + 8 <= nb; b += 8) {
      alignas(32) float s0[8], s1[8];
      v2_prefetch(wr0 + b * 16); v2_prefetch(wr1 + b * 16);
      const __m256 sa = _mm256_loadu_ps(sa_f32 + b);
      _mm256_store_ps(s0, _mm256_mul_ps(sa, e8m0x8(sr0 + b)));
      _mm256_store_ps(s1, _mm256_mul_ps(sa, e8m0x8(sr1 + b)));
      for (int j = 0; j < 8; ++j) {
        const float* ap = a_f32 + (b + j) * 32;
        __m256 w0, w1, w2, w3;
        nib_to_f32x4_v2(wr0 + (b + j) * 16, t, w0, w1, w2, w3);
        const __m256 d0 = blk_dot(w0, w1, w2, w3, ap);
        nib_to_f32x4_v2(wr1 + (b + j) * 16, t, w0, w1, w2, w3);
        const __m256 d1 = blk_dot(w0, w1, w2, w3, ap);
        acc0 = _mm256_fmadd_ps(d0, _mm256_broadcast_ss(s0 + j), acc0);
        acc1 = _mm256_fmadd_ps(d1, _mm256_broadcast_ss(s1 + j), acc1);
      }
    }
    for (; b < nb; ++b) {  // nb % 8 tail (none for the real shapes K = 5120 / 2304, nb = 160 / 72) — scalar scale formula as in _ref
      const float* ap = a_f32 + b * 32;
      __m256 w0, w1, w2, w3;
      nib_to_f32x4_v2(wr0 + b * 16, t, w0, w1, w2, w3);
      const __m256 d0 = blk_dot(w0, w1, w2, w3, ap);
      nib_to_f32x4_v2(wr1 + b * 16, t, w0, w1, w2, w3);
      const __m256 d1 = blk_dot(w0, w1, w2, w3, ap);
      const float sa = sa_f32[b];
      acc0 = _mm256_fmadd_ps(d0, _mm256_set1_ps(sa * e8m0_fast(sr0[b])), acc0);
      acc1 = _mm256_fmadd_ps(d1, _mm256_set1_ps(sa * e8m0_fast(sr1[b])), acc1);
    }
    y[n] = hsum256(acc0);
    y[n + 1] = hsum256(acc1);
  }
  for (; n < n1; ++n) {
    const uint8_t* wr0 = W + (size_t)n * (K / 2);
    const uint8_t* sr0 = sw + (size_t)n * nb;
    __m256 acc0 = _mm256_setzero_ps();
    int b = 0;
    for (; b + 8 <= nb; b += 8) {
      alignas(32) float s0[8];
      v2_prefetch(wr0 + b * 16);
      _mm256_store_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(sa_f32 + b), e8m0x8(sr0 + b)));
      for (int j = 0; j < 8; ++j) {
        __m256 w0, w1, w2, w3;
        nib_to_f32x4_v2(wr0 + (b + j) * 16, t, w0, w1, w2, w3);
        acc0 = _mm256_fmadd_ps(blk_dot(w0, w1, w2, w3, a_f32 + (b + j) * 32), _mm256_broadcast_ss(s0 + j), acc0);
      }
    }
    for (; b < nb; ++b) {
      __m256 w0, w1, w2, w3;
      nib_to_f32x4_v2(wr0 + b * 16, t, w0, w1, w2, w3);
      acc0 = _mm256_fmadd_ps(blk_dot(w0, w1, w2, w3, a_f32 + b * 32), _mm256_set1_ps(sa_f32[b] * e8m0_fast(sr0[b])), acc0);
    }
    y[n] = hsum256(acc0);
  }
}

void gemv_e2m1_rows_multi_v2(const uint8_t* W, const uint8_t* sw, int N, int K, const float* const* a_f32, const float* const* sa_f32, int R,
                             int n0, int n1, float* const* y) {
  if (R == 1) { gemv_e2m1_rows_v2(W, sw, N, K, a_f32[0], sa_f32[0], n0, n1, y[0]); return; }  // same R=1 branch as _ref (two-row unrolled, same accumulation)
  const int nb = K / 32;
  const V2Tabs t;
  for (int n = n0; n < n1; ++n) {
    const uint8_t* wr = W + (size_t)n * (K / 2);
    const uint8_t* sr = sw + (size_t)n * nb;
    __m256 acc[8];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    int b = 0;
    for (; b + 8 <= nb; b += 8) {
      alignas(32) float s[8][8];
      v2_prefetch(wr + b * 16);
      const __m256 swb = e8m0x8(sr + b);
      for (int r = 0; r < R; ++r) _mm256_store_ps(s[r], _mm256_mul_ps(_mm256_loadu_ps(sa_f32[r] + b), swb));  // _ref: sa_f32[r][b] * swb
      for (int j = 0; j < 8; ++j) {
        __m256 w0, w1, w2, w3;
        nib_to_f32x4_v2(wr + (b + j) * 16, t, w0, w1, w2, w3);
        for (int r = 0; r < R; ++r) acc[r] = _mm256_fmadd_ps(blk_dot(w0, w1, w2, w3, a_f32[r] + (b + j) * 32), _mm256_broadcast_ss(s[r] + j), acc[r]);
      }
    }
    for (; b < nb; ++b) {
      __m256 w0, w1, w2, w3;
      nib_to_f32x4_v2(wr + b * 16, t, w0, w1, w2, w3);
      const float swb = e8m0_fast(sr[b]);
      for (int r = 0; r < R; ++r) acc[r] = _mm256_fmadd_ps(blk_dot(w0, w1, w2, w3, a_f32[r] + b * 32), _mm256_set1_ps(sa_f32[r][b] * swb), acc[r]);
    }
    for (int r = 0; r < R; ++r) y[r][n] = hsum256(acc[r]);
  }
}

// ---- HIVE_CPU_MULTIROW2: multi-row (R ≥ 2) K-tiled version (_v3 below) --------------------------------------------------------------------
// Rationale (bench_pool_cpu on Zen 2: 1×1 row 0.32, 1×4 rows 0.44, 1×8 rows 0.79 ms — extra rows are not free): _v2 multi-row walks the whole K for every output row n
//   and re-reads the R activations ([K] fp32 = 20 KB/row — 160 KB for R=8) per block → R·20 KB of activations from L2 per output row (exceeds the 32 KB L1), vs 2.5 KB of weights.
//   _v3: loops over groups of NT output rows × K tiles (KT blocks = R·KT·128 B of activations — 16 KB for R=8, KT=16, fits in L1): one tile is used for all NT rows before the next tile.
//   The partial sums acc of output (n, r) are parked in a stack buffer between tiles and reloaded (ymm store/load preserves values). R is a template constant (R accumulators in registers).
//   Numerics are **bit-identical** to _v2 (= _ref): each output (n, r) keeps the same operation chain — blocks b in ascending order, d = w0·a0 → fma(w1,a1) → fma(w2,a2) → fma(w3,a3),
//   acc = fma(d, sa·sw, acc), the same final hsum256. Only the **iteration order** over (n, b) changes (tile-major within a row group); each output's accumulation order does not.
//   The scales sa·sw are the same fp32 products as in _v2 (8-block vector multiply = scalar multiply, IEEE multiplication is element-wise identical); nibble unpack = nib_to_f32x4_v2 (same values as the cvtph path).
namespace {
constexpr int kV3NT = 16;  // output-row group (acc buffer = NT·R·8 fp32 ≤ 4 KB)
// K tile (in blocks) chosen so the activation tile R·KT·128 B is ~16 KB (half of L1): R 2 → 64, 4 → 32, 8 → 16 (multiples of 8, so the 8-block scale vectors close within a tile)
constexpr int v3_kt(int R) { return R <= 2 ? 64 : R <= 4 ? 32 : 16; }
template <int R>
void multi_v3(const uint8_t* W, const uint8_t* sw, int K, const float* const* a_f32, const float* const* sa_f32, int n0, int n1, float* const* y) {
  constexpr int KT = v3_kt(R);
  const int nb = K / 32;
  const V2Tabs t;
  alignas(32) float accb[kV3NT][R][8];
  const float* ar[R];
  const float* sr_a[R];
  for (int r = 0; r < R; ++r) { ar[r] = a_f32[r]; sr_a[r] = sa_f32[r]; }
  for (int nc = n0; nc < n1; nc += kV3NT) {
    const int ne = std::min(nc + kV3NT, n1);
    for (int kb = 0; kb < nb; kb += KT) {
      const int ke = std::min(kb + KT, nb);
      for (int n = nc; n < ne; ++n) {
        const uint8_t* wr = W + (size_t)n * (K / 2);
        const uint8_t* sr = sw + (size_t)n * nb;
        float (*ab)[8] = accb[n - nc];
        __m256 acc[R];
        if (kb == 0) for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
        else for (int r = 0; r < R; ++r) acc[r] = _mm256_load_ps(ab[r]);
        if (ke < nb) {  // prefetch this row's next tile (weights KT·16 B + scales KT B) — used after NT rows (does not affect values, load hint)
          const uint8_t* pw = wr + ke * 16;
          for (int o = 0; o < std::min(KT, nb - ke) * 16; o += 64) _mm_prefetch(reinterpret_cast<const char*>(pw + o), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(sr + ke), _MM_HINT_T0);
        }
        int b = kb;
        for (; b + 8 <= ke; b += 8) {
          alignas(32) float s[R][8];
          const __m256 swb = e8m0x8(sr + b);
          for (int r = 0; r < R; ++r) _mm256_store_ps(s[r], _mm256_mul_ps(_mm256_loadu_ps(sr_a[r] + b), swb));  // same sa·sw as _v2
          for (int j = 0; j < 8; ++j) {
            __m256 w0, w1, w2, w3;
            nib_to_f32x4_v2(wr + (b + j) * 16, t, w0, w1, w2, w3);
            for (int r = 0; r < R; ++r) acc[r] = _mm256_fmadd_ps(blk_dot(w0, w1, w2, w3, ar[r] + (b + j) * 32), _mm256_broadcast_ss(s[r] + j), acc[r]);
          }
        }
        for (; b < ke; ++b) {  // nb % 8 tail (last tile only — none for the real shapes, nb = 160 / 72) — same scalar scale formula as the _v2 tail
          __m256 w0, w1, w2, w3;
          nib_to_f32x4_v2(wr + b * 16, t, w0, w1, w2, w3);
          const float swb = e8m0_fast(sr[b]);
          for (int r = 0; r < R; ++r) acc[r] = _mm256_fmadd_ps(blk_dot(w0, w1, w2, w3, ar[r] + b * 32), _mm256_set1_ps(sr_a[r][b] * swb), acc[r]);
        }
        if (ke < nb) for (int r = 0; r < R; ++r) _mm256_store_ps(ab[r], acc[r]);
        else for (int r = 0; r < R; ++r) y[r][n] = hsum256(acc[r]);
      }
    }
  }
}
}  // namespace

void gemv_e2m1_rows_multi_v3(const uint8_t* W, const uint8_t* sw, int N, int K, const float* const* a_f32, const float* const* sa_f32, int R,
                             int n0, int n1, float* const* y) {
  (void)N;
  switch (R) {
    case 2: multi_v3<2>(W, sw, K, a_f32, sa_f32, n0, n1, y); return;
    case 3: multi_v3<3>(W, sw, K, a_f32, sa_f32, n0, n1, y); return;
    case 4: multi_v3<4>(W, sw, K, a_f32, sa_f32, n0, n1, y); return;
    case 5: multi_v3<5>(W, sw, K, a_f32, sa_f32, n0, n1, y); return;
    case 6: multi_v3<6>(W, sw, K, a_f32, sa_f32, n0, n1, y); return;
    case 7: multi_v3<7>(W, sw, K, a_f32, sa_f32, n0, n1, y); return;
    case 8: multi_v3<8>(W, sw, K, a_f32, sa_f32, n0, n1, y); return;
    default: gemv_e2m1_rows_multi_v2(W, sw, N, K, a_f32, sa_f32, R, n0, n1, y); return;  // R = 1 (two-row unrolled version) — same accumulation
  }
}

// Switch HIVE_CPU_MULTIROW2 — unset / empty / "0" = off (0 — default entry unchanged). On = multi-row calls with R ≥ the minimum row count go to _v3:
//   an integer 2..8 = that minimum; any other "on" value ("1" included — absorbed, not rejected) = 5. Basis (Ryzen 9 9955HX, 16 threads, w1 5120×2304, 32-row jobs):
//   _v2 → _v3 ms/matrix R=3 0.114→0.108 · 4 0.114→0.109 · 5 0.143→0.121 · 6 0.163→0.139 · 7 0.192→0.162 · 8 0.213→0.184 — for R ≤ 4 DRAM streaming is
//   the limit (jumping between rows per tile disturbs the prefetcher), the gain is within noise and the pool bench loses (R=2 0.33→0.48 ms, first version); for R ≥ 5 compute is the limit, ~15 %.
//   Zen 2 has lower L2 bandwidth, so the threshold may differ there → it is a tunable (tools/bench_pool_cpu.py). Read once per process.
int multirow2_min_rows() {
  static const int n = [] {
    const char* v = getenv("HIVE_CPU_MULTIROW2");
    if (!(v && *v && strcmp(v, "0") != 0)) return 0;
    char* end = nullptr;
    const long x = strtol(v, &end, 10);
    return end != v && end && *end == 0 && x >= 2 && x <= 8 ? (int)x : 5;
  }();
  return n;
}
bool multirow2_enabled() { return multirow2_min_rows() > 0; }

// Switch HIVE_CPU_GEMV2 — unset / empty / "0" = off (_ref), anything else = on (_v2). Read once per process (function-local static — thread-safe initialization).
bool gemv2_enabled() {
  static const bool on = [] { const char* v = getenv("HIVE_CPU_GEMV2"); return v && *v && strcmp(v, "0") != 0; }();
  return on;
}
void gemv_e2m1_rows(const uint8_t* W, const uint8_t* sw, int N, int K, const float* a_f32, const float* sa_f32, int n0, int n1, float* y) {
  if (gemv2_enabled()) gemv_e2m1_rows_v2(W, sw, N, K, a_f32, sa_f32, n0, n1, y);
  else gemv_e2m1_rows_ref(W, sw, N, K, a_f32, sa_f32, n0, n1, y);
}
void gemv_e2m1_rows_multi(const uint8_t* W, const uint8_t* sw, int N, int K, const float* const* a_f32, const float* const* sa_f32, int R,
                          int n0, int n1, float* const* y) {
  if (R >= 2 && multirow2_enabled() && R >= multirow2_min_rows()) { gemv_e2m1_rows_multi_v3(W, sw, N, K, a_f32, sa_f32, R, n0, n1, y); return; }  // HIVE_CPU_MULTIROW2 (same bits)
  if (gemv2_enabled()) gemv_e2m1_rows_multi_v2(W, sw, N, K, a_f32, sa_f32, R, n0, n1, y);
  else gemv_e2m1_rows_multi_ref(W, sw, N, K, a_f32, sa_f32, R, n0, n1, y);
}

// ---- HIVE_CPU_UNPACK2: unpack CPU job input activations (e4m3 → fp32, e8m0 → fp32) 8 at a time with vectors -------------------------------------------------
//   Scalar path (runtime.cpp moe_decode_experts unpack_row) = a 256-entry table lookup per element (e4m3_lut = e4m3_to_f32) + e8m0_to_f32 (ldexpf) per block — 5120 + 160
//   per row sent to the CPU in batch decode. Here the same values are produced by bit assembly (no table lookups, no libm):
//   e4m3 v = s|e(4)|m(3): e ≥ 1 and (e,m) ≠ (15,7) → fp32 bits = s<<31 | (v&0x7F)<<20 + 120<<23 (exponent e−7+127 = e+120, mantissa m<<20);
//   e = 0 → m·2^-9 (integer m converted to fp32 times 2^-9 — both exact); (e,m) = (15,7) → nanf("") bits 0x7FC00000; the sign is ORed in last (same bits as the scalar s ? -r : r —
//   ±0 and ±NaN included, all 256 codes checked: bench_expert_cpu). e8m0 b: b<<23, b = 0 → 0x00400000 (2^-127), b = 255 → 0x7FC00000 (nanf) — all 256 codes checked.
namespace {
inline __m256 e4m3x8(const uint8_t* q) {
  const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q)));
  const __m256i em = _mm256_and_si256(b, _mm256_set1_epi32(0x7F));
  const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24);
  const __m256i norm = _mm256_add_epi32(_mm256_slli_epi32(em, 20), _mm256_set1_epi32(120 << 23));
  const __m256i sub = _mm256_castps_si256(_mm256_mul_ps(_mm256_cvtepi32_ps(em), _mm256_set1_ps(1.0f / 512.0f)));
  __m256i v = _mm256_blendv_epi8(norm, sub, _mm256_cmpgt_epi32(_mm256_set1_epi32(8), em));        // em < 8 → subnormal (0 included)
  v = _mm256_blendv_epi8(v, _mm256_set1_epi32(0x7FC00000), _mm256_cmpeq_epi32(em, _mm256_set1_epi32(0x7F)));  // NaN
  return _mm256_castsi256_ps(_mm256_or_si256(v, sign));
}
inline __m256 e8m0x8_exact(const uint8_t* s) {
  const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s)));
  __m256i v = _mm256_blendv_epi8(_mm256_slli_epi32(b, 23), _mm256_set1_epi32(0x00400000), _mm256_cmpeq_epi32(b, _mm256_setzero_si256()));
  v = _mm256_blendv_epi8(v, _mm256_set1_epi32(0x7FC00000), _mm256_cmpeq_epi32(b, _mm256_set1_epi32(255)));
  return _mm256_castsi256_ps(v);
}
}  // namespace
void unpack_act_row_ref(const uint8_t* q, const uint8_t* s, int dim, float* a_f, float* a_s) {
  for (int d = 0; d < dim; ++d) a_f[d] = e4m3_to_f32(q[d]);
  for (int b = 0; b < dim / 32; ++b) a_s[b] = e8m0_to_f32(s[b]);
}
void unpack_act_row(const uint8_t* q, const uint8_t* s, int dim, float* a_f, float* a_s) {
  int d = 0;
  for (; d + 8 <= dim; d += 8) _mm256_storeu_ps(a_f + d, e4m3x8(q + d));
  for (; d < dim; ++d) a_f[d] = e4m3_to_f32(q[d]);
  const int nb = dim / 32;
  int b = 0;
  for (; b + 8 <= nb; b += 8) _mm256_storeu_ps(a_s + b, e8m0x8_exact(s + b));
  for (; b < nb; ++b) a_s[b] = e8m0_to_f32(s[b]);
}
bool unpack2_enabled() {  // HIVE_CPU_UNPACK2 — unset / "" / "0" = off. Read once per process.
  static const bool on = [] { const char* v = getenv("HIVE_CPU_UNPACK2"); return v && *v && strcmp(v, "0") != 0; }();
  return on;
}

void expert_forward(const ExpertDesc& e, const uint8_t* a_q, const uint8_t* a_s, float route_w, float swiglu_limit,
                    float* out, float* tmp) {
  HIVE_CHECK(e.fmt == ExpertFormat::E2M1_B32, "expert format");
  const int K = e.dim, I = e.inter;
  float* a_f = tmp;                 // [K]
  float* sa_f = a_f + K;            // [K/32]
  float* gate = sa_f + K / 32;      // [I]
  float* up = gate + I;             // [I]
  float* yq_f = up + I;             // [I]  (w2 input values)
  float* sy_f = yq_f + I;           // [I/32]
  for (int k = 0; k < K; ++k) a_f[k] = e4m3_to_f32(a_q[k]);
  for (int b = 0; b < K / 32; ++b) sa_f[b] = e8m0_to_f32(a_s[b]);
  gemv_e2m1_rows(e.w1, e.s1, I, K, a_f, sa_f, 0, I, gate);
  gemv_e2m1_rows(e.w3, e.s3, I, K, a_f, sa_f, 0, I, up);
  // GEMM output is bf16 → fp32 (output dtype of the reference linear())
  for (int i = 0; i < I; ++i) {
    float g = bf16_round(gate[i]), u = bf16_round(up[i]);
    if (swiglu_limit > 0.f) {
      u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
      g = fminf(g, swiglu_limit);
    }
    float v = silu(g) * u * route_w;
    yq_f[i] = bf16_round(v);  // y.to(bf16)
  }
  // act_quant(fp8, 32, e8m0) of yq
  for (int b = 0; b < I / 32; ++b) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(yq_f[b * 32 + i]));
    amax = fmaxf(amax, 1e-4f);
    const float sc = e8m0_to_f32(f32_ceil_pow2_e8m0(amax * (1.0f / 448.0f)));
    sy_f[b] = sc;
    for (int i = 0; i < 32; ++i) {
      float t = fminf(fmaxf(yq_f[b * 32 + i] / sc, -448.f), 448.f);
      yq_f[b * 32 + i] = e4m3_to_f32(f32_to_e4m3_host(t));
    }
  }
  gemv_e2m1_rows(e.w2, e.s2, K, I, yq_f, sy_f, 0, K, out);
  for (int k = 0; k < K; ++k) out[k] = bf16_round(out[k]);
}

}  // namespace hive::cpu
