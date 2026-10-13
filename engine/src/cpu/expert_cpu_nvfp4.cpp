// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// NVFP4 CPU expert kernels (AVX2 + FMA + F16C) — see hive/expert_cpu.h. Kept apart from the E2M1_B32 kernels so that the DeepSeek
//   path is untouched.
#include <immintrin.h>

#include <cmath>
#include <cstdlib>
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
void gemv_nvfp4_rows_ref(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* a, int n0, int n1, float* y) {
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

void gemv_nvfp4_rows_multi_ref(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* const* a, int R, int n0, int n1,
                               float* const* y) {
  HIVE_CHECK(K % 32 == 0 && R >= 1 && R <= 8, "nvfp4 multi gemv");
  if (R == 1) { gemv_nvfp4_rows_ref(W, sw, gscale, N, K, a[0], n0, n1, y[0]); return; }
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

// ---- _v2: faster kernels, same bits -------------------------------------------------------------------------------------------------------------------
// Where the time went (engine/tests/bench_nvfp4_pool_cpu.cpp: the GlmExperts::cpu_experts task shape — 2 NUMA nodes × 16 pinned threads,
//   node-local half records, phase 1 gate+up / phase 2 down, J experts × R rows per MoE layer, 128→32-row chunks; 3975WX, read ceiling
//   ~120 GB/s; kernels interleaved in one process, median of 3 — separate runs on the shared machine differed by up to ±25 %):
//   _ref at R = 1 is memory bound; at R ≥ 2 it is compute bound: acc[r] of a runtime R lives on the stack (a load + store per chain step)
//   and every (row, block) also pays the cvtph unpack share. GLM R=2 55-83, R=3 53-68, R=4 43-61, R=8 14-31 GB/s.
//   Tiling K so the R activation rows stay in L1 (tried first, like the e2m1 _v3) LOST 5-15 % at R ≤ 4: DRAM streaming of whole weight
//   rows is the limit, and jumping between rows per tile disturbs the hardware prefetcher — so each weight row is walked once, whole.
//   A pre-scaled variant (w·e4m3 scale folded into the decoded weights, 4 fma per (row, block), fp32 reassociation) was within ±2 % of
//   this _v2 (R=2, 3: −6 … +2 %) once _v2 kept its accumulators in registers — not kept (no gain, not bit-identical).
// _v2 keeps the _ref operation chain for every output (n, r) — per 32-column block in ascending order
//   lo = w0·a[0:8] → fma(w2, a[8:16], lo) · hi = w1·a[16:24] → fma(w3, a[24:32], hi) · acc = fma(lo, e4m3(s[2b]), acc) → fma(hi, e4m3(s[2b+1]), acc),
//   y = hsum256(acc) · gscale — and changes only how the work is issued: (a) nibble → fp32 by byte shuffles (the expert_cpu.cpp _v2 unpack:
//   same layout, same values for all 16 codes, no fp16 convert), (b) R = 1: 4 weight rows per pass (4 independent fma chains, activations
//   loaded once per block), (c) R ≥ 2: R is a template constant — accumulators in registers, the block's weights decoded once for all R rows.
//   tests/test_nvfp4_cpu.cpp compares all output bits _v2 vs _ref (GLM / FN half shapes, R = 1..8, odd row ranges).
//   Pool bench GB/s _ref → _v2 (J = 1..8 experts per layer):  GLM (4096/2048) R=1 75-108 → 78-107 · R=2 55-83 → 82-111 · R=3 53-68 → 64-97 ·
//   R=4 43-61 → 50-88 · R=5 32-61 → 35-71 · R=6 24-48 → 24-52 · R=8 14-31 → 25-44.  FN (2560/640) R=1 56-86 → 63-89 · R=2 47-64 → 53-88 ·
//   R=3 40-55 → 54-81 · R=4 37-49 → 46-67 · R=5 27-46 → 30-56 · R=8 17-31 → 20-34.
namespace {
alignas(32) const uint8_t kV2IdxLo[32] = {0x00, 0x80, 0x01, 0x80, 0x08, 0x80, 0x09, 0x80, 0x04, 0x80, 0x05, 0x80, 0x0C, 0x80, 0x0D, 0x80,
                                          0x02, 0x80, 0x03, 0x80, 0x0A, 0x80, 0x0B, 0x80, 0x06, 0x80, 0x07, 0x80, 0x0E, 0x80, 0x0F, 0x80};
alignas(32) const uint8_t kV2IdxHi[32] = {0x80, 0x00, 0x80, 0x01, 0x80, 0x08, 0x80, 0x09, 0x80, 0x04, 0x80, 0x05, 0x80, 0x0C, 0x80, 0x0D,
                                          0x80, 0x02, 0x80, 0x03, 0x80, 0x0A, 0x80, 0x0B, 0x80, 0x06, 0x80, 0x07, 0x80, 0x0E, 0x80, 0x0F};
alignas(32) const uint8_t kV2B3[32] = {0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0,
                                       0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};
alignas(32) const uint8_t kV2B2[32] = {0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
                                       0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
struct NvTabs {
  __m256i lo, hi, b3, b2, m;
  NvTabs() : lo(_mm256_load_si256(reinterpret_cast<const __m256i*>(kV2IdxLo))), hi(_mm256_load_si256(reinterpret_cast<const __m256i*>(kV2IdxHi))),
             b3(_mm256_load_si256(reinterpret_cast<const __m256i*>(kV2B3))), b2(_mm256_load_si256(reinterpret_cast<const __m256i*>(kV2B2))),
             m(_mm256_set1_epi8(0x0F)) {}
};
// 16 nibble bytes → w0 = elements 0-7, w1 = 16-23, w2 = 8-15, w3 = 24-31 (the nib_to_f32x4 layout and values)
inline void nib4_v2(const uint8_t* wp, const NvTabs& t, __m256& w0, __m256& w1, __m256& w2, __m256& w3) {
  const __m256i raw = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wp)));
  const __m256i lo = _mm256_and_si256(raw, t.m), hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), t.m);
  const __m256i x = _mm256_or_si256(_mm256_shuffle_epi8(lo, t.lo), _mm256_shuffle_epi8(hi, t.hi));
  const __m256i b3 = _mm256_shuffle_epi8(t.b3, x), b2 = _mm256_shuffle_epi8(t.b2, x);
  const __m256i wl = _mm256_unpacklo_epi8(b2, b3), wh = _mm256_unpackhi_epi8(b2, b3);
  const __m256i z = _mm256_setzero_si256();
  w0 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(z, wl));
  w1 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(z, wl));
  w2 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(z, wh));
  w3 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(z, wh));
}
// one block of one activation row into acc — the _ref chain
inline __m256 nv_blk(const __m256& w0, const __m256& w1, const __m256& w2, const __m256& w3, const float* ap, const __m256& sl, const __m256& sh,
                     __m256 acc) {
  __m256 lo = _mm256_mul_ps(w0, _mm256_loadu_ps(ap));
  lo = _mm256_fmadd_ps(w2, _mm256_loadu_ps(ap + 8), lo);
  __m256 hi = _mm256_mul_ps(w1, _mm256_loadu_ps(ap + 16));
  hi = _mm256_fmadd_ps(w3, _mm256_loadu_ps(ap + 24), hi);
  acc = _mm256_fmadd_ps(lo, sl, acc);
  return _mm256_fmadd_ps(hi, sh, acc);
}

// NR weight rows × 1 activation row (_v2, R = 1)
template <int NR>
inline void rows1_v2(const uint8_t* W, const uint8_t* sw, float gscale, int K, const float* a, int n, float* y, const NvTabs& t, const float* L) {
  const int nb = K / 32;
  const uint8_t* wr[NR]; const uint8_t* sr[NR];
  __m256 acc[NR];
#pragma GCC unroll 4
  for (int i = 0; i < NR; ++i) { wr[i] = W + (size_t)(n + i) * (K / 2); sr[i] = sw + (size_t)(n + i) * (K / 16); acc[i] = _mm256_setzero_ps(); }
  for (int b = 0; b < nb; ++b) {
    const float* ap = a + b * 32;
    const __m256 a0 = _mm256_loadu_ps(ap), a1 = _mm256_loadu_ps(ap + 8), a2 = _mm256_loadu_ps(ap + 16), a3 = _mm256_loadu_ps(ap + 24);
#pragma GCC unroll 4
    for (int i = 0; i < NR; ++i) {
      __m256 w0, w1, w2, w3;
      nib4_v2(wr[i] + b * 16, t, w0, w1, w2, w3);
      __m256 lo = _mm256_mul_ps(w0, a0);
      lo = _mm256_fmadd_ps(w2, a1, lo);
      __m256 hi = _mm256_mul_ps(w1, a2);
      hi = _mm256_fmadd_ps(w3, a3, hi);
      acc[i] = _mm256_fmadd_ps(lo, _mm256_broadcast_ss(L + sr[i][2 * b]), acc[i]);
      acc[i] = _mm256_fmadd_ps(hi, _mm256_broadcast_ss(L + sr[i][2 * b + 1]), acc[i]);
    }
  }
#pragma GCC unroll 4
  for (int i = 0; i < NR; ++i) y[n + i] = hsum256(acc[i]) * gscale;
}

// G activation rows (a[0..G)) over one whole weight row into acc[0..G) (_v2)
template <int G>
inline void row_v2(const uint8_t* wr, const uint8_t* sr, int nb, const float* const* a, __m256* acc, const NvTabs& t, const float* L) {
  for (int b = 0; b < nb; ++b) {
    __m256 w0, w1, w2, w3;
    nib4_v2(wr + b * 16, t, w0, w1, w2, w3);
    const __m256 sl = _mm256_broadcast_ss(L + sr[2 * b]), sh = _mm256_broadcast_ss(L + sr[2 * b + 1]);
#pragma GCC unroll 8
    for (int r = 0; r < G; ++r) acc[r] = nv_blk(w0, w1, w2, w3, a[r] + b * 32, sl, sh, acc[r]);
  }
}
// R activation rows (2..8). R ≤ 6: one pass (R accumulators + 4 weights + 2 scales in the 16 ymm registers); R = 7, 8: two passes over the same
//   weight row (the second reads it from L1) — pbench GLM R=5 single 74 vs split 59 GB/s, R=6 47-55 vs 51, R=7 34-37 vs 46, R=8 14-28 vs 23-37 (spills).
constexpr int kSplitV2 = 6;
template <int R>
void multi_v2(const uint8_t* W, const uint8_t* sw, float gscale, int K, const float* const* a, int n0, int n1, float* const* y, const NvTabs& t,
              const float* L) {
  constexpr int G0 = R <= kSplitV2 ? R : (R + 1) / 2, G1 = R - G0;
  const int nb = K / 32;
  for (int n = n0; n < n1; ++n) {
    const uint8_t* wr = W + (size_t)n * (K / 2);
    const uint8_t* sr = sw + (size_t)n * (K / 16);
    __m256 acc[G0];
#pragma GCC unroll 8
    for (int r = 0; r < G0; ++r) acc[r] = _mm256_setzero_ps();
    row_v2<G0>(wr, sr, nb, a, acc, t, L);
#pragma GCC unroll 8
    for (int r = 0; r < G0; ++r) y[r][n] = hsum256(acc[r]) * gscale;
    if constexpr (G1 > 0) {
      __m256 acc1[G1 > 0 ? G1 : 1];
#pragma GCC unroll 8
      for (int r = 0; r < G1; ++r) acc1[r] = _mm256_setzero_ps();
      row_v2<G1>(wr, sr, nb, a + G0, acc1, t, L);
#pragma GCC unroll 8
      for (int r = 0; r < G1; ++r) y[G0 + r][n] = hsum256(acc1[r]) * gscale;
    }
  }
}

}  // namespace

void gemv_nvfp4_rows_v2(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* a, int n0, int n1, float* y) {
  HIVE_CHECK(K % 32 == 0, "nvfp4 gemv needs K % 32 == 0");
  (void)N;
  const NvTabs t;
  const float* L = e4m3_lut().v;
  int n = n0;
  for (; n + 4 <= n1; n += 4) rows1_v2<4>(W, sw, gscale, K, a, n, y, t, L);
  for (; n < n1; ++n) rows1_v2<1>(W, sw, gscale, K, a, n, y, t, L);
}

#define HIVE_NV_DISPATCH(fn)                                                \
  switch (R) {                                                              \
    case 2: fn<2>(W, sw, gscale, K, a, n0, n1, y, t, L); return;            \
    case 3: fn<3>(W, sw, gscale, K, a, n0, n1, y, t, L); return;            \
    case 4: fn<4>(W, sw, gscale, K, a, n0, n1, y, t, L); return;            \
    case 5: fn<5>(W, sw, gscale, K, a, n0, n1, y, t, L); return;            \
    case 6: fn<6>(W, sw, gscale, K, a, n0, n1, y, t, L); return;            \
    case 7: fn<7>(W, sw, gscale, K, a, n0, n1, y, t, L); return;            \
    default: fn<8>(W, sw, gscale, K, a, n0, n1, y, t, L); return;           \
  }
void gemv_nvfp4_rows_multi_v2(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* const* a, int R, int n0, int n1,
                              float* const* y) {
  HIVE_CHECK(K % 32 == 0 && R >= 1 && R <= 8, "nvfp4 multi gemv");
  if (R == 1) { gemv_nvfp4_rows_v2(W, sw, gscale, N, K, a[0], n0, n1, y[0]); return; }
  const NvTabs t;
  const float* L = e4m3_lut().v;
  HIVE_NV_DISPATCH(multi_v2)
}
#undef HIVE_NV_DISPATCH

// HIVE_CPU_NVFP4_V2: unset / anything but "0" = _v2 (bit-identical to _ref) · "0" = the _ref kernels (A/B, fallback). Read once per process.
bool nvfp4_v2_enabled() {
  static const bool on = [] { const char* v = getenv("HIVE_CPU_NVFP4_V2"); return !(v && strcmp(v, "0") == 0); }();
  return on;
}
void gemv_nvfp4_rows(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* a, int n0, int n1, float* y) {
  if (nvfp4_v2_enabled()) gemv_nvfp4_rows_v2(W, sw, gscale, N, K, a, n0, n1, y);
  else gemv_nvfp4_rows_ref(W, sw, gscale, N, K, a, n0, n1, y);
}
void gemv_nvfp4_rows_multi(const uint8_t* W, const uint8_t* sw, float gscale, int N, int K, const float* const* a, int R, int n0, int n1,
                           float* const* y) {
  if (nvfp4_v2_enabled()) gemv_nvfp4_rows_multi_v2(W, sw, gscale, N, K, a, R, n0, n1, y);
  else gemv_nvfp4_rows_multi_ref(W, sw, gscale, N, K, a, R, n0, n1, y);
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
