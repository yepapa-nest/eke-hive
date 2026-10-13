// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// A/B measurement only: the CPU matmul as of 2026-10-09 19:30 (before the GEMV register blocking and the GEMM changes of that evening),
//   compiled into namespace hive::cpu::ab_legacy. HIVE_CPU_MATMUL_LEGACY=1 routes hive::cpu::matmul / matmul_multi here, so an engine's
//   own check binary can time old against new in one build. Results are bit-identical to the current kernels (tests/ab_cpu_matmul.cpp).
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hive/common.h"
#include "hive/cpu/cpu_backend.h"

namespace hive::cpu::ab_legacy {

namespace {

alignas(16) const uint8_t kNibF16Hi[16] = {0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46, 0x80, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6};

struct E4M3Lut {
  float v[256];
  E4M3Lut() { for (int i = 0; i < 256; ++i) v[i] = hive::e4m3_to_f32((uint8_t)i); }
};
const E4M3Lut& e4m3_lut() { static const E4M3Lut L; return L; }

inline __m256 dec_bf16(const uint8_t* p) {
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16));
}
inline __m256 dec_fp8(const uint8_t* p) {  // value / 2^8
  const __m128i b = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)p));
  const __m128i h = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(b, _mm_set1_epi16(0x7f)), 7), _mm_slli_epi16(_mm_and_si128(b, _mm_set1_epi16(0x80)), 8));
  return _mm256_cvtph_ps(h);
}
// 16 bytes → 32 values in element order (w[0..3] = elements 0-7, 8-15, 16-23, 24-31)
inline void dec_e2m1_32(const uint8_t* p, __m256 (&w)[4]) {
  const __m128i lut = _mm_load_si128((const __m128i*)kNibF16Hi);
  const __m128i raw = _mm_loadu_si128((const __m128i*)p);
  const __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0f));
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0f));
  const __m128i c0 = _mm_unpacklo_epi8(lo, hi), c1 = _mm_unpackhi_epi8(lo, hi);   // codes 0-15, 16-31
  const __m128i h0 = _mm_shuffle_epi8(lut, c0), h1 = _mm_shuffle_epi8(lut, c1);
  const __m128i z = _mm_setzero_si128();
  w[0] = _mm256_cvtph_ps(_mm_unpacklo_epi8(z, h0));
  w[1] = _mm256_cvtph_ps(_mm_unpackhi_epi8(z, h0));
  w[2] = _mm256_cvtph_ps(_mm_unpacklo_epi8(z, h1));
  w[3] = _mm256_cvtph_ps(_mm_unpackhi_epi8(z, h1));
}
inline __m256 dec_i8(const uint8_t* p) { return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)p))); }
inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}

// where row r of the matrix lives
struct RowPtr { const uint8_t* w; const uint8_t* s; };
inline RowPtr row_ptr(const Mat& m, int r) {
  const int k = (m.nodes > 1 && r >= m.row0[1]) ? 1 : 0;
  const int lr = r - m.row0[k];
  RowPtr p;
  p.w = m.w[k] + (size_t)lr * m.row_bytes();
  if (m.t == WType::FP8) p.s = m.s[k] + (size_t)(lr / m.bn) * m.scale_row_bytes();
  else p.s = m.s[k] ? m.s[k] + (size_t)lr * m.scale_row_bytes() : nullptr;
  return p;
}

// ---- GEMV: one weight row, M activation rows, K range [k0, k1) -------------------------------------------------------------------------------
// Accumulates into acc[M] (8 lanes each); k0 / k1 are multiples of the scale block.
// measured (3975WX, 2026-10-09, bench_cpu_backend): slicing K so the X rows stay in L1 and applying the FP4 block scales to the
// weights once (instead of once per activation row) were both slower at M = 4 / 8 (MoE layer 367 → 392-499 µs), so neither is done.
template <int M>
inline void row_part(const Mat& m, RowPtr rp, const float* X, int ldx, int k0, int k1, __m256* acc) {
  switch (m.t) {
    case WType::BF16: {
      __m256 acc2[M];
      for (int i = 0; i < M; ++i) acc2[i] = _mm256_setzero_ps();
      int k = k0;
      for (; k + 16 <= k1; k += 16) {
        const __m256 w0 = dec_bf16(rp.w + (size_t)k * 2), w1 = dec_bf16(rp.w + (size_t)(k + 8) * 2);
        for (int i = 0; i < M; ++i) {
          acc[i] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(X + (size_t)i * ldx + k), acc[i]);
          acc2[i] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(X + (size_t)i * ldx + k + 8), acc2[i]);
        }
      }
      for (; k < k1; k += 8) {
        const __m256 w0 = dec_bf16(rp.w + (size_t)k * 2);
        for (int i = 0; i < M; ++i) acc[i] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(X + (size_t)i * ldx + k), acc[i]);
      }
      for (int i = 0; i < M; ++i) acc[i] = _mm256_add_ps(acc[i], acc2[i]);
      return;
    }
    case WType::F32: {
      const float* w = reinterpret_cast<const float*>(rp.w);
      for (int k = k0; k < k1; k += 8) {
        const __m256 w0 = _mm256_loadu_ps(w + k);
        for (int i = 0; i < M; ++i) acc[i] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(X + (size_t)i * ldx + k), acc[i]);
      }
      return;
    }
    case WType::FP8: {
      const float* sc = reinterpret_cast<const float*>(rp.s);
      const int bk = m.bk;
      for (int kb = k0; kb < k1; kb += bk) {
        const int ke = std::min(k1, kb + bk);
        __m256 blk[M];
        for (int i = 0; i < M; ++i) blk[i] = _mm256_setzero_ps();
        for (int k = kb; k < ke; k += 8) {
          const __m256 w0 = dec_fp8(rp.w + k);
          for (int i = 0; i < M; ++i) blk[i] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(X + (size_t)i * ldx + k), blk[i]);
        }
        const __m256 s = _mm256_set1_ps(sc[kb / bk] * 256.f);
        for (int i = 0; i < M; ++i) acc[i] = _mm256_fmadd_ps(blk[i], s, acc[i]);
      }
      return;
    }
    case WType::I8: {
      const float* sc = reinterpret_cast<const float*>(rp.s);
      const int bk = m.bk;
      for (int kb = k0; kb < k1; kb += bk) {
        const int ke = std::min(k1, kb + bk);
        __m256 blk[M];
        for (int i = 0; i < M; ++i) blk[i] = _mm256_setzero_ps();
        for (int k = kb; k < ke; k += 8) {
          const __m256 w0 = dec_i8(rp.w + k);
          for (int i = 0; i < M; ++i) blk[i] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(X + (size_t)i * ldx + k), blk[i]);
        }
        const __m256 s = _mm256_set1_ps(sc[kb / bk]);
        for (int i = 0; i < M; ++i) acc[i] = _mm256_fmadd_ps(blk[i], s, acc[i]);
      }
      return;
    }
    case WType::NVFP4: {
      const float* L = e4m3_lut().v;
      for (int k = k0; k < k1; k += 32) {
        __m256 w[4];
        dec_e2m1_32(rp.w + k / 2, w);
        const __m256 s0 = _mm256_set1_ps(L[rp.s[k / 16]]), s1 = _mm256_set1_ps(L[rp.s[k / 16 + 1]]);
        {
          for (int i = 0; i < M; ++i) {
            const float* x = X + (size_t)i * ldx + k;
            const __m256 b0 = _mm256_fmadd_ps(w[1], _mm256_loadu_ps(x + 8), _mm256_mul_ps(w[0], _mm256_loadu_ps(x)));
            const __m256 b1 = _mm256_fmadd_ps(w[3], _mm256_loadu_ps(x + 24), _mm256_mul_ps(w[2], _mm256_loadu_ps(x + 16)));
            acc[i] = _mm256_fmadd_ps(b0, s0, acc[i]);
            acc[i] = _mm256_fmadd_ps(b1, s1, acc[i]);
          }
        }
      }
      return;
    }
    case WType::MXFP4: {
      for (int k = k0; k < k1; k += 32) {
        __m256 w[4];
        dec_e2m1_32(rp.w + k / 2, w);
        const __m256 s = _mm256_set1_ps(e8m0_to_f32(rp.s[k / 32]));
        {
          for (int i = 0; i < M; ++i) {
            const float* x = X + (size_t)i * ldx + k;
            __m256 b = _mm256_mul_ps(w[0], _mm256_loadu_ps(x));
            b = _mm256_fmadd_ps(w[1], _mm256_loadu_ps(x + 8), b);
            b = _mm256_fmadd_ps(w[2], _mm256_loadu_ps(x + 16), b);
            b = _mm256_fmadd_ps(w[3], _mm256_loadu_ps(x + 24), b);
            acc[i] = _mm256_fmadd_ps(b, s, acc[i]);
          }
        }
      }
      return;
    }
  }
}

template <int M>
void row_dot(const Mat& m, RowPtr rp, const float* X, int ldx, float gs, float* out /* M */) {
  __m256 acc[M];
  for (int i = 0; i < M; ++i) acc[i] = _mm256_setzero_ps();
  row_part<M>(m, rp, X, ldx, 0, m.K, acc);
  for (int i = 0; i < M; ++i) out[i] = hsum(acc[i]) * gs;
}

using RowFn = void (*)(const Mat&, RowPtr, const float*, int, float, float*);
RowFn row_fn(int M) {
  switch (M) {
    case 1: return &row_dot<1>; case 2: return &row_dot<2>; case 3: return &row_dot<3>; case 4: return &row_dot<4>;
    case 5: return &row_dot<5>; case 6: return &row_dot<6>; case 7: return &row_dot<7>; default: return &row_dot<8>;
  }
}

// rows per GEMV task: enough work per dispatch item, enough items per worker
int gemv_rows_per_task(const Mat& m, int rows) {
  const size_t rb = std::max<size_t>(1, m.row_bytes());
  int r = (int)std::max<size_t>(4, (64u << 10) / rb);   // ~64 KiB of weights per task
  const int workers = Pool::get().threads();
  while (r > 4 && (rows + r - 1) / r < 4 * workers) r /= 2;
  return std::max(1, r);
}

// ---- GEMM ------------------------------------------------------------------------------------------------------------------------------------
constexpr int kNR = 16, kMR = 6, kKC = 256, kNB = 64, kMB = 192;

// decode one weight row's k range [k0, k1) into dst (scales applied)
inline void decode_row(const Mat& m, int r, int k0, int k1, float* dst) {
  const float* L = e4m3_lut().v;
  const RowPtr rp = row_ptr(m, r);
  switch (m.t) {
    case WType::BF16:
      for (int k = k0; k < k1; k += 8) _mm256_store_ps(dst + (k - k0), dec_bf16(rp.w + (size_t)k * 2));
      break;
    case WType::F32:
      std::memcpy(dst, reinterpret_cast<const float*>(rp.w) + k0, (size_t)(k1 - k0) * 4);
      break;
    case WType::FP8: {
      const float* sc = reinterpret_cast<const float*>(rp.s);
      for (int k = k0; k < k1; k += 8)
        _mm256_store_ps(dst + (k - k0), _mm256_mul_ps(dec_fp8(rp.w + k), _mm256_set1_ps(sc[k / m.bk] * 256.f)));
      break;
    }
    case WType::NVFP4:
      for (int k = k0; k < k1; k += 32) {
        __m256 w[4];
        dec_e2m1_32(rp.w + k / 2, w);
        const __m256 s0 = _mm256_set1_ps(L[rp.s[k / 16]] * m.gscale), s1 = _mm256_set1_ps(L[rp.s[k / 16 + 1]] * m.gscale);
        _mm256_store_ps(dst + (k - k0), _mm256_mul_ps(w[0], s0)); _mm256_store_ps(dst + (k - k0) + 8, _mm256_mul_ps(w[1], s0));
        _mm256_store_ps(dst + (k - k0) + 16, _mm256_mul_ps(w[2], s1)); _mm256_store_ps(dst + (k - k0) + 24, _mm256_mul_ps(w[3], s1));
      }
      break;
    case WType::MXFP4:
      for (int k = k0; k < k1; k += 32) {
        __m256 w[4];
        dec_e2m1_32(rp.w + k / 2, w);
        const __m256 s = _mm256_set1_ps(e8m0_to_f32(rp.s[k / 32]));
        for (int q = 0; q < 4; ++q) _mm256_store_ps(dst + (k - k0) + 8 * q, _mm256_mul_ps(w[q], s));
      }
      break;
    case WType::I8: {
      const float* sc = reinterpret_cast<const float*>(rp.s);
      for (int k = k0; k < k1; k += 8) _mm256_store_ps(dst + (k - k0), _mm256_mul_ps(dec_i8(rp.w + k), _mm256_set1_ps(sc[k / m.bk])));
      break;
    }
  }
}

// 8×8 transpose of fp32 rows r[0..7] in place
inline void transpose8(__m256 (&r)[8]) {
  __m256 t[8];
  for (int i = 0; i < 4; ++i) { t[2 * i] = _mm256_unpacklo_ps(r[2 * i], r[2 * i + 1]); t[2 * i + 1] = _mm256_unpackhi_ps(r[2 * i], r[2 * i + 1]); }
  __m256 u[8];
  u[0] = _mm256_shuffle_ps(t[0], t[2], 0x44); u[1] = _mm256_shuffle_ps(t[0], t[2], 0xEE);
  u[2] = _mm256_shuffle_ps(t[1], t[3], 0x44); u[3] = _mm256_shuffle_ps(t[1], t[3], 0xEE);
  u[4] = _mm256_shuffle_ps(t[4], t[6], 0x44); u[5] = _mm256_shuffle_ps(t[4], t[6], 0xEE);
  u[6] = _mm256_shuffle_ps(t[5], t[7], 0x44); u[7] = _mm256_shuffle_ps(t[5], t[7], 0xEE);
  r[0] = _mm256_permute2f128_ps(u[0], u[4], 0x20); r[1] = _mm256_permute2f128_ps(u[1], u[5], 0x20);
  r[2] = _mm256_permute2f128_ps(u[2], u[6], 0x20); r[3] = _mm256_permute2f128_ps(u[3], u[7], 0x20);
  r[4] = _mm256_permute2f128_ps(u[0], u[4], 0x31); r[5] = _mm256_permute2f128_ps(u[1], u[5], 0x31);
  r[6] = _mm256_permute2f128_ps(u[2], u[6], 0x31); r[7] = _mm256_permute2f128_ps(u[3], u[7], 0x31);
}

// decode weight rows [n0, n0 + 16) (fewer at the end: zero rows) × k [k0, k1) into panel[(k − k0) · 16 + j], scales applied
//   (rows decoded 8 at a time and transposed in registers; the values are the decoded weights, unchanged)
void pack_panel(const Mat& m, int n0, int nrows, int k0, int k1, float* panel) {
  alignas(32) float tmp[8][kKC];
  const int kc = k1 - k0;
  for (int h = 0; h < 2; ++h) {
    for (int j = 0; j < 8; ++j) {
      const int jj = h * 8 + j;
      if (jj < nrows) decode_row(m, n0 + jj, k0, k1, tmp[j]);
      else std::memset(tmp[j], 0, (size_t)kc * 4);
    }
    int k = 0;
    for (; k + 8 <= kc; k += 8) {
      __m256 r[8];
      for (int j = 0; j < 8; ++j) r[j] = _mm256_load_ps(tmp[j] + k);
      transpose8(r);
      for (int q = 0; q < 8; ++q) _mm256_store_ps(panel + (size_t)(k + q) * kNR + h * 8, r[q]);
    }
    for (; k < kc; ++k) for (int j = 0; j < 8; ++j) panel[(size_t)k * kNR + h * 8 + j] = tmp[j][k];
  }
}

// C[mr rows, 16] (+)= X[mr, kc] · panel[kc][16]
//   Accumulators are named variables: with arrays (c0[MR], c1[MR]) GCC 13 kept a stack copy and stored all 12 accumulators on every k step
//   (perf: 2.25 IPC but ~0.65 FMA/cycle — store-port bound), half the FMA rate. Same FMAs in the same order per element.
template <int MR>
inline void micro(const float* X, int ldx, const float* panel, int kc, float* C, int ldc, int ncols, bool first) {
  __m256 a0 = _mm256_setzero_ps(), b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0, a4 = a0, b4 = a0, a5 = a0, b5 = a0;
  const float* x0 = X;
  const float* x1 = X + (size_t)(MR > 1 ? 1 : 0) * ldx;
  const float* x2 = X + (size_t)(MR > 2 ? 2 : 0) * ldx;
  const float* x3 = X + (size_t)(MR > 3 ? 3 : 0) * ldx;
  const float* x4 = X + (size_t)(MR > 4 ? 4 : 0) * ldx;
  const float* x5 = X + (size_t)(MR > 5 ? 5 : 0) * ldx;
  for (int k = 0; k < kc; ++k) {
    const __m256 p0 = _mm256_load_ps(panel + (size_t)k * kNR), p1 = _mm256_load_ps(panel + (size_t)k * kNR + 8);
    __m256 v;
    v = _mm256_broadcast_ss(x0 + k); a0 = _mm256_fmadd_ps(v, p0, a0); b0 = _mm256_fmadd_ps(v, p1, b0);
    if constexpr (MR > 1) { v = _mm256_broadcast_ss(x1 + k); a1 = _mm256_fmadd_ps(v, p0, a1); b1 = _mm256_fmadd_ps(v, p1, b1); }
    if constexpr (MR > 2) { v = _mm256_broadcast_ss(x2 + k); a2 = _mm256_fmadd_ps(v, p0, a2); b2 = _mm256_fmadd_ps(v, p1, b2); }
    if constexpr (MR > 3) { v = _mm256_broadcast_ss(x3 + k); a3 = _mm256_fmadd_ps(v, p0, a3); b3 = _mm256_fmadd_ps(v, p1, b3); }
    if constexpr (MR > 4) { v = _mm256_broadcast_ss(x4 + k); a4 = _mm256_fmadd_ps(v, p0, a4); b4 = _mm256_fmadd_ps(v, p1, b4); }
    if constexpr (MR > 5) { v = _mm256_broadcast_ss(x5 + k); a5 = _mm256_fmadd_ps(v, p0, a5); b5 = _mm256_fmadd_ps(v, p1, b5); }
  }
  auto put = [&](int i, __m256 lo, __m256 hi) {
    float* c = C + (size_t)i * ldc;
    if (ncols == kNR) {
      if (first) { _mm256_storeu_ps(c, lo); _mm256_storeu_ps(c + 8, hi); }
      else { _mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), lo)); _mm256_storeu_ps(c + 8, _mm256_add_ps(_mm256_loadu_ps(c + 8), hi)); }
      return;
    }
    alignas(32) float t[16];
    _mm256_store_ps(t, lo); _mm256_store_ps(t + 8, hi);
    if (first) for (int j = 0; j < ncols; ++j) c[j] = t[j];
    else for (int j = 0; j < ncols; ++j) c[j] += t[j];
  };
  put(0, a0, b0);
  if constexpr (MR > 1) put(1, a1, b1);
  if constexpr (MR > 2) put(2, a2, b2);
  if constexpr (MR > 3) put(3, a3, b3);
  if constexpr (MR > 4) put(4, a4, b4);
  if constexpr (MR > 5) put(5, a5, b5);
}
using MicroFn = void (*)(const float*, int, const float*, int, float*, int, int, bool);
MicroFn micro_fn(int mr) {
  switch (mr) {
    case 1: return &micro<1>; case 2: return &micro<2>; case 3: return &micro<3>; case 4: return &micro<4>; case 5: return &micro<5>; default: return &micro<6>;
  }
}

// previous microkernel (array accumulators) — A/B measurement only, with gemm_legacy
template <int MR>
inline void micro_legacy(const float* X, int ldx, const float* panel, int kc, float* C, int ldc, int ncols, bool first) {
  __m256 c0[MR], c1[MR];
  for (int i = 0; i < MR; ++i) { c0[i] = _mm256_setzero_ps(); c1[i] = _mm256_setzero_ps(); }
  for (int k = 0; k < kc; ++k) {
    const __m256 b0 = _mm256_load_ps(panel + (size_t)k * kNR), b1 = _mm256_load_ps(panel + (size_t)k * kNR + 8);
    for (int i = 0; i < MR; ++i) {
      const __m256 a = _mm256_broadcast_ss(X + (size_t)i * ldx + k);
      c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
      c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
    }
  }
  for (int i = 0; i < MR; ++i) {
    alignas(32) float t[16];
    _mm256_store_ps(t, c0[i]); _mm256_store_ps(t + 8, c1[i]);
    float* c = C + (size_t)i * ldc;
    if (first) for (int j = 0; j < ncols; ++j) c[j] = t[j];
    else for (int j = 0; j < ncols; ++j) c[j] += t[j];
  }
}

MicroFn micro_legacy_fn(int mr) {
  switch (mr) {
    case 1: return &micro_legacy<1>; case 2: return &micro_legacy<2>; case 3: return &micro_legacy<3>; case 4: return &micro_legacy<4>; case 5: return &micro_legacy<5>; default: return &micro_legacy<6>;
  }
}

// ---- previous dispatch (A/B measurement only: set_dispatch_legacy) -------------------------------------------------------------------------
// One GEMM per job, a task = (kNB weight rows, kMB activation rows) that packs its panels again for every row block, panels transposed one
//   value at a time. Same per-element arithmetic as gemm_task (the results are identical); kept to measure the new dispatch against.
std::atomic<bool> g_legacy{false};
void pack_panel_legacy(const Mat& m, int n0, int nrows, int k0, int k1, float* panel) {
  alignas(32) float tmp[kKC];
  for (int j = 0; j < kNR; ++j) {
    if (j >= nrows) { for (int k = k0; k < k1; ++k) panel[(size_t)(k - k0) * kNR + j] = 0.f; continue; }
    decode_row(m, n0 + j, k0, k1, tmp);
    for (int k = k0; k < k1; ++k) panel[(size_t)(k - k0) * kNR + j] = tmp[k - k0];
  }
}
void gemm_legacy(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy) {
  const Mat& m = *W.m;
  const int N = W.r1 - W.r0, K = m.K;
  const int nb = (N + kNB - 1) / kNB, mb = (M + kMB - 1) / kMB;
  Pool::get().run(nb * mb, [&](int t) {
    const int bi = t % nb, mi = t / nb;
    const int n0 = bi * kNB, n1 = std::min(N, n0 + kNB);
    const int m0 = mi * kMB, m1 = std::min(M, m0 + kMB);
    thread_local std::vector<float> panels;
    panels.resize((size_t)(kNB / kNR) * kKC * kNR + 8);
    float* pan = reinterpret_cast<float*>(((uintptr_t)panels.data() + 31) & ~(uintptr_t)31);
    for (int k0 = 0; k0 < K; k0 += kKC) {
      const int k1 = std::min(K, k0 + kKC), kc = k1 - k0;
      const int np = (n1 - n0 + kNR - 1) / kNR;
      for (int p = 0; p < np; ++p) pack_panel_legacy(m, W.r0 + n0 + p * kNR, std::min(kNR, n1 - n0 - p * kNR), k0, k1, pan + (size_t)p * kKC * kNR);
      for (int p = 0; p < np; ++p) {
        const int nc = std::min(kNR, n1 - n0 - p * kNR);
        for (int i = m0; i < m1; i += kMR) {
          const int mr = std::min(kMR, m1 - i);
          micro_legacy_fn(mr)(X + (size_t)i * ldx + k0, ldx, pan + (size_t)p * kKC * kNR, kc, Y + (size_t)i * ldy + n0 + p * kNR, ldy, nc, k0 == 0);
        }
      }
    }
  });
}

// ---- one dispatch for any set of jobs ------------------------------------------------------------------------------------------------------
// Every job keeps the kernel its own M selects (GEMV for M ≤ 8, GEMM above) and every output element its summation order, so a job's results
//   do not depend on what else runs in the same dispatch or on how its rows are split into tasks:
//   GEMV — per weight row, the M activation rows accumulate independently (row_dot<M> applies the same operations to each row).
//   GEMM — per element, K slices of kKC in ascending order (the first slice writes, the next ones add), k ascending inside a slice.
// Tasks are node-local (a task's weight rows live on the node whose workers run it); a GEMM task = kNB weight rows × a chunk of activation
//   rows: for each K slice it packs the slice's panels once and sweeps them over the whole chunk (in sub-blocks of kMB rows).
// weight rows per GEMM task (multiple of kNR): every task reads its activation chunk once per K slice, so wider tasks read X fewer times.
//   Default: the widest of 256 / 128 / 64 that still gives the dispatch ≥ 3 tasks per worker (bench_gemm_fn, 2026-10-09, M = 2048:
//   FN prefill shapes 565 → 526 ms per layer set at 256 against 64, but a 324-row matrix 15.6 → 26.4 ms — too few tasks). HIVE_CPU_GEMM_NB
//   fixes it.
const int kTaskRowsEnv = getenv("HIVE_CPU_GEMM_NB") ? std::max(kNR, atoi(getenv("HIVE_CPU_GEMM_NB")) / kNR * kNR) : 0;
constexpr int kTaskRowsMax = 256;
const bool kPackX = getenv("HIVE_CPU_GEMM_PACKX") && atoi(getenv("HIVE_CPU_GEMM_PACKX")) > 0;
const int kChunkMin = getenv("HIVE_CPU_GEMM_CHUNK") ? std::max(kMB, atoi(getenv("HIVE_CPU_GEMM_CHUNK"))) : kMB;
const int kTasksPerWorker = getenv("HIVE_CPU_GEMM_TPW") ? std::max(1, atoi(getenv("HIVE_CPU_GEMM_TPW"))) : 6;
struct Job { const Mat* m; int r0, r1; float* Y; int ldy; const float* X; int M, ldx; int rpt; };
struct Task { int job; int a, b; int m0, m1; };  // weight rows [a, b) (absolute), activation rows [m0, m1) (GEMM) or the whole M (GEMV)

void gemm_task(const Job& j, const Task& t) {
  const Mat& m = *j.m;
  const int K = m.K, nrows = t.b - t.a;
  const int np = (nrows + kNR - 1) / kNR;
  thread_local std::vector<float> panels;
  panels.resize((size_t)(std::max(kTaskRowsMax, kTaskRowsEnv) / kNR) * kKC * kNR + 8);
  float* pan = reinterpret_cast<float*>(((uintptr_t)panels.data() + 31) & ~(uintptr_t)31);
  const int n0 = t.a - j.r0;  // first output column of the task
  // K slice outer: the slice's panels are packed once (L2-sized) and swept over every activation row of the chunk
  for (int k0 = 0; k0 < K; k0 += kKC) {
    const int k1 = std::min(K, k0 + kKC), kc = k1 - k0;
    for (int p = 0; p < np; ++p) pack_panel(m, t.a + p * kNR, std::min(kNR, nrows - p * kNR), k0, k1, pan + (size_t)p * kKC * kNR);
    for (int mb = t.m0; mb < t.m1; mb += kMB) {
      const int me = std::min(t.m1, mb + kMB);
      // optionally copy the activation block [mb, me) × [k0, k1) to a contiguous buffer first (same values; measurement switch)
      const float* xb = j.X + (size_t)mb * j.ldx + k0;
      int ldb = j.ldx;
      if (kPackX) {
        thread_local std::vector<float> xbuf;
        xbuf.resize((size_t)kMB * kKC);
        for (int i = mb; i < me; ++i) std::memcpy(xbuf.data() + (size_t)(i - mb) * kc, j.X + (size_t)i * j.ldx + k0, (size_t)kc * 4);
        xb = xbuf.data(); ldb = kc;
      }
      for (int p = 0; p < np; ++p) {
        const int nc = std::min(kNR, nrows - p * kNR);
        for (int i = mb; i < me; i += kMR) {
          const int mr = std::min(kMR, me - i);
          micro_fn(mr)(xb + (size_t)(i - mb) * ldb, ldb, pan + (size_t)p * kKC * kNR, kc, j.Y + (size_t)i * j.ldy + n0 + p * kNR, j.ldy, nc, k0 == 0);
        }
      }
    }
  }
}

void gemv_task(const Job& j, const Task& t) {
  const RowFn fn = row_fn(j.M);
  const float gs = j.m->t == WType::NVFP4 ? j.m->gscale : 1.f;
  float out[8];
  for (int r = t.a; r < t.b; ++r) {
    fn(*j.m, row_ptr(*j.m, r), j.X, j.ldx, gs, out);
    for (int i = 0; i < j.M; ++i) j.Y[(size_t)i * j.ldy + (r - j.r0)] = out[i];
  }
}

void run_jobs(std::vector<Job>& jobs) {
  Pool& pool = Pool::get();
  const int workers = pool.threads();
  // GEMM: N blocks per node part of every job, then activation-row chunks so the dispatch has a few tasks per worker
  int gemv_rows = 0;
  for (const Job& j : jobs) if (j.M <= 8) gemv_rows += j.r1 - j.r0;
  // GEMM task width: per node part, blocks × the most activation chunks a job can have
  auto gemm_blocks_of = [&](int nb, bool with_chunks) {
    long b = 0;
    for (const Job& j : jobs) {
      if (j.M <= 8) continue;
      const Mat& m = *j.m;
      for (int k = 0; k < std::max(1, m.nodes); ++k) {
        const int lo = std::max(j.r0, m.nodes > 1 ? m.row0[k] : 0), hi = std::min(j.r1, m.nodes > 1 ? m.row0[k + 1] : m.N);
        if (hi > lo) b += (long)((hi - lo + nb - 1) / nb) * (with_chunks ? (j.M + kChunkMin - 1) / kChunkMin : 1);
      }
    }
    return b;
  };
  int kTaskRows = kTaskRowsEnv;
  if (!kTaskRows) {
    kTaskRows = 64;
    for (int nb : {256, 128}) if (gemm_blocks_of(nb, true) >= 3L * workers) { kTaskRows = nb; break; }
  }
  const int gemm_blocks = (int)std::min<long>(gemm_blocks_of(kTaskRows, false), 1 << 30);
  for (Job& j : jobs) if (j.M <= 8) j.rpt = gemv_rows_per_task(*j.m, std::max(gemv_rows, j.r1 - j.r0));
  std::vector<Task> tasks[2];
  for (int ji = 0; ji < (int)jobs.size(); ++ji) {
    const Job& j = jobs[ji];
    const Mat& m = *j.m;
    // activation-row chunks: at least kChunkMin rows (packed panels reused over ≥ 4 sub-blocks), enough tasks for an even spread
    int chunks = 1;
    if (j.M > 8 && gemm_blocks > 0)
      chunks = std::max(1, std::min((kTasksPerWorker * workers + gemm_blocks - 1) / gemm_blocks, (j.M + kChunkMin - 1) / kChunkMin));
    for (int k = 0; k < std::max(1, m.nodes); ++k) {
      const int lo = std::max(j.r0, m.nodes > 1 ? m.row0[k] : 0), hi = std::min(j.r1, m.nodes > 1 ? m.row0[k + 1] : m.N);
      if (hi <= lo) continue;
      if (j.M <= 8) {
        for (int a = lo; a < hi; a += j.rpt) tasks[k].push_back(Task{ji, a, std::min(hi, a + j.rpt), 0, j.M});
      } else {
        const int per = (j.M + chunks - 1) / chunks;
        for (int a = lo; a < hi; a += kTaskRows)
          for (int m0 = 0; m0 < j.M; m0 += per) tasks[k].push_back(Task{ji, a, std::min(hi, a + kTaskRows), m0, std::min(j.M, m0 + per)});
      }
    }
  }
  auto exec = [&](const Task& t) {
    const Job& j = jobs[t.job];
    if (j.M > 8) gemm_task(j, t); else gemv_task(j, t);
  };
  const int n0 = (int)tasks[0].size(), n1 = (int)tasks[1].size();
  if (n0 + n1 == 0) return;
  if (pool.nodes() == 1 || n0 == 0 || n1 == 0) {  // one node holds every row: any worker may run any task
    const std::vector<Task>& all = n0 ? tasks[0] : tasks[1];
    if (n0 && n1) {
      pool.run(n0 + n1, [&](int i) { exec(i < n0 ? tasks[0][i] : tasks[1][i - n0]); });
    } else {
      pool.run((int)all.size(), [&](int i) { exec(all[i]); });
    }
    return;
  }
  pool.run_nodes({n0, n1}, [&](int k, int i) { exec(tasks[k][i]); });
}

}  // namespace

void set_dispatch_legacy(bool on) { g_legacy = on; }

void matmul(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy) {
  if (M <= 0 || W.r1 <= W.r0) return;
  if (M > 8 && g_legacy) { gemm_legacy(W, X, M, ldx, Y, ldy); return; }
  std::vector<Job> jobs{Job{W.m, W.r0, W.r1, Y, ldy, X, M, ldx, 0}};
  run_jobs(jobs);
}

void matmul_multi(const MatJob* jobs_in, int n_jobs, const float* X, int M, int ldx) {
  std::vector<Job> jobs;
  jobs.reserve(n_jobs);
  for (int i = 0; i < n_jobs; ++i) {
    const MatJob& q = jobs_in[i];
    const int m = q.M > 0 ? q.M : M;
    if (m <= 0 || q.W.r1 <= q.W.r0) continue;
    if (m > 8 && g_legacy) { gemm_legacy(q.W, q.X ? q.X : X, m, q.ldx > 0 ? q.ldx : ldx, q.Y, q.ldy); continue; }
    jobs.push_back(Job{q.W.m, q.W.r0, q.W.r1, q.Y, q.ldy, q.X ? q.X : X, m, q.ldx > 0 ? q.ldx : ldx, 0});
  }
  if (!jobs.empty()) run_jobs(jobs);
}

}  // namespace hive::cpu::ab_legacy
