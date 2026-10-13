// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU matmul kernels (AVX2 + FMA + F16C) — see hive/cpu/cpu_backend.h.
//
// Weight decoding, 8 values at a time into fp32 (exact for every format):
//   BF16  — widen to 32 bits and shift left 16.
//   FP8   — e4m3 byte b → fp16 bits ((b & 0x7f) << 7) | ((b & 0x80) << 8), F16C conversion, × 2^8 (folded into the block scale): the e4m3
//           exponent bias 7 vs fp16's 15 is a factor 2^8 for normals and subnormals alike (an e4m3 subnormal lands on an fp16 subnormal).
//   e2m1  — 16 bytes = 32 nibbles in element order (low nibble first), mapped to fp16 high bytes by a byte shuffle (the code table of
//           hive's expert kernels), F16C conversion.
// GEMV (M ≤ 8): a task = a block of consecutive weight rows of one node; a kernel call covers R weight rows × G activation rows (gv_shape:
//   the shape measured best per format and M), decoding each weight vector once per pass and sharing each activation load over the R rows.
//   Per output element: Σ over scale blocks of (Σ in the block of x·w, 8 lanes in fixed order, then the lanes summed in a fixed tree) × scale
//   — fixed, independent of the kernel shape and the thread count.
// GEMM (M > 8): a task = (64 weight rows, up to 192 activation rows); for each K slice of 256 the 64 rows are decoded (scales applied) into a
//   packed fp32 panel [256][16] per 16 rows, then a 6×16 register tile runs FMAs over the slice. Y is written by the first slice and
//   accumulated by the next ones — the per-element order (slice by slice, k ascending inside) does not depend on the thread count.
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hive/common.h"
#include "hive/cpu/cpu_backend.h"

namespace hive::cpu {

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
// value / 2^8. Sign-extended to 16 bits and shifted left by 7, a byte s·e·m lands as s at bit 15, a copy of s at bit 14 and e·m at bits 7-13;
//   clearing bit 14 leaves the fp16 bits ((b & 0x7f) << 7) | ((b & 0x80) << 8) — 3 integer ops instead of 5.
inline __m256 dec_fp8(const uint8_t* p) {
  const __m128i b = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)p));
  return _mm256_cvtph_ps(_mm_and_si128(_mm_slli_epi16(b, 7), _mm_set1_epi16((short)0xBF80)));
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

// ---- GEMV: R weight rows × G activation rows per kernel call ---------------------------------------------------------------------------------
// Per output element the operations are fixed (they define the results; every kernel shape below applies exactly these to each (row, x) pair):
//   BF16  two chains: k in steps of 16, c += w[k..k+8)·x, d += w[k+8..k+16)·x; the K mod 16 tail into c; then c + d.
//   F32   one chain, steps of 8.
//   FP8 / I8  per scale block: b = Σ w·x (steps of 8, from zero), then c = fma(b, block scale, c).
//   NVFP4 per 32: b0 = fma(w8..15, x8..15, w0..7 · x0..7), c = fma(b0, s0, c); b1 likewise for 16..31 with s1, c = fma(b1, s1, c).
//   MXFP4 per 32: b = w0..7 · x0..7, then three fmas for 8..31, c = fma(b, s, c).
//   Then y = hsum(c) (fixed lane tree) × gscale (NVFP4) — independent of the kernel shape, the task split and the thread count.
// The kernel shape only changes how many accumulators live in registers at once: R weight rows (one x load serves R rows) × G activation
//   rows (one decoded weight vector serves G rows); M rows are covered by ⌈M / G⌉ passes over the (L1-resident) weight rows. The old kernel
//   (R = 1, G = M) kept 2M accumulators for BF16 / FP8 — beyond 16 ymm registers from M = 7 on, so GCC kept them on the stack.
struct GvRow { const uint8_t* w; const uint8_t* s; };

const float* e8m0_lut() {
  static const struct L { float v[256]; L() { for (int i = 0; i < 256; ++i) v[i] = e8m0_to_f32((uint8_t)i); } } l;
  return l.v;
}

// e2m1 elements [16h, 16h + 16) of a 32-element group (8 bytes at p) → 2 vectors of 8, the same values as dec_e2m1_32
inline void dec_e2m1_16(const uint8_t* p, __m256& w0, __m256& w1) {
  const __m128i lut = _mm_load_si128((const __m128i*)kNibF16Hi);
  const __m128i raw = _mm_loadl_epi64((const __m128i*)p);
  const __m128i m = _mm_set1_epi8(0x0f);
  const __m128i c = _mm_unpacklo_epi8(_mm_and_si128(raw, m), _mm_and_si128(_mm_srli_epi16(raw, 4), m));
  const __m128i h = _mm_shuffle_epi8(lut, c);
  w0 = _mm256_cvtph_ps(_mm_unpacklo_epi8(_mm_setzero_si128(), h));
  w1 = _mm256_cvtph_ps(_mm_unpackhi_epi8(_mm_setzero_si128(), h));
}

#define GV_FOR_R for (int r = 0; r < R; ++r)
#define GV_FOR_I for (int i = 0; i < G; ++i)
#define GV_U4 _Pragma("GCC unroll 4")
#define GV_U8 _Pragma("GCC unroll 8")

// One kernel call: R weight rows × G activation rows over the whole row (k in [k0, k1) = [0, K)).
struct GvArgs { const GvRow* rw; const float* X; int ldx; int k0, k1, bk; float gs; float* Y; int ldy; };

template <int R, int G, int S>
inline void gv_begin(const GvArgs&, __m256 (&c)[R][G], __m256 (*d)[G]) {
  GV_U4 GV_FOR_R GV_U8 GV_FOR_I { c[r][i] = _mm256_setzero_ps(); if constexpr (S > 1) d[r][i] = _mm256_setzero_ps(); }
}
template <int R, int G, int S>
inline void gv_end(const GvArgs& a, __m256 (&c)[R][G], __m256 (*d)[G]) {
  if constexpr (S > 1) { GV_U4 GV_FOR_R GV_U8 GV_FOR_I c[r][i] = _mm256_add_ps(c[r][i], d[r][i]); }
  GV_U8 GV_FOR_I GV_U4 GV_FOR_R a.Y[(size_t)i * a.ldy + r] = hsum(c[r][i]) * a.gs;
}

template <int R, int G>
void gv_bf16(const GvArgs& a) {
  __m256 c[R][G], d[R][G];
  gv_begin<R, G, 2>(a, c, d);
  const float* X = a.X;
  const int ldx = a.ldx, k1 = a.k1;
  int k = a.k0;
  for (; k + 16 <= k1; k += 16) {
    __m256 w[R];
    GV_U4 GV_FOR_R w[r] = dec_bf16(a.rw[r].w + (size_t)k * 2);
    GV_U8 GV_FOR_I {
      const __m256 x = _mm256_loadu_ps(X + (size_t)i * ldx + k);
      GV_U4 GV_FOR_R c[r][i] = _mm256_fmadd_ps(w[r], x, c[r][i]);
    }
    GV_U4 GV_FOR_R w[r] = dec_bf16(a.rw[r].w + (size_t)(k + 8) * 2);
    GV_U8 GV_FOR_I {
      const __m256 x = _mm256_loadu_ps(X + (size_t)i * ldx + k + 8);
      GV_U4 GV_FOR_R d[r][i] = _mm256_fmadd_ps(w[r], x, d[r][i]);
    }
  }
  for (; k < k1; k += 8) {  // K mod 16 tail
    __m256 w[R];
    GV_U4 GV_FOR_R w[r] = dec_bf16(a.rw[r].w + (size_t)k * 2);
    GV_U8 GV_FOR_I {
      const __m256 x = _mm256_loadu_ps(X + (size_t)i * ldx + k);
      GV_U4 GV_FOR_R c[r][i] = _mm256_fmadd_ps(w[r], x, c[r][i]);
    }
  }
  gv_end<R, G, 2>(a, c, d);
}

template <int R, int G>
void gv_f32(const GvArgs& a) {
  __m256 c[R][G];
  gv_begin<R, G, 1>(a, c, nullptr);
  for (int k = a.k0; k < a.k1; k += 8) {
    __m256 w[R];
    GV_U4 GV_FOR_R w[r] = _mm256_loadu_ps(reinterpret_cast<const float*>(a.rw[r].w) + k);
    GV_U8 GV_FOR_I {
      const __m256 x = _mm256_loadu_ps(a.X + (size_t)i * a.ldx + k);
      GV_U4 GV_FOR_R c[r][i] = _mm256_fmadd_ps(w[r], x, c[r][i]);
    }
  }
  gv_end<R, G, 1>(a, c, nullptr);
}

// FP8 (I8 = false) and I8 (I8 = true): per scale block of bk columns
template <int R, int G, bool I8>
void gv_blk(const GvArgs& a) {
  __m256 c[R][G];
  gv_begin<R, G, 1>(a, c, nullptr);
  const int bk = a.bk;
  for (int kb = a.k0; kb < a.k1; kb += bk) {
    const int ke = std::min(a.k1, kb + bk);
    __m256 b[R][G];
    GV_U4 GV_FOR_R GV_U8 GV_FOR_I b[r][i] = _mm256_setzero_ps();
    for (int k = kb; k < ke; k += 8) {
      __m256 w[R];
      GV_U4 GV_FOR_R w[r] = I8 ? dec_i8(a.rw[r].w + k) : dec_fp8(a.rw[r].w + k);
      GV_U8 GV_FOR_I {
        const __m256 x = _mm256_loadu_ps(a.X + (size_t)i * a.ldx + k);
        GV_U4 GV_FOR_R b[r][i] = _mm256_fmadd_ps(w[r], x, b[r][i]);
      }
    }
    GV_U4 GV_FOR_R {
      const float sv = reinterpret_cast<const float*>(a.rw[r].s)[kb / bk];
      const __m256 s = _mm256_set1_ps(I8 ? sv : sv * 256.f);
      GV_U8 GV_FOR_I c[r][i] = _mm256_fmadd_ps(b[r][i], s, c[r][i]);
    }
  }
  gv_end<R, G, 1>(a, c, nullptr);
}

// NVFP4: FULL decodes the 32 weights of a group at once (one 16-byte load, as the old kernel) when the registers allow it, otherwise in
//   two halves of 16 (two 8-byte loads) so that only 2 weight vectors per row are live.
template <int R, int G>
void gv_nvfp4(const GvArgs& a) {
  constexpr bool FULL = R * (G + 6) + 3 <= 16;
  const float* L = e4m3_lut().v;
  __m256 c[R][G];
  gv_begin<R, G, 1>(a, c, nullptr);
  for (int k = a.k0; k < a.k1; k += 32) {
    if constexpr (FULL) {
      __m256 w[R][4], s0[R], s1[R];
      GV_U4 GV_FOR_R {
        dec_e2m1_32(a.rw[r].w + k / 2, w[r]);
        s0[r] = _mm256_set1_ps(L[a.rw[r].s[k / 16]]); s1[r] = _mm256_set1_ps(L[a.rw[r].s[k / 16 + 1]]);
      }
      GV_U8 GV_FOR_I {
        const float* x = a.X + (size_t)i * a.ldx + k;
        GV_U4 GV_FOR_R {
          const __m256 b0 = _mm256_fmadd_ps(w[r][1], _mm256_loadu_ps(x + 8), _mm256_mul_ps(w[r][0], _mm256_loadu_ps(x)));
          const __m256 b1 = _mm256_fmadd_ps(w[r][3], _mm256_loadu_ps(x + 24), _mm256_mul_ps(w[r][2], _mm256_loadu_ps(x + 16)));
          c[r][i] = _mm256_fmadd_ps(b0, s0[r], c[r][i]);
          c[r][i] = _mm256_fmadd_ps(b1, s1[r], c[r][i]);
        }
      }
    } else {
      _Pragma("GCC unroll 2") for (int h = 0; h < 2; ++h) {
        __m256 w0[R], w1[R], s[R];
        GV_U4 GV_FOR_R {
          dec_e2m1_16(a.rw[r].w + k / 2 + 8 * h, w0[r], w1[r]);
          s[r] = _mm256_set1_ps(L[a.rw[r].s[k / 16 + h]]);
        }
        GV_U8 GV_FOR_I {
          const float* x = a.X + (size_t)i * a.ldx + k + 16 * h;
          const __m256 x0 = _mm256_loadu_ps(x), x1 = _mm256_loadu_ps(x + 8);
          GV_U4 GV_FOR_R {
            const __m256 b = _mm256_fmadd_ps(w1[r], x1, _mm256_mul_ps(w0[r], x0));
            c[r][i] = _mm256_fmadd_ps(b, s[r], c[r][i]);
          }
        }
      }
    }
  }
  gv_end<R, G, 1>(a, c, nullptr);
}

template <int R, int G>
void gv_mxfp4(const GvArgs& a) {
  const float* E = e8m0_lut();
  __m256 c[R][G];
  gv_begin<R, G, 1>(a, c, nullptr);
  for (int k = a.k0; k < a.k1; k += 32) {
    __m256 w[R][4], s[R];
    GV_U4 GV_FOR_R {
      dec_e2m1_32(a.rw[r].w + k / 2, w[r]);
      s[r] = _mm256_set1_ps(E[a.rw[r].s[k / 32]]);
    }
    GV_U8 GV_FOR_I {
      const float* x = a.X + (size_t)i * a.ldx + k;
      GV_U4 GV_FOR_R {
        __m256 b = _mm256_mul_ps(w[r][0], _mm256_loadu_ps(x));
        b = _mm256_fmadd_ps(w[r][1], _mm256_loadu_ps(x + 8), b);
        b = _mm256_fmadd_ps(w[r][2], _mm256_loadu_ps(x + 16), b);
        b = _mm256_fmadd_ps(w[r][3], _mm256_loadu_ps(x + 24), b);
        c[r][i] = _mm256_fmadd_ps(b, s[r], c[r][i]);
      }
    }
  }
  gv_end<R, G, 1>(a, c, nullptr);
}
#undef GV_FOR_R
#undef GV_FOR_I
#undef GV_U4
#undef GV_U8

using GvFn = void (*)(const GvArgs&);
constexpr int kGvMaxR = 4, kGvMaxG = 8;
template <int R, int G>
GvFn gv_fn_t(WType t) {
  switch (t) {
    case WType::BF16: return &gv_bf16<R, G>;
    case WType::F32: return &gv_f32<R, G>;
    case WType::FP8: return &gv_blk<R, G, false>;
    case WType::I8: return &gv_blk<R, G, true>;
    case WType::NVFP4: return &gv_nvfp4<R, G>;
    case WType::MXFP4: return &gv_mxfp4<R, G>;
  }
  return nullptr;
}
template <int R>
GvFn gv_fn_r(WType t, int g) {
  switch (g) {
    case 1: return gv_fn_t<R, 1>(t); case 2: return gv_fn_t<R, 2>(t); case 3: return gv_fn_t<R, 3>(t); case 4: return gv_fn_t<R, 4>(t);
    case 5: return gv_fn_t<R, 5>(t); case 6: return gv_fn_t<R, 6>(t); case 7: return gv_fn_t<R, 7>(t); default: return gv_fn_t<R, 8>(t);
  }
}
GvFn gv_fn(WType t, int r, int g) {
  switch (r) { case 1: return gv_fn_r<1>(t, g); case 2: return gv_fn_r<2>(t, g); case 3: return gv_fn_r<3>(t, g); default: return gv_fn_r<4>(t, g); }
}

}  // namespace
// Kernel shape per (format, M): R weight rows, G activation rows per pass (the passes split M as evenly as possible).
//   HIVE_CPU_GEMV_SHAPE="RxG" (or tune::gemv_force, measurement only) forces one shape for every format and M — results are identical for
//   every shape.
namespace tune {
int gemv_force_r = 0, gemv_force_g = 0;
const bool gemv_env_read = [] {
  if (const char* e = getenv("HIVE_CPU_GEMV_SHAPE")) if (sscanf(e, "%dx%d", &gemv_force_r, &gemv_force_g) != 2) gemv_force_r = gemv_force_g = 0;
  return true;
}();
}  // namespace tune
namespace {
struct GvShape { int r, g; };
// measured (3975WX, 32 workers, 2026-10-09, bench_cpu_matmul_ab shapes: FN / GLM / DS / 27B matrices, every R×G for M = 1..8):
//   the GEMV is memory-bound at M = 1 (~112 GB/s of the 121.7 GB/s node-local STREAM read) and turns compute- or L2-bound as M grows;
//   the cost per extra activation row is the weight decode (once per pass) plus the activation loads (shared by R weight rows, from L2
//   when M · K · 4 bytes exceed L1). Best shapes (µs, M = 8 unless noted, legacy = the 2026-10-09 kernel):
//   NVFP4 27B 17408×5120 1×4: 1087 (legacy 1254) · GLM 2048×4096 1×4: 129 (165) · FN 2560×640 (X in L1) 1×8: 47 (49); M ≤ 7: 1×M
//         (the whole 32-group decoded at once — the legacy order of work, without its spilled accumulators; R = 2 needs the half-group
//         decode and lost 12-15 % in the MoE decode layer at M = 1-3)
//   MXFP4 DS 2304×5120 2×8: 172 (285), M = 4 1×4: 95 (182); the e8m0 table instead of ldexpf halves M = 1 (67 vs 132)
//   FP8   GLM 1536×4096 (X 128 KiB at M = 8) 2×8: 86 (122); 12288×1536 (X 48 KiB) 1×8: 262 (268)
//   BF16  memory-bound up to M = 8 with every shape within ±3 %.
//   Slicing K so the activation slice stays in L1 was slower in every case (strided weight rows defeat the prefetchers: 1×1 at M = 1
//   on the 27B matrix 448 → 943 µs with 256-column slices), so the kernels always run over the whole row.
GvShape gv_shape(WType t, int M, int K) {
  if (tune::gemv_force_r > 0 && tune::gemv_force_g > 0)
    return GvShape{std::min(tune::gemv_force_r, kGvMaxR), std::min(tune::gemv_force_g, std::min(M, kGvMaxG))};
  const size_t xb = (size_t)M * K * 4;  // activation bytes
  switch (t) {
    case WType::NVFP4: return GvShape{1, M <= 7 || xb <= (32u << 10) ? M : 4};
    case WType::MXFP4: return GvShape{M <= 4 ? 1 : 2, M};
    case WType::FP8: case WType::I8: return GvShape{xb > (64u << 10) ? 2 : 1, M};
    case WType::BF16: return M <= 4 ? GvShape{2, M} : GvShape{4, 4};
    default: return GvShape{1, M};
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
  const float* E = e8m0_lut();
  const RowPtr rp = row_ptr(m, r);
  switch (m.t) {
    case WType::BF16:
      for (int k = k0; k < k1; k += 8) _mm256_store_ps(dst + (k - k0), dec_bf16(rp.w + (size_t)k * 2));
      break;
    case WType::F32:
      std::memcpy(dst, reinterpret_cast<const float*>(rp.w) + k0, (size_t)(k1 - k0) * 4);
      break;
    case WType::FP8: {  // per scale block (no integer division per 8 values)
      const float* sc = reinterpret_cast<const float*>(rp.s);
      for (int kb = k0; kb < k1;) {
        const int b = kb / m.bk, ke = std::min(k1, (b + 1) * m.bk);
        const __m256 s = _mm256_set1_ps(sc[b] * 256.f);
        for (int k = kb; k < ke; k += 8) _mm256_store_ps(dst + (k - k0), _mm256_mul_ps(dec_fp8(rp.w + k), s));
        kb = ke;
      }
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
        const __m256 s = _mm256_set1_ps(E[rp.s[k / 32]]);
        for (int q = 0; q < 4; ++q) _mm256_store_ps(dst + (k - k0) + 8 * q, _mm256_mul_ps(w[q], s));
      }
      break;
    case WType::I8: {
      const float* sc = reinterpret_cast<const float*>(rp.s);
      for (int kb = k0; kb < k1;) {
        const int b = kb / m.bk, ke = std::min(k1, (b + 1) * m.bk);
        const __m256 s = _mm256_set1_ps(sc[b]);
        for (int k = kb; k < ke; k += 8) _mm256_store_ps(dst + (k - k0), _mm256_mul_ps(dec_i8(rp.w + k), s));
        kb = ke;
      }
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

// ---- one dispatch for any set of jobs ------------------------------------------------------------------------------------------------------
// Every job keeps the kernel its own M selects (GEMV for M ≤ 8, GEMM above) and every output element its summation order, so a job's results
//   do not depend on what else runs in the same dispatch or on how its rows are split into tasks:
//   GEMV — per weight row, the M activation rows accumulate independently (every GEMV kernel shape applies the same operations to each row).
//   GEMM — per element, K slices of kKC in ascending order (the first slice writes, the next ones add), k ascending inside a slice.
// Tasks are node-local (a task's weight rows live on the node whose workers run it); a GEMM task = kNB weight rows × a chunk of activation
//   rows: for each K slice it packs the slice's panels once and sweeps them over the whole chunk (in sub-blocks of kMB rows).
// weight rows per GEMM task (multiple of kNR): every task reads its activation chunk once per K slice, so wider tasks read X fewer times.
//   Default: the widest of 256 / 128 / 64 that still gives the dispatch ≥ 3 tasks per worker (bench_gemm_fn, 2026-10-09, M = 2048:
//   FN prefill shapes 565 → 526 ms per layer set at 256 against 64, but a 324-row matrix 15.6 → 26.4 ms — too few tasks). HIVE_CPU_GEMM_NB
//   fixes it.
const int kTaskRowsEnv = getenv("HIVE_CPU_GEMM_NB") ? std::max(kNR, atoi(getenv("HIVE_CPU_GEMM_NB")) / kNR * kNR) : 0;
constexpr int kTaskRowsMax = 256;
const int kChunkMin = getenv("HIVE_CPU_GEMM_CHUNK") ? std::max(kMB, atoi(getenv("HIVE_CPU_GEMM_CHUNK"))) : kMB;
const int kTasksPerWorker = getenv("HIVE_CPU_GEMM_TPW") ? std::max(1, atoi(getenv("HIVE_CPU_GEMM_TPW"))) : 6;
struct Job { const Mat* m; int r0, r1; float* Y; int ldy; const float* X; int M, ldx; int rpt; };
struct Task { int job; int a, b; int m0, m1; };  // weight rows [a, b) (absolute), activation rows [m0, m1) (GEMM) or the whole M (GEMV)

}  // namespace
// GEMM loop variants (measurement switches; every variant applies the same operations to every element — slices ascending, the first writes):
//   gemm_mb    activation rows per sub-block (HIVE_CPU_GEMM_MB, default 192)
//   gemm_order 0: K slice outer — the slice's panels are packed once per task and swept over the whole chunk (Y of the chunk is revisited
//              once per slice); 1: sub-block outer — panels re-packed per sub-block, its Y tile stays in L2 across the slices
//              (HIVE_CPU_GEMM_ORDER)
//   gemm_iloop 0: panels outer, 6-row groups inner (a panel stays in L1); 1: 6-row groups outer (their X stays in L1) (HIVE_CPU_GEMM_ILOOP)
namespace tune {
int gemm_mb = getenv("HIVE_CPU_GEMM_MB") ? std::max(6, atoi(getenv("HIVE_CPU_GEMM_MB"))) : kMB;
int gemm_order = getenv("HIVE_CPU_GEMM_ORDER") ? atoi(getenv("HIVE_CPU_GEMM_ORDER")) : 0;
int gemm_iloop = getenv("HIVE_CPU_GEMM_ILOOP") ? atoi(getenv("HIVE_CPU_GEMM_ILOOP")) : 0;
}  // namespace tune
namespace {
// one K slice of a sub-block [mb, me): panels already packed
inline void gemm_block(const Job& j, const Task& t, const float* pan, int np, int nrows, int n0, int mb, int me, int k0, int kc) {
  const float* xb = j.X + (size_t)mb * j.ldx + k0;
  const int ldb = j.ldx;
  if (tune::gemm_iloop) {
    for (int i = mb; i < me; i += kMR) {
      const int mr = std::min(kMR, me - i);
      const MicroFn f = micro_fn(mr);
      for (int p = 0; p < np; ++p)
        f(xb + (size_t)(i - mb) * ldb, ldb, pan + (size_t)p * kKC * kNR, kc, j.Y + (size_t)i * j.ldy + n0 + p * kNR, j.ldy, std::min(kNR, nrows - p * kNR), k0 == 0);
    }
    return;
  }
  for (int p = 0; p < np; ++p) {
    const int nc = std::min(kNR, nrows - p * kNR);
    for (int i = mb; i < me; i += kMR) {
      const int mr = std::min(kMR, me - i);
      micro_fn(mr)(xb + (size_t)(i - mb) * ldb, ldb, pan + (size_t)p * kKC * kNR, kc, j.Y + (size_t)i * j.ldy + n0 + p * kNR, j.ldy, nc, k0 == 0);
    }
  }
  (void)t;
}

void gemm_task(const Job& j, const Task& t) {
  const Mat& m = *j.m;
  const int K = m.K, nrows = t.b - t.a;
  const int np = (nrows + kNR - 1) / kNR;
  thread_local std::vector<float> panels;
  panels.resize((size_t)(std::max(kTaskRowsMax, kTaskRowsEnv) / kNR) * kKC * kNR + 8);
  float* pan = reinterpret_cast<float*>(((uintptr_t)panels.data() + 31) & ~(uintptr_t)31);
  const int n0 = t.a - j.r0;  // first output column of the task
  const int MB = tune::gemm_mb;
  auto pack = [&](int k0, int k1) {
    for (int p = 0; p < np; ++p) pack_panel(m, t.a + p * kNR, std::min(kNR, nrows - p * kNR), k0, k1, pan + (size_t)p * kKC * kNR);
  };
  if (tune::gemm_order == 1) {
    for (int mb = t.m0; mb < t.m1; mb += MB)
      for (int k0 = 0; k0 < K; k0 += kKC) {
        const int k1 = std::min(K, k0 + kKC);
        pack(k0, k1);
        gemm_block(j, t, pan, np, nrows, n0, mb, std::min(t.m1, mb + MB), k0, k1 - k0);
      }
    return;
  }
  for (int k0 = 0; k0 < K; k0 += kKC) {
    const int k1 = std::min(K, k0 + kKC);
    pack(k0, k1);
    for (int mb = t.m0; mb < t.m1; mb += MB) gemm_block(j, t, pan, np, nrows, n0, mb, std::min(t.m1, mb + MB), k0, k1 - k0);
  }
}

void gemv_task(const Job& j, const Task& t) {
  const Mat& m = *j.m;
  const float gs = m.t == WType::NVFP4 ? m.gscale : 1.f;
  const GvShape sh = gv_shape(m.t, j.M, m.K);
  const int passes = (j.M + sh.g - 1) / sh.g;
  GvArgs a{nullptr, nullptr, j.ldx, 0, m.K, m.bk, gs, nullptr, j.ldy};
  for (int r = t.a; r < t.b; r += sh.r) {
    const int nr = std::min(sh.r, t.b - r);
    GvRow rw[kGvMaxR];
    for (int q = 0; q < nr; ++q) { const RowPtr p = row_ptr(m, r + q); rw[q] = GvRow{p.w, p.s}; }
    a.rw = rw;
    for (int p = 0, i0 = 0; p < passes; ++p) {  // activation rows split evenly: the first M mod passes passes take one more
      const int g = j.M / passes + (p < j.M % passes ? 1 : 0);
      a.X = j.X + (size_t)i0 * j.ldx;
      a.Y = j.Y + (size_t)i0 * j.ldy + (r - j.r0);
      gv_fn(m.t, nr, g)(a);
      i0 += g;
    }
  }
}

void run_jobs(std::vector<Job>& jobs) {
  Pool& pool = Pool::get();
  bool any_gemm = false;
  for (const Job& j : jobs) any_gemm |= j.M > 8;
  const int workers = pool.threads() + (any_gemm ? pool.smt_threads() : 0);
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
      pool.run(n0 + n1, [&](int i) { exec(i < n0 ? tasks[0][i] : tasks[1][i - n0]); }, any_gemm);
    } else {
      pool.run((int)all.size(), [&](int i) { exec(all[i]); }, any_gemm);
    }
    return;
  }
  pool.run_nodes({n0, n1}, [&](int k, int i) { exec(tasks[k][i]); }, any_gemm);
}

}  // namespace

namespace ab_legacy {  // cpu_matmul_legacy.cpp — A/B measurement only
void matmul(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy);
void matmul_multi(const MatJob* jobs_in, int n_jobs, const float* X, int M, int ldx);
}  // namespace ab_legacy
namespace {
// the whole matmul of the snapshot: HIVE_CPU_MATMUL_LEGACY=1 (process-wide) or set_dispatch_legacy(true) (in-process A/B)
std::atomic<bool> g_ab_legacy{getenv("HIVE_CPU_MATMUL_LEGACY") && atoi(getenv("HIVE_CPU_MATMUL_LEGACY")) > 0};
}  // namespace

void set_dispatch_legacy(bool on) { g_ab_legacy.store(on || (getenv("HIVE_CPU_MATMUL_LEGACY") && atoi(getenv("HIVE_CPU_MATMUL_LEGACY")) > 0)); }

void matmul(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy) {
  if (g_ab_legacy.load(std::memory_order_relaxed)) { ab_legacy::matmul(W, X, M, ldx, Y, ldy); return; }
  if (M <= 0 || W.r1 <= W.r0) return;
  std::vector<Job> jobs{Job{W.m, W.r0, W.r1, Y, ldy, X, M, ldx, 0}};
  run_jobs(jobs);
}

void matmul_multi(const MatJob* jobs_in, int n_jobs, const float* X, int M, int ldx) {
  if (g_ab_legacy.load(std::memory_order_relaxed)) { ab_legacy::matmul_multi(jobs_in, n_jobs, X, M, ldx); return; }
  std::vector<Job> jobs;
  jobs.reserve(n_jobs);
  for (int i = 0; i < n_jobs; ++i) {
    const MatJob& q = jobs_in[i];
    const int m = q.M > 0 ? q.M : M;
    if (m <= 0 || q.W.r1 <= q.W.r0) continue;
    jobs.push_back(Job{q.W.m, q.W.r0, q.W.r1, q.Y, q.ldy, q.X ? q.X : X, m, q.ldx > 0 ? q.ldx : ldx, 0});
  }
  if (!jobs.empty()) run_jobs(jobs);
}

}  // namespace hive::cpu
