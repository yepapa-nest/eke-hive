// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Long-context decode — CPU test (no GPU). Runs the **same kernel source** from decode_longctx.cuh under a SIMT emulation (block = 1024/128 OS threads ·
//   __syncthreads = std::barrier · warp ops = barrier exchange among the 32 threads of a warp · smem = static) and compares bit for bit against rule-based references.
//   ① HIVE_DECODE_TOPK2 topk2_kernel: reference = the specification of the original dec_topk_kernel (header of attn_decode_fused.h: top min(k, T) by (key descending, index ascending) →
//      drop -inf → index ascending → fill with -1 · cand/offset mapping), computed directly by sorting. Massive ties · -inf · NaN · chunk boundaries (CH±1) · per-row trows · k 1..2048.
//   ② HIVE_DECODE_IDXSCORE2 idx_scores2_kernel<CAND, KPT 1·2·4>: reference = the statements of the original idx_score32 ported to the host (same fmaf order, -ffp-contract=off).
//   ③ topk2_plan invariants (CH ≥ max(4096, 2k) · power of two · chunks cover T_in · union slots ≥ slots written).
//   Comparison against the original kernels on the GPU, and timing, is in engine/tests/test_decode_longctx.cu (run on a GPU machine).
// Build (-fno-strict-aliasing: the uint4↔bf16 reinterpretation in device code is tolerated by nvcc, but g++ -O2 drops it under aliasing rules — emulation-only flag):
//   g++ -std=c++20 -O2 -ffp-contract=off -fno-strict-aliasing -pthread -Itools/cpu_fake/include -Iengine/include -Iengine/src/kernels engine/tests/test_decode_longctx_cpu.cpp
// GPU model: a NaN that passes through a float register can become the canonical NaN (0x7FFFFFFF) on the GPU — nvcc 13.0 for sm_120a compiled the original okey(float)'s
//   `u | 0x80000000` into `FADD R, -|f|, -RZ`, turning the +NaN key 0xFFC00000 into 0x7FFFFFFF (4 FAILs measured on the GPU, confirmed in SASS). So this emulation's
//   __float_as_uint returns canonical NaN bits for NaN (disabled when HIVE_EMU_EXACT_FLOAT_BITS is defined) — a kernel that handles keys as float values fails here too.
//   The reference (ref_topk) does not use kernel functions; it sorts directly by the key **bytes** (independent of the kernel).
//   Negative control (does the unfixed kernel FAIL here?): build and run the same command with -DLCTX_CUH='"<path to the unfixed decode_longctx.cuh>"'.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <unistd.h>
#include <vector>

#include "hive/common.h"
#include "hive/kv_pack.h"

// ---- SIMT emulation ---------------------------------------------------------------------------------------------------------------
struct dim3 { unsigned x = 1, y = 1, z = 1; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint4 { unsigned x, y, z, w; };
struct float4 { float x, y, z, w; };
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return uint4{a, b, c, d}; }
#define __global__
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
namespace emu {
struct Warp { std::barrier<> bar{32}; uint32_t slot[32]; };
struct Block {
  explicit Block(int n) : bar(n), warps((n + 31) / 32) {}
  std::barrier<> bar;
  std::atomic<int> orv{0};
  std::vector<Warp> warps;
};
inline thread_local Block* blk = nullptr;
inline thread_local int tid = 0;
inline Warp& warp() { return blk->warps[tid >> 5]; }
inline uint32_t xch_read(uint32_t v, const std::function<uint32_t(const uint32_t*)>& f) {
  Warp& w = warp();
  w.slot[tid & 31] = v;
  w.bar.arrive_and_wait();
  const uint32_t r = f(w.slot);
  w.bar.arrive_and_wait();
  return r;
}
}  // namespace emu
inline thread_local dim3 threadIdx, blockIdx;
inline dim3 gridDim, blockDim;
inline void __syncthreads() { emu::blk->bar.arrive_and_wait(); }
inline int __syncthreads_or(int p) {
  if (p) emu::blk->orv.fetch_or(1);
  __syncthreads();
  const int r = emu::blk->orv.load();
  __syncthreads();
  if (emu::tid == 0) emu::blk->orv.store(0);
  __syncthreads();
  return r;
}
inline int __shfl_up_sync(unsigned, int v, int o) {
  const int lane = emu::tid & 31;
  return (int)emu::xch_read((uint32_t)v, [&](const uint32_t* s) { return lane >= o ? s[lane - o] : (uint32_t)v; });
}
inline int __shfl_sync(unsigned, int v, int src) { return (int)emu::xch_read((uint32_t)v, [&](const uint32_t* s) { return s[src & 31]; }); }
inline unsigned __ballot_sync(unsigned, int p) {
  return emu::xch_read(p ? 1u : 0u, [](const uint32_t* s) { uint32_t m = 0; for (int i = 0; i < 32; ++i) m |= (s[i] ? 1u : 0u) << i; return m; });
}
inline unsigned __match_any_sync(unsigned, unsigned v) {
  return emu::xch_read(v, [&](const uint32_t* s) { uint32_t m = 0; for (int i = 0; i < 32; ++i) m |= (s[i] == v ? 1u : 0u) << i; return m; });
}
inline int atomicAdd(int* p, int v) { return std::atomic_ref<int>(*p).fetch_add(v); }
inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
template <class T> inline T __ldcg(const T* p) { return *reinterpret_cast<const volatile T*>(p); }
inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline int __ffs(unsigned x) { return __builtin_ffs((int)x); }
// GPU model (see header): extracting the bits of a float value yields the canonical NaN for NaN — do not rely on sign or payload being preserved
inline uint32_t __float_as_uint(float f) {
  uint32_t u; memcpy(&u, &f, 4);
#ifndef HIVE_EMU_EXACT_FLOAT_BITS
  if (f != f) u = 0x7FFFFFFFu;
#endif
  return u;
}
inline float __uint_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
inline unsigned short __bfloat16_as_ushort(__nv_bfloat16 v) { return v.x; }
inline __nv_bfloat16 __ushort_as_bfloat16(unsigned short u) { return __nv_bfloat16{u}; }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
namespace hive::kvp {  // same formulas as the device versions in kv_pack.h (__CUDACC__ only)
inline float e2m1_val(uint32_t nib) {
  const uint32_t e = (nib >> 1) & 3u, m = nib & 1u;
  const uint32_t mag = e ? (((126u + e) << 23) | (m << 22)) : (m ? (126u << 23) : 0u);
  return __uint_as_float(mag | ((nib & 8u) << 28));
}
}  // namespace hive::kvp

#ifndef LCTX_CUH
#define LCTX_CUH "decode_longctx.cuh"
#endif
#include LCTX_CUH
#include "hive/decode_longctx.h"

namespace emu {
template <class F> void launch(dim3 grid, int nthreads, F body) {
  gridDim = grid; blockDim = dim3(nthreads);
  for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
      Block b(nthreads);
      std::vector<std::thread> th;
      th.reserve(nthreads);
      for (int t = 0; t < nthreads; ++t)
        th.emplace_back([&, t] {
          blk = &b; tid = t; threadIdx = dim3(t); blockIdx = dim3(bx, by);
          body();
        });
      for (auto& x : th) x.join();
    }
}
}  // namespace emu

using namespace hive;
using hive::k::DecRow;
namespace L = hive::k::lctx;

static int g_fail = 0, g_cases = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++g_fail; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// ---- ① top-k ---------------------------------------------------------------------------------------------------------------
// Reference = the **compiled** rule of the original dec_topk_kernel (confirmed from nvcc 13.0 sm_120a PTX/SASS, matches the detailed output of the GPU rerun):
//   K = bit sort key (histogram and tie-count loops — integer LOP3). S = the key of the two selection loops (nsel, emit) — for f32 `u | 0x80000000` is FADD -|f|, so
//   +NaN becomes 0x7FFFFFFF (canonical NaN bits; all other values equal K) · for bf16 S = K. thr and want come from K (= the keff-th element after sorting);
//   per thread range (C = ⌈T/1024⌉) r starts at the count of K == thr in the preceding ranges and increments for each S == thr; selected = S > thr || (S == thr && r < want), -inf excluded.
//   (The unfixed new version and the first reference followed a spec that ranks +NaN highest — against this reference that version FAILs on the 4 GPU-failing shapes: negative control.)
//   Computed directly from the key bytes (independent of the kernel functions and of the emulated __float_as_uint).
static uint32_t ref_okey(float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static uint32_t ref_okey(bf16 v) { const uint32_t u = v.x; return ((u & 0x8000u) ? ~u : (u | 0x8000u)) & 0xFFFFu; }
static uint32_t ref_skey(float f) { uint32_t u; memcpy(&u, &f, 4); return (u > 0x7F800000u && u < 0x80000000u) ? 0x7FFFFFFFu : ref_okey(f); }
static uint32_t ref_skey(bf16 v) { return ref_okey(v); }
static bool ref_neg_inf(float f) { uint32_t u; memcpy(&u, &f, 4); return u == 0xFF800000u; }
static bool ref_neg_inf(bf16 v) { return v.x == 0xFF80u; }
template <class KeyT>
std::vector<int32_t> ref_topk(const KeyT* kr, int T, int k, const int32_t* candrow, int bs, int offset, bool map_offset, int vis) {
  std::vector<int> ord(T);
  std::iota(ord.begin(), ord.end(), 0);
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return ref_okey(kr[a]) > ref_okey(kr[b]); });  // ties: smaller index first (stable)
  const int keff = std::min(k, T);
  std::vector<int> sel;
  if (keff > 0) {
    const uint32_t thr = ref_okey(kr[ord[keff - 1]]);
    int cgt = 0;
    for (int t = 0; t < T; ++t) cgt += ref_okey(kr[t]) > thr;
    const int want = keff - cgt;
    const int C = (T + 1023) / 1024;
    int eq_before = 0;  // count of K == thr in the preceding thread ranges (as block_excl_scan)
    for (int t0 = 0; t0 < T; t0 += C) {
      const int t1 = std::min(T, t0 + C);
      int r = eq_before;
      for (int t = t0; t < t1; ++t) {
        const uint32_t u = ref_skey(kr[t]);
        bool s = u > thr;
        if (u == thr) { s = r < want; ++r; }
        if (s && !ref_neg_inf(kr[t])) sel.push_back(t);
      }
      for (int t = t0; t < t1; ++t) eq_before += ref_okey(kr[t]) == thr;
    }
  }
  std::sort(sel.begin(), sel.end());
  std::vector<int32_t> out(k, -1);
  for (size_t j = 0; j < sel.size(); ++j) {
    int32_t p = sel[j];
    if (candrow && p >= 0) { const int32_t b = candrow[p / bs]; p = b >= 0 ? b * bs + p % bs : -1; }
    if (map_offset) p = (p >= 0 && p < vis) ? p + offset : -1;
    out[j] = p;
  }
  for (int j = (int)sel.size(); j < k; ++j) {
    int32_t p = -1;
    if (map_offset) p = -1;
    out[j] = p;
  }
  return out;
}
static float bf_of(float x) { return bf2f(f2bf(x)); }
template <class KeyT> KeyT mk(float x);
template <> float mk<float>(float x) { return x; }
template <> bf16 mk<bf16>(float x) { return f2bf(x); }

// kind: 0 = continuous random · 1 = massive ties (8 values) · 2 = narrow exponent (score-like) · 3 = half -inf + a few NaN · 4 = all equal · 5 = 98 % -inf + negative NaN (after -inf) · 6 = 3 % +NaN
template <class KeyT, int NBITS>
void topk_case(const char* name, int M, int T_in, bool use_trows, const std::vector<int>& trows, int k, int kind, bool with_cand, bool map_offset, uint32_t seed) {
  ++g_cases;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(0.f, 1.f);
  const int row_stride = T_in + 3;
  std::vector<KeyT> keys((size_t)M * row_stride);
  for (auto& v : keys) {
    float x;
    switch (kind) {
      case 0: x = U(rng) * 20.f - 5.f; break;
      case 1: x = (float)(int)(U(rng) * 8.f); break;
      case 2: x = 2.f + U(rng) * 1.5f; break;
      case 3: { const float r = U(rng); x = r < 0.5f ? -INFINITY : r < 0.51f ? NAN : (r < 0.52f ? -NAN : U(rng) * 4.f); break; }
      case 5: { const float r = U(rng); x = r < 0.98f ? -INFINITY : r < 0.99f ? -NAN : U(rng) * 4.f; break; }  // fewer than k valid per chunk — if the chunk stage dropped -inf the union would be empty
      case 6: { const float r = U(rng); x = r < 0.3f ? -INFINITY : r < 0.33f ? std::fabs(NAN) : U(rng) * 4.f; break; }  // +NaN only (highest key)
      default: x = 1.25f;
    }
    v = mk<KeyT>(x);
  }
  const int bs = 8, CB = (T_in + bs - 1) / bs + 2;
  std::vector<int32_t> cand((size_t)M * CB);
  for (auto& c : cand) c = U(rng) < 0.1f ? -1 : (int)(U(rng) * 1000);
  std::vector<DecRow> tab(M);
  for (int m = 0; m < M; ++m) {
    memset(&tab[m], 0, sizeof(DecRow));
    tab[m].trows = use_trows ? trows[m] : T_in;
    tab[m].visible = (int)(U(rng) * (T_in + 10));
  }
  const int offset = 133, out_stride = k + 5;
  std::vector<int32_t> out((size_t)M * out_stride, 777);
  const k::Topk2Plan p = k::topk2_plan(T_in, k, M, sizeof(KeyT));
  std::vector<uint8_t> scratch(p.bytes + 64);
  std::vector<int> cnt(M, 0);
  KeyT* ukeys = p.ncmax > 1 ? reinterpret_cast<KeyT*>(scratch.data()) : nullptr;
  int32_t* uidx = p.ncmax > 1 ? reinterpret_cast<int32_t*>(scratch.data() + p.idx_off) : nullptr;
  emu::launch(dim3(p.ncmax, M), L::TPK2, [&] {
    L::topk2_kernel<KeyT, NBITS>(keys.data(), T_in, use_trows, tab.data(), k, row_stride, out.data(), out_stride, with_cand ? cand.data() : nullptr, CB, bs,
                                 offset, map_offset, p.CH, ukeys, uidx, p.ustride, cnt.data());
  });
  int bad = 0;
  for (int m = 0; m < M; ++m) {
    const int T = use_trows ? std::min(trows[m], T_in) : T_in;
    const auto r = ref_topk<KeyT>(keys.data() + (size_t)m * row_stride, T, k, with_cand ? cand.data() + (size_t)m * CB : nullptr, bs, offset, map_offset, tab[m].visible);
    int first = -1;
    for (int j = 0; j < k; ++j) { const bool d = out[(size_t)m * out_stride + j] != r[j]; bad += d; if (d && first < 0) first = j; }
    if (first >= 0) {  // first mismatch position and value (for f32 including the key bits)
      const int32_t o = out[(size_t)m * out_stride + first], e = r[first];
      auto kb = [&](int32_t p) -> unsigned { if (with_cand || map_offset || p < 0 || p >= T) return 0; uint32_t u = 0; memcpy(&u, &keys[(size_t)m * row_stride + p], sizeof(KeyT)); return u; };
      printf("    first diff row %d pos %d: got %d (key 0x%x) want %d (key 0x%x)\n", m, first, o, kb(o), e, kb(e));
    }
    for (int j = k; j < out_stride; ++j) bad += out[(size_t)m * out_stride + j] != 777;  // nothing is written beyond the width
    EXPECT(cnt[m] == 0, "%s: counter not restored (row %d = %d)", name, m, cnt[m]);
  }
  EXPECT(bad == 0, "%s: %d mismatches (M %d T_in %d k %d kind %d CH %d nc %d)", name, bad, M, T_in, k, kind, p.CH, p.ncmax);
  printf("  topk %-28s M %d T_in %6d k %4d kind %d CH %5d nc %2d : %s\n", name, M, T_in, k, kind, p.CH, p.ncmax, bad ? "FAIL" : "ok");
}

// ---- ② scores -----------------------------------------------------------------------------------------------------------------
// the statements of the original idx_score32 (attn_decode_fused.cu) on the host — qT[d·32 + h] = bf2f(q[h·128 + d])
static float ref_score32(const float* qT, const float* wsm, const uint8_t* kr) {
  float acc[32];
  for (int h = 0; h < 32; ++h) acc[h] = 0.f;
  for (int qd = 0; qd < 4; ++qd) {
    const float sc = e8m0_to_f32(kr[qd]);
    uint32_t wv[4];
    memcpy(wv, kr + kvp::IDX_HDR + 16 * qd, 16);
    for (int w = 0; w < 4; ++w)
      for (int j = 0; j != 8; ++j) {
        const int d = qd * 32 + w * 8 + j;
        const float kv = kvp::e2m1_val((wv[w] >> (4 * j)) & 0xFu) * sc;
        for (int h = 0; h < 32; ++h) acc[h] = fmaf(qT[d * 32 + h], kv, acc[h]);
      }
  }
  float total = 0.f;
  for (int h = 0; h < 32; ++h) {
    float sb = bf_of(acc[h]);
    sb = fmaxf(sb, 0.f);
    sb = bf_of(sb * wsm[h]);
    total += sb;
  }
  return total;
}
template <bool CAND, int KPT>
void score_case(const char* name, int M, int ncols, int T, uint32_t seed) {
  ++g_cases;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<bf16> q((size_t)M * 32 * 128), w((size_t)M * 32);
  for (auto& v : q) v = f2bf(U(rng) * 3.f);
  for (auto& v : w) v = f2bf(U(rng) * 0.2f);
  std::vector<uint8_t> keys((size_t)std::max(T, 1) * kvp::IDX_ROW);
  for (int t = 0; t < T; ++t) {
    uint8_t* r = keys.data() + (size_t)t * kvp::IDX_ROW;
    for (int b = 0; b < 4; ++b) r[b] = (uint8_t)(122 + (rng() % 8));
    for (int b = 4; b < kvp::IDX_HDR; ++b) r[b] = 0;
    for (int b = kvp::IDX_HDR; b < kvp::IDX_ROW; ++b) r[b] = (uint8_t)rng();
  }
  const int bs = 8, CB = (ncols + bs - 1) / bs;
  std::vector<int32_t> cand((size_t)M * CB);
  for (auto& c : cand) c = (rng() % 10 == 0) ? -1 : (int)(rng() % ((T + bs - 1) / bs + 3));
  std::vector<DecRow> tab(M);
  for (int m = 0; m < M; ++m) {
    memset(&tab[m], 0, sizeof(DecRow));
    tab[m].kptr = keys.data();
    tab[m].trows = m == 0 ? T : T - (int)(rng() % 3);
    tab[m].visible = m == 0 ? T : (int)(rng() % (T + 1));
  }
  std::vector<bf16> score((size_t)M * ncols, f2bf(12345.f));
  const int per = L::IDX2_T * KPT;
  emu::launch(dim3((ncols + per - 1) / per, M), L::IDX2_T, [&] {
    L::idx_scores2_kernel<CAND, KPT>(q.data(), w.data(), tab.data(), ncols, CAND ? cand.data() : nullptr, CB, bs, score.data());
  });
  int bad = 0, finite = 0;
  for (int m = 0; m < M; ++m) {
    float qT[128 * 32], wsm[32];
    for (int i = 0; i < 32 * 128; ++i) { const int h = i / 128, d = i % 128; qT[d * 32 + h] = bf2f(q[(size_t)m * 4096 + i]); }
    for (int h = 0; h < 32; ++h) wsm[h] = bf2f(w[(size_t)m * 32 + h]);
    const int vis = std::min(tab[m].visible, tab[m].trows);
    for (int ci = 0; ci < ncols; ++ci) {
      int t = ci;
      if (CAND) { const int32_t b = cand[(size_t)m * CB + ci / bs]; t = b >= 0 ? b * bs + ci % bs : -1; }
      const bf16 r = (t >= 0 && t < vis) ? f2bf(ref_score32(qT, wsm, keys.data() + (size_t)t * kvp::IDX_ROW)) : f2bf(-INFINITY);

      bad += score[(size_t)m * ncols + ci].x != r.x;
      finite += std::isfinite(bf2f(r)) && bf2f(r) != 0.f;
    }
  }
  EXPECT(bad == 0, "%s: %d mismatches", name, bad);
  EXPECT(finite > 0, "%s: degenerate case (no finite nonzero score)", name);
  printf("  score %-27s M %d ncols %5d T %5d KPT %d : %s (%d finite)\n", name, M, ncols, T, KPT, bad ? "FAIL" : "ok", finite);
}

int main() {
  alarm(1800);  // an emulation deadlock (non-converged warp op, non-uniform return) must kill the test, not hang it
  // ③ plan invariants
  {
    int bad = 0;
    for (int k : {1, 7, 256, 512, 1000, 2048})
      for (int T : {0, 1, 511, 512, 4095, 4096, 4097, 8192, 20000, 40960, 65536, 81920, 102400, 131072, 1048576}) {
        const k::Topk2Plan p = k::topk2_plan(T, k, 8, 2);
        const bool pow2 = (p.CH & (p.CH - 1)) == 0;
        bad += !(p.CH >= 4096 && p.CH >= 2 * k && pow2 && (double)p.CH >= std::sqrt((double)T * k) - 1e-9 && (long long)p.ncmax * p.CH >= T);
        bad += p.ncmax > 1 ? !(p.ustride == p.ncmax * k && p.idx_off % 16 == 0 && p.bytes >= p.idx_off + (size_t)8 * p.ustride * 4) : !(p.bytes == 0 && T <= p.CH);
        // union slot count U of the longest row ≤ ustride
        if (p.ncmax > 1) bad += (p.ncmax - 1) * k + std::min(k, T - (p.ncmax - 1) * p.CH) > p.ustride;
      }
    ++g_cases;
    EXPECT(bad == 0, "plan invariants: %d violations", bad);
    const k::Topk2Plan a = k::topk2_plan(81920, 512, 1, 2), b = k::topk2_plan(131072, 512, 1, 2), c = k::topk2_plan(16384, 2048, 1, 4);
    printf("  plan: T 81920 k 512 → CH %d nc %d · T 131072 k 512 → CH %d nc %d · T 16384 k 2048 → CH %d nc %d : %s\n", a.CH, a.ncmax, b.CH, b.ncmax, c.CH,
           c.ncmax, bad ? "FAIL" : "ok");
  }
  // ① top-k
  topk_case<bf16, 16>("bf16 single chunk", 2, 3000, false, {}, 512, 0, false, true, 1);
  topk_case<bf16, 16>("bf16 T<k", 1, 300, false, {}, 512, 2, false, true, 2);
  topk_case<bf16, 16>("bf16 T=0 rows", 2, 5000, true, {0, 1}, 512, 0, false, true, 3);
  topk_case<bf16, 16>("bf16 2 chunks ties", 1, 4097, false, {}, 512, 1, false, true, 4);
  topk_case<bf16, 16>("bf16 score-like 5 chunks", 1, 20000, false, {}, 512, 2, false, true, 5);
  topk_case<bf16, 16>("bf16 -inf/NaN", 2, 13000, false, {}, 512, 3, false, true, 6);
  topk_case<bf16, 16>("bf16 all equal", 1, 12289, false, {}, 512, 4, false, true, 7);
  topk_case<bf16, 16>("bf16 trows per row", 3, 24576, true, {24576, 4096, 8193}, 512, 2, false, true, 8);
  topk_case<bf16, 16>("bf16 cand map", 2, 16384, false, {}, 512, 0, true, true, 9);
  topk_case<bf16, 16>("bf16 k small", 1, 9000, false, {}, 7, 1, false, false, 10);
  topk_case<bf16, 16>("bf16 k 1", 1, 9000, false, {}, 1, 4, false, false, 11);
  topk_case<bf16, 16>("bf16 mostly -inf, -NaN", 2, 20000, false, {}, 512, 5, false, true, 18);
  topk_case<float, 32>("f32 mostly -inf, -NaN", 1, 16384, false, {}, 2048, 5, false, false, 19);
  topk_case<float, 32>("f32 blocks k2048", 1, 16384, false, {}, 2048, 0, false, false, 12);
  topk_case<float, 32>("f32 blocks ties", 2, 12000, false, {}, 2048, 1, false, false, 13);
  topk_case<float, 32>("f32 -inf/NaN", 1, 10000, false, {}, 2048, 3, false, false, 14);
  topk_case<float, 32>("f32 kb = nblocks", 1, 512, false, {}, 512, 0, false, false, 15);
  // same shapes as the 4 GPU-measured FAILs (f32 · k 2048 · kind 3 ±NaN · 2 chunks) + two rows + +NaN only
  topk_case<float, 32>("f32 NaN 2 chunks (ws 5120)", 1, 5120, false, {}, 2048, 3, false, false, 31);
  topk_case<float, 32>("f32 NaN 2 chunks (ws 16384)", 2, 16384, false, {}, 2048, 3, false, false, 32);
  topk_case<float, 32>("f32 +NaN only", 1, 12800, false, {}, 2048, 6, false, false, 33);
  topk_case<bf16, 16>("bf16 +NaN only", 1, 20000, false, {}, 512, 6, false, true, 34);
  topk_case<bf16, 16>("bf16 boundary CH-1", 1, 8191, false, {}, 1024, 2, false, true, 16);
  topk_case<bf16, 16>("bf16 boundary CH+1", 1, 8193, false, {}, 1024, 2, false, true, 17);
  // ② scores
  score_case<false, 1>("full KPT1", 2, 700, 650, 21);
  score_case<false, 2>("full KPT2 (tail blocks)", 2, 1100, 400, 22);
  score_case<false, 4>("full KPT4", 1, 1500, 1500, 23);
  score_case<false, 2>("vis 1 (only t 0)", 1, 300, 1, 26);
  score_case<true, 2>("cand KPT2", 2, 640, 900, 24);
  score_case<true, 4>("cand KPT4", 1, 520, 700, 25);
  printf("%d cases — %s\n", g_cases, g_fail ? "SOME FAIL" : "ALL PASS");
  return g_fail ? 1 : 0;
}
