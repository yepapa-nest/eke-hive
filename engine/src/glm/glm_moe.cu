// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/glm/glm_moe.h"

#include <cuda_fp8.h>

#include <algorithm>

#include "hive/common.h"

namespace hive::glm {

namespace {
__device__ __constant__ float kE2M1g[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
__device__ __forceinline__ float e4m3g(uint8_t v) { __nv_fp8_e4m3 t; t.__x = v; return float(t); }

// Dot of one NVFP4 row (K columns) with an activation in shared memory (bf16), warp-cooperative; returns the warp sum in all lanes.
__device__ __forceinline__ float row_dot(const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ srow, int K, const __nv_bfloat16* xs) {
  const uint4* wr = reinterpret_cast<const uint4*>(wrow);
  const int lane = threadIdx.x & 31;
  float acc = 0.f;
  for (int c = lane; c < K / 32; c += 32) {
    const uint4 q = __ldg(wr + c);
    const float s0 = e4m3g(srow[2 * c]), s1 = e4m3g(srow[2 * c + 1]);
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    const __nv_bfloat16* xr = xs + c * 32;
    float lo = 0.f, hi = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (words[j] >> (8 * b)) & 0xFF;
        const int k = j * 8 + b * 2;
        const float v = kE2M1g[byte & 15] * __bfloat162float(xr[k]) + kE2M1g[byte >> 4] * __bfloat162float(xr[k + 1]);
        if (k < 16) lo += v; else hi += v;
      }
    acc = fmaf(lo, s0, fmaf(hi, s1, acc));
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
  return acc;
}

constexpr int kWarps = 8;
// gate/up: block (pair p, tile of I rows); each warp computes gate and up for rows of the tile → y[p, i] = bf16(silu·up)
__global__ void __launch_bounds__(kWarps * 32) gu_kernel(ExpertRecLayout L, const MoePair* __restrict__ pairs, const __nv_bfloat16* __restrict__ x,
                                                         int H, int I, float limit, __nv_bfloat16* __restrict__ y) {
  extern __shared__ __nv_bfloat16 xs[];
  const MoePair pr = pairs[blockIdx.y];
  for (int i = threadIdx.x; i < H; i += blockDim.x) xs[i] = x[(size_t)pr.row * H + i];
  __syncthreads();
  const float g1 = *reinterpret_cast<const float*>(pr.rec + L.g), g3 = *reinterpret_cast<const float*>(pr.rec + L.g + 4);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rows_per_block = (I + gridDim.x - 1) / gridDim.x;
  const int r0 = blockIdx.x * rows_per_block, r1 = min(I, r0 + rows_per_block);
  for (int i = r0 + warp; i < r1; i += kWarps) {
    float g = row_dot(pr.rec + L.w1 + (size_t)i * (H / 2), pr.rec + L.s1 + (size_t)i * (H / 16), H, xs) * g1;
    float u = row_dot(pr.rec + L.w3 + (size_t)i * (H / 2), pr.rec + L.s3 + (size_t)i * (H / 16), H, xs) * g3;
    if (lane == 0) {
      g = __bfloat162float(__float2bfloat16(g)); u = __bfloat162float(__float2bfloat16(u));
      if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
      y[(size_t)blockIdx.y * I + i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
    }
  }
}
// down: block (pair p, tile of H rows) → part[p, n] = bf16(w2 · y[p]) · w   (fp32 store)
__global__ void __launch_bounds__(kWarps * 32) down_kernel(ExpertRecLayout L, const MoePair* __restrict__ pairs, const __nv_bfloat16* __restrict__ y,
                                                           int H, int I, float* __restrict__ part) {
  extern __shared__ __nv_bfloat16 ys[];
  const MoePair pr = pairs[blockIdx.y];
  for (int i = threadIdx.x; i < I; i += blockDim.x) ys[i] = y[(size_t)blockIdx.y * I + i];
  __syncthreads();
  const float g2 = *reinterpret_cast<const float*>(pr.rec + L.g + 8);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rows_per_block = (H + gridDim.x - 1) / gridDim.x;
  const int r0 = blockIdx.x * rows_per_block, r1 = min(H, r0 + rows_per_block);
  for (int n = r0 + warp; n < r1; n += kWarps) {
    const float v = row_dot(pr.rec + L.w2 + (size_t)n * (I / 2), pr.rec + L.s2 + (size_t)n * (I / 16), I, ys) * g2;
    if (lane == 0) part[(size_t)blockIdx.y * H + n] = __bfloat162float(__float2bfloat16(v)) * pr.w;
  }
}
// out[row] += Σ parts of that row in pair order (pairs are grouped by the caller in ascending row order)
__global__ void reduce_kernel(const MoePair* __restrict__ pairs, int n_pairs, const float* __restrict__ part, int H, float* __restrict__ out) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= H) return;
  for (int p = 0; p < n_pairs; ++p) out[(size_t)pairs[p].row * H + n] += part[(size_t)p * H + n];
}

__global__ void dequant_rec_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, float g, int N, int K, __nv_bfloat16* __restrict__ out) {
  const size_t pairs = (size_t)N * K / 2;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += (size_t)gridDim.x * blockDim.x) {
    const size_t n = i / (K / 2), kp = i % (K / 2);
    const float sc = e4m3g(s[n * (K / 16) + (kp * 2) / 16]) * g;
    const uint8_t b = w[i];
    __nv_bfloat162 v; v.x = __float2bfloat16(kE2M1g[b & 15] * sc); v.y = __float2bfloat16(kE2M1g[b >> 4] * sc);
    reinterpret_cast<__nv_bfloat162*>(out)[i] = v;
  }
}
__global__ void swiglu_rows_kernel(const __nv_bfloat16* __restrict__ gu, int R, int I, float limit, __nv_bfloat16* __restrict__ y) {
  const size_t total = (size_t)R * I;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
    const size_t r = idx / I, i = idx % I;
    float g = __bfloat162float(gu[r * 2 * I + i]), u = __bfloat162float(gu[r * 2 * I + I + i]);
    if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
    y[idx] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
  }
}
__global__ void scatter_add_kernel(const __nv_bfloat16* __restrict__ e, const int32_t* __restrict__ rows, const float* __restrict__ w, int R, int H, float* __restrict__ out) {
  const size_t total = (size_t)R * H;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
    const size_t r = idx / H, n = idx % H;
    out[(size_t)rows[r] * H + n] += __bfloat162float(e[idx]) * w[r];
  }
}
__global__ void gather_rows_kernel(const __nv_bfloat16* __restrict__ x, const int32_t* __restrict__ rows, int R, int H, __nv_bfloat16* __restrict__ out) {
  const size_t total = (size_t)R * H;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x)
    out[idx] = x[(size_t)rows[idx / H] * H + idx % H];
}
__global__ void add_f32_kernel(float* __restrict__ a, const float* __restrict__ b, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a[i] += b[i];
}

// ==== v1 decode kernels ====================================================================================================================
// Design (measured on the RTX PRO 6000: the first CUDA-core version reached the streaming-read ceiling only with its arithmetic
//   removed — decode + fp32 FMAs did not overlap with the loads — so the dot products run on the tensor cores):
//   (1) both kernels first group the pairs by record pointer into chunks of ≤ kChunk pairs in shared memory (order of first appearance,
//       members in pair order, larger groups split in order), so a record shared by several rows is streamed once for all of them; no
//       extra launch, no host round trip.
//   (2) gu2_kernel (persistent grid, equal rounds per block): block item = (chunk, 8 rows of I) as one m16n8k16 tile — mma rows 0-7 = gate
//       rows, 8-15 = up rows, mma columns = the chunk's activation rows. K is split over the 8 warps; every load instruction reads 64
//       contiguous bytes per row. e2m1 is decoded exactly (bit trick → e4m3 → f16) and multiplied by its e4m3 scale in f16 (exact), the
//       activations are converted bf16 → f16, fp32 accumulation; split-K partials are summed in warp order in smem, then bf16 → clamp →
//       SwiGLU → bf16 into y[pair, i].
//   (3) down2_kernel (programmatic dependent launch: it groups and prefetches its first weights while gu2 drains, then waits for y):
//       block = 16 output columns n for ALL pairs, warp item = one chunk over the full K; bf16(w2·y·g2)·w_p goes to a smem table
//       [pair][n]; after a barrier the block adds the table into out in pair order — per element the same fp32 add sequence as the old
//       reduce_kernel → deterministic (no atomics, fixed reduction orders everywhere), no fp32 partial buffer in global memory.
//   An activation outside the f16 range (|x| ≥ 65520) or a non-finite input makes the item's accumulator non-finite; that item is then
//   recomputed exactly on the CUDA cores (row_dot_cc), so the f16 operand path never changes a finite result into inf/NaN.
constexpr int kChunk = 8, kV1Warps = 8, kV1Threads = kV1Warps * 32, kTN = 16, kMaxPairs = 512, kInBlockGroupMax = 64;
struct Chunk { const uint8_t* rec; int cnt; uint16_t p[kChunk]; int pad; };
static_assert(sizeof(Chunk) == 32, "Chunk layout");
// shared memory: head = Chunk[n] + prow[n] (kept), scratch = recs[n] + info[n] + base[n] + scan tmp (dead after build_chunks)
__host__ __device__ constexpr size_t chunk_head(int n) { return ((size_t)n * (sizeof(Chunk) + 4) + 15) / 16 * 16; }
__host__ __device__ constexpr size_t chunk_scratch(int n) { return (size_t)n * (8 + 4 + 4) + (kV1Threads + 1) * 4; }
__host__ __device__ constexpr size_t chunk_smem(int n) { return chunk_head(n) + chunk_scratch(n); }

// Group info of pair p: leader (first pair with the same record), rank among that record's pairs, group size — packed lead | rank << 10 |
//   size << 20 (n ≤ 512).
__device__ __forceinline__ int pair_info(const uint8_t* const* recs, int n, int p) {
  const uint8_t* r = recs[p];
  int l = p, rk = 0, sz = 0;
  for (int q = 0; q < n; ++q)
    if (recs[q] == r) { if (q < p) { ++rk; l = min(l, q); } ++sz; }
  return l | rk << 10 | sz << 20;
}
// n > kInBlockGroupMax: the O(n²) part runs once here (one warp per pair, ballots) instead of in every block of both kernels (measured at
//   n = 512: +43 µs in gu when every block did it).
__global__ void pair_info_kernel(const MoePair* __restrict__ pairs, int n, int* __restrict__ info) {
  const int lane = threadIdx.x & 31, p = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (p >= n) return;
  const uint8_t* r = pairs[p].rec;
  int l = -1, rk = 0, sz = 0;
  for (int q0 = 0; q0 < n; q0 += 32) {
    const int q = q0 + lane;
    const unsigned m = __ballot_sync(0xffffffffu, q < n && pairs[q].rec == r);
    if (l < 0 && m) l = q0 + __ffs(m) - 1;
    sz += __popc(m);
    rk += __popc(m & (q0 + 32 <= p ? 0xffffffffu : q0 >= p ? 0u : (1u << (p - q0)) - 1u));
  }
  if (lane == 0) info[p] = l | rk << 10 | sz << 20;
}

// Fills ch[0..nch) and prow[p] = pairs[p].row; returns nch (same value in all threads). Must be called by the whole block.
//   ginfo: per-pair info from pair_info_kernel, or nullptr (computed here, n ≤ kInBlockGroupMax).
__device__ int build_chunks(const MoePair* __restrict__ pairs, int n, const int* __restrict__ ginfo, uint8_t* sm, Chunk*& ch, int*& prow) {
  ch = reinterpret_cast<Chunk*>(sm);
  prow = reinterpret_cast<int*>(ch + n);
  const uint8_t** recs = reinterpret_cast<const uint8_t**>(sm + chunk_head(n));
  int* info = reinterpret_cast<int*>(recs + n);
  int* base = info + n;
  int* tmp = base + n;  // [kV1Threads + 1]
  for (int p = threadIdx.x; p < n; p += blockDim.x) { recs[p] = pairs[p].rec; prow[p] = pairs[p].row; }
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    const int inf = ginfo ? ginfo[p] : pair_info(recs, n, p);
    info[p] = inf;
    base[p] = ((inf & 1023) == p) ? ((inf >> 20) + kChunk - 1) / kChunk : 0;  // chunks of the group, counted at its leader
  }
  if (threadIdx.x == 0) tmp[kV1Threads] = 0;  // carry
  __syncthreads();
  for (int s0 = 0; s0 < n; s0 += kV1Threads) {  // inclusive scan of base (Hillis-Steele per 256-wide segment + carry)
    const int p = s0 + threadIdx.x;
    int v = p < n ? base[p] : 0;
    for (int o = 1; o < min(kV1Threads, n - s0); o <<= 1) {
      tmp[threadIdx.x] = v;
      __syncthreads();
      if ((int)threadIdx.x >= o) v += tmp[threadIdx.x - o];
      __syncthreads();
    }
    const int carry = tmp[kV1Threads];
    if (p < n) base[p] = v + carry;
    __syncthreads();
    if (threadIdx.x == min(kV1Threads, n - s0) - 1) tmp[kV1Threads] = carry + v;
    __syncthreads();
  }
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    const int inf = info[p], l = inf & 1023, rk = (inf >> 10) & 1023, sz = inf >> 20;
    const int c = base[l] - (sz + kChunk - 1) / kChunk + rk / kChunk, slot = rk % kChunk;  // exclusive base of the group + sub-chunk
    ch[c].p[slot] = (uint16_t)p;
    if (slot == 0) { ch[c].rec = recs[p]; ch[c].cnt = min(kChunk, sz - (rk / kChunk) * kChunk); }
  }
  const int nch = tmp[kV1Threads];
  __syncthreads();
  return nch;
}

// e2m1 → e4m3 bit trick: e4m3 bits (n & 8) << 4 | (n & 7) << 2 represent exactly e2m1(n) · 2^-6 (also for the e2m1 subnormal 0.5, which
//   lands on the e4m3 subnormal 2^-7); cvt.rn.f16x2.e4m3x2 (sm_89+) is exact. (cvt.*.e2m1x2 needs the arch-specific sm_120a PTX target,
//   which this build does not emit — measured: ptxas "not supported on .target sm_120".)
// The f16 product with the scale pre-multiplied by 2^6 is exact: e2m1·e4m3 has ≤ 6 significant bits and lies in [2^-10, 2688].
// {s·64, s·64} f16x2 pairs for the two 16-column halves of a 32-column block from its two e4m3 scale bytes
__device__ __forceinline__ void scales64(uint16_t s2, uint32_t& s_lo, uint32_t& s_hi) {
  uint32_t sc; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sc) : "h"(s2));
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sc) : "r"(sc), "r"(0x54005400u));  // × 64 (exact)
#ifdef HIVE_GLM_MOE_NEGCTRL_SCALE
  s_lo = __byte_perm(sc, 0, 0x3232); s_hi = __byte_perm(sc, 0, 0x1010);  // negative control: swapped scale pair
#else
  s_lo = __byte_perm(sc, 0, 0x1010); s_hi = __byte_perm(sc, 0, 0x3232);
#endif
}
// One nibble word (8 columns) → f16x2 e2m1·scale (sc from scales64): h0 = (c0, c2), h1 = (c4, c6), h2 = (c1, c3), h3 = (c5, c7)
__device__ __forceinline__ void decode8h(uint32_t w, uint32_t sc, uint32_t (&h)[4]) {
#ifdef HIVE_GLM_MOE_NEGCTRL
  w = ((w >> 4) & 0x0F0F0F0Fu) | ((w & 0x0F0F0F0Fu) << 4);  // negative control: swapped nibble order inside each byte
#endif
  const uint32_t ev = ((w << 4) & 0x80808080u) | ((w << 2) & 0x1C1C1C1Cu);  // even columns (low nibble of each byte) → e4m3·2^6
  const uint32_t od = (w & 0x80808080u) | ((w >> 2) & 0x1C1C1C1Cu);         // odd columns
  asm("{\n.reg .b16 a, b, c, d;\nmov.b32 {a, b}, %4;\nmov.b32 {c, d}, %5;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n"
      "cvt.rn.f16x2.e4m3x2 %2, c;\ncvt.rn.f16x2.e4m3x2 %3, d;\n}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(ev), "r"(od));
#pragma unroll
  for (int k = 0; k < 4; ++k) asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(h[k]) : "r"(sc));
}
// Same, widened to fp32 in column order o[0..7] (CUDA-core fallback)
__device__ __forceinline__ void decode8(uint32_t w, uint32_t sc, float (&o)[8]) {
  uint32_t h[4];
  decode8h(w, sc, h);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float lo, hi; asm("{.reg .f16 l, u;\nmov.b32 {l, u}, %2;\ncvt.f32.f16 %0, l;\ncvt.f32.f16 %1, u;}" : "=f"(lo), "=f"(hi) : "r"(h[k]));
    const int c = (k & 1) * 4 + (k >> 1);
    o[c] = lo; o[c + 2] = hi;
  }
}
__device__ __forceinline__ float dot8(const uint4 xv, const float (&w)[8], float acc) {
  const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    acc = fmaf(__uint_as_float(xs[b] << 16), w[2 * b], acc);
    acc = fmaf(__uint_as_float(xs[b] & 0xFFFF0000u), w[2 * b + 1], acc);
  }
  return acc;
}
__device__ __forceinline__ uint4 ld_stream(const void* p) {  // weights: read once, do not allocate in L1 (keeps x / y there)
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// B operands from 8 bf16 activations (one uint4) in the same column order as decode8h: b0 = (x0, x2), b1 = (x4, x6), b2 = (x1, x3), b3 = (x5, x7).
//   bf16 → f16 is exact for 2^-14 ≤ |x| < 65520; below, the f16 subnormal rounding error is ≤ 2^-25 absolute; above, the value becomes inf and
//   the accumulator non-finite → the caller recomputes that item exactly on the CUDA cores.
__device__ __forceinline__ void x8_to_h(const uint4 xv, uint32_t (&b)[4]) {
  const uint32_t u[4] = {xv.x, xv.y, xv.z, xv.w};
  float f[8];
#pragma unroll
  for (int k = 0; k < 4; ++k) { f[2 * k] = __uint_as_float(u[k] << 16); f[2 * k + 1] = __uint_as_float(u[k] & 0xFFFF0000u); }
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(b[0]) : "f"(f[0]), "f"(f[2]));
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(b[1]) : "f"(f[4]), "f"(f[6]));
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(b[2]) : "f"(f[1]), "f"(f[3]));
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(b[3]) : "f"(f[5]), "f"(f[7]));
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 16 mma steps of one 32-column word group: rows A (mma rows g) and B (mma rows g+8) of 32 columns each, activation 32 bf16 (4 × uint4).
__device__ __forceinline__ void mma32(float (&c)[4], const uint4 qa, uint16_t sa, const uint4 qb, uint16_t sb, const uint4 (&xv)[4]) {
  uint32_t al, ah, bl, bh;
  scales64(sa, al, ah);
  scales64(sb, bl, bh);
  const uint32_t wa[4] = {qa.x, qa.y, qa.z, qa.w}, wb[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    uint32_t ha[4], hb[4], xb[4];
    decode8h(wa[t], t < 2 ? al : ah, ha);
    decode8h(wb[t], t < 2 ? bl : bh, hb);
    x8_to_h(xv[t], xb);
    mma16816(c, ha[0], hb[0], ha[1], hb[1], xb[0], xb[1]);
    mma16816(c, ha[2], hb[2], ha[3], hb[3], xb[2], xb[3]);
  }
}
// CUDA-core exact row dot (fallback): K columns, lane-strided 32-column blocks, fp32 accumulation; result in all lanes.
__device__ float row_dot_cc(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, int K, const __nv_bfloat16* __restrict__ a) {
  const int lane = threadIdx.x & 31;
  float acc = 0.f;
  for (int cb = lane; cb < K / 32; cb += 32) {
    const uint4 q = __ldg(reinterpret_cast<const uint4*>(w) + cb);
    uint32_t sl, sh;
    scales64(__ldg(reinterpret_cast<const uint16_t*>(s) + cb), sl, sh);
    const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      float o[8];
      decode8(ws[t], t < 2 ? sl : sh, o);
      acc = dot8(__ldg(reinterpret_cast<const uint4*>(a + cb * 32 + 8 * t)), o, acc);
    }
  }
  return warp_sum(acc);
}
__device__ __forceinline__ float swiglu1(float g, float u, float limit) {
  g = __bfloat162float(__float2bfloat16(g)); u = __bfloat162float(__float2bfloat16(u));
  if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
  return g / (1.f + __expf(-g)) * u;
}

// gate/up (tensor cores): block item = (chunk c, 8 rows i0..i0+7 of I). mma rows 0-7 = gate rows, 8-15 = up rows; mma columns = the chunk's
//   activation rows (≤ 8); K = H split over the 8 warps (H/8 columns each), every lane of quad t owning the 16-byte words 4j+t of its warp
//   segment (each load instruction reads 64 contiguous bytes per row). Split-K partials are reduced in smem in warp order (deterministic).
template <int KB>
__global__ void __launch_bounds__(kV1Threads, 2) gu2_kernel(ExpertRecLayout L, const MoePair* __restrict__ pairs, int n,
                                                            const int* __restrict__ ginfo, const __nv_bfloat16* __restrict__ x, int H, int I, float limit,
                                                            __nv_bfloat16* __restrict__ y) {
  asm volatile("griddepcontrol.launch_dependents;");
  extern __shared__ __align__(16) uint8_t sm[];
  __shared__ float red[kV1Warps][32][4];
  Chunk* ch; int* prow;
  const int nch = build_chunks(pairs, n, ginfo, sm, ch, prow);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tiles = I / 8, items = nch * tiles;
  const int seg = warp * (H / 8);  // first column of this warp's K segment
  // weights of item `it` → registers; called for the next item right after each word group is consumed (prefetch, no extra registers)
  const uint8_t *wg, *wu, *sg, *su;
  auto addr = [&](int it) {
    const int c = it / tiles, i0 = (it - c * tiles) * 8;
    const uint8_t* rec = ch[c].rec;
    wg = rec + L.w1 + (size_t)(i0 + g) * (H / 2) + seg / 2;
    wu = rec + L.w3 + (size_t)(i0 + g) * (H / 2) + seg / 2;
    sg = rec + L.s1 + (size_t)(i0 + g) * (H / 16) + seg / 16;
    su = rec + L.s3 + (size_t)(i0 + g) * (H / 16) + seg / 16;
  };
  uint4 qg[KB], qu[KB];
  uint16_t s2g[KB], s2u[KB];
  auto load = [&](int j) {
    const int wi = 4 * j + t;  // 16-byte word index inside the warp segment
    qg[j] = ld_stream(wg + wi * 16); qu[j] = ld_stream(wu + wi * 16);
    s2g[j] = __ldg(reinterpret_cast<const uint16_t*>(sg) + wi); s2u[j] = __ldg(reinterpret_cast<const uint16_t*>(su) + wi);
  };
  if ((int)blockIdx.x < items) {
    addr(blockIdx.x);
#pragma unroll
    for (int j = 0; j < KB; ++j) load(j);
  }
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int c = it / tiles, i0 = (it - c * tiles) * 8;
    const Chunk& C = ch[c];
    const int cnt = C.cnt;
    const uint8_t* rec = C.rec;
    const bool more = it + (int)gridDim.x < items;
    if (more) addr(it + gridDim.x);
    const bool act = g < cnt;
    const __nv_bfloat16* xr = x + (size_t)(act ? prow[C.p[g]] : 0) * H + seg;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < KB; ++j) {
      uint4 xv[4];
      const uint4* xp = reinterpret_cast<const uint4*>(xr + 32 * (4 * j + t));
#pragma unroll
      for (int k = 0; k < 4; ++k) xv[k] = act ? __ldg(xp + k) : make_uint4(0, 0, 0, 0);
      mma32(acc, qg[j], s2g[j], qu[j], s2u[j], xv);
      if (more) load(j);
    }
    // acc: [0], [1] = gate row i0+g · act 2t, 2t+1; [2], [3] = up row i0+g · act 2t, 2t+1
#pragma unroll
    for (int k = 0; k < 4; ++k) red[warp][lane][k] = acc[k];
    __syncthreads();
    float yv = 0.f;
    bool bad = false;
    const int fr = threadIdx.x >> 3, fa = threadIdx.x & 7;  // finalize: thread (row fr < 8, act fa)
    if (threadIdx.x < 64 && fa < cnt) {
      const int ln = fr * 4 + (fa >> 1), k = fa & 1;
      float gs = 0.f, us = 0.f;
#pragma unroll
      for (int w = 0; w < kV1Warps; ++w) { gs += red[w][ln][k]; us += red[w][ln][2 + k]; }
      gs *= __ldg(reinterpret_cast<const float*>(rec + L.g));
      us *= __ldg(reinterpret_cast<const float*>(rec + L.g + 4));
      bad = !isfinite(gs) || !isfinite(us);
      yv = swiglu1(gs, us, limit);
    }
#ifdef HIVE_GLM_MOE_NO_FALLBACK
    bad = false;  // control build: shows that the out-of-f16-range test cases need the fallback
#endif
    if (__syncthreads_or(bad)) {  // rare: an activation outside the f16 range (or a non-finite input) → exact CUDA-core recompute
      const int row = i0 + warp;
      for (int a = 0; a < cnt; ++a) {
        const __nv_bfloat16* xa = x + (size_t)prow[C.p[a]] * H;
        const float gs = row_dot_cc(rec + L.w1 + (size_t)row * (H / 2), rec + L.s1 + (size_t)row * (H / 16), H, xa) *
                         __ldg(reinterpret_cast<const float*>(rec + L.g));
        const float us = row_dot_cc(rec + L.w3 + (size_t)row * (H / 2), rec + L.s3 + (size_t)row * (H / 16), H, xa) *
                         __ldg(reinterpret_cast<const float*>(rec + L.g + 4));
        if (lane == 0) y[(size_t)C.p[a] * I + row] = __float2bfloat16(swiglu1(gs, us, limit));
      }
    } else if (threadIdx.x < 64 && fa < cnt) {
      y[(size_t)C.p[fa] * I + i0 + fr] = __float2bfloat16(yv);
    }
  }
}

// down + reduce (tensor cores): block ↔ 16 output columns [n0, n0+16) for all pairs; warp item = one chunk over the full K = I
//   (mma rows = the 16 columns n, mma columns = the chunk's y rows); per lane the 16-byte words 4j+t of each row, 4 words per row in flight.
//   bf16(w2·y·g2)·w_p goes to a smem table [pair][16]; after a barrier the block adds it into out in pair order.
template <int KB>
__global__ void __launch_bounds__(kV1Threads, 2) down2_kernel(ExpertRecLayout L, const MoePair* __restrict__ pairs, int n,
                                                              const int* __restrict__ ginfo, const __nv_bfloat16* __restrict__ y, int H, int I, float* __restrict__ out) {
  static_assert(kTN == 16, "down2 tile = one mma M tile");
  extern __shared__ __align__(16) uint8_t sm[];
  Chunk* ch; int* prow;
  const int nch = build_chunks(pairs, n, ginfo, sm, ch, prow);
  __nv_bfloat16* part = reinterpret_cast<__nv_bfloat16*>(sm + chunk_head(n));  // [n][16] bf16(w2·y·g2), over the dead grouping scratch
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * kTN;
  constexpr int WPR = KB * 1024 / 32 / 4;  // 16-byte words per row per lane: I/32 words per row over the 4 lanes of a quad
  constexpr int BATCH = 4;
  static_assert(WPR % BATCH == 0, "down2 batch");
  bool waited = false;
  for (int c = warp; c < nch; c += kV1Warps) {
    const Chunk& C = ch[c];
    const int cnt = C.cnt;
    const uint8_t* rec = C.rec;
    const uint8_t* wa = rec + L.w2 + (size_t)(n0 + g) * (I / 2);
    const uint8_t* wb = rec + L.w2 + (size_t)(n0 + g + 8) * (I / 2);
    const uint16_t* sa = reinterpret_cast<const uint16_t*>(rec + L.s2 + (size_t)(n0 + g) * (I / 16));
    const uint16_t* sb = reinterpret_cast<const uint16_t*>(rec + L.s2 + (size_t)(n0 + g + 8) * (I / 16));
    uint4 qa[BATCH], qb[BATCH];
    uint16_t s2a[BATCH], s2b[BATCH];
    auto load = [&](int j0) {
#pragma unroll
      for (int j = 0; j < BATCH; ++j) {
        const int wi = 4 * (j0 + j) + t;
        qa[j] = ld_stream(wa + wi * 16); qb[j] = ld_stream(wb + wi * 16);
        s2a[j] = __ldg(sa + wi); s2b[j] = __ldg(sb + wi);
      }
    };
    load(0);
    if (!waited) { asm volatile("griddepcontrol.wait;" ::: "memory"); waited = true; }  // y is complete after this (weights prefetched before)
    const bool act = g < cnt;
    const __nv_bfloat16* yr = y + (size_t)(act ? C.p[g] : 0) * I;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j0 = 0; j0 < WPR; j0 += BATCH) {
      uint4 ca[BATCH], cb[BATCH];
      uint16_t csa[BATCH], csb[BATCH];
#pragma unroll
      for (int j = 0; j < BATCH; ++j) { ca[j] = qa[j]; cb[j] = qb[j]; csa[j] = s2a[j]; csb[j] = s2b[j]; }
      if (j0 + BATCH < WPR) load(j0 + BATCH);
#pragma unroll
      for (int j = 0; j < BATCH; ++j) {
        uint4 yv[4];
        const uint4* yp = reinterpret_cast<const uint4*>(yr + 32 * (4 * (j0 + j) + t));
#pragma unroll
        for (int k = 0; k < 4; ++k) yv[k] = act ? __ldg(yp + k) : make_uint4(0, 0, 0, 0);
        mma32(acc, ca[j], csa[j], cb[j], csb[j], yv);
      }
    }
    // acc: [0], [1] = column n0+g · act 2t, 2t+1; [2], [3] = column n0+g+8 · act 2t, 2t+1
    const float g2 = __ldg(reinterpret_cast<const float*>(rec + L.g + 8));
    bool bad = false;
#pragma unroll
    for (int k = 0; k < 4; ++k) bad |= (2 * t + (k & 1) < cnt) && !isfinite(acc[k]);
#ifdef HIVE_GLM_MOE_NO_FALLBACK
    bad = false;
#endif
    if (__any_sync(0xffffffffu, bad)) {  // rare: y outside the f16 range → exact CUDA-core recompute of this item
      for (int rr = 0; rr < kTN; ++rr)
        for (int a = 0; a < cnt; ++a) {
          const float v = row_dot_cc(rec + L.w2 + (size_t)(n0 + rr) * (I / 2), rec + L.s2 + (size_t)(n0 + rr) * (I / 16), I,
                                     y + (size_t)C.p[a] * I) * g2;
          if (lane == 0) part[(size_t)C.p[a] * kTN + rr] = __float2bfloat16(v);
        }
    } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int a = 2 * t + (k & 1);
        if (a < cnt) part[(size_t)C.p[a] * kTN + g + (k >> 1) * 8] = __float2bfloat16(acc[k] * g2);
      }
    }
  }
  if (!waited) asm volatile("griddepcontrol.wait;" ::: "memory");
  __syncthreads();
  // out[row, n] += bf16(…) · w_p in pair order; rows are split over thread groups (row % G): each element sees the old reduce_kernel's
  //   sequence of fp32 adds (same values: bf16 → fp32 exact, × w_p).
  constexpr int G = kV1Threads / kTN;
  const int nl = threadIdx.x % kTN, cls = threadIdx.x / kTN;
  for (int p = 0; p < n; ++p) {
    const int row = prow[p];
    if (row % G == cls) out[(size_t)row * H + n0 + nl] += __bfloat162float(part[(size_t)p * kTN + nl]) * __ldg(&pairs[p].w);
  }
}
}  // namespace

void gather_rows(const __nv_bfloat16* x, const int32_t* rows, int R, int H, __nv_bfloat16* out, cudaStream_t st) {
  if (R <= 0) return;
  gather_rows_kernel<<<std::min<size_t>(((size_t)R * H + 255) / 256, 4096), 256, 0, st>>>(x, rows, R, H, out);
  CUDA_CHECK(cudaGetLastError());
}
void add_f32(float* a, const float* b, size_t n, cudaStream_t st) {
  if (!n) return;
  add_f32_kernel<<<std::min<size_t>((n + 255) / 256, 4096), 256, 0, st>>>(a, b, n);
  CUDA_CHECK(cudaGetLastError());
}

// ---- v0 (reference / comparison): gu_kernel → down_kernel → reduce_kernel, one pair per block row ------------------------------------------
size_t moe_decode_v0_ws_bytes(int n_pairs, int H, int I) { return (size_t)n_pairs * I * 2 + (size_t)n_pairs * H * 4 + 256; }
void moe_decode_v0(const ExpertRecLayout& L, const MoePair* pairs, int n_pairs, const __nv_bfloat16* x, int H, int I, float limit, float* out,
                   void* ws, cudaStream_t st) {
  if (n_pairs <= 0) return;
  __nv_bfloat16* y = reinterpret_cast<__nv_bfloat16*>(ws);
  float* part = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws) + ((size_t)n_pairs * I * 2 + 255) / 256 * 256);
  // enough blocks to fill the GPU: ~ (SMs · 4) / pairs tiles per pair
  const int tiles_gu = std::max(1, std::min(I / kWarps, 768 / n_pairs + 1));
  const int tiles_d = std::max(1, std::min(H / kWarps, 768 / n_pairs + 1));
  gu_kernel<<<dim3(tiles_gu, n_pairs), kWarps * 32, H * 2, st>>>(L, pairs, x, H, I, limit, y);
  CUDA_CHECK(cudaGetLastError());
  down_kernel<<<dim3(tiles_d, n_pairs), kWarps * 32, I * 2, st>>>(L, pairs, y, H, I, part);
  CUDA_CHECK(cudaGetLastError());
  reduce_kernel<<<(H + 255) / 256, 256, 0, st>>>(pairs, n_pairs, part, H, out);
  CUDA_CHECK(cudaGetLastError());
}

// ---- v1 -----------------------------------------------------------------------------------------------------------------------------------
namespace {
int sm_count() {
  static int n = [] { int d = 0, v = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d); return v > 0 ? v : 1; }();
  return n;
}
size_t down_smem(int n) { return chunk_head(n) + std::max(chunk_scratch(n), (size_t)n * kTN * sizeof(__nv_bfloat16)); }
// Workspace: y bf16 [n, I] + per-pair group info int [n] (used when n > kInBlockGroupMax)
size_t v1_ws_bytes(int n, int I) { return (size_t)n * I * 2 + ((size_t)n * 4 + 255) / 256 * 256 + 256; }
template <int KBG, int KBD>
void moe_decode_v1(const ExpertRecLayout& L, const MoePair* pairs, int n, const __nv_bfloat16* x, int H, int I, float limit, float* out, void* ws,
                   cudaStream_t st) {
  static const int gu_occ = [] {
    CUDA_CHECK(cudaFuncSetAttribute(gu2_kernel<KBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)chunk_smem(kMaxPairs)));
    CUDA_CHECK(cudaFuncSetAttribute(down2_kernel<KBD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)down_smem(kMaxPairs)));
    int v = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, gu2_kernel<KBG>, kV1Threads, chunk_smem(64)));
    return v > 0 ? v : 1;
  }();
  __nv_bfloat16* y = reinterpret_cast<__nv_bfloat16*>(ws);
  int* ginfo = nullptr;
  if (n > kInBlockGroupMax) {
    ginfo = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(ws) + ((size_t)n * I * 2 + 255) / 256 * 256);
    pair_info_kernel<<<(n + 7) / 8, 256, 0, st>>>(pairs, n, ginfo);
    CUDA_CHECK(cudaGetLastError());
  }
  // block items = nchunks · I/8 ≤ n · I/8 (the chunk count is only known on the device): equal rounds per block for the upper bound
  const long items = (long)n * (I / 8), slots = (long)sm_count() * gu_occ;
  const long rounds = (items + slots - 1) / slots;
  const int gu_blocks = (int)((items + rounds - 1) / rounds);
  gu2_kernel<KBG><<<gu_blocks, kV1Threads, chunk_smem(n), st>>>(L, pairs, n, ginfo, x, H, I, limit, y);
  CUDA_CHECK(cudaGetLastError());
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(H / kTN); cfg.blockDim = dim3(kV1Threads); cfg.dynamicSmemBytes = down_smem(n); cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, down2_kernel<KBD>, L, pairs, n, (const int*)ginfo, (const __nv_bfloat16*)y, H, I, out));
}
using V1Fn = void (*)(const ExpertRecLayout&, const MoePair*, int, const __nv_bfloat16*, int, int, float, float*, void*, cudaStream_t);
// v1 needs both K dims to be multiples of 1024 (one 16-byte nibble word per lane per 1024 columns), H a multiple of the down tile and
//   n ≤ kMaxPairs (shared-memory tables); everything else takes the v0 path (and gets the v0 workspace size).
V1Fn v1_for(int n, int H, int I) {
  if (n > kMaxPairs || H % 1024 || I % 1024 || H % kTN) return nullptr;
  const int kg = H / 1024, kd = I / 1024;
#define HIVE_V1(A, B) if (kg == A && kd == B) return &moe_decode_v1<A, B>;
  HIVE_V1(4, 2)  // GLM-5.3-Flash (H 4096, I 2048); add instantiations here for other shapes
#undef HIVE_V1
  return nullptr;
}
}  // namespace

// Workspace (v1): y bf16 [n_pairs, I] + group info; shapes / pair counts v1 does not cover use the v0 layout.
size_t moe_decode_ws_bytes(int n_pairs, int H, int I) { return v1_for(n_pairs, H, I) ? v1_ws_bytes(n_pairs, I) : moe_decode_v0_ws_bytes(n_pairs, H, I); }

void moe_decode(const ExpertRecLayout& L, const MoePair* pairs, int n_pairs, const __nv_bfloat16* x, int H, int I, float limit, float* out,
                void* ws, cudaStream_t st) {
  if (n_pairs <= 0) return;
  if (V1Fn f = v1_for(n_pairs, H, I)) { f(L, pairs, n_pairs, x, H, I, limit, out, ws, st); return; }
  moe_decode_v0(L, pairs, n_pairs, x, H, I, limit, out, ws, st);
}

void expert_dequant(const ExpertRecLayout& L, const uint8_t* rec, int H, int I, __nv_bfloat16* gu, __nv_bfloat16* d, cudaStream_t st) {
  float g[3];
  CUDA_CHECK(cudaMemcpyAsync(g, rec + L.g, 12, cudaMemcpyDeviceToHost, st));  // tiny; prefill path only
  CUDA_CHECK(cudaStreamSynchronize(st));
  dequant_rec_kernel<<<1024, 256, 0, st>>>(rec + L.w1, rec + L.s1, g[0], I, H, gu);
  dequant_rec_kernel<<<1024, 256, 0, st>>>(rec + L.w3, rec + L.s3, g[1], I, H, gu + (size_t)I * H);
  dequant_rec_kernel<<<1024, 256, 0, st>>>(rec + L.w2, rec + L.s2, g[2], H, I, d);
  CUDA_CHECK(cudaGetLastError());
}

void swiglu_rows(const __nv_bfloat16* gu_out, int R, int I, float limit, __nv_bfloat16* y, cudaStream_t st) {
  if (R <= 0) return;
  swiglu_rows_kernel<<<std::min<size_t>(((size_t)R * I + 255) / 256, 4096), 256, 0, st>>>(gu_out, R, I, limit, y);
  CUDA_CHECK(cudaGetLastError());
}

void scatter_add_rows(const __nv_bfloat16* e, const int32_t* rows, const float* w, int R, int H, float* out, cudaStream_t st) {
  if (R <= 0) return;
  scatter_add_kernel<<<std::min<size_t>(((size_t)R * H + 255) / 256, 4096), 256, 0, st>>>(e, rows, w, R, H, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::glm
