// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Decode GEMV for FN's small dense matrices (src/qwen/fn_gemv.h). Adapted from hive's own tensor-core GEMV (src/glm/glm_decode.cu,
//   gemv_tc_kernel: mma.m16n8k16 over 16 weight rows × ≤ 8 activation rows, 64-byte row reads, ping-pong batches). That kernel splits K
//   only inside a block, so a matrix with few 16-row tiles runs on few SMs — measured (bench_fn_gemv, 2026-10-09, M = 4):
//   hc down 324×10240 447 GB/s, shared down 2560×640 390, shared gate|up 1280×2560 759, hc up 10240×320 907, against ≈ 1,470 for the big
//   projections. Here K is also split ACROSS blocks (ks parts): every block writes fp32 partials of its 16-row tile and K part, and a small
//   pass adds the parts in part order (fixed order, no atomics) and rounds to bf16 (or keeps fp32 for the router logits).
#include <cuda_bf16.h>
#include <cstring>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>

#include "hive/common.h"
#include "hive/qwen/fn_pdl.h"
#include "hive/qwen/fn_gemv.h"

namespace hive::qwen {

namespace {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint4 ld_stream(const void* p) {  // weights: read once, no L1 allocation
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ld_stream8(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr int kTB = 2;
// block = 8 warps = 8/wt tiles of 16 rows; the wt warps of a tile split the block's K part (K/ks columns) in wt sub-parts; block
//   (tile group, K part kpi) writes part[(kpi · M + a) · N + n] (fp32), sub-parts added in order
template <int NT>   // NT activation tiles of 8 rows (M ≤ 8·NT), each weight fragment feeding NT mma
__global__ void __launch_bounds__(256, NT <= 1 ? 3 : 1) fn_gemv_kernel(const bf16* __restrict__ W, const bf16* __restrict__ x, int ldx, int M, int N, int K,
                                                          int ks, int wt, float* __restrict__ part, bf16* __restrict__ y = nullptr, int ldy = 0,
                                                          float* __restrict__ yf = nullptr) {
  // part == null (ks = 1): the result is written directly — y bf16 or yf fp32 (fn_gemv_tc)
  __shared__ float red[8][32][4 * NT];
  const int tpb = 8 / wt, tiles = (N + 15) / 16, tgroups = (tiles + tpb - 1) / tpb;
  const int tg = blockIdx.x % tgroups, kpi = blockIdx.x / tgroups;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int kpart = K / ks, ksub = kpart / wt, ns = ksub / 32;
  const int tl = warp / wt, sub = warp % wt;
  const int k0 = kpi * kpart + sub * ksub;
  const int n0 = (tg * tpb + tl) * 16;
  const int ra = n0 + g < N ? n0 + g : 0, rb = n0 + g + 8 < N ? n0 + g + 8 : 0;   // rows past N: valid memory, results dropped
  const uint4* wa = reinterpret_cast<const uint4*>(W + (size_t)ra * K + k0) + t;
  const uint4* wb = reinterpret_cast<const uint4*>(W + (size_t)rb * K + k0) + t;
  bool act[NT];
  const uint4* xr[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) { act[u] = u * 8 + g < M; xr[u] = reinterpret_cast<const uint4*>(x + (size_t)(act[u] ? u * 8 + g : 0) * ldx + k0) + t; }
  float c[NT][4];
#pragma unroll
  for (int u = 0; u < NT; ++u) c[u][0] = c[u][1] = c[u][2] = c[u][3] = 0.f;
  for (int j0 = 0; j0 < ns; j0 += kTB) {
    uint4 qa[kTB], qb[kTB];
#pragma unroll
    for (int j = 0; j < kTB; ++j)
      if (j0 + j < ns) { qa[j] = ld_stream(wa + 4 * (j0 + j)); qb[j] = ld_stream(wb + 4 * (j0 + j)); }
    if (j0 == 0) pdl_wait();   // the first weights are in flight; x comes from the previous kernel
#pragma unroll
    for (int j = 0; j < kTB; ++j)
      if (j0 + j < ns) {
#pragma unroll
        for (int u = 0; u < NT; ++u) {
          if (u * 8 >= M) break;
          const uint4 xv = act[u] ? __ldg(xr[u] + 4 * (j0 + j)) : make_uint4(0, 0, 0, 0);
          mma_bf16(c[u], qa[j].x, qb[j].x, qa[j].y, qb[j].y, xv.x, xv.y);
          mma_bf16(c[u], qa[j].z, qb[j].z, qa[j].w, qb[j].w, xv.z, xv.w);
        }
      }
  }
#pragma unroll
  for (int u = 0; u < NT; ++u)
#pragma unroll
    for (int k = 0; k < 4; ++k) red[warp][lane][u * 4 + k] = c[u][k];
  __syncthreads();
  const int MA = 8 * NT;
  for (int i = threadIdx.x; i < tpb * 16 * MA; i += 256) {  // (tile tl2, row r < 16, act a < 8·NT)
    const int tl2 = i / (16 * MA), r = (i / MA) % 16, a = i % MA;
    const int n = (tg * tpb + tl2) * 16 + r;
    if (a >= M || n >= N) continue;
    const int u = a >> 3, a8 = a & 7;
    const int ln = (r & 7) * 4 + (a8 >> 1), k = u * 4 + (a8 & 1) + (r >> 3) * 2;
    float v = 0.f;
    for (int q = 0; q < wt; ++q) v += red[tl2 * wt + q][ln][k];
    if (part) part[((size_t)kpi * M + a) * N + n] = v;
    else if (yf) yf[(size_t)a * ldy + n] = v;
    else y[(size_t)a * ldy + n] = __float2bfloat16(v);
  }
}

// y[a, n] = Σ_parts in order (bf16 out unless yf)
__global__ void fn_gemv_reduce(const float* __restrict__ part, int ks, int M, int N, bf16* __restrict__ y, int ldy, float* __restrict__ yf) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * N) return;
  const int a = i / N, n = i % N;
  float v = 0.f;
  for (int q = 0; q < ks; ++q) v += part[((size_t)q * M + a) * N + n];
  if (yf) yf[(size_t)a * ldy + n] = v;
  else y[(size_t)a * ldy + n] = __float2bfloat16(v);
}

// fused epilogues of the split-K reduce (fn_gemv_epi): 2 = hyper-connection mix down projection (dl bf16 and, for the first `rank` columns,
//   s = bf16(silu(bf16(dl / hc)))) · 3 = router logits + shared expert gate|up (columns < nr → fp32 logits; the next 2I columns → y = bf16(
//   silu(bf16(gate)) · bf16(up)) — the rounding points of the separate GEMV → hc_mix_act / swiglu_rows kernels)
__global__ void fn_gemv_reduce_epi(const float* __restrict__ part, int ks, int M, int N, FnEpi e) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (e.mode == 2) {
    if (i >= M * N) return;
    const int a = i / N, n = i % N;
    float v = 0.f;
    for (int q = 0; q < ks; ++q) v += part[((size_t)q * M + a) * N + n];
    const bf16 d = __float2bfloat16(v);
    e.y[(size_t)a * e.ldy + n] = d;
    if (n < e.rank) {
      const float t = __bfloat162float(__float2bfloat16(__bfloat162float(d) * e.inv_hc));
      e.s[(size_t)a * e.rank + n] = __float2bfloat16(t * (1.f / (1.f + expf(-t))));
    }
    return;
  }
  // mode 3: threads over (a, j < nr + I)
  const int W = e.nr + e.I;
  if (i >= M * W) return;
  const int a = i / W, j = i % W;
  auto sum = [&](int n) { float v = 0.f; for (int q = 0; q < ks; ++q) v += part[((size_t)q * M + a) * N + n]; return v; };
  if (j < e.nr) { e.yf[(size_t)a * e.ldyf + j] = sum(j); return; }
  const int ii = j - e.nr;
  const float g = __bfloat162float(__float2bfloat16(sum(e.nr + ii))), u = __bfloat162float(__float2bfloat16(sum(e.nr + e.I + ii)));
  e.ys[(size_t)a * e.I + ii] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
}

int sm_count() {
  static int n = [] { int d = 0, v = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d); return v > 0 ? v : 1; }();
  return n;
}
}  // namespace

static void fn_gemv_launch(dim3 grid, cudaStream_t st, const bf16* W, const bf16* x, int ldx, int M, int N, int K, int ks, int wt, float* ws,
                           bf16* y = nullptr, int ldy = 0, float* yf = nullptr) {
  const int nt = (M + 7) / 8;
  if (nt == 1) launch_pdl(fn_gemv_kernel<1>, grid, dim3(256), 0, st, W, x, ldx, M, N, K, ks, wt, ws, y, ldy, yf);
  else if (nt == 2) launch_pdl(fn_gemv_kernel<2>, grid, dim3(256), 0, st, W, x, ldx, M, N, K, ks, wt, ws, y, ldy, yf);
  else if (nt <= 4) launch_pdl(fn_gemv_kernel<4>, grid, dim3(256), 0, st, W, x, ldx, M, N, K, ks, wt, ws, y, ldy, yf);
  else launch_pdl(fn_gemv_kernel<8>, grid, dim3(256), 0, st, W, x, ldx, M, N, K, ks, wt, ws, y, ldy, yf);
}

// Batch-invariant bf16 GEMV for any 1..64 rows: the hive GLM tensor-core GEMV's exact schedule (K split only inside the block, wt warps per
//   tile by its rule, the same mma order and part sum) with up to 8 activation tiles per weight read — row a's result does not depend on how
//   many rows share the call (the GLM kernel takes ≤ 8 rows; cuBLAS / split-K changed the rounding with the batch size: bf16 batched
//   decode of 12 sequences diverged from single-sequence decode, 2026-10-10). y bf16 [M, ldy], or yf fp32 when given.
void fn_gemv_tc(const bf16* W, int N, int K, const bf16* x, int ldx, int M, bf16* y, int ldy, float* yf, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0 && M >= 1 && M <= 64 && ldx % 8 == 0 && ((uintptr_t)W % 16) == 0 && ((uintptr_t)x % 16) == 0, "fn_gemv_tc: shape / alignment");
  int wt = 1;
  while (wt < 8 && K % (2 * wt * 32) == 0 && K / (2 * wt) >= 512) wt *= 2;
  const int tiles = (N + 15) / 16, tpb = 8 / wt;
  fn_gemv_launch(dim3((tiles + tpb - 1) / tpb), st, W, x, ldx, M, N, K, 1, wt, nullptr, y, ldy, yf);
  CUDA_CHECK(cudaGetLastError());
}

// split: wt warps per tile × ks parts across blocks, each warp ≥ one 32-column chunk; the largest wt · ks (≤ 64) that divides K's chunk
//   count, preferring parts across blocks while the grid is under 2 blocks per SM
struct Split { int ks = 0, wt = 0; };
Split fn_split(int N, int K) {
  Split sp;
  if (K % 32) return sp;
  const int chunks = K / 32, tiles = (N + 15) / 16;
  int best = -1;
  for (int wt = 1; wt <= 8; wt *= 2)
    for (int ks = 1; ks <= 16; ks *= 2) {
      if (chunks % (wt * ks)) continue;
      const long blocks = (long)(tiles + 8 / wt - 1) / (8 / wt) * ks;
      // score: enough blocks first (up to 3 per SM), then fewer K parts (less partial traffic)
      const long score = std::min<long>(blocks, 3L * sm_count()) * 64 - ks;
      if (score > best) { best = (int)score; sp.ks = ks; sp.wt = wt; }
    }
  return sp;
}
int fn_gemv_parts(int N, int K) { return fn_split(N, K).ks; }

bool fn_gemv_ok(int N, int K, int M) { return M >= 1 && M <= 64 && N > 0 && fn_gemv_parts(N, K) > 0; }

size_t fn_gemv_ws_bytes(int N, int K, int M) { return (size_t)std::max(1, fn_gemv_parts(N, K)) * M * N * 4; }

void fn_gemv(const bf16* W, int N, int K, const bf16* x, int ldx, int M, bf16* y, int ldy, float* yf, float* ws, cudaStream_t st) {
  const Split sp = fn_split(N, K);
  HIVE_CHECK(sp.ks > 0 && M >= 1 && M <= 64 && ldx % 8 == 0 && ((uintptr_t)W % 16) == 0 && ((uintptr_t)x % 16) == 0, "fn_gemv: shape / alignment");
  const int ks = sp.ks, tpb = 8 / sp.wt, tiles = (N + 15) / 16;
  fn_gemv_launch(dim3((tiles + tpb - 1) / tpb * ks), st, W, x, ldx, M, N, K, ks, sp.wt, ws);
  CUDA_CHECK(cudaGetLastError());
  launch_pdl(fn_gemv_reduce, dim3((M * N + 255) / 256), dim3(256), 0, st, ws, ks, M, N, y, ldy, yf);
  CUDA_CHECK(cudaGetLastError());
}

void fn_gemv_epi(const bf16* W, int N, int K, const bf16* x, int ldx, int M, const FnEpi& e, float* ws, cudaStream_t st) {
  const Split sp = fn_split(N, K);
  HIVE_CHECK(sp.ks > 0 && M >= 1 && M <= 64 && ldx % 8 == 0 && ((uintptr_t)W % 16) == 0 && ((uintptr_t)x % 16) == 0 && (e.mode == 2 || e.mode == 3),
             "fn_gemv_epi: shape / alignment / mode");
  HIVE_CHECK(e.mode != 3 || e.nr + 2 * e.I == N, "fn_gemv_epi: router + shared rows");
  const int ks = sp.ks, tpb = 8 / sp.wt, tiles = (N + 15) / 16;
  fn_gemv_launch(dim3((tiles + tpb - 1) / tpb * ks), st, W, x, ldx, M, N, K, ks, sp.wt, ws);
  CUDA_CHECK(cudaGetLastError());
  const int items = e.mode == 2 ? M * N : M * (e.nr + e.I);
  launch_pdl(fn_gemv_reduce_epi, dim3((items + 255) / 256), dim3(256), 0, st, ws, ks, M, N, e);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen

// ---- 8-bit dense weights (HIVE_QWEN_DENSE8): FP8 e4m3 or INT8 [N, K] with fp32 scales per (row, group of gs columns) --------------------------
//   W[n, k] = q[n, k] · s[n, k / gs]. The weights are expanded to bf16 in registers (exact for both formats) and fed to the same mma as the
//   bf16 GEMV; each group's partial sums are scaled when the group ends. Block = 8 warps = 8/wt tiles of 16 rows, wt warps per tile split K.
namespace hive::qwen {
namespace {
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
// 16 weights (16 bytes) → 8 packed bf16 pairs (exact)
template <int FMT>
__device__ __forceinline__ void dec16(uint4 q, uint32_t (&o)[8]) {
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const uint32_t h = (w[j] >> (16 * b)) & 0xFFFF;
      float f0, f1;
      if (FMT == 1) {   // e4m3
        const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)h, __NV_E4M3);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
        f0 = f.x; f1 = f.y;
      } else {          // int8
        f0 = (float)(int8_t)(h & 0xFF); f1 = (float)(int8_t)(h >> 8);
      }
      o[j * 2 + b] = pack_bf16(f0, f1);
    }
}
// Lane (g, t) streams 16 bytes (16 columns) of weight rows g and g+8 per step at columns 64·j + 16t, and the same 16 activation columns
//   (two 16-byte words); the 16 columns feed 4 mma steps (k slots {0,1},{8,9} of each — the same permutation on both operands, so the dot
//   product is unchanged). Batches of kB8 steps are requested ahead (ping-pong), like the bf16 kernel.
constexpr int kB8 = 4;
// NT activation tiles of 8 rows (M ≤ 8·NT): every weight fragment feeds NT mma, so a batch of verify rows reads the weights once
template <int FMT, int NT, int NWM = 8, bool LEAN = false>
__global__ void __launch_bounds__(32 * NWM, LEAN ? 2 : NT <= 1 ? 3 : 1) gemv8_kernel(const uint8_t* __restrict__ W, const float* __restrict__ S, int gs, const bf16* __restrict__ x,
                                                        int ldx, int M, int N, int K, int wt, bf16* __restrict__ y, int ldy, float* __restrict__ yf,
                                                        int ks, float* __restrict__ part) {
  // ks > 1 (split K across blocks, part != null): block = (tile group, K part kpi) writes fp32 partials part[(kpi · M + a) · N + n]
  __shared__ float red[NWM][32][4 * NT];   // NWM 16: 9..32 rows (HIVE_QWEN_GEMV8_NWB)
  const int tpb = (int)(blockDim.x >> 5) / wt;   // blocks of 8 warps, or fewer for matrices with few tiles (fn_gemv8)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tl = warp / wt, sub = warp % wt;
  const int tgroups = ((N + 15) / 16 + tpb - 1) / tpb, tg = (int)blockIdx.x % tgroups, kpi = (int)blockIdx.x / tgroups;
  const int kp = K / ks / wt, k0 = kpi * (K / ks) + sub * kp, ns = kp / 64;
  const int n0 = (tg * tpb + tl) * 16;
  const int ra = n0 + g < N ? n0 + g : 0, rb = n0 + g + 8 < N ? n0 + g + 8 : 0;
  const uint4* wa = reinterpret_cast<const uint4*>(W + (size_t)ra * K + k0) + t;
  const uint4* wb = reinterpret_cast<const uint4*>(W + (size_t)rb * K + k0) + t;
  const int ng = (K + gs - 1) / gs;
  const float* sa = S + (size_t)ra * ng;
  const float* sb = S + (size_t)rb * ng;
  bool act[NT];
  const uint4* xr[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) {
    act[u] = u * 8 + g < M;
    xr[u] = reinterpret_cast<const uint4*>(x + (size_t)(act[u] ? u * 8 + g : 0) * ldx + k0) + 2 * t;
  }
  float c[NT][4], cg[NT][4];
#pragma unroll
  for (int u = 0; u < NT; ++u)
#pragma unroll
    for (int k = 0; k < 4; ++k) { c[u][k] = 0.f; cg[u][k] = 0.f; }
  int grp = k0 / gs;
  constexpr int KB = LEAN ? 2 : kB8;   // LEAN: 2 blocks per SM (≤ 128 registers) with half the weight batch in flight per warp
  uint4 qa[2][KB], qb[2][KB];
  auto load = [&](int buf, int j0) {
#pragma unroll
    for (int j = 0; j < KB; ++j)
      if (j0 + j < ns) { qa[buf][j] = ld_stream(wa + 4 * (j0 + j)); qb[buf][j] = ld_stream(wb + 4 * (j0 + j)); }
  };
  auto flush = [&] {
    const float s0 = __ldg(sa + grp), s1 = __ldg(sb + grp);
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      c[u][0] = fmaf(cg[u][0], s0, c[u][0]); c[u][1] = fmaf(cg[u][1], s0, c[u][1]); c[u][2] = fmaf(cg[u][2], s1, c[u][2]); c[u][3] = fmaf(cg[u][3], s1, c[u][3]);
      cg[u][0] = cg[u][1] = cg[u][2] = cg[u][3] = 0.f;
    }
  };
  auto compute = [&](int buf, int j0) {
#pragma unroll
    for (int j = 0; j < KB; ++j) {
      if (j0 + j >= ns) break;
      const int col = k0 + 64 * (j0 + j);
      if (col / gs != grp) { flush(); grp = col / gs; }   // group ended: scale its partial sums into the totals
      uint32_t A[8], B[8];
      dec16<FMT>(qa[buf][j], A); dec16<FMT>(qb[buf][j], B);
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        if (u * 8 >= M) break;
        const uint4 x0 = act[u] ? __ldg(xr[u] + 8 * (j0 + j)) : make_uint4(0, 0, 0, 0);
        const uint4 x1 = act[u] ? __ldg(xr[u] + 8 * (j0 + j) + 1) : make_uint4(0, 0, 0, 0);
        mma_bf16(cg[u], A[0], B[0], A[1], B[1], x0.x, x0.y);
        mma_bf16(cg[u], A[2], B[2], A[3], B[3], x0.z, x0.w);
        mma_bf16(cg[u], A[4], B[4], A[5], B[5], x1.x, x1.y);
        mma_bf16(cg[u], A[6], B[6], A[7], B[7], x1.z, x1.w);
      }
    }
  };
  load(0, 0);
  load(1, KB);
  pdl_wait();   // weights of the first two batches in flight; x comes from the previous kernel
  for (int j0 = 0; j0 < ns; j0 += 2 * KB) {
    compute(0, j0);
    load(0, j0 + 2 * KB);
    compute(1, j0 + KB);
    load(1, j0 + 3 * KB);
  }
  flush();
#pragma unroll
  for (int u = 0; u < NT; ++u)
#pragma unroll
    for (int k = 0; k < 4; ++k) red[warp][lane][u * 4 + k] = c[u][k];
  __syncthreads();
  const int MA = 8 * NT;
  for (int i = threadIdx.x; i < tpb * 16 * MA; i += blockDim.x) {
    const int tl2 = i / (16 * MA), r = (i / MA) % 16, a = i % MA;
    const int n = (tg * tpb + tl2) * 16 + r;
    if (a >= M || n >= N) continue;
    const int u = a >> 3, a8 = a & 7;
    const int ln = (r & 7) * 4 + (a8 >> 1), k = u * 4 + (a8 & 1) + (r >> 3) * 2;
    float v = 0.f;
    for (int q = 0; q < wt; ++q) v += red[tl2 * wt + q][ln][k];
    if (part) part[((size_t)kpi * M + a) * N + n] = v;
    else if (yf) yf[(size_t)a * ldy + n] = v;
    else y[(size_t)a * ldy + n] = __float2bfloat16(v);
  }
}

// w[n, k] = bf16(q · s) — the bf16 operand of the prefill GEMMs when the bf16 original was dropped (HIVE_QWEN_DENSE8)
template <int FMT>
__global__ void dequant8_kernel(const uint8_t* __restrict__ q, const float* __restrict__ s, int gs, int N, int K, bf16* __restrict__ w) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= (size_t)N * K) return;
  const int n = (int)(i / K), k = (int)(i % K), ng = (K + gs - 1) / gs;
  const float sc = s[(size_t)n * ng + k / gs];
  const uint32_t v = *reinterpret_cast<const uint32_t*>(q + i);
  float f[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint8_t b = (uint8_t)(v >> (8 * j));
    if (FMT == 1) { __nv_fp8_e4m3 t; t.__x = b; f[j] = float(t); } else f[j] = (float)(int8_t)b;
  }
  *reinterpret_cast<__nv_bfloat162*>(w + i) = __floats2bfloat162_rn(f[0] * sc, f[1] * sc);
  *reinterpret_cast<__nv_bfloat162*>(w + i + 2) = __floats2bfloat162_rn(f[2] * sc, f[3] * sc);
}

// q[n, k] = round(w / s), s = max |w| over the group / (448 for e4m3, 127 for int8)
template <int FMT>
__global__ void quant8_kernel(const bf16* __restrict__ w, int N, int K, int gs, uint8_t* __restrict__ q, float* __restrict__ s) {
  const int ng = (K + gs - 1) / gs;
  const int n = blockIdx.x, grp = blockIdx.y;
  const int c0 = grp * gs, c1 = min(K, c0 + gs);
  __shared__ float red[256];
  float amax = 0.f;
  for (int k = c0 + threadIdx.x; k < c1; k += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(w[(size_t)n * K + k])));
  red[threadIdx.x] = amax;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
  const float sc = red[0] > 0.f ? red[0] / (FMT == 1 ? 448.f : 127.f) : 1.f;
  if (threadIdx.x == 0) s[(size_t)n * ng + grp] = sc;
  const float inv = 1.f / sc;
  for (int k = c0 + threadIdx.x; k < c1; k += blockDim.x) {
    const float v = __bfloat162float(w[(size_t)n * K + k]) * inv;
    if (FMT == 1) q[(size_t)n * K + k] = __nv_fp8_e4m3(v).__x;
    else q[(size_t)n * K + k] = (uint8_t)(int8_t)__float2int_rn(fminf(127.f, fmaxf(-127.f, v)));
  }
}
}  // namespace

bool fn_gemv8_ok(int N, int K, int gs, int M) { return M >= 1 && M <= 64 && N > 0 && K % 64 == 0 && gs % 64 == 0 && gs > 0; }

void fn_quant8(const bf16* w, int N, int K, int fmt, int gs, uint8_t* q, float* s, cudaStream_t st) {
  const dim3 grid(N, (K + gs - 1) / gs);
  if (fmt == 1) quant8_kernel<1><<<grid, 256, 0, st>>>(w, N, K, gs, q, s);
  else quant8_kernel<2><<<grid, 256, 0, st>>>(w, N, K, gs, q, s);
  CUDA_CHECK(cudaGetLastError());
}

void fn_gemv8(const uint8_t* W, const float* S, int fmt, int gs, int N, int K, const bf16* x, int ldx, int M, bf16* y, int ldy, cudaStream_t st, float* yf) {
  HIVE_CHECK(fn_gemv8_ok(N, K, gs, M) && ldx % 8 == 0 && ((uintptr_t)W % 16) == 0 && ((uintptr_t)x % 16) == 0, "fn_gemv8: shape / alignment");
  int wt = 1;   // warps per tile: K parts of ≥ 512 columns (as the bf16 kernel), at most 8, each a multiple of 64 columns
  while (wt < 8 && K % (2 * wt * 64) == 0 && K / (2 * wt) >= 512) wt *= 2;
  // warps per block: 8, halved while the grid stays under 2 blocks per SM (hc up 10240 × 320: 80 blocks of 8 warps on 188 SMs) —
  //   HIVE_QWEN_GEMV8_MINW (default 4: hc up 6.7 → 6.2 µs, shared gate|up 10.2 → 8.6 µs at M = 4; 1 and 2 were slower — 2026-10-10) is the floor
  static const int minw = getenv("HIVE_QWEN_GEMV8_MINW") ? std::max(1, std::min(8, atoi(getenv("HIVE_QWEN_GEMV8_MINW")))) : 4;
  const int tiles = (N + 15) / 16;
  int nw = 8;
  while (nw / 2 >= std::max(wt, minw) && (tiles + nw / wt - 1) / (nw / wt) < 2 * sm_count()) nw /= 2;
  const int nt = (M + 7) / 8;
  // 9..32 rows: x is read once per row tile of N per K part (x traffic = 2·M/16 bytes per weight byte: 4× the weights at 32 rows), so more
  //   tiles per block share it through L1 — HIVE_QWEN_GEMV8_NWB warps per block (8 or 16); the per-tile arithmetic is unchanged
  static const int nwb = getenv("HIVE_QWEN_GEMV8_NWB") ? (atoi(getenv("HIVE_QWEN_GEMV8_NWB")) >= 16 ? 16 : 8) : 8;
  if (fmt == 2 && nt >= 2 && nt <= 4 && nwb == 16 && (tiles + 16 / wt - 1) / (16 / wt) >= sm_count()) nw = 16;
  // 9..32 rows: LEAN instances (2 blocks per SM, 128 registers instead of 181 / 155 — one block per SM left too few loads in flight);
  //   HIVE_QWEN_GEMV8_LEAN=0 restores the old instances. bench_fn_gemv 2026-10-10: gdn_in 16 rows 41.0 → 35.0 us, 32 rows 55.6 → 46.7,
  //   lm_head 449 → 408 / 608 → 488 us; same arithmetic and order (only the weight batch in flight changes)
  static const bool lean = !(getenv("HIVE_QWEN_GEMV8_LEAN") && strcmp(getenv("HIVE_QWEN_GEMV8_LEAN"), "0") == 0);
  const int tpb = nw / wt, blocks = (tiles + tpb - 1) / tpb;
  auto go = [&](auto kern) { launch_pdl(kern, dim3(blocks), dim3(32 * nw), 0, st, W, S, gs, x, ldx, M, N, K, wt, y, ldy, yf, 1, (float*)nullptr); };
  if (fmt == 1) {
    if (nt == 1) go(gemv8_kernel<1, 1>); else if (nt == 2) go(gemv8_kernel<1, 2>); else if (nt <= 4) go(gemv8_kernel<1, 4>); else go(gemv8_kernel<1, 8>);
  } else {
    if (nt == 1) go(gemv8_kernel<2, 1>);
    else if (nt == 2) nw == 16 ? go(gemv8_kernel<2, 2, 16>) : lean ? go(gemv8_kernel<2, 2, 8, true>) : go(gemv8_kernel<2, 2>);
    else if (nt <= 4) nw == 16 ? go(gemv8_kernel<2, 4, 16>) : lean ? go(gemv8_kernel<2, 4, 8, true>) : go(gemv8_kernel<2, 4>);
    else go(gemv8_kernel<2, 8>);
  }
  CUDA_CHECK(cudaGetLastError());
}
void fn_dequant8(const uint8_t* q, const float* s, int fmt, int gs, int N, int K, __nv_bfloat16* w, cudaStream_t st) {
  HIVE_CHECK(K % 4 == 0 && gs % 4 == 0, "fn_dequant8 shape");
  const size_t n4 = (size_t)N * K / 4;
  const unsigned blocks = (unsigned)((n4 + 255) / 256);
  if (fmt == 1) dequant8_kernel<1><<<blocks, 256, 0, st>>>(q, s, gs, N, K, w);
  else dequant8_kernel<2><<<blocks, 256, 0, st>>>(q, s, gs, N, K, w);
  CUDA_CHECK(cudaGetLastError());
}
// split-K 8-bit GEMV with the fn_gemv epilogues (the hyper-connection down projection, 324 × 10240: few row tiles, a long K)
int fn_gemv8_split_parts(int N, int K, int gs) {
  const Split sp = fn_split(N, K);
  int ks = sp.ks;
  while (ks > 1 && ((K / ks) % 64 || (K / ks) % gs)) ks /= 2;
  return ks;
}
void fn_gemv8_epi(const uint8_t* W, const float* S, int fmt, int gs, int N, int K, const bf16* x, int ldx, int M, const FnEpi& e, float* ws, cudaStream_t st) {
  const int ks = fn_gemv8_split_parts(N, K, gs);
  HIVE_CHECK(ks >= 1 && M >= 1 && M <= 64 && K % 64 == 0 && ldx % 8 == 0 && (e.mode == 2 || e.mode == 3), "fn_gemv8_epi: shape");
  int wt = 1;
  while (wt < 8 && (K / ks) % (2 * wt * 64) == 0 && (K / ks) / (2 * wt) >= 256) wt *= 2;
  const int nw = 8, tpb = nw / wt, tiles = (N + 15) / 16, blocks = (tiles + tpb - 1) / tpb * ks;
  auto go = [&](auto kern) { launch_pdl(kern, dim3(blocks), dim3(32 * nw), 0, st, W, S, gs, x, ldx, M, N, K, wt, (bf16*)nullptr, 0, (float*)nullptr, ks, ws); };
  const int nt = (M + 7) / 8;
  if (fmt == 1) {
    if (nt == 1) go(gemv8_kernel<1, 1>); else if (nt == 2) go(gemv8_kernel<1, 2>); else if (nt <= 4) go(gemv8_kernel<1, 4>); else go(gemv8_kernel<1, 8>);
  } else {
    if (nt == 1) go(gemv8_kernel<2, 1>); else if (nt == 2) go(gemv8_kernel<2, 2>); else if (nt <= 4) go(gemv8_kernel<2, 4>); else go(gemv8_kernel<2, 8>);
  }
  const int items = e.mode == 2 ? M * N : M * (e.nr + e.I);
  launch_pdl(fn_gemv_reduce_epi, dim3((items + 255) / 256), dim3(256), 0, st, ws, ks, M, N, e);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen
