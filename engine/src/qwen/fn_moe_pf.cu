// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next routed-expert prefill on the GPU when every expert of the layer is resident: one grouped W4A16 GEMM pass instead of
//   GlmExperts::prefill_layer's per-expert path (host routing, an NVFP4 → bf16 dequant of every routed expert, a cuBLAS GEMM per expert
//   and per-expert gather / swiglu / scatter launches). nsys 2026-10-09 (one 16K-token prefill, 3.8 s): 50,895 dequant launches
//   (265 ms), ~17K gather / swiglu / scatter launches each, 17K cudaMemcpyAsync calls (2.6 s of host API time).
//   Per sub-chunk of rows (the per-pair buffers are sized for it):
//   (1) count / scan / fill: the (row, k) pairs sorted by expert (counting sort; the order inside an expert does not matter — every
//       pair's result depends only on its own row), and a list of 128-pair tiles (a warp skips its 64 rows when they are all past the tile);
//   (2) gate|up GEMM: block = (tile, 64 rows of I); the expert's NVFP4 gate and up rows are expanded to bf16 in shared memory (e2m1 ×
//       e4m3 scale is exact in bf16), x rows gathered with cp.async, bf16 mma.sync m16n8k16 with fp32 accumulation; epilogue
//       g = bf16(acc·g1), u = bf16(acc·g3), y = bf16(silu(g)·u) — the decode kernels' rounding points;
//   (3) down GEMM: block = (tile, 128 rows of H), y rows of the tile are contiguous; part = bf16(acc·g2) per (row, k) pair;
//   (4) reduce: out[row] = Σ_k part·w in k order (the decode reduce order; deterministic, no atomics on the outputs).
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>

#include "hive/common.h"
#include "hive/qwen/fn_moe.h"

namespace hive::qwen {

namespace {
using bf16 = __nv_bfloat16;
constexpr int BM = 128, BMS = 32, BK = 64, PK = BK + 8, kThreads = 256;   // PK: padded row (bf16) — ldmatrix rows land on distinct banks
constexpr int kGuN = 64;                                        // I rows per gate|up block (64 gate + 64 up = 128 B rows)
constexpr int kDnN = 128;                                       // H rows per down block
constexpr int BNmax = 128;
constexpr int a_stages(int bm) { return bm == 128 ? 2 : 4; }   // A (gathered rows) stages: deeper for the short-prompt tiles
constexpr size_t smem_for(int bm, int bn = BNmax) {   // A stages + B double buffer; the gate|up epilogue reuses it as fp32 [bm][bn + 4]
  return std::max((size_t)a_stages(bm) * bm * PK * 2 + (size_t)2 * bn * PK * 2, (size_t)bm * (bn + 4) * 4);
}

struct PfTile { int e, slot0, m; };

__global__ void pf_count_kernel(const int32_t* __restrict__ ids, int P, int E, int* __restrict__ cnt) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  const int e = ids[p];
  if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
}
// one block of 1024 threads (E ≤ 1024): exclusive scans of the counts and of the tile counts (ceil(count / bm)) → fill cursors, tile list
__global__ void __launch_bounds__(1024) pf_scan_kernel(const int* __restrict__ cnt, int E, int bm, int* __restrict__ cursor, PfTile* __restrict__ tiles,
                                                       int* __restrict__ n_tiles) {
  __shared__ int sc[1024], st[1024];
  const int e = threadIdx.x;
  const int c = e < E ? cnt[e] : 0, t = (c + bm - 1) / bm;
  sc[e] = c; st[e] = t;
  __syncthreads();
  for (int o = 1; o < 1024; o <<= 1) {   // inclusive Hillis–Steele scans
    const int a = e >= o ? sc[e - o] : 0, b2 = e >= o ? st[e - o] : 0;
    __syncthreads();
    sc[e] += a; st[e] += b2;
    __syncthreads();
  }
  if (e < E) {
    const int off = sc[e] - c, toff = st[e] - t;
    cursor[e] = off;
    for (int i = 0; i < t; ++i) tiles[toff + i] = PfTile{e, off + i * bm, min(bm, c - i * bm)};
  }
  if (e == 1023) *n_tiles = st[1023];
}
__global__ void pf_fill_kernel(const int32_t* __restrict__ ids, int P, int E, int* __restrict__ cursor, int* __restrict__ perm) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  const int e = ids[p];
  if (e >= 0 && e < E) perm[atomicAdd(&cursor[e], 1)] = p;
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldm4(unsigned (&r)[4], const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float e4m3f(uint8_t v) { __nv_fp8_e4m3 t; t.__x = v; return float(t); }
__device__ __forceinline__ unsigned pack_bf2(float a, float b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const unsigned*>(&h);
}
// 32 e2m1 values (16 bytes, low nibble first) of two 16-element scale blocks → 32 bf16 at dst (exact: e2m1 · e4m3 has ≤ 6 significant bits)
__device__ __forceinline__ void dequant32(uint4 q, float s0, float s1, bf16* dst) {
  const unsigned w[4] = {q.x, q.y, q.z, q.w};
  unsigned o[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float s = j < 2 ? s0 : s1;
#pragma unroll
    for (int b = 0; b < 4; b += 1) {
      const __nv_fp4x2_storage_t byte = (__nv_fp4x2_storage_t)((w[j] >> (8 * b)) & 0xFF);
      const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1);
      const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
      o[j * 4 + b] = pack_bf2(f.x * s, f.y * s);
    }
  }
  uint4* d = reinterpret_cast<uint4*>(dst);
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}

// GU = true: gate|up (A = gathered x rows, Kd = H, B rows: 64 gate + 64 up rows of I) → y; false: down (A = y rows of the tile, Kd = I,
//   B = 128 rows of H) → part
// GN (gate|up only): I rows per block — 128 halves the gathered-x traffic (each block re-reads the tile's x rows: as many bytes as its
//   weights at 64) for the same per-element arithmetic
template <bool GU, int BMT, int GN = kGuN, int DN = kDnN>
__global__ void __launch_bounds__(kThreads) pf_gemm_kernel(glm::ExpertRecLayout L, const uint8_t* const* __restrict__ tab, const PfTile* __restrict__ tiles,
                                                           const int* __restrict__ n_tiles, const int* __restrict__ perm, int K, const bf16* __restrict__ a_src,
                                                           int H, int I, bf16* __restrict__ out) {
  if ((int)blockIdx.x >= *n_tiles) return;
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* As = reinterpret_cast<bf16*>(smem);                       // [2][BM][PK]
  constexpr int BN = GU ? 2 * GN : DN, RB = BN / 128;   // B rows per block, B rows per thread
  constexpr int WM = BMT == 128 ? 2 : 1, WN = 8 / WM, WROWS = BMT / WM, MT = WROWS / 16, WCOLS = BN / WN, NTW = WCOLS / 8, NP = NTW / 2;
  constexpr int AI = BMT / 32;   // 16-byte A chunks per thread per k-tile
  constexpr int SA = a_stages(BMT);
  bf16* Bs = As + SA * BMT * PK;                                   // [2][BN][PK]
  const PfTile tl = tiles[blockIdx.x];
  const uint8_t* rec = tab[tl.e];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp % WN;
  const int Kd = GU ? H : I;
  const int nb0 = blockIdx.y * (GU ? GN : DN);
  // this thread's A rows (2 chunks of 16 B per k-tile) and B row (half of 64 k)
  const bf16* arow[AI];
  int acol[AI], arr[AI];
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int c = tid + i * kThreads, r = c >> 3;
    arr[i] = r; acol[i] = (c & 7) * 8;
    const int rr = r < tl.m ? r : 0;
    arow[i] = GU ? a_src + (size_t)(perm[tl.slot0 + rr] / K) * H : a_src + (size_t)(tl.slot0 + rr) * I;
  }
  const int bh = tid & 1;
  const uint8_t* bw[RB];
  const uint8_t* bs[RB];
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    const int br = (tid >> 1) + r * 128;
    if (GU) {
      const int i = nb0 + (br % GN);
      bw[r] = rec + (br < GN ? L.w1 : L.w3) + (size_t)i * (H / 2);
      bs[r] = rec + (br < GN ? L.s1 : L.s3) + (size_t)i * (H / 16);
    } else {
      const int n = nb0 + br;
      bw[r] = rec + L.w2 + (size_t)n * (I / 2);
      bs[r] = rec + L.s2 + (size_t)n * (I / 16);
    }
  }
  auto load_a = [&](int buf, int k0) {
#pragma unroll
    if (k0 >= 0)
      for (int i = 0; i < AI; ++i) cp16(As + ((size_t)buf * BMT + arr[i]) * PK + acol[i], arow[i] + k0 + acol[i]);
    cp_commit();   // (an empty group past the last tile keeps the group count — and the wait below — uniform)
  };
  // B (the NVFP4 weight stream) is requested 3 k-tiles ahead into a 4-slot register ring (slots are compile-time indices: the k loop is
  //   unrolled by 4) — with one tile ahead the stream ran at ~600-800 GB/s on short prompts (nsys 2026-10-10, 1K tokens)
  uint4 bq[4][RB]; uint16_t bsc[4][RB];
  auto load_b = [&](int slot, int k0) {
    const int kk = k0 + bh * 32;
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      bq[slot][r] = __ldg(reinterpret_cast<const uint4*>(bw[r] + kk / 2));
      bsc[slot][r] = __ldg(reinterpret_cast<const uint16_t*>(bs[r] + kk / 16));
    }
  };
  auto store_b = [&](int buf, int slot) {
#pragma unroll
    for (int r = 0; r < RB; ++r)
      dequant32(bq[slot][r], e4m3f((uint8_t)(bsc[slot][r] & 0xFF)), e4m3f((uint8_t)(bsc[slot][r] >> 8)),
                Bs + ((size_t)buf * BN + (tid >> 1) + r * 128) * PK + bh * 32);
  };
  float acc[MT][NTW][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NTW; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  const int nk = Kd / BK;
#pragma unroll
  for (int q = 0; q < SA - 1; ++q) load_a(q, q < nk ? q * BK : -1);
#pragma unroll
  for (int q = 0; q < 3; ++q) if (q < nk) load_b(q, q * BK);
  store_b(0, 0);
  cp_wait<SA - 2>();
  __syncthreads();
  for (int kt0 = 0; kt0 < nk; kt0 += 4)
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    const int kt = kt0 + jj;
    if (kt >= nk) break;
    const int cur = jj & 1, nxt = cur ^ 1;
    if (kt + 3 < nk) load_b((jj + 3) & 3, (kt + 3) * BK);
    load_a((kt + SA - 1) % SA, kt + SA - 1 < nk ? (kt + SA - 1) * BK : -1);
    const bf16* Ab = As + (size_t)(kt % SA) * BMT * PK;
    const bf16* Bb = Bs + (size_t)cur * BN * PK;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      if (wm * WROWS >= tl.m) break;   // this warp's rows are all past the tile's pairs (warp-uniform)
      unsigned af[MT][4], bfr[NP][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) ldm4(af[mt], Ab + (size_t)(wm * WROWS + mt * 16 + (lane & 15)) * PK + ks * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int np = 0; np < NP; ++np)
        ldm4(bfr[np], Bb + (size_t)(wn * WCOLS + np * 16 + (lane & 7) + (lane >> 4) * 8) * PK + ks * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) mma16816(acc[mt][nt], af[mt], bfr[nt >> 1][(nt & 1) * 2], bfr[nt >> 1][(nt & 1) * 2 + 1]);
    }
    if (kt + 1 < nk) store_b(nxt, (jj + 1) & 3);
    cp_wait<SA - 2>();   // A of tile kt + 1 has landed (SA − 2 newer groups may still be in flight)
    __syncthreads();
  }
  const int g = lane >> 2, q2 = (lane & 3) * 2;
  if (GU) {
    float* Cs = reinterpret_cast<float*>(smem);   // [BM][BN + 4] (the pipeline buffers are free after the last barrier)
    constexpr int CS = BN + 4;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NTW; ++nt) {
        const int r = wm * WROWS + mt * 16 + g, c = wn * WCOLS + nt * 8 + q2;
        Cs[r * CS + c] = acc[mt][nt][0]; Cs[r * CS + c + 1] = acc[mt][nt][1];
        Cs[(r + 8) * CS + c] = acc[mt][nt][2]; Cs[(r + 8) * CS + c + 1] = acc[mt][nt][3];
      }
    __syncthreads();
    const float g1 = *reinterpret_cast<const float*>(rec + L.g), g3 = *reinterpret_cast<const float*>(rec + L.g + 4);
    for (int idx = tid; idx < BMT * GN; idx += kThreads) {
      const int r = idx / GN, i = idx % GN;
      if (r >= tl.m) break;
      const float gv = __bfloat162float(__float2bfloat16(Cs[r * CS + i] * g1));
      const float uv = __bfloat162float(__float2bfloat16(Cs[r * CS + GN + i] * g3));
      out[(size_t)(tl.slot0 + r) * I + nb0 + i] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
    }
  } else {
    const float g2 = *reinterpret_cast<const float*>(rec + L.g + 8);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        const int r = wm * WROWS + mt * 16 + g + h2 * 8;
        if (r >= tl.m) continue;
        bf16* o = out + (size_t)perm[tl.slot0 + r] * H + nb0;
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
          const int c = wn * WCOLS + nt * 8 + q2;
          *reinterpret_cast<__nv_bfloat162*>(o + c) = __floats2bfloat162_rn(acc[mt][nt][h2 * 2] * g2, acc[mt][nt][h2 * 2 + 1] * g2);
        }
      }
  }
}

// out[row, :] = Σ_k part[row·K + k, :] · w[row·K + k] in k order (pairs with an invalid id contribute nothing)
__global__ void pf_reduce_kernel(const bf16* __restrict__ part, const int32_t* __restrict__ ids, const float* __restrict__ w, int M, int K, int E,
                                 int H, float* __restrict__ out) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (i >= (size_t)M * H) return;
  const int m = (int)(i / H), n = (int)(i % H);
  float a0 = 0.f, a1 = 0.f;
  for (int k = 0; k < K; ++k) {
    const int p = m * K + k, e = ids[p];
    if (e < 0 || e >= E) continue;
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(part + (size_t)p * H + n));
    const float wv = w[p];
    a0 += v.x * wv; a1 += v.y * wv;
  }
  out[i] = a0; out[i + 1] = a1;
}

// ---- W4A16 v2 (default; HIVE_QWEN_MOE_PF_V2=0 = the kernels above) ------------------------------------------------------------------------
// ncu 2026-10-31 (pf_gemm_kernel<true, 128>, 8,192 rows, 146 TFLOP/s of the 551 TFLOP/s bf16 mma.sync peak): stalls wait / mio_throttle /
//   barrier at 8 warps per SM — every block expands its NVFP4 weight tile to bf16 in shared memory (STS of 4× the bytes, then ldmatrix).
//   Here the NVFP4 bytes and their e4m3 scales go to shared memory as they are (cp.async, 3 stages) and each lane converts the two bytes of
//   its own B fragment (k 2t, 2t+1 and 2t+8, 2t+9 of output column g) — e2m1 → half2 (hardware), × scale, → bf16x2: the same exact values
//   the expansion produced — right before the mma. 64 × 64 warp tiles (8 warps over 128 rows × 256 columns), gate|up rows interleaved so the
//   epilogue stays in registers. Same mma.sync m16n8k16 in the same k order, same epilogue arithmetic: bit-identical to the kernels above.
constexpr int W2_BK = 64, W2_ST = 3, W2_BN = 256;
constexpr int W2_AB = W2_BK * 2;     // A row bytes per stage (64 bf16)
constexpr int W2_BB = W2_BK / 2;     // B row bytes per stage (64 e2m1)
constexpr size_t w2_smem(int bm) { return (size_t)W2_ST * ((size_t)bm * W2_AB + (size_t)W2_BN * (W2_BB + 4)); }
__device__ __forceinline__ int w2_asw(int r, int c) { return (c ^ (r & 7)) * 16; }   // 16-byte chunk c of a 128-byte A row, XOR-swizzled
__device__ __forceinline__ void cp4(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ unsigned e2m1x2_bf16x2(uint8_t b, float s) {
  const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
  return pack_bf2(f.x * s, f.y * s);
}
template <bool GU, int BMT>
__global__ void __launch_bounds__(256, 1) w16v2_gemm_kernel(glm::ExpertRecLayout L, const uint8_t* const* __restrict__ tab, const PfTile* __restrict__ tiles,
                                                           const int* __restrict__ n_tiles, const int* __restrict__ perm, int K, const bf16* __restrict__ a_src,
                                                           int H, int I, bf16* __restrict__ out) {
  if ((int)blockIdx.x >= *n_tiles) return;
  extern __shared__ __align__(16) unsigned char smem[];
  constexpr int BN = W2_BN;
  uint8_t* As = smem;                                            // [ST][BMT][128]
  uint8_t* Bs = smem + (size_t)W2_ST * BMT * W2_AB;              // [ST][BN][32]
  uint8_t* Ss = Bs + (size_t)W2_ST * BN * W2_BB;                 // [ST][BN][4]
  constexpr int WM = BMT == 128 ? 2 : 1, WN = 8 / WM, WROWS = BMT / WM, MT = WROWS / 16, WCOLS = BN / WN, NT = WCOLS / 8;
  const PfTile tl = tiles[blockIdx.x];
  const uint8_t* rec = tab[tl.e];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp % WN;
  const int Kd = GU ? H : I;
  const int nb0 = blockIdx.y * (GU ? BN / 2 : BN);
  // A: BMT rows × 8 chunks of 16 B per stage; B: BN rows × 2 chunks of 16 B; scales: BN rows × 4 B
  constexpr int CA = (BMT * 8 + 255) / 256, CB = BN * 2 / 256, CS = BN / 256;
  const bf16* arow[CA];
#pragma unroll
  for (int i = 0; i < CA; ++i) {
    const int c = tid + 256 * i, r = c >> 3 < BMT ? c >> 3 : 0, rr = r < tl.m ? r : 0;
    arow[i] = GU ? a_src + (size_t)(perm[tl.slot0 + rr] / K) * H : a_src + (size_t)(tl.slot0 + rr) * I;
  }
  auto brow = [&](int br, const uint8_t*& w, const uint8_t*& s) {
    if (GU) {
      const int i = nb0 + (br >> 1);
      w = rec + ((br & 1) ? L.w3 : L.w1) + (size_t)i * (H / 2); s = rec + ((br & 1) ? L.s3 : L.s1) + (size_t)i * (H / 16);
    } else {
      w = rec + L.w2 + (size_t)(nb0 + br) * (I / 2); s = rec + L.s2 + (size_t)(nb0 + br) * (I / 16);
    }
  };
  const uint8_t* bsrc[CB];
  const uint8_t* ssrc[CS];
#pragma unroll
  for (int i = 0; i < CB; ++i) { const int c = tid + 256 * i; const uint8_t* s; brow(c >> 1, bsrc[i], s); bsrc[i] += (c & 1) * 16; }
#pragma unroll
  for (int i = 0; i < CS; ++i) { const uint8_t* w; brow(tid + 256 * i, w, ssrc[i]); }
  auto load = [&](int st, int kt) {
    if (kt >= 0) {
      uint8_t* Ab = As + (size_t)st * BMT * W2_AB;
#pragma unroll
      for (int i = 0; i < CA; ++i) {
        const int c = tid + 256 * i, r = c >> 3;
        if (c < BMT * 8) cp16(Ab + (size_t)r * W2_AB + w2_asw(r, c & 7), arow[i] + kt * W2_BK + (c & 7) * 8);
      }
      uint8_t* Bb = Bs + (size_t)st * BN * W2_BB;
#pragma unroll
      for (int i = 0; i < CB; ++i) { const int c = tid + 256 * i; cp16(Bb + (size_t)(c >> 1) * W2_BB + (c & 1) * 16, bsrc[i] + kt * (W2_BK / 2)); }
      uint8_t* Sb = Ss + (size_t)st * BN * 4;
#pragma unroll
      for (int i = 0; i < CS; ++i) cp4(Sb + (size_t)(tid + 256 * i) * 4, ssrc[i] + kt * (W2_BK / 16));
    }
    cp_commit();
  };
  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  const int nk = Kd / W2_BK;
#pragma unroll
  for (int s = 0; s < W2_ST - 1; ++s) load(s, s < nk ? s : -1);
  const int g = lane >> 2, t = lane & 3;
  const bool live = wm * WROWS < tl.m;
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<W2_ST - 2>();
    __syncthreads();
    load((kt + W2_ST - 1) % W2_ST, kt + W2_ST - 1 < nk ? kt + W2_ST - 1 : -1);
    if (!live) continue;
    const int st = kt % W2_ST;
    const uint8_t* Ab = As + (size_t)st * BMT * W2_AB;
    const uint8_t* Bb = Bs + (size_t)st * BN * W2_BB;
    const uint8_t* Sb = Ss + (size_t)st * BN * 4;
    unsigned sc[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) sc[nt] = *reinterpret_cast<const unsigned*>(Sb + (size_t)(wn * WCOLS + nt * 8 + g) * 4);
#pragma unroll
    for (int ks = 0; ks < W2_BK / 16; ++ks) {
      unsigned af[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r = wm * WROWS + mt * 16 + (lane & 15);
        ldm4(af[mt], Ab + (size_t)r * W2_AB + w2_asw(r, ks * 2 + (lane >> 4)));
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const uint2 q = *reinterpret_cast<const uint2*>(Bb + (size_t)(wn * WCOLS + nt * 8 + g) * W2_BB + ks * 8);
        const float s = e4m3f((uint8_t)(sc[nt] >> (8 * ks)));
        const unsigned b0 = e2m1x2_bf16x2((uint8_t)(q.x >> (8 * t)), s);   // k 2t, 2t + 1
        const unsigned b1 = e2m1x2_bf16x2((uint8_t)(q.y >> (8 * t)), s);   // k 2t + 8, 2t + 9
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma16816(acc[mt][nt], af[mt], b0, b1);
      }
    }
  }
  cp_wait0();
  if (!live) return;
  if (GU) {
    const float g1 = *reinterpret_cast<const float*>(rec + L.g), g3 = *reinterpret_cast<const float*>(rec + L.g + 4);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        const int r = wm * WROWS + mt * 16 + g + h2 * 8;
        if (r >= tl.m) continue;
        bf16* o = out + (size_t)(tl.slot0 + r) * I + nb0;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int j = (wn * WCOLS + nt * 8 + 2 * t) >> 1;
          const float gv = __bfloat162float(__float2bfloat16(acc[mt][nt][h2 * 2] * g1));
          const float uv = __bfloat162float(__float2bfloat16(acc[mt][nt][h2 * 2 + 1] * g3));
          o[j] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
        }
      }
  } else {
    const float g2 = *reinterpret_cast<const float*>(rec + L.g + 8);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        const int r = wm * WROWS + mt * 16 + g + h2 * 8;
        if (r >= tl.m) continue;
        bf16* o = out + (size_t)perm[tl.slot0 + r] * H + nb0;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int c = wn * WCOLS + nt * 8 + 2 * t;
          *reinterpret_cast<__nv_bfloat162*>(o + c) = __floats2bfloat162_rn(acc[mt][nt][h2 * 2] * g2, acc[mt][nt][h2 * 2 + 1] * g2);
        }
      }
  }
}

// ---- W4A4 (HIVE_QWEN_MOE_A4) ---------------------------------------------------------------------------------------------------------------
// The checkpoint's experts are calibrated for NVFP4 activations (config.json quantization_config group_0: input_activations num_bits 4,
//   group 16). Here the activations are quantized per row: global gx = amax / (6 · 448), one e4m3 scale per 16 values (amax_b / 6 / gx,
//   round to nearest), e2m1 values round-to-nearest of v / (scale · gx) — the checkpoint's format with a dynamic per-row global instead of
//   its static input_scale (nothing is clipped). The GEMMs run on mma.sync kind::mxf4nvf4 m16n8k64 (the layout validated by q27's
//   W4A4 decode GEMM, q27_kernels.cu): both operands' e4m3 block scales applied by the tensor core, fp32 accumulation; epilogues
//   × gx · g (the record's global). The rounding points after the GEMMs are the W4A16 kernels' (g, u → bf16, y = bf16(silu(g)·u),
//   part = bf16(·)). Lossy (activation quantization): a switch, default decided by the quality gate (docs/qwen.md).
constexpr int A4_BK = 128;     // k values per pipeline stage (64 bytes of a row)
constexpr int A4_RB = 80;      // smem row pitch (bytes): the 8 rows of an ldmatrix land on distinct banks
constexpr int A4_ST = 4;       // pipeline stages
constexpr int A4_BN = 128;     // B rows per block (gate|up: 64 gate + 64 up rows of I; down: 128 rows of H)
constexpr size_t a4_smem(int bm) {
  return std::max((size_t)A4_ST * (bm + A4_BN) * (A4_RB + 8), (size_t)bm * (A4_BN + 4) * 4 + (size_t)bm * 4);
}

// one warp per row: x [n, K] bf16 (row pitch ld) → q [n, K/2] (e2m1 pairs, low nibble = even column), s [n, K/16] (e4m3), g [n]
__global__ void a4_quant_kernel(const bf16* __restrict__ x, int ld, int n, int K, uint8_t* __restrict__ q, uint8_t* __restrict__ s,
                                float* __restrict__ g) {
  const int row = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
  if (row >= n) return;
  const bf16* xr = x + (size_t)row * ld;
  const int nb = K / 16;
  auto load16 = [&](int b, float (&v)[16]) {
    const uint4* p = reinterpret_cast<const uint4*>(xr + (size_t)b * 16);
    const uint4 u[2] = {p[0], p[1]};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const unsigned w[4] = {u[h].x, u[h].y, u[h].z, u[h].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[j]));
        v[h * 8 + j * 2] = f.x; v[h * 8 + j * 2 + 1] = f.y;
      }
    }
  };
  float a = 0.f;
  for (int b = lane; b < nb; b += 32) {
    float v[16];
    load16(b, v);
#pragma unroll
    for (int j = 0; j < 16; ++j) a = fmaxf(a, fabsf(v[j]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const float gx = a > 0.f ? a / (6.f * 448.f) : 1.f, inv = 1.f / gx;
  if (lane == 0) g[row] = gx;
  for (int b = lane; b < nb; b += 32) {
    float v[16], ab = 0.f;
    load16(b, v);
#pragma unroll
    for (int j = 0; j < 16; ++j) ab = fmaxf(ab, fabsf(v[j]));
    const __nv_fp8_storage_t se = __nv_cvt_float_to_fp8(ab / 6.f * inv, __NV_SATFINITE, __NV_E4M3);
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(se, __NV_E4M3);
    const float sv = __half2float(*reinterpret_cast<const __half*>(&hr));
    const float f = sv > 0.f ? inv / sv : 0.f;
    unsigned o[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j)
      o[j / 4] |= (unsigned)(uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(v[2 * j] * f, v[2 * j + 1] * f), __NV_E2M1, cudaRoundNearest) << (8 * (j % 4));
    *reinterpret_cast<uint2*>(q + (size_t)row * (K / 2) + (size_t)b * 8) = make_uint2(o[0], o[1]);
    s[(size_t)row * nb + b] = (uint8_t)se;
  }
}

__device__ __forceinline__ void cp8(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void mma_f4(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1, unsigned sa, unsigned sb) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%12, %13};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(0), "n"(0));
}

// GU = true: gate|up (A = the tile's token rows of xq, Kd = H; B = 64 gate + 64 up rows of I) → y bf16 [pair slot, I];
//   false: down (A = the tile's y rows, Kd = I; B = 128 rows of H) → part bf16 [pair, H]
template <bool GU, int BMT>
__global__ void __launch_bounds__(256) a4_gemm_kernel(glm::ExpertRecLayout L, const uint8_t* const* __restrict__ tab, const PfTile* __restrict__ tiles,
                                                     const int* __restrict__ n_tiles, const int* __restrict__ perm, int K, const uint8_t* __restrict__ aq,
                                                     const uint8_t* __restrict__ asc, const float* __restrict__ ag, int H, int I, bf16* __restrict__ out) {
  if ((int)blockIdx.x >= *n_tiles) return;
  extern __shared__ __align__(16) unsigned char smem[];
  constexpr int BN = A4_BN;
  uint8_t* Aq = smem;                                   // [ST][BMT][RB]
  uint8_t* Bq = Aq + A4_ST * BMT * A4_RB;               // [ST][BN][RB]
  uint8_t* As = Bq + A4_ST * BN * A4_RB;                // [ST][BMT][8]
  uint8_t* Bs = As + A4_ST * BMT * 8;                   // [ST][BN][8]
  constexpr int WM = BMT == 128 ? 2 : 1, WN = 8 / WM, WROWS = BMT / WM, MT = WROWS / 16, WCOLS = BN / WN, NT = WCOLS / 8, NP = NT / 2;
  static_assert(NT % 2 == 0, "a4 warp tile");
  const PfTile tl = tiles[blockIdx.x];
  const uint8_t* rec = tab[tl.e];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp % WN;
  const int Kd = GU ? H : I, kb = Kd / 2, kg = Kd / 16;
  const int nb0 = blockIdx.y * (GU ? BN / 2 : BN);
  // loads of a stage: A rows × 4 chunks of 16 B, B rows × 4 chunks, one 8-byte scale piece per A / B row
  constexpr int CA = (BMT * 4 + 255) / 256, CB = BN * 4 / 256;
  const uint8_t* arow[CA];
  const uint8_t* brow[CB];
#pragma unroll
  for (int i = 0; i < CA; ++i) {
    const int c = tid + i * 256, r = (c >> 2) < BMT ? (c >> 2) : 0, rr = r < tl.m ? r : 0;
    arow[i] = aq + (size_t)(GU ? perm[tl.slot0 + rr] / K : tl.slot0 + rr) * kb + (c & 3) * 16;
  }
#pragma unroll
  for (int i = 0; i < CB; ++i) {
    const int c = tid + i * 256, br = c >> 2;
    const size_t w = GU ? (br < BN / 2 ? L.w1 : L.w3) + (size_t)(nb0 + br % (BN / 2)) * kb : L.w2 + (size_t)(nb0 + br) * kb;
    brow[i] = rec + w + (c & 3) * 16;
  }
  const uint8_t* srow = nullptr;   // this thread's scale row (tid < BMT: A row tid; BMT ≤ tid < BMT + BN: B row tid − BMT)
  if (tid < BMT) {
    const int rr = tid < tl.m ? tid : 0;
    srow = asc + (size_t)(GU ? perm[tl.slot0 + rr] / K : tl.slot0 + rr) * kg;
  } else if (tid < BMT + BN) {
    const int br = tid - BMT;
    srow = rec + (GU ? (br < BN / 2 ? L.s1 : L.s3) + (size_t)(nb0 + br % (BN / 2)) * kg : L.s2 + (size_t)(nb0 + br) * kg);
  }
  auto load = [&](int st, int kt) {
    if (kt >= 0) {
#pragma unroll
      for (int i = 0; i < CA; ++i) {
        const int c = tid + i * 256;
        if (c < BMT * 4) cp16(Aq + ((size_t)st * BMT + (c >> 2)) * A4_RB + (c & 3) * 16, arow[i] + kt * 64);
      }
#pragma unroll
      for (int i = 0; i < CB; ++i) {
        const int c = tid + i * 256;
        cp16(Bq + ((size_t)st * BN + (c >> 2)) * A4_RB + (c & 3) * 16, brow[i] + kt * 64);
      }
      if (tid < BMT) cp8(As + ((size_t)st * BMT + tid) * 8, srow + kt * 8);
      else if (tid < BMT + BN) cp8(Bs + ((size_t)st * BN + tid - BMT) * 8, srow + kt * 8);
    }
    cp_commit();
  };
  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  const int nk = Kd / A4_BK;
#pragma unroll
  for (int s = 0; s < A4_ST - 1; ++s) load(s, s < nk ? s : -1);
  const int g = lane >> 2, t = lane & 3;
  const bool live = wm * WROWS < tl.m;   // this warp's rows hold pairs (warp-uniform)
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<A4_ST - 2>();
    __syncthreads();
    load((kt + A4_ST - 1) % A4_ST, kt + A4_ST - 1 < nk ? kt + A4_ST - 1 : -1);
    if (!live) continue;
    const int st = kt % A4_ST;
    const uint8_t* Ab = Aq + (size_t)st * BMT * A4_RB;
    const uint8_t* Bb = Bq + (size_t)st * BN * A4_RB;
    const uint8_t* Asb = As + (size_t)st * BMT * 8;
    const uint8_t* Bsb = Bs + (size_t)st * BN * 8;
#pragma unroll
    for (int ks = 0; ks < A4_BK / 64; ++ks) {
      unsigned af[MT][4], bfr[NP][4], sa[MT], sb[NT];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r0 = wm * WROWS + mt * 16;
        ldm4(af[mt], Ab + (size_t)(r0 + (lane & 7) + ((lane >> 3) & 1) * 8) * A4_RB + ks * 32 + (lane >> 4) * 16);
        sa[mt] = *reinterpret_cast<const unsigned*>(Asb + (size_t)(r0 + g + (t & 1) * 8) * 8 + ks * 4);
      }
#pragma unroll
      for (int np = 0; np < NP; ++np)
        ldm4(bfr[np], Bb + (size_t)(wn * WCOLS + np * 16 + ((lane >> 4) & 1) * 8 + (lane & 7)) * A4_RB + ks * 32 + ((lane >> 3) & 1) * 16);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) sb[nt] = *reinterpret_cast<const unsigned*>(Bsb + (size_t)(wn * WCOLS + nt * 8 + g) * 8 + ks * 4);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) mma_f4(acc[mt][nt], af[mt], bfr[nt >> 1][(nt & 1) * 2], bfr[nt >> 1][(nt & 1) * 2 + 1], sa[mt], sb[nt]);
    }
  }
  cp_wait0();
  __syncthreads();
  const int q2 = t * 2;
  float* rs = reinterpret_cast<float*>(smem + (size_t)BMT * (BN + 4) * 4);   // [BMT] the rows' activation globals
  if (tid < BMT) rs[tid] = tid < tl.m ? ag[GU ? perm[tl.slot0 + tid] / K : tl.slot0 + tid] : 0.f;
  if (GU) {
    float* Cs = reinterpret_cast<float*>(smem);   // [BMT][BN + 4]
    constexpr int CS = BN + 4, GN = BN / 2;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int r = wm * WROWS + mt * 16 + g, c = wn * WCOLS + nt * 8 + q2;
        Cs[r * CS + c] = acc[mt][nt][0]; Cs[r * CS + c + 1] = acc[mt][nt][1];
        Cs[(r + 8) * CS + c] = acc[mt][nt][2]; Cs[(r + 8) * CS + c + 1] = acc[mt][nt][3];
      }
    __syncthreads();
    const float g1 = *reinterpret_cast<const float*>(rec + L.g), g3 = *reinterpret_cast<const float*>(rec + L.g + 4);
    for (int idx = tid; idx < BMT * GN; idx += 256) {
      const int r = idx / GN, i = idx % GN;
      if (r >= tl.m) break;
      const float gr = rs[r];
      const float gv = __bfloat162float(__float2bfloat16(Cs[r * CS + i] * gr * g1));
      const float uv = __bfloat162float(__float2bfloat16(Cs[r * CS + GN + i] * gr * g3));
      out[(size_t)(tl.slot0 + r) * I + nb0 + i] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
    }
  } else {
    __syncthreads();
    const float g2 = *reinterpret_cast<const float*>(rec + L.g + 8);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        const int r = wm * WROWS + mt * 16 + g + h2 * 8;
        if (r >= tl.m) continue;
        const float f = rs[r] * g2;
        bf16* o = out + (size_t)perm[tl.slot0 + r] * H + nb0;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int c = wn * WCOLS + nt * 8 + q2;
          *reinterpret_cast<__nv_bfloat162*>(o + c) = __floats2bfloat162_rn(acc[mt][nt][h2 * 2] * f, acc[mt][nt][h2 * 2 + 1] * f);
        }
      }
  }
}

// ---- W4A4 v2: 64 × 64 warp tiles, gate / up rows interleaved, swizzled 64-byte smem rows ---------------------------------------------------
// ncu 2026-10-30 (a4_gemm_kernel<GU, 128>, 8,192 rows): stalls mostly mio_throttle / wait / barrier — 14 shared-memory instructions per 16 mma
//   (64 × 32 warp tiles, a 32-bit scale load per m and per n tile and k step). Here: 256 B rows per block (gate|up: 128 gate rows interleaved
//   with their 128 up rows, so a thread's accumulator pair (2t, 2t + 1) is the gate and the up value of one output column and the epilogue
//   stays in registers; down: 256 rows of H), 64 × 64 warp tiles (28 shared-memory instructions per 64 mma: the scales of both k64 steps of a
//   stage in one 64-bit load), 3 stages of 64-byte rows with the 16-byte chunks XOR-swizzled instead of padded. Same products, same k order
//   and the same epilogue arithmetic as a4_gemm_kernel: bit-identical outputs.
constexpr int A2_BN = 256, A2_ST = 3, A2_RB = 64;
constexpr size_t a2_smem(int bm) { return (size_t)A2_ST * (bm + A2_BN) * (A2_RB + 8); }
__device__ __forceinline__ int a2_sw(int r, int c) { return (c ^ ((r >> 1) & 3)) * 16; }   // byte offset of 16-byte chunk c in row r
template <bool GU, int BMT>
__global__ void __launch_bounds__(256, 1) a4v2_gemm_kernel(glm::ExpertRecLayout L, const uint8_t* const* __restrict__ tab, const PfTile* __restrict__ tiles,
                                                          const int* __restrict__ n_tiles, const int* __restrict__ perm, int K, const uint8_t* __restrict__ aq,
                                                          const uint8_t* __restrict__ asc, const float* __restrict__ ag, int H, int I, bf16* __restrict__ out) {
  if ((int)blockIdx.x >= *n_tiles) return;
  extern __shared__ __align__(16) unsigned char smem[];
  constexpr int BN = A2_BN, NR = BMT + BN;
  uint8_t* Q = smem;                              // [ST][NR][64]: rows 0..BMT−1 = A, then the B rows
  uint8_t* Sc = smem + (size_t)A2_ST * NR * A2_RB;   // [ST][NR][8]
  constexpr int WM = BMT == 128 ? 2 : 1, WN = 8 / WM, WROWS = BMT / WM, MT = WROWS / 16, WCOLS = BN / WN, NT = WCOLS / 8, NP = NT / 2;
  const PfTile tl = tiles[blockIdx.x];
  const uint8_t* rec = tab[tl.e];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp % WN;
  const int Kd = GU ? H : I, kb = Kd / 2, kg = Kd / 16;
  const int nb0 = blockIdx.y * (GU ? BN / 2 : BN);   // first output column (gate|up: of I; down: of H)
  // source of smem row r (A row, or B row r − BMT)
  auto row_src = [&](int r, const uint8_t*& d, const uint8_t*& s) {
    if (r < BMT) {
      const int rr = r < tl.m ? r : 0;
      const int src = GU ? perm[tl.slot0 + rr] / K : tl.slot0 + rr;
      d = aq + (size_t)src * kb; s = asc + (size_t)src * kg;
    } else {
      const int br = r - BMT;
      if (GU) {   // interleaved: even = gate row, odd = up row of output column nb0 + br / 2
        const int i = nb0 + (br >> 1);
        d = rec + ((br & 1) ? L.w3 : L.w1) + (size_t)i * kb; s = rec + ((br & 1) ? L.s3 : L.s1) + (size_t)i * kg;
      } else {
        d = rec + L.w2 + (size_t)(nb0 + br) * kb; s = rec + L.s2 + (size_t)(nb0 + br) * kg;
      }
    }
  };
  // this thread's loads per stage: data chunks c = tid + 256·i (row c / 4, chunk c % 4), one 8-byte scale piece per row tid (+256)
  constexpr int CD = (NR * 4 + 255) / 256, CS = (NR + 255) / 256;
  const uint8_t* dsrc[CD];
  const uint8_t* ssrc[CS];
#pragma unroll
  for (int i = 0; i < CD; ++i) {
    const int c = tid + 256 * i;
    if (c < NR * 4) { const uint8_t* s; row_src(c >> 2, dsrc[i], s); dsrc[i] += (c & 3) * 16; } else dsrc[i] = nullptr;
  }
#pragma unroll
  for (int i = 0; i < CS; ++i) { const int r = tid + 256 * i; const uint8_t* d; if (r < NR) row_src(r, d, ssrc[i]); else ssrc[i] = nullptr; }
  auto load = [&](int st, int kt) {
    if (kt >= 0) {
      uint8_t* Qs = Q + (size_t)st * NR * A2_RB;
#pragma unroll
      for (int i = 0; i < CD; ++i) { const int c = tid + 256 * i, r = c >> 2; if (c < NR * 4) cp16(Qs + (size_t)r * A2_RB + a2_sw(r, c & 3), dsrc[i] + kt * 64); }
#pragma unroll
      for (int i = 0; i < CS; ++i) { const int r = tid + 256 * i; if (r < NR) cp8(Sc + ((size_t)st * NR + r) * 8, ssrc[i] + kt * 8); }
    }
    cp_commit();
  };
  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  const int nk = Kd / A4_BK;
#pragma unroll
  for (int s = 0; s < A2_ST - 1; ++s) load(s, s < nk ? s : -1);
  const int g = lane >> 2, t = lane & 3;
  const bool live = wm * WROWS < tl.m;
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<A2_ST - 2>();
    __syncthreads();
    load((kt + A2_ST - 1) % A2_ST, kt + A2_ST - 1 < nk ? kt + A2_ST - 1 : -1);
    if (!live) continue;
    const int st = kt % A2_ST;
    const uint8_t* Qs = Q + (size_t)st * NR * A2_RB;
    const uint8_t* Ss = Sc + (size_t)st * NR * 8;
    uint2 sa[MT], sb[NT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) sa[mt] = *reinterpret_cast<const uint2*>(Ss + (size_t)(wm * WROWS + mt * 16 + g + (t & 1) * 8) * 8);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) sb[nt] = *reinterpret_cast<const uint2*>(Ss + (size_t)(BMT + wn * WCOLS + nt * 8 + g) * 8);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      unsigned af[MT][4], bfr[NP][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r = wm * WROWS + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldm4(af[mt], Qs + (size_t)r * A2_RB + a2_sw(r, ks * 2 + (lane >> 4)));
      }
#pragma unroll
      for (int np = 0; np < NP; ++np) {
        const int r = BMT + wn * WCOLS + np * 16 + ((lane >> 4) & 1) * 8 + (lane & 7);
        ldm4(bfr[np], Qs + (size_t)r * A2_RB + a2_sw(r, ks * 2 + ((lane >> 3) & 1)));
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
          mma_f4(acc[mt][nt], af[mt], bfr[nt >> 1][(nt & 1) * 2], bfr[nt >> 1][(nt & 1) * 2 + 1], ks ? sa[mt].y : sa[mt].x, ks ? sb[nt].y : sb[nt].x);
    }
  }
  cp_wait0();
  if (!live) return;
  if (GU) {
    const float g1 = *reinterpret_cast<const float*>(rec + L.g), g3 = *reinterpret_cast<const float*>(rec + L.g + 4);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        const int r = wm * WROWS + mt * 16 + g + h2 * 8;
        if (r >= tl.m) continue;
        const float gr = ag[perm[tl.slot0 + r] / K];
        bf16* o = out + (size_t)(tl.slot0 + r) * I + nb0;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int j = (wn * WCOLS + nt * 8 + 2 * t) >> 1;   // output column: gate in c[2·h2], up in c[2·h2 + 1]
          const float gv = __bfloat162float(__float2bfloat16(acc[mt][nt][h2 * 2] * gr * g1));
          const float uv = __bfloat162float(__float2bfloat16(acc[mt][nt][h2 * 2 + 1] * gr * g3));
          o[j] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
        }
      }
  } else {
    const float g2 = *reinterpret_cast<const float*>(rec + L.g + 8);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        const int r = wm * WROWS + mt * 16 + g + h2 * 8;
        if (r >= tl.m) continue;
        const float f = ag[tl.slot0 + r] * g2;
        bf16* o = out + (size_t)perm[tl.slot0 + r] * H + nb0;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int c = wn * WCOLS + nt * 8 + 2 * t;
          *reinterpret_cast<__nv_bfloat162*>(o + c) = __floats2bfloat162_rn(acc[mt][nt][h2 * 2] * f, acc[mt][nt][h2 * 2 + 1] * f);
        }
      }
  }
}

// the reduce fused with the MoE combine (moe_combine_kernel's arithmetic): out[row] = bf16(bf16(Σ_k part·w) + bf16(bf16(sigmoid(bf16(gate
//   logit))) · shared)) — the fp32 sum is the one pf_reduce_kernel stores, so the result is bit-identical to reduce → moe_combine
__global__ void pf_reduce_combine_kernel(const bf16* __restrict__ part, const int32_t* __restrict__ ids, const float* __restrict__ w, int M, int K, int E,
                                         int H, const bf16* __restrict__ shared, const float* __restrict__ lg, int ld, int col, bf16* __restrict__ out) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (i >= (size_t)M * H) return;
  const int m = (int)(i / H), n = (int)(i % H);
  float a0 = 0.f, a1 = 0.f;
  for (int k = 0; k < K; ++k) {
    const int p = m * K + k, e = ids[p];
    if (e < 0 || e >= E) continue;
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(part + (size_t)p * H + n));
    const float wv = w[p];
    a0 += v.x * wv; a1 += v.y * wv;
  }
  auto rb = [](float x) { return __bfloat162float(__float2bfloat16(x)); };
  const float g = rb(1.f / (1.f + expf(-rb(lg[(size_t)m * ld + col]))));
  const float2 sh = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(shared + i));
  *reinterpret_cast<__nv_bfloat162*>(out + i) = __halves2bfloat162(__float2bfloat16(rb(a0) + rb(g * sh.x)), __float2bfloat16(rb(a1) + rb(g * sh.y)));
}

int g_pf_gn = getenv("HIVE_QWEN_MOE_PF_GN") && atoi(getenv("HIVE_QWEN_MOE_PF_GN")) == 128 ? 128 : 64;   // short-prompt gate|up block width
int g_pf_dn = getenv("HIVE_QWEN_MOE_PF_DN") && atoi(getenv("HIVE_QWEN_MOE_PF_DN")) == 256 ? 256 : 128;   // short-prompt down block rows
struct PfWs { int* cnt; int* cursor; int* n_tiles; PfTile* tiles; int* perm; bf16* y; bf16* part; uint8_t *xq, *xs, *yq, *ys; float *xg, *yg; };
PfWs pf_layout(void* ws, int S, int K, int E, int H, int I) {
  char* p = static_cast<char*>(ws);
  auto take = [&](size_t b) { char* r = p; p += (b + 255) / 256 * 256; return r; };
  const size_t P = (size_t)S * K;
  PfWs w;
  w.cnt = (int*)take((size_t)E * 4); w.cursor = (int*)take((size_t)E * 4); w.n_tiles = (int*)take(4);
  w.tiles = (PfTile*)take((P / BMS + E) * sizeof(PfTile)); w.perm = (int*)take(P * 4);
  w.y = (bf16*)take(P * I * 2); w.part = (bf16*)take(P * H * 2);
  w.xq = (uint8_t*)take((size_t)S * H / 2); w.xs = (uint8_t*)take((size_t)S * H / 16); w.xg = (float*)take((size_t)S * 4);   // W4A4
  w.yq = (uint8_t*)take(P * I / 2); w.ys = (uint8_t*)take(P * I / 16); w.yg = (float*)take(P * 4);
  return w;
}
}  // namespace

bool g_pf_a4 = getenv("HIVE_QWEN_MOE_A4") && atoi(getenv("HIVE_QWEN_MOE_A4")) > 0;
bool g_pf_v2 = !(getenv("HIVE_QWEN_MOE_PF_V2") && atoi(getenv("HIVE_QWEN_MOE_PF_V2")) == 0);   // W4A16 v2 kernels (0 = the first ones, A/B)
void fn_moe_prefill_set_v2(bool on) { g_pf_v2 = on; }
bool g_pf_a4v2 = !(getenv("HIVE_QWEN_MOE_A4V2") && atoi(getenv("HIVE_QWEN_MOE_A4V2")) == 0);   // 0 = the v1 kernels (A/B)
void fn_moe_prefill_set_a4(bool on) { g_pf_a4 = on; }
void fn_moe_prefill_set_a4v2(bool on) { g_pf_a4v2 = on; }
bool fn_moe_prefill_a4() { return g_pf_a4; }
void fn_moe_prefill_set_gn(int gn, int dn) { g_pf_gn = gn == 128 ? 128 : 64; g_pf_dn = dn == 256 ? 256 : 128; }
bool fn_moe_prefill_supported(int H, int I) { return H % kDnN == 0 && I % kGuN == 0 && H % BK == 0 && I % BK == 0; }

size_t fn_moe_prefill_ws_bytes(int S, int K, int E, int H, int I) {
  const size_t P = (size_t)S * K;
  auto r = [](size_t b) { return (b + 255) / 256 * 256; };
  return r((size_t)E * 4) * 2 + r(4) + r((P / BMS + E) * sizeof(PfTile)) + r(P * 4) + r(P * I * 2) + r(P * H * 2) +
         r((size_t)S * H / 2) + r((size_t)S * H / 16) + r((size_t)S * 4) + r(P * I / 2) + r(P * I / 16) + r(P * 4);
}

void fn_moe_prefill(const glm::ExpertRecLayout& L, const uint8_t* const* tab, int li, int E, const int32_t* ids, const float* w, int M, int K,
                    const __nv_bfloat16* x, int H, int I, float* out, void* ws, int S, cudaStream_t st, const FnPfCombine* cmb) {
  if (M <= 0) return;
  HIVE_CHECK(fn_moe_prefill_supported(H, I) && S >= 1 && E <= 1024, "fn_moe_prefill shape");
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(pf_gemm_kernel<true, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_for(BM)));
    CUDA_CHECK(cudaFuncSetAttribute(pf_gemm_kernel<false, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_for(BM)));
    CUDA_CHECK(cudaFuncSetAttribute(pf_gemm_kernel<true, BMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_for(BMS)));
    CUDA_CHECK(cudaFuncSetAttribute(pf_gemm_kernel<false, BMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_for(BMS)));
    CUDA_CHECK(cudaFuncSetAttribute(pf_gemm_kernel<true, BMS, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_for(BMS, 256)));
    CUDA_CHECK(cudaFuncSetAttribute(pf_gemm_kernel<false, BMS, kGuN, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_for(BMS, 256)));
    CUDA_CHECK(cudaFuncSetAttribute(w16v2_gemm_kernel<true, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)w2_smem(BM)));
    CUDA_CHECK(cudaFuncSetAttribute(w16v2_gemm_kernel<false, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)w2_smem(BM)));
    CUDA_CHECK(cudaFuncSetAttribute(a4v2_gemm_kernel<true, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a2_smem(BM)));
    CUDA_CHECK(cudaFuncSetAttribute(a4v2_gemm_kernel<false, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a2_smem(BM)));
    CUDA_CHECK(cudaFuncSetAttribute(a4v2_gemm_kernel<true, BMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a2_smem(BMS)));
    CUDA_CHECK(cudaFuncSetAttribute(a4v2_gemm_kernel<false, BMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a2_smem(BMS)));
    CUDA_CHECK(cudaFuncSetAttribute(a4_gemm_kernel<true, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a4_smem(BM)));
    CUDA_CHECK(cudaFuncSetAttribute(a4_gemm_kernel<false, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a4_smem(BM)));
    CUDA_CHECK(cudaFuncSetAttribute(a4_gemm_kernel<true, BMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a4_smem(BMS)));
    CUDA_CHECK(cudaFuncSetAttribute(a4_gemm_kernel<false, BMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a4_smem(BMS)));
    attr = true;
  }
  const PfWs W = pf_layout(ws, S, K, E, H, I);
  const uint8_t* const* lt = tab + (size_t)li * E;
  for (int s0 = 0; s0 < M; s0 += S) {
    const int n = std::min(S, M - s0), P = n * K;
    const int32_t* sid = ids + (size_t)s0 * K;
    CUDA_CHECK(cudaMemsetAsync(W.cnt, 0, (size_t)E * 4, st));
    pf_count_kernel<<<(P + 255) / 256, 256, 0, st>>>(sid, P, E, W.cnt);
    // tile rows: 128 when experts average ≥ 96 pairs, else 32 (short prompts: ~20 pairs per expert at 1K tokens — the 128-row tiles were
    //   mostly empty and their smem allowed one block per SM; nsys 2026-10-10: 97 ms of a 180 ms 1K-token prefill)
    const int bm = P >= 96 * E ? BM : BMS;
    pf_scan_kernel<<<1, 1024, 0, st>>>(W.cnt, E, bm, W.cursor, W.tiles, W.n_tiles);
    pf_fill_kernel<<<(P + 255) / 256, 256, 0, st>>>(sid, P, E, W.cursor, W.perm);
    const int max_tiles = P / bm + E;
    if (g_pf_a4 && g_pf_a4v2) {   // W4A4, v2 kernels (bit-identical to v1)
      a4_quant_kernel<<<(n + 7) / 8, 256, 0, st>>>(x + (size_t)s0 * H, H, n, H, W.xq, W.xs, W.xg);
      if (bm == BM) a4v2_gemm_kernel<true, BM><<<dim3(max_tiles, I / (A2_BN / 2)), 256, a2_smem(BM), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.xq, W.xs, W.xg, H, I, W.y);
      else a4v2_gemm_kernel<true, BMS><<<dim3(max_tiles, I / (A2_BN / 2)), 256, a2_smem(BMS), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.xq, W.xs, W.xg, H, I, W.y);
      a4_quant_kernel<<<(P + 7) / 8, 256, 0, st>>>(W.y, I, P, I, W.yq, W.ys, W.yg);
      if (bm == BM) a4v2_gemm_kernel<false, BM><<<dim3(max_tiles, H / A2_BN), 256, a2_smem(BM), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.yq, W.ys, W.yg, H, I, W.part);
      else a4v2_gemm_kernel<false, BMS><<<dim3(max_tiles, H / A2_BN), 256, a2_smem(BMS), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.yq, W.ys, W.yg, H, I, W.part);
    } else if (g_pf_a4) {   // W4A4: x rows quantized once, y rows after the gate|up pass
      a4_quant_kernel<<<(n + 7) / 8, 256, 0, st>>>(x + (size_t)s0 * H, H, n, H, W.xq, W.xs, W.xg);
      if (bm == BM) a4_gemm_kernel<true, BM><<<dim3(max_tiles, I / (A4_BN / 2)), 256, a4_smem(BM), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.xq, W.xs, W.xg, H, I, W.y);
      else a4_gemm_kernel<true, BMS><<<dim3(max_tiles, I / (A4_BN / 2)), 256, a4_smem(BMS), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.xq, W.xs, W.xg, H, I, W.y);
      a4_quant_kernel<<<(P + 7) / 8, 256, 0, st>>>(W.y, I, P, I, W.yq, W.ys, W.yg);
      if (bm == BM) a4_gemm_kernel<false, BM><<<dim3(max_tiles, H / A4_BN), 256, a4_smem(BM), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.yq, W.ys, W.yg, H, I, W.part);
      else a4_gemm_kernel<false, BMS><<<dim3(max_tiles, H / A4_BN), 256, a4_smem(BMS), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.yq, W.ys, W.yg, H, I, W.part);
    } else if (g_pf_v2 && bm == BM) {   // W4A16 v2 for the 128-row tiles (bit-identical to the kernels below; measured 8,192 rows 5.59 → 4.12 ms
      //   per layer — the 32-row tiles of short prompts are weight-bandwidth bound and stayed faster on the first kernels: 1,024 rows 2.32 vs 2.45 ms)
      w16v2_gemm_kernel<true, BM><<<dim3(max_tiles, I / (W2_BN / 2)), 256, w2_smem(BM), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, x + (size_t)s0 * H, H, I, W.y);
      w16v2_gemm_kernel<false, BM><<<dim3(max_tiles, H / W2_BN), 256, w2_smem(BM), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.y, H, I, W.part);
    } else if (bm == BM) {
      pf_gemm_kernel<true, BM><<<dim3(max_tiles, I / kGuN), kThreads, smem_for(BM), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, x + (size_t)s0 * H, H, I, W.y);
      pf_gemm_kernel<false, BM><<<dim3(max_tiles, H / kDnN), kThreads, smem_for(BM), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.y, H, I, W.part);
    } else {
      if (g_pf_gn == 128 && I % 128 == 0)
        pf_gemm_kernel<true, BMS, 128><<<dim3(max_tiles, I / 128), kThreads, smem_for(BMS, 256), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, x + (size_t)s0 * H, H, I, W.y);
      else
        pf_gemm_kernel<true, BMS><<<dim3(max_tiles, I / kGuN), kThreads, smem_for(BMS), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, x + (size_t)s0 * H, H, I, W.y);
      if (g_pf_dn == 256 && H % 256 == 0)
        pf_gemm_kernel<false, BMS, kGuN, 256><<<dim3(max_tiles, H / 256), kThreads, smem_for(BMS, 256), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.y, H, I, W.part);
      else
        pf_gemm_kernel<false, BMS><<<dim3(max_tiles, H / kDnN), kThreads, smem_for(BMS), st>>>(L, lt, W.tiles, W.n_tiles, W.perm, K, W.y, H, I, W.part);
    }
    const size_t pairs2 = (size_t)n * H / 2;
    if (cmb)
      pf_reduce_combine_kernel<<<(unsigned)((pairs2 + 255) / 256), 256, 0, st>>>(W.part, sid, w + (size_t)s0 * K, n, K, E, H, cmb->shared + (size_t)s0 * H,
                                                                                 cmb->lg + (size_t)s0 * cmb->ld, cmb->ld, cmb->col, cmb->out + (size_t)s0 * H);
    else
      pf_reduce_kernel<<<(unsigned)((pairs2 + 255) / 256), 256, 0, st>>>(W.part, sid, w + (size_t)s0 * K, n, K, E, H, out + (size_t)s0 * H);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen
