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

int g_pf_gn = getenv("HIVE_QWEN_MOE_PF_GN") && atoi(getenv("HIVE_QWEN_MOE_PF_GN")) == 128 ? 128 : 64;   // short-prompt gate|up block width
int g_pf_dn = getenv("HIVE_QWEN_MOE_PF_DN") && atoi(getenv("HIVE_QWEN_MOE_PF_DN")) == 256 ? 256 : 128;   // short-prompt down block rows
struct PfWs { int* cnt; int* cursor; int* n_tiles; PfTile* tiles; int* perm; bf16* y; bf16* part; };
PfWs pf_layout(void* ws, int S, int K, int E, int H, int I) {
  char* p = static_cast<char*>(ws);
  auto take = [&](size_t b) { char* r = p; p += (b + 255) / 256 * 256; return r; };
  const size_t P = (size_t)S * K;
  PfWs w;
  w.cnt = (int*)take((size_t)E * 4); w.cursor = (int*)take((size_t)E * 4); w.n_tiles = (int*)take(4);
  w.tiles = (PfTile*)take((P / BMS + E) * sizeof(PfTile)); w.perm = (int*)take(P * 4);
  w.y = (bf16*)take(P * I * 2); w.part = (bf16*)take(P * H * 2);
  return w;
}
}  // namespace

void fn_moe_prefill_set_gn(int gn, int dn) { g_pf_gn = gn == 128 ? 128 : 64; g_pf_dn = dn == 256 ? 256 : 128; }
bool fn_moe_prefill_supported(int H, int I) { return H % kDnN == 0 && I % kGuN == 0 && H % BK == 0 && I % BK == 0; }

size_t fn_moe_prefill_ws_bytes(int S, int K, int E, int H, int I) {
  const size_t P = (size_t)S * K;
  auto r = [](size_t b) { return (b + 255) / 256 * 256; };
  return r((size_t)E * 4) * 2 + r(4) + r((P / BMS + E) * sizeof(PfTile)) + r(P * 4) + r(P * I * 2) + r(P * H * 2);
}

void fn_moe_prefill(const glm::ExpertRecLayout& L, const uint8_t* const* tab, int li, int E, const int32_t* ids, const float* w, int M, int K,
                    const __nv_bfloat16* x, int H, int I, float* out, void* ws, int S, cudaStream_t st) {
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
    if (bm == BM) {
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
    pf_reduce_kernel<<<(unsigned)((pairs2 + 255) / 256), 256, 0, st>>>(W.part, sid, w + (size_t)s0 * K, n, K, E, H, out + (size_t)s0 * H);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen
