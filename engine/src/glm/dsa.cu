// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash DSA kernels — see hive/glm/dsa.h for the math and the reference correspondence.
//
// Pipeline per layer and step:
//   dsa_append    : copy c / kI / gs rows into the cache, then build pooled keys of pools completed by the new tokens.
//   dsa_select    : (only for queries with > 512 visible pools) fused tensor-core scorer → fp32 scores [q, pool], then a
//                   per-query block radix-select of the top 512 pools (ties at the threshold: lowest pool id), expanded to
//                   ascending token ids + the incomplete tail pool. Queries with ≤ 512 visible pools attend densely.
//   dsa_attention : q_lat = W_UKᵀ q (cuBLAS, fp32 out → fp16) → sparse gather / online-softmax core on fp16
//                   tensor cores (fp32 accumulation, 16 heads × 32 tokens per step, Q in registers, cp.async gather of
//                   latent rows) with optional split over tokens + combine → out = W_UV (out_lat hi + lo) (cuBLAS, fp32).
#include "hive/glm/dsa.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cuda_fp16.h>

#include <cub/cub.cuh>

namespace hive::glm {

namespace {

#define DSA_CUBLAS_CHECK(expr)                                                              \
  do {                                                                                      \
    cublasStatus_t _s = (expr);                                                             \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                                      \
      fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n", (int)_s, __FILE__, __LINE__, #expr); \
      abort();                                                                              \
    }                                                                                       \
  } while (0)

cublasHandle_t g_user_handle = nullptr;
cublasHandle_t blas(cudaStream_t st) {
  static thread_local cublasHandle_t h = nullptr;
  cublasHandle_t use = g_user_handle;
  if (!use) {
    if (!h) DSA_CUBLAS_CHECK(cublasCreate(&h));
    use = h;
  }
  DSA_CUBLAS_CHECK(cublasSetStream(use, st));
  return use;
}

// Per-query sources: either one sequence (prefill: positions pos0 + i) or per-row decode descriptors.
struct QSrc {
  const DsaDecodeRow* rows;
  const bf16* c;
  const bf16* pooled;
  int pos0;
  __device__ __forceinline__ int pos(int i) const { return rows ? rows[i].pos : pos0 + i; }
  __device__ __forceinline__ const bf16* cp(int i) const { return rows ? rows[i].c : c; }
  __device__ __forceinline__ const bf16* pl(int i) const { return rows ? rows[i].pooled : pooled; }
};

__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t ld32(const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); }

// =====================================================================================================================
// append
// =====================================================================================================================
// Copy row i of c/kI/gs to the cache at its position (one block per new token).
__global__ void append_rows_kernel(QSrc dst, bf16* c, bf16* kI, bf16* gs, const bf16* __restrict__ c_new,
                                   const bf16* __restrict__ kI_new, const bf16* __restrict__ gs_new,
                                   const DsaDecodeRow* rows) {
  const int i = blockIdx.x, t = threadIdx.x;  // 64 threads × 16 B = 1024 B (one latent row)
  const int p = dst.pos(i);
  bf16* cc = rows ? rows[i].c : c;
  bf16* kk = rows ? rows[i].kI : kI;
  bf16* gg = rows ? rows[i].gs : gs;
  reinterpret_cast<uint4*>(cc + (size_t)p * kDsaLat)[t] = reinterpret_cast<const uint4*>(c_new + (size_t)i * kDsaLat)[t];
  if (t < 16)
    reinterpret_cast<uint4*>(kk + (size_t)p * kDsaIdxDim)[t] = reinterpret_cast<const uint4*>(kI_new + (size_t)i * kDsaIdxDim)[t];
  else if (t < 32)
    reinterpret_cast<uint4*>(gg + (size_t)p * kDsaIdxDim)[t - 16] =
        reinterpret_cast<const uint4*>(gs_new + (size_t)i * kDsaIdxDim)[t - 16];
}

// Pooled key of pool p (one block of 128 channels per pool), mirroring get_pooled_states in the reference:
//   logits = gs.float() + ape.float(); prob = softmax over the 4 slots (fp32) → bf16; K = Σ_j bf16(prob_j · k_j) (fp32 sum
//   of the bf16 products, rounded once to bf16).
__device__ __forceinline__ void pool_one(const bf16* kI, const bf16* gs, const bf16* ape, bf16* pooled, int p, int ch) {
  float g[kDsaPool], m = -INFINITY;
#pragma unroll
  for (int j = 0; j < kDsaPool; ++j) {
    g[j] = bf(gs[(size_t)(kDsaPool * p + j) * kDsaIdxDim + ch]) + bf(ape[j * kDsaIdxDim + ch]);
    m = fmaxf(m, g[j]);
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < kDsaPool; ++j) {
    g[j] = expf(g[j] - m);
    s += g[j];
  }
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < kDsaPool; ++j) {
    const float pr = bf(__float2bfloat16(g[j] / s));
    acc += bf(__float2bfloat16(pr * bf(kI[(size_t)(kDsaPool * p + j) * kDsaIdxDim + ch])));
  }
  pooled[(size_t)p * kDsaIdxDim + ch] = __float2bfloat16(acc);
}

__global__ void pool_prefill_kernel(const bf16* kI, const bf16* gs, const bf16* __restrict__ ape, bf16* pooled, int p0) {
  pool_one(kI, gs, ape, pooled, p0 + blockIdx.x, threadIdx.x);
}
__global__ void pool_decode_kernel(const DsaDecodeRow* rows, const bf16* __restrict__ ape) {
  const DsaDecodeRow r = rows[blockIdx.x];
  if ((r.pos + 1) % kDsaPool != 0) return;  // this token does not complete its pool
  pool_one(r.kI, r.gs, ape, r.pooled, r.pos / kDsaPool, threadIdx.x);
}

// =====================================================================================================================
// indexer scores: score[q, p] = Σ_h (w_h·32^-0.5) · relu(qI_h · K_p · 128^-0.5)
// Block = QPB queries × 64 pools; 4 warps per query; warp = 32 heads (2 m16 tiles) × 16 pools (2 n8 tiles), K = 128.
// =====================================================================================================================
constexpr int kScPools = 64, kScLd = kDsaIdxDim + 8;  // padded bf16 row (conflict-free 32-bit fragment loads)

template <int QPB>
__global__ void __launch_bounds__(QPB * 128) score_kernel(QSrc src, int q0, int nq, const bf16* __restrict__ qI,
                                                          const float* __restrict__ wI, float* __restrict__ scores, int ld) {
  __shared__ __align__(16) bf16 As[QPB][kDsaIdxHeads][kScLd];
  __shared__ __align__(16) bf16 Bs[kScPools][kScLd];
  __shared__ float Ws[QPB][kDsaIdxHeads];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int qb = q0 + blockIdx.y * QPB;  // absolute index of the block's first query
  const int p0 = blockIdx.x * kScPools;
  // visible pools of the last (largest) query in the block; queries with ≤ 512 pools are not scored (dense)
  int nmax = 0;
  for (int j = 0; j < QPB; ++j)
    if (qb + j < q0 + nq) nmax = max(nmax, (src.pos(qb + j) + 1) / kDsaPool);
  if (p0 >= nmax || nmax <= kDsaTopPools) return;
  const bf16* pooled = src.pl(qb);  // QPB > 1 only for single-sequence prefill (shared pooled keys)
  for (int e = tid; e < QPB * kDsaIdxHeads * (kDsaIdxDim / 8); e += blockDim.x) {
    const int j = e / (kDsaIdxHeads * 16), r = (e / 16) % kDsaIdxHeads, c8 = e % 16;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (qb + j < q0 + nq) v = reinterpret_cast<const uint4*>(qI + ((size_t)(qb + j) * kDsaIdxHeads + r) * kDsaIdxDim)[c8];
    *reinterpret_cast<uint4*>(&As[j][r][c8 * 8]) = v;
  }
  for (int e = tid; e < kScPools * 16; e += blockDim.x) {
    const int r = e / 16, c8 = e % 16;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (p0 + r < nmax) v = reinterpret_cast<const uint4*>(pooled + (size_t)(p0 + r) * kDsaIdxDim)[c8];
    *reinterpret_cast<uint4*>(&Bs[r][c8 * 8]) = v;
  }
  for (int e = tid; e < QPB * kDsaIdxHeads; e += blockDim.x) {
    const int j = e / kDsaIdxHeads;
    Ws[j][e % kDsaIdxHeads] = (qb + j < q0 + nq) ? wI[(size_t)(qb + j) * kDsaIdxHeads + e % kDsaIdxHeads] : 0.f;
  }
  __syncthreads();
  const int j = warp >> 2, wq = warp & 3;  // query in block, pool sub-tile
  float acc[2][2][4] = {};
#pragma unroll
  for (int k0 = 0; k0 < kDsaIdxDim; k0 += 16) {
    uint32_t a[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int r = mt * 16 + (lane >> 2), c = k0 + (lane & 3) * 2;
      a[mt][0] = ld32(&As[j][r][c]);
      a[mt][1] = ld32(&As[j][r + 8][c]);
      a[mt][2] = ld32(&As[j][r][c + 8]);
      a[mt][3] = ld32(&As[j][r + 8][c + 8]);
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int n = wq * 16 + nt * 8 + (lane >> 2), c = k0 + (lane & 3) * 2;
      const uint32_t b0 = ld32(&Bs[n][c]), b1 = ld32(&Bs[n][c + 8]);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) mma_bf16(acc[mt][nt], a[mt], b0, b1);
    }
  }
  const float ss = 0.08838834764831845f;  // 128^-0.5 (indexer softmax_scale)
  const float hs = 0.17677669529663687f;  // 32^-0.5 (n_heads^-0.5 on the head weights)
  float col[2][2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      float v = 0.f;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int h = mt * 16 + hh * 8 + (lane >> 2);
          v += (Ws[j][h] * hs) * fmaxf(acc[mt][nt][hh * 2 + cc] * ss, 0.f);
        }
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      col[nt][cc] = v;
    }
  const int q = qb + j;
  if (lane < 4 && q < q0 + nq) {
    const int n = (src.pos(q) + 1) / kDsaPool;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int p = p0 + wq * 16 + nt * 8 + lane * 2 + cc;
        if (p < n) scores[(size_t)(q - q0) * ld + p] = col[nt][cc];
      }
  }
}

// =====================================================================================================================
// top-k (per query one block): dense below 513 visible pools, radix-select of the top 512 pools otherwise.
// =====================================================================================================================
constexpr int kTkThreads = 512;

__device__ __forceinline__ uint32_t fkey(float f) {
  if (f == 0.f) f = 0.f;  // -0 == +0 for the reference's comparisons
  const uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(kTkThreads) topk_kernel(QSrc src, int q0, const float* __restrict__ scores, int ld,
                                                         int32_t* __restrict__ idx_out) {
  using Scan = cub::BlockScan<int, kTkThreads>;
  __shared__ typename Scan::TempStorage scan_tmp;
  __shared__ uint32_t hist[256];
  __shared__ uint32_t s_prefix, s_krem;
  __shared__ int s_base_sel, s_base_eq;
  const int q = q0 + blockIdx.x, tid = threadIdx.x;
  const int pos = src.pos(q), n = (pos + 1) / kDsaPool;
  int32_t* out = idx_out + (size_t)q * kDsaIdxWidth;
  if (n <= kDsaTopPools) {  // every visible token (pools + tail) — pos + 1 ≤ 2051
    for (int e = tid; e < kDsaIdxWidth; e += kTkThreads) out[e] = e <= pos ? e : -1;
    return;
  }
  const float* row = scores + (size_t)blockIdx.x * ld;
  // radix select on the order-preserving uint32 key: find the 512th largest key K*
  uint32_t prefix = 0, mask = 0, krem = kDsaTopPools;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int e = tid; e < 256; e += kTkThreads) hist[e] = 0;
    __syncthreads();
    for (int e = tid; e < n; e += kTkThreads) {
      const uint32_t k = fkey(row[e]);
      if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      uint32_t cum = 0;
      int b = 255;
      for (; b > 0; --b) {
        if (cum + hist[b] >= krem) break;
        cum += hist[b];
      }
      s_prefix = prefix | ((uint32_t)b << shift);
      s_krem = krem - cum;
    }
    __syncthreads();
    prefix = s_prefix;
    krem = s_krem;
    mask |= 255u << shift;
  }
  // ordered compaction: keys > K* plus the first krem keys == K* (lowest pool ids), pool ids ascending
  if (tid == 0) s_base_sel = s_base_eq = 0;
  __syncthreads();
  for (int base = 0; base < n; base += kTkThreads) {
    const int e = base + tid;
    uint32_t k = 0;
    if (e < n) k = fkey(row[e]);
    const int gt = (e < n && k > prefix), eq = (e < n && k == prefix);
    int eq_ex, eq_tot;
    Scan(scan_tmp).ExclusiveSum(eq, eq_ex, eq_tot);
    __syncthreads();
    const int sel = gt || (eq && (uint32_t)(s_base_eq + eq_ex) < krem);
    int sel_ex, sel_tot;
    Scan(scan_tmp).ExclusiveSum(sel, sel_ex, sel_tot);
    if (sel) {
      const int o = (s_base_sel + sel_ex) * kDsaPool;
#pragma unroll
      for (int j = 0; j < kDsaPool; ++j) out[o + j] = e * kDsaPool + j;
    }
    __syncthreads();
    if (tid == 0) {
      s_base_eq += eq_tot;
      s_base_sel += sel_tot;
    }
    __syncthreads();
  }
  // tail rule: the incomplete pool of the query (tokens n·4 .. pos), then -1
  if (tid < kDsaPool - 1) {
    const int t = n * kDsaPool + tid;
    out[kDsaTopk + tid] = t <= pos ? t : -1;
  }
}

// =====================================================================================================================
// sparse absorbed attention core
// Block = (16-head group, token split, query); 4 warps, 2 blocks/SM. Per 32-token tile: S[16,32] = Q[16,512]·Cᵀ (warp =
// 128 latent columns of every token, reduced through smem),
// online softmax (8 threads per head row), O[16,512] += P[16,32]·C (warp = 128 latent columns).
// Precision: the core runs on fp16 tensor-core operands with fp32 accumulation — q_lat arrives pre-scaled by
// scaling·log2(e) as fp16 (11-bit mantissa vs bf16's 8; |q_lat·scaling| is far from the fp16 range limit), latent rows are
// gathered as bf16 and widened to fp16 in shared memory (exact: the RMSNorm output range is well inside fp16's normal
// range), and probabilities enter the PV product as fp16. out_lat leaves as bf16 hi + lo so the W_UV GEMM sees ~fp32.
// =====================================================================================================================
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ld32h(const __half* p) { return *reinterpret_cast<const uint32_t*>(p); }

// out_lat is stored as two bf16 terms (hi + lo) so the W_UV product sees it at ~fp32 precision.
__device__ __forceinline__ void store_hilo(bf16* hi, bf16* lo, size_t off, float x, float y) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  *reinterpret_cast<__nv_bfloat162*>(hi + off) = h;
  *reinterpret_cast<__nv_bfloat162*>(lo + off) = __floats2bfloat162_rn(x - __low2float(h), y - __high2float(h));
}

constexpr int kAtHeads = 16, kAtTile = 32, kAtLd = kDsaLat + 8, kAtThreads = 128;
// ≤ 49 KB so that two blocks share an SM (one block's gather overlaps the other's math)
constexpr int kAtSmem = kAtTile * kAtLd * 2 + 4 * kAtHeads * (kAtTile + 1) * 4 + kAtHeads * (kAtTile + 8) * 2 +
                        kAtHeads * 4 * 3 + kAtTile * 4;

// qlat: fp16 [nq, 64, 512] (pre-scaled). olat/olat_lo may alias qlat's storage (each block reads its own Q rows first).
// Q lives in registers: warp w owns latent columns [128w, 128w+128) of the 16 head rows; S is reduced over the 4 warps
// through shared memory.
__global__ void __launch_bounds__(kAtThreads, 2) attn_core_kernel(QSrc src, int q0, const __half* qlat,
                                                                 const int32_t* __restrict__ idx, bf16* olat,
                                                                 bf16* olat_lo, int nsplit, float* __restrict__ part_o,
                                                                 float* __restrict__ part_ml) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* Cs = reinterpret_cast<__half*>(smem);                     // [32][kAtLd] (bf16 on arrival, fp16 after widening)
  float* Sp = reinterpret_cast<float*>(Cs + kAtTile * kAtLd);       // [4 warps][16][33] partial scores
  __half* Ps = reinterpret_cast<__half*>(Sp + 4 * kAtHeads * (kAtTile + 1));  // [16][40]
  float* alpha_s = reinterpret_cast<float*>(Ps + kAtHeads * (kAtTile + 8));
  float* m_s = alpha_s + kAtHeads;
  float* l_s = m_s + kAtHeads;
  int* tok = reinterpret_cast<int*>(l_s + kAtHeads);  // [32]
  __shared__ int s_cnt;

  const int g = blockIdx.x, split = blockIdx.y, ql = blockIdx.z, q = q0 + ql;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int32_t* irow = idx + (size_t)ql * kDsaIdxWidth;
  const bf16* cbase = src.cp(q);

  // number of index slots to scan = last valid slot + 1
  if (tid == 0) s_cnt = 0;
  __syncthreads();
  int last = 0;
  for (int e = tid; e < kDsaIdxWidth; e += kAtThreads)
    if (irow[e] >= 0) last = e + 1;
  atomicMax(&s_cnt, last);
  __syncthreads();
  const int ntiles = (s_cnt + kAtTile - 1) / kAtTile;
  const int tps = (ntiles + nsplit - 1) / nsplit;
  const int t_beg = split * tps, t_end = min(ntiles, t_beg + tps);

  // Q fragments of this warp's 128 latent columns (8 k-steps of m16n8k16) → registers
  uint32_t qa[8][4];
  {
    const __half* qg = qlat + ((size_t)ql * kDsaHeads + g * kAtHeads) * kDsaLat;
    const __half* r0 = qg + (size_t)(lane >> 2) * kDsaLat + warp * 128 + (lane & 3) * 2;
    const __half* r1 = r0 + 8 * kDsaLat;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      qa[ks][0] = ld32h(r0 + ks * 16);
      qa[ks][1] = ld32h(r1 + ks * 16);
      qa[ks][2] = ld32h(r0 + ks * 16 + 8);
      qa[ks][3] = ld32h(r1 + ks * 16 + 8);
    }
  }
  __syncthreads();  // every warp holds its Q before out_lat may overwrite the (aliased) q_lat rows

  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;  // per head row (replicated in its 8 softmax threads)
  const int srow = tid >> 3, scol = (tid & 7) * 4;

  for (int t = t_beg; t < t_end; ++t) {
    // gather 32 latent rows (16-byte chunks: e = tid + k·128 → row e/64, chunk e%64); widen own chunks bf16 → fp16
    for (int e = tid; e < kAtTile * (kDsaLat / 8); e += kAtThreads) {
      const int r = e >> 6, c8 = e & 63;
      const int slot = t * kAtTile + r;
      const int j = slot < kDsaIdxWidth ? irow[slot] : -1;
      cp_async16(Cs + r * kAtLd + c8 * 8, cbase + (size_t)max(j, 0) * kDsaLat + c8 * 8, j >= 0 ? 16 : 0);
      if (c8 == 0) tok[r] = j;
    }
    cp_commit();
    cp_wait<0>();
    for (int e = tid; e < kAtTile * (kDsaLat / 8); e += kAtThreads) {
      uint4* p = reinterpret_cast<uint4*>(Cs + (e >> 6) * kAtLd + (e & 63) * 8);
      uint4 v = *p;
      uint32_t* w = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w[i]);
        const __half2 h = __floats2half2_rn(__low2float(b), __high2float(b));
        w[i] = *reinterpret_cast<const uint32_t*>(&h);
      }
      *p = v;
    }
    __syncthreads();
    // ---- partial S = Q[:, 128w:128w+128]·C[:, 128w:128w+128]ᵀ for all 32 tokens (4 independent n8 chains)
    {
      float s[4][4] = {};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const __half* b = Cs + (nt * 8 + (lane >> 2)) * kAtLd + warp * 128 + ks * 16 + (lane & 3) * 2;
          mma_f16(s[nt], qa[ks], ld32h(b), ld32h(b + 8));
        }
      }
      float* sp = Sp + warp * kAtHeads * (kAtTile + 1);
      const int r = lane >> 2;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int c = nt * 8 + (lane & 3) * 2;
        sp[r * (kAtTile + 1) + c] = s[nt][0];
        sp[r * (kAtTile + 1) + c + 1] = s[nt][1];
        sp[(r + 8) * (kAtTile + 1) + c] = s[nt][2];
        sp[(r + 8) * (kAtTile + 1) + c + 1] = s[nt][3];
      }
    }
    __syncthreads();
    // ---- online softmax (log2 domain)
    {
      float v[4], mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int o_ = srow * (kAtTile + 1) + scol + i;
        const float x = (Sp[o_] + Sp[kAtHeads * (kAtTile + 1) + o_]) +
                        (Sp[2 * kAtHeads * (kAtTile + 1) + o_] + Sp[3 * kAtHeads * (kAtTile + 1) + o_]);
        v[i] = tok[scol + i] >= 0 ? x : -INFINITY;
        mx = fmaxf(mx, v[i]);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
      const float m_new = fmaxf(m_run, mx);
      float alpha = 1.f, sum = 0.f;
      __half ph[4];
      if (m_new == -INFINITY) {
#pragma unroll
        for (int i = 0; i < 4; ++i) ph[i] = __float2half_rn(0.f);
      } else {
        alpha = exp2f(m_run - m_new);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ph[i] = __float2half_rn(exp2f(v[i] - m_new));
          sum += __half2float(ph[i]);  // normalise with exactly the probabilities that enter the PV product
        }
      }
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      sum += __shfl_xor_sync(0xffffffffu, sum, 4);
      l_run = l_run * alpha + sum;
      m_run = m_new;
#pragma unroll
      for (int i = 0; i < 4; ++i) Ps[srow * (kAtTile + 8) + scol + i] = ph[i];
      if ((tid & 7) == 0) alpha_s[srow] = alpha;
    }
    __syncthreads();
    // ---- O = O·alpha + P·C: warp → latent columns warp*128 .. +128
    {
      const float a_lo = alpha_s[lane >> 2], a_hi = alpha_s[(lane >> 2) + 8];
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        o[nt][0] *= a_lo;
        o[nt][1] *= a_lo;
        o[nt][2] *= a_hi;
        o[nt][3] *= a_hi;
      }
#pragma unroll
      for (int k0 = 0; k0 < kAtTile; k0 += 16) {
        uint32_t a[4];
        const __half* prow = Ps + (lane >> 2) * (kAtTile + 8) + k0 + (lane & 3) * 2;
        a[0] = ld32h(prow);
        a[1] = ld32h(prow + 8 * (kAtTile + 8));
        a[2] = ld32h(prow + 8);
        a[3] = ld32h(prow + 8 * (kAtTile + 8) + 8);
        const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
        for (int np = 0; np < 8; ++np) {  // pairs of n8 tiles
          const int n0 = warp * 128 + np * 16;
          const __half* addr = Cs + (k0 + rr + (mi & 1) * 8) * kAtLd + n0 + (mi >> 1) * 8;
          uint32_t b0, b1, b2, b3;
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                       : "r"(smem_u32(addr)));
          mma_f16(o[2 * np], a, b0, b1);
          mma_f16(o[2 * np + 1], a, b2, b3);
        }
      }
    }
    __syncthreads();
  }
  if ((tid & 7) == 0) {
    m_s[srow] = m_run;
    l_s[srow] = l_run;
  }
  __syncthreads();
  const int r_lo = lane >> 2, r_hi = r_lo + 8;
  if (nsplit == 1) {
    const float inv_lo = l_s[r_lo] > 0.f ? 1.f / l_s[r_lo] : 0.f, inv_hi = l_s[r_hi] > 0.f ? 1.f / l_s[r_hi] : 0.f;
    const size_t ob = ((size_t)ql * kDsaHeads + g * kAtHeads) * kDsaLat;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int c = warp * 128 + nt * 8 + (lane & 3) * 2;
      store_hilo(olat, olat_lo, ob + (size_t)r_lo * kDsaLat + c, o[nt][0] * inv_lo, o[nt][1] * inv_lo);
      store_hilo(olat, olat_lo, ob + (size_t)r_hi * kDsaLat + c, o[nt][2] * inv_hi, o[nt][3] * inv_hi);
    }
  } else {
    // partial: unnormalised O, m (log2 domain), l per (query, head, split)
    const size_t hb = (size_t)ql * kDsaHeads + g * kAtHeads;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int c = warp * 128 + nt * 8 + (lane & 3) * 2;
      *reinterpret_cast<float2*>(part_o + ((hb + r_lo) * nsplit + split) * kDsaLat + c) = make_float2(o[nt][0], o[nt][1]);
      *reinterpret_cast<float2*>(part_o + ((hb + r_hi) * nsplit + split) * kDsaLat + c) = make_float2(o[nt][2], o[nt][3]);
    }
    if (tid < kAtHeads) {
      part_ml[((hb + tid) * nsplit + split) * 2] = m_s[tid];
      part_ml[((hb + tid) * nsplit + split) * 2 + 1] = l_s[tid];
    }
  }
}

// Combine token splits: one block (128 threads × 4 columns) per (head, query).
__global__ void attn_combine_kernel(int nsplit, const float* __restrict__ part_o, const float* __restrict__ part_ml,
                                    bf16* __restrict__ olat, bf16* __restrict__ olat_lo) {
  const size_t hq = (size_t)blockIdx.y * kDsaHeads + blockIdx.x;
  const float* ml = part_ml + hq * nsplit * 2;
  float m = -INFINITY;
  for (int s = 0; s < nsplit; ++s) m = fmaxf(m, ml[2 * s]);
  float l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
  if (m != -INFINITY) {
    for (int s = 0; s < nsplit; ++s) {
      if (ml[2 * s] == -INFINITY) continue;
      const float wgt = exp2f(ml[2 * s] - m);
      l += wgt * ml[2 * s + 1];
      const float4 v = reinterpret_cast<const float4*>(part_o + (hq * nsplit + s) * kDsaLat)[threadIdx.x];
      acc[0] += wgt * v.x;
      acc[1] += wgt * v.y;
      acc[2] += wgt * v.z;
      acc[3] += wgt * v.w;
    }
  }
  const float inv = l > 0.f ? 1.f / l : 0.f;
  const size_t off = hq * kDsaLat + threadIdx.x * 4;
  store_hilo(olat, olat_lo, off, acc[0] * inv, acc[1] * inv);
  store_hilo(olat, olat_lo, off + 2, acc[2] * inv, acc[3] * inv);
}

// =====================================================================================================================
// host helpers
// =====================================================================================================================
constexpr int kTargetBlocks = 384;  // ~2 waves on the 188-SM RTX PRO 6000; used to choose the token split
constexpr int kMaxSplit = 32;
constexpr size_t kSelectChunkBytes = 128ull << 20;
constexpr int kAttnChunk = 256;  // queries per chunk (256 KB of workspace per query, see attention_chunk)

inline size_t align256(size_t b) { return (b + 255) & ~size_t(255); }
inline int split_for(int nq) { return nq * 4 >= kTargetBlocks ? 1 : std::min(kMaxSplit, (kTargetBlocks + nq * 4 - 1) / (nq * 4)); }
inline size_t partial_bytes(int nq, int ns) {
  if (ns == 1) return 0;
  return align256((size_t)nq * kDsaHeads * ns * kDsaLat * 4) + align256((size_t)nq * kDsaHeads * ns * 2 * 4);
}
inline size_t lat_bytes(int nq) { return align256((size_t)nq * kDsaHeads * kDsaLat * 2); }
// workspace: [fp32 q_lat, later fp32 out: 2·lat] [fp16 q_lat → bf16 out_lat hi: lat] [bf16 out_lat lo: lat] [split partials]
inline size_t attn_bytes(int nq, int ns) { return 4 * lat_bytes(nq) + partial_bytes(nq, ns); }
inline size_t attn_chunk_bytes(int nq) { return attn_bytes(nq, split_for(nq)); }
// Largest token split ≤ the preferred one that fits the workspace (1 always fits once the q_lat buffer does).
inline int split_fit(int nq, size_t ws_bytes) {
  int ns = split_for(nq);
  while (ns > 1 && attn_bytes(nq, ns) > ws_bytes) ns /= 2;
  return ns;
}

void attn_core_once() {
  static bool done = false;  // set the dynamic smem limit once per process (idempotent)
  if (!done) {
    CUDA_CHECK(cudaFuncSetAttribute(attn_core_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kAtSmem));
    done = true;
  }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ x, __half* __restrict__ y, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    const float4 v = reinterpret_cast<const float4*>(x)[i];
    reinterpret_cast<__half2*>(y)[2 * i] = __floats2half2_rn(v.x, v.y);
    reinterpret_cast<__half2*>(y)[2 * i + 1] = __floats2half2_rn(v.z, v.w);
  }
}
// q [nq, 64, 256] → qlat [nq, 64, 512] = scaling·log2(e) · W_UKᵀ q: one strided-batched GEMM over the heads (fp32 out),
// then narrowed to fp16 for the tensor-core core.
void absorb_q(cublasHandle_t h, const bf16* kv_b, const bf16* q, float* qlat32, __half* qlat, int nq, cudaStream_t st) {
  const float alpha = (1.0f / 16.0f) * 1.4426950408889634f, zero = 0.f;  // 256^-0.5 · log2(e)
  DSA_CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kDsaLat, nq, kDsaQkDim, &alpha, kv_b, CUDA_R_16BF,
                                              kDsaLat, (long long)2 * kDsaQkDim * kDsaLat, q, CUDA_R_16BF,
                                              kDsaHeads * kDsaQkDim, kDsaQkDim, &zero, qlat32, CUDA_R_32F,
                                              kDsaHeads * kDsaLat, kDsaLat, kDsaHeads, CUBLAS_COMPUTE_32F,
                                              CUBLAS_GEMM_DEFAULT));
  const size_t n4 = (size_t)nq * kDsaHeads * kDsaLat / 4;
  f32_to_f16_kernel<<<(unsigned)std::min<size_t>((n4 + 255) / 256, 4096), 256, 0, st>>>(qlat32, qlat, n4);
  CUDA_CHECK(cudaGetLastError());
}
// (olat_hi + olat_lo) [nq, 64, 512] → out32 [nq, 64, 256] fp32 (two accumulating GEMMs) → out bf16
__global__ void f32_to_bf16_kernel(const float* __restrict__ x, bf16* __restrict__ y, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    const float4 v = reinterpret_cast<const float4*>(x)[i];
    reinterpret_cast<__nv_bfloat162*>(y)[2 * i] = __floats2bfloat162_rn(v.x, v.y);
    reinterpret_cast<__nv_bfloat162*>(y)[2 * i + 1] = __floats2bfloat162_rn(v.z, v.w);
  }
}
void expand_v(cublasHandle_t h, const bf16* kv_b, const bf16* olat, const bf16* olat_lo, float* out32, bf16* out, int nq,
              cudaStream_t st) {
  const float one = 1.f;
  for (int part = 0; part < 2; ++part) {
    const float beta = part ? 1.f : 0.f;
    DSA_CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kDsaVDim, nq, kDsaLat, &one,
                                                kv_b + (size_t)kDsaQkDim * kDsaLat, CUDA_R_16BF, kDsaLat,
                                                (long long)2 * kDsaQkDim * kDsaLat, part ? olat_lo : olat, CUDA_R_16BF,
                                                kDsaHeads * kDsaLat, kDsaLat, &beta, out32, CUDA_R_32F,
                                                kDsaHeads * kDsaVDim, kDsaVDim, kDsaHeads, CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT));
  }
  const size_t n4 = (size_t)nq * kDsaHeads * kDsaVDim / 4;
  f32_to_bf16_kernel<<<(unsigned)std::min<size_t>((n4 + 255) / 256, 4096), 256, 0, st>>>(out32, out, n4);
  CUDA_CHECK(cudaGetLastError());
}

// Attention for queries [q0, q0+nq) of src (q / idx / out already offset to q0).
void attention_chunk(const DsaLayerW& w, const QSrc& src, int q0, int nq, const bf16* q, const int32_t* idx, bf16* out,
                     void* ws, size_t ws_bytes, cudaStream_t st) {
  attn_core_once();
  cublasHandle_t h = blas(st);
  char* base = reinterpret_cast<char*>(ws);
  float* f32buf = reinterpret_cast<float*>(base);                 // fp32 q_lat, later the fp32 output
  __half* qlat = reinterpret_cast<__half*>(base + 2 * lat_bytes(nq));
  bf16* lat = reinterpret_cast<bf16*>(qlat);                       // out_lat hi overwrites q_lat in place
  bf16* lat_lo = reinterpret_cast<bf16*>(base + 3 * lat_bytes(nq));
  const int ns = split_fit(nq, ws_bytes);
  HIVE_CHECK(attn_bytes(nq, ns) <= ws_bytes, "dsa attention: workspace too small");
  float* part_o = nullptr;
  float* part_ml = nullptr;
  if (ns > 1) {
    part_o = reinterpret_cast<float*>(base + 4 * lat_bytes(nq));
    part_ml = reinterpret_cast<float*>(reinterpret_cast<char*>(part_o) +
                                       align256((size_t)nq * kDsaHeads * ns * kDsaLat * 4));
  }
  absorb_q(h, w.kv_b, q, f32buf, qlat, nq, st);
  attn_core_kernel<<<dim3(kDsaHeads / kAtHeads, ns, nq), kAtThreads, kAtSmem, st>>>(src, q0, qlat, idx, lat, lat_lo, ns,
                                                                                    part_o, part_ml);
  CUDA_CHECK(cudaGetLastError());
  if (ns > 1) {
    attn_combine_kernel<<<dim3(kDsaHeads, nq), kDsaLat / 4, 0, st>>>(ns, part_o, part_ml, lat, lat_lo);
    CUDA_CHECK(cudaGetLastError());
  }
  expand_v(h, w.kv_b, lat, lat_lo, f32buf, out, nq, st);
}

}  // namespace

// =====================================================================================================================
// public API
// =====================================================================================================================
void dsa_set_cublas_handle(void* hnd) { g_user_handle = reinterpret_cast<cublasHandle_t>(hnd); }

void dsa_append(const DsaLayerW& w, DsaSeqCache& cache, int pos0, int T, const bf16* c_new, const bf16* kI_new,
                const bf16* gs_new, cudaStream_t st) {
  if (T <= 0) return;
  HIVE_CHECK(pos0 >= 0 && pos0 + T <= cache.cap, "dsa_append: cache overflow");
  QSrc dst{nullptr, nullptr, nullptr, pos0};
  append_rows_kernel<<<T, 64, 0, st>>>(dst, cache.c, cache.kI, cache.gs, c_new, kI_new, gs_new, nullptr);
  CUDA_CHECK(cudaGetLastError());
  const int p_beg = pos0 / kDsaPool, p_end = (pos0 + T) / kDsaPool;  // pools whose last token is in [pos0, pos0+T)
  if (p_end > p_beg) {
    pool_prefill_kernel<<<p_end - p_beg, kDsaIdxDim, 0, st>>>(cache.kI, cache.gs, w.ape, cache.pooled, p_beg);
    CUDA_CHECK(cudaGetLastError());
  }
}

size_t dsa_select_ws_bytes(int T, int ctx) {
  const int ld = ctx / kDsaPool;
  if (ld <= kDsaTopPools || T <= 0) return 0;
  const size_t row = (size_t)ld * 4;
  const size_t rows = std::min<size_t>((size_t)T, std::max<size_t>(64, kSelectChunkBytes / row));
  return rows * row;
}

void dsa_select(const DsaSeqCache& cache, int pos0, int T, const bf16* qI, const float* wI, int32_t* idx, void* ws,
                size_t ws_bytes, cudaStream_t st) {
  if (T <= 0) return;
  QSrc src{nullptr, cache.c, cache.pooled, pos0};
  const int ld = (pos0 + T) / kDsaPool;
  // queries at pos ≥ 2051 have > 512 visible pools and need scores; earlier ones are dense
  const int i_sparse = std::min(T, std::max(0, kDsaIdxWidth - pos0));
  if (i_sparse > 0) {
    topk_kernel<<<i_sparse, kTkThreads, 0, st>>>(src, 0, nullptr, 0, idx);
    CUDA_CHECK(cudaGetLastError());
  }
  if (i_sparse == T) return;
  const size_t row = (size_t)ld * 4;
  HIVE_CHECK(ws && ws_bytes >= row, "dsa_select: workspace too small");
  const int chunk = (int)std::min<size_t>((size_t)(T - i_sparse), ws_bytes / row);
  float* scores = reinterpret_cast<float*>(ws);
  for (int q0 = i_sparse; q0 < T; q0 += chunk) {
    const int nq = std::min(chunk, T - q0);
    const int ld_c = (pos0 + q0 + nq) / kDsaPool;  // pools visible to the chunk's last query
    score_kernel<2><<<dim3((ld_c + kScPools - 1) / kScPools, (nq + 1) / 2), 256, 0, st>>>(src, q0, nq, qI, wI, scores, ld);
    CUDA_CHECK(cudaGetLastError());
    topk_kernel<<<nq, kTkThreads, 0, st>>>(src, q0, scores, ld, idx);
    CUDA_CHECK(cudaGetLastError());
  }
}

size_t dsa_attention_ws_bytes(int T) { return T <= 0 ? 0 : attn_chunk_bytes(std::min(T, kAttnChunk)); }

void dsa_attention(const DsaLayerW& w, const DsaSeqCache& cache, int pos0, int T, const bf16* q, const int32_t* idx,
                   bf16* out, void* ws, size_t ws_bytes, cudaStream_t st) {
  if (T <= 0) return;
  QSrc src{nullptr, cache.c, cache.pooled, pos0};
  int chunk = std::min(T, kAttnChunk);
  while (chunk > 1 && attn_bytes(chunk, 1) > ws_bytes) chunk /= 2;
  HIVE_CHECK(attn_bytes(chunk, 1) <= ws_bytes, "dsa_attention: workspace too small");
  for (int q0 = 0; q0 < T; q0 += chunk) {
    const int nq = std::min(chunk, T - q0);
    attention_chunk(w, src, q0, nq, q + (size_t)q0 * kDsaHeads * kDsaQkDim, idx + (size_t)q0 * kDsaIdxWidth,
                    out + (size_t)q0 * kDsaHeads * kDsaVDim, ws, ws_bytes, st);
  }
}

void dsa_append_decode(const DsaLayerW& w, const DsaDecodeRow* rows, int M, const bf16* c_new, const bf16* kI_new,
                       const bf16* gs_new, cudaStream_t st) {
  if (M <= 0) return;
  QSrc dst{rows, nullptr, nullptr, 0};
  append_rows_kernel<<<M, 64, 0, st>>>(dst, nullptr, nullptr, nullptr, c_new, kI_new, gs_new, rows);
  CUDA_CHECK(cudaGetLastError());
  pool_decode_kernel<<<M, kDsaIdxDim, 0, st>>>(rows, w.ape);
  CUDA_CHECK(cudaGetLastError());
}

size_t dsa_select_decode_ws_bytes(int M, int max_ctx) {
  const int ld = max_ctx / kDsaPool;
  return M <= 0 ? 0 : (size_t)M * std::max(ld, 1) * 4;
}

void dsa_select_decode(const DsaDecodeRow* rows, int M, int max_ctx, const bf16* qI, const float* wI, int32_t* idx,
                       void* ws, size_t ws_bytes, cudaStream_t st) {
  if (M <= 0) return;
  QSrc src{rows, nullptr, nullptr, 0};
  const int ld = std::max(max_ctx / kDsaPool, 1);
  HIVE_CHECK(ws && ws_bytes >= dsa_select_decode_ws_bytes(M, max_ctx), "dsa_select_decode: workspace too small");
  float* scores = reinterpret_cast<float*>(ws);
  if (ld > kDsaTopPools) {
    score_kernel<1><<<dim3((ld + kScPools - 1) / kScPools, M), 128, 0, st>>>(src, 0, M, qI, wI, scores, ld);
    CUDA_CHECK(cudaGetLastError());
  }
  topk_kernel<<<M, kTkThreads, 0, st>>>(src, 0, scores, ld, idx);
  CUDA_CHECK(cudaGetLastError());
}

size_t dsa_attention_decode_ws_bytes(int M) { return M <= 0 ? 0 : attn_chunk_bytes(M); }

void dsa_attention_decode(const DsaLayerW& w, const DsaDecodeRow* rows, int M, const bf16* q, const int32_t* idx,
                          bf16* out, void* ws, size_t ws_bytes, cudaStream_t st) {
  if (M <= 0) return;
  HIVE_CHECK(ws_bytes >= attn_bytes(M, 1), "dsa_attention_decode: workspace too small");
  QSrc src{rows, nullptr, nullptr, 0};
  attention_chunk(w, src, 0, M, q, idx, out, ws, ws_bytes, st);
}

}  // namespace hive::glm
