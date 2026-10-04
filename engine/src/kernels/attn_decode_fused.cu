// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_ATTN_FUSED — fused attention front end for decode (M ≤ 8). Contract and scope: header comment of hive/attn_decode_fused.h.
//
// Launches per layer (M=1, the attention_decode_dev share of graph A — the wo_a/wo_b launches are common to both paths and not counted):
//   window/compressed consumer layers (0·1·3~7·9~13·15~19·21~23·…): unfused q_a · q_b · kv · attn · merge = 5 → qkv_a · qb · attn = 3
//   candidate-pool index layers (24·28·32·36): unfused 15 → 7 (qkv_a · qb+iq · cuBLAS iw · scores · top-k (+cand_to_pos·offset) · attn)
//   kv + index sources (2·8·14 ratio 2 · 20 ratio 1): unfused 26 → 12 (20 includes block_max and the block top-k)
// Row table: the unfused kernels read pos·KvRow·kptrs·trows·visible·ring/cache pointers from mapped pinned memory in every block (a ~1–2 µs PCIe round trip
//   at each block's tail, and the kv tail read pos three times serially per row, ~17 round trips at M=8). Here the last kv block copies them once to a device copy (same values — no numeric effect).
#include <cuda_fp8.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>

#include "hive/attn_decode_fused.h"
#include "hive/decode_attn2.h"  // HIVE_DECODE_ATTN2 (q_b columns — dec_attention_front ②)
#include "hive/decode_attn3.h"  // HIVE_DECODE_QKV3 · HIVE_DECODE_SPARSE3 (① q_a‖kv · ⑤ sparse attention)
#include "hive/decode_longctx.h"  // HIVE_DECODE_TOPK2 · HIVE_DECODE_IDXSCORE2 (④ indexer — indexer_stage)
#include "hive/cublas_ops.h"
#include "hive/kernels.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "hive/verify_decode.h"  // HIVE_MTP_VERIFY2_FUSED: verify-mode compressor (compressor_step_seq — the verify-path kernel unchanged)
#include "hive/warp_reduce.cuh"

namespace hive::k {

namespace {

constexpr int FT = 256;
constexpr int FW = FT / 32;
constexpr int OUT = 8;      // q_a·kv·q_b block = 8 columns (1 per warp) — same parallelism as fused.cu
constexpr int OUT_I = 32;   // indexer query block = 32 columns (4 per warp) — the 32-element fp4 round-trip block closes within one block
constexpr int MMAX = 8;
constexpr float kFp8MaxInv = 1.0f / 448.0f;
constexpr float kFp4MaxInv = 1.0f / 6.0f;

// ---- same code as the device helpers in fused.cu (copied because they live in an anonymous namespace — formulas and order unchanged) --------------------------------------
__device__ __forceinline__ float2 cvt_e4m3x2_f(uint32_t pair16) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)pair16));
  return __half22float2(*reinterpret_cast<__half2*>(&h2));
}
__device__ __forceinline__ float e8m0_fast_f(uint8_t b) { return b ? __uint_as_float((uint32_t)b << 23) : __uint_as_float(0x00400000u); }
__device__ __forceinline__ void decode8(const uint4 r0, const uint4 r1, float (&w)[32]) {
  const uint32_t x[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 a = cvt_e4m3x2_f(x[i] & 0xFFFFu), b = cvt_e4m3x2_f(x[i] >> 16);
    w[4 * i] = a.x; w[4 * i + 1] = a.y; w[4 * i + 2] = b.x; w[4 * i + 3] = b.y;
  }
}
__device__ __forceinline__ void quant_rows_smem(const bf16* __restrict__ A, int lda, int M, int K, uint8_t* aq, float* asv, uint8_t* asc) {
  const int nb = K / 32;
  for (int idx = threadIdx.x; idx < M * nb; idx += FT) {
    const int m = idx / nb, b = idx % nb;
    const bf16* p = A + (size_t)m * lda + b * 32;
    float v[32];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { v[i] = bf2f(p[i]); amax = fmaxf(amax, fabsf(v[i])); }
    amax = fmaxf(amax, 1e-4f);
    const uint8_t code = f32_ceil_pow2_e8m0(amax * kFp8MaxInv);
    const float sc = e8m0_to_f32(code);
    uint8_t* q = aq + (size_t)m * K + b * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) q[i] = f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.f), 448.f));
    asv[m * nb + b] = sc;
    asc[m * nb + b] = code;
  }
}
// Warp: column n × M rows, A = smem (e4m3 + scale values). Same order as fused.cu warp_dot<true> / gemm.cu gemv_bs_kernel (same scale values: e8m0_to_f32 == e8m0_fast).
__device__ __forceinline__ void warp_dot(const uint8_t* __restrict__ aq, const float* __restrict__ asv, int lda_q, const uint8_t* __restrict__ B,
                                         const uint8_t* __restrict__ sb, int M, int K, int n, int lane, float (&acc)[MMAX]) {
  const int nb = K / 32;
#pragma unroll
  for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
  for (int b = lane; b < nb; b += 32) {
    float w[32];
    const uint8_t* pb = B + (size_t)n * K + b * 32;
    decode8(*reinterpret_cast<const uint4*>(pb), *reinterpret_cast<const uint4*>(pb + 16), w);
    const float sbv = e8m0_fast_f(sb[(size_t)n * nb + b]);
#pragma unroll
    for (int m = 0; m < MMAX; ++m) {
      if (m < M) {
        const uint8_t* pa = aq + (size_t)m * lda_q + b * 32;
        float av[32];
        decode8(*reinterpret_cast<const uint4*>(pa), *reinterpret_cast<const uint4*>(pa + 16), av);
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) d = fmaf(av[i], w[i], d);
        acc[m] = fmaf(d, asv[m * nb + b] * sbv, acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MMAX; ++m) {
    float v = acc[m];
    v = hive::cu::warp_allreduce(v);
    acc[m] = v;
  }
}
// Same reduction as rmsnorm_kernel<256> (256 threads, i = tid, tid+256, … sequential, warp xor, red[8] via xor over lane<8). out may alias x.
__device__ __forceinline__ void rmsnorm_row_256(const bf16* x, const bf16* __restrict__ w, float eps, int N, bf16* out, float* red) {
  float ss = 0.f;
  for (int i = threadIdx.x; i < N; i += 256) { float v = bf2f(x[i]); ss += v * v; }
  ss = hive::cu::warp_allreduce(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    t = hive::cu::warp_allreduce(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / (float)N + eps);
  for (int i = threadIdx.x; i < N; i += 256) out[i] = f2bf(bf2f(w[i]) * (bf2f(x[i]) * rs));
  __syncthreads();
}
// Is this the last of n blocks (threadfence reduction)? The last block resets the counter to 0 (safe for graph replay).
__device__ __forceinline__ bool last_of(int* counter, int n, int* flag_smem) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) *flag_smem = (atomicAdd(counter, 1) == n - 1);
  __syncthreads();
  const bool last = *flag_smem != 0;
  if (last) { __threadfence(); if (threadIdx.x == 0) *counter = 0; }
  return last;
}
__device__ __forceinline__ void stage_row(const DecRowSrc& s, int m, DecRow* tab, DecSoA* soa) {
  DecRow r;
  r.kv = s.kv[m];
  r.kptr = s.kptr ? s.kptr[m] : nullptr;
  r.dstp = s.dstp ? s.dstp[m] : nullptr;
  r.dstk = s.dstk ? s.dstk[m] : nullptr;
  r.skv = s.skv ? s.skv[m] : nullptr;
  r.ssc = s.ssc ? s.ssc[m] : nullptr;
  r.pos = s.pos[m];
  r.gpos = s.gpos ? s.gpos[m] : 0;
  r.visible = s.visible ? s.visible[m] : 0;
  r.trows = s.trows ? s.trows[m] : 0;
  r.dsti = s.dsti ? s.dsti[m] : 0;
  r.pad_[0] = r.pad_[1] = r.pad_[2] = 0;
  tab[m] = r;
  if (soa) { soa->kptr[m] = r.kptr; soa->trows[m] = r.trows; soa->visible[m] = r.visible; }
}

// ---- ① q_a ‖ kv: the two GEMVs that consume the same quant(xn) in one launch. Blocks [0, Nqa/8) = wq_a columns, [Nqa/8, Nqa/8 + D/8) = wkv columns.
//   Last q_a block: qrn = rmsnorm(qr) (gemv_qa_kernel tail). Last kv block: row-table copy + window indices + kv chain (gemv_kv_kernel tail).
//   If xf is given, the blocks also share writing the fp32 copy of xn (bf16_to_f32 — cuBLAS input of the ratio>1 compressor).
__global__ void __launch_bounds__(FT) dec_qkv_a_kernel(const bf16* __restrict__ xn, int K, const uint8_t* __restrict__ wqa, const uint8_t* __restrict__ sqa,
                                                       int Nqa, const bf16* __restrict__ q_norm, const uint8_t* __restrict__ wkv,
                                                       const uint8_t* __restrict__ skv, int D, const bf16* __restrict__ kv_norm, float eps, int rd,
                                                       const float2* __restrict__ freqs, int M, bf16* __restrict__ qr, bf16* __restrict__ qrn,
                                                       bf16* __restrict__ kv, bf16* const* __restrict__ ring_ptrs, int win, int32_t* __restrict__ idx_out,
                                                       int idx_stride, float* __restrict__ xf, DecRowSrc src, DecRow* __restrict__ tab,
                                                       DecSoA* __restrict__ soa, int* __restrict__ counters, const int32_t* __restrict__ vgrp) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (K / 32));
  __shared__ float outv[MMAX][OUT];
  __shared__ __align__(16) bf16 row[512];
  __shared__ float red[8];
  __shared__ int flag;
  __shared__ int spos[MMAX];
  __shared__ int sgrp[MMAX];  // verify mode: first row of the part that row m belongs to (vgrp — mapped pinned)
  __shared__ bf16* sring[MMAX];
  quant_rows_smem(xn, K, M, K, aq, asv, asc);
  if (xf) for (int i = blockIdx.x * FT + threadIdx.x; i < M * K; i += gridDim.x * FT) xf[i] = bf2f(xn[i]);
  __syncthreads();
  const int nqb = Nqa / OUT;
  const bool isq = (int)blockIdx.x < nqb;
  const int n0 = (isq ? blockIdx.x : blockIdx.x - nqb) * OUT;
  const uint8_t* B = isq ? wqa : wkv;
  const uint8_t* sb = isq ? sqa : skv;
  const int N = isq ? Nqa : D;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  {
    float acc[MMAX];
    warp_dot(aq, asv, K, B, sb, M, K, n0 + warp, lane, acc);
    if (lane == 0) for (int m = 0; m < M; ++m) outv[m][warp] = acc[m];
  }
  __syncthreads();
  bf16* C = isq ? qr : kv;
  if (threadIdx.x < M * OUT) { const int m = threadIdx.x / OUT, c = threadIdx.x % OUT; C[(size_t)m * N + n0 + c] = f2bf(outv[m][c]); }
  if (isq) {
    if (!last_of(counters + 0, nqb, &flag)) return;
    for (int m = 0; m < M; ++m) rmsnorm_row_256(qr + (size_t)m * Nqa, q_norm, eps, Nqa, qrn + (size_t)m * Nqa, red);
    return;
  }
  if (!last_of(counters + 1, D / OUT, &flag)) return;
  if (threadIdx.x < M) {  // mapped pinned tables in one go (independent loads — one round trip) → device copy; pos and ring pointers go to smem
    const int m = threadIdx.x;
    stage_row(src, m, tab, soa);
    spos[m] = src.pos[m];
    sgrp[m] = vgrp ? vgrp[m] : m;
    sring[m] = ring_ptrs ? ring_ptrs[m] : nullptr;
  }
  __syncthreads();
  if (idx_out && vgrp) {  // verify mode: same formula as window_idxs_verify_kernel (before the part's first position = ring slot, inside the part = chunk row win + g + (src − p0))
    for (int i = threadIdx.x; i < M * win; i += FT) {
      const int m = i / win, j = i % win, g = sgrp[m];
      const int64_t p = spos[m], p0 = spos[g], srcp = p - (win - 1) + j;
      idx_out[(size_t)m * idx_stride + j] = srcp < 0 ? -1 : (srcp < p0 ? (int32_t)(srcp % win) : (int32_t)(win + g + (srcp - p0)));
    }
  } else if (idx_out) {  // same formula as window_idxs_rows
    for (int i = threadIdx.x; i < M * win; i += FT) {
      const int m = i / win, j = i % win;
      const int64_t p = spos[m], srcp = p - (win - 1) + j;
      idx_out[(size_t)m * idx_stride + j] = srcp < 0 ? -1 : (srcp < p ? (int32_t)(srcp % win) : (int32_t)(win + m));
    }
  }
  const int half = rd / 2;
  for (int m = 0; m < M; ++m) {  // same chain as the gemv_kv_kernel tail (rmsnorm → RoPE → fp8 round trip → kv row + ring)
    bf16* kvm = kv + (size_t)m * D;
    rmsnorm_row_256(kvm, kv_norm, eps, D, row, red);
    if (threadIdx.x < half) {
      const float2 f = freqs[(size_t)spos[m] * half + threadIdx.x];
      bf16* base = row + (D - rd) + 2 * threadIdx.x;
      const float a = bf2f(base[0]), b = bf2f(base[1]);
      base[0] = f2bf(a * f.x - b * f.y);
      base[1] = f2bf(a * f.y + b * f.x);
    }
    __syncthreads();
    if (threadIdx.x < D / 32) {
      bf16* p = row + threadIdx.x * 32;
      float v[32];
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) { v[i] = bf2f(p[i]); amax = fmaxf(amax, fabsf(v[i])); }
      amax = fmaxf(amax, 1e-4f);
      const float sc = e8m0_to_f32(f32_ceil_pow2_e8m0(amax * kFp8MaxInv));
#pragma unroll
      for (int i = 0; i < 32; i++) {
        const float t = fminf(fmaxf(v[i] / sc, -448.f), 448.f);
        p[i] = f2bf(e4m3_to_f32(f32_to_e4m3(t)) * sc);
      }
    }
    __syncthreads();
    bf16* ring = sring[m] ? sring[m] + (size_t)(spos[m] % win) * D : nullptr;
    for (int i = threadIdx.x; i < D; i += FT) { kvm[i] = row[i]; if (ring) ring[i] = row[i]; }
    __syncthreads();
  }
}

// ---- ② q_b (+RoPE) ‖ indexer query (idx_wq_b + RoPE + fp4 round trip). Blocks [0, HD/8) = the same work as gemv_qb_rope_kernel (pos from the device copy),
//   [HD/8, HD/8 + NI/32) = 32 indexer query columns: same values as gemm_bs(qrq·qrs) (gemv_bs_kernel) → bf16 → rope_last → fp4_quant_roundtrip(32, e8m0).
__global__ void __launch_bounds__(FT) dec_qb_kernel(const bf16* __restrict__ qrn, int K, const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb,
                                                    int M, int N, int D, int rd, const float2* __restrict__ freqs, const DecRow* __restrict__ tab,
                                                    bf16* __restrict__ q, uint8_t* __restrict__ side_q, uint8_t* __restrict__ side_s,
                                                    const uint8_t* __restrict__ Bi, const uint8_t* __restrict__ sbi, int NI, int Di,
                                                    const float2* __restrict__ freqs_i, bf16* __restrict__ iq, bf16* __restrict__ iw, int n_iw, float wscale,
                                                    uint8_t* __restrict__ iqp) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (K / 32));
  __shared__ float outv[MMAX][OUT_I];
  __shared__ int spos[MMAX];
  quant_rows_smem(qrn, K, M, K, aq, asv, asc);
  if (threadIdx.x < M) spos[threadIdx.x] = tab[threadIdx.x].pos;
  __syncthreads();
  if (blockIdx.x == 0 && side_q) {
    const int nb = K / 32;
    for (int i = threadIdx.x; i < M * K; i += FT) side_q[i] = aq[i];
    for (int i = threadIdx.x; i < M * nb; i += FT) side_s[i] = asc[i];
  }
  if (blockIdx.x == 0 && iw) for (int i = threadIdx.x; i < n_iw; i += FT) iw[i] = f2bf(bf2f(iw[i]) * wscale);  // scale_bf16(iw) — the consumer (scores) is the next launch
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nqb = N / OUT;
  if ((int)blockIdx.x < nqb) {
    const int n0 = blockIdx.x * OUT;
    {
      float acc[MMAX];
      warp_dot(aq, asv, K, B, sb, M, K, n0 + warp, lane, acc);
      if (lane == 0) for (int m = 0; m < M; ++m) outv[m][warp] = acc[m];
    }
    __syncthreads();
    if (threadIdx.x >= M * OUT) return;
    const int m = threadIdx.x / OUT, c = threadIdx.x % OUT, n = n0 + c;
    const int dpos = n % D;
    if (dpos < D - rd) { q[(size_t)m * N + n] = f2bf(outv[m][c]); return; }
    if (c & 1) return;
    const int p = (dpos - (D - rd)) >> 1;
    const float2 f = freqs[(size_t)spos[m] * (rd / 2) + p];
    const float a = bf2f(f2bf(outv[m][c])), b = bf2f(f2bf(outv[m][c + 1]));
    q[(size_t)m * N + n] = f2bf(a * f.x - b * f.y);
    q[(size_t)m * N + n + 1] = f2bf(a * f.y + b * f.x);
    return;
  }
  // indexer query blocks
  const int n0 = (blockIdx.x - nqb) * OUT_I;
#pragma unroll
  for (int j = 0; j < OUT_I / FW; ++j) {
    const int c = warp * (OUT_I / FW) + j;
    float acc[MMAX];
    warp_dot(aq, asv, K, Bi, sbi, M, K, n0 + c, lane, acc);
    if (lane == 0) for (int m = 0; m < M; ++m) outv[m][c] = bf2f(f2bf(acc[m]));  // gemv output is bf16
  }
  __syncthreads();
  const int dpos0 = n0 % Di;
  if (dpos0 >= Di - rd && threadIdx.x < M * (OUT_I / 2)) {  // rope_last(iq): pairs (2p, 2p+1) — all 32 columns of the block are in the rotary range
    const int m = threadIdx.x / (OUT_I / 2), c = 2 * (threadIdx.x % (OUT_I / 2));
    const int p = (dpos0 + c - (Di - rd)) >> 1;
    const float2 f = freqs_i[(size_t)spos[m] * (rd / 2) + p];
    const float a = outv[m][c], b = outv[m][c + 1];
    outv[m][c] = bf2f(f2bf(a * f.x - b * f.y));
    outv[m][c + 1] = bf2f(f2bf(a * f.y + b * f.x));
  }
  __syncthreads();
  if (threadIdx.x < M) {  // fp4_roundtrip_kernel<32, false>: thread = (row, 32-block); with idx_mode 1 also fp4_pack_kernel<32, false>(iqp) on the pre-round-trip values
    const int m = threadIdx.x;
    float v[32];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { v[i] = outv[m][i]; amax = fmaxf(amax, fabsf(v[i])); }
    amax = fmaxf(amax, 6.f * ldexpf(1.f, -126));
    const uint8_t code = f32_ceil_pow2_e8m0(amax * kFp4MaxInv);
    const float sc = e8m0_to_f32(code);
    if (iqp) {  // row = m·Hi + head (n0 / Di), block = (n0 % Di) / 32 — same bytes as fp4_pack(iq, M·Hi, Di, 32, false, iqp, IDX_ROW)
      uint8_t* prow = iqp + ((size_t)m * (NI / Di) + n0 / Di) * kvp::IDX_ROW;
      const int b = (n0 % Di) / 32;
      prow[b] = code;
      uint8_t* nib = prow + (kvp::IDX_ROW - Di / 2) + (b * 32) / 2;
#pragma unroll
      for (int i = 0; i < 32; i += 2) {
        const uint8_t lo = f32_to_e2m1(fminf(fmaxf(v[i] / sc, -6.f), 6.f)), hi = f32_to_e2m1(fminf(fmaxf(v[i + 1] / sc, -6.f), 6.f));
        nib[i / 2] = (uint8_t)(lo | (hi << 4));
      }
    }
    bf16* dst = iq + (size_t)m * NI + n0;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float t = fminf(fmaxf(v[i] / sc, -6.f), 6.f);
      dst[i] = f2bf(e2m1_to_f32(f32_to_e2m1(t)) * sc);
    }
  }
}

// ---- ③ compressor tail (row = block, 256 threads): [ratio>1: compressor_step_rows → f32_to_bf16] → rmsnorm(comp_norm) → latent (normalized, pre-RoPE — idx_wk input)
//   → RoPE(gpos) → fp4_pack(16, e4m3) → compressed cache. ratio==1 reads the latent (cuBLAS output) and sets valid=1 (replaces a memset).
//   Difference from the unfused path: !valid rows (ratio>1, incomplete group) leave the latent untouched (the unfused path computed it from a stale cout; the consumer, idx_wk → ik pack, discarded it via valid).
__global__ void __launch_bounds__(256) dec_compress_kernel(int ratio, int D, const float* __restrict__ ckv, const float* __restrict__ cscore,
                                                           float* __restrict__ cout, bf16* __restrict__ latent, const bf16* __restrict__ norm_w, float eps,
                                                           int rd, const float2* __restrict__ freqs, const DecRow* __restrict__ tab, uint8_t* __restrict__ valid,
                                                           bool stepped) {
  __shared__ __align__(16) bf16 row[512];
  __shared__ float red[8];
  const int m = blockIdx.x;
  const DecRow& r = tab[m];
  if (ratio > 1 && stepped) {  // verify mode: state, cout and valid were written by the preceding compressor_step_seq launch (row order) — only that cout is read here (same value as row[d] = f2bf(acc) in the other branch)
    if (!valid[m]) return;
    for (int d = threadIdx.x; d < D; d += blockDim.x) row[d] = f2bf(cout[(size_t)m * D + d]);
  } else if (ratio > 1) {
    const int p = r.pos;
    const int slot = p % ratio;
    float* skv = r.skv;
    float* ssc = r.ssc;
    const bool should = ((p + 1) % ratio) == 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      skv[slot * D + d] = ckv[(size_t)m * D + d];
      ssc[slot * D + d] = cscore[(size_t)m * D + d];
    }
    __syncthreads();
    if (threadIdx.x == 0) valid[m] = should ? 1 : 0;
    if (!should) return;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float mx = -FLT_MAX;
      for (int rr = 0; rr < ratio; ++rr) mx = fmaxf(mx, ssc[rr * D + d]);
      float sum = 0.f, acc = 0.f, e[8];
      for (int rr = 0; rr < ratio; ++rr) { e[rr] = expf(ssc[rr * D + d] - mx); sum += e[rr]; }
      for (int rr = 0; rr < ratio; ++rr) acc += skv[rr * D + d] * (e[rr] / sum);
      cout[(size_t)m * D + d] = acc;
      row[d] = f2bf(acc);
    }
  } else {
    for (int d = threadIdx.x; d < D; d += blockDim.x) row[d] = latent[(size_t)m * D + d];
    if (threadIdx.x == 0) valid[m] = 1;
  }
  __syncthreads();
  rmsnorm_row_256(row, norm_w, eps, D, row, red);
  for (int d = threadIdx.x; d < D; d += blockDim.x) latent[(size_t)m * D + d] = row[d];
  const int half = rd / 2;
  if (threadIdx.x < half) {  // rope_last(latent, H=1, gpos)
    const float2 f = freqs[(size_t)r.gpos * half + threadIdx.x];
    bf16* base = row + (D - rd) + 2 * threadIdx.x;
    const float a = bf2f(base[0]), b = bf2f(base[1]);
    base[0] = f2bf(a * f.x - b * f.y);
    base[1] = f2bf(a * f.y + b * f.x);
  }
  __syncthreads();
  if (threadIdx.x < D / 16) {  // fp4_pack_kernel<16, true>
    const int b = threadIdx.x;
    uint8_t* dst = r.dstp + (size_t)r.dsti * kvp::COMP_ROW;
    float v[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float x = bf2f(row[b * 16 + i]);
      v[i] = x;
      amax = fmaxf(amax, fabsf(x));
    }
    amax = fmaxf(amax, 6.f * ldexpf(1.f, -9));
    const uint8_t code = f32_to_e4m3(amax * kFp4MaxInv);
    const float sc = e4m3_to_f32(code);
    dst[b] = code;
    uint8_t* nib = dst + (kvp::COMP_ROW - D / 2) + (b * 16) / 2;
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      const uint8_t lo = f32_to_e2m1(fminf(fmaxf(v[i] / sc, -6.f), 6.f)), hi = f32_to_e2m1(fminf(fmaxf(v[i + 1] / sc, -6.f), 6.f));
      nib[i / 2] = (uint8_t)(lo | (hi << 4));
    }
  }
}

// ---- ④ index-key tail (row = block, 256 threads): rmsnorm(idx_k_norm) → RoPE(gpos) → fp4_pack(32, e8m0) → index-key cache (valid rows only)
__global__ void __launch_bounds__(256) dec_ik_kernel(int Di, const bf16* __restrict__ ik, const bf16* __restrict__ norm_w, float eps, int rd,
                                                     const float2* __restrict__ freqs, const DecRow* __restrict__ tab, const uint8_t* __restrict__ valid) {
  __shared__ __align__(16) bf16 row[128];
  __shared__ float red[8];
  const int m = blockIdx.x;
  if (!valid[m]) return;
  const DecRow& r = tab[m];
  for (int d = threadIdx.x; d < Di; d += blockDim.x) row[d] = ik[(size_t)m * Di + d];
  __syncthreads();
  rmsnorm_row_256(row, norm_w, eps, Di, row, red);
  const int half = rd / 2;
  if (threadIdx.x < half) {
    const float2 f = freqs[(size_t)r.gpos * half + threadIdx.x];
    bf16* base = row + (Di - rd) + 2 * threadIdx.x;
    const float a = bf2f(base[0]), b = bf2f(base[1]);
    base[0] = f2bf(a * f.x - b * f.y);
    base[1] = f2bf(a * f.y + b * f.x);
  }
  __syncthreads();
  if (threadIdx.x < Di / 32) {  // fp4_pack_kernel<32, false>, row_bytes = IDX_ROW
    const int b = threadIdx.x;
    uint8_t* dst = r.dstk + (size_t)r.dsti * kvp::IDX_ROW;
    float v[32];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { v[i] = bf2f(row[b * 32 + i]); amax = fmaxf(amax, fabsf(v[i])); }
    amax = fmaxf(amax, 6.f * ldexpf(1.f, -126));
    const uint8_t code = f32_ceil_pow2_e8m0(amax * kFp4MaxInv);
    const float sc = e8m0_to_f32(code);
    dst[b] = code;
    uint8_t* nib = dst + (kvp::IDX_ROW - Di / 2) + (b * 32) / 2;
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
      const uint8_t lo = f32_to_e2m1(fminf(fmaxf(v[i] / sc, -6.f), 6.f)), hi = f32_to_e2m1(fminf(fmaxf(v[i + 1] / sc, -6.f), 6.f));
      nib[i / 2] = (uint8_t)(lo | (hi << 4));
    }
  }
}

// ---- ⑤ indexer scores (fixed Hi = 32, Di = 128): same arithmetic as idx_score_one (per head fmaf over ascending d → bf16 → relu → ×w (bf16) → fp32 sum over ascending h → bf16).
//   Changes: with a constant head count the head-loop branch (32 per dimension) disappears, q is transposed to [d][h] and read as 8 float4 per dimension,
//   and the key row (80 B) is read as 4 uint4 + 4 scales instead of byte by byte per dimension. The weight scale (scale_bf16) is applied in place by dec_qb block 0 (same formula).
constexpr int IDX_T = 128, IHI = 32, IDI = 128;
__device__ __forceinline__ float idx_score32(const float* __restrict__ qT, const float* __restrict__ wsm, const uint8_t* __restrict__ kr) {
  float acc[IHI];
#pragma unroll
  for (int h = 0; h < IHI; ++h) acc[h] = 0.f;
  // 32 dimensions = one scale block = 16 B of nibbles (one uint4). Dimension d = 32·qd + 8·w + j is bits 4·j of word w (same as nibble (d&1) of byte d>>1)
  for (int qd = 0; qd < IDI / 32; ++qd) {
    const float sc = e8m0_to_f32(kr[qd]);
    const uint4 u = reinterpret_cast<const uint4*>(kr + kvp::IDX_HDR)[qd];
    const uint32_t wv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t word = wv[w];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int d = qd * 32 + w * 8 + j;
        const float kv = kvp::e2m1_val((word >> (4 * j)) & 0xFu) * sc;
        const float4* qd4 = reinterpret_cast<const float4*>(qT + d * IHI);
#pragma unroll
        for (int jj = 0; jj < IHI / 4; ++jj) {
          const float4 qq = qd4[jj];
          acc[4 * jj] = fmaf(qq.x, kv, acc[4 * jj]);
          acc[4 * jj + 1] = fmaf(qq.y, kv, acc[4 * jj + 1]);
          acc[4 * jj + 2] = fmaf(qq.z, kv, acc[4 * jj + 2]);
          acc[4 * jj + 3] = fmaf(qq.w, kv, acc[4 * jj + 3]);
        }
      }
    }
  }
  float total = 0.f;
#pragma unroll
  for (int h = 0; h < IHI; ++h) {
    float sb = bf2f(f2bf(acc[h]));
    sb = fmaxf(sb, 0.f);
    sb = bf2f(f2bf(sb * wsm[h]));
    total += sb;
  }
  return total;
}
template <bool CAND>
__global__ void __launch_bounds__(IDX_T) dec_idx_scores_kernel(const bf16* __restrict__ q, const bf16* __restrict__ w, const DecRow* __restrict__ tab,
                                                               int ncols, const int32_t* __restrict__ cand, int cand_stride, int bs, bf16* __restrict__ score) {
  __shared__ __align__(16) float qT[IDI * IHI];
  __shared__ float wsm[IHI];
  const int m = blockIdx.y;
  for (int i = threadIdx.x; i < IHI * IDI; i += IDX_T) { const int h = i / IDI, d = i % IDI; qT[d * IHI + h] = bf2f(q[(size_t)m * IHI * IDI + i]); }
  if (threadIdx.x < IHI) wsm[threadIdx.x] = bf2f(w[(size_t)m * IHI + threadIdx.x]);  // w = value after scale_bf16 (applied in place by dec_qb block 0)
  __syncthreads();
  const int ci = blockIdx.x * IDX_T + threadIdx.x;
  if (ci >= ncols) return;
  const DecRow& r = tab[m];
  const int vis = min(r.visible, r.trows);
  int t = ci;
  if (CAND) {
    const int32_t b = cand[(size_t)m * cand_stride + ci / bs];
    t = b >= 0 ? b * bs + ci % bs : -1;
  }
  score[(size_t)m * ncols + ci] = (t >= 0 && t < vis) ? f2bf(idx_score32(qT, wsm, r.kptr + (size_t)t * kvp::IDX_ROW)) : f2bf(-INFINITY);
}

// Same formula as block_max_kernel (visible from the device copy)
__global__ void dec_block_max_kernel(const bf16* __restrict__ score, int M, int T, int bs, const DecRow* __restrict__ tab, float* __restrict__ bmax) {
  const int nblocks = (T + bs - 1) / bs;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * nblocks) return;
  int m = i / nblocks, b = i % nblocks;
  float mx = -INFINITY;
  for (int j = b * bs; j < min(T, (b + 1) * bs); ++j) mx = fmaxf(mx, bf2f(score[(size_t)m * T + j]));
  int last = (tab[m].visible - 1) / bs;
  if (b == last) mx = INFINITY;
  bmax[i] = mx;
}

// ---- ⑥ per-row top-k (block = row, 1024 threads). Same result as topk_select_kernel:
//   selection = top keff = min(k, T) by (sort key descending, index ascending) → drop those whose key is -inf → index ascending → rest -1.
//   topk_select_kernel: 4 histogram passes over 8-bit digits → collect those above the threshold (atomic order) → ties with the threshold picked by thread 0 from the smallest index (worst case O(want·neq) serial) → bitonic sort.
//   Here: the same histogram yields the threshold (thr) and the count above it (count_gt); each thread owns a contiguous range and two block prefix sums
//   (tie rank → selection rank) give the output slot directly — no sort, no serial tie loop. For bf16 keys the low 16 bits of the sort key are fixed by the sign (0x0000 / 0xFFFF), so two 16-bit passes give the same result.
//   Output mapping (optional): with cand, cand_to_pos (ci → block·position); with map_offset, offset_idxs((p ≥ 0 && p < visible) ? p + offset : -1).
constexpr int TPK = 1024;
__device__ __forceinline__ uint32_t ord_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ uint32_t ord_key(bf16 v) {
  const uint32_t u = (uint32_t)__bfloat16_as_ushort(v);
  return ((u & 0x8000u) ? ~u : (u | 0x8000u)) & 0xFFFFu;  // == ord_key(float(v)) >> 16
}
__device__ __forceinline__ bool is_neg_inf(float f) { return f == -INFINITY; }
__device__ __forceinline__ bool is_neg_inf(bf16 v) { return bf2f(v) == -INFINITY; }
// Block exclusive prefix sum (1024 threads). Returns the sum before this thread; *total = the full sum
__device__ __forceinline__ int block_excl_scan(int v, int* wsum, int* total) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int x = hive::cu::warp_inclusive_sum(v);
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (warp == 0) wsum[lane] = hive::cu::warp_inclusive_sum(wsum[lane]);  // inclusive prefix
  __syncthreads();
  const int before = (warp ? wsum[warp - 1] : 0) + x - v;
  *total = wsum[31];
  __syncthreads();
  return before;
}
template <class KeyT, int NBITS>
__global__ void __launch_bounds__(TPK) dec_topk_kernel(const KeyT* __restrict__ keys, int T_in, bool use_trows, const DecRow* __restrict__ tab, int k,
                                                       int row_stride, int32_t* __restrict__ out, int out_stride, const int32_t* __restrict__ cand,
                                                       int cand_stride, int bs, int offset, bool map_offset) {
  __shared__ int hist[32][256];
  __shared__ uint32_t prefix;
  __shared__ int need, cgt;
  __shared__ int wsum[32];
  const int row = blockIdx.x;
  const int T = use_trows ? min(tab[row].trows, T_in) : T_in;
  const int keff = min(k, T);
  const KeyT* kr = keys + (size_t)row * row_stride;
  const int tid = threadIdx.x, warp = tid >> 5;
  if (tid == 0) { prefix = 0; need = keff - 1; cgt = 0; }
  __syncthreads();
  constexpr uint32_t FULL = NBITS == 32 ? 0xFFFFFFFFu : 0xFFFFu;
  for (int pass = 0; pass < NBITS / 8; ++pass) {
    const int shift = NBITS - 8 - pass * 8;
    const uint32_t mask = pass == 0 ? 0u : ((FULL << (NBITS - pass * 8)) & FULL);
    for (int i = tid; i < 32 * 256; i += TPK) (&hist[0][0])[i] = 0;
    __syncthreads();
    const uint32_t pf = prefix;
    for (int t = tid; t < T; t += TPK) {
      const uint32_t u = ord_key(kr[t]);
      if ((u & mask) == pf) atomicAdd(&hist[warp][(u >> shift) & 0xFF], 1);
    }
    __syncthreads();
    for (int b = tid; b < 256; b += TPK) {
      int sum = 0;
      for (int w = 0; w < 32; ++w) sum += hist[w][b];
      hist[0][b] = sum;
    }
    __syncthreads();
    if (tid == 0) {
      int acc = 0, bin = 0;
      for (int b = 255; b >= 0; --b) {
        if (acc + hist[0][b] > need) { bin = b; break; }
        acc += hist[0][b];
      }
      need -= acc;
      cgt += acc;
      prefix |= ((uint32_t)bin) << shift;
    }
    __syncthreads();
  }
  const uint32_t thr = prefix;
  const int want = keff - cgt;  // number of threshold ties to take from the front (≥ 1; 0 when T == 0)
  // thread = contiguous range [t0, t1)
  const int C = (T + TPK - 1) / TPK;
  const int t0 = min(T, tid * C), t1 = min(T, t0 + C);
  int neq = 0;
  for (int t = t0; t < t1; ++t) neq += ord_key(kr[t]) == thr;
  int tot;
  const int eq_base = block_excl_scan(neq, wsum, &tot);
  int nsel = 0;
  {
    int r = eq_base;
    for (int t = t0; t < t1; ++t) {
      const KeyT v = kr[t];
      const uint32_t u = ord_key(v);
      bool s = u > thr;
      if (u == thr) { s = r < want; ++r; }
      nsel += (s && !is_neg_inf(v));
    }
  }
  int nall;
  int pos = block_excl_scan(nsel, wsum, &nall);
  int32_t* orow = out + (size_t)row * out_stride;
  auto emit = [&](int j, int32_t p) {
    if (cand && p >= 0) { const int32_t b = cand[(size_t)row * cand_stride + p / bs]; p = b >= 0 ? b * bs + p % bs : -1; }
    if (map_offset) p = (p >= 0 && p < tab[row].visible) ? p + offset : -1;
    orow[j] = p;
  };
  {
    int r = eq_base;
    for (int t = t0; t < t1; ++t) {
      const KeyT v = kr[t];
      const uint32_t u = ord_key(v);
      bool s = u > thr;
      if (u == thr) { s = r < want; ++r; }
      if (s && !is_neg_inf(v)) emit(pos++, t);
    }
  }
  for (int j = nall + tid; j < k; j += TPK) emit(j, -1);
}

// ---- ⑦ sparse attention (same arithmetic as attn_tc.cu attn_decode_rows_kernel + attn_decode_merge_kernel). Block = (row m, head h, split s).
//   Changes: row table from the device copy · 8 keys per warp in the score phase (8 loads issued first) · 16 P·V loads issued first (same accumulation order — fmaf over ascending t) ·
//   the split merge runs the same merge code in the last split block of (m, h) (saves one launch). With S == 1 the result is stored directly.
constexpr int AD = 512;
constexpr int DEC_THREADS = 256;
__device__ __forceinline__ uint32_t pack2bf(float a, float b) {
  return (uint32_t)__bfloat16_as_ushort(f2bf(a)) | ((uint32_t)__bfloat16_as_ushort(f2bf(b)) << 16);
}
__device__ __forceinline__ void store_o_pair(bf16* __restrict__ orow_base, int d0, float v0, float v1, int D, int rd, const float2* __restrict__ rope_freqs,
                                             int pos) {
  if (rope_freqs && d0 >= D - rd) {
    const int half = rd / 2, p = (d0 - (D - rd)) >> 1;
    float2 f = rope_freqs[(size_t)pos * half + p];
    f.y = -f.y;
    const float a = bf2f(f2bf(v0)), b = bf2f(f2bf(v1));
    orow_base[d0] = f2bf(a * f.x - b * f.y);
    orow_base[d0 + 1] = f2bf(a * f.y + b * f.x);
  } else {
    orow_base[d0] = f2bf(v0);
    orow_base[d0 + 1] = f2bf(v1);
  }
}
__global__ void __launch_bounds__(DEC_THREADS, 2) dec_attn_kernel(const bf16* __restrict__ q, int H, const DecRow* __restrict__ tab, const bf16* __restrict__ chunk,
                                                               int chunk_len, int win, const int32_t* __restrict__ idx, int idx_stride,
                                                               const float* __restrict__ sink, float scale, int S, bf16* __restrict__ o,
                                                               float* __restrict__ pacc, float* __restrict__ pm, float* __restrict__ ps,
                                                               const float2* __restrict__ rope_freqs, int rd, int* __restrict__ cnt) {
  extern __shared__ float sdyn[];
  __shared__ float qs[AD];
  __shared__ float red[DEC_THREADS / 32];
  __shared__ float stat[2];
  __shared__ float scs[16];
  __shared__ int flag;
  const int s = blockIdx.x % S;
  const int mh = blockIdx.x / S;
  const int h = mh % H, m = mh / H;
  const KvRow row = tab[m].kv;
  const int rpos = tab[m].pos;
  const int ncols = win + row.topk;
  const int chunk_n = (ncols + S - 1) / S;
  const int tb = s * chunk_n, te = min(ncols, tb + chunk_n), nloc = te > tb ? te - tb : 0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int32_t* irow = idx + (size_t)m * idx_stride + tb;
  const bf16* qh = q + ((size_t)m * H + h) * AD;
  const int ncomp0 = win + chunk_len;
  auto krow_bf16 = [&](int ix) -> const bf16* { return ix < win ? row.ring + (size_t)ix * AD : chunk + (size_t)(ix - win) * AD; };
  auto load_key16 = [&](int ix, uint4& ra, uint4& rb) {
    if (ix >= ncomp0) {
      float v[16];
      kvp::comp_block16(row.comp + (size_t)(ix - ncomp0) * kvp::COMP_ROW, lane, v);
      ra = make_uint4(pack2bf(v[0], v[1]), pack2bf(v[2], v[3]), pack2bf(v[4], v[5]), pack2bf(v[6], v[7]));
      rb = make_uint4(pack2bf(v[8], v[9]), pack2bf(v[10], v[11]), pack2bf(v[12], v[13]), pack2bf(v[14], v[15]));
    } else {
      const uint4* kp = reinterpret_cast<const uint4*>(krow_bf16(ix)) + lane * 2;
      ra = kp[0]; rb = kp[1];
    }
  };
  auto load_pair = [&](int ix, int d0) -> uint32_t {
    if (ix >= ncomp0) { const float2 p = kvp::comp_pair(row.comp + (size_t)(ix - ncomp0) * kvp::COMP_ROW, d0); return pack2bf(p.x, p.y); }
    return *reinterpret_cast<const uint32_t*>(krow_bf16(ix) + d0);
  };
  for (int i = tid; i < AD; i += DEC_THREADS) qs[i] = bf2f(qh[i]);
  __syncthreads();
  float qv[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) qv[i] = qs[lane * 16 + i];
  auto dot16 = [&](const uint4 a, const uint4 b) -> float {
    const bf16* ka = reinterpret_cast<const bf16*>(&a);
    const bf16* kb = reinterpret_cast<const bf16*>(&b);
    float d = fmaf(qv[0], bf2f(ka[0]), 0.f);
#pragma unroll
    for (int i = 1; i < 8; ++i) d = fmaf(qv[i], bf2f(ka[i]), d);
#pragma unroll
    for (int i = 0; i < 8; ++i) d = fmaf(qv[8 + i], bf2f(kb[i]), d);
    return d;
  };
  constexpr int NW = DEC_THREADS / 32;
  constexpr int KPW = 4;  // keys per warp (loads first — 4 as in attn_tc.cu: 8 needs 192 registers → 1 block per SM, and the 256 blocks of M=1 split into two waves)
  for (int t0 = warp * KPW; t0 < nloc; t0 += NW * KPW) {
    int ixs[KPW];
    uint4 ra[KPW], rb[KPW];
#pragma unroll
    for (int u = 0; u < KPW; ++u) {
      const int t = t0 + u;
      ixs[u] = (t < nloc) ? irow[t] : -1;
      if (ixs[u] >= 0) load_key16(ixs[u], ra[u], rb[u]);
      else { ra[u] = make_uint4(0, 0, 0, 0); rb[u] = ra[u]; }
    }
#pragma unroll
    for (int u = 0; u < KPW; ++u) {
      float d = dot16(ra[u], rb[u]);
      d = hive::cu::warp_allreduce(d);
      const int t = t0 + u;
      if (lane == 0 && t < nloc) sdyn[t] = (ixs[u] >= 0) ? d * scale : -INFINITY;
    }
  }
  __syncthreads();
  float mx = -INFINITY;
  for (int t = tid; t < nloc; t += DEC_THREADS) mx = fmaxf(mx, sdyn[t]);
  mx = hive::cu::warp_allreduce<hive::cu::Max>(mx);
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  if (tid == 0) { float v = red[0]; for (int i = 1; i < DEC_THREADS / 32; ++i) v = fmaxf(v, red[i]); stat[0] = v; }
  __syncthreads();
  mx = stat[0];
  const bool empty = !(mx > -INFINITY);
  float su = 0.f;
  if (!empty) for (int t = tid; t < nloc; t += DEC_THREADS) su += expf(sdyn[t] - mx);
  su = hive::cu::warp_allreduce(su);
  __syncthreads();
  if (lane == 0) red[warp] = su;
  __syncthreads();
  if (tid == 0) { float v = 0.f; for (int i = 0; i < DEC_THREADS / 32; ++i) v += red[i]; stat[1] = v; }
  if (!empty) for (int t = tid; t < nloc; t += DEC_THREADS) sdyn[t] = bf2f(f2bf(expf(sdyn[t] - mx)));
  __syncthreads();
  float acc0 = 0.f, acc1 = 0.f;
  const int d0 = tid * 2;
  int t = 0;
  if (!empty) {
    // attn_tc.cu: groups of 8 + remainder (skipping p == 0 / ix < 0). Terms with (p == 0 or ix < 0) inside a group add a +0 product — acc starts at +0, so it
    // can never become −0, and adding ±0 keeps value and sign → changing the group size gives the same result bits.
    constexpr int PV = 16;
    for (; t + PV <= nloc; t += PV) {
      uint32_t raw[PV];
      float pv[PV];
#pragma unroll
      for (int u = 0; u < PV; ++u) {
        pv[u] = sdyn[t + u];
        const int ix = irow[t + u];
        raw[u] = (ix >= 0 && pv[u] != 0.f) ? load_pair(ix, d0) : 0u;
      }
#pragma unroll
      for (int u = 0; u < PV; ++u) {
        acc0 = fmaf(pv[u], __uint_as_float(raw[u] << 16), acc0);
        acc1 = fmaf(pv[u], __uint_as_float(raw[u] & 0xFFFF0000u), acc1);
      }
    }
    for (; t < nloc; ++t) {
      const float p = sdyn[t];
      const int ix = irow[t];
      if (p == 0.f || ix < 0) continue;
      const uint32_t raw = load_pair(ix, d0);
      acc0 = fmaf(p, __uint_as_float(raw << 16), acc0);
      acc1 = fmaf(p, __uint_as_float(raw & 0xFFFF0000u), acc1);
    }
  }
  bf16* orow = o + ((size_t)m * H + h) * AD;
  if (S == 1) {
    const float denom = stat[1] + expf(sink[h] - mx);
    store_o_pair(orow, d0, acc0 / denom, acc1 / denom, AD, rd, rope_freqs, rpos);
    return;
  }
  const size_t pi = ((size_t)m * H + h) * S + s;
  float* pa = pacc + pi * AD + d0;
  pa[0] = acc0; pa[1] = acc1;
  if (tid == 0) { pm[pi] = mx; ps[pi] = stat[1]; }
  if (!last_of(cnt + mh, S, &flag)) return;
  // same merge as attn_decode_merge_kernel (thread 0 statistics, 2 dimensions per thread, fmaf over ascending s)
  const size_t base = (size_t)mh * S;
  if (tid == 0) {
    float M_g = -INFINITY;
    for (int ss = 0; ss < S; ++ss) M_g = fmaxf(M_g, __ldcg(pm + base + ss));
    float sum = 0.f;
    for (int ss = 0; ss < S; ++ss) {
      const float pmv = __ldcg(pm + base + ss);
      const float f = pmv > -INFINITY ? expf(pmv - M_g) : 0.f;
      scs[ss] = f;
      sum += f * __ldcg(ps + base + ss);
    }
    stat[0] = M_g; stat[1] = sum + expf(sink[h] - M_g);
  }
  __syncthreads();
  float a0 = 0.f, a1 = 0.f;
  for (int ss = 0; ss < S; ++ss) {
    const float* pp = pacc + (base + ss) * AD + d0;
    a0 = fmaf(scs[ss], __ldcg(pp), a0); a1 = fmaf(scs[ss], __ldcg(pp + 1), a1);
  }
  store_o_pair(orow, d0, a0 / stat[1], a1 / stat[1], AD, rd, rope_freqs, rpos);
}

__global__ void dec_stage_rows_kernel(DecRowSrc src, int M, DecRow* __restrict__ tab, DecSoA* __restrict__ soa) {
  if ((int)threadIdx.x < M) stage_row(src, threadIdx.x, tab, soa);
}

inline size_t smem_bytes(int M, int K) { return (size_t)M * K + (size_t)M * (K / 32) * 4 + (size_t)M * (K / 32) + 16; }
template <class Kern>
void ensure_smem(Kern kern, size_t bytes, size_t& configured) {
  if (bytes + 4096 > 48 * 1024 && bytes > configured) {
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(bytes + 4096)));
    configured = bytes;
  }
}
inline void launch_check(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) return;
  throw std::runtime_error(std::string("dec fused launch failed: ").append(what).append(": ").append(cudaGetErrorString(e)));
}
inline int grid1d(size_t n, int block = 256) { return (int)((n + block - 1) / block); }

template <class KeyT, int NBITS>
void topk_launch(const KeyT* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                 const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, cudaStream_t st) {
  HIVE_CHECK(k >= 1 && k <= 2048 && M >= 1, "dec_topk k");
  HIVE_CHECK(!(use_trows || map_offset) || tab, "dec_topk tab");
  dec_topk_kernel<KeyT, NBITS><<<M, TPK, 0, st>>>(keys, T_in, use_trows, tab, k, row_stride, out, out_stride, cand, cand_stride, bs, offset, map_offset);
  launch_check("topk");
}

}  // namespace

void dec_topk_bf16(const bf16* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                   const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, cudaStream_t st) {
  topk_launch<bf16, 16>(keys, M, T_in, use_trows, tab, k, row_stride, out, out_stride, cand, cand_stride, bs, offset, map_offset, st);
}
void dec_topk_f32(const float* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                  const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, cudaStream_t st) {
  topk_launch<float, 32>(keys, M, T_in, use_trows, tab, k, row_stride, out, out_stride, cand, cand_stride, bs, offset, map_offset, st);
}
void dec_stage_rows(const DecRowSrc& src, int M, DecRow* tab, DecSoA* soa, cudaStream_t st) {
  dec_stage_rows_kernel<<<1, 32, 0, st>>>(src, M, tab, soa);
  launch_check("stage");
}

// ④ indexer (the former dec_attention_front body moved verbatim — called only when idx_on is true). New-version selection (hive/decode_longctx.h):
//   bit 1 = HIVE_DECODE_IDXSCORE2 (scores; the candidate-pool tc version (pack_q) is unchanged) · bit 2 = HIVE_DECODE_TOPK2 (all three top-k). When a new version declines (false), the original launches run.
//   The first M slots of a.attn_cnt ([M·H] int, 0) are borrowed as row counters for the split top-k — the kernel resets them to 0, and the sparse attention later on the same stream uses them afterwards.
static bool indexer_stage(const DecAttnArgs& a, cudaStream_t st) {
  const int M = a.M;
  const bool cand_src = (a.layer == a.cand_source_layer);
  const bool uses_cand = (a.cand_source_layer >= 0 && a.cand_source_layer < a.layer);
  const bool pack_q = uses_cand && a.idx_mode == 1;  // pack_dec (HIVE_IDX_TC, candidate-pool layers)
  const int lm = a.lctx >= 0 ? a.lctx : ((decode_idxscore2_on() ? 1 : 0) | (decode_topk2_on() ? 2 : 0));
  const bool sc2 = (lm & 1) != 0, tk2 = (lm & 2) != 0;
  bool produced = false;
  {
    const int bs = a.cand_block, nblocks = (a.Tmax + bs - 1) / bs, CB = a.cand_topk_blocks;
    const int topk = std::min(a.index_topk, a.Tmax);
    const int off = a.win + M;
    if (uses_cand) {
      const int kbc = std::min(CB, nblocks);
      HIVE_CHECK(a.have_candidates && kbc > 0, "candidates missing");
      const int ncand = kbc * bs;
      if (pack_q)
      {  // same kernel as before; only the row arrays come from the device copy (DecSoA — a device pointer, so member addresses via offsetof)
        char* sb = reinterpret_cast<char*>(a.soa);
        indexer_scores_cand_tc(a.iqp, nullptr, reinterpret_cast<const uint8_t* const*>(sb + offsetof(DecSoA, kptr)),
                               reinterpret_cast<const int32_t*>(sb + offsetof(DecSoA, trows)), a.iw, M, a.Hi, a.cand, CB, kbc, bs,
                               reinterpret_cast<const int32_t*>(sb + offsetof(DecSoA, visible)), a.iscore, st);
      }
      else if (!(sc2 && dec_idx_scores2(a.iq, a.iw, a.tab, M, ncand, true, a.cand, CB, bs, a.iscore, 0, st)))
        dec_idx_scores_kernel<true><<<dim3((ncand + IDX_T - 1) / IDX_T, M), IDX_T, 0, st>>>(a.iq, a.iw, a.tab, ncand, a.cand, CB, bs, a.iscore);
      launch_check("scores_cand");
      if (!(tk2 && dec_topk2_bf16(a.iscore, M, ncand, false, a.tab, topk, ncand, a.idx + a.win, a.idx_stride, a.cand, CB, bs, off, true, a.scratch,
                                  a.scratch_bytes, a.attn_cnt, st)))
      topk_launch<bf16, 16>(a.iscore, M, ncand, false, a.tab, topk, ncand, a.idx + a.win, a.idx_stride, a.cand, CB, bs, off, true, st);
    } else {
      if (!(sc2 && dec_idx_scores2(a.iq, a.iw, a.tab, M, a.Tmax, false, nullptr, 0, 1, a.iscore, 0, st)))
      dec_idx_scores_kernel<false><<<dim3((a.Tmax + IDX_T - 1) / IDX_T, M), IDX_T, 0, st>>>(a.iq, a.iw, a.tab, a.Tmax, nullptr, 0, 1, a.iscore);
      launch_check("scores");
      if (cand_src) {
        dec_block_max_kernel<<<grid1d((size_t)M * nblocks), 256, 0, st>>>(a.iscore, M, a.Tmax, bs, a.tab, a.bmax);
        launch_check("block_max");
        const int kb = std::min(CB, nblocks);
        if (!(tk2 && dec_topk2_f32(a.bmax, M, nblocks, false, a.tab, kb, nblocks, a.cand, CB, nullptr, 0, 1, 0, false, a.scratch, a.scratch_bytes, a.attn_cnt,
                                   st)))
        topk_launch<float, 32>(a.bmax, M, nblocks, false, a.tab, kb, nblocks, a.cand, CB, nullptr, 0, 1, 0, false, st);  // → cand directly (instead of topk_pos + memcpy2D)
        produced = true;
      }
      if (!(tk2 && dec_topk2_bf16(a.iscore, M, a.Tmax, true, a.tab, topk, a.Tmax, a.idx + a.win, a.idx_stride, nullptr, 0, 1, off, true, a.scratch,
                                  a.scratch_bytes, a.attn_cnt, st)))
      topk_launch<bf16, 16>(a.iscore, M, a.Tmax, true, a.tab, topk, a.Tmax, a.idx + a.win, a.idx_stride, nullptr, 0, 1, off, true, st);
    }
  }
  return produced;
}

bool dec_attention_front(const DecAttnArgs& a, Blas* blas, cudaStream_t st) {
  const int M = a.M, D = a.D, dim = a.dim, HD = a.H * a.D;
  HIVE_CHECK(M >= 1 && M <= MMAX && dim % 32 == 0 && a.q_lora % OUT == 0 && D % OUT == 0 && D == AD && a.rd % 2 == 0 && a.rd / 2 <= FT &&
                 (D - a.rd) % OUT == 0 && HD % OUT == 0 && a.splits >= 1 && a.splits <= 16 && a.tab && a.soa && a.counters && a.attn_cnt,
             "dec_attention_front shape");
  const bool idx_on = a.ratio && a.index_source && a.Tmax > 0;
  if (idx_on) HIVE_CHECK(a.Hi == IHI && a.Di == IDI && a.Di % OUT_I == 0 && (a.Di - a.rd) % OUT_I == 0 && a.Di == kvp::IDX_D, "dec indexer shape (Hi 32 · Di 128)");
  const bool uses_cand = idx_on && (a.cand_source_layer >= 0 && a.cand_source_layer < a.layer);
  const bool pack_q = uses_cand && a.idx_mode == 1;  // pack_dec (HIVE_IDX_TC, candidate-pool layers)
  if (pack_q) HIVE_CHECK(a.iqp, "dec idx tc: iqp");
  const bool xf_on = a.ratio > 1 && a.kv_source;
  // v3 bit 1 = new q_a‖kv version (HIVE_DECODE_QKV3) · bit 2 = new sparse-attention version (HIVE_DECODE_SPARSE3) — both bit-identical (test_decode_attn3).
  const int v3 = a.v3 >= 0 ? a.v3 : ((decode_qkv3_on() ? 1 : 0) | (decode_sparse3_on() ? 2 : 0));
  // ① q_a ‖ kv (+ fp32 copy of xn · row table · window indices · ring write)
  static size_t cfg1 = 0, cfg2 = 0;
  const size_t s1 = smem_bytes(M, dim), s2 = smem_bytes(M, a.q_lora);
  ensure_smem(dec_qkv_a_kernel, s1, cfg1);
  ensure_smem(dec_qb_kernel, s2, cfg2);
  if (!((v3 & 1) && attn3_qkv_a(a, xf_on, st))) {  // if the new version does not accept the shape, fall back to this kernel (absorbed)
  dec_qkv_a_kernel<<<a.q_lora / OUT + D / OUT, FT, s1, st>>>(a.xn, dim, a.wqa, a.sqa, a.q_lora, a.q_norm, a.wkv, a.skv, D, a.kv_norm, a.eps, a.rd, a.freqs, M,
                                                             a.qr, a.qrn, a.kv, a.vgrp ? nullptr : a.ring_ptrs, a.win, a.idx, a.idx_stride, xf_on ? a.xf : nullptr,
                                                             a.src, a.tab, a.soa, a.counters, a.vgrp);
  launch_check("qkv_a");
  }
  // indexer weight projection (cuBLAS — same call as the unfused path). The scale (scale_bf16) is applied in place by block 0 of the following dec_qb.
  if (idx_on) blas->gemm_bf16(a.xn, a.idx_wproj, a.iw, M, a.Hi, dim);
  // ② q_b (+RoPE) ‖ indexer query (+RoPE · fp4 round trip · [tc] packing) · iw scale
  const int NI = idx_on ? a.Hi * a.Di : 0;
  const float wscale = (1.0f / sqrtf((float)a.Di)) * (1.0f / sqrtf((float)a.Hi));
  // HIVE_DECODE_ATTN2: the q_b columns (+RoPE, side outputs qrq/qrs) use the decode_attn2.cu version (bit-identical, positions = row-table pos). Only the indexer query blocks
  //   and the iw scale (index layers) run this kernel with N = 0 (all blocks are indexer blocks, no side_q) — if that version declines the shape, the original launch runs.
  static_assert(sizeof(DecRow) % 4 == 0, "DecRow pos stride");
  if (decode_attn2_on() && attn2_qb(a.qrn, a.q_lora, a.wqb, a.sqb, M, HD, D, a.rd, a.freqs,
                                    reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(a.tab) + offsetof(DecRow, pos)), (int)(sizeof(DecRow) / 4), a.q,
                                    a.qrq, a.qrs, st)) {
    if (NI > 0)
      dec_qb_kernel<<<NI / OUT_I, FT, s2, st>>>(a.qrn, a.q_lora, a.wqb, a.sqb, M, 0, D, a.rd, a.freqs, a.tab, a.q, nullptr, nullptr, a.idx_wqb, a.idx_sqb, NI,
                                               a.Di, a.freqs_idx, a.iq, idx_on ? a.iw : nullptr, M * a.Hi, wscale, pack_q ? a.iqp : nullptr);
  } else
  dec_qb_kernel<<<HD / OUT + NI / OUT_I, FT, s2, st>>>(a.qrn, a.q_lora, a.wqb, a.sqb, M, HD, D, a.rd, a.freqs, a.tab, a.q, a.qrq, a.qrs, a.idx_wqb, a.idx_sqb,
                                                        NI, a.Di, a.freqs_idx, a.iq, idx_on ? a.iw : nullptr, M * a.Hi, wscale, pack_q ? a.iqp : nullptr);
  launch_check("qb");
  // ③ compressor (kv source layers) — cuBLAS calls same as the unfused path
  if (a.ratio && a.kv_source) {
    const bool vstep = a.vgrp && a.ratio > 1;  // verify mode: rows of the same sequence continue the same state → row-ordered compressor (same launch as attention_verify_dev ②)
    if (a.ratio > 1) {
      blas->gemm_f32(a.xf, static_cast<const float*>(a.comp_wkv), a.ckv, M, D, dim);
      blas->gemm_f32(a.xf, a.comp_wgate, a.cscore, M, D, dim);
      if (vstep && a.vsave) {  // keep the inputs for rollback (kv ‖ score, row capacity vsave_rows) — same layout and location as attention_verify_dev
        CUDA_CHECK(cudaMemcpyAsync(a.vsave, a.ckv, (size_t)M * D * 4, cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(a.vsave + a.vsave_rows * D, a.cscore, (size_t)M * D * 4, cudaMemcpyDeviceToDevice, st));
      }
      if (vstep) compressor_step_seq(a.ckv, a.cscore, M, D, a.ratio, a.src.pos, a.src.skv, a.src.ssc, a.cout, a.valid, st);
    } else {
      blas->gemm_bf16(a.xn, static_cast<const bf16*>(a.comp_wkv), a.latent, M, D, dim);
    }
    dec_compress_kernel<<<M, 256, 0, st>>>(a.ratio, D, a.ckv, a.cscore, a.cout, a.latent, a.comp_norm, a.eps, a.rd, a.freqs, a.tab, a.valid, vstep);
    launch_check("compress");
    if (a.index_source) {
      blas->gemm_bf16(a.latent, a.idx_wk, a.ik, M, a.Di, D);
      dec_ik_kernel<<<M, 256, 0, st>>>(a.Di, a.ik, a.idx_k_norm, a.eps, a.rd, a.freqs, a.tab, a.valid);
      launch_check("ik");
    }
  }
  // ④ indexer — the body lives in indexer_stage (above) (same statements and launch order; with the switches off the original launches run)
  const bool produced = idx_on && indexer_stage(a, st);
  // ⑤ sparse attention (+inverse RoPE, split merge included)
  const int ncols_max = a.win + (a.ratio ? a.index_topk : 0);
  const int S = a.splits;
  const size_t smem = (size_t)((ncols_max + S - 1) / S) * 4 + 16;
  if (!((v3 & 2) && attn3_sparse(a, a.v3_g, st))) {  // if the new version does not accept the shape, fall back to this kernel (absorbed)
  dec_attn_kernel<<<M * a.H * S, DEC_THREADS, smem, st>>>(a.q, a.H, a.tab, a.kv, M, a.win, a.idx, a.idx_stride, a.sink, 1.0f / sqrtf((float)D), S, a.o, a.pacc,
                                                          a.pm, a.ps, a.freqs, a.rd, a.attn_cnt);
  launch_check("attn");
  }
  // ⑥ verify mode: the ring write comes **after** attention (same kernel as attention_verify_dev ③) — all rows have read their windows before a later row of the same sequence overwrites the oldest slot of an earlier row's window
  if (a.vgrp) {
    ring_write_rows(a.kv, M, D, a.win, a.src.pos, a.ring_ptrs, st);
    launch_check("verify ring");
  }
  return produced;
}

// For tests (test_decode_longctx): stage ④ only — same shape checks and same function as dec_attention_front
bool dec_indexer_stage(const DecAttnArgs& a, cudaStream_t st) {
  const bool idx_on = a.ratio && a.index_source && a.Tmax > 0;
  if (!idx_on) return false;
  HIVE_CHECK(a.M >= 1 && a.M <= MMAX && a.tab, "dec_indexer_stage shape");
  HIVE_CHECK(a.Hi == IHI && a.Di == IDI && a.Di == kvp::IDX_D, "dec indexer shape (Hi 32 · Di 128)");
  if ((a.cand_source_layer >= 0 && a.cand_source_layer < a.layer) && a.idx_mode == 1) HIVE_CHECK(a.iqp, "dec idx tc: iqp");
  return indexer_stage(a, st);
}

void dec_idx_scores_ref(const bf16* q, const bf16* w, const DecRow* tab, int M, int ncols, bool cand_mode, const int32_t* cand, int cand_stride, int bs,
                        bf16* score, cudaStream_t st) {
  if (cand_mode) dec_idx_scores_kernel<true><<<dim3((ncols + IDX_T - 1) / IDX_T, M), IDX_T, 0, st>>>(q, w, tab, ncols, cand, cand_stride, bs, score);
  else dec_idx_scores_kernel<false><<<dim3((ncols + IDX_T - 1) / IDX_T, M), IDX_T, 0, st>>>(q, w, tab, ncols, nullptr, 0, 1, score);
  launch_check("scores(ref)");
}

// For tests (test_decode_attn3): run one stage in the original or new version — launch arguments use the same formulas as dec_attention_front above
bool dec_front_stage(const DecAttnArgs& a, int stage, int variant, cudaStream_t st) {
  const int M = a.M, D = a.D, dim = a.dim;
  if (stage == 0) {
    const bool xf_on = a.ratio > 1 && a.kv_source;
    if (variant >= 0) return attn3_qkv_a(a, xf_on, st);
    static size_t cfg1 = 0;
    const size_t s1 = smem_bytes(M, dim);
    ensure_smem(dec_qkv_a_kernel, s1, cfg1);
    dec_qkv_a_kernel<<<a.q_lora / OUT + D / OUT, FT, s1, st>>>(a.xn, dim, a.wqa, a.sqa, a.q_lora, a.q_norm, a.wkv, a.skv, D, a.kv_norm, a.eps, a.rd, a.freqs, M,
                                                               a.qr, a.qrn, a.kv, a.vgrp ? nullptr : a.ring_ptrs, a.win, a.idx, a.idx_stride,
                                                               xf_on ? a.xf : nullptr, a.src, a.tab, a.soa, a.counters, a.vgrp);
    launch_check("qkv_a(stage)");
    return true;
  }
  if (variant >= 0) return attn3_sparse(a, variant, st);
  const int ncols_max = a.win + (a.ratio ? a.index_topk : 0);
  const int S = a.splits;
  const size_t smem = (size_t)((ncols_max + S - 1) / S) * 4 + 16;
  dec_attn_kernel<<<M * a.H * S, DEC_THREADS, smem, st>>>(a.q, a.H, a.tab, a.kv, M, a.win, a.idx, a.idx_stride, a.sink, 1.0f / sqrtf((float)D), S, a.o, a.pacc,
                                                          a.pm, a.ps, a.freqs, a.rd, a.attn_cnt);
  launch_check("attn(stage)");
  return true;
}

}  // namespace hive::k
