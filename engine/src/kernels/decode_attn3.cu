// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// New versions of the four remaining decode (M ≤ 8) attention-side kernels (router · sparse attention · q_a‖kv · hc mix). Contracts and switches: header comment of include/hive/decode_attn3.h.
//
// Background (nsys, serving build with the ATTN2 kernels on and the fused front end, warm cache, last 128 decode steps at M=1, per step → per layer = /40):
//   a2_router 0.91ms → 22.8µs (gate_w 7.9MB → 0.35 TB/s — ATTN2 expected ~8µs) · dec_attn 2.43 → 61µs · dec_qkv_a 1.13 → 28µs · hc_mix_pre_norm 0.78 (2× per layer) → 9.8µs each.
//   head_logits 0.80ms reads head 1.32GB (129280×5120 bf16) at 1.65 TB/s — bandwidth-bound, left alone. wo_a 0.93ms (1.47 TB/s) also unchanged.
//
// ① router — gap between the ATTN2 estimate (8µs) and the measurement (22.8µs) (ptxas: a2_router 40 registers, smem 20KB → 4 blocks per SM → 384 blocks in one wave; occupancy is not the cause):
//   (a) the xn chain of the dot product: warp m loads xn only one 16-element group ahead — computing a group (16 fmaf) is ~100 cycles while an L2 round trip is several hundred, so K=5120 = 10 groups
//       means 10 serialized L2 round trips (`xa = xb` at the end of a group waits on the load just issued). The second group only starts **after** the whole gate row has arrived.
//   (b) tail (the single last block): row m is handled by warp m alone — softplus/sqrt (log1pf, expf, sqrtf, ~50 instructions) in a non-unrolled loop of 12 per lane ·
//       every one of the 6 top-k rounds does an smem scan (loop, compare chain) + 10 shuffles + lane-0 smem/global writes + __syncwarp · normalization re-reads w from global.
//       Arithmetic estimate: tail ≈ 3–5µs + (a) ≈ 2–4µs + atomic join ~0.5µs, serialized after the weight drain (7.9MB/1.8TB/s ≈ 4.4µs + latency).
//   Fix: for M=1 the gate row and xn go to smem in the same cp.async (30KB per block → 3 blocks per SM, still 384 blocks in one wave) → the dot chain reads only smem.
//     For M ≥ 2 smem would break the single wave (40KB → 2 blocks per SM = 376 < 384), so xn uses a 3-stage register ring (two groups ahead, rotating three buffers without moves).
//     Tail: softplus/sqrt/bias element-wise over the block's 256 threads (same formula) → smem · top-k: each warp holds 16 values per lane in registers, register scan + shuffle per round
//     (selection rule as in ATTN2: ascending `>` within a lane → xor shuffle (larger value, on a tie the smaller index)) · weight sum and normalization in lane-0 registers/smem (no global re-read).
//   Expected (arithmetic): drain ~5µs + dot ~0.3µs + join ~0.5µs + tail ~1µs ≈ 7–8µs per layer (22.8 → ~8) ⇒ −0.55–0.6ms per step. The GPU test measures phases 1/2 (load only, load + dot) separately.
//
// ② sparse attention — arithmetic per layer (M=1, layers 2–39: columns = window 128 + compressed top 512 = 640; the same for 4K/32K/128K context — context only grows the indexer (scores, top-k)):
//   unique bytes = window 128 × 1KB (bf16) + compressed 512 × 288B (fp4+e4m3) ≈ 275KB (at 128K context, 147KB scattered over a 37MB compressed cache — only first touch hits DRAM) ·
//   FLOP = 64 heads × 640 columns × (score 512 + P·V 512) fma ≈ 42M fma (≈ 0.1 TFLOP/s class — negligible).
//   What the 61µs really is = **instruction count** (duplicated key decoding): the previous 256 blocks (row, head, split) each decode the same keys **separately per head**, and P·V decodes one
//   dimension pair per thread, recomputing the e4m3 scale (ldexpf path) per pair. Instruction estimate (per thread and key): score ~250 (16 nibble decodes, bf16 round trip, 16 fmaf) · P·V ~65
//   → ≈ 40K + 83K warp instructions per block, at most 2 blocks per SM = 246K / (4 IPC) ≈ 62K cycles ≈ 25µs (issue-bound) + load latency (index → key address, 2-level dependent chain) ⇒ ~60µs.
//   The split count 4 (merge order = bits) is kept. New version: block = (row, G heads, split) · a tile of 16 keys is decoded once by the block's 256 threads (16-dim group = one thread,
//   one scale) into smem (bf16), and both the scores of G heads (warp = one head, lane = 16 dims — same chain as the previous dot16, xor 16..1) and P·V (thread = dim pair × G heads,
//   fmaf in ascending t) read it · the index split goes to smem once at the start · the next tile's loads are issued before computing the current tile. Softmax, merge and output formulas are unchanged per head.
//   Instructions (per block, decode 2 passes ≈ 51K + score 7K·G + P·V (3.8+3.8·G)K warp instructions): G=2 ≈ 77K (38K per head — 1/3 of the previous 123K) · G=8 ≈ 141K (18K per head).
//   Expected (arithmetic): M=1 G=2 (128 blocks) 61 → ~12–15µs per layer · M=8 G=8 (256 blocks) 1/7 of the instructions per head. Automatic G satisfies M·H·S/G ≤ SM count — the test measures G 1/2/4/8.
//
// ③ q_a‖kv — previous version: 224 blocks (i) first quantize the input (K/32 = 160 threads, 32 elements each, serially) and only **then** load weights · (ii) each of the 5 warp_dot iterations
//   does its own DRAM round trip · (iii) the tail of the last kv block reads the mapped pinned row table (PCIe round trip) → window indices → serial per-row rmsnorm (4 syncs) → RoPE → fp8 round trip
//   (16 threads, 32 elements each, serial divisions) → ring. At M=8 the tail is 8×. New version (same grid, same chains): 3 weight blocks are issued into registers before quantization (the other 2 in the loop,
//   3 slots ahead) · quantization on 256 threads (same split and formula as a2_quant_rows) · row table, window indices, pos and ring pointers handled up front by the kv block · tail: row = warp
//   (emulates the 256 virtual threads of rmsnorm_row_256 as 32 lanes × 8 — same partial sums, same xor tree) · RoPE pairs via lane xor 1 · fp8 block (32 elements) via warp max.
//   Expected (arithmetic): 28 → ~12–15µs per layer (M=1) · at M=8 the serialized 8-row tail → one row's time.
//
// ④ hc_mix_pre_norm — confirmed in PTX: the mix dot loop is unrolled by 16 (80 iterations → 5 round trips), but the x loop of the pre+norm block (one block per row) is **not unrolled**: 20 iterations × 4 loads
//   = 20 serialized L2 round trips — this block is the tail of the launch. New version (same grid (M, 26), same thread → element mapping and reductions): mix/sum of squares = 2 × 40 · pre+norm =
//   80 h + 20 norm_w loaded all at once before the sync; x is kept in registers to write xn (previously it re-read the x just written — same values).
//   Expected (arithmetic): 9.8 → ~3–4µs per launch × 80 ⇒ −0.5ms per step.
#include <cuda_fp8.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hive/attn_decode_fused.h"
#include "hive/decode_attn2.h"
#include "hive/decode_attn3.h"
#include "hive/kernels.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

namespace hive::k {

namespace {

// switch test = same rule as runtime.cpp env_on (unset, "", "0" = off)
inline bool a3_env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }

constexpr float kFp8MaxInv3 = 1.0f / 448.0f;

// ---- device helpers (same formulas as the existing files — bit-identity is the contract, so the formulas are not changed) ------------------------------------------------------
__device__ __forceinline__ float2 a3_cvt_e4m3x2(uint32_t pair16) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)pair16));
  return __half22float2(*reinterpret_cast<__half2*>(&h2));
}
__device__ __forceinline__ float a3_e8m0_fast(uint8_t b) { return b ? __uint_as_float((uint32_t)b << 23) : __uint_as_float(0x00400000u); }
__device__ __forceinline__ void a3_decode8(const uint4 r0, const uint4 r1, float (&w)[32]) {
  const uint32_t x[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 lo = a3_cvt_e4m3x2(x[i] & 0xFFFFu), hi = a3_cvt_e4m3x2(x[i] >> 16);
    w[4 * i] = lo.x; w[4 * i + 1] = lo.y; w[4 * i + 2] = hi.x; w[4 * i + 3] = hi.y;
  }
}
// cp.async — the most conservative form, as in decode_attn2.cu and the F2 kernels (no hint, no prefetch: the cache_hint variant is an illegal instruction on sm_120a, measured)
__device__ __forceinline__ uint32_t a3_su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void a3_cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a3_su32(s)), "l"(g) : "memory"); }
__device__ __forceinline__ void a3_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void a3_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
// same as fused.cu last_block / attn_decode_fused.cu last_of (the last block resets the counter to 0 — safe for graph replay)
__device__ __forceinline__ bool a3_last_of(int* counter, int n, int* flag_smem) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) *flag_smem = (atomicAdd(counter, 1) == n - 1);
  __syncthreads();
  const bool last = *flag_smem != 0;
  if (last) { __threadfence(); if (threadIdx.x == 0) *counter = 0; }
  return last;
}

// ================================================================================================================================
// ① router
// ================================================================================================================================
constexpr int R3T = 256;  // block = one expert · row m = warp m
constexpr int R3U = 16;   // xn group of the register version (16 per lane)
constexpr int R3EV = 16;  // top-k register version: 16 values per lane (E ≤ 512)
constexpr int R3KMAX = 16;
// XS: xn in smem (M·K·2 bytes) — the caller has verified a single wave (all E blocks resident). phase: 0 full · 1 load only · 2 load + dot
template <bool XS>
__global__ void __launch_bounds__(R3T, 3) a3_router_kernel(const bf16* __restrict__ xn, int K, const float* __restrict__ W, int M, int E,
                                                            const float* __restrict__ bias, const float* __restrict__ bias_vl,
                                                            const int8_t* __restrict__ is_image, int k, float route_scale, float* __restrict__ scores,
                                                            int32_t* __restrict__ ids, float* __restrict__ w, int* __restrict__ counter, int phase,
                                                            CachePriorArgs cp) {
  extern __shared__ __align__(16) uint8_t r3_sm[];  // [K] f32 gate row | XS: [M][K] bf16 xn · reused in the tail as s[M][E] · sl[M][E] · wsel[M][16]
  __shared__ int flag;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, e = blockIdx.x;
  float* gs = reinterpret_cast<float*>(r3_sm);
  const float* We = W + (size_t)e * K;
  for (int c = threadIdx.x; c < K / 4; c += R3T) a3_cp16(gs + 4 * c, We + 4 * c);
  bf16* xs = reinterpret_cast<bf16*>(r3_sm + (size_t)K * 4);
  if constexpr (XS) {
    for (int c = threadIdx.x; c < M * K / 8; c += R3T) a3_cp16(xs + 8 * c, xn + 8 * c);
  }
  a3_commit();
  const bool act = warp < M;
  float d = 0.f;  // previous: d = fmaf(bf2f(xm[kk]), We[kk], d), kk = lane, lane+32, … ascending → xor 16..1
  if constexpr (XS) {
    a3_wait<0>();
    __syncthreads();
    if (phase == 1) return;
    if (act) {
      const bf16* xm = xs + (size_t)warp * K;
      int kk = lane;
      for (; kk + 32 * 7 < K; kk += 32 * 8) {  // issue 8 smem load pairs first (the chain is fmaf only)
        float xv[8], gv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) { xv[u] = bf2f(xm[kk + 32 * u]); gv[u] = gs[kk + 32 * u]; }
#pragma unroll
        for (int u = 0; u < 8; ++u) d = fmaf(xv[u], gv[u], d);
      }
      for (; kk < K; kk += 32) d = fmaf(bf2f(xm[kk]), gs[kk], d);
    }
  } else {
    // 3-stage xn register ring: the loads for group c are issued before computing c−2. The three buffers (xa, xb, xc) rotate with no register moves
    //   (the previous `xa = xb` waited on the load it had just issued).
    const bf16* xm = xn + (size_t)(act ? warp : 0) * K;
    const int nch = (K + 32 * R3U - 1) / (32 * R3U);
    float xa[R3U], xb[R3U], xc[R3U];
    auto ld = [&](float (&x)[R3U], int c) {
#pragma unroll
      for (int u = 0; u < R3U; ++u) { const int kk = c * 32 * R3U + lane + 32 * u; x[u] = (act && kk < K) ? bf2f(xm[kk]) : 0.f; }
    };
    auto dot = [&](const float (&x)[R3U], int c) {
      if (c >= nch) return;
#pragma unroll
      for (int u = 0; u < R3U; ++u) { const int kk = c * 32 * R3U + lane + 32 * u; if (kk < K) d = fmaf(x[u], gs[kk], d); }
    };
    ld(xa, 0);
    ld(xb, 1);
    a3_wait<0>();
    __syncthreads();
    if (phase == 1) return;
    if (act) {
      for (int c = 0; c < nch; c += 3) {
        ld(xc, c + 2); dot(xa, c);
        ld(xa, c + 3); dot(xb, c + 1);
        ld(xb, c + 4); dot(xc, c + 2);
      }
    }
  }
  if (act) {
    d = hive::cu::warp_allreduce(d);
    if (lane == 0) scores[(size_t)warp * E + e] = d;
  }
  if (phase == 2) return;
  if (!a3_last_of(counter, (int)gridDim.x, &flag)) return;
  // ---- tail (last block) ----
  float* s = reinterpret_cast<float*>(r3_sm);
  float* sl = s + (size_t)M * E;
  float* wsel = sl + (size_t)M * E;
  for (int i = threadIdx.x; i < M * E; i += R3T) {  // previous formula unchanged (element-wise — only spread over 256 threads)
    const int m = i / E, ee = i - m * E;
    const float* b = (bias_vl && is_image && is_image[m]) ? bias_vl : bias;
    const float x = __ldcg(scores + i);
    const float sp = x > 20.f ? x : log1pf(expf(x));
    s[i] = sqrtf(sp);
    sl[i] = s[i] + b[ee];
  }
  __syncthreads();
  // Cache prior (HIVE_CACHE_PRIOR — CachePriorArgs in decode_attn3.h): selection scores sl of VRAM-resident experts and of the row's top-J
  //   get + λ·Δavg (Δavg = running average of the per-row range of sl in this layer); the weights still come from s
  if (cp.mask && cp.lambda > 0.f) {
    __shared__ float rng[8];
    __shared__ float bonus;
    __shared__ int topj[8][4];
    if (warp < M) {
      float mx = -FLT_MAX, mn = FLT_MAX;
      for (int ee = lane; ee < E; ee += 32) { const float v = sl[(size_t)warp * E + ee]; mx = fmaxf(mx, v); mn = fminf(mn, v); }
      for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o)); }
      if (lane == 0) rng[warp] = mx - mn;
      for (int j = 0; j < cp.top_j; ++j) {
        float bv = -FLT_MAX; int bi = E;
        for (int ee = lane; ee < E; ee += 32) {
          bool taken = false;
          for (int q = 0; q < j; ++q) taken |= topj[warp][q] == ee;
          const float v = sl[(size_t)warp * E + ee];
          if (!taken && (v > bv || (v == bv && ee < bi))) { bv = v; bi = ee; }
        }
        for (int o = 16; o > 0; o >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
          if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) topj[warp][j] = bi;
        __syncwarp();
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float r = 0.f;
      for (int q = 0; q < M; ++q) r += rng[q];
      r /= M;
      const float a0 = *cp.range_avg;
      const float a = a0 > 0.f ? 0.99f * a0 + 0.01f * r : r;
      *cp.range_avg = a;
      bonus = cp.lambda * a;
    }
    __syncthreads();
    if (warp < M)
      for (int ee = lane; ee < E; ee += 32) {
        bool boost = cp.mask[ee] != 0;
        for (int q = 0; q < cp.top_j; ++q) boost |= topj[warp][q] == ee;
        if (boost) sl[(size_t)warp * E + ee] += bonus;
      }
    __syncthreads();
  }
  if (warp < M) {
    const int m = warp;
    const float* sm = s + (size_t)m * E;
    float v[R3EV];
#pragma unroll
    for (int u = 0; u < R3EV; ++u) { const int ee = lane + 32 * u; v[u] = ee < E ? sl[(size_t)m * E + ee] : -FLT_MAX; }  // -FLT_MAX is never picked (`>`)
    float wsum = 0.f;
#pragma unroll 1
    for (int j = 0; j < k; ++j) {
      float bv = -FLT_MAX;
      int bi = -1;
#pragma unroll
      for (int u = 0; u < R3EV; ++u) if (v[u] > bv) { bv = v[u]; bi = lane + 32 * u; }  // first maximum in ascending order within a lane (same as ATTN2)
#pragma unroll
      for (int o = 16; o >= 1; o /= 2) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
      }
      {  // clear the picked slot (ATTN2: sl[bi] = -FLT_MAX) — select expression on every slot (a dynamically indexed store would push v[] to local memory: confirmed with ptxas)
        const int kill = (bi >= 0 && (bi & 31) == lane) ? (bi >> 5) : -1;
#pragma unroll
        for (int u = 0; u < R3EV; ++u) v[u] = (u == kill) ? -FLT_MAX : v[u];
      }
      if (lane == 0) {
        const float sv = bi >= 0 ? sm[bi] : 0.f;
        ids[m * k + j] = bi;
        wsel[m * R3KMAX + j] = sv;
        wsum += sv;
      }
    }
    if (lane == 0) for (int j = 0; j < k; ++j) w[m * k + j] = wsel[m * R3KMAX + j] / (wsum + 1e-20f) * route_scale;
  }
}

// ================================================================================================================================
// ④ hc_mix_pre_norm (same formulas as model_kernels.cu hc_mix_body / hc_pre_norm_body — only the load timing differs)
// ================================================================================================================================
constexpr int H3T = 256;
constexpr int H3U = 40;  // mix / sum of squares: load group per thread
template <int NK>        // NK = hcdim / 256 (elements per thread)
__device__ __forceinline__ void h3_mix_body(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps,
                                            float* __restrict__ mixes, float* __restrict__ rsq, int m, int j, float* red) {
  const bf16* p = h + (size_t)m * hcdim;
  float acc = 0.f;
  if (j < mix_hc) {
    const float* wr = W + (size_t)j * hcdim;
#pragma unroll
    for (int b0 = 0; b0 < NK; b0 += H3U) {
      float hv[H3U], wv[H3U];
#pragma unroll
      for (int u = 0; u < H3U; ++u) {
        if (b0 + u < NK) { const int i = threadIdx.x + 256 * (b0 + u); hv[u] = bf2f(p[i]); wv[u] = wr[i]; }
      }
#pragma unroll
      for (int u = 0; u < H3U; ++u) if (b0 + u < NK) acc = fmaf(hv[u], wv[u], acc);
    }
  } else {
#pragma unroll
    for (int b0 = 0; b0 < NK; b0 += H3U) {
      float hv[H3U];
#pragma unroll
      for (int u = 0; u < H3U; ++u) if (b0 + u < NK) hv[u] = bf2f(p[threadIdx.x + 256 * (b0 + u)]);
#pragma unroll
      for (int u = 0; u < H3U; ++u) {
        if (b0 + u < NK) { float v = hv[u]; acc += v * v; }  // previous formula unchanged (changing it could change bits)
      }
    }
  }
  acc = hive::cu::warp_allreduce(acc);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += red[i];
    if (j < mix_hc) mixes[(size_t)m * mix_hc + j] = t;
    else rsq[m] = rsqrtf(t / (float)hcdim + eps);
  }
}
template <int HC, int ND>  // ND = dim / 256
__device__ __forceinline__ void h3_pre_norm_body(const bf16* __restrict__ h, const float* __restrict__ pre_in, int dim, const bf16* __restrict__ norm_w,
                                                 float norm_eps, bf16* __restrict__ x, bf16* __restrict__ xn, int m, float* red, float* pin) {
  constexpr int hc = HC;
  const bf16* hm = h + (size_t)m * hc * dim;
  bf16 hv[ND][HC];
  bf16 nw[ND];
#pragma unroll
  for (int k = 0; k < ND; ++k) {  // issue everything before the sync (independent of pin)
    const int d = threadIdx.x + 256 * k;
#pragma unroll
    for (int c = 0; c < hc; ++c) hv[k][c] = hm[(size_t)c * dim + d];
    nw[k] = norm_w[d];
  }
  if (threadIdx.x < hc) pin[threadIdx.x] = pre_in[m * hc + threadIdx.x];
  __syncthreads();
  float ss = 0.f;
  float xv[ND];
#pragma unroll
  for (int k = 0; k < ND; ++k) {
    const int d = threadIdx.x + 256 * k;
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < hc; ++c) s += pin[c] * bf2f(hv[k][c]);
    const bf16 xb = f2bf(s);
    x[(size_t)m * dim + d] = xb;
    const float v = bf2f(xb);
    xv[k] = v;
    ss += v * v;
  }
  ss = hive::cu::warp_allreduce(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    t = hive::cu::warp_allreduce(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / (float)dim + norm_eps);
#pragma unroll
  for (int k = 0; k < ND; ++k) {  // previous: re-read v = bf2f(x[·]) — same value as the xb this thread just wrote
    const int d = threadIdx.x + 256 * k;
    xn[(size_t)m * dim + d] = f2bf(bf2f(nw[k]) * (xv[k] * rs));
  }
}
template <int HC, int NK, int ND>
__global__ void __launch_bounds__(H3T, 2) hc_mix_pre_norm3_kernel(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps,
                                                                  float* __restrict__ mixes, float* __restrict__ rsq, const float* __restrict__ pre_in, int dim,
                                                                  const bf16* __restrict__ norm_w, float norm_eps, bf16* __restrict__ x, bf16* __restrict__ xn) {
  __shared__ float red[8];
  __shared__ float pin[HC];
  if ((int)blockIdx.y <= mix_hc) h3_mix_body<NK>(h, W, hcdim, mix_hc, eps, mixes, rsq, blockIdx.x, blockIdx.y, red);
  else h3_pre_norm_body<HC, ND>(h, pre_in, dim, norm_w, norm_eps, x, xn, blockIdx.x, red, pin);
}

// ================================================================================================================================
// ③ q_a‖kv (same grid and chains as attn_decode_fused.cu dec_qkv_a_kernel)
// ================================================================================================================================
constexpr int Q3T = 256;
constexpr int Q3OUT = 8;  // block = 8 columns (1 column per warp)
constexpr int Q3M = 8;
constexpr int Q3PF = 3;   // weight blocks issued before quantization (32B per lane + scale)
// Same values as quant_rows_smem (aq e4m3 · asv = e8m0_to_f32(code) · asc = code) — only the split differs: one 32-element block = 4 lanes (8 elements, 16B loads), amax via xor 1·2
//   (fmaxf — order-independent, NaN dropped: same result as sequential fmaxf). Same split as decode_attn2.cu a2_quant_rows, and also writes asv.
__device__ __forceinline__ void q3_quant_rows(const bf16* __restrict__ A, int M, int K, uint8_t* aq, float* asv, uint8_t* asc) {
  const int nb = K / 32, total = M * nb, q = threadIdx.x & 3;
  constexpr int R = 4;
  for (int i0 = 0; i0 < total; i0 += R * (Q3T / 4)) {
    uint4 raw[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int idx = i0 + r * (Q3T / 4) + (int)(threadIdx.x >> 2);
      if (idx < total) {
        const int m = idx / nb, b = idx - m * nb;
        raw[r] = *reinterpret_cast<const uint4*>(A + (size_t)m * K + (size_t)b * 32 + q * 8);
      } else {
        raw[r] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int idx = i0 + r * (Q3T / 4) + (int)(threadIdx.x >> 2);
      const bool ok = idx < total;
      const uint32_t xw[4] = {raw[r].x, raw[r].y, raw[r].z, raw[r].w};
      float v[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) { v[2 * i] = __uint_as_float(xw[i] << 16); v[2 * i + 1] = __uint_as_float(xw[i] & 0xFFFF0000u); }  // = bf2f(A[·])
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
      if (ok) {
        const int m = idx / nb, b = idx - m * nb;
        amax = fmaxf(amax, 1e-4f);
        const uint8_t code = f32_ceil_pow2_e8m0(amax * kFp8MaxInv3);
        const float sc = e8m0_to_f32(code);
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) lo |= (uint32_t)f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.f), 448.f)) << (8 * i);
#pragma unroll
        for (int i = 0; i < 4; ++i) hi |= (uint32_t)f32_to_e4m3(fminf(fmaxf(v[4 + i] / sc, -448.f), 448.f)) << (8 * i);
        *reinterpret_cast<uint2*>(aq + (size_t)m * K + (size_t)b * 32 + q * 8) = make_uint2(lo, hi);
        if (q == 0) { asc[idx] = code; asv[idx] = sc; }  // idx = m·nb + b
      }
    }
  }
}
// rmsnorm_row_256 (x → out, N = 256·NQ) on one warp: lane l plays virtual threads l + 32·vw (vw < 8) — each virtual thread accumulates elements i = vt + 256·kq
//   in ascending kq as `ss += v * v` (previous formula) · xor 16..1 of virtual warp vw = real lane xor · red[vw] → for lanes < 8, t = red[lane] → xor 16..1 → rs.
//   Returns: normalized values (bf16) in rv[j], j = vw + 8·kq ↔ element i = lane + 32·j.
template <int NQ>
__device__ __forceinline__ void q3_warp_rmsnorm(const bf16* x, const bf16* __restrict__ w, float eps, int lane, bf16 (&rv)[8 * NQ]) {
  constexpr int N = 256 * NQ;
  float xv[8 * NQ];
#pragma unroll
  for (int j = 0; j < 8 * NQ; ++j) xv[j] = bf2f(__ldcg(x + lane + 32 * j));  // written by other blocks (after the fence of the last-block test)
  float redv[8];
#pragma unroll
  for (int vw = 0; vw < 8; ++vw) {
    float ss = 0.f;
#pragma unroll
    for (int kq = 0; kq < NQ; ++kq) { float v = xv[vw + 8 * kq]; ss += v * v; }
    ss = hive::cu::warp_allreduce(ss);
    redv[vw] = ss;
  }
  float t = 0.f;
#pragma unroll
  for (int vw = 0; vw < 8; ++vw) if (lane == vw) t = redv[vw];
  t = hive::cu::warp_allreduce(t);
  const float rs = rsqrtf(t / (float)N + eps);
#pragma unroll
  for (int j = 0; j < 8 * NQ; ++j) rv[j] = f2bf(bf2f(w[lane + 32 * j]) * (xv[j] * rs));
}
__device__ __forceinline__ void q3_stage_row(const DecRowSrc& s, int m, DecRow* tab, DecSoA* soa) {  // same as attn_decode_fused.cu stage_row
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
// NBL = (K/32)/32 blocks per lane (K = 5120 → 5) · NQ = Nqa/256 (1280 → 5) · ND = D/256 (512 → 2)
template <int NBL, int NQ, int ND>
__global__ void __launch_bounds__(Q3T, 2) dec_qkv3_kernel(const bf16* __restrict__ xn, int K, const uint8_t* __restrict__ wqa, const uint8_t* __restrict__ sqa,
                                                          int Nqa, const bf16* __restrict__ q_norm, const uint8_t* __restrict__ wkv,
                                                          const uint8_t* __restrict__ skv, int D, const bf16* __restrict__ kv_norm, float eps, int rd,
                                                          const float2* __restrict__ freqs, int M, bf16* __restrict__ qr, bf16* __restrict__ qrn,
                                                          bf16* __restrict__ kv, bf16* const* __restrict__ ring_ptrs, int win, int32_t* __restrict__ idx_out,
                                                          int idx_stride, float* __restrict__ xf, DecRowSrc src, DecRow* __restrict__ tab,
                                                          DecSoA* __restrict__ soa, int* __restrict__ counters, const int32_t* __restrict__ vgrp) {
  extern __shared__ __align__(16) uint8_t q3_sm[];
  uint8_t* aq = q3_sm;
  float* asv = reinterpret_cast<float*>(q3_sm + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (K / 32));
  __shared__ int flag;
  __shared__ int spos[Q3M];
  __shared__ int sgrp[Q3M];  // verify version (HIVE_MTP_VERIFY2_FUSED): first row of the part that row m belongs to (vgrp — mapped pinned)
  __shared__ bf16* sring[Q3M];
  const int nqb = Nqa / Q3OUT;
  const bool isq = (int)blockIdx.x < nqb;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = (isq ? blockIdx.x : blockIdx.x - nqb) * Q3OUT + warp;
  const uint8_t* B = isq ? wqa : wkv;
  const uint8_t* sb = isq ? sqa : skv;
  const int N = isq ? Nqa : D;
  const int nb = K / 32;
  // (1) the first Q3PF weight blocks (lane l: block l + 32t) — independent of the previous kernel's output, so issued before quantization
  uint4 wr0[Q3PF], wr1[Q3PF];
  uint8_t wsc[Q3PF];
#pragma unroll
  for (int t = 0; t < Q3PF; ++t) {
    const int b = lane + 32 * t;
    if (t < NBL && b < nb) {
      const uint8_t* pb = B + (size_t)n * K + (size_t)b * 32;
      wr0[t] = *reinterpret_cast<const uint4*>(pb); wr1[t] = *reinterpret_cast<const uint4*>(pb + 16);
      wsc[t] = sb[(size_t)n * nb + b];
    } else {
      wr0[t] = make_uint4(0u, 0u, 0u, 0u); wr1[t] = wr0[t]; wsc[t] = 0;
    }
  }
  // (2) kv block: reads the mapped pinned row table (PCIe) up front — previously read in the tail of the last kv block (same values). The first kv block also writes the device copies (tab, soa)
  //     (no consumer inside this kernel — later kernels read them).
  if (!isq && threadIdx.x < M) {
    const int m = threadIdx.x;
    spos[m] = src.pos[m];
    sgrp[m] = vgrp ? vgrp[m] : m;
    sring[m] = ring_ptrs ? ring_ptrs[m] : nullptr;
    if ((int)blockIdx.x == nqb) q3_stage_row(src, m, tab, soa);
  }
  // (3) input quantization (256 threads) + fp32 copy of xn (same split as before)
  q3_quant_rows(xn, M, K, aq, asv, asc);
  if (xf) for (int i = blockIdx.x * Q3T + threadIdx.x; i < M * K; i += gridDim.x * Q3T) xf[i] = bf2f(xn[i]);
  __syncthreads();
  if (!isq && idx_out && (int)blockIdx.x == nqb + (D / Q3OUT > 1 ? 1 : 0)) {  // same formula as window_idxs_rows (previously: in the last kv block)
    if (vgrp) {  // verify version: same formula as window_idxs_verify_kernel (before the part's first position = ring slot · inside the part = chunk row win + g + (src − p0))
      for (int i = threadIdx.x; i < M * win; i += Q3T) {
        const int m = i / win, j = i % win, g = sgrp[m];
        const int64_t p = spos[m], p0 = spos[g], srcp = p - (win - 1) + j;
        idx_out[(size_t)m * idx_stride + j] = srcp < 0 ? -1 : (srcp < p0 ? (int32_t)(srcp % win) : (int32_t)(win + g + (srcp - p0)));
      }
    } else
    for (int i = threadIdx.x; i < M * win; i += Q3T) {
      const int m = i / win, j = i % win;
      const int64_t p = spos[m], srcp = p - (win - 1) + j;
      idx_out[(size_t)m * idx_stride + j] = srcp < 0 ? -1 : (srcp < p ? (int32_t)(srcp % win) : (int32_t)(win + m));
    }
  }
  // (4) dot product — same chain as warp_dot (blocks b = lane + 32t ascending · d = Σ fmaf(a_i, w_i) · acc = fmaf(d, sa·sb, acc) · xor 16..1)
  float acc[Q3M];
#pragma unroll
  for (int m = 0; m < Q3M; ++m) acc[m] = 0.f;
#pragma unroll
  for (int t = 0; t < NBL; ++t) {
    const int slot = t % Q3PF;
    const uint4 r0 = wr0[slot], r1 = wr1[slot];
    const uint8_t scb = wsc[slot];
    if (t + Q3PF < NBL) {  // a slot is free, so issue the block Q3PF slots ahead
      const int b2 = lane + 32 * (t + Q3PF);
      if (b2 < nb) {
        const uint8_t* pb = B + (size_t)n * K + (size_t)b2 * 32;
        wr0[slot] = *reinterpret_cast<const uint4*>(pb); wr1[slot] = *reinterpret_cast<const uint4*>(pb + 16);
        wsc[slot] = sb[(size_t)n * nb + b2];
      }
    }
    const int b = lane + 32 * t;
    if (b < nb) {
      float wv[32];
      a3_decode8(r0, r1, wv);
      const float sbv = a3_e8m0_fast(scb);
#pragma unroll
      for (int m = 0; m < Q3M; ++m) {
        if (m < M) {
          const uint8_t* pa = aq + (size_t)m * K + (size_t)b * 32;
          float av[32];
          a3_decode8(*reinterpret_cast<const uint4*>(pa), *reinterpret_cast<const uint4*>(pa + 16), av);
          float d = 0.f;
#pragma unroll
          for (int i = 0; i < 32; ++i) d = fmaf(av[i], wv[i], d);
          acc[m] = fmaf(d, asv[m * nb + b] * sbv, acc[m]);
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < Q3M; ++m) {
    float v = acc[m];
    v = hive::cu::warp_allreduce(v);
    acc[m] = v;
  }
  bf16* C = isq ? qr : kv;
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < Q3M; ++m) if (m < M) C[(size_t)m * N + n] = f2bf(acc[m]);  // previous: outv → thread m·8+c writes f2bf(outv[m][c])
  }
  if (isq) {  // q tail: qrn = rmsnorm(qr, q_norm) — row = warp
    if (!a3_last_of(counters + 0, nqb, &flag)) return;
    if (warp < M) {
      bf16 rv[8 * NQ];
      q3_warp_rmsnorm<NQ>(qr + (size_t)warp * Nqa, q_norm, eps, lane, rv);
      bf16* out = qrn + (size_t)warp * Nqa;
#pragma unroll
      for (int j = 0; j < 8 * NQ; ++j) out[lane + 32 * j] = rv[j];
    }
    return;
  }
  // kv tail (same chain as the gemv_kv_kernel tail — rmsnorm → RoPE → fp8 round trip → kv row + ring) — row = warp
  if (!a3_last_of(counters + 1, D / Q3OUT, &flag)) return;
  if (warp >= M) return;
  const int m = warp;
  bf16* kvm = kv + (size_t)m * D;
  bf16 rv[8 * ND];
  q3_warp_rmsnorm<ND>(kvm, kv_norm, eps, lane, rv);
  // RoPE: pairs (2p, 2p+1) of elements i ∈ [D − rd, D) — since i = lane + 32j, the partner is lane xor 1 (same j). Previously: thread p took a = row[2p], b = row[2p+1]
  const int half = rd / 2, pos = spos[m];
#pragma unroll
  for (int j = 0; j < 8 * ND; ++j) {
    const int i = lane + 32 * j;
    const bf16 other = __shfl_xor_sync(0xffffffff, rv[j], 1);
    if (i >= D - rd) {
      const int p = (i - (D - rd)) >> 1;
      const float2 f = freqs[(size_t)pos * half + p];
      if ((lane & 1) == 0) { const float a = bf2f(rv[j]), b = bf2f(other); rv[j] = f2bf(a * f.x - b * f.y); }
      else { const float a = bf2f(other), b = bf2f(rv[j]); rv[j] = f2bf(a * f.y + b * f.x); }
    }
  }
  // fp8 round trip (block of 32 elements = the 32 lanes of one j): amax = fmaxf(|v|) from 0 → fmaxf(·, 1e-4) · sc = e8m0(ceil pow2(amax/448)) · element formula unchanged
  bf16* ring = sring[m] ? sring[m] + (size_t)(pos % win) * D : nullptr;
#pragma unroll
  for (int j = 0; j < 8 * ND; ++j) {
    const float v = bf2f(rv[j]);
    float amax = fmaxf(0.f, fabsf(v));
    amax = hive::cu::warp_allreduce<hive::cu::Max>(amax);
    amax = fmaxf(amax, 1e-4f);
    const float sc = e8m0_to_f32(f32_ceil_pow2_e8m0(amax * kFp8MaxInv3));
    const float t = fminf(fmaxf(v / sc, -448.f), 448.f);
    const bf16 ob = f2bf(e4m3_to_f32(f32_to_e4m3(t)) * sc);
    const int i = lane + 32 * j;
    kvm[i] = ob;
    if (ring) ring[i] = ob;
  }
}

// ================================================================================================================================
// ② sparse attention (same arithmetic as attn_decode_fused.cu dec_attn_kernel · block = (row m, G heads, split s))
// ================================================================================================================================
constexpr int S3T = 256;
constexpr int S3TK = 16;   // key tile (256 threads per block × 2 units = 16 keys × 32 groups of 16 dims)
constexpr int S3AD = 512;
__device__ __forceinline__ uint32_t s3_pack2bf(float a, float b) {
  return (uint32_t)__bfloat16_as_ushort(f2bf(a)) | ((uint32_t)__bfloat16_as_ushort(f2bf(b)) << 16);
}
__device__ __forceinline__ void s3_store_o_pair(bf16* __restrict__ orow_base, int d0, float v0, float v1, int D, int rd, const float2* __restrict__ rope_freqs,
                                                int pos) {  // same as attn_decode_fused.cu store_o_pair
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
struct S3Raw { uint4 a, b; };  // bf16 key: 32B · compressed key: a.x/a.y = 8B of nibbles, a.z = scale byte
template <int G>
__global__ void __launch_bounds__(S3T, 2) dec_attn3_kernel(const bf16* __restrict__ q, int H, const DecRow* __restrict__ tab, const bf16* __restrict__ chunk,
                                                           int chunk_len, int win, const int32_t* __restrict__ idx, int idx_stride,
                                                           const float* __restrict__ sink, float scale, int S, int cmax, bf16* __restrict__ o,
                                                           float* __restrict__ pacc, float* __restrict__ pm, float* __restrict__ ps,
                                                           const float2* __restrict__ rope_freqs, int rd, int* __restrict__ cnt) {
  extern __shared__ __align__(16) uint8_t s3_sm[];
  bf16* kt = reinterpret_cast<bf16*>(s3_sm);                                    // [2][S3TK][512] bf16 key tile (double-buffered)
  int* sidx = reinterpret_cast<int*>(s3_sm + (size_t)2 * S3TK * S3AD * 2);      // [cmax] column indices of this split
  float* sd = reinterpret_cast<float*>(sidx + cmax);                            // [G][cmax] scores → p
  __shared__ float red[G][8];
  __shared__ float stat[G][2];
  __shared__ float scs[G][16];
  __shared__ int flag;
  const int HG = H / G;
  const int s = blockIdx.x % S;
  const int hg = (blockIdx.x / S) % HG, m = blockIdx.x / (S * HG);
  const int h0 = hg * G;
  const KvRow row = tab[m].kv;
  const int rpos = tab[m].pos;
  const int ncols = win + row.topk;
  const int chunk_n = (ncols + S - 1) / S;
  const int tb = s * chunk_n, te = min(ncols, tb + chunk_n), nloc = te > tb ? te - tb : 0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int32_t* irow = idx + (size_t)m * idx_stride + tb;
  const int ncomp0 = win + chunk_len;
  for (int t = tid; t < nloc; t += S3T) sidx[t] = irow[t];
  // this warp's head (score phase): g = warp % G — qv = same values as the previous qs[lane·16 + i] = bf2f(q[·])
  const int gq = warp % G;
  float qv[16];
  {
    const uint4* qp = reinterpret_cast<const uint4*>(q + ((size_t)m * H + h0 + gq) * S3AD + lane * 16);
    const uint4 q0 = qp[0], q1 = qp[1];
    const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) { qv[2 * i] = __uint_as_float(qw[i] << 16); qv[2 * i + 1] = __uint_as_float(qw[i] & 0xFFFF0000u); }
  }
  __syncthreads();  // sidx
  const int ntile = (nloc + S3TK - 1) / S3TK;
  // unit u = tid + 256·rr → key j = warp + 8·rr, 16-dim group b = lane
  auto fetch = [&](int tile, S3Raw (&r)[2]) {
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int j = warp + 8 * rr, t = tile * S3TK + j;
      const int ix = t < nloc ? sidx[t] : -1;
      if (ix >= ncomp0) {
        const uint8_t* cr = row.comp + (size_t)(ix - ncomp0) * kvp::COMP_ROW;
        const uint2 nbv = *reinterpret_cast<const uint2*>(cr + kvp::COMP_SCALES + 8 * lane);
        r[rr].a = make_uint4(nbv.x, nbv.y, (uint32_t)cr[lane], 0u);
        r[rr].b = make_uint4(0u, 0u, 0u, 0u);
      } else if (ix >= 0) {
        const bf16* kr = ix < win ? row.ring + (size_t)ix * S3AD : chunk + (size_t)(ix - win) * S3AD;
        const uint4* kp = reinterpret_cast<const uint4*>(kr) + lane * 2;
        r[rr].a = kp[0]; r[rr].b = kp[1];
      } else {
        r[rr].a = make_uint4(0u, 0u, 0u, 0u); r[rr].b = r[rr].a;
      }
    }
  };
  auto store = [&](int buf, int tile, const S3Raw (&r)[2]) {
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int j = warp + 8 * rr, t = tile * S3TK + j;
      const int ix = t < nloc ? sidx[t] : -1;
      uint4 A = r[rr].a, Bv = r[rr].b;
      if (ix >= ncomp0) {  // same values as kvp::comp_block16 → pack2bf of the previous load_key16
        const float sc = e4m3_to_f32((uint8_t)r[rr].a.z);
        const uint32_t xw[2] = {r[rr].a.x, r[rr].a.y};
        float v[16];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int jj = 0; jj < 8; ++jj) v[8 * i + jj] = kvp::e2m1_val((xw[i] >> (4 * jj)) & 0xF) * sc;
        A = make_uint4(s3_pack2bf(v[0], v[1]), s3_pack2bf(v[2], v[3]), s3_pack2bf(v[4], v[5]), s3_pack2bf(v[6], v[7]));
        Bv = make_uint4(s3_pack2bf(v[8], v[9]), s3_pack2bf(v[10], v[11]), s3_pack2bf(v[12], v[13]), s3_pack2bf(v[14], v[15]));
      }
      uint4* dst = reinterpret_cast<uint4*>(kt + ((size_t)buf * S3TK + j) * S3AD) + lane * 2;
      dst[0] = A; dst[1] = Bv;
    }
  };
  auto dot16 = [&](const uint4 a, const uint4 b) -> float {  // same chain as the previous dot16
    const bf16* ka = reinterpret_cast<const bf16*>(&a);
    const bf16* kb = reinterpret_cast<const bf16*>(&b);
    float d = fmaf(qv[0], bf2f(ka[0]), 0.f);
#pragma unroll
    for (int i = 1; i < 8; ++i) d = fmaf(qv[i], bf2f(ka[i]), d);
#pragma unroll
    for (int i = 0; i < 8; ++i) d = fmaf(qv[8 + i], bf2f(kb[i]), d);
    return d;
  };
  S3Raw rr_[2];
  // ---- scores: per tile 16·G (head g, key j) pairs — warp w takes g = w % G, key j = w / G + (8/G)·r ----
  if (ntile > 0) fetch(0, rr_);
  for (int tile = 0; tile < ntile; ++tile) {
    const int buf = tile & 1;
    store(buf, tile, rr_);
    if (tile + 1 < ntile) fetch(tile + 1, rr_);  // the next tile's loads are in flight while this tile computes
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2 * G; ++r) {
      const int j = warp / G + (8 / G) * r, t = tile * S3TK + j;
      const uint4* kp = reinterpret_cast<const uint4*>(kt + ((size_t)buf * S3TK + j) * S3AD) + lane * 2;
      float d = dot16(kp[0], kp[1]);
      d = hive::cu::warp_allreduce(d);
      if (lane == 0 && t < nloc) sd[gq * cmax + t] = (sidx[t] >= 0) ? d * scale : -INFINITY;
    }
  }
  if (ntile > 0) fetch(0, rr_);  // first P·V tile — loads during the softmax
  __syncthreads();
  // ---- softmax (previous formula and reduction unchanged, per head) ----
  float mxg[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float mx = -INFINITY;
    for (int t = tid; t < nloc; t += S3T) mx = fmaxf(mx, sd[g * cmax + t]);
    mx = hive::cu::warp_allreduce<hive::cu::Max>(mx);
    if (lane == 0) red[g][warp] = mx;
  }
  __syncthreads();
  if (tid < G) { float v = red[tid][0]; for (int i = 1; i < S3T / 32; ++i) v = fmaxf(v, red[tid][i]); stat[tid][0] = v; }
  __syncthreads();
  bool emp[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float mx = stat[g][0];
    mxg[g] = mx;
    emp[g] = !(mx > -INFINITY);
    float su = 0.f;
    if (!emp[g]) for (int t = tid; t < nloc; t += S3T) su += expf(sd[g * cmax + t] - mx);
    su = hive::cu::warp_allreduce(su);
    if (lane == 0) red[g][warp] = su;
  }
  __syncthreads();
  if (tid < G) { float v = 0.f; for (int i = 0; i < S3T / 32; ++i) v += red[tid][i]; stat[tid][1] = v; }
#pragma unroll
  for (int g = 0; g < G; ++g)
    if (!emp[g]) for (int t = tid; t < nloc; t += S3T) sd[g * cmax + t] = bf2f(f2bf(expf(sd[g * cmax + t] - mxg[g])));
  __syncthreads();
  // ---- P·V: thread = dim pair d0 × G heads, fmaf in ascending t (previously (ix < 0 or p == 0) terms either added a +0 product or were skipped — acc starts at +0 and can never
  //      become −0, so this is bit-identical to skipping) ----
  float a0[G], a1[G];
#pragma unroll
  for (int g = 0; g < G; ++g) { a0[g] = 0.f; a1[g] = 0.f; }
  const int d0 = tid * 2;
  for (int tile = 0; tile < ntile; ++tile) {
    const int buf = tile & 1;
    store(buf, tile, rr_);
    if (tile + 1 < ntile) fetch(tile + 1, rr_);
    __syncthreads();
    const int tn = min(S3TK, nloc - tile * S3TK);
    for (int j = 0; j < tn; ++j) {
      const int t = tile * S3TK + j;
      const int ix = sidx[t];
      const uint32_t raw = *reinterpret_cast<const uint32_t*>(kt + ((size_t)buf * S3TK + j) * S3AD + d0);
      const float k0 = __uint_as_float(raw << 16), k1 = __uint_as_float(raw & 0xFFFF0000u);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        if (emp[g]) continue;
        const float p = sd[g * cmax + t];
        if (ix >= 0 && p != 0.f) { a0[g] = fmaf(p, k0, a0[g]); a1[g] = fmaf(p, k1, a1[g]); }
      }
    }
  }
  if (S == 1) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const int h = h0 + g;
      const float denom = stat[g][1] + expf(sink[h] - mxg[g]);
      s3_store_o_pair(o + ((size_t)m * H + h) * S3AD, d0, a0[g] / denom, a1[g] / denom, S3AD, rd, rope_freqs, rpos);
    }
    return;
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const size_t pi = ((size_t)m * H + h0 + g) * S + s;
    float* pa = pacc + pi * S3AD + d0;
    pa[0] = a0[g]; pa[1] = a1[g];
  }
  if (tid < G) { const size_t pi = ((size_t)m * H + h0 + tid) * S + s; pm[pi] = stat[tid][0]; ps[pi] = stat[tid][1]; }
  if (!a3_last_of(cnt + (size_t)m * H + h0, S, &flag)) return;
  // same merge as attn_decode_merge_kernel (per head: thread g computes the stats · 2 dims per thread, fmaf in ascending s)
  if (tid < G) {
    const int g = tid;
    const size_t base = ((size_t)m * H + h0 + g) * S;
    float M_g = -INFINITY;
    for (int ss = 0; ss < S; ++ss) M_g = fmaxf(M_g, __ldcg(pm + base + ss));
    float sum = 0.f;
    for (int ss = 0; ss < S; ++ss) {
      const float pmv = __ldcg(pm + base + ss);
      const float f = pmv > -INFINITY ? expf(pmv - M_g) : 0.f;
      scs[g][ss] = f;
      sum += f * __ldcg(ps + base + ss);
    }
    stat[g][0] = M_g; stat[g][1] = sum + expf(sink[h0 + g] - M_g);
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const size_t base = ((size_t)m * H + h0 + g) * S;
    float b0 = 0.f, b1 = 0.f;
    for (int ss = 0; ss < S; ++ss) {
      const float* pp = pacc + (base + ss) * S3AD + d0;
      b0 = fmaf(scs[g][ss], __ldcg(pp), b0); b1 = fmaf(scs[g][ss], __ldcg(pp + 1), b1);
    }
    s3_store_o_pair(o + ((size_t)m * H + h0 + g) * S3AD, d0, b0 / stat[g][1], b1 / stat[g][1], S3AD, rd, rope_freqs, rpos);
  }
}

// ---- launch helpers -------------------------------------------------------------------------------------------------------------
inline void a3_check(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) return;
  throw std::runtime_error(std::string("attn3 launch failed: ").append(what).append(": ").append(cudaGetErrorString(e)));
}
inline int a3_sms() {
  static const int n = [] {
    int dev = 0, v = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
    return v;
  }();
  return n;
}
inline bool a3_al(const void* p, size_t a) { return ((uintptr_t)p % a) == 0; }
constexpr size_t kA3SmemMax = 98 * 1024;
// Dynamic smem limit attribute (only above 48KB) + resident blocks per SM — once per (kernel, smem). Different kernels with the same signature (differing only in template arguments)
//   share a function pointer type, so the kernel address is part of the cache key.
template <class Kern>
int a3_occupancy(Kern kern, int threads, size_t smem) {
  struct Ent { const void* k; size_t smem; int n; };
  static std::mutex mu;
  static std::vector<Ent> cache;
  static std::vector<std::pair<const void*, size_t>> configured;
  const void* key = reinterpret_cast<const void*>(kern);
  std::lock_guard<std::mutex> lk(mu);
  for (const auto& p : cache) if (p.k == key && p.smem == smem) return p.n;
  if (smem > 48 * 1024) {
    size_t* have = nullptr;
    for (auto& c : configured) if (c.first == key) have = &c.second;
    if (!have || *have < smem) {
      CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      if (have) *have = smem; else configured.emplace_back(key, smem);
    }
  }
  int n = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, threads, smem));
  cache.push_back({key, smem, n});
  return n;
}

inline size_t r3_tail_bytes(int M, int E) { return ((size_t)2 * M * E + (size_t)M * R3KMAX) * 4; }
void router3_launch(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                    float route_scale, float* scores, int32_t* ids, float* w, int* counter, int phase, int mode, cudaStream_t st, CachePriorArgs cp = {}) {
  const bool ok = M >= 1 && M <= R3T / 32 && E >= 1 && E <= 32 * R3EV && k >= 1 && k <= R3KMAX && k <= E && K >= 8 && K % 8 == 0 && a3_al(gate_w, 16) &&
                  a3_al(xn, 16);
  const size_t sm_reg = std::max((size_t)K * 4, r3_tail_bytes(M, E));
  const size_t sm_xs = std::max((size_t)K * 4 + (size_t)M * K * 2, r3_tail_bytes(M, E));
  if (!ok || sm_reg > kA3SmemMax) {  // absorbed: outside this version's range, fall back to the serving (ATTN2) version
    attn2_router(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter, st);
    return;
  }
  bool xs = false;
  if (mode == 1) xs = sm_xs <= kA3SmemMax;
  else if (mode < 0 && sm_xs <= kA3SmemMax) xs = (long)a3_occupancy(a3_router_kernel<true>, R3T, sm_xs) * a3_sms() >= E;  // only for a single wave (all E blocks resident)
  if (xs) a3_router_kernel<true><<<E, R3T, sm_xs, st>>>(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter, phase, cp);
  else {
    (void)a3_occupancy(a3_router_kernel<false>, R3T, sm_reg);  // attribute above 48KB (if applicable)
    a3_router_kernel<false><<<E, R3T, sm_reg, st>>>(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter, phase, cp);
  }
  a3_check(xs ? "router3(xs)" : "router3");
}

}  // namespace

bool decode_router3_on() { static const bool on = a3_env_on("HIVE_DECODE_ROUTER3"); return on; }
bool decode_sparse3_on() { static const bool on = a3_env_on("HIVE_DECODE_SPARSE3"); return on; }
bool decode_qkv3_on() { static const bool on = a3_env_on("HIVE_DECODE_QKV3"); return on; }
bool decode_hcmix3_on() { static const bool on = a3_env_on("HIVE_DECODE_HCMIX3"); return on; }
int decode_sparse3_g() {
  static const int g = [] {
    const char* v = getenv("HIVE_DECODE_SPARSE3_G");
    const int x = v && *v ? atoi(v) : 0;
    return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 0;  // any other value = automatic (normalized — not rejected)
  }();
  return g;
}

void attn3_router(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                  float route_scale, float* scores, int32_t* ids, float* w, int* counter, cudaStream_t st) {
  router3_launch(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter, 0, -1, st);
}
void attn3_router_cp(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                     float route_scale, float* scores, int32_t* ids, float* w, int* counter, const CachePriorArgs& cp, cudaStream_t st) {
  router3_launch(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter, 0, -1, st, cp);
}
void attn3_router_phase(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                        float route_scale, float* scores, int32_t* ids, float* w, int* counter, int phase, int mode, cudaStream_t st) {
  router3_launch(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter, phase, mode, st);
}

void hc_mix_pre_norm3(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                      const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, cudaStream_t st) {
  // shapes of this version (model dim 5120, hc 4): elements per thread NK = 80 (mix), ND = 20 (pre+norm). Anything else uses the previous kernel (absorbed)
  if (!(hc == 4 && dim == 5120 && hcdim == hc * dim && M >= 1)) {
    hc_mix_pre_norm(h, W, M, hcdim, mix_hc, eps, mixes, rsq, pre_in, hc, dim, norm_w, norm_eps, x, xn, st);
    return;
  }
  hc_mix_pre_norm3_kernel<4, 80, 20><<<dim3(M, mix_hc + 2), H3T, 0, st>>>(h, W, hcdim, mix_hc, eps, mixes, rsq, pre_in, dim, norm_w, norm_eps, x, xn);
  a3_check("hc_mix3");
}

bool attn3_qkv_a(const DecAttnArgs& a, bool xf_on, cudaStream_t st) {
  const int M = a.M, K = a.dim, D = a.D;
  // shapes of this version: K = 5120 (5 blocks per lane) · q_lora 1280 (tail NQ 5) · D 512 (ND 2) · rd ≤ D · input 16B-aligned · M 1..8
  if (!(M >= 1 && M <= Q3M && K == 5120 && a.q_lora == 1280 && D == 512 && a.rd % 2 == 0 && a.rd <= D && (D - a.rd) % 2 == 0 && a3_al(a.xn, 16)))
    return false;
  const size_t smem = (size_t)M * K + (size_t)M * (K / 32) * 4 + (size_t)M * (K / 32) + 16;  // same layout as the previous smem_bytes
  auto kern = dec_qkv3_kernel<5, 5, 2>;
  if (smem > kA3SmemMax) return false;
  (void)a3_occupancy(kern, Q3T, smem);
  kern<<<a.q_lora / Q3OUT + D / Q3OUT, Q3T, smem, st>>>(a.xn, K, a.wqa, a.sqa, a.q_lora, a.q_norm, a.wkv, a.skv, D, a.kv_norm, a.eps, a.rd, a.freqs, M, a.qr,
                                                         a.qrn, a.kv, a.vgrp ? nullptr : a.ring_ptrs, a.win, a.idx, a.idx_stride, xf_on ? a.xf : nullptr, a.src,
                                                         a.tab, a.soa, a.counters, a.vgrp);  // verify version: ring written at the end of the front end (after attention) · window indices from the vgrp formula
  a3_check("qkv3");
  return true;
}

int attn3_sparse_auto_g(int M, int H, int S) {
  const long blocks = (long)M * H * S;
  const int sms = a3_sms();
  for (int g : {1, 2, 4, 8})
    if (H % g == 0 && blocks / g <= sms) return g;
  return H % 8 == 0 ? 8 : (H % 4 == 0 ? 4 : (H % 2 == 0 ? 2 : 1));
}

bool attn3_sparse(const DecAttnArgs& a, int G, cudaStream_t st) {
  const int M = a.M, H = a.H, S = a.splits;
  if (!(M >= 1 && M <= 8 && a.D == S3AD && S >= 1 && S <= 16 && H >= 1)) return false;
  if (G <= 0) G = decode_sparse3_g();
  if (G <= 0) G = attn3_sparse_auto_g(M, H, S);
  if (H % G != 0) G = 1;
  const int ncols_max = a.win + (a.ratio ? a.index_topk : 0);
  const int cmax = std::max(1, (ncols_max + S - 1) / S);
  const size_t smem = (size_t)2 * S3TK * S3AD * 2 + (size_t)cmax * 4 * (G + 1) + 16;
  if (smem > kA3SmemMax) return false;
  const int grid = M * (H / G) * S;
  const float scale = 1.0f / sqrtf((float)a.D);
#define A3_SPARSE(GG)                                                                                                                            \
  do {                                                                                                                                           \
    (void)a3_occupancy(dec_attn3_kernel<GG>, S3T, smem);                                                                                        \
    dec_attn3_kernel<GG><<<grid, S3T, smem, st>>>(a.q, H, a.tab, a.kv, M, a.win, a.idx, a.idx_stride, a.sink, scale, S, cmax, a.o, a.pacc, a.pm, a.ps, \
                                                  a.freqs, a.rd, a.attn_cnt);                                                                    \
  } while (0)
  switch (G) {
    case 8: A3_SPARSE(8); break;
    case 4: A3_SPARSE(4); break;
    case 2: A3_SPARSE(2); break;
    default: A3_SPARSE(1); break;
  }
#undef A3_SPARSE
  a3_check("sparse3");
  return true;
}

__global__ void hc_inject_rows_kernel(bf16* __restrict__ h, const float* __restrict__ post, const float* __restrict__ y, const int32_t* __restrict__ rows,
                                      int n, int M, int hc, int dim) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * dim) return;
  const int m = (int)(i / dim), d = (int)(i % dim);
  float acc = 0.f;
  for (int r = 0; r < n; ++r) if (rows[r] == m) acc += y[(size_t)r * dim + d];
  if (acc == 0.f) return;
  for (int c = 0; c < hc; ++c) {
    bf16* p = h + ((size_t)m * hc + c) * dim + d;
    *p = __float2bfloat16(__bfloat162float(*p) + post[m * hc + c] * acc);
  }
}
void hc_inject_rows(bf16* h, const float* post, const float* y, const int32_t* rows, int n, int M, int hc, int dim, cudaStream_t st) {
  if (n <= 0) return;
  const size_t tot = (size_t)M * dim;
  hc_inject_rows_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(h, post, y, rows, n, M, hc, dim);
  a3_check("hc_inject_rows");
}
__global__ void copy_f32_kernel(float* __restrict__ d, const float* __restrict__ s, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = s[i];
}
void copy_f32(float* dst, const float* src, size_t n, cudaStream_t st) {
  if (!n) return;
  copy_f32_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(dst, src, n);
  a3_check("copy_f32");
}

}  // namespace hive::k
