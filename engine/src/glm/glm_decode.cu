// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash decode-path kernels (M ≤ 8). Contracts are in include/hive/glm/glm_decode.h; validation in tests/test_glm_decode.cu.
//
// Why the old path was slow (read from the code, measured in the test):
//   · hc: k::hc_mix runs one block per (row, coefficient) whose 256 threads each walk 64 fp32 weights in a dependent fmaf chain with the
//     load inside the loop → ~64 serialized memory round trips; Sinkhorn runs on ONE thread per token; three launches per stage.
//   · BF16 projections: one cuBLAS GEMM per matrix (8 per KDA layer) at M ≤ 8.
//   · dense NVFP4 / FP8 shared expert: separate GEMVs + conversion + SwiGLU launches, loads not issued ahead of the arithmetic.
// Provenance: original code for this repository. The NVFP4 tensor-core helpers (scales64, decode8h, decode8, x8_to_h, mma16816, mma32,
//   row_dot_cc) are copied from engine/src/glm/glm_moe.cu and the per-chunk FP8 dot follows engine/src/glm/fp8b.cu; the hc bodies reproduce
//   model_kernels.cu hc_mix_body / hc_pre_norm_body / hc_post_kernel and include hc_decode_fused.cuh (all this repository, MIT). No external code.
// Negative controls (each must make the test FAIL): -DHIVE_GLM_DEC_NEG_SINKHORN (one Sinkhorn iteration less) · -DHIVE_GLM_DEC_NEG_POST
//   (comb transposed in the fused hc_post) · -DHIVE_GLM_DEC_NEG_STRIDE (segment output row stride ignored) · -DHIVE_GLM_DEC_NEG_TIE (router
//   ties go to the larger id) · -DHIVE_GLM_DEC_NEG_NVFP4 (e4m3 scale pair swapped) · -DHIVE_GLM_DEC_NEG_FP8 (wrong block-scale column).
#include "hive/glm/glm_decode.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cfloat>
#include <cstdlib>

#include "hive/common.h"
#include "hive/warp_reduce.cuh"
#include "../kernels/hc_decode_fused.cuh"  // 16-lane Sinkhorn, bit-identical to model_kernels.cu sinkhorn_one

namespace hive::glm {

namespace {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_stream(const void* p) {  // weights: read once, no L1 allocation
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(int* p, int v) { asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }

// Launch with programmatic stream serialization (PDL) allowed.
template <class... KArgs, class... Args>
void launch_pdl(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  static const bool no_pdl = [] { const char* v = getenv("HIVE_GLM_DEC_NO_PDL"); return v && *v && *v != '0'; }();  // A/B switch
  cfg.attrs = at; cfg.numAttrs = no_pdl ? 0 : 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, std::forward<Args>(args)...));
}

// ======================================================================================================================================
// 1/2. hyper-connections
// ======================================================================================================================================
constexpr int kHC = 4, kMix = (2 + kHC) * kHC;  // 24
#ifdef HIVE_GLM_DEC_NEG_SINKHORN
constexpr int kItersDelta = -1;
#else
constexpr int kItersDelta = 0;
#endif

// h values of row m that thread t owns: element i = t + 256·s of the flattened [4·DIM] row (s < 4·DIM/256), i.e. stream c = s / Q,
//   column d = t + 256·(s % Q) — the same element set the mix dot products (hc_mix_body) and the x/rmsnorm loop (hc_pre_norm_body) use.
//   With post_x: the new h after hc_post (same expression as model_kernels.cu hc_post_kernel), rounded to bf16 and widened back.
//   h, post and comb are written later in the same grid → plain loads (no .nc).
template <int DIM>
__device__ __forceinline__ void hc_load_h(const bf16* hm, const bf16* __restrict__ px, const float* post, const float* comb, float (&hv)[kHC * DIM / 256]) {
  constexpr int Q = DIM / 256;
  const int t = threadIdx.x;
  if (px) {
    float pp[kHC], cc[kHC * kHC];
#pragma unroll
    for (int c = 0; c < kHC; ++c) pp[c] = post[c];
#pragma unroll
    for (int q = 0; q < kHC * kHC; ++q) cc[q] = comb[q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int d = t + 256 * q;
      float res[kHC];
#pragma unroll
      for (int j = 0; j < kHC; ++j) res[j] = bf2f(hm[(size_t)j * DIM + d]);
      const float xv = bf2f(__ldg(px + d));
#pragma unroll
      for (int c = 0; c < kHC; ++c) {
        float s = __fmul_rn(pp[c], xv);  // explicit order = hc_post_kernel's SASS: FMUL post·x, then one FFMA per stream
#pragma unroll
        for (int j = 0; j < kHC; ++j) {
#ifdef HIVE_GLM_DEC_NEG_POST
          s = fmaf(cc[c * kHC + j], res[j], s);  // negative control: comb transposed
#else
          s = fmaf(cc[j * kHC + c], res[j], s);
#endif
        }
        hv[c * Q + q] = bf2f(f2bf(s));
      }
    }
  } else {
#pragma unroll
    for (int s = 0; s < kHC * Q; ++s) hv[s] = bf2f(hm[t + 256 * s]);
  }
}

template <int DIM>
__global__ void __launch_bounds__(256) hc_pre_kernel(bf16* h, const bf16* __restrict__ post_x, const float* __restrict__ fn, const float* __restrict__ scale,
                                                     const float* __restrict__ base, const bf16* __restrict__ norm_w, int M, float rms_eps, float hc_eps,
                                                     int iters, float* mixes, float* rsq, float* pre, float* post, float* comb, bf16* __restrict__ x,
                                                     bf16* __restrict__ xn, int* sync) {
  constexpr int HCD = kHC * DIM, PER = HCD / 256, Q = DIM / 256;
  __shared__ float red[8];
  __shared__ float pin[kHC];
  __shared__ int last;
  const int t = threadIdx.x;
  const int nmix = M * (kMix + 1);
  const bool is_mix = (int)blockIdx.x < nmix;
  const int m = is_mix ? blockIdx.x / (kMix + 1) : blockIdx.x - nmix;
  const int j = is_mix ? blockIdx.x % (kMix + 1) : kMix + 1;
  pdl_trigger();
  // weights (independent of the previous kernel): issued before the dependency wait
  float wv[PER];
  if (j < kMix) {
    const float* wr = fn + (size_t)j * HCD + t;
#pragma unroll
    for (int s = 0; s < PER; ++s) wv[s] = __ldg(wr + 256 * s);
  }
  pdl_wait();
  bf16* hm = h + (size_t)m * HCD;
  float hv[PER];
  hc_load_h<DIM>(hm, post_x ? post_x + (size_t)m * DIM : nullptr, post + m * kHC, comb + (size_t)m * kHC * kHC, hv);
  if (is_mix) {  // == hc_mix_body: fixed per-thread order s = 0..PER-1, xor butterfly, serial sum of the 8 warp partials
    float acc = 0.f;
    if (j < kMix) {
#pragma unroll
      for (int s = 0; s < PER; ++s) acc = fmaf(hv[s], wv[s], acc);
    } else {
#pragma unroll
      for (int s = 0; s < PER; ++s) { const float v = hv[s]; acc += v * v; }
    }
    acc = hive::cu::warp_allreduce(acc);
    if ((t & 31) == 0) red[t >> 5] = acc;
    __syncthreads();
    if (t == 0) {
      float tt = 0.f;
      for (int i = 0; i < 8; ++i) tt += red[i];
      if (j < kMix) mixes[(size_t)m * kMix + j] = tt;
      else rsq[m] = rsqrtf(tt / (float)HCD + rms_eps);
    }
  }
  // arrival (every block: its reads of the old h/post/comb and its mixes write are ordered before the counter)
  __threadfence();
  __syncthreads();
  if (t == 0) last = atomicAdd(&sync[0], 1) == (int)gridDim.x - 1;
  __syncthreads();
  if (last) {
    __threadfence();
    if (t == 0) sync[0] = 0;
    k::hcdf::sinkhorn_block<kHC>(mixes, rsq, scale, base, M, iters + kItersDelta, hc_eps, pre, post, comb);
    __threadfence();
    __syncthreads();
    if (t == 0) st_release(&sync[1], 1);
  }
  if (is_mix) return;
  // norm block m: wait for Sinkhorn, then (hc_post write-back) + x = Σ pre·h + rmsnorm (== hc_pre_norm_body)
  if (t == 0) while (ld_acquire(&sync[1]) == 0) __nanosleep(20);
  __syncthreads();
  if (post_x) {
#pragma unroll
    for (int s = 0; s < PER; ++s) hm[t + 256 * (s % Q) + (size_t)(s / Q) * DIM] = f2bf(hv[s]);
  }
  if (t < kHC) pin[t] = __ldcg(pre + m * kHC + t);
  __syncthreads();
  float ss = 0.f;
  float xvv[Q];
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < kHC; ++c) s += pin[c] * hv[c * Q + q];
    const bf16 xb = f2bf(s);
    x[(size_t)m * DIM + t + 256 * q] = xb;
    const float v = bf2f(xb);
    xvv[q] = v;
    ss += v * v;
  }
  ss = hive::cu::warp_allreduce(ss);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {
    float tt = t < 8 ? red[t] : 0.f;
    tt = hive::cu::warp_allreduce(tt);
    if (t == 0) red[0] = tt;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / (float)DIM + rms_eps);
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    const int d = t + 256 * q;
    xn[(size_t)m * DIM + d] = f2bf(bf2f(norm_w[d]) * (xvv[q] * rs));
  }
  if (t == 0 && atomicAdd(&sync[2], 1) == M - 1) { sync[2] = 0; st_release(&sync[1], 0); }  // every norm block has seen the flag
}

// hc_post: thread = 8 consecutive columns of one row
__global__ void __launch_bounds__(256) hc_post_kernel(const bf16* __restrict__ x, const float* __restrict__ post, const float* __restrict__ comb, int M, int dim,
                                                      bf16* h) {
  pdl_trigger();
  const int i8 = blockIdx.x * blockDim.x + threadIdx.x, per = dim / 8;
  if (i8 >= M * per) return;
  const int m = i8 / per, d0 = (i8 % per) * 8;
  pdl_wait();
  float pp[kHC], cc[kHC * kHC];
#pragma unroll
  for (int c = 0; c < kHC; ++c) pp[c] = post[m * kHC + c];
#pragma unroll
  for (int q = 0; q < kHC * kHC; ++q) cc[q] = comb[(size_t)m * kHC * kHC + q];
  const uint4 xv4 = *reinterpret_cast<const uint4*>(x + (size_t)m * dim + d0);
  uint4 r4[kHC];
#pragma unroll
  for (int jj = 0; jj < kHC; ++jj) r4[jj] = *reinterpret_cast<const uint4*>(h + ((size_t)m * kHC + jj) * dim + d0);
  const bf16* xe = reinterpret_cast<const bf16*>(&xv4);
  uint4 o4[kHC];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float res[kHC];
#pragma unroll
    for (int jj = 0; jj < kHC; ++jj) res[jj] = bf2f(reinterpret_cast<const bf16*>(&r4[jj])[e]);
    const float xv = bf2f(xe[e]);
#pragma unroll
    for (int c = 0; c < kHC; ++c) {
      float s = __fmul_rn(pp[c], xv);
#pragma unroll
      for (int jj = 0; jj < kHC; ++jj) s = fmaf(cc[jj * kHC + c], res[jj], s);
      reinterpret_cast<bf16*>(&o4[c])[e] = f2bf(s);
    }
  }
#pragma unroll
  for (int c = 0; c < kHC; ++c) *reinterpret_cast<uint4*>(h + ((size_t)m * kHC + c) * dim + d0) = o4[c];
}

// ======================================================================================================================================
// 3/4. BF16 GEMV (segment table) and router
// ======================================================================================================================================
// 8 bf16 activations (one uint4) · 8 fp32 weights (column order) into acc
__device__ __forceinline__ float dot8(const uint4 xv, const float (&w)[8], float acc) {
  const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    acc = fmaf(__uint_as_float(xs[b] << 16), w[2 * b], acc);
    acc = fmaf(__uint_as_float(xs[b] & 0xFFFF0000u), w[2 * b + 1], acc);
  }
  return acc;
}

struct SegTab {
  GemvSeg s[kGemvMaxSeg];
  int blk0[kGemvMaxSeg + 1];   // first block of each segment (prefix sums; a block never spans two segments)
  int wt[kGemvMaxSeg];         // warps per 16-row tile (K split into wt parts, combined in part order)
  int nseg;
};
__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// Tensor-core GEMV. Block = 8 warps = (8/wt) tiles of 16 output rows; the wt warps of a tile split K into equal parts.
//   mma.m16n8k16 (bf16 operands, fp32 accumulation): mma rows = 16 weight rows (A), mma columns = the ≤ 8 activation rows (B).
//   Lane (g = lane/4, t = lane%4) streams 16-byte words of weight rows n0+g and n0+g+8 at columns 32·j + 8t (64 contiguous bytes per row
//   and instruction) and the matching 16 bytes of activation row g; one word pair = two mma steps (k pairs {0,1},{2,3} and {4,5},{6,7}
//   of the word in the A and B k-slots — the same permutation on both operands, so the dot product is unchanged).
//   Partials: c[0..1] = (row g, act 2t, 2t+1), c[2..3] = (row g+8, …) → smem → summed over the wt parts in part order → bf16.
//   Ping-pong batches of kTB words per row keep 2·kTB·2 = 16 words (256 B) per lane in flight; the first two batches are requested
//   before griddepcontrol.wait. Deterministic: fixed mma order per warp, fixed part order, no atomics.
#ifndef HIVE_GLM_DEC_TB
#define HIVE_GLM_DEC_TB 2  // measured: 1 → 1.29 TB/s, 2 → 1.49, 4 → 1.44, 8 → 1.42 (q|k|v set); o_proj 1.17 / 1.31-1.37 / 1.26 / 1.02
#endif
constexpr int kTB = HIVE_GLM_DEC_TB;
__device__ __forceinline__ void tc_load(uint4 (&qa)[kTB], uint4 (&qb)[kTB], const uint4* wa, const uint4* wb, int j0, int ns) {
#pragma unroll
  for (int j = 0; j < kTB; ++j)
    if (j0 + j < ns) { qa[j] = ld_stream(wa + 4 * (j0 + j)); qb[j] = ld_stream(wb + 4 * (j0 + j)); }
}
__device__ __forceinline__ void tc_mma(float (&c)[4], const uint4 (&qa)[kTB], const uint4 (&qb)[kTB], const uint4* xr, bool act, int j0, int ns) {
#pragma unroll
  for (int j = 0; j < kTB; ++j)
    if (j0 + j < ns) {
      const uint4 xv = act ? __ldg(xr + 4 * (j0 + j)) : make_uint4(0, 0, 0, 0);
      mma_bf16(c, qa[j].x, qb[j].x, qa[j].y, qb[j].y, xv.x, xv.y);
      mma_bf16(c, qa[j].z, qb[j].z, qa[j].w, qb[j].w, xv.z, xv.w);
    }
}
#ifndef HIVE_GLM_DEC_TCMIN
#define HIVE_GLM_DEC_TCMIN 3  // blocks per SM (≤ 85 registers): measured best with TB 2
#endif
__global__ void __launch_bounds__(256, HIVE_GLM_DEC_TCMIN) gemv_tc_kernel(const __grid_constant__ SegTab T, int M) {
  __shared__ float red[8][32][4];
  pdl_trigger();
  if ((int)blockIdx.x >= T.blk0[T.nseg]) return;
  int s = 0;
  while (s + 1 < T.nseg && (int)blockIdx.x >= T.blk0[s + 1]) ++s;
  const GemvSeg& G = T.s[s];
  const int wt = T.wt[s], tpb = 8 / wt;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tile0 = (blockIdx.x - T.blk0[s]) * tpb;
  const int n0 = (tile0 + warp / wt) * 16, part = warp % wt;
  const int kp = G.K / wt, ns = kp / 32;
  const int ra = n0 + g < G.N ? n0 + g : 0, rb = n0 + g + 8 < G.N ? n0 + g + 8 : 0;  // rows past N: valid memory, results dropped
  const uint4* wa = reinterpret_cast<const uint4*>(G.W + (size_t)ra * G.K + (size_t)part * kp) + t;
  const uint4* wb = reinterpret_cast<const uint4*>(G.W + (size_t)rb * G.K + (size_t)part * kp) + t;
  const bool act = g < M;
  const uint4* xr = reinterpret_cast<const uint4*>(G.x + (size_t)(act ? g : 0) * G.ldx + (size_t)part * kp) + t;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  uint4 qa0[kTB], qb0[kTB], qa1[kTB], qb1[kTB];
  tc_load(qa0, qb0, wa, wb, 0, ns);
  tc_load(qa1, qb1, wa, wb, kTB, ns);
  pdl_wait();
  for (int j0 = 0; j0 < ns; j0 += 2 * kTB) {
    tc_mma(c, qa0, qb0, xr, act, j0, ns);
    tc_load(qa0, qb0, wa, wb, j0 + 2 * kTB, ns);
    tc_mma(c, qa1, qb1, xr, act, j0 + kTB, ns);
    tc_load(qa1, qb1, wa, wb, j0 + 3 * kTB, ns);
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) red[warp][lane][k] = c[k];
  __syncthreads();
  for (int i = threadIdx.x; i < tpb * 128; i += 256) {  // (tile tl, row r < 16, act a < 8)
    const int tl = i >> 7, r = (i >> 3) & 15, a = i & 7;
    const int n = (tile0 + tl) * 16 + r;
    if (a >= M || n >= G.N) continue;
    const int ln = (r & 7) * 4 + (a >> 1), k = (a & 1) + (r >> 3) * 2;
    float v = 0.f;
    for (int q = 0; q < wt; ++q) v += red[tl * wt + q][ln][k];
#ifdef HIVE_GLM_DEC_NEG_STRIDE
    G.y[(size_t)a * G.N + n] = f2bf(v);  // negative control: output row stride ignored
#else
    G.y[(size_t)a * G.ldy + n] = f2bf(v);
#endif
  }
}

// Router, two launches. (1) Logits on the tensor cores: block = ONE warp = (16-expert tile, K part of kRouterKP columns) → E/16 · K/kRouterKP
//   blocks spread over the GPU (measured: 36 blocks with a warp per expert ≈ 12 µs for the 2.4 MB gate — 36 SMs pulling 64 KB each); each
//   stores its fp32 partial tile into part[p][m][e]. (2) PDL tail: parts added in ascending p (fixed order), warp m = top-k of row m
//   (measured: a single-warp tail inside kernel 1 cost ≈ 5 µs per row).
constexpr int kRouterKP = 512, kRouterMaxE = 512;
__global__ void __launch_bounds__(32) router_logits_kernel(const bf16* __restrict__ x, int ldx, int M, const bf16* __restrict__ W, int E, int K,
                                                           float* __restrict__ part_out) {
  pdl_trigger();
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int P = K / kRouterKP, tile = blockIdx.x / P, part = blockIdx.x % P;
  const int n0 = tile * 16;
  const int ra = n0 + g < E ? n0 + g : 0, rb = n0 + g + 8 < E ? n0 + g + 8 : 0;
  const uint4* wa = reinterpret_cast<const uint4*>(W + (size_t)ra * K + (size_t)part * kRouterKP) + t;
  const uint4* wb = reinterpret_cast<const uint4*>(W + (size_t)rb * K + (size_t)part * kRouterKP) + t;
  const bool act = g < M;
  const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(act ? g : 0) * ldx + (size_t)part * kRouterKP) + t;
  constexpr int ns = kRouterKP / 32;  // 16 word steps
  uint4 qa[ns], qb[ns];
#pragma unroll
  for (int j = 0; j < ns; ++j) { qa[j] = ld_stream(wa + 4 * j); qb[j] = ld_stream(wb + 4 * j); }
  pdl_wait();
  float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < ns; ++j) {
    const uint4 xv = act ? __ldg(xr + 4 * j) : make_uint4(0, 0, 0, 0);
    mma_bf16(c, qa[j].x, qb[j].x, qa[j].y, qb[j].y, xv.x, xv.y);
    mma_bf16(c, qa[j].z, qb[j].z, qa[j].w, qb[j].w, xv.z, xv.w);
  }
  // c[0..1] = (expert n0+g, rows 2t, 2t+1), c[2..3] = (expert n0+g+8, …) → part_out[part][m][e]
  float* pp = part_out + (size_t)part * M * E;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int a = 2 * t + (k & 1), e = n0 + g + (k >> 1) * 8;
    if (a < M && e < E) pp[(size_t)a * E + e] = c[k];
  }
}
// Top-k of one row by one warp, equivalent to router_topk's serial scan (k times: first maximum of c by strict '>', ascending id).
//   Lane l owns candidates e = l + 32q; it sorts them ONCE (odd-even transposition network on registers, order: larger c first, equal c →
//   smaller id first; values that the serial scan can never pick — NaN or ≤ -FLT_MAX — are mapped to -inf and never selected). Each round
//   is then a 5-level butterfly over the lane heads (larger value; equal → smaller id) and a predicated pop in the winning lane.
//   (Measured: re-scanning 9 candidates per lane per round = one dependent chain of ~1 k instructions at 17.5 cycles each ≈ 9 µs.)
template <int NPL>
__device__ __forceinline__ void router_select(const float* c, const float* sr, int E, int kk, float scale, int32_t* ids, float* wts, int32_t* ids_h,
                                              float* w_h) {
  const int lane = threadIdx.x & 31;
  float v[NPL];
  int id[NPL];
#pragma unroll
  for (int q = 0; q < NPL; ++q) {
    const int e = lane + 32 * q;
    const float x = e < E ? c[e] : -INFINITY;
    v[q] = (x > -FLT_MAX) ? x : -INFINITY;  // NaN, -inf, -FLT_MAX: never selectable
    id[q] = e;
  }
#pragma unroll
  for (int pass = 0; pass < NPL; ++pass)
#pragma unroll
    for (int q = pass & 1; q + 1 < NPL; q += 2) {
      // ids within a lane ascend with q, so "q+1 before q" only when strictly larger (ties keep the smaller id first)
      const bool sw = v[q + 1] > v[q];
      const float tv = v[q]; const int ti = id[q];
      v[q] = sw ? v[q + 1] : v[q]; id[q] = sw ? id[q + 1] : id[q];
      v[q + 1] = sw ? tv : v[q + 1]; id[q + 1] = sw ? ti : id[q + 1];
    }
  float sum = 0.f, my_s = 0.f;
  int my_id = -1;  // lane j keeps selection j
#pragma unroll 1
  for (int j = 0; j < kk; ++j) {
    float bv = v[0] > -FLT_MAX ? v[0] : -FLT_MAX;
    int bi = v[0] > -FLT_MAX ? id[0] : -1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float pv = __shfl_xor_sync(0xffffffffu, bv, o);
      const int pi = __shfl_xor_sync(0xffffffffu, bi, o);
#ifdef HIVE_GLM_DEC_NEG_TIE
      const bool take = pv > bv || (pv == bv && pi >= 0 && (bi < 0 || pi > bi));  // negative control: ties → larger id
#else
      const bool take = pv > bv || (pv == bv && pi >= 0 && (bi < 0 || pi < bi));  // ties → smaller id (= serial first maximum)
#endif
      if (take) { bv = pv; bi = pi; }
    }
    if (bi >= 0 && (bi & 31) == lane) {  // pop the head of the winning lane
#pragma unroll
      for (int q = 0; q + 1 < NPL; ++q) { v[q] = v[q + 1]; id[q] = id[q + 1]; }
      v[NPL - 1] = -INFINITY;
    }
    const float bs = bi >= 0 ? sr[bi] : 0.f;
    sum += bs;  // all lanes: the same sequence (selection order) as router_topk's serial sum
    if (lane == j) { my_id = bi; my_s = bs; }
  }
  if (lane < kk) {
    const float inv = 1.f / (sum + 1e-20f);
    const float wv = my_id >= 0 ? my_s * inv * scale : 0.f;
    ids[lane] = my_id; wts[lane] = wv;
    if (ids_h) ids_h[lane] = my_id;
    if (w_h) w_h[lane] = wv;
  }
}
// logits[m][e] = Σ_p part[p][m][e] (ascending p, written over part 0: each thread reads all parts of its elements first) and, in shared
//   memory, s = 1/(1+expf(-logit)) and c = s + bias (router_topk's expressions); then warp m selects the top-k of row m from shared memory.
//   Loops are kept rolled on purpose: a register-array version unrolled over 16 rounds × 16 candidates was 7.6 k SASS instructions and,
//   with a cold L2 (as in decode, where the expert weights stream through L2 between two routers), cost ~10 µs of instruction fetch.
__global__ void __launch_bounds__(256) router_topk_kernel(float* logits, int M, int E, int P, const float* __restrict__ bias, int kk, float scale,
                                                          int32_t* __restrict__ ids, float* __restrict__ wts, int32_t* ids_h, float* w_h, GlmCachePrior cp) {
  extern __shared__ float rsm[];  // s [M·E], c [M·E]
  float* ss = rsm;
  float* cs = rsm + M * E;
  pdl_trigger();
  pdl_wait();
  for (int i = threadIdx.x; i < M * E; i += 256) {
    float v = 0.f;
#pragma unroll 8
    for (int q = 0; q < P; ++q) v += logits[(size_t)q * M * E + i];
    logits[i] = v;
    const float sg = 1.f / (1.f + expf(-v));
    ss[i] = sg;
    cs[i] = sg + __ldg(bias + i % E);
  }
  __syncthreads();
  const int m = threadIdx.x >> 5;
  // Cache prior (HIVE_CACHE_PRIOR — see GlmCachePrior in glm_decode.h): the selection score c of experts resident in VRAM, and of the row's
  //   top-J by c, gets + λ·Δavg; Δavg = running average of the per-row range max(c) − min(c) of this layer. Only the selection changes —
  //   router_select still weights the chosen experts by s.
  if (cp.mask && cp.lambda > 0.f) {
    __shared__ float rng[8];
    __shared__ float bonus;
    __shared__ int topj[8][4];
    const int lane = threadIdx.x & 31;
    if (m < M) {
      float mx = -INFINITY, mn = INFINITY;
      for (int e = lane; e < E; e += 32) { const float v = cs[m * E + e]; mx = fmaxf(mx, v); mn = fminf(mn, v); }
      for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o)); }
      if (lane == 0) rng[m] = mx - mn;
      for (int j = 0; j < cp.top_j; ++j) {  // top-J of the original selection scores (ties → lower id)
        float bv = -INFINITY; int bi = E;
        for (int e = lane; e < E; e += 32) {
          bool taken = false;
          for (int q = 0; q < j; ++q) taken |= topj[m][q] == e;
          const float v = cs[m * E + e];
          if (!taken && (v > bv || (v == bv && e < bi))) { bv = v; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
          if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) topj[m][j] = bi;
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
    if (m < M)
      for (int e = lane; e < E; e += 32) {
        bool boost = cp.mask[e] != 0;
        for (int q = 0; q < cp.top_j; ++q) boost |= topj[m][q] == e;
        if (boost) cs[m * E + e] += bonus;
      }
    __syncthreads();
  }
  if (m >= M) return;
  if (E <= 9 * 32) router_select<9>(cs + m * E, ss + m * E, E, kk, scale, ids + (size_t)m * kk, wts + (size_t)m * kk, ids_h ? ids_h + (size_t)m * kk : nullptr,
                                    w_h ? w_h + (size_t)m * kk : nullptr);
  else router_select<16>(cs + m * E, ss + m * E, E, kk, scale, ids + (size_t)m * kk, wts + (size_t)m * kk, ids_h ? ids_h + (size_t)m * kk : nullptr,
                         w_h ? w_h + (size_t)m * kk : nullptr);
}

// ======================================================================================================================================
// 5. dense NVFP4 MLP — tensor-core dot products (same exact decode as glm_moe.cu, copied)
// ======================================================================================================================================
__device__ __forceinline__ void scales64(uint16_t s2, uint32_t& s_lo, uint32_t& s_hi) {
  uint32_t sc; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sc) : "h"(s2));
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sc) : "r"(sc), "r"(0x54005400u));  // × 64 (exact)
#ifdef HIVE_GLM_DEC_NEG_NVFP4
  s_lo = __byte_perm(sc, 0, 0x3232); s_hi = __byte_perm(sc, 0, 0x1010);  // negative control: swapped scale pair
#else
  s_lo = __byte_perm(sc, 0, 0x1010); s_hi = __byte_perm(sc, 0, 0x3232);
#endif
}
// e2m1 → e4m3 bit trick (exact): see glm_moe.cu. h0 = (c0, c2), h1 = (c4, c6), h2 = (c1, c3), h3 = (c5, c7)
__device__ __forceinline__ void decode8h(uint32_t w, uint32_t sc, uint32_t (&h)[4]) {
  const uint32_t ev = ((w << 4) & 0x80808080u) | ((w << 2) & 0x1C1C1C1Cu);
  const uint32_t od = (w & 0x80808080u) | ((w >> 2) & 0x1C1C1C1Cu);
  asm("{\n.reg .b16 a, b, c, d;\nmov.b32 {a, b}, %4;\nmov.b32 {c, d}, %5;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n"
      "cvt.rn.f16x2.e4m3x2 %2, c;\ncvt.rn.f16x2.e4m3x2 %3, d;\n}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(ev), "r"(od));
#pragma unroll
  for (int k = 0; k < 4; ++k) asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(h[k]) : "r"(sc));
}
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
// 16 mma steps of one 32-column word group: rows A (mma rows g) and B (mma rows g+8), activation 32 bf16 (4 × uint4)
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
// exact CUDA-core row dot (fallback), result in all lanes
__device__ float row_dot_cc(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, int K, const bf16* __restrict__ a) {
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
  return hive::cu::warp_allreduce(acc);
}
// == swiglu_rows_kernel on bf16-rounded gate/up values
__device__ __forceinline__ float swiglu_bf(float g, float u, float limit) {
  g = bf2f(f2bf(g)); u = bf2f(f2bf(u));
  if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
  return g / (1.f + __expf(-g)) * u;
}

// gate/up: block = 8 rows i0..i0+7 of I (one m16n8k16 tile: mma rows 0-7 gate, 8-15 up; mma columns = the M activation rows).
//   K = H split over the 8 warps (H/8 columns each); lane quad t owns the 16-byte words 4j+t of its segment.
template <int KB>
__global__ void __launch_bounds__(256) dense_gu_kernel(Nvfp4Mat G, Nvfp4Mat U, const bf16* __restrict__ x, int ldx, int M, float limit, bf16* __restrict__ y) {
  __shared__ float red[8][32][4];
  pdl_trigger();
  const int H = G.K, I = G.N;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int i0 = blockIdx.x * 8, seg = warp * (H / 8);
  const uint8_t* wg = G.w + (size_t)(i0 + g) * (H / 2) + seg / 2;
  const uint8_t* wu = U.w + (size_t)(i0 + g) * (H / 2) + seg / 2;
  const uint16_t* sg = reinterpret_cast<const uint16_t*>(G.s + (size_t)(i0 + g) * (H / 16) + seg / 16);
  const uint16_t* su = reinterpret_cast<const uint16_t*>(U.s + (size_t)(i0 + g) * (H / 16) + seg / 16);
  uint4 qg[KB], qu[KB];
  uint16_t s2g[KB], s2u[KB];
#pragma unroll
  for (int j = 0; j < KB; ++j) {
    const int wi = 4 * j + t;
    qg[j] = ld_stream(wg + wi * 16); qu[j] = ld_stream(wu + wi * 16);
    s2g[j] = __ldg(sg + wi); s2u[j] = __ldg(su + wi);
  }
  pdl_wait();
  const bool act = g < M;
  const bf16* xr = x + (size_t)(act ? g : 0) * ldx + seg;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < KB; ++j) {
    uint4 xv[4];
    const uint4* xp = reinterpret_cast<const uint4*>(xr + 32 * (4 * j + t));
#pragma unroll
    for (int k = 0; k < 4; ++k) xv[k] = act ? __ldg(xp + k) : make_uint4(0, 0, 0, 0);
    mma32(acc, qg[j], s2g[j], qu[j], s2u[j], xv);
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) red[warp][lane][k] = acc[k];
  __syncthreads();
  float yv = 0.f;
  bool bad = false;
  const int fr = threadIdx.x >> 3, fa = threadIdx.x & 7;  // finalize thread: (row fr < 8, activation row fa)
  if (threadIdx.x < 64 && fa < M) {
    const int ln = fr * 4 + (fa >> 1), k = fa & 1;
    float gs = 0.f, us = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) { gs += red[w][ln][k]; us += red[w][ln][2 + k]; }
    gs *= G.g; us *= U.g;
    bad = !isfinite(gs) || !isfinite(us);
    yv = swiglu_bf(gs, us, limit);
  }
  if (__syncthreads_or(bad)) {  // rare: activation outside the f16 range or non-finite input → exact CUDA-core recompute
    const int row = i0 + warp;
    for (int a = 0; a < M; ++a) {
      const bf16* xa = x + (size_t)a * ldx;
      const float gs = row_dot_cc(G.w + (size_t)row * (H / 2), G.s + (size_t)row * (H / 16), H, xa) * G.g;
      const float us = row_dot_cc(U.w + (size_t)row * (H / 2), U.s + (size_t)row * (H / 16), H, xa) * U.g;
      if (lane == 0) y[(size_t)a * I + row] = f2bf(swiglu_bf(gs, us, limit));
    }
  } else if (threadIdx.x < 64 && fa < M) {
    y[(size_t)fa * I + i0 + fr] = f2bf(yv);
  }
}

// down: block = 16 output rows n0..n0+15 (mma rows), mma columns = the M rows of y; K = I split over the NW warps, lane quad t owns
//   words 4j+t of its segment, batches of BATCH words per row in flight (prefetch of the next batch while the current one is multiplied).
//   NW = 16 (measured: with 8 warps the 256 blocks reached 22 % achieved occupancy and ~63 % of DRAM peak).
template <int NW, int WPL, int BATCH>
__global__ void __launch_bounds__(NW * 32) dense_down_kernel(Nvfp4Mat D, const bf16* __restrict__ y, int M, bf16* __restrict__ out, int ldo) {
  static_assert(WPL % BATCH == 0, "down batch");
  __shared__ float red[NW][32][4];
  pdl_trigger();
  const int K = D.K;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * 16, seg = warp * (K / NW);
  const uint8_t* wa = D.w + (size_t)(n0 + g) * (K / 2) + seg / 2;
  const uint8_t* wb = D.w + (size_t)(n0 + g + 8) * (K / 2) + seg / 2;
  const uint16_t* sa = reinterpret_cast<const uint16_t*>(D.s + (size_t)(n0 + g) * (K / 16) + seg / 16);
  const uint16_t* sb = reinterpret_cast<const uint16_t*>(D.s + (size_t)(n0 + g + 8) * (K / 16) + seg / 16);
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
  pdl_wait();  // y complete after this
  const bool act = g < M;
  const bf16* yr = y + (size_t)(act ? g : 0) * K + seg;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j0 = 0; j0 < WPL; j0 += BATCH) {
    uint4 ca[BATCH], cb[BATCH];
    uint16_t csa[BATCH], csb[BATCH];
#pragma unroll
    for (int j = 0; j < BATCH; ++j) { ca[j] = qa[j]; cb[j] = qb[j]; csa[j] = s2a[j]; csb[j] = s2b[j]; }
    if (j0 + BATCH < WPL) load(j0 + BATCH);
#pragma unroll
    for (int j = 0; j < BATCH; ++j) {
      uint4 yv[4];
      const uint4* yp = reinterpret_cast<const uint4*>(yr + 32 * (4 * (j0 + j) + t));
#pragma unroll
      for (int k = 0; k < 4; ++k) yv[k] = act ? __ldg(yp + k) : make_uint4(0, 0, 0, 0);
      mma32(acc, ca[j], csa[j], cb[j], csb[j], yv);
    }
  }
  // acc: [0], [1] = output row n0+g · y rows 2t, 2t+1; [2], [3] = output row n0+g+8 · y rows 2t, 2t+1
#pragma unroll
  for (int k = 0; k < 4; ++k) red[warp][lane][k] = acc[k];
  __syncthreads();
  const int r = threadIdx.x & 15, a = threadIdx.x >> 4;  // finalize thread (output row r, y row a < 8): threads 0..127
  float v = 0.f;
  bool bad = false;
  if (threadIdx.x < 128 && a < M) {
    const int ln = (r & 7) * 4 + (a >> 1), k = (a & 1) + (r >> 3) * 2;
#pragma unroll
    for (int w = 0; w < NW; ++w) v += red[w][ln][k];
    v *= D.g;
    bad = !isfinite(v);
  }
  if (__syncthreads_or(bad)) {
    for (int rr = warp; rr < 16; rr += NW)
      for (int aa = 0; aa < M; ++aa) {
        const float vv = row_dot_cc(D.w + (size_t)(n0 + rr) * (K / 2), D.s + (size_t)(n0 + rr) * (K / 16), K, y + (size_t)aa * K) * D.g;
        if (lane == 0) out[(size_t)aa * ldo + n0 + rr] = f2bf(vv);
      }
  } else if (threadIdx.x < 128 && a < M) {
    out[(size_t)a * ldo + n0 + r] = f2bf(v);
  }
}

// ======================================================================================================================================
// 6. FP8 (128×128 block scales) shared expert — CUDA cores, bit-identical to fp8b_gemv
// ======================================================================================================================================
__device__ __forceinline__ void e4m3x16(const uint4 q, float (&w)[16]) {
  const uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t h0, h1;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"((uint16_t)(words[j] & 0xFFFFu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"((uint16_t)(words[j] >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&h0)), b = __half22float2(*reinterpret_cast<const __half2*>(&h1));
    w[4 * j] = a.x; w[4 * j + 1] = a.y; w[4 * j + 2] = b.x; w[4 * j + 3] = b.y;
  }
}
__device__ __forceinline__ void bf16x8_to_f(const uint4 v, float* f) {  // f[0..7] = 8 bf16 (exact widening)
  const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) { f[2 * k] = __uint_as_float(u[k] << 16); f[2 * k + 1] = __uint_as_float(u[k] & 0xFFFF0000u); }
}
__device__ __forceinline__ void bf16x8_to_f8(const uint4 v, float* f) { bf16x8_to_f(v, f + 8); }
// fp8b_gemv's per-chunk dot: d = Σ_{i<16} w[i]·x[i] (fmaf in column order) for 16 bf16 activations at xp
__device__ __forceinline__ float fp8_chunk_dot(const float (&w)[16], const bf16* xp) {
  const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(xp)), a1 = __ldg(reinterpret_cast<const uint4*>(xp) + 1);
  const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
  const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
  float d = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 f = __bfloat1622float2(p0[i]); d = fmaf(w[2 * i], f.x, d); d = fmaf(w[2 * i + 1], f.y, d); }
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 f = __bfloat1622float2(p1[i]); d = fmaf(w[8 + 2 * i], f.x, d); d = fmaf(w[8 + 2 * i + 1], f.y, d); }
  return d;
}
__device__ __forceinline__ float fp8_scale(const float* srow, int c) {
#ifdef HIVE_GLM_DEC_NEG_FP8
  return __ldg(srow + (c >> 4));  // negative control: wrong block-scale column
#else
  return __ldg(srow + (c >> 3));
#endif
}
__device__ __forceinline__ float xor_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// gate+up: warp = row i of I; lane chunks c = lane + 32·ii (16 columns each), all loaded before the arithmetic
template <int MM, int CPL>
#ifndef HIVE_GLM_DEC_SHMIN
#define HIVE_GLM_DEC_SHMIN 2  // measured M=8: 1 → 32.8 µs (167 regs), 2 → 30.7 µs, 3 → spills (158 µs)
#endif
__global__ void __launch_bounds__(256, HIVE_GLM_DEC_SHMIN) sh_gu_kernel(Fp8BMat G, Fp8BMat U, const bf16* __restrict__ x, int ldx, float limit, bf16* __restrict__ y) {
  pdl_trigger();
  const int lane = threadIdx.x & 31, i = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (i >= G.N) return;
  const int K = G.K, kb_n = (K + 127) / 128;
  const uint4* wg = reinterpret_cast<const uint4*>(G.w + (size_t)i * K);
  const uint4* wu = reinterpret_cast<const uint4*>(U.w + (size_t)i * K);
  const float* sgr = G.s + (size_t)(i / 128) * kb_n;
  const float* sur = U.s + (size_t)(i / 128) * kb_n;
  uint4 qg[CPL], qu[CPL];
  float sg[CPL], su[CPL];
#pragma unroll
  for (int ii = 0; ii < CPL; ++ii) {
    const int c = lane + 32 * ii;
    qg[ii] = ld_stream(wg + c); qu[ii] = ld_stream(wu + c);
    sg[ii] = fp8_scale(sgr, c); su[ii] = fp8_scale(sur, c);
  }
  pdl_wait();
  float ag[MM], au[MM];
#pragma unroll
  for (int m = 0; m < MM; ++m) { ag[m] = 0.f; au[m] = 0.f; }
#pragma unroll
  for (int ii = 0; ii < CPL; ++ii) {
    const int c = lane + 32 * ii;
    float wg[16], wu[16];
    e4m3x16(qg[ii], wg);
    e4m3x16(qu[ii], wu);
#pragma unroll
    for (int m = 0; m < MM; ++m) {  // x chunk loaded and widened once for both gate and up (same per-dot fma order as fp8b_gemv)
      const bf16* xp = x + (size_t)m * ldx + c * 16;
      const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(xp)), a1 = __ldg(reinterpret_cast<const uint4*>(xp) + 1);
      float xf[16];
      bf16x8_to_f(a0, xf); bf16x8_to_f8(a1, xf);
      float dg = 0.f, du = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) { dg = fmaf(wg[i], xf[i], dg); du = fmaf(wu[i], xf[i], du); }
      ag[m] = fmaf(dg, sg[ii], ag[m]);
      au[m] = fmaf(du, su[ii], au[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    const float gv = xor_sum(ag[m]), uv = xor_sum(au[m]);
    if (lane == 0) y[(size_t)m * G.N + i] = f2bf(swiglu_bf(gv, uv, limit));
  }
}
// down: warp = output row n
template <int MM, int CPL>
__global__ void __launch_bounds__(256, HIVE_GLM_DEC_SHMIN) sh_down_kernel(Fp8BMat D, const bf16* __restrict__ y, bf16* __restrict__ out, int ldo) {
  pdl_trigger();
  const int lane = threadIdx.x & 31, n = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (n >= D.N) return;
  const int K = D.K, kb_n = (K + 127) / 128;
  const uint4* wr = reinterpret_cast<const uint4*>(D.w + (size_t)n * K);
  const float* sr = D.s + (size_t)(n / 128) * kb_n;
  uint4 q[CPL];
  float sc[CPL];
#pragma unroll
  for (int ii = 0; ii < CPL; ++ii) { const int c = lane + 32 * ii; q[ii] = ld_stream(wr + c); sc[ii] = fp8_scale(sr, c); }
  pdl_wait();  // y complete after this
  float acc[MM];
#pragma unroll
  for (int m = 0; m < MM; ++m) acc[m] = 0.f;
#pragma unroll
  for (int ii = 0; ii < CPL; ++ii) {
    const int c = lane + 32 * ii;
    float w[16];
    e4m3x16(q[ii], w);
#pragma unroll
    for (int m = 0; m < MM; ++m) acc[m] = fmaf(fp8_chunk_dot(w, y + (size_t)m * K + c * 16), sc[ii], acc[m]);
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    const float v = xor_sum(acc[m]);
    if (lane == 0) out[(size_t)m * ldo + n] = f2bf(v);
  }
}

#define HIVE_DISPATCH_M(M, CALL) \
  switch (M) {                   \
    case 1: CALL(1); break;      \
    case 2: CALL(2); break;      \
    case 3: CALL(3); break;      \
    case 4: CALL(4); break;      \
    case 5: CALL(5); break;      \
    case 6: CALL(6); break;      \
    case 7: CALL(7); break;      \
    default: CALL(8); break;     \
  }

}  // namespace

// ---- host wrappers --------------------------------------------------------------------------------------------------------------------------
void glm_hc_pre_decode(bf16* h, int M, int dim, const float* fn, const float* scale, const float* base, const bf16* norm_w, float rms_eps,
                       float hc_eps, int iters, float* mixes, float* rsq, float* pre, float* post, float* comb, bf16* x, bf16* xn, int* sync,
                       cudaStream_t st, const bf16* post_x) {
  HIVE_CHECK(M >= 1 && M <= 8 && dim == 4096, "glm_hc_pre_decode shape (M ≤ 8, dim 4096)");
  launch_pdl(hc_pre_kernel<4096>, dim3(M * (kMix + 2)), dim3(256), 0, st, h, post_x, fn, scale, base, norm_w, M, rms_eps, hc_eps, iters, mixes, rsq, pre, post,
             comb, x, xn, sync);
}

void glm_hc_post_decode(const bf16* x, const float* post, const float* comb, int M, int dim, bf16* h, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 8 && dim % 8 == 0, "glm_hc_post_decode shape");
  const int n = M * dim / 8;
  launch_pdl(hc_post_kernel, dim3((n + 255) / 256), dim3(256), 0, st, x, post, comb, M, dim, h);
}

void glm_gemv_bf16_multi(const GemvSeg* segs, int nseg, int M, cudaStream_t st) {
  HIVE_CHECK(nseg >= 1 && nseg <= kGemvMaxSeg && M >= 1 && M <= 8, "glm_gemv_bf16_multi: nseg / M");
  SegTab T{};
  T.nseg = nseg;
  int blocks = 0;
  for (int s = 0; s < nseg; ++s) {
    const GemvSeg& G = segs[s];
    HIVE_CHECK(G.K % 32 == 0 && G.N > 0 && G.ldx % 8 == 0 && ((uintptr_t)G.W % 16) == 0 && ((uintptr_t)G.x % 16) == 0, "glm_gemv_bf16_multi: shape/alignment");
    int wt = 1;  // warps per tile: K parts of ≥ 512 columns, at most 8
    while (wt < 8 && G.K % (2 * wt * 32) == 0 && G.K / (2 * wt) >= 512) wt *= 2;
    T.s[s] = G;
    T.wt[s] = wt;
    T.blk0[s] = blocks;
    const int tiles = (G.N + 15) / 16, tpb = 8 / wt;
    blocks += (tiles + tpb - 1) / tpb;
  }
  T.blk0[nseg] = blocks;
  launch_pdl(gemv_tc_kernel, dim3(blocks), dim3(256), 0, st, T, M);
}

size_t glm_router_logits_floats(int M, int E, int K) { return (size_t)std::max(1, K / kRouterKP) * M * E; }

void glm_router_decode(const bf16* x, int ldx, int M, const bf16* gate_w, const float* bias, int E, int K, int k, float scale, float* logits, int32_t* ids,
                       float* w, int32_t* ids_host, float* w_host, cudaStream_t st, const GlmCachePrior* cache_prior) {
  HIVE_CHECK(M >= 1 && M <= 8 && E >= 1 && E <= kRouterMaxE && k >= 1 && k <= 16 && K % kRouterKP == 0 && ldx % 8 == 0, "glm_router_decode shape");
  const int P = K / kRouterKP;
  launch_pdl(router_logits_kernel, dim3((E + 15) / 16 * P), dim3(32), 0, st, x, ldx, M, gate_w, E, K, logits);
  launch_pdl(router_topk_kernel, dim3(1), dim3(256), (size_t)2 * M * E * sizeof(float), st, logits, M, E, P, bias, k, scale, ids, w, ids_host, w_host,
             cache_prior ? *cache_prior : GlmCachePrior{});
}

size_t glm_dense_nvfp4_ws_bytes(int M, int I) { return (size_t)M * I * 2; }

void glm_dense_nvfp4_decode(const Nvfp4Mat& gate, const Nvfp4Mat& up, const Nvfp4Mat& down, const bf16* x, int ldx, int M, float limit, bf16* y_ws,
                            bf16* out, int ldo, cudaStream_t st) {
  const int H = gate.K, I = gate.N;
  HIVE_CHECK(M >= 1 && M <= 8 && up.N == I && up.K == H && down.N == H && down.K == I && H % 1024 == 0 && I % 1024 == 0 && ldx % 8 == 0,
             "glm_dense_nvfp4_decode shape");
  const int kb = H / 1024, wpl = I / 1024;
  if (kb == 4) launch_pdl(dense_gu_kernel<4>, dim3(I / 8), dim3(256), 0, st, gate, up, x, ldx, M, limit, y_ws);
  else HIVE_CHECK(false, "glm_dense_nvfp4_decode: add a dense_gu_kernel instantiation for this H");
  if (wpl == 12) launch_pdl(dense_down_kernel<16, 6, 3>, dim3(H / 16), dim3(512), 0, st, down, (const bf16*)y_ws, M, out, ldo);
  else if (wpl == 4) launch_pdl(dense_down_kernel<8, 4, 4>, dim3(H / 16), dim3(256), 0, st, down, (const bf16*)y_ws, M, out, ldo);
  else HIVE_CHECK(false, "glm_dense_nvfp4_decode: add a dense_down_kernel instantiation for this I");
}

void glm_shared_fp8_decode(const Fp8BMat& gate, const Fp8BMat& up, const Fp8BMat& down, const bf16* x, int ldx, int M, float limit, bf16* y_ws, bf16* out,
                           int ldo, cudaStream_t st) {
  const int H = gate.K, I = gate.N;
  HIVE_CHECK(M >= 1 && M <= 8 && up.N == I && up.K == H && down.N == H && down.K == I && ldx % 8 == 0, "glm_shared_fp8_decode shape");
  HIVE_CHECK(H == 4096 && I == 2048, "glm_shared_fp8_decode: instantiated for H 4096 · I 2048");
#define HIVE_SHGU(MM) launch_pdl(sh_gu_kernel<MM, 8>, dim3((I + 7) / 8), dim3(256), 0, st, gate, up, x, ldx, limit, y_ws)
  HIVE_DISPATCH_M(M, HIVE_SHGU)
#undef HIVE_SHGU
#define HIVE_SHD(MM) launch_pdl(sh_down_kernel<MM, 4>, dim3((H + 7) / 8), dim3(256), 0, st, down, (const bf16*)y_ws, out, ldo)
  HIVE_DISPATCH_M(M, HIVE_SHD)
#undef HIVE_SHD
}

}  // namespace hive::glm
