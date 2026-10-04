// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// KDA (Kimi Delta Attention) kernels for GLM-5.3-Flash — see hive/glm/kda.h for the math contract.
//
// Work split. The delta-rule recurrence is independent per value column v: column v of S only ever reads k, q, g,
// beta and v[v]. So a warp owns 8 columns × all 128 key rows, lane = (ksub = lane>>3, col = lane&7) and keeps rows
// ksub + 4j (j = 0..31) of its column in 32 registers. Reductions over k are then 32 in-thread FMAs plus two shuffles
// (xor 8, xor 16), with no block barrier per token.
//   o = Σ_k (S'+k·δ)·q = Σ_k S'·q + δ·(k·q)  where S' = decayed state, so kv and the pre-update part of o are reduced
//   together, and k·q (per token and head) comes from the prep stage.
//
// Prefill (kda_forward_seq), per internal chunk of up to kChunk tokens:
//   kda_prep_kernel   (token, head) blocks: conv + SiLU + bf16 round, l2norm, decay factor exp(g), beta, k·q →
//                     workspace (q/k/exp(g) stored in the lane-permuted row order, so they stage with 16-byte copies)
//   kda_recur_kernel  (column tile, head) blocks: sequential recurrence over the chunk, state in registers,
//                     inputs double-buffered into shared memory with cp.async → o (bf16) in workspace
//   kda_norm_kernel   one warp per (token, head): RMSNorm + o_norm_w + sigmoid(gate) → out
//   kda_conv_state_kernel (after the last chunk) — conv state = last 3 pre-conv inputs.
// Decode (kda_decode_batch): one 512-thread block per (row, head) runs the same prep, step and norm device functions,
// so a decoded token is bit-identical to a T = 1 prefill.
//
// Negative-control builds (tests only): -DKDA_NEGCTRL=1 drops the l2norm eps, =2 applies the decay after the update,
// =3 skips the HF bf16 roundings of the conv output, beta and o.
#include "hive/glm/kda.h"

#include <algorithm>

#include "hive/common.h"

#ifndef KDA_NEGCTRL
#define KDA_NEGCTRL 0
#endif

namespace hive::glm {
namespace {

constexpr int H = kKdaHeads;
constexpr int D = kKdaHeadDim;
constexpr int W = kKdaWidth;
constexpr int C = kKdaConvChannels;
constexpr int kChunk = 1024;        // tokens per internal prefill chunk (bounds the workspace)
constexpr int kSB = 16;             // tokens per shared-memory stage in the recurrence
constexpr int kGS = 36;             // padded stride of one ksub group (32 rows + 4) — keeps the 4 groups on distinct banks
constexpr int kRowS = 4 * kGS;      // 144 floats per token for q / k / exp(g)
constexpr int kRecurWarps = 8;      // warps per recurrence block (8 columns each) — 2 column tiles per head
// (measured per 1024-token chunk on the RTX PRO 6000: 1/2/4/8/16 warps = 2045/750/505/300/504 us; sm_120 has ~100 KB
//  of shared memory per SM, so kSB = 32 does not fit with 8 warps and kSB = 8 was slower)
constexpr float kQScale = 0.08838834764831845f;   // 1/sqrt(128)
constexpr float kLowerBound = -5.0f;
constexpr float kL2Eps = 1e-6f;
constexpr float kNormEps = 1e-5f;

__device__ __forceinline__ float rbf(float x) {
#if KDA_NEGCTRL == 3
  return x;
#else
  return __bfloat162float(__float2bfloat16(x));
#endif
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
// Row r of a head → its slot in the lane-permuted order (group r&3, position r>>2).
__device__ __forceinline__ int perm_row(int r) { return (r & 3) * 32 + (r >> 2); }
__device__ __forceinline__ int perm_row_padded(int r) { return (r & 3) * kGS + (r >> 2); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Causal conv (4 taps) of one channel: x0 = token t-3 … x3 = token t. Same expression in prefill and decode.
__device__ __forceinline__ float conv_silu(const float* w4, float x0, float x1, float x2, float x3) {
  float y = w4[0] * x0;
  y = fmaf(w4[1], x1, y);
  y = fmaf(w4[2], x2, y);
  y = fmaf(w4[3], x3, y);
  return rbf(y / (1.0f + expf(-y)));
}

// Per-head prep shared by prefill and decode. Called by ALL threads of the block (it contains __syncthreads); threads
// 0..127 own channel i = threadIdx.x of the head and pass their conv outputs, the others pass anything.
// red: __shared__ float[12]. Results for threads < 128: qn, kn (normalized, q scaled), eg = exp(g); thread 0: beta, kq.
struct PrepOut { float qn, kn, eg, beta, kq; };
__device__ __forceinline__ PrepOut prep_head(float xq, float xk, float gp, float dtb, float A_log, float beta_logit,
                                             float* red) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float sq = xq * xq, sk = xk * xk, qk = xq * xk;
  sq = warp_sum(sq); sk = warp_sum(sk); qk = warp_sum(qk);
  if (warp < 4 && lane == 0) { red[warp] = sq; red[4 + warp] = sk; red[8 + warp] = qk; }
  __syncthreads();
  const float tq = ((red[0] + red[1]) + red[2]) + red[3];
  const float tk = ((red[4] + red[5]) + red[6]) + red[7];
  const float tqk = ((red[8] + red[9]) + red[10]) + red[11];
  __syncthreads();   // red may be reused by the caller
#if KDA_NEGCTRL == 1
  const float nq = sqrtf(tq), nk = sqrtf(tk);
#else
  const float nq = sqrtf(tq + kL2Eps), nk = sqrtf(tk + kL2Eps);
#endif
  PrepOut r;
  r.qn = (xq / nq) * kQScale;
  r.kn = xk / nk;
  r.kq = ((tqk / nq) / nk) * kQScale;
  const float g = kLowerBound * sigmoidf_(expf(A_log) * (gp + dtb));
  r.eg = expf(g);
  r.beta = rbf(sigmoidf_(beta_logit));
  return r;
}

// One recurrence step for the calling lane (rows ksub+4j of one column). q/k/e point at the lane's ksub group
// (32 contiguous floats, 16-byte aligned). Returns o for the column (identical on all lanes of the column).
__device__ __forceinline__ float kda_step(float (&s)[32], const float* __restrict__ q, const float* __restrict__ k,
                                          const float* __restrict__ e, float v, float beta, float kq) {
  float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f, so0 = 0.f, so1 = 0.f, so2 = 0.f, so3 = 0.f;
#pragma unroll
  for (int j4 = 0; j4 < 8; ++j4) {
    const float4 e4 = *reinterpret_cast<const float4*>(e + 4 * j4);
    const float4 k4 = *reinterpret_cast<const float4*>(k + 4 * j4);
    const float4 q4 = *reinterpret_cast<const float4*>(q + 4 * j4);
#if KDA_NEGCTRL == 2
    // Negative control: decay after the update — kv / o use the undecayed state here.
    (void)e4;
#else
    s[4 * j4 + 0] *= e4.x; s[4 * j4 + 1] *= e4.y; s[4 * j4 + 2] *= e4.z; s[4 * j4 + 3] *= e4.w;
#endif
    kv0 = fmaf(s[4 * j4 + 0], k4.x, kv0); so0 = fmaf(s[4 * j4 + 0], q4.x, so0);
    kv1 = fmaf(s[4 * j4 + 1], k4.y, kv1); so1 = fmaf(s[4 * j4 + 1], q4.y, so1);
    kv2 = fmaf(s[4 * j4 + 2], k4.z, kv2); so2 = fmaf(s[4 * j4 + 2], q4.z, so2);
    kv3 = fmaf(s[4 * j4 + 3], k4.w, kv3); so3 = fmaf(s[4 * j4 + 3], q4.w, so3);
  }
  float kv = (kv0 + kv1) + (kv2 + kv3), so = (so0 + so1) + (so2 + so3);
  kv += __shfl_xor_sync(0xffffffffu, kv, 8);
  so += __shfl_xor_sync(0xffffffffu, so, 8);
  kv += __shfl_xor_sync(0xffffffffu, kv, 16);
  so += __shfl_xor_sync(0xffffffffu, so, 16);
  const float delta = (v - kv) * beta;
#pragma unroll
  for (int j4 = 0; j4 < 8; ++j4) {
    const float4 k4 = *reinterpret_cast<const float4*>(k + 4 * j4);
    s[4 * j4 + 0] = fmaf(k4.x, delta, s[4 * j4 + 0]);
    s[4 * j4 + 1] = fmaf(k4.y, delta, s[4 * j4 + 1]);
    s[4 * j4 + 2] = fmaf(k4.z, delta, s[4 * j4 + 2]);
    s[4 * j4 + 3] = fmaf(k4.w, delta, s[4 * j4 + 3]);
  }
#if KDA_NEGCTRL == 2
  float o0 = 0.f;
#pragma unroll
  for (int j4 = 0; j4 < 8; ++j4) {
    const float4 e4 = *reinterpret_cast<const float4*>(e + 4 * j4);
    const float4 q4 = *reinterpret_cast<const float4*>(q + 4 * j4);
    s[4 * j4 + 0] *= e4.x; s[4 * j4 + 1] *= e4.y; s[4 * j4 + 2] *= e4.z; s[4 * j4 + 3] *= e4.w;
    o0 = fmaf(s[4 * j4 + 0], q4.x, o0); o0 = fmaf(s[4 * j4 + 1], q4.y, o0);
    o0 = fmaf(s[4 * j4 + 2], q4.z, o0); o0 = fmaf(s[4 * j4 + 3], q4.w, o0);
  }
  o0 += __shfl_xor_sync(0xffffffffu, o0, 8);
  o0 += __shfl_xor_sync(0xffffffffu, o0, 16);
  (void)so; (void)kq;
  return o0;
#else
  return fmaf(delta, kq, so);
#endif
}

// RMSNorm(128) + weight + sigmoid(gate) for one (token, head), one warp. o: 128 bf16-rounded values (lane + 32j).
__device__ __forceinline__ void norm_gate_warp(const float (&o)[4], const bf16* __restrict__ gate,
                                               const bf16* __restrict__ w, bf16* __restrict__ out, int lane) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) ss = fmaf(o[j], o[j], ss);
  ss = warp_sum(ss);
  const float r = 1.0f / sqrtf(ss * (1.0f / D) + kNormEps);
  float gv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) gv[j] = __bfloat162float(gate[lane + 32 * j]);   // read before out (may alias)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float y = __bfloat162float(w[lane + 32 * j]) * (o[j] * r);
    out[lane + 32 * j] = __float2bfloat16(y * sigmoidf_(gv[j]));
  }
}

// ---- prefill -------------------------------------------------------------------------------------------------------
struct PrefillArgs {
  const float* conv_w; const float* A_log; const float* dt_bias; const bf16* o_norm_w;
  const bf16* qkv_pre; const bf16* g_pre; const bf16* beta_logit; const bf16* gate;
  const float* conv_state; float* state; bf16* out;
  // workspace (chunk-local token index)
  float* qp; float* kp; float* ep;   // [n][64][128] lane-permuted rows
  float* vw;                         // [n][8192]
  float2* bk;                        // [n][64] (beta, k·q)
  bf16* ow;                          // [n][8192]
  int t0, n;
};

__global__ void __launch_bounds__(128) kda_prep_kernel(PrefillArgs a) {
  __shared__ float red[12];
  const int tl = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  const int t = a.t0 + tl;
  float y[3];
#pragma unroll
  for (int part = 0; part < 3; ++part) {
    const int c = part * W + h * D + i;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int tt = t - 3 + j;
      x[j] = tt >= 0 ? __bfloat162float(a.qkv_pre[(size_t)tt * C + c]) : a.conv_state[(size_t)(t + j) * C + c];
    }
    y[part] = conv_silu(a.conv_w + (size_t)c * 4, x[0], x[1], x[2], x[3]);
  }
  const float gp = __bfloat162float(a.g_pre[(size_t)t * W + h * D + i]);
  const PrepOut r = prep_head(y[0], y[1], gp, a.dt_bias[h * D + i], a.A_log[h],
                              __bfloat162float(a.beta_logit[(size_t)t * H + h]), red);
  const size_t base = ((size_t)tl * H + h) * D;
  const int pr = perm_row(i);
  a.qp[base + pr] = r.qn;
  a.kp[base + pr] = r.kn;
  a.ep[base + pr] = r.eg;
  a.vw[(size_t)tl * W + h * D + i] = y[2];
  if (i == 0) a.bk[(size_t)tl * H + h] = make_float2(r.beta, r.kq);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int WARPS>
struct RecurSmem {
  static constexpr int kCols = 8 * WARPS;
  static constexpr int kStageFloats = kSB * (3 * kRowS + kCols + 2);   // q, k, e, v tile, (beta, kq)
  static constexpr size_t kBytes = 2 * (size_t)kStageFloats * sizeof(float);
};

template <int WARPS>
__global__ void __launch_bounds__(32 * WARPS) kda_recur_kernel(PrefillArgs a) {
  constexpr int kCols = 8 * WARPS;
  constexpr int kStage = RecurSmem<WARPS>::kStageFloats;
  extern __shared__ __align__(16) float sm[];
  const int h = blockIdx.y, tile = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ksub = lane >> 3, colw = warp * 8 + (lane & 7), col = tile * kCols + colw;
  float* st = a.state + (size_t)h * D * D;

  float s[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) s[j] = st[(size_t)(ksub + 4 * j) * D + col];

  // Stage layout: sq[kSB][144] | sk | se | sv[kSB][kCols] | sbk[kSB] (float2)
  auto issue = [&](int b, int buf) {
    float* base = sm + (size_t)buf * kStage;
    const int tb = b * kSB, cnt = min(kSB, a.n - tb);
    // q/k/e: per token 3 arrays × 4 groups × 8 chunks of 16 B
    for (int idx = tid; idx < cnt * 96; idx += 32 * WARPS) {
      const int tok = idx / 96, rem = idx % 96, arr = rem >> 5, g = (rem >> 3) & 3, ch = rem & 7;
      const float* src = (arr == 0 ? a.qp : arr == 1 ? a.kp : a.ep) + ((size_t)(tb + tok) * H + h) * D + g * 32 + ch * 4;
      float* dst = base + (size_t)arr * kSB * kRowS + tok * kRowS + g * kGS + ch * 4;
      cp_async16(dst, src);
    }
    constexpr int vch = kCols / 4;
    for (int idx = tid; idx < cnt * vch; idx += 32 * WARPS) {
      const int tok = idx / vch, ch = idx % vch;
      cp_async16(base + 3 * kSB * kRowS + tok * kCols + ch * 4,
                 a.vw + (size_t)(tb + tok) * W + h * D + tile * kCols + ch * 4);
    }
    for (int idx = tid; idx < cnt; idx += 32 * WARPS)
      cp_async8(base + 3 * kSB * kRowS + kSB * kCols + 2 * idx, a.bk + (size_t)(tb + idx) * H + h);
  };

  const int nb = (a.n + kSB - 1) / kSB;
  issue(0, 0);
  cp_async_commit();
  for (int b = 0; b < nb; ++b) {
    if (b + 1 < nb) issue(b + 1, (b + 1) & 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const float* base = sm + (size_t)(b & 1) * kStage;
    const int tb = b * kSB, cnt = min(kSB, a.n - tb);
    for (int tok = 0; tok < cnt; ++tok) {
      const float* q = base + tok * kRowS + ksub * kGS;
      const float* k = q + kSB * kRowS;
      const float* e = k + kSB * kRowS;
      const float v = base[3 * kSB * kRowS + tok * kCols + colw];
      const float2 bk = *reinterpret_cast<const float2*>(base + 3 * kSB * kRowS + kSB * kCols + 2 * tok);
      const float o = kda_step(s, q, k, e, v, bk.x, bk.y);
      if (ksub == 0) a.ow[(size_t)(tb + tok) * W + h * D + col] = __float2bfloat16(o);
    }
    __syncthreads();   // the buffer is refilled by the next iteration's issue
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) st[(size_t)(ksub + 4 * j) * D + col] = s[j];
}

__global__ void __launch_bounds__(256) kda_norm_kernel(PrefillArgs a) {
  const int idx = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (idx >= a.n * H) return;
  const int tl = idx / H, h = idx % H;
  const bf16* ow = a.ow + (size_t)tl * W + h * D;
  float o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) o[j] = rbf(__bfloat162float(ow[lane + 32 * j]));
  const size_t g = (size_t)(a.t0 + tl) * W + h * D;
  norm_gate_warp(o, a.gate + g, a.o_norm_w, a.out + g, lane);
}

// conv_state ← last 3 pre-conv inputs after T tokens (old slot s holds token s-3 relative to the call start).
__global__ void kda_conv_state_kernel(const bf16* __restrict__ qkv_pre, float* __restrict__ cs, int T) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float nv[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int tt = T - 3 + j;
    nv[j] = tt >= 0 ? __bfloat162float(qkv_pre[(size_t)tt * C + c]) : cs[(size_t)(T + j) * C + c];
  }
#pragma unroll
  for (int j = 0; j < 3; ++j) cs[(size_t)j * C + c] = nv[j];
}

// ---- decode --------------------------------------------------------------------------------------------------------
struct DecodeArgs {
  const float* conv_w; const float* A_log; const float* dt_bias; const bf16* o_norm_w;
  const bf16* qkv_pre; const bf16* g_pre; const bf16* beta_logit; const bf16* gate;
  float* const* conv_states; float* const* states; bf16* out;
};

__global__ void __launch_bounds__(512) kda_decode_kernel(DecodeArgs a) {
  __shared__ float red[12];
  __shared__ float xs[3][D];
  __shared__ __align__(16) float sq[kRowS], sk[kRowS], se[kRowS];
  __shared__ float so[D];
  __shared__ float2 sbk;
  const int m = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float* cs = a.conv_states[m];

  // Issue the state loads first so the 64 KB per head is in flight while the prep runs.
  const int ksub = lane >> 3, col = warp * 8 + (lane & 7);
  float* st = a.states[m] + (size_t)h * D * D;
  float s[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) s[j] = st[(size_t)(ksub + 4 * j) * D + col];

  if (tid < 3 * D) {
    const int part = tid / D, i = tid % D, c = part * W + h * D + i;
    const float x0 = cs[c], x1 = cs[(size_t)C + c], x2 = cs[(size_t)2 * C + c];
    const float x3 = __bfloat162float(a.qkv_pre[(size_t)m * C + c]);
    xs[part][i] = conv_silu(a.conv_w + (size_t)c * 4, x0, x1, x2, x3);
    cs[c] = x1; cs[(size_t)C + c] = x2; cs[(size_t)2 * C + c] = x3;   // each thread owns its channel
  }
  __syncthreads();
  const int i = tid & (D - 1);
  const PrepOut r = prep_head(xs[0][i], xs[1][i], __bfloat162float(a.g_pre[(size_t)m * W + h * D + i]),
                              a.dt_bias[h * D + i], a.A_log[h], __bfloat162float(a.beta_logit[(size_t)m * H + h]), red);
  if (tid < D) {
    const int pr = perm_row_padded(tid);
    sq[pr] = r.qn; sk[pr] = r.kn; se[pr] = r.eg;
    if (tid == 0) sbk = make_float2(r.beta, r.kq);
  }
  __syncthreads();

  const float2 bk = sbk;
  const float o = kda_step(s, sq + ksub * kGS, sk + ksub * kGS, se + ksub * kGS, xs[2][col], bk.x, bk.y);
#pragma unroll
  for (int j = 0; j < 32; ++j) st[(size_t)(ksub + 4 * j) * D + col] = s[j];
  if (ksub == 0) so[col] = rbf(o);
  __syncthreads();
  if (warp == 0) {
    float ov[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) ov[j] = so[lane + 32 * j];
    const size_t g = (size_t)m * W + h * D;
    norm_gate_warp(ov, a.gate + g, a.o_norm_w, a.out + g, lane);
  }
}

// Verify rows: R consecutive tokens of ONE sequence in one launch. Same per-row arithmetic as kda_decode_kernel (bit-identical to R
//   calls of it in sequence), but the state stays in registers between rows: it is read once, written once at the end, and after each
//   row but the last its value is also stored to snap_state[r] / the conv window to snap_conv[r] (for rollback to that row).
struct DecodeRowsArgs {
  DecodeArgs d;
  float* const* snap_conv; float* const* snap_state;
  int R;
};

__global__ void __launch_bounds__(512) kda_decode_rows_kernel(DecodeRowsArgs ra) {
  const DecodeArgs& a = ra.d;
  __shared__ float red[12];
  __shared__ float xs[3][D];
  __shared__ __align__(16) float sq[kRowS], sk[kRowS], se[kRowS];
  __shared__ float so[D];
  __shared__ float2 sbk;
  const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float* cs = a.conv_states[0];
  const int ksub = lane >> 3, col = warp * 8 + (lane & 7);
  float* st = a.states[0] + (size_t)h * D * D;
  float s[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) s[j] = st[(size_t)(ksub + 4 * j) * D + col];
  for (int r = 0; r < ra.R; ++r) {
    const bool snap = r + 1 < ra.R;
    if (tid < 3 * D) {
      const int part = tid / D, i = tid % D, c = part * W + h * D + i;
      const float x0 = cs[c], x1 = cs[(size_t)C + c], x2 = cs[(size_t)2 * C + c];
      const float x3 = __bfloat162float(a.qkv_pre[(size_t)r * C + c]);
      xs[part][i] = conv_silu(a.conv_w + (size_t)c * 4, x0, x1, x2, x3);
      cs[c] = x1; cs[(size_t)C + c] = x2; cs[(size_t)2 * C + c] = x3;
      if (snap) { float* sc = ra.snap_conv[r]; sc[c] = x1; sc[(size_t)C + c] = x2; sc[(size_t)2 * C + c] = x3; }
    }
    __syncthreads();
    const int i = tid & (D - 1);
    const PrepOut po = prep_head(xs[0][i], xs[1][i], __bfloat162float(a.g_pre[(size_t)r * W + h * D + i]),
                                 a.dt_bias[h * D + i], a.A_log[h], __bfloat162float(a.beta_logit[(size_t)r * H + h]), red);
    if (tid < D) {
      const int pr = perm_row_padded(tid);
      sq[pr] = po.qn; sk[pr] = po.kn; se[pr] = po.eg;
      if (tid == 0) sbk = make_float2(po.beta, po.kq);
    }
    __syncthreads();
    const float2 bk = sbk;
    const float o = kda_step(s, sq + ksub * kGS, sk + ksub * kGS, se + ksub * kGS, xs[2][col], bk.x, bk.y);
    if (snap) {
      float* sn = ra.snap_state[r] + (size_t)h * D * D;
#pragma unroll
      for (int j = 0; j < 32; ++j) sn[(size_t)(ksub + 4 * j) * D + col] = s[j];
    }
    if (ksub == 0) so[col] = rbf(o);
    __syncthreads();
    if (warp == 0) {
      float ov[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) ov[j] = so[lane + 32 * j];
      const size_t g = (size_t)r * W + h * D;
      norm_gate_warp(ov, a.gate + g, a.o_norm_w, a.out + g, lane);
    }
    __syncthreads();  // xs, so, sq/sk/se and red are rewritten by the next row
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) st[(size_t)(ksub + 4 * j) * D + col] = s[j];
}

struct WsLayout { size_t qp, kp, ep, vw, bk, ow, total; };
WsLayout ws_layout(int n) {
  WsLayout L{};
  size_t off = 0;
  auto take = [&](size_t bytes) { const size_t o = off; off = align_up(off + bytes, 256); return o; };
  const size_t f = (size_t)n * W * sizeof(float);
  L.qp = take(f); L.kp = take(f); L.ep = take(f); L.vw = take(f);
  L.bk = take((size_t)n * H * sizeof(float2));
  L.ow = take((size_t)n * W * sizeof(bf16));
  L.total = off;
  return L;
}

}  // namespace

size_t kda_workspace_bytes(int T) { return T <= 0 ? 0 : ws_layout(std::min(T, kChunk)).total; }

void kda_forward_seq(const KdaParams& p, const bf16* qkv_pre, const bf16* g_pre, const bf16* beta_logit,
                     const bf16* gate, int T, float* conv_state, float* state, bf16* out, void* workspace,
                     size_t ws_bytes, cudaStream_t st) {
  if (T <= 0) return;
  HIVE_CHECK(ws_bytes >= kda_workspace_bytes(T), "kda_forward_seq: workspace too small");
  HIVE_CHECK(((uintptr_t)workspace & 255) == 0, "kda_forward_seq: workspace must be 256-byte aligned");
  constexpr size_t kSmem = RecurSmem<kRecurWarps>::kBytes;
  static bool attr_set = false;   // per-process; the attribute is per function and device-independent here
  if (!attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(kda_recur_kernel<kRecurWarps>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kSmem));
    attr_set = true;
  }
  const WsLayout L = ws_layout(std::min(T, kChunk));
  char* w = static_cast<char*>(workspace);
  PrefillArgs a{p.conv_w, p.A_log, p.dt_bias, p.o_norm_w, qkv_pre, g_pre, beta_logit, gate, conv_state, state, out,
                reinterpret_cast<float*>(w + L.qp), reinterpret_cast<float*>(w + L.kp),
                reinterpret_cast<float*>(w + L.ep), reinterpret_cast<float*>(w + L.vw),
                reinterpret_cast<float2*>(w + L.bk), reinterpret_cast<bf16*>(w + L.ow), 0, 0};
  for (int t0 = 0; t0 < T; t0 += kChunk) {
    a.t0 = t0;
    a.n = std::min(kChunk, T - t0);
    kda_prep_kernel<<<dim3(a.n, H), D, 0, st>>>(a);
    kda_recur_kernel<kRecurWarps><<<dim3(D / (8 * kRecurWarps), H), 32 * kRecurWarps, kSmem, st>>>(a);
    kda_norm_kernel<<<(a.n * H + 7) / 8, 256, 0, st>>>(a);
  }
  kda_conv_state_kernel<<<(C + 255) / 256, 256, 0, st>>>(qkv_pre, conv_state, T);
  CUDA_CHECK(cudaGetLastError());
}

void kda_decode_batch(const KdaParams& p, const bf16* qkv_pre, const bf16* g_pre, const bf16* beta_logit,
                      const bf16* gate, int M, float* const* conv_states, float* const* states, bf16* out,
                      cudaStream_t st) {
  if (M <= 0) return;
  DecodeArgs a{p.conv_w, p.A_log, p.dt_bias, p.o_norm_w, qkv_pre, g_pre, beta_logit, gate, conv_states, states, out};
  kda_decode_kernel<<<dim3(M, H), 512, 0, st>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

void kda_decode_rows(const KdaParams& p, const bf16* qkv_pre, const bf16* g_pre, const bf16* beta_logit, const bf16* gate, int R,
                     float* const* conv_state, float* const* state, float* const* snap_conv, float* const* snap_state, bf16* out, cudaStream_t st) {
  if (R <= 0) return;
  DecodeRowsArgs a{{p.conv_w, p.A_log, p.dt_bias, p.o_norm_w, qkv_pre, g_pre, beta_logit, gate, conv_state, state, out}, snap_conv, snap_state, R};
  kda_decode_rows_kernel<<<dim3(1, H), 512, 0, st>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::glm
