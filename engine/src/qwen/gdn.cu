// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gated DeltaNet kernels for Qwen3.8-Flash-Next — see hive/qwen/gdn.h for the math contract. Adapted from hive/glm/kda.cu (same work
// split and state layout); what differs is noted where it happens.
//
// Work split. The delta-rule recurrence is independent per value column v of a value head: column v of S reads only q, k, beta, the
// decay and v[v]. A warp owns 8 columns × all 128 key rows, lane = (ksub = lane>>3, col = lane&7), rows ksub + 4j (j = 0..31) in 32
// registers; reductions over k are 32 in-thread FMAs plus two shuffles (xor 8, xor 16).
//   o = Σ_k (S'+k·δ)·q = Σ_k S'·q + δ·(k·q), S' = decayed state — kv and the pre-update part of o are reduced together and k·q comes
//   from the prep stage. The decay is one scalar per value head here (KDA: one per key row).
//
// Prefill (gdn_forward_seq), per internal chunk of up to kChunk tokens:
//   gdn_prep_kernel   (token, 16 key heads + 48 value heads) blocks: key head → conv + SiLU + bf16 round of q and k, l2norm, k·q (lane-
//                     permuted q / k rows to the workspace; k·q into the .z of its three value heads' records); value head → conv of v,
//                     beta, exp(g) (.x / .y of its record)
//   gdn_recur_kernel  (column tile, value head) blocks: sequential recurrence over the chunk, state in registers, inputs double-buffered
//                     into shared memory with cp.async → o (bf16) in workspace
//   gdn_norm_kernel   one warp per (token, value head): RMSNorm · w · sigmoid(z) → out
//   gdn_conv_tail_kernel (after the last chunk) — conv state = last 3 pre-conv inputs.
// Decode (gdn_decode_batch): one 512-thread block per (row, value head) runs the same conv, prep, step and norm device functions (its key
// head's prep is recomputed by each of its three value heads — identical arithmetic), so a decoded token is bit-identical to a T = 1
// prefill. Unlike KDA, three blocks read the same q / k conv channels, so no block writes the conv state: a tail kernel shifts it after
// the step (also for the verify rows, where it writes the per-row conv snapshots).
#include "hive/qwen/gdn.h"

#include <algorithm>
#include <cstdlib>

#include "hive/common.h"
#include "hive/qwen/fn_pdl.h"

namespace hive::qwen {
namespace {

constexpr int HK = kGdnKHeads;
constexpr int HV = kGdnVHeads;
constexpr int D = kGdnDim;
constexpr int C = kGdnConvCh;
constexpr int kKOff = kGdnKWidth;          // k channels start
constexpr int kVOff = 2 * kGdnKWidth;      // v channels start
constexpr int kGroup = HV / HK;            // value heads per key head (3)
constexpr int kChunk = 1024;               // tokens per internal prefill chunk (bounds the workspace)
constexpr int kSB = 16;                    // tokens per shared-memory stage in the recurrence
constexpr int kGS = 36;                    // padded stride of one ksub group (32 rows + 4) — keeps the 4 groups on distinct banks
constexpr int kRowS = 4 * kGS;             // 144 floats per token for q / k
constexpr int kRecurWarps = 4;             // warps per recurrence block (8 columns each) — 4 column tiles per value head
// (kept from the KDA measurements on the RTX PRO 6000 — 8 warps were fastest for 64 heads; not re-measured for 48 heads)
constexpr float kQScale = 0.08838834764831845f;   // 1/sqrt(128)
constexpr float kL2Eps = 1e-6f;

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ int perm_row(int r) { return (r & 3) * 32 + (r >> 2); }
__device__ __forceinline__ int perm_row_padded(int r) { return (r & 3) * kGS + (r >> 2); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Causal conv (4 taps) of one channel: x0 = token t-3 … x3 = token t, SiLU, bf16 round. Same expression in prefill and decode.
__device__ __forceinline__ float conv_silu(const float* w4, float x0, float x1, float x2, float x3) {
  float y = w4[0] * x0;
  y = fmaf(w4[1], x1, y);
  y = fmaf(w4[2], x2, y);
  y = fmaf(w4[3], x3, y);
  return rbf(y / (1.0f + expf(-y)));
}
// Pre-conv input of channel c at call-relative token tt (tt < 0: the conv state of the call start, slot tt + 3).
__device__ __forceinline__ float conv_in(const bf16* __restrict__ proj, int ld, const float* __restrict__ cs, int tt, int c) {
  return tt >= 0 ? __bfloat162float(proj[(size_t)tt * ld + c]) : cs[(size_t)(tt + 3) * C + c];
}
// Conv output of channel c at call-relative token t.
__device__ __forceinline__ float conv_at(const float* __restrict__ conv_w, const bf16* __restrict__ proj, int ld, const float* __restrict__ cs,
                                         int t, int c) {
  return conv_silu(conv_w + (size_t)c * 4, conv_in(proj, ld, cs, t - 3, c), conv_in(proj, ld, cs, t - 2, c), conv_in(proj, ld, cs, t - 1, c),
                   conv_in(proj, ld, cs, t, c));
}

// Key-head prep shared by prefill and decode. Called by ALL threads of the block (it contains __syncthreads); threads 0..127 own channel
// i = threadIdx.x of the key head and pass their conv outputs, the others pass anything. red: __shared__ float[12].
// Results for threads < 128: qn (scaled), kn; every thread: kq.
struct PrepK { float qn, kn, kq; };
__device__ __forceinline__ PrepK prep_key(float xq, float xk, float* red) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float sq = xq * xq, sk = xk * xk, qk = xq * xk;
  sq = warp_sum(sq); sk = warp_sum(sk); qk = warp_sum(qk);
  if (warp < 4 && lane == 0) { red[warp] = sq; red[4 + warp] = sk; red[8 + warp] = qk; }
  __syncthreads();
  const float tq = ((red[0] + red[1]) + red[2]) + red[3];
  const float tk = ((red[4] + red[5]) + red[6]) + red[7];
  const float tqk = ((red[8] + red[9]) + red[10]) + red[11];
  __syncthreads();   // red may be reused by the caller
  const float nq = sqrtf(tq + kL2Eps), nk = sqrtf(tk + kL2Eps);
  PrepK r;
  r.qn = (xq / nq) * kQScale;
  r.kn = xk / nk;
  r.kq = ((tqk / nq) / nk) * kQScale;
  return r;
}
// Value-head scalars from the projection row: beta = bf16(sigmoid(b)), decay = exp(−exp(A_log)·softplus(a + dt_bias)).
__device__ __forceinline__ float2 prep_value(const bf16* __restrict__ row, int h, const float* __restrict__ A_log, const float* __restrict__ dt_bias) {
  const float b = __bfloat162float(row[kGdnBOff + h]), a = __bfloat162float(row[kGdnAOff + h]);
  const float g = -expf(A_log[h]) * softplusf_(a + dt_bias[h]);
  return make_float2(rbf(sigmoidf_(b)), expf(g));
}

// One recurrence step for the calling lane (rows ksub+4j of one column). q/k point at the lane's ksub group (32 contiguous floats,
// 16-byte aligned). Returns o for the column (identical on all lanes of the column).
__device__ __forceinline__ float gdn_step(float (&s)[32], const float* __restrict__ q, const float* __restrict__ k, float v, float beta,
                                          float decay, float kq) {
  float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f, so0 = 0.f, so1 = 0.f, so2 = 0.f, so3 = 0.f;
#pragma unroll
  for (int j4 = 0; j4 < 8; ++j4) {
    const float4 k4 = *reinterpret_cast<const float4*>(k + 4 * j4);
    const float4 q4 = *reinterpret_cast<const float4*>(q + 4 * j4);
    s[4 * j4 + 0] *= decay; s[4 * j4 + 1] *= decay; s[4 * j4 + 2] *= decay; s[4 * j4 + 3] *= decay;
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
  return fmaf(delta, kq, so);
}

// RMSNorm(128) · w · sigmoid(z) for one (token, value head), one warp. o: 128 bf16-rounded values (lane + 32j).
__device__ __forceinline__ void norm_gate_warp(const float (&o)[4], const bf16* __restrict__ z, const bf16* __restrict__ w, float eps,
                                               bf16* __restrict__ out, int lane) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) ss = fmaf(o[j], o[j], ss);
  ss = warp_sum(ss);
  const float r = 1.0f / sqrtf(ss * (1.0f / D) + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float y = __bfloat162float(w[lane + 32 * j]) * (o[j] * r);
    out[lane + 32 * j] = __float2bfloat16(y * sigmoidf_(__bfloat162float(z[lane + 32 * j])));
  }
}

// ---- prefill -------------------------------------------------------------------------------------------------------
struct PrefillArgs {
  const float* conv_w; const float* A_log; const float* dt_bias; const bf16* norm_w; float eps;
  const bf16* proj; int ld;
  const float* conv_state; float* state; bf16* out;
  // workspace (chunk-local token index)
  float* qp; float* kp;   // [n][16][128] lane-permuted rows
  float* vw;              // [n][6144]
  float4* rec;            // [n][48] (beta, decay, k·q, 0)
  bf16* ow;               // [n][6144]
  int t0, n;
};

__global__ void __launch_bounds__(128) gdn_prep_kernel(PrefillArgs a) {
  __shared__ float red[12];
  const int tl = blockIdx.x, y = blockIdx.y, i = threadIdx.x;
  const int t = a.t0 + tl;
  if (y < HK) {
    const int kh = y;
    const float xq = conv_at(a.conv_w, a.proj, a.ld, a.conv_state, t, kh * D + i);
    const float xk = conv_at(a.conv_w, a.proj, a.ld, a.conv_state, t, kKOff + kh * D + i);
    const PrepK r = prep_key(xq, xk, red);
    const size_t base = ((size_t)tl * HK + kh) * D;
    a.qp[base + perm_row(i)] = r.qn;
    a.kp[base + perm_row(i)] = r.kn;
    if (i < kGroup) a.rec[(size_t)tl * HV + kh * kGroup + i].z = r.kq;
  } else {
    const int h = y - HK;
    a.vw[(size_t)tl * kGdnVWidth + h * D + i] = conv_at(a.conv_w, a.proj, a.ld, a.conv_state, t, kVOff + h * D + i);
    if (i == 0) {
      const float2 bd = prep_value(a.proj + (size_t)t * a.ld, h, a.A_log, a.dt_bias);
      float4& r = a.rec[(size_t)tl * HV + h];
      r.x = bd.x; r.y = bd.y; r.w = 0.f;
    }
  }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int WARPS>
struct RecurSmem {
  static constexpr int kCols = 8 * WARPS;
  static constexpr int kStageFloats = kSB * (2 * kRowS + kCols + 4);   // q, k, v tile, (beta, decay, kq, 0)
  static constexpr size_t kBytes = 2 * (size_t)kStageFloats * sizeof(float);
};

template <int WARPS>
__global__ void __launch_bounds__(32 * WARPS) gdn_recur_kernel(PrefillArgs a) {
  constexpr int kCols = 8 * WARPS;
  constexpr int kStage = RecurSmem<WARPS>::kStageFloats;
  extern __shared__ __align__(16) float sm[];
  const int h = blockIdx.y, kh = h / kGroup, tile = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ksub = lane >> 3, colw = warp * 8 + (lane & 7), col = tile * kCols + colw;
  float* st = a.state + (size_t)h * D * D;

  float s[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) s[j] = st[(size_t)(ksub + 4 * j) * D + col];

  // Stage layout: sq[kSB][144] | sk | sv[kSB][kCols] | srec[kSB] (float4)
  auto issue = [&](int b, int buf) {
    float* base = sm + (size_t)buf * kStage;
    const int tb = b * kSB, cnt = min(kSB, a.n - tb);
    // q/k: per token 2 arrays × 4 groups × 8 chunks of 16 B
    for (int idx = tid; idx < cnt * 64; idx += 32 * WARPS) {
      const int tok = idx / 64, rem = idx % 64, arr = rem >> 5, g = (rem >> 3) & 3, ch = rem & 7;
      const float* src = (arr == 0 ? a.qp : a.kp) + ((size_t)(tb + tok) * HK + kh) * D + g * 32 + ch * 4;
      float* dst = base + (size_t)arr * kSB * kRowS + tok * kRowS + g * kGS + ch * 4;
      cp_async16(dst, src);
    }
    constexpr int vch = kCols / 4;
    for (int idx = tid; idx < cnt * vch; idx += 32 * WARPS) {
      const int tok = idx / vch, ch = idx % vch;
      cp_async16(base + 2 * kSB * kRowS + tok * kCols + ch * 4, a.vw + (size_t)(tb + tok) * kGdnVWidth + h * D + tile * kCols + ch * 4);
    }
    for (int idx = tid; idx < cnt; idx += 32 * WARPS)
      cp_async16(base + 2 * kSB * kRowS + kSB * kCols + 4 * idx, a.rec + (size_t)(tb + idx) * HV + h);
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
      const float v = base[2 * kSB * kRowS + tok * kCols + colw];
      const float4 r = *reinterpret_cast<const float4*>(base + 2 * kSB * kRowS + kSB * kCols + 4 * tok);
      const float o = gdn_step(s, q, k, v, r.x, r.y, r.z);
      if (ksub == 0) a.ow[(size_t)(tb + tok) * kGdnVWidth + h * D + col] = __float2bfloat16(o);
    }
    __syncthreads();   // the buffer is refilled by the next iteration's issue
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) st[(size_t)(ksub + 4 * j) * D + col] = s[j];
}

__global__ void __launch_bounds__(256) gdn_norm_kernel(PrefillArgs a) {
  const int idx = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (idx >= a.n * HV) return;
  const int tl = idx / HV, h = idx % HV;
  const bf16* ow = a.ow + (size_t)tl * kGdnVWidth + h * D;
  float o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) o[j] = rbf(__bfloat162float(ow[lane + 32 * j]));
  const int t = a.t0 + tl;
  norm_gate_warp(o, a.proj + (size_t)t * a.ld + kGdnZOff + h * D, a.norm_w, a.eps, a.out + (size_t)t * kGdnVWidth + h * D, lane);
}

// Conv state after R rows of one sequence (grid.y = sequences; row block m's projection rows start at proj + m·R·ld): the last 3
//   pre-conv inputs. snap (optional, one sequence): snap[r] for r < R−1 = the window after row r (verify rollback).
//   Each thread owns one channel and reads its old state before writing it.
__global__ void gdn_conv_tail_kernel(const bf16* __restrict__ proj, int ld, int R, float* cs_single, float* const* cs_arr,
                                     float* const* snap, float* const* snap0 = nullptr) {
  pdl_wait();
  const int c = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
  if (c >= C) return;
  float* cs = cs_arr ? cs_arr[m] : cs_single;
  const bf16* pr = proj + (size_t)m * R * ld;
  const float o0 = cs[c], o1 = cs[(size_t)C + c], o2 = cs[(size_t)2 * C + c];
  if (snap0) { float* s0 = *snap0; s0[c] = o0; s0[(size_t)C + c] = o1; s0[(size_t)2 * C + c] = o2; }   // window before row 0 (replay rollback)
  auto in = [&](int tt) { return tt >= 0 ? __bfloat162float(pr[(size_t)tt * ld + c]) : (tt == -3 ? o0 : tt == -2 ? o1 : o2); };
  if (snap)
    for (int r = 0; r + 1 < R; ++r) {
      float* sc = snap[r];
#pragma unroll
      for (int j = 0; j < 3; ++j) sc[(size_t)j * C + c] = in(r - 2 + j);
    }
  const float n0 = in(R - 3), n1 = in(R - 2), n2 = in(R - 1);
  cs[c] = n0; cs[(size_t)C + c] = n1; cs[(size_t)2 * C + c] = n2;
}

// ---- decode --------------------------------------------------------------------------------------------------------
struct DecodeArgs {
  const float* conv_w; const float* A_log; const float* dt_bias; const bf16* norm_w; float eps;
  const bf16* proj; int ld;
  float* const* conv_states; float* const* states; bf16* out;
};

// One token of value head h: conv window from cs (call start) + projection rows (row `t` is the token, rows before it in this call
// precede it). Shared by the batched decode (t = 0, proj = the row) and the verify rows (t = r).
struct DecodeSmem {
  float red[12];
  float xs[3][D];
  __align__(16) float sq[kRowS];
  __align__(16) float sk[kRowS];
  float so[D];
  float4 srec;
};
__device__ __forceinline__ float decode_token(const DecodeArgs& a, DecodeSmem& S, const bf16* __restrict__ proj, const float* __restrict__ cs,
                                              int t, int h, float (&s)[32]) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, kh = h / kGroup;
  const int ksub = lane >> 3, col = warp * 8 + (lane & 7);
  if (tid < 3 * D) {
    const int part = tid / D, i = tid % D;
    const int c = part == 0 ? kh * D + i : part == 1 ? kKOff + kh * D + i : kVOff + h * D + i;
    S.xs[part][i] = conv_at(a.conv_w, proj, a.ld, cs, t, c);
  }
  __syncthreads();
  const int i = tid & (D - 1);
  const PrepK r = prep_key(S.xs[0][i], S.xs[1][i], S.red);
  if (tid < D) {
    const int pr = perm_row_padded(tid);
    S.sq[pr] = r.qn; S.sk[pr] = r.kn;
    if (tid == 0) {
      const float2 bd = prep_value(proj + (size_t)t * a.ld, h, a.A_log, a.dt_bias);
      S.srec = make_float4(bd.x, bd.y, r.kq, 0.f);
    }
  }
  __syncthreads();
  const float4 rc = S.srec;
  return gdn_step(s, S.sq + ksub * kGS, S.sk + ksub * kGS, S.xs[2][col], rc.x, rc.y, rc.z);
}

__global__ void __launch_bounds__(512) gdn_decode_kernel(DecodeArgs a) {
  pdl_wait();
  __shared__ DecodeSmem S;
  const int m = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ksub = lane >> 3, col = warp * 8 + (lane & 7);
  float* st = a.states[m] + (size_t)h * D * D;
  float s[32];   // state loads first: the 64 KB per head are in flight while the prep runs
#pragma unroll
  for (int j = 0; j < 32; ++j) s[j] = st[(size_t)(ksub + 4 * j) * D + col];
  const bf16* proj = a.proj + (size_t)m * a.ld;
  const float o = decode_token(a, S, proj, a.conv_states[m], 0, h, s);
#pragma unroll
  for (int j = 0; j < 32; ++j) st[(size_t)(ksub + 4 * j) * D + col] = s[j];
  if (ksub == 0) S.so[col] = rbf(o);
  __syncthreads();
  if (warp == 0) {
    float ov[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) ov[j] = S.so[lane + 32 * j];
    norm_gate_warp(ov, proj + kGdnZOff + h * D, a.norm_w, a.eps, a.out + (size_t)m * kGdnVWidth + h * D, lane);
  }
}

// Verify rows: R consecutive tokens of ONE sequence; the state stays in registers between rows (read once, written once), and after
//   each row but the last it is also stored to snap_state[r]. The conv state is untouched here (gdn_conv_tail_kernel follows).
// snap_init (replay rollback): the state BEFORE row 0 goes to snap_state[0] and no per-row snapshot is written; snap_state == null: none
struct DecodeRowsArgs { DecodeArgs d; float* const* snap_state; int R; int snap_init; };

__global__ void __launch_bounds__(512) gdn_decode_rows_kernel(DecodeRowsArgs ra) {
  pdl_wait();
  const DecodeArgs& a = ra.d;
  __shared__ DecodeSmem S;
  const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ksub = lane >> 3, col = warp * 8 + (lane & 7);
  const float* cs = a.conv_states[0];
  float* st = a.states[0] + (size_t)h * D * D;
  float s[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) s[j] = st[(size_t)(ksub + 4 * j) * D + col];
  for (int r = 0; r < ra.R; ++r) {
    const float o = decode_token(a, S, a.proj, cs, r, h, s);
    if (r + 1 < ra.R) {
      float* sn = ra.snap_state[r] + (size_t)h * D * D;
#pragma unroll
      for (int j = 0; j < 32; ++j) sn[(size_t)(ksub + 4 * j) * D + col] = s[j];
    }
    if (ksub == 0) S.so[col] = rbf(o);
    __syncthreads();
    if (warp == 0) {
      float ov[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) ov[j] = S.so[lane + 32 * j];
      norm_gate_warp(ov, a.proj + (size_t)r * a.ld + kGdnZOff + h * D, a.norm_w, a.eps, a.out + (size_t)r * kGdnVWidth + h * D, lane);
    }
    __syncthreads();  // xs, so, sq/sk, srec and red are rewritten by the next row
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) st[(size_t)(ksub + 4 * j) * D + col] = s[j];
}

// Verify rows, prep hoisted (R ≤ kPrepRows): the conv outputs, key norms and value scalars of every row do not depend on the state, so they
//   are computed for all rows first (4 rows at a time, 128 threads per row — the same reductions and order as prep_key); then the R
//   recurrence steps run back to back from shared memory with no block barrier between rows, and the R output norms run one warp per row.
//   Same arithmetic per element as gdn_decode_rows_kernel; 3 barriers in place of 3 per row.
constexpr int kPrepRows = 8;
struct RowsSmem {
  float xs[kPrepRows][3][D];
  __align__(16) float sq[kPrepRows][kRowS];
  __align__(16) float sk[kPrepRows][kRowS];
  float so[kPrepRows][D];
  float red[kPrepRows][12];
  float4 srec[kPrepRows];
};
__device__ __forceinline__ void rows2_body(const DecodeRowsArgs& ra) {
  const DecodeArgs& a = ra.d;
  __shared__ RowsSmem S;
  const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, kh = h / kGroup, R = ra.R;
  const int ksub = lane >> 3, col = warp * 8 + (lane & 7);
  const float* cs = a.conv_states[0];
  float* st = a.states[0] + (size_t)h * D * D;
  float s[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) s[j] = st[(size_t)(ksub + 4 * j) * D + col];
  if (ra.snap_init && ra.snap_state) {
    float* sn = ra.snap_state[0] + (size_t)h * D * D;
#pragma unroll
    for (int j = 0; j < 32; ++j) sn[(size_t)(ksub + 4 * j) * D + col] = s[j];
  }
  for (int e = tid; e < R * 3 * D; e += blockDim.x) {
    const int r = e / (3 * D), part = (e / D) % 3, i = e % D;
    const int c = part == 0 ? kh * D + i : part == 1 ? kKOff + kh * D + i : kVOff + h * D + i;
    S.xs[r][part][i] = conv_at(a.conv_w, a.proj, a.ld, cs, r, c);
  }
  __syncthreads();
  const int i = tid & (D - 1), sub = tid >> 7, wk = (tid >> 5) & 3;   // 4 rows per pass: rows sub, sub + 4
  for (int r = sub; r < R; r += 4) {
    const float xq = S.xs[r][0][i], xk = S.xs[r][1][i];
    float sq = warp_sum(xq * xq), sk = warp_sum(xk * xk), qk = warp_sum(xq * xk);
    if (lane == 0) { S.red[r][wk] = sq; S.red[r][4 + wk] = sk; S.red[r][8 + wk] = qk; }
  }
  __syncthreads();
  for (int r = sub; r < R; r += 4) {
    const float* rd = S.red[r];
    const float tq = ((rd[0] + rd[1]) + rd[2]) + rd[3];
    const float tk = ((rd[4] + rd[5]) + rd[6]) + rd[7];
    const float tqk = ((rd[8] + rd[9]) + rd[10]) + rd[11];
    const float nq = sqrtf(tq + kL2Eps), nk = sqrtf(tk + kL2Eps);
    const int pr = perm_row_padded(i);
    S.sq[r][pr] = (S.xs[r][0][i] / nq) * kQScale;
    S.sk[r][pr] = S.xs[r][1][i] / nk;
    if (i == 0) {
      const float2 bd = prep_value(a.proj + (size_t)r * a.ld, h, a.A_log, a.dt_bias);
      S.srec[r] = make_float4(bd.x, bd.y, ((tqk / nq) / nk) * kQScale, 0.f);
    }
  }
  __syncthreads();
  for (int r = 0; r < R; ++r) {
    const float4 rc = S.srec[r];
    const float o = gdn_step(s, S.sq[r] + ksub * kGS, S.sk[r] + ksub * kGS, S.xs[r][2][col], rc.x, rc.y, rc.z);
    if (r + 1 < R && !ra.snap_init && ra.snap_state) {
      float* sn = ra.snap_state[r] + (size_t)h * D * D;
#pragma unroll
      for (int j = 0; j < 32; ++j) sn[(size_t)(ksub + 4 * j) * D + col] = s[j];
    }
    if (ksub == 0) S.so[r][col] = rbf(o);
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) st[(size_t)(ksub + 4 * j) * D + col] = s[j];
  __syncthreads();
  if (warp < R) {
    float ov[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) ov[j] = S.so[warp][lane + 32 * j];
    norm_gate_warp(ov, a.proj + (size_t)warp * a.ld + kGdnZOff + h * D, a.norm_w, a.eps, a.out + (size_t)warp * kGdnVWidth + h * D, lane);
  }
}

__global__ void __launch_bounds__(512) gdn_decode_rows2_kernel(DecodeRowsArgs ra) {
  pdl_wait();
  rows2_body(ra);
}
// several sequences' verify rows in one launch (block = (part, value head)): the per-part launches used 48 blocks each on 188 SMs —
//   nsys 2026-10-10, 6 sequences × 5 rows: 4.2 ms of gdn_decode_rows2 + 0.7 ms of conv tails in 378 + 378 launches per step
constexpr int kMultiParts = 16;
struct RowsMulti { DecodeRowsArgs a[kMultiParts]; };
__global__ void __launch_bounds__(512) gdn_decode_rows2_multi_kernel(const __grid_constant__ RowsMulti m) {
  pdl_wait();
  rows2_body(m.a[blockIdx.x]);
}
struct TailPart { const bf16* proj; int R; float* const* cs; float* const* snap; float* const* snap0; };
struct TailMulti { TailPart a[kMultiParts]; };
__device__ __forceinline__ void tail_body(const TailPart& t, int ld) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float* cs = t.cs[0];
  const bf16* pr = t.proj;
  const int R = t.R;
  const float o0 = cs[c], o1 = cs[(size_t)C + c], o2 = cs[(size_t)2 * C + c];
  if (t.snap0) { float* s0 = *t.snap0; s0[c] = o0; s0[(size_t)C + c] = o1; s0[(size_t)2 * C + c] = o2; }
  auto in = [&](int tt) { return tt >= 0 ? __bfloat162float(pr[(size_t)tt * ld + c]) : (tt == -3 ? o0 : tt == -2 ? o1 : o2); };
  if (t.snap)
    for (int r = 0; r + 1 < R; ++r) {
      float* sc = t.snap[r];
#pragma unroll
      for (int j = 0; j < 3; ++j) sc[(size_t)j * C + c] = in(r - 2 + j);
    }
  const float n0 = in(R - 3), n1 = in(R - 2), n2 = in(R - 1);
  cs[c] = n0; cs[(size_t)C + c] = n1; cs[(size_t)2 * C + c] = n2;
}
__global__ void gdn_conv_tail_multi_kernel(const __grid_constant__ TailMulti m, int ld) {
  pdl_wait();
  tail_body(m.a[blockIdx.y], ld);
}
// The same launch pair with the parts' (first row, rows) read from device memory (vpt: int pairs): one captured verify graph then serves any
//   split of its rows over the P sequences (the per-part arguments of the kernels above are baked into a graph)
struct RowsDyn { DecodeArgs base; float* const* snap; int snap_stride, replay_from; };
__global__ void __launch_bounds__(512) gdn_decode_rows2_dyn_kernel(const __grid_constant__ RowsDyn m, const int* __restrict__ vpt) {
  pdl_wait();
  const int p = blockIdx.x, r0 = vpt[2 * p], R = vpt[2 * p + 1];
  DecodeArgs a = m.base;
  a.proj += (size_t)r0 * a.ld; a.conv_states += r0; a.states += r0; a.out += (size_t)r0 * kGdnVWidth;
  float* const* sp = m.snap + (size_t)p * m.snap_stride;
  rows2_body(DecodeRowsArgs{a, sp + 8, R, p >= m.replay_from ? 1 : 0});
}
__global__ void gdn_conv_tail_dyn_kernel(const __grid_constant__ RowsDyn m, int ld, const int* __restrict__ vpt) {
  pdl_wait();
  const int p = blockIdx.y, r0 = vpt[2 * p];
  float* const* sp = m.snap + (size_t)p * m.snap_stride;
  tail_body(TailPart{m.base.proj + (size_t)r0 * ld, vpt[2 * p + 1], m.base.conv_states + r0, sp, p >= m.replay_from ? sp + 7 : nullptr}, ld);
}

struct WsLayout { size_t qp, kp, vw, rec, ow, total; };
WsLayout ws_layout(int n) {
  WsLayout L{};
  size_t off = 0;
  auto take = [&](size_t bytes) { const size_t o = off; off = align_up(off + bytes, 256); return o; };
  L.qp = take((size_t)n * kGdnKWidth * sizeof(float));
  L.kp = take((size_t)n * kGdnKWidth * sizeof(float));
  L.vw = take((size_t)n * kGdnVWidth * sizeof(float));
  L.rec = take((size_t)n * HV * sizeof(float4));
  L.ow = take((size_t)n * kGdnVWidth * sizeof(bf16));
  L.total = off;
  return L;
}

}  // namespace

size_t gdn_workspace_bytes(int T) { return T <= 0 ? 0 : ws_layout(std::min(T, kChunk)).total; }

void gdn_forward_seq(const GdnParams& p, const bf16* proj, int ld, int T, float* conv_state, float* state, bf16* out, void* workspace,
                     size_t ws_bytes, cudaStream_t st) {
  if (T <= 0) return;
  HIVE_CHECK(ws_bytes >= gdn_workspace_bytes(T), "gdn_forward_seq: workspace too small");
  HIVE_CHECK(((uintptr_t)workspace & 255) == 0, "gdn_forward_seq: workspace must be 256-byte aligned");
  HIVE_CHECK(ld >= kGdnProjWidth && ld % 8 == 0, "gdn_forward_seq: projection row stride");
  // warps per recurrence block (HIVE_QWEN_GDN_WARPS, 2 / 4 / 8; default kRecurWarps): a column's arithmetic does not depend on it (bit-identical
  //   outputs). Measured 2026-10-30, 16K-token prefill (36 layers): 8 warps (96 blocks, half the SMs idle) 166 ms · 4 warps (192 blocks) 154 ms ·
  //   2 warps 272 ms. A chunked form (the in-chunk part as a triangular solve, the chunk recurrence on CUDA cores or 3xTF32 tensor cores) was
  //   slower end to end on this card: 8,192 tokens 3.5–4.2 ms against 2.84 ms here (its extra prep kernels alone cost 1.3 ms).
  static const int warps = [] { const char* e = getenv("HIVE_QWEN_GDN_WARPS"); const int w = e ? atoi(e) : kRecurWarps; return w == 2 || w == 8 ? w : 4; }();
  static bool attr_set = false;
  if (!attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(gdn_recur_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)RecurSmem<8>::kBytes));
    CUDA_CHECK(cudaFuncSetAttribute(gdn_recur_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)RecurSmem<4>::kBytes));
    CUDA_CHECK(cudaFuncSetAttribute(gdn_recur_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)RecurSmem<2>::kBytes));
    attr_set = true;
  }
  const WsLayout L = ws_layout(std::min(T, kChunk));
  char* w = static_cast<char*>(workspace);
  PrefillArgs a{p.conv_w, p.A_log, p.dt_bias, p.norm_w, p.norm_eps, proj, ld, conv_state, state, out,
                reinterpret_cast<float*>(w + L.qp), reinterpret_cast<float*>(w + L.kp), reinterpret_cast<float*>(w + L.vw),
                reinterpret_cast<float4*>(w + L.rec), reinterpret_cast<bf16*>(w + L.ow), 0, 0};
  for (int t0 = 0; t0 < T; t0 += kChunk) {
    a.t0 = t0;
    a.n = std::min(kChunk, T - t0);
    gdn_prep_kernel<<<dim3(a.n, HK + HV), D, 0, st>>>(a);
    if (warps == 2) gdn_recur_kernel<2><<<dim3(D / 16, HV), 64, RecurSmem<2>::kBytes, st>>>(a);
    else if (warps == 8) gdn_recur_kernel<8><<<dim3(D / 64, HV), 256, RecurSmem<8>::kBytes, st>>>(a);
    else gdn_recur_kernel<4><<<dim3(D / 32, HV), 128, RecurSmem<4>::kBytes, st>>>(a);
    gdn_norm_kernel<<<(a.n * HV + 7) / 8, 256, 0, st>>>(a);
  }
  // the prep kernels read the conv state of the call start for every chunk; it is shifted only now
  launch_pdl(gdn_conv_tail_kernel, dim3(dim3((C + 255) / 256, 1)), dim3(256), 0, st, proj, ld, T, conv_state, nullptr, nullptr, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void gdn_decode_batch(const GdnParams& p, const bf16* proj, int ld, int M, float* const* conv_states, float* const* states, bf16* out,
                      cudaStream_t st) {
  if (M <= 0) return;
  HIVE_CHECK(ld >= kGdnProjWidth && ld % 8 == 0, "gdn_decode_batch: projection row stride");
  DecodeArgs a{p.conv_w, p.A_log, p.dt_bias, p.norm_w, p.norm_eps, proj, ld, conv_states, states, out};
  launch_pdl(gdn_decode_kernel, dim3(dim3(M, HV)), dim3(512), 0, st, a);
  launch_pdl(gdn_conv_tail_kernel, dim3(dim3((C + 255) / 256, M)), dim3(256), 0, st, proj, ld, 1, nullptr, conv_states, nullptr, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void gdn_decode_rows(const GdnParams& p, const bf16* proj, int ld, int R, float* const* conv_state, float* const* state, float* const* snap_conv,
                     float* const* snap_state, bf16* out, cudaStream_t st, int snap_init, float* const* snap_conv0) {
  if (R <= 0) return;
  HIVE_CHECK(ld >= kGdnProjWidth && ld % 8 == 0, "gdn_decode_rows: projection row stride");
  DecodeRowsArgs a{{p.conv_w, p.A_log, p.dt_bias, p.norm_w, p.norm_eps, proj, ld, conv_state, state, out}, snap_state, R, snap_init};
  static const bool legacy = getenv("HIVE_QWEN_GDN_ROWS_LEGACY") && atoi(getenv("HIVE_QWEN_GDN_ROWS_LEGACY")) > 0;
  if (R <= kPrepRows && !legacy) launch_pdl(gdn_decode_rows2_kernel, dim3(dim3(1, HV)), dim3(512), 0, st, a);
  else launch_pdl(gdn_decode_rows_kernel, dim3(dim3(1, HV)), dim3(512), 0, st, a);
  launch_pdl(gdn_conv_tail_kernel, dim3(dim3((C + 255) / 256, 1)), dim3(256), 0, st, proj, ld, R, nullptr, conv_state, snap_conv, snap_conv0);
  CUDA_CHECK(cudaGetLastError());
}

void gdn_decode_rows_multi_dyn(const GdnParams& p, int ld, int P, const int* vpt, const bf16* proj, float* const* conv_state, float* const* state,
                               float* const* snap, int snap_stride, int replay_from, bf16* out, cudaStream_t st) {
  if (P <= 0) return;
  HIVE_CHECK(P <= kMultiParts && ld >= kGdnProjWidth && ld % 8 == 0, "gdn_decode_rows_multi_dyn");
  const RowsDyn m{{p.conv_w, p.A_log, p.dt_bias, p.norm_w, p.norm_eps, proj, ld, conv_state, state, out}, snap, snap_stride, replay_from};
  launch_pdl(gdn_decode_rows2_dyn_kernel, dim3(dim3(P, HV)), dim3(512), 0, st, m, vpt);
  launch_pdl(gdn_conv_tail_dyn_kernel, dim3(dim3((C + 255) / 256, P)), dim3(256), 0, st, m, ld, vpt);
  CUDA_CHECK(cudaGetLastError());
}

void gdn_decode_rows_multi(const GdnParams& p, int ld, int P, const GdnRowsPart* parts, cudaStream_t st) {
  if (P <= 0) return;
  bool fit = P <= kMultiParts;
  for (int i = 0; i < P && fit; ++i) fit = parts[i].R >= 1 && parts[i].R <= kPrepRows;
  static const bool legacy = getenv("HIVE_QWEN_GDN_ROWS_LEGACY") && atoi(getenv("HIVE_QWEN_GDN_ROWS_LEGACY")) > 0;
  if (!fit || legacy) {
    for (int i = 0; i < P; ++i) {
      const GdnRowsPart& q = parts[i];
      gdn_decode_rows(p, q.proj, ld, q.R, q.conv_state, q.state, q.snap_conv, q.snap_state, q.out, st, q.snap_init, q.snap_conv0);
    }
    return;
  }
  HIVE_CHECK(ld >= kGdnProjWidth && ld % 8 == 0, "gdn_decode_rows_multi: projection row stride");
  RowsMulti rm{};
  TailMulti tm{};
  for (int i = 0; i < P; ++i) {
    const GdnRowsPart& q = parts[i];
    rm.a[i] = DecodeRowsArgs{{p.conv_w, p.A_log, p.dt_bias, p.norm_w, p.norm_eps, q.proj, ld, q.conv_state, q.state, q.out}, q.snap_state, q.R, q.snap_init};
    tm.a[i] = TailPart{q.proj, q.R, q.conv_state, q.snap_conv, q.snap_conv0};
  }
  launch_pdl(gdn_decode_rows2_multi_kernel, dim3(dim3(P, HV)), dim3(512), 0, st, rm);
  launch_pdl(gdn_conv_tail_multi_kernel, dim3(dim3((C + 255) / 256, P)), dim3(256), 0, st, tm, ld);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen
