// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3-ASR kernels — see include/hive/qasr/qasr_kernels.h for the contracts.
#include "hive/qasr/qasr_kernels.h"

#include <cfloat>

#include "hive/warp_reduce.cuh"

namespace hive::qasr {
namespace {

using cu::warp_allreduce;

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// Block-wide sum (blockDim.x multiple of 32, <= 1024); fixed order. `red` = 32 floats of shared memory.
__device__ __forceinline__ float block_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_allreduce(v);
  __syncthreads();
  if (lane == 0) red[wid] = v;
  __syncthreads();
  float t = lane < nw ? red[lane] : 0.f;
  return warp_allreduce(t);
}

// ---------------------------------------------------------------------------------------------------------------------------
// norms

__global__ void rmsnorm_k(const float* __restrict__ x, const float* __restrict__ w, bf16* __restrict__ y, int n, float eps) {
  __shared__ float red[32];
  const float* xr = x + (size_t)blockIdx.x * n;
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
  s = block_sum(s, red);
  const float r = rsqrtf(s / (float)n + eps);
  bf16* yr = y + (size_t)blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = __float2bfloat16(xr[i] * r * w[i]);
}

__global__ void layernorm_k(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, bf16* __restrict__ y,
                            int n, float eps) {
  __shared__ float red[32];
  const float* xr = x + (size_t)blockIdx.x * n;
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
  const float mean = block_sum(s, red) / (float)n;
  float v = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) { const float d = xr[i] - mean; v += d * d; }
  const float r = rsqrtf(block_sum(v, red) / (float)n + eps);
  bf16* yr = y + (size_t)blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = __float2bfloat16((xr[i] - mean) * r * w[i] + b[i]);
}

// ---------------------------------------------------------------------------------------------------------------------------
// element-wise

__global__ void bias_epi_k(const float* __restrict__ in, const float* __restrict__ bias, long total, int n, int mode, bf16* __restrict__ ob,
                           float* __restrict__ of) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    float v = in[i] + (bias ? bias[i % n] : 0.f);
    if (mode == 0) ob[i] = __float2bfloat16(v);
    else if (mode == 1) ob[i] = __float2bfloat16(gelu_erf(v));
    else of[i] += v;
  }
}

__global__ void swiglu_k(const float* __restrict__ gu, bf16* __restrict__ out, long total, int inter) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const long r = i / inter, c = i % inter;
    const float g = gu[r * 2 * inter + c], u = gu[r * 2 * inter + inter + c];
    out[i] = __float2bfloat16(silu(g) * u);
  }
}

__global__ void embed_k(const bf16* __restrict__ table, const int32_t* __restrict__ ids, float* __restrict__ x, int n) {
  const bf16* src = table + (size_t)ids[blockIdx.x] * n;
  float* dst = x + (size_t)blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = __bfloat162float(src[i]);
}

int grid_for(long total, int bs = 256) {
  long g = (total + bs - 1) / bs;
  return (int)(g < 65535 * 16 ? (g < 1 ? 1 : g) : 65535 * 16);
}

// ---------------------------------------------------------------------------------------------------------------------------
// q/k RMSNorm + RoPE + KV store (head_dim 128: lane owns dims 4*lane .. 4*lane+3; the rotate-half partner lives in lane ^ 16)

__global__ void qk_norm_rope_k(const float* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                               const float* __restrict__ rc, const float* __restrict__ rs, const int32_t* __restrict__ pos,
                               const int32_t* __restrict__ slot, int rows, int H, int KV, float eps, bf16* __restrict__ q_out,
                               bf16* __restrict__ kc, bf16* __restrict__ vc, long slot_stride) {
  const int lane = threadIdx.x & 31;
  const long gw = (blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5;
  const int nh = H + 2 * KV;
  if (gw >= (long)rows * nh) return;
  const int r = (int)(gw / nh), h = (int)(gw % nh);
  const float4 v4 = *reinterpret_cast<const float4*>(qkv + (size_t)r * nh * 128 + h * 128 + lane * 4);
  float x[4] = {v4.x, v4.y, v4.z, v4.w};
  const int p = pos[r];
  const long base = (long)slot[r] * slot_stride + (long)p * KV * 128;
  if (h >= H + KV) {  // v head: plain copy
    bf16* dst = vc + base + (h - H - KV) * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = __float2bfloat16(x[i]);
    return;
  }
  float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
  ss = warp_allreduce(ss);
  const float rr = rsqrtf(ss / 128.f + eps);
  const float* nw = h < H ? qn : kn;
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = x[i] * rr * nw[lane * 4 + i];
  float y[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float partner = __shfl_xor_sync(0xffffffffu, x[i], 16);
    const int f = (lane & 15) * 4 + i;  // frequency index 0..63
    const float c = rc[(size_t)p * 64 + f], s = rs[(size_t)p * 64 + f];
    y[i] = lane < 16 ? x[i] * c - partner * s : x[i] * c + partner * s;
  }
  bf16* dst = h < H ? q_out + (size_t)r * H * 128 + h * 128 + lane * 4 : kc + base + (h - H) * 128 + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = __float2bfloat16(y[i]);
}

// ---------------------------------------------------------------------------------------------------------------------------
// tiled attention: one warp per query, 8 queries per block share K/V tiles of 32 keys in shared memory.
// D = head dim (128 decoder, 64 encoder); lane owns D/32 consecutive dims. Scores reduced with a warp all-reduce (every lane sees every
// score), softmax kept online per tile in fp32.

template <int D>
struct AttnArgs {
  const bf16* q; long q_stride;  // query i, head h at q + i*q_stride + h*D
  const bf16* k; const bf16* v; long kv_stride;  // key j, kv head g at k + j*kv_stride + g*D
  bf16* out; long o_stride;
  int nq, H, rep;  // rep = H / KV
  int pos0;        // causal: query i sits at position pos0 + i and sees keys 0..pos0+i
  int win, n;      // window mode: query i sees keys [i/win*win, min(i/win*win + win, n))
};

template <int D, bool CAUSAL>
__global__ void __launch_bounds__(256) attn_tile_k(AttnArgs<D> a) {
  constexpr int E = D / 32;  // dims per lane
  constexpr int TK = 32;     // keys per tile
  __shared__ __align__(16) bf16 ks[TK][D];
  __shared__ __align__(16) bf16 vs[TK][D];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int h = blockIdx.y, g = h / a.rep;
  const int q0 = blockIdx.x * 8, qi = q0 + wid;
  const bool active = qi < a.nq;
  const int qlast = min(q0 + 7, a.nq - 1);
  int lo_blk, hi_blk, lo, hi;  // key ranges [lo, hi) for the block and for this warp's query
  if (CAUSAL) {
    lo_blk = 0; hi_blk = a.pos0 + qlast + 1;
    lo = 0; hi = a.pos0 + qi + 1;
  } else {
    lo_blk = (q0 / a.win) * a.win; hi_blk = min((qlast / a.win) * a.win + a.win, a.n);
    lo = (qi / a.win) * a.win; hi = min(lo + a.win, a.n);
  }
  const float scale = rsqrtf((float)D);
  float q[E], acc[E];
  if (active) {
    const bf16* qp = a.q + (long)qi * a.q_stride + h * D + lane * E;
#pragma unroll
    for (int e = 0; e < E; ++e) { q[e] = __bfloat162float(qp[e]) * scale; acc[e] = 0.f; }
  }
  float m = -FLT_MAX, l = 0.f;
  constexpr int VEC = 8;  // bf16 per 16-byte load
  for (int t0 = lo_blk; t0 < hi_blk; t0 += TK) {
    const int nt = min(TK, hi_blk - t0);
    __syncthreads();
    for (int i = threadIdx.x; i < TK * D / VEC; i += blockDim.x) {
      const int j = i / (D / VEC), c = (i % (D / VEC)) * VEC;
      uint4 kv4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
      if (j < nt) {
        kv4 = *reinterpret_cast<const uint4*>(a.k + (long)(t0 + j) * a.kv_stride + g * D + c);
        vv4 = *reinterpret_cast<const uint4*>(a.v + (long)(t0 + j) * a.kv_stride + g * D + c);
      }
      *reinterpret_cast<uint4*>(&ks[j][c]) = kv4;
      *reinterpret_cast<uint4*>(&vs[j][c]) = vv4;
    }
    __syncthreads();
    if (!active) continue;
    const int jlo = max(lo - t0, 0), jhi = min(hi - t0, nt);
    if (jlo >= jhi) continue;
    float s[TK];
    float tmax = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < TK; ++j) {
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < E; ++e) d += q[e] * __bfloat162float(ks[j][lane * E + e]);
      d = warp_allreduce(d);
      s[j] = (j >= jlo && j < jhi) ? d : -FLT_MAX;
      tmax = fmaxf(tmax, s[j]);
    }
    const float mn = fmaxf(m, tmax);
    const float corr = expf(m - mn);
    l *= corr;
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] *= corr;
#pragma unroll
    for (int j = 0; j < TK; ++j) {
      const float p = (j >= jlo && j < jhi) ? expf(s[j] - mn) : 0.f;
      l += p;
#pragma unroll
      for (int e = 0; e < E; ++e) acc[e] += p * __bfloat162float(vs[j][lane * E + e]);
    }
    m = mn;
  }
  if (active) {
    bf16* op = a.out + (long)qi * a.o_stride + h * D + lane * E;
    const float inv = 1.f / l;
#pragma unroll
    for (int e = 0; e < E; ++e) op[e] = __float2bfloat16(acc[e] * inv);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------
// decode attention: block = (kv head g, sequence i), 8 warps split the keys 0..len-1 into contiguous ranges; each warp handles the REP
// query heads of g, 8 keys at a time (independent shuffle chains), online softmax per head; the warps' partial states are merged
// in shared memory.

template <int REP>
__global__ void __launch_bounds__(256) attn_decode_k(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                                                     const int32_t* __restrict__ pos, const int32_t* __restrict__ slot, int H, int KV,
                                                     long slot_stride, bf16* __restrict__ out) {
  constexpr int NW = 8, KB = 8;
  __shared__ float sm_m[NW][REP], sm_l[NW][REP];
  __shared__ __align__(16) float sm_acc[NW][REP][128];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int g = blockIdx.x, i = blockIdx.y;
  const int len = pos[i] + 1;
  const bf16* kb = kc + (long)slot[i] * slot_stride + g * 128 + lane * 4;
  const bf16* vb = vc + (long)slot[i] * slot_stride + g * 128 + lane * 4;
  const long ks = (long)KV * 128;
  const float scale = rsqrtf(128.f);
  float qv[REP][4], acc[REP][4], m[REP], l[REP];
#pragma unroll
  for (int r = 0; r < REP; ++r) {
    const bf16* qp = q + (long)i * H * 128 + (g * REP + r) * 128 + lane * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) { qv[r][e] = __bfloat162float(qp[e]) * scale; acc[r][e] = 0.f; }
    m[r] = -FLT_MAX; l[r] = 0.f;
  }
  const int per = (len + NW - 1) / NW;
  const int j0 = wid * per, j1 = min(len, j0 + per);
  for (int jb = j0; jb < j1; jb += KB) {
    float s[KB][REP];
    float kf[KB][4];
#pragma unroll
    for (int u = 0; u < KB; ++u) {
      const int j = jb + u;
      if (j < j1) {
        const uint2 raw = *reinterpret_cast<const uint2*>(kb + (long)j * ks);
        const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), a1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        kf[u][0] = __low2float(a0); kf[u][1] = __high2float(a0); kf[u][2] = __low2float(a1); kf[u][3] = __high2float(a1);
      } else {
        kf[u][0] = kf[u][1] = kf[u][2] = kf[u][3] = 0.f;
      }
    }
#pragma unroll
    for (int u = 0; u < KB; ++u)
#pragma unroll
      for (int r = 0; r < REP; ++r) s[u][r] = qv[r][0] * kf[u][0] + qv[r][1] * kf[u][1] + qv[r][2] * kf[u][2] + qv[r][3] * kf[u][3];
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1)
#pragma unroll
      for (int u = 0; u < KB; ++u)
#pragma unroll
        for (int r = 0; r < REP; ++r) s[u][r] += __shfl_xor_sync(0xffffffffu, s[u][r], o);
#pragma unroll
    for (int r = 0; r < REP; ++r) {
      float tmax = -FLT_MAX;
#pragma unroll
      for (int u = 0; u < KB; ++u) if (jb + u < j1) tmax = fmaxf(tmax, s[u][r]);
      const float mn = fmaxf(m[r], tmax), corr = expf(m[r] - mn);
      l[r] *= corr;
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][e] *= corr;
      m[r] = mn;
    }
#pragma unroll
    for (int u = 0; u < KB; ++u) {
      const int j = jb + u;
      if (j >= j1) break;
      const uint2 raw = *reinterpret_cast<const uint2*>(vb + (long)j * ks);
      const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), a1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
      const float vf[4] = {__low2float(a0), __high2float(a0), __low2float(a1), __high2float(a1)};
#pragma unroll
      for (int r = 0; r < REP; ++r) {
        const float p = expf(s[u][r] - m[r]);
        l[r] += p;
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[r][e] += p * vf[e];
      }
    }
  }
#pragma unroll
  for (int r = 0; r < REP; ++r) {
    if (lane == 0) { sm_m[wid][r] = m[r]; sm_l[wid][r] = l[r]; }
#pragma unroll
    for (int e = 0; e < 4; ++e) sm_acc[wid][r][lane * 4 + e] = acc[r][e];
  }
  __syncthreads();
  for (int t = threadIdx.x; t < REP * 128; t += blockDim.x) {
    const int r = t / 128, d = t % 128;
    float M = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < NW; ++w) M = fmaxf(M, sm_m[w][r]);
    float L = 0.f, O = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float c = sm_l[w][r] > 0.f ? expf(sm_m[w][r] - M) : 0.f;
      L += sm_l[w][r] * c;
      O += sm_acc[w][r][d] * c;
    }
    out[(long)i * H * 128 + (g * REP + r) * 128 + d] = __float2bfloat16(O / L);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------
// decode GEMV: one warp per output row (two rows — gate and up — for SwiGLU); lanes stream the weight row in 16-byte vectors with
// no L1 allocation; activations (tiny, shared by every warp) are read through the cache.

__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ float dot8(const uint4& w, const uint4& x) {
  const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w);
  const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&x);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 a = __bfloat1622float2(wp[i]), b = __bfloat1622float2(xp[i]);
    s = fmaf(a.x, b.x, s);
    s = fmaf(a.y, b.y, s);
  }
  return s;
}

// MODE 0: out = acc (fp32) · 1: out += acc (fp32) · 2: SwiGLU (bf16)
template <int M, int K, int MODE>
__global__ void __launch_bounds__(256) gemv_k(const bf16* __restrict__ W, const bf16* __restrict__ x, int N, int inter, float* __restrict__ of,
                                              bf16* __restrict__ ob) {
  constexpr int NV = K / 256;  // 16-byte vectors per lane
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= (MODE == 2 ? inter : N)) return;
  const bf16* w0 = W + (size_t)row * K + lane * 8;
  const bf16* w1 = W + (size_t)(row + inter) * K + lane * 8;
  uint4 wa[NV], wb[MODE == 2 ? NV : 1];
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    wa[v] = ld_stream(w0 + v * 256);
    if constexpr (MODE == 2) wb[v] = ld_stream(w1 + v * 256);
  }
  float sa[M], sb[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { sa[r] = 0.f; sb[r] = 0.f; }
#pragma unroll
  for (int v = 0; v < NV; ++v) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (size_t)r * K + v * 256 + lane * 8));
      sa[r] += dot8(wa[v], xv);
      if constexpr (MODE == 2) sb[r] += dot8(wb[v], xv);
    }
  }
#pragma unroll
  for (int r = 0; r < M; ++r) {
    sa[r] = warp_allreduce(sa[r]);
    if constexpr (MODE == 2) sb[r] = warp_allreduce(sb[r]);
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      if constexpr (MODE == 0) of[(size_t)r * N + row] = sa[r];
      else if constexpr (MODE == 1) of[(size_t)r * N + row] += sa[r];
      else ob[(size_t)r * inter + row] = __float2bfloat16(silu(sa[r]) * sb[r]);
    }
  }
}

template <int MODE>
void gemv_launch(const bf16* W, const bf16* x, int m, int N, int K, int inter, float* of, bf16* ob, cudaStream_t st) {
  const int rows = MODE == 2 ? inter : N;
  const dim3 grid((rows + 7) / 8), blk(256);
#define HIVE_QASR_GEMV(MM, KK) gemv_k<MM, KK, MODE><<<grid, blk, 0, st>>>(W, x, N, inter, of, ob)
#define HIVE_QASR_GEMV_K(MM)                                  \
  if (K == 1024) HIVE_QASR_GEMV(MM, 1024);                    \
  else if (K == 2048) HIVE_QASR_GEMV(MM, 2048);               \
  else if (K == 3072) HIVE_QASR_GEMV(MM, 3072);               \
  else if (K == 6144) HIVE_QASR_GEMV(MM, 6144);               \
  else HIVE_CHECK(false, "qasr gemv: unsupported K " + std::to_string(K));
  switch (m) {
    case 1: HIVE_QASR_GEMV_K(1); break;
    case 2: HIVE_QASR_GEMV_K(2); break;
    case 3: HIVE_QASR_GEMV_K(3); break;
    case 4: HIVE_QASR_GEMV_K(4); break;
    default: HIVE_CHECK(false, "qasr gemv: m must be 1..4");
  }
#undef HIVE_QASR_GEMV_K
#undef HIVE_QASR_GEMV
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------------
// argmax per row (first index on ties)

__global__ void argmax_k(const float* __restrict__ logits, int V, int32_t* __restrict__ ids) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  const float* lr = logits + (size_t)blockIdx.x * V;
  float best = -FLT_MAX;
  int idx = 0x7fffffff;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float v = lr[i];
    if (v > best) { best = v; idx = i; }  // strided scan: indices grow, so '>' keeps the first one
  }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) { bv[wid] = best; bi[wid] = idx; }
  __syncthreads();
  if (wid == 0) {
    const int nw = blockDim.x >> 5;
    best = lane < nw ? bv[lane] : -FLT_MAX;
    idx = lane < nw ? bi[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
      if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (lane == 0) ids[blockIdx.x] = idx == 0x7fffffff ? 0 : idx;
  }
}

// ---------------------------------------------------------------------------------------------------------------------------
// audio front convolutions

__global__ void conv1_k(const float* __restrict__ mel, int n_mels, int n_frames, int c0, int nc, int lp, const float* __restrict__ w,
                        const float* __restrict__ b, int C, bf16* __restrict__ out) {
  const int F1 = (n_mels + 1) / 2, T1 = (lp + 1) / 2;
  const long total = (long)nc * F1 * T1 * C;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int o = (int)(i % C);
    long r = i / C;
    const int t = (int)(r % T1); r /= T1;
    const int f = (int)(r % F1);
    const int n = (int)(r / F1);
    const int chunk = c0 + n;
    const int fr0 = chunk * 100, frn = min(100, n_frames - fr0);
    float s = b[o];
#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
      const int fi = 2 * f + kh - 1;
      if (fi < 0 || fi >= n_mels) continue;
#pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        const int ti = 2 * t + kw - 1;
        if (ti < 0 || ti >= frn) continue;  // padded frames (>= the chunk's length, < lp) are zero like the conv padding
        s += w[o * 9 + kh * 3 + kw] * mel[(size_t)fi * n_frames + fr0 + ti];
      }
    }
    out[i] = __float2bfloat16(gelu_erf(s));
  }
}

__global__ void im2col_k(const bf16* __restrict__ in, int nc, int Fin, int Tin, int C, int time_major, bf16* __restrict__ col) {
  const int Fo = (Fin + 1) / 2, To = (Tin + 1) / 2, CV = C / 8;
  const long total = (long)nc * Fo * To * 9 * CV;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int cv = (int)(i % CV);
    long r = i / CV;
    const int tap = (int)(r % 9);
    const long row = r / 9;
    int n, fo, to;
    if (time_major) { fo = (int)(row % Fo); to = (int)((row / Fo) % To); n = (int)(row / ((long)Fo * To)); }
    else { to = (int)(row % To); fo = (int)((row / To) % Fo); n = (int)(row / ((long)Fo * To)); }
    const int fi = 2 * fo + tap / 3 - 1, ti = 2 * to + tap % 3 - 1;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (fi >= 0 && fi < Fin && ti >= 0 && ti < Tin) v = *reinterpret_cast<const uint4*>(in + (((long)n * Fin + fi) * Tin + ti) * C + cv * 8);
    *reinterpret_cast<uint4*>(col + row * 9 * C + tap * C + cv * 8) = v;
  }
}

__global__ void add_pos_gather_k(const float* __restrict__ in, const float* __restrict__ pos, int nc, int T3, int last_valid, int d,
                                 float* __restrict__ h) {
  const long rows = (long)(nc - 1) * T3 + last_valid;
  const long total = rows * d;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int c = (int)(i % d);
    const long r = i / d;  // output row; the first nc-1 chunks are full, so input row == output row
    const int t = (int)(r % T3);
    h[i] = in[r * d + c] + pos[(long)t * d + c];
  }
}

}  // namespace

// ===========================================================================================================================
// launchers

void rmsnorm(const float* x, const float* w, bf16* y, int rows, int n, float eps, cudaStream_t st) {
  if (rows <= 0) return;
  rmsnorm_k<<<rows, 256, 0, st>>>(x, w, y, n, eps);
  CUDA_CHECK(cudaGetLastError());
}

void layernorm(const float* x, const float* w, const float* b, bf16* y, int rows, int n, float eps, cudaStream_t st) {
  if (rows <= 0) return;
  layernorm_k<<<rows, 256, 0, st>>>(x, w, b, y, n, eps);
  CUDA_CHECK(cudaGetLastError());
}

void bias_epilogue(const float* in, const float* bias, int rows, int n, Epi mode, bf16* out_bf16, float* out_f32, cudaStream_t st) {
  const long total = (long)rows * n;
  if (total <= 0) return;
  bias_epi_k<<<grid_for(total), 256, 0, st>>>(in, bias, total, n, (int)mode, out_bf16, out_f32);
  CUDA_CHECK(cudaGetLastError());
}

void swiglu(const float* gu, bf16* out, int rows, int inter, cudaStream_t st) {
  const long total = (long)rows * inter;
  if (total <= 0) return;
  swiglu_k<<<grid_for(total), 256, 0, st>>>(gu, out, total, inter);
  CUDA_CHECK(cudaGetLastError());
}

void embed_rows(const bf16* table, const int32_t* ids, float* x, int rows, int n, cudaStream_t st) {
  if (rows <= 0) return;
  embed_k<<<rows, 256, 0, st>>>(table, ids, x, n);
  CUDA_CHECK(cudaGetLastError());
}

void qk_norm_rope_store(const float* qkv, const float* qn, const float* kn, const float* rope_cos, const float* rope_sin,
                        const int32_t* pos, const int32_t* slot, int rows, int H, int KV, float eps, bf16* q_out, bf16* kcache,
                        bf16* vcache, long slot_stride, cudaStream_t st) {
  if (rows <= 0) return;
  const long warps = (long)rows * (H + 2 * KV);
  qk_norm_rope_k<<<(unsigned)((warps + 7) / 8), 256, 0, st>>>(qkv, qn, kn, rope_cos, rope_sin, pos, slot, rows, H, KV, eps, q_out, kcache,
                                                               vcache, slot_stride);
  CUDA_CHECK(cudaGetLastError());
}

void attn_prefill_causal(const bf16* q, const bf16* kc, const bf16* vc, int nq, int pos0, int H, int KV, bf16* out, cudaStream_t st) {
  if (nq <= 0) return;
  AttnArgs<128> a{q, (long)H * 128, kc, vc, (long)KV * 128, out, (long)H * 128, nq, H, H / KV, pos0, 1, 0};
  attn_tile_k<128, true><<<dim3((nq + 7) / 8, H), 256, 0, st>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

void attn_encoder(const bf16* qkv, int n, int H, int win, bf16* out, cudaStream_t st) {
  if (n <= 0) return;
  const long s = 3L * H * 64;
  AttnArgs<64> a{qkv, s, qkv + H * 64, qkv + 2 * H * 64, s, out, (long)H * 64, n, H, 1, 0, win, n};
  attn_tile_k<64, false><<<dim3((n + 7) / 8, H), 256, 0, st>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

void attn_decode(const bf16* q, const bf16* kc, const bf16* vc, const int32_t* pos, const int32_t* slot, int m, int H, int KV,
                 long slot_stride, bf16* out, cudaStream_t st) {
  if (m <= 0) return;
  HIVE_CHECK(H == 2 * KV, "qasr attn_decode: only 2 query heads per kv head are built");
  attn_decode_k<2><<<dim3(KV, m), 256, 0, st>>>(q, kc, vc, pos, slot, H, KV, slot_stride, out);
  CUDA_CHECK(cudaGetLastError());
}

void gemv(const bf16* W, const bf16* x, int m, int N, int K, float* out, cudaStream_t st) { gemv_launch<0>(W, x, m, N, K, 0, out, nullptr, st); }
void gemv_add(const bf16* W, const bf16* x, int m, int N, int K, float* out, cudaStream_t st) {
  gemv_launch<1>(W, x, m, N, K, 0, out, nullptr, st);
}
void gemv_swiglu(const bf16* W, const bf16* x, int m, int inter, int K, bf16* out, cudaStream_t st) {
  gemv_launch<2>(W, x, m, 2 * inter, K, inter, nullptr, out, st);
}

void argmax_rows(const float* logits, int rows, int V, int32_t* ids, cudaStream_t st) {
  if (rows <= 0) return;
  argmax_k<<<rows, 1024, 0, st>>>(logits, V, ids);
  CUDA_CHECK(cudaGetLastError());
}

void conv1_gelu(const float* mel, int n_mels, int n_frames, int c0, int nc, int lp, const float* w, const float* b, int C, bf16* out,
                cudaStream_t st) {
  const long total = (long)nc * ((n_mels + 1) / 2) * ((lp + 1) / 2) * C;
  if (total <= 0) return;
  conv1_k<<<grid_for(total), 256, 0, st>>>(mel, n_mels, n_frames, c0, nc, lp, w, b, C, out);
  CUDA_CHECK(cudaGetLastError());
}

void im2col_s2(const bf16* in, int nc, int Fin, int Tin, int C, bool time_major, bf16* col, cudaStream_t st) {
  HIVE_CHECK(C % 8 == 0, "qasr im2col: channels must be a multiple of 8");
  const long total = (long)nc * ((Fin + 1) / 2) * ((Tin + 1) / 2) * 9 * (C / 8);
  if (total <= 0) return;
  im2col_k<<<grid_for(total), 256, 0, st>>>(in, nc, Fin, Tin, C, time_major ? 1 : 0, col);
  CUDA_CHECK(cudaGetLastError());
}

void add_pos_gather(const float* in, const float* pos, int nc, int T3, int last_valid, int d, float* h, cudaStream_t st) {
  const long total = ((long)(nc - 1) * T3 + last_valid) * d;
  if (total <= 0) return;
  add_pos_gather_k<<<grid_for(total), 256, 0, st>>>(in, pos, nc, T3, last_valid, d, h);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qasr
