// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// qtts kernels (see hive/qtts/qtts_kernels.h for the contracts). The math reference is oracle/qwen3_tts_oracle.py.
#include "hive/qtts/qtts_kernels.h"

#include <cfloat>
#include <cmath>

namespace hive::qtts {

namespace {

constexpr float kNegInf = -INFINITY;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float dot8(const uint4 w, const float4 a, const float4 b) {
  const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x));
  const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
  const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.z));
  const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.w));
  float s = f0.x * a.x;
  s = fmaf(f0.y, a.y, s);
  s = fmaf(f1.x, a.z, s);
  s = fmaf(f1.y, a.w, s);
  s = fmaf(f2.x, b.x, s);
  s = fmaf(f2.y, b.y, s);
  s = fmaf(f3.x, b.z, s);
  s = fmaf(f3.y, b.w, s);
  return s;
}

__device__ __forceinline__ uint4 ldg_stream(const bf16* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// ---- GEMV --------------------------------------------------------------------------------------------------------------------------------
constexpr int kGemvWarps = 8;

template <int M, int MODE, bool NORM, bool TABLE>
__global__ void __launch_bounds__(kGemvWarps * 32) gemv_kernel(const GemvParams p) {
  extern __shared__ float xs[];  // [M][K]
  __shared__ float red[M][kGemvWarps];
  const int K = p.K;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float ss[M];
#pragma unroll
  for (int m = 0; m < M; ++m) ss[m] = 0.f;
  if constexpr (TABLE) {
    const bf16* src = p.table + (size_t)(*p.idx) * K;
    for (int k = tid * 8; k < K; k += kGemvWarps * 32 * 8) {
      const uint4 u = *reinterpret_cast<const uint4*>(src + k);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        xs[k + 2 * e] = f.x;
        xs[k + 2 * e + 1] = f.y;
        ss[0] += f.x * f.x + f.y * f.y;
      }
    }
  } else {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float* src = p.x + (size_t)m * p.ldx;
      for (int k = tid * 4; k < K; k += kGemvWarps * 32 * 4) {
        const float4 v = *reinterpret_cast<const float4*>(src + k);
        *reinterpret_cast<float4*>(xs + m * K + k) = v;
        ss[m] += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
      }
    }
  }
  if constexpr (NORM) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float s = warp_sum(ss[m]);
      if (lane == 0) red[m][warp] = s;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float t = 0.f;
#pragma unroll
      for (int w = 0; w < kGemvWarps; ++w) t += red[m][w];
      const float r = rsqrtf(t / (float)K + p.eps);
      for (int k = tid; k < K; k += kGemvWarps * 32) xs[m * K + k] *= r * p.norm_w[k];
    }
  }
  __syncthreads();

  // two weight rows per warp: (2n, 2n+1) outputs, or (n, n + N) = (gate, up) for kGemvSiluMul
  const int wi = blockIdx.x * kGemvWarps + warp;
  int r0, r1;
  bool v0, v1;
  if constexpr (MODE == kGemvSiluMul) {
    r0 = wi; r1 = wi + p.N; v0 = v1 = wi < p.N;
  } else {
    r0 = 2 * wi; r1 = 2 * wi + 1; v0 = r0 < p.N; v1 = r1 < p.N;
  }
  if (!v0) return;
  const bf16* w0 = p.W + (size_t)r0 * K;
  const bf16* w1 = p.W + (size_t)(v1 ? r1 : r0) * K;
  float a0[M], a1[M];
#pragma unroll
  for (int m = 0; m < M; ++m) a0[m] = a1[m] = 0.f;
#pragma unroll 4
  for (int k = lane * 8; k < K; k += 256) {
    const uint4 u0 = ldg_stream(w0 + k);
    const uint4 u1 = ldg_stream(w1 + k);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float4 xa = *reinterpret_cast<const float4*>(xs + m * K + k);
      const float4 xb = *reinterpret_cast<const float4*>(xs + m * K + k + 4);
      a0[m] += dot8(u0, xa, xb);
      a1[m] += dot8(u1, xa, xb);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    a0[m] = warp_sum(a0[m]);
    a1[m] = warp_sum(a1[m]);
  }
  if (lane < M) {
    // lane m writes row m (keeps the per-row values in registers: select by unrolled compare)
    float x0 = 0.f, x1 = 0.f;
#pragma unroll
    for (int m = 0; m < M; ++m)
      if (lane == m) { x0 = a0[m]; x1 = a1[m]; }
    float* y = p.y + (size_t)lane * p.ldy;
    if constexpr (MODE == kGemvSiluMul) {
      y[r0] = silu(x0) * x1;
    } else if constexpr (MODE == kGemvAccum) {
      y[r0] += (p.scale ? p.scale[r0] : 1.f) * x0;
      if (v1) y[r1] += (p.scale ? p.scale[r1] : 1.f) * x1;
    } else {
      y[r0] = x0 + (p.bias ? p.bias[r0] : 0.f);
      if (v1) y[r1] = x1 + (p.bias ? p.bias[r1] : 0.f);
    }
  }
}

template <int M, int MODE>
void gemv_launch(const GemvParams& p, cudaStream_t st) {
  const int outs_per_block = MODE == kGemvSiluMul ? kGemvWarps : 2 * kGemvWarps;
  const int grid = (p.N + outs_per_block - 1) / outs_per_block;
  const size_t smem = (size_t)M * p.K * sizeof(float);
  const bool norm = p.norm_w != nullptr, table = p.table != nullptr;
  if (table) {
    if constexpr (M == 1) {
      if (norm) gemv_kernel<M, MODE, true, true><<<grid, kGemvWarps * 32, smem, st>>>(p);
      else gemv_kernel<M, MODE, false, true><<<grid, kGemvWarps * 32, smem, st>>>(p);
    }
  } else {
    if (norm) gemv_kernel<M, MODE, true, false><<<grid, kGemvWarps * 32, smem, st>>>(p);
    else gemv_kernel<M, MODE, false, false><<<grid, kGemvWarps * 32, smem, st>>>(p);
  }
}

template <int M>
void gemv_mode(const GemvParams& p, cudaStream_t st) {
  if (p.mode == kGemvSiluMul) gemv_launch<M, kGemvSiluMul>(p, st);
  else if (p.mode == kGemvAccum) gemv_launch<M, kGemvAccum>(p, st);
  else gemv_launch<M, kGemvStore>(p, st);
}

// ---- attention ---------------------------------------------------------------------------------------------------------------------------
// one warp per head (q, k or v) of one row; D = 128: lane holds dims 4*lane .. 4*lane+3, the rotate-half partner is lane ^ 16
__global__ void qk_norm_rope_store_kernel(const float* qkv, int H, int KV, const float* qn, const float* kn, float eps, const float* cos_t,
                                          const float* sin_t, const int* pos_dev, int pos_off, float* q_out, bf16* kc, bf16* vc, int cap) {
  constexpr int D = 128;
  const int m = blockIdx.x;
  const int head = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = (pos_dev ? *pos_dev : 0) + pos_off + m;
  const float* row = qkv + (size_t)m * (H + 2 * KV) * D;
  const float4 v = *reinterpret_cast<const float4*>(row + head * D + lane * 4);
  float x[4] = {v.x, v.y, v.z, v.w};
  if (head >= H + KV) {  // v: stored as is
    bf16* dst = vc + ((size_t)(head - H - KV) * cap + pos) * D + lane * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[e] = __float2bfloat16(x[e]);
    return;
  }
  const bool is_q = head < H;
  const float* nw = is_q ? qn : kn;
  const float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  const float r = rsqrtf(ss / (float)D + eps);
#pragma unroll
  for (int e = 0; e < 4; ++e) x[e] = x[e] * r * nw[lane * 4 + e];
  float y[4];
  const int i0 = (lane & 15) * 4;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float partner = __shfl_xor_sync(0xffffffffu, x[e], 16);
    const float c = cos_t[(size_t)pos * (D / 2) + i0 + e], s = sin_t[(size_t)pos * (D / 2) + i0 + e];
    y[e] = lane < 16 ? x[e] * c - partner * s : x[e] * c + partner * s;
  }
  if (is_q) {
    *reinterpret_cast<float4*>(q_out + (size_t)m * H * D + head * D + lane * 4) = make_float4(y[0], y[1], y[2], y[3]);
  } else {
    bf16* dst = kc + ((size_t)(head - H) * cap + pos) * D + lane * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[e] = __float2bfloat16(y[e]);
  }
}

__device__ __forceinline__ void load4_bf16(const bf16* p, float* out) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
}

// grid (KV, splits), 4 warps; GQA group G = H / KV heads per block (G <= 4); D = 128
template <int G>
__global__ void __launch_bounds__(128) attn_decode_kernel(const float* q, const bf16* kc, const bf16* vc, int H, int KV, int cap, const int* pos_dev,
                                                          int pos_off, float* part, float* out) {
  constexpr int D = 128;
  __shared__ float sm_m[4][G], sm_l[4][G];
  __shared__ float sm_acc[4][G][D];
  const int kvh = blockIdx.x, split = blockIdx.y, splits = gridDim.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = (pos_dev ? *pos_dev : 0) + pos_off + 1;
  const int chunk = (n + splits - 1) / splits;
  const int kb = split * chunk, ke = min(n, kb + chunk);
  const float scale = rsqrtf((float)D);
  float qv[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float4 t = *reinterpret_cast<const float4*>(q + (size_t)(kvh * G + g) * D + lane * 4);
    qv[g][0] = t.x * scale; qv[g][1] = t.y * scale; qv[g][2] = t.z * scale; qv[g][3] = t.w * scale;
  }
  float mx[G], l[G], acc[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    mx[g] = kNegInf; l[g] = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[g][e] = 0.f;
  }
  const bf16* kbase = kc + (size_t)kvh * cap * D + lane * 4;
  const bf16* vbase = vc + (size_t)kvh * cap * D + lane * 4;
  for (int k = kb + warp; k < ke; k += 4) {
    float kv[4], vv[4];
    load4_bf16(kbase + (size_t)k * D, kv);
    load4_bf16(vbase + (size_t)k * D, vv);
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float s = qv[g][0] * kv[0] + qv[g][1] * kv[1] + qv[g][2] * kv[2] + qv[g][3] * kv[3];
      s = warp_sum(s);
      const float mn = fmaxf(mx[g], s);
      const float corr = __expf(mx[g] - mn), pw = __expf(s - mn);
      l[g] = l[g] * corr + pw;
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[g][e] = acc[g][e] * corr + pw * vv[e];
      mx[g] = mn;
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (lane == 0) { sm_m[warp][g] = mx[g]; sm_l[warp][g] = l[g]; }
#pragma unroll
    for (int e = 0; e < 4; ++e) sm_acc[warp][g][lane * 4 + e] = acc[g][e];
  }
  __syncthreads();
  const int d = threadIdx.x;  // 128 threads = D
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float M = kNegInf;
#pragma unroll
    for (int w = 0; w < 4; ++w) M = fmaxf(M, sm_m[w][g]);
    float L = 0.f, A = 0.f;
    if (M != kNegInf) {
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        if (sm_m[w][g] == kNegInf) continue;
        const float c = __expf(sm_m[w][g] - M);
        L += sm_l[w][g] * c;
        A += sm_acc[w][g][d] * c;
      }
    }
    const int h = kvh * G + g;
    if (splits == 1) {
      out[(size_t)h * D + d] = A / L;
    } else {
      float* pp = part + ((size_t)h * splits + split) * (D + 2);
      if (d == 0) { pp[0] = M; pp[1] = L; }
      pp[2 + d] = A;
    }
  }
}

__global__ void attn_combine_kernel(const float* part, int splits, int D, float* out) {
  const int h = blockIdx.x, d = threadIdx.x;
  const float* pp = part + (size_t)h * splits * (D + 2);
  float M = kNegInf;
  for (int s = 0; s < splits; ++s) M = fmaxf(M, pp[s * (D + 2)]);
  float L = 0.f, A = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float m = pp[s * (D + 2)];
    if (m == kNegInf) continue;
    const float c = __expf(m - M);
    L += pp[s * (D + 2) + 1] * c;
    A += pp[s * (D + 2) + 2 + d] * c;
  }
  out[(size_t)h * D + d] = A / L;
}

// grid (M rows, H heads), 4 warps; row m attends keys 0..m
__global__ void __launch_bounds__(128) attn_prefill_kernel(const float* q, const bf16* kc, const bf16* vc, int H, int KV, int cap, bf16* out) {
  constexpr int D = 128;
  __shared__ float sm_m[4], sm_l[4], sm_acc[4][D];
  const int m = blockIdx.x, h = blockIdx.y, kvh = h / (H / KV);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float scale = rsqrtf((float)D);
  const float4 t = *reinterpret_cast<const float4*>(q + ((size_t)m * H + h) * D + lane * 4);
  const float qv[4] = {t.x * scale, t.y * scale, t.z * scale, t.w * scale};
  float mx = kNegInf, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
  const bf16* kbase = kc + (size_t)kvh * cap * D + lane * 4;
  const bf16* vbase = vc + (size_t)kvh * cap * D + lane * 4;
  for (int k = warp; k <= m; k += 4) {
    float kv[4], vv[4];
    load4_bf16(kbase + (size_t)k * D, kv);
    load4_bf16(vbase + (size_t)k * D, vv);
    const float s = warp_sum(qv[0] * kv[0] + qv[1] * kv[1] + qv[2] * kv[2] + qv[3] * kv[3]);
    const float mn = fmaxf(mx, s);
    const float corr = __expf(mx - mn), pw = __expf(s - mn);
    l = l * corr + pw;
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = acc[e] * corr + pw * vv[e];
    mx = mn;
  }
  if (lane == 0) { sm_m[warp] = mx; sm_l[warp] = l; }
#pragma unroll
  for (int e = 0; e < 4; ++e) sm_acc[warp][lane * 4 + e] = acc[e];
  __syncthreads();
  const int d = threadIdx.x;
  float M = kNegInf;
  for (int w = 0; w < 4; ++w) M = fmaxf(M, sm_m[w]);
  float L = 0.f, A = 0.f;
  for (int w = 0; w < 4; ++w) {
    if (sm_m[w] == kNegInf) continue;
    const float c = __expf(sm_m[w] - M);
    L += sm_l[w] * c;
    A += sm_acc[w][d] * c;
  }
  out[((size_t)m * H + h) * D + d] = __float2bfloat16(A / L);
}

// ---- prefill helpers -------------------------------------------------------------------------------------------------------------------
__global__ void rms_norm_bf16_kernel(const float* x, const float* w, float eps, int K, bf16* out) {
  __shared__ float red[32];
  const float* r = x + (size_t)blockIdx.x * K;
  float s = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) s += r[k] * r[k];
  s = warp_sum(s);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < (int)(blockDim.x >> 5); ++i) t += red[i];
  const float rs = rsqrtf(t / (float)K + eps);
  for (int k = threadIdx.x; k < K; k += blockDim.x) out[(size_t)blockIdx.x * K + k] = __float2bfloat16(r[k] * rs * w[k]);
}

__global__ void silu_mul_bf16_kernel(const float* gu, int M, int N, bf16* out) {
  const size_t n = (size_t)M * N;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t m = i / N, j = i % N;
    out[i] = __float2bfloat16(silu(gu[m * 2 * N + j]) * gu[m * 2 * N + N + j]);
  }
}

__global__ void f32_to_bf16_kernel(const float* x, size_t n, bf16* out) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) out[i] = __float2bfloat16(x[i]);
}

__global__ void bias_silu_bf16_kernel(const float* x, const float* b, int M, int N, bf16* out) {
  const size_t n = (size_t)M * N;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    out[i] = __float2bfloat16(silu(x[i] + b[i % N]));
}

__global__ void bias_add_kernel(float* x, const float* b, int M, int N) {
  const size_t n = (size_t)M * N;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] += b[i % N];
}

__global__ void assemble_rows_kernel(const float* P, const bf16* codec_emb, const int* tslot, const int* cid, int K, float* E) {
  const int r = blockIdx.x;
  const int ts = tslot[r], c = cid[r];
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = 0.f;
    if (ts >= 0) v += P[(size_t)ts * K + k];
    if (c >= 0) v += __bfloat162float(codec_emb[(size_t)c * K + k]);
    E[(size_t)r * K + k] = v;
  }
}

// one block of 256 threads, 8 dims each (K <= 2048 * ... any multiple of 8)
__global__ void frame_input_kernel(const bf16* codec_emb, const bf16* pred_emb, int pred_vocab, const float* pad, const int* tok, int K, float* h,
                                   int* codes, uint8_t* hist, Ctl* ctl) {
  __shared__ int t[16];
  __shared__ int frame;
  if (threadIdx.x < 16) t[threadIdx.x] = tok[threadIdx.x];
  if (threadIdx.x == 0) frame = ctl->frame;
  __syncthreads();
  for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8) {
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = pad[k + e];
    for (int i = 0; i < 16; ++i) {
      const bf16* row = i == 0 ? codec_emb + (size_t)t[0] * K : pred_emb + ((size_t)(i - 1) * pred_vocab + t[i]) * K;
      const uint4 u = *reinterpret_cast<const uint4*>(row + k);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        acc[2 * e] += f.x;
        acc[2 * e + 1] += f.y;
      }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) h[k + e] = acc[e];
  }
  if (threadIdx.x < 16) codes[(size_t)frame * 16 + threadIdx.x] = t[threadIdx.x];
  if (threadIdx.x == 0) {
    hist[t[0]] = 1;
    ctl->frame = frame + 1;
  }
}

// ---- sampling ------------------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ unsigned fkey(float x) {
  const unsigned u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

constexpr int kSampleThreads = 1024;
constexpr int kSamplePer = 4;  // V <= 4096

__global__ void __launch_bounds__(kSampleThreads) sample_kernel(const float* logits, const SampleSpec sp, const uint8_t* hist, Ctl* ctl,
                                                                int* tok_out, const int* forced, float* dump) {
  __shared__ unsigned bins[256];
  __shared__ unsigned sel_bin, sel_rem;
  __shared__ float wv[32];
  __shared__ int wi[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int frame = ctl->frame;
  const unsigned call = ctl->rng;
  const unsigned long long seed = ctl->seed;
  const int si = sp.cb == 0 ? 0 : 1;
  const float temperature = sp.from_ctl ? ctl->temperature[si] : sp.temperature;
  const int top_k = sp.from_ctl ? ctl->top_k[si] : sp.top_k;
  const float rep_penalty = sp.from_ctl && si == 0 ? ctl->rep_penalty : sp.rep_penalty;
  const int min_frames = sp.from_ctl && si == 0 ? ctl->min_frames : sp.min_frames;
  if (ctl->dump && frame < ctl->dump_frames)
    for (int i = tid; i < sp.V; i += kSampleThreads) dump[((size_t)frame * 16 + sp.cb) * 3072 + i] = logits[i];

  float v[kSamplePer];
  int nfin = 0;
#pragma unroll
  for (int j = 0; j < kSamplePer; ++j) {
    const int i = tid + j * kSampleThreads;
    float x = kNegInf;
    if (i < sp.V) {
      x = logits[i];
      if (rep_penalty != 1.f && hist && hist[i]) x = x > 0.f ? x / rep_penalty : x * rep_penalty;
      if (i >= sp.sup_lo && i < sp.sup_hi && i != sp.keep_id) x = kNegInf;
      if (i == sp.eos_id && frame < min_frames) x = kNegInf;
    }
    v[j] = x;
    nfin += x != kNegInf;
  }
  const bool greedy = temperature <= 0.f;
  unsigned thr = 0;
  if (!greedy) {
    // exact k-th largest key (radix select, 8 bits per pass); ties at the threshold are kept (logits >= k-th value)
    __shared__ int nf_s;
    if (tid == 0) nf_s = 0;
    __syncthreads();
    int nf = nfin;
    nf += __shfl_xor_sync(0xffffffffu, nf, 16);
    nf += __shfl_xor_sync(0xffffffffu, nf, 8);
    nf += __shfl_xor_sync(0xffffffffu, nf, 4);
    nf += __shfl_xor_sync(0xffffffffu, nf, 2);
    nf += __shfl_xor_sync(0xffffffffu, nf, 1);
    if (lane == 0) atomicAdd(&nf_s, nf);
    __syncthreads();
    unsigned rem = (unsigned)min(top_k > 0 ? top_k : nf_s, nf_s);
    if (rem == 0) rem = 1;
    unsigned prefix = 0, pmask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tid < 256) bins[tid] = 0;
      __syncthreads();
#pragma unroll
      for (int j = 0; j < kSamplePer; ++j) {
        if (v[j] == kNegInf) continue;
        const unsigned k = fkey(v[j]);
        if ((k & pmask) == prefix) atomicAdd(&bins[(k >> shift) & 255u], 1u);
      }
      __syncthreads();
      if (warp == 0) {
        // lane l owns bins 255-8l .. 248-8l (descending)
        unsigned c[8], s = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) { c[e] = bins[255 - 8 * lane - e]; s += c[e]; }
        unsigned incl = s;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const unsigned y = __shfl_up_sync(0xffffffffu, incl, o);
          if (lane >= o) incl += y;
        }
        const unsigned excl = incl - s;
        if (excl < rem && incl >= rem) {
          unsigned cum = excl;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            if (cum + c[e] >= rem) { sel_bin = 255 - 8 * lane - e; sel_rem = rem - cum; break; }
            cum += c[e];
          }
        }
      }
      __syncthreads();
      prefix |= sel_bin << shift;
      pmask |= 255u << shift;
      rem = sel_rem;
      __syncthreads();
    }
    thr = prefix;
  }
  // argmax of (greedy) v, or of v / T + Gumbel over the kept set
  float best = kNegInf;
  int bi = 0x7fffffff;
#pragma unroll
  for (int j = 0; j < kSamplePer; ++j) {
    const int i = tid + j * kSampleThreads;
    if (v[j] == kNegInf) continue;
    float z = v[j];
    if (!greedy) {
      if (fkey(v[j]) < thr) continue;
      const unsigned long long r = mix64(seed ^ mix64(((unsigned long long)call << 32) ^ (unsigned long long)i));
      const float u = ((float)(r >> 40) + 0.5f) * (1.0f / 16777216.0f);
      z = v[j] / temperature - logf(-logf(u));
    }
    if (z > best || (z == best && i < bi)) { best = z; bi = i; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ob = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
  }
  if (lane == 0) { wv[warp] = best; wi[warp] = bi; }
  __syncthreads();
  if (tid == 0) {
    float b = wv[0];
    int id = wi[0];
    for (int w = 1; w < kSampleThreads / 32; ++w)
      if (wv[w] > b || (wv[w] == b && wi[w] < id)) { b = wv[w]; id = wi[w]; }
    if (id == 0x7fffffff) id = sp.eos_id >= 0 ? sp.eos_id : 0;  // nothing finite: stop
    if (ctl->forced_n > 0) {
      if (frame < ctl->forced_n) id = forced[(size_t)frame * 16 + sp.cb];
      else if (sp.cb == 0) id = sp.eos_id;
    }
    *tok_out = id;
    ctl->rng = call + 1;
    if (sp.first) {
      if (sp.cb == 0 && sp.advance) ctl->pos += 1;
      if (id == sp.eos_id || frame >= ctl->max_frames || ctl->pos >= ctl->max_pos) {
        ctl->done = 1;
        if (ctl->eos_frame < 0) ctl->eos_frame = frame;
      }
    }
  }
}

// ---- codec ----------------------------------------------------------------------------------------------------------------------------------
__global__ void codec_rvq_kernel(const int* codes, const Ctl* ctl, const float* cb, int cb_size, int cb_dim, const float* Pf, const float* Pr,
                                 int out_dim, float* x) {
  extern __shared__ float s[];  // s0[cb_dim], s1[cb_dim]
  float* s0 = s;
  float* s1 = s + cb_dim;
  const int t = blockIdx.x;
  const int* c = codes + (size_t)(ctl->codec_frame + t) * 16;
  for (int d = threadIdx.x; d < cb_dim; d += blockDim.x) {
    const int c0 = min(max(c[0], 0), cb_size - 1);
    s0[d] = cb[(size_t)c0 * cb_dim + d];
    float a = 0.f;
    for (int q = 1; q < 16; ++q) {
      const int cq = min(max(c[q], 0), cb_size - 1);
      a += cb[((size_t)q * cb_size + cq) * cb_dim + d];
    }
    s1[d] = a;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  for (int o = warp; o < out_dim; o += nw) {
    float a = 0.f;
    for (int d = lane; d < cb_dim; d += 32) a += Pf[(size_t)o * cb_dim + d] * s0[d] + Pr[(size_t)o * cb_dim + d] * s1[d];
    a = warp_sum(a);
    if (lane == 0) x[(size_t)t * out_dim + o] = a;
  }
}

__device__ __forceinline__ float snake(float x, float ea, float ib) {
  const float s = sinf(x * ea);
  return x + s * s * ib;
}

__global__ void im2col_kernel(const float* x, const float* state, int T, int C, int k, int dil, const float* ea, const float* ib, bf16* A,
                              int lda) {
  const int H = (k - 1) * dil;
  const size_t n = (size_t)T * k * C;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(i % C);
    const size_t rest = i / C;
    const int j = (int)(rest % k);
    const int t = (int)(rest / k);
    const int r = t + j * dil;  // row of [state; x]
    float v = r < H ? state[(size_t)r * C + c] : x[(size_t)(r - H) * C + c];
    if (ea) v = snake(v, ea[c], ib[c]);
    A[(size_t)t * lda + (size_t)j * C + c] = __float2bfloat16(v);
  }
}

__global__ void save_state_kernel(const float* x, int T, int C, int H, float* state) {
  const size_t n = (size_t)H * C;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    state[i] = x[(size_t)(T - H) * C + i];
}

__global__ void fill_bias_kernel(float* y, int M, int N, const float* b, int nb) {
  const size_t n = (size_t)M * N;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = b[(i % N) % nb];
}

__global__ void ru_mid_kernel(const float* v, const float* b1, const float* ea, const float* ib, const float* b2, int T, int C, bf16* a, float* r) {
  const size_t n = (size_t)T * C;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(i % C);
    a[i] = __float2bfloat16(snake(v[i] + b1[c], ea[c], ib[c]));
    r[i] += b2[c];
  }
}

// grid T rows; C = 1024, 256 threads x 4 channels
__global__ void convnext_dw_ln_kernel(const float* x, const float* state, int C, const float* dw_w, const float* dw_b, const float* ln_w,
                                      const float* ln_b, bf16* out) {
  constexpr int H = 6;
  __shared__ float red[2][32];
  const int t = blockIdx.x;
  float y[4];
  float s = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int c = threadIdx.x + e * blockDim.x;
    float a = dw_b[c];
#pragma unroll
    for (int j = 0; j < 7; ++j) {
      const int r = t + j;
      const float xv = r < H ? state[(size_t)r * C + c] : x[(size_t)(r - H) * C + c];
      a += dw_w[c * 7 + j] * xv;
    }
    y[e] = a;
    s += a;
  }
  s = warp_sum(s);
  if ((threadIdx.x & 31) == 0) red[0][threadIdx.x >> 5] = s;
  __syncthreads();
  float mean = 0.f;
  for (int i = 0; i < (int)(blockDim.x >> 5); ++i) mean += red[0][i];
  mean /= (float)C;
  float q = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) q += (y[e] - mean) * (y[e] - mean);
  q = warp_sum(q);
  if ((threadIdx.x & 31) == 0) red[1][threadIdx.x >> 5] = q;
  __syncthreads();
  float var = 0.f;
  for (int i = 0; i < (int)(blockDim.x >> 5); ++i) var += red[1][i];
  const float rs = rsqrtf(var / (float)C + 1e-6f);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int c = threadIdx.x + e * blockDim.x;
    out[(size_t)t * C + c] = __float2bfloat16((y[e] - mean) * rs * ln_w[c] + ln_b[c]);
  }
}

__global__ void bias_gelu_bf16_kernel(const float* x, const float* b, int M, int N, bf16* out) {
  const size_t n = (size_t)M * N;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float v = x[i] + b[i % N];
    out[i] = __float2bfloat16(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
  }
}

__global__ void scale_bias_residual_kernel(float* x, const float* y, const float* b, const float* g, int M, int N) {
  const size_t n = (size_t)M * N;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(i % N);
    x[i] += g[c] * (y[i] + b[c]);
  }
}

// grid (T, H), D threads (D = 64): RoPE on q and k of row t, k / v into the ring
__global__ void codec_rope_store_kernel(const float* qkv, const Ctl* ctl, int H, int D, int ring, const float* cos_t, const float* sin_t,
                                        float* kr, float* vr, float* q_tmp) {
  extern __shared__ float sh[];  // q[D], k[D]
  const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int pos = ctl->codec_frame + t;
  const float* row = qkv + (size_t)t * 3 * H * D;
  sh[d] = row[h * D + d];
  sh[D + d] = row[H * D + h * D + d];
  __syncthreads();
  const int half = D / 2, i = d % half;
  const float c = cos_t[(size_t)pos * half + i], s = sin_t[(size_t)pos * half + i];
  float qo, ko;
  if (d < half) {
    qo = sh[d] * c - sh[d + half] * s;
    ko = sh[D + d] * c - sh[D + d + half] * s;
  } else {
    qo = sh[d] * c + sh[d - half] * s;
    ko = sh[D + d] * c + sh[D + d - half] * s;
  }
  const int slot = pos % ring;
  q_tmp[((size_t)t * H + h) * D + d] = qo;
  kr[((size_t)h * ring + slot) * D + d] = ko;
  vr[((size_t)h * ring + slot) * D + d] = row[2 * H * D + h * D + d];
}

// grid (H, T), 128 threads (>= window keys), D <= 128
__global__ void codec_attn_kernel(const Ctl* ctl, int H, int D, int window, int ring, const float* kr, const float* vr, const float* q_tmp,
                                  float* out) {
  __shared__ float qs[128], ps[128], red[4];
  const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const int pos = ctl->codec_frame + t;
  if (tid < D) qs[tid] = q_tmp[((size_t)t * H + h) * D + tid];
  __syncthreads();
  const int n = min(window, pos + 1);  // keys pos-n+1 .. pos
  const float scale = rsqrtf((float)D);
  float s = kNegInf;
  if (tid < n) {
    const float* kk = kr + ((size_t)h * ring + (pos - tid) % ring) * D;
    float a = 0.f;
    for (int d = 0; d < D; ++d) a += qs[d] * kk[d];
    s = a * scale;
  }
  float m = warp_max(s);
  if ((tid & 31) == 0) red[tid >> 5] = m;
  __syncthreads();
  m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  __syncthreads();
  const float p = tid < n ? __expf(s - m) : 0.f;
  ps[tid] = p;
  float l = warp_sum(p);
  if ((tid & 31) == 0) red[tid >> 5] = l;
  __syncthreads();
  l = red[0] + red[1] + red[2] + red[3];
  if (tid < D) {
    float a = 0.f;
    for (int j = 0; j < n; ++j) a += ps[j] * vr[((size_t)h * ring + (pos - j) % ring) * D + tid];
    out[((size_t)t * H + h) * D + tid] = a / l;
  }
}

__global__ void codec_out_kernel(const float* x, const float* state, int T, int C, const float* ea, const float* ib, const float* w, float b,
                                 int16_t* pcm) {
  constexpr int H = 6;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  float a = b;
  for (int j = 0; j < 7; ++j) {
    const int r = t + j;
    const float* row = r < H ? state + (size_t)r * C : x + (size_t)(r - H) * C;
    for (int c = 0; c < C; ++c) a += w[j * C + c] * snake(row[c], ea[c], ib[c]);
  }
  a = fminf(fmaxf(a, -1.f), 1.f);
  pcm[t] = (int16_t)__float2int_rz(a * 32767.f);
}

__global__ void ctl_advance_codec_kernel(Ctl* ctl, int frames) { ctl->codec_frame += frames; }

inline int grid_for(size_t n, int threads = 256) {
  const size_t g = (n + threads - 1) / threads;
  return (int)(g > 65535 * 16 ? 65535 * 16 : (g == 0 ? 1 : g));
}

}  // namespace

// ---- launchers ------------------------------------------------------------------------------------------------------------------------------
void gemv(const GemvParams& p, cudaStream_t st) {
  HIVE_CHECK(p.K % 256 == 0, "gemv: K must be a multiple of 256");
  HIVE_CHECK((size_t)p.M * p.K * sizeof(float) <= 48 * 1024, "gemv: input rows exceed the 48 KiB staging");
  HIVE_CHECK(p.table == nullptr || p.M == 1, "gemv: table input needs M = 1");
  switch (p.M) {
    case 1: gemv_mode<1>(p, st); break;
    case 2: gemv_mode<2>(p, st); break;
    case 3: gemv_mode<3>(p, st); break;
    case 4: gemv_mode<4>(p, st); break;
    default: HIVE_CHECK(false, "gemv: M must be 1..4");
  }
}

void qk_norm_rope_store(const float* qkv, int M, int H, int KV, int D, const float* qn, const float* kn, float eps, const float* cos_t,
                        const float* sin_t, const int* pos_dev, int pos_off, float* q_out, bf16* kc, bf16* vc, int cap, cudaStream_t st) {
  HIVE_CHECK(D == 128 && (H + 2 * KV) * 32 <= 1024, "qk_norm_rope_store: D must be 128");
  qk_norm_rope_store_kernel<<<M, (H + 2 * KV) * 32, 0, st>>>(qkv, H, KV, qn, kn, eps, cos_t, sin_t, pos_dev, pos_off, q_out, kc, vc, cap);
}

void attn_decode(const float* q, const bf16* kc, const bf16* vc, int H, int KV, int D, int cap, const int* pos_dev, int pos_off, int splits,
                 float* part, float* out, cudaStream_t st) {
  HIVE_CHECK(D == 128, "attn_decode: D must be 128");
  const dim3 grid(KV, splits);
  switch (H / KV) {
    case 1: attn_decode_kernel<1><<<grid, 128, 0, st>>>(q, kc, vc, H, KV, cap, pos_dev, pos_off, part, out); break;
    case 2: attn_decode_kernel<2><<<grid, 128, 0, st>>>(q, kc, vc, H, KV, cap, pos_dev, pos_off, part, out); break;
    case 4: attn_decode_kernel<4><<<grid, 128, 0, st>>>(q, kc, vc, H, KV, cap, pos_dev, pos_off, part, out); break;
    default: HIVE_CHECK(false, "attn_decode: unsupported GQA group");
  }
}

void attn_combine(const float* part, int H, int D, int splits, float* out, cudaStream_t st) {
  attn_combine_kernel<<<H, D, 0, st>>>(part, splits, D, out);
}

void attn_prefill(const float* q, const bf16* kc, const bf16* vc, int M, int H, int KV, int D, int cap, bf16* out, cudaStream_t st) {
  HIVE_CHECK(D == 128, "attn_prefill: D must be 128");
  attn_prefill_kernel<<<dim3(M, H), 128, 0, st>>>(q, kc, vc, H, KV, cap, out);
}

void rms_norm_bf16(const float* x, const float* w, float eps, int M, int K, bf16* out, cudaStream_t st) {
  rms_norm_bf16_kernel<<<M, 256, 0, st>>>(x, w, eps, K, out);
}
void silu_mul_bf16(const float* gu, int M, int N, bf16* out, cudaStream_t st) {
  silu_mul_bf16_kernel<<<grid_for((size_t)M * N), 256, 0, st>>>(gu, M, N, out);
}
void f32_to_bf16(const float* x, size_t n, bf16* out, cudaStream_t st) { f32_to_bf16_kernel<<<grid_for(n), 256, 0, st>>>(x, n, out); }
void bias_silu_bf16(const float* x, const float* bias, int M, int N, bf16* out, cudaStream_t st) {
  bias_silu_bf16_kernel<<<grid_for((size_t)M * N), 256, 0, st>>>(x, bias, M, N, out);
}
void bias_add(float* x, const float* bias, int M, int N, cudaStream_t st) {
  bias_add_kernel<<<grid_for((size_t)M * N), 256, 0, st>>>(x, bias, M, N);
}
void assemble_rows(const float* P, const bf16* codec_emb, const int* tslot, const int* cid, int R, int K, float* E, cudaStream_t st) {
  assemble_rows_kernel<<<R, 256, 0, st>>>(P, codec_emb, tslot, cid, K, E);
}
void frame_input(const bf16* codec_emb, const bf16* pred_emb, int pred_vocab, const float* pad, const int* tok, int K, float* h, int* codes,
                 uint8_t* hist, Ctl* ctl, cudaStream_t st) {
  HIVE_CHECK(K % 8 == 0, "frame_input: K must be a multiple of 8");
  frame_input_kernel<<<1, 256, 0, st>>>(codec_emb, pred_emb, pred_vocab, pad, tok, K, h, codes, hist, ctl);
}
void sample(const float* logits, const SampleSpec& sp, const uint8_t* hist, Ctl* ctl, int* tok_out, const int* forced, float* dump,
            cudaStream_t st) {
  HIVE_CHECK(sp.V <= kSampleThreads * kSamplePer && sp.V <= 3072, "sample: vocabulary too large");
  sample_kernel<<<1, kSampleThreads, 0, st>>>(logits, sp, hist, ctl, tok_out, forced, dump);
}

void codec_rvq(const int* codes, const Ctl* ctl, int T, const float* cb, int cb_size, int cb_dim, const float* Pf, const float* Pr, int out_dim,
               float* x, cudaStream_t st) {
  codec_rvq_kernel<<<T, 256, 2 * cb_dim * sizeof(float), st>>>(codes, ctl, cb, cb_size, cb_dim, Pf, Pr, out_dim, x);
}
void im2col_causal(const float* x, const float* state, int T, int C, int k, int dil, const float* ea, const float* ib, bf16* A, int lda,
                   cudaStream_t st) {
  im2col_kernel<<<grid_for((size_t)T * k * C), 256, 0, st>>>(x, state, T, C, k, dil, ea, ib, A, lda);
}
void save_state(const float* x, int T, int C, int H, float* state, cudaStream_t st) {
  HIVE_CHECK(T >= H, "save_state: chunk shorter than the conv history");
  if (H > 0) save_state_kernel<<<grid_for((size_t)H * C), 256, 0, st>>>(x, T, C, H, state);
}
void fill_bias(float* y, int M, int N, const float* bias, int nb, cudaStream_t st) {
  fill_bias_kernel<<<grid_for((size_t)M * N), 256, 0, st>>>(y, M, N, bias, nb);
}
void ru_mid(const float* v, const float* b1, const float* ea, const float* ib, const float* b2, int T, int C, bf16* a, float* r, cudaStream_t st) {
  ru_mid_kernel<<<grid_for((size_t)T * C), 256, 0, st>>>(v, b1, ea, ib, b2, T, C, a, r);
}
void convnext_dw_ln(const float* x, float* state, int T, int C, const float* dw_w, const float* dw_b, const float* ln_w, const float* ln_b,
                    bf16* out, cudaStream_t st) {
  HIVE_CHECK(C == 1024, "convnext_dw_ln: C must be 1024");
  convnext_dw_ln_kernel<<<T, 256, 0, st>>>(x, state, C, dw_w, dw_b, ln_w, ln_b, out);
  save_state(x, T, C, 6, state, st);
}
void bias_gelu_bf16(const float* x, const float* b, int M, int N, bf16* out, cudaStream_t st) {
  bias_gelu_bf16_kernel<<<grid_for((size_t)M * N), 256, 0, st>>>(x, b, M, N, out);
}
void scale_bias_residual(float* x, const float* y, const float* b, const float* gamma, int M, int N, cudaStream_t st) {
  scale_bias_residual_kernel<<<grid_for((size_t)M * N), 256, 0, st>>>(x, y, b, gamma, M, N);
}
void codec_attn(const float* qkv, const Ctl* ctl, int T, int H, int D, int window, int ring, const float* cos_t, const float* sin_t, float* kr,
                float* vr, float* q_tmp, float* out, cudaStream_t st) {
  HIVE_CHECK(D <= 128 && window <= 128 && ring >= window + T, "codec_attn: window / ring too small");
  codec_rope_store_kernel<<<dim3(T, H), D, 2 * D * sizeof(float), st>>>(qkv, ctl, H, D, ring, cos_t, sin_t, kr, vr, q_tmp);
  codec_attn_kernel<<<dim3(H, T), 128, 0, st>>>(ctl, H, D, window, ring, kr, vr, q_tmp, out);
}
void codec_out(const float* x, float* state, int T, int C, const float* ea, const float* ib, const float* w, float b, int16_t* pcm,
               cudaStream_t st) {
  codec_out_kernel<<<(T + 127) / 128, 128, 0, st>>>(x, state, T, C, ea, ib, w, b, pcm);
  save_state(x, T, C, 6, state, st);
}
void ctl_advance_codec(Ctl* ctl, int frames, cudaStream_t st) { ctl_advance_codec_kernel<<<1, 1, 0, st>>>(ctl, frames); }

void gemm(cublasHandle_t h, const bf16* A, int lda, const bf16* B, int ldb, float* C, int ldc, int M, int N, int K, float beta) {
  const float alpha = 1.f;
  const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, ldb, A, CUDA_R_16BF, lda, &beta, C,
                                        CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  HIVE_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed (status " + std::to_string((int)s) + ")");
}

}  // namespace hive::qtts
