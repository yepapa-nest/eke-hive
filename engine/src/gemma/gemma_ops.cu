// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Element-wise and per-row kernels of the Gemma family (hive/gemma/gemma_kernels.h): RMS norms (w, not 1 + w), the residual adds with
//   their post-norms and the layer scalar, the GELU-tanh gate, embeddings, per-head norm + RoPE, logit softcap. fp32 arithmetic, bf16 I/O.
#include <cmath>

#include "hive/common.h"
#include "hive/gemma/gemma_kernels.h"
#include "gemma_pdl.cuh"

namespace hive::gemma::k {

namespace {

constexpr int NT = 256;

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float red[32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  v = l < (int)(blockDim.x >> 5) ? red[l] : 0.f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// rows of K ≤ 8 · NT · VPT bf16 held in registers (one 16-byte vector per thread and step): one read, one write
constexpr int VPT = 4;
__device__ __forceinline__ void unpack8(const uint4& u, float (&f)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) { f[2 * i] = __uint_as_float(w[i] << 16); f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u); }
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    w[i] = *reinterpret_cast<const uint32_t*>(&h);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ __forceinline__ float rnd(float v) { return __bfloat162float(__float2bfloat16(v)); }

__global__ void rmsnorm_kernel(const bf16* x, int ldx, const bf16* w, int K, float eps, bf16* out, int ldo) {
  pdl_trigger();
  pdl_wait();
  const uint4* r = reinterpret_cast<const uint4*>(x + (size_t)blockIdx.x * ldx);
  const int nv = K / 8;
  float v[VPT][8];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * NT;
    if (i < nv) {
      unpack8(r[i], v[j]);
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += v[j][e] * v[j][e];
    }
  }
  const float inv = rsqrtf(block_sum(ss) / K + eps);
  uint4* o = reinterpret_cast<uint4*>(out + (size_t)blockIdx.x * ldo);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * NT;
    if (i >= nv) continue;
    float f[8], ww[8];
    if (w) unpack8(wv[i], ww);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = w ? v[j][e] * inv * ww[e] : v[j][e] * inv;
    o[i] = pack8(f);
  }
}

// h = bf16(h + bf16(rmsnorm(y) · w)) [· scalar, bf16]; then, if wn, out = rmsnorm(h) · wn (the next norm, fused)
__global__ void add_postnorm_kernel(bf16* h, const bf16* y, int ldy, const bf16* w, int K, float eps, const bf16* scalar, const bf16* wn,
                                    bf16* out, int ldo) {
  pdl_trigger();
  pdl_wait();
  const uint4* r = reinterpret_cast<const uint4*>(y + (size_t)blockIdx.x * ldy);
  uint4* hv = reinterpret_cast<uint4*>(h + (size_t)blockIdx.x * K);
  const int nv = K / 8;
  float v[VPT][8];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * NT;
    if (i < nv) {
      unpack8(r[i], v[j]);
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += v[j][e] * v[j][e];
    }
  }
  const float inv = rsqrtf(block_sum(ss) / K + eps);
  const float sc = scalar ? __bfloat162float(*scalar) : 1.f;
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  float ss2 = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * NT;
    if (i >= nv) continue;
    float ww[8], hh[8];
    unpack8(wv[i], ww);
    unpack8(hv[i], hh);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float n = rnd(v[j][e] * inv * ww[e]);
      float t = rnd(hh[e] + n);
      if (scalar) t = rnd(t * sc);
      v[j][e] = t;
      ss2 += t * t;
    }
    hv[i] = pack8(v[j]);
  }
  if (!wn) return;
  const float inv2 = rsqrtf(block_sum(ss2) / K + eps);
  const uint4* wnv = reinterpret_cast<const uint4*>(wn);
  uint4* o = reinterpret_cast<uint4*>(out + (size_t)blockIdx.x * ldo);
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * NT;
    if (i >= nv) continue;
    float ww[8], f[8];
    unpack8(wnv[i], ww);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = v[j][e] * inv2 * ww[e];
    o[i] = pack8(f);
  }
}

__global__ void gelu_mul_kernel(const bf16* g, int ldg, const bf16* u, int ldu, int I, bf16* act, int lda) {
  pdl_trigger();
  pdl_wait();
  const int m = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < I; i += gridDim.x * blockDim.x)
    act[(size_t)m * lda + i] = __float2bfloat16(gelu_tanh(__bfloat162float(g[(size_t)m * ldg + i])) * __bfloat162float(u[(size_t)m * ldu + i]));
}

__global__ void embed_kernel(const bf16* E, const int32_t* ids, int K, float scale, bf16* out, int ldo) {
  pdl_trigger();
  pdl_wait();
  const bf16* r = E + (size_t)ids[blockIdx.x] * K;
  bf16* o = out + (size_t)blockIdx.x * ldo;
  for (int i = threadIdx.x; i < K; i += blockDim.x) o[i] = __float2bfloat16(__bfloat162float(r[i]) * scale);
}

// one block per (row, head of the concatenated parts); thread i handles the rotate-half pair (i, i + D/2)
__global__ void head_norm_rope_kernel(HnrParts P, int D, float eps, const int32_t* pos) {
  pdl_trigger();
  pdl_wait();
  int h = blockIdx.y, j = 0;
  while (j + 1 < P.n && h >= P.p[j].heads) { h -= P.p[j].heads; ++j; }
  const HnrPart& q = P.p[j];
  const int m = blockIdx.x, half = D / 2;
  const bf16* r = q.x + (size_t)m * q.ldx + (size_t)h * D;
  float a = 0.f, b = 0.f;
  const int i = threadIdx.x;
  if (i < half) { a = __bfloat162float(r[i]); b = __bfloat162float(r[i + half]); }
  const float inv = rsqrtf(block_sum(a * a + b * b) / D + eps);
  a *= inv; b *= inv;
  if (i >= half) return;
  if (q.w) { a *= __bfloat162float(q.w[i]); b *= __bfloat162float(q.w[i + half]); }
  if (q.inv_freq) {
    const float f = q.inv_freq[i];
    if (f != 0.f) {
      float sn, cs;
      sincosf((float)pos[m] * f, &sn, &cs);
      const float a2 = a * cs - b * sn, b2 = b * cs + a * sn;
      a = a2; b = b2;
    }
  }
  bf16* o = q.out + (size_t)m * q.ldo + (size_t)h * D;
  o[i] = __float2bfloat16(a);
  o[i + half] = __float2bfloat16(b);
}

__global__ void softcap_kernel(float* l, size_t n, float cap) {
  pdl_trigger();
  pdl_wait();
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) l[i] = cap * tanhf(l[i] / cap);
}

struct Ids4 { int v[4]; };
__global__ void suppress_kernel(float* l, int V, Ids4 ids, int n) {
  pdl_trigger();
  pdl_wait();
  if ((int)threadIdx.x < n) l[(size_t)blockIdx.x * V + ids.v[threadIdx.x]] = -INFINITY;
}

// LayerNorm (w, b) of one row per block, the row held in registers (hive/gemma/gemma_kernels.h layernorm)
__global__ void layernorm_kernel(const bf16* x, int ldx, const bf16* w, const bf16* b, int K, float eps, const bf16* pos, int gw, bf16* out,
                                 int ldo) {
  pdl_trigger();
  pdl_wait();
  const int m = blockIdx.x, nv = K / 8;
  const uint4* r = reinterpret_cast<const uint4*>(x + (size_t)m * ldx);
  const uint4* p0 = pos ? reinterpret_cast<const uint4*>(pos + ((size_t)(m % gw) * 2 + 0) * K) : nullptr;
  const uint4* p1 = pos ? reinterpret_cast<const uint4*>(pos + ((size_t)(m / gw) * 2 + 1) * K) : nullptr;
  float v[VPT][8];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * NT;
    if (i >= nv) continue;
    unpack8(r[i], v[j]);
    if (pos) {
      float a[8], c[8];
      unpack8(p0[i], a);
      unpack8(p1[i], c);
#pragma unroll
      for (int e = 0; e < 8; ++e) v[j][e] = rnd(v[j][e] + rnd(a[e] + c[e]));
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) s += v[j][e];
  }
  const float mean = block_sum(s) / K;
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    if (threadIdx.x + j * NT >= nv) continue;
#pragma unroll
    for (int e = 0; e < 8; ++e) { const float d = v[j][e] - mean; ss += d * d; }
  }
  const float inv = rsqrtf(block_sum(ss) / K + eps);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  const uint4* bv = reinterpret_cast<const uint4*>(b);
  uint4* o = reinterpret_cast<uint4*>(out + (size_t)m * ldo);
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * NT;
    if (i >= nv) continue;
    float ww[8], bb[8], f[8];
    unpack8(wv[i], ww);
    unpack8(bv[i], bb);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = (v[j][e] - mean) * inv * ww[e] + bb[e];
    o[i] = pack8(f);
  }
}

__global__ void bias_rows_kernel(const bf16* b, int N, bf16* out) {
  pdl_trigger();
  pdl_wait();
  const uint4* bv = reinterpret_cast<const uint4*>(b);
  uint4* o = reinterpret_cast<uint4*>(out + (size_t)blockIdx.x * N);
  for (int i = threadIdx.x; i < N / 8; i += blockDim.x) o[i] = bv[i];
}

// argmax (ties → the smaller id) and Σ exp(l - max) of each row, in two passes over AS_NB column blocks (fixed order: deterministic)
constexpr int AS_NB = 64;
__global__ void argmax_sumexp_part(const float* __restrict__ l, int V, float* __restrict__ part) {
  pdl_trigger();
  pdl_wait();
  const int row = blockIdx.y, b = blockIdx.x, per = (V + AS_NB - 1) / AS_NB, v0 = b * per, v1 = min(V, v0 + per);
  const float* r = l + (size_t)row * V;
  float mx = -INFINITY;
  int arg = 0x7fffffff;
  for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x) { const float x = r[v]; if (x > mx) { mx = x; arg = v; } }
  __shared__ float smx[NT / 32];
  __shared__ int sarg[NT / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float om = __shfl_xor_sync(0xffffffffu, mx, o);
    const int oa = __shfl_xor_sync(0xffffffffu, arg, o);
    if (om > mx || (om == mx && oa < arg)) { mx = om; arg = oa; }
  }
  const int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
  if (ln == 0) { smx[w] = mx; sarg[w] = arg; }
  __syncthreads();
  if (w == 0) {
    mx = ln < NT / 32 ? smx[ln] : -INFINITY;
    arg = ln < NT / 32 ? sarg[ln] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float om = __shfl_xor_sync(0xffffffffu, mx, o);
      const int oa = __shfl_xor_sync(0xffffffffu, arg, o);
      if (om > mx || (om == mx && oa < arg)) { mx = om; arg = oa; }
    }
    if (ln == 0) { smx[0] = mx; sarg[0] = arg; }
  }
  __syncthreads();
  mx = smx[0];
  float s = 0.f;
  for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x) s += expf(r[v] - mx);
  s = block_sum(s);
  if (threadIdx.x == 0) {
    float* o = part + ((size_t)row * AS_NB + b) * 3;
    o[0] = mx; o[1] = __int_as_float(sarg[0]); o[2] = s;
  }
}
__global__ void argmax_sumexp_final(const float* __restrict__ part, int32_t* __restrict__ arg_out, float* __restrict__ sum_out) {
  pdl_trigger();
  pdl_wait();
  const int row = blockIdx.x, ln = threadIdx.x;   // one warp per row, AS_NB = 64 partials: 2 per lane
  const float* p = part + (size_t)row * AS_NB * 3;
  float m0 = p[ln * 3], m1 = p[(ln + 32) * 3];
  int a0 = __float_as_int(p[ln * 3 + 1]), a1 = __float_as_int(p[(ln + 32) * 3 + 1]);
  float mx = m0;
  int arg = a0;
  if (m1 > mx || (m1 == mx && a1 < arg)) { mx = m1; arg = a1; }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float om = __shfl_xor_sync(0xffffffffu, mx, o);
    const int oa = __shfl_xor_sync(0xffffffffu, arg, o);
    if (om > mx || (om == mx && oa < arg)) { mx = om; arg = oa; }
  }
  float s = (m0 == -INFINITY ? 0.f : p[ln * 3 + 2] * expf(m0 - mx)) + (m1 == -INFINITY ? 0.f : p[(ln + 32) * 3 + 2] * expf(m1 - mx));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (ln == 0) { arg_out[row] = arg; sum_out[row] = s; }
}

}  // namespace

void rmsnorm(const bf16* x, int ldx, const bf16* w, int M, int K, float eps, bf16* out, int ldo, cudaStream_t st) {
  HIVE_CHECK(K % 8 == 0 && K <= 8 * NT * VPT && ldx % 8 == 0 && ldo % 8 == 0, "rmsnorm: row size / alignment");
  if (M > 0) launch_pdl(rmsnorm_kernel, dim3(M), dim3(NT), 0, st, x, ldx, w, K, eps, out, ldo);
}
void add_postnorm(bf16* h, const bf16* y, int ldy, const bf16* w, int M, int K, float eps, const bf16* scalar, cudaStream_t st,
                  const bf16* wn, bf16* out, int ldo) {
  HIVE_CHECK(K % 8 == 0 && K <= 8 * NT * VPT && ldy % 8 == 0 && (!wn || ldo % 8 == 0), "add_postnorm: row size / alignment");
  if (M > 0) launch_pdl(add_postnorm_kernel, dim3(M), dim3(NT), 0, st, h, y, ldy, w, K, eps, scalar, wn, out, ldo);
}
void gelu_mul(const bf16* g, int ldg, const bf16* u, int ldu, int M, int I, bf16* act, int lda, cudaStream_t st) {
  if (M > 0) launch_pdl(gelu_mul_kernel, dim3((I + NT - 1) / NT, M), dim3(NT), 0, st, g, ldg, u, ldu, I, act, lda);
}
void embed(const bf16* E, const int32_t* ids, int M, int K, float scale, bf16* out, int ldo, cudaStream_t st) {
  if (M > 0) launch_pdl(embed_kernel, dim3(M), dim3(NT), 0, st, E, ids, K, scale, out, ldo);
}
void head_norm_rope(const HnrParts& P, int M, int D, float eps, const int32_t* pos, cudaStream_t st) {
  HIVE_CHECK(D % 64 == 0 && D <= 1024 && P.n >= 1 && P.n <= 3, "head_norm_rope: head size / parts");
  int heads = 0;
  for (int i = 0; i < P.n; ++i) heads += P.p[i].heads;
  if (M > 0) launch_pdl(head_norm_rope_kernel, dim3(M, heads), dim3(D / 2), 0, st, P, D, eps, pos);
}
void head_norm_rope(const bf16* x, int ldx, int M, int heads, int D, const bf16* w, float eps, const float* inv_freq, const int32_t* pos,
                    bf16* out, int ldo, cudaStream_t st) {
  HnrParts P;
  P.n = 1;
  P.p[0] = {x, ldx, heads, w, inv_freq, out, ldo};
  head_norm_rope(P, M, D, eps, pos, st);
}
void argmax_sumexp(const float* logits, int M, int V, float* part, int32_t* arg, float* sumexp, cudaStream_t st) {
  if (M <= 0) return;
  launch_pdl(argmax_sumexp_part, dim3(AS_NB, M), dim3(NT), 0, st, logits, V, part);
  launch_pdl(argmax_sumexp_final, dim3(M), dim3(32), 0, st, (const float*)part, arg, sumexp);
}
void softcap(float* l, size_t n, float cap, cudaStream_t st) {
  if (n) launch_pdl(softcap_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, st, l, n, cap);
}
void suppress_cols(float* l, int M, int V, const int (&ids)[4], int n, cudaStream_t st) {
  if (M <= 0 || n <= 0) return;
  HIVE_CHECK(n <= 4, "suppress_cols: at most 4 ids");
  Ids4 v;
  for (int i = 0; i < 4; ++i) v.v[i] = ids[i];
  launch_pdl(suppress_kernel, dim3(M), dim3(32), 0, st, l, V, v, n);
}
void layernorm(const bf16* x, int ldx, const bf16* w, const bf16* b, int M, int K, float eps, const bf16* pos, int gw, bf16* out, int ldo,
               cudaStream_t st) {
  HIVE_CHECK(K % 8 == 0 && K <= 8 * NT * VPT && ldx % 8 == 0 && ldo % 8 == 0 && (!pos || gw > 0), "layernorm: row size / alignment");
  if (M > 0) launch_pdl(layernorm_kernel, dim3(M), dim3(NT), 0, st, x, ldx, w, b, K, eps, pos, gw, out, ldo);
}
void bias_rows(const bf16* b, int M, int N, bf16* out, cudaStream_t st) {
  HIVE_CHECK(N % 8 == 0, "bias_rows: row size");
  if (M > 0) launch_pdl(bias_rows_kernel, dim3(M), dim3(NT), 0, st, b, N, out);
}

}  // namespace hive::gemma::k
