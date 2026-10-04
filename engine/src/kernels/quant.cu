// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Quantization and normalization kernels — follow the act_quant / fp4_act_quant / RMSNorm definitions of the reference kernel.py exactly.
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

namespace hive::k {

namespace {

constexpr float kFp8MaxInv = 1.0f / 448.0f;
constexpr float kFp4MaxInv = 1.0f / 6.0f;

// one thread = (row, one 32-element block)
template <bool ROUNDTRIP>
__global__ void act_quant_fp8_kernel(const bf16* __restrict__ x, bf16* __restrict__ xr, int M, int K,
                                     uint8_t* __restrict__ y, uint8_t* __restrict__ s) {
  const int nb = K / 32;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * nb) return;
  const int m = idx / nb, b = idx % nb;
  const bf16* p = x + (size_t)m * K + b * 32;
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    v[i] = bf2f(p[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  amax = fmaxf(amax, 1e-4f);
  const uint8_t code = f32_ceil_pow2_e8m0(amax * kFp8MaxInv);
  const float sc = e8m0_to_f32(code);
  if (ROUNDTRIP) {
    bf16* q = xr + (size_t)m * K + b * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      float t = fminf(fmaxf(v[i] / sc, -448.f), 448.f);
      uint8_t e = f32_to_e4m3(t);
      q[i] = f2bf(e4m3_to_f32(e) * sc);
    }
  } else {
    uint8_t* q = y + (size_t)m * K + b * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      float t = fminf(fmaxf(v[i] / sc, -448.f), 448.f);
      q[i] = f32_to_e4m3(t);
    }
    s[(size_t)m * nb + b] = code;
  }
}

// fp4 round trip. BLOCK = 16 or 32. SCALE_E4M3: compressed-KV convention (amax≥6·2^-9, s=e4m3(amax/6)); otherwise e8m0 ceil-pow2.
template <int BLOCK, bool SCALE_E4M3>
__global__ void fp4_roundtrip_kernel(bf16* __restrict__ x, int M, int K) {
  const int nb = K / BLOCK;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * nb) return;
  const int m = idx / nb, b = idx % nb;
  bf16* p = x + (size_t)m * K + b * BLOCK;
  float v[BLOCK];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < BLOCK; ++i) {
    v[i] = bf2f(p[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  float sc;
  if (SCALE_E4M3) {
    amax = fmaxf(amax, 6.f * ldexpf(1.f, -9));
    sc = e4m3_to_f32(f32_to_e4m3(amax * kFp4MaxInv));  // reference: T.Cast(FP8, amax/fp4_max) → fp32
  } else {
    amax = fmaxf(amax, 6.f * ldexpf(1.f, -126));
    sc = e8m0_to_f32(f32_ceil_pow2_e8m0(amax * kFp4MaxInv));
  }
#pragma unroll
  for (int i = 0; i < BLOCK; ++i) {
    float t = fminf(fmaxf(v[i] / sc, -6.f), 6.f);
    p[i] = f2bf(e2m1_to_f32(f32_to_e2m1(t)) * sc);
  }
}

// fp4 packing (kv_pack.h): thread = (row, block). Same scale and rounding as fp4_roundtrip_kernel. row_bytes = hdr + K/2, scales at row[b], nibbles at row[hdr + d/2] (low nibble = even d)
template <int BLOCK, bool SCALE_E4M3>
__global__ void fp4_pack_kernel(const bf16* __restrict__ x, int M, int K, uint8_t* __restrict__ dst, uint8_t* const* __restrict__ dst_ptrs,
                                const int32_t* __restrict__ dst_index, const uint8_t* __restrict__ valid, int row_bytes) {
  const int nb = K / BLOCK;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * nb) return;
  const int m = idx / nb, b = idx % nb;
  if (valid && !valid[m]) return;
  uint8_t* row = dst_ptrs ? dst_ptrs[m] + (size_t)dst_index[m] * row_bytes : dst + (size_t)m * row_bytes;
  const bf16* p = x + (size_t)m * K + b * BLOCK;
  float v[BLOCK];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < BLOCK; ++i) { v[i] = bf2f(p[i]); amax = fmaxf(amax, fabsf(v[i])); }
  float sc;
  uint8_t code;
  if (SCALE_E4M3) {
    amax = fmaxf(amax, 6.f * ldexpf(1.f, -9));
    code = f32_to_e4m3(amax * kFp4MaxInv);
    sc = e4m3_to_f32(code);
  } else {
    amax = fmaxf(amax, 6.f * ldexpf(1.f, -126));
    code = f32_ceil_pow2_e8m0(amax * kFp4MaxInv);
    sc = e8m0_to_f32(code);
  }
  row[b] = code;
  const int hdr = row_bytes - K / 2;
  uint8_t* nib = row + hdr + (b * BLOCK) / 2;
#pragma unroll
  for (int i = 0; i < BLOCK; i += 2) {
    const uint8_t lo = f32_to_e2m1(fminf(fmaxf(v[i] / sc, -6.f), 6.f)), hi = f32_to_e2m1(fminf(fmaxf(v[i + 1] / sc, -6.f), 6.f));
    nib[i / 2] = (uint8_t)(lo | (hi << 4));
  }
}

template <int THREADS>
__global__ void rmsnorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w, float eps, int K,
                               bf16* __restrict__ out) {
  const int m = blockIdx.x;
  const bf16* p = x + (size_t)m * K;
  float ss = 0.f;
  for (int i = threadIdx.x; i < K; i += THREADS) {
    float v = bf2f(p[i]);
    ss += v * v;
  }
  __shared__ float red[THREADS / 32];
  // warp reduce
  ss = hive::cu::warp_allreduce(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < THREADS / 32 ? red[threadIdx.x] : 0.f;
    t = hive::cu::warp_allreduce(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / (float)K + eps);
  bf16* q = out + (size_t)m * K;
  for (int i = threadIdx.x; i < K; i += THREADS) {
    q[i] = f2bf(bf2f(w[i]) * (bf2f(p[i]) * rs));
  }
}

}  // namespace

void act_quant_fp8(const bf16* x, int M, int K, uint8_t* y, uint8_t* s, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0, "K % 32");
  const int n = M * (K / 32);
  act_quant_fp8_kernel<false><<<(n + 127) / 128, 128, 0, st>>>(x, nullptr, M, K, y, s);
}

void act_quant_fp8_roundtrip(bf16* x, int M, int K, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0, "K % 32");
  const int n = M * (K / 32);
  act_quant_fp8_kernel<true><<<(n + 127) / 128, 128, 0, st>>>(x, x, M, K, nullptr, nullptr);
}

void fp4_quant_roundtrip(bf16* x, int M, int K, int block, bool scale_e4m3, cudaStream_t st) {
  HIVE_CHECK(block == 16 || block == 32, "fp4 block");
  HIVE_CHECK(K % block == 0, "K % block");
  const int n = M * (K / block);
  const int g = (n + 127) / 128;
  if (block == 32 && !scale_e4m3) fp4_roundtrip_kernel<32, false><<<g, 128, 0, st>>>(x, M, K);
  else if (block == 16 && scale_e4m3) fp4_roundtrip_kernel<16, true><<<g, 128, 0, st>>>(x, M, K);
  else if (block == 32 && scale_e4m3) fp4_roundtrip_kernel<32, true><<<g, 128, 0, st>>>(x, M, K);
  else fp4_roundtrip_kernel<16, false><<<g, 128, 0, st>>>(x, M, K);
}

static void fp4_pack_launch(const bf16* x, int M, int K, int block, bool e4, uint8_t* dst, uint8_t* const* dst_ptrs, const int32_t* dst_index,
                            const uint8_t* valid, int row_bytes, cudaStream_t st) {
  HIVE_CHECK((block == 16 || block == 32) && K % block == 0 && row_bytes >= K / 2 + K / block, "fp4_pack shape");
  const int n = M * (K / block), g = (n + 127) / 128;
  if (block == 16 && e4) fp4_pack_kernel<16, true><<<g, 128, 0, st>>>(x, M, K, dst, dst_ptrs, dst_index, valid, row_bytes);
  else if (block == 32 && !e4) fp4_pack_kernel<32, false><<<g, 128, 0, st>>>(x, M, K, dst, dst_ptrs, dst_index, valid, row_bytes);
  else if (block == 32 && e4) fp4_pack_kernel<32, true><<<g, 128, 0, st>>>(x, M, K, dst, dst_ptrs, dst_index, valid, row_bytes);
  else fp4_pack_kernel<16, false><<<g, 128, 0, st>>>(x, M, K, dst, dst_ptrs, dst_index, valid, row_bytes);
}
void fp4_pack(const bf16* x, int M, int K, int block, bool scale_e4m3, uint8_t* dst, int row_bytes, cudaStream_t st) {
  if (M <= 0) return;
  fp4_pack_launch(x, M, K, block, scale_e4m3, dst, nullptr, nullptr, nullptr, row_bytes, st);
}
void fp4_pack_rows(const bf16* x, int M, int K, int block, bool scale_e4m3, uint8_t* const* dst_ptrs, const int32_t* dst_index, const uint8_t* valid,
                   int row_bytes, cudaStream_t st) {
  if (M <= 0) return;
  fp4_pack_launch(x, M, K, block, scale_e4m3, nullptr, dst_ptrs, dst_index, valid, row_bytes, st);
}

void rmsnorm(const bf16* x, const bf16* w, float eps, int M, int K, bf16* out, cudaStream_t st) {
  rmsnorm_kernel<256><<<M, 256, 0, st>>>(x, w, eps, K, out);
}

}  // namespace hive::k
