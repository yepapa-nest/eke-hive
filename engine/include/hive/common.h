// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive common — error macros · host/device conversions for the low-precision formats (e4m3 · e8m0 · e2m1).
// Format contract (established from the reference inference/kernel.py and convert.py):
//  · dense fp8 = e4m3 values + one e8m0 scale per 32×32 block (the engine expands it per row as [N, K/32])
//  · experts = two e2m1 nibbles per byte (low nibble = earlier element) + one e8m0 scale per row and 32-element block
//  · activations = e4m3 + one e8m0 per row and 32-element block (2^ceil(log2(amax/448)), amax≥1e-4)
//  · e8m0 value = 2^(b-127) (b=0 → 2^-127, 255 = NaN)
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(expr)                                                                      \
  do {                                                                                        \
    cudaError_t _e = (expr);                                                                  \
    if (_e != cudaSuccess) {                                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(_e), __FILE__, __LINE__, #expr); \
      abort();                                                                                \
    }                                                                                         \
  } while (0)

#define HIVE_CHECK(cond, msg)                                                                 \
  do {                                                                                        \
    if (!(cond)) {                                                                            \
      fprintf(stderr, "hive check failed at %s:%d: %s\n", __FILE__, __LINE__, std::string(msg).c_str()); \
      abort();                                                                                \
    }                                                                                         \
  } while (0)

namespace hive {

using bf16 = __nv_bfloat16;

// ---- e8m0 -------------------------------------------------------------------------------
__host__ __device__ inline float e8m0_to_f32(uint8_t b) {
  if (b == 255) return nanf("");
  // 2^(b-127) — for b<1 this is in the fp32 subnormal range (2^-127 is subnormal). ldexp makes it exact.
  return ldexpf(1.0f, (int)b - 127);
}

// e8m0 code of 2^ceil(log2(x)) (x>0, fp32). Reference fast_log2_ceil: exp-127 + (mant!=0)
__host__ __device__ inline uint8_t f32_ceil_pow2_e8m0(float x) {
  uint32_t bits;
#ifdef __CUDA_ARCH__
  bits = __float_as_uint(x);
#else
  static_assert(sizeof(float) == 4, "");
  __builtin_memcpy(&bits, &x, 4);
#endif
  int e = (int)((bits >> 23) & 0xFF) - 127;
  uint32_t man = bits & 0x7FFFFF;
  int c = e + (man != 0 ? 1 : 0);
  // subnormal inputs (e=-127): the reference fast_log2_ceil also uses -127+(man!=0) → identical
  int code = c + 127;
  if (code < 0) code = 0;
  if (code > 254) code = 254;
  return (uint8_t)code;
}

// ---- e4m3 -------------------------------------------------------------------------------
__host__ __device__ inline float e4m3_to_f32(uint8_t v) {
  uint32_t s = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
  float r;
  if (e == 0) {
    r = ldexpf((float)m, -9);  // m/8 · 2^-6
  } else if (e == 15 && m == 7) {
    r = nanf("");
  } else {
    r = ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
  }
  return s ? -r : r;
}

// Reference kernel: clamp(±448) then T.Cast(FP8) — RNE. The satfinite conversion in cuda_fp8.h gives the same result (host and device).
__host__ __device__ inline uint8_t f32_to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// Host-side e4m3 RNE (satfinite). For verification and CPU kernels.
inline uint8_t f32_to_e4m3_host(float x) {
  if (std::isnan(x)) return 0x7F;
  uint32_t bits;
  __builtin_memcpy(&bits, &x, 4);
  uint8_t sign = (uint8_t)(bits >> 31);
  float a = std::fabs(x);
  if (a > 448.0f) a = 448.0f;
  if (a < ldexpf(1.0f, -10)) return (uint8_t)(sign << 7);  // < half the smallest subnormal → 0 (tie at 2^-10 rounds to even = 0)
  // subnormal: a < 2^-6 → m = round(a·2^9)
  if (a < ldexpf(1.0f, -6)) {
    float q = a * 512.0f;
    float r = nearbyintf(q);  // RNE (default rounding mode)
    uint8_t m = (uint8_t)r;
    if (m == 8) return (uint8_t)((sign << 7) | (1 << 3));  // rounded up to the smallest normal
    return (uint8_t)((sign << 7) | m);
  }
  int e;
  float f = frexpf(a, &e);  // a = f·2^e, f∈[0.5,1)
  // normal: a = (1+m/8)·2^(E-7), E = e-1+7
  int E = e - 1 + 7;
  float mant = (f * 2.0f - 1.0f) * 8.0f;  // [0,8)
  float r = nearbyintf(mant);
  if (r == 8.0f) { r = 0; E += 1; }
  if (E > 15 || (E == 15 && r >= 7)) return (uint8_t)((sign << 7) | 0x7E);  // saturate at 448
  return (uint8_t)((sign << 7) | (E << 3) | (uint8_t)r);
}

// ---- e2m1 -------------------------------------------------------------------------------
__host__ __device__ inline float e2m1_to_f32(uint8_t nib) {
  // 0:0 1:.5 2:1 3:1.5 4:2 5:3 6:4 7:6, 8..15 = negative
  const float t = (nib & 7) == 0 ? 0.f : (nib & 7) == 1 ? 0.5f : (nib & 7) == 2 ? 1.f : (nib & 7) == 3 ? 1.5f
               : (nib & 7) == 4 ? 2.f : (nib & 7) == 5 ? 3.f : (nib & 7) == 6 ? 4.f : 6.f;
  return (nib & 8) ? -t : t;
}

// |v|≤6 → e2m1 code (RNE, ties to the even code). Matches T.Cast(FP4) in the reference fp4_act_quant.
__host__ __device__ inline uint8_t f32_to_e2m1(float v) {
  float a = fabsf(v);
  uint8_t c;
  if (a <= 0.25f) c = 0;            // tie at 0.25 → 0 (even)
  else if (a < 0.75f) c = 1;
  else if (a <= 1.25f) c = 2;       // tie at 0.75 → 2 (even), tie at 1.25 → 2
  else if (a < 1.75f) c = 3;
  else if (a <= 2.5f) c = 4;        // 1.75 → 4, 2.5 → 4
  else if (a < 3.5f) c = 5;
  else if (a <= 5.0f) c = 6;        // 3.5 → 6, 5.0 → 6
  else c = 7;
  return (uint8_t)(c | (v < 0 ? 8 : 0));
}

// ---- bf16 -------------------------------------------------------------------------------
__host__ __device__ inline float bf2f(bf16 v) { return __bfloat162float(v); }
__host__ __device__ inline bf16 f2bf(float v) { return __float2bfloat16(v); }  // RNE

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

}  // namespace hive
