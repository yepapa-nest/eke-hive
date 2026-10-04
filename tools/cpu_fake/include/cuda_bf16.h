// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Fake bf16 for CPU-only tests: round-to-nearest-even like __float2bfloat16.
#pragma once
#include <cstdint>
#include <cstring>
struct __nv_bfloat16 { uint16_t x; };
inline __nv_bfloat16 __float2bfloat16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return __nv_bfloat16{(uint16_t)((u >> 16) | 0x40)};  // quiet NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return __nv_bfloat16{(uint16_t)(u >> 16)};
}
inline float __bfloat162float(__nv_bfloat16 b) { uint32_t u = (uint32_t)b.x << 16; float f; std::memcpy(&f, &u, sizeof f); return f; }
