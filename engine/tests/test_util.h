// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Small helpers shared by the engine test and benchmark programs.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hive/clock.h"

namespace hive::test {

// bf16 <-> fp32 by bit pattern (bf16 is the upper half of an fp32).
inline float bf16_bits_to_f32(uint16_t hi) {
  const uint32_t w = uint32_t{hi} << 16;
  float f;
  std::memcpy(&f, &w, sizeof f);
  return f;
}

// Relative L2 error and cosine similarity of `got` against `want`; false if either has a non-finite value.
struct Closeness {
  double rel = 0, cos = 1, max_abs = 0;
  bool finite = true;
};
inline Closeness closeness(const float* got, const float* want, size_t n) {
  Closeness c;
  double num = 0, den = 0, gg = 0, ww = 0, gw = 0;
  for (size_t k = 0; k < n; ++k) {
    const double g = got[k], w = want[k];
    if (!std::isfinite(g) || !std::isfinite(w)) { c.finite = false; continue; }
    const double d = g - w;
    num += d * d; den += w * w; gg += g * g; ww += w * w; gw += g * w;
    c.max_abs = std::fmax(c.max_abs, std::fabs(d));
  }
  c.rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  c.cos = (gg > 0 && ww > 0) ? gw / std::sqrt(gg * ww) : (gg == ww ? 1.0 : 0.0);
  return c;
}
inline Closeness closeness(const std::vector<float>& got, const std::vector<float>& want) {
  return closeness(got.data(), want.data(), got.size() < want.size() ? got.size() : want.size());
}

}  // namespace hive::test
