// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// AVX2 element-wise kernels of the Qwen3.8-27B CPU engine. The shared ones (exp, sigmoid / SiLU, bf16 rounding, Σx², the (1 + w) norm, the
//   attention output gate) are Flash-Next's (src/qwen/cpu/qwen_cpu_simd.h — the same math), used as they are; the two that differ in 27B are
//   here, each with a scalar twin performing the same IEEE operations in the same order (vector body + identical scalar tail):
//   · the MLP activation bf16(bf16(silu(g)) · u) (Flash-Next: bf16(silu(g) · u) — 27B rounds the SiLU output like the reference's bf16 act)
//   · the GDN output norm with a SiLU gate and three roundings (Flash-Next: sigmoid gate, one rounding).
#pragma once
#include "../../qwen/cpu/qwen_cpu_simd.h"

namespace hive::q27::cpu::simd {

using namespace ::hive::qwen::cpu::simd;

// out = bf16(bf16(silu(g)) · u)
inline void silu_mul_ref(const float* g, const float* u, int n, float* out) {
  for (int i = 0; i < n; ++i) out[i] = bf16r_s(bf16r_s(silu_s(g[i])) * u[i]);
}
inline void silu_mul_v(const float* g, const float* u, int n, float* out) {
  int i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(out + i, bf16r8(_mm256_mul_ps(bf16r8(silu8(_mm256_loadu_ps(g + i))), _mm256_loadu_ps(u + i))));
  for (; i < n; ++i) out[i] = bf16r_s(bf16r_s(silu_s(g[i])) * u[i]);
}
// gated RMS norm of one value head in place: o = bf16(bf16(bf16(o · rs) · w) · silu(z)), rs = 1/sqrt(Σo²/n + eps)
inline void gated_norm_silu_ref(float* o, const float* z, const float* w, int n, float eps) {
  const float rs = 1.f / std::sqrt(sumsq_ref(o, n) / (float)n + eps);
  for (int j = 0; j < n; ++j) o[j] = bf16r_s(bf16r_s(bf16r_s(o[j] * rs) * w[j]) * silu_s(z[j]));
}
inline void gated_norm_silu_v(float* o, const float* z, const float* w, int n, float eps) {
  const float rs = 1.f / std::sqrt(sumsq(o, n) / (float)n + eps);
  const __m256 rv = _mm256_set1_ps(rs);
  int j = 0;
  for (; j + 8 <= n; j += 8) {
    const __m256 a = bf16r8(_mm256_mul_ps(_mm256_loadu_ps(o + j), rv));
    const __m256 b = bf16r8(_mm256_mul_ps(a, _mm256_loadu_ps(w + j)));
    _mm256_storeu_ps(o + j, bf16r8(_mm256_mul_ps(b, silu8(_mm256_loadu_ps(z + j)))));
  }
  for (; j < n; ++j) o[j] = bf16r_s(bf16r_s(bf16r_s(o[j] * rs) * w[j]) * silu_s(z[j]));
}
// residual add: h = bf16(h + y)
inline void add_round_v(float* h, const float* y, int n) {
  int i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(h + i, bf16r8(_mm256_add_ps(_mm256_loadu_ps(h + i), _mm256_loadu_ps(y + i))));
  for (; i < n; ++i) h[i] = bf16r_s(h[i] + y[i]);
}

}  // namespace hive::q27::cpu::simd
