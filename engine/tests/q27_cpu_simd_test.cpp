// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// The 27B CPU element-wise kernels (src/q27/cpu/q27_cpu_simd.h) against their scalar twins — bit for bit, on random and edge values.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../src/q27/cpu/q27_cpu_simd.h"

using namespace hive::q27::cpu::simd;

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 3.f);
  int bad = 0;
  for (int n : {1, 7, 8, 13, 128, 1000}) {
    std::vector<float> g(n), u(n), w(n), o(n), a(n), b(n);
    for (int i = 0; i < n; ++i) { g[i] = bf16r_s(nd(rng)); u[i] = bf16r_s(nd(rng)); w[i] = bf16r_s(1.f + 0.1f * nd(rng)); o[i] = bf16r_s(nd(rng)); }
    if (n > 2) { g[0] = 90.f; g[1] = -90.f; }
    silu_mul_ref(g.data(), u.data(), n, a.data());
    silu_mul_v(g.data(), u.data(), n, b.data());
    if (memcmp(a.data(), b.data(), n * 4)) { printf("silu_mul n=%d DIFFER\n", n); ++bad; }
    a = o; b = o;
    gated_norm_silu_ref(a.data(), g.data(), w.data(), n, 1e-6f);
    gated_norm_silu_v(b.data(), g.data(), w.data(), n, 1e-6f);
    if (memcmp(a.data(), b.data(), n * 4)) { printf("gated_norm_silu n=%d DIFFER\n", n); ++bad; }
    a = o; b = o;
    for (int i = 0; i < n; ++i) a[i] = bf16r_s(a[i] + u[i]);
    add_round_v(b.data(), u.data(), n);
    if (memcmp(a.data(), b.data(), n * 4)) { printf("add_round n=%d DIFFER\n", n); ++bad; }
  }
  printf("%s\n", bad ? "FAIL" : "PASS: q27 simd kernels bit-identical to their scalar twins");
  return bad ? 1 : 0;
}
