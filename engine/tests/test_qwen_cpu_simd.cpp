// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// The Qwen CPU engine's AVX2 element-wise kernels (src/qwen/cpu/qwen_cpu_simd.h) must equal their scalar twins bit for bit, and the shared exp
//   must stay within a few ulps of libm. Lengths include non-multiples of 8 (scalar tails).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../src/qwen/cpu/qwen_cpu_simd.h"

using namespace hive::qwen::cpu::simd;

namespace {
int fails = 0;
bool same(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * 4) == 0; }
void check(bool ok, const char* what, int n) {
  if (!ok) { ++fails; printf("FAIL %s n=%d\n", what, n); }
}
// values with bf16-like magnitudes, some exact bf16 values (rounding ties), zeros and large arguments of exp
std::vector<float> rnd(std::mt19937& g, size_t n, float scale) {
  std::normal_distribution<float> d(0.f, scale);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = d(g);
    if (i % 17 == 3) v[i] = bf16r_s(v[i]);
    if (i % 29 == 5) { uint32_t u; std::memcpy(&u, &v[i], 4); u = (u & 0xffff0000u) | 0x8000u; std::memcpy(&v[i], &u, 4); }  // exact tie
    if (i % 31 == 7) v[i] = 0.f;
    if (i % 37 == 11) v[i] = (i & 1 ? 1 : -1) * 95.f;
  }
  return v;
}
}  // namespace

int main() {
  std::mt19937 g(1234);
  // exp accuracy
  double worst = 0;
  for (float x = -87.f; x <= 88.f; x += 0.0137f) worst = std::max(worst, std::fabs((double)exp_s(x) - std::exp((double)x)) / std::exp((double)x));
  printf("exp_s max relative error %.3g\n", worst);
  check(worst < 4e-7, "exp accuracy", 0);
  for (int n : {1, 7, 8, 9, 128, 255, 2560, 10240 + 3}) {
    std::vector<float> a = rnd(g, n, 3.f), b = rnd(g, n, 3.f), w = rnd(g, n, 0.2f), o1(n), o2(n);
    // exp / sigmoid / silu lanes
    for (int i = 0; i + 8 <= n; i += 8) {
      alignas(32) float e[8], s[8], l[8];
      _mm256_store_ps(e, exp8(_mm256_loadu_ps(&a[i]))); _mm256_store_ps(s, sigm8(_mm256_loadu_ps(&a[i]))); _mm256_store_ps(l, silu8(_mm256_loadu_ps(&a[i])));
      for (int j = 0; j < 8; ++j) {
        const float es = exp_s(a[i + j]), ss = sigm_s(a[i + j]), ls = silu_s(a[i + j]);
        if (std::memcmp(&e[j], &es, 4) || std::memcmp(&s[j], &ss, 4) || std::memcmp(&l[j], &ls, 4)) { check(false, "exp8/sigm8/silu8", n); i = n; break; }
      }
    }
    o1 = a; o2 = a; round_n(o1.data(), n); round_n_ref(o2.data(), n); check(same(o1.data(), o2.data(), n), "round_n", n);
    check(sumsq(a.data(), n) == sumsq_ref(a.data(), n), "sumsq", n);
    norm1p_v(a.data(), w.data(), n, 1e-6f, o1.data()); norm1p_ref(a.data(), w.data(), n, 1e-6f, o2.data()); check(same(o1.data(), o2.data(), n), "norm1p", n);
    silu_mul_v(a.data(), b.data(), n, 0.37f, 1.9f, o1.data()); silu_mul_ref(a.data(), b.data(), n, 0.37f, 1.9f, o2.data());
    check(same(o1.data(), o2.data(), n), "silu_mul", n);
    gate_v(a.data(), b.data(), n, o1.data()); gate_ref(a.data(), b.data(), n, o2.data()); check(same(o1.data(), o2.data(), n), "gate", n);
    o1 = a; o2 = a;
    gated_norm_v(o1.data(), b.data(), w.data(), n, 1e-6f); gated_norm_ref(o2.data(), b.data(), w.data(), n, 1e-6f); check(same(o1.data(), o2.data(), n), "gated_norm", n);
    // hyper-connection mix / combine over 4 copies of H = n
    const int hc = 4;
    std::vector<float> u = rnd(g, (size_t)hc * n, 2.f), hn = rnd(g, (size_t)hc * n, 1.f), x1(n), x2(n);
    hc_mix_v(u.data(), hn.data(), n, hc, x1.data()); hc_mix_ref(u.data(), hn.data(), n, hc, x2.data()); check(same(x1.data(), x2.data(), n), "hc_mix", n);
    std::vector<float> h1 = rnd(g, (size_t)hc * n, 1.f), h2 = h1;
    const float gg[4] = {0.75f, 1.25f, 0.5f, 1.875f};
    hc_combine_v(h1.data(), a.data(), gg, n, hc); hc_combine_ref(h2.data(), a.data(), gg, n, hc); check(same(h1.data(), h2.data(), h1.size()), "hc_combine", n);
  }
  printf(fails ? "test_qwen_cpu_simd: %d failure(s)\n" : "test_qwen_cpu_simd: ALL PASS\n", fails);
  return fails ? 1 : 0;
}
