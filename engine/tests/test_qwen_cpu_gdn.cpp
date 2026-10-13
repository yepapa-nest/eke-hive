// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// The fused L1-slab Gated DeltaNet scan (src/qwen/cpu/qwen_cpu_gdn.h) must equal the two-pass reference bit for bit — outputs and final
//   state — for every column width and token count; prints the single-thread speed of both on a prefill-size run.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../src/qwen/cpu/qwen_cpu_gdn.h"

using namespace hive::qwen::cpu::gdn;

int main() {
  const int D = 128;
  std::mt19937 g(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.5f, 1.f);
  int fails = 0;
  for (int n : {1, 2, 3, 7, 64, 300}) {
    std::vector<float> S0((size_t)D * D), K((size_t)n * D), Q((size_t)n * D), Vv((size_t)n * D), E(n), B(n);
    for (auto& x : S0) x = nd(g) * 0.1f;
    for (int t = 0; t < n; ++t) {
      float sk = 0, sq = 0;
      for (int d = 0; d < D; ++d) { K[t * D + d] = nd(g); Q[t * D + d] = nd(g); Vv[t * D + d] = nd(g); sk += K[t * D + d] * K[t * D + d]; sq += Q[t * D + d] * Q[t * D + d]; }
      for (int d = 0; d < D; ++d) { K[t * D + d] /= std::sqrt(sk); Q[t * D + d] /= std::sqrt(sq) * std::sqrt((float)D); }
      E[t] = ud(g); B[t] = ud(g) * 0.8f;
    }
    std::vector<float> Sr = S0, Or((size_t)n * D);
    ScanIn ir{K.data(), (size_t)D, Q.data(), (size_t)D, Vv.data(), (size_t)D, E.data(), 1, B.data(), 1, Or.data(), (size_t)D};
    gdn_scan_ref(Sr.data(), D, n, ir, 0, D);
    for (int W : {8, 16, 24, 32}) {
      if (D % W) continue;
      std::vector<float> Ss = S0, Os((size_t)n * D);
      ScanIn is = ir; is.o = Os.data();
      for (int j0 = 0; j0 < D; j0 += W) gdn_scan_slab(Ss.data(), D, n, is, j0, j0 + W);
      const bool ok = std::memcmp(Sr.data(), Ss.data(), Sr.size() * 4) == 0 && std::memcmp(Or.data(), Os.data(), Or.size() * 4) == 0;
      if (!ok) { ++fails; printf("FAIL n=%d W=%d\n", n, W); }
    }
  }
  // speed, one thread, 2048 tokens
  const int n = 2048;
  std::vector<float> S((size_t)D * D, 0.01f), K((size_t)n * D, 0.05f), Q((size_t)n * D, 0.01f), Vv((size_t)n * D, 0.1f), E(n, 0.99f), B(n, 0.5f), O((size_t)n * D);
  ScanIn in{K.data(), (size_t)D, Q.data(), (size_t)D, Vv.data(), (size_t)D, E.data(), 1, B.data(), 1, O.data(), (size_t)D};
  auto now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  double t0 = now(); gdn_scan_ref(S.data(), D, n, in, 0, D); const double tr = now() - t0;
  t0 = now(); for (int j0 = 0; j0 < D; j0 += 32) gdn_scan_slab(S.data(), D, n, in, j0, j0 + 32); const double ts = now() - t0;
  printf("one head, %d tokens, one thread: two-pass %.1f ms · fused slab %.1f ms\n", n, tr, ts);
  printf(fails ? "test_qwen_cpu_gdn: %d failure(s)\n" : "test_qwen_cpu_gdn: ALL PASS\n", fails);
  return fails ? 1 : 0;
}
