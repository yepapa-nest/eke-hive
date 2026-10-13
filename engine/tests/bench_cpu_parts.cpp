// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Single-thread cost of the CPU matmul's parts (includes src/cpu/cpu_matmul.cpp to reach its internal kernels), cache-resident data:
//   pack    — pack_panel (decode + transpose of 16 weight rows × 256 k) per weight format, ns per panel and ns per weight value
//   micro   — micro<6> over a 256-deep panel (12 accumulators), ns and FMA/cycle at the given GHz
//   gemv    — GEMV kernels per shape R×G and format on one L1/L2-resident weight row block, ns per (weight row · activation row)
// usage: bench_cpu_parts [GHz = 3.5]   (pin it: taskset -c 2)
#include "../src/cpu/cpu_matmul.cpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>

using namespace hive::cpu;

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::vector<uint8_t> rnd(size_t n, uint8_t mask = 0xff) {
  std::vector<uint8_t> v(n);
  std::mt19937 r(7);
  for (auto& b : v) b = (uint8_t)(r() & mask);
  return v;
}
Mat make_one(WType t, int N, int K) {
  switch (t) {
    case WType::BF16: { auto w = rnd((size_t)N * K * 2, 0x3f); return make_mat(t, N, K, w.data()); }
    case WType::F32: { std::vector<float> w((size_t)N * K, 0.01f); return make_mat(t, N, K, w.data()); }
    case WType::FP8: { auto w = rnd((size_t)N * K); for (auto& b : w) if ((b & 0x7f) == 0x7f) b = 0x30;
                       std::vector<float> s((size_t)((N + 127) / 128) * ((K + 127) / 128), 0.01f); return make_mat(t, N, K, w.data(), s.data(), SType::F32); }
    case WType::NVFP4: { auto w = rnd((size_t)N * K / 2); auto s = rnd((size_t)N * K / 16, 0x3f); for (auto& b : s) b |= 0x20;
                         return make_mat(t, N, K, w.data(), s.data(), SType::E4M3, 0.01f); }
    case WType::MXFP4: { auto w = rnd((size_t)N * K / 2); std::vector<uint8_t> s((size_t)N * K / 32, 127); return make_mat(t, N, K, w.data(), s.data(), SType::E8M0); }
    case WType::I8: { std::vector<uint16_t> w((size_t)N * K, 0x3c00); return make_mat_i8(N, K, w.data(), 64); }
  }
  return Mat{};
}
const char* tn(WType t) {
  switch (t) { case WType::BF16: return "BF16"; case WType::F32: return "F32"; case WType::FP8: return "FP8"; case WType::NVFP4: return "NVFP4";
               case WType::MXFP4: return "MXFP4"; case WType::I8: return "I8"; }
  return "?";
}
volatile float g_sink;
}  // namespace

int main(int argc, char** argv) {
  if (argc > 2 && std::string(argv[1]) == "micro") {  // micro<6> only, for perf stat: FMA instructions = 12 · 256 · calls
    std::vector<float> X((size_t)6 * 4096, 0.5f), C((size_t)6 * 1024 + 64 * 16);
    alignas(32) static float pan[kKC * kNR];
    for (auto& v : pan) v = 0.25f;
    long calls = 0; const double t0 = now(), dur = atof(argv[2]);
    while (now() - t0 < dur) { for (int r = 0; r < 64; ++r) micro<6>(X.data() + (r % 16) * 256, 4096, pan, kKC, C.data() + r * 16, 1024, 16, (r & 1)); calls += 64; }
    printf("micro<6> calls %ld · FMA instructions %.4g\n", calls, 3072.0 * calls);
    g_sink = C[5];
    return 0;
  }
  const double ghz = argc > 1 ? atof(argv[1]) : 3.5;
  const WType ts[] = {WType::BF16, WType::FP8, WType::NVFP4, WType::MXFP4, WType::I8, WType::F32};
  alignas(32) static float panel[kKC * kNR];
  // ---- pack
  for (WType t : ts) {
    Mat m = make_one(t, 64, 2048);
    int it = 0; const double t0 = now();
    while (now() - t0 < 0.4) { for (int p = 0; p < 4; ++p) for (int k0 = 0; k0 < 2048; k0 += kKC) pack_panel(m, p * 16, 16, k0, k0 + kKC, panel); ++it; }
    const double ns = (now() - t0) / it / 32 * 1e9;
    printf("pack  %-6s 16×256 panel: %8.0f ns · %.2f ns/value · %.1f cycles/value\n", tn(t), ns, ns / 4096, ns / 4096 * ghz);
    free_mat(m);
  }
  // ---- micro
  {
    std::vector<float> X((size_t)6 * 4096, 0.5f), C((size_t)6 * 1024 + 64 * 16);
    for (auto& v : panel) v = 0.25f;
    long it = 0; const double t0 = now();
    while (now() - t0 < 0.4) { for (int r = 0; r < 64; ++r) micro<6>(X.data() + (r % 16) * 256, 4096, panel, kKC, C.data() + r * 16, 1024, 16, (r & 1)); ++it; }
    const double ns = (now() - t0) / (it * 64) * 1e9;
    const double fma = 12.0 * 256;
    printf("micro<6> 256 deep: %.0f ns · %.2f FMA/cycle (peak 2) at %.2f GHz\n", ns, fma / (ns * ghz), ghz);
    printf("  pack share at a chunk of C activation rows: pack / (pack + C/6 · micro) — see the pack lines above (e.g. BF16)\n");
  }
  // ---- gemv kernels, L1/L2-resident: 16 weight rows, K = 2560 (BF16 80 KiB, NVFP4 20 KiB), X M×K
  for (WType t : ts) {
    const int N = 16, K = 2560;
    Mat m = make_one(t, N, K);
    std::vector<float> X((size_t)8 * K, 0.5f), Y((size_t)8 * N);
    const float gs = t == WType::NVFP4 ? m.gscale : 1.f;
    printf("gemv %-6s K=%d ns per (weight row · activation row) [cycles per 32 weights]:\n", tn(t), K);
    for (int M = 1; M <= 8; ++M) {
      printf("  M=%d", M);
      for (int R = 1; R <= 4; ++R)
        for (int G : {1, 2, 3, 4, 5, 6, 7, 8}) {
          if (G > M || (M % G && G != M && G * ((M + G - 1) / G) != M && false)) continue;
          if (R * G > 16) continue;
          const int passes = (M + G - 1) / G;
          long it = 0; const double t0 = now();
          while (now() - t0 < 0.05) {
            for (int r = 0; r < N; r += R) {
              GvRow rw[4];
              const int nr = std::min(R, N - r);
              for (int q = 0; q < nr; ++q) { const RowPtr p = row_ptr(m, r + q); rw[q] = GvRow{p.w, p.s}; }
              for (int p = 0, i0 = 0; p < passes; ++p) {
                const int g = M / passes + (p < M % passes ? 1 : 0);
                GvArgs a{rw, X.data() + (size_t)i0 * K, K, 0, K, m.bk, gs, Y.data() + (size_t)i0 * N + r, N};
                gv_fn(t, nr, g)(a);
                i0 += g;
              }
            }
            ++it;
          }
          const double ns = (now() - t0) / it / (N * M) * 1e9;
          printf(" %dx%d=%.0f[%.1f]", R, G, ns, ns * ghz / (K / 32));
        }
      printf("\n");
    }
    g_sink = Y[0];
    free_mat(m);
  }
  return 0;
}
