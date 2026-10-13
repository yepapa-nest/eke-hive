// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU backend (hive/cpu/cpu_backend.h) against scalar fp64 references:
//   matmul for every weight format × M ∈ {1, 2, 3, 8, 9, 33, 128} (GEMV and GEMM paths) on odd shapes, MatView row ranges, matmul_multi,
//   FP8 block scales given as F32 / BF16 / E8M0 (128×128 and 32×32 blocks), attention (causal range and index lists with −1 holes), and
//   bit-identical results for different pool sizes (the test re-runs itself with HIVE_CPU_POOL_THREADS = 1, 3 and the default and compares
//   a hash of every output).
//   Tolerance: |y − ref| ≤ 4·K·2^-24 · Σ|x·w| + 1e-30 — the fp32 accumulation bound for K terms (the kernels sum in a different order).
// Build (CPU only): g++ -std=c++20 -O3 -mavx2 -mfma -mf16c -Iinclude -Icpu_shim -Ithird_party tests/test_cpu_backend_cpu.cpp
//   src/cpu/cpu_backend.cpp src/cpu/cpu_matmul.cpp src/cpu/cpu_attention.cpp -lnuma -lpthread
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hive/common.h"
#include "hive/cpu/cpu_backend.h"

using namespace hive;
using namespace hive::cpu;

namespace {
int fails = 0;
uint64_t hash_acc = 1469598103934665603ull;
void hash_floats(const float* p, size_t n) {
  const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n * 4; ++i) hash_acc = (hash_acc ^ b[i]) * 1099511628211ull;
}
const double kE2M1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0., -0.5, -1, -1.5, -2, -3, -4, -6};
uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

struct Case {
  WType t; int N, K; SType sdt; int bn = 128, bk = 128;
  std::vector<uint8_t> w, s; float g = 1.f;
  std::vector<double> deq;  // [N, K]
};

Case make_case(WType t, int N, int K, SType sdt, std::mt19937& rng, int bn = 128, int bk = 128) {
  Case c{t, N, K, sdt, bn, bk};
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  c.deq.assign((size_t)N * K, 0.0);
  switch (t) {
    case WType::BF16: {
      c.w.resize((size_t)N * K * 2);
      for (size_t i = 0; i < (size_t)N * K; ++i) { const uint16_t b = f2b(U(rng)); memcpy(&c.w[i * 2], &b, 2); c.deq[i] = bf16_to_f32(b); }
      break;
    }
    case WType::F32: {
      c.w.resize((size_t)N * K * 4);
      for (size_t i = 0; i < (size_t)N * K; ++i) { const float v = U(rng); memcpy(&c.w[i * 4], &v, 4); c.deq[i] = v; }
      break;
    }
    case WType::FP8: {
      c.w.resize((size_t)N * K);
      for (auto& b : c.w) { do b = (uint8_t)(rng() & 0xff); while ((b & 0x7f) == 0x7f); }
      const int nbr = (N + bn - 1) / bn, nbk = (K + bk - 1) / bk;
      std::vector<double> sv((size_t)nbr * nbk);
      const size_t es = sdt == SType::F32 ? 4 : sdt == SType::BF16 ? 2 : 1;
      c.s.resize(sv.size() * es);
      for (size_t i = 0; i < sv.size(); ++i) {
        if (sdt == SType::F32) { const float v = 0.001f + 0.01f * (float)(rng() % 1000) / 1000.f; memcpy(&c.s[i * 4], &v, 4); sv[i] = v; }
        else if (sdt == SType::BF16) { const uint16_t b = f2b(0.001f + 0.01f * (float)(rng() % 1000) / 1000.f); memcpy(&c.s[i * 2], &b, 2); sv[i] = bf16_to_f32(b); }
        else { const uint8_t b = (uint8_t)(110 + rng() % 20); c.s[i] = b; sv[i] = std::ldexp(1.0, (int)b - 127); }
      }
      for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) c.deq[(size_t)n * K + k] = hive::e4m3_to_f32(c.w[(size_t)n * K + k]) * sv[(size_t)(n / bn) * nbk + k / bk];
      break;
    }
    case WType::NVFP4: case WType::MXFP4: {
      c.w.resize((size_t)N * K / 2);
      for (auto& b : c.w) b = (uint8_t)(rng() & 0xff);
      const int blk = t == WType::NVFP4 ? 16 : 32;
      c.s.resize((size_t)N * K / blk);
      for (auto& b : c.s) {
        if (t == WType::NVFP4) { do b = (uint8_t)(0x20 + rng() % 0x40); while ((b & 0x7f) == 0x7f); }
        else b = (uint8_t)(118 + rng() % 16);
      }
      c.g = t == WType::NVFP4 ? 0.0123f : 1.f;
      for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
          const uint8_t byte = c.w[((size_t)n * K + k) / 2];
          const double v = kE2M1[(k & 1) ? (byte >> 4) : (byte & 15)];
          const uint8_t sb = c.s[(size_t)n * (K / blk) + k / blk];
          const double sc = t == WType::NVFP4 ? (double)hive::e4m3_to_f32(sb) * c.g : std::ldexp(1.0, (int)sb - 127);
          c.deq[(size_t)n * K + k] = v * sc;
        }
      break;
    }
  }
  return c;
}

const char* tname(WType t) {
  switch (t) { case WType::BF16: return "BF16"; case WType::F32: return "F32"; case WType::FP8: return "FP8"; case WType::NVFP4: return "NVFP4"; case WType::MXFP4: return "MXFP4"; }
  return "?";
}

void check_matmul(Case& c, int M, int r0, int r1, std::mt19937& rng, bool multi = false) {
  Mat m = make_mat(c.t, c.N, c.K, c.w.data(), c.s.empty() ? nullptr : c.s.data(), c.sdt, c.g, c.bn, c.bk);
  const int ldx = c.K + 3, ldy = (r1 - r0) + 5;
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<float> X((size_t)M * ldx), Y((size_t)M * ldy, -7.f), Y2;
  for (auto& v : X) v = U(rng);
  if (multi) {  // two jobs over halves of the range
    const int mid = (r0 + r1) / 2;
    Y2.assign((size_t)M * ldy, -7.f);
    MatJob j[2] = {{MatView{&m, r0, mid}, Y.data(), ldy}, {MatView{&m, mid, r1}, Y.data() + (mid - r0), ldy}};
    matmul_multi(j, 2, X.data(), M, ldx);
  } else {
    matmul(MatView{&m, r0, r1}, X.data(), M, ldx, Y.data(), ldy);
  }
  double worst = 0;
  for (int i = 0; i < M; ++i)
    for (int n = r0; n < r1; ++n) {
      double ref = 0, mag = 0;
      for (int k = 0; k < c.K; ++k) { const double p = (double)X[(size_t)i * ldx + k] * c.deq[(size_t)n * c.K + k]; ref += p; mag += std::fabs(p); }
      const double got = Y[(size_t)i * ldy + (n - r0)];
      const double tol = 4.0 * c.K * std::ldexp(1.0, -24) * mag + 1e-30;
      worst = std::max(worst, std::fabs(got - ref) / tol);
    }
  hash_floats(Y.data(), Y.size());
  const bool ok = worst <= 1.0;
  if (!ok) ++fails;
  if (!ok || getenv("VERBOSE"))
    printf("%s matmul %-5s N=%d K=%d M=%d rows [%d,%d)%s sdt=%d bn=%d bk=%d: worst/tol %.3f\n", ok ? "PASS" : "FAIL", tname(c.t), c.N, c.K, M, r0, r1,
           multi ? " multi" : "", (int)c.sdt, c.bn, c.bk, worst);
  free_mat(m);
}

void check_attention(std::mt19937& rng) {
  const int M = 3, Hq = 8, Hkv = 2, D = 72, T = 100;
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<float> q((size_t)M * Hq * D), out((size_t)M * Hq * D), out2((size_t)M * Hq * D);
  for (auto& v : q) v = U(rng);
  std::vector<uint16_t> K((size_t)T * Hkv * D), V((size_t)T * Hkv * D);
  for (auto& v : K) v = f2b(U(rng));
  for (auto& v : V) v = f2b(U(rng));
  const int32_t ce[M] = {1, 37, 100};
  const int S = 40;
  std::vector<int32_t> idx((size_t)M * S);
  for (int m = 0; m < M; ++m)
    for (int j = 0; j < S; ++j) idx[(size_t)m * S + j] = (j % 7 == 3) ? -1 : (int32_t)((j * 13 + m * 5) % T);
  const float sc = 1.f / std::sqrt((float)D);
  for (int mode = 0; mode < 2; ++mode) {
    attention(q.data(), M, Hq, Hkv, D, K.data(), V.data(), mode ? idx.data() : nullptr, S, mode ? nullptr : ce, sc, out.data());
    double worst = 0;
    for (int m = 0; m < M; ++m)
      for (int h = 0; h < Hq; ++h) {
        const int g = h / (Hq / Hkv);
        std::vector<int> ps;
        if (mode) { for (int j = 0; j < S; ++j) if (idx[(size_t)m * S + j] >= 0) ps.push_back(idx[(size_t)m * S + j]); }
        else for (int p = 0; p < ce[m]; ++p) ps.push_back(p);
        std::vector<double> s(ps.size());
        double mx = -1e300;
        for (size_t j = 0; j < ps.size(); ++j) {
          double d = 0;
          for (int e = 0; e < D; ++e) d += (double)q[((size_t)m * Hq + h) * D + e] * bf16_to_f32(K[((size_t)ps[j] * Hkv + g) * D + e]);
          s[j] = d * sc; mx = std::max(mx, s[j]);
        }
        double z = 0;
        for (auto& v : s) { v = std::exp(v - mx); z += v; }
        for (int e = 0; e < D; ++e) {
          double o = 0;
          for (size_t j = 0; j < ps.size(); ++j) o += s[j] / z * bf16_to_f32(V[((size_t)ps[j] * Hkv + g) * D + e]);
          worst = std::max(worst, std::fabs(o - out[((size_t)m * Hq + h) * D + e]));
        }
      }
    hash_floats(out.data(), out.size());
    const bool ok = worst < 2e-5;
    if (!ok) ++fails;
    printf("%s attention %s: max abs error %.2e\n", ok ? "PASS" : "FAIL", mode ? "index list (-1 holes)" : "causal", worst);
  }
}

void run_all() {
  std::mt19937 rng(1234);
  const int Ms[] = {1, 2, 3, 8, 9, 33, 128};
  struct Shape { WType t; int N, K; SType s; int bn, bk; };
  const Shape shapes[] = {
      {WType::BF16, 37, 264, SType::F32, 128, 128}, {WType::F32, 29, 136, SType::F32, 128, 128},
      {WType::FP8, 300, 384, SType::F32, 128, 128}, {WType::FP8, 70, 256, SType::BF16, 128, 128}, {WType::FP8, 45, 96, SType::E8M0, 32, 32},
      {WType::NVFP4, 41, 320, SType::E4M3, 128, 128}, {WType::MXFP4, 39, 288, SType::E8M0, 128, 128},
  };
  for (const Shape& sh : shapes) {
    Case c = make_case(sh.t, sh.N, sh.K, sh.s, rng, sh.bn, sh.bk);
    for (int M : Ms) check_matmul(c, M, 0, sh.N, rng);
    check_matmul(c, 4, 3, sh.N - 2, rng);          // view, GEMV
    check_matmul(c, 20, 5, sh.N - 1, rng);         // view, GEMM
    check_matmul(c, 6, 1, sh.N, rng, true);        // matmul_multi, GEMV
    check_matmul(c, 12, 0, sh.N, rng, true);       // matmul_multi, GEMM
  }
  check_attention(rng);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--hash") {
    run_all();
    printf("HASH %016llx FAILS %d\n", (unsigned long long)hash_acc, fails);
    return fails ? 1 : 0;
  }
  printf("pool: %d threads on %d node(s)\n", Pool::get().threads(), Pool::get().nodes());
  run_all();
  // determinism across pool sizes: children with different thread counts must hash identically
  std::string hashes[3];
  const char* th[3] = {"1", "3", ""};
  for (int i = 0; i < 3; ++i) {
    const std::string cmd = std::string(th[i][0] ? "HIVE_CPU_POOL_THREADS=" + std::string(th[i]) + " " : "") + argv[0] + " --hash";
    FILE* f = popen(cmd.c_str(), "r");
    char line[256] = {0};
    while (f && fgets(line, sizeof line, f)) if (!strncmp(line, "HASH", 4)) hashes[i] = line;
    if (f) pclose(f);
  }
  const bool det = !hashes[0].empty() && hashes[0] == hashes[1] && hashes[1] == hashes[2];
  if (!det) ++fails;
  printf("%s determinism across pool sizes 1 / 3 / default: %s", det ? "PASS" : "FAIL", hashes[0].c_str());
  printf("%s: %d failure(s)\n", fails ? "FAIL" : "ALL PASS", fails);
  return fails ? 1 : 0;
}
