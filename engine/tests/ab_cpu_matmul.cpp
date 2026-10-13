// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// In-process bitwise A/B of the CPU matmul: hive::cpu::matmul / matmul_multi (current kernels) against hive::cpu::ab_legacy (the snapshot in
//   src/cpu/cpu_matmul_legacy.cpp) on every weight format (BF16, F32, FP8 with 128×128 / 32×32 blocks and F32 / BF16 / E8M0 scales, NVFP4,
//   MXFP4, I8), M = 1..16 and larger GEMM sizes up to 450, odd shapes, MatView row ranges, and matmul_multi with per-job X / M / ldx.
//   Every output must be byte-identical (outputs start from different garbage so an unwritten element also fails).
// usage: ab_cpu_matmul            (exit 1 on any difference)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "hive/cpu/cpu_backend.h"

namespace hive::cpu::ab_legacy {
void matmul(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy);
void matmul_multi(const MatJob* jobs_in, int n_jobs, const float* X, int M, int ldx);
}  // namespace hive::cpu::ab_legacy

using namespace hive::cpu;

namespace {
std::mt19937 g(42);
long cases = 0, diffs = 0;
std::vector<float> rf(size_t n) { std::normal_distribution<float> d(0, 1); std::vector<float> v(n); for (auto& x : v) x = d(g); return v; }
std::vector<uint8_t> rb(size_t n) { std::vector<uint8_t> v(n); for (auto& x : v) x = (uint8_t)g(); return v; }
uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

struct M_ { Mat m; const char* name; };

void cmp(const char* what, const M_& mm, int M, const std::vector<float>& a, const std::vector<float>& b) {
  ++cases;
  if (a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * 4) == 0) return;
  ++diffs;
  size_t first = 0;
  while (first < a.size() && memcmp(&a[first], &b[first], 4) == 0) ++first;
  printf("DIFF %s %s N=%d K=%d M=%d at %zu: %.9g vs %.9g\n", what, mm.name, mm.m.N, mm.m.K, M, first, first < a.size() ? a[first] : 0.f,
         first < b.size() ? b[first] : 0.f);
}

void run_one(const M_& mm, int M) {
  const Mat& m = mm.m;
  const int ldx = m.K + 5, ldy = m.N + 3;
  const auto X = rf((size_t)M * ldx);
  std::vector<float> A((size_t)M * ldy, -1.f), B((size_t)M * ldy, -2.f);
  for (int i = 0; i < M; ++i) for (int n = m.N; n < ldy; ++n) A[(size_t)i * ldy + n] = B[(size_t)i * ldy + n] = 0.f;  // padding untouched
  matmul(m, X.data(), M, ldx, A.data(), ldy);
  ab_legacy::matmul(MatView{&m, 0, m.N}, X.data(), M, ldx, B.data(), ldy);
  cmp("full", mm, M, A, B);
  // a view that crosses the NUMA split (if any) and starts mid-block
  const int r0 = std::min(m.N - 1, 7), r1 = std::max(r0 + 1, m.N - 5), w = r1 - r0;
  std::vector<float> C((size_t)M * w, 3.f), D((size_t)M * w, 3.f);
  matmul(MatView{&m, r0, r1}, X.data(), M, ldx, C.data(), w);
  ab_legacy::matmul(MatView{&m, r0, r1}, X.data(), M, ldx, D.data(), w);
  cmp("view", mm, M, C, D);
}

// many jobs with their own X, M and ldx (MoE-like) over row ranges of one matrix, plus a shared-X job
void run_multi(const M_& mm) {
  const Mat& m = mm.m;
  const int Mj[] = {1, 3, 8, 9, 17, 40, 2, 120, 5, 64, 11, 7, 4, 6};
  const int nj = 14, rows = std::max(1, m.N / nj);
  std::vector<std::vector<float>> X(nj), Ya(nj), Yb(nj);
  std::vector<MatJob> ja, jb;
  for (int i = 0; i < nj; ++i) {
    const int r0 = std::min(m.N - 1, i * rows), r1 = i == nj - 1 ? m.N : std::min(m.N, r0 + rows);
    if (r1 <= r0) continue;
    const int ldx = m.K + (i % 3) * 8;
    X[i] = rf((size_t)Mj[i] * ldx);
    Ya[i].assign((size_t)Mj[i] * (r1 - r0), -1.f); Yb[i].assign((size_t)Mj[i] * (r1 - r0), -2.f);
    ja.push_back(MatJob{MatView{&m, r0, r1}, Ya[i].data(), r1 - r0, X[i].data(), Mj[i], ldx});
    jb.push_back(MatJob{MatView{&m, r0, r1}, Yb[i].data(), r1 - r0, X[i].data(), Mj[i], ldx});
  }
  matmul_multi(ja.data(), (int)ja.size(), nullptr, 0, 0);
  ab_legacy::matmul_multi(jb.data(), (int)jb.size(), nullptr, 0, 0);
  for (int i = 0; i < nj; ++i) if (!Ya[i].empty()) cmp("multi", mm, Mj[i], Ya[i], Yb[i]);
  // shared X, call-level M, two halves
  for (int M : {2, 6, 8, 30}) {
    const auto Xs = rf((size_t)M * m.K);
    const int mid = m.N / 2;
    std::vector<float> A((size_t)M * m.N, -1.f), B((size_t)M * m.N, -2.f);
    MatJob a2[2] = {{MatView{&m, 0, mid}, A.data(), m.N}, {MatView{&m, mid, m.N}, A.data() + mid, m.N}};
    MatJob b2[2] = {{MatView{&m, 0, mid}, B.data(), m.N}, {MatView{&m, mid, m.N}, B.data() + mid, m.N}};
    matmul_multi(a2, 2, Xs.data(), M, m.K);
    ab_legacy::matmul_multi(b2, 2, Xs.data(), M, m.K);
    cmp("multi-shared", mm, M, A, B);
  }
}

std::vector<M_> make_mats(int N, int K) {
  std::vector<M_> v;
  std::normal_distribution<float> d(0, .05f);
  std::vector<uint16_t> bw((size_t)N * K);
  for (auto& x : bw) x = f2b(d(g));
  v.push_back({make_mat(WType::BF16, N, K, bw.data()), "BF16"});
  std::vector<float> fw((size_t)N * K);
  for (auto& x : fw) x = d(g);
  v.push_back({make_mat(WType::F32, N, K, fw.data()), "F32"});
  auto f8 = rb((size_t)N * K);
  for (auto& x : f8) if ((x & 0x7f) == 0x7f) x &= 0xfe;
  {
    std::vector<float> s((size_t)((N + 127) / 128) * ((K + 127) / 128));
    for (auto& x : s) x = 0.001f + 0.01f * (float)(g() % 1000) / 1000.f;
    v.push_back({make_mat(WType::FP8, N, K, f8.data(), s.data(), SType::F32), "FP8/128/f32"});
    std::vector<uint16_t> sb(s.size());
    for (auto& x : sb) x = (uint16_t)(0x3c00 + (g() & 0xff));
    v.push_back({make_mat(WType::FP8, N, K, f8.data(), sb.data(), SType::BF16), "FP8/128/bf16"});
    std::vector<uint8_t> se((size_t)((N + 31) / 32) * ((K + 31) / 32));
    for (auto& x : se) x = (uint8_t)(110 + g() % 20);
    v.push_back({make_mat(WType::FP8, N, K, f8.data(), se.data(), SType::E8M0, 1.f, 32, 32), "FP8/32/e8m0"});
  }
  if (K % 32 == 0) {
    auto nw = rb((size_t)N * K / 2);
    auto ns = rb((size_t)N * K / 16);
    for (auto& x : ns) { x = (uint8_t)(0x20 + (x & 0x3f)); if ((x & 0x7f) == 0x7f) x = 0x30; }
    v.push_back({make_mat(WType::NVFP4, N, K, nw.data(), ns.data(), SType::E4M3, 0.0123f), "NVFP4"});
    auto ms = rb((size_t)N * K / 32);
    for (auto& x : ms) x = (uint8_t)(118 + x % 16);
    v.push_back({make_mat(WType::MXFP4, N, K, nw.data(), ms.data(), SType::E8M0), "MXFP4"});
  }
  v.push_back({make_mat_i8(N, K, bw.data(), 64), "I8/64"});
  v.push_back({make_mat_i8(N, K, bw.data(), 128), "I8/128"});
  return v;
}
}  // namespace

int main() {
  printf("pool: %d threads on %d node(s)\n", Pool::get().threads(), Pool::get().nodes());
  struct S { int N, K; };
  const S shapes[] = {{37, 264}, {300, 640}, {1300, 2560}, {517, 1056}, {64, 7168}};
  std::vector<int> Ms;
  for (int M = 1; M <= 16; ++M) Ms.push_back(M);
  for (int M : {24, 33, 64, 100, 192, 193, 450}) Ms.push_back(M);
  for (const S& s : shapes) {
    auto mats = make_mats(s.N, s.K);
    for (auto& mm : mats) {
      for (int M : Ms) run_one(mm, M);
      run_multi(mm);
    }
    for (auto& mm : mats) free_mat(mm.m);
    printf("shape N=%d K=%d done: %ld cases, %ld diffs\n", s.N, s.K, cases, diffs);
  }
  printf("%s: %ld cases, %ld differ\n", diffs ? "FAIL" : "ALL IDENTICAL", cases, diffs);
  return diffs ? 1 : 0;
}
