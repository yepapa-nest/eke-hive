// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Decode / verify GEMV microbenchmark of the 27B shapes (hive/q27/q27_kernels.h k::gemv / k::gemv_gu), NVFP4 weights with random codes:
//   time per call and weight GB/s for M = 1 … 64 rows, and bit identity of every row across M (the M-row call against M = 1 calls).
//   HIVE_Q27_SW chooses the kernel variant as in the engine. Usage: q27_gemv_bench [iters] [shape index | -1 all] [M list, comma]
//   Each shape is allocated in enough copies (≥ 512 MB) that consecutive calls read DRAM, not L2 (the engine reads each matrix once a step).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "hive/q27/q27_kernels.h"
using namespace hive::q27::k;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
static GMat make(int N, int K, std::mt19937& rng) {
  GMat W; W.fmt = WFmt::NVFP4; W.N = N; W.K = K; W.g = 1.f / 1024.f;
  W.act = getenv("HIVE_Q27_A4_MLP") && atoi(getenv("HIVE_Q27_A4_MLP")) == 1 ? 4 : 0;   // the W4A4 route under test
  std::vector<uint8_t> w((size_t)N * K / 2), s((size_t)N * K / 16);
  for (auto& b : w) b = (uint8_t)rng();
  for (auto& b : s) b = (uint8_t)(0x30 + rng() % 16);   // e4m3 group scales in a tame range
  uint8_t *dw, *ds;
  CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&ds, s.size()));
  CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds, s.data(), s.size(), cudaMemcpyHostToDevice));
  W.w = dw; W.s = ds;
  if (W.act == 4) {   // the W4A4 route reads a repacked copy
    uint8_t* pk; const size_t nb = a4_repack(W, nullptr, 0);
    CK(cudaMalloc(&pk, nb)); a4_repack(W, pk, 0); CK(cudaDeviceSynchronize());
    W.a4w = pk; W.a4s = pk + (size_t)((N + 15) / 16) * 16 * K / 2;
  }
  return W;
}
int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 50;
  const int only = argc > 2 ? atoi(argv[2]) : -1;
  std::mt19937 rng(7);
  struct Shape { const char* name; int N, K; bool gu; };
  const Shape shapes[] = {{"gate+up 17408x5120", 17408, 5120, true}, {"down 5120x17408", 5120, 17408, false},
                          {"lin qkvz 16480x5120", 16480, 5120, false}, {"o / lin_out 5120x6144", 5120, 6144, false},
                          {"lm_head 248320x5120", 248320, 5120, false}};
  const int Ms[] = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64};
  constexpr int NM = sizeof(Ms) / sizeof(Ms[0]), MMAX = 64;
  cudaStream_t st; CK(cudaStreamCreate(&st));
  gemv_prepare();
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double tot_by_m[NM] = {};
  bool all_ok = true;
  for (int si = 0; si < 5; ++si) {
    const Shape& sh = shapes[si];
    if (only >= 0 && si != only) continue;
    const double one = (double)sh.N * sh.K * (0.5 + 1.0 / 16) * (sh.gu ? 2 : 1);
    const int copies = std::max(2, (int)std::ceil(512e6 / one));
    std::vector<GMat> Ws, Us;
    for (int c = 0; c < copies; ++c) { Ws.push_back(make(sh.N, sh.K, rng)); Us.push_back(sh.gu ? make(sh.N, sh.K, rng) : Ws.back()); }
    GMat W = Ws[0], U = Us[0];
    const int K = sh.K, ldy = sh.gu ? 2 * sh.N : sh.N;
    std::vector<uint16_t> xh((size_t)MMAX * K);
    for (auto& v : xh) { const float f = (float)((int)(rng() % 2001) - 1000) / 1000.f; uint32_t u; memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
    bf16 *x, *y, *y1;
    CK(cudaMalloc(&x, xh.size() * 2)); CK(cudaMalloc(&y, (size_t)MMAX * ldy * 2)); CK(cudaMalloc(&y1, (size_t)ldy * 2));
    CK(cudaMemcpy(x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    const double wbytes = (double)sh.N * K * (0.5 + 1.0 / 16) * (sh.gu ? 2 : 1);
    printf("%s\n", sh.name);
    for (int mi = 0; mi < NM; ++mi) {
      const int M = Ms[mi];
      if (getenv("GB_M") && atoi(getenv("GB_M")) != M) continue;   // one row count only (profiling)
      int cur = 0;
      auto run = [&](int m, const bf16* xi, bf16* yo) {
        const GMat& w = Ws[cur], & u = Us[cur];
        if (sh.gu) gemv_gu(w, u, xi, K, m, yo, sh.N, st); else gemv(w, xi, K, m, yo, nullptr, ldy, st);
      };
      auto runc = [&](int m, const bf16* xi, bf16* yo) { cur = (cur + 1) % copies; run(m, xi, yo); };
      for (int i = 0; i < 3; ++i) runc(M, x, y);
      CK(cudaEventRecord(e0, st));
      for (int i = 0; i < iters; ++i) runc(M, x, y);
      CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
      float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
      const double us = 1000.0 * ms / iters;
      tot_by_m[mi] += us;
      // bit identity: rows 0 and M−1 of the M-row call against single-row calls
      bool ok = true;
      const int outw = sh.gu ? sh.N : sh.N;
      std::vector<uint16_t> a(outw), b(outw);
      for (int r : {0, std::min(M - 1, 32), M - 1}) {
        run(1, x + (size_t)r * K, y1);
        CK(cudaMemcpy(a.data(), y + (size_t)r * (sh.gu ? sh.N : ldy), outw * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y1, outw * 2, cudaMemcpyDeviceToHost));
        ok &= memcmp(a.data(), b.data(), outw * 2) == 0;
      }
      all_ok &= ok;
      double rel = -1;
      if (Ws[cur].act == 4) {   // error of the W4A4 route against the bf16-activation route (same weights)
        GMat w0 = Ws[cur], u0 = Us[cur]; w0.act = 0; u0.act = 0;
        bf16* yr; CK(cudaMalloc(&yr, (size_t)MMAX * ldy * 2));
        run(M, x, y);
        if (sh.gu) gemv_gu(w0, u0, x, K, M, yr, sh.N, st); else gemv(w0, x, K, M, yr, nullptr, ldy, st);
        std::vector<uint16_t> ya((size_t)M * ldy), yo((size_t)M * ldy);
        CK(cudaMemcpy(ya.data(), y, ya.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(yo.data(), yr, yo.size() * 2, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        const int w = sh.gu ? sh.N : sh.N;
        for (int r = 0; r < M; ++r) for (int c = 0; c < w; ++c) {
          uint32_t ua = (uint32_t)ya[(size_t)r * (sh.gu ? sh.N : ldy) + c] << 16, ub = (uint32_t)yo[(size_t)r * (sh.gu ? sh.N : ldy) + c] << 16;
          float fa, fb; memcpy(&fa, &ua, 4); memcpy(&fb, &ub, 4);
          num += (double)(fa - fb) * (fa - fb); den += (double)fb * fb;
        }
        rel = std::sqrt(num / std::max(den, 1e-30));
        CK(cudaFree(yr));
      }
      printf("  M=%2d  %8.1f us  %6.0f GB/s  rows identical to M=1: %s  rel err vs bf16 act %.4f\n", M, us, wbytes / us / 1e3, ok ? "yes" : "NO", rel);
    }
    CK(cudaFree(x)); CK(cudaFree(y)); CK(cudaFree(y1));
    for (int c = 0; c < copies; ++c) { cudaFree((void*)Ws[c].w); cudaFree((void*)Ws[c].s); if (sh.gu) { cudaFree((void*)Us[c].w); cudaFree((void*)Us[c].s); } }
  }
  printf("sum over shapes (one layer-ish set) us:");
  for (int mi = 0; mi < NM; ++mi) printf(" M%d=%.0f", Ms[mi], tot_by_m[mi]);
  printf("\nbit identity across M: %s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
