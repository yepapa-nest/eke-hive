// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// FN dense projections at the batched-verify row counts (M = 8 · 16 · 32 · 48): time and bandwidth per shape for every kernel
//   QwenEngine::lin can take — bf16: glm_gemv_bf16_multi (≤ 8 rows), fn_gemv (split-K, the HC down shape), fn_gemv_tc (tensor cores,
//   batch-invariant); 8-bit: fn_gemv8 with fmt 1 (FP8) · 2 (INT8) · 3 (MXFP8, the FP4-Local dense format). Each shape cycles over ≥ 2 GB
//   of copies so L2 holds none of them (a decode step streams every layer's weights once). Timing only: zero weights, unit scales.
//   2026-10-31: the C8 kernel profile put fn_gemv + gemv8 at ~32 % of the step — this tells whether that is bandwidth or kernel shape.
#include <cuda_bf16.h>
#include <cstdio>
#include <algorithm>
#include <vector>
#include "hive/common.h"
#include "hive/glm/glm_decode.h"
#include "hive/qwen/fn_gemv.h"

using bf16 = __nv_bfloat16;
int main() {
  struct S { const char* name; int N, K; };
  const S shapes[] = {{"hc_down 324x10240", 324, 10240}, {"hc_up 10240x320", 10240, 320}, {"gdn_in 16480x2560", 16480, 2560},
                      {"gdn_out 2560x6144", 2560, 6144}, {"attn_in 13952x2560", 13952, 2560}, {"attn_out 2560x6144", 2560, 6144},
                      {"sh_gu 1280x2560", 1280, 2560}, {"sh_down 2560x640", 2560, 640}};
  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
  bf16 *x, *y; CUDA_CHECK(cudaMalloc(&x, 64 * 16384 * 2)); CUDA_CHECK(cudaMalloc(&y, (size_t)64 * 262144 * 2));
  CUDA_CHECK(cudaMemset(x, 0, 64 * 16384 * 2));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto timeit = [&](int reps, auto&& f) { cudaEventRecord(a, st); for (int i = 0; i < reps; ++i) f(i); cudaEventRecord(b, st); cudaEventSynchronize(b);
                                          float ms = 0; cudaEventElapsedTime(&ms, a, b); return 1000.0 * ms / reps; };   // µs per call
  printf("%-22s %3s  %-26s %-26s %-26s | %-22s %-22s %-22s\n", "shape", "M", "bf16 glm(<=8)", "bf16 fn_gemv", "bf16 fn_gemv_tc", "fp8 gemv8", "int8 gemv8", "mxfp8 gemv8");
  for (const S& s : shapes) {
    const size_t bytes = (size_t)s.N * s.K * 2;
    const int copies = (int)std::max<size_t>(1, ((size_t)2 << 30) / bytes);
    std::vector<bf16*> W(copies);
    for (auto& w : W) { CUDA_CHECK(cudaMalloc(&w, bytes)); CUDA_CHECK(cudaMemset(w, 0, bytes)); }
    const int c8 = (int)std::max<size_t>(1, ((size_t)2 << 30) / (bytes / 2));
    std::vector<uint8_t*> Q(c8);
    for (auto& q : Q) { CUDA_CHECK(cudaMalloc(&q, bytes / 2)); CUDA_CHECK(cudaMemset(q, 0, bytes / 2)); }
    float* S32; CUDA_CHECK(cudaMalloc(&S32, (size_t)s.N * ((s.K + 127) / 128) * 4)); CUDA_CHECK(cudaMemset(S32, 0, (size_t)s.N * ((s.K + 127) / 128) * 4));
    uint8_t* Smx; CUDA_CHECK(cudaMalloc(&Smx, (size_t)s.N * (s.K / 32))); CUDA_CHECK(cudaMemset(Smx, 127, (size_t)s.N * (s.K / 32)));
    for (int M : {8, 16, 32, 48}) {
      char col[6][64]; for (auto& c : col) c[0] = 0;
      const int reps = std::max(copies, 64), r8 = std::max(c8, 64);
      auto fmt = [&](char* out, double us, size_t bytes_read) { snprintf(out, 64, "%7.1f us %5.0f GB/s", us, bytes_read / (us * 1e-6) / 1e9); };
      if (M <= 8) {
        for (int i = 0; i < copies; ++i) { hive::glm::GemvSeg g{W[i], x, y, s.N, s.K, s.K, s.N}; hive::glm::glm_gemv_bf16_multi(&g, 1, M, st); }
        fmt(col[0], timeit(reps, [&](int i) { hive::glm::GemvSeg g{W[i % copies], x, y, s.N, s.K, s.K, s.N}; hive::glm::glm_gemv_bf16_multi(&g, 1, M, st); }), bytes);
      }
      if (hive::qwen::fn_gemv_ok(s.N, s.K, M)) {
        float* ws; CUDA_CHECK(cudaMalloc(&ws, hive::qwen::fn_gemv_ws_bytes(s.N, s.K, M)));
        for (int i = 0; i < copies; ++i) hive::qwen::fn_gemv(W[i], s.N, s.K, x, s.K, M, y, s.N, nullptr, ws, st);
        fmt(col[1], timeit(reps, [&](int i) { hive::qwen::fn_gemv(W[i % copies], s.N, s.K, x, s.K, M, y, s.N, nullptr, ws, st); }), bytes);
        cudaFree(ws);
      }
      if (s.K % 32 == 0) {
        for (int i = 0; i < copies; ++i) hive::qwen::fn_gemv_tc(W[i], s.N, s.K, x, s.K, M, y, s.N, nullptr, st);
        fmt(col[2], timeit(reps, [&](int i) { hive::qwen::fn_gemv_tc(W[i % copies], s.N, s.K, x, s.K, M, y, s.N, nullptr, st); }), bytes);
      }
      for (int f : {1, 2, 3}) {
        const int gs = f == 3 ? 32 : (f == 1 ? s.K : (s.K % 128 == 0 ? 128 : s.K));
        if (f == 3 && s.K % 32 != 0) continue;
        if (!hive::qwen::fn_gemv8_ok(s.N, s.K, gs, M)) continue;
        const float* Sp = f == 3 ? reinterpret_cast<const float*>(Smx) : S32;
        for (int i = 0; i < c8; ++i) hive::qwen::fn_gemv8(Q[i], Sp, f, gs, s.N, s.K, x, s.K, M, y, s.N, st);
        const size_t rb = bytes / 2 + (f == 3 ? (size_t)s.N * (s.K / 32) : 0);
        fmt(col[2 + f], timeit(r8, [&](int i) { hive::qwen::fn_gemv8(Q[i % c8], Sp, f, gs, s.N, s.K, x, s.K, M, y, s.N, st); }), rb);
      }
      printf("%-22s %3d  %-26s %-26s %-26s | %-22s %-22s %-22s\n", s.name, M, col[0], col[1], col[2], col[3], col[4], col[5]);
    }
    for (auto& w : W) cudaFree(w);
    for (auto& q : Q) cudaFree(q);
    cudaFree(S32); cudaFree(Smx);
  }
}
