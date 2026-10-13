// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// FN decode GEMV shapes on the GPU (glm_gemv_bf16_multi, the path QwenEngine::lin takes for ≤ 8 rows): time and bandwidth per shape at
//   M = 1 and 4, each shape cycling over enough copies (≥ 2 GB in all) that L2 holds none of them — like a decode step's distinct layers.
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
                      {"sh_gu 1280x2560", 1280, 2560}, {"sh_down 2560x640", 2560, 640}, {"lm_head 248320x2560", 248320, 2560}};
  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
  bf16 *x, *y; CUDA_CHECK(cudaMalloc(&x, 64 * 16384 * 2)); CUDA_CHECK(cudaMalloc(&y, (size_t)64 * 262144 * 2));
  CUDA_CHECK(cudaMemset(x, 0, 64 * 16384 * 2));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (const S& s : shapes) {
    const size_t bytes = (size_t)s.N * s.K * 2;
    const int copies = (int)std::max<size_t>(1, ((size_t)2 << 30) / bytes);
    std::vector<bf16*> W(copies);
    for (auto& w : W) { CUDA_CHECK(cudaMalloc(&w, bytes)); CUDA_CHECK(cudaMemset(w, 0, bytes)); }
    for (int M : {1, 4, 16, 32}) {   // ≥ 16: the batched verify rows (int8 only)
      if (M <= 4) {
      for (int i = 0; i < copies; ++i) { hive::glm::GemvSeg g{W[i], x, y, s.N, s.K, s.K, s.N}; hive::glm::glm_gemv_bf16_multi(&g, 1, M, st); }
      const int reps = std::max(copies, 64);
      cudaEventRecord(a, st);
      for (int i = 0; i < reps; ++i) { hive::glm::GemvSeg g{W[i % copies], x, y, s.N, s.K, s.K, s.N}; hive::glm::glm_gemv_bf16_multi(&g, 1, M, st); }
      cudaEventRecord(b, st); cudaEventSynchronize(b);
      float ms = 0; cudaEventElapsedTime(&ms, a, b);
      printf("%-22s M=%d glm: %8.1f us · %6.0f GB/s", s.name, M, 1000.0 * ms / reps, bytes / (ms / reps * 1e-3) / 1e9);
      if (hive::qwen::fn_gemv_ok(s.N, s.K, M)) {
        float* ws; CUDA_CHECK(cudaMalloc(&ws, hive::qwen::fn_gemv_ws_bytes(s.N, s.K, M)));
        for (int i = 0; i < copies; ++i) hive::qwen::fn_gemv(W[i], s.N, s.K, x, s.K, M, y, s.N, nullptr, ws, st);
        cudaEventRecord(a, st);
        for (int i = 0; i < reps; ++i) hive::qwen::fn_gemv(W[i % copies], s.N, s.K, x, s.K, M, y, s.N, nullptr, ws, st);
        cudaEventRecord(b, st); cudaEventSynchronize(b);
        float ms2 = 0; cudaEventElapsedTime(&ms2, a, b);
        printf(" · fn (parts %d): %8.1f us · %6.0f GB/s", hive::qwen::fn_gemv_parts(s.N, s.K), 1000.0 * ms2 / reps, bytes / (ms2 / reps * 1e-3) / 1e9);
        cudaFree(ws);
      }
      } else printf("%-22s M=%d", s.name, M);
      for (int fmt : {1, 2}) {   // 8-bit copies (timing only: zero weights / unit scales)
        const int gs = fmt == 1 ? s.K : (s.K % 128 == 0 ? 128 : s.K);
        if (!hive::qwen::fn_gemv8_ok(s.N, s.K, gs, M)) continue;
        const int ng = (s.K + gs - 1) / gs;
        const int c8 = (int)std::max<size_t>(1, ((size_t)2 << 30) / (bytes / 2));
        std::vector<uint8_t*> Q(c8); float* S;
        for (auto& q : Q) { CUDA_CHECK(cudaMalloc(&q, bytes / 2)); CUDA_CHECK(cudaMemset(q, 0, bytes / 2)); }
        CUDA_CHECK(cudaMalloc(&S, (size_t)s.N * ng * 4)); CUDA_CHECK(cudaMemset(S, 0, (size_t)s.N * ng * 4));
        for (int i = 0; i < c8; ++i) hive::qwen::fn_gemv8(Q[i], S, fmt, gs, s.N, s.K, x, s.K, M, y, s.N, st);
        const int r8 = std::max(c8, 64);
        cudaEventRecord(a, st);
        for (int i = 0; i < r8; ++i) hive::qwen::fn_gemv8(Q[i % c8], S, fmt, gs, s.N, s.K, x, s.K, M, y, s.N, st);
        cudaEventRecord(b, st); cudaEventSynchronize(b);
        float m8 = 0; cudaEventElapsedTime(&m8, a, b);
        printf(" · %s: %7.1f us %5.0f GB/s", fmt == 1 ? "fp8" : "int8", 1000.0 * m8 / r8, bytes / 2 / (m8 / r8 * 1e-3) / 1e9);
        for (auto& q : Q) cudaFree(q);
        cudaFree(S);
      }
      printf("\n");
    }
    for (auto& w : W) cudaFree(w);
  }
}
