// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Decode dense GEMV microbenchmark — kernel time and effective bandwidth (weight bytes / time) for M=1/8 at the real per-layer shapes.
// How far this falls short of VRAM bandwidth (~1.6 TB/s) motivates the segA optimizations. Usage: ./build/bench_gemv
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hive/common.h"
#include "hive/cublas_ops.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "test_util.h"

using namespace hive;
static float bits2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }

namespace {
uint8_t* dev_rand(size_t n, uint8_t lo, uint8_t hi) {
  std::vector<uint8_t> h(n);
  uint32_t s = 12345;
  for (size_t i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; h[i] = (uint8_t)(lo + (s >> 24) % (hi - lo + 1)); }
  uint8_t* d; CUDA_CHECK(cudaMalloc(&d, n)); CUDA_CHECK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice));
  return d;
}
template <class F>
float time_ms(cudaStream_t st, F f, int iters = 50) {
  for (int i = 0; i < 3; ++i) f();
  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0, st));
  for (int i = 0; i < iters; ++i) f();
  CUDA_CHECK(cudaEventRecord(e1, st));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}
}  // namespace

int main() {
  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
  Blas blas(st);
  struct Shape { const char* name; int N, K; bool fp4; };
  const Shape shapes[] = {{"wq_a", 1280, 5120, false}, {"wq_b", 32768, 1280, false}, {"wkv", 512, 5120, false},
                          {"wo_b", 5120, 8192, false}, {"sh_w1", 2304, 5120, false}, {"sh_w2", 5120, 2304, false},
                          {"idx_wq_b", 4096, 1280, false}, {"expert_w1(fp4)", 2304, 5120, true}, {"expert_w2(fp4)", 5120, 2304, true}};
  printf("%-16s %7s %6s | %-9s %-9s | %-9s %-9s   (GB/s over weight bytes)\n", "shape", "N", "K", "M=1 ms", "GB/s", "M=8 ms", "GB/s");
  for (const Shape& s : shapes) {
    const size_t wb = s.fp4 ? (size_t)s.N * s.K / 2 : (size_t)s.N * s.K;
    const size_t sb = (size_t)s.N * s.K / 32;
    uint8_t* B = dev_rand(wb, 0, 255);
    uint8_t* Bs = dev_rand(sb, 120, 130);
    uint8_t* A = dev_rand((size_t)8 * s.K, 0, 0x7E);
    uint8_t* As = dev_rand((size_t)8 * s.K / 32, 120, 130);
    bf16* C; CUDA_CHECK(cudaMalloc(&C, (size_t)8 * s.N * 2));
    float t1 = time_ms(st, [&] { k::gemm_bs(A, As, B, Bs, s.fp4, 1, s.N, s.K, C, nullptr, st); });
    float t8 = time_ms(st, [&] { k::gemm_bs(A, As, B, Bs, s.fp4, 8, s.N, s.K, C, nullptr, st); });
    printf("%-16s %7d %6d | %7.4f   %7.0f   | %7.4f   %7.0f\n", s.name, s.N, s.K, t1, (wb + sb) / t1 / 1e6, t8, (wb + sb) / t8 / 1e6);
    cudaFree(B); cudaFree(Bs); cudaFree(A); cudaFree(As); cudaFree(C);
  }
  // Grouped experts (6 resident, one M=1 row each): w13 + w2
  {
    const int I = 2304, K = 5120, G = 6;
    const size_t wsz = (size_t)I * K / 2, ssz = (size_t)I * K / 32;
    const size_t rec = 3 * (wsz + ssz);
    uint8_t* recs = dev_rand(rec * G, 0, 255);
    std::vector<k::GroupDesc> gd(G);
    for (int g = 0; g < G; ++g) {
      uint8_t* r = recs + rec * g;
      gd[g] = {r, r + wsz, r + wsz + ssz, r + 2 * wsz + ssz, r + 2 * wsz + 2 * ssz, r + 3 * wsz + 2 * ssz, g, 1};
    }
    k::GroupDesc* gdd; CUDA_CHECK(cudaMalloc(&gdd, sizeof(k::GroupDesc) * G)); CUDA_CHECK(cudaMemcpy(gdd, gd.data(), sizeof(k::GroupDesc) * G, cudaMemcpyHostToDevice));
    uint8_t* A = dev_rand((size_t)G * K, 0, 0x7E); uint8_t* As = dev_rand((size_t)G * K / 32, 120, 130);
    float* rw; CUDA_CHECK(cudaMalloc(&rw, G * 4)); CUDA_CHECK(cudaMemset(rw, 0, G * 4));
    bf16* y; CUDA_CHECK(cudaMalloc(&y, (size_t)G * I * 2));
    uint8_t* yq = dev_rand((size_t)G * I, 0, 0x7E); uint8_t* ys = dev_rand((size_t)G * I / 32, 120, 130);
    bf16* eo; CUDA_CHECK(cudaMalloc(&eo, (size_t)G * K * 2));
    float t13 = time_ms(st, [&] { k::gemv_grouped_w13(gdd, G, A, As, I, K, rw, 10.f, y, st); });
    float t2 = time_ms(st, [&] { k::gemv_grouped_w2(gdd, G, yq, ys, K, I, eo, st); });
    printf("%-16s %7s %6s | w13 %7.4f ms %5.0f GB/s | w2 %7.4f ms %5.0f GB/s | sum %.3f ms (6 experts = %.1f MB)\n", "grouped x6", "", "",
           t13, 2.0 * G * (wsz + ssz) / t13 / 1e6, t2, 1.0 * G * (wsz + ssz) / t2 / 1e6, t13 + t2, 3.0 * G * (wsz + ssz) / 1e6);
    // Tensor-core (mxf8f6f4) version: compare results on the same input (only the fp32 accumulation order differs) + timing. rw is 1 (so the post-SwiGLU comparison is meaningful)
    {
      std::vector<float> ones(G, 1.f); CUDA_CHECK(cudaMemcpy(rw, ones.data(), G * 4, cudaMemcpyHostToDevice));
      bf16 *y2, *eo2; CUDA_CHECK(cudaMalloc(&y2, (size_t)G * I * 2)); CUDA_CHECK(cudaMalloc(&eo2, (size_t)G * K * 2));
      k::gemv_grouped_w13(gdd, G, A, As, I, K, rw, 10.f, y, st);
      k::mx_grouped_w13(gdd, G, A, As, I, K, rw, 10.f, y2, st);
      k::gemv_grouped_w2(gdd, G, yq, ys, K, I, eo, st);
      k::mx_grouped_w2(gdd, G, yq, ys, K, I, eo2, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      auto cmp = [&](const bf16* a, const bf16* b, size_t n, const char* name) {
        std::vector<uint16_t> ha(n), hb(n);
        CUDA_CHECK(cudaMemcpy(ha.data(), a, n * 2, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hb.data(), b, n * 2, cudaMemcpyDeviceToHost));
        double num = 0, den = 0, mx = 0; size_t neq = 0;
        for (size_t i = 0; i < n; ++i) {
          float x = bits2f(ha[i]), z = bits2f(hb[i]);
          num += (double)(x - z) * (x - z); den += (double)x * x; mx = std::max(mx, (double)fabsf(x - z)); neq += ha[i] != hb[i];
        }
        printf("  mx vs cuda-core %s: rel RMS %.3e · max|Δ| %.3e · bf16 mismatches %zu/%zu\n", name, sqrt(num / (den + 1e-30)), mx, neq, n);
      };
      cmp(y, y2, (size_t)G * I, "w13");
      cmp(eo, eo2, (size_t)G * K, "w2");
      float m13 = time_ms(st, [&] { k::mx_grouped_w13(gdd, G, A, As, I, K, rw, 10.f, y2, st); });
      float m2 = time_ms(st, [&] { k::mx_grouped_w2(gdd, G, yq, ys, K, I, eo2, st); });
      printf("%-16s %7s %6s | w13 %7.4f ms %5.0f GB/s | w2 %7.4f ms %5.0f GB/s | sum %.3f ms\n", "mx tensorcore x6", "", "",
             m13, 2.0 * G * (wsz + ssz) / m13 / 1e6, m2, 1.0 * G * (wsz + ssz) / m2 / 1e6, m13 + m2);
      // 8 rows per group (batched decode, short prefill)
      for (int g = 0; g < G; ++g) gd[g].row0 = g * 8, gd[g].n = 8;
      CUDA_CHECK(cudaMemcpy(gdd, gd.data(), sizeof(k::GroupDesc) * G, cudaMemcpyHostToDevice));
      uint8_t* A8 = dev_rand((size_t)G * 8 * K, 0, 0x7E); uint8_t* As8 = dev_rand((size_t)G * 8 * K / 32, 120, 130);
      float* rw8; CUDA_CHECK(cudaMalloc(&rw8, G * 8 * 4)); { std::vector<float> o(G * 8, 1.f); CUDA_CHECK(cudaMemcpy(rw8, o.data(), G * 8 * 4, cudaMemcpyHostToDevice)); }
      bf16 *y8a, *y8b; CUDA_CHECK(cudaMalloc(&y8a, (size_t)G * 8 * I * 2)); CUDA_CHECK(cudaMalloc(&y8b, (size_t)G * 8 * I * 2));
      k::gemv_grouped_w13(gdd, G, A8, As8, I, K, rw8, 10.f, y8a, st); k::mx_grouped_w13(gdd, G, A8, As8, I, K, rw8, 10.f, y8b, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      cmp(y8a, y8b, (size_t)G * 8 * I, "w13 rows=8");
      float c8 = time_ms(st, [&] { k::gemv_grouped_w13(gdd, G, A8, As8, I, K, rw8, 10.f, y8a, st); });
      float x8 = time_ms(st, [&] { k::mx_grouped_w13(gdd, G, A8, As8, I, K, rw8, 10.f, y8b, st); });
      printf("%-16s rows=8/group: cuda-core %.4f ms · mx %.4f ms\n", "w13 x6", c8, x8);
    }
  }
  // cuBLAS: router f32, hc mix f32, wo_a bf16 batch
  {
    float *A, *B, *C; CUDA_CHECK(cudaMalloc(&A, 8 * 20480 * 4)); CUDA_CHECK(cudaMalloc(&B, (size_t)384 * 20480 * 4)); CUDA_CHECK(cudaMalloc(&C, 8 * 384 * 4));
    CUDA_CHECK(cudaMemset(A, 0, 8 * 20480 * 4)); CUDA_CHECK(cudaMemset(B, 0, (size_t)384 * 20480 * 4));
    float tr = time_ms(st, [&] { blas.gemm_f32(A, B, C, 1, 384, 5120); });
    float th = time_ms(st, [&] { blas.gemm_f32(A, B, C, 1, 24, 20480); });
    printf("%-16s router f32 [384x5120] %.4f ms %4.0f GB/s · hc mix f32 [24x20480] %.4f ms %4.0f GB/s\n", "cublas f32", tr, 384.0 * 5120 * 4 / tr / 1e6, th,
           24.0 * 20480 * 4 / th / 1e6);
    const int H = 64, D = 512, G = 8, R = 1024, sub = H * D / G;
    bf16 *o, *woa, *og; CUDA_CHECK(cudaMalloc(&o, 8 * H * D * 2)); CUDA_CHECK(cudaMalloc(&woa, (size_t)G * R * sub * 2)); CUDA_CHECK(cudaMalloc(&og, 8 * G * R * 2));
    CUDA_CHECK(cudaMemset(o, 0, 8 * H * D * 2)); CUDA_CHECK(cudaMemset(woa, 0, (size_t)G * R * sub * 2));
    float tw = time_ms(st, [&] { blas.gemm_bf16_batched_ld(o, H * D, sub, woa, sub, (long)R * sub, og, G * R, R, 1, R, sub, G); });
    float tw8 = time_ms(st, [&] { blas.gemm_bf16_batched_ld(o, H * D, sub, woa, sub, (long)R * sub, og, G * R, R, 8, R, sub, G); });
    printf("%-16s wo_a bf16 8x[1024x4096] M=1 %.4f ms %4.0f GB/s · M=8 %.4f ms (%.1f MB)\n", "cublas bf16", tw, (double)G * R * sub * 2 / tw / 1e6, tw8,
           (double)G * R * sub * 2 / 1e6);
  }
  // Launch/execution floor of small kernels: rmsnorm, act_quant (M=1, K=5120)
  {
    bf16 *x, *w, *y; uint8_t *q, *s;
    CUDA_CHECK(cudaMalloc(&x, 8 * 5120 * 2)); CUDA_CHECK(cudaMalloc(&w, 5120 * 2)); CUDA_CHECK(cudaMalloc(&y, 8 * 5120 * 2));
    CUDA_CHECK(cudaMalloc(&q, 8 * 5120)); CUDA_CHECK(cudaMalloc(&s, 8 * 160));
    CUDA_CHECK(cudaMemset(x, 0, 8 * 5120 * 2)); CUDA_CHECK(cudaMemset(w, 0, 5120 * 2));
    float tn = time_ms(st, [&] { k::rmsnorm(x, w, 1e-6f, 1, 5120, y, st); }, 200);
    float tq = time_ms(st, [&] { k::act_quant_fp8(x, 1, 5120, q, s, st); }, 200);
    printf("%-16s rmsnorm(1x5120) %.4f ms · act_quant(1x5120) %.4f ms\n", "small kernels", tn, tq);
  }
  return 0;
}
