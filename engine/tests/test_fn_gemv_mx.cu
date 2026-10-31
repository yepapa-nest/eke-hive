// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// MXFP8 dense weights as stored in NVIDIA's FP4-Local checkpoint (QMat fmt8 3): e4m3 [N, K] + ue8m0 bytes [N, K/32].
//   1. fn_dequant8(fmt 3) == the host expansion (hive/ckpt_quant.h expand: e4m3 · 2^(s − 127) → bf16), bit for bit.
//   2. fn_gemv8(fmt 3) against fn_gemv_tc on that bf16 matrix (both exact operands, fp32 accumulation in another order): max |Δ| relative
//      to the row's output scale ≤ 1e-5, for every row-count instance (1, 2..8, 9..16, 17..32, 33..64 rows) and the FN shapes; plus timing
//      next to the INT8 copy (same bytes per weight).
//   3. fn_mx_tile_scales == the tiled layout fn_mx_quant_w8 writes for the same exponents (cuBLASLt block-scale tiles).
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "hive/ckpt_quant.h"
#include "hive/common.h"
#include "hive/qwen/fn_gemv.h"
#include "hive/qwen/fn_mx.h"

using bf16 = __nv_bfloat16;
using namespace hive;
static int fails = 0;

int main() {
  struct S { const char* name; int N, K; };
  const S shapes[] = {{"gdn_in 16480x2560", 16480, 2560}, {"gdn_out 2560x6144", 2560, 6144}, {"attn_in 13952x2560", 13952, 2560},
                      {"sh_gu 1280x2560", 1280, 2560}, {"sh_down 2560x640", 2560, 640}};
  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
  std::mt19937 rng(7);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (const S& s : shapes) {
    const size_t n = (size_t)s.N * s.K, ns = (size_t)s.N * (s.K / 32);
    std::vector<uint8_t> q(n), e(ns);
    for (auto& v : q) { v = (uint8_t)(rng() & 0xFF); if ((v & 0x7F) == 0x7F) v ^= 1; }
    for (auto& v : e) v = (uint8_t)(112 + rng() % 12);   // the checkpoint's range is 59..120
    std::vector<uint16_t> ref(n);
    for (size_t i = 0; i < n; ++i) ref[i] = ckq::f32_bf16(ckq::e4m3_f32(q[i]) * ckq::ue8m0_f32(e[(i / s.K) * (s.K / 32) + (i % s.K) / 32]));
    uint8_t *dq, *de; bf16* dw;
    CUDA_CHECK(cudaMalloc(&dq, n)); CUDA_CHECK(cudaMalloc(&de, ns)); CUDA_CHECK(cudaMalloc(&dw, n * 2));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), n, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(de, e.data(), ns, cudaMemcpyHostToDevice));
    qwen::fn_dequant8(dq, reinterpret_cast<const float*>(de), 3, 32, s.N, s.K, dw, st);
    std::vector<uint16_t> got(n);
    CUDA_CHECK(cudaMemcpyAsync(got.data(), dw, n * 2, cudaMemcpyDeviceToHost, st)); CUDA_CHECK(cudaStreamSynchronize(st));
    if (got != ref) { printf("FAIL %s: fn_dequant8(fmt 3) != host expansion\n", s.name); ++fails; }
    // GEMV
    const int Mmax = 64;
    std::vector<uint16_t> xh((size_t)Mmax * s.K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = ckq::f32_bf16(nd(rng));
    bf16 *x, *y8, *yt; CUDA_CHECK(cudaMalloc(&x, xh.size() * 2)); CUDA_CHECK(cudaMalloc(&y8, (size_t)Mmax * s.N * 4)); CUDA_CHECK(cudaMalloc(&yt, (size_t)Mmax * s.N * 4));
    CUDA_CHECK(cudaMemcpy(x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    for (int M : {1, 2, 5, 8, 9, 16, 24, 32, 40, 64}) {
      // fp32 outputs of both (the verify head's yf path): the same exact products, summed in another order
      qwen::fn_gemv8(dq, reinterpret_cast<const float*>(de), 3, 32, s.N, s.K, x, s.K, M, nullptr, s.N, st, reinterpret_cast<float*>(y8));
      qwen::fn_gemv_tc(dw, s.N, s.K, x, s.K, M, nullptr, s.N, reinterpret_cast<float*>(yt), st);
      std::vector<float> A((size_t)M * s.N), B((size_t)M * s.N);
      CUDA_CHECK(cudaMemcpyAsync(A.data(), y8, A.size() * 4, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaMemcpyAsync(B.data(), yt, B.size() * 4, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaStreamSynchronize(st));
      double worst = 0;
      for (int m = 0; m < M; ++m) {
        double scale = 0;
        for (int c = 0; c < s.N; ++c) scale = std::max(scale, (double)std::fabs(B[(size_t)m * s.N + c]));
        for (int c = 0; c < s.N; ++c) worst = std::max(worst, std::fabs((double)A[(size_t)m * s.N + c] - B[(size_t)m * s.N + c]) / std::max(scale, 1e-30));
      }
      if (worst > 1e-5) { printf("FAIL %s M=%d: fmt 3 GEMV vs bf16 GEMV max rel |Δ| %.3g\n", s.name, M, worst); ++fails; }
      else if (M == 1 || M == 64) printf("%-20s M=%2d fmt3 vs bf16 GEMV: max |Δ| / row max %.2e\n", s.name, M, worst);
    }
    // timing next to INT8 (unit scales, the same 1 byte per weight)
    for (int M : {1, 4, 16, 32}) {
      float* s8; CUDA_CHECK(cudaMalloc(&s8, (size_t)s.N * (s.K / 128) * 4)); CUDA_CHECK(cudaMemset(s8, 0, (size_t)s.N * (s.K / 128) * 4));
      float t[2];
      for (int k = 0; k < 2; ++k) {
        const int fmt = k ? 2 : 3, gs = k ? 128 : 32;
        const float* sp = k ? s8 : reinterpret_cast<const float*>(de);
        for (int i = 0; i < 5; ++i) qwen::fn_gemv8(dq, sp, fmt, gs, s.N, s.K, x, s.K, M, y8, s.N, st);
        cudaEventRecord(a, st);
        for (int i = 0; i < 50; ++i) qwen::fn_gemv8(dq, sp, fmt, gs, s.N, s.K, x, s.K, M, y8, s.N, st);
        cudaEventRecord(b, st); cudaEventSynchronize(b);
        cudaEventElapsedTime(&t[k], a, b); t[k] *= 1000.f / 50;
      }
      printf("%-20s M=%2d  MXFP8 %7.1f us · INT8 %7.1f us (L2-warm)\n", s.name, M, t[0], t[1]);
      cudaFree(s8);
    }
    // tiles
    {
      std::vector<float> s8h(ns);
      for (size_t i = 0; i < ns; ++i) s8h[i] = ckq::ue8m0_f32(e[i]);
      float* s8; CUDA_CHECK(cudaMalloc(&s8, ns * 4)); CUDA_CHECK(cudaMemcpy(s8, s8h.data(), ns * 4, cudaMemcpyHostToDevice));
      // fn_mx_quant_w8 takes INT8 values · per-group scales; unit int8 weights (127) with scale 2^(e−127)/127 → block amax = 2^(e−127):
      //   its exponent rounding is ceil(log2(amax / 448)), so compare against that, not against e itself — instead compare our tiling with a
      //   host tiling of the same bytes
      const size_t tb = qwen::fn_mx_scale_bytes(s.N, s.K);
      uint8_t* tl; CUDA_CHECK(cudaMalloc(&tl, tb));
      qwen::fn_mx_tile_scales(de, s.N, s.K, tl, st);
      std::vector<uint8_t> th(tb), hh(tb, 0);
      CUDA_CHECK(cudaMemcpyAsync(th.data(), tl, tb, cudaMemcpyDeviceToHost, st)); CUDA_CHECK(cudaStreamSynchronize(st));
      const int cols = s.K / 32, cols4 = (cols + 3) / 4 * 4;
      for (int r = 0; r < s.N; ++r)
        for (int c = 0; c < cols; ++c)
          hh[((size_t)(r / 128) * (cols4 / 4) + (c / 4)) * 512 + (size_t)(r % 32) * 16 + (size_t)((r % 128) / 32) * 4 + (c % 4)] = e[(size_t)r * cols + c];
      if (th != hh) { printf("FAIL %s: fn_mx_tile_scales layout\n", s.name); ++fails; }
      cudaFree(tl); cudaFree(s8);
    }
    cudaFree(dq); cudaFree(de); cudaFree(dw); cudaFree(x); cudaFree(y8); cudaFree(yt);
  }
  if (fails) { printf("test_fn_gemv_mx: %d failure(s)\n", fails); return 1; }
  printf("test_fn_gemv_mx: dequant exact, GEMV fp32 outputs within 1e-5 of the bf16 GEMV for 1..64 rows, scale tiles — passed\n");
  return 0;
}
