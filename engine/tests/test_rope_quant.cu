// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// rope_last_quant (inverse RoPE on the last rd dims + act_quant(x, 32, ue8m0) round trip in one pass) vs the two separate launches
//   rope_last -> act_quant_fp8_roundtrip, and with rd 0 (quantize only) vs act_quant_fp8_roundtrip: outputs must be bit-identical. Also times both (the prefill wo_a input, M rows of H*D).
//   Run: scripts/hive-run.sh "./build-svc/test_rope_quant"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"

using namespace hive;
static uint16_t f2bits(float f) { bf16 b = f2bf(f); uint16_t u; memcpy(&u, &b, 2); return u; }

int main() {
  const int H = 64, D = 512, rd = 64, half = rd / 2, maxpos = 70000;
  bool all_ok = true;
  std::vector<float2> fr((size_t)maxpos * half);
  for (int p = 0; p < maxpos; ++p)
    for (int i = 0; i < half; ++i) {
      const double th = p * std::pow(10000.0, -2.0 * i / rd);
      fr[(size_t)p * half + i] = make_float2((float)std::cos(th), (float)std::sin(th));
    }
  float2* dfr; CUDA_CHECK(cudaMalloc(&dfr, fr.size() * sizeof(float2)));
  CUDA_CHECK(cudaMemcpy(dfr, fr.data(), fr.size() * sizeof(float2), cudaMemcpyHostToDevice));
  for (const int RD : {rd, 0})
  for (const int M : {1, 7, 129, 2048, 16384}) {
    std::mt19937 rng(11 + M);
    std::normal_distribution<float> nd;
    std::vector<uint16_t> x((size_t)M * H * D);
    for (size_t i = 0; i < x.size(); ++i) x[i] = f2bits(nd(rng) * (i % 97 == 0 ? 3.f : 0.05f));  // a few outliers per block set the scale
    std::vector<int32_t> pos(M);
    for (int m = 0; m < M; ++m) pos[m] = (int)(rng() % maxpos);
    bf16 *a, *b; int32_t* dpos;
    CUDA_CHECK(cudaMalloc(&a, x.size() * 2)); CUDA_CHECK(cudaMalloc(&b, x.size() * 2)); CUDA_CHECK(cudaMalloc(&dpos, M * 4));
    CUDA_CHECK(cudaMemcpy(dpos, pos.data(), M * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1, e2; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    float ms_two = 0, ms_one = 0;
    for (int rep = 0; rep < 6; ++rep) {
      CUDA_CHECK(cudaMemcpy(a, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(b, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      cudaEventRecord(e0);
      if (RD) k::rope_last(a, M, H, D, RD, dfr, dpos, true, 0);
      k::act_quant_fp8_roundtrip(a, M, H * D, 0);
      cudaEventRecord(e1);
      k::rope_last_quant(b, M, H, D, RD, RD ? dfr : nullptr, RD ? dpos : nullptr, RD != 0, 0);
      cudaEventRecord(e2);
      CUDA_CHECK(cudaEventSynchronize(e2));
      float t1, t2; cudaEventElapsedTime(&t1, e0, e1); cudaEventElapsedTime(&t2, e1, e2);
      if (rep) { ms_two += t1; ms_one += t2; }
    }
    std::vector<uint16_t> ra(x.size()), rb(x.size());
    CUDA_CHECK(cudaMemcpy(ra.data(), a, x.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rb.data(), b, x.size() * 2, cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < x.size(); ++i) diff += ra[i] != rb[i];
    if (diff && M == 1) {
      int shown = 0;
      for (size_t i = 0; i < x.size() && shown < 8; ++i)
        if (ra[i] != rb[i]) { printf("  i=%zu (head %zu dim %zu) in %04x two %04x fused %04x\n", i, i / D, i % D, x[i], ra[i], rb[i]); ++shown; }
    }
    const bool ok = diff == 0;
    all_ok &= ok;
    printf("rd=%-2d M=%-6d differing elements %zu / %zu  %s · two launches %.3f ms · fused %.3f ms\n", RD, M, diff, x.size(), ok ? "PASS" : "FAIL", ms_two / 5, ms_one / 5);
    cudaFree(a); cudaFree(b); cudaFree(dpos);
  }
  cudaFree(dfr);
  printf("%s\n", all_ok ? "ALL PASS (bit-identical)" : "FAIL");
  return all_ok ? 0 : 1;
}
