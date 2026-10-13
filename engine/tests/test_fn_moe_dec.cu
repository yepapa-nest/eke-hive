// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// fn_moe_decode_routed v2 (expert-id grouping, vector activation loads, weight prefetch) against the first kernels (fn_moe_decode_set_ref):
//   same arithmetic in the same order → every output bit must match. Also times both on 4 layers × 512 synthetic NVFP4 experts with distinct
//   random routing (each step reads ~M·10 records from HBM, like a served batch) and reports the record bandwidth.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/qwen/fn_moe.h"

using namespace hive;
using namespace hive::qwen;
using bf16 = __nv_bfloat16;

int main() {
  const int H = 2560, I = 640, K = 10, E = 512, NL = 4;
  const glm::ExpertRecLayout L = glm::ExpertRecLayout::make(H, I);
  std::mt19937 rng(11);
  std::vector<uint8_t> rec(L.total);
  uint8_t* drec = nullptr;
  CUDA_CHECK(cudaMalloc(&drec, (size_t)NL * E * L.total));
  for (int e = 0; e < NL * E; ++e) {
    for (size_t i = 0; i < L.g; ++i) rec[i] = (uint8_t)rng();
    for (size_t i : {L.s1, L.s3}) for (size_t j = 0; j < (size_t)I * H / 16; ++j) rec[i + j] = (uint8_t)(((3 + rng() % 6) << 3) | (rng() & 7));
    for (size_t j = 0; j < (size_t)H * I / 16; ++j) rec[L.s2 + j] = (uint8_t)(((3 + rng() % 6) << 3) | (rng() & 7));
    const float g[3] = {0.0021f * (1 + e % 3), 0.0017f, 0.0013f * (1 + e % 2)};
    std::memcpy(rec.data() + L.g, g, sizeof g);
    CUDA_CHECK(cudaMemcpy(drec + (size_t)e * L.total, rec.data(), L.total, cudaMemcpyHostToDevice));
  }
  std::vector<const uint8_t*> tab((size_t)NL * E);
  for (int e = 0; e < NL * E; ++e) tab[e] = drec + (size_t)e * L.total;
  tab[(size_t)1 * E + 7] = nullptr;   // one non-resident expert (the weight-0 fallback path)
  const uint8_t** dtab = nullptr;
  CUDA_CHECK(cudaMalloc(&dtab, tab.size() * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(dtab, tab.data(), tab.size() * sizeof(void*), cudaMemcpyHostToDevice));
  const int MX = 64;
  bf16 *x, *sh, *o1, *o2;
  float* lg;
  int32_t* ids;
  float* w;
  void* ws;
  CUDA_CHECK(cudaMalloc(&x, (size_t)MX * H * 2)); CUDA_CHECK(cudaMalloc(&sh, (size_t)MX * H * 2));
  CUDA_CHECK(cudaMalloc(&o1, (size_t)MX * H * 2)); CUDA_CHECK(cudaMalloc(&o2, (size_t)MX * H * 2));
  CUDA_CHECK(cudaMalloc(&lg, (size_t)MX * 4)); CUDA_CHECK(cudaMalloc(&ids, (size_t)MX * K * 4)); CUDA_CHECK(cudaMalloc(&w, (size_t)MX * K * 4));
  CUDA_CHECK(cudaMalloc(&ws, fn_moe_decode_ws_bytes(MX * K, H, I)));
  {
    std::vector<bf16> hx((size_t)MX * H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = __float2bfloat16(nd(rng));
    CUDA_CHECK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    for (auto& v : hx) v = __float2bfloat16(0.1f * nd(rng));
    CUDA_CHECK(cudaMemcpy(sh, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> hl(MX);
    for (auto& v : hl) v = nd(rng);
    CUDA_CHECK(cudaMemcpy(lg, hl.data(), MX * 4, cudaMemcpyHostToDevice));
  }
  auto route = [&](int M, bool dup) {   // dup: rows in pairs share their experts (MTP rows of one context often do)
    std::vector<int32_t> hi((size_t)M * K);
    std::vector<float> hw((size_t)M * K);
    for (int m = 0; m < M; ++m) {
      std::vector<int> perm(E);
      for (int e = 0; e < E; ++e) perm[e] = e;
      std::shuffle(perm.begin(), perm.end(), rng);
      for (int k = 0; k < K; ++k) {
        hi[(size_t)m * K + k] = (dup && m % 2) ? hi[(size_t)(m - 1) * K + k] : perm[k];
        hw[(size_t)m * K + k] = 0.05f + 0.1f * (rng() % 8);
      }
    }
    if (M > 3) { hi[3 * K + 2] = 7; hi[2 * K + 1] = -1; }   // non-resident (layer 1) / invalid id
    CUDA_CHECK(cudaMemcpy(ids, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  };
  bool ok = true;
  std::vector<uint16_t> h1((size_t)MX * H), h2((size_t)MX * H);
  for (int M : {1, 5, 20, 30, 64})
    for (bool dup : {false, true})
      for (int li = 0; li < NL; ++li) {
        route(M, dup);
        fn_moe_decode_set_ref(true);
        fn_moe_decode_routed(L, ids, w, M, K, dtab, li, E, x, H, I, sh, lg, 1, 0, o1, ws, 0);
        fn_moe_decode_set_ref(false);
        fn_moe_decode_routed(L, ids, w, M, K, dtab, li, E, x, H, I, sh, lg, 1, 0, o2, ws, 0);
        CUDA_CHECK(cudaMemcpy(h1.data(), o1, (size_t)M * H * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h2.data(), o2, (size_t)M * H * 2, cudaMemcpyDeviceToHost));
        const long bad = (long)M * H - std::inner_product(h1.begin(), h1.begin() + (size_t)M * H, h2.begin(), 0L, std::plus<long>(),
                                                          [](uint16_t a, uint16_t b) { return (long)(a == b); });
        if (bad) { printf("M %d dup %d layer %d: %ld outputs differ\n", M, dup, li, bad); ok = false; }
      }
  printf("bit-exact v2 vs ref: %s\n", ok ? "yes" : "NO");
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  for (int M : {5, 20, 30, 40}) {
    for (int ref = 1; ref >= 0; --ref) {
      fn_moe_decode_set_ref(ref);
      double tot = 0, bytes = 0;
      for (int it = 0; it < 24; ++it) {
        route(M, false);
        const int li = it % NL;
        CUDA_CHECK(cudaEventRecord(a));
        fn_moe_decode_routed(L, ids, w, M, K, dtab, li, E, x, H, I, sh, lg, 1, 0, o1, ws, 0);
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        if (it >= 4) { tot += ms; bytes += 0; }
      }
      // distinct records ≈ E·(1 − (1 − K/E)^M)
      double dist = E * (1 - std::pow(1 - (double)K / E, M));
      const double us = tot / 20 * 1000;
      printf("M %2d %s: %.1f us/layer · ~%.0f records · %.0f GB/s\n", M, ref ? "ref" : "v2 ", us, dist, dist * (L.g + 12) / (us * 1e-6) / 1e9);
    }
  }
  return ok ? 0 : 1;
}
