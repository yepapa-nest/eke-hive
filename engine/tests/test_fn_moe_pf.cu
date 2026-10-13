// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// fn_moe_prefill (grouped W4A16 GEMMs) against fn_moe_decode (the validated decode kernels) on synthetic NVFP4 experts: same rows, same
//   routing; the two differ only in fp32 summation order, so every output must agree to a few fp32 ulps of its row's scale.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/qwen/fn_moe.h"

using namespace hive;
using namespace hive::qwen;

static bool run(int E, int M);
static bool gn_check();
int main() {
  bool ok = run(64, 51);   // ~8 pairs per expert: 32-row tiles
  ok = run(8, 102) && ok;  // ~128 pairs per expert: 128-row tiles
  ok = gn_check() && ok;
  return ok ? 0 : 1;
}
// real shape (E 512, 1036 rows: ~20 pairs per expert, 32-row tiles): gate|up blocks of 128 I rows must equal 64 bit for bit; both timed
static bool gn_check() {
  const int H = 2560, I = 640, K = 10, E = 512, M = 1036, U = 32;   // U distinct records, repeated over the 512 slots (timing still reads 1.4 GB)
  const glm::ExpertRecLayout L = glm::ExpertRecLayout::make(H, I);
  std::mt19937 rng(9);
  std::vector<uint8_t> recs((size_t)U * L.total);
  for (int e = 0; e < U; ++e) {
    uint8_t* r = recs.data() + (size_t)e * L.total;
    for (size_t i = 0; i < L.g; ++i) r[i] = (uint8_t)rng();
    for (size_t i : {L.s1, L.s3}) for (size_t j = 0; j < (size_t)I * H / 16; ++j) r[i + j] = (uint8_t)(((3 + rng() % 6) << 3) | (rng() & 7));
    for (size_t j = 0; j < (size_t)H * I / 16; ++j) r[L.s2 + j] = (uint8_t)(((3 + rng() % 6) << 3) | (rng() & 7));
    const float g[3] = {0.0021f * (1 + e % 3), 0.0017f, 0.0013f * (1 + e % 2)};
    std::memcpy(r + L.g, g, sizeof g);
  }
  uint8_t* drec = nullptr;
  CUDA_CHECK(cudaMalloc(&drec, (size_t)E * L.total));
  for (int e = 0; e < E; ++e) CUDA_CHECK(cudaMemcpy(drec + (size_t)e * L.total, recs.data() + (size_t)(e % U) * L.total, L.total, cudaMemcpyHostToDevice));
  std::vector<const uint8_t*> tab(E);
  for (int e = 0; e < E; ++e) tab[e] = drec + (size_t)e * L.total;
  const uint8_t** dtab = nullptr;
  CUDA_CHECK(cudaMalloc(&dtab, E * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(dtab, tab.data(), E * sizeof(void*), cudaMemcpyHostToDevice));
  std::normal_distribution<float> N01(0.f, 1.f);
  std::vector<__nv_bfloat16> x((size_t)M * H);
  for (auto& v : x) v = __float2bfloat16(N01(rng));
  std::vector<int32_t> ids((size_t)M * K);
  std::vector<float> w((size_t)M * K);
  for (int m = 0; m < M; ++m) {
    std::vector<int> pe(E);
    for (int e = 0; e < E; ++e) pe[e] = e;
    std::shuffle(pe.begin(), pe.end(), rng);
    for (int k = 0; k < K; ++k) { ids[m * K + k] = pe[k]; w[m * K + k] = 0.05f + 0.01f * k; }
  }
  __nv_bfloat16* dx; int32_t* dids; float* dw; float *o1, *o2; void* ws;
  CUDA_CHECK(cudaMalloc(&dx, x.size() * 2)); CUDA_CHECK(cudaMalloc(&dids, ids.size() * 4)); CUDA_CHECK(cudaMalloc(&dw, w.size() * 4));
  CUDA_CHECK(cudaMalloc(&o1, (size_t)M * H * 4)); CUDA_CHECK(cudaMalloc(&o2, (size_t)M * H * 4));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&ws, fn_moe_prefill_ws_bytes(8192, K, E, H, I)));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  float t[4] = {0, 0, 0, 0};
  bool ok = true;
  std::vector<float> h1((size_t)M * H), h2((size_t)M * H);
  for (int v = 0; v < 4; ++v) {   // (gate|up width, down rows): (64,128) reference · (128,128) · (64,256) · (128,256)
    fn_moe_prefill_set_gn(v & 1 ? 128 : 64, v & 2 ? 256 : 128);
    float* o = v ? o2 : o1;
    for (int it = 0; it < 6; ++it) {
      CUDA_CHECK(cudaMemset(o, 0, (size_t)M * H * 4));
      CUDA_CHECK(cudaEventRecord(a));
      fn_moe_prefill(L, dtab, 0, E, dids, dw, M, K, dx, H, I, o, ws, 8192, 0);
      CUDA_CHECK(cudaEventRecord(b));
      CUDA_CHECK(cudaEventSynchronize(b));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      if (it >= 2) t[v] += ms / 4;
    }
    if (v == 0) CUDA_CHECK(cudaMemcpy(h1.data(), o1, h1.size() * 4, cudaMemcpyDeviceToHost));
    else {
      CUDA_CHECK(cudaMemcpy(h2.data(), o2, h2.size() * 4, cudaMemcpyDeviceToHost));
      const bool same = std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0;
      ok = ok && same;
      printf("%s blocks (gate|up %d, down %d) == (64, 128) bit for bit · layer %.3f ms vs %.3f ms\n", same ? "PASS" : "FAIL", v & 1 ? 128 : 64,
             v & 2 ? 256 : 128, t[v], t[0]);
    }
  }
  fn_moe_prefill_set_gn(64, 128);
  return ok;
}
static bool run(int E, int M) {
  const int H = 2560, I = 640, K = 10;   // M · K ≤ 1024: the decode kernels take every pair at once
  const glm::ExpertRecLayout L = glm::ExpertRecLayout::make(H, I);
  std::mt19937 rng(7);
  std::vector<uint8_t> recs((size_t)E * L.total);
  for (int e = 0; e < E; ++e) {
    uint8_t* r = recs.data() + (size_t)e * L.total;
    for (size_t i = 0; i < L.g; ++i) r[i] = (uint8_t)rng();
    // e4m3 scales in [2^-4, 2^2): exponent field 3..8, any mantissa, sign 0
    for (size_t i : {L.s1, L.s3}) for (size_t j = 0; j < (size_t)I * H / 16; ++j) r[i + j] = (uint8_t)(((3 + rng() % 6) << 3) | (rng() & 7));
    for (size_t j = 0; j < (size_t)H * I / 16; ++j) r[L.s2 + j] = (uint8_t)(((3 + rng() % 6) << 3) | (rng() & 7));
    const float g[3] = {0.0021f * (1 + e % 3), 0.0017f, 0.0013f * (1 + e % 2)};
    std::memcpy(r + L.g, g, sizeof g);
  }
  uint8_t* drec = nullptr;
  CUDA_CHECK(cudaMalloc(&drec, recs.size()));
  CUDA_CHECK(cudaMemcpy(drec, recs.data(), recs.size(), cudaMemcpyHostToDevice));
  std::vector<const uint8_t*> tab(E);
  for (int e = 0; e < E; ++e) tab[e] = drec + (size_t)e * L.total;
  const uint8_t** dtab = nullptr;
  CUDA_CHECK(cudaMalloc(&dtab, E * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(dtab, tab.data(), E * sizeof(void*), cudaMemcpyHostToDevice));
  std::normal_distribution<float> N01(0.f, 1.f);
  std::vector<__nv_bfloat16> x((size_t)M * H);
  for (auto& v : x) v = __float2bfloat16(N01(rng));
  std::vector<int32_t> ids((size_t)M * K);
  std::vector<float> w((size_t)M * K);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) { ids[m * K + k] = (int)((k * E / K + rng() % std::max(1, E / K)) % E); w[m * K + k] = 0.05f + 0.01f * k; }   // repeats: shared tiles
  std::vector<glm::MoePair> pairs((size_t)M * K);
  for (int p = 0; p < M * K; ++p) pairs[p] = glm::MoePair{p / K, tab[ids[p]], w[p]};
  __nv_bfloat16* dx; int32_t* dids; float* dw; glm::MoePair* dp; float *o1, *o2;
  CUDA_CHECK(cudaMalloc(&dx, x.size() * 2)); CUDA_CHECK(cudaMalloc(&dids, ids.size() * 4)); CUDA_CHECK(cudaMalloc(&dw, w.size() * 4));
  CUDA_CHECK(cudaMalloc(&dp, pairs.size() * sizeof(glm::MoePair)));
  CUDA_CHECK(cudaMalloc(&o1, (size_t)M * H * 4)); CUDA_CHECK(cudaMalloc(&o2, (size_t)M * H * 4));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dp, pairs.data(), pairs.size() * sizeof(glm::MoePair), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(o1, 0, (size_t)M * H * 4));
  void *ws1, *ws2;
  CUDA_CHECK(cudaMalloc(&ws1, fn_moe_decode_ws_bytes(M * K, H, I)));
  const int S = E >= 64 ? 32 : M;   // E 64: two sub-chunks; E 8: one, 1,020 pairs (128-row tiles)
  CUDA_CHECK(cudaMalloc(&ws2, fn_moe_prefill_ws_bytes(S, K, E, H, I)));
  fn_moe_decode(L, dp, M * K, dx, H, I, o1, ws1, 0);
  fn_moe_prefill(L, dtab, 0, E, dids, dw, M, K, dx, H, I, o2, ws2, S, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> a((size_t)M * H), b((size_t)M * H);
  CUDA_CHECK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
  double dot = 0, na = 0, nb = 0, maxrel = 0;
  size_t same = 0, bad = 0;
  for (int m = 0; m < M; ++m) {
    double rms = 0;
    for (int n = 0; n < H; ++n) rms += (double)a[(size_t)m * H + n] * a[(size_t)m * H + n];
    rms = std::sqrt(rms / H) + 1e-30;
    for (int n = 0; n < H; ++n) {
      const double u = a[(size_t)m * H + n], v = b[(size_t)m * H + n];
      dot += u * v; na += u * u; nb += v * v;
      same += u == v;
      const double rel = std::fabs(u - v) / rms;
      maxrel = std::max(maxrel, rel);
      if (rel > 1e-2 && bad < 6) printf("  row %d col %d: decode %.6f prefill %.6f (row rms %.4f)\n", m, n, u, v, rms);
      bad += rel > 1e-2;   // a bf16 rounding flip of a part (2^-8 of one of 10 terms) stays well below this
    }
  }
  const double cos = dot / std::sqrt(na * nb);
  printf("fn_moe_prefill vs fn_moe_decode: %d rows · cos %.8f · identical %.2f %% · max |d| / row rms %.2e · over 1e-2: %zu\n", M, cos,
         100.0 * same / a.size(), maxrel, bad);
  const bool ok = cos > 0.99999 && bad <= (size_t)M;   // a bf16 flip of an intermediate y can move a few outputs by ~1 % of their row rms
  printf("%s (E %d, M %d)\n", ok ? "PASS" : "FAIL", E, M);
  return ok;
}
