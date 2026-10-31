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
#include "hive/qwen/qwen_kernels.h"
#include <cuda_fp4.h>
#include <cuda_fp8.h>

using namespace hive;
using namespace hive::qwen;

static bool run(int E, int M);
static bool gn_check();
static bool a4_check(int E, int M, int S);
static bool a4_time();
int main() {
  bool ok = run(64, 51);   // ~8 pairs per expert: 32-row tiles
  ok = run(8, 102) && ok;  // ~128 pairs per expert: 128-row tiles
  ok = gn_check() && ok;   // (W4A16 v1 block shapes)
  ok = a4_check(64, 51, 32) && ok;   // W4A4: 32-row tiles, two sub-chunks
  ok = a4_check(8, 102, 102) && ok;  // W4A4: 128-row tiles
  fn_moe_prefill_set_a4v2(false);    // v1 kernels against the same host emulation
  ok = a4_check(64, 51, 32) && ok;
  ok = a4_check(8, 102, 102) && ok;
  fn_moe_prefill_set_a4v2(true);
  ok = a4_time() && ok;
  return ok ? 0 : 1;
}
// real shape (E 512, 1036 rows: ~20 pairs per expert, 32-row tiles): gate|up blocks of 128 I rows must equal 64 bit for bit; both timed
static bool gn_check() {
  fn_moe_prefill_set_v2(false);   // the first W4A16 kernels' block shapes against each other
  struct V2 { ~V2() { fn_moe_prefill_set_v2(true); } } v2_restore;
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

// ---- W4A4 -----------------------------------------------------------------------------------------------------------------------------
// Host emulation of the W4A4 path with the same conversions (cuda_fp8 / cuda_fp4 host functions): per-row activation quantization, exact
//   products of the e2m1 values and e4m3 scales summed in double (the GPU sums in fp32 — the only difference), the same bf16 rounding points.
static float e4m3_val(uint8_t v) { __nv_fp8_e4m3 t; t.__x = v; return float(t); }
static float e2m1_val(uint8_t nib) {
  static const float tbl[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (nib & 8 ? -1.f : 1.f) * tbl[nib & 7];
}
static float bfr(float v) { return __bfloat162float(__float2bfloat16(v)); }
// row of K floats → dequantized values (without the global) and the global
static void quant_row(const float* v, int K, std::vector<float>& dq, float& gx) {
  float a = 0.f;
  for (int i = 0; i < K; ++i) a = std::max(a, std::fabs(v[i]));
  gx = a > 0.f ? a / (6.f * 448.f) : 1.f;
  const float inv = 1.f / gx;
  dq.assign(K, 0.f);
  for (int b = 0; b < K / 16; ++b) {
    float ab = 0.f;
    for (int j = 0; j < 16; ++j) ab = std::max(ab, std::fabs(v[b * 16 + j]));
    const __nv_fp8_storage_t se = __nv_cvt_float_to_fp8(ab / 6.f * inv, __NV_SATFINITE, __NV_E4M3);
    const float sv = e4m3_val(se), f = sv > 0.f ? inv / sv : 0.f;
    for (int j = 0; j < 16; j += 2) {
      const uint8_t byte = (uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(v[b * 16 + j] * f, v[b * 16 + j + 1] * f), __NV_E2M1, cudaRoundNearest);
      dq[b * 16 + j] = e2m1_val(byte & 15) * sv; dq[b * 16 + j + 1] = e2m1_val(byte >> 4) * sv;
    }
  }
}
static void dq_weight_row(const uint8_t* w, const uint8_t* s, int K, std::vector<float>& o) {
  o.resize(K);
  for (int k = 0; k < K; ++k) o[k] = e2m1_val(k & 1 ? w[k / 2] >> 4 : w[k / 2] & 15) * e4m3_val(s[k / 16]);
}
static bool a4_check(int E, int M, int S) {
  const int H = 2560, I = 640, K = 10;
  const glm::ExpertRecLayout L = glm::ExpertRecLayout::make(H, I);
  std::mt19937 rng(11);
  std::vector<uint8_t> recs((size_t)E * L.total);
  for (int e = 0; e < E; ++e) {
    uint8_t* r = recs.data() + (size_t)e * L.total;
    for (size_t i = 0; i < L.g; ++i) r[i] = (uint8_t)rng();
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
  for (size_t i = 0; i < x.size(); ++i) x[i] = __float2bfloat16(N01(rng) * (i % 97 == 0 ? 8.f : 1.f));   // some outliers per row
  std::vector<int32_t> ids((size_t)M * K);
  std::vector<float> w((size_t)M * K);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) { ids[m * K + k] = (int)((k * E / K + rng() % std::max(1, E / K)) % E); w[m * K + k] = 0.05f + 0.01f * k; }
  if (M > 3) ids[3 * K + 2] = -1;   // an invalid id contributes nothing
  __nv_bfloat16* dx; int32_t* dids; float* dw; float* o; void* ws;
  CUDA_CHECK(cudaMalloc(&dx, x.size() * 2)); CUDA_CHECK(cudaMalloc(&dids, ids.size() * 4)); CUDA_CHECK(cudaMalloc(&dw, w.size() * 4));
  CUDA_CHECK(cudaMalloc(&o, (size_t)M * H * 4));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&ws, fn_moe_prefill_ws_bytes(S, K, E, H, I)));
  fn_moe_prefill_set_a4(true);
  fn_moe_prefill(L, dtab, 0, E, dids, dw, M, K, dx, H, I, o, ws, S, 0);
  fn_moe_prefill_set_a4(false);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got((size_t)M * H);
  CUDA_CHECK(cudaMemcpy(got.data(), o, got.size() * 4, cudaMemcpyDeviceToHost));
  // host reference
  std::vector<float> ref((size_t)M * H, 0.f), xr(H), dqx, wrow, yv(I), dqy, part(H);
  double dot = 0, na = 0, nb = 0, maxrel = 0;
  size_t bad = 0;
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < H; ++k) xr[k] = __bfloat162float(x[(size_t)m * H + k]);
    float gx; quant_row(xr.data(), H, dqx, gx);
    std::vector<float> acc(H, 0.f);
    for (int kk = 0; kk < K; ++kk) {
      const int e = ids[m * K + kk];
      if (e < 0) continue;
      const uint8_t* r = recs.data() + (size_t)e * L.total;
      float g1, g3, g2; std::memcpy(&g1, r + L.g, 4); std::memcpy(&g3, r + L.g + 4, 4); std::memcpy(&g2, r + L.g + 8, 4);
      for (int i = 0; i < I; ++i) {
        double sg = 0, su = 0;
        dq_weight_row(r + L.w1 + (size_t)i * H / 2, r + L.s1 + (size_t)i * H / 16, H, wrow);
        for (int k = 0; k < H; ++k) sg += (double)dqx[k] * wrow[k];
        dq_weight_row(r + L.w3 + (size_t)i * H / 2, r + L.s3 + (size_t)i * H / 16, H, wrow);
        for (int k = 0; k < H; ++k) su += (double)dqx[k] * wrow[k];
        const float gv = bfr((float)sg * gx * g1), uv = bfr((float)su * gx * g3);
        yv[i] = bfr(gv / (1.f + std::exp(-gv)) * uv);
      }
      float gy; quant_row(yv.data(), I, dqy, gy);
      for (int n = 0; n < H; ++n) {
        double sd = 0;
        dq_weight_row(r + L.w2 + (size_t)n * I / 2, r + L.s2 + (size_t)n * I / 16, I, wrow);
        for (int k = 0; k < I; ++k) sd += (double)dqy[k] * wrow[k];
        part[n] = bfr((float)sd * gy * g2);
      }
      for (int n = 0; n < H; ++n) acc[n] += part[n] * w[m * K + kk];
    }
    double rms = 0;
    for (int n = 0; n < H; ++n) rms += (double)acc[n] * acc[n];
    rms = std::sqrt(rms / H) + 1e-30;
    for (int n = 0; n < H; ++n) {
      const double u = acc[n], v = got[(size_t)m * H + n];
      dot += u * v; na += u * u; nb += v * v;
      const double rel = std::fabs(u - v) / rms;
      maxrel = std::max(maxrel, rel);
      if (rel > 1e-2 && bad < 6) printf("  row %d col %d: host %.6f gpu %.6f (row rms %.4f)\n", m, n, u, v, rms);
      bad += rel > 1e-2;
    }
  }
  const double cos = dot / std::sqrt(na * nb);
  const bool ok = cos > 0.99999 && bad <= (size_t)M;   // fp32 vs double sums can flip a bf16 rounding of g / u / y / part
  printf("%s W4A4 prefill vs host emulation (E %d, M %d, S %d): cos %.8f · max |d| / row rms %.2e · over 1e-2: %zu\n", ok ? "PASS" : "FAIL", E, M, S,
         cos, maxrel, bad);
  CUDA_CHECK(cudaFree(drec)); CUDA_CHECK(cudaFree(dtab)); CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dids)); CUDA_CHECK(cudaFree(dw));
  CUDA_CHECK(cudaFree(o)); CUDA_CHECK(cudaFree(ws));
  return ok;
}
// real shape timing (E 512, distinct records): W4A16 vs W4A4 at 1K / 8K rows, and the W4A4 output's cosine to the W4A16 one
static bool a4_time() {
  const int H = 2560, I = 640, K = 10, E = 512;
  const glm::ExpertRecLayout L = glm::ExpertRecLayout::make(H, I);
  std::mt19937 rng(5);
  std::vector<uint8_t> rec(L.total);
  uint8_t* drec = nullptr;
  CUDA_CHECK(cudaMalloc(&drec, (size_t)E * L.total));
  for (int e = 0; e < E; ++e) {
    for (size_t i = 0; i < L.g; ++i) rec[i] = (uint8_t)rng();
    for (size_t i : {L.s1, L.s3}) for (size_t j = 0; j < (size_t)I * H / 16; ++j) rec[i + j] = (uint8_t)(((3 + rng() % 6) << 3) | (rng() & 7));
    for (size_t j = 0; j < (size_t)H * I / 16; ++j) rec[L.s2 + j] = (uint8_t)(((3 + rng() % 6) << 3) | (rng() & 7));
    const float g[3] = {0.0021f, 0.0017f, 0.0013f};
    std::memcpy(rec.data() + L.g, g, sizeof g);
    CUDA_CHECK(cudaMemcpy(drec + (size_t)e * L.total, rec.data(), L.total, cudaMemcpyHostToDevice));
  }
  std::vector<const uint8_t*> tab(E);
  for (int e = 0; e < E; ++e) tab[e] = drec + (size_t)e * L.total;
  const uint8_t** dtab = nullptr;
  CUDA_CHECK(cudaMalloc(&dtab, E * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(dtab, tab.data(), E * sizeof(void*), cudaMemcpyHostToDevice));
  bool ok = true;
  for (int M : {1024, 8192}) {
    std::normal_distribution<float> N01(0.f, 1.f);
    std::vector<__nv_bfloat16> x((size_t)M * H);
    for (auto& v : x) v = __float2bfloat16(N01(rng));
    std::vector<int32_t> ids((size_t)M * K);
    std::vector<float> w((size_t)M * K);
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        int e;
        do { e = (int)(rng() % E); } while (std::find(ids.begin() + (size_t)m * K, ids.begin() + (size_t)m * K + k, e) != ids.begin() + (size_t)m * K + k);
        ids[(size_t)m * K + k] = e; w[(size_t)m * K + k] = 0.1f;
      }
    }
    __nv_bfloat16* dx; int32_t* dids; float* dw; float *o1, *o2; void* ws;
    CUDA_CHECK(cudaMalloc(&dx, x.size() * 2)); CUDA_CHECK(cudaMalloc(&dids, ids.size() * 4)); CUDA_CHECK(cudaMalloc(&dw, w.size() * 4));
    CUDA_CHECK(cudaMalloc(&o1, (size_t)M * H * 4)); CUDA_CHECK(cudaMalloc(&o2, (size_t)M * H * 4));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ws, fn_moe_prefill_ws_bytes(M, K, E, H, I)));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    float t[4] = {0, 0, 0, 0};
    float *o3 = nullptr, *o4 = nullptr;
    CUDA_CHECK(cudaMalloc(&o3, (size_t)M * H * 4)); CUDA_CHECK(cudaMalloc(&o4, (size_t)M * H * 4));
    for (int v = 0; v < 4; ++v) {   // W4A16 v2 · W4A4 v2 · W4A4 v1 · W4A16 v1
      fn_moe_prefill_set_a4(v == 1 || v == 2);
      fn_moe_prefill_set_a4v2(v == 1);
      fn_moe_prefill_set_v2(v != 3);
      for (int it = 0; it < 6; ++it) {
        CUDA_CHECK(cudaEventRecord(a));
        fn_moe_prefill(L, dtab, 0, E, dids, dw, M, K, dx, H, I, v == 0 ? o1 : v == 1 ? o2 : v == 2 ? o3 : o4, ws, M, 0);
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        if (it >= 2) t[v] += ms / 4;
      }
    }
    fn_moe_prefill_set_a4(false);
    fn_moe_prefill_set_a4v2(true);
    fn_moe_prefill_set_v2(true);
    {
      std::vector<float> e1((size_t)M * H), e4((size_t)M * H);
      CUDA_CHECK(cudaMemcpy(e1.data(), o1, e1.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(e4.data(), o4, e4.size() * 4, cudaMemcpyDeviceToHost));
      const bool same16 = std::memcmp(e1.data(), e4.data(), e1.size() * 4) == 0;
      printf("%s W4A16 v2 == v1 bit for bit (M %d) · v2 %.3f ms · v1 %.3f ms\n", same16 ? "PASS" : "FAIL", M, t[0], t[3]);
      ok = ok && same16;
    }
    CUDA_CHECK(cudaFree(o4));
    std::vector<float> h1((size_t)M * H), h2((size_t)M * H), h3((size_t)M * H);
    CUDA_CHECK(cudaMemcpy(h1.data(), o1, h1.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), o2, h2.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h3.data(), o3, h3.size() * 4, cudaMemcpyDeviceToHost));
    const bool same = std::memcmp(h2.data(), h3.data(), h2.size() * 4) == 0;
    printf("%s W4A4 v2 == v1 bit for bit (M %d) · v1 %.3f ms\n", same ? "PASS" : "FAIL", M, t[2]);
    ok = ok && same;
    {   // the reduce fused with the combine == fn_moe_prefill (fp32) + moe_combine, bit for bit
      const int ld = E + 1;
      std::vector<__nv_bfloat16> sh((size_t)M * H);
      std::vector<float> lg((size_t)M * ld);
      for (auto& v : sh) v = __float2bfloat16(N01(rng));
      for (auto& v : lg) v = N01(rng);
      __nv_bfloat16 *dsh, *y1, *y2; float* dlg;
      CUDA_CHECK(cudaMalloc(&dsh, sh.size() * 2)); CUDA_CHECK(cudaMalloc(&dlg, lg.size() * 4));
      CUDA_CHECK(cudaMalloc(&y1, sh.size() * 2)); CUDA_CHECK(cudaMalloc(&y2, sh.size() * 2));
      CUDA_CHECK(cudaMemcpy(dsh, sh.data(), sh.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dlg, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
      fn_moe_prefill(L, dtab, 0, E, dids, dw, M, K, dx, H, I, o1, ws, M, 0);
      moe_combine(o1, dsh, dlg, ld, E, M, H, y1, 0);
      const FnPfCombine cmb{dsh, dlg, ld, E, y2};
      fn_moe_prefill(L, dtab, 0, E, dids, dw, M, K, dx, H, I, nullptr, ws, M, 0, &cmb);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<__nv_bfloat16> a1(sh.size()), a2(sh.size());
      CUDA_CHECK(cudaMemcpy(a1.data(), y1, a1.size() * 2, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(a2.data(), y2, a2.size() * 2, cudaMemcpyDeviceToHost));
      const bool sc = std::memcmp(a1.data(), a2.data(), a1.size() * 2) == 0;
      printf("%s reduce fused with the combine == reduce + moe_combine (M %d)\n", sc ? "PASS" : "FAIL", M);
      ok = ok && sc;
      CUDA_CHECK(cudaFree(dsh)); CUDA_CHECK(cudaFree(dlg)); CUDA_CHECK(cudaFree(y1)); CUDA_CHECK(cudaFree(y2));
    }
    CUDA_CHECK(cudaFree(o3));
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < h1.size(); ++i) { dot += (double)h1[i] * h2[i]; na += (double)h1[i] * h1[i]; nb += (double)h2[i] * h2[i]; }
    const double cos = dot / std::sqrt(na * nb);
    const double tf = 2.0 * M * K * 3.0 * H * I / 1e12;
    printf("M %5d: W4A16 %.3f ms (%.0f TFLOP/s) · W4A4 %.3f ms (%.0f TFLOP/s) · cos(W4A4, W4A16) %.5f\n", M, t[0], tf / t[0] * 1e3, t[1],
           tf / t[1] * 1e3, cos);
    ok = ok && cos > 0.98;
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dids)); CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(o1)); CUDA_CHECK(cudaFree(o2));
    CUDA_CHECK(cudaFree(ws));
  }
  CUDA_CHECK(cudaFree(drec)); CUDA_CHECK(cudaFree(dtab));
  return ok;
}
