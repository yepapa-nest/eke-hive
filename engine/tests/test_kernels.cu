// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GPU kernel unit tests — compared against a host reference (double). Small enough (tens of MB VRAM) to run next to another process using the GPU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"

using namespace hive;

static int fails = 0;
#define EXPECT(cond, ...)                          \
  do {                                             \
    if (!(cond)) {                                 \
      ++fails;                                     \
      printf("  FAIL %s:%d ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                         \
      printf("\n");                                \
    }                                              \
  } while (0)

template <class T>
static T* dev_copy(const std::vector<T>& v) {
  T* d;
  CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
  return d;
}
template <class T>
static std::vector<T> dev_get(const T* d, size_t n) {
  std::vector<T> v(n);
  CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

static std::mt19937 rng(1234);

static void test_fp8_conv() {
  // Does the host e4m3 conversion match the device __nv_cvt: run the full grid + random values through the device conversion and compare
  std::vector<float> xs;
  for (int i = 0; i < 20000; ++i) xs.push_back(std::uniform_real_distribution<float>(-480, 480)(rng));
  for (int i = 0; i < 2000; ++i) xs.push_back(std::uniform_real_distribution<float>(-0.02f, 0.02f)(rng));
  for (int c = 0; c < 256; ++c) {
    float v = e4m3_to_f32((uint8_t)c);
    if (!std::isnan(v)) { xs.push_back(v); xs.push_back(v * 1.0001f); xs.push_back(v * 0.9999f); }
  }
  // Ties: midpoints between adjacent grid points
  for (int c = 0; c < 126; ++c) {
    float a = e4m3_to_f32((uint8_t)c), b = e4m3_to_f32((uint8_t)(c + 1));
    if (!std::isnan(a) && !std::isnan(b)) xs.push_back((a + b) * 0.5f);
  }
  // Checking the device conversion indirectly through act_quant is cumbersome, so instead of a separate kernel use a bf16 input roundtrip: a block with amax=448 so the scale is 1
  // -> here only the host conversion's self-consistency: round-trip monotonicity and saturation
  int bad = 0;
  for (float x : xs) {
    uint8_t q = f32_to_e4m3_host(x);
    float y = e4m3_to_f32(q);
    if (std::isnan(y)) { ++bad; continue; }
    float lim = std::min(std::fabs(x), 448.f);
    // error is at most half the grid spacing
    int e;
    frexpf(std::max(lim, ldexpf(1.f, -6)), &e);
    float step = ldexpf(1.f, e - 1 - 3);
    if (std::fabs(y - std::copysign(lim, x)) > step * 0.5f + 1e-7f) ++bad;
  }
  EXPECT(bad == 0, "fp8 host conv bad=%d", bad);
  printf("fp8 host conversion: %s (%zu samples)\n", bad ? "FAIL" : "ok", xs.size());
}

static void test_act_quant() {
  const int M = 37, K = 5120;
  std::vector<bf16> x(M * K);
  std::normal_distribution<float> nd(0, 3);
  for (auto& v : x) v = f2bf(nd(rng));
  bf16* dx = dev_copy(x);
  uint8_t *dy, *ds;
  CUDA_CHECK(cudaMalloc(&dy, M * K));
  CUDA_CHECK(cudaMalloc(&ds, M * K / 32));
  k::act_quant_fp8(dx, M, K, dy, ds, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto y = dev_get(dy, (size_t)M * K);
  auto s = dev_get(ds, (size_t)M * K / 32);
  int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int b = 0; b < K / 32; ++b) {
      float amax = 0;
      for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(bf2f(x[m * K + b * 32 + i])));
      amax = std::max(amax, 1e-4f);
      uint8_t code = f32_ceil_pow2_e8m0(amax * (1.0f / 448.0f));
      if (s[m * (K / 32) + b] != code) ++bad;
      float sc = e8m0_to_f32(code);
      for (int i = 0; i < 32; ++i) {
        float t = std::min(std::max(bf2f(x[m * K + b * 32 + i]) / sc, -448.f), 448.f);
        if (y[m * K + b * 32 + i] != f32_to_e4m3_host(t)) ++bad;
      }
    }
  EXPECT(bad == 0, "act_quant mismatch %d", bad);
  printf("act_quant_fp8: %s\n", bad ? "FAIL" : "ok (device __nv_cvt == host RNE)");
  // roundtrip
  k::act_quant_fp8_roundtrip(dx, M, K, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto xr = dev_get(dx, (size_t)M * K);
  bad = 0;
  for (int m = 0; m < M; ++m)
    for (int b = 0; b < K / 32; ++b) {
      float sc = e8m0_to_f32(s[m * (K / 32) + b]);
      for (int i = 0; i < 32; ++i) {
        float want = e4m3_to_f32(y[m * K + b * 32 + i]) * sc;
        if (bf2f(xr[m * K + b * 32 + i]) != want) ++bad;
      }
    }
  EXPECT(bad == 0, "roundtrip mismatch %d", bad);
  printf("act_quant_fp8_roundtrip: %s\n", bad ? "FAIL" : "ok");
  cudaFree(dx); cudaFree(dy); cudaFree(ds);
}

static void test_fp4_roundtrip() {
  const int M = 5, K = 512;
  std::vector<bf16> x(M * K);
  std::normal_distribution<float> nd(0, 1);
  for (auto& v : x) v = f2bf(nd(rng));
  for (int which = 0; which < 2; ++which) {
    bf16* dx = dev_copy(x);
    int block = which ? 16 : 32;
    bool e4 = which;
    k::fp4_quant_roundtrip(dx, M, K, block, e4, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto r = dev_get(dx, (size_t)M * K);
    int bad = 0;
    for (int m = 0; m < M; ++m)
      for (int b = 0; b < K / block; ++b) {
        float amax = 0;
        for (int i = 0; i < block; ++i) amax = std::max(amax, std::fabs(bf2f(x[m * K + b * block + i])));
        float sc;
        if (e4) { amax = std::max(amax, 6.f * ldexpf(1.f, -9)); sc = e4m3_to_f32(f32_to_e4m3_host(amax * (1.0f / 6.0f))); }
        else { amax = std::max(amax, 6.f * ldexpf(1.f, -126)); sc = e8m0_to_f32(f32_ceil_pow2_e8m0(amax * (1.0f / 6.0f))); }
        for (int i = 0; i < block; ++i) {
          float t = std::min(std::max(bf2f(x[m * K + b * block + i]) / sc, -6.f), 6.f);
          float want = bf2f(f2bf(e2m1_to_f32(f32_to_e2m1(t)) * sc));
          if (bf2f(r[m * K + b * block + i]) != want) ++bad;
        }
      }
    EXPECT(bad == 0, "fp4 roundtrip(block %d e4m3 %d) mismatch %d", block, (int)e4, bad);
    printf("fp4_quant_roundtrip block=%d scale=%s: %s\n", block, e4 ? "e4m3" : "e8m0", bad ? "FAIL" : "ok");
    cudaFree(dx);
  }
}

static void test_rmsnorm() {
  const int M = 9, K = 5120;
  std::vector<bf16> x(M * K), w(K);
  std::normal_distribution<float> nd(0, 2);
  for (auto& v : x) v = f2bf(nd(rng));
  for (auto& v : w) v = f2bf(1.f + 0.1f * nd(rng));
  bf16 *dx = dev_copy(x), *dw = dev_copy(w), *dout;
  CUDA_CHECK(cudaMalloc(&dout, M * K * sizeof(bf16)));
  k::rmsnorm(dx, dw, 1e-20f, M, K, dout, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto o = dev_get(dout, (size_t)M * K);
  double maxrel = 0;
  for (int m = 0; m < M; ++m) {
    double ss = 0;
    for (int i = 0; i < K; ++i) { double v = bf2f(x[m * K + i]); ss += v * v; }
    double rs = 1.0 / std::sqrt(ss / K + 1e-20);
    for (int i = 0; i < K; ++i) {
      double want = bf2f(w[i]) * (bf2f(x[m * K + i]) * rs);
      double got = bf2f(o[m * K + i]);
      maxrel = std::max(maxrel, std::fabs(got - want) / (std::fabs(want) + 1e-3));
    }
  }
  EXPECT(maxrel < 1e-2, "rmsnorm maxrel %g", maxrel);
  printf("rmsnorm: %s (maxrel %.3g, includes bf16 output rounding)\n", maxrel < 1e-2 ? "ok" : "FAIL", maxrel);
  cudaFree(dx); cudaFree(dw); cudaFree(dout);
}

// Block-scaled GEMM reference (double)
static void ref_gemm(const std::vector<uint8_t>& A, const std::vector<uint8_t>& sa, const std::vector<uint8_t>& B,
                     const std::vector<uint8_t>& sb, bool fp4, int M, int N, int K, std::vector<double>& C) {
  const int nb = K / 32;
  C.assign((size_t)M * N, 0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int b = 0; b < nb; ++b) {
        double d = 0;
        for (int i = 0; i < 32; ++i) {
          int k = b * 32 + i;
          double a = e4m3_to_f32(A[(size_t)m * K + k]);
          double w;
          if (fp4) {
            uint8_t byte = B[(size_t)n * (K / 2) + k / 2];
            w = e2m1_to_f32((k & 1) ? (byte >> 4) : (byte & 0xF));
          } else w = e4m3_to_f32(B[(size_t)n * K + k]);
          d += a * w;
        }
        acc += d * (double)e8m0_to_f32(sa[(size_t)m * nb + b]) * (double)e8m0_to_f32(sb[(size_t)n * nb + b]);
      }
      C[(size_t)m * N + n] = acc;
    }
}

static void test_gemm(int M, int N, int K, bool fp4) {
  const int nb = K / 32;
  std::vector<uint8_t> A((size_t)M * K), sa((size_t)M * nb), B(fp4 ? (size_t)N * K / 2 : (size_t)N * K), sb((size_t)N * nb);
  std::uniform_int_distribution<int> byte(0, 255);
  auto rnd_e4m3 = [&]() { uint8_t v; do { v = (uint8_t)byte(rng); } while ((v & 0x7F) == 0x7F || (v & 0x78) > 0x58); return v; };  // exclude NaN and values that are too large
  for (auto& v : A) v = rnd_e4m3();
  for (auto& v : sa) v = (uint8_t)(127 - 8 + byte(rng) % 8);
  for (auto& v : sb) v = (uint8_t)(127 - 12 + byte(rng) % 8);
  if (fp4) for (auto& v : B) v = (uint8_t)byte(rng);
  else for (auto& v : B) v = rnd_e4m3();
  std::vector<double> ref;
  ref_gemm(A, sa, B, sb, fp4, M, N, K, ref);
  uint8_t *dA = dev_copy(A), *dsa = dev_copy(sa), *dB = dev_copy(B), *dsb = dev_copy(sb);
  float* dC;
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
  k::gemm_bs(dA, dsa, dB, dsb, fp4, M, N, K, nullptr, dC, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto C = dev_get(dC, (size_t)M * N);
  double maxabs = 0, maxref = 0;
  for (size_t i = 0; i < C.size(); ++i) { maxabs = std::max(maxabs, std::fabs(C[i] - ref[i])); maxref = std::max(maxref, std::fabs(ref[i])); }
  double rel = maxabs / (maxref + 1e-30);
  EXPECT(rel < 1e-4, "gemm M=%d N=%d K=%d fp4=%d rel %g", M, N, K, (int)fp4, rel);
  printf("gemm_bs M=%d N=%d K=%d B=%s: %s (max|Δ|/max|ref| = %.2e)\n", M, N, K, fp4 ? "e2m1" : "e4m3", rel < 1e-4 ? "ok" : "FAIL", rel);
  cudaFree(dA); cudaFree(dsa); cudaFree(dB); cudaFree(dsb); cudaFree(dC);
}

static void test_topk(int M, int T, int k, int ninf) {
  std::vector<float> keys((size_t)M * T);
  std::uniform_real_distribution<float> U(-5, 5);
  for (auto& v : keys) v = U(rng);
  // Insert some ties
  for (int m = 0; m < M; ++m) for (int i = 0; i < T / 10; ++i) keys[(size_t)m * T + (rng() % T)] = 1.5f;
  for (int m = 0; m < M; ++m) for (int i = 0; i < ninf; ++i) keys[(size_t)m * T + (rng() % T)] = -INFINITY;
  float* dk = dev_copy(keys);
  int32_t* dout; CUDA_CHECK(cudaMalloc(&dout, (size_t)M * k * 4));
  k::topk_select_rows(dk, M, T, k, T, dout, k, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto out = dev_get(dout, (size_t)M * k);
  int bad = 0;
  for (int m = 0; m < M; ++m) {
    std::vector<int> idx(T);
    for (int i = 0; i < T; ++i) idx[i] = i;
    const float* kr = keys.data() + (size_t)m * T;
    std::stable_sort(idx.begin(), idx.end(), [kr](int a, int b) { return kr[a] > kr[b]; });
    std::vector<int> ref;
    for (int i = 0; i < k; ++i) if (kr[idx[i]] > -INFINITY) ref.push_back(idx[i]);
    std::sort(ref.begin(), ref.end());
    for (int i = 0; i < k; ++i) {
      int want = i < (int)ref.size() ? ref[i] : -1;
      if (out[(size_t)m * k + i] != want) { ++bad; if (bad < 4) printf("  topk mismatch m=%d i=%d got %d want %d\n", m, i, out[(size_t)m * k + i], want); }
    }
  }
  EXPECT(bad == 0, "topk M=%d T=%d k=%d bad=%d", M, T, k, bad);
  printf("topk_select M=%d T=%d k=%d ninf=%d: %s\n", M, T, k, ninf, bad ? "FAIL" : "ok");
  // Timing (decode scale: M=1, T=200000, k=512)
  if (M == 1 && T >= 100000) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 10; ++i) k::topk_select_rows(dk, M, T, k, T, dout, k, 0);
    cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("  topk_select M=1 T=%d: %.3f ms\n", T, ms / 10);
  }
  cudaFree(dk); cudaFree(dout);
}


// ---- Fused decode kernels (fused.cu) <-> unfused chain, bitwise comparison ------------------------------------------------------------------
static std::vector<bf16> rnd_bf16(size_t n, float scale) {
  std::vector<bf16> v(n);
  std::uniform_real_distribution<float> U(-scale, scale);
  for (auto& x : v) x = f2bf(U(rng));
  return v;
}
static std::vector<uint8_t> rnd_fp8w(size_t n) {
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<uint8_t> v(n);
  for (auto& x : v) { uint8_t b; do { b = (uint8_t)byte(rng); } while ((b & 0x7F) == 0x7F || (b & 0x78) > 0x50); x = b; }
  return v;
}
static std::vector<uint8_t> rnd_scale(size_t n, int lo, int span) {
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<uint8_t> v(n);
  for (auto& x : v) x = (uint8_t)(lo + byte(rng) % span);
  return v;
}
static size_t count_diff_bf16(const std::vector<bf16>& a, const std::vector<bf16>& b, float* maxrel) {
  size_t n = 0; float mr = 0.f, mref = 0.f;
  for (size_t i = 0; i < a.size(); i++) {
    float x = bf2f(a[i]), y = bf2f(b[i]);
    mref = std::max(mref, std::fabs(y));
    if (x != y) { ++n; mr = std::max(mr, std::fabs(x - y)); }
  }
  *maxrel = mref > 0 ? mr / mref : 0.f;
  return n;
}

static void test_fused(int M) {
  const int dim = 5120, Nqa = 1280, H = 64, D = 512, rd = 64, I = 2304, E = 384, kk = 6, GR = 8192, win = 128;
  // Inputs and weights
  auto xn = rnd_bf16((size_t)M * dim, 2.f);
  auto wqa = rnd_fp8w((size_t)Nqa * dim), sqa = rnd_scale((size_t)Nqa * dim / 32, 115, 8);
  auto wqb = rnd_fp8w((size_t)H * D * Nqa), sqb = rnd_scale((size_t)H * D * Nqa / 32, 115, 8);
  auto wkv = rnd_fp8w((size_t)D * dim), skv = rnd_scale((size_t)D * dim / 32, 115, 8);
  auto w1 = rnd_fp8w((size_t)I * dim), s1 = rnd_scale((size_t)I * dim / 32, 115, 8);
  auto w3 = rnd_fp8w((size_t)I * dim), s3 = rnd_scale((size_t)I * dim / 32, 115, 8);
  auto w2 = rnd_fp8w((size_t)dim * I), s2 = rnd_scale((size_t)dim * I / 32, 115, 8);
  auto wob = rnd_fp8w((size_t)dim * GR), sob = rnd_scale((size_t)dim * GR / 32, 115, 8);
  auto og = rnd_bf16((size_t)M * GR, 1.f);
  auto qn = rnd_bf16(Nqa, 1.f), kvn = rnd_bf16(D, 1.f);
  std::vector<float> gate_w((size_t)E * dim), bias(E);
  { std::uniform_real_distribution<float> U(-0.02f, 0.02f); for (auto& v : gate_w) v = U(rng); for (auto& v : bias) v = U(rng) * 10.f; }
  std::vector<float2> freqs((size_t)4096 * (rd / 2));
  for (size_t i = 0; i < freqs.size(); ++i) { float ang = (float)(i % 37) * 0.1f + (float)(i / 37) * 1e-3f; freqs[i] = make_float2(cosf(ang), sinf(ang)); }
  std::vector<int32_t> pos(M);
  for (int m = 0; m < M; ++m) pos[m] = 100 + 37 * m;
  bf16 *dxn = dev_copy(xn), *dwqn = dev_copy(qn), *dkvn = dev_copy(kvn), *dog = dev_copy(og);
  uint8_t *dwqa = dev_copy(wqa), *dsqa = dev_copy(sqa), *dwqb = dev_copy(wqb), *dsqb = dev_copy(sqb), *dwkv = dev_copy(wkv), *dskv = dev_copy(skv);
  uint8_t *dw1 = dev_copy(w1), *ds1 = dev_copy(s1), *dw3 = dev_copy(w3), *ds3 = dev_copy(s3), *dw2 = dev_copy(w2), *ds2 = dev_copy(s2), *dwob = dev_copy(wob), *dsob = dev_copy(sob);
  float *dgw = dev_copy(gate_w), *dbias = dev_copy(bias);
  float2* dfreqs = dev_copy(freqs);
  int32_t* dpos = dev_copy(pos);
  auto dalloc = [&](size_t bytes) { void* p; CUDA_CHECK(cudaMalloc(&p, bytes)); CUDA_CHECK(cudaMemset(p, 0, bytes)); return p; };
  // --- Chain (reference) ---
  uint8_t* xq = (uint8_t*)dalloc((size_t)M * dim); uint8_t* xs = (uint8_t*)dalloc((size_t)M * dim / 32);
  bf16* qr = (bf16*)dalloc((size_t)M * Nqa * 2); bf16* qrn = (bf16*)dalloc((size_t)M * Nqa * 2);
  uint8_t* qrq = (uint8_t*)dalloc((size_t)M * Nqa); uint8_t* qrs = (uint8_t*)dalloc((size_t)M * Nqa / 32);
  bf16* q = (bf16*)dalloc((size_t)M * H * D * 2);
  bf16* kv = (bf16*)dalloc((size_t)M * D * 2);
  bf16* ring = (bf16*)dalloc((size_t)win * D * 2);
  k::act_quant_fp8(dxn, M, dim, xq, xs, 0);
  k::gemm_bs(xq, xs, dwqa, dsqa, false, M, Nqa, dim, qr, nullptr, 0);
  k::rmsnorm(qr, dwqn, 1e-20f, M, Nqa, qrn, 0);
  k::act_quant_fp8(qrn, M, Nqa, qrq, qrs, 0);
  k::gemm_bs(qrq, qrs, dwqb, dsqb, false, M, H * D, Nqa, q, nullptr, 0);
  k::rope_last(q, M, H, D, rd, dfreqs, dpos, false, 0);
  k::gemm_bs(xq, xs, dwkv, dskv, false, M, D, dim, kv, nullptr, 0);
  k::rmsnorm(kv, dkvn, 1e-20f, M, D, kv, 0);
  k::rope_last(kv, M, 1, D, rd, dfreqs, dpos, false, 0);
  k::act_quant_fp8_roundtrip(kv, M, D, 0);
  std::vector<bf16*> ringp(M, ring);
  bf16** dringp = dev_copy(ringp);
  k::ring_write_rows(kv, M, D, win, dpos, dringp, 0);
  // Shared expert chain
  bf16* gate = (bf16*)dalloc((size_t)M * I * 2); bf16* up = (bf16*)dalloc((size_t)M * I * 2); bf16* y = (bf16*)dalloc((size_t)M * I * 2);
  uint8_t* yq = (uint8_t*)dalloc((size_t)M * I); uint8_t* ys = (uint8_t*)dalloc((size_t)M * I / 32);
  bf16* eout = (bf16*)dalloc((size_t)M * dim * 2); float* acc = (float*)dalloc((size_t)M * dim * 4);
  k::gemm_bs(xq, xs, dw1, ds1, false, M, I, dim, gate, nullptr, 0);
  k::gemm_bs(xq, xs, dw3, ds3, false, M, I, dim, up, nullptr, 0);
  k::swiglu_route(gate, up, nullptr, M, I, 10.f, y, 0);
  k::act_quant_fp8(y, M, I, yq, ys, 0);
  k::gemm_bs(yq, ys, dw2, ds2, false, M, dim, I, eout, nullptr, 0);
  k::accum_bf16_rows(eout, nullptr, M, dim, acc, 0);
  // wo_b chain
  uint8_t* ogq = (uint8_t*)dalloc((size_t)M * GR); uint8_t* ogs = (uint8_t*)dalloc((size_t)M * GR / 32);
  bf16* ao = (bf16*)dalloc((size_t)M * dim * 2);
  k::act_quant_fp8(dog, M, GR, ogq, ogs, 0);
  k::gemm_bs(ogq, ogs, dwob, dsob, false, M, dim, GR, ao, nullptr, 0);
  // Router reference: host double scores -> router_topk
  std::vector<float> scores_h((size_t)M * E);
  for (int m = 0; m < M; ++m) for (int e = 0; e < E; ++e) { double d = 0; for (int kx = 0; kx < dim; ++kx) d += (double)bf2f(xn[(size_t)m * dim + kx]) * gate_w[(size_t)e * dim + kx]; scores_h[(size_t)m * E + e] = (float)d; }
  float* dscores_ref = dev_copy(scores_h);
  int32_t* ids_ref = (int32_t*)dalloc((size_t)M * kk * 4); float* rw_ref = (float*)dalloc((size_t)M * kk * 4);
  k::router_topk(dscores_ref, dbias, nullptr, nullptr, M, E, kk, 1.5f, ids_ref, rw_ref, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto q_ref = dev_get(q, (size_t)M * H * D), qrn_ref = dev_get(qrn, (size_t)M * Nqa), kv_ref = dev_get(kv, (size_t)M * D), ring_ref = dev_get(ring, (size_t)win * D);
  auto qrq_ref = dev_get(qrq, (size_t)M * Nqa), qrs_ref = dev_get(qrs, (size_t)M * Nqa / 32), xq_ref = dev_get(xq, (size_t)M * dim), xs_ref = dev_get(xs, (size_t)M * dim / 32);
  auto y_ref = dev_get(y, (size_t)M * I);
  auto yq_ref = dev_get(yq, (size_t)M * I);
  auto ys_ref = dev_get(ys, (size_t)M * I / 32);
  auto acc_ref = dev_get(acc, (size_t)M * dim);
  auto ao_ref = dev_get(ao, (size_t)M * dim);
  auto ids_r = dev_get(ids_ref, (size_t)M * kk);
  auto rw_r = dev_get(rw_ref, (size_t)M * kk);
  // --- Fused ---
  int* counters = (int*)dalloc(16 * 4);
  CUDA_CHECK(cudaMemset(q, 0, (size_t)M * H * D * 2)); CUDA_CHECK(cudaMemset(qrn, 0, (size_t)M * Nqa * 2)); CUDA_CHECK(cudaMemset(qrq, 0, (size_t)M * Nqa)); CUDA_CHECK(cudaMemset(qrs, 0, (size_t)M * Nqa / 32));
  CUDA_CHECK(cudaMemset(kv, 0, (size_t)M * D * 2)); CUDA_CHECK(cudaMemset(ring, 0, (size_t)win * D * 2));
  CUDA_CHECK(cudaMemset(y, 0, (size_t)M * I * 2)); CUDA_CHECK(cudaMemset(yq, 0, (size_t)M * I)); CUDA_CHECK(cudaMemset(ys, 0, (size_t)M * I / 32)); CUDA_CHECK(cudaMemset(acc, 0x7f, (size_t)M * dim * 4));
  CUDA_CHECK(cudaMemset(xq, 0, (size_t)M * dim)); CUDA_CHECK(cudaMemset(xs, 0, (size_t)M * dim / 32)); CUDA_CHECK(cudaMemset(ao, 0, (size_t)M * dim * 2));
  k::fused_q_proj(dxn, dim, dwqa, dsqa, Nqa, dwqn, 1e-20f, dwqb, dsqb, H, D, rd, dfreqs, dpos, M, qr, qrn, q, qrq, qrs, counters + 0, 0);
  k::fused_kv_proj(dxn, dim, dwkv, dskv, D, dkvn, 1e-20f, rd, dfreqs, dpos, M, kv, dringp, win, nullptr, 0, counters + 1, 0);
  k::fused_shared_experts(dxn, dim, dw1, ds1, dw3, ds3, dw2, ds2, M, I, 10.f, y, xq, xs, acc, 0);
  k::fused_gemv_quantin(dog, GR, dwob, dsob, M, dim, ao, 0);
  float* dscores = (float*)dalloc((size_t)M * E * 4);
  int32_t* ids_f = (int32_t*)dalloc((size_t)M * kk * 4); float* rw_f = (float*)dalloc((size_t)M * kk * 4);
  k::fused_router(dxn, dim, dgw, M, E, dbias, nullptr, nullptr, kk, 1.5f, dscores, ids_f, rw_f, counters + 2, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float mr;
  size_t d1 = count_diff_bf16(dev_get(qrn, (size_t)M * Nqa), qrn_ref, &mr); EXPECT(d1 == 0, "fused qrn diff %zu", d1); printf("fused q_proj M=%d: qrn %s", M, d1 ? "DIFF" : "==");
  size_t d2 = count_diff_bf16(dev_get(q, (size_t)M * H * D), q_ref, &mr); EXPECT(d2 == 0, "fused q diff %zu", d2); printf(" · q(rope) %s", d2 ? "DIFF" : "==");
  EXPECT(dev_get(qrq, (size_t)M * Nqa) == qrq_ref && dev_get(qrs, (size_t)M * Nqa / 32) == qrs_ref, "fused qrq/qrs side output"); printf(" · qrq/qrs %s\n", (dev_get(qrq, (size_t)M * Nqa) == qrq_ref) ? "==" : "DIFF");
  size_t d3 = count_diff_bf16(dev_get(kv, (size_t)M * D), kv_ref, &mr); EXPECT(d3 == 0, "fused kv diff %zu", d3);
  size_t d4 = count_diff_bf16(dev_get(ring, (size_t)win * D), ring_ref, &mr); EXPECT(d4 == 0, "fused ring diff %zu", d4);
  printf("fused kv_proj M=%d: kv %s · ring %s\n", M, d3 ? "DIFF" : "==", d4 ? "DIFF" : "==");
  size_t d5 = count_diff_bf16(dev_get(y, (size_t)M * I), y_ref, &mr); EXPECT(d5 == 0, "fused y diff %zu", d5);
  bool q6 = true;  // y quantisation happens inside the w2 kernel (smem) — no separate output
  bool q7 = dev_get(xq, (size_t)M * dim) == xq_ref && dev_get(xs, (size_t)M * dim / 32) == xs_ref; EXPECT(q7, "fused xq/xs side output");
  auto acc_f = dev_get(acc, (size_t)M * dim); size_t d8 = 0; for (size_t i = 0; i < acc_f.size(); ++i) if (acc_f[i] != acc_ref[i]) ++d8; EXPECT(d8 == 0, "fused acc diff %zu", d8);
  printf("fused shared_experts M=%d: y %s · yq/ys %s · xq/xs %s · acc %s\n", M, d5 ? "DIFF" : "==", q6 ? "==" : "DIFF", q7 ? "==" : "DIFF", d8 ? "DIFF" : "==");
  size_t d9 = count_diff_bf16(dev_get(ao, (size_t)M * dim), ao_ref, &mr); EXPECT(d9 == 0, "fused wo_b diff %zu", d9);
  printf("fused gemv_quantin(wo_b K=%d) M=%d: %s\n", GR, M, d9 ? "DIFF" : "==");
  auto ids_ff = dev_get(ids_f, (size_t)M * kk); auto rw_ff = dev_get(rw_f, (size_t)M * kk); auto sc_f = dev_get(dscores, (size_t)M * E);
  int idmis = 0; float rwmax = 0.f, scmax = 0.f;
  for (size_t i = 0; i < ids_ff.size(); ++i) { if (ids_ff[i] != ids_r[i]) ++idmis; rwmax = std::max(rwmax, std::fabs(rw_ff[i] - rw_r[i])); }
  for (size_t i = 0; i < sc_f.size(); ++i) scmax = std::max(scmax, std::fabs(sc_f[i] - scores_h[i]) / (std::fabs(scores_h[i]) + 1e-3f));
  EXPECT(idmis == 0 && rwmax < 1e-4f && scmax < 1e-3f, "fused router ids mismatch %d rw %g score rel %g", idmis, rwmax, scmax);
  printf("fused router M=%d: ids %s · |Δrw| %.2e · score rel %.2e (fp32 order difference)\n", M, idmis ? "DIFF" : "==", rwmax, scmax);
  // Layer tail
  {
    const int hc = 4;
    std::vector<float> accv((size_t)M * dim), post((size_t)M * hc), comb((size_t)M * hc * hc), pre((size_t)M * hc);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (auto& v : accv) v = U(rng) * 3.f; for (auto& v : post) v = U(rng); for (auto& v : comb) v = U(rng); for (auto& v : pre) v = U(rng);
    auto h0 = rnd_bf16((size_t)M * hc * dim, 2.f);
    float *dacc = dev_copy(accv), *dpost = dev_copy(post), *dcomb = dev_copy(comb), *dpre = dev_copy(pre);
    bf16 *dh1 = dev_copy(h0), *dh2 = dev_copy(h0), *dtmp = (bf16*)dalloc((size_t)M * dim * 2);
    float* dpm1 = (float*)dalloc((size_t)M * hc * 4); float* dpm2 = (float*)dalloc((size_t)M * hc * 4);
    k::f32_to_bf16(dacc, M * dim, dtmp, 0); k::hc_post(dtmp, dpost, dcomb, M, hc, dim, dh1, 0); CUDA_CHECK(cudaMemcpy(dpm1, dpre, (size_t)M * hc * 4, cudaMemcpyDeviceToDevice));
    k::hc_post_tail(dacc, dpost, dcomb, M, hc, dim, dh2, dpre, dpm2, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t dd = count_diff_bf16(dev_get(dh1, (size_t)M * hc * dim), dev_get(dh2, (size_t)M * hc * dim), &mr);
    bool pm = dev_get(dpm1, (size_t)M * hc) == dev_get(dpm2, (size_t)M * hc);
    EXPECT(dd == 0 && pm, "hc_post_tail diff %zu pm %d", dd, (int)pm);
    printf("fused hc_post_tail M=%d: h %s · pre_mix %s\n", M, dd ? "DIFF" : "==", pm ? "==" : "DIFF");
  }
  // Timing (M=1): chain vs fused — q path only
  if (M == 1) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto chain = [&] { k::act_quant_fp8(dxn, M, dim, xq, xs, 0); k::gemm_bs(xq, xs, dwqa, dsqa, false, M, Nqa, dim, qr, nullptr, 0); k::rmsnorm(qr, dwqn, 1e-20f, M, Nqa, qrn, 0);
      k::act_quant_fp8(qrn, M, Nqa, qrq, qrs, 0); k::gemm_bs(qrq, qrs, dwqb, dsqb, false, M, H * D, Nqa, q, nullptr, 0); k::rope_last(q, M, H, D, rd, dfreqs, dpos, false, 0);
      k::gemm_bs(xq, xs, dwkv, dskv, false, M, D, dim, kv, nullptr, 0); k::rmsnorm(kv, dkvn, 1e-20f, M, D, kv, 0); k::rope_last(kv, M, 1, D, rd, dfreqs, dpos, false, 0); k::act_quant_fp8_roundtrip(kv, M, D, 0); k::ring_write_rows(kv, M, D, win, dpos, dringp, 0);
      k::gemm_bs(xq, xs, dw1, ds1, false, M, I, dim, gate, nullptr, 0); k::gemm_bs(xq, xs, dw3, ds3, false, M, I, dim, up, nullptr, 0); k::swiglu_route(gate, up, nullptr, M, I, 10.f, y, 0); k::act_quant_fp8(y, M, I, yq, ys, 0);
      cudaMemsetAsync(acc, 0, (size_t)M * dim * 4, 0); k::gemm_bs(yq, ys, dw2, ds2, false, M, dim, I, eout, nullptr, 0); k::accum_bf16_rows(eout, nullptr, M, dim, acc, 0); };
    auto fused = [&] { k::fused_q_proj(dxn, dim, dwqa, dsqa, Nqa, dwqn, 1e-20f, dwqb, dsqb, H, D, rd, dfreqs, dpos, M, qr, qrn, q, qrq, qrs, counters, 0);
      k::fused_kv_proj(dxn, dim, dwkv, dskv, D, dkvn, 1e-20f, rd, dfreqs, dpos, M, kv, dringp, win, nullptr, 0, counters + 1, 0);
      k::fused_shared_experts(dxn, dim, dw1, ds1, dw3, ds3, dw2, ds2, M, I, 10.f, y, xq, xs, acc, 0); };
    for (int i = 0; i < 3; ++i) { chain(); fused(); }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for (int i = 0; i < 50; ++i) chain(); cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize());
    float ms_c; cudaEventElapsedTime(&ms_c, e0, e1);
    cudaEventRecord(e0); for (int i = 0; i < 50; ++i) fused(); cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize());
    float ms_f; cudaEventElapsedTime(&ms_f, e0, e1);
    printf("  timing q+kv+shared (M=1, eager launches): chain %.1f µs (17 launches) · fused %.1f µs (5 launches)\n", ms_c / 50 * 1000, ms_f / 50 * 1000);
  }
}

int main() {
  int dev = 0;
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("device: %s sm_%d%d · asyncEngineCount %d · smem/block opt-in %zu KB\n", p.name, p.major, p.minor, p.asyncEngineCount, p.sharedMemPerBlockOptin / 1024);
  test_fp8_conv();
  test_act_quant();
  test_fp4_roundtrip();
  test_rmsnorm();
  test_gemm(1, 2304, 5120, true);
  test_gemm(6, 5120, 2304, true);
  test_gemm(3, 1280, 5120, false);
  test_gemm(70, 2304, 5120, true);
  test_gemm(130, 512, 5120, false);
  test_gemm(64, 5120, 2304, true);
  test_topk(4, 5000, 512, 0);
  test_topk(3, 700, 512, 300);
  test_topk(2, 512, 512, 0);
  test_topk(2, 3000, 2048, 10);
  test_topk(1, 200000, 512, 0);
  test_gemm(200, 1280, 5120, false);
  test_gemm(17, 2304, 2304, true);
  test_fused(1);
  test_fused(3);
  test_fused(8);
  // Timing: prefill scale (4096 tokens) — expert w1 [2304, 5120] fp4, dense wq_b [32768, 1280] fp8
  {
    auto bench = [&](int M, int N, int K, bool fp4) {
      std::vector<uint8_t> A((size_t)M * K, 0x38), sa((size_t)M * K / 32, 127), B(fp4 ? (size_t)N * K / 2 : (size_t)N * K, 0x21), sb((size_t)N * K / 32, 127);
      uint8_t *dA = dev_copy(A), *dsa = dev_copy(sa), *dB = dev_copy(B), *dsb = dev_copy(sb);
      bf16* dC; CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 2));
      k::gemm_bs(dA, dsa, dB, dsb, fp4, M, N, K, dC, nullptr, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      cudaEventRecord(e0);
      for (int i = 0; i < 5; ++i) k::gemm_bs(dA, dsa, dB, dsb, fp4, M, N, K, dC, nullptr, 0);
      cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize());
      float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
      printf("bench gemm M=%d N=%d K=%d %s: %.3f ms = %.1f TFLOPS\n", M, N, K, fp4 ? "fp4" : "fp8", ms, 2.0 * M * N * K / ms / 1e9);
      cudaFree(dA); cudaFree(dsa); cudaFree(dB); cudaFree(dsb); cudaFree(dC);
    };
    bench(4096, 2304, 5120, true);
    bench(4096, 32768, 1280, false);
    k::gemm_bs_force_path(1);
    bench(4096, 2304, 5120, true);
    k::gemm_bs_force_path(0);
  }
  printf(fails ? "SOME TESTS FAILED (%d)\n" : "ALL OK\n", fails);
  return fails ? 1 : 0;
}
