// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Check: hc_mix_pre_norm (one launch) vs hc_mix + hc_pre_norm (two launches) — mixes, rsq, x and xn must be
//   bit-identical (same device functions). Decode shapes (M 1/3/8, hc 4, dim 5120).
//   Run: scripts/hive-run.sh "./build-dev/test_hc_fuse"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/model_kernels.h"

using namespace hive;
static uint16_t f2bits(float f) { bf16 b = f2bf(f); uint16_t u; memcpy(&u, &b, 2); return u; }

template <class T>
static std::vector<T> get(const T* d, size_t n) { std::vector<T> v(n); CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }

int main() {
  const int hc = 4, dim = 5120, hcdim = hc * dim, mix = (2 + hc) * hc;
  const float eps = 1e-20f;
  bool all_ok = true;
  for (const int M : {1, 3, 8}) {
    std::mt19937 rng(17 + M);
    std::normal_distribution<float> nd;
    std::vector<uint16_t> h((size_t)M * hcdim), nw(dim);
    std::vector<float> W((size_t)mix * hcdim), pre((size_t)M * hc);
    for (auto& v : h) v = f2bits(nd(rng));
    for (auto& v : nw) v = f2bits(1.f + 0.1f * nd(rng));
    for (auto& v : W) v = nd(rng) * 0.01f;
    for (auto& v : pre) v = 0.25f + 0.1f * nd(rng);
    bf16 *dh, *dnw, *x1, *xn1, *x2, *xn2; float *dW, *dpre, *m1, *r1, *m2, *r2;
    CUDA_CHECK(cudaMalloc(&dh, h.size() * 2)); CUDA_CHECK(cudaMalloc(&dnw, nw.size() * 2)); CUDA_CHECK(cudaMalloc(&dW, W.size() * 4)); CUDA_CHECK(cudaMalloc(&dpre, pre.size() * 4));
    for (bf16** p : {&x1, &xn1, &x2, &xn2}) CUDA_CHECK(cudaMalloc(p, (size_t)M * dim * 2));
    for (float** p : {&m1, &m2}) CUDA_CHECK(cudaMalloc(p, (size_t)M * mix * 4));
    for (float** p : {&r1, &r2}) CUDA_CHECK(cudaMalloc(p, (size_t)M * 4));
    CUDA_CHECK(cudaMemcpy(dh, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dnw, nw.data(), nw.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * 4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dpre, pre.data(), pre.size() * 4, cudaMemcpyHostToDevice));
    k::hc_mix(dh, dW, M, hcdim, mix, eps, m1, r1, 0);
    k::hc_pre_norm(dh, dpre, M, hc, dim, dnw, eps, x1, xn1, 0);
    k::hc_mix_pre_norm(dh, dW, M, hcdim, mix, eps, m2, r2, dpre, hc, dim, dnw, eps, x2, xn2, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    const bool eq_m = get(m1, (size_t)M * mix) == get(m2, (size_t)M * mix), eq_r = get(r1, (size_t)M) == get(r2, (size_t)M);
    auto bx1 = get(reinterpret_cast<uint16_t*>(x1), (size_t)M * dim), bx2 = get(reinterpret_cast<uint16_t*>(x2), (size_t)M * dim);
    auto bn1 = get(reinterpret_cast<uint16_t*>(xn1), (size_t)M * dim), bn2 = get(reinterpret_cast<uint16_t*>(xn2), (size_t)M * dim);
    const bool ok = eq_m && eq_r && bx1 == bx2 && bn1 == bn2;
    printf("M=%d  mixes %s · rsq %s · x %s · xn %s  %s\n", M, eq_m ? "=" : "≠", eq_r ? "=" : "≠", bx1 == bx2 ? "=" : "≠", bn1 == bn2 ? "=" : "≠", ok ? "PASS" : "FAIL");
    all_ok = all_ok && ok;
    for (void* p : {(void*)dh, (void*)dnw, (void*)dW, (void*)dpre, (void*)x1, (void*)xn1, (void*)x2, (void*)xn2, (void*)m1, (void*)m2, (void*)r1, (void*)r2}) cudaFree(p);
  }
  printf(all_ok ? "ALL PASS\n" : "SOME FAIL\n");
  return all_ok ? 0 : 1;
}
