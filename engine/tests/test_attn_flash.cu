// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Validates sparse_attn_flash64 (v4, 64-head blocks) == sparse_attn_flash_v3 (v3) bit for bit + v3 vs sparse_attn_tc (v2) vs sparse_attn (v1) — same inputs (ring bf16 · chunk bf16 · compressed fp4 packing · indices mixed with -1 · sink).
//   The versions differ only in fp32 accumulation order and the reference for P's bf16 rounding (v3 = running max) → judged by relative error against max|o| per row (m,h). The v1↔v2 difference is printed as the noise floor.
//   Run: scripts/hive-run.sh "./build-dev/test_attn_flash"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "test_util.h"

using namespace hive;
static float bits2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }
static uint16_t f2bits(float f) { bf16 b = f2bf(f); uint16_t u; memcpy(&u, &b, 2); return u; }

static double cmp(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int M, int H, int D, double* mean_out) {
  double worst = 0, tot = 0; size_t n = 0;
  for (int m = 0; m < M; ++m)
    for (int h = 0; h < H; ++h) {
      double scale = 1e-6, e = 0;
      for (int d = 0; d < D; ++d) scale = std::max(scale, (double)std::fabs(bits2f(a[((size_t)m * H + h) * D + d])));
      for (int d = 0; d < D; ++d) { const double df = std::fabs((double)bits2f(a[((size_t)m * H + h) * D + d]) - bits2f(b[((size_t)m * H + h) * D + d])) / scale; e = std::max(e, df); tot += df; ++n; }
      worst = std::max(worst, e);
    }
  *mean_out = tot / n;
  return worst;
}

int main() {
  const int H = 64, D = 512, win = 128, topk_cols = 512, idx_stride = win + topk_cols, T = 700;
  bool all_ok = true;
  struct Case { int M; int topk_total; unsigned seed; };
  for (const Case cs : {Case{1, 640, 1}, Case{16, 640, 2}, Case{37, 640, 3}, Case{40, 128 + 300, 4}, Case{33, 128, 5}}) {
    const int M = cs.M, topk_total = cs.topk_total;
    std::mt19937 rng(100 + cs.seed);
    std::normal_distribution<float> nd;
    std::vector<uint16_t> q((size_t)M * H * D), ring((size_t)win * D), chunk((size_t)M * D), comp_bf((size_t)T * D);
    for (auto& v : q) v = f2bits(nd(rng) * 1.5f);  // scale 0.05 made softmax nearly uniform and left rescaling · P rounding · sink untested — S std ≈ 1.5
    for (auto& v : ring) v = f2bits(nd(rng));
    for (auto& v : chunk) v = f2bits(nd(rng));
    for (auto& v : comp_bf) v = f2bits(nd(rng));
    std::vector<float> sink(H);
    for (auto& v : sink) v = nd(rng) * 2.0f;
    std::vector<int32_t> idx((size_t)M * idx_stride, -1);
    std::uniform_int_distribution<int> ur(0, win + M - 1), uc(0, T - 1), coin(0, 9);
    for (int m = 0; m < M; ++m) {
      for (int j = 0; j < win; ++j) idx[(size_t)m * idx_stride + j] = coin(rng) == 0 ? -1 : ur(rng);
      const int nc = topk_total - win;
      std::vector<int> pos(T);
      for (int t = 0; t < T; ++t) pos[t] = t;
      std::shuffle(pos.begin(), pos.end(), rng);
      for (int j = 0; j < nc; ++j) idx[(size_t)m * idx_stride + win + j] = (j >= nc - 3 || coin(rng) == 0) ? -1 : win + M + pos[j];
    }
    // edge cases: row 1 is all -1 (output must be 0), one key block (16 keys) of row 0 is all -1
    if (M > 1) for (int j = 0; j < idx_stride; ++j) idx[(size_t)1 * idx_stride + j] = -1;
    for (int j = 16; j < 32 && j < topk_total; ++j) idx[j] = -1;
    bf16 *dq, *dring, *dchunk, *dcomp_bf, *o1, *o2, *o3, *o4; uint8_t* dcomp; int32_t* didx; float *dsink, *S, *mx, *sum;
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 2)); CUDA_CHECK(cudaMalloc(&dring, ring.size() * 2)); CUDA_CHECK(cudaMalloc(&dchunk, chunk.size() * 2));
    CUDA_CHECK(cudaMalloc(&dcomp_bf, comp_bf.size() * 2)); CUDA_CHECK(cudaMalloc(&dcomp, (size_t)T * kvp::COMP_ROW));
    CUDA_CHECK(cudaMalloc(&didx, idx.size() * 4)); CUDA_CHECK(cudaMalloc(&dsink, H * 4));
    CUDA_CHECK(cudaMalloc(&o1, q.size() * 2)); CUDA_CHECK(cudaMalloc(&o2, q.size() * 2)); CUDA_CHECK(cudaMalloc(&o3, q.size() * 2)); CUDA_CHECK(cudaMalloc(&o4, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&S, (size_t)M * H * topk_total * 4)); CUDA_CHECK(cudaMalloc(&mx, (size_t)M * H * 4)); CUDA_CHECK(cudaMalloc(&sum, (size_t)M * H * 4));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dring, ring.data(), ring.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dchunk, chunk.data(), chunk.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dcomp_bf, comp_bf.data(), comp_bf.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dsink, sink.data(), H * 4, cudaMemcpyHostToDevice));
    k::fp4_pack(dcomp_bf, T, D, 16, true, dcomp, kvp::COMP_ROW, 0);  // same compressed KV format as the runtime
    k::KvSources kv{dring, win, dchunk, M, dcomp, T};
    const float scale = 1.0f / sqrtf((float)D);
    k::sparse_attn(dq, M, H, D, kv, didx, idx_stride, topk_total, dsink, scale, o1, 0);
    k::sparse_attn_tc(dq, M, H, D, kv, didx, idx_stride, topk_total, dsink, scale, S, mx, sum, o2, 0);
    k::sparse_attn_flash_v3(dq, M, H, D, kv, didx, idx_stride, topk_total, dsink, scale, o3, 0);
    k::sparse_attn_flash64(dq, M, H, D, kv, didx, idx_stride, topk_total, dsink, scale, o4, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> h1(q.size()), h2(q.size()), h3(q.size()), h4(q.size());
    CUDA_CHECK(cudaMemcpy(h1.data(), o1, h1.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), o2, h2.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h3.data(), o3, h3.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h4.data(), o4, h4.size() * 2, cudaMemcpyDeviceToHost));
    // host double reference (rows 0..2): kv uses the bf16 rows as is / compressed rows are unpacked on the host (e4m3 scale per 16 dims · low nibble = even d)
    std::vector<uint8_t> compp((size_t)T * kvp::COMP_ROW);
    CUDA_CHECK(cudaMemcpy(compp.data(), dcomp, compp.size(), cudaMemcpyDeviceToHost));
    auto kvv = [&](int ix, int d) -> double {
      if (ix < win) return bits2f(ring[(size_t)ix * D + d]);
      if (ix < win + M) return bits2f(chunk[(size_t)(ix - win) * D + d]);
      const uint8_t* row = &compp[(size_t)(ix - win - M) * kvp::COMP_ROW];
      const uint8_t byte = row[kvp::COMP_SCALES + d / 2];
      return (double)e2m1_to_f32((d & 1) ? (byte >> 4) : (byte & 0xF)) * e4m3_to_f32(row[d / 16]);
    };
    double e_ref = 0;
    for (int m = 0; m < std::min(M, 3); ++m)
      for (int h = 0; h < H; ++h) {
        std::vector<double> sc(topk_total, -INFINITY);
        double mx = -INFINITY;
        for (int t = 0; t < topk_total; ++t) {
          const int ix = idx[(size_t)m * idx_stride + t];
          if (ix < 0) continue;
          double dd = 0;
          for (int d = 0; d < D; ++d) dd += (double)bits2f(q[((size_t)m * H + h) * D + d]) * kvv(ix, d);
          sc[t] = dd * scale; mx = std::max(mx, sc[t]);
        }
        if (!(mx > -INFINITY)) continue;  // all -1 rows are covered by the zero-row check below
        const double mref = std::max(mx, (double)sink[h]);
        double den = std::exp(sink[h] - mref);
        std::vector<double> acc(D, 0.0);
        for (int t = 0; t < topk_total; ++t) {
          if (!(sc[t] > -INFINITY)) continue;
          const double p = std::exp(sc[t] - mref); den += p;
          const int ix = idx[(size_t)m * idx_stride + t];
          for (int d = 0; d < D; ++d) acc[d] += p * kvv(ix, d);
        }
        double scl = 1e-6;
        for (int d = 0; d < D; ++d) scl = std::max(scl, std::fabs(acc[d] / den));
        for (int d = 0; d < D; ++d) e_ref = std::max(e_ref, std::fabs(bits2f(h4[((size_t)m * H + h) * D + d]) - acc[d] / den) / scl);
      }
    double mean12, mean23, mean13;
    const double e12 = cmp(h1, h2, M, H, D, &mean12), e23 = cmp(h2, h3, M, H, D, &mean23), e13 = cmp(h1, h3, M, H, D, &mean13);
    size_t nan3 = 0;
    for (auto v : h4) if (std::isnan(bits2f(v)) || std::isinf(bits2f(v))) ++nan3;
    size_t diff34 = 0;  // v4 uses the same operation order as v3 for every head row → must be bit-identical
    for (size_t i = 0; i < h3.size(); ++i) diff34 += h3[i] != h4[i];
    bool zero_row = true;  // an all -1 row must be 0 (same convention as v2)
    if (M > 1) for (int i = 0; i < H * D; ++i) if (bits2f(h4[(size_t)1 * H * D + i]) != 0.f) zero_row = false;
    const bool ok = nan3 == 0 && e23 < 3e-2 && mean23 < 3e-3 && e13 < 3e-2 && e_ref < 3e-2 && zero_row && diff34 == 0;
    printf("M=%-3d topk=%-4d  v1↔v2 max %.2e mean %.2e | v2↔v3 max %.2e mean %.2e | v1↔v3 max %.2e | v4↔ref %.2e | v3≠v4 %zu | zero-row %s | nan %zu  %s\n",
           M, topk_total, e12, mean12, e23, mean23, e13, e_ref, diff34, zero_row ? "ok" : "BAD", nan3, ok ? "PASS" : "FAIL");
    all_ok = all_ok && ok;
    cudaFree(dq); cudaFree(dring); cudaFree(dchunk); cudaFree(dcomp_bf); cudaFree(dcomp); cudaFree(didx); cudaFree(dsink);
    cudaFree(o1); cudaFree(o2); cudaFree(o3); cudaFree(o4); cudaFree(S); cudaFree(mx); cudaFree(sum);
  }
  // speed sample: M=2048 · topk 640 · v2 vs v3 (events) — one layer's worth
  {
    const int M = 2048, topk_total = 640;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd;
    std::vector<uint16_t> q((size_t)M * H * D), ring((size_t)win * D), chunk((size_t)M * D), comp_bf((size_t)T * D);
    for (auto& v : q) v = f2bits(nd(rng) * 0.05f);
    for (auto& v : ring) v = f2bits(nd(rng));
    for (auto& v : chunk) v = f2bits(nd(rng));
    for (auto& v : comp_bf) v = f2bits(nd(rng));
    std::vector<float> sink(H, 0.f);
    std::vector<int32_t> idx((size_t)M * idx_stride, -1);
    std::uniform_int_distribution<int> ur(0, win + M - 1), uc(0, T - 1);
    for (auto& v : idx) v = 0;
    for (int m = 0; m < M; ++m) {
      for (int j = 0; j < win; ++j) idx[(size_t)m * idx_stride + j] = ur(rng);
      for (int j = 0; j < topk_cols; ++j) idx[(size_t)m * idx_stride + win + j] = win + M + (int)(((size_t)m * 131 + j) % T);  // no duplicates within a row
    }
    bf16 *dq, *dring, *dchunk, *dcomp_bf, *o2, *o3, *o4; uint8_t* dcomp; int32_t* didx; float *dsink, *S, *mx, *sum;
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 2)); CUDA_CHECK(cudaMalloc(&dring, ring.size() * 2)); CUDA_CHECK(cudaMalloc(&dchunk, chunk.size() * 2));
    CUDA_CHECK(cudaMalloc(&dcomp_bf, comp_bf.size() * 2)); CUDA_CHECK(cudaMalloc(&dcomp, (size_t)T * kvp::COMP_ROW));
    CUDA_CHECK(cudaMalloc(&didx, idx.size() * 4)); CUDA_CHECK(cudaMalloc(&dsink, H * 4));
    CUDA_CHECK(cudaMalloc(&o2, q.size() * 2)); CUDA_CHECK(cudaMalloc(&o3, q.size() * 2)); CUDA_CHECK(cudaMalloc(&o4, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&S, (size_t)M * H * topk_total * 4)); CUDA_CHECK(cudaMalloc(&mx, (size_t)M * H * 4)); CUDA_CHECK(cudaMalloc(&sum, (size_t)M * H * 4));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dring, ring.data(), ring.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dchunk, chunk.data(), chunk.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dcomp_bf, comp_bf.data(), comp_bf.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dsink, sink.data(), H * 4, cudaMemcpyHostToDevice));
    k::fp4_pack(dcomp_bf, T, D, 16, true, dcomp, kvp::COMP_ROW, 0);
    k::KvSources kv{dring, win, dchunk, M, dcomp, T};
    const float scale = 1.0f / sqrtf((float)D);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int rep = 0; rep < 2; ++rep) {
      cudaEventRecord(e0); k::sparse_attn_tc(dq, M, H, D, kv, didx, idx_stride, topk_total, dsink, scale, S, mx, sum, o2, 0); cudaEventRecord(e1);
      CUDA_CHECK(cudaDeviceSynchronize()); float ms2; cudaEventElapsedTime(&ms2, e0, e1);
      cudaEventRecord(e0); k::sparse_attn_flash_v3(dq, M, H, D, kv, didx, idx_stride, topk_total, dsink, scale, o3, 0); cudaEventRecord(e1);
      CUDA_CHECK(cudaDeviceSynchronize()); float ms3; cudaEventElapsedTime(&ms3, e0, e1);
      cudaEventRecord(e0); k::sparse_attn_flash64(dq, M, H, D, kv, didx, idx_stride, topk_total, dsink, scale, o4, 0); cudaEventRecord(e1);
      CUDA_CHECK(cudaDeviceSynchronize()); float ms4; cudaEventElapsedTime(&ms4, e0, e1);
      const double flop = (double)M * H * topk_total * D * 4;
      if (rep == 1)
        printf("speed M=2048 topk=640 (one layer): v2 %.2f ms (%.1f TFLOPS) · v3 %.2f ms (%.1f TFLOPS) · v4 %.2f ms (%.1f TFLOPS) · v3/v4 ×%.2f\n", ms2,
               flop / ms2 / 1e9, ms3, flop / ms3 / 1e9, ms4, flop / ms4 / 1e9, ms3 / ms4);
    }
  }
  printf(all_ok ? "ALL PASS\n" : "SOME FAIL\n");
  return all_ok ? 0 : 1;
}
