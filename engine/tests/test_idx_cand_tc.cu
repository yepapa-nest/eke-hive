// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Validation: indexer_scores_cand_tc (candidate pool, e2m1 x e2m1 tensor cores) vs indexer_scores_cand (CUDA cores, round-tripped bf16 query) vs a host double reference
//   + bitwise comparison with the full-T tensor-core version (indexer_scores_tc) gathered at the candidate positions (informational — same mma and block order, so 0 is expected; not part of the verdict).
//   Two modes: single key cache (prefill Reindex layer) and row table (k_ptrs + T_rows, batched decode — a different cache and length per row).
//   Candidates: block ids ascending + trailing -1; invisible blocks (beyond visible) mixed in; kb·bs chosen not to divide evenly by the warp tile (32) or block (128).
//   Pass = identical -inf positions + finite-value relative error (relative to the row max |reference|) < 2e-2 on both GPU paths (same criterion as test_idx_tc).
//   Run: scripts/hive-run.sh "./build-dev/test_idx_cand_tc"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/kernels.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "test_util.h"

using namespace hive;
static float bits2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }
static uint16_t f2bits(float f) { bf16 b = f2bf(f); uint16_t u; memcpy(&u, &b, 2); return u; }
static float bfr(double v) { return bits2f(f2bits((float)v)); }
static double pval(const uint8_t* row, int d) {
  const int nib = (row[kvp::IDX_HDR + (d >> 1)] >> (4 * (d & 1))) & 0xF;
  return (double)e2m1_to_f32(nib) * e8m0_to_f32(row[d >> 5]);
}

int main() {
  bool all_ok = true;
  struct Case { int M, T, Hi, kb, rows_mode, seed; };
  for (const Case cs : {Case{1, 3000, 32, 300, 0, 1}, Case{9, 3000, 32, 301, 0, 2}, Case{5, 700, 64, 70, 0, 3}, Case{6, 2500, 32, 250, 1, 4},
                        Case{3, 20000, 32, 2048, 0, 5}}) {
    const int M = cs.M, T = cs.T, Hi = cs.Hi, Di = kvp::IDX_D, R = kvp::IDX_ROW, bs = 8, kb = cs.kb, ncand = kb * bs, CB = kb + 3;
    const int nblocks = (T + bs - 1) / bs;
    const bool rows_mode = cs.rows_mode != 0;
    std::mt19937 rng(21 + cs.seed);
    std::normal_distribution<float> nd;
    // Row-table mode: two caches (lengths T, T/2) alternating per row. Single mode: one cache.
    const int ncache = rows_mode ? 2 : 1;
    std::vector<int> clen(ncache);
    for (int c = 0; c < ncache; ++c) clen[c] = c == 0 ? T : T / 2;
    std::vector<uint16_t> Q((size_t)M * Hi * Di), W((size_t)M * Hi);
    std::vector<std::vector<uint16_t>> K(ncache);
    for (auto& v : Q) v = f2bits(nd(rng) * 2.f);
    for (auto& v : W) v = f2bits(nd(rng) * 0.02f);
    for (int c = 0; c < ncache; ++c) { K[c].resize((size_t)clen[c] * Di); for (auto& v : K[c]) v = f2bits(nd(rng)); }
    std::vector<int32_t> vis(M), trows(M);
    for (int m = 0; m < M; ++m) { vis[m] = (m == 0) ? T : 1 + (int)(rng() % T); trows[m] = clen[m % ncache]; }
    // Candidate blocks: a random ascending subset per row + a few trailing -1
    std::vector<int32_t> cand((size_t)M * CB, -1);
    for (int m = 0; m < M; ++m) {
      std::vector<int> ids(nblocks);
      for (int b = 0; b < nblocks; ++b) ids[b] = b;
      std::shuffle(ids.begin(), ids.end(), rng);
      const int nreal = std::min(nblocks, kb - (m % 3));  // depending on the row, 1–2 fewer than kb (-1 tail)
      std::sort(ids.begin(), ids.begin() + nreal);
      for (int j = 0; j < nreal; ++j) cand[(size_t)m * CB + j] = ids[j];
    }
    bf16 *dQ, *dW, *s_ref, *s_tc, *s_full; uint8_t* dQp; int32_t *dV, *dC, *dTr; const uint8_t** dKptrs;
    std::vector<bf16*> dK(ncache); std::vector<uint8_t*> dKp(ncache);
    CUDA_CHECK(cudaMalloc(&dQ, Q.size() * 2)); CUDA_CHECK(cudaMalloc(&dW, W.size() * 2)); CUDA_CHECK(cudaMalloc(&dQp, (size_t)M * Hi * R));
    CUDA_CHECK(cudaMalloc(&dV, M * 4)); CUDA_CHECK(cudaMalloc(&dTr, M * 4)); CUDA_CHECK(cudaMalloc(&dC, cand.size() * 4));
    CUDA_CHECK(cudaMalloc(&s_ref, (size_t)M * ncand * 2)); CUDA_CHECK(cudaMalloc(&s_tc, (size_t)M * ncand * 2)); CUDA_CHECK(cudaMalloc(&s_full, (size_t)M * T * 2));
    CUDA_CHECK(cudaMalloc(&dKptrs, M * sizeof(uint8_t*)));
    for (int c = 0; c < ncache; ++c) {
      CUDA_CHECK(cudaMalloc(&dK[c], K[c].size() * 2)); CUDA_CHECK(cudaMalloc(&dKp[c], (size_t)clen[c] * R));
      CUDA_CHECK(cudaMemcpy(dK[c], K[c].data(), K[c].size() * 2, cudaMemcpyHostToDevice));
      k::fp4_pack(dK[c], clen[c], Di, 32, false, dKp[c], R, 0);
    }
    std::vector<const uint8_t*> kptrs(M);
    for (int m = 0; m < M; ++m) kptrs[m] = dKp[m % ncache];
    CUDA_CHECK(cudaMemcpy(dKptrs, kptrs.data(), M * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, vis.data(), M * 4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dTr, trows.data(), M * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, cand.data(), cand.size() * 4, cudaMemcpyHostToDevice));
    k::fp4_pack(dQ, M * Hi, Di, 32, false, dQp, R, 0);  // same order as the runtime: pack before the round trip
    k::fp4_quant_roundtrip(dQ, M, Hi * Di, 32, false, 0);
    const uint8_t* const* kp_arg = rows_mode ? dKptrs : nullptr;
    const int32_t* tr_arg = rows_mode ? dTr : nullptr;
    const uint8_t* k_arg = rows_mode ? nullptr : dKp[0];
    k::indexer_scores_cand(dQ, k_arg, kp_arg, tr_arg, dW, M, Hi, Di, dC, CB, kb, bs, dV, s_ref, 0);
    k::indexer_scores_cand_tc(dQp, k_arg, kp_arg, tr_arg, dW, M, Hi, dC, CB, kb, bs, dV, s_tc, 0);
    if (!rows_mode) k::indexer_scores_tc(dQp, dKp[0], dW, M, Hi, T, dV, s_full, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> Qp((size_t)M * Hi * R);
    std::vector<std::vector<uint8_t>> Kp(ncache);
    std::vector<uint16_t> h_ref((size_t)M * ncand), h_tc((size_t)M * ncand), h_full((size_t)M * T);
    CUDA_CHECK(cudaMemcpy(Qp.data(), dQp, Qp.size(), cudaMemcpyDeviceToHost));
    for (int c = 0; c < ncache; ++c) { Kp[c].resize((size_t)clen[c] * R); CUDA_CHECK(cudaMemcpy(Kp[c].data(), dKp[c], Kp[c].size(), cudaMemcpyDeviceToHost)); }
    CUDA_CHECK(cudaMemcpy(h_ref.data(), s_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tc.data(), s_tc, h_tc.size() * 2, cudaMemcpyDeviceToHost));
    if (!rows_mode) CUDA_CHECK(cudaMemcpy(h_full.data(), s_full, h_full.size() * 2, cudaMemcpyDeviceToHost));
    double e_ref = 0, e_tc = 0; size_t inf_mis = 0, full_mis = 0, nfinite = 0;
    for (int m = 0; m < M; ++m) {
      const int v = rows_mode ? std::min(vis[m], trows[m]) : vis[m];
      const uint8_t* kc = Kp[m % ncache].data();
      std::vector<double> host(ncand, 0.0); std::vector<int> pos(ncand, -1); double scale = 1e-6;
      for (int ci = 0; ci < ncand; ++ci) {
        const int32_t b = cand[(size_t)m * CB + ci / bs];
        const int t = b >= 0 ? b * bs + ci % bs : -1;
        if (t < 0 || t >= v) continue;
        pos[ci] = t;
        double tot = 0;
        for (int h = 0; h < Hi; ++h) {
          double d = 0;
          for (int dd = 0; dd < Di; ++dd) d += pval(&Qp[((size_t)m * Hi + h) * R], dd) * pval(&kc[(size_t)t * R], dd);
          double sb = bfr(d); sb = sb > 0 ? sb : 0; sb = bfr(sb * bits2f(W[(size_t)m * Hi + h]));
          tot += sb;
        }
        host[ci] = tot; scale = std::max(scale, std::fabs(tot));
      }
      for (int ci = 0; ci < ncand; ++ci) {
        const float a = bits2f(h_ref[(size_t)m * ncand + ci]), b = bits2f(h_tc[(size_t)m * ncand + ci]);
        const bool ia = std::isinf(a) && a < 0, ib = std::isinf(b) && b < 0, ih = pos[ci] < 0;
        if (ia != ih || ib != ih) { ++inf_mis; continue; }
        if (ih) continue;
        ++nfinite;
        e_ref = std::max(e_ref, std::fabs(a - host[ci]) / scale);
        e_tc = std::max(e_tc, std::fabs(b - host[ci]) / scale);
        if (!rows_mode && h_full[(size_t)m * T + pos[ci]] != h_tc[(size_t)m * ncand + ci]) ++full_mis;
      }
    }
    const bool ok = inf_mis == 0 && e_ref < 2e-2 && e_tc < 2e-2 && nfinite > 0 && (rows_mode || full_mis == 0);  // bit identity with the full-T tensor-core version is the sharpest check of the fragment layout
    printf("M=%-2d T=%-5d Hi=%d kb=%-4d %s  finite=%zu inf_mismatch=%zu err(cuda-core)=%.3e err(tc)=%.3e  tc≠full-T-tc=%zu%s  %s\n", M, T, Hi, kb,
           rows_mode ? "rows" : "single", nfinite, inf_mis, e_ref, e_tc, full_mis, rows_mode ? "(n/a)" : "", ok ? "PASS" : "FAIL");
    all_ok = all_ok && ok;
    cudaFree(dQ); cudaFree(dW); cudaFree(dQp); cudaFree(dV); cudaFree(dTr); cudaFree(dC); cudaFree(s_ref); cudaFree(s_tc); cudaFree(s_full);
    cudaFree(dKptrs);
    for (int c = 0; c < ncache; ++c) { cudaFree(dK[c]); cudaFree(dKp[c]); }
  }
  // Speed sample: production shape (Hi 32, kb 2048, bs 8 = 16,384 candidates), M=2048, T=20000
  {
    const int M = 2048, T = 20000, Hi = 32, Di = kvp::IDX_D, R = kvp::IDX_ROW, bs = 8, kb = 2048, ncand = kb * bs;
    const int nblocks = (T + bs - 1) / bs;
    std::mt19937 rng(77);
    std::normal_distribution<float> nd;
    std::vector<uint16_t> Q((size_t)M * Hi * Di), K((size_t)T * Di), W((size_t)M * Hi);
    for (auto& v : Q) v = f2bits(nd(rng) * 2.f);
    for (auto& v : K) v = f2bits(nd(rng));
    for (auto& v : W) v = f2bits(nd(rng) * 0.02f);
    std::vector<int32_t> vis(M, T), cand((size_t)M * kb);
    for (int m = 0; m < M; ++m)
      for (int j = 0; j < kb; ++j) cand[(size_t)m * kb + j] = (int)(((size_t)j * nblocks) / kb);  // ascending, evenly spread
    bf16 *dQ, *dK, *dW, *s_ref, *s_tc; uint8_t *dQp, *dKp; int32_t *dV, *dC;
    CUDA_CHECK(cudaMalloc(&dQ, Q.size() * 2)); CUDA_CHECK(cudaMalloc(&dK, K.size() * 2)); CUDA_CHECK(cudaMalloc(&dW, W.size() * 2));
    CUDA_CHECK(cudaMalloc(&dQp, (size_t)M * Hi * R)); CUDA_CHECK(cudaMalloc(&dKp, (size_t)T * R)); CUDA_CHECK(cudaMalloc(&dV, M * 4));
    CUDA_CHECK(cudaMalloc(&dC, cand.size() * 4)); CUDA_CHECK(cudaMalloc(&s_ref, (size_t)M * ncand * 2)); CUDA_CHECK(cudaMalloc(&s_tc, (size_t)M * ncand * 2));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dV, vis.data(), M * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, cand.data(), cand.size() * 4, cudaMemcpyHostToDevice));
    k::fp4_pack(dK, T, Di, 32, false, dKp, R, 0);
    k::fp4_pack(dQ, M * Hi, Di, 32, false, dQp, R, 0);
    k::fp4_quant_roundtrip(dQ, M, Hi * Di, 32, false, 0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int rep = 0; rep < 2; ++rep) {
      cudaEventRecord(e0); k::indexer_scores_cand(dQ, dKp, nullptr, nullptr, dW, M, Hi, Di, dC, kb, kb, bs, dV, s_ref, 0); cudaEventRecord(e1);
      CUDA_CHECK(cudaDeviceSynchronize()); float ms_ref; cudaEventElapsedTime(&ms_ref, e0, e1);
      cudaEventRecord(e0); k::indexer_scores_cand_tc(dQp, dKp, nullptr, nullptr, dW, M, Hi, dC, kb, kb, bs, dV, s_tc, 0); cudaEventRecord(e1);
      CUDA_CHECK(cudaDeviceSynchronize()); float ms_tc; cudaEventElapsedTime(&ms_tc, e0, e1);
      if (rep == 1) printf("speed M=2048 candidates 16384 (one layer): cuda-core %.2f ms · tc %.2f ms · ×%.1f\n", ms_ref, ms_tc, ms_ref / ms_tc);
    }
    cudaFree(dQ); cudaFree(dK); cudaFree(dW); cudaFree(dQp); cudaFree(dKp); cudaFree(dV); cudaFree(dC); cudaFree(s_ref); cudaFree(s_tc);
  }
  printf(all_ok ? "ALL PASS\n" : "SOME FAIL\n");
  return all_ok ? 0 : 1;
}
