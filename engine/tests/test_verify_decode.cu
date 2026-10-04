// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Kernel checks for HIVE_MTP_VERIFY2 / HIVE_MTP_BATCH (hive/verify_decode.h): do the three kernels changed to run verify rows on the decode path give **the same values** as the original kernels?
//   Shapes (the checkpoint's config.json): sliding_window 128 · head_dim 512 · compress_ratios ∈ {1, 2} (+ 4·8 to check the kernel limits) · verify rows M 1..8 · parts (sequences) 1..4.
//   ① window_idxs_verify vs window_idxs (prefill — the original verify path, start_pos = first position of the part, min_src 0): per part the window columns of rows [row0, row0+M_p)
//      must match (chunk columns ≥ win are shifted by the part's first row row0). A one-row part must also match window_idxs_rows (normal decode).
//      Positions: 0 · 1 · 5 · 126 · 127 · 128 · 129 · 255 · 4095 · 100000 (ring wrap and negative-position boundaries).
//   ② compressor_step_seq vs compressor_step_rows once per row (= M one-row decode steps — continuing the same state): byte comparison of out and valid for rows that complete a group,
//      and of the final state (kv·score). Also with several parts (row order mixing different state pointers). Completed-group outputs are also compared byte for byte with compressor_pool
//      (the prefill compressor of the original verify path; same formula, same order).
//   ③ compressor_apply_rows: re-applying n accepted rows (1..M) to a snapshot state == processing just those n rows sequentially with compressor_step_rows (bytes).
//   ④ timing (µs per launch, average of 1000 without graphs): window_idxs_verify vs window_idxs_rows · compressor_step_seq vs compressor_step_rows.
//   ⑤ HIVE_MTP_VERIFY2_FUSED: the whole verify front end — the unfused chain (original attention_verify_dev) vs the verify mode of the fused decode front end (HIVE_DECODE_ATTN_FUSED, incl. QKV3/SPARSE3 and IDXSCORE2/TOPK2)
//      on real layer shapes · part layouts · 6 layer kinds — byte comparison of outputs and state + immediate/graph time (µs) of one front end. See the header comment of namespace front below.
//   Verdict: all bytes equal = ALL PASS (exit code 0). Run: scripts/hive-run.sh "./build-dev/test_verify_decode [ctx,ctx,…]"   (⑤ contexts — default 4096,32768)
//   Whole-model comparison (layer chain, logits, greedy acceptance, rollback, timing) is in the CLI: HIVE_MTP_VERIFY2=1 hive ... --verify2-test (hive_main.cpp).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hive/attn_decode_fused.h"  // ⑤
#include "hive/common.h"
#include "hive/cublas_ops.h"
#include "hive/decode_attn2.h"
#include "hive/gemv_decode.h"
#include "hive/kernels.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "hive/verify_decode.h"
#include "hive/clock.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

constexpr int WIN = 128, D = 512;
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) { ++g_fail; fprintf(stderr, "  FAIL %s\n", what); }
}
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T))); CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T))); return p; }
template <class T> std::vector<T> host(const T* d, size_t n) { std::vector<T> v(n); CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
template <class T> void up(T* d, const std::vector<T>& v) { CUDA_CHECK(cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice)); }
uint32_t rng_state = 12345;
float frand(float a, float b) { rng_state = rng_state * 1664525u + 1013904223u; return a + (b - a) * (float)(rng_state >> 8) / 16777216.f; }

// ① window indices
void test_window() {
  const int stride = WIN + 512;
  const int64_t starts[] = {0, 1, 5, 126, 127, 128, 129, 255, 4095, 100000};
  int32_t *pos = dmalloc<int32_t>(8), *grp = dmalloc<int32_t>(8), *out = dmalloc<int32_t>((size_t)8 * stride), *ref = dmalloc<int32_t>((size_t)8 * stride);
  int cases = 0;
  // part layout: 1..4 parts, row total ≤ 8
  const std::vector<std::vector<int>> layouts = {{1}, {2}, {3}, {6}, {8}, {1, 1}, {2, 3}, {4, 4}, {1, 5}, {3, 1, 2}, {2, 2, 2, 2}, {1, 1, 1, 1}};
  for (const auto& lay : layouts)
    for (int si = 0; si < (int)(sizeof(starts) / sizeof(starts[0])); ++si) {
      std::vector<int32_t> ph, gh;
      std::vector<int64_t> p0s;
      int r = 0;
      for (size_t pi = 0; pi < lay.size(); ++pi) {
        const int64_t p0 = starts[(si + 3 * pi) % (sizeof(starts) / sizeof(starts[0]))];
        p0s.push_back(p0);
        for (int i = 0; i < lay[pi]; ++i) { ph.push_back((int32_t)(p0 + i)); gh.push_back(r); }
        r += lay[pi];
      }
      const int M = r;
      up(pos, ph); up(grp, gh);
      CUDA_CHECK(cudaMemset(out, 0x7f, (size_t)8 * stride * 4));
      k::window_idxs_verify(M, WIN, pos, grp, out, stride, 0);
      const std::vector<int32_t> got = host(out, (size_t)M * stride);
      bool ok = true;
      int row0 = 0;
      for (size_t pi = 0; pi < lay.size(); ++pi) {
        k::window_idxs(lay[pi], WIN, p0s[pi], ref, stride, 0, 0);
        const std::vector<int32_t> want = host(ref, (size_t)lay[pi] * stride);
        for (int i = 0; i < lay[pi]; ++i)
          for (int j = 0; j < WIN; ++j) {
            int32_t w = want[(size_t)i * stride + j];
            if (w >= WIN) w += row0;  // chunk column = kv row of this call (shifted by the part's first row)
            if (got[(size_t)(row0 + i) * stride + j] != w) ok = false;
          }
        if (lay[pi] == 1) {  // one-row part = normal decode row (self of window_idxs_rows = win + m)
          k::window_idxs_rows(M, WIN, pos, ref, stride, 0);
          const std::vector<int32_t> dec = host(ref, (size_t)M * stride);
          for (int j = 0; j < WIN; ++j) if (got[(size_t)row0 * stride + j] != dec[(size_t)row0 * stride + j]) ok = false;
        }
        row0 += lay[pi];
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      char buf[96]; snprintf(buf, sizeof buf, "window layout %zu parts M=%d start %lld", lay.size(), M, (long long)starts[si]);
      check(ok, buf);
      ++cases;
    }
  printf("① window_idxs_verify == window_idxs (per part) / window_idxs_rows (one-row part): %d cases\n", cases);
  for (void* p : {(void*)pos, (void*)grp, (void*)out, (void*)ref}) CUDA_CHECK(cudaFree(p));
}

// ② ③ compressor
struct Seqs {  // per part (sequence): state [ratio, D] kv·score
  std::vector<float*> skv, ssc;
  void alloc(int n, int ratio) { for (int i = 0; i < n; ++i) { skv.push_back(dmalloc<float>((size_t)ratio * D)); ssc.push_back(dmalloc<float>((size_t)ratio * D)); } }
  void free_all() { for (float* p : skv) CUDA_CHECK(cudaFree(p)); for (float* p : ssc) CUDA_CHECK(cudaFree(p)); skv.clear(); ssc.clear(); }
};
void test_compressor() {
  int cases = 0;
  const std::vector<std::vector<int>> layouts = {{1}, {2}, {3}, {5}, {6}, {8}, {2, 3}, {1, 1}, {4, 4}, {3, 1, 2}, {2, 2, 2, 2}};
  float *kv = dmalloc<float>((size_t)8 * D), *sc = dmalloc<float>((size_t)8 * D), *out_a = dmalloc<float>((size_t)8 * D), *out_b = dmalloc<float>((size_t)8 * D);
  float *pout = dmalloc<float>((size_t)8 * D);
  uint8_t *val_a = dmalloc<uint8_t>(8), *val_b = dmalloc<uint8_t>(8);
  int32_t* pos = dmalloc<int32_t>(8);
  int32_t* pos1 = dmalloc<int32_t>(1);
  float** pkv = dmalloc<float*>(8); float** psc = dmalloc<float*>(8);
  float** pkv1 = dmalloc<float*>(1); float** psc1 = dmalloc<float*>(1);
  for (int ratio : {2, 4, 8, 1})
    for (const auto& lay : layouts)
      for (int64_t base : {0, 1, 3, 127, 4096}) {
        const int P = (int)lay.size();
        Seqs A, B, S;  // A = new (one launch) · B = original (one one-row launch per row) · S = snapshot (initial state)
        A.alloc(P, ratio); B.alloc(P, ratio); S.alloc(P, ratio);
        // initial state: random (partial groups left by previous steps) — the same values in all three copies
        for (int p = 0; p < P; ++p) {
          std::vector<float> a((size_t)ratio * D), b((size_t)ratio * D);
          for (auto& x : a) x = frand(-2, 2);
          for (auto& x : b) x = frand(-4, 4);
          up(A.skv[p], a); up(B.skv[p], a); up(S.skv[p], a);
          up(A.ssc[p], b); up(B.ssc[p], b); up(S.ssc[p], b);
        }
        std::vector<float> kh((size_t)8 * D), sh((size_t)8 * D);
        for (auto& x : kh) x = frand(-2, 2);
        for (auto& x : sh) x = frand(-6, 6);
        up(kv, kh); up(sc, sh);
        std::vector<int32_t> ph; std::vector<float*> pk, ps; std::vector<int> part_of, row0_of;
        int r = 0;
        for (int p = 0; p < P; ++p) {
          const int64_t p0 = base + 5 * p;  // different position per part (group boundaries at different places)
          row0_of.push_back(r);
          for (int i = 0; i < lay[p]; ++i, ++r) { ph.push_back((int32_t)(p0 + i)); pk.push_back(A.skv[p]); ps.push_back(A.ssc[p]); part_of.push_back(p); }
        }
        const int M = r;
        up(pos, ph); up(pkv, pk); up(psc, ps);
        CUDA_CHECK(cudaMemset(out_a, 0, (size_t)8 * D * 4)); CUDA_CHECK(cudaMemset(out_b, 0, (size_t)8 * D * 4));
        k::compressor_step_seq(kv, sc, M, D, ratio, pos, pkv, psc, out_a, val_a, 0);
        // original: one-row compressor_step_rows per row (same launch as a normal decode step)
        for (int m = 0; m < M; ++m) {
          up(pos1, std::vector<int32_t>{ph[m]});
          up(pkv1, std::vector<float*>{B.skv[part_of[m]]}); up(psc1, std::vector<float*>{B.ssc[part_of[m]]});
          k::compressor_step_rows(kv + (size_t)m * D, sc + (size_t)m * D, 1, D, ratio, pos1, pkv1, psc1, out_b + (size_t)m * D, val_b + m, 0);
          CUDA_CHECK(cudaDeviceSynchronize());
        }
        const auto va = host(val_a, M), vb = host(val_b, M);
        const auto oa = host(out_a, (size_t)M * D), ob = host(out_b, (size_t)M * D);
        bool ok = va == vb;
        for (int m = 0; m < M; ++m) if (va[m] && memcmp(&oa[(size_t)m * D], &ob[(size_t)m * D], (size_t)D * 4) != 0) ok = false;
        for (int p = 0; p < P; ++p) {
          if (host(A.skv[p], (size_t)ratio * D) != host(B.skv[p], (size_t)ratio * D)) ok = false;
          if (host(A.ssc[p], (size_t)ratio * D) != host(B.ssc[p], (size_t)ratio * D)) ok = false;
        }
        char buf[128]; snprintf(buf, sizeof buf, "compressor_step_seq ratio %d parts %d M=%d base %lld", ratio, P, M, (long long)base);
        check(ok, buf);
        // original verify path (compressor_pool, per part, from the snapshot state) — byte comparison of completed-group outputs
        bool okp = true;
        for (int p = 0; p < P; ++p) {
          Seqs T; T.alloc(1, ratio);
          CUDA_CHECK(cudaMemcpy(T.skv[0], S.skv[p], (size_t)ratio * D * 4, cudaMemcpyDeviceToDevice));
          CUDA_CHECK(cudaMemcpy(T.ssc[0], S.ssc[p], (size_t)ratio * D * 4, cudaMemcpyDeviceToDevice));
          const int64_t p0 = ph[row0_of[p]];
          // compressor_pool reads only "the partial group before the start position" from the state (slots r < p0 % ratio) — the same slots as the decode state
          k::compressor_pool(kv + (size_t)row0_of[p] * D, sc + (size_t)row0_of[p] * D, lay[p], D, ratio, p0, T.skv[0], T.ssc[0], pout, 0);
          CUDA_CHECK(cudaDeviceSynchronize());
          const int64_t g0 = p0 / ratio, G = (p0 + lay[p]) / ratio - g0;
          const auto po = host(pout, (size_t)std::max<int64_t>(G, 1) * D);
          for (int64_t g = 0; g < G; ++g) {
            const int64_t last = (g0 + g) * ratio + ratio - 1;  // position that completes the group → its row
            const int m = row0_of[p] + (int)(last - p0);
            if (m < row0_of[p] || m >= row0_of[p] + lay[p] || !va[m] || memcmp(&po[(size_t)g * D], &oa[(size_t)m * D], (size_t)D * 4) != 0) okp = false;
          }
          T.free_all();
        }
        snprintf(buf, sizeof buf, "compressor_step_seq == compressor_pool groups ratio %d parts %d M=%d base %lld", ratio, P, M, (long long)base);
        check(okp, buf);
        // ③ n accepted: snapshot + compressor_apply_rows == n rows of sequential compressor_step_rows
        bool oka = true;
        for (int p = 0; p < P; ++p)
          for (int n = 1; n <= lay[p]; ++n) {
            Seqs X, Y; X.alloc(1, ratio); Y.alloc(1, ratio);
            for (Seqs* Z : {&X, &Y}) {
              CUDA_CHECK(cudaMemcpy(Z->skv[0], S.skv[p], (size_t)ratio * D * 4, cudaMemcpyDeviceToDevice));
              CUDA_CHECK(cudaMemcpy(Z->ssc[0], S.ssc[p], (size_t)ratio * D * 4, cudaMemcpyDeviceToDevice));
            }
            const int64_t p0 = ph[row0_of[p]];
            k::compressor_apply_rows(kv + (size_t)row0_of[p] * D, sc + (size_t)row0_of[p] * D, n, D, ratio, p0, X.skv[0], X.ssc[0], 0);
            for (int i = 0; i < n; ++i) {
              up(pos1, std::vector<int32_t>{(int32_t)(p0 + i)});
              up(pkv1, std::vector<float*>{Y.skv[0]}); up(psc1, std::vector<float*>{Y.ssc[0]});
              k::compressor_step_rows(kv + (size_t)(row0_of[p] + i) * D, sc + (size_t)(row0_of[p] + i) * D, 1, D, ratio, pos1, pkv1, psc1, out_b, val_b, 0);
              CUDA_CHECK(cudaDeviceSynchronize());
            }
            if (host(X.skv[0], (size_t)ratio * D) != host(Y.skv[0], (size_t)ratio * D) || host(X.ssc[0], (size_t)ratio * D) != host(Y.ssc[0], (size_t)ratio * D)) oka = false;
            X.free_all(); Y.free_all();
          }
        snprintf(buf, sizeof buf, "compressor_apply_rows == sequential ratio %d parts %d base %lld", ratio, P, (long long)base);
        check(oka, buf);
        A.free_all(); B.free_all(); S.free_all();
        ++cases;
      }
  printf("②③ compressor_step_seq == per-row compressor_step_rows (== compressor_pool complete groups) · compressor_apply_rows == sequential state: %d cases\n", cases);
  for (void* p : {(void*)kv, (void*)sc, (void*)out_a, (void*)out_b, (void*)pout, (void*)val_a, (void*)val_b, (void*)pos, (void*)pos1, (void*)pkv, (void*)psc,
                  (void*)pkv1, (void*)psc1})
    CUDA_CHECK(cudaFree(p));
}

// ⑤ HIVE_MTP_VERIFY_ROWIND: accum_f32_rows_ordered == values summed on the host in r order (bytes) · two runs give the same bytes (deterministic)
void test_accum_ordered() {
  const int dim = 4096, M = 8;
  int cases = 0;
  for (int R : {1, 3, 8, 17, 48}) {
    std::vector<float> src((size_t)R * dim), acc0((size_t)M * dim);
    std::vector<int32_t> rows(R);
    for (auto& x : src) x = frand(-1, 1);
    for (auto& x : acc0) x = frand(-3, 3);
    for (int r = 0; r < R; ++r) rows[r] = (int32_t)((r * 5 + R) % M);
    std::vector<float> want = acc0;
    for (int r = 0; r < R; ++r) for (int d = 0; d < dim; ++d) want[(size_t)rows[r] * dim + d] += src[(size_t)r * dim + d];
    float *s = dmalloc<float>(src.size()), *a = dmalloc<float>(acc0.size());
    int32_t* rw = dmalloc<int32_t>(R);
    up(s, src); up(rw, rows);
    bool ok = true;
    for (int rep = 0; rep < 2; ++rep) {
      up(a, acc0);
      k::accum_f32_rows_ordered(s, rw, R, dim, a, 0);
      const auto got = host(a, acc0.size());
      if (memcmp(got.data(), want.data(), want.size() * 4) != 0) ok = false;
    }
    char buf[64]; snprintf(buf, sizeof buf, "accum_f32_rows_ordered R=%d", R);
    check(ok, buf);
    ++cases;
    for (void* p : {(void*)s, (void*)a, (void*)rw}) CUDA_CHECK(cudaFree(p));
  }
  printf("⑤ accum_f32_rows_ordered == fixed-order host sum (twice · bytes): %d cases\n", cases);
}

// ④ timing
template <class F> double us_per(F f, int reps = 1000) {
  for (int i = 0; i < 20; ++i) f();
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto t0 = hive::SteadyClock::now();
  for (int i = 0; i < reps; ++i) f();
  CUDA_CHECK(cudaDeviceSynchronize());
  return std::chrono::duration<double, std::micro>(hive::SteadyClock::now() - t0).count() / reps;
}
void timing() {
  const int stride = WIN + 512;
  int32_t *pos = dmalloc<int32_t>(8), *grp = dmalloc<int32_t>(8), *out = dmalloc<int32_t>((size_t)8 * stride);
  float *kv = dmalloc<float>((size_t)8 * D), *sc = dmalloc<float>((size_t)8 * D), *o = dmalloc<float>((size_t)8 * D);
  uint8_t* val = dmalloc<uint8_t>(8);
  Seqs S; S.alloc(8, 2);
  float** pk = dmalloc<float*>(8); float** ps = dmalloc<float*>(8);
  std::vector<float*> kk, ss;
  for (int m = 0; m < 8; ++m) { kk.push_back(S.skv[m]); ss.push_back(S.ssc[m]); }
  up(pk, kk); up(ps, ss);
  for (int M : {1, 4, 6, 8}) {
    std::vector<int32_t> ph(M), gh(M, 0);
    for (int m = 0; m < M; ++m) ph[m] = 5000 + m;
    up(pos, ph); up(grp, gh);
    const double a = us_per([&] { k::window_idxs_verify(M, WIN, pos, grp, out, stride, 0); });
    const double b = us_per([&] { k::window_idxs_rows(M, WIN, pos, out, stride, 0); });
    const double c = us_per([&] { k::compressor_step_seq(kv, sc, M, D, 2, pos, pk, ps, o, val, 0); });
    const double d = us_per([&] { k::compressor_step_rows(kv, sc, M, D, 2, pos, pk, ps, o, val, 0); });  // (different state per row — normal decode batch)
    printf("④ M=%d: window_idxs_verify %.2f µs vs window_idxs_rows %.2f · compressor_step_seq %.2f µs vs compressor_step_rows %.2f (per launch, no graph)\n", M, a, b, c, d);
  }
  S.free_all();
  for (void* p : {(void*)pos, (void*)grp, (void*)out, (void*)kv, (void*)sc, (void*)o, (void*)val, (void*)pk, (void*)ps}) CUDA_CHECK(cudaFree(p));
}


// ⑤ HIVE_MTP_VERIFY2_FUSED — verify front end: the unfused chain (the !front_done branch of runtime attention_verify_dev — wo_a·wo_b are common to both and left out)
//   vs the verify mode of the fused decode front end (k::dec_attention_front, DecAttnArgs::vgrp). Real layer shapes (dim 5120 · H 64 · D 512 · q_lora 1280 · indexer 32×128 · top 512 ·
//   win 128) × contexts (default 4096·32768) × part layouts (consecutive positions of the same sequence, 1..4 parts, row total 2..8) × 6 layer kinds (W·C·K2·K20·U·Utc — as in test_decode_attn)
//   × 5 variants (v3 bits 0·1·2·3 = QKV3/SPARSE3 with lctx 0, plus v3 3 + lctx 3 = including IDXSCORE2/TOPK2).
//   Compared (bits): qr·qrn·q·qrq·qrs·kv·iq·idx (window + top-k columns)·o·ring (whole ring per part — the ring is written after attention)·compressed cache·index keys (row range a part writes)·compressor state·
//   valid·cand·iqp·iw·the rollback input copy (first M rows of vckv, kv ‖ score)·have_cand. Timing: one front end, immediate and graph, average of 20 (µs) — the new variant takes v3/lctx from the environment (-1).
namespace front {

constexpr int DIM = 5120, H = 64, RD = 64, QL = 1280, HI = 32, DI = 128, TOPK = 512, BS = 8, CB = 2048, SPLITS = 4, VROWS = 8;
constexpr float EPS = 1e-20f;
constexpr int CAND_SRC = 20;

__device__ __forceinline__ uint32_t hsh(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
__device__ __forceinline__ float u01(uint32_t x) { return (hsh(x) >> 8) * (1.0f / 16777216.0f); }
__global__ void fill_e4m3(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) {
    const uint32_t h = hsh((uint32_t)i * 2654435761u ^ seed);
    p[i] = (uint8_t)(((h >> 8) & 0x80u) | ((4u + (h % 6u)) << 3) | ((h >> 4) & 7u));
  }
}
__global__ void fill_u8c(uint8_t* p, size_t n, uint8_t lo, uint8_t span, uint32_t seed) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) p[i] = (uint8_t)(lo + hsh((uint32_t)i ^ seed) % span);
}
__global__ void fill_f32(float* p, size_t n, float a, float b, uint32_t seed) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) p[i] = a + (b - a) * u01((uint32_t)i ^ seed);
}
__global__ void fill_bf16(bf16* p, size_t n, float a, float b, uint32_t seed) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) p[i] = f2bf(a + (b - a) * u01((uint32_t)i ^ seed));
}
__global__ void fill_comp(uint8_t* p, size_t rows, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * kvp::COMP_ROW; i += (size_t)gridDim.x * blockDim.x) {
    const int j = (int)(i % kvp::COMP_ROW);
    const uint32_t h = hsh((uint32_t)i ^ seed);
    p[i] = j < kvp::COMP_SCALES ? (uint8_t)(0x20 + h % 24) : (uint8_t)h;
  }
}
__global__ void fill_idxk(uint8_t* p, size_t rows, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * kvp::IDX_ROW; i += (size_t)gridDim.x * blockDim.x) {
    const int j = (int)(i % kvp::IDX_ROW);
    const uint32_t h = hsh((uint32_t)i ^ seed);
    p[i] = j < kvp::IDX_SCALES ? (uint8_t)(120 + h % 7) : j < kvp::IDX_HDR ? (uint8_t)0 : (uint8_t)h;
  }
}
__global__ void fill_rope(float2* f, int npos, int half, float theta) {
  const size_t n = (size_t)npos * half;
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) {
    const int p = (int)(i / half), k = (int)(i % half);
    const float ang = (float)p * powf(theta, -2.0f * k / (2.0f * half));
    float s, c;
    sincosf(fmodf(ang, 6.2831853f), &s, &c);
    f[i] = make_float2(c, s);
  }
}
int gridn(size_t n) { return (int)std::max<size_t>(1, std::min<size_t>(4096, (n + 255) / 256)); }
template <class T> T* dalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc(&p, n * sizeof(T) + 64)); CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T) + 64)); return p; }
template <class T> T* mapped(size_t n, T** dev) {
  T* p; CUDA_CHECK(cudaHostAlloc((void**)&p, n * sizeof(T), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void**)dev, p, 0)); memset(p, 0, n * sizeof(T));
  return p;
}
template <class T> std::vector<uint8_t> snap(const T* d, size_t n) { std::vector<uint8_t> v(n * sizeof(T)); CUDA_CHECK(cudaMemcpy(v.data(), d, v.size(), cudaMemcpyDeviceToHost)); return v; }
void put(void* d, const std::vector<uint8_t>& v) { if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice)); }

struct Fp8 { uint8_t* w; uint8_t* s; };
Fp8 make_fp8(int N, int K, uint32_t seed) {
  Fp8 f{dalloc<uint8_t>((size_t)N * K), dalloc<uint8_t>((size_t)N * (K / 32))};
  fill_e4m3<<<gridn((size_t)N * K), 256>>>(f.w, (size_t)N * K, seed);
  fill_u8c<<<gridn((size_t)N * K / 32), 256>>>(f.s, (size_t)N * (K / 32), 117, 4, seed * 7 + 1);
  return f;
}
bf16* make_bf16(size_t n, float a, float b, uint32_t seed) { bf16* p = dalloc<bf16>(n); fill_bf16<<<gridn(n), 256>>>(p, n, a, b, seed); return p; }
float* make_f32(size_t n, float a, float b, uint32_t seed) { float* p = dalloc<float>(n); fill_f32<<<gridn(n), 256>>>(p, n, a, b, seed); return p; }

struct LayerW {
  Fp8 wqa, wqb, wkv, idx_wqb;
  bf16 *q_norm, *kv_norm, *comp_norm, *idx_wproj, *idx_wk, *idx_k_norm, *comp_wkv_bf;
  float *sink, *comp_wkv_f, *comp_wgate;
};
LayerW make_layer(uint32_t seed) {
  LayerW L;
  L.wqa = make_fp8(QL, DIM, seed + 1); L.wqb = make_fp8(H * D, QL, seed + 2); L.wkv = make_fp8(D, DIM, seed + 3); L.idx_wqb = make_fp8(HI * DI, QL, seed + 4);
  L.q_norm = make_bf16(QL, 0.8f, 1.2f, seed + 5); L.kv_norm = make_bf16(D, 0.8f, 1.2f, seed + 6); L.comp_norm = make_bf16(D, 0.8f, 1.2f, seed + 7);
  L.idx_wproj = make_bf16((size_t)HI * DIM, -0.03f, 0.03f, seed + 8); L.idx_wk = make_bf16((size_t)DI * D, -0.08f, 0.08f, seed + 9);
  L.idx_k_norm = make_bf16(DI, 0.8f, 1.2f, seed + 10); L.comp_wkv_bf = make_bf16((size_t)D * DIM, -0.02f, 0.02f, seed + 11);
  L.sink = make_f32(H, -0.5f, 1.0f, seed + 12); L.comp_wkv_f = make_f32((size_t)D * DIM, -0.02f, 0.02f, seed + 13);
  L.comp_wgate = make_f32((size_t)D * DIM, -0.02f, 0.02f, seed + 14);
  return L;
}
struct SeqState { bf16* ring; uint8_t* comp; uint8_t* idxk; float* skv; float* ssc; };  // state of one part (sequence)
struct Work {
  int Tcap = 0;
  bf16 *xn, *qr, *qrn, *q, *kv, *latent, *ik, *iq, *iw, *iscore, *o;
  float *xf, *ckv, *cscore, *cout, *iscore_f, *bmax, *pacc, *pm, *ps, *vckv;
  uint8_t *qrq, *qrs, *valid, *iqp;
  int32_t *idx, *topk_pos, *cand;
  int *counters, *attn_cnt;
  k::DecRow* tab;
  k::DecSoA* soa;
  k::KvRow *kvrows_h, *kvrows_d;
  const uint8_t **kptrs_h, **kptrs_d;
  int32_t *trows_h, *trows_d, *dsti_h, *dsti_d, *pos_h, *pos_d, *gpos_h, *gpos_d, *visible_h, *visible_d, *vgrp_h, *vgrp_d;
  bf16 **ringp_h, **ringp_d;
  uint8_t **dstp_h, **dstp_d, **dstk_h, **dstk_d;
  float **skv_h, **skv_d, **ssc_h, **ssc_d;
};
void make_work(Work& w, int Tcap) {
  const int M = VROWS;
  w.Tcap = Tcap;
  w.xn = dalloc<bf16>((size_t)M * DIM); w.qr = dalloc<bf16>((size_t)M * QL); w.qrn = dalloc<bf16>((size_t)M * QL); w.q = dalloc<bf16>((size_t)M * H * D);
  w.kv = dalloc<bf16>((size_t)M * D); w.latent = dalloc<bf16>((size_t)M * D); w.ik = dalloc<bf16>((size_t)M * DI); w.iq = dalloc<bf16>((size_t)M * HI * DI);
  w.iw = dalloc<bf16>((size_t)M * HI); w.iscore = dalloc<bf16>((size_t)M * (Tcap + 8)); w.o = dalloc<bf16>((size_t)M * H * D);
  w.xf = dalloc<float>((size_t)M * DIM); w.ckv = dalloc<float>((size_t)M * D); w.cscore = dalloc<float>((size_t)M * D); w.cout = dalloc<float>((size_t)M * D);
  w.iscore_f = dalloc<float>((size_t)M * (Tcap + 8)); w.bmax = dalloc<float>((size_t)M * ((Tcap + BS - 1) / BS));
  w.pacc = dalloc<float>((size_t)M * H * 16 * D); w.pm = dalloc<float>((size_t)M * H * 16); w.ps = dalloc<float>((size_t)M * H * 16);
  w.vckv = dalloc<float>((size_t)2 * VROWS * D);
  w.qrq = dalloc<uint8_t>((size_t)M * QL); w.qrs = dalloc<uint8_t>((size_t)M * QL / 32); w.valid = dalloc<uint8_t>(M);
  w.idx = dalloc<int32_t>((size_t)M * (WIN + TOPK)); w.topk_pos = dalloc<int32_t>((size_t)M * std::max(TOPK, CB)); w.cand = dalloc<int32_t>((size_t)M * CB);
  w.counters = dalloc<int>(16); w.attn_cnt = dalloc<int>((size_t)M * H); w.tab = dalloc<k::DecRow>(M); w.soa = dalloc<k::DecSoA>(1);
  w.iqp = dalloc<uint8_t>((size_t)M * HI * kvp::IDX_ROW);
  w.kvrows_h = mapped<k::KvRow>(M, &w.kvrows_d); w.kptrs_h = mapped<const uint8_t*>(M, &w.kptrs_d);
  w.trows_h = mapped<int32_t>(M, &w.trows_d); w.dsti_h = mapped<int32_t>(M, &w.dsti_d); w.pos_h = mapped<int32_t>(M, &w.pos_d);
  w.gpos_h = mapped<int32_t>(M, &w.gpos_d); w.visible_h = mapped<int32_t>(M, &w.visible_d); w.vgrp_h = mapped<int32_t>(M, &w.vgrp_d);
  w.ringp_h = mapped<bf16*>(M, &w.ringp_d);
  w.dstp_h = mapped<uint8_t*>(M, &w.dstp_d); w.dstk_h = mapped<uint8_t*>(M, &w.dstk_d); w.skv_h = mapped<float*>(M, &w.skv_d); w.ssc_h = mapped<float*>(M, &w.ssc_d);
}
struct Case { const char* name; int layer, ratio; bool kv_source, index_source, idx_tc; };

// same tables as runtime attention_verify_host (row = verify row · sequence = part). Returns Tmax (index layers)
int fill_tables(Work& w, const Case& cs, const std::vector<SeqState>& seqs, const std::vector<int>& part, const std::vector<int>& pos, int M,
                std::vector<int>& shared_topk) {
  for (int m = 0; m < M; ++m) {
    const SeqState& s = seqs[part[m]];
    w.kvrows_h[m].ring = s.ring; w.kvrows_h[m].comp = nullptr; w.kvrows_h[m].comp_len = 0; w.kvrows_h[m].topk = 0;
    w.ringp_h[m] = s.ring;
    w.pos_h[m] = pos[m];
  }
  if (!cs.ratio) return 0;
  const int ratio = cs.ratio;
  for (int m = 0; m < M; ++m) {
    const SeqState& s = seqs[part[m]];
    const int p = pos[m];
    w.kvrows_h[m].comp = s.comp; w.kvrows_h[m].comp_len = (p + 1) / ratio;
    w.kptrs_h[m] = s.idxk; w.trows_h[m] = w.kvrows_h[m].comp_len; w.visible_h[m] = (p + 1) / ratio;
  }
  if (cs.kv_source)
    for (int m = 0; m < M; ++m) {
      const SeqState& s = seqs[part[m]];
      const int p = pos[m];
      w.gpos_h[m] = p + 1 - ratio; w.dsti_h[m] = p / ratio; w.dstp_h[m] = s.comp;
      if (ratio > 1) { w.skv_h[m] = s.skv; w.ssc_h[m] = s.ssc; }
      if (cs.index_source) w.dstk_h[m] = s.idxk;
    }
  int Tmax = 0;
  if (cs.index_source) {
    for (int m = 0; m < M; ++m) Tmax = std::max(Tmax, w.trows_h[m]);
    for (int m = 0; m < M; ++m) shared_topk[m] = std::min(TOPK, w.trows_h[m]);
  }
  for (int m = 0; m < M; ++m) w.kvrows_h[m].topk = shared_topk[m];
  return Tmax;
}

// ---- unfused chain: the !front_done branch of runtime attention_verify_dev (HIVE_ATTN_TC · HIVE_IDX_F32 off) verbatim — up to before wo + ring write ----------------------------
void r1_front(const Case& cs, const LayerW& A, Work& w, Blas& blas, int M, int Tmax, bool& have_cand, const float2* fr_win, const float2* fr_cmp, cudaStream_t st) {
  const int dim = DIM, win = WIN, ratio = cs.ratio, rd = RD;
  const float2* freqs = ratio ? fr_cmp : fr_win;
  (k::decode_attn2_on() ? k::attn2_q_proj : k::decode_gemv2_on() ? k::gemv2_q_proj : k::fused_q_proj)(w.xn, dim, A.wqa.w, A.wqa.s, QL, A.q_norm, EPS, A.wqb.w,
      A.wqb.s, H, D, rd, freqs, w.pos_d, M, w.qr, w.qrn, w.q, w.qrq, w.qrs, w.counters + 0, st);
  k::fused_kv_proj(w.xn, dim, A.wkv.w, A.wkv.s, D, A.kv_norm, EPS, rd, freqs, w.pos_d, M, w.kv, nullptr, win, nullptr, 0, w.counters + 1, st);
  const int idx_stride = win + TOPK;
  k::window_idxs_verify(M, win, w.pos_d, w.vgrp_d, w.idx, idx_stride, st);
  const int topk_max_cols = ratio ? TOPK : 0;
  if (ratio) {
    if (cs.kv_source) {
      if (ratio > 1) {
        k::bf16_to_f32(w.xn, M * dim, w.xf, st);
        blas.gemm_f32(w.xf, A.comp_wkv_f, w.ckv, M, D, dim);
        blas.gemm_f32(w.xf, A.comp_wgate, w.cscore, M, D, dim);
        CUDA_CHECK(cudaMemcpyAsync(w.vckv, w.ckv, (size_t)M * D * 4, cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(w.vckv + (size_t)VROWS * D, w.cscore, (size_t)M * D * 4, cudaMemcpyDeviceToDevice, st));
        k::compressor_step_seq(w.ckv, w.cscore, M, D, ratio, w.pos_d, w.skv_d, w.ssc_d, w.cout, w.valid, st);
        k::f32_to_bf16(w.cout, M * D, w.latent, st);
      } else {
        blas.gemm_bf16(w.xn, A.comp_wkv_bf, w.latent, M, D, dim);
        CUDA_CHECK(cudaMemsetAsync(w.valid, 1, (size_t)M, st));
      }
      k::rmsnorm(w.latent, A.comp_norm, EPS, M, D, w.latent, st);
      if (cs.index_source) {
        blas.gemm_bf16(w.latent, A.idx_wk, w.ik, M, DI, D);
        k::rmsnorm(w.ik, A.idx_k_norm, EPS, M, DI, w.ik, st);
        k::rope_last(w.ik, M, 1, DI, rd, freqs, w.gpos_d, false, st);
        k::fp4_pack_rows(w.ik, M, DI, 32, false, w.dstk_d, w.dsti_d, w.valid, kvp::IDX_ROW, st);
      }
      k::rope_last(w.latent, M, 1, D, rd, freqs, w.gpos_d, false, st);
      k::fp4_pack_rows(w.latent, M, D, 16, true, w.dstp_d, w.dsti_d, w.valid, kvp::COMP_ROW, st);
    }
    if (cs.index_source && Tmax > 0) {
      const bool pack_dec = cs.idx_tc && CAND_SRC >= 0 && CAND_SRC < cs.layer;
      k::gemm_bs(w.qrq, w.qrs, A.idx_wqb.w, A.idx_wqb.s, false, M, HI * DI, QL, w.iq, nullptr, st);
      k::rope_last(w.iq, M, HI, DI, rd, fr_cmp, w.pos_d, false, st);
      if (pack_dec) k::fp4_pack(w.iq, M * HI, DI, 32, false, w.iqp, kvp::IDX_ROW, st);
      k::fp4_quant_roundtrip(w.iq, M, HI * DI, 32, false, st);
      blas.gemm_bf16(w.xn, A.idx_wproj, w.iw, M, HI, dim);
      k::scale_bf16(w.iw, M * HI, (1.0f / sqrtf((float)DI)) * (1.0f / sqrtf((float)HI)), st);
      const bool cand_src = (cs.layer == CAND_SRC);
      const bool uses_cand = (CAND_SRC >= 0 && CAND_SRC < cs.layer);
      const int nblocks = (Tmax + BS - 1) / BS;
      const int topk = std::min(TOPK, Tmax);
      if (uses_cand) {
        const int kbc = std::min(CB, nblocks);
        HIVE_CHECK(have_cand && kbc > 0, "candidates missing");
        const int ncand = kbc * BS;
        if (pack_dec) k::indexer_scores_cand_tc(w.iqp, nullptr, w.kptrs_d, w.trows_d, w.iw, M, HI, w.cand, CB, kbc, BS, w.visible_d, w.iscore, st);
        else k::indexer_scores_cand(w.iq, nullptr, w.kptrs_d, w.trows_d, w.iw, M, HI, DI, w.cand, CB, kbc, BS, w.visible_d, w.iscore, st);
        k::bf16_rows_to_f32(w.iscore, M * ncand, w.iscore_f, st);
        k::topk_select_rows(w.iscore_f, M, ncand, topk, ncand, w.topk_pos, topk, st);
        k::cand_to_pos(w.topk_pos, M, topk, w.cand, CB, BS, st);
      } else {
        k::indexer_scores_rows(w.iq, w.kptrs_d, w.trows_d, Tmax, w.iw, M, HI, DI, w.visible_d, w.iscore, st);
        if (cand_src) {
          k::block_max(w.iscore, M, Tmax, BS, w.visible_d, w.bmax, st);
          const int kb = std::min(CB, nblocks);
          k::topk_select_rows(w.bmax, M, nblocks, kb, nblocks, w.topk_pos, kb, st);
          CUDA_CHECK(cudaMemcpy2DAsync(w.cand, (size_t)CB * 4, w.topk_pos, (size_t)kb * 4, (size_t)kb * 4, M, cudaMemcpyDeviceToDevice, st));
          have_cand = true;
        }
        k::bf16_rows_to_f32(w.iscore, M * Tmax, w.iscore_f, st);
        k::topk_select_rows_t(w.iscore_f, M, w.trows_d, Tmax, topk, Tmax, w.topk_pos, topk, st);
      }
      k::offset_idxs(w.topk_pos, M, topk, w.visible_d, win + M, w.idx + win, idx_stride, st);
    }
  }
  k::sparse_attn_decode_rows(w.q, M, H, D, w.kvrows_d, w.kv, M, win, w.idx, idx_stride, win + topk_max_cols, A.sink, 1.0f / sqrtf((float)D), w.o,
                             (win + topk_max_cols) > 256 ? SPLITS : 1, w.pacc, w.pm, w.ps, st, freqs, w.pos_d, rd);
  k::ring_write_rows(w.kv, M, D, win, w.pos_d, w.ringp_d, st);
}

// ---- fused front end in verify mode (same arguments as the front_done branch of runtime attention_verify_dev) ----------------------------------------------------------------
void q2_front(const Case& cs, const LayerW& A, Work& w, Blas& blas, int M, int Tmax, bool& have_cand, const float2* fr_win, const float2* fr_cmp, int v3, int lctx,
              cudaStream_t st) {
  k::DecAttnArgs a;
  a.M = M; a.dim = DIM; a.H = H; a.D = D; a.rd = RD; a.win = WIN; a.q_lora = QL; a.Hi = HI; a.Di = DI; a.index_topk = TOPK; a.ratio = cs.ratio; a.layer = cs.layer;
  a.cand_source_layer = CAND_SRC; a.cand_block = BS; a.cand_topk_blocks = CB; a.kv_source = cs.kv_source; a.index_source = cs.index_source;
  a.have_candidates = have_cand; a.Tmax = Tmax; a.eps = EPS; a.splits = (WIN + (cs.ratio ? TOPK : 0)) > 256 ? SPLITS : 1;
  a.wqa = A.wqa.w; a.sqa = A.wqa.s; a.wqb = A.wqb.w; a.sqb = A.wqb.s; a.wkv = A.wkv.w; a.skv = A.wkv.s; a.q_norm = A.q_norm; a.kv_norm = A.kv_norm; a.sink = A.sink;
  a.comp_wkv = cs.ratio > 1 ? (const void*)A.comp_wkv_f : (const void*)A.comp_wkv_bf; a.comp_wgate = A.comp_wgate; a.comp_norm = A.comp_norm;
  a.idx_wqb = A.idx_wqb.w; a.idx_sqb = A.idx_wqb.s; a.idx_wproj = A.idx_wproj; a.idx_wk = A.idx_wk; a.idx_k_norm = A.idx_k_norm;
  a.freqs = cs.ratio ? fr_cmp : fr_win; a.freqs_idx = fr_cmp;
  a.xn = w.xn; a.xf = w.xf; a.qr = w.qr; a.qrn = w.qrn; a.q = w.q; a.kv = w.kv; a.qrq = w.qrq; a.qrs = w.qrs; a.ckv = w.ckv; a.cscore = w.cscore; a.cout = w.cout;
  a.latent = w.latent; a.valid = w.valid; a.ik = w.ik; a.iq = w.iq; a.iw = w.iw; a.iscore = w.iscore; a.bmax = w.bmax; a.cand = w.cand; a.idx = w.idx;
  a.idx_stride = WIN + TOPK; a.o = w.o; a.pacc = w.pacc; a.pm = w.pm; a.ps = w.ps; a.counters = w.counters + 12; a.attn_cnt = w.attn_cnt; a.tab = w.tab; a.soa = w.soa;
  a.idx_mode = cs.idx_tc ? 1 : 0; a.iqp = w.iqp;
  a.src = k::DecRowSrc{w.kvrows_d, w.kptrs_d, w.dstp_d, w.dstk_d, w.skv_d, w.ssc_d, w.pos_d, w.gpos_d, w.visible_d, w.trows_d, w.dsti_d};
  a.ring_ptrs = w.ringp_d;
  a.scratch = w.iscore_f; a.scratch_bytes = (size_t)VROWS * (w.Tcap + 8) * 4;
  a.v3 = v3; a.lctx = lctx;
  a.vgrp = w.vgrp_d;
  if (cs.ratio > 1 && cs.kv_source) { a.vsave = w.vckv; a.vsave_rows = VROWS; }
  if (k::dec_attention_front(a, &blas, st)) have_cand = true;
}

const char* kOut[] = {"qr", "qrn", "q", "qrq", "qrs", "kv", "iq", "idx", "o", "ring", "comp", "idxk", "state", "valid", "cand", "iqp", "iw", "vckv"};
// cache row range [lo, hi) written by part p
void rows_of(const Case& cs, const std::vector<int>& pos, const std::vector<int>& part, int p, int& lo, int& hi) {
  lo = 1 << 30; hi = 0;
  for (size_t m = 0; m < pos.size(); ++m)
    if (part[m] == p) { const int di = cs.ratio ? pos[m] / cs.ratio : 0; lo = std::min(lo, di); hi = std::max(hi, di + 1); }
  if (lo > hi) lo = hi = 0;
}
std::vector<std::vector<uint8_t>> grab(const Case& cs, Work& w, const std::vector<SeqState>& seqs, const std::vector<int>& part, const std::vector<int>& pos, int M,
                                       int P) {
  std::vector<std::vector<uint8_t>> o;
  const bool idx_on = cs.index_source && cs.ratio;
  o.push_back(snap(w.qr, (size_t)M * QL)); o.push_back(snap(w.qrn, (size_t)M * QL)); o.push_back(snap(w.q, (size_t)M * H * D));
  o.push_back(snap(w.qrq, (size_t)M * QL)); o.push_back(snap(w.qrs, (size_t)M * QL / 32)); o.push_back(snap(w.kv, (size_t)M * D));
  o.push_back(idx_on ? snap(w.iq, (size_t)M * HI * DI) : std::vector<uint8_t>());
  o.push_back(snap(w.idx, (size_t)M * (WIN + TOPK))); o.push_back(snap(w.o, (size_t)M * H * D));
  std::vector<uint8_t> ring, comp, idxk, state;
  for (int p = 0; p < P; ++p) {
    auto r = snap(seqs[p].ring, (size_t)WIN * D); ring.insert(ring.end(), r.begin(), r.end());
    if (cs.ratio && cs.kv_source) {
      int lo, hi; rows_of(cs, pos, part, p, lo, hi);
      auto c = snap(seqs[p].comp + (size_t)lo * kvp::COMP_ROW, (size_t)(hi - lo) * kvp::COMP_ROW); comp.insert(comp.end(), c.begin(), c.end());
      if (cs.index_source) { auto x = snap(seqs[p].idxk + (size_t)lo * kvp::IDX_ROW, (size_t)(hi - lo) * kvp::IDX_ROW); idxk.insert(idxk.end(), x.begin(), x.end()); }
      if (cs.ratio > 1) {
        auto a = snap(seqs[p].skv, (size_t)cs.ratio * D), b = snap(seqs[p].ssc, (size_t)cs.ratio * D);
        state.insert(state.end(), a.begin(), a.end()); state.insert(state.end(), b.begin(), b.end());
      }
    }
  }
  o.push_back(ring); o.push_back(comp); o.push_back(idxk); o.push_back(state);
  o.push_back(cs.ratio && cs.kv_source ? snap(w.valid, M) : std::vector<uint8_t>());
  o.push_back(cs.layer == CAND_SRC ? snap(w.cand, (size_t)M * CB) : std::vector<uint8_t>());
  o.push_back(cs.idx_tc ? snap(w.iqp, (size_t)M * HI * kvp::IDX_ROW) : std::vector<uint8_t>());
  o.push_back(idx_on ? snap(w.iw, (size_t)M * HI) : std::vector<uint8_t>());
  if (cs.ratio > 1 && cs.kv_source) {
    auto a = snap(w.vckv, (size_t)M * D), b = snap(w.vckv + (size_t)VROWS * D, (size_t)M * D);
    a.insert(a.end(), b.begin(), b.end());
    o.push_back(a);
  } else o.push_back(std::vector<uint8_t>());
  return o;
}
struct Saved { std::vector<std::vector<uint8_t>> ring, comp, idxk, skv, ssc; std::vector<int> lo; std::vector<uint8_t> idx, cand, vckv; };
Saved save(const Case& cs, Work& w, const std::vector<SeqState>& seqs, const std::vector<int>& part, const std::vector<int>& pos, int P) {
  Saved s;
  for (int p = 0; p < P; ++p) {
    int lo, hi; rows_of(cs, pos, part, p, lo, hi);
    s.lo.push_back(lo);
    s.ring.push_back(snap(seqs[p].ring, (size_t)WIN * D));
    s.comp.push_back(snap(seqs[p].comp + (size_t)lo * kvp::COMP_ROW, (size_t)(hi - lo) * kvp::COMP_ROW));
    s.idxk.push_back(snap(seqs[p].idxk + (size_t)lo * kvp::IDX_ROW, (size_t)(hi - lo) * kvp::IDX_ROW));
    s.skv.push_back(snap(seqs[p].skv, (size_t)2 * D)); s.ssc.push_back(snap(seqs[p].ssc, (size_t)2 * D));
  }
  s.idx = snap(w.idx, (size_t)VROWS * (WIN + TOPK)); s.cand = snap(w.cand, (size_t)VROWS * CB); s.vckv = snap(w.vckv, (size_t)2 * VROWS * D);
  return s;
}
void restore(Work& w, const std::vector<SeqState>& seqs, int P, const Saved& s) {
  for (int p = 0; p < P; ++p) {
    put(seqs[p].ring, s.ring[p]);
    put(seqs[p].comp + (size_t)s.lo[p] * kvp::COMP_ROW, s.comp[p]); put(seqs[p].idxk + (size_t)s.lo[p] * kvp::IDX_ROW, s.idxk[p]);
    put(seqs[p].skv, s.skv[p]); put(seqs[p].ssc, s.ssc[p]);
  }
  put(w.idx, s.idx); put(w.cand, s.cand); put(w.vckv, s.vckv);
  CUDA_CHECK(cudaDeviceSynchronize());
}
template <class F> float time_eager(F&& f, cudaStream_t st, int iters) {
  for (int i = 0; i < 3; ++i) f();
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0, st));
  for (int i = 0; i < iters; ++i) f();
  CUDA_CHECK(cudaEventRecord(e1, st));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms * 1000.f / iters;
}
template <class F> float time_graph(F&& f, cudaStream_t st, int iters) {
  f();
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaGraph_t g; cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  f();
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaGraphLaunch(ge, st));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0, st));
  for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(ge, st));
  CUDA_CHECK(cudaEventRecord(e1, st));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  return ms * 1000.f / iters;
}
int next_pow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }

void run(const std::vector<int>& ctxs) {
  const int max_ctx = *std::max_element(ctxs.begin(), ctxs.end());
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));  // blocking stream (ordered with default-stream cudaMemcpy)
  Blas blas(st);
  const int npos = max_ctx + 16;
  float2 *fr_win = dalloc<float2>((size_t)npos * (RD / 2)), *fr_cmp = dalloc<float2>((size_t)npos * (RD / 2));
  fill_rope<<<4096, 256>>>(fr_win, npos, RD / 2, 10000.f);
  fill_rope<<<4096, 256>>>(fr_cmp, npos, RD / 2, 160000.f);
  constexpr int PMAX = 4;
  std::vector<SeqState> seqs(PMAX);
  const size_t crow = (size_t)max_ctx + 16;
  for (int p = 0; p < PMAX; ++p) {
    seqs[p].ring = make_bf16((size_t)WIN * D, -2.f, 2.f, 1000 + p);
    seqs[p].comp = dalloc<uint8_t>(crow * kvp::COMP_ROW); seqs[p].idxk = dalloc<uint8_t>(crow * kvp::IDX_ROW);
    fill_comp<<<gridn(crow * kvp::COMP_ROW), 256>>>(seqs[p].comp, crow, 2000 + p);
    fill_idxk<<<gridn(crow * kvp::IDX_ROW), 256>>>(seqs[p].idxk, crow, 3000 + p);
    seqs[p].skv = make_f32((size_t)2 * D, -1.f, 1.f, 4000 + p); seqs[p].ssc = make_f32((size_t)2 * D, -2.f, 2.f, 5000 + p);
  }
  Work w;
  make_work(w, max_ctx + 8);
  CUDA_CHECK(cudaMemset(w.cand, 0xFF, (size_t)VROWS * CB * 4));
  const Case cases[] = {{"W", 0, 0, false, false, false}, {"C", 22, 1, false, false, false}, {"K2", 2, 2, true, true, false}, {"K20", 20, 1, true, true, false},
                        {"U", 24, 1, false, true, false}, {"Utc", 28, 1, false, true, true}};
  std::vector<LayerW> LW;
  for (int i = 0; i < 6; ++i) LW.push_back(make_layer(100 * (i + 1)));
  CUDA_CHECK(cudaDeviceSynchronize());
  // part layouts (row total 2..8 — runtime v2_rows_ ≤ 8) · variants (v3 bits · lctx bits)
  const std::vector<std::vector<int>> layouts = {{2}, {3}, {4}, {8}, {1, 1}, {2, 3}, {4, 4}, {3, 1, 2}, {2, 2, 2, 2}};
  const int variants[][2] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {3, 3}};
  int n_cases = 0, n_fail = 0;
  printf("⑤ verify front R1 chain vs fused verify version (HIVE_MTP_VERIFY2_FUSED) — timing: new version = env v3/lctx\n");
  printf("%-4s %6s %-10s | %-44s | %8s %8s %5s | %8s %8s %5s\n", "L", "ctx", "parts", "bit match (v3,lctx = 00 10 20 30 33)", "R1 µs", "Q2 µs", "x", "R1 G",
         "Q2 G", "x");
  for (const int ctx : ctxs)
    for (const auto& lay : layouts) {
      const int P = (int)lay.size();
      std::vector<int> pos, part, grp;
      int r = 0;
      for (int p = 0; p < P; ++p) {
        const int p0 = ctx - 1 - 16 * p - (p & 1) - lay[p];  // even/odd start per part (ratio-2 group boundaries at different places) · end position < ctx
        for (int i = 0; i < lay[p]; ++i) { pos.push_back(p0 + i); part.push_back(p); grp.push_back(r); }
        r += lay[p];
      }
      const int M = r;
      std::string lname;
      for (int p = 0; p < P; ++p) lname += (p ? "+" : "") + std::to_string(lay[p]);
      for (int m = 0; m < M; ++m) w.vgrp_h[m] = grp[m];
      std::vector<int> shared_topk(M, 0);
      bool have_cand = false;
      for (const Case& cs : cases) {
        const LayerW& A = LW[&cs - cases];
        fill_bf16<<<gridn((size_t)M * DIM), 256>>>(w.xn, (size_t)M * DIM, -1.7f, 1.7f, 77 + cs.layer * 13 + ctx + M * 7 + P);
        {  // reference top-k columns (like the preceding index layer's result read by non-index layers — chunk offset win + M)
          std::vector<int32_t> idx0((size_t)VROWS * (WIN + TOPK), -1);
          for (int m = 0; m < M; ++m)
            for (int j = 0; j < TOPK; ++j) {
              const int T = cs.ratio ? (pos[m] + 1) / cs.ratio : 0;
              const int p = j * std::max(1, T / TOPK);
              idx0[(size_t)m * (WIN + TOPK) + WIN + j] = (T > 0 && p < T) ? p + WIN + M : -1;
            }
          CUDA_CHECK(cudaMemcpy(w.idx, idx0.data(), idx0.size() * 4, cudaMemcpyHostToDevice));
        }
        const int Tact = fill_tables(w, cs, seqs, part, pos, M, shared_topk);
        if (!cs.index_source && cs.ratio) for (int m = 0; m < M; ++m) { shared_topk[m] = std::min(TOPK, w.trows_h[m]); w.kvrows_h[m].topk = shared_topk[m]; }
        const int Tb = cs.index_source && cs.ratio ? std::min(w.Tcap - 8, std::max(1024, next_pow2(Tact))) : 0;
        CUDA_CHECK(cudaDeviceSynchronize());
        const Saved base = save(cs, w, seqs, part, pos, P);
        bool hc_ref = have_cand;
        r1_front(cs, A, w, blas, M, Tb, hc_ref, fr_win, fr_cmp, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaGetLastError());
        const auto o_ref = grab(cs, w, seqs, part, pos, M, P);
        const std::vector<uint8_t> cand_ref = snap(w.cand, (size_t)VROWS * CB);
        std::string res;
        bool ok_all = true;
        for (const auto& v : variants) {
          restore(w, seqs, P, base);
          bool hc = have_cand;
          q2_front(cs, A, w, blas, M, Tb, hc, fr_win, fr_cmp, v[0], v[1], st);
          CUDA_CHECK(cudaStreamSynchronize(st));
          CUDA_CHECK(cudaGetLastError());
          const auto o = grab(cs, w, seqs, part, pos, M, P);
          std::string bad;
          for (size_t i = 0; i < o.size(); i++)
            if (o[i] != o_ref[i]) {
              size_t n = 0, first = 0;
              for (size_t b = 0; b < std::min(o[i].size(), o_ref[i].size()); ++b) if (o[i][b] != o_ref[i][b]) { if (!n) first = b; ++n; }
              char buf[80]; snprintf(buf, sizeof buf, "%s≠(%zu B@%zu)", kOut[i], n + (o[i].size() != o_ref[i].size() ? 1 : 0), first); bad += buf;
            }
          if (hc != hc_ref) bad += "have_cand≠";
          if (!bad.empty()) { ok_all = false; fprintf(stderr, "  FAIL ⑤ %s ctx %d parts %s v3 %d lctx %d: %s\n", cs.name, ctx, lname.c_str(), v[0], v[1], bad.c_str()); }
          res += bad.empty() ? "ok " : "NO ";
        }
        restore(w, seqs, P, base);
        bool h1 = have_cand, h2 = have_cand;
        const float t_ref = time_eager([&] { r1_front(cs, A, w, blas, M, Tb, h1, fr_win, fr_cmp, st); }, st, 20);
        restore(w, seqs, P, base);
        const float t_new = time_eager([&] { q2_front(cs, A, w, blas, M, Tb, h2, fr_win, fr_cmp, -1, -1, st); }, st, 20);
        restore(w, seqs, P, base);
        const float g_ref = time_graph([&] { r1_front(cs, A, w, blas, M, Tb, h1, fr_win, fr_cmp, st); }, st, 20);
        restore(w, seqs, P, base);
        const float g_new = time_graph([&] { q2_front(cs, A, w, blas, M, Tb, h2, fr_win, fr_cmp, -1, -1, st); }, st, 20);
        restore(w, seqs, P, base);  // candidates for the next layer = the unfused chain's result (identical if the comparison passed)
        put(w.cand, cand_ref);
        have_cand = hc_ref;
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("%-4s %6d %-10s | %-44s | %8.1f %8.1f %5.2f | %8.1f %8.1f %5.2f  %s\n", cs.name, ctx, lname.c_str(), res.c_str(), t_ref, t_new, t_ref / t_new, g_ref,
               g_new, g_ref / g_new, ok_all ? "PASS" : "FAIL");
        fflush(stdout);
        ++n_cases;
        if (!ok_all) ++n_fail;
        check(ok_all, "verify front R1 == Q2 fused");
      }
    }
  printf("⑤ %d cases, %d FAIL\n", n_cases, n_fail);
}

}  // namespace front

}  // namespace

int main(int argc, char** argv) {
  test_window();
  test_compressor();
  test_accum_ordered();
  timing();
  {  // ⑤ context list (comma-separated) — default 4096,32768
    std::vector<int> ctxs;
    const std::string t = argc > 1 ? argv[1] : "4096,32768";
    for (size_t p = 0; p < t.size();) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size(); const int v = atoi(t.substr(p, q - p).c_str()); if (v > 64) ctxs.push_back(v); p = q + 1; }
    if (ctxs.empty()) ctxs = {4096, 32768};
    front::run(ctxs);
  }
  printf("test_verify_decode: %d checks, %d failures — %s\n", g_checks, g_fail, g_fail ? "FAIL" : "ALL PASS");
  return g_fail ? 1 : 0;
}
