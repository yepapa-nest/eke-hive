// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_GROUPED_PREFILL validation: runs one prefill expert layer through (A) the per-expert loop (the run_gpu_expert / run_joined_expert chain of
//   runtime.cpp moe_experts_multi: gather_rows_u8 ×2 → gemm_bs w1/w3 → swiglu_route → act_quant_fp8 → gemm_bs w2 → accum_bf16_rows) and
//   (B) the grouped path (hive/moe_grouped_plan.h plan → k::GroupedMoe::run batch execution), compares acc (fp32) **bit for bit** and measures time.
//   Shape = the real layer (the checkpoint's config.json: dim 5120, moe_inter 2304, 384 experts, top-6, swiglu_limit 10). Weights = random data in the real storage
//   format (ExpertLayout: e2m1 nibbles + e8m0/32 scales, one record per expert). Routing = 6 distinct experts per token with Zipf popularity (skewed like
//   real traffic — many experts with fewer than 128 rows).
//   Cases: 16384 rows (1 sub-chunk) and 49152 (3 sub-chunks × 16384, both joined = the HIVE_TILE_GROUP_GEMM version and unjoined). Paths: mx tiled
//   (the HIVE_MX_GEMM=1 version, default) and WMMA (tc forced).
//   Streaming emulation: ~70 % of the experts are marked late (computed after waiting for the copy) and planned with a staging ring of 48 (no real copies — wait events only).
//   Verdict: acc bit-identical in every case = PASS; mismatch within the fp32 reordering tolerance (relative 1e-5) = PASS-TOL (bit identity was claimed, so the cause
//   must be investigated); anything else = FAIL.
//   Run: scripts/hive-run.sh "./build-dev/test_grouped_prefill"   (optional: first argument = repetitions (default 3); with HIVE_MX_GEMM=0, tc twice instead of the tiled path)
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/expert_store.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/moe_grouped_plan.h"
#include "hive/clock.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

// Shape = the real layer. HIVE_GP_SMALL is a reduced shape for the CPU emulation (logic comparison) only — GPU runs always use the real shape.
#ifdef HIVE_GP_SMALL
constexpr int DIM = 256, INTER = 128, E = 160, TOPK = 6, WM = 100;
constexpr double ZIPF = 1.6;  // so that some experts' joined segs exceed WM (multi-seg expert path)
#else
constexpr int DIM = 5120, INTER = 2304, E = 384, TOPK = 6, WM = 16384;  // WM = work buffer rows (max_chunk 16384)
constexpr double ZIPF = 1.1;  // popularity ∝ (rank + 8)^-1.1 — in a 16K chunk ~216 experts get fewer than 128 rows (same shape as the observed skew)
#endif
constexpr float LIMIT = 10.f;

__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed, int mode) {
  const size_t i = hive::cu::global_tid();
  if (i >= n) return;
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  uint8_t v = (uint8_t)h;
  if (mode == 1) {  // e4m3 activations: excluding NaN (0x7F/0xFF), magnitude around <= 2^2
    v = (uint8_t)((v & 0x80) | ((v & 0x7F) % 0x48));
  } else if (mode == 2) {  // activation scales e8m0 ∈ [120, 127]
    v = (uint8_t)(120 + (h >> 8) % 8);
  } else if (mode == 3) {  // weight scales e8m0 ∈ [118, 123]
    v = (uint8_t)(118 + (h >> 8) % 6);
  }  // mode 0: random bytes = two e2m1 nibbles (all 16 values valid)
  p[i] = v;
}
void fill(void* p, size_t n, uint32_t seed, int mode) { fill_kernel<<<(unsigned)((n + 255) / 256), 256>>>((uint8_t*)p, n, seed, mode); CUDA_CHECK(cudaGetLastError()); }

struct Ref { int s, m, j; };
double now_ms() { return hive::mono_ms(); }
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, n * sizeof(T))); return p; }

struct Layer {
  ExpertLayout lay;
  std::vector<uint8_t*> rec;  // per-expert records (device)
};

struct Case {
  int S, M;       // number of sub-chunks, rows per sub-chunk
  bool joined;    // HIVE_TILE_GROUP_GEMM version (sub-chunks > 1)
  const char* name;
};

struct Result { bool exact = false, tol = false; double max_abs = 0, max_rel = 0; size_t mism = 0; double ms_old = 0, ms_new = 0; int batches = 0; };

Result run_case(const Layer& L, const Case& cs, int reps, uint32_t seed) {
  const int S = cs.S, M = cs.M, dim = DIM, I = INTER, k = TOPK;
  std::mt19937 rng(seed);
  // Routing (Zipf popularity, random permutation) — k distinct experts per token
  std::vector<int> perm(E); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<double> pw(E); for (int r = 0; r < E; ++r) pw[perm[r]] = 1.0 / std::pow(r + 8.0, ZIPF);
  std::discrete_distribution<int> pick(pw.begin(), pw.end());
  std::uniform_real_distribution<float> urw(0.05f, 0.5f);
  std::vector<std::vector<int>> ids(S, std::vector<int>((size_t)M * k));
  std::vector<std::vector<float>> rws(S, std::vector<float>((size_t)M * k));
  for (int s = 0; s < S; ++s)
    for (int m = 0; m < M; ++m)
      for (int j = 0; j < k; ++j) {
        int e;
        bool dup;
        do { e = pick(rng); dup = false; for (int q = 0; q < j; ++q) dup |= ids[s][(size_t)m * k + q] == e; } while (dup);
        ids[s][(size_t)m * k + j] = e; rws[s][(size_t)m * k + j] = urw(rng);
      }
  // Per expert (s, m, j) — same order as the runtime (s → m → j)
  std::vector<std::vector<Ref>> by_e(E);
  for (int s = 0; s < S; ++s) for (int m = 0; m < M; ++m) for (int j = 0; j < k; ++j) by_e[ids[s][(size_t)m * k + j]].push_back({s, m, j});
  std::vector<int> order(E); for (int e = 0; e < E; ++e) order[e] = e; std::shuffle(std::begin(order), std::end(order), rng);  // emulates an arbitrary order (prefetch-first etc.)
  int small = 0, lt128 = 0, used = 0, mx = 0;  // skew summary
  for (int e = 0; e < E; e++) { const int n = (int)by_e[e].size(); if (!n) continue; ++used; small += n <= 8 * S; lt128 += n < 128; mx = std::max(mx, n); }
  printf("  routing: experts used %d · <128 rows %d · ≤8/sub %d · max rows %d\n", used, lt128, small, mx);
  // Activations (per-sub-chunk xq/xs) and initial acc (emulates the shared-expert contribution)
  std::vector<uint8_t*> xq(S), xs(S);
  std::vector<float*> acc_old(S), acc_new(S), acc_init(S);
  for (int s = 0; s < S; ++s) {
    xq[s] = dmalloc<uint8_t>((size_t)M * dim); xs[s] = dmalloc<uint8_t>((size_t)M * dim / 32);
    fill(xq[s], (size_t)M * dim, seed * 7 + s, 1); fill(xs[s], (size_t)M * dim / 32, seed * 11 + s, 2);
    acc_old[s] = dmalloc<float>((size_t)M * dim); acc_new[s] = dmalloc<float>((size_t)M * dim); acc_init[s] = dmalloc<float>((size_t)M * dim);
    std::vector<float> h((size_t)M * dim);
    for (auto& v : h) v = std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
    CUDA_CHECK(cudaMemcpy(acc_init[s], h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  }
  // Work buffers (same capacity as the runtime Work — WM rows)
  bf16 *gate = dmalloc<bf16>((size_t)WM * I), *up = dmalloc<bf16>((size_t)WM * I), *y = dmalloc<bf16>((size_t)WM * I), *eout = dmalloc<bf16>((size_t)WM * dim);
  uint8_t *yq = dmalloc<uint8_t>((size_t)WM * I), *ys = dmalloc<uint8_t>((size_t)WM * I / 32), *gxq = dmalloc<uint8_t>((size_t)WM * dim),
          *gxs = dmalloc<uint8_t>((size_t)WM * dim / 32);
  cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const ExpertLayout& lay = L.lay;
  // ---- (A) per-expert loop: build and upload the row table first (hoff order of the runtime g_rows_h/g_rw_h)
  std::vector<int32_t> rows_h; std::vector<float> rw_h;
  struct Call { int e, s, off, n, begin_off; bool joined_head; std::vector<std::array<int, 4>> parts; };  // parts: s, off, begin, n
  std::vector<Call> calls;
  for (int e : order) {
    const auto& refs = by_e[e];
    if (refs.empty()) continue;
    if (cs.joined && S > 1) {
      for (size_t i = 0; i < refs.size();) {
        Call c{e, -1, 0, 0, (int)rows_h.size(), true, {}};
        int n = 0;
        while (i < refs.size() && n < WM) {
          const int s = refs[i].s, off = (int)rows_h.size(), begin = n;
          while (i < refs.size() && refs[i].s == s && n < WM) { const Ref& r = refs[i++]; rows_h.push_back(r.m); rw_h.push_back(rws[s][(size_t)r.m * k + r.j]); ++n; }
          c.parts.push_back({s, off, begin, n - begin});
        }
        c.n = n;
        calls.push_back(c);
      }
    } else {
      for (size_t i = 0; i < refs.size();) {
        const int s = refs[i].s, off = (int)rows_h.size();
        int n = 0;
        while (i < refs.size() && refs[i].s == s) { rows_h.push_back(refs[i].m); rw_h.push_back(rws[s][(size_t)refs[i].m * k + refs[i].j]); ++n; ++i; }
        calls.push_back(Call{e, s, off, n, off, false, {}});
      }
    }
  }
  int32_t* rows_d = dmalloc<int32_t>(rows_h.size()); float* rw_d = dmalloc<float>(rw_h.size());
  CUDA_CHECK(cudaMemcpy(rows_d, rows_h.data(), rows_h.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rw_d, rw_h.data(), rw_h.size() * 4, cudaMemcpyHostToDevice));
  auto run_old = [&] {
    for (const Call& c : calls) {
      const uint8_t* rec = L.rec[c.e];
      if (!c.joined_head) {
        k::gather_rows_u8(xq[c.s], rows_d + c.off, c.n, dim, gxq, st);
        k::gather_rows_u8(xs[c.s], rows_d + c.off, c.n, dim / 32, gxs, st);
        k::gemm_bs(gxq, gxs, rec + lay.w1, rec + lay.s1, true, c.n, I, dim, gate, nullptr, st);
        k::gemm_bs(gxq, gxs, rec + lay.w3, rec + lay.s3, true, c.n, I, dim, up, nullptr, st);
        k::swiglu_route(gate, up, rw_d + c.off, c.n, I, LIMIT, y, st);
        k::act_quant_fp8(y, c.n, I, yq, ys, st);
        k::gemm_bs(yq, ys, rec + lay.w2, rec + lay.s2, true, c.n, dim, I, eout, nullptr, st);
        k::accum_bf16_rows(eout, rows_d + c.off, c.n, dim, acc_old[c.s], st);
      } else {
        for (const auto& p : c.parts) {
          k::gather_rows_u8(xq[p[0]], rows_d + p[1], p[3], dim, gxq + (size_t)p[2] * dim, st);
          k::gather_rows_u8(xs[p[0]], rows_d + p[1], p[3], dim / 32, gxs + (size_t)p[2] * (dim / 32), st);
        }
        k::gemm_bs(gxq, gxs, rec + lay.w1, rec + lay.s1, true, c.n, I, dim, gate, nullptr, st);
        k::gemm_bs(gxq, gxs, rec + lay.w3, rec + lay.s3, true, c.n, I, dim, up, nullptr, st);
        k::swiglu_route(gate, up, rw_d + c.begin_off, c.n, I, LIMIT, y, st);
        k::act_quant_fp8(y, c.n, I, yq, ys, st);
        k::gemm_bs(yq, ys, rec + lay.w2, rec + lay.s2, true, c.n, dim, I, eout, nullptr, st);
        for (const auto& p : c.parts) k::accum_bf16_rows(eout + (size_t)p[2] * dim, rows_d + p[1], p[3], dim, acc_old[p[0]], st);
      }
    }
  };
  // ---- (B) grouped path: same segs, plan and execution as the grouped branch of runtime moe_experts_multi
  const int SS = 48;
  std::vector<gp::Expert> gex; std::vector<int> gex_e, seg_rows, seg_i0;
  std::bernoulli_distribution is_late(0.7);
  for (int e : order) {
    const auto& refs = by_e[e];
    if (refs.empty()) continue;
    gp::Expert X{};
    X.seg_begin = (int)seg_rows.size();
    if (cs.joined && S > 1) for (size_t i = 0; i < refs.size(); i += WM) { seg_i0.push_back((int)i); seg_rows.push_back((int)std::min(refs.size() - i, (size_t)WM)); }
    else for (size_t i = 0; i < refs.size();) { size_t j = i; while (j < refs.size() && refs[j].s == refs[i].s) ++j; seg_i0.push_back((int)i); seg_rows.push_back((int)(j - i)); i = j; }
    X.seg_end = (int)seg_rows.size();
    X.kind = is_late(rng) ? gp::kStream : gp::kResident;
    gex.push_back(X); gex_e.push_back(e);
  }
  uint32_t next = 0;
  std::vector<gp::Action> acts;
  gp::plan_batches(gex, seg_rows, WM, SS / 2, SS, next, acts);
  k::GroupedMoe gm(WM, S);
  std::vector<const uint8_t*> xq_sub(xq.begin(), xq.end()), xs_sub(xs.begin(), xs.end());
  std::vector<int> sub_rows(S, M);
  cudaEvent_t copied; CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
  const k::GroupedBufs gb{gate, up, y, eout, yq, ys, gxq, gxs};
  auto run_new = [&] {
    std::vector<k::GroupedSeg> bsegs; std::vector<k::GroupedRef> brefs;
    CUDA_CHECK(cudaEventRecord(copied, st));  // emulates copy completion (an already completed event)
    for (const gp::Action& a : acts) {
      if (a.type == gp::kCopy) continue;
      if (a.type == gp::kSeg) {
        const gp::Expert& X = gex[a.x];
        const int e = gex_e[a.x];
        const uint8_t* rec = L.rec[e];
        const int i0 = seg_i0[a.seg], n = seg_rows[a.seg];
        bsegs.push_back(k::GroupedSeg{rec + lay.w1, rec + lay.s1, rec + lay.w3, rec + lay.s3, rec + lay.w2, rec + lay.s2, (int)brefs.size(), n, X.kind != gp::kResident});
        for (int t = i0; t < i0 + n; ++t) { const Ref& r = by_e[e][t]; brefs.push_back(k::GroupedRef{r.s, r.m, rws[r.s][(size_t)r.m * k + r.j]}); }
      } else {
        gm.run(bsegs.data(), (int)bsegs.size(), brefs.data(), xq_sub.data(), xs_sub.data(), acc_new.data(), S, sub_rows.data(), dim, I, LIMIT, gb, st,
               [&] { CUDA_CHECK(cudaStreamWaitEvent(st, copied, 0)); });
        bsegs.clear(); brefs.clear();
      }
    }
  };
  Result res;
  res.ms_old = res.ms_new = 1e30;
  const int b0 = gm.batches();
  for (int rep = 0; rep < reps; ++rep) {
    for (int s = 0; s < S; ++s) {
      CUDA_CHECK(cudaMemcpyAsync(acc_old[s], acc_init[s], (size_t)M * dim * 4, cudaMemcpyDeviceToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(acc_new[s], acc_init[s], (size_t)M * dim * 4, cudaMemcpyDeviceToDevice, st));
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    double t0 = now_ms(); run_old(); CUDA_CHECK(cudaStreamSynchronize(st)); res.ms_old = std::min(res.ms_old, now_ms() - t0);
    t0 = now_ms(); run_new(); CUDA_CHECK(cudaStreamSynchronize(st)); res.ms_new = std::min(res.ms_new, now_ms() - t0);
  }
  res.batches = (gm.batches() - b0) / std::max(1, reps);
  CUDA_CHECK(cudaGetLastError());
  // Comparison
  res.exact = true;
  bool finite = true;
  for (int s = 0; s < S; ++s) {
    std::vector<float> a((size_t)M * dim), b((size_t)M * dim);
    CUDA_CHECK(cudaMemcpy(a.data(), acc_old[s], a.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), acc_new[s], b.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); i++) {
      const float x = a[i], y = b[i];
      if (!std::isfinite(x) || !std::isfinite(y)) finite = false;
      if (std::memcmp(&x, &y, sizeof x) != 0) {
        ++res.mism; res.exact = false;
        const double d = std::fabs((double)x - y);
        res.max_abs = std::max(res.max_abs, d);
        res.max_rel = std::max(res.max_rel, d / std::max(1e-3, std::fabs((double)a[i])));
      }
    }
  }
  res.tol = finite && res.max_rel < 1e-5;
  if (!finite) { res.exact = false; printf("  ⚠️ non-finite values in acc (test data scale)\n"); }
  for (int s = 0; s < S; ++s) { cudaFree(xq[s]); cudaFree(xs[s]); cudaFree(acc_old[s]); cudaFree(acc_new[s]); cudaFree(acc_init[s]); }
  cudaFree(gate); cudaFree(up); cudaFree(y); cudaFree(eout); cudaFree(yq); cudaFree(ys); cudaFree(gxq); cudaFree(gxs); cudaFree(rows_d); cudaFree(rw_d);
  cudaEventDestroy(copied); cudaStreamDestroy(st);
  return res;
}

const char* route_name(int r) {
  switch (r) { case k::kGemmMxTiled: return "mx-tiled"; case k::kGemmMxRows: return "mx-rows"; case k::kGemmTc: return "wmma-tc"; case k::kGemmGemv: return "gemv"; default: return "cuda-core"; }
}

}  // namespace

int main(int argc, char** argv) {
  setenv("HIVE_MX_GEMM", "1", 0);  // serving default (HIVE_MX_GEMM=1, see docs/configuration.md) — an externally set value takes precedence
  const int reps = argc > 1 ? std::max(1, atoi(argv[1])) : 3;
  Layer L;
  L.lay = ExpertLayout::make(DIM, INTER);
  L.rec.resize(E);
  for (int e = 0; e < E; ++e) {
    L.rec[e] = dmalloc<uint8_t>(L.lay.total);
    fill(L.rec[e] + L.lay.w1, (size_t)INTER * DIM / 2, 1000 + e, 0); fill(L.rec[e] + L.lay.s1, (size_t)INTER * DIM / 32, 2000 + e, 3);
    fill(L.rec[e] + L.lay.w3, (size_t)INTER * DIM / 2, 3000 + e, 0); fill(L.rec[e] + L.lay.s3, (size_t)INTER * DIM / 32, 4000 + e, 3);
    fill(L.rec[e] + L.lay.w2, (size_t)DIM * INTER / 2, 5000 + e, 0); fill(L.rec[e] + L.lay.s2, (size_t)DIM * INTER / 32, 6000 + e, 3);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const Case cases[] = {{1, WM, false, "rows 16384 (1 sub-chunk)"}, {3, WM, false, "rows 49152 (3 sub-chunks, per-sub GEMM)"},
                        {3, WM, true, "rows 49152 (3 sub-chunks, joined = HIVE_TILE_GROUP_GEMM)"}};
  bool all_exact = true, any_fail = false;
  for (int pass = 0; pass < 2; ++pass) {
    k::gemm_bs_force_path(pass == 0 ? 0 : 2);  // 0 = auto (mx tiled with HIVE_MX_GEMM), 2 = force WMMA tc
    const int r13 = k::gemm_bs_route(true, 9, DIM), r2 = k::gemm_bs_route(true, 9, INTER);
    printf("== pass %d: route w13 %s · w2 %s · n ≤ 8 → gemv (grouped supported: %s)\n", pass, route_name(r13), route_name(r2),
           k::GroupedMoe::supported(DIM, INTER) ? "yes" : "no");
    if (!k::GroupedMoe::supported(DIM, INTER)) { printf("  skip (no grouped kernel for this route)\n"); continue; }
    for (int ci = 0; ci < 3; ++ci) {
      const Case& cs = cases[ci];
      printf("-- %s\n", cs.name);
      const Result r = run_case(L, cs, reps, 20260930u + 17u * ci + 101u * pass);
      const char* verdict = r.exact ? "PASS (bit-exact)" : (r.tol ? "PASS-TOL (not bit-exact!)" : "FAIL");
      printf("  old per-expert %.2f ms · grouped %.2f ms (%d batches) · speedup ×%.2f · mismatches %zu · max abs %.3e · max rel %.3e  → %s\n", r.ms_old,
             r.ms_new, r.batches, r.ms_old / std::max(1e-9, r.ms_new), r.mism, r.max_abs, r.max_rel, verdict);
      all_exact &= r.exact;
      any_fail |= !r.exact && !r.tol;
    }
  }
  k::gemm_bs_force_path(0);
  for (auto* p : L.rec) cudaFree(p);
  printf("RESULT: %s\n", any_fail ? "FAIL" : (all_exact ? "PASS (all bit-exact)" : "PASS-TOL"));
  return any_fail ? 1 : 0;
}
