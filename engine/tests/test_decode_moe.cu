// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_FUSED check: runs one decode expert layer (the GPU share of runtime.cpp moe_decode_experts) as
//   (A) the unfused chain — exactly the fuse_ && use_mx path: mx_grouped_w13 (row map + yq/ys) → mx_grouped_w2 → accum_bf16_rows_seq, twice (resident groups, DMA groups)
//   (B) fused — one k::moe_decode_fused launch for each of the same two parts
//   and compares acc (fp32), yq, ys and eout **bit for bit**; with L2 flushed it measures µs per layer and effective GB/s (expert record bytes used / time).
//   Shape = a real layer (the checkpoint's config.json: hidden 5120 · moe_intermediate 2304 · top-6 · swiglu_limit 10). Weights = random data in the real storage format (ExpertLayout:
//   e2m1 nibbles + e8m0/32 scales, one record per expert). Groups follow the runtime rules (experts ascending · (m,j) order within an expert · ≤ 8 rows per group ·
//   rw in group row order) · the last ~1/4 of the groups are split off as the "DMA share" (second launch) so the accumulation order (resident → DMA) is covered too.
//   Cases: M = 1, 2, 4, 6, 8 (6 distinct experts per row from a pool of NE, skewed popularity) + M=8 with a pool of 12 (many rows per group).
//   (B) runs per variant — D1 = HIVE_DECODE_FUSED · F2 = HIVE_DECODE_FUSED2 with S2 = 3/4/6 stages × the host-table variant (moe_decode_fused) and the DEV variant (moe_decode_fused_dev,
//   device-side counts plus spare grid capacity, as in the step graph) — selected with k::moe_decode_fused_force (independent of the environment). Each line shows µs · GB/s · ratio vs the unfused chain ·
//   ratio vs D1 with the same launch mode · bit verdict. The fastest F2 S2 is the basis for the HIVE_DECODE_FUSED2_STAGES default.
//   F3 = HIVE_DECODE_FUSED3 (S = 3/4/5 stages — k::moe_decode_fused_force3), 6 variants (host, DEV). Besides the bit verdict against the unfused chain, F3 lines are also compared byte for byte
//   against the D1 and F2s4 results of the same case (all of acc; "==D1 ==F2s4" columns) · the "vs F2s4" column = ratio vs F2s4 with the same launch mode. A summary of the fastest variant per M follows at the end.
//   GB/s = expert record bytes used in this layer (w1·w3·w2 + scales, once per expert) / (time of the two launches, resident + DMA) — L2 flushed.
//   Each variant prints "launching <variant>" before synchronizing — if a launch crashes (e.g. illegal instruction), the variant on the last line is the culprit.
//   Verdict: bit identity in every case = PASS, otherwise FAIL (bit identity by design — no tolerance). If a fused kernel does not finish within 10 s: FAIL (hang) and exit.
//   Run: scripts/hive-run.sh "./build-dev/test_decode_moe [reps=20] [NE=96]"   — needs ≈ NE × 18.9 MB + 0.3 GB of free VRAM (NE is reduced, with a notice, if short).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "hive/common.h"
#include "hive/expert_store.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/clock.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

constexpr int DIM = 5120, INTER = 2304, TOPK = 6;
constexpr float LIMIT = 10.f;

__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed, int mode) {
  const size_t i = hive::cu::global_tid();
  if (i >= n) return;
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  uint8_t v = (uint8_t)h;
  if (mode == 1) v = (uint8_t)((v & 0x80) | ((v & 0x7F) % 0x48));  // e4m3 activations (no NaN)
  else if (mode == 2) v = (uint8_t)(120 + (h >> 8) % 8);            // activation scales e8m0
  else if (mode == 3) v = (uint8_t)(118 + (h >> 8) % 6);            // weight scales e8m0
  p[i] = v;                                                         // mode 0: two e2m1 nibbles
}
void fill(void* p, size_t n, uint32_t seed, int mode) {
  fill_kernel<<<(unsigned)((n + 255) / 256), 256>>>((uint8_t*)p, n, seed, mode);
  CUDA_CHECK(cudaGetLastError());
}
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T))); return p; }
double now_ms() { return hive::mono_ms(); }

struct Bufs {  // decode buffers of the runtime Work (rows = M·k)
  uint8_t *xq, *xs, *yq, *ys, *tbl;
  bf16 *y, *eout;
  float *acc, *acc_init;
};

// variant = (S2, dev) — S2 0 = D1 · 3/4/6 = F2 (HIVE_DECODE_FUSED2, stage count) · dev = moe_decode_fused_dev (step-graph variant, device-side counts)
// f3 > 0 = F3 variant (stage count f3 — moe_decode_fused_force3); otherwise moe_decode_fused_force with s2
struct Variant { const char* name; int s2; bool dev; int f3 = 0; };
const Variant kVariants[] = {{"D1", 0, false}, {"D1-dev", 0, true}, {"F2s3", 3, false}, {"F2s4", 4, false}, {"F2s6", 6, false},
                             {"F2s3-dev", 3, true}, {"F2s4-dev", 4, true}, {"F2s6-dev", 6, true},
                             {"F3s3", 0, false, 3}, {"F3s4", 0, false, 4}, {"F3s5", 0, false, 5},
                             {"F3s3-dev", 0, true, 3}, {"F3s4-dev", 0, true, 4}, {"F3s5-dev", 0, true, 5}};
constexpr int NV = sizeof(kVariants) / sizeof(kVariants[0]);
struct VRes { bool exact = true; size_t mism_acc = 0, mism_q = 0; double us = 0; int eq_d1 = -1, eq_f2 = -1; };  // eq_*: F3 only (-1 = not compared)
int vidx(const char* name) { for (int i = 0; i < NV; ++i) if (!strcmp(kVariants[i].name, name)) return i; return -1; }
struct Result { bool exact = true; double us_old = 0; double mb = 0; int ng = 0, n_exp = 0; VRes v[NV]; };

bool wait_done(cudaStream_t st, double limit_ms) {  // fused-kernel deadlock watchdog (false if it does not finish)
  cudaEvent_t e; CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(e, st));
  const double t0 = now_ms();
  cudaError_t r;
  while ((r = cudaEventQuery(e)) == cudaErrorNotReady) if (now_ms() - t0 > limit_ms) return false;
  CUDA_CHECK(r);
  cudaEventDestroy(e);
  return true;
}

Result run_case(const std::vector<uint8_t*>& rec, const ExpertLayout& lay, int M, int pool, int reps, uint32_t seed, uint8_t* flush, size_t flush_n) {
  const int dim = DIM, I = INTER, k = TOPK, R = M * k;
  std::mt19937 rng(seed);
  // routing: k distinct experts per row from the pool (popularity ∝ (rank+4)^-1 — so batch rows share some experts)
  std::vector<int> perm(pool); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<double> pw(pool); for (int r = 0; r < pool; ++r) pw[perm[r]] = 1.0 / (r + 4.0);
  std::discrete_distribution<int> pick(pw.begin(), pw.end());
  std::uniform_real_distribution<float> urw(0.05f, 0.5f);
  std::vector<int> ids(R); std::vector<float> rwv(R);
  for (int m = 0; m < M; ++m)
    for (int j = 0; j < k; ++j) {
      int e; bool dup;
      do { e = pick(rng); dup = false; for (int q = 0; q < j; ++q) dup |= ids[m * k + q] == e; } while (dup);
      ids[m * k + j] = e; rwv[m * k + j] = urw(rng);
    }
  // grouping as in the runtime: experts ascending · within an expert i = m·k+j ascending · ≤ 8 rows per group
  std::vector<k::GroupDesc> gd; std::vector<int32_t> grows; std::vector<float> grw;
  std::vector<int> used;
  for (int e = 0; e < pool; ++e) {
    std::vector<int> mj;
    for (int i = 0; i < R; ++i) if (ids[i] == e) mj.push_back(i);
    if (mj.empty()) continue;
    used.push_back(e);
    for (size_t i0 = 0; i0 < mj.size(); i0 += 8) {
      k::GroupDesc d{};
      const uint8_t* rp = rec[e];
      d.w1 = rp + lay.w1; d.s1 = rp + lay.s1; d.w3 = rp + lay.w3; d.s3 = rp + lay.s3; d.w2 = rp + lay.w2; d.s2 = rp + lay.s2;
      d.row0 = (int)grows.size(); d.n = (int)std::min<size_t>(8, mj.size() - i0);
      for (int i = (int)i0; i < (int)i0 + d.n; ++i) { grows.push_back(mj[i] / k); grw.push_back(rwv[mj[i]]); }
      gd.push_back(d);
    }
  }
  const int ng = (int)gd.size(), goff = (int)grows.size();
  const int ng_hit = ng >= 2 ? ng - std::max(1, ng / 4) : ng;  // last ~1/4 = DMA share (second launch)
  const int R_hit = ng_hit < ng ? gd[ng_hit].row0 : goff;
  Result res; res.ng = ng; res.n_exp = (int)used.size();
  res.mb = (double)used.size() * 3.0 * ((double)I * dim / 2 + (double)I * dim / 32) / 1e6;
  bool aligned = true;
  for (const auto& d : gd) aligned &= k::moe_decode_desc_aligned(d);
  if (!k::moe_decode_fused_ok(dim, I, M, ng) || !aligned) { printf("  fused path not applicable (shape/alignment) → FAIL\n"); res.exact = false; return res; }
  // table (one block [gdesc][rows][rw], as in the runtime) · buffers
  const size_t off_rows = sizeof(k::GroupDesc) * ng, off_rw = off_rows + (size_t)goff * 4, total = off_rw + (size_t)goff * 4;
  std::vector<uint8_t> tbl(total);
  std::memcpy(tbl.data(), gd.data(), off_rows); std::memcpy(tbl.data() + off_rows, grows.data(), goff * 4); std::memcpy(tbl.data() + off_rw, grw.data(), goff * 4);
  Bufs A{}, B{};
  for (Bufs* b : {&A, &B}) {
    b->xq = dmalloc<uint8_t>((size_t)M * dim); b->xs = dmalloc<uint8_t>((size_t)M * dim / 32);
    b->yq = dmalloc<uint8_t>((size_t)R * I); b->ys = dmalloc<uint8_t>((size_t)R * I / 32);
    b->y = dmalloc<bf16>((size_t)R * I); b->eout = dmalloc<bf16>((size_t)R * dim);
    b->acc = dmalloc<float>((size_t)M * dim); b->acc_init = dmalloc<float>((size_t)M * dim);
    b->tbl = dmalloc<uint8_t>(total);
    CUDA_CHECK(cudaMemcpy(b->tbl, tbl.data(), total, cudaMemcpyHostToDevice));
  }
  auto clear = [&](Bufs& b) {
    CUDA_CHECK(cudaMemset(b.yq, 0, (size_t)R * I)); CUDA_CHECK(cudaMemset(b.ys, 0, (size_t)R * I / 32)); CUDA_CHECK(cudaMemset(b.eout, 0, (size_t)R * dim * 2));
    CUDA_CHECK(cudaMemset(b.y, 0, (size_t)R * I * 2));
  };
  clear(A);
  fill(A.xq, (size_t)M * dim, seed * 7, 1); fill(A.xs, (size_t)M * dim / 32, seed * 11, 2);
  CUDA_CHECK(cudaMemcpy(B.xq, A.xq, (size_t)M * dim, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(B.xs, A.xs, (size_t)M * dim / 32, cudaMemcpyDeviceToDevice));
  {
    std::vector<float> h((size_t)M * dim);
    for (auto& v : h) v = std::uniform_real_distribution<float>(-1.f, 1.f)(rng);  // stands in for the shared-expert share
    CUDA_CHECK(cudaMemcpy(A.acc_init, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B.acc_init, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  }
  // device-side counts for the DEV variant (same layout as the first four fields of hs::LayerCounts: ng_hit · ng_str · R_hit · R_str)
  const int R_str = goff - R_hit, ng_str = ng - ng_hit;
  int* dcnt = dmalloc<int>(4);
  { const int h[4] = {ng_hit, ng_str, R_hit, R_str}; CUDA_CHECK(cudaMemcpy(dcnt, h, sizeof h, cudaMemcpyHostToDevice)); }
  cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  auto tp = [&](const Bufs& b) {
    return std::make_tuple(reinterpret_cast<const k::GroupDesc*>(b.tbl), reinterpret_cast<const int32_t*>(b.tbl + off_rows), reinterpret_cast<const float*>(b.tbl + off_rw));
  };
  auto run_old = [&] {
    auto [g, rows, rw] = tp(A);
    auto part = [&](int g0, int n, int r0, int nr) {
      if (n <= 0) return;
      k::mx_grouped_w13(g + g0, n, A.xq, A.xs, I, dim, rw, LIMIT, A.y, st, rows, A.yq, A.ys);
      k::mx_grouped_w2(g + g0, n, A.yq, A.ys, dim, I, A.eout, st);
      k::accum_bf16_rows_seq(A.eout + (size_t)r0 * dim, rows + r0, nr, M, dim, A.acc, st);
    };
    part(0, ng_hit, 0, R_hit);
    part(ng_hit, ng - ng_hit, R_hit, goff - R_hit);
  };
  auto run_new = [&](bool dev) {
    auto [g, rows, rw] = tp(B);
    if (!dev) {
      k::moe_decode_fused(g, ng_hit, B.xq, B.xs, rows, rw, 0, R_hit, M, dim, I, LIMIT, B.y, B.yq, B.ys, B.eout, B.acc, st);
      if (ng > ng_hit) k::moe_decode_fused(g + ng_hit, ng - ng_hit, B.xq, B.xs, rows, rw, R_hit, goff - R_hit, M, dim, I, LIMIT, B.y, B.yq, B.ys, B.eout, B.acc, st);
      return;
    }
    // as in the step graph: grid = capacity (here groups + 2 — also checks that surplus CTAs exit immediately); the actual group count and row range come from device-side counts. Launched even when the DMA share is 0 (zero items).
    k::moe_decode_fused_dev(g, ng_hit + 2, k::DfDevCount{dcnt, 0, -1, 2}, B.xq, B.xs, rows, rw, M, dim, I, LIMIT, B.y, B.yq, B.ys, B.eout, B.acc, st);
    k::moe_decode_fused_dev(g + ng_hit, ng_str + 2, k::DfDevCount{dcnt, 1, 2, 3}, B.xq, B.xs, rows, rw, M, dim, I, LIMIT, B.y, B.yq, B.ys, B.eout, B.acc, st);
  };
  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  const int nt = std::max(1, reps - 1);
  // (A) unfused chain
  double t_old = 0;
  for (int rep = 0; rep < reps; ++rep) {
    CUDA_CHECK(cudaMemcpyAsync(A.acc, A.acc_init, (size_t)M * dim * 4, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemsetAsync(flush, rep & 0xFF, flush_n, st));  // flush L2 (so weights come from HBM)
    float ms = 0;
    if (rep == 0) { printf("  [M=%d groups %d] launching old chain\n", M, ng); fflush(stdout); }
    CUDA_CHECK(cudaEventRecord(e0, st)); run_old(); CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1)); CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); if (rep) t_old += ms;  // first rep = warm-up
  }
  CUDA_CHECK(cudaGetLastError());
  res.us_old = t_old / nt * 1000.0;
  std::vector<float> ref((size_t)M * dim), got((size_t)M * dim), first;
  std::vector<float> acc_d1, acc_f2;  // D1 and F2s4 (host variant) results — compared directly with F3
  CUDA_CHECK(cudaMemcpy(ref.data(), A.acc, ref.size() * 4, cudaMemcpyDeviceToHost));
  bool finite = true; for (float v : ref) finite &= std::isfinite(v);
  if (!finite) printf("  ⚠️ non-finite values in reference acc (test data scale)\n");
  auto cmp_dev = [&](const void* a, const void* b, size_t n) {
    std::vector<uint8_t> x(n), y(n);
    CUDA_CHECK(cudaMemcpy(x.data(), a, n, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(y.data(), b, n, cudaMemcpyDeviceToHost));
    size_t d = 0; for (size_t i = 0; i < n; ++i) d += x[i] != y[i];
    return d;
  };
  // (B) per fused variant: same inputs · L2 flushed · determinism of the first and last rep · bit comparison with the unfused chain (all of acc + y/yq/ys/eout of the rows written)
  for (int vi = 0; vi < NV; ++vi) {
    const Variant& V = kVariants[vi];
    VRes& vr = res.v[vi];
    if (V.f3 > 0) k::moe_decode_fused_force3(V.f3);
    else k::moe_decode_fused_force(V.s2);
    // print which variant is launched before synchronizing (if it crashes, the last line names it — errors are sticky and could surface in the next variant, so synchronize after each variant)
    printf("  [M=%d groups %d] launching %s (variant %d, %s)\n", M, ng, V.name, k::moe_decode_fused_variant(), V.dev ? "dev" : "host");
    fflush(stdout);
    clear(B);
    double t_new = 0;
    for (int rep = 0; rep < reps; ++rep) {
      CUDA_CHECK(cudaMemcpyAsync(B.acc, B.acc_init, (size_t)M * dim * 4, cudaMemcpyDeviceToDevice, st));
      CUDA_CHECK(cudaMemsetAsync(flush, (rep + vi + 1) & 0xFF, flush_n, st));
      float ms = 0;
      CUDA_CHECK(cudaEventRecord(e0, st)); run_new(V.dev); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(e1, st));
      if (!wait_done(st, 10000.0)) { printf("  FAIL (hang): %s did not finish in 10 s (M=%d, groups %d)\n", V.name, M, ng); fflush(stdout); std::_Exit(3); }
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); if (rep) t_new += ms;
      if (rep == 0 || rep == reps - 1) {  // counter self-reset holds across reps + determinism between reps
        CUDA_CHECK(cudaMemcpy(got.data(), B.acc, got.size() * 4, cudaMemcpyDeviceToHost));
        if (rep == 0) first = got;
        else if (std::memcmp(first.data(), got.data(), got.size() * 4) != 0) { printf("  ⚠️ %s result differs between repetitions\n", V.name); vr.exact = false; }
      }
    }
    CUDA_CHECK(cudaGetLastError());
    vr.us = t_new / nt * 1000.0;
    CUDA_CHECK(cudaMemcpy(got.data(), B.acc, got.size() * 4, cudaMemcpyDeviceToHost));
    double max_abs = 0;
    for (size_t i = 0; i < ref.size(); i++)
      if (std::memcmp(&ref[i], &got[i], 4) != 0) { ++vr.mism_acc; max_abs = std::max(max_abs, (double)std::fabs(ref[i] - got[i])); }
    vr.mism_q = cmp_dev(A.yq, B.yq, (size_t)goff * I) + cmp_dev(A.ys, B.ys, (size_t)goff * I / 32) + cmp_dev(A.eout, B.eout, (size_t)goff * dim * 2) +
                cmp_dev(A.y, B.y, (size_t)goff * I * 2);
    if (vi == vidx("D1")) acc_d1 = got;
    if (vi == vidx("F2s4")) acc_f2 = got;
    if (V.f3 > 0) {  // F3 ↔ D1 · F2s4 directly (all acc bytes)
      if (!acc_d1.empty()) vr.eq_d1 = std::memcmp(acc_d1.data(), got.data(), got.size() * 4) == 0;
      if (!acc_f2.empty()) vr.eq_f2 = std::memcmp(acc_f2.data(), got.data(), got.size() * 4) == 0;
      vr.exact = vr.exact && vr.eq_d1 != 0 && vr.eq_f2 != 0;
    }
    vr.exact = vr.exact && finite && vr.mism_acc == 0 && vr.mism_q == 0;
    if (vr.mism_acc) printf("  %s: acc mismatches %zu (max abs %.3e)\n", V.name, vr.mism_acc, max_abs);
    if (vr.mism_q) printf("  %s: y/yq/ys/eout byte mismatches %zu\n", V.name, vr.mism_q);
    res.exact = res.exact && vr.exact;
  }
  k::moe_decode_fused_force(-1);
  for (Bufs* b : {&A, &B}) { cudaFree(b->xq); cudaFree(b->xs); cudaFree(b->yq); cudaFree(b->ys); cudaFree(b->y); cudaFree(b->eout); cudaFree(b->acc); cudaFree(b->acc_init); cudaFree(b->tbl); }
  cudaFree(dcnt);
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(st);
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::max(2, atoi(argv[1])) : 20;
  int NE = argc > 2 ? std::max(12, atoi(argv[2])) : 96;
  const ExpertLayout lay = ExpertLayout::make(DIM, INTER);
  const size_t flush_n = (size_t)256 << 20;  // larger than L2 (128 MB)
  size_t vfree = 0, vtot = 0;
  CUDA_CHECK(cudaMemGetInfo(&vfree, &vtot));
  const size_t reserve = flush_n + ((size_t)64 << 20);
  const int fit = vfree > reserve ? (int)((vfree - reserve) / lay.total) : 0;
  if (fit < NE) { printf("VRAM free %.0f MiB → experts %d → %d\n", vfree / 1048576.0, NE, fit); NE = fit; }
  if (NE < 12) { printf("RESULT: FAIL (not enough VRAM for 12 expert records)\n"); return 2; }
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("device %s · SMs %d · L2 %d MB · experts %d × %.1f MB (dim %d, inter %d, top-%d)\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, NE,
         lay.total / 1e6, DIM, INTER, TOPK);
  std::vector<uint8_t*> rec(NE);
  for (int e = 0; e < NE; ++e) {
    rec[e] = dmalloc<uint8_t>(lay.total);
    fill(rec[e] + lay.w1, (size_t)INTER * DIM / 2, 1000 + e, 0); fill(rec[e] + lay.s1, (size_t)INTER * DIM / 32, 2000 + e, 3);
    fill(rec[e] + lay.w3, (size_t)INTER * DIM / 2, 3000 + e, 0); fill(rec[e] + lay.s3, (size_t)INTER * DIM / 32, 4000 + e, 3);
    fill(rec[e] + lay.w2, (size_t)DIM * INTER / 2, 5000 + e, 0); fill(rec[e] + lay.s2, (size_t)DIM * INTER / 32, 6000 + e, 3);
  }
  uint8_t* flush = dmalloc<uint8_t>(flush_n);
  CUDA_CHECK(cudaDeviceSynchronize());
  struct Case { int M, pool; };
  const Case cases[] = {{1, NE}, {2, NE}, {4, NE}, {6, NE}, {8, NE}, {8, 12}};
  bool ok = true;
  printf("%-4s %-5s %-7s %-7s %-9s %-10s %-10s %-9s %-8s %-9s %s\n", "M", "pool", "experts", "groups", "path", "µs", "GB/s", "vs old", "vs D1", "vs F2s4", "verdict");
  std::vector<std::string> summary;
  for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
    const Case& cs = cases[ci];
    const Result r = run_case(rec, lay, cs.M, cs.pool, reps, 20260930u + 31u * (uint32_t)ci, flush, flush_n);
    auto gbs = [&](double us) { return r.mb / std::max(1e-9, us) * 1e3; };
    printf("%-4d %-5d %-7d %-7d %-9s %-10.1f %-10.0f %-9s %-8s %-9s %s\n", cs.M, cs.pool, r.n_exp, r.ng, "old", r.us_old, gbs(r.us_old), "", "", "", "(reference)");
    int best = 0;
    for (int vi = 0; vi < NV; ++vi) {
      const VRes& v = r.v[vi];
      const double d1 = r.v[kVariants[vi].dev ? 1 : 0].us;  // vs D1 with the same launch mode (host/DEV)
      const double f2 = r.v[vidx(kVariants[vi].dev ? "F2s4-dev" : "F2s4")].us;  // vs F2s4 with the same launch mode
      char eq[48] = "";
      if (kVariants[vi].f3 > 0) snprintf(eq, sizeof eq, " ==D1 %s ==F2s4 %s", v.eq_d1 == 1 ? "yes" : "NO", v.eq_f2 == 1 ? "yes" : "NO");
      printf("%-4s %-5s %-7s %-7s %-9s %-10.1f %-10.0f ×%-8.2f ×%-7.2f ×%-8.2f %s%s\n", "", "", "", "", kVariants[vi].name, v.us, gbs(v.us), r.us_old / std::max(1e-9, v.us),
             d1 / std::max(1e-9, v.us), f2 / std::max(1e-9, v.us), v.exact ? "PASS (bit-exact)" : "FAIL", eq);
      if (v.us < r.v[best].us) best = vi;
    }
    char line[160];
    snprintf(line, sizeof line, "M=%d pool %d: fastest %s %.1f µs %.0f GB/s (D1 %.0f · F2s4 %.0f · F3s4 %.0f GB/s)", cs.M, cs.pool, kVariants[best].name, r.v[best].us,
             gbs(r.v[best].us), gbs(r.v[vidx("D1")].us), gbs(r.v[vidx("F2s4")].us), gbs(r.v[vidx("F3s4")].us));
    summary.push_back(line);
    ok &= r.exact;
  }
  for (auto* p : rec) cudaFree(p);
  cudaFree(flush);
  printf("summary (host + dev variants):\n");
  for (const auto& l : summary) printf("  %s\n", l.c_str());
  printf("RESULT: %s\n", ok ? "PASS (all bit-exact)" : "FAIL");
  return ok ? 0 : 1;
}
