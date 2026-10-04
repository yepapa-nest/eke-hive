// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM decode kernels (glm_decode.h) vs the current composition in glm_engine.cpp, M ∈ {1, 2, 3, 4, 8}:
//   (1) hc pre-stage: glm_hc_pre_decode vs k::hc_mix → k::hc_split_sinkhorn → k::hc_pre_norm — byte equality of mixes, rsq, pre, post, comb,
//       x, xn (Sinkhorn scale 1 and 5); fused form (post_x) vs k::hc_post + the three kernels — byte equality incl. h; glm_hc_post_decode vs
//       k::hc_post (bytes); sync words back at 0.
//   (2) glm_gemv_bf16_multi (KDA q|k|v|f_a|g_a|b set, f_b|g_b set, o_proj) vs an fp64 CPU reference over ALL outputs and vs cuBLAS's
//       fp32-output GEMM, both with the fp32-order error budget ulp_bf16 + K·2^-24·Σ|x·w|; the bf16-output cuBLAS GEMM the engine uses today
//       is reported against the same fp64 reference (information).
//   (3) glm_router_decode vs cuBLAS fp32-out GEMM + router_topk: identical ids, weights to 1e-5, host-mapped copies equal, router_topk on
//       this kernel's logits bit-identical to its own ids/weights, and a tie case (every expert row duplicated, equal bias) with identical ids.
//   (4) glm_dense_nvfp4_decode: y and out vs an fp64 CPU reference with a propagated error budget (fp32 order + f16 activation rounding of
//       the tensor-core path; out against the path's own y); the old composition (M ≤ 4 — it aborts above) gets the same check; includes
//       activations outside the f16 range (exact CUDA-core fallback).
//   (5) glm_shared_fp8_decode vs fp8b_gemv ×2 / swiglu_rows / fp8b_gemv — byte equality.
//   Every new kernel is run twice (byte-identical = deterministic).
// Timing: CUDA events around the op, the L2 flushed before every iteration by READING a 256 MB buffer; median of 25.
// Negative controls: build glm_decode.cu with one of -DHIVE_GLM_DEC_NEG_{SINKHORN,POST,STRIDE,TIE,NVFP4,FP8} → this test must FAIL.
// Run: test_glm_decode [hc,gemv,router,dense,shared]   (default: all)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include <cuda_fp16.h>

#include "hive/common.h"
#include "hive/cublas_ops.h"
#include "hive/glm/fp8b.h"
#include "hive/glm/glm_decode.h"
#include "hive/glm/glm_kernels.h"
#include "hive/glm/glm_moe.h"
#include "hive/glm/nvfp4.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"

using namespace hive;
using namespace hive::glm;

namespace {
int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
void verdict(bool ok, const char* what) { printf("  %-74s %s\n", what, ok ? "PASS" : "FAIL"); if (!ok) ++fails; }

__device__ __forceinline__ uint32_t hsh(size_t i, uint32_t seed) {
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed ^ (uint32_t)(i >> 32) * 0x9E3779B9u;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  return h;
}
__device__ __forceinline__ float uni(size_t i, uint32_t seed) { return (float)(hsh(i, seed) >> 8) * (2.f / 16777216.f) - 1.f; }  // [-1, 1)
__global__ void fill_bf16_k(bf16* p, size_t n, uint32_t seed, float a, float b) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = f2bf(a + b * uni(i, seed));
}
__global__ void fill_f32_k(float* p, size_t n, uint32_t seed, float a, float b) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = a + b * uni(i, seed);
}
__global__ void fill_u8_k(uint8_t* p, size_t n, uint32_t seed, int lo, int span, int mode) {  // mode 0: any byte · 1: lo + h%span · 2: e4m3 non-NaN
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t h = hsh(i, seed);
    uint8_t v = (uint8_t)(h >> 8);
    if (mode == 1) v = (uint8_t)(lo + (h >> 8) % span);
    if (mode == 2 && (v & 0x7F) == 0x7F) v &= 0xFE;
    p[i] = v;
  }
}
__global__ void flush_read(const uint4* __restrict__ p, size_t n, unsigned* o) {
  unsigned a = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const uint4 v = p[i]; a ^= v.x ^ v.w; }
  if (a == 0x9E3779B9u) *o = a;
}
void fill_bf16(bf16* p, size_t n, uint32_t s, float a, float b) { fill_bf16_k<<<2048, 256>>>(p, n, s, a, b); CUDA_CHECK(cudaGetLastError()); }
void fill_f32(float* p, size_t n, uint32_t s, float a, float b) { fill_f32_k<<<2048, 256>>>(p, n, s, a, b); CUDA_CHECK(cudaGetLastError()); }
void fill_u8(uint8_t* p, size_t n, uint32_t s, int lo, int span, int mode) { fill_u8_k<<<2048, 256>>>(p, n, s, lo, span, mode); CUDA_CHECK(cudaGetLastError()); }
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T))); CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T))); return p; }
template <class T> std::vector<T> host(const T* d, size_t n) { std::vector<T> v(n); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
template <class T> bool same_bytes(const T* a, const T* b, size_t n) { auto x = host(a, n), y = host(b, n); return memcmp(x.data(), y.data(), n * sizeof(T)) == 0; }
float bf(bf16 v) { return bf2f(v); }
double ulp_bf16(double a) {  // spacing of bf16 values at |a|
  a = std::fabs(a);
  if (a < 1e-30) return 1e-38;
  int e; std::frexp(a, &e);  // a = f · 2^e, f in [0.5, 1)
  return std::ldexp(1.0, e - 8);
}
struct UlpStat { double max_ulp = 0, max_rel = 0; size_t diff = 0, n = 0; };
// distance of a to b in bf16 steps at max(|a|, |b|); max relative error over elements with |b| ≥ 1e-3·rms(b)
UlpStat ulp_cmp(const std::vector<bf16>& a, const std::vector<bf16>& b) {
  UlpStat s; s.n = a.size();
  double rms = 0;
  for (auto v : b) rms += (double)bf(v) * bf(v);
  rms = std::sqrt(rms / std::max<size_t>(1, b.size()));
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = bf(a[i]), y = bf(b[i]);
    if (!std::isfinite(x) || !std::isfinite(y)) { s.max_ulp = 1e30; continue; }
    if (x != y) ++s.diff;
    s.max_ulp = std::max(s.max_ulp, std::fabs(x - y) / ulp_bf16(std::max(std::fabs(x), std::fabs(y))));
    if (std::fabs(y) >= 1e-3 * rms) s.max_rel = std::max(s.max_rel, std::fabs(x - y) / std::fabs(y));
  }
  return s;
}

cudaStream_t g_st;
void* g_flush;
const size_t kFlushB = 256u << 20;
cudaEvent_t g_ea, g_eb;
float time_med(const std::function<void()>& f, int reps = 25) {
  std::vector<float> v;
  f(); CUDA_CHECK(cudaStreamSynchronize(g_st));  // warm-up (first-launch costs)
  for (int r = 0; r < reps; ++r) {
    flush_read<<<1024, 256, 0, g_st>>>(reinterpret_cast<const uint4*>(g_flush), kFlushB / 16, reinterpret_cast<unsigned*>(g_flush));
    CUDA_CHECK(cudaEventRecord(g_ea, g_st));
    f();
    CUDA_CHECK(cudaEventRecord(g_eb, g_st));
    CUDA_CHECK(cudaEventSynchronize(g_eb));
    float t; CUDA_CHECK(cudaEventElapsedTime(&t, g_ea, g_eb));
    v.push_back(t * 1000.f);
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
struct TRow { const char* what; int M; float old_us, new_us; double mb; };
std::vector<TRow> g_rows;
void trow(const char* what, int M, float o, float n, double mb) { g_rows.push_back({what, M, o, n, mb}); }

const int H = 4096, HCD = 4 * H, MIX = 24, ITERS = 20;
const float RMS_EPS = 1e-5f, HC_EPS = 1e-6f, LIMIT = 10.f;
const int Ms[] = {1, 2, 3, 4, 8};
bool timed_M(int M) { return M == 1 || M == 4 || M == 8; }

// ===================================================================================================================================
void test_hc(int* sync) {
  printf("[1] hc pre / post (bit identity with k::hc_mix → k::hc_split_sinkhorn → k::hc_pre_norm, k::hc_post)\n");
  float* fn = dmalloc<float>((size_t)MIX * HCD); float* scale = dmalloc<float>(3); float* base = dmalloc<float>(MIX); bf16* nw = dmalloc<bf16>(H);
  fill_f32(fn, (size_t)MIX * HCD, 11, 0.f, 0.02f); fill_f32(base, MIX, 13, 0.f, 1.f); fill_bf16(nw, H, 14, 1.f, 0.1f);
  struct O { float *mixes, *rsq, *pre, *post, *comb; bf16 *x, *xn, *h; };
  auto mk = [] { O o; o.mixes = dmalloc<float>(8 * MIX); o.rsq = dmalloc<float>(8); o.pre = dmalloc<float>(32); o.post = dmalloc<float>(32); o.comb = dmalloc<float>(128);
                 o.x = dmalloc<bf16>(8 * H); o.xn = dmalloc<bf16>(8 * H); o.h = dmalloc<bf16>((size_t)8 * HCD); return o; };
  O a = mk(), b = mk();
  bf16* h0 = dmalloc<bf16>((size_t)8 * HCD); bf16* px = dmalloc<bf16>(8 * H);
  float* post0 = dmalloc<float>(32); float* comb0 = dmalloc<float>(128);
  auto eq = [&](const O& p, const O& q, int M, bool with_h) {
    bool ok = same_bytes(p.mixes, q.mixes, M * MIX) && same_bytes(p.rsq, q.rsq, M) && same_bytes(p.pre, q.pre, M * 4) && same_bytes(p.post, q.post, M * 4) &&
              same_bytes(p.comb, q.comb, M * 16) && same_bytes(p.x, q.x, (size_t)M * H) && same_bytes(p.xn, q.xn, (size_t)M * H);
    if (with_h) ok = ok && same_bytes(p.h, q.h, (size_t)M * HCD);
    return ok;
  };
  auto old_pre = [&](const O& o, int M) {
    k::hc_mix(o.h, fn, M, HCD, MIX, RMS_EPS, o.mixes, o.rsq, g_st);
    k::hc_split_sinkhorn(o.mixes, o.rsq, scale, base, M, 4, ITERS, HC_EPS, o.pre, o.post, o.comb, g_st);
    k::hc_pre_norm(o.h, o.pre, M, 4, H, nw, RMS_EPS, o.x, o.xn, g_st);
  };
  auto new_pre = [&](const O& o, int M, const bf16* pxp) {
    glm_hc_pre_decode(o.h, M, H, fn, scale, base, nw, RMS_EPS, HC_EPS, ITERS, o.mixes, o.rsq, o.pre, o.post, o.comb, o.x, o.xn, sync, g_st, pxp);
  };
  auto reset = [&](O& o, int M) {  // h = h0, post/comb = previous-stage values
    CUDA_CHECK(cudaMemcpyAsync(o.h, h0, (size_t)M * HCD * 2, cudaMemcpyDeviceToDevice, g_st));
    CUDA_CHECK(cudaMemcpyAsync(o.post, post0, 32 * 4, cudaMemcpyDeviceToDevice, g_st));
    CUDA_CHECK(cudaMemcpyAsync(o.comb, comb0, 128 * 4, cudaMemcpyDeviceToDevice, g_st));
  };
  for (const float smag : {1.f, 5.f}) {
    fill_f32(scale, 3, 12 + (uint32_t)smag, smag, 0.5f * smag);
    fill_bf16(h0, (size_t)8 * HCD, 21 + (uint32_t)smag, 0.f, 2.f); fill_bf16(px, 8 * H, 22, 0.f, 1.f);
    // previous-stage post/comb: a real Sinkhorn output
    {
      float* mx = dmalloc<float>(8 * MIX); float* rq = dmalloc<float>(8); float* pr = dmalloc<float>(32);
      k::hc_mix(h0, fn, 8, HCD, MIX, RMS_EPS, mx, rq, g_st);
      k::hc_split_sinkhorn(mx, rq, scale, base, 8, 4, ITERS, HC_EPS, pr, post0, comb0, g_st);
      CUDA_CHECK(cudaStreamSynchronize(g_st));
      cudaFree(mx); cudaFree(rq); cudaFree(pr);
    }
    for (const int M : Ms) {
      char msg[160];
      // plain pre
      reset(a, M); reset(b, M);
      old_pre(a, M); new_pre(b, M, nullptr);
      const bool e1 = eq(a, b, M, true);
      new_pre(b, M, nullptr);
      const bool e2 = eq(a, b, M, true);
      const auto sy = host(sync, kGlmDecodeSyncInts);
      const bool s0 = std::all_of(sy.begin(), sy.end(), [](int v) { return v == 0; });
      snprintf(msg, sizeof msg, "pre  scale %.0f M=%d: bytes equal (run 1, run 2), sync words 0", smag, M);
      verdict(e1 && e2 && s0, msg);
      // fused post + pre
      reset(a, M); reset(b, M);
      k::hc_post(px, a.post, a.comb, M, 4, H, a.h, g_st); old_pre(a, M);
      new_pre(b, M, px);
      const bool f1 = eq(a, b, M, true);
      reset(b, M); new_pre(b, M, px);
      const bool f2 = eq(a, b, M, true);
      snprintf(msg, sizeof msg, "post+pre scale %.0f M=%d: bytes equal incl. h (run 1, run 2)", smag, M);
      verdict(f1 && f2, msg);
      // standalone post
      reset(a, M); reset(b, M);
      k::hc_post(px, a.post, a.comb, M, 4, H, a.h, g_st);
      glm_hc_post_decode(px, b.post, b.comb, M, H, b.h, g_st);
      snprintf(msg, sizeof msg, "post scale %.0f M=%d: h bytes equal", smag, M);
      verdict(same_bytes(a.h, b.h, (size_t)M * HCD), msg);
      if (smag == 1.f && timed_M(M)) {
        const double mb = (MIX * HCD * 4.0 + M * HCD * 2.0) / 1e6;
        trow("hc pre (mix+sinkhorn+pre_norm)", M, time_med([&] { old_pre(a, M); }), time_med([&] { new_pre(b, M, nullptr); }), mb);
        trow("hc post", M, time_med([&] { k::hc_post(px, a.post, a.comb, M, 4, H, a.h, g_st); }),
             time_med([&] { glm_hc_post_decode(px, b.post, b.comb, M, H, b.h, g_st); }), M * (2.0 * HCD * 2 + H * 2) / 1e6);
        trow("hc post + next pre (fused)", M, time_med([&] { k::hc_post(px, a.post, a.comb, M, 4, H, a.h, g_st); old_pre(a, M); }),
             time_med([&] { new_pre(b, M, px); }), mb);
      }
    }
  }
  for (void* p : {(void*)fn, (void*)scale, (void*)base, (void*)nw, (void*)h0, (void*)px, (void*)post0, (void*)comb0}) cudaFree(p);
}

// ===================================================================================================================================
// fp64 dot of bf16 rows
double dot_bf(const bf16* a, const bf16* b, int K, double* abs_sum = nullptr) {
  double s = 0, sa = 0;
  for (int k = 0; k < K; ++k) { const double p = (double)bf(a[k]) * bf(b[k]); s += p; sa += std::fabs(p); }
  if (abs_sum) *abs_sum = sa;
  return s;
}
// Error budget of an fp32 dot product in ANY summation order, then one rounding to bf16: |y − exact| ≤ ulp_bf16 + γ_K·Σ|x·w|,
//   γ_K = K·2^-24 (standard worst-case bound). Returns max |a − b| / budget (≤ 1 ⇔ the difference is explainable by fp32 order alone);
//   `scale` widens the budget when b is itself an fp32-order result (2: both sides carry the error).
double order_ratio(const std::vector<bf16>& a, const std::vector<double>& b, const std::vector<double>& absum, int K, double scale = 1.0) {
  double r = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double budget = scale * (ulp_bf16(b[i]) + K * std::ldexp(1.0, -24) * absum[i]);
    r = std::max(r, std::fabs((double)bf(a[i]) - b[i]) / budget);
  }
  return r;
}

void test_gemv(Blas& blas) {
  printf("[2] BF16 GEMV segment table vs cuBLAS bf16 GEMM\n");
  const int W = 8192, R = 128, NB = 64;
  bf16 *wq = dmalloc<bf16>((size_t)W * H), *wk = dmalloc<bf16>((size_t)W * H), *wv = dmalloc<bf16>((size_t)W * H);
  bf16 *wfa = dmalloc<bf16>((size_t)R * H), *wga = dmalloc<bf16>((size_t)R * H), *wb = dmalloc<bf16>((size_t)NB * H);
  bf16 *wfb = dmalloc<bf16>((size_t)W * R), *wgb = dmalloc<bf16>((size_t)W * R), *wo = dmalloc<bf16>((size_t)H * W);
  uint32_t sd = 100;
  for (auto [p, n] : {std::pair<bf16*, size_t>{wq, (size_t)W * H}, {wk, (size_t)W * H}, {wv, (size_t)W * H}, {wfa, (size_t)R * H}, {wga, (size_t)R * H},
                      {wb, (size_t)NB * H}, {wfb, (size_t)W * R}, {wgb, (size_t)W * R}, {wo, (size_t)H * W}})
    fill_bf16(p, n, sd++, 0.f, 0.03f);
  bf16* xn = dmalloc<bf16>(8 * H); fill_bf16(xn, 8 * H, 7, 0.f, 1.5f);
  bf16* xo = dmalloc<bf16>(8 * W); fill_bf16(xo, 8 * W, 8, 0.f, 1.f);
  struct Out { bf16 *qkv, *fa, *ga, *b, *fb, *gb, *o; };
  auto mk = [&] { Out o; o.qkv = dmalloc<bf16>(8 * 3 * W); o.fa = dmalloc<bf16>(8 * R); o.ga = dmalloc<bf16>(8 * R); o.b = dmalloc<bf16>(8 * NB);
                  o.fb = dmalloc<bf16>(8 * W); o.gb = dmalloc<bf16>(8 * W); o.o = dmalloc<bf16>(8 * H); return o; };
  Out a = mk(), b = mk(), c = mk();
  auto old1 = [&](Out& o, int M) {
    blas.gemm_bf16_batched_ld(xn, H, 0, wq, H, 0, o.qkv, 3 * W, 0, M, W, H, 1);
    blas.gemm_bf16_batched_ld(xn, H, 0, wk, H, 0, o.qkv + W, 3 * W, 0, M, W, H, 1);
    blas.gemm_bf16_batched_ld(xn, H, 0, wv, H, 0, o.qkv + 2 * W, 3 * W, 0, M, W, H, 1);
    blas.gemm_bf16(xn, wfa, o.fa, M, R, H);
    blas.gemm_bf16(xn, wga, o.ga, M, R, H);
    blas.gemm_bf16(xn, wb, o.b, M, NB, H);
  };
  auto old2 = [&](Out& o, const bf16* fa, const bf16* ga, int M) { blas.gemm_bf16(fa, wfb, o.fb, M, W, R); blas.gemm_bf16(ga, wgb, o.gb, M, W, R); };
  auto oldo = [&](Out& o, int M) { blas.gemm_bf16(xo, wo, o.o, M, H, W); };
  auto new1 = [&](Out& o, int M) {
    const GemvSeg s[6] = {{wq, xn, o.qkv, W, H, H, 3 * W}, {wk, xn, o.qkv + W, W, H, H, 3 * W}, {wv, xn, o.qkv + 2 * W, W, H, H, 3 * W},
                          {wfa, xn, o.fa, R, H, H, R}, {wga, xn, o.ga, R, H, H, R}, {wb, xn, o.b, NB, H, H, NB}};
    glm_gemv_bf16_multi(s, 6, M, g_st);
  };
  auto new2 = [&](Out& o, const bf16* fa, const bf16* ga, int M) {
    const GemvSeg s[2] = {{wfb, fa, o.fb, W, R, R, W}, {wgb, ga, o.gb, W, R, R, W}};
    glm_gemv_bf16_multi(s, 2, M, g_st);
  };
  auto newo = [&](Out& o, int M) { const GemvSeg s{wo, xo, o.o, H, W, W, H}; glm_gemv_bf16_multi(&s, 1, M, g_st); };
  // References: (A) fp64 CPU dot over ALL outputs, (B) cuBLAS fp32-output GEMM rounded once to bf16 (= fp32 accumulation, other order).
  //   The bf16-output cuBLAS GEMM the engine uses today (C) is reported against (A) as information: at these decode shapes it is NOT a
  //   single rounding of an fp32 sum (measured below), so it is not a valid "fp32-order noise" reference by itself.
  const auto hx = host(xn, 8 * H), hxo = host(xo, 8 * W);
  struct Seg { const char* n; const bf16* w; int N, K; };
  const Seg segs[] = {{"q", wq, W, H}, {"k", wk, W, H}, {"v", wv, W, H}, {"f_a", wfa, R, H}, {"g_a", wga, R, H}, {"b", wb, NB, H}, {"o_proj", wo, H, W}};
  std::vector<std::vector<double>> ref(7), absr(7);  // [seg][m·N + n] for 8 rows: exact value, Σ|x·w|
  for (int si = 0; si < 7; ++si) {
    const auto hw = host(segs[si].w, (size_t)segs[si].N * segs[si].K);
    const bf16* xx = si == 6 ? hxo.data() : hx.data();
    ref[si].resize((size_t)8 * segs[si].N); absr[si].resize(ref[si].size());
    for (int m = 0; m < 8; ++m)
      for (int n = 0; n < segs[si].N; ++n) {
        const size_t i = (size_t)m * segs[si].N + n;
        ref[si][i] = dot_bf(xx + (size_t)m * segs[si].K, hw.data() + (size_t)n * segs[si].K, segs[si].K, &absr[si][i]);
      }
  }
  const auto hfb = host(wfb, (size_t)W * R), hgb = host(wgb, (size_t)W * R);
  float* f32 = dmalloc<float>((size_t)8 * W);
  for (const int M : Ms) {
    old1(a, M); old2(a, a.fa, a.ga, M); oldo(a, M);
    new1(b, M); new2(b, a.fa, a.ga, M); newo(b, M);   // set 2 on the SAME inputs as cuBLAS (isolates this kernel)
    new1(c, M); new2(c, a.fa, a.ga, M); newo(c, M);   // determinism
    CUDA_CHECK(cudaStreamSynchronize(g_st));
    auto take = [&](const Out& o, int si) {  // bf16 [M, N] of segment si
      std::vector<bf16> v((size_t)M * segs[si].N);
      const bf16* base = si < 3 ? o.qkv + si * W : si == 3 ? o.fa : si == 4 ? o.ga : si == 5 ? o.b : o.o;
      const int ld = si < 3 ? 3 * W : segs[si].N;
      for (int m = 0; m < M; ++m) CUDA_CHECK(cudaMemcpy(v.data() + (size_t)m * segs[si].N, base + (size_t)m * ld, segs[si].N * 2, cudaMemcpyDeviceToHost));
      return v;
    };
    for (int si = 0; si < 7; ++si) {
      const int N = segs[si].N;
      std::vector<bf16> rA((size_t)M * N);
      for (size_t i = 0; i < rA.size(); ++i) rA[i] = f2bf((float)ref[si][i]);
      if (si == 6) blas.gemm_bf16_f32out(xo, wo, f32, M, N, W); else blas.gemm_bf16_f32out(xn, segs[si].w, f32, M, N, H);
      const auto hf = host(f32, (size_t)M * N);
      std::vector<bf16> rB(hf.size());
      for (size_t i = 0; i < hf.size(); ++i) rB[i] = f2bf(hf[i]);
      const auto nb = take(b, si), nc2 = take(c, si), ob = take(a, si);
      const int K = segs[si].K;
      const std::vector<double> rv(ref[si].begin(), ref[si].begin() + (size_t)M * N), av(absr[si].begin(), absr[si].begin() + (size_t)M * N);
      std::vector<double> bv(hf.begin(), hf.end());
      const double rA_ = order_ratio(nb, rv, av, K), rB_ = order_ratio(nb, bv, av, K, 2.0), rC_ = order_ratio(ob, rv, av, K);
      const auto eA = ulp_cmp(nb, rA), eB = ulp_cmp(nb, rB), eC = ulp_cmp(ob, rA);
      size_t offC = 0;
      for (size_t i = 0; i < ob.size(); ++i) offC += std::fabs(bf(ob[i]) - rv[i]) > ulp_bf16(rv[i]) + K * std::ldexp(1.0, -24) * av[i];
      const bool det = memcmp(nb.data(), nc2.data(), nb.size() * 2) == 0;
      char msg[240];
      snprintf(msg, sizeof msg, "M=%d %-6s new/fp64 %.2f st (budget %.2f) · new/cuBLAS-f32out %.0f st (budget %.2f, %zu/%zu differ, rel %.1e) · det", M,
               segs[si].n, eA.max_ulp, rA_, eB.max_ulp, rB_, eB.diff, eB.n, eB.max_rel);
      verdict(rA_ <= 1.0 && rB_ <= 1.0 && det, msg);
      printf("      engine path today (cuBLAS bf16-out) vs fp64: max %.0f steps, budget ratio %.2f, %zu/%zu elements outside the fp32-order budget\n", eC.max_ulp,
             rC_, offC, eC.n);
    }
    {  // f_b / g_b on cuBLAS's f_a / g_a (fp64 reference on the same inputs)
      const auto hfa = host(a.fa, (size_t)M * R), hga = host(a.ga, (size_t)M * R);
      const auto nfb = host(b.fb, (size_t)M * W), ngb = host(b.gb, (size_t)M * W), cfb = host(c.fb, (size_t)M * W), cgb = host(c.gb, (size_t)M * W);
      std::vector<double> rfb((size_t)M * W), rgb((size_t)M * W), afb((size_t)M * W), agb((size_t)M * W);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < W; ++n) {
          const size_t i = (size_t)m * W + n;
          rfb[i] = dot_bf(hfa.data() + (size_t)m * R, hfb.data() + (size_t)n * R, R, &afb[i]);
          rgb[i] = dot_bf(hga.data() + (size_t)m * R, hgb.data() + (size_t)n * R, R, &agb[i]);
        }
      const double e1 = order_ratio(nfb, rfb, afb, R), e2 = order_ratio(ngb, rgb, agb, R);
      const bool det = memcmp(nfb.data(), cfb.data(), nfb.size() * 2) == 0 && memcmp(ngb.data(), cgb.data(), ngb.size() * 2) == 0;
      char msg[200];
      snprintf(msg, sizeof msg, "M=%d f_b|g_b (K=128, 2 rows/warp) vs fp64: budget ratio %.2f / %.2f · det", M, e1, e2);
      verdict(e1 <= 1.0 && e2 <= 1.0 && det, msg);
    }
    if (timed_M(M)) {
      const double mb1 = ((3.0 * W + 2 * R + NB) * H * 2) / 1e6, mb2 = 2.0 * W * R * 2 / 1e6, mbo = (double)H * W * 2 / 1e6;
      trow("KDA q|k|v|f_a|g_a|b (6 cuBLAS → 1)", M, time_med([&] { old1(a, M); }), time_med([&] { new1(b, M); }), mb1);
      trow("KDA f_b|g_b (2 cuBLAS → 1)", M, time_med([&] { old2(a, a.fa, a.ga, M); }), time_med([&] { new2(b, b.fa, b.ga, M); }), mb2);
      trow("KDA all input projections (8 → 2)", M, time_med([&] { old1(a, M); old2(a, a.fa, a.ga, M); }),
           time_med([&] { new1(b, M); new2(b, b.fa, b.ga, M); }), mb1 + mb2);
      trow("KDA o_proj [4096,8192]", M, time_med([&] { oldo(a, M); }), time_med([&] { newo(b, M); }), mbo);
    }
  }
  cudaFree(f32);
  for (void* p : {(void*)wq, (void*)wk, (void*)wv, (void*)wfa, (void*)wga, (void*)wb, (void*)wfb, (void*)wgb, (void*)wo, (void*)xn, (void*)xo}) cudaFree(p);
}

// ===================================================================================================================================
void test_router(Blas& blas) {
  printf("[3] router logits + top-k vs cuBLAS fp32-out GEMM + router_topk\n");
  const int E = 288, KK = 8;
  const float SCALE = 2.5f;
  bf16* gw = dmalloc<bf16>((size_t)E * H); float* bias = dmalloc<float>(E); bf16* x = dmalloc<bf16>(8 * H);
  float *lo = dmalloc<float>(8 * E), *ln = dmalloc<float>(glm_router_logits_floats(8, E, H)), *wo = dmalloc<float>(8 * KK), *wn = dmalloc<float>(8 * KK), *wr = dmalloc<float>(8 * KK);
  int32_t *io = dmalloc<int32_t>(8 * KK), *in = dmalloc<int32_t>(8 * KK), *ir = dmalloc<int32_t>(8 * KK);
  int32_t* ih; float* wh;
  CUDA_CHECK(cudaHostAlloc((void**)&ih, 8 * KK * 4, cudaHostAllocMapped)); CUDA_CHECK(cudaHostAlloc((void**)&wh, 8 * KK * 4, cudaHostAllocMapped));
  int32_t* ihd; float* whd;
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&ihd, ih, 0)); CUDA_CHECK(cudaHostGetDevicePointer((void**)&whd, wh, 0));
  std::vector<int32_t> h_ids(8 * KK); std::vector<float> h_w(8 * KK);
  auto oldr = [&](int M, bool d2h) {
    blas.gemm_bf16_f32out(x, gw, lo, M, E, H);
    router_topk(lo, bias, M, E, KK, SCALE, io, wo, g_st);
    if (d2h) {
      CUDA_CHECK(cudaMemcpyAsync(h_ids.data(), io, (size_t)M * KK * 4, cudaMemcpyDeviceToHost, g_st));
      CUDA_CHECK(cudaMemcpyAsync(h_w.data(), wo, (size_t)M * KK * 4, cudaMemcpyDeviceToHost, g_st));
    }
  };
  auto newr = [&](int M) { glm_router_decode(x, H, M, gw, bias, E, H, KK, SCALE, ln, in, wn, ihd, whd, g_st); };
  for (int tie = 0; tie < 2; ++tie) {
    fill_bf16(gw, (size_t)E * H, 31, 0.f, 0.05f); fill_f32(bias, E, 32, 0.f, 0.02f); fill_bf16(x, 8 * H, 33 + tie, 0.f, 1.f);
    if (tie) {  // every odd expert row = its even neighbour, same bias → every score has an exact twin
      for (int e = 0; e < E; e += 2) {
        CUDA_CHECK(cudaMemcpy(gw + (size_t)(e + 1) * H, gw + (size_t)e * H, H * 2, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(bias + e + 1, bias + e, 4, cudaMemcpyDeviceToDevice));
      }
    }
    for (const int M : Ms) {
      oldr(M, false); newr(M);
      CUDA_CHECK(cudaStreamSynchronize(g_st));
      const auto a_ids = host(io, M * KK), b_ids = host(in, M * KK);
      const auto a_w = host(wo, M * KK), b_w = host(wn, M * KK);
      double wmax = 0;
      for (int i = 0; i < M * KK; ++i) wmax = std::max(wmax, (double)std::fabs(a_w[i] - b_w[i]) / std::max(1e-30f, std::fabs(a_w[i])));
      const bool host_ok = memcmp(ih, b_ids.data(), M * KK * 4) == 0 && memcmp(wh, b_w.data(), M * KK * 4) == 0;
      // top-k semantics: router_topk on this kernel's logits must give the same bits
      router_topk(ln, bias, M, E, KK, SCALE, ir, wr, g_st);
      const bool topk_same = same_bytes(ir, in, M * KK) && same_bytes(wr, wn, M * KK);
      // logits vs cuBLAS
      const auto la = host(lo, M * E), lb = host(ln, M * E);
      double lmax = 0, lsc = 0;
      for (int i = 0; i < M * E; ++i) { lmax = std::max(lmax, (double)std::fabs(la[i] - lb[i])); lsc = std::max(lsc, (double)std::fabs(la[i])); }
      bool twins = true;
      if (tie) for (int m = 0; m < M; ++m) for (int e = 0; e < E; e += 2) twins = twins && la[m * E + e] == la[m * E + e + 1] && lb[m * E + e] == lb[m * E + e + 1];
      // determinism
      auto ln1 = host(ln, M * E); newr(M); const bool det = same_bytes(in, ir, M * KK) && host(ln, M * E) == ln1;
      const bool s0 = true;
      char msg[220];
      snprintf(msg, sizeof msg, "%sM=%d: ids equal, w rel %.1e, host-mapped copy equal, router_topk(own logits) bit-equal, det", tie ? "ties " : "", M, wmax);
      verdict(a_ids == b_ids && wmax <= 1e-5 && host_ok && topk_same && det && s0 && twins, msg);
      if (!(a_ids == b_ids && wmax <= 1e-5 && host_ok && topk_same && det && s0 && twins)) {
        printf("      flags: ids %d w %d host %d topk %d det %d sync %d twins %d\n", a_ids == b_ids, wmax <= 1e-5, host_ok, topk_same, det, s0, twins);
        for (int i = 0; i < std::min(M * KK, 16); ++i) printf("      %d: old %d new %d\n", i, a_ids[i], b_ids[i]);
      }
      if (M == 8) printf("      logits max |Δ| vs cuBLAS %.2e (max |logit| %.2f)%s\n", lmax, lsc, tie ? (twins ? " · twin logits exactly equal in both" : " · TWINS DIFFER") : "");
      if (!tie && timed_M(M)) {
        trow("router (GEMM+topk, no D2H)", M, time_med([&] { oldr(M, false); }), time_med([&] { newr(M); }), (double)E * H * 2 / 1e6);
        trow("router incl. D2H ids/w (2 copies vs mapped)", M, time_med([&] { oldr(M, true); }), time_med([&] { newr(M); }), (double)E * H * 2 / 1e6);
      }
    }
  }
  cudaFreeHost(ih); cudaFreeHost(wh);
  for (void* p : {(void*)gw, (void*)bias, (void*)x, (void*)lo, (void*)ln, (void*)wo, (void*)wn, (void*)wr, (void*)io, (void*)in, (void*)ir}) cudaFree(p);
}

// ===================================================================================================================================
const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
float e4m3_host(uint8_t v) { __nv_fp8_e4m3 t; t.__x = v; return float(t); }
// fp64 NVFP4 row dot: value, Σ|w·x|, and Σ|w|·|x − f16(x)| (the f16 operand rounding of the tensor-core path, non-zero only for |x| < 2^-14)
struct Dot3 { double v = 0, abs = 0, f16err = 0; };
Dot3 nv_dot3(const uint8_t* w, const uint8_t* s, int K, const float* a) {
  Dot3 d;
  for (int k = 0; k < K; k += 16) {
    const double sc = e4m3_host(s[k / 16]);
    for (int i = 0; i < 16; ++i) {
      const uint8_t b = w[(k + i) / 2];
      const double wv = (double)kE2M1[(i & 1) ? (b >> 4) : (b & 15)] * sc, xv = a[k + i];
      d.v += wv * xv; d.abs += std::fabs(wv * xv);
      d.f16err += std::fabs(wv) * std::fabs(xv - (double)__half2float(__float2half_rn((float)xv)));
    }
  }
  return d;
}
template <class F> void par_for(int n, F f) {  // rows in parallel on the host (reference only)
  const int T = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t) th.emplace_back([&, t] { for (int i = t; i < n; i += T) f(i); });
  for (auto& x : th) x.join();
}
void test_dense() {
  printf("[4] dense NVFP4 MLP (layers 0-2: gate/up [12288,4096], down [4096,12288])\n");
  const int I = 12288;
  uint8_t *w[3], *s[3];
  const int N[3] = {I, I, H}, K[3] = {H, H, I};
  const float gsc[3] = {0.0031f, 0.0027f, 0.0042f};
  Nvfp4Mat mat[3];
  for (int j = 0; j < 3; ++j) {
    w[j] = dmalloc<uint8_t>((size_t)N[j] * K[j] / 2); s[j] = dmalloc<uint8_t>((size_t)N[j] * K[j] / 16);
    fill_u8(w[j], (size_t)N[j] * K[j] / 2, 41 + j, 0, 0, 0); fill_u8(s[j], (size_t)N[j] * K[j] / 16, 51 + j, 0x28, 0x20, 1);
    mat[j] = Nvfp4Mat{w[j], s[j], gsc[j], N[j], K[j]};
  }
  bf16* x = dmalloc<bf16>(8 * H);
  bf16 *ya = dmalloc<bf16>(8 * I), *yb = dmalloc<bf16>(8 * I), *yc = dmalloc<bf16>(8 * I), *gu = dmalloc<bf16>(8 * 2 * I);
  bf16 *oa = dmalloc<bf16>(8 * H), *ob = dmalloc<bf16>(8 * H), *oc = dmalloc<bf16>(8 * H);
  float* tmp = dmalloc<float>(8 * 2 * I);
  auto oldd = [&](int M) {
    nvfp4_gemv(mat[0], x, H, M, tmp, 2 * I, g_st);
    nvfp4_gemv(mat[1], x, H, M, tmp + I, 2 * I, g_st);
    f32_to_bf16(tmp, (size_t)M * 2 * I, gu, g_st);
    swiglu_rows(gu, M, I, LIMIT, ya, g_st);
    nvfp4_gemv(mat[2], ya, I, M, tmp, H, g_st);
    f32_to_bf16(tmp, (size_t)M * H, oa, g_st);
  };
  auto newd = [&](bf16* y, bf16* o, int M) { glm_dense_nvfp4_decode(mat[0], mat[1], mat[2], x, H, M, LIMIT, y, o, H, g_st); };
  std::vector<uint8_t> hw[3], hs[3];
  for (int j = 0; j < 3; ++j) { hw[j] = host(w[j], (size_t)N[j] * K[j] / 2); hs[j] = host(s[j], (size_t)N[j] * K[j] / 16); }
  // y check (linearized error propagation): |y − y_ref| ≤ |∂y/∂g|·(δg + ulp(g)) + |∂y/∂u|·(δu + ulp(u)) + ulp(y) + 2^-20·|y|
  //   δ = scale · (K·2^-24·Σ|w·x| + Σ|w|·|x − f16(x)|) (worst-case fp32 order + the f16 activation rounding of the tensor-core path),
  //   ulp(g), ulp(u): the bf16 rounding of gate/up may flip; 2^-20: __expf vs exp. Returns the max ratio (≤ 1 passes).
  auto check_y = [&](const std::vector<bf16>& yg, const std::vector<Dot3>& g3, const std::vector<Dot3>& u3, int M) {
    double worst = 0;
    const double gam = H * std::ldexp(1.0, -24), L = LIMIT;
    for (size_t i = 0; i < (size_t)M * I; ++i) {
      const double g = g3[i].v * gsc[0], u = u3[i].v * gsc[1];
      const double dg = gsc[0] * (gam * g3[i].abs + g3[i].f16err) + ulp_bf16(g), du = gsc[1] * (gam * u3[i].abs + u3[i].f16err) + ulp_bf16(u);
      const double gc = std::min(g, L), uc = std::min(std::max(u, -L), L);
      const double sg = 1.0 / (1.0 + std::exp(-gc)), silu = gc * sg, dsilu = sg * (1.0 + gc * (1.0 - sg));
      const double yref = silu * uc;
      const double dyg = (g - dg > L) ? 0.0 : std::fabs(dsilu * uc), dyu = (std::fabs(u) - du > L) ? 0.0 : std::fabs(silu);
      const double tol = dyg * dg + dyu * du + ulp_bf16(yref) + std::ldexp(std::fabs(yref), -20);
      worst = std::max(worst, std::fabs((double)bf(yg[i]) - yref) / tol);
    }
    return worst;
  };
  // out check: fp64 down · (this path's own y), ratio to the budget ulp + γ·Σ|w·y| + Σ|w|·|y − f16(y)|
  auto check_out = [&](const std::vector<bf16>& yg, const std::vector<bf16>& og, int M) {
    std::vector<float> yf(yg.size());
    for (size_t i = 0; i < yg.size(); ++i) yf[i] = bf(yg[i]);
    std::vector<double> ratio((size_t)M * H);
    par_for(M * H, [&](int idx) {
      const int m = idx / H, n = idx % H;
      const Dot3 d = nv_dot3(hw[2].data() + (size_t)n * I / 2, hs[2].data() + (size_t)n * I / 16, I, yf.data() + (size_t)m * I);
      const double ref = d.v * gsc[2], budget = ulp_bf16(ref) + gsc[2] * (I * std::ldexp(1.0, -24) * d.abs + d.f16err);
      ratio[idx] = std::fabs((double)bf(og[idx]) - ref) / budget;
    });
    return *std::max_element(ratio.begin(), ratio.end());
  };
  for (int big = 0; big < 2; ++big) {
    fill_bf16(x, 8 * H, 61, 0.f, 1.5f);
    if (big) {  // activations outside the f16 range (exact CUDA-core fallback path)
      std::vector<bf16> hx = host(x, 8 * H);
      for (size_t i = 0; i < hx.size(); i += 97) hx[i] = f2bf(i & 1 ? -3e5f : 3e5f);
      CUDA_CHECK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    }
    const auto hx = host(x, 8 * H);
    std::vector<float> xf(8 * H);
    for (int i = 0; i < 8 * H; ++i) xf[i] = bf(hx[i]);
    std::vector<Dot3> g3((size_t)8 * I), u3((size_t)8 * I);
    par_for(8 * I, [&](int idx) {
      const int m = idx / I, i = idx % I;
      g3[idx] = nv_dot3(hw[0].data() + (size_t)i * H / 2, hs[0].data() + (size_t)i * H / 16, H, xf.data() + (size_t)m * H);
      u3[idx] = nv_dot3(hw[1].data() + (size_t)i * H / 2, hs[1].data() + (size_t)i * H / 16, H, xf.data() + (size_t)m * H);
    });
    for (const int M : Ms) {
      if (M <= 4) oldd(M);
      newd(yb, ob, M); newd(yc, oc, M);
      CUDA_CHECK(cudaStreamSynchronize(g_st));
      const auto hyb = host(yb, (size_t)M * I), hob = host(ob, (size_t)M * H);
      const bool det = same_bytes(yb, yc, (size_t)M * I) && same_bytes(ob, oc, (size_t)M * H);
      const double yrat = check_y(hyb, g3, u3, M);
      const double orat = check_out(hyb, hob, M);
      char msg[240];
      snprintf(msg, sizeof msg, "%sM=%d new: y/fp64 budget ratio %.2f · out/fp64(own y) budget ratio %.2f · deterministic", big ? "big-x " : "", M, yrat, orat);
      verdict(yrat <= 1.0 && orat <= 1.0 && det, msg);
      if (M <= 4) {
        const auto hya = host(ya, (size_t)M * I), hoa = host(oa, (size_t)M * H);
        const double yrat_o = check_y(hya, g3, u3, M);
        const double orat_o = check_out(hya, hoa, M);
        const auto dy = ulp_cmp(hyb, hya), dout = ulp_cmp(hob, hoa);
        snprintf(msg, sizeof msg, "%sM=%d old (same check): y ratio %.2f · out ratio %.2f   [new−old: y %zu/%zu differ, out %zu/%zu differ]", big ? "big-x " : "",
                 M, yrat_o, orat_o, dy.diff, dy.n, dout.diff, dout.n);
        verdict(yrat_o <= 1.0 && orat_o <= 1.0, msg);
      }
      if (!big && timed_M(M)) {
        const double mb = (2.0 * I * H / 2 + 2.0 * I * H / 16 + (double)H * I / 2 + (double)H * I / 16) / 1e6;
        trow("dense NVFP4 MLP (6 launches → 2)", M, M <= 4 ? time_med([&] { oldd(M); }) : -1.f, time_med([&] { newd(yb, ob, M); }), mb);
      }
    }
  }
  for (int j = 0; j < 3; ++j) { cudaFree(w[j]); cudaFree(s[j]); }
  for (void* p : {(void*)x, (void*)ya, (void*)yb, (void*)yc, (void*)gu, (void*)oa, (void*)ob, (void*)oc, (void*)tmp}) cudaFree(p);
}

// ===================================================================================================================================
void test_shared() {
  printf("[5] shared expert FP8 (gate/up [2048,4096], down [4096,2048], 128×128 block scales)\n");
  const int I = 2048;
  const int N[3] = {I, I, H}, K[3] = {H, H, I};
  uint8_t* w[3]; float* s[3]; Fp8BMat mat[3];
  for (int j = 0; j < 3; ++j) {
    const size_t ns = (size_t)((N[j] + 127) / 128) * ((K[j] + 127) / 128);
    w[j] = dmalloc<uint8_t>((size_t)N[j] * K[j]); s[j] = dmalloc<float>(ns);
    fill_u8(w[j], (size_t)N[j] * K[j], 71 + j, 0, 0, 2); fill_f32(s[j], ns, 81 + j, 1.0e-3f, 0.5e-3f);
    mat[j] = Fp8BMat{w[j], s[j], N[j], K[j]};
  }
  bf16* x = dmalloc<bf16>(8 * H); fill_bf16(x, 8 * H, 91, 0.f, 1.5f);
  bf16 *sgu = dmalloc<bf16>(8 * 2 * I), *ya = dmalloc<bf16>(8 * I), *yb = dmalloc<bf16>(8 * I), *yc = dmalloc<bf16>(8 * I);
  bf16 *oa = dmalloc<bf16>(8 * H), *ob = dmalloc<bf16>(8 * H), *oc = dmalloc<bf16>(8 * H);
  auto olds = [&](int M) {
    fp8b_gemv(mat[0], x, H, M, sgu, 2 * I, g_st);
    fp8b_gemv(mat[1], x, H, M, sgu + I, 2 * I, g_st);
    swiglu_rows(sgu, M, I, LIMIT, ya, g_st);
    fp8b_gemv(mat[2], ya, I, M, oa, H, g_st);
  };
  auto news = [&](bf16* y, bf16* o, int M) { glm_shared_fp8_decode(mat[0], mat[1], mat[2], x, H, M, LIMIT, y, o, H, g_st); };
  for (const int M : Ms) {
    olds(M); news(yb, ob, M); news(yc, oc, M);
    CUDA_CHECK(cudaStreamSynchronize(g_st));
    const bool ey = same_bytes(ya, yb, (size_t)M * I), eo = same_bytes(oa, ob, (size_t)M * H);
    const bool det = same_bytes(yb, yc, (size_t)M * I) && same_bytes(ob, oc, (size_t)M * H);
    const auto so = ulp_cmp(host(ob, (size_t)M * H), host(oa, (size_t)M * H));
    char msg[200];
    snprintf(msg, sizeof msg, "M=%d: y bytes %s, out bytes %s (max %.2f step), deterministic", M, ey ? "equal" : "DIFFER", eo ? "equal" : "DIFFER", so.max_ulp);
    verdict(ey && eo && det, msg);
    if (timed_M(M)) {
      const double mb = (3.0 * I * H + 3.0 * 16 * 32 * 4) / 1e6;
      trow("shared expert FP8 (4 launches → 2)", M, time_med([&] { olds(M); }), time_med([&] { news(yb, ob, M); }), mb);
    }
  }
  for (int j = 0; j < 3; ++j) { cudaFree(w[j]); cudaFree(s[j]); }
  for (void* p : {(void*)x, (void*)sgu, (void*)ya, (void*)yb, (void*)yc, (void*)oa, (void*)ob, (void*)oc}) cudaFree(p);
}
}  // namespace

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : "";
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&g_ea)); CUDA_CHECK(cudaEventCreate(&g_eb));
  CUDA_CHECK(cudaMalloc(&g_flush, kFlushB)); CUDA_CHECK(cudaMemset(g_flush, 1, kFlushB));
  Blas blas(g_st);
  int* sync = dmalloc<int>(kGlmDecodeSyncInts);
  auto on = [&](const char* k) { return !*only || strstr(only, k); };
  if (on("hc")) test_hc(sync);
  if (on("gemv")) test_gemv(blas);
  if (on("router")) test_router(blas);
  if (on("dense")) test_dense();
  if (on("shared")) test_shared();
  printf("\nTiming (µs, median of 25, L2 flushed by a 256 MB read before each iteration; GB/s = weight bytes / new time)\n");
  printf("  %-44s %2s %10s %10s %7s %8s\n", "op", "M", "old", "new", "x", "GB/s");
  for (const TRow& r : g_rows) {
    if (r.old_us < 0) printf("  %-44s %2d %10s %10.1f %7s %8.0f\n", r.what, r.M, "abort", r.new_us, "-", r.mb / r.new_us * 1e3);
    else printf("  %-44s %2d %10.1f %10.1f %6.2fx %8.0f\n", r.what, r.M, r.old_us, r.new_us, r.old_us / r.new_us, r.mb / r.new_us * 1e3);
  }
  printf("\n%s (%d failure%s)\n", fails ? "SOME FAILED" : "ALL PASS", fails, fails == 1 ? "" : "s");
  return fails ? 1 : 0;
}
