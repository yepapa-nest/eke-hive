// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// glm_moe_fp8_decode (glm_moe_fp8.h) vs the composition glm_mtp.cpp runs today for the MTP routed experts:
//   per row m, per slot k (in order), id valid → glm_shared_fp8_decode(expert[id], x[m], M = 1) → accum_scaled(out[m], y, w[m, k]).
//   E = 16 random FP8 experts of the real shapes (gate/up [2048, 4096], down [4096, 2048], fp32 128×128 block scales; e4m3 NaN codes
//   excluded), random route weights, out pre-filled with random values (the op accumulates). Cases per M ∈ {1, 2, 3, 4}: random distinct ids
//   per row (repeats across rows), and for M ≥ 2 one with invalid ids (-1, ≥ E) and a duplicate inside a row.
//   Checks: out bytes equal to the reference, the per-slot SwiGLU output y equal to the reference's y_ws, two runs byte-identical, and a
//   CUDA-graph capture of the op replaying to the same bytes.
//   Information: fraction of gate/up values the SwiGLU clamp changes (so the clamp control is meaningful).
// Timing: CUDA events, L2 flushed before every iteration by READING a 256 MB buffer, median of 25. Old = the engine's path including the
//   host stream sync per row (ids then read on the host); new = one glm_moe_fp8_decode. GB/s = distinct expert bytes / time.
// Negative controls: build src/glm/glm_moe_fp8.cu with -DHIVE_GLM_MOE_FP8_NEG_{SCALE,ID,CLAMP} → this test must FAIL.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <vector>

#include "hive/common.h"
#include "hive/glm/fp8b.h"
#include "hive/glm/glm_decode.h"
#include "hive/glm/glm_kernels.h"
#include "hive/glm/glm_moe_fp8.h"

using namespace hive;
using namespace hive::glm;

namespace {
int fails = 0;
void verdict(bool ok, const char* what) { printf("  %-96s %s\n", what, ok ? "PASS" : "FAIL"); if (!ok) ++fails; }

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
__global__ void fill_e4m3_k(uint8_t* p, size_t n, uint32_t seed) {  // any e4m3 code except NaN (0x7F / 0xFF)
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint8_t v = (uint8_t)(hsh(i, seed) >> 8);
    if ((v & 0x7F) == 0x7F) v &= 0xFE;
    p[i] = v;
  }
}
__global__ void flush_read(const uint4* __restrict__ p, size_t n, unsigned* o) {
  unsigned a = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const uint4 v = p[i]; a ^= v.x ^ v.w; }
  if (a == 0x9E3779B9u) *o = a;
}
template <class T> T* dmalloc(size_t n) {
  T* p;
  CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
  std::vector<T> v(n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
template <class T> bool same_bytes(const T* a, const T* b, size_t n) { auto x = host(a, n), y = host(b, n); return memcmp(x.data(), y.data(), n * sizeof(T)) == 0; }

cudaStream_t g_st;
void* g_flush;
const size_t kFlushB = 256u << 20;
cudaEvent_t g_ea, g_eb;
float time_med(const std::function<void()>& f, int reps = 25) {
  std::vector<float> v;
  f(); CUDA_CHECK(cudaStreamSynchronize(g_st));  // warm-up
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

const int H = 4096, I = 2048, E = 16, K = 8;
const float LIMIT = 10.f;

}  // namespace

int main() {
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&g_ea)); CUDA_CHECK(cudaEventCreate(&g_eb));
  CUDA_CHECK(cudaMalloc(&g_flush, kFlushB)); CUDA_CHECK(cudaMemset(g_flush, 1, kFlushB));

  // ---- experts
  std::vector<Fp8BMat> ex(3 * E);
  const int N3[3] = {I, I, H}, K3[3] = {H, H, I};
  std::vector<void*> frees;
  for (int e = 0; e < E; ++e)
    for (int j = 0; j < 3; ++j) {
      const size_t nw = (size_t)N3[j] * K3[j], ns = (size_t)(N3[j] / 128) * (K3[j] / 128);
      uint8_t* w = dmalloc<uint8_t>(nw); float* s = dmalloc<float>(ns);
      fill_e4m3_k<<<2048, 256>>>(w, nw, 1000 + 7 * e + j);
      fill_f32_k<<<64, 256>>>(s, ns, 5000 + 7 * e + j, 1.0e-3f, 0.5e-3f);
      CUDA_CHECK(cudaGetLastError());
      ex[3 * e + j] = Fp8BMat{w, s, N3[j], K3[j]};
      frees.push_back(w); frees.push_back(s);
    }
  const double expert_mb = (3.0 * I * H + 3.0 * 16 * 32 * 4) / 1e6;
  Fp8BMat* tab = glm_moe_fp8_table_create(ex.data(), E, H, I);

  bf16* x = dmalloc<bf16>(4 * H);
  CUDA_CHECK(cudaMemsetAsync(x, 0, 4 * H * 2));
  fill_bf16_k<<<256, 256>>>(x, 4 * H, 91, 0.f, 1.5f);
  float* out0 = dmalloc<float>(4 * H);
  fill_f32_k<<<256, 256>>>(out0, 4 * H, 92, 0.f, 3.f);
  float *oref = dmalloc<float>(4 * H), *oa = dmalloc<float>(4 * H), *ob = dmalloc<float>(4 * H);
  bf16 *ytmp = dmalloc<bf16>(I), *tmp = dmalloc<bf16>(H), *yref = dmalloc<bf16>(32 * I);
  int32_t* ids = dmalloc<int32_t>(32);
  float* wts = dmalloc<float>(32);
  const size_t wsb = glm_moe_fp8_ws_bytes(4, K, I);
  void* ws = dmalloc<uint8_t>(wsb);
  CUDA_CHECK(cudaDeviceSynchronize());

  // reference: the engine's loop (host ids/weights)
  std::vector<int32_t> hid;
  std::vector<float> hw;
  auto run_old = [&](int M, float* out, bool with_sync, bool keep_y) {
    for (int m = 0; m < M; ++m) {
      if (with_sync) CUDA_CHECK(cudaStreamSynchronize(g_st));  // routing on the host (glm_mtp.cpp)
      for (int k = 0; k < K; ++k) {
        const int id = hid[m * K + k];
        if (id < 0 || id >= E) continue;
        glm_shared_fp8_decode(ex[3 * id], ex[3 * id + 1], ex[3 * id + 2], x + (size_t)m * H, H, 1, LIMIT, ytmp, tmp, H, g_st);
        if (keep_y) CUDA_CHECK(cudaMemcpyAsync(yref + (size_t)(m * K + k) * I, ytmp, I * 2, cudaMemcpyDeviceToDevice, g_st));
        accum_scaled(out + (size_t)m * H, tmp, hw[m * K + k], H, g_st);
      }
    }
  };
  auto run_new = [&](int M, float* out) { glm_moe_fp8_decode(tab, E, ids, wts, x, M, K, H, I, LIMIT, out, ws, g_st); };

  srand(12345);
  struct Case { int M; int kind; };  // kind 0: distinct ids per row · 1: invalid ids + duplicate within a row
  const Case cases[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {2, 1}, {4, 1}};
  printf("[1] glm_moe_fp8_decode vs glm_shared_fp8_decode + accum_scaled per (row, slot)  (E=%d, K=%d, H=%d, I=%d, limit %.0f)\n", E, K, H, I, LIMIT);
  for (const Case& c : cases) {
    const int M = c.M;
    hid.assign(M * K, 0); hw.assign(M * K, 0.f);
    for (int m = 0; m < M; ++m) {
      std::vector<int> perm(E);
      for (int e = 0; e < E; ++e) perm[e] = e;
      for (int e = E - 1; e > 0; --e) std::swap(perm[e], perm[rand() % (e + 1)]);
      for (int k = 0; k < K; ++k) { hid[m * K + k] = perm[k]; hw[m * K + k] = 0.1f + 2.4f * (float)rand() / RAND_MAX; }
    }
    if (c.kind == 1) { hid[1] = -1; hid[K + 2] = E + 83; hid[K + 5] = hid[K + 3]; }
    CUDA_CHECK(cudaMemcpy(ids, hid.data(), M * K * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wts, hw.data(), M * K * 4, cudaMemcpyHostToDevice));
    std::set<int> distinct;
    for (int v : hid) if (v >= 0 && v < E) distinct.insert(v);

    CUDA_CHECK(cudaMemcpy(oref, out0, M * H * 4, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(oa, out0, M * H * 4, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(ob, out0, M * H * 4, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(yref, 0, 32 * I * 2));
    CUDA_CHECK(cudaMemset(ws, 0, wsb));
    CUDA_CHECK(cudaDeviceSynchronize());  // the setup above runs on the legacy stream, which does not order with the non-blocking g_st
    run_old(M, oref, true, true);
    run_new(M, oa);
    CUDA_CHECK(cudaStreamSynchronize(g_st));
    // per-slot y of the new path (ws) vs the reference's y_ws — valid slots only
    auto ynew = host((const bf16*)ws, (size_t)M * K * I), yr = host(yref, (size_t)M * K * I);
    bool ey = true;
    for (int s = 0; s < M * K; ++s)
      if (hid[s] >= 0 && hid[s] < E && memcmp(&ynew[(size_t)s * I], &yr[(size_t)s * I], I * 2) != 0) {
        ey = false;
        // diagnose: recompute this slot alone (after a full sync) and say which side differs
        const int id = hid[s], m = s / K;
        glm_shared_fp8_decode(ex[3 * id], ex[3 * id + 1], ex[3 * id + 2], x + (size_t)m * H, H, 1, LIMIT, ytmp, tmp, H, g_st);
        auto y1 = host(ytmp, I);
        printf("    y slot %d (id %d): new %s the isolated recompute, reference copy %s it\n", s, id,
               memcmp(y1.data(), &ynew[(size_t)s * I], I * 2) ? "DIFFERS from" : "equals", memcmp(y1.data(), &yr[(size_t)s * I], I * 2) ? "DIFFERS from" : "equals");
      }
    run_new(M, ob);
    CUDA_CHECK(cudaStreamSynchronize(g_st));
    auto a = host(oa, (size_t)M * H), r = host(oref, (size_t)M * H), b = host(ob, (size_t)M * H);
    size_t ndiff = 0; double maxd = 0;
    for (size_t i = 0; i < a.size(); ++i) if (memcmp(&a[i], &r[i], 4)) { ++ndiff; maxd = std::max(maxd, (double)std::fabs(a[i] - r[i])); }
    const bool det = memcmp(a.data(), b.data(), a.size() * 4) == 0;
    // the op accumulated something (sanity: out changed from out0)
    auto o0 = host(out0, (size_t)M * H);
    size_t changed = 0;
    for (size_t i = 0; i < a.size(); ++i) changed += a[i] != o0[i];
    char msg[256];
    snprintf(msg, sizeof msg, "M=%d %s (%zu distinct experts): out bytes %s (%zu diff, max |d| %.3g), y bytes %s, deterministic %s, %zu/%zu changed",
             M, c.kind ? "invalid+dup ids" : "random ids     ", distinct.size(), ndiff ? "DIFFER" : "equal", ndiff, maxd, ey ? "equal" : "DIFFER",
             det ? "yes" : "NO", changed, a.size());
    verdict(!ndiff && ey && det && changed > a.size() / 2, msg);
  }

  // CUDA graph capture of the last case (no host sync inside the op → capturable); replay must give the same bytes
  {
    const int M = 4;
    CUDA_CHECK(cudaMemcpy(ob, out0, M * H * 4, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaGraph_t g; cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(g_st, cudaStreamCaptureModeGlobal));
    run_new(M, ob);
    CUDA_CHECK(cudaStreamEndCapture(g_st, &g));
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    CUDA_CHECK(cudaGraphLaunch(ge, g_st));
    CUDA_CHECK(cudaStreamSynchronize(g_st));
    verdict(same_bytes(oa, ob, (size_t)M * H), "M=4 invalid+dup ids captured in a CUDA graph: replay bytes equal to the direct launch");
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  }

  // clamp coverage (information): gate/up values of the M=4 case's experts on row 0
  {
    bf16* g = dmalloc<bf16>(I); bf16* u = dmalloc<bf16>(I);
    size_t cg = 0, cu = 0, n = 0;
    for (int k = 0; k < K; ++k) {
      const int id = hid[k];
      if (id < 0 || id >= E) continue;
      fp8b_gemv(ex[3 * id], x, H, 1, g, I, g_st); fp8b_gemv(ex[3 * id + 1], x, H, 1, u, I, g_st);
      auto hg = host(g, I), hu = host(u, I);
      for (int i = 0; i < I; ++i) { cg += bf2f(hg[i]) > LIMIT; cu += std::fabs(bf2f(hu[i])) > LIMIT; ++n; }
    }
    printf("  info: clamp active on %.1f %% of gate values (g > %.0f) and %.1f %% of up values (|u| > %.0f)\n", 100.0 * cg / n, LIMIT, 100.0 * cu / n,
           LIMIT);
    cudaFree(g); cudaFree(u);
  }

  // ---- timing
  printf("\nTiming (µs, median of 25, L2 flushed by a 256 MB read before each iteration; random distinct ids per row from E=%d)\n", E);
  printf("  %-3s %8s %13s %13s %8s %10s %10s\n", "M", "experts", "old(+sync)", "new", "speedup", "old GB/s", "new GB/s");
  for (const int M : {1, 2, 4}) {
    hid.assign(M * K, 0); hw.assign(M * K, 0.f);
    for (int m = 0; m < M; ++m) {
      std::vector<int> perm(E);
      for (int e = 0; e < E; ++e) perm[e] = e;
      for (int e = E - 1; e > 0; --e) std::swap(perm[e], perm[rand() % (e + 1)]);
      for (int k = 0; k < K; ++k) { hid[m * K + k] = perm[k]; hw[m * K + k] = 0.1f + 2.4f * (float)rand() / RAND_MAX; }
    }
    CUDA_CHECK(cudaMemcpy(ids, hid.data(), M * K * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wts, hw.data(), M * K * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::set<int> distinct(hid.begin(), hid.end());
    const float to = time_med([&] { run_old(M, oa, true, false); });
    const float tn = time_med([&] { run_new(M, ob); });
    const double mb_new = distinct.size() * expert_mb, mb_old = (double)M * K * expert_mb;
    printf("  %-3d %8zu %13.1f %13.1f %7.2fx %10.0f %10.0f\n", M, distinct.size(), to, tn, to / tn, mb_old / to * 1e6 / 1e3, mb_new / tn * 1e6 / 1e3);
  }

  glm_moe_fp8_table_free(tab);
  for (void* p : frees) cudaFree(p);
  printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASS", fails, fails == 1 ? "" : "s");
  return fails ? 1 : 0;
}
