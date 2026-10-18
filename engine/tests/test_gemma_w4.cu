// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma int4 group-32 kernels (hive/gemma/gemma_kernels.h): w4_gemv / w4_gemv_gu / w4_dequant against an fp64 host reference,
//   row results independent of the row count (batch invariance), and decode bandwidth at the model's shapes.
//   Run: scripts/hive-run.sh "./build-svc/test_gemma_w4"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/devbuf.h"
#include "hive/gemma/gemma_kernels.h"

using namespace hive;
using namespace hive::gemma::k;

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

struct HostMat {
  int N, K;
  std::vector<uint32_t> q;
  std::vector<uint16_t> s;
  double w(int n, int k) const { return (double)((int)((q[(size_t)n * (K / 8) + k / 8] >> (4 * (k % 8))) & 0xF) - 8) * bf2f(s[(size_t)n * (K / 32) + k / 32]); }
};

static HostMat make(int N, int K, std::mt19937& rng) {
  HostMat m{N, K, std::vector<uint32_t>((size_t)N * K / 8), std::vector<uint16_t>((size_t)N * K / 32)};
  for (auto& v : m.q) v = rng();
  std::normal_distribution<float> nd(0.f, 0.01f);
  for (auto& v : m.s) v = f2bf(std::fabs(nd(rng)) + 1e-4f);
  return m;
}

struct DevMat {   // uploaded in checkpoint order, repacked to fragment order on the GPU (as the loader does)
  DevBuf q, s;
  W4Mat W;
  DevMat(const HostMat& h) {
    DevBuf qc(h.q.size() * 4), sc(h.s.size() * 2);
    cudaMemcpy(qc.p, h.q.data(), h.q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(sc.p, h.s.data(), h.s.size() * 2, cudaMemcpyHostToDevice);
    q.alloc(h.q.size() * 4); s.alloc(h.s.size() * 2);
    w4_repack(qc.as<uint32_t>(), sc.as<bf16>(), h.N, h.K, q.as<uint32_t>(), s.as<uint32_t>(), nullptr);
    cudaDeviceSynchronize();
    W.q = q.as<uint32_t>(); W.s = s.as<uint32_t>(); W.N = h.N; W.K = h.K; W.split = w4_split_for(h.N, h.K);
  }
};

static bool check(bool ok, const char* what) { printf("%-80s %s\n", what, ok ? "PASS" : "FAIL"); return ok; }

// cold-L2 time of one call (µs, median of R): a 512 MiB read evicts the weights from L2 before every timed call (the model streams
//   ~8.5 GB per step, so nothing of a matrix is in L2 when its turn comes)
__global__ void flush_read(const uint4* p, size_t n, unsigned* sink) {
  uint32_t a = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const uint4 v = p[i]; a ^= v.x ^ v.w; }
  if (a == 0x12345678u) *sink = a;
}

template <class F>
static double cold_us(F&& fn, int R = 30) {
  static DevBuf flush((size_t)512 << 20), sink(64);
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> t;
  for (int i = 0; i < R + 3; ++i) {
    if (i == 0) cudaMemset(flush.p, 1, flush.n);
    flush_read<<<1024, 256>>>(flush.as<uint4>(), flush.n / 16, sink.as<unsigned>());
    cudaEventRecord(a, nullptr);
    fn();
    cudaEventRecord(b, nullptr);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    if (i >= 3) t.push_back(ms * 1000.f);
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  bool all = true;
  std::mt19937 rng(7);
  cudaStream_t st = nullptr;
  // correctness on a small shape with 1, 3 (uneven K ranges) and 8 warps per tile
  for (int split : {1, 3, 8}) {
    const int N = 256, K = 1024, M = 32;
    HostMat h = make(N, K, rng);
    DevMat d(h);
    d.W.split = split;
    std::vector<uint16_t> xh((size_t)M * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = f2bf(nd(rng));
    DevBuf x((size_t)M * K * 2), y((size_t)M * N * 4), ws((size_t)split * M * N * 8 + 16);
    cudaMemcpy(x.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    double maxrel = 0;
    std::vector<float> yM((size_t)M * N);
    w4_gemv(d.W, x.as<bf16>(), K, M, nullptr, y.as<float>(), N, ws.as<float>(), st);
    cudaMemcpy(yM.data(), y.p, yM.size() * 4, cudaMemcpyDeviceToHost);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double r = 0, a = 0;
        for (int k = 0; k < K; ++k) { const double t = h.w(n, k) * bf2f(xh[(size_t)m * K + k]); r += t; a += std::fabs(t); }
        maxrel = std::max(maxrel, std::fabs(r - yM[(size_t)m * N + n]) / (a + 1e-12));
      }
    char b[128];
    snprintf(b, sizeof b, "w4_gemv fp32 vs fp64 reference (split %d, M 32): max |err| / sum|terms| = %.2e", split, maxrel);
    all &= check(maxrel < 1e-5, b);
    // batch invariance: row m alone == row m inside M rows
    bool inv = true;
    for (int Mi : {1, 3, 8, 9, 16, 17, 24, 25}) {
      std::vector<float> yi((size_t)Mi * N);
      w4_gemv(d.W, x.as<bf16>(), K, Mi, nullptr, y.as<float>(), N, ws.as<float>(), st);
      cudaMemcpy(yi.data(), y.p, yi.size() * 4, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < yi.size(); ++i) inv &= yi[i] == yM[i];
    }
    snprintf(b, sizeof b, "  … rows identical for M = 1, 3, 8, 9, 16, 17, 24, 25, 32 (split %d)", split);
    all &= check(inv, b);
    // gu: act == bf16(gelu(g)·u) from the fp32 gemv results
    HostMat hu = make(N, K, rng);
    DevMat du(hu);
    du.W.split = split;
    DevBuf yu((size_t)M * N * 4), act((size_t)M * N * 2);
    w4_gemv(du.W, x.as<bf16>(), K, M, nullptr, yu.as<float>(), N, ws.as<float>(), st);
    w4_gemv_gu(d.W, du.W, x.as<bf16>(), K, M, act.as<bf16>(), N, ws.as<float>(), st);
    std::vector<float> uM((size_t)M * N);
    std::vector<uint16_t> aM((size_t)M * N);
    cudaMemcpy(uM.data(), yu.p, uM.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(aM.data(), act.p, aM.size() * 2, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (size_t i = 0; i < aM.size(); ++i) {
      const float g = yM[i], ref = 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) * uM[i];
      if (std::fabs(bf2f(aM[i]) - ref) > 1e-2f * std::fabs(ref) + 1e-6f) ++bad;
    }
    snprintf(b, sizeof b, "w4_gemv_gu == bf16(gelu_tanh(G x) · (U x)) (split %d): %d mismatches", split, bad);
    all &= check(bad == 0, b);
    // dequant
    DevBuf deq((size_t)N * K * 2);
    w4_dequant(d.W, deq.as<bf16>(), st);
    std::vector<uint16_t> dh((size_t)N * K);
    cudaMemcpy(dh.data(), deq.p, dh.size() * 2, cudaMemcpyDeviceToHost);
    bool dq = true;
    for (int n = 0; n < N && dq; ++n)
      for (int k = 0; k < K; ++k) dq &= dh[(size_t)n * K + k] == f2bf((float)h.w(n, k));
    snprintf(b, sizeof b, "w4_dequant == bf16((nibble - 8) · s) (split %d)", split);
    all &= check(dq, b);
  }
  // w4_gemv_multi (q / k / v in one launch) == separate calls, bit for bit
  {
    const int K = 1024, M = 5;
    HostMat ha = make(256, K, rng), hb = make(128, K, rng), hc = make(64, K, rng);
    DevMat da(ha), db(hb), dc(hc);
    std::vector<uint16_t> xh((size_t)M * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = f2bf(nd(rng));
    DevBuf x((size_t)M * K * 2), y1((size_t)M * 448 * 4), y2((size_t)M * 448 * 4);
    cudaMemcpy(x.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    w4_gemv(da.W, x.as<bf16>(), K, M, nullptr, y1.as<float>(), 448, nullptr, st);
    w4_gemv(db.W, x.as<bf16>(), K, M, nullptr, y1.as<float>() + 256, 448, nullptr, st);
    w4_gemv(dc.W, x.as<bf16>(), K, M, nullptr, y1.as<float>() + 384, 448, nullptr, st);
    W4Multi J;
    J.n = 3;
    J.W[0] = da.W; J.W[1] = db.W; J.W[2] = dc.W;
    J.yf[0] = y2.as<float>(); J.yf[1] = y2.as<float>() + 256; J.yf[2] = y2.as<float>() + 384;
    J.ldy[0] = J.ldy[1] = J.ldy[2] = 448;
    w4_gemv_multi(J, x.as<bf16>(), K, M, st);
    std::vector<float> a((size_t)M * 448), b((size_t)M * 448);
    cudaMemcpy(a.data(), y1.p, a.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), y2.p, b.size() * 4, cudaMemcpyDeviceToHost);
    all &= check(memcmp(a.data(), b.data(), a.size() * 4) == 0, "w4_gemv_multi (3 matrices) == 3 w4_gemv calls (bit exact)");
  }
  // decode bandwidth at the model's shapes (one layer's matrices of Gemma 4 12B, sliding layer)
  struct Shape { const char* name; int N, K; } shapes[] = {{"q (sliding)", 4096, 3840}, {"k / v", 2048, 3840}, {"o", 3840, 4096},
                                                           {"gate / up", 15360, 3840}, {"down", 3840, 15360}, {"q (full)", 8192, 3840}};
  for (const auto& sh : shapes) {
    HostMat h = make(sh.N, sh.K, rng);
    DevMat d(h);
    DevBuf x((size_t)16 * sh.K * 2), y((size_t)16 * sh.N * 2), ws(w4_ws_bytes(d.W, 16) + 16);
    cudaMemset(x.p, 0, x.n);
    for (int M : {1, 8, 16}) {
      const double us = cold_us([&] { w4_gemv(d.W, x.as<bf16>(), sh.K, M, y.as<bf16>(), nullptr, sh.N, ws.as<float>(), st); });
      const double bytes = (double)sh.N * sh.K / 2 + (double)sh.N * sh.K / 32 * 2;
      printf("  %-12s N %5d K %5d split %d  M %2d: %7.1f us  %6.0f GB/s\n", sh.name, sh.N, sh.K, d.W.split, M, us, bytes / us / 1e3);
    }
  }
  {   // q + k + v of a sliding layer in one launch
    HostMat hq = make(4096, 3840, rng), hk = make(2048, 3840, rng), hv = make(2048, 3840, rng);
    DevMat dq(hq), dk(hk), dv(hv);
    DevBuf x((size_t)16 * 3840 * 2), y((size_t)16 * 8192 * 2);
    cudaMemset(x.p, 0, x.n);
    W4Multi J;
    J.n = 3;
    J.W[0] = dq.W; J.W[1] = dk.W; J.W[2] = dv.W;
    J.yb[0] = y.as<bf16>(); J.yb[1] = y.as<bf16>() + 4096; J.yb[2] = y.as<bf16>() + 6144;
    J.ldy[0] = J.ldy[1] = J.ldy[2] = 8192;
    for (int M : {1, 4, 16}) {
      const double us = cold_us([&] { w4_gemv_multi(J, x.as<bf16>(), 3840, M, st); });
      const double bytes = 8192.0 * 3840 / 2 + 8192.0 * 3840 / 32 * 2;
      printf("  %-12s N %5d K %5d split %d  M %2d: %7.1f us  %6.0f GB/s\n", "q+k+v fused", 8192, 3840, dq.W.split, M, us, bytes / us / 1e3);
    }
  }
  {   // gate + up (one launch)
    HostMat hg = make(15360, 3840, rng), hu = make(15360, 3840, rng);
    DevMat dg(hg), du(hu);
    DevBuf x((size_t)16 * 3840 * 2), act((size_t)16 * 15360 * 2);
    cudaMemset(x.p, 0, x.n);
    for (int M : {1, 4, 16}) {
      const double us = cold_us([&] { w4_gemv_gu(dg.W, du.W, x.as<bf16>(), 3840, M, act.as<bf16>(), 15360, nullptr, st); });
      const double bytes = 2 * (15360.0 * 3840 / 2 + 15360.0 * 3840 / 32 * 2);
      printf("  %-12s N %5d K %5d split %d  M %2d: %7.1f us  %6.0f GB/s\n", "gate+up", 15360, 3840, dg.W.split, M, us, bytes / us / 1e3);
    }
  }
  const cudaError_t e = cudaGetLastError();
  all &= check(e == cudaSuccess, cudaGetErrorString(e));
  printf("%s\n", all ? "ALL PASS" : "FAIL");
  return all ? 0 : 1;
}
