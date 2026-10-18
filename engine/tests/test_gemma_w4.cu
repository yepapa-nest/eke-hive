// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma int4 group-32 kernels (hive/gemma/gemma_kernels.h): w4_gemv / w4_gemv_gu / w4_dequant against an fp64 host reference,
//   row results independent of the row count (batch invariance), the prefill GEMM (w4_gemm_multi / w4_gemm_gu) bit for bit against the
//   GEMV at every row count, decode bandwidth at the model's shapes, and prefill times (GEMV passes / GEMM / dequant + cuBLAS).
//   Run: scripts/hive-run.sh "./build-svc/test_gemma_w4"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "hive/cublas_ops.h"
#include "hive/devbuf.h"
#include "hive/gemma/gemma_kernels.h"

using namespace hive;
using namespace hive::gemma::k;

static uint16_t tf2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
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
  for (auto& v : m.s) v = tf2bf(std::fabs(nd(rng)) + 1e-4f);
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

// one flush buffer for every cold_us instantiation (a static inside the template was one 512 MiB buffer per lambda type, never freed)
static DevBuf& flush_buf() {
  static DevBuf flush((size_t)(getenv("HIVE_W4_FLUSH_MB") ? atoi(getenv("HIVE_W4_FLUSH_MB")) : 512) << 20);
  return flush;
}

template <class F>
static double cold_us(F&& fn, int R = 30) {
  static DevBuf sink(64);
  DevBuf& flush = flush_buf();
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
    for (auto& v : xh) v = tf2bf(nd(rng));
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
      for (int k = 0; k < K; ++k) dq &= dh[(size_t)n * K + k] == tf2bf((float)h.w(n, k));
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
    for (auto& v : xh) v = tf2bf(nd(rng));
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
  // prefill GEMM == GEMV bit for bit (rows of any count; the GEMV runs in passes of kGemvRows), plain / three matrices / gate + up
  //   for every variant (HIVE_GEMMA_GEMM_V 0 … 6, and the default); N multiples of 256 so the cluster variants run themselves
  const int kVariants = 9;
  for (int split : {1, 3, 8}) {
    const int K = 1024, MM = 200, NA = 512, NB = 256, NC = 256, NY = NA + NB + NC;
    HostMat ha = make(NA, K, rng), hb = make(NB, K, rng), hc = make(NC, K, rng), hu = make(NA, K, rng);
    DevMat da(ha), db(hb), dc(hc), du(hu);
    da.W.split = db.W.split = dc.W.split = du.W.split = split;
    std::vector<uint16_t> xh((size_t)MM * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = tf2bf(nd(rng));
    DevBuf x((size_t)MM * K * 2), y1((size_t)MM * NY * 4), y2((size_t)MM * NY * 4), a1((size_t)MM * NA * 2), a2((size_t)MM * NA * 2);
    cudaMemcpy(x.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    std::string bad, bad_gu;
    for (int v = -1; v < kVariants; ++v) {
      w4_gemm_set_variant(v);
      bool same = true, same_gu = true;
      for (int M : {1, 5, 16, 17, 31, 32, 33, 48, 70, 79, 80, 81, 128, 150, 192, 200}) {
        W4Multi J;
        J.n = 3; J.W[0] = da.W; J.W[1] = db.W; J.W[2] = dc.W;
        J.yf[0] = y1.as<float>(); J.yf[1] = y1.as<float>() + NA; J.yf[2] = y1.as<float>() + NA + NB;
        J.ldy[0] = J.ldy[1] = J.ldy[2] = NY;
        cudaMemset(y1.p, 0, y1.n); cudaMemset(y2.p, 0xFF, y2.n);
        for (int r0 = 0; r0 < M; r0 += kGemvRows) {
          W4Multi P = J;
          for (int i = 0; i < 3; ++i) P.yf[i] = J.yf[i] + (size_t)r0 * NY;
          w4_gemv_multi(P, x.as<bf16>() + (size_t)r0 * K, K, std::min(kGemvRows, M - r0), st);
        }
        W4Multi Q = J;
        for (int i = 0; i < 3; ++i) Q.yf[i] = J.yf[i] - y1.as<float>() + y2.as<float>();
        w4_gemm_multi(Q, x.as<bf16>(), K, M, st);
        std::vector<float> ra((size_t)M * NY), rb((size_t)M * NY);
        cudaMemcpy(ra.data(), y1.p, ra.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(rb.data(), y2.p, rb.size() * 4, cudaMemcpyDeviceToHost);
        same &= memcmp(ra.data(), rb.data(), ra.size() * 4) == 0;
        cudaMemset(a2.p, 0xFF, a2.n);
        for (int r0 = 0; r0 < M; r0 += kGemvRows)
          w4_gemv_gu(da.W, du.W, x.as<bf16>() + (size_t)r0 * K, K, std::min(kGemvRows, M - r0), a1.as<bf16>() + (size_t)r0 * NA, NA, nullptr, st);
        w4_gemm_gu(da.W, du.W, x.as<bf16>(), K, M, a2.as<bf16>(), NA, st);
        std::vector<uint16_t> ga((size_t)M * NA), gb((size_t)M * NA);
        cudaMemcpy(ga.data(), a1.p, ga.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(gb.data(), a2.p, gb.size() * 2, cudaMemcpyDeviceToHost);
        same_gu &= memcmp(ga.data(), gb.data(), ga.size() * 2) == 0;
      }
      if (!same) bad += " " + std::to_string(v);
      if (!same_gu) bad_gu += " " + std::to_string(v);
    }
    w4_gemm_set_variant(-2);
    char b[200];
    snprintf(b, sizeof b, "w4_gemm_multi (3 matrices) == w4_gemv_multi passes, bit for bit, M = 1 … 200, variants -1 … 8 (split %d)%s%s", split,
             bad.empty() ? "" : " — differ:", bad.c_str());
    all &= check(bad.empty(), b);
    snprintf(b, sizeof b, "w4_gemm_gu == w4_gemv_gu passes, bit for bit, M = 1 … 200, variants -1 … 8 (split %d)%s%s", split,
             bad_gu.empty() ? "" : " — differ:", bad_gu.c_str());
    all &= check(bad_gu.empty(), b);
  }
  // the same at the model's shapes (K 3840 / 4096 / 15360, the engine's splits) and row counts of a turn tail
  {
    struct RShape { const char* name; int N, K; } rs[] = {{"q (sliding)", 4096, 3840}, {"k (full)", 512, 3840}, {"o (sliding)", 3840, 4096},
                                                           {"o (full)", 3840, 8192}, {"down", 3840, 15360}};
    for (const auto& sh : rs) {
      HostMat h = make(sh.N, sh.K, rng);
      DevMat d(h);
      const int MM = 150;
      std::vector<uint16_t> xh((size_t)MM * sh.K);
      std::normal_distribution<float> nd(0.f, 1.f);
      for (auto& v : xh) v = tf2bf(nd(rng));
      DevBuf x((size_t)MM * sh.K * 2), y1((size_t)MM * sh.N * 2), y2((size_t)MM * sh.N * 2);
      cudaMemcpy(x.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
      bool same = true;
      for (int M : {17, 21, 70, 81, 150}) {
        cudaMemset(y1.p, 0, y1.n);
        for (int r0 = 0; r0 < M; r0 += kGemvRows)
          w4_gemv(d.W, x.as<bf16>() + (size_t)r0 * sh.K, sh.K, std::min(kGemvRows, M - r0), y1.as<bf16>() + (size_t)r0 * sh.N, nullptr, sh.N, nullptr, st);
        std::vector<uint16_t> a((size_t)M * sh.N), bb((size_t)M * sh.N);
        cudaMemcpy(a.data(), y1.p, a.size() * 2, cudaMemcpyDeviceToHost);
        for (int v = -1; v < kVariants; ++v) {
          w4_gemm_set_variant(v);
          cudaMemset(y2.p, 0xFF, y2.n);
          W4Multi J; J.n = 1; J.W[0] = d.W; J.yb[0] = y2.as<bf16>(); J.ldy[0] = sh.N;
          w4_gemm_multi(J, x.as<bf16>(), sh.K, M, st);
          cudaMemcpy(bb.data(), y2.p, bb.size() * 2, cudaMemcpyDeviceToHost);
          long diff = 0;
          for (size_t i = 0; i < a.size(); ++i) diff += a[i] != bb[i];
          if (diff) printf("    %s M %d variant %d: %ld of %zu outputs differ\n", sh.name, M, v, diff, a.size());
          same &= diff == 0;
        }
        w4_gemm_set_variant(-2);
      }
      char b[160];
      snprintf(b, sizeof b, "w4_gemm == w4_gemv passes at %s (N %d K %d split %d), M = 17, 21, 70, 81, 150, variants -1 … 8", sh.name, sh.N, sh.K,
               d.W.split);
      all &= check(same, b);
    }
    HostMat hg = make(15360, 3840, rng), hu = make(15360, 3840, rng);
    DevMat dg(hg), du(hu);
    const int MM = 150;
    std::vector<uint16_t> xh((size_t)MM * 3840);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = tf2bf(nd(rng));
    DevBuf x((size_t)MM * 3840 * 2), a1((size_t)MM * 15360 * 2), a2((size_t)MM * 15360 * 2);
    cudaMemcpy(x.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    bool same = true;
    for (int M : {17, 21, 70, 81, 150}) {
      for (int r0 = 0; r0 < M; r0 += kGemvRows)
        w4_gemv_gu(dg.W, du.W, x.as<bf16>() + (size_t)r0 * 3840, 3840, std::min(kGemvRows, M - r0), a1.as<bf16>() + (size_t)r0 * 15360, 15360, nullptr, st);
      std::vector<uint16_t> a((size_t)M * 15360), bb((size_t)M * 15360);
      cudaMemcpy(a.data(), a1.p, a.size() * 2, cudaMemcpyDeviceToHost);
      for (int v = -1; v < kVariants; ++v) {
        w4_gemm_set_variant(v);
        cudaMemset(a2.p, 0xFF, a2.n);
        w4_gemm_gu(dg.W, du.W, x.as<bf16>(), 3840, M, a2.as<bf16>(), 15360, st);
        cudaMemcpy(bb.data(), a2.p, bb.size() * 2, cudaMemcpyDeviceToHost);
        long diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != bb[i];
        if (diff) printf("    gate+up M %d variant %d: %ld of %zu outputs differ\n", M, v, diff, a.size());
        same &= diff == 0;
      }
      w4_gemm_set_variant(-2);
    }
    all &= check(same, "w4_gemm_gu == w4_gemv_gu passes at gate / up (N 15360 K 3840), M = 17, 21, 70, 81, 150, variants -1 … 8");
  }
  // prefill GEMM time of a whole layer (cold L2 before the layer; its matrices in the engine's order, each cold when its turn comes):
  //   sliding layer q + k + v (one launch), o, gate + up, down; full layer q + k, o (K 8192), gate + up, down. Variants interleaved per
  //   repetition (the card is shared with served models: back to back, medians). HIVE_W4_LAYER_V: variants to time (default "2 1 6 -1";
  //   100 = dequant + cuBLAS), HIVE_W4_PARTS=1: the sliding layer's matrices one by one as well.
  {
    const char* ve = getenv("HIVE_W4_LAYER_V");
    std::vector<int> vs;
    for (const char* q = ve ? ve : "2 1 6 -1"; *q;) { char* e; const long v = strtol(q, &e, 10); if (e == q) break; vs.push_back((int)v); q = e; }
    const int reps = getenv("HIVE_W4_LAYER_REPS") ? atoi(getenv("HIVE_W4_LAYER_REPS")) : 15;
    std::vector<int> lms;   // HIVE_W4_LAYER_M: row counts (default "17 40 70 128 192")
    {
      const char* me = getenv("HIVE_W4_LAYER_M");
      for (const char* q = me ? me : "17 40 70 128 192"; *q;) { char* e; const long v = strtol(q, &e, 10); if (e == q) break; lms.push_back((int)v); q = e; }
    }
    const int maxm = *std::max_element(lms.begin(), lms.end());
    HostMat hq = make(4096, 3840, rng), hk = make(2048, 3840, rng), hv = make(2048, 3840, rng), ho = make(3840, 4096, rng);
    HostMat hqf = make(8192, 3840, rng), hkf = make(512, 3840, rng), hof = make(3840, 8192, rng);
    HostMat hg = make(15360, 3840, rng), hu = make(15360, 3840, rng), hd = make(3840, 15360, rng);
    DevMat dq(hq), dk(hk), dv(hv), dO(ho), dqf(hqf), dkf(hkf), dof(hof), dg(hg), du(hu), dd(hd);
    DevBuf x((size_t)maxm * 15360 * 2), y((size_t)maxm * 15360 * 2);
    cudaMemset(x.p, 0, x.n);
    int cur_m = 0;
    Blas blas(st);
    DevBuf deq((size_t)15360 * 3840 * 2), y2((size_t)maxm * 15360 * 2);
    int cur_v = 0;
    auto lin = [&](const W4Mat& W, bf16* out) {   // the engine's long-forward path: dequantize, cuBLAS
      w4_dequant(W, deq.as<bf16>(), st);
      blas.gemm_bf16(x.as<bf16>(), deq.as<bf16>(), out, cur_m, W.N, W.K);
    };
    auto layer = [&](bool full, int M, int part = -1) {   // part 0 … 3: that matrix alone
      cur_m = M;
      if (cur_v == 100) {   // "variant" 100: dequant + cuBLAS per matrix (gate and up separately; without the engine's gelu_mul pass)
        if (part < 0 || part == 0) { lin(full ? dqf.W : dq.W, y.as<bf16>()); lin(full ? dkf.W : dk.W, y.as<bf16>()); if (!full) lin(dv.W, y.as<bf16>()); }
        if (part < 0 || part == 1) lin(full ? dof.W : dO.W, y.as<bf16>());
        if (part < 0 || part == 2) { lin(dg.W, y.as<bf16>()); lin(du.W, y2.as<bf16>()); }
        if (part < 0 || part == 3) lin(dd.W, y.as<bf16>());
        return;
      }
      W4Multi J;
      J.n = full ? 2 : 3;
      J.W[0] = full ? dqf.W : dq.W; J.W[1] = full ? dkf.W : dk.W; J.W[2] = dv.W;
      for (int i = 0; i < 3; ++i) { J.yb[i] = y.as<bf16>(); J.ldy[i] = 8192; }
      if (part < 0 || part == 0) w4_gemm_multi(J, x.as<bf16>(), 3840, M, st);
      W4Multi O; O.n = 1; O.W[0] = full ? dof.W : dO.W; O.yb[0] = y.as<bf16>(); O.ldy[0] = 3840;
      if (part < 0 || part == 1) w4_gemm_multi(O, x.as<bf16>(), O.W[0].K, M, st);
      if (part < 0 || part == 2) w4_gemm_gu(dg.W, du.W, x.as<bf16>(), 3840, M, y.as<bf16>(), 15360, st);
      W4Multi D; D.n = 1; D.W[0] = dd.W; D.yb[0] = y.as<bf16>(); D.ldy[0] = 3840;
      if (part < 0 || part == 3) w4_gemm_multi(D, x.as<bf16>(), 15360, M, st);
    };
    printf("  prefill GEMM per layer (cold L2, µs, median of %d)    variant:", reps);
    for (int v : vs) printf(" %7d", v);
    printf("\n");
    for (int full = 0; full < 2; ++full)
      for (int M : lms) {
        std::vector<std::vector<double>> t(vs.size());
        for (int r = 0; r < reps; ++r)
          for (size_t i = 0; i < vs.size(); ++i) {
            w4_gemm_set_variant(vs[i]);
            cur_v = vs[i];
            t[i].push_back(cold_us([&] { layer(full, M); }, 1));
          }
        w4_gemm_set_variant(-2);
        printf("  %-7s layer M %3d %40s", full ? "full" : "sliding", M, "");
        for (auto& ti : t) { std::sort(ti.begin(), ti.end()); printf(" %7.1f", ti[ti.size() / 2]); }
        printf("\n");
        if (!getenv("HIVE_W4_PARTS") || full) continue;   // HIVE_W4_PARTS=1: the sliding layer's matrices one by one too
        const char* pn[4] = {"q+k+v", "o", "gate+up", "down"};
        for (int part = 0; part < 4; ++part) {
          std::vector<std::vector<double>> tp(vs.size());
          for (int r = 0; r < reps; ++r)
            for (size_t i = 0; i < vs.size(); ++i) {
              w4_gemm_set_variant(vs[i]);
              cur_v = vs[i];
              tp[i].push_back(cold_us([&] { layer(false, M, part); }, 1));
            }
          w4_gemm_set_variant(-2);
          printf("    %-10s M %3d %41s", pn[part], M, "");
          for (auto& ti : tp) { std::sort(ti.begin(), ti.end()); printf(" %7.1f", ti[ti.size() / 2]); }
          printf("\n");
        }
      }
  }
  if (getenv("HIVE_W4_QUICK")) {   // the layer times only (skips the per-matrix table and the decode bandwidth lines)
    const cudaError_t e = cudaGetLastError();
    all &= check(e == cudaSuccess, cudaGetErrorString(e));
    printf("%s\n", all ? "ALL PASS" : "FAIL");
    return all ? 0 : 1;
  }
  // prefill time per call at the model's shapes: GEMV passes of kGemvRows / GEMM / dequant + cuBLAS (the engine's three paths)
  {
    struct PShape { const char* name; int N, K; } ps[] = {{"q+k+v", 8192, 3840}, {"o", 3840, 4096}, {"gate+up", 15360, 3840}, {"down", 3840, 15360}};
    // HIVE_W4_MAXM: largest row count timed (default 2048; a smaller one needs less memory — beside running daemons)
    const int maxm = getenv("HIVE_W4_MAXM") ? atoi(getenv("HIVE_W4_MAXM")) : 2048;
    Blas blas(st);
    DevBuf deq((size_t)15360 * 3840 * 2), outb((size_t)maxm * 15360 * 2), outb2((size_t)maxm * 15360 * 2);
    printf("  prefill (cold L2, µs)           M:");
    const int MsAll[] = {16, 24, 32, 48, 70, 96, 128, 192, 256, 512, 1024, 2048};
    std::vector<int> Ms;
    const int only = getenv("HIVE_W4_ONLY_M") ? atoi(getenv("HIVE_W4_ONLY_M")) : 0;   // profiling: time this row count alone
    for (int M : MsAll) if (M <= maxm && (!only || M == only)) Ms.push_back(M);
    const int nm = (int)Ms.size();
    for (int M : Ms) printf(" %7d", M);
    printf("\n");
    for (const auto& sh : ps) {
      const bool gu = std::string(sh.name) == "gate+up";
      HostMat h = make(sh.N, sh.K, rng), h2 = make(gu ? sh.N : 64, sh.K, rng);
      DevMat d(h), d2(h2);
      DevBuf x((size_t)maxm * sh.K * 2);
      cudaMemset(x.p, 0, x.n);
      double tv[3][12];
      for (int mi = 0; mi < nm; ++mi) {
        const int M = Ms[mi];
        tv[0][mi] = cold_us([&] {
          for (int r0 = 0; r0 < M; r0 += kGemvRows) {
            const int R = std::min(kGemvRows, M - r0);
            if (gu) w4_gemv_gu(d.W, d2.W, x.as<bf16>() + (size_t)r0 * sh.K, sh.K, R, outb.as<bf16>() + (size_t)r0 * sh.N, sh.N, nullptr, st);
            else w4_gemv(d.W, x.as<bf16>() + (size_t)r0 * sh.K, sh.K, R, outb.as<bf16>() + (size_t)r0 * sh.N, nullptr, sh.N, nullptr, st);
          }
        }, 10);
        tv[1][mi] = cold_us([&] {
          if (gu) w4_gemm_gu(d.W, d2.W, x.as<bf16>(), sh.K, M, outb.as<bf16>(), sh.N, st);
          else { W4Multi J; J.n = 1; J.W[0] = d.W; J.yb[0] = outb.as<bf16>(); J.ldy[0] = sh.N; w4_gemm_multi(J, x.as<bf16>(), sh.K, M, st); }
        }, 10);
        tv[2][mi] = cold_us([&] {
          w4_dequant(d.W, deq.as<bf16>(), st);
          blas.gemm_bf16(x.as<bf16>(), deq.as<bf16>(), outb.as<bf16>(), M, sh.N, sh.K);
          if (gu) { w4_dequant(d2.W, deq.as<bf16>(), st); blas.gemm_bf16(x.as<bf16>(), deq.as<bf16>(), outb2.as<bf16>(), M, sh.N, sh.K); }
        }, 10);
      }
      const char* lab[3] = {"GEMV passes", "GEMM", "dequant+cuBLAS"};
      for (int k = 0; k < 3; ++k) {
        printf("  %-8s %-22s", sh.name, lab[k]);
        for (int mi = 0; mi < nm; ++mi) printf(" %7.1f", tv[k][mi]);
        printf("\n");
      }
    }
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
