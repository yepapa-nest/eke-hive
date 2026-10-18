// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3-ASR kernels (src/qasr/qasr_kernels.cu) against straightforward CPU references on small shapes, plus the decode GEMV bandwidth
//   at the real decoder layer shapes (one layer's four weight matrices, m = 1 and 2) next to cuBLAS for the same products.
//   Needs ~700 MB of free VRAM. Exit code = number of failed checks.
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/qasr/qasr_kernels.h"
#include "test_util.h"

using hive::bf16;
using namespace hive::qasr;

static int fails = 0;
static void ok(bool c, const char* what, double v = 0) {
  fails += !c;
  printf("%s %s (%.3g)\n", c ? "ok  " : "FAIL", what, v);
}

static std::mt19937 rng(7);
static std::vector<float> randv(size_t n, float s = 1.f) {
  std::normal_distribution<float> nd(0.f, s);
  std::vector<float> v(n);
  for (auto& x : v) x = nd(rng);
  return v;
}
static uint16_t f2b(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float b2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }
static std::vector<float> round_bf16(std::vector<float> v) {
  for (auto& x : v) x = b2f(f2b(x));
  return v;
}
template <class T> static T* dev(const std::vector<T>& h) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, h.size() * sizeof(T) + 16));
  CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
static bf16* devb(const std::vector<float>& h) {
  std::vector<uint16_t> b(h.size());
  for (size_t i = 0; i < h.size(); ++i) b[i] = f2b(h[i]);
  return reinterpret_cast<bf16*>(dev(b));
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
static std::vector<float> hostb(const bf16* d, size_t n) {
  auto b = host(reinterpret_cast<const uint16_t*>(d), n);
  std::vector<float> f(n);
  for (size_t i = 0; i < n; ++i) f[i] = b2f(b[i]);
  return f;
}
static double rel(const std::vector<float>& g, const std::vector<float>& w) { return hive::test::closeness(g, w).rel; }

static float silu(float x) { return x / (1.f + std::exp(-x)); }
static float gelu(float x) { return 0.5f * x * (1.f + std::erf(x / std::sqrt(2.f))); }

// CPU attention: q [nq][H][D], k/v rows j at kv + j*stride + g*D; key range [lo(i), hi(i))
template <class LoHi>
static std::vector<float> cpu_attn(const std::vector<float>& q, int nq, int H, int D, int rep, const std::vector<float>& k, const std::vector<float>& v,
                                   long stride, LoHi lohi) {
  std::vector<float> o((size_t)nq * H * D, 0.f);
  for (int i = 0; i < nq; ++i)
    for (int h = 0; h < H; ++h) {
      const int g = h / rep;
      int lo, hi;
      lohi(i, lo, hi);
      std::vector<double> s(hi - lo);
      double mx = -1e30;
      for (int j = lo; j < hi; ++j) {
        double d = 0;
        for (int e = 0; e < D; ++e) d += (double)q[((size_t)i * H + h) * D + e] * k[j * stride + g * D + e];
        s[j - lo] = d / std::sqrt((double)D);
        mx = std::max(mx, s[j - lo]);
      }
      double l = 0;
      for (auto& x : s) { x = std::exp(x - mx); l += x; }
      for (int j = lo; j < hi; ++j)
        for (int e = 0; e < D; ++e) o[((size_t)i * H + h) * D + e] += (float)(s[j - lo] / l * v[j * stride + g * D + e]);
    }
  return o;
}

static void test_gemv() {
  for (int K : {2048, 6144}) {
    for (int m : {1, 2}) {
      const int N = 520;
      auto w = round_bf16(randv((size_t)N * K, 0.05f));
      auto x = round_bf16(randv((size_t)m * K));
      auto y0 = randv((size_t)m * N);
      std::vector<float> ref((size_t)m * N);
      for (int r = 0; r < m; ++r)
        for (int n = 0; n < N; ++n) {
          double s = 0;
          for (int k = 0; k < K; ++k) s += (double)w[(size_t)n * K + k] * x[(size_t)r * K + k];
          ref[(size_t)r * N + n] = (float)s;
        }
      bf16 *dw = devb(w), *dx = devb(x);
      float* dy = dev(std::vector<float>((size_t)m * N));
      gemv(dw, dx, m, N, K, dy, 0);
      char what[96];
      snprintf(what, sizeof what, "gemv K=%d m=%d rel err", K, m);
      ok(rel(host(dy, (size_t)m * N), ref) < 1e-5, what, rel(host(dy, (size_t)m * N), ref));
      CUDA_CHECK(cudaMemcpy(dy, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice));
      gemv_add(dw, dx, m, N, K, dy, 0);
      auto refa = ref;
      for (size_t i = 0; i < refa.size(); ++i) refa[i] += y0[i];
      snprintf(what, sizeof what, "gemv_add K=%d m=%d rel err", K, m);
      ok(rel(host(dy, (size_t)m * N), refa) < 1e-5, what, rel(host(dy, (size_t)m * N), refa));
      if (K == 2048) {
        const int inter = N / 2;
        bf16* dout;
        CUDA_CHECK(cudaMalloc(&dout, (size_t)m * inter * 2));
        gemv_swiglu(dw, dx, m, inter, K, dout, 0);
        std::vector<float> rs((size_t)m * inter);
        for (int r = 0; r < m; ++r)
          for (int i = 0; i < inter; ++i) rs[(size_t)r * inter + i] = silu(ref[(size_t)r * N + i]) * ref[(size_t)r * N + inter + i];
        snprintf(what, sizeof what, "gemv_swiglu m=%d rel err", m);
        ok(rel(hostb(dout, rs.size()), rs) < 8e-3, what, rel(hostb(dout, rs.size()), rs));
        cudaFree(dout);
      }
      cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }
  }
}

static void test_norms() {
  const int rows = 5, n = 2048;
  auto x = randv((size_t)rows * n, 2.f), w = randv(n), b = randv(n);
  float *dx = dev(x), *dw = dev(w), *db = dev(b);
  bf16* dy;
  CUDA_CHECK(cudaMalloc(&dy, (size_t)rows * n * 2));
  rmsnorm(dx, dw, dy, rows, n, 1e-6f, 0);
  std::vector<float> ref((size_t)rows * n);
  for (int r = 0; r < rows; ++r) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)x[r * n + i] * x[r * n + i];
    const double rr = 1.0 / std::sqrt(s / n + 1e-6);
    for (int i = 0; i < n; ++i) ref[r * n + i] = (float)(x[r * n + i] * rr * w[i]);
  }
  ok(rel(hostb(dy, ref.size()), ref) < 4e-3, "rmsnorm rel err", rel(hostb(dy, ref.size()), ref));
  layernorm(dx, dw, db, dy, rows, 1024, 1e-5f, 0);  // n = 1024 view over the first rows*1024 values
  std::vector<float> rl((size_t)rows * 1024);
  for (int r = 0; r < rows; ++r) {
    double mu = 0, var = 0;
    for (int i = 0; i < 1024; ++i) mu += x[r * 1024 + i];
    mu /= 1024;
    for (int i = 0; i < 1024; ++i) var += (x[r * 1024 + i] - mu) * (x[r * 1024 + i] - mu);
    var /= 1024;
    for (int i = 0; i < 1024; ++i) rl[r * 1024 + i] = (float)((x[r * 1024 + i] - mu) / std::sqrt(var + 1e-5) * w[i] + b[i]);
  }
  ok(rel(hostb(dy, rl.size()), rl) < 4e-3, "layernorm rel err", rel(hostb(dy, rl.size()), rl));
  // epilogues
  float* dout;
  CUDA_CHECK(cudaMalloc(&dout, (size_t)rows * 1024 * 4));
  CUDA_CHECK(cudaMemcpy(dout, x.data(), (size_t)rows * 1024 * 4, cudaMemcpyHostToDevice));
  bias_epilogue(dx, db, rows, 1024, Epi::kGeluBf16, dy, nullptr, 0);
  bias_epilogue(dx, db, rows, 1024, Epi::kAddF32, nullptr, dout, 0);
  std::vector<float> rg((size_t)rows * 1024), ra((size_t)rows * 1024);
  for (int i = 0; i < rows * 1024; ++i) { rg[i] = gelu(x[i] + b[i % 1024]); ra[i] = 2 * x[i] + b[i % 1024]; }
  ok(rel(hostb(dy, rg.size()), rg) < 4e-3, "bias + gelu epilogue", rel(hostb(dy, rg.size()), rg));
  ok(rel(host(dout, ra.size()), ra) < 1e-6, "bias + residual add epilogue", rel(host(dout, ra.size()), ra));
  cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dy); cudaFree(dout);
}

static void test_qk_rope() {
  const int rows = 3, H = 4, KV = 2, D = 128, maxpos = 64, nslot = 2;
  const int nh = H + 2 * KV;
  auto qkv = randv((size_t)rows * nh * D), qn = randv(D), kn = randv(D);
  std::vector<float> cs((size_t)maxpos * 64), sn((size_t)maxpos * 64);
  for (int p = 0; p < maxpos; ++p)
    for (int i = 0; i < 64; ++i) {
      const double a = p * std::pow(1e6, -2.0 * i / 128);
      cs[p * 64 + i] = (float)std::cos(a); sn[p * 64 + i] = (float)std::sin(a);
    }
  std::vector<int32_t> pos = {5, 17, 63}, slot = {1, 0, 1};
  const long ss = (long)maxpos * KV * D;
  bf16 *dq, *dk, *dv;
  CUDA_CHECK(cudaMalloc(&dq, (size_t)rows * H * D * 2));
  CUDA_CHECK(cudaMalloc(&dk, (size_t)nslot * ss * 2));
  CUDA_CHECK(cudaMalloc(&dv, (size_t)nslot * ss * 2));
  qk_norm_rope_store(dev(qkv), dev(qn), dev(kn), dev(cs), dev(sn), dev(pos), dev(slot), rows, H, KV, 1e-6f, dq, dk, dv, ss, 0);
  auto gq = hostb(dq, (size_t)rows * H * D), gk = hostb(dk, (size_t)nslot * ss), gv = hostb(dv, (size_t)nslot * ss);
  double worst = 0;
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < nh; ++h) {
      const float* src = &qkv[((size_t)r * nh + h) * D];
      std::vector<float> y(D);
      if (h < H + KV) {
        double s = 0;
        for (int i = 0; i < D; ++i) s += (double)src[i] * src[i];
        const double rr = 1.0 / std::sqrt(s / D + 1e-6);
        std::vector<float> xn(D);
        for (int i = 0; i < D; ++i) xn[i] = (float)(src[i] * rr * (h < H ? qn[i] : kn[i]));
        for (int i = 0; i < 64; ++i) {
          const float c = cs[pos[r] * 64 + i], s2 = sn[pos[r] * 64 + i];
          y[i] = xn[i] * c - xn[i + 64] * s2;
          y[i + 64] = xn[i + 64] * c + xn[i] * s2;
        }
      } else {
        for (int i = 0; i < D; ++i) y[i] = src[i];
      }
      const float* got = h < H ? &gq[((size_t)r * H + h) * D]
                               : (h < H + KV ? &gk[slot[r] * ss + (long)pos[r] * KV * D + (h - H) * D]
                                             : &gv[slot[r] * ss + (long)pos[r] * KV * D + (h - H - KV) * D]);
      for (int i = 0; i < D; ++i) worst = std::max(worst, (double)std::fabs(got[i] - y[i]) / (std::fabs(y[i]) + 0.05));
    }
  ok(worst < 1e-2, "q/k norm + rope + kv store (max rel err)", worst);
}

static void test_attention() {
  // decoder prefill: chunk of 37 queries at pos0 = 20 over a cache of 57 rows (slot view)
  {
    const int H = 4, KV = 2, D = 128, nq = 37, pos0 = 20, nk = pos0 + nq;
    const long stride = KV * D;
    auto q = round_bf16(randv((size_t)nq * H * D)), k = round_bf16(randv((size_t)nk * stride)), v = round_bf16(randv((size_t)nk * stride));
    bf16* dout;
    CUDA_CHECK(cudaMalloc(&dout, (size_t)nq * H * D * 2));
    attn_prefill_causal(devb(q), devb(k), devb(v), nq, pos0, H, KV, dout, 0);
    auto ref = cpu_attn(q, nq, H, D, H / KV, k, v, stride, [&](int i, int& lo, int& hi) { lo = 0; hi = pos0 + i + 1; });
    ok(rel(hostb(dout, ref.size()), ref) < 6e-3, "attn_prefill_causal rel err", rel(hostb(dout, ref.size()), ref));
  }
  // encoder windows: n = 230 tokens, window 104
  {
    const int H = 3, D = 64, n = 230, win = 104;
    auto qkv = round_bf16(randv((size_t)n * 3 * H * D));
    std::vector<float> q((size_t)n * H * D), k((size_t)n * H * D), v((size_t)n * H * D);
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < H * D; ++c) {
        q[(size_t)i * H * D + c] = qkv[(size_t)i * 3 * H * D + c];
        k[(size_t)i * H * D + c] = qkv[(size_t)i * 3 * H * D + H * D + c];
        v[(size_t)i * H * D + c] = qkv[(size_t)i * 3 * H * D + 2 * H * D + c];
      }
    bf16* dout;
    CUDA_CHECK(cudaMalloc(&dout, (size_t)n * H * D * 2));
    attn_encoder(devb(qkv), n, H, win, dout, 0);
    auto ref = cpu_attn(q, n, H, D, 1, k, v, (long)H * D, [&](int i, int& lo, int& hi) { lo = i / win * win; hi = std::min(lo + win, n); });
    ok(rel(hostb(dout, ref.size()), ref) < 6e-3, "attn_encoder (block-diagonal windows) rel err", rel(hostb(dout, ref.size()), ref));
  }
  // decode: 2 sequences in slots 1 and 0 with lengths 300 and 9
  {
    const int H = 16, KV = 8, D = 128, maxpos = 320, nslot = 2, m = 2;
    const long ss = (long)maxpos * KV * D;
    auto q = round_bf16(randv((size_t)m * H * D)), k = round_bf16(randv((size_t)nslot * ss)), v = round_bf16(randv((size_t)nslot * ss));
    std::vector<int32_t> pos = {299, 8}, slot = {1, 0};
    bf16* dout;
    CUDA_CHECK(cudaMalloc(&dout, (size_t)m * H * D * 2));
    attn_decode(devb(q), devb(k), devb(v), dev(pos), dev(slot), m, H, KV, ss, dout, 0);
    auto got = hostb(dout, (size_t)m * H * D);
    double worst = 0;
    for (int i = 0; i < m; ++i) {
      std::vector<float> qi(q.begin() + (size_t)i * H * D, q.begin() + (size_t)(i + 1) * H * D);
      std::vector<float> ki(k.begin() + slot[i] * ss, k.begin() + (slot[i] + 1) * ss), vi(v.begin() + slot[i] * ss, v.begin() + (slot[i] + 1) * ss);
      auto ref = cpu_attn(qi, 1, H, D, H / KV, ki, vi, (long)KV * D, [&](int, int& lo, int& hi) { lo = 0; hi = pos[i] + 1; });
      std::vector<float> g(got.begin() + (size_t)i * H * D, got.begin() + (size_t)(i + 1) * H * D);
      worst = std::max(worst, rel(g, ref));
    }
    ok(worst < 6e-3, "attn_decode (2 sequences, different slots/lengths) rel err", worst);
  }
}

static void test_argmax_embed() {
  const int rows = 3, V = 151936;
  auto lg = randv((size_t)rows * V);
  lg[17] = 50.f; lg[V + 151935] = 60.f; lg[2 * V + 5] = 40.f; lg[2 * V + 900] = 40.f;  // tie -> first index
  int32_t* dids;
  CUDA_CHECK(cudaMalloc(&dids, rows * 4));
  argmax_rows(dev(lg), rows, V, dids, 0);
  auto ids = host(dids, rows);
  ok(ids[0] == 17 && ids[1] == 151935 && ids[2] == 5, "argmax (incl. last index and first-index tie)", ids[2]);
  const int n = 64;
  auto table = round_bf16(randv((size_t)100 * n));
  std::vector<int32_t> tok = {3, 99, 0};
  float* dx;
  CUDA_CHECK(cudaMalloc(&dx, (size_t)3 * n * 4));
  embed_rows(devb(table), dev(tok), dx, 3, n, 0);
  auto x = host(dx, (size_t)3 * n);
  bool same = true;
  for (int r = 0; r < 3; ++r)
    for (int i = 0; i < n; ++i) same &= x[r * n + i] == table[(size_t)tok[r] * n + i];
  ok(same, "embed rows exact");
}

static void test_conv() {
  // conv1 + im2col (both orders) vs direct 3x3/s2/p1 convolutions on a 2-chunk input with a padded tail
  const int n_mels = 128, nfr = 137, C = 16, nc = 2, lp = 100;
  auto mel = randv((size_t)n_mels * nfr);
  auto w1 = randv((size_t)C * 9, 0.3f), b1 = randv(C, 0.1f);
  const int F1 = 64, T1 = 50;
  bf16* c1;
  CUDA_CHECK(cudaMalloc(&c1, (size_t)nc * F1 * T1 * C * 2));
  conv1_gelu(dev(mel), n_mels, nfr, 0, nc, lp, dev(w1), dev(b1), C, c1, 0);
  auto g1 = hostb(c1, (size_t)nc * F1 * T1 * C);
  std::vector<float> r1(g1.size());
  for (int n = 0; n < nc; ++n) {
    const int len = std::min(100, nfr - n * 100);
    for (int f = 0; f < F1; ++f)
      for (int t = 0; t < T1; ++t)
        for (int o = 0; o < C; ++o) {
          double s = b1[o];
          for (int kh = 0; kh < 3; ++kh)
            for (int kw = 0; kw < 3; ++kw) {
              const int fi = 2 * f + kh - 1, ti = 2 * t + kw - 1;
              if (fi < 0 || fi >= n_mels || ti < 0 || ti >= len) continue;
              s += w1[o * 9 + kh * 3 + kw] * mel[(size_t)fi * nfr + n * 100 + ti];
            }
          r1[(((size_t)n * F1 + f) * T1 + t) * C + o] = gelu((float)s);
        }
  }
  ok(rel(g1, r1) < 4e-3, "conv1 + gelu (padded tail chunk) rel err", rel(g1, r1));
  const int Fo = 32, To = 25;
  bf16* col;
  CUDA_CHECK(cudaMalloc(&col, (size_t)nc * Fo * To * 9 * C * 2));
  for (int tm = 0; tm < 2; ++tm) {
    im2col_s2(c1, nc, F1, T1, C, tm == 1, col, 0);
    auto gc = hostb(col, (size_t)nc * Fo * To * 9 * C);
    bool same = true;
    for (int n = 0; n < nc; ++n)
      for (int fo = 0; fo < Fo; ++fo)
        for (int to = 0; to < To; ++to) {
          const long row = tm ? ((long)n * To + to) * Fo + fo : ((long)n * Fo + fo) * To + to;
          for (int tap = 0; tap < 9; ++tap)
            for (int c = 0; c < C; ++c) {
              const int fi = 2 * fo + tap / 3 - 1, ti = 2 * to + tap % 3 - 1;
              const float want = (fi < 0 || fi >= F1 || ti < 0 || ti >= T1) ? 0.f : g1[(((size_t)n * F1 + fi) * T1 + ti) * C + c];
              same &= gc[row * 9 * C + tap * C + c] == want;
            }
        }
    ok(same, tm ? "im2col (time-major rows) exact" : "im2col (mel-major rows) exact");
  }
}

static void bench_gemv() {
  // kCopies distinct layers (~600 MB) are cycled so the weights stream from DRAM, not from the 128 MB L2 as one repeated layer would
  constexpr int kCopies = 6;
  const int h = 2048, inter = 6144, nqkv = 4096;
  std::vector<std::pair<int, int>> mats = {{nqkv, h}, {h, h}, {2 * inter, h}, {h, inter}};
  size_t bytes = 0;
  std::vector<bf16*> ws;
  for (int cpy = 0; cpy < kCopies; ++cpy)
    for (auto [n, k] : mats) {
      bf16* p;
      CUDA_CHECK(cudaMalloc(&p, (size_t)n * k * 2));
      CUDA_CHECK(cudaMemset(p, 0x3c, (size_t)n * k * 2));
      ws.push_back(p);
      if (cpy == 0) bytes += (size_t)n * k * 2;
    }
  bf16 *x, *act;
  float *out, *res;
  CUDA_CHECK(cudaMalloc(&x, (size_t)4 * inter * 2));
  CUDA_CHECK(cudaMemset(x, 0, (size_t)4 * inter * 2));
  CUDA_CHECK(cudaMalloc(&act, (size_t)4 * inter * 2));
  CUDA_CHECK(cudaMalloc(&out, (size_t)4 * 2 * inter * 4));
  CUDA_CHECK(cudaMalloc(&res, (size_t)4 * h * 4));
  cublasHandle_t cb;
  cublasCreate(&cb);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int m : {1, 2}) {
    for (int lib = 0; lib < 2; ++lib) {
      int cur = 0;
      [[maybe_unused]] auto layer = [&] {
        bf16** w = &ws[(size_t)(cur++ % kCopies) * mats.size()];
        if (lib == 0) {
          gemv(w[0], x, m, nqkv, h, out, 0);
          gemv_add(w[1], x, m, h, h, res, 0);
          gemv_swiglu(w[2], x, m, inter, h, act, 0);
          gemv_add(w[3], act, m, h, inter, res, 0);
        } else {
          const float a = 1.f, b0 = 0.f;
          for (size_t i = 0; i < mats.size(); ++i)
            cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, mats[i].first, m, mats[i].second, &a, w[i], CUDA_R_16BF, mats[i].second, x, CUDA_R_16BF,
                         mats[i].second, &b0, out, CUDA_R_32F, mats[i].first, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
      };
      // captured in a CUDA graph (as the engine's decode step is): kCopies * 4 layers per launch, no launch gaps
      cudaStream_t cs;
      cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
      cublasSetStream(cb, cs);
      cudaGraph_t g;
      cudaGraphExec_t ge;
      const int per_graph = kCopies * 4;
      CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < per_graph; ++i) {
        bf16** w = &ws[(size_t)(i % kCopies) * mats.size()];
        if (lib == 0) {
          gemv(w[0], x, m, nqkv, h, out, cs);
          gemv_add(w[1], x, m, h, h, res, cs);
          gemv_swiglu(w[2], x, m, inter, h, act, cs);
          gemv_add(w[3], act, m, h, inter, res, cs);
        } else {
          const float a = 1.f, b0 = 0.f;
          for (size_t k = 0; k < mats.size(); ++k)
            cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, mats[k].first, m, mats[k].second, &a, w[k], CUDA_R_16BF, mats[k].second, x, CUDA_R_16BF,
                         mats[k].second, &b0, out, CUDA_R_32F, mats[k].first, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
      }
      CUDA_CHECK(cudaStreamEndCapture(cs, &g));
      CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
      for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, cs);
      const int reps = 10, it = reps * per_graph;
      cudaEventRecord(e0, cs);
      for (int i = 0; i < reps; ++i) cudaGraphLaunch(ge, cs);
      cudaEventRecord(e1, cs);
      cudaEventSynchronize(e1);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      printf("info decoder layer GEMVs (%.1f MB) m=%d %-6s %.1f us/layer  %.0f GB/s\n", bytes / 1e6, m, lib ? "cuBLAS" : "qasr", 1e3 * ms / it,
             bytes * it / (ms * 1e-3) / 1e9);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  cublasDestroy(cb);
  for (auto p : ws) cudaFree(p);
}

int main() {
  test_gemv();
  test_norms();
  test_qk_rope();
  test_attention();
  test_argmax_embed();
  test_conv();
  CUDA_CHECK(cudaDeviceSynchronize());
  bench_gemv();
  printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
  return fails;
}
