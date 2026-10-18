// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// qtts kernels against plain CPU references (double accumulation) on small random inputs: GEMV (all modes / prologues / M), q/k norm +
//   RoPE + KV store, decode attention (split and unsplit), prefill attention, sampling (greedy with penalty / suppression, top-k
//   membership and frequencies), causal im2col with SnakeBeta, ConvNeXt front, codec attention window, codec output layer, RVQ.
//   Exit code 0 = all pass. Needs < 200 MiB of VRAM.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <cstdlib>
#include <random>
#include <vector>

#include "hive/qtts/qtts_kernels.h"

using namespace hive;
using namespace hive::qtts;

static int g_fail = 0;
static void check(const char* name, double err, double tol) {
  const bool ok = err <= tol && std::isfinite(err);
  printf("%-34s err %.3e  tol %.1e  %s\n", name, err, tol, ok ? "ok" : "FAIL");
  if (!ok) ++g_fail;
}

static std::mt19937 rng(1234);
static std::vector<float> randv(size_t n, float s = 1.f) {
  std::normal_distribution<float> d(0.f, s);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}
static std::vector<float> round_bf16(std::vector<float> v) {
  for (auto& x : v) x = __bfloat162float(__float2bfloat16(x));
  return v;
}
template <class T>
static T* dev(const std::vector<T>& v) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)));
  if (!v.empty()) CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
static bf16* dev_bf16(const std::vector<float>& v) {
  std::vector<bf16> b(v.size());
  for (size_t i = 0; i < v.size(); ++i) b[i] = __float2bfloat16(v[i]);
  return dev(b);
}
template <class T>
static std::vector<T> host(const T* p, size_t n) {
  std::vector<T> v(n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
static double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
  double m = 0, s = 0;
  for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(a[i] - b[i])); s = std::max(s, std::fabs(b[i])); }
  return m / (s + 1e-30);
}
static double snake(double x, double ea, double ib) { const double s = std::sin(x * ea); return x + s * s * ib; }

static void test_gemv() {
  cudaStream_t st = 0;
  for (int M : {1, 2, 4}) {
    for (int mode : {kGemvStore, kGemvAccum, kGemvSiluMul}) {
      for (int norm = 0; norm < 2; ++norm) {
        const int N = 200, K = 768;
        const int rows = mode == kGemvSiluMul ? 2 * N : N;
        auto W = round_bf16(randv((size_t)rows * K, 0.05f));
        auto x = randv((size_t)M * K);
        auto nw = randv(K);
        auto bias = randv(N);
        auto scale = randv(N);
        auto y0 = randv((size_t)M * N);
        float* dx = dev(x); float* dnw = dev(nw); float* db = dev(bias); float* ds = dev(scale); float* dy = dev(y0);
        bf16* dW = dev_bf16(W);
        GemvParams p;
        p.W = dW; p.N = N; p.K = K; p.M = M; p.x = dx; p.ldx = K; p.norm_w = norm ? dnw : nullptr; p.eps = 1e-6f;
        p.bias = mode == kGemvStore ? db : nullptr; p.scale = mode == kGemvAccum ? ds : nullptr; p.y = dy; p.ldy = N; p.mode = mode;
        gemv(p, st);
        auto y = host(dy, (size_t)M * N);
        std::vector<double> ref((size_t)M * N);
        for (int m = 0; m < M; ++m) {
          std::vector<double> xs(K);
          double ss = 0;
          for (int k = 0; k < K; ++k) ss += (double)x[m * K + k] * x[m * K + k];
          const double r = norm ? 1.0 / std::sqrt(ss / K + 1e-6) : 1.0;
          for (int k = 0; k < K; ++k) xs[k] = x[m * K + k] * r * (norm ? nw[k] : 1.0);
          for (int n = 0; n < N; ++n) {
            double a = 0, b = 0;
            for (int k = 0; k < K; ++k) a += W[(size_t)n * K + k] * xs[k];
            if (mode == kGemvSiluMul) for (int k = 0; k < K; ++k) b += W[(size_t)(n + N) * K + k] * xs[k];
            double o;
            if (mode == kGemvStore) o = a + bias[n];
            else if (mode == kGemvAccum) o = y0[m * N + n] + scale[n] * a;
            else o = a / (1 + std::exp(-a)) * b;
            ref[(size_t)m * N + n] = o;
          }
        }
        char name[64];
        snprintf(name, sizeof name, "gemv M%d mode%d norm%d", M, mode, norm);
        check(name, rel_err(y, ref), 2e-5);
        cudaFree(dx); cudaFree(dnw); cudaFree(db); cudaFree(ds); cudaFree(dy); cudaFree(dW);
      }
    }
  }
  // table input (bf16 row picked by a device index) with norm
  {
    const int N = 64, K = 512, V = 10, id = 7;
    auto W = round_bf16(randv((size_t)N * K, 0.05f));
    auto tab = round_bf16(randv((size_t)V * K));
    auto nw = randv(K);
    bf16* dW = dev_bf16(W); bf16* dt = dev_bf16(tab); float* dnw = dev(nw);
    int* di = dev(std::vector<int>{id});
    float* dy = dev(std::vector<float>(N));
    GemvParams p;
    p.W = dW; p.N = N; p.K = K; p.table = dt; p.idx = di; p.norm_w = dnw; p.eps = 1e-6f; p.y = dy; p.ldy = N;
    gemv(p, 0);
    auto y = host(dy, N);
    double ss = 0;
    for (int k = 0; k < K; ++k) ss += (double)tab[id * K + k] * tab[id * K + k];
    const double r = 1.0 / std::sqrt(ss / K + 1e-6);
    std::vector<double> ref(N);
    for (int n = 0; n < N; ++n) {
      double a = 0;
      for (int k = 0; k < K; ++k) a += W[(size_t)n * K + k] * tab[id * K + k] * r * nw[k];
      ref[n] = a;
    }
    check("gemv table+norm", rel_err(y, ref), 2e-5);
  }
}

// CPU: norm + rope of one head vector
static void cpu_norm_rope(const float* x, const float* w, int pos, double theta, double* out, bool norm, bool rope) {
  const int D = 128;
  double ss = 0;
  for (int d = 0; d < D; ++d) ss += (double)x[d] * x[d];
  const double r = norm ? 1.0 / std::sqrt(ss / D + 1e-6) : 1.0;
  std::vector<double> y(D);
  for (int d = 0; d < D; ++d) y[d] = x[d] * r * (norm ? w[d] : 1.0);
  if (!rope) { for (int d = 0; d < D; ++d) out[d] = y[d]; return; }
  for (int i = 0; i < 64; ++i) {
    const double a = pos / std::pow(theta, 2.0 * i / D);
    out[i] = y[i] * std::cos(a) - y[i + 64] * std::sin(a);
    out[i + 64] = y[i + 64] * std::cos(a) + y[i] * std::sin(a);
  }
}

static void test_attention() {
  const int H = 16, KV = 8, D = 128, cap = 1024, splits = 16;
  const double theta = 1e6;
  std::vector<float> co((size_t)cap * 64), si((size_t)cap * 64);
  for (int p = 0; p < cap; ++p)
    for (int i = 0; i < 64; ++i) {
      const double a = p / std::pow(theta, 2.0 * i / D);
      co[p * 64 + i] = (float)std::cos(a);
      si[p * 64 + i] = (float)std::sin(a);
    }
  float* dco = dev(co); float* dsi = dev(si);
  auto qn = randv(D), kn = randv(D);
  float* dqn = dev(qn); float* dkn = dev(kn);
  bf16 *kc, *vc;
  CUDA_CHECK(cudaMalloc(&kc, (size_t)KV * cap * D * 2));
  CUDA_CHECK(cudaMalloc(&vc, (size_t)KV * cap * D * 2));
  // fill positions 0..P with prompt rows (M = P + 1) through the store kernel, then attend from the last row
  for (int P : {0, 37, 700}) {
    const int M = P + 1;
    auto qkv = randv((size_t)M * (H + 2 * KV) * D);
    float* dqkv = dev(qkv);
    float* dq = dev(std::vector<float>((size_t)M * H * D));
    qk_norm_rope_store(dqkv, M, H, KV, D, dqn, dkn, 1e-6f, dco, dsi, nullptr, 0, dq, kc, vc, cap, 0);
    // CPU reference K, V (bf16-rounded like the cache), q of the last row
    std::vector<double> K((size_t)M * KV * D), V((size_t)M * KV * D), Q((size_t)M * H * D);
    for (int m = 0; m < M; ++m) {
      const float* row = qkv.data() + (size_t)m * (H + 2 * KV) * D;
      for (int h = 0; h < H; ++h) cpu_norm_rope(row + h * D, qn.data(), m, theta, &Q[((size_t)m * H + h) * D], true, true);
      for (int h = 0; h < KV; ++h) {
        cpu_norm_rope(row + (H + h) * D, kn.data(), m, theta, &K[((size_t)m * KV + h) * D], true, true);
        for (int d = 0; d < D; ++d) V[((size_t)m * KV + h) * D + d] = row[(H + KV + h) * D + d];
      }
    }
    for (auto& x : K) x = __bfloat162float(__float2bfloat16((float)x));
    for (auto& x : V) x = __bfloat162float(__float2bfloat16((float)x));
    auto qdev = host(dq, (size_t)M * H * D);
    std::vector<double> qref(Q.begin(), Q.end());
    check(("q norm+rope P=" + std::to_string(P)).c_str(), rel_err(qdev, qref), 1e-5);
    auto attn_ref = [&](int m) {
      std::vector<double> o((size_t)H * D);
      for (int h = 0; h < H; ++h) {
        const int g = h / (H / KV);
        std::vector<double> s(m + 1);
        double mx = -1e300;
        for (int k = 0; k <= m; ++k) {
          double a = 0;
          for (int d = 0; d < D; ++d) a += (double)qdev[((size_t)m * H + h) * D + d] * K[((size_t)k * KV + g) * D + d];
          s[k] = a / std::sqrt((double)D);
          mx = std::max(mx, s[k]);
        }
        double l = 0;
        for (auto& x : s) { x = std::exp(x - mx); l += x; }
        for (int d = 0; d < D; ++d) {
          double a = 0;
          for (int k = 0; k <= m; ++k) a += s[k] * V[((size_t)k * KV + g) * D + d];
          o[(size_t)h * D + d] = a / l;
        }
      }
      return o;
    };
    // decode: last row, positions from a device counter
    int* dpos = dev(std::vector<int>{P});
    float* part = dev(std::vector<float>((size_t)H * splits * (D + 2)));
    float* out = dev(std::vector<float>((size_t)H * D));
    const auto ref = attn_ref(P);
    for (int sp : {1, splits}) {
      attn_decode(dq + (size_t)P * H * D, kc, vc, H, KV, D, cap, dpos, 0, sp, part, out, 0);
      if (sp > 1) attn_combine(part, H, D, sp, out, 0);
      auto o = host(out, (size_t)H * D);
      check(("attn_decode P=" + std::to_string(P) + " splits=" + std::to_string(sp)).c_str(), rel_err(o, ref), 1e-5);
    }
    // prefill: all rows
    bf16* po;
    CUDA_CHECK(cudaMalloc(&po, (size_t)M * H * D * 2));
    attn_prefill(dq, kc, vc, M, H, KV, D, cap, po, 0);
    auto pb = host(po, (size_t)M * H * D);
    double worst = 0;
    for (int m : {0, M / 2, M - 1}) {
      const auto r = attn_ref(m);
      std::vector<float> got((size_t)H * D);
      for (size_t i = 0; i < got.size(); ++i) got[i] = __bfloat162float(pb[(size_t)m * H * D + i]);
      worst = std::max(worst, rel_err(got, r));
    }
    check(("attn_prefill P=" + std::to_string(P)).c_str(), worst, 1e-2);
    cudaFree(dqkv); cudaFree(dq); cudaFree(dpos); cudaFree(part); cudaFree(out); cudaFree(po);
  }
  cudaFree(kc); cudaFree(vc); cudaFree(dco); cudaFree(dsi); cudaFree(dqn); cudaFree(dkn);
}

static void test_sample() {
  const int V = 3072;
  Ctl c{};
  c.max_frames = 1 << 20; c.max_pos = 1 << 30; c.eos_frame = -1; c.seed = 42;
  Ctl* dc = dev(std::vector<Ctl>{c});
  int* dtok = dev(std::vector<int>{0});
  // greedy with penalty and suppression: the largest logit is suppressed, the second largest is penalized below the third
  std::vector<float> l = randv(V);
  l[2500] = 50.f;  // suppressed (2048..3071 except 2150)
  l[100] = 10.f;   // in hist: 10 / 1.05 = 9.52
  l[200] = 9.6f;   // wins
  l[2150] = 9.55f; // EOS: suppressed while frame < 2
  std::vector<uint8_t> hist(V, 0);
  hist[100] = 1;
  float* dl = dev(l);
  uint8_t* dh = dev(hist);
  SampleSpec s;
  s.V = V; s.temperature = 0.f; s.rep_penalty = 1.05f; s.sup_lo = 2048; s.sup_hi = 3072; s.keep_id = 2150; s.eos_id = 2150; s.min_frames = 2;
  s.first = 1;
  sample(dl, s, dh, dc, dtok, nullptr, nullptr, 0);
  int t = host(dtok, 1)[0];
  check("sample greedy (penalty, suppress)", t == 200 ? 0 : 1, 0);
  // frame >= min_frames: EOS allowed (9.55 > 9.52 but < 9.6: still 200); raise EOS
  l[2150] = 9.7f;
  CUDA_CHECK(cudaMemcpy(dl, l.data(), V * 4, cudaMemcpyHostToDevice));
  c.frame = 2;
  CUDA_CHECK(cudaMemcpy(dc, &c, sizeof c, cudaMemcpyHostToDevice));
  sample(dl, s, dh, dc, dtok, nullptr, nullptr, 0);
  t = host(dtok, 1)[0];
  const Ctl after = host(dc, 1)[0];
  check("sample greedy EOS -> done", (t == 2150 && after.done == 1 && after.eos_frame == 2) ? 0 : 1, 0);
  // top-k frequencies: 2048 logits, k = 50, T = 0.9, many draws
  const int V2 = 2048, k = 50, draws = 20000;
  std::vector<float> l2 = randv(V2, 2.f);
  l2[5] = l2[6];  // a tie
  float* dl2 = dev(l2);
  c = Ctl{}; c.max_frames = 1 << 20; c.max_pos = 1 << 30; c.eos_frame = -1; c.seed = 7;
  CUDA_CHECK(cudaMemcpy(dc, &c, sizeof c, cudaMemcpyHostToDevice));
  SampleSpec s2;
  s2.V = V2; s2.temperature = 0.9f; s2.top_k = k; s2.cb = 3;
  int* dtoks = dev(std::vector<int>(draws));
  for (int i = 0; i < draws; ++i) sample(dl2, s2, nullptr, dc, dtoks + i, nullptr, nullptr, 0);
  auto toks = host(dtoks, draws);
  std::vector<float> sorted = l2;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());
  const float kth = sorted[k - 1];
  std::vector<double> p(V2, 0.0);
  double Z = 0;
  for (int i = 0; i < V2; ++i)
    if (l2[i] >= kth) { p[i] = std::exp((l2[i] - sorted[0]) / 0.9); Z += p[i]; }
  std::vector<int> cnt(V2, 0);
  int outside = 0;
  for (int x : toks) { if (x < 0 || x >= V2 || l2[x] < kth) ++outside; else ++cnt[x]; }
  double maxdev = 0;
  for (int i = 0; i < V2; ++i) maxdev = std::max(maxdev, std::fabs(cnt[i] / (double)draws - p[i] / Z));
  check("sample top-k: outside the top-k", outside, 0);
  check("sample top-k: max |freq - p|", maxdev, 0.012);
  cudaFree(dc); cudaFree(dtok); cudaFree(dl); cudaFree(dh); cudaFree(dl2); cudaFree(dtoks);
}

static void test_codec_ops() {
  // im2col with snake, dilation 3, state rows; save_state
  {
    const int T = 20, C = 24, k = 7, dil = 3, H = (k - 1) * dil;
    auto x = randv((size_t)T * C), stv = randv((size_t)H * C), ea = randv(C), ib = randv(C);
    for (auto& v : ea) v = std::exp(v * 0.3f);
    float* dx = dev(x); float* ds = dev(stv); float* dea = dev(ea); float* dib = dev(ib);
    bf16* dA;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)T * k * C * 2));
    im2col_causal(dx, ds, T, C, k, dil, dea, dib, dA, k * C, 0);
    auto A = host(dA, (size_t)T * k * C);
    double e = 0;
    for (int t = 0; t < T; ++t)
      for (int j = 0; j < k; ++j)
        for (int c = 0; c < C; ++c) {
          const int r = t + j * dil;
          const double v = r < H ? stv[r * C + c] : x[(r - H) * C + c];
          const double ref = snake(v, ea[c], ib[c]);
          e = std::max(e, std::fabs(__bfloat162float(A[(size_t)t * k * C + j * C + c]) - ref) / (std::fabs(ref) + 1.0));
        }
    check("im2col snake dil3", e, 8e-3);
    save_state(dx, T, C, H > T ? T : H, ds, 0);
    auto s2 = host(ds, (size_t)std::min(H, T) * C);
    double e2 = 0;
    for (int i = 0; i < std::min(H, T) * C; ++i) e2 = std::max(e2, (double)std::fabs(s2[i] - x[(size_t)(T - std::min(H, T)) * C + i]));
    check("save_state", e2, 0);
    cudaFree(dx); cudaFree(ds); cudaFree(dea); cudaFree(dib); cudaFree(dA);
  }
  // ConvNeXt front: depthwise k7 with state + LayerNorm
  {
    const int T = 8, C = 1024;
    auto x = randv((size_t)T * C), stv = randv((size_t)6 * C), w = randv((size_t)C * 7), b = randv(C), lw = randv(C), lb = randv(C);
    float* dx = dev(x); float* ds = dev(stv); float* dw = dev(w); float* db = dev(b); float* dlw = dev(lw); float* dlb = dev(lb);
    bf16* dout;
    CUDA_CHECK(cudaMalloc(&dout, (size_t)T * C * 2));
    convnext_dw_ln(dx, ds, T, C, dw, db, dlw, dlb, dout, 0);
    auto o = host(dout, (size_t)T * C);
    double e = 0;
    for (int t = 0; t < T; ++t) {
      std::vector<double> y(C);
      double mean = 0, var = 0;
      for (int c = 0; c < C; ++c) {
        double a = b[c];
        for (int j = 0; j < 7; ++j) { const int r = t + j; a += w[c * 7 + j] * (r < 6 ? stv[r * C + c] : x[(r - 6) * C + c]); }
        y[c] = a; mean += a;
      }
      mean /= C;
      for (int c = 0; c < C; ++c) var += (y[c] - mean) * (y[c] - mean);
      var /= C;
      for (int c = 0; c < C; ++c) {
        const double ref = (y[c] - mean) / std::sqrt(var + 1e-6) * lw[c] + lb[c];
        e = std::max(e, std::fabs(__bfloat162float(o[(size_t)t * C + c]) - ref) / (std::fabs(ref) + 1.0));
      }
    }
    check("convnext dw+ln", e, 8e-3);
    auto s2 = host(ds, (size_t)6 * C);
    double e2 = 0;
    for (int i = 0; i < 6 * C; ++i) e2 = std::max(e2, (double)std::fabs(s2[i] - x[(size_t)2 * C + i]));
    check("convnext state", e2, 0);
    cudaFree(dx); cudaFree(ds); cudaFree(dw); cudaFree(db); cudaFree(dlw); cudaFree(dlb); cudaFree(dout);
  }
  // codec attention: ring 128, window 72, positions 70..73 then 200..203 (wrapped ring, window cut)
  {
    const int T = 4, H = 16, D = 64, window = 72, ring = 128;
    const int npos = 256;
    std::vector<float> co((size_t)npos * 32), si((size_t)npos * 32);
    for (int p = 0; p < npos; ++p)
      for (int i = 0; i < 32; ++i) {
        const double a = p / std::pow(1e4, 2.0 * i / D);
        co[p * 32 + i] = (float)std::cos(a); si[p * 32 + i] = (float)std::sin(a);
      }
    float* dco = dev(co); float* dsi = dev(si);
    float* kr = dev(std::vector<float>((size_t)H * ring * D));
    float* vr = dev(std::vector<float>((size_t)H * ring * D));
    float* qt = dev(std::vector<float>((size_t)T * H * D));
    float* out = dev(std::vector<float>((size_t)T * H * D));
    // CPU history of k / v / q by position
    std::vector<std::vector<double>> Kp(npos), Vp(npos);
    Ctl c{};
    Ctl* dc = dev(std::vector<Ctl>{c});
    double worst = 0;
    for (int f0 = 0; f0 < 208; f0 += T) {
      auto qkv = randv((size_t)T * 3 * H * D);
      float* dq = dev(qkv);
      c.codec_frame = f0;
      CUDA_CHECK(cudaMemcpy(dc, &c, sizeof c, cudaMemcpyHostToDevice));
      codec_attn(dq, dc, T, H, D, window, ring, dco, dsi, kr, vr, qt, out, 0);
      auto o = host(out, (size_t)T * H * D);
      auto rope = [&](const float* x, int p, double* y) {
        for (int i = 0; i < 32; ++i) {
          const double a = p / std::pow(1e4, 2.0 * i / D);
          y[i] = x[i] * std::cos(a) - x[i + 32] * std::sin(a);
          y[i + 32] = x[i + 32] * std::cos(a) + x[i] * std::sin(a);
        }
      };
      for (int t = 0; t < T; ++t) {
        const int p = f0 + t;
        Kp[p].assign((size_t)H * D, 0); Vp[p].assign((size_t)H * D, 0);
        for (int h = 0; h < H; ++h) {
          rope(&qkv[(size_t)t * 3 * H * D + H * D + h * D], p, &Kp[p][(size_t)h * D]);
          for (int d = 0; d < D; ++d) Vp[p][(size_t)h * D + d] = qkv[(size_t)t * 3 * H * D + 2 * H * D + h * D + d];
        }
      }
      if (f0 == 68 || f0 == 200) {
        std::vector<double> ref((size_t)T * H * D);
        for (int t = 0; t < T; ++t) {
          const int p = f0 + t;
          for (int h = 0; h < H; ++h) {
            double qv[64];
            rope(&qkv[(size_t)t * 3 * H * D + h * D], p, qv);
            std::vector<double> s;
            std::vector<int> ps;
            for (int j = std::max(0, p - window + 1); j <= p; ++j) {
              double a = 0;
              for (int d = 0; d < D; ++d) a += qv[d] * Kp[j][(size_t)h * D + d];
              s.push_back(a / 8.0); ps.push_back(j);
            }
            double mx = -1e300, l = 0;
            for (double x : s) mx = std::max(mx, x);
            for (auto& x : s) { x = std::exp(x - mx); l += x; }
            for (int d = 0; d < D; ++d) {
              double a = 0;
              for (size_t i = 0; i < s.size(); ++i) a += s[i] * Vp[ps[i]][(size_t)h * D + d];
              ref[((size_t)t * H + h) * D + d] = a / l;
            }
          }
        }
        worst = std::max(worst, rel_err(o, ref));
      }
      cudaFree(dq);
    }
    check("codec attention (ring, window)", worst, 1e-5);
    cudaFree(dco); cudaFree(dsi); cudaFree(kr); cudaFree(vr); cudaFree(qt); cudaFree(out); cudaFree(dc);
  }
  // codec output layer: snake + conv C->1 k7 with state, clamp, int16
  {
    const int T = 300, C = 96;
    auto x = randv((size_t)T * C, 0.3f), stv = randv((size_t)6 * C, 0.3f), ea = randv(C), ib = randv(C, 0.1f), w = randv((size_t)7 * C, 0.05f);
    for (auto& v : ea) v = std::exp(v * 0.3f);
    float* dx = dev(x); float* ds = dev(stv); float* dea = dev(ea); float* dib = dev(ib); float* dw = dev(w);
    int16_t* dp = dev(std::vector<int16_t>(T));
    codec_out(dx, ds, T, C, dea, dib, dw, 0.01f, dp, 0);
    auto pcm = host(dp, T);
    int worst = 0;
    for (int t = 0; t < T; ++t) {
      double a = 0.01;
      for (int j = 0; j < 7; ++j)
        for (int c = 0; c < C; ++c) {
          const int r = t + j;
          a += w[j * C + c] * snake(r < 6 ? stv[r * C + c] : x[(r - 6) * C + c], ea[c], ib[c]);
        }
      a = std::min(1.0, std::max(-1.0, a));
      worst = std::max(worst, std::abs((int)pcm[t] - (int)(a * 32767.0)));
    }
    check("codec out (pcm16 lsb)", worst, 1);
    cudaFree(dx); cudaFree(ds); cudaFree(dea); cudaFree(dib); cudaFree(dw); cudaFree(dp);
  }
  // RVQ
  {
    const int T = 4, Q = 16, S = 64, Dc = 256, O = 512;
    auto cb = randv((size_t)Q * S * Dc), pf = randv((size_t)O * Dc, 0.05f), pr = randv((size_t)O * Dc, 0.05f);
    std::vector<int> codes((size_t)(T + 8) * 16);
    for (auto& v : codes) v = (int)(rng() % S);
    float* dcb = dev(cb); float* dpf = dev(pf); float* dpr = dev(pr); int* dcodes = dev(codes);
    float* dx = dev(std::vector<float>((size_t)T * O));
    Ctl c{};
    c.codec_frame = 3;
    Ctl* dc = dev(std::vector<Ctl>{c});
    codec_rvq(dcodes, dc, T, dcb, S, Dc, dpf, dpr, O, dx, 0);
    auto x = host(dx, (size_t)T * O);
    std::vector<double> ref((size_t)T * O);
    for (int t = 0; t < T; ++t) {
      const int* cc = &codes[(size_t)(3 + t) * 16];
      for (int o = 0; o < O; ++o) {
        double a = 0;
        for (int d = 0; d < Dc; ++d) {
          double s1 = 0;
          for (int q = 1; q < Q; ++q) s1 += cb[((size_t)q * S + cc[q]) * Dc + d];
          a += pf[(size_t)o * Dc + d] * cb[(size_t)cc[0] * Dc + d] + pr[(size_t)o * Dc + d] * s1;
        }
        ref[(size_t)t * O + o] = a;
      }
    }
    check("codec rvq", rel_err(x, ref), 1e-5);
    cudaFree(dcb); cudaFree(dpf); cudaFree(dpr); cudaFree(dcodes); cudaFree(dx); cudaFree(dc);
  }
}

int main() {
  test_gemv();
  test_attention();
  test_sample();
  test_codec_ops();
  printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
