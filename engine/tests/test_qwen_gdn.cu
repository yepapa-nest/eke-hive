// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next Gated DeltaNet kernels (src/qwen/gdn.cu) against a host fp64 reference written from the hive/qwen/gdn.h contract,
//   and against themselves. Synthetic inputs only (no checkpoint, no model).
//   1. prefill T = 300 from zero state vs the host reference: outputs, final recurrent state, conv state
//   2. state continuity: T = 2500 (crosses the internal 1024-token chunk) in one call vs calls of 700 + 1 + 1799 — bit-identical
//   3. batched decode M = 4 (different sequences, after a prefill) vs a T = 1 prefill of each sequence — bit-identical
//   4. verify rows R = 4 vs 4 single-row decodes — bit-identical outputs and states; snapshots = the state / conv window after each row
// Run: test_qwen_gdn  (exit status 1 on any failure)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/qwen/gdn.h"
#include "test_util.h"

using namespace hive;
using namespace hive::qwen;

namespace {

constexpr int HK = kGdnKHeads, HV = kGdnVHeads, D = kGdnDim, C = kGdnConvCh, LD = kGdnProjWidth;
constexpr float kEps = 1e-6f;
int g_fail = 0;

void verdict(const char* name, bool ok, const char* detail) {
  printf("%s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
  if (!ok) ++g_fail;
}

template <class T>
T* dev_upload(const std::vector<T>& h) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
  if (!h.empty()) CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T>
std::vector<T> dev_download(const T* d, size_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0; }
std::vector<float> to_f32(const std::vector<bf16>& v) { std::vector<float> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = bf2f(v[i]); return o; }
double rb(double x) { return (double)bf2f(f2bf((float)x)); }
int key_ch(int kh, int i) { return kGdnKWidth + kh * kGdnDim + i; }  // k channels follow the 2048 q channels

struct Weights {
  std::vector<float> conv_w, A_log, dt_bias;
  std::vector<bf16> norm_w;
  float* d_conv = nullptr; float* d_A = nullptr; float* d_dt = nullptr; bf16* d_norm = nullptr;
  GdnParams params() const { return GdnParams{d_conv, d_A, d_dt, d_norm, kEps}; }
};

Weights make_weights(std::mt19937& rng) {
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::normal_distribution<float> n(0.f, 1.f);
  Weights w;
  w.conv_w.resize((size_t)C * 4); for (auto& x : w.conv_w) x = 0.5f * u(rng);
  w.A_log.resize(HV); for (auto& x : w.A_log) x = u(rng);
  w.dt_bias.resize(HV); for (auto& x : w.dt_bias) x = u(rng);
  w.norm_w.resize(D); for (auto& x : w.norm_w) x = f2bf(1.f + 0.1f * n(rng));
  w.d_conv = dev_upload(w.conv_w); w.d_A = dev_upload(w.A_log); w.d_dt = dev_upload(w.dt_bias); w.d_norm = dev_upload(w.norm_w);
  return w;
}
std::vector<bf16> make_proj(std::mt19937& rng, int T) {
  std::normal_distribution<float> n(0.f, 1.f);
  std::vector<bf16> p((size_t)T * LD);
  for (auto& x : p) x = f2bf(n(rng));
  return p;
}

// Host reference (fp64 arithmetic, the contract's bf16 roundings): T tokens of one sequence, state / conv updated in place.
void reference(const Weights& w, const std::vector<bf16>& proj, int T, std::vector<double>& conv, std::vector<double>& S, std::vector<double>& out) {
  out.assign((size_t)T * kGdnVWidth, 0.0);
  std::vector<double> win((size_t)(T + 3) * C);
  for (int j = 0; j < 3; ++j) for (int c = 0; c < C; ++c) win[(size_t)j * C + c] = conv[(size_t)j * C + c];
  for (int t = 0; t < T; ++t) for (int c = 0; c < C; ++c) win[(size_t)(t + 3) * C + c] = bf2f(proj[(size_t)t * LD + c]);
  std::vector<double> x(C), qn(kGdnKWidth), kn(kGdnKWidth), o(D);
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < C; ++c) {
      double y = 0;
      for (int j = 0; j < 4; ++j) y += (double)w.conv_w[(size_t)c * 4 + j] * win[(size_t)(t + j) * C + c];
      x[c] = rb(y / (1.0 + std::exp(-y)));
    }
    for (int kh = 0; kh < HK; ++kh) {
      double sq = 0, sk = 0;
      for (int i = 0; i < D; ++i) { sq += x[kh * D + i] * x[kh * D + i]; sk += x[key_ch(kh, i)] * x[key_ch(kh, i)]; }
      for (int i = 0; i < D; ++i) {
        qn[kh * D + i] = x[kh * D + i] / std::sqrt(sq + 1e-6) / std::sqrt(128.0);
        kn[kh * D + i] = x[key_ch(kh, i)] / std::sqrt(sk + 1e-6);
      }
    }
    const bf16* row = proj.data() + (size_t)t * LD;
    for (int h = 0; h < HV; ++h) {
      const int kh = h / (HV / HK);
      const double b = bf2f(row[kGdnBOff + h]), a = bf2f(row[kGdnAOff + h]);
      const double sp = a + w.dt_bias[h] > 20 ? a + w.dt_bias[h] : std::log1p(std::exp(a + w.dt_bias[h]));
      const double decay = std::exp(-std::exp((double)w.A_log[h]) * sp);
      const double beta = rb(1.0 / (1.0 + std::exp(-b)));
      double* Sh = S.data() + (size_t)h * D * D;
      for (int v = 0; v < D; ++v) {
        double kv = 0;
        for (int k = 0; k < D; ++k) { Sh[(size_t)k * D + v] *= decay; kv += Sh[(size_t)k * D + v] * kn[kh * D + k]; }
        const double delta = (x[2 * kGdnKWidth + h * D + v] - kv) * beta;
        double ov = 0;
        for (int k = 0; k < D; ++k) { Sh[(size_t)k * D + v] += kn[kh * D + k] * delta; ov += Sh[(size_t)k * D + v] * qn[kh * D + k]; }
        o[v] = rb(ov);
      }
      double ss = 0;
      for (int v = 0; v < D; ++v) ss += o[v] * o[v];
      const double r = 1.0 / std::sqrt(ss / D + kEps);
      for (int v = 0; v < D; ++v) {
        const double z = bf2f(row[kGdnZOff + h * D + v]);
        out[(size_t)t * kGdnVWidth + h * D + v] = rb((double)bf2f(w.norm_w[v]) * (o[v] * r) / (1.0 + std::exp(-z)));
      }
    }
  }
  for (int j = 0; j < 3; ++j) for (int c = 0; c < C; ++c) conv[(size_t)j * C + c] = win[(size_t)(T + j) * C + c];
}

struct DevSeq {
  float* conv = nullptr; float* state = nullptr;
  DevSeq() {
    CUDA_CHECK(cudaMalloc(&conv, kGdnConvStateFloats * 4)); CUDA_CHECK(cudaMemset(conv, 0, kGdnConvStateFloats * 4));
    CUDA_CHECK(cudaMalloc(&state, kGdnStateFloats * 4)); CUDA_CHECK(cudaMemset(state, 0, kGdnStateFloats * 4));
  }
  void copy_from(const DevSeq& o) {
    CUDA_CHECK(cudaMemcpy(conv, o.conv, kGdnConvStateFloats * 4, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state, o.state, kGdnStateFloats * 4, cudaMemcpyDeviceToDevice));
  }
};
float* const* dev_ptrs(const std::vector<float*>& v) { return dev_upload(v); }

void prefill(const Weights& w, const bf16* d_proj, int T, DevSeq& s, bf16* d_out) {
  const size_t wsb = gdn_workspace_bytes(T);
  void* ws = nullptr;
  CUDA_CHECK(cudaMalloc(&ws, std::max<size_t>(wsb, 256)));
  gdn_forward_seq(w.params(), d_proj, LD, T, s.conv, s.state, d_out, ws, wsb, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(ws));
}

}  // namespace

int main() {
  std::mt19937 rng(20261009);
  const Weights w = make_weights(rng);
  // 1. prefill vs host reference. Tolerances: the GPU recurrence is fp32 (state and reductions) against fp64 here; the outputs are
  //    rounded to bf16 at two points (o and the gated norm), so a ±1 bf16 step (2^-8 relative) on a minority of elements is expected.
  //    rel-L2 < 4e-3 bounds that; the fp32 state must agree to ~1e-5 relative (300 steps of fp32 rounding).
  {
    const int T = 300;
    const std::vector<bf16> proj = make_proj(rng, T);
    bf16* d_proj = dev_upload(proj);
    bf16* d_out = nullptr; CUDA_CHECK(cudaMalloc(&d_out, (size_t)T * kGdnVWidth * 2));
    DevSeq s;
    prefill(w, d_proj, T, s, d_out);
    std::vector<double> conv(kGdnConvStateFloats, 0.0), S(kGdnStateFloats, 0.0), out;
    reference(w, proj, T, conv, S, out);
    const std::vector<float> got = to_f32(dev_download(d_out, (size_t)T * kGdnVWidth));
    std::vector<float> want(out.begin(), out.end());
    const test::Closeness co = test::closeness(got, want);
    char buf[256];
    snprintf(buf, sizeof buf, "rel %.2e cos %.7f max_abs %.3e", co.rel, co.cos, co.max_abs);
    verdict("prefill output vs reference", co.finite && co.rel < 4e-3 && co.cos > 0.99999, buf);
    const std::vector<float> gs = dev_download(s.state, kGdnStateFloats);
    std::vector<float> ws(S.begin(), S.end());
    const test::Closeness cs = test::closeness(gs, ws);
    snprintf(buf, sizeof buf, "rel %.2e max_abs %.3e", cs.rel, cs.max_abs);
    verdict("prefill state vs reference", cs.finite && cs.rel < 1e-5, buf);
    const std::vector<float> gc = dev_download(s.conv, kGdnConvStateFloats);
    std::vector<float> wc(conv.begin(), conv.end());
    verdict("prefill conv state vs reference", same_bytes(gc, wc), "(the last 3 projection rows, exact)");
  }
  // 2. continuity: one call vs three calls (across the internal chunk boundary)
  {
    const int T = 2500, parts[3] = {700, 1, 1799};
    const std::vector<bf16> proj = make_proj(rng, T);
    bf16* d_proj = dev_upload(proj);
    bf16 *o1 = nullptr, *o2 = nullptr;
    CUDA_CHECK(cudaMalloc(&o1, (size_t)T * kGdnVWidth * 2)); CUDA_CHECK(cudaMalloc(&o2, (size_t)T * kGdnVWidth * 2));
    DevSeq a, b;
    prefill(w, d_proj, T, a, o1);
    int off = 0;
    for (int p : parts) { prefill(w, d_proj + (size_t)off * LD, p, b, o2 + (size_t)off * kGdnVWidth); off += p; }
    const bool ok = same_bytes(dev_download(o1, (size_t)T * kGdnVWidth), dev_download(o2, (size_t)T * kGdnVWidth)) &&
                    same_bytes(dev_download(a.state, kGdnStateFloats), dev_download(b.state, kGdnStateFloats)) &&
                    same_bytes(dev_download(a.conv, kGdnConvStateFloats), dev_download(b.conv, kGdnConvStateFloats));
    verdict("prefill split 700+1+1799 == one call", ok, "(outputs, state, conv state bit-identical)");
  }
  // 3. batched decode vs T = 1 prefill per sequence
  {
    const int M = 4;
    std::vector<DevSeq> sa(M), sb(M);
    for (int m = 0; m < M; ++m) {  // different histories
      const int T = 17 + 13 * m;
      const std::vector<bf16> proj = make_proj(rng, T);
      bf16* d_proj = dev_upload(proj);
      bf16* d_out = nullptr; CUDA_CHECK(cudaMalloc(&d_out, (size_t)T * kGdnVWidth * 2));
      prefill(w, d_proj, T, sa[m], d_out);
      sb[m].copy_from(sa[m]);
    }
    const std::vector<bf16> proj = make_proj(rng, M);
    bf16* d_proj = dev_upload(proj);
    bf16 *oa = nullptr, *ob = nullptr;
    CUDA_CHECK(cudaMalloc(&oa, (size_t)M * kGdnVWidth * 2)); CUDA_CHECK(cudaMalloc(&ob, (size_t)M * kGdnVWidth * 2));
    std::vector<float*> cp, sp;
    for (int m = 0; m < M; ++m) { cp.push_back(sa[m].conv); sp.push_back(sa[m].state); }
    gdn_decode_batch(w.params(), d_proj, LD, M, dev_ptrs(cp), dev_ptrs(sp), oa, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int m = 0; m < M; ++m) prefill(w, d_proj + (size_t)m * LD, 1, sb[m], ob + (size_t)m * kGdnVWidth);
    bool ok = same_bytes(dev_download(oa, (size_t)M * kGdnVWidth), dev_download(ob, (size_t)M * kGdnVWidth));
    for (int m = 0; m < M; ++m)
      ok = ok && same_bytes(dev_download(sa[m].state, kGdnStateFloats), dev_download(sb[m].state, kGdnStateFloats)) &&
           same_bytes(dev_download(sa[m].conv, kGdnConvStateFloats), dev_download(sb[m].conv, kGdnConvStateFloats));
    verdict("decode batch M=4 == T=1 prefill per sequence", ok, "(outputs, states, conv states bit-identical)");
  }
  // 4. verify rows vs sequential single-row decodes, with snapshots
  {
    const int R = 4;
    DevSeq base;
    {
      const int T = 33;
      const std::vector<bf16> proj = make_proj(rng, T);
      bf16* d_out = nullptr; CUDA_CHECK(cudaMalloc(&d_out, (size_t)T * kGdnVWidth * 2));
      prefill(w, dev_upload(proj), T, base, d_out);
    }
    DevSeq a, b;
    a.copy_from(base); b.copy_from(base);
    const std::vector<bf16> proj = make_proj(rng, R);
    bf16* d_proj = dev_upload(proj);
    std::vector<DevSeq> snaps(R - 1);
    std::vector<float*> sc, ss;
    for (auto& x : snaps) { sc.push_back(x.conv); ss.push_back(x.state); }
    bf16 *oa = nullptr, *ob = nullptr;
    CUDA_CHECK(cudaMalloc(&oa, (size_t)R * kGdnVWidth * 2)); CUDA_CHECK(cudaMalloc(&ob, (size_t)R * kGdnVWidth * 2));
    gdn_decode_rows(w.params(), d_proj, LD, R, dev_ptrs({a.conv}), dev_ptrs({a.state}), dev_ptrs(sc), dev_ptrs(ss), oa, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* const* bc = dev_ptrs({b.conv});
    float* const* bs = dev_ptrs({b.state});
    bool snap_ok = true;
    for (int r = 0; r < R; ++r) {
      gdn_decode_batch(w.params(), d_proj + (size_t)r * LD, LD, 1, bc, bs, ob + (size_t)r * kGdnVWidth, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      if (r + 1 < R)
        snap_ok = snap_ok && same_bytes(dev_download(snaps[r].state, kGdnStateFloats), dev_download(b.state, kGdnStateFloats)) &&
                  same_bytes(dev_download(snaps[r].conv, kGdnConvStateFloats), dev_download(b.conv, kGdnConvStateFloats));
    }
    const bool ok = same_bytes(dev_download(oa, (size_t)R * kGdnVWidth), dev_download(ob, (size_t)R * kGdnVWidth)) &&
                    same_bytes(dev_download(a.state, kGdnStateFloats), dev_download(b.state, kGdnStateFloats)) &&
                    same_bytes(dev_download(a.conv, kGdnConvStateFloats), dev_download(b.conv, kGdnConvStateFloats));
    verdict("verify rows R=4 == 4 single-row decodes", ok, "(outputs, final state, conv state bit-identical)");
    verdict("verify snapshots == per-row states", snap_ok, "(state and conv window after rows 0..2)");
  }
  // 5. gdn_decode_rows_multi (3 sequences, rows 5 / 1 / 3, snapshots and a snap_init part) == one gdn_decode_rows per part
  {
    const int P = 3, Rs[P] = {5, 1, 3}, init[P] = {0, 0, 1};
    std::vector<DevSeq> a(P), b(P), sa(P * 8), sb(P * 8), c0a(P), c0b(P);
    std::vector<bf16*> pj(P), oa(P), ob(P);
    std::vector<GdnRowsPart> parts(P);
    std::vector<float* const*> keep;
    for (int p = 0; p < P; ++p) {
      const int T = 21 + 9 * p;
      const std::vector<bf16> hist = make_proj(rng, T);
      bf16* d_out = nullptr; CUDA_CHECK(cudaMalloc(&d_out, (size_t)T * kGdnVWidth * 2));
      prefill(w, dev_upload(hist), T, a[p], d_out);
      b[p].copy_from(a[p]);
      pj[p] = dev_upload(make_proj(rng, Rs[p]));
      CUDA_CHECK(cudaMalloc(&oa[p], (size_t)Rs[p] * kGdnVWidth * 2)); CUDA_CHECK(cudaMalloc(&ob[p], (size_t)Rs[p] * kGdnVWidth * 2));
    }
    auto ptrs = [&](std::vector<DevSeq>& v, int p, bool conv) {
      std::vector<float*> x;
      for (int r = 0; r < 8; ++r) x.push_back(conv ? v[p * 8 + r].conv : v[p * 8 + r].state);
      return dev_ptrs(x);
    };
    for (int p = 0; p < P; ++p) {
      parts[p] = GdnRowsPart{pj[p], Rs[p], dev_ptrs({a[p].conv}), dev_ptrs({a[p].state}), ptrs(sa, p, true), ptrs(sa, p, false), oa[p], init[p],
                             init[p] ? dev_ptrs({c0a[p].conv}) : nullptr};
      gdn_decode_rows(w.params(), pj[p], LD, Rs[p], dev_ptrs({b[p].conv}), dev_ptrs({b[p].state}), ptrs(sb, p, true), ptrs(sb, p, false), ob[p], 0,
                      init[p], init[p] ? dev_ptrs({c0b[p].conv}) : nullptr);
    }
    gdn_decode_rows_multi(w.params(), LD, P, parts.data(), 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = true;
    for (int p = 0; p < P; ++p) {
      ok = ok && same_bytes(dev_download(oa[p], (size_t)Rs[p] * kGdnVWidth), dev_download(ob[p], (size_t)Rs[p] * kGdnVWidth)) &&
           same_bytes(dev_download(a[p].state, kGdnStateFloats), dev_download(b[p].state, kGdnStateFloats)) &&
           same_bytes(dev_download(a[p].conv, kGdnConvStateFloats), dev_download(b[p].conv, kGdnConvStateFloats));
      for (int r = 0; r < 8; ++r)
        ok = ok && same_bytes(dev_download(sa[p * 8 + r].state, kGdnStateFloats), dev_download(sb[p * 8 + r].state, kGdnStateFloats)) &&
             same_bytes(dev_download(sa[p * 8 + r].conv, kGdnConvStateFloats), dev_download(sb[p * 8 + r].conv, kGdnConvStateFloats));
      if (init[p]) ok = ok && same_bytes(dev_download(c0a[p].conv, kGdnConvStateFloats), dev_download(c0b[p].conv, kGdnConvStateFloats));
    }
    verdict("verify rows multi (3 parts) == per-part calls", ok, "(outputs, states, conv, snapshots bit-identical)");
  }
  printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASS", g_fail, g_fail == 1 ? "" : "s");
  return g_fail ? 1 : 0;
}
