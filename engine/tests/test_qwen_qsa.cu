// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next full-attention / QSA kernels (hive/qwen/qsa.h) against a CPU reference written from the header's math contract
//   (double precision, with the contract's bf16 rounding points). Random weights and inputs — no checkpoint.
//   Sequence A: prefilled in two calls (positions 0..1001, 1002..2599 — group 250 spans both calls), > 512 visible groups at the end so the
//     indexer scores and selects. Sequence B: 7 prefill positions with distinct (t, h, w) rope coordinates (axis mapping), then one decode
//     row of A and one of B in the same call (rows of two sequences).
//   Checks: q / k / iq / group keys (≤ 1 bf16 step from the reference, ≥ 99 % identical), v / raw key copies (exact), selection (dense rows:
//     exactly 0..pos; scored rows: 512 distinct groups whose reference scores reach the reference 512th score within 1e-4 relative, plus the
//     open-group tail), attention output (cosine ≥ 0.9999 and relative L2 ≤ 1e-2 against the reference fed with the GPU's own q/k/v and
//     indices — the softmax and P·V run in fp32 on the GPU; the bf16 output rounding dominates the difference).
// Usage: test_qwen_qsa            (exit status 1 on any FAIL)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include "hive/common.h"
#include "hive/qwen/qsa.h"
#include "test_util.h"

using namespace hive;
using namespace hive::qwen;

namespace {

int g_fail = 0;
#define EXPECT(cond, ...)             \
  do {                                \
    if (!(cond)) {                    \
      printf("  FAIL: " __VA_ARGS__); \
      printf("\n");                   \
      ++g_fail;                       \
    }                                 \
  } while (0)

float b2f(uint16_t v) { return test::bf16_bits_to_f32(v); }
uint16_t f2b(float f) {  // RNE
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
double rbd(double x) { return b2f(f2b((float)x)); }

template <class T> T* dev(size_t n) { T* p = nullptr; CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); return p; }
template <class T> void up(T* d, const std::vector<T>& h) { CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
template <class T> std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

constexpr int D = kQsaDim, DI = kQsaIdxDim, LD = kQsaProjWidth + 32;  // projection rows wider than needed (stride ≠ width)
constexpr float kEps = 1e-6f, kTheta = 1e7f;
const int kSection[3] = {11, 11, 10};

struct Seq {
  int cap = 0;
  bf16 *k = nullptr, *v = nullptr, *ik = nullptr, *gk = nullptr;
  int32_t* rope = nullptr;
  std::vector<int32_t> rope_h;   // [cap, 3]
  std::vector<uint16_t> proj_h;  // [cap, LD] — the projection row each position was written from
  QsaCache cache() const { return QsaCache{k, v, ik, gk, rope}; }
};

// CPU reference -----------------------------------------------------------------------------------------------------------------------
std::vector<double> inv_freq_ref() {
  std::vector<double> f(32);
  for (int i = 0; i < 32; ++i) f[i] = 1.0f / powf(kTheta, (float)(2 * i) / 64.f);  // the same fp32 values the kernels use
  return f;
}
int axis_ref(int i) { return (i % 3 == 1 && i < 3 * kSection[1]) ? 1 : (i % 3 == 2 && i < 3 * kSection[2]) ? 2 : 0; }
// (1 + w) norm (bf16 out) then NeoX rope on dims 0..63 (bf16 out) of one head
std::vector<double> head_ref(const uint16_t* x, int n, const std::vector<uint16_t>& w, const int32_t* coord) {
  static const std::vector<double> inv = inv_freq_ref();
  double ss = 0;
  for (int d = 0; d < n; ++d) { const double v = b2f(x[d]); ss += v * v; }
  const double r = 1.0 / std::sqrt(ss / n + kEps);
  std::vector<double> y(n);
  for (int d = 0; d < n; ++d) y[d] = rbd(b2f(x[d]) * r * (1.0 + b2f(w[d])));
  for (int i = 0; i < 32; ++i) {
    const double ang = (double)(float)((float)coord[axis_ref(i)] * (float)inv[i]);
    const double c = std::cos(ang), s = std::sin(ang), a = y[i], b = y[i + 32];
    y[i] = rbd(a * c - b * s);
    y[i + 32] = rbd(b * c + a * s);
  }
  return y;
}
std::vector<double> group_key_ref(const Seq& S, int G, const std::vector<uint16_t>& w) {
  std::vector<uint16_t> m(DI);
  for (int d = 0; d < DI; ++d) {
    double s = 0;
    for (int j = 0; j < 4; ++j) s += b2f(S.proj_h[(size_t)(4 * G + j) * LD + kQsaIkOff + d]);
    m[d] = f2b((float)(s * 0.25));
  }
  return head_ref(m.data(), DI, w, &S.rope_h[(size_t)(4 * G) * 3]);
}

// counts of bf16-step differences. scale = the largest |reference| of the vector the value belongs to: a value that is tiny against its
//   vector (a rotary pair that nearly cancels) is judged by the absolute error — one bf16 step of the vector's scale — not by its own
//   relative error, which the fp32 kernel cannot meet against a double reference (2026-10-31: 2-7 of 6-10 M q values at rel 1e-2..1e-1,
//   99.999 % identical otherwise; the kernels were unchanged since the test last passed).
struct Cmp { size_t n = 0, same = 0, over1 = 0; double maxrel = 0; };
double vmax(const double* y, int n) { double m = 0; for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(y[i])); return m; }
void cmp_vals(Cmp& c, double got, double want, double scale = 0) {
  ++c.n;
  if (got == want) { ++c.same; return; }
  const double rel = std::fabs(got - want) / std::max(std::fabs(want), 1e-30);
  c.maxrel = std::max(c.maxrel, rel);
  if (rel > 1.0 / 128 + 1e-12 && std::fabs(got - want) > scale / 128) ++c.over1;   // more than one bf16 step of the value (8 mantissa bits incl. the implicit one → ≤ 2^-7 relative) and more than two steps of the vector's scale
}
void report(const char* what, const Cmp& c) {
  printf("  %-10s %zu values · identical %.3f %% · > 1 bf16 step %zu · max rel %.2e\n", what, c.n, 100.0 * c.same / std::max<size_t>(c.n, 1), c.over1, c.maxrel);
  EXPECT(c.over1 == 0 && c.same >= c.n * 99 / 100, "%s: values differ from the reference beyond one bf16 rounding", what);
}

}  // namespace

int main() {
  setenv("HIVE_QWEN_QSA_MR", "1", 1);   // the multi-row attention is opt-in; the test checks it
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  auto rnd_bf16 = [&](size_t n, float scale) { std::vector<uint16_t> v(n); for (auto& x : v) x = f2b(U(rng) * scale); return v; };
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  // weights
  const std::vector<uint16_t> wq = rnd_bf16(D, 0.5f), wk = rnd_bf16(D, 0.5f), wiq = rnd_bf16(DI, 0.5f), wik = rnd_bf16(DI, 0.5f);
  bf16* dwq = (bf16*)dev<uint16_t>(D); bf16* dwk = (bf16*)dev<uint16_t>(D); bf16* dwiq = (bf16*)dev<uint16_t>(DI); bf16* dwik = (bf16*)dev<uint16_t>(DI);
  up((uint16_t*)dwq, wq); up((uint16_t*)dwk, wk); up((uint16_t*)dwiq, wiq); up((uint16_t*)dwik, wik);
  const QsaWeights W{dwq, dwk, dwiq, dwik, kEps};
  float* inv = dev<float>(32);
  int32_t* axis = dev<int32_t>(32);
  qsa_rope_tables(kTheta, kSection, true, inv, axis, st);
  const QsaRope rope{inv, axis};
  // sequences
  Seq A, B;
  for (Seq* S : {&A, &B}) {
    S->cap = 4096;
    S->k = (bf16*)dev<uint16_t>((size_t)S->cap * 2 * D); S->v = (bf16*)dev<uint16_t>((size_t)S->cap * 2 * D);
    S->ik = (bf16*)dev<uint16_t>((size_t)S->cap * DI); S->gk = (bf16*)dev<uint16_t>((size_t)(S->cap / 4 + 1) * DI);
    S->rope = dev<int32_t>((size_t)S->cap * 3);
    S->rope_h.assign((size_t)S->cap * 3, 0);
    S->proj_h.assign((size_t)S->cap * LD, 0);
  }
  for (int p = 0; p < A.cap; ++p) for (int a = 0; a < 3; ++a) A.rope_h[(size_t)p * 3 + a] = p;            // text
  for (int p = 0; p < B.cap; ++p) { B.rope_h[(size_t)p * 3] = p; B.rope_h[(size_t)p * 3 + 1] = p + 3; B.rope_h[(size_t)p * 3 + 2] = 2 * p + 5; }  // image-like
  up(A.rope, A.rope_h); up(B.rope, B.rope_h);
  // device buffers for a call of up to NMAX rows
  const int NMAX = 1600;
  bf16* proj = (bf16*)dev<uint16_t>((size_t)NMAX * LD);
  bf16* q = (bf16*)dev<uint16_t>((size_t)NMAX * kQsaHeads * D);
  bf16* iq = (bf16*)dev<uint16_t>((size_t)NMAX * kQsaIdxHeads * DI);
  int32_t* idx = dev<int32_t>((size_t)NMAX * kQsaSel);
  bf16* out = (bf16*)dev<uint16_t>((size_t)NMAX * kQsaHeads * D);
  QsaRow* rows_d = dev<QsaRow>(NMAX);
  const size_t wsb = qsa_select_ws_bytes(NMAX, 4096);
  void* ws = dev<uint8_t>(wsb);

  // one call: rows (seq, pos) — runs prep, select, attention and checks everything against the reference
  auto run_call = [&](const char* name, const std::vector<std::pair<Seq*, int>>& rs, bool small_ws) {
    printf("%s: %zu rows\n", name, rs.size());
    const int n = (int)rs.size();
    std::vector<uint16_t> ph = rnd_bf16((size_t)n * LD, 1.0f);
    std::vector<QsaRow> rh(n);
    int max_pos = 0;
    for (int r = 0; r < n; ++r) {
      Seq* S = rs[r].first;
      const int p = rs[r].second;
      std::memcpy(&S->proj_h[(size_t)p * LD], &ph[(size_t)r * LD], LD * 2);
      rh[r] = QsaRow{S->cache(), p};
      max_pos = std::max(max_pos, p);
    }
    up((uint16_t*)proj, ph);
    up(rows_d, rh);
    qsa_prep(W, rope, rows_d, n, proj, LD, q, iq, st);
    const size_t wsu = small_ws ? qsa_select_ws_bytes(1, max_pos + 1) : wsb;   // small: forces one-row sub-blocks
    bool one = true;
    for (const auto& x : rs) one = one && x.first == rs[0].first;
    qsa_select(rows_d, n, max_pos, iq, idx, ws, wsu, st, 0, one);
    qsa_attention(rows_d, n, q, proj, LD, idx, out, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> qh = down((const uint16_t*)q, (size_t)n * kQsaHeads * D);
    const std::vector<uint16_t> iqh = down((const uint16_t*)iq, (size_t)n * kQsaIdxHeads * DI);
    const std::vector<int32_t> ih = down(idx, (size_t)n * kQsaSel);
    const std::vector<uint16_t> oh = down((const uint16_t*)out, (size_t)n * kQsaHeads * D);
    // prep
    Cmp cq, ck, ciq, cgk;
    size_t copy_bad = 0;
    for (int r = 0; r < n; ++r) {
      Seq& S = *rs[r].first;
      const int p = rs[r].second;
      const uint16_t* pr = &ph[(size_t)r * LD];
      const int32_t* co = &S.rope_h[(size_t)p * 3];
      for (int h = 0; h < kQsaHeads; ++h) {
        const auto y = head_ref(pr + (size_t)h * 2 * D, D, wq, co);
        { const double sc = vmax(y.data(), D); for (int d = 0; d < D; ++d) cmp_vals(cq, b2f(qh[((size_t)r * kQsaHeads + h) * D + d]), y[d], sc); }
      }
      const std::vector<uint16_t> kc = down((const uint16_t*)S.k + (size_t)p * 2 * D, 2 * D);
      const std::vector<uint16_t> vc = down((const uint16_t*)S.v + (size_t)p * 2 * D, 2 * D);
      const std::vector<uint16_t> ikc = down((const uint16_t*)S.ik + (size_t)p * DI, DI);
      for (int g = 0; g < 2; ++g) {
        const auto y = head_ref(pr + kQsaKOff + (size_t)g * D, D, wk, co);
        { const double sc = vmax(y.data(), D); for (int d = 0; d < D; ++d) cmp_vals(ck, b2f(kc[(size_t)g * D + d]), y[d], sc); }
      }
      for (int i = 0; i < 2 * D; ++i) copy_bad += vc[i] != pr[kQsaVOff + i];
      for (int i = 0; i < DI; ++i) copy_bad += ikc[i] != pr[kQsaIkOff + i];
      for (int h = 0; h < kQsaIdxHeads; ++h) {
        const auto y = head_ref(pr + kQsaIqOff + (size_t)h * DI, DI, wiq, co);
        { const double sc = vmax(y.data(), DI); for (int d = 0; d < DI; ++d) cmp_vals(ciq, b2f(iqh[((size_t)r * kQsaIdxHeads + h) * DI + d]), y[d], sc); }
      }
      if ((p & 3) == 3) {
        const int G = p >> 2;
        const auto y = group_key_ref(S, G, wik);
        const std::vector<uint16_t> g = down((const uint16_t*)S.gk + (size_t)G * DI, DI);
        { const double sc = vmax(y.data(), DI); for (int d = 0; d < DI; ++d) cmp_vals(cgk, b2f(g[d]), y[d], sc); }
      }
    }
    report("q", cq); report("k", ck); report("iq", ciq);
    if (cgk.n) report("group key", cgk);
    EXPECT(copy_bad == 0, "v / raw indexer key copies differ in %zu values", copy_bad);
    // select
    size_t sel_bad = 0, scored = 0;
    for (int r = 0; r < n && sel_bad < 8; ++r) {
      Seq& S = *rs[r].first;
      const int p = rs[r].second, V = (p + 1) / 4;
      const int32_t* ir = &ih[(size_t)r * kQsaSel];
      std::multiset<int> got;
      for (int j = 0; j < kQsaSel; ++j) if (ir[j] >= 0) got.insert(ir[j]);
      if (V <= kQsaTopBlocks) {
        std::multiset<int> want;
        for (int j = 0; j <= p; ++j) want.insert(j);
        if (got != want) { ++sel_bad; printf("  row %d (pos %d): dense selection differs\n", r, p); }
        continue;
      }
      ++scored;
      // reference scores from the GPU's own iq and group keys (isolates the selection from the prep's rounding)
      const std::vector<uint16_t> gk = down((const uint16_t*)S.gk, (size_t)V * DI);
      std::vector<double> sc(V);
      for (int G = 0; G < V; ++G) {
        double t = 0;
        for (int h = 0; h < kQsaIdxHeads; ++h) {
          double s = 0;
          for (int d = 0; d < DI; ++d) s += (double)b2f(iqh[((size_t)r * kQsaIdxHeads + h) * DI + d]) * b2f(gk[(size_t)G * DI + d]);
          t += std::max(s, 0.0);
        }
        sc[G] = t / std::sqrt(128.0);
      }
      std::vector<double> sorted = sc;
      std::nth_element(sorted.begin(), sorted.begin() + (kQsaTopBlocks - 1), sorted.end(), std::greater<double>());
      const double kth = sorted[kQsaTopBlocks - 1];
      std::set<int> groups;
      bool ok = true;
      for (int x : got) {
        if (x >= 4 * V) { if (x > p) ok = false; continue; }   // open-group tail
        groups.insert(x / 4);
      }
      for (int G : groups) if (sc[G] < kth - 1e-4 * std::fabs(kth)) ok = false;
      ok = ok && groups.size() == (size_t)kQsaTopBlocks && got.size() == (size_t)(kQsaTopBlocks * 4 + (p + 1 - 4 * V));
      for (int G : groups) ok = ok && got.count(4 * G) == 1 && got.count(4 * G + 3) == 1;
      for (int t = 4 * V; t <= p; ++t) ok = ok && got.count(t) == 1;
      if (!ok) { ++sel_bad; printf("  row %d (pos %d): scored selection wrong (groups %zu, positions %zu)\n", r, p, groups.size(), got.size()); }
    }
    printf("  select: %d rows (%zu scored) · bad %zu\n", n, scored, sel_bad);
    EXPECT(sel_bad == 0, "selection");
    // attention: reference from the GPU's q, the cache's k / v and the GPU's indices (sampled rows to keep the test quick)
    std::vector<float> gotv, wantv;
    for (int r = 0; r < n; r += std::max(1, n / 24)) {
      Seq& S = *rs[r].first;
      const int32_t* ir = &ih[(size_t)r * kQsaSel];
      std::vector<int> sel;
      for (int j = 0; j < kQsaSel; ++j) if (ir[j] >= 0) sel.push_back(ir[j]);
      std::vector<std::vector<uint16_t>> kk, vv;
      for (int pos : sel) {
        kk.push_back(down((const uint16_t*)S.k + (size_t)pos * 2 * D, 2 * D));
        vv.push_back(down((const uint16_t*)S.v + (size_t)pos * 2 * D, 2 * D));
      }
      for (int h = 0; h < kQsaHeads; ++h) {
        const int g = h / 12;
        std::vector<double> s(sel.size());
        double mx = -1e300;
        for (size_t j = 0; j < sel.size(); ++j) {
          double t = 0;
          for (int d = 0; d < D; ++d) t += (double)b2f(qh[((size_t)r * kQsaHeads + h) * D + d]) * b2f(kk[j][(size_t)g * D + d]);
          s[j] = t / 16.0;
          mx = std::max(mx, s[j]);
        }
        double l = 0;
        for (double& x : s) { x = std::exp(x - mx); l += x; }
        for (int d = 0; d < D; ++d) {
          double a = 0;
          for (size_t j = 0; j < sel.size(); ++j) a += s[j] * b2f(vv[j][(size_t)g * D + d]);
          a = sel.empty() ? 0.0 : rbd(a / l);
          const double gt = b2f(ph[(size_t)r * LD + (size_t)h * 2 * D + D + d]);
          wantv.push_back((float)(a / (1.0 + std::exp(-gt))));
          gotv.push_back(b2f(oh[((size_t)r * kQsaHeads + h) * D + d]));
        }
      }
    }
    const test::Closeness c = test::closeness(gotv, wantv);
    printf("  attention: %zu values · cos %.6f · rel %.2e · max abs %.2e\n", gotv.size(), c.cos, c.rel, c.max_abs);
    EXPECT(c.finite && c.cos >= 0.9999 && c.rel <= 1e-2, "attention output");
    if (one && n > 64) {   // several rows per block (prefill chunks of one sequence) against the per-row kernel's output
      const size_t mb = qsa_attention_mr_ws_bytes(n);
      void* mws = nullptr;
      bf16* om = nullptr;
      CUDA_CHECK(cudaMalloc(&mws, mb)); CUDA_CHECK(cudaMalloc(&om, (size_t)n * kQsaHeads * D * 2));
      const bool ran = qsa_attention_mr(rows_d, n, max_pos, q, proj, LD, idx, om, mws, mb, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      EXPECT(ran, "qsa_attention_mr did not run");
      const std::vector<uint16_t> sm = down((const uint16_t*)om, (size_t)n * kQsaHeads * D);
      std::vector<float> a, b;
      for (size_t i = 0; i < sm.size(); ++i) { a.push_back(b2f(sm[i])); b.push_back(b2f(oh[i])); }
      const test::Closeness cm = test::closeness(a, b);
      size_t same = 0;
      for (size_t i = 0; i < sm.size(); ++i) same += sm[i] == oh[i];
      printf("  multi-row attention (%d rows per block) vs the per-row kernel: cos %.7f · rel %.2e · identical %.1f %%\n", 4, cm.cos, cm.rel,
             100.0 * same / sm.size());
      EXPECT(cm.finite && cm.cos >= 0.99999 && cm.rel <= 5e-3, "multi-row attention output");
      std::vector<float> gm;   // and against the reference on the sampled rows
      for (int r = 0; r < n; r += std::max(1, n / 24))
        for (int h = 0; h < kQsaHeads; ++h)
          for (int d = 0; d < D; ++d) gm.push_back(b2f(sm[((size_t)r * kQsaHeads + h) * D + d]));
      const test::Closeness cr = test::closeness(gm, wantv);
      printf("  multi-row attention vs reference: cos %.6f · rel %.2e\n", cr.cos, cr.rel);
      EXPECT(cr.finite && cr.cos >= 0.9999 && cr.rel <= 1e-2, "multi-row attention vs reference");
      cudaFree(mws); cudaFree(om);
    }
    {   // split-KV attention (decode / verify rows) on the last ≤ 8 rows of the call, against the reference attention of those rows
      const int m8 = std::min(n, 8), off = n - m8;
      static void* sws = nullptr;
      if (!sws) CUDA_CHECK(cudaMalloc(&sws, qsa_attention_split_ws_bytes(8, 16)));
      bf16* o8 = nullptr;
      CUDA_CHECK(cudaMalloc(&o8, (size_t)m8 * kQsaHeads * D * 2));
      qsa_attention_split(rows_d + off, m8, q + (size_t)off * kQsaHeads * D, proj + (size_t)off * LD, LD, idx + (size_t)off * kQsaSel, o8, 16, sws, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      const std::vector<uint16_t> s8 = down((const uint16_t*)o8, (size_t)m8 * kQsaHeads * D);
      std::vector<float> a8, b8;
      for (size_t i = 0; i < s8.size(); ++i) { a8.push_back(b2f(s8[i])); b8.push_back(b2f(oh[(size_t)off * kQsaHeads * D + i])); }
      const test::Closeness c8 = test::closeness(a8, b8);
      printf("  split-KV attention (16 splits, last %d rows) vs the full-row kernel: cos %.6f · rel %.2e\n", m8, c8.cos, c8.rel);
      EXPECT(c8.finite && c8.cos >= 0.9999 && c8.rel <= 1e-2, "split attention output");
      cudaFree(o8);
      // decode scorer (≤ 8 rows of one sequence): the same rows selected again in one small call — position sets vs the big call's
      if (one) {
        int32_t* i8 = nullptr;
        CUDA_CHECK(cudaMalloc(&i8, (size_t)m8 * kQsaSel * 4));
        qsa_select(rows_d + off, m8, max_pos, iq + (size_t)off * kQsaIdxHeads * DI, i8, ws, wsb, st, 0, true);
        CUDA_CHECK(cudaStreamSynchronize(st));
        const std::vector<int32_t> h8 = down(i8, (size_t)m8 * kQsaSel);
        size_t same = 0, tot = 0;
        for (int r = 0; r < m8; ++r) {
          std::set<int> a, b;
          for (int j = 0; j < kQsaSel; ++j) { if (h8[(size_t)r * kQsaSel + j] >= 0) a.insert(h8[(size_t)r * kQsaSel + j]); if (ih[(size_t)(off + r) * kQsaSel + j] >= 0) b.insert(ih[(size_t)(off + r) * kQsaSel + j]); }
          for (int x : b) same += a.count(x);
          tot += b.size();
        }
        printf("  decode-size selection (last %d rows, one call): %.4f of the positions agree\n", m8, tot ? (double)same / tot : 1.0);
        EXPECT(tot == 0 || (double)same / tot >= 0.995, "decode scorer selection");
        cudaFree(i8);
      }
    }
  };

  // sequence A in two prefill calls (group 250 = positions 1000..1003 spans both), B in one
  std::vector<std::pair<Seq*, int>> rs;
  for (int p = 0; p < 1002; ++p) rs.push_back({&A, p});
  run_call("A prefill 0..1001", rs, false);
  rs.clear();
  for (int p = 1002; p < 2600; p += 1) if (rs.size() < (size_t)NMAX) rs.push_back({&A, p});
  run_call("A prefill 1002..2599", rs, false);
  rs.clear();
  for (int p = 0; p < 7; ++p) rs.push_back({&B, p});
  run_call("B prefill 0..6 (image-like rope)", rs, true);
  // decode: one row of A (scored) and one of B (dense) in the same call; one-row selection sub-blocks
  rs = {{&A, 2600}, {&B, 7}};
  run_call("decode A@2600 + B@7", rs, true);
  // verify-style: 4 consecutive rows of A completing a group in the middle
  rs = {{&A, 2601}, {&A, 2602}, {&A, 2603}, {&A, 2604}};
  run_call("verify rows A@2601..2604", rs, false);

  printf(g_fail ? "FAIL (%d)\n" : "PASS\n", g_fail);
  return g_fail ? 1 : 0;
}
