// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash KDA kernels (src/glm/kda.cu) vs the real HF module and against themselves.
//   Reference files (bf16 / fp32 raw dumps) come from gen_ref.py (transformers 5.18 Glm5NextTextLinearAttention on CPU,
//   bf16 with conv1d / dt_bias / A_log kept fp32); default directory /tmp/kda_ref.
//   1. prefill T tokens from zero state vs HF output before o_proj (bf16) and HF final recurrent state
//   2. state continuity: prefill split into chunks and token-by-token decode vs one-shot (outputs + final states),
//      also across the internal 1024-token chunk boundary (T = 2500, synthetic inputs)
//   3. batched decode M = 4 different sequences vs each sequence decoded alone
//   4. negative controls: a host fp32 reference with the decay applied after the update must fail vs HF (the correct
//      host reference must pass); a kernel build with -DKDA_NEGCTRL=n must fail test 1 (the binary inverts its verdict)
//   5. timings: prefill T = 1·64·2048·16384, decode M = 1·4·8 (hot and with L2 flushed)
// Run: test_glm_kda [ref_dir=/tmp/kda_ref] [skip_bench=0]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <string>
#include <vector>

#include "hive/common.h"
#include "hive/glm/kda.h"

#ifndef KDA_NEGCTRL
#define KDA_NEGCTRL 0
#endif

using namespace hive;
using namespace hive::glm;

namespace {

constexpr int H = kKdaHeads, D = kKdaHeadDim, W = kKdaWidth, C = kKdaConvChannels;
int g_fail = 0;

template <class T>
std::vector<T> load_bin(const std::string& path, size_t n) {
  std::vector<T> v(n);
  FILE* f = fopen(path.c_str(), "rb");
  HIVE_CHECK(f, "cannot open " + path);
  const size_t got = fread(v.data(), sizeof(T), n, f);
  fclose(f);
  HIVE_CHECK(got == n, "short read " + path);
  return v;
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

struct Cmp { double max_abs = 0, ref_absmax = 0, rel_l2 = 0; size_t n_diff = 0, n = 0; };
template <class A, class B>
Cmp compare(const A* a, const B* b, size_t n) {
  auto tf = [](auto x) -> double {
    if constexpr (std::is_same_v<std::decay_t<decltype(x)>, bf16>) return (double)bf2f(x);
    else return (double)x;
  };
  Cmp c; c.n = n;
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = tf(a[i]), y = tf(b[i]), d = x - y;
    c.max_abs = std::max(c.max_abs, std::fabs(d));
    c.ref_absmax = std::max(c.ref_absmax, std::fabs(y));
    num += d * d; den += y * y;
    if (d != 0) ++c.n_diff;
  }
  c.rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  return c;
}
void print_cmp(const char* tag, const Cmp& c) {
  printf("  %-46s max_abs %.3e (ref absmax %.3e, max_abs/absmax %.3e)  rel_l2 %.3e  differing %zu/%zu\n", tag, c.max_abs,
         c.ref_absmax, c.ref_absmax > 0 ? c.max_abs / c.ref_absmax : 0.0, c.rel_l2, c.n_diff, c.n);
}
void check(bool ok, const char* what) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_fail;
}

// ---- host fp32 reference (HF semantics incl. bf16 roundings); mode 2 = decay after the update -----------------------
float rbf_h(float x) { return bf2f(f2bf(x)); }
float sigm_h(float x) { return 1.0f / (1.0f + std::exp(-x)); }
struct HostParams { std::vector<float> conv_w, A_log, dt_bias; std::vector<bf16> o_norm; };

void host_ref(const HostParams& P, const bf16* qkv, const bf16* gp, const bf16* bl, const bf16* gate, int T,
              std::vector<float>& cs, std::vector<float>& S, std::vector<float>& out, int mode) {
  out.assign((size_t)T * W, 0.f);
  std::vector<float> y(C), hist((size_t)(T + 3) * C);
  for (int j = 0; j < 3; ++j) for (int c = 0; c < C; ++c) hist[(size_t)j * C + c] = cs[(size_t)j * C + c];
  for (int t = 0; t < T; ++t) for (int c = 0; c < C; ++c) hist[(size_t)(t + 3) * C + c] = bf2f(qkv[(size_t)t * C + c]);
  std::vector<float> q(D), k(D), eg(D), o(D), kv(D);
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < C; ++c) {
      float acc = 0.f;
      for (int j = 0; j < 4; ++j) acc += P.conv_w[(size_t)c * 4 + j] * hist[(size_t)(t + j) * C + c];
      y[c] = rbf_h(acc * sigm_h(acc));
    }
    for (int h = 0; h < H; ++h) {
      float sq = 0, sk = 0;
      for (int i = 0; i < D; ++i) { sq += y[h * D + i] * y[h * D + i]; sk += y[W + h * D + i] * y[W + h * D + i]; }
      const float nq = std::sqrt(sq + 1e-6f), nk = std::sqrt(sk + 1e-6f);
      for (int i = 0; i < D; ++i) {
        q[i] = y[h * D + i] / nq * 0.08838834764831845f;
        k[i] = y[W + h * D + i] / nk;
        const float g = -5.0f * sigm_h(std::exp(P.A_log[h]) * (bf2f(gp[(size_t)t * W + h * D + i]) + P.dt_bias[h * D + i]));
        eg[i] = std::exp(g);
      }
      const float beta = rbf_h(sigm_h(bf2f(bl[(size_t)t * H + h])));
      float* Sh = S.data() + (size_t)h * D * D;
      if (mode != 2) for (int a = 0; a < D; ++a) for (int b = 0; b < D; ++b) Sh[a * D + b] *= eg[a];
      std::fill(kv.begin(), kv.end(), 0.f);
      for (int a = 0; a < D; ++a) for (int b = 0; b < D; ++b) kv[b] += Sh[a * D + b] * k[a];
      for (int b = 0; b < D; ++b) kv[b] = (y[2 * W + h * D + b] - kv[b]) * beta;   // delta
      for (int a = 0; a < D; ++a) for (int b = 0; b < D; ++b) Sh[a * D + b] += k[a] * kv[b];
      if (mode == 2) for (int a = 0; a < D; ++a) for (int b = 0; b < D; ++b) Sh[a * D + b] *= eg[a];
      std::fill(o.begin(), o.end(), 0.f);
      for (int a = 0; a < D; ++a) for (int b = 0; b < D; ++b) o[b] += Sh[a * D + b] * q[a];
      float ss = 0;
      for (int b = 0; b < D; ++b) { o[b] = rbf_h(o[b]); ss += o[b] * o[b]; }
      const float r = 1.0f / std::sqrt(ss / D + 1e-5f);
      for (int b = 0; b < D; ++b)
        out[(size_t)t * W + h * D + b] =
            rbf_h(bf2f(P.o_norm[b]) * (o[b] * r) * sigm_h(bf2f(gate[(size_t)t * W + h * D + b])));
    }
  }
  for (int j = 0; j < 3; ++j) for (int c = 0; c < C; ++c) cs[(size_t)j * C + c] = hist[(size_t)(T + j) * C + c];
}

// ---- device sequence wrapper ----------------------------------------------------------------------------------------
struct DevSeq {   // inputs of one sequence on the device
  bf16 *qkv = nullptr, *gp = nullptr, *bl = nullptr, *gate = nullptr;
  int T = 0;
  void upload(const bf16* q, const bf16* g, const bf16* b, const bf16* ga, int T_) {
    T = T_;
    qkv = dev_upload(std::vector<bf16>(q, q + (size_t)T * C));
    gp = dev_upload(std::vector<bf16>(g, g + (size_t)T * W));
    bl = dev_upload(std::vector<bf16>(b, b + (size_t)T * H));
    gate = dev_upload(std::vector<bf16>(ga, ga + (size_t)T * W));
  }
  void free_all() { cudaFree(qkv); cudaFree(gp); cudaFree(bl); cudaFree(gate); }
};

struct DevState {
  float *cs = nullptr, *S = nullptr;
  DevState() {
    CUDA_CHECK(cudaMalloc(&cs, kKdaConvStateFloats * 4)); CUDA_CHECK(cudaMalloc(&S, kKdaStateFloats * 4));
    zero();
  }
  void zero() { CUDA_CHECK(cudaMemset(cs, 0, kKdaConvStateFloats * 4)); CUDA_CHECK(cudaMemset(S, 0, kKdaStateFloats * 4)); }
  void copy_from(const DevState& o) {
    CUDA_CHECK(cudaMemcpy(cs, o.cs, kKdaConvStateFloats * 4, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(S, o.S, kKdaStateFloats * 4, cudaMemcpyDeviceToDevice));
  }
  void release() { cudaFree(cs); cudaFree(S); }
};

struct Ws {
  void* p = nullptr; size_t bytes = 0;
  void ensure(size_t b) { if (b > bytes) { if (p) cudaFree(p); CUDA_CHECK(cudaMalloc(&p, b)); bytes = b; } }
};

// Run tokens [t0, t0+n) of sq through kda_forward_seq into out rows [t0, t0+n).
void run_prefill(const KdaParams& P, const DevSeq& sq, int t0, int n, DevState& s, bf16* out, Ws& ws) {
  ws.ensure(kda_workspace_bytes(n));
  kda_forward_seq(P, sq.qkv + (size_t)t0 * C, sq.gp + (size_t)t0 * W, sq.bl + (size_t)t0 * H, sq.gate + (size_t)t0 * W, n,
                  s.cs, s.S, out + (size_t)t0 * W, ws.p, ws.bytes, 0);
}
// Token-by-token through kda_decode_batch (M = 1).
void run_decode_seq(const KdaParams& P, const DevSeq& sq, DevState& s, bf16* out) {
  float** pp;
  CUDA_CHECK(cudaMalloc(&pp, 2 * sizeof(float*)));
  CUDA_CHECK(cudaMemcpy(pp, &s.cs, sizeof(float*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pp + 1, &s.S, sizeof(float*), cudaMemcpyHostToDevice));
  for (int t = 0; t < sq.T; ++t)
    kda_decode_batch(P, sq.qkv + (size_t)t * C, sq.gp + (size_t)t * W, sq.bl + (size_t)t * H, sq.gate + (size_t)t * W, 1,
                     pp, pp + 1, out + (size_t)t * W, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(pp);
}

void fill_random(std::mt19937& rng, std::vector<bf16>& v, float sd, float mean = 0.f) {
  std::normal_distribution<float> nd(mean, sd);
  for (auto& x : v) x = f2bf(nd(rng));
}

__global__ void fill_bf16_kernel(bf16* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = f2bf(((x & 0xFFFF) / 65535.0f - 0.5f) * 2.f * scale);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "/tmp/kda_ref";
  const bool skip_bench = argc > 2 && atoi(argv[2]) != 0;
  printf("test_glm_kda  ref=%s  KDA_NEGCTRL=%d\n", dir.c_str(), KDA_NEGCTRL);

  int T = 0;
  { FILE* f = fopen((dir + "/meta.txt").c_str(), "r"); HIVE_CHECK(f && fscanf(f, "%d", &T) == 1, "meta.txt"); fclose(f); }
  HostParams HP;
  HP.conv_w = load_bin<float>(dir + "/conv_w.bin", (size_t)C * 4);
  HP.A_log = load_bin<float>(dir + "/A_log.bin", H);
  HP.dt_bias = load_bin<float>(dir + "/dt_bias.bin", W);
  HP.o_norm = load_bin<bf16>(dir + "/o_norm.bin", D);
  const auto qkv = load_bin<bf16>(dir + "/qkv_pre.bin", (size_t)T * C);
  const auto gp = load_bin<bf16>(dir + "/g_pre.bin", (size_t)T * W);
  const auto bl = load_bin<bf16>(dir + "/beta_logit.bin", (size_t)T * H);
  const auto gate = load_bin<bf16>(dir + "/gate.bin", (size_t)T * W);
  const auto ref_out = load_bin<bf16>(dir + "/out.bin", (size_t)T * W);
  const auto ref_state = load_bin<float>(dir + "/state.bin", kKdaStateFloats);

  KdaParams P{dev_upload(HP.conv_w), dev_upload(HP.A_log), dev_upload(HP.dt_bias), dev_upload(HP.o_norm)};
  DevSeq seq; seq.upload(qkv.data(), gp.data(), bl.data(), gate.data(), T);
  Ws ws;
  bf16* d_out; CUDA_CHECK(cudaMalloc(&d_out, (size_t)T * W * 2));
  bf16* d_out2; CUDA_CHECK(cudaMalloc(&d_out2, (size_t)T * W * 2));

  // ---- 1. prefill vs HF ----
  printf("\n[1] prefill T=%d from zero state vs HF Glm5NextTextLinearAttention (bf16 output before o_proj)\n", T);
  DevState s1;
  run_prefill(P, seq, 0, T, s1, d_out, ws);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto k_out = dev_download(d_out, (size_t)T * W);
  const auto k_state = dev_download(s1.S, kKdaStateFloats);
  const Cmp c_out = compare(k_out.data(), ref_out.data(), k_out.size());
  const Cmp c_st = compare(k_state.data(), ref_state.data(), k_state.size());
  print_cmp("kernel out vs HF", c_out);
  print_cmp("kernel final state vs HF (fp32)", c_st);
  const bool hf_ok = c_out.rel_l2 < 2e-3 && c_st.rel_l2 < 2e-3;
#if KDA_NEGCTRL == 0
  check(hf_ok, "kernel matches HF (rel_l2 < 2e-3 for output and final state)");
#else
  printf("  negative-control build %d: comparison %s\n", KDA_NEGCTRL,
         hf_ok ? "still PASSES -> this mutation is NOT detected" : "FAILS as expected -> mutation detected");
#endif

  // ---- 4a. host reference & negative control (needs only test-1 data) ----
  printf("\n[4] host fp32 reference (HF semantics) and its negative control (decay applied after the update)\n");
  {
    std::vector<float> cs(kKdaConvStateFloats, 0.f), S(kKdaStateFloats, 0.f), hout;
    host_ref(HP, qkv.data(), gp.data(), bl.data(), gate.data(), T, cs, S, hout, 0);
    const Cmp a = compare(hout.data(), ref_out.data(), hout.size());
    const Cmp as = compare(S.data(), ref_state.data(), S.size());
    const Cmp ak = compare(k_out.data(), hout.data(), hout.size());
    const Cmp aks = compare(k_state.data(), S.data(), S.size());
    print_cmp("host ref out vs HF", a);
    print_cmp("host ref state vs HF", as);
    print_cmp("kernel out vs host ref", ak);
    print_cmp("kernel state vs host ref", aks);
    check(a.rel_l2 < 2e-3 && as.rel_l2 < 2e-3, "host reference matches HF");
    std::vector<float> cs2(kKdaConvStateFloats, 0.f), S2(kKdaStateFloats, 0.f), nout;
    host_ref(HP, qkv.data(), gp.data(), bl.data(), gate.data(), T, cs2, S2, nout, 2);
    const Cmp n = compare(nout.data(), ref_out.data(), nout.size());
    print_cmp("NEG host (decay after update) out vs HF", n);
    check(n.rel_l2 > 2e-3, "negative control (decay after update) fails the same comparison");
  }

  // ---- 2. state continuity ----
  printf("\n[2] state continuity on the HF sequence (T=%d): chunked prefill and token-by-token decode vs one-shot\n", T);
  auto cont = [&](const char* tag, const std::vector<int>& splits, bool decode) {
    DevState s; Ws w2;
    CUDA_CHECK(cudaMemset(d_out2, 0, (size_t)T * W * 2));
    if (decode) run_decode_seq(P, seq, s, d_out2);
    else { int t0 = 0; for (int n : splits) { run_prefill(P, seq, t0, n, s, d_out2, w2); t0 += n; } HIVE_CHECK(t0 == T, "splits"); }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto o = dev_download(d_out2, (size_t)T * W);
    const auto S = dev_download(s.S, kKdaStateFloats);
    const auto cs = dev_download(s.cs, kKdaConvStateFloats);
    const auto cs1 = dev_download(s1.cs, kKdaConvStateFloats);
    const Cmp co = compare(o.data(), k_out.data(), o.size()), cs_ = compare(S.data(), k_state.data(), S.size());
    const Cmp cc = compare(cs.data(), cs1.data(), cs.size());
    printf("  %s\n", tag);
    print_cmp("  out vs one-shot", co);
    print_cmp("  final state vs one-shot", cs_);
    print_cmp("  conv state vs one-shot", cc);
    check(co.rel_l2 < 1e-6 && cs_.rel_l2 < 1e-6 && cc.max_abs == 0, tag);
    s.release(); if (w2.p) cudaFree(w2.p);
  };
  cont("prefill 200 + 120", {200, 120}, false);
  cont("prefill 1 + 2 + 61 + 256 (T=1 and T<3 conv-state paths)", {1, 2, 61, 256}, false);
  cont("token-by-token kda_decode_batch (M=1)", {}, true);
  {
    // across the internal 1024-token chunk boundary, synthetic inputs
    const int T2 = 2500;
    std::mt19937 rng(7);
    std::vector<bf16> q2((size_t)T2 * C), g2((size_t)T2 * W), b2((size_t)T2 * H), ga2((size_t)T2 * W);
    fill_random(rng, q2, 1.0f); fill_random(rng, g2, 0.5f); fill_random(rng, b2, 1.0f); fill_random(rng, ga2, 1.0f);
    DevSeq sq2; sq2.upload(q2.data(), g2.data(), b2.data(), ga2.data(), T2);
    bf16 *oa, *ob; CUDA_CHECK(cudaMalloc(&oa, (size_t)T2 * W * 2)); CUDA_CHECK(cudaMalloc(&ob, (size_t)T2 * W * 2));
    DevState sa, sb, sc; Ws w2;
    run_prefill(P, sq2, 0, T2, sa, oa, w2);
    run_prefill(P, sq2, 0, 1000, sb, ob, w2); run_prefill(P, sq2, 1000, 1500, sb, ob, w2);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto o1 = dev_download(oa, (size_t)T2 * W), o2 = dev_download(ob, (size_t)T2 * W);
    auto S1 = dev_download(sa.S, kKdaStateFloats), S2 = dev_download(sb.S, kKdaStateFloats);
    printf("  T=2500 synthetic (one-shot crosses the 1024-token internal chunks)\n");
    Cmp x = compare(o2.data(), o1.data(), o1.size()), y = compare(S2.data(), S1.data(), S1.size());
    print_cmp("  1000+1500 out vs one-shot", x); print_cmp("  1000+1500 state vs one-shot", y);
    check(x.rel_l2 < 1e-6 && y.rel_l2 < 1e-6, "T=2500 prefill 1000+1500 == one-shot");
    run_decode_seq(P, sq2, sc, ob);
    o2 = dev_download(ob, (size_t)T2 * W); S2 = dev_download(sc.S, kKdaStateFloats);
    x = compare(o2.data(), o1.data(), o1.size()); y = compare(S2.data(), S1.data(), S1.size());
    print_cmp("  decode x2500 out vs one-shot", x); print_cmp("  decode x2500 state vs one-shot", y);
    check(x.rel_l2 < 1e-6 && y.rel_l2 < 1e-6, "T=2500 token-by-token decode == one-shot prefill");
    // fp32 host reference on the first 64 tokens of this sequence (independent check with synthetic magnitudes)
    std::vector<float> hcs(kKdaConvStateFloats, 0.f), hS(kKdaStateFloats, 0.f), hout;
    host_ref(HP, q2.data(), g2.data(), b2.data(), ga2.data(), 64, hcs, hS, hout, 0);
    DevState sd; run_prefill(P, sq2, 0, 64, sd, ob, w2); CUDA_CHECK(cudaDeviceSynchronize());
    o2 = dev_download(ob, (size_t)64 * W);
    x = compare(o2.data(), hout.data(), hout.size());
    print_cmp("  synthetic T=64 kernel vs host ref", x);
    check(x.rel_l2 < 2e-3, "synthetic inputs: kernel matches host fp32 reference");
    sa.release(); sb.release(); sc.release(); sd.release(); sq2.free_all(); cudaFree(oa); cudaFree(ob); cudaFree(w2.p);
  }

  // ---- 3. batched decode M=4 vs alone ----
  printf("\n[3] batched decode M=4 (different sequences, different histories) vs each decoded alone\n");
  {
    const int M = 4, steps = 6;
    const int hist_len[M] = {5, 17, 64, 1};
    std::mt19937 rng(11);
    DevState sb[M], sa[M];
    Ws w2;
    // histories
    for (int m = 0; m < M; ++m) {
      std::vector<bf16> q((size_t)hist_len[m] * C), g((size_t)hist_len[m] * W), b((size_t)hist_len[m] * H), ga((size_t)hist_len[m] * W);
      fill_random(rng, q, 1.0f); fill_random(rng, g, 0.5f); fill_random(rng, b, 1.0f); fill_random(rng, ga, 1.0f);
      DevSeq s; s.upload(q.data(), g.data(), b.data(), ga.data(), hist_len[m]);
      bf16* o; CUDA_CHECK(cudaMalloc(&o, (size_t)hist_len[m] * W * 2));
      run_prefill(P, s, 0, hist_len[m], sb[m], o, w2);
      CUDA_CHECK(cudaDeviceSynchronize());
      sa[m].copy_from(sb[m]);
      s.free_all(); cudaFree(o);
    }
    // decode inputs: row m of each step belongs to sequence m
    std::vector<bf16> q((size_t)steps * M * C), g((size_t)steps * M * W), b((size_t)steps * M * H), ga((size_t)steps * M * W);
    fill_random(rng, q, 1.0f); fill_random(rng, g, 0.5f); fill_random(rng, b, 1.0f); fill_random(rng, ga, 1.0f);
    DevSeq in; in.upload(q.data(), g.data(), b.data(), ga.data(), steps * M);
    // batched pointer tables (rows in reverse allocation order to exercise the indirection)
    std::vector<float*> hcs(M), hS(M);
    for (int m = 0; m < M; ++m) { hcs[m] = sb[m].cs; hS[m] = sb[m].S; }
    float** dcs = dev_upload(hcs);
    float** dS = dev_upload(hS);
    bf16 *ob, *oa; CUDA_CHECK(cudaMalloc(&ob, (size_t)steps * M * W * 2)); CUDA_CHECK(cudaMalloc(&oa, (size_t)steps * M * W * 2));
    std::vector<float**> single(M);
    for (int m = 0; m < M; ++m) { std::vector<float*> v{sa[m].cs, sa[m].S}; single[m] = dev_upload(v); }
    for (int t = 0; t < steps; ++t) {
      const size_t r = (size_t)t * M;
      kda_decode_batch(P, in.qkv + r * C, in.gp + r * W, in.bl + r * H, in.gate + r * W, M, dcs, dS, ob + r * W, 0);
      for (int m = 0; m < M; ++m)
        kda_decode_batch(P, in.qkv + (r + m) * C, in.gp + (r + m) * W, in.bl + (r + m) * H, in.gate + (r + m) * W, 1,
                         single[m], single[m] + 1, oa + (r + m) * W, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto vb = dev_download(ob, (size_t)steps * M * W), va = dev_download(oa, (size_t)steps * M * W);
    const Cmp co = compare(vb.data(), va.data(), vb.size());
    print_cmp("batched out vs alone (6 steps x 4 rows)", co);
    double smax = 0;
    for (int m = 0; m < M; ++m) {
      const auto x = dev_download(sb[m].S, kKdaStateFloats), y = dev_download(sa[m].S, kKdaStateFloats);
      const auto xc = dev_download(sb[m].cs, kKdaConvStateFloats), yc = dev_download(sa[m].cs, kKdaConvStateFloats);
      smax = std::max({smax, compare(x.data(), y.data(), x.size()).max_abs, compare(xc.data(), yc.data(), xc.size()).max_abs});
    }
    printf("  max |state diff| over 4 sequences (recurrent + conv): %.3e\n", smax);
    check(co.max_abs == 0 && smax == 0, "batched decode M=4 is bit-identical to per-sequence decode");
    // sanity: the 4 sequences really differ
    const Cmp cross = compare(vb.data(), vb.data() + W, W);
    printf("  (row 0 vs row 1 of step 0 differ in %zu/%d elements)\n", cross.n_diff, W);
    for (int m = 0; m < M; ++m) { sb[m].release(); sa[m].release(); cudaFree(single[m]); }
    cudaFree(dcs); cudaFree(dS); cudaFree(ob); cudaFree(oa); in.free_all(); if (w2.p) cudaFree(w2.p);
  }

  // ---- 4. verify rows: kda_decode_rows vs per-row kda_decode_batch (M=1) + snapshot copies ----
  printf("\n[4] verify rows R=4 of one sequence: kda_decode_rows vs per-row decode + snapshot copies\n");
  {
    const int R = 4, hist = 37;
    std::mt19937 rng(23);
    DevState sr, sp;
    Ws w2;
    {
      std::vector<bf16> q((size_t)hist * C), g((size_t)hist * W), b((size_t)hist * H), ga((size_t)hist * W);
      fill_random(rng, q, 1.0f); fill_random(rng, g, 0.5f); fill_random(rng, b, 1.0f); fill_random(rng, ga, 1.0f);
      DevSeq s; s.upload(q.data(), g.data(), b.data(), ga.data(), hist);
      bf16* o; CUDA_CHECK(cudaMalloc(&o, (size_t)hist * W * 2));
      run_prefill(P, s, 0, hist, sr, o, w2);
      CUDA_CHECK(cudaDeviceSynchronize());
      sp.copy_from(sr);
      s.free_all(); cudaFree(o);
    }
    std::vector<bf16> q((size_t)R * C), g((size_t)R * W), b((size_t)R * H), ga((size_t)R * W);
    fill_random(rng, q, 1.0f); fill_random(rng, g, 0.5f); fill_random(rng, b, 1.0f); fill_random(rng, ga, 1.0f);
    DevSeq in; in.upload(q.data(), g.data(), b.data(), ga.data(), R);
    std::vector<float*> snc(R - 1), sns(R - 1);
    for (int r = 0; r < R - 1; ++r) {
      CUDA_CHECK(cudaMalloc(&snc[r], kKdaConvStateFloats * 4)); CUDA_CHECK(cudaMalloc(&sns[r], kKdaStateFloats * 4));
      CUDA_CHECK(cudaMemset(snc[r], 0, kKdaConvStateFloats * 4)); CUDA_CHECK(cudaMemset(sns[r], 0, kKdaStateFloats * 4));
    }
    std::vector<float*> one_r{sr.cs, sr.S}, one_p{sp.cs, sp.S};
    float** pr = dev_upload(one_r); float** pp = dev_upload(one_p);
    float** dsnc = dev_upload(snc); float** dsns = dev_upload(sns);
    bf16 *orow, *oper; CUDA_CHECK(cudaMalloc(&orow, (size_t)R * W * 2)); CUDA_CHECK(cudaMalloc(&oper, (size_t)R * W * 2));
    kda_decode_rows(P, in.qkv, in.gp, in.bl, in.gate, R, pr, pr + 1, dsnc, dsns, orow, 0);
    std::vector<std::vector<float>> ref_s, ref_c;
    for (int r = 0; r < R; ++r) {
      kda_decode_batch(P, in.qkv + (size_t)r * C, in.gp + (size_t)r * W, in.bl + (size_t)r * H, in.gate + (size_t)r * W, 1, pp, pp + 1,
                       oper + (size_t)r * W, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      if (r + 1 < R) { ref_s.push_back(dev_download(sp.S, kKdaStateFloats)); ref_c.push_back(dev_download(sp.cs, kKdaConvStateFloats)); }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto vr = dev_download(orow, (size_t)R * W), vp = dev_download(oper, (size_t)R * W);
    const Cmp co = compare(vr.data(), vp.data(), vr.size());
    double smax = std::max(compare(dev_download(sr.S, kKdaStateFloats).data(), dev_download(sp.S, kKdaStateFloats).data(), kKdaStateFloats).max_abs,
                           compare(dev_download(sr.cs, kKdaConvStateFloats).data(), dev_download(sp.cs, kKdaConvStateFloats).data(), kKdaConvStateFloats).max_abs);
    double snmax = 0;
    for (int r = 0; r < R - 1; ++r) {
      snmax = std::max(snmax, compare(dev_download(sns[r], kKdaStateFloats).data(), ref_s[r].data(), kKdaStateFloats).max_abs);
      snmax = std::max(snmax, compare(dev_download(snc[r], kKdaConvStateFloats).data(), ref_c[r].data(), kKdaConvStateFloats).max_abs);
    }
    print_cmp("rows out vs per-row out", co);
    printf("  max |final state diff| %.3e · max |snapshot diff| over rows 0..%d %.3e\n", smax, R - 2, snmax);
    check(co.max_abs == 0 && smax == 0 && snmax == 0, "kda_decode_rows is bit-identical to per-row decode (outputs, final state, snapshots)");
    // timing: one layer, R = 2..4 (per-row path = R launches + 2(R−1) snapshot copies, as the engine's former verify path)
    if (!skip_bench) {
      cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
      for (int RR = 2; RR <= R; ++RR) {
        float ms_rows = 0, ms_per = 0;
        const int reps = 200;
        CUDA_CHECK(cudaEventRecord(e0));
        for (int it = 0; it < reps; ++it) kda_decode_rows(P, in.qkv, in.gp, in.bl, in.gate, RR, pr, pr + 1, dsnc, dsns, orow, 0);
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1)); CUDA_CHECK(cudaEventElapsedTime(&ms_rows, e0, e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int it = 0; it < reps; ++it)
          for (int r = 0; r < RR; ++r) {
            kda_decode_batch(P, in.qkv + (size_t)r * C, in.gp + (size_t)r * W, in.bl + (size_t)r * H, in.gate + (size_t)r * W, 1, pp, pp + 1, oper + (size_t)r * W, 0);
            if (r + 1 < RR) { CUDA_CHECK(cudaMemcpyAsync(sns[r], sp.S, kKdaStateFloats * 4, cudaMemcpyDeviceToDevice, 0));
                              CUDA_CHECK(cudaMemcpyAsync(snc[r], sp.cs, kKdaConvStateFloats * 4, cudaMemcpyDeviceToDevice, 0)); }
          }
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1)); CUDA_CHECK(cudaEventElapsedTime(&ms_per, e0, e1));
        printf("  R=%d: rows kernel %.1f us · per-row + copies %.1f us (one layer)\n", RR, 1000.0 * ms_rows / reps, 1000.0 * ms_per / reps);
      }
    }
    sr.release(); sp.release(); for (int r = 0; r < R - 1; ++r) { cudaFree(snc[r]); cudaFree(sns[r]); }
    cudaFree(pr); cudaFree(pp); cudaFree(dsnc); cudaFree(dsns); cudaFree(orow); cudaFree(oper); in.free_all(); if (w2.p) cudaFree(w2.p);
  }

  s1.release(); seq.free_all(); cudaFree(d_out); cudaFree(d_out2); if (ws.p) cudaFree(ws.p);

  // ---- 5. timings ----
  if (!skip_bench && KDA_NEGCTRL == 0) {
    printf("\n[5] timings (RTX PRO 6000, one layer)\n");
    const int Tmax = 16384;
    bf16 *qkv_b, *gate_b, *bl_b;
    CUDA_CHECK(cudaMalloc(&qkv_b, (size_t)Tmax * C * 2));
    CUDA_CHECK(cudaMalloc(&gate_b, (size_t)Tmax * W * 2));   // also used as g_pre and out (out may alias gate)
    CUDA_CHECK(cudaMalloc(&bl_b, (size_t)Tmax * H * 2));
    fill_bf16_kernel<<<1024, 256>>>(qkv_b, (size_t)Tmax * C, 1, 2.f);
    fill_bf16_kernel<<<1024, 256>>>(gate_b, (size_t)Tmax * W, 2, 1.f);
    fill_bf16_kernel<<<1024, 256>>>(bl_b, (size_t)Tmax * H, 3, 2.f);
    DevState s;
    Ws w2; w2.ensure(kda_workspace_bytes(Tmax));
    size_t fr, tot; CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    printf("  prefill buffers: qkv %.0f MB, gate/g_pre/out %.0f MB, workspace %.0f MB\n", Tmax * (double)C * 2 / 1e6,
           Tmax * (double)W * 2 / 1e6, kda_workspace_bytes(Tmax) / 1e6);
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    for (int Tb : {1, 64, 2048, 16384}) {
      const int reps = Tb >= 16384 ? 5 : Tb >= 2048 ? 20 : 200;
      for (int i = 0; i < 2; ++i)
        kda_forward_seq(P, qkv_b, gate_b, bl_b, gate_b, Tb, s.cs, s.S, gate_b, w2.p, w2.bytes, 0);
      CUDA_CHECK(cudaEventRecord(e0));
      for (int i = 0; i < reps; ++i)
        kda_forward_seq(P, qkv_b, gate_b, bl_b, gate_b, Tb, s.cs, s.S, gate_b, w2.p, w2.bytes, 0);
      CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
      float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      printf("  prefill T=%-6d %9.3f ms/layer  (%.2f us/token)\n", Tb, ms / reps, ms / reps * 1000.0 / Tb);
    }
    s.release(); cudaFree(qkv_b); cudaFree(gate_b); cudaFree(bl_b); cudaFree(w2.p);

    // decode
    const int Mmax = 8;
    bf16 *dq, *dg; CUDA_CHECK(cudaMalloc(&dq, (size_t)Mmax * C * 2)); CUDA_CHECK(cudaMalloc(&dg, (size_t)Mmax * W * 2));
    bf16 *dbl, *dout; CUDA_CHECK(cudaMalloc(&dbl, (size_t)Mmax * H * 2)); CUDA_CHECK(cudaMalloc(&dout, (size_t)Mmax * W * 2));
    fill_bf16_kernel<<<256, 256>>>(dq, (size_t)Mmax * C, 4, 2.f);
    fill_bf16_kernel<<<256, 256>>>(dg, (size_t)Mmax * W, 5, 1.f);
    fill_bf16_kernel<<<256, 256>>>(dbl, (size_t)Mmax * H, 6, 2.f);
    std::vector<DevState> ds(Mmax);
    std::vector<float*> hcs(Mmax), hS(Mmax);
    for (int m = 0; m < Mmax; ++m) { hcs[m] = ds[m].cs; hS[m] = ds[m].S; }
    float** dcs = dev_upload(hcs); float** dS = dev_upload(hS);
    void* flush; const size_t flush_bytes = 256ull << 20; CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
    for (int M : {1, 4, 8}) {
      const int reps = 200;
      for (int i = 0; i < 5; ++i) kda_decode_batch(P, dq, dg, dbl, dg, M, dcs, dS, dout, 0);
      CUDA_CHECK(cudaEventRecord(e0));
      for (int i = 0; i < reps; ++i) kda_decode_batch(P, dq, dg, dbl, dg, M, dcs, dS, dout, 0);
      CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
      float hot; CUDA_CHECK(cudaEventElapsedTime(&hot, e0, e1));
      float cold = 0;
      for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush, i, flush_bytes));
        CUDA_CHECK(cudaEventRecord(e0));
        kda_decode_batch(P, dq, dg, dbl, dg, M, dcs, dS, dout, 0);
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); cold += ms;
      }
      const double bytes = (double)M * kKdaStateFloats * 4 * 2;
      printf("  decode M=%d  hot %7.2f us   L2-flushed %7.2f us  (state r+w %.1f MB -> %.0f GB/s cold)\n", M,
             hot / reps * 1000, cold / 20 * 1000, bytes / 1e6, bytes / (cold / 20 * 1e-3) / 1e9);
    }
    for (auto& d : ds) d.release();
    cudaFree(dcs); cudaFree(dS); cudaFree(flush); cudaFree(dq); cudaFree(dg); cudaFree(dbl); cudaFree(dout);
  }

#if KDA_NEGCTRL != 0
  // Negative-control build: the only verdict that matters is whether test 1 rejected the mutated kernel.
  printf("\nNEGATIVE CONTROL %d: %s\n", KDA_NEGCTRL, hf_ok ? "NOT DETECTED" : "DETECTED (test 1 rejects the mutated kernel)");
  return hf_ok ? 1 : 0;
#else
  printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "FAILED", g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
#endif
}
