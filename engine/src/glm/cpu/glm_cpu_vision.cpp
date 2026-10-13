// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash CPU engine — the vision encoder (model.visual, BF16), the math of engine/src/glm/glm_vision.cu (hive/glm/glm_vision.h) with the
//   same bf16 rounding points: patch embedding, 24 blocks (RMSNorm, qkv + bias, per-head RMSNorm, axial 2D rotary, non-causal attention,
//   clamped SwiGLU), RMSNorm, 2×2 downsample, merger (linear → LayerNorm → GELU → clamped SwiGLU). Loaded on the first image.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "glm_cpu_internal.h"
#include "hive/clock.h"

namespace hive::glm::cpu {

using ::hive::cpu::WType;

namespace {
constexpr int kPatchDim = 3 * 2 * 14 * 14;
void add_bias(float* x, const std::vector<float>& b, int M, int N) {
  par_for((size_t)M, 64, [&](size_t a, size_t e) { for (size_t m = a; m < e; ++m) for (int i = 0; i < N; ++i) x[m * N + i] = bf16r(x[m * N + i] + b[i]); });
}
// x = bf16(x + Y) (the cuBLAS β = 1 accumulate into a bf16 matrix)
void add_into(float* x, const float* y, size_t n) {
  par_for(n, 1 << 16, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) x[i] = bf16r(x[i] + y[i]); });
}
void swiglu_clamp(const float* g, const float* u, float* out, size_t n, float lim) {
  par_for(n, 1 << 16, [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) {
      const float ga = std::fmin(g[i], lim), ub = std::fmin(std::fmax(u[i], -lim), lim);
      out[i] = bf16r(bf16r(ga / (1.f + std::exp(-ga))) * ub);
    }
  });
}
}  // namespace

void GlmCpuEngine::Impl::ensure_vision() {
  if (vision) return;
  const double t0 = hive::mono_ms();
  auto V = std::make_unique<VisionW>();
  V->dim = c.vision_dim; V->heads = c.vision_heads; V->inter = c.vision_inter; V->out = c.vision_out; V->pinter = c.vision_proj_inter;
  if (V->dim / V->heads != 64 || c.vision_patch != 14 || c.vision_temporal != 2 || c.vision_merge != 2) throw std::runtime_error("glm cpu vision: unsupported geometry");
  Checkpoint& k = *ck;
  const std::string p = "model.visual.";
  {  // conv3d [1024, 3, 2, 14, 14] = linear [1024, 1176] (the patches are flattened in the same order)
    const TensorInfo& t = k.get(p + "patch_embed.proj.weight");
    if (t.dtype != "BF16" || (size_t)t.numel() != (size_t)V->dim * kPatchDim) throw std::runtime_error("glm cpu vision: patch_embed.proj.weight");
    V->patch = ::hive::cpu::make_mat(WType::BF16, V->dim, kPatchDim, t.data);
  }
  V->patch_b = load_vec(k, p + "patch_embed.proj.bias");
  V->blocks.resize(c.vision_layers);
  for (int l = 0; l < c.vision_layers; ++l) {
    const std::string b = p + "blocks." + std::to_string(l) + ".";
    VisionW::Block& B = V->blocks[l];
    B.n1 = load_vec(k, b + "norm1.weight"); B.n2 = load_vec(k, b + "norm2.weight");
    B.qkv = load_bf16_mat(k, {b + "attn.qkv.weight"}); B.qkv_b = load_vec(k, b + "attn.qkv.bias");
    B.q_norm = load_vec(k, b + "attn.q_norm.weight"); B.k_norm = load_vec(k, b + "attn.k_norm.weight");
    B.proj = load_bf16_mat(k, {b + "attn.proj.weight"}); B.proj_b = load_vec(k, b + "attn.proj.bias");
    B.gate = load_bf16_mat(k, {b + "mlp.gate_proj.weight"}); B.gate_b = load_vec(k, b + "mlp.gate_proj.bias");
    B.up = load_bf16_mat(k, {b + "mlp.up_proj.weight"}); B.up_b = load_vec(k, b + "mlp.up_proj.bias");
    B.down = load_bf16_mat(k, {b + "mlp.down_proj.weight"}); B.down_b = load_vec(k, b + "mlp.down_proj.bias");
  }
  V->post_norm = load_vec(k, p + "post_layernorm.weight");
  {  // downsample conv2d [out, dim, 2, 2] → linear [out, 4·dim], input order (i·2 + j)·dim + ch (4 consecutive patches of a merge block)
    const TensorInfo& t = k.get(p + "downsample.weight");
    const int O = V->out, D = V->dim;
    if (t.dtype != "BF16" || (size_t)t.numel() != (size_t)O * D * 4) throw std::runtime_error("glm cpu vision: downsample.weight");
    const uint16_t* src = (const uint16_t*)t.data;
    std::vector<uint16_t> dst((size_t)t.numel());
    par_for((size_t)O, 64, [&](size_t a, size_t e) {
      for (size_t o = a; o < e; ++o)
        for (int ch = 0; ch < D; ++ch)
          for (int ij = 0; ij < 4; ++ij) dst[o * 4 * D + (size_t)ij * D + ch] = src[(o * D + ch) * 4 + ij];
    });
    V->down = ::hive::cpu::make_mat(WType::BF16, O, 4 * D, dst.data());
  }
  V->down_b = load_vec(k, p + "downsample.bias");
  V->m_proj = load_bf16_mat(k, {p + "merger.proj.weight"});
  V->m_ln_w = load_vec(k, p + "merger.post_projection_norm.weight"); V->m_ln_b = load_vec(k, p + "merger.post_projection_norm.bias");
  V->m_gate = load_bf16_mat(k, {p + "merger.gate_proj.weight"}); V->m_up = load_bf16_mat(k, {p + "merger.up_proj.weight"});
  V->m_down = load_bf16_mat(k, {p + "merger.down_proj.weight"});
  vision = std::move(V);
  fprintf(stderr, "[glm-cpu] vision encoder loaded in %.0f ms\n", hive::mono_ms() - t0);
}

void GlmCpuEngine::Impl::encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out) {
  ensure_vision();
  const VisionW& V = *vision;
  if (gh <= 0 || gw <= 0 || gh % 2 || gw % 2) throw std::runtime_error("glm cpu vision: grid must be even");
  const int N = gh * gw, D = V.dim, NH = V.heads, HD = 64, I = V.inter, NB = N / 4;
  std::vector<float> X((size_t)N * D), XN((size_t)N * D), QKV((size_t)N * 3 * D), ATT((size_t)N * D), T((size_t)N * std::max(D, I));
  std::vector<float> G((size_t)N * I), U((size_t)N * I);
  std::vector<int32_t> pos((size_t)N * 2);
  {
    size_t r = 0;
    for (int bh = 0; bh < gh / 2; ++bh)
      for (int bw = 0; bw < gw / 2; ++bw)
        for (int mh = 0; mh < 2; ++mh)
          for (int mw = 0; mw < 2; ++mw) { pos[r * 2] = bh * 2 + mh; pos[r * 2 + 1] = bw * 2 + mw; ++r; }
  }
  float inv[16];
  for (int i = 0; i < 16; ++i) inv[i] = 1.f / std::pow(10000.f, (2.f * i) / 32.f);
  {
    std::vector<float> pf((size_t)N * kPatchDim);
    par_for(pf.size(), 1 << 16, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) pf[i] = b2f(patches[i]); });
    lin(V.patch, pf.data(), N, kPatchDim, X.data(), D);
    add_bias(X.data(), V.patch_b, N, D);
  }
  const float scale = 1.f / 8.f;
  for (const VisionW::Block& B : V.blocks) {
    rmsnorm_rows(X.data(), D, B.n1.data(), N, D, V.eps, XN.data(), D);
    lin(B.qkv, XN.data(), N, D, QKV.data(), 3 * D);
    add_bias(QKV.data(), B.qkv_b, N, 3 * D);
    // q / k: per-head RMSNorm (bf16, then × weight, bf16) and the axial rotary embedding (bf16)
    par_for((size_t)N * NH * 2, 64, [&](size_t a, size_t b) {
      for (size_t i = a; i < b; ++i) {
        const size_t row = i / (NH * 2), head = (i / 2) % NH, which = i % 2;
        float* p = QKV.data() + row * 3 * D + which * D + head * HD;
        float ss = 0.f;
        for (int d = 0; d < HD; ++d) ss += p[d] * p[d];
        const float r = 1.f / std::sqrt(ss / 64.f + V.eps);
        const float* w = which ? B.k_norm.data() : B.q_norm.data();
        float v[64];
        for (int d = 0; d < HD; ++d) v[d] = bf16r(bf16r(p[d] * r) * w[d]);
        for (int d = 0; d < HD; ++d) {
          const int j = d & 31;
          const float ang = (float)(j < 16 ? pos[row * 2] : pos[row * 2 + 1]) * inv[j & 15];
          const float rot = d < 32 ? -v[d + 32] : v[d - 32];
          p[d] = bf16r(v[d] * std::cos(ang) + rot * std::sin(ang));
        }
      }
    });
    // non-causal attention per head: P = bf16(softmax(Q·Kᵀ·scale)), ATT = bf16(P·V)
    for (int h = 0; h < NH; ++h) {
      std::vector<float> Kh((size_t)N * HD), Vh((size_t)N * HD);
      for (int r = 0; r < N; ++r)
        for (int d = 0; d < HD; ++d) { Kh[(size_t)r * HD + d] = QKV[(size_t)r * 3 * D + D + h * HD + d]; Vh[(size_t)r * HD + d] = QKV[(size_t)r * 3 * D + 2 * D + h * HD + d]; }
      par_for((size_t)N, 4, [&](size_t a, size_t b) {
        std::vector<float> s(N);
        for (size_t r = a; r < b; ++r) {
          const float* q = QKV.data() + r * 3 * D + h * HD;
          float mx = -INFINITY;
          for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            const float* kr = Kh.data() + (size_t)j * HD;
            for (int d = 0; d < HD; ++d) acc += q[d] * kr[d];
            s[j] = acc * scale; mx = std::max(mx, s[j]);
          }
          float sum = 0.f;
          for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
          const float is = 1.f / sum;
          float o[64] = {0};
          for (int j = 0; j < N; ++j) {
            const float pj = bf16r(s[j] * is);
            const float* vr = Vh.data() + (size_t)j * HD;
            for (int d = 0; d < HD; ++d) o[d] += pj * vr[d];
          }
          for (int d = 0; d < HD; ++d) ATT[r * D + h * HD + d] = bf16r(o[d]);
        }
      });
    }
    lin(B.proj, ATT.data(), N, D, T.data(), D, false);
    add_into(X.data(), T.data(), (size_t)N * D);
    add_bias(X.data(), B.proj_b, N, D);
    rmsnorm_rows(X.data(), D, B.n2.data(), N, D, V.eps, XN.data(), D);
    lin(B.gate, XN.data(), N, D, G.data(), I); add_bias(G.data(), B.gate_b, N, I);
    lin(B.up, XN.data(), N, D, U.data(), I); add_bias(U.data(), B.up_b, N, I);
    swiglu_clamp(G.data(), U.data(), G.data(), (size_t)N * I, V.limit);
    lin(B.down, G.data(), N, I, T.data(), D, false);
    add_into(X.data(), T.data(), (size_t)N * D);
    add_bias(X.data(), B.down_b, N, D);
  }
  rmsnorm_rows(X.data(), D, V.post_norm.data(), N, D, V.eps, XN.data(), D);
  const int O = V.out, PI = V.pinter;
  std::vector<float> ds((size_t)NB * O), mt((size_t)NB * O), mg((size_t)NB * PI), mu((size_t)NB * PI);
  lin(V.down, XN.data(), NB, 4 * D, ds.data(), O);
  add_bias(ds.data(), V.down_b, NB, O);
  lin(V.m_proj, ds.data(), NB, O, mt.data(), O);
  par_for((size_t)NB, 4, [&](size_t a, size_t b) {  // LayerNorm (eps 1e-5) then exact GELU, each rounded to bf16
    for (size_t m = a; m < b; ++m) {
      const float* x = mt.data() + m * O;
      float s = 0.f, s2 = 0.f;
      for (int i = 0; i < O; ++i) { s += x[i]; s2 += x[i] * x[i]; }
      const float mean = s / O, var = std::max(s2 / O - mean * mean, 0.f), iv = 1.f / std::sqrt(var + 1e-5f);
      for (int i = 0; i < O; ++i) {
        const float v = bf16r((x[i] - mean) * iv * V.m_ln_w[i] + V.m_ln_b[i]);
        ds[m * O + i] = bf16r(0.5f * v * (1.f + std::erf(v * 0.70710678118654752f)));
      }
    }
  });
  lin(V.m_gate, ds.data(), NB, O, mg.data(), PI);
  lin(V.m_up, ds.data(), NB, O, mu.data(), PI);
  swiglu_clamp(mg.data(), mu.data(), mg.data(), (size_t)NB * PI, V.limit);
  out.resize((size_t)NB * O);
  lin(V.m_down, mg.data(), NB, PI, out.data(), O);
}

}  // namespace hive::glm::cpu
