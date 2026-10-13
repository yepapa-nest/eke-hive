// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B CPU engine — the vision encoder (model.visual) on the CPU, adapted from the Flash-Next CPU encoder
//   (src/qwen/cpu/qwen_cpu_vision.cpp: the same structure; 27B only differs in the merger width, 5120) — bf16 rounding points as
//   engine/src/qwen/qwen_vision.cu (see hive/qwen/qwen_vision.h): patch linear + bias, bilinear position grid, 27 pre-LN blocks with axial
//   2D rotary attention (non-causal) and a GELU-tanh MLP, then the 2×2 merger (LayerNorm, concat 4, fc1, GELU-erf, fc2).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include "q27_cpu_internal.h"

namespace hive::q27::cpu {


namespace {
std::mutex g_vision_mu;

void layernorm(const float* x, const std::vector<float>& w, const std::vector<float>& b, int N, int D, float eps, float* y) {
  par_for((size_t)N, 16, [&](size_t a, size_t e) {
    for (size_t r = a; r < e; ++r) {
      const float* xr = x + r * D;
      double m = 0, v = 0;
      for (int d = 0; d < D; ++d) m += xr[d];
      m /= D;
      for (int d = 0; d < D; ++d) v += (xr[d] - m) * (xr[d] - m);
      const float rs = 1.f / std::sqrt((float)(v / D) + eps);
      for (int d = 0; d < D; ++d) y[r * D + d] = bf16r(((float)(xr[d] - m)) * rs * w[d] + b[d]);
    }
  });
}
// y = bf16(bf16(X·Wᵀ) + bias)
void lin_bias(const Mat& W, const std::vector<float>& bias, const float* X, int N, float* Y) {
  lin(W, X, N, W.K, Y, W.N);
  par_for((size_t)N, 8, [&](size_t a, size_t e) {
    for (size_t r = a; r < e; ++r) for (int j = 0; j < W.N; ++j) Y[r * W.N + j] = bf16r(Y[r * W.N + j] + bias[j]);
  });
}
// x = bf16(bf16(x + X·Wᵀ) + bias)  (the residual GEMM accumulates into the bf16 stream, then the bias)
void acc_bias(const Mat& W, const std::vector<float>& bias, const float* X, int N, float* x, std::vector<float>& tmp) {
  tmp.resize((size_t)N * W.N);
  ::hive::cpu::matmul(W, X, N, W.K, tmp.data(), W.N);
  par_for((size_t)N, 8, [&](size_t a, size_t e) {
    for (size_t r = a; r < e; ++r)
      for (int j = 0; j < W.N; ++j) { float& v = x[r * W.N + j]; v = bf16r(bf16r(v + tmp[r * W.N + j]) + bias[j]); }
  });
}
}  // namespace

void Q27CpuEngine::Impl::ensure_vision() {
  std::lock_guard<std::mutex> lk(g_vision_mu);
  if (vision) return;
  // the checkpoint was released after loading the language model: reopen it for the (lazily loaded) encoder
  if (!ck) ck = std::make_unique<Checkpoint>(ckpt_dir);
  if (!ck->has("model.visual.patch_embed.proj.weight")) throw std::runtime_error("q27 cpu: the checkpoint has no vision encoder");
  auto v = std::make_unique<VisionW>();
  v->dim = c.vision_dim; v->heads = c.vision_heads; v->inter = c.vision_inter; v->out = c.vision_out;
  v->side = (int)std::lround(std::sqrt((double)c.vision_pos));
  if (v->dim / v->heads != 72 || v->side * v->side != c.vision_pos || c.vision_merge != 2) throw std::runtime_error("q27 cpu: unsupported vision geometry");
  const std::string p = "model.visual.";
  {  // conv3d weight [dim, 3, 2, 16, 16] (5-D, so not through load_bf16_mat) is read as the matrix [dim, 1536]
    const TensorInfo& t = ck->get(p + "patch_embed.proj.weight");
    if (t.dtype != "BF16" || t.shape.empty() || (int)t.shape[0] != v->dim || t.numel() != (int64_t)v->dim * 1536)
      throw std::runtime_error("q27 cpu: unexpected patch_embed.proj.weight layout");
    v->patch = ::hive::cpu::make_mat(::hive::cpu::WType::BF16, v->dim, 1536, t.data);
  }
  v->patch_b = load_vec(*ck, p + "patch_embed.proj.bias");
  v->pos = load_vec(*ck, p + "pos_embed.weight");
  v->blocks.resize(c.vision_layers);
  for (int l = 0; l < c.vision_layers; ++l) {
    const std::string b = p + "blocks." + std::to_string(l) + ".";
    VisionW::Block& B = v->blocks[l];
    B.n1w = load_vec(*ck, b + "norm1.weight"); B.n1b = load_vec(*ck, b + "norm1.bias");
    B.n2w = load_vec(*ck, b + "norm2.weight"); B.n2b = load_vec(*ck, b + "norm2.bias");
    B.qkv = load_mat(*ck, {b + "attn.qkv"}); B.qkv_b = load_vec(*ck, b + "attn.qkv.bias");
    B.proj = load_mat(*ck, {b + "attn.proj"}); B.proj_b = load_vec(*ck, b + "attn.proj.bias");
    B.fc1 = load_mat(*ck, {b + "mlp.linear_fc1"}); B.fc1_b = load_vec(*ck, b + "mlp.linear_fc1.bias");
    B.fc2 = load_mat(*ck, {b + "mlp.linear_fc2"}); B.fc2_b = load_vec(*ck, b + "mlp.linear_fc2.bias");
  }
  v->mn_w = load_vec(*ck, p + "merger.norm.weight"); v->mn_b = load_vec(*ck, p + "merger.norm.bias");
  v->m1 = load_mat(*ck, {p + "merger.linear_fc1"}); v->m1_b = load_vec(*ck, p + "merger.linear_fc1.bias");
  v->m2 = load_mat(*ck, {p + "merger.linear_fc2"}); v->m2_b = load_vec(*ck, p + "merger.linear_fc2.bias");
  ck->release_all();
  vision = std::move(v);
}

void Q27CpuEngine::Impl::encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out) {
  ensure_vision();
  const VisionW& v = *vision;
  if (gh <= 0 || gw <= 0 || gh % 2 || gw % 2) throw std::runtime_error("q27 cpu vision: the grid must be even");
  const int N = gh * gw, D = v.dim, HD = D / v.heads, NB = N / 4, PD = v.patch.K, side = v.side;
  std::vector<float> x((size_t)N * D), xn((size_t)N * D), qkv((size_t)N * 3 * D), att((size_t)N * D), f1((size_t)N * v.inter), tmp;
  {
    std::vector<float> pf((size_t)N * PD);
    ::hive::cpu::bf16_to_f32_n(patches, pf.size(), pf.data());
    lin_bias(v.patch, v.patch_b, pf.data(), N, x.data());
  }
  // position grid of every patch row (merge-block order) and the bilinear position embedding (align-corners samples)
  std::vector<int32_t> rc((size_t)N * 2);
  {
    auto samples = [&](int n, std::vector<int>& f, std::vector<int>& cl, std::vector<float>& fr) {
      f.resize(n); cl.resize(n); fr.resize(n);
      for (int i = 0; i < n; ++i) {
        const float t = n == 1 ? 0.f : (float)i * (float)(side - 1) / (float)(n - 1);
        const int fl = std::min((int)t, side - 1);
        f[i] = fl; cl[i] = std::min(fl + 1, side - 1); fr[i] = t - (float)fl;
      }
    };
    std::vector<int> hf, hc, wf, wc;
    std::vector<float> dh, dw;
    samples(gh, hf, hc, dh);
    samples(gw, wf, wc, dw);
    size_t r = 0;
    for (int bh = 0; bh < gh / 2; ++bh)
      for (int bw = 0; bw < gw / 2; ++bw)
        for (int mh = 0; mh < 2; ++mh)
          for (int mw = 0; mw < 2; ++mw, ++r) {
            const int y = bh * 2 + mh, xx = bw * 2 + mw;
            rc[r * 2] = y; rc[r * 2 + 1] = xx;
            const float a = dh[y], b = dw[xx];
            const int ci[4] = {hf[y] * side + wf[xx], hf[y] * side + wc[xx], hc[y] * side + wf[xx], hc[y] * side + wc[xx]};
            const float cw[4] = {(1.f - a) * (1.f - b), (1.f - a) * b, a * (1.f - b), a * b};
            for (int d = 0; d < D; ++d) {
              float s = 0.f;
              for (int i = 0; i < 4; ++i) s = std::fma(cw[i], v.pos[(size_t)ci[i] * D + d], s);
              x[r * D + d] = bf16r(x[r * D + d] + bf16r(s));
            }
          }
  }
  float inv[18];
  for (int j = 0; j < 18; ++j) inv[j] = 1.f / std::pow(10000.f, (2.f * j) / 36.f);
  std::vector<uint16_t> Kc((size_t)N * D), Vc((size_t)N * D);
  std::vector<float> Q((size_t)N * D);
  std::vector<int32_t> ce(N, N);
  auto to_b = [](float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); };
  for (const VisionW::Block& B : v.blocks) {
    layernorm(x.data(), B.n1w, B.n1b, N, D, v.eps, xn.data());
    lin_bias(B.qkv, B.qkv_b, xn.data(), N, qkv.data());
    // axial rotary on q and k, then q fp32 and k / v as bf16 caches [N, heads, 72]
    par_for((size_t)N, 16, [&](size_t a, size_t e) {
      for (size_t r = a; r < e; ++r) {
        float cs[36], sn[36];
        for (int d = 0; d < 36; ++d) {
          const int j = d;
          const float ang = (float)(j < 18 ? rc[r * 2] : rc[r * 2 + 1]) * inv[j < 18 ? j : j - 18];
          cs[d] = std::cos(ang); sn[d] = std::sin(ang);
        }
        for (int which = 0; which < 2; ++which)
          for (int h = 0; h < v.heads; ++h) {
            const float* src = qkv.data() + r * 3 * D + (size_t)which * D + (size_t)h * HD;
            float o[72];
            for (int d = 0; d < HD; ++d) {
              const int j = d < 36 ? d : d - 36;
              const float rot = d < 36 ? -src[d + 36] : src[d - 36];
              o[d] = bf16r(src[d] * cs[j] + rot * sn[j]);
            }
            for (int d = 0; d < HD; ++d) {
              if (which == 0) Q[r * D + (size_t)h * HD + d] = o[d];
              else Kc[r * D + (size_t)h * HD + d] = to_b(o[d]);
            }
          }
        for (int d = 0; d < D; ++d) Vc[r * D + d] = to_b(qkv[r * 3 * D + 2 * D + d]);
      }
    });
    ::hive::cpu::attention(Q.data(), N, v.heads, v.heads, HD, Kc.data(), Vc.data(), nullptr, 0, ce.data(), 1.f / std::sqrt((float)HD), att.data());
    ::hive::cpu::round_bf16(att.data(), att.size());
    acc_bias(B.proj, B.proj_b, att.data(), N, x.data(), tmp);
    layernorm(x.data(), B.n2w, B.n2b, N, D, v.eps, xn.data());
    lin_bias(B.fc1, B.fc1_b, xn.data(), N, f1.data());
    par_for(f1.size(), 65536, [&](size_t a, size_t e) {
      for (size_t i = a; i < e; ++i) {
        const float t = f1[i];
        f1[i] = bf16r(0.5f * t * (1.f + std::tanh(0.7978845608028654f * (t + 0.044715f * t * t * t))));
      }
    });
    acc_bias(B.fc2, B.fc2_b, f1.data(), N, x.data(), tmp);
  }
  layernorm(x.data(), v.mn_w, v.mn_b, N, D, v.eps, xn.data());   // [N, D] = [NB, 4·D] rows of one merge block
  std::vector<float> mh((size_t)NB * 4 * D);
  lin_bias(v.m1, v.m1_b, xn.data(), NB, mh.data());
  for (float& t : mh) t = bf16r(0.5f * t * (1.f + std::erf(t * 0.70710678118654752f)));
  out.resize((size_t)NB * v.out);
  lin_bias(v.m2, v.m2_b, mh.data(), NB, out.data());
}

void Q27CpuEngine::encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out) { p_->encode_image(patches, gh, gw, out); }

}  // namespace hive::q27::cpu
