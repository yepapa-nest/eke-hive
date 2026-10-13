// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash CPU engine — vision encoder (reference inference/vision.py, as hive/vision.h on the GPU): 32-layer bf16 ViT with 2D RoPE
//   (h positions on the first 16 rotary frequencies, w positions on the next 16; halves rotation) and bidirectional attention, final RMS
//   norm, then the Aligner (3×3 unfold of the patch grid, zero padded → Linear → GELU (erf) → Linear). Every bf16 module output is rounded to
//   bf16; linears add their bias to the fp32 accumulator before the rounding; the attention computes an exact fp32 softmax.
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ds_cpu_internal.h"
#include "hive/clock.h"
#include "hive/common.h"

namespace hive::ds::cpu {

using ::hive::cpu::Pool;

bool DsCpuEngine::has_vision() const { return p_->opt.vision && p_->ck && p_->ck->has("vision.patch_embed.proj.weight") && !p_->image_start.empty(); }

void DsCpuEngine::encode_image(const uint16_t* patches, int n_vit_h, int n_vit_w, std::vector<float>& out) {
  p_->encode_image(patches, n_vit_h, n_vit_w, out, "x");
}

void DsCpuEngine::Impl::ensure_vision() {
  if (vision) return;
  HIVE_CHECK(pub->has_vision(), "ds cpu: the checkpoint has no vision encoder (or vision is off)");
  const double t0 = hive::mono_ms();
  auto v = std::make_unique<VisionW>();
  Checkpoint& K = *ck;
  {  // patch embedding [vdim, 3·p·p = 588]: K padded with zero columns to a multiple of 8 (the matmul's granule)
    const TensorInfo& t = K.get("vision.patch_embed.proj.weight");
    const int N = (int)t.shape[0], Kp = (int)t.shape[1], Kq = (Kp + 7) / 8 * 8;
    std::vector<uint16_t> w((size_t)N * Kq, 0);
    for (int n = 0; n < N; ++n) memcpy(&w[(size_t)n * Kq], t.data + (size_t)n * Kp * 2, (size_t)Kp * 2);
    v->patch = ::hive::cpu::make_mat(::hive::cpu::WType::BF16, N, Kq, w.data());
  }
  v->patch_b = load_vec(K, "vision.patch_embed.proj.bias");
  v->blocks.resize(c.vision_layers);
  for (int i = 0; i < c.vision_layers; ++i) {
    const std::string p = "vision.blocks." + std::to_string(i);
    VisionW::Block& b = v->blocks[i];
    b.qkv = load_mat(K, p + ".attn.wqkv"); b.bqkv = load_vec(K, p + ".attn.wqkv.bias");
    b.wo = load_mat(K, p + ".attn.wo"); b.bo = load_vec(K, p + ".attn.wo.bias");
    b.w1 = load_mat(K, p + ".mlp.w1"); b.w2 = load_mat(K, p + ".mlp.w2");
    b.n1 = load_vec(K, p + ".norm1.weight"); b.n2 = load_vec(K, p + ".norm2.weight");
  }
  v->norm = load_vec(K, "vision.norm.weight");
  v->a1 = load_mat(K, "aligner.w1"); v->a1_b = load_vec(K, "aligner.w1.bias");
  v->a2 = load_mat(K, "aligner.w2"); v->a2_b = load_vec(K, "aligner.w2.bias");
  vision = std::move(v);
  fprintf(stderr, "[ds-cpu] vision encoder loaded (%.1fs)\n", (hive::mono_ms() - t0) / 1000.0);
}

namespace {
// y = bf16(x · Wᵀ + b)
void linb(const Mat& W, const float* x, int M, int ldx, const std::vector<float>* b, float* y, int ldy) {
  ::hive::cpu::matmul(W, x, M, ldx, y, ldy);
  par_for((size_t)M, 16, [&](size_t a, size_t e) {
    for (size_t m = a; m < e; ++m) {
      float* r = y + m * ldy;
      if (b) for (int i = 0; i < W.N; ++i) r[i] = bf16r(r[i] + (*b)[i]);
      else for (int i = 0; i < W.N; ++i) r[i] = bf16r(r[i]);
    }
  });
}
inline float hsum8(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo)); lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}
// bidirectional attention of one head over n tokens: q/k/v [n, hd] (head-contiguous copies), out [n, hd] (bf16), exact fp32 softmax
void vit_head_attn(const float* q, const float* kT /* [hd, n] */, const float* v, int n, int hd, float scale, float* out, int ldo, int r0, int r1) {
  std::vector<float> s((size_t)n);
  for (int r = r0; r < r1; ++r) {
    const float* qr = q + (size_t)r * hd;
    int j = 0;
    for (; j + 8 <= n; j += 8) {
      __m256 a = _mm256_setzero_ps();
      for (int d = 0; d < hd; ++d) a = _mm256_fmadd_ps(_mm256_set1_ps(qr[d]), _mm256_loadu_ps(kT + (size_t)d * n + j), a);
      _mm256_storeu_ps(s.data() + j, _mm256_mul_ps(a, _mm256_set1_ps(scale)));
    }
    for (; j < n; ++j) { float a = 0.f; for (int d = 0; d < hd; ++d) a += qr[d] * kT[(size_t)d * n + j]; s[j] = a * scale; }
    float mx = -INFINITY;
    for (int t = 0; t < n; ++t) mx = std::max(mx, s[t]);
    float sum = 0.f;
    for (int t = 0; t < n; ++t) { s[t] = std::exp(s[t] - mx); sum += s[t]; }
    __m256 acc[8];
    for (int i = 0; i < hd / 8; ++i) acc[i] = _mm256_setzero_ps();
    for (int t = 0; t < n; ++t) {
      const __m256 p = _mm256_set1_ps(s[t]);
      const float* vr = v + (size_t)t * hd;
      for (int i = 0; i < hd / 8; ++i) acc[i] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vr + 8 * i), acc[i]);
    }
    float* o = out + (size_t)r * ldo;
    alignas(32) float tmp[8];
    for (int i = 0; i < hd / 8; ++i) {
      _mm256_store_ps(tmp, acc[i]);
      for (int z = 0; z < 8; ++z) o[8 * i + z] = bf16r(tmp[z] / sum);
    }
  }
  (void)hsum8;
}
}  // namespace

void DsCpuEngine::Impl::encode_image(const uint16_t* patches, int nh, int nw, std::vector<float>& out, const std::string& tag) {
  ensure_vision();
  const double t0 = hive::mono_ms();
  const VisionW& V = *vision;
  const int n = nh * nw, vd = c.vision_dim, nH = c.vision_heads, hd = vd / nH, P = 3 * c.vision_patch * c.vision_patch, I = c.vision_inter;
  const int rdim = hd / 2;  // rotary frequencies: rdim / 2 per axis
  auto hook = [&](const std::string& name, const std::vector<float>& a) { if (pub->vision_hook) pub->vision_hook(name.c_str(), a.data(), a.size()); };
  const int Pq = V.patch.K;   // padded patch width
  std::vector<float> pin((size_t)n * Pq, 0.f);
  for (int p = 0; p < n; ++p) for (int i = 0; i < P; ++i) pin[(size_t)p * Pq + i] = b2f(patches[(size_t)p * P + i]);
  std::vector<float> x((size_t)n * vd), xn((size_t)n * vd), qkv((size_t)n * 3 * vd), att((size_t)n * vd), o((size_t)n * vd), h1((size_t)n * 2 * I),
      mlp((size_t)n * I);
  linb(V.patch, pin.data(), n, Pq, &V.patch_b, x.data(), vd);
  hook("vit_" + tag + "_patch", x);
  // 2D RoPE tables: freqs[p] = [hpos · inv_freq (rdim/2), wpos · inv_freq (rdim/2)]
  std::vector<float> inv(rdim / 2), cs((size_t)n * rdim), sn((size_t)n * rdim);
  for (int i = 0; i < rdim / 2; ++i) inv[i] = 1.0f / std::pow(c.vision_rope_theta, (float)(2 * i) / (float)rdim);
  for (int p = 0; p < n; ++p)
    for (int i = 0; i < rdim; ++i) {
      const float pos = i < rdim / 2 ? (float)(p / nw) : (float)(p % nw);
      const float a = pos * inv[i % (rdim / 2)];
      cs[(size_t)p * rdim + i] = (float)std::cos((double)a);
      sn[(size_t)p * rdim + i] = (float)std::sin((double)a);
    }
  const float scale = 1.f / std::sqrt((float)hd);
  std::vector<float> qh((size_t)nH * n * hd), kT((size_t)nH * hd * n), vh((size_t)nH * n * hd);
  for (int bi = 0; bi < (int)V.blocks.size(); ++bi) {
    const VisionW::Block& B = V.blocks[bi];
    rmsnorm_rows(x.data(), vd, B.n1.data(), n, vd, 1e-6f, xn.data(), vd);
    linb(B.qkv, xn.data(), n, vd, &B.bqkv, qkv.data(), 3 * vd);
    // rotary on q and k (halves: x1 = dims [0, hd/2), x2 = [hd/2, hd)), head-major copies (k transposed)
    par_for((size_t)n, 32, [&](size_t a, size_t b) {
      for (size_t p = a; p < b; ++p)
        for (int h = 0; h < nH; ++h)
          for (int w = 0; w < 2; ++w) {
            const float* src = qkv.data() + p * 3 * vd + (size_t)w * vd + (size_t)h * hd;
            float tmp[256];
            for (int i = 0; i < rdim; ++i) {
              const float x1 = src[i], x2 = src[rdim + i], cv = cs[p * rdim + i], sv = sn[p * rdim + i];
              tmp[i] = bf16r(x1 * cv - x2 * sv);
              tmp[rdim + i] = bf16r(x2 * cv + x1 * sv);
            }
            if (w == 0) memcpy(qh.data() + ((size_t)h * n + p) * hd, tmp, (size_t)hd * 4);
            else for (int d = 0; d < hd; ++d) kT[((size_t)h * hd + d) * n + p] = tmp[d];
          }
        for (size_t p = a; p < b; ++p)
          for (int h = 0; h < nH; ++h) memcpy(vh.data() + ((size_t)h * n + p) * hd, qkv.data() + p * 3 * vd + 2 * (size_t)vd + (size_t)h * hd, (size_t)hd * 4);
    });
    const int QB = 32, nqb = (n + QB - 1) / QB;
    Pool::get().run(nH * nqb, [&](int t) {
      const int h = t / nqb, qb = t % nqb;
      vit_head_attn(qh.data() + (size_t)h * n * hd, kT.data() + (size_t)h * hd * n, vh.data() + (size_t)h * n * hd, n, hd, scale,
                    att.data() + (size_t)h * hd, vd, qb * QB, std::min(n, (qb + 1) * QB));
    });
    linb(B.wo, att.data(), n, vd, &B.bo, o.data(), vd);
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf16r(x[i] + o[i]);
    rmsnorm_rows(x.data(), vd, B.n2.data(), n, vd, 1e-6f, xn.data(), vd);
    linb(B.w1, xn.data(), n, vd, nullptr, h1.data(), 2 * I);
    par_for((size_t)n, 16, [&](size_t a, size_t b) {
      for (size_t p = a; p < b; ++p)
        for (int i = 0; i < I; ++i) {
          const float g = h1[p * 2 * I + i], u = h1[p * 2 * I + I + i];
          mlp[p * I + i] = bf16r(bf16r(g / (1.f + std::exp(-g))) * u);
        }
    });
    linb(B.w2, mlp.data(), n, I, nullptr, o.data(), vd);
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf16r(x[i] + o[i]);
    if (bi == 0 || bi == 1 || bi == 7 || bi == 15 || bi == 31) hook("vit_" + tag + "_blk" + std::to_string(bi), x);
  }
  rmsnorm_rows(x.data(), vd, V.norm.data(), n, vd, 1e-6f, xn.data(), vd);
  hook("vit_" + tag + "_norm", xn);
  // Aligner: unfold r×r blocks (zero padded): feature (ch · r + ki) · r + kj
  const int r = c.vision_downsample, Lh = (nh + r - 1) / r, Lw = (nw + r - 1) / r, L = Lh * Lw, F = vd * r * r;
  std::vector<float> u((size_t)L * F, 0.f);
  for (int bh = 0; bh < Lh; ++bh)
    for (int bw = 0; bw < Lw; ++bw)
      for (int ki = 0; ki < r; ++ki)
        for (int kj = 0; kj < r; ++kj) {
          const int ph = bh * r + ki, pw = bw * r + kj;
          if (ph >= nh || pw >= nw) continue;
          const float* src = xn.data() + ((size_t)ph * nw + pw) * vd;
          float* dst = u.data() + (size_t)(bh * Lw + bw) * F + ki * r + kj;
          for (int ch = 0; ch < vd; ++ch) dst[(size_t)ch * r * r] = src[ch];
        }
  std::vector<float> a1((size_t)L * c.dim);
  linb(V.a1, u.data(), L, F, &V.a1_b, a1.data(), c.dim);
  for (float& z : a1) z = bf16r(0.5f * z * (1.f + std::erf(z * (float)M_SQRT1_2)));
  out.resize((size_t)L * c.dim);
  linb(V.a2, a1.data(), L, c.dim, &V.a2_b, out.data(), c.dim);
  hook("vit_" + tag + "_out", out);
  hook("image_emb_" + tag, out);
  fprintf(stderr, "[ds-cpu] image %dx%d patches → %d rows in %.1fs\n", nh, nw, L, (hive::mono_ms() - t0) / 1000.0);
}

}  // namespace hive::ds::cpu
