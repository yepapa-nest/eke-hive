// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B on the CPU — loading and the layer forward (hive/q27/q27_cpu.h). The arithmetic and every bf16 rounding point follow
//   oracle/q27_oracle.py, so the residual streams can be compared with it layer by layer. Structure adapted from the Flash-Next CPU engine
//   (src/qwen/cpu/qwen_cpu_engine.cpp): the Gated DeltaNet recurrence, the attention row prep and the verify snapshots are the same code
//   with 27B's shapes (4 KV heads, no QSA indexer, a SiLU-gated output norm) and the dense pre-norm residual instead of hyper-connections.
#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "hive/clock.h"
#include "q27_cpu_internal.h"

namespace hive::q27::cpu {

using ::hive::cpu::Pool;
using ::hive::cpu::SType;
using ::hive::cpu::WType;

namespace {
inline float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
inline uint16_t f2b_exact(float x) { uint32_t u; memcpy(&u, &x, 4); return (uint16_t)(u >> 16); }  // x already a bf16 value
}  // namespace

// ---- helpers ------------------------------------------------------------------------------------------------------------------------------
void par_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f) {
  if (n == 0) return;
  const size_t nb = (n + grain - 1) / grain;
  if (nb <= 1) { f(0, n); return; }
  Pool::get().run((int)nb, [&](int b) { const size_t a = (size_t)b * grain; f(a, std::min(n, a + grain)); });
}

void round_rows(float* Y, int M, int N, int ldy) {
  par_for((size_t)M, std::max<size_t>(1, 262144 / std::max(1, N)), [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) simd::round_n(Y + m * ldy, (size_t)N);
  });
}

void lin(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy, bool round) {
  ::hive::cpu::matmul(W, X, M, ldx, Y, ldy);
  if (round) round_rows(Y, M, W.r1 - W.r0, ldy);
}

void norm1p(const float* x, int ldx, const float* w, int M, int K, int group, float eps, float* out, int ldo) {
  const int G = K / group;
  par_for((size_t)M * G, std::max<size_t>(1, 131072 / group), [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) {
      const size_t m = i / G, g = i % G;
      simd::norm1p_v(x + m * ldx + g * group, w + g * group, group, eps, out + m * ldo + g * group);
    }
  });
}

std::vector<float> load_vec(Checkpoint& ck, const std::string& name) {
  const TensorInfo& t = ck.get(name);
  std::vector<float> v((size_t)t.numel());
  if (t.dtype == "BF16") { const uint16_t* p = (const uint16_t*)t.data; for (size_t i = 0; i < v.size(); ++i) v[i] = b2f(p[i]); }
  else if (t.dtype == "F32") memcpy(v.data(), t.data, v.size() * 4);
  else throw std::runtime_error("q27 cpu: unexpected dtype " + t.dtype + " for " + name);
  return v;
}

// BF16 ("<base>.weight") or FP8 ("<base>.weight" e4m3 + "<base>.weight_scale" bf16 [N, 1]) linears, rows stacked; all of one format
Mat load_mat(Checkpoint& ck, const std::vector<std::string>& bases) {
  int N = 0, K = -1;
  std::string fmt;
  std::vector<const TensorInfo*> ws, ss;
  for (const auto& b : bases) {
    const TensorInfo& t = ck.get(b + ".weight");
    if (t.shape.size() != 2) throw std::runtime_error("q27 cpu: expected a matrix: " + b);
    if (fmt.empty()) fmt = t.dtype;
    if (t.dtype != fmt || (t.dtype != "BF16" && t.dtype != "F8_E4M3")) throw std::runtime_error("q27 cpu: unexpected / mixed format at " + b + " (" + t.dtype + ")");
    if (K >= 0 && K != (int)t.shape[1]) throw std::runtime_error("q27 cpu: stacked widths differ: " + b);
    K = (int)t.shape[1]; N += (int)t.shape[0];
    ws.push_back(&t);
    if (fmt == "F8_E4M3") {
      const TensorInfo& s = ck.get(b + ".weight_scale");
      if (s.dtype != "BF16" || s.numel() != t.shape[0]) throw std::runtime_error("q27 cpu: FP8 needs one bf16 scale per row: " + b);
      ss.push_back(&s);
    }
  }
  const bool fp8 = fmt == "F8_E4M3";
  const void* w = ws[0]->data;
  const void* s = fp8 ? ss[0]->data : nullptr;
  std::vector<uint8_t> tw, ts;
  if (ws.size() > 1) {
    for (auto* t : ws) tw.insert(tw.end(), t->data, t->data + t->nbytes);
    for (auto* t : ss) ts.insert(ts.end(), t->data, t->data + t->nbytes);
    w = tw.data(); s = fp8 ? ts.data() : nullptr;
  }
  if (!fp8) return ::hive::cpu::make_mat(WType::BF16, N, K, w);
  return ::hive::cpu::make_mat(WType::FP8, N, K, w, s, SType::BF16, 1.f, 1, K);   // one block per row (per-channel scale)
}

// compressed-tensors NVFP4: weight_packed u8 [N, K/2] · weight_scale e4m3 [N, K/16] · weight_global_scale g (W = e2m1 · s / g)
Mat load_nvfp4(Checkpoint& ck, const std::string& b) {
  const TensorInfo& w = ck.get(b + ".weight_packed");
  const TensorInfo& s = ck.get(b + ".weight_scale");
  const TensorInfo& g = ck.get(b + ".weight_global_scale");
  if (w.dtype != "U8" || s.dtype != "F8_E4M3" || g.dtype != "F32" || w.shape.size() != 2 || s.shape[1] * 8 != w.shape[1])
    throw std::runtime_error("q27 cpu: unexpected NVFP4 layout at " + b);
  float gs; memcpy(&gs, g.data, 4);
  return ::hive::cpu::make_mat(WType::NVFP4, (int)w.shape[0], (int)w.shape[1] * 2, w.data, s.data, SType::E4M3, 1.f / gs);
}

// ---- loading -------------------------------------------------------------------------------------------------------------------------------
namespace {
const std::string P = "model.language_model.";

void load_mlp(Checkpoint& ck, const std::string& p, MlpW& w) {
  auto one = [&](const char* n) { const std::string b = p + n; return ck.has(b + ".weight_packed") ? load_nvfp4(ck, b) : load_mat(ck, {b}); };
  w.gate = one("gate_proj"); w.up = one("up_proj"); w.down = one("down_proj");
}
void load_attn(Checkpoint& ck, const std::string& a, AttW& w) {
  w.qkv = load_mat(ck, {a + "q_proj", a + "k_proj", a + "v_proj"});
  w.out = load_mat(ck, {a + "o_proj"});
  w.q_norm = load_vec(ck, a + "q_norm.weight"); w.k_norm = load_vec(ck, a + "k_norm.weight");
}
}  // namespace

void Q27CpuEngine::Impl::load() {
  const double t0 = hive::mono_ms();
  c = Q27Config::load(ckpt_dir);
  ck = std::make_unique<Checkpoint>(ckpt_dir);
  if (const char* v = getenv("HIVE_Q27_CPU_CHUNK"); v && atoi(v) > 0) max_chunk = atoi(v);
  if (!opt.layers.empty() && (int)opt.layers.size() != c.n_layers) throw std::runtime_error("q27 cpu: layer mask size");
  layers.resize(c.n_layers);
  lin_index.assign(c.n_layers, -1); attn_index.assign(c.n_layers, -1);
  int nl = 0, na = 0;
  for (int l = 0; l < c.n_layers; ++l) (c.is_attn[l] ? attn_index[l] = na++ : lin_index[l] = nl++);
  int loaded = 0;
  for (int l = 0; l < c.n_layers; ++l) {
    LayerW& L = layers[l];
    L.attn = c.is_attn[l];
    L.on = opt.layers.empty() || opt.layers[l];
    if (!L.on) continue;
    const std::string p = P + "layers." + std::to_string(l) + ".";
    L.ln1 = load_vec(*ck, p + "input_layernorm.weight"); L.ln2 = load_vec(*ck, p + "post_attention_layernorm.weight");
    if (L.attn) load_attn(*ck, p + "self_attn.", L.att);
    else {
      const std::string a = p + "linear_attn.";
      L.lin.qkvz = load_mat(*ck, {a + "in_proj_qkv", a + "in_proj_z"});
      L.lin.ba = load_mat(*ck, {a + "in_proj_b", a + "in_proj_a"});
      L.lin.out = load_mat(*ck, {a + "out_proj"});
      L.lin.conv_w = load_vec(*ck, a + "conv1d.weight");
      L.lin.A_log = load_vec(*ck, a + "A_log"); L.lin.dt_bias = load_vec(*ck, a + "dt_bias");
      L.lin.norm = load_vec(*ck, a + "norm.weight");
      if ((int)L.lin.qkvz.N != c.lin_conv_ch() + c.lin_v_width() || L.lin.ba.N != 2 * c.lin_v_heads) throw std::runtime_error("q27 cpu: linear attention shape");
    }
    load_mlp(*ck, p + "mlp.", L.mlp);
    for (const Mat* m : {&L.lin.qkvz, &L.lin.ba, &L.lin.out, &L.att.qkv, &L.att.out, &L.mlp.gate, &L.mlp.up, &L.mlp.down}) wbytes += m->bytes;
    if (++loaded % 16 == 0) fprintf(stderr, "[q27-cpu] %d layers loaded (%.1f s)\n", loaded, (hive::mono_ms() - t0) / 1000);
  }
  if (opt.head && opt.lm_head) {
    lm_head = load_mat(*ck, {"lm_head"});
    wbytes += lm_head.bytes;
  }
  if (opt.head) {
    final_norm = load_vec(*ck, P + "norm.weight");
    const TensorInfo& e = ck->get(P + "embed_tokens.weight");
    if (e.dtype != "BF16") throw std::runtime_error("q27 cpu: embeddings must be BF16");
    embed.resize((size_t)e.numel());
    memcpy(embed.data(), e.data, e.nbytes);
  }
  if (opt.load_mtp && opt.head && opt.lm_head && c.n_mtp > 0 && ck->has("mtp.fc.weight")) {
    mtp = std::make_unique<MtpW>();
    mtp->fc = load_mat(*ck, {"mtp.fc"});
    if (mtp->fc.K != 2 * c.hidden) throw std::runtime_error("q27 cpu: mtp.fc must be [hidden, 2·hidden]");
    mtp->pre_emb_norm = load_vec(*ck, "mtp.pre_fc_norm_embedding.weight");
    mtp->pre_hidden_norm = load_vec(*ck, "mtp.pre_fc_norm_hidden.weight");
    mtp->norm = load_vec(*ck, "mtp.norm.weight");
    LayerW& L = mtp->layer;
    L.attn = true; L.on = true;
    L.ln1 = load_vec(*ck, "mtp.layers.0.input_layernorm.weight"); L.ln2 = load_vec(*ck, "mtp.layers.0.post_attention_layernorm.weight");
    load_attn(*ck, "mtp.layers.0.self_attn.", L.att);
    load_mlp(*ck, "mtp.layers.0.mlp.", L.mlp);
    for (const Mat* m : {&mtp->fc, &L.att.qkv, &L.att.out, &L.mlp.gate, &L.mlp.up, &L.mlp.down}) wbytes += m->bytes;
    mtp_post = getenv("HIVE_Q27_MTP_INPUT") && strcmp(getenv("HIVE_Q27_MTP_INPUT"), "post") == 0;
    mtp_k = getenv("HIVE_Q27_MTP_K") ? std::max(0, std::min(7, atoi(getenv("HIVE_Q27_MTP_K")))) : 3;
  }
  const int half = c.rotary_dim / 2;
  rope_inv.resize(half); rope_axis.assign(half, 0);
  for (int i = 0; i < half; ++i) {
    rope_inv[i] = std::pow((double)c.rope_theta, -2.0 * i / c.rotary_dim);
    if (c.mrope_interleaved) rope_axis[i] = (i % 3 == 1 && i < 3 * c.mrope_section[1]) ? 1 : (i % 3 == 2 && i < 3 * c.mrope_section[2]) ? 2 : 0;
    else rope_axis[i] = i < c.mrope_section[0] ? 0 : i < c.mrope_section[0] + c.mrope_section[1] ? 1 : 2;
  }
  vis_avail = c.vision_layers > 0 && ck->has("model.visual.patch_embed.proj.weight");
  ck->release_all();   // every weight was copied
  fprintf(stderr, "[q27-cpu] %d layer(s) loaded in %.1f s · %.2f GiB · %d threads on %d NUMA node(s) · MTP k %d · chunk %d\n", loaded,
          (hive::mono_ms() - t0) / 1000, wbytes / 1073741824.0, Pool::get().threads(), Pool::get().nodes(), mtp_k, max_chunk);
}

// ---- building blocks -------------------------------------------------------------------------------------------------------------------------
void Q27CpuEngine::Impl::rope_row(float* v, int n_heads, int head_dim, const int32_t* coord) const {
  const int half = (int)rope_inv.size();
  float cs[64], sn[64];
  for (int i = 0; i < half; ++i) {
    const double a = (double)coord[rope_axis[i]] * rope_inv[i];
    cs[i] = (float)std::cos(a); sn[i] = (float)std::sin(a);
  }
  for (int h = 0; h < n_heads; ++h) {
    float* x = v + (size_t)h * head_dim;
    for (int i = 0; i < half; ++i) {
      const float a = x[i], b = x[i + half];
      x[i] = bf16r(a * cs[i] - b * sn[i]);
      x[i + half] = bf16r(b * cs[i] + a * sn[i]);
    }
  }
}

void Q27CpuEngine::Impl::embed_rows(const int32_t* ids, int n, float* out) const {
  const int H = c.hidden;
  par_for((size_t)n, 16, [&](size_t a, size_t b) {
    for (size_t r = a; r < b; ++r) ::hive::cpu::bf16_to_f32_n(embed.data() + (size_t)ids[r] * H, (size_t)H, out + r * H);
  });
}

void Q27CpuEngine::Impl::head(const float* h, int M, const std::vector<float>& norm_w, float* logits) {
  thread_local std::vector<float> xn_tl;
  auto& xn = xn_tl;
  xn.resize((size_t)M * c.hidden);
  norm1p(h, c.hidden, norm_w.data(), M, c.hidden, c.hidden, c.rms_eps, xn.data(), c.hidden);
  ::hive::cpu::matmul(lm_head, xn.data(), M, c.hidden, logits, c.vocab);
}

// ---- MLP -------------------------------------------------------------------------------------------------------------------------------------
void Q27CpuEngine::Impl::mlp(const MlpW& w, int M, const float* x, float* out) {
  const int H = c.hidden, I = w.gate.N;
  thread_local std::vector<float> g_tl, u_tl;
  auto& g = g_tl; auto& u = u_tl;   // the lambdas run on worker threads: they must use the caller's buffers
  g.resize((size_t)M * I); u.resize((size_t)M * I);
  MatJob jobs[2] = {MatJob{MatView{&w.gate, 0, I}, g.data(), I}, MatJob{MatView{&w.up, 0, I}, u.data(), I}};
  ::hive::cpu::matmul_multi(jobs, 2, x, M, H);
  par_for((size_t)M * 8, 1, [&](size_t a, size_t b) {   // act = bf16(bf16(silu(bf16 g)) · bf16 u), in 8 slices per row
    for (size_t i = a; i < b; ++i) {
      const size_t m = i / 8, q = i % 8, s0 = I * q / 8, s1 = I * (q + 1) / 8;
      float* gr = g.data() + m * I;
      float* ur = u.data() + m * I;
      simd::round_n(gr + s0, s1 - s0);
      simd::round_n(ur + s0, s1 - s0);
      simd::silu_mul_v(gr + s0, ur + s0, (int)(s1 - s0), gr + s0);
    }
  });
  lin(w.down, g.data(), M, I, out, H);
}

// ---- Gated DeltaNet ------------------------------------------------------------------------------------------------------------------------
void Q27CpuEngine::Impl::lin_layer(const LinW& w, int li, const std::vector<RowInfo>& rows, const float* x, float* out) {
  const int M = (int)rows.size(), H = c.hidden;
  const int KH = c.lin_k_heads, VH = c.lin_v_heads, D = c.lin_k_dim, KW = KH * D, VW = VH * D, C = 2 * KW + VW;
  const int PW = C + VW + 2 * VH, ZO = C, BO = C + VW, AO = BO + VH;
  thread_local std::vector<float> proj_tl, y_tl, qn_tl, kn_tl, gg_tl, beta_tl, o_tl;
  auto& proj = proj_tl; auto& y = y_tl; auto& qn = qn_tl; auto& kn = kn_tl; auto& gg = gg_tl; auto& beta = beta_tl; auto& o = o_tl;
  proj.resize((size_t)M * PW); y.resize((size_t)M * C); qn.resize((size_t)M * KW); kn.resize((size_t)M * KW);
  gg.resize((size_t)M * VH); beta.resize((size_t)M * VH); o.resize((size_t)M * VW);
  {
    MatJob jobs[2] = {MatJob{MatView{&w.qkvz, 0, w.qkvz.N}, proj.data(), PW}, MatJob{MatView{&w.ba, 0, w.ba.N}, proj.data() + C + VW, PW}};
    ::hive::cpu::matmul_multi(jobs, 2, x, M, H);
    round_rows(proj.data(), M, PW, PW);
  }
  const float* cw = w.conv_w.data();
  const float qs = 1.f / std::sqrt((float)D);
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s && rows[r + n].pos == rows[r].pos + n) ++n;
    int part = -1;
    if (verify) for (size_t p = 0; p < vparts.size(); ++p) if (vparts[p].r0 == r) part = (int)p;
    std::vector<float>& hist = s.conv[li];   // [3, C] oldest first
    par_for((size_t)C, 512, [&](size_t a, size_t b) {
      for (size_t ch = a; ch < b; ++ch) {
        const float* w4 = cw + ch * 4;
        auto X = [&](int t) { return t >= 0 ? proj[(size_t)(r + t) * PW + ch] : hist[(size_t)(3 + t) * C + ch]; };
        for (int t = 0; t < n; ++t) {
          float v = w4[0] * X(t - 3);
          v += w4[1] * X(t - 2); v += w4[2] * X(t - 1); v += w4[3] * X(t);
          y[(size_t)(r + t) * C + ch] = bf16r(siluf(v));
          if (part >= 0 && t + 1 < n)
            for (int k = 0; k < 3; ++k) snaps[part][t].conv[li][(size_t)k * C + ch] = X(t - 2 + k);
        }
        float nv[3];
        for (int k = 0; k < 3; ++k) nv[k] = X(n - 3 + k);
        for (int k = 0; k < 3; ++k) hist[(size_t)k * C + ch] = nv[k];
      }
    });
    par_for((size_t)n, 8, [&](size_t ta, size_t tb) {
      for (int t = (int)ta; t < (int)tb; ++t) {
        const float* yr = y.data() + (size_t)(r + t) * C;
        for (int kh = 0; kh < KH; ++kh) {
          const float nq = std::sqrt(simd::sumsq(yr + kh * D, D) + 1e-6f), nk = std::sqrt(simd::sumsq(yr + KW + kh * D, D) + 1e-6f);
          for (int d = 0; d < D; ++d) {
            qn[(size_t)(r + t) * KW + kh * D + d] = yr[kh * D + d] / nq * qs;
            kn[(size_t)(r + t) * KW + kh * D + d] = yr[KW + kh * D + d] / nk;
          }
        }
        const float* pr = proj.data() + (size_t)(r + t) * PW;
        for (int vh = 0; vh < VH; ++vh) {
          float sp = pr[AO + vh] + w.dt_bias[vh];
          sp = sp > 20.f ? sp : std::log1p(std::exp(sp));
          gg[(size_t)(r + t) * VH + vh] = std::exp(-std::exp(w.A_log[vh]) * sp);
          beta[(size_t)(r + t) * VH + vh] = bf16r(sigm(pr[BO + vh]));
        }
      }
    });
    // recurrence: S [k][v] fp32 per value head, one task per value head (Flash-Next measured this split as the fastest, 2026-10-09)
    const int rep = VH / KH;
    Pool::get().run(VH, [&](int vh) {
      float* S = s.state[li].data() + (size_t)vh * D * D;
      const int kh = vh / rep;
      alignas(32) float kv[128], dl[128], oo[128];
      for (int t = 0; t < n; ++t) {
        const size_t row = (size_t)(r + t);
        const float e = gg[row * VH + vh], bt = beta[row * VH + vh];
        const float* kk = kn.data() + row * KW + kh * D;
        const float* qq = qn.data() + row * KW + kh * D;
        const float* vv = y.data() + row * C + 2 * KW + (size_t)vh * D;
        const __m256 ev = _mm256_set1_ps(e);
        for (int j = 0; j < D; j += 8) _mm256_store_ps(kv + j, _mm256_setzero_ps());
        for (int k = 0; k < D; ++k) {
          float* Sr = S + (size_t)k * D;
          const __m256 kb = _mm256_set1_ps(kk[k]);
          for (int j = 0; j < D; j += 8) {
            const __m256 sv = _mm256_mul_ps(_mm256_loadu_ps(Sr + j), ev);
            _mm256_storeu_ps(Sr + j, sv);
            _mm256_store_ps(kv + j, _mm256_fmadd_ps(sv, kb, _mm256_load_ps(kv + j)));
          }
        }
        for (int j = 0; j < D; ++j) { dl[j] = (vv[j] - kv[j]) * bt; oo[j] = 0.f; }
        for (int k = 0; k < D; ++k) {
          float* Sr = S + (size_t)k * D;
          const __m256 kb = _mm256_set1_ps(kk[k]), qb = _mm256_set1_ps(qq[k]);
          for (int j = 0; j < D; j += 8) {
            const __m256 sv = _mm256_fmadd_ps(kb, _mm256_load_ps(dl + j), _mm256_loadu_ps(Sr + j));
            _mm256_storeu_ps(Sr + j, sv);
            _mm256_store_ps(oo + j, _mm256_fmadd_ps(sv, qb, _mm256_load_ps(oo + j)));
          }
        }
        float* orow = o.data() + row * VW + (size_t)vh * D;
        for (int j = 0; j < D; ++j) orow[j] = bf16r(oo[j]);
        if (part >= 0 && t + 1 < n) memcpy(snaps[part][t].state[li].data() + (size_t)vh * D * D, S, (size_t)D * D * 4);
      }
    });
    par_for((size_t)n * VH, 64, [&](size_t ia, size_t ib) {
      for (size_t i = ia; i < ib; ++i) {
        const size_t row = (size_t)r + i / VH, vh = i % VH;
        simd::gated_norm_silu_v(o.data() + row * VW + vh * D, proj.data() + row * PW + ZO + vh * D, w.norm.data(), D, c.rms_eps);
      }
    });
    r += n;
  }
  lin(w.out, o.data(), M, VW, out, H);
}

// ---- full attention ------------------------------------------------------------------------------------------------------------------------
void Q27CpuEngine::Impl::attn_layer(const AttW& w, int ai, bool mtp_layer, const std::vector<RowInfo>& rows, const int32_t* coords, const float* x,
                                    float* out, bool kv_only) {
  const int M = (int)rows.size(), H = c.hidden, NH = c.n_heads, NKV = c.n_kv_heads, D = c.head_dim;
  const int QW = 2 * NH * D, KO = QW, VO = KO + NKV * D, PW = VO + NKV * D, KVW = NKV * D;
  thread_local std::vector<float> proj_tl, q_tl, att_tl, o_tl;
  auto& proj = proj_tl; auto& q = q_tl; auto& att = att_tl; auto& o = o_tl;
  proj.resize((size_t)M * PW); q.resize((size_t)M * NH * D);
  lin(w.qkv, x, M, H, proj.data(), PW);
  par_for((size_t)M, 1, [&](size_t a, size_t b) {
    for (size_t r = a; r < b; ++r) {
      CpuSeq& s = *rows[r].s;
      CpuSeq::Attn& ca = mtp_layer ? s.mtp : s.attn[ai];
      const int64_t p = rows[r].pos;
      const float* pr = proj.data() + r * PW;
      const int32_t* cd = coords + r * 3;
      float kbuf[8 * 256];
      norm1p(pr + KO, D, w.k_norm.data(), NKV, D, D, c.rms_eps, kbuf, D);
      rope_row(kbuf, NKV, D, cd);
      for (int i = 0; i < KVW; ++i) { ca.k[(size_t)p * KVW + i] = f2b_exact(kbuf[i]); ca.v[(size_t)p * KVW + i] = f2b_exact(pr[VO + i]); }
      if (kv_only) continue;
      float* qr = q.data() + r * NH * D;
      for (int h = 0; h < NH; ++h) norm1p(pr + (size_t)h * 2 * D, 2 * D, w.q_norm.data(), 1, D, D, c.rms_eps, qr + (size_t)h * D, D);
      rope_row(qr, NH, D, cd);
    }
  });
  if (kv_only) return;
  att.resize((size_t)M * NH * D);
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s) ++n;
    CpuSeq::Attn& ca = mtp_layer ? s.mtp : s.attn[ai];
    std::vector<int32_t> ce(n);
    for (int t = 0; t < n; ++t) ce[t] = (int32_t)rows[r + t].pos + 1;
    ::hive::cpu::attention(q.data() + (size_t)r * NH * D, n, NH, NKV, D, ca.k.data(), ca.v.data(), nullptr, 0, ce.data(), 1.f / std::sqrt((float)D),
                           att.data() + (size_t)r * NH * D);
    r += n;
  }
  o.resize((size_t)M * NH * D);
  par_for((size_t)M * NH, 64, [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) {
      const size_t r = i / NH, h = i % NH;
      simd::gate_v(att.data() + i * D, proj.data() + r * PW + h * 2 * D + D, D, o.data() + i * D);
    }
  });
  lin(w.out, o.data(), M, NH * D, out, H);
}

// ---- one decoder layer: h += mixer(n1(h)); h += mlp(n2(h)) ---------------------------------------------------------------------------------
void Q27CpuEngine::Impl::layer(int l, const LayerW& L, bool mtp_layer, const std::vector<RowInfo>& rows, const int32_t* coords, float* h) {
  const int M = (int)rows.size(), H = c.hidden;
  thread_local std::vector<float> x_tl, y_tl;
  auto& x = x_tl; auto& y = y_tl;
  x.resize((size_t)M * H); y.resize((size_t)M * H);
  double tc = hive::mono_ms();
  auto lap = [&](double& acc) { const double t = hive::mono_ms(); acc += t - tc; tc = t; };
  norm1p(h, H, L.ln1.data(), M, H, H, c.rms_eps, x.data(), H);
  lap(st.ms_norm);
  if (L.attn) attn_layer(L.att, mtp_layer ? 0 : attn_index[l], mtp_layer, rows, coords, x.data(), y.data());
  else lin_layer(L.lin, lin_index[l], rows, x.data(), y.data());
  lap(L.attn ? st.ms_attn : st.ms_lin);
  par_for((size_t)M, 4, [&](size_t a, size_t b) {
    for (size_t r = a; r < b; ++r) simd::add_round_v(h + r * H, y.data() + r * H, H);
  });
  norm1p(h, H, L.ln2.data(), M, H, H, c.rms_eps, x.data(), H);
  lap(st.ms_norm);
  mlp(L.mlp, M, x.data(), y.data());
  lap(st.ms_mlp);
  par_for((size_t)M, 4, [&](size_t a, size_t b) {
    for (size_t r = a; r < b; ++r) simd::add_round_v(h + r * H, y.data() + r * H, H);
  });
  lap(st.ms_norm);
}

// ---- rope coordinates (text: p + delta; image spans: (st, st + i / gw, st + i % gw)) --------------------------------------------------------
void Q27CpuEngine::Impl::set_rope(const std::vector<RowInfo>& rows, bool prefill, std::vector<int32_t>& coords) {
  const int M = (int)rows.size();
  coords.resize((size_t)M * 3);
  for (int r = 0; r < M; ++r) {
    CpuSeq& s = *rows[r].s;
    const int64_t p = rows[r].pos, gi = emb_base + r;
    const EmbedOverride* sp = nullptr;
    if (prefill) for (const EmbedOverride& e : emb_over) if (e.gw > 0 && gi >= e.row && gi < e.row + e.n) { sp = &e; break; }
    int32_t* o = coords.data() + (size_t)r * 3;
    if (!sp) { o[0] = o[1] = o[2] = (int32_t)(p + s.rope_delta); }
    else {
      const int64_t k = gi - sp->row, st0 = (p - k) + s.rope_delta;
      o[0] = (int32_t)st0; o[1] = (int32_t)(st0 + k / sp->gw); o[2] = (int32_t)(st0 + k % sp->gw);
      if (k == sp->n - 1) s.rope_delta += std::max(sp->gh, sp->gw) - sp->n;
    }
    if ((int64_t)s.rope.size() < (p + 1) * 3) s.rope.resize((size_t)(p + 1) * 3);
    memcpy(s.rope.data() + (size_t)p * 3, o, 12);
  }
}

namespace {
void grow_attn(CpuSeq::Attn& a, int64_t tokens, int kvw) {
  if ((int64_t)a.k.size() < tokens * kvw) {
    const size_t n = (size_t)std::max<int64_t>(tokens, (int64_t)a.k.size() / kvw * 3 / 2) * kvw;   // amortized growth
    a.k.resize(n); a.v.resize(n);
  }
}
}  // namespace

void Q27CpuEngine::Impl::forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, float* logits, bool all_logits) {
  const int M = (int)rows.size(), H = c.hidden;
  const double t0 = hive::mono_ms();
  for (const RowInfo& ri : rows) {
    CpuSeq& s = *ri.s;
    if (ri.pos + 1 > s.cap) throw std::runtime_error("q27 cpu: sequence capacity exceeded");
    for (auto& a : s.attn) grow_attn(a, ri.pos + 1, c.kv_width());
    if (mtp) grow_attn(s.mtp, ri.pos + 1 + 8, c.kv_width());
  }
  ++depth;
  struct Depth { int& d; ~Depth() { --d; } } dguard{depth};
  if ((int)h_stack.size() < depth) h_stack.resize(depth);
  std::vector<float>& hv = h_stack[depth - 1];
  hv.resize((size_t)M * H);
  float* h = hv.data();
  std::vector<int32_t> coords;
  set_rope(rows, prefill, coords);
  embed_rows(ids, M, h);
  if (prefill)
    for (const EmbedOverride& o : emb_over) {
      const int64_t a = std::max<int64_t>(o.row, emb_base), b = std::min<int64_t>(o.row + o.n, emb_base + M);
      for (int64_t g = a; g < b; ++g) memcpy(h + (size_t)(g - emb_base) * H, o.emb.data() + (size_t)(g - o.row) * H, (size_t)H * 4);
    }
  for (int l = 0; l < c.n_layers; ++l) {
    if (prefill && l > 0 && layer_inject && *layer_inject) (*layer_inject)(l, h, M);
    layer(l, layers[l], false, rows, coords.data(), h);
    if (layer_hook && *layer_hook) (*layer_hook)(l, h, M);
    if (prefill && depth == 1 && layer_boundary && *layer_boundary && l + 1 < c.n_layers) {
      std::vector<EmbedOverride> parked = std::move(emb_over);  // inner forwards are other sequences' decode steps
      const int64_t base = emb_base;
      emb_over.clear(); emb_base = 0;
      (*layer_boundary)(l);
      emb_over = std::move(parked); emb_base = base;
    }
  }
  if (mtp) fh.assign(h, h + (size_t)M * H);
  const int r0 = all_logits ? 0 : M - 1, nr = all_logits ? M : 1;
  double th = hive::mono_ms();
  if (logits) head(h + (size_t)r0 * H, nr, final_norm, logits);
  st.ms_head += hive::mono_ms() - th;
  th = hive::mono_ms();
  if (mtp && mtp_k > 0) mtp_after_forward(rows, ids, prefill);
  st.ms_mtp += hive::mono_ms() - th;
  for (int r = 0; r < M; ++r) { CpuSeq& s = *rows[r].s; s.pos = rows[r].pos + 1; s.tokens.push_back(ids[r]); }
  if (!prefill) ++st.steps;
  st.ms_total += hive::mono_ms() - t0;
}

void Q27CpuEngine::Impl::alloc_snaps(CpuSeq* const* seqs, const int* Ms, int S) {
  snaps.resize(S);
  for (int p = 0; p < S; ++p) {
    snaps[p].resize(std::max(0, Ms[p] - 1));
    for (auto& sn : snaps[p]) {
      sn.conv.resize(seqs[p]->conv.size()); sn.state.resize(seqs[p]->state.size());
      for (size_t i = 0; i < sn.conv.size(); ++i) { sn.conv[i].resize(seqs[p]->conv[i].size()); sn.state[i].resize(seqs[p]->state[i].size()); }
    }
  }
}

Q27CpuEngine::Impl::~Impl() {
  using ::hive::cpu::free_mat;
  auto layer_free = [](LayerW& L) {
    for (Mat* m : {&L.lin.qkvz, &L.lin.ba, &L.lin.out, &L.att.qkv, &L.att.out, &L.mlp.gate, &L.mlp.up, &L.mlp.down}) free_mat(*m);
  };
  for (LayerW& L : layers) layer_free(L);
  free_mat(lm_head);
  if (mtp) { free_mat(mtp->fc); layer_free(mtp->layer); }
  if (vision) {
    free_mat(vision->patch); free_mat(vision->m1); free_mat(vision->m2);
    for (auto& B : vision->blocks) { free_mat(B.qkv); free_mat(B.proj); free_mat(B.fc1); free_mat(B.fc2); }
  }
}

// ---- public API ------------------------------------------------------------------------------------------------------------------------
Q27CpuEngine::Q27CpuEngine(const std::string& ckpt_dir, const CpuOptions& opt) : p_(std::make_unique<Impl>()) {
  p_->ckpt_dir = ckpt_dir;
  p_->opt = opt;
  p_->layer_hook = &layer_hook;
  p_->layer_boundary = &layer_boundary;
  p_->layer_inject = &layer_inject;
  p_->load();
}
Q27CpuEngine::~Q27CpuEngine() = default;
const Q27Config& Q27CpuEngine::cfg() const { return p_->c; }
bool Q27CpuEngine::owns(int l) const { return p_->layers[l].on; }
size_t Q27CpuEngine::weight_bytes() const { return p_->wbytes; }
CpuStats& Q27CpuEngine::stats() { return p_->st; }
bool Q27CpuEngine::mtp_on() const { return p_->mtp && p_->mtp_k > 0; }
int Q27CpuEngine::mtp_k() const { return p_->mtp_k; }
void Q27CpuEngine::set_embed_override(std::vector<EmbedOverride> v) { p_->emb_over = std::move(v); }

void Q27CpuEngine::init_seq(CpuSeq& s, int64_t cap) {
  const Impl& I = *p_;
  const Q27Config& c = I.c;
  s.cap = cap;
  s.conv.assign(c.n_lin(), {}); s.state.assign(c.n_lin(), {});
  for (int l = 0; l < c.n_layers; ++l)
    if (I.layers[l].on && !c.is_attn[l]) {
      const int li = I.lin_index[l];
      s.conv[li].assign((size_t)3 * c.lin_conv_ch(), 0.f);
      s.state[li].assign((size_t)c.lin_v_heads * c.lin_k_dim * c.lin_v_dim, 0.f);
    }
  s.attn.assign(c.n_attn(), {});
  reset_seq(s);
}

void Q27CpuEngine::reset_seq(CpuSeq& s) {
  for (auto& v : s.conv) std::fill(v.begin(), v.end(), 0.f);
  for (auto& v : s.state) std::fill(v.begin(), v.end(), 0.f);
  for (auto& a : s.attn) { a.k.clear(); a.v.clear(); }
  s.mtp.k.clear(); s.mtp.v.clear();
  s.pos = 0; s.tokens.clear(); s.rope_delta = 0; s.rope.clear(); s.broken = false;
  s.h_last.assign(p_->mtp ? (size_t)p_->c.hidden : 0, 0.f); s.h_last_valid = false; s.mtp_written = 0;
}

void Q27CpuEngine::prefill(CpuSeq& s, const int32_t* ids, int T, float* logits) {
  if (T <= 0) return;
  if ((int64_t)s.tokens.size() != s.pos) throw std::runtime_error("q27 cpu: sequence tokens out of step with its position");
  try {
    for (int off = 0; off < T; off += p_->max_chunk) {
      const int n = std::min(p_->max_chunk, T - off);
      p_->emb_base = off;
      std::vector<RowInfo> rows(n);
      for (int i = 0; i < n; ++i) rows[i] = {&s, s.pos + i};
      p_->forward(rows, ids + off, true, off + n == T ? logits : nullptr, false);
    }
  } catch (...) { s.broken = true; p_->emb_base = 0; p_->emb_over.clear(); throw; }
  p_->emb_base = 0;
  p_->emb_over.clear();
}

void Q27CpuEngine::decode(CpuSeq* const* seqs, const int32_t* ids, int M, float* logits) {
  std::vector<RowInfo> rows(M);
  for (int m = 0; m < M; ++m) rows[m] = {seqs[m], seqs[m]->pos};
  try { p_->forward(rows, ids, false, logits, true); } catch (...) { for (int m = 0; m < M; ++m) seqs[m]->broken = true; throw; }
}

void Q27CpuEngine::verify_batch(CpuSeq* const* seqs, const int* Ms, int S, const int32_t* ids, float* logits) {
  Impl& I = *p_;
  I.vparts.clear();
  std::vector<RowInfo> rows;
  int R = 0;
  for (int p = 0; p < S; ++p) {
    if (Ms[p] < 1 || R + Ms[p] > 8) throw std::runtime_error("q27 cpu verify rows");
    I.vparts.push_back({seqs[p], seqs[p]->pos, Ms[p], R});
    for (int r = 0; r < Ms[p]; ++r) rows.push_back({seqs[p], seqs[p]->pos + r});
    R += Ms[p];
  }
  I.alloc_snaps(seqs, Ms, S);
  I.verify_ids.assign(ids, ids + R);
  I.verify = true;
  try { I.forward(rows, ids, false, logits, true); } catch (...) { I.verify = false; I.vparts.clear(); for (int p = 0; p < S; ++p) seqs[p]->broken = true; throw; }
  I.verify = false;
}

void Q27CpuEngine::rollback_batch(const int* n_keep, int S) {
  Impl& I = *p_;
  if (S != (int)I.vparts.size()) throw std::runtime_error("q27 cpu rollback without a matching verify");
  const int H = I.c.hidden;
  for (int p = 0; p < S; ++p) {
    const Impl::VPart& vp = I.vparts[p];
    CpuSeq& s = *vp.s;
    const int M = vp.M, nk = n_keep[p];
    if (nk < 1 || nk > M || s.pos != vp.pos0 + M) throw std::runtime_error("q27 cpu rollback without a matching verify");
    if (nk < M) {
      const Impl::Snap& sn = I.snaps[p][nk - 1];
      for (size_t i = 0; i < s.conv.size(); ++i) if (!s.conv[i].empty()) { s.conv[i] = sn.conv[i]; s.state[i] = sn.state[i]; }
    }
    s.pos = vp.pos0 + nk;
    s.tokens.resize(s.tokens.size() - (size_t)(M - nk));
    if (I.mtp && I.mtp_k > 0) {
      if (nk >= 2) I.mtp_kv(s, vp.pos0, nk - 1, I.fh.data() + (size_t)vp.r0 * H, I.verify_ids.data() + vp.r0 + 1);
      memcpy(s.h_last.data(), I.fh.data() + (size_t)(vp.r0 + nk - 1) * H, (size_t)H * 4);
      s.h_last_valid = true;
    }
  }
  I.vparts.clear();
}

const std::vector<float>& Q27CpuEngine::last_draft_logits() const { return p_->draft_logits; }
int Q27CpuEngine::mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf) { return p_->mtp_draft(s, tok, k, drafts, conf); }

// ---- hybrid placement ----------------------------------------------------------------------------------------------------------------
void Q27CpuEngine::run_layer(int l, const Row* rows, int M, const int32_t* coords, float* h) {
  Impl& I = *p_;
  if (!I.layers[l].on) throw std::runtime_error("q27 cpu: layer " + std::to_string(l) + " is not on the CPU");
  std::vector<RowInfo> rv(rows, rows + M);
  if (I.c.is_attn[l])
    for (const RowInfo& ri : rv) grow_attn(ri.s->attn[I.attn_index[l]], ri.pos + 1, I.c.kv_width());
  const double t0 = hive::mono_ms();
  I.layer(l, I.layers[l], false, rv, coords, h);
  I.st.ms_total += hive::mono_ms() - t0;
}

void Q27CpuEngine::begin_verify(CpuSeq* const* seqs, const int64_t* pos0, const int* Ms, int S) {
  Impl& I = *p_;
  I.vparts.clear();
  int R = 0;
  for (int p = 0; p < S; ++p) { I.vparts.push_back({seqs[p], pos0[p], Ms[p], R}); R += Ms[p]; }
  I.alloc_snaps(seqs, Ms, S);
  I.verify = true;
}
void Q27CpuEngine::end_verify() { p_->verify = false; }
void Q27CpuEngine::rollback_layers(CpuSeq* const* seqs, const int* Ms, const int* n_keep, int S) {
  Impl& I = *p_;
  for (int p = 0; p < S && p < (int)I.snaps.size(); ++p) {
    if (n_keep[p] >= Ms[p] || n_keep[p] < 1) continue;
    const Impl::Snap& sn = I.snaps[p][n_keep[p] - 1];
    CpuSeq& s = *seqs[p];
    for (size_t i = 0; i < s.conv.size(); ++i) if (!s.conv[i].empty()) { s.conv[i] = sn.conv[i]; s.state[i] = sn.state[i]; }
  }
  I.vparts.clear();
}

const uint16_t* Q27CpuEngine::embed_table() const { return p_->embed.data(); }
void Q27CpuEngine::head_logits(const float* h, int M, float* logits) {
  if (!p_->lm_head.N) throw std::runtime_error("q27 cpu: no lm_head on the CPU");
  p_->head(h, M, p_->final_norm, logits);
}

bool Q27CpuEngine::has_vision() const { return p_->vis_avail; }

}  // namespace hive::q27::cpu
