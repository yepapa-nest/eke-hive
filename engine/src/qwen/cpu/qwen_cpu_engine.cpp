// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next on the CPU — loading and the layer forward (hive/qwen/qwen_cpu.h). The arithmetic and every bf16 rounding point follow
//   oracle/qwen4_oracle.py (the per-layer reference), so the engine's hidden streams can be compared with it layer by layer.
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "hive/clock.h"
#include "qwen_cpu_gdn.h"
#include "qwen_cpu_internal.h"

namespace hive::qwen::cpu {

using ::hive::cpu::Pool;
using ::hive::cpu::WType;
using ::hive::cpu::SType;

namespace {
inline float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
inline uint16_t f2b_exact(float x) { uint32_t u; memcpy(&u, &x, 4); return (uint16_t)(u >> 16); }  // x already a bf16 value
uint16_t f2b(float x) { return f2b_exact(bf16r(x)); }
bool env_off(const char* n) { const char* v = getenv(n); return v && strcmp(v, "0") == 0; }
}  // namespace

// ---- helpers ------------------------------------------------------------------------------------------------------------------------------
void par_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f) {
  if (n == 0) return;
  const size_t nb = (n + grain - 1) / grain;
  if (nb <= 1) { f(0, n); return; }
  Pool::get().run((int)nb, [&](int b) { const size_t a = (size_t)b * grain; f(a, std::min(n, a + grain)); });
}

void lin(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy, bool round) {
  ::hive::cpu::matmul(W, X, M, ldx, Y, ldy);
  if (!round) return;
  const int N = W.r1 - W.r0;
  par_for((size_t)M, std::max<size_t>(1, 262144 / std::max(1, N)), [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) simd::round_n(Y + m * ldy, (size_t)N);
  });
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
  else throw std::runtime_error("qwen cpu: unexpected dtype " + t.dtype + " for " + name);
  return v;
}

// Opt-in lossy I8 compression of bf16 dense weights (decode reads half the bytes): HIVE_QWEN_CPU_DENSE_I8 = comma list of categories
//   gdn, attn, hc, head, ple, mtp, moe (router + shared expert — routing is sensitive to it), or "all" (every category but moe);
//   HIVE_QWEN_CPU_I8_GROUP = columns per scale (default 64). Off by default.
static bool i8_enabled(const char* cat) {
  static const std::string list = [] { const char* v = getenv("HIVE_QWEN_CPU_DENSE_I8"); return std::string(v ? v : ""); }();
  if (!cat || list.empty() || list == "0") return false;
  const std::string c(cat);
  if (list == "all") return c != "moe";
  size_t a = 0;
  while (a <= list.size()) {
    const size_t b = std::min(list.find(',', a), list.size());
    if (list.compare(a, b - a, c) == 0) return true;
    a = b + 1;
  }
  return false;
}
static int i8_group() { static const int g = getenv("HIVE_QWEN_CPU_I8_GROUP") ? std::max(8, atoi(getenv("HIVE_QWEN_CPU_I8_GROUP")) / 8 * 8) : 64; return g; }

Mat load_bf16_mat(Checkpoint& ck, const std::vector<std::string>& names, const char* cat) {
  int N = 0, K = -1;
  std::vector<const TensorInfo*> ts;
  for (const auto& n : names) {
    const TensorInfo& t = ck.get(n);
    if (t.dtype != "BF16" || t.shape.size() != 2) throw std::runtime_error("qwen cpu: expected a BF16 matrix: " + n);
    if (K >= 0 && K != (int)t.shape[1]) throw std::runtime_error("qwen cpu: stacked widths differ: " + n);
    K = (int)t.shape[1]; N += (int)t.shape[0];
    ts.push_back(&t);
  }
  const bool i8 = i8_enabled(cat) && K % i8_group() == 0;
  if (ts.size() == 1)
    return i8 ? ::hive::cpu::make_mat_i8(N, K, (const uint16_t*)ts[0]->data, i8_group()) : ::hive::cpu::make_mat(WType::BF16, N, K, ts[0]->data);
  std::vector<uint8_t> tmp((size_t)N * K * 2);
  size_t off = 0;
  for (auto* t : ts) { memcpy(tmp.data() + off, t->data, t->nbytes); off += t->nbytes; }
  return i8 ? ::hive::cpu::make_mat_i8(N, K, (const uint16_t*)tmp.data(), i8_group()) : ::hive::cpu::make_mat(WType::BF16, N, K, tmp.data());
}

void QwenCpuEngine::Impl::ppar(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f) {
  if (legacy & 4) f(0, n); else par_for(n, grain, f);
}

// ---- loading -------------------------------------------------------------------------------------------------------------------------------
namespace {
const std::string P = "model.language_model.";

void load_hc(Checkpoint& ck, const std::string& p, HcW& w, bool inject) {
  std::vector<std::string> rows{p + "input_mix_weight_down.weight"};
  if (inject) rows.push_back(p + "block_inject_weight.weight");
  w.down_inj = load_bf16_mat(ck, rows, "hc");
  w.up = load_bf16_mat(ck, {p + "input_mix_weight_up.weight"}, "hc");
  w.norm = load_vec(ck, p + "hc_norm.weight");
  w.inject = inject;
}

void load_attn(Checkpoint& ck, const std::string& a, AttW& w) {
  w.in = load_bf16_mat(ck, {a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight", a + "indexer.index_qk_proj.weight"}, "attn");
  w.out = load_bf16_mat(ck, {a + "o_proj.weight"}, "attn");
  w.q_norm = load_vec(ck, a + "q_norm.weight"); w.k_norm = load_vec(ck, a + "k_norm.weight");
  w.iq_norm = load_vec(ck, a + "indexer.q_layernorm.weight"); w.ik_norm = load_vec(ck, a + "indexer.k_layernorm.weight");
}

// Routed experts of one layer (see MoeW): NVFP4 (main layers) or FP8 128×128 (the MTP layer), stacked into the two NUMA-friendly matrices.
void load_experts(Checkpoint& ck, const std::string& p, int E, int H, int I, bool fp8, MoeW& w) {
  w.fp8 = fp8;
  const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
  auto name = [&](int e, int j, const char* suf) { return p + "experts." + std::to_string(e) + "." + proj[j] + "." + suf; };
  w.g_gate.assign(E, 1.f); w.g_up.assign(E, 1.f); w.g_down.assign(E, 1.f);
  if (!fp8) {
    const size_t gw = (size_t)I * H / 2, gs = (size_t)I * H / 16, dw = (size_t)H * I / 2, ds = (size_t)H * I / 16;
    std::vector<uint8_t> GW(2 * E * gw), GS(2 * E * gs), DW(E * dw), DS(E * ds);
    for (int e = 0; e < E; ++e) {
      for (int j = 0; j < 2; ++j) {
        const TensorInfo& tw = ck.get(name(e, j, "weight"));
        const TensorInfo& ts = ck.get(name(e, j, "weight_scale"));
        const TensorInfo& tg = ck.get(name(e, j, "weight_scale_2"));
        if (tw.dtype != "U8" || ts.dtype != "F8_E4M3" || tw.nbytes != gw || ts.nbytes != gs) throw std::runtime_error("qwen cpu: expert must be NVFP4 " + name(e, j, ""));
        memcpy(GW.data() + ((size_t)j * E + e) * gw, tw.data, gw);
        memcpy(GS.data() + ((size_t)j * E + e) * gs, ts.data, gs);
        float g; memcpy(&g, tg.data, 4);
        (j == 0 ? w.g_gate : w.g_up)[e] = g;
      }
      const TensorInfo& tw = ck.get(name(e, 2, "weight"));
      const TensorInfo& ts = ck.get(name(e, 2, "weight_scale"));
      const TensorInfo& tg = ck.get(name(e, 2, "weight_scale_2"));
      if (tw.nbytes != dw || ts.nbytes != ds) throw std::runtime_error("qwen cpu: expert down shape " + name(e, 2, ""));
      const size_t hw = dw / 2, hs = ds / 2;  // rows 0..H/2 and H/2..H
      memcpy(DW.data() + (size_t)e * hw, tw.data, hw);
      memcpy(DW.data() + (size_t)E * hw + (size_t)e * hw, tw.data + hw, hw);
      memcpy(DS.data() + (size_t)e * hs, ts.data, hs);
      memcpy(DS.data() + (size_t)E * hs + (size_t)e * hs, ts.data + hs, hs);
      float g; memcpy(&g, tg.data, 4); w.g_down[e] = g;
    }
    w.gu = ::hive::cpu::make_mat(WType::NVFP4, 2 * E * I, H, GW.data(), GS.data(), SType::E4M3, 1.f);
    w.d = ::hive::cpu::make_mat(WType::NVFP4, E * H, I, DW.data(), DS.data(), SType::E4M3, 1.f);
    return;
  }
  // FP8 e4m3 [N, K] with BF16 "weight_scale_inv" [N/128, K/128]
  const size_t gw = (size_t)I * H, dw = (size_t)H * I;
  const int gsr = (I + 127) / 128, gsc = (H + 127) / 128, dsr = (H + 127) / 128, dsc = (I + 127) / 128;
  if (I % 128 || H % 256) throw std::runtime_error("qwen cpu: FP8 experts need 128-row blocks per half");
  std::vector<uint8_t> GW(2 * E * gw), DW(E * dw);
  std::vector<uint16_t> GS((size_t)2 * E * gsr * gsc), DS((size_t)E * dsr * dsc);
  for (int e = 0; e < E; ++e) {
    for (int j = 0; j < 3; ++j) {
      const TensorInfo& tw = ck.get(name(e, j, "weight"));
      const TensorInfo& ts = ck.get(name(e, j, "weight_scale_inv"));
      if (tw.dtype != "F8_E4M3" || ts.dtype != "BF16") throw std::runtime_error("qwen cpu: MTP expert must be FP8 + BF16 scales " + name(e, j, ""));
      if (j < 2) {
        memcpy(GW.data() + ((size_t)j * E + e) * gw, tw.data, gw);
        memcpy(GS.data() + ((size_t)j * E + e) * gsr * gsc, ts.data, (size_t)gsr * gsc * 2);
      } else {
        const size_t hw = dw / 2, hr = (size_t)dsr / 2 * dsc;
        memcpy(DW.data() + (size_t)e * hw, tw.data, hw);
        memcpy(DW.data() + (size_t)E * hw + (size_t)e * hw, tw.data + hw, hw);
        memcpy(DS.data() + (size_t)e * hr, ts.data, hr * 2);
        memcpy(DS.data() + (size_t)E * hr + (size_t)e * hr, (const uint8_t*)ts.data + hr * 2, hr * 2);
      }
    }
  }
  w.gu = ::hive::cpu::make_mat(WType::FP8, 2 * E * I, H, GW.data(), GS.data(), SType::BF16, 1.f, 128, 128);
  w.d = ::hive::cpu::make_mat(WType::FP8, E * H, I, DW.data(), DS.data(), SType::BF16, 1.f, 128, 128);
}

void load_moe(Checkpoint& ck, const std::string& p, const QwenConfig& c, bool fp8, MoeW& w) {
  w.front = load_bf16_mat(ck, {p + "gate.weight", p + "shared_expert_gate.weight", p + "shared_expert.gate_proj.weight", p + "shared_expert.up_proj.weight"}, "moe");
  w.sh_down = load_bf16_mat(ck, {p + "shared_expert.down_proj.weight"}, "moe");
  load_experts(ck, p, c.n_experts, c.hidden, c.moe_inter, fp8, w);
}
}  // namespace

void QwenCpuEngine::Impl::load(bool load_mtp) {
  const double t0 = hive::mono_ms();
  c = QwenConfig::load(ckpt_dir);
  ck = std::make_unique<Checkpoint>(ckpt_dir);
  if (const char* v = getenv("HIVE_QWEN_CPU_CHUNK"); v && atoi(v) > 0) max_chunk = atoi(v);
  if (!c.ple_layer_ids.empty()) ple_layer = c.ple_layer_ids[0] - 1;
  first_layer = std::max(0, std::min(first_layer, c.n_layers));
  if (ple_layer < first_layer) ple_layer = -1;   // hybrid placement: the PLE layer runs elsewhere
  layers.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; ++l) layers[l].attn = c.is_attn[l];
  // checkpoint lookups are not thread-safe: layers load one after another (each make_mat copy is the bulk of the time)
  for (int l = first_layer; l < c.n_layers; ++l) {
    LayerW& L = layers[l];
    const std::string p = P + "layers." + std::to_string(l) + ".";
    L.attn = c.is_attn[l];
    load_hc(*ck, p + "attn_hyper_connection.", L.ha, true);
    load_hc(*ck, p + "mlp_hyper_connection.", L.hm, true);
    if (L.attn) load_attn(*ck, p + "self_attn.", L.att);
    else {
      const std::string a = p + "linear_attn.";
      L.lin.in = load_bf16_mat(*ck, {a + "in_proj_qkv.weight", a + "in_proj_z.weight", a + "in_proj_b.weight", a + "in_proj_a.weight"}, "gdn");
      L.lin.out = load_bf16_mat(*ck, {a + "out_proj.weight"}, "gdn");
      L.lin.conv_w = load_vec(*ck, a + "conv1d.weight");
      L.lin.A_log = load_vec(*ck, a + "A_log"); L.lin.dt_bias = load_vec(*ck, a + "dt_bias");
      L.lin.norm = load_vec(*ck, a + "norm.weight");
    }
    load_moe(*ck, p + "mlp.", c, false, L.moe);
    if (l == ple_layer) {
      L.ple = true;
      const std::string q = p + "ple.";
      L.plew.key = load_bf16_mat(*ck, {q + "key_proj.weight"}, "ple");
      L.plew.value = load_bf16_mat(*ck, {q + "value_proj.weight"}, "ple");
      L.plew.norm_key = load_vec(*ck, q + "norm_key.weight"); L.plew.norm_query = load_vec(*ck, q + "norm_query.weight");
      L.plew.norm_conv = load_vec(*ck, q + "norm_conv.weight"); L.plew.conv_w = load_vec(*ck, q + "conv1d.weight");
    }
    if ((l + 1) % 12 == 0) fprintf(stderr, "[qwen-cpu] layers 0..%d loaded (%.1f s)\n", l, (hive::mono_ms() - t0) / 1000);
  }
  if (with_head) {  // final mixer, lm_head, embedding (a hybrid tail engine gets its input streams and returns them)
    load_hc(*ck, P + "hyper_connection_mixer.", mixer, false);
    lm_head = load_bf16_mat(*ck, {"lm_head.weight"}, "head");
    const TensorInfo& e = ck->get(P + "embed_tokens.weight");
    embed.resize((size_t)e.numel());
    memcpy(embed.data(), e.data, e.nbytes);
  }
  if (load_mtp && with_head && first_layer == 0 && c.n_mtp > 0 && ck->has("mtp.fc_embedding.weight")) {
    mtp = std::make_unique<MtpW>();
    mtp->fc_emb = load_bf16_mat(*ck, {"mtp.fc_embedding.weight"}, "mtp");
    mtp->fc_hidden = load_bf16_mat(*ck, {"mtp.fc_hidden.weight"}, "mtp");
    mtp->pre_emb_norm = load_vec(*ck, "mtp.pre_fc_norm_embedding.weight");
    mtp->pre_hidden_norm = load_vec(*ck, "mtp.pre_fc_norm_hidden.weight");
    LayerW& L = mtp->layer;
    L.attn = true;
    load_hc(*ck, "mtp.layers.0.attn_hyper_connection.", L.ha, true);
    load_hc(*ck, "mtp.layers.0.mlp_hyper_connection.", L.hm, true);
    load_attn(*ck, "mtp.layers.0.self_attn.", L.att);
    load_moe(*ck, "mtp.layers.0.mlp.", c, true, L.moe);
    load_hc(*ck, "mtp.hyper_connection_mixer.", mtp->mixer, false);
    mtp_k = getenv("HIVE_QWEN_MTP_K") ? std::max(0, std::min(7, atoi(getenv("HIVE_QWEN_MTP_K")))) : 3;
  }
  if (const char* rd = getenv("HIVE_QWEN_CPU_ROUTE_DUMP"); rd && *rd) route_dump = fopen(rd, "w");
  if (ple_layer >= 0) {
    ple_hash.ngram = c.ngram; ple_hash.heads_per = c.heads_per_ngram; ple_hash.eos = c.eos_id; ple_hash.vocab = c.vocab;
    ple = ple_open(*ck, P + "layers." + std::to_string(ple_layer) + ".ple.", ple_hash, ple_scale);
  }
  const int half = c.rotary_dim / 2;
  rope_inv.resize(half); rope_axis.assign(half, 0);
  for (int i = 0; i < half; ++i) {
    rope_inv[i] = std::pow((double)c.rope_theta, -2.0 * i / c.rotary_dim);
    if (c.mrope_interleaved) rope_axis[i] = (i % 3 == 1 && i < 3 * c.mrope_section[1]) ? 1 : (i % 3 == 2 && i < 3 * c.mrope_section[2]) ? 2 : 0;
    else rope_axis[i] = i < c.mrope_section[0] ? 0 : i < c.mrope_section[0] + c.mrope_section[1] ? 1 : 2;
  }
  // the checkpoint pages are no longer needed (weights were copied); PLE in SSD mode reads the file by its own descriptor
  if (const char* v = getenv("HIVE_QWEN_CPU_DENSE_I8_AB"); v && *v && strcmp(v, "0") != 0) build_i8_alt(v);
  ck->release_all();
  fprintf(stderr, "[qwen-cpu] model loaded in %.1f s · %d threads on %d NUMA node(s) · PLE %s · MTP k %d · chunk %d\n", (hive::mono_ms() - t0) / 1000,
          Pool::get().threads(), Pool::get().nodes(), ple ? ple->kind() : "none", mtp_k, max_chunk);
}

// ---- building blocks -------------------------------------------------------------------------------------------------------------------------
void QwenCpuEngine::Impl::rope_row(float* v, int n_heads, int head_dim, const int32_t* coord) const {
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

void QwenCpuEngine::Impl::hc_mix(const HcW& w, const float* h, int M, float* x, std::vector<float>& hn, std::vector<float>& dl) {
  const int H = c.hidden, HC = c.hcw(), rk = c.hc_rank, nd = w.down_inj.N, hc = c.hc;
  hn.resize((size_t)M * HC); dl.resize((size_t)M * nd);
  norm1p(h, HC, w.norm.data(), M, HC, H, c.rms_eps, hn.data(), HC);
  lin(w.down_inj, hn.data(), M, HC, dl.data(), nd);
  thread_local std::vector<float> s_tl, u_tl;
  // worker threads see their own thread_local instances: the lambdas below must use the caller's through these references
  auto& s = s_tl; auto& u = u_tl;
  s.resize((size_t)M * rk); u.resize((size_t)M * HC);
  ppar((size_t)M, 32, [&](size_t a, size_t b) {  // s = bf16(silu(bf16(dl / hc)))
    for (size_t m = a; m < b; ++m) {
      const float* dr = dl.data() + m * nd;
      float* sr = s.data() + m * rk;
      for (int j = 0; j < rk; ++j) sr[j] = bf16r(dr[j] / hc);
      for (int j = 0; j < rk; ++j) sr[j] = bf16r(siluf(sr[j]));
    }
  });
  lin(w.up, s.data(), M, rk, u.data(), HC);
  par_for((size_t)M, 8, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) simd::hc_mix_v(u.data() + m * HC, hn.data() + m * HC, H, hc, x + m * H);
  });
}

void QwenCpuEngine::Impl::hc_combine(const HcW& w, float* h, const float* y, int M, const std::vector<float>& dl) {
  const int H = c.hidden, HC = c.hcw(), rk = c.hc_rank, nd = w.down_inj.N, hc = c.hc;
  par_for((size_t)M, 8, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      float g[8];
      for (int cc = 0; cc < hc; ++cc) g[cc] = bf16r(2.f * sigm(bf16r(dl[m * nd + rk + cc] / hc)));
      simd::hc_combine_v(h + m * HC, y + m * H, g, H, hc);
    }
  });
}

void QwenCpuEngine::Impl::head(const float* h, int M, float* logits) {
  thread_local std::vector<float> xf_tl, hn_tl, dl_tl;
  // worker threads see their own thread_local instances: the lambdas below must use the caller's through these references
  auto& xf = xf_tl; auto& hn = hn_tl; auto& dl = dl_tl;
  xf.resize((size_t)M * c.hidden);
  hc_mix(mixer, h, M, xf.data(), hn, dl);
  ::hive::cpu::matmul(lm_head, xf.data(), M, c.hidden, logits, c.vocab);
}

// ---- PLE -----------------------------------------------------------------------------------------------------------------------------------
void QwenCpuEngine::Impl::ple_block(const std::vector<RowInfo>& rows, const int32_t* ids, float* h, bool prefill) {
  const int M = (int)rows.size(), H = c.hidden, HC = c.hcw(), hc = c.hc;
  const PleW& w = layers[ple_layer].plew;
  std::vector<int64_t> rid((size_t)M * kPleHeads);
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s && rows[r + n].pos == rows[r].pos + n) ++n;
    const int64_t p0 = rows[r].pos;
    const int32_t prev1 = p0 >= 1 ? s.tokens[p0 - 1] : -1, prev2 = p0 >= 2 ? s.tokens[p0 - 2] : -1;
    ple_hash.rows_of(ids + r, n, prev2, prev1, rid.data() + (size_t)r * kPleHeads);
    r += n;
  }
  std::vector<uint8_t> raw((size_t)M * kPleRow);
  ple->gather(rid.data(), M, raw.data());
  std::vector<float> e((size_t)M * kPleRow), key((size_t)M * HC), val((size_t)M * H), gv((size_t)M * HC), gvn((size_t)M * HC);
  ppar(e.size(), 65536, [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) e[i] = bf16r(::hive::cpu::e4m3_to_f32(raw[i]) * ple_scale);
  });
  lin(w.key, e.data(), M, c.ple_dim, key.data(), HC);
  lin(w.value, e.data(), M, c.ple_dim, val.data(), H);
  std::vector<float> kn((size_t)M * HC), qn((size_t)M * HC);
  norm1p(key.data(), HC, w.norm_key.data(), M, HC, H, c.rms_eps, kn.data(), HC);
  norm1p(h, HC, w.norm_query.data(), M, HC, H, c.rms_eps, qn.data(), HC);
  const float isd = 1.f / std::sqrt((float)H);
  ppar((size_t)M * hc, 8, [&](size_t ia, size_t ib) {
    for (size_t i = ia; i < ib; ++i) {
      const size_t m = i / hc, cc = i % hc;
      const float* a = kn.data() + m * HC + cc * H;
      const float* b = qn.data() + (size_t)m * HC + (size_t)cc * H;
      float g = 0.f;
      for (int d = 0; d < H; ++d) g += a[d] * b[d];
      g *= isd;
      const float sg = g > 0.f ? 1.f : g < 0.f ? -1.f : 0.f;
      g = bf16r(sigm(sg * std::sqrt(std::max(std::fabs(g), 1e-6f))));
      for (int d = 0; d < H; ++d) gv[m * HC + cc * H + d] = bf16r(g * val[m * H + d]);
    }
  });
  norm1p(gv.data(), HC, w.norm_conv.data(), M, HC, H, c.rms_eps, gvn.data(), HC);
  // causal conv over each sequence's rows (taps t−9, t−6, t−3, t), state = the last 9 inputs
  const float* cw = w.conv_w.data();
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s && rows[r + n].pos == rows[r].pos + n) ++n;
    int part = -1;
    if (verify) for (size_t p = 0; p < vparts.size(); ++p) if (vparts[p].r0 == r) part = (int)p;
    std::vector<float>& st = s.ple_state;  // [9, HC] oldest first
    par_for((size_t)HC, 1024, [&](size_t a, size_t b) {
      for (size_t ch = a; ch < b; ++ch) {
        float win[9 + 64];
        const int nn = n;
        std::vector<float> big;
        float* seq = win;
        if (nn > 64) { big.resize(9 + nn); seq = big.data(); }
        for (int k = 0; k < 9; ++k) seq[k] = st[(size_t)k * HC + ch];
        for (int t = 0; t < nn; ++t) seq[9 + t] = gvn[(size_t)(r + t) * HC + ch];
        const float* w4 = cw + ch * 4;
        for (int t = 0; t < nn; ++t) {
          const int j = 9 + t;
          const float y = w4[0] * seq[j - 9] + w4[1] * seq[j - 6] + w4[2] * seq[j - 3] + w4[3] * seq[j];
          const float o = bf16r(siluf(bf16r(y)));
          const size_t i = (size_t)(r + t) * HC + ch;
          h[i] = bf16r(h[i] + bf16r(gv[i] + o));
          if (part >= 0 && t + 1 < nn) {
            std::vector<float>& sp = snaps[part][t].ple;
            for (int k = 0; k < 9; ++k) sp[(size_t)k * HC + ch] = seq[t + 1 + k];
          }
        }
        for (int k = 0; k < 9; ++k) st[(size_t)k * HC + ch] = seq[nn + k];
      }
    });
    r += n;
  }
  (void)prefill;
}

// ---- Gated DeltaNet ------------------------------------------------------------------------------------------------------------------------
void QwenCpuEngine::Impl::lin_layer(const LinW& w, int li, const std::vector<RowInfo>& rows, bool prefill, const float* x, float* out) {
  (void)prefill;
  const int M = (int)rows.size(), H = c.hidden;
  const int KH = c.lin_k_heads, VH = c.lin_v_heads, D = c.lin_k_dim, KW = KH * D, VW = VH * D, C = 2 * KW + VW;
  const int PW = C + VW + 2 * VH, ZO = C, BO = C + VW, AO = BO + VH;
  thread_local std::vector<float> proj_tl, y_tl, qn_tl, kn_tl, gg_tl, beta_tl, o_tl;
  // worker threads see their own thread_local instances: the lambdas below must use the caller's through these references
  auto& proj = proj_tl; auto& y = y_tl; auto& qn = qn_tl; auto& kn = kn_tl; auto& gg = gg_tl; auto& beta = beta_tl; auto& o = o_tl;
  proj.resize((size_t)M * PW); y.resize((size_t)M * C); qn.resize((size_t)M * KW); kn.resize((size_t)M * KW);
  gg.resize((size_t)M * VH); beta.resize((size_t)M * VH); o.resize((size_t)M * VW);
  lin(w.in, x, M, H, proj.data(), PW);
  const float* cw = w.conv_w.data();
  const float qs = 1.f / std::sqrt((float)D);
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s && rows[r + n].pos == rows[r].pos + n) ++n;
    int part = -1;
    if (verify) for (size_t p = 0; p < vparts.size(); ++p) if (vparts[p].r0 == r) part = (int)p;
    std::vector<float>& hist = s.conv[li];   // [3, C] oldest first
    // conv + SiLU per channel (tap 3 = current token)
    par_for((size_t)C, 512, [&](size_t a, size_t b) {
      for (size_t ch = a; ch < b; ++ch) {
        const float* w4 = cw + ch * 4;
        auto X = [&](int t) { return t >= 0 ? proj[(size_t)(r + t) * PW + ch] : hist[(size_t)(3 + t) * C + ch]; };
        for (int t = 0; t < n; ++t) {
          float v = w4[0] * X(t - 3);
          v += w4[1] * X(t - 2); v += w4[2] * X(t - 1); v += w4[3] * X(t);
          y[(size_t)(r + t) * C + ch] = bf16r(siluf(v));
          if (part >= 0 && t + 1 < n)
            for (int k = 0; k < 3; ++k) snaps[part][t].conv[li][(size_t)k * C + ch] = X(t - 2 + k);  // the 3 inputs up to token t
        }
        float nv[3];
        for (int k = 0; k < 3; ++k) nv[k] = X(n - 3 + k);
        for (int k = 0; k < 3; ++k) hist[(size_t)k * C + ch] = nv[k];
      }
    });
    // q, k l2 norm per key head; decay and step size per value head (tokens in parallel)
    ppar((size_t)n, 8, [&](size_t ta, size_t tb) {
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
    // recurrence (qwen_cpu_gdn.h): S [k][v] fp32 per value head, columns independent. Verify parts (state snapshots per token) and single
    //   tokens run the two-pass form, one task per head; longer runs take the fused one-pass form on L1-resident 32-column slabs, four tasks
    //   per head (bit-identical; the plain 32-column split without the slab copy was measured slower: GDN 9.1 → 14.9 s at 2K).
    const int rep = VH / KH;
    const bool slab = part < 0 && n > 1 && !(legacy & 8) && D % 32 == 0 && D <= 128;
    const int CG = slab ? D / 32 : 1;
    Pool::get().run(VH * CG, [&](int task) {
      const int cw = D / CG, vh = task / CG, j0 = (task % CG) * cw, j1 = j0 + cw;
      float* S = s.state[li].data() + (size_t)vh * D * D;
      const int kh = vh / rep;
      gdn::ScanIn in{kn.data() + (size_t)r * KW + kh * D, (size_t)KW, qn.data() + (size_t)r * KW + kh * D, (size_t)KW,
                     y.data() + (size_t)r * C + 2 * KW + (size_t)vh * D, (size_t)C, gg.data() + (size_t)r * VH + vh, (size_t)VH,
                     beta.data() + (size_t)r * VH + vh, (size_t)VH, o.data() + (size_t)r * VW + (size_t)vh * D, (size_t)VW};
      if (slab) { gdn::gdn_scan_slab(S, D, n, in, j0, j1); return; }
      if (part < 0) { gdn::gdn_scan_ref(S, D, n, in, j0, j1); return; }
      for (int t = 0; t < n; ++t) {  // verify: one token at a time, the state after each token kept for rollback
        gdn::ScanIn one = in;
        one.k += (size_t)t * in.ks; one.q += (size_t)t * in.qs; one.v += (size_t)t * in.vs; one.e += (size_t)t * in.es; one.b += (size_t)t * in.bs;
        one.o += (size_t)t * in.os;
        gdn::gdn_scan_ref(S, D, 1, one, j0, j1);
        if (t + 1 < n) memcpy(snaps[part][t].state[li].data() + (size_t)vh * D * D, S, (size_t)D * D * 4);
      }
    });
    // gated RMS norm per value head
    ppar((size_t)n * VH, 64, [&](size_t ia, size_t ib) {
      for (size_t i = ia; i < ib; ++i) {
        const size_t row = (size_t)r + i / VH, vh = i % VH;
        simd::gated_norm_v(o.data() + row * VW + vh * D, proj.data() + row * PW + ZO + vh * D, w.norm.data(), D, c.rms_eps);
      }
    });
    r += n;
  }
  lin(w.out, o.data(), M, VW, out, H);
}

// ---- full attention + QSA ------------------------------------------------------------------------------------------------------------------
void QwenCpuEngine::Impl::attn_layer(const AttW& w, int ai, bool mtp_layer, const std::vector<RowInfo>& rows, const std::vector<int32_t>& coords,
                                     const float* x, float* out, bool kv_only) {
  const int M = (int)rows.size(), H = c.hidden, NH = c.n_heads, NKV = c.n_kv_heads, D = c.head_dim, IH = c.idx_heads, ID = c.idx_dim, R = c.idx_ratio;
  const int QW = 2 * NH * D, KO = QW, VO = KO + NKV * D, IQO = VO + NKV * D, IKO = IQO + IH * ID, PW = IKO + ID;
  const int KVW = NKV * D, NB = c.idx_topk_blocks(), SEL = c.idx_width();
  thread_local std::vector<float> proj_tl, q_tl, iq_tl, att_tl, o_tl;
  // worker threads see their own thread_local instances: the lambdas below must use the caller's through these references
  auto& proj = proj_tl; auto& q = q_tl; auto& iq = iq_tl; auto& att = att_tl; auto& o = o_tl;
  proj.resize((size_t)M * PW); q.resize((size_t)M * NH * D); iq.resize((size_t)M * IH * ID);
  lin(w.in, x, M, H, proj.data(), PW);
  // per row: q / k / iq norm + rope; cache writes (k, v, raw indexer key)
  par_for((size_t)M, 1, [&](size_t a, size_t b) {
    for (size_t r = a; r < b; ++r) {
      CpuSeq& s = *rows[r].s;
      CpuSeq::Attn& ca = mtp_layer ? s.mtp : s.attn[ai];
      const int64_t p = rows[r].pos;
      const float* pr = proj.data() + r * PW;
      const int32_t* cd = coords.data() + r * 3;
      float kbuf[2 * 256];
      norm1p(pr + KO, D, w.k_norm.data(), NKV, D, D, c.rms_eps, kbuf, D);
      rope_row(kbuf, NKV, D, cd);
      for (int i = 0; i < KVW; ++i) { ca.k[(size_t)p * KVW + i] = f2b_exact(kbuf[i]); ca.v[(size_t)p * KVW + i] = f2b_exact(pr[VO + i]); }
      for (int i = 0; i < ID; ++i) ca.ik[(size_t)p * ID + i] = f2b_exact(pr[IKO + i]);
      if (kv_only) continue;
      float* qr = q.data() + r * NH * D;
      for (int h = 0; h < NH; ++h) norm1p(pr + (size_t)h * 2 * D, 2 * D, w.q_norm.data(), 1, D, D, c.rms_eps, qr + (size_t)h * D, D);
      rope_row(qr, NH, D, cd);
      float* iqr = iq.data() + r * IH * ID;
      norm1p(pr + IQO, ID, w.iq_norm.data(), IH, ID, ID, c.rms_eps, iqr, ID);
      rope_row(iqr, IH, ID, cd);
    }
  });
  // complete the groups whose last token is one of these rows
  for (int r = 0; r < M; ++r) {
    const int64_t p = rows[r].pos;
    if ((p + 1) % R) continue;
    CpuSeq& s = *rows[r].s;
    CpuSeq::Attn& ca = mtp_layer ? s.mtp : s.attn[ai];
    const int64_t G = p / R;
    float mean[128], kn[128];
    for (int i = 0; i < ID; ++i) {
      float sum = 0.f;
      for (int j = 0; j < R; ++j) sum += b2f(ca.ik[(size_t)(G * R + j) * ID + i]);
      mean[i] = bf16r(sum / R);
    }
    norm1p(mean, ID, w.ik_norm.data(), 1, ID, ID, c.rms_eps, kn, ID);
    // coordinates of the group's first token: this call's rows (MTP drafts run ahead of the sequence) or the sequence's record
    const int32_t* c0 = nullptr;
    for (int q2 = 0; q2 < M; ++q2) if (rows[q2].s == &s && rows[q2].pos == G * R) c0 = coords.data() + (size_t)q2 * 3;
    int32_t ahead[3];
    if (!c0 && (int64_t)s.rope.size() >= (G * R + 1) * 3) c0 = s.rope.data() + (size_t)(G * R) * 3;
    if (!c0) { ahead[0] = ahead[1] = ahead[2] = (int32_t)(G * R + s.rope_delta); c0 = ahead; }
    rope_row(kn, 1, ID, c0);
    for (int i = 0; i < ID; ++i) ca.gk[(size_t)G * ID + i] = f2b_exact(kn[i]);
    if (G <= ca.gkt_n) {  // keep the fp32 scoring mirror in step (a rewritten group — rollback, MTP drafts — overwrites its slot)
      if ((int64_t)ca.gkt.size() < (G / 8 + 1) * 8 * ID) ca.gkt.resize((size_t)(G / 8 + 1) * 8 * ID, 0.f);
      for (int i = 0; i < ID; ++i) ca.gkt[((size_t)(G / 8) * ID + i) * 8 + G % 8] = kn[i];
      if (G == ca.gkt_n) ++ca.gkt_n;
    }
  }
  if (kv_only) return;
  att.resize((size_t)M * NH * D);
  // attention per sequence group (the backend reads one K / V cache per call)
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s) ++n;
    CpuSeq::Attn& ca = mtp_layer ? s.mtp : s.attn[ai];
    bool sparse = false;
    for (int t = 0; t < n; ++t) sparse = sparse || (rows[r + t].pos + 1) / R > NB;
    if (!sparse) {
      std::vector<int32_t> ce(n);
      for (int t = 0; t < n; ++t) ce[t] = (int32_t)rows[r + t].pos + 1;
      ::hive::cpu::attention(q.data() + (size_t)r * NH * D, n, NH, NKV, D, ca.k.data(), ca.v.data(), nullptr, 0, ce.data(), 1.f / std::sqrt((float)D),
                           att.data() + (size_t)r * NH * D);
    } else {
      std::vector<int32_t> idx((size_t)n * SEL, -1);
      // fp32 mirror of the group keys up to the last visible group (rebuilt after a restore)
      int64_t vmax = 0;
      for (int t = 0; t < n; ++t) vmax = std::max<int64_t>(vmax, (rows[r + t].pos + 1) / R);
      if ((int64_t)ca.gkt.size() < (vmax / 8 + 1) * 8 * ID) ca.gkt.resize((size_t)(vmax / 8 + 1) * 8 * ID, 0.f);
      for (; ca.gkt_n < vmax; ++ca.gkt_n)
        for (int i = 0; i < ID; ++i) ca.gkt[((size_t)(ca.gkt_n / 8) * ID + i) * 8 + ca.gkt_n % 8] = b2f(ca.gk[(size_t)ca.gkt_n * ID + i]);
      par_for((size_t)n, 1, [&](size_t a, size_t b) {
        std::vector<float> sc;
        std::vector<int> ord;
        for (size_t t = a; t < b; ++t) {
          const int64_t p = rows[r + t].pos;
          const int vis = (int)((p + 1) / R);
          int32_t* ix = idx.data() + t * SEL;
          if (vis <= NB) { for (int64_t j = 0; j <= p; ++j) ix[j] = (int32_t)j; continue; }
          const float* iqr = iq.data() + (size_t)(r + t) * IH * ID;
          // score of group g = Σ_h relu(q_h · k_g) / √d, 8 groups per vector: each lane runs the scalar form's operations in its order
          //   (products rounded, then added in i order — how the compiler builds the scalar loop — relu, heads added in order)
          sc.resize((size_t)(vis + 7) / 8 * 8);
          if (legacy & 16) {  // A/B: the scalar form
            for (int g = 0; g < vis; ++g) {
              const uint16_t* gk = ca.gk.data() + (size_t)g * ID;
              float kf[128];
              for (int i = 0; i < ID; ++i) kf[i] = b2f(gk[i]);
              float tot = 0.f;
              for (int h = 0; h < IH; ++h) {
                float d = 0.f;
                for (int i = 0; i < ID; ++i) d += iqr[h * ID + i] * kf[i];
                tot += std::max(d, 0.f);
              }
              sc[g] = tot / std::sqrt((float)ID);
            }
          } else {
          const __m256 rs = _mm256_set1_ps(std::sqrt((float)ID)), zero = _mm256_setzero_ps();
          for (int blk = 0; blk < (vis + 7) / 8; ++blk) {
            const float* kt = ca.gkt.data() + (size_t)blk * ID * 8;
            __m256 tot = _mm256_setzero_ps();
            for (int h = 0; h < IH; ++h) {
              const float* qh = iqr + h * ID;
              __m256 d = _mm256_setzero_ps();
              for (int i = 0; i < ID; ++i) d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_set1_ps(qh[i]), _mm256_loadu_ps(kt + (size_t)i * 8)));
              tot = _mm256_add_ps(tot, _mm256_max_ps(d, zero));
            }
            _mm256_storeu_ps(sc.data() + (size_t)blk * 8, _mm256_div_ps(tot, rs));
          }
          }
          ord.resize(vis);
          std::iota(ord.begin(), ord.end(), 0);
          std::partial_sort(ord.begin(), ord.begin() + NB, ord.end(), [&](int x1, int x2) { return sc[x1] > sc[x2] || (sc[x1] == sc[x2] && x1 < x2); });
          int k = 0;
          for (int i = 0; i < NB; ++i) for (int j = 0; j < R; ++j) ix[k++] = ord[i] * R + j;
          for (int64_t tpos = (int64_t)vis * R; tpos <= p; ++tpos) ix[k++] = (int32_t)tpos;
        }
      });
      ::hive::cpu::attention(q.data() + (size_t)r * NH * D, n, NH, NKV, D, ca.k.data(), ca.v.data(), idx.data(), SEL, nullptr, 1.f / std::sqrt((float)D),
                           att.data() + (size_t)r * NH * D);
    }
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

// ---- MoE ------------------------------------------------------------------------------------------------------------------------------------
void QwenCpuEngine::Impl::moe_layer(const MoeW& w, int M, const float* x, float* out) {
  const int H = c.hidden, E = c.n_experts, K = c.n_act, I = c.moe_inter, SI = c.shared_inter, FW = w.front.N;
  thread_local std::vector<float> front_tl, ys_tl, sh_tl, pair_out_tl;
  // worker threads see their own thread_local instances: the lambdas below must use the caller's through these references
  auto& front = front_tl; auto& ys = ys_tl; auto& sh = sh_tl; auto& pair_out = pair_out_tl;
  front.resize((size_t)M * FW); ys.resize((size_t)M * SI); sh.resize((size_t)M * H);
  lin(w.front, x, M, H, front.data(), FW);   // router logits, shared gate logit, shared gate, shared up — every output rounded (bf16 linears)
  // routing: softmax (fp32) over the bf16 logits, top-K (ties → smaller id), renormalised — rows in parallel
  thread_local std::vector<int> ids_tl;
  thread_local std::vector<float> wts_tl;
  auto& ids = ids_tl; auto& wts = wts_tl;
  ids.resize((size_t)M * K); wts.resize((size_t)M * K);
  ppar((size_t)M, 16, [&](size_t a, size_t b) {
    thread_local std::vector<float> pr;   // this worker's scratch
    thread_local std::vector<int> ord;
    pr.resize(E); ord.resize(E);
    for (size_t m = a; m < b; ++m) {
      const float* lg = front.data() + m * FW;
      float mx = -INFINITY;
      for (int e = 0; e < E; ++e) mx = std::max(mx, lg[e]);
      float sum = 0.f;
      for (int e = 0; e < E; ++e) { pr[e] = std::exp(lg[e] - mx); sum += pr[e]; }
      for (int e = 0; e < E; ++e) pr[e] /= sum;
      std::iota(ord.begin(), ord.end(), 0);
      std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int x1, int x2) { return pr[x1] > pr[x2] || (pr[x1] == pr[x2] && x1 < x2); });
      float tot = 0.f;
      for (int k = 0; k < K; ++k) tot += pr[ord[k]];
      for (int k = 0; k < K; ++k) { ids[m * K + k] = ord[k]; wts[m * K + k] = c.norm_topk ? pr[ord[k]] / tot : pr[ord[k]]; }
    }
  });
  if (route_dump) {
    fprintf(route_dump, "%d %d\n", cur_layer, M);
    for (int m = 0; m < M; ++m) { for (int k = 0; k < K; ++k) fprintf(route_dump, "%s%d", k ? " " : "", ids[(size_t)m * K + k]); fputc('\n', route_dump); }
    fflush(route_dump);
  }
  // shared expert activation
  ppar((size_t)M, 64, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const float* fr = front.data() + m * FW + E + 1;
      simd::silu_mul_v(fr, fr + SI, SI, 1.f, 1.f, ys.data() + m * SI);
    }
  });
  // routed experts: the (row, slot) pairs of each expert gathered contiguously (experts ascending, rows ascending inside). One dispatch for
  //   the shared expert's down projection and every expert's gate | up (per-job rows and activations; gu is stacked so the gates sit on one
  //   NUMA half and the ups on the other), one for every expert's down halves (one per node).
  const int P = M * K, hh = H / 2;
  thread_local std::vector<float> Xe_tl, G_tl, U_tl, Y_tl, D2_tl;
  // worker threads see their own thread_local instances: the lambdas below must use the caller's through these references
  auto& Xe = Xe_tl; auto& G = G_tl; auto& U = U_tl; auto& Y = Y_tl; auto& D2 = D2_tl;
  Xe.resize((size_t)P * H); G.resize((size_t)P * I); U.resize((size_t)P * I); Y.resize((size_t)P * I); D2.resize((size_t)P * H);
  pair_out.resize((size_t)P * H);
  thread_local std::vector<int> cnt_tl, off_tl, prow_tl, pslot_tl, pexp_tl;
  auto& cnt = cnt_tl; auto& off = off_tl; auto& prow = prow_tl; auto& pslot = pslot_tl; auto& pexp = pexp_tl;
  cnt.assign(E, 0); off.assign(E + 1, 0); prow.resize(P); pslot.resize(P); pexp.resize(P);
  for (int i = 0; i < P; ++i) ++cnt[ids[i]];
  for (int e = 0; e < E; ++e) off[e + 1] = off[e] + cnt[e];
  {
    std::vector<int> fill(off.begin(), off.end() - 1);
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k) { const int e = ids[(size_t)m * K + k], q = fill[e]++; prow[q] = m; pslot[q] = k; pexp[q] = e; }
  }
  ppar((size_t)P, 64, [&](size_t a, size_t b) {
    for (size_t q = a; q < b; ++q) memcpy(Xe.data() + q * H, x + (size_t)prow[q] * H, (size_t)H * 4);
  });
  std::vector<MatJob> jobs;
  jobs.reserve(2 * (size_t)std::min(P, E) + 1);
  // A/B legacy: the shared down alone, experts dispatched per equal row count (the previous scheme; identical results)
  auto dispatch = [&](std::vector<MatJob>& js) {
    if (!(legacy & 2)) { ::hive::cpu::matmul_multi(js.data(), (int)js.size(), nullptr, 0, 0); return; }
    for (int n = 1; n <= M; ++n) {
      std::vector<MatJob> b;
      for (const MatJob& j : js) if (j.M == n) b.push_back(j);
      if (!b.empty()) ::hive::cpu::matmul_multi(b.data(), (int)b.size(), nullptr, n, 0);
    }
  };
  if (legacy & 2) lin(w.sh_down, ys.data(), M, SI, sh.data(), H);
  else jobs.push_back(MatJob{MatView{&w.sh_down, 0, w.sh_down.N}, sh.data(), H, ys.data(), M, SI});
  for (int e = 0; e < E; ++e) {
    if (!cnt[e]) continue;
    const int r = off[e];
    jobs.push_back(MatJob{MatView{&w.gu, e * I, (e + 1) * I}, G.data() + (size_t)r * I, I, Xe.data() + (size_t)r * H, cnt[e], H});
    jobs.push_back(MatJob{MatView{&w.gu, E * I + e * I, E * I + (e + 1) * I}, U.data() + (size_t)r * I, I, Xe.data() + (size_t)r * H, cnt[e], H});
  }
  dispatch(jobs);
  if (!(legacy & 2)) par_for((size_t)M, 64, [&](size_t a, size_t b) { for (size_t m = a; m < b; ++m) simd::round_n(sh.data() + m * H, (size_t)H); });  // lin's rounding
  par_for((size_t)P, 2, [&](size_t a, size_t b) {
    for (size_t r = a; r < b; ++r) {
      const int e = pexp[r];
      simd::silu_mul_v(G.data() + r * I, U.data() + r * I, I, w.g_gate[e], w.g_up[e], Y.data() + r * I);
    }
  });
  jobs.clear();
  for (int e = 0; e < E; ++e) {
    if (!cnt[e]) continue;
    const int r = off[e];
    jobs.push_back(MatJob{MatView{&w.d, e * hh, (e + 1) * hh}, D2.data() + (size_t)r * H, H, Y.data() + (size_t)r * I, cnt[e], I});
    jobs.push_back(MatJob{MatView{&w.d, E * hh + e * hh, E * hh + (e + 1) * hh}, D2.data() + (size_t)r * H + hh, H, Y.data() + (size_t)r * I, cnt[e], I});
  }
  dispatch(jobs);
  par_for((size_t)P, 4, [&](size_t a, size_t b) {
    for (size_t r = a; r < b; ++r) {
      const int e = pexp[r];
      float* po = pair_out.data() + ((size_t)prow[r] * K + pslot[r]) * H;
      const __m256 gd = _mm256_set1_ps(w.g_down[e]);
      int d = 0;
      for (; d + 8 <= H; d += 8) _mm256_storeu_ps(po + d, simd::bf16r8(_mm256_mul_ps(_mm256_loadu_ps(D2.data() + r * H + d), gd)));
      for (; d < H; ++d) po[d] = bf16r(D2[r * H + d] * w.g_down[e]);
    }
  });
  // out = bf16(bf16(Σ_k w_k · y_k) + bf16(sigmoid(gate logit) · shared))  (Σ in k order, as the scalar form)
  par_for((size_t)M, 4, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const float g = sigm(front[m * FW + E]);
      const __m256 gv = _mm256_set1_ps(g);
      int d = 0;
      for (; d + 8 <= H; d += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < K; ++k)
          acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(wts[m * K + k]), _mm256_loadu_ps(pair_out.data() + (m * K + k) * H + d)));
        _mm256_storeu_ps(out + m * H + d, simd::bf16r8(_mm256_add_ps(simd::bf16r8(acc), simd::bf16r8(_mm256_mul_ps(gv, _mm256_loadu_ps(sh.data() + m * H + d))))));
      }
      for (; d < H; ++d) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc = acc + wts[m * K + k] * pair_out[(m * K + k) * H + d];
        out[m * H + d] = bf16r(bf16r(acc) + bf16r(g * sh[m * H + d]));
      }
    }
  });
}

// ---- rope coordinates (text: (p + d)·3; image spans: (st, st + i / gw, st + i % gw), see QwenEngine::set_rope) ---------------------------
void QwenCpuEngine::Impl::set_rope(const std::vector<RowInfo>& rows, bool prefill, std::vector<int32_t>& coords) {
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
void grow_attn(CpuSeq::Attn& a, int64_t tokens, int kvw, int id, int ratio) {
  if ((int64_t)a.k.size() < tokens * kvw) { a.k.resize((size_t)tokens * kvw); a.v.resize((size_t)tokens * kvw); }
  if ((int64_t)a.ik.size() < tokens * id) a.ik.resize((size_t)tokens * id);
  if ((int64_t)a.gk.size() < (tokens / ratio + 1) * id) a.gk.resize((size_t)(tokens / ratio + 1) * id);
}
}  // namespace

// layers [l0, n_layers) over the rows' hidden streams h [M, 10240] (rope coordinates already in `coords`)
void QwenCpuEngine::Impl::run_range(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, const std::vector<int32_t>& coords, float* h,
                                    int l0) {
  const int M = (int)rows.size(), H = c.hidden;
  std::vector<float> x((size_t)M * H), y((size_t)M * H), hn, dl;
  int li = 0, ai = 0;
  for (int l = 0; l < l0; ++l) (c.is_attn[l] ? ai : li)++;
  for (int l = l0; l < c.n_layers; ++l) {
    const LayerW& L = layers[l];
    if (prefill && l > 0 && layer_inject && *layer_inject) (*layer_inject)(l, h, M);
    double tc = hive::mono_ms();
    auto lap = [&](double& acc) { const double t = hive::mono_ms(); acc += t - tc; tc = t; };
    if (L.ple) { ple_block(rows, ids, h, prefill); lap(st.ms_ple); }
    hc_mix(L.ha, h, M, x.data(), hn, dl);
    lap(st.ms_hc);
    if (L.attn) attn_layer(L.att, ai++, false, rows, coords, x.data(), y.data());
    else lin_layer(L.lin, li++, rows, prefill, x.data(), y.data());
    lap(L.attn ? st.ms_attn : st.ms_lin);
    hc_combine(L.ha, h, y.data(), M, dl);
    hc_mix(L.hm, h, M, x.data(), hn, dl);
    lap(st.ms_hc);
    cur_layer = l;
    moe_layer(L.moe, M, x.data(), y.data());
    cur_layer = -1;
    lap(st.ms_moe);
    hc_combine(L.hm, h, y.data(), M, dl);
    lap(st.ms_hc);
    if (layer_hook && *layer_hook) (*layer_hook)(l, h, M);
    if (prefill && depth == 1 && layer_boundary && *layer_boundary && l + 1 < c.n_layers) {
      std::vector<EmbedOverride> parked = std::move(emb_over);  // inner forwards are other sequences' decode steps
      const int64_t base = emb_base;
      emb_over.clear(); emb_base = 0;
      (*layer_boundary)(l);
      emb_over = std::move(parked); emb_base = base;
    }
  }
}

void QwenCpuEngine::Impl::forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, float* logits, bool all_logits) {
  const int M = (int)rows.size(), H = c.hidden, HC = c.hcw();
  const double t0 = hive::mono_ms();
  for (const RowInfo& ri : rows) {
    CpuSeq& s = *ri.s;
    if (ri.pos + 1 > s.cap) throw std::runtime_error("qwen cpu: sequence capacity exceeded");
    for (auto& a : s.attn) grow_attn(a, ri.pos + 1, c.kv_width(), c.idx_dim, c.idx_ratio);
    if (mtp) grow_attn(s.mtp, ri.pos + 1 + 8, c.kv_width(), c.idx_dim, c.idx_ratio);
  }
  ++depth;
  struct Depth { int& d; ~Depth() { --d; } } dguard{depth};
  if ((int)h_stack.size() < depth) h_stack.resize(depth);
  std::vector<float>& hv = h_stack[depth - 1];
  hv.resize((size_t)M * HC);
  float* h = hv.data();
  std::vector<int32_t> coords;
  set_rope(rows, prefill, coords);
  // embeddings (image rows from the vision encoder) → every copy of the stream
  {
    std::vector<float> x((size_t)M * H);
    ppar((size_t)M, 64, [&](size_t a, size_t b) {
      for (size_t r = a; r < b; ++r) for (int d = 0; d < H; ++d) x[r * H + d] = b2f(embed[(size_t)ids[r] * H + d]);
    });
    if (prefill)
      for (const EmbedOverride& o : emb_over) {
        const int64_t a = std::max<int64_t>(o.row, emb_base), b = std::min<int64_t>(o.row + o.n, emb_base + M);
        for (int64_t g = a; g < b; ++g) memcpy(x.data() + (size_t)(g - emb_base) * H, o.emb.data() + (size_t)(g - o.row) * H, (size_t)H * 4);
      }
    ppar((size_t)M, 64, [&](size_t a, size_t b) {
      for (size_t r = a; r < b; ++r) for (int cc = 0; cc < c.hc; ++cc) memcpy(h + r * HC + (size_t)cc * H, x.data() + r * H, (size_t)H * 4);
    });
  }
  run_range(rows, ids, prefill, coords, h, 0);
  if (mtp) fh.assign(h, h + (size_t)M * HC);
  const int r0 = all_logits ? 0 : M - 1, nr = all_logits ? M : 1;
  double th = hive::mono_ms();
  if (logits) head(h + (size_t)r0 * HC, nr, logits);
  st.ms_head += hive::mono_ms() - th;
  th = hive::mono_ms();
  if (mtp && mtp_k > 0) mtp_after_forward(rows, ids, prefill);
  st.ms_mtp += hive::mono_ms() - th;
  for (int r = 0; r < M; ++r) { CpuSeq& s = *rows[r].s; s.pos = rows[r].pos + 1; s.tokens.push_back(ids[r]); }
  if (!prefill) ++st.steps;
  st.ms_total += hive::mono_ms() - t0;
}

QwenCpuEngine::Impl::~Impl() {
  if (route_dump) fclose(route_dump);
  for (auto& pm : i8_alt) ::hive::cpu::free_mat(pm.second);
  using ::hive::cpu::free_mat;
  auto hc = [](HcW& w) { free_mat(w.down_inj); free_mat(w.up); };
  auto moe = [](MoeW& w) { free_mat(w.front); free_mat(w.sh_down); free_mat(w.gu); free_mat(w.d); };
  auto layer = [&](LayerW& L) {
    hc(L.ha); hc(L.hm);
    free_mat(L.lin.in); free_mat(L.lin.out);
    free_mat(L.att.in); free_mat(L.att.out);
    moe(L.moe);
    free_mat(L.plew.key); free_mat(L.plew.value);
  };
  for (LayerW& L : layers) layer(L);
  hc(mixer);
  free_mat(lm_head);
  if (mtp) { free_mat(mtp->fc_emb); free_mat(mtp->fc_hidden); layer(mtp->layer); hc(mtp->mixer); }
  if (vision) {
    free_mat(vision->patch); free_mat(vision->m1); free_mat(vision->m2);
    for (auto& B : vision->blocks) { free_mat(B.qkv); free_mat(B.proj); free_mat(B.fc1); free_mat(B.fc2); }
  }
}

// ---- public API ------------------------------------------------------------------------------------------------------------------------
QwenCpuEngine::QwenCpuEngine(const std::string& ckpt_dir, bool load_mtp) : QwenCpuEngine(ckpt_dir, Options{load_mtp, 0, true}) {}
QwenCpuEngine::QwenCpuEngine(const std::string& ckpt_dir, const Options& o) : p_(std::make_unique<Impl>()) {
  p_->ckpt_dir = ckpt_dir;
  p_->layer_hook = &layer_hook;
  p_->layer_boundary = &layer_boundary;
  p_->layer_inject = &layer_inject;
  p_->first_layer = o.first_layer;
  p_->with_head = o.head;
  p_->load(o.load_mtp);
}
int QwenCpuEngine::first_layer() const { return p_->first_layer; }
size_t QwenCpuEngine::host_bytes() const {
  size_t b = p_->embed.size() * 2;
  auto mats = [&](const LayerW& L) {
    for (const Mat* m : {&L.ha.down_inj, &L.ha.up, &L.hm.down_inj, &L.hm.up, &L.lin.in, &L.lin.out, &L.att.in, &L.att.out, &L.moe.front, &L.moe.sh_down,
                         &L.moe.gu, &L.moe.d, &L.plew.key, &L.plew.value})
      b += m->bytes;
  };
  for (const LayerW& L : p_->layers) mats(L);
  b += p_->lm_head.bytes + p_->mixer.down_inj.bytes + p_->mixer.up.bytes;
  if (p_->ple) b += p_->ple->host_bytes();
  return b;
}

void QwenCpuEngine::run_layers(CpuSeq* const* seqs, const int64_t* pos, int M, const int32_t* ids, bool prefill, const int32_t* coords, float* h) {
  Impl& I = *p_;
  const QwenConfig& c = I.c;
  std::vector<RowInfo> rows(M);
  std::vector<int32_t> cd(coords, coords + (size_t)M * 3);
  for (int r = 0; r < M; ++r) {
    CpuSeq& s = *seqs[r];
    rows[r] = {&s, pos[r]};
    if (pos[r] + 1 > s.cap) throw std::runtime_error("qwen cpu: sequence capacity exceeded");
    int ai = 0;
    for (int l = 0; l < c.n_layers; ++l)
      if (c.is_attn[l]) { if (l >= I.first_layer) grow_attn(s.attn[ai], pos[r] + 1, c.kv_width(), c.idx_dim, c.idx_ratio); ++ai; }
    if ((int64_t)s.rope.size() < (pos[r] + 1) * 3) s.rope.resize((size_t)(pos[r] + 1) * 3);
    memcpy(s.rope.data() + (size_t)pos[r] * 3, coords + (size_t)r * 3, 12);
  }
  const double t0 = hive::mono_ms();
  try { I.run_range(rows, ids, prefill, cd, h, I.first_layer); } catch (...) { for (int r = 0; r < M; ++r) seqs[r]->broken = true; throw; }
  for (int r = 0; r < M; ++r) {
    CpuSeq& s = *seqs[r];
    if ((int64_t)s.tokens.size() > pos[r]) s.tokens.resize((size_t)pos[r]);   // a rolled-back / restarted position is rewritten
    s.tokens.push_back(ids[r]);
    s.pos = pos[r] + 1;
  }
  if (!prefill) ++I.st.steps;
  I.st.ms_total += hive::mono_ms() - t0;
}
QwenCpuEngine::~QwenCpuEngine() = default;
const QwenConfig& QwenCpuEngine::cfg() const { return p_->c; }
CpuStats& QwenCpuEngine::stats() { return p_->st; }
bool QwenCpuEngine::mtp_on() const { return p_->mtp && p_->mtp_k > 0; }
int QwenCpuEngine::mtp_k() const { return p_->mtp_k; }
// ---- I8 A/B: an I8 twin of every dense matrix of the chosen categories, swapped in and out by set_dense_i8 --------------------------------
namespace {
Mat i8_of(const Mat& m, int group) {
  std::vector<uint16_t> w((size_t)m.N * m.K);
  for (int k = 0; k < m.nodes; ++k) {
    const int r0 = m.row0[k], r1 = m.nodes > 1 ? m.row0[k + 1] : m.N;
    if (r1 > r0) memcpy(w.data() + (size_t)r0 * m.K, m.w[k], (size_t)(r1 - r0) * m.K * 2);
  }
  return ::hive::cpu::make_mat_i8(m.N, m.K, w.data(), group);
}
}  // namespace

void QwenCpuEngine::Impl::build_i8_alt(const std::string& cats) {
  auto want = [&](const char* c) {
    if (cats == "all" || cats == "1") return strcmp(c, "moe") != 0;
    return ("," + cats + ",").find(std::string(",") + c + ",") != std::string::npos;
  };
  const int g = getenv("HIVE_QWEN_CPU_I8_GROUP") ? std::max(8, atoi(getenv("HIVE_QWEN_CPU_I8_GROUP")) / 8 * 8) : 64;
  auto add = [&](Mat& m, const char* c) { if (m.t == WType::BF16 && m.N > 0 && want(c) && m.K % g == 0) i8_alt.push_back({&m, i8_of(m, g)}); };
  auto hc = [&](HcW& w) { add(w.down_inj, "hc"); add(w.up, "hc"); };
  auto layer = [&](LayerW& L) {
    hc(L.ha); hc(L.hm);
    add(L.lin.in, "gdn"); add(L.lin.out, "gdn"); add(L.att.in, "attn"); add(L.att.out, "attn");
    add(L.moe.front, "moe"); add(L.moe.sh_down, "moe"); add(L.plew.key, "ple"); add(L.plew.value, "ple");
  };
  for (LayerW& L : layers) layer(L);
  hc(mixer);
  add(lm_head, "head");
  if (mtp) { add(mtp->fc_emb, "mtp"); add(mtp->fc_hidden, "mtp"); layer(mtp->layer); hc(mtp->mixer); }
  size_t b16 = 0, b8 = 0;
  for (auto& [p, m] : i8_alt) { b16 += p->bytes; b8 += m.bytes; }
  fprintf(stderr, "[qwen-cpu] I8 A/B twins of %zu dense matrices (%s, group %d): %.2f GB bf16 → %.2f GB i8\n", i8_alt.size(), cats.c_str(), g, b16 / 1e9,
          b8 / 1e9);
}

void QwenCpuEngine::set_dense_i8(bool on) {
  Impl& I = *p_;
  if (on == I.i8_on) return;
  for (auto& [p, m] : I.i8_alt) std::swap(*p, m);
  I.i8_on = on;
}
bool QwenCpuEngine::has_dense_i8() const { return !p_->i8_alt.empty(); }

void QwenCpuEngine::set_ab_legacy(int mask) { p_->legacy = mask; ::hive::cpu::set_dispatch_legacy(mask & 1); }
void QwenCpuEngine::set_embed_override(std::vector<EmbedOverride> v) { p_->emb_over = std::move(v); }

void QwenCpuEngine::init_seq(CpuSeq& s, int64_t cap) {
  const QwenConfig& c = p_->c;
  s.cap = cap;
  int nl = 0, na = 0;
  for (int l = 0; l < c.n_layers; ++l) (c.is_attn[l] ? na : nl)++;
  s.conv.assign(nl, {}); s.state.assign(nl, {});
  for (int l = 0, i = 0; l < c.n_layers; ++l) {   // states only for the layers this engine runs (all, or a hybrid tail)
    if (c.is_attn[l]) continue;
    if (l >= p_->first_layer) { s.conv[i].assign((size_t)3 * c.lin_conv_ch(), 0.f); s.state[i].assign((size_t)c.lin_v_heads * c.lin_k_dim * c.lin_v_dim, 0.f); }
    ++i;
  }
  s.attn.assign(na, {});
  reset_seq(s);
}

void QwenCpuEngine::reset_seq(CpuSeq& s) {
  for (auto& v : s.conv) std::fill(v.begin(), v.end(), 0.f);
  for (auto& v : s.state) std::fill(v.begin(), v.end(), 0.f);
  s.ple_state.assign(p_->ple_layer >= 0 ? (size_t)kPleConvState * p_->c.hcw() : 0, 0.f);
  s.pos = 0; s.tokens.clear(); s.rope_delta = 0; s.rope.clear(); s.broken = false;
  for (auto& a : s.attn) a.gkt_n = 0;
  s.mtp.gkt_n = 0;
  s.h_last.assign(p_->mtp ? (size_t)p_->c.hcw() : 0, 0.f); s.h_last_valid = false; s.mtp_written = 0;
}

void QwenCpuEngine::prefill(CpuSeq& s, const int32_t* ids, int T, float* logits) {
  if (T <= 0) return;
  if ((int64_t)s.tokens.size() != s.pos) throw std::runtime_error("qwen cpu: sequence tokens out of step with its position");
  if (p_->ple && strcmp(p_->ple->kind(), "ram") != 0) {  // SSD: read the prompt's PLE rows ahead
    std::vector<int64_t> pr((size_t)T * kPleHeads);
    p_->ple_hash.rows_of(ids, T, s.pos >= 2 ? s.tokens[s.pos - 2] : -1, s.pos >= 1 ? s.tokens[s.pos - 1] : -1, pr.data());
    p_->ple->prefetch(pr.data(), T);
  }
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
  p_->emb_over.clear();  // the overrides belong to this prefill only
}

void QwenCpuEngine::decode(CpuSeq* const* seqs, const int32_t* ids, int M, float* logits) {
  std::vector<RowInfo> rows(M);
  for (int m = 0; m < M; ++m) rows[m] = {seqs[m], seqs[m]->pos};
  try { p_->forward(rows, ids, false, logits, true); } catch (...) { for (int m = 0; m < M; ++m) seqs[m]->broken = true; throw; }
}

void QwenCpuEngine::verify_batch(CpuSeq* const* seqs, const int* Ms, int S, const int32_t* ids, float* logits) {
  Impl& I = *p_;
  I.vparts.clear();
  std::vector<RowInfo> rows;
  int R = 0;
  for (int p = 0; p < S; ++p) {
    if (Ms[p] < 1 || R + Ms[p] > 8) throw std::runtime_error("qwen cpu verify rows");
    I.vparts.push_back({seqs[p], seqs[p]->pos, Ms[p], R});
    for (int r = 0; r < Ms[p]; ++r) rows.push_back({seqs[p], seqs[p]->pos + r});
    R += Ms[p];
  }
  // snapshot buffers: [part][row < M−1]
  I.snaps.resize(S);
  for (int p = 0; p < S; ++p) {
    I.snaps[p].resize(std::max(0, Ms[p] - 1));
    for (auto& sn : I.snaps[p]) {
      sn.conv.resize(seqs[p]->conv.size()); sn.state.resize(seqs[p]->state.size());
      for (size_t i = 0; i < sn.conv.size(); ++i) { sn.conv[i].resize(seqs[p]->conv[i].size()); sn.state[i].resize(seqs[p]->state[i].size()); }
      sn.ple.resize(seqs[p]->ple_state.size());
    }
  }
  I.verify_ids.assign(ids, ids + R);
  I.verify = true;
  try { I.forward(rows, ids, false, logits, true); } catch (...) { I.verify = false; I.vparts.clear(); for (int p = 0; p < S; ++p) seqs[p]->broken = true; throw; }
  I.verify = false;
}

void QwenCpuEngine::rollback_batch(const int* n_keep, int S) {
  Impl& I = *p_;
  if (S != (int)I.vparts.size()) throw std::runtime_error("qwen cpu rollback without a matching verify");
  const int HC = I.c.hcw();
  for (int p = 0; p < S; ++p) {
    const Impl::VPart& vp = I.vparts[p];
    CpuSeq& s = *vp.s;
    const int M = vp.M, nk = n_keep[p];
    if (nk < 1 || nk > M || s.pos != vp.pos0 + M) throw std::runtime_error("qwen cpu rollback without a matching verify");
    if (nk < M) {
      const Impl::Snap& sn = I.snaps[p][nk - 1];
      for (size_t i = 0; i < s.conv.size(); ++i) { s.conv[i] = sn.conv[i]; s.state[i] = sn.state[i]; }
      s.ple_state = sn.ple;
    }
    s.pos = vp.pos0 + nk;
    s.tokens.resize(s.tokens.size() - (size_t)(M - nk));
    if (I.mtp && I.mtp_k > 0) {
      if (nk >= 2) I.mtp_kv(s, vp.pos0, nk - 1, I.fh.data() + (size_t)vp.r0 * HC, I.verify_ids.data() + vp.r0 + 1);
      memcpy(s.h_last.data(), I.fh.data() + (size_t)(vp.r0 + nk - 1) * HC, (size_t)HC * 4);
      s.h_last_valid = true;
    }
  }
  I.vparts.clear();
}

int QwenCpuEngine::mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf) { return p_->mtp_draft(s, tok, k, drafts, conf); }
bool QwenCpuEngine::has_vision() const { return p_->c.vision_layers > 0 && p_->ck && p_->ck->has("model.visual.patch_embed.proj.weight"); }

}  // namespace hive::qwen::cpu
