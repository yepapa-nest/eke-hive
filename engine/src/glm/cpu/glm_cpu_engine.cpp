// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash on the CPU — loading and the layer forward (hive/glm/glm_cpu.h). Arithmetic and bf16 rounding points follow the GPU family
//   (glm_engine.cpp, kda.h, dsa.h, glm_moe.cu, glm_kernels.cu, the shared hyper-connection kernels), so the engine's hc streams can be
//   compared with oracle/glm5_oracle.py layer by layer.
#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "glm_cpu_internal.h"
#include "hive/clock.h"
#include "nlohmann/json.hpp"

namespace hive::glm::cpu {

using ::hive::cpu::Pool;
using ::hive::cpu::SType;
using ::hive::cpu::WType;

namespace {
inline float hsum8(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}
inline float dotf(const float* a, const float* b, int n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  float s = hsum8(_mm256_add_ps(s0, s1));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
inline void bf16_row(const uint16_t* src, int n, float* dst) {
  int d = 0;
  for (; d + 8 <= n; d += 8)
    _mm256_storeu_ps(dst + d, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(src + d))), 16)));
  for (; d < n; ++d) dst[d] = b2f(src[d]);
}
inline float silu_lim_act(float g, float u, float lim) {  // GPU swiglu: inputs already bf16; clamp; bf16(silu(g)·u)
  if (lim > 0.f) { u = std::fmin(std::fmax(u, -lim), lim); g = std::fmin(g, lim); }
  return bf16r(g / (1.f + std::exp(-g)) * u);
}
}  // namespace

// ---- helpers ------------------------------------------------------------------------------------------------------------------------------
void par_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f) {
  if (n == 0) return;
  const size_t nb = (n + grain - 1) / grain;
  if (nb <= 1) { f(0, n); return; }
  Pool::get().run((int)nb, [&](int b) { const size_t a = (size_t)b * grain; f(a, std::min(n, a + grain)); });
}

void round_rows(float* Y, int M, int ldy, int N) {
  par_for((size_t)M, std::max<size_t>(1, 262144 / std::max(1, N)), [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      float* y = Y + m * ldy;
      for (int i = 0; i < N; ++i) y[i] = bf16r(y[i]);
    }
  });
}

void lin(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy, bool round) {
  ::hive::cpu::matmul(W, X, M, ldx, Y, ldy);
  if (round) round_rows(Y, M, ldy, W.r1 - W.r0);
}

void rmsnorm_rows(const float* x, int ldx, const float* w, int M, int K, float eps, float* out, int ldo) {
  par_for((size_t)M, std::max<size_t>(1, 65536 / K), [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const float* xr = x + m * ldx;
      float* o = out + m * ldo;
      const float rs = 1.f / std::sqrt(dotf(xr, xr, K) / (float)K + eps);
      for (int i = 0; i < K; ++i) o[i] = bf16r(w[i] * (xr[i] * rs));
    }
  });
}

std::vector<float> load_vec(Checkpoint& ck, const std::string& name) {
  const TensorInfo& t = ck.get(name);
  std::vector<float> v((size_t)t.numel());
  if (t.dtype == "BF16") { const uint16_t* p = (const uint16_t*)t.data; for (size_t i = 0; i < v.size(); ++i) v[i] = b2f(p[i]); }
  else if (t.dtype == "F32") std::memcpy(v.data(), t.data, v.size() * 4);
  else throw std::runtime_error("glm cpu: unexpected dtype " + t.dtype + " for " + name);
  return v;
}

Mat load_bf16_mat(Checkpoint& ck, const std::vector<std::string>& names) {
  int N = 0, K = -1;
  std::vector<const TensorInfo*> ts;
  for (const auto& n : names) {
    const TensorInfo& t = ck.get(n);
    if (t.dtype != "BF16" || t.shape.size() != 2) throw std::runtime_error("glm cpu: expected a BF16 matrix: " + n + " (" + t.dtype + ")");
    if (K >= 0 && K != (int)t.shape[1]) throw std::runtime_error("glm cpu: stacked widths differ: " + n);
    K = (int)t.shape[1]; N += (int)t.shape[0];
    ts.push_back(&t);
  }
  if (ts.size() == 1) return ::hive::cpu::make_mat(WType::BF16, N, K, ts[0]->data);
  std::vector<uint8_t> tmp((size_t)N * K * 2);
  size_t off = 0;
  for (auto* t : ts) { std::memcpy(tmp.data() + off, t->data, t->nbytes); off += t->nbytes; }
  return ::hive::cpu::make_mat(WType::BF16, N, K, tmp.data());
}

// ---- config -------------------------------------------------------------------------------------------------------------------------------
CpuConfig CpuConfig::load(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) throw std::runtime_error("no config.json in " + dir);
  const nlohmann::json all = nlohmann::json::parse(f);
  const nlohmann::json& t = all.contains("text_config") ? all["text_config"] : all;
  CpuConfig c;
  auto gi = [&](const char* k, int& v) { if (t.contains(k) && t[k].is_number()) v = t[k].get<int>(); };
  auto gf = [&](const char* k, float& v) { if (t.contains(k) && t[k].is_number()) v = t[k].get<float>(); };
  gi("hidden_size", c.hidden); gi("num_hidden_layers", c.n_layers); gi("vocab_size", c.vocab); gi("num_attention_heads", c.n_heads);
  gi("q_lora_rank", c.q_lora); gi("kv_lora_rank", c.kv_lora); gi("qk_head_dim", c.qk_head); gi("v_head_dim", c.v_head);
  gi("index_n_heads", c.index_heads); gi("index_head_dim", c.index_dim); gi("index_topk", c.index_topk); gi("index_kpool", c.index_kpool);
  gi("intermediate_size", c.dense_inter); gi("moe_intermediate_size", c.moe_inter); gi("n_routed_experts", c.n_routed);
  gi("num_experts_per_tok", c.n_act); gi("hc_mult", c.hc); gi("hc_sinkhorn_iters", c.sinkhorn_iters); gi("num_nextn_predict_layers", c.n_mtp);
  gf("routed_scaling_factor", c.routed_scale); gf("swiglu_limit", c.swiglu_limit); gf("rms_norm_eps", c.rms_eps); gf("hc_eps", c.hc_eps);
  if (t.contains("eos_token_id") && t["eos_token_id"].is_array()) c.eos_ids = t["eos_token_id"].get<std::vector<int>>();
  if (!t.contains("linear_attn_config")) throw std::runtime_error("config: linear_attn_config missing (not a GLM-5.3-Flash checkpoint?)");
  const nlohmann::json& la = t["linear_attn_config"];
  if (la.contains("num_heads")) c.kda_heads = la["num_heads"].get<int>();
  if (la.contains("head_dim")) c.kda_dim = la["head_dim"].get<int>();
  if (la.contains("short_conv_kernel_size")) c.conv_k = la["short_conv_kernel_size"].get<int>();
  if (la.contains("gate_lower_bound")) c.kda_lower_bound = la["gate_lower_bound"].get<float>();
  c.is_kda.assign(c.n_layers, false);
  for (int l : la["kda_layers"].get<std::vector<int>>()) if (l < c.n_layers) c.is_kda[l] = true;
  int first_dense = 3;
  if (t.contains("first_k_dense_replace")) first_dense = t["first_k_dense_replace"].get<int>();
  c.is_moe.assign(c.n_layers, false);
  for (int l = first_dense; l < c.n_layers; ++l) c.is_moe[l] = true;
  if (all.contains("vision_config")) {
    const nlohmann::json& v = all["vision_config"];
    auto vi = [&](const char* k, int& x) { if (v.contains(k)) x = v[k].get<int>(); };
    vi("depth", c.vision_layers); vi("hidden_size", c.vision_dim); vi("num_heads", c.vision_heads); vi("intermediate_size", c.vision_inter);
    vi("patch_size", c.vision_patch); vi("spatial_merge_size", c.vision_merge); vi("temporal_patch_size", c.vision_temporal);
    vi("out_hidden_size", c.vision_out); vi("projection_intermediate_size", c.vision_proj_inter);
  }
  if (all.contains("image_token_id")) c.image_token_id = all["image_token_id"].get<int>();
  if (c.hc != 4 || c.kda_dim != 128 || c.kv_lora != 512 || c.qk_head != 256 || c.v_head != 256 || c.index_dim != 128 || c.index_kpool != 4)
    throw std::runtime_error("glm cpu: unsupported geometry (hc 4, KDA head 128, latent 512, heads 256, indexer 128, pool 4)");
  return c;
}

// ---- loading -------------------------------------------------------------------------------------------------------------------------------
namespace {
const std::string P = "model.language_model.";

void load_hc(Checkpoint& ck, const std::string& p, const char* which, HcW& w) {
  std::vector<float> fn = load_vec(ck, p + "hc_" + which + "_fn");
  const TensorInfo& t = ck.get(p + "hc_" + which + "_fn");
  w.fn = ::hive::cpu::make_mat(WType::F32, (int)t.shape[0], (int)t.shape[1], fn.data());
  w.base = load_vec(ck, p + "hc_" + which + "_base");
  w.scale = load_vec(ck, p + "hc_" + which + "_scale");
}

Mat load_nvfp4(Checkpoint& ck, const std::string& b) {
  const TensorInfo& tw = ck.get(b + ".weight");
  const TensorInfo& ts = ck.get(b + ".weight_scale");
  const TensorInfo& tg = ck.get(b + ".weight_scale_2");
  if (tw.dtype != "U8" || ts.dtype != "F8_E4M3" || tg.dtype != "F32") throw std::runtime_error("glm cpu: expected NVFP4: " + b);
  float g; std::memcpy(&g, tg.data, 4);
  return ::hive::cpu::make_mat(WType::NVFP4, (int)tw.shape[0], (int)tw.shape[1] * 2, tw.data, ts.data, SType::E4M3, g);
}
}  // namespace

Mat GlmCpuEngine::Impl::mat_fp8ok(const std::string& name) {
  if (f8 && f8->has(name) && f8->has(name + "_scale_inv")) {
    const TensorInfo& t = f8->get(name);
    const TensorInfo& s = f8->get(name + "_scale_inv");
    if (t.dtype != "F8_E4M3" || t.shape.size() != 2 || s.dtype != "F32") throw std::runtime_error("glm cpu fp8: unexpected tensor " + name);
    return ::hive::cpu::make_mat(WType::FP8, (int)t.shape[0], (int)t.shape[1], t.data, s.data, SType::F32, 1.f, 128, 128);
  }
  return load_bf16_mat(*ck, {name});
}

void GlmCpuEngine::Impl::load_dsa(Checkpoint& k, const std::string& a, DsaW& w) {
  w.q_a = mat_fp8ok(a + "q_a_proj.weight"); w.q_b = mat_fp8ok(a + "q_b_proj.weight");
  w.kv_a = mat_fp8ok(a + "kv_a_proj_with_mqa.weight"); w.o = mat_fp8ok(a + "o_proj.weight");
  w.kv_b = load_bf16_mat(k, {a + "kv_b_proj.weight"});
  w.q_a_norm = load_vec(k, a + "q_a_layernorm.weight"); w.kv_a_norm = load_vec(k, a + "kv_a_layernorm.weight");
  const std::string i = a + "indexer.";
  w.ix = load_bf16_mat(k, {i + "wk.weight", i + "index_kpool_compress_gate", i + "weights_proj.weight"});
  w.wq_b = load_bf16_mat(k, {i + "wq_b.weight"});
  w.k_norm_w = load_vec(k, i + "k_norm.weight"); w.k_norm_b = load_vec(k, i + "k_norm.bias"); w.ape = load_vec(k, i + "index_kpool_compress_ape");
  // W_UKᵀ per head: ukt[h·512 + l][d] = kv_b[h·512 + d][l] (d < 256 = k_nope rows of head h) — q_lat[h] = W_UKᵀ q[h]
  const TensorInfo& t = k.get(a + "kv_b_proj.weight");
  const int NH = c.n_heads, L = c.kv_lora, D = c.qk_head, R = D + c.v_head;
  if ((int)t.shape[0] != NH * R || (int)t.shape[1] != L) throw std::runtime_error("glm cpu: kv_b_proj shape " + a);
  const uint16_t* src = (const uint16_t*)t.data;
  std::vector<uint16_t> tr((size_t)NH * L * D);
  par_for((size_t)NH, 1, [&](size_t h0, size_t h1) {
    for (size_t h = h0; h < h1; ++h)
      for (int d = 0; d < D; ++d)
        for (int l = 0; l < L; ++l) tr[((size_t)h * L + l) * D + d] = src[((size_t)h * R + d) * L + l];
  });
  w.ukt = ::hive::cpu::make_mat(WType::BF16, NH * L, D, tr.data());
}

void GlmCpuEngine::Impl::load_moe(const std::string& p, MoeW& w, bool mtp_layer) {
  const int E = c.n_routed, H = c.hidden, I = c.moe_inter;
  w.router = load_bf16_mat(*ck, {p + "gate.weight"});
  w.bias = load_vec(*ck, p + "gate.e_score_correction_bias");
  w.sh_g = mat_fp8ok(p + "shared_experts.gate_proj.weight");
  w.sh_u = mat_fp8ok(p + "shared_experts.up_proj.weight");
  w.sh_d = mat_fp8ok(p + "shared_experts.down_proj.weight");
  w.g_gate.assign(E, 1.f); w.g_up.assign(E, 1.f); w.g_down.assign(E, 1.f);
  if (ext_experts && !mtp_layer) return;  // hybrid: the routed experts are computed by the caller's expert store
  const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
  auto name = [&](int e, int j) { return p + "experts." + std::to_string(e) + "." + proj[j] + ".weight"; };
  if (!mtp_layer) {
    const size_t gw = (size_t)I * H / 2, gs = (size_t)I * H / 16, dw = (size_t)H * I / 2, ds = (size_t)H * I / 16;
    std::vector<uint8_t> GW(2 * E * gw), GS(2 * E * gs), DW(E * dw), DS(E * ds);
    for (int e = 0; e < E; ++e) {
      for (int j = 0; j < 3; ++j) {
        const std::string b = name(e, j);
        const TensorInfo& tw = ck->get(b);
        const TensorInfo& ts = ck->get(b.substr(0, b.size() - 7) + ".weight_scale");
        const TensorInfo& tg = ck->get(b.substr(0, b.size() - 7) + ".weight_scale_2");
        if (tw.dtype != "U8" || ts.dtype != "F8_E4M3" || tw.nbytes != (j < 2 ? gw : dw) || ts.nbytes != (j < 2 ? gs : ds))
          throw std::runtime_error("glm cpu: expert must be NVFP4 " + b);
        float g; std::memcpy(&g, tg.data, 4);
        if (j < 2) {
          std::memcpy(GW.data() + ((size_t)j * E + e) * gw, tw.data, gw);
          std::memcpy(GS.data() + ((size_t)j * E + e) * gs, ts.data, gs);
          (j == 0 ? w.g_gate : w.g_up)[e] = g;
        } else {
          const size_t hw = dw / 2, hs = ds / 2;
          std::memcpy(DW.data() + (size_t)e * hw, tw.data, hw);
          std::memcpy(DW.data() + (size_t)E * hw + (size_t)e * hw, tw.data + hw, hw);
          std::memcpy(DS.data() + (size_t)e * hs, ts.data, hs);
          std::memcpy(DS.data() + (size_t)E * hs + (size_t)e * hs, ts.data + hs, hs);
          w.g_down[e] = g;
        }
      }
    }
    w.gu = ::hive::cpu::make_mat(WType::NVFP4, 2 * E * I, H, GW.data(), GS.data(), SType::E4M3, 1.f);
    w.d = ::hive::cpu::make_mat(WType::NVFP4, E * H, I, DW.data(), DS.data(), SType::E4M3, 1.f);
    return;
  }
  // NextN routed experts: the Z.ai FP8 copy (fp32 128×128 block scales "weight_scale_inv")
  const size_t gw = (size_t)I * H, dw = (size_t)H * I;
  const int gsr = I / 128, gsc = H / 128, dsr = H / 128, dsc = I / 128;
  if (I % 128 || H % 256) throw std::runtime_error("glm cpu: FP8 experts need 128-row blocks per half");
  std::vector<uint8_t> GW(2 * E * gw), DW(E * dw);
  std::vector<float> GS((size_t)2 * E * gsr * gsc), DS((size_t)E * dsr * dsc);
  for (int e = 0; e < E; ++e)
    for (int j = 0; j < 3; ++j) {
      const TensorInfo& tw = f8->get(name(e, j));
      const TensorInfo& ts = f8->get(name(e, j) + "_scale_inv");
      if (tw.dtype != "F8_E4M3" || ts.dtype != "F32") throw std::runtime_error("glm cpu: NextN expert must be FP8 " + name(e, j));
      if (j < 2) {
        std::memcpy(GW.data() + ((size_t)j * E + e) * gw, tw.data, gw);
        std::memcpy(GS.data() + ((size_t)j * E + e) * gsr * gsc, ts.data, (size_t)gsr * gsc * 4);
      } else {
        const size_t hw = dw / 2, hr = (size_t)dsr / 2 * dsc;
        std::memcpy(DW.data() + (size_t)e * hw, tw.data, hw);
        std::memcpy(DW.data() + (size_t)E * hw + (size_t)e * hw, tw.data + hw, hw);
        std::memcpy(DS.data() + (size_t)e * hr, ts.data, hr * 4);
        std::memcpy(DS.data() + (size_t)E * hr + (size_t)e * hr, (const uint8_t*)ts.data + hr * 4, hr * 4);
      }
    }
  w.gu = ::hive::cpu::make_mat(WType::FP8, 2 * E * I, H, GW.data(), GS.data(), SType::F32, 1.f, 128, 128);
  w.d = ::hive::cpu::make_mat(WType::FP8, E * H, I, DW.data(), DS.data(), SType::F32, 1.f, 128, 128);
}

void GlmCpuEngine::Impl::load(bool load_mtp) {
  const double t0 = hive::mono_ms();
  c = CpuConfig::load(ckpt_dir);
  ck = std::make_unique<Checkpoint>(ckpt_dir);
  if (const char* fd = getenv("HIVE_GLM_FP8_DIR"); fd && *fd && strcmp(fd, "0") != 0) {
    f8 = std::make_unique<Checkpoint>(fd);
    fprintf(stderr, "[glm-cpu] fp8 tensors from %s\n", fd);
  }
  if (const char* v = getenv("HIVE_GLM_CPU_CHUNK"); v && atoi(v) > 0) max_chunk = atoi(v);
  if (hybrid) { last = last < 0 ? c.n_layers - 1 : std::min(last, c.n_layers - 1); first = std::max(0, std::min(first, last)); }
  else first = 0, last = c.n_layers - 1;
  if (const char* v = getenv("HIVE_GLM_CPU_LAYERS"); !hybrid && v && *v) {
    int a = 0, b = 0;
    if (sscanf(v, "%d:%d", &a, &b) == 2 && a >= 0 && b < c.n_layers && a <= b) { first = a; last = b; }
  }
  kda_index.assign(c.n_layers, -1); dsa_index.assign(c.n_layers, -1);
  for (int l = 0; l < c.n_layers; ++l) { if (c.is_kda[l]) kda_index[l] = n_kda++; else dsa_index[l] = n_dsa++; }
  layers.resize(c.n_layers);
  for (int l = first; l <= last; ++l) {
    LayerW& L = layers[l];
    const std::string p = P + "layers." + std::to_string(l) + ".";
    L.kda = c.is_kda[l]; L.moe = c.is_moe[l]; L.loaded = true;
    L.in_norm = load_vec(*ck, p + "input_layernorm.weight"); L.post_norm = load_vec(*ck, p + "post_attention_layernorm.weight");
    load_hc(*ck, p, "attn", L.ha); load_hc(*ck, p, "ffn", L.hf);
    const std::string a = p + "self_attn.";
    if (L.kda) {
      KdaW& w = L.kw;
      w.in = load_bf16_mat(*ck, {a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight", a + "f_a_proj.weight", a + "g_a_proj.weight",
                                 a + "b_proj.weight"});
      w.fa_dim = (int)ck->get(a + "f_a_proj.weight").shape[0]; w.ga_dim = (int)ck->get(a + "g_a_proj.weight").shape[0];
      w.f_b = load_bf16_mat(*ck, {a + "f_b_proj.weight"}); w.g_b = load_bf16_mat(*ck, {a + "g_b_proj.weight"});
      w.o = load_bf16_mat(*ck, {a + "o_proj.weight"});
      w.conv_w.clear();
      for (const char* n : {"q_conv1d", "k_conv1d", "v_conv1d"}) { auto v = load_vec(*ck, a + n + ".weight"); w.conv_w.insert(w.conv_w.end(), v.begin(), v.end()); }
      w.A_log = load_vec(*ck, a + "A_log"); w.dt_bias = load_vec(*ck, a + "dt_bias"); w.o_norm = load_vec(*ck, a + "o_norm.weight");
    } else {
      load_dsa(*ck, a, L.dw);
    }
    if (L.moe) load_moe(p + "mlp.", L.mw, false);
    else for (int j = 0; j < 3; ++j) {
      const char* n[3] = {"gate_proj", "up_proj", "down_proj"};
      Mat m = load_nvfp4(*ck, p + "mlp." + n[j]);
      (j == 0 ? L.dense.g : j == 1 ? L.dense.u : L.dense.d) = m;
    }
    if ((l + 1) % 9 == 0 || l == last) fprintf(stderr, "[glm-cpu] layers %d..%d loaded (%.1f s)\n", first, l, (hive::mono_ms() - t0) / 1000);
  }
  if (!hybrid) {
    const TensorInfo& e = ck->get(P + "embed_tokens.weight");
    embed.resize((size_t)e.numel());
    std::memcpy(embed.data(), e.data, e.nbytes);
  }
  if (last == c.n_layers - 1 && !hybrid) {
    lm_head = load_bf16_mat(*ck, {"lm_head.weight"});
    final_norm = load_vec(*ck, P + "norm.weight");
    const std::string p = P + "layers." + std::to_string(c.n_layers) + ".";
    if (load_mtp && c.n_mtp > 0 && ck->has(p + "eh_proj.weight")) {
      const std::string e0 = p + "mlp.experts.0.gate_proj.weight";
      if (f8 && f8->has(e0) && f8->has(e0 + "_scale_inv")) {
        mtp = std::make_unique<MtpW>();
        mtp->enorm = load_vec(*ck, p + "enorm.weight"); mtp->hnorm = load_vec(*ck, p + "hnorm.weight");
        mtp->head_norm = load_vec(*ck, p + "shared_head.norm.weight");
        mtp->in_norm = load_vec(*ck, p + "input_layernorm.weight"); mtp->post_norm = load_vec(*ck, p + "post_attention_layernorm.weight");
        mtp->eh = load_bf16_mat(*ck, {p + "eh_proj.weight"});
        load_dsa(*ck, p + "self_attn.", mtp->dw);
        load_moe(p + "mlp.", mtp->mw, true);
        mtp_k = getenv("HIVE_GLM_MTP_K") ? std::max(0, std::min(7, atoi(getenv("HIVE_GLM_MTP_K")))) : 3;
      } else {
        fprintf(stderr, "[glm-cpu] MTP off: the NextN layer's routed experts need the Z.ai FP8 copy (HIVE_GLM_FP8_DIR, scripts/glm-fetch-fp8.py --mtp-experts)\n");
      }
    }
    // HIVE_GLM_DRAFT_VOCAB (as the GPU family): the draft head scores only these rows of lm_head (a copy here); verify uses the full head
    if (const char* dv = getenv("HIVE_GLM_DRAFT_VOCAB"); mtp && dv && *dv && strcmp(dv, "0") != 0) {
      std::vector<int32_t> rows;
      if (FILE* f = fopen(dv, "rb")) { int32_t v; while (fread(&v, 4, 1, f) == 1) if (v >= 0 && v < c.vocab) rows.push_back(v); fclose(f); }
      std::sort(rows.begin(), rows.end());
      rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
      if (!rows.empty() && (int)rows.size() < c.vocab) {
        const TensorInfo& t = ck->get("lm_head.weight");
        std::vector<uint16_t> tmp(rows.size() * (size_t)c.hidden);
        for (size_t i = 0; i < rows.size(); ++i) std::memcpy(tmp.data() + i * c.hidden, t.data + (size_t)rows[i] * c.hidden * 2, (size_t)c.hidden * 2);
        draft_head = ::hive::cpu::make_mat(WType::BF16, (int)rows.size(), c.hidden, tmp.data());
        draft_rows = std::move(rows);
        fprintf(stderr, "[glm-cpu] MTP draft head: %zu of %d tokens (%s)\n", draft_rows.size(), c.vocab, dv);
      } else {
        fprintf(stderr, "[glm-cpu] HIVE_GLM_DRAFT_VOCAB=%s: no usable token ids — the draft head scores the whole vocabulary\n", dv);
      }
    }
  }
  if (const char* rd = getenv("HIVE_GLM_CPU_ROUTE_DUMP"); rd && *rd) route_dump = fopen(rd, "w");
  if (!hybrid && !(getenv("HIVE_GLM_CPU_KEEP_PAGES") && atoi(getenv("HIVE_GLM_CPU_KEEP_PAGES")) > 0)) {
    ck->release_all();
    if (f8) f8->release_all();
  }
  fprintf(stderr, "[glm-cpu] model loaded in %.1f s · layers %d..%d · %d threads on %d NUMA node(s) · MTP k %d · fp8 %s · chunk %d\n",
          (hive::mono_ms() - t0) / 1000, first, last, Pool::get().threads(), Pool::get().nodes(), mtp ? mtp_k : 0, f8 ? "on" : "off", max_chunk);
}

// ---- hyper-connections ---------------------------------------------------------------------------------------------------------------------
// mixes = fn · h (fp32), r = rsqrt(mean(h²) + eps); pre = sigmoid(mix·r·s0 + b) + hc_eps, post = 2·sigmoid(…), comb = Sinkhorn(softmax rows
//   + hc_eps); x = bf16(Σ_c pre_c · h_c); xn = rmsnorm(x)·w  (model_kernels.cu sinkhorn_one / hc_pre, glm_engine.cpp hc_pre_norm)
void GlmCpuEngine::Impl::hc_pre(const HcW& w, const std::vector<float>& norm, const float* h, int M, float* xn, float* post, float* comb) {
  const int H = c.hidden, HC = c.hc * H, hc = c.hc, mixn = (2 + hc) * hc;
  thread_local std::vector<float> mix_tl, x_tl;
  auto& mix = mix_tl; auto& x = x_tl;
  mix.resize((size_t)M * mixn); x.resize((size_t)M * H);
  ::hive::cpu::matmul(w.fn, h, M, HC, mix.data(), mixn);
  const float* sc = w.scale.data(); const float* bs = w.base.data();
  const float eps = c.hc_eps;
  par_for((size_t)M, 1, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const float* hr = h + m * HC;
      const float r = 1.f / std::sqrt(dotf(hr, hr, HC) / (float)HC + c.rms_eps);
      const float* mx = mix.data() + m * mixn;
      float pre[4], cm[16];
      for (int j = 0; j < hc; ++j) pre[j] = 1.f / (1.f + std::exp(-(mx[j] * r * sc[0] + bs[j]))) + eps;
      for (int j = 0; j < hc; ++j) post[m * hc + j] = 2.f / (1.f + std::exp(-(mx[j + hc] * r * sc[1] + bs[j + hc])));
      for (int j = 0; j < hc * hc; ++j) cm[j] = mx[j + 2 * hc] * r * sc[2] + bs[j + 2 * hc];
      for (int j = 0; j < hc; ++j) {
        float mv = -INFINITY;
        for (int k = 0; k < hc; ++k) mv = std::max(mv, cm[j * hc + k]);
        float s = 0.f;
        for (int k = 0; k < hc; ++k) { cm[j * hc + k] = std::exp(cm[j * hc + k] - mv); s += cm[j * hc + k]; }
        for (int k = 0; k < hc; ++k) cm[j * hc + k] = cm[j * hc + k] / s + eps;
      }
      auto colnorm = [&] {
        for (int k = 0; k < hc; ++k) {
          float s = 0.f;
          for (int j = 0; j < hc; ++j) s += cm[j * hc + k];
          for (int j = 0; j < hc; ++j) cm[j * hc + k] = cm[j * hc + k] / (s + eps);
        }
      };
      colnorm();
      for (int it = 0; it < c.sinkhorn_iters - 1; ++it) {
        for (int j = 0; j < hc; ++j) {
          float s = 0.f;
          for (int k = 0; k < hc; ++k) s += cm[j * hc + k];
          for (int k = 0; k < hc; ++k) cm[j * hc + k] = cm[j * hc + k] / (s + eps);
        }
        colnorm();
      }
      std::memcpy(comb + m * hc * hc, cm, sizeof(float) * hc * hc);
      float* xr = x.data() + m * H;
      for (int d = 0; d < H; ++d) {
        float s = 0.f;
        for (int cc = 0; cc < hc; ++cc) s += pre[cc] * hr[(size_t)cc * H + d];
        xr[d] = bf16r(s);
      }
    }
  });
  rmsnorm_rows(x.data(), H, norm.data(), M, H, c.rms_eps, xn, H);
}

// h[m, c, :] = bf16(post[c]·y + Σ_j comb[j][c]·h[m, j, :])
void GlmCpuEngine::Impl::hc_post(float* h, const float* y, int M, const float* post, const float* comb) {
  const int H = c.hidden, HC = 4 * H;
  const int blk = 512, nb = H / blk;
  par_for((size_t)M * nb, 4, [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) {
      const size_t m = i / nb, d0 = (i % nb) * blk;
      float* hr = h + m * HC;
      const float* p = post + m * 4;
      const float* cb = comb + m * 16;
      for (size_t d = d0; d < d0 + blk; d += 8) {
        const __m256 yv = _mm256_loadu_ps(y + m * H + d);
        __m256 r[4];
        for (int j = 0; j < 4; ++j) r[j] = _mm256_loadu_ps(hr + (size_t)j * H + d);
        for (int cc = 0; cc < 4; ++cc) {
          __m256 s = _mm256_mul_ps(_mm256_set1_ps(p[cc]), yv);
          for (int j = 0; j < 4; ++j) s = _mm256_fmadd_ps(_mm256_set1_ps(cb[j * 4 + cc]), r[j], s);
          alignas(32) float t[8];
          _mm256_store_ps(t, s);
          for (int q = 0; q < 8; ++q) t[q] = bf16r(t[q]);
          _mm256_storeu_ps(hr + (size_t)cc * H + d, _mm256_load_ps(t));
        }
      }
    }
  });
}

// ---- KDA (kda.h contract) --------------------------------------------------------------------------------------------------------------------
void GlmCpuEngine::Impl::kda_layer(const KdaW& w, int ki, const std::vector<RowInfo>& rows, const float* xn, float* out) {
  const double t0 = hive::mono_ms();
  const int M = (int)rows.size(), H = c.hidden, NH = c.kda_heads, D = c.kda_dim, W = NH * D, C = 3 * W, NI = w.in.N;
  const int FA = C, GA = C + w.fa_dim, BO = C + w.fa_dim + w.ga_dim;
  thread_local std::vector<float> proj_tl, gp_tl, gt_tl, y_tl, qn_tl, kn_tl, eg_tl, beta_tl, o_tl;
  auto& proj = proj_tl; auto& gp = gp_tl; auto& gt = gt_tl; auto& y = y_tl; auto& qn = qn_tl; auto& kn = kn_tl; auto& eg = eg_tl; auto& beta = beta_tl;
  auto& o = o_tl;
  proj.resize((size_t)M * NI); gp.resize((size_t)M * W); gt.resize((size_t)M * W); y.resize((size_t)M * C); qn.resize((size_t)M * W);
  kn.resize((size_t)M * W); eg.resize((size_t)M * W); beta.resize((size_t)M * NH); o.resize((size_t)M * W);
  lin(w.in, xn, M, H, proj.data(), NI);
  {
    MatJob jobs[2] = {{MatView{&w.f_b, 0, w.f_b.N}, gp.data(), W, proj.data() + FA, M, NI}, {MatView{&w.g_b, 0, w.g_b.N}, gt.data(), W, proj.data() + GA, M, NI}};
    ::hive::cpu::matmul_multi(jobs, 2, nullptr, 0, 0);
    round_rows(gp.data(), M, W, W); round_rows(gt.data(), M, W, W);
  }
  st.ms_kda_proj += hive::mono_ms() - t0;
  const double t1 = hive::mono_ms();
  const float* cw = w.conv_w.data();
  const float qs = 1.f / std::sqrt((float)D), lb = c.kda_lower_bound;
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s && rows[r + n].pos == rows[r].pos + n) ++n;
    int part = -1;
    if (verify) for (size_t p = 0; p < vparts.size(); ++p) if (vparts[p].r0 == r) part = (int)p;
    std::vector<float>& hist = s.conv[ki];   // [3, C] oldest first
    // causal conv (4 taps) + SiLU, bf16 (kda.h step 1); conv state = the last 3 pre-conv inputs
    par_for((size_t)C, 1024, [&](size_t a, size_t b) {
      for (size_t ch = a; ch < b; ++ch) {
        const float* w4 = cw + ch * 4;
        auto X = [&](int t) { return t >= 0 ? proj[(size_t)(r + t) * NI + ch] : hist[(size_t)(3 + t) * C + ch]; };
        for (int t = 0; t < n; ++t) {
          float v = w4[0] * X(t - 3);
          v = std::fma(w4[1], X(t - 2), v); v = std::fma(w4[2], X(t - 1), v); v = std::fma(w4[3], X(t), v);
          y[(size_t)(r + t) * C + ch] = bf16r(v / (1.f + std::exp(-v)));
          if (part >= 0 && t + 1 < n)
            for (int k = 0; k < 3; ++k) snaps[part][t].conv[ki][(size_t)k * C + ch] = X(t - 2 + k);
        }
        float nv[3];
        for (int k = 0; k < 3; ++k) nv[k] = X(n - 3 + k);
        for (int k = 0; k < 3; ++k) hist[(size_t)k * C + ch] = nv[k];
      }
    });
    // l2 norms (q scaled by 1/√128), decay exp(g), beta (kda.h steps 2-3)
    par_for((size_t)n * NH, 16, [&](size_t a, size_t b) {
      for (size_t i = a; i < b; ++i) {
        const size_t row = r + i / NH, h = i % NH;
        const float* yq = y.data() + row * C + h * D;
        const float* yk = yq + W;
        const float nq = std::sqrt(dotf(yq, yq, D) + 1e-6f), nk = std::sqrt(dotf(yk, yk, D) + 1e-6f);
        const float A = std::exp(w.A_log[h]);
        for (int d = 0; d < D; ++d) {
          qn[row * W + h * D + d] = (yq[d] / nq) * qs;
          kn[row * W + h * D + d] = yk[d] / nk;
          const float g = lb * sigm(A * (gp[row * W + h * D + d] + w.dt_bias[h * D + d]));
          eg[row * W + h * D + d] = std::exp(g);
        }
        beta[row * NH + h] = bf16r(sigm(proj[row * NI + BO + h]));
      }
    });
    // delta rule per head, state S[k][v] fp32: S *= exp(g)[k]; kv = Σ_k S·k; δ = (v − kv)·β; S += k ⊗ δ; o = Σ_k S·q (kda.h step 4)
    Pool::get().run(NH, [&](int h) {
      float* S = s.state[ki].data() + (size_t)h * D * D;
      alignas(32) float kv[128], dl[128], oo[128];
      for (int t = 0; t < n; ++t) {
        const size_t row = (size_t)(r + t);
        const float* kk = kn.data() + row * W + h * D;
        const float* qq = qn.data() + row * W + h * D;
        const float* ee = eg.data() + row * W + h * D;
        const float* vv = y.data() + row * C + 2 * W + (size_t)h * D;
        const float bt = beta[row * NH + h];
        for (int j = 0; j < D; j += 8) _mm256_store_ps(kv + j, _mm256_setzero_ps());
        for (int k = 0; k < D; ++k) {
          float* Sr = S + (size_t)k * D;
          const __m256 ev = _mm256_set1_ps(ee[k]), kb = _mm256_set1_ps(kk[k]);
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
        float* orow = o.data() + row * W + (size_t)h * D;
        for (int j = 0; j < D; ++j) orow[j] = bf16r(oo[j]);
        if (part >= 0 && t + 1 < n) std::memcpy(snaps[part][t].state[ki].data() + (size_t)h * D * D, S, (size_t)D * D * 4);
      }
    });
    // RMSNorm(128) · o_norm_w · sigmoid(gate) per head (kda.h step 5)
    par_for((size_t)n * NH, 64, [&](size_t a, size_t b) {
      for (size_t i = a; i < b; ++i) {
        const size_t row = r + i / NH, h = i % NH;
        float* oh = o.data() + row * W + h * D;
        const float* gh = gt.data() + row * W + h * D;
        const float rr = 1.f / std::sqrt(dotf(oh, oh, D) * (1.f / D) + 1e-5f);
        for (int d = 0; d < D; ++d) oh[d] = bf16r((w.o_norm[d] * (oh[d] * rr)) * sigm(gh[d]));
      }
    });
    r += n;
  }
  st.ms_kda_core += hive::mono_ms() - t1;
  const double t2 = hive::mono_ms();
  lin(w.o, o.data(), M, W, out, H);
  st.ms_kda_proj += hive::mono_ms() - t2;
}

// ---- DSA (dsa.h contract) --------------------------------------------------------------------------------------------------------------------
namespace {
void grow_dsa(CpuSeq::Dsa& a, int64_t tokens) {
  if ((int64_t)a.c.size() < tokens * 512) {
    const size_t want = (size_t)std::max<int64_t>(tokens, (int64_t)a.c.size() / 512 * 3 / 2 + 64);
    a.c.resize(want * 512); a.kI.resize(want * 128); a.gs.resize(want * 128); a.pooled.resize((want / 4 + 1) * 128);
  }
}
// pooled key of pool p: K[ch] = bf16(Σ_j bf16(bf16(softmax_j(gs + ape)) · kI)) (dsa.cu pool_one)
void pool_one(CpuSeq::Dsa& a, const float* ape, int64_t p) {
  for (int ch = 0; ch < 128; ++ch) {
    float g[4], m = -INFINITY;
    for (int j = 0; j < 4; ++j) { g[j] = b2f(a.gs[(size_t)(4 * p + j) * 128 + ch]) + ape[j * 128 + ch]; m = std::max(m, g[j]); }
    float s = 0.f;
    for (int j = 0; j < 4; ++j) { g[j] = std::exp(g[j] - m); s += g[j]; }
    float acc = 0.f;
    for (int j = 0; j < 4; ++j) acc += bf16r(bf16r(g[j] / s) * b2f(a.kI[(size_t)(4 * p + j) * 128 + ch]));
    a.pooled[(size_t)p * 128 + ch] = f2b(acc);
  }
}
}  // namespace

void GlmCpuEngine::Impl::dsa_layer(const DsaW& w, int di, bool mtp_layer, const std::vector<RowInfo>& rows, const float* xn, float* out, bool kv_only) {
  double t0 = hive::mono_ms();
  const int M = (int)rows.size(), H = c.hidden, QL = c.q_lora, L = c.kv_lora, NH = c.n_heads, QD = c.qk_head, VD = c.v_head;
  const int IH = c.index_heads, ID = c.index_dim, IX = w.ix.N, QW = NH * QD, OW = NH * VD, NB = c.index_topk / c.index_kpool, kPool = c.index_kpool;
  thread_local std::vector<float> qa_tl, ckv_tl, ixo_tl, qres_tl, cn_tl, kI_tl, q_tl, qI_tl, qlat_tl, olat_tl, ao_tl;
  auto& qa = qa_tl; auto& ckv = ckv_tl; auto& ixo = ixo_tl; auto& qres = qres_tl; auto& cn = cn_tl; auto& kI = kI_tl; auto& q = q_tl; auto& qI = qI_tl;
  auto& qlat = qlat_tl; auto& olat = olat_tl; auto& ao = ao_tl;
  qa.resize((size_t)M * QL); ckv.resize((size_t)M * L); ixo.resize((size_t)M * IX); qres.resize((size_t)M * QL); cn.resize((size_t)M * L);
  kI.resize((size_t)M * ID);
  {
    std::vector<MatJob> jobs;
    jobs.push_back({MatView{&w.kv_a, 0, w.kv_a.N}, ckv.data(), L});
    jobs.push_back({MatView{&w.ix, 0, w.ix.N}, ixo.data(), IX});
    if (!kv_only) jobs.push_back({MatView{&w.q_a, 0, w.q_a.N}, qa.data(), QL});
    ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), xn, M, H);
    round_rows(ckv.data(), M, L, L); round_rows(ixo.data(), M, IX, 2 * ID);   // weights_proj stays fp32
  }
  rmsnorm_rows(ckv.data(), L, w.kv_a_norm.data(), M, L, c.rms_eps, cn.data(), L);
  // indexer key: LayerNorm(wk(x)) eps 1e-6 (fp32 math, bf16 out)
  par_for((size_t)M, 16, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const float* x = ixo.data() + m * IX;
      float s = 0.f, s2 = 0.f;
      for (int i = 0; i < ID; ++i) { s += x[i]; s2 += x[i] * x[i]; }
      const float mean = s / ID, var = std::max(s2 / ID - mean * mean, 0.f), inv = 1.f / std::sqrt(var + 1e-6f);
      for (int i = 0; i < ID; ++i) kI[m * ID + i] = bf16r((x[i] - mean) * inv * w.k_norm_w[i] + w.k_norm_b[i]);
    }
  });
  // cache rows, then the pools completed by these rows
  for (int r = 0; r < M; ++r) {
    CpuSeq& s = *rows[r].s;
    CpuSeq::Dsa& ca = mtp_layer ? s.mtp : s.dsa[di];
    const int64_t p = rows[r].pos;
    grow_dsa(ca, p + 1);
    for (int i = 0; i < L; ++i) ca.c[(size_t)p * L + i] = f2b(cn[(size_t)r * L + i]);
    for (int i = 0; i < ID; ++i) { ca.kI[(size_t)p * ID + i] = f2b(kI[(size_t)r * ID + i]); ca.gs[(size_t)p * ID + i] = f2b(ixo[(size_t)r * IX + ID + i]); }
  }
  for (int r = 0; r < M; ++r) {
    const int64_t p = rows[r].pos;
    if ((p + 1) % kPool) continue;
    CpuSeq& s = *rows[r].s;
    pool_one(mtp_layer ? s.mtp : s.dsa[di], w.ape.data(), p / kPool);
  }
  if (kv_only) { st.ms_dsa_proj += hive::mono_ms() - t0; return; }
  rmsnorm_rows(qa.data(), QL, w.q_a_norm.data(), M, QL, c.rms_eps, qres.data(), QL);
  q.resize((size_t)M * QW); qI.resize((size_t)M * IH * ID);
  {
    MatJob jobs[2] = {{MatView{&w.q_b, 0, w.q_b.N}, q.data(), QW}, {MatView{&w.wq_b, 0, w.wq_b.N}, qI.data(), IH * ID}};
    ::hive::cpu::matmul_multi(jobs, 2, qres.data(), M, QL);
    round_rows(q.data(), M, QW, QW); round_rows(qI.data(), M, IH * ID, IH * ID);
  }
  // absorbed MLA: q_lat[h] = W_UKᵀ q[h] (fp32)
  qlat.resize((size_t)M * NH * L);
  {
    std::vector<MatJob> jobs(NH);
    for (int h = 0; h < NH; ++h) jobs[h] = MatJob{MatView{&w.ukt, h * L, (h + 1) * L}, qlat.data() + (size_t)h * L, NH * L, q.data() + (size_t)h * QD, M, QW};
    ::hive::cpu::matmul_multi(jobs.data(), NH, nullptr, 0, 0);
  }
  st.ms_dsa_proj += hive::mono_ms() - t0;
  t0 = hive::mono_ms();
  // indexer selection per row (dsa.h): ≤ 512 visible pools → every position ≤ pos; else the top 512 pools by
  //   Σ_h (wI_h·32^-½)·relu(qI_h·K_p·128^-½) (ties → lower pool id) expanded to their tokens, plus the incomplete tail pool
  std::vector<std::vector<int32_t>> sel(M);
  const float wsc = 1.f / std::sqrt((float)IH), ksc = 1.f / std::sqrt((float)ID);
  // pool scores of every sparse row, in tasks of (row, block of 256 pools) so that one decode row at a long context uses every worker
  std::vector<size_t> soff(M + 1, 0);
  for (int r = 0; r < M; ++r) { const int vis = (int)((rows[r].pos + 1) / kPool); soff[r + 1] = soff[r] + (vis > NB ? (size_t)vis : 0); }
  std::vector<float> scores(soff[M]);
  {
    constexpr int kPB = 256;
    std::vector<std::pair<int, int>> tasks;
    for (int r = 0; r < M; ++r) for (size_t g = 0; g < soff[r + 1] - soff[r]; g += kPB) tasks.push_back({r, (int)g});
    par_for(tasks.size(), 1, [&](size_t a, size_t b) {
      alignas(32) float kf[128];
      for (size_t ti = a; ti < b; ++ti) {
        const int r = tasks[ti].first, g0 = tasks[ti].second, g1 = std::min<int>(g0 + kPB, (int)(soff[r + 1] - soff[r]));
        CpuSeq& s = *rows[r].s;
        const CpuSeq::Dsa& ca = mtp_layer ? s.mtp : s.dsa[di];
        const float* qr = qI.data() + (size_t)r * IH * ID;
        const float* wr = ixo.data() + (size_t)r * IX + 2 * ID;
        for (int g = g0; g < g1; ++g) {
          bf16_row(ca.pooled.data() + (size_t)g * ID, ID, kf);
          float tot = 0.f;
          for (int hh = 0; hh < IH; ++hh) tot += (wr[hh] * wsc) * std::max(dotf(qr + hh * ID, kf, ID) * ksc, 0.f);
          scores[soff[r] + g] = tot;
        }
      }
    });
  }
  par_for((size_t)M, 1, [&](size_t a, size_t b) {
    std::vector<int32_t> ord;
    for (size_t r = a; r < b; ++r) {
      const int64_t p = rows[r].pos;
      const int vis = (int)((p + 1) / kPool);
      std::vector<int32_t>& ix = sel[r];
      ix.clear();
      if (vis <= NB) { ix.resize(p + 1); std::iota(ix.begin(), ix.end(), 0); continue; }
      const float* sc = scores.data() + soff[r];
      ord.resize(vis);
      std::iota(ord.begin(), ord.end(), 0);
      std::nth_element(ord.begin(), ord.begin() + NB, ord.end(), [&](int x1, int x2) { return sc[x1] > sc[x2] || (sc[x1] == sc[x2] && x1 < x2); });
      std::sort(ord.begin(), ord.begin() + NB);
      ix.reserve((size_t)NB * kPool + kPool);
      for (int i = 0; i < NB; ++i) for (int j = 0; j < kPool; ++j) ix.push_back(ord[i] * kPool + j);
      for (int64_t tp = (int64_t)vis * kPool; tp <= p; ++tp) ix.push_back((int32_t)tp);
    }
  });
  st.ms_dsa_idx += hive::mono_ms() - t0;
  t0 = hive::mono_ms();
  // sparse attention over the latent cache: s_j = q_lat[h]·c_j·256^-½, out_lat[h] = Σ softmax(s)_j c_j. Tasks: row × group of heads.
  olat.resize((size_t)M * NH * L);
  const float scale = 1.f / std::sqrt((float)QD);
  constexpr int HG = 8, kTile = 16;
  const int ngr = NH / HG;
  // few rows (decode): the positions are split over several tasks (partial softmax states combined below) so every worker has work
  size_t maxsel = 0;
  for (int r = 0; r < M; ++r) maxsel = std::max(maxsel, sel[r].size());
  int nsp = 1;
  while (M * ngr * nsp < Pool::get().threads() && (size_t)(nsp * 2) * 256 <= maxsel) nsp *= 2;
  thread_local std::vector<float> part_tl, pml_tl;
  auto& part = part_tl; auto& pml = pml_tl;
  if (nsp > 1) { part.resize((size_t)M * NH * nsp * L); pml.resize((size_t)M * NH * nsp * 2); }
  Pool::get().run(M * ngr * nsp, [&](int task) {
    const int sp = task % nsp, rg = task / nsp, r = rg / ngr, h0 = (rg % ngr) * HG;
    CpuSeq& s = *rows[r].s;
    const CpuSeq::Dsa& ca = mtp_layer ? s.mtp : s.dsa[di];
    const std::vector<int32_t>& ix = sel[r];
    thread_local std::vector<float> kt;
    kt.resize((size_t)kTile * L);
    float mx[HG], ls[HG], scs[HG][kTile];
    // split sp writes head h to part[((r·NH + h0 + h)·nsp + sp)·L] — o of head h at oh + h·ostride
    float* oh = nsp > 1 ? part.data() + (((size_t)r * NH + h0) * nsp + sp) * L : olat.data() + ((size_t)r * NH + h0) * L;
    const size_t ostride = nsp > 1 ? (size_t)nsp * L : (size_t)L;
    const float* qh = qlat.data() + ((size_t)r * NH + h0) * L;
    for (int h = 0; h < HG; ++h) { mx[h] = -INFINITY; ls[h] = 0.f; std::memset(oh + (size_t)h * ostride, 0, (size_t)L * 4); }
    const int nall = (int)ix.size(), per = ((nall + nsp - 1) / nsp + kTile - 1) / kTile * kTile;
    const int jb = std::min(nall, sp * per), n = std::min(nall, jb + per);
    for (int t0i = jb; t0i < n; t0i += kTile) {
      const int cnt = std::min(kTile, n - t0i);
      for (int j = 0; j < cnt; ++j) bf16_row(ca.c.data() + (size_t)ix[t0i + j] * L, L, kt.data() + (size_t)j * L);
      if (cnt < kTile) std::memset(kt.data() + (size_t)cnt * L, 0, (size_t)(kTile - cnt) * L * 4);
      // scores, 4 heads × 2 positions per register block (6 loads per 8 FMAs instead of 2 per FMA)
      for (int h = 0; h < HG; h += 4)
        for (int j = 0; j < kTile; j += 2) {
          __m256 a[4][2];
          for (int u = 0; u < 4; ++u) a[u][0] = a[u][1] = _mm256_setzero_ps();
          const float* k0 = kt.data() + (size_t)j * L;
          const float* k1 = k0 + L;
          for (int d = 0; d < L; d += 8) {
            const __m256 x0 = _mm256_loadu_ps(k0 + d), x1 = _mm256_loadu_ps(k1 + d);
            for (int u = 0; u < 4; ++u) {
              const __m256 qv = _mm256_loadu_ps(qh + (size_t)(h + u) * L + d);
              a[u][0] = _mm256_fmadd_ps(qv, x0, a[u][0]);
              a[u][1] = _mm256_fmadd_ps(qv, x1, a[u][1]);
            }
          }
          for (int u = 0; u < 4; ++u) { scs[h + u][j] = hsum8(a[u][0]) * scale; scs[h + u][j + 1] = hsum8(a[u][1]) * scale; }
        }
      float pw[HG][kTile];
      for (int h = 0; h < HG; ++h) {
        float tm = -INFINITY;
        for (int j = 0; j < cnt; ++j) tm = std::max(tm, scs[h][j]);
        const float nm = std::max(mx[h], tm);
        const float corr = mx[h] == -INFINITY ? 0.f : std::exp(mx[h] - nm);
        float sum = ls[h] * corr;
        for (int j = 0; j < kTile; ++j) { pw[h][j] = j < cnt ? std::exp(scs[h][j] - nm) : 0.f; sum += pw[h][j]; }
        ls[h] = sum; mx[h] = nm;
        scs[h][0] = corr;  // reuse: the rescale of the running output
      }
      // o[h] = corr·o[h] + Σ_j p[h][j]·c_j, per 8 columns with the 8 heads' accumulators in registers
      for (int d = 0; d < L; d += 8) {
        __m256 acc[HG];
        for (int h = 0; h < HG; ++h) acc[h] = _mm256_mul_ps(_mm256_set1_ps(scs[h][0]), _mm256_loadu_ps(oh + (size_t)h * ostride + d));
        for (int j = 0; j < kTile; j += 4) {
          const __m256 c0 = _mm256_loadu_ps(kt.data() + (size_t)j * L + d), c1 = _mm256_loadu_ps(kt.data() + (size_t)(j + 1) * L + d);
          const __m256 c2 = _mm256_loadu_ps(kt.data() + (size_t)(j + 2) * L + d), c3 = _mm256_loadu_ps(kt.data() + (size_t)(j + 3) * L + d);
          for (int h = 0; h < HG; ++h) {
            acc[h] = _mm256_fmadd_ps(_mm256_set1_ps(pw[h][j]), c0, acc[h]);
            acc[h] = _mm256_fmadd_ps(_mm256_set1_ps(pw[h][j + 1]), c1, acc[h]);
            acc[h] = _mm256_fmadd_ps(_mm256_set1_ps(pw[h][j + 2]), c2, acc[h]);
            acc[h] = _mm256_fmadd_ps(_mm256_set1_ps(pw[h][j + 3]), c3, acc[h]);
          }
        }
        for (int h = 0; h < HG; ++h) _mm256_storeu_ps(oh + (size_t)h * ostride + d, acc[h]);
      }
    }
    if (nsp > 1) {
      for (int h = 0; h < HG; ++h) { float* q2 = pml.data() + (((size_t)r * NH + h0 + h) * nsp + sp) * 2; q2[0] = mx[h]; q2[1] = ls[h]; }
      return;
    }
    for (int h = 0; h < HG; ++h) {
      const float inv = ls[h] > 0.f ? 1.f / ls[h] : 0.f;
      float* o = oh + (size_t)h * L;
      for (int d = 0; d < L; ++d) o[d] *= inv;
    }
  });
  if (nsp > 1)  // combine the splits: o = Σ_s e^(m_s − m) o_s / Σ_s e^(m_s − m) l_s
    par_for((size_t)M * NH, 4, [&](size_t a, size_t b) {
      for (size_t i = a; i < b; ++i) {
        const float* q2 = pml.data() + i * nsp * 2;
        float m = -INFINITY;
        for (int k = 0; k < nsp; ++k) m = std::max(m, q2[2 * k]);
        float* o = olat.data() + i * L;
        std::fill(o, o + L, 0.f);
        float l = 0.f;
        for (int k = 0; k < nsp; ++k) {
          if (q2[2 * k] == -INFINITY) continue;
          const float w = std::exp(q2[2 * k] - m);
          l += w * q2[2 * k + 1];
          const float* ps = part.data() + (i * nsp + k) * L;
          for (int d = 0; d < L; ++d) o[d] += w * ps[d];
        }
        const float inv = l > 0.f ? 1.f / l : 0.f;
        for (int d = 0; d < L; ++d) o[d] *= inv;
      }
    });
  // out[h] = W_UV[h] out_lat[h] (bf16), then o_proj
  ao.resize((size_t)M * OW);
  {
    std::vector<MatJob> jobs(NH);
    for (int h = 0; h < NH; ++h)
      jobs[h] = MatJob{MatView{&w.kv_b, h * (QD + VD) + QD, (h + 1) * (QD + VD)}, ao.data() + (size_t)h * VD, OW, olat.data() + (size_t)h * L, M, NH * L};
    ::hive::cpu::matmul_multi(jobs.data(), NH, nullptr, 0, 0);
    round_rows(ao.data(), M, OW, OW);
  }
  st.ms_dsa_attn += hive::mono_ms() - t0;
  t0 = hive::mono_ms();
  lin(w.o, ao.data(), M, OW, out, H);
  st.ms_dsa_proj += hive::mono_ms() - t0;
}

// ---- dense MLP (layers 0-2, NVFP4) -----------------------------------------------------------------------------------------------------------
void GlmCpuEngine::Impl::dense_mlp(const DenseW& w, int M, const float* xn, float* out) {
  const int H = c.hidden, I = w.g.N;
  thread_local std::vector<float> g_tl, u_tl;
  auto& g = g_tl; auto& u = u_tl;
  g.resize((size_t)M * I); u.resize((size_t)M * I);
  MatJob jobs[2] = {{MatView{&w.g, 0, I}, g.data(), I}, {MatView{&w.u, 0, I}, u.data(), I}};
  ::hive::cpu::matmul_multi(jobs, 2, xn, M, H);
  const float lim = c.swiglu_limit;
  par_for((size_t)M * I, 65536, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) g[i] = silu_lim_act(bf16r(g[i]), bf16r(u[i]), lim); });
  lin(w.d, g.data(), M, I, out, H);
}

// ---- MoE (router sigmoid + correction bias, top-8, NVFP4 experts on the CPU, shared expert) ---------------------------------------------------
void GlmCpuEngine::Impl::moe(const MoeW& w, int M, const float* xn, float* out) {
  double t0 = hive::mono_ms();
  const int H = c.hidden, E = c.n_routed, K = c.n_act, I = c.moe_inter, SI = w.sh_g.N;
  thread_local std::vector<float> lg_tl, wts_tl, sg_tl, su_tl, sh_tl;
  thread_local std::vector<int> ids_tl;
  auto& lg = lg_tl; auto& wts = wts_tl; auto& ids = ids_tl; auto& sg = sg_tl; auto& su = su_tl; auto& sh = sh_tl;
  lg.resize((size_t)M * E); wts.resize((size_t)M * K); ids.resize((size_t)M * K);
  ::hive::cpu::matmul(w.router, xn, M, H, lg.data(), E);   // fp32 logits
  for (int m = 0; m < M; ++m) {
    float sv[512], cv[512];
    const float* l = lg.data() + (size_t)m * E;
    for (int e = 0; e < E; ++e) { sv[e] = 1.f / (1.f + std::exp(-l[e])); cv[e] = sv[e] + w.bias[e]; }
    float sum = 0.f;
    int sel[16];
    for (int j = 0; j < K; ++j) {
      int best = -1; float bv = -INFINITY;
      for (int e = 0; e < E; ++e) if (cv[e] > bv) { bv = cv[e]; best = e; }
      sel[j] = best; cv[best] = -INFINITY; sum += sv[best];
    }
    const float inv = 1.f / (sum + 1e-20f);
    for (int j = 0; j < K; ++j) { ids[(size_t)m * K + j] = sel[j]; wts[(size_t)m * K + j] = sv[sel[j]] * inv * c.routed_scale; }
    if (route_dump) {
      float nxt = -INFINITY;
      for (int e = 0; e < E; ++e) nxt = std::max(nxt, cv[e]);
      fprintf(route_dump, "%d %d %.6g", cur_layer, m, sv[sel[K - 1]] + w.bias[sel[K - 1]] - nxt);
      for (int j = 0; j < K; ++j) fprintf(route_dump, " %d", sel[j]);
      fputc('\n', route_dump);
    }
  }
  st.ms_router += hive::mono_ms() - t0;
  t0 = hive::mono_ms();
  if (ext_experts && &w != &mtp_moe()) {  // hybrid: shared expert here, routed experts by the caller's expert store
    sg.resize((size_t)M * SI); su.resize((size_t)M * SI); sh.resize((size_t)M * H);
    MatJob sj[2] = {{MatView{&w.sh_g, 0, SI}, sg.data(), SI, xn, M, H}, {MatView{&w.sh_u, 0, SI}, su.data(), SI, xn, M, H}};
    ::hive::cpu::matmul_multi(sj, 2, nullptr, 0, 0);
    const float lim = c.swiglu_limit;
    par_for((size_t)M * SI, 65536, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) sg[i] = silu_lim_act(bf16r(sg[i]), bf16r(su[i]), lim); });
    ::hive::cpu::matmul(w.sh_d, sg.data(), M, SI, sh.data(), H);
    thread_local std::vector<float> yr_tl;
    auto& yr = yr_tl;
    yr.assign((size_t)M * H, 0.f);
    ext_experts(cur_layer, M, ids.data(), wts.data(), xn, yr.data());
    par_for((size_t)M * H, 65536, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) out[i] = bf16r(bf16r(yr[i]) + bf16r(sh[i])); });
    st.ms_experts += hive::mono_ms() - t0;
    return;
  }
  // pairs grouped by expert (experts ascending, rows ascending inside); one dispatch for every expert's gate | up and the shared gate / up,
  //   one for every expert's down halves and the shared down
  const int Pn = M * K, hh = H / 2;
  thread_local std::vector<float> Xe_tl, G_tl, U_tl, Y_tl, D2_tl;
  thread_local std::vector<int> cnt_tl, off_tl, prow_tl, pslot_tl, pexp_tl;
  auto& Xe = Xe_tl; auto& G = G_tl; auto& U = U_tl; auto& Y = Y_tl; auto& D2 = D2_tl;
  auto& cnt = cnt_tl; auto& off = off_tl; auto& prow = prow_tl; auto& pslot = pslot_tl; auto& pexp = pexp_tl;
  Xe.resize((size_t)Pn * H); G.resize((size_t)Pn * I); U.resize((size_t)Pn * I); Y.resize((size_t)Pn * I); D2.resize((size_t)Pn * H);
  sg.resize((size_t)M * SI); su.resize((size_t)M * SI); sh.resize((size_t)M * H);
  cnt.assign(E, 0); off.assign(E + 1, 0); prow.resize(Pn); pslot.resize(Pn); pexp.resize(Pn);
  for (int i = 0; i < Pn; ++i) ++cnt[ids[i]];
  for (int e = 0; e < E; ++e) off[e + 1] = off[e] + cnt[e];
  {
    std::vector<int> fill(off.begin(), off.end() - 1);
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k) { const int e = ids[(size_t)m * K + k], q = fill[e]++; prow[q] = m; pslot[q] = k; pexp[q] = e; }
  }
  par_for((size_t)Pn, 64, [&](size_t a, size_t b) { for (size_t q = a; q < b; ++q) std::memcpy(Xe.data() + q * H, xn + (size_t)prow[q] * H, (size_t)H * 4); });
  std::vector<MatJob> jobs;
  jobs.reserve(2 * (size_t)std::min(Pn, E) + 3);
  jobs.push_back(MatJob{MatView{&w.sh_g, 0, SI}, sg.data(), SI, xn, M, H});
  jobs.push_back(MatJob{MatView{&w.sh_u, 0, SI}, su.data(), SI, xn, M, H});
  for (int e = 0; e < E; ++e) {
    if (!cnt[e]) continue;
    const int r = off[e];
    jobs.push_back(MatJob{MatView{&w.gu, e * I, (e + 1) * I}, G.data() + (size_t)r * I, I, Xe.data() + (size_t)r * H, cnt[e], H});
    jobs.push_back(MatJob{MatView{&w.gu, E * I + e * I, E * I + (e + 1) * I}, U.data() + (size_t)r * I, I, Xe.data() + (size_t)r * H, cnt[e], H});
  }
  ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
  const float lim = c.swiglu_limit;
  par_for((size_t)M, 16, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m)
      for (int i = 0; i < SI; ++i) sg[m * SI + i] = silu_lim_act(bf16r(sg[m * SI + i]), bf16r(su[m * SI + i]), lim);
  });
  par_for((size_t)Pn, 2, [&](size_t a, size_t b) {
    for (size_t r = a; r < b; ++r) {
      const int e = pexp[r];
      const float gg = w.g_gate[e], gu = w.g_up[e];
      for (int i = 0; i < I; ++i) Y[r * I + i] = silu_lim_act(bf16r(G[r * I + i] * gg), bf16r(U[r * I + i] * gu), lim);
    }
  });
  jobs.clear();
  jobs.push_back(MatJob{MatView{&w.sh_d, 0, H}, sh.data(), H, sg.data(), M, SI});
  for (int e = 0; e < E; ++e) {
    if (!cnt[e]) continue;
    const int r = off[e];
    jobs.push_back(MatJob{MatView{&w.d, e * hh, (e + 1) * hh}, D2.data() + (size_t)r * H, H, Y.data() + (size_t)r * I, cnt[e], I});
    jobs.push_back(MatJob{MatView{&w.d, E * hh + e * hh, E * hh + (e + 1) * hh}, D2.data() + (size_t)r * H + hh, H, Y.data() + (size_t)r * I, cnt[e], I});
  }
  ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
  // out = bf16(bf16(Σ_k w_k · bf16(down_k)) + bf16(shared))   (glm_moe.cu: bf16 expert outputs × weight, fp32 sum; f32_to_bf16 + add_bf16)
  thread_local std::vector<int> slot_tl;
  auto& slot = slot_tl;
  slot.resize(Pn);
  for (int q = 0; q < Pn; ++q) slot[(size_t)prow[q] * K + pslot[q]] = q;
  par_for((size_t)M * 8, 1, [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) {
      const size_t m = i / 8, d0 = (i % 8) * (H / 8), d1 = d0 + H / 8;
      for (size_t d = d0; d < d1; ++d) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
          const int q = slot[m * K + k];
          acc += bf16r(D2[(size_t)q * H + d] * w.g_down[pexp[q]]) * wts[m * K + k];
        }
        out[m * H + d] = bf16r(bf16r(acc) + bf16r(sh[m * H + d]));
      }
    }
  });
  st.ms_experts += hive::mono_ms() - t0;
}

const MoeW& GlmCpuEngine::Impl::mtp_moe() const { static const MoeW none; return mtp ? mtp->mw : none; }

void GlmCpuEngine::Impl::final_hidden(const float* h, int M, float* out) {
  const int H = c.hidden;
  std::vector<float> x((size_t)M * H);
  par_for((size_t)M, 16, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m)
      for (int d = 0; d < H; ++d) {
        float s = 0.f;
        for (int cc = 0; cc < c.hc; ++cc) s += h[m * 4 * H + (size_t)cc * H + d];
        x[m * H + d] = bf16r(s / c.hc);
      }
  });
  rmsnorm_rows(x.data(), H, final_norm.data(), M, H, c.rms_eps, out, H);
}

// ---- forward ---------------------------------------------------------------------------------------------------------------------------------
void GlmCpuEngine::Impl::run_range(const std::vector<RowInfo>& rows, float* h, bool prefill) {
  const int M = (int)rows.size(), H = c.hidden;
  std::vector<float> xn((size_t)M * H), y((size_t)M * H), post((size_t)M * 4), comb((size_t)M * 16);
  for (int l = first; l <= last; ++l) {
    const LayerW& L = layers[l];
    if (prefill && layer_inject && *layer_inject && (l > first || first > 0)) (*layer_inject)(l, h, M);
    double tc = hive::mono_ms();
    auto lap = [&](double& acc) { const double t = hive::mono_ms(); acc += t - tc; tc = t; };
    hc_pre(L.ha, L.in_norm, h, M, xn.data(), post.data(), comb.data());
    lap(st.ms_hc);
    if (L.kda) kda_layer(L.kw, kda_index[l], rows, xn.data(), y.data());
    else dsa_layer(L.dw, dsa_index[l], false, rows, xn.data(), y.data());
    tc = hive::mono_ms();
    hc_post(h, y.data(), M, post.data(), comb.data());
    hc_pre(L.hf, L.post_norm, h, M, xn.data(), post.data(), comb.data());
    lap(st.ms_hc);
    cur_layer = l;
    if (L.moe) { moe(L.mw, M, xn.data(), y.data()); cur_layer = -1; }
    else { dense_mlp(L.dense, M, xn.data(), y.data()); lap(st.ms_dense); }
    tc = hive::mono_ms();
    hc_post(h, y.data(), M, post.data(), comb.data());
    lap(st.ms_hc);
    if (layer_hook && *layer_hook) (*layer_hook)(l, h, M);
    if (prefill && depth == 1 && layer_boundary && *layer_boundary && l < last) {
      std::vector<EmbedOverride> parked = std::move(emb_over);  // inner forwards are other sequences' decode / verify steps
      const int64_t base = emb_base;
      emb_over.clear(); emb_base = 0;
      (*layer_boundary)(l);
      emb_over = std::move(parked); emb_base = base;
    }
  }
}

void GlmCpuEngine::Impl::forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, float* logits, bool all_logits) {
  const int M = (int)rows.size(), H = c.hidden, HC = c.hc * H;
  const double t0 = hive::mono_ms();
  for (const RowInfo& ri : rows)
    if (ri.pos + 1 > ri.s->cap) throw std::runtime_error("glm cpu: sequence capacity exceeded");
  ++depth;
  struct Depth { int& d; ~Depth() { --d; } } dguard{depth};
  if ((int)h_stack.size() < depth) h_stack.resize(depth);
  std::vector<float>& hv = h_stack[depth - 1];
  hv.resize((size_t)M * HC);
  float* h = hv.data();
  {
    std::vector<float> x((size_t)M * H);
    par_for((size_t)M, 64, [&](size_t a, size_t b) { for (size_t r = a; r < b; ++r) bf16_row(embed.data() + (size_t)ids[r] * H, H, x.data() + r * H); });
    if (prefill)
      for (const EmbedOverride& o : emb_over) {
        const int64_t a = std::max<int64_t>(o.row, emb_base), b = std::min<int64_t>(o.row + o.n, emb_base + M);
        for (int64_t g = a; g < b; ++g) std::memcpy(x.data() + (size_t)(g - emb_base) * H, o.emb.data() + (size_t)(g - o.row) * H, (size_t)H * 4);
      }
    par_for((size_t)M, 64, [&](size_t a, size_t b) {
      for (size_t r = a; r < b; ++r) for (int cc = 0; cc < c.hc; ++cc) std::memcpy(h + r * HC + (size_t)cc * H, x.data() + r * H, (size_t)H * 4);
    });
  }
  st.ms_embed += hive::mono_ms() - t0;
  run_range(rows, h, prefill);
  const double th = hive::mono_ms();
  if (last == c.n_layers - 1) {
    const int r0 = all_logits ? 0 : M - 1, nr = all_logits ? M : 1;
    if (mtp && mtp_k > 0) {
      fh.resize((size_t)M * H);
      final_hidden(h, M, fh.data());
      if (logits) ::hive::cpu::matmul(lm_head, fh.data() + (size_t)r0 * H, nr, H, logits, c.vocab);
    } else if (logits) {
      std::vector<float> f((size_t)nr * H);
      final_hidden(h + (size_t)r0 * HC, nr, f.data());
      ::hive::cpu::matmul(lm_head, f.data(), nr, H, logits, c.vocab);
    }
  }
  st.ms_head += hive::mono_ms() - th;
  const double tm = hive::mono_ms();
  if (mtp && mtp_k > 0 && last == c.n_layers - 1) mtp_after_forward(rows, ids, prefill);
  st.ms_mtp += hive::mono_ms() - tm;
  for (int r = 0; r < M; ++r) { CpuSeq& s = *rows[r].s; s.pos = rows[r].pos + 1; s.tokens.push_back(ids[r]); }
  if (!prefill) ++st.steps;
  st.ms_total += hive::mono_ms() - t0;
}

GlmCpuEngine::Impl::~Impl() {
  if (route_dump) fclose(route_dump);
  using ::hive::cpu::free_mat;
  auto dsa = [](DsaW& w) { for (Mat* m : {&w.q_a, &w.kv_a, &w.ix, &w.q_b, &w.wq_b, &w.ukt, &w.kv_b, &w.o}) free_mat(*m); };
  auto moe = [](MoeW& w) { for (Mat* m : {&w.router, &w.sh_g, &w.sh_u, &w.sh_d, &w.gu, &w.d}) free_mat(*m); };
  for (LayerW& L : layers) {
    free_mat(L.ha.fn); free_mat(L.hf.fn);
    for (Mat* m : {&L.kw.in, &L.kw.f_b, &L.kw.g_b, &L.kw.o, &L.dense.g, &L.dense.u, &L.dense.d}) free_mat(*m);
    dsa(L.dw); moe(L.mw);
  }
  free_mat(lm_head); free_mat(draft_head);
  if (mtp) { free_mat(mtp->eh); dsa(mtp->dw); moe(mtp->mw); }
  if (vision) {
    for (Mat* m : {&vision->patch, &vision->down, &vision->m_proj, &vision->m_gate, &vision->m_up, &vision->m_down}) free_mat(*m);
    for (auto& B : vision->blocks) for (Mat* m : {&B.qkv, &B.proj, &B.gate, &B.up, &B.down}) free_mat(*m);
  }
}

// ---- public API ------------------------------------------------------------------------------------------------------------------------
GlmCpuEngine::GlmCpuEngine(const std::string& ckpt_dir, bool load_mtp) : p_(std::make_unique<Impl>()) {
  p_->ckpt_dir = ckpt_dir;
  p_->layer_hook = &layer_hook;
  p_->layer_boundary = &layer_boundary;
  p_->layer_inject = &layer_inject;
  p_->load(load_mtp);
}
GlmCpuEngine::GlmCpuEngine(const std::string& ckpt_dir, const Hybrid& hy) : p_(std::make_unique<Impl>()) {
  p_->ckpt_dir = ckpt_dir;
  p_->layer_hook = &layer_hook;
  p_->layer_boundary = &layer_boundary;
  p_->layer_inject = &layer_inject;
  p_->hybrid = true;
  p_->first = hy.first; p_->last = hy.last;
  p_->ext_experts = hy.experts;
  p_->load(false);
}
GlmCpuEngine::~GlmCpuEngine() = default;
const CpuConfig& GlmCpuEngine::cfg() const { return p_->c; }
int GlmCpuEngine::first_layer() const { return p_->first; }
int GlmCpuEngine::last_layer() const { return p_->last; }
CpuStats& GlmCpuEngine::stats() { return p_->st; }
bool GlmCpuEngine::mtp_on() const { return p_->mtp && p_->mtp_k > 0; }
int GlmCpuEngine::mtp_k() const { return p_->mtp_k; }
void GlmCpuEngine::set_embed_override(std::vector<EmbedOverride> v) { p_->emb_over = std::move(v); }

void GlmCpuEngine::init_seq(CpuSeq& s, int64_t cap) {
  const CpuConfig& c = p_->c;
  s.cap = cap;
  const size_t W = (size_t)c.kda_heads * c.kda_dim;
  s.conv.assign(p_->n_kda, {}); s.state.assign(p_->n_kda, {});
  for (int l = 0; l < c.n_layers; ++l) {  // only the loaded layers' states (all of them unless a layer range / hybrid use)
    const int i = p_->kda_index[l];
    if (i < 0 || l < p_->first || l > p_->last) continue;
    s.conv[i].assign(3 * 3 * W, 0.f); s.state[i].assign((size_t)c.kda_heads * c.kda_dim * c.kda_dim, 0.f);
  }
  s.dsa.assign(p_->n_dsa, {});
  reset_seq(s);
}

void GlmCpuEngine::reset_seq(CpuSeq& s) {
  for (auto& v : s.conv) std::fill(v.begin(), v.end(), 0.f);
  for (auto& v : s.state) std::fill(v.begin(), v.end(), 0.f);
  s.pos = 0; s.tokens.clear(); s.broken = false;
  s.h_last.assign(p_->mtp ? (size_t)p_->c.hidden : 0, 0.f); s.h_last_valid = false; s.mtp_written = 0;
}

void GlmCpuEngine::prefill(CpuSeq& s, const int32_t* ids, int T, float* logits) {
  if (T <= 0) return;
  if ((int64_t)s.tokens.size() != s.pos) throw std::runtime_error("glm cpu: sequence tokens out of step with its position");
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

void GlmCpuEngine::decode(CpuSeq* const* seqs, const int32_t* ids, int M, float* logits) {
  std::vector<RowInfo> rows(M);
  for (int m = 0; m < M; ++m) rows[m] = {seqs[m], seqs[m]->pos};
  try { p_->forward(rows, ids, false, logits, true); } catch (...) { for (int m = 0; m < M; ++m) seqs[m]->broken = true; throw; }
}

void GlmCpuEngine::verify_batch(CpuSeq* const* seqs, const int* Ms, int S, const int32_t* ids, float* logits) {
  Impl& I = *p_;
  I.vparts.clear();
  std::vector<RowInfo> rows;
  int R = 0;
  for (int p = 0; p < S; ++p) {
    if (Ms[p] < 1 || R + Ms[p] > 8) throw std::runtime_error("glm cpu verify rows");
    I.vparts.push_back({seqs[p], seqs[p]->pos, Ms[p], R});
    for (int r = 0; r < Ms[p]; ++r) rows.push_back({seqs[p], seqs[p]->pos + r});
    R += Ms[p];
  }
  I.prepare_snaps(seqs, Ms, S);
  I.verify_ids.assign(ids, ids + R);
  I.verify = true;
  try { I.forward(rows, ids, false, logits, true); } catch (...) { I.verify = false; I.vparts.clear(); for (int p = 0; p < S; ++p) seqs[p]->broken = true; throw; }
  I.verify = false;
}

void GlmCpuEngine::Impl::prepare_snaps(CpuSeq* const* seqs, const int* Ms, int S) {
  if ((int)snaps.size() < S) snaps.resize(S);
  for (int p = 0; p < S; ++p) {
    if ((int)snaps[p].size() < Ms[p] - 1) snaps[p].resize(Ms[p] - 1);
    for (int r = 0; r < Ms[p] - 1; ++r) {
      Snap& sn = snaps[p][r];
      sn.conv.resize(seqs[p]->conv.size()); sn.state.resize(seqs[p]->state.size());
      for (size_t i = 0; i < sn.conv.size(); ++i) {  // hybrid: only the loaded layers' states are kept
        const bool used = seqs[p]->conv[i].size() > 0;
        sn.conv[i].resize(used ? seqs[p]->conv[i].size() : 0); sn.state[i].resize(used ? seqs[p]->state[i].size() : 0);
      }
    }
  }
}

void GlmCpuEngine::run_layers(CpuSeq* const* seqs, const int64_t* pos, int M, float* h, bool prefill, const int* part_rows, int S) {
  Impl& I = *p_;
  std::vector<RowInfo> rows(M);
  for (int r = 0; r < M; ++r) rows[r] = {seqs[r], pos[r]};
  I.vparts.clear();
  if (S > 0) {
    std::vector<CpuSeq*> ps(S);
    int R = 0;
    for (int p = 0; p < S; ++p) { ps[p] = seqs[R]; I.vparts.push_back({seqs[R], pos[R], part_rows[p], R}); R += part_rows[p]; }
    if (R != M) throw std::runtime_error("glm cpu run_layers: parts do not cover the rows");
    I.prepare_snaps(ps.data(), part_rows, S);
    I.verify = true;
  }
  ++I.depth;
  struct Guard { Impl& I; ~Guard() { --I.depth; I.verify = false; } } g{I};
  const double t0 = hive::mono_ms();
  for (const RowInfo& ri : rows) if (ri.pos + 1 > ri.s->cap) throw std::runtime_error("glm cpu: sequence capacity exceeded");
  I.run_range(rows, h, prefill);
  I.st.ms_total += hive::mono_ms() - t0;
  if (!prefill) ++I.st.steps;
}

void GlmCpuEngine::restore_verify(const int* n_keep, int S) {
  Impl& I = *p_;
  if (S != (int)I.vparts.size()) throw std::runtime_error("glm cpu restore_verify without a matching run_layers");
  for (int p = 0; p < S; ++p) {
    const Impl::VPart& vp = I.vparts[p];
    CpuSeq& s = *vp.s;
    const int nk = n_keep[p];
    if (nk < 1 || nk > vp.M) throw std::runtime_error("glm cpu restore_verify: n_keep");
    if (nk == vp.M) continue;
    const Impl::Snap& sn = I.snaps[p][nk - 1];
    par_for(s.conv.size(), 1, [&](size_t a, size_t b) {
      for (size_t i = a; i < b; ++i) {
        if (sn.conv[i].empty()) continue;
        std::memcpy(s.conv[i].data(), sn.conv[i].data(), s.conv[i].size() * 4);
        std::memcpy(s.state[i].data(), sn.state[i].data(), s.state[i].size() * 4);
      }
    });
  }
  I.vparts.clear();
}

void GlmCpuEngine::rollback_batch(const int* n_keep, int S) {
  Impl& I = *p_;
  if (S != (int)I.vparts.size()) throw std::runtime_error("glm cpu rollback without a matching verify");
  const int H = I.c.hidden;
  for (int p = 0; p < S; ++p) {
    const Impl::VPart& vp = I.vparts[p];
    CpuSeq& s = *vp.s;
    const int M = vp.M, nk = n_keep[p];
    if (nk < 1 || nk > M || s.pos != vp.pos0 + M) throw std::runtime_error("glm cpu rollback without a matching verify");
    if (nk < M) {
      const Impl::Snap& sn = I.snaps[p][nk - 1];
      par_for(s.conv.size(), 1, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) { s.conv[i] = sn.conv[i]; std::memcpy(s.state[i].data(), sn.state[i].data(), s.state[i].size() * 4); }
      });
    }
    s.pos = vp.pos0 + nk;
    s.tokens.resize(s.tokens.size() - (size_t)(M - nk));
    if (I.mtp && I.mtp_k > 0) {
      if (nk >= 2) I.mtp_kv(s, vp.pos0, nk - 1, I.fh.data() + (size_t)vp.r0 * H, I.verify_ids.data() + vp.r0 + 1);
      std::memcpy(s.h_last.data(), I.fh.data() + (size_t)(vp.r0 + nk - 1) * H, (size_t)H * 4);
      s.h_last_valid = true;
    }
  }
  I.vparts.clear();
}

int GlmCpuEngine::mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf, float* logits0) {
  return p_->mtp_draft(s, tok, k, drafts, conf, logits0);
}
bool GlmCpuEngine::has_vision() const { return p_->c.vision_layers > 0 && p_->ck && p_->ck->has("model.visual.patch_embed.proj.weight"); }
void GlmCpuEngine::encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out) { p_->encode_image(patches, gh, gw, out); }

}  // namespace hive::glm::cpu
