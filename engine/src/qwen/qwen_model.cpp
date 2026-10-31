// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/qwen/qwen_model.h"
#include "hive/ckpt_quant.h"
#include "hive/glm/glm_moe.h"
#include "hive/qwen/fn_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace hive::qwen {

using json = nlohmann::json;

namespace {
const std::string P = "model.language_model.";
}  // namespace

QMat QwenModel::mat(const std::string& name) { return stack({name}); }

// HIVE_QWEN_DENSE8 = fp8 | int8 (default off): 8-bit copies of the dense projections for the decode GEMV (prefill keeps the bf16 GEMM).
//   Scales per row and group of HIVE_QWEN_DENSE8_GS columns (default: the whole row for fp8, 128 for int8). Not the router (expert choice)
//   and not lm_head unless HIVE_QWEN_DENSE8_HEAD=1.
void QwenModel::make_dense8() {
  const char* v = getenv("HIVE_QWEN_DENSE8");
  if (!v || !*v || !strcmp(v, "0") || !strcmp(v, "bf16")) return;
  dense8_ = !strcmp(v, "int8") ? 2 : 1;
  const int gs_env = getenv("HIVE_QWEN_DENSE8_GS") ? atoi(getenv("HIVE_QWEN_DENSE8_GS")) : 0;
  size_t bytes = 0, dropped = 0;
  int n = 0;
  // HIVE_QWEN_DENSE8_KEEP_BF16=1 keeps the bf16 originals beside the 8-bit copies (prefill GEMMs on the originals); by default they are
  //   freed — the 8-bit copy is the only weight, and a prefill GEMM expands it to bf16 in a scratch first (QwenEngine::lin). Keeping both
  //   cost 3.2 GiB at 96 GB: the expert cache then held 99.1 % of the experts and decode lost device routing (2026-10-10)
  const bool drop = !(getenv("HIVE_QWEN_DENSE8_KEEP_BF16") && atoi(getenv("HIVE_QWEN_DENSE8_KEEP_BF16")) > 0);
  auto conv = [&](QMat& m, bool keep_bf16 = false) {
    if (!m.buf.p) return;
    if (m.in % 64) return;
    int gs = gs_env > 0 ? gs_env : (dense8_ == 1 ? m.in : 128);
    gs = std::max(64, gs / 64 * 64);
    if (m.in % gs && gs != m.in) gs = m.in;
    const int ng = (m.in + gs - 1) / gs;
    m.q8.alloc((size_t)m.out * m.in); m.s8.alloc((size_t)m.out * ng * 4);
    fn_quant8(m.p(), m.out, m.in, dense8_, gs, m.q8.as<uint8_t>(), m.s8.as<float>(), 0);
    m.fmt8 = dense8_; m.gs8 = gs;
    bytes += m.q8.n + m.s8.n; ++n;
    dense8_max_ = std::max(dense8_max_, (size_t)m.out * m.in);
    if (drop && !keep_bf16) { CUDA_CHECK(cudaDeviceSynchronize()); dropped += m.buf.n; m.buf.free(); }
  };
  // the down projection (324 rows × 10240) stays bf16 unless HIVE_QWEN_DENSE8_HCDOWN=1 (split-K 8-bit GEMV, fn_gemv8_epi)
  static const bool hcdown = getenv("HIVE_QWEN_DENSE8_HCDOWN") && atoi(getenv("HIVE_QWEN_DENSE8_HCDOWN")) > 0;
  auto hc = [&](HcW& w) { conv(w.up); if (hcdown) conv(w.down_inj); };
  auto layer = [&](LayerW& L) {
    hc(L.hc_attn); hc(L.hc_mlp);
    conv(L.att.in); conv(L.att.out); conv(L.lin.in); conv(L.lin.out);
    conv(L.moe.sh_gu); conv(L.moe.sh_down);
    conv(L.plew.key); conv(L.plew.value);
  };
  // HIVE_QWEN_DENSE8_SCOPE=head: only lm_head gets the 8-bit copy (with HIVE_QWEN_DENSE8_HEAD=1). FP4-Local, 2026-10-31: its dense projections
  //   are MXFP8 as stored (fmt8 3, not touched here); INT8 copies of the remaining bf16 ones (hyper-connection up, PLE key / value, the MTP
  //   layer) failed the quality gate twice (164 · 166 against 168; the same with them bf16: 169)
  const char* scope = getenv("HIVE_QWEN_DENSE8_SCOPE");
  if (!(scope && !strcmp(scope, "head"))) {
    for (LayerW& L : layers_) layer(L);
    hc(mixer_);
    if (mtp_) { conv(mtp_->fc_emb); conv(mtp_->fc_hidden); layer(mtp_->layer); hc(mtp_->mixer); }
  }
  if (getenv("HIVE_QWEN_DENSE8_HEAD") && atoi(getenv("HIVE_QWEN_DENSE8_HEAD")) > 0) conv(lm_head_, true);   // bf16 kept: prefill head, the draft head's copy
  CUDA_CHECK(cudaDeviceSynchronize());
  vram_bytes_ = vram_bytes_ + bytes - dropped;
  fprintf(stderr, "[qwen] dense 8-bit weights (%s, group %s): %d matrices · %.2f GiB VRAM%s\n", dense8_ == 1 ? "FP8 e4m3" : "INT8",
          gs_env > 0 ? std::to_string(gs_env).c_str() : (dense8_ == 1 ? "row" : "128"), n, bytes / 1073741824.0,
          drop ? (" · bf16 originals freed (" + std::to_string((long long)(dropped >> 20)) + " MiB) — prefill expands them per GEMM").c_str()
               : " beside the bf16 originals (prefill)");
}

// Rows of several matrices with the same input width. BF16 parts are copied. When every part is MXFP8 (NVIDIA FP4-Local: e4m3 + ue8m0 per
//   32 columns) the stack keeps that format as the QMat's only copy (fmt8 3, hive/qwen/fn_gemv.h) — exact, half the bytes of bf16, and the
//   decode GEMV / prefill MXFP8 GEMM read it as stored. Any other mix (or HIVE_QWEN_MX_EXPAND=1, for A/B) is expanded to bf16 on the host:
//   exact for MXFP8, one rounding for NVFP4 (hive/ckpt_quant.h).
QMat QwenModel::stack(const std::vector<std::string>& names) {
  int rows = 0, in = -1;
  bool all_bf16 = true, all_mx = true;
  for (const auto& n : names) {
    HIVE_CHECK(n.size() > 6 && n.compare(n.size() - 6, 6, "weight") == 0, "qwen: stack expects .weight tensors: " + n);
    const std::string base = n.substr(0, n.size() - 6);
    const ckq::Dense f = ckq::dense_format(ckpt_, base);
    HIVE_CHECK(f != ckq::Dense::Other, "qwen: unsupported matrix format: " + n + " (" + ckpt_.get(n).dtype + ")");
    all_bf16 = all_bf16 && f == ckq::Dense::BF16;
    all_mx = all_mx && f == ckq::Dense::MXFP8;
    int64_t N = 0, K = 0;
    ckq::dense_shape(ckpt_, base, N, K);
    HIVE_CHECK(in < 0 || in == (int)K, "qwen: stacked matrices differ in width: " + n);
    in = (int)K; rows += (int)N;
  }
  static const bool mx_expand = getenv("HIVE_QWEN_MX_EXPAND") && atoi(getenv("HIVE_QWEN_MX_EXPAND")) > 0;
  QMat m; m.out = rows; m.in = in;
  if (all_mx && !mx_expand && in % 64 == 0) {
    m.q8.alloc((size_t)rows * in); m.s8.alloc((size_t)rows * (in / 32));
    size_t oq = 0, os = 0;
    for (const auto& n : names) {
      const uint8_t *q = nullptr, *sc = nullptr;
      int64_t N = 0, K = 0;
      ckq::mxfp8_parts(ckpt_, n.substr(0, n.size() - 6), q, sc, N, K);
      CUDA_CHECK(cudaMemcpy(m.q8.as<uint8_t>() + oq, q, (size_t)N * K, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(m.s8.as<uint8_t>() + os, sc, (size_t)N * (K / 32), cudaMemcpyHostToDevice));
      oq += (size_t)N * K; os += (size_t)N * (K / 32);
    }
    m.fmt8 = 3; m.gs8 = 32;
    vram_bytes_ += oq + os;
    dense8_max_ = std::max(dense8_max_, (size_t)rows * in);
    ++mx8_;
    return m;
  }
  m.buf.alloc((size_t)rows * in * 2);
  size_t off = 0;
  for (const auto& n : names) {
    const TensorInfo& t = ckpt_.get(n);
    if (all_bf16 || t.dtype == "BF16") {
      CUDA_CHECK(cudaMemcpy(m.buf.as<uint8_t>() + off, t.data, t.nbytes, cudaMemcpyHostToDevice));
      off += t.nbytes;
    } else {
      const std::vector<uint16_t> h = ckq::expand_bf16(ckpt_, n.substr(0, n.size() - 6), 16);
      CUDA_CHECK(cudaMemcpy(m.buf.as<uint8_t>() + off, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
      off += h.size() * 2;
    }
  }
  vram_bytes_ += off;
  return m;
}

// bf16 vector. F32 (the FP4-Local checkpoint keeps the hyper-connection and PLE norms in fp32) is rounded to bf16 — the kernels take bf16
//   norm weights; the NVFP4 checkpoint stores these norms in bf16 already.
DevBuf QwenModel::vec_bf16(const std::string& name) {
  const TensorInfo& t = ckpt_.get(name);
  HIVE_CHECK(t.dtype == "BF16" || t.dtype == "F32", "qwen: expected BF16 or F32: " + name);
  if (t.dtype == "F32") {
    const std::vector<float> f = ckq::vec_f32(ckpt_, name);
    std::vector<uint16_t> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = ckq::f32_bf16(f[i]);
    DevBuf b(h.size() * 2);
    CUDA_CHECK(cudaMemcpy(b.p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    vram_bytes_ += h.size() * 2;
    ++f32_rounded_;
    return b;
  }
  DevBuf b(t.nbytes);
  CUDA_CHECK(cudaMemcpy(b.p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  vram_bytes_ += t.nbytes;
  return b;
}

// (1 + w) norm weight: an F32 tensor (FP4-Local's hyper-connection and PLE norms — QAD-trained values that are not bf16 numbers) stays fp32
//   unless HIVE_QWEN_NORM_BF16=1 (A/B: the bf16 rounding)
NormW QwenModel::vec_norm(const std::string& name) {
  static const bool to_bf16 = getenv("HIVE_QWEN_NORM_BF16") && atoi(getenv("HIVE_QWEN_NORM_BF16")) > 0;
  NormW w;
  if (ckpt_.get(name).dtype == "F32" && !to_bf16) {
    w.buf = vec_f32(name);
    w.f32 = true;
    ++f32_kept_;
  } else {
    w.buf = vec_bf16(name);
  }
  return w;
}

DevBuf QwenModel::vec_f32(const std::string& name) {
  const TensorInfo& t = ckpt_.get(name);
  std::vector<float> h((size_t)t.numel());
  if (t.dtype == "F32") std::memcpy(h.data(), t.data, t.nbytes);
  else if (t.dtype == "BF16") { const bf16* s = reinterpret_cast<const bf16*>(t.data); for (size_t i = 0; i < h.size(); ++i) h[i] = bf2f(s[i]); }
  else throw std::runtime_error("qwen: unexpected dtype " + t.dtype + " for " + name);
  DevBuf b(h.size() * 4);
  CUDA_CHECK(cudaMemcpy(b.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  vram_bytes_ += h.size() * 4;
  return b;
}

DevBuf QwenModel::conv_f32(const std::string& name, int channels) {
  const TensorInfo& t = ckpt_.get(name);
  HIVE_CHECK(t.shape.size() == 3 && t.shape[0] == channels && t.shape[1] == 1 && t.shape[2] == 4, "qwen: conv1d weight shape " + name);
  return vec_f32(name);  // [C, 1, 4] row-major = [C, 4]
}

void QwenModel::load_hc(const std::string& p, HcW& w, bool inject) {
  std::vector<std::string> rows{p + "input_mix_weight_down.weight"};
  if (inject) rows.push_back(p + "block_inject_weight.weight");
  w.down_inj = stack(rows);
  w.up = mat(p + "input_mix_weight_up.weight");
  w.norm = vec_norm(p + "hc_norm.weight");
  w.inject = inject;
  HIVE_CHECK(w.down_inj.in == cfg_.hcw() && w.down_inj.out == cfg_.hc_rank + (inject ? cfg_.hc : 0) && w.up.out == cfg_.hcw() && w.up.in == cfg_.hc_rank,
             "qwen: hyper-connection shapes " + p);
}

void QwenModel::load_moe(const std::string& p, MoeW& w) {
  w.router = stack({p + "mlp.gate.weight", p + "mlp.shared_expert_gate.weight"});
  w.sh_gu = stack({p + "mlp.shared_expert.gate_proj.weight", p + "mlp.shared_expert.up_proj.weight"});
  w.sh_down = mat(p + "mlp.shared_expert.down_proj.weight");
  HIVE_CHECK(w.router.out == cfg_.n_experts + 1 && w.sh_gu.out == 2 * cfg_.shared_inter && w.sh_down.in == cfg_.shared_inter, "qwen: MoE shapes " + p);
}

void QwenModel::load_layer(const std::string& p, bool attn, LayerW& L) {
  const QwenConfig& c = cfg_;
  L.attn = attn;
  load_hc(p + "attn_hyper_connection.", L.hc_attn, true);
  load_hc(p + "mlp_hyper_connection.", L.hc_mlp, true);
  if (attn) {
    const std::string a = p + "self_attn.";
    L.att.in = stack({a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight", a + "indexer.index_qk_proj.weight"});
    HIVE_CHECK(L.att.in.out == 2 * c.q_width() + 2 * c.kv_width() + (c.idx_heads + 1) * c.idx_dim, "qwen: attention projection rows " + a);
    L.att.out = mat(a + "o_proj.weight");
    L.att.q_norm = vec_bf16(a + "q_norm.weight"); L.att.k_norm = vec_bf16(a + "k_norm.weight");
    L.att.iq_norm = vec_bf16(a + "indexer.q_layernorm.weight"); L.att.ik_norm = vec_bf16(a + "indexer.k_layernorm.weight");
  } else {
    const std::string a = p + "linear_attn.";
    L.lin.in = stack({a + "in_proj_qkv.weight", a + "in_proj_z.weight", a + "in_proj_b.weight", a + "in_proj_a.weight"});
    HIVE_CHECK(L.lin.in.out == c.lin_conv_ch() + c.lin_v_width() + 2 * c.lin_v_heads, "qwen: linear-attention projection rows " + a);
    L.lin.conv_w = conv_f32(a + "conv1d.weight", c.lin_conv_ch());
    L.lin.A_log = vec_f32(a + "A_log"); L.lin.dt_bias = vec_f32(a + "dt_bias");
    L.lin.norm = vec_bf16(a + "norm.weight");
    L.lin.out = mat(a + "out_proj.weight");
  }
  load_moe(p, L.moe);
}

QwenModel::QwenModel(const std::string& dir, bool load_mtp, int gpu_layers) : cfg_(QwenConfig::load(dir)), ckpt_(dir) {
  const QwenConfig& c = cfg_;
  { size_t fr = 0, tot = 0; CUDA_CHECK(cudaMemGetInfo(&fr, &tot)); free_at_load_ = fr; }
  gpu_layers_ = gpu_layers < 0 || gpu_layers > c.n_layers ? c.n_layers : gpu_layers;
  if (gpu_layers_ < c.n_layers) load_mtp = false;   // the MTP block reads the final stream of every layer — not with a CPU tail
  layers_.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; ++l) layers_[l].attn = c.is_attn[l];
  if (!c.ple_layer_ids.empty()) ple_layer_ = c.ple_layer_ids[0] - 1;
  if (ple_layer_ >= gpu_layers_) ple_layer_ = -1;   // the PLE layer is in the CPU tail (it opens the table)
  for (int l = 0; l < gpu_layers_; ++l) {
    const std::string p = P + "layers." + std::to_string(l) + ".";
    load_layer(p, c.is_attn[l], layers_[l]);
    if (l == ple_layer_) {
      LayerW& L = layers_[l];
      L.ple = true;
      const std::string q = p + "ple.";
      L.plew.key = mat(q + "key_proj.weight"); L.plew.value = mat(q + "value_proj.weight");
      L.plew.norm_key = vec_norm(q + "norm_key.weight"); L.plew.norm_query = vec_norm(q + "norm_query.weight");
      L.plew.norm_conv = vec_norm(q + "norm_conv.weight");
      HIVE_CHECK(L.plew.norm_key.f32 == L.plew.norm_query.f32, "qwen: PLE key / query norms in different precisions");
      L.plew.conv_w = conv_f32(q + "conv1d.weight", c.hcw());
      HIVE_CHECK(L.plew.key.out == c.hcw() && L.plew.key.in == c.ple_dim && L.plew.value.out == c.hidden, "qwen: PLE projection shapes");
    }
    if ((l + 1) % 12 == 0) fprintf(stderr, "[qwen] layers 0..%d loaded · VRAM %.1f GiB\n", l, vram_bytes_ / 1073741824.0);
  }
  {
    const TensorInfo& e = ckpt_.get(P + "embed_tokens.weight");
    HIVE_CHECK(e.dtype == "BF16" && e.shape[0] == c.vocab && e.shape[1] == c.hidden, "qwen: embed_tokens shape");
    embed_host_.resize((size_t)e.numel());
    std::memcpy(embed_host_.data(), e.data, e.nbytes);
  }
  lm_head_ = mat("lm_head.weight");
  HIVE_CHECK(lm_head_.out == c.vocab && lm_head_.in == c.hidden, "qwen: lm_head shape");
  load_hc(P + "hyper_connection_mixer.", mixer_, false);
  if (load_mtp && c.n_mtp > 0 && ckpt_.has("mtp.fc_embedding.weight")) {
    mtp_ = std::make_unique<MtpW>();
    MtpW& w = *mtp_;
    w.fc_emb = mat("mtp.fc_embedding.weight"); w.fc_hidden = mat("mtp.fc_hidden.weight");
    w.pre_emb_norm = vec_bf16("mtp.pre_fc_norm_embedding.weight"); w.pre_hidden_norm = vec_bf16("mtp.pre_fc_norm_hidden.weight");
    HIVE_CHECK(w.pre_hidden_norm.n == (size_t)c.hcw() * 2, "qwen: MTP pre_fc_norm_hidden width");
    // the NextN layer: full attention, no PLE; its routed experts are FP8 (below), everything else like a main layer
    const std::string p = "mtp.layers.0.";
    LayerW& L = w.layer;
    L.attn = true;
    load_hc(p + "attn_hyper_connection.", L.hc_attn, true);
    load_hc(p + "mlp_hyper_connection.", L.hc_mlp, true);
    {
      const std::string a = p + "self_attn.";
      L.att.in = stack({a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight", a + "indexer.index_qk_proj.weight"});
      L.att.out = mat(a + "o_proj.weight");
      L.att.q_norm = vec_bf16(a + "q_norm.weight"); L.att.k_norm = vec_bf16(a + "k_norm.weight");
      L.att.iq_norm = vec_bf16(a + "indexer.q_layernorm.weight"); L.att.ik_norm = vec_bf16(a + "indexer.k_layernorm.weight");
    }
    load_moe(p, L.moe);
    load_hc("mtp.hyper_connection_mixer.", w.mixer, false);
    const int E = c.n_experts, I = c.moe_inter, H = c.hidden;
    const char* names[3] = {"gate_proj", "up_proj", "down_proj"};
    if (ckpt_.get(p + "mlp.experts.0.gate_proj.weight").dtype == "U8") {
      // NVFP4 experts (NVIDIA FP4-Local: W4A16, plain E4M3 block scales + fp32 global): records in the main layers' format
      //   (glm::ExpertRecLayout — gate weights, scales, up weights, scales, down weights, scales, the three globals), one arena in VRAM
      const glm::ExpertRecLayout RL = glm::ExpertRecLayout::make(H, I);
      w.nvfp4 = true; w.rec_bytes = RL.total;
      w.expert_arena.alloc((size_t)E * RL.total);
      std::vector<uint8_t> rec(RL.total);
      for (int e = 0; e < E; ++e) {
        std::fill(rec.begin(), rec.end(), 0);
        const size_t wo[3] = {RL.w1, RL.w3, RL.w2}, so[3] = {RL.s1, RL.s3, RL.s2};
        float g[3];
        for (int j = 0; j < 3; ++j) {
          const std::string b = p + "mlp.experts." + std::to_string(e) + "." + names[j] + ".";
          const TensorInfo& tw = ckpt_.get(b + "weight");
          const TensorInfo& tg = ckpt_.get(b + "weight_scale_2");
          const int N = (int)tw.shape[0], K = (int)tw.shape[1] * 2;
          HIVE_CHECK(tw.dtype == "U8" && tg.dtype == "F32" && ((j < 2 && N == I && K == H) || (j == 2 && N == H && K == I)), "qwen: MTP NVFP4 expert shape " + b);
          std::memcpy(rec.data() + wo[j], tw.data, tw.nbytes);
          ckq::nvfp4_scales(ckpt_, b, N, K / 16, rec.data() + so[j]);
          std::memcpy(&g[j], tg.data, 4);
        }
        std::memcpy(rec.data() + RL.g, g, 12);
        CUDA_CHECK(cudaMemcpy(w.expert_arena.as<uint8_t>() + (size_t)e * RL.total, rec.data(), RL.total, cudaMemcpyHostToDevice));
      }
      vram_bytes_ += (size_t)E * RL.total;
      fprintf(stderr, "[qwen] MTP: NextN layer loaded · %d routed experts NVFP4 in VRAM (%.2f GiB)\n", E, (double)E * RL.total / 1073741824.0);
    } else {
    // FP8 experts: weight e4m3 [N, K] + weight_scale_inv BF16 [N/128, K/128] → fp32 scales (glm::Fp8BMat), one arena in VRAM
    const size_t wgu = (size_t)I * H, sgu = (size_t)((I + 127) / 128) * ((H + 127) / 128) * 4;
    const size_t wd = (size_t)H * I, sd = (size_t)((H + 127) / 128) * ((I + 127) / 128) * 4;
    const size_t per = (wgu * 2 + wd + 255) / 256 * 256 + (sgu * 2 + sd + 255) / 256 * 256 + 1024;
    w.expert_arena.alloc((size_t)E * per);
    uint8_t* base = w.expert_arena.as<uint8_t>();
    std::vector<float> sc;
    for (int e = 0; e < E; ++e) {
      uint8_t* q = base + (size_t)e * per;
      std::array<glm::Fp8BMat, 3> m{};
      for (int j = 0; j < 3; ++j) {
        const std::string b = p + "mlp.experts." + std::to_string(e) + "." + names[j] + ".weight";
        const TensorInfo& tw = ckpt_.get(b);
        const TensorInfo& ts = ckpt_.get(b + "_scale_inv");
        HIVE_CHECK(tw.dtype == "F8_E4M3" && (ts.dtype == "BF16" || ts.dtype == "F32"), "qwen: MTP expert must be FP8 with block scales: " + b);
        const int N = (int)tw.shape[0], K = (int)tw.shape[1];
        HIVE_CHECK((j < 2 && N == I && K == H) || (j == 2 && N == H && K == I), "qwen: MTP expert shape " + b);
        HIVE_CHECK(ts.shape.size() == 2 && ts.shape[0] == (N + 127) / 128 && ts.shape[1] == (K + 127) / 128, "qwen: MTP expert scale shape " + b);
        CUDA_CHECK(cudaMemcpy(q, tw.data, tw.nbytes, cudaMemcpyHostToDevice));
        uint8_t* sq = q + (tw.nbytes + 255) / 256 * 256;
        sc.resize((size_t)ts.numel());
        if (ts.dtype == "F32") std::memcpy(sc.data(), ts.data, ts.nbytes);
        else { const bf16* s = reinterpret_cast<const bf16*>(ts.data); for (size_t i = 0; i < sc.size(); ++i) sc[i] = bf2f(s[i]); }
        CUDA_CHECK(cudaMemcpy(sq, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
        m[j] = glm::Fp8BMat{q, reinterpret_cast<const float*>(sq), N, K};
        q = sq + (sc.size() * 4 + 255) / 256 * 256;
        HIVE_CHECK(q <= base + (size_t)(e + 1) * per, "qwen: MTP expert arena overflow");
      }
      w.experts.push_back(m);
    }
    vram_bytes_ += (size_t)E * per;
    fprintf(stderr, "[qwen] MTP: NextN layer loaded · %d routed experts FP8 in VRAM (%.2f GiB)\n", E, (double)E * per / 1073741824.0);
    }
  } else if (load_mtp) {
    fprintf(stderr, "[qwen] no MTP layer in the checkpoint — speculative decoding off\n");
  }
  make_dense8();
  if (mx8_ || f32_rounded_ || f32_kept_)
    fprintf(stderr, "[qwen] checkpoint formats: %d dense matrices kept as MXFP8 (e4m3 + ue8m0 / 32)%s · fp32 norms: %d kept fp32, %d rounded to bf16\n",
            mx8_, mx8_ ? " — decode GEMV and prefill read them as stored" : "", f32_kept_, f32_rounded_);
  const int n_attn = (int)std::count(c.is_attn.begin(), c.is_attn.end(), true);
  fprintf(stderr, "[qwen] model loaded: %d layers (%d linear · %d full attention) · PLE at layer %d · non-expert VRAM %.2f GiB · MTP %s%s\n", c.n_layers,
          c.n_layers - n_attn, n_attn, ple_layer_, vram_bytes_ / 1073741824.0, mtp_ ? "on" : "off",
          gpu_layers_ < c.n_layers ? (" · layers " + std::to_string(gpu_layers_) + ".." + std::to_string(c.n_layers - 1) + " on the CPU").c_str() : "");
}

}  // namespace hive::qwen
