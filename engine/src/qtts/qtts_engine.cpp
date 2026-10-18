// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// qtts engine (hive/qtts/qtts_engine.h): weights, buffers, prompt prefill, the per-frame graph (code predictor + talker step + sampling) and
//   the per-chunk codec graph. The math contract is oracle/qwen3_tts_oracle.py.
#include "hive/qtts/qtts_engine.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "hive/devbuf.h"
#include "hive/qtts/qtts_kernels.h"
#include "hive/safetensors.h"
#include "hive/vram_limit.h"
#include "nlohmann/json.hpp"

namespace hive::qtts {

using json = nlohmann::json;

// ---- config ---------------------------------------------------------------------------------------------------------------------------------
QttsConfig QttsConfig::load(const std::string& dir) {
  QttsConfig c;
  std::ifstream f(dir + "/config.json");
  if (!f) throw std::runtime_error("cannot open " + dir + "/config.json");
  json j = json::parse(f);
  c.tts_bos = j.value("tts_bos_token_id", c.tts_bos);
  c.tts_eos = j.value("tts_eos_token_id", c.tts_eos);
  c.tts_pad = j.value("tts_pad_token_id", c.tts_pad);
  const json& t = j.at("talker_config");
  c.hidden = t.at("hidden_size");
  c.layers = t.at("num_hidden_layers");
  c.heads = t.at("num_attention_heads");
  c.kv_heads = t.at("num_key_value_heads");
  c.head_dim = t.at("head_dim");
  c.inter = t.at("intermediate_size");
  c.vocab = t.at("vocab_size");
  c.text_vocab = t.at("text_vocab_size");
  c.text_hidden = t.at("text_hidden_size");
  c.groups = t.at("num_code_groups");
  c.eps = t.at("rms_norm_eps");
  c.theta = t.at("rope_theta");
  c.codec_bos = t.at("codec_bos_id");
  c.codec_eos = t.at("codec_eos_token_id");
  c.codec_pad = t.at("codec_pad_id");
  c.codec_think = t.at("codec_think_id");
  c.codec_nothink = t.at("codec_nothink_id");
  c.think_bos = t.at("codec_think_bos_id");
  c.think_eos = t.at("codec_think_eos_id");
  for (auto it = t.at("spk_id").begin(); it != t.at("spk_id").end(); ++it) {
    c.speakers[it.key()] = it.value().get<int>();
    c.speaker_order.push_back(it.key());
  }
  for (auto it = t.at("codec_language_id").begin(); it != t.at("codec_language_id").end(); ++it) c.languages[it.key()] = it.value().get<int>();
  if (t.contains("spk_is_dialect"))
    for (auto it = t.at("spk_is_dialect").begin(); it != t.at("spk_is_dialect").end(); ++it)
      c.dialect[it.key()] = it.value().is_string() ? it.value().get<std::string>() : std::string();
  const json& p = t.at("code_predictor_config");
  c.p_hidden = p.at("hidden_size");
  c.p_layers = p.at("num_hidden_layers");
  c.p_heads = p.at("num_attention_heads");
  c.p_kv_heads = p.at("num_key_value_heads");
  c.p_head_dim = p.at("head_dim");
  c.p_inter = p.at("intermediate_size");
  c.p_vocab = p.at("vocab_size");
  c.p_eps = p.at("rms_norm_eps");
  c.p_theta = p.at("rope_theta");

  std::ifstream g(dir + "/speech_tokenizer/config.json");
  if (!g) throw std::runtime_error("cannot open " + dir + "/speech_tokenizer/config.json");
  json s = json::parse(g);
  const json& d = s.at("decoder_config");
  c.c_quantizers = d.at("num_quantizers");
  c.c_codebook_size = d.at("codebook_size");
  c.c_codebook_dim = d.at("codebook_dim");
  c.c_latent = d.at("latent_dim");
  c.c_hidden = d.at("hidden_size");
  c.c_inter = d.at("intermediate_size");
  c.c_heads = d.at("num_attention_heads");
  c.c_head_dim = d.at("head_dim");
  c.c_layers = d.at("num_hidden_layers");
  c.c_window = d.at("sliding_window");
  c.c_decoder_dim = d.at("decoder_dim");
  c.c_eps = d.at("rms_norm_eps");
  c.c_theta = d.at("rope_theta");
  c.c_upsample_rates = d.at("upsample_rates").get<std::vector<int>>();
  c.c_upsampling_ratios = d.at("upsampling_ratios").get<std::vector<int>>();
  return c;
}

namespace {

constexpr int kChunk = 4;  // codec frames per streamed chunk

// Bump allocator: a measuring pass (base == nullptr) gives the size, the second pass hands out the pointers.
struct Arena {
  char* base = nullptr;
  size_t off = 0;
  template <class T>
  T* take(size_t n) {
    off = align_up(off, 256);
    T* p = base ? reinterpret_cast<T*>(base + off) : nullptr;
    off += n * sizeof(T);
    return p;
  }
};

std::vector<float> to_f32(const TensorInfo& t) {
  std::vector<float> v((size_t)t.numel());
  if (t.dtype == "F32") {
    memcpy(v.data(), t.data, v.size() * 4);
  } else if (t.dtype == "BF16") {
    const uint16_t* s = reinterpret_cast<const uint16_t*>(t.data);
    for (size_t i = 0; i < v.size(); ++i) {
      const uint32_t b = (uint32_t)s[i] << 16;
      memcpy(&v[i], &b, 4);
    }
  } else {
    throw std::runtime_error("unsupported dtype " + t.dtype + " for " + t.name);
  }
  return v;
}

void put_f32(float* dst, const std::vector<float>& v) { CUDA_CHECK(cudaMemcpy(dst, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); }
void put_bf16(bf16* dst, const std::vector<float>& v) {
  std::vector<bf16> b(v.size());
  for (size_t i = 0; i < v.size(); ++i) b[i] = __float2bfloat16(v[i]);
  CUDA_CHECK(cudaMemcpy(dst, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
}

struct Layer {
  float *in_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
  bf16 *wqkv = nullptr, *wo = nullptr, *wgu = nullptr, *wd = nullptr;
};

struct CLayer {  // codec transformer
  float *in_norm, *post_norm, *attn_scale, *mlp_scale;
  bf16 *wqkv, *wo, *wgu, *wd;
};
struct CRes {  // residual unit
  float *ea1, *ib1, *b1, *ea2, *ib2, *b2;
  bf16 *w1 /*[C][7C]*/, *w2 /*[C][C]*/;
  float* state;  // 6 * dilation rows
};
struct CBlock {
  int cin, cout, stride, tin, tout;
  float *ea, *ib, *bt;
  bf16* wt;  // [stride * cout][2 * cin]
  float* st_t;  // 1 row
  CRes res[3];
  float* out;  // [tout][cout]
};

}  // namespace

struct Engine::Impl {
  QttsConfig cfg;
  EngineOptions opt;
  EngineInfo inf;
  std::string dir;
  std::unique_ptr<SafetensorsFile> talker_file;  // kept open: text embedding rows are gathered on the host
  const TensorInfo* text_emb = nullptr;

  cudaStream_t st = nullptr;
  cublasHandle_t blas = nullptr;
  DevBuf wbuf, bbuf;  // weights, buffers
  int max_seq = 0, max_prompt = 0, splits = 16;
  int fs = 1920;      // samples per frame

  // ---- talker / predictor weights
  std::vector<Layer> tl, pl;
  float *t_norm = nullptr, *p_norm = nullptr, *mtp_b = nullptr, *tp_b1 = nullptr, *tp_b2 = nullptr;
  bf16 *codec_emb = nullptr, *codec_head = nullptr, *tp_w1 = nullptr, *tp_w2 = nullptr, *mtp_w = nullptr, *pred_emb = nullptr,
       *pred_head = nullptr;
  // ---- codec weights
  float *cb = nullptr, *Pf = nullptr, *Pr = nullptr;
  bf16* pre_w = nullptr;
  float* pre_b = nullptr;
  bf16 *in_proj_w = nullptr, *out_proj_w = nullptr;
  float *in_proj_b = nullptr, *out_proj_b = nullptr, *c_norm = nullptr;
  std::vector<CLayer> cl;
  bf16 *up_w[2] = {}, *pw1[2] = {}, *pw2[2] = {};
  float *up_b[2] = {}, *dw_w[2] = {}, *dw_b[2] = {}, *ln_w[2] = {}, *ln_b[2] = {}, *pw1_b[2] = {}, *pw2_b[2] = {}, *gamma[2] = {};
  bf16* dec0_w = nullptr;
  float* dec0_b = nullptr;
  std::vector<CBlock> blocks;
  float *out_ea = nullptr, *out_ib = nullptr, *out_w = nullptr;
  float out_b = 0.f;

  // ---- buffers
  Ctl* ctl = nullptr;
  Ctl* h_ctl = nullptr;  // pinned
  float *h = nullptr, *ph = nullptr, *qkv = nullptr, *q = nullptr, *att = nullptr, *act = nullptr, *part = nullptr, *logits = nullptr,
        *pad_vec = nullptr;
  int *tok = nullptr, *codes = nullptr, *forced = nullptr;
  uint8_t* hist = nullptr;
  float *rope_cos = nullptr, *rope_sin = nullptr, *c_cos = nullptr, *c_sin = nullptr;
  std::vector<bf16*> kc, vc, pkc, pvc;
  float* dump = nullptr;
  int dump_frames = 0;
  // prefill
  float *pf_h = nullptr, *pf_qkv = nullptr, *pf_q = nullptr, *pf_gu = nullptr, *pf_P = nullptr;
  bf16 *pf_x = nullptr, *pf_act = nullptr;
  int *pf_tslot = nullptr, *pf_cid = nullptr;
  bf16* h_stage = nullptr;    // pinned text rows
  int* h_recipe = nullptr;    // pinned (tslot, cid)
  // codec
  float *c_z0 = nullptr, *c_x1 = nullptr, *c_h = nullptr, *c_qkv = nullptr, *c_att = nullptr, *c_qt = nullptr, *c_gu = nullptr,
        *c_x2 = nullptr, *c_kr = nullptr, *c_vr = nullptr;
  float *c_u[2] = {}, *c_cnh = nullptr, *c_cno = nullptr, *c_d0 = nullptr, *c_v = nullptr;
  bf16 *c_A = nullptr, *c_cna = nullptr, *c_cng = nullptr, *c_a = nullptr;
  float *st_pre = nullptr, *st_dw[2] = {}, *st_dec0 = nullptr, *st_out = nullptr;
  float* state_base = nullptr;
  size_t state_bytes = 0;
  int16_t* c_pcm = nullptr;
  int16_t* h_pcm = nullptr;  // pinned
  int ring = 128;
  void* blas_ws = nullptr;
  size_t blas_ws_bytes = 8ull << 20;  // the codec GEMMs are small (largest 7680 x 96 x 672)

  cudaGraphExec_t g_frame = nullptr, g_codec = nullptr;
  bool active = false;
  int prompt_len = 0;

  // ------------------------------------------------------------------------------------------------------------------------------------------
  void layout_weights(Arena& a) {
    const QttsConfig& c = cfg;
    auto layer = [&](Layer& L, int hid, int H, int KV, int D, int inter) {
      L.in_norm = a.take<float>(hid);
      L.post_norm = a.take<float>(hid);
      L.q_norm = a.take<float>(D);
      L.k_norm = a.take<float>(D);
      L.wqkv = a.take<bf16>((size_t)(H + 2 * KV) * D * hid);
      L.wo = a.take<bf16>((size_t)hid * H * D);
      L.wgu = a.take<bf16>((size_t)2 * inter * hid);
      L.wd = a.take<bf16>((size_t)hid * inter);
    };
    if (!opt.codec_only) layout_talker(a, layer);
    layout_codec(a);
  }

  template <class LayerFn>
  void layout_talker(Arena& a, LayerFn&& layer) {
    const QttsConfig& c = cfg;
    tl.resize(c.layers);
    for (auto& L : tl) layer(L, c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter);
    t_norm = a.take<float>(c.hidden);
    codec_emb = a.take<bf16>((size_t)c.vocab * c.hidden);
    codec_head = a.take<bf16>((size_t)c.vocab * c.hidden);
    tp_w1 = a.take<bf16>((size_t)c.text_hidden * c.text_hidden);
    tp_b1 = a.take<float>(c.text_hidden);
    tp_w2 = a.take<bf16>((size_t)c.hidden * c.text_hidden);
    tp_b2 = a.take<float>(c.hidden);
    pl.resize(c.p_layers);
    for (auto& L : pl) layer(L, c.p_hidden, c.p_heads, c.p_kv_heads, c.p_head_dim, c.p_inter);
    p_norm = a.take<float>(c.p_hidden);
    mtp_w = a.take<bf16>((size_t)c.p_hidden * c.hidden);
    mtp_b = a.take<float>(c.p_hidden);
    pred_emb = a.take<bf16>((size_t)(c.groups - 1) * c.p_vocab * c.hidden);
    pred_head = a.take<bf16>((size_t)(c.groups - 1) * c.p_vocab * c.p_hidden);
  }

  void layout_codec(Arena& a) {
    const QttsConfig& c = cfg;
    const int cbd = c.c_codebook_dim / 2;
    cb = a.take<float>((size_t)c.c_quantizers * c.c_codebook_size * cbd);
    Pf = a.take<float>((size_t)c.c_codebook_dim * cbd);
    Pr = a.take<float>((size_t)c.c_codebook_dim * cbd);
    pre_w = a.take<bf16>((size_t)c.c_latent * 3 * c.c_codebook_dim);
    pre_b = a.take<float>(c.c_latent);
    in_proj_w = a.take<bf16>((size_t)c.c_hidden * c.c_latent);
    in_proj_b = a.take<float>(c.c_hidden);
    cl.resize(c.c_layers);
    const int chd = c.c_heads * c.c_head_dim;
    for (auto& L : cl) {
      L.in_norm = a.take<float>(c.c_hidden);
      L.post_norm = a.take<float>(c.c_hidden);
      L.attn_scale = a.take<float>(c.c_hidden);
      L.mlp_scale = a.take<float>(c.c_hidden);
      L.wqkv = a.take<bf16>((size_t)3 * chd * c.c_hidden);
      L.wo = a.take<bf16>((size_t)c.c_hidden * chd);
      L.wgu = a.take<bf16>((size_t)2 * c.c_inter * c.c_hidden);
      L.wd = a.take<bf16>((size_t)c.c_hidden * c.c_inter);
    }
    c_norm = a.take<float>(c.c_hidden);
    out_proj_w = a.take<bf16>((size_t)c.c_latent * c.c_hidden);
    out_proj_b = a.take<float>(c.c_latent);
    for (int i = 0; i < 2; ++i) {
      const int L = c.c_latent, r = c.c_upsampling_ratios[i];
      up_w[i] = a.take<bf16>((size_t)r * L * L);
      up_b[i] = a.take<float>(L);
      dw_w[i] = a.take<float>((size_t)L * 7);
      dw_b[i] = a.take<float>(L);
      ln_w[i] = a.take<float>(L);
      ln_b[i] = a.take<float>(L);
      pw1[i] = a.take<bf16>((size_t)4 * L * L);
      pw1_b[i] = a.take<float>(4 * L);
      pw2[i] = a.take<bf16>((size_t)4 * L * L);
      pw2_b[i] = a.take<float>(L);
      gamma[i] = a.take<float>(L);
    }
    dec0_w = a.take<bf16>((size_t)c.c_decoder_dim * 7 * c.c_latent);
    dec0_b = a.take<float>(c.c_decoder_dim);
    blocks.resize(c.c_upsample_rates.size());
    int T = kChunk;
    for (int r : c.c_upsampling_ratios) T *= r;
    for (size_t b = 0; b < blocks.size(); ++b) {
      CBlock& B = blocks[b];
      B.cin = c.c_decoder_dim >> b;
      B.cout = c.c_decoder_dim >> (b + 1);
      B.stride = c.c_upsample_rates[b];
      B.tin = T;
      B.tout = T * B.stride;
      T = B.tout;
      B.ea = a.take<float>(B.cin);
      B.ib = a.take<float>(B.cin);
      B.wt = a.take<bf16>((size_t)B.stride * B.cout * 2 * B.cin);
      B.bt = a.take<float>(B.cout);
      for (auto& R : B.res) {
        R.ea1 = a.take<float>(B.cout);
        R.ib1 = a.take<float>(B.cout);
        R.ea2 = a.take<float>(B.cout);
        R.ib2 = a.take<float>(B.cout);
        R.b1 = a.take<float>(B.cout);
        R.b2 = a.take<float>(B.cout);
        R.w1 = a.take<bf16>((size_t)B.cout * 7 * B.cout);
        R.w2 = a.take<bf16>((size_t)B.cout * B.cout);
      }
    }
    const int oc = blocks.back().cout;
    out_ea = a.take<float>(oc);
    out_ib = a.take<float>(oc);
    out_w = a.take<float>((size_t)7 * oc);
  }

  void layout_buffers(Arena& a) {
    const QttsConfig& c = cfg;
    ctl = a.take<Ctl>(1);
    h = a.take<float>(c.hidden);
    ph = a.take<float>(c.p_hidden);
    const int qkv_n = std::max((c.heads + 2 * c.kv_heads) * c.head_dim, (c.p_heads + 2 * c.p_kv_heads) * c.p_head_dim);
    qkv = a.take<float>(qkv_n);
    q = a.take<float>(std::max(c.heads * c.head_dim, c.p_heads * c.p_head_dim));
    att = a.take<float>(std::max(c.heads * c.head_dim, c.p_heads * c.p_head_dim));
    act = a.take<float>(std::max(c.inter, c.p_inter));
    part = a.take<float>((size_t)c.heads * splits * (c.head_dim + 2));
    logits = a.take<float>(std::max(c.vocab, c.p_vocab));
    pad_vec = a.take<float>(c.hidden);
    tok = a.take<int>(16);
    codes = a.take<int>((size_t)(max_seq + 2 * kChunk) * 16);
    forced = a.take<int>((size_t)(max_seq + kChunk) * 16);
    hist = a.take<uint8_t>(c.vocab);
    rope_cos = a.take<float>((size_t)max_seq * c.head_dim / 2);
    rope_sin = a.take<float>((size_t)max_seq * c.head_dim / 2);
    c_cos = a.take<float>((size_t)(max_seq + 2 * kChunk) * c.c_head_dim / 2);
    c_sin = a.take<float>((size_t)(max_seq + 2 * kChunk) * c.c_head_dim / 2);
    kc.resize(c.layers); vc.resize(c.layers);
    for (int l = 0; l < c.layers; ++l) {
      kc[l] = a.take<bf16>((size_t)c.kv_heads * max_seq * c.head_dim);
      vc[l] = a.take<bf16>((size_t)c.kv_heads * max_seq * c.head_dim);
    }
    pkc.resize(c.p_layers); pvc.resize(c.p_layers);
    for (int l = 0; l < c.p_layers; ++l) {
      pkc[l] = a.take<bf16>((size_t)c.p_kv_heads * c.groups * c.p_head_dim);
      pvc[l] = a.take<bf16>((size_t)c.p_kv_heads * c.groups * c.p_head_dim);
    }
    if (dump_frames > 0) dump = a.take<float>((size_t)dump_frames * 16 * 3072);
    // prefill (max_prompt rows; the text projection also needs max_prompt rows)
    const int P = max_prompt;
    pf_h = a.take<float>((size_t)P * c.hidden);
    pf_qkv = a.take<float>((size_t)P * (c.heads + 2 * c.kv_heads) * c.head_dim);
    pf_q = a.take<float>((size_t)P * c.heads * c.head_dim);
    pf_gu = a.take<float>((size_t)P * 2 * c.inter);
    pf_P = a.take<float>((size_t)P * c.hidden);
    pf_x = a.take<bf16>((size_t)P * std::max(c.hidden, c.heads * c.head_dim));
    pf_act = a.take<bf16>((size_t)P * c.inter);
    pf_tslot = a.take<int>(P);
    pf_cid = a.take<int>(P);
    // codec
    const int T0 = kChunk;
    c_z0 = a.take<float>((size_t)T0 * c.c_codebook_dim);
    c_x1 = a.take<float>((size_t)T0 * c.c_latent);
    c_h = a.take<float>((size_t)T0 * c.c_hidden);
    const int chd = c.c_heads * c.c_head_dim;
    c_qkv = a.take<float>((size_t)T0 * 3 * chd);
    c_att = a.take<float>((size_t)T0 * chd);
    c_qt = a.take<float>((size_t)T0 * chd);
    c_gu = a.take<float>((size_t)T0 * c.c_inter);
    c_x2 = a.take<float>((size_t)T0 * c.c_latent);
    int T = T0;
    size_t amax = (size_t)T0 * 3 * c.c_codebook_dim;
    for (int i = 0; i < 2; ++i) {
      amax = std::max(amax, (size_t)T * c.c_latent);
      T *= c.c_upsampling_ratios[i];
      c_u[i] = a.take<float>((size_t)T * c.c_latent);
    }
    c_cnh = a.take<float>((size_t)T * 4 * c.c_latent);
    c_cno = a.take<float>((size_t)T * c.c_latent);
    c_cna = a.take<bf16>((size_t)T * c.c_latent);
    c_cng = a.take<bf16>((size_t)T * 4 * c.c_latent);
    c_d0 = a.take<float>((size_t)T * c.c_decoder_dim);
    amax = std::max(amax, (size_t)T * 7 * c.c_latent);
    size_t vmax = 0;
    for (auto& B : blocks) {
      B.out = a.take<float>((size_t)B.tout * B.cout);
      amax = std::max(amax, (size_t)B.tin * 2 * B.cin);
      amax = std::max(amax, (size_t)B.tout * 7 * B.cout);
      vmax = std::max(vmax, (size_t)B.tout * B.cout);
    }
    c_v = a.take<float>(vmax);
    c_a = a.take<bf16>(vmax);
    c_A = a.take<bf16>(amax);
    c_pcm = a.take<int16_t>((size_t)kChunk * fs);
    // codec state (zeroed at every start)
    const size_t s0 = align_up(a.off, 256);
    a.off = s0;
    state_base = a.take<float>(0);
    st_pre = a.take<float>((size_t)2 * c.c_codebook_dim);
    for (int i = 0; i < 2; ++i) st_dw[i] = a.take<float>((size_t)6 * c.c_latent);
    st_dec0 = a.take<float>((size_t)6 * c.c_latent);
    for (auto& B : blocks) {
      B.st_t = a.take<float>(B.cin);
      int dil = 1;
      for (auto& R : B.res) { R.state = a.take<float>((size_t)6 * dil * B.cout); dil *= 3; }
    }
    st_out = a.take<float>((size_t)6 * blocks.back().cout);
    c_kr = a.take<float>((size_t)c.c_layers * c.c_heads * ring * c.c_head_dim);
    c_vr = a.take<float>((size_t)c.c_layers * c.c_heads * ring * c.c_head_dim);
    state_bytes = align_up(a.off, 256) - s0;
  }

  size_t weight_bytes_needed() {
    Arena a;
    layout_weights(a);
    return align_up(a.off, 256);
  }
  size_t buffer_bytes_needed() {
    Arena a;
    layout_buffers(a);
    return align_up(a.off, 256);
  }

  // ------------------------------------------------------------------------------------------------------------------------------------------
  void load_weights() {
    if (opt.codec_only) {
      load_codec();
      return;
    }
    const QttsConfig& c = cfg;
    SafetensorsFile& f = *talker_file;
    auto T = [&](const std::string& n) -> const TensorInfo& {
      const TensorInfo* t = f.find(n);
      if (!t) throw std::runtime_error("missing tensor " + n);
      return *t;
    };
    auto raw_bf16 = [&](bf16* dst, const TensorInfo& t) {
      if (t.dtype == "BF16") CUDA_CHECK(cudaMemcpy(dst, t.data, t.nbytes, cudaMemcpyHostToDevice));
      else put_bf16(dst, to_f32(t));
    };
    auto stack = [&](std::vector<Layer>& Ls, const std::string& pre, int hid, int H, int KV, int D, int inter) {
      for (size_t i = 0; i < Ls.size(); ++i) {
        const std::string p = pre + "layers." + std::to_string(i) + ".";
        Layer& L = Ls[i];
        put_f32(L.in_norm, to_f32(T(p + "input_layernorm.weight")));
        put_f32(L.post_norm, to_f32(T(p + "post_attention_layernorm.weight")));
        put_f32(L.q_norm, to_f32(T(p + "self_attn.q_norm.weight")));
        put_f32(L.k_norm, to_f32(T(p + "self_attn.k_norm.weight")));
        raw_bf16(L.wqkv, T(p + "self_attn.q_proj.weight"));
        raw_bf16(L.wqkv + (size_t)H * D * hid, T(p + "self_attn.k_proj.weight"));
        raw_bf16(L.wqkv + (size_t)(H + KV) * D * hid, T(p + "self_attn.v_proj.weight"));
        raw_bf16(L.wo, T(p + "self_attn.o_proj.weight"));
        raw_bf16(L.wgu, T(p + "mlp.gate_proj.weight"));
        raw_bf16(L.wgu + (size_t)inter * hid, T(p + "mlp.up_proj.weight"));
        raw_bf16(L.wd, T(p + "mlp.down_proj.weight"));
      }
    };
    stack(tl, "talker.model.", c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter);
    put_f32(t_norm, to_f32(T("talker.model.norm.weight")));
    raw_bf16(codec_emb, T("talker.model.codec_embedding.weight"));
    raw_bf16(codec_head, T("talker.codec_head.weight"));
    raw_bf16(tp_w1, T("talker.text_projection.linear_fc1.weight"));
    put_f32(tp_b1, to_f32(T("talker.text_projection.linear_fc1.bias")));
    raw_bf16(tp_w2, T("talker.text_projection.linear_fc2.weight"));
    put_f32(tp_b2, to_f32(T("talker.text_projection.linear_fc2.bias")));
    stack(pl, "talker.code_predictor.model.", c.p_hidden, c.p_heads, c.p_kv_heads, c.p_head_dim, c.p_inter);
    put_f32(p_norm, to_f32(T("talker.code_predictor.model.norm.weight")));
    raw_bf16(mtp_w, T("talker.code_predictor.small_to_mtp_projection.weight"));
    put_f32(mtp_b, to_f32(T("talker.code_predictor.small_to_mtp_projection.bias")));
    for (int i = 0; i < c.groups - 1; ++i) {
      raw_bf16(pred_emb + (size_t)i * c.p_vocab * c.hidden, T("talker.code_predictor.model.codec_embedding." + std::to_string(i) + ".weight"));
      raw_bf16(pred_head + (size_t)i * c.p_vocab * c.p_hidden, T("talker.code_predictor.lm_head." + std::to_string(i) + ".weight"));
    }
    text_emb = &T("talker.model.text_embedding.weight");
    if (text_emb->dtype != "BF16") throw std::runtime_error("text_embedding must be BF16");
    load_codec();
  }

  void load_codec() {
    const QttsConfig& c = cfg;
    SafetensorsFile f(dir + "/speech_tokenizer/model.safetensors");
    auto W = [&](const std::string& n) {
      const TensorInfo* t = f.find("decoder." + n);
      if (!t) throw std::runtime_error("missing tensor decoder." + n);
      return to_f32(*t);
    };
    auto exp_v = [](std::vector<float> v) { for (auto& x : v) x = std::exp(x); return v; };
    auto inv_exp_v = [](std::vector<float> v) { for (auto& x : v) x = 1.f / (std::exp(x) + 1e-9f); return v; };
    // causal conv [cout][cin][k] -> GEMM B [cout][k * cin]
    auto conv_b = [](const std::vector<float>& w, int cout, int cin, int k) {
      std::vector<float> b((size_t)cout * k * cin);
      for (int o = 0; o < cout; ++o)
        for (int i = 0; i < cin; ++i)
          for (int j = 0; j < k; ++j) b[((size_t)o * k + j) * cin + i] = w[((size_t)o * cin + i) * k + j];
      return b;
    };
    // causal transposed conv [cin][cout][k] (k = 2s or s) -> GEMM B [(jj, cout)][taps * cin]: taps (x[n-1], x[n]) for k = 2s, (x[n]) for k = s
    auto tconv_b = [](const std::vector<float>& w, int cin, int cout, int k, int s) {
      const int taps = k / s;
      std::vector<float> b((size_t)s * cout * taps * cin);
      for (int jj = 0; jj < s; ++jj)
        for (int o = 0; o < cout; ++o)
          for (int i = 0; i < cin; ++i) {
            const size_t row = ((size_t)jj * cout + o) * taps * cin;
            if (taps == 2) {
              b[row + i] = w[((size_t)i * cout + o) * k + jj + s];
              b[row + cin + i] = w[((size_t)i * cout + o) * k + jj];
            } else {
              b[row + i] = w[((size_t)i * cout + o) * k + jj];
            }
          }
      return b;
    };
    const int cbd = c.c_codebook_dim / 2;
    {
      std::vector<float> all((size_t)c.c_quantizers * c.c_codebook_size * cbd);
      for (int qi = 0; qi < c.c_quantizers; ++qi) {
        const std::string p = qi == 0 ? "quantizer.rvq_first.vq.layers.0._codebook." :
                                        "quantizer.rvq_rest.vq.layers." + std::to_string(qi - 1) + "._codebook.";
        auto es = W(p + "embedding_sum"), cu = W(p + "cluster_usage");
        for (int e = 0; e < c.c_codebook_size; ++e)
          for (int d = 0; d < cbd; ++d)
            all[((size_t)qi * c.c_codebook_size + e) * cbd + d] = es[(size_t)e * cbd + d] / std::max(cu[e], 1e-5f);
      }
      put_f32(cb, all);
    }
    put_f32(Pf, W("quantizer.rvq_first.output_proj.weight"));
    put_f32(Pr, W("quantizer.rvq_rest.output_proj.weight"));
    put_bf16(pre_w, conv_b(W("pre_conv.conv.weight"), c.c_latent, c.c_codebook_dim, 3));
    put_f32(pre_b, W("pre_conv.conv.bias"));
    put_bf16(in_proj_w, W("pre_transformer.input_proj.weight"));
    put_f32(in_proj_b, W("pre_transformer.input_proj.bias"));
    const int chd = c.c_heads * c.c_head_dim;
    for (int l = 0; l < c.c_layers; ++l) {
      const std::string p = "pre_transformer.layers." + std::to_string(l) + ".";
      CLayer& L = cl[l];
      put_f32(L.in_norm, W(p + "input_layernorm.weight"));
      put_f32(L.post_norm, W(p + "post_attention_layernorm.weight"));
      put_f32(L.attn_scale, W(p + "self_attn_layer_scale.scale"));
      put_f32(L.mlp_scale, W(p + "mlp_layer_scale.scale"));
      put_bf16(L.wqkv, W(p + "self_attn.q_proj.weight"));
      put_bf16(L.wqkv + (size_t)chd * c.c_hidden, W(p + "self_attn.k_proj.weight"));
      put_bf16(L.wqkv + (size_t)2 * chd * c.c_hidden, W(p + "self_attn.v_proj.weight"));
      put_bf16(L.wo, W(p + "self_attn.o_proj.weight"));
      put_bf16(L.wgu, W(p + "mlp.gate_proj.weight"));
      put_bf16(L.wgu + (size_t)c.c_inter * c.c_hidden, W(p + "mlp.up_proj.weight"));
      put_bf16(L.wd, W(p + "mlp.down_proj.weight"));
    }
    put_f32(c_norm, W("pre_transformer.norm.weight"));
    put_bf16(out_proj_w, W("pre_transformer.output_proj.weight"));
    put_f32(out_proj_b, W("pre_transformer.output_proj.bias"));
    for (int i = 0; i < 2; ++i) {
      const std::string p = "upsample." + std::to_string(i) + ".";
      const int L = c.c_latent, r = c.c_upsampling_ratios[i];
      put_bf16(up_w[i], tconv_b(W(p + "0.conv.weight"), L, L, r, r));
      put_f32(up_b[i], W(p + "0.conv.bias"));
      put_f32(dw_w[i], W(p + "1.dwconv.conv.weight"));
      put_f32(dw_b[i], W(p + "1.dwconv.conv.bias"));
      put_f32(ln_w[i], W(p + "1.norm.weight"));
      put_f32(ln_b[i], W(p + "1.norm.bias"));
      put_bf16(pw1[i], W(p + "1.pwconv1.weight"));
      put_f32(pw1_b[i], W(p + "1.pwconv1.bias"));
      put_bf16(pw2[i], W(p + "1.pwconv2.weight"));
      put_f32(pw2_b[i], W(p + "1.pwconv2.bias"));
      put_f32(gamma[i], W(p + "1.gamma"));
    }
    put_bf16(dec0_w, conv_b(W("decoder.0.conv.weight"), c.c_decoder_dim, c.c_latent, 7));
    put_f32(dec0_b, W("decoder.0.conv.bias"));
    for (size_t b = 0; b < blocks.size(); ++b) {
      CBlock& B = blocks[b];
      const std::string p = "decoder." + std::to_string(b + 1) + ".block.";
      put_f32(B.ea, exp_v(W(p + "0.alpha")));
      put_f32(B.ib, inv_exp_v(W(p + "0.beta")));
      put_bf16(B.wt, tconv_b(W(p + "1.conv.weight"), B.cin, B.cout, 2 * B.stride, B.stride));
      put_f32(B.bt, W(p + "1.conv.bias"));
      for (int r = 0; r < 3; ++r) {
        const std::string q = p + std::to_string(r + 2) + ".";
        CRes& R = B.res[r];
        put_f32(R.ea1, exp_v(W(q + "act1.alpha")));
        put_f32(R.ib1, inv_exp_v(W(q + "act1.beta")));
        put_f32(R.ea2, exp_v(W(q + "act2.alpha")));
        put_f32(R.ib2, inv_exp_v(W(q + "act2.beta")));
        put_bf16(R.w1, conv_b(W(q + "conv1.conv.weight"), B.cout, B.cout, 7));
        put_f32(R.b1, W(q + "conv1.conv.bias"));
        put_bf16(R.w2, W(q + "conv2.conv.weight"));
        put_f32(R.b2, W(q + "conv2.conv.bias"));
      }
    }
    const int nb = (int)blocks.size();
    const int oc = blocks.back().cout;
    put_f32(out_ea, exp_v(W("decoder." + std::to_string(nb + 1) + ".alpha")));
    put_f32(out_ib, inv_exp_v(W("decoder." + std::to_string(nb + 1) + ".beta")));
    {
      auto w = W("decoder." + std::to_string(nb + 2) + ".conv.weight");  // [1][oc][7]
      std::vector<float> o((size_t)7 * oc);
      for (int j = 0; j < 7; ++j)
        for (int ch = 0; ch < oc; ++ch) o[(size_t)j * oc + ch] = w[(size_t)ch * 7 + j];
      put_f32(out_w, o);
      out_b = W("decoder." + std::to_string(nb + 2) + ".conv.bias")[0];
    }
  }

  void init_tables() {
    const QttsConfig& c = cfg;
    auto table = [](int npos, int D, double theta, float* dc, float* ds) {
      const int half = D / 2;
      std::vector<float> co((size_t)npos * half), si((size_t)npos * half);
      for (int p = 0; p < npos; ++p)
        for (int i = 0; i < half; ++i) {
          const double inv = 1.0 / std::pow(theta, (2.0 * i) / D);
          const double a = (double)p * inv;
          co[(size_t)p * half + i] = (float)std::cos(a);
          si[(size_t)p * half + i] = (float)std::sin(a);
        }
      CUDA_CHECK(cudaMemcpy(dc, co.data(), co.size() * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(ds, si.data(), si.size() * 4, cudaMemcpyHostToDevice));
    };
    HIVE_CHECK(c.theta == c.p_theta && c.head_dim == c.p_head_dim, "talker and code predictor RoPE differ");
    table(max_seq, c.head_dim, c.theta, rope_cos, rope_sin);
    table(max_seq + 2 * kChunk, c.c_head_dim, c.c_theta, c_cos, c_sin);
  }

  // ---- the per-frame step ---------------------------------------------------------------------------------------------------------------------

  SampleSpec talker_spec(bool advance) const {
    SampleSpec s;
    s.V = cfg.vocab;
    s.sup_lo = cfg.vocab - 1024;
    s.sup_hi = cfg.vocab;
    s.keep_id = cfg.codec_eos;
    s.eos_id = cfg.codec_eos;
    s.cb = 0;
    s.first = 1;
    s.advance = advance ? 1 : 0;
    s.from_ctl = 1;
    return s;
  }

  void decoder_layer_step(const Layer& L, float* x, int hid, int H, int KV, int D, int inter, float eps, bf16* kcache, bf16* vcache, int cap,
                          const int* pos_dev, int pos_off, int nsplit) {
    GemvParams g;
    g.W = L.wqkv; g.N = (H + 2 * KV) * D; g.K = hid; g.x = x; g.ldx = hid; g.norm_w = L.in_norm; g.eps = eps; g.y = qkv; g.ldy = g.N;
    gemv(g, st);
    qk_norm_rope_store(qkv, 1, H, KV, D, L.q_norm, L.k_norm, eps, rope_cos, rope_sin, pos_dev, pos_off, q, kcache, vcache, cap, st);
    attn_decode(q, kcache, vcache, H, KV, D, cap, pos_dev, pos_off, nsplit, part, att, st);
    if (nsplit > 1) attn_combine(part, H, D, nsplit, att, st);
    GemvParams o;
    o.W = L.wo; o.N = hid; o.K = H * D; o.x = att; o.ldx = H * D; o.y = x; o.ldy = hid; o.mode = kGemvAccum;
    gemv(o, st);
    GemvParams gu;
    gu.W = L.wgu; gu.N = inter; gu.K = hid; gu.x = x; gu.ldx = hid; gu.norm_w = L.post_norm; gu.eps = eps; gu.y = act; gu.ldy = inter;
    gu.mode = kGemvSiluMul;
    gemv(gu, st);
    GemvParams dn;
    dn.W = L.wd; dn.N = hid; dn.K = inter; dn.x = act; dn.ldx = inter; dn.y = x; dn.ldy = hid; dn.mode = kGemvAccum;
    gemv(dn, st);
  }

  void frame_body() {
    const QttsConfig& c = cfg;
    // code predictor: row 0 = projected talker hidden (talker final norm), row 1 = codec_embedding[c0], row j >= 2 = predictor embedding
    //   j-2 of code j-1; rows j >= 1 sample code j through lm_head[j-1]
    for (int j = 0; j < c.groups; ++j) {
      GemvParams in;
      in.W = mtp_w; in.N = c.p_hidden; in.K = c.hidden; in.bias = mtp_b; in.y = ph; in.ldy = c.p_hidden;
      if (j == 0) {
        in.x = h; in.ldx = c.hidden; in.norm_w = t_norm; in.eps = c.eps;
      } else if (j == 1) {
        in.table = codec_emb; in.idx = tok;
      } else {
        in.table = pred_emb + (size_t)(j - 2) * c.p_vocab * c.hidden; in.idx = tok + (j - 1);
      }
      gemv(in, st);
      for (int l = 0; l < c.p_layers; ++l)
        decoder_layer_step(pl[l], ph, c.p_hidden, c.p_heads, c.p_kv_heads, c.p_head_dim, c.p_inter, c.p_eps, pkc[l], pvc[l], c.groups,
                           nullptr, j, 1);
      if (j >= 1) {
        GemvParams hd;
        hd.W = pred_head + (size_t)(j - 1) * c.p_vocab * c.p_hidden; hd.N = c.p_vocab; hd.K = c.p_hidden; hd.x = ph; hd.ldx = c.p_hidden;
        hd.norm_w = p_norm; hd.eps = c.p_eps; hd.y = logits; hd.ldy = c.p_vocab;
        gemv(hd, st);
        SampleSpec s;
        s.V = c.p_vocab; s.cb = j; s.from_ctl = 1;
        sample(logits, s, nullptr, ctl, tok + j, forced, dump, st);
      }
    }
    frame_input(codec_emb, pred_emb, c.p_vocab, pad_vec, tok, c.hidden, h, codes, hist, ctl, st);
    // talker decode step at ctl->pos
    for (int l = 0; l < c.layers; ++l)
      decoder_layer_step(tl[l], h, c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter, c.eps, kc[l], vc[l], max_seq, &ctl->pos, 0, splits);
    head_and_sample(true);
    CUDA_CHECK(cudaMemcpyAsync(h_ctl, ctl, sizeof(Ctl), cudaMemcpyDeviceToHost, st));
  }

  void head_and_sample(bool advance) {
    const QttsConfig& c = cfg;
    GemvParams hd;
    hd.W = codec_head; hd.N = c.vocab; hd.K = c.hidden; hd.x = h; hd.ldx = c.hidden; hd.norm_w = t_norm; hd.eps = c.eps; hd.y = logits;
    hd.ldy = c.vocab;
    gemv(hd, st);
    sample(logits, talker_spec(advance), hist, ctl, tok, forced, dump, st);
  }

  // ---- codec chunk ---------------------------------------------------------------------------------------------------------------------------
  void codec_body() {
    const QttsConfig& c = cfg;
    const int T0 = kChunk;
    const int cbd = c.c_codebook_dim / 2;
    codec_rvq(codes, ctl, T0, cb, c.c_codebook_size, cbd, Pf, Pr, c.c_codebook_dim, c_z0, st);
    // pre_conv k3
    im2col_causal(c_z0, st_pre, T0, c.c_codebook_dim, 3, 1, nullptr, nullptr, c_A, 3 * c.c_codebook_dim, st);
    save_state(c_z0, T0, c.c_codebook_dim, 2, st_pre, st);
    fill_bias(c_x1, T0, c.c_latent, pre_b, c.c_latent, st);
    gemm(blas, c_A, 3 * c.c_codebook_dim, pre_w, 3 * c.c_codebook_dim, c_x1, c.c_latent, T0, c.c_latent, 3 * c.c_codebook_dim, 1.f);
    // transformer
    const int chd = c.c_heads * c.c_head_dim;
    {
      GemvParams g;
      g.W = in_proj_w; g.N = c.c_hidden; g.K = c.c_latent; g.M = T0; g.x = c_x1; g.ldx = c.c_latent; g.bias = in_proj_b; g.y = c_h;
      g.ldy = c.c_hidden;
      gemv(g, st);
    }
    for (int l = 0; l < c.c_layers; ++l) {
      const CLayer& L = cl[l];
      GemvParams g;
      g.W = L.wqkv; g.N = 3 * chd; g.K = c.c_hidden; g.M = T0; g.x = c_h; g.ldx = c.c_hidden; g.norm_w = L.in_norm; g.eps = c.c_eps;
      g.y = c_qkv; g.ldy = 3 * chd;
      gemv(g, st);
      codec_attn(c_qkv, ctl, T0, c.c_heads, c.c_head_dim, c.c_window, ring, c_cos, c_sin,
                 c_kr + (size_t)l * c.c_heads * ring * c.c_head_dim, c_vr + (size_t)l * c.c_heads * ring * c.c_head_dim, c_qt, c_att, st);
      GemvParams o;
      o.W = L.wo; o.N = c.c_hidden; o.K = chd; o.M = T0; o.x = c_att; o.ldx = chd; o.scale = L.attn_scale; o.y = c_h; o.ldy = c.c_hidden;
      o.mode = kGemvAccum;
      gemv(o, st);
      GemvParams gu;
      gu.W = L.wgu; gu.N = c.c_inter; gu.K = c.c_hidden; gu.M = T0; gu.x = c_h; gu.ldx = c.c_hidden; gu.norm_w = L.post_norm;
      gu.eps = c.c_eps; gu.y = c_gu; gu.ldy = c.c_inter; gu.mode = kGemvSiluMul;
      gemv(gu, st);
      GemvParams dn;
      dn.W = L.wd; dn.N = c.c_hidden; dn.K = c.c_inter; dn.M = T0; dn.x = c_gu; dn.ldx = c.c_inter; dn.scale = L.mlp_scale; dn.y = c_h;
      dn.ldy = c.c_hidden; dn.mode = kGemvAccum;
      gemv(dn, st);
    }
    {
      GemvParams g;
      g.W = out_proj_w; g.N = c.c_latent; g.K = c.c_hidden; g.M = T0; g.x = c_h; g.ldx = c.c_hidden; g.norm_w = c_norm; g.eps = c.c_eps;
      g.bias = out_proj_b; g.y = c_x2; g.ldy = c.c_latent;
      gemv(g, st);
    }
    // 2 x (transposed conv k=s=2, ConvNeXt)
    const float* xin = c_x2;
    int T = T0;
    const int Lt = c.c_latent;
    for (int i = 0; i < 2; ++i) {
      const int r = c.c_upsampling_ratios[i];
      im2col_causal(xin, nullptr, T, Lt, 1, 1, nullptr, nullptr, c_A, Lt, st);
      fill_bias(c_u[i], T, r * Lt, up_b[i], Lt, st);
      gemm(blas, c_A, Lt, up_w[i], Lt, c_u[i], r * Lt, T, r * Lt, Lt, 1.f);
      T *= r;
      convnext_dw_ln(c_u[i], st_dw[i], T, Lt, dw_w[i], dw_b[i], ln_w[i], ln_b[i], c_cna, st);
      gemm(blas, c_cna, Lt, pw1[i], Lt, c_cnh, 4 * Lt, T, 4 * Lt, Lt, 0.f);
      bias_gelu_bf16(c_cnh, pw1_b[i], T, 4 * Lt, c_cng, st);
      gemm(blas, c_cng, 4 * Lt, pw2[i], 4 * Lt, c_cno, Lt, T, Lt, 4 * Lt, 0.f);
      scale_bias_residual(c_u[i], c_cno, pw2_b[i], gamma[i], T, Lt, st);
      xin = c_u[i];
    }
    // decoder conv k7
    const int Dd = c.c_decoder_dim;
    im2col_causal(xin, st_dec0, T, Lt, 7, 1, nullptr, nullptr, c_A, 7 * Lt, st);
    save_state(xin, T, Lt, 6, st_dec0, st);
    fill_bias(c_d0, T, Dd, dec0_b, Dd, st);
    gemm(blas, c_A, 7 * Lt, dec0_w, 7 * Lt, c_d0, Dd, T, Dd, 7 * Lt, 1.f);
    xin = c_d0;
    for (auto& B : blocks) {
      // SnakeBeta + causal transposed conv (k = 2s)
      im2col_causal(xin, B.st_t, B.tin, B.cin, 2, 1, B.ea, B.ib, c_A, 2 * B.cin, st);
      save_state(xin, B.tin, B.cin, 1, B.st_t, st);
      fill_bias(B.out, B.tin, B.stride * B.cout, B.bt, B.cout, st);
      gemm(blas, c_A, 2 * B.cin, B.wt, 2 * B.cin, B.out, B.stride * B.cout, B.tin, B.stride * B.cout, 2 * B.cin, 1.f);
      int dil = 1;
      for (auto& R : B.res) {
        const int Tt = B.tout, C = B.cout;
        im2col_causal(B.out, R.state, Tt, C, 7, dil, R.ea1, R.ib1, c_A, 7 * C, st);
        save_state(B.out, Tt, C, 6 * dil, R.state, st);
        gemm(blas, c_A, 7 * C, R.w1, 7 * C, c_v, C, Tt, C, 7 * C, 0.f);
        ru_mid(c_v, R.b1, R.ea2, R.ib2, R.b2, Tt, C, c_a, B.out, st);
        gemm(blas, c_a, C, R.w2, C, B.out, C, Tt, C, C, 1.f);
        dil *= 3;
      }
      xin = B.out;
    }
    const CBlock& Lb = blocks.back();
    codec_out(Lb.out, st_out, Lb.tout, Lb.cout, out_ea, out_ib, out_w, out_b, c_pcm, st);
    ctl_advance_codec(ctl, kChunk, st);
    CUDA_CHECK(cudaMemcpyAsync(h_pcm, c_pcm, (size_t)kChunk * fs * sizeof(int16_t), cudaMemcpyDeviceToHost, st));
  }

  void capture(cudaGraphExec_t* exec, const std::function<void()>& body) {
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
    body();
    cudaGraph_t g = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(st, &g));
    CUDA_CHECK(cudaGraphInstantiate(exec, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
  }

  bool graphs_on = true;
  void run_frame() {
    if (g_frame && graphs_on) CUDA_CHECK(cudaGraphLaunch(g_frame, st));
    else frame_body();
  }
  void run_codec() {
    if (g_codec && graphs_on) CUDA_CHECK(cudaGraphLaunch(g_codec, st));
    else codec_body();
  }

  void reset_codec() {
    CUDA_CHECK(cudaMemsetAsync(state_base, 0, state_bytes, st));
  }

  // ---- prefill ----------------------------------------------------------------------------------------------------------------------------------
  void prefill(int L) {
    const QttsConfig& c = cfg;
    const int qkv_n = (c.heads + 2 * c.kv_heads) * c.head_dim;
    for (int l = 0; l < c.layers; ++l) {
      const Layer& Ly = tl[l];
      rms_norm_bf16(pf_h, Ly.in_norm, c.eps, L, c.hidden, pf_x, st);
      gemm(blas, pf_x, c.hidden, Ly.wqkv, c.hidden, pf_qkv, qkv_n, L, qkv_n, c.hidden, 0.f);
      qk_norm_rope_store(pf_qkv, L, c.heads, c.kv_heads, c.head_dim, Ly.q_norm, Ly.k_norm, c.eps, rope_cos, rope_sin, nullptr, 0, pf_q, kc[l],
                         vc[l], max_seq, st);
      attn_prefill(pf_q, kc[l], vc[l], L, c.heads, c.kv_heads, c.head_dim, max_seq, pf_x, st);
      gemm(blas, pf_x, c.heads * c.head_dim, Ly.wo, c.heads * c.head_dim, pf_h, c.hidden, L, c.hidden, c.heads * c.head_dim, 1.f);
      rms_norm_bf16(pf_h, Ly.post_norm, c.eps, L, c.hidden, pf_x, st);
      gemm(blas, pf_x, c.hidden, Ly.wgu, c.hidden, pf_gu, 2 * c.inter, L, 2 * c.inter, c.hidden, 0.f);
      silu_mul_bf16(pf_gu, L, c.inter, pf_act, st);
      gemm(blas, pf_act, c.inter, Ly.wd, c.inter, pf_h, c.hidden, L, c.hidden, c.inter, 1.f);
    }
    CUDA_CHECK(cudaMemcpyAsync(h, pf_h + (size_t)(L - 1) * c.hidden, c.hidden * sizeof(float), cudaMemcpyDeviceToDevice, st));
  }
};

// ---- Engine ------------------------------------------------------------------------------------------------------------------------------------
Engine::Engine(const std::string& model_dir, const EngineOptions& opt) : d_(new Impl) {
  Impl& d = *d_;
  d.dir = model_dir;
  d.opt = opt;
  d.cfg = QttsConfig::load(model_dir);
  d.fs = d.cfg.frame_samples();
  d.splits = std::max(1, opt.attn_splits);
  d.max_prompt = std::max(32, opt.max_prompt);
  d.max_seq = std::max(d.max_prompt + 16, opt.max_seq);
  d.dump_frames = opt.dump_frames;
  HIVE_CHECK(d.cfg.groups == 16, "expected 16 code groups");

  // VRAM: context measured as NVML free before it exists minus cudaMemGetInfo free after (vram_limit.h)
  size_t limit = opt.vram_limit_mb ? opt.vram_limit_mb * vram_limit::kMiB : vram_limit::limit_bytes();
  size_t f0 = 0;
  bool have_f0 = false;
  {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    char pci[32] = {0};
    if (cudaDeviceGetPCIBusId(pci, sizeof pci, dev) == cudaSuccess) have_f0 = vram_limit::nvml_free(pci, &f0);
  }
  CUDA_CHECK(cudaFree(nullptr));
  // The context reserves local memory for the per-thread stack limit (default 1 KiB) on every resident thread: ~210 MiB of the ~550 MiB
  //   context on a 188-SM card (measured 2026-10-18: free +212 MiB after lowering it to 256 B). The qtts kernels use at most 32 B
  //   (ptxas -v) and the driver raises the limit by itself for a launch that needs more, so it only costs a reallocation if ever needed.
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 256));
  size_t f1 = 0, tot = 0;
  CUDA_CHECK(cudaMemGetInfo(&f1, &tot));
  d.inf.context_measured = have_f0 && f0 > f1;
  d.inf.context_bytes = d.inf.context_measured ? f0 - f1 : vram_limit::kCtxFallback;
  d.inf.limit_bytes = limit;

  const size_t wbytes = d.weight_bytes_needed();
  const size_t fixed = d.blas_ws_bytes + (64ull << 20);  // cuBLAS workspace + handle, graphs, allocator slack
  if (limit) {
    // lower max_seq until context + weights + buffers fit
    while (true) {
      const size_t need = d.inf.context_bytes + wbytes + d.buffer_bytes_needed() + fixed;
      if (need <= limit) break;
      if (d.max_seq <= d.max_prompt + 256) {
        char msg[256];
        snprintf(msg, sizeof msg, "VRAM limit %.0f MiB is too small: context %.0f + weights %.0f + buffers %.0f + fixed %.0f MiB",
                 limit / 1048576.0, d.inf.context_bytes / 1048576.0, wbytes / 1048576.0, d.buffer_bytes_needed() / 1048576.0,
                 fixed / 1048576.0);
        throw std::runtime_error(msg);
      }
      d.max_seq = std::max(d.max_prompt + 256, d.max_seq - 256);
    }
  }
  d.inf.max_seq = d.max_seq;
  d.inf.max_prompt = d.max_prompt;

  if (!opt.codec_only) d.talker_file = std::make_unique<SafetensorsFile>(model_dir + "/model.safetensors");
  d.wbuf.alloc(wbytes);
  {
    Arena a;
    a.base = d.wbuf.as<char>();
    d.layout_weights(a);
  }
  const size_t bbytes = d.buffer_bytes_needed();
  d.bbuf.alloc(bbytes);
  {
    Arena a;
    a.base = d.bbuf.as<char>();
    d.layout_buffers(a);
  }
  d.inf.weight_bytes = wbytes;
  d.inf.buffer_bytes = bbytes;
  CUDA_CHECK(cudaMemset(d.bbuf.p, 0, bbytes));
  d.load_weights();
  d.init_tables();

  CUDA_CHECK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
  if (cublasCreate(&d.blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
  cublasSetStream(d.blas, d.st);
  CUDA_CHECK(cudaMalloc(&d.blas_ws, d.blas_ws_bytes));
  cublasSetWorkspace(d.blas, d.blas_ws, d.blas_ws_bytes);
  CUDA_CHECK(cudaHostAlloc(&d.h_ctl, sizeof(Ctl), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&d.h_pcm, (size_t)kChunk * d.fs * sizeof(int16_t), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&d.h_stage, (size_t)d.max_prompt * d.cfg.text_hidden * sizeof(bf16), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&d.h_recipe, (size_t)2 * d.max_prompt * sizeof(int), cudaHostAllocDefault));
  memset(d.h_ctl, 0, sizeof(Ctl));
  d.h_ctl->done = 1;
  d.h_ctl->eos_frame = -1;

  // run both bodies once eagerly (cuBLAS picks its kernels outside of capture), then capture them
  {
    Ctl c{};
    c.pos = 0; c.max_frames = 1; c.max_pos = d.max_seq; c.eos_frame = -1;
    CUDA_CHECK(cudaMemcpyAsync(d.ctl, &c, sizeof c, cudaMemcpyHostToDevice, d.st));
    if (!opt.codec_only) d.frame_body();
    d.reset_codec();
    d.codec_body();
    CUDA_CHECK(cudaStreamSynchronize(d.st));
  }
  if (opt.graphs) {
    if (!opt.codec_only) d.capture(&d.g_frame, [&] { d.frame_body(); });
    d.capture(&d.g_codec, [&] { d.codec_body(); });
  }
  CUDA_CHECK(cudaStreamSynchronize(d.st));
  size_t f2 = 0;
  CUDA_CHECK(cudaMemGetInfo(&f2, &tot));
  d.inf.process_bytes = d.inf.context_measured ? f0 - f2 : d.inf.context_bytes + (f1 - f2);
  d.h_ctl->done = 1;
}

Engine::~Engine() {
  if (!d_) return;
  Impl& d = *d_;
  if (d.st) cudaStreamSynchronize(d.st);
  if (d.g_frame) cudaGraphExecDestroy(d.g_frame);
  if (d.g_codec) cudaGraphExecDestroy(d.g_codec);
  if (d.blas) cublasDestroy(d.blas);
  if (d.blas_ws) cudaFree(d.blas_ws);
  if (d.h_ctl) cudaFreeHost(d.h_ctl);
  if (d.h_pcm) cudaFreeHost(d.h_pcm);
  if (d.h_stage) cudaFreeHost(d.h_stage);
  if (d.h_recipe) cudaFreeHost(d.h_recipe);
  if (d.st) cudaStreamDestroy(d.st);
}

const QttsConfig& Engine::config() const { return d_->cfg; }
const EngineInfo& Engine::info() const { return d_->inf; }
int Engine::prompt_rows() const { return d_->prompt_len; }
int Engine::frames() const { return d_->h_ctl->frame; }

void Engine::start(const int32_t* ids, int n_ids, const int32_t* instruct, int n_instruct, int speaker_id, int language_id, uint64_t seed,
                   const SamplingParams& sp, const int32_t* forced, int n_forced) {
  Impl& d = *d_;
  const QttsConfig& c = d.cfg;
  if (d.opt.codec_only) throw std::logic_error("engine was built codec-only");
  if (n_ids < 8) throw std::invalid_argument("prompt ids must hold the assistant template (at least 8 tokens)");
  if (n_instruct < 0 || (n_instruct > 0 && !instruct)) throw std::invalid_argument("bad instruct ids");
  if (speaker_id >= c.vocab || language_id >= c.vocab) throw std::invalid_argument("speaker / language id out of the codec vocabulary");
  for (int i = 0; i < n_ids; ++i)
    if (ids[i] < 0 || ids[i] >= c.text_vocab) throw std::invalid_argument("token id out of range");
  for (int i = 0; i < n_instruct; ++i)
    if (instruct[i] < 0 || instruct[i] >= c.text_vocab) throw std::invalid_argument("instruct token id out of range");
  const int n_text = n_ids - 8;
  std::vector<int> codec_list;
  if (language_id >= 0) codec_list = {c.codec_think, c.think_bos, language_id, c.think_eos};
  else codec_list = {c.codec_nothink, c.think_bos, c.think_eos};
  if (speaker_id >= 0) codec_list.push_back(speaker_id);
  codec_list.push_back(c.codec_pad);
  codec_list.push_back(c.codec_bos);
  const int nc = (int)codec_list.size();
  const int R = n_instruct + 3 + n_text + 3;  // projected text rows
  const int L = n_instruct + 3 + (nc - 1) + (n_text + 1) + 1;
  if (R > d.max_prompt || L > d.max_prompt) throw std::invalid_argument("prompt too long (" + std::to_string(L) + " rows, limit " +
                                                                        std::to_string(d.max_prompt) + ")");
  if (n_forced > d.max_seq - L - 1) throw std::invalid_argument("too many forced frames");
  // gather the text embedding rows (bf16) on the host
  const size_t row_bytes = (size_t)c.text_hidden * sizeof(bf16);
  std::vector<int> tid;
  tid.reserve(R);
  for (int i = 0; i < n_instruct; ++i) tid.push_back(instruct[i]);
  for (int i = 0; i < 3 + n_text; ++i) tid.push_back(ids[i]);
  tid.push_back(c.tts_bos);
  tid.push_back(c.tts_eos);
  tid.push_back(c.tts_pad);
  for (int r = 0; r < R; ++r) memcpy(d.h_stage + (size_t)r * c.text_hidden, d.text_emb->data + (size_t)tid[r] * row_bytes, row_bytes);
  const int s_bos = R - 3, s_eos = R - 2, s_pad = R - 1;
  int* ts = d.h_recipe;
  int* ci = d.h_recipe + d.max_prompt;
  int r = 0;
  for (int i = 0; i < n_instruct; ++i, ++r) { ts[r] = i; ci[r] = -1; }
  for (int i = 0; i < 3; ++i, ++r) { ts[r] = n_instruct + i; ci[r] = -1; }
  for (int j = 0; j < nc - 1; ++j, ++r) { ts[r] = j < nc - 2 ? s_pad : s_bos; ci[r] = codec_list[j]; }
  for (int i = 0; i < n_text; ++i, ++r) { ts[r] = n_instruct + 3 + i; ci[r] = c.codec_pad; }
  ts[r] = s_eos; ci[r] = c.codec_pad; ++r;
  ts[r] = s_pad; ci[r] = c.codec_bos; ++r;
  HIVE_CHECK(r == L, "prompt row count");
  d.prompt_len = L;

  cudaStream_t st = d.st;
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaMemcpyAsync(d.pf_x, d.h_stage, (size_t)R * row_bytes, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(d.pf_tslot, ts, L * sizeof(int), cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(d.pf_cid, ci, L * sizeof(int), cudaMemcpyHostToDevice, st));
  // text projection: P = W2 silu(W1 e + b1) + b2   (pf_q / pf_act as scratch)
  gemm(d.blas, d.pf_x, c.text_hidden, d.tp_w1, c.text_hidden, d.pf_q, c.text_hidden, R, c.text_hidden, c.text_hidden, 0.f);
  bias_silu_bf16(d.pf_q, d.tp_b1, R, c.text_hidden, d.pf_act, st);
  gemm(d.blas, d.pf_act, c.text_hidden, d.tp_w2, c.text_hidden, d.pf_P, c.hidden, R, c.hidden, c.text_hidden, 0.f);
  bias_add(d.pf_P, d.tp_b2, R, c.hidden, st);
  CUDA_CHECK(cudaMemcpyAsync(d.pad_vec, d.pf_P + (size_t)s_pad * c.hidden, c.hidden * sizeof(float), cudaMemcpyDeviceToDevice, st));
  assemble_rows(d.pf_P, d.codec_emb, d.pf_tslot, d.pf_cid, L, c.hidden, d.pf_h, st);

  // generation state
  Ctl x{};
  x.pos = L;
  x.frame = 0;
  x.done = 0;
  const int room = d.max_seq - L - 1;
  x.max_frames = sp.max_frames > 0 ? std::min(sp.max_frames, room) : room;
  x.max_pos = d.max_seq;
  x.forced_n = 0;
  if (forced && n_forced > 0) {
    CUDA_CHECK(cudaMemcpyAsync(d.forced, forced, (size_t)n_forced * 16 * sizeof(int), cudaMemcpyHostToDevice, st));
    x.forced_n = n_forced;
    x.max_frames = n_forced;
  }
  x.dump = d.dump ? 1 : 0;
  x.dump_frames = d.dump_frames;
  x.seed = seed;
  x.rng = 0;
  x.codec_frame = 0;
  x.eos_frame = -1;
  x.temperature[0] = sp.temperature;
  x.temperature[1] = sp.sub_temperature;
  x.top_k[0] = sp.top_k;
  x.top_k[1] = sp.sub_top_k;
  x.rep_penalty = sp.rep_penalty;
  x.min_frames = sp.min_frames;
  CUDA_CHECK(cudaMemcpyAsync(d.ctl, &x, sizeof x, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemsetAsync(d.hist, 0, c.vocab, st));
  d.reset_codec();
  d.prefill(L);
  d.head_and_sample(false);
  CUDA_CHECK(cudaMemcpyAsync(d.h_ctl, d.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  d.active = true;
}

int Engine::next_chunk(int16_t* pcm, int max_samples, const volatile int32_t* cancel) {
  Impl& d = *d_;
  if (!d.active) return 0;
  int produced = 0;
  while (produced < kChunk && !d.h_ctl->done) {
    if (cancel && *cancel) {
      d.active = false;
      return -2;
    }
    d.run_frame();
    CUDA_CHECK(cudaStreamSynchronize(d.st));
    ++produced;
  }
  if (produced == 0) {
    d.active = false;
    return 0;
  }
  d.run_codec();
  CUDA_CHECK(cudaStreamSynchronize(d.st));
  const int n = std::min(produced * d.fs, max_samples);
  memcpy(pcm, d.h_pcm, (size_t)n * sizeof(int16_t));
  return n;
}

void Engine::use_graphs(bool on) { d_->graphs_on = on; }

std::vector<int32_t> Engine::codes() const {
  const Impl& d = *d_;
  std::vector<int32_t> v((size_t)d.h_ctl->frame * 16);
  if (!v.empty()) CUDA_CHECK(cudaMemcpy(v.data(), d.codes, v.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  return v;
}

std::vector<float> Engine::logits() const {
  const Impl& d = *d_;
  if (!d.dump) return {};
  const int n = std::min(d.h_ctl->frame + 1, d.dump_frames);
  std::vector<float> v((size_t)n * 16 * 3072);
  CUDA_CHECK(cudaMemcpy(v.data(), d.dump, v.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return v;
}

int Engine::decode_codes(const int32_t* codes, int n_frames, int16_t* pcm, int max_samples) {
  Impl& d = *d_;
  if (n_frames <= 0) return 0;
  if (n_frames > d.max_seq) throw std::invalid_argument("too many frames");
  cudaStream_t st = d.st;
  CUDA_CHECK(cudaStreamSynchronize(st));
  d.active = false;
  CUDA_CHECK(cudaMemcpyAsync(d.codes, codes, (size_t)n_frames * 16 * sizeof(int), cudaMemcpyHostToDevice, st));
  Ctl x{};
  x.done = 1;
  x.eos_frame = -1;
  x.max_pos = d.max_seq;
  CUDA_CHECK(cudaMemcpyAsync(d.ctl, &x, sizeof x, cudaMemcpyHostToDevice, st));
  d.reset_codec();
  int out = 0;
  for (int f0 = 0; f0 < n_frames; f0 += kChunk) {
    d.run_codec();
    CUDA_CHECK(cudaStreamSynchronize(st));
    const int n = std::min(std::min(kChunk, n_frames - f0) * d.fs, max_samples - out);
    if (n <= 0) break;
    memcpy(pcm + out, d.h_pcm, (size_t)n * sizeof(int16_t));
    out += n;
  }
  *d.h_ctl = x;
  return out;
}

}  // namespace hive::qtts
