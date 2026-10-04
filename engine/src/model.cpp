// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/model.h"

#include <cmath>
#include <cstring>
#include <fstream>

#include "hive/model_kernels.h"
#include "nlohmann/json.hpp"

namespace hive {

using json = nlohmann::json;

Model::~Model() {
  if (embed_owned_) cudaFreeHost(embed_owned_);
}

bool Config::is_kv_source(int l) const {
  for (int x : kv_source_layers) if (x == l) return true;
  return false;
}
bool Config::is_index_source(int l) const {
  for (int x : index_source_layers) if (x == l) return true;
  return false;
}

Config Config::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open config " + path);
  json j = json::parse(f);
  json t = j.contains("text_config") ? j["text_config"] : j;
  Config c;
  auto gi = [&](const char* k, int& v) { if (t.contains(k)) v = t[k].get<int>(); };
  auto gf = [&](const char* k, float& v) { if (t.contains(k)) v = t[k].get<float>(); };
  gi("hidden_size", c.dim); gi("num_hidden_layers", c.n_layers); gi("num_attention_heads", c.n_heads); gi("head_dim", c.head_dim);
  gi("qk_rope_head_dim", c.rd); gi("q_lora_rank", c.q_lora_rank); gi("o_groups", c.o_groups); gi("o_lora_rank", c.o_lora_rank);
  gi("sliding_window", c.window); gi("vocab_size", c.vocab); gi("moe_intermediate_size", c.moe_inter); gi("n_routed_experts", c.n_routed);
  gi("num_experts_per_tok", c.n_act); gi("n_shared_experts", c.n_shared); gi("index_n_heads", c.index_n_heads);
  gi("index_head_dim", c.index_head_dim); gi("index_topk", c.index_topk); gi("candidate_source_layer_id", c.cand_source_layer);
  gi("candidate_topk_blocks", c.cand_topk_blocks); gi("candidate_block_size", c.cand_block); gi("hc_mult", c.hc);
  gi("hc_sinkhorn_iters", c.sinkhorn_iters); gf("hc_eps", c.hc_eps); gf("rms_norm_eps", c.norm_eps);
  gf("routed_scaling_factor", c.route_scale); gf("swiglu_limit", c.swiglu_limit); gf("rope_theta", c.rope_theta);
  gf("compress_rope_theta", c.compress_rope_theta); gi("image_token_id", c.image_token_id);
  gi("engram_max_ngram_size", c.engram_max_ngram); gi("engram_n_heads", c.engram_n_heads); gi("engram_head_dim", c.engram_head_dim);
  gi("engram_vocab_size", c.engram_vocab_size); gi("engram_compressed_vocab_size", c.engram_compressed_vocab);
  gi("engram_pad_token_id", c.engram_pad_id); gi("dspark_block_size", c.dspark_block); gi("dspark_n_routed_experts", c.dspark_experts);
  gi("dspark_num_experts_per_tok", c.dspark_act); gi("dspark_noise_token_id", c.dspark_noise_token); gi("dspark_markov_rank", c.dspark_markov_rank);
  if (t.contains("max_position_embeddings")) c.max_position = t["max_position_embeddings"].get<int64_t>();
  if (t.contains("rope_scaling")) {
    auto r = t["rope_scaling"];
    c.rope_factor = r["factor"].get<float>();
    c.original_seq_len = r["original_max_position_embeddings"].get<int64_t>();
    c.beta_fast = r["beta_fast"].get<float>();
    c.beta_slow = r["beta_slow"].get<float>();
  }
  for (auto& v : t["compress_ratios"]) c.compress_ratios.push_back(v.get<int>());
  for (auto& v : t["kv_source_layer_ids"]) c.kv_source_layers.push_back(v.get<int>());
  for (auto& v : t["index_source_layer_ids"]) c.index_source_layers.push_back(v.get<int>());
  for (auto& v : t["engram_layer_ids"]) c.engram_layer_ids.push_back(v.get<int>());
  for (auto& v : t["engram_num_embeddings"]) c.engram_num_embeddings.push_back(v.get<int64_t>());
  if (t.contains("dspark_target_layer_ids")) for (auto& v : t["dspark_target_layer_ids"]) c.dspark_targets.push_back(v.get<int>());
  if (j.contains("vision_config")) {
    auto v = j["vision_config"];
    c.vision_layers = v["num_hidden_layers"].get<int>();
    c.vision_dim = v["hidden_size"].get<int>();
    c.vision_heads = v["num_attention_heads"].get<int>();
    c.vision_inter = v["intermediate_size"].get<int>();
    c.vision_patch = v["patch_size"].get<int>();
    c.vision_downsample = v["downsample_ratio"].get<int>();
    c.vision_max_tokens = v["max_image_tokens"].get<int>();
    c.vision_min_pixels = v["min_pixels"].get<int>();
    c.vision_rope_theta = v["rope_theta"].get<float>();
  }
  HIVE_CHECK((int)c.compress_ratios.size() >= c.n_layers, "compress_ratios");
  return c;
}

// ---------------------------------------------------------------------------------------------------------------------
Model::Model(const std::string& dir, int max_layer, int64_t rope_len) : cfg_(Config::load(dir + "/config.json")), ckpt_(dir) {
  rope_req_ = rope_len;
  {  // embedding table (1.3 GB bf16) into pinned host memory — using the mmap directly cost 3 ms per token in page faults on row lookups (measured)
    const TensorInfo& t = ckpt_.get("embed.weight");
    void* p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, t.nbytes, cudaHostAllocDefault));
    memcpy(p, t.data, t.nbytes);
    embed_host_ = reinterpret_cast<const bf16*>(p);
    embed_owned_ = p;
    ckpt_.close_shard("embed.weight");
  }
  try {
    norm_ = load_raw("norm.weight");
    head_ = load_raw("head.weight");
  } catch (const std::exception& e) {
    fprintf(stderr, "[model] head/norm shard missing (partial checkpoint): %s\n", e.what());
  }
  build_rope();
  int nl = max_layer < 0 ? cfg_.n_layers : std::min(max_layer + 1, cfg_.n_layers);
  for (int l = 0; l < nl; ++l) load_layer(l);
}

DevBuf Model::load_raw(const std::string& name) {
  const TensorInfo& t = ckpt_.get(name);
  DevBuf b(t.nbytes);
  CUDA_CHECK(cudaMemcpy(b.p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  return b;
}

DevBuf Model::load_as_f32(const std::string& name) {
  const TensorInfo& t = ckpt_.get(name);
  std::vector<float> tmp(t.numel());
  if (t.dtype == "F32") memcpy(tmp.data(), t.data, t.nbytes);
  else if (t.dtype == "BF16") {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(t.data);
    for (size_t i = 0; i < tmp.size(); ++i) { uint32_t u = (uint32_t)p[i] << 16; memcpy(&tmp[i], &u, 4); }
  } else throw std::runtime_error("load_as_f32 dtype " + t.dtype + " " + name);
  DevBuf b(tmp.size() * 4);
  CUDA_CHECK(cudaMemcpy(b.p, tmp.data(), tmp.size() * 4, cudaMemcpyHostToDevice));
  return b;
}

Fp8Linear Model::load_fp8(const std::string& name) {
  const TensorInfo& w = ckpt_.get(name + ".weight");
  const TensorInfo& s = ckpt_.get(name + ".scale");
  HIVE_CHECK(w.dtype == "F8_E4M3" && s.dtype == "F8_E8M0", "fp8 linear dtype " + name);
  HIVE_CHECK(w.shape.size() == 2, "fp8 shape");
  Fp8Linear L;
  L.N = (int)w.shape[0];
  L.K = (int)w.shape[1];
  HIVE_CHECK(s.shape[0] == (L.N + 31) / 32 && s.shape[1] == (L.K + 31) / 32, "fp8 scale shape " + name);
  HIVE_CHECK(L.N % 32 == 0 && L.K % 32 == 0, "fp8 dims " + name);
  L.w.alloc(w.nbytes);
  CUDA_CHECK(cudaMemcpy(L.w.p, w.data, w.nbytes, cudaMemcpyHostToDevice));
  DevBuf tmp(s.nbytes);
  CUDA_CHECK(cudaMemcpy(tmp.p, s.data, s.nbytes, cudaMemcpyHostToDevice));
  L.s.alloc((size_t)L.N * (L.K / 32));
  k::expand_block_scale(tmp.as<uint8_t>(), L.N, L.K, L.s.as<uint8_t>(), 0);
  CUDA_CHECK(cudaStreamSynchronize(0));
  return L;
}

void Model::build_rope() {
  // Reference precompute_freqs_cis: dim=rd, seqlen=max_seq_len. Two tables: compressed (YaRN, 160000) and window (10000, no YaRN).
  rope_len_ = std::min<int64_t>(cfg_.max_position, rope_req_);
  const int half = cfg_.rd / 2;
  auto make = [&](bool yarn, float base) {
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) freqs[i] = 1.0f / powf(base, (float)(2 * i) / (float)cfg_.rd);
    if (yarn) {
      const int dim = cfg_.rd;
      auto corrected = [&](float rot) { return dim * logf((float)cfg_.original_seq_len / (rot * 2.f * (float)M_PI)) / (2.f * logf(base)); };
      int low = std::max((int)floorf(corrected(cfg_.beta_fast)), 0);
      int high = std::min((int)ceilf(corrected(cfg_.beta_slow)), dim - 1);
      for (int i = 0; i < half; ++i) {
        float ramp = ((float)i - low) / std::max((float)(high - low), 1e-3f);
        ramp = std::min(std::max(ramp, 0.f), 1.f);
        float smooth = 1.f - ramp;
        freqs[i] = freqs[i] / cfg_.rope_factor * (1.f - smooth) + freqs[i] * smooth;
      }
    }
    std::vector<float2> tab((size_t)rope_len_ * half);
    for (int64_t p = 0; p < rope_len_; ++p)
      for (int i = 0; i < half; ++i) {
        // torch.polar(1, outer(arange, freqs)) — fp32 product, then cos/sin
        float ang = (float)p * freqs[i];
        tab[(size_t)p * half + i] = make_float2(cosf(ang), sinf(ang));
      }
    DevBuf b(tab.size() * sizeof(float2));
    CUDA_CHECK(cudaMemcpy(b.p, tab.data(), tab.size() * sizeof(float2), cudaMemcpyHostToDevice));
    return b;
  };
  rope_window_ = make(false, cfg_.rope_theta);
  rope_compress_ = make(true, cfg_.compress_rope_theta);
}

void Model::load_layer(int l) {
  auto L = std::make_unique<LayerWeights>();
  L->id = l;
  load_block(*L, "layers." + std::to_string(l), true);
  if ((int)layers_.size() <= l) layers_.resize(l + 1);
  layers_[l] = std::move(L);
}

// Dense weights of one layer. With backbone=false (draft stage) the compressor, indexer and engram are not read (they are not in the checkpoint either).
void Model::load_block(LayerWeights& L, const std::string& p, bool backbone) {
  const int l = L.id;
  const std::string ap = p + ".attn";
  auto& A = L.attn;
  A.wq_a = load_fp8(ap + ".wq_a");
  A.wq_b = load_fp8(ap + ".wq_b");
  A.wkv = load_fp8(ap + ".wkv");
  A.wo_b = load_fp8(ap + ".wo_b");
  A.q_norm = load_raw(ap + ".q_norm.weight");
  A.kv_norm = load_raw(ap + ".kv_norm.weight");
  A.attn_sink = load_as_f32(ap + ".attn_sink");
  // wo_a keeps only the fp8 original (the bf16 dequantized copy, 67 MB/layer = 2.7 GB, is dropped — prefill also uses a bf16 x fp8 tensor-core GEMM, same values)
  A.wo_a_q = load_fp8(ap + ".wo_a");
  if (backbone && cfg_.is_kv_source(l)) {
    A.comp_norm = load_raw(ap + ".compressor.norm.weight");
    if (cfg_.ratio(l) > 1) {
      A.comp_wkv = load_as_f32(ap + ".compressor.wkv.weight");
      A.comp_wgate = load_as_f32(ap + ".compressor.wgate.weight");
    } else {
      A.comp_wkv = load_raw(ap + ".compressor.wkv.weight");
    }
  }
  if (backbone && cfg_.is_index_source(l)) {
    A.idx_wq_b = load_fp8(ap + ".indexer.wq_b");
    A.idx_weights_proj = load_raw(ap + ".indexer.weights_proj.weight");
    if (cfg_.is_kv_source(l)) {
      A.idx_wk = load_raw(ap + ".indexer.wk.weight");
      A.idx_k_norm = load_raw(ap + ".indexer.k_norm.weight");
    }
  }
  auto& F = L.ffn;
  F.gate_w = load_as_f32(p + ".ffn.gate.weight");
  F.gate_bias = load_as_f32(p + ".ffn.gate.bias");
  if (ckpt_.has(p + ".ffn.gate.bias_vl")) F.gate_bias_vl = load_as_f32(p + ".ffn.gate.bias_vl");
  F.sh_w1 = load_fp8(p + ".ffn.shared_experts.w1");
  F.sh_w2 = load_fp8(p + ".ffn.shared_experts.w2");
  F.sh_w3 = load_fp8(p + ".ffn.shared_experts.w3");
  L.attn_norm = load_raw(p + ".attn_norm.weight");
  L.ffn_norm = load_raw(p + ".ffn_norm.weight");
  L.hc_attn_fn = load_as_f32(p + ".hc_attn_fn");
  L.hc_attn_scale = load_as_f32(p + ".hc_attn_scale");
  L.hc_attn_base = load_as_f32(p + ".hc_attn_base");
  L.hc_ffn_fn = load_as_f32(p + ".hc_ffn_fn");
  L.hc_ffn_scale = load_as_f32(p + ".hc_ffn_scale");
  L.hc_ffn_base = load_as_f32(p + ".hc_ffn_base");
  if (!backbone) return;
  for (size_t i = 0; i < cfg_.engram_layer_ids.size(); ++i) {
    if (cfg_.engram_layer_ids[i] != l) continue;
    auto E = std::make_unique<EngramWeights>();
    E->hash_index = (int)i;
    E->wkv = load_fp8(p + ".engram.wkv");
    // qk = q_weight * k_weight (fp32)
    const TensorInfo& q = ckpt_.get(p + ".engram.q_weight");
    const TensorInfo& kk = ckpt_.get(p + ".engram.k_weight");
    std::vector<float> qk(q.numel());
    const uint16_t* qp = reinterpret_cast<const uint16_t*>(q.data);
    const uint16_t* kp = reinterpret_cast<const uint16_t*>(kk.data);
    for (size_t j = 0; j < qk.size(); ++j) {
      uint32_t a = (uint32_t)qp[j] << 16, b = (uint32_t)kp[j] << 16;
      float fa, fb; memcpy(&fa, &a, 4); memcpy(&fb, &b, 4);
      qk[j] = fa * fb;
    }
    E->qk.alloc(qk.size() * 4);
    CUDA_CHECK(cudaMemcpy(E->qk.p, qk.data(), qk.size() * 4, cudaMemcpyHostToDevice));
    L.engram = std::move(E);
  }
}

// DSpark draft stage (mtp.{s}): reference DSparkBlock. Stage 0 = main_proj/main_norm, last stage = norm/markov_head/confidence_head.
//   The draft entries of compress_ratios must be 0 (window-only attention — reference `assert self.compress_ratio == 0`).
void Model::load_mtp() {
  HIVE_CHECK(n_loaded_layers() == cfg_.n_layers, "load_mtp needs the full backbone");
  mtp_.clear();
  for (int s = 0;; ++s) {
    const std::string p = "mtp." + std::to_string(s);
    if (!ckpt_.has(p + ".attn_norm.weight")) break;
    auto W = std::make_unique<MtpWeights>();
    W->L.id = cfg_.n_layers + s;
    W->L.n_routed = cfg_.dspark_experts;
    W->L.n_act = cfg_.dspark_act;
    if ((int)cfg_.compress_ratios.size() > W->L.id) HIVE_CHECK(cfg_.compress_ratios[W->L.id] == 0, "dspark stage must be window-only");
    load_block(W->L, p, false);
    if (s == 0) {
      W->main_proj = load_fp8(p + ".main_proj");
      HIVE_CHECK(W->main_proj.N == cfg_.dim && W->main_proj.K == cfg_.dim * (int)cfg_.dspark_targets.size(), "main_proj shape");
      W->main_norm = load_raw(p + ".main_norm.weight");
    }
    if (ckpt_.has(p + ".markov_head.embed.weight")) {
      W->norm = load_raw(p + ".norm.weight");
      W->markov_embed = load_raw(p + ".markov_head.embed.weight");
      W->markov_head = load_raw(p + ".markov_head.head.weight");
      W->conf_proj = load_as_f32(p + ".confidence_head.proj.weight");
      HIVE_CHECK(W->markov_embed.n == (size_t)cfg_.vocab * cfg_.dspark_markov_rank * 2 && W->conf_proj.n == (size_t)(cfg_.dim + cfg_.dspark_markov_rank) * 4,
                 "markov/confidence shape");
    }
    mtp_.push_back(std::move(W));
  }
  if (!mtp_.empty()) HIVE_CHECK(mtp_.back()->markov_embed.p != nullptr && mtp_.front()->main_proj.w.p != nullptr, "mtp stages incomplete");
  size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
  fprintf(stderr, "[model] dspark stages %d · block %d · experts %d/top%d · VRAM free %.0f MiB\n", (int)mtp_.size(), cfg_.dspark_block,
          cfg_.dspark_experts, cfg_.dspark_act, fr / 1048576.0);
}

}  // namespace hive
