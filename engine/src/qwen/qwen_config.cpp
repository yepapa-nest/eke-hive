// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// QwenConfig::load — config.json parsing only (no device code), shared by the GPU family (hived_qwen) and the CPU engine (hived_qwen_cpu).
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "hive/qwen/qwen_model.h"
#include "nlohmann/json.hpp"

namespace hive::qwen {

using json = nlohmann::json;

QwenConfig QwenConfig::load(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) throw std::runtime_error("no config.json in " + dir);
  json all = json::parse(f);
  if (all.value("model_type", std::string()) != "qwen4_exp") throw std::runtime_error("config: model_type is not qwen4_exp");
  const json& t = all.contains("text_config") ? all["text_config"] : all;
  QwenConfig c;
  auto gi = [&](const json& j, const char* k, int& v) { if (j.contains(k) && j[k].is_number()) v = j[k].get<int>(); };
  auto gf = [&](const json& j, const char* k, float& v) { if (j.contains(k) && j[k].is_number()) v = j[k].get<float>(); };
  gi(t, "hidden_size", c.hidden); gi(t, "num_hidden_layers", c.n_layers); gi(t, "vocab_size", c.vocab); gi(t, "hc_count", c.hc);
  gi(t, "hc_lowrank", c.hc_rank); gf(t, "rms_norm_eps", c.rms_eps);
  gi(t, "linear_num_key_heads", c.lin_k_heads); gi(t, "linear_num_value_heads", c.lin_v_heads); gi(t, "linear_key_head_dim", c.lin_k_dim);
  gi(t, "linear_value_head_dim", c.lin_v_dim); gi(t, "linear_conv_kernel_dim", c.conv_k);
  gi(t, "num_attention_heads", c.n_heads); gi(t, "num_key_value_heads", c.n_kv_heads); gi(t, "head_dim", c.head_dim);
  gi(t, "indexer_n_heads", c.idx_heads); gi(t, "indexer_head_dim", c.idx_dim); gi(t, "indexer_budget", c.idx_budget);
  gi(t, "indexer_compress_ratio", c.idx_ratio);
  gi(t, "num_experts", c.n_experts); gi(t, "num_experts_per_tok", c.n_act); gi(t, "moe_intermediate_size", c.moe_inter);
  gi(t, "shared_expert_intermediate_size", c.shared_inter);
  if (t.contains("norm_topk_prob")) c.norm_topk = t["norm_topk_prob"].get<bool>();
  gi(t, "ple_embed_dim", c.ple_dim); gi(t, "ple_conv_kernel_size", c.ple_conv); gi(t, "ngram_size", c.ngram); gi(t, "heads_per_ngram", c.heads_per_ngram);
  gi(t, "split_ngram_parts", c.ngram_parts);
  if (t.contains("ngram_vocab_size_base")) c.ngram_vocab_base = t["ngram_vocab_size_base"].get<int64_t>();
  if (t.contains("ple_layer_ids")) c.ple_layer_ids = t["ple_layer_ids"].get<std::vector<int>>();
  if (t.contains("eos_token_id") && t["eos_token_id"].is_number()) c.eos_id = t["eos_token_id"].get<int>();
  gi(t, "mtp_num_hidden_layers", c.n_mtp);
  if (t.contains("max_position_embeddings")) c.max_position = t["max_position_embeddings"].get<int64_t>();
  const json& rp = t.contains("rope_parameters") ? t["rope_parameters"] : t.contains("rope_scaling") ? t["rope_scaling"] : json::object();
  if (rp.contains("rope_theta")) c.rope_theta = rp["rope_theta"].get<float>();
  else gf(t, "rope_theta", c.rope_theta);
  float prf = 0.25f;
  if (rp.contains("partial_rotary_factor")) prf = rp["partial_rotary_factor"].get<float>();
  else gf(t, "partial_rotary_factor", prf);
  c.rotary_dim = (int)std::lround(c.head_dim * prf);
  if (rp.contains("mrope_section")) {
    const auto s = rp["mrope_section"].get<std::vector<int>>();
    HIVE_CHECK(s.size() == 3, "qwen: mrope_section must have 3 entries");
    c.mrope_section = {s[0], s[1], s[2]};
  }
  if (rp.contains("mrope_interleaved")) c.mrope_interleaved = rp["mrope_interleaved"].get<bool>();
  HIVE_CHECK(t.contains("layer_types") && t["layer_types"].is_array(), "qwen: text_config.layer_types missing");
  const auto lt = t["layer_types"].get<std::vector<std::string>>();
  HIVE_CHECK((int)lt.size() == c.n_layers, "qwen: layer_types length");
  c.is_attn.assign(c.n_layers, false);
  for (int l = 0; l < c.n_layers; ++l) {
    HIVE_CHECK(lt[l] == "linear_attention" || lt[l] == "full_attention", "qwen: unknown layer type " + lt[l]);
    c.is_attn[l] = lt[l] == "full_attention";
  }
  if (all.contains("vision_config")) {
    const json& v = all["vision_config"];
    gi(v, "depth", c.vision_layers); gi(v, "hidden_size", c.vision_dim); gi(v, "num_heads", c.vision_heads); gi(v, "intermediate_size", c.vision_inter);
    gi(v, "patch_size", c.vision_patch); gi(v, "spatial_merge_size", c.vision_merge); gi(v, "temporal_patch_size", c.vision_temporal);
    gi(v, "out_hidden_size", c.vision_out); gi(v, "num_position_embeddings", c.vision_pos);
  }
  gi(all, "image_token_id", c.image_token_id); gi(all, "video_token_id", c.video_token_id); gi(all, "vision_start_token_id", c.vision_start_id);
  gi(all, "vision_end_token_id", c.vision_end_id);
  // the kernels are written for these shapes (gdn.h, qsa.h, ple.h constants) — a different checkpoint must fail here, not inside a kernel
  HIVE_CHECK(c.hidden == 2560 && c.hc == 4 && c.hc_rank == 320, "qwen: hidden 2560 · hc 4 · hc_lowrank 320 expected");
  HIVE_CHECK(c.lin_k_heads == 16 && c.lin_v_heads == 48 && c.lin_k_dim == 128 && c.lin_v_dim == 128 && c.conv_k == 4, "qwen: GatedDeltaNet shape");
  HIVE_CHECK(c.n_heads == 24 && c.n_kv_heads == 2 && c.head_dim == 256 && c.rotary_dim == 64, "qwen: attention shape");
  HIVE_CHECK(c.idx_heads == 4 && c.idx_dim == 128 && c.idx_budget == 2048 && c.idx_ratio == 4, "qwen: QSA indexer shape");
  HIVE_CHECK(c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2] == c.rotary_dim / 2, "qwen: mrope sections must cover the rotary pairs");
  HIVE_CHECK(c.n_act <= 16 && c.n_experts <= 1024 && c.moe_inter % 32 == 0, "qwen: MoE shape");
  HIVE_CHECK(c.ple_layer_ids.size() <= 1, "qwen: at most one PLE layer supported");
  HIVE_CHECK(c.ple_layer_ids.empty() || (c.ple_dim == 2560 && c.ple_conv == 4 && c.ngram == 3 && c.heads_per_ngram == 8), "qwen: PLE shape");
  return c;
}

}  // namespace hive::qwen
