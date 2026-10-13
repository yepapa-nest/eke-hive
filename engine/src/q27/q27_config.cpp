// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Q27Config::load — config.json parsing only (no device code), shared by hived_q27 and hived_q27_cpu.
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "hive/q27/q27_model.h"
#include "nlohmann/json.hpp"

namespace hive::q27 {

using json = nlohmann::json;

namespace {
void need(bool c, const std::string& m) { if (!c) throw std::runtime_error("q27 config: " + m); }
}  // namespace

Q27Config Q27Config::load(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  need((bool)f, "no config.json in " + dir);
  json all = json::parse(f);
  const std::string mt = all.value("model_type", std::string());
  need(mt == "qwen3_5" || mt == "qwen3_5_text", "model_type is not qwen3_5 (got '" + mt + "')");
  const json& t = all.contains("text_config") ? all["text_config"] : all;
  Q27Config c;
  auto gi = [&](const json& j, const char* k, int& v) { if (j.contains(k) && j[k].is_number()) v = j[k].get<int>(); };
  auto gf = [&](const json& j, const char* k, float& v) { if (j.contains(k) && j[k].is_number()) v = j[k].get<float>(); };
  gi(t, "hidden_size", c.hidden); gi(t, "num_hidden_layers", c.n_layers); gi(t, "vocab_size", c.vocab); gi(t, "intermediate_size", c.inter);
  gf(t, "rms_norm_eps", c.rms_eps);
  gi(t, "linear_num_key_heads", c.lin_k_heads); gi(t, "linear_num_value_heads", c.lin_v_heads); gi(t, "linear_key_head_dim", c.lin_k_dim);
  gi(t, "linear_value_head_dim", c.lin_v_dim); gi(t, "linear_conv_kernel_dim", c.conv_k);
  gi(t, "num_attention_heads", c.n_heads); gi(t, "num_key_value_heads", c.n_kv_heads); gi(t, "head_dim", c.head_dim);
  gi(t, "mtp_num_hidden_layers", c.n_mtp);
  if (t.contains("eos_token_id") && t["eos_token_id"].is_number()) c.eos_id = t["eos_token_id"].get<int>();
  if (t.contains("max_position_embeddings")) c.max_position = t["max_position_embeddings"].get<int64_t>();
  const json& rp = t.contains("rope_parameters") ? t["rope_parameters"] : t.contains("rope_scaling") ? t["rope_scaling"] : json::object();
  if (rp.contains("rope_theta")) c.rope_theta = rp["rope_theta"].get<float>(); else gf(t, "rope_theta", c.rope_theta);
  float prf = 0.25f;
  if (rp.contains("partial_rotary_factor")) prf = rp["partial_rotary_factor"].get<float>(); else gf(t, "partial_rotary_factor", prf);
  c.rotary_dim = (int)std::lround(c.head_dim * prf);
  if (rp.contains("mrope_section")) {
    const auto s = rp["mrope_section"].get<std::vector<int>>();
    need(s.size() == 3, "mrope_section must have 3 entries");
    c.mrope_section = {s[0], s[1], s[2]};
  }
  if (rp.contains("mrope_interleaved")) c.mrope_interleaved = rp["mrope_interleaved"].get<bool>();
  need(t.contains("layer_types") && t["layer_types"].is_array(), "text_config.layer_types missing");
  const auto lt = t["layer_types"].get<std::vector<std::string>>();
  need((int)lt.size() == c.n_layers, "layer_types length");
  c.is_attn.assign(c.n_layers, false);
  for (int l = 0; l < c.n_layers; ++l) {
    need(lt[l] == "linear_attention" || lt[l] == "full_attention", "unknown layer type " + lt[l]);
    c.is_attn[l] = lt[l] == "full_attention";
  }
  if (all.contains("vision_config")) {
    const json& v = all["vision_config"];
    gi(v, "depth", c.vision_layers); gi(v, "hidden_size", c.vision_dim); gi(v, "num_heads", c.vision_heads); gi(v, "intermediate_size", c.vision_inter);
    gi(v, "patch_size", c.vision_patch); gi(v, "spatial_merge_size", c.vision_merge); gi(v, "temporal_patch_size", c.vision_temporal);
    gi(v, "out_hidden_size", c.vision_out); gi(v, "num_position_embeddings", c.vision_pos);
    if (v.contains("deepstack_visual_indexes")) need(v["deepstack_visual_indexes"].empty(), "deepstack vision layers are not supported");
  }
  gi(all, "image_token_id", c.image_token_id); gi(all, "video_token_id", c.video_token_id); gi(all, "vision_start_token_id", c.vision_start_id);
  gi(all, "vision_end_token_id", c.vision_end_id);
  // the kernels assume these shapes; a different checkpoint must fail here, not inside a kernel
  need(c.hidden % 256 == 0 && c.inter % 256 == 0, "hidden / intermediate must be multiples of 256");
  need(c.lin_k_dim == 128 && c.lin_v_dim == 128 && c.conv_k == 4 && c.lin_v_heads % c.lin_k_heads == 0, "Gated DeltaNet shape (128-wide heads, kernel 4)");
  need(c.head_dim == 256 && c.rotary_dim == 64 && c.n_heads % c.n_kv_heads == 0, "attention shape (head 256, rotary 64)");
  need(c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2] == c.rotary_dim / 2, "mrope sections must cover the rotary pairs");
  need(c.n_mtp <= 1, "at most one MTP layer");
  return c;
}

}  // namespace hive::q27
