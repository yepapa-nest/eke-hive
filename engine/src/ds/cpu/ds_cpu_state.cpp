// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash CPU engine — config.json parsing, engine options from the environment and CpuSeq serialization for hived's session
//   images. Image order: header, tokens, engram history, every layer's window ring, per kv source (compressed KV rows and indexer key rows
//   below pos / ratio, compressor state), every DSpark ring, the DSpark hidden.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "ds_cpu_internal.h"
#include "nlohmann/json.hpp"

namespace hive::ds::cpu {

using json = nlohmann::json;

bool DsConfig::is_kv_source(int l) const { return l < n_layers && std::find(kv_source_layers.begin(), kv_source_layers.end(), l) != kv_source_layers.end(); }
bool DsConfig::is_index_source(int l) const {
  return l < n_layers && std::find(index_source_layers.begin(), index_source_layers.end(), l) != index_source_layers.end();
}

DsConfig DsConfig::load(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) throw std::runtime_error("no config.json in " + dir);
  const json all = json::parse(f);
  const json& t = all.contains("text_config") ? all["text_config"] : all;
  DsConfig c;
  auto gi = [&](const json& j, const char* k, int& v) { if (j.contains(k) && j[k].is_number()) v = j[k].get<int>(); };
  auto gf = [&](const json& j, const char* k, float& v) { if (j.contains(k) && j[k].is_number()) v = j[k].get<float>(); };
  auto gv = [&](const json& j, const char* k, std::vector<int>& v) { if (j.contains(k) && j[k].is_array()) v = j[k].get<std::vector<int>>(); };
  gi(t, "hidden_size", c.dim); gi(t, "num_hidden_layers", c.n_layers); gi(t, "num_attention_heads", c.n_heads); gi(t, "head_dim", c.head_dim);
  gi(t, "qk_rope_head_dim", c.rd); gi(t, "q_lora_rank", c.q_lora_rank); gi(t, "o_groups", c.o_groups); gi(t, "o_lora_rank", c.o_lora_rank);
  gi(t, "sliding_window", c.window); gi(t, "vocab_size", c.vocab); gi(t, "moe_intermediate_size", c.moe_inter);
  gi(t, "n_routed_experts", c.n_routed); gi(t, "num_experts_per_tok", c.n_act);
  gi(t, "index_n_heads", c.index_n_heads); gi(t, "index_head_dim", c.index_head_dim); gi(t, "index_topk", c.index_topk);
  gi(t, "candidate_source_layer_id", c.cand_source_layer); gi(t, "candidate_topk_blocks", c.cand_topk_blocks); gi(t, "candidate_block_size", c.cand_block);
  gi(t, "hc_mult", c.hc); gi(t, "hc_sinkhorn_iters", c.sinkhorn_iters); gf(t, "hc_eps", c.hc_eps); gf(t, "rms_norm_eps", c.norm_eps);
  gf(t, "routed_scaling_factor", c.route_scale); gf(t, "swiglu_limit", c.swiglu_limit);
  gf(t, "rope_theta", c.rope_theta); gf(t, "compress_rope_theta", c.compress_rope_theta);
  if (t.contains("rope_scaling") && t["rope_scaling"].is_object()) {
    const json& r = t["rope_scaling"];
    gf(r, "factor", c.rope_factor); gf(r, "beta_fast", c.beta_fast); gf(r, "beta_slow", c.beta_slow);
    if (r.contains("original_max_position_embeddings")) c.original_seq_len = r["original_max_position_embeddings"].get<int64_t>();
  }
  gv(t, "compress_ratios", c.compress_ratios); gv(t, "kv_source_layer_ids", c.kv_source_layers); gv(t, "index_source_layer_ids", c.index_source_layers);
  gv(t, "engram_layer_ids", c.engram_layer_ids);
  gi(t, "engram_max_ngram_size", c.engram_max_ngram); gi(t, "engram_n_heads", c.engram_n_heads); gi(t, "engram_head_dim", c.engram_head_dim);
  gi(all, "image_token_id", c.image_token_id);
  gi(t, "dspark_block_size", c.dspark_block); gi(t, "dspark_n_routed_experts", c.dspark_experts); gi(t, "dspark_num_experts_per_tok", c.dspark_act);
  gi(t, "dspark_noise_token_id", c.dspark_noise_token); gi(t, "dspark_markov_rank", c.dspark_markov_rank); gv(t, "dspark_target_layer_ids", c.dspark_targets);
  if (all.contains("vision_config") && all["vision_config"].is_object()) {
    const json& v = all["vision_config"];
    gi(v, "num_hidden_layers", c.vision_layers); gi(v, "hidden_size", c.vision_dim); gi(v, "num_attention_heads", c.vision_heads);
    gi(v, "intermediate_size", c.vision_inter); gi(v, "patch_size", c.vision_patch); gi(v, "downsample_ratio", c.vision_downsample);
    gi(v, "max_image_tokens", c.vision_max_tokens); gf(v, "rope_theta", c.vision_rope_theta);
  }
  if ((int)c.compress_ratios.size() < c.n_layers) c.compress_ratios.resize(c.n_layers, 0);
  return c;
}

EngineOptions EngineOptions::from_env() {
  EngineOptions o;
  auto is = [](const char* n, const char* v) { const char* e = getenv(n); return e && strcmp(e, v) == 0; };
  o.experts_mmap = is("HIVE_DS_CPU_EXPERTS", "mmap");
  o.engram_mmap = is("HIVE_DS_CPU_ENGRAM", "mmap");
  if (const char* v = getenv("HIVE_DS_CPU_CHUNK"); v && atoi(v) > 0) o.max_chunk = atoi(v);
  if (const char* v = getenv("HIVE_DS_CPU_TAIL"); v && *v) o.decoder_tail = std::max(0, atoi(v));
  return o;
}

namespace {
template <class T>
std::vector<uint8_t> bytes_of(const T* p, size_t n) {
  std::vector<uint8_t> b(n * sizeof(T));
  if (n) memcpy(b.data(), p, b.size());
  return b;
}
template <class T>
void restore(std::vector<T>& v, const std::vector<uint8_t>& b, size_t min_n = 0) {
  if (b.size() % sizeof(T)) throw std::runtime_error("ds cpu: bad state image");
  v.resize(std::max(min_n, b.size() / sizeof(T)));
  if (!b.empty()) memcpy(v.data(), b.data(), b.size());
}
}  // namespace

std::vector<std::vector<uint8_t>> CpuSeq::save_bytes() const {
  std::vector<std::vector<uint8_t>> out;
  const int64_t hdr[8] = {pos, cap, mtp_hidden_valid ? 1 : 0, mtp_pos, broken ? 1 : 0, (int64_t)ring.size(), (int64_t)comp.size(), (int64_t)mtp_ring.size()};
  out.push_back(bytes_of(hdr, 8));
  out.push_back(bytes_of(tokens.data(), tokens.size()));
  out.push_back(bytes_of(engram_history.data(), engram_history.size()));
  for (const auto& r : ring) out.push_back(bytes_of(r.data(), r.size()));
  for (const Comp& c : comp) {
    // only the completed group rows exist below pos / ratio; the vectors may be larger (grown ahead) — the width is fixed per vector
    const size_t ratio_rows = c.st_kv.empty() ? (size_t)pos : (size_t)pos / std::max<size_t>(1, c.st_kv.size() / 512);
    out.push_back(bytes_of(c.kv.data(), std::min(c.kv.size(), ratio_rows * 512)));
    out.push_back(bytes_of(c.ik.data(), std::min(c.ik.size(), ratio_rows * 128)));
    out.push_back(bytes_of(c.st_kv.data(), c.st_kv.size()));
    out.push_back(bytes_of(c.st_sc.data(), c.st_sc.size()));
  }
  for (const auto& r : mtp_ring) out.push_back(bytes_of(r.data(), r.size()));
  out.push_back(bytes_of(mtp_hidden.data(), mtp_hidden.size()));
  return out;
}

void CpuSeq::load_bytes(const std::vector<std::vector<uint8_t>>& b) {
  if (b.empty() || b[0].size() != 64) throw std::runtime_error("ds cpu: bad state image");
  int64_t hdr[8];
  memcpy(hdr, b[0].data(), 64);
  const size_t nl = (size_t)hdr[5], nc = (size_t)hdr[6], nm = (size_t)hdr[7];
  if (b.size() != 3 + nl + 4 * nc + nm + 1 || nl != ring.size() || nc != comp.size() || nm != mtp_ring.size())
    throw std::runtime_error("ds cpu: state image does not match the model");
  pos = hdr[0]; cap = hdr[1]; mtp_hidden_valid = hdr[2] != 0; mtp_pos = hdr[3]; broken = hdr[4] != 0;
  size_t k = 1;
  restore(tokens, b[k++]);
  restore(engram_history, b[k++]);
  for (auto& r : ring) restore(r, b[k++], r.size());
  for (Comp& c : comp) {
    // keep the grown capacity: copy the saved rows over the front
    const std::vector<uint8_t>& kv = b[k++];
    const std::vector<uint8_t>& ik = b[k++];
    if (c.kv.size() * 2 < kv.size()) c.kv.resize(kv.size() / 2);
    if (c.ik.size() * 4 < ik.size()) c.ik.resize(ik.size() / 4);
    if (!kv.empty()) memcpy(c.kv.data(), kv.data(), kv.size());
    if (!ik.empty()) memcpy(c.ik.data(), ik.data(), ik.size());
    restore(c.st_kv, b[k++]);
    restore(c.st_sc, b[k++]);
  }
  for (auto& r : mtp_ring) restore(r, b[k++], r.size());
  restore(mtp_hidden, b[k++]);
}

}  // namespace hive::ds::cpu
