// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Model config + loader for the VRAM-resident (dense) weights. Experts and engram tables are kept in pinned RAM by expert_store.h.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hive/common.h"
#include "hive/devbuf.h"
#include "hive/devmem.h"
#include "hive/safetensors.h"

namespace hive {

struct Config {
  int dim = 5120, n_layers = 40, n_heads = 64, head_dim = 512, rd = 64, q_lora_rank = 1280, o_groups = 8, o_lora_rank = 1024;
  int window = 128, vocab = 129280, moe_inter = 2304, n_routed = 384, n_act = 6, n_shared = 1;
  int index_n_heads = 32, index_head_dim = 128, index_topk = 512;
  int cand_source_layer = 20, cand_topk_blocks = 2048, cand_block = 8;
  int hc = 4, sinkhorn_iters = 20;
  float hc_eps = 1e-6f, norm_eps = 1e-20f, route_scale = 1.5f, swiglu_limit = 10.f;
  float rope_theta = 10000.f, compress_rope_theta = 160000.f;
  float rope_factor = 16.f, beta_fast = 32.f, beta_slow = 1.f;
  int64_t original_seq_len = 65536, max_position = 1048576;
  std::vector<int> compress_ratios, kv_source_layers, index_source_layers, engram_layer_ids;
  std::vector<int64_t> engram_num_embeddings;
  int engram_max_ngram = 4, engram_n_heads = 8, engram_head_dim = 256, engram_vocab_size = 16000000, engram_compressed_vocab = 99092,
      engram_pad_id = 2;
  int image_token_id = 129264;
  // vision
  int vision_layers = 32, vision_dim = 1024, vision_heads = 16, vision_inter = 2816, vision_patch = 14, vision_downsample = 3,
      vision_max_tokens = 1024, vision_min_pixels = 295936;
  float vision_rope_theta = 10000.f;
  // dspark (draft block, MTP) — reference DSparkBlock: drafts a block of 5 tokens at once, 128 experts top-3, target-layer attention-input mean (hc) fed to main_proj
  int dspark_block = 5, dspark_experts = 128, dspark_act = 3, dspark_noise_token = 128799, dspark_markov_rank = 256;
  std::vector<int> dspark_targets;

  bool is_kv_source(int l) const;
  bool is_index_source(int l) const;
  int ratio(int l) const { return compress_ratios[l]; }
  int n_hash_cols() const { return (engram_max_ngram - 1) * engram_n_heads; }
  static Config load(const std::string& path);
};

// fp8 dense linear: values [N,K] e4m3 + expanded scales [N, K/32] e8m0
struct Fp8Linear {
  DevBuf w, s;
  int N = 0, K = 0;
};

struct AttnWeights {
  Fp8Linear wq_a, wq_b, wkv, wo_b;
  DevBuf q_norm, kv_norm;  // bf16
  DevBuf wo_a;             // bf16 [o_groups·o_lora_rank, n_heads·head_dim/o_groups] (dequantized — for prefill cuBLAS)
  Fp8Linear wo_a_q;        // original fp8 of the same weights (+ per-row block scales) — decode GEMV reads 33MB instead of 67MB (same values, dequantization is exact)
  DevBuf attn_sink;        // f32 [H]
  // compressor(kv source)
  DevBuf comp_norm;        // bf16 [head_dim]
  DevBuf comp_wkv;         // ratio>1: f32 [head_dim, dim] / ratio==1: bf16 [head_dim, dim]
  DevBuf comp_wgate;       // ratio>1: f32 [head_dim, dim]
  // indexer(index source)
  Fp8Linear idx_wq_b;      // [Hi·Di, q_lora_rank]
  DevBuf idx_weights_proj; // bf16 [Hi, dim]
  DevBuf idx_wk;           // bf16 [Di, head_dim] (owns_k)
  DevBuf idx_k_norm;       // bf16 [Di]
};

struct MoeWeights {
  DevBuf gate_w;      // f32 [E, dim]
  DevBuf gate_bias;   // f32 [E]
  DevBuf gate_bias_vl;
  Fp8Linear sh_w1, sh_w2, sh_w3;
};

struct EngramWeights {
  Fp8Linear wkv;   // [dim·(hc+1), cols·hd]
  DevBuf qk;       // f32 [hc, dim] = q_weight*k_weight
  int hash_index = 0;
};

struct LayerWeights {
  int id = 0;
  int n_routed = 0, n_act = 0;  // 0 = backbone value from Config (draft layers: 128 / 3)
  AttnWeights attn;
  MoeWeights ffn;
  DevBuf attn_norm, ffn_norm;                  // bf16 [dim]
  DevBuf hc_attn_fn, hc_attn_scale, hc_attn_base;  // f32
  DevBuf hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
  std::unique_ptr<EngramWeights> engram;
};

// DSpark draft stage (checkpoint mtp.{s}.*): same structure as a backbone layer (no compressor, indexer or engram; attention window comes from the backbone main_x) +
//   stage 0's main_proj/main_norm (target-layer hidden → window-KV source of the draft input x) · last stage's norm, markov (bigram bias) and confidence
struct MtpWeights {
  LayerWeights L;
  Fp8Linear main_proj;   // [dim, dim·n_targets]
  DevBuf main_norm;      // bf16 [dim]
  DevBuf norm;           // bf16 [dim] (last stage)
  DevBuf markov_embed;   // bf16 [V, rank]
  DevBuf markov_head;    // bf16 [V, rank]
  DevBuf conf_proj;      // f32 [dim + rank]
};

struct VisionWeights;  // vision.h

class Model {
 public:
  Model(const std::string& dir, int max_layer = -1, int64_t rope_len = 1048576);
  ~Model();
  const Config& cfg() const { return cfg_; }
  Checkpoint& ckpt() { return ckpt_; }
  const LayerWeights& layer(int l) const { return *layers_[l]; }
  int n_loaded_layers() const { return (int)layers_.size(); }
  const bf16* embed_host() const { return embed_host_; }  // [vocab, dim] bf16 (checkpoint mmap)
  const bf16* head() const { return head_.as<bf16>(); }
  const bf16* norm() const { return norm_.as<bf16>(); }
  bool has_head() const { return head_.p != nullptr; }
  // rope tables: [max_pos, rd/2] float2 — for the window (theta 10000, no YaRN) and for compression (160000, YaRN)
  const float2* rope_window() const { return rope_window_.as<float2>(); }
  const float2* rope_compress() const { return rope_compress_.as<float2>(); }
  int64_t rope_len() const { return rope_len_; }
  void load_layer(int l);
  // DSpark draft stages: loads as many as the checkpoint has mtp.{s}.attn_norm.weight (0 if none). Only after all 40 backbone layers are loaded.
  void load_mtp();
  int n_mtp() const { return (int)mtp_.size(); }
  const MtpWeights& mtp(int s) const { return *mtp_[s]; }

 private:
  Config cfg_;
  Checkpoint ckpt_;
  std::vector<std::unique_ptr<LayerWeights>> layers_;
  std::vector<std::unique_ptr<MtpWeights>> mtp_;
  void load_block(LayerWeights& L, const std::string& p, bool backbone);
  const bf16* embed_host_ = nullptr;
  void* embed_owned_ = nullptr;
  DevBuf head_, norm_, rope_window_, rope_compress_;
  int64_t rope_len_ = 0, rope_req_ = 1048576;
  Fp8Linear load_fp8(const std::string& name);
  DevBuf load_raw(const std::string& name);
  DevBuf load_as_f32(const std::string& name);
  void build_rope();
};

}  // namespace hive
