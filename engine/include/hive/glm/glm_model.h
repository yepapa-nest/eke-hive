// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash (transformers model_type glm5_next) — configuration and non-expert weights.
//   45 decoder layers: 34 KDA linear-attention layers + 11 DSA (MLA without positions + pooled lightning indexer) layers,
//   hyper-connections (hc_mult 4, Sinkhorn), layers 0-2 dense MLP (NVFP4), layers 3-44 MoE (288 routed experts top-8, sigmoid
//   router with correction bias, one BF16 shared expert). Layer 45 is the NextN (MTP) layer. Routed experts are handled by
//   GlmExperts (glm_experts.h); everything else here lives in VRAM, except the embedding table (host, rows gathered per token).
#pragma once
#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "hive/common.h"
#include "hive/glm/fp8b.h"
#include "hive/glm/nvfp4.h"
#include "hive/devbuf.h"
#include "hive/safetensors.h"

namespace hive::glm {

struct GlmConfig {
  int hidden = 4096, n_layers = 45, vocab = 154880;
  int n_heads = 64;                          // DSA attention heads
  int q_lora = 1536, kv_lora = 512, qk_head = 256, v_head = 256;
  int kda_heads = 64, kda_dim = 128, conv_k = 4;
  float kda_lower_bound = -5.f;
  int index_heads = 32, index_dim = 128, index_topk = 2048, index_kpool = 4;
  int dense_inter = 12288, moe_inter = 2048, n_routed = 288, n_act = 8, n_shared = 1;
  float routed_scale = 2.5f, swiglu_limit = 10.f, rms_eps = 1e-5f, hc_eps = 1e-6f;
  int hc = 4, sinkhorn_iters = 20;
  int n_mtp = 1;
  int64_t max_position = 1048576;
  std::vector<int> eos_ids{154820, 154827, 154829};
  std::vector<bool> is_kda;     // per layer: linear attention
  std::vector<bool> is_moe;     // per layer: sparse MLP
  std::vector<int> moe_index;   // layer → index among MoE layers (-1 for dense)
  int n_moe = 0;
  // vision (Glm5NextVisionModel)
  int vision_layers = 24, vision_dim = 1024, vision_heads = 16, vision_inter = 4096, vision_patch = 14, vision_merge = 2,
      vision_temporal = 2, vision_out = 4096, vision_proj_inter = 10240;
  int image_token_id = 154854, video_token_id = 154855, image_start_id = 154830, image_end_id = 154831, video_start_id = 154832,
      video_end_id = 154833;
  static GlmConfig load(const std::string& ckpt_dir);
};

// A BF16 matrix in VRAM ([out, in], row-major as stored).
struct Mat {
  DevBuf buf;                 // bf16 weight (empty when fp8)
  DevBuf f8w, f8s;            // fp8 e4m3 [out, in] + fp32 128×128 block scales (Z.ai FP8 tensors, HIVE_GLM_FP8_DIR)
  bool fp8 = false;
  int out = 0, in = 0;
  const bf16* p() const { return buf.as<bf16>(); }
  Fp8BMat f8() const { return Fp8BMat{f8w.as<uint8_t>(), f8s.as<float>(), out, in}; }
};

struct HcW { DevBuf fn; DevBuf base, scale; };  // fn fp32 [24, 4·hidden] · base fp32 [24] · scale fp32 [3]

struct KdaW {
  Mat q, k, v, f_a, f_b, g_a, g_b, b, o;   // projections
  DevBuf conv_w;                           // fp32 [3·8192, 4] (q, k, v stacked)
  DevBuf A_log, dt_bias;                   // fp32 [64], [8192]
  DevBuf o_norm;                           // bf16 [128]
};
struct DsaW {
  Mat q_a, q_b, kv_a, kv_b, o;             // MLA
  DevBuf q_a_norm, kv_a_norm;              // bf16 [1536], [512]
  Mat idx_wq_b, idx_wk, idx_weights, idx_gate;  // indexer: [4096,1536] [128,4096] [32,4096] [128,4096]
  DevBuf idx_k_norm_w, idx_k_norm_b, idx_ape;    // bf16 [128] [128] [4,128]
};
struct DenseMlpW { DevBuf w[3], s[3]; float g[3] = {1, 1, 1}; };  // NVFP4 gate, up, down (device)
struct MoeW {
  Mat gate;                                // router [288, 4096]
  DevBuf bias;                             // fp32 [288]
  Mat sh_gate, sh_up, sh_down;             // shared expert BF16
};
struct LayerW {
  bool kda = false, moe = false;
  DevBuf in_norm, post_norm;               // bf16 [4096]
  HcW hc_attn, hc_ffn;
  KdaW kda_w; DsaW dsa_w;
  DenseMlpW dense; MoeW moe_w;
};
struct MtpW {
  DevBuf enorm, hnorm, head_norm, in_norm, post_norm;
  Mat eh_proj;                             // [4096, 8192]: [enorm(embedding) | hnorm(hidden)] → hidden
  DsaW dsa_w; MoeW moe_w;
  // routed experts of the NextN layer: Z.ai FP8 (128×128 block scales), all resident in VRAM (gate, up [I, H] · down [H, I])
  DevBuf expert_arena;
  std::vector<std::array<Fp8BMat, 3>> experts;
};

class GlmModel {
 public:
  GlmModel(const std::string& ckpt_dir, bool load_mtp = true);
  const GlmConfig& cfg() const { return cfg_; }
  Checkpoint& ckpt() { return ckpt_; }
  const LayerW& layer(int l) const { return layers_[l]; }
  const MtpW* mtp() const { return mtp_.get(); }
  const bf16* embed_host() const { return embed_host_.data(); }  // [vocab, hidden] bf16 in host RAM
  const Mat& lm_head() const { return lm_head_; }
  const DevBuf& final_norm() const { return final_norm_; }
  size_t vram_bytes() const { return vram_bytes_; }
  int n_fp8() const { return n_fp8_; }
  // Dense MLP of layers 0-2 as NVFP4 views.
  Nvfp4Mat dense_mat(int l, int which) const;

 private:
  Mat mat(const std::string& name);
  Mat mat_fp8ok(const std::string& name);  // FP8 from the Z.ai checkpoint when present there, else BF16
  DevBuf vec_bf16(const std::string& name);
  DevBuf vec_f32(const std::string& name);  // stored dtype F32 or BF16 → fp32 on device
  void load_kda(const std::string& p, KdaW& w);
  void load_dsa(const std::string& p, DsaW& w);
  void load_moe(const std::string& p, MoeW& w);
  void load_hc(const std::string& p, const char* which, HcW& w);
  GlmConfig cfg_;
  Checkpoint ckpt_;
  std::unique_ptr<Checkpoint> fp8_;
  int n_fp8_ = 0;
  std::vector<LayerW> layers_;
  std::unique_ptr<MtpW> mtp_;
  std::vector<bf16> embed_host_;
  Mat lm_head_;
  DevBuf final_norm_;
  size_t vram_bytes_ = 0;
};

}  // namespace hive::glm
