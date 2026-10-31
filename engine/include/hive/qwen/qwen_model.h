// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next (config.json model_type "qwen4_exp", NVIDIA ModelOpt NVFP4 checkpoint) — configuration and non-expert weights.
//   48 decoder layers, every fourth one (3, 7, …, 47) full attention, the others Gated DeltaNet linear attention. Every layer:
//     hidden stream of hc_count = 4 copies of the 2560-wide hidden (hyper-connections, "gated residual" form: a low-rank sigmoid mix of
//     the four normed copies feeds the block, the block output is added back to each copy with a per-copy sigmoid weight);
//     MoE MLP: 512 routed experts (NVFP4), top-10 of a softmax router, renormalised; one shared expert (BF16) gated by sigmoid(x·g).
//   Full attention: 24 query heads / 2 KV heads × 256, per-head (1 + w) RMS norm of q and k, interleaved M-RoPE on the first 64 dims
//     (θ = 1e7, sections 11/11/10), an output gate sigmoid(g) per head, and a sparse "QSA" indexer that picks ≤ 2048 earlier tokens
//     (4-token groups scored by a 4-head, 128-dim indexer over mean-pooled keys) — dense attention while ≤ 512 groups are visible.
//   PLE (per-layer n-gram embedding) in front of layer 1's mixing: 16 hashed 2-/3-gram lookups of 160 FP8 values per token from a
//     320 M-row table (47.7 GiB), gated against the hidden stream and passed through a dilated causal conv (hive/qwen/ple.h).
//   NextN (MTP): one full-attention layer with its own FP8 (128×128 block) experts, fed by the token embedding and the main stream.
//   Vision encoder (27 blocks, 1152 wide, patch 16, 2×2 merge) → 2560 (hive/qwen/qwen_vision.h).
// Everything except the routed experts (GlmExperts, hive/glm/glm_experts.h — the same NVFP4 record format and cache) lives in VRAM;
//   the token embedding table stays in host RAM (rows gathered per token).
// Every value here was read from the checkpoint's config.json / tensor shapes (2026-10-09); the loader checks the shapes it relies on.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hive/common.h"
#include "hive/devbuf.h"
#include "hive/glm/fp8b.h"
#include "hive/qwen/qwen_kernels.h"
#include "hive/safetensors.h"

namespace hive::qwen {

struct QwenConfig {
  int hidden = 2560, n_layers = 48, vocab = 248320, hc = 4, hc_rank = 320;
  float rms_eps = 1e-6f;
  // linear attention (Gated DeltaNet)
  int lin_k_heads = 16, lin_v_heads = 48, lin_k_dim = 128, lin_v_dim = 128, conv_k = 4;
  // full attention
  int n_heads = 24, n_kv_heads = 2, head_dim = 256, rotary_dim = 64;
  float rope_theta = 1e7f;
  std::array<int, 3> mrope_section{11, 11, 10};
  bool mrope_interleaved = true;
  // QSA indexer
  int idx_heads = 4, idx_dim = 128, idx_budget = 2048, idx_ratio = 4;
  // MoE
  int n_experts = 512, n_act = 10, moe_inter = 640, shared_inter = 640;
  bool norm_topk = true;
  // PLE
  std::vector<int> ple_layer_ids{2};   // config value: the PLE runs in front of layer (id − 1)
  int ple_dim = 2560, ple_conv = 4, ngram = 3, heads_per_ngram = 8, ngram_parts = 128;
  int64_t ngram_vocab_base = 20000000;
  int eos_id = 248044;                 // text_config.eos_token_id — the n-gram context resets at it
  // MTP
  int n_mtp = 1;
  int64_t max_position = 262144;
  std::vector<bool> is_attn;           // per layer: full attention (else linear)
  // vision
  int vision_layers = 27, vision_dim = 1152, vision_heads = 16, vision_inter = 4304, vision_patch = 16, vision_merge = 2,
      vision_temporal = 2, vision_out = 2560, vision_pos = 2304;
  int image_token_id = 248056, video_token_id = 248057, vision_start_id = 248053, vision_end_id = 248054;
  // derived
  int hcw() const { return hc * hidden; }
  int q_width() const { return n_heads * head_dim; }            // 6144
  int kv_width() const { return n_kv_heads * head_dim; }        // 512
  int lin_k_width() const { return lin_k_heads * lin_k_dim; }   // 2048
  int lin_v_width() const { return lin_v_heads * lin_v_dim; }   // 6144
  int lin_conv_ch() const { return 2 * lin_k_width() + lin_v_width(); }  // 10240
  int idx_topk_blocks() const { return idx_budget / idx_ratio; }         // 512
  int idx_width() const { return idx_budget + idx_ratio - 1; }           // 2051 selected positions per query (incl. its open group)
  static QwenConfig load(const std::string& ckpt_dir);
};

// A BF16 matrix in VRAM ([out, in], row-major as stored).
struct QMat {
  DevBuf buf;
  int out = 0, in = 0;
  const bf16* p() const { return buf.as<bf16>(); }
  // HIVE_QWEN_DENSE8: an 8-bit copy for the decode GEMV (fn_gemv8: fmt 1 = FP8 e4m3, 2 = INT8; scales per row and group of gs columns).
  //   fmt 3 = MXFP8 straight from the checkpoint (NVIDIA FP4-Local: e4m3 [out, in] + ue8m0 bytes [out, in/32] in s8, gs8 32) — the only
  //   copy of the weight (no bf16 buf); prefill expands it exactly or runs it as MXFP8 (HIVE_QWEN_DENSE_MX)
  DevBuf q8, s8;
  int fmt8 = 0, gs8 = 0;
};

// Hyper-connection weights of one block side ("gated residual"): down [hc_rank (+ hc rows of inject), hc·hidden] stacked so one GEMM
//   gives the mix's low-rank input and the combine's per-copy logits; up [hc·hidden, hc_rank]; norm (1 + w) [hc·hidden].
struct HcW {
  QMat down_inj;     // [hc_rank + hc, hc·hidden] — rows 0..hc_rank-1 = input_mix_weight_down, then block_inject_weight (absent for the final mixer)
  QMat up;           // [hc·hidden, hc_rank]
  NormW norm;        // [hc·hidden] (grouped per hidden, 1 + w) — bf16, or fp32 as stored (FP4-Local)
  bool inject = false;
};

struct LinW {        // Gated DeltaNet
  QMat in;           // [qkv 10240 | z 6144 | b 48 | a 48 = 16480, 2560] (rows of in_proj_qkv, in_proj_z, in_proj_b, in_proj_a)
  DevBuf conv_w;     // fp32 [10240, 4]
  DevBuf A_log, dt_bias;  // fp32 [48]
  DevBuf norm;       // bf16 [128] (plain weight — RMSNormGated)
  QMat out;          // [2560, 6144]
};
struct AttnW {       // full attention + QSA indexer
  QMat in;           // [q|gate 12288 (per head: 256 q then 256 gate) | k 512 | v 512 | indexer q 512 | indexer k 128 = 13952, 2560]
  DevBuf q_norm, k_norm;     // bf16 [256] (1 + w)
  DevBuf iq_norm, ik_norm;   // bf16 [128] (1 + w)
  QMat out;          // [2560, 6144]
};
struct MoeW {
  QMat router;       // [512 + 1, 2560]: router rows, then shared_expert_gate
  QMat sh_gu;        // [2·640, 2560]: shared gate rows then up rows
  QMat sh_down;      // [2560, 640]
};
struct PleW {
  QMat key, value;   // key_proj [10240, 2560] · value_proj [2560, 2560]
  NormW norm_key, norm_query, norm_conv;   // [10240] (grouped per hidden, 1 + w) — bf16, or fp32 as stored (FP4-Local)
  DevBuf conv_w;     // fp32 [10240, 4] (dilation = ngram)
};
struct LayerW {
  bool attn = false;
  HcW hc_attn, hc_mlp;
  LinW lin; AttnW att;
  MoeW moe;
  bool ple = false; PleW plew;
};
struct MtpW {
  QMat fc_emb, fc_hidden;              // [2560, 2560]
  DevBuf pre_emb_norm, pre_hidden_norm;  // bf16 [2560] · [10240] (whole-width, 1 + w)
  LayerW layer;                        // one full-attention layer (no PLE)
  HcW mixer;                           // final mixer of the MTP stream
  // routed experts: FP8 e4m3 with 128×128 block scales (stored as bf16 "weight_scale_inv", converted to fp32), all in VRAM
  DevBuf expert_arena;
  std::vector<std::array<glm::Fp8BMat, 3>> experts;   // gate [640, 2560] · up [640, 2560] · down [2560, 640]
  // or NVFP4 (W4A16, NVIDIA FP4-Local): one glm::ExpertRecLayout record per expert in expert_arena (the main layers' record format —
  //   fn_moe_decode_routed runs them); experts stays empty then
  bool nvfp4 = false;
  size_t rec_bytes = 0;                                // ExpertRecLayout::make(hidden, moe_inter).total
};

class QwenModel {
 public:
  // gpu_layers: layers [0, gpu_layers) are loaded into VRAM, the rest run on the CPU (hybrid placement, QwenEngine's tail engine);
  //   < 0 or ≥ n_layers = all. A split model loads no MTP layer.
  QwenModel(const std::string& ckpt_dir, bool load_mtp = true, int gpu_layers = -1);
  int gpu_layers() const { return gpu_layers_; }
  int dense8() const { return dense8_; }
  size_t dense8_max() const { return dense8_max_; }
  int mx8() const { return mx8_; }                     // matrices kept as the checkpoint's MXFP8 (QMat fmt8 3)   // elements of the largest 8-bit matrix (the engine's expansion scratch)   // 0 = bf16 decode GEMVs, 1 = FP8, 2 = INT8 (HIVE_QWEN_DENSE8)
  const QwenConfig& cfg() const { return cfg_; }
  Checkpoint& ckpt() { return ckpt_; }
  const LayerW& layer(int l) const { return layers_[l]; }
  const MtpW* mtp() const { return mtp_.get(); }
  const bf16* embed_host() const { return embed_host_.data(); }   // [vocab, hidden] bf16 in host RAM
  const QMat& lm_head() const { return lm_head_; }
  const HcW& mixer() const { return mixer_; }
  size_t vram_bytes() const { return vram_bytes_; }
  int ple_layer() const { return ple_layer_; }                     // decoder layer whose front runs the PLE (-1 = none)
  size_t free_at_load() const { return free_at_load_; }            // free VRAM before the first weight was copied (the VRAM budget's base)

 private:
  QMat mat(const std::string& name);
  QMat stack(const std::vector<std::string>& names);              // rows of several BF16 matrices with the same input width
  DevBuf vec_bf16(const std::string& name);
  NormW vec_norm(const std::string& name);                         // BF16 as is; F32 kept fp32 (HIVE_QWEN_NORM_BF16=1: rounded to bf16)
  DevBuf vec_f32(const std::string& name);                         // F32 or BF16 → fp32 on the device
  DevBuf conv_f32(const std::string& name, int channels);          // [C, 1, k] BF16 → fp32 [C, k]
  void load_hc(const std::string& p, HcW& w, bool inject);
  void load_layer(const std::string& p, bool attn, LayerW& L);
  void load_moe(const std::string& p, MoeW& w);
  QwenConfig cfg_;
  Checkpoint ckpt_;
  std::vector<LayerW> layers_;
  std::unique_ptr<MtpW> mtp_;
  std::vector<bf16> embed_host_;
  QMat lm_head_;
  HcW mixer_;
  size_t vram_bytes_ = 0;
  int ple_layer_ = -1;
  int gpu_layers_ = 0;
  int dense8_ = 0;
  size_t dense8_max_ = 0;
  int mx8_ = 0, f32_rounded_ = 0, f32_kept_ = 0;
  void make_dense8();
  size_t free_at_load_ = 0;
};

}  // namespace hive::qwen
