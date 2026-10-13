// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash on the CPU only (hived_glm_cpu). The same model as hive/glm/glm_engine.h — the math contract is the GPU family's (glm_model.h,
//   kda.h, dsa.h, glm_kernels.h, glm_moe.h, glm_mtp.cpp, glm_vision.h) and the per-layer reference is oracle/glm5_oracle.py (the transformers
//   GLM-5 Next modules in float32) — computed with hive/cpu/cpu_backend.h: every weight in NUMA-split host memory (NVFP4 routed experts and
//   dense MLP, BF16 KDA / indexer / router / lm_head, the Z.ai FP8 tensors where HIVE_GLM_FP8_DIR provides them: DSA q_a / q_b / kv_a / o,
//   shared experts, the NextN layer's routed experts), activations fp32 with bf16 roundings where the GPU path keeps bf16.
// Contract towards fam::Runtime (hive/glm/hived_family_cpu.h): prefill one sequence, decode M ≤ 8 sequences (one token each), MTP draft /
//   verify / rollback like GlmEngine; per-sequence state in host memory, saved / restored as byte buffers.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hive/cpu/cpu_backend.h"

namespace hive::glm::cpu {

// config.json (text_config + vision_config) — the fields GlmConfig (glm_model.h) reads, parsed here without the GPU library
struct CpuConfig {
  int hidden = 4096, n_layers = 45, vocab = 154880;
  int n_heads = 64, q_lora = 1536, kv_lora = 512, qk_head = 256, v_head = 256;
  int kda_heads = 64, kda_dim = 128, conv_k = 4;
  float kda_lower_bound = -5.f;
  int index_heads = 32, index_dim = 128, index_topk = 2048, index_kpool = 4;
  int dense_inter = 12288, moe_inter = 2048, n_routed = 288, n_act = 8;
  float routed_scale = 2.5f, swiglu_limit = 10.f, rms_eps = 1e-5f, hc_eps = 1e-6f;
  int hc = 4, sinkhorn_iters = 20, n_mtp = 1;
  std::vector<int> eos_ids{154820, 154827, 154829};
  std::vector<bool> is_kda, is_moe;
  int vision_layers = 24, vision_dim = 1024, vision_heads = 16, vision_inter = 4096, vision_patch = 14, vision_merge = 2, vision_temporal = 2,
      vision_out = 4096, vision_proj_inter = 10240;
  int image_token_id = 154854;
  static CpuConfig load(const std::string& ckpt_dir);
};

struct CpuSeq {
  int64_t pos = 0, cap = 0;
  std::vector<int32_t> tokens;
  std::vector<std::vector<float>> conv, state;   // per KDA layer: [3·24576] (slot 0 = oldest input), [64·128·128] (S[h][k][v])
  struct Dsa { std::vector<uint16_t> c, kI, gs, pooled; };  // bf16 rows: [n, 512] latent · [n, 128] indexer key · [n, 128] pool gate · [n/4 + 1, 128]
  std::vector<Dsa> dsa;                          // per DSA layer
  Dsa mtp;                                       // NextN layer cache (entry j = (h_j, x_{j+1}))
  std::vector<float> h_last;                     // [4096] final normed hidden of the last processed position (MTP input)
  bool h_last_valid = false;
  int64_t mtp_written = 0;
  bool broken = false;
  std::vector<std::vector<uint8_t>> save_bytes() const;
  void load_bytes(const std::vector<std::vector<uint8_t>>& b);
};

// wall time per section (ms)
struct CpuStats {
  double ms_total = 0; long steps = 0;
  double ms_embed = 0, ms_hc = 0, ms_kda_proj = 0, ms_kda_core = 0, ms_dsa_proj = 0, ms_dsa_idx = 0, ms_dsa_attn = 0, ms_dense = 0, ms_router = 0,
         ms_experts = 0, ms_head = 0, ms_mtp = 0;
};

class GlmCpuEngine {
 public:
  explicit GlmCpuEngine(const std::string& ckpt_dir, bool load_mtp = true);
  // Hybrid use (the GPU family's CPU-placed layers, GlmEngine with HIVE_GLM_GPU_LAYERS / HIVE_VRAM_LIMIT_MB): only layers first..last are
  //   loaded — no embedding, head or NextN layer — and the routed experts are not loaded either: `experts` computes them (the GPU family's
  //   GlmExperts CPU pool, which already holds every expert in RAM). experts(layer, M, ids [M, K], w [M, K], x [M, H], y [M, H]) must
  //   write y = Σ_k w_k · bf16(expert_k(x)) (fp32; y arrives zeroed).
  struct Hybrid {
    int first = 0, last = -1;
    std::function<void(int layer, int M, const int32_t* ids, const float* w, const float* x, float* y)> experts;
  };
  GlmCpuEngine(const std::string& ckpt_dir, const Hybrid& hy);
  ~GlmCpuEngine();
  const CpuConfig& cfg() const;
  // HIVE_GLM_CPU_LAYERS=a:b (validation): only layers a..b are loaded and run; without the last layer there is no head (logits untouched)
  int first_layer() const;
  int last_layer() const;
  void init_seq(CpuSeq& s, int64_t cap);
  void reset_seq(CpuSeq& s);
  // Vision rows of the next prefill: rows [row, row + n) of the ids passed to prefill take emb [n, 4096] (bf16 values)
  struct EmbedOverride { int64_t row; int n; std::vector<float> emb; };
  void set_embed_override(std::vector<EmbedOverride> v);
  bool has_vision() const;
  // patches bf16 [gh·gw, 1176] (merge-block order) → [gh·gw/4, 4096] (bf16 values)
  void encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out);
  void prefill(CpuSeq& s, const int32_t* ids, int T, float* logits);
  void decode(CpuSeq* const* seqs, const int32_t* ids, int M, float* logits);
  bool mtp_on() const;
  int mtp_k() const;
  // drafts from token tok (not yet processed) and the sequence's last hidden; conf = softmax probability of each draft
  int mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf, float* logits0 = nullptr);
  void verify_batch(CpuSeq* const* seqs, const int* M, int S, const int32_t* ids, float* logits);
  void rollback_batch(const int* n_keep, int S);
  // Hybrid: run the loaded layers on M rows — row r = position pos[r] of seqs[r]; h [M, 4·hidden] fp32 (bf16 values) is read and overwritten
  //   with the streams after the last loaded layer. Sequence positions are not advanced (the caller owns them). With parts (S > 0, rows =
  //   S runs of consecutive positions, part_rows[p] each — a verify step), the KDA state after every row but a part's last is kept;
  //   restore_verify(n_keep, S) then returns each part's state to its first n_keep rows (no-op when all are kept).
  void run_layers(CpuSeq* const* seqs, const int64_t* pos, int M, float* h, bool prefill, const int* part_rows = nullptr, int S = 0);
  void restore_verify(const int* n_keep, int S);
  // validation hooks: after every layer the hc streams [rows, 4·4096]; before layer l (l > first, or l == first > 0) of a prefill the
  //   streams may be overwritten (per-layer injection of the reference's layer inputs)
  std::function<void(int layer, const float* h, int rows)> layer_hook;
  std::function<void(int layer, float* h, int rows)> layer_inject;
  std::function<void(int layer)> layer_boundary;
  CpuStats& stats();
  struct Impl;
 private:
  std::unique_ptr<Impl> p_;
};

}  // namespace hive::glm::cpu
