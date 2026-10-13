// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B (config.json model_type "qwen3_5", architecture Qwen3_5ForConditionalGeneration; the official compressed-tensors checkpoint
//   Qwen3.8-27B-NVFP4) — configuration and the math contract shared by the CPU engine (hive/q27/q27_cpu.h, hived_q27_cpu) and the GPU engine
//   (hive/q27/q27_engine.h, hived_q27). oracle/q27_oracle.py is the executable form of this contract (per-layer reference).
//
// Model (every value read from config.json / the tensor shapes on 2026-10-09; the loader checks the shapes it relies on):
//   64 decoder layers, layer_types: every 4th one (3, 7, …, 63) full attention, the other 48 Gated DeltaNet linear attention. Dense — no MoE,
//   no hyper-connections, no PLE, no sparse attention indexer. hidden 5120, vocab 248320, untied lm_head.
//   Layer: h = bf16(h + mixer(n1(h))); h = bf16(h + mlp(n2(h))), n = (1 + w) RMS norm (eps 1e-6, fp32 math, one bf16 rounding).
//   Gated DeltaNet: in_proj_qkv [10240 = q 2048 | k 2048 | v 6144], in_proj_z [6144], in_proj_b / in_proj_a [48]; 16 key heads, 48 value
//     heads × 128 (value head h reads key head h / 3); causal depthwise conv (kernel 4, no bias) over the qkv channels, SiLU, bf16;
//     q, k l2-normed (x / sqrt(Σx² + 1e-6)), q · 128^−0.5; decay exp(−exp(A_log) · softplus(a + dt_bias)), beta bf16(sigmoid(b));
//     state S[k][v] fp32: S ·= decay; δ = (v − Sᵀk)·beta; S += k ⊗ δ; o = Sᵀq (bf16);
//     output norm: bf16(bf16(bf16(o · rsqrt(mean o² + eps)) · w) · silu(z))  — plain w (not 1 + w), SiLU gate (config output_gate_type
//     "swish"; Flash-Next uses sigmoid here) — then out_proj [5120, 6144].
//   Full attention: q_proj [24 · 2 · 256] (per head: 256 query then 256 gate), k_proj / v_proj [4 · 256]; (1 + w) RMS norm of q and k per
//     head; interleaved M-RoPE on dims 0..63 (θ = 1e7, sections 11/11/10, NeoX halves, fp32 cos / sin, one rounding); GQA (head h reads kv
//     head h / 6), softmax(q·k / 16) fp32; out = bf16(bf16(att) · sigmoid(gate)); o_proj [5120, 6144].
//   MLP: bf16(bf16(silu(gate)) · up) → down; gate / up [17408, 5120], down [5120, 17408].
//   Head: final (1 + w) norm, lm_head (fp32 logits).
//   MTP (mtp.*, BF16): entry j = (residual stream h_j before the final norm, next token x_{j+1}) at rope position j:
//     fc([n_e(emb(x_{j+1})) | n_h(h_j)]) [5120, 10240] → one full-attention decoder layer (its own KV cache) → (1 + w) mtp.norm → lm_head.
//   Vision (model.visual, BF16): 27 blocks × 1152 (16 heads × 72), patch 16, temporal 2, 2×2 merge → 5120 (no deepstack) — the same encoder
//     structure as Flash-Next (hive/qwen/qwen_vision.h) with a 5120-wide merger output.
// Weights: FP8 e4m3 with one bf16 scale per output row — attention q/k/v/o, linear-attention in_proj_qkv / in_proj_z / out_proj, the MLP of
//   layers 56-63, lm_head; NVFP4 (compressed-tensors: e2m1 pairs, low nibble = even column, e4m3 scale per 16 columns, fp32 weight_global_scale
//   g: W = e2m1 · s16 / g) — the MLP of layers 0-55; BF16 — in_proj_a / in_proj_b, conv, A_log, dt_bias, norms, embeddings, MTP, vision.
//   Activations are not quantized (W8A16 / W4A16). The checkpoint's FP8 KV scales (k_scale / v_scale) are not used: KV caches are bf16.
// Rounding: every GEMM output is rounded to bf16 (fp32 accumulation over the dequantized weights); see oracle/q27_oracle.py for each point.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace hive::q27 {

struct Q27Config {
  int hidden = 5120, n_layers = 64, vocab = 248320, inter = 17408;
  float rms_eps = 1e-6f;
  // Gated DeltaNet
  int lin_k_heads = 16, lin_v_heads = 48, lin_k_dim = 128, lin_v_dim = 128, conv_k = 4;
  // full attention
  int n_heads = 24, n_kv_heads = 4, head_dim = 256, rotary_dim = 64;
  float rope_theta = 1e7f;
  std::array<int, 3> mrope_section{11, 11, 10};
  bool mrope_interleaved = true;
  int n_mtp = 1;
  int64_t max_position = 262144;
  std::vector<bool> is_attn;          // per layer
  // vision
  int vision_layers = 27, vision_dim = 1152, vision_heads = 16, vision_inter = 4304, vision_patch = 16, vision_merge = 2, vision_temporal = 2,
      vision_out = 5120, vision_pos = 2304;
  int image_token_id = 248056, video_token_id = 248057, vision_start_id = 248053, vision_end_id = 248054, eos_id = 248044;
  // derived
  int q_width() const { return n_heads * head_dim; }                    // 6144
  int kv_width() const { return n_kv_heads * head_dim; }                // 1024
  int lin_k_width() const { return lin_k_heads * lin_k_dim; }           // 2048
  int lin_v_width() const { return lin_v_heads * lin_v_dim; }           // 6144
  int lin_conv_ch() const { return 2 * lin_k_width() + lin_v_width(); }  // 10240
  int n_attn() const { int n = 0; for (bool b : is_attn) n += b; return n; }
  int n_lin() const { return n_layers - n_attn(); }
  static Q27Config load(const std::string& ckpt_dir);   // throws std::runtime_error on a different model
};

}  // namespace hive::q27
