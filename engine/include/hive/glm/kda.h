// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// KDA (Kimi Delta Attention) linear-attention layers of GLM-5.3-Flash — everything between the input projections and
// o_proj (both done by the caller with cuBLAS). Reference: transformers 5.18 Glm5NextTextLinearAttention.
//   H = 64 heads, head_dim 128 (K = V = 128), q/k/v width 8192 each.
//   1. causal depthwise conv1d (kernel 4, no bias) over the 24576 channels [q|k|v], SiLU, rounded to bf16
//      (HF casts the conv output back to the activation dtype).
//   2. q, k: per-head l2norm in fp32, x / sqrt(Σx² + 1e-6); q *= 1/sqrt(128). v as is.
//   3. g = -5 · sigmoid(exp(A_log[h]) · (g_pre + dt_bias)) (fp32, gate_lower_bound = -5); beta = sigmoid(beta_logit)
//      rounded to bf16 (HF computes it in the activation dtype).
//   4. per head, state S[k][v] fp32: S *= exp(g)[k]; kv = Σ_k S·k; δ = (v − kv)·beta; S += k ⊗ δ; o = Σ_k S·q
//      (o rounded to bf16 like HF's core_attn_out).
//   5. out = bf16( (o_norm_w · (o · rsqrt(mean(o²) + 1e-5))) · sigmoid(gate) ) per head (RMSNormGated).
// State layouts (per sequence): conv_state fp32 [3][24576] — slot 0 = oldest pre-conv input (token t-3), slot 2 =
// newest (t-1); zeros for a fresh sequence. state fp32 [64][128 (k)][128 (v)] (= HF recurrent_states [H, K, V]).
#pragma once
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::glm {

constexpr int kKdaHeads = 64;
constexpr int kKdaHeadDim = 128;
constexpr int kKdaWidth = kKdaHeads * kKdaHeadDim;   // 8192
constexpr int kKdaConvChannels = 3 * kKdaWidth;      // 24576
constexpr int kKdaConvTaps = 4;
constexpr size_t kKdaConvStateFloats = (size_t)(kKdaConvTaps - 1) * kKdaConvChannels;          // 73728
constexpr size_t kKdaStateFloats = (size_t)kKdaHeads * kKdaHeadDim * kKdaHeadDim;              // 1048576

struct KdaParams {
  const float* conv_w;               // [24576, 4] fp32, rows = q_conv1d | k_conv1d | v_conv1d (tap 3 = current token)
  const float* A_log;                // [64] fp32
  const float* dt_bias;              // [8192] fp32
  const __nv_bfloat16* o_norm_w;     // [128] bf16, shared by all heads
};

// Prefill/extend T tokens of ONE sequence (also used for MTP verify chunks). Rows are tokens:
//   qkv_pre [T, 24576] · g_pre [T, 8192] · beta_logit [T, 64] · gate [T, 8192] · out [T, 8192] (bf16, dense rows).
// conv_state fp32 [3*24576] and state fp32 [64*128*128] are read and updated in place (state after the last token).
// workspace must hold kda_workspace_bytes(T) bytes (256-byte aligned). out may alias gate (each element of gate is
// read before the same element of out is written); it must not alias anything else. T = 0 is a no-op.
void kda_forward_seq(const KdaParams& p, const __nv_bfloat16* qkv_pre, const __nv_bfloat16* g_pre,
                     const __nv_bfloat16* beta_logit, const __nv_bfloat16* gate, int T, float* conv_state, float* state,
                     __nv_bfloat16* out, void* workspace, size_t ws_bytes, cudaStream_t st);
size_t kda_workspace_bytes(int T);

// Batched decode: M sequences, one token each (row m of every input), per-row state pointers (device arrays of M
// device pointers). Bit-identical to kda_forward_seq with T = 1 on the same sequence. Rows must use distinct states.
void kda_decode_batch(const KdaParams& p, const __nv_bfloat16* qkv_pre, const __nv_bfloat16* g_pre,
                      const __nv_bfloat16* beta_logit, const __nv_bfloat16* gate, int M, float* const* conv_states,
                      float* const* states, __nv_bfloat16* out, cudaStream_t st);

// Verify rows: R consecutive tokens of ONE sequence (rows of every input), bit-identical to R kda_decode_batch calls with M = 1 in
//   order. conv_state / state: device arrays holding the sequence's single pointer. After row r < R−1 the conv window and state are
//   also stored to snap_conv[r] / snap_state[r] (device arrays of R−1 device pointers) — what the per-row path copied with dcopy.
void kda_decode_rows(const KdaParams& p, const __nv_bfloat16* qkv_pre, const __nv_bfloat16* g_pre, const __nv_bfloat16* beta_logit,
                     const __nv_bfloat16* gate, int R, float* const* conv_state, float* const* state, float* const* snap_conv,
                     float* const* snap_state, __nv_bfloat16* out, cudaStream_t st);

}  // namespace hive::glm
