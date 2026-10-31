// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gated DeltaNet linear-attention layers of Qwen3.8-Flash-Next — everything between the fused input projection and out_proj (both
//   cuBLAS / GEMV in the engine). Adapted from the KDA kernels of hive/glm/kda.h (same work split and state layout); the differences:
//   16 key heads shared by 48 value heads (value head h reads key head h / 3), one decay value per head instead of per key channel, the
//   gate / step size from a softplus, and the output norm gated by sigmoid(z) with a plain (not 1 + w) weight.
// Math contract, per token and value head h (kh = h / 3):
//   1. causal depthwise conv1d (kernel 4, no bias) over the 10240 channels [q 2048 | k 2048 | v 6144] of the projection, SiLU, rounded
//      to bf16 (the reference's conv output dtype).
//   2. q, k: per key head l2norm in fp32, x / sqrt(Σx² + 1e-6); q *= 1/sqrt(128). v as is.
//   3. g = −exp(A_log[h]) · softplus(a + dt_bias[h]) (softplus(x) = x above 20) · decay = exp(g) · beta = bf16(sigmoid(b)).
//   4. state S[k][v] fp32 per value head: S *= decay; kv = Σ_k S·k; δ = (v − kv)·beta; S += k ⊗ δ; o = Σ_k S·q   (o rounded to bf16)
//   5. out = bf16( w · (o · rsqrt(mean(o²) + eps)) · sigmoid(z) ) per value head (128 wide).
// State layouts (per sequence): conv_state fp32 [3][10240] (slot 0 = oldest pre-conv input t−3 … slot 2 = t−1; zeros when fresh) ·
//   state fp32 [48][128 (k)][128 (v)].
// Projection rows (engine fused GEMM output, row stride ld): [qkv 10240 | z 6144 | b 48 | a 48].
#pragma once
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::qwen {

constexpr int kGdnKHeads = 16, kGdnVHeads = 48, kGdnDim = 128;
constexpr int kGdnKWidth = kGdnKHeads * kGdnDim;          // 2048
constexpr int kGdnVWidth = kGdnVHeads * kGdnDim;          // 6144
constexpr int kGdnConvCh = 2 * kGdnKWidth + kGdnVWidth;   // 10240
constexpr int kGdnConvTaps = 4;
constexpr int kGdnProjWidth = kGdnConvCh + kGdnVWidth + 2 * kGdnVHeads;  // 16480
constexpr int kGdnZOff = kGdnConvCh, kGdnBOff = kGdnConvCh + kGdnVWidth, kGdnAOff = kGdnBOff + kGdnVHeads;
constexpr size_t kGdnConvStateFloats = (size_t)(kGdnConvTaps - 1) * kGdnConvCh;            // 30720
constexpr size_t kGdnStateFloats = (size_t)kGdnVHeads * kGdnDim * kGdnDim;                // 786432

struct GdnParams {
  const float* conv_w;       // [10240, 4] fp32 (tap 3 = current token)
  const float* A_log;        // [48] fp32
  const float* dt_bias;      // [48] fp32
  const __nv_bfloat16* norm_w;  // [128] bf16
  float norm_eps;
};

// Prefill/extend T tokens of ONE sequence. proj bf16 [T, ld] (layout above); out bf16 [T, 6144]. conv_state / state are read and updated
//   in place. workspace ≥ gdn_workspace_bytes(T) (256-byte aligned). T = 0 is a no-op.
size_t gdn_workspace_bytes(int T);
void gdn_forward_seq(const GdnParams& p, const __nv_bfloat16* proj, int ld, int T, float* conv_state, float* state, __nv_bfloat16* out,
                     void* workspace, size_t ws_bytes, cudaStream_t st);
// Batched decode: M sequences, one token each (row m), per-row state pointers (device-readable arrays of M pointers). Bit-identical to
//   gdn_forward_seq with T = 1 on the same sequence.
void gdn_decode_batch(const GdnParams& p, const __nv_bfloat16* proj, int ld, int M, float* const* conv_states, float* const* states,
                      __nv_bfloat16* out, cudaStream_t st);
// Verify rows: R consecutive tokens of ONE sequence, bit-identical to R gdn_decode_batch calls with M = 1; after row r < R−1 the conv
//   window / state are also stored to snap_conv[r] / snap_state[r] (rollback to that row).
// snap_init = 1 (replay rollback): snap_state[0] gets the state before row 0, no per-row state snapshots; snap_conv0 (optional): the
//   conv window before row 0. snap_state / snap_conv may be null (no snapshots).
void gdn_decode_rows(const GdnParams& p, const __nv_bfloat16* proj, int ld, int R, float* const* conv_state, float* const* state,
                     float* const* snap_conv, float* const* snap_state, __nv_bfloat16* out, cudaStream_t st, int snap_init = 0, float* const* snap_conv0 = nullptr);
// gdn_decode_rows for several sequences in one launch (same per-block arithmetic → bit-identical); parts of > 8 rows or more than 16 parts
//   fall back to one gdn_decode_rows per part
struct GdnRowsPart {
  const __nv_bfloat16* proj; int R; float* const* conv_state; float* const* state; float* const* snap_conv; float* const* snap_state;
  __nv_bfloat16* out; int snap_init; float* const* snap_conv0;
};
void gdn_decode_rows_multi(const GdnParams& p, int ld, int P, const GdnRowsPart* parts, cudaStream_t st);
// gdn_decode_rows_multi with the parts' rows in device memory — vpt: int pairs (first row, rows ≤ 8) per part, P ≤ 16 — and the per-part
//   arguments derived from bases: part p reads proj / out rows from its first row, conv_state / state = the row pointer tables + first row,
//   its snapshot pointers at snap + p · snap_stride (conv 0..6 · conv window before the verify 7 · state 8..), replay snapshots (snap_init)
//   for parts ≥ replay_from. Same arithmetic per block as gdn_decode_rows_multi (bit-identical); a captured graph stays valid for any split.
void gdn_decode_rows_multi_dyn(const GdnParams& p, int ld, int P, const int* vpt, const __nv_bfloat16* proj, float* const* conv_state,
                               float* const* state, float* const* snap, int snap_stride, int replay_from, __nv_bfloat16* out, cudaStream_t st);

}  // namespace hive::qwen
