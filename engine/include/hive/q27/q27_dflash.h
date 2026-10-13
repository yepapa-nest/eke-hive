// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DFlash2 block drafter for Qwen3.8-27B (HIVE_Q27_DFLASH=<draft checkpoint dir>, e.g. Qwen3.8-27B-DFlash2) — kernels.
//
// The drafter's math follows the reference implementation of z-lab/dflash (dflash/model_mlx.py: DFlashAttention, GroupedDynamicCausalConv,
//   DFlash2DecoderLayer, CandidateSelector, DFlash2DraftModel), MIT License, Copyright (c) 2026 Z Lab — adapted here to CUDA kernels; no code
//   was copied. Summary of that math:
//   context  : target hidden of layers 5/19/33/47/61 (decoder-layer outputs) concatenated [25600] → fc → RMSNorm (hidden_norm) = h_ctx;
//              every draft layer projects h_ctx with its own k_proj / v_proj (k_norm, rope at the token position) into its context KV
//              (sliding window 2048: a block query at position q sees context keys > q − 2048).
//   block    : rows = [anchor (the next token, not yet fed to the target), mask × 7] at positions ctx_len … ctx_len + 7; embeddings = the
//              target's table. Each of 5 layers (Qwen3-style, head_dim 128, 32 / 8 heads, plain-weight RMSNorm):
//                x' = conv_a(norm1(x)) · attention (block keys/values + context, non-causal inside the block) → x += conv_b(attn)
//                x' = conv_a(norm2(x)) · MLP (SiLU)                                                       → x += conv_b(mlp)
//              conv (grouped dynamic causal conv, two taps over block rows, groups of 16 channels):
//                dyn = kernel_projection(norm(x)) [2 · 2 · 320];  y[t, c] = Σ_o (base[o, c] + dyn[t, o, c / 16]) · in[t − o, c]
//   output   : rows 1..7 → final norm → target lm_head (logits) → top-16 candidates per row; candidate selector (greedy):
//                score(c) = logit(c) + Σ_r P[prev, r] · Hp[t, r] · S[c, r],  Hp = hidden_projection(normed hidden), prev = previous pick
//                (the anchor for row 1).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::q27::df {

using bf16 = __nv_bfloat16;

constexpr int kLayers = 5, kHeads = 32, kKvHeads = 8, kHeadDim = 128, kBlock = 8, kWindow = 2048, kTopK = 16, kRank = 256, kGroup = 16;
constexpr int kRing = kWindow + 32;          // context ring slots per layer: rejected verify positions never land in a live window slot
constexpr int kKvW = kKvHeads * kHeadDim;    // 1024

// Context ring of a sequence: k / v [kLayers][slots][kKvW], position p in slot p % slots (DFlash2: kRing slots with a 2048 window;
//   DSpark: full context, slots grown beyond the position so nothing wraps).
struct CtxRow { bf16* k; bf16* v; int pos; int slots; };   // the ring and the position of this row
struct BlkSeq { const bf16* k; const bf16* v; int ctx; int anchor; int slots; };   // a drafted sequence: ring, context length, anchor
// Draft blocks of S sequences are stacked: rows [B·s, B·s + B) belong to sequence s (BlkSeq array of S entries on the device);
//   B = 8 (DFlash2: anchor + 7 masks, outputs from rows 1..7) or 7 (DSpark: anchor + 6 masks, outputs from rows 0..6).
// Rope: inv_freq table (DSpark: YaRN-scaled) and mscale (YaRN attention factor on cos / sin; 1 = none).

// out = bf16(x · rsqrt(mean(x²) + eps) · w) per row of K (plain-weight RMSNorm)
void rmsnorm(const bf16* x, int ldx, const bf16* w, int M, int K, float eps, bf16* out, int ldo, cudaStream_t st);
// out[r, :] = y ? bf16(h + y) : h — a decoder layer's output while its MLP output still waits in y (row stride of out: ldo)
void capture(const bf16* h, const bf16* y, int M, int H, bf16* out, int ldo, cudaStream_t st);
// context K / V of M rows: kv [M, kLayers · 2 · kKvW] (per layer [k | v]) → k_norm + rope(pos) / copy → ring slot pos % kRing
void ctx_kv(const bf16* kv, const bf16* k_norm, const CtxRow* rows, int M, float eps, const float* inv_freq, float mscale, cudaStream_t st);
// block rows (8 per sequence): proj [8S, 32·128 | 8·128 | 8·128] → q (q_norm + rope) [8S, 32, 128], k (k_norm + rope) / v [8S, 8, 128];
//   positions ctx_s + i
void blk_prep(const bf16* proj, const bf16* q_norm, const bf16* k_norm, const BlkSeq* seqs, int S, int B, float eps, const float* inv_freq,
              float mscale, bf16* q, bf16* k, bf16* v, cudaStream_t st);
// attention of each sequence's 8 block queries over its context ring (layer slice, positions [max(0, q − 2047), ctx)) and its 8 block keys
//   (non-causal); out bf16 [8S, 32 · 128]. ring_off: element offset of the layer slice added to BlkSeq::k / v
//   layer: the ring's layer slice · window: keys of positions > q − window (0 = the whole context)
void attention_prepare();   // scratch of the grouped attention (call once outside graph capture)
void attention(const bf16* q, const bf16* bk, const bf16* bv, const BlkSeq* seqs, int layer, int S, int B, int window, float scale, bf16* out,
               cudaStream_t st);
// grouped dynamic causal conv over the block rows (taps stay inside each sequence's 8 rows): dyn fp32 [8S, 1280] (half a = 0 prepare /
//   1 finish), base fp32 [2 taps, H] of that half; residual = false: out = bf16(conv(in)); true: out = bf16(out + bf16(conv(in)))
void conv(const bf16* in, int ld_in, const float* dyn, int half, const float* base, int rows, int B, int H, bf16* out, bool residual, cudaStream_t st);
// greedy candidate selection, sequence s = block s: rows [7s, 7s + 7) of cidx / cval [., 16] (cidx through map when map != nullptr), hp fp32
//   [., 256], codebooks bf16 [V, 256], first predecessor = seqs[s].anchor; out ids / conf [7S] (softmax probability of the pick among the 16).
//   edges: work fp32 [S, 7, 16, 16]
//   hp == nullptr: no hidden factor (DSpark's Markov bias: pred = w1, succ = w2). DSpark confidence (conf_h != nullptr): conf = sigmoid(conf_h[row]
//   + conf_w · w1[prev] + conf_b) — the confidence head on [hidden ; w1[prev]] — instead of the softmax probability.
void select(const int32_t* cidx, const float* cval, const float* hp, const bf16* pred, const bf16* succ, const int32_t* map, const BlkSeq* seqs,
            int S, float* edges, int32_t* ids, float* conf, cudaStream_t st, const float* conf_h = nullptr, const bf16* conf_w = nullptr, float conf_b = 0.f);
// the 16 largest logits of each row (ties → smaller index): idx / val [rows, 16]; work fp32 [rows · 32 · 16 · 2]; N ≤ 131072
void top16(const float* logits, int rows, int N, float* work, int32_t* idx, float* val, cudaStream_t st);
// bf16 [N, K] → FP8 e4m3 [N, K] with an fp32 per-row scale (max |w| / 448) — HIVE_Q27_DFLASH_FP8=1 (drafter only: drafts never change outputs)
void to_fp8_rows(const bf16* w, int N, int K, uint8_t* q, float* scale, cudaStream_t st);

// debug: number of non-finite values among n bf16 / fp32 values → *out (device int)
void count_nonfinite_bf16(const bf16* x, size_t n, int* out, cudaStream_t st);
void count_nonfinite_f32(const float* x, size_t n, int* out, cudaStream_t st);

}  // namespace hive::q27::df
