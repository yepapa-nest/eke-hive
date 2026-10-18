// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3-ASR kernels (qasr_kernels.cu): the small element-wise / norm / attention / GEMV kernels around the cuBLAS GEMMs.
// Conventions: activations between GEMMs are bf16, everything that is accumulated (residual streams, GEMM outputs that are post-
//   processed) is fp32; all reductions are done in fp32 in a fixed order (deterministic run to run, independent of batch size for the
//   per-row kernels). Shapes are row-major; "rows" are tokens.
#pragma once
#include <cstdint>

#include "hive/common.h"

namespace hive::qasr {

// ---- norms ----
// y[r] = bf16( x[r] * rsqrt(mean(x[r]^2) + eps) * w )          (decoder RMSNorm, n = 2048)
void rmsnorm(const float* x, const float* w, bf16* y, int rows, int n, float eps, cudaStream_t st);
// y[r] = bf16( (x[r] - mean) * rsqrt(var + eps) * w + b )       (encoder LayerNorm, biased variance)
void layernorm(const float* x, const float* w, const float* b, bf16* y, int rows, int n, float eps, cudaStream_t st);

// ---- epilogues for fp32 GEMM outputs ----
enum class Epi { kBf16 = 0, kGeluBf16 = 1, kAddF32 = 2 };
// kBf16: out_bf16 = in + bias · kGeluBf16: out_bf16 = gelu_erf(in + bias) · kAddF32: out_f32 += in + bias  (bias may be null)
void bias_epilogue(const float* in, const float* bias, int rows, int n, Epi mode, bf16* out_bf16, float* out_f32, cudaStream_t st);
// out[r, i] = bf16( silu(gu[r, i]) * gu[r, inter + i] )   (gu = [rows, 2*inter] = gate || up)
void swiglu(const float* gu, bf16* out, int rows, int inter, cudaStream_t st);

// ---- embeddings ----
// x[r] = float(table[ids[r]])  (n = row width)
void embed_rows(const bf16* table, const int32_t* ids, float* x, int rows, int n, cudaStream_t st);

// ---- decoder attention prologue ----
// qkv: fp32 [rows, (H + 2*KV) * 128] (q heads, then k heads, then v heads). Per row r (token at position pos[r] of sequence slot[r]):
//   q heads: RMSNorm(head, qn) -> RoPE -> q_out[r, h, :] (bf16)
//   k heads: RMSNorm(head, kn) -> RoPE -> kcache[slot][pos][kvh][:] (bf16);  v heads -> vcache[slot][pos][kvh][:] (bf16)
// RoPE: rotate-half, (x[i], x[i+64]) rotated by the angle in rope_cos/sin[pos][i] (tables [max_pos, 64], fp32).
// Cache layout per layer: [n_slots][max_pos][KV][128] (slot_stride = max_pos*KV*128 elements).
void qk_norm_rope_store(const float* qkv, const float* qn, const float* kn, const float* rope_cos, const float* rope_sin,
                        const int32_t* pos, const int32_t* slot, int rows, int H, int KV, float eps, bf16* q_out, bf16* kcache,
                        bf16* vcache, long slot_stride, cudaStream_t st);

// ---- attention ----
// Decoder prefill (causal): queries q [nq, H, 128] at positions pos0 + i of one sequence; keys/values = the cache rows 0..pos0+i of
//   that slot (kc/vc point at slot's [max_pos][KV][128]). out [nq, H*128] bf16. Scale 1/sqrt(128).
void attn_prefill_causal(const bf16* q, const bf16* kc, const bf16* vc, int nq, int pos0, int H, int KV, bf16* out, cudaStream_t st);
// Encoder (bidirectional, block-diagonal): qkv bf16 [n, 3*H*64]; token i attends to tokens of its window [w*win, min((w+1)*win, n)).
void attn_encoder(const bf16* qkv, int n, int H, int win, bf16* out, cudaStream_t st);
// Decoder decode step: m sequences, query q [m, H, 128]; sequence i reads slot[i] of the cache, keys 0..pos[i] (inclusive).
//   kc/vc = layer base ([n_slots][max_pos][KV][128]). out [m, H*128] bf16.
void attn_decode(const bf16* q, const bf16* kc, const bf16* vc, const int32_t* pos, const int32_t* slot, int m, int H, int KV,
                 long slot_stride, bf16* out, cudaStream_t st);

// ---- decode GEMVs (m <= 4 rows of activations, bf16 weights [N, K] row-major, K % 256 == 0) ----
// out[r, n] = sum_k x[r, k] * W[n, k]   (fp32)
void gemv(const bf16* W, const bf16* x, int m, int N, int K, float* out, cudaStream_t st);
// out[r, n] += sum_k x[r, k] * W[n, k]  (fp32 residual add)
void gemv_add(const bf16* W, const bf16* x, int m, int N, int K, float* out, cudaStream_t st);
// out[r, i] = bf16( silu(g) * u ), g = x[r]·W[i], u = x[r]·W[inter + i]   (W = gate || up, [2*inter, K])
void gemv_swiglu(const bf16* W, const bf16* x, int m, int inter, int K, bf16* out, cudaStream_t st);

// ---- sampling helpers ----
// ids[r] = argmax(logits[r, :V]) (first index on ties)
void argmax_rows(const float* logits, int rows, int V, int32_t* ids, cudaStream_t st);

// ---- audio encoder front ----
// mel [n_mels, n_frames] fp32 (column = frame). For chunks c = c0 .. c0+nc-1 (chunk c covers frames [c*100, min(c*100+100, n_frames)),
//   zero beyond), conv1 (1 -> C, 3x3, stride 2, pad 1) + GELU over a padded chunk of `lp` frames.
//   out bf16 [nc][F1][T1][C] (F1 = (n_mels+1)/2, T1 = (lp+1)/2).  w [C][9] fp32, b [C] fp32.
void conv1_gelu(const float* mel, int n_mels, int n_frames, int c0, int nc, int lp, const float* w, const float* b, int C, bf16* out,
                cudaStream_t st);
// im2col for a 3x3 / stride 2 / pad 1 conv over channel-last input in [nc][Fin][Tin][C] (bf16):
//   col[row][(kh*3 + kw)*C + c] = in[n][2*fo + kh - 1][2*to + kw - 1][c] (0 outside), Fo = (Fin+1)/2, To = (Tin+1)/2.
//   time_major = false: row = (n*Fo + fo)*To + to;  true: row = (n*To + to)*Fo + fo.
void im2col_s2(const bf16* in, int nc, int Fin, int Tin, int C, bool time_major, bf16* col, cudaStream_t st);
// h[tok0 + n*13-ish + t] = in[(n*T3 + t)] + pos[t] for t < valid(n): chunks are full (13 tokens) except possibly the last one, whose
//   valid length is `last_valid`. in fp32 [nc*T3, d]; pos fp32 [>= T3, d]. Writes nc_full*T3... rows contiguous from h.
void add_pos_gather(const float* in, const float* pos, int nc, int T3, int last_valid, int d, float* h, cudaStream_t st);

}  // namespace hive::qasr
