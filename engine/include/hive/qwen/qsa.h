// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Full-attention layers of Qwen3.8-Flash-Next: Q/K prep (per-head (1 + w) RMS norm, interleaved M-RoPE), the KV cache, the QSA indexer
//   (4-token groups, mean-pooled and normed keys, ReLU-summed multi-head scores, top 512 groups) and the sparse GQA attention with
//   the sigmoid output gate.
// Math contract (per query token at sequence position p, rope coordinates r = (rt, rh, rw)):
//   · q_h = rope(norm_q(q_h)), k_g = rope(norm_k(k_g)) (head 256, rotary on dims 0..63, NeoX halves 0..31 / 32..63; pair i < 32 uses
//     frequency θ^(−2i/64) of axis a(i): a = 1 if i % 3 == 1 and i < 3·sec[1], a = 2 if i % 3 == 2 and i < 3·sec[2], else 0).
//   · indexer: iq_h = rope(norm_iq(iq_h)) (4 heads × 128, the same rope on dims 0..63); the raw key ik (128) of every token is kept.
//     Group G = tokens [4G, 4G+3] is complete once token 4G+3 is processed; its key = rope_at(r of token 4G)(norm_ik(bf16(mean_fp32 of
//     the 4 raw keys))). score_G = Σ_h relu(iq_h · key_G) / sqrt(128) (fp32) over the visible groups G < (p+1)/4.
//   · selection: the top min(512, visible) groups (ties: implementation-defined) expanded to their 4 tokens, plus the open group's
//     tokens [4·floor((p+1)/4), p]. With ≤ 512 visible groups that is every token ≤ p (dense attention).
//   · out_h = Σ_j softmax_j(q_h · k_{g(h)}(j) / 16) v_{g(h)}(j) over the selected j (fp32), g(h) = h / 12; then out_h ·= sigmoid(gate_h);
//     the result is bf16 [24·256] per token (o_proj input).
// Positions vs rope coordinates: p is the token's index in the sequence (causality, grouping); r are its M-RoPE coordinates (equal to
//   (p', p', p') for text, p' = p plus the sequence's image offset — hive/qwen/qwen_engine.h).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::qwen {

constexpr int kQsaHeads = 24, kQsaKvHeads = 2, kQsaDim = 256, kQsaRot = 64;
constexpr int kQsaIdxHeads = 4, kQsaIdxDim = 128, kQsaRatio = 4, kQsaTopBlocks = 512;
constexpr int kQsaSel = kQsaTopBlocks * kQsaRatio + kQsaRatio - 1;   // 2051 selected positions per query (−1 padded)
// attention projection row layout (engine fused GEMM output, row stride ld)
constexpr int kQsaQGW = 2 * kQsaHeads * kQsaDim;                      // 12288: per head [q 256 | gate 256]
constexpr int kQsaKOff = kQsaQGW, kQsaVOff = kQsaKOff + kQsaKvHeads * kQsaDim, kQsaIqOff = kQsaVOff + kQsaKvHeads * kQsaDim;
constexpr int kQsaIkOff = kQsaIqOff + kQsaIdxHeads * kQsaIdxDim;
constexpr int kQsaProjWidth = kQsaIkOff + kQsaIdxDim;                 // 13952

// Rope tables: inv_freq fp32 [32] and the axis of each pair int32 [32] (qsa_rope_tables).
struct QsaRope { const float* inv_freq; const int32_t* axis; };
void qsa_rope_tables(float theta, const int* section, bool interleaved, float* inv_freq_dev, int32_t* axis_dev, cudaStream_t st);

// One sequence's cache for one full-attention layer (device memory, sized for cap tokens).
struct QsaCache {
  __nv_bfloat16* k;      // [cap, 2, 256] normed + roped keys
  __nv_bfloat16* v;      // [cap, 2, 256]
  __nv_bfloat16* ik;     // [cap, 128] raw indexer keys
  __nv_bfloat16* gk;     // [cap/4 + 1, 128] group keys (normed + roped), row G valid once group G is complete
  const int32_t* rope;   // [cap, 3] rope coordinates of every processed token (shared by all layers of the sequence)
};
// One query row: its sequence's cache and its position p (the row's token is written at p).
struct QsaRow { QsaCache c; int pos; };

struct QsaWeights { const __nv_bfloat16* q_norm; const __nv_bfloat16* k_norm; const __nv_bfloat16* iq_norm; const __nv_bfloat16* ik_norm; float eps; };

// Prep for n rows (rows: device-readable array of n QsaRow; each row's rope coordinates must already be in its cache at rope[pos]):
//   from proj bf16 [n, ld] writes q bf16 [n, 24, 256] (normed + roped), iq bf16 [n, 4, 128] (normed + roped), and into each row's cache
//   k[pos], v[pos], ik[pos]; then completes every group whose last token is one of these rows (gk). Rows of the same sequence must be in
//   ascending position order within the call (prefill chunks, verify rows); groups whose earlier tokens came from an earlier call are read
//   from the cache.
void qsa_prep(const QsaWeights& w, const QsaRope& rope, const QsaRow* rows, int n, const __nv_bfloat16* proj, int ld, __nv_bfloat16* q,
              __nv_bfloat16* iq, cudaStream_t st);
// Selection for n rows → idx int32 [n, 2051] (−1 padded; any order). ws ≥ qsa_select_ws_bytes(n, max_pos + 1); rows whose visible groups
//   number ≤ 512 skip the scoring. Rows are processed in sub-blocks that fit the workspace (≥ qsa_select_ws_bytes(1, ctx) is enough).
size_t qsa_select_ws_bytes(int n, int ctx);
//   ctx_cap > max_pos + 1 sizes the score layout for a longer context (the kernels bound every row by its own position), so a captured CUDA
//   graph stays valid while the positions grow within the cap; 0 = max_pos + 1.
void qsa_select(const QsaRow* rows, int n, int max_pos, const __nv_bfloat16* iq, int32_t* idx, void* ws, size_t ws_bytes, cudaStream_t st,
                int ctx_cap = 0, bool one_seq = false);   // one_seq: every row belongs to one sequence (prefill chunk) — tensor-core scoring
// Attention for n rows: q [n, 24, 256], gate = proj's gate halves (row stride ld), idx [n, 2051] → out bf16 [n, 6144].
void qsa_attention(const QsaRow* rows, int n, const __nv_bfloat16* q, const __nv_bfloat16* proj, int ld, const int32_t* idx,
                   __nv_bfloat16* out, cudaStream_t st);

// Prefill rows of ONE sequence (consecutive positions, ascending), n > 64: several rows per block over the union of their selected groups
//   (qsa.cu qsa_attn_mr_kernel) — the same math as qsa_attention, fp32 sums in another order. max_pos = the largest row position.
//   Returns false (nothing launched) unless switched on (HIVE_QWEN_QSA_MR=1), n ≤ 64 or the workspace is too small; the caller then runs
//   qsa_attention. ws ≥ qsa_attention_mr_ws_bytes(n).
size_t qsa_attention_mr_ws_bytes(int n);
bool qsa_attention_mr(const QsaRow* rows, int n, int max_pos, const __nv_bfloat16* q, const __nv_bfloat16* proj, int ld, const int32_t* idx,
                      __nv_bfloat16* out, void* ws, size_t ws_bytes, cudaStream_t st);

// Decode / verify rows: the tensor-core attention with the selected positions split over `splits` blocks per (row, kv head) (2..32) and a
//   fixed-order combine — a 4-row verify otherwise runs on 8 blocks. ws ≥ qsa_attention_split_ws_bytes(n, splits).
size_t qsa_attention_split_ws_bytes(int n, int splits);
void qsa_attention_split(const QsaRow* rows, int n, const __nv_bfloat16* q, const __nv_bfloat16* proj, int ld, const int32_t* idx,
                         __nv_bfloat16* out, int splits, void* ws, cudaStream_t st);

}  // namespace hive::qwen
