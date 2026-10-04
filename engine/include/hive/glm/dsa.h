// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash DSA layer core: MLA without positions (NoPE) + pooled lightning indexer ("k-pool") + sparse attention.
//
// Reference: transformers 5.18 modeling_glm5_next.py — Glm5NextTextAttention and Glm5NextTextIndexer (verified line by line):
//  · 64 heads, q = q_b_proj(RMSNorm1536(q_a_proj(x))) → [64, 256] (qk_nope 256, qk_rope 0). scaling = 256^-0.5.
//  · latent c = RMSNorm512(kv_a_proj_with_mqa(x)) → 512 per token; the cache stores only c.
//  · kv_b_proj [32768, 512]: rows of head h are [h·512, h·512+256) = k_nope, [h·512+256, (h+1)·512) = v
//    (expand_kv: view(B, S, 64, 512) then split [256, 256]).
//  · absorbed core: q_lat[h] = W_UK[h]ᵀ q[h] (512); s_j = q_lat[h]·c_j · scaling; out_lat[h] = Σ softmax(s)_j c_j;
//    out[h] = W_UV[h] out_lat[h] (256). Output [T, 64·256] (head-major per token) is the o_proj input.
//  · indexer (index_kpool = 4, index_topk = 2048 → 512 pools, 32 heads × 128):
//      K_p[ch] = Σ_{j<4} bf16( bf16(softmax_j(gs[4p+j, ch] + ape[j, ch])) · kI[4p+j, ch] )  (fp32 sum, bf16 result)
//      score_p = Σ_h (wI_h · 32^-0.5) · relu(qI_h · K_p · 128^-0.5)  (fp32)
//    candidates = complete pools whose last token is ≤ the query position; top min(512, #candidates) are expanded to their
//    4 tokens, then the tail rule appends the query's incomplete pool: tokens [pos+1 - (pos+1)%4, pos]. When (pos+1)%4 == 0
//    the query's own pool is complete and is an ordinary candidate (the own token is NOT forced in). Contexts with
//    ≤ 512 visible pools therefore attend densely (all tokens ≤ pos).
//
// Cache dtype choice: kI and gs are kept in bf16 — that is exactly what the reference stores (the bf16 module produces
// k_norm(wk(x)) and F.linear(x, gate) in bf16 and caches them packed), so pooled keys are bit-compatible; fp32 would cost
// 2× memory for no fidelity gain. Pooled keys are bf16 too (the reference casts the softmax probabilities and the pooled sum
// to the key dtype).
//
// Index layout produced by dsa_select: per query 2051 int32 (= 2048 + kpool-1), ascending token ids, -1 padded. The
// attention kernels accept any layout with unique entries and -1 holes (e.g. the reference's score-ordered layout).
#pragma once
#include <cstddef>
#include <cstdint>

#include "hive/common.h"

namespace hive::glm {

constexpr int kDsaHeads = 64, kDsaLat = 512, kDsaQkDim = 256, kDsaVDim = 256;
constexpr int kDsaIdxHeads = 32, kDsaIdxDim = 128, kDsaPool = 4, kDsaTopk = 2048;
constexpr int kDsaTopPools = kDsaTopk / kDsaPool;          // 512
constexpr int kDsaIdxWidth = kDsaTopk + kDsaPool - 1;      // 2051

struct DsaLayerW {
  const bf16* kv_b;  // [32768, 512] (kv_b_proj.weight)
  const bf16* ape;   // [4, 128]    (index_kpool_compress_ape)
};

// Device buffers of one sequence for one DSA layer, sized for cap tokens (cap multiple of 4 recommended).
struct DsaSeqCache {
  bf16* c;       // [cap, 512]  latent (kv_a_layernorm output)
  bf16* kI;      // [cap, 128]  indexer key (LayerNorm(wk(x)))
  bf16* gs;      // [cap, 128]  pool gate scores (x @ index_kpool_compress_gate.T)
  bf16* pooled;  // [cap/4, 128] pooled keys, row p valid once tokens 4p..4p+3 are appended
  int cap;
};

// One decode row (sequence) of a batched decode step. pos = absolute position of this step's token.
// Arrays of rows must live in device memory (graph-capturable).
struct DsaDecodeRow {
  bf16* c;
  bf16* kI;
  bf16* gs;
  bf16* pooled;
  int pos;
};

// ---------------- prefill / single-sequence path (T new tokens at positions [pos0, pos0+T)) ----------------
// Writes c/kI/gs rows pos0..pos0+T-1 and computes the pooled keys of every pool completed by them (the pool may begin in an
// earlier append).
void dsa_append(const DsaLayerW& w, DsaSeqCache& cache, int pos0, int T, const bf16* c_new, const bf16* kI_new,
                const bf16* gs_new, cudaStream_t st);

// qI [T, 32, 128] bf16 (wq_b(q_resid)), wI [T, 32] fp32 = weights_proj(x) WITHOUT the 32^-0.5 factor (applied here).
// idx [T, 2051] int32. Requires dsa_append of these tokens first.
size_t dsa_select_ws_bytes(int T, int ctx);  // ctx = pos0 + T
void dsa_select(const DsaSeqCache& cache, int pos0, int T, const bf16* qI, const float* wI, int32_t* idx, void* ws,
                size_t ws_bytes, cudaStream_t st);

// q [T, 64, 256] bf16 → out [T, 64·256] bf16. The workspace may be smaller than dsa_attention_ws_bytes(T) as long as it holds
// dsa_attention_ws_bytes(1): queries are then processed in chunks.
size_t dsa_attention_ws_bytes(int T);
void dsa_attention(const DsaLayerW& w, const DsaSeqCache& cache, int pos0, int T, const bf16* q, const int32_t* idx,
                   bf16* out, void* ws, size_t ws_bytes, cudaStream_t st);

// ---------------- batched decode: M sequences, one token each ----------------
// rows: device array [M]. c_new [M,512], kI_new [M,128], gs_new [M,128]; row r is written at rows[r].pos.
void dsa_append_decode(const DsaLayerW& w, const DsaDecodeRow* rows, int M, const bf16* c_new, const bf16* kI_new,
                       const bf16* gs_new, cudaStream_t st);
// max_ctx ≥ max(rows[r].pos)+1 (sizes the score buffer).
size_t dsa_select_decode_ws_bytes(int M, int max_ctx);
void dsa_select_decode(const DsaDecodeRow* rows, int M, int max_ctx, const bf16* qI, const float* wI, int32_t* idx, void* ws,
                       size_t ws_bytes, cudaStream_t st);
size_t dsa_attention_decode_ws_bytes(int M);
void dsa_attention_decode(const DsaLayerW& w, const DsaDecodeRow* rows, int M, const bf16* q, const int32_t* idx, bf16* out,
                          void* ws, size_t ws_bytes, cudaStream_t st);

// Optional: use the caller's cuBLAS handle (default: one lazily created handle per host thread). The attention entry points
// call cuBLAS (two strided-batched GEMMs over the 64 heads); for CUDA-graph capture pass a handle that has a workspace set
// (cublasSetWorkspace). Everything else is plain kernels without host synchronisation.
void dsa_set_cublas_handle(void* cublas_handle);

}  // namespace hive::glm
