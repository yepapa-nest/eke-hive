// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma 4 12B (text path) GPU kernels (sm_120a). The math contract is oracle/gemma4_oracle.py; the arithmetic is fp32 between the
//   points where the engine rounds to bf16 (activations between ops, the KV cache in fp8 e4m3).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::gemma::k {

using bf16 = __nv_bfloat16;

// Weight matrix [N, K], int4 group-32 (checkpoint: compressed-tensors pack-quantized — int32 [N, K / 8], 8 values per word, low nibble
//   first, value = nibble - 8; bf16 scale [N, K / 32]; W[n, k] = (nibble - 8) · s[n, k / 32]), held on the GPU in the decode GEMV's
//   fragment order (w4_repack): q = words [N / 16 tiles][K / 64 group pairs][32 lanes][4] — lane (g = lane / 4, c = lane % 4) holds, for
//   each group of the pair and each k16 half, one word with the nibble pairs of its mma A fragment (a0..a3 = rows g / g+8 at k 2c / 2c+8,
//   pair p's first value at bits 4p, second at bits 16 + 4p); s = bf16 pairs [N / 16][K / 32][8] = {s[row g], s[row g + 8]}.
struct W4Mat {
  const uint32_t* q = nullptr;
  const uint32_t* s = nullptr;
  int N = 0, K = 0;   // K % 64 == 0, N % 16 == 0
  int split = 1;      // warps per 16-row tile in the decode GEMV (their K split; a function of K only — never of the row count, so rows
                      //   stay batch-invariant)
};
// up to 3 matrices with the same K and the same input in one decode GEMV launch (e.g. q / k / v): output i = yb[i] (bf16) or yf[i] (fp32)
struct W4Multi {
  W4Mat W[3];
  bf16* yb[3] = {};
  float* yf[3] = {};
  int ldy[3] = {};
  int n = 0;
};
// checkpoint layout (q_ck int32 [N, K / 8], s_ck bf16 [N, K / 32], device) → fragment order (q: N · K / 8 words, s: N · K / 32 · 2 bytes)
void w4_repack(const uint32_t* q_ck, const bf16* s_ck, int N, int K, uint32_t* q, uint32_t* s, cudaStream_t st);
// split for a matrix: enough warps in flight on narrow matrices (N small) — a function of the shape only
int w4_split_for(int N, int K);

// Decode / verify GEMV, M ≤ 16 rows: y[m, n] = Σ_g s[n, g] · Σ_{k in g} (nibble - 8) · x[m, k], fp32 accumulation, written as bf16
//   (rounded) to yb or fp32 to yf (exactly one non-null). x bf16 [M, K] (row stride ldx). Each row's result does not depend on M.
//   ws: unused (kept for the call sites).
void w4_gemv(const W4Mat& W, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, float* ws, cudaStream_t st);
void w4_gemv_multi(const W4Multi& J, const bf16* x, int ldx, int M, cudaStream_t st);
// MLP front, M ≤ 16: act[m, n] = bf16(gelu_tanh(g) · u), g / u = rows n of G / U (fp32 sums, same as two w4_gemv calls in fp32)
void w4_gemv_gu(const W4Mat& G, const W4Mat& U, const bf16* x, int ldx, int M, bf16* act, int lda, float* ws, cudaStream_t st);
// bytes of fp32 scratch the two calls above need for M rows (max over the model's matrices is the caller's)
inline size_t w4_ws_bytes(const W4Mat&, int) { return 0; }   // the K split is summed inside the block now (no scratch)

// Prefill: the weight as bf16 [N, K] (bf16((nibble - 8) · s) — rounded once), for a cuBLAS GEMM with fp32 accumulation.
void w4_dequant(const W4Mat& W, bf16* out, cudaStream_t st);

// out = bf16(x · (mean(x²) + eps)^-0.5 · w) over rows of K (w null: no scale)
void rmsnorm(const bf16* x, int ldx, const bf16* w, int M, int K, float eps, bf16* out, int ldo, cudaStream_t st);
//   (K % 8 == 0, K ≤ 8192, row strides multiples of 8 — rows are read as 16-byte vectors)
// h = bf16(h + bf16(rmsnorm(y) · w)) — the post-attention / post-feedforward residual add; optionally then h *= scalar (bf16); with wn,
//   also out = rmsnorm(h) · wn (the norm that reads the new residual, fused: pre-feedforward or the next layer's input norm)
void add_postnorm(bf16* h, const bf16* y, int ldy, const bf16* w, int M, int K, float eps, const bf16* scalar, cudaStream_t st,
                  const bf16* wn = nullptr, bf16* out = nullptr, int ldo = 0);
// act = bf16(gelu_tanh(g) · u) over [M, I] (prefill path; g / u fp32 or bf16 rows from the GEMM)
void gelu_mul(const bf16* g, int ldg, const bf16* u, int ldu, int M, int I, bf16* act, int lda, cudaStream_t st);
// embedding rows: out[m, :K] = bf16(E[ids[m]] · scale) (E bf16 [vocab, K]; out row stride ldo)
void embed(const bf16* E, const int32_t* ids, int M, int K, float scale, bf16* out, int ldo, cudaStream_t st);
// per head (heads of D values in each row of x, row stride ldx): y = rmsnorm(x) · w (w null: no scale), then the rotate-half RoPE at
//   pos[m] with inv_freq [D / 2] (null: no RoPE; zero entries = unrotated pairs, the "proportional" partial RoPE), out bf16 (row stride ldo)
void head_norm_rope(const bf16* x, int ldx, int M, int heads, int D, const bf16* w, float eps, const float* inv_freq, const int32_t* pos,
                    bf16* out, int ldo, cudaStream_t st);
// the same for up to 3 inputs of the same head size in one launch (q / k / v of a layer)
struct HnrPart { const bf16* x; int ldx, heads; const bf16* w; const float* inv_freq; bf16* out; int ldo; };
struct HnrParts { HnrPart p[3]; int n = 0; };
void head_norm_rope(const HnrParts& P, int M, int D, float eps, const int32_t* pos, cudaStream_t st);
// per row of M logits rows of V: arg = argmax (ties → the smaller id), sumexp = Σ exp(l - max) (p(argmax) = 1 / sumexp); part: scratch of
//   argmax_sumexp_part_floats(M) floats
inline size_t argmax_sumexp_part_floats(int M) { return (size_t)M * 64 * 3; }
void argmax_sumexp(const float* logits, int M, int V, float* part, int32_t* arg, float* sumexp, cudaStream_t st);
// logits in place: l = cap · tanh(l / cap) over n values (cap > 0)
void softcap(float* l, size_t n, float cap, cudaStream_t st);


// ---- Paged fp8 KV cache and attention ------------------------------------------------------------------------------------------------
// The KV of one layer type (sliding / full) lives in pages of kPage tokens; one page holds every layer of that type:
//   page = [token (kPage)][layer (n_layers)][K | V][kv head (Hkv)][D] as fp8 e4m3 (scale 1.0, saturating) — token-major, so the filled
//   part of a page is always a contiguous prefix (session images copy only what grew since the base image) and a token's K of one head
//   is D contiguous bytes (rows of one layer are token_bytes() apart). A sequence's page table holds page base *pointers* (device memory, or mapped pinned host memory when VRAM is
//   short — the kernels read both the same way). Position p of a sequence lives in page p / kPage, slot p % kPage. Sliding-layer pages
//   older than every key a query can still see may be released (their table entries become null and are never read).
constexpr int kPage = 64;
struct KvLayout {
  int n_layers = 0;   // layers of this type
  int Hkv = 0, D = 0;
  size_t token_bytes() const { return (size_t)n_layers * 2 * Hkv * D; }
  size_t page_bytes() const { return token_bytes() * kPage; }
  // byte offset of (slot, layer, K/V, head 0) inside a page
  size_t k_off(int slot, int layer) const { return (size_t)slot * token_bytes() + (size_t)layer * 2 * Hkv * D; }
  size_t v_off(int slot, int layer) const { return k_off(slot, layer) + (size_t)Hkv * D; }
};
// One sequence's slice of an attention call: query rows [q_row0, q_row0 + n_q) at positions q_pos0 .. q_pos0 + n_q - 1, keys 0 ..
//   q_pos0 + n_q - 1 (causal; the new rows' K/V are already written), window > 0: key j visible to query position i iff i - window < j ≤ i.
struct AttnSeq {
  const uint8_t* const* pages = nullptr;   // device array of page pointers
  int q_row0 = 0, n_q = 0, q_pos0 = 0;
};
// out[r, h, :] = softmax_j(q[r, h, :] · K[j, h / (H / Hkv), :]) · V[j, h / (H / Hkv), :]  (scaling 1.0 — q / k are RMS-normalized),
//   fp32 softmax and accumulation, K / V read as fp8 → fp32. q / out bf16 [rows, H, D]. seqs: device array of n_seqs AttnSeq.
//   D ∈ {256, 512}; H / Hkv ∈ {2, 16}. Rows of one call may mix decode (n_q = 1), verify (n_q ≤ 16) and prefill chunks.
void attn_paged(const bf16* q, bf16* out, int H, const KvLayout& L, int layer, const AttnSeq* seqs, int n_seqs, int max_q_per_seq, int window,
                cudaStream_t st);
// write the new rows' K / V (bf16 [rows, Hkv, D], already normed / rotated) into the pages at positions pos[r] of sequence seq_of[r]
//   (seq_of / pos: device arrays)
void kv_write(const bf16* k, const bf16* v, int rows, const int32_t* seq_of, const int32_t* pos, const AttnSeq* seqs, const KvLayout& L, int layer,
              cudaStream_t st);

}  // namespace hive::gemma::k
