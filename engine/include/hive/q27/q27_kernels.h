// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B GPU kernels (sm_120a). The math contract is hive/q27/q27_model.h (oracle/q27_oracle.py); every point the reference keeps in
//   bf16 is rounded to bf16 here at the same place, the arithmetic in between is fp32. Written for correctness first (simple work splits);
//   the decode GEMVs read every weight byte once per step and are the part that sets decode speed.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::q27::k {

using bf16 = __nv_bfloat16;

// One weight matrix [N, K] in VRAM in its checkpoint format.
enum class WFmt : int { BF16 = 0, FP8ROW = 1, NVFP4 = 2 };
struct GMat {
  WFmt fmt = WFmt::BF16;
  const uint8_t* w = nullptr;   // BF16 [N, K] · FP8 e4m3 [N, K] · NVFP4 e2m1 pairs [N, K/2] (low nibble = even column)
  const void* s = nullptr;      // FP8: fp32 [N] per-row scale · NVFP4: e4m3 [N, K/16]
  float g = 1.f;                // NVFP4: factor applied to the sum (1 / weight_global_scale)
  int N = 0, K = 0;
  const uint8_t* s_swz = nullptr;   // NVFP4, low-precision prefill: the block scales in cuBLASLt's tiled layout (hive/q27/q27_lowp.h)
  float in_gs = 0.f;                // NVFP4: the checkpoint's input_global_scale (activation quantization of the low-precision prefill)
  // HIVE_Q27_W4=1 (lossy, off by default): an NVFP4 copy of an FP8 matrix used by the decode / verify GEMVs (≤ 16 rows); prefill keeps W
  const uint8_t* w4 = nullptr;      // e2m1 pairs [N, K/2]
  const uint8_t* s4 = nullptr;      // e4m3 [N, K/16]
  float g4 = 1.f;                   // 1 / global scale
  const uint8_t* s4_swz = nullptr;  // the same scales in cuBLASLt's tiled layout (W4A4 prefill of the copy, dynamic activation scale)
  // decode / verify activation policy of this use (NInfer-style per-use table): 0 = bf16 activations (A16); 4 = NVFP4 activations with the
  //   W4A4 block-scaled mma for every row count (HIVE_Q27_A4_MLP / HIVE_Q27_A4_PROJ; lossy — quality gate)
  int act = 0;
  const uint8_t* a4w = nullptr;   // act == 4: codes repacked in mma fragment order [N/16][K/64][32 lanes][16 bytes]
  const uint8_t* a4s = nullptr;   //           scales [N/16][K/64][16 rows][4 bytes]
};
// the matrix the decode GEMVs read: the NVFP4 copy when there is one, else W
inline GMat decode_view(const GMat& W) {
  if (!W.w4) return W;
  GMat v = W;
  v.fmt = WFmt::NVFP4; v.w = W.w4; v.s = W.s4; v.g = W.g4; v.s_swz = nullptr; v.in_gs = 0.f; v.w4 = nullptr;
  return v;
}
// NVFP4 copy of an FP8 row-scaled matrix (global scale from max |W|, e4m3 scale per 16 values, e2m1 values, round to nearest); amax: fp32
//   device scratch (one float). Returns g4 (synchronizes the stream).
float requant_nvfp4(const GMat& W, uint8_t* w4, uint8_t* s4, float* amax, cudaStream_t st);

// Decode GEMV, M ≤ 8 rows: y[m, n] = Σ_k x[m, k] · W[n, k] (fp32 accumulation, scales applied), written as bf16 (rounded) to yb or fp32 to yf
//   (exactly one non-null). x bf16 [M, K] (row stride ldx).
void gemv(const GMat& W, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st);
// scratch of the split-N GEMV (HIVE_Q27_SW=20) — call once before any graph capture
// drafter rows 33..64 in one pass (false: not handled — the caller splits into ≤ 32-row passes)
bool gemv_draft(const GMat& W, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st);
void gemv_prepare();
// W4A4 decode / verify GEMV (GMat.act == 4, NVFP4 view): the M rows of x are quantized to NVFP4 per row (dynamic global scale, e4m3 scale
//   per 16 values, e2m1) and multiplied with mma m16n8k64 kind::mxf4nvf4 block_scale. Returns false when it does not apply.
// repack an NVFP4 matrix (codes / scales of W) for the W4A4 route; dst: (N/16 rounded up) · K/2 bytes of codes, then the scales
//   (N/16 rounded up) · K/16 bytes. Returns the byte count when dst is null.
size_t a4_repack(const GMat& W, uint8_t* dst, cudaStream_t st);
bool gemv_a4(const GMat& W, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st);
bool gemv_gu_a4(const GMat& G, const GMat& U, const bf16* x, int ldx, int M, bf16* act, int lda, cudaStream_t st);
// Fused MLP front for M ≤ 8: act[m, n] = bf16(bf16(silu(bf16(g))) · bf16(u)), g / u = rows n of G / U (same K and format) — the same
//   values as two gemv calls + silu_mul.
void gemv_gu(const GMat& G, const GMat& U, const bf16* x, int ldx, int M, bf16* act, int lda, cudaStream_t st);
// h = bf16(h + y) then out = rmsnorm1p(h) (whole rows of K) — add_round + rmsnorm1p in one kernel, same values
void add_norm(bf16* h, const bf16* y, const bf16* w, int M, int K, float eps, bf16* out, cudaStream_t st);
// Prefill helpers: the exact pre-scale weight as bf16 [N, K] (FP8 value / e2m1 · s16 — both exact in bf16), and the epilogue
//   y[m, n] = round_bf16(acc[m, n] · scale(n)) (scale = 1, the row scale, or g) from the fp32 GEMM result acc [M, N] (row stride lda).
void dequant_pre(const GMat& W, bf16* out, cudaStream_t st);
void epilogue(const GMat& W, const float* acc, int lda, int M, bf16* yb, float* yf, int ldy, cudaStream_t st);

// out = bf16(x · rsqrt(mean_group(x²) + eps) · (1 + w)) over groups of `group` (group | K); x / out bf16 rows
void rmsnorm1p(const bf16* x, int ldx, const bf16* w, int M, int K, int group, float eps, bf16* out, int ldo, cudaStream_t st);
// h = bf16(h + y)
void add_round(bf16* h, const bf16* y, size_t n, cudaStream_t st);
// MLP activation in place over gu = [gate | up] rows (row stride ld, I columns each): g = bf16(bf16(silu(g)) · u)
void silu_mul(bf16* g, int ldg, const bf16* u, int ldu, int M, int I, cudaStream_t st);
// out[m, h·D + d] = bf16(bf16(att) · sigmoid(gate)); att fp32 [M, H, D], gate = proj[m · ld + h · 2D + D + d]
void attn_gate(const float* att, const bf16* proj, int ld, int M, int H, int D, bf16* out, cudaStream_t st);

// ---- full attention ----------------------------------------------------------------------------------------------------------------------
struct AttnRow { bf16* k; bf16* v; int pos; int n_ctx; };   // row's KV cache (layer, sequence) [cap, Hkv·D] · its position · keys visible
// Per row: q (per head (1 + w) norm + rope) → q_out bf16 [M, H, D]; k (norm + rope) and v → cache row pos. proj rows [q|gate per head ·
//   k · v] (stride ld); coords int32 [M, 3]; inv_freq fp64 [rot/2] and axis int32 [rot/2] (interleaved M-RoPE). kv_only: no q.
void attn_prep(const AttnRow* rows, int M, const bf16* proj, int ld, const bf16* q_norm, const bf16* k_norm, float eps, const int32_t* coords,
               const double* inv_freq, const int32_t* axis, int rot, int H, int Hkv, int D, bf16* q_out, bool kv_only, cudaStream_t st);
// cos / sin of every (position < n, frequency) pair, float2 [n, rot/2] in VRAM, used by attn_prep for coordinates inside it (the same
//   values it computes otherwise: (float) of the fp64 sincos)
void set_rope_table(const float2* tab, int n);
// Causal GQA attention per row over keys [0, n_ctx): out fp32 [M, H, D]
void attention(const AttnRow* rows, int M, const bf16* q, int H, int Hkv, int D, float scale, float* out, cudaStream_t st);
// split-key decode attention (HIVE_Q27_ATTN2): device scratch for its partial results, sized for `rows` rows (attention_scratch_bytes)
void set_attention_scratch(float* p, int rows);
size_t attention_scratch_bytes(int rows, int Hkv);

// Prefill attention (tensor cores, mma m16n8k16 bf16): M query rows of ONE sequence at consecutive positions pos0 … pos0 + M − 1 (causal),
//   keys / values of that sequence's cache [0, pos0 + M). Scores fp32; probabilities rounded to bf16 for the P·V product (fp32 accumulate);
//   online softmax per 32-key tile. out fp32 [M, H, D]. D = 256.
void attention_prefill(const bf16* q, int M, int pos0, const bf16* K, const bf16* V, int H, int Hkv, int D, float scale, float* out, cudaStream_t st);

// ---- Gated DeltaNet ----------------------------------------------------------------------------------------------------------------------
// A segment = consecutive rows [row0, row0 + n) of one sequence: its conv state fp32 [3, C] and recurrent state fp32 [VH, 128, 128]; with
//   snapshots (verify), the conv state after row t < n − 1 is copied to snap_conv[t] and the recurrent state before row 0 to snap_state[0].
// keep_state = 1 (verify without a state snapshot): the recurrence reads `state` but does not write it back — the rollback replays the
//   kept rows from that untouched state in place (gdn_replay with init == state)
struct GdnSeg { int row0, n; float* conv; float* state; float* const* snap_conv; float* const* snap_state; int keep_state = 0; };
// rollback replay item: rows [row0, row0 + n) of a verify's saved GDN inputs (q / k, gate / beta, v — that layer's buffers) applied to the
//   state `init` (the verify's snapshot, or the untouched state itself: init == state, in place), written to `state`; the same row
//   arithmetic as the forward
struct GdnReplay { const float* qk; const float* gb; const bf16* y; int row0, n; const float* init; float* state; };
void gdn_replay(const GdnReplay* items, int n, int KH, int VH, int C, cudaStream_t st);
// many small device copies in one launch (rollback conv states): item i copies n floats src → dst
struct CopyItem { const float* src; float* dst; int n; };
void multi_copy(const CopyItem* items, int n, cudaStream_t st);
// many device copies of arbitrary byte sizes in one launch: chunk i copies n bytes src → dst (≤ 64 KiB each; 16-byte pieces when all three
//   are 16-byte aligned, else bytes)
struct ByteChunk { const uint8_t* src; uint8_t* dst; unsigned n; };
void copy_chunks(const ByteChunk* chunks, int n, cudaStream_t st);
struct GdnW { const float* conv_w; const float* A_log; const float* dt_bias; const bf16* norm; float eps; };
// proj bf16 [rows, ld] = [qkv C | z VW | b VH | a VH]; out bf16 [rows, VW]. Work buffers: y bf16 [rows, C], qk fp32 [rows, 2·KW],
//   gb fp32 [rows, 2·VH], o fp32 [rows, VW].
void gdn(const GdnW& w, const GdnSeg* segs, int n_segs, int rows, const bf16* proj, int ld, int KH, int VH, bf16* y, float* qk, float* gb,
         float* o, bf16* out, cudaStream_t st);
// the chunked prefill workspaces of gdn() for up to `rows` rows, allocated now (load time) instead of on the first long chunk
void gdn_reserve(int rows, int KH, int VH);

// argmax of one fp32 row (ties → smaller index) and its softmax probability 1 / Σ exp(l − max) → out {id (int32 bits), p}; out holds
//   2 + 3·64 floats (work area after the result)
void argmax_prob(const float* logits, int V, float* out, cudaStream_t st);
// out[d] = table[ids[*id_idx] (or *id_ptr when ids == nullptr)][d] — one embedding row picked by a device-side token id
void embed_by_id(const bf16* table, const int32_t* id_ptr, int dim, bf16* out, cudaStream_t st);
// out[r, :] = table[ids[r], :] for M rows (ids device), bf16 rows of width dim
void embed_rows(const bf16* table, const int32_t* ids, int M, int dim, bf16* out, cudaStream_t st);
// *id = map[*id] (draft vocabulary index → token id), on the device
void map_ids(int32_t* id, const int32_t* map, cudaStream_t st);
// rows of bf16 → fp32 and back (CPU-placed layers)
void bf16_to_f32(const bf16* x, size_t n, float* y, cudaStream_t st);
void f32_to_bf16(const float* x, size_t n, bf16* y, cudaStream_t st);

}  // namespace hive::q27::k
