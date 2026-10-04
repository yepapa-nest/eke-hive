// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Model kernels — each operation of the reference model.py ported one by one. Batches are flattened into the row (M) dimension.
#pragma once
#include <cstdint>

#include "hive/common.h"
#include "hive/kv_pack.h"

namespace hive::k {

// ---- block-scale expansion (at load): [N/32, K/32] → [N, K/32] (32 rows share a scale) --------------------------------
void expand_block_scale(const uint8_t* src, int N, int K, uint8_t* dst, cudaStream_t st);
// 32×32-block fp8 → bf16 dequantization (for wo_a; what convert.py used to do)
void dequant_fp8_block_to_bf16(const uint8_t* w, const uint8_t* s, int N, int K, bf16* out, cudaStream_t st);
// fp8 with per-row expanded scales ([N, K/32], output of load_fp8) → bf16 (values identical to the reference bf16 dequantization). K % 8 == 0
void dequant_fp8_rows_to_bf16(const uint8_t* w, const uint8_t* s, int N, int K, bf16* out, cudaStream_t st);
// prefill hc mix coefficients + row rsqrt in one kernel (replaces hc_flatten_f32 + cuBLAS fp32 GEMM + hc_row_rsqrt). mix_hc == 24, hcdim % 2048 == 0
void hc_mix_rows(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, cudaStream_t st);
// the two variants hc_mix_rows chooses between (called directly for tests and A/B): 16 = 16-row blocks sharing a W window in smem (default, hcdim % 512) · 4 = previous 4-row blocks (HIVE_HC_MIX_ROWS4=1)
void hc_mix_rows16(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, cudaStream_t st);
void hc_mix_rows4(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, cudaStream_t st);
// dump only: positions outside the candidates set to -inf (same index_score shape as the keep-mask output)
void mask_by_cand(bf16* score, int M, int T, int bs, const int32_t* cand, int cand_stride, int kb, cudaStream_t st);

// ---- embedding / residual stream -----------------------------------------------------------------------------------
// h[m, c, :] = embed[ids[m]] (replicated for c = 0..hc-1)
void embed_expand(const bf16* embed, const int32_t* ids, int M, int dim, int hc, bf16* h, cudaStream_t st);
// pre_mix init: [M,hc] fp32, only copy 0 is 1
void identity_pre_mix(float* pre, int M, int hc, cudaStream_t st);

// ---- Hyper-Connections ----------------------------------------------------------------------------------------
// mixes = (fn · flatten(h)) · rsqrt(mean(flatten(h)²)+eps) → sinkhorn → pre[M,hc] post[M,hc] comb[M,hc,hc] (fp32)
//   fn [mix_hc, hc·dim] fp32, scale[3], base[mix_hc]. mixes is computed outside with cuBLAS (fp32); the rest happens here.
void hc_row_rsqrt(const bf16* h, int M, int hcdim, float eps, float* rsq, cudaStream_t st);  // rsq[M]
void hc_flatten_f32(const bf16* h, int M, int hcdim, float* out, cudaStream_t st);           // bf16 → fp32 copy (cuBLAS input)
void hc_split_sinkhorn(const float* mixes, const float* rsq, const float* scale, const float* base, int M, int hc, int iters,
                       float eps, float* pre, float* post, float* comb, cudaStream_t st);
// x[m,:] = bf16( Σ_c pre[m,c] · h[m,c,:] )
void hc_pre(const bf16* h, const float* pre, int M, int hc, int dim, bf16* x, cudaStream_t st);
// h[m,i,:] = bf16( post[m,i]·x[m,:] + Σ_j comb[m,j,i]·h[m,j,:] )   (in-place update allowed: rows are read first internally)
void hc_post(const bf16* x, const float* post, const float* comb, int M, int hc, int dim, bf16* h, cudaStream_t st);
// decode hc fusion (M ≤ 8): hc_mix (mix coefficients + row rsqrt, blocks (M, mix_hc+1)) → hc_pre_fused (sinkhorn + hc_pre + rmsnorm, M blocks)
void hc_mix(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, cudaStream_t st);
void hc_pre_norm(const bf16* h, const float* pre_in, int M, int hc, int dim, const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, cudaStream_t st);
// hc_mix + hc_pre_norm in a single launch (blocks are split, same device functions — bit-identical to the two kernels). Enabled in decode with HIVE_HC_FUSE2=1 (off by default — not measured)
void hc_mix_pre_norm(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                     const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, cudaStream_t st);
// HIVE_HC_DECODE_FUSED (env_on, off by default): hc_mix_pre_norm + sinkhorn (the last mix block runs M tokens × 16 lanes — hc_decode_fused.cuh).
//   Output is bit-identical to hc_mix_pre_norm (mixes, rsq, x, xn) + hc_split_sinkhorn (pre_out, post, comb). cnt = one zero-initialized int (the kernel resets it to 0) — launches
//   using the same cnt must not run concurrently. M ≤ 16. No branch stream or fork/join needed (graph-capturable).
void hc_mix_pre_norm_sk(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                        const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, const float* scale, const float* base, int iters, float hc_eps, float* pre_out,
                        float* post, float* comb, int* cnt, cudaStream_t st);
void hc_mix_pre_norm_sk_preload();
// 16-lane parallel version of hc_split_sinkhorn (one block, bit-identical). hc ≠ 4 or M > 16 falls back to the original. Used on the branch stream by HIVE_HC_SINKHORN_PAR (env_on, off by default).
void hc_split_sinkhorn_par(const float* mixes, const float* rsq, const float* scale, const float* base, int M, int hc, int iters, float eps, float* pre,
                           float* post, float* comb, cudaStream_t st);
bool hc_decode_fused_on();   // env_on("HIVE_HC_DECODE_FUSED") — read once
bool hc_sinkhorn_par_on();   // env_on("HIVE_HC_SINKHORN_PAR") — read once
void hc_decode_fused_preload();  // preloads the lazy modules of the kernels above (v1, v2, parallel sinkhorn) — outside capture and the step-graph spin wait, same reason as hs_preload
void hc_pre_fused(const bf16* h, const float* mixes, const float* rsq, const float* scale, const float* base, const float* pre_in, int M, int hc,
                  int dim, int iters, float hc_eps, const bf16* norm_w, float norm_eps, float* pre_out, float* post, float* comb, bf16* x, bf16* xn,
                  cudaStream_t st);
// routing result (ids, rw [n_route]) + activations (xq [n_act], xs [n_act/32]; skipped when n_act=0) → mapped pinned host buffers
void route_to_host(const int32_t* ids, const float* rw, const uint8_t* xq, const uint8_t* xs, int n_route, int n_act, int32_t* ids_h,
                   float* rw_h, uint8_t* xq_h, uint8_t* xs_h, cudaStream_t st);

// ---- RoPE -------------------------------------------------------------------------------------------------------
// freqs_cis: [max_pos, rd/2] complex (cos,sin pairs) fp32. x: rotates the last rd dims of [M, H, D]. pos[m] = position.
// inverse = conjugate. Reference apply_rotary_emb: computes in fp32 and writes back bf16.
void rope_last(bf16* x, int M, int H, int D, int rd, const float2* freqs, const int32_t* pos, bool inverse, cudaStream_t st);
// For group positions of the compressed latents: also passed as pos[m] as above, even when it is not a plain position array.

// ---- attention ---------------------------------------------------------------------------------------------------------
// q [M,H,D] bf16 · 3 KV sources (ring[win,D] · chunk[M,D] · comp[T,D]) · idx [M, topk] int32 (global index: [0,win) ring,
// [win, win+M) chunk, [win+M, …) comp; -1 = none) · sink[H] fp32 → o [M,H,D] bf16. Same online softmax as the reference sparse_attn.
// Per-row (per-sequence) KV sources: in batched decode each row sees the ring and compressed cache of its own sequence. The chunk (this step's new kv) is a shared [M,D].
struct KvRow {
  const bf16* ring;   // [win, D]
  const uint8_t* comp;  // [T, kvp::COMP_ROW] packed fp4 compressed KV cache (of the source layer) — kv_pack.h
  int comp_len;       // number of compressed positions this row may see (end_pos // ratio)
  int topk;           // compressed column count of this row (min(index_topk, comp_len)) — attention columns = win + topk
};

struct KvSources {
  const bf16* ring;   // [win, D]
  int win;            // ring slot count (decode 128 / prefill 128; 0 if chunk only)
  const bf16* chunk;  // [M, D] (prefill: the chunk's own kv) or nullptr
  int chunk_len;
  const uint8_t* comp;  // [T, kvp::COMP_ROW] packed fp4 compressed KV cache
  int comp_len;
};
void sparse_attn(const bf16* q, int M, int H, int D, const KvSources& kv, const int32_t* idx, int idx_stride, int topk,
                 const float* sink, float scale, bf16* o, cudaStream_t st);
// tensor-core version (attn_tc.cu): needs S [M,H,topk] fp32 and mx/sum [M,H] work buffers
void sparse_attn_tc(const bf16* q, int M, int H, int D, const KvSources& kv, const int32_t* idx, int idx_stride, int topk,
                    const float* sink, float scale, float* S, float* mx, float* sum, bf16* o, cudaStream_t st);
// flash version (attn_flash.cu): one kernel, no scratch, block = (query, 32 heads), 46KB smem. H % 32 == 0, D == 512. Default for prefill (M ≥ 16).
void sparse_attn_flash(const bf16* q, int M, int H, int D, const KvSources& kv, const int32_t* idx, int idx_stride, int topk, const float* sink,
                       float scale, bf16* o, cudaStream_t st);
// v4: block = (query, 64 heads), 512 threads, 84KB smem (1 block/SM) — gathers and unpacks each key tile only once per query. Bit-identical to v3.
//   sparse_attn_flash routes here when H % 64 == 0 (v3 with HIVE_ATTN_FLASH32=1). The two below are called directly for tests and A/B.
void sparse_attn_flash64(const bf16* q, int M, int H, int D, const KvSources& kv, const int32_t* idx, int idx_stride, int topk, const float* sink,
                         float scale, bf16* o, cudaStream_t st);
void sparse_attn_flash_v3(const bf16* q, int M, int H, int D, const KvSources& kv, const int32_t* idx, int idx_stride, int topk, const float* sink,
                          float scale, bf16* o, cudaStream_t st);
// row-table version (batched decode): rows[m] gives the ring, compressed cache and column count. Global index [0,win) ring · [win, win+chunk_len) chunk · rest compressed.
//   topk_max = max (win + topk) over rows; the row stride of S.
void sparse_attn_tc_rows(const bf16* q, int M, int H, int D, const KvRow* rows, const bf16* chunk, int chunk_len, int win,
                         const int32_t* idx, int idx_stride, int topk_max, const float* sink, float scale, float* S, float* mx, float* sum,
                         bf16* o, cudaStream_t st);
// fused decode version (M ≤ 8, one kernel): block = (row, head). ncols_max = win + index_topk (smem size)
// with splits > 1 the keys are split and run as blocks (row, head, split), then merged (pacc [M,H,S,D] f32 · pm/ps [M,H,S])
// with rope_freqs, inverse RoPE (same formula as rope_last(inverse)) is applied to the trailing rd dims of the output o — saves one launch
void sparse_attn_decode_rows(const bf16* q, int M, int H, int D, const KvRow* rows, const bf16* chunk, int chunk_len, int win,
                             const int32_t* idx, int idx_stride, int ncols_max, const float* sink, float scale, bf16* o, int splits, float* pacc,
                             float* pm, float* ps, cudaStream_t st, const float2* rope_freqs = nullptr, const int32_t* rope_pos = nullptr, int rd = 0);

// Window index generation. Decode (ring): slot order oldest→newest, -1 if empty. Prefill (chunk): the positions query s sees, as
// global indices (ring slot / chunk row). start_pos = position of the chunk's first token. out [M, win] int32.
// min_src (bounded replay): keys before this position are -1 — so only the decoder replay range [s, i] is visible. Normally 0.
void window_idxs(int M, int win, int64_t start_pos, int32_t* out, int out_stride, cudaStream_t st, int64_t min_src = 0);
// batched decode: row m at position pos[m]; its own kv is chunk row m (global win+m). Earlier tokens come from its own ring.
void window_idxs_rows(int M, int win, const int32_t* pos, int32_t* out, int out_stride, cudaStream_t st);
// batched decode ring write: ring_ptrs[m][(pos[m] % win)] = kv[m]
void ring_write_rows(const bf16* kv, int M, int D, int win, const int32_t* pos, bf16* const* ring_ptrs, cudaStream_t st);
// one batched-decode compressor step (reference Compressor decode): fills the state slot per row and pools when the group is full → out[m], valid[m]=1
void compressor_step_rows(const float* kv, const float* score, int M, int D, int ratio, const int32_t* pos, float* const* state_kv,
                          float* const* state_score, float* out, uint8_t* valid, cudaStream_t st);
// per-row cache write: if valid[m], dst_ptrs[m][dst_index[m]*D ..] = src[m]
void write_cache_rows(const bf16* src, int M, int D, bf16* const* dst_ptrs, const int32_t* dst_index, const uint8_t* valid, cudaStream_t st);
// fp4 packed write (kv_pack.h format): x[M,K] bf16 → per row [K/block scales … K/2 nibbles] (row_bytes = header + K/2). Same quantization as fp4_quant_roundtrip.
//   contiguous rows: dst + m·row_bytes · per-row pointers: if valid[m], dst_ptrs[m] + dst_index[m]·row_bytes
void fp4_pack(const bf16* x, int M, int K, int block, bool scale_e4m3, uint8_t* dst, int row_bytes, cudaStream_t st);
void fp4_pack_rows(const bf16* x, int M, int K, int block, bool scale_e4m3, uint8_t* const* dst_ptrs, const int32_t* dst_index, const uint8_t* valid,
                   int row_bytes, cudaStream_t st);
// writes chunk kv [M,D] into the ring: ring[(start_pos+s) % win] = kv[s] (later entries win)
void ring_write(const bf16* kv, int M, int D, int win, int64_t start_pos, bf16* ring, cudaStream_t st);
// x = bf16(x · c)  (reference: bf16 tensor × Python float)
void scale_bf16(bf16* x, int n, float c, cudaStream_t st);
// fills 0..T-1 per row (int32 [M,T]) — for CUB sort
void iota_rows(int32_t* out, int M, int T, cudaStream_t st);
// bf16 → fp32 (sort keys)
void bf16_rows_to_f32(const bf16* src, int n, float* dst, cudaStream_t st);

// ---- compressor / indexer ------------------------------------------------------------------------------------------------
// ratio>1: kv[M,D] score[M,D] fp32 (per-token projections of the chunk) + state (state_kv/state_score [ratio, D], fill count of the first token's group slot)
//   → softmax-weighted pooling for each completed group → out[G, D] fp32 (G = groups completed now). The leftover tail stays in the state.
//   Group boundaries come from start_pos (global position // ratio). The returned G is computed by the host.
void compressor_pool(const float* kv, const float* score, int M, int D, int ratio, int64_t start_pos, float* state_kv,
                     float* state_score, float* out, cudaStream_t st);
// Indexer scores: q [M,Hi,Di] bf16 · k [T,Di] bf16 · w [M,Hi] bf16 → score[M,T] bf16 = Σ_h relu(q·k)·w  (reference bf16 einsum semantics:
//   fp32 accumulate then bf16 round, relu, bf16 multiply, head sum accumulated in fp32 then bf16). Then -inf where t ≥ visible[m].
//   k = packed fp4 indexer keys [T, kvp::IDX_ROW] (kv_pack.h)
void indexer_scores(const bf16* q, const uint8_t* k, const bf16* w, int M, int Hi, int Di, int T, const int32_t* visible,
                    bf16* score, cudaStream_t st);
// per-row indexer keys (k_ptrs[m], length T_rows[m]); score [M, Tmax] (-inf past T_rows[m] and past visible)
void indexer_scores_rows(const bf16* q, const uint8_t* const* k_ptrs, const int32_t* T_rows, int Tmax, const bf16* w, int M, int Hi, int Di,
                         const int32_t* visible, bf16* score, cudaStream_t st);
// candidate block mask (candidate_source): score[M,T] → keep[M,T] bool. Block max → last (partial) block +inf → topk blocks.
//   topk runs on the GPU instead of a host sort: block scores are sorted (CUB) — here only block_max is produced; runtime sorts with CUB.
void block_max(const bf16* score, int M, int T, int bs, const int32_t* visible, float* bmax, cudaStream_t st);
// keep[m, pos[m,j]] = 1 (only where bmax[m, pos] > -inf)
void scatter_keep(const int32_t* pos, const float* bmax, int M, int k, int nblocks, uint8_t* keep, int keep_stride, cudaStream_t st);
void apply_block_mask(bf16* score, int M, int T, int bs, const uint8_t* keep_block, int nblocks, cudaStream_t st);
// per-row top-k (radix select, one block per row): keys [M, row_stride] fp32 (-inf allowed) → out[M, out_stride] top-k indices ascending (-1 if short)
// candidate-pool scores: cand[m][0..kb) = block ids chosen by layer 20 (ascending, -1 if short) · score[m, kb·bs] (indexed by candidate ordinal ci) — values bit-identical to the full-T version
void indexer_scores_cand(const bf16* q, const uint8_t* k, const uint8_t* const* k_ptrs, const int32_t* T_rows, const bf16* w, int M, int Hi, int Di,
                         const int32_t* cand, int cand_stride, int kb, int bs, const int32_t* visible, bf16* score, cudaStream_t st);
void cand_to_pos(int32_t* sel, int M, int k, const int32_t* cand, int cand_stride, int bs, cudaStream_t st);
// fp32 indexer scores (HIVE_IDX_F32=1 — fp32 logits without per-head bf16 rounding, fewer ties; off by default until a quality A/B). Outputs fp32 scores directly (no bf16→f32 conversion)
void indexer_scores_f32(const bf16* q, const uint8_t* k, const bf16* w, int M, int Hi, int Di, int T, const int32_t* visible, float* score, cudaStream_t st);
void indexer_scores_rows_f32(const bf16* q, const uint8_t* const* k_ptrs, const int32_t* T_rows, int Tmax, const bf16* w, int M, int Hi, int Di,
                             const int32_t* visible, float* score, cudaStream_t st);
void indexer_scores_cand_f32(const bf16* q, const uint8_t* k, const uint8_t* const* k_ptrs, const int32_t* T_rows, const bf16* w, int M, int Hi, int Di,
                             const int32_t* cand, int cand_stride, int kb, int bs, const int32_t* visible, float* score, cudaStream_t st);
void block_max_f32(const float* score, int M, int T, int bs, const int32_t* visible, float* bmax, cudaStream_t st);  // (cand_to_pos above: candidate ordinal → position, in place)
void topk_select_rows(const float* keys, int M, int T, int k, int row_stride, int32_t* out, int out_stride, cudaStream_t st);
// per-row T (T_rows[m] ≤ T) version: each row only looks at its own length
void topk_select_rows_t(const float* keys, int M, const int32_t* T_rows, int T, int k, int row_stride, int32_t* out, int out_stride,
                        cudaStream_t st);
// sorted top-topk positions (pos_sorted [M, topk] int32, ascending, -1 if not visible) → add offset to get global indices
void offset_idxs(const int32_t* pos, int M, int topk, const int32_t* visible, int32_t offset, int32_t* out, int out_stride,
                 cudaStream_t st);

// ---- MoE ----------------------------------------------------------------------------------------------------------------
// router: scores[M,E] fp32 (= x·Wgᵀ, cuBLAS) → sqrt(softplus) → (+bias) top-k → ids[M,k] int32, w[M,k] fp32 (normalized, scaled)
// with bias_vl and is_image, image tokens are selected with bias_vl (reference noaux_tc_for_vl)
void router_topk(const float* scores, const float* bias, const float* bias_vl, const int8_t* is_image, int M, int E, int k,
                 float route_scale, int32_t* ids, float* w, cudaStream_t st);
// expert intermediate: gate[M,I] up[M,I] bf16 (GEMM output) → y[M,I] bf16 = bf16( silu(clamp(gate)) · clamp(up) · route_w[m] )
//   route_w nullptr means 1 (shared expert).
void swiglu_route(const bf16* gate, const bf16* up, const float* route_w, int M, int I, float limit, bf16* y, cudaStream_t st);
// accumulate: acc[m,:] += bf16 value(src[m,:]) (fp32). With row_map, acc row = row_map[m].
void accum_bf16_rows(const bf16* src, const int32_t* row_map, int M, int dim, float* acc, cudaStream_t st);
void accum_f32_rows(const float* src, const int32_t* row_map, int M, int dim, float* acc, cudaStream_t st);
// deterministic version: all R rows at once, added in ascending r for each acc row m (grouped expert path)
void accum_bf16_rows_seq(const bf16* src, const int32_t* row_map, int R, int M, int dim, float* acc, cudaStream_t st);
// fp32 → bf16
void f32_to_bf16(const float* src, int n, bf16* dst, cudaStream_t st);
// row gather: dst[i,:] = src[rows[i],:] (bf16, per-expert token batches)
void gather_rows_bf16(const bf16* src, const int32_t* rows, int n, int dim, bf16* dst, cudaStream_t st);
void gather_rows_u8(const uint8_t* src, const int32_t* rows, int n, int row_bytes, uint8_t* dst, cudaStream_t st);

// ---- Engram -------------------------------------------------------------------------------------------------------------
// looked-up rows (fp8 [M, cols, hd] + e8m0 [M, cols, hd/32]) → bf16 [M, cols·hd]
void engram_dequant(const uint8_t* vals, const uint8_t* scales, int M, int cols, int hd, bf16* out, cudaStream_t st);
// gate: kv [M, dim·(hc+1)] bf16 (key = first hc·dim, value = last dim), qk_weight [hc, dim] fp32 (q·k product), h [M,hc,dim] in place
// tokens with is_image[m] == 1 are left untouched (reference token_mask → gate 0)
void engram_gate(const bf16* kv, const float* qk_weight, const int8_t* is_image, int M, int hc, int dim, float eps, bf16* h,
                 cudaStream_t st);
// image span merge: h[start+p, c, :] = (type 0: image_start / 2: newline / 3: image_end / 1: rows[row_of[p]]) — into every hc copy
void merge_image(bf16* h, int hc, int dim, int start, int span, const int8_t* types, const int32_t* row_of, const bf16* rows,
                 const bf16* v_start, const bf16* v_newline, const bf16* v_end, cudaStream_t st);

// ---- output head --------------------------------------------------------------------------------------------------------
// logits[m, v] = Σ_d x[m,d]·head[v,d]  (x fp32 [M,dim], head bf16 [V,dim], fp32 accumulation)
void head_logits(const float* x, const bf16* head, int M, int V, int dim, float* logits, cudaStream_t st);
void argmax_rows(const float* logits, int M, int V, int32_t* out, cudaStream_t st);
void bf16_to_f32(const bf16* src, int n, float* dst, cudaStream_t st);

// ---- DSpark (MTP) helpers ----------------------------------------------------------------------------------------------------
// hc mean of the target-layer attention input (reference h.mean(dim=2)): dst[i, col_off + :dim] = mean_c h[row0+i, c, :]  (i < n)
void hc_mean_rows(const bf16* h, int hc, int dim, int row0, int n, bf16* dst, int dst_stride, int col_off, cudaStream_t st);
// markov head: dst[rank] = embed[*id] · logits[v] += Σ_r head[v,r]·e[r] (fp32)
void markov_gather(const bf16* embed, const int32_t* id, int rank, bf16* dst, cudaStream_t st);
void markov_add_bias(float* logits, const bf16* head, const bf16* e, int V, int rank, cudaStream_t st);
// confidence head: out[b] = w[:dim]·x[b] + w[dim:]·e[b]  (fp32)
void conf_dot(const bf16* x, int dim, const bf16* e, int rank, const float* w, int B, float* out, cudaStream_t st);
// window ring snapshot/restore (rollback after speculative verification): slots (start_pos+i)%win of L layers, i<n. Restore only covers i ∈ [i0, n).
void ring_gather_layers(bf16* const* rings, int L, int win, int64_t start_pos, int n, int D, bf16* dst, cudaStream_t st);
void ring_scatter_layers(const bf16* src, bf16* const* rings, int L, int win, int64_t start_pos, int i0, int n, int D, cudaStream_t st);
// draft block attention indices (reference get_dspark_topk_idxs): [0,filled) ring slots + [win, win+B) block rows + -1. out [B, stride], win+B columns
void dspark_idxs(int B, int filled, int win, int32_t* out, int stride, cudaStream_t st);
// dst_ptrs[m][0..n) = src[m, 0..n)
void scatter_rows_ptrs(const bf16* src, int M, int n, bf16* const* dst_ptrs, cudaStream_t st);
// ---- GPU sampler helpers: rowmax[m] = max_v logits · rowsum[m] = Σ_v exp((l−max)·inv_temp[m]) · vals[m,j] = logits[m, idx[m,j]] (-1 → -inf)
void row_max_sumexp(const float* logits, int M, int V, const float* inv_temp, float* rowmax, float* rowsum, cudaStream_t st);
void gather_vals(const float* logits, const int32_t* idx, int M, int NC, int V, float* vals, cudaStream_t st);

}  // namespace hive::k
