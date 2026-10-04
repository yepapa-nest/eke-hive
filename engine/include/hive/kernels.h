// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive GPU kernel API (v1 = CUDA-core, correctness first; v2 = to be replaced by sm_120a block-scaled tensor-core mma)
// All functions are asynchronous (stream) and all input pointers are device memory.
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "hive/common.h"

namespace hive::k {

// Activation quantization (reference act_quant(x, 32, ue8m0)): x bf16 [M,K] -> y e4m3 [M,K], s e8m0 [M, K/32]
void act_quant_fp8(const bf16* x, int M, int K, uint8_t* y, uint8_t* s, cudaStream_t st);
// Round trip of the same quantization (dequantized values written back in place as bf16) — for windowed KV storage
void act_quant_fp8_roundtrip(bf16* x, int M, int K, cudaStream_t st);
// fp4 round trip: block=32 with e8m0 (indexer q/k) or block=16 with e4m3 scales (compressed KV)
void fp4_quant_roundtrip(bf16* x, int M, int K, int block, bool scale_e4m3, cudaStream_t st);
// RMSNorm (fp32 compute): out = bf16( w * x * rsqrt(mean(x^2)+eps) )
void rmsnorm(const bf16* x, const bf16* w, float eps, int M, int K, bf16* out, cudaStream_t st);

// Block-scaled GEMM: C[M,N] = A(e4m3 [M,K] · sa e8m0 [M,K/32]) x B(e4m3 [N,K] or e2m1 nibble-packed [N,K/2] · sb e8m0 [N,K/32])^T
// fp32 accumulation. Writes whichever of C (bf16) and C32 (fp32) is non-null (both allowed).
void gemm_bs(const uint8_t* A, const uint8_t* sa, const uint8_t* B, const uint8_t* sb, bool b_fp4, int M, int N, int K,
             bf16* C, float* C32, cudaStream_t st);
// Tensor-core (WMMA bf16) version — gemm_bs routes here automatically for large M
void gemm_bs_tc(const uint8_t* A, const uint8_t* sa, const uint8_t* B, const uint8_t* sb, bool b_fp4, int M, int N, int K, bf16* C,
                float* C32, cudaStream_t st);
// bf16 activations x fp8 weights (per-row e8m0/32) tensor-core GEMM with explicit row strides: C[M, ldc] = A[M, lda](bf16) · B[N,K](fp8 dequantized)^T — wo_a prefill (no bf16 copy needed)
void gemm_bf16a_fp8b_tc(const bf16* A, int lda, const uint8_t* B, const uint8_t* sb, int M, int N, int K, bf16* C, int ldc, cudaStream_t st);
// Force a path (tests): 0 = auto, 1 = CUDA-core tiles, 2 = tensor cores
void gemm_bs_force_path(int p);
// D-1 prefill block-scaled GEMM (fp4 weights only, no M limit). gemm_bs routes here for M > 8 when HIVE_MX_PREFILL=1.
void gemm_bs_mx(const uint8_t* A, const uint8_t* sa, const uint8_t* B, const uint8_t* sb, int M, int N, int K, bf16* C, float* C32, cudaStream_t st);
// Block-scaled tiled GEMM (fp8 and fp4 B, 128x128x64, cp.async 2 stages for fp8 / 3 for fp4, native mma kind::mxf8f6f4 — no dequantization). K % 64 == 0. Enabled with HIVE_MX_GEMM=1.
void gemm_bs_mx_tiled(const uint8_t* A, const uint8_t* sa, const uint8_t* B, const uint8_t* sb, bool b_fp4, int M, int N, int K, bf16* C, float* C32,
                      cudaStream_t st);
// D-2 tensor-core indexer scores — query and keys both in kvp::IDX_ROW packing (query row = (m, h)). Same output as indexer_scores (only the fp32 accumulation order differs).
void indexer_scores_tc(const uint8_t* q_packed, const uint8_t* k_packed, const bf16* w, int M, int Hi, int T, const int32_t* visible, bf16* score,
                       cudaStream_t st);
// Candidate-pool version (tensor-core version of indexer_scores_cand): q_packed = (m, h) rows in IDX_ROW packing (fp4_pack, before the round trip);
//   keys from k_packed (single cache) or k_ptrs[m] (+ T_rows[m], batched decode); cand[m][0..kb) block ids ascending (-1 = none);
//   score[m, kb*bs] (indexed by candidate position). Hi % 16 == 0, Hi <= 64.
void indexer_scores_cand_tc(const uint8_t* q_packed, const uint8_t* k_packed, const uint8_t* const* k_ptrs, const int32_t* T_rows, const bf16* w, int M,
                            int Hi, const int32_t* cand, int cand_stride, int kb, int bs, const int32_t* visible, bf16* score, cudaStream_t st);

// Decode grouped expert GEMV — all resident experts of a layer in one launch (8 launches per expert -> 6 per layer).
// Group g = one expert: weight pointers + row range [row0, row0+n) of the gathered activation buffer (n <= 8).
struct GroupDesc {
  const uint8_t *w1, *s1, *w3, *s3, *w2, *s2;  // e2m1 nibbles, e8m0/32
  int row0, n;
};
// y[row, :I] = swiglu(bf16(A·w1^T), bf16(A·w3^T)) · rw[row]  — bit-identical to gemm_bs(gemv) + swiglu_route
void gemv_grouped_w13(const GroupDesc* g, int ngroups, const uint8_t* A, const uint8_t* sa, int I, int K, const float* rw, float limit,
                      bf16* y, cudaStream_t st);
// Decode wo_a: for each group g, C[m, g*R + r] = sum_k A[m, g*sub + k] · deq(B[g*R + r, k])  (A bf16 [M, lda], B fp8 [G*R, sub] + sb e8m0 [G*R, sub/32]) — M <= 8
void gemv_bf16_fp8_grouped(const bf16* A, int lda, const uint8_t* B, const uint8_t* sb, int M, int G, int R, int sub, bf16* C, cudaStream_t st);
// Tensor-core version (sm_120a mma.sync kind::mxf8f6f4.block_scale) — same contract, only the fp32 accumulation order differs
//   rows: group row -> A row index (nullptr = consecutive rows of the gathered buffer). yq/ys: if given, this kernel also writes
//   the 32-block quantization of y (same formula as act_quant_fp8)
void mx_grouped_w13(const GroupDesc* g, int ngroups, const uint8_t* A, const uint8_t* sa, int I, int K, const float* rw, float limit, bf16* y,
                    cudaStream_t st, const int32_t* rows = nullptr, uint8_t* yq = nullptr, uint8_t* ys = nullptr);
void mx_grouped_w2(const GroupDesc* g, int ngroups, const uint8_t* Yq, const uint8_t* ys, int dim, int I, bf16* eout, cudaStream_t st);
// eout[row, :dim] = bf16(Yq·w2ᵀ)
void gemv_grouped_w2(const GroupDesc* g, int ngroups, const uint8_t* Yq, const uint8_t* ys, int dim, int I, bf16* eout, cudaStream_t st);
// D1 decode expert fusion (HIVE_DECODE_FUSED, moe_decode.cu): the chain mx_grouped_w13 (rows map + yq/ys) -> mx_grouped_w2 ->
//   accum_bf16_rows_seq(eout + r0*dim, rows + r0, n_rows, M, ...) in one launch — same output (y, yq, ys, eout, acc bit-identical, same order).
//   g/rows/rw are device tables (row0 = global row); rows [r0, r0+n_rows) belong to these groups. Usable iff moe_decode_fused_ok and
//   moe_decode_desc_aligned for every group.
bool moe_decode_fused_ok(int dim, int I, int M, int ngroups);
// G1 HIVE_DECODE_STEP_GRAPH: device-count version of the same kernel — ng = base[ng], r0 = r0 < 0 ? 0 : base[r0], n_rows = base[nrows] (int indices).
struct DfDevCount { const int* base = nullptr; int ng = 0, r0 = -1, nrows = 0; };
void moe_decode_fused_dev(const GroupDesc* g, int max_groups, DfDevCount dc, const uint8_t* xq, const uint8_t* xs, const int32_t* rows, const float* rw,
                          int M, int dim, int I, float limit, bf16* y, uint8_t* yq, uint8_t* ys, bf16* eout, float* acc, cudaStream_t st);
void moe_decode_fused_preload(cudaStream_t st);  // performs lazy module loading and stream-counter setup ahead of time (outside capture and spin waits)
// E2 HIVE_DECODE_FUSED2 (env_on): replaces the bodies of the two versions above (moe_decode_fused and _dev) with F2 (cp.async lane
//   ring, w2 weight prefetch, persistent grid) — bit-identical values. Stage count HIVE_DECODE_FUSED2_STAGES (S2 in {3,4,6}, default 4).
//   Test override: -1 = environment, 0 = D1, anything else = F2 with S2 (normalized to 3/4/6). variant() = current version (0 = D1).
void moe_decode_fused_force(int stages);
// R4 HIVE_DECODE_FUSED3 (env_on — takes precedence over F2 when on): replaces the bodies of the same two versions with F3 (warp-level
//   work items, 128B contiguous stages per row) — bit-identical values. Stage count HIVE_DECODE_FUSED3_STAGES (S in {3,4,5}, default 4).
//   Test override force3: -1 = environment, anything else = F3 with S. variant() = 100 + S (F3), 3/4/6 (F2), 0 (D1).
void moe_decode_fused_force3(int stages);
int moe_decode_fused_variant();
bool moe_decode_desc_aligned(const GroupDesc& d);
void moe_decode_fused(const GroupDesc* g, int ngroups, const uint8_t* xq, const uint8_t* xs, const int32_t* rows, const float* rw, int r0, int n_rows,
                      int M, int dim, int I, float limit, bf16* y, uint8_t* yq, uint8_t* ys, bf16* eout, float* acc, cudaStream_t st);

// ---- H1 grouped prefill experts (HIVE_GROUPED_PREFILL) ----------------------------------------------------------------------
// Path selection for gemm_bs (single place): gemm_bs itself uses it too, so the grouped path picks **the same kernel formula** for
// every expert GEMM call (with M rows).
enum GemmRoute { kGemmGemv = 0, kGemmMxTiled = 1, kGemmMxRows = 2, kGemmTc = 3, kGemmTile = 4 };
int gemm_bs_route(bool b_fp4, int M, int K);
// Grouped GEMM (fp4 expert weights): block = (column tile 128, row tile tiles[y], z). Row tile = GroupDesc (row0 = gathered row index, n <= 128, 6 expert weights).
//   which 0: z=0 -> w1 -> C0, z=1 -> w3 -> C1 (N = I, K = dim); which 1: w2 -> C0 (N = dim, K = I).
//   A row (gathered row r): with rref, a_base[rref[r] >> 24] + (rref[r] & 0xFFFFFF)*K (sub-chunk activations addressed directly via the row map — no gather),
//   otherwise A + r*K.
//   route = kGemmMxTiled (same 128x128x64 mma loop as gemm_bs_mx_tiled) or kGemmTc (same WMMA loop as gemm_bs_tc). The output of one row tile is
//   **bit-identical** to calling gemm_bs separately for the same expert (per-row independent mma, same k order — tile boundaries are also every
//   128 rows from expert row 0).
void gemm_fp4_grouped(int route, const GroupDesc* tiles, int ntiles, int which, const uint8_t* const* a_base, const uint8_t* const* s_base,
                      const int32_t* rref, const uint8_t* A, const uint8_t* sa, int N, int K, bf16* C0, bf16* C1, cudaStream_t st);
void gemm_bs_mx_tiled_grouped(const GroupDesc* tiles, int ntiles, int which, const uint8_t* const* a_base, const uint8_t* const* s_base, const int32_t* rref,
                              const uint8_t* A, const uint8_t* sa, int N, int K, bf16* C0, bf16* C1, cudaStream_t st);
void gemm_bs_tc_grouped(const GroupDesc* tiles, int ntiles, int which, const uint8_t* const* a_base, const uint8_t* const* s_base, const int32_t* rref,
                        const uint8_t* A, const uint8_t* sa, int N, int K, bf16* C0, bf16* C1, cudaStream_t st);
// Row-map gather: gathered row r in [r0, r0+n) -> dq[r, :dim] = xq_base[s][m, :], ds[r, :dim/32] = xs_base[s][m, :]  (s, m decoded from rref[r])
void gather_rows_ref(const uint8_t* const* xq_base, const uint8_t* const* xs_base, const int32_t* rref, int r0, int n, int dim, uint8_t* dq, uint8_t* ds,
                     cudaStream_t st);
// swiglu + 32-block quantization in one launch: rows r in [r0, r2). r < r1: swiglu_route formula (gate/up bf16 · rw[r]) -> bf16 y -> act_quant_fp8 formula;
//   r >= r1: quantize y[r] as is (gemv_grouped_w13 already wrote y for those rows). y is not written for r < r1 (later stages read only yq/ys).
//   Bit-identical to the swiglu_route -> act_quant_fp8 chain.
void swiglu_quant_rows(const bf16* gate, const bf16* up, const bf16* y, const float* rw, int r0, int r1, int r2, int I, float limit, uint8_t* yq, uint8_t* ys,
                       cudaStream_t st);
// Order-fixed accumulation: for each destination d, acc[d] <- (((acc[d] + v_0) + v_1) + ...), v_i = bf16 value(src[contrib[c0+i], :]) — same order and
//   same fp32 additions as the previous loop's per-expert accum_bf16_rows (1 atomic add = 1 rounded add), hence bit-identical. Destinations are distinct
//   within a launch (no atomics needed).
struct AccDest { float* acc; int c0, n; };
void accum_ordered_rows(const AccDest* dests, int ndest, const int32_t* contrib, const bf16* src, int dim, cudaStream_t st);

// Batch executor — builds the tables (gathered row -> sub-chunk row and routing weight, row tiles, gemv groups, accumulation destinations) in a pinned
//   ring, copies them in one transfer and launches stage by stage.
//   One seg = one expert GEMM call of the previous loop (same row set, same row count -> same path: n <= 8 gemv, otherwise route13/route2). seg order =
//   accumulation order.
//   late segs (streaming — must wait for their copy to complete) are computed after before_late() (the caller waits on the stage_copied event there).
struct GroupedRef { int s, m; float rw; };
struct GroupedSeg { const uint8_t *w1, *s1, *w3, *s3, *w2, *s2; int i0, n; bool late; };  // rows = refs[i0, i0+n)
struct GroupedBufs { bf16 *gate, *up, *y, *eout; uint8_t *yq, *ys, *gxq, *gxs; };
class GroupedMoe {
 public:
  GroupedMoe(int max_rows, int max_subs);  // one batch <= max_rows rows (must not exceed the row capacity of buffers gate/up/y/yq/eout/gxq); sub-chunks <= max_subs
  ~GroupedMoe();
  GroupedMoe(const GroupedMoe&) = delete;
  GroupedMoe& operator=(const GroupedMoe&) = delete;
  int max_rows() const { return max_rows_; }
  int max_subs() const { return max_subs_; }
  void run(const GroupedSeg* segs, int nseg, const GroupedRef* refs, const uint8_t* const* xq_sub, const uint8_t* const* xs_sub, float* const* acc_sub,
           int n_sub, const int* sub_rows, int dim, int I, float limit, const GroupedBufs& b, cudaStream_t st, const std::function<void()>& before_late);
  // Whether, for this shape, the path of an n > 8 expert GEMM (gemm_bs_route) has a grouped version (kGemmMxTiled, kGemmTc) — otherwise (e.g. the HIVE_MX_PREFILL version) the caller uses the previous loop.
  static bool supported(int dim, int I);
  int batches() const { return n_batches_; }

 private:
  static constexpr int kRing = 4;  // table ring (the host may run up to 4 batches ahead of the GPU — so side-stream copy issue is never blocked)
  int max_rows_, max_subs_;
  size_t region_ = 0;
  uint8_t* host_[kRing] = {};
  uint8_t* dev_[kRing] = {};
  cudaEvent_t done_[kRing] = {};
  bool used_[kRing] = {};
  int next_ = 0, n_batches_ = 0;
  std::vector<int> stamp_, dslot_;  // (sub-chunk, row) -> destination index (stamp per batch avoids re-initialization)
  std::vector<int> sub_off_, cnt_, row_of_seg_;
  int cur_stamp_ = 0;
};

// ---- Decode fusion (fused.cu, M <= 8) — bit-identical to the unfused chain. counter = device int (last-block counter initialized to 0, a different one per call) --------
size_t fused_smem_max_k();
// qr = wq_a(quant xn) -> qrn = rmsnorm(qr) -> q = rope(wq_b(quant qrn)); qrq/qrs = quantized qrn (for the indexer, may be nullptr)
void fused_q_proj(const bf16* xn, int K, const uint8_t* wqa, const uint8_t* sqa, int Nqa, const bf16* q_norm, float eps, const uint8_t* wqb,
                  const uint8_t* sqb, int H, int D, int rd, const float2* freqs, const int32_t* pos, int M, bf16* qr, bf16* qrn, bf16* q, uint8_t* qrq,
                  uint8_t* qrs, int* counter, cudaStream_t st);
// kv = fp8 round trip(rope(rmsnorm(wkv(quant xn)))) -> kv[M,D] (also to ring_ptrs[m][(pos[m]%win)]); if idx_out is given, also writes the window indices (window_idxs_rows)
void fused_kv_proj(const bf16* xn, int K, const uint8_t* wkv, const uint8_t* skv, int D, const bf16* kv_norm, float eps, int rd, const float2* freqs,
                   const int32_t* pos, int M, bf16* kv, bf16* const* ring_ptrs, int win, int32_t* idx_out, int idx_stride, int* counter, cudaStream_t st);
// C = B(quant A) bf16
void fused_gemv_quantin(const bf16* A, int K, const uint8_t* B, const uint8_t* sb, int M, int N, bf16* C, cudaStream_t st);
// Shared experts: y = swiglu(w1·quant xn, w3·quant xn) -> yq/ys -> acc = bf16 value(w2·yq) (fp32 write). xq/xs = quantized xn (for the expert path)
void fused_shared_experts(const bf16* xn, int K, const uint8_t* w1, const uint8_t* s1, const uint8_t* w3, const uint8_t* s3, const uint8_t* w2,
                          const uint8_t* s2, int M, int I, float limit, bf16* y, uint8_t* xq, uint8_t* xs, float* acc, cudaStream_t st);
// Router: scores (fp32 GEMV, only the order differs from cuBLAS) -> top-k (same formula as router_topk)
void fused_router(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                  float route_scale, float* scores, int32_t* ids, float* w, int* counter, cudaStream_t st);
// Layer tail: h = hc_post(bf16(acc)), pre_mix = pre_f
void hc_post_tail(const float* acc, const float* post, const float* comb, int M, int hc, int dim, bf16* h, const float* pre_f, float* pre_mix, cudaStream_t st);

}  // namespace hive::k
