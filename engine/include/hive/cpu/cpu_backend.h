// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU compute backend shared by every model family's CPU path (CPU-only daemons and the CPU side of hybrid placements).
//   Target: Zen2 (AVX2 + FMA + F16C, no AVX-512), 2 NUMA nodes, DDR4. Decode is bound by memory bandwidth, so every weight matrix is split
//   by output rows across the NUMA nodes (node k holds rows [split_k, split_{k+1}) in its own memory) and only that node's pinned workers
//   read it — the layout GlmExperts measured at 86 GB/s against ~45 GB/s for interleaved records. Prefill is bound by arithmetic: the
//   matmul packs activation panels and runs an FMA micro-kernel over weight rows converted once per row block.
//   Activations are fp32 on the CPU side; the model code rounds them to bf16 wherever the reference keeps bf16 (see each family).
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hive::cpu {

// ---- thread pool ---------------------------------------------------------------------------------------------------------------------
// HIVE_CPU_POOL_THREADS (default: the physical cores — SMT siblings add no memory bandwidth), split evenly over the NUMA nodes, each worker
//   pinned to its node's CPUs. Workers spin HIVE_CPU_POOL_SPIN_US (default 200) after a job before sleeping. HIVE_CPU_POOL_SMT=1 adds the
//   SMT helpers (see run()); they spin HIVE_CPU_POOL_SMT_SPIN_US (default 50) after a compute-bound job.
class Pool {
 public:
  static Pool& get();
  int threads() const { return n_; }
  int nodes() const { return nodes_; }
  int threads_on(int node) const { return per_node_[node]; }
  // f(i) for i in [0, n): tasks taken by every worker (no node affinity). The caller waits.
  //   compute_bound: the SMT helper workers (HIVE_CPU_POOL_SMT=1, one per remaining hardware thread) join this dispatch too — for
  //   arithmetic-bound work such as the GEMM (ws 3975WX: +14-16 % GFLOP/s with both hardware threads of every core); others never wake them.
  void run(int n, const std::function<void(int)>& f, bool compute_bound = false);
  // f(node, i) for i in [0, n_k) on node k's workers only (and its SMT helpers when compute_bound). The caller waits.
  void run_nodes(const std::vector<int>& n_per_node, const std::function<void(int, int)>& f, bool compute_bound = false);
  int smt_threads() const;  // SMT helper workers (0 unless HIVE_CPU_POOL_SMT=1)
  ~Pool();
  struct Impl;
 private:
  Pool();
  std::unique_ptr<Impl> impl_;
  int n_ = 1, nodes_ = 1;
  std::vector<int> per_node_;
};

// NUMA-placed host memory (falls back to plain allocation without libnuma or with one node).
void* numa_alloc_node(size_t bytes, int node);
void numa_free_node(void* p, size_t bytes);

// ---- weight matrices -------------------------------------------------------------------------------------------------------------------
enum class WType : uint8_t {
  BF16 = 0,      // w bf16 [N, K]
  F32 = 1,       // w fp32 [N, K]
  FP8 = 2,       // w e4m3 [N, K] · s fp32 [ceil(N/bn), ceil(K/bk)] block scales (Z.ai / Qwen MTP 128×128, DeepSeek dense 32×32 e8m0 —
                 //   every scale format is converted to fp32 by make_mat)
  NVFP4 = 3,     // w e2m1 nibbles [N, K/2] (low nibble = even column) · s e4m3 [N, K/16] · gscale fp32 (ModelOpt)
  MXFP4 = 4,     // w e2m1 nibbles [N, K/2] · s e8m0 [N, K/32] (DeepSeek experts)
  I8 = 5,        // w int8 [N, K] · s fp32 [N, ceil(K/bk)] per-row group scales (opt-in lossy compression of bf16 weights, make_mat_i8)
};
enum class SType : uint8_t { F32 = 0, BF16 = 1, E8M0 = 2, E4M3 = 3 };  // source scale dtype (FP8 block scales are stored as fp32)

// One matrix [N, K] (out × in, the checkpoint's row-major layout), rows split across the NUMA nodes.
struct Mat {
  WType t = WType::BF16;
  int N = 0, K = 0;
  float gscale = 1.f;            // NVFP4 global scale
  int bn = 128, bk = 128;        // FP8 block shape
  int nodes = 1;
  int row0[3] = {0, 0, 0};       // node k holds rows [row0[k], row0[k+1])
  uint8_t* w[2] = {nullptr, nullptr};  // per node: weight bytes of its rows (row r of node k at w[k] + (r − row0[k]) · row_bytes)
  uint8_t* s[2] = {nullptr, nullptr};  // per node: scales of its rows (FP8: the block rows covering its rows, fp32)
  size_t bytes = 0;              // host bytes held (both nodes)
  size_t row_bytes() const;
  size_t scale_row_bytes() const;
};
// Copies a checkpoint matrix into NUMA-split memory (the node boundary is a multiple of the FP8 block rows). w / s point at the checkpoint
//   bytes (mmap); sdt = the dtype of s (FP8 block scales of any dtype are stored as fp32). Frees with free_mat.
Mat make_mat(WType t, int N, int K, const void* w, const void* s = nullptr, SType sdt = SType::F32, float gscale = 1.f, int bn = 128, int bk = 128);
void free_mat(Mat& m);
// Lossy: bf16 [N, K] quantized to I8 with one symmetric scale per row and group of `group` columns (scale = max|w| / 127, round to nearest).
//   Halves the bytes a GEMV reads; the caller decides where the accuracy cost is acceptable.
Mat make_mat_i8(int N, int K, const uint16_t* w_bf16, int group = 64);
// Rows [a, b) of m as a new matrix that aliases m's memory (no copy, no ownership) — e.g. the q/k/v parts of a stacked projection.
//   Only valid for row ranges that do not cross a NUMA boundary in a way the kernels cannot handle: the kernels accept any range.
struct MatView { const Mat* m; int r0, r1; };

// ---- matmul ------------------------------------------------------------------------------------------------------------------------------
// Y[m, n] = Σ_k X[m, k] · W[r0 + n, k] for m < M, n < r1 − r0 (fp32 accumulation; Y fp32, row stride ldy). X fp32 [M, K] (row stride ldx).
//   M ≤ 8: row-streaming GEMV (each weight row read once for all M rows), node-local rows per worker.
//   M > 8: blocked GEMM (weight row blocks converted to fp32 once per block, FMA micro-kernel), all workers.
//   Deterministic: every output element has a fixed summation order independent of the thread count.
void matmul(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy);
inline void matmul(const Mat& W, const float* X, int M, int ldx, float* Y, int ldy) { matmul(MatView{&W, 0, W.N}, X, M, ldx, Y, ldy); }
// Several independent matmuls in one pool dispatch (decode: q|k|v|… projections; MoE: every expert of a layer).
// X: this job's own activations; nullptr = the shared X of the call. M / ldx: this job's row count and row stride; 0 = the call's.
//   Each job's results are identical to a matmul of its own (its M selects GEMV or GEMM as above).
struct MatJob { MatView W; float* Y; int ldy; const float* X = nullptr; int M = 0; int ldx = 0; };
void matmul_multi(const MatJob* jobs, int n_jobs, const float* X, int M, int ldx);
// A/B measurement only: true = the whole matmul as of 2026-10-09 19:30 (src/cpu/cpu_matmul_legacy.cpp — GEMV with per-M accumulator
//   arrays, the GEMM before the decode fixes); HIVE_CPU_MATMUL_LEGACY=1 does the same for the whole process. Identical results.
void set_dispatch_legacy(bool on);

// ---- small helpers -------------------------------------------------------------------------------------------------------------------------
inline float bf16r(float x) {  // round to the nearest bf16 (ties to even), back to fp32
  uint32_t u; __builtin_memcpy(&u, &x, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  float r; __builtin_memcpy(&r, &u, 4); return r;
}
void round_bf16(float* x, size_t n);                    // in place, parallel for large n
float bf16_to_f32(uint16_t b);
void bf16_to_f32_n(const uint16_t* src, size_t n, float* dst);
float e4m3_to_f32(uint8_t v);

// ---- attention ---------------------------------------------------------------------------------------------------------------------------
// out[m, h, :] = Σ_j softmax_j(q[m, h, :] · K[pos_j, g(h), :] · scale) · V[pos_j, g(h), :]  (fp32), g(h) = h / (Hq / Hkv).
//   K / V caches bf16 [T, Hkv, D] (row stride Hkv·D). The positions of row m: idx[m · idx_stride + j] for j < n_idx(m) (−1 entries skipped)
//   or, with idx == nullptr, the causal range [0, causal_end[m]). Parallel over (row, kv head).
void attention(const float* q, int M, int Hq, int Hkv, int D, const uint16_t* K, const uint16_t* V, const int32_t* idx, int idx_stride,
               const int32_t* causal_end, float scale, float* out);

}  // namespace hive::cpu
