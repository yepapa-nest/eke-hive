// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// G1 HIVE_DECODE_STEP_GRAPH — decode-layer experts without host synchronization (see the decode_handshake.h header comment).
//   hs_plan: routing (device) + slot table (device copy) → the same classification, group table and CPU job table as the default
//     moe_decode_experts · posts to the mailbox when needed.
//   hs_layer_experts: plan → resident groups (mx_grouped_w13/w2 + fixed-order accumulation) → wait for DMA → DMA groups → wait
//     for CPU → accumulate CPU results.
//   Every wait is bounded via %globaltimer (timeout = DevMisc.dev_err + Ctrl.err — the host checks after the step and absorbs it
//   by marking the sequence broken).
#include "hive/decode_handshake.h"
#include "hive/kernels.h"

namespace hive {
namespace k {
namespace {

__device__ __forceinline__ unsigned long long hs_now_ns() {
  uint64_t ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  return ns;
}
__device__ __forceinline__ uint32_t ld_volatile(const uint32_t* p) { return *reinterpret_cast<const volatile uint32_t*>(p); }
__device__ __forceinline__ void st_volatile(uint32_t* p, uint32_t v) { *reinterpret_cast<volatile uint32_t*>(p) = v; }
__device__ __forceinline__ bool seq_reached(uint32_t have, uint32_t want) { return (int32_t)(have - want) >= 0; }

constexpr int kPlanThreads = 256;

// One block. Builds in shared memory the same values as the default host loop (cnt · offsets · rows_by_e · resident/miss lists · sort · shares).
__global__ void __launch_bounds__(kPlanThreads) hs_plan_kernel(HsPlanArgs a) {
  __shared__ int32_t s_ids[hs::kMaxR];
  __shared__ float s_rw[hs::kMaxR];
  __shared__ int32_t s_rbe[hs::kMaxR];
  __shared__ int32_t s_cnt[hs::kMaxE], s_off[hs::kMaxE];
  __shared__ int32_t s_hit[hs::kMaxE], s_hslot[hs::kMaxE], s_miss[hs::kMaxE];
  __shared__ int s_nh, s_nm, s_err, s_post, s_njobs;
  __shared__ hs::LayerCounts s_c;
  const int tid = threadIdx.x, R = a.M * a.k, E = a.E;
  if (tid == 0) s_err = a.misc->dev_err;
  for (int i = tid; i < R; i += blockDim.x) { s_ids[i] = a.ids[i]; s_rw[i] = a.rw[i]; }
  for (int e = tid; e < E; e += blockDim.x) s_cnt[e] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) atomicAdd(&s_cnt[s_ids[i]], 1);
  __syncthreads();
  // warp 0: offsets in ascending expert order (exclusive scan) · resident/miss lists (same order as the default `for e: n_of(e) > 0 → slot ≥ 0 ? hit : miss`)
  if (tid < 32) {
    const int lane = tid;
    int carry = 0, nh = 0, nm = 0;
    for (int base = 0; base < E; base += 32) {
      const int e = base + lane;
      const int c = e < E ? s_cnt[e] : 0;
      const int sl = (e < E && c > 0) ? a.slot_row[e] : -1;
      int inc = c;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, inc, d); if (lane >= d) inc += v; }
      if (e < E) s_off[e] = carry + inc - c;
      carry += __shfl_sync(0xffffffffu, inc, 31);
      const unsigned lt = (1u << lane) - 1u;
      const unsigned hb = __ballot_sync(0xffffffffu, c > 0 && sl >= 0), mb = __ballot_sync(0xffffffffu, c > 0 && sl < 0);
      if (c > 0 && sl >= 0) { const int p = nh + __popc(hb & lt); s_hit[p] = e; s_hslot[p] = sl; }
      if (c > 0 && sl < 0) s_miss[nm + __popc(mb & lt)] = e;
      nh += __popc(hb); nm += __popc(mb);
    }
    if (lane == 0) { s_nh = nh; s_nm = nm; }
  }
  __syncthreads();
  // rows_by_e: as the default `rows_by_e[fill[id]++] = i` (ascending i) = off[e] + (number of rows before i with the same expert)
  for (int i = tid; i < R; i += blockDim.x) {
    const int e = s_ids[i];
    int rank = 0;
    for (int j = 0; j < i; ++j) rank += s_ids[j] == e;
    s_rbe[s_off[e] + rank] = i;
  }
  __syncthreads();
  if (tid == 0) {
    const bool err = s_err != 0;
    // if an earlier layer timed out (the step will end as failed), do not handle misses — so the mailbox (the dispatcher may still be reading it) is left untouched
    const int nm = err ? 0 : s_nm;
    const int share = err ? 0 : a.prm->share[s_nm < hs::kMaxE ? s_nm : hs::kMaxE];
    hs::PostHdr* hdr = a.hdr;
    // same functions as the sort, split, add_group and CPU-job loop of the default moe_decode_experts (shared with host replay and the CPU test — decode_handshake.h)
    const hs::LayerCounts c = hs::build_tables(s_cnt, s_off, s_rbe, s_rw, a.k, s_hit, s_hslot, s_nh, s_miss, nm, share, a.slots_base, a.staging_base, a.rec,
                                               a.prm->stage_base, &a.misc->stage_ctr, a.prm->n_staging, a.gd, a.gd + a.Gh, a.g_rows, a.g_rw, a.cpu_rows,
                                               a.jobs, hdr->dma);
    *a.cnt = c;
    s_c = c;
    s_njobs = c.n_jobs;
    s_post = !err && (c.n_jobs > 0 || c.n_dma > 0);
    if (s_post) { hdr->seq = a.prm->seq_base + (uint32_t)a.l + 1u; hdr->layer = a.l; hdr->M = a.M; hdr->n_jobs = c.n_jobs; hdr->n_dma = c.n_dma; hdr->job_rows = c.job_rows; }
  }
  __syncthreads();
  // empty group entries (n = 0 — the kernel exits immediately) · log
  {
    const hs::LayerCounts c = s_c;
    for (int g = c.ng_hit + tid; g < a.Gh; g += blockDim.x) { GroupDesc d{}; d.n = 0; a.gd[g] = d; }
    for (int g = c.ng_str + tid; g < a.Gd; g += blockDim.x) { GroupDesc d{}; d.n = 0; a.gd[a.Gh + g] = d; }
    for (int i = tid; i < R; i += blockDim.x) a.route_log_l[i] = s_ids[i];
    if (tid == 0) *a.counts_log_l = c;
  }
  if (s_post) {
    if (s_njobs > 0) {  // activations for the CPU (all rows — the default path also expanded all M rows to fp32)
      const size_t nq = (size_t)a.M * a.dim, ns = nq / 32;
      const uint4* xq_src = reinterpret_cast<const uint4*>(a.xq);
      uint4* xq_dst = reinterpret_cast<uint4*>(a.bxq);
      for (size_t i = tid; i < nq / 16; i += blockDim.x) xq_dst[i] = xq_src[i];
      for (size_t i = tid; i < ns; i += blockDim.x) a.bxs[i] = a.xs[i];
    }
    __threadfence_system();  // after each thread has made its own writes (tables, activations) visible to the host
    __syncthreads();
    if (tid == 0) {
      a.time_log_l[0] = hs_now_ns();
      st_volatile(&a.ctrl->post_seq, a.prm->seq_base + (uint32_t)a.l + 1u);  // post (last)
      __threadfence_system();
    }
  } else if (tid == 0) {
    a.time_log_l[0] = hs_now_ns();
  }
}

__global__ void hs_step_reset_kernel(hs::DevMisc* misc) { misc->dev_err = 0; misc->stage_ctr = 0; }

// If there is a DMA share, wait for the dispatcher's signal (hs_signal after the side-stream copy)
__global__ void hs_wait_dma_kernel(const hs::Params* prm, hs::DevMisc* misc, const hs::LayerCounts* cnt, hs::Ctrl* ctrl, int l) {
  if (cnt->ng_str == 0 || ld_volatile(reinterpret_cast<const uint32_t*>(&misc->dev_err)) != 0) return;
  const uint32_t want = prm->seq_base + (uint32_t)l + 1u;
  const unsigned long long t0 = hs_now_ns(), lim = prm->timeout_ns;
  while (!seq_reached(ld_volatile(&misc->dma_flag), want)) {
    if (hs_now_ns() - t0 > lim) {
      misc->dev_err = 2;
      st_volatile(&ctrl->err, ld_volatile(&ctrl->err) | 2u);  // no atomics on mapped memory (no PCIe AtomicOps needed — by stream order there is a single writer)
      if (ld_volatile(&ctrl->err_seq) == 0) st_volatile(&ctrl->err_seq, want);
      __threadfence_system();
      return;
    }
    __nanosleep(500);
  }
  __threadfence();
}

// If there are CPU jobs, wait for done_seq. Records the time on entry (in the default path dma_e1_ = right after DMA group accumulation).
__global__ void hs_wait_cpu_kernel(const hs::Params* prm, hs::DevMisc* misc, const hs::LayerCounts* cnt, hs::Ctrl* ctrl, unsigned long long* time_log_l,
                                   int l) {
  time_log_l[1] = hs_now_ns();
  if (cnt->n_jobs == 0 || ld_volatile(reinterpret_cast<const uint32_t*>(&misc->dev_err)) != 0) return;
  const uint32_t want = prm->seq_base + (uint32_t)l + 1u;
  const unsigned long long t0 = hs_now_ns(), lim = prm->timeout_ns;
  while (!seq_reached(ld_volatile(&ctrl->done_seq), want)) {
    if (hs_now_ns() - t0 > lim) {
      misc->dev_err = 1;
      st_volatile(&ctrl->err, ld_volatile(&ctrl->err) | 1u);
      if (ld_volatile(&ctrl->err_seq) == 0) st_volatile(&ctrl->err_seq, want);
      __threadfence_system();
      return;
    }
    __nanosleep(1000);
  }
  __threadfence_system();
}

// Same formula and addition order (ascending row r) as accum_bf16_rows_seq_kernel — only r0/R are read from device counts. which 0 = resident [0, R_hit) · 1 = DMA [R_hit, R_hit+R_str)
__global__ void hs_accum_seq_kernel(const bf16* __restrict__ eout, const int32_t* __restrict__ rows, const hs::LayerCounts* __restrict__ cnt, int which, int M,
                                    int dim, float* __restrict__ acc) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * dim) return;
  const int r0 = which == 0 ? 0 : cnt->R_hit, R = which == 0 ? cnt->R_hit : cnt->R_str;
  if (R <= 0) return;  // default path: accum_bf16_rows_seq is not launched for R ≤ 0 (acc unchanged)
  const int m = (int)(i / dim), d = (int)(i % dim);
  const bf16* src = eout + (size_t)r0 * dim;
  const int32_t* rm = rows + r0;
  float s = acc[i];
  for (int r = 0; r < R; ++r)
    if (rm[r] == m) s += bf2f(src[(size_t)r * dim + d]);
  acc[i] = s;
}

// Accumulate CPU results — fixed row order (ascending job row). The default accum_f32_rows uses atomic adds, so when a row had
//   more than one CPU result the addition order varied between runs (this is one of those possible orders). The results are in
//   mapped pinned memory (just written by the host), hence ld.global.cv (treat cached system-memory lines as stale and re-read).
__global__ void hs_accum_cpu_kernel(const float* __restrict__ cpu_out, const int32_t* __restrict__ cpu_rows, const hs::LayerCounts* __restrict__ cnt,
                                    const hs::DevMisc* __restrict__ misc, int M, int dim, float* __restrict__ acc) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * dim) return;
  const int n = cnt->job_rows;
  if (n <= 0 || misc->dev_err != 0) return;
  const int m = (int)(i / dim), d = (int)(i % dim);
  float s = acc[i];
  for (int r = 0; r < n; ++r)
    if (cpu_rows[r] == m) s += __ldcv(cpu_out + (size_t)r * dim + d);
  acc[i] = s;
}

__global__ void hs_signal_kernel(uint32_t* flag, uint32_t value) {
  __threadfence();
  st_volatile(flag, value);
  __threadfence();
}

// S1 HIVE_DECODE_EARLY_ROUTE: the same copy as the routing half of route_to_host_kernel (int32/fp32 as is) + a post number. The
//   number comes from a device counter (a graph captures arguments by value, so a value that differs per replay must come from
//   device memory). Each thread makes its writes visible at system scope, then barrier → thread 0 writes the number.
__global__ void er_route_post_kernel(const int32_t* __restrict__ ids, const float* __restrict__ rw, int n, int32_t* __restrict__ ids_h, float* __restrict__ rw_h,
                                     uint32_t* __restrict__ ctr, uint32_t* __restrict__ flag_h) {
  for (int i = (int)threadIdx.x; i < n; i += blockDim.x) { ids_h[i] = ids[i]; rw_h[i] = rw[i]; }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0u) {
    const uint32_t v = *ctr + 1u;
    *ctr = v;
    st_volatile(flag_h, v);
    __threadfence_system();
  }
}

}  // namespace

void er_route_post(const int32_t* ids, const float* rw, int n, int32_t* ids_h, float* rw_h, uint32_t* ctr, uint32_t* flag_h, cudaStream_t st) {
  er_route_post_kernel<<<1, 128, 0, st>>>(ids, rw, n, ids_h, rw_h, ctr, flag_h);
  CUDA_CHECK(cudaGetLastError());
}

void hs_plan(const HsPlanArgs& a, cudaStream_t st) {
  HIVE_CHECK(a.E <= hs::kMaxE && a.M * a.k <= hs::kMaxR && a.dim % 16 == 0, "hs_plan bounds");
  hs_plan_kernel<<<1, kPlanThreads, 0, st>>>(a);
  CUDA_CHECK(cudaGetLastError());
}
void hs_step_reset(hs::DevMisc* misc, cudaStream_t st) { hs_step_reset_kernel<<<1, 1, 0, st>>>(misc); }
void hs_signal(uint32_t* flag, uint32_t value, cudaStream_t st) { hs_signal_kernel<<<1, 1, 0, st>>>(flag, value); }
void hs_preload(cudaStream_t side, cudaStream_t fused_st) {
  cudaFuncAttributes fa;
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hs_plan_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hs_step_reset_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hs_wait_dma_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hs_wait_cpu_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hs_accum_seq_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hs_accum_cpu_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hs_signal_kernel));
  uint32_t* tmp = nullptr;  // actually done once on the dispatcher stream (load + first-launch path)
  CUDA_CHECK(cudaMalloc((void**)&tmp, sizeof(uint32_t)));
  hs_signal_kernel<<<1, 1, 0, side>>>(tmp, 0u);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(side));
  CUDA_CHECK(cudaFree(tmp));
  moe_decode_fused_preload(fused_st);
  CUDA_CHECK(cudaStreamSynchronize(fused_st));
}

void hs_layer_experts(const HsLayerArgs& a, cudaStream_t st) {
  const HsPlanArgs& p = a.plan;
  const int M = p.M, dim = p.dim, I = a.I;
  const int nthr = 256, nblk = (int)(((size_t)M * dim + nthr - 1) / nthr);
  hs_plan(p, st);
  // Group execution: same choice as run_groups in the default moe_decode_experts — D1 fusion (one launch) or the plain chain (w13 → w2 → fixed-order accumulation)
  const int* cb = reinterpret_cast<const int*>(p.cnt);  // LayerCounts = {ng_hit, ng_str, R_hit, R_str, …}
  auto run = [&](const GroupDesc* g, int gmax, int which) {
    if (a.fused) {
      const DfDevCount dc{cb, which == 0 ? 0 : 1, which == 0 ? -1 : 2, which == 0 ? 2 : 3};
      moe_decode_fused_dev(g, gmax, dc, p.xq, p.xs, p.g_rows, p.g_rw, M, dim, I, a.limit, a.y, a.yq, a.ys, a.eout, a.acc, st);
      return;
    }
    mx_grouped_w13(g, gmax, p.xq, p.xs, I, dim, p.g_rw, a.limit, a.y, st, p.g_rows, a.yq, a.ys);  // empty entries (n = 0) exit at the kernel head
    mx_grouped_w2(g, gmax, a.yq, a.ys, dim, I, a.eout, st);
    hs_accum_seq_kernel<<<nblk, nthr, 0, st>>>(a.eout, p.g_rows, p.cnt, which, M, dim, a.acc);
  };
  run(p.gd, p.Gh, 0);  // resident groups [0, Gh)
  // DMA groups [Gh, Gh+Gd) — after the dispatcher signals that the staging copy is done
  hs_wait_dma_kernel<<<1, 1, 0, st>>>(p.prm, p.misc, p.cnt, p.ctrl, p.l);
  run(p.gd + p.Gh, p.Gd, 1);
  // CPU results
  hs_wait_cpu_kernel<<<1, 1, 0, st>>>(p.prm, p.misc, p.cnt, p.ctrl, p.time_log_l, p.l);
  hs_accum_cpu_kernel<<<nblk, nthr, 0, st>>>(a.cpu_out_d, p.cpu_rows, p.cnt, p.misc, M, dim, a.acc);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace k
}  // namespace hive
