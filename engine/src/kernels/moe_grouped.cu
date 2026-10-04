// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Grouped prefill experts (HIVE_GROUPED_PREFILL) — batch executor + small kernels (row-map gather, swiglu + quantization, fixed-order accumulation).
//   The default prefill loop (runtime.cpp moe_experts_multi) launches, per expert, 2 gathers → gemm w1/w3 → swiglu → quantize → gemm w2 → accumulate
//   back to back on one stream (16K chunk: ~256 rows per expert on average but skewed, many under 128 rows → grids of 18/40 blocks queued one after
//   another on a large GPU). Here one batch (dozens of experts, rows <= work buffer capacity) is run with one launch per stage.
//   Values: each expert GEMM call (seg) sees **the same row set and row count** as the per-expert path, so the kernel gemm_bs would choose (gemv, mx tiled,
//   WMMA) is used as is (the grouped version has independent per-row mmas with the same k order); the swiglu and quantization formulas are the same; and
//   accumulation adds once per destination row in fp32, in the per-expert path's expert order → bit-identical.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "hive/kernels.h"

namespace hive::k {

namespace {

constexpr float kFp8MaxInvG = 1.0f / 448.0f;  // same value as quant.cu kFp8MaxInv

__global__ void gather_rows_ref_kernel(const uint8_t* const* __restrict__ xq_base, const uint8_t* const* __restrict__ xs_base,
                                       const int32_t* __restrict__ rref, int r0, int n, int dim, uint8_t* __restrict__ dq, uint8_t* __restrict__ ds) {
  const int nbq = dim / 16, nbs = dim / 32;  // 16 B units (xq), byte units (xs)
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nq = (size_t)n * nbq;
  if (i < nq) {
    const int r = r0 + (int)(i / nbq), c = (int)(i % nbq);
    const int v = rref[r];
    const uint4* src = reinterpret_cast<const uint4*>(xq_base[v >> 24] + (size_t)(v & 0xFFFFFF) * dim);
    reinterpret_cast<uint4*>(dq + (size_t)r * dim)[c] = src[c];
    return;
  }
  const size_t j = i - nq;
  if (j >= (size_t)n * nbs) return;
  const int r = r0 + (int)(j / nbs), c = (int)(j % nbs);
  const int v = rref[r];
  ds[(size_t)r * nbs + c] = xs_base[v >> 24][(size_t)(v & 0xFFFFFF) * nbs + c];
}

// Thread = (row, 32-element block). r < r1: same formula as swiglu_route_kernel → bf16 → same formula as act_quant_fp8_kernel<false>. r >= r1: quantize y as is.
__global__ void swiglu_quant_rows_kernel(const bf16* __restrict__ gate, const bf16* __restrict__ up, const bf16* __restrict__ y, const float* __restrict__ rw,
                                         int r0, int r1, int r2, int I, float limit, uint8_t* __restrict__ yq, uint8_t* __restrict__ ys) {
  const int nb = I / 32;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)(r2 - r0) * nb) return;
  const int r = r0 + (int)(idx / nb), b = (int)(idx % nb);
  const size_t o = (size_t)r * I + (size_t)b * 32;
  float v[32];
  float amax = 0.f;
  if (r < r1) {
    const float rwv = rw[r];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      float g = bf2f(gate[o + i]), u = bf2f(up[o + i]);
      if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
      float t = g / (1.f + expf(-g)) * u;
      t *= rwv;
      v[i] = bf2f(f2bf(t));
      amax = fmaxf(amax, fabsf(v[i]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < 32; ++i) { v[i] = bf2f(y[o + i]); amax = fmaxf(amax, fabsf(v[i])); }
  }
  amax = fmaxf(amax, 1e-4f);
  const uint8_t code = f32_ceil_pow2_e8m0(amax * kFp8MaxInvG);
  const float sc = e8m0_to_f32(code);
  uint8_t* q = yq + o;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    float t = fminf(fmaxf(v[i] / sc, -448.f), 448.f);
    q[i] = f32_to_e4m3(t);
  }
  ys[(size_t)r * nb + b] = code;
}

// Block = one destination, threads split the columns. Addition order = contrib order (= the per-expert path's expert order).
__global__ void accum_ordered_rows_kernel(const AccDest* __restrict__ dests, const int32_t* __restrict__ contrib, const bf16* __restrict__ src, int dim) {
  const AccDest d = dests[blockIdx.x];
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    float s = d.acc[c];
    for (int i = 0; i < d.n; ++i) s += bf2f(src[(size_t)contrib[d.c0 + i] * dim + c]);
    d.acc[c] = s;
  }
}

inline size_t align16(size_t x) { return (x + 15) & ~(size_t)15; }

}  // namespace

void gather_rows_ref(const uint8_t* const* xq_base, const uint8_t* const* xs_base, const int32_t* rref, int r0, int n, int dim, uint8_t* dq, uint8_t* ds,
                     cudaStream_t st) {
  if (n <= 0) return;
  HIVE_CHECK(dim % 32 == 0, "gather_rows_ref dim");
  const size_t tot = (size_t)n * (dim / 16 + dim / 32);
  gather_rows_ref_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(xq_base, xs_base, rref, r0, n, dim, dq, ds);
}

void swiglu_quant_rows(const bf16* gate, const bf16* up, const bf16* y, const float* rw, int r0, int r1, int r2, int I, float limit, uint8_t* yq, uint8_t* ys,
                       cudaStream_t st) {
  if (r2 <= r0) return;
  HIVE_CHECK(I % 32 == 0 && r0 <= r1 && r1 <= r2, "swiglu_quant_rows shape");
  const size_t n = (size_t)(r2 - r0) * (I / 32);
  const unsigned grid = (unsigned)((n + 127) / 128);
  swiglu_quant_rows_kernel<<<grid, 128, 0, st>>>(gate, up, y, rw, r0, r1, r2, I, limit, yq, ys);
}

void accum_ordered_rows(const AccDest* dests, int ndest, const int32_t* contrib, const bf16* src, int dim, cudaStream_t st) {
  if (ndest <= 0) return;
  accum_ordered_rows_kernel<<<ndest, 256, 0, st>>>(dests, contrib, src, dim);
}

// ---- Batch executor ----------------------------------------------------------------------------------------------------------
bool GroupedMoe::supported(int dim, int I) {
  if (I % 32 != 0 || dim % 32 != 0) return false;
  auto ok = [](int r) { return r == kGemmMxTiled || r == kGemmTc; };
  return ok(gemm_bs_route(true, 9, dim)) && ok(gemm_bs_route(true, 9, I));  // the path for n > 8 does not depend on the row count (the selection only tests M > 8)
}

GroupedMoe::GroupedMoe(int max_rows, int max_subs) : max_rows_(std::max(1, max_rows)), max_subs_(std::max(1, max_subs)) {
  // Table bounds (one batch): row tiles + gemv groups <= rows/128 + seg count <= rows/128 + rows, destinations <= rows, rref/rw/contrib = rows
  const size_t R = (size_t)max_rows_;
  region_ = align16(2 * (size_t)max_subs_ * sizeof(void*)) + align16((R + R / 128 + 4) * sizeof(GroupDesc)) + align16(R * sizeof(AccDest)) +
            3 * align16(R * 4) + 64;
  for (int i = 0; i < kRing; ++i) {
    CUDA_CHECK(cudaHostAlloc((void**)&host_[i], region_, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc((void**)&dev_[i], region_));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_[i], cudaEventDisableTiming));
  }
}

GroupedMoe::~GroupedMoe() {
  for (int i = 0; i < kRing; ++i) {
    if (done_[i]) { cudaEventSynchronize(done_[i]); cudaEventDestroy(done_[i]); }
    if (host_[i]) cudaFreeHost(host_[i]);
    if (dev_[i]) cudaFree(dev_[i]);
  }
}

void GroupedMoe::run(const GroupedSeg* segs, int nseg, const GroupedRef* refs, const uint8_t* const* xq_sub, const uint8_t* const* xs_sub, float* const* acc_sub,
                     int n_sub, const int* sub_rows, int dim, int I, float limit, const GroupedBufs& b, cudaStream_t st,
                     const std::function<void()>& before_late) {
  if (nseg <= 0) return;
  HIVE_CHECK(n_sub >= 1 && n_sub <= max_subs_ && n_sub < 128, "grouped moe subs");
  const int route13 = gemm_bs_route(true, 9, dim), route2 = gemm_bs_route(true, 9, I);
  HIVE_CHECK((route13 == kGemmMxTiled || route13 == kGemmTc) && (route2 == kGemmMxTiled || route2 == kGemmTc), "grouped moe route");
  // 1) Gathered row layout: part 0 (resident — computed immediately), part 1 (late — after the copy completes). Each part is [row-tile segs | gemv segs].
  int R = 0;
  row_of_seg_.assign(nseg, 0);
  int part_a[2], part_b[2], part_c[2], nt[2] = {0, 0}, ng[2] = {0, 0};
  for (int p = 0; p < 2; ++p) {
    part_a[p] = R;
    for (int i = 0; i < nseg; ++i)
      if (segs[i].late == (p == 1) && segs[i].n > 8) { row_of_seg_[i] = R; R += segs[i].n; nt[p] += (segs[i].n + 127) / 128; }
    part_b[p] = R;
    for (int i = 0; i < nseg; ++i)
      if (segs[i].late == (p == 1) && segs[i].n <= 8 && segs[i].n > 0) { row_of_seg_[i] = R; R += segs[i].n; ++ng[p]; }
    part_c[p] = R;
  }
  HIVE_CHECK(R <= max_rows_, "grouped moe batch rows");  // prevents overflowing the work buffers (gate/up/y/yq/eout/gxq) — the planner (moe_grouped_plan.h) cuts batches at rows <= capacity
  if (R == 0) return;
  for (int i = 0; i < nseg; ++i) HIVE_CHECK(gemm_bs_route(true, segs[i].n, dim) == (segs[i].n > 8 ? route13 : kGemmGemv), "grouped moe seg route");
  // 2) Contribution list per destination (sub-chunk row) — in seg order (= the per-expert path's expert order)
  sub_off_.assign(n_sub + 1, 0);
  for (int s = 0; s < n_sub; ++s) sub_off_[s + 1] = sub_off_[s] + std::max(0, sub_rows[s]);
  if ((int)stamp_.size() < sub_off_[n_sub]) { stamp_.assign(sub_off_[n_sub], 0); dslot_.assign(sub_off_[n_sub], 0); cur_stamp_ = 0; }
  if (++cur_stamp_ == 0x7FFFFFFF) { std::fill(stamp_.begin(), stamp_.end(), 0); cur_stamp_ = 1; }
  cnt_.clear();
  int nd = 0;
  for (int i = 0; i < nseg; ++i)
    for (int t = 0; t < segs[i].n; ++t) {
      const GroupedRef& rf = refs[segs[i].i0 + t];
      HIVE_CHECK(rf.s >= 0 && rf.s < n_sub && rf.m >= 0 && rf.m < sub_rows[rf.s] && rf.m < (1 << 24), "grouped moe ref");
      const int key = sub_off_[rf.s] + rf.m;
      if (stamp_[key] != cur_stamp_) { stamp_[key] = cur_stamp_; dslot_[key] = nd++; cnt_.push_back(0); }
      ++cnt_[dslot_[key]];
    }
  // 3) Table ring slot — the host may only overwrite it once the slot's previous copy has finished (usually already done: 4 ring slots = 4 batches ahead of the GPU)
  const int ring = next_++ % kRing;
  if (used_[ring]) CUDA_CHECK(cudaEventSynchronize(done_[ring]));
  uint8_t* H = host_[ring];
  size_t off = 0;
  auto take = [&](size_t bytes) { const size_t o = off; off += align16(bytes); return o; };
  const size_t o_xq = take(sizeof(void*) * n_sub), o_xs = take(sizeof(void*) * n_sub);
  const size_t o_tiles = take(sizeof(GroupDesc) * (nt[0] + nt[1])), o_gemv = take(sizeof(GroupDesc) * (ng[0] + ng[1]));
  const size_t o_dest = take(sizeof(AccDest) * nd), o_rref = take(4 * (size_t)R), o_rw = take(4 * (size_t)R), o_con = take(4 * (size_t)R);
  HIVE_CHECK(off <= region_, "grouped moe table size");
  std::memcpy(H + o_xq, xq_sub, sizeof(void*) * n_sub);
  std::memcpy(H + o_xs, xs_sub, sizeof(void*) * n_sub);
  GroupDesc* tiles = reinterpret_cast<GroupDesc*>(H + o_tiles);
  GroupDesc* gemv = reinterpret_cast<GroupDesc*>(H + o_gemv);
  AccDest* dest = reinterpret_cast<AccDest*>(H + o_dest);
  int32_t* rref = reinterpret_cast<int32_t*>(H + o_rref);
  float* rw = reinterpret_cast<float*>(H + o_rw);
  int32_t* con = reinterpret_cast<int32_t*>(H + o_con);
  int it = 0, ig = 0;
  for (int p = 0; p < 2; ++p) {  // row tiles → gemv groups, in part order (part 0 at the front of the table)
    for (int i = 0; i < nseg; ++i) {
      const GroupedSeg& g = segs[i];
      if (g.late != (p == 1) || g.n <= 8) continue;
      for (int m0 = 0; m0 < g.n; m0 += 128) tiles[it++] = GroupDesc{g.w1, g.s1, g.w3, g.s3, g.w2, g.s2, row_of_seg_[i] + m0, std::min(128, g.n - m0)};
    }
    for (int i = 0; i < nseg; ++i) {
      const GroupedSeg& g = segs[i];
      if (g.late != (p == 1) || g.n > 8 || g.n <= 0) continue;
      gemv[ig++] = GroupDesc{g.w1, g.s1, g.w3, g.s3, g.w2, g.s2, row_of_seg_[i], g.n};
    }
  }
  {
    int c0 = 0;
    for (int d = 0; d < nd; ++d) { dest[d].c0 = c0; dest[d].n = 0; c0 += cnt_[d]; }
    for (int i = 0; i < nseg; ++i)
      for (int t = 0; t < segs[i].n; ++t) {
        const GroupedRef& rf = refs[segs[i].i0 + t];
        const int r = row_of_seg_[i] + t;
        rref[r] = (rf.s << 24) | rf.m;
        rw[r] = rf.rw;
        const int d = dslot_[sub_off_[rf.s] + rf.m];
        dest[d].acc = acc_sub[rf.s] + (size_t)rf.m * dim;
        con[dest[d].c0 + dest[d].n++] = r;
      }
  }
  uint8_t* D = dev_[ring];
  CUDA_CHECK(cudaMemcpyAsync(D, H, off, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaEventRecord(done_[ring], st));
  used_[ring] = true;
  const uint8_t* const* xq_d = reinterpret_cast<const uint8_t* const*>(D + o_xq);
  const uint8_t* const* xs_d = reinterpret_cast<const uint8_t* const*>(D + o_xs);
  const GroupDesc* tiles_d = reinterpret_cast<const GroupDesc*>(D + o_tiles);
  const GroupDesc* gemv_d = reinterpret_cast<const GroupDesc*>(D + o_gemv);
  const int32_t* rref_d = reinterpret_cast<const int32_t*>(D + o_rref);
  const float* rw_d = reinterpret_cast<const float*>(D + o_rw);
  // 4) Per part: [gemv row gather → gemv w13 (+swiglu)] · [grouped GEMM w1‖w3] → swiglu + quantization → [grouped GEMM w2] · [gemv w2]
  int t_off = 0, g_off = 0;
  for (int p = 0; p < 2; ++p) {
    if (part_c[p] == part_a[p]) continue;
    if (p == 1 && before_late) before_late();
    if (ng[p] > 0) {
      gather_rows_ref(xq_d, xs_d, rref_d, part_b[p], part_c[p] - part_b[p], dim, b.gxq, b.gxs, st);
      gemv_grouped_w13(gemv_d + g_off, ng[p], b.gxq, b.gxs, I, dim, rw_d, limit, b.y, st);
    }
    if (nt[p] > 0) gemm_fp4_grouped(route13, tiles_d + t_off, nt[p], 0, xq_d, xs_d, rref_d, nullptr, nullptr, I, dim, b.gate, b.up, st);
    swiglu_quant_rows(b.gate, b.up, b.y, rw_d, part_a[p], part_b[p], part_c[p], I, limit, b.yq, b.ys, st);
    if (nt[p] > 0) gemm_fp4_grouped(route2, tiles_d + t_off, nt[p], 1, nullptr, nullptr, nullptr, b.yq, b.ys, dim, I, b.eout, nullptr, st);
    if (ng[p] > 0) gemv_grouped_w2(gemv_d + g_off, ng[p], b.yq, b.ys, dim, I, b.eout, st);
    t_off += nt[p]; g_off += ng[p];
  }
  // 5) Fixed-order accumulation (whole batch — seg order = the per-expert path's expert order)
  accum_ordered_rows(reinterpret_cast<const AccDest*>(D + o_dest), nd, reinterpret_cast<const int32_t*>(D + o_con), b.eout, dim, st);
  ++n_batches_;
}

}  // namespace hive::k
