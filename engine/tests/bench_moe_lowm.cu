// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Low-row decode expert timing: one fused launch (moe_decode_fused, the variant chosen by the environment or forced) for M rows over
//   ng distinct experts, with and without an L2 flush before each launch. Shape = a real DeepSeek-V4.1-Flash layer (5120 · 2304 · top-6).
//   Prints µs and effective GB/s per (M, groups) so the fixed cost of a launch can be separated from the per-expert cost
//   (fit t = a + b · groups over the M=1 cases). Weights and activations are random data in the storage format (values are not checked here —
//   test_decode_moe checks bit identity).
//   Run: ./build/bench_moe_lowm [reps=50] [variant: 0 = environment · 103/104/105 = FUSED3 S=3/4/5]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "hive/common.h"
#include "hive/expert_store.h"
#include "hive/kernels.h"

using namespace hive;

namespace {
constexpr int DIM = 5120, INTER = 2304, TOPK = 6;

__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed, int mode) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  uint8_t v = (uint8_t)h;
  if (mode == 1) v = (uint8_t)((v & 0x80) | ((v & 0x7F) % 0x48));
  else if (mode == 2) v = (uint8_t)(120 + (h >> 8) % 8);
  else if (mode == 3) v = (uint8_t)(118 + (h >> 8) % 6);
  p[i] = v;
}
void fill(void* p, size_t n, uint32_t seed, int mode) { fill_kernel<<<(unsigned)((n + 255) / 256), 256>>>((uint8_t*)p, n, seed, mode); }
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T))); return p; }
}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::max(3, atoi(argv[1])) : 50;
  const int force = argc > 2 ? atoi(argv[2]) : 0;
  if (force > 100) k::moe_decode_fused_force3(force - 100);
  const ExpertLayout lay = ExpertLayout::make(DIM, INTER);
  const int NE = 48;
  std::vector<uint8_t*> rec(NE);
  for (int e = 0; e < NE; ++e) {
    CUDA_CHECK(cudaMalloc((void**)&rec[e], lay.total));
    fill(rec[e] + lay.w1, (size_t)INTER * DIM / 2, 11 + e, 0); fill(rec[e] + lay.w3, (size_t)INTER * DIM / 2, 13 + e, 0);
    fill(rec[e] + lay.w2, (size_t)INTER * DIM / 2, 17 + e, 0);
    fill(rec[e] + lay.s1, (size_t)INTER * DIM / 32, 19 + e, 3); fill(rec[e] + lay.s3, (size_t)INTER * DIM / 32, 23 + e, 3);
    fill(rec[e] + lay.s2, (size_t)INTER * DIM / 32, 29 + e, 3);
  }
  const size_t flush_n = 256u << 20;
  uint8_t* flush = dmalloc<uint8_t>(flush_n);
  const int R = 8 * TOPK;
  uint8_t *xq = dmalloc<uint8_t>((size_t)8 * DIM), *xs = dmalloc<uint8_t>((size_t)8 * DIM / 32), *yq = dmalloc<uint8_t>((size_t)R * INTER),
          *ys = dmalloc<uint8_t>((size_t)R * INTER / 32), *tbl = dmalloc<uint8_t>(64 << 10);
  bf16 *y = dmalloc<bf16>((size_t)R * INTER), *eout = dmalloc<bf16>((size_t)R * DIM);
  float* acc = dmalloc<float>((size_t)8 * DIM);
  fill(xq, (size_t)8 * DIM, 7, 1); fill(xs, (size_t)8 * DIM / 32, 9, 2);
  CUDA_CHECK(cudaMemset(acc, 0, (size_t)8 * DIM * 4));
  cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  k::moe_decode_fused_preload(st);
  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  printf("variant %d · reps %d\n", k::moe_decode_fused_variant(), reps);
  printf("%-3s %-7s %-12s %-10s %-12s %-10s\n", "M", "groups", "flushed µs", "GB/s", "warm-L2 µs", "GB/s");
  struct Case { int M, ng; };
  const Case cases[] = {{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 12}, {4, 24}, {8, 48}};
  for (const Case& cs : cases) {
    // groups: each expert gets the rows that chose it (M rows, k experts per row → experts 0..ng−1, rows assigned round robin)
    std::vector<k::GroupDesc> gd; std::vector<int32_t> rows; std::vector<float> rw;
    for (int g = 0; g < cs.ng; ++g) {
      k::GroupDesc d{};
      const uint8_t* rp = rec[g % NE];
      d.w1 = rp + lay.w1; d.s1 = rp + lay.s1; d.w3 = rp + lay.w3; d.s3 = rp + lay.s3; d.w2 = rp + lay.w2; d.s2 = rp + lay.s2;
      d.row0 = (int)rows.size(); d.n = 1;
      rows.push_back(g % cs.M); rw.push_back(0.1f);
      gd.push_back(d);
    }
    const size_t off_rows = sizeof(k::GroupDesc) * gd.size(), off_rw = off_rows + rows.size() * 4, total = off_rw + rw.size() * 4;
    std::vector<uint8_t> h(total);
    std::memcpy(h.data(), gd.data(), off_rows); std::memcpy(h.data() + off_rows, rows.data(), rows.size() * 4); std::memcpy(h.data() + off_rw, rw.data(), rw.size() * 4);
    CUDA_CHECK(cudaMemcpy(tbl, h.data(), total, cudaMemcpyHostToDevice));
    const auto* g = reinterpret_cast<const k::GroupDesc*>(tbl);
    const auto* rp = reinterpret_cast<const int32_t*>(tbl + off_rows);
    const auto* wp = reinterpret_cast<const float*>(tbl + off_rw);
    double t[2] = {0, 0};
    for (int mode = 0; mode < 2; ++mode)
      for (int rep = 0; rep < reps; ++rep) {
        if (mode == 0) CUDA_CHECK(cudaMemsetAsync(flush, rep & 0xFF, flush_n, st));
        float ms = 0;
        CUDA_CHECK(cudaEventRecord(e0, st));
        k::moe_decode_fused(g, cs.ng, xq, xs, rp, wp, 0, (int)rows.size(), cs.M, DIM, INTER, 10.f, y, yq, ys, eout, acc, st);
        CUDA_CHECK(cudaEventRecord(e1, st));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (rep) t[mode] += ms;
      }
    const double mb = cs.ng * 3.0 * ((double)INTER * DIM / 2 + (double)INTER * DIM / 32) / 1e6;
    const double us0 = t[0] / (reps - 1) * 1000, us1 = t[1] / (reps - 1) * 1000;
    printf("%-3d %-7d %-12.1f %-10.0f %-12.1f %-10.0f\n", cs.M, cs.ng, us0, mb / us0 * 1e3, us1, mb / us1 * 1e3);
  }
  return 0;
}
