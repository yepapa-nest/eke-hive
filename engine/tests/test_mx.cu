// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Measures the fragment layout of mma.sync kind::mxf8f6f4.block_scale — verified with single elements instead of assumptions.
//   ① e2m1 container bit position (<<0 / <<2 / <<4) ② A/B/D fragment layout ③ scale thread layout.  Usage: ./build/test_mx
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hive/common.h"

namespace {
#define MX_U32(x) "r"(x)
__device__ __forceinline__ void mma_mx(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sfa, uint32_t sfb) {
  const uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : MX_U32(a[0]), MX_U32(a[1]), MX_U32(a[2]), MX_U32(a[3]), MX_U32(b[0]), MX_U32(b[1]), MX_U32(sfa), "h"(zero), "h"(zero), MX_U32(sfb), "h"(zero), "h"(zero));
}
#undef MX_U32

// Thread t's registers come straight from the host-provided table (no layout assumption). Returns the 4 D values as is.
__global__ void k_one(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B, const uint32_t* __restrict__ SFA,
                      const uint32_t* __restrict__ SFB, float* __restrict__ D) {
  const int t = threadIdx.x;
  uint32_t a[4] = {A[t * 4], A[t * 4 + 1], A[t * 4 + 2], A[t * 4 + 3]};
  uint32_t b[2] = {B[t * 2], B[t * 2 + 1]};
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  mma_mx(d, a, b, SFA[t], SFB[t]);
  for (int i = 0; i < 4; ++i) D[t * 4 + i] = d[i];
}
}  // namespace

int main() {
  uint32_t *dA, *dB, *dSA, *dSB;
  float* dD;
  CUDA_CHECK(cudaMalloc(&dA, 32 * 4 * 4)); CUDA_CHECK(cudaMalloc(&dB, 32 * 2 * 4));
  CUDA_CHECK(cudaMalloc(&dSA, 32 * 4)); CUDA_CHECK(cudaMalloc(&dSB, 32 * 4)); CUDA_CHECK(cudaMalloc(&dD, 32 * 4 * 4));
  auto run = [&](const std::vector<uint32_t>& A, const std::vector<uint32_t>& B, const std::vector<uint32_t>& SA, const std::vector<uint32_t>& SB,
                 std::vector<float>& D) {
    CUDA_CHECK(cudaMemcpy(dA, A.data(), 32 * 16, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dB, B.data(), 32 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSA, SA.data(), 32 * 4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dSB, SB.data(), 32 * 4, cudaMemcpyHostToDevice));
    k_one<<<1, 32>>>(dA, dB, dSA, dSB, dD);
    CUDA_CHECK(cudaDeviceSynchronize());
    D.resize(128);
    CUDA_CHECK(cudaMemcpy(D.data(), dD, 128 * 4, cudaMemcpyDeviceToHost));
  };
  std::vector<uint32_t> A(128, 0), B(64, 0), SA(32, 127), SB(32, 127);
  std::vector<float> D;
  // ① container bits: A = byte 0 of thread 0's a0 = e4m3 1.0 (0x38) · B = e2m1 code 2 (=1.0) at byte 0 of thread 0's b0, per shift
  for (int sh : {0, 2, 4}) {
    std::fill(A.begin(), A.end(), 0); std::fill(B.begin(), B.end(), 0);
    A[0] = 0x38u;                 // t0 a0 byte0
    B[0] = (uint32_t)(0x2u << sh);  // t0 b0 byte0
    run(A, B, SA, SB, D);
    int nz = 0; float v = 0; int where = -1;
    for (int i = 0; i < 128; ++i) if (D[i] != 0.f) { ++nz; v = D[i]; where = i; }
    printf("e2m1 shift %d: nonzero %d, value %g at D[t=%d, i=%d]\n", sh, nz, v, where / 4, where % 4);
  }
  // ② A fragment layout: put one A value 1.0 in register r, byte c of thread t, and B = 1.0 for the whole column (all threads b0,b1 = 4 bytes of 1.0) → position in D = (row, all cols)
  //    B all 1.0: every (k, n) is 1 → D[m][n] = Σ_k A[m][k] = 1 in all columns of row m
  const uint32_t one4 = 0u;  // filled in below
  (void)one4;
  const int shift_ok = 2;   // adjust to the result of ① above (default assumption 2)
  auto fill_B_ones = [&](std::vector<uint32_t>& Bv, int sh) { uint32_t byte = (0x2u << sh); uint32_t w = byte | (byte << 8) | (byte << 16) | (byte << 24); std::fill(Bv.begin(), Bv.end(), w); };
  printf("A fragment (thread t, register r, byte c) → D row:\n");
  for (int t : {0, 1, 4, 5}) for (int r = 0; r < 4; ++r) for (int c : {0, 3}) {
    std::fill(A.begin(), A.end(), 0); fill_B_ones(B, shift_ok);
    A[t * 4 + r] = 0x38u << (8 * c);
    run(A, B, SA, SB, D);
    int rows[16] = {0}; int cnt = 0;
    for (int i = 0; i < 128; ++i) if (D[i] != 0.f) { int tt = i / 4, ii = i % 4; int row = tt / 4 + (ii >= 2 ? 8 : 0); rows[row] = 1; ++cnt; }
    printf("  t=%2d r=%d c=%d → nonzero %d, rows:", t, r, c, cnt);
    for (int m = 0; m < 16; ++m) if (rows[m]) printf(" %d", m);
    printf("\n");
  }
  // ③ B fragment layout: put one B value 1.0 in b register r, byte c of thread t, and A = all 1.0 → column n of D
  printf("B fragment (thread t, register r, byte c) → D column:\n");
  for (int t : {0, 1, 2, 3, 4, 8}) for (int r = 0; r < 2; ++r) for (int c : {0, 3}) {
    std::fill(A.begin(), A.end(), 0x38383838u); std::fill(B.begin(), B.end(), 0);
    B[t * 2 + r] = (0x2u << shift_ok) << (8 * c);
    run(A, B, SA, SB, D);
    int cols[8] = {0}; int cnt = 0;
    for (int i = 0; i < 128; ++i) if (D[i] != 0.f) { int tt = i / 4, ii = i % 4; int col = (tt % 4) * 2 + (ii & 1); cols[col] = 1; ++cnt; }
    printf("  t=%2d r=%d c=%d → nonzero %d, cols:", t, r, c, cnt);
    for (int n = 0; n < 8; ++n) if (cols[n]) printf(" %d", n);
    printf("\n");
  }
  // ④ scale layout: A·B all 1.0, all SA 127, only thread t's SA 128 (×2) → which row becomes 2
  printf("SFA thread t byte0=128 → D rows with value 2:\n");
  for (int t : {0, 1, 2, 3, 4, 5, 8, 9, 28, 29}) {
    std::fill(A.begin(), A.end(), 0x38383838u); fill_B_ones(B, shift_ok); std::fill(SA.begin(), SA.end(), 127); std::fill(SB.begin(), SB.end(), 127);
    SA[t] = 128;
    run(A, B, SA, SB, D);
    int rows[16] = {0};
    for (int i = 0; i < 128; ++i) { int tt = i / 4, ii = i % 4; int row = tt / 4 + (ii >= 2 ? 8 : 0); if (D[i] > 40.f) rows[row] = 1; }
    printf("  t=%2d →", t);
    for (int m = 0; m < 16; ++m) if (rows[m]) printf(" %d", m);
    printf("   (reference D=%g)\n", D[0]);
  }
  printf("SFB thread t byte0=128 → D columns with value 2:\n");
  for (int t : {0, 1, 2, 3, 4, 8, 12, 28}) {
    std::fill(A.begin(), A.end(), 0x38383838u); fill_B_ones(B, shift_ok); std::fill(SA.begin(), SA.end(), 127); std::fill(SB.begin(), SB.end(), 127);
    SB[t] = 128;
    run(A, B, SA, SB, D);
    int cols[8] = {0};
    for (int i = 0; i < 128; ++i) { int tt = i / 4, ii = i % 4; int col = (tt % 4) * 2 + (ii & 1); if (D[i] > 40.f) cols[col] = 1; }
    printf("  t=%2d →", t);
    for (int n = 0; n < 8; ++n) if (cols[n]) printf(" %d", n);
    printf("\n");
  }
  return 0;
}
