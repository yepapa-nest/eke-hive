// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// fp4 packing format for compressed KV and indexer keys. Storing the fp4-rounded values in bf16 slots (16 bits) would take 3.6–4× the information content.
//   compressed KV row (D=512, 16-element blocks, e4m3 scales):  [scale e4m3 ×32][nibble ×256 (low nibble = even dim)] = 288 B  (bf16: 1,024 B)
//   indexer key row (Di=128, 32-element blocks, e8m0 scales):  [scale e8m0 ×4, pad 12][nibble ×64] = 80 B          (bf16: 256 B)
// Numeric contract: stored value = e2m1 value × scale value. The product e2m1 (2 significant bits) × e4m3 (4 significant bits) has ≤ 6 significant bits and fits bf16 (8 bits) exactly,
//   so a bf16-stored value == the fp32 product recomputed here — attention and indexer results are **bit-identical** (verified by comparing dumps before/after packing).
#pragma once
#include <cstdint>

#include "hive/common.h"

namespace hive::kvp {

constexpr int COMP_D = 512, COMP_BLK = 16, COMP_SCALES = COMP_D / COMP_BLK, COMP_ROW = COMP_SCALES + COMP_D / 2;  // 32 + 256 = 288
constexpr int IDX_D = 128, IDX_BLK = 32, IDX_SCALES = IDX_D / IDX_BLK, IDX_HDR = 16, IDX_ROW = IDX_HDR + IDX_D / 2;  // 16 + 64 = 80
static_assert(COMP_ROW % 16 == 0 && IDX_ROW % 16 == 0, "packed rows must stay 16-byte aligned");

#ifdef __CUDACC__
namespace {
__constant__ float kE2M1v[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
}
// Reading the __constant__ table above with a data-dependent index serializes for every distinct address within a warp (up to 16 replays).
//   So e2m1 is built directly from the bits — bit-identical to the table (including ±0): exponent e = (n>>1)&3, mantissa m = n&1 · e>0 → 2^(e-1)·(1+m/2) · e=0 → m/2.
__device__ __forceinline__ float e2m1_val(uint32_t nib) {
  const uint32_t e = (nib >> 1) & 3u, m = nib & 1u;
  const uint32_t mag = e ? (((126u + e) << 23) | (m << 22)) : (m ? (126u << 23) : 0u);
  return __uint_as_float(mag | ((nib & 8u) << 28));
}
// value of dimension d of a compressed KV row (fp32 — equal to the bf16-stored value)
__device__ __forceinline__ float comp_val(const uint8_t* __restrict__ row, int d) {
  const int nib = (row[COMP_SCALES + (d >> 1)] >> (4 * (d & 1))) & 0xF;
  return e2m1_val(nib) * e4m3_to_f32(row[d >> 4]);
}
// the 16 dims [16·b, 16·b+16) at once (one block per lane in decode attention)
__device__ __forceinline__ void comp_block16(const uint8_t* __restrict__ row, int b, float (&v)[16]) {
  const uint2 nb = *reinterpret_cast<const uint2*>(row + COMP_SCALES + 8 * b);
  const float sc = e4m3_to_f32(row[b]);
  const uint32_t x[2] = {nb.x, nb.y};
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) v[8 * i + j] = e2m1_val((x[i] >> (4 * j)) & 0xF) * sc;
}
__device__ __forceinline__ float2 comp_pair(const uint8_t* __restrict__ row, int d0) {  // (d0, d0+1), d0 even
  const uint8_t byte = row[COMP_SCALES + (d0 >> 1)];
  const float sc = e4m3_to_f32(row[d0 >> 4]);
  return make_float2(e2m1_val(byte & 0xF) * sc, e2m1_val(byte >> 4) * sc);
}
__device__ __forceinline__ float idx_val(const uint8_t* __restrict__ row, int d) {
  const int nib = (row[IDX_HDR + (d >> 1)] >> (4 * (d & 1))) & 0xF;
  return e2m1_val(nib) * e8m0_to_f32(row[d >> 5]);
}
#endif

}  // namespace hive::kvp
