// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_ATTN2 (env_on, default off) — bandwidth-oriented versions (decode_attn2.cu) of the attention-side GPU work in decode
//   (M <= 8): router, wq_b, wo_b, shared experts.
//   Numeric contract: **bit-identical** to the baseline kernels — for every output element the same lane→block assignment (lane l
//   takes blocks l, l+32, … in ascending order), the same fmaf chain, the same xor reduction (16,8,4,2,1) and the same epilogue
//   formula. What changes: (1) the path by which weights reach registers (per-warp cp.async ring, issued *before* input
//   quantization), (2) input quantization is split across the block's 256 threads (same per-element formula; amax uses fmaxf, so
//   order does not matter), (3) a grid in which each block handles several columns (one wave by occupancy), (4) router = one
//   expert per block (row m on warp m) + warp-parallel top-k (larger value first, ties → smaller index = the first maximum of the
//   baseline lane-0 sequential scan).
//   The only case that may differ: a router row with fewer than k selectable candidates (not NaN / -FLT_MAX) — the baseline
//   kernel read out of bounds with bi = -1, this version writes id -1 with weight 0 (only happens when activations are NaN; same
//   handling as D2b).
//   Signatures match the baseline functions — call sites use `(k::decode_attn2_on() ? k::attn2_x : …)(...)`. If shape or alignment
//   is outside what this version supports, the function calls the baseline function internally (absorbed — never rejected).
#pragma once
#include <cstdint>

#include "hive/common.h"

namespace hive::k {

bool decode_attn2_on();  // env_on("HIVE_DECODE_ATTN2") — read once (unset / "" / "0" = off)

// Same signatures and same (bitwise) outputs as the baseline fused.cu functions
void attn2_q_proj(const bf16* xn, int K, const uint8_t* wqa, const uint8_t* sqa, int Nqa, const bf16* q_norm, float eps, const uint8_t* wqb,
                  const uint8_t* sqb, int H, int D, int rd, const float2* freqs, const int32_t* pos, int M, bf16* qr, bf16* qrn, bf16* q, uint8_t* qrq,
                  uint8_t* qrs, int* counter, cudaStream_t st);
void attn2_gemv_quantin(const bf16* A, int K, const uint8_t* B, const uint8_t* sb, int M, int N, bf16* C, cudaStream_t st);
void attn2_shared_experts(const bf16* xn, int K, const uint8_t* w1, const uint8_t* s1, const uint8_t* w3, const uint8_t* s3, const uint8_t* w2,
                          const uint8_t* s2, int M, int I, float limit, bf16* y, uint8_t* xq, uint8_t* xs, float* acc, cudaStream_t st);
void attn2_router(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                  float route_scale, float* scores, int32_t* ids, float* w, int* counter, cudaStream_t st);
// q_b (+RoPE) columns only (same values as the q blocks of D2 dec_qb_kernel and gemv_qb_rope_kernel). Position = pos[m · pos_stride]
//   (with a D2 row table: &tab->pos, stride = sizeof(DecRow)/4). Also writes side_q/side_s (quantized qrn, for the indexer and MTP).
//   For a shape this version does not accept it does nothing and returns false (the caller falls back to the baseline kernel).
bool attn2_qb(const bf16* qrn, int K, const uint8_t* B, const uint8_t* sb, int M, int N, int D, int rd, const float2* freqs, const int32_t* pos,
              int pos_stride, bf16* q, uint8_t* side_q, uint8_t* side_s, cudaStream_t st);

}  // namespace hive::k
