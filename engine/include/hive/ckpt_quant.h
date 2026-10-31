// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Checkpoint weight formats beyond plain BF16 / NVFP4 records (host side, no CUDA — the CPU engines and tests use it too):
//
//  · NVFP4 block scales stored compressed ("nvfp4_csf", NVIDIA's Qwen3.8-Flash-Next-FP4-Local routed experts): the E4M3 scale matrix
//    [R, C] (C = K / 16) of a projection is stored as
//      `<proj>.weight_scale.nvfp4_csf_fixed`      U8  [R / 16, 16 + 8·C]
//      `<proj>.weight_scale.nvfp4_csf_exceptions` U32 [n]
//    Per group of 16 rows: 16 base bytes (one E4M3 code byte per row of the group), then 16·C 4-bit deltas, row-major inside the group
//    (row 0's C deltas, then row 1's …), two per byte, the even position in the low nibble. scale code = base + delta (integer addition
//    of code bytes). Each exception word replaces one entry: high 8 bits = the E4M3 code byte, low 24 bits = the flat index r·C + c; the
//    delta stored at an exception is 0. This description is ours (from the tensor shapes and bytes of the checkpoint, 2026-10-31) — no
//    third-party code was used.
//    Invariants checked on decode (a violation means a damaged or misread tensor — the loader refuses it; data integrity, not
//    availability): fixed shape = [R / 16, 16 + 8·C]; every exception index < R·C; the delta at an exception index is 0; every decoded code
//    byte is a finite non-negative E4M3 value (≤ 0x7E). Measured over the whole checkpoint (73,728 tensors) before these checks were made
//    fatal — see docs/qwen.md "FP4-Local".
//  · MXFP8 dense projections: `<name>.weight` F8_E4M3 [N, K] + `<name>.weight_scale` U8 (UE8M0, 2^(b − 127)) [N, K / 32].
//  · NVFP4 dense projections (W4A16 vision / MTP experts, PLE rows): `<name>.weight` U8 [N, K / 2] (E2M1 pairs, low nibble first) +
//    `<name>.weight_scale` F8_E4M3 [N, K / 16] + `<name>.weight_scale_2` F32 (one value).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hive/safetensors.h"

namespace hive::ckq {

// E4M3 / E2M1 / UE8M0 code → float (exact)
float e4m3_f32(uint8_t b);
float e2m1_f32(uint8_t nibble);
float ue8m0_f32(uint8_t b);
// float → bf16 bits, round to nearest even (finite inputs)
uint16_t f32_bf16(float v);

// CSF decode into out [R, C] (E4M3 code bytes). Returns false and sets *err on an invariant violation.
bool csf_decode(const uint8_t* fixed, int64_t fixed_rows, int64_t fixed_cols, const uint32_t* exc, size_t n_exc, int R, int C, uint8_t* out,
                std::string* err);

// `<base>weight_scale` (F8_E4M3) or its CSF pair present? base ends with "." (e.g. "…experts.7.gate_proj.")
bool has_csf(Checkpoint& ck, const std::string& base);
// The NVFP4 block-scale matrix [R, C] of `<base>weight` into out (plain F8_E4M3 tensor copied, CSF decoded). Throws on a violation.
void nvfp4_scales(Checkpoint& ck, const std::string& base, int R, int C, uint8_t* out);

enum class Dense { BF16, MXFP8, NVFP4, Other };
// format of the projection `<base>weight` (base ends with ".")
Dense dense_format(Checkpoint& ck, const std::string& base);
// rows / columns (logical: NVFP4 K = 2 · stored bytes)
void dense_shape(Checkpoint& ck, const std::string& base, int64_t& N, int64_t& K);
// the projection as BF16 bits [N, K]: BF16 copied; MXFP8 expanded exactly (e4m3 · 2^k is a bf16 value); NVFP4 = bf16(e2m1 · e4m3 · g)
//   (one rounding). threads > 1 splits the rows.
std::vector<uint16_t> expand_bf16(Checkpoint& ck, const std::string& base, int threads = 8);
// MXFP8 pieces of `<base>`: e4m3 bytes [N, K] and the ue8m0 bytes [N, K / 32]
void mxfp8_parts(Checkpoint& ck, const std::string& base, const uint8_t*& q, const uint8_t*& s, int64_t& N, int64_t& K);

// a vector (norm / bias): F32 or BF16 → float
std::vector<float> vec_f32(Checkpoint& ck, const std::string& name);

}  // namespace hive::ckq
