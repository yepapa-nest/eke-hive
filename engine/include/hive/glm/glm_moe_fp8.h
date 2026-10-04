// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash routed MoE with Z.ai FP8 experts (128×128 block scales) resident in VRAM — the NextN (MTP) layer's 288 experts — for decode
//   rows M ≤ 4, in two launches with no host round trip (expert ids and route weights are read on the device; graph-capturable).
// Validation and timings: engine/tests/test_glm_moe_fp8.cu.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "hive/glm/fp8b.h"

namespace hive::glm {

// ---- Device expert table ---------------------------------------------------------------------------------------------------------------------
// A device array of E·3 Fp8BMat: entry [e·3 + 0] = gate [I, H], [e·3 + 1] = up [I, H], [e·3 + 2] = down [H, I] (the layout of
//   std::vector<std::array<Fp8BMat, 3>>::data(), glm_model.h experts). The matrices' weight / scale pointers must be device pointers.
// glm_moe_fp8_table_create checks every shape against (H, I) on the host, cudaMalloc's the table and copies it (synchronous). Build it
//   once at load; free with glm_moe_fp8_table_free.
Fp8BMat* glm_moe_fp8_table_create(const Fp8BMat* host_triples, int E, int H, int I);
void glm_moe_fp8_table_free(Fp8BMat* table);

// ---- Routed MoE decode -------------------------------------------------------------------------------------------------------------------------
// For every row m < M and slot k < K with e = ids[m, k] ∈ [0, E):
//   y_mk = bf16( down_e · bf16( swiglu( bf16(gate_e · x[m]), bf16(up_e · x[m]), limit ) ) )
//   out[m, :] += w[m, k] · y_mk        (fp32, k = 0..K-1 in order, one fmaf per element: out = fmaf(w, y, out))
// Slots with an id outside [0, E) contribute nothing (the engine's `continue`). out is ACCUMULATED into (zero it first for a plain sum).
//   table: device table from glm_moe_fp8_table_create · ids int32 [M, K], w fp32 [M, K] (device) · x bf16 [M, H] (row stride H, 16-byte aligned)
//   · out fp32 [M, H] · ws ≥ glm_moe_fp8_ws_bytes(M, K, I) bytes of device scratch (16-byte aligned).
//   Instantiated for H 4096 · I 2048 (GLM-5.3-Flash), 1 ≤ M ≤ 4, 1 ≤ M·K ≤ 32.
// Numeric contract: BIT-IDENTICAL to the composition glm_mtp.cpp runs today, per row m and k = 0..K-1:
//   glm_shared_fp8_decode(table[e], x[m], M = 1) → accum_scaled(out[m], y, w[m, k])   (same lane → 16-column chunk mapping, fma order,
//   block-scale application, butterfly, SwiGLU expression and accumulate expression; tested byte-for-byte).
// Structure: launch 1 = gate/up + SwiGLU, grid (I/8 rows, M·K slots), warp = one row of I; launch 2 = down + weighted accumulate, block =
//   4 output rows of H, warps take the distinct experts, partial results in shared memory, then out += Σ_k in fixed order (no atomics).
//   An expert chosen by several slots (other rows of the same step) is streamed ONCE: the first slot holding the id computes every slot that
//   shares it. Both launches use PDL (launch 2 prefetches its down weights while launch 1 drains). Expert ids are read only after the
//   predecessor completed (launch 1 waits before it triggers launch 2), so ids/w may be produced by the immediately preceding kernel
//   (glm_router_decode) on the same stream.
size_t glm_moe_fp8_ws_bytes(int M, int K, int I);
void glm_moe_fp8_decode(const Fp8BMat* table, int E, const int32_t* ids, const float* w, const __nv_bfloat16* x, int M, int K, int H, int I,
                        float limit, float* out, void* ws, cudaStream_t st);

}  // namespace hive::glm
