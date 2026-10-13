// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next routed-expert decode on the GPU (src/qwen/fn_moe.cu): out[row] += Σ_pairs w · expert(x[row]) over device pairs whose
//   records are resident (NVFP4, glm::ExpertRecLayout). Same contract as glm::moe_decode with limit 0 (plain SiLU); for H, I where
//   fn_moe_decode_supported() holds (FN: 2560 / 640).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "hive/glm/fp8b.h"
#include "hive/glm/glm_moe.h"

namespace hive::qwen {

bool fn_moe_decode_supported(int H, int I);
// tests: true = the first decode kernels (O(n²) grouping, scalar activation loads) — v2 must match them bit for bit
void fn_moe_decode_set_ref(bool ref);
size_t fn_moe_decode_ws_bytes(int n_pairs, int H, int I);
void fn_moe_decode(const glm::ExpertRecLayout& L, const glm::MoePair* pairs, int n_pairs, const __nv_bfloat16* x, int H, int I, float* out,
                   void* ws, cudaStream_t st);

// The whole routed part of a decode MoE layer from the router's device output, fused with the final combine:
//   out[m] = bf16( bf16(Σ_k w[m,k] · expert(ids[m,k]) (x[m])) + bf16( bf16(sigmoid(bf16(lg[m·ld + col]))) · shared[m] ) )
//   (records from rec_tab[li · E + id]; the rounding points of build_pairs → fn_moe_decode → moe_combine). ws ≥ fn_moe_decode_ws_bytes(M·K).
void fn_moe_decode_routed(const glm::ExpertRecLayout& L, const int32_t* ids, const float* w, int M, int K, const uint8_t* const* rec_tab, int li, int E,
                          const __nv_bfloat16* x, int H, int I, const __nv_bfloat16* shared, const float* lg, int ld, int col, __nv_bfloat16* out,
                          void* ws, cudaStream_t st, const float* add = nullptr, cudaEvent_t join = nullptr,
                          const uint8_t* const* host_tab = nullptr, int pcie_div = 0);
//   Hybrid route (QwenEngine, part of the experts not resident): rec_tab holds nullptr for them — those pairs are skipped, and `add`
//   (fp32 [M, H], the CPU's weighted sum for them; pinned host memory is fine) is added to the routed sum before the combine, once `join`
//   (an event the caller records when the CPU share is written) has completed. With host_tab (pinned host record per key, device array) and
//   pcie_div > 0, every pcie_div-th non-resident expert of the layer (by id) is read over PCIe by these kernels instead (the caller's CPU
//   share leaves the same ones out).

// The same for FP8 experts with 128×128 block scales (the MTP layer): table = device array of E·3 glm::Fp8BMat (gate, up, down per expert).
//   Drafts only (no output depends on it); no host round trip, so the draft steps can be captured.
void fn_moe_fp8_routed(const glm::Fp8BMat* table, int E, const int32_t* ids, const float* w, int M, int K, const __nv_bfloat16* x, int H, int I,
                       const __nv_bfloat16* shared, const float* lg, int ld, int col, __nv_bfloat16* out, void* ws, cudaStream_t st);

// FP8 e4m3 copy with 128×128 block scales (glm::Fp8BMat layout) of a bf16 [N, K] matrix — the MTP draft head (drafts are proposals; the
//   verify step scores them with the bf16 head, so outputs do not depend on it)
void fn_quant_fp8b(const __nv_bfloat16* w, int N, int K, uint8_t* q, float* s, cudaStream_t st);
// Prefill (rows > 64, every expert of layer li resident; tab = the engine's record table [layer · E + e]): out[row] = Σ_k part·w in k order
//   (fp32, overwritten) — grouped W4A16 GEMMs (src/qwen/fn_moe_pf.cu). Rows are processed in sub-chunks of S; ws ≥ fn_moe_prefill_ws_bytes(S, ...).
bool fn_moe_prefill_supported(int H, int I);
// gate|up block width of the short-prompt tiles (64 default; 128 = HIVE_QWEN_MOE_PF_GN) — tests compare the two bit for bit
void fn_moe_prefill_set_gn(int gn, int dn = 128);   // + down block rows (128 default; 256 = HIVE_QWEN_MOE_PF_DN)
size_t fn_moe_prefill_ws_bytes(int S, int K, int E, int H, int I);
void fn_moe_prefill(const glm::ExpertRecLayout& L, const uint8_t* const* tab, int li, int E, const int32_t* ids, const float* w, int M, int K,
                    const __nv_bfloat16* x, int H, int I, float* out, void* ws, int S, cudaStream_t st);

// draft token of one bf16 logit row: id = argmax (lowest index on ties; map[id] when map != null), conf = {max, Σexp(l − max)} (the
//   row_max_sumexp pair) — two small kernels (fixed-order combine) instead of a bf16→fp32 copy + argmax + max/sum over the fp32 row.
//   part: 3·fn_draft_pick_parts(V) floats
int fn_draft_pick_parts(int V);
void fn_draft_pick(const __nv_bfloat16* lg, int V, const int32_t* map, int32_t* id, float* conf, float* part, cudaStream_t st);
// dst[i, :] = src[rows[i], :] for a bf16 [*, K] matrix (device row ids)
void fn_gather_rows(const __nv_bfloat16* src, const int32_t* rows, int n, int K, __nv_bfloat16* dst, cudaStream_t st);
void fn_bf16_to_f32(const __nv_bfloat16* a, size_t n, float* b, cudaStream_t st);

}  // namespace hive::qwen
