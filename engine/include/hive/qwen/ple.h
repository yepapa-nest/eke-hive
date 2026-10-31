// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// PLE — Qwen3.8-Flash-Next's per-layer n-gram embedding in front of layer 1's mixing.
//   Lookup (host, PleHash): for the token x_t with the two tokens before it (x_{t−1}, x_{t−2}; the EOS id 248044 before the start of the
//   sequence), b = x_{t−1}, c = (x_{t−1} == EOS) ? EOS : x_{t−2}; the 2-gram mix = x_t·m0 ⊕ b·m1, the 3-gram mix = x_t·m0 ⊕ b·m1 ⊕ c·m2
//   (int64, ⊕ = xor); head j of n-gram n (8 heads each) reads row (mix_n mod size_j) + offset_j. m, size and offset are the checkpoint's
//   own buffers (layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets). 16 rows × 160 FP8 values per token, × weight_scale.
//   The table (320,001,536 rows × 160 B = 47.7 GiB) is stored as 128 tensors "…ngram_embedding.shard_i.weight" of equal row count.
//   Table placement (HIVE_QWEN_PLE): "ram" (default) — pinned host memory, read in place · "ssd" — rows read on demand from the checkpoint
//   file with O_DIRECT through a row cache (PleSsd below; hive/engram_ssd.h's design: thread-pool pread, CLOCK cache, read-ahead).
//   Both give the same bytes.
//   NVIDIA's FP4-Local checkpoint stores the table in NVFP4 instead (`ple_embedding_dtype` "nvfp4"): per shard `.weight` U8 [rows, 80]
//   (E2M1 pairs, low nibble first) + `.weight_scale` F8_E4M3 [rows, 10] (one per 16 values) and one fp32 `ngram_embedding.weight_scale_2`;
//   value = bf16(e2m1 · (e4m3 · scale_2)). A gathered row then holds the 80 data bytes followed by the 10 scale bytes in its 160-byte slot
//   (the same [n, 16, 160] buffer). RAM mode only: 26.8 GiB (28.8 GB) pinned (90 B per row). Checked against the NVFP4 checkpoint's FP8 table
//   2026-10-31: the same rows dequantized both ways agree to cos 0.994–0.996 (QAD retrained the table), the high-nibble-first reading to ≈ 0.
//   Block (GPU, ple_* below): key = key_proj(e) [hc·hidden], value = value_proj(e) [hidden]; per copy c:
//     gate_c = Σ_d norm_key(key)_c,d · norm_query(h)_c,d / sqrt(hidden); gate_c = sigmoid(sign·sqrt(max(|gate_c|, 1e-6)));
//     gv_c,d = gate_c · value_d; y = gv + silu(conv(norm_conv(gv))) with a depthwise causal conv of 4 taps at dilation 3 over the
//     hc·hidden channels (taps t−9, t−6, t−3, t; per-sequence state = the last 9 conv inputs, zeros when fresh); h += y.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "hive/safetensors.h"

namespace hive::qwen {

constexpr int kPleHeads = 16, kPleHeadDim = 160, kPleRow = kPleHeads * kPleHeadDim;  // 2560 values per token
constexpr int kPleConvState = 9;                                                     // (taps − 1) · dilation

// ---- Hash (host) ------------------------------------------------------------------------------------------------------------------------
struct PleHash {
  int ngram = 3, heads_per = 8, eos = 0;
  int64_t vocab = 0;                         // token ids are below it (the overflow bound of the products)
  std::vector<int64_t> mult;                 // [ngram]
  std::vector<int64_t> size, offset;         // [(ngram − 1) · heads_per]
  int64_t rows = 0;                          // total rows of the table
  // rows of n tokens tok[i] whose two predecessors are prev2 / prev1 (−1 = before the sequence start → EOS): out [n, 16]
  //   (header-only so the CPU test builds without CUDA)
  void rows_of(const int32_t* tok, int n, int32_t prev2, int32_t prev1, int64_t* out) const {
    const int64_t E = eos;
    const int nh = heads_per;
    for (int i = 0; i < n; ++i) {
      const int64_t x = tok[i];
      const int64_t w1 = i >= 1 ? tok[i - 1] : (prev1 < 0 ? E : prev1);
      const int64_t w0 = i >= 2 ? tok[i - 2] : i == 1 ? (prev1 < 0 ? E : prev1) : (prev2 < 0 ? E : prev2);
      const int64_t b = w1, c = w1 == E ? E : w0;  // the n-gram window does not reach back across an EOS
      // tokens < vocab and multipliers ≤ INT64_MAX / vocab (ple_open checks the bound), so every product fits int64
      const int64_t mix2 = (x * mult[0]) ^ (b * mult[1]);
      const int64_t mix3 = mix2 ^ (c * mult[2]);
      int64_t* o = out + (size_t)i * kPleHeads;
      for (int j = 0; j < nh; ++j) o[j] = mix2 % size[j] + offset[j];
      for (int j = 0; j < nh; ++j) o[nh + j] = mix3 % size[nh + j] + offset[nh + j];
    }
  }
};

// ---- Table ----------------------------------------------------------------------------------------------------------------------------------
// Gathers the FP8 rows of n tokens (16 rows each, in head order) into dst [n, 16, 160] (host memory; the engine passes pinned memory).
class PleTable {
 public:
  virtual ~PleTable() = default;
  virtual void gather(const int64_t* rows, int n_tokens, uint8_t* dst) = 0;
  virtual void prefetch(const int64_t* /*rows*/, int /*n_tokens*/) {}   // read-ahead hint (SSD)
  virtual size_t host_bytes() const = 0;
  virtual const char* kind() const = 0;
};
// Opens the table from the checkpoint (mode from HIVE_QWEN_PLE: ram | ssd), reads the hash buffers and the weight scale (FP8: the bf16
//   per-table scale; NVFP4: scale_2). nvfp4 (optional): set to whether the table is NVFP4; a caller that passes nullptr gets an error for
//   an NVFP4 table.
std::unique_ptr<PleTable> ple_open(Checkpoint& ck, const std::string& prefix, PleHash& hash, float& weight_scale, bool* nvfp4 = nullptr);

// ---- Block (GPU) -----------------------------------------------------------------------------------------------------------------------------
// e[m, :] = bf16(fp8(rows[m, :]) · scale) — rows uint8 [M, 2560] (may be pinned host memory, UVA), e bf16 [M, 2560]
void ple_dequant(const uint8_t* rows, int M, float scale, __nv_bfloat16* e, cudaStream_t st);
// the NVFP4 table's rows (each 160-byte slot = 80 data bytes + 10 E4M3 scales): e = bf16(e2m1 · (e4m3 · scale2))
void ple_dequant_nvfp4(const uint8_t* rows, int M, float scale2, __nv_bfloat16* e, cudaStream_t st);
// gv[m, c, d] = bf16(gate(m, c) · value[m, d]) with gate from norm_key(key) · norm_query(h) (both (1 + w) grouped per dim) as above.
//   key, h bf16 [M, hc·dim] · value bf16 [M, dim] · gv bf16 [M, hc·dim]
void ple_gate(const __nv_bfloat16* key, const __nv_bfloat16* h, const __nv_bfloat16* norm_key, const __nv_bfloat16* norm_query,
              const __nv_bfloat16* value, int M, int hc, int dim, float eps, __nv_bfloat16* gv, cudaStream_t st);
// the same with fp32 norm weights (FP4-Local)
void ple_gate(const __nv_bfloat16* key, const __nv_bfloat16* h, const float* norm_key, const float* norm_query, const __nv_bfloat16* value, int M,
              int hc, int dim, float eps, __nv_bfloat16* gv, cudaStream_t st);
// One sequence, T consecutive rows: h[t, :] += bf16( gv[t] + silu(conv(gvn)[t]) ) with the conv state (fp32 [9, C], oldest first) read
//   and updated in place. gvn = norm_conv(gv) (bf16 [T, C]); C = hc·dim.
void ple_conv_seq(const __nv_bfloat16* gv, const __nv_bfloat16* gvn, const float* conv_w, int T, int C, float* state, __nv_bfloat16* h,
                  cudaStream_t st);
// M sequences, one row each (per-row state pointers, device-readable array); snap (optional, verify rows of one sequence: R rows with
//   snap[r] receiving the state after row r < R−1) — see ple_conv_rows.
void ple_conv_batch(const __nv_bfloat16* gv, const __nv_bfloat16* gvn, const float* conv_w, int M, int C, float* const* states, __nv_bfloat16* h,
                    cudaStream_t st);
void ple_conv_rows(const __nv_bfloat16* gv, const __nv_bfloat16* gvn, const float* conv_w, int R, int C, float* const* state, float* const* snap,
                   __nv_bfloat16* h, cudaStream_t st);
// ple_conv_rows for P parts in one launch: part p = rows [vpt[2p], vpt[2p] + vpt[2p + 1]) (device int pairs), state through state_base[p],
//   snapshots (parts of more than one row) at snap_base + p · snap_stride — the same per-element arithmetic
void ple_conv_rows_dyn(const __nv_bfloat16* gv, const __nv_bfloat16* gvn, const float* conv_w, int P, const int* vpt, int C, float* const* state_base,
                       float* const* snap_base, int snap_stride, __nv_bfloat16* h, cudaStream_t st);

}  // namespace hive::qwen
