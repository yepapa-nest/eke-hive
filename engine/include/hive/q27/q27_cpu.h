// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B on the CPU (hived_q27_cpu, and the CPU-placed layers of hived_q27). The math contract is hive/q27/q27_model.h and
//   oracle/q27_oracle.py, which this engine must match layer by layer; computed with hive/cpu/cpu_backend.h: every weight in NUMA-split host
//   memory in its checkpoint format (FP8 per-row scales, NVFP4, BF16), activations fp32 with bf16 roundings where the reference keeps bf16.
// Contract towards cpufam::Runtime (hive/q27/hived_family_cpu.h): prefill one sequence, decode M ≤ 8 sequences (one token each), MTP
//   draft / verify / rollback; per-sequence state in host memory, saved / restored as byte buffers.
// Layer placement (hived_q27 under a VRAM cap): CpuOptions::layers selects the layers this engine loads; run_layer runs one of them for a
//   set of rows on hidden-stream rows handed over by the GPU engine (hive/q27/q27_engine.h), with the sequence's state of those layers kept
//   in a CpuSeq.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hive/cpu/cpu_backend.h"
#include "hive/q27/q27_model.h"

namespace hive::q27::cpu {

struct CpuSeq {
  int64_t pos = 0, cap = 0;
  std::vector<int32_t> tokens;
  int64_t rope_delta = 0;
  std::vector<std::vector<float>> conv, state;        // per linear layer (index among linear layers): [3·10240], [48·128·128]; empty if
                                                      //   that layer is not on the CPU
  struct Attn { std::vector<uint16_t> k, v; };        // bf16 [pos, 4·256] each (grown as needed)
  std::vector<Attn> attn;                             // per full-attention layer (index among attention layers)
  std::vector<int32_t> rope;                          // [pos, 3]
  Attn mtp;
  std::vector<float> h_last;                          // [5120] residual stream of the last position (MTP input), valid if h_last_valid
  bool h_last_valid = false;
  int64_t mtp_written = 0;
  bool broken = false;
  // Serialized state (fixed-size parts and the rows below pos) for hived's session images; load_bytes restores it.
  std::vector<std::vector<uint8_t>> save_bytes() const;
  void load_bytes(const std::vector<std::vector<uint8_t>>& b);
};

// wall time per section (ms): total, MLP, full attention, Gated DeltaNet, norms + residual adds, final norm + lm_head, MTP entry writes
struct CpuStats {
  double ms_total = 0; long steps = 0; double ms_mlp = 0, ms_attn = 0, ms_lin = 0, ms_norm = 0, ms_head = 0, ms_mtp = 0;
};

struct CpuOptions {
  bool load_mtp = true;
  std::vector<bool> layers;   // layers this engine loads and runs (empty = all)
  bool head = true;           // embeddings + final norm (always needed: the hybrid engine gathers its embedding rows from here)
  bool lm_head = true;        // lm_head on the CPU (the hybrid engine sets it only when the lm_head does not fit in VRAM)
};

class Q27CpuEngine {
 public:
  explicit Q27CpuEngine(const std::string& ckpt_dir, const CpuOptions& opt = {});
  ~Q27CpuEngine();
  const Q27Config& cfg() const;
  bool owns(int layer) const;
  size_t weight_bytes() const;
  void init_seq(CpuSeq& s, int64_t cap);
  void reset_seq(CpuSeq& s);
  // Vision rows of the next prefill (rows [row, row + n) of the ids passed to prefill take emb [n, hidden] fp32; gh × gw merged grid).
  struct EmbedOverride { int64_t row; int n; std::vector<float> emb; int gh, gw; };
  void set_embed_override(std::vector<EmbedOverride> v);
  bool has_vision() const;
  // patches bf16 [gh·gw, 1536] (merge-block order) → [gh·gw/4, 5120] fp32
  void encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out);
  // Prefill T tokens of one sequence (chunks of max_chunk inside); logits of the last token → logits (vocab floats) if non-null.
  void prefill(CpuSeq& s, const int32_t* ids, int T, float* logits);
  // One token for each of M ≤ 8 sequences; logits [M, vocab].
  void decode(CpuSeq* const* seqs, const int32_t* ids, int M, float* logits);
  bool mtp_on() const;
  int mtp_k() const;
  int mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  const std::vector<float>& last_draft_logits() const;   // logits of the first draft of the last mtp_draft call (validation)
  // S parts, part p = M[p] consecutive tokens of seqs[p] (Σ M ≤ 8); logits of every row.
  void verify_batch(CpuSeq* const* seqs, const int* M, int S, const int32_t* ids, float* logits);
  void rollback_batch(const int* n_keep, int S);
  // Validation hook: after every layer with the residual stream rows [rows, 5120] (fp32).
  std::function<void(int layer, const float* h, int rows)> layer_hook;
  // Validation hook: before every layer l ≥ 1 of a prefill, may overwrite the residual stream rows (per-layer injection of a reference).
  std::function<void(int layer, float* h, int rows)> layer_inject;
  // Layer boundary inside a prefill (hived layer yields): the callback may run decode / verify of OTHER sequences.
  std::function<void(int layer)> layer_boundary;
  CpuStats& stats();

  // ---- hybrid placement (hive/q27/q27_engine.h) ----------------------------------------------------------------------------------------
  struct Row { CpuSeq* s; int64_t pos; };
  // Layer l (owned) over M rows: h [M, 5120] fp32 (bf16 values) is the residual stream in front of the layer, updated in place; coords
  //   [M, 3] the rows' rope coordinates. Rows of one sequence are consecutive positions in ascending order.
  void run_layer(int l, const Row* rows, int M, const int32_t* coords, float* h);
  // Verify rows of the next run_layer calls: S parts (sequence, first position, row count, first row); the CPU layers keep a snapshot of
  //   their state after every row but the last of each part. end_verify drops the part list; rollback_layers restores the snapshot after
  //   row n_keep[p] − 1 of part p (n_keep[p] < M[p]) in every owned linear layer.
  void begin_verify(CpuSeq* const* seqs, const int64_t* pos0, const int* M, int S);
  void end_verify();
  void rollback_layers(CpuSeq* const* seqs, const int* M, const int* n_keep, int S);
  // the bf16 embedding table [vocab, 5120]; final norm + lm_head of M residual-stream rows (fp32) → logits [M, vocab] (needs lm_head)
  const uint16_t* embed_table() const;
  void head_logits(const float* h, int M, float* logits);
  struct Impl;
 private:
  std::unique_ptr<Impl> p_;
};

}  // namespace hive::q27::cpu
