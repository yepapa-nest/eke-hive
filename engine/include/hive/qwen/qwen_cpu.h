// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next on the CPU only (hived_qwen_cpu). The same model as hive/qwen/qwen_engine.h — the math contracts are in the GPU
//   family's headers (qwen_model.h, gdn.h, qsa.h, ple.h, qwen_kernels.h) and in oracle/qwen4_oracle.py, which this engine must match — computed
//   with hive/cpu/cpu_backend.h: every weight in NUMA-split host memory (BF16 dense, NVFP4 routed experts, FP8 MTP experts, FP8 PLE table
//   through hive/qwen/ple.h), activations fp32 with bf16 roundings where the reference keeps bf16.
// Contract towards fam::Runtime (hive/qwen/hived_family_cpu.h): prefill one sequence, decode M ≤ 8 sequences (one token each), MTP draft /
//   verify / rollback like QwenEngine; per-sequence state in host memory, saved / restored as byte buffers.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hive/cpu/cpu_backend.h"
#include "hive/qwen/ple.h"
#include "hive/qwen/qwen_model.h"   // QwenConfig (config.json parsing only — no device code is used)

namespace hive::qwen::cpu {

struct CpuSeq {
  int64_t pos = 0, cap = 0;
  std::vector<int32_t> tokens;
  int64_t rope_delta = 0;
  std::vector<std::vector<float>> conv, state;        // per linear layer: [3·10240], [48·128·128]
  struct Attn {
    std::vector<uint16_t> k, v, ik, gk;  // bf16: [pos, 2, 256] [pos, 2, 256] [pos, 128] [pos/4 + 1, 128] (grown as needed)
    std::vector<float> gkt;              // derived (not serialized): gk as fp32 in blocks of 8 groups [g/8][128][8] for the QSA scoring
    int64_t gkt_n = 0;                   //   groups 0..gkt_n−1 mirrored
  };
  std::vector<Attn> attn;                              // per full-attention layer
  std::vector<int32_t> rope;                           // [pos, 3]
  std::vector<float> ple_state;                        // [9, 10240]
  Attn mtp;
  std::vector<float> h_last;                           // [10240] final hidden stream (MTP input), valid if h_last_valid
  bool h_last_valid = false;
  int64_t mtp_written = 0;
  bool broken = false;
  // Serialized state (fixed-size parts and the rows below pos) for hived's session images, in a fixed order; load_bytes restores it.
  std::vector<std::vector<uint8_t>> save_bytes() const;
  void load_bytes(const std::vector<std::vector<uint8_t>>& b);
};

// wall time per section (ms): total, routed+shared MoE, full attention, Gated DeltaNet, hyper-connection mix / combine, PLE, final mixer +
//   lm_head, MTP entry writes after a forward (drafts are not counted)
struct CpuStats {
  double ms_total = 0; long steps = 0; double ms_matmul = 0, ms_moe = 0, ms_attn = 0, ms_lin = 0, ms_hc = 0, ms_ple = 0, ms_head = 0, ms_mtp = 0;
};

class QwenCpuEngine {
 public:
  explicit QwenCpuEngine(const std::string& ckpt_dir, bool load_mtp = true);
  // Hybrid placement (a GPU engine runs layers [0, first_layer), this one the rest): only layers ≥ first_layer are loaded; head = false
  //   also skips the embedding, final mixer and lm_head (and MTP). init_seq allocates states of those layers only.
  struct Options { bool load_mtp = true; int first_layer = 0; bool head = true; };
  QwenCpuEngine(const std::string& ckpt_dir, const Options& o);
  int first_layer() const;
  size_t host_bytes() const;          // host memory of the weights this engine holds (incl. the PLE table when it has the PLE layer)
  // Layers [first_layer, n_layers) of M rows (row r = position pos[r] of seqs[r]; consecutive rows of one sequence are consecutive
  //   tokens): h fp32 [M, 10240] in / out, rope coordinates [M, 3], token ids for the PLE n-grams. Advances each sequence to pos + 1.
  void run_layers(CpuSeq* const* seqs, const int64_t* pos, int M, const int32_t* ids, bool prefill, const int32_t* coords, float* h);
  ~QwenCpuEngine();
  const QwenConfig& cfg() const;
  void init_seq(CpuSeq& s, int64_t cap);
  void reset_seq(CpuSeq& s);
  // Vision rows of the next prefill (rows [row, row + n) of the ids passed to prefill take emb [n, hidden] fp32; gh × gw merged grid).
  struct EmbedOverride { int64_t row; int n; std::vector<float> emb; int gh, gw; };
  void set_embed_override(std::vector<EmbedOverride> v);
  bool has_vision() const;
  // patches bf16 [gh·gw, 1536] (merge-block order) → [gh·gw/4, 2560] fp32
  void encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out);
  // Prefill T tokens of one sequence (chunks of max_chunk inside); logits of the last token → logits (vocab floats) if non-null.
  void prefill(CpuSeq& s, const int32_t* ids, int T, float* logits);
  // One token for each of M ≤ 8 sequences; logits [M, vocab].
  void decode(CpuSeq* const* seqs, const int32_t* ids, int M, float* logits);
  bool mtp_on() const;
  int mtp_k() const;
  int mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  // S parts, part p = M[p] consecutive tokens of seqs[p] (Σ M ≤ 8); logits of every row.
  void verify_batch(CpuSeq* const* seqs, const int* M, int S, const int32_t* ids, float* logits);
  void rollback_batch(const int* n_keep, int S);
  // Validation hook: after every layer with the hidden stream rows [rows, 10240] (fp32).
  std::function<void(int layer, const float* h, int rows)> layer_hook;
  // Validation hook: before every layer l ≥ 1 of a prefill, may overwrite the hidden stream rows [rows, 10240] (per-layer injection of a
  //   reference's layer inputs, so each layer is compared alone).
  std::function<void(int layer, float* h, int rows)> layer_inject;
  // Layer boundary inside a prefill (hived layer yields): the callback may run decode / verify of OTHER sequences.
  std::function<void(int layer)> layer_boundary;
  CpuStats& stats();
  // A/B measurement only: run parts of the previous schedule (bit 1 GEMM dispatch with re-packed panels, 2 MoE dispatch per expert row
  //   count, 4 serial loops, 8 GDN two-pass scan with one task per value head, 16 scalar QSA indexer scoring; 31 = all). Identical results.
  void set_ab_legacy(int mask);
  // A/B of the lossy I8 dense weights (HIVE_QWEN_CPU_DENSE_I8_AB=<categories> at load keeps an I8 twin of each): swap them in / out.
  void set_dense_i8(bool on);
  bool has_dense_i8() const;
  struct Impl;
 private:
  std::unique_ptr<Impl> p_;
};

}  // namespace hive::qwen::cpu
