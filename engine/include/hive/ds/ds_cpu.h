// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash on the CPU only (hived_ds_cpu). The same model as the default (GPU) family — the math contract is the GPU family's
//   (hive/model.h, hive/runtime.h, hive/model_kernels.h, hive/kv_pack.h, hive/vision.h) and oracle/dsv41_oracle.py, which this engine
//   follows rounding point by rounding point — computed with hive/cpu/cpu_backend.h: every weight in NUMA-split host memory (FP8 32×32-block
//   dense, MXFP4 routed experts, BF16 / F32 small matrices), activations fp32 holding bf16 values wherever the reference keeps bf16, and the
//   reference's fp8 activation quantization (32-element blocks, e8m0 scales) in front of every fp8 / fp4 linear.
//   Engram tables (2 × 98 GB) live in host memory (HIVE_DS_CPU_ENGRAM=ram, default) or are read from the checkpoint mapping (=mmap: page cache
//   / SSD — the same bytes); routed experts likewise (HIVE_DS_CPU_EXPERTS=ram|mmap).
// Contract towards cpufam::Runtime (hive/ds/hived_family_cpu.h): prefill one sequence (internally in chunks), decode M ≤ 8 sequences, DSpark
//   draft / verify / rollback like the GPU Runtime; per-sequence state in host memory, saved / restored as byte buffers.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hive/cpu/cpu_backend.h"

namespace hive {
class EngramHash;
}

namespace hive::ds::cpu {

// config.json (text_config + vision_config) — the fields the engine uses
struct DsConfig {
  int dim = 5120, n_layers = 40, n_heads = 64, head_dim = 512, rd = 64, q_lora_rank = 1280, o_groups = 8, o_lora_rank = 1024;
  int window = 128, vocab = 129280, moe_inter = 2304, n_routed = 384, n_act = 6;
  int index_n_heads = 32, index_head_dim = 128, index_topk = 512;
  int cand_source_layer = 20, cand_topk_blocks = 2048, cand_block = 8;
  int hc = 4, sinkhorn_iters = 20;
  float hc_eps = 1e-6f, norm_eps = 1e-20f, route_scale = 1.5f, swiglu_limit = 10.f;
  float rope_theta = 10000.f, compress_rope_theta = 160000.f, rope_factor = 16.f, beta_fast = 32.f, beta_slow = 1.f;
  int64_t original_seq_len = 65536;
  std::vector<int> compress_ratios, kv_source_layers, index_source_layers, engram_layer_ids;
  int engram_max_ngram = 4, engram_n_heads = 8, engram_head_dim = 256;
  int image_token_id = 129264;
  int vision_layers = 32, vision_dim = 1024, vision_heads = 16, vision_inter = 2816, vision_patch = 14, vision_downsample = 3, vision_max_tokens = 1024;
  float vision_rope_theta = 10000.f;
  int dspark_block = 5, dspark_experts = 128, dspark_act = 3, dspark_noise_token = 128799, dspark_markov_rank = 256;
  std::vector<int> dspark_targets;
  int ratio(int l) const { return l < (int)compress_ratios.size() && l < n_layers ? compress_ratios[l] : 0; }
  bool is_kv_source(int l) const;
  bool is_index_source(int l) const;
  int n_hash_cols() const { return (engram_max_ngram - 1) * engram_n_heads; }
  static DsConfig load(const std::string& ckpt_dir);
};

struct CpuSeq {
  int64_t pos = 0, cap = 0;
  std::vector<int32_t> tokens;
  std::vector<int64_t> engram_history;               // compressed token ids (engram n-grams), -1 = image token
  std::vector<std::vector<uint16_t>> ring;           // per layer: window KV bf16 [win, head_dim] (slot = position % win)
  struct Comp {                                      // per kv-source layer (ordinal in kv_source_layers)
    std::vector<uint16_t> kv;                        // compressed KV bf16 [groups, head_dim] (fp4-rounded values, as the reference cache)
    std::vector<float> ik;                           // indexer keys [groups, index_head_dim] (fp4-rounded values; fp32 for the score kernel)
    std::vector<float> st_kv, st_sc;                 // ratio > 1: compressor state [ratio, head_dim] (score −inf = empty)
  };
  std::vector<Comp> comp;
  std::vector<std::vector<uint16_t>> mtp_ring;       // per DSpark stage: window KV bf16 [win, head_dim]
  std::vector<float> mtp_hidden;                     // target-layer hidden (bf16 values) [n_targets · dim] of position mtp_pos
  bool mtp_hidden_valid = false;
  int64_t mtp_pos = -1;
  bool broken = false;
  // Serialized state (fixed-size parts and the cache rows below pos) for hived's session images, in a fixed order; load_bytes restores it.
  std::vector<std::vector<uint8_t>> save_bytes() const;
  void load_bytes(const std::vector<std::vector<uint8_t>>& b);
};

// wall time per section (ms) accumulated over forwards
struct CpuStats {
  long steps = 0;
  double ms_total = 0, ms_engram = 0, ms_hc = 0, ms_attn_proj = 0, ms_compress = 0, ms_index = 0, ms_attn = 0, ms_attn_out = 0, ms_router = 0, ms_moe = 0,
         ms_head = 0, ms_mtp = 0;
};

// Routed experts held by someone else in the half-record layout of the GPU ExpertStore (hive/expert_store.h HalfLayout: per node n = 0/1 the rows
//   [n·I/2, (n+1)·I/2) of w1 and w3 and [n·dim/2, (n+1)·dim/2) of w2, each followed by its e8m0 scales) — the engine aliases them (no copy).
struct ExpertHalves {
  size_t w1 = 0, s1 = 0, w3 = 0, s3 = 0, w2 = 0, s2 = 0;
  std::function<const uint8_t*(int layer, int expert, int node)> get;
};

struct EngineOptions {
  int max_layer = -1;          // load layers [0, max_layer] only (−1 = all) — partial validation; no head then
  bool vision = true;          // load the ViT + aligner (lazily, on the first image)
  int decoder_tail = 128 * 21; // prefill chunks longer than this run layers ≥ the last kv source (20) on the last rows only (exact for those rows)
  int max_chunk = 4096;        // rows per internal prefill chunk (HIVE_DS_CPU_CHUNK)
  bool experts_mmap = false;   // HIVE_DS_CPU_EXPERTS=mmap: routed experts read from the checkpoint mapping (no copy)
  bool engram_mmap = false;    // HIVE_DS_CPU_ENGRAM=mmap: engram rows read from the checkpoint mapping (no copy)
  // Layer block mode (the GPU daemon's CPU-placed layers, run_block): load layers [first_layer, n_layers) only, no embedding / head;
  //   experts from ext_experts, engram rows handed in by Block::engram_rows (the tables stay where the daemon keeps them).
  int first_layer = 0;
  int tail_min_rows = 0;       // block mode: decoder-tail rows only for chunks of at least this many rows (the GPU runtime's prefill threshold)
  std::shared_ptr<ExpertHalves> ext_experts;
  bool ext_engram = false;
  static EngineOptions from_env();
};

// A run of consecutive layers [first_layer, n_layers) on the CPU for rows another engine produced (DsCpuEngine::run_block). The hidden
//   stream and pre-mix are read and written in place; the caller keeps tokens / positions / engram history and runs the head.
struct BlockPart { CpuSeq* s; int64_t pos0; int r0, n; };   // rows [r0, r0 + n) = positions pos0.. of s
struct Block {
  std::vector<BlockPart> parts;
  int M = 0;
  const int64_t* pos = nullptr;              // [M]
  const int8_t* img = nullptr;               // [M] image-span rows (router VL bias, engram gate 0)
  float* h = nullptr;                        // [M, hc·dim] (bf16 values) in / out
  float* premix = nullptr;                   // [M, hc] in / out
  const int64_t* hashes = nullptr;           // engram hashes [M, n_engram_layers, cols] of these rows (for engram layers in the block)
  // engram rows of hash table `hash_index` for the M rows: vals [M, cols, hd] e4m3 · scales [M, cols, hd/32] e8m0
  std::function<void(int hash_index, int M, uint8_t* vals, uint8_t* scales)> engram_rows;
  const int32_t* topk = nullptr;             // selection of the last index-source layer before the block: positions [M, topk_stride], topk_n per row
  const int* topk_n = nullptr;
  int topk_stride = 0;
  const int32_t* cand = nullptr;             // candidate blocks of the candidate-source layer (if before the block)
  const int* cand_n = nullptr;
  int cand_stride = 0;
  int64_t win_min = 0;                       // window keys before this position are not seen (decoder-tail rows of the caller)
  bool prefill = false;                      // a prefill chunk: the block may switch to decoder-tail rows itself (out_r0 / out_M)
  int out_r0 = 0, out_M = 0;                 // on return: the rows that went through the whole block are h rows [out_r0, out_r0 + out_M)
};

class DsCpuEngine {
 public:
  DsCpuEngine(const std::string& ckpt_dir, const EngineOptions& opt);
  ~DsCpuEngine();
  const DsConfig& cfg() const;
  int n_loaded_layers() const;
  bool has_head() const;          // head + final norm loaded (all layers present)
  void set_engram_hash(const hive::EngramHash* eh);
  void set_decoder_tail(int rows);  // 0 = off
  // A/B measurement only: the previous schedule of parts of the prefill (bit 1 attention in row blocks, 2 compressor pooling row by row) —
  //   identical results
  void set_ab_legacy(int mask);
  // DSpark stages (only after a full load); returns the stage count
  int load_mtp();
  int n_mtp() const;
  void init_seq(CpuSeq& s, int64_t cap);
  void reset_seq(CpuSeq& s);
  // One image inside the next prefill chunk: types per span position (0 start, 1 image row, 2 newline, 3 end), patches bf16 [nh·nw, 3·14·14].
  struct Image { int start = 0, n_vit_h = 0, n_vit_w = 0; std::vector<int8_t> types; const uint16_t* patches = nullptr; };
  bool has_vision() const;
  // patches → aligner rows bf16 values [ceil(nh/3)·ceil(nw/3), dim] (fp32)
  void encode_image(const uint16_t* patches, int n_vit_h, int n_vit_w, std::vector<float>& out);
  // Prefill T tokens of one sequence (internal chunks); logits of the last token → logits (vocab floats) if non-null and the head is loaded.
  void prefill(CpuSeq& s, const int32_t* ids, int T, const std::vector<Image>* images, float* logits);
  // One token for each of M ≤ 8 sequences; logits [M, vocab].
  void decode(CpuSeq* const* seqs, const int32_t* ids, int M, float* logits);
  // DSpark: draft block from the hidden of position pos − 1 and token tok (position pos). drafts [block], conf [block] (raw confidence logits).
  bool mtp_on() const;
  int mtp_block() const;
  int mtp_draft(CpuSeq& s, int32_t tok, int32_t* drafts, float* conf);
  // S parts, part p = M[p] consecutive tokens of seqs[p] (Σ M ≤ 8); logits of every row [Σ M, vocab]. Always followed by rollback_batch.
  void verify_batch(CpuSeq* const* seqs, const int* M, int S, const int32_t* ids, float* logits);
  void rollback_batch(const int* n_keep, int S);

  // Layer block mode (EngineOptions::first_layer > 0): layers [first_layer, n_layers) for the rows of b (see Block).
  void run_block(Block& b);
  // Mirror of a kv source that runs elsewhere: compressed KV rows (kv_pack.h format, 288 B: 32 e4m3 scales + 256 nibble bytes) and indexer key
  //   rows (80 B: 4 e8m0 scales, 12 pad, 64 nibble bytes) of groups [g0, g1) → s's caches of that source (values identical — the packed form is exact).
  void mirror_source(CpuSeq& s, int src_layer, const uint8_t* comp_rows, const uint8_t* idx_rows, int64_t g0, int64_t g1);
  // ---- validation hooks (all optional; positions are absolute) ----------------------------------------------------------------------
  // after a stage of layer l for `rows` rows: what = "h" (after the layer, [rows, hc·dim]), "premix" ([rows, hc]), "engram" (after the engram)
  std::function<void(const char* what, int layer, const int64_t* pos, int rows, const float* data, int cols)> dump_hook;
  // before layer l ≥ 1 of a forward: may overwrite h [rows, hc·dim] and premix [rows, hc] (reference inputs of layer l → per-layer isolation)
  std::function<void(int layer, const int64_t* pos, int rows, float* h, float* premix)> layer_inject;
  // routing of layer l: ids / weights [rows, k]
  std::function<void(int layer, const int64_t* pos, int rows, const int32_t* ids, const float* w, int k)> route_hook;
  // compressed positions selected by index-source layer l: row r has n[r] ascending positions sel[r · stride ..]
  std::function<void(int layer, const int64_t* pos, int rows, const int32_t* sel, int stride, const int* n)> index_hook;
  // DSpark intermediate arrays (names of the oracle's spec.npz: spec_main_x, spec_h_M{s}, spec_x, spec_logits, ...)
  std::function<void(const char* name, const float* data, size_t n)> spec_hook;
  // aligner rows / ViT stages of an image (names of the oracle: vit_{start}_patch, ..., image_emb_{start})
  std::function<void(const char* name, const float* data, size_t n)> vision_hook;
  // layer boundary inside a long prefill (hived layer yields — the callback may run decode / verify of OTHER sequences)
  std::function<void(int layer)> layer_boundary;
  CpuStats& stats();
  struct Impl;
 private:
  std::unique_ptr<Impl> p_;
};

}  // namespace hive::ds::cpu
