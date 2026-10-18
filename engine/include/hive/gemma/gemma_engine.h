// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma 4 (Unified, text path) engine: int4 group-32 weights (compressed-tensors pack-quantized, repacked for the decode GEMV), bf16
//   embedding shared with the lm_head, a paged fp8 KV cache (sliding-window layers release pages behind the window), and the Frozen-KV MTP
//   assistant (it reads the target's last sliding / full layer KV). The math contract is oracle/gemma4_oracle.py.
//   The vision / audio towers of the checkpoint are not loaded (text serving).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "hive/devbuf.h"
#include "hive/gemma/gemma_kernels.h"

namespace hive::gemma {

struct GemmaConfig {
  int n_layers = 0, hidden = 0, inter = 0, heads = 0, vocab = 0;
  int kv_heads = 0, head_dim = 0;                 // sliding layers
  int g_kv_heads = 0, g_head_dim = 0;             // full layers
  bool k_eq_v = false;                            // full layers: V = norm(k_proj x) (no v_proj)
  int window = 0;
  float eps = 1e-6f, softcap = 0.f, embed_scale = 1.f;
  float theta_s = 1e4f, theta_f = 1e6f, partial_f = 1.f;
  std::vector<bool> sliding;                      // per layer
  std::vector<int> type_index;                    // per layer: index among the layers of its type
  int n_sliding = 0, n_full = 0;
  std::vector<int> eos;
  static GemmaConfig load(const std::string& ckpt_dir);
};

// One sequence: tokens and position, the page tables of its KV (sliding / full), the last row's hidden for the drafter.
struct GemmaSeq {
  int64_t pos = 0;                                // tokens in the KV
  std::vector<int32_t> tokens;
  int64_t swa_lo = 0;                             // first position whose sliding-layer page is still held (page aligned)
  std::vector<uint8_t*> pages_s, pages_f;         // page base pointers (index = position / kPage; released sliding pages null)
  DevBuf tab_s, tab_f;                            // device copies of the tables (capacity max_ctx / kPage)
  int64_t max_ctx = 0;
  size_t tb_s = 0, tb_f = 0, h_bytes = 0;            // token bytes of a sliding / full page row, bytes of h_last (set by init_seq)
  DevBuf h_last;                                  // bf16 [hidden] final-normed hidden of the last processed row (MTP seed)
  bool h_last_valid = false;
  DevBuf h_rows;                                  // bf16 [16, hidden] verify rows' hidden (rollback picks the kept row)
  bool broken = false;
  // session images: the pages' filled prefixes in order, then the small state (the filled part of a page only grows while the sequence
  //   only grows, so each page piece is an `append` piece)
  struct Piece { const void* host; const void* dev; size_t n; bool append; };
  void save_plan(std::vector<Piece>& out, std::vector<std::vector<uint8_t>>& hold) const;
};

struct Placement { bool head_gpu = true; };

class GemmaEngine {
 public:
  GemmaEngine(const std::string& ckpt_dir, const std::string& assistant_dir, int64_t max_ctx, int max_batch);
  ~GemmaEngine();
  const GemmaConfig& cfg() const;
  const Placement& placement() const;
  void init_seq(GemmaSeq& s, int64_t cap);
  void reset_seq(GemmaSeq& s);
  void release_kv(GemmaSeq& s);
  // load a session image's pieces (the order of save_plan) into the sequence's (new) pages
  void load_views(GemmaSeq& s, const std::vector<std::pair<const uint8_t*, size_t>>& v,
                  const std::function<void(void*, const uint8_t*, size_t)>& h2d);
  std::function<bool()> kv_reclaim;               // pool exhausted: evict one idle session (true = retry)
  std::function<void(int layer)> layer_boundary;
  std::function<bool()> layer_boundary_due;
  void warmup(int max_batch, int max_k);
  // prefill T tokens of one sequence (chunked internally); logits of the last row → logits (fp32 [vocab], host or null)
  void prefill(GemmaSeq& s, const int32_t* ids, int T, float* logits);
  // one token for each of M sequences; logits [M, vocab] (host, or null with set_host_logits(false))
  void decode(GemmaSeq* const* seqs, const int32_t* ids, int M, float* logits);
  // verify rows: M[s] rows of sequence s (ids concatenated); logits [Σ M, vocab]
  void verify_batch(GemmaSeq* const* seqs, const int* M, int S, const int32_t* ids, float* logits);
  void rollback_batch(const int* n_keep, int S);  // the sequences of the last verify keep n_keep rows each
  bool mtp_on() const;
  int mtp_k() const;
  int mtp_draft(GemmaSeq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  void mtp_draft_batch(GemmaSeq* const* seqs, const int32_t* toks, int S, int k, int32_t* drafts, float* conf, int* n_out);
  const float* logits_dev() const;
  float* logits_host();
  void set_host_logits(bool on);
  long graph_captures() const;
  cudaStream_t stream() const;
  // VRAM the engine holds (weights, KV pool, work) and the KV pages in use / free / on the host
  struct MemStats { size_t weights = 0, kv_pool = 0, work = 0; int pages_s = 0, pages_f = 0, free_s = 0, free_f = 0, host_pages = 0; };
  MemStats mem() const;
  struct Impl;
 private:
  std::unique_ptr<Impl> p_;
};

}  // namespace hive::gemma
