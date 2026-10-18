// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma 4 (Unified) engine: int4 group-32 weights (compressed-tensors pack-quantized, repacked for the decode GEMV), bf16
//   embedding shared with the lm_head, a paged fp8 KV cache (sliding-window layers release pages behind the window), and the Frozen-KV MTP
//   assistant (it reads the target's last sliding / full layer KV), and the encoder-free vision embedder (merged pixel patches → LayerNorm →
//   Dense → LayerNorm → + 2-D position → LayerNorm → RMSNorm → Linear into the text space; an image's soft tokens attend to each other both
//   ways). The math contract is oracle/gemma4_oracle.py. The audio projection of the checkpoint is not loaded.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
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
  std::vector<int> suppress;                      // generation_config suppress_tokens (input-only image / audio end markers)
  // vision (config.json vision_config): merged patch = model_patch_size² · 3 values, mm_embed_dim, position table rows, soft tokens per image
  int v_patch = 0, v_dim = 0, v_posemb = 0, v_max_tokens = 0;
  float v_eps = 1e-6f;
  int image_token = -1;
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

// One image of a prefill: its gh × gw soft tokens are the rows [row, row + gh · gw) of the ids handed to prefill (placeholders), in raster
//   order; patches = bf16 [gh · gw, v_patch] merged pixel patches (values / 255, rows of (y, x, channel)), device memory.
struct GemmaImage {
  int row = 0, gh = 0, gw = 0;
  const __nv_bfloat16* patches = nullptr;
};

class GemmaEngine {
 public:
  GemmaEngine(const std::string& ckpt_dir, const std::string& assistant_dir, int64_t max_ctx, int max_batch);
  ~GemmaEngine();
  const GemmaConfig& cfg() const;
  const Placement& placement() const;
  void init_seq(GemmaSeq& s, int64_t cap);
  void reset_seq(GemmaSeq& s);
  void release_kv(GemmaSeq& s);
  // cut the sequence back to its first n tokens (0 < n < pos) when its KV still covers them — every full-layer position is kept, and the
  //   sliding layers need [n − window, n), which the keep margin (HIVE_GEMMA_SWA_KEEP) holds for a while. false = not possible (unchanged).
  bool rewind(GemmaSeq& s, int64_t n);
  // load a session image's pieces (the order of save_plan) into the sequence's (new) pages
  void load_views(GemmaSeq& s, const std::vector<std::pair<const uint8_t*, size_t>>& v,
                  const std::function<void(void*, const uint8_t*, size_t)>& h2d);
  std::function<bool()> kv_reclaim;               // pool exhausted: evict one idle session (true = retry)
  std::function<void(int layer)> layer_boundary;
  std::function<bool()> layer_boundary_due;
  void warmup(int max_batch, int max_k);
  // prefill T tokens of one sequence (chunked internally, never inside an image); logits of the last row → logits (fp32 [vocab], host or
  //   null); images (or null): their rows take the vision embedding instead of the placeholder token's
  void prefill(GemmaSeq& s, const int32_t* ids, int T, float* logits, const std::vector<GemmaImage>* images = nullptr);
  bool has_vision() const;                        // the checkpoint has the vision embedder and HIVE_GEMMA_VISION is not 0
  int vision_max_tokens() const;                  // soft tokens of one image at most (0 = no vision)
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
