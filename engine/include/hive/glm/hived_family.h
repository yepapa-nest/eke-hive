// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash family for hived (built as hived_glm with -DHIVE_FAMILY_GLM). hived.cpp is shared by every model family: it only sees
//   the names below — Config, Model, ExpertStore, RuntimeOptions, ForwardStats, ImageInput, PrefillPart, Seq, SeqImage, Runtime and the
//   load helpers — which the DeepSeek build takes from model.h / expert_store.h / runtime.h / bulk_load.h. Here they wrap GlmModel,
//   GlmExperts and GlmEngine. Scheduling, sessions, prompt-prefix reuse, sampling and the socket protocol stay hived's own code.
//   Not supported yet in this family (reported, not faked): batched prefill. Images and video frames go through GlmVision (one
//   "image" span per temporal patch — the server sends each pair of video frames as its own span).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hive/common.h"
#include "hive/devbuf.h"
#include "hive/devmem.h"
#include "hive/glm/glm_engine.h"
#include "hive/glm/glm_experts.h"
#include "hive/glm/glm_model.h"
#include "hive/glm/glm_vision.h"
#include "hive/host_image.h"
#include "hive/layer_yield.h"

namespace hive::glm::fam {

constexpr const char* kFamily = "glm5_next";

struct Config {
  int vocab = 0, n_layers = 0;
  std::vector<int> engram_layer_ids;    // none (no engram in GLM)
  int vision_max_tokens = 0, vision_downsample = 1;
  int window = 0;
  GlmModel* glm = nullptr;              // back pointer for the store/runtime wrappers
};

// Checkpoint page cache release after the load (same call hived makes on the DeepSeek checkpoint).
class CkptRef {
 public:
  explicit CkptRef(GlmModel* m) : m_(m) {}
  size_t release_all() { return m_->ckpt().release_all(); }
 private:
  GlmModel* m_;
};

class Model {
 public:
  Model(const std::string& ckpt, int max_layer, int64_t max_ctx);
  const Config& cfg() const { return cfg_; }
  int n_loaded_layers() const { return cfg_.n_layers; }
  bool has_head() const { return true; }
  void load_mtp() {}                    // the NextN layer is loaded by the constructor (HIVE_GLM_MTP_K=0 skips it)
  int n_mtp() const { return 0; }       // (no separate expert layers for the store: the NextN experts live in VRAM, GlmModel)
  CkptRef& ckpt() { return ckpt_ref_; }
  GlmModel& glm() { return *m_; }
 private:
  std::unique_ptr<GlmModel> m_;
  Config cfg_;
  CkptRef ckpt_ref_{nullptr};
};

struct RecLayout { size_t total = 0; };

class ExpertStore {
 public:
  struct CacheStats {
    int resident = 0, unique = 0, pending = 0, duplicates = 0, invalid_mappings = 0;
    uint64_t promotions = 0, commits = 0, evictions = 0, duplicate_skips = 0, h2d_records = 0, d2d_records = 0;
  };
  ExpertStore(const Config& c, int n_layers, size_t cache_bytes, int cpu_threads, int n_mtp, bool defer_slots);
  void load_layer_experts(CkptRef&, int) { ensure_loaded(); }  // the first call loads every routed expert (GlmExperts::load_all)
  template <class... A> void load_bulk(A&&...) { ensure_loaded(); }
  void ensure_loaded();
  void pin_all() {}
  template <class... A> void load_engram(A&&...) {}
  template <class... A> void engram_ssd_finish(A&&...) {}
  std::vector<uint64_t> host_checksums(int) const { return {}; }
  const RecLayout& layout() const { return lay_; }
  int alloc_cache(int slots) { return x_->alloc_cache((size_t)std::max(0, slots) * lay_.total); }
  static int fit_slots(size_t free_bytes, size_t reserve, size_t rec) { return free_bytes > reserve ? (int)((free_bytes - reserve) / rec) : 0; }
  CacheStats cache_stats() const;
  int n_slots() const { return x_->n_slots(); }
  int n_resident() const { return x_->n_resident(); }
  uint64_t total_uses() const { return x_->stats().routed; }
  size_t experts_host_bytes() const { return x_->host_bytes(); }
  size_t engram_host_bytes() const { return 0; }
  std::vector<std::pair<int, double>> coverage(const std::vector<int>&) const { return {}; }
  int staging_slots() const { return x_->staging_slots(); }
  int elastic_slots() const { return 0; }
  int sleep_phys_slots() const { return x_->slots_before_sleep(); }
  std::vector<int32_t> resident_keys_by_score() const { return x_->resident_keys_by_score(); }
  int n_layers() const { return x_->n_moe_layers(); }
  int E() const { return x_->n_experts(); }
  GlmExperts& experts() { return *x_; }
 private:
  std::unique_ptr<GlmExperts> x_;
  RecLayout lay_;
  int threads_;
  bool loaded_ = false;
};

// load helpers (DeepSeek: bulk_load.h / expert_store.h) — the GLM load is one parallel pass (GlmExperts::load_all)
inline int load_par_threads() { return 0; }
inline int load_prefault_threads() { return 0; }
inline bool load_checksum_on() { return false; }
inline int load_opts_from_env(int) { return 0; }
struct DensePrefetch {
  double bytes_v = 0, sec = 0;
  struct { double load() const { return 0; } } bytes;
  void start(const std::string&, int, int) {}
  void join() {}
};
struct HostPrefault {
  static int predict(const std::string&, int, bool) { return 0; }
  static void start(int, int) {}
  static void finish() {}
};

struct RuntimeOptions {
  int max_chunk = 64, max_batch = 8;
  int64_t max_ctx = 4096;
  std::string dump_dir, inject_dir;
  int prefill_threshold = 1024, decode_gpu_share = 0, promote_per_token = 8, promote_misses = 0;
  float promote_miss_ratio = 0.f;
  int warm_cap = 2048;
  bool vision = false, vision_lazy = true, decoder_replay = false, mtp = false;
  int decoder_tail = 0, prefill_tile = 1, vision_max_patches = 0;
  int sampler_cands = 1024;
};

struct ForwardStats {
  int n_routed = 0, n_hit = 0, n_cpu = 0, n_streamed = 0, tail_rows = 0, n_dma_rows = 0, n_promoted = 0;
  double ms_cpu_span = 0, ms_total = 0, ms_cpu_wait = 0, ms_host = 0;
  std::vector<int> row_hit, row_cpu, row_dma;
};

struct ImageInput {
  int start = 0, n_vit_h = 0, n_vit_w = 0;
  std::vector<int8_t> types;
  const bf16* patches_dev = nullptr;
};

struct Seq;
struct PrefillPart {
  Seq* seq = nullptr;
  const int32_t* ids = nullptr;
  int M = 0;
  std::vector<float>* logits_out = nullptr;
  bool upper_needed = true;
  int32_t next = -1;
  int tail_rows = 0;
};

struct Seq : GlmSeq {
  static uint32_t next_uid() { static uint32_t n = 0; return ++n; }
  uint32_t uid = next_uid();
  uint32_t generation = uid;
  std::vector<int64_t> engram_history;  // always empty (hived copies it around)
  bool mtp_hidden_valid = false;
};

// Host image of a sequence: KDA conv + recurrent state (fixed size), DSA latent / indexer caches up to pos.
struct SeqImage {
  int64_t pos = 0;
  bool h_last_valid = false;
  int64_t mtp_written = 0;
  uint32_t generation = 0;
  std::vector<int32_t> tokens;
  std::vector<int64_t> engram_history;
  bool mtp_hidden_valid = false;
  // in Runtime::save_image order; HIVE_CKPT_DELTA shares the still-exact prefix segments of the previous image (host_image.h)
  std::vector<HostImageBuffer> bufs;
  size_t allocated_bytes(std::set<const void*>& seen) const {
    if (!seen.insert(this).second) return 0;
    size_t b = sizeof(*this) + tokens.capacity() * 4 + bufs.capacity() * sizeof(HostImageBuffer);
    for (auto& v : bufs) b += v.allocated_bytes(seen);
    return b;
  }
  size_t bytes() const { std::set<const void*> seen; return allocated_bytes(seen); }
};

class Runtime {
 public:
  Runtime(Model& model, ExpertStore& store, const void* engram_hash, const RuntimeOptions& opt);
  ~Runtime();
  std::unique_ptr<Seq> new_seq() const;
  void reset_seq(Seq& s) const;
  void save_image(const Seq& s, SeqImage& img, const SeqImage* base = nullptr) const;
  void load_image(Seq& s, const SeqImage& img) const;
  void quiesce_after_host_error();
  int32_t forward(Seq& seq, const int32_t* ids, int M, const std::vector<ImageInput>* images, std::vector<float>* logits_out,
                  ForwardStats* stats, bool upper_needed = true);
  int32_t forward(Seq& seq, const int32_t* ids, int M, std::vector<float>* logits_out, ForwardStats* stats) {
    return forward(seq, ids, M, nullptr, logits_out, stats);
  }
  void forward_batch(std::vector<Seq*>& seqs, const int32_t* ids, std::vector<int32_t>& next, std::vector<std::vector<float>>* logits_out,
                     ForwardStats* stats);
  bool tail_mode_for(int) const { return false; }
  // Vision encoder (model.visual, BF16, ~1.1 GB): loaded on the first image and kept until the next sleep (release_vision_if_idle is a
  //   no-op — reloading it for every request would shrink and regrow the expert cache each time).
  bool has_vision() const { return opt_.vision && const_cast<Model&>(model_).glm().ckpt().has("model.visual.patch_embed.proj.weight"); }
  void ensure_vision();
  void release_vision_if_idle() {}
  int max_batch() const { return opt_.max_batch; }
  // prefill "slots" = blocks of max_chunk rows the engine takes per call (layer-major prefill, HIVE_GLM_PREFILL_TILES): hived sizes its
  //   prompt chunks as max_chunk × prefill_slots()
  int prefill_tile() const { return eng_->prefill_tiles(); }
  int prefill_slots() const { return eng_->prefill_tiles(); }
  int host_slots() const { return 0; }
  bool batch_prefill() const { return false; }
  int slots_for(int M) const { const int c = std::max(opt_.max_chunk, 64); return std::max(1, (M + c - 1) / c); }
  void forward_multi(std::vector<PrefillPart>& parts, ForwardStats* stats);
  void set_prefill_pending(bool on) { prefill_pending_ = on; }
  bool prefill_pending() const { return prefill_pending_; }
  std::vector<uint64_t> prefetch_stats() const { return {0, 0, 0}; }
  double prefetch_copy_ms() const { return 0; }
  size_t snapshot_pool_cached_bytes() const { return 0; }
  void trim_snapshot_pool() {}
  int warm_cache() { return 0; }  // warming is paced inside GlmExperts::after_step (warm quota set after each prefill)
  int warm_cap() const { return opt_.warm_cap; }
  void set_warm_cap(int n) { opt_.warm_cap = n; }
  void warm_defer(int) {}
  int warm_left() const { return 0; }
  int image_patch_bytes() const { return GlmVision::kPatchDim * 2; }  // one temporal patch: 3 × 2 × 14 × 14 bf16
  int promote_after_step() { return 0; }
  int warm_from_keys(const std::vector<int32_t>& keys) { return store_.experts().warm_from_keys(keys); }
  // Sleep (hived {"op":"sleep"} — level 1 here; level 2 offloads sessions through save_image, level 3 releases every VMM region: weights,
  //   sessions, scratch, cuBLAS workspace, all allocated through hive/devmem.h when HIVE_SLEEP_VMM=1): the expert slots and the staging ring
  //   are freed; wake re-allocates them (reserve > 0 = HIVE_CACHE_FIT: as many slots as the free VRAM minus reserve holds, else the count
  //   before the sleep) and wake_warm refills them from the resident list hived saved before the sleep.
  size_t sleep_release() { vision_.reset(); return store_.experts().sleep_release(); }
  int wake_restore(size_t reserve, std::string* why) {
    GlmExperts& x = store_.experts();
    const int n = x.wake_alloc(reserve > 0 ? -1 : x.slots_before_sleep(), reserve);
    if (n < 0 && why) {
      size_t fr = 0, tot = 0;
      cudaMemGetInfo(&fr, &tot);
      *why = "not enough free VRAM for the GLM staging ring (free " + std::to_string((long long)(fr >> 20)) + " MiB)";
    }
    return n;
  }
  int wake_warm(const std::vector<int32_t>& keys, const std::function<void(int, int)>& progress = {}) {
    if (store_.experts().asleep()) return 0;
    const int n = warm_from_keys(keys);
    if (progress) progress((int)keys.size(), (int)keys.size());
    return n;
  }
  bool asleep() const { return const_cast<ExpertStore&>(store_).experts().asleep(); }
  // Layer yield (HIVE_LAYER_YIELD — hive/layer_yield.h, the same hooks hived gives the DeepSeek runtime): a prefill forward pauses at layer
  //   boundaries (GlmEngine::layer_boundary) so hived can admit short requests and run decode steps of the active set. Nothing is copied:
  //   the inner forwards use rows the outer prefill does not hold (GlmEngine::yield_enter).
  void set_layer_yield(ly::Hooks h) { ly_.h = std::move(h); }
  bool layer_yield_on() const { return ly_.on(); }
  int layer_yield_kind() const { return ly::kLayer; }  // HIVE_LAYER_YIELD_INTRA is DeepSeek-only: GLM pauses at layer boundaries only (hived admits there as before)
  bool in_layer_yield() const { return ly_.depth > 0; }
  const ly::Stats& layer_yield_stats() const { return ly_.st; }
  double layer_yield_resume_ms() const { return ly_.last; }
  double layer_yield_progress() const { return ly_progress_; }  // (layers done) / layers of the paused prefill forward
  const RuntimeOptions& opt() const { return opt_; }
  int n_cands() const { return opt_.sampler_cands; }
  float* row_inv_temp() { return cand_it_h_; }
  const int32_t* cand_idx(int m) const { return cand_idx_h_ + (size_t)m * opt_.sampler_cands; }
  const float* cand_val(int m) const { return cand_val_h_ + (size_t)m * opt_.sampler_cands; }
  float row_max(int m) const { return cand_max_h_[m]; }
  float row_sumexp(int m) const { return cand_sum_h_[m]; }
  int32_t row_argmax(int m) const { return next_h_[m]; }
  bool mtp_enabled() const { return eng_->mtp_on(); }
  int mtp_block() const { return eng_->mtp_on() ? eng_->mtp_k() : 0; }
  void mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats* stats);
  void forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats);
  void rollback(Seq& seq, int n_keep);
  struct VerifyPart { Seq* seq = nullptr; const int32_t* ids = nullptr; int M = 0; };
  void forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats);
  void rollback_batch(const std::vector<int>& n_keep);
  bool mtp_batch_enabled() const { return mtp_batch_; }
  // {"op":"set","mtp_batch":...} (hived): run-time switch. Batched verify buffers exist only when HIVE_MTP_BATCH was on at startup, so turning it on
  //   is absorbed (stays off) unless the startup value was on; turning it off and back on is allowed.
  void set_mtp_batch(bool on) {
    if (!mtp_batch_seen_) { mtp_batch_seen_ = true; mtp_batch_start_ = mtp_batch_; }
    mtp_batch_ = on && mtp_batch_start_;
  }
  int mtp_batch_rows() const { return mtp_batch_ ? std::min(8, rows_cap_) : 0; }
  // No CUDA graphs here. hived's MTP gate (HIVE_MTP_GATE3) drops step-cost samples taken while this counter moved — the GLM runtime
  //   moves it for verify steps within kWarmSteps decode steps after a prefill, when the expert cache is still filling: the first verify of
  //   a request measured 130 ms for 4 rows against ~50 ms warm and kept the gate on k = 1 (hived trace).
  long graph_captures() const { return warm_samples_; }
  long graph_eager_runs() const { return warm_samples_; }

 private:
  void cands_dev(int M);
  void stats_delta(ForwardStats* st, const GlmCacheStats& before) const;
  void cache_line(int M, const GlmCacheStats& before, double t0);  // HIVE_TRACE_CACHE: the [cache] line of a decode / verify step
  long cache_step_ = 0;
  // HIVE_PROFILE=N: host time inside a decode / verify call, one [call-host <kind>] line per N calls of a kind (tools/hive_monitor.py) —
  //   eng = the engine's forward call (its GPU part is in [fwd-host]) · logits = copying the pinned logits into the caller's vectors ·
  //   cands = sampler candidates on the GPU and their copy back (cands_dev) · rest = the remainder of the call (stats, [cache] line).
  //   Kinds: 0 decode (forward_batch) · 1 verify · 2 verify-batch.
  struct CallHost { long n = 0, rows = 0; double eng = 0, logits = 0, cands = 0, total = 0; };
  CallHost call_host_[3];
  void call_host(int kind, int rows, double t0, double eng_ms, double logits_ms, double cands_ms);  // t0 = call start (mono_ms)
  Model& model_;
  ExpertStore& store_;
  RuntimeOptions opt_;
  std::unique_ptr<GlmEngine> eng_;
  std::unique_ptr<GlmVision> vision_;
  bool prefill_pending_ = false;
  static constexpr int kWarmSteps = 32;
  long warm_samples_ = 0;
  int since_prefill_ = 1 << 30;
  ly::State ly_;
  bool ly_small_ = false;                  // HIVE_LAYER_YIELD_SMALL: also yield inside prefills below prefill_threshold (above the decode-path size)
  std::vector<float> prefill_logits_;      // logits of a prefill's last row (not logits_h_: decode steps run inside its layer yields resize that)
  double ly_progress_ = 0;
  void layer_yield_point();
  int rows_cap_ = 0;
  bool mtp_batch_ = false;                 // HIVE_MTP_BATCH (constructor)
  bool mtp_batch_seen_ = false, mtp_batch_start_ = false;  // set_mtp_batch: the startup value (captured on the first call, after the constructor)
  std::vector<GlmSeq*> vb_seqs_;           // parts of the last forward_verify_batch (for rollback_batch)
  float* cand_it_h_ = nullptr; float* cand_it_d_ = nullptr;
  int32_t* cand_idx_h_ = nullptr; float* cand_val_h_ = nullptr; float* cand_max_h_ = nullptr; float* cand_sum_h_ = nullptr;
  int32_t* next_h_ = nullptr;
  DevBuf cand_idx_, cand_val_, cand_max_, cand_sum_, next_d_;
  // logits of the last decode / verify step, pinned: a copy into pageable memory that hived had just allocated took ~4.5 ms per verify step
  //   (CUPTI timeline, real chat c1: 66 gaps of 4.47 ms in 3 s between the head GEMM and the candidate kernels — ~10 % of the step time)
  float* logits_pin_ = nullptr;
};

}  // namespace hive::glm::fam
