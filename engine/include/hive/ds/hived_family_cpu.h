// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash on the CPU only, for hived (built as hived_ds_cpu with -DHIVE_FAMILY_DS_CPU against engine/cpu_shim, the CUDA
//   stand-in headers: "device" memory is host memory). The surface hived sees is the one of the default (GPU) DeepSeek family (hive/model.h,
//   hive/expert_store.h, hive/runtime.h) in namespace hive::ds::cpufam; here it wraps DsCpuEngine (hive/ds/ds_cpu.h). There is no expert
//   cache: every weight lives in host memory, so the ExpertStore below only answers hived's bookkeeping questions (the engram tables are
//   loaded by the engine; hived still loads the engram hash from --engram DIR and hands it to the Runtime). Sleep releases nothing.
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hive/common.h"
#include "hive/devbuf.h"   // hived uses DevBuf / devmem (host memory under the CUDA stand-in)
#include "hive/devmem.h"
#include "hive/host_image.h"
#include "hive/layer_yield.h"
#include "hive/ds/ds_cpu.h"

namespace hive::ds::cpufam {

constexpr const char* kFamily = "deepseek_v41";

struct Config {
  int vocab = 0, n_layers = 0;
  std::vector<int> engram_layer_ids;
  int vision_max_tokens = 0, vision_downsample = 1;
  int window = 0;
  cpu::DsCpuEngine* eng = nullptr;
};

class CkptRef {
 public:
  size_t release_all() { return 0; }   // the CPU engine copied what it needs into its own memory; its mmap is closed by the engine
};

class Model {
 public:
  Model(const std::string& ckpt, int max_layer, int64_t max_ctx);
  const Config& cfg() const { return cfg_; }
  int n_loaded_layers() const { return cfg_.n_layers; }
  bool has_head() const { return e_->has_head(); }
  void load_mtp() { e_->load_mtp(); }
  int n_mtp() const { return e_->n_mtp(); }
  CkptRef& ckpt() { return ckpt_ref_; }
  cpu::DsCpuEngine& eng() const { return *e_; }
 private:
  std::unique_ptr<cpu::DsCpuEngine> e_;
  Config cfg_;
  CkptRef ckpt_ref_;
};

struct RecLayout { size_t total = 1; };

class ExpertStore {
 public:
  struct CacheStats {
    int resident = 0, unique = 0, pending = 0, duplicates = 0, invalid_mappings = 0;
    uint64_t promotions = 0, commits = 0, evictions = 0, duplicate_skips = 0, h2d_records = 0, d2d_records = 0;
  };
  ExpertStore(const Config&, int, size_t, int, int, bool) {}
  void load_layer_experts(CkptRef&, int) {}
  template <class... A> void load_bulk(A&&...) {}
  void ensure_loaded() {}
  void pin_all() {}
  template <class... A> void load_engram(A&&...) {}
  template <class... A> void engram_ssd_finish(A&&...) {}
  std::vector<uint64_t> host_checksums(int) const { return {}; }
  const RecLayout& layout() const { return lay_; }
  int alloc_cache(int) { return 0; }
  static int fit_slots(size_t, size_t, size_t) { return 0; }
  CacheStats cache_stats() const { return {}; }
  int n_slots() const { return 0; }
  int n_resident() const { return 0; }
  uint64_t total_uses() const { return 0; }
  size_t experts_host_bytes() const { return 0; }
  size_t engram_host_bytes() const { return 0; }
  std::vector<std::pair<int, double>> coverage(const std::vector<int>&) const { return {}; }
  int staging_slots() const { return 0; }
  int elastic_slots() const { return 0; }
  int sleep_phys_slots() const { return 0; }
  std::vector<int32_t> resident_keys_by_score() const { return {}; }
  int n_layers() const { return 0; }
  int E() const { return 0; }
 private:
  RecLayout lay_;
};

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
  int decoder_tail = 128 * 21, prefill_tile = 1, vision_max_patches = 0;
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
  const bf16* patches_dev = nullptr;   // host memory under the CUDA stand-in
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

struct Seq : cpu::CpuSeq {   // engram_history, mtp_hidden_valid, mtp_pos come from CpuSeq
  static uint32_t next_uid() { static uint32_t n = 0; return ++n; }
  uint32_t uid = next_uid();
  uint32_t generation = uid;
};

struct SeqImage {
  int64_t pos = 0;
  uint32_t generation = 0;
  std::vector<int32_t> tokens;
  std::vector<int64_t> engram_history;
  bool mtp_hidden_valid = false;
  std::vector<HostImageBuffer> bufs;   // CpuSeq::save_bytes, one buffer each (engram history included)
  // session store (hive/session_store.h): every member in a fixed order · nothing in flight (host state)
  template <class A> void fields(A& a) { a.i64(pos); a.u32(generation); a.vi32(tokens); a.vi64(engram_history); a.flag(mtp_hidden_valid); a.bufs(bufs); }
  void wait_ready() const {}
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
  ~Runtime() = default;
  std::unique_ptr<Seq> new_seq() const;
  void reset_seq(Seq& s) const;
  void save_image(const Seq& s, SeqImage& img, const SeqImage* base = nullptr) const;
  void load_image(Seq& s, const SeqImage& img) const;
  void quiesce_after_host_error() {}
  int32_t forward(Seq& seq, const int32_t* ids, int M, const std::vector<ImageInput>* images, std::vector<float>* logits_out,
                  ForwardStats* stats, bool upper_needed = true);
  int32_t forward(Seq& seq, const int32_t* ids, int M, std::vector<float>* logits_out, ForwardStats* stats) {
    return forward(seq, ids, M, nullptr, logits_out, stats);
  }
  void forward_batch(std::vector<Seq*>& seqs, const int32_t* ids, std::vector<int32_t>& next, std::vector<std::vector<float>>* logits_out,
                     ForwardStats* stats);
  bool tail_mode_for(int) const { return false; }
  bool has_vision() const { return opt_.vision && model_.eng().has_vision(); }
  void ensure_vision() {}
  void release_vision_if_idle() {}
  int max_batch() const { return opt_.max_batch; }
  int prefill_tile() const { return 1; }
  int prefill_slots() const { return 1; }
  int host_slots() const { return 0; }
  bool batch_prefill() const { return false; }
  int slots_for(int M) const { const int c = std::max(opt_.max_chunk, 64); return std::max(1, (M + c - 1) / c); }
  void forward_multi(std::vector<PrefillPart>& parts, ForwardStats* stats) {
    for (auto& p : parts) p.next = forward(*p.seq, p.ids, p.M, nullptr, p.logits_out, stats, p.upper_needed);
  }
  void set_prefill_pending(bool on) { prefill_pending_ = on; }
  bool prefill_pending() const { return prefill_pending_; }
  std::vector<uint64_t> prefetch_stats() const { return {0, 0, 0}; }
  double prefetch_copy_ms() const { return 0; }
  size_t snapshot_pool_cached_bytes() const { return 0; }
  void trim_snapshot_pool() {}
  int warm_cache() { return 0; }
  int warm_cap() const { return opt_.warm_cap; }
  void set_warm_cap(int n) { opt_.warm_cap = n; }
  void warm_defer(int) {}
  int warm_left() const { return 0; }
  int image_patch_bytes() const { const auto& c = model_.eng().cfg(); return 3 * c.vision_patch * c.vision_patch * 2; }
  int promote_after_step() { return 0; }
  int warm_from_keys(const std::vector<int32_t>&) { return 0; }
  size_t sleep_release() { return 0; }
  int wake_restore(size_t, std::string*) { return 0; }
  int wake_warm(const std::vector<int32_t>&, const std::function<void(int, int)>& = {}) { return 0; }
  bool asleep() const { return false; }
  void set_layer_yield(ly::Hooks h) { ly_.h = std::move(h); }
  void set_progress(ly::Progress p) { prog_ = std::move(p); }  // read progress inside a prefill forward (layer_yield.h Progress)
  void progress_point() { ly::progress_point(prog_, ly_progress_, ly_.depth, ly::steady_ms); }
  bool layer_yield_on() const { return ly_.on(); }
  int layer_yield_kind() const { return ly::kLayer; }
  bool in_layer_yield() const { return ly_.depth > 0; }
  const ly::Stats& layer_yield_stats() const { return ly_.st; }
  double layer_yield_resume_ms() const { return ly_.last; }
  double layer_yield_progress() const { return ly_progress_; }
  const RuntimeOptions& opt() const { return opt_; }
  int n_cands() const { return opt_.sampler_cands; }
  float* row_inv_temp() { return inv_temp_.data(); }
  const int32_t* cand_idx(int m) const { return cand_idx_.data() + (size_t)m * opt_.sampler_cands; }
  const float* cand_val(int m) const { return cand_val_.data() + (size_t)m * opt_.sampler_cands; }
  float row_max(int m) const { return rmax_[m]; }
  float row_sumexp(int m) const { return rsum_[m]; }
  int32_t row_argmax(int m) const { return next_[m]; }
  bool mtp_enabled() const { return model_.eng().mtp_on(); }
  int mtp_block() const { return model_.eng().mtp_on() ? model_.eng().mtp_block() : 0; }
  void mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats* stats);
  void forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats);
  void rollback(Seq& seq, int n_keep) { std::vector<int> k{n_keep}; vb_seqs_ = {&seq}; rollback_batch(k); }
  struct VerifyPart { Seq* seq = nullptr; const int32_t* ids = nullptr; int M = 0; };
  void forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats);
  void rollback_batch(const std::vector<int>& n_keep);
  bool mtp_batch_enabled() const { return mtp_batch_; }
  void set_mtp_batch(bool on) {
    if (!mtp_batch_seen_) { mtp_batch_seen_ = true; mtp_batch_start_ = mtp_batch_; }
    mtp_batch_ = on && mtp_batch_start_;
  }
  int mtp_batch_rows() const { return mtp_batch_ ? 8 : 0; }
  long graph_captures() const { return 0; }
  long graph_eager_runs() const { return 0; }

 private:
  void cands(const float* logits, int M);   // sampler candidates of M logit rows (host)
  void layer_yield_point();
  Model& model_;
  ExpertStore& store_;
  RuntimeOptions opt_;
  bool prefill_pending_ = false;
  ly::State ly_;
  double ly_progress_ = 0;
  ly::Progress prog_;                        // set_progress
  bool mtp_batch_ = false, mtp_batch_seen_ = false, mtp_batch_start_ = false;
  std::vector<Seq*> vb_seqs_;
  std::vector<float> logits_, inv_temp_, cand_val_, rmax_, rsum_;
  std::vector<int32_t> cand_idx_, next_;
};

}  // namespace hive::ds::cpufam
