// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/qwen/hived_family.h"
#include "hive/image_reaper.h"

#include <algorithm>
#include <thread>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

#include "hive/cache_reserve.h"
#include "hive/clock.h"
#include "hive/model_kernels.h"
#include "hive/vram_limit.h"

namespace hive::qwen::fam {

namespace {
template <class T> T* pinned(size_t n) { void* p = nullptr; CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(n, 1) * sizeof(T))); return (T*)p; }
template <class T> T* pinned_mapped(size_t n, T** dev) {
  void* p = nullptr; CUDA_CHECK(cudaHostAlloc(&p, std::max<size_t>(n, 1) * sizeof(T), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)dev, p, 0)); return (T*)p;
}
}  // namespace

// ---- Model / ExpertStore ----------------------------------------------------------------------------------------------------------------
// Hybrid placement — layers [0, N) on the GPU, the rest on the CPU (QwenEngine's tail engine):
//   HIVE_QWEN_GPU_LAYERS = N fixes it; otherwise, under HIVE_VRAM_LIMIT_MB, N is the most layers whose dense weights fit beside the rest of
//   the startup (estimated as hived's plan does, hive/vram_limit.h, at its lowest knob settings — hived's plan then sets the knobs) and a
//   minimum expert cache (HIVE_QWEN_PLAN_CACHE_MB, default 256). Unset limit = every layer on the GPU.
static int gpu_layers_for(const std::string& ckpt) {
  if (const char* v = getenv("HIVE_QWEN_GPU_LAYERS"); v && *v) return atoi(v);
  const size_t limit = vram_limit::limit_bytes();
  if (!limit) return -1;
  const QwenConfig q = QwenConfig::load(ckpt);
  Checkpoint ck(ckpt);
  auto bytes_of = [&](const std::string& prefix, bool skip_experts) {
    size_t b = 0;
    for (const std::string& n : ck.names_with_prefix(prefix)) {
      if (skip_experts && n.find(".mlp.experts.") != std::string::npos) continue;
      if (n.find(".ple.ple_embedding.") != std::string::npos) continue;   // the PLE table stays in host RAM (or on the SSD)
      b += ck.get(n).nbytes;
    }
    return b;
  };
  const std::string P = "model.language_model.";
  std::vector<double> layer(q.n_layers);
  for (int l = 0; l < q.n_layers; ++l) layer[l] = (double)bytes_of(P + "layers." + std::to_string(l) + ".", true);
  // always on the GPU: lm_head and the final mixer (the embedding stays in host RAM)
  const double head = (double)bytes_of("lm_head.", false) + (double)bytes_of(P + "hyper_connection_mixer.", false);
  vram_limit::Knobs k;
  k.max_chunk = 1024; k.max_batch = 1; k.max_sessions = 1; k.mtp = false; k.max_ctx = 32768;
  const double cache = (getenv("HIVE_QWEN_PLAN_CACHE_MB") ? atof(getenv("HIVE_QWEN_PLAN_CACHE_MB")) : 256.0) * 1048576.0;
  const double room = (double)limit - (double)vram_limit::kCtxFallback - head - vram_limit::fixed_bytes(vram_limit::Family::Qwen, k) -
                      (double)cache_reserve_bytes() - cache;
  int n = 0;
  double used = 0;
  while (n < q.n_layers && used + layer[n] <= room) used += layer[n++];
  fprintf(stderr, "[qwen] VRAM limit %.0f MiB: %d of %d layers fit on the GPU (%.0f MiB of dense weights; room %.0f MiB after context %.0f, head %.0f, "
          "lowest-setting work buffers %.0f, reserve %.0f, cache %.0f MiB)%s\n", limit / 1048576.0, n, q.n_layers, used / 1048576.0, room / 1048576.0,
          vram_limit::kCtxFallback / 1048576.0, head / 1048576.0, vram_limit::fixed_bytes(vram_limit::Family::Qwen, k) / 1048576.0,
          cache_reserve_bytes() / 1048576.0, cache / 1048576.0, n < q.n_layers ? " — the rest run on the CPU" : "");
  return n >= q.n_layers ? -1 : std::max(1, n);
}
static std::unique_ptr<QwenModel> make_model(const std::string& ckpt) {
  const int n = gpu_layers_for(ckpt);
  // a CPU tail has no MTP (QwenModel skips it): say so to everything that reads the switch (hived's VRAM plan, the runtime)
  if (n >= 0 && n < QwenConfig::load(ckpt).n_layers) setenv("HIVE_QWEN_MTP_K", "0", 1);
  return std::make_unique<QwenModel>(ckpt, !(getenv("HIVE_QWEN_MTP_K") && strcmp(getenv("HIVE_QWEN_MTP_K"), "0") == 0), n);
}
Model::Model(const std::string& ckpt, int /*max_layer*/, int64_t /*max_ctx*/) : m_(make_model(ckpt)) {
  const QwenConfig& q = m_->cfg();
  cfg_.vocab = q.vocab; cfg_.n_layers = q.n_layers; cfg_.qm = m_.get();
  // hived's image admission: at most vision_max_tokens · downsample² patches per span (the server keeps every span within 8000 merged tokens)
  cfg_.vision_max_tokens = 8000; cfg_.vision_downsample = q.vision_merge;
  ckpt_ref_ = CkptRef(m_.get());
}

ExpertStore::ExpertStore(const Config& c, int, size_t cache_bytes, int cpu_threads, int, bool defer_slots) : threads_(std::max(1, cpu_threads)) {
  HIVE_CHECK(c.qm, "qwen store without a model");
  qm_ = c.qm;
  const QwenConfig& q = c.qm->cfg();
  glm::ExpertSpec sp;
  sp.hidden = q.hidden; sp.inter = q.moe_inter; sp.n_routed = q.n_experts; sp.n_act = q.n_act; sp.limit = 0.f;  // plain SiLU (no clamp)
  for (int l = 0; l < c.qm->gpu_layers(); ++l) sp.moe_layers.push_back(l);   // every layer is MoE (a CPU tail holds its own experts)
  sp.ckpt = &c.qm->ckpt();
  x_ = std::make_unique<GlmExperts>(sp, 0, threads_, 8);
  lay_.total = x_->layout().total;
  if (!defer_slots && cache_bytes > 0) alloc_cache((int)(cache_bytes / lay_.total));
}
void ExpertStore::ensure_loaded() {
  if (loaded_) return;
  x_->load_all(std::max(16, threads_));
  loaded_ = true;
}
// HIVE_QWEN_VRAM_MB: the process may use at most this much device memory in all — the expert slots get what the budget leaves after the
//   weights, the session pool and HIVE_CACHE_RESERVE_MB (the headroom the KV growth and the vision encoder take from)
int ExpertStore::clamp_slots(int slots) const {
  slots = std::min(std::max(0, slots), x_->n_keys());
  if (const char* v = getenv("HIVE_QWEN_VRAM_MB"); v && *v && atof(v) > 0 && qm_) {
    const size_t budget = (size_t)(atof(v) * 1048576.0);
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    const size_t used = qm_->free_at_load() > fr ? qm_->free_at_load() - fr : 0;
    // the staging ring (allocated before the slots) is part of `used`; the reserve stays free
    const size_t left = budget > used + cache_reserve_bytes() ? budget - used - cache_reserve_bytes() : 0;
    const int cap = (int)(left / lay_.total);
    if (slots > cap) {
      fprintf(stderr, "[qwen] VRAM budget %.0f MiB: expert cache %d → %d slots (used %.0f MiB · reserve %.0f MiB)\n", budget / 1048576.0, slots, cap,
              used / 1048576.0, cache_reserve_bytes() / 1048576.0);
      slots = cap;
    }
  }
  return slots;
}

int ExpertStore::alloc_cache(int slots) {
  const int n = x_->alloc_cache((size_t)clamp_slots(slots) * lay_.total);
  fprintf(stderr, "[qwen] expert cache: %d of %d experts can be resident (%.1f %%)%s\n", n, x_->n_keys(), 100.0 * n / std::max(1, x_->n_keys()),
          n >= x_->n_keys() ? " — every expert fits: decode routes on the device once they are loaded" : "");
  return n;
}

int Runtime::wake_restore(size_t reserve, std::string* why) {
  GlmExperts& x = store_.experts();
  // the slot count after the wake follows the startup rules: the free VRAM minus the reserve (HIVE_CACHE_FIT) or the count before the
  //   sleep, then the expert count and the VRAM budget (ExpertStore::clamp_slots) — GlmExperts::wake_alloc(-1) would apply neither
  int slots = x.slots_before_sleep();
  if (reserve > 0) {
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    const size_t ring = (size_t)x.staging_slots() * x.layout().total;  // allocated by wake_alloc before the slots
    slots = fr > reserve + ring ? (int)((fr - reserve - ring) / x.layout().total) : 0;
  }
  const int n = x.wake_alloc(store_.clamp_slots(slots), 0);
  eng_->clear_graphs();   // device buffers may have moved
  if (n < 0 && why) {
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    *why = "not enough free VRAM for the expert staging ring (free " + std::to_string((long long)(fr >> 20)) + " MiB)";
  }
  return n;
}

ExpertStore::CacheStats ExpertStore::cache_stats() const {
  CacheStats s;
  s.resident = s.unique = x_->n_resident();
  const GlmCacheStats& g = x_->stats();
  s.promotions = g.promoted; s.commits = g.promoted; s.evictions = g.evicted; s.h2d_records = g.promoted + g.streamed;
  return s;
}

// ---- Runtime ---------------------------------------------------------------------------------------------------------------
Runtime::Runtime(Model& model, ExpertStore& store, const void*, const RuntimeOptions& opt) : model_(model), store_(store), opt_(opt) {
  opt_.max_batch = std::max(1, std::min(opt_.max_batch, kDecRows / 4));  // decode / verify kernels take kDecRows (64) rows: 16 sequences × (MTP k 3 + 1)
  store_.ensure_loaded();
  eng_ = std::make_unique<QwenEngine>(model_.qm(), store_.experts(), std::max(opt_.max_chunk, 64));
  rows_cap_ = std::max(opt_.max_batch, eng_->mtp_on() ? eng_->mtp_k() + 1 : 1);
  // HIVE_MTP_BATCH (same switch as DeepSeek): hived verifies the drafts of several sequences in one step (forward_verify_batch) —
  //   linear-attention / PLE snapshots for max_batch parts, rows up to 8 (the decode kernels' M limit)
  {
    const char* mb = getenv("HIVE_MTP_BATCH");
    mtp_batch_ = mb && *mb && strcmp(mb, "0") != 0 && eng_->mtp_on() && opt_.max_batch >= 2;
    if (mtp_batch_) { eng_->enable_batch_verify(opt_.max_batch); rows_cap_ = std::max(rows_cap_, std::min(kDecRows, opt_.max_batch * (eng_->mtp_k() + 1))); }
  }
  const int NC = std::max(1, opt_.sampler_cands);
  cand_it_h_ = pinned_mapped<float>(rows_cap_, &cand_it_d_);
  for (int i = 0; i < rows_cap_; ++i) cand_it_h_[i] = 1.f;
  cand_idx_h_ = pinned<int32_t>((size_t)rows_cap_ * NC); cand_val_h_ = pinned<float>((size_t)rows_cap_ * NC);
  cand_max_h_ = pinned<float>(rows_cap_); cand_sum_h_ = pinned<float>(rows_cap_); next_h_ = pinned<int32_t>(rows_cap_);
  logits_pin_ = pinned<float>((size_t)rows_cap_ * model_.qm().cfg().vocab);
  cand_idx_.alloc((size_t)rows_cap_ * NC * 4); cand_val_.alloc((size_t)rows_cap_ * NC * 4);
  cand_max_.alloc((size_t)rows_cap_ * 4); cand_sum_.alloc((size_t)rows_cap_ * 4); next_d_.alloc((size_t)rows_cap_ * 4);
  // HIVE_LAYER_YIELD: hc-stream rows for the forwards run inside a yield while a chunked prefill keeps its rows in h_ — the largest of them
  //   is a short request hived admits there (below prefill_threshold rows) or a decode / verify step
  if (ly::parse_period(getenv("HIVE_LAYER_YIELD")) > 0) {
    eng_->alloc_yield_rows(std::max(opt_.prefill_threshold, rows_cap_));
    const char* v = getenv("HIVE_LAYER_YIELD_SMALL");
    ly_small_ = v && *v && strcmp(v, "0") != 0;
  }
  fprintf(stderr, "[qwen] runtime: family %s · max batch %d · chunk %d · context %lld · sampler candidates %d\n", kFamily, opt_.max_batch,
          std::max(opt_.max_chunk, 64), (long long)opt_.max_ctx, opt_.sampler_cands);
}

Runtime::~Runtime() {
  for (void* p : {(void*)cand_it_h_, (void*)cand_idx_h_, (void*)cand_val_h_, (void*)cand_max_h_, (void*)cand_sum_h_, (void*)next_h_, (void*)logits_pin_})
    if (p) cudaFreeHost(p);
  if (ckpt_thr_.joinable()) {
    { std::lock_guard<std::mutex> lk(ckpt_mu_); ckpt_stop_ = true; }
    ckpt_cv_.notify_all();
    ckpt_thr_.join();
  }
  for (int i = 0; i < 2; ++i) { if (cstage_[i]) cudaFreeHost(cstage_[i]); if (cstage_ev_[i]) cudaEventDestroy(cstage_ev_[i]); }
  if (ckpt_st_) cudaStreamDestroy(ckpt_st_);
  for (int i = 0; i < 2; ++i) { if (stage_[i]) cudaFreeHost(stage_[i]); if (stage_ev_[i]) cudaEventDestroy(stage_ev_[i]); }
}

// HIVE_QWEN_TRACE_HOST=1: one [qwen-host] line per session-level call (where a request's time goes before its prefill starts)
static bool trace_host() { static const bool on = getenv("HIVE_QWEN_TRACE_HOST") && atoi(getenv("HIVE_QWEN_TRACE_HOST")) > 0; return on; }

std::unique_ptr<Seq> Runtime::new_seq() const {
  const double t0 = hive::mono_ms();
  auto s = std::make_unique<Seq>();
  eng_->init_seq(*s, opt_.max_ctx);
  if (trace_host()) fprintf(stderr, "[qwen-host] new_seq %.2f ms\n", hive::mono_ms() - t0);
  return s;
}

void Runtime::reset_seq(Seq& s) const {
  const double t0 = hive::mono_ms();
  struct T { double t0; ~T() { if (trace_host()) fprintf(stderr, "[qwen-host] reset_seq %.2f ms\n", hive::mono_ms() - t0); } } tr{t0};
  s.ckpt_fence->wait();   // a checkpoint copy may still read the rows reset_seq releases
  eng_->reset_seq(s);
  s.generation = Seq::next_uid();
  s.engram_history.clear();
  s.mtp_hidden_valid = false;
}

// Image layout: per linear layer conv, state (whole) · the PLE conv state · rope coordinates [0, pos) · per attention layer k, v, ik rows
//   [0, pos) and group keys [0, pos/4) · with MTP: its k, v, ik, group-key rows and the last hidden stream.
// HIVE_CKPT_DELTA (the DeepSeek switch, as in the GLM family): when `base` is an earlier image of the same sequence (same generation, its
//   tokens a prefix of the sequence's), the rows it already holds are still exact — rows below a committed position are never rewritten —
//   so their segments are shared and only the new rows are copied. MTP rows are shared only below the base's mtp_written; the fixed-size
//   states change every step and are always copied.
// Session images live in pageable host memory: cudaMemcpy from / to it ran at ~2-3 GB/s (save_image 52 ms per finished c1 request, on
//   the next admission's path when the pool evicts). Two pinned 16 MiB staging buffers: the DMA of chunk i+1 overlaps the host copy of chunk i.
// host copy of a staging chunk on 4 threads (one thread ran ~8-10 GB/s incl. first-touch faults of the image buffers — a 64K prompt-end
//   checkpoint is ~2 GB, copied while the request decodes)
static void par_memcpy(void* dst, const void* src, size_t n) {
  constexpr int kT = 4;
  if (n < ((size_t)2 << 20)) { std::memcpy(dst, src, n); return; }
  std::thread th[kT - 1];
  const size_t part = (n / kT + 4095) / 4096 * 4096;
  for (int t = 1; t < kT; ++t) {
    const size_t o = std::min(n, part * t), e = std::min(n, part * (t + 1));
    th[t - 1] = std::thread([=] { if (e > o) std::memcpy(static_cast<uint8_t*>(dst) + o, static_cast<const uint8_t*>(src) + o, e - o); });
  }
  std::memcpy(dst, src, std::min(n, part));
  for (auto& x : th) x.join();
}
void Runtime::stage_d2h(void* dst, const void* src, size_t n) const {
  constexpr size_t kChunk = (size_t)16 << 20;
  if (!stage_[0]) for (int i = 0; i < 2; ++i) { CUDA_CHECK(cudaMallocHost(&stage_[i], kChunk)); CUDA_CHECK(cudaEventCreateWithFlags(&stage_ev_[i], cudaEventDisableTiming)); }
  cudaStream_t st = eng_->stream();
  const size_t nch = (n + kChunk - 1) / kChunk;
  auto issue = [&](size_t c) {
    const size_t off = c * kChunk, len = std::min(kChunk, n - off);
    CUDA_CHECK(cudaMemcpyAsync(stage_[c & 1], static_cast<const uint8_t*>(src) + off, len, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaEventRecord(stage_ev_[c & 1], st));
  };
  if (nch) issue(0);
  for (size_t c = 0; c < nch; ++c) {
    if (c + 1 < nch) issue(c + 1);
    CUDA_CHECK(cudaEventSynchronize(stage_ev_[c & 1]));
    const size_t off = c * kChunk, len = std::min(kChunk, n - off);
    par_memcpy(static_cast<uint8_t*>(dst) + off, stage_[c & 1], len);
  }
}
void Runtime::stage_h2d(void* dst, const void* src, size_t n) const {
  constexpr size_t kChunk = (size_t)16 << 20;
  if (!stage_[0]) for (int i = 0; i < 2; ++i) { CUDA_CHECK(cudaMallocHost(&stage_[i], kChunk)); CUDA_CHECK(cudaEventCreateWithFlags(&stage_ev_[i], cudaEventDisableTiming)); }
  cudaStream_t st = eng_->stream();
  for (size_t off = 0, c = 0; off < n; off += kChunk, ++c) {
    const size_t len = std::min(kChunk, n - off);
    if (c >= 2) CUDA_CHECK(cudaEventSynchronize(stage_ev_[c & 1]));   // the DMA that last read this buffer is done
    par_memcpy(stage_[c & 1], static_cast<const uint8_t*>(src) + off, len);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(dst) + off, stage_[c & 1], len, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaEventRecord(stage_ev_[c & 1], st));
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
}

// device snapshot slot for a job's per-step buffers (2 slots; waits for a free one)
int Runtime::ckpt_slot(size_t bytes) const {
  std::unique_lock<std::mutex> lk(ckpt_mu_);
  if (!ckpt_thr_.joinable()) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&ckpt_st_, cudaStreamNonBlocking));
    ckpt_thr_ = std::thread([this] { ckpt_worker(); });
  }
  ckpt_cv_.wait(lk, [&] { return !ckpt_busy_[0] || !ckpt_busy_[1]; });
  const int sl = ckpt_busy_[0] ? 1 : 0;
  ckpt_busy_[sl] = true;
  lk.unlock();
  if (ckpt_scratch_[sl].n < bytes) ckpt_scratch_[sl].alloc(bytes);
  return sl;
}
void Runtime::ckpt_worker() const {
  constexpr size_t kChunk = (size_t)16 << 20;
  CUDA_CHECK(cudaSetDevice(0));
  for (int i = 0; i < 2; ++i) { CUDA_CHECK(cudaMallocHost(&cstage_[i], kChunk)); CUDA_CHECK(cudaEventCreateWithFlags(&cstage_ev_[i], cudaEventDisableTiming)); }
  for (;;) {
    std::shared_ptr<CkptJob> job;
    {
      std::unique_lock<std::mutex> lk(ckpt_mu_);
      ckpt_cv_.wait(lk, [&] { return ckpt_stop_ || !ckpt_q_.empty(); });
      if (ckpt_q_.empty()) return;
      job = ckpt_q_.front(); ckpt_q_.pop_front();
    }
    CUDA_CHECK(cudaStreamWaitEvent(ckpt_st_, job->ready, 0));
    size_t c = 0;   // global chunk counter over all parts: staging buffer c & 1
    std::vector<std::pair<uint8_t*, size_t>> pend(2);
    auto drain = [&](int b) { if (pend[b].first) { CUDA_CHECK(cudaEventSynchronize(cstage_ev_[b])); par_memcpy(pend[b].first, cstage_[b], pend[b].second); pend[b] = {}; } };
    for (auto& [dst, src, n] : job->parts)
      for (size_t off = 0; off < n; off += kChunk, ++c) {
        const int b = (int)(c & 1);
        drain(b);   // the buffer's previous chunk is copied out before it is refilled
        const size_t len = std::min(kChunk, n - off);
        CUDA_CHECK(cudaMemcpyAsync(cstage_[b], src + off, len, cudaMemcpyDeviceToHost, ckpt_st_));
        CUDA_CHECK(cudaEventRecord(cstage_ev_[b], ckpt_st_));
        pend[b] = {dst + off, len};
        drain(b ^ 1);
      }
    drain(0); drain(1);
    CUDA_CHECK(cudaEventDestroy(job->ready));
    { std::lock_guard<std::mutex> lk(ckpt_mu_); ckpt_busy_[job->slot] = false; }
    ckpt_cv_.notify_all();
    job->img_fence->done();
    job->seq_fence->done();
  }
}

void Runtime::save_image(const Seq& s, SeqImage& img, const SeqImage* base) const {
  const QwenConfig& c = model_.qm().cfg();
  const double t0 = hive::mono_ms();
  struct T { double t0; ~T() { if (trace_host()) fprintf(stderr, "[qwen-host] save_image %.2f ms\n", hive::mono_ms() - t0); } } tr{t0};
  const size_t n_bufs = s.conv.size() * 2 + (s.ple_state.p ? 1 : 0) + 1 + s.attn.size() * 4 + (s.h_last.p ? 5 : 0);
  static const bool delta = [] { const char* v = getenv("HIVE_CKPT_DELTA"); return v && *v && strcmp(v, "0") != 0; }();
  if (!delta || !base || base == &img || base->generation != s.generation || base->pos > s.pos || base->bufs.size() != n_bufs ||
      base->tokens.size() > s.tokens.size() || !std::equal(base->tokens.begin(), base->tokens.end(), s.tokens.begin())) base = nullptr;
  if (base && base->fence) base->fence->wait();   // its segments are shared below: their copies must be complete
  // HIVE_QWEN_CKPT_ASYNC (default on): the host copy runs on a worker thread and stream — the per-step state (conv, recurrent, PLE, last
  //   hidden) is first snapshotted on the device (~120 MB, ~0.1 ms); the KV rows below pos are never rewritten while the copy runs (reset_seq,
  //   load_image and ~Seq wait for it). A 64K prompt-end checkpoint (2.1 GB) cost the decode 1.3 s synchronous (pageable) / 0.27 s staged.
  static const bool async = !(getenv("HIVE_QWEN_CKPT_ASYNC") && strcmp(getenv("HIVE_QWEN_CKPT_ASYNC"), "0") == 0);
  std::shared_ptr<CkptJob> job;
  size_t mut_bytes = 0, kv_all = 0;
  bool snap_all = false;
  if (async) {
    for (size_t i = 0; i < s.conv.size(); ++i) mut_bytes += s.conv[i].n + s.state[i].n;
    mut_bytes += s.ple_state.n + s.h_last.n + 256 * 8;
    // short sessions: the KV rows are snapshotted on the device too (≤ HIVE_QWEN_CKPT_SNAP_MB in all, default 192 ≈ 2.4K tokens; a 1K-token image is 150 MB), so the job no longer reads
    //   the sequence and reset_seq / load_image need not wait for it — a pool eviction's copy blocked the next admission 33 ms (c1, 1K tokens)
    {
      const size_t Pp = (size_t)s.pos, rkv = (size_t)c.kv_width() * 2, ri = (size_t)c.idx_dim * 2, Rr = (size_t)c.idx_ratio;
      const size_t per_attn = 2 * Pp * rkv + Pp * ri + (Pp / Rr) * ri;
      kv_all = Pp * 12 + per_attn * (s.attn.size() + (s.h_last.p ? 1 : 0)) + 256 * (4 * s.attn.size() + 8);
      static const size_t budget = (size_t)(getenv("HIVE_QWEN_CKPT_SNAP_MB") ? atof(getenv("HIVE_QWEN_CKPT_SNAP_MB")) : 192.0) * 1048576;
      snap_all = mut_bytes + kv_all <= budget;
    }
    job = std::make_shared<CkptJob>();
    job->slot = ckpt_slot(snap_all ? mut_bytes + kv_all : mut_bytes);
  } else {
    CUDA_CHECK(cudaStreamSynchronize(eng_->stream()));
  }
  size_t snap_off = 0;
  auto snap = [&](const void* p, size_t n) -> const uint8_t* {   // device snapshot of a per-step buffer (main stream, ordered after its writers)
    uint8_t* d = ckpt_scratch_[job->slot].as<uint8_t>() + snap_off;
    CUDA_CHECK(cudaMemcpyAsync(d, p, n, cudaMemcpyDeviceToDevice, eng_->stream()));
    snap_off += (n + 255) / 256 * 256;
    return d;
  };
  SeqImage out;
  out.pos = s.pos; out.generation = s.generation; out.tokens = s.tokens; out.mtp_hidden_valid = s.mtp_hidden_valid; out.rope_delta = s.rope_delta;
  out.h_last_valid = s.h_last_valid; out.mtp_written = s.mtp_written;
  out.bufs.reserve(n_bufs);
  size_t copied = 0, total = 0;
  auto grab = [&](const void* p, size_t bytes, size_t keep, bool per_step = false) {  // keep = leading bytes of the base's buffer that are still exact
    HostImageBuffer dst;
    size_t prefix = 0;
    if (base) {
      keep = std::min(keep, bytes);
      for (const auto& sg : base->bufs[out.bufs.size()].segments) {
        if (prefix >= keep) break;
        const size_t n = std::min(sg.n, keep - prefix);
        dst.segments.push_back({sg.data, n});
        prefix += n;
      }
    }
    if (bytes > prefix) {
      std::shared_ptr<uint8_t> q = hive::reaped_bytes(bytes - prefix);   // freed off the engine thread (hive/image_reaper.h)
      if (job) {
        const uint8_t* src = static_cast<const uint8_t*>(p) + prefix;
        if (per_step || snap_all) src = snap(src, bytes - prefix);
        job->parts.emplace_back(q.get(), src, bytes - prefix);
        job->keep.push_back(q);   // the piece outlives the copy even if the image is dropped first (reaper safety rule)
      } else {
        stage_d2h(q.get(), static_cast<const uint8_t*>(p) + prefix, bytes - prefix);
      }
      dst.segments.push_back({std::move(q), bytes - prefix});
      copied += bytes - prefix;
    }
    total += bytes;
    out.bufs.push_back(std::move(dst));
  };
  const size_t P = (size_t)s.pos, BP = base ? (size_t)base->pos : 0, R = (size_t)c.idx_ratio;
  const size_t BM = base ? (size_t)std::min<int64_t>(base->pos, base->mtp_written) : 0;
  const size_t row_kv = (size_t)c.kv_width() * 2, row_i = (size_t)c.idx_dim * 2;
  for (size_t i = 0; i < s.conv.size(); ++i) { grab(s.conv[i].p, s.conv[i].n, 0, true); grab(s.state[i].p, s.state[i].n, 0, true); }
  if (s.ple_state.p) grab(s.ple_state.p, s.ple_state.n, 0, true);
  grab(s.rope.p, P * 12, BP * 12);
  for (const AttnCache& a : s.attn) {
    grab(a.k.p, P * row_kv, BP * row_kv); grab(a.v.p, P * row_kv, BP * row_kv); grab(a.ik.p, P * row_i, BP * row_i);
    grab(a.gk.p, (P / R) * row_i, (BP / R) * row_i);
  }
  if (s.h_last.p) {
    grab(s.mtp.k.p, P * row_kv, BM * row_kv); grab(s.mtp.v.p, P * row_kv, BM * row_kv); grab(s.mtp.ik.p, P * row_i, BM * row_i);
    grab(s.mtp.gk.p, (P / R) * row_i, (BM / R) * row_i); grab(s.h_last.p, s.h_last.n, 0, true);
  }
  if (s.tail) { out.tail = s.tail->save_bytes(); for (const auto& t : out.tail) { copied += t.size(); total += t.size(); } }
  if (job) {
    CUDA_CHECK(cudaEventCreateWithFlags(&job->ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(job->ready, eng_->stream()));   // after the snapshots and every earlier write of the KV rows
    job->seq_fence = snap_all ? std::make_shared<CkptFence>() : s.ckpt_fence;   // snap_all: the job reads only the device snapshot
    job->img_fence = std::make_shared<CkptFence>();
    job->seq_fence->add(); job->img_fence->add();
    out.fence = job->img_fence;
    { std::lock_guard<std::mutex> lk(ckpt_mu_); ckpt_q_.push_back(job); }
    ckpt_cv_.notify_all();
  }
  img = std::move(out);
  static const bool trace = getenv("HIVE_PROFILE") && atoi(getenv("HIVE_PROFILE")) > 0;
  if (trace)
    fprintf(stderr, "[qwen-ckpt] pos %lld base %lld · copied %.1f of %.1f MB · %.1f ms\n", (long long)s.pos, base ? (long long)base->pos : -1LL, copied / 1e6,
            total / 1e6, hive::mono_ms() - t0);
}

void Runtime::load_image(Seq& s, const SeqImage& img) const {
  struct T { double t0; ~T() { if (trace_host()) fprintf(stderr, "[qwen-host] load_image %.2f ms\n", hive::mono_ms() - t0); } } tr{hive::mono_ms()};
  HIVE_CHECK(img.pos <= s.cap, "qwen image longer than the sequence capacity");
  if (img.fence) img.fence->wait();   // its copies finished
  s.ckpt_fence->wait();               // no copy still reads this sequence's rows (they are rewritten below)
  eng_->kv_reserve(s, std::min<int64_t>(s.cap, img.pos + QwenEngine::kKvSlack));
  size_t k = 0;
  auto put = [&](void* p, size_t cap) {
    HIVE_CHECK(k < img.bufs.size() && img.bufs[k].size() <= cap, "qwen image layout mismatch");
    size_t off = 0;
    for (const auto& sg : img.bufs[k].segments) {
      stage_h2d(static_cast<uint8_t*>(p) + off, sg.data.get(), sg.n);
      off += sg.n;
    }
    ++k;
  };
  for (size_t i = 0; i < s.conv.size(); ++i) { put(s.conv[i].p, s.conv[i].n); put(s.state[i].p, s.state[i].n); }
  if (s.ple_state.p) put(s.ple_state.p, s.ple_state.n);
  put(s.rope.p, s.rope.mapped);
  for (AttnCache& a : s.attn) { put(a.k.p, a.k.mapped); put(a.v.p, a.v.mapped); put(a.ik.p, a.ik.mapped); put(a.gk.p, a.gk.mapped); }
  if (s.h_last.p) { put(s.mtp.k.p, s.mtp.k.mapped); put(s.mtp.v.p, s.mtp.v.mapped); put(s.mtp.ik.p, s.mtp.ik.mapped); put(s.mtp.gk.p, s.mtp.gk.mapped); put(s.h_last.p, s.h_last.n); }
  HIVE_CHECK(k == img.bufs.size(), "qwen image layout mismatch (count)");
  HIVE_CHECK(!s.tail == img.tail.empty(), "qwen image layout mismatch (CPU-tail state)");
  if (s.tail) s.tail->load_bytes(img.tail);
  s.pos = img.pos; s.tokens = img.tokens; s.generation = img.generation; s.broken = false; s.rope_delta = img.rope_delta;
  s.engram_history.clear();
  s.h_last_valid = img.h_last_valid && s.h_last.p; s.mtp_written = img.mtp_written; s.mtp_hidden_valid = s.h_last_valid && eng_->mtp_on();
}

// Every expert fits the cache (the whole card, no budget): the slots are allocated after this runtime (hived sizes them after the session
//   pool), so the missing records are copied in on the first call that finds room for all of them — and again after the cache shrank for a
//   long context and grew back. Decode then routes on the device (QwenEngine::moe_device).
void Runtime::fill_cache() {
  GlmExperts& x = store_.experts();
  // mapped device routing (QwenEngine, HIVE_QWEN_MAPPED_ROUTE) with a cache that cannot hold every expert: its free slots are filled (every
  //   layer alike, expert ids in turn — no scores yet at startup); the step bookkeeping then promotes the hot ones
  if (!x.asleep() && x.n_slots() < x.n_keys() && eng_ && eng_->mapped_route() && x.n_resident() < x.n_slots() - x.n_pending()) {
    std::vector<int32_t> keys;
    const int E = x.n_experts(), L = x.n_moe_layers();
    for (int e = 0; e < E; ++e) for (int li = 0; li < L; ++li) if (x.slot_of(li * E + e) < 0) keys.push_back(li * E + e);
    const double t0 = hive::mono_ms();
    const int n = x.warm_from_keys(keys);
    if (n > 0)
      fprintf(stderr, "[qwen] expert cache warmed: %d experts loaded in %.1f s (%d / %d resident, %d slots)\n", n, (hive::mono_ms() - t0) / 1000,
              x.n_resident(), x.n_keys(), x.n_slots());
    return;
  }
  if (x.asleep() || x.n_slots() < x.n_keys()) return;
  if (x.n_resident() == x.n_keys()) return;
  std::vector<int32_t> miss;
  for (int k = 0; k < x.n_keys(); ++k) if (x.slot_of(k) < 0) miss.push_back(k);
  const double t0 = hive::mono_ms();
  const int n = x.warm_from_keys(miss);
  fprintf(stderr, "[qwen] expert cache filled: %d of %zu missing experts loaded in %.1f s (%d / %d resident)\n", n, miss.size(), (hive::mono_ms() - t0) / 1000,
          x.n_resident(), x.n_keys());
}

void Runtime::quiesce_after_host_error() { cudaStreamSynchronize(eng_->stream()); }

void Runtime::ensure_vision() {
  if (vision_) return;
  const double t0 = hive::mono_ms();
  eng_->make_room((size_t)1100 << 20);  // the weights are ~0.9 GB (BF16)
  vision_ = std::make_unique<QwenVision>(model_.qm());
  fprintf(stderr, "[qwen] vision encoder loaded: %.0f MiB in %.0f ms\n", vision_->weight_bytes() / 1048576.0, hive::mono_ms() - t0);
}

// HIVE_QWEN_PROF (measurement only): per call kind (decode by rows, verify by rows, draft), the wall time of hived's calls and the expert
//   counters, accumulated and printed every 400 calls of a kind. Engine thread only.
namespace {
struct ProfKind {
  long n = 0;
  double wall = 0, moe = 0;
  GlmCacheStats x;
  long dev = 0;
};
std::map<std::string, ProfKind>& prof_kinds() { static std::map<std::string, ProfKind> m; return m; }
bool prof_on() { static const bool on = getenv("HIVE_QWEN_PROF") && atoi(getenv("HIVE_QWEN_PROF")) > 0; return on; }
void prof_add(const std::string& kind, double wall, const QwenForwardStats& e0, const QwenForwardStats& e1, const GlmCacheStats& x0, const GlmCacheStats& x1) {
  ProfKind& k = prof_kinds()[kind];
  ++k.n; k.wall += wall; k.moe += e1.ms_moe - e0.ms_moe; k.dev += e1.device_route_steps - e0.device_route_steps;
  auto dx = [&](uint64_t GlmCacheStats::*f) { k.x.*f += x1.*f - x0.*f; };
  for (auto f : {&GlmCacheStats::routed, &GlmCacheStats::hit, &GlmCacheStats::cpu, &GlmCacheStats::promoted, &GlmCacheStats::cpu_jobs}) dx(f);
  auto dt = [&](double GlmCacheStats::*f) { k.x.*f += x1.*f - x0.*f; };
  for (auto f : {&GlmCacheStats::ms_cpu, &GlmCacheStats::ms_layer}) dt(f);
  if (k.n % 400) return;
  const double n = (double)k.n;
  fprintf(stderr, "[qwen-prof] %s · calls %ld · wall %.2f ms/call · device-routed %.0f %% · hit %.1f %% of %.1f routed/call · cpu %.1f/call · "
                  "host MoE %.2f ms/call (CPU experts %.2f) · promoted %.2f/call\n",
          kind.c_str(), k.n, k.wall / n, 100.0 * k.dev / n, k.x.routed ? 100.0 * k.x.hit / k.x.routed : 0.0, k.x.routed / n, k.x.cpu / n, k.moe / n,
          k.x.ms_cpu / n, k.x.promoted / n);
}
}  // namespace

void Runtime::stats_delta(ForwardStats* st, const GlmCacheStats& b) const {
  if (!st) return;
  const GlmCacheStats& a = store_.experts().stats();
  st->n_routed += (int)(a.routed - b.routed); st->n_hit += (int)(a.hit - b.hit); st->n_cpu += (int)(a.cpu - b.cpu);
  st->n_streamed += (int)(a.streamed - b.streamed); st->n_promoted += (int)(a.promoted - b.promoted);
}

// HIVE_PROFILE=N [call-host <kind>] (hived_family.h CallHost)
void Runtime::call_host(int kind, int rows, double t0, double eng_ms, double logits_ms, double cands_ms) {
  static const int every = getenv("HIVE_PROFILE") ? std::max(0, atoi(getenv("HIVE_PROFILE"))) : 0;
  if (every <= 0) return;
  static const char* const names[3] = {"decode", "verify", "verify-batch"};
  CallHost& c = call_host_[kind];
  const double t1 = hive::mono_ms();
  ++c.n; c.rows += rows; c.eng += eng_ms; c.logits += logits_ms; c.cands += cands_ms; c.total += t1 - t0;
  if (c.n < every) return;
  const double n = (double)c.n;
  fprintf(stderr, "[call-host %s] calls %ld · rows %.2f · eng %.3f · logits %.3f · cands %.3f · rest %.3f ms/call\n", names[kind], c.n, c.rows / n, c.eng / n,
          c.logits / n, c.cands / n, (c.total - c.eng - c.logits - c.cands) / n);
  c = CallHost{};
}

// HIVE_TRACE_CACHE (the DeepSeek switch and line format — tools/hive_monitor.py): one [cache] line per decode / verify step, with the VRAM
//   headroom every 64 steps and the wall time of the step (hived's call, candidates included)
void Runtime::cache_line(int M, const GlmCacheStats& b, double t0) {
  static const bool on = getenv("HIVE_TRACE_CACHE") && *getenv("HIVE_TRACE_CACHE") && strcmp(getenv("HIVE_TRACE_CACHE"), "0") != 0;
  if (!on) return;
  GlmExperts& x = store_.experts();
  const GlmCacheStats& a = x.stats();
  size_t vfree = 0, vtot = 0;
  if ((cache_step_ & 63) == 0) cudaMemGetInfo(&vfree, &vtot);
  fprintf(stderr, "[cache] step %ld M=%d routed %d hit %d cpu %d streamed %d · resident %d pending %d / %d%s · wall %.2f ms\n", cache_step_, M,
          (int)(a.routed - b.routed), (int)(a.hit - b.hit), (int)(a.cpu - b.cpu), (int)(a.streamed - b.streamed), x.n_resident(), x.n_pending(), x.n_slots(),
          vtot ? (" · vram free " + std::to_string(vfree >> 20) + " MiB").c_str() : "", hive::mono_ms() - t0);
  ++cache_step_;
}

void Runtime::cands_dev(int M) {
  const int NC = opt_.sampler_cands, V = model_.qm().cfg().vocab;
  cudaStream_t st = eng_->stream();
  const float* lg = eng_->logits_dev();
  k::argmax_rows(lg, M, V, next_d_.as<int32_t>(), st);
  CUDA_CHECK(cudaMemcpyAsync(next_h_, next_d_.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
  if (NC > 0) {
    k::row_max_sumexp(lg, M, V, cand_it_d_, cand_max_.as<float>(), cand_sum_.as<float>(), st);
    k::topk_select_rows(lg, M, V, NC, V, cand_idx_.as<int32_t>(), NC, st);
    k::gather_vals(lg, cand_idx_.as<int32_t>(), M, NC, V, cand_val_.as<float>(), st);
    CUDA_CHECK(cudaMemcpyAsync(cand_idx_h_, cand_idx_.p, (size_t)M * NC * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(cand_val_h_, cand_val_.p, (size_t)M * NC * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(cand_max_h_, cand_max_.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(cand_sum_h_, cand_sum_.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
}

int32_t Runtime::forward(Seq& seq, const int32_t* ids, int M, const std::vector<ImageInput>* images, std::vector<float>* logits_out,
                         ForwardStats* stats, bool) {
  HIVE_CHECK(M >= 1, "qwen forward: no tokens");
  fill_cache();
  // images: encode each span (one temporal patch: an image, or a pair of video frames), then let the prefill take those rows' embeddings
  //   from the encoder output; the merged grid (n_vit_h/2 × n_vit_w/2) gives the span's rope coordinates
  std::vector<DevBuf> vis_out;
  const bool has_images = images && !images->empty();
  if (has_images) {
    ensure_vision();
    const size_t H = model_.qm().cfg().hidden;
    std::vector<QwenEngine::EmbedOverride> ov;
    vis_out.reserve(images->size());
    for (const ImageInput& im : *images) {
      const int n = im.n_vit_h * im.n_vit_w;
      HIVE_CHECK(im.n_vit_h % 2 == 0 && im.n_vit_w % 2 == 0 && (int)im.types.size() == n / 4 && im.start >= 0 && im.start + n / 4 <= M,
                 "qwen: image span does not match its patches");
      eng_->make_room(QwenVision::scratch_bytes(n) + (size_t)(n / 4) * H * 2);
      vis_out.emplace_back((size_t)(n / 4) * H * 2);
      vision_->encode(im.patches_dev, im.n_vit_h, im.n_vit_w, vis_out.back().as<bf16>(), eng_->stream());
      ov.push_back({im.start, n / 4, vis_out.back().as<bf16>(), im.n_vit_h / 2, im.n_vit_w / 2});
    }
    eng_->set_embed_override(std::move(ov));
  }
  struct ClearOv {
    QwenEngine& e; bool on;
    ~ClearOv() { if (on) e.set_embed_override({}); }
  } clear_ov{*eng_, has_images};
  if (M == 1) {
    std::vector<Seq*> seqs{&seq};
    std::vector<int32_t> next;
    std::vector<std::vector<float>> lg;
    forward_batch(seqs, ids, next, logits_out ? &lg : nullptr, stats);
    if (logits_out) *logits_out = std::move(lg[0]);
    return next[0];
  }
  const QwenConfig& c = model_.qm().cfg();
  const GlmCacheStats before = store_.experts().stats();
  const double t0 = hive::mono_ms();
  // layer yields: only an outer prefill (not one run inside a yield) of at least prefill_threshold rows — or, with HIVE_LAYER_YIELD_SMALL,
  //   of more rows than the decode-path size (64: those run in tens of milliseconds)
  const bool ly_ok = ly_.on() && ly_.depth < ly_.h.max_depth && (M >= opt_.prefill_threshold || (ly_small_ && M > 64));
  // read progress (layer_yield.h Progress — the API server's stage status): the layer hook is also installed when only it is on;
  //   the yield point still runs only for ly_ok
  const bool pg_ok = prog_.on() && ly_.depth == 0 && M >= opt_.prefill_threshold;
  const bool hook_on = ly_ok || pg_ok;
  struct LyHook {  // restores the hook of the forward this one runs inside (a level-2 yield)
    QwenEngine& e; bool on; std::function<void(int)> prev; bool prev_yields;
    ~LyHook() { if (on) { e.layer_boundary = std::move(prev); e.layer_boundary_yields = prev_yields; } }
  } ly_hook{*eng_, hook_on, hook_on ? eng_->layer_boundary : std::function<void(int)>{}, eng_->layer_boundary_yields};
  if (pg_ok) pg_marks_.begin();
  if (hook_on) {
    if (ly_ok && ly_.depth == 0) ly_.last = hive::mono_ms();
    eng_->layer_boundary_yields = ly_ok;
    eng_->layer_boundary = [this, ly_ok, pg_ok](int l) { ly_progress_ = eng_->boundary_base + eng_->boundary_span * (l + 1.0) / std::max(1, model_.qm().cfg().n_layers); if (pg_ok && ly_.depth == 0) { pg_marks_.mark(eng_->stream(), ly_progress_); progress_point_gpu(); } if (ly_ok) layer_yield_point(); };
  }
  try {
    prefill_logits_.resize(c.vocab);
    eng_->prefill(seq, ids, M, prefill_logits_.data());
  } catch (...) { seq.broken = true; throw; }
  if (has_images) {  // the encoder outputs are no longer needed: free them and let the expert cache take that memory back
    eng_->set_embed_override({});
    vis_out.clear();
    eng_->give_back_room();
  }
  since_prefill_ = 0;
  cand_it_h_[0] = 1.f;
  cands_dev(1);
  if (logits_out) *logits_out = prefill_logits_;
  seq.mtp_hidden_valid = seq.h_last_valid && eng_->mtp_on();
  stats_delta(stats, before);
  if (stats) stats->ms_total += hive::mono_ms() - t0;
  return next_h_[0];
}

void Runtime::forward_batch(std::vector<Seq*>& seqs, const int32_t* ids, std::vector<int32_t>& next, std::vector<std::vector<float>>* logits_out,
                            ForwardStats* stats) {
  const int M = (int)seqs.size();
  HIVE_CHECK(M >= 1 && M <= rows_cap_, "qwen decode batch size");
  fill_cache();
  const int V = model_.qm().cfg().vocab;
  const GlmCacheStats before = store_.experts().stats();
  const double t0 = hive::mono_ms();
  std::vector<QwenSeq*> gs(M);
  for (int m = 0; m < M; ++m) gs[m] = seqs[m];
  store_.experts().reset_row_stats();
  const QwenForwardStats e0 = eng_->stats();
  try {
    eng_->decode(gs.data(), ids, M, logits_out ? logits_pin_ : nullptr);   // no full-logit copy unless asked (16 rows: 16 MB per step)
  } catch (...) { for (Seq* s : seqs) s->broken = true; throw; }
  const double t_eng = hive::mono_ms();
  if (prof_on()) prof_add("decode M=" + std::to_string(M), t_eng - t0, e0, eng_->stats(), before, store_.experts().stats());
  if (since_prefill_ < (1 << 30)) ++since_prefill_;
  const double t_c0 = hive::mono_ms();
  cands_dev(M);
  const double t_c1 = hive::mono_ms();
  for (Seq* q : seqs) q->mtp_hidden_valid = q->h_last_valid && eng_->mtp_on();
  next.assign(next_h_, next_h_ + M);
  if (logits_out) {
    logits_out->assign(M, std::vector<float>());
    for (int m = 0; m < M; ++m) (*logits_out)[m].assign(logits_pin_ + (size_t)m * V, logits_pin_ + (size_t)(m + 1) * V);
  }
  const double t_lg = hive::mono_ms();
  stats_delta(stats, before);
  if (stats) {
    stats->row_hit.assign(M, 0); stats->row_cpu.assign(M, 0); stats->row_dma.assign(M, 0);
    for (int m = 0; m < M; ++m) { stats->row_hit[m] = store_.experts().row_hit(m); stats->row_cpu[m] = store_.experts().row_cpu(m); }
    stats->ms_total += hive::mono_ms() - t0;
  }  cache_line(M, before, t0);
  call_host(0, M, t0, t_eng - t0, t_lg - t_c1, t_c1 - t_c0);
}

void Runtime::mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats* stats) {
  const GlmCacheStats before = store_.experts().stats();
  const int k = eng_->mtp_k();
  drafts.assign(k, 0); conf.assign(k, 0.f);
  const QwenForwardStats e0 = eng_->stats();
  const double t0 = hive::mono_ms();
  const int n = eng_->mtp_draft(seq, tok, k, drafts.data(), conf.data());
  if (prof_on()) prof_add("draft k=" + std::to_string(k), hive::mono_ms() - t0, e0, eng_->stats(), before, store_.experts().stats());
  drafts.resize(n); conf.resize(n);
  // hived's MTP gate buckets a logit-scale confidence (DeepSeek: its confidence head); this family gives the draft's softmax probability p —
  //   pass log(p / (1 − p)), monotone in p, so the gate's online acceptance table per bucket applies (HIVE_MTP_GATE2 learns it)
  for (float& c : conf) { const float p = std::min(std::max(c, 1e-6f), 1.f - 1e-6f); c = std::log(p / (1.f - p)); }
  stats_delta(stats, before);
}

void Runtime::mtp_draft_batch(const std::vector<Seq*>& seqs, const std::vector<int32_t>& toks, std::vector<std::vector<int32_t>>& drafts,
                              std::vector<std::vector<float>>& conf, ForwardStats* stats) {
  const GlmCacheStats before = store_.experts().stats();
  const int S = (int)seqs.size(), k = eng_->mtp_k();
  drafts.assign(S, {}); conf.assign(S, {});
  std::vector<QwenSeq*> qs(S);
  std::vector<int32_t> tk(S);
  for (int s = 0; s < S; ++s) { qs[s] = seqs[s]; tk[s] = toks[s]; }
  std::vector<int32_t> d((size_t)S * k);
  std::vector<float> c((size_t)S * k);
  std::vector<int> n(S);
  eng_->mtp_draft_batch(qs.data(), tk.data(), S, k, d.data(), c.data(), n.data());
  for (int s = 0; s < S; ++s) {
    drafts[s].assign(d.begin() + (size_t)s * k, d.begin() + (size_t)s * k + n[s]);
    conf[s].assign(c.begin() + (size_t)s * k, c.begin() + (size_t)s * k + n[s]);
    for (float& x : conf[s]) { const float p = std::min(std::max(x, 1e-6f), 1.f - 1e-6f); x = std::log(p / (1.f - p)); }   // as mtp_draft
  }
  stats_delta(stats, before);
}

void Runtime::forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats) {
  HIVE_CHECK(M >= 1 && M <= rows_cap_, "qwen verify rows");
  fill_cache();
  const int V = model_.qm().cfg().vocab;
  const GlmCacheStats before = store_.experts().stats();
  const double t0 = hive::mono_ms();
  const QwenForwardStats e0 = eng_->stats();
  try {
    eng_->verify(seq, ids, M, verify_logits_ ? logits_pin_ : nullptr);
  } catch (...) { seq.broken = true; throw; }
  const double t_eng = hive::mono_ms();
  if (verify_logits_) logits_rows.assign(logits_pin_, logits_pin_ + (size_t)M * V);
  else logits_rows.clear();   // greedy rows only (set_verify_logits(false)): the candidates carry the argmax
  const double t_lg = hive::mono_ms();
  if (prof_on()) prof_add(std::string(since_prefill_ < kWarmSteps ? "verify(warm) M=" : "verify M=") + std::to_string(M), hive::mono_ms() - t0, e0, eng_->stats(), before,
                          store_.experts().stats());
  if (since_prefill_ < kWarmSteps) ++warm_samples_;  // cache still warming: not a representative step cost (see graph_captures)
  if (since_prefill_ < (1 << 30)) ++since_prefill_;
  const double t_c0 = hive::mono_ms();
  cands_dev(M);  // candidates of every verify row (row_inv_temp set by the caller)
  const double t_c1 = hive::mono_ms();
  stats_delta(stats, before);
  if (stats) stats->ms_total += hive::mono_ms() - t0;  cache_line(M, before, t0);
  call_host(1, M, t0, t_eng - t0, t_lg - t_eng, t_c1 - t_c0);
}

void Runtime::rollback(Seq& seq, int n_keep) {
  eng_->rollback(seq, n_keep);
  seq.mtp_hidden_valid = seq.h_last_valid && eng_->mtp_on();
}

void Runtime::forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats) {
  const int S = (int)parts.size();
  HIVE_CHECK(mtp_batch_ && S >= 1 && S <= eng_->verify_parts_cap(), "qwen batched verify: parts");
  const int V = model_.qm().cfg().vocab;
  std::vector<QwenSeq*> seqs(S);
  std::vector<int> Ms(S);
  std::vector<int32_t> ids;
  for (int p = 0; p < S; ++p) { seqs[p] = parts[p].seq; Ms[p] = parts[p].M; ids.insert(ids.end(), parts[p].ids, parts[p].ids + parts[p].M); }
  const int R = (int)ids.size();
  HIVE_CHECK(R >= 1 && R <= rows_cap_, "qwen batched verify rows");
  fill_cache();
  const GlmCacheStats before = store_.experts().stats();
  const QwenForwardStats e0 = eng_->stats();
  const double t0 = hive::mono_ms();
  try {
    eng_->verify_batch(seqs.data(), Ms.data(), S, ids.data(), verify_logits_ ? logits_pin_ : nullptr);
  } catch (...) { for (auto& p : parts) p.seq->broken = true; throw; }
  const double t_eng = hive::mono_ms();
  if (verify_logits_) logits_rows.assign(logits_pin_, logits_pin_ + (size_t)R * V);
  else logits_rows.clear();
  const double t_lg = hive::mono_ms();
  if (prof_on()) prof_add("verify-batch S=" + std::to_string(S) + " R=" + std::to_string(R), hive::mono_ms() - t0, e0, eng_->stats(), before,
                          store_.experts().stats());
  if (since_prefill_ < kWarmSteps) ++warm_samples_;
  if (since_prefill_ < (1 << 30)) ++since_prefill_;
  const double t_c0 = hive::mono_ms();
  cands_dev(R);
  const double t_c1 = hive::mono_ms();
  stats_delta(stats, before);
  if (stats) stats->ms_total += hive::mono_ms() - t0;
  vb_seqs_ = seqs;  cache_line(R, before, t0);
  call_host(2, R, t0, t_eng - t0, t_lg - t_eng, t_c1 - t_c0);
}

void Runtime::rollback_batch(const std::vector<int>& n_keep) {
  HIVE_CHECK(n_keep.size() == vb_seqs_.size(), "qwen batched rollback: parts");
  eng_->rollback_batch(n_keep.data(), (int)n_keep.size());
  for (QwenSeq* s : vb_seqs_) static_cast<Seq*>(s)->mtp_hidden_valid = s->h_last_valid && eng_->mtp_on();
  vb_seqs_.clear();
}

// A layer boundary of an outer prefill: yield_point decides (period, hived's want) and runs hived's yield body between yield_enter and yield_exit.
//   floor = the most rows a decode / verify step uses; cap = the rows the inner forwards may use at this boundary. The stream is drained first so
//   the time hived measures as prefill ends here.
void Runtime::layer_yield_point() {
  if (!ly_.on() || ly_.depth >= ly_.h.max_depth) return;
  ly_.floor = rows_cap_;
  ly_.cap = eng_->yield_rows_cap();
  QwenEngine::YieldSave sv;
  double t_body = 0;  // yield time (QwenEngine::pf_yield_body_ms): entry after the stream is drained → exit
  ly::yield_point(
      ly_,
      [&](int) {
        CUDA_CHECK(cudaStreamSynchronize(eng_->stream()));
        const bool ok = eng_->yield_enter(sv);
        if (ok) t_body = hive::mono_ms();
        return ok;
      },
      [&](int) {
        if (t_body > 0) { eng_->pf_yield_body_ms += hive::mono_ms() - t_body; t_body = 0; }
        cudaStreamSynchronize(eng_->stream());  // (no throw — this also runs while an exception from the yield body unwinds)
        eng_->yield_exit(sv);
      },
      [] { return hive::mono_ms(); });
}

void Runtime::forward_multi(std::vector<PrefillPart>& parts, ForwardStats* stats) {
  for (auto& p : parts) p.next = forward(*p.seq, p.ids, p.M, nullptr, p.logits_out, stats, p.upper_needed);
}

}  // namespace hive::qwen::fam
