// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hived surface of the Gemma family — see hive/gemma/hived_family.h (derived from src/q27/q27_runtime.cpp, with GemmaEngine).
#include "hive/gemma/hived_family.h"
#include "hive/vram_reserve.h"
#include "hive/image_reaper.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <sys/mman.h>

#include "hive/clock.h"
#include "hive/model_kernels.h"

namespace hive::gemma::fam {

Model::Model(const std::string& ckpt, int, int64_t max_ctx) {
  // the MTP assistant (Frozen-KV drafter) is a separate checkpoint: HIVE_GEMMA_ASSISTANT (unset or "" = no drafter)
  const char* a = getenv("HIVE_GEMMA_ASSISTANT");
  const int mb = getenv("HIVE_MAX_BATCH") ? std::max(1, atoi(getenv("HIVE_MAX_BATCH"))) : 4;
  e_ = std::make_unique<GemmaEngine>(ckpt, a ? a : "", max_ctx, mb);
  const GemmaConfig& q = e_->cfg();
  cfg_.vocab = q.vocab; cfg_.n_layers = q.n_layers; cfg_.eng = e_.get();
  cfg_.window = q.window;
  cfg_.vision_max_tokens = e_->vision_max_tokens(); cfg_.vision_downsample = 1;   // hived's patch budget per image (one patch = one soft token)
}

Runtime::Runtime(Model& model, ExpertStore& store, const void*, const RuntimeOptions& opt) : model_(model), store_(store), opt_(opt) {
  opt_.max_batch = std::max(1, std::min(opt_.max_batch, 16));
  // sampler candidates per row (HIVE_GEMMA_CANDS, default ≤ 256): top-k over 248K logits per row is part of every step; the sampler takes the
  //   full-logit path whenever the candidates do not hold the needed mass or top_k exceeds them, so outputs follow the same distribution
  opt_.sampler_cands = getenv("HIVE_GEMMA_CANDS") ? std::max(1, atoi(getenv("HIVE_GEMMA_CANDS"))) : std::min(opt_.sampler_cands, 256);
  const char* mb = getenv("HIVE_MTP_BATCH");
  mtp_batch_ = mb && *mb && strcmp(mb, "0") != 0 && model_.eng().mtp_on() && opt_.max_batch >= 2;
  const int rows = 32, NC = std::max(1, opt_.sampler_cands), V = model_.eng().cfg().vocab;
  logits_.resize((size_t)rows * V); inv_temp_.assign(rows, 1.f); cand_val_.resize((size_t)rows * NC); cand_idx_.resize((size_t)rows * NC);
  rmax_.resize(rows); rsum_.resize(rows); next_.resize(rows);
  c_it_.alloc((size_t)rows * 4); c_idx_.alloc((size_t)rows * NC * 4); c_val_.alloc((size_t)rows * NC * 4); c_max_.alloc((size_t)rows * 4);
  c_sum_.alloc((size_t)rows * 4); c_next_.alloc((size_t)rows * 4);
  if (!(getenv("HIVE_GEMMA_WARMUP") && strcmp(getenv("HIVE_GEMMA_WARMUP"), "0") == 0)) model_.eng().warmup(opt_.max_batch, model_.eng().mtp_k());
  staged_flush();   // allocates the pinned image staging buffers now (not on the first eviction)
  const auto m = model_.eng().mem();
  fprintf(stderr, "[gemma] runtime: family %s · MTP %s (k %d) · max batch %d · chunk %d · context %lld · weights %.0f MiB · KV pool %.0f MiB · work %.0f MiB\n",
          kFamily, model_.eng().mtp_on() ? "on" : "off", model_.eng().mtp_k(), opt_.max_batch, std::max(opt_.max_chunk, 64), (long long)opt_.max_ctx,
          m.weights / 1048576.0, m.kv_pool / 1048576.0, m.work / 1048576.0);
}

std::unique_ptr<Seq> Runtime::new_seq() const {
  auto s = std::make_unique<Seq>();
  model_.eng().init_seq(*s, opt_.max_ctx);
  return s;
}

bool Runtime::rewind(Seq& s, size_t n) const {
  if (!model_.eng().rewind(s, (int64_t)n)) return false;
  s.generation = Seq::next_uid();
  s.mtp_hidden_valid = false;
  return true;
}

void Runtime::reset_seq(Seq& s) const {
  model_.eng().reset_seq(s);
  s.generation = Seq::next_uid();
  s.engram_history.clear();
  s.mtp_hidden_valid = false;
}

// images straight from the device into the host buffers (no intermediate vectors); KV pieces share the base image's prefix (same lineage,
//   only grown since) through HostImageBuffer::extend
// Device → pageable host copies of session images through two pinned 64 MiB staging buffers: the D2H of one block overlaps the parallel
//   host copy of the previous one. Measured 2026-10-10 (perf-core 600 W): direct pageable D2H ran at ~2.6 GB/s — the deferred prompt
//   checkpoint of a 32K prompt (2.1 GB KV + 151 MB GDN state) stalled decode ~0.9 s (prefill32k TPOT 7.6 → 13.9 ms), c1 lost ~90 ms per
//   request (TPOT 5.39 → 5.87), and pool evictions put ~116 ms into every c1 queue_ms.
namespace {

// Host memory for session images, faulted in ahead of time. A save writes ~150 MB (GDN states) into fresh pages; the page faults of those
//   writes (16 copy threads contending in the kernel) made every pool eviction ~30 ms (measured 2026-10-10, bench c1 1K: TTFT 131-136 ms
//   for the first 16 sessions, 163-168 ms from the 17th on — each admission evicts and saves the oldest idle session). A background thread
//   keeps two blocks mapped and populated (transparent huge pages where available); a save whose new bytes fit one block carves its
//   pieces from it (aliasing shared_ptrs: the block is unmapped with its last piece), larger saves allocate as before.
// image memory is released through hive::image_reaper (hive/image_reaper.h — shared by all families since 2026-10-11)
struct ImageArena {
  // kStep: block sizes follow the largest image in 32 MiB steps — 256 MiB steps put the ~300 MiB images of short voice conversations
  //   into 512 MiB blocks (2026-10-19: 90 of 115 saves 289-302 MiB, 41 % of each block unused)
  static constexpr size_t kMin = (size_t)256 << 20, kMax = (size_t)1 << 30, kStep = (size_t)32 << 20, kAlign = 4096;
  struct Block { uint8_t* p; size_t n; };
  std::mutex mu;
  std::condition_variable cv;
  std::vector<Block> ready;
  size_t want = kMin;   // block size: follows the latest image (kMin … kMax)
  std::thread th;
  static Block make(size_t n) {
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return {nullptr, 0};
    madvise(p, n, MADV_HUGEPAGE);
    for (size_t o = 0; o < n; o += 4096) static_cast<volatile uint8_t*>(p)[o] = 0;   // fault in now, off the request path
    return {static_cast<uint8_t*>(p), n};
  }
  ImageArena() {
    th = std::thread([this] {
      for (;;) {
        size_t n;
        {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&] { return ready.size() < 2; });
          n = want;
        }
        Block b = make(n);
        if (!b.p) { std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
        std::lock_guard<std::mutex> lk(mu);
        ready.push_back(b);
      }
    });
    th.detach();
  }
  // a populated block of at least n and at most 2n bytes (nullptr: none ready / too large / mostly unused).
  //   The pieces of one save are carved from the block's front and nothing else is ever carved from the rest, while the block stays mapped
  //   until its last piece goes — and the session budget (allocated_bytes) counts only the pieces. Taking a block for any save of 8 MiB
  //   or more made each follow-up turn (a few tens of MB of new KV) hold a whole 256 MiB-1 GiB block: 2026-10-19 hived_gemma (voice
  //   conversations beside a DeepSeek daemon, a few hours) had VmRSS 59.9 GB against 16.96 GB of counted session images (160 archived),
  //   every large anonymous mapping a multiple of 0.5 GiB. A save the block would not fit within 2x allocates its own bytes; its host copy
  //   runs on the checkpoint worker (async path).
  std::shared_ptr<uint8_t> take(size_t n) {
    std::lock_guard<std::mutex> lk(mu);
    // the latest size, not the largest seen: one long prompt (a ~1 GiB image) used to make every later block 1 GiB for good, so the
    //   ~300 MiB saves of the conversations after it each held 1 GiB. Ready blocks more than twice the new size go back (the thread refills).
    want = std::min(kMax, std::max(kMin, (n + kStep - 1) / kStep * kStep));
    for (auto it = ready.begin(); it != ready.end();)
      if (it->n / 2 > want) { hive::image_reaper().put(it->p, it->n, hive::ImageReaper::kMunmap); it = ready.erase(it); }
      else ++it;
    static const bool tmi = getenv("HIVE_GEMMA_TIMING") && *getenv("HIVE_GEMMA_TIMING") && strcmp(getenv("HIVE_GEMMA_TIMING"), "0") != 0;
    size_t first_fit = 0;   // the block the former rule (any ready block >= n) would have taken — for the timing log
    for (size_t i = 0; i < ready.size(); ++i) if (ready[i].n >= n) { first_fit = ready[i].n; break; }
    for (size_t i = 0; i < ready.size(); ++i)
      if (ready[i].n >= n && ready[i].n / 2 <= n) {
        Block b = ready[i];
        ready.erase(ready.begin() + i);
        cv.notify_one();
        if (tmi) fprintf(stderr, "[gemma-timing] image arena: %zu MiB from a %zu MiB block\n", n >> 20, b.n >> 20);
        return std::shared_ptr<uint8_t>(b.p, [n = b.n](uint8_t* q) { hive::image_reaper().put(q, n, hive::ImageReaper::kMunmap); });
      }
    if (tmi) fprintf(stderr, "[gemma-timing] image arena: %zu MiB allocated (%s)\n", n >> 20,
                     first_fit ? ("former rule: a " + std::to_string(first_fit >> 20) + " MiB block").c_str() : "no ready block");
    cv.notify_one();
    return nullptr;
  }
};
ImageArena& image_arena() { static ImageArena a; return a; }
struct Stage {
  uint8_t* buf[2] = {nullptr, nullptr};
  cudaEvent_t ev[2] = {nullptr, nullptr};
  uint8_t* dst[2] = {nullptr, nullptr};
  size_t n[2] = {0, 0};
  int cur = 0;
  static constexpr size_t kBytes = 64ull << 20;
};
Stage& stage() {
  static Stage s;
  if (!s.buf[0]) for (int i = 0; i < 2; ++i) { CUDA_CHECK(cudaMallocHost(&s.buf[i], Stage::kBytes)); CUDA_CHECK(cudaEventCreateWithFlags(&s.ev[i], cudaEventDisableTiming)); }
  return s;
}
void stage_drain(Stage& S, int b) {
  if (!S.n[b]) return;
  CUDA_CHECK(cudaEventSynchronize(S.ev[b]));
  const size_t n = S.n[b], per = (n + 15) / 16;
  uint8_t* d = S.dst[b];
  const uint8_t* src = S.buf[b];
  (void)per;
  memcpy(d, src, n);
  S.n[b] = 0;
}
}  // namespace
void Runtime::staged_d2h(uint8_t* dst, const uint8_t* dev, size_t n) const {
  Stage& S = stage();
  cudaStream_t st = model_.eng().stream();
  for (size_t off = 0; off < n; off += Stage::kBytes) {
    const size_t m = std::min(Stage::kBytes, n - off);
    const int b = S.cur;
    stage_drain(S, b);   // the buffer's previous block goes to its destination first
    CUDA_CHECK(cudaMemcpyAsync(S.buf[b], dev + off, m, cudaMemcpyDefault, st));   // (Default: KV pages may be mapped host memory)
    CUDA_CHECK(cudaEventRecord(S.ev[b], st));
    S.dst[b] = dst + off; S.n[b] = m;
    S.cur ^= 1;
  }
}
void Runtime::staged_h2d(void* dev, const uint8_t* src, size_t n) const {
  Stage& S = stage();
  cudaStream_t st = model_.eng().stream();
  for (size_t off = 0; off < n; off += Stage::kBytes) {
    const size_t m = std::min(Stage::kBytes, n - off);
    const int b = S.cur;
    stage_drain(S, b);
    CUDA_CHECK(cudaEventSynchronize(S.ev[b]));   // the buffer's previous upload has left it
    uint8_t* stg = S.buf[b];
    const size_t per = (m + 15) / 16;
    (void)per;
    memcpy(stg, src + off, m);
    CUDA_CHECK(cudaMemcpyAsync((uint8_t*)dev + off, stg, m, cudaMemcpyDefault, st));   // (Default: the destination may be a host KV page)
    CUDA_CHECK(cudaEventRecord(S.ev[b], st));
    S.cur ^= 1;
  }
}
void Runtime::staged_flush() const {
  Stage& S = stage();
  stage_drain(S, S.cur);
  stage_drain(S, S.cur ^ 1);
}

Runtime::~Runtime() {
  { std::lock_guard<std::mutex> lk(ck_mu_); ck_stop_ = true; }
  ck_cv_.notify_all();
  if (ck_thr_.joinable()) ck_thr_.join();
}

int Runtime::ck_slot(size_t bytes) const {
  std::unique_lock<std::mutex> lk(ck_mu_);
  if (!ck_thr_.joinable()) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&ck_st_, cudaStreamNonBlocking));
    ck_thr_ = std::thread([this] { ck_worker(); });
  }
  ck_cv_.wait(lk, [&] { return !ck_busy_[0] || !ck_busy_[1]; });
  const int sl = ck_busy_[0] ? 1 : 0;
  ck_busy_[sl] = true;
  lk.unlock();
  if (ck_scratch_[sl].n < bytes) {
    // Grow with headroom (+1/8, whole 256 MiB) and never lose the slot: measured 2026-10-11 (hived, the bench's 20 prefill64k prompts in
    //   order): prompts of 63-65.5K tokens grew the slot a little at a time; once the pooled 64K sessions had filled VRAM (≥ 78 GB) the
    //   regrowth (free, then cudaMalloc) failed, the slot stayed empty and every later prompt checkpoint (~4.3 GB) went synchronous on the
    //   request path — decode of 128 tokens 414-817 ms → 846-1484 ms (+300-450 ms per request). Now: the new buffer first, the old one
    //   freed after; on failure free and retry, else take the old size back.
    //   ★2026-10-18 (Gemma beside DeepSeek under HIVE_VRAM_LIMIT_MB): the slots are DevBuf allocations, so they come out of the limit's
    //   reserve like every runtime buffer and are refused past it (→ this image is copied synchronously). They were raw cudaMalloc
    //   pointers stored in a DevBuf: the growth bypassed the limit (the process grew 12,982 → 13,998 MiB under a 13,000 MiB limit in
    //   256 MiB steps during the live harnesses) and every regrowth's free credited the reserve with memory it never gave.
    const size_t want = ((bytes + bytes / 8) + ((size_t)256 << 20) - 1) & ~(((size_t)256 << 20) - 1), old = ck_scratch_[sl].n;
    //   The slots may not take the reserve's last kFloor bytes: they belong to the small buffers a new conversation needs (page
    //   tables, seeds) — with the reserve at 0 those would be refused and the request fail (measured: a 512 MiB slot left 0 MiB).
    constexpr size_t kFloor = (size_t)128 << 20;
    auto try_alloc = [](DevBuf& b, size_t n) {
      if (vram_reserve::cap_on() && vram_reserve::held() < n + kFloor) return false;
      try { b.alloc(n); } catch (const DeviceOutOfMemory&) {}
      return b.p != nullptr;
    };
    DevBuf nb;
    bool ok = try_alloc(nb, want);
    if (!ok) { ck_scratch_[sl] = DevBuf(); ok = try_alloc(nb, want); }
    if (ok) {
      ck_scratch_[sl] = std::move(nb);
    } else {
      if (old && !ck_scratch_[sl].p) try_alloc(ck_scratch_[sl], old);
      static int warned = 0;
      if (warned++ < 8) fprintf(stderr, "[gemma] checkpoint snapshot slot %d: no VRAM for %zu MiB — this image is copied synchronously\n", sl, want >> 20);
      { std::lock_guard<std::mutex> lk2(ck_mu_); ck_busy_[sl] = false; }   // no room for the snapshot: the caller copies synchronously
      ck_cv_.notify_all();
      return -1;
    }
  }
  return sl;
}

void Runtime::ck_worker() const {
  constexpr size_t kChunk = (size_t)16 << 20;
  CUDA_CHECK(cudaSetDevice(0));
  uint8_t* stg[2];
  cudaEvent_t ev[2];
  for (int i = 0; i < 2; ++i) { CUDA_CHECK(cudaMallocHost(&stg[i], kChunk)); CUDA_CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming)); }
  for (;;) {
    std::shared_ptr<CkptJob> job;
    {
      std::unique_lock<std::mutex> lk(ck_mu_);
      ck_cv_.wait(lk, [&] { return ck_stop_ || !ck_q_.empty(); });
      if (ck_q_.empty()) break;
      job = ck_q_.front(); ck_q_.pop_front();
    }
    CUDA_CHECK(cudaStreamWaitEvent(ck_st_, job->ready, 0));
    size_t c = 0;
    std::pair<uint8_t*, size_t> pend[2] = {};
    auto drain = [&](int b) { if (pend[b].first) { CUDA_CHECK(cudaEventSynchronize(ev[b])); memcpy(pend[b].first, stg[b], pend[b].second); pend[b] = {}; } };
    for (auto& [dst, src, n] : job->parts)
      for (size_t off = 0; off < n; off += kChunk, ++c) {
        const int b = (int)(c & 1);
        drain(b);
        const size_t len = std::min(kChunk, n - off);
        CUDA_CHECK(cudaMemcpyAsync(stg[b], src + off, len, cudaMemcpyDeviceToHost, ck_st_));
        CUDA_CHECK(cudaEventRecord(ev[b], ck_st_));
        pend[b] = {dst + off, len};
        drain(b ^ 1);
      }
    drain(0); drain(1);
    CUDA_CHECK(cudaEventDestroy(job->ready));
    { std::lock_guard<std::mutex> lk(ck_mu_); ck_busy_[job->slot] = false; }
    ck_cv_.notify_all();
    job->fence->done();
  }
  for (int i = 0; i < 2; ++i) { cudaFreeHost(stg[i]); cudaEventDestroy(ev[i]); }
}

// Pool evictions (and deferred prompt checkpoints) save on the request path: the copy to host ran synchronously, ~17 ms per arrival after the
//   pre-faulted arena (bench c1 1K, 2026-10-10). Default (HIVE_GEMMA_CKPT_ASYNC): when the image's new device bytes fit a scratch slot
//   (HIVE_GEMMA_CKPT_SNAP_MB, default 512) they are snapshotted device → device on the engine stream (ordered after every writer; ~0.2 ms for a
//   1K session) and a worker thread copies the snapshot to the host bytes (default 8192 MB since 2026-10-11: a 32K prompt checkpoint (2.3 GB)
//   stalled decode ~150 ms per request with 512 — hived, 5 × 32K: decode 900 / 863 → 745 / 727 ms for 128 tokens); the job never reads the sequence, so the sequence can be reset,
//   trimmed or reloaded at once. The image's fence makes load_image (and a later save sharing its segments) wait for the copy.
void Runtime::save_image(const Seq& s, SeqImage& img, const SeqImage* base) const {
  static const bool tmi = getenv("HIVE_GEMMA_TIMING") && *getenv("HIVE_GEMMA_TIMING") && strcmp(getenv("HIVE_GEMMA_TIMING"), "0") != 0;
  const double ti0 = hive::mono_ms();
  double ti_plan = 0, ti_fence = 0, ti_arena = 0, ti_slot = 0, ti_copy = 0;
  SeqImage out;
  out.pos = s.pos; out.generation = s.generation; out.tokens = s.tokens; out.mtp_hidden_valid = s.mtp_hidden_valid;
  std::vector<GemmaSeq::Piece> pc;
  std::vector<std::vector<uint8_t>> hold;
  s.save_plan(pc, hold);
  ti_plan = hive::mono_ms();
  const bool lineage = base && base->generation == s.generation && base->pos <= s.pos && base->bufs.size() == pc.size() &&
                       std::equal(base->tokens.begin(), base->tokens.end(), s.tokens.begin());
  if (lineage && base->fence) base->fence->wait();   // its segments are shared below: their copies must be complete
  ti_fence = hive::mono_ms();
  cudaStream_t st = model_.eng().stream();
  out.bufs.resize(pc.size());
  // new bytes of this image (the same prefix rule as HostImageBuffer::extend), carved from one pre-faulted arena block when they fit
  size_t need = 0;
  for (size_t i = 0; i < pc.size(); ++i) {
    const size_t pre = lineage && pc[i].append && base->bufs[i].size() <= pc[i].n ? base->bufs[i].size() : 0;
    if (pc[i].n > pre) need += (pc[i].n - pre + 63) & ~(size_t)63;
  }
  std::shared_ptr<uint8_t> blk = need >= ((size_t)8 << 20) ? image_arena().take(need) : nullptr;
  ti_arena = hive::mono_ms();
  size_t used = 0;
  auto alloc = [&](size_t n) {
    const size_t a = (n + 63) & ~(size_t)63;
    if (blk && used + a <= ImageArena::kMax && used + a <= need) { std::shared_ptr<uint8_t> r(blk, blk.get() + used); used += a; return r; }
    return hive::reaped_bytes(n);   // large pieces: freed off the engine thread
  };
  static const bool async = !(getenv("HIVE_GEMMA_CKPT_ASYNC") && strcmp(getenv("HIVE_GEMMA_CKPT_ASYNC"), "0") == 0);
  // default 8192 MB; under a VRAM cap (HIVE_VRAM_LIMIT_MB) 512 MB — the slots would take the VRAM kept for
  //   the KV of a full-context prompt (GemmaEngine KV plan, 2026-10-12)
  static const bool vcap = (getenv("HIVE_VRAM_LIMIT_MB") && atof(getenv("HIVE_VRAM_LIMIT_MB")) > 0) || false;
  static const size_t snap_budget = (size_t)(getenv("HIVE_GEMMA_CKPT_SNAP_MB") ? atof(getenv("HIVE_GEMMA_CKPT_SNAP_MB")) : vcap ? 512.0 : 8192.0) * 1048576;
  size_t dev_new = 0;
  for (size_t i = 0; i < pc.size(); ++i) {
    const size_t pre = lineage && pc[i].append && base->bufs[i].size() <= pc[i].n ? base->bufs[i].size() : 0;
    if (pc[i].dev && pc[i].n > pre) dev_new += (pc[i].n - pre + 255) & ~(size_t)255;
  }
  std::shared_ptr<CkptJob> job;
  if (async && dev_new > 0 && dev_new <= snap_budget) {
    const int sl = ck_slot(dev_new);
    if (sl >= 0) { job = std::make_shared<CkptJob>(); job->slot = sl; }
  }
  ti_slot = hive::mono_ms();
  size_t snap_off = 0;
  // the snapshot's device → device copies go as one copy_chunks launch (2026-10-11 hived nsys, 1K request: ~150 piece copies, one
  //   cudaMemcpyAsync each, left ~10 ms of GPU idle in the admission / prompt-checkpoint path)
  for (size_t i = 0; i < pc.size(); ++i) {
    const GemmaSeq::Piece& p = pc[i];
    HostImageBuffer& hb = out.bufs[i];
    if (!p.n) continue;
    const size_t pre = hb.extend(lineage && p.append ? &base->bufs[i] : nullptr, p.n, alloc);
    if (pre >= p.n) continue;
    uint8_t* dst = hb.segments.back().data.get();
    if (p.dev && job) {   // device snapshot now (engine stream), host copy on the worker
      uint8_t* d = ck_scratch_[job->slot].as<uint8_t>() + snap_off;
      CUDA_CHECK(cudaMemcpyAsync(d, (const uint8_t*)p.dev + pre, p.n - pre, cudaMemcpyDefault, st));
      job->parts.emplace_back(dst, d, p.n - pre);
      job->keep.push_back(hb.segments.back().data);
      snap_off += (p.n - pre + 255) & ~(size_t)255;
    } else if (p.dev) staged_d2h(dst, (const uint8_t*)p.dev + pre, p.n - pre);
    else memcpy(dst, (const uint8_t*)p.host + pre, p.n - pre);
  }
  ti_copy = hive::mono_ms();
  if (job) {
    CUDA_CHECK(cudaEventCreateWithFlags(&job->ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(job->ready, st));   // after the snapshots and every earlier write of the pieces
    job->fence = std::make_shared<CkptFence>();
    job->fence->add();
    out.fence = job->fence;
    { std::lock_guard<std::mutex> lk(ck_mu_); ck_q_.push_back(job); }
    ck_cv_.notify_all();
  } else {
    staged_flush();
  }
  img = std::move(out);
  if (tmi && hive::mono_ms() - ti0 > 20)   // a slow save on the request path, by phase
    fprintf(stderr, "[gemma-timing] save_image %.0f ms: plan %.0f · base fence %.0f · arena %.0f · slot %.0f · pieces %.0f · tail %.0f · need %zu MiB dev %zu MiB async %d\n",
            hive::mono_ms() - ti0, ti_plan - ti0, ti_fence - ti_plan, ti_arena - ti_fence, ti_slot - ti_arena, ti_copy - ti_slot, hive::mono_ms() - ti_copy,
            need >> 20, dev_new >> 20, job ? 1 : 0);
}

void Runtime::load_image(Seq& s, const SeqImage& img) const {
  if (img.fence) img.fence->wait();   // its host copies finished
  std::vector<std::pair<const uint8_t*, size_t>> v;
  std::vector<std::vector<uint8_t>> flat;   // small pieces held in several segments (header, tokens, rope — read on the host) are joined
  flat.reserve(img.bufs.size());
  // Large pieces in several segments (an image sharing a boundary snapshot's prefix) go to the device segment by segment: joining them
  //   first copied the whole KV once more on the engine thread — measured 2026-10-14, a 192K-token prompt checkpoint built on its boundary
  //   snapshot: 12 GB joined at ~1.5 GB/s (page faults) held the next request 12.5 s in the queue. Only the device copy reads them (load_views
  //   hands each such view whole to h2d), so the view's pointer is a key into `segs`.
  std::unordered_map<const uint8_t*, const HostImageBuffer*> segs;
  constexpr size_t kJoinMax = (size_t)64 << 20;
  for (const auto& hb : img.bufs) {
    if (hb.segments.empty()) { v.push_back({nullptr, 0}); continue; }
    if (hb.segments.size() == 1) { v.push_back({hb.segments[0].data.get(), hb.segments[0].n}); continue; }
    if (hb.size() > kJoinMax) { segs[hb.segments[0].data.get()] = &hb; v.push_back({hb.segments[0].data.get(), hb.size()}); continue; }
    std::vector<uint8_t> f(hb.size());
    size_t off = 0;
    for (const auto& sg : hb.segments) { memcpy(f.data() + off, sg.data.get(), sg.n); off += sg.n; }
    flat.push_back(std::move(f));
    v.push_back({flat.back().data(), flat.back().size()});
  }
  model_.eng().load_views(s, v, [&](void* d, const uint8_t* h, size_t n) {
    const auto it = segs.find(h);
    if (it == segs.end()) { staged_h2d(d, h, n); return; }
    HIVE_CHECK(n == it->second->size(), "gemma segmented image piece read partially");
    size_t off = 0;
    for (const auto& sg : it->second->segments) { staged_h2d((uint8_t*)d + off, sg.data.get(), sg.n); off += sg.n; }
  });
  HIVE_CHECK(s.pos == img.pos, "gemma image position mismatch");
  s.tokens = img.tokens; s.generation = img.generation; s.broken = false; s.engram_history.clear();
  s.mtp_hidden_valid = s.h_last_valid && model_.eng().mtp_on();
}

// Sampler candidates on the host: argmax, max / Σ exp((l − max) · inv_temp) and the NC largest logits (any order — hived sorts what it needs).
void Runtime::cands(const float* lg, int M) {
  const int V = model_.eng().cfg().vocab, NC = opt_.sampler_cands;
  if (model_.eng().placement().head_gpu) {   // the same quantities on the device (FN's GPU runtime kernels) — the logits are already there
    cudaStream_t st = model_.eng().stream();
    const float* dl = model_.eng().logits_dev();
    CUDA_CHECK(cudaMemcpyAsync(c_it_.p, inv_temp_.data(), (size_t)M * 4, cudaMemcpyHostToDevice, st));
    hive::k::argmax_rows(dl, M, V, c_next_.as<int32_t>(), st);
    hive::k::row_max_sumexp(dl, M, V, c_it_.as<float>(), c_max_.as<float>(), c_sum_.as<float>(), st);
    if (NC > 0) {
      hive::k::topk_select_rows(dl, M, V, NC, V, c_idx_.as<int32_t>(), NC, st);
      hive::k::gather_vals(dl, c_idx_.as<int32_t>(), M, NC, V, c_val_.as<float>(), st);
      CUDA_CHECK(cudaMemcpyAsync(cand_idx_.data(), c_idx_.p, (size_t)M * NC * 4, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaMemcpyAsync(cand_val_.data(), c_val_.p, (size_t)M * NC * 4, cudaMemcpyDeviceToHost, st));
    }
    CUDA_CHECK(cudaMemcpyAsync(next_.data(), c_next_.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(rmax_.data(), c_max_.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(rsum_.data(), c_sum_.p, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    return;
  }
  for (int m = 0; m < M; ++m) {
    const float* r = lg + (size_t)m * V;
    int am = 0; float mx = r[0];
    for (int i = 1; i < V; ++i) if (r[i] > mx) { mx = r[i]; am = i; }
    const float it = inv_temp_[m];
    double s = 0;
    for (int i = 0; i < V; ++i) s += std::exp((double)(r[i] - mx) * it);
    next_[m] = am; rmax_[m] = mx; rsum_[m] = (float)s;
    if (NC > 0) {
      std::vector<int32_t> id(V);
      for (int i = 0; i < V; ++i) id[i] = i;
      const int k = std::min(NC, V);
      std::nth_element(id.begin(), id.begin() + k, id.end(), [&](int a, int b) { return r[a] > r[b] || (r[a] == r[b] && a < b); });
      for (int i = 0; i < k; ++i) { cand_idx_[(size_t)m * NC + i] = id[i]; cand_val_[(size_t)m * NC + i] = r[id[i]]; }
    }
  }
}

int32_t Runtime::forward(Seq& seq, const int32_t* ids, int M, const std::vector<ImageInput>* images, std::vector<float>* logits_out, ForwardStats* stats,
                         bool) {
  HIVE_CHECK(M >= 1, "gemma forward: no tokens");
  GemmaEngine& e = model_.eng();
  std::vector<GemmaImage> gi;
  if (images) for (const ImageInput& x : *images) {
    HIVE_CHECK(has_vision(), "gemma: image inputs without the vision embedder");
    HIVE_CHECK((int)x.types.size() == x.n_vit_h * x.n_vit_w, "gemma: an image span is one soft token per patch");
    gi.push_back({x.start, x.n_vit_h, x.n_vit_w, x.patches_dev});
  }
  if (M == 1 && gi.empty()) {
    std::vector<Seq*> s{&seq};
    std::vector<int32_t> next;
    std::vector<std::vector<float>> lg;
    forward_batch(s, ids, next, logits_out ? &lg : nullptr, stats);
    if (logits_out) *logits_out = std::move(lg[0]);
    return next[0];
  }
  const double t0 = hive::mono_ms();
  // no layer yield: GemmaEngine has one set of forward buffers (residual, norms, q / k / v), so a decode run inside a prefill's layer
  //   boundary would overwrite the prefill's residual; prompts are prefilled in chunks (HIVE_GEMMA_CHUNK) between decode steps instead
  constexpr bool kLayerYield = false;
  const bool ly_ok = kLayerYield && ly_.on() && ly_.depth < ly_.h.max_depth && M >= opt_.prefill_threshold;
  struct LyHook {
    GemmaEngine& e; bool on; std::function<void(int)> prev; std::function<bool()> prev_due;
    ~LyHook() { if (on) { e.layer_boundary = std::move(prev); e.layer_boundary_due = std::move(prev_due); } }
  } lh{e, ly_ok, ly_ok ? e.layer_boundary : std::function<void(int)>{}, ly_ok ? e.layer_boundary_due : std::function<bool()>{}};
  if (ly_ok) {
    if (ly_.depth == 0) ly_.last = hive::mono_ms();
    e.layer_boundary = [this](int l) { ly_progress_ = (l + 1.0) / std::max(1, model_.eng().cfg().n_layers); layer_yield_point(); };
    // the same preconditions ly::yield_point checks first (period elapsed since the last resume, work waiting): otherwise no sync at all
    e.layer_boundary_due = [this] { return hive::mono_ms() - ly_.last >= ly_.h.period_ms && ly_.h.want && ly_.h.want() > 0; };
  }
  const int V = e.cfg().vocab;
  std::vector<float> lg(V);
  try { e.prefill(seq, ids, M, lg.data(), gi.empty() ? nullptr : &gi); } catch (...) { seq.broken = true; throw; }
  inv_temp_[0] = 1.f;
  cands(lg.data(), 1);
  if (logits_out) *logits_out = std::move(lg);
  seq.mtp_hidden_valid = seq.h_last_valid && e.mtp_on();
  if (stats) stats->ms_total += hive::mono_ms() - t0;
  return next_[0];
}

void Runtime::forward_batch(std::vector<Seq*>& seqs, const int32_t* ids, std::vector<int32_t>& next, std::vector<std::vector<float>>* logits_out,
                            ForwardStats* stats) {
  const int M = (int)seqs.size(), V = model_.eng().cfg().vocab;
  HIVE_CHECK(M >= 1 && M <= 16, "gemma decode batch size");
  const double t0 = hive::mono_ms();
  std::vector<GemmaSeq*> cs(seqs.begin(), seqs.end());
  float* lg = model_.eng().logits_host();   // logits stay in the engine's pinned staging buffer (no intermediate copy)
  const double gap = tm_last_ > 0 ? t0 - tm_last_ : 0;
  try { model_.eng().decode(cs.data(), ids, M, lg); } catch (...) { for (Seq* s : seqs) s->broken = true; throw; }
  const double t1 = hive::mono_ms();
  cands(lg, M);
  const double t2 = hive::mono_ms();
  for (Seq* q : seqs) q->mtp_hidden_valid = q->h_last_valid && model_.eng().mtp_on();
  next.assign(next_.begin(), next_.begin() + M);
  if (logits_out) {   // one vector per row, filled in parallel (the allocation's page faults included)
    logits_out->assign(M, std::vector<float>());
    for (int m = 0; m < M; ++m) (*logits_out)[m].assign(lg + (size_t)m * V, lg + (size_t)(m + 1) * V);
  }
  tm_add("decode M" + std::to_string(M), t1 - t0, hive::mono_ms() - t2, t2 - t1, gap);
  tm_last_ = hive::mono_ms();
  if (stats) {  // hived reads one entry per row (row_*.at(i)); no expert cache on the CPU path, so every count is 0
    stats->row_hit.assign(M, 0); stats->row_cpu.assign(M, 0); stats->row_dma.assign(M, 0);
    stats->ms_total += hive::mono_ms() - t0;
  }
}

void Runtime::tm_add(const std::string& k, double eng, double cp, double cd, double outside) const {
  static const bool on = getenv("HIVE_GEMMA_TIMING") && *getenv("HIVE_GEMMA_TIMING") && strcmp(getenv("HIVE_GEMMA_TIMING"), "0") != 0;
  if (!on) return;
  Tm& t = tm_[k];
  t.sum[0] += eng; t.sum[1] += cp; t.sum[2] += cd; t.sum[3] += outside; ++t.n;
  static const long every = std::max(10L, atol(getenv("HIVE_GEMMA_TIMING")) > 1 ? atol(getenv("HIVE_GEMMA_TIMING")) : 300L);   // HIVE_GEMMA_TIMING=N: print every N calls
  if (++tm_calls_ % every == 0) {
    for (auto& [name, v] : tm_)
      fprintf(stderr, "[gemma-timing] %-14s n %6ld · engine %6.2f · copy %5.2f · cands %5.2f · before (hived) %6.2f ms/call\n", name.c_str(), v.n, v.sum[0] / v.n,
              v.sum[1] / v.n, v.sum[2] / v.n, v.sum[3] / v.n);
    tm_.clear();
  }
}

void Runtime::mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats*) {
  GemmaEngine& e = model_.eng();
  const int k = e.mtp_k();
  drafts.assign(k, 0); conf.assign(k, 0.f);
  const double t0 = hive::mono_ms(), gap = tm_last_ > 0 ? t0 - tm_last_ : 0;
  const int n = e.mtp_draft(seq, tok, k, drafts.data(), conf.data());
  tm_add("draft S1", hive::mono_ms() - t0, 0, 0, gap);
  tm_last_ = hive::mono_ms();
  drafts.resize(n); conf.resize(n);
  for (float& c : conf) { const float p = std::min(std::max(c, 1e-6f), 1.f - 1e-6f); c = std::log(p / (1.f - p)); }  // as the GPU family
}

void Runtime::mtp_draft_batch(const std::vector<Seq*>& seqs, const std::vector<int32_t>& toks, std::vector<std::vector<int32_t>>& drafts,
                              std::vector<std::vector<float>>& conf, ForwardStats*) {
  GemmaEngine& e = model_.eng();
  const int S = (int)seqs.size(), k = e.mtp_k();
  std::vector<GemmaSeq*> qs(seqs.begin(), seqs.end());
  std::vector<int32_t> d((size_t)S * 7, 0);
  std::vector<float> c((size_t)S * 7, 0.f);
  std::vector<int> n(S, 0);
  const double t0 = hive::mono_ms(), gap = tm_last_ > 0 ? t0 - tm_last_ : 0;
  e.mtp_draft_batch(qs.data(), toks.data(), S, k, d.data(), c.data(), n.data());
  tm_add("draft S" + std::to_string(S), hive::mono_ms() - t0, 0, 0, gap);
  tm_last_ = hive::mono_ms();
  drafts.assign(S, {}); conf.assign(S, {});
  for (int s = 0; s < S; ++s) {
    drafts[s].assign(d.begin() + (size_t)s * 7, d.begin() + (size_t)s * 7 + n[s]);
    for (int i = 0; i < n[s]; ++i) { const float p = std::min(std::max(c[(size_t)s * 7 + i], 1e-6f), 1.f - 1e-6f); conf[s].push_back(std::log(p / (1.f - p))); }
  }
}

void Runtime::forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats) {
  std::vector<VerifyPart> p{{&seq, ids, M}};
  forward_verify_batch(p, logits_rows, stats);
}

void Runtime::forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats) {
  const int S = (int)parts.size(), V = model_.eng().cfg().vocab;
  std::vector<GemmaSeq*> seqs(S);
  std::vector<int> Ms(S);
  std::vector<int32_t> ids;
  for (int p = 0; p < S; ++p) { seqs[p] = parts[p].seq; Ms[p] = parts[p].M; ids.insert(ids.end(), parts[p].ids, parts[p].ids + parts[p].M); }
  const int R = (int)ids.size();
  HIVE_CHECK(R >= 1 && R <= 32, "gemma verify rows");
  const double t0 = hive::mono_ms();
  const double gap = tm_last_ > 0 ? t0 - tm_last_ : 0;
  // hived's row buffer is reused for every verify: reserved once at 32 rows and pinned, the graph's D2H writes into it directly
  const bool rows_on = verify_logits_ || !model_.eng().placement().head_gpu;   // set_verify_logits(false): greedy rows, logits stay on the device
  const size_t cap = (size_t)32 * V;
  if (rows_on && logits_rows.capacity() < cap) logits_rows.reserve(cap);
  if (rows_on) logits_rows.resize((size_t)R * V); else logits_rows.clear();
  // direct D2H into hived's buffer is off: hived may reallocate or free that vector while it is registered (measured 2026-10-10: cudaMalloc
  //   then failed with "resource already mapped" and requests were refused). (q27 history; not offered here).
  const bool direct = false;
  float* lg = direct ? logits_rows.data() : model_.eng().logits_host();
  struct HostLogits { GemmaEngine& e; ~HostLogits() { e.set_host_logits(true); } } hl_restore{model_.eng()};
  model_.eng().set_host_logits(rows_on);
  try { model_.eng().verify_batch(seqs.data(), Ms.data(), S, ids.data(), lg); } catch (...) { for (auto& p : parts) p.seq->broken = true; throw; }
  const double t1 = hive::mono_ms();
  if (rows_on && !direct) {
    const size_t n = (size_t)R * V, per = (n + 15) / 16;
    float* dst = logits_rows.data();
    (void)per;
    memcpy(dst, lg, n * 4);
  }
  const double t2 = hive::mono_ms();
  cands(lg, R);
  const double t3 = hive::mono_ms();
  tm_add("verify S" + std::to_string(S) + " R" + std::to_string(R), t1 - t0, t2 - t1, t3 - t2, gap);
  tm_last_ = t3;
  vb_seqs_.clear();
  for (auto& p : parts) vb_seqs_.push_back(p.seq);
  if (stats) {
    stats->row_hit.assign(R, 0); stats->row_cpu.assign(R, 0); stats->row_dma.assign(R, 0);
    stats->ms_total += hive::mono_ms() - t0;
  }
}

void Runtime::rollback_batch(const std::vector<int>& n_keep) {
  HIVE_CHECK(n_keep.size() == vb_seqs_.size(), "gemma rollback: parts");
  const double t0 = hive::mono_ms(), gap = tm_last_ > 0 ? t0 - tm_last_ : 0;
  model_.eng().rollback_batch(n_keep.data(), (int)n_keep.size());
  tm_add("rollback S" + std::to_string(n_keep.size()), hive::mono_ms() - t0, 0, 0, gap);
  tm_last_ = hive::mono_ms();
  for (Seq* s : vb_seqs_) s->mtp_hidden_valid = s->h_last_valid && model_.eng().mtp_on();
  vb_seqs_.clear();
}

void Runtime::layer_yield_point() {
  if (!ly_.on() || ly_.depth >= ly_.h.max_depth) return;
  ly_.floor = 8;
  ly_.cap = 8;   // the CPU engine's inner forwards use their own buffers: decode / verify rows only
  ly::yield_point(ly_, [&](int) { return true; }, [&](int) {}, [] { return hive::mono_ms(); });
}

}  // namespace hive::gemma::fam
