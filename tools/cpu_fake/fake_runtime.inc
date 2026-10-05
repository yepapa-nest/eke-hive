// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Fake Model/Runtime for running the REAL hived.cpp on CPU (tools/test_daemon_cpu.py).
// The generator (harness.runtime_source) prepends runtime.h and the PRODUCTION text of
// env_on, the Runtime ctor option/pinned-pool block (cudaHostAlloc there and in save_image is routed through
// fake_inj_cudaHostAlloc below for failure injection), reset_seq, tail_mode_for,
// quiesce_after_host_error, save_image, load_image, promote_after_step, slots_for, the round-2/3 host helpers (tile_plan,
// host_tile_count, plan_units) and the HostStager class (sliced from engine/src/runtime.cpp; ctor option block sets
// batch_prefill_/host_slots_ from HIVE_BATCH_PREFILL / HIVE_PREFILL_HOST_TILES / HIVE_PREFILL_HOST_MB exactly as production).
//
// "Model state" oracle: every processed position p gets content(p) = token | image_hash<<32.
//   * comp_cache[l] (append-only, ratio 1) holds content(p) in row p      (saved as prefix rows)
//   * ring[l]       (window W = cfg.window) holds content(p) at p % W       (saved whole)
// After every forward the fake checks ring rows of the last W positions == comp rows and
// emits argmax = 10 + FNV64(content[0..pos)) % 4000. A test can therefore predict every
// token and any wrong prefix reuse / restore / fence ordering shows up as a wrong token or
// an explicit "fake state" error. Device writes of a forward run on st_ as a stream op, so
// with FAKE_CUDA_ASYNC=1 they are ordered against the snapshot stream only by the
// production fence waits (copied below from runtime.cpp's forward/forward_batch preambles).
#include <atomic>
#include <new>
#include <thread>
#include "hive/clock.h"

namespace hive {
void fake_ckpt_put(const std::string&, const uint8_t*, size_t, const std::string& = "U8");
constexpr int kFakePatchBytes = 16;  // bytes per ViT patch in the fake image payload
constexpr uint64_t kFnvBasis = 1469598103934665603ull, kFnvPrime = 1099511628211ull;
static uint64_t fnv(const uint8_t* p, size_t n, uint64_t h = kFnvBasis) { for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * kFnvPrime; return h; }
using FakeMs = std::chrono::milliseconds;
static int fake_env(const char* n, int d) { const char* v = getenv(n); return v && *v ? atoi(v) : d; }

// ---- Model ----
Model::Model(const std::string& dir, int max_layer, int64_t rope_len) : ckpt_(dir) {
  cfg_.dim = 512; cfg_.moe_inter = 512; cfg_.n_routed = 8; cfg_.n_act = 2; cfg_.dspark_experts = 4; cfg_.n_layers = 2;
  cfg_.vocab = 4096; cfg_.window = 64; cfg_.compress_ratios = {1, 1}; cfg_.swiglu_limit = 10.f;
  if (fake_env("FAKE_KV_SOURCE", 0)) cfg_.kv_source_layers = {0};  // tail_mode_for needs a kv source layer (C1/tail-split observable in test_daemon_cpu)
  const int nl = max_layer >= 0 ? std::min(max_layer + 1, cfg_.n_layers) : cfg_.n_layers;
  for (int l = 0; l < nl; ++l) layers_.push_back(std::make_unique<LayerWeights>());
  head_.alloc(16);
  rope_len_ = rope_len;
  static std::vector<std::vector<uint8_t>> blobs;  // expert tensors for ExpertStore::load_layer_experts
  const size_t wsz = (size_t)cfg_.moe_inter * cfg_.dim / 2, ssz = (size_t)cfg_.moe_inter * cfg_.dim / 32;
  uint32_t x = 12345;
  for (int l = 0; l < cfg_.n_layers; ++l)
    for (int e = 0; e < cfg_.n_routed; ++e)
      for (const char* m : {"w1", "w2", "w3"}) {
        const std::string p = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + "." + m;
        blobs.emplace_back(wsz); for (auto& v : blobs.back()) { x = x * 1664525u + 1013904223u; v = (uint8_t)(x >> 24); }
        fake_ckpt_put(p + ".weight", blobs.back().data(), wsz);
        blobs.emplace_back(ssz); for (auto& v : blobs.back()) { x = x * 1664525u + 1013904223u; v = (uint8_t)(117 + (x >> 24) % 6); }
        fake_ckpt_put(p + ".scale", blobs.back().data(), ssz);
      }
}
Model::~Model() {}
void Model::load_mtp() {}
Blas::~Blas() {}
void EngramHash::load(const std::string&, const std::string&) { throw std::runtime_error("fake: no engram"); }

struct Runtime::Work {};

// Pinned-allocation failure injection (snapshot pinned path; a CKPT_ASYNC pinned-allocation failure must not abort):
// FAKE_HOST_ALLOC_FAIL=N makes every N-th cudaHostAlloc issued by the production text below fail
// with cudaErrorMemoryAllocation (1 = every call; unset/0 = off). Counters read by test_snapshot_cpu.cpp.
std::atomic<long> fake_host_alloc_calls{0}, fake_host_alloc_injected{0};
static cudaError_t fake_inj_cudaHostAlloc(void** p, size_t n, unsigned flags) {
  static const int every = fake_env("FAKE_HOST_ALLOC_FAIL", 0);
  if (every > 0 && ++fake_host_alloc_calls % every == 0) { *p = nullptr; ++fake_host_alloc_injected; return cudaErrorMemoryAllocation; }
  return ::cudaHostAlloc(p, n, flags);
}
#define cudaHostAlloc hive::fake_inj_cudaHostAlloc

Runtime::Runtime(Model& model, ExpertStore& store, const EngramHash* eh, const RuntimeOptions& opt)
    : model_(model), store_(store), eh_(eh), opt_(opt) {
@@RUNTIME_CTOR_OPTIONS@@
  CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&promo_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&snapshot_st_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&promo_evt_, cudaEventDisableTiming));  // promote_after_step (production text) records it under REUSE_STAGE
  opt_.sampler_cands = 0;  // host sampling from full (fake) logits
  cand_it_h_ = new float[256];
  vision_can_ = true;
  mtp_on_ = opt_.mtp && fake_env("FAKE_MTP", 0) > 0;  // FAKE_MTP=B -> oracle drafts for DSpark block B (mtp_draft below)
  // verify2_/mtp_batch_ are read by the PRODUCTION option block above (HIVE_MTP_VERIFY2, HIVE_MTP_BATCH). The usability condition at the end of
  //   the constructor (fuse_, no dumps) is always true in the fake; row cap = min(8, max_batch) (production v2_rows_ = min(8, Mb, Msub, max_chunk, snapshot rows)).
  v2_rows_ = std::min(8, std::max(1, opt_.max_batch));
  v2_ok_ = mtp_on_ && (verify2_ || mtp_batch_) && v2_rows_ >= 2;
  // Production ctor's layer-yield floor/cap (runtime.cpp — same formula)
  { const Config& c = model_.cfg(); ly_.cap = opt_.max_chunk; ly_.floor = std::max({c.window, std::max(1, std::min(opt_.max_batch, opt_.max_chunk)), c.dspark_block + 1}); }
}
Runtime::~Runtime() {
  if (snapshot_st_) { cudaStreamSynchronize(snapshot_st_); cudaStreamDestroy(snapshot_st_); }
  for (auto s : {st_, side_, promo_}) if (s) { cudaStreamSynchronize(s); cudaStreamDestroy(s); }
  if (promo_evt_) cudaEventDestroy(promo_evt_);
  delete[] cand_it_h_;
}
std::unique_ptr<Seq> Runtime::new_seq() const {
  const Config& c = model_.cfg();
  auto s = std::make_unique<Seq>();
  s->cap = opt_.max_ctx;
  s->ring.resize(model_.n_loaded_layers());
  for (int l = 0; l < model_.n_loaded_layers(); ++l) {
    s->ring[l].alloc((size_t)c.window * 8);
    CUDA_CHECK(cudaMemset(s->ring[l].p, 0, s->ring[l].n));
    s->comp_cache[l].alloc((size_t)(s->cap + 2) * kvp::COMP_ROW);
  }
  return s;
}
void Runtime::ensure_vision() {}
void Runtime::release_vision_if_idle() {}
int Runtime::warm_cache() { return 0; }
int Runtime::image_patch_bytes() const { return 16; }  // the CPU harness sends 16-byte fake patches (test_daemon_cpu PATCH_BYTES)
void Runtime::warm_defer(int per_step) { warm_left_ = opt_.warm_cap; warm_step_ = std::max(1, per_step); warm_steps_ = 0; warm_done_ = 0; }  // HIVE_WARM_DEFER: quota only (the fake store has no slots)
int Runtime::warm_from_keys(const std::vector<int32_t>&) { return 0; }
int32_t Runtime::row_argmax(int) const { return -1; }
int Runtime::mtp_block() const { return mtp_on_ ? fake_env("FAKE_MTP", 0) : 0; }
// tools/test_mtp_cpu.py: fake DSpark. Drafts = the oracle's own continuation (fake_next's FNV chain), so a
// test controls acceptance exactly: FAKE_MTP_WRONG=N corrupts draft i when (pos0 + i) % N == 0 (then rejection/rollback run).
// FAKE_MTP_CONF (default 5) = every draft's confidence, except the process's first FAKE_MTP_LOWCONF_DRAFTS draft calls (conf −5: declined, so
// the daemon measures normal steps first — reproduces the service state where the default gate's stale verify table locks MTP out). Timing: FAKE_MTP_DRAFT_MS per draft, FAKE_VERIFY_MS + FAKE_VERIFY_ROW_MS·(M−1) per verify.
// forward_verify = M single-row fake forwards (so every row's logits are the real oracle's) after a ring snapshot; rollback restores
// pos/tokens/history and the ring (snapshot, then re-applies the kept rows from comp rows) — a wrong rollback shows up as a fake-state error.
static std::vector<std::vector<uint8_t>> fake_ring_snap;
static void fake_sleep_ms(int ms) { if (ms > 0) std::this_thread::sleep_for(FakeMs(ms)); }
// tools/test_sleep_cpu.py: sleep/wake on the REAL ExpertStore (sleep_release · wake_alloc · restore_keys). The fake has no Work,
// graphs or vision, so only the store part of runtime.cpp's Runtime::sleep_release/wake_restore/wake_warm is modelled. FAKE_WAKE_FAIL=N: the first N
// wakes report "not enough VRAM" (nothing allocated) · FAKE_WAKE_MS: delay per warm progress report (observe "waking" from stats) · every fake forward
// aborts while asleep (the scheduler must never run one).
size_t Runtime::sleep_release() {
  if (asleep_) return 0;
  store_.wait_jobs();
  store_.flush_promotions();
  for (auto s : {st_, side_, promo_, snapshot_st_}) CUDA_CHECK(cudaStreamSynchronize(s));
  const size_t f = store_.sleep_release();
  asleep_ = true;
  return f;
}
int Runtime::wake_restore(size_t reserve_bytes, std::string* why) {
  if (!asleep_) return store_.n_slots();
  static int fails = fake_env("FAKE_WAKE_FAIL", 0);
  if (fails > 0) { --fails; if (why) *why = "fake: not enough free VRAM"; return -1; }
  const int n = store_.wake_alloc(reserve_bytes);
  if (n < 0) { if (why) *why = "store wake_alloc failed"; return -1; }
  asleep_ = false;
  return n;
}
int Runtime::wake_warm(const std::vector<int32_t>& keys, const std::function<void(int, int)>& progress) {
  if (asleep_) return 0;
  return store_.restore_keys(keys, promo_, [&](int d, int t) { if (progress) progress(d, t); fake_sleep_ms(fake_env("FAKE_WAKE_MS", 0)); });
}
static void fake_awake_check(bool asleep) { if (asleep) { fprintf(stderr, "fake: forward while asleep\n"); abort(); } }
void Runtime::mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats*) {
  HIVE_CHECK(mtp_on_ && seq.mtp_hidden_valid && seq.mtp_pos == seq.pos - 1, "mtp_draft needs the hidden of pos-1");
  CUDA_CHECK(cudaStreamSynchronize(st_));
  const int B = mtp_block(), wrong = fake_env("FAKE_MTP_WRONG", 0);
  const char* cf = getenv("FAKE_MTP_CONF");
  static int n_calls = 0;
  const float cv = n_calls++ < fake_env("FAKE_MTP_LOWCONF_DRAFTS", 0) ? -5.f : cf && *cf ? (float)atof(cf) : 5.f;
  uint64_t h = kFnvBasis;
  const uint8_t* comp = seq.comp_cache.at(0).as<uint8_t>();
  for (int64_t p = 0; p < seq.pos; ++p) h = fnv(comp + (size_t)p * kvp::COMP_ROW, 8, h);
  uint64_t v = (uint32_t)tok;
  drafts.clear(); conf.assign(B, cv);
  for (int i = 0; i < B; ++i) {
    h = fnv(reinterpret_cast<const uint8_t*>(&v), 8, h);
    int32_t d = (int32_t)(10 + h % 4000);
    if (wrong > 0 && (seq.pos + i) % wrong == 0) d = d == 4009 ? 10 : d + 1;
    drafts.push_back(d);
    v = (uint32_t)d;
  }
  fake_sleep_ms(fake_env("FAKE_MTP_DRAFT_MS", 0));
}
void Runtime::forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats) {
  HIVE_CHECK(mtp_on_ && M >= 2 && M <= mtp_block() + 1 && M <= 8, "verify rows");
  HIVE_CHECK(v_M_ == 0, "rollback pending");
  CUDA_CHECK(cudaStreamSynchronize(st_));
  fake_ring_snap.clear();
  for (auto& r : seq.ring) fake_ring_snap.emplace_back(r.as<uint8_t>(), r.as<uint8_t>() + r.n);
  v_pos0_ = seq.pos; v_M_ = M; v_hist_ = seq.engram_history.size();
  verify_ = true;
  struct G { Runtime* r; ~G() { r->verify_ = false; if (std::uncaught_exceptions()) r->v_M_ = 0; } } g{this};
  const int V = model_.cfg().vocab;
  logits_rows.assign((size_t)M * V, 0.f);
  std::vector<float> row;
  for (int i = 0; i < M; ++i) {
    forward(seq, ids + i, 1, nullptr, &row, stats);
    std::copy(row.begin(), row.end(), logits_rows.begin() + (size_t)i * V);
  }
  fake_sleep_ms(fake_env("FAKE_VERIFY_MS", 0) + fake_env("FAKE_VERIFY_ROW_MS", 0) * (M - 1));
}
void Runtime::rollback(Seq& seq, int n_keep) {
  HIVE_CHECK(v_M_ > 0 && n_keep >= 1 && n_keep <= v_M_, "rollback range");
  CUDA_CHECK(cudaStreamSynchronize(st_));
  if (n_keep < v_M_) {
    seq.pos = v_pos0_ + n_keep;
    seq.tokens.resize(seq.tokens.size() - (size_t)(v_M_ - n_keep));
    seq.engram_history.resize(v_hist_ + (size_t)n_keep);
    const int W = model_.cfg().window;
    for (size_t l = 0; l < seq.ring.size(); ++l) {
      uint8_t* ring = seq.ring[l].as<uint8_t>();
      memcpy(ring, fake_ring_snap[l].data(), fake_ring_snap[l].size());
      const uint8_t* comp = seq.comp_cache.at((int)l).as<uint8_t>();
      for (int64_t p = v_pos0_; p < seq.pos; ++p) memcpy(ring + (size_t)(p % W) * 8, comp + (size_t)p * kvp::COMP_ROW, 8);
    }
  }
  seq.mtp_hidden_valid = true; seq.mtp_pos = seq.pos - 1;
  v_M_ = 0;
}

// HIVE_MTP_BATCH (tools/test_mtp_cpu.py): forward_verify_batch = per part, the same as the fake forward_verify (ring snapshot of that
// sequence, then M_p single-row oracle forwards) — parts must be distinct sequences, M_p >= 1, sum <= mtp_batch_rows() (production contract; a violation
// aborts, like forward_multi's). Timing: FAKE_VERIFY_MS + FAKE_VERIFY_ROW_MS·(rows−1) once for the whole batch. rollback_batch = the fake rollback per part.
static std::vector<std::vector<std::vector<uint8_t>>> fake_ring_snaps;
void Runtime::forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats) {
  int rows = 0;
  for (auto& p : parts) {
    for (auto& q : parts) if (&q != &p && q.seq == p.seq) { fprintf(stderr, "fake: forward_verify_batch one part per sequence\n"); abort(); }
    if (!p.seq || p.M < 1 || p.M > mtp_block() + 1) { fprintf(stderr, "fake: forward_verify_batch part rows %d\n", p.M); abort(); }
    rows += p.M;
  }
  if (!mtp_batch_enabled() || parts.empty() || (int)parts.size() > kVParts || rows > mtp_batch_rows() || v_M_ != 0 || !vparts_.empty()) {
    fprintf(stderr, "fake: forward_verify_batch contract (enabled %d parts %zu rows %d > %d pending %d)\n", mtp_batch_enabled() ? 1 : 0, parts.size(), rows,
            mtp_batch_rows(), v_M_);
    abort();
  }
  CUDA_CHECK(cudaStreamSynchronize(st_));
  fake_ring_snaps.clear();
  vparts_.clear();
  verify_ = true;
  struct G { Runtime* r; ~G() { r->verify_ = false; if (std::uncaught_exceptions()) { r->vparts_.clear(); r->v_M_ = 0; } } } g{this};
  const int V = model_.cfg().vocab;
  logits_rows.assign((size_t)rows * V, 0.f);
  std::vector<float> row;
  int r = 0;
  for (auto& p : parts) {
    Seq& seq = *p.seq;
    fake_ring_snaps.emplace_back();
    for (auto& rg : seq.ring) fake_ring_snaps.back().emplace_back(rg.as<uint8_t>(), rg.as<uint8_t>() + rg.n);
    vparts_.push_back(VPart{&seq, seq.pos, p.M, r, seq.engram_history.size()});
    for (int i = 0; i < p.M; ++i, ++r) {
      forward(seq, p.ids + i, 1, nullptr, &row, stats);
      std::copy(row.begin(), row.end(), logits_rows.begin() + (size_t)r * V);
    }
  }
  v_M_ = rows;
  if (stats) { stats->row_hit.assign(rows, 0); stats->row_cpu.assign(rows, 0); stats->row_dma.assign(rows, 0); }
  // FAKE_VERIFY_FIRST_MS = ms added to the first verify of each row count (production run_graph: first use of a graph key runs eagerly) and bumps
  //   graph_eager_ (not bumped with FAKE_VERIFY_FIRST_SILENT=1 — a negative control that mimics a runtime unable to notice first uses).
  static bool seen_rows[64] = {};
  int first_ms = 0;
  if (rows < 64 && !seen_rows[rows]) {
    seen_rows[rows] = true;
    first_ms = fake_env("FAKE_VERIFY_FIRST_MS", 0);
    if (first_ms > 0 && !fake_env("FAKE_VERIFY_FIRST_SILENT", 0)) ++graph_eager_;
  }
  fake_sleep_ms(fake_env("FAKE_VERIFY_MS", 0) + fake_env("FAKE_VERIFY_ROW_MS", 0) * (rows - 1) + first_ms);
}
void Runtime::rollback_batch(const std::vector<int>& n_keep) {
  HIVE_CHECK(!vparts_.empty() && n_keep.size() == vparts_.size(), "rollback parts");
  CUDA_CHECK(cudaStreamSynchronize(st_));
  const int W = model_.cfg().window;
  for (size_t pi = 0; pi < vparts_.size(); ++pi) {
    const VPart& vp = vparts_[pi];
    Seq& seq = *vp.seq;
    HIVE_CHECK(n_keep[pi] >= 1 && n_keep[pi] <= vp.M, "rollback range");
    if (n_keep[pi] < vp.M) {
      seq.pos = vp.pos0 + n_keep[pi];
      seq.tokens.resize(seq.tokens.size() - (size_t)(vp.M - n_keep[pi]));
      seq.engram_history.resize(vp.hist + (size_t)n_keep[pi]);
      for (size_t l = 0; l < seq.ring.size(); ++l) {
        uint8_t* ring = seq.ring[l].as<uint8_t>();
        memcpy(ring, fake_ring_snaps[pi][l].data(), fake_ring_snaps[pi][l].size());
        const uint8_t* comp = seq.comp_cache.at((int)l).as<uint8_t>();
        for (int64_t p = vp.pos0; p < seq.pos; ++p) memcpy(ring + (size_t)(p % W) * 8, comp + (size_t)p * kvp::COMP_ROW, 8);
      }
    }
    seq.mtp_hidden_valid = true; seq.mtp_pos = seq.pos - 1;
  }
  vparts_.clear();
  v_M_ = 0;
}

namespace {
// Runs a small batch of real CPU expert jobs on the production pool (concurrency exposure).
void fake_cpu_experts(ExpertStore& store, const Config& c, int M) {
  static std::vector<float> af(c.dim, 0.5f), saf(c.dim / 32, 1.f / 64);
  const int nj = std::min(M, 3), R = std::min(M, 4);
  std::vector<ExpertStore::Job> jobs(nj);
  std::vector<std::vector<float>> outs, scratch;
  outs.reserve(nj * R); scratch.reserve(nj);
  for (int j = 0; j < nj; ++j) {
    auto& J = jobs[j]; J.layer = j % c.n_layers; J.e = (M + j) % c.n_routed; J.R = R;
    scratch.emplace_back(ExpertStore::job_scratch_floats(c.moe_inter) * R); J.scratch = scratch.back().data();
    for (int r = 0; r < R; ++r) { J.a_f[r] = af.data(); J.a_s[r] = saf.data(); J.route_w[r] = 0.5f; outs.emplace_back(c.dim); J.out[r] = outs.back().data(); }
  }
  store.start_jobs(jobs);
  store.wait_jobs();
}
uint64_t content_at(const std::vector<ImageInput>* images, const int32_t* ids, int m) {
  uint64_t v = (uint32_t)ids[m];
  if (images) for (const auto& im : *images)
    if (m >= im.start && m < im.start + (int)im.types.size()) {
      const uint64_t h = fnv(reinterpret_cast<const uint8_t*>(im.patches_dev), (size_t)im.n_vit_h * im.n_vit_w * kFakePatchBytes);
      v |= (h & 0xffffffffull) << 32;
    }
  return v;
}
}  // namespace

// Writes contents for positions [pos, pos+n) on st_ (a stream op, like GPU kernels).
static void fake_write(cudaStream_t st, Seq& seq, int W, std::shared_ptr<std::vector<uint64_t>> vals) {
  const int64_t pos = seq.pos;
  std::vector<std::pair<uint8_t*, uint8_t*>> dst;  // (comp base, ring base) per layer
  for (size_t l = 0; l < seq.ring.size(); ++l) dst.push_back({seq.comp_cache.at((int)l).as<uint8_t>(), seq.ring[l].as<uint8_t>()});
  fakecuda::enqueue(st, [dst, vals, pos, W] {
    for (auto [comp, ring] : dst)
      for (size_t i = 0; i < vals->size(); ++i) {
        const int64_t p = pos + (int64_t)i;
        memcpy(comp + (size_t)p * kvp::COMP_ROW, &(*vals)[i], 8);
        memcpy(ring + (size_t)(p % W) * 8, &(*vals)[i], 8);
      }
  });
}
// Host-side check + argmax after st_ drained. Throws on inconsistent restored state.
static int32_t fake_next(Seq& seq, int W) {
  const int64_t pos = seq.pos;
  // engram_history models the non-idempotent host state (production: engram n-gram history,
  // compressor state): one entry per processed position, saved/restored by save/load_image.
  if ((int64_t)seq.engram_history.size() != pos)
    throw std::runtime_error("fake state: history " + std::to_string(seq.engram_history.size()) + " != pos " + std::to_string(pos));
  uint64_t h = kFnvBasis;
  for (size_t l = 0; l < seq.ring.size(); ++l) {
    const uint8_t* comp = seq.comp_cache.at((int)l).as<uint8_t>();
    const uint8_t* ring = seq.ring[l].as<uint8_t>();
    for (int64_t p = std::max<int64_t>(0, pos - W); p < pos; ++p)
      if (memcmp(comp + (size_t)p * kvp::COMP_ROW, ring + (size_t)(p % W) * 8, 8) != 0)
        throw std::runtime_error("fake state: ring/comp mismatch at layer " + std::to_string(l) + " pos " + std::to_string(p));
    if (l == 0) for (int64_t p = 0; p < pos; ++p) h = fnv(comp + (size_t)p * kvp::COMP_ROW, 8, h);
  }
  return (int32_t)(10 + h % 4000);
}

// HIVE_LAYER_YIELD (tools/test_layer_yield_cpu.py): a model of the production layer loop and of Work.
//   fake_work = Work's rows (slot 0 = the first max_chunk rows of the running forward). EVERY fake forward writes its rows [0, min(M, cap)) at its head
//   (salt 0) and a decode step writes rows [0, floor) — like production, where any forward reuses Work from row 0. A yield-eligible forward
//   (production condition: original rows >= prefill_threshold, not verify, not inside a yield) runs FAKE_LAYERS layers (FAKE_PREFILL_MS split evenly);
//   after each layer it checks its rows still hold its own salt-l values, calls layer_yield_point() (production hive/layer_yield.h yield_point with
//   a fake park/unpark of rows [0, R)), checks again (a clobbered row = the yield body touched more than the parked rows, or park/unpark broken →
//   "fake state: layer-yield work rows clobbered"), then writes salt l+1. FAKE_LY_NO_PARK=1 = negative control (park/unpark do nothing).
//   FAKE_PREFILL_SHORT_MS: sleep of a forward with M < prefill_threshold instead of FAKE_PREFILL_MS (a short prompt is cheap in production).
static std::vector<uint64_t> fake_work;
static constexpr uint64_t kFakeSalt = 0x9e3779b97f4a7c15ull;
static double fake_now_ms() { return hive::mono_ms(); }
static void fake_work_put(const std::vector<uint64_t>& vals, int salt, size_t cap) {
  if (fake_work.size() < cap) fake_work.resize(cap, 0);
  for (size_t m = 0; m < std::min(vals.size(), cap); ++m) fake_work[m] = vals[m] ^ (kFakeSalt * (uint64_t)(salt + 1));
}
static void fake_work_check(const std::vector<uint64_t>& vals, int salt, size_t cap, const char* where) {
  for (size_t m = 0; m < std::min(vals.size(), cap); ++m)
    if (fake_work[m] != (vals[m] ^ (kFakeSalt * (uint64_t)(salt + 1))))
      throw std::runtime_error("fake state: layer-yield work rows clobbered at row " + std::to_string(m) + " (" + where + ")");
}
struct Runtime::LyPark { std::vector<uint64_t> rows; };
void Runtime::ly_begin(std::vector<std::pair<Seq*, int>> owners) {
  if (ly_.depth > 1) return;
  if (ly_.depth == 0) ly_.last = fake_now_ms();
  ly_owner_[ly_.depth] = std::move(owners);
}
void Runtime::layer_yield_point() {
  if (!ly_.on() || ly_.depth >= ly_.h.max_depth) return;
  const int lvl = std::min(ly_.depth, 1);
  if (!lypark_[lvl]) lypark_[lvl] = std::make_shared<LyPark>();
  LyPark& P = *lypark_[lvl];
  static const bool no_park = fake_env("FAKE_LY_NO_PARK", 0) > 0;
  bool pend = false;
  ly::yield_point(ly_,
      [&](int R) {
        CUDA_CHECK(cudaStreamSynchronize(st_));
        if (!no_park) P.rows.assign(fake_work.begin(), fake_work.begin() + std::min<size_t>((size_t)R, fake_work.size()));
        pend = prefill_pending_; prefill_pending_ = true;  // production: promotion pause / no elastic lending during the yield
        return true;
      },
      [&](int) {
        CUDA_CHECK(cudaStreamSynchronize(st_));
        if (!no_park) std::copy(P.rows.begin(), P.rows.end(), fake_work.begin());
        prefill_pending_ = pend;
      },
      [] { return fake_now_ms(); });
}
static void fake_layers(Runtime* rt, bool ly_ok, int ms, const std::vector<uint64_t>& vals, size_t cap, const std::function<void()>& yield) {
  const int FL = std::max(1, fake_env("FAKE_LAYERS", 1));
  for (int l = 0; l < FL; ++l) {
    if (ms > 0) std::this_thread::sleep_for(std::chrono::microseconds((long)ms * 1000 / FL));
    if (l + 1 < FL && ly_ok) {
      fake_work_check(vals, l, cap, "before yield");
      yield();
      fake_work_check(vals, l, cap, "after yield");
      fake_work_put(vals, l + 1, cap);
    }
  }
  (void)rt;
}

int32_t Runtime::forward(Seq& seq, const int32_t* ids, int M, const std::vector<ImageInput>* images, std::vector<float>* logits_out,
                         ForwardStats* stats, bool upper_needed) {
  (void)upper_needed; (void)stats;
  fake_awake_check(asleep_);
  if (!fake_env("FAKE_SKIP_SNAPSHOT_FENCE", 0))
    if (seq.snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, seq.snapshot_fence->event, 0));  // runtime.cpp forward preamble
  // Same guard as runtime.cpp forward: a host exception mid-forward marks the sequence broken.
  struct BrokenGuard { Seq* q; int ex; ~BrokenGuard() { if (std::uncaught_exceptions() > ex) q->broken = true; } } broken_guard{&seq, std::uncaught_exceptions()};
  HIVE_CHECK(seq.pos + M <= seq.cap, "context overflow");
  // capacity contract of the production forward: a chunk fits the tile slots (max_chunk × prefill_slots, H2 host tiles included).
  // Opt-in (FAKE_CHECK_CHUNK=1, set by tools/test_daemon_cpu.py): other fixtures drive this fake forward directly with larger chunks.
  if (fake_env("FAKE_CHECK_CHUNK", 0) && M > opt_.max_chunk * prefill_slots()) { fprintf(stderr, "fake: forward chunk %d > max_chunk %d x slots %d\n", M, opt_.max_chunk, prefill_slots()); abort(); }
  auto vals = std::make_shared<std::vector<uint64_t>>(M);
  for (int m = 0; m < M; ++m) (*vals)[m] = content_at(images, ids, m);
  // Production forward's yield eligibility (runtime.cpp ly_ok — here without the GPU-only conditions) + Work rows from row 0
  //   HIVE_LAYER_YIELD_SMALL (production: also HIVE_PREFILL_SMALL forwards below the threshold) — here: any forward below the threshold
  static const bool ly_small = [] { const char* v = getenv("HIVE_LAYER_YIELD_SMALL"); return v && *v && strcmp(v, "0") != 0; }();
  const bool ly_ok = ly_.on() && ly_.depth < ly_.h.max_depth && !verify_ && (M >= opt_.prefill_threshold || ly_small);
  if (ly_ok) ly_begin({{&seq, M}});
  fake_work_put(*vals, 0, (size_t)opt_.max_chunk);
  fake_cpu_experts(store_, model_.cfg(), M);
  if (fake_env("FAKE_PLACE", 0) && store_.n_slots() > 0)  // create resident experts (resident keys before sleep -> compared after wake; default off)
    for (int m = 0; m < std::min(M, 4); ++m) {
      const int l = (int)((uint32_t)ids[m] % (uint32_t)model_.cfg().n_layers), e = (int)((uint32_t)ids[m] / 7u % (uint32_t)model_.cfg().n_routed);
      store_.observe(l, e, 1.f + (float)(ids[m] % 5));
      store_.place(l, e, st_);
    }
  fake_write(st_, seq, model_.cfg().window, vals);
  seq.engram_history.insert(seq.engram_history.end(), vals->begin(), vals->end());
  seq.tokens.insert(seq.tokens.end(), ids, ids + M);
  seq.pos += M;
  if (const char* f = getenv("FAKE_FAIL_TOKEN"))  // host exception AFTER state advanced (half-written forward)
    for (int m = 0; m < M; ++m) if (ids[m] == atoi(f)) { CUDA_CHECK(cudaStreamSynchronize(st_)); throw std::runtime_error("fake injected forward failure"); }
  {
    const int ms = M < opt_.prefill_threshold && getenv("FAKE_PREFILL_SHORT_MS") ? fake_env("FAKE_PREFILL_SHORT_MS", 0) : fake_env("FAKE_PREFILL_MS", 0);
    fake_layers(this, ly_ok, ms, *vals, (size_t)opt_.max_chunk, [this] { layer_yield_point(); });  // FAKE_LAYERS 1 = one sleep (no per-layer yield)
  }
  CUDA_CHECK(cudaStreamSynchronize(st_));
  const int32_t next = fake_next(seq, model_.cfg().window);
  if (logits_out) { logits_out->assign(model_.cfg().vocab, 0.f); (*logits_out)[next] = 1.f; }
  if (mtp_on_ && !verify_) { seq.mtp_hidden_valid = true; seq.mtp_pos = seq.pos - 1; }  // runtime.cpp forward: mtp_sync_seq (not inside verify)
  return next;
}

// Several sequences' text prefill chunks in one call. Contract of the production forward_multi (runtime.cpp):
// one part per sequence, sum of slots_for(M) <= prefill_slots(), more than one part only with HIVE_BATCH_PREFILL, no images.
// Every part advances like fake forward; FAKE_FAIL_TOKEN in any part throws after ALL parts advanced (all seqs broken, as in production).
void Runtime::forward_multi(std::vector<PrefillPart>& parts, ForwardStats* stats) {
  (void)stats;
  fake_awake_check(asleep_);
  int slots = 0, rows = 0;
  for (auto& p : parts) {
    if (!fake_env("FAKE_SKIP_SNAPSHOT_FENCE", 0))
      if (p.seq->snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, p.seq->snapshot_fence->event, 0));  // runtime.cpp forward_multi preamble
    for (auto& q : parts) if (&q != &p && q.seq == p.seq) { fprintf(stderr, "fake: forward_multi one part per sequence\n"); abort(); }
    HIVE_CHECK(p.M >= 1 && p.seq->pos + p.M <= p.seq->cap, "forward_multi part");
    slots += slots_for(p.M); rows += p.M;
    p.next = -1; p.tail_rows = 0;
  }
  if (parts.empty() || slots > prefill_slots() || (parts.size() > 1 && !batch_prefill_)) {
    fprintf(stderr, "fake: forward_multi contract (parts %zu slots %d > %d or batch off)\n", parts.size(), slots, prefill_slots()); abort();
  }
  fprintf(stderr, "[fake] forward_multi parts %zu rows %d slots %d/%d\n", parts.size(), rows, slots, prefill_slots());
  struct BrokenGuard { std::vector<PrefillPart>* ps; int ex; ~BrokenGuard() { if (std::uncaught_exceptions() > ex) for (auto& p : *ps) p.seq->broken = true; } } guard{&parts, std::uncaught_exceptions()};
  fake_cpu_experts(store_, model_.cfg(), rows);
  // forward_multi is always a prefill (production ly_ok) — slot 0 = the first part's first max_chunk rows
  const bool ly_ok = ly_.on() && ly_.depth < ly_.h.max_depth;
  if (ly_ok) { std::vector<std::pair<Seq*, int>> own; for (auto& p : parts) own.push_back({p.seq, p.M}); ly_begin(std::move(own)); }
  std::vector<uint64_t> vals0(parts[0].M);
  for (int m = 0; m < parts[0].M; ++m) vals0[m] = content_at(nullptr, parts[0].ids, m);
  fake_work_put(vals0, 0, (size_t)opt_.max_chunk);
  bool fail = false;
  for (auto& p : parts) {
    auto vals = std::make_shared<std::vector<uint64_t>>(p.M);
    for (int m = 0; m < p.M; ++m) (*vals)[m] = content_at(nullptr, p.ids, m);
    fake_write(st_, *p.seq, model_.cfg().window, vals);
    p.seq->engram_history.insert(p.seq->engram_history.end(), vals->begin(), vals->end());
    p.seq->tokens.insert(p.seq->tokens.end(), p.ids, p.ids + p.M);
    p.seq->pos += p.M;
    if (const char* f = getenv("FAKE_FAIL_TOKEN")) for (int m = 0; m < p.M; ++m) if (p.ids[m] == atoi(f)) fail = true;
  }
  if (fail) { CUDA_CHECK(cudaStreamSynchronize(st_)); throw std::runtime_error("fake injected forward failure"); }
  fake_layers(this, ly_ok, fake_env("FAKE_PREFILL_MS", 0), vals0, (size_t)opt_.max_chunk, [this] { layer_yield_point(); });  // one layer-first pass for all parts (FAKE_LAYERS)
  CUDA_CHECK(cudaStreamSynchronize(st_));
  for (auto& p : parts) {
    p.next = fake_next(*p.seq, model_.cfg().window);
    if (mtp_on_) { p.seq->mtp_hidden_valid = true; p.seq->mtp_pos = p.seq->pos - 1; }  // runtime.cpp forward_multi: mtp_sync_seq
    if (p.logits_out) { p.logits_out->assign(model_.cfg().vocab, 0.f); (*p.logits_out)[p.next] = 1.f; }
  }
}

void Runtime::forward_batch(std::vector<Seq*>& seqs, const int32_t* ids, std::vector<int32_t>& next,
                            std::vector<std::vector<float>>* logits_out, ForwardStats* stats) {
  fake_awake_check(asleep_);
  if (!fake_env("FAKE_SKIP_SNAPSHOT_FENCE", 0))
    for (auto* seq : seqs) if (seq->snapshot_fence) CUDA_CHECK(cudaStreamWaitEvent(st_, seq->snapshot_fence->event, 0));  // runtime.cpp
  const int M = (int)seqs.size();
  HIVE_CHECK(M >= 1 && M <= opt_.max_batch, "batch size");
  struct BatchGuard { std::vector<Seq*>* ss; int ex; ~BatchGuard() { if (std::uncaught_exceptions() > ex) for (Seq* q : *ss) q->broken = true; } } guard{&seqs, std::uncaught_exceptions()};
  fake_cpu_experts(store_, model_.cfg(), M);
  {  // a decode step uses Work rows [0, floor) (production: rows ≤ max(window, max_batch, dspark_block+1))
    std::vector<uint64_t> dv((size_t)std::min(ly_.floor, opt_.max_chunk));
    for (size_t i = 0; i < dv.size(); ++i) dv[i] = 0xdec0de00ull + i + (uint64_t)seqs[0]->pos * 131;
    fake_work_put(dv, 7, (size_t)opt_.max_chunk);
  }
  const int fail_pos = fake_env("FAKE_FAIL_DECODE_POS", -1);
  for (int m = 0; m < M; ++m) {
    auto vals = std::make_shared<std::vector<uint64_t>>(1, content_at(nullptr, ids, m));
    fake_write(st_, *seqs[m], model_.cfg().window, vals);
    seqs[m]->engram_history.push_back((*vals)[0]);
    // half-advanced state (history grew, pos did not), like an exception between layers
    if (seqs[m]->pos == fail_pos) { CUDA_CHECK(cudaStreamSynchronize(st_)); throw std::runtime_error("fake injected decode failure"); }
    seqs[m]->tokens.push_back(ids[m]); seqs[m]->pos += 1;
  }
  if (int ms = fake_env("FAKE_STEP_MS", 0)) std::this_thread::sleep_for(FakeMs(ms));
  CUDA_CHECK(cudaStreamSynchronize(st_));
  next.assign(M, -1);
  if (logits_out) logits_out->assign(M, {});
  for (int m = 0; m < M; ++m) {
    next[m] = fake_next(*seqs[m], model_.cfg().window);
    if (mtp_on_) { seqs[m]->mtp_hidden_valid = true; seqs[m]->mtp_pos = seqs[m]->pos - 1; }  // runtime.cpp forward_batch: mtp_sync_batch
    if (logits_out) { (*logits_out)[m].assign(model_.cfg().vocab, 0.f); (*logits_out)[m][next[m]] = 1.f; }
  }
  if (stats) { stats->row_hit.assign(M, 0); stats->row_cpu.assign(M, 0); stats->row_dma.assign(M, 0); }
}
}  // namespace hive
