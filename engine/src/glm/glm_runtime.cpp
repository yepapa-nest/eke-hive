// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/glm/hived_family.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

#include "hive/clock.h"
#include "hive/model_kernels.h"

namespace hive::glm::fam {

namespace {
template <class T> T* pinned(size_t n) { void* p = nullptr; CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(n, 1) * sizeof(T))); return (T*)p; }
template <class T> T* pinned_mapped(size_t n, T** dev) {
  void* p = nullptr; CUDA_CHECK(cudaHostAlloc(&p, std::max<size_t>(n, 1) * sizeof(T), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)dev, p, 0)); return (T*)p;
}
}  // namespace

// ---- Model / ExpertStore ----------------------------------------------------------------------------------------------------
Model::Model(const std::string& ckpt, int /*max_layer*/, int64_t /*max_ctx*/)
    : m_(std::make_unique<GlmModel>(ckpt, !(getenv("HIVE_GLM_MTP_K") && strcmp(getenv("HIVE_GLM_MTP_K"), "0") == 0))) {
  const GlmConfig& g = m_->cfg();
  cfg_.vocab = g.vocab; cfg_.n_layers = g.n_layers; cfg_.glm = m_.get();
  // hived's image admission: at most vision_max_tokens · downsample² patches per image span (the processor's max_image_tokens 8000;
  //   the server sends each pair of video frames as its own span, within the same limit)
  cfg_.vision_max_tokens = 8000; cfg_.vision_downsample = g.vision_merge;
  ckpt_ref_ = CkptRef(m_.get());
}

ExpertStore::ExpertStore(const Config& c, int, size_t cache_bytes, int cpu_threads, int, bool defer_slots) : threads_(std::max(1, cpu_threads)) {
  HIVE_CHECK(c.glm, "GLM store without a model");
  // load threads: the copy from the checkpoint mmap is memory/IO bound — use at least 16 even if the CPU expert pool is smaller
  x_ = std::make_unique<GlmExperts>(*c.glm, 0, threads_, 8);
  lay_.total = x_->layout().total;
  if (!defer_slots && cache_bytes > 0) x_->alloc_cache(cache_bytes);
}
void ExpertStore::ensure_loaded() {
  if (loaded_) return;
  x_->load_all(std::max(16, threads_));
  loaded_ = true;
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
  opt_.max_batch = std::max(1, std::min(opt_.max_batch, 8));  // decode kernels (nvfp4/fp8 gemv, moe_decode grouping) take M ≤ 8
  store_.ensure_loaded();
  eng_ = std::make_unique<GlmEngine>(model_.glm(), store_.experts(), std::max(opt_.max_chunk, 64));
  rows_cap_ = std::max(opt_.max_batch, eng_->mtp_on() ? eng_->mtp_k() + 1 : 1);
  // HIVE_MTP_BATCH (same switch as DeepSeek): hived verifies the drafts of several sequences in one step (forward_verify_batch) —
  //   KDA snapshots for max_batch parts, rows up to 8 (the decode kernels' M limit)
  {
    const char* mb = getenv("HIVE_MTP_BATCH");
    mtp_batch_ = mb && *mb && strcmp(mb, "0") != 0 && eng_->mtp_on() && opt_.max_batch >= 2;
    if (mtp_batch_) { eng_->enable_batch_verify(opt_.max_batch); rows_cap_ = std::max(rows_cap_, 8); }
  }
  const int NC = std::max(1, opt_.sampler_cands);
  cand_it_h_ = pinned_mapped<float>(rows_cap_, &cand_it_d_);
  for (int i = 0; i < rows_cap_; ++i) cand_it_h_[i] = 1.f;
  cand_idx_h_ = pinned<int32_t>((size_t)rows_cap_ * NC); cand_val_h_ = pinned<float>((size_t)rows_cap_ * NC);
  cand_max_h_ = pinned<float>(rows_cap_); cand_sum_h_ = pinned<float>(rows_cap_); next_h_ = pinned<int32_t>(rows_cap_);
  cand_idx_.alloc((size_t)rows_cap_ * NC * 4); cand_val_.alloc((size_t)rows_cap_ * NC * 4);
  cand_max_.alloc((size_t)rows_cap_ * 4); cand_sum_.alloc((size_t)rows_cap_ * 4); next_d_.alloc((size_t)rows_cap_ * 4);
  fprintf(stderr, "[glm] runtime: family %s · max batch %d · chunk %d · context %lld · sampler candidates %d\n", kFamily, opt_.max_batch,
          std::max(opt_.max_chunk, 64), (long long)opt_.max_ctx, opt_.sampler_cands);
}

Runtime::~Runtime() {
  for (void* p : {(void*)cand_it_h_, (void*)cand_idx_h_, (void*)cand_val_h_, (void*)cand_max_h_, (void*)cand_sum_h_, (void*)next_h_})
    if (p) cudaFreeHost(p);
}

std::unique_ptr<Seq> Runtime::new_seq() const {
  auto s = std::make_unique<Seq>();
  eng_->init_seq(*s, opt_.max_ctx);
  return s;
}

void Runtime::reset_seq(Seq& s) const {
  eng_->reset_seq(s);
  s.generation = Seq::next_uid();
  s.engram_history.clear();
  s.mtp_hidden_valid = false;
}

// Image layout: per KDA layer conv, state (whole) · per DSA layer c, kI, gs rows [0, pos) and pooled rows [0, pos/pool).
void Runtime::save_image(const Seq& s, SeqImage& img, const SeqImage*) const {
  const GlmConfig& c = model_.glm().cfg();
  CUDA_CHECK(cudaStreamSynchronize(eng_->stream()));
  img.pos = s.pos; img.generation = s.generation; img.tokens = s.tokens; img.engram_history.clear(); img.mtp_hidden_valid = s.mtp_hidden_valid;
  img.h_last_valid = s.h_last_valid; img.mtp_written = s.mtp_written;
  img.bufs.clear();
  auto grab = [&](const auto& b, size_t bytes) {
    img.bufs.emplace_back(bytes);
    if (bytes) CUDA_CHECK(cudaMemcpy(img.bufs.back().data(), b.p, bytes, cudaMemcpyDeviceToHost));
  };
  for (size_t i = 0; i < s.conv.size(); ++i) { grab(s.conv[i], s.conv[i].n); grab(s.state[i], s.state[i].n); }
  const size_t P = (size_t)s.pos;
  for (size_t i = 0; i < s.c.size(); ++i) {
    grab(s.c[i], P * c.kv_lora * 2); grab(s.kI[i], P * c.index_dim * 2); grab(s.gs[i], P * c.index_dim * 2);
    grab(s.pooled[i], (P / c.index_kpool) * c.index_dim * 2);
  }
  if (s.h_last.p) {  // MTP cache (entries < mtp_written are exact; copy up to pos) + last hidden
    grab(s.mtp_c, P * c.kv_lora * 2); grab(s.mtp_kI, P * c.index_dim * 2); grab(s.mtp_gs, P * c.index_dim * 2);
    grab(s.mtp_pooled, (P / c.index_kpool) * c.index_dim * 2); grab(s.h_last, s.h_last.n);
  }
}

void Runtime::load_image(Seq& s, const SeqImage& img) const {
  HIVE_CHECK(img.pos <= s.cap, "GLM image longer than the sequence capacity");
  eng_->kv_reserve(s, std::min<int64_t>(s.cap, img.pos + GlmEngine::kKvSlack));
  size_t k = 0;
  auto put = [&](auto& b) {
    HIVE_CHECK(k < img.bufs.size() && img.bufs[k].size() <= b.n, "GLM image layout mismatch");
    if (!img.bufs[k].empty()) CUDA_CHECK(cudaMemcpy(b.p, img.bufs[k].data(), img.bufs[k].size(), cudaMemcpyHostToDevice));
    ++k;
  };
  for (size_t i = 0; i < s.conv.size(); ++i) { put(s.conv[i]); put(s.state[i]); }
  for (size_t i = 0; i < s.c.size(); ++i) { put(s.c[i]); put(s.kI[i]); put(s.gs[i]); put(s.pooled[i]); }
  if (s.h_last.p) { put(s.mtp_c); put(s.mtp_kI); put(s.mtp_gs); put(s.mtp_pooled); put(s.h_last); }
  HIVE_CHECK(k == img.bufs.size(), "GLM image layout mismatch (count)");
  s.pos = img.pos; s.tokens = img.tokens; s.generation = img.generation; s.broken = false;
  s.engram_history.clear();
  s.h_last_valid = img.h_last_valid && s.h_last.p; s.mtp_written = img.mtp_written; s.mtp_hidden_valid = s.h_last_valid && eng_->mtp_on();
}

void Runtime::quiesce_after_host_error() { cudaStreamSynchronize(eng_->stream()); }

void Runtime::ensure_vision() {
  if (vision_) return;
  const double t0 = hive::mono_ms();
  eng_->make_room((size_t)1200 << 20);  // the weights are ~1.1 GB (BF16)
  vision_ = std::make_unique<GlmVision>(model_.glm());
  fprintf(stderr, "[glm] vision encoder loaded: %.0f MiB in %.0f ms\n", vision_->weight_bytes() / 1048576.0, hive::mono_ms() - t0);
}

// HIVE_GLM_PROF (measurement only): per call kind (decode by rows, verify by rows, draft), the wall time of hived's calls, the engine's phase
//   times (GlmForwardStats — with HIVE_GLM_PROF=2 the GPU event intervals, which include the gaps where the stream waits for the host) and the
//   expert-cache counters, accumulated and printed every 400 calls of a kind. Engine thread only.
namespace {
struct ProfKind {
  long n = 0;
  double wall = 0;
  GlmForwardStats e;
  GlmCacheStats x;
};
std::map<std::string, ProfKind>& prof_kinds() { static std::map<std::string, ProfKind> m; return m; }
bool prof_on() { static const bool on = getenv("HIVE_GLM_PROF") && atoi(getenv("HIVE_GLM_PROF")) > 0; return on; }
void prof_add(const std::string& kind, double wall, const GlmForwardStats& e0, const GlmForwardStats& e1, const GlmCacheStats& x0, const GlmCacheStats& x1) {
  ProfKind& k = prof_kinds()[kind];
  ++k.n; k.wall += wall;
  auto de = [&](double GlmForwardStats::*f) { k.e.*f += e1.*f - e0.*f; };
  for (auto f : {&GlmForwardStats::ms_hc, &GlmForwardStats::ms_kda, &GlmForwardStats::ms_dsa, &GlmForwardStats::ms_dense, &GlmForwardStats::ms_router,
                 &GlmForwardStats::ms_shared, &GlmForwardStats::ms_experts, &GlmForwardStats::ms_head, &GlmForwardStats::ms_embed, &GlmForwardStats::ms_kda_proj,
                 &GlmForwardStats::ms_kda_core, &GlmForwardStats::ms_predict})
    de(f);
  auto dx = [&](uint64_t GlmCacheStats::*f) { k.x.*f += x1.*f - x0.*f; };
  for (auto f : {&GlmCacheStats::routed, &GlmCacheStats::hit, &GlmCacheStats::cpu, &GlmCacheStats::prefetched, &GlmCacheStats::promoted, &GlmCacheStats::cpu_layers, &GlmCacheStats::skipped, &GlmCacheStats::deferred,
                 &GlmCacheStats::cpu_jobs, &GlmCacheStats::cpu_bytes}) dx(f);
  auto dt = [&](double GlmCacheStats::*f) { k.x.*f += x1.*f - x0.*f; };
  for (auto f : {&GlmCacheStats::ms_layer, &GlmCacheStats::ms_cpu, &GlmCacheStats::ms_wait_x, &GlmCacheStats::ms_p1, &GlmCacheStats::ms_p2, &GlmCacheStats::ms_busy1, &GlmCacheStats::ms_busy2}) dt(f);
  if (k.n % 400) return;
  const double n = (double)k.n;
  const GlmForwardStats& E = k.e;
  fprintf(stderr, "[glm-prof] %s · calls %ld · wall %.2f ms/call · hit %.1f %% of %.1f routed/call · cpu %.1f/call · skipped %.2f/call · deferred %.2f/call · prefetched %.2f · promoted %.2f/call\n",
          kind.c_str(), k.n, k.wall / n, k.x.routed ? 100.0 * k.x.hit / k.x.routed : 0.0, k.x.routed / n, k.x.cpu / n, k.x.skipped / n, k.x.deferred / n, k.x.prefetched / n, k.x.promoted / n);
  fprintf(stderr, "[glm-prof] %s · per call ms: embed %.2f · hc %.2f · kda %.2f (proj %.2f · core %.2f) · dsa %.2f · dense %.2f · router %.2f · shared %.2f · "
                  "experts %.2f · head %.2f · predict %.2f · (host: CPU experts %.2f [gate/up %.2f · down %.2f · worker busy %.0f / %.0f %%] over %.1f layers · %.1f experts · %.1f GB/s · wait for x %.2f)\n",
          kind.c_str(), E.ms_embed / n, E.ms_hc / n, (E.ms_kda + E.ms_kda_proj + E.ms_kda_core) / n, E.ms_kda_proj / n, E.ms_kda_core / n, E.ms_dsa / n,
          E.ms_dense / n, E.ms_router / n, E.ms_shared / n, E.ms_experts / n, E.ms_head / n, E.ms_predict / n, k.x.ms_cpu / n, k.x.ms_p1 / n, k.x.ms_p2 / n,
          k.x.ms_p1 > 0 ? 100.0 * k.x.ms_busy1 / (k.x.ms_p1 * std::max(1, x1.workers)) : 0.0, k.x.ms_p2 > 0 ? 100.0 * k.x.ms_busy2 / (k.x.ms_p2 * std::max(1, x1.workers)) : 0.0,
          k.x.cpu_layers / n,
          k.x.cpu_jobs / n, k.x.ms_cpu > 0 ? k.x.cpu_bytes / (k.x.ms_cpu * 1e6) : 0.0, k.x.ms_wait_x / n);
}
}  // namespace

void Runtime::stats_delta(ForwardStats* st, const GlmCacheStats& b) const {
  if (!st) return;
  const GlmCacheStats& a = store_.experts().stats();
  st->n_routed += (int)(a.routed - b.routed); st->n_hit += (int)(a.hit - b.hit); st->n_cpu += (int)(a.cpu - b.cpu);
  st->n_streamed += (int)(a.streamed - b.streamed); st->n_promoted += (int)(a.promoted - b.promoted);
}

void Runtime::cands_dev(int M) {
  const int NC = opt_.sampler_cands, V = model_.glm().cfg().vocab;
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
  HIVE_CHECK(M >= 1, "GLM forward: no tokens");
  // images: encode each span, then let the prefill take those rows' embeddings from the encoder output
  std::vector<DevBuf> vis_out;
  const bool has_images = images && !images->empty();
  if (has_images) {
    ensure_vision();
    const size_t H = model_.glm().cfg().hidden;
    std::vector<GlmEngine::EmbedOverride> ov;
    vis_out.reserve(images->size());
    for (const ImageInput& im : *images) {
      const int n = im.n_vit_h * im.n_vit_w;
      HIVE_CHECK(n % 4 == 0 && (int)im.types.size() == n / 4 && im.start >= 0 && im.start + n / 4 <= M, "GLM: image span does not match its patches");
      eng_->make_room(GlmVision::scratch_bytes(n) + (size_t)(n / 4) * H * 2);
      vis_out.emplace_back((size_t)(n / 4) * H * 2);
      vision_->encode(im.patches_dev, im.n_vit_h, im.n_vit_w, vis_out.back().as<bf16>(), eng_->stream());
      ov.push_back({im.start, n / 4, vis_out.back().as<bf16>()});
    }
    eng_->set_embed_override(std::move(ov));
  }
  struct ClearOv {
    GlmEngine& e; bool on;
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
  const GlmConfig& c = model_.glm().cfg();
  const GlmCacheStats before = store_.experts().stats();
  const double t0 = hive::mono_ms();
  try {
    logits_h_.resize(c.vocab);
    eng_->prefill(seq, ids, M, logits_h_.data());
  } catch (...) { seq.broken = true; throw; }
  if (has_images) {  // the encoder outputs are no longer needed: free them and let the expert cache take that memory back
    eng_->set_embed_override({});
    vis_out.clear();
    eng_->give_back_room();
  }
  since_prefill_ = 0;
  cand_it_h_[0] = 1.f;
  cands_dev(1);
  if (logits_out) *logits_out = logits_h_;
  seq.mtp_hidden_valid = seq.h_last_valid && eng_->mtp_on();
  stats_delta(stats, before);
  if (stats) stats->ms_total += hive::mono_ms() - t0;
  return next_h_[0];
}

void Runtime::forward_batch(std::vector<Seq*>& seqs, const int32_t* ids, std::vector<int32_t>& next, std::vector<std::vector<float>>* logits_out,
                            ForwardStats* stats) {
  const int M = (int)seqs.size();
  HIVE_CHECK(M >= 1 && M <= rows_cap_, "GLM decode batch size");
  const int V = model_.glm().cfg().vocab;
  const GlmCacheStats before = store_.experts().stats();
  const double t0 = hive::mono_ms();
  std::vector<GlmSeq*> gs(M);
  for (int m = 0; m < M; ++m) gs[m] = seqs[m];
  logits_h_.resize((size_t)M * V);
  store_.experts().reset_row_stats();
  const GlmForwardStats e0 = eng_->stats();
  try {
    eng_->decode(gs.data(), ids, M, logits_h_.data());
  } catch (...) { for (Seq* s : seqs) s->broken = true; throw; }
  if (prof_on()) prof_add("decode M=" + std::to_string(M), hive::mono_ms() - t0, e0, eng_->stats(), before, store_.experts().stats());
  if (since_prefill_ < (1 << 30)) ++since_prefill_;
  cands_dev(M);
  for (Seq* q : seqs) q->mtp_hidden_valid = q->h_last_valid && eng_->mtp_on();
  next.assign(next_h_, next_h_ + M);
  if (logits_out) {
    logits_out->assign(M, std::vector<float>());
    for (int m = 0; m < M; ++m) (*logits_out)[m].assign(logits_h_.begin() + (size_t)m * V, logits_h_.begin() + (size_t)(m + 1) * V);
  }
  stats_delta(stats, before);
  if (stats) {
    stats->row_hit.assign(M, 0); stats->row_cpu.assign(M, 0); stats->row_dma.assign(M, 0);
    for (int m = 0; m < M; ++m) { stats->row_hit[m] = store_.experts().row_hit(m); stats->row_cpu[m] = store_.experts().row_cpu(m); }
    stats->ms_total += hive::mono_ms() - t0;
  }
}

void Runtime::mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats* stats) {
  const GlmCacheStats before = store_.experts().stats();
  const int k = eng_->mtp_k();
  drafts.assign(k, 0); conf.assign(k, 0.f);
  const GlmForwardStats e0 = eng_->stats();
  const double t0 = hive::mono_ms();
  const int n = eng_->mtp_draft(seq, tok, k, drafts.data(), conf.data());
  if (prof_on()) prof_add("draft k=" + std::to_string(k), hive::mono_ms() - t0, e0, eng_->stats(), before, store_.experts().stats());
  drafts.resize(n); conf.resize(n);
  // hived's MTP gate buckets a logit-scale confidence (DeepSeek: its confidence head); GLM gives the draft's softmax probability p —
  //   pass log(p / (1 − p)), monotone in p, so the gate's online acceptance table per bucket applies (HIVE_MTP_GATE2 learns it)
  for (float& c : conf) { const float p = std::min(std::max(c, 1e-6f), 1.f - 1e-6f); c = std::log(p / (1.f - p)); }
  stats_delta(stats, before);
}

void Runtime::forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats) {
  HIVE_CHECK(M >= 1 && M <= rows_cap_, "GLM verify rows");
  const int V = model_.glm().cfg().vocab;
  const GlmCacheStats before = store_.experts().stats();
  const double t0 = hive::mono_ms();
  logits_rows.resize((size_t)M * V);
  const GlmForwardStats e0 = eng_->stats();
  try {
    eng_->verify(seq, ids, M, logits_rows.data());
  } catch (...) { seq.broken = true; throw; }
  if (prof_on()) prof_add(std::string(since_prefill_ < kWarmSteps ? "verify(warm) M=" : "verify M=") + std::to_string(M), hive::mono_ms() - t0, e0, eng_->stats(), before,
                          store_.experts().stats());
  if (since_prefill_ < kWarmSteps) ++warm_samples_;  // cache still warming: not a representative step cost (see graph_captures)
  if (since_prefill_ < (1 << 30)) ++since_prefill_;
  cands_dev(M);  // candidates of every verify row (row_inv_temp set by the caller)
  stats_delta(stats, before);
  if (stats) stats->ms_total += hive::mono_ms() - t0;
}

void Runtime::rollback(Seq& seq, int n_keep) {
  eng_->rollback(seq, n_keep);
  seq.mtp_hidden_valid = seq.h_last_valid && eng_->mtp_on();
}

void Runtime::forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats) {
  const int S = (int)parts.size();
  HIVE_CHECK(mtp_batch_ && S >= 1 && S <= eng_->verify_parts_cap(), "GLM batched verify: parts");
  const int V = model_.glm().cfg().vocab;
  std::vector<GlmSeq*> seqs(S);
  std::vector<int> Ms(S);
  std::vector<int32_t> ids;
  for (int p = 0; p < S; ++p) { seqs[p] = parts[p].seq; Ms[p] = parts[p].M; ids.insert(ids.end(), parts[p].ids, parts[p].ids + parts[p].M); }
  const int R = (int)ids.size();
  HIVE_CHECK(R >= 1 && R <= rows_cap_, "GLM batched verify rows");
  const GlmCacheStats before = store_.experts().stats();
  const GlmForwardStats e0 = eng_->stats();
  const double t0 = hive::mono_ms();
  logits_rows.resize((size_t)R * V);
  try {
    eng_->verify_batch(seqs.data(), Ms.data(), S, ids.data(), logits_rows.data());
  } catch (...) { for (auto& p : parts) p.seq->broken = true; throw; }
  if (prof_on()) prof_add("verify-batch S=" + std::to_string(S) + " R=" + std::to_string(R), hive::mono_ms() - t0, e0, eng_->stats(), before,
                          store_.experts().stats());
  if (since_prefill_ < kWarmSteps) ++warm_samples_;
  if (since_prefill_ < (1 << 30)) ++since_prefill_;
  cands_dev(R);
  stats_delta(stats, before);
  if (stats) stats->ms_total += hive::mono_ms() - t0;
  vb_seqs_ = seqs;
}

void Runtime::rollback_batch(const std::vector<int>& n_keep) {
  HIVE_CHECK(n_keep.size() == vb_seqs_.size(), "GLM batched rollback: parts");
  eng_->rollback_batch(n_keep.data(), (int)n_keep.size());
  for (GlmSeq* s : vb_seqs_) static_cast<Seq*>(s)->mtp_hidden_valid = s->h_last_valid && eng_->mtp_on();
  vb_seqs_.clear();
}

void Runtime::forward_multi(std::vector<PrefillPart>& parts, ForwardStats* stats) {
  for (auto& p : parts) p.next = forward(*p.seq, p.ids, p.M, nullptr, p.logits_out, stats, p.upper_needed);
}

}  // namespace hive::glm::fam
