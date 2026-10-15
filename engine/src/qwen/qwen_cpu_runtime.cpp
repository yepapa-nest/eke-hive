// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hived surface of the CPU-only Qwen3.8-Flash-Next family — see hive/qwen/hived_family_cpu.h.
#include "hive/qwen/hived_family_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "hive/clock.h"
#include "hive/cpu/cpu_backend.h"

namespace hive::qwen::cpufam {

// HIVE_QWEN_MTP_K: "0" = do not load the MTP head; unset, "" or any other value = the default (loaded).
static bool want_mtp() { return !(getenv("HIVE_QWEN_MTP_K") && strcmp(getenv("HIVE_QWEN_MTP_K"), "0") == 0); }

Model::Model(const std::string& ckpt, int, int64_t) : e_(std::make_unique<cpu::QwenCpuEngine>(ckpt, want_mtp())) {
  const QwenConfig& q = e_->cfg();
  cfg_.vocab = q.vocab; cfg_.n_layers = q.n_layers; cfg_.eng = e_.get();
  cfg_.vision_max_tokens = 8000; cfg_.vision_downsample = q.vision_merge;
}

Runtime::Runtime(Model& model, ExpertStore& store, const void*, const RuntimeOptions& opt) : model_(model), store_(store), opt_(opt) {
  opt_.max_batch = std::max(1, std::min(opt_.max_batch, 8));
  const char* mb = getenv("HIVE_MTP_BATCH");
  mtp_batch_ = mb && *mb && strcmp(mb, "0") != 0 && model_.eng().mtp_on() && opt_.max_batch >= 2;
  const int rows = 8, NC = std::max(1, opt_.sampler_cands), V = model_.eng().cfg().vocab;
  logits_.resize((size_t)rows * V); inv_temp_.assign(rows, 1.f); cand_val_.resize((size_t)rows * NC); cand_idx_.resize((size_t)rows * NC);
  rmax_.resize(rows); rsum_.resize(rows); next_.resize(rows);
  fprintf(stderr, "[qwen-cpu] runtime: family %s · CPU only (%d pool threads on %d NUMA nodes) · max batch %d · chunk %d · context %lld\n", kFamily,
          hive::cpu::Pool::get().threads(), hive::cpu::Pool::get().nodes(), opt_.max_batch, std::max(opt_.max_chunk, 64), (long long)opt_.max_ctx);
}

std::unique_ptr<Seq> Runtime::new_seq() const {
  auto s = std::make_unique<Seq>();
  model_.eng().init_seq(*s, opt_.max_ctx);
  return s;
}

void Runtime::reset_seq(Seq& s) const {
  model_.eng().reset_seq(s);
  s.generation = Seq::next_uid();
  s.engram_history.clear();
  s.mtp_hidden_valid = false;
}

void Runtime::save_image(const Seq& s, SeqImage& img, const SeqImage*) const {
  SeqImage out;
  out.pos = s.pos; out.generation = s.generation; out.tokens = s.tokens; out.mtp_hidden_valid = s.mtp_hidden_valid;
  for (auto& b : s.save_bytes()) {
    HostImageBuffer hb;
    std::shared_ptr<uint8_t> p(new uint8_t[std::max<size_t>(b.size(), 1)], std::default_delete<uint8_t[]>());
    if (!b.empty()) std::memcpy(p.get(), b.data(), b.size());
    hb.segments.push_back({std::move(p), b.size()});
    out.bufs.push_back(std::move(hb));
  }
  img = std::move(out);
}

void Runtime::load_image(Seq& s, const SeqImage& img) const {
  std::vector<std::vector<uint8_t>> b;
  for (const auto& hb : img.bufs) {
    std::vector<uint8_t> v(hb.size());
    size_t off = 0;
    for (const auto& sg : hb.segments) { std::memcpy(v.data() + off, sg.data.get(), sg.n); off += sg.n; }
    b.push_back(std::move(v));
  }
  s.load_bytes(b);
  HIVE_CHECK(s.pos == img.pos, "qwen-cpu image position mismatch");
  s.tokens = img.tokens; s.generation = img.generation; s.broken = false; s.engram_history.clear();
  s.mtp_hidden_valid = s.h_last_valid && model_.eng().mtp_on();
}

// Sampler candidates on the host: argmax, max / Σ exp((l − max) · inv_temp) and the NC largest logits (any order — hived sorts what it needs).
void Runtime::cands(const float* lg, int M) {
  const int V = model_.eng().cfg().vocab, NC = opt_.sampler_cands;
  hive::cpu::Pool::get().run(M, [&](int m) {
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
  });
}

int32_t Runtime::forward(Seq& seq, const int32_t* ids, int M, const std::vector<ImageInput>* images, std::vector<float>* logits_out, ForwardStats* stats,
                         bool) {
  HIVE_CHECK(M >= 1, "qwen-cpu forward: no tokens");
  cpu::QwenCpuEngine& e = model_.eng();
  if (images && !images->empty()) {
    std::vector<cpu::QwenCpuEngine::EmbedOverride> ov;
    for (const ImageInput& im : *images) {
      const int n = im.n_vit_h * im.n_vit_w;
      HIVE_CHECK(im.n_vit_h % 2 == 0 && im.n_vit_w % 2 == 0 && (int)im.types.size() == n / 4 && im.start >= 0 && im.start + n / 4 <= M,
                 "qwen-cpu: image span does not match its patches");
      cpu::QwenCpuEngine::EmbedOverride o{im.start, n / 4, {}, im.n_vit_h / 2, im.n_vit_w / 2};
      e.encode_image(reinterpret_cast<const uint16_t*>(im.patches_dev), im.n_vit_h, im.n_vit_w, o.emb);
      ov.push_back(std::move(o));
    }
    e.set_embed_override(std::move(ov));
  }
  struct ClearOv { cpu::QwenCpuEngine& e; bool on; ~ClearOv() { if (on) e.set_embed_override({}); } } clear{e, images && !images->empty()};
  if (M == 1 && !(images && !images->empty())) {
    std::vector<Seq*> s{&seq};
    std::vector<int32_t> next;
    std::vector<std::vector<float>> lg;
    forward_batch(s, ids, next, logits_out ? &lg : nullptr, stats);
    if (logits_out) *logits_out = std::move(lg[0]);
    return next[0];
  }
  const double t0 = hive::mono_ms();
  const bool ly_ok = ly_.on() && ly_.depth < ly_.h.max_depth && M >= opt_.prefill_threshold;
  // read progress (layer_yield.h Progress — the API server's stage status): the layer hook is also installed when only it is on;
  //   the yield point still runs only for ly_ok
  const bool pg_ok = prog_.on() && ly_.depth == 0 && M >= opt_.prefill_threshold;
  const bool hook_on = ly_ok || pg_ok;
  struct LyHook { cpu::QwenCpuEngine& e; bool on; std::function<void(int)> prev; ~LyHook() { if (on) e.layer_boundary = std::move(prev); } }
      lh{e, hook_on, hook_on ? e.layer_boundary : std::function<void(int)>{}};
  if (hook_on) {
    if (ly_ok && ly_.depth == 0) ly_.last = hive::mono_ms();
    e.layer_boundary = [this, ly_ok](int l) { ly_progress_ = (l + 1.0) / std::max(1, model_.eng().cfg().n_layers); progress_point(); if (ly_ok) layer_yield_point(); };
  }
  const int V = e.cfg().vocab;
  std::vector<float> lg(V);
  try { e.prefill(seq, ids, M, lg.data()); } catch (...) { seq.broken = true; throw; }
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
  HIVE_CHECK(M >= 1 && M <= 8, "qwen-cpu decode batch size");
  const double t0 = hive::mono_ms();
  std::vector<cpu::CpuSeq*> cs(seqs.begin(), seqs.end());
  try { model_.eng().decode(cs.data(), ids, M, logits_.data()); } catch (...) { for (Seq* s : seqs) s->broken = true; throw; }
  cands(logits_.data(), M);
  for (Seq* q : seqs) q->mtp_hidden_valid = q->h_last_valid && model_.eng().mtp_on();
  next.assign(next_.begin(), next_.begin() + M);
  if (logits_out) {
    logits_out->assign(M, std::vector<float>());
    for (int m = 0; m < M; ++m) (*logits_out)[m].assign(logits_.data() + (size_t)m * V, logits_.data() + (size_t)(m + 1) * V);
  }
  if (stats) {  // hived reads one entry per row (row_*.at(i)); no expert cache on the CPU path, so every count is 0
    stats->row_hit.assign(M, 0); stats->row_cpu.assign(M, 0); stats->row_dma.assign(M, 0);
    stats->ms_total += hive::mono_ms() - t0;
  }
}

void Runtime::mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats*) {
  cpu::QwenCpuEngine& e = model_.eng();
  const int k = e.mtp_k();
  drafts.assign(k, 0); conf.assign(k, 0.f);
  const int n = e.mtp_draft(seq, tok, k, drafts.data(), conf.data());
  drafts.resize(n); conf.resize(n);
  for (float& c : conf) { const float p = std::min(std::max(c, 1e-6f), 1.f - 1e-6f); c = std::log(p / (1.f - p)); }  // as the GPU family
}

void Runtime::forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats) {
  std::vector<VerifyPart> p{{&seq, ids, M}};
  forward_verify_batch(p, logits_rows, stats);
}

void Runtime::forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats) {
  const int S = (int)parts.size(), V = model_.eng().cfg().vocab;
  std::vector<cpu::CpuSeq*> seqs(S);
  std::vector<int> Ms(S);
  std::vector<int32_t> ids;
  for (int p = 0; p < S; ++p) { seqs[p] = parts[p].seq; Ms[p] = parts[p].M; ids.insert(ids.end(), parts[p].ids, parts[p].ids + parts[p].M); }
  const int R = (int)ids.size();
  HIVE_CHECK(R >= 1 && R <= 8, "qwen-cpu verify rows");
  const double t0 = hive::mono_ms();
  try { model_.eng().verify_batch(seqs.data(), Ms.data(), S, ids.data(), logits_.data()); } catch (...) { for (auto& p : parts) p.seq->broken = true; throw; }
  logits_rows.assign(logits_.begin(), logits_.begin() + (size_t)R * V);
  cands(logits_.data(), R);
  vb_seqs_.clear();
  for (auto& p : parts) vb_seqs_.push_back(p.seq);
  if (stats) {
    stats->row_hit.assign(R, 0); stats->row_cpu.assign(R, 0); stats->row_dma.assign(R, 0);
    stats->ms_total += hive::mono_ms() - t0;
  }
}

void Runtime::rollback_batch(const std::vector<int>& n_keep) {
  HIVE_CHECK(n_keep.size() == vb_seqs_.size(), "qwen-cpu rollback: parts");
  model_.eng().rollback_batch(n_keep.data(), (int)n_keep.size());
  for (Seq* s : vb_seqs_) s->mtp_hidden_valid = s->h_last_valid && model_.eng().mtp_on();
  vb_seqs_.clear();
}

void Runtime::layer_yield_point() {
  if (!ly_.on() || ly_.depth >= ly_.h.max_depth) return;
  ly_.floor = 8;
  ly_.cap = 8;   // the CPU engine's inner forwards use their own buffers: decode / verify rows only
  ly::yield_point(ly_, [&](int) { return true; }, [&](int) {}, [] { return hive::mono_ms(); });
}

}  // namespace hive::qwen::cpufam
