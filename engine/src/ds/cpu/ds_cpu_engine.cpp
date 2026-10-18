// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash CPU engine — the forward (hive/ds/ds_cpu.h). One code path for every kind of forward: a forward is a list of parts
//   (consecutive positions of one sequence): a prefill chunk = 1 part, a batched decode = M parts of one row, a DSpark verify = S parts.
//   The arithmetic and every bf16 rounding point follow oracle/dsv41_oracle.py (reference model.py / kernel.py):
//   · hidden stream h [rows, hc, dim] and every activation between ops hold bf16 values in fp32;
//   · fp8 / fp4 linears quantize their input with act_quant (32-blocks, e8m0) and round their fp32 GEMM output to bf16; plain bf16 linears
//     (compressor ratio 1, indexer weights / keys, ViT) round their output; fp32 linears (hc mixes, router, ratio-2 compressor, head) do not;
//   · attention keys are seen per row: the window = positions [p − 127, p] (this forward's own rows, else the sequence's ring), the compressed
//     keys = the top-k positions of the last index-source layer; the ring is written after the attention (no row sees a later row's key).
//   Decoder tail (as the GPU Runtime): a prefill chunk longer than EngineOptions::decoder_tail runs the last kv-source layer's compressor for
//   all rows, then layers ≥ it on the last decoder_tail rows only, whose window does not reach before the tail — exact for the last
//   128 rows (each layer can spoil at most 128 more rows from the front; 128 × 21 ≥ that), which are all a later chunk or decode reads.
#include <immintrin.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "ds_cpu_internal.h"
#include "hive/clock.h"
#include "hive/common.h"

namespace hive::ds::cpu {

using ::hive::cpu::Pool;

namespace {
struct Timer {
  double& acc; double t0;
  explicit Timer(double& a) : acc(a), t0(hive::mono_ms()) {}
  ~Timer() { acc += hive::mono_ms() - t0; }
};
inline float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float siluf(float x) { return x / (1.f + std::exp(-x)); }
void mm(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy) { ::hive::cpu::matmul(W, X, M, ldx, Y, ldy); }
}  // namespace

// ---- public surface ------------------------------------------------------------------------------------------------------------------------
DsCpuEngine::DsCpuEngine(const std::string& ckpt_dir, const EngineOptions& opt) : p_(std::make_unique<Impl>()) {
  p_->dir = ckpt_dir;
  p_->opt = opt;
  p_->pub = this;
  p_->load();
}
DsCpuEngine::~DsCpuEngine() = default;
const DsConfig& DsCpuEngine::cfg() const { return p_->c; }
int DsCpuEngine::n_loaded_layers() const { return (int)p_->layers.size(); }
bool DsCpuEngine::has_head() const { return p_->head_ok; }
void DsCpuEngine::set_engram_hash(const hive::EngramHash* eh) { p_->eh = eh; }
void DsCpuEngine::set_decoder_tail(int rows) { p_->opt.decoder_tail = std::max(0, rows); }
void DsCpuEngine::set_ab_legacy(int mask) { p_->legacy = mask; }
int DsCpuEngine::load_mtp() { p_->load_mtp(); return (int)p_->mtp.size(); }
int DsCpuEngine::n_mtp() const { return (int)p_->mtp.size(); }
bool DsCpuEngine::mtp_on() const { return !p_->mtp.empty(); }
int DsCpuEngine::mtp_block() const { return p_->c.dspark_block; }
CpuStats& DsCpuEngine::stats() { return p_->st; }
int DsCpuEngine::mtp_draft(CpuSeq& s, int32_t tok, int32_t* drafts, float* conf) { return p_->mtp_draft(s, tok, drafts, conf); }

void DsCpuEngine::init_seq(CpuSeq& s, int64_t cap) {
  const DsConfig& c = p_->c;
  s.cap = cap;
  s.ring.assign(p_->layers.size(), std::vector<uint16_t>((size_t)c.window * c.head_dim, 0));
  s.comp.assign(p_->kv_src.size(), CpuSeq::Comp{});
  for (size_t i = 0; i < p_->kv_src.size(); ++i) {
    const int r = c.ratio(p_->kv_src[i]);
    if (r > 1) { s.comp[i].st_kv.assign((size_t)r * c.head_dim, 0.f); s.comp[i].st_sc.assign((size_t)r * c.head_dim, -INFINITY); }
  }
  s.mtp_ring.assign(p_->mtp.size(), std::vector<uint16_t>((size_t)c.window * c.head_dim, 0));
  s.mtp_hidden.assign((size_t)c.dspark_targets.size() * c.dim, 0.f);
  reset_seq(s);
}

void DsCpuEngine::reset_seq(CpuSeq& s) {
  s.pos = 0; s.broken = false; s.tokens.clear(); s.engram_history.clear();
  for (auto& r : s.ring) std::fill(r.begin(), r.end(), 0);
  for (auto& cc : s.comp) { std::fill(cc.st_kv.begin(), cc.st_kv.end(), 0.f); std::fill(cc.st_sc.begin(), cc.st_sc.end(), -INFINITY); }
  for (auto& r : s.mtp_ring) std::fill(r.begin(), r.end(), 0);
  s.mtp_hidden_valid = false; s.mtp_pos = -1;
}

namespace {
void add_part(Fwd& f, CpuSeq* s, const int32_t* ids, int n) {
  Part p{s, s->pos, f.M, n};
  f.parts.push_back(p);
  for (int i = 0; i < n; ++i) {
    f.pos.push_back(s->pos + i);
    f.part_of.push_back((int)f.parts.size() - 1);
    f.ids.push_back(ids[i]);
    f.img.push_back(0);
  }
  f.M += n;
}
}  // namespace

void DsCpuEngine::prefill(CpuSeq& s, const int32_t* ids, int T, const std::vector<Image>* images, float* logits) {
  HIVE_CHECK(T >= 1 && s.pos + T <= s.cap, "ds cpu prefill: context overflow");
  const int chunk = std::max(1, p_->opt.max_chunk);
  int a = 0;
  while (a < T) {
    int b = std::min(T, a + chunk);
    // never cut an image span: end the chunk before a span it would split (or after it when the span starts the chunk)
    if (images)
      for (const Image& im : *images) {
        const int s0 = im.start, s1 = im.start + (int)im.types.size();
        if (s0 < b && s1 > b) b = s0 > a ? s0 : s1;
      }
    std::vector<Image> sub;
    if (images)
      for (const Image& im : *images)
        if (im.start >= a && im.start < b) { Image x = im; x.start -= a; sub.push_back(std::move(x)); }
    Fwd f;
    f.prefill = true;
    add_part(f, &s, ids + a, b - a);
    try { p_->run(f, sub.empty() ? nullptr : &sub, b == T ? logits : nullptr, false); } catch (...) { s.broken = true; throw; }
    a = b;
  }
}

void DsCpuEngine::decode(CpuSeq* const* seqs, const int32_t* ids, int M, float* logits) {
  HIVE_CHECK(M >= 1 && M <= 8, "ds cpu decode: batch");
  Fwd f;
  for (int m = 0; m < M; ++m) { HIVE_CHECK(seqs[m]->pos + 1 <= seqs[m]->cap, "ds cpu decode: context overflow"); add_part(f, seqs[m], ids + m, 1); }
  try { p_->run(f, nullptr, logits, true); } catch (...) { for (int m = 0; m < M; ++m) seqs[m]->broken = true; throw; }
}

void DsCpuEngine::verify_batch(CpuSeq* const* seqs, const int* Ms, int S, const int32_t* ids, float* logits) {
  Impl& I = *p_;
  const DsConfig& c = I.c;
  HIVE_CHECK(I.vparts.empty(), "ds cpu verify: rollback pending");
  Fwd f;
  f.verify = true;
  int off = 0;
  for (int p = 0; p < S; ++p) {
    HIVE_CHECK(Ms[p] >= 1 && seqs[p]->pos + Ms[p] <= seqs[p]->cap, "ds cpu verify: rows");
    Impl::VPart v{seqs[p], seqs[p]->pos, Ms[p], off, seqs[p]->engram_history.size(), {}, {}};
    // rollback snapshot: ring slots of the part's positions in every layer + the ratio > 1 compressor states
    const size_t row = (size_t)c.head_dim;
    v.ring_snap.resize((size_t)I.layers.size() * Ms[p] * row);
    for (size_t l = 0; l < I.layers.size(); ++l)
      for (int i = 0; i < Ms[p]; ++i)
        memcpy(&v.ring_snap[(l * Ms[p] + i) * row], &seqs[p]->ring[l][((seqs[p]->pos + i) % c.window) * row], row * 2);
    for (int o : I.src2) {
      const CpuSeq::Comp& cc = seqs[p]->comp[o];
      v.st_snap.insert(v.st_snap.end(), cc.st_kv.begin(), cc.st_kv.end());
      v.st_snap.insert(v.st_snap.end(), cc.st_sc.begin(), cc.st_sc.end());
    }
    I.vparts.push_back(std::move(v));
    add_part(f, seqs[p], ids + off, Ms[p]);
    off += Ms[p];
  }
  HIVE_CHECK(off <= 8, "ds cpu verify: at most 8 rows");
  try { I.run(f, nullptr, logits, true); } catch (...) { for (int p = 0; p < S; ++p) seqs[p]->broken = true; I.vparts.clear(); throw; }
  I.v_mh = std::move(f.mh);
  I.v_comp_in = std::move(f.comp_in);
  I.v_rows = f.M;
}

void DsCpuEngine::rollback_batch(const int* n_keep, int S) {
  Impl& I = *p_;
  const DsConfig& c = I.c;
  HIVE_CHECK((int)I.vparts.size() == S, "ds cpu rollback: parts");
  struct Clear { Impl& I; ~Clear() { I.vparts.clear(); I.v_mh.clear(); I.v_comp_in.clear(); I.v_rows = 0; } } clr{I};
  const size_t row = (size_t)c.head_dim, D = (size_t)c.head_dim;
  const size_t K = (size_t)c.dspark_targets.size() * c.dim;
  for (int p = 0; p < S; ++p) {
    Impl::VPart& v = I.vparts[p];
    CpuSeq& s = *v.s;
    const int nk = n_keep[p];
    HIVE_CHECK(nk >= 1 && nk <= v.M, "ds cpu rollback: range");
    if (nk < v.M) {
      s.pos = v.pos0 + nk;
      s.tokens.resize(s.tokens.size() - (size_t)(v.M - nk));
      if (s.engram_history.size() > v.hist + (size_t)nk) s.engram_history.resize(v.hist + (size_t)nk);
      for (size_t l = 0; l < I.layers.size(); ++l)
        for (int i = nk; i < v.M; ++i)
          memcpy(&s.ring[l][((v.pos0 + i) % c.window) * row], &v.ring_snap[(l * v.M + i) * row], row * 2);
      // compressor state: the snapshot, then the kept rows' inputs put back in order (completed groups of kept rows are already in the cache)
      size_t o = 0;
      for (size_t q = 0; q < I.src2.size(); ++q) {
        CpuSeq::Comp& cc = s.comp[I.src2[q]];
        const int r = c.ratio(I.kv_src[I.src2[q]]);
        std::copy(v.st_snap.begin() + o, v.st_snap.begin() + o + cc.st_kv.size(), cc.st_kv.begin()); o += cc.st_kv.size();
        std::copy(v.st_snap.begin() + o, v.st_snap.begin() + o + cc.st_sc.size(), cc.st_sc.begin()); o += cc.st_sc.size();
        for (int i = 0; i < nk; ++i) {
          const float* in = &I.v_comp_in[((q * (size_t)I.v_rows) + v.r0 + i) * 2 * D];
          const int slot = (int)((v.pos0 + i) % r);
          memcpy(&cc.st_kv[slot * D], in, D * 4);
          memcpy(&cc.st_sc[slot * D], in + D, D * 4);
        }
      }
    }
    if (!I.mtp.empty() && !I.v_mh.empty()) I.mtp_sync(s, &I.v_mh[(size_t)v.r0 * K], nk, v.pos0);
  }
}

// ---- forward ---------------------------------------------------------------------------------------------------------------------------------
void DsCpuEngine::Impl::dump(const char* what, int l, const Fwd& f, const float* data, int cols) {
  if (pub->dump_hook) pub->dump_hook(what, l, f.pos.data(), f.M, data, cols);
}

void DsCpuEngine::Impl::run(Fwd& f, const std::vector<DsCpuEngine::Image>* images, float* logits, bool all_logits) {
  const double t0 = hive::mono_ms();
  const int D = c.dim, HC = c.hc, HD = HC * D;
  const int M0 = f.M;
  const std::vector<Part> parts0 = f.parts;
  const std::vector<int32_t> ids0 = f.ids;
  f.h.resize((size_t)M0 * HD);
  f.premix.assign((size_t)M0 * HC, 0.f);
  par_for((size_t)M0, 16, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const uint16_t* e = embed.data() + (size_t)f.ids[m] * D;
      float* h = f.h.data() + m * HD;
      for (int d = 0; d < D; ++d) h[d] = b2f(e[d]);
      for (int cc = 1; cc < HC; ++cc) memcpy(h + (size_t)cc * D, h, (size_t)D * 4);
      f.premix[m * HC] = 1.f;
    }
  });
  if (images && !images->empty()) {
    HIVE_CHECK(f.parts.size() == 1, "ds cpu: images only in a prefill chunk");
    for (const DsCpuEngine::Image& im : *images) {
      HIVE_CHECK(im.start >= 0 && im.start + (int)im.types.size() <= M0, "ds cpu: image span outside the chunk");
      std::vector<float> rows;
      encode_image(im.patches, im.n_vit_h, im.n_vit_w, rows, std::to_string(f.parts[0].pos0 + im.start));
      size_t next = 0;
      for (size_t q = 0; q < im.types.size(); ++q) {
        const int m = im.start + (int)q;
        const float* src = nullptr;
        switch (im.types[q]) {
          case 0: src = image_start.data(); break;
          case 2: src = image_newline.data(); break;
          case 3: src = image_end.data(); break;
          default: HIVE_CHECK((next + 1) * D <= rows.size(), "ds cpu: image rows"); src = rows.data() + next * D; ++next; break;
        }
        for (int cc = 0; cc < HC; ++cc) memcpy(f.h.data() + (size_t)m * HD + (size_t)cc * D, src, (size_t)D * 4);
        f.img[m] = 1;
      }
    }
  }
  // engram hashes (history extended once per forward, in row order)
  const bool engram_on = eh && std::any_of(layers.begin(), layers.end(), [](const LayerW& L) { return L.engram != nullptr; });
  if (engram_on) {
    const int nL = eh->n_layers(), C = eh->n_cols();
    f.hash_layers = nL;
    f.hashes.assign((size_t)M0 * nL * C, 0);
    for (const Part& p : f.parts) {
      std::vector<int64_t> out;
      eh->compute(f.ids.data() + p.r0, f.img.data() + p.r0, p.n, p.s->engram_history, out);
      std::copy(out.begin(), out.end(), f.hashes.begin() + (size_t)p.r0 * nL * C);
    }
  }
  if (!mtp.empty()) f.mh.assign((size_t)M0 * c.dspark_targets.size() * D, 0.f);
  if (f.verify && !src2.empty()) f.comp_in.assign(src2.size() * (size_t)M0 * 2 * c.head_dim, 0.f);
  const int nl = (int)layers.size();
  const bool tail = f.prefill && opt.decoder_tail > 0 && M0 > opt.decoder_tail && tail_layer >= 0 && tail_layer < nl;
  for (int l = 0; l < nl; ++l) {
    LayerW& L = layers[l];
    bool skip = false;
    if (tail && l == tail_layer) { tail_prepare(L, f, opt.decoder_tail); skip = true; }
    if (pub->layer_inject && l > 0) pub->layer_inject(l, f.pos.data(), f.M, f.h.data(), f.premix.data());
    layer_forward(L, f, skip);
    dump("h", l, f, f.h.data(), HD);
    dump("premix", l, f, f.premix.data(), HC);
    if (f.prefill && pub->layer_boundary && l + 1 < nl) pub->layer_boundary(l);
  }
  for (const Part& p : parts0) {
    p.s->tokens.insert(p.s->tokens.end(), ids0.begin() + p.r0, ids0.begin() + p.r0 + p.n);
    p.s->pos += p.n;
  }
  if (logits && head_ok) {
    Timer tm(st.ms_head);
    if (all_logits) head_rows(f.h.data(), f.premix.data(), f.M, norm, logits, nullptr);
    else head_rows(f.h.data() + (size_t)(f.M - 1) * HD, f.premix.data() + (size_t)(f.M - 1) * HC, 1, norm, logits, nullptr);
    if (pub->dump_hook) {
      const int n = all_logits ? f.M : 1;
      pub->dump_hook("logits", -1, f.pos.data() + (f.M - n), n, logits, c.vocab);
    }
  }
  if (!mtp.empty() && !f.verify) {
    Timer tm(st.ms_mtp);
    const size_t K = (size_t)c.dspark_targets.size() * D;
    for (const Part& p : f.parts) {
      const int n = std::min(p.n, c.window);
      mtp_sync(*p.s, f.mh.data() + (size_t)(p.r0 + p.n - n) * K, n, p.pos0 + p.n - n);
    }
  }
  st.ms_total += hive::mono_ms() - t0;
  ++st.steps;
}

void DsCpuEngine::Impl::tail_prepare(LayerW& L, Fwd& f, int tail) {
  const int M = f.M, D = c.dim, HC = c.hc, HD = HC * D;
  HIVE_CHECK(f.parts.size() == 1 && L.kv_src && L.ratio >= 1, "ds cpu: decoder tail needs one part and a kv source");
  // the layer's attention input of every row → its compressor (all rows), as the layer front computes it
  std::vector<float> pre((size_t)M * HC), post((size_t)M * HC), comb((size_t)M * HC * HC), xn((size_t)M * D);
  hc_front(L, f, false, f.premix.data(), pre.data(), post.data(), comb.data(), xn.data());
  {
    Timer tm(st.ms_compress);
    std::vector<float> ckv((size_t)M * c.head_dim), csc;
    std::vector<MatJob> jobs{MatJob{MatView{&L.at.comp_wkv, 0, L.at.comp_wkv.N}, ckv.data(), c.head_dim, xn.data(), M, D}};
    if (L.ratio > 1) { csc.resize(ckv.size()); jobs.push_back(MatJob{MatView{&L.at.comp_wgate, 0, L.at.comp_wgate.N}, csc.data(), c.head_dim, xn.data(), M, D}); }
    ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
    if (L.ratio == 1) round_rows(ckv.data(), M, c.head_dim, c.head_dim);
    compress(L, f, ckv.data(), csc.empty() ? nullptr : csc.data());
  }
  // keep the last `tail` rows
  const int r0 = M - tail;
  memmove(f.h.data(), f.h.data() + (size_t)r0 * HD, (size_t)tail * HD * 4);
  memmove(f.premix.data(), f.premix.data() + (size_t)r0 * HC, (size_t)tail * HC * 4);
  f.h.resize((size_t)tail * HD); f.premix.resize((size_t)tail * HC);
  f.pos.erase(f.pos.begin(), f.pos.begin() + r0);
  f.ids.erase(f.ids.begin(), f.ids.begin() + r0);
  f.img.erase(f.img.begin(), f.img.begin() + r0);
  f.part_of.erase(f.part_of.begin(), f.part_of.begin() + r0);
  if (!f.mh.empty()) f.mh.resize((size_t)tail * c.dspark_targets.size() * D);
  f.parts[0].pos0 += r0; f.parts[0].n = tail; f.parts[0].r0 = 0;
  f.M = tail;
  f.win_min = f.parts[0].pos0;
  f.have_topk = false; f.have_cand = false;
}

void DsCpuEngine::Impl::layer_forward(LayerW& L, Fwd& f, bool skip_compress) {
  const int M = f.M, D = c.dim, HC = c.hc, HD = HC * D;
  if (L.engram && !f.hashes.empty()) { engram(L, f); dump("engram", L.id, f, f.h.data(), HD); }
  // DSpark target layer: hc mean of the attention input (reference h.mean(dim=1) of the bf16 stream)
  for (size_t t = 0; t < c.dspark_targets.size() && !f.mh.empty(); ++t)
    if (c.dspark_targets[t] == L.id) {
      const size_t K = (size_t)c.dspark_targets.size() * D;
      par_for((size_t)M, 64, [&](size_t a, size_t b) {
        for (size_t m = a; m < b; ++m) {
          const float* h = f.h.data() + m * HD;
          float* o = f.mh.data() + m * K + t * D;
          for (int d = 0; d < D; ++d) {
            float s = 0.f;
            for (int cc = 0; cc < HC; ++cc) s += h[(size_t)cc * D + d];
            o[d] = bf16r(s / (float)HC);
          }
        }
      });
    }
  std::vector<float> pre_a((size_t)M * HC), post_a((size_t)M * HC), comb_a((size_t)M * HC * HC), xn((size_t)M * D), y((size_t)M * D);
  hc_front(L, f, false, f.premix.data(), pre_a.data(), post_a.data(), comb_a.data(), xn.data());
  attention(L, f, xn.data(), y.data(), skip_compress, false, -1);
  dump("attn_out", L.id, f, y.data(), D);
  hc_post(f, y.data(), post_a.data(), comb_a.data());
  std::vector<float> pre_f((size_t)M * HC), post_f((size_t)M * HC), comb_f((size_t)M * HC * HC);
  hc_front(L, f, true, pre_a.data(), pre_f.data(), post_f.data(), comb_f.data(), xn.data());
  dump("ffn_in", L.id, f, xn.data(), D);
  moe(L, f, xn.data(), y.data());
  dump("ffn_out", L.id, f, y.data(), D);
  hc_post(f, y.data(), post_f.data(), comb_f.data());
  f.premix = pre_f;
}

// hc_mixes (fp32: mixes = (fn · flatten(h)) · rsqrt(mean(h²) + eps)) → Sinkhorn → x = bf16(Σ_c pre_in[c] · h[c]) → xn = rmsnorm(x)
void DsCpuEngine::Impl::hc_front(LayerW& L, Fwd& f, bool ffn, const float* pre_in, float* pre, float* post, float* comb, float* xn) {
  Timer tm(st.ms_hc);
  const int M = f.M, D = c.dim, HC = c.hc, HD = HC * D, MIX = (2 + HC) * HC;
  const Mat& fn = ffn ? L.hc_ffn_fn : L.hc_attn_fn;
  const float* scale = ffn ? L.hc_ffn_scale : L.hc_attn_scale;
  const float* base = ffn ? L.hc_ffn_base.data() : L.hc_attn_base.data();
  const std::vector<float>& nw = ffn ? L.ffn_norm : L.attn_norm;
  std::vector<float> mixes((size_t)M * MIX), x((size_t)M * D);
  mm(MatView{&fn, 0, fn.N}, f.h.data(), M, HD, mixes.data(), MIX);
  par_for((size_t)M, 8, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      const float* h = f.h.data() + m * HD;
      __m256 acc = _mm256_setzero_ps();
      for (int i = 0; i < HD; i += 8) { const __m256 v = _mm256_loadu_ps(h + i); acc = _mm256_fmadd_ps(v, v, acc); }
      alignas(32) float t[8];
      _mm256_store_ps(t, acc);
      float ss = 0.f;
      for (int i = 0; i < 8; ++i) ss += t[i];
      const float rsq = 1.f / std::sqrt(ss / (float)HD + c.norm_eps);
      float* mx = mixes.data() + m * MIX;
      for (int i = 0; i < MIX; ++i) mx[i] *= rsq;
      sinkhorn_row(mx, scale, base, HC, c.sinkhorn_iters, c.hc_eps, pre + m * HC, post + m * HC, comb + m * HC * HC);
      const float* p = pre_in + m * HC;
      float* xr = x.data() + m * D;
      for (int d = 0; d < D; ++d) {
        float s = p[0] * h[d];
        for (int cc = 1; cc < HC; ++cc) s += p[cc] * h[(size_t)cc * D + d];
        xr[d] = bf16r(s);
      }
    }
  });
  rmsnorm_rows(x.data(), D, nw.data(), M, D, c.norm_eps, xn, D);
}

// hc_post: h[i] = bf16(post[i] · y + Σ_j comb[j][i] · h[j])
void DsCpuEngine::Impl::hc_post(Fwd& f, const float* y, const float* post, const float* comb) {
  Timer tm(st.ms_hc);
  const int M = f.M, D = c.dim, HC = c.hc, HD = HC * D;
  HIVE_CHECK(HC == 4, "ds cpu: hc == 4");
  // every output copy reads all four input copies: a row buffer, then the row is replaced
  par_for((size_t)M, 4, [&](size_t a, size_t b) {
    std::vector<float> nh((size_t)HD);
    for (size_t m = a; m < b; ++m) {
      const float* cb = comb + m * HC * HC;
      const float* yr = y + m * D;
      float* h = f.h.data() + m * HD;
      for (int i = 0; i < HC; ++i) {
        const __m256 pv = _mm256_set1_ps(post[m * HC + i]), c0v = _mm256_set1_ps(cb[0 * HC + i]), c1v = _mm256_set1_ps(cb[1 * HC + i]),
                     c2v = _mm256_set1_ps(cb[2 * HC + i]), c3v = _mm256_set1_ps(cb[3 * HC + i]);
        for (int d = 0; d < D; d += 8) {
          __m256 s = _mm256_mul_ps(c0v, _mm256_loadu_ps(h + d));
          s = _mm256_add_ps(s, _mm256_mul_ps(c1v, _mm256_loadu_ps(h + D + d)));
          s = _mm256_add_ps(s, _mm256_mul_ps(c2v, _mm256_loadu_ps(h + 2 * D + d)));
          s = _mm256_add_ps(s, _mm256_mul_ps(c3v, _mm256_loadu_ps(h + 3 * D + d)));
          const __m256 v = _mm256_add_ps(_mm256_mul_ps(pv, _mm256_loadu_ps(yr + d)), s);
          // bf16 rounding
          const __m256i u = _mm256_castps_si256(v);
          const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
          const __m256i r = _mm256_and_si256(_mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), lsb), _mm256_set1_epi32((int)0xffff0000u));
          _mm256_storeu_ps(nh.data() + (size_t)i * D + d, _mm256_castsi256_ps(r));
        }
      }
      memcpy(h, nh.data(), (size_t)HD * 4);
    }
  });
}

// ---- engram (reference EngramLayer) ------------------------------------------------------------------------------------------------------------
void DsCpuEngine::Impl::engram(LayerW& L, Fwd& f) {
  Timer tm(st.ms_engram);
  const EngramW& E = *L.engram;
  const int M = f.M, D = c.dim, HC = c.hc, HD = HC * D, cols = c.n_hash_cols(), hd = c.engram_head_dim, ns = hd / 32, nL = f.hash_layers;
  const int K = cols * hd;
  std::vector<float> emb((size_t)M * K), kv((size_t)M * E.wkv.N);
  std::vector<uint8_t> gv, gs;   // block mode: rows gathered by the caller
  if (f.engram_rows) {
    gv.resize((size_t)M * cols * hd); gs.resize((size_t)M * cols * ns);
    f.engram_rows(E.hash_index, M, gv.data(), gs.data());
  }
  par_for((size_t)M * cols, 8, [&](size_t a, size_t b) {
    for (size_t q = a; q < b; ++q) {
      const size_t m = q / cols, col = q % cols;
      const uint8_t* v;
      const uint8_t* s;
      if (f.engram_rows) { v = gv.data() + q * hd; s = gs.data() + q * ns; }
      else {
        const uint64_t row = (uint64_t)f.hashes[(m * nL + E.hash_index) * cols + col];
        HIVE_CHECK(row < E.tab.rows, "ds cpu: engram row out of range");
        v = E.tab.val(row, hd);
        s = E.tab.scl(row, ns);
      }
      float* o = emb.data() + m * K + col * hd;
      for (int i = 0; i < hd; ++i) o[i] = bf16r(::hive::e4m3_to_f32(v[i]) * ::hive::e8m0_to_f32(s[i / 32]));
    }
  });
  act_quant_fp8(emb.data(), M, K, K, emb.data(), K);
  mm(MatView{&E.wkv, 0, E.wkv.N}, emb.data(), M, K, kv.data(), E.wkv.N);
  round_rows(kv.data(), M, E.wkv.N, E.wkv.N);
  const float scl = (float)std::pow((double)D, -0.5);
  par_for((size_t)M * HC, 4, [&](size_t a, size_t b) {
    for (size_t q = a; q < b; ++q) {
      const size_t m = q / HC; const int cc = (int)(q % HC);
      float* h = f.h.data() + m * HD + (size_t)cc * D;
      const float* key = kv.data() + m * E.wkv.N + (size_t)cc * D;
      const float* val = kv.data() + m * E.wkv.N + (size_t)HC * D;
      const float* w = E.qk.data() + (size_t)cc * D;
      float sh = 0.f, sk = 0.f, dot = 0.f;
      for (int d = 0; d < D; ++d) { sh += h[d] * h[d]; sk += key[d] * key[d]; dot += h[d] * w[d] * key[d]; }
      const float rstd = (1.f / std::sqrt(sh / (float)D + c.norm_eps)) * (1.f / std::sqrt(sk / (float)D + c.norm_eps));
      const float dt = dot * rstd * scl;
      float gate = sigm(std::copysign(std::sqrt(std::max(std::fabs(dt), 1e-6f)), dt));
      if (f.img[m]) gate = 0.f;
      for (int d = 0; d < D; ++d) h[d] = bf16r(h[d] + gate * val[d]);
    }
  });
}

// ---- attention (reference Attention.__call__) ---------------------------------------------------------------------------------------------
void DsCpuEngine::Impl::attention(LayerW& L, Fwd& f, const float* xn, float* out, bool skip_compress, bool, int) {
  const AttnW& A = L.at;
  const int M = f.M, D = c.dim, H = c.n_heads, HDm = c.head_dim, R = c.q_lora_rank, rd = c.rd, win = c.window;
  const int QW = H * HDm, Hi = c.index_n_heads, Di = c.index_head_dim;
  const bool comp_here = L.kv_src && !skip_compress;
  const Rope& rope = rope_of(L);
  std::vector<float> xq((size_t)M * D), qa((size_t)M * R), kv((size_t)M * HDm), ckv, csc, iw, qr((size_t)M * R), q((size_t)M * QW), iq;
  {
    Timer tm(st.ms_attn_proj);
    act_quant_fp8(xn, M, D, D, xq.data(), D);
    std::vector<MatJob> jobs{MatJob{MatView{&A.wq_a, 0, A.wq_a.N}, qa.data(), R, xq.data(), M, D}, MatJob{MatView{&A.wkv, 0, A.wkv.N}, kv.data(), HDm, xq.data(), M, D}};
    if (comp_here) {
      ckv.resize((size_t)M * HDm);
      jobs.push_back(MatJob{MatView{&A.comp_wkv, 0, A.comp_wkv.N}, ckv.data(), HDm, xn, M, D});
      if (L.ratio > 1) { csc.resize(ckv.size()); jobs.push_back(MatJob{MatView{&A.comp_wgate, 0, A.comp_wgate.N}, csc.data(), HDm, xn, M, D}); }
    }
    if (L.idx_src) { iw.resize((size_t)M * Hi); jobs.push_back(MatJob{MatView{&A.idx_wproj, 0, A.idx_wproj.N}, iw.data(), Hi, xn, M, D}); }
    ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
    round_rows(qa.data(), M, R, R);
    round_rows(kv.data(), M, HDm, HDm);
    if (comp_here && L.ratio == 1) round_rows(ckv.data(), M, HDm, HDm);
    if (L.idx_src) {
      const float ws = (float)((1.0 / std::sqrt((double)Di)) * (1.0 / std::sqrt((double)Hi)));
      for (float& v : iw) v = bf16r(bf16r(v) * ws);
    }
    rmsnorm_rows(qa.data(), R, A.q_norm.data(), M, R, c.norm_eps, qr.data(), R);
    rmsnorm_rows(kv.data(), HDm, A.kv_norm.data(), M, HDm, c.norm_eps, kv.data(), HDm);
    par_for((size_t)M, 16, [&](size_t a, size_t b) {
      for (size_t m = a; m < b; ++m) rope.apply(kv.data() + m * HDm, 1, HDm, rd, f.pos[m], false);
    });
    act_quant_fp8(kv.data(), M, HDm, HDm, kv.data(), HDm);
    std::vector<float> qrq((size_t)M * R);
    act_quant_fp8(qr.data(), M, R, R, qrq.data(), R);
    jobs.clear();
    jobs.push_back(MatJob{MatView{&A.wq_b, 0, A.wq_b.N}, q.data(), QW, qrq.data(), M, R});
    if (L.idx_src) { iq.resize((size_t)M * Hi * Di); jobs.push_back(MatJob{MatView{&A.idx_wq_b, 0, A.idx_wq_b.N}, iq.data(), Hi * Di, qrq.data(), M, R}); }
    ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
    round_rows(q.data(), M, QW, QW);
    if (L.idx_src) round_rows(iq.data(), M, Hi * Di, Hi * Di);
    par_for((size_t)M, 4, [&](size_t a, size_t b) {
      for (size_t m = a; m < b; ++m) rope.apply(q.data() + m * QW, H, HDm, rd, f.pos[m], false);
    });
  }
  if (L.ratio) {
    if (comp_here) { Timer tm(st.ms_compress); compress(L, f, ckv.data(), csc.empty() ? nullptr : csc.data()); }
    if (L.idx_src) { Timer tm(st.ms_index); indexer(L, f, iq.data(), iw.data()); }
    HIVE_CHECK(f.have_topk, "ds cpu: no index selection before a compressed-attention layer");
  }
  // sparse attention. Many rows (prefill): one task per row in a single dispatch — the row gathers its keys into a worker-local buffer, runs all
  //   heads and its inverse RoPE (no barrier per row block). Few rows (decode, verify): rows in blocks, gather tasks, then (row, head group)
  //   tasks so every worker has work. Every head's arithmetic is the same in both forms (head pairs for the scores, 4-head blocks for P·V) —
  //   the results are bit-identical; set_ab_legacy(1) forces the block form for A/B measurements.
  std::vector<float> o((size_t)M * QW);
  const bool row_tasks = !(legacy & 1) && M >= 2 * Pool::get().threads();
  if (row_tasks) {
    Timer tm(st.ms_attn);
    const int KT = c.index_topk, NMAX = win + KT;
    const float scale = 1.f / std::sqrt((float)HDm);
    Pool::get().run(M, [&](int m) {
      thread_local std::vector<float> keys, ws;
      keys.resize((size_t)NMAX * HDm);
      ws.resize((size_t)H * NMAX);
      const Part& pt = f.parts[f.part_of[m]];
      const int64_t p = f.pos[m];
      const int64_t w0 = std::max<int64_t>({0, p - win + 1, f.win_min});
      const int nw = (int)(p - w0 + 1);
      const int n = nw + (L.ratio ? f.topk_n[m] : 0);
      for (int j = 0; j < n; ++j) {
        float* dst = keys.data() + (size_t)j * HDm;
        if (j < nw) {
          const int64_t qp = w0 + j;
          if (qp >= pt.pos0) memcpy(dst, kv.data() + (size_t)(pt.r0 + (qp - pt.pos0)) * HDm, (size_t)HDm * 4);
          else { const uint16_t* src = pt.s->ring[L.id].data() + (size_t)(qp % win) * HDm; for (int d = 0; d < HDm; ++d) dst[d] = b2f(src[d]); }
        } else {
          const int g = f.topk[(size_t)m * KT + (j - nw)];
          const uint16_t* src = pt.s->comp[L.comp_ord].kv.data() + (size_t)g * HDm;
          for (int d = 0; d < HDm; ++d) dst[d] = b2f(src[d]);
        }
      }
      sparse_attn_row(q.data() + (size_t)m * QW, H, HDm, keys.data(), n, A.sink.data(), scale, o.data() + (size_t)m * QW, ws.data());
      rope.apply(o.data() + (size_t)m * QW, H, HDm, rd, p, true);
    });
  } else {
    Timer tm(st.ms_attn);
    const int KT = c.index_topk, NMAX = win + KT;
    const int RB = std::max(1, std::min(M, 32));
    // heads per task: 8, or 4 when few rows would leave workers idle (decode: 1–8 rows) — the P·V kernel takes 4-head blocks
    const int HG = M * (H / 8) >= Pool::get().threads() ? 8 : 4;
    std::vector<float> keys((size_t)RB * NMAX * HDm);
    std::vector<int> nkeys(RB);
    const float scale = 1.f / std::sqrt((float)HDm);
    for (int b0 = 0; b0 < M; b0 += RB) {
      const int nb = std::min(RB, M - b0);
      // gather: (row, 64-key chunk) tasks
      for (int r = 0; r < nb; ++r) {
        const int m = b0 + r;
        const int64_t p = f.pos[m];
        const int64_t w0 = std::max<int64_t>({0, p - win + 1, f.win_min});
        nkeys[r] = (int)(p - w0 + 1) + (L.ratio ? f.topk_n[m] : 0);
      }
      Pool::get().run(nb * ((NMAX + 63) / 64), [&](int t) {
        const int r = t / ((NMAX + 63) / 64), ch = t % ((NMAX + 63) / 64);
        const int m = b0 + r;
        const Part& pt = f.parts[f.part_of[m]];
        const int64_t p = f.pos[m];
        const int64_t w0 = std::max<int64_t>({0, p - win + 1, f.win_min});
        const int nw = (int)(p - w0 + 1);
        float* K = keys.data() + (size_t)r * NMAX * HDm;
        for (int j = ch * 64; j < std::min(nkeys[r], ch * 64 + 64); ++j) {
          float* dst = K + (size_t)j * HDm;
          if (j < nw) {
            const int64_t qp = w0 + j;
            if (qp >= pt.pos0) memcpy(dst, kv.data() + (size_t)(pt.r0 + (qp - pt.pos0)) * HDm, (size_t)HDm * 4);
            else { const uint16_t* src = pt.s->ring[L.id].data() + (size_t)(qp % win) * HDm; for (int d = 0; d < HDm; ++d) dst[d] = b2f(src[d]); }
          } else {
            const int g = f.topk[(size_t)m * KT + (j - nw)];
            const uint16_t* src = pt.s->comp[L.comp_ord].kv.data() + (size_t)g * HDm;
            for (int d = 0; d < HDm; ++d) dst[d] = b2f(src[d]);
          }
        }
      });
      Pool::get().run(nb * (H / HG), [&](int t) {
        const int r = t / (H / HG), g = t % (H / HG);
        const int m = b0 + r;
        thread_local std::vector<float> ws;
        ws.resize((size_t)HG * NMAX);
        sparse_attn_row(q.data() + (size_t)m * QW + (size_t)g * HG * HDm, HG, HDm, keys.data() + (size_t)r * NMAX * HDm, nkeys[r],
                        A.sink.data() + g * HG, scale, o.data() + (size_t)m * QW + (size_t)g * HG * HDm, ws.data());
      });
    }
    par_for((size_t)M, 4, [&](size_t a, size_t b) {
      for (size_t m = a; m < b; ++m) rope.apply(o.data() + m * QW, H, HDm, rd, f.pos[m], true);
    });
  }
  {
    Timer tm(st.ms_attn_out);
    const int G = c.o_groups, OR = c.o_lora_rank, sub = QW / G;
    // reference act_quant(o, 32, ue8m0) in place before wo_a (model.py since deepseek-ai/DeepSeek-V4.1-Flash 4dd75a8, 2026-10-12) — o is bf16 there
    round_rows(o.data(), M, QW, QW);
    act_quant_fp8(o.data(), M, QW, QW, o.data(), QW);
    std::vector<float> og((size_t)M * G * OR);
    std::vector<MatJob> jobs;
    for (int g = 0; g < G; ++g) jobs.push_back(MatJob{MatView{&A.wo_a, g * OR, (g + 1) * OR}, og.data() + (size_t)g * OR, G * OR, o.data() + (size_t)g * sub, M, QW});
    ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
    round_rows(og.data(), M, G * OR, G * OR);
    act_quant_fp8(og.data(), M, G * OR, G * OR, og.data(), G * OR);
    mm(MatView{&A.wo_b, 0, A.wo_b.N}, og.data(), M, G * OR, out, D);
    round_rows(out, M, D, D);
  }
  // ring: this forward's window keys (rows in position order — the latest row of a slot wins)
  for (const Part& pt : f.parts)
    for (int i = 0; i < pt.n; ++i) {
      uint16_t* dst = pt.s->ring[L.id].data() + (size_t)((pt.pos0 + i) % win) * HDm;
      const float* src = kv.data() + (size_t)(pt.r0 + i) * HDm;
      for (int d = 0; d < HDm; ++d) dst[d] = f2b(src[d]);
    }
}

// ---- compressor (reference Compressor + the indexer's own keys) ----------------------------------------------------------------------------
void DsCpuEngine::Impl::compress(LayerW& L, Fwd& f, const float* ckv, const float* csc) {
  const AttnW& A = L.at;
  const int HDm = c.head_dim, Di = c.index_head_dim, rd = c.rd, r = L.ratio;
  int src2_ord = -1;
  for (size_t q = 0; q < src2.size(); ++q) if (src2[q] == L.own_ord) src2_ord = (int)q;
  for (const Part& pt : f.parts) {
    CpuSeq::Comp& C = pt.s->comp[L.own_ord];
    const int64_t gmax = (pt.pos0 + pt.n) / r;
    if ((int64_t)C.kv.size() < gmax * HDm) {
      const int64_t want = std::min<int64_t>(std::max<int64_t>(gmax, (int64_t)C.kv.size() / HDm * 2), std::max<int64_t>(gmax, pt.s->cap / r + 1));
      C.kv.resize((size_t)want * HDm);
      C.ik.resize((size_t)want * Di);
    }
    std::vector<float> lat;   // [G, HDm] bf16 values
    std::vector<int64_t> gid;
    std::vector<float> pooled(HDm);
    if (r > 1 && !(legacy & 2)) {
      // groups completing in this part, pooled in parallel: slot k of group g = position g·r + k — this part's row, or the compressor state
      //   carried in from earlier positions (read before the state is updated below). Same per-group arithmetic as the row-by-row form.
      if (f.verify && src2_ord >= 0)
        for (int i = 0; i < pt.n; ++i) {
          float* dst = &f.comp_in[(((size_t)src2_ord * f.M) + pt.r0 + i) * 2 * HDm];
          memcpy(dst, ckv + (size_t)(pt.r0 + i) * HDm, (size_t)HDm * 4); memcpy(dst + HDm, csc + (size_t)(pt.r0 + i) * HDm, (size_t)HDm * 4);
        }
      for (int i = 0; i < pt.n; ++i) if ((pt.pos0 + i + 1) % r == 0) gid.push_back((pt.pos0 + i) / r);
      lat.resize(gid.size() * HDm);
      par_for(gid.size(), 16, [&](size_t a, size_t b) {
        for (size_t q = a; q < b; ++q) {
          const float* kvs[16];
          const float* scs[16];
          for (int k = 0; k < r; ++k) {
            const int64_t qp = gid[q] * r + k;
            if (qp >= pt.pos0) { kvs[k] = ckv + (size_t)(pt.r0 + (qp - pt.pos0)) * HDm; scs[k] = csc + (size_t)(pt.r0 + (qp - pt.pos0)) * HDm; }
            else { kvs[k] = &C.st_kv[(size_t)(qp % r) * HDm]; scs[k] = &C.st_sc[(size_t)(qp % r) * HDm]; }
          }
          float* out = lat.data() + q * HDm;
          for (int d = 0; d < HDm; ++d) {
            float mx = -INFINITY;
            for (int s2 = 0; s2 < r; ++s2) mx = std::max(mx, scs[s2][d]);
            float e[16], sum = 0.f;
            for (int s2 = 0; s2 < r; ++s2) { e[s2] = std::exp(scs[s2][d] - mx); sum += e[s2]; }
            float acc = 0.f;
            for (int s2 = 0; s2 < r; ++s2) acc += kvs[s2][d] * (e[s2] / sum);
            out[d] = bf16r(acc);
          }
        }
      });
      for (int i = std::max(0, pt.n - r); i < pt.n; ++i) {  // the state keeps the latest row of each slot
        const int slot = (int)((pt.pos0 + i) % r);
        memcpy(&C.st_kv[(size_t)slot * HDm], ckv + (size_t)(pt.r0 + i) * HDm, (size_t)HDm * 4);
        memcpy(&C.st_sc[(size_t)slot * HDm], csc + (size_t)(pt.r0 + i) * HDm, (size_t)HDm * 4);
      }
    } else
    for (int i = 0; i < pt.n; ++i) {
      const int64_t p = pt.pos0 + i;
      const float* kr = ckv + (size_t)(pt.r0 + i) * HDm;
      if (r == 1) {
        lat.insert(lat.end(), kr, kr + HDm);
        gid.push_back(p);
        continue;
      }
      const float* sr = csc + (size_t)(pt.r0 + i) * HDm;
      if (f.verify && src2_ord >= 0) {
        float* dst = &f.comp_in[(((size_t)src2_ord * f.M) + pt.r0 + i) * 2 * HDm];
        memcpy(dst, kr, (size_t)HDm * 4); memcpy(dst + HDm, sr, (size_t)HDm * 4);
      }
      const int slot = (int)(p % r);
      memcpy(&C.st_kv[(size_t)slot * HDm], kr, (size_t)HDm * 4);
      memcpy(&C.st_sc[(size_t)slot * HDm], sr, (size_t)HDm * 4);
      if ((p + 1) % r == 0) {
        for (int d = 0; d < HDm; ++d) {
          float mx = -INFINITY;
          for (int s2 = 0; s2 < r; ++s2) mx = std::max(mx, C.st_sc[(size_t)s2 * HDm + d]);
          float e[16], sum = 0.f;
          for (int s2 = 0; s2 < r; ++s2) { e[s2] = std::exp(C.st_sc[(size_t)s2 * HDm + d] - mx); sum += e[s2]; }
          float acc = 0.f;
          for (int s2 = 0; s2 < r; ++s2) acc += C.st_kv[(size_t)s2 * HDm + d] * (e[s2] / sum);
          pooled[d] = bf16r(acc);
        }
        lat.insert(lat.end(), pooled.begin(), pooled.end());
        gid.push_back(p / r);
      }
    }
    const int G = (int)gid.size();
    if (!G) continue;
    rmsnorm_rows(lat.data(), HDm, A.comp_norm.data(), G, HDm, c.norm_eps, lat.data(), HDm);
    if (L.idx_src) {
      std::vector<float> k((size_t)G * Di);
      mm(MatView{&A.idx_wk, 0, A.idx_wk.N}, lat.data(), G, HDm, k.data(), Di);
      round_rows(k.data(), G, Di, Di);
      rmsnorm_rows(k.data(), Di, A.idx_k_norm.data(), G, Di, c.norm_eps, k.data(), Di);
      par_for((size_t)G, 64, [&](size_t a, size_t b) {
        for (size_t g = a; g < b; ++g) {
          float* kr = k.data() + g * Di;
          rope_comp.apply(kr, 1, Di, rd, gid[g] * r, false);
          fp4_quant_row(kr, Di, 32, false);
          memcpy(&C.ik[(size_t)gid[g] * Di], kr, (size_t)Di * 4);
        }
      });
    }
    par_for((size_t)G, 32, [&](size_t a, size_t b) {
      for (size_t g = a; g < b; ++g) {
        float* lr = lat.data() + g * HDm;
        rope_comp.apply(lr, 1, HDm, rd, gid[g] * r, false);
        fp4_quant_row(lr, HDm, 16, true);
        uint16_t* dst = &C.kv[(size_t)gid[g] * HDm];
        for (int d = 0; d < HDm; ++d) dst[d] = f2b(lr[d]);
      }
    });
  }
}

// ---- indexer (reference Indexer) ---------------------------------------------------------------------------------------------------------------
void DsCpuEngine::Impl::indexer(LayerW& L, Fwd& f, float* iq, const float* iw) {
  const int M = f.M, Hi = c.index_n_heads, Di = c.index_head_dim, rd = c.rd, r = L.ratio, KT = c.index_topk, bs = c.cand_block, CB = c.cand_topk_blocks;
  const bool cand_src = L.id == c.cand_source_layer;
  const bool uses_cand = c.cand_source_layer >= 0 && c.cand_source_layer < L.id;
  par_for((size_t)M, 4, [&](size_t a, size_t b) {
    for (size_t m = a; m < b; ++m) {
      float* qm = iq + m * Hi * Di;
      rope_comp.apply(qm, Hi, Di, rd, f.pos[m], false);
      fp4_quant_row(qm, Hi * Di, 32, false);
    }
  });
  f.topk.assign((size_t)M * KT, -1);
  f.topk_n.assign(M, 0);
  if (cand_src) { f.cand.assign((size_t)M * CB, -1); f.cand_n.assign(M, 0); }
  if (uses_cand) HIVE_CHECK(f.have_cand, "ds cpu: candidate blocks missing");
  // rows in blocks; tasks = (row, chunk of 1024 candidate keys) for the scores, then one task per row for the selection
  std::vector<int> T(M);
  int Tmax = 0;
  for (int m = 0; m < M; ++m) { T[m] = (int)((f.pos[m] + 1) / r); Tmax = std::max(Tmax, T[m]); }
  const int RB = std::max(1, std::min(M, std::max(1, (int)((64ll << 20) / 4 / std::max(1, Tmax)))));
  std::vector<float> score((size_t)RB * std::max(1, Tmax));
  std::vector<std::vector<int32_t>> cpos(RB);
  constexpr int CH = 1024;
  for (int b0 = 0; b0 < M; b0 += RB) {
    const int nb = std::min(RB, M - b0);
    std::vector<int> nk(nb), nch(nb), first(nb + 1, 0);
    for (int i = 0; i < nb; ++i) {
      const int m = b0 + i;
      cpos[i].clear();
      if (uses_cand) {
        for (int q = 0; q < f.cand_n[m]; ++q) {
          const int blk = f.cand[(size_t)m * CB + q];
          if (blk < 0 || (int64_t)blk * bs >= T[m]) continue;   // entries past the row's blocks are not candidates
          for (int t = blk * bs; t < std::min(T[m], (blk + 1) * bs); ++t) cpos[i].push_back(t);
        }
        nk[i] = (int)cpos[i].size();
      } else nk[i] = T[m];
      nch[i] = (nk[i] + CH - 1) / CH;
      first[i + 1] = first[i] + nch[i];
    }
    Pool::get().run(first[nb], [&](int t) {
      const int i = (int)(std::upper_bound(first.begin(), first.end(), t) - first.begin()) - 1;
      const int m = b0 + i, ch = t - first[i];
      const int j0 = ch * CH, j1 = std::min(nk[i], j0 + CH);
      const CpuSeq& s = *f.parts[f.part_of[m]].s;
      const float* keys = s.comp[L.comp_ord].ik.data();
      // candidate rows: key j = position cpos[j]; otherwise key j = position j
      index_scores_row(iq + (size_t)m * Hi * Di, iw + (size_t)m * Hi, Hi, Di, uses_cand ? keys : keys + (size_t)j0 * Di,
                       uses_cand ? cpos[i].data() + j0 : nullptr, j1 - j0, score.data() + (size_t)i * Tmax + j0);
    });
    Pool::get().run(nb, [&](int i) {
      const int m = b0 + i;
      thread_local std::vector<int32_t> scratch;
      float* sc = score.data() + (size_t)i * Tmax;
      if (cand_src) {
        const int nblocks = (T[m] + bs - 1) / bs;
        std::vector<float> bmax(nblocks, -INFINITY);
        for (int t = 0; t < T[m]; ++t) bmax[t / bs] = std::max(bmax[t / bs], sc[t]);
        if (nblocks > 0) bmax[(T[m] - 1) / bs] = INFINITY;
        const int n = topk_sorted(bmax.data(), nullptr, nblocks, std::min(CB, nblocks), &f.cand[(size_t)m * CB], scratch);
        f.cand_n[m] = n;
      }
      f.topk_n[m] = topk_sorted(sc, uses_cand ? cpos[i].data() : nullptr, nk[i], std::min(KT, T[m]), &f.topk[(size_t)m * KT], scratch);
    });
  }
  if (cand_src) f.have_cand = true;
  f.have_topk = true;
  if (pub->index_hook) pub->index_hook(L.id, f.pos.data(), M, f.topk.data(), KT, f.topk_n.data());
}

// ---- MoE (reference MoE: sqrt-softplus router with bias top-k, MXFP4 experts, FP8 shared expert) -------------------------------------------
void DsCpuEngine::Impl::moe(LayerW& L, Fwd& f, const float* xn, float* out) {
  const MoeW& W = L.moe;
  const int M = f.M, D = c.dim, I = c.moe_inter, E = W.E, k = W.k, hh = D / 2;
  std::vector<int32_t> ids((size_t)M * k);
  std::vector<float> wts((size_t)M * k);
  {
    Timer tm(st.ms_router);
    std::vector<float> sc((size_t)M * E);
    mm(MatView{&W.gate, 0, W.gate.N}, xn, M, D, sc.data(), E);
    par_for((size_t)M, 16, [&](size_t a, size_t b) {
      std::vector<float> sel(E);
      std::vector<int> ord(E);
      for (size_t m = a; m < b; ++m) {
        float* s = sc.data() + m * E;
        for (int e = 0; e < E; ++e) { const float x = s[e]; s[e] = std::sqrt(x > 20.f ? x : std::log1p(std::exp(x))); }
        const std::vector<float>& bias = (f.img[m] && !W.bias_vl.empty()) ? W.bias_vl : W.bias;
        for (int e = 0; e < E; ++e) sel[e] = s[e] + bias[e];
        std::iota(ord.begin(), ord.end(), 0);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int x1, int x2) { return sel[x1] > sel[x2] || (sel[x1] == sel[x2] && x1 < x2); });
        float tot = 0.f;
        for (int j = 0; j < k; ++j) tot += s[ord[j]];
        for (int j = 0; j < k; ++j) { ids[m * k + j] = ord[j]; wts[m * k + j] = s[ord[j]] / (tot + 1e-20f) * c.route_scale; }
      }
    });
    if (pub->route_hook) pub->route_hook(L.id, f.pos.data(), M, ids.data(), wts.data(), k);
  }
  Timer tm(st.ms_moe);
  std::vector<float> xq((size_t)M * D);
  act_quant_fp8(xn, M, D, D, xq.data(), D);
  // (row, slot) pairs gathered per expert (experts ascending, rows ascending inside)
  const int P = M * k;
  std::vector<int> cnt(E, 0), off(E + 1, 0), prow(P), pslot(P), pexp(P);
  for (int i = 0; i < P; ++i) ++cnt[ids[i]];
  for (int e = 0; e < E; ++e) off[e + 1] = off[e] + cnt[e];
  {
    std::vector<int> fill(off.begin(), off.end() - 1);
    for (int m = 0; m < M; ++m) for (int j = 0; j < k; ++j) { const int e = ids[(size_t)m * k + j], q = fill[e]++; prow[q] = m; pslot[q] = j; pexp[q] = e; }
  }
  std::vector<float> Xe((size_t)P * D), G((size_t)P * I), U((size_t)P * I), Gs((size_t)M * I), Us((size_t)M * I);
  par_for((size_t)P, 64, [&](size_t a, size_t b) { for (size_t q = a; q < b; ++q) memcpy(Xe.data() + q * D, xq.data() + (size_t)prow[q] * D, (size_t)D * 4); });
  std::vector<MatJob> jobs;
  jobs.push_back(MatJob{MatView{&W.sh1, 0, W.sh1.N}, Gs.data(), I, xq.data(), M, D});
  jobs.push_back(MatJob{MatView{&W.sh3, 0, W.sh3.N}, Us.data(), I, xq.data(), M, D});
  for (int e = 0; e < E; ++e) {
    if (!cnt[e]) continue;
    const int r = off[e];
    jobs.push_back(MatJob{W.ex[e].g, G.data() + (size_t)r * I, I, Xe.data() + (size_t)r * D, cnt[e], D});
    jobs.push_back(MatJob{W.ex[e].u, U.data() + (size_t)r * I, I, Xe.data() + (size_t)r * D, cnt[e], D});
  }
  ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
  // y = bf16(w · silu(min(bf16 gate, limit)) · clamp(bf16 up, ±limit)), then act_quant for the down projection
  const float lim = c.swiglu_limit;
  auto act = [&](float* g, const float* u, int n, float w) {
    for (int i = 0; i < n; ++i) {
      float gv = bf16r(g[i]), uv = bf16r(u[i]);
      if (lim > 0) { uv = std::min(std::max(uv, -lim), lim); gv = std::min(gv, lim); }
      g[i] = bf16r(w * (siluf(gv) * uv));
    }
  };
  par_for((size_t)P + M, 4, [&](size_t a, size_t b) {
    for (size_t q = a; q < b; ++q) {
      if (q < (size_t)P) act(G.data() + q * I, U.data() + q * I, I, wts[(size_t)prow[q] * k + pslot[q]]);
      else {
        const size_t m = q - P;
        // the shared expert has no weight: y = silu(gate) · up (multiplying by 1 is exact)
        act(Gs.data() + m * I, Us.data() + m * I, I, 1.f);
      }
    }
  });
  act_quant_fp8(G.data(), P, I, I, G.data(), I);
  act_quant_fp8(Gs.data(), M, I, I, Gs.data(), I);
  std::vector<float> Oe((size_t)P * D), Os((size_t)M * D);
  jobs.clear();
  jobs.push_back(MatJob{MatView{&W.sh2, 0, W.sh2.N}, Os.data(), D, Gs.data(), M, I});
  for (int e = 0; e < E; ++e) {
    if (!cnt[e]) continue;
    const int r = off[e];
    jobs.push_back(MatJob{W.ex[e].d0, Oe.data() + (size_t)r * D, D, G.data() + (size_t)r * I, cnt[e], I});
    if (W.ex[e].d1.m) jobs.push_back(MatJob{W.ex[e].d1, Oe.data() + (size_t)r * D + hh, D, G.data() + (size_t)r * I, cnt[e], I});
  }
  ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
  // out = bf16(Σ_{experts ascending} bf16(y_e) + bf16(shared)) — the reference accumulates expert by expert in id order, then the shared one
  std::vector<int> slot_of((size_t)M * k);
  for (int q = 0; q < P; ++q) slot_of[(size_t)prow[q] * k + pslot[q]] = q;
  par_for((size_t)M, 2, [&](size_t a, size_t b) {
    std::vector<std::pair<int, int>> order(k);
    for (size_t m = a; m < b; ++m) {
      for (int j = 0; j < k; ++j) order[j] = {ids[m * k + j], slot_of[m * k + j]};
      std::sort(order.begin(), order.end());
      float* o = out + m * D;
      for (int d = 0; d < D; ++d) {
        float acc = 0.f;
        for (int j = 0; j < k; ++j) acc += bf16r(Oe[(size_t)order[j].second * D + d]);
        acc += bf16r(Os[m * D + d]);
        o[d] = bf16r(acc);
      }
    }
  });
}

// ---- layer block mode ---------------------------------------------------------------------------------------------------------------------
void DsCpuEngine::run_block(Block& b) {
  Impl& I = *p_;
  const DsConfig& c = I.c;
  const double t0 = hive::mono_ms();
  const int D = c.dim, HC = c.hc, HD = HC * D, first = I.opt.first_layer, nl = (int)I.layers.size(), M = b.M;
  HIVE_CHECK(M >= 1 && first >= 1 && first < nl, "ds cpu block: rows / layers");
  Fwd f;
  f.prefill = b.prefill;
  f.M = M;
  for (const BlockPart& bp : b.parts) {
    f.parts.push_back(Part{bp.s, bp.pos0, bp.r0, bp.n});
    for (int i = 0; i < bp.n; ++i) f.part_of.push_back((int)f.parts.size() - 1);
  }
  HIVE_CHECK((int)f.part_of.size() == M, "ds cpu block: parts do not cover the rows");
  f.pos.assign(b.pos, b.pos + M);
  f.ids.assign(M, 0);
  f.img.assign(M, 0);
  if (b.img) f.img.assign(b.img, b.img + M);
  f.h.assign(b.h, b.h + (size_t)M * HD);
  f.premix.assign(b.premix, b.premix + (size_t)M * HC);
  if (b.hashes) {
    f.hash_layers = (int)c.engram_layer_ids.size();
    f.hashes.assign(b.hashes, b.hashes + (size_t)M * f.hash_layers * c.n_hash_cols());
    f.engram_rows = b.engram_rows;
  }
  if (b.topk) {
    f.topk.assign((size_t)M * c.index_topk, -1);
    f.topk_n.assign(M, 0);
    for (int m = 0; m < M; ++m) {
      const int n = std::min(b.topk_n[m], c.index_topk);
      std::copy(b.topk + (size_t)m * b.topk_stride, b.topk + (size_t)m * b.topk_stride + n, f.topk.begin() + (size_t)m * c.index_topk);
      f.topk_n[m] = n;
    }
    f.have_topk = true;
  }
  if (b.cand) {
    f.cand.assign((size_t)M * c.cand_topk_blocks, -1);
    f.cand_n.assign(M, 0);
    for (int m = 0; m < M; ++m) {
      const int n = std::min(b.cand_n[m], c.cand_topk_blocks);
      std::copy(b.cand + (size_t)m * b.cand_stride, b.cand + (size_t)m * b.cand_stride + n, f.cand.begin() + (size_t)m * c.cand_topk_blocks);
      f.cand_n[m] = n;
    }
    f.have_cand = true;
  }
  f.win_min = b.win_min;
  const bool tail = f.prefill && I.opt.decoder_tail > 0 && M > I.opt.decoder_tail && M >= I.opt.tail_min_rows && I.tail_layer >= first && I.tail_layer < nl;
  int r_off = 0;
  for (int l = first; l < nl; ++l) {
    LayerW& L = I.layers[l];
    bool skip = false;
    if (tail && l == I.tail_layer) { r_off = f.M - I.opt.decoder_tail; I.tail_prepare(L, f, I.opt.decoder_tail); skip = true; }
    if (layer_inject) layer_inject(l, f.pos.data(), f.M, f.h.data(), f.premix.data());
    I.layer_forward(L, f, skip);
    I.dump("h", l, f, f.h.data(), HD);
    I.dump("premix", l, f, f.premix.data(), HC);
  }
  b.out_r0 = r_off;
  b.out_M = f.M;
  std::copy(f.h.begin(), f.h.end(), b.h + (size_t)r_off * HD);
  std::copy(f.premix.begin(), f.premix.end(), b.premix + (size_t)r_off * HC);
  I.st.ms_total += hive::mono_ms() - t0;
  ++I.st.steps;
}

void DsCpuEngine::mirror_source(CpuSeq& s, int src_layer, const uint8_t* comp_rows, const uint8_t* idx_rows, int64_t g0, int64_t g1) {
  Impl& I = *p_;
  const DsConfig& c = I.c;
  const int ord = (int)(std::find(I.kv_src.begin(), I.kv_src.end(), src_layer) - I.kv_src.begin());
  HIVE_CHECK(ord < (int)I.kv_src.size() && g1 >= g0, "ds cpu mirror: not a kv source");
  if (g1 == g0) return;
  CpuSeq::Comp& C = s.comp[ord];
  const int HDm = c.head_dim, Di = c.index_head_dim, r = c.ratio(src_layer);
  if ((int64_t)C.kv.size() < g1 * HDm) {
    const int64_t want = std::min<int64_t>(std::max<int64_t>(g1, (int64_t)C.kv.size() / HDm * 2), std::max<int64_t>(g1, s.cap / r + 1));
    C.kv.resize((size_t)want * HDm);
    C.ik.resize((size_t)want * Di);
  }
  static const float e2m1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  constexpr int CROW = 288, CSC = 32, IROW = 80, IHDR = 16;
  par_for((size_t)(g1 - g0), 256, [&](size_t a, size_t e) {
    for (size_t q = a; q < e; ++q) {
      const int64_t g = g0 + (int64_t)q;
      const uint8_t* cr = comp_rows + q * CROW;
      uint16_t* dst = &C.kv[(size_t)g * HDm];
      for (int d = 0; d < HDm; ++d) {
        const int nib = (cr[CSC + d / 2] >> (4 * (d & 1))) & 0xF;
        dst[d] = f2b(e2m1[nib] * ::hive::e4m3_to_f32(cr[d / 16]));
      }
      if (idx_rows) {
        const uint8_t* ir = idx_rows + q * IROW;
        float* kd = &C.ik[(size_t)g * Di];
        for (int d = 0; d < Di; ++d) {
          const int nib = (ir[IHDR + d / 2] >> (4 * (d & 1))) & 0xF;
          kd[d] = e2m1[nib] * ::hive::e8m0_to_f32(ir[d / 32]);
        }
      }
    }
  });
}

// final head: x = bf16(Σ_c premix[c] · h[c]) → rmsnorm → logits (fp32 head)
void DsCpuEngine::Impl::head_rows(const float* h, const float* premix, int M, const std::vector<float>& nw, float* logits, float* x_out) {
  const int D = c.dim, HC = c.hc, HD = HC * D;
  std::vector<float> x((size_t)M * D), xn((size_t)M * D);
  for (int m = 0; m < M; ++m)
    for (int d = 0; d < D; ++d) {
      float s = premix[(size_t)m * HC] * h[(size_t)m * HD + d];
      for (int cc = 1; cc < HC; ++cc) s += premix[(size_t)m * HC + cc] * h[(size_t)m * HD + (size_t)cc * D + d];
      x[(size_t)m * D + d] = bf16r(s);
    }
  if (x_out) memcpy(x_out, x.data(), x.size() * 4);
  rmsnorm_rows(x.data(), D, nw.data(), M, D, c.norm_eps, xn.data(), D);
  mm(MatView{&head, 0, head.N}, xn.data(), M, D, logits, c.vocab);
}

}  // namespace hive::ds::cpu
