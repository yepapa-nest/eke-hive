// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next CPU engine — the MTP block (mtp_draft of oracle/qwen4_oracle.py). Entry j of a sequence = (final hidden stream h_j,
//   token x_{j+1}) at rope position j; its attention layer has its own caches, so every entry whose next token is known is written once
//   (K / V / indexer key only) and a draft runs the whole block for the pending entry and chains its own output stream for further drafts.
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "qwen_cpu_internal.h"

namespace hive::qwen::cpu {

namespace {
inline float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

void grow(CpuSeq::Attn& a, int64_t tokens, int kvw, int id, int ratio) {
  if ((int64_t)a.k.size() < tokens * kvw) { a.k.resize((size_t)tokens * kvw); a.v.resize((size_t)tokens * kvw); }
  if ((int64_t)a.ik.size() < tokens * id) a.ik.resize((size_t)tokens * id);
  if ((int64_t)a.gk.size() < (tokens / ratio + 1) * id) a.gk.resize((size_t)(tokens / ratio + 1) * id);
}

// rope coordinates of MTP entry j: the sequence's own record when the position was already processed, else text positions
void entry_coords(const CpuSeq& s, int64_t j, int32_t* o) {
  if ((int64_t)s.rope.size() >= (j + 1) * 3) { memcpy(o, s.rope.data() + (size_t)j * 3, 12); return; }
  o[0] = o[1] = o[2] = (int32_t)(j + s.rope_delta);
}
}  // namespace

// hc [n, 10240] = bf16(fc_embedding(norm(emb)) + fc_hidden(norm(h)) per copy)
void QwenCpuEngine::Impl::mtp_input(const float* emb, const float* h_rows, int n, float* hc) {
  const int H = c.hidden, HC = c.hcw();
  std::vector<float> en((size_t)n * H), e((size_t)n * H), hn((size_t)n * HC), fh((size_t)n * HC);
  norm1p(emb, H, mtp->pre_emb_norm.data(), n, H, H, c.rms_eps, en.data(), H);
  lin(mtp->fc_emb, en.data(), n, H, e.data(), H);
  norm1p(h_rows, HC, mtp->pre_hidden_norm.data(), n, HC, HC, c.rms_eps, hn.data(), HC);
  lin(mtp->fc_hidden, hn.data(), n * c.hc, H, fh.data(), H);
  for (int r = 0; r < n; ++r)
    for (int cc = 0; cc < c.hc; ++cc)
      for (int d = 0; d < H; ++d) {
        const size_t i = (size_t)r * HC + (size_t)cc * H + d;
        hc[i] = bf16r(e[(size_t)r * H + d] + fh[i]);
      }
}

namespace {
// K / V / indexer-key writes of MTP entries rows[i] (sequence, entry position) from streams h [n, HC] and next-token embeddings emb [n, H]
void mtp_kv_rows(QwenCpuEngine::Impl& I, const std::vector<RowInfo>& rows, const float* h, const float* emb) {
  const QwenConfig& c = I.c;
  const int n = (int)rows.size();
  if (n == 0) return;
  std::vector<float> hc((size_t)n * c.hcw()), x((size_t)n * c.hidden), hn, dl;
  std::vector<int32_t> coords((size_t)n * 3);
  for (int r = 0; r < n; ++r) {
    grow(rows[r].s->mtp, rows[r].pos + 1 + 8, c.kv_width(), c.idx_dim, c.idx_ratio);
    entry_coords(*rows[r].s, rows[r].pos, coords.data() + (size_t)r * 3);
  }
  I.mtp_input(emb, h, n, hc.data());
  I.hc_mix(I.mtp->layer.ha, hc.data(), n, x.data(), hn, dl);
  I.attn_layer(I.mtp->layer.att, 0, true, rows, coords, x.data(), nullptr, true);
  for (const RowInfo& ri : rows) ri.s->mtp_written = std::max(ri.s->mtp_written, ri.pos + 1);
}

void embed_rows(const QwenCpuEngine::Impl& I, const int32_t* ids, int n, float* out) {
  const int H = I.c.hidden;
  for (int r = 0; r < n; ++r) for (int d = 0; d < H; ++d) out[(size_t)r * H + d] = b2f(I.embed[(size_t)ids[r] * H + d]);
}
}  // namespace

void QwenCpuEngine::Impl::mtp_kv(CpuSeq& s, int64_t pos0, int n, const float* h_rows, const int32_t* next_ids) {
  if (n <= 0) return;
  std::vector<RowInfo> rows(n);
  for (int i = 0; i < n; ++i) rows[i] = {&s, pos0 + i};
  std::vector<float> emb((size_t)n * c.hidden);
  embed_rows(*this, next_ids, n, emb.data());
  mtp_kv_rows(*this, rows, h_rows, emb.data());
}

// After a main forward (fh = its final streams): writes the entries whose next token became known. Verify rows only write the entry in
//   front of each part (the part's first token is always kept); rollback_batch writes the accepted ones.
void QwenCpuEngine::Impl::mtp_after_forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill) {
  const int M = (int)rows.size(), H = c.hidden, HC = c.hcw();
  std::vector<RowInfo> er;
  std::vector<float> eh, ee;
  auto add = [&](CpuSeq* s, int64_t j, const float* h, int r_next) {
    er.push_back({s, j});
    eh.insert(eh.end(), h, h + HC);
    const size_t o = ee.size();
    ee.resize(o + H);
    embed_rows(*this, ids + r_next, 1, ee.data() + o);
    if (prefill) {  // an image row's entry takes the vision embedding, as the main stream did
      const int64_t g = emb_base + r_next;
      for (const EmbedOverride& ov : emb_over)
        if (g >= ov.row && g < ov.row + ov.n) memcpy(ee.data() + o, ov.emb.data() + (size_t)(g - ov.row) * H, (size_t)H * 4);
    }
  };
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s && rows[r + n].pos == rows[r].pos + n) ++n;
    const int64_t p0 = rows[r].pos;
    if (s.h_last_valid && p0 >= 1) add(&s, p0 - 1, s.h_last.data(), r);
    if (!verify) {
      for (int t = 0; t + 1 < n; ++t) add(&s, p0 + t, fh.data() + (size_t)(r + t) * HC, r + t + 1);
      s.h_last.assign(fh.data() + (size_t)(r + n - 1) * HC, fh.data() + (size_t)(r + n) * HC);
      s.h_last_valid = true;
    }
    r += n;
  }
  mtp_kv_rows(*this, er, eh.data(), ee.data());
}

// Drafts for the pending entry j = pos − 1 (stream h_last, next token tok), chaining: entry j + i takes the block's output stream and the
//   previous draft. conf[i] = softmax probability of draft i.
int QwenCpuEngine::Impl::mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf) {
  if (!mtp || k <= 0 || !s.h_last_valid || s.pos < 1) return 0;
  k = std::min(k, 7);
  const int H = c.hidden, HC = c.hcw(), V = c.vocab;
  std::vector<float> h(s.h_last), hc(HC), emb(H), x(H), y(H), xf(H), hn, dl;
  std::vector<float> logits(V);
  int64_t j = s.pos - 1;
  int32_t t = tok;
  const LayerW& L = mtp->layer;
  for (int i = 0; i < k; ++i, ++j) {
    std::vector<RowInfo> rows{{&s, j}};
    std::vector<int32_t> coords(3);
    entry_coords(s, j, coords.data());
    grow(s.mtp, j + 1 + 8, c.kv_width(), c.idx_dim, c.idx_ratio);
    embed_rows(*this, &t, 1, emb.data());
    mtp_input(emb.data(), h.data(), 1, hc.data());
    hc_mix(L.ha, hc.data(), 1, x.data(), hn, dl);
    attn_layer(L.att, 0, true, rows, coords, x.data(), y.data());
    hc_combine(L.ha, hc.data(), y.data(), 1, dl);
    hc_mix(L.hm, hc.data(), 1, x.data(), hn, dl);
    moe_layer(L.moe, 1, x.data(), y.data());
    hc_combine(L.hm, hc.data(), y.data(), 1, dl);
    hc_mix(mtp->mixer, hc.data(), 1, xf.data(), hn, dl);
    ::hive::cpu::matmul(lm_head, xf.data(), 1, H, logits.data(), V);
    int best = 0;
    for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
    double z = 0;
    for (int v = 0; v < V; ++v) z += std::exp((double)logits[v] - logits[best]);
    drafts[i] = best;
    if (conf) conf[i] = (float)(1.0 / z);
    h = hc;
    t = best;
  }
  return k;
}

}  // namespace hive::qwen::cpu
