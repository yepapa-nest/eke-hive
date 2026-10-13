// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B CPU engine — the MTP block (mtp_block of oracle/q27_oracle.py). Entry j of a sequence = (residual stream h_j before the final
//   norm, token x_{j+1}) at rope position j; its attention layer has its own KV cache, so every entry whose next token is known is written
//   once (K / V only) and a draft runs the whole block for the pending entry and chains its own output stream for further drafts.
//   Bookkeeping adapted from the Flash-Next CPU engine (src/qwen/cpu/qwen_cpu_mtp.cpp).
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "q27_cpu_internal.h"

namespace hive::q27::cpu {

namespace {
void grow(CpuSeq::Attn& a, int64_t tokens, int kvw) {
  if ((int64_t)a.k.size() < tokens * kvw) {
    const size_t n = (size_t)std::max<int64_t>(tokens, (int64_t)a.k.size() / kvw * 3 / 2) * kvw;
    a.k.resize(n); a.v.resize(n);
  }
}
// rope coordinates of MTP entry j: the sequence's own record when the position was already processed, else text positions
void entry_coords(const CpuSeq& s, int64_t j, int32_t* o) {
  if ((int64_t)s.rope.size() >= (j + 1) * 3) { memcpy(o, s.rope.data() + (size_t)j * 3, 12); return; }
  o[0] = o[1] = o[2] = (int32_t)(j + s.rope_delta);
}
}  // namespace

// h [n, 5120] = bf16(fc([n_e(emb) | n_h(h_rows)]))
//   With HIVE_Q27_MTP_INPUT=post the hidden input is first passed through the norm of the stream it came from (prenorm: the final norm for
//   main-model streams, mtp.norm for a chained draft) — the alternative input definition, kept for the A/B in docs/q27.md.
void Q27CpuEngine::Impl::mtp_input(const float* emb, const float* h_rows, int n, float* h, const std::vector<float>* prenorm) {
  const int H = c.hidden;
  std::vector<float> cat((size_t)n * 2 * H), hn;
  norm1p(emb, H, mtp->pre_emb_norm.data(), n, H, H, c.rms_eps, cat.data(), 2 * H);
  if (mtp_post) {
    hn.resize((size_t)n * H);
    norm1p(h_rows, H, (prenorm ? *prenorm : final_norm).data(), n, H, H, c.rms_eps, hn.data(), H);
    h_rows = hn.data();
  }
  norm1p(h_rows, H, mtp->pre_hidden_norm.data(), n, H, H, c.rms_eps, cat.data() + H, 2 * H);
  lin(mtp->fc, cat.data(), n, 2 * H, h, H);
}

namespace {
// K / V writes of MTP entries rows[i] from streams h [n, H] and next-token embeddings emb [n, H]
void mtp_kv_rows(Q27CpuEngine::Impl& I, const std::vector<RowInfo>& rows, const float* h, const float* emb) {
  const Q27Config& c = I.c;
  const int n = (int)rows.size(), H = c.hidden;
  if (n == 0) return;
  std::vector<float> hm((size_t)n * H), x((size_t)n * H);
  std::vector<int32_t> coords((size_t)n * 3);
  for (int r = 0; r < n; ++r) {
    grow(rows[r].s->mtp, rows[r].pos + 1 + 8, c.kv_width());
    entry_coords(*rows[r].s, rows[r].pos, coords.data() + (size_t)r * 3);
  }
  I.mtp_input(emb, h, n, hm.data());
  norm1p(hm.data(), H, I.mtp->layer.ln1.data(), n, H, H, c.rms_eps, x.data(), H);
  I.attn_layer(I.mtp->layer.att, 0, true, rows, coords.data(), x.data(), nullptr, true);
  for (const RowInfo& ri : rows) ri.s->mtp_written = std::max(ri.s->mtp_written, ri.pos + 1);
}
}  // namespace

void Q27CpuEngine::Impl::mtp_kv(CpuSeq& s, int64_t pos0, int n, const float* h_rows, const int32_t* next_ids) {
  if (n <= 0) return;
  std::vector<RowInfo> rows(n);
  for (int i = 0; i < n; ++i) rows[i] = {&s, pos0 + i};
  std::vector<float> emb((size_t)n * c.hidden);
  embed_rows(next_ids, n, emb.data());
  mtp_kv_rows(*this, rows, h_rows, emb.data());
}

// After a main forward (fh = its residual streams): writes the entries whose next token became known. Verify rows only write the entry in
//   front of each part (the part's first token is always kept); rollback_batch writes the accepted ones.
void Q27CpuEngine::Impl::mtp_after_forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill) {
  const int M = (int)rows.size(), H = c.hidden;
  std::vector<RowInfo> er;
  std::vector<float> eh, ee;
  auto add = [&](CpuSeq* s, int64_t j, const float* h, int r_next) {
    er.push_back({s, j});
    eh.insert(eh.end(), h, h + H);
    const size_t o = ee.size();
    ee.resize(o + H);
    embed_rows(ids + r_next, 1, ee.data() + o);
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
      for (int t = 0; t + 1 < n; ++t) add(&s, p0 + t, fh.data() + (size_t)(r + t) * H, r + t + 1);
      s.h_last.assign(fh.data() + (size_t)(r + n - 1) * H, fh.data() + (size_t)(r + n) * H);
      s.h_last_valid = true;
    }
    r += n;
  }
  mtp_kv_rows(*this, er, eh.data(), ee.data());
}

// Drafts for the pending entry j = pos − 1 (stream h_last, next token tok), chaining: entry j + i takes the block's output stream and the
//   previous draft. conf[i] = softmax probability of draft i.
int Q27CpuEngine::Impl::mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf) {
  if (!mtp || k <= 0 || !s.h_last_valid || s.pos < 1) return 0;
  k = std::min(k, 7);
  const int H = c.hidden, V = c.vocab;
  std::vector<float> h(s.h_last), hm(H), emb(H), logits(V);
  int64_t j = s.pos - 1;
  int32_t t = tok;
  for (int i = 0; i < k; ++i, ++j) {
    std::vector<RowInfo> rows{{&s, j}};
    int32_t coords[3];
    entry_coords(s, j, coords);
    grow(s.mtp, j + 1 + 8, c.kv_width());
    embed_rows(&t, 1, emb.data());
    mtp_input(emb.data(), h.data(), 1, hm.data(), i == 0 ? nullptr : &mtp->norm);
    layer(-1, mtp->layer, true, rows, coords, hm.data());
    head(hm.data(), 1, mtp->norm, logits.data());
    if (i == 0) draft_logits = logits;
    int best = 0;
    for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
    double z = 0;
    for (int v = 0; v < V; ++v) z += std::exp((double)logits[v] - logits[best]);
    drafts[i] = best;
    if (conf) conf[i] = (float)(1.0 / z);
    h = hm;
    t = best;
  }
  return k;
}

}  // namespace hive::q27::cpu
