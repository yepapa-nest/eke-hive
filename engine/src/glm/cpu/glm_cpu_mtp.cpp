// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash CPU engine — the NextN (MTP) layer, as engine/src/glm/glm_mtp.cpp: entry j of a sequence pairs the main model's final normed
//   hidden h_j with token x_{j+1} (embedding zeroed for j = 0); x0 = eh_proj([enorm(emb) | hnorm(h)]); one plain-residual layer: DSA attention
//   with its own cache, MoE of 288 FP8 experts + shared expert; shared_head.norm → the main lm_head. The normed output is the next draft's h.
//   Entries whose next token is known are written once (latent / indexer cache only); a draft runs the whole layer for the pending entry.
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "glm_cpu_internal.h"
#include "hive/clock.h"

namespace hive::glm::cpu {

namespace {
void embed_rows(const GlmCpuEngine::Impl& I, const int32_t* ids, const int64_t* pos, int n, float* out) {
  const int H = I.c.hidden;
  for (int r = 0; r < n; ++r) {
    if (pos[r] == 0) { std::fill(out + (size_t)r * H, out + (size_t)(r + 1) * H, 0.f); continue; }
    const uint16_t* e = I.embed.data() + (size_t)ids[r] * H;
    for (int d = 0; d < H; ++d) out[(size_t)r * H + d] = b2f(e[d]);
  }
}
}  // namespace

// x0 [n, H] = bf16(eh_proj([bf16(rmsnorm(emb)·enorm) | bf16(rmsnorm(h)·hnorm)]))
void GlmCpuEngine::Impl::mtp_input(const float* emb, const float* h_rows, int n, float* x0) {
  const int H = c.hidden;
  std::vector<float> cat((size_t)n * 2 * H);
  rmsnorm_rows(emb, H, mtp->enorm.data(), n, H, c.rms_eps, cat.data(), 2 * H);
  rmsnorm_rows(h_rows, H, mtp->hnorm.data(), n, H, c.rms_eps, cat.data() + H, 2 * H);
  lin(mtp->eh, cat.data(), n, 2 * H, x0, H);
}

void GlmCpuEngine::Impl::mtp_kv(CpuSeq& s, int64_t pos0, int n, const float* h_rows, const int32_t* next_ids) {
  if (n <= 0) return;
  const int H = c.hidden;
  std::vector<RowInfo> rows(n);
  std::vector<int64_t> pos(n);
  for (int i = 0; i < n; ++i) { rows[i] = {&s, pos0 + i}; pos[i] = pos0 + i; }
  std::vector<float> emb((size_t)n * H), x0((size_t)n * H), xn((size_t)n * H);
  embed_rows(*this, next_ids, pos.data(), n, emb.data());
  mtp_input(emb.data(), h_rows, n, x0.data());
  rmsnorm_rows(x0.data(), H, mtp->in_norm.data(), n, H, c.rms_eps, xn.data(), H);
  dsa_layer(mtp->dw, 0, true, rows, xn.data(), nullptr, true);
  s.mtp_written = std::max(s.mtp_written, pos0 + n);
}

// After a main forward (fh = final normed hidden of every row): the entries whose next token became known. Verify rows only write the entry in
//   front of each part; rollback_batch writes the accepted ones.
void GlmCpuEngine::Impl::mtp_after_forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill) {
  const int M = (int)rows.size(), H = c.hidden;
  for (int r = 0; r < M;) {
    CpuSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < M && rows[r + n].s == &s && rows[r + n].pos == rows[r].pos + n) ++n;
    const int64_t p0 = rows[r].pos;
    if (s.h_last_valid && p0 > 0 && s.mtp_written < p0) mtp_kv(s, p0 - 1, 1, s.h_last.data(), ids + r);
    if (!verify) {
      if (n > 1) mtp_kv(s, p0, n - 1, fh.data() + (size_t)r * H, ids + r + 1);
      std::memcpy(s.h_last.data(), fh.data() + (size_t)(r + n - 1) * H, (size_t)H * 4);
      s.h_last_valid = true;
    }
    r += n;
  }
  (void)prefill;
}

int GlmCpuEngine::Impl::mtp_draft(CpuSeq& s, int32_t tok, int k, int32_t* drafts, float* conf, float* logits0) {
  if (!mtp || mtp_k <= 0 || !s.h_last_valid || k <= 0) return 0;
  k = std::min(k, mtp_k);
  const int64_t p = s.pos - 1;
  if (p < 0 || p + k + 1 > s.cap) return 0;
  const double t0 = hive::mono_ms();
  const int H = c.hidden;
  const bool dv = !draft_rows.empty();
  const int V = dv ? (int)draft_rows.size() : c.vocab;
  std::vector<float> h(s.h_last), emb(H), x0(H), xn(H), y(H), h1(H), mo(H), hout(H), logits(V);
  int32_t t = tok;
  for (int i = 0; i < k; ++i) {
    const int64_t j = p + i;
    std::vector<RowInfo> rows{{&s, j}};
    embed_rows(*this, &t, &j, 1, emb.data());
    mtp_input(emb.data(), h.data(), 1, x0.data());
    rmsnorm_rows(x0.data(), H, mtp->in_norm.data(), 1, H, c.rms_eps, xn.data(), H);
    dsa_layer(mtp->dw, 0, true, rows, xn.data(), y.data());
    for (int d = 0; d < H; ++d) h1[d] = bf16r(x0[d] + y[d]);
    rmsnorm_rows(h1.data(), H, mtp->post_norm.data(), 1, H, c.rms_eps, xn.data(), H);
    moe(mtp->mw, 1, xn.data(), mo.data());
    for (int d = 0; d < H; ++d) mo[d] = bf16r(mo[d] + h1[d]);
    rmsnorm_rows(mo.data(), H, mtp->head_norm.data(), 1, H, c.rms_eps, hout.data(), H);
    ::hive::cpu::matmul(dv ? draft_head : lm_head, hout.data(), 1, H, logits.data(), V);
    if (i == 0 && logits0 && !dv) std::memcpy(logits0, logits.data(), (size_t)V * 4);
    int best = 0;
    for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
    double z = 0;
    for (int v = 0; v < V; ++v) z += std::exp((double)logits[v] - logits[best]);
    drafts[i] = dv ? draft_rows[best] : best;
    if (conf) conf[i] = (float)(1.0 / z);
    h = hout;
    t = drafts[i];
  }
  s.mtp_written = std::max(s.mtp_written, p + 1);  // the first entry used the real hidden: exact
  st.ms_mtp += hive::mono_ms() - t0;
  return k;
}

}  // namespace hive::glm::cpu
