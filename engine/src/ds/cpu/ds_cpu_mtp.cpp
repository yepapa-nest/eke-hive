// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash CPU engine — DSpark speculative drafting (reference DSparkAttention / DSparkBlock / Transformer.forward_spec, mirrored by
//   the oracle's forward_spec and the GPU Runtime's mtp_sync_seq / mtp_attention / mtp_draft).
//   · sync: the target layers' attention-input hc means of every processed position → main_x = rmsnorm(main_proj(·)) → per stage a window
//     KV ring entry (stage wkv, window RoPE, fp8 roundtrip), and the last position's hidden is kept for the next draft;
//   · draft: block [tok, noise × (B − 1)] at positions pos.. through every stage (the block rows see the whole ring [0, min(win, pos)) and the
//     whole block, non-causal), then the trunk head on the last stage's norm, the Markov bigram bias and a greedy chain, and the confidence head.
#include <algorithm>
#include <cstring>

#include "ds_cpu_internal.h"
#include "hive/clock.h"
#include "hive/common.h"

namespace hive::ds::cpu {

using ::hive::cpu::Pool;

void DsCpuEngine::Impl::mtp_sync(CpuSeq& s, const float* mh, int n, int64_t pos0) {
  if (mtp.empty() || n <= 0) return;
  const int D = c.dim, HDm = c.head_dim, win = c.window, K = (int)c.dspark_targets.size() * D;
  const MtpStage& S0 = *mtp[0];
  std::vector<float> q((size_t)n * K), mx((size_t)n * D), kv((size_t)n * HDm);
  act_quant_fp8(mh, n, K, K, q.data(), K);
  ::hive::cpu::matmul(S0.main_proj, q.data(), n, K, mx.data(), D);
  round_rows(mx.data(), n, D, D);
  rmsnorm_rows(mx.data(), D, S0.main_norm.data(), n, D, c.norm_eps, mx.data(), D);
  if (pub->spec_hook) pub->spec_hook("sync_main_x", mx.data(), mx.size());
  std::vector<float> xq((size_t)n * D);
  act_quant_fp8(mx.data(), n, D, D, xq.data(), D);
  for (size_t st_i = 0; st_i < mtp.size(); ++st_i) {
    const AttnW& A = mtp[st_i]->L.at;
    ::hive::cpu::matmul(A.wkv, xq.data(), n, D, kv.data(), HDm);
    round_rows(kv.data(), n, HDm, HDm);
    rmsnorm_rows(kv.data(), HDm, A.kv_norm.data(), n, HDm, c.norm_eps, kv.data(), HDm);
    for (int i = 0; i < n; ++i) rope_win.apply(kv.data() + (size_t)i * HDm, 1, HDm, c.rd, pos0 + i, false);
    act_quant_fp8(kv.data(), n, HDm, HDm, kv.data(), HDm);
    for (int i = 0; i < n; ++i) {
      uint16_t* dst = s.mtp_ring[st_i].data() + (size_t)((pos0 + i) % win) * HDm;
      for (int d = 0; d < HDm; ++d) dst[d] = f2b(kv[(size_t)i * HDm + d]);
    }
  }
  s.mtp_hidden.assign(mh + (size_t)(n - 1) * K, mh + (size_t)n * K);
  s.mtp_hidden_valid = true;
  s.mtp_pos = pos0 + n - 1;
}

namespace {
// DSpark stage attention: q / kv of the block rows; keys = ring slots [0, filled) + every block row
void mtp_attention(DsCpuEngine::Impl& I, const LayerW& L, const CpuSeq& s, const Fwd& f, const float* xn, float* out, int filled) {
  const DsConfig& c = I.c;
  const AttnW& A = L.at;
  const int B = f.M, D = c.dim, H = c.n_heads, HDm = c.head_dim, R = c.q_lora_rank, rd = c.rd, QW = H * HDm;
  std::vector<float> xq((size_t)B * D), qa((size_t)B * R), kv((size_t)B * HDm), qr((size_t)B * R), q((size_t)B * QW);
  act_quant_fp8(xn, B, D, D, xq.data(), D);
  std::vector<MatJob> jobs{MatJob{MatView{&A.wq_a, 0, A.wq_a.N}, qa.data(), R, xq.data(), B, D}, MatJob{MatView{&A.wkv, 0, A.wkv.N}, kv.data(), HDm, xq.data(), B, D}};
  ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
  round_rows(qa.data(), B, R, R);
  round_rows(kv.data(), B, HDm, HDm);
  rmsnorm_rows(qa.data(), R, A.q_norm.data(), B, R, c.norm_eps, qr.data(), R);
  rmsnorm_rows(kv.data(), HDm, A.kv_norm.data(), B, HDm, c.norm_eps, kv.data(), HDm);
  for (int m = 0; m < B; ++m) I.rope_win.apply(kv.data() + (size_t)m * HDm, 1, HDm, rd, f.pos[m], false);
  act_quant_fp8(kv.data(), B, HDm, HDm, kv.data(), HDm);
  act_quant_fp8(qr.data(), B, R, R, qr.data(), R);
  ::hive::cpu::matmul(A.wq_b, qr.data(), B, R, q.data(), QW);
  round_rows(q.data(), B, QW, QW);
  for (int m = 0; m < B; ++m) I.rope_win.apply(q.data() + (size_t)m * QW, H, HDm, rd, f.pos[m], false);
  const int n = filled + B;
  std::vector<float> keys((size_t)n * HDm), o((size_t)B * QW);
  const std::vector<uint16_t>& ring = s.mtp_ring[L.id - c.n_layers];
  for (int j = 0; j < filled; ++j) for (int d = 0; d < HDm; ++d) keys[(size_t)j * HDm + d] = b2f(ring[(size_t)j * HDm + d]);
  memcpy(keys.data() + (size_t)filled * HDm, kv.data(), (size_t)B * HDm * 4);
  const int HG = 4;   // block rows × 16 head groups of 4 keep the pool busy
  const float scale = 1.f / std::sqrt((float)HDm);
  Pool::get().run(B * (H / HG), [&](int t) {
    const int m = t / (H / HG), g = t % (H / HG);
    thread_local std::vector<float> ws;
    ws.resize((size_t)HG * n);
    sparse_attn_row(q.data() + (size_t)m * QW + (size_t)g * HG * HDm, HG, HDm, keys.data(), n, A.sink.data() + g * HG, scale,
                    o.data() + (size_t)m * QW + (size_t)g * HG * HDm, ws.data());
  });
  for (int m = 0; m < B; ++m) I.rope_win.apply(o.data() + (size_t)m * QW, H, HDm, rd, f.pos[m], true);
  const int G = c.o_groups, OR = c.o_lora_rank, sub = QW / G;
  std::vector<float> og((size_t)B * G * OR);
  jobs.clear();
  for (int g = 0; g < G; ++g) jobs.push_back(MatJob{MatView{&A.wo_a, g * OR, (g + 1) * OR}, og.data() + (size_t)g * OR, G * OR, o.data() + (size_t)g * sub, B, QW});
  ::hive::cpu::matmul_multi(jobs.data(), (int)jobs.size(), nullptr, 0, 0);
  round_rows(og.data(), B, G * OR, G * OR);
  act_quant_fp8(og.data(), B, G * OR, G * OR, og.data(), G * OR);
  ::hive::cpu::matmul(A.wo_b, og.data(), B, G * OR, out, D);
  round_rows(out, B, D, D);
}
}  // namespace

int DsCpuEngine::Impl::mtp_draft(CpuSeq& s, int32_t tok, int32_t* drafts, float* conf) {
  HIVE_CHECK(!mtp.empty() && s.mtp_hidden_valid && s.mtp_pos == s.pos - 1, "ds cpu mtp_draft needs the hidden of pos − 1");
  const double t0 = hive::mono_ms();
  const int B = c.dspark_block, D = c.dim, HC = c.hc, HD = HC * D, V = c.vocab, rank = c.dspark_markov_rank, win = c.window;
  const int64_t pos0 = s.pos;
  const int filled = (int)std::min<int64_t>(win, pos0);
  Fwd f;
  f.M = B;
  for (int i = 0; i < B; ++i) { f.pos.push_back(pos0 + i); f.ids.push_back(i == 0 ? tok : c.dspark_noise_token); f.img.push_back(0); f.part_of.push_back(0); }
  f.parts.push_back(Part{&s, pos0, 0, B});
  f.h.resize((size_t)B * HD);
  f.premix.assign((size_t)B * HC, 0.f);
  for (int m = 0; m < B; ++m) {
    const uint16_t* e = embed.data() + (size_t)f.ids[m] * D;
    for (int cc = 0; cc < HC; ++cc) for (int d = 0; d < D; ++d) f.h[(size_t)m * HD + (size_t)cc * D + d] = b2f(e[d]);
    f.premix[(size_t)m * HC] = 1.f;
  }
  std::vector<float> pre_a((size_t)B * HC), post_a((size_t)B * HC), comb_a((size_t)B * HC * HC), xn((size_t)B * D), y((size_t)B * D);
  std::vector<float> pre_f((size_t)B * HC), post_f((size_t)B * HC), comb_f((size_t)B * HC * HC);
  for (size_t si = 0; si < mtp.size(); ++si) {
    LayerW& L = mtp[si]->L;
    hc_front(L, f, false, f.premix.data(), pre_a.data(), post_a.data(), comb_a.data(), xn.data());
    mtp_attention(*this, L, s, f, xn.data(), y.data(), filled);
    hc_post(f, y.data(), post_a.data(), comb_a.data());
    hc_front(L, f, true, pre_a.data(), pre_f.data(), post_f.data(), comb_f.data(), xn.data());
    moe(L, f, xn.data(), y.data());
    hc_post(f, y.data(), post_f.data(), comb_f.data());
    f.premix = pre_f;
    if (pub->spec_hook) {
      pub->spec_hook(("spec_h_M" + std::to_string(si)).c_str(), f.h.data(), f.h.size());
      pub->spec_hook(("spec_premix_M" + std::to_string(si)).c_str(), f.premix.data(), f.premix.size());
    }
  }
  const MtpStage& Ls = *mtp.back();
  std::vector<float> x((size_t)B * D), logits((size_t)B * V), emb((size_t)B * rank), bias(V);
  head_rows(f.h.data(), f.premix.data(), B, Ls.norm, logits.data(), x.data());
  std::vector<int32_t> out(B + 1);
  out[0] = tok;
  for (int i = 0; i < B; ++i) {
    float* e = emb.data() + (size_t)i * rank;
    for (int r = 0; r < rank; ++r) e[r] = b2f(Ls.markov_embed[(size_t)out[i] * rank + r]);
    ::hive::cpu::matmul(Ls.markov_head, e, 1, rank, bias.data(), V);
    float* lg = logits.data() + (size_t)i * V;
    int am = 0;
    for (int v = 0; v < V; ++v) { lg[v] += bias[v]; if (lg[v] > lg[am]) am = v; }
    out[i + 1] = am;
  }
  for (int i = 0; i < B; ++i) {
    float acc = 0.f;
    for (int d = 0; d < D; ++d) acc += x[(size_t)i * D + d] * Ls.conf[d];
    for (int r = 0; r < rank; ++r) acc += emb[(size_t)i * rank + r] * Ls.conf[D + r];
    conf[i] = acc;
    drafts[i] = out[i + 1];
  }
  if (pub->spec_hook) {
    pub->spec_hook("spec_x", x.data(), x.size());
    pub->spec_hook("spec_markov", emb.data(), emb.size());
    pub->spec_hook("spec_logits", logits.data(), logits.size());
    std::vector<float> idf(out.begin(), out.end());
    pub->spec_hook("spec_ids", idf.data(), idf.size());
    pub->spec_hook("spec_conf", conf, B);
  }
  st.ms_mtp += hive::mono_ms() - t0;
  return B;
}

}  // namespace hive::ds::cpu
