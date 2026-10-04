// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash forward validation against a transformers reference (ref.bin from glm_ref.py + glm_ref2bin.py):
//   1) prefill each prompt, compare the hc streams after every layer (relative L2 error, cosine) and the last-token top-20 logits;
//   2) decode path: prefill T-1 tokens, decode the last one, compare its logits with the prefill logits of step 1;
//   3) greedy continuation of a few tokens (ids printed for detokenization).
// usage: glm_check <ckpt dir> <ref.bin> [cache GiB = 40] [cpu threads = 24]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <vector>

#include <cuda_fp16.h>

#include "hive/glm/glm_engine.h"

using namespace hive;
using namespace hive::glm;

static float h2f(uint16_t h) { __half x; std::memcpy(&x, &h, 2); return __half2float(x); }
static float b2f(bf16 b) { return __bfloat162float(b); }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: glm_check <ckpt> <ref.bin> [cache GiB] [threads]\n"); return 2; }
  const double cache_gib = argc > 3 ? atof(argv[3]) : 40.0;
  const int threads = argc > 4 ? atoi(argv[4]) : 24;
  std::ifstream f(argv[2], std::ios::binary);
  char magic[8]; f.read(magic, 8);
  if (std::memcmp(magic, "GLMREF1", 7)) { fprintf(stderr, "bad ref file\n"); return 2; }
  int32_t hdr[4]; f.read((char*)hdr, 16);
  const int P = hdr[0], L = hdr[1], HC = hdr[2], H = hdr[3];
  struct Ref { std::vector<int32_t> ids; std::vector<uint16_t> layers; std::vector<uint16_t> top_v; std::vector<int32_t> top_i; };
  std::vector<Ref> refs(P);
  for (auto& r : refs) {
    int32_t T; f.read((char*)&T, 4);
    r.ids.resize(T); f.read((char*)r.ids.data(), 4 * T);
    r.layers.resize((size_t)L * T * HC * H); f.read((char*)r.layers.data(), r.layers.size() * 2);
    r.top_v.resize(20); f.read((char*)r.top_v.data(), 40);
    r.top_i.resize(20); f.read((char*)r.top_i.data(), 80);
  }
  printf("ref: %d prompts · %d layers · hc %d · hidden %d\n", P, L, HC, H);

  GlmModel model(argv[1], getenv("GLM_CHECK_MTP") && atoi(getenv("GLM_CHECK_MTP")) > 0);
  GlmExperts experts(model, 0, threads, 8);
  experts.load_all(threads);
  experts.alloc_cache((size_t)(cache_gib * 1073741824.0));
  int max_T = 0; for (auto& r : refs) max_T = std::max<int>(max_T, r.ids.size());
  GlmEngine eng(model, experts, getenv("GLM_CHECK_CHUNK") ? atoi(getenv("GLM_CHECK_CHUNK")) : std::max(64, max_T));
  const int V = model.cfg().vocab;
  int fails = 0;
  for (int p = 0; p < P; ++p) {
    Ref& r = refs[p];
    const int T = r.ids.size();
    printf("\n== prompt %d (T=%d)\n", p, T);
    std::vector<bf16> hbuf;
    double worst = 0;
    eng.layer_hook = [&](int l, const bf16* hd, int rows) {
      if (l >= L) return;
      hbuf.resize((size_t)rows * HC * H);
      CUDA_CHECK(cudaMemcpy(hbuf.data(), hd, hbuf.size() * 2, cudaMemcpyDeviceToHost));
      const uint16_t* ref = r.layers.data() + (size_t)l * T * HC * H;
      double num = 0, den = 0, dot = 0, na = 0, nb = 0, last_num = 0, last_den = 0;
      for (size_t i = 0; i < hbuf.size(); ++i) {
        const double a = b2f(hbuf[i]), b = h2f(ref[i]);
        num += (a - b) * (a - b); den += b * b; dot += a * b; na += a * a; nb += b * b;
        if (i >= (size_t)(rows - 1) * HC * H) { last_num += (a - b) * (a - b); last_den += b * b; }
      }
      const double rel = std::sqrt(num / std::max(den, 1e-30)), cos = dot / std::sqrt(std::max(na * nb, 1e-30));
      worst = std::max(worst, rel);
      printf("  layer %2d  rel %.4f  cos %.6f  last-row rel %.4f\n", l, rel, cos, std::sqrt(last_num / std::max(last_den, 1e-30)));
    };
    // keep the last row of every layer from the full prefill (decode-path comparison below)
    std::vector<std::vector<bf16>> last_rows(model.cfg().n_layers);
    auto ref_hook = eng.layer_hook;
    eng.layer_hook = [&](int l, const bf16* hd, int rows) {
      if (ref_hook) ref_hook(l, hd, rows);
      last_rows[l].resize((size_t)HC * H);
      CUDA_CHECK(cudaMemcpy(last_rows[l].data(), hd + (size_t)(rows - 1) * HC * H, (size_t)HC * H * 2, cudaMemcpyDeviceToHost));
    };
    auto s = eng.new_seq(4096);
    std::vector<float> logits(V), logits2(V);
    eng.prefill(*s, r.ids.data(), T, logits.data());
    eng.layer_hook = nullptr;
    // top-20 comparison
    std::vector<int> ord(V); for (int i = 0; i < V; ++i) ord[i] = i;
    std::partial_sort(ord.begin(), ord.begin() + 20, ord.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    int overlap = 0; for (int i = 0; i < 20; ++i) for (int j = 0; j < 20; ++j) overlap += ord[i] == r.top_i[j];
    double max_dv = 0; for (int j = 0; j < 20; ++j) max_dv = std::max(max_dv, (double)std::fabs(logits[r.top_i[j]] - h2f(r.top_v[j])));
    const bool top1 = ord[0] == r.top_i[0];
    printf("  logits: top1 %s (ours %d, ref %d) · top20 overlap %d/20 · max |Δlogit| on ref top20 %.3f · worst layer rel %.4f\n",
           top1 ? "MATCH" : "DIFF", ord[0], r.top_i[0], overlap, max_dv, worst);
    if (!top1) ++fails;
    // decode path vs prefill
    auto s2 = eng.new_seq(4096);
    eng.prefill(*s2, r.ids.data(), T - 1, nullptr);
    GlmSeq* sp[1] = {s2.get()};
    const bool verbose = getenv("GLM_CHECK_DECODE_LAYERS") != nullptr;
    eng.layer_hook = [&](int l, const bf16* hd, int rows) {
      std::vector<bf16> d((size_t)HC * H);
      CUDA_CHECK(cudaMemcpy(d.data(), hd, d.size() * 2, cudaMemcpyDeviceToHost));
      double num = 0, den = 0;
      for (size_t i = 0; i < d.size(); ++i) { const double a = b2f(d[i]), b = b2f(last_rows[l][i]); num += (a - b) * (a - b); den += b * b; }
      if (verbose) printf("  decode layer %2d  rel vs prefill %.5f\n", l, std::sqrt(num / std::max(den, 1e-30)));
    };
    eng.decode(sp, r.ids.data() + T - 1, 1, logits2.data());
    eng.layer_hook = nullptr;
    double dmax = 0, dn = 0, dd = 0; int a1 = 0;
    for (int i = 0; i < V; ++i) { dmax = std::max(dmax, (double)std::fabs(logits[i] - logits2[i])); dn += std::pow(logits[i] - logits2[i], 2); dd += logits[i] * logits[i]; if (logits2[i] > logits2[a1]) a1 = i; }
    printf("  decode-vs-prefill: argmax %s · max |Δ| %.4f · rel %.5f\n", a1 == ord[0] ? "MATCH" : "DIFF", dmax, std::sqrt(dn / dd));
    if (a1 != ord[0]) ++fails;
    // greedy continuation
    printf("  greedy:");
    int tok = ord[0];
    const int n_greedy = getenv("GLM_CHECK_GREEDY") ? atoi(getenv("GLM_CHECK_GREEDY")) : 16;
    const auto c0 = experts.stats();
    eng.stats() = GlmForwardStats{};
    const auto t0 = std::chrono::steady_clock::now();
    int produced = 0;
    const int mtp_k = getenv("GLM_CHECK_MTP") ? atoi(getenv("GLM_CHECK_MTP")) : 0;
    if (mtp_k > 0 && eng.mtp_on()) {  // greedy with speculative drafts: verify rows, accept while argmax == draft
      long steps = 0, drafted = 0, accepted = 0;
      double ms_draft = 0, ms_verify = 0, ms_rollback = 0;
      auto tnow = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
      std::vector<float> rows((size_t)8 * V);
      while (produced < n_greedy) {
        printf(" %d", tok); fflush(stdout); ++produced;
        bool eos = false; for (int e : model.cfg().eos_ids) eos |= tok == e;
        if (eos || produced >= n_greedy) break;
        int32_t d[8]; float cf[8];
        double ta = tnow();
        const int k = eng.mtp_draft(*s, tok, mtp_k, d, cf);
        ms_draft += tnow() - ta;
        if (k == 0) {
          GlmSeq* q[1] = {s.get()};
          eng.decode(q, &tok, 1, logits.data());
          int b = 0; for (int i = 1; i < V; ++i) if (logits[i] > logits[b]) b = i;
          tok = b; continue;
        }
        int32_t ids[8]; ids[0] = tok; for (int i = 0; i < k; ++i) ids[i + 1] = d[i];
        ta = tnow();
        eng.verify(*s, ids, k + 1, rows.data());
        ms_verify += tnow() - ta;
        auto am = [&](int r) { const float* L = rows.data() + (size_t)r * V; int b = 0; for (int i = 1; i < V; ++i) if (L[i] > L[b]) b = i; return b; };
        int n_keep = 1, next = am(0);
        for (int i = 0; i < k && produced < n_greedy; ++i) {
          if (next != d[i]) break;
          printf(" %d", d[i]); ++produced; ++n_keep;
          bool e2 = false; for (int e : model.cfg().eos_ids) e2 |= d[i] == e;
          next = am(i + 1);
          if (e2) { produced = n_greedy; break; }
        }
        ta = tnow();
        eng.rollback(*s, n_keep);
        ms_rollback += tnow() - ta;
        ++steps; drafted += k; accepted += n_keep - 1;
        tok = next;
      }
      printf("\n  mtp: k %d · steps %ld · drafted %ld · accepted %ld (%.1f %%) · %.2f tokens per verify step · per step: draft %.2f verify %.2f rollback %.2f ms", mtp_k, steps, drafted, accepted,
             100.0 * accepted / std::max(1L, drafted), (double)(accepted + steps) / std::max(1L, steps), ms_draft / std::max(1L, steps),
             ms_verify / std::max(1L, steps), ms_rollback / std::max(1L, steps));
    } else
    for (int k = 0; k < n_greedy; ++k, ++produced) {
      printf(" %d", tok); fflush(stdout);
      bool eos = false; for (int e : model.cfg().eos_ids) eos |= tok == e;
      if (eos) break;
      GlmSeq* q[1] = {s.get()};
      eng.decode(q, &tok, 1, logits.data());
      int b = 0; for (int i = 1; i < V; ++i) if (logits[i] > logits[b]) b = i;
      tok = b;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto c1 = experts.stats();
    printf("\n  decode layers: per token %.2f ms in decode_layer · x to host %.2f · cpu experts %.2f (phase1 %.2f phase2 %.2f) over %.1f layers with CPU work",
           (c1.ms_layer - c0.ms_layer) / std::max(1, produced), (c1.ms_wait_x - c0.ms_wait_x) / std::max(1, produced), (c1.ms_cpu - c0.ms_cpu) / std::max(1, produced),
           (c1.ms_p1 - c0.ms_p1) / std::max(1, produced), (c1.ms_p2 - c0.ms_p2) / std::max(1, produced), (double)(c1.cpu_layers - c0.cpu_layers) / std::max(1, produced));
    printf("\n  decode: %d tokens in %.2f s = %.2f tok/s · hit %.1f %% (routed %llu, cpu %llu, promoted %llu, prefetched %llu)\n", produced, sec, produced / sec,
           100.0 * (c1.hit - c0.hit) / std::max<double>(1, c1.routed - c0.routed), (unsigned long long)(c1.routed - c0.routed),
           (unsigned long long)(c1.cpu - c0.cpu), (unsigned long long)(c1.promoted - c0.promoted), (unsigned long long)(c1.prefetched - c0.prefetched));
    // GLM_CHECK_BATCHV=N (first prompt only): batched verify of two sequences (verify_batch + rollback_batch) against the same two sequences
    //   verified one by one (verify + rollback), N steps of greedy drafting k = 2. Pairs: A = the whole prompt, B = the prompt minus 5 tokens.
    //   Residency differs run to run (CPU vs GPU for a missed expert), so rows are compared by argmax and relative error, not bit for bit.
    static int batchv_done = 0;
    const int batchv = getenv("GLM_CHECK_BATCHV") ? atoi(getenv("GLM_CHECK_BATCHV")) : 0;
    if (batchv > 0 && !batchv_done++ && eng.mtp_on() && T > 8) {
      eng.enable_batch_verify(2);
      auto A1 = eng.new_seq(4096), B1 = eng.new_seq(4096), A2 = eng.new_seq(4096), B2 = eng.new_seq(4096), A3 = eng.new_seq(4096), B3 = eng.new_seq(4096);
      std::vector<float> la(V), lb(V);
      eng.prefill(*A1, r.ids.data(), T, la.data()); eng.prefill(*A2, r.ids.data(), T, nullptr);
      eng.prefill(*B1, r.ids.data(), T - 5, lb.data()); eng.prefill(*B2, r.ids.data(), T - 5, nullptr);
      eng.prefill(*A3, r.ids.data(), T, nullptr); eng.prefill(*B3, r.ids.data(), T - 5, nullptr);  // control: one-by-one again
      auto am = [&](const float* L) { int b = 0; for (int i = 1; i < V; ++i) if (L[i] > L[b]) b = i; return b; };
      int ta = am(la.data()), tb = am(lb.data());
      std::vector<float> rb((size_t)8 * V), r1((size_t)4 * V), r2((size_t)4 * V), r3((size_t)4 * V), r4((size_t)4 * V);
      double worst_part[2] = {0, 0}, worst_ctl = 0; long ctl_argmax_diff = 0;
      long rows_cmp = 0, rows_argmax_diff = 0, keep_diff = 0, tok_diff = 0; double worst_rel = 0;
      // the one-by-one pair follows the batched pair (same drafts, same accepted lengths, same next tokens), so every step compares the
      //   same inputs; it only checks that the batched rows match the rows of separate verifies
      for (int step = 0; step < batchv; ++step) {
        int32_t da[4], db[4], dx[4]; float cf[4];
        const int ka = eng.mtp_draft(*A1, ta, 2, da, cf), kb = eng.mtp_draft(*B1, tb, 2, db, cf);
        eng.mtp_draft(*A2, ta, 2, dx, cf); eng.mtp_draft(*B2, tb, 2, dx, cf);  // same MTP cache writes as the batched pair
        eng.mtp_draft(*A3, ta, 2, dx, cf); eng.mtp_draft(*B3, tb, 2, dx, cf);
        if (ka != 2 || kb != 2) { printf("\n  batchv: draft count %d %d — stop", ka, kb); break; }
        int32_t ids[6] = {ta, da[0], da[1], tb, db[0], db[1]};
        GlmSeq* ps[2] = {A1.get(), B1.get()}; const int Ms[2] = {3, 3};
        eng.verify_batch(ps, Ms, 2, ids, rb.data());
        auto accept = [&](const float* rows, const int32_t* d, int& next) { int keep = 1; next = am(rows); for (int i = 0; i < 2; ++i) { if (next != d[i]) break; ++keep; next = am(rows + (size_t)(i + 1) * V); } return keep; };
        int na = 0, nb = 0; const int kak = accept(rb.data(), da, na), kbk = accept(rb.data() + (size_t)3 * V, db, nb);
        const int keeps[2] = {kak, kbk};
        eng.rollback_batch(keeps, 2);
        eng.verify(*A2, ids, 3, r1.data());
        int na2 = 0; keep_diff += accept(r1.data(), da, na2) != kak; tok_diff += na2 != na;
        eng.rollback(*A2, kak);
        eng.verify(*B2, ids + 3, 3, r2.data());
        int nb2 = 0; keep_diff += accept(r2.data(), db, nb2) != kbk; tok_diff += nb2 != nb;
        eng.rollback(*B2, kbk);
        eng.verify(*A3, ids, 3, r3.data()); eng.rollback(*A3, kak);
        eng.verify(*B3, ids + 3, 3, r4.data()); eng.rollback(*B3, kbk);
        auto rel = [&](const float* x, const float* y) { double num = 0, den = 0; for (int i = 0; i < V; ++i) { num += (double)(x[i] - y[i]) * (x[i] - y[i]); den += (double)y[i] * y[i]; } return std::sqrt(num / std::max(den, 1e-30)); };
        for (int row = 0; row < 6; ++row) {  // control: one-by-one vs one-by-one (the run-to-run noise of the same path)
          const float* x = row < 3 ? r3.data() + (size_t)row * V : r4.data() + (size_t)(row - 3) * V;
          const float* y = row < 3 ? r1.data() + (size_t)row * V : r2.data() + (size_t)(row - 3) * V;
          worst_ctl = std::max(worst_ctl, rel(x, y)); ctl_argmax_diff += am(x) != am(y);
          const float* b = rb.data() + (size_t)row * V;
          worst_part[row / 3] = std::max(worst_part[row / 3], rel(b, y));
        }
        for (int row = 0; row < 6; ++row) {
          const float* x = rb.data() + (size_t)row * V;
          const float* y = row < 3 ? r1.data() + (size_t)row * V : r2.data() + (size_t)(row - 3) * V;
          double num = 0, den = 0; for (int i = 0; i < V; ++i) { num += (double)(x[i] - y[i]) * (x[i] - y[i]); den += (double)y[i] * y[i]; }
          worst_rel = std::max(worst_rel, std::sqrt(num / std::max(den, 1e-30))); ++rows_cmp; rows_argmax_diff += am(x) != am(y);
        }
        ta = na; tb = nb;
      }
      printf("\n  batchv: %d steps · rows compared %ld · argmax differs %ld · worst row rel %.5f · accepted-length differs %ld · next token differs %ld · positions A %lld/%lld B %lld/%lld",
             batchv, rows_cmp, rows_argmax_diff, worst_rel, keep_diff, tok_diff, (long long)A1->pos, (long long)A2->pos, (long long)B1->pos, (long long)B2->pos);
      printf("\n  batchv control (one-by-one twice): worst row rel %.5f · argmax differs %ld · batched vs one-by-one worst rel part A %.5f · part B %.5f",
             worst_ctl, ctl_argmax_diff, worst_part[0], worst_part[1]);
      // pass = the batched rows differ from separate verifies no more than separate verifies differ from each other (2× margin)
      if (worst_rel > std::max(0.05, 2.0 * worst_ctl) || rows_argmax_diff > std::max(rows_cmp / 10, 2 * ctl_argmax_diff + 1) || A1->pos != A2->pos || B1->pos != B2->pos) {
        ++fails; printf(" · FAIL");
      }
    }
  }
  {
    auto& st = eng.stats();
    if (st.steps > 0) {
      const double n = st.steps;
      printf("\nper decode step (HIVE_GLM_PROF, last prompt greedy only; %d steps): embed %.2f hc %.2f kda %.2f dsa %.2f dense %.2f router %.2f shared %.2f experts %.2f head %.2f ms\n",
             st.steps, st.ms_embed / n, st.ms_hc / n, st.ms_kda / n, st.ms_dsa / n, st.ms_dense / n, st.ms_router / n, st.ms_shared / n, st.ms_experts / n, st.ms_head / n);
      printf("  kda split: projections %.2f · recurrence %.2f · o_proj+rest %.2f ms · prediction %.2f ms (GPU time with HIVE_GLM_PROF=2)\n", st.ms_kda_proj / n, st.ms_kda_core / n, st.ms_kda / n, st.ms_predict / n);
    }
  }
  {
    auto& ps = eng.predict_stats();
    for (int d = 1; d <= 2; ++d)
      if (ps.pairs[d])
        printf("predict next-%d MoE layer: recall %.1f %% of (row, expert) choices · misses %.1f %% of choices · misses predicted %.1f %% · prefetch candidates %.2f per layer-step, used %.1f %%\n", d,
               100.0 * ps.pred[d] / ps.pairs[d], 100.0 * ps.miss[d] / ps.pairs[d], 100.0 * ps.miss_pred[d] / std::max<uint64_t>(1, ps.miss[d]),
               (double)ps.pf_issued[d] / std::max<uint64_t>(1, ps.pairs[d] / 8), 100.0 * ps.pf_used[d] / std::max<uint64_t>(1, ps.pf_issued[d]));
  }
  // GLM_CHECK_PF_IDS=<int32 file>: prefill timing of a real tokenized prompt (phase breakdown with HIVE_GLM_PROF)
  if (const char* pf = getenv("GLM_CHECK_PF_IDS")) {
    std::ifstream fi(pf, std::ios::binary | std::ios::ate);
    std::vector<int32_t> ids((size_t)fi.tellg() / 4);
    fi.seekg(0); fi.read((char*)ids.data(), ids.size() * 4);
    for (int rep = 0; rep < 2; ++rep) {
      auto sq = eng.new_seq((int64_t)ids.size() + 64);
      eng.stats() = GlmForwardStats{};
      const auto e0 = experts.stats();
      std::vector<float> lg(V);
      const auto t0 = std::chrono::steady_clock::now();
      eng.prefill(*sq, ids.data(), (int)ids.size(), lg.data());
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const auto e1 = experts.stats();
      auto& st = eng.stats();
      printf("prefill %zu tokens (run %d): %.2f s = %.0f tok/s · streamed %llu · cpu rows %llu · hit rows %llu\n", ids.size(), rep + 1, sec, ids.size() / sec,
             (unsigned long long)(e1.streamed - e0.streamed), (unsigned long long)(e1.cpu - e0.cpu), (unsigned long long)(e1.hit - e0.hit));
      printf("  phases (ms): embed %.0f hc %.0f kda %.0f (proj %.0f rec %.0f) dsa %.0f dense %.0f router %.0f shared %.0f experts %.0f head %.0f\n", st.ms_embed,
             st.ms_hc, st.ms_kda + st.ms_kda_proj + st.ms_kda_core, st.ms_kda_proj, st.ms_kda_core, st.ms_dsa, st.ms_dense, st.ms_router, st.ms_shared, st.ms_experts, st.ms_head);
    }
  }
  auto& cs = experts.stats();
  printf("\ncache: routed %llu hit %llu cpu %llu streamed %llu promoted %llu\n", (unsigned long long)cs.routed, (unsigned long long)cs.hit,
         (unsigned long long)cs.cpu, (unsigned long long)cs.streamed, (unsigned long long)cs.promoted);
  printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
