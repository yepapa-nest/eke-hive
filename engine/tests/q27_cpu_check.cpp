// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next CPU engine (hive/q27/q27_cpu.h) forward dump for validation against the CPU oracle (oracle/q27_oracle.py) — the
//   same outputs as engine/tests/q27_check.cpp (GPU), compare with tools/compare_q27.py: h_LNN.npy (float32 [T, 5120], hidden stream after
//   every layer of the one-chunk prefill), logits.npy (the last prompt row, then one row per greedy decode step), decode.json, and
//   timing.json (load, prefill = time to first token, every decode step). With mtp = 1 the MTP block is loaded and one draft per step is
//   checked against the next greedy token (draft_hits in timing.json).
// usage: q27_cpu_check (q27_check) <ckpt dir> <out dir> <ids: comma list or file> [decode N = 8] [mtp 0|1 = 0]
//   Q27_CHECK_NO_DUMP=1 skips the per-layer dumps; HIVE_CPU_POOL_THREADS sets the worker count.
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hive/clock.h"
#include "hive/q27/q27_cpu.h"
#ifdef Q27_CHECK_GPU
#include "hive/q27/q27_engine.h"
#endif

using namespace hive;
using namespace hive::q27::cpu;
#ifdef Q27_CHECK_GPU
using Engine = hive::q27::Q27Engine;
using Seq = hive::q27::Q27Seq;
static CpuStats& stats_of(Engine& e) { return e.cpu_stats(); }   // the CPU-placed layers (GPU sections: Engine::stats, printed at the end)
#define Q27_CHECK_NAME "q27_check"
#else
using Engine = Q27CpuEngine;
using Seq = CpuSeq;
static CpuStats& stats_of(Engine& e) { return e.stats(); }
#define Q27_CHECK_NAME "q27_cpu_check"
#endif

namespace {
void write_npy_f32(const std::string& path, const float* data, int rows, int cols) {
  std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " + std::to_string(cols) + "), }";
  const size_t base = 10 + hdr.size() + 1;
  hdr.append((64 - base % 64) % 64, ' ');
  hdr.push_back('\n');
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  fwrite(magic, 1, 8, f);
  const uint16_t hl = (uint16_t)hdr.size();
  fwrite(&hl, 2, 1, f);
  fwrite(hdr.data(), 1, hdr.size(), f);
  fwrite(data, 4, (size_t)rows * cols, f);
  fclose(f);
}
// a crash prints its stack (addresses resolve with addr2line on a -g build)
void on_fatal(int sig) {
  void* fr[64];
  const int n = backtrace(fr, 64);
  dprintf(2, "q27_cpu_check: signal %d\n", sig);
  backtrace_symbols_fd(fr, n, 2);
  _exit(128 + sig);
}
// float32 .npy (version 1 header) → values
std::vector<float> read_npy_f32(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(2); }
  unsigned char pre[10];
  if (fread(pre, 1, 10, f) != 10 || pre[6] != 1) { fprintf(stderr, "bad npy %s\n", path.c_str()); exit(2); }
  const size_t hl = pre[8] | (pre[9] << 8);
  std::string hdr(hl, ' ');
  if (fread(hdr.data(), 1, hl, f) != hl || hdr.find("<f4") == std::string::npos) { fprintf(stderr, "npy %s is not float32\n", path.c_str()); exit(2); }
  fseek(f, 0, SEEK_END);
  const size_t n = ((size_t)ftell(f) - 10 - hl) / 4;
  fseek(f, (long)(10 + hl), SEEK_SET);
  std::vector<float> v(n);
  if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "short npy %s\n", path.c_str()); exit(2); }
  fclose(f);
  return v;
}
int argmax(const float* r, int V) { return (int)(std::max_element(r, r + V) - r); }
std::vector<int32_t> parse_ids(const std::string& a) {
  std::string text = a;
  if (std::ifstream f(a); f) { std::stringstream ss; ss << f.rdbuf(); text = ss.str(); }
  std::vector<int32_t> ids;
  std::string cur;
  for (char ch : text + ",") {
    if ((ch >= '0' && ch <= '9') || ch == '-') cur.push_back(ch);
    else if (!cur.empty()) { ids.push_back(atoi(cur.c_str())); cur.clear(); }
  }
  return ids;
}
}  // namespace

int main(int argc, char** argv) {
  signal(SIGSEGV, on_fatal); signal(SIGBUS, on_fatal); signal(SIGFPE, on_fatal); signal(SIGABRT, on_fatal);
  if (argc < 4) { fprintf(stderr, "usage: " Q27_CHECK_NAME " <ckpt> <out dir> <ids> [decode N] [mtp 0|1]\n"); return 2; }
  const std::string out = argv[2];
  const std::vector<int32_t> ids = parse_ids(argv[3]);
  const int n_dec = argc > 4 ? atoi(argv[4]) : 8;
  const bool use_mtp = argc > 5 && atoi(argv[5]) > 0;
  const bool dump = !(getenv("Q27_CHECK_NO_DUMP") && atoi(getenv("Q27_CHECK_NO_DUMP")) > 0);
  if (ids.empty()) { fprintf(stderr, "no prompt ids\n"); return 2; }
  const double t0 = mono_ms();
#ifdef Q27_CHECK_GPU
  Engine eng(argv[1], use_mtp);
#else
  CpuOptions co; co.load_mtp = use_mtp;
  Engine eng(argv[1], co);
#endif
  const double load_ms = mono_ms() - t0;
  const hive::q27::Q27Config& c = eng.cfg();
  const int T = (int)ids.size(), V = c.vocab;
  Seq s;
  eng.init_seq(s, T + n_dec + 64);
  // vision: Q27_CHECK_VISION="<raw bf16 patches [gh·gw, 1536]>:gh,gw[:row]" — the encoder output goes to vision.npy; with a row, the prompt
  //   rows [row, row + gh·gw/4) take the image embeddings (and M-RoPE coordinates) in the dumped prefill below
  std::vector<Engine::EmbedOverride> image;
  if (const char* vz = getenv("Q27_CHECK_VISION"); vz && *vz) {
    std::string spec(vz);
    const size_t c1 = spec.find(':'), c2 = spec.find(':', c1 + 1);
    const std::string file = spec.substr(0, c1), grid = spec.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1);
    const int gh = atoi(grid.c_str()), gw = atoi(grid.c_str() + grid.find(',') + 1);
    std::vector<uint16_t> pt((size_t)gh * gw * 1536);
    FILE* f = fopen(file.c_str(), "rb");
    if (!f || fread(pt.data(), 2, pt.size(), f) != pt.size()) { fprintf(stderr, "cannot read %s\n", file.c_str()); return 2; }
    fclose(f);
    std::vector<float> emb;
    const double tv = mono_ms();
    eng.encode_image(pt.data(), gh, gw, emb);
    printf("vision: %dx%d patches encoded in %.2f s\n", gh, gw, (mono_ms() - tv) / 1000);
    write_npy_f32(out + "/vision.npy", emb.data(), gh * gw / 4, c.hidden);
    // Q27_CHECK_VISION_EMB=<oracle vision.npy>: the prompt takes the reference's encoder output instead (separates the language-model
    //   path of image rows from the encoder's own rounding differences)
    if (const char* ve = getenv("Q27_CHECK_VISION_EMB"); ve && *ve) {
      const std::vector<float> ref = read_npy_f32(ve);
      if (ref.size() != emb.size()) { fprintf(stderr, "%s: %zu values, need %zu\n", ve, ref.size(), emb.size()); return 2; }
      emb = ref;
      printf("vision: the prompt uses the reference embeddings %s\n", ve);
    }
    if (c2 != std::string::npos) image.push_back({atoll(spec.c_str() + c2 + 1), gh * gw / 4, emb, gh / 2, gw / 2});
  }
  // per-layer injection: every layer l ≥ 1 takes the reference's input (its h_L{l−1}); outputs go to <out>/inject
  if (const char* ref = getenv("Q27_CHECK_INJECT"); ref && *ref) {
    const std::string idir = out + "/inject";
    if (system(("mkdir -p '" + idir + "'").c_str()) != 0) return 2;
    Seq si;
    eng.init_seq(si, T + 8);
    eng.layer_inject = [&](int l, float* h, int rows) {
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l - 1);
      const std::vector<float> v = read_npy_f32(std::string(ref) + name);
      if (v.size() != (size_t)rows * c.hidden) { fprintf(stderr, "inject: %s has %zu values, need %d rows\n", name, v.size(), rows); exit(2); }
      memcpy(h, v.data(), v.size() * 4);
    };
    eng.layer_hook = [&](int l, const float* h, int rows) {
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
      write_npy_f32(idir + name, h, rows, c.hidden);
    };
    std::vector<float> lg(V);
    if (!image.empty()) eng.set_embed_override(image);
    eng.prefill(si, ids.data(), T, lg.data());
    write_npy_f32(idir + "/logits.npy", lg.data(), 1, V);
    eng.layer_inject = nullptr; eng.layer_hook = nullptr;
    stats_of(eng) = CpuStats{};
    printf("inject: per-layer outputs in %s\n", idir.c_str());
  }
  if (dump)
    eng.layer_hook = [&](int l, const float* h, int rows) {
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
      write_npy_f32(out + name, h, rows, c.hidden);
    };
  std::vector<float> logits((size_t)(n_dec + 1) * V);
  const double tp = mono_ms();
  if (!image.empty()) eng.set_embed_override(image);
  eng.prefill(s, ids.data(), T, logits.data());
  const double prefill_ms = mono_ms() - tp;
  eng.layer_hook = nullptr;
  const CpuStats pst = stats_of(eng);
  stats_of(eng) = CpuStats{};
  std::vector<int32_t> dec;
  std::vector<double> step_ms;
  int drafts_checked = 0, draft_hits = 0;
  std::vector<int32_t> draft_at(n_dec + 3, -1);
  for (int i = 0; i < n_dec; ++i) {
    const float* lr = logits.data() + (size_t)i * V;
    const int32_t tok = (int32_t)(std::max_element(lr, lr + V) - lr);
    if (draft_at[i] >= 0) { ++drafts_checked; draft_hits += draft_at[i] == tok; }
    dec.push_back(tok);
    Seq* sp1[1] = {&s};
    const double ts = mono_ms();
    eng.decode(sp1, &tok, 1, logits.data() + (size_t)(i + 1) * V);
    step_ms.push_back(mono_ms() - ts);
    if (use_mtp && eng.mtp_on() && i + 1 < n_dec) {
      // pending entry = (stream of tok, next greedy token): its draft predicts the greedy token two steps on
      const float* nl = logits.data() + (size_t)(i + 1) * V;
      const int32_t nxt = (int32_t)(std::max_element(nl, nl + V) - nl);
      int32_t d[1]; float cf[1];
      if (eng.mtp_draft(s, nxt, 1, d, cf) == 1) draft_at[i + 2] = d[0];
    }
  }
  const CpuStats dec_st = stats_of(eng);
  write_npy_f32(out + "/logits.npy", logits.data(), n_dec + 1, V);
  {
    FILE* f = fopen((out + "/decode.json").c_str(), "w");
    fprintf(f, "{\"prompt\": %d, \"decode\": [", T);
    for (size_t i = 0; i < dec.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", dec[i]);
    fprintf(f, "]}\n");
    fclose(f);
  }
  // prefill bench: Q27_CHECK_BENCH="2048,16384" — the prompt tiled to each length, fresh sequence, TTFT and the section breakdown
  if (const char* bl = getenv("Q27_CHECK_BENCH"); bl && *bl) {
    std::vector<int> lens;
    for (const char* q = bl; *q;) { lens.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    for (int L : lens) {
      if (L <= 0) continue;
      std::vector<int32_t> big(L);
      for (int i = 0; i < L; ++i) big[i] = ids[i % T];
      Seq sb;
      eng.init_seq(sb, L + 16);
      std::vector<float> lg(V);
      stats_of(eng) = CpuStats{};
      const double tb = mono_ms();
      eng.prefill(sb, big.data(), L, lg.data());
      const double ms = mono_ms() - tb;
      const CpuStats bs = stats_of(eng);
      printf("bench prefill %d tokens: TTFT %.2f s (%.1f tok/s) · mlp %.0f · lin %.0f · attn %.0f · norm %.0f · head %.0f · mtp %.0f ms\n", L,
             ms / 1000, L / (ms / 1000), bs.ms_mlp, bs.ms_lin, bs.ms_attn, bs.ms_norm, bs.ms_head, bs.ms_mtp);
      fflush(stdout);
    }
  }
  // M > 1 decode: B copies of the prompt decoded together must give the single-sequence greedy tokens
  if (const char* bs = getenv("Q27_CHECK_BATCH"); bs && atoi(bs) > 1 && n_dec > 1) {
    const int B = std::min(16, atoi(bs)), steps = std::min(n_dec - 1, 4);
    std::vector<Seq> seqs(B);
    std::vector<Seq*> sp(B);
    std::vector<float> lg((size_t)B * V);
    for (int b = 0; b < B; ++b) { eng.init_seq(seqs[b], T + 16); sp[b] = &seqs[b]; if (!image.empty()) eng.set_embed_override(image); eng.prefill(seqs[b], ids.data(), T, lg.data()); }
    int agree = 0, total = 0;
    float maxd = 0.f;
    double bms = 0;
    for (int i = 0; i < steps; ++i) {
      std::vector<int32_t> tk(B, dec[i]);
      const double ts = mono_ms();
      eng.decode(sp.data(), tk.data(), B, lg.data());
      bms += mono_ms() - ts;
      const float* ref = logits.data() + (size_t)(i + 1) * V;
      for (int b = 0; b < B; ++b) {
        const float* r = lg.data() + (size_t)b * V;
        agree += argmax(r, V) == argmax(ref, V); ++total;
        for (int v = 0; v < V; ++v) maxd = std::max(maxd, std::fabs(r[v] - ref[v]));
      }
    }
    printf("batch decode M=%d: %d steps · top-1 agreement with M=1 %d/%d · max|logit diff| %.4g · %.1f ms/step (%.2f tok/s aggregate)\n", B, steps,
           agree, total, maxd, bms / steps, B * 1000.0 * steps / bms);
  }
#ifndef Q27_CHECK_GPU
  // MTP against the oracle's last entry (mtp_top.json row T−2): prefill T−1 tokens, then the draft for (h_{T−2}, x_{T−1}) → mtp_last.npy
  if (use_mtp && eng.mtp_on() && T > 2) {
    Seq sm;
    eng.init_seq(sm, T + 16);
    std::vector<float> lg(V);
    if (!image.empty()) eng.set_embed_override(image);
    eng.prefill(sm, ids.data(), T - 1, lg.data());
    int32_t d[1]; float cf[1];
    if (eng.mtp_draft(sm, ids[T - 1], 1, d, cf) == 1) {
      write_npy_f32(out + "/mtp_last.npy", eng.last_draft_logits().data(), 1, V);
      printf("MTP last prompt entry: draft %d (p %.4f) → mtp_last.npy\n", d[0], cf[0]);
    }
  }
#endif
  // speculative decode with MTP drafts (verify + rollback): the emitted tokens must equal the greedy ones
  if (use_mtp && eng.mtp_on() && n_dec > 1) {
    const int k = std::max(1, std::min(7, eng.mtp_k()));
    Seq ss;
    eng.init_seq(ss, T + n_dec + 32);
    std::vector<float> lg((size_t)(k + 1) * V);
    if (!image.empty()) eng.set_embed_override(image);
    eng.prefill(ss, ids.data(), T, lg.data());
    std::vector<int32_t> emitted{(int32_t)argmax(lg.data(), V)};
    int rounds = 0, accepted = 0, proposed = 0;
    const double ts = mono_ms();
    while ((int)emitted.size() < n_dec) {
      const int32_t tok = emitted.back();
      int32_t dr[8]; float cf[8];
      const int nd = eng.mtp_draft(ss, tok, k, dr, cf);
      std::vector<int32_t> vi{tok};
      for (int j = 0; j < nd; ++j) vi.push_back(dr[j]);
      Seq* sp1[1] = {&ss};
      const int Mv = (int)vi.size();
      eng.verify_batch(sp1, &Mv, 1, vi.data(), lg.data());
      int keep = 1;
      while (keep < Mv && argmax(lg.data() + (size_t)(keep - 1) * V, V) == vi[keep]) ++keep;
      eng.rollback_batch(&keep, 1);
      for (int j = 1; j < keep; ++j) emitted.push_back(vi[j]);
      emitted.push_back((int32_t)argmax(lg.data() + (size_t)(keep - 1) * V, V));
      ++rounds; accepted += keep - 1; proposed += nd;
    }
    emitted.resize(n_dec);
    const bool same = emitted == dec;
    printf("MTP spec decode k=%d: %d tokens in %d rounds · accepted %d / %d drafts · %.2f tok/s · tokens %s greedy\n", k, n_dec, rounds, accepted,
           proposed, n_dec * 1000.0 / (mono_ms() - ts), same ? "==" : "!=");
    if (!same) { for (int32_t t : emitted) printf(" %d", t); printf("\n"); }
  }
#ifdef Q27_CHECK_GPU
  // session image round trip (Q27_CHECK_IMAGE=1): prefill, save_bytes, load into a fresh sequence, one decode step on both — same logits
  if (getenv("Q27_CHECK_IMAGE") && n_dec > 1) {
    Seq a, b;
    eng.init_seq(a, T + 32); eng.init_seq(b, T + 32);
    std::vector<float> l0(V), la(V), lb(V);
    eng.prefill(a, ids.data(), T, l0.data());
    const int32_t t = (int32_t)argmax(l0.data(), V);
    b.load_bytes(a.save_bytes());
    Seq* pa[1] = {&a};
    Seq* pb[1] = {&b};
    eng.decode(pa, &t, 1, la.data());
    eng.decode(pb, &t, 1, lb.data());
    float md = 0.f;
    for (int v = 0; v < V; ++v) md = std::max(md, std::fabs(la[v] - lb[v]));
    printf("image round trip: pos %lld · max|logit diff| after one step %.4g · %s\n", (long long)b.pos, md, md == 0.f ? "identical" : "DIFFERENT");
  }
#endif
#ifdef Q27_CHECK_GPU
  // batched speculative decode (Q27_CHECK_SPECB=S, 2 ≤ S ≤ 8): S sequences on prompts ids[0 .. T − 2s), drafts of all of them in one
  //   mtp_draft_batch, one verify_batch of Σ (k_s + 1) ≤ 16 rows, rollback_batch — every sequence's tokens must equal its own greedy decode
  if (const char* sb = getenv("Q27_CHECK_SPECB"); sb && atoi(sb) > 1 && use_mtp && eng.mtp_on() && n_dec > 1 && T > 16) {
    const int S = std::min(8, atoi(sb)), kb = std::max(1, std::min(eng.mtp_k(), (getenv("Q27_CHECK_SPECB_ROWS") ? atoi(getenv("Q27_CHECK_SPECB_ROWS")) : 16) / S - 1));
    std::vector<std::vector<int32_t>> ref(S);
    std::vector<float> lg1(V);
    for (int s = 0; s < S; ++s) {   // greedy reference per sequence
      Seq q;
      const int Ts = T - 2 * s;
      eng.init_seq(q, Ts + n_dec + 32);
      eng.prefill(q, ids.data(), Ts, lg1.data());
      ref[s].push_back((int32_t)argmax(lg1.data(), V));
      while ((int)ref[s].size() < n_dec) {
        Seq* qp[1] = {&q};
        const int32_t t = ref[s].back();
        eng.decode(qp, &t, 1, lg1.data());
        ref[s].push_back((int32_t)argmax(lg1.data(), V));
      }
    }
    std::vector<Seq> sq(S);
    std::vector<std::vector<int32_t>> em(S);
    for (int s = 0; s < S; ++s) {
      eng.init_seq(sq[s], T + n_dec + 32);
      eng.prefill(sq[s], ids.data(), T - 2 * s, lg1.data());
      em[s].push_back((int32_t)argmax(lg1.data(), V));
    }
    std::vector<float> lg((size_t)32 * V);
    int rounds = 0, acc = 0, prop = 0;
    const double ts = mono_ms();
    for (;;) {
      std::vector<Seq*> act;
      std::vector<int> ai;
      for (int s = 0; s < S; ++s) if ((int)em[s].size() < n_dec) { act.push_back(&sq[s]); ai.push_back(s); }
      if (act.empty()) break;
      const int A = (int)act.size();
      std::vector<int32_t> tk(A), dr((size_t)A * 7), vi;
      std::vector<float> cf((size_t)A * 7);
      std::vector<int> nd(A), Ms(A);
      for (int a = 0; a < A; ++a) tk[a] = em[ai[a]].back();
      eng.mtp_draft_batch(act.data(), tk.data(), A, kb, dr.data(), cf.data(), nd.data());
      for (int a = 0; a < A; ++a) {
        const int k = std::min(nd[a], kb);
        Ms[a] = k + 1;
        vi.push_back(tk[a]);
        for (int j = 0; j < k; ++j) vi.push_back(dr[(size_t)a * 7 + j]);
        prop += k;
      }
      eng.verify_batch(act.data(), Ms.data(), A, vi.data(), lg.data());
      std::vector<int> keep(A);
      int r0 = 0;
      for (int a = 0; a < A; ++a) {
        int kp = 1;
        while (kp < Ms[a] && argmax(lg.data() + (size_t)(r0 + kp - 1) * V, V) == vi[r0 + kp]) ++kp;
        keep[a] = kp;
        for (int j = 1; j < kp; ++j) em[ai[a]].push_back(vi[r0 + j]);
        em[ai[a]].push_back((int32_t)argmax(lg.data() + (size_t)(r0 + kp - 1) * V, V));
        acc += kp - 1;
        r0 += Ms[a];
      }
      eng.rollback_batch(keep.data(), A);
      ++rounds;
    }
    int same = 0, tot = 0;
    for (int s = 0; s < S; ++s) { em[s].resize(n_dec); same += em[s] == ref[s]; tot += n_dec; }
    printf("batched spec S=%d k=%d: %d rounds · accepted %d / %d drafts · %.1f tok/s aggregate · %d / %d sequences == greedy\n", S, kb, rounds, acc,
           prop, tot * 1000.0 / (mono_ms() - ts), same, S);
  }
#endif
  std::vector<double> sorted = step_ms;
  std::sort(sorted.begin(), sorted.end());
  const double med = sorted.empty() ? 0 : sorted[sorted.size() / 2];
  const CpuStats& dst = dec_st;
  {
    FILE* f = fopen((out + "/timing.json").c_str(), "w");
    fprintf(f, "{\"load_s\": %.3f, \"prompt_tokens\": %d, \"prefill_s\": %.3f, \"prefill_tok_s\": %.3f, \"decode_step_s\": [", load_ms / 1000, T,
            prefill_ms / 1000, T / (prefill_ms / 1000));
    for (size_t i = 0; i < step_ms.size(); ++i) fprintf(f, "%s%.4f", i ? ", " : "", step_ms[i] / 1000);
    fprintf(f, "], \"decode_median_s\": %.4f, \"decode_tok_s\": %.3f, \"prefill_ms_mlp\": %.1f, \"prefill_ms_attn\": %.1f, \"prefill_ms_lin\": %.1f, "
               "\"decode_ms_mlp\": %.1f, \"decode_ms_attn\": %.1f, \"decode_ms_lin\": %.1f, \"mtp\": %s, \"draft_hits\": %d, \"drafts\": %d}\n",
            med / 1000, med > 0 ? 1000.0 / med : 0.0, pst.ms_mlp, pst.ms_attn, pst.ms_lin, dst.ms_mlp, dst.ms_attn, dst.ms_lin, use_mtp ? "true" : "false",
            draft_hits, drafts_checked);
    fclose(f);
  }
  if (use_mtp) printf("  MTP draft-1 hits %d / %d\n", draft_hits, drafts_checked);
  printf(Q27_CHECK_NAME ": load %.1f s · prompt %d tokens, TTFT %.2f s (%.2f tok/s) · decode %d steps, median %.1f ms (%.2f tok/s) · out %s\n",
         load_ms / 1000, T, prefill_ms / 1000, T / (prefill_ms / 1000), n_dec, med, med > 0 ? 1000.0 / med : 0.0, out.c_str());
  auto brk = [](const char* what, const CpuStats& t, double per) {
    const double parts = t.ms_mlp + t.ms_attn + t.ms_lin + t.ms_norm + t.ms_head + t.ms_mtp;
    printf("  %s (ms%s): total %.1f · mlp %.1f · lin %.1f · attn %.1f · norm %.1f · head %.1f · mtp %.1f · other %.1f\n", what,
           per > 1 ? "/step" : "", t.ms_total / per, t.ms_mlp / per, t.ms_lin / per, t.ms_attn / per, t.ms_norm / per, t.ms_head / per,
           t.ms_mtp / per, (t.ms_total - parts) / per);
  };
  brk("prefill", pst, 1);
  brk("decode", dec_st, std::max(1, n_dec));
#ifdef Q27_CHECK_GPU
  {
    const auto& g = eng.stats();
    const auto& pl = eng.placement();
    printf("  GPU engine (whole run, ms): total %.1f · gpu layers %.1f · cpu layers %.1f · transfers %.1f · head %.1f · mtp %.1f · steps %ld · placement: %d GPU layers, lm_head %s, MTP %s\n",
           g.ms_total, g.ms_gpu_layers, g.ms_cpu_layers, g.ms_xfer, g.ms_head, g.ms_mtp, g.steps, pl.gpu_layers, pl.head_gpu ? "GPU" : "CPU", pl.mtp_gpu ? "GPU" : "off");
  }
#endif
  return 0;
}
