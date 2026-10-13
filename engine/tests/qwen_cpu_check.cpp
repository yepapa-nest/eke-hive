// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next CPU engine (hive/qwen/qwen_cpu.h) forward dump for validation against the CPU oracle (oracle/qwen4_oracle.py) — the
//   same outputs as engine/tests/qwen_check.cpp, compare with tools/compare_qwen.py: h_LNN.npy (float32 [T, 10240], hidden stream after
//   every layer of the one-chunk prefill), logits.npy (the last prompt row, then one row per greedy decode step), decode.json, and
//   timing.json (load, prefill = time to first token, every decode step). With mtp = 1 the MTP block is loaded and one draft per step is
//   checked against the next greedy token (draft_hits in timing.json).
// usage: qwen_cpu_check <ckpt dir> <out dir> <ids: comma list or file> [decode N = 8] [mtp 0|1 = 0]
//   QWEN_CHECK_NO_DUMP=1 skips the per-layer dumps; HIVE_CPU_POOL_THREADS sets the worker count.
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
#include "hive/qwen/qwen_cpu.h"

using namespace hive;
using namespace hive::qwen::cpu;

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
  dprintf(2, "qwen_cpu_check: signal %d\n", sig);
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
  if (argc < 4) { fprintf(stderr, "usage: qwen_cpu_check <ckpt> <out dir> <ids> [decode N] [mtp 0|1]\n"); return 2; }
  const std::string out = argv[2];
  const std::vector<int32_t> ids = parse_ids(argv[3]);
  const int n_dec = argc > 4 ? atoi(argv[4]) : 8;
  const bool use_mtp = argc > 5 && atoi(argv[5]) > 0;
  const bool dump = !(getenv("QWEN_CHECK_NO_DUMP") && atoi(getenv("QWEN_CHECK_NO_DUMP")) > 0);
  if (ids.empty()) { fprintf(stderr, "no prompt ids\n"); return 2; }
  const double t0 = mono_ms();
  QwenCpuEngine eng(argv[1], use_mtp);
  const double load_ms = mono_ms() - t0;
  if (getenv("QWEN_CHECK_I8") && atoi(getenv("QWEN_CHECK_I8")) > 0) {   // dumps / decode of the lossy I8 dense weights (needs HIVE_QWEN_CPU_DENSE_I8_AB)
    if (!eng.has_dense_i8()) { fprintf(stderr, "QWEN_CHECK_I8 needs HIVE_QWEN_CPU_DENSE_I8_AB at load\n"); return 2; }
    eng.set_dense_i8(true);
  }
  const hive::qwen::QwenConfig& c = eng.cfg();
  const int T = (int)ids.size(), V = c.vocab;
  CpuSeq s;
  eng.init_seq(s, T + n_dec + 64);
  // per-layer injection: every layer l ≥ 1 takes the reference's input (its h_L{l−1}); outputs go to <out>/inject
  if (const char* ref = getenv("QWEN_CHECK_INJECT"); ref && *ref) {
    const std::string idir = out + "/inject";
    if (system(("mkdir -p '" + idir + "'").c_str()) != 0) return 2;
    CpuSeq si;
    eng.init_seq(si, T + 8);
    eng.layer_inject = [&](int l, float* h, int rows) {
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l - 1);
      const std::vector<float> v = read_npy_f32(std::string(ref) + name);
      if (v.size() != (size_t)rows * c.hcw()) { fprintf(stderr, "inject: %s has %zu values, need %d rows\n", name, v.size(), rows); exit(2); }
      memcpy(h, v.data(), v.size() * 4);
    };
    eng.layer_hook = [&](int l, const float* h, int rows) {
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
      write_npy_f32(idir + name, h, rows, c.hcw());
    };
    std::vector<float> lg(V);
    eng.prefill(si, ids.data(), T, lg.data());
    write_npy_f32(idir + "/logits.npy", lg.data(), 1, V);
    eng.layer_inject = nullptr; eng.layer_hook = nullptr;
    eng.stats() = CpuStats{};
    printf("inject: per-layer outputs in %s\n", idir.c_str());
  }
  if (dump)
    eng.layer_hook = [&](int l, const float* h, int rows) {
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
      write_npy_f32(out + name, h, rows, c.hcw());
    };
  std::vector<float> logits((size_t)(n_dec + 1) * V);
  const double tp = mono_ms();
  eng.prefill(s, ids.data(), T, logits.data());
  const double prefill_ms = mono_ms() - tp;
  eng.layer_hook = nullptr;
  const CpuStats pst = eng.stats();
  eng.stats() = CpuStats{};
  std::vector<int32_t> dec;
  std::vector<double> step_ms;
  int drafts_checked = 0, draft_hits = 0;
  std::vector<int32_t> draft_at(n_dec + 3, -1);
  for (int i = 0; i < n_dec; ++i) {
    const float* lr = logits.data() + (size_t)i * V;
    const int32_t tok = (int32_t)(std::max_element(lr, lr + V) - lr);
    if (draft_at[i] >= 0) { ++drafts_checked; draft_hits += draft_at[i] == tok; }
    dec.push_back(tok);
    CpuSeq* sp1[1] = {&s};
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
  const CpuStats dec_st = eng.stats();
  write_npy_f32(out + "/logits.npy", logits.data(), n_dec + 1, V);
  {
    FILE* f = fopen((out + "/decode.json").c_str(), "w");
    fprintf(f, "{\"prompt\": %d, \"decode\": [", T);
    for (size_t i = 0; i < dec.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", dec[i]);
    fprintf(f, "]}\n");
    fclose(f);
  }
  // prefill bench: QWEN_CHECK_BENCH="2048,16384" — the prompt tiled to each length, fresh sequence, TTFT and the section breakdown
  if (const char* bl = getenv("QWEN_CHECK_BENCH"); bl && *bl) {
    std::vector<int> lens;
    for (const char* q = bl; *q;) { lens.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    for (int L : lens) {
      if (L <= 0) continue;
      std::vector<int32_t> big(L);
      for (int i = 0; i < L; ++i) big[i] = ids[i % T];
      CpuSeq sb;
      eng.init_seq(sb, L + 16);
      std::vector<float> lg(V);
      eng.stats() = CpuStats{};
      const double tb = mono_ms();
      eng.prefill(sb, big.data(), L, lg.data());
      const double ms = mono_ms() - tb;
      const CpuStats bs = eng.stats();
      printf("bench prefill %d tokens: TTFT %.2f s (%.1f tok/s) · moe %.0f · lin %.0f · attn %.0f · hc %.0f · ple %.0f · head %.0f · mtp %.0f ms\n", L,
             ms / 1000, L / (ms / 1000), bs.ms_moe, bs.ms_lin, bs.ms_attn, bs.ms_hc, bs.ms_ple, bs.ms_head, bs.ms_mtp);
      fflush(stdout);
    }
  }
  // A/B (QWEN_CHECK_AB="2048x3,16384x1,decx3,2048x2@8" — length or dec, repetitions, legacy mask of the base side, default 31): the previous schedule (set_ab_legacy) and the new one alternately in this process, each
  //   measurement with the CPU use of another process (QWEN_CHECK_WATCH_PID, e.g. a co-located server; needs --pid=host in docker) over the
  //   same interval; the two modes' logits must be bit-identical.
  if (const char* ab = getenv("QWEN_CHECK_AB"); ab && *ab) {
    const int wpid = getenv("QWEN_CHECK_WATCH_PID") ? atoi(getenv("QWEN_CHECK_WATCH_PID")) : 0;   // 0: all other processes on the host
    // without a pid: every other process on the host = host busy ticks (/proc/stat) − this process's ticks
    auto self_ticks = []() -> long long {
      FILE* f = fopen("/proc/self/stat", "r");
      if (!f) return 0;
      char buf[2048]; const size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
      const char* q = strrchr(buf, ')');
      long long ut = 0, st = 0; int field = 2;
      for (const char* t = q ? q + 1 : buf; *t && field < 15; ++t) if (*t == ' ') { ++field; if (field == 14) ut = atoll(t + 1); if (field == 15) st = atoll(t + 1); }
      return ut + st;
    };
    auto ticks = [&]() -> long long {
      if (wpid <= 0) {
        FILE* f = fopen("/proc/stat", "r");
        if (!f) return -1;
        long long v[8] = {0};
        if (fscanf(f, "cpu %lld %lld %lld %lld %lld %lld %lld %lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) != 8) { fclose(f); return -1; }
        fclose(f);
        return v[0] + v[1] + v[2] + v[5] + v[6] + v[7] - self_ticks();
      }
      char path[64]; snprintf(path, sizeof path, "/proc/%d/stat", wpid);
      FILE* f = fopen(path, "r");
      if (!f) return -1;
      char buf[2048]; const size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
      const char* q = strrchr(buf, ')');
      if (!q) return -1;
      long long ut = 0, st = 0; int field = 2;
      for (const char* t = q + 1; *t && field < 15; ++t) if (*t == ' ') { ++field; if (field == 14) ut = atoll(t + 1); if (field == 15) st = atoll(t + 1); }
      return ut + st;
    };
    const double hz = (double)sysconf(_SC_CLK_TCK);
    struct Item { int len; int reps; int mask; bool i8; };
    std::vector<Item> items;
    for (const char* q = ab; *q;) {
      const char* e = q; while (*e && *e != ',') ++e;
      std::string tok(q, e);
      const size_t x = tok.find('x'), at = tok.find('@');
      const int reps = x == std::string::npos ? 1 : atoi(tok.c_str() + x + 1);
      const bool i8 = at != std::string::npos && tok.compare(at + 1, 2, "i8") == 0;   // "@i8": base = bf16 dense, new = I8 twins (lossy)
      const int mask = at == std::string::npos || i8 ? 31 : atoi(tok.c_str() + at + 1);   // "len x reps @ legacy-mask"
      if (i8 && !eng.has_dense_i8()) { fprintf(stderr, "AB @i8 needs HIVE_QWEN_CPU_DENSE_I8_AB at load\n"); continue; }
      items.push_back({tok.rfind("dec", 0) == 0 ? 0 : atoi(tok.c_str()), reps, mask, i8});
      q = *e ? e + 1 : e;
    }
    for (const Item& it : items)
      for (int rep = 0; rep < it.reps; ++rep) {
        std::vector<float> ref;
        std::vector<int32_t> ref_toks;
        for (int k = 0; k < 2; ++k) {
          const bool legacy = (k == 0) == (rep % 2 == 0);
          if (it.i8) { eng.set_ab_legacy(0); eng.set_dense_i8(!legacy); }
          else eng.set_ab_legacy(legacy ? it.mask : 0);
          std::vector<int32_t> toks;
          eng.stats() = CpuStats{};
          const long long g0 = ticks();
          const double t0 = mono_ms();
          std::vector<float> lg(V);
          double ms = 0;
          std::vector<double> st;
          if (it.len > 0) {
            std::vector<int32_t> big(it.len);
            for (int i = 0; i < it.len; ++i) big[i] = ids[i % T];
            CpuSeq sb; eng.init_seq(sb, it.len + 16);
            eng.prefill(sb, big.data(), it.len, lg.data());
            ms = mono_ms() - t0;
          } else {
            CpuSeq sb; eng.init_seq(sb, T + n_dec + 16);
            eng.prefill(sb, ids.data(), T, lg.data());
            eng.stats() = CpuStats{};
            CpuSeq* sp1[1] = {&sb};
            int32_t tk = it.i8 ? (int32_t)argmax(lg.data(), V) : dec[0];   // lossy A/B: each mode follows its own greedy tokens
            for (int i = 0; i < n_dec; ++i) {
              if (!it.i8) tk = dec[i];
              toks.push_back(tk);
              const double ts = mono_ms(); eng.decode(sp1, &tk, 1, lg.data()); st.push_back(mono_ms() - ts);
              tk = (int32_t)argmax(lg.data(), V);
            }
            std::sort(st.begin(), st.end());
            ms = st[st.size() / 2];
          }
          const double wall = mono_ms() - t0;
          const long long g1 = ticks();
          const double glm = (g0 >= 0 && g1 >= 0) ? (g1 - g0) / hz / (wall / 1000) * 100 : -1;
          const CpuStats b = eng.stats();
          const double per = it.len > 0 ? 1 : std::max(1, n_dec);
          if (it.len > 0)
            printf("AB %s@%s prefill %d rep %d: TTFT %.2f s (%.1f tok/s)", legacy ? "base" : "new ", it.i8 ? "i8" : std::to_string(it.mask).c_str(), it.len, rep, ms / 1000, it.len / (ms / 1000));
          else
            printf("AB %s@%s decode c1 rep %d: median %.1f ms (%.2f tok/s)", legacy ? "base" : "new ", it.i8 ? "i8" : std::to_string(it.mask).c_str(), rep, ms, 1000 / ms);
          printf(" · moe %.1f lin %.1f attn %.1f hc %.1f ple %.1f head %.1f mtp %.1f%s · other-process CPU %.0f%%\n", b.ms_moe / per, b.ms_lin / per,
                 b.ms_attn / per, b.ms_hc / per, b.ms_ple / per, b.ms_head / per, b.ms_mtp / per, it.len > 0 ? " ms" : " ms/step", glm);
          if (k == 0) { ref = lg; ref_toks = toks; }
          else if (!it.i8) printf("AB    logits base == new: %s\n", memcmp(ref.data(), lg.data(), (size_t)V * 4) == 0 ? "bit-identical" : "DIFFER");
          else {
            double ab = 0, aa = 0, bb = 0;
            for (int v = 0; v < V; ++v) { ab += (double)ref[v] * lg[v]; aa += (double)ref[v] * ref[v]; bb += (double)lg[v] * lg[v]; }
            int same = 0;
            for (size_t i = 0; i < std::min(ref_toks.size(), toks.size()); ++i) { if (ref_toks[i] != toks[i]) break; ++same; }
            printf("AB    i8 vs bf16: last logits cos %.6f · top-1 %s%s\n", ab / std::sqrt(aa * bb), argmax(ref.data(), V) == argmax(lg.data(), V) ? "same" : "DIFFERENT",
                   it.len > 0 ? "" : (" · greedy tokens identical for " + std::to_string(same) + " / " + std::to_string(toks.size())).c_str());
          }
          fflush(stdout);
        }
      }
    eng.set_ab_legacy(0);
    if (eng.has_dense_i8()) eng.set_dense_i8(false);
  }
  // M > 1 decode: B copies of the prompt decoded together must give the single-sequence greedy tokens
  if (const char* bs = getenv("QWEN_CHECK_BATCH"); bs && atoi(bs) > 1 && n_dec > 1) {
    const int B = std::min(8, atoi(bs)), steps = std::min(n_dec - 1, 4);
    std::vector<CpuSeq> seqs(B);
    std::vector<CpuSeq*> sp(B);
    std::vector<float> lg((size_t)B * V);
    for (int b = 0; b < B; ++b) { eng.init_seq(seqs[b], T + 16); sp[b] = &seqs[b]; eng.prefill(seqs[b], ids.data(), T, lg.data()); }
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
  // speculative decode with MTP drafts (verify + rollback): the emitted tokens must equal the greedy ones
  if (use_mtp && eng.mtp_on() && n_dec > 1) {
    const int k = std::max(1, std::min(3, eng.mtp_k()));
    CpuSeq ss;
    eng.init_seq(ss, T + n_dec + 32);
    std::vector<float> lg((size_t)(k + 1) * V);
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
      CpuSeq* sp1[1] = {&ss};
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
  std::vector<double> sorted = step_ms;
  std::sort(sorted.begin(), sorted.end());
  const double med = sorted.empty() ? 0 : sorted[sorted.size() / 2];
  const CpuStats& dst = dec_st;
  {
    FILE* f = fopen((out + "/timing.json").c_str(), "w");
    fprintf(f, "{\"load_s\": %.3f, \"prompt_tokens\": %d, \"prefill_s\": %.3f, \"prefill_tok_s\": %.3f, \"decode_step_s\": [", load_ms / 1000, T,
            prefill_ms / 1000, T / (prefill_ms / 1000));
    for (size_t i = 0; i < step_ms.size(); ++i) fprintf(f, "%s%.4f", i ? ", " : "", step_ms[i] / 1000);
    fprintf(f, "], \"decode_median_s\": %.4f, \"decode_tok_s\": %.3f, \"prefill_ms_moe\": %.1f, \"prefill_ms_attn\": %.1f, \"prefill_ms_lin\": %.1f, "
               "\"decode_ms_moe\": %.1f, \"decode_ms_attn\": %.1f, \"decode_ms_lin\": %.1f, \"mtp\": %s, \"draft_hits\": %d, \"drafts\": %d}\n",
            med / 1000, med > 0 ? 1000.0 / med : 0.0, pst.ms_moe, pst.ms_attn, pst.ms_lin, dst.ms_moe, dst.ms_attn, dst.ms_lin, use_mtp ? "true" : "false",
            draft_hits, drafts_checked);
    fclose(f);
  }
  if (use_mtp) printf("  MTP draft-1 hits %d / %d\n", draft_hits, drafts_checked);
  printf("qwen_cpu_check: load %.1f s · prompt %d tokens, TTFT %.2f s (%.2f tok/s) · decode %d steps, median %.1f ms (%.2f tok/s) · out %s\n",
         load_ms / 1000, T, prefill_ms / 1000, T / (prefill_ms / 1000), n_dec, med, med > 0 ? 1000.0 / med : 0.0, out.c_str());
  auto brk = [](const char* what, const CpuStats& t, double per) {
    const double parts = t.ms_moe + t.ms_attn + t.ms_lin + t.ms_hc + t.ms_ple + t.ms_head + t.ms_mtp;
    printf("  %s (ms%s): total %.1f · moe %.1f · lin %.1f · attn %.1f · hc %.1f · ple %.1f · head %.1f · mtp %.1f · other %.1f\n", what,
           per > 1 ? "/step" : "", t.ms_total / per, t.ms_moe / per, t.ms_lin / per, t.ms_attn / per, t.ms_hc / per, t.ms_ple / per, t.ms_head / per,
           t.ms_mtp / per, (t.ms_total - parts) / per);
  };
  brk("prefill", pst, 1);
  brk("decode", dec_st, std::max(1, n_dec));
  return 0;
}
