// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash CPU engine (hive/glm/glm_cpu.h) forward dump for validation against oracle/glm5_oracle.py, compared with tools/compare_glm.py:
//   h_LNN.npy (float32 [T, 4·4096], hc streams after every layer of the prefill), logits.npy (the last prompt row, then one row per greedy decode
//   step), decode.json and timing.json (load, prefill = time to first token, every decode step with the section breakdown).
// usage: glm_cpu_check <ckpt dir> <out dir> <ids: comma list or file> [decode N = 8] [mtp 0|1 = 0]
//   GLM_CHECK_NO_DUMP=1           no per-layer dumps
//   GLM_CHECK_INJECT=<oracle dir> first a prefill where every layer takes the oracle's input (h_L{l−1}); outputs in <out>/inject
//   GLM_CHECK_MTP_NEXT=<tok>      NextN draft logits of entry T−1 = (h_{T−1}, tok) → mtp_logits.npy (separate sequence)
//   GLM_CHECK_BATCH=B             B copies of the prompt decoded together (M = B) against the single-sequence logits
//   GLM_CHECK_BENCH=2048,...      prefill TTFT of the prompt tiled to each length (fresh sequence), with the section breakdown
//   GLM_CHECK_VISION=<patches.npy float32 [N, 1176]> GLM_CHECK_GRID=gh,gw [GLM_CHECK_IMAGE_AT=row]: encoder output → vision.npy; with
//                                 IMAGE_AT the prompt rows [row, row + N/4) take it (as hived does for an image span)
//   GLM_CHECK_STATE=1             session image round trip: save after the prefill, load into a new sequence, one decode step must match
//   mtp = 1: speculative decoding with drafts (k = HIVE_GLM_MTP_K, verify + rollback) must emit the greedy tokens.
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hive/clock.h"
#include "hive/glm/glm_cpu.h"

using namespace hive;
using namespace hive::glm::cpu;

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
void on_fatal(int sig) {
  void* fr[64];
  const int n = backtrace(fr, 64);
  dprintf(2, "glm_cpu_check: signal %d\n", sig);
  backtrace_symbols_fd(fr, n, 2);
  _exit(128 + sig);
}
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
uint16_t f2b(float x) { uint32_t u; memcpy(&u, &x, 4); u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16; return (uint16_t)u; }
void brk(const char* what, const CpuStats& t, double per) {
  const double parts = t.ms_embed + t.ms_hc + t.ms_kda_proj + t.ms_kda_core + t.ms_dsa_proj + t.ms_dsa_idx + t.ms_dsa_attn + t.ms_dense + t.ms_router +
                       t.ms_experts + t.ms_head + t.ms_mtp;
  printf("  %s (ms%s): total %.1f · embed %.1f · hc %.1f · kda proj %.1f core %.1f · dsa proj %.1f idx %.1f attn %.1f · dense %.1f · router %.1f · "
         "experts+shared %.1f · head %.1f · mtp %.1f · other %.1f\n",
         what, per > 1 ? "/step" : "", t.ms_total / per, t.ms_embed / per, t.ms_hc / per, t.ms_kda_proj / per, t.ms_kda_core / per, t.ms_dsa_proj / per,
         t.ms_dsa_idx / per, t.ms_dsa_attn / per, t.ms_dense / per, t.ms_router / per, t.ms_experts / per, t.ms_head / per, t.ms_mtp / per,
         (t.ms_total - parts) / per);
}
}  // namespace

int main(int argc, char** argv) {
  signal(SIGSEGV, on_fatal); signal(SIGBUS, on_fatal); signal(SIGFPE, on_fatal); signal(SIGABRT, on_fatal);
  if (argc < 4) { fprintf(stderr, "usage: glm_cpu_check <ckpt> <out dir> <ids> [decode N] [mtp 0|1]\n"); return 2; }
  const std::string out = argv[2];
  const std::vector<int32_t> ids = parse_ids(argv[3]);
  const int n_dec = argc > 4 ? atoi(argv[4]) : 8;
  const bool use_mtp = argc > 5 && atoi(argv[5]) > 0;
  const bool dump = !(getenv("GLM_CHECK_NO_DUMP") && atoi(getenv("GLM_CHECK_NO_DUMP")) > 0);
  if (ids.empty()) { fprintf(stderr, "no prompt ids\n"); return 2; }
  if (system(("mkdir -p '" + out + "'").c_str()) != 0) return 2;
  const double t0 = mono_ms();
  GlmCpuEngine eng(argv[1], use_mtp || getenv("GLM_CHECK_MTP_NEXT"));
  const double load_ms = mono_ms() - t0;
  const CpuConfig& c = eng.cfg();
  const int T = (int)ids.size(), V = c.vocab, HC = 4 * c.hidden;
  const bool full = eng.first_layer() == 0 && eng.last_layer() == c.n_layers - 1;
  printf("glm_cpu_check: loaded in %.1f s · layers %d..%d · prompt %d tokens\n", load_ms / 1000, eng.first_layer(), eng.last_layer(), T);
  fflush(stdout);
  // vision
  std::vector<GlmCpuEngine::EmbedOverride> ov;
  if (const char* vp = getenv("GLM_CHECK_VISION"); vp && *vp) {
    int gh = 0, gw = 0;
    if (!getenv("GLM_CHECK_GRID") || sscanf(getenv("GLM_CHECK_GRID"), "%d,%d", &gh, &gw) != 2) { fprintf(stderr, "GLM_CHECK_GRID=gh,gw needed\n"); return 2; }
    const std::vector<float> pf = read_npy_f32(vp);
    if (pf.size() != (size_t)gh * gw * 1176) { fprintf(stderr, "patches: %zu values, need %d·1176\n", pf.size(), gh * gw); return 2; }
    std::vector<uint16_t> pb(pf.size());
    for (size_t i = 0; i < pf.size(); ++i) pb[i] = f2b(pf[i]);
    std::vector<float> vo;
    const double tv = mono_ms();
    eng.encode_image(pb.data(), gh, gw, vo);
    printf("vision: %d patches (%dx%d) → %d rows in %.2f s\n", gh * gw, gh, gw, gh * gw / 4, (mono_ms() - tv) / 1000);
    write_npy_f32(out + "/vision.npy", vo.data(), gh * gw / 4, c.hidden);
    if (const char* at = getenv("GLM_CHECK_IMAGE_AT"); at && *at) ov.push_back({atoi(at), gh * gw / 4, vo});
  }
  CpuSeq s;
  eng.init_seq(s, T + n_dec + 64);
  if (const char* ref = getenv("GLM_CHECK_INJECT"); ref && *ref) {
    const std::string idir = out + "/inject";
    if (system(("mkdir -p '" + idir + "'").c_str()) != 0) return 2;
    CpuSeq si;
    eng.init_seq(si, T + 8);
    // prefill chunks (HIVE_GLM_CPU_CHUNK) arrive as separate forwards: a layer number below the previous one starts the next chunk
    int64_t off = 0; int prev_l = 1 << 30, prev_rows = 0;
    std::vector<std::vector<float>> acc(c.n_layers);
    eng.layer_inject = [&](int l, float* h, int rows) {
      if (l < prev_l && prev_l != (1 << 30)) off += prev_rows;
      prev_l = l; prev_rows = rows;
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l - 1);
      const std::vector<float> v = read_npy_f32(std::string(ref) + name);
      if (v.size() < (size_t)(off + rows) * HC) { fprintf(stderr, "inject: %s has %zu values, need %lld rows\n", name, v.size(), (long long)(off + rows)); exit(2); }
      memcpy(h, v.data() + (size_t)off * HC, (size_t)rows * HC * 4);
    };
    eng.layer_hook = [&](int l, const float* h, int rows) { acc[l].insert(acc[l].end(), h, h + (size_t)rows * HC); };
    std::vector<float> lg(V);
    if (!ov.empty()) eng.set_embed_override(ov);
    const double ti = mono_ms();
    eng.prefill(si, ids.data(), T, lg.data());
    if (full) write_npy_f32(idir + "/logits.npy", lg.data(), 1, V);
    for (int l = 0; l < c.n_layers; ++l) {
      if (acc[l].empty()) continue;
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
      write_npy_f32(idir + name, acc[l].data(), (int)(acc[l].size() / HC), HC);
    }
    eng.layer_inject = nullptr; eng.layer_hook = nullptr;
    eng.stats() = CpuStats{};
    printf("inject: per-layer outputs in %s (%.1f s)\n", idir.c_str(), (mono_ms() - ti) / 1000);
    fflush(stdout);
  }
  std::vector<std::vector<float>> dacc(c.n_layers);
  if (dump) eng.layer_hook = [&](int l, const float* h, int rows) { dacc[l].insert(dacc[l].end(), h, h + (size_t)rows * HC); };
  std::vector<float> logits((size_t)(n_dec + 1) * V);
  if (!ov.empty()) eng.set_embed_override(ov);
  const double tp = mono_ms();
  eng.prefill(s, ids.data(), T, logits.data());
  const double prefill_ms = mono_ms() - tp;
  eng.layer_hook = nullptr;
  for (int l = 0; l < c.n_layers; ++l) {
    if (dacc[l].empty()) continue;
    char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
    write_npy_f32(out + name, dacc[l].data(), (int)(dacc[l].size() / HC), HC);
    std::vector<float>().swap(dacc[l]);
  }
  const CpuStats pst = eng.stats();
  printf("prefill %d tokens: TTFT %.2f s (%.1f tok/s)\n", T, prefill_ms / 1000, T / (prefill_ms / 1000));
  brk("prefill", pst, 1);
  fflush(stdout);
  if (!full) {  // partial layer range: no head — decode steps with prompt tokens, for the per-section timing of the loaded layers only
    eng.stats() = CpuStats{};
    std::vector<double> ms;
    for (int i = 0; i < n_dec; ++i) {
      CpuSeq* sp1[1] = {&s};
      const int32_t tok = ids[i % T];
      const double ts = mono_ms();
      eng.decode(sp1, &tok, 1, logits.data());
      ms.push_back(mono_ms() - ts);
    }
    std::sort(ms.begin(), ms.end());
    if (!ms.empty()) printf("partial layer range: decode median %.2f ms/step over layers %d..%d\n", ms[ms.size() / 2], eng.first_layer(), eng.last_layer());
    brk("decode (partial)", eng.stats(), std::max(1, n_dec));
    return 0;
  }
  // session image round trip
  std::vector<uint8_t> state_ok;
  std::vector<std::vector<uint8_t>> img;
  if (getenv("GLM_CHECK_STATE")) img = s.save_bytes();
  eng.stats() = CpuStats{};
  std::vector<int32_t> dec;
  std::vector<double> step_ms;
  for (int i = 0; i < n_dec; ++i) {
    const float* lr = logits.data() + (size_t)i * V;
    const int32_t tok = argmax(lr, V);
    dec.push_back(tok);
    CpuSeq* sp1[1] = {&s};
    const double ts = mono_ms();
    eng.decode(sp1, &tok, 1, logits.data() + (size_t)(i + 1) * V);
    step_ms.push_back(mono_ms() - ts);
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
  printf("greedy:");
  for (int32_t t : dec) printf(" %d", t);
  printf("\n");
  if (!img.empty() && n_dec > 0) {
    CpuSeq s2;
    eng.init_seq(s2, T + n_dec + 64);
    s2.load_bytes(img);
    std::vector<float> lg(V);
    CpuSeq* sp1[1] = {&s2};
    eng.decode(sp1, &dec[0], 1, lg.data());
    printf("state round trip: first decode step after save / load %s\n", memcmp(lg.data(), logits.data() + V, (size_t)V * 4) == 0 ? "bit-identical" : "DIFFERS");
  }
  if (const char* mn = getenv("GLM_CHECK_MTP_NEXT"); mn && *mn && eng.mtp_on()) {
    CpuSeq sm;
    eng.init_seq(sm, T + 16);
    std::vector<float> lg(V), ml(V);
    eng.prefill(sm, ids.data(), T, lg.data());
    int32_t d[1]; float cf[1];
    const int32_t nxt = atoi(mn);
    if (eng.mtp_draft(sm, nxt, 1, d, cf, ml.data()) == 1) {
      write_npy_f32(out + "/mtp_logits.npy", ml.data(), 1, V);
      printf("MTP draft of entry %d (next token %d): %d (p %.3f)\n", T - 1, nxt, d[0], cf[0]);
    }
  }
  if (const char* bs = getenv("GLM_CHECK_BATCH"); bs && atoi(bs) > 1 && n_dec > 1) {
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
  if (use_mtp && eng.mtp_on() && n_dec > 1) {
    const int k = std::max(1, eng.mtp_k());
    CpuSeq ss;
    eng.init_seq(ss, T + n_dec + 32);
    std::vector<float> lg((size_t)(k + 1) * V);
    eng.prefill(ss, ids.data(), T, lg.data());
    std::vector<int32_t> emitted{(int32_t)argmax(lg.data(), V)};
    int rounds = 0, accepted = 0, proposed = 0;
    eng.stats() = CpuStats{};
    double draft_ms = 0, verify_ms = 0;
    const double ts = mono_ms();
    while ((int)emitted.size() < n_dec) {
      const int32_t tok = emitted.back();
      int32_t dr[8]; float cf[8];
      const double td = mono_ms();
      const int nd = eng.mtp_draft(ss, tok, k, dr, cf);
      draft_ms += mono_ms() - td;
      std::vector<int32_t> vi{tok};
      for (int j = 0; j < nd; ++j) vi.push_back(dr[j]);
      CpuSeq* sp1[1] = {&ss};
      const int Mv = (int)vi.size();
      const double tv = mono_ms();
      eng.verify_batch(sp1, &Mv, 1, vi.data(), lg.data());
      int keep = 1;
      while (keep < Mv && argmax(lg.data() + (size_t)(keep - 1) * V, V) == vi[keep]) ++keep;
      eng.rollback_batch(&keep, 1);
      verify_ms += mono_ms() - tv;
      for (int j = 1; j < keep; ++j) emitted.push_back(vi[j]);
      emitted.push_back((int32_t)argmax(lg.data() + (size_t)(keep - 1) * V, V));
      ++rounds; accepted += keep - 1; proposed += nd;
    }
    const double wall = mono_ms() - ts;
    emitted.resize(n_dec);
    const bool same = emitted == dec;
    printf("MTP spec decode k=%d: %d tokens in %d rounds · accepted %d / %d drafts (%.1f %%) · %.2f tok/s · draft %.1f ms/round · verify+rollback %.1f ms/round · "
           "tokens %s greedy\n", k, n_dec, rounds, accepted, proposed, proposed ? 100.0 * accepted / proposed : 0.0, n_dec * 1000.0 / wall,
           draft_ms / rounds, verify_ms / rounds, same ? "==" : "!=");
    if (!same) { printf("  spec:"); for (int32_t t : emitted) printf(" %d", t); printf("\n"); }
  }
  if (const char* bl = getenv("GLM_CHECK_BENCH"); bl && *bl) {
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
      printf("bench prefill %d tokens: TTFT %.2f s (%.1f tok/s)\n", L, ms / 1000, L / (ms / 1000));
      brk("bench prefill", eng.stats(), 1);
      fflush(stdout);
    }
  }
  std::vector<double> sorted = step_ms;
  std::sort(sorted.begin(), sorted.end());
  const double med = sorted.empty() ? 0 : sorted[sorted.size() / 2];
  {
    FILE* f = fopen((out + "/timing.json").c_str(), "w");
    fprintf(f, "{\"load_s\": %.3f, \"prompt_tokens\": %d, \"prefill_s\": %.3f, \"decode_step_s\": [", load_ms / 1000, T, prefill_ms / 1000);
    for (size_t i = 0; i < step_ms.size(); ++i) fprintf(f, "%s%.4f", i ? ", " : "", step_ms[i] / 1000);
    fprintf(f, "], \"decode_median_s\": %.4f}\n", med / 1000);
    fclose(f);
  }
  printf("glm_cpu_check: load %.1f s · prompt %d tokens, TTFT %.2f s · decode %d steps, median %.1f ms (%.2f tok/s) · out %s\n", load_ms / 1000, T,
         prefill_ms / 1000, n_dec, med, med > 0 ? 1000.0 / med : 0.0, out.c_str());
  brk("decode", dec_st, std::max(1, n_dec));
  return 0;
}
