// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash CPU engine (hive/ds/ds_cpu.h) validation and timing driver. Runs a raw golden's steps (tools/compare_ds_cpu.py: the
//   prompt, then every decode step's token — teacher forced) and writes the same files the golden has (s{k}_h_L{l}.f32 + s{k}_pos_L{l}.i64
//   with the rows' positions, premix, engram, routing / index json, logits) for tools/compare_ds_cpu.py compare.
// usage: ds_cpu_check <ckpt> <golden dir | -> <out dir> [options]
//   --engram DIR       engram hash (engram_hash.json + token_map.bin; the oracle's export)
//   --max-layer N      load layers 0..N only (no head)
//   --inject           feed every layer l ≥ 1 the golden's output of layer l − 1 (per-layer isolation)
//   --steps K          decode steps to replay (default: all of the golden)
//   --images           the golden's image_inputs.json (prompt ids, types, patches)
//   --spec             DSpark check (oracle --spec golden): prefill, the decode step, then a draft at the golden's spec.json position
//   --ids LIST|FILE    prompt without a golden (timing / generation runs)
//   --gen N            greedy generation of N tokens after the prompt (decode timing, ms per step and the section breakdown)
//   --gen-mtp N        the same N tokens with DSpark draft → verify → rollback; checks the tokens equal plain greedy decoding
//   --no-dump          no per-layer files (timing runs)
//   --log-rows N       routing / index json for the last N rows of a step (default 64)
//   --batch B          after the prompt, B copies of the sequence decode together (batched decode timing; rows must agree)
//   --ab [--ab-reps N] before the prompt, prefill it N (4) times alternating the previous schedule (set_ab_legacy 3) and the current one; logits must be identical
//   --legacy MASK      run everything on the previous schedule (set_ab_legacy) — A/B across processes when one in-process A/B is too long
//   --short N          before the prompt, time a prefill of its first N tokens on a scratch sequence (short-prompt TTFT; run twice, the second counts)
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hive/clock.h"
#include "hive/ds/ds_cpu.h"
#include "hive/engram_hash.h"
#include "nlohmann/json.hpp"

using namespace hive;
using namespace hive::ds::cpu;
using json = nlohmann::json;

namespace {
void on_fatal(int sig) {
  void* fr[64];
  const int n = backtrace(fr, 64);
  dprintf(2, "ds_cpu_check: signal %d\n", sig);
  backtrace_symbols_fd(fr, n, 2);
  _exit(128 + sig);
}
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
int argmax(const float* r, int V) { return (int)(std::max_element(r, r + V) - r); }

struct Step { int64_t start = 0; int rows = 0; std::vector<int32_t> ids; };

// per-step dump state
struct Dumper {
  std::string out;
  int step = 0;
  int64_t start = 0, rows = 0;
  int log_rows = 64;
  bool on = true;
  std::map<std::string, FILE*> files;
  json route = json::object(), index = json::object();
  void begin(int k, int64_t s, int64_t r) { finish(); step = k; start = s; rows = r; route = json::object(); index = json::object(); }
  FILE* file(const std::string& name) {
    auto it = files.find(name);
    if (it != files.end()) return it->second;
    FILE* f = fopen((out + "/" + name).c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", name.c_str()); exit(2); }
    files[name] = f;
    return f;
  }
  void rows_out(const char* what, int layer, const int64_t* pos, int n, const float* data, int cols) {
    if (!on) return;
    const std::string base = "s" + std::to_string(step) + "_" + what + (layer >= 0 ? "_L" + std::to_string(layer) : "");
    fwrite(data, 4, (size_t)n * cols, file(base + ".f32"));
    fwrite(pos, 8, (size_t)n, file(base + ".pos"));
  }
  void route_out(int layer, const int64_t* pos, int n, const int32_t* ids, const float* w, int k) {
    char key[16]; snprintf(key, sizeof key, "L%02d", layer);
    json& e = route[key];
    if (e.is_null()) e = {{"rows", json::array()}, {"ids", json::array()}, {"w", json::array()}};
    for (int m = 0; m < n; ++m) {
      if (pos[m] < start + rows - log_rows) continue;
      e["rows"].push_back(pos[m] - start);
      e["ids"].push_back(std::vector<int32_t>(ids + (size_t)m * k, ids + (size_t)(m + 1) * k));
      e["w"].push_back(std::vector<float>(w + (size_t)m * k, w + (size_t)(m + 1) * k));
    }
  }
  void index_out(int layer, const int64_t* pos, int n, const int32_t* sel, int stride, const int* cnt) {
    char key[16]; snprintf(key, sizeof key, "L%02d", layer);
    json& e = index[key];
    if (e.is_null()) e = {{"rows", json::array()}, {"sel", json::array()}};
    for (int m = 0; m < n; ++m) {
      if (pos[m] < start + rows - log_rows) continue;
      e["rows"].push_back(pos[m] - start);
      e["sel"].push_back(std::vector<int32_t>(sel + (size_t)m * stride, sel + (size_t)m * stride + cnt[m]));
    }
  }
  void finish() {
    for (auto& [n, f] : files) fclose(f);
    files.clear();
    if (on && (!route.empty() || !index.empty())) {
      std::ofstream(out + "/s" + std::to_string(step) + "_route.json") << route.dump();
      std::ofstream(out + "/s" + std::to_string(step) + "_index.json") << index.dump();
    }
  }
};

// golden rows for injection: s{k}_h_L{l}.f32 rows of step k (row = position − step start)
struct Injector {
  std::string dir;
  std::vector<Step>* steps = nullptr;
  int D = 5120, HC = 4;
  int step_of(int64_t p) const {
    for (size_t k = 0; k < steps->size(); ++k) if (p >= (*steps)[k].start && p < (*steps)[k].start + (*steps)[k].rows) return (int)k;
    return -1;
  }
  bool read_row(const std::string& name, int64_t row, float* dst, size_t n) const {
    const int fd = open((dir + "/" + name).c_str(), O_RDONLY);
    if (fd < 0) return false;
    const ssize_t got = pread(fd, dst, n * 4, (off_t)(row * (int64_t)n * 4));
    close(fd);
    return got == (ssize_t)(n * 4);
  }
  void inject(int l, const int64_t* pos, int M, float* h, float* pm) const {
    int done = 0;
    for (int m = 0; m < M; ++m) {
      const int k = step_of(pos[m]);
      if (k < 0) continue;
      const int64_t row = pos[m] - (*steps)[k].start;
      const std::string pre = "s" + std::to_string(k) + "_";
      if (read_row(pre + "h_L" + std::to_string(l - 1) + ".f32", row, h + (size_t)m * HC * D, (size_t)HC * D) &&
          read_row(pre + "premix_L" + std::to_string(l - 1) + ".f32", row, pm + (size_t)m * HC, HC))
        ++done;
    }
    if (done != M && l == 1) fprintf(stderr, "[inject] layer %d: %d of %d rows from the golden\n", l, done, M);
  }
};
}  // namespace

int main(int argc, char** argv) {
  signal(SIGSEGV, on_fatal); signal(SIGABRT, on_fatal); signal(SIGBUS, on_fatal);
  if (argc < 4) { fprintf(stderr, "usage: ds_cpu_check <ckpt> <golden|-> <out> [options]\n"); return 2; }
  const std::string ckpt = argv[1], gdir = argv[2], out = argv[3];
  std::string engram_dir, ids_arg;
  int max_layer = -1, steps_max = -1, gen = 0, gen_mtp = 0, batch = 0, short_n = 0;
  bool inject = false, images = false, spec = false, ab = false;
  int ab_reps = 4, legacy_mask = 0;
  Dumper dmp;
  dmp.out = out;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
    if (a == "--engram") engram_dir = next();
    else if (a == "--max-layer") max_layer = atoi(next().c_str());
    else if (a == "--inject") inject = true;
    else if (a == "--steps") steps_max = atoi(next().c_str());
    else if (a == "--images") images = true;
    else if (a == "--spec") spec = true;
    else if (a == "--ids") ids_arg = next();
    else if (a == "--gen") gen = atoi(next().c_str());
    else if (a == "--gen-mtp") gen_mtp = atoi(next().c_str());
    else if (a == "--batch") batch = atoi(next().c_str());
    else if (a == "--short") short_n = atoi(next().c_str());
    else if (a == "--ab") ab = true;
    else if (a == "--legacy") legacy_mask = atoi(next().c_str());
    else if (a == "--ab-reps") ab_reps = std::max(2, atoi(next().c_str()));
    else if (a == "--no-dump") dmp.on = false;
    else if (a == "--log-rows") dmp.log_rows = atoi(next().c_str());
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  mkdir(out.c_str(), 0755);
  // golden steps
  std::vector<Step> steps;
  json meta;
  if (gdir != "-") {
    std::ifstream f(gdir + "/meta.json");
    if (!f) { fprintf(stderr, "no meta.json in %s\n", gdir.c_str()); return 2; }
    meta = json::parse(f);
    for (auto& s : meta["steps"]) steps.push_back(Step{s["start"].get<int64_t>(), s["rows"].get<int>(), s["ids"].get<std::vector<int32_t>>()});
  }
  if (!ids_arg.empty()) { steps.clear(); steps.push_back(Step{0, 0, parse_ids(ids_arg)}); steps[0].rows = (int)steps[0].ids.size(); }
  if (steps.empty()) { fprintf(stderr, "no prompt\n"); return 2; }
  if (steps_max >= 0 && (int)steps.size() > steps_max + 1) steps.resize(steps_max + 1);
  if (spec) steps.resize(std::min<size_t>(steps.size(), 2));

  const double t_load0 = mono_ms();
  EngineOptions opt = EngineOptions::from_env();
  opt.max_layer = max_layer;
  DsCpuEngine eng(ckpt, opt);
  const DsConfig& c = eng.cfg();
  EngramHash eh;
  if (!engram_dir.empty()) { eh.load(engram_dir + "/engram_hash.json", engram_dir + "/token_map.bin"); eng.set_engram_hash(&eh); }
  else if (!c.engram_layer_ids.empty() && c.engram_layer_ids[0] < eng.n_loaded_layers()) fprintf(stderr, "⚠ no --engram: engram layers are skipped\n");
  if ((spec || gen_mtp) && eng.has_head()) eng.load_mtp();
  const double t_load = mono_ms() - t_load0;
  fprintf(stderr, "[check] load %.1fs · layers %d · head %d · DSpark stages %d\n", t_load / 1000.0, eng.n_loaded_layers(), eng.has_head(), eng.n_mtp());

  eng.set_ab_legacy(legacy_mask);   // --legacy MASK: the whole run on the previous schedule (A/B across processes)
  eng.dump_hook = [&](const char* what, int l, const int64_t* pos, int n, const float* d, int cols) { dmp.rows_out(what, l, pos, n, d, cols); };
  eng.route_hook = [&](int l, const int64_t* pos, int n, const int32_t* ids, const float* w, int k) { if (dmp.on) dmp.route_out(l, pos, n, ids, w, k); };
  eng.index_hook = [&](int l, const int64_t* pos, int n, const int32_t* sel, int stride, const int* cnt) { if (dmp.on) dmp.index_out(l, pos, n, sel, stride, cnt); };
  Injector inj;
  inj.dir = gdir; inj.steps = &steps; inj.D = c.dim; inj.HC = c.hc;
  if (inject) eng.layer_inject = [&](int l, const int64_t* pos, int M, float* h, float* pm) { inj.inject(l, pos, M, h, pm); };
  std::map<std::string, std::vector<float>> spec_arrays;
  eng.spec_hook = [&](const char* name, const float* d, size_t n) { spec_arrays[name].assign(d, d + n); };
  eng.vision_hook = [&](const char* name, const float* d, size_t n) {
    if (!dmp.on) return;
    FILE* f = fopen((out + "/s0_" + name + ".f32").c_str(), "wb");
    if (f) { fwrite(d, 4, n, f); fclose(f); }
  };

  const int V = c.vocab;
  std::vector<float> logits((size_t)8 * V);
  int64_t cap = 0;
  for (auto& s : steps) cap = std::max<int64_t>(cap, s.start + s.rows);
  cap += std::max(gen, gen_mtp) + 64;
  CpuSeq seq;
  eng.init_seq(seq, cap);
  json timing = json::object();

  // ---- prompt -------------------------------------------------------------------------------------------------------------------------
  std::vector<DsCpuEngine::Image> imgs;
  std::vector<std::vector<uint16_t>> patch_bufs;
  if (images) {
    std::ifstream f(gdir + "/image_inputs.json");
    if (!f) { fprintf(stderr, "no image_inputs.json in %s\n", gdir.c_str()); return 2; }
    json ii = json::parse(f);
    steps[0].ids = ii["ids"].get<std::vector<int32_t>>();
    for (auto& im : ii["images"]) {
      DsCpuEngine::Image x;
      x.start = im["start"].get<int>(); x.n_vit_h = im["n_vit_h"].get<int>(); x.n_vit_w = im["n_vit_w"].get<int>();
      for (int t : im["types"].get<std::vector<int>>()) x.types.push_back((int8_t)t);
      std::ifstream pf(im["file"].get<std::string>(), std::ios::binary);
      std::vector<uint16_t> buf((size_t)x.n_vit_h * x.n_vit_w * 3 * c.vision_patch * c.vision_patch);
      pf.read((char*)buf.data(), buf.size() * 2);
      if (!pf) { fprintf(stderr, "short patch file\n"); return 2; }
      patch_bufs.push_back(std::move(buf));
      x.patches = patch_bufs.back().data();
      imgs.push_back(std::move(x));
    }
  }
  if (short_n > 0 && eng.has_head()) {
    const bool was = dmp.on;
    dmp.on = false;
    std::vector<double> ms;
    for (int rep = 0; rep < 2; ++rep) {
      CpuSeq tmp;
      eng.init_seq(tmp, short_n + 8);
      const double ts = mono_ms();
      eng.prefill(tmp, steps[0].ids.data(), std::min<int>(short_n, (int)steps[0].ids.size()), nullptr, logits.data());
      ms.push_back(mono_ms() - ts);
    }
    timing["short_prefill_tokens"] = short_n;
    timing["short_prefill_ms"] = ms;
    fprintf(stderr, "[check] short prefill %d tokens: %.1f ms (first run %.1f ms)\n", short_n, ms[1], ms[0]);
    dmp.on = was;
  }
  if (ab && eng.has_head()) {
    const bool was = dmp.on;
    dmp.on = false;
    std::vector<double> ms[2];
    std::vector<float> ref;
    bool same = true;
    for (int rep = 0; rep < ab_reps; ++rep) {
      const int legacy = rep % 2 == 0 ? 3 : 0;
      eng.set_ab_legacy(legacy);
      CpuSeq tmp;
      eng.init_seq(tmp, (int64_t)steps[0].ids.size() + 8);
      std::vector<float> lg(V);
      const double ts = mono_ms();
      eng.prefill(tmp, steps[0].ids.data(), (int)steps[0].ids.size(), nullptr, lg.data());
      ms[legacy ? 0 : 1].push_back(mono_ms() - ts);
      if (ref.empty()) ref = lg; else same = same && memcmp(ref.data(), lg.data(), (size_t)V * 4) == 0;
    }
    eng.set_ab_legacy(legacy_mask);
    timing["ab"] = {{"legacy_ms", ms[0]}, {"new_ms", ms[1]}, {"logits_bit_identical", same}};
    fprintf(stderr, "[check] A/B prefill %zu tokens: previous %s ms · current %s ms · logits bit-identical: %s\n", steps[0].ids.size(),
            json(ms[0]).dump().c_str(), json(ms[1]).dump().c_str(), same ? "yes" : "NO");
    dmp.on = was;
  }
  dmp.begin(0, 0, (int64_t)steps[0].ids.size());
  const CpuStats st0 = eng.stats();
  double t0 = mono_ms();
  eng.prefill(seq, steps[0].ids.data(), (int)steps[0].ids.size(), imgs.empty() ? nullptr : &imgs, eng.has_head() ? logits.data() : nullptr);
  const double t_prefill = mono_ms() - t0;
  auto sect = [&](const CpuStats& a, const CpuStats& b) {
    return json{{"total", b.ms_total - a.ms_total}, {"engram", b.ms_engram - a.ms_engram}, {"hc", b.ms_hc - a.ms_hc}, {"attn_proj", b.ms_attn_proj - a.ms_attn_proj},
                {"compress", b.ms_compress - a.ms_compress}, {"index", b.ms_index - a.ms_index}, {"attn", b.ms_attn - a.ms_attn},
                {"attn_out", b.ms_attn_out - a.ms_attn_out}, {"router", b.ms_router - a.ms_router}, {"moe", b.ms_moe - a.ms_moe}, {"head", b.ms_head - a.ms_head},
                {"mtp", b.ms_mtp - a.ms_mtp}};
  };
  timing["prompt_tokens"] = steps[0].ids.size();
  timing["prefill_ms"] = t_prefill;
  timing["prefill_sections"] = sect(st0, eng.stats());
  int next = eng.has_head() ? argmax(logits.data(), V) : -1;
  fprintf(stderr, "[check] prefill %zu tokens: %.1f ms (%.1f tok/s) · next %d\n", steps[0].ids.size(), t_prefill, steps[0].ids.size() / (t_prefill / 1000.0), next);
  std::vector<int32_t> generated;
  if (next >= 0) generated.push_back(next);

  // ---- teacher-forced decode steps of the golden ------------------------------------------------------------------------------------------
  std::vector<double> step_ms;
  for (size_t k = 1; k < steps.size(); ++k) {
    const int32_t tok = steps[k].ids.empty() ? next : steps[k].ids[0];
    dmp.begin((int)k, seq.pos, 1);
    CpuSeq* sp = &seq;
    t0 = mono_ms();
    eng.decode(&sp, &tok, 1, logits.data());
    step_ms.push_back(mono_ms() - t0);
    if (eng.has_head()) {
      next = argmax(logits.data(), V);
      generated.push_back(next);
    }
    fprintf(stderr, "[check] step %zu: token %d at pos %lld → next %d · %.1f ms\n", k, tok, (long long)(seq.pos - 1), next, step_ms.back());
  }
  dmp.finish();
  dmp.on = false;   // the generation / DSpark phases below must not overwrite the golden steps' dumps
  timing["golden_step_ms"] = step_ms;

  // ---- DSpark check: draft at the golden's position ---------------------------------------------------------------------------------------
  if (spec && eng.mtp_on()) {
    std::ifstream sf(gdir + "/spec.json");
    json sj = sf ? json::parse(sf) : json::object();
    const int32_t tok = sj.contains("tok") ? sj["tok"].get<int32_t>() : next;
    std::vector<int32_t> drafts(c.dspark_block);
    std::vector<float> conf(c.dspark_block);
    const std::vector<float> main_x = spec_arrays["sync_main_x"];   // the last sync = the decode step's row
    t0 = mono_ms();
    eng.mtp_draft(seq, tok, drafts.data(), conf.data());
    const double tm = mono_ms() - t0;
    json js = {{"tok", tok}, {"pos", seq.pos}, {"ids", json::array()}, {"conf", conf}, {"ms", tm}};
    js["ids"].push_back(tok);
    for (int32_t d : drafts) js["ids"].push_back(d);
    std::ofstream(out + "/spec.json") << js.dump();
    spec_arrays["spec_main_x"] = main_x;
    for (auto& [name, v] : spec_arrays) {
      FILE* f = fopen((out + "/" + name + ".f32").c_str(), "wb");
      fwrite(v.data(), 4, v.size(), f);
      fclose(f);
    }
    fprintf(stderr, "[check] DSpark at pos %lld tok %d: drafts", (long long)seq.pos, tok);
    for (int32_t d : drafts) fprintf(stderr, " %d", d);
    fprintf(stderr, " · %.1f ms\n", tm);
  }

  // ---- greedy generation (timing) and the DSpark speculative loop (must reproduce it) -------------------------------------------------
  if ((gen > 0 || gen_mtp > 0) && eng.has_head()) {
    CpuSeq base_state = seq;   // the state after the prompt (and golden steps)
    const int32_t first = next;
    std::vector<int32_t> greedy;
    if (gen > 0) {
      std::vector<double> ms;
      const CpuStats s0 = eng.stats();
      int32_t tok = first;
      for (int i = 0; i < gen; ++i) {
        greedy.push_back(tok);
        CpuSeq* sp = &seq;
        t0 = mono_ms();
        eng.decode(&sp, &tok, 1, logits.data());
        ms.push_back(mono_ms() - t0);
        tok = argmax(logits.data(), V);
      }
      greedy.push_back(tok);
      const CpuStats s1 = eng.stats();
      std::vector<double> sorted = ms;
      std::sort(sorted.begin(), sorted.end());
      const double med = sorted[sorted.size() / 2];
      timing["decode_c1_ms"] = ms;
      timing["decode_c1_median_ms"] = med;
      json secs = sect(s0, s1);
      for (auto& [k2, v] : secs.items()) v = v.get<double>() / gen;
      timing["decode_c1_sections_per_step"] = secs;
      timing["greedy"] = greedy;
      fprintf(stderr, "[check] greedy %d tokens: median %.1f ms/step (%.2f tok/s) · per step: %s\n", gen, med, 1000.0 / med, secs.dump().c_str());
    }
    if (batch > 1 && gen > 0) {
      std::vector<CpuSeq> bs(batch, base_state);
      std::vector<CpuSeq*> ptr;
      for (auto& s : bs) ptr.push_back(&s);
      std::vector<int32_t> toks(batch, first);
      std::vector<double> ms;
      bool same = true;
      for (int i = 0; i < std::min(gen, 16); ++i) {
        t0 = mono_ms();
        eng.decode(ptr.data(), toks.data(), batch, logits.data());
        ms.push_back(mono_ms() - t0);
        for (int b = 0; b < batch; ++b) { toks[b] = argmax(logits.data() + (size_t)b * V, V); same = same && toks[b] == toks[0]; }
        if (i + 1 < (int)greedy.size()) same = same && toks[0] == greedy[i + 1];
      }
      std::sort(ms.begin(), ms.end());
      timing["decode_batch"] = {{"rows", batch}, {"median_ms", ms[ms.size() / 2]}, {"rows_agree_with_c1", same}};
      fprintf(stderr, "[check] batched decode %d rows: median %.1f ms/step (%.2f tok/s aggregate) · rows equal to c1 greedy: %s\n", batch, ms[ms.size() / 2],
              batch * 1000.0 / ms[ms.size() / 2], same ? "yes" : "NO");
    }
    if (gen_mtp > 0 && eng.mtp_on()) {
      CpuSeq s = base_state;
      std::vector<int32_t> toks{first};
      int drafted = 0, accepted = 0, steps_n = 0;
      std::vector<int32_t> drafts(c.dspark_block);
      std::vector<float> conf(c.dspark_block);
      t0 = mono_ms();
      while ((int)toks.size() < gen_mtp + 1) {
        const int32_t tok = toks.back();
        eng.mtp_draft(s, tok, drafts.data(), conf.data());
        std::vector<int32_t> row{tok};
        const int k = std::min<int>(c.dspark_block, 7);
        row.insert(row.end(), drafts.begin(), drafts.begin() + k);
        const int R = (int)row.size();
        CpuSeq* sp = &s;
        eng.verify_batch(&sp, &R, 1, row.data(), logits.data());
        int keep = 1;
        std::vector<int32_t> out;
        for (int i = 0; i < R; ++i) {
          const int32_t a = argmax(logits.data() + (size_t)i * V, V);
          out.push_back(a);
          if (i + 1 < R && a == row[i + 1]) ++keep; else break;
        }
        eng.rollback_batch(&keep, 1);
        drafted += k; accepted += keep - 1; ++steps_n;
        for (int32_t t : out) toks.push_back(t);
      }
      const double tm = mono_ms() - t0;
      toks.resize(gen_mtp + 1);
      bool same = true;
      for (size_t i = 0; i < std::min(toks.size(), greedy.size()); ++i) same = same && toks[i] == greedy[i];
      timing["mtp"] = {{"tokens", gen_mtp}, {"steps", steps_n}, {"drafted", drafted}, {"accepted", accepted}, {"ms", tm}, {"tokens_equal_greedy", same}, {"tokens_out", toks}};
      fprintf(stderr, "[check] DSpark loop: %d tokens in %d verify steps (accepted %d of %d drafts, %.2f tokens/step) · %.1f ms/token · tokens equal greedy: %s\n",
              gen_mtp, steps_n, accepted, drafted, (double)gen_mtp / steps_n, tm / gen_mtp, greedy.empty() ? "n/a" : (same ? "yes" : "NO"));
    }
  }
  timing["generated"] = generated;
  timing["load_ms"] = t_load;
  std::ofstream(out + "/timing.json") << timing.dump(1);
  if (gdir != "-") {
    json m = meta;
    std::ofstream(out + "/meta.json") << m.dump();
  }
  fprintf(stderr, "[check] done → %s\n", out.c_str());
  return 0;
}
