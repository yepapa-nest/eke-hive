// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// R2 short-prefill (TTFT) A/B benchmark — GPU only, run on the GPU host with the hived service stopped (one load, ~4–8 minutes).
//   Prefills 1K–4K prompts as a single chunk with the real weights and a real cache state (hived warm-start file --cache-state),
//   alternating switch combinations (runtime.h ShortPrefillOpts) **within one load** (same token windows · same cache):
//     base  = all off (= service default)        tail  = HIVE_PREFILL_TAIL_SHORT       scale = HIVE_PREFETCH_SCALE
//     adapt = HIVE_PREFILL_SHORT_ADAPT (=--adapt-rows) all = all three   (listing base twice gives the A vs A noise floor — the default list does)
//     name@N = that variant + a per-layer DMA cap N for short prefill (HIVE_SHORT_DMA_CAP for that run only — e.g. tail@32,tail@96 —
//       to find the share cap of the short tail path)
//   Q3: variant names can be joined with '+' (e.g. small+adapt+scale@16). Two features:
//     small = HIVE_PREFILL_SMALL (floor --small-rows · default = the floor of switch value "1", max(max_batch+1, 9)) — experts of
//       chunks below threshold go through the streaming path
//     mtail = HIVE_PREFILL_MULTI_TAIL_SHORT — upper-layer units of forward_multi on the short path. Only meaningful in the two cases
//       that use forward_multi: H2 host tiles (HIVE_PREFILL_HOST_TILES on and a length with M > max_chunk × prefill_tile — e.g.
//       --lens 60000) · H3 (--multi P + HIVE_BATCH_PREFILL=1 — P sequences of the same length in one forward_multi; logits are
//       compared per sequence and the worst is reported)
//   Measuring the 1K cliff (observed in service: 906 tokens 7.3 s > 4,213 tokens 4.2 s): --lens 256,512,1000,2048,4096 --variants base,small,small+adapt,small+adapt+scale,base
//   Instrumentation: HIVE_PREFILL_PROF is on for every forward → [prefill-prof] stage breakdown on stderr (runtime.cpp PrefillProf).
//     At the end: median TTFT per length × variant and % vs base.
//   Correctness (lossless check): at the same repetition and length, compare the variant's last-token logits with the first base
//     run — argmax match · top-5 overlap · max|Δ| · KL(base‖variant).
//     The difference between the two base runs (A vs A) is the "run-to-run floor of the same computation" (when the CPU/GPU split
//     adapts, even base runs may differ in fp32 accumulation order).
//     The switches only change the fp32 accumulation order (runtime.cpp R2 comment) — the criterion is whether the difference is of
//     the same size as base-vs-base (bit identity is not expected).
//   Fixed cache: this benchmark disables promotion (promote 0 · warm_cache not called) — prefills with rows ≥ prefill_threshold never
//     promote anyway, and this keeps the cache unchanged for short lengths too.
//   --warm-prev N: before each measurement, prefill N tokens into another sequence (not measured) — the "short chunk after a long
//     chunk" situation (the copy-ahead prediction comes from the previous long chunk; target of HIVE_PREFETCH_SCALE).
//   Example (export the same HIVE_* environment as the server first):
//     scripts/hive-run.sh "./build-dev/bench_short_prefill --ckpt /path/to/DeepSeek-V4.1-Flash --engram /out/engram --cache-state /out/cache-state.bin
//        --ids-file /out/prompt-ids.txt --lens 1024,2048,4096 --reps 3 --vram-cache-mb 68000 --cpu-threads 16 --max-chunk 16384 --prefill-tile 3 --decoder-replay"
//     Q3 examples: ... --lens 256,512,1000,2048,4096 --variants base,small,small+adapt,small+adapt+scale,base
//             HIVE_BATCH_PREFILL=1 ... --multi 2 --lens 4096,8192 --variants base,mtail,mtail@32,base
//             HIVE_PREFILL_HOST_TILES=2 ... --lens 60000 --reps 2 --variants base,mtail,base
#include <algorithm>
#include <chrono>
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

#include "hive/engram_hash.h"
#include "hive/expert_store.h"
#include "hive/model.h"
#include "hive/runtime.h"
#include "hive/short_prefill.h"
#include "hive/clock.h"

using namespace hive;

namespace {

double now_ms() { return hive::mono_ms(); }

std::vector<int32_t> parse_list(const std::string& s) {
  std::vector<int32_t> v;
  std::string cur;
  for (char ch : s + ",") {
    if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') { if (!cur.empty()) { v.push_back(atoi(cur.c_str())); cur.clear(); } }
    else cur += ch;
  }
  return v;
}

struct Variant { std::string name; Runtime::ShortPrefillOpts o; int dma_cap = 0; };  // dma_cap = per-layer DMA cap for short prefill (0 = constructor value — HIVE_SHORT_DMA_CAP)

// Logit comparison: argmax · top-5 overlap · max|Δ| · KL(a‖b) (log-softmax, double)
struct Cmp { bool argmax_eq = false; int top5 = 0; double max_abs = 0, kl = 0; };
using Logits = std::vector<float>;
Cmp compare(const Logits& a, const Logits& b) {
  Cmp r;
  if (a.empty() || a.size() != b.size()) { r.max_abs = INFINITY; r.kl = INFINITY; return r; }
  const size_t V = a.size();
  auto top = [&](const std::vector<float>& x) {
    std::vector<int> id(V);
    for (size_t i = 0; i < V; ++i) id[i] = (int)i;
    std::partial_sort(id.begin(), id.begin() + 5, id.end(), [&](int p, int q) { return x[p] > x[q]; });
    id.resize(5);
    return id;
  };
  const auto ta = top(a), tb = top(b);
  r.argmax_eq = ta[0] == tb[0];
  for (int i : ta) r.top5 += std::count(tb.begin(), tb.end(), i) ? 1 : 0;
  double ma = -INFINITY, mb = -INFINITY;
  for (size_t i = 0; i < V; ++i) { ma = std::max(ma, (double)a[i]); mb = std::max(mb, (double)b[i]); r.max_abs = std::max(r.max_abs, std::fabs((double)a[i] - b[i])); }
  double sa = 0, sb = 0;
  for (size_t i = 0; i < V; ++i) { sa += std::exp(a[i] - ma); sb += std::exp(b[i] - mb); }
  const double la = std::log(sa) + ma, lb = std::log(sb) + mb;
  for (size_t i = 0; i < V; ++i) { const double pa = a[i] - la, pb = b[i] - lb; r.kl += std::exp(pa) * (pa - pb); }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = getenv("HIVE_CKPT") ? getenv("HIVE_CKPT") : "", engram_dir, cache_state, ids_file, lens_s = "256,512,1000,2048,4096",
              variants_s = "base,small,small+adapt,tail,all,base";  // Q3 default lengths and variants (1K cliff — see the header comment)
  int reps = 3, cpu_threads = 16, max_chunk = 16384, prefill_tile = 1, adapt_rows = 4096, warm_prev = 0, prefill_threshold = -1, decoder_tail = -1, max_batch = 8;
  int small_rows = 0, multi = 1;  // Q3 --small-rows (0 = floor of switch value "1") · --multi P (number of H3 forward_multi sequences)
  double cache_mb = 60000;
  int64_t max_ctx = 131072;
  bool decoder_replay = false, no_mtp = false;
  for (int i = 1; i < argc; i++) {
    const std::string a(argv[i]);
    auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "%s: missing value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--engram") engram_dir = next();
    else if (a == "--cache-state") cache_state = next();
    else if (a == "--ids-file") ids_file = next();
    else if (a == "--lens") lens_s = next();
    else if (a == "--variants") variants_s = next();
    else if (a == "--reps") reps = std::max(1, atoi(next().c_str()));
    else if (a == "--cpu-threads") cpu_threads = atoi(next().c_str());
    else if (a == "--vram-cache-mb") cache_mb = atof(next().c_str());
    else if (a == "--max-ctx") max_ctx = atoll(next().c_str());
    else if (a == "--max-chunk") max_chunk = atoi(next().c_str());
    else if (a == "--max-batch") max_batch = atoi(next().c_str());
    else if (a == "--prefill-tile") prefill_tile = std::max(1, atoi(next().c_str()));
    else if (a == "--prefill-threshold") prefill_threshold = atoi(next().c_str());
    else if (a == "--decoder-tail") decoder_tail = atoi(next().c_str());
    else if (a == "--decoder-replay") decoder_replay = true;
    else if (a == "--adapt-rows") adapt_rows = std::max(2, atoi(next().c_str()));
    else if (a == "--warm-prev") warm_prev = std::max(0, atoi(next().c_str()));
    else if (a == "--no-mtp") no_mtp = true;
    else if (a == "--small-rows") small_rows = std::max(0, atoi(next().c_str()));
    else if (a == "--multi") multi = std::max(1, atoi(next().c_str()));
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (ckpt.empty()) { fprintf(stderr, "--ckpt DIR (or HIVE_CKPT) is required\n"); return 2; }
  const std::vector<int32_t> lens = parse_list(lens_s);
  std::vector<Variant> vars;
  { std::stringstream ss(variants_s); std::string v;
    while (std::getline(ss, v, ',')) {
      Variant x{v, {}};
      x.o.prof = true;
      std::string kind = v;
      if (const size_t at = v.find('@'); at != std::string::npos) { kind = v.substr(0, at); x.dma_cap = std::max(1, atoi(v.c_str() + at + 1)); }  // e.g. tail@32
      std::stringstream fs(kind);  // Q3: features joined with '+'
      std::string f;
      bool bad = false;
      while (std::getline(fs, f, '+')) {
        if (f == "base") {}
        else if (f == "tail") x.o.tail_short = true;
        else if (f == "scale") x.o.prefetch_scale = true;
        else if (f == "adapt") x.o.adapt_rows = adapt_rows;
        else if (f == "all") { x.o.tail_short = true; x.o.prefetch_scale = true; x.o.adapt_rows = adapt_rows; }
        else if (f == "small") x.o.small_rows = -1;  // the value is filled in after loading (once max_batch is known)
        else if (f == "mtail") x.o.multi_tail_short = true;
        else bad = true;
      }
      if (bad) { fprintf(stderr, "unknown variant %s (base|tail|scale|adapt|all|small|mtail joined with '+' · [@short DMA cap])\n", v.c_str()); return 2; }
      vars.push_back(x);
    } }
  if (lens.empty() || vars.empty()) { fprintf(stderr, "--lens / --variants is empty\n"); return 2; }
  const int max_len = *std::max_element(lens.begin(), lens.end());
  std::vector<int32_t> ids;
  if (!ids_file.empty()) { std::ifstream f(ids_file); std::stringstream ss; ss << f.rdbuf(); ids = parse_list(ss.str()); }
  const size_t need = (size_t)max_len * reps * multi + (size_t)warm_prev + 1;
  if (ids.size() < need) {
    fprintf(stderr, "[bench] WARNING: ids %zu < needed %zu — padding with deterministic pseudo-tokens (not a natural-language distribution — cache hits and routing differ from real use)\n", ids.size(), need);
    for (size_t i = ids.size(); i < need; ++i) ids.push_back(1000 + (int32_t)((i * 7919u) % 100000u));
  }

  fprintf(stderr, "[bench] ckpt=%s lens=%s reps=%d variants=%s\n", ckpt.c_str(), lens_s.c_str(), reps, variants_s.c_str());
  const double t_load = now_ms();
  Model model(ckpt, -1, max_ctx);
  const Config& c = model.cfg();
  const int nl = model.n_loaded_layers();
  if (!no_mtp && nl == c.n_layers && model.has_head()) model.load_mtp();
  const int n_mtp = model.n_mtp();
  ExpertStore store(c, nl, (size_t)(cache_mb * 1048576.0), cpu_threads, n_mtp);
  for (int l = 0; l < nl + n_mtp; ++l) store.load_layer_experts(model.ckpt(), l);
  store.pin_all();
  EngramHash eh;
  bool have_eh = false;
  for (size_t i = 0; i < c.engram_layer_ids.size(); ++i) {
    const int el = c.engram_layer_ids[i];
    if (el >= nl) continue;
    if (engram_dir.empty()) { fprintf(stderr, "layer %d needs engram — --engram DIR\n", el); return 2; }
    if (!have_eh) { eh.load(engram_dir + "/engram_hash.json", engram_dir + "/token_map.bin"); have_eh = true; }
    store.load_engram(model.ckpt(), el, (int)i);
  }
  RuntimeOptions opt;
  opt.max_chunk = max_chunk;
  opt.prefill_tile = prefill_tile;
  opt.max_ctx = max_ctx;
  opt.max_batch = max_batch;
  opt.vision = false;
  opt.promote_per_token = 0;  // fixed cache (see the header comment)
  opt.promote_misses = 0;
  if (decoder_tail >= 0) opt.decoder_tail = decoder_tail;
  if (decoder_replay) { opt.decoder_replay = true; if (decoder_tail < 0) opt.decoder_tail = c.window; }
  if (prefill_threshold > 0) opt.prefill_threshold = prefill_threshold;
  opt.mtp = !no_mtp;
  Runtime rt(model, store, have_eh ? &eh : nullptr, opt);
  if (!cache_state.empty()) {
    FILE* f = fopen(cache_state.c_str(), "rb");
    std::vector<int32_t> keys;
    if (f) {  // same format as the hived warm-start file (hived.cpp save_cache_state)
      int32_t hdr[4] = {0, 0, 0, 0};
      if (fread(hdr, 4, 4, f) == 4 && hdr[0] == 0x48564353 && hdr[1] == store.n_layers() && hdr[2] == store.E() && hdr[3] > 0 && hdr[3] < 100000) {
        keys.resize(hdr[3]);
        if (fread(keys.data(), 4, keys.size(), f) != keys.size()) keys.clear();
      }
      fclose(f);
    }
    if (keys.empty()) fprintf(stderr, "[bench] WARNING: could not read cache-state %s — measuring with an empty cache (hits near 0, unlike real use)\n", cache_state.c_str());
    else fprintf(stderr, "[bench] warm start: %d/%zu experts\n", rt.warm_from_keys(keys), keys.size());
  } else fprintf(stderr, "[bench] WARNING: no --cache-state — measuring with an empty cache\n");
  fprintf(stderr, "[bench] ready in %.0f s · env switches (constructor values) tail %d scale %d adapt %d prof %d — variants override them\n", (now_ms() - t_load) / 1000,
          rt.short_prefill_opts().tail_short, rt.short_prefill_opts().prefetch_scale, rt.short_prefill_opts().adapt_rows, rt.short_prefill_opts().prof);
  const Runtime::ShortPrefillOpts env_opts = rt.short_prefill_opts();
  const int env_cap = rt.short_dma_cap();
  {  // Q3 small floor: with --small-rows N use that value (max(N, max_batch+1, 9), as the switch parsing does) · otherwise the floor of switch value "1"
    const int floor = sp::parse_small(small_rows > 0 ? std::to_string(small_rows).c_str() : "1", rt.max_batch());
    for (Variant& v : vars) if (v.o.small_rows < 0) v.o.small_rows = floor;
    fprintf(stderr, "[bench] small floor %d rows · env small %d mtail %d · prefill_threshold %d · multi %d (batch_prefill %d · slots %d)\n", floor, env_opts.small_rows,
            env_opts.multi_tail_short, rt.opt().prefill_threshold, multi, rt.batch_prefill(), rt.prefill_slots());
  }
  if (multi > 1) {  // H3: multi sequences of the same length must fit in one forward_multi (same formula as hived's condition — otherwise stop here: benchmark argument error)
    if (!rt.batch_prefill()) { fprintf(stderr, "--multi requires HIVE_BATCH_PREFILL=1\n"); return 2; }
    for (int M : lens) if (rt.slots_for(M) * multi > rt.prefill_slots()) { fprintf(stderr, "--multi %d × M=%d: slots %d > %d\n", multi, M, rt.slots_for(M) * multi, rt.prefill_slots()); return 2; }
  }

  std::vector<std::unique_ptr<Seq>> seqs;
  for (int i = 0; i < multi; ++i) seqs.push_back(rt.new_seq());
  auto& seq = seqs[0];
  auto seq_prev = warm_prev > 0 ? rt.new_seq() : nullptr;
  std::vector<std::vector<float>> part_logits;  // Q3 --multi: logits of sequences 1.. (sequence 0 is in logits)
  // One run: (optional) a preceding long chunk → the measured chunk. Returns wall ms · logits
  auto run_one = [&](const Variant& v, const int32_t* p, int M, const int32_t* prev, std::vector<float>& logits, ForwardStats& st) {
    Runtime::ShortPrefillOpts o = v.o;
    if (prev) {  // the preceding chunk uses the base settings (only its copy-ahead prediction remains · no instrumentation)
      Runtime::ShortPrefillOpts b{};
      rt.set_short_prefill_opts(b);
      rt.set_short_dma_cap(env_cap);
      rt.reset_seq(*seq_prev);
      ForwardStats s0{};
      for (int off = 0; off < warm_prev;) {
        const int m = std::min(warm_prev - off, max_chunk * rt.prefill_slots());
        rt.forward(*seq_prev, prev + off, m, nullptr, &s0);
        off += m;
      }
    }
    rt.set_short_prefill_opts(o);
    rt.set_short_dma_cap(v.dma_cap > 0 ? v.dma_cap : env_cap);
    for (auto& q : seqs) rt.reset_seq(*q);
    st = ForwardStats{};
    if (multi > 1) {  // Q3 H3: sequence i = token window p + i·max_len·reps (different texts)
      part_logits.assign((size_t)multi, {});
      std::vector<PrefillPart> parts((size_t)multi);
      for (int i = 0; i < multi; ++i) {
        parts[i].seq = seqs[i].get(); parts[i].ids = p + (size_t)i * max_len * reps; parts[i].M = M;
        parts[i].logits_out = i == 0 ? &logits : &part_logits[(size_t)i];
      }
      const double t0 = now_ms();
      rt.forward_multi(parts, &st);
      return now_ms() - t0;
    }
    const double t0 = now_ms();
    rt.forward(*seq, p, M, &logits, &st);
    return now_ms() - t0;
  };
  // Run once per length without measuring (base settings) so first-use costs (cuBLAS workspace, lazy module loading, grouped GEMM buffer creation, ...) do not land on the first variant
  for (int M : lens) {
    std::vector<float> lg; ForwardStats st{};
    Variant w{"warmup", {}};
    run_one(w, ids.data(), M, nullptr, lg, st);
  }
  std::map<std::pair<int, std::string>, std::vector<double>> times;
  bool all_ok = true;
  for (int r = 0; r < reps; ++r)
    for (int M : lens) {
      const int32_t* p = ids.data() + (size_t)r * max_len;
      const int32_t* prev = warm_prev > 0 ? ids.data() + (size_t)reps * max_len * multi : nullptr;  // Q3: after the --multi windows
      std::vector<float> ref;
      std::vector<std::vector<float>> ref_parts;  // Q3 --multi
      bool have_ref = false;
      // rotate the variant order by one per repetition (so order effects — adaptive state, temperature — do not concentrate on one variant)
      for (size_t q = 0; q < vars.size(); ++q) {
        const Variant& v = vars[(q + (size_t)r) % vars.size()];
        std::vector<float> lg;
        ForwardStats st{};
        const double ms = run_one(v, p, M, prev, lg, st);
        times[{M, v.name}].push_back(ms);
        char cmp[256] = "";
        if (v.name == "base" && !have_ref) { ref = lg; ref_parts = part_logits; have_ref = true; snprintf(cmp, sizeof cmp, "(reference)"); }
        else if (have_ref) {
          Cmp k = compare(ref, lg);
          for (int i = 1; i < multi; ++i) {  // Q3: compare per sequence — the worst value (argmax counts as different if any differs)
            const Cmp q = compare(ref_parts[(size_t)i], part_logits[(size_t)i]);
            k.argmax_eq = k.argmax_eq && q.argmax_eq; k.top5 = std::min(k.top5, q.top5); k.max_abs = std::max(k.max_abs, q.max_abs); k.kl = std::max(k.kl, q.kl);
          }
          snprintf(cmp, sizeof cmp, "vs base: argmax %s · top5 %d/5 · max|Δ| %.4g · KL %.3g", k.argmax_eq ? "same" : "differs", k.top5, k.max_abs, k.kl);
          if (!k.argmax_eq) all_ok = false;
        } else snprintf(cmp, sizeof cmp, "(before reference — no comparison)");
        fprintf(stderr, "[bench] rep %d M=%d %-12s %8.1f ms · routed %d hit %d cpu %d streamed %d dma rows %d · cpu wait %.1f · tail %d · %s\n", r, M, v.name.c_str(), ms,
                st.n_routed, st.n_hit, st.n_cpu, st.n_streamed, st.n_dma_rows, st.ms_cpu_wait, st.tail_rows, cmp);
      }
    }
  rt.set_short_prefill_opts(env_opts);
  rt.set_short_dma_cap(env_cap);
  // Summary: median per length × variant · vs base
  auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); const size_t n = v.size(); return n ? (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0; };
  printf("M\tvariant\tn\tmedian_ms\tmin_ms\tmax_ms\tvs_base\n");
  for (int M : lens) {
    const double b = times.count({M, "base"}) ? median(times[{M, "base"}]) : 0.0;
    std::vector<std::string> seen;
    for (const Variant& v : vars) {
      if (std::find(seen.begin(), seen.end(), v.name) != seen.end()) continue;
      seen.push_back(v.name);
      const auto& t = times[{M, v.name}];
      const double md = median(t);
      char rel[32] = "-";
      if (b > 0) snprintf(rel, sizeof rel, "%+.1f%%", (md / b - 1.0) * 100.0);
      printf("%d\t%s\t%zu\t%.1f\t%.1f\t%.1f\t%s\n", M, v.name.c_str(), t.size(), md, *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()), rel);
    }
  }
  printf("%s\n", all_ok ? "ARGMAX: all variants agree with base" : "ARGMAX: some variant disagrees with base (see [bench] lines — compare with base-vs-base)");
  return 0;
}
