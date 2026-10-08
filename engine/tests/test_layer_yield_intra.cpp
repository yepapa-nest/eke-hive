// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// T11b HIVE_LAYER_YIELD_INTRA — GPU comparison test (needs the GPU to itself: no running hived · loads the model once). Not run by the CPU suite.
//   One scenario per process, run with the switch off and on (and off again = noise floor), compared by tools/compare_layer_yield_intra.py:
//     (1) --dec decoders prefill --dec-len tokens each and take one greedy step (2) a long prompt L (--len rows) is prefilled while a yield body
//     (ly::Hooks — period --period ms) gives the decoders --dec-steps greedy steps per yield (the same body at layer boundaries and, switch on, at
//     intra-layer points) (3) L decodes --steps greedy tokens.
//   Output: <out>.bin = L's logits (last prefill row, then every decode step) as raw f32 · <out>.img = L's SeqImage right after the prefill (every
//   buffer's bytes in a fixed order) · <out>.json = tokens of L and of every decoder, yield counters, prefill time.
//   Expected: L's logits bytes and image bytes identical off vs on (the paused prefill runs the same kernels on the same inputs in the same order;
//   only the decoders' steps are interleaved) whenever off vs off is identical — run with fixed DMA shares and no promotion so the cache and the
//   CPU/DMA split cannot depend on timing (HIVE_DMA_FRAC, HIVE_DMA_FRAC_PREFILL, --promote 0; HIVE_PREFILL_SPLIT/HIVE_DECODE_SPLIT unset).
//   Decoder streams: compared on the common prefix (their step count differs — more yields with the switch on).
//   HIVE_PREFETCH=8 exercises the staging hold (pre-copies outstanding at intra points → decoder misses on the CPU).
//   Example (tools/validate_gpu.sh "layer_yield_intra" runs exactly this): see the comment there.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hive/clock.h"
#include "hive/engram_hash.h"
#include "hive/expert_store.h"
#include "hive/model.h"
#include "hive/runtime.h"

using namespace hive;

namespace {
std::vector<int32_t> parse_list(const std::string& s) {
  std::vector<int32_t> v;
  std::string cur;
  for (char ch : s + ",") {
    if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') { if (!cur.empty()) { v.push_back(atoi(cur.c_str())); cur.clear(); } }
    else cur += ch;
  }
  return v;
}
void put_buf(FILE* f, const HostImageBuffer& b) {
  const uint64_t n = b.size();
  fwrite(&n, 8, 1, f);
  for (const auto& s : b.segments) fwrite(s.data.get(), 1, s.n, f);
}
void put_img(const std::string& path, const SeqImage& img) {
  if (img.fence) img.fence->wait();
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  fwrite("HVLI", 1, 4, f);
  fwrite(&img.pos, 8, 1, f);
  const uint64_t nt = img.tokens.size(), nh = img.engram_history.size();
  fwrite(&nt, 8, 1, f); fwrite(img.tokens.data(), 4, nt, f);
  fwrite(&nh, 8, 1, f); fwrite(img.engram_history.data(), 8, nh, f);
  for (const auto& b : img.ring) put_buf(f, b);
  for (const auto& b : img.mtp_ring) put_buf(f, b);
  for (const auto* m : {&img.comp, &img.idx, &img.st_kv, &img.st_score})
    for (const auto& [l, b] : *m) { const int32_t k = l; fwrite(&k, 4, 1, f); put_buf(f, b); }
  put_buf(f, img.mtp_hidden);
  const uint8_t v = img.mtp_hidden_valid ? 1 : 0;
  fwrite(&v, 1, 1, f); fwrite(&img.mtp_pos, 8, 1, f);
  fclose(f);
}
int32_t argmax(const std::vector<float>& v) { return v.empty() ? -1 : (int32_t)(std::max_element(v.begin(), v.end()) - v.begin()); }
}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = getenv("HIVE_CKPT") ? getenv("HIVE_CKPT") : "", engram_dir, ids_file, out = "ly-intra";
  int len = 65536, dec = 2, dec_len = 2000, dec_steps = 2, steps = 32, cpu_threads = 16, max_chunk = 16384, max_batch = 8, promote = 0, tile = 3;
  double cache_mb = 60000, period = 50;
  int64_t max_ctx = 131072;
  for (int i = 1; i < argc; i++) {
    const std::string a(argv[i]);
    auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "%s: missing value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--engram") engram_dir = next();
    else if (a == "--ids-file") ids_file = next();
    else if (a == "--out") out = next();
    else if (a == "--len") len = atoi(next().c_str());
    else if (a == "--dec") dec = atoi(next().c_str());
    else if (a == "--dec-len") dec_len = atoi(next().c_str());
    else if (a == "--dec-steps") dec_steps = atoi(next().c_str());
    else if (a == "--steps") steps = atoi(next().c_str());
    else if (a == "--period") period = atof(next().c_str());
    else if (a == "--promote") promote = atoi(next().c_str());
    else if (a == "--prefill-tile") tile = atoi(next().c_str());
    else if (a == "--cpu-threads") cpu_threads = atoi(next().c_str());
    else if (a == "--vram-cache-mb") cache_mb = atof(next().c_str());
    else if (a == "--max-ctx") max_ctx = atoll(next().c_str());
    else if (a == "--max-chunk") max_chunk = atoi(next().c_str());
    else if (a == "--max-batch") max_batch = atoi(next().c_str());
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (ckpt.empty()) { fprintf(stderr, "--ckpt DIR (or HIVE_CKPT) is required\n"); return 2; }
  dec = std::max(1, std::min(dec, max_batch));
  std::vector<int32_t> ids;
  if (!ids_file.empty()) { std::ifstream f(ids_file); std::stringstream ss; ss << f.rdbuf(); ids = parse_list(ss.str()); }
  const size_t need = (size_t)len + (size_t)dec * dec_len;
  for (size_t i = ids.size(); i < need; ++i) ids.push_back(1000 + (int32_t)((i * 7919u) % 100000u));  // deterministic pseudo-tokens
  const char* sw = getenv("HIVE_LAYER_YIELD_INTRA");
  fprintf(stderr, "[ly-intra-test] HIVE_LAYER_YIELD_INTRA=%s · L %d · decoders %d × %d · %d steps per yield · period %.0f ms\n", sw ? sw : "(unset)", len, dec,
          dec_len, dec_steps, period);
  Model model(ckpt, -1, max_ctx);
  const Config& c = model.cfg();
  const int nl = model.n_loaded_layers();
  if (nl == c.n_layers && model.has_head()) model.load_mtp();
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
  opt.max_chunk = max_chunk; opt.max_ctx = max_ctx; opt.max_batch = max_batch; opt.vision = false;
  opt.promote_per_token = promote; opt.prefill_tile = tile; opt.decoder_tail = 128; opt.decoder_replay = true;  // service shape (--prefill-tile 3 --decoder-replay)
  Runtime rt(model, store, have_eh ? &eh : nullptr, opt);
  // decoders
  std::vector<std::unique_ptr<Seq>> dseq;
  std::vector<Seq*> dptr;
  std::vector<int32_t> dtok;
  std::vector<std::vector<int32_t>> dstream((size_t)dec);
  size_t off = (size_t)len;
  for (int d = 0; d < dec; ++d) {
    dseq.push_back(rt.new_seq());
    std::vector<float> lg;
    ForwardStats st{};
    dtok.push_back(rt.forward(*dseq.back(), ids.data() + off, dec_len, &lg, &st));
    off += (size_t)dec_len;
    dptr.push_back(dseq.back().get());
    dstream[(size_t)d].push_back(dtok.back());
  }
  long steps_in_yields = 0;
  auto dstep = [&] {
    std::vector<int32_t> nxt;
    std::vector<std::vector<float>> lg;
    ForwardStats st{};
    rt.forward_batch(dptr, dtok.data(), nxt, &lg, &st);
    for (int d = 0; d < dec; ++d) { dtok[(size_t)d] = nxt[(size_t)d]; dstream[(size_t)d].push_back(nxt[(size_t)d]); }
  };
  dstep();
  ly::Hooks h;
  h.period_ms = period;
  h.max_depth = 1;
  h.want = [] { return 1; };  // decoders are always running (decode only — no admission, so R stays the decode floor at boundaries too)
  h.run = [&](int, double) { for (int s = 0; s < dec_steps; ++s) { dstep(); ++steps_in_yields; } };
  rt.set_layer_yield(std::move(h));
  // long prompt
  auto L = rt.new_seq();
  std::vector<float> lg;
  ForwardStats st{};
  int32_t tok = -1;
  const double t0 = mono_ms();
  for (int i = 0; i < len;) {
    const int m = std::min(len - i, max_chunk * std::max(1, rt.prefill_slots()));
    lg.clear();
    rt.set_prefill_pending(i + m < len);
    tok = rt.forward(*L, ids.data() + i, m, &lg, &st);
    i += m;
  }
  rt.set_prefill_pending(false);
  const double prefill_ms = mono_ms() - t0;
  rt.set_layer_yield(ly::Hooks{});  // L's decode below runs alone
  FILE* fo = fopen((out + ".bin").c_str(), "wb");
  if (!fo) { fprintf(stderr, "cannot open %s.bin\n", out.c_str()); return 2; }
  const uint32_t V = (uint32_t)c.vocab;
  fwrite("HVLY", 1, 4, fo); fwrite(&V, 4, 1, fo);
  auto dump = [&](int32_t t, const std::vector<float>& v) {
    std::vector<float> z(V, 0.f);
    fwrite(&t, 4, 1, fo);
    fwrite(v.size() == V ? v.data() : z.data(), 4, V, fo);
  };
  dump(tok, lg);
  {
    SeqImage img;
    rt.save_image(*L, img);
    put_img(out + ".img", img);
  }
  std::vector<int32_t> ltoks{tok};
  for (int s = 0; s < steps; ++s) {
    std::vector<float> lg2;
    ForwardStats st2{};
    const int32_t nxt = rt.forward(*L, &tok, 1, &lg2, &st2);
    dump(nxt, lg2);
    ltoks.push_back(nxt);
    tok = nxt;
  }
  fclose(fo);
  const ly::Stats& ys = rt.layer_yield_stats();
  std::ofstream js(out + ".json");
  js << "{\"label\":\"" << (sw && *sw && strcmp(sw, "0") ? "intra" : "off") << "\",\"len\":" << len << ",\"prefill_ms\":" << prefill_ms << ",\"yields\":" << ys.yields
     << ",\"intra_yields\":" << ys.intra_yields << ",\"yield_inner_ms\":" << ys.inner_ms << ",\"max_gap_ms\":" << ys.max_gap_ms
     << ",\"steps_in_yields\":" << steps_in_yields << ",\"L\":[";
  for (size_t i = 0; i < ltoks.size(); ++i) js << (i ? "," : "") << ltoks[i];
  js << "],\"decoders\":[";
  for (int d = 0; d < dec; ++d) {
    js << (d ? "," : "") << "[";
    for (size_t i = 0; i < dstream[(size_t)d].size(); ++i) js << (i ? "," : "") << dstream[(size_t)d][i];
    js << "]";
  }
  js << "]}\n";
  fprintf(stderr, "[ly-intra-test] done: prefill %.0f ms · yields %ld (intra %ld) · inner %.0f ms · max gap %.0f ms · decoder steps in yields %ld · L argmax %d\n",
          prefill_ms, ys.yields, ys.intra_yields, ys.inner_ms, ys.max_gap_ms, steps_in_yields, argmax(lg));
  return 0;
}
