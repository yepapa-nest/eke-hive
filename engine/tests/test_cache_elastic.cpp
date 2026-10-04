// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Q1 HIVE_CACHE_ELASTIC — GPU comparison test (needs the GPU to itself: no running hived · loads the model once).
//   Runs the same scenario twice, switch off and on (one Runtime per process — VRAM), and compares the output files
//   with tools/compare_cache_elastic.py:
//     (1) prefill prompt A (--len-a, chunk max_chunk) -> (2) decode --steps tokens (greedy · promotion on — with the
//     switch on, the elastic slots fill up) -> (3) prefill prompt B (with the switch on, reclaim happens here: the
//     elastic slots' residents are evicted and that VRAM becomes the prefill work buffer) -> (4) decode --steps
//     -> (5) prompt C (short turn --len-c; rows <= S takes the small variant — no reclaim) -> (6) decode --steps
//   After each phase: byte comparison of the VRAM records of all resident slots (or --check-max of them) against the
//   host half records (catches the work buffer overwriting lent slots, or a promotion overwriting the work-buffer
//   region) · slot counts (n_slots · base · elastic) · reclaim statistics (count · evicted residents · dropped
//   promotions · wait ms) · phase time (prefill TTFT).
//   Output (--out): per-phase logits (last prefill row · every decode step) as raw f32 + tokens — the comparator
//   reports off vs on argmax · top-5 · max|Δ| · KL.
//   Expected: phase (1), starting with an empty cache (no --cache-state), has 0 hits both off and on, so it is
//   **bit-identical** (the large variant only differs in addresses). Phases (2)–(6) can differ because the switch-on
//   run has more slots and therefore different hits, by the fp32 accumulation-order difference between the CPU and
//   GPU expert paths — the baseline is the difference between two switch-off runs (A vs A; give the comparator three
//   files to get it as well).
//   Record mismatches must be 0 (the direct evidence of losslessness — any mismatch is a FAIL).
//   Example (in a shell that exports the same HIVE_* as production):
//     scripts/hive-run.sh "./build-dev/test_cache_elastic --ckpt /path/to/DeepSeek-V4.1-Flash --engram /out/engram --ids-file /out/prompt-ids.txt
//        --vram-cache-mb 60000 --cpu-threads 16 --max-chunk 16384 --out /out/elastic-off.bin"             (HIVE_CACHE_ELASTIC unset)
//     HIVE_CACHE_ELASTIC=1 … --out /out/elastic-on.bin   -> python3 tools/compare_cache_elastic.py /out/elastic-off.bin /out/elastic-on.bin [/out/elastic-off2.bin]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hive/engram_hash.h"
#include "hive/expert_store.h"
#include "hive/model.h"
#include "hive/runtime.h"
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
// resident slot record == the two host half records (GPU record format — the same 12 pieces as expert_store.cpp for_each_rec_piece)
int check_records(ExpertStore& s, const Config& c, int max_n, int* checked) {
  const ExpertLayout& L = s.layout();
  const HalfLayout& H = s.half_layout();
  const size_t w13 = H.s1 - H.w1, s13 = H.w3 - H.s1, w2 = H.s2 - H.w2, s2 = H.w2_rows * c.moe_inter / 32;
  static uint8_t* buf = nullptr;
  if (!buf) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&buf), L.total));
  CUDA_CHECK(cudaDeviceSynchronize());
  int bad = 0, n = 0;
  for (size_t k = 0; k < s.slot_table_size() && (max_n <= 0 || n < max_n); ++k) {
    const int slot = s.slot_table()[k];
    if (slot < 0) continue;
    ++n;
    CUDA_CHECK(cudaMemcpy(buf, s.dev_rec(slot), L.total, cudaMemcpyDeviceToHost));
    const int l = (int)(k / s.E()), e = (int)(k % s.E());
    bool ok = slot < s.n_slots();
    for (int h = 0; h < 2 && ok; ++h) {
      const uint8_t* src = s.host_half(l, e, h);
      ok = ok && !memcmp(buf + L.w1 + h * w13, src + H.w1, w13) && !memcmp(buf + L.s1 + h * s13, src + H.s1, s13) &&
           !memcmp(buf + L.w3 + h * w13, src + H.w3, w13) && !memcmp(buf + L.s3 + h * s13, src + H.s3, s13) &&
           !memcmp(buf + L.w2 + h * w2, src + H.w2, w2) && !memcmp(buf + L.s2 + h * s2, src + H.s2, s2);
    }
    if (!ok) { if (bad < 8) fprintf(stderr, "[elastic-test] ✗ slot %d (layer %d expert %d) record differs\n", slot, l, e); ++bad; }
  }
  *checked = n;
  return bad;
}
}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = getenv("HIVE_CKPT") ? getenv("HIVE_CKPT") : "", engram_dir, ids_file, out = "elastic.bin", cache_state;
  int len_a = 20000, len_b = 6000, len_c = 100, steps = 48, cpu_threads = 16, max_chunk = 16384, max_batch = 8, check_max = 0, promote = 8;
  double cache_mb = 60000;
  int64_t max_ctx = 65536;
  for (int i = 1; i < argc; i++) {
    const std::string a(argv[i]);
    auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "%s: missing value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--engram") engram_dir = next();
    else if (a == "--ids-file") ids_file = next();
    else if (a == "--cache-state") cache_state = next();
    else if (a == "--out") out = next();
    else if (a == "--len-a") len_a = atoi(next().c_str());
    else if (a == "--len-b") len_b = atoi(next().c_str());
    else if (a == "--len-c") len_c = atoi(next().c_str());
    else if (a == "--steps") steps = atoi(next().c_str());
    else if (a == "--promote") promote = atoi(next().c_str());
    else if (a == "--check-max") check_max = atoi(next().c_str());
    else if (a == "--cpu-threads") cpu_threads = atoi(next().c_str());
    else if (a == "--vram-cache-mb") cache_mb = atof(next().c_str());
    else if (a == "--max-ctx") max_ctx = atoll(next().c_str());
    else if (a == "--max-chunk") max_chunk = atoi(next().c_str());
    else if (a == "--max-batch") max_batch = atoi(next().c_str());
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (ckpt.empty()) { fprintf(stderr, "--ckpt DIR (or HIVE_CKPT) is required\n"); return 2; }
  std::vector<int32_t> ids;
  if (!ids_file.empty()) { std::ifstream f(ids_file); std::stringstream ss; ss << f.rdbuf(); ids = parse_list(ss.str()); }
  const size_t need = (size_t)len_a + len_b + len_c;
  if (ids.size() < need) {
    fprintf(stderr, "[elastic-test] WARNING: ids %zu < %zu — padding with deterministic pseudo-tokens\n", ids.size(), need);
    for (size_t i = ids.size(); i < need; ++i) ids.push_back(1000 + (int32_t)((i * 7919u) % 100000u));
  }
  const char* sw = getenv("HIVE_CACHE_ELASTIC");
  fprintf(stderr, "[elastic-test] HIVE_CACHE_ELASTIC=%s · A %d · B %d · C %d · steps %d\n", sw ? sw : "(unset)", len_a, len_b, len_c, steps);
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
  opt.promote_per_token = promote;
  Runtime rt(model, store, have_eh ? &eh : nullptr, opt);
  if (!cache_state.empty()) {
    FILE* f = fopen(cache_state.c_str(), "rb");
    std::vector<int32_t> keys;
    if (f) {
      int32_t hdr[4] = {0, 0, 0, 0};
      if (fread(hdr, 4, 4, f) == 4 && hdr[0] == 0x48564353 && hdr[1] == store.n_layers() && hdr[2] == store.E() && hdr[3] > 0 && hdr[3] < 100000) {
        keys.resize(hdr[3]);
        if (fread(keys.data(), 4, keys.size(), f) != keys.size()) keys.clear();
      }
      fclose(f);
    }
    fprintf(stderr, "[elastic-test] warm start %d/%zu\n", keys.empty() ? 0 : rt.warm_from_keys(keys), keys.size());
  }
  FILE* fo = fopen(out.c_str(), "wb");
  if (!fo) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 2; }
  const uint32_t V = (uint32_t)c.vocab;
  fwrite("HVEL", 1, 4, fo); fwrite(&V, 4, 1, fo);
  // record: u8 phase · u8 kind (0 prefill · 1 decode) · u16 step · i32 token (argmax) · V × f32 logits
  auto dump = [&](int phase, int kind, int step, int32_t tok, const std::vector<float>& lg) {
    const uint8_t pk[2] = {(uint8_t)phase, (uint8_t)kind};
    const uint16_t s16 = (uint16_t)step;
    fwrite(pk, 1, 2, fo); fwrite(&s16, 2, 1, fo); fwrite(&tok, 4, 1, fo);
    std::vector<float> z(V, 0.f);
    fwrite(lg.size() == V ? lg.data() : z.data(), 4, V, fo);
  };
  int total_bad = 0;
  auto check = [&](const char* what) {
    int n = 0;
    const double t0 = now_ms();
    const int bad = check_records(store, c, check_max, &n);
    total_bad += bad;
    const auto& es = store.elastic_stats();
    fprintf(stderr, "[elastic-test] %s: records checked %d · bad %d (%.0f ms) · slots %d (base %d · elastic %d · lent %d) · resident %d · reclaims %llu lends %llu evicted %llu dropped %llu wait %.2f ms\n",
            what, n, bad, now_ms() - t0, store.n_slots(), store.base_slots(), store.elastic_slots(), (int)store.elastic_lent(), store.n_resident(),
            (unsigned long long)es.reclaims, (unsigned long long)es.lends, (unsigned long long)es.evicted, (unsigned long long)es.dropped, es.wait_ms);
  };
  auto seq = rt.new_seq();
  size_t off = 0;
  auto prefill = [&](int phase, int len) {
    std::vector<float> lg;
    ForwardStats st{};
    int32_t tok = -1;
    const double t0 = now_ms();
    for (int i = 0; i < len;) {
      const int m = std::min(len - i, max_chunk * std::max(1, rt.prefill_slots()));
      lg.clear();
      rt.set_prefill_pending(i + m < len);  // same as the hived M7 contract (true during a multi-chunk prefill)
      tok = rt.forward(*seq, ids.data() + off + i, m, &lg, &st);
      i += m;
    }
    rt.set_prefill_pending(false);
    off += (size_t)len;
    const double ms = now_ms() - t0;
    if (len >= rt.opt().prefill_threshold) rt.warm_cache();
    fprintf(stderr, "[elastic-test] phase %d prefill %d tokens: %.1f ms · next %d\n", phase, len, ms, tok);
    dump(phase, 0, 0, tok, lg);
    return tok;
  };
  auto decode = [&](int phase, int32_t tok) {
    ForwardStats st{};
    const double t0 = now_ms();
    for (int s = 0; s < steps; ++s) {
      std::vector<float> lg;
      const int32_t nxt = rt.forward(*seq, &tok, 1, &lg, &st);
      dump(phase, 1, s, nxt, lg);
      tok = nxt;
    }
    fprintf(stderr, "[elastic-test] phase %d decode %d steps: %.1f ms (hit %d · cpu %d · streamed %d)\n", phase, steps, now_ms() - t0, st.n_hit, st.n_cpu, st.n_streamed);
    return tok;
  };
  check("start");
  int32_t t = prefill(1, len_a); check("after prefill A");
  t = decode(2, t); check("after decode A");
  t = prefill(3, len_b); check("after prefill B (reclaim)");
  t = decode(4, t); check("after decode B");
  t = prefill(5, len_c); check("after short turn C");
  decode(6, t); check("after decode C");
  fclose(fo);
  printf("elastic-test %s: record mismatches %d\n", total_bad == 0 ? "PASS" : "FAIL", total_bad);
  return total_bad == 0 ? 0 : 1;
}
