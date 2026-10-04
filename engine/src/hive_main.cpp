// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive CLI — load -> prefill -> (optional) decode. Compared against the oracle with per-layer dumps.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
#include <random>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "hive/clock.h"
#include "hive/engram_hash.h"
#include "hive/expert_store.h"
#include "hive/model.h"
#include "hive/runtime.h"
#include "nlohmann/json.hpp"

using namespace hive;

// image_inputs.json + patches_NNN.bf16 saved by the oracle -> ImageInput (patches uploaded to the device)
static std::vector<ImageInput> load_images(const std::string& dir, std::vector<DevBuf>& keep) {
  std::vector<ImageInput> out;
  std::ifstream f(dir + "/image_inputs.json");
  if (!f) return out;
  nlohmann::json j = nlohmann::json::parse(f);
  for (auto& im : j["images"]) {
    ImageInput x;
    x.start = im["start"].get<int>();
    x.n_vit_h = im["n_vit_h"].get<int>();
    x.n_vit_w = im["n_vit_w"].get<int>();
    for (auto& t : im["types"]) x.types.push_back((int8_t)t.get<int>());
    std::ifstream pf(dir + "/" + im["file"].get<std::string>(), std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
    DevBuf d(raw.size());
    CUDA_CHECK(cudaMemcpy(d.p, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    x.patches_dev = d.as<bf16>();
    keep.push_back(std::move(d));
    out.push_back(std::move(x));
  }
  return out;
}

static std::vector<int32_t> parse_ids(const std::string& s) {
  std::vector<int32_t> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) v.push_back(atoi(s.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return v;
}

int main(int argc, char** argv) {
  std::string logits_out;
  int interleave_tiled = 0;
  std::string ckpt = getenv("HIVE_CKPT") ? getenv("HIVE_CKPT") : "", ids_s, engram_dir, dump_dir, inject_dir, cont_s, images_from, batch_golden;
  int max_layer = -1, decode = 0, cpu_threads = 16, max_chunk = 64;
  double cache_mb = 512;
  int64_t max_ctx = 4096;
  bool no_cpu = false;
  int vision_max_patches = 0, bench_cpu = 0, promote = -1, gpu_share = -1, decoder_tail = -1, spec = 0;
  bool decoder_replay = false;
  int promote_misses = -1;
  float promote_miss_ratio = 0.f;
  int prefill_tile = 1;
  bool no_mtp = false;
  bool verify_test = false;
  int verify2_test = 0;  // --verify2-test [iterations] — HIVE_MTP_VERIFY2/HIVE_MTP_BATCH comparison and timing (below)
  for (int i = 1; i < argc; i++) {
    const std::string a(argv[i]);
    auto next = [&]() -> std::string { return argv[++i]; };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--ids") ids_s = next();
    else if (a == "--ids-file") {  // long prompts go through a file (a single argument is limited to 128 KB — 40K tokens ~= 244 KB is rejected by execve)
      std::ifstream f(next()); std::stringstream ss; ss << f.rdbuf(); ids_s = ss.str();
      for (char& ch : ids_s) if (ch == '\n' || ch == ' ' || ch == '\t' || ch == '\r') ch = ',';
    }
    else if (a == "--logits-out") logits_out = next();  // final prefill logits (fp32 vocab) to a file — for path comparison (A/B)
    else if (a == "--interleave-tiled") interleave_tiled = atoi(next().c_str());  // after 4 decode steps, tile-prefill another sequence with N tokens and continue decoding (graph address safety)
    else if (a == "--continue-ids") cont_s = next();
    else if (a == "--images-from") images_from = next();
    else if (a == "--batch-golden") batch_golden = next();
    else if (a == "--max-layer") max_layer = atoi(next().c_str());
    else if (a == "--engram") engram_dir = next();
    else if (a == "--dump") dump_dir = next();
    else if (a == "--inject") inject_dir = next();  // per-layer golden injection (output of export_golden_raw.py)
    else if (a == "--decode") decode = atoi(next().c_str());
    else if (a == "--cpu-threads") cpu_threads = atoi(next().c_str());
    else if (a == "--vram-cache-mb") cache_mb = atof(next().c_str());
    else if (a == "--max-ctx") max_ctx = atoll(next().c_str());
    else if (a == "--max-chunk") max_chunk = atoi(next().c_str());
    else if (a == "--prefill-tile") prefill_tile = std::max(1, atoi(next().c_str()));
    else if (a == "--no-cpu") no_cpu = true;
    else if (a == "--vision-max-patches") vision_max_patches = atoi(next().c_str());
    else if (a == "--bench-cpu") bench_cpu = atoi(next().c_str());
    else if (a == "--promote") promote = atoi(next().c_str());
    else if (a == "--promote-misses") {  // N | auto | auto:cap
      const std::string v = next();
      if (v == "auto") { promote_misses = 32; promote_miss_ratio = 0.25f; }
      else if (v.rfind("auto:", 0) == 0 && atoi(v.c_str() + 5) > 0) { promote_misses = atoi(v.c_str() + 5); promote_miss_ratio = 0.25f; }
      else if (!v.empty() && v.find_first_not_of("0123456789") == std::string::npos) { promote_misses = atoi(v.c_str()); promote_miss_ratio = 0.f; }
      else { fprintf(stderr, "--promote-misses: must be N | auto | auto:N (N>0): %s\n", v.c_str()); return 1; }
    }  // miss promotion (N per step -> LRU slots)
    else if (a == "--gpu-share") gpu_share = atoi(next().c_str());
    else if (a == "--decoder-tail") decoder_tail = atoi(next().c_str());
    else if (a == "--decoder-replay") decoder_replay = true;  // shrink the decoder tail to the window length (128) and look only at the replay span (deployment recipe, approximate)
    else if (a == "--no-mtp") no_mtp = true;
    else if (a == "--verify-test") verify_test = true;
    else if (a == "--verify2-test") verify2_test = std::max(4, atoi(next().c_str()));  // iterations (first 3 = eager run and capture — excluded from timing)
    else if (a == "--spec") spec = atoi(next().c_str());  // N DSpark speculative steps after decode (greedy verify) — with --dump, dumps the draft outputs
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
  }
  if (ckpt.empty()) { fprintf(stderr, "--ckpt DIR (or HIVE_CKPT) is required\n"); return 1; }
  std::vector<int32_t> ids = parse_ids(ids_s);
  if (bench_cpu > 0) ids = {0};
  std::vector<DevBuf> keep;
  std::vector<ImageInput> images;
  if (!images_from.empty()) {
    std::ifstream f(images_from + "/image_inputs.json");
    nlohmann::json j = nlohmann::json::parse(f);
    ids = j["ids"].get<std::vector<int32_t>>();
  }
  if (ids.empty()) { fprintf(stderr, "--ids is required\n"); return 1; }
  fprintf(stderr, "[hive] ckpt=%s layers≤%d ids=%zu\n", ckpt.c_str(), max_layer, ids.size());
  Model model(ckpt, max_layer, max_ctx);
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
    int el = c.engram_layer_ids[i];
    if (el >= nl) continue;
    if (engram_dir.empty()) { fprintf(stderr, "[hive] layer %d needs engram — pass --engram DIR\n", el); return 1; }
    if (!have_eh) { eh.load(engram_dir + "/engram_hash.json", engram_dir + "/token_map.bin"); have_eh = true; }
    store.load_engram(model.ckpt(), el, (int)i);
  }
  store.engram_ssd_finish(model.ckpt());  // HIVE_ENGRAM_SSD (off = no-op) — for comparing GPU-window RAM vs SSD dumps
  RuntimeOptions opt;
  opt.max_chunk = max_chunk;
  opt.max_ctx = max_ctx;
  opt.dump_dir = dump_dir;
  opt.inject_dir = inject_dir;
  opt.cpu_for_misses = !no_cpu;
  opt.vision_max_patches = vision_max_patches;
  if (promote >= 0) opt.promote_per_token = promote;
  if (promote_misses >= 0) opt.promote_misses = promote_misses;
  if (promote_miss_ratio > 0.f) opt.promote_miss_ratio = promote_miss_ratio;
  if (gpu_share >= 0) opt.decode_gpu_share = gpu_share;
  if (decoder_tail >= 0) opt.decoder_tail = decoder_tail;
  opt.prefill_tile = prefill_tile;
  if (decoder_replay) { opt.decoder_replay = true; if (decoder_tail < 0) opt.decoder_tail = model.cfg().window; }
  opt.mtp = !no_mtp;
  if (bench_cpu > 0) {
    // One decode token = 6 missed experts (random) per layer sent to the CPU pool. Repeat bench_cpu times to measure per-layer and per-token cost.
    std::vector<float> a_f(c.dim), a_s(c.dim / 32), outs((size_t)6 * c.dim), scratch((size_t)6 * ExpertStore::job_scratch_floats(c.moe_inter));
    for (auto& v : a_f) v = 0.01f * (float)(rand() % 200 - 100);
    for (auto& v : a_s) v = 1.f;
    std::mt19937 rng(1);
    double total = 0;
    for (int it = 0; it < bench_cpu; ++it) {
      std::vector<ExpertStore::Job> jobs;
      for (int j = 0; j < 6; ++j) {
        int e = (int)(rng() % c.n_routed);
        ExpertStore::Job jb{}; jb.layer = 0; jb.e = e; jb.R = 1; jb.a_f[0] = a_f.data(); jb.a_s[0] = a_s.data(); jb.route_w[0] = 0.3f;
        jb.out[0] = outs.data() + (size_t)j * c.dim; jb.scratch = scratch.data() + (size_t)j * ExpertStore::job_scratch_floats(c.moe_inter);
        jobs.push_back(jb);
      }
      auto t0 = hive::SteadyClock::now();
      store.run_jobs(jobs);
      total += hive::ms_since(t0);
    }
    // Decode critical path: latency with 1 miss (one node) or 2 misses (same node) on a layer
    for (int nj : {1, 2}) {
      double t_sum = 0;
      for (int it = 0; it < bench_cpu; ++it) {
        std::vector<ExpertStore::Job> jobs;
        for (int j = 0; j < nj; ++j) {
          ExpertStore::Job jb{}; jb.layer = 0; jb.e = (int)(2 * (rng() % (c.n_routed / 2))); jb.R = 1; jb.a_f[0] = a_f.data(); jb.a_s[0] = a_s.data();
          jb.route_w[0] = 0.3f; jb.out[0] = outs.data() + (size_t)j * c.dim; jb.scratch = scratch.data() + (size_t)j * ExpertStore::job_scratch_floats(c.moe_inter);
          jobs.push_back(jb);
        }
        auto t0 = hive::SteadyClock::now();
        store.run_jobs(jobs);
        t_sum += hive::ms_since(t0);
      }
      fprintf(stderr, "[bench-cpu] %d job(s) same node: %.3f ms per run\n", nj, t_sum / bench_cpu);
    }
    fprintf(stderr, "[bench-cpu] 6 misses/layer × %d: %.2f ms/layer → %.0f ms/token (40 layers) · %d threads/node\n", bench_cpu, total / bench_cpu, total / bench_cpu * 40, cpu_threads);
    return 0;
  }
  Runtime rt(model, store, have_eh ? &eh : nullptr, opt);
  if (!images_from.empty()) images = load_images(images_from, keep);
  auto seq = rt.new_seq();
  ForwardStats st{};
  std::vector<float> logits;
  // Prefill (in chunks)
  int32_t next = -1;
  const size_t chunk_cap = images.empty() ? (size_t)max_chunk * (size_t)rt.prefill_slots() : (size_t)max_chunk;  // tiling (images use a single chunk)
  for (size_t i = 0; i < ids.size(); i += chunk_cap) {
    int M = (int)std::min<size_t>(chunk_cap, ids.size() - i);
    if (!images.empty()) { HIVE_CHECK(i == 0 && M == (int)ids.size(), "image tests need a single chunk"); }
    next = rt.forward(*seq, ids.data() + i, M, images.empty() ? nullptr : &images, &logits, &st);
  }
  if (!logits_out.empty()) {
    std::ofstream lf(logits_out, std::ios::binary);
    lf.write((const char*)logits.data(), (std::streamsize)(logits.size() * 4));
    fprintf(stderr, "[hive] logits → %s (%zu)\n", logits_out.c_str(), logits.size());
  }
  fprintf(stderr, "[hive] prefill %zu tokens: %.1f ms · routed %d hit %d cpu %d streamed %d · cpu wait %.1f ms · tail %d · next=%d\n", ids.size(),
          st.ms_total, st.n_routed, st.n_hit, st.n_cpu, st.n_streamed, st.ms_cpu_wait, st.tail_rows, next);
  if (ids.size() >= 64) {
    auto tw = hive::SteadyClock::now();
    int nw = rt.warm_cache();
    fprintf(stderr, "[hive] warm cache: %d experts in %.0f ms\n", nw, hive::ms_since(tw));
  }
  if (verify_test) {
    // Explicit test mode only: no serving guard, no automatic GPU execution.
    if (!rt.mtp_enabled() || next < 0) { fprintf(stderr,"verify-test requires full model + MTP\n"); return 1; }
    SeqImage base; rt.save_image(*seq,base);
    const int N=std::min(8,c.dspark_block+1);
    std::vector<int32_t> probe(N,next);
    for (int i=1;i<N;++i) probe[i]=(1000+i*7919)%c.vocab;
    auto close_logits = [&](const float* a,const float* b,size_t n,const char* label) {
      double dot=0,aa=0,bb=0,md=0,scale=0;
      for(size_t i=0;i!=n;++i) {
        const double x=a[i],y=b[i];
        if(!(std::isfinite(x) && std::isfinite(y))) return false;
        dot+=x*y; aa+=x*x; bb+=y*y;
        md=std::max(md,std::abs(x-y));scale=std::max(scale,std::abs(y));
      }
      const double rel=md/std::max(scale,1e-30),cos=dot/std::max(std::sqrt(aa*bb),1e-30);
      const bool ok=md==0 || (rel<=.01 && cos>=.99999);
      fprintf(stderr,"[verify-test] %s rel %.6g cos %.9g %s\n",label,rel,cos,ok?"OK":"FAIL"); return ok;
    };
    std::vector<float> baseline;
    rt.forward_verify(*seq,probe.data(),N,baseline,nullptr);rt.rollback(*seq,N);
    bool ok=true;
    for(int cut=1;cut<N;++cut) {
      rt.load_image(*seq,base);
      auto changed=probe;for(int j=cut;j<N;++j) changed[j]=(changed[j]+97)%c.vocab;
      std::vector<float> rows;rt.forward_verify(*seq,changed.data(),N,rows,nullptr);rt.rollback(*seq,cut);
      ok &= close_logits(rows.data(),baseline.data(),(size_t)cut*c.vocab,"future independence");
    }
    for(int keep_n=1;keep_n<=N;++keep_n) {
      rt.load_image(*seq,base);
      std::vector<float> rows,after,reference;
      rt.forward_verify(*seq,probe.data(),N,rows,nullptr);rt.rollback(*seq,keep_n);
      int32_t follow=13;rt.forward(*seq,&follow,1,nullptr,&after,nullptr);
      rt.load_image(*seq,base);
      for(int j=0;j<keep_n;++j) rt.forward(*seq,&probe[j],1,nullptr,&reference,nullptr);
      ok &= close_logits(rows.data()+(size_t)(keep_n-1)*c.vocab,reference.data(),c.vocab,"verify vs sequential");
      rt.forward(*seq,&follow,1,nullptr,&reference,nullptr);
      ok &= close_logits(after.data(),reference.data(),c.vocab,"rollback continuation");
    }
    return ok?0:1;
  }
  if (verify2_test > 0) {
    // HIVE_MTP_VERIFY2 / HIVE_MTP_BATCH GPU comparison (test mode only). HIVE_MTP_VERIFY2=1 (or HIVE_MTP_BATCH=1) must be set at startup so the verify-decode path buffers are allocated.
    //   Within one process, rt.set_mtp_verify2_for_test alternates the old and new verify paths. Every comparison also carries a **routing comparison** (rt.set_route_capture):
    //   the first layer where the per-row/per-layer expert set differs (first flip). It separates logit differences caused by cascading amplification after a routing flip (= fp-order differences of the same computation)
    //   from logits that differ with identical routing (= a different computation — suspected leak, "LEAK?"). Verdicts:
    //     OK       = within the --verify-test tolerance (rel <= 0.01, cos >= 0.99999)
    //     BASELINE = outside tolerance, but (1) the two sides' routing diverged at some layer and (2) rel <= the old path's value for the same comparison or the noise floor of repeating the same input
    //     FAIL     = anything else (in particular identical routing throughout but outside tolerance = LEAK?), a flipped greedy row argmax, or greedy accept counts old != new
    //   Noise floor: same state and input twice — expert CPU/DMA placement (cache state, DMA share adaptation), atomic accumulation order of CPU results and DMA group order (rows descending — depends on other rows)
    //   may differ between runs (runtime.cpp moe_decode_experts, moe_decode_accum). With --no-cpu --promote 0 there is no CPU share or promotion, so noise shrinks (DMA order remains).
    //   (1) rows N = 2..B+1, greedy continuation (all accepted); (2) random drafts: future independence, continuation after rollback for every keep (old and new); (3) timing; (4) HIVE_MTP_BATCH partial batches.
    if (!rt.mtp_enabled() || next < 0 || rt.mtp_batch_rows() < 2) {
      fprintf(stderr, "verify2-test requires full model + MTP + HIVE_MTP_VERIFY2=1 (or HIVE_MTP_BATCH=1) at start (rows %d)\n", rt.mtp_batch_rows());
      return 1;
    }
    const int V = c.vocab, B = c.dspark_block, NL = c.n_layers;
    const int Nmax = std::min(B + 1, rt.mtp_batch_rows());
    fprintf(stderr, "[verify2-test] cpu_for_misses %d · promote/token %d · promote_misses %d (to isolate noise: --no-cpu --promote 0)\n", rt.opt().cpu_for_misses ? 1 : 0,
            rt.opt().promote_per_token, rt.opt().promote_misses);
    SeqImage base; rt.save_image(*seq, base);
    using Routes = std::vector<std::vector<std::vector<int32_t>>>;  // [row][layer] sorted expert set
    auto take_routes = [&](Routes& R, int rows_expected) {  // main-layer entries only (draft layers l >= NL excluded); entry order = call order
      R.assign(rows_expected, {});
      const int row_base = 0;  // collects a single forward only (sequential tokens are separate per call — run_seq)
      for (const auto& e : rt.route_log()) {
        const int l = e[0], M = e[1], k = e[2];
        if (l >= NL) continue;
        for (int m = 0; m < M && row_base + m < rows_expected; ++m) {
          std::vector<int32_t> s(e.begin() + 3 + (size_t)m * k, e.begin() + 3 + (size_t)(m + 1) * k);
          std::sort(s.begin(), s.end());
          R[row_base + m].push_back(s);
        }
      }
      rt.set_route_capture(false);
    };
    auto first_flip = [&](const std::vector<std::vector<int32_t>>& a, const std::vector<std::vector<int32_t>>& b, int* n_diff) {
      int first = -1, n = 0;
      for (size_t l = 0; l < std::min(a.size(), b.size()); ++l) if (a[l] != b[l]) { ++n; if (first < 0) first = (int)l; }
      if (n_diff) *n_diff = n;
      return first;  // -1 = identical on all layers
    };
    auto argmax = [&](const float* r) { return (int32_t)(std::max_element(r, r + V) - r); };
    auto margin = [&](const float* r) { float a = -INFINITY, b = -INFINITY; for (int i = 0; i < V; ++i) { if (r[i] > a) { b = a; a = r[i]; } else if (r[i] > b) b = r[i]; } return a - b; };
    auto relcos = [&](const float* a, const float* b, size_t n, double* rel_o, double* cos_o) {
      double dot = 0, aa = 0, bb = 0, md = 0, sc = 0;
      for (size_t i = 0; i != n; ++i) {
        const double x = a[i], y = b[i];
        if (!(std::isfinite(x) && std::isfinite(y))) { *rel_o = INFINITY; *cos_o = 0; return false; }
        dot += x * y; aa += x * x; bb += y * y;
        md = std::max(md, std::abs(x - y)); sc = std::max(sc, std::abs(y));
      }
      *rel_o = md / std::max(sc, 1e-30); *cos_o = dot / std::max(std::sqrt(aa * bb), 1e-30);
      return md == 0 || (*rel_o <= .01 && *cos_o >= .99999);  // same tolerance as --verify-test
    };
    // One verify (v2 = new path) + routing, rollback(keep)
    auto run_verify = [&](bool v2, const std::vector<int32_t>& ids_v, int N, int keep, std::vector<float>& rows, Routes* R) {
      rt.set_mtp_verify2_for_test(v2);
      rt.load_image(*seq, base);
      rt.set_route_capture(R != nullptr);
      rt.forward_verify(*seq, ids_v.data(), N, rows, nullptr);
      if (R) take_routes(*R, N);
      rt.rollback(*seq, keep);
    };
    // n sequential (M=1 forward) tokens + per-row logits and routing
    auto run_seq = [&](SeqImage& img, Seq& s, const std::vector<int32_t>& toks, int n, std::vector<std::vector<float>>& rows, Routes* R) {
      rt.load_image(s, img);
      rows.assign(n, {});
      if (R) R->assign(n, {});
      for (int j = 0; j < n; ++j) {
        rt.set_route_capture(R != nullptr);
        rt.forward(s, &toks[j], 1, nullptr, &rows[j], nullptr);
        if (R) { Routes one; take_routes(one, 1); (*R)[j] = one[0]; }
      }
    };
    bool ok = true;
    int flips = 0;
    // Verdict helpers: tolerance, routing, reference value (old path on the same comparison, noise floor)
    auto judge = [&](const char* what, const float* a, const float* b, size_t n, int flip_layer, int n_flip, double ref_rel) {
      double rl, cs;
      const bool close = relcos(a, b, n, &rl, &cs);
      const char* st = close ? "OK" : (flip_layer >= 0 && rl <= ref_rel) ? "BASELINE" : "FAIL";
      if (!close && flip_layer < 0) st = "FAIL(LEAK? routing identical)";
      ok &= close || (flip_layer >= 0 && rl <= ref_rel);
      fprintf(stderr, "[verify2-test] %s: rel %.3g cos %.9f · routing first flip layer %d (%d layers differ) · ref rel %.3g → %s\n", what, rl, cs, flip_layer, n_flip, ref_rel, st);
      return rl;
    };
    // Sequential greedy continuation (draft = the model's own next token)
    std::vector<int32_t> probe(1, next);
    {
      rt.load_image(*seq, base);
      for (int j = 0; j + 1 < Nmax; ++j) { std::vector<float> lg; probe.push_back(rt.forward(*seq, &probe[j], 1, nullptr, &lg, nullptr)); }
    }
    std::vector<std::vector<float>> seq_rows, seq_rows2;
    Routes seq_R, seq_R2;
    run_seq(base, *seq, probe, Nmax, seq_rows, &seq_R);
    run_seq(base, *seq, probe, Nmax, seq_rows2, &seq_R2);
    // Noise floor (same input twice): sequential, old, new
    double floor_seq = 0, floor_old = 0, floor_new = 0;
    for (int j = 0; j < Nmax; ++j) {
      int nd; const int f = first_flip(seq_R[j], seq_R2[j], &nd);
      double rl, cs; relcos(seq_rows2[j].data(), seq_rows[j].data(), V, &rl, &cs);
      floor_seq = std::max(floor_seq, rl);
      fprintf(stderr, "[verify2-test] noise floor sequential row %d: rel %.3g · routing first flip %d (%d)\n", j, rl, f, nd);
    }
    {
      std::vector<float> a, b; Routes Ra, Rb;
      for (int v2 = 0; v2 < 2; ++v2) {
        run_verify(v2 != 0, probe, Nmax, Nmax, a, &Ra);
        run_verify(v2 != 0, probe, Nmax, Nmax, b, &Rb);
        for (int i = 0; i < Nmax; ++i) {
          int nd; const int f = first_flip(Ra[i], Rb[i], &nd);
          double rl, cs; relcos(b.data() + (size_t)i * V, a.data() + (size_t)i * V, V, &rl, &cs);
          (v2 ? floor_new : floor_old) = std::max(v2 ? floor_new : floor_old, rl);
          fprintf(stderr, "[verify2-test] noise floor %s verify N %d row %d: rel %.3g · routing first flip %d (%d)\n", v2 ? "new" : "old", Nmax, i, rl, f, nd);
        }
      }
    }
    const double floor_all = std::max({floor_seq, floor_old, floor_new});
    // (1) greedy continuation
    for (int N = 2; N <= Nmax; ++N) {
      std::vector<float> ro, rn; Routes Ro, Rn;
      run_verify(false, probe, N, N, ro, &Ro);
      run_verify(true, probe, N, N, rn, &Rn);
      int acc_o = 0, acc_n = 0;
      for (int i = 0; i + 1 < N && argmax(ro.data() + (size_t)i * V) == probe[i + 1]; ++i) ++acc_o;
      for (int i = 0; i + 1 < N && argmax(rn.data() + (size_t)i * V) == probe[i + 1]; ++i) ++acc_n;
      for (int i = 0; i < N; ++i) {
        const float *o = ro.data() + (size_t)i * V, *n = rn.data() + (size_t)i * V, *q = seq_rows[i].data();
        int d_os, d_ns, d_no;
        const int f_os = first_flip(Ro[i], seq_R[i], &d_os), f_ns = first_flip(Rn[i], seq_R[i], &d_ns), f_no = first_flip(Rn[i], Ro[i], &d_no);
        double r_os, c_os; relcos(o, q, V, &r_os, &c_os);
        char w[96];
        snprintf(w, sizeof w, "N %d row %d new~old", N, i); judge(w, n, o, V, f_no, d_no, std::max(r_os, floor_all));
        snprintf(w, sizeof w, "N %d row %d new~seq", N, i); judge(w, n, q, V, f_ns, d_ns, std::max(r_os, floor_all));
        const int32_t ao = argmax(o), an = argmax(n), as = argmax(q);
        const bool flip = ao != an;
        flips += flip ? 1 : 0;
        ok &= !flip;
        fprintf(stderr, "[verify2-test] N %d row %d: argmax old %d new %d seq %d%s · old~seq rel %.3g (routing flip %d) · seq margin %.4f\n", N, i, ao, an, as,
                flip ? " FLIP → FAIL" : "", r_os, f_os, margin(q));
      }
      fprintf(stderr, "[verify2-test] N %d greedy accepted: old %d new %d / %d %s\n", N, acc_o, acc_n, N - 1, acc_o == acc_n ? "OK" : "FAIL");
      ok &= acc_o == acc_n;
    }
    // (2) random drafts: future independence, continuation after rollback — old runs first so its rel serves as the reference for the same comparison
    {
      const int N = Nmax;
      std::vector<int32_t> rnd(N, next);
      for (int i = 1; i < N; ++i) rnd[i] = (1000 + i * 7919) % V;
      //   variant 0 = old, 1 = new, 2 = new + HIVE_MTP_VERIFY_ROWIND (row m's expert sum is independent of other rows: future independence must be **bit-identical** (rel 0), else FAIL)
      const char* vname[3] = {"old", "new", "new+rowind"};
      std::vector<float> base_rows[3]; Routes base_R[3];
      auto set_variant = [&](int v) { rt.set_mtp_verify_rowind_for_test(v == 2); };
      for (int v = 0; v < 3; ++v) { set_variant(v); run_verify(v != 0, rnd, N, N, base_rows[v], &base_R[v]); }
      for (int cut = 1; cut < N; ++cut) {
        auto ch = rnd; for (int j = cut; j < N; ++j) ch[j] = (ch[j] + 97) % V;
        double ref = floor_all;
        for (int v = 0; v < 3; ++v) {
          set_variant(v);
          std::vector<float> rows; Routes R;
          run_verify(v != 0, ch, N, cut, rows, &R);
          int f = -1, nd = 0;
          for (int i = 0; i < cut; ++i) { int d; const int fi = first_flip(R[i], base_R[v][i], &d); nd += d; if (fi >= 0 && (f < 0 || fi < f)) f = fi; }
          char w[96]; snprintf(w, sizeof w, "future independence %s cut %d", vname[v], cut);
          if (v == 2) {  // strict: bit-identical
            bool same = true;
            for (size_t i = 0; i < (size_t)cut * V; ++i) if (rows[i] != base_rows[2][i]) { same = false; break; }
            ok &= same;
            fprintf(stderr, "[verify2-test] %s: %s (routing first flip %d)\n", w, same ? "bit-identical OK" : "NOT bit-identical → FAIL", f);
            continue;
          }
          const double rl = judge(w, rows.data(), base_rows[v].data(), (size_t)cut * V, f, nd, std::max(ref, floor_all));
          if (!v) ref = std::max(ref, rl);  // reference for new = old's value on the same comparison
        }
      }
      set_variant(0);
      for (int keep = 1; keep <= N; ++keep) {
        std::vector<std::vector<float>> refr; Routes refR;
        std::vector<int32_t> toks(rnd.begin(), rnd.begin() + keep); toks.push_back(13);
        run_seq(base, *seq, toks, keep + 1, refr, &refR);  // sequential: keep accepted tokens + one continuation token
        double ref_v = floor_all, ref_c = floor_all;
        for (int v2 = 0; v2 < 2; ++v2) {
          std::vector<float> rows, after; Routes R, Ra;
          run_verify(v2 != 0, rnd, N, keep, rows, &R);
          int32_t follow = 13;
          rt.set_route_capture(true);
          rt.forward(*seq, &follow, 1, nullptr, &after, nullptr);
          take_routes(Ra, 1);
          int nd; int f = first_flip(R[keep - 1], refR[keep - 1], &nd);
          char w[96]; snprintf(w, sizeof w, "%s keep %d: verify row vs sequential", v2 ? "new" : "old", keep);
          double rl = judge(w, rows.data() + (size_t)(keep - 1) * V, refr[keep - 1].data(), V, f, nd, ref_v);
          if (!v2) ref_v = std::max(ref_v, rl);
          // Routing verdict for the continuation = continuation token + accepted rows (the rows that built the state — if they were routed differently from sequential, ring and compressed cache differ).
          //   Checking only the continuation token would wrongly report "identical routing -> LEAK?" when accepted rows diverged and the state differs.
          f = first_flip(Ra[0], refR[keep], &nd);
          for (int j = 0; j < keep; ++j) { int d; const int fj = first_flip(R[j], refR[j], &d); nd += d; if (fj >= 0 && (f < 0 || fj < f)) f = fj; }
          snprintf(w, sizeof w, "%s keep %d: rollback continuation", v2 ? "new" : "old", keep);
          rl = judge(w, after.data(), refr[keep].data(), V, f, nd, ref_c);
          if (!v2) ref_c = std::max(ref_c, rl);
        }
      }
    }
    // (3) timing
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    auto ms_since = [](hive::SteadyClock::time_point t) { return hive::ms_since(t); };
    {
      std::vector<double> t1;
      for (int r = 0; r < verify2_test; ++r) {
        rt.load_image(*seq, base);
        const auto t0 = hive::SteadyClock::now();
        rt.forward(*seq, &probe[0], 1, nullptr, nullptr, nullptr);
        if (r >= 3) t1.push_back(ms_since(t0));
      }
      fprintf(stderr, "[verify2-test] time: sequential step %.2f ms (median of %zu)\n", med(t1), t1.size());
      for (int N = 2; N <= Nmax; ++N) {
        std::vector<double> to, tn;
        for (int v2 = 0; v2 < 2; ++v2)
          for (int r = 0; r < verify2_test; ++r) {
            std::vector<float> rows;
            rt.set_mtp_verify2_for_test(v2 != 0);
            rt.load_image(*seq, base);
            const auto t0 = hive::SteadyClock::now();
            rt.forward_verify(*seq, probe.data(), N, rows, nullptr);
            rt.rollback(*seq, N);
            if (r >= 3) (v2 ? tn : to).push_back(ms_since(t0));
          }
        fprintf(stderr, "[verify2-test] time: N %d rows — old verify %.2f ms · new verify %.2f ms (x%.2f) · new / sequential step x%.2f\n", N, med(to), med(tn),
                med(tn) > 0 ? med(to) / med(tn) : 0.0, med(t1) > 0 ? med(tn) / med(t1) : 0.0);
      }
    }
    // (4) multi-sequence verify: part rows vs a standalone new verify of that part (single-row parts are sequential), continuation after rollback_batch == sequential, timing
    if (rt.mtp_batch_enabled()) {
      rt.set_mtp_verify2_for_test(true);
      std::vector<int32_t> ids2(ids.rbegin(), ids.rend());
      auto seq2 = rt.new_seq();
      int32_t next2 = -1;
      for (size_t i = 0; i < ids2.size(); i += chunk_cap) next2 = rt.forward(*seq2, ids2.data() + i, (int)std::min(chunk_cap, ids2.size() - i), nullptr, nullptr, nullptr);
      SeqImage base2; rt.save_image(*seq2, base2);
      std::vector<int32_t> probe2(1, next2);
      for (int j = 0; j + 1 < Nmax; ++j) { std::vector<float> lg; probe2.push_back(rt.forward(*seq2, &probe2[j], 1, nullptr, &lg, nullptr)); }
      std::vector<std::vector<float>> s2rows; Routes s2R;
      run_seq(base2, *seq2, probe2, Nmax, s2rows, &s2R);
      const std::vector<std::pair<int, int>> lays = {{3, 3}, {1, 4}, {4, 1}, {2, 2}};
      for (auto [m1, m2] : lays) {
        if (m1 + m2 > rt.mtp_batch_rows() || m1 > Nmax || m2 > Nmax) continue;
        std::vector<float> single[2]; Routes singleR[2];
        const int mm[2] = {m1, m2};
        Seq* sq[2] = {seq.get(), seq2.get()};
        SeqImage* bs[2] = {&base, &base2};
        const std::vector<int32_t>* pr[2] = {&probe, &probe2};
        for (int p = 0; p < 2; ++p) {
          if (mm[p] < 2) continue;
          rt.load_image(*sq[p], *bs[p]);
          rt.set_route_capture(true);
          rt.forward_verify(*sq[p], pr[p]->data(), mm[p], single[p], nullptr);
          take_routes(singleR[p], mm[p]);
          rt.rollback(*sq[p], mm[p]);
        }
        rt.load_image(*seq, base); rt.load_image(*seq2, base2);
        std::vector<Runtime::VerifyPart> parts{{seq.get(), probe.data(), m1}, {seq2.get(), probe2.data(), m2}};
        std::vector<float> rows; Routes R;
        rt.set_route_capture(true);
        rt.forward_verify_batch(parts, rows, nullptr);
        take_routes(R, m1 + m2);
        const int k1 = std::max(1, m1 - 1), k2 = std::max(1, m2 - 1);  // accept one row less (exercises the rollback path)
        rt.rollback_batch({k1, k2});
        for (int i = 0; i < m1 + m2; ++i) {
          const int p = i < m1 ? 0 : 1, j = p ? i - m1 : i;
          const bool one = mm[p] < 2;
          const float* ref = one ? (p ? s2rows[j].data() : seq_rows[j].data()) : single[p].data() + (size_t)j * V;
          const auto& refR = one ? (p ? s2R[j] : seq_R[j]) : singleR[p][j];
          int nd; const int f = first_flip(R[i], refR, &nd);
          char w[96]; snprintf(w, sizeof w, "batch [%d,%d] part %d row %d vs %s", m1, m2, p, j, one ? "sequential" : "single verify");
          judge(w, rows.data() + (size_t)i * V, ref, V, f, nd, floor_all);
          const bool flip = argmax(rows.data() + (size_t)i * V) != argmax(ref);
          if (flip) { ++flips; ok = false; fprintf(stderr, "[verify2-test]   ↑ argmax FLIP → FAIL\n"); }
        }
        // continuation == sequential (each sequence: k accepted + one token)
        int32_t follow = 13;
        std::vector<float> a[2];
        Routes aR[2];
        for (int p = 0; p < 2; ++p) { rt.set_route_capture(true); rt.forward(*sq[p], &follow, 1, nullptr, &a[p], nullptr); take_routes(aR[p], 1); }
        const int kk[2] = {k1, k2};
        for (int p = 0; p < 2; ++p) {
          std::vector<int32_t> toks(pr[p]->begin(), pr[p]->begin() + kk[p]); toks.push_back(follow);
          std::vector<std::vector<float>> rr; Routes rR;
          run_seq(*bs[p], *sq[p], toks, kk[p] + 1, rr, &rR);
          int nd; int f = first_flip(aR[p][0], rR[kk[p]], &nd);
          for (int j = 0; j < kk[p]; ++j) {  // routing of the accepted rows (row p*j in the batch) too — same verdict as the keep continuation above
            int d; const int fj = first_flip(R[(p ? m1 : 0) + j], rR[j], &d); nd += d; if (fj >= 0 && (f < 0 || fj < f)) f = fj;
          }
          char w[96]; snprintf(w, sizeof w, "batch [%d,%d] keep %d part %d continuation", m1, m2, kk[p], p);
          judge(w, a[p].data(), rr[kk[p]].data(), V, f, nd, floor_all);
        }
        std::vector<double> tb, ts, tn;
        for (int r = 0; r < verify2_test; ++r) {
          rt.load_image(*seq, base); rt.load_image(*seq2, base2);
          auto t0 = hive::SteadyClock::now();
          rt.forward_verify_batch(parts, rows, nullptr); rt.rollback_batch({m1, m2});
          if (r >= 3) tb.push_back(ms_since(t0));
          rt.load_image(*seq, base); rt.load_image(*seq2, base2);
          t0 = hive::SteadyClock::now();
          for (int p = 0; p < 2; ++p) {
            if (mm[p] >= 2) { rt.forward_verify(*sq[p], pr[p]->data(), mm[p], rows, nullptr); rt.rollback(*sq[p], mm[p]); }
            else rt.forward(*sq[p], pr[p]->data(), 1, nullptr, nullptr, nullptr);
          }
          if (r >= 3) ts.push_back(ms_since(t0));
          rt.load_image(*seq, base); rt.load_image(*seq2, base2);
          std::vector<Seq*> two{seq.get(), seq2.get()};
          int32_t toks[2] = {probe[0], probe2[0]};
          std::vector<int32_t> nx;
          t0 = hive::SteadyClock::now();
          rt.forward_batch(two, toks, nx, nullptr, nullptr);
          if (r >= 3) tn.push_back(ms_since(t0));
        }
        fprintf(stderr, "[verify2-test] time: batch [%d,%d] = %d rows %.2f ms · two single verifies %.2f ms · plain batch step (2 rows) %.2f ms\n", m1, m2, m1 + m2, med(tb),
                med(ts), med(tn));
      }
    }
    fprintf(stderr, "[verify2-test] noise floors: sequential %.3g · old %.3g · new %.3g\n", floor_seq, floor_old, floor_new);
    fprintf(stderr, "[verify2-test] %s (argmax flips %d)\n", ok ? "ALL OK" : "FAIL", flips);
    return ok ? 0 : 1;
  }
  if (!batch_golden.empty()) {
    // Batched decode verification: prefill the second golden's prompt as a separate sequence, then feed continue-ids to both sequences at once (batched).
    std::ifstream f(batch_golden + "/prefill.json");
    nlohmann::json j = nlohmann::json::parse(f);
    std::vector<int32_t> ids2 = j["ids"].get<std::vector<int32_t>>();
    auto seq2 = rt.new_seq();
    ForwardStats s2{};
    for (size_t i = 0; i < ids2.size(); i += (size_t)max_chunk * rt.prefill_slots()) {
      int M = (int)std::min<size_t>((size_t)max_chunk * rt.prefill_slots(), ids2.size() - i);
      rt.forward(*seq2, ids2.data() + i, M, nullptr, &s2);
    }
    std::vector<Seq*> seqs{seq.get(), seq2.get()};
    for (int32_t t : parse_ids(cont_s)) {
      ForwardStats ds{};
      int32_t toks[2] = {t, t};
      std::vector<int32_t> nxt;
      rt.forward_batch(seqs, toks, nxt, nullptr, &ds);
      fprintf(stderr, "[hive] batch continue id=%d: %.1f ms (cpu wait %.1f, host %.1f, hit %d cpu %d streamed %d)\n", t, ds.ms_total, ds.ms_cpu_wait, ds.ms_host,
              ds.n_hit, ds.n_cpu, ds.n_streamed);
    }
    return 0;
  }
  for (int32_t t : parse_ids(cont_s)) {
    ForwardStats ds{};
    int32_t n2 = rt.forward(*seq, &t, 1, nullptr, &ds);
    fprintf(stderr, "[hive] continue id=%d: %.1f ms (cpu wait %.1f, host %.1f, hit %d cpu %d streamed %d) → %d\n", t, ds.ms_total, ds.ms_cpu_wait, ds.ms_host,
            ds.n_hit, ds.n_cpu, ds.n_streamed, n2);
  }
  std::vector<int32_t> out;
  std::unique_ptr<Seq> seq_other;
  for (int d = 0; d < decode && next >= 0; ++d) {
    if (interleave_tiled > 0 && d == 4) {  // after the graphs are captured (steps 1-3), tile-prefill another sequence — Work must be restored afterwards for the following decode to be correct
      seq_other = rt.new_seq();
      std::vector<int32_t> ids2(interleave_tiled);
      for (int i = 0; i < interleave_tiled; ++i) ids2[i] = 1000 + (int32_t)((i * 7919u) % 100000u);
      ForwardStats s2{};
      for (size_t i = 0; i < ids2.size(); i += chunk_cap) rt.forward(*seq_other, ids2.data() + i, (int)std::min<size_t>(chunk_cap, ids2.size() - i), nullptr, &s2);
      fprintf(stderr, "[hive] interleaved tiled prefill %d tokens: %.0f ms\n", interleave_tiled, s2.ms_total);
    }
    ForwardStats ds{};
    out.push_back(next);
    int32_t t = next;
    next = rt.forward(*seq, &t, 1, nullptr, &ds);
    fprintf(stderr, "[hive] decode %d: %.1f ms (cpu wait %.1f, hit %d cpu %d) → %d\n", d, ds.ms_total, ds.ms_cpu_wait, ds.n_hit, ds.n_cpu,
            next);
  }
  // DSpark speculative steps (greedy): draft -> verify (per-row argmax comparison) -> rollback. Compared with the oracle --spec golden through the --dump outputs (mtp_*).
  for (int st_i = 0; st_i < spec && next >= 0 && rt.mtp_enabled(); ++st_i) {
    out.push_back(next);  // each verify consumes this token, including reject-at-0
    std::vector<int32_t> drafts; std::vector<float> conf;
    ForwardStats dst{};
    auto t0 = hive::SteadyClock::now();
    rt.mtp_draft(*seq, next, drafts, conf, &dst);
    double ms_d = hive::ms_since(t0);
    std::vector<int32_t> ids(1, next);
    ids.insert(ids.end(), drafts.begin(), drafts.end());
    std::vector<float> rows;
    ForwardStats ds{};
    t0 = hive::SteadyClock::now();
    rt.forward_verify(*seq, ids.data(), (int)ids.size(), rows, &ds);
    double ms_v = hive::ms_since(t0);
    int n_keep = 1, nxt = -1;
    for (int i = 0; i < (int)drafts.size(); ++i) {
      const float* row = rows.data() + (size_t)i * c.vocab;
      int32_t am = (int32_t)(std::max_element(row, row + c.vocab) - row);
      if (am != drafts[i]) { nxt = am; break; }
      ++n_keep;
    }
    if (nxt < 0) { const float* row = rows.data() + (size_t)drafts.size() * c.vocab; nxt = (int32_t)(std::max_element(row, row + c.vocab) - row); }
    rt.rollback(*seq, n_keep);
    fprintf(stderr, "[spec %d] pos %lld tok %d → drafts [", st_i, (long long)seq->pos, next);
    for (size_t i = 0; i < drafts.size(); ++i) fprintf(stderr, "%s%d(%.2f)", i ? " " : "", drafts[i], conf[i]);
    fprintf(stderr, "] accepted %d/%zu · draft %.1f ms (hit %d cpu %d) · verify %.1f ms (hit %d cpu %d) → next %d\n", n_keep - 1, drafts.size(), ms_d,
            dst.n_hit, dst.n_cpu, ms_v, ds.n_hit, ds.n_cpu, nxt);
    for (int i = 1; i < n_keep; ++i) out.push_back(ids[i]);
    next = nxt;
  }
  if (next >= 0) out.push_back(next);
  printf("generated:");
  for (int32_t t : out) printf(" %d", t);
  printf("\n");
  fprintf(stderr, "[hive] generated:");
  for (int32_t t : out) fprintf(stderr, " %d", t);
  fprintf(stderr, "\n");
  return 0;
}
