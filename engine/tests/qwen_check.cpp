// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next forward dump for validation against the CPU oracle (oracle/qwen4_oracle.py) — compare with tools/compare_qwen.py.
//   Prefills the prompt in one chunk (the oracle's single chunk), writes the hidden stream after every layer (h_LNN.npy, float32
//   [T, 10240]) and the logits (logits.npy: the last prompt row, then one row per greedy decode step — the oracle's layout), and the
//   decoded ids (decode.json). With QWEN_CHECK_DECODE_PATH=1 the prompt's last token is decoded instead of prefilled (the decode path's
//   logits must match the prefill path's for the same position).
// usage: qwen_check <ckpt dir> <out dir> <ids: comma list or file> [decode N = 8] [cache GiB = 70] [cpu threads = 24]
//   (GPU; loads the whole model — run only when the GPU is free for it.)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hive/clock.h"
#include "hive/qwen/qwen_engine.h"
#include <cuda_profiler_api.h>

using namespace hive;
using namespace hive::qwen;

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
  if (argc < 4) { fprintf(stderr, "usage: qwen_check <ckpt> <out dir> <ids> [decode N] [cache GiB] [threads]\n"); return 2; }
  const std::string out = argv[2];
  const std::vector<int32_t> ids = parse_ids(argv[3]);
  const int n_dec = argc > 4 ? atoi(argv[4]) : 8;
  const double cache_gib = argc > 5 ? atof(argv[5]) : 70.0;
  const int threads = argc > 6 ? atoi(argv[6]) : 24;
  const bool decode_path = getenv("QWEN_CHECK_DECODE_PATH") && atoi(getenv("QWEN_CHECK_DECODE_PATH")) > 0;
  if (ids.empty()) { fprintf(stderr, "no prompt ids\n"); return 2; }
  // QWEN_CHECK_SPEC=N: after the dumps, N steps of hived's decode loop with MTP (draft → verify → rollback) on a filled cache, timed per
  //   step (QWEN_CHECK_PROFILE=1: cudaProfilerStart/Stop around the timed steps, for nsys --capture-range=cudaProfilerApi)
  const int spec_steps = getenv("QWEN_CHECK_SPEC") ? atoi(getenv("QWEN_CHECK_SPEC")) : 0;
  // QWEN_CHECK_PREFILL=N[,chunk]: prefill speed only — N pseudo-random token ids (fixed seed), 3 fresh sequences, engine chunk = chunk
  //   (default 16384, hived's); QWEN_CHECK_PROFILE wraps the third in cudaProfilerStart/Stop
  int pre_n = 0, pre_chunk = 16384;
  if (const char* e = getenv("QWEN_CHECK_PREFILL"); e && *e) { pre_n = atoi(e); if (const char* c2 = strchr(e, ',')) pre_chunk = atoi(c2 + 1); }
  QwenModel model(argv[1], spec_steps > 0, getenv("HIVE_QWEN_GPU_LAYERS") ? atoi(getenv("HIVE_QWEN_GPU_LAYERS")) : -1);   // < n_layers: hybrid
  const QwenConfig& c = model.cfg();
  glm::ExpertSpec sp;
  sp.hidden = c.hidden; sp.inter = c.moe_inter; sp.n_routed = c.n_experts; sp.n_act = c.n_act; sp.limit = 0.f;
  for (int l = 0; l < model.gpu_layers(); ++l) sp.moe_layers.push_back(l);
  sp.ckpt = &model.ckpt();
  glm::GlmExperts experts(sp, 0, threads, 8);
  experts.load_all(threads);
  experts.alloc_cache((size_t)(cache_gib * 1073741824.0));
  // QWEN_CHECK_FILL=1: fill the cache before the dumps too, so the prompt takes the serving path (device-routed grouped prefill experts)
  const bool fill = getenv("QWEN_CHECK_FILL") && atoi(getenv("QWEN_CHECK_FILL")) > 0;
  if ((spec_steps > 0 || pre_n > 0 || fill) && experts.n_slots() >= experts.n_keys()) {   // the runtime's fill_cache: every expert resident → device routing
    std::vector<int32_t> miss;
    for (int k = 0; k < experts.n_keys(); ++k) if (experts.slot_of(k) < 0) miss.push_back(k);
    const int n = experts.warm_from_keys(miss);
    printf("qwen_check: expert cache filled with %d records (%d / %d resident)\n", n, experts.n_resident(), experts.n_keys());
  }
  const int T = (int)ids.size();
  QwenEngine eng(model, experts, pre_n > 0 ? std::min(pre_n, pre_chunk) : std::max(64, std::min(T, 16384)));   // long prompts: hived's chunk
  if (pre_n > 0) {
    std::vector<int32_t> pid(pre_n);
    uint64_t z = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < pre_n; ++i) { z = z * 6364136223846793005ull + 1442695040888963407ull; pid[i] = 1000 + (int32_t)((z >> 33) % 150000); }
    float* lp = nullptr;
    CUDA_CHECK(cudaMallocHost(&lp, (size_t)c.vocab * 4));
    const bool prof = getenv("QWEN_CHECK_PROFILE") && atoi(getenv("QWEN_CHECK_PROFILE")) == 1;
    for (int rep = 0; rep < 3; ++rep) {
      QwenSeq sp;
      eng.init_seq(sp, pre_n + 64);
      if (prof && rep == 2) CUDA_CHECK(cudaProfilerStart());
      const double t0 = hive::mono_ms();
      eng.prefill(sp, pid.data(), pre_n, lp);
      CUDA_CHECK(cudaDeviceSynchronize());
      const double ms = hive::mono_ms() - t0;
      if (prof && rep == 2) CUDA_CHECK(cudaProfilerStop());
      printf("qwen_check: prefill %d tok (chunk %d) rep %d · %.0f ms · %.0f tok/s\n", pre_n, std::min(pre_n, pre_chunk), rep, ms, pre_n / ms * 1e3);
      fflush(stdout);
    }
    if (FILE* f = fopen((out + "/pre_logits.bin").c_str(), "wb")) { fwrite(lp, 4, c.vocab, f); fclose(f); }   // the last row's logits (A/B of paths)
    cudaFreeHost(lp);
    return 0;
  }
  QwenSeq s;
  eng.init_seq(s, T + n_dec + 64);
  const int HC = c.hcw(), V = c.vocab;
  std::vector<bf16> hb;
  std::vector<float> hf;
  const int T_pre = decode_path ? T - 1 : T;
  eng.layer_hook = [&](int l, const bf16* hd, int rows) {
    hb.resize((size_t)rows * HC); hf.resize(hb.size());
    CUDA_CHECK(cudaMemcpy(hb.data(), hd, hb.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = bf2f(hb[i]);
    char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
    write_npy_f32(out + name, hf.data(), rows, HC);
  };
  // hybrid placement (HIVE_QWEN_GPU_LAYERS): the CPU-tail layers report through the CPU engine's host hooks
  auto host_dump = [&](const std::string& dir) {
    return [&, dir](int l, const float* h, int rows) {
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
      write_npy_f32(dir + name, h, rows, HC);
    };
  };
  if (eng.tail()) eng.tail()->layer_hook = host_dump(out);
  // per-layer injection (QWEN_CHECK_INJECT=<oracle dir>): layer l ≥ 1 takes the reference's h_L{l−1}; outputs → <out>/inject
  if (const char* ref = getenv("QWEN_CHECK_INJECT"); ref && *ref) {
    const std::string idir = out + "/inject";
    if (system(("mkdir -p '" + idir + "'").c_str()) != 0) return 2;
    QwenSeq si;
    eng.init_seq(si, T + 64);
    std::vector<bf16> hbuf;
    eng.layer_inject = [&](int l, bf16* hd, int rows) {
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l - 1);
      const std::vector<float> v = read_npy_f32(std::string(ref) + name);
      if (v.size() != (size_t)rows * HC) { fprintf(stderr, "inject: %s has %zu values, need %d rows\n", name, v.size(), rows); exit(2); }
      hbuf.resize(v.size());
      for (size_t i = 0; i < v.size(); ++i) hbuf[i] = __float2bfloat16(v[i]);   // the reference values are bf16 already: exact
      CUDA_CHECK(cudaMemcpy(hd, hbuf.data(), hbuf.size() * 2, cudaMemcpyHostToDevice));
    };
    auto hook = eng.layer_hook;
    eng.layer_hook = [&](int l, const bf16* hd, int rows) {
      hb.resize((size_t)rows * HC); hf.resize(hb.size());
      CUDA_CHECK(cudaMemcpy(hb.data(), hd, hb.size() * 2, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < hb.size(); ++i) hf[i] = bf2f(hb[i]);
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
      write_npy_f32(idir + name, hf.data(), rows, HC);
      // the layer's routed expert ids (QWEN_CHECK_ROUTE: routing flips vs the oracle's routing.json), as float [rows, k]
      if (getenv("QWEN_CHECK_ROUTE")) {
        const int K = c.n_act;
        std::vector<int32_t> ri((size_t)rows * K);
        CUDA_CHECK(cudaMemcpy(ri.data(), eng.route_ids_dev(), ri.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<float> rf(ri.begin(), ri.end());
        snprintf(name, sizeof name, "/route_L%02d.npy", l);
        write_npy_f32(idir + name, rf.data(), rows, K);
      }
    };
    if (eng.tail()) {
      eng.tail()->layer_hook = host_dump(idir);
      eng.tail()->layer_inject = [&](int l, float* h, int rows) {
        char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l - 1);
        const std::vector<float> v = read_npy_f32(std::string(ref) + name);
        if (v.size() != (size_t)rows * HC) { fprintf(stderr, "inject: %s has %zu values, need %d rows\n", name, v.size(), rows); exit(2); }
        std::memcpy(h, v.data(), v.size() * 4);
      };
    }
    std::vector<float> lg((size_t)V);
    eng.prefill(si, ids.data(), T, lg.data());
    write_npy_f32(idir + "/logits.npy", lg.data(), 1, V);
    eng.layer_inject = nullptr;
    eng.layer_hook = hook;
    if (eng.tail()) { eng.tail()->layer_inject = nullptr; eng.tail()->layer_hook = host_dump(out); }
    eng.reset_seq(si);
    printf("qwen_check: per-layer injection outputs in %s\n", idir.c_str());
  }
  // QWEN_CHECK_DECODE_ALL=1: the prompt goes through the DECODE path one token at a time (the first through prefill) — the per-layer dumps
  //   and the injection then test the decode GEMVs (e.g. HIVE_QWEN_DENSE8); rows are assembled per layer
  if (getenv("QWEN_CHECK_DECODE_ALL") && atoi(getenv("QWEN_CHECK_DECODE_ALL")) > 0) {
    auto run_decode_all = [&](QwenSeq& sq, const std::string& dir, const char* ref) {
      std::vector<std::vector<float>> rows_of_layer(c.n_layers);
      int cur = 0;
      std::vector<bf16> hbuf;
      eng.layer_hook = [&](int l, const bf16* hd, int rows) {
        hb.resize((size_t)rows * HC);
        CUDA_CHECK(cudaMemcpy(hb.data(), hd, hb.size() * 2, cudaMemcpyDeviceToHost));
        auto& v = rows_of_layer[l];
        for (size_t i = 0; i < hb.size(); ++i) v.push_back(bf2f(hb[i]));
      };
      if (ref) {
        eng.layer_inject = [&](int l, bf16* hd, int rows) {
          char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l - 1);
          const std::vector<float> v = read_npy_f32(std::string(ref) + name);
          hbuf.resize((size_t)rows * HC);
          for (int r = 0; r < rows; ++r)
            for (int i = 0; i < HC; ++i) hbuf[(size_t)r * HC + i] = __float2bfloat16(v[(size_t)(cur + r) * HC + i]);
          CUDA_CHECK(cudaMemcpy(hd, hbuf.data(), hbuf.size() * 2, cudaMemcpyHostToDevice));
        };
      }
      std::vector<float> lg((size_t)V);
      cur = 0;
      eng.prefill(sq, ids.data(), 1, lg.data());
      for (int t = 1; t < T; ++t) {
        cur = t;
        QwenSeq* sp1[1] = {&sq};
        eng.decode(sp1, ids.data() + t, 1, lg.data());
      }
      for (int l = 0; l < c.n_layers; ++l) {
        char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
        write_npy_f32(dir + name, rows_of_layer[l].data(), T, HC);
      }
      write_npy_f32(dir + "/logits.npy", lg.data(), 1, V);
      eng.layer_inject = nullptr; eng.layer_hook = nullptr;
    };
    if (const char* ref = getenv("QWEN_CHECK_INJECT"); ref && *ref) {
      if (system(("mkdir -p '" + out + "/inject_dec'").c_str()) != 0) return 2;
      QwenSeq si; eng.init_seq(si, T + 64);
      run_decode_all(si, out + "/inject_dec", ref);
    }
    if (system(("mkdir -p '" + out + "/dec'").c_str()) != 0) return 2;
    QwenSeq sa; eng.init_seq(sa, T + 64);
    run_decode_all(sa, out + "/dec", nullptr);
    printf("qwen_check: decode-path dumps in %s/dec%s\n", out.c_str(), getenv("QWEN_CHECK_INJECT") ? " and inject_dec" : "");
    eng.layer_hook = [&](int l, const bf16* hd, int rows) {   // back to the main dump hook
      hb.resize((size_t)rows * HC); hf.resize(hb.size());
      CUDA_CHECK(cudaMemcpy(hb.data(), hd, hb.size() * 2, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < hb.size(); ++i) hf[i] = bf2f(hb[i]);
      char name[64]; snprintf(name, sizeof name, "/h_L%02d.npy", l);
      write_npy_f32(out + name, hf.data(), rows, HC);
    };
  }
  std::vector<float> logits((size_t)(n_dec + 1) * V);
  eng.prefill(s, ids.data(), T_pre, logits.data());
  eng.layer_hook = nullptr;
  if (eng.tail()) eng.tail()->layer_hook = nullptr;
  int32_t tok = 0;
  if (decode_path) {
    QwenSeq* sp1[1] = {&s};
    eng.decode(sp1, ids.data() + T - 1, 1, logits.data());
  }
  std::vector<int32_t> dec;
  for (int i = 0; i < n_dec; ++i) {
    const float* lr = logits.data() + (size_t)i * V;
    tok = (int32_t)(std::max_element(lr, lr + V) - lr);
    dec.push_back(tok);
    QwenSeq* sp1[1] = {&s};
    eng.decode(sp1, &tok, 1, logits.data() + (size_t)(i + 1) * V);
  }
  write_npy_f32(out + "/logits.npy", logits.data(), n_dec + 1, V);
  FILE* f = fopen((out + "/decode.json").c_str(), "w");
  fprintf(f, "{\"prompt\": %d, \"decode\": [", T);
  for (size_t i = 0; i < dec.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", dec[i]);
  fprintf(f, "]}\n");
  fclose(f);
  if (spec_steps > 0) {
    // plain decode (one token per step) on a fresh sequence, then the MTP loop on another
    auto argmax = [&](const float* r) { return (int32_t)(std::max_element(r, r + V) - r); };
    const bool prof = getenv("QWEN_CHECK_PROFILE") && atoi(getenv("QWEN_CHECK_PROFILE")) == 1;
    float* lgp = nullptr;   // pinned, as hived's logits buffer (the decode / verify CUDA graphs copy the logits into it)
    CUDA_CHECK(cudaMallocHost(&lgp, (size_t)8 * V * 4));
    struct Lg { float* p; float* data() { return p; } } lg{lgp};
    QwenSeq sd;
    eng.init_seq(sd, T + 2 * spec_steps * 8 + 64);
    eng.prefill(sd, ids.data(), T, lg.data());
    int32_t tk = argmax(lg.data());
    std::vector<double> ms;
    std::vector<int32_t> plain;
    for (int i = 0; i < spec_steps; ++i) {
      QwenSeq* sp1[1] = {&sd};
      const double t0 = hive::mono_ms();
      eng.decode(sp1, &tk, 1, lg.data());
      ms.push_back(hive::mono_ms() - t0);
      plain.push_back(tk);
      tk = argmax(lg.data());
    }
    std::vector<double> srt(ms.begin() + std::min<size_t>(4, ms.size() - 1), ms.end());
    std::sort(srt.begin(), srt.end());
    printf("qwen_check: plain decode %d steps · median %.2f ms/step (%.1f tok/s) · device-routed steps %ld\n", spec_steps, srt[srt.size() / 2],
           1000.0 / srt[srt.size() / 2], eng.stats().device_route_steps);
    // QWEN_CHECK_BATCH=n1,n2,…: n copies of the prompt decoded together (one forward of n rows per step, the batched GEMV / MoE / GDN
    //   paths up to 64 rows); every copy's greedy tokens must equal the single-sequence decode above
    if (const char* bl = getenv("QWEN_CHECK_BATCH"); bl && *bl) {
      float* lgb = nullptr;
      CUDA_CHECK(cudaMallocHost(&lgb, (size_t)64 * V * 4));
      for (const char* q = bl; *q;) {
        const int n = std::max(1, std::min(64, atoi(q)));
        while (*q && *q != ',') ++q;
        if (*q == ',') ++q;
        std::vector<QwenSeq> bs(n);
        std::vector<QwenSeq*> bp(n);
        std::vector<int32_t> bt(n);
        for (int b = 0; b < n; ++b) {
          eng.init_seq(bs[b], T + spec_steps + 64);
          eng.prefill(bs[b], ids.data(), T, lgb);
          bt[b] = argmax(lgb);
          bp[b] = &bs[b];
        }
        const int steps = std::min(spec_steps, 24);
        int bad = 0;
        std::vector<double> bms;
        const bool bprof = getenv("QWEN_CHECK_PROFILE") && atoi(getenv("QWEN_CHECK_PROFILE")) == 2;   // 2: profile the batched steps
        for (int i = 0; i < steps; ++i) {
          if (bprof && i == 4) CUDA_CHECK(cudaProfilerStart());
          for (int b = 0; b < n; ++b) bad += bt[b] != plain[i];
          const double t0 = hive::mono_ms();
          eng.decode(bp.data(), bt.data(), n, lgb);
          bms.push_back(hive::mono_ms() - t0);
          for (int b = 0; b < n; ++b) bt[b] = argmax(lgb + (size_t)b * V);
        }
        if (bprof) CUDA_CHECK(cudaProfilerStop());
        std::sort(bms.begin(), bms.end());
        printf("qwen_check: batched decode %d sequences × %d steps · median %.2f ms/step · tokens == single-sequence decode: %s (%d differ)\n", n, steps,
               bms[bms.size() / 2], bad ? "NO" : "yes", bad);
      }
      cudaFreeHost(lgb);
    }
    if (eng.mtp_on()) {
      QwenSeq sm;
      eng.init_seq(sm, T + 2 * spec_steps * 8 + 64);
      eng.prefill(sm, ids.data(), T, lg.data());
      tk = argmax(lg.data());
      std::vector<int32_t> emitted;
      std::vector<double> sms, tdr, tvf, trb;
      int acc = 0, prop = 0;
      const int k = eng.mtp_k();
      for (int i = 0; i < spec_steps; ++i) {
        if (prof && i == 4) CUDA_CHECK(cudaProfilerStart());
        const double t0 = hive::mono_ms();
        int32_t d[8]; float cf[8];
        const int nd = eng.mtp_draft(sm, tk, k, d, cf);
        const double t1 = hive::mono_ms();
        std::vector<int32_t> vi{tk};
        for (int j = 0; j < nd; ++j) vi.push_back(d[j]);
        eng.verify(sm, vi.data(), (int)vi.size(), lg.data());
        const double t2 = hive::mono_ms();
        int keep = 1;
        while (keep < (int)vi.size() && argmax(lg.data() + (size_t)(keep - 1) * V) == vi[keep]) ++keep;
        eng.rollback(sm, keep);
        sms.push_back(hive::mono_ms() - t0);
        if (i >= 4) { tdr.push_back(t1 - t0); tvf.push_back(t2 - t1); trb.push_back(hive::mono_ms() - t2); }
        for (int j = 0; j < keep; ++j) emitted.push_back(vi[j]);
        tk = argmax(lg.data() + (size_t)(keep - 1) * V);
        acc += keep - 1; prop += nd;
      }
      if (prof) CUDA_CHECK(cudaProfilerStop());
      std::vector<double> s2(sms.begin() + std::min<size_t>(4, sms.size() - 1), sms.end());
      std::sort(s2.begin(), s2.end());
      double tot = 0; for (double x : sms) tot += x;
      const double tps = emitted.size() / (tot / 1000);
      bool same = true;
      for (size_t i = 0; i < std::min(emitted.size(), plain.size()); ++i) if (emitted[i] != plain[i]) { same = false; break; }
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
      printf("qwen_check: MTP step parts (median ms): draft %.2f · verify %.2f · rollback %.2f\n", med(tdr), med(tvf), med(trb));
      printf("qwen_check: MTP k=%d %d steps · median %.2f ms/step · accept %.2f tokens/step (%d / %d drafts) · %.1f tok/s · greedy tokens %s plain decode\n", k,
             spec_steps, s2[s2.size() / 2], (double)emitted.size() / spec_steps, acc, prop, tps, same ? "==" : "!=");
    }
    // QWEN_CHECK_VBATCH=n: n copies of the prompt with MTP — drafts per sequence, ONE batched verify (verify_batch, n parts) and
    //   rollback_batch per step; every copy's emitted tokens must equal the single-sequence greedy decode
    if (const char* vb = getenv("QWEN_CHECK_VBATCH"); vb && atoi(vb) > 0 && eng.mtp_on()) {
      eng.enable_batch_verify(std::min(atoi(vb), kDecRows / 4));
      const int n = std::min(atoi(vb), eng.verify_parts_cap());
      std::vector<QwenSeq> vs(n);
      std::vector<int32_t> vtk(n);
      std::vector<std::vector<int32_t>> em(n);
      float* lgv = nullptr;
      CUDA_CHECK(cudaMallocHost(&lgv, (size_t)64 * V * 4));
      // QWEN_CHECK_VBATCH_DISTINCT=1: copy b reads the prompt rotated by b·131 tokens — distinct contexts route to distinct experts like a
      //   served batch (identical copies share every expert, which hides the MoE weight traffic); the token check is then skipped
      const bool distinct = getenv("QWEN_CHECK_VBATCH_DISTINCT") && atoi(getenv("QWEN_CHECK_VBATCH_DISTINCT")) > 0;
      for (int b = 0; b < n; ++b) {
        std::vector<int32_t> pb(ids.begin(), ids.begin() + T);
        if (distinct) std::rotate(pb.begin(), pb.begin() + (size_t)(b * 131) % T, pb.end());
        eng.init_seq(vs[b], T + 2 * spec_steps * 8 + 64); eng.prefill(vs[b], pb.data(), T, lgv); vtk[b] = argmax(lgv);
      }
      std::vector<double> vms;
      const int k = eng.mtp_k();
      const int steps = std::min(spec_steps, 24);
      const bool vprof = getenv("QWEN_CHECK_PROFILE") && atoi(getenv("QWEN_CHECK_PROFILE")) == 3;   // 3: profile the batched MTP steps
      std::vector<double> vdr, vvf, vrb;
      long vacc = 0, vprop = 0;   // accepted / proposed drafts (the drafts read the MTP entries rollback_batch writes)
      for (int i = 0; i < steps; ++i) {
        if (vprof && i == 4) CUDA_CHECK(cudaProfilerStart());
        const double t0 = hive::mono_ms();
        std::vector<int32_t> all;
        std::vector<int> Ms(n), keep(n);
        std::vector<QwenSeq*> sp(n);
        for (int b = 0; b < n; ++b) sp[b] = &vs[b];
        std::vector<int32_t> dd((size_t)n * k);
        std::vector<float> cc((size_t)n * k);
        std::vector<int> nd(n);
        eng.mtp_draft_batch(sp.data(), vtk.data(), n, k, dd.data(), cc.data(), nd.data());   // one chain of n rows
        for (int b = 0; b < n; ++b) {
          all.push_back(vtk[b]);
          for (int j = 0; j < nd[b]; ++j) all.push_back(dd[(size_t)b * k + j]);
          Ms[b] = nd[b] + 1;
        }
        const double t1 = hive::mono_ms();
        eng.verify_batch(sp.data(), Ms.data(), n, all.data(), lgv);
        const double t2 = hive::mono_ms();
        int r0 = 0;
        for (int b = 0; b < n; ++b) {
          int kk = 1;
          while (kk < Ms[b] && argmax(lgv + (size_t)(r0 + kk - 1) * V) == all[r0 + kk]) ++kk;
          keep[b] = kk;
          vacc += kk - 1; vprop += Ms[b] - 1;
          for (int j = 0; j < kk; ++j) em[b].push_back(all[r0 + j]);
          vtk[b] = argmax(lgv + (size_t)(r0 + kk - 1) * V);
          r0 += Ms[b];
        }
        const double t3 = hive::mono_ms();   // host argmax above is not the engine's
        eng.rollback_batch(keep.data(), n);
        vms.push_back(hive::mono_ms() - t0);
        if (i >= 4) { vdr.push_back(t1 - t0); vvf.push_back(t2 - t1); vrb.push_back(hive::mono_ms() - t3); }
      }
      if (vprof) CUDA_CHECK(cudaProfilerStop());
      {
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
        printf("qwen_check: batched MTP step parts (median ms): draft %.2f · verify %.2f · rollback %.2f · drafts accepted %ld / %ld\n", med(vdr), med(vvf),
               med(vrb), vacc, vprop);
      }
      int bad = 0;
      for (int b = 0; b < n; ++b)
        for (size_t j = 0; !distinct && j < std::min(em[b].size(), plain.size()); ++j) bad += em[b][j] != plain[j];
      std::sort(vms.begin(), vms.end());
      printf("qwen_check: batched MTP verify %d sequences × %d steps · median %.2f ms/step (draft + verify + rollback) · tokens == single-sequence decode: %s (%d differ)\n",
             n, steps, vms[vms.size() / 2], bad ? "NO" : "yes", bad);
      cudaFreeHost(lgv);
    }
  }
  printf("qwen_check: %d prompt tokens%s · %d decoded · device-routed steps %ld · host-routed %ld · dumps in %s\n", T,
         decode_path ? " (last one decoded)" : "", n_dec, eng.stats().device_route_steps, eng.stats().host_route_steps, out.c_str());
  return 0;
}
