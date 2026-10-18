// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma engine end to end against oracle/gemma4_oracle.py and a reference server: prefill a prompt (token ids from a JSON list), greedy
//   decode N tokens (one row per step), and the drafts the MTP assistant proposes at every step. Writes OUT/logits.f32 (last prompt row),
//   OUT/decode.json {"tokens": [...], "drafts": [[...], ...]} and timings to stdout.
//   gemma_check CKPT ASSISTANT_DIR|- PROMPT.json OUT N [chunk]
//   HIVE_CHECK_IMAGES=images.json: the prompt's images (the oracle's --images format: start, gh, gw, raw bf16 patch file)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "hive/common.h"
#include "hive/gemma/gemma_engine.h"
#include "nlohmann/json.hpp"

using namespace hive::gemma;
using hive::DevBuf;

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: gemma_check CKPT ASSISTANT|- PROMPT.json OUT N\n"); return 2; }
  const std::string ckpt = argv[1], asst = std::string(argv[2]) == "-" ? "" : argv[2], out = argv[4];
  const int N = atoi(argv[5]);
  std::vector<int32_t> ids = nlohmann::json::parse(std::ifstream(argv[3])).get<std::vector<int32_t>>();
  const double t0 = now_ms();
  GemmaEngine e(ckpt, asst, 32768, 4);
  const double t1 = now_ms();
  GemmaSeq s;
  e.init_seq(s, 32768);
  const int V = e.cfg().vocab;
  std::vector<float> lg(V);
  std::vector<GemmaImage> images;
  std::vector<DevBuf> patch_bufs;
  if (const char* ip = getenv("HIVE_CHECK_IMAGES")) {
    const std::string p = ip, base = p.find('/') == std::string::npos ? "." : p.substr(0, p.rfind('/'));
    for (const auto& im : nlohmann::json::parse(std::ifstream(p))) {
      std::ifstream f(base + "/" + im["file"].get<std::string>(), std::ios::binary);
      std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      DevBuf d(raw.size());
      CUDA_CHECK(cudaMemcpy(d.p, raw.data(), raw.size(), cudaMemcpyHostToDevice));
      images.push_back({im["start"].get<int>(), im["gh"].get<int>(), im["gw"].get<int>(), d.as<__nv_bfloat16>()});
      patch_bufs.push_back(std::move(d));
    }
  }
  const double t2 = now_ms();
  // HIVE_CHECK_SPLIT=N: prefill the first N tokens, then the rest (a server request that continues a shared prefix)
  const int split = getenv("HIVE_CHECK_SPLIT") ? atoi(getenv("HIVE_CHECK_SPLIT")) : 0;
  if (split > 0 && split < (int)ids.size() && images.empty()) {
    e.prefill(s, ids.data(), split, nullptr);
    e.prefill(s, ids.data() + split, (int)ids.size() - split, lg.data());
  } else {
    e.prefill(s, ids.data(), (int)ids.size(), lg.data(), images.empty() ? nullptr : &images);
  }
  const double t3 = now_ms();
  { std::ofstream f(out + "/logits.f32", std::ios::binary); f.write((const char*)lg.data(), (size_t)V * 4); }
  auto argmax = [&](const float* r) { int a = 0; for (int i = 1; i < V; ++i) if (r[i] > r[a]) a = i; return a; };
  // HIVE_CHECK_FORCE=<json with "output_ids">: teacher-forced — feed the reference tokens and count where the engine's argmax agrees
  std::vector<int> force;
  if (const char* fp = getenv("HIVE_CHECK_FORCE")) force = nlohmann::json::parse(std::ifstream(fp))["output_ids"].get<std::vector<int>>();
  int agree = 0, total = 0;
  nlohmann::json misses = nlohmann::json::array();
  auto judge = [&](int step, const float* r) {   // the engine's argmax vs force[step]
    if (step >= (int)force.size()) return;
    const int a = argmax(r), want = force[step];
    ++total;
    if (a == want) { ++agree; return; }
    misses.push_back({{"step", step}, {"engine", a}, {"ref", want}, {"gap", r[a] - r[want]}});
  };
  int tok = argmax(lg.data());
  judge(0, lg.data());
  if (!force.empty()) tok = force[0];
  std::vector<int> toks, eos = e.cfg().eos;
  std::vector<std::vector<int>> drafts;
  double tdec = 0, tdraft = 0;
  for (int step = 0; step < N; ++step) {
    toks.push_back(tok);
    bool stop = false;
    for (int x : eos) stop |= x == tok;
    if (stop) break;
    GemmaSeq* q[1] = {&s};
    const double a = now_ms();
    e.decode(q, &tok, 1, lg.data());
    tdec += now_ms() - a;
    int nxt = argmax(lg.data());
    if (!force.empty()) {
      judge(step + 1, lg.data());
      if (step + 1 >= (int)force.size()) break;
      nxt = force[step + 1];
    }
    if (e.mtp_on()) {
      int32_t d[7]; float cf[7];
      const double b = now_ms();
      const int n = e.mtp_draft(s, nxt, e.mtp_k(), d, cf);
      tdraft += now_ms() - b;
      drafts.push_back(std::vector<int>(d, d + n));
    }
    tok = nxt;
  }
  nlohmann::json j;
  j["tokens"] = toks;
  j["drafts"] = drafts;
  if (!force.empty()) {
    j["forced_agree"] = agree; j["forced_total"] = total; j["forced_misses"] = misses;
    printf("teacher-forced: argmax agrees with the reference at %d / %d steps; misses %s\n", agree, total, misses.dump().c_str());
  }
  std::ofstream(out + "/decode.json") << j.dump();
  const auto m = e.mem();
  printf("load %.1f s · prefill %zu tokens %.1f ms · decode %zu steps %.2f ms/step · draft %.2f ms/chain · weights %.0f MiB · KV pool %.0f MiB · "
         "work %.0f MiB\n", (t1 - t0) / 1e3, ids.size(), t3 - t2, toks.size(), toks.empty() ? 0.0 : tdec / toks.size(),
         drafts.empty() ? 0.0 : tdraft / drafts.size(), m.weights / 1048576.0, m.kv_pool / 1048576.0, m.work / 1048576.0);
  return 0;
}
