// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// qtts engine runs on the real checkpoint, for comparison with oracle/qwen3_tts_oracle.py dumps (needs ~4.3 GB of VRAM).
//   test_qtts_engine <model> forced   <ids.json> <codes.npy> <out>   teacher forcing: logits.npy [F+1,16,3072], codes.npy, ids.json
//   test_qtts_engine <model> decode   <codes.npy> <out>              codec only: wav.npy (pcm16 / 32767)
//   test_qtts_engine <model> generate <ids.json> <out> [greedy] [seed=N] [frames=N] [eager]   codes.npy, wav.npy, timing on stdout
//   test_qtts_engine <model> graphcheck <ids.json>                   same seed with graphs and eagerly: codes and pcm must be identical
//   test_qtts_engine <model> bench <ids.json> [runs=N]               first-chunk latency, real-time factor, ms per frame
// ids.json = {"ids": [...], "instruct_ids": [...] | null, "speaker_id": n | null, "language_id": n | null} (the oracle writes it).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hive/qtts/qtts_engine.h"
#include "nlohmann/json.hpp"

using namespace hive::qtts;
using json = nlohmann::json;

namespace {

struct Prompt {
  std::vector<int32_t> ids, ins;
  int spk = -1, lang = -1;
};

Prompt read_prompt(const std::string& path) {
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  json j = json::parse(f);
  Prompt p;
  p.ids = j.at("ids").get<std::vector<int32_t>>();
  if (j.contains("instruct_ids") && !j["instruct_ids"].is_null()) p.ins = j["instruct_ids"].get<std::vector<int32_t>>();
  if (j.contains("speaker_id") && !j["speaker_id"].is_null()) p.spk = j["speaker_id"];
  if (j.contains("language_id") && !j["language_id"].is_null()) p.lang = j["language_id"];
  return p;
}

// ---- minimal .npy (C order, little endian) ----
std::vector<int64_t> npy_shape(const std::string& header) {
  std::vector<int64_t> s;
  const size_t a = header.find('('), b = header.find(')');
  std::stringstream ss(header.substr(a + 1, b - a - 1));
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (tok.find_first_not_of(" ") != std::string::npos) s.push_back(std::stoll(tok));
  return s;
}
std::vector<int32_t> read_codes_npy(const std::string& path, int* frames) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  char magic[8];
  f.read(magic, 8);
  uint16_t hl = 0;
  if (magic[6] == 1) { f.read((char*)&hl, 2); } else { uint32_t h4; f.read((char*)&h4, 4); hl = (uint16_t)h4; }
  std::string header(hl, ' ');
  f.read(&header[0], hl);
  const auto shape = npy_shape(header);
  const int64_t n = shape.at(0) * shape.at(1);
  std::vector<int32_t> out((size_t)n);
  if (header.find("<i8") != std::string::npos) {
    std::vector<int64_t> v((size_t)n);
    f.read((char*)v.data(), n * 8);
    for (int64_t i = 0; i < n; ++i) out[i] = (int32_t)v[i];
  } else if (header.find("<i4") != std::string::npos) {
    f.read((char*)out.data(), n * 4);
  } else {
    fprintf(stderr, "codes must be int32/int64\n");
    exit(2);
  }
  *frames = (int)shape[0];
  return out;
}
void write_npy(const std::string& path, const void* data, const char* descr, size_t esize, const std::vector<size_t>& shape) {
  std::string sh = "(";
  size_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) { sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : ""); n *= shape[i]; }
  sh += ")";
  std::string h = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + sh + ", }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::ofstream f(path, std::ios::binary);
  const char magic[] = "\x93NUMPY\x01\x00";
  f.write(magic, 8);
  const uint16_t hl = (uint16_t)h.size();
  f.write((const char*)&hl, 2);
  f.write(h.data(), h.size());
  f.write((const char*)data, n * esize);
}
void write_prompt(const std::string& path, const Prompt& p) {
  json j;
  j["ids"] = p.ids;
  j["instruct_ids"] = p.ins.empty() ? json(nullptr) : json(p.ins);
  j["speaker_id"] = p.spk >= 0 ? json(p.spk) : json(nullptr);
  j["language_id"] = p.lang >= 0 ? json(p.lang) : json(nullptr);
  std::ofstream(path) << j.dump();
}

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Run {
  std::vector<int16_t> pcm;
  std::vector<int32_t> codes;
  double first_ms = 0, total_ms = 0, start_ms = 0;
  int frames = 0;
};

Run generate(Engine& e, const Prompt& p, uint64_t seed, const SamplingParams& sp) {
  Run r;
  std::vector<int16_t> buf(4 * 1920);
  const double t0 = now_ms();
  e.start(p.ids.data(), (int)p.ids.size(), p.ins.data(), (int)p.ins.size(), p.spk, p.lang, seed, sp);
  r.start_ms = now_ms() - t0;
  while (true) {
    const int n = e.next_chunk(buf.data(), (int)buf.size(), nullptr);
    if (n <= 0) break;
    if (r.pcm.empty()) r.first_ms = now_ms() - t0;
    r.pcm.insert(r.pcm.end(), buf.begin(), buf.begin() + n);
  }
  r.total_ms = now_ms() - t0;
  r.frames = e.frames();
  r.codes = e.codes();
  return r;
}

void save_wav_npy(const std::string& path, const std::vector<int16_t>& pcm) {
  std::vector<float> w(pcm.size());
  for (size_t i = 0; i < pcm.size(); ++i) w[i] = pcm[i] / 32767.f;
  write_npy(path, w.data(), "<f4", 4, {w.size()});
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: see the header of tests/test_qtts_engine.cu\n"); return 2; }
  const std::string model = argv[1], mode = argv[2];
  auto opt_of = [&](const char* key, long dflt) {
    for (int i = 3; i < argc; ++i)
      if (!strncmp(argv[i], key, strlen(key)) && argv[i][strlen(key)] == '=') return strtol(argv[i] + strlen(key) + 1, nullptr, 10);
    return dflt;
  };
  auto has = [&](const char* key) {
    for (int i = 3; i < argc; ++i) if (!strcmp(argv[i], key)) return true;
    return false;
  };
  EngineOptions o;
  o.max_seq = (int)opt_of("max_seq", 2048);
  if (mode == "forced" || has("dump")) o.dump_frames = 512;
  if (mode == "decode") { o.codec_only = true; o.max_seq = std::max(o.max_seq, 4096); o.max_prompt = 32; }
  const double t0 = now_ms();
  Engine e(model, o);
  const auto& inf = e.info();
  printf("[engine] ready in %.0f ms: weights %.1f MiB, buffers %.1f MiB, context %.1f MiB (%s), process %.1f MiB, max_seq %d\n", now_ms() - t0,
         inf.weight_bytes / 1048576.0, inf.buffer_bytes / 1048576.0, inf.context_bytes / 1048576.0,
         inf.context_measured ? "measured" : "estimated", inf.process_bytes / 1048576.0, inf.max_seq);
  fflush(stdout);
  if (mode == "forced") {
    const Prompt p = read_prompt(argv[3]);
    int F = 0;
    const auto forced = read_codes_npy(argv[4], &F);
    const std::string out = argv[5];
    SamplingParams sp;
    e.start(p.ids.data(), (int)p.ids.size(), p.ins.data(), (int)p.ins.size(), p.spk, p.lang, 0, sp, forced.data(), F);
    std::vector<int16_t> buf(4 * 1920);
    while (e.next_chunk(buf.data(), (int)buf.size(), nullptr) > 0) {}
    const auto lg = e.logits();
    const size_t rows = lg.size() / (16 * 3072);
    write_npy(out + "/logits.npy", lg.data(), "<f4", 4, {rows, 16, 3072});
    const auto c = e.codes();
    std::vector<int64_t> c64(c.begin(), c.end());
    write_npy(out + "/codes.npy", c64.data(), "<i8", 8, {c.size() / 16, 16});
    write_prompt(out + "/ids.json", p);
    printf("[forced] %d frames replayed, %zu logit rows\n", e.frames(), rows);
  } else if (mode == "decode") {
    int F = 0;
    const auto codes = read_codes_npy(argv[3], &F);
    std::vector<int16_t> pcm((size_t)F * 1920);
    const double t1 = now_ms();
    const int n = e.decode_codes(codes.data(), F, pcm.data(), (int)pcm.size());
    printf("[decode] %d frames -> %d samples in %.1f ms\n", F, n, now_ms() - t1);
    pcm.resize(n);
    save_wav_npy(std::string(argv[4]) + "/wav.npy", pcm);
  } else if (mode == "generate") {
    const Prompt p = read_prompt(argv[3]);
    const std::string out = argv[4];
    SamplingParams sp;
    if (has("greedy")) { sp.temperature = 0.f; sp.sub_temperature = 0.f; }
    sp.max_frames = (int)opt_of("frames", 0);
    e.use_graphs(!has("eager"));
    const Run r = generate(e, p, (uint64_t)opt_of("seed", 0), sp);
    std::vector<int64_t> c64(r.codes.begin(), r.codes.end());
    write_npy(out + "/codes.npy", c64.data(), "<i8", 8, {r.codes.size() / 16, 16});
    save_wav_npy(out + "/wav.npy", r.pcm);
    write_prompt(out + "/ids.json", p);
    if (has("dump")) {
      const auto lg = e.logits();
      write_npy(out + "/logits.npy", lg.data(), "<f4", 4, {lg.size() / (16 * 3072), 16, 3072});
    }
    printf("[generate] %d frames, audio %.0f ms, prompt %d rows, start %.1f ms, first chunk %.1f ms, total %.1f ms, RTF %.3f\n", r.frames,
           r.pcm.size() / 24.0, e.prompt_rows(), r.start_ms, r.first_ms, r.total_ms, r.total_ms / (r.pcm.size() / 24.0));
  } else if (mode == "graphcheck") {
    const Prompt p = read_prompt(argv[3]);
    SamplingParams sp;
    sp.max_frames = (int)opt_of("frames", 60);
    e.use_graphs(true);
    const Run a = generate(e, p, 123, sp);
    e.use_graphs(false);
    const Run b = generate(e, p, 123, sp);
    const bool same = a.codes == b.codes && a.pcm == b.pcm;
    printf("[graphcheck] graph %d frames, eager %d frames: %s (graph %.1f ms, eager %.1f ms)\n", a.frames, b.frames,
           same ? "identical" : "DIFFERENT", a.total_ms, b.total_ms);
    return same ? 0 : 1;
  } else if (mode == "bench") {
    const Prompt p = read_prompt(argv[3]);
    const int runs = (int)opt_of("runs", 5);
    SamplingParams sp;
    std::vector<double> first, rtf, per_frame;
    for (int i = 0; i < runs; ++i) {
      const Run r = generate(e, p, 1000 + i, sp);
      first.push_back(r.first_ms);
      rtf.push_back(r.total_ms / (r.pcm.size() / 24.0));
      per_frame.push_back((r.total_ms - r.start_ms) / std::max(1, r.frames));
      printf("[bench] run %d: %d frames, start %.1f ms, first %.1f ms, total %.1f ms, RTF %.3f, %.2f ms/frame\n", i, r.frames, r.start_ms,
             r.first_ms, r.total_ms, rtf.back(), per_frame.back());
    }
    std::sort(first.begin(), first.end());
    std::sort(rtf.begin(), rtf.end());
    printf("[bench] first chunk p50 %.1f ms max %.1f ms, RTF p50 %.3f\n", first[first.size() / 2], first.back(), rtf[rtf.size() / 2]);
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
  }
  return 0;
}
