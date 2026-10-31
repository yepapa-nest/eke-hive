// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive/ckpt_quant.h on the CPU: CSF decode == the scale matrix it was built from (encoder written here from the format description),
//   every invariant violation refused (negative controls), MXFP8 expansion exact, NVFP4 expansion = one rounding of the fp32 product,
//   nvfp4_scales reads a plain and a compressed tensor alike — through real safetensors files.
//   With --dump DIR OUT BASE...: writes the decoded scales of the checkpoint projections BASE (".…proj.") to OUT/<i>.bin for
//   tools/fp4l_check.py (comparison against an independent Python decoder).
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "hive/ckpt_quant.h"
#include "nlohmann/json.hpp"

using namespace hive;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); ++fails; } } while (0)

// encoder: base = the row's minimum code; entries more than 15 above it become exceptions (delta 0)
static void csf_encode(const std::vector<uint8_t>& s, int R, int C, std::vector<uint8_t>& fixed, std::vector<uint32_t>& exc) {
  const int fc = 16 + 8 * C;
  fixed.assign((size_t)(R / 16) * fc, 0);
  exc.clear();
  for (int r = 0; r < R; ++r) {
    int mn = 255;
    for (int c = 0; c < C; ++c) mn = std::min(mn, (int)s[(size_t)r * C + c]);
    const int g = r / 16;
    fixed[(size_t)g * fc + (r % 16)] = (uint8_t)mn;
    for (int c = 0; c < C; ++c) {
      int d = s[(size_t)r * C + c] - mn;
      if (d > 15) { exc.push_back(((uint32_t)s[(size_t)r * C + c] << 24) | (uint32_t)(r * C + c)); d = 0; }
      const size_t n = (size_t)(r % 16) * C + c;
      fixed[(size_t)g * fc + 16 + (n >> 1)] |= (uint8_t)(d << ((n & 1) * 4));
    }
  }
}

// a small checkpoint directory: one shard + index
struct Shard {
  nlohmann::json hdr = nlohmann::json::object();
  std::vector<uint8_t> data;
  void add(const std::string& name, const std::string& dt, std::vector<int64_t> shape, const void* p, size_t n) {
    hdr[name] = {{"dtype", dt}, {"shape", shape}, {"data_offsets", {data.size(), data.size() + n}}};
    data.insert(data.end(), (const uint8_t*)p, (const uint8_t*)p + n);
  }
  void write(const std::string& dir) {
    const std::string h = hdr.dump();
    std::ofstream f(dir + "/m.safetensors", std::ios::binary);
    const uint64_t n = h.size();
    f.write((const char*)&n, 8); f.write(h.data(), h.size()); f.write((const char*)data.data(), data.size());
    nlohmann::json idx; idx["weight_map"] = nlohmann::json::object();
    for (auto& [k, v] : hdr.items()) idx["weight_map"][k] = "m.safetensors";
    std::ofstream(dir + "/model.safetensors.index.json") << idx.dump();
  }
};

static int dump(int argc, char** argv) {
  Checkpoint ck(argv[2]);
  const std::string out = argv[3];
  for (int i = 4; i < argc; ++i) {
    const std::string b = argv[i];
    const TensorInfo& w = ck.get(b + "weight");
    const int R = (int)w.shape[0], C = (int)(w.shape[1] * 2 / 16);
    std::vector<uint8_t> s((size_t)R * C);
    ckq::nvfp4_scales(ck, b, R, C, s.data());
    std::ofstream(out + "/" + std::to_string(i - 4) + ".bin", std::ios::binary).write((const char*)s.data(), s.size());
  }
  printf("dumped %d\n", argc - 4);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 4 && std::string(argv[1]) == "--dump") return dump(argc, argv);
  std::mt19937 rng(20261031);
  // 1. round trip on random scale matrices (outliers included) + the shapes of the checkpoint (640 × 160, 2560 × 40)
  for (auto [R, C] : std::vector<std::pair<int, int>>{{640, 160}, {2560, 40}, {16, 1}, {32, 3}}) {
    std::vector<uint8_t> s((size_t)R * C);
    for (auto& v : s) v = (uint8_t)(40 + rng() % 12 + (rng() % 50 == 0 ? rng() % 70 : 0));   // codes ≤ 0x7E
    for (auto& v : s) v = std::min<uint8_t>(v, 0x7E);
    std::vector<uint8_t> fx; std::vector<uint32_t> ex;
    csf_encode(s, R, C, fx, ex);
    std::vector<uint8_t> o((size_t)R * C);
    std::string err;
    EXPECT(ckq::csf_decode(fx.data(), R / 16, 16 + 8 * C, ex.data(), ex.size(), R, C, o.data(), &err), err.c_str());
    EXPECT(o == s, "CSF round trip");
    // negative controls
    { auto e2 = ex; e2.push_back((60u << 24) | (uint32_t)(R * C)); EXPECT(!ckq::csf_decode(fx.data(), R / 16, 16 + 8 * C, e2.data(), e2.size(), R, C, o.data(), &err), "index range"); }
    if (!ex.empty()) {   // non-zero delta under an exception
      auto f2 = fx; const uint32_t idx = ex[0] & 0xFFFFFF; const int r = idx / C, c = idx % C; const size_t n = (size_t)(r % 16) * C + c;
      f2[(size_t)(r / 16) * (16 + 8 * C) + 16 + (n >> 1)] |= (uint8_t)(1 << ((n & 1) * 4));
      EXPECT(!ckq::csf_decode(f2.data(), R / 16, 16 + 8 * C, ex.data(), ex.size(), R, C, o.data(), &err), "delta at exception");
    }
    { auto f2 = fx; f2[0] = 0x7E; f2[16] |= 0x0F;   // base 0x7E + delta of (0, 0) ≥ 1 → past the largest finite E4M3 code (unless (0,0) is an exception)
      bool exc00 = false; for (uint32_t w : ex) exc00 = exc00 || (w & 0xFFFFFF) == 0;
      if (!exc00) EXPECT(!ckq::csf_decode(f2.data(), R / 16, 16 + 8 * C, ex.data(), ex.size(), R, C, o.data(), &err), "code overflow"); }
    { auto e2 = ex; if (!e2.empty()) { e2[0] = (0x7Fu << 24) | (e2[0] & 0xFFFFFF); EXPECT(!ckq::csf_decode(fx.data(), R / 16, 16 + 8 * C, e2.data(), e2.size(), R, C, o.data(), &err), "NaN exception"); } }
    EXPECT(!ckq::csf_decode(fx.data(), R / 16, 16 + 8 * C + 1, ex.data(), ex.size(), R, C, o.data(), &err), "fixed shape");
  }
  // 2. formats through a real safetensors file
  char tmpl[] = "/tmp/ckq-XXXXXX";
  const std::string dir = mkdtemp(tmpl);
  Shard sh;
  const int N = 48, K = 128;
  std::vector<uint8_t> mq((size_t)N * K), ms((size_t)N * K / 32);
  for (auto& v : mq) { v = (uint8_t)(rng() & 0xFF); if ((v & 0x7F) == 0x7F) v ^= 1; }   // no NaN codes
  for (auto& v : ms) v = (uint8_t)(110 + rng() % 30);
  sh.add("mx.weight", "F8_E4M3", {N, K}, mq.data(), mq.size());
  sh.add("mx.weight_scale", "U8", {N, K / 32}, ms.data(), ms.size());
  std::vector<uint8_t> nq((size_t)N * K / 2), ns((size_t)N * K / 16);
  for (auto& v : nq) v = (uint8_t)(rng() & 0xFF);
  for (auto& v : ns) v = (uint8_t)(30 + rng() % 40);
  const float g = 3.3e-5f;
  sh.add("nv.weight", "U8", {N, K / 2}, nq.data(), nq.size());
  sh.add("nv.weight_scale", "F8_E4M3", {N, K / 16}, ns.data(), ns.size());
  sh.add("nv.weight_scale_2", "F32", {}, &g, 4);
  std::vector<uint8_t> fx; std::vector<uint32_t> ex;
  const int RC = 32, CC = 8;
  std::vector<uint8_t> cs((size_t)RC * CC);
  for (auto& v : cs) v = (uint8_t)(50 + rng() % 30);
  csf_encode(cs, RC, CC, fx, ex);
  std::vector<uint8_t> cw((size_t)RC * CC * 8);
  sh.add("csf.weight", "U8", {RC, CC * 8}, cw.data(), cw.size());
  sh.add("csf.weight_scale.nvfp4_csf_fixed", "U8", {RC / 16, 16 + 8 * CC}, fx.data(), fx.size());
  sh.add("csf.weight_scale.nvfp4_csf_exceptions", "U32", {(int64_t)ex.size()}, ex.data(), ex.size() * 4);
  sh.add("csf.weight_scale_2", "F32", {}, &g, 4);
  sh.write(dir);
  {
    Checkpoint ck(dir);
    EXPECT(ckq::dense_format(ck, "mx.") == ckq::Dense::MXFP8 && ckq::dense_format(ck, "nv.") == ckq::Dense::NVFP4, "format detection");
    const auto bm = ckq::expand_bf16(ck, "mx.", 3);
    bool exact = true;
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        const float ref = ckq::e4m3_f32(mq[(size_t)n * K + k]) * std::ldexp(1.f, (int)ms[(size_t)n * (K / 32) + k / 32] - 127);
        const uint32_t u = (uint32_t)bm[(size_t)n * K + k] << 16; float got; std::memcpy(&got, &u, 4);
        exact = exact && got == ref;
      }
    EXPECT(exact, "MXFP8 → bf16 is exact");
    const auto bn = ckq::expand_bf16(ck, "nv.", 2);
    bool one = true;
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        const uint8_t b = nq[(size_t)n * (K / 2) + k / 2];
        const float ref = ckq::e2m1_f32((k & 1) ? b >> 4 : b & 15) * (ckq::e4m3_f32(ns[(size_t)n * (K / 16) + k / 16]) * g);
        one = one && bn[(size_t)n * K + k] == ckq::f32_bf16(ref);
      }
    EXPECT(one, "NVFP4 → bf16 = one rounding of e2m1 · (e4m3 · g)");
    std::vector<uint8_t> o((size_t)RC * CC), p((size_t)N * K / 16);
    EXPECT(ckq::has_csf(ck, "csf.") && !ckq::has_csf(ck, "nv."), "CSF detection");
    ckq::nvfp4_scales(ck, "csf.", RC, CC, o.data());
    EXPECT(o == cs, "nvfp4_scales: compressed tensor");
    ckq::nvfp4_scales(ck, "nv.", N, K / 16, p.data());
    EXPECT(p == ns, "nvfp4_scales: plain tensor");
  }
  std::remove((dir + "/m.safetensors").c_str()); std::remove((dir + "/model.safetensors.index.json").c_str()); rmdir(dir.c_str());
  if (fails) { fprintf(stderr, "test_ckpt_quant_cpu: %d failure(s)\n", fails); return 1; }
  printf("test_ckpt_quant_cpu: CSF round trip + 5 negative controls × 4 shapes, MXFP8 exact, NVFP4 one rounding, plain / CSF scales — passed\n");
  return 0;
}
