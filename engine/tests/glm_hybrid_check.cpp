// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM GPU engine with CPU-placed layers (HIVE_GLM_GPU_LAYERS — GlmEngine's hybrid path): prefill a prompt, decode greedily, write the logits
//   of the last prompt row and of every step (logits.npy, float32 [N + 1, vocab]) and the tokens (decode.json), so a hybrid run can be compared
//   with an all-GPU run of the same binary (tools/compare_glm.py or numpy): same tokens, logit cosine per step. Prints TTFT and ms per step.
// usage: glm_hybrid_check <ckpt> <ids: comma list or file> <out dir> [decode N = 32] [cache GiB = 40] [cpu threads = 32]
//   GLM_CHECK_CHUNK = prefill chunk (default 4096); GLM_CHECK_MTP=k: speculative decode with k drafts after the greedy run (tokens must match)
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
#include "hive/glm/glm_engine.h"

using namespace hive;
using namespace hive::glm;

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
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: glm_hybrid_check <ckpt> <ids> <out dir> [decode N] [cache GiB] [threads]\n"); return 2; }
  const std::vector<int32_t> ids = parse_ids(argv[2]);
  const std::string out = argv[3];
  const int n_dec = argc > 4 ? atoi(argv[4]) : 32;
  const double cache_gib = argc > 5 ? atof(argv[5]) : 40.0;
  const int threads = argc > 6 ? atoi(argv[6]) : 32;
  const int mtp_k = getenv("GLM_CHECK_MTP") ? atoi(getenv("GLM_CHECK_MTP")) : 0;
  if (system(("mkdir -p '" + out + "'").c_str()) != 0) return 2;
  const double t0 = mono_ms();
  GlmModel model(argv[1], mtp_k > 0);
  GlmExperts experts(model, 0, threads, 8);
  experts.load_all(std::max(16, threads));
  experts.alloc_cache((size_t)(cache_gib * 1073741824.0));
  GlmEngine eng(model, experts, getenv("GLM_CHECK_CHUNK") ? atoi(getenv("GLM_CHECK_CHUNK")) : 4096);
  const int V = model.cfg().vocab, T = (int)ids.size();
  printf("glm_hybrid_check: loaded in %.1f s · %d of %d layers on the GPU · prompt %d tokens · cache %d slots\n", (mono_ms() - t0) / 1000,
         eng.n_gpu_layers(), model.cfg().n_layers, T, experts.n_slots());
  fflush(stdout);
  auto s = eng.new_seq(T + n_dec + 64);
  std::vector<float> logits((size_t)(n_dec + 1) * V);
  const double tp = mono_ms();
  eng.prefill(*s, ids.data(), T, logits.data());
  const double ttft = mono_ms() - tp;
  eng.stats() = GlmForwardStats{};
  std::vector<int32_t> dec;
  std::vector<double> step;
  for (int i = 0; i < n_dec; ++i) {
    const int32_t tok = argmax(logits.data() + (size_t)i * V, V);
    dec.push_back(tok);
    GlmSeq* sp[1] = {s.get()};
    const double ts = mono_ms();
    eng.decode(sp, &tok, 1, logits.data() + (size_t)(i + 1) * V);
    step.push_back(mono_ms() - ts);
  }
  const GlmForwardStats st = eng.stats();
  write_npy_f32(out + "/logits.npy", logits.data(), n_dec + 1, V);
  {
    FILE* f = fopen((out + "/decode.json").c_str(), "w");
    fprintf(f, "{\"prompt\": %d, \"decode\": [", T);
    for (size_t i = 0; i < dec.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", dec[i]);
    fprintf(f, "]}\n");
    fclose(f);
  }
  std::vector<double> so = step;
  std::sort(so.begin(), so.end());
  const double med = so.empty() ? 0 : so[so.size() / 2];
  printf("greedy:");
  for (int32_t t : dec) printf(" %d", t);
  printf("\nTTFT %.2f s (%d tokens, %.1f tok/s) · decode median %.1f ms/step (%.2f tok/s) · CPU layers %.1f ms/step · cache hit %.1f %%\n", ttft / 1000, T,
         T / (ttft / 1000), med, med > 0 ? 1000 / med : 0.0, st.ms_cpu_layers / std::max(1, n_dec),
         experts.stats().routed ? 100.0 * experts.stats().hit / experts.stats().routed : 0.0);
  if (mtp_k > 0 && eng.mtp_on() && n_dec > 1) {
    auto ss = eng.new_seq(T + n_dec + 64);
    std::vector<float> lg((size_t)(mtp_k + 1) * V);
    eng.prefill(*ss, ids.data(), T, lg.data());
    std::vector<int32_t> emitted{(int32_t)argmax(lg.data(), V)};
    int rounds = 0, acc = 0;
    const double ts = mono_ms();
    while ((int)emitted.size() < n_dec) {
      const int32_t tok = emitted.back();
      int32_t dr[8]; float cf[8];
      const int nd = eng.mtp_draft(*ss, tok, mtp_k, dr, cf);
      std::vector<int32_t> vi{tok};
      for (int j = 0; j < nd; ++j) vi.push_back(dr[j]);
      eng.verify(*ss, vi.data(), (int)vi.size(), lg.data());
      int keep = 1;
      while (keep < (int)vi.size() && argmax(lg.data() + (size_t)(keep - 1) * V, V) == vi[keep]) ++keep;
      eng.rollback(*ss, keep);
      for (int j = 1; j < keep; ++j) emitted.push_back(vi[j]);
      emitted.push_back(argmax(lg.data() + (size_t)(keep - 1) * V, V));
      ++rounds; acc += keep - 1;
    }
    emitted.resize(n_dec);
    printf("MTP k=%d: %d rounds · %d drafts accepted · %.2f tok/s · tokens %s greedy\n", mtp_k, rounds, acc, n_dec * 1000.0 / (mono_ms() - ts),
           emitted == dec ? "==" : "!=");
  }
  return 0;
}
