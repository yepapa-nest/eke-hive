// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma engine paths agree: the residual of the last prompt token after the loaded layers (HIVE_GEMMA_DUMP) must be the same whether the
//   prompt is prefilled at once (A), its last 3 tokens decoded one by one (B — the CUDA-graph decode step, run twice so the second replays),
//   or its last 4 tokens sent as one verify block (C), and after a verify rolled back and redone (D). Works with HIVE_GEMMA_LAYERS=N.
//   gemma_consistency CKPT PROMPT.json DUMP_PATH
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "hive/gemma/gemma_engine.h"
#include "nlohmann/json.hpp"

using namespace hive::gemma;

static std::vector<float> rd(const char* p) {
  std::vector<float> v;
  FILE* f = fopen(p, "rb");
  if (!f) return v;
  float x;
  while (fread(&x, 4, 1, f) == 1) v.push_back(x);
  fclose(f);
  return v;
}
static double cosv(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
  return d / std::sqrt(na * nb + 1e-30);
}

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  std::vector<int32_t> ids = nlohmann::json::parse(std::ifstream(argv[2])).get<std::vector<int32_t>>();
  const char* dump = argv[3];
  setenv("HIVE_GEMMA_DUMP", dump, 1);
  GemmaEngine e(argv[1], "", 4096, 4);
  const int n = (int)ids.size(), V = e.cfg().vocab;
  std::vector<float> lg((size_t)16 * V);
  GemmaSeq a, b, c;
  e.init_seq(a, 4096); e.init_seq(b, 4096); e.init_seq(c, 4096);
  e.prefill(a, ids.data(), n, lg.data());
  const auto A = rd(dump);
  e.prefill(b, ids.data(), n - 3, lg.data());
  for (int i = n - 3; i < n; ++i) { GemmaSeq* q[1] = {&b}; e.decode(q, &ids[i], 1, lg.data()); }
  const auto B = rd(dump);
  e.prefill(c, ids.data(), n - 4, lg.data());
  GemmaSeq* q[1] = {&c};
  int m = 4;
  e.verify_batch(q, &m, 1, &ids[n - 4], lg.data());
  const auto C = rd(dump);
  int keep = 1;
  e.rollback_batch(&keep, 1);   // back to n - 3 tokens, then the last 3 again as a verify
  m = 3;
  e.verify_batch(q, &m, 1, &ids[n - 3], lg.data());
  const auto D = rd(dump);
  // batch of two decodes: a (one more token) beside b (one more token) == each alone
  printf("A prefill vs B decode x3: cos %.6f\nA prefill vs C verify 4: cos %.6f\nA prefill vs D rollback + verify 3: cos %.6f\n", cosv(A, B), cosv(A, C), cosv(A, D));
  printf("positions a %lld b %lld c %lld · graphs %ld\n", (long long)a.pos, (long long)b.pos, (long long)c.pos, e.graph_captures());
  const bool ok = cosv(A, B) > 0.9995 && cosv(A, C) > 0.9995 && cosv(A, D) > 0.9995 && a.pos == n && b.pos == n && c.pos == n;
  printf("%s\n", ok ? "ALL PASS" : "FAIL");
  return ok ? 0 : 1;
}
