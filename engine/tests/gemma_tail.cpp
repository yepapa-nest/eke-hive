// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// The prefill of a conversation turn's tail: a context of CTX tokens (the prompt repeated), then R times "rewind TAIL_BACK tokens and
//   prefill TAIL new ones" — what a follow-up turn of a long chat costs before its first token. Prints the per-round times (ms).
//   gemma_tail CKPT PROMPT.json CTX TAIL TAIL_BACK ROUNDS
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include "hive/gemma/gemma_engine.h"
#include "nlohmann/json.hpp"

using namespace hive::gemma;

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage: gemma_tail CKPT PROMPT.json CTX TAIL TAIL_BACK ROUNDS\n"); return 2; }
  const std::vector<int32_t> p = nlohmann::json::parse(std::ifstream(argv[2])).get<std::vector<int32_t>>();
  const int ctx = atoi(argv[3]), tail = atoi(argv[4]), back = atoi(argv[5]), rounds = atoi(argv[6]);
  GemmaEngine e(argv[1], "", 131072, 4);
  GemmaSeq s;
  e.init_seq(s, 131072);
  std::vector<int32_t> ids;
  while ((int)ids.size() < ctx) ids.insert(ids.end(), p.begin() + 1, p.end());
  ids.resize(ctx);
  std::vector<float> lg(e.cfg().vocab);
  double t = now_ms();
  e.prefill(s, ids.data(), ctx, lg.data());
  printf("context %d tokens prefilled in %.1f ms\n", ctx, now_ms() - t);
  std::vector<int32_t> tl(p.begin() + 1, p.begin() + 1 + std::min<size_t>(tail, p.size() - 1));
  while ((int)tl.size() < tail) tl.push_back(tl[tl.size() % 16]);
  for (int r = 0; r < rounds; ++r) {
    if (!e.rewind(s, s.pos - back)) { fprintf(stderr, "rewind failed\n"); return 1; }
    cudaDeviceSynchronize();
    t = now_ms();
    e.prefill(s, tl.data(), tail, lg.data());
    printf("round %d: pos %lld · tail %d rows prefilled in %.2f ms\n", r, (long long)s.pos, tail, now_ms() - t);
  }
  return 0;
}
