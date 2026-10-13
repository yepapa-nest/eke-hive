// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// PLE n-gram hash (hive/qwen/ple.h PleHash) against vectors written by oracle/qwen4_ple_vectors.py — the oracle's own implementation of the
//   same rule. CPU only (no GPU, no checkpoint): the default vectors use made-up constants (tools/fixtures/qwen_ple_selftest.json); pass the
//   file the script writes with --ckpt to check the checkpoint's constants as well.
//   Also checks that hashing a sequence in pieces (with the two preceding tokens passed in) equals hashing it whole.
//   usage: test_qwen_ple_cpu [vectors.json ...]
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "hive/qwen/ple.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using hive::qwen::kPleHeads;
using hive::qwen::PleHash;

static int check(const std::string& path) {
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "FAIL cannot open %s\n", path.c_str()); return 1; }
  const json d = json::parse(f);
  PleHash h;
  h.ngram = 3; h.heads_per = 8; h.eos = d["eos"].get<int>();
  h.mult = d["mult"].get<std::vector<int64_t>>();
  h.size = d["size"].get<std::vector<int64_t>>();
  h.offset = d["offset"].get<std::vector<int64_t>>();
  const auto tok = d["tokens"].get<std::vector<int32_t>>();
  const auto want = d["rows"].get<std::vector<std::vector<int64_t>>>();
  const int n = (int)tok.size();
  std::vector<int64_t> got((size_t)n * kPleHeads);
  h.rows_of(tok.data(), n, -1, -1, got.data());
  int bad = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < kPleHeads; ++j)
      if (got[(size_t)i * kPleHeads + j] != want[i][j]) {
        if (bad++ < 5) fprintf(stderr, "  token %d head %d: got %lld want %lld\n", i, j, (long long)got[(size_t)i * kPleHeads + j], (long long)want[i][j]);
      }
  // pieces: every split point gives the same rows when the two preceding tokens are passed
  for (int cut = 1; cut < n; ++cut) {
    std::vector<int64_t> a((size_t)n * kPleHeads);
    h.rows_of(tok.data(), cut, -1, -1, a.data());
    h.rows_of(tok.data() + cut, n - cut, cut >= 2 ? tok[cut - 2] : -1, tok[cut - 1], a.data() + (size_t)cut * kPleHeads);
    if (a != got && bad++ < 8) fprintf(stderr, "  split at %d differs from the whole sequence\n", cut);
  }
  printf("%s %s: %d tokens × %d heads\n", bad ? "FAIL" : "PASS", path.c_str(), n, kPleHeads);
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) files.push_back(argv[i]);
  if (files.empty()) files.push_back(std::string(HIVE_SOURCE_DIR_HINT) + "/../tools/fixtures/qwen_ple_selftest.json");
  int rc = 0;
  for (const auto& f : files) rc |= check(f);
  return rc;
}
