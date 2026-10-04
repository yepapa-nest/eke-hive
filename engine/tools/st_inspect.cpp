// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// safetensors inspection tool — tensor list and shapes of a shard, byte sum of one tensor (for loader verification).
#include <cstdio>
#include <cstring>
#include <string>

#include "hive/safetensors.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: st_inspect <file.safetensors> [tensor-name]\n");
    return 1;
  }
  hive::SafetensorsFile f(argv[1]);
  if (argc == 2) {
    for (auto& [name, t] : f.tensors()) {
      printf("%-60s %-8s [", name.c_str(), t.dtype.c_str());
      for (size_t i = 0; i < t.shape.size(); ++i) printf("%s%lld", i ? "," : "", (long long)t.shape[i]);
      printf("] %zu B\n", t.nbytes);
    }
    printf("%zu tensors, file %zu B\n", f.tensors().size(), f.size());
    return 0;
  }
  const hive::TensorInfo* t = f.find(argv[2]);
  if (!t) { fprintf(stderr, "no such tensor\n"); return 1; }
  unsigned long long sum = 0;
  for (size_t i = 0; i < t->nbytes; ++i) sum = sum * 1000003ULL + t->data[i];
  printf("%s %s nbytes=%zu hash=%llx first=%02x %02x %02x %02x\n", t->name.c_str(), t->dtype.c_str(), t->nbytes, sum, t->data[0], t->data[1],
         t->data[2], t->data[3]);
  return 0;
}
