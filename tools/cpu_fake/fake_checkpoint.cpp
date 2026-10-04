// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Fake Checkpoint/SafetensorsFile for CPU-only tests: tensors come from an in-memory
// registry filled by the test (fake_ckpt_put). No file is opened.
#include "hive/safetensors.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace hive {
std::map<std::string, TensorInfo>& fake_ckpt_registry() { static std::map<std::string, TensorInfo> r; return r; }
void fake_ckpt_put(const std::string& name, const uint8_t* data, size_t nbytes, const std::string& dtype = "U8") {
  TensorInfo t; t.name = name; t.dtype = dtype; t.nbytes = nbytes; t.data = data; t.shape = {(int64_t)nbytes};
  fake_ckpt_registry()[name] = t;
}
SafetensorsFile::SafetensorsFile(const std::string& path) : path_(path) {}
SafetensorsFile::~SafetensorsFile() {}
void SafetensorsFile::advise_sequential() const {}
void SafetensorsFile::advise_random() const {}
void SafetensorsFile::advise_dontneed() const {}
void SafetensorsFile::drop_cache() const {}
Checkpoint::Checkpoint(const std::string& dir) : dir_(dir) {}
const TensorInfo& Checkpoint::get(const std::string& name) {
  auto it = fake_ckpt_registry().find(name);
  if (it == fake_ckpt_registry().end()) { fprintf(stderr, "fake checkpoint: missing %s\n", name.c_str()); abort(); }
  return it->second;
}
SafetensorsFile& Checkpoint::shard_for(const std::string&) {
  static std::mutex mu; std::lock_guard<std::mutex> l(mu);
  auto& p = open_["fake"]; if (!p) p = std::make_unique<SafetensorsFile>("fake"); return *p;
}
size_t Checkpoint::release_all() { return 0; }
}  // namespace hive
