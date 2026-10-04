// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace hive {

using json = nlohmann::json;

size_t dtype_size(const std::string& d) {
  if (d == "F8_E4M3" || d == "F8_E8M0" || d == "I8" || d == "U8" || d == "BOOL") return 1;
  if (d == "BF16" || d == "F16" || d == "I16") return 2;
  if (d == "F32" || d == "I32") return 4;
  if (d == "F64" || d == "I64") return 8;
  throw std::runtime_error("unknown safetensors dtype: " + d);
}

SafetensorsFile::SafetensorsFile(const std::string& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("cannot open " + path);
  struct stat st{};
  if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed " + path);
  size_ = (size_t)st.st_size;
  map_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (map_ == MAP_FAILED) throw std::runtime_error("mmap failed " + path);
  uint64_t hlen;
  memcpy(&hlen, map_, 8);
  if (hlen + 8 > size_) throw std::runtime_error("bad safetensors header " + path);
  json hdr = json::parse(map_ + 8, map_ + 8 + hlen);
  const size_t base = 8 + (size_t)hlen;
  for (auto& [name, v] : hdr.items()) {
    if (name == "__metadata__") continue;
    TensorInfo t;
    t.name = name;
    t.dtype = v["dtype"].get<std::string>();
    for (auto& d : v["shape"]) t.shape.push_back(d.get<int64_t>());
    auto off = v["data_offsets"];
    size_t s = off[0].get<size_t>(), e = off[1].get<size_t>();
    t.offset = base + s;
    t.nbytes = e - s;
    if (t.offset + t.nbytes > size_) throw std::runtime_error("tensor out of file: " + name);
    if ((size_t)t.numel() * dtype_size(t.dtype) != t.nbytes)
      throw std::runtime_error("tensor size mismatch: " + name);
    t.data = map_ + t.offset;
    tensors_.emplace(name, std::move(t));
  }
}

SafetensorsFile::~SafetensorsFile() {
  if (map_ && map_ != MAP_FAILED) munmap((void*)map_, size_);
  if (fd_ >= 0) ::close(fd_);
}

const TensorInfo* SafetensorsFile::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

void SafetensorsFile::advise_sequential() const { madvise((void*)map_, size_, MADV_SEQUENTIAL); }
void SafetensorsFile::advise_random() const { madvise((void*)map_, size_, MADV_RANDOM); }
void SafetensorsFile::advise_dontneed() const { madvise((void*)map_, size_, MADV_DONTNEED); }
void SafetensorsFile::drop_cache() const {
  if (map_ && map_ != MAP_FAILED) madvise((void*)map_, size_, MADV_DONTNEED);
  if (fd_ >= 0) posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
}

Checkpoint::Checkpoint(const std::string& dir) : dir_(dir) {
  std::ifstream f(dir + "/model.safetensors.index.json");
  if (!f) throw std::runtime_error("no index in " + dir);
  json idx = json::parse(f);
  for (auto& [k, v] : idx["weight_map"].items()) weight_map_[k] = v.get<std::string>();
}

SafetensorsFile& Checkpoint::shard_for(const std::string& name) {
  auto it = weight_map_.find(name);
  if (it == weight_map_.end()) throw std::runtime_error("tensor not in index: " + name);
  auto& shard = it->second;
  auto o = open_.find(shard);
  if (o == open_.end()) o = open_.emplace(shard, std::make_unique<SafetensorsFile>(dir_ + "/" + shard)).first;
  return *o->second;
}

const TensorInfo& Checkpoint::get(const std::string& name) {
  auto& sf = shard_for(name);
  const TensorInfo* t = sf.find(name);
  if (!t) throw std::runtime_error("tensor missing in shard: " + name);
  return *t;
}

std::vector<std::string> Checkpoint::names_with_prefix(const std::string& prefix) const {
  std::vector<std::string> out;
  for (auto it = weight_map_.lower_bound(prefix); it != weight_map_.end(); ++it) {
    if (it->first.compare(0, prefix.size(), prefix) != 0) break;
    out.push_back(it->first);
  }
  return out;
}

size_t Checkpoint::release_all() {
  size_t n = 0;
  for (auto& [path, f] : open_) { f->drop_cache(); n += f->size(); }
  open_.clear();
  return n;
}

void Checkpoint::close_shard(const std::string& name) {
  auto it = weight_map_.find(name);
  if (it == weight_map_.end()) return;
  open_.erase(it->second);
}

}  // namespace hive
