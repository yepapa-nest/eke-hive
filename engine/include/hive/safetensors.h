// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// safetensors reader — header (JSON) parsing + mmap. Tensors are exposed as (dtype string, shape, data pointer, byte count).
// Even very large shards (engram, 94 GB) are mmapped, so opening costs nothing. Actual bytes are pulled by memcpy/read where needed.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace hive {

struct TensorInfo {
  std::string name;
  std::string dtype;  // "F8_E4M3" "F8_E8M0" "BF16" "F32" "I8" "U8" ...
  std::vector<int64_t> shape;
  size_t offset = 0;  // start of the data in the file (absolute offset, header included)
  size_t nbytes = 0;
  const uint8_t* data = nullptr;  // mmap pointer (valid while the file is open)
  int64_t numel() const {
    int64_t n = 1;
    for (size_t k = 0; k < shape.size(); ++k) n *= shape[k];
    return n;
  }
};

class SafetensorsFile {
 public:
  explicit SafetensorsFile(const std::string& path);
  ~SafetensorsFile();
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  const TensorInfo* find(const std::string& tensor_name) const;
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  const std::string& path() const { return path_; }
  size_t size() const { return size_; }
  int fd() const { return fd_; }
  // madvise hints: sequential reads (loading) / random (engram lookups)
  void advise_sequential() const;
  void advise_random() const;
  void advise_dontneed() const;  // the pages may be dropped from the page cache (shard fully loaded)
  void drop_cache() const;       // posix_fadvise(DONTNEED): actually evicts this shard's (clean) pages from the page cache — madvise only unmaps

 private:
  std::string path_;
  int fd_ = -1;
  size_t size_ = 0;
  const uint8_t* map_ = nullptr;
  std::map<std::string, TensorInfo> tensors_;
};

// Resolves name → shard via model.safetensors.index.json; shards are opened on demand.
class Checkpoint {
 public:
  explicit Checkpoint(const std::string& dir);
  bool has(const std::string& name) const { return weight_map_.count(name) > 0; }
  const TensorInfo& get(const std::string& name);
  SafetensorsFile& shard_for(const std::string& name);
  const std::string& dir() const { return dir_; }
  std::vector<std::string> names_with_prefix(const std::string& prefix) const;
  void close_shard(const std::string& name);  // closes the mmap of a fully loaded shard
  // After loading: evicts the page cache of every open shard and closes them. Without this the 532 GB checkpoint stayed in the page cache;
  //   together with 460 GB of pinned memory only 16 GB were left free, kswapd and direct reclaim scanned 1.4 billion pages (pgscan) and
  //   kcompactd burned a full core (CPU at 92 °C). Experts, dense weights, engram and embeddings are all copied to the arena/VRAM/pinned
  //   memory, so the shards are never read again.
  size_t release_all();  // returns the number of bytes evicted

 private:
  std::string dir_;
  std::map<std::string, std::string> weight_map_;
  std::map<std::string, std::unique_ptr<SafetensorsFile>> open_;
};

size_t dtype_size(const std::string& dtype);

}  // namespace hive
