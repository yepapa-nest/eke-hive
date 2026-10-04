// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Vision encoder (reference vision.py): 32-layer ViT (bf16, 2D RoPE, bidirectional attention) + Aligner (3×3 unfold → Linear → GELU → Linear).
// One image → aligner rows (bf16 [n_llm_h·n_llm_w, dim]); the runtime writes them into the IMAGE slots of the image span.
#pragma once
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <vector>

#include "hive/cublas_ops.h"
#include "hive/model.h"

namespace hive {

struct VisionLayer {
  DevBuf norm1, norm2;        // f32 [vdim]
  DevBuf wqkv, bqkv;          // bf16 [3·vdim, vdim], [3·vdim]
  DevBuf wo, bo;              // bf16 [vdim, vdim], [vdim]
  DevBuf w1, w2;              // bf16 [2·inter, vdim], [vdim, inter]
};

struct VisionWeights {
  DevBuf patch_w, patch_b;    // bf16 [vdim, 3·p·p], [vdim]
  std::vector<VisionLayer> layers;
  DevBuf norm;                // f32 [vdim]
  DevBuf al_w1, al_b1, al_w2, al_b2;  // bf16 aligner
  DevBuf image_start, image_end, image_newline;  // bf16 [dim]
};

class Vision {
 public:
  Vision(Model& model, Blas& blas, cudaStream_t st, int max_patches);
  // patches: bf16 [n_vit_h·n_vit_w, 3·p·p] (device) → out: bf16 [n_llm_h·n_llm_w, dim] (device)
  void encode(const bf16* patches, int n_vit_h, int n_vit_w, bf16* out);
  const VisionWeights& w() const { return w_; }

 private:
  Model& model_;
  Blas& blas_;
  cudaStream_t st_;
  VisionWeights w_;
  int max_patches_;
  DevBuf x, xn, qkv, q, k, v, attn, o, mlp, act, cos_, sin_, unfold, al_hidden, scores, probs;
  void load();
};

// Opt-in HIVE_VIT_CACHE_MB=N (default 0 = off): keeps the encoder output of an image (aligner rows, bf16) in host RAM as an LRU — when the same picture
//   comes back on a later turn, the 32 ViT layers are skipped and the stored rows are uploaded. A hit requires an **exact byte match**: the hash only selects
//   candidates; patch bytes, types and grid are compared in full (a hash collision cannot return the wrong image). Value = exactly the bytes encode produced for that input.
//   Host only (no CUDA) — tested directly on the CPU by tools/test_snapshot_cpu.py (engine/tests/test_prefill_host_cpu.cpp).
class VitCache {
 public:
  struct Entry {
    uint64_t hash = 0;
    int n_vit_h = 0, n_vit_w = 0;
    std::vector<int8_t> types;
    std::vector<uint8_t> patches, rows;  // input patch bf16 bytes · output row bf16 bytes
    size_t bytes() const { return patches.size() + rows.size() + types.size() + sizeof(Entry); }
  };
  explicit VitCache(size_t cap_bytes) : cap_(cap_bytes) {}
  // 64-bit candidate hash (multiplicative mixing over 8-byte words + tail bytes + types + grid). Hits are decided by the full comparison in find.
  static uint64_t hash_of(const uint8_t* p, size_t n, int n_vit_h, int n_vit_w, const int8_t* types, size_t n_types) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t)n;
    auto mix = [&h](uint64_t v) { h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 31; };
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t v; std::memcpy(&v, p + i, 8); mix(v); }
    for (; i < n; ++i) mix(p[i]);
    for (size_t t = 0; t < n_types; ++t) mix((uint8_t)types[t]);
    mix((uint32_t)n_vit_h); mix((uint32_t)n_vit_w);
    return h;
  }
  // On a hit returns the entry (moved to the front), otherwise nullptr. hash is the hash_of value (the test inserts different bytes under the same hash to check the full comparison).
  const Entry* find(uint64_t hash, const uint8_t* p, size_t n, int n_vit_h, int n_vit_w, const int8_t* types, size_t n_types) {
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
      const Entry& e = *it;
      if (e.hash != hash || e.n_vit_h != n_vit_h || e.n_vit_w != n_vit_w || e.patches.size() != n || e.types.size() != n_types) continue;
      if ((n && std::memcmp(e.patches.data(), p, n) != 0) || (n_types && std::memcmp(e.types.data(), types, n_types) != 0)) continue;
      lru_.splice(lru_.begin(), lru_, it);
      ++hits_;
      return &lru_.front();
    }
    ++misses_;
    return nullptr;
  }
  // Insert: evicts least recently used entries while over the cap. An entry larger than the cap is not inserted (absorbed, not an error).
  void insert(Entry&& e) {
    const size_t b = e.bytes();
    if (b > cap_) { ++skipped_; return; }
    while (!lru_.empty() && bytes_ + b > cap_) { bytes_ -= lru_.back().bytes(); lru_.pop_back(); ++evicted_; }
    bytes_ += b;
    lru_.push_front(std::move(e));
  }
  size_t bytes() const { return bytes_; }
  size_t cap() const { return cap_; }
  size_t size() const { return lru_.size(); }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t evicted() const { return evicted_; }
  uint64_t skipped() const { return skipped_; }

 private:
  size_t cap_ = 0, bytes_ = 0;
  std::list<Entry> lru_;
  uint64_t hits_ = 0, misses_ = 0, evicted_ = 0, skipped_ = 0;
};

}  // namespace hive
