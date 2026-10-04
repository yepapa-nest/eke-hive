// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace hive {
// Exact-sized host allocations. A shared owner survives Runtime when an archived
// image still owns memory. Returned blocks are cached only within a byte budget.
class HostImagePool : public std::enable_shared_from_this<HostImagePool> {
  size_t limit_, cached_=0;
  std::function<uint8_t*(size_t)> alloc_;
  std::function<void(uint8_t*)> free_;
  std::map<size_t,std::vector<uint8_t*>> bins_;
  mutable std::mutex mu_;
  void release(uint8_t* p,size_t n) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mu_);
      if(n<=limit_ && cached_<=limit_-n) { bins_[n].push_back(p);cached_+=n;return; }
    } catch (...) {} // an optional cache allocation must not fail image destruction
    free_(p);
  }
 public:
  HostImagePool(size_t limit,std::function<uint8_t*(size_t)> alloc,std::function<void(uint8_t*)> free)
      :limit_(limit),alloc_(std::move(alloc)),free_(std::move(free)){}
  ~HostImagePool(){trim();}
  std::shared_ptr<uint8_t> acquire(size_t n) {
    auto owner=shared_from_this();uint8_t* p=nullptr;
    {std::lock_guard<std::mutex> lock(mu_);auto it=bins_.find(n);
      if(it!=bins_.end() && !it->second.empty()){p=it->second.back();it->second.pop_back();cached_-=n;if(it->second.empty())bins_.erase(it);}}
    if(!p) p=alloc_(n);
    return std::shared_ptr<uint8_t>(p,[owner,n](uint8_t* q){owner->release(q,n);});
  }
  size_t cached_bytes() const {std::lock_guard<std::mutex> lock(mu_);return cached_;}
  void trim() {
    std::map<size_t,std::vector<uint8_t*>> old;
    {std::lock_guard<std::mutex> lock(mu_);old.swap(bins_);cached_=0;}
    for(auto& [n,items]:old) for(auto* p:items) free_(p);
  }
};

// Immutable completed-prefix segments. Only the new suffix is transferred; no
// CPU memcpy of the old prefix and no mutable sharing between snapshots.
struct HostImageBuffer {
  struct Segment { std::shared_ptr<uint8_t> data; size_t n; };
  std::vector<Segment> segments;
  size_t size() const { size_t n = 0; for (auto& s : segments) n += s.n; return n; }
  bool empty() const { return segments.empty(); }
  size_t extend(const HostImageBuffer* base, size_t n,
                const std::function<std::shared_ptr<uint8_t>(size_t)>& alloc) {
    segments.clear();
    if (base && base->size() <= n) segments = base->segments;
    const size_t prefix = size();
    if (n > prefix) segments.push_back({alloc(n - prefix), n - prefix});
    return prefix;
  }
  size_t allocated_bytes(std::set<const void*>& seen) const {
    size_t n = segments.capacity() * sizeof(Segment);
    for (auto& s : segments) if (seen.insert(s.data.get()).second) n += s.n;
    return n;
  }
};
}  // namespace hive
