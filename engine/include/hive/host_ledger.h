// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#pragma once
// Incremental host-image accounting for hived's session budget (host_total).
//
// host_total used to walk every held image and insert every segment into one std::set to count shared segments once.
//   Images share their unchanged prefix with the image they were saved from, so one image is thousands of segments
//   (measured 2026-10-21, DS service: median 5,994 · p90 8,466 per image) and the walk grew with everything kept —
//   1,649 boundary snapshots + 111 archived sessions after 39 h. It ran on the engine thread once a second during
//   decode (stats) and several times per request (checkpoint, boundary snapshot, chunk end, budget trim): 3.1 s of a
//   13 s profile, ~15 ms of idle engine time per decode step, 13–175-token follow-up prefills taking 3–4 s, and
//   single-stream decode falling from ~45 to ~24 tok/s over the process lifetime.
//
// The ledger gives the same number from per-image deltas. Images are immutable once saved, so an image's segments
//   are read once when it first appears (through the family's own allocated_bytes, observed via segment_sink) and
//   subtracted when it is no longer referenced. A pass costs O(images referenced) plus the segments of images that
//   appeared or disappeared since the last pass.
//
// Same total as the old walk: each image counted once (its own metadata = allocated_bytes minus its segments), each
//   segment counted once however many images hold it (a data pointer listed with two lengths counts one of them, as
//   the walk did). Lifetimes are unchanged: entries keep weak references — for a make_shared image that keeps only
//   its small allocation block, never its segments — and the shared references taken during a pass end with it.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hive/host_image.h"

namespace hive {
template <class Image>
class HostLedger {
  struct Seg { size_t n; uint32_t holders; };
  struct Entry { std::weak_ptr<Image> w; size_t meta = 0; std::vector<const void*> segs; uint64_t pass = 0; };
  std::unordered_map<const Image*, Entry> imgs_;
  std::unordered_map<const void*, Seg> segs_;
  std::vector<std::shared_ptr<Image>> cur_;
  size_t meta_ = 0, seg_bytes_ = 0;
  uint64_t pass_ = 0;

  // An image's own metadata and its distinct segments (data, bytes), read through its allocated_bytes.
  static size_t read(const Image& img, std::vector<std::pair<const void*, size_t>>& out) {
    out.clear();
    std::set<const void*> seen;
    segment_sink = &out;
    const size_t all = img.allocated_bytes(seen);
    segment_sink = nullptr;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first == b.first; }), out.end());
    size_t own = 0;
    for (const auto& [p, n] : out) own += n;
    return all - own;
  }
  void add(const std::shared_ptr<Image>& img) {
    std::vector<std::pair<const void*, size_t>> v;
    Entry e;
    e.w = img;
    e.meta = read(*img, v);
    e.pass = pass_;
    e.segs.reserve(v.size());
    for (const auto& [p, n] : v) {
      e.segs.push_back(p);
      auto [it, fresh] = segs_.try_emplace(p, Seg{n, 0});
      if (fresh) seg_bytes_ += n;
      ++it->second.holders;
    }
    meta_ += e.meta;
    imgs_.emplace(img.get(), std::move(e));
  }
  void drop(Entry& e) {
    for (const void* p : e.segs) {
      auto it = segs_.find(p);
      if (--it->second.holders == 0) { seg_bytes_ -= it->second.n; segs_.erase(it); }
    }
    meta_ -= e.meta;
  }
  static bool same(const std::weak_ptr<Image>& w, const std::shared_ptr<Image>& p) { return !w.owner_before(p) && !p.owner_before(w); }

 public:
  // One pass: begin(), visit() every image referenced now (repeats are fine), end() -> bytes of all of them.
  void begin() { cur_.clear(); }
  void visit(const std::shared_ptr<Image>& img) { if (img) cur_.push_back(img); }
  size_t end() {
    ++pass_;
    std::vector<std::shared_ptr<Image>> fresh;
    for (const auto& p : cur_) {
      auto it = imgs_.find(p.get());
      if (it != imgs_.end() && same(it->second.w, p)) it->second.pass = pass_;
      else fresh.push_back(p);
    }
    // Unreferenced (or a new image at a freed image's address) leave before anything is added: a freed segment's
    //   address may already belong to a segment of a newly added image.
    for (auto it = imgs_.begin(); it != imgs_.end();)
      if (it->second.pass != pass_) { drop(it->second); it = imgs_.erase(it); } else ++it;
    for (const auto& p : fresh)
      if (!imgs_.count(p.get())) add(p);
    cur_.clear();
    return meta_ + seg_bytes_;
  }
  // What an image not referenced by the last pass would add (0 if it was referenced): its metadata plus its segments
  //   that no referenced image holds — the replacement peak while a new image is being swapped in.
  size_t extra(const Image* img) const {
    if (!img || imgs_.count(img)) return 0;
    std::vector<std::pair<const void*, size_t>> v;
    size_t n = read(*img, v);
    for (const auto& [p, b] : v) if (!segs_.count(p)) n += b;
    return n;
  }
  size_t images() const { return imgs_.size(); }
  size_t segments() const { return segs_.size(); }
};
}  // namespace hive
