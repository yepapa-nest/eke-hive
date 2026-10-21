// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive/host_ledger.h == the full deduplicating walk it replaced in hived's host_total, over a random schedule of images
//   that share prefix segments (HostImageBuffer::extend, as save_image does), are dropped, re-referenced from several
//   places, and reallocated at freed addresses. Also: extra() == what the old walk added for a transient image, entries
//   keep no image alive, and a pass costs only the images that changed (timing printed, not asserted). No GPU.
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "hive/host_ledger.h"

using namespace hive;

// Same shape as the family images (hive/*/hived_family.h): metadata + buffers, counted once per image
struct Image {
  std::vector<int32_t> tokens;
  std::vector<HostImageBuffer> bufs;
  size_t allocated_bytes(std::set<const void*>& seen) const {
    if (!seen.insert(this).second) return 0;
    size_t b = sizeof(*this) + tokens.capacity() * 4 + bufs.capacity() * sizeof(HostImageBuffer);
    for (auto& v : bufs) b += v.allocated_bytes(seen);
    return b;
  }
};

static size_t full_walk(const std::vector<std::shared_ptr<Image>>& refs, const Image* transient) {
  std::set<const void*> seen;
  size_t n = 0;
  for (const auto& r : refs) if (r) n += r->allocated_bytes(seen);
  if (transient) n += transient->allocated_bytes(seen);
  return n;
}

int main() {
  std::mt19937_64 rng(20261021);
  auto alloc = [](size_t n) { return std::shared_ptr<uint8_t>(new uint8_t[n], std::default_delete<uint8_t[]>()); };
  auto make = [&](const Image* base, int nbuf, size_t grow) {
    auto img = std::make_shared<Image>();
    img->tokens.resize(1 + rng() % 64);
    img->bufs.resize(nbuf);
    for (int b = 0; b < nbuf; ++b) {
      const HostImageBuffer* pb = base && b < (int)base->bufs.size() ? &base->bufs[b] : nullptr;
      const size_t have = pb ? pb->size() : 0;
      // mostly append to the base (shared prefix); sometimes shrink (copied whole, nothing shared)
      const size_t n = (rng() % 8 == 0) ? 1 + rng() % (have + 1) : have + 1 + rng() % grow;
      img->bufs[b].extend(pb, n, alloc);
    }
    return img;
  };
  // "places" that hold images (sessions' ckpt/live/history, shared snapshots): a slot may repeat another slot's image
  std::vector<std::shared_ptr<Image>> refs(48);
  HostLedger<Image> ledger;
  size_t checks = 0;
  for (int step = 0; step < 6000; ++step) {
    const size_t i = rng() % refs.size();
    switch (rng() % 6) {
      case 0: case 1: { const size_t j = rng() % refs.size(); refs[i] = make(refs[j].get(), 6, 32); break; }  // delta from another image
      case 2: refs[i] = make(nullptr, 6, 64); break;                                                        // fresh image
      case 3: refs[i] = refs[rng() % refs.size()]; break;                                                   // same image held twice
      case 4: refs[i].reset(); break;                                                                       // dropped
      case 5: { refs[i].reset(); refs[i] = make(nullptr, 6, 16); break; }                                   // likely at a freed address
    }
    if (rng() % 3) continue;
    std::shared_ptr<Image> transient = (rng() % 2) ? make(refs[rng() % refs.size()].get(), 6, 32) : refs[rng() % refs.size()];
    ledger.begin();
    for (const auto& r : refs) ledger.visit(r);
    const size_t got = ledger.end() + ledger.extra(transient.get());
    const size_t want = full_walk(refs, transient.get());
    if (got != want) { fprintf(stderr, "step %d: ledger %zu != walk %zu\n", step, got, want); return 1; }
    ++checks;
  }
  // a new image at a freed image's address is a different image: its segments replace the old ones. (A make_shared image
  //   cannot move into a tracked address — the entry's weak reference keeps that block — so images here own their memory
  //   separately, as a shared_ptr adopting a raw pointer would.)
  int reused = 0;
  auto make_sep = [&](int nbuf, size_t grow) { auto m = make(nullptr, nbuf, grow); return std::shared_ptr<Image>(new Image(std::move(*m))); };
  for (int k = 0; k < 64; ++k) {
    std::vector<std::shared_ptr<Image>> one{make_sep(3, 64)};
    ledger.begin(); ledger.visit(one[0]); ledger.end();
    const Image* old = one[0].get();
    one[0].reset();
    one[0] = make_sep(1 + k % 5, 4096);
    reused += one[0].get() == old;
    ledger.begin(); ledger.visit(one[0]);
    if (ledger.end() != full_walk(one, nullptr)) { fprintf(stderr, "address reuse %d: ledger != walk\n", k); return 1; }
  }
  assert(reused > 0);
  // the ledger holds no image: dropping every reference frees every image
  std::weak_ptr<Image> probe = refs[0] ? refs[0] : (refs[0] = make(nullptr, 2, 8));
  ledger.begin(); for (const auto& r : refs) ledger.visit(r); ledger.end();
  for (auto& r : refs) r.reset();
  assert(probe.expired());
  ledger.begin();
  assert(ledger.end() == 0 && ledger.images() == 0 && ledger.segments() == 0);

  // cost: 2,000 images of 200 buffers in delta chains (thousands of segments each) — one image changes per pass
  std::vector<std::shared_ptr<Image>> big;
  std::shared_ptr<Image> chain;
  for (int k = 0; k < 2000; ++k) { chain = make(k % 40 ? chain.get() : nullptr, 200, 4); big.push_back(chain); }
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  const size_t walk = full_walk(big, nullptr);
  const double walk_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  ledger.begin(); for (const auto& r : big) ledger.visit(r);
  const size_t first = ledger.end();
  assert(first == walk);
  t0 = clk::now();
  for (int k = 0; k < 20; ++k) {
    big[rng() % big.size()] = make(big[rng() % big.size()].get(), 200, 4);
    ledger.begin(); for (const auto& r : big) ledger.visit(r);
    ledger.end();
  }
  const double pass_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / 20;
  ledger.begin(); for (const auto& r : big) ledger.visit(r);
  assert(ledger.end() == full_walk(big, nullptr));
  printf("host ledger CPU: %zu random passes == full walk, address reuse %d/64, no image kept alive · 2,000 images / %zu segments: full walk %.1f ms, ledger pass %.2f ms\n",
         checks, reused, ledger.segments(), walk_ms, pass_ms);
  return 0;
}
