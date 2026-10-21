// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// The session store's segment index (known_ in hive/session_store.h) under a long random sequence of writes and releases:
//   (1) its size stays within a constant multiple of the live segments (expired entries are pruned on writes),
//   (2) a live entry is never dropped — an image made only of segments already on disk writes no new bytes.
// Build: g++ -std=c++20 -O1 -pthread -Iengine/include -Iengine/third_party engine/tests/test_session_store_index_cpu.cpp
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <unistd.h>

#include "hive/session_store.h"

using namespace hive;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

struct TestImage {
  int64_t pos = 0;
  std::vector<int32_t> tokens;            // empty: writes no token bytes, so a write of known segments adds nothing
  std::vector<HostImageBuffer> bufs;
  template <class A> void fields(A& a) { a.i64(pos); a.vi32(tokens); a.bufs(bufs); }
  void wait_ready() const {}
};

int main() {
  char tmpl[] = "/tmp/hive-store-index-XXXXXX";
  const std::string dir = mkdtemp(tmpl);
  {
    store::SessionStore<TestImage> st(dir, "test-index", (size_t)64 << 30);
    std::mt19937_64 rng(7);
    std::vector<std::shared_ptr<TestImage>> live;
    auto segment = [&](size_t n) { std::shared_ptr<uint8_t> p(new uint8_t[n], std::default_delete<uint8_t[]>()); for (size_t i = 0; i < n; ++i) p.get()[i] = (uint8_t)rng(); return HostImageBuffer::Segment{p, n}; };
    auto live_segments = [&] { std::set<const void*> s; for (auto& im : live) for (auto& b : im->bufs) for (auto& sg : b.segments) s.insert(sg.data.get()); return s.size(); };
    uint64_t key = 0;
    size_t worst = 0, known_max = 0, checks = 0;
    for (int it = 0; it < 4000; ++it) {
      auto im = std::make_shared<TestImage>();
      im->pos = it;
      HostImageBuffer b;
      if (!live.empty() && rng() % 2) {  // share a prefix with a live image (as delta images do)
        const auto& base = live[rng() % live.size()]->bufs[0].segments;
        b.segments.assign(base.begin(), base.begin() + (long)(rng() % (base.size() + 1)));
      }
      for (int k = 1 + (int)(rng() % 6); k > 0; --k) b.segments.push_back(segment(1 + rng() % 64));
      im->bufs.push_back(std::move(b));
      st.put_snap(++key, im, {}, false);
      live.push_back(im);
      while (live.size() > 24 || (live.size() > 2 && rng() % 3 == 0)) live.erase(live.begin() + (long)(rng() % live.size()));
      st.drain();
      const size_t known = st.stats()["known"].get<size_t>(), L = live_segments();
      known_max = std::max(known_max, known);
      worst = std::max(worst, L ? known / L : known);
      EXPECT(known <= 4 * L + 16, "iteration %d: index %zu entries for %zu live segments", it, known, L);
      if (it % 50 == 49 && !live.empty()) {  // (2) every segment of a live image is still known: re-writing it adds no bytes
        const size_t w0 = st.stats()["written"].get<size_t>();
        auto again = std::make_shared<TestImage>();
        again->pos = -1;
        again->bufs = live[rng() % live.size()]->bufs;
        st.put_snap(++key, again, {}, false);
        st.drain();
        const size_t w1 = st.stats()["written"].get<size_t>();
        EXPECT(w1 == w0, "iteration %d: re-writing a live image wrote %zu new bytes — a live index entry was dropped", it, w1 - w0);
        ++checks;
      }
      if (fails > 5) break;
    }
    printf("session store index CPU: 4000 random writes/releases · index max %zu entries · worst index/live ratio %zu · %zu live re-writes added 0 bytes\n",
           known_max, worst, checks);
  }
  const std::string rm = "rm -rf '" + dir + "'";
  if (system(rm.c_str()) != 0) {}
  return fails ? 1 : 0;
}
