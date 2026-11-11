// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// The session store's pack reference counts and budget eviction (hive/session_store.h) under a long random sequence of checkpoint and
//   snapshot writes that share segments, with a budget small enough that most writes evict:
//   (1) the counts kept as entries come and go equal a full recount — closing and reopening the store (the start-up scan recounts from the
//       metas on disk and drops every pack nothing refers to) finds the same entries and bytes and removes no file,
//   (2) the pack files on disk are exactly the indexed ones and their sizes add up to the store's byte count,
//   (3) the store stays within its budget, every entry it still lists reads back, and a segment whose pack was dropped is written again,
//   (4) the budget drops the least recently *used* entry: an old entry that was read or touched outlives newer ones that were not,
//   (5) a write's time does not grow with the number of entries (the full recount it replaced did).
// Build: g++ -std=c++20 -O1 -pthread -Iengine/include -Iengine/third_party engine/tests/test_session_store_evict_cpu.cpp
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>

#include "hive/session_store.h"

using namespace hive;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); putchar(10); } } while (0)

struct TestImage {
  int64_t pos = 0;
  uint32_t generation = 0;
  std::vector<int32_t> tokens;
  std::vector<HostImageBuffer> bufs;
  template <class A> void fields(A& a) { a.i64(pos); a.vi32(tokens); a.bufs(bufs); }
  void wait_ready() const {}
};
using Store = store::SessionStore<TestImage>;

static size_t dir_bytes(const std::string& d, size_t* files = nullptr) {
  size_t n = 0, f = 0;
  for (const auto& x : store::list_dir(d)) { const long long s = store::file_size(d + "/" + x); if (s > 0) n += (size_t)s; ++f; }
  if (files) *files = f;
  return n;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
  // the store logs one line per write
  if (!freopen("/dev/null", "w", stderr)) {}
  char tmpl[] = "/tmp/hive-store-evict-XXXXXX";
  const std::string dir = mkdtemp(tmpl);
  std::mt19937_64 rng(11);
  auto segment = [&](size_t n) {
    std::shared_ptr<uint8_t> p(new uint8_t[n], std::default_delete<uint8_t[]>());
    for (size_t i = 0; i < n; ++i) p.get()[i] = (uint8_t)rng();
    return HostImageBuffer::Segment{std::move(p), n};
  };

  // ---- (1)(2)(3): random writes under a small budget, reopened every 150 writes -------------------------------------------------------
  {
    const size_t budget = 300 << 10;
    const std::string key = "test-evict", root = dir + "/" + key;
    auto st = std::make_unique<Store>(dir, key, budget);
    std::vector<std::shared_ptr<TestImage>> live;  // images whose segments later ones may share
    std::map<std::string, int64_t> last_ckpt;       // session id -> pos of its latest checkpoint
    uint64_t snap_key = 0;
    long evicted_seen = 0, reopened = 0, rewrites = 0, loads = 0;
    for (int it = 0; it < 3000 && fails < 6; ++it) {
      auto im = std::make_shared<TestImage>();
      im->pos = it + 1;
      im->tokens.assign(1 + rng() % 8, (int32_t)it);
      HostImageBuffer b;
      if (!live.empty() && rng() % 3) {  // share a prefix with an earlier image (delta images)
        const auto& base = live[rng() % live.size()]->bufs[0].segments;
        b.segments.assign(base.begin(), base.begin() + (long)(rng() % (base.size() + 1)));
      }
      for (int k = 1 + (int)(rng() % 5); k > 0; --k) b.segments.push_back(segment(512 + rng() % 4096));
      im->bufs.push_back(std::move(b));
      if (rng() % 2) {
        const std::string sid = "s" + std::to_string(rng() % 12);  // replaces that session's previous checkpoint
        st->put_ckpt(sid, im, std::vector<float>(4, (float)it), {});
        last_ckpt[sid] = im->pos;
      } else {
        st->put_snap(++snap_key, im, {}, rng() % 4 == 0);
      }
      live.push_back(im);
      while (live.size() > 16) live.erase(live.begin() + (long)(rng() % live.size()));
      st->drain();
      const auto s = st->stats();
      const size_t bytes = s["bytes"].get<size_t>(), entries = s["ckpts"].get<size_t>() + s["snaps"].get<size_t>();
      evicted_seen = s["evicted"].get<long>();
      EXPECT(bytes <= budget || entries <= 1, "iteration %d: %zu bytes over the %zu budget with %zu entries", it, bytes, budget, entries);
      EXPECT(s["write_fail"].get<long>() == 0, "iteration %d: a write failed", it);
      if (it % 25 == 24) {  // (2) disk == index
        size_t files = 0;
        const size_t on_disk = dir_bytes(root + "/packs", &files);
        EXPECT(on_disk == bytes, "iteration %d: packs hold %zu bytes on disk, the index counts %zu", it, on_disk, bytes);
        size_t metas = 0; dir_bytes(root + "/metas", &metas);
        EXPECT(metas == entries, "iteration %d: %zu meta files for %zu entries", it, metas, entries);
      }
      if (it % 40 == 39) {  // (3) what the index lists reads back; a checkpoint the budget dropped is gone for good
        for (auto& [sid, pos] : last_ckpt) {
          if (!st->has_ckpt(sid)) continue;
          auto L = st->load_ckpt(sid);
          EXPECT(L && L->image && L->image->pos == pos, "iteration %d: checkpoint %s did not read back", it, sid.c_str());
          ++loads;
        }
        EXPECT(st->stats()["load_fail"].get<long>() == 0, "iteration %d: a load failed", it);
      }
      if (it % 60 == 59 && !live.empty()) {  // (3) re-writing a live image: bytes are written only for segments whose pack is gone, and it reads back
        auto again = std::make_shared<TestImage>(*live[rng() % live.size()]);
        st->put_ckpt("again", again, {}, {});
        st->drain();
        auto L = st->load_ckpt("again");
        bool same = L && L->image && L->image->bufs.size() == 1 && L->image->bufs[0].segments.size() == again->bufs[0].segments.size();
        for (size_t k = 0; same && k < again->bufs[0].segments.size(); ++k) {
          const auto &a = again->bufs[0].segments[k], &b2 = L->image->bufs[0].segments[k];
          same = a.n == b2.n && !memcmp(a.data.get(), b2.data.get(), a.n);
        }
        EXPECT(same, "iteration %d: a re-written image did not read back byte for byte", it);
        last_ckpt["again"] = again->pos;
        ++rewrites;
      }
      if (it % 150 == 149) {  // (1) the running counts == a full recount
        const auto before = st->stats();
        size_t files0 = 0; const size_t disk0 = dir_bytes(root + "/packs", &files0);
        st.reset();
        st = std::make_unique<Store>(dir, key, budget);
        const auto after = st->stats();
        size_t files1 = 0; const size_t disk1 = dir_bytes(root + "/packs", &files1);
        EXPECT(after["ckpts"] == before["ckpts"] && after["snaps"] == before["snaps"], "reopen at %d: entries %s/%s -> %s/%s", it,
               before["ckpts"].dump().c_str(), before["snaps"].dump().c_str(), after["ckpts"].dump().c_str(), after["snaps"].dump().c_str());
        EXPECT(after["bytes"] == before["bytes"], "reopen at %d: %s bytes -> %s", it, before["bytes"].dump().c_str(), after["bytes"].dump().c_str());
        EXPECT(files1 == files0 && disk1 == disk0, "reopen at %d: the start-up scan removed packs (%zu files %zu bytes -> %zu files %zu bytes) — a pack no entry referred to was left on disk",
               it, files0, disk0, files1, disk1);
        ++reopened;
      }
    }
    EXPECT(evicted_seen > 0 || reopened > 0, "the budget never evicted — the test does not exercise eviction");
    printf("session store evict CPU: 3000 writes under a %zu KB budget · %ld reopen recounts equal · %ld re-writes read back · %ld loads\n",
           budget >> 10, reopened, rewrites, loads);
  }

  // ---- (4): recency follows use ----------------------------------------------------------------------------------------------------------
  {
    const std::string key = "test-lru";
    Store st(dir, key, (size_t)40 << 10);  // room for ~4 of the 8 KB entries below
    auto put = [&](uint64_t k) {
      auto im = std::make_shared<TestImage>();
      im->pos = (int64_t)k; im->tokens.assign(2, (int32_t)k);
      HostImageBuffer b; b.segments.push_back(segment(8 << 10)); im->bufs.push_back(std::move(b));
      st.put_snap(k, im, {}, false);
      st.drain();
      std::this_thread::sleep_for(std::chrono::milliseconds(3));  // distinct last_used
    };
    put(1); put(2); put(3); put(4);
    st.touch_snap(1);                 // the engine reused the in-memory copy of 1
    EXPECT(st.load_snap(2) != nullptr, "snapshot 2 should still be stored");  // a read refreshes 2
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    put(5); put(6);                   // two must go: the least recently used are 3 and 4
    EXPECT(st.load_snap(1) != nullptr, "the touched snapshot 1 was evicted (recency = write time?)");
    EXPECT(st.load_snap(2) != nullptr, "the read snapshot 2 was evicted (recency = write time?)");
    EXPECT(st.load_snap(3) == nullptr && st.load_snap(4) == nullptr, "snapshots 3 and 4 (never used again) should have been evicted first");
    EXPECT(st.load_snap(5) != nullptr && st.load_snap(6) != nullptr, "the newest snapshots must stay");
    printf("session store evict CPU: recency follows use (touched and read entries outlive newer unused ones)\n");
  }

  // ---- (5): the time of a write does not grow with the entries --------------------------------------------------------------------------
  {
    const std::string key = "test-scale";
    Store st(dir, key, (size_t)8 << 30);
    std::vector<std::shared_ptr<TestImage>> chain;
    auto write_n = [&](int n, uint64_t k0) {
      const double t0 = now_ms();
      for (int i = 0; i < n; ++i) {
        auto im = std::make_shared<TestImage>();
        im->pos = (int64_t)(k0 + (uint64_t)i); im->tokens.assign(1, (int32_t)i);
        HostImageBuffer b;
        if (!chain.empty()) { const auto& base = chain.back()->bufs[0].segments; b.segments.assign(base.begin(), base.end()); }  // every image refers to every earlier pack
        if (b.segments.size() > 48) b.segments.erase(b.segments.begin(), b.segments.begin() + 1);
        b.segments.push_back(segment(64));
        im->bufs.push_back(std::move(b));
        st.put_snap(k0 + (uint64_t)i, im, {}, false);
        chain.push_back(im);
        if (chain.size() > 2) chain.erase(chain.begin());
      }
      st.drain();
      return (now_ms() - t0) / n;
    };
    const double early = write_n(400, 1000);
    write_n(6000, 10000);
    const double late = write_n(400, 100000);
    // fsync dominates both; the full recount it replaced made `late` grow with the 6,800 entries x 48 packs (several ms each here)
    EXPECT(late < 3 * early + 1.0, "a write took %.3f ms with 6,800 entries against %.3f ms with 400 — it grows with the store", late, early);
    printf("session store evict CPU: write %.3f ms at 400 entries, %.3f ms at 6,800 (x 48 pack references each)\n", early, late);
  }
  const std::string rm = "rm -rf '" + dir + "'";
  if (system(rm.c_str()) != 0) {}
  return fails ? 1 : 0;
}
