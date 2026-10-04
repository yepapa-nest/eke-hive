// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_CACHE_FIT — CPU test (fake CUDA/NUMA, no GPU). Built and run by tools/test_cache_fit_cpu.py.
//
// Two real ExpertStores (engine/src/expert_store.cpp) — A = eager (slot area of n × record bytes allocated in the constructor) · B = deferred (defer_cache → alloc_cache(n) later).
//  1. fit_slots : (free − operating reserve − allocation overhead) divided by the record size (bounds: free ≤ reserve + overhead → 0 · exact multiples · record 0 → 0).
//  2. deferral  : B has 0 slots before alloc_cache · the first call takes n (one contiguous chunk — dev_rec(i) = dev_rec(0) + i·total) · a second call, and a call on the non-deferred A,
//                 do nothing. The 'B' line of the cache event trace is exactly one line for both stores and carries the final slot count.
//  3. lockstep  : with the same observations, same promotions (promote / promote_keys) and same place order, A == B every round — slot table · stats · resident list order · VRAM bytes of resident slots
//                 (= reference records built with copy_rec_async). I.e. deferred allocation is the same state machine as the eager cache except for the slot count.
//  4. zero      : alloc_cache(0) = no cache (same state as --vram-cache-mb 0: 0 slots, 0 resident).
#define main hs_handshake_main
#include "test_decode_handshake_cpu.cpp"
#undef main

#include <fstream>
#include <sstream>

namespace {

World& world() { static World w; return w; }
size_t rec_total() {
  static const size_t t = [] { ExpertStore probe(world().cfg, L, 0, 1, 0); return probe.layout().total; }();
  return t;
}
std::vector<uint8_t> ref_rec(ExpertStore& s, int key) {
  std::vector<uint8_t> r(rec_total(), 0xCD);
  s.copy_rec_async(r.data(), key / s.E(), key % s.E(), nullptr);
  return r;
}
int check_bytes(ExpertStore& s, ExpertStore& ref_src, const char* what) {
  int bad = 0;
  for (size_t k = 0; k < s.slot_table_size(); ++k) {
    const int slot = s.slot_table()[k];
    if (slot < 0) continue;
    const std::vector<uint8_t> r = ref_rec(ref_src, (int)k);
    bad += std::memcmp(s.dev_rec(slot), r.data(), r.size()) != 0;
  }
  EXPECT(bad == 0, "%s: %d resident slot(s) differ from the reference record", what, bad);
  return bad;
}
int count_b_lines(const std::string& path, int* slots) {
  std::ifstream f(path);
  std::string line;
  int n = 0;
  while (std::getline(f, line))
    if (!line.empty() && line[0] == 'B') { ++n; if (slots) *slots = atoi(line.c_str() + 2); }
  return n;
}

void test_fit_slots() {
  const size_t MiB = 1ull << 20, rec = 18800640;
  EXPECT(ExpertStore::fit_slots(0, 0, rec) == 0, "free 0");
  EXPECT(ExpertStore::fit_slots(2400 * MiB, 2400 * MiB, rec) == 0, "free == reserve");
  EXPECT(ExpertStore::fit_slots(2400 * MiB + (64 * MiB), 2400 * MiB, rec) == 0, "free == reserve + overhead");
  EXPECT(ExpertStore::fit_slots(2400 * MiB + 64 * MiB + rec, 2400 * MiB, rec) == 1, "exactly one record");
  EXPECT(ExpertStore::fit_slots(2400 * MiB + 64 * MiB + rec - 1, 2400 * MiB, rec) == 0, "one byte short");
  EXPECT(ExpertStore::fit_slots(2400 * MiB + 64 * MiB + 10 * rec + 5, 2400 * MiB, rec) == 10, "ten records");
  EXPECT(ExpertStore::fit_slots(1000 * MiB, 0, rec, 0) == (int)(1000 * MiB / rec), "no reserve/overhead");
  EXPECT(ExpertStore::fit_slots(1000 * MiB, 0, 0) == 0, "record 0");
  // serving arithmetic: 2,731 MiB free after the pool + 67,989 MiB used by 3,792 slots · operating reserve 2,400 → 3,792 + 15 slots (after subtracting the 64 MiB allocation overhead)
  const size_t svc_free = 2731 * MiB + 3792 * rec;
  EXPECT(ExpertStore::fit_slots(svc_free, 2400 * MiB, rec) == 3792 + (int)((331 * MiB - 64 * MiB) / rec), "service arithmetic: %d",
         ExpertStore::fit_slots(svc_free, 2400 * MiB, rec));
  printf("cache-fit CPU: fit_slots arithmetic (bounds, exact multiples, service numbers) ok\n");
}

void test_lockstep(int rounds, const std::string& tmp) {
  World& w = world();
  Checkpoint ck("fake");
  const int n = 5;
  (void)rec_total();  // create the size-probe store before setting the trace path (the probe writes a 'B' line too)
  const std::string ta = tmp + "/events-a.csv", tb = tmp + "/events-b.csv";
  setenv("HIVE_CACHE_EVENTS", ta.c_str(), 1);
  ExpertStore A(w.cfg, L, (size_t)n * rec_total(), 1, 0);
  setenv("HIVE_CACHE_EVENTS", tb.c_str(), 1);
  ExpertStore B(w.cfg, L, 123 * rec_total() /* ignored when deferred */, 1, 0, /*defer_cache=*/true);
  unsetenv("HIVE_CACHE_EVENTS");
  ExpertStore R(w.cfg, L, 0, 1, 0);
  for (int l = 0; l < L; ++l) { A.load_layer_experts(ck, l, 2); B.load_layer_experts(ck, l, 2); R.load_layer_experts(ck, l, 2); }
  EXPECT(!A.cache_deferred() && B.cache_deferred(), "deferred flags A %d B %d", (int)A.cache_deferred(), (int)B.cache_deferred());
  EXPECT(B.n_slots() == 0 && B.n_resident() == 0, "deferred store has %d slots before alloc_cache", B.n_slots());
  EXPECT(A.alloc_cache(n + 7) == n && A.n_slots() == n, "alloc_cache on an eager store changed it (%d)", A.n_slots());
  EXPECT(B.alloc_cache(n) == n && B.n_slots() == n && !B.cache_deferred(), "alloc_cache(%d) gave %d", n, B.n_slots());
  EXPECT(B.alloc_cache(n + 3) == n && B.n_slots() == n, "second alloc_cache changed the slot count (%d)", B.n_slots());
  for (int i = 0; i < n; ++i) EXPECT(B.dev_rec(i) == B.dev_rec(0) + (size_t)i * rec_total(), "slot %d not contiguous", i);
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  std::mt19937 rng(7);
  long promos = 0, places = 0;
  for (int r = 0; r < rounds && fails == 0; ++r) {
    for (int i = 0; i < 6; ++i) { const int l = (int)(rng() % L), e = (int)(rng() % E); A.observe(l, e); B.observe(l, e); }
    A.decay_scores(0.9f); B.decay_scores(0.9f);
    int na, nb;
    const int mode = r % 4;
    if (mode == 0) { na = A.promote(2, 0.5f, st); nb = B.promote(2, 0.5f, st); }
    else if (mode == 3) {  // place during prefill streaming (LRU victim) — synchronous H2D
      const int l = (int)(rng() % L), e = (int)(rng() % E);
      na = A.place(l, e, st); nb = B.place(l, e, st); ++places;
    } else {
      std::vector<int> keys(L * E);
      for (int k = 0; k < L * E; ++k) keys[k] = k;
      std::shuffle(keys.begin(), keys.end(), rng);
      keys.resize(1 + rng() % 5);
      const bool sv = rng() % 2;
      na = A.promote_keys(keys, 2, st, 2, sv); nb = B.promote_keys(keys, 2, st, 2, sv);
    }
    EXPECT(na == nb, "round %d mode %d: result %d vs %d", r, mode, na, nb);
    if (mode != 3) promos += nb;
    if (r % 2) { A.commit_all(); B.commit_all(); } else { A.commit_pending(); B.commit_pending(); }
    EXPECT(std::memcmp(A.slot_table(), B.slot_table(), A.slot_table_size() * 4) == 0, "round %d: slot tables differ", r);
    const auto ca = A.cache_stats(), cb = B.cache_stats();
    EXPECT(ca.promotions == cb.promotions && ca.commits == cb.commits && ca.evictions == cb.evictions && ca.h2d_records == cb.h2d_records &&
               ca.resident == cb.resident && ca.pending == cb.pending && ca.duplicates == 0 && cb.duplicates == 0 && cb.invalid_mappings == 0,
           "round %d: stats differ", r);
    EXPECT(A.resident_keys_by_score() == B.resident_keys_by_score(), "round %d: resident order differs", r);
    if (r % 2) { check_bytes(A, R, "lockstep A"); check_bytes(B, R, "lockstep B"); }
  }
  A.commit_all(); B.commit_all();
  check_bytes(B, R, "lockstep B final");
  cudaStreamDestroy(st);
  int sa = -1, sb = -1;
  const int ba = count_b_lines(ta, &sa), bb = count_b_lines(tb, &sb);
  EXPECT(ba == 1 && sa == n, "eager trace: %d 'B' line(s), slots %d", ba, sa);
  EXPECT(bb == 1 && sb == n, "deferred trace: %d 'B' line(s), slots %d (must be the allocated count)", bb, sb);
  printf("cache-fit CPU: deferred == eager over %d rounds (%ld promotions · %ld places): slot table, stats, resident order, VRAM bytes · trace 'B' once\n",
         rounds, promos, places);
}

void test_zero() {
  World& w = world();
  Checkpoint ck("fake");
  ExpertStore B(w.cfg, L, 0, 1, 0, true);
  for (int l = 0; l < L; ++l) B.load_layer_experts(ck, l, 2);
  EXPECT(B.alloc_cache(0) == 0 && B.n_slots() == 0 && !B.cache_deferred(), "alloc_cache(0)");
  EXPECT(B.alloc_cache(4) == 0, "alloc_cache after a zero allocation changed the count");
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  B.observe(0, 1, 5.f);
  EXPECT(B.promote(4, 0.5f, st) == 0 && B.promote_keys({1, 2}, 2, st, 2) == 0 && B.n_resident() == 0, "zero-slot cache promoted something");
  cudaStreamDestroy(st);
  printf("cache-fit CPU: alloc_cache(0) = no cache (no promotions, no residents)\n");
}

}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? atoi(argv[1]) : 400;
  const std::string tmp = argc > 2 ? argv[2] : "/tmp";
  test_fit_slots();
  test_lockstep(rounds, tmp);
  test_zero();
  if (fails) { fprintf(stderr, "cache-fit CPU: %d failure(s)\n", fails); return 1; }
  printf("cache-fit CPU suite OK\n");
  return 0;
}
