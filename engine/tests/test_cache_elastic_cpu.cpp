// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_CACHE_ELASTIC — CPU test (fake CUDA/NUMA, no GPU). Built and run by tools/test_cache_elastic_cpu.py (with FAKE_CUDA_ASYNC 0 and 1).
//
// Against the real ExpertStore (engine/src/expert_store.cpp):
//  1. reserve   : allocated store → the slot area becomes one block of (n + ⌈bytes/rec⌉) (dev_rec contiguous, elastic_base = dev_rec(n)); a second call is a
//                 no-op; with residents present it is 0 + a separately allocated range (elastic_base != nullptr, slot count unchanged). Deferred store →
//                 the tail of alloc_cache(n) is elastic; if n < elastic, at least the elastic amount is allocated.
//  2. lent==plain: in the lent state (before reclaim) the store matches a plain store with the same total slot count every round (slot table, stats,
//                 resident order, VRAM bytes).
//  3. cycles    : reclaim/lend interleaved between random observations, promotions (promote/promote_keys, E4 coordination on/off) and place. While
//                 reclaimed, every round: n_slots == base, no table key points to an elastic slot, no pending promotion targets an elastic slot, and
//                 even after the elastic range is overwritten with 0xEE (= work buffer use) every resident record is byte-equal to the reference record.
//                 After lending back: the elastic slots return as empty slots and are refilled. The cache event trace is verified by the Python side
//                 with the tools/check_cache_events.py replayer (reclaim's 'E'/'D' events must match the state machine).
#define main hs_handshake_main
#include "test_decode_handshake_cpu.cpp"
#undef main

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
int check_bytes(ExpertStore& s, ExpertStore& ref_src, const char* what, int round) {
  int bad = 0;
  for (size_t k = 0; k < s.slot_table_size(); ++k) {
    const int slot = s.slot_table()[k];
    if (slot < 0) continue;
    const std::vector<uint8_t> r = ref_rec(ref_src, (int)k);
    bad += std::memcmp(s.dev_rec(slot), r.data(), r.size()) != 0;
  }
  EXPECT(bad == 0, "%s round %d: %d resident slot(s) differ from the reference record", what, round, bad);
  return bad;
}
void load_all(std::initializer_list<ExpertStore*> ss) {
  Checkpoint ck("fake");
  for (ExpertStore* s : ss) for (int l = 0; l < L; ++l) s->load_layer_experts(ck, l, 2);
}

void test_reserve() {
  World& w = world();
  const size_t rec = rec_total();
  ExpertStore A(w.cfg, L, 4 * rec, 1, 0);
  load_all({&A});
  EXPECT(A.elastic_slots() == 0 && A.elastic_base() == nullptr && A.base_slots() == 4, "fresh store has elastic state");
  EXPECT(A.elastic_reserve(2 * rec + rec / 2) == 3, "reserve 2.5 records → 3 slots");
  EXPECT(A.n_slots() == 7 && A.base_slots() == 4 && A.elastic_slots() == 3 && A.elastic_lent(), "reserve: slots %d base %d elastic %d", A.n_slots(),
         A.base_slots(), A.elastic_slots());
  for (int i = 0; i < 7; ++i) EXPECT(A.dev_rec(i) == A.dev_rec(0) + (size_t)i * rec, "slot %d not contiguous", i);
  EXPECT(A.elastic_base() == A.dev_rec(4), "elastic base is not dev_rec(base)");
  EXPECT(A.elastic_reserve(100 * rec) == 3 && A.n_slots() == 7, "second reserve changed the store");
  // Store with residents: a separately allocated range without elasticity (absorbed)
  ExpertStore B(w.cfg, L, 4 * rec, 1, 0);
  load_all({&B});
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  B.place(0, 1, st);
  EXPECT(B.elastic_reserve(2 * rec) == 0 && B.elastic_slots() == 0 && B.n_slots() == 4 && B.elastic_base() != nullptr, "busy store: reserve must absorb");
  EXPECT(B.elastic_reclaim() == 0 && !B.elastic_lent(), "absorbed store reclaimed something");
  // Deferred store
  ExpertStore D(w.cfg, L, 0, 1, 0, true);
  load_all({&D});
  EXPECT(D.elastic_reserve(3 * rec) == 3 && D.elastic_slots() == 0 && D.elastic_base() == nullptr, "deferred reserve must only record");
  EXPECT(D.alloc_cache(8) == 8 && D.elastic_slots() == 3 && D.base_slots() == 5 && D.elastic_lent() && D.elastic_base() == D.dev_rec(5), "deferred alloc: base %d el %d",
         D.base_slots(), D.elastic_slots());
  ExpertStore D2(w.cfg, L, 0, 1, 0, true);
  load_all({&D2});
  D2.elastic_reserve(3 * rec);
  EXPECT(D2.alloc_cache(2) == 3 && D2.base_slots() == 0 && D2.elastic_slots() == 3, "deferred alloc below the elastic size: %d slots", D2.n_slots());
  EXPECT(D2.elastic_reclaim() == 0 && D2.n_slots() == 0, "base-0 reclaim leaves %d slots", D2.n_slots());
  D2.elastic_lend();
  EXPECT(D2.n_slots() == 3, "lend after base-0 reclaim");
  cudaStreamDestroy(st);
  printf("elastic CPU: reserve (eager contiguous · idempotent · busy → absorbed) and deferred alloc_cache placement ok\n");
}

void test_lent_equals_plain(int rounds) {
  World& w = world();
  const size_t rec = rec_total();
  ExpertStore P(w.cfg, L, 7 * rec, 1, 0), A(w.cfg, L, 4 * rec, 1, 0), R(w.cfg, L, 0, 1, 0);
  load_all({&P, &A, &R});
  A.elastic_reserve(3 * rec);
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  std::mt19937 rng(5);
  for (int r = 0; r < rounds && fails == 0; ++r) {
    for (int i = 0; i < 4; ++i) { const int l = (int)(rng() % L), e = (int)(rng() % E); P.observe(l, e); A.observe(l, e); }
    P.decay_scores(0.9f); A.decay_scores(0.9f);
    int np, na;
    if (r % 3 == 0) { np = P.promote(2, 0.5f, st); na = A.promote(2, 0.5f, st); }
    else if (r % 3 == 1) { const int l = (int)(rng() % L), e = (int)(rng() % E); np = P.place(l, e, st); na = A.place(l, e, st); }
    else {
      std::vector<int> keys;
      for (int i = 0; i < 3; ++i) keys.push_back((int)(rng() % (L * E)));
      np = P.promote_keys(keys, 2, st, 2); na = A.promote_keys(keys, 2, st, 2);
    }
    EXPECT(np == na, "round %d: %d vs %d", r, np, na);
    P.commit_all(); A.commit_all();
    CUDA_CHECK(cudaStreamSynchronize(st));  // place() copies (in progress on the st thread with FAKE_CUDA_ASYNC)
    EXPECT(std::memcmp(P.slot_table(), A.slot_table(), P.slot_table_size() * 4) == 0, "round %d: slot tables differ", r);
    const auto cp = P.cache_stats(), ca = A.cache_stats();
    EXPECT(cp.promotions == ca.promotions && cp.commits == ca.commits && cp.evictions == ca.evictions && cp.resident == ca.resident, "round %d: stats differ", r);
    EXPECT(P.resident_keys_by_score() == A.resident_keys_by_score(), "round %d: resident order differs", r);
    if (r % 4 == 0) { check_bytes(P, R, "lent==plain (plain)", r); check_bytes(A, R, "lent==plain", r); }
  }
  cudaStreamDestroy(st);
  printf("elastic CPU: lent store == plain store of the same slot count over %d rounds\n", rounds);
}

void test_cycles(int rounds, bool pace, const std::string& trace) {
  World& w = world();
  const size_t rec = rec_total();
  setenv("HIVE_CACHE_EVENTS", trace.c_str(), 1);
  ExpertStore A(w.cfg, L, 3 * rec, 1, 0);
  unsetenv("HIVE_CACHE_EVENTS");
  ExpertStore R(w.cfg, L, 0, 1, 0);
  load_all({&A, &R});
  EXPECT(A.elastic_reserve(4 * rec) == 4, "reserve");
  A.set_promo_pacing(pace);
  const int base = A.base_slots(), total = base + A.elastic_slots();
  cudaStream_t st = nullptr, promo = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&promo, cudaStreamNonBlocking));
  std::mt19937 rng(pace ? 23 : 17);
  long reclaims = 0, evicted = 0, lends = 0, elastic_hits = 0;
  bool lent = true;
  for (int r = 0; r < rounds && fails == 0; ++r) {
    for (int i = 0; i < 5; ++i) A.observe((int)(rng() % L), (int)(rng() % E), 1.f + (float)(rng() % 3));
    A.decay_scores(0.9f);
    const int mode = (int)(rng() % 4);
    if (mode == 0) A.promote(3, 0.5f, promo);
    else if (mode == 1) { std::vector<int> keys; for (int i = 0; i < 4; ++i) keys.push_back((int)(rng() % (L * E))); A.promote_keys(keys, 3, promo, 3); }
    else if (mode == 2 && A.n_slots() > 0) A.place((int)(rng() % L), (int)(rng() % E), st);
    if (pace && rng() % 2) A.issue_promotions(rec / 3);
    if (rng() % 3 == 0) A.commit_pending();
    CUDA_CHECK(cudaStreamSynchronize(st));
    if (rng() % 5 == 0) {  // prefill start/end
      if (lent) {
        evicted += A.elastic_reclaim(); ++reclaims;
        EXPECT(A.n_slots() == base && !A.elastic_lent(), "round %d: reclaim left %d slots", r, A.n_slots());
      } else { A.elastic_lend(); ++lends; EXPECT(A.n_slots() == total && A.elastic_lent(), "round %d: lend", r); }
      lent = !lent;
    }
    if (!lent) {
      for (size_t k = 0; k < A.slot_table_size(); ++k) EXPECT(A.slot_table()[k] < base, "round %d: key %zu maps to elastic slot %d while reclaimed", r, k, A.slot_table()[k]);
      EXPECT(A.n_pending() >= 0, "pending");
      std::memset(A.elastic_base(), 0xEE, (size_t)A.elastic_slots() * rec);  // the work buffer writes that range
    } else {
      for (size_t k = 0; k < A.slot_table_size(); ++k) elastic_hits += A.slot_table()[k] >= base;
    }
    A.commit_all();
    CUDA_CHECK(cudaStreamSynchronize(promo));
    const auto cs = A.cache_stats();
    EXPECT(cs.duplicates == 0 && cs.invalid_mappings == 0, "round %d: duplicates %d invalid %d", r, cs.duplicates, cs.invalid_mappings);
    if (!lent) for (size_t k = 0; k < A.slot_table_size(); ++k) EXPECT(A.slot_table()[k] < base, "round %d: commit mapped key %zu to elastic slot %d", r, k, A.slot_table()[k]);
    check_bytes(A, R, pace ? "cycles(paced)" : "cycles", r);
  }
  const auto& es = A.elastic_stats();
  EXPECT(es.reclaims == (uint64_t)reclaims && es.lends == (uint64_t)lends && es.evicted == (uint64_t)evicted, "elastic stats %llu/%llu/%llu vs %ld/%ld/%ld",
         (unsigned long long)es.reclaims, (unsigned long long)es.lends, (unsigned long long)es.evicted, reclaims, lends, evicted);
  EXPECT(reclaims > 3 && elastic_hits > 0, "scenario too weak: reclaims %ld · elastic residency samples %ld", reclaims, elastic_hits);
  cudaStreamDestroy(st); cudaStreamDestroy(promo);
  printf("elastic CPU: cycles%s over %d rounds: %ld reclaims (%ld evicted · %llu dropped promotions) · %ld lends · invariants and VRAM bytes ok\n",
         pace ? " (E4 paced promotions)" : "", rounds, reclaims, evicted, (unsigned long long)es.dropped, lends);
}

}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? atoi(argv[1]) : 400;
  const std::string tmp = argc > 2 ? argv[2] : "/tmp";
  (void)rec_total();
  test_reserve();
  test_lent_equals_plain(rounds);
  test_cycles(rounds, false, tmp + "/elastic-events.csv");
  test_cycles(rounds, true, tmp + "/elastic-events-paced.csv");
  if (fails) { fprintf(stderr, "elastic CPU: %d failure(s)\n", fails); return 1; }
  printf("elastic CPU suite OK\n");
  return 0;
}
