// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Z1 sleep/wake — store CPU test (fake CUDA/NUMA, no GPU). Built and run by tools/test_sleep_cpu.py (FAKE_CUDA_ASYNC 0 and 1).
//
// Real ExpertStore (engine/src/expert_store.cpp):
//  1. plain    : after random observe/promote/place (some promotions not yet committed — async copies in flight)
//                sleep_release -> slot and staging device bytes 0 · whole table -1 · second sleep frees 0 ·
//                wake_alloc -> same slot count · restore_keys (resident keys at sleep time) -> same resident set ·
//                same score-ordered list · resident records byte-equal to the reference records · scores and use
//                counts unchanged. Promote/place work normally after waking (invariants · bytes). Repeated many times.
//  2. elastic  : elastic store (base 4 + elastic 3), in both the lent and the reclaimed state, wakes into the same state
//                (n_slots · base · elastic · elastic_base == dev_rec(base)) · reclaim/lend keep working.
//  3. short    : FAKE_DEV_MALLOC_MAX simulates a VRAM shortage — if not even the elastic region fits -> -1 (stays asleep
//                · device bytes 0) · if more than the elastic region fits, wake shrunk (absorbed) with a smaller base ·
//                a store without elastic slots absorbs down to 0 slots. The reserve (HIVE_CACHE_FIT) path fits to
//                free − reserve.
//  4. paced    : sleep with E4 pacing (set_promo_pacing) on (commit_all issues all pieces even if some remain) · restore_keys.
// The Python side replays the cache event trace with tools/check_cache_events.py ('B' 0 on sleep · 'B' n on wake).
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
std::vector<int32_t> table_copy(const ExpertStore& s) { return std::vector<int32_t>(s.slot_table(), s.slot_table() + s.slot_table_size()); }
std::vector<int> resident_set(const ExpertStore& s) {
  std::vector<int> v;
  for (size_t k = 0; k < s.slot_table_size(); ++k) if (s.slot_table()[k] >= 0) v.push_back((int)k);
  return v;
}
// mixes observe, promote (leaving uncommitted batches) and place
void churn(ExpertStore& A, std::mt19937& rng, cudaStream_t st, cudaStream_t promo, int n) {
  for (int r = 0; r < n; ++r) {
    for (int i = 0; i < 4; ++i) A.observe((int)(rng() % L), (int)(rng() % E), 1.f + (float)(rng() % 3));
    A.decay_scores(0.95f);
    const int mode = (int)(rng() % 3);
    if (mode == 0) A.promote(3, 0.5f, promo);
    else if (mode == 1) { std::vector<int> keys; for (int i = 0; i < 4; ++i) keys.push_back((int)(rng() % (L * E))); A.promote_keys(keys, 3, promo, 3); }
    else if (A.n_slots() > 0) {
      A.place((int)(rng() % L), (int)(rng() % E), st);
      CUDA_CHECK(cudaStreamSynchronize(st));  // runtime: the next promotion may pick that slot as a victim only after the kernels reading the place() copy (st_) (same rule as the elastic test)
    }
    if (rng() % 3 == 0) A.commit_pending();
  }
}

void sleep_wake_cycle(ExpertStore& A, ExpertStore& R, cudaStream_t st, cudaStream_t promo, const char* what, int round, size_t reserve = 0) {
  CUDA_CHECK(cudaStreamSynchronize(st));  // the runtime calls this after a device sync (place copy = st)
  const int slots0 = A.n_slots(), base0 = A.base_slots(), el0 = A.elastic_slots();
  const bool lent0 = A.elastic_lent();
  const std::vector<uint64_t> uses0 = A.use_counts();
  std::vector<float> score0;
  for (int k = 0; k < L * E; ++k) score0.push_back(A.score_of(k));
  // runtime order: device sync -> sleep_release (commit_all inside) — hived snapshots the resident keys right before
  // sleep, not from the list afterwards (uncommitted promotions are not resident yet)
  const std::vector<int32_t> keys = A.resident_keys_by_score();
  const size_t freed = A.sleep_release();
  EXPECT(freed > 0, "%s round %d: nothing freed", what, round);
  EXPECT(A.asleep() && A.n_slots() == 0 && A.device_bytes() == 0, "%s round %d: asleep %d slots %d bytes %zu", what, round, A.asleep(), A.n_slots(), A.device_bytes());
  bool clean = true;
  for (size_t k = 0; k < A.slot_table_size(); ++k) clean = clean && A.slot_table()[k] < 0;
  EXPECT(clean && A.n_pending() == 0, "%s round %d: slot table not empty while asleep", what, round);
  EXPECT(A.sleep_release() == 0, "%s round %d: second sleep freed something", what, round);
  EXPECT(A.restore_keys(keys, promo) == 0, "%s round %d: restore while asleep", what, round);
  const int n = A.wake_alloc(reserve);
  EXPECT(n == A.sleep_phys_slots() && !A.asleep(), "%s round %d: wake %d of %d", what, round, n, A.sleep_phys_slots());
  EXPECT(A.n_slots() == slots0 && A.base_slots() == base0 && A.elastic_slots() == el0 && A.elastic_lent() == lent0, "%s round %d: shape %d/%d/%d/%d vs %d/%d/%d/%d",
         what, round, A.n_slots(), A.base_slots(), A.elastic_slots(), (int)A.elastic_lent(), slots0, base0, el0, (int)lent0);
  if (el0) EXPECT(A.elastic_base() == A.dev_rec(A.base_slots()), "%s round %d: elastic base is not dev_rec(base)", what, round);
  for (int i = 1; i < (int)A.sleep_phys_slots(); ++i) if (A.dev_rec(i) != A.dev_rec(0) + (size_t)i * rec_total()) { EXPECT(false, "%s: slot %d not contiguous", what, i); break; }
  EXPECT(A.wake_alloc(reserve) == n, "%s round %d: second wake changed the store", what, round);
  int calls = 0, last_done = -1, last_total = -1;
  const int placed = A.restore_keys(keys, promo, [&](int d, int t) { ++calls; EXPECT(d >= last_done, "progress went back"); last_done = d; last_total = t; });
  const int want = std::min<int>((int)keys.size(), A.n_slots());
  EXPECT(placed == want && last_done == want && last_total == want && calls >= 1, "%s round %d: restored %d of %d (progress %d/%d, %d calls)", what, round, placed,
         want, last_done, last_total, calls);
  std::vector<int> want_set(keys.begin(), keys.begin() + want), got = resident_set(A);
  std::sort(want_set.begin(), want_set.end());
  EXPECT(got == want_set, "%s round %d: resident set differs after wake (%zu vs %zu)", what, round, got.size(), want_set.size());
  EXPECT(A.use_counts() == uses0, "%s round %d: use counts changed", what, round);
  bool same = true;
  for (int k = 0; k < L * E; ++k) same = same && A.score_of(k) == score0[(size_t)k];
  EXPECT(same, "%s round %d: scores changed by sleep/wake", what, round);
  CUDA_CHECK(cudaStreamSynchronize(promo));
  check_bytes(A, R, what, round);
  const auto cs = A.cache_stats();
  EXPECT(cs.duplicates == 0 && cs.invalid_mappings == 0 && cs.pending == 0, "%s round %d: dup %d invalid %d pending %d", what, round, cs.duplicates,
         cs.invalid_mappings, cs.pending);
}

void test_plain(int rounds, bool pace, const std::string& trace) {
  World& w = world();
  const size_t rec = rec_total();
  setenv("HIVE_CACHE_EVENTS", trace.c_str(), 1);
  ExpertStore A(w.cfg, L, 6 * rec, 1, 0);
  unsetenv("HIVE_CACHE_EVENTS");
  ExpertStore R(w.cfg, L, 0, 1, 0);
  load_all({&A, &R});
  A.set_promo_pacing(pace);
  cudaStream_t st = nullptr, promo = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&promo, cudaStreamNonBlocking));
  std::mt19937 rng(pace ? 31 : 29);
  int cycles = 0, max_res = 0;
  for (int r = 0; r < rounds && fails == 0; ++r) {
    churn(A, rng, st, promo, 6);
    if (pace && rng() % 2) A.issue_promotions(rec / 3);  // sleep with only some pieces issued (sleep's commit_all issues the rest)
    max_res = std::max(max_res, A.n_resident());
    if (r % 3 == 0) { sleep_wake_cycle(A, R, st, promo, pace ? "plain(paced)" : "plain", r); ++cycles; }
    else { A.commit_all(); CUDA_CHECK(cudaStreamSynchronize(st)); check_bytes(A, R, "plain churn", r); }
  }
  EXPECT(cycles > 5 && max_res >= 4, "scenario too weak: cycles %d max resident %d", cycles, max_res);
  cudaStreamDestroy(st); cudaStreamDestroy(promo);
  printf("sleep CPU: plain store%s · %d sleep/wake cycles over %d rounds · resident set/scores/bytes kept\n", pace ? " (E4 paced)" : "", cycles, rounds);
}

void test_elastic(int rounds) {
  World& w = world();
  const size_t rec = rec_total();
  ExpertStore A(w.cfg, L, 4 * rec, 1, 0), R(w.cfg, L, 0, 1, 0);
  load_all({&A, &R});
  EXPECT(A.elastic_reserve(3 * rec) == 3, "reserve");
  cudaStream_t st = nullptr, promo = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&promo, cudaStreamNonBlocking));
  std::mt19937 rng(41);
  int lent_cycles = 0, reclaimed_cycles = 0;
  for (int r = 0; r < rounds && fails == 0; ++r) {
    churn(A, rng, st, promo, 4);
    if (rng() % 4 == 0) {
      CUDA_CHECK(cudaStreamSynchronize(st));
      if (A.elastic_lent()) A.elastic_reclaim(); else A.elastic_lend();
    }
    if (r % 3 == 0) {
      (A.elastic_lent() ? lent_cycles : reclaimed_cycles)++;
      sleep_wake_cycle(A, R, st, promo, A.elastic_lent() ? "elastic(lent)" : "elastic(reclaimed)", r);
      if (!A.elastic_lent()) for (size_t k = 0; k < A.slot_table_size(); ++k) EXPECT(A.slot_table()[k] < A.base_slots(), "reclaimed wake mapped an elastic slot");
    }
  }
  EXPECT(lent_cycles > 2 && reclaimed_cycles > 0, "scenario too weak: lent %d reclaimed %d", lent_cycles, reclaimed_cycles);
  cudaStreamDestroy(st); cudaStreamDestroy(promo);
  printf("sleep CPU: elastic store · %d lent + %d reclaimed sleep/wake cycles · shape (base/elastic/lent/elastic_base) kept\n", lent_cycles, reclaimed_cycles);
}

void test_short() {
  World& w = world();
  const size_t rec = rec_total();
  char buf[64];
  cudaStream_t promo = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&promo, cudaStreamNonBlocking));
  {  // elastic store: base 4 + elastic 9 (larger than the 8-record staging ring — a per-allocation size cap then gives
     // "staging fits, elastic does not")
    ExpertStore A(w.cfg, L, 4 * rec, 1, 0);
    load_all({&A});
    EXPECT(A.elastic_reserve(9 * rec) == 9, "reserve 9");
    A.sleep_release();
    snprintf(buf, sizeof buf, "%zu", 8 * rec + rec / 2);  // staging (8) fits, the 9 elastic slots do not
    setenv("FAKE_DEV_MALLOC_MAX", buf, 1);
    EXPECT(A.wake_alloc(0) == -1 && A.asleep() && A.device_bytes() == 0 && A.n_slots() == 0, "short wake must stay asleep with nothing allocated");
    snprintf(buf, sizeof buf, "%zu", 11 * rec + rec / 2);  // up to 11 slots (shrinking 13 -> 12 -> 11)
    setenv("FAKE_DEV_MALLOC_MAX", buf, 1);
    const int n = A.wake_alloc(0);
    EXPECT(n == 11 && A.base_slots() == 2 && A.elastic_slots() == 9 && A.n_slots() == 11 && A.elastic_base() == A.dev_rec(2), "shrunk wake: %d slots base %d",
           n, A.base_slots());
    unsetenv("FAKE_DEV_MALLOC_MAX");
    EXPECT(A.elastic_reclaim() == 0 && A.n_slots() == 2, "reclaim after shrunk wake");
    A.elastic_lend();
    EXPECT(A.n_slots() == 11, "lend after shrunk wake");
    // the next sleep/wake uses the (shrunk) physical count of that time
    A.sleep_release();
    EXPECT(A.wake_alloc(0) == 11, "second wake keeps the shrunk size");
  }
  {  // no elastic slots: absorbed down to 0 slots
    ExpertStore A(w.cfg, L, 6 * rec, 1, 0);
    load_all({&A});
    A.sleep_release();
    snprintf(buf, sizeof buf, "%zu", rec / 2);  // not even one slot fits — nor the staging ring (8 records) -> -1
    setenv("FAKE_DEV_MALLOC_MAX", buf, 1);
    EXPECT(A.wake_alloc(0) == -1 && A.asleep(), "no room for the staging ring must stay asleep");
    snprintf(buf, sizeof buf, "%zu", 8 * rec);  // staging fits, and the 6-slot block too (6·rec <= 8·rec) -> 6, no shrinking
    setenv("FAKE_DEV_MALLOC_MAX", buf, 1);
    EXPECT(A.wake_alloc(0) == 6, "plain wake under the limit");
    A.sleep_release();
    snprintf(buf, sizeof buf, "%zu", 4 * rec);  // staging (8·rec) does not fit -> -1
    setenv("FAKE_DEV_MALLOC_MAX", buf, 1);
    EXPECT(A.wake_alloc(0) == -1, "staging over the limit");
    unsetenv("FAKE_DEV_MALLOC_MAX");
    // reserve: fake cudaMemGetInfo free 64 GiB − reserve (64 GiB − 2.5 rec) -> 2 slots
    const size_t reserve = (64ull << 30) - 64ull * 1048576 - 2 * rec - rec / 2;
    EXPECT(A.wake_alloc(reserve) == 2 && A.n_slots() == 2, "reserve-fitted wake: %d slots", A.n_slots());
    A.sleep_release();
    EXPECT(A.wake_alloc(0) == 2, "phys after a fitted wake is the fitted size");
    std::vector<int32_t> keys = {0, 1, 2, 3, 99999, -5, 1};
    EXPECT(A.restore_keys(keys, promo) == 2 && A.n_resident() == 2 && A.slot_of(0, 0) >= 0 && A.slot_of(0, 1) >= 0, "restore into 2 slots (bad keys absorbed)");
  }
  CUDA_CHECK(cudaStreamSynchronize(promo));
  cudaStreamDestroy(promo);
  printf("sleep CPU: VRAM-short wake (elastic: -1 below the elastic region, shrink above it · plain: staging mandatory, slots absorbed) · reserve fit ok\n");
}

}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? atoi(argv[1]) : 120;
  const std::string tmp = argc > 2 ? argv[2] : "/tmp";
  (void)rec_total();
  test_plain(rounds, false, tmp + "/sleep-events.csv");
  test_plain(rounds, true, tmp + "/sleep-events-paced.csv");
  test_elastic(rounds);
  test_short();
  if (fails) { fprintf(stderr, "sleep CPU: %d failure(s)\n", fails); return 1; }
  printf("sleep CPU suite OK\n");
  return 0;
}
