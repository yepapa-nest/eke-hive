// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_COPY_PRIO — CPU test (fake CUDA/NUMA, no GPU). Built and run by tools/test_copy_prio_cpu.py.
//
// Two real ExpertStores (engine/src/expert_store.cpp) — A = immediate (H2D issued at decision time) · B = paced (set_promo_pacing: H2D pieces are queued and issued per layer within a budget).
//  1. lockstep : with the same observations and promotion call order, A runs commit_all after the decision, B runs commit_pending at the step head (not yet issued → nothing committed) → paced issue over nL layers
//                (budget ⌈remaining / remaining layers⌉, per layer ≤ budget + one piece, all issued by the last layer) → commit_pending at the next head (committed). Every round A == B:
//                slot table · stats (promotions, commits, evictions, h2d records) · promotion H2D bytes · VRAM bytes of resident slots (= reference records built with copy_rec_async).
//  2. t+2     : decision at the end of step t → not committed at the head of t+1 → paced over the layers of t+1 → new decision at the end of t+1 (leftover pieces of the earlier batch are issued first) → only the earlier batch commits at the head of t+2 (FIFO).
//  3. dispatcher: the step-graph dispatcher (hs::Dispatcher, real CPU pool) issues B's pieces via after_dma on every post (dispatcher thread), while the engine thread
//                makes promotion decisions after disarm and runs commit_pending before the next arm — committed slot bytes == reference. Under TSAN all store, fake CUDA and pool accesses are race-checked.
// The fake GPU, world (weights) and mailbox tools are reused from test_decode_handshake_cpu.cpp (included with only main renamed).
#define main hs_handshake_main
#include "test_decode_handshake_cpu.cpp"
#undef main

namespace {

// There is a single world (weight blob) — the fake checkpoint holds the pointer from the first registration, so World must not be created and destroyed repeatedly
World& world() { static World w; return w; }
size_t rec_total() {
  static const size_t t = [] { ExpertStore probe(world().cfg, L, 0, 1, 0); return probe.layout().total; }();
  return t;
}

// Reference record: bytes built with copy_rec_async (same 12 pieces) (0xCD = fake cudaMalloc fill value — same as the empty parts of a slot)
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

// Layer pacing (budget formula of runtime promo_pump — the gate only orders work under fake CUDA, so it is omitted here; ordering is covered by test_cache_cpu)
void pump_step(ExpertStore& b, int nL, size_t max_piece) {
  for (int l = 0; l < nL; ++l) {
    if (!b.promo_backlog()) break;
    const size_t left = b.promo_backlog_bytes(), layers_left = (size_t)std::max(1, nL - l);
    const size_t budget = (left + layers_left - 1) / layers_left;
    const size_t sent = b.issue_promotions(budget);
    EXPECT(sent <= budget + max_piece, "layer %d sent %zu > budget %zu + piece", l, sent, budget);
    EXPECT(left == 0 || sent > 0, "layer %d made no progress", l);
  }
  EXPECT(!b.promo_backlog() && b.promo_backlog_bytes() == 0, "backlog left after the last layer");
}

void test_lockstep(int rounds) {
  World& w = world();
  Checkpoint ck("fake");
  const size_t slots_bytes = 5 * rec_total();
  ExpertStore A(w.cfg, L, slots_bytes, 1, 0), B(w.cfg, L, slots_bytes, 1, 0), R(w.cfg, L, 0, 1, 0);  // R = reference records only (so the stats are untouched)
  for (int l = 0; l < L; ++l) { A.load_layer_experts(ck, l, 2); B.load_layer_experts(ck, l, 2); R.load_layer_experts(ck, l, 2); }
  B.set_promo_pacing(true);
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const HalfLayout& hl = B.half_layout();
  const size_t max_piece = std::max(hl.s1 - hl.w1, hl.s2 - hl.w2);  // largest piece (half of w1/w3 or half of w2)
  std::mt19937 rng(99);
  long batches = 0, promos = 0;
  for (int r = 0; r < rounds && fails == 0; ++r) {
    // same observations (scores) — observe on both stores in the same order
    for (int i = 0; i < 6; ++i) { const int l = (int)(rng() % L), e = (int)(rng() % E); A.observe(l, e); B.observe(l, e); }
    A.decay_scores(0.9f); B.decay_scores(0.9f);
    int na, nb;
    if (r % 3 == 2) { na = A.promote(2, 0.5f, st); nb = B.promote(2, 0.5f, st); }
    else {
      std::vector<int> keys(L * E);
      for (int k = 0; k < L * E; ++k) keys[k] = k;
      std::shuffle(keys.begin(), keys.end(), rng);
      keys.resize(1 + rng() % 5);
      const bool sv = rng() % 2;
      na = A.promote_keys(keys, 2, st, 2, sv); nb = B.promote_keys(keys, 2, st, 2, sv);
    }
    EXPECT(na == nb, "round %d: promote count %d vs %d", r, na, nb);
    A.commit_all();
    // B: head of the next step — a batch not yet issued is not committed
    const int pend = B.n_pending();
    B.commit_pending();
    EXPECT(B.n_pending() == pend, "round %d: unissued batch committed at the head", r);
    pump_step(B, 40, max_piece);
    B.commit_pending();  // the head after that
    EXPECT(B.n_pending() == 0, "round %d: paced batch not resident at the second head", r);
    EXPECT(std::memcmp(A.slot_table(), B.slot_table(), A.slot_table_size() * 4) == 0, "round %d: slot tables differ", r);
    const auto ca = A.cache_stats(), cb = B.cache_stats();
    EXPECT(ca.promotions == cb.promotions && ca.commits == cb.commits && ca.evictions == cb.evictions && ca.h2d_records == cb.h2d_records &&
               ca.resident == cb.resident,
           "round %d: stats differ", r);
    EXPECT(A.promo_h2d_bytes() == B.promo_h2d_bytes(), "round %d: promotion bytes %llu vs %llu", r, (unsigned long long)A.promo_h2d_bytes(),
           (unsigned long long)B.promo_h2d_bytes());
    check_bytes(B, R, "lockstep B");
    check_bytes(A, R, "lockstep A");
    batches += nb > 0; promos += nb;
  }
  printf("copy-prio CPU: lockstep %d rounds (%ld batches · %ld promotions) — paced == immediate: slot table, stats, H2D bytes, VRAM bytes\n", rounds, batches,
         promos);
  cudaStreamDestroy(st);
}

void test_t2() {
  World& w = world();
  Checkpoint ck("fake");
  ExpertStore B(w.cfg, L, 5 * rec_total(), 1, 0), R(w.cfg, L, 0, 1, 0);
  for (int l = 0; l < L; ++l) { B.load_layer_experts(ck, l, 2); R.load_layer_experts(ck, l, 2); }
  B.set_promo_pacing(true);
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  EXPECT(B.promote_keys({0}, 1, st, 1) == 1, "t: decide");       // end of step t
  B.commit_pending();                                           // head of t+1
  EXPECT(B.slot_of(0, 0) < 0, "t+1 head: resident before its copies were issued");
  B.issue_promotions(1);                                        // layers of t+1: only some pieces
  EXPECT(B.promo_backlog(), "fixture: expected a partial issue");
  EXPECT(B.promote_keys({1}, 1, st, 1) == 1, "t+1: decide");     // end of t+1: leftover pieces of the earlier batch first (flush), new batch held
  B.commit_pending();                                           // head of t+2
  EXPECT(B.slot_of(0, 0) >= 0 && B.slot_of(0, 1) < 0, "t+2 head: expected only the first batch resident");
  B.issue_promotions(SIZE_MAX);
  B.commit_pending();                                           // head of t+3
  EXPECT(B.slot_of(0, 1) >= 0, "t+3 head: second batch not resident");
  check_bytes(B, R, "t2");
  printf("copy-prio CPU: residency = first head after the last piece (t+2 for a step-t decision), FIFO, flush before the next decision\n");
  cudaStreamDestroy(st);
}

void test_dispatcher(int steps, int threads) {
  World& w = world();
  Checkpoint ck("fake");
  auto store = std::make_unique<ExpertStore>(w.cfg, L, 6 * rec_total(), threads, 0);
  auto ref = std::make_unique<ExpertStore>(w.cfg, L, 0, 1, 0);
  for (int l = 0; l < L; ++l) { store->load_layer_experts(ck, l, 2); ref->load_layer_experts(ck, l, 2); }
  store->set_promo_pacing(true);
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Rig rg;
  std::atomic<long> pumps{0};
  hs::Dispatcher::Io io = rg.io(store.get(), -1, 0);
  ExpertStore* sp = store.get();
  io.after_dma = [sp, &pumps](int l) {  // same budget formula as runtime promo_pump (dispatcher thread)
    if (!sp->promo_backlog()) return;
    const size_t left = sp->promo_backlog_bytes(), layers_left = (size_t)std::max(1, L - l);
    sp->issue_promotions((left + layers_left - 1) / layers_left);
    pumps.fetch_add(1, std::memory_order_relaxed);
  };
  auto disp = std::make_unique<hs::Dispatcher>(std::move(io));
  std::mt19937 rng(2024);
  uint32_t seq_base = 0, stage_next = 0;
  long posts = 0, committed = 0;
  for (int s = 0; s < steps && fails == 0; ++s) {
    const int before = store->n_resident();
    store->commit_pending();  // step head (engine thread, before arm)
    committed += std::max(0, store->n_resident() - before);
    check_bytes(*store, *ref, "dispatcher head");
    seq_base += L + 1;
    disp->arm();
    bool dev_err = false;
    const GpuResult r = fake_gpu_step(w, rg, rng, seq_base, 1 + (int)(rng() % MB), 20000.0, stage_next, dev_err);
    disp->disarm();
    EXPECT(!dev_err && r.bad_rows == 0 && r.dma_bad == 0 && ld_acq(&rg.ctrl()->cpu_err) == 0, "step %d: handshake not clean", s);
    posts += r.posts;
    // end of step (engine thread, after disarm): promotion decision — leftover pieces of the previous decision go out here first
    for (int i = 0; i < 4; ++i) store->observe((int)(rng() % L), (int)(rng() % E));
    std::vector<int> keys;
    for (int i = 0; i < 3; ++i) keys.push_back((int)(rng() % (L * E)));
    store->promote_keys(keys, 2, st, 2);
  }
  store->commit_all();
  check_bytes(*store, *ref, "dispatcher end");
  printf("copy-prio CPU: dispatcher after_dma pacing %d steps · %ld posts · %ld pumps · %ld commits — resident bytes exact (threads/node %d)\n", steps, posts,
         pumps.load(), committed, threads);
  disp.reset();
  store.reset();
  cudaStreamDestroy(st);
}

}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? atoi(argv[1]) : 400;
  const int steps = argc > 2 ? atoi(argv[2]) : 300;
  const int threads = argc > 3 ? atoi(argv[3]) : 3;
  test_lockstep(rounds);
  test_t2();
  test_dispatcher(steps, threads);
  if (fails) { fprintf(stderr, "copy-prio CPU: %d failure(s)\n", fails); return 1; }
  printf("copy-prio CPU suite OK\n");
  return 0;
}
