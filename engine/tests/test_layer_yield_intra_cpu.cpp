// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// T11b HIVE_LAYER_YIELD_INTRA — CPU test (fake CUDA/NUMA, no GPU). Built and run by tools/test_layer_yield_intra_cpu.py.
//   1. hive/layer_yield.h: HIVE_LAYER_YIELD_CAP parsing (absorbs junk) · intra_due truth table (period, look-ahead, cap 0, no sample, cost bound) ·
//      segment EMA attribution (a segment belongs to the point that opened it; the resume after a yield restarts it).
//   2. yield_point_intra: off (s.intra false) never asks want · decode only — R = the floor whatever want reports · the point kind is visible to
//      want/run and reset afterwards · an exception in run still unparks once (depth, kind, stats, last) · look-ahead yields before the period and
//      not right after a resume · park failure = skipped, clock and segment restart · the boundary yield_point never changes s.kind.
//   3. The REAL ExpertStore: copy_to_staging works unheld (record bytes = copy_rec_async) and the hold is off by default; copy_to_staging while held
//      aborts in a child process (the data-corruption guard). Mode --staging-overwrite (used by the Python negative control with a mutant
//      expert_store.cpp that lacks the check): a held copy_to_staging overwrites a "pre-copied" record → prints OVERWRITTEN.
#define main hs_handshake_main
#include "test_decode_handshake_cpu.cpp"
#undef main

#include <csignal>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

#include "hive/layer_yield.h"

namespace {

void test_parse() {
  using hive::ly::parse_cap;
  EXPECT(parse_cap(nullptr) == 200.0 && parse_cap("") == 200.0, "unset/empty = 200");
  EXPECT(parse_cap("0") == 0.0, "0 = look-ahead off");
  EXPECT(parse_cap("abc") == 200.0 && parse_cap("12x") == 200.0 && parse_cap("49") == 200.0 && parse_cap("-5") == 200.0 && parse_cap("nan") == 200.0, "junk/below 50 = default");
  EXPECT(parse_cap("50") == 50.0 && parse_cap("300") == 300.0 && parse_cap("1e9") == 60000.0, "numeric ≥ 50 (capped 60000)");
  EXPECT(std::string(hive::ly::kind_name(hive::ly::kIndex)) == "index" && std::string(hive::ly::kind_name(9)) == "?", "kind names");
  printf("layer-yield intra CPU: HIVE_LAYER_YIELD_CAP parsing ok\n");
}

void test_due() {
  using hive::ly::intra_due;
  EXPECT(intra_due(150, 150, 200, 0, 0), "period reached");
  EXPECT(!intra_due(100, 150, 200, 0, 0), "no sample: period only");
  EXPECT(intra_due(100, 150, 200, 120, 0), "look-ahead: 100 + 120 > 200");
  EXPECT(!intra_due(100, 150, 200, 100, 0), "look-ahead: 100 + 100 = cap (not beyond)");
  EXPECT(!intra_due(100, 150, 0, 5000, 0), "cap 0 = look-ahead off");
  EXPECT(!intra_due(20, 150, 200, 5000, 30), "right after a resume: removed stall (20) < yield cost (30)");
  EXPECT(intra_due(30, 150, 200, 5000, 30), "removed stall = yield cost");
  hive::ly::Segs g;
  g.close(7, 10.0);  // first point: nothing to attribute
  EXPECT(g.next(7) == 0 && g.key == 7 && g.t0 == 10.0, "first close attributes nothing");
  g.close(9, 40.0);  // segment 10 → 40 belongs to key 7
  EXPECT(g.next(7) == 30.0 && g.next(9) == 0, "segment belongs to the opening key (first sample = value)");
  g.close(7, 50.0);
  g.close(9, 70.0);  // key 7 again: 50 → 70 = 20 → EMA 0.9·30 + 0.1·20 = 29
  EXPECT(std::fabs(g.next(7) - 29.0) < 1e-9, "EMA a = 0.1: %f", g.next(7));
  EXPECT(hive::ly::seg_key(hive::ly::kIndex, 3, true) != hive::ly::seg_key(hive::ly::kIndex, 3, false) &&
         hive::ly::seg_key(hive::ly::kIndex, 3, false) != hive::ly::seg_key(hive::ly::kMoe, 3, false) &&
         hive::ly::seg_key(hive::ly::kIndex, 3, false) != hive::ly::seg_key(hive::ly::kIndex, 4, false), "segment keys distinct");
  printf("layer-yield intra CPU: intra_due truth table and segment EMA ok\n");
}

struct Rec {  // a fake runtime side for yield_point_intra
  double t = 0;
  int parks = 0, unparks = 0, wants = 0, runs = 0, run_R = -1, kind_in_want = -1, kind_in_run = -1;
  bool park_ok = true, run_throws = false;
  int want_ret = 1;
  double run_ms = 10;
};
hive::ly::State make_state(Rec& r, double period) {
  hive::ly::State s;
  s.h.period_ms = period;
  s.h.max_depth = 1;
  s.intra = true;
  s.floor = 64; s.cap = 1024;
  s.h.want = [&r, &s] { ++r.wants; r.kind_in_want = s.kind; return r.want_ret; };
  s.h.run = [&r, &s](int R, double) {
    ++r.runs; r.run_R = R; r.kind_in_run = s.kind; r.t += r.run_ms;
    if (r.run_throws) throw std::runtime_error("decode failed");
  };
  return s;
}
bool point(hive::ly::State& s, Rec& r, int kind, int key) {
  return hive::ly::yield_point_intra(s, kind, key, [&](int) { ++r.parks; return r.park_ok; }, [&](int) { ++r.unparks; }, [&] { return r.t; });
}

void test_point() {
  using namespace hive::ly;
  {  // off: no want, no park
    Rec r; State s = make_state(r, 50); s.intra = false; r.t = 1000;
    EXPECT(!point(s, r, kIndex, 1) && r.wants == 0 && r.parks == 0, "intra off: nothing");
  }
  {  // decode only: R = floor even when want reports a prompt's rows; kind visible to want/run, reset after
    Rec r; State s = make_state(r, 50); r.t = 100; r.want_ret = 900;
    EXPECT(point(s, r, kUnit, 1), "yield after the period");
    EXPECT(r.run_R == 64, "R = floor (64), got %d", r.run_R);
    EXPECT(r.kind_in_want == kUnit && r.kind_in_run == kUnit && s.kind == kLayer, "kind during want/run, reset after");
    EXPECT(s.depth == 0 && r.parks == 1 && r.unparks == 1 && s.st.yields == 1 && s.st.intra_yields == 1, "counters");
    EXPECT(s.last == 110 && s.seg.t0 == 110 && s.cost_ms == 10, "resume: clock, segment restart and yield cost");
  }
  {  // exception in run: unpark once, state restored, exception propagates
    Rec r; State s = make_state(r, 50); r.t = 100; r.run_throws = true;
    bool threw = false;
    try { point(s, r, kIndex, 2); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw && r.unparks == 1 && s.depth == 0 && s.kind == kLayer && s.last == 110, "exception: unparked once, depth/kind/clock restored");
    r.run_throws = false; r.t = 200;
    EXPECT(point(s, r, kIndex, 2) && r.unparks == 2, "next point works after the exception");
  }
  {  // look-ahead
    Rec r; State s = make_state(r, 150); s.cap_ms = 200;
    r.t = 0; EXPECT(!point(s, r, kMoe, 5), "t=0: no");
    r.t = 500; EXPECT(point(s, r, kUnit, 6), "t=500 (period) — the segment 0→500 is attributed to key 5");  // resume at 510
    EXPECT(s.seg.next(5) == 500, "segment of key 5 = 500 ms");
    r.t = 520; EXPECT(point(s, r, kMoe, 5), "10 ms after the resume (= cost 10) with a 500 ms segment next: look-ahead yield");
  }
  {  // the cost bound: right after a resume with a long next segment, a yield only if the stall it removes ≥ the yield cost
    Rec r; State s = make_state(r, 150); s.cap_ms = 200; r.run_ms = 40;
    r.t = 0; point(s, r, kMoe, 5);
    r.t = 500; point(s, r, kUnit, 6);   // yields (period); cost 40; resume 540; key-5 sample = 500
    r.t = 560; EXPECT(!point(s, r, kMoe, 5), "20 ms after the resume, cost 40: no look-ahead yield");
    r.t = 650; EXPECT(!point(s, r, kUnit, 6), "110 ms after the resume, key 6's segment (20 ms) ends within the cap: no");
    r.t = 760; EXPECT(point(s, r, kMoe, 5), "220 ms after the resume ≥ period");
    Rec r2; State s2 = make_state(r2, 1000); s2.cap_ms = 200; r2.run_ms = 5;
    r2.t = 0; point(s2, r2, kMoe, 5);
    r2.t = 300; point(s2, r2, kUnit, 6);  // key 5 sample = 300; no yield (period 1000, key 6 unsampled)
    r2.t = 350; EXPECT(point(s2, r2, kMoe, 5), "look-ahead: 350 + 300 > 200 (before the 1000 ms period)");
    EXPECT(r2.runs == 1, "exactly one yield");
  }
  {  // want 0 = nothing; park failure = skipped, clock restarts
    Rec r; State s = make_state(r, 50); r.t = 100; r.want_ret = 0;
    EXPECT(!point(s, r, kUnit, 1) && r.parks == 0 && s.kind == kLayer, "want 0: no park, kind reset");
    r.want_ret = 1; r.park_ok = false;
    EXPECT(!point(s, r, kUnit, 1) && s.st.skipped_park == 1 && s.last == 100 && s.depth == 0, "park failure absorbed");
  }
  {  // the boundary yield_point never touches kind (hived admits there)
    Rec r; State s = make_state(r, 50); r.t = 100; int k_run = -1;
    s.h.run = [&](int, double) { k_run = s.kind; };
    EXPECT(yield_point(s, [&](int) { return true; }, [&](int) {}, [&] { return r.t; }) && k_run == kLayer, "boundary: kind stays kLayer");
  }
  printf("layer-yield intra CPU: yield_point_intra (off, decode-only rows, kind, exceptions, look-ahead, cost bound, park failure) ok\n");
}

// ---- real ExpertStore ----------------------------------------------------------------------------------------------------------------------------
World& world() { static World w; return w; }

void test_store_hold() {
  World& w = world();
  Checkpoint ck("fake");
  ExpertStore st(w.cfg, L, 0, 1, 0);
  for (int l = 0; l < L; ++l) st.load_layer_experts(ck, l, 2);
  const size_t total = st.layout().total;
  std::vector<uint8_t> ref(total);
  st.copy_rec_async(ref.data(), 0, 1, nullptr);
  EXPECT(!st.staging_held(), "hold off by default");
  st.copy_to_staging(0, 0, 1, nullptr);
  EXPECT(std::memcmp(st.staging_rec(0), ref.data(), total) == 0, "unheld copy_to_staging = record");
  st.set_staging_hold(true);
  EXPECT(st.staging_held(), "hold set");
  fflush(stdout); fflush(stderr);
  const pid_t pid = fork();
  if (pid == 0) {  // child: a writer that ignores the hold
    st.copy_to_staging(0, 1, 2, nullptr);
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  EXPECT(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "copy_to_staging while held must abort (status %d)", status);
  st.set_staging_hold(false);
  EXPECT(std::memcmp(st.staging_rec(0), ref.data(), total) == 0, "parent's staging record untouched");
  printf("layer-yield intra CPU: real ExpertStore staging hold (default off, unheld copy, held copy aborts) ok\n");
}

// Negative-control mode (Python builds it with a mutant expert_store.cpp): a held copy_to_staging overwrites a "pre-copied" record.
int staging_overwrite() {
  World& w = world();
  Checkpoint ck("fake");
  ExpertStore st(w.cfg, L, 0, 1, 0);
  for (int l = 0; l < L; ++l) st.load_layer_experts(ck, l, 2);
  const size_t total = st.layout().total;
  st.copy_to_staging(0, 0, 1, nullptr);  // the paused prefill's pre-copy
  std::vector<uint8_t> pre(st.staging_rec(0), st.staging_rec(0) + total);
  st.set_staging_hold(true);
  st.copy_to_staging(0, 1, 2, nullptr);  // a decode writer that ignored the hold
  const bool over = std::memcmp(st.staging_rec(0), pre.data(), total) != 0;
  printf("%s\n", over ? "OVERWRITTEN" : "INTACT");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--staging-overwrite") return staging_overwrite();
  test_parse();
  test_due();
  test_point();
  test_store_hold();
  if (fails) { fprintf(stderr, "layer-yield intra CPU: %d failures\n", fails); return 1; }
  printf("layer-yield intra CPU: all passed\n");
  return 0;
}
