// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU-only check of the PRODUCTION Runtime::promote_after_step (text sliced from engine/src/runtime.cpp by
// tools/cpu_fake/harness.py) on the REAL ExpertStore (engine/src/expert_store.cpp) with fake CUDA. Built/run by
// tools/test_snapshot_cpu.py. No GPU.
//
// M7 (HIVE_PREFILL_PAUSE_PROMOTE): while hived marks a multi-chunk prefill (set_prefill_pending(true)) and the
// switch is on, promote_after_step issues NO promotions but still decays scores and still clears the step/draft miss
// lists, so the store state equals the unpaused path minus the promotions. Switch off (unset/""/"0") = the mark has no effect.
// argv[1] = expected mode: "on" (switch set by the python driver) or "off".
// Both promotion policies are covered: score promotion (promote_per_token) and miss promotion (promote_misses, step_miss_).
// Every header runtime.h pulls in is included BEFORE the access override below (the override must only touch hive's classes).
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>
// step_miss_/draft_miss_ are private inputs filled by the GPU decode path; this TU reaches them directly (same layout).
#define private public
#include "hive/runtime.h"
#undef private

using namespace hive;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

namespace {
struct Rig {
  Model model{"fake", -1, 0};
  std::unique_ptr<ExpertStore> store;
  std::unique_ptr<Runtime> rt;
  Rig(const RuntimeOptions& opt) {
    const size_t total = ExpertLayout::make(model.cfg().dim, model.cfg().moe_inter).total;
    store = std::make_unique<ExpertStore>(model.cfg(), model.cfg().n_layers, 4 * total, 1, 0);
    for (int l = 0; l < model.cfg().n_layers; ++l) store->load_layer_experts(model.ckpt(), l, 2);
    rt = std::make_unique<Runtime>(model, *store, nullptr, opt);
  }
  // deterministic usage history: key k observed (k % 5 + 1) · 3 times
  void seed() {
    const int E = model.cfg().n_routed;
    for (int l = 0; l < model.cfg().n_layers; ++l)
      for (int e = 0; e < E; ++e)
        for (int t = 0; t < ((l * E + e) % 5 + 1) * 3; ++t) store->observe(l, e);
  }
  std::vector<float> scores() const {
    std::vector<float> v;
    for (int k = 0; k < model.cfg().n_layers * model.cfg().n_routed; ++k) v.push_back(store->score_of(k));
    return v;
  }
  std::vector<int> mapping() const {
    std::vector<int> v;
    for (int l = 0; l < model.cfg().n_layers; ++l) for (int e = 0; e < model.cfg().n_routed; ++e) v.push_back(store->slot_of(l, e));
    return v;
  }
  void misses() { rt->step_miss_ = {3, 5, 9, 3}; rt->draft_miss_ = {11}; }
};

void run(const char* policy, bool expect_pause) {
  RuntimeOptions opt;
  opt.max_ctx = 256;
  if (!strcmp(policy, "score")) { opt.promote_per_token = 2; opt.promote_misses = 0; opt.promote_min_score = 1.f; }
  else { opt.promote_per_token = 0; opt.promote_misses = 2; opt.mtp_cache = true; }
  Rig paused(opt), ref(opt);
  paused.seed(); ref.seed();
  paused.rt->set_prefill_pending(true);  // hived: multi-chunk prefill in progress
  EXPECT(paused.rt->prefill_pending(), "contract getter");
  int n_paused = 0, n_ref = 0;
  for (int step = 0; step < 3; ++step) {
    paused.misses(); ref.misses();
    n_paused += paused.rt->promote_after_step();
    n_ref += ref.rt->promote_after_step();
    EXPECT(paused.rt->step_miss_.empty() && paused.rt->draft_miss_.empty(), "[%s] miss lists not cleared while paused", policy);
    EXPECT(ref.rt->step_miss_.empty() && ref.rt->draft_miss_.empty(), "[%s] miss lists not cleared", policy);
    // promotions never change scores: the decay is identical on both paths
    EXPECT(paused.scores() == ref.scores(), "[%s] step %d: scores differ between paused and unpaused paths", policy, step);
  }
  paused.store->commit_all(); ref.store->commit_all();
  EXPECT(n_ref > 0, "[%s] reference path issued no promotion (fixture too weak)", policy);
  if (expect_pause) {
    EXPECT(n_paused == 0 && paused.store->cache_stats().promotions == 0, "[%s] paused path issued %d promotions", policy, n_paused);
    EXPECT(paused.store->n_resident() == 0 && paused.store->n_pending() == 0, "[%s] paused path changed residency", policy);
  } else {
    EXPECT(n_paused == n_ref && paused.mapping() == ref.mapping(), "[%s] switch off: prefill mark changed promotions (%d vs %d)", policy, n_paused, n_ref);
  }
  // prefill done: promotions resume and the next step behaves like a normal step on the same (decayed) scores
  paused.rt->set_prefill_pending(false);
  paused.misses();
  const int after = paused.rt->promote_after_step();
  paused.store->commit_all();
  EXPECT(after > 0, "[%s] no promotion after the prefill mark was cleared", policy);
  printf("runtime host CPU: M7 [%s] pause %s: paused %d · unpaused %d · after clear %d · scores identical\n", policy,
         expect_pause ? "on" : "off", n_paused, n_ref, after);
}
}  // namespace

int main(int argc, char** argv) {
  const bool on = argc > 1 && !strcmp(argv[1], "on");
  run("score", on);
  run("miss", on);
  return fails ? 1 : 0;
}
