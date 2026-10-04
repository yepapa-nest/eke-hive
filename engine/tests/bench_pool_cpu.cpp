// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Measures one decode layer's worth of CPU misses on the real ExpertStore CPU pool (engine/src/expert_store.cpp) at real shapes (dim 5120, inter 2304).
//   Fake CUDA/NUMA (tools/cpu_fake) — tools/bench_pool_cpu.py builds it and runs it twice with HIVE_CPU_GEMV2=0/1 for comparison (the switch is read once per process).
//   Unit = one start_jobs -> wait_jobs (= the CPU share of one moe_decode_experts layer), job counts 1/2/3/6 (R=1, M=1 decode), and 2 jobs with R=4 (batch).
//   Experts rotate over E experts of one layer (working set > L3 — read from DRAM). The first batch's output is bit-compared against expert_forward.
//   Note: the build machine's CPU/memory may differ from the deployment machine (e.g. Zen2 3975WX, 2 nodes, 16 threads per node) — numbers are for relative comparison.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hive/clock.h"
#include "hive/expert_cpu.h"
#include "hive/expert_store.h"
#include "hive/safetensors.h"

namespace hive { void fake_ckpt_put(const std::string&, const uint8_t*, size_t, const std::string& = "U8"); }
using namespace hive;

int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::max(1, atoi(argv[1])) : 8;   // per node (2 fake nodes -> twice the total)
  const int E = argc > 2 ? std::max(4, atoi(argv[2])) : 16;
  const int iters = argc > 3 ? std::max(1, atoi(argv[3])) : 200;
  constexpr int DIM = 5120, INTER = 2304;
  Config cfg;
  cfg.dim = DIM; cfg.moe_inter = INTER; cfg.n_routed = E; cfg.dspark_experts = std::min(E, 4); cfg.swiglu_limit = 10.f;
  std::mt19937 rng(20260930);
  std::uniform_int_distribution<int> byte(0, 255);
  const size_t wsz = (size_t)INTER * DIM / 2, ssz = (size_t)INTER * DIM / 32;
  std::vector<std::vector<uint8_t>> blobs;
  blobs.reserve((size_t)E * 6);
  for (int e = 0; e < E; ++e)
    for (const char* m : {"w1", "w2", "w3"}) {
      blobs.emplace_back(wsz); for (auto& v : blobs.back()) v = (uint8_t)byte(rng);
      const std::string p = "layers.0.ffn.experts." + std::to_string(e) + "." + m;
      fake_ckpt_put(p + ".weight", blobs.back().data(), wsz);
      blobs.emplace_back(ssz); for (auto& v : blobs.back()) v = (uint8_t)(117 + byte(rng) % 6);
      fake_ckpt_put(p + ".scale", blobs.back().data(), ssz);
    }
  Checkpoint ck("fake");
  ExpertStore st(cfg, 1, 0, threads, 0);
  st.load_layer_experts(ck, 0, 4);
  auto e4m3 = [&] { uint8_t v; do { v = (uint8_t)byte(rng); } while ((v & 0x7F) == 0x7F || (v & 0x78) > 0x50); return v; };
  constexpr int NA = 8;
  std::vector<uint8_t> aq[NA], as[NA];
  std::vector<float> af[NA], saf[NA];
  for (int a = 0; a < NA; ++a) {
    aq[a].resize(DIM); as[a].resize(DIM / 32); af[a].resize(DIM); saf[a].resize(DIM / 32);
    for (auto& v : aq[a]) v = e4m3();
    for (auto& v : as[a]) v = (uint8_t)(127 - 6 + byte(rng) % 4);
    for (int k = 0; k < DIM; ++k) af[a][k] = e4m3_to_f32(aq[a][k]);
    for (int b = 0; b < DIM / 32; ++b) saf[a][b] = e8m0_to_f32(as[a][b]);
  }
  const size_t per = ExpertStore::job_scratch_floats(INTER);
  std::vector<float> scratch(per * ExpertStore::kMaxRows * 8), outs((size_t)DIM * ExpertStore::kMaxRows * 8);
  int bad = 0, checked = 0;
  auto sw_on = [](const char* n) { const char* v = getenv(n); return v && *v && strcmp(v, "0") != 0; };
  printf("bench_pool_cpu: [B3] MULTIROW2 min rows %d (0 = off) · FINE %s\n", cpu::multirow2_min_rows(), sw_on("HIVE_CPU_FINE") ? "on" : "off");
  printf("bench_pool_cpu: HIVE_CPU_GEMV2 → %s · SPLIT13 %s · P2_PREFETCH %s · %d threads per node x nodes %s · experts %d x 18.8 MB · reps %d\n", cpu::gemv2_enabled() ? "_v2" : "_ref (baseline)", getenv("HIVE_CPU_SPLIT13") && *getenv("HIVE_CPU_SPLIT13") && strcmp(getenv("HIVE_CPU_SPLIT13"), "0") ? "on" : "off",
         getenv("HIVE_CPU_P2_PREFETCH") && *getenv("HIVE_CPU_P2_PREFETCH") && strcmp(getenv("HIVE_CPU_P2_PREFETCH"), "0") ? "on" : "off", threads,
         getenv("FAKE_NUMA_NODES") && atoi(getenv("FAKE_NUMA_NODES")) == 1 ? "1 (fake)" : "2 (fake — same memory)", E, iters);
  // same = all jobs use the same expert (as when a batch's R rows calling one expert are split into "one job per row" — the weights are re-read per job).
  //   moe_decode_experts groups rows of the same expert into one job (R <= kMaxRows) — {1,R} vs {R,1,same} is the value of that grouping.
  struct Case { int nj, R; bool same = false; };
  for (const Case cs : {Case{1, 1}, Case{2, 1}, Case{3, 1}, Case{6, 1}, Case{2, 4}, Case{1, 2}, Case{1, 4}, Case{4, 1, true}, Case{1, 8}, Case{8, 1, true}, Case{2, 8}, Case{3, 4}}) {
    std::vector<double> ms;
    int pick = 0;
    for (int it = 0; it < iters + 3; ++it) {
      std::vector<ExpertStore::Job> jobs((size_t)cs.nj);
      for (int j = 0; j < cs.nj; ++j) {
        auto& J = jobs[(size_t)j];
        J.layer = 0; J.e = cs.same && j > 0 ? jobs[0].e : (pick++) % E; J.R = cs.R;
        J.scratch = scratch.data() + (size_t)j * per * ExpertStore::kMaxRows;
        for (int r = 0; r < cs.R; ++r) {
          const int a = (cs.same ? j : j + r) % NA;
          J.a_f[r] = af[a].data(); J.a_s[r] = saf[a].data(); J.route_w[r] = 0.2f + 0.1f * r;
          J.out[r] = outs.data() + ((size_t)j * ExpertStore::kMaxRows + r) * DIM;
        }
      }
      const auto t0 = hive::SteadyClock::now();
      st.start_jobs(jobs);
      st.wait_jobs();
      const double t = hive::ms_since(t0);
      if (it >= 3) ms.push_back(t);  // first 3 iterations are warm-up
      if (it == 0) {  // bit comparison against expert_forward (same switch entry point)
        std::vector<float> ref(DIM), tmp(4 * INTER + 2 * DIM + 64);
        for (int j = 0; j < cs.nj; ++j) {
          const auto& J = jobs[(size_t)j];
          const size_t base = (size_t)J.e * 6;
          cpu::ExpertDesc d{cpu::ExpertFormat::E2M1_B32, DIM, INTER, blobs[base].data(), blobs[base + 1].data(), blobs[base + 2].data(),
                            blobs[base + 3].data(), blobs[base + 4].data(), blobs[base + 5].data()};
          for (int r = 0; r < cs.R; ++r) {
            const int a = (cs.same ? j : j + r) % NA;
            cpu::expert_forward(d, aq[a].data(), as[a].data(), J.route_w[r], cfg.swiglu_limit, ref.data(), tmp.data());
            bad += memcmp(ref.data(), J.out[r], DIM * 4) != 0; ++checked;
          }
        }
      }
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2], p10 = ms[ms.size() / 10], p90 = ms[ms.size() * 9 / 10];
    printf("  jobs %d x R=%d%s: median %.3f ms (p10 %.3f · p90 %.3f) · per job %.3f ms\n", cs.nj, cs.R, cs.same ? " (same expert)" : "", med, p10, p90, med / cs.nj);
  }
  printf("bench_pool_cpu: expert_forward bit comparison: mismatches %d/%d %s\n", bad, checked, bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
