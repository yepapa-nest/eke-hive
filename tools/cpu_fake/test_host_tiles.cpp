// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// H2/H3 host-side machinery on CPU: PRODUCTION text of runtime.cpp (tile_plan · host_tile_count · plan_units ·
// HostStager incl. run_pass) sliced by harness.host_tiles_source() into host_tiles_gen.h, compiled against fake CUDA.
// With FAKE_CUDA_ASYNC=1 every stream is a real worker thread and events order them exactly like the production calls, so a
// missing wait shows up as a wrong value (and as a TSAN report under HIVE_TEST_SANITIZER=thread). tools/test_daemon_cpu.py
// also builds mutants of the stager (each wait removed) that must fail here.
// NOT covered (GPU only): the forward_multi kernels, real copy-engine overlap, VRAM accounting, pinned-memory limits.
#include "host_tiles_gen.h"

#include <cstdio>
#include <random>
#include <vector>

using namespace hive;
static int fails = 0, checks = 0;
#define EXPECT(c, ...) do { ++checks; if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// Reference: the original C2 tile formula as it stood inline in Runtime::forward, for M > Wm.
static std::vector<int> ref_plan(int M, int Wm, int need) {
  std::vector<int> tM;
  if (M <= Wm) return {M};
  const int T = (M + Wm - 1) / Wm;
  int last = M - (T - 1) * ((M + T - 1) / T);
  if (last < need) last = std::min(Wm, need);
  const int rest = M - last, Ms = T > 1 ? (rest + T - 2) / (T - 1) : 0;
  for (int s = 0, off = 0; s < T; ++s) { const int n = s < T - 1 ? std::min(Ms, rest - off) : last; tM.push_back(n); off += n; }
  return tM;
}

static void test_tile_plan() {
  std::vector<int> got;
  long valid = 0;
  for (int Wm : {256, 1536, 2048, 16384})
    for (int need : {1, 129, 1101, 1025, 2689})
      for (int M = 1; M <= Wm * 7; M += (Wm >= 16384 ? 97 : 1)) {
        tile_plan(M, Wm, need, got);
        const auto ref = ref_plan(M, Wm, need);
        if (got != ref) { EXPECT(false, "tile_plan(%d,%d,%d) differs from the pre-refactor formula", M, Wm, need); return; }
        const int T = M <= Wm ? 1 : (M + Wm - 1) / Wm;
        EXPECT((int)got.size() == T, "tile_plan count %zu != ceil(%d/%d)", got.size(), M, Wm);
        bool ok = true; int sum = 0;
        for (int n : got) { ok = ok && n >= 1 && n <= Wm; sum += n; }
        if (ok && need <= Wm) {  // a plan the production checks accept: covers M exactly, last sub-chunk >= need when tiled
          ++valid;
          EXPECT(sum == M, "tile_plan(%d,%d,%d) sum %d", M, Wm, need, sum);
          if (T > 1) EXPECT(got.back() >= std::min(Wm, need), "tile_plan last %d < need %d", got.back(), need);
        }
      }
  EXPECT(valid > 10000, "tile_plan: too few valid plans checked (%ld)", valid);
}

static void test_host_tile_count() {
  const size_t hb = (size_t)16384 * 4 * 5120 * 2;  // real shape: 640 MiB per 16K sub-chunk
  EXPECT(host_tile_count(false, nullptr, hb, true, 3, 16384, 262144) == 0, "off");
  EXPECT(host_tile_count(true, nullptr, hb, true, 3, 16384, 262144) == 2, "unset MB = 2 tiles");
  EXPECT(host_tile_count(true, "", hb, true, 3, 16384, 262144) == 2, "empty MB = 2 tiles");
  EXPECT(host_tile_count(true, "0", hb, true, 3, 16384, 262144) == 0, "0 MB = 0");
  EXPECT(host_tile_count(true, "-5", hb, true, 3, 16384, 262144) == 0, "negative = 0 (absorbed)");
  EXPECT(host_tile_count(true, "abc", hb, true, 3, 16384, 262144) == 0, "non-numeric = 0 (absorbed)");
  EXPECT(host_tile_count(true, "1280", hb, true, 3, 16384, 262144) == 2, "1280 MiB = 2 x 640 MiB");
  EXPECT(host_tile_count(true, "1279.5", hb, true, 3, 16384, 262144) == 1, "1279.5 MiB < 2 x 640 MiB");
  EXPECT(host_tile_count(true, "100000", hb, true, 3, 16384, 262144) == 13, "capped at max_ctx / max_chunk - resident (16 - 3)");
  EXPECT(host_tile_count(true, "100000", hb, true, 3, 16384, 49152) == 0, "no host tile when resident slots already cover max_ctx");
  EXPECT(host_tile_count(true, nullptr, hb, false, 3, 16384, 262144) == 0, "tiles impossible (no tail mode) = 0");
  EXPECT(host_tile_count(true, "inf", hb, true, 3, 16384, 262144) == 0, "inf = 0 (absorbed)");
}

static void test_plan_units() {
  std::vector<MultiUnitPlan> u;
  auto tail_ok = [](int m) { return m >= 1025 && m > 1100; };
  EXPECT(plan_units({1536, 1200, 300}, 1536, 1101, 3, tail_ok, u) && u.size() == 3, "three single-sub parts in 3 slots");
  EXPECT(u.size() == 3 && u[0].part == 0 && u[1].part == 1 && u[2].part == 2 && u[2].slot == 2 && u[0].last && u[1].last && u[2].last, "order/slots/last");
  EXPECT(!plan_units({1536, 1200, 300}, 1536, 1101, 2, tail_ok, u), "slot overflow rejected");
  EXPECT(plan_units({4000, 1200}, 1536, 1101, 4, tail_ok, u) && u.size() == 4, "multi-sub part + single");
  EXPECT(u.size() == 4 && u[0].part == 0 && u[2].part == 0 && u[2].last && !u[0].last && u[3].part == 1 && u[3].slot == 3, "multi-sub order");
  int off = 0; bool offs = true;
  for (int s = 0; s < 3; ++s) { offs = offs && u[(size_t)s].off == off; off += u[(size_t)s].M; }
  EXPECT(offs && off == 4000, "multi-sub offsets cover the chunk");
  EXPECT(!plan_units({4000}, 1536, 1101, 8, [](int) { return false; }, u), "multi-sub part without tail mode rejected");
  EXPECT(!plan_units({0}, 1536, 1101, 8, tail_ok, u), "empty part rejected");
}

// ---- HostStager: simulated layer-first forwards over resident + host units --------------------------------------------------
// compute(u, l) on the compute stream: every byte b[i] = b[i]*31 + (l*7 + u*13 + i) — order-sensitive, so any stale or early copy changes the result.
static void compute(cudaStream_t st, uint8_t* p, size_t n, int u, int l) {
  fakecuda::enqueue(st, [p, n, u, l] {
    if (fakecuda::delay_us() > 0) std::this_thread::sleep_for(std::chrono::microseconds(fakecuda::delay_us() / 2));
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)(p[i] * 31u + (uint32_t)(l * 7 + u * 13) + (uint32_t)i);
  });
}
static void init(cudaStream_t st, uint8_t* p, size_t n, int u, int f) {
  fakecuda::enqueue(st, [p, n, u, f] { for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)(u * 5 + f * 3 + i * 11); });
}

static void test_stager(int seed, int R, int H, int n_stage, int L, int forwards) {
  std::mt19937 rng((unsigned)seed);
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const size_t rowb = 24, rows_max = 64, bytes = rows_max * rowb;
  {
    HostStager hs(st, H, bytes, n_stage);
    EXPECT(hs.n_host() == H && hs.n_stage() == std::max(1, std::min(n_stage, H)), "stager shape");
    const int U = R + H;
    std::vector<DevBuf> resident((size_t)R);
    for (auto& b : resident) b.alloc(bytes);
    std::vector<DevBuf> slot_h((size_t)H);  // TileSlot.h of the host slots — empty until a visit swaps the stage buffer in
    for (int f = 0; f < forwards; ++f) {
      hs.begin();
      std::vector<size_t> rows((size_t)U);
      for (auto& r : rows) r = 8 + rng() % (rows_max - 8);
      std::vector<std::vector<uint8_t>> ref((size_t)U);
      std::vector<int> up;  // units that continue past the "tail layer" with fewer rows (like member-last units)
      for (int u = 0; u < U; ++u) if (rng() % 2 || u == U - 1) up.push_back(u);
      std::vector<std::vector<uint8_t>> out((size_t)U);
      for (int u = 0; u < U; ++u) { ref[(size_t)u].resize(bytes); for (size_t i = 0; i < rows[(size_t)u] * rowb; ++i) ref[(size_t)u][i] = (uint8_t)(u * 5 + f * 3 + i * 11); }
      auto cpu_compute = [&](int u, size_t n, int l) { for (size_t i = 0; i < n; ++i) ref[(size_t)u][i] = (uint8_t)(ref[(size_t)u][i] * 31u + (uint32_t)(l * 7 + u * 13) + (uint32_t)i); };
      const int T = L / 2;  // "tail layer": units in `up` shrink to 1/2..all of their rows, the rest stop (dead h)
      std::vector<size_t> rows2 = rows;
      for (int u : up) rows2[(size_t)u] = 1 + rng() % rows[(size_t)u];
      auto run = [&](const std::vector<int>& us, int l, bool fresh, int phase) {  // phase 0 = before T, 1 = T, 2 = after T, 3 = final
        std::vector<HostStager::Item> items(us.size());
        for (size_t i = 0; i < us.size(); ++i) {
          const int u = us[i];
          items[i].k = u >= R ? u - R : -1;
          const size_t r = phase >= 2 ? rows2[(size_t)u] : rows[(size_t)u];
          items[i].load = items[i].k >= 0 && !fresh ? r * rowb : 0;
        }
        hs.run_pass(items, [&](size_t i, DevBuf* stage) {
          const int u = us[i];
          if (stage) std::swap(slot_h[(size_t)(u - R)], *stage);
          uint8_t* p = (u < R ? resident[(size_t)u] : slot_h[(size_t)(u - R)]).as<uint8_t>();
          const size_t r = phase >= 2 ? rows2[(size_t)u] : rows[(size_t)u];
          if (fresh) init(st, p, r * rowb, u, f);
          compute(st, p, r * rowb, u, l);
          cpu_compute(u, r * rowb, l);
          const bool is_up = std::find(up.begin(), up.end(), u) != up.end();
          size_t store = r * rowb;
          if (phase == 1) {  // tail layer: shrink (rows move to the front like prepare_decoder_tail) or die
            store = is_up ? rows2[(size_t)u] * rowb : 0;
          }
          if (phase == 3) {  // head: read the result on the compute stream, h dead afterwards
            out[(size_t)u].resize(r * rowb);
            uint8_t* dst = out[(size_t)u].data();
            fakecuda::enqueue(st, [dst, p, n = r * rowb] { memcpy(dst, p, n); });
            store = 0;
          }
          if (stage) std::swap(slot_h[(size_t)(u - R)], *stage);
          items[i].store = u >= R ? store : 0;
        });
      };
      std::vector<int> all((size_t)U);
      for (int u = 0; u < U; ++u) all[(size_t)u] = u;
      for (int l = 0; l < T; ++l) run(all, l, l == 0, 0);
      run(all, T, T == 0, 1);
      for (int l = T + 1; l < L; ++l) run(up, l, false, 2);
      run(up, L, false, 3);
      CUDA_CHECK(cudaStreamSynchronize(st));
      hs.sync();
      for (int u : up) {
        const size_t n = rows2[(size_t)u] * rowb;
        EXPECT(out[(size_t)u].size() == n && std::equal(out[(size_t)u].begin(), out[(size_t)u].end(), ref[(size_t)u].begin()),
               "seed %d R %d H %d stages %d forward %d unit %d (%s): result differs", seed, R, H, n_stage, f, u, u >= R ? "host" : "resident");
      }
      for (int k = 0; k < H; ++k) EXPECT(!hs.staged(k) && slot_h[(size_t)k].p == nullptr, "stage buffer returned");
    }
  }
  CUDA_CHECK(cudaStreamDestroy(st));
}

int main(int argc, char** argv) {
  const bool stress_only = argc > 1 && std::string(argv[1]) == "stress";
  if (!stress_only) { test_tile_plan(); test_host_tile_count(); test_plan_units(); }
  int seed = 1;
  for (int R : {1, 2, 3})
    for (int H : {1, 2, 3, 5})
      for (int n_stage : {1, 2})
        test_stager(seed++, R, H, n_stage, 6, 3);
  printf("host tiles CPU: %d checks, %d failures (fake CUDA async=%d)\n", checks, fails, (int)fakecuda::async_mode());
  return fails ? 1 : 0;
}
