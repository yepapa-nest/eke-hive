// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_VRAM_LIMIT_MB — run a GPU daemon as if the card had this much memory (docs/vram-footprint.md §4.2, docs/configuration.md).
//
// Why: the RTX 50 / RTX PRO Blackwell cards hive builds for range from 8 to 96 GB. The limit lets one card stand in for a smaller one, counting
//   everything the process holds: CUDA context, weights, work buffers, KV, CUDA graphs, cuBLAS and allocator granularity.
// How (hived.cpp, GPU families only; unset = nothing in this header runs):
//   1. Context: NVML's free memory before the CUDA context exists (F0) minus cudaMemGetInfo's free right after it is created (F1). NVML is
//      dlopen'ed (hive links no NVML); without it the context is taken as kCtxFallback (an estimate — the 984 MiB a GLM daemon kept at sleep
//      level 3, 2026-10-05, docs/glm.md, which includes the context; the log line says which one was used).
//   2. Weights: loaded as usual, then measured (free before − free after the model).
//   3. Plan (plan() below): the rest of the startup allocations is estimated from per-row / per-sequence costs (docs/vram-footprint.md §1.3,
//      derived from the allocation code and checked against the measured 96 GB startups: DS 4,452 vs 4,447 slots, GLM 4,330 vs 4,323). If it
//      does not fit under the limit with HIVE_CACHE_RESERVE_MB beside it, the knobs that do not change outputs are lowered in a fixed order.
//      If even the lowest setting does not fit, the shortfall and the number of layers that would have to leave the GPU are printed and hived
//      exits with code 2 — the same point where a real card of that size would abort in a cudaMalloc, with the reason spelled out.
//   4. Ballast: one cudaMalloc of (free − (limit − context − weights)), held for the life of the process. Every later sizing decision
//      (HIVE_CACHE_FIT, the reserve, KV growth, host tiles, vision, wake) then sees the smaller card without any change in the families.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(HIVE_CPU_ONLY) && !defined(HIVE_FAKE_CUDA)
#include <dlfcn.h>
#endif

namespace hive::vram_limit {

constexpr size_t kMiB = 1ull << 20;
constexpr size_t kCtxFallback = 984 * kMiB;

// HIVE_VRAM_LIMIT_MB: unset, empty, non-numeric or <= 0 = off (0 bytes)
inline size_t limit_bytes() {
  const char* v = getenv("HIVE_VRAM_LIMIT_MB");
  if (!v || !*v) return 0;
  char* end = nullptr;
  const double x = strtod(v, &end);
  if (end == v || !end || *end != 0 || !std::isfinite(x) || x <= 0) return 0;
  return (size_t)(std::min(x, 1e9) * (double)kMiB);
}

#if !defined(HIVE_CPU_ONLY) && !defined(HIVE_FAKE_CUDA)
// Free device memory from NVML (no CUDA context needed). pci = "dddddddd:bb:dd.0". Returns false when NVML is not available.
inline bool nvml_free(const char* pci, size_t* free_bytes) {
  struct Mem { unsigned long long total, free, used; };
  void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) return false;
  using InitF = int (*)();
  using ByPciF = int (*)(const char*, void**);
  using MemF = int (*)(void*, Mem*);
  auto init = (InitF)dlsym(h, "nvmlInit_v2");
  auto shut = (InitF)dlsym(h, "nvmlShutdown");
  auto by_pci = (ByPciF)dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
  auto mem = (MemF)dlsym(h, "nvmlDeviceGetMemoryInfo");
  bool ok = false;
  if (init && shut && by_pci && mem && init() == 0) {
    void* dev = nullptr;
    Mem m{};
    if (by_pci(pci, &dev) == 0 && mem(dev, &m) == 0) { *free_bytes = (size_t)m.free; ok = true; }
    shut();
  }
  dlclose(h);
  return ok;
}
#endif

// ---- startup plan ----------------------------------------------------------------------------------------------------------------------
enum class Family { None, DeepSeek, Glm, Qwen };
inline const char* family_name(Family f) { return f == Family::DeepSeek ? "deepseek" : f == Family::Glm ? "glm" : f == Family::Qwen ? "qwen" : "none"; }

// The knobs the plan may lower (all output-neutral: chunk boundaries and tiles process the same tokens in the same causal order; staging,
//   host tiles and verify parts only change how much is prepared at once; batch and sessions only change how many requests run together).
struct Knobs {
  int max_chunk = 4096, prefill_tile = 1, host_tiles = 0, staging = 8, max_batch = 8, max_sessions = 2;
  bool mtp = false, mtp_batch = false;
  int64_t max_ctx = 262144;
  int64_t kv_base = -1;  // GLM / Qwen: positions mapped per sequence at creation (HIVE_*_KV_BASE; < 0 = max_chunk)
};

// Startup VRAM after the weights, by family (bytes) — estimates, see docs/vram-footprint.md §1.2–1.3 for where each constant comes from.
struct Costs {
  double row = 0;          // work buffers per --max-chunk row
  double tile_row = 0;     // DS: per row per extra VRAM tile slot (--prefill-tile N: N−1 slots)
  double host_row = 0;     // DS: device part per row per host tile
  double host_stage_row = 0;  // DS: h staging per row, at most 2 buffers
  double fixed = 0;        // small buffers, cuBLAS workspaces, indexer, logits, process residual
  double record = 0;       // one expert record (the staging ring holds `staging` of them)
  double snap_part = 0;    // MTP verify snapshot part (GLM / Qwen; parts = max_batch with HIVE_MTP_BATCH)
  double seq_fixed = 0;    // per sequence: recurrent states / windows
  double seq_pos = 0;      // per sequence position (DS: preallocated for max_ctx; GLM / Qwen: kv_base positions at creation)
};

inline Costs costs(Family f, bool mtp) {
  Costs c;
  switch (f) {
    case Family::DeepSeek:  // Work 352,168 B/row (sum of Runtime::Runtime allocations; reproduces the measured 10,335 MiB prefill-only total)
      c.row = 352168; c.tile_row = 69392; c.host_row = 28432; c.host_stage_row = 40960;
      c.fixed = 1083.0 * kMiB;  // measured 2026-10-09 (runtime VRAM 2,804 MiB − staging 1,721 MiB, config/hive.env)
      c.record = 18800640; c.seq_fixed = 5.1 * kMiB; c.seq_pos = 979;  // 250 MiB per sequence at 262,144 measured
      break;
    case Family::Glm:  // GlmEngine work 426,000 B/row with MTP (65,536 of them MTP rows)
      c.row = mtp ? 426000 : 360464;
      c.fixed = (128 + 144 + 38 + 128 + 1300) * (double)kMiB;  // deq, attention ws, logits, 2 cuBLAS ws, residual (GLM 1,824 MiB measured incl. context, ~0.5 GiB of it context)
      c.record = 13.5 * kMiB;
      c.snap_part = mtp ? 437.0 * kMiB : 0;  // measured: 8 parts 3,494 MiB
      c.seq_fixed = 146 * kMiB; c.seq_pos = 19200;  // 456 MiB per sequence at the 16K base measured
      break;
    case Family::Qwen:  // QwenEngine work 256,152 B/row with MTP (78,800 of them MTP rows) — not run on a GPU yet
      c.row = mtp ? 256152 : 179352;
      c.fixed = (61 + 100 + 128 + 1300) * (double)kMiB;
      c.record = 2.64 * kMiB;
      c.snap_part = mtp ? 337.0 * kMiB : 0;
      c.seq_fixed = 112 * kMiB; c.seq_pos = 30796;
      break;
    default: break;
  }
  return c;
}

inline int sessions_of(const Knobs& k) { return std::max(k.max_sessions, k.max_batch); }

// Estimated device bytes the daemon allocates after the weights, without the expert cache and without the reserve.
inline double fixed_bytes(Family f, const Knobs& k) {
  const Costs c = costs(f, k.mtp);
  const double M = std::max(1, k.max_chunk);
  double b = c.fixed + c.row * M;
  if (f == Family::DeepSeek) {
    b += (double)std::max(0, k.prefill_tile - 1) * c.tile_row * M;
    b += (double)std::max(0, k.host_tiles) * c.host_row * M + (double)std::min(2, std::max(0, k.host_tiles)) * c.host_stage_row * M;
    b += (double)std::max(8, k.staging) * c.record;
    b += sessions_of(k) * (c.seq_fixed + c.seq_pos * (double)k.max_ctx);
  } else {
    b += 8 * c.record;  // GlmExperts staging ring
    const int parts = k.mtp ? (k.mtp_batch && k.max_batch >= 2 ? std::min(8, k.max_batch) : 1) : 0;
    b += parts * c.snap_part;
    const double base = (double)std::min<int64_t>(k.kv_base >= 0 ? k.kv_base : k.max_chunk, k.max_ctx);
    b += sessions_of(k) * (c.seq_fixed + c.seq_pos * base);
  }
  return b;
}

struct Step { std::string what; double before = 0, after = 0; };

// Lower the knobs, in a fixed order, until fixed_bytes(k) <= room (room = limit − context − weights − reserve). Every applied step is
//   returned; *fits says whether the result fits. Order (docs/vram-footprint.md §4.2): DS host tiles → DS prefill tiles → DS staging →
//   MTP verify parts → halve --max-chunk (not below 1,024) → halve --max-batch with --max-sessions capped at it (to 1).
inline std::vector<Step> plan(Family f, Knobs& k, double room, bool* fits) {
  std::vector<Step> steps;
  auto fit = [&] { return fixed_bytes(f, k) <= room; };
  auto apply = [&](const std::string& what, auto&& change) {
    if (fit()) return;
    const double before = fixed_bytes(f, k);
    Knobs saved = k;
    change();
    const double after = fixed_bytes(f, k);
    if (after < before) steps.push_back({what, before, after}); else k = saved;
  };
  if (f == Family::DeepSeek) {
    apply("host tiles " + std::to_string(k.host_tiles) + " → 0 (HIVE_PREFILL_HOST_TILES=0)", [&] { k.host_tiles = 0; });
    apply("prefill tiles " + std::to_string(k.prefill_tile) + " → 1 (--prefill-tile 1)", [&] { k.prefill_tile = 1; });
    apply("staging " + std::to_string(k.staging) + " → 8 records (HIVE_STAGING=8)", [&] { k.staging = 8; });
  }
  apply("MTP verify parts → 1 (HIVE_MTP_BATCH=0)", [&] { k.mtp_batch = false; });
  while (!fit() && k.max_chunk / 2 >= 1024) {
    const int from = k.max_chunk;
    apply("--max-chunk " + std::to_string(from) + " → " + std::to_string(from / 2), [&] { k.max_chunk = from / 2; });
    if (k.max_chunk == from) break;
  }
  while (!fit() && (k.max_batch > 1 || k.max_sessions > 1)) {  // fewer requests at once; the session pool follows (sessions = max of both)
    const int from = k.max_batch, sfrom = k.max_sessions, to = std::max(1, from / 2);
    apply("--max-batch " + std::to_string(from) + " → " + std::to_string(to) + ", --max-sessions " + std::to_string(sfrom) + " → " +
              std::to_string(std::min(sfrom, to)),
          [&] { k.max_batch = to; k.max_sessions = std::min(sfrom, to); });
    if (k.max_batch == from && k.max_sessions == sfrom) break;
  }
  *fits = fit();
  return steps;
}

}  // namespace hive::vram_limit
