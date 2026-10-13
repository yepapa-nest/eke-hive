// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU test of the HIVE_VRAM_LIMIT_MB startup plan (hive/vram_limit.h): the cost model against the measured 96 GB DeepSeek startup, the
//   shrink order, and the "does not fit" answer. No CUDA.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "hive/vram_limit.h"

using namespace hive::vram_limit;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); ++fails; } } while (0)

int main() {
  const double MiB = (double)kMiB;
  // 1. DeepSeek, config/hive.env on the RTX PRO 6000 (2026-10-09 log): runtime 2,804 MiB (staging included) + prefill-only 10,335 MiB + pool
  //    8 × 250 MiB = 15,139 MiB measured after the weights
  Knobs ds;
  ds.max_chunk = 16384; ds.prefill_tile = 3; ds.host_tiles = 3; ds.staging = 96; ds.max_batch = 8; ds.max_sessions = 2; ds.mtp = true; ds.max_ctx = 262144;
  const double f = fixed_bytes(Family::DeepSeek, ds) / MiB;
  EXPECT(std::fabs(f - 15139) / 15139 < 0.03, "DeepSeek shipped fixed part %.0f MiB, measured 15139", f);
  // 2. plenty of room: nothing changes
  {
    Knobs k = ds; bool fits = false;
    const auto st = plan(Family::DeepSeek, k, 80000 * MiB, &fits);
    EXPECT(fits && st.empty() && k.max_chunk == 16384 && k.host_tiles == 3, "96 GB: no change expected (%zu steps)", st.size());
  }
  // 3. 16 GB card: weights 10,071 + context 786 MiB, reserve 1,700 → room ~3,746 MiB; the plan must reach a fitting state, host tiles first
  {
    Knobs k = ds; bool fits = false;
    const auto st = plan(Family::DeepSeek, k, (16303 - 10071 - 786 - 1700) * MiB, &fits);
    EXPECT(fits, "DeepSeek 16 GB should fit after shrinking (fixed %.0f MiB)", fixed_bytes(Family::DeepSeek, k) / MiB);
    EXPECT(!st.empty() && st[0].what.rfind("host tiles", 0) == 0, "first step must be the host tiles");
    EXPECT(k.max_chunk >= 1024 && k.max_batch >= 1, "knobs out of range");
    for (const auto& s : st) printf("  DS 16GB: %s (%.0f → %.0f MiB)\n", s.what.c_str(), s.before / MiB, s.after / MiB);
  }
  // 4. steps never increase the estimate, and a hopeless room reports not fitting with the knobs at their floor
  {
    Knobs k = ds; bool fits = true;
    const auto st = plan(Family::DeepSeek, k, 100 * MiB, &fits);
    EXPECT(!fits, "100 MiB of room cannot fit");
    for (const auto& s : st) EXPECT(s.after < s.before, "step '%s' did not lower the estimate", s.what.c_str());
    EXPECT(k.max_chunk == 1024 && k.max_batch == 1 && k.max_sessions == 1 && k.prefill_tile == 1 && k.host_tiles == 0 && k.staging == 8,
           "floor: chunk %d batch %d sessions %d", k.max_chunk, k.max_batch, k.max_sessions);
  }
  // 5. GLM shipped (config/glm.env): snapshots 8 parts, pool 8 × 456 MiB measured; MTP batch is the first GLM step
  {
    Knobs k;
    k.max_chunk = 16384; k.max_batch = 8; k.max_sessions = 2; k.mtp = true; k.mtp_batch = true; k.max_ctx = 262144;
    const double pool = sessions_of(k) * (costs(Family::Glm, true).seq_fixed + costs(Family::Glm, true).seq_pos * 16384) / MiB;
    EXPECT(std::fabs(pool - 8 * 456) / (8 * 456) < 0.03, "GLM pool %.0f MiB, measured 3648", pool);
    bool fits = false;
    const auto st = plan(Family::Glm, k, 8000 * MiB, &fits);
    EXPECT(!st.empty() && st[0].what.rfind("MTP verify", 0) == 0 && !k.mtp_batch, "GLM: MTP verify parts first");
    for (const auto& s : st) printf("  GLM 8000 MiB room: %s (%.0f → %.0f MiB)\n", s.what.c_str(), s.before / MiB, s.after / MiB);
  }
  // 6. limit parsing
  setenv("HIVE_VRAM_LIMIT_MB", "16303", 1); EXPECT(limit_bytes() == (size_t)(16303 * MiB), "parse");
  setenv("HIVE_VRAM_LIMIT_MB", "abc", 1);   EXPECT(limit_bytes() == 0, "non-numeric = off");
  setenv("HIVE_VRAM_LIMIT_MB", "0", 1);     EXPECT(limit_bytes() == 0, "0 = off");
  unsetenv("HIVE_VRAM_LIMIT_MB");           EXPECT(limit_bytes() == 0, "unset = off");
  printf("test_vram_plan_cpu: DeepSeek shipped fixed %.0f MiB (measured 15139) · %s\n", f, fails ? "FAILED" : "OK");
  return fails ? 1 : 0;
}
