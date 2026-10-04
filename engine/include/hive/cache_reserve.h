// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace hive {

// HIVE_CACHE_RESERVE_MB: VRAM left free beside the expert cache (unset, empty, non-numeric or negative = 2400 MiB). One rule for hived's
//   HIVE_CACHE_FIT sizing and for every runtime that hands cache memory to something else and must keep this headroom.
inline size_t cache_reserve_bytes() {
  const char* v = getenv("HIVE_CACHE_RESERVE_MB");
  char* end = nullptr;  // atof turns a non-number into 0, which left zero headroom (vision VRAM shortage) — use the value only if the whole string is numeric
  const double x = v && *v ? strtod(v, &end) : 2400.0;
  const double mb = v && *v && !(end != v && end && *end == 0) ? 2400.0 : x;
  return (size_t)((std::isfinite(mb) && mb >= 0 ? std::min(mb, 1e7) : 2400.0) * 1048576.0);
}

}  // namespace hive
