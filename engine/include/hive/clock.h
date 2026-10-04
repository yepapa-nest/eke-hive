// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Monotonic wall-clock helpers shared by the engine, daemon, tools and tests.
#pragma once
#include <chrono>

namespace hive {

using SteadyClock = std::chrono::steady_clock;
using Millis = std::chrono::duration<double, std::milli>;
using Seconds = std::chrono::duration<double>;

// Milliseconds on the monotonic clock (arbitrary origin; only differences are meaningful).
inline double mono_ms() { return Millis(SteadyClock::now().time_since_epoch()).count(); }
// Time elapsed since t0.
inline double ms_since(SteadyClock::time_point t0) { return Millis(SteadyClock::now() - t0).count(); }
inline double sec_since(SteadyClock::time_point t0) { return Seconds(SteadyClock::now() - t0).count(); }

}  // namespace hive
