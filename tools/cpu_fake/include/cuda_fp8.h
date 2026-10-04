// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Fake fp8 conversion header for CPU-only tests. The device conversion is not
// used by host paths under test; calling it aborts so a silent stub cannot pass.
#pragma once
#include <cstdlib>
enum __nv_saturation_t { __NV_NOSAT = 0, __NV_SATFINITE = 1 };
enum __nv_fp8_interpretation_t { __NV_E4M3 = 0, __NV_E5M2 = 1 };
typedef unsigned char __nv_fp8_storage_t;
inline __nv_fp8_storage_t __nv_cvt_float_to_fp8(float, int, int) { abort(); }
