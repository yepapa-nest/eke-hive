// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B GPU engine (hive/q27/q27_engine.h, with its CPU-placed layers) forward dump for validation against oracle/q27_oracle.py —
//   the same program and outputs as engine/tests/q27_cpu_check.cpp (compare with tools/compare_q27.py). HIVE_Q27_VRAM_MB / HIVE_Q27_GPU_LAYERS
//   choose the placement.
#define Q27_CHECK_GPU 1
#include "q27_cpu_check.cpp"
