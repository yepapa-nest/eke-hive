# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Shared CPU backend (src/cpu, include/hive/cpu/cpu_backend.h) — measurement and A/B tools, CPU only. Not a model family: this file only
#   rides on the cmake/family_*.cmake include hook of CMakeLists.txt; every target is guarded by the existence of its sources.
#   ab_cpu_matmul        current matmul against the 2026-10-09 snapshot (src/cpu/cpu_matmul_legacy.cpp), bitwise, in one process
#   bench_cpu_matmul_ab  GEMV / MoE / GEMM timings, current vs snapshot; STREAM read bandwidth per NUMA node
#   bench_cpu_parts      single-thread cost of the matmul's parts (compiles src/cpu/cpu_matmul.cpp into itself to reach its kernels)
set(_cpub ${CMAKE_CURRENT_SOURCE_DIR})
if(TARGET hive_cpu AND EXISTS ${_cpub}/src/cpu/cpu_matmul_legacy.cpp)
  foreach(t ab_cpu_matmul bench_cpu_matmul_ab)
    if(EXISTS ${_cpub}/tests/${t}.cpp)
      add_executable(${t} tests/${t}.cpp)
      target_link_libraries(${t} hive_cpu)
    endif()
  endforeach()
  if(EXISTS ${_cpub}/tests/bench_cpu_parts.cpp)
    add_executable(bench_cpu_parts tests/bench_cpu_parts.cpp src/cpu/cpu_backend.cpp src/cpu/cpu_matmul_legacy.cpp src/cpu/cpu_attention.cpp)
    target_include_directories(bench_cpu_parts BEFORE PRIVATE ${_cpub}/cpu_shim)
    target_include_directories(bench_cpu_parts PRIVATE ${_cpub}/include ${_cpub}/third_party)
    target_compile_definitions(bench_cpu_parts PRIVATE HIVE_CPU_ONLY=1)
    target_link_libraries(bench_cpu_parts ${NUMA_LIB} Threads::Threads)
  endif()
endif()
