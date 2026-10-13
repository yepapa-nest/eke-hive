# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# DeepSeek-V4.1-Flash on the CPU only (hive/ds/ds_cpu.h): the engine library, the daemon (hived.cpp with the family surface of
#   hive/ds/hived_family_cpu.h) and the validation driver (tools/compare_ds_cpu.py compares its dumps with the oracle).
#   Included by every configure (CMakeLists.txt family hook): every target is guarded by the existence of its sources.
file(GLOB HIVE_DS_CPU_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/ds/cpu/*.cpp)
if(HIVE_DS_CPU_SRC AND TARGET hive_cpu)
  add_library(hive_ds_cpu STATIC ${HIVE_DS_CPU_SRC})
  target_link_libraries(hive_ds_cpu PUBLIC hive_cpu)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/ds/ds_cpu_runtime.cpp AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/hive/ds/hived_family_cpu.h)
    add_executable(hived_ds_cpu src/hived.cpp src/ds/ds_cpu_runtime.cpp)
    target_link_libraries(hived_ds_cpu hive_ds_cpu)
    target_compile_definitions(hived_ds_cpu PRIVATE HIVE_FAMILY_DS_CPU HIVE_BUILD_ID="${HIVE_SOURCE_SHA}")
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/ds_cpu_check.cpp)
    add_executable(ds_cpu_check tests/ds_cpu_check.cpp)
    target_link_libraries(ds_cpu_check hive_ds_cpu)
  endif()
endif()
# Layer-level placement in the default (GPU) DeepSeek family (runtime.cpp Runtime::CpuLayers): the same CPU engine, compiled into hive_core
#   with the real CUDA headers (hive_ds_cpu above is the cpu_shim build for hived_ds_cpu).
file(GLOB HIVE_CPU_BACKEND_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpu_*.cpp)
if(TARGET hive_core AND HIVE_DS_CPU_SRC AND HIVE_CPU_BACKEND_SRC)
  target_sources(hive_core PRIVATE ${HIVE_CPU_BACKEND_SRC} ${HIVE_DS_CPU_SRC})
endif()
