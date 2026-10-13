# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# GLM-5.3-Flash on the CPU only (hive/glm/glm_cpu.h): the engine library, the daemon hived_glm_cpu (same hived.cpp, family types from
#   hive/glm/hived_family_cpu.h, built against engine/cpu_shim like hived_qwen_cpu) and the validation tool glm_cpu_check.
#   Every target is guarded by its sources so that a partial checkout still configures.
file(GLOB HIVE_GLM_CPU_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/glm/cpu/*.cpp)
if(HIVE_GLM_CPU_SRC AND TARGET hive_cpu)
  add_library(hive_glm_cpu STATIC ${HIVE_GLM_CPU_SRC})
  target_link_libraries(hive_glm_cpu PUBLIC hive_cpu)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/glm/glm_cpu_runtime.cpp AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/hive/glm/hived_family_cpu.h)
    add_executable(hived_glm_cpu ${CMAKE_CURRENT_SOURCE_DIR}/src/hived.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/glm/glm_cpu_runtime.cpp)
    target_link_libraries(hived_glm_cpu hive_glm_cpu)
    target_compile_definitions(hived_glm_cpu PRIVATE HIVE_FAMILY_GLM_CPU HIVE_BUILD_ID="${HIVE_SOURCE_SHA}")
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_cpu_check.cpp)
    add_executable(glm_cpu_check ${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_cpu_check.cpp)
    target_link_libraries(glm_cpu_check hive_glm_cpu)
  endif()
endif()

# Hybrid placement in the GPU family (GlmEngine, HIVE_GLM_GPU_LAYERS / HIVE_VRAM_LIMIT_MB): the same CPU engine and CPU backend compiled for
#   the GPU build (real CUDA headers, no cpu_shim) and linked into hive_glm.
file(GLOB HIVE_CPU_BACKEND_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpu_*.cpp)
if(HIVE_GLM_CPU_SRC AND HIVE_CPU_BACKEND_SRC AND TARGET hive_glm AND TARGET hive_core)
  add_library(hive_glm_hybrid STATIC ${HIVE_CPU_BACKEND_SRC} ${HIVE_GLM_CPU_SRC})
  target_link_libraries(hive_glm_hybrid PUBLIC hive_core)
  target_link_libraries(hive_glm PUBLIC hive_glm_hybrid)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_hybrid_check.cpp)
    add_executable(glm_hybrid_check ${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_hybrid_check.cpp)
    target_link_libraries(glm_hybrid_check hive_glm)
  endif()
endif()
