# SPDX-License-Identifier: MIT
# HIVE_VRAM_LIMIT_MB startup plan (include/hive/vram_limit.h): CPU test, no CUDA. Not a model family — this file only rides on the
#   cmake/family_*.cmake include hook of CMakeLists.txt.
add_executable(test_vram_plan_cpu tests/test_vram_plan_cpu.cpp)
target_include_directories(test_vram_plan_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(test_vram_plan_cpu PRIVATE HIVE_CPU_ONLY=1)
