# SPDX-License-Identifier: MIT
# Qwen3.8-Flash-Next extras of hive_qwen: FN decode kernels (src/qwen/fn_moe.cu) and hybrid placement (hived_qwen with HIVE_QWEN_GPU_LAYERS / a VRAM limit): the GPU family runs the first layers, the FN CPU
#   engine (src/qwen/cpu) the rest. The CPU backend and the CPU engine are compiled into hive_qwen against the real CUDA headers (the
#   hive_cpu library carries the cpu_shim stand-ins and must not be linked into a GPU target).
if(TARGET hive_qwen)
  file(GLOB HIVE_FN_HYBRID_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen/cpu/*.cpp)
  target_sources(hive_qwen PRIVATE ${HIVE_FN_HYBRID_SRC})
  # the CPU backend once per link: hive_core may already carry it (cmake/family_ds_cpu.cmake adds it for the DeepSeek placement) — a second
  #   copy in hive_qwen mixed two builds of the same symbols (2026-10-09: hived_qwen segfaulted in the backend during a long prefill)
  get_target_property(_core_src hive_core SOURCES)
  set(_core_has_backend OFF)
  foreach(_f ${_core_src})
    if(_f MATCHES "src/cpu/cpu_backend\\.cpp$")
      set(_core_has_backend ON)
    endif()
  endforeach()
  if(NOT _core_has_backend)
    file(GLOB HIVE_FN_CPU_BACKEND_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/cpu_*.cpp)
    target_sources(hive_qwen PRIVATE ${HIVE_FN_CPU_BACKEND_SRC})
  endif()
  target_link_libraries(hive_qwen PUBLIC ${NUMA_LIB})
  # FN-specific GPU kernels kept outside the shared CMakeLists source list
  target_sources(hive_qwen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen/fn_moe.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen/fn_moe_pf.cu
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen/fn_gemv.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen/fn_mx.cu)
endif()
# FN decode GEMV shapes (GPU micro-benchmark)
if(TARGET hive_glm AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_fn_gemv.cu)
  add_executable(bench_fn_gemv tests/bench_fn_gemv.cu)
  target_link_libraries(bench_fn_gemv hive_qwen)
endif()
# FN grouped prefill MoE vs the decode kernels (GPU unit test)
if(TARGET hive_glm AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fn_moe_dec.cu)
  add_executable(test_fn_moe_dec tests/test_fn_moe_dec.cu)
  target_link_libraries(test_fn_moe_dec hive_qwen)
endif()
if(TARGET hive_glm AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fn_gemv_mx.cu)
  add_executable(test_fn_gemv_mx tests/test_fn_gemv_mx.cu)   # FP4-Local MXFP8 dense weights: dequant exact, GEMV vs bf16, scale tiles
  target_link_libraries(test_fn_gemv_mx hive_qwen)
endif()
if(TARGET hive_glm AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fn_moe_pf.cu)
  add_executable(test_fn_moe_pf tests/test_fn_moe_pf.cu)
  target_link_libraries(test_fn_moe_pf hive_qwen)
endif()
