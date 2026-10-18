# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Qwen3-ASR speech recognition (config model_type "qwen3_asr") — include/hive/qasr/qasr.h, served by server/speech/asr_server.py.
#   libhive_qasr.so: the engine behind a small C API (loaded with ctypes). Self-contained: only the safetensors reader comes from the
#   shared sources, so it builds without hive_core.
#   test_qasr_kernels: the kernels against CPU references on small shapes (+ decode GEMV bandwidth at the real layer shapes).
# Every target is guarded by the existence of its sources: this file is included by every configure of the tree.
set(_qasr ${CMAKE_CURRENT_SOURCE_DIR})
if(EXISTS ${_qasr}/src/qasr/qasr_engine.cpp AND EXISTS ${_qasr}/src/qasr/qasr_kernels.cu)
  add_library(hive_qasr SHARED src/qasr/qasr_engine.cpp src/qasr/qasr_kernels.cu src/safetensors.cpp)
  target_include_directories(hive_qasr PRIVATE include third_party)
  target_link_libraries(hive_qasr PRIVATE CUDA::cudart CUDA::cublas ${CMAKE_DL_LIBS} Threads::Threads)
  if(EXISTS ${_qasr}/tests/test_qasr_kernels.cu)
    add_executable(test_qasr_kernels tests/test_qasr_kernels.cu src/qasr/qasr_kernels.cu)
    target_include_directories(test_qasr_kernels PRIVATE include third_party)
    target_link_libraries(test_qasr_kernels PRIVATE CUDA::cudart CUDA::cublas)
  endif()
endif()
