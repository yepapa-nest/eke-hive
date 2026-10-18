# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Qwen3-TTS 12Hz CustomVoice text-to-speech (hive/qtts/qtts_engine.h, math contract oracle/qwen3_tts_oracle.py).
#   hive_qtts: shared library with the C API (hive/qtts/qtts_capi.h) — loaded by server/speech/tts_server.py through ctypes.
#   test_qtts_*: GPU tests (kernels against CPU references; engine runs against oracle dumps).
# Every target is guarded by the existence of its sources: this file is included by every configure of the tree.
set(_qtts ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB HIVE_QTTS_SRC CONFIGURE_DEPENDS ${_qtts}/src/qtts/*.cu ${_qtts}/src/qtts/*.cpp)
if(HIVE_QTTS_SRC)
  # one shared CUDA runtime for the library and everything linked to it (a static runtime inside a dlopen'ed library would be a second copy)
  add_library(hive_qtts SHARED ${HIVE_QTTS_SRC} ${_qtts}/src/safetensors.cpp)
  target_include_directories(hive_qtts PUBLIC ${_qtts}/include ${_qtts}/third_party)
  target_link_libraries(hive_qtts PUBLIC CUDA::cudart CUDA::cublas ${CMAKE_DL_LIBS})
  set_target_properties(hive_qtts PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  file(GLOB HIVE_QTTS_TESTS CONFIGURE_DEPENDS ${_qtts}/tests/test_qtts_*.cu)
  foreach(t IN LISTS HIVE_QTTS_TESTS)
    get_filename_component(n ${t} NAME_WE)
    add_executable(${n} ${t})
    target_link_libraries(${n} hive_qtts)
    set_target_properties(${n} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  endforeach()
endif()
