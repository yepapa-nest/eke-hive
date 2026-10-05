# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
cmake_minimum_required(VERSION 3.24)
project(hive LANGUAGES CXX CUDA)

# hive — inference engine dedicated to DeepSeek-V4.1-Flash (reference host: RTX PRO 6000 96GB + TR PRO 3975WX (Zen2, AVX2) + 1TB DDR4, 2 NUMA nodes).
#  · dense weights and the expert cache live in VRAM; all experts and the engram tables live in pinned RAM (split across the 2 NUMA nodes)
#  · missed experts are computed in place by the CPU from RAM (primary path); part of them goes to the GPU over PCIe DMA (secondary path)
#  · build = the image from docker/Dockerfile (CUDA 13.0 toolkit, cmake, g++13); scripts/build.sh runs cmake inside it
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# sm_120a: block-scaled mma (kind::mxf8f6f4) is only available on the 'a' variant
set(CMAKE_CUDA_ARCHITECTURES 120a)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR "CUDA Toolkit ${CUDAToolkit_VERSION} cannot target sm_120a (Blackwell); 12.8 or newer is required — docs/requirements.md")
endif()
find_library(NUMA_LIB numa REQUIRED)
find_package(Threads REQUIRED)

# Zen2: AVX2 + FMA + F16C (no AVX-512)
set(HIVE_CPU_FLAGS -mavx2 -mfma -mf16c -O3)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${HIVE_CPU_FLAGS}>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-mavx2,-mfma,-mf16c,-O3>")

add_library(hive_core STATIC
  src/safetensors.cpp
  src/cpu/expert_cpu.cpp
  src/kernels/quant.cu
  src/kernels/gemm.cu
  src/kernels/gemm_tc.cu
  src/kernels/gemm_mx.cu
  src/kernels/moe_grouped.cu
  src/kernels/moe_decode.cu
  src/kernels/decode_handshake.cu
  src/kernels/fused.cu
  src/kernels/attn_tc.cu
  src/kernels/attn_decode_fused.cu
  src/kernels/gemv_decode.cu
  src/kernels/decode_attn2.cu
  src/kernels/decode_attn3.cu
  src/kernels/decode_longctx.cu
  src/kernels/hc_decode_fused.cu
  src/kernels/verify_decode.cu
  src/kernels/attn_flash.cu
  src/kernels/topk_select.cu
  src/kernels/model_kernels.cu
  src/kernels/topk.cu
  src/model.cpp
  src/cublas_ops.cpp
  src/expert_store.cpp
  src/engram_hash.cpp
  src/runtime.cpp
  src/vision.cu
)
target_include_directories(hive_core PUBLIC include third_party)
target_link_libraries(hive_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt ${NUMA_LIB} Threads::Threads)

# GLM-5.3-Flash family (NVFP4 experts, KDA + DSA attention): its own library on top of hive_core
add_library(hive_glm STATIC
  src/cpu/expert_cpu_nvfp4.cpp
  src/glm/nvfp4.cu
  src/glm/fp8b.cu
  src/glm/glm_decode.cu
  src/glm/glm_moe_fp8.cu
  src/glm/glm_moe.cu
  src/glm/glm_kernels.cu
  src/glm/glm_vision.cu
  src/glm/kda.cu
  src/glm/dsa.cu
  src/glm/glm_model.cpp
  src/glm/glm_experts.cpp
  src/glm/glm_engine.cpp
  src/glm/glm_mtp.cpp
  src/glm/glm_prefill.cpp
  src/glm/glm_runtime.cpp
)
target_link_libraries(hive_glm PUBLIC hive_core)
foreach(t test_nvfp4_cpu)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} hive_glm)
endforeach()
foreach(t test_glm_nvfp4 test_glm_fp8b test_glm_decode test_glm_moe_fp8 test_glm_moe test_glm_kda test_glm_dsa)
  add_executable(${t} tests/${t}.cu)
  target_link_libraries(${t} hive_glm)
endforeach()
# forward validation against the transformers reference (ref.bin)
add_executable(glm_check tests/glm_check.cpp)
target_link_libraries(glm_check hive_glm)

add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels hive_core)

add_executable(test_mx tests/test_mx.cu)
target_link_libraries(test_mx hive_core)

add_executable(test_mx2 tests/test_mx2.cu)
target_link_libraries(test_mx2 hive_core)
add_executable(test_mx3 tests/test_mx3.cu)
target_link_libraries(test_mx3 hive_core)
add_executable(test_idx_tc tests/test_idx_tc.cu)
target_link_libraries(test_idx_tc hive_core)
add_executable(test_attn_flash tests/test_attn_flash.cu)
target_link_libraries(test_attn_flash hive_core)
add_executable(test_idx_cand tests/test_idx_cand.cu)
target_link_libraries(test_idx_cand hive_core)
add_executable(test_idx_cand_tc tests/test_idx_cand_tc.cu)
target_link_libraries(test_idx_cand_tc hive_core)
add_executable(test_hc_mix tests/test_hc_mix.cu)
target_link_libraries(test_hc_mix hive_core)
add_executable(test_woa tests/test_woa.cu)
target_link_libraries(test_woa hive_core)
add_executable(test_mxg tests/test_mxg.cu)
target_link_libraries(test_mxg hive_core)
add_executable(test_hc_fuse tests/test_hc_fuse.cu)
target_link_libraries(test_hc_fuse hive_core)
add_executable(test_idx_f32 tests/test_idx_f32.cu)
target_link_libraries(test_idx_f32 hive_core)
# HIVE_GROUPED_PREFILL: grouped prefill experts vs the per-expert loop (real layer shapes, bit comparison + timing) and the grouping-plan CPU test
add_executable(test_grouped_prefill tests/test_grouped_prefill.cu)
target_link_libraries(test_grouped_prefill hive_core)
# HIVE_DECODE_FUSED: fused decode experts vs the unfused chain (real layer shapes, M 1..8, bit comparison + µs and GB/s) — test_decode_moe below
add_executable(test_grow_buf tests/test_grow_buf.cu)  # GrowBuf: one address range, memory mapped and trimmed in steps (GLM sequence KV)
target_link_libraries(test_grow_buf hive_core)
add_executable(test_devmem tests/test_devmem.cu)  # sleep level 3: VMM release/restore at the same address + replay of a captured graph (GPU)
target_link_libraries(test_devmem hive_core)
add_executable(test_decode_moe tests/test_decode_moe.cu)
target_link_libraries(test_decode_moe hive_core)
add_executable(test_head_rows tests/test_head_rows.cu)  # HIVE_HEAD_ROWS: multi-row head kernel vs the per-row kernel (bit comparison + µs)
target_link_libraries(test_head_rows hive_core)
add_executable(bench_moe_lowm tests/bench_moe_lowm.cu)  # low-row decode expert launch timing (fixed cost vs per-expert cost)
target_link_libraries(bench_moe_lowm hive_core)
# HIVE_DECODE_ATTN_FUSED: fused decode attention front end vs the fuse_ version (5 layer kinds × context 4K/32K/128K/262K × M 1/4/8, bit comparison + µs per layer)
add_executable(test_decode_attn tests/test_decode_attn.cu)
target_link_libraries(test_decode_attn hive_core)
# HIVE_DECODE_GEMV2/HIVE_DECODE_PDL: new decode dense GEMV, router and hc mix vs the previous kernels (real layer shapes, M 1/4/8, bit comparison + µs and GB/s, layer chain with PDL/graph)
add_executable(test_decode_gemv tests/test_decode_gemv.cu)
target_link_libraries(test_decode_gemv hive_core)
# HIVE_DECODE_ATTN2: new router, q_b, wo_b and shared-expert kernels vs the previous ones (real layer shapes, M 1/3/4/8, bit comparison + µs and GB/s with L2 flushed, layer chain eager/graph)
add_executable(test_decode_attn2 tests/test_decode_attn2.cu)
target_link_libraries(test_decode_attn2 hive_core)
# HIVE_DECODE_ROUTER3/SPARSE3/QKV3/HCMIX3: new router, sparse attention, q_a‖kv and hc mix kernels vs the serving kernels (M 1/4/8, context 4K/32K/128K, bit comparison + µs with L2 flushed, layer chain as graph)
add_executable(test_decode_attn3 tests/test_decode_attn3.cu)
target_link_libraries(test_decode_attn3 hive_core)
add_executable(test_decode_longctx tests/test_decode_longctx.cu)  # long-context decode indexer (TOPK2, IDXSCORE2) vs the previous kernels + timing
target_link_libraries(test_decode_longctx hive_core)
# HIVE_DECODE_STEP_GRAPH: per-layer expert handshake (GPU plan + CPU dispatcher) vs the host-synchronized chain (synthetic layer stack, N-step bit comparison + step time)
add_executable(test_decode_step_graph tests/test_decode_step_graph.cu)
target_link_libraries(test_decode_step_graph hive_core)
# HIVE_DECODE_EARLY_ROUTE: post after the router (er_route_post + er::Gate) and CPU release after the front end vs per-layer synchronization (synthetic layer stack, N-step bit comparison + step time)
add_executable(test_early_route tests/test_early_route.cu)
target_link_libraries(test_early_route hive_core)
# HIVE_HC_DECODE_FUSED/HIVE_HC_SINKHORN_PAR: fused hc mix + sinkhorn vs the unfused path (mix → separate sinkhorn) — M 1/4/8 bit comparison + µs per layer (graph)
add_executable(test_hc_decode tests/test_hc_decode.cu)
target_link_libraries(test_hc_decode hive_core)
# HIVE_DECODE_UBATCH: expert GPU share for two half batches vs the full batch (per-row bit comparison vs the unfused chain / fused decode / F2) + timing model of the production ordering (dov::run_ubatch)
add_executable(test_decode_ubatch tests/test_decode_ubatch.cu)
target_link_libraries(test_decode_ubatch hive_core)
# HIVE_MTP_VERIFY2/HIVE_MTP_BATCH: verify-row kernels (window_idxs_verify, compressor_step_seq, compressor_apply_rows) vs the previous kernels (window_idxs, M calls of
#   compressor_step_rows, compressor_pool) — bit comparison on real shapes (win 128, D 512, ratio 1/2, positions 0/1/127/128/4095 …). Model-level comparison and timing: hive --verify2-test
add_executable(test_verify_decode tests/test_verify_decode.cu)
target_link_libraries(test_verify_decode hive_core)
add_executable(test_grouped_plan_cpu tests/test_grouped_plan_cpu.cpp)
target_include_directories(test_grouped_plan_cpu PRIVATE include)

add_executable(bench_gemv tests/bench_gemv.cu)
target_link_libraries(bench_gemv hive_core)

add_executable(test_dma tests/test_dma.cu)
target_link_libraries(test_dma hive_core)

add_executable(test_expert_cpu tests/test_expert_cpu.cpp)
target_link_libraries(test_expert_cpu hive_core)
# HIVE_CPU_GEMV2: _v2 == _ref bit comparison + ms per expert (single and multi-threaded) — run on the target CPU (Zen2) to compare
add_executable(bench_expert_cpu tests/bench_expert_cpu.cpp)
target_link_libraries(bench_expert_cpu hive_core)

# short prefill (TTFT): 1K–4K prefill A/B with real weights (HIVE_PREFILL_TAIL_SHORT, HIVE_PREFETCH_SCALE, HIVE_PREFILL_SHORT_ADAPT) + [prefill-prof] phase breakdown and logit comparison
add_executable(bench_short_prefill tests/bench_short_prefill.cpp)
target_link_libraries(bench_short_prefill hive_core)

# HIVE_CACHE_ELASTIC: prefill/decode scenarios with real weights (off/on, two runs each) — byte comparison of resident records and logit files (tools/compare_cache_elastic.py)
add_executable(test_cache_elastic tests/test_cache_elastic.cpp)
target_link_libraries(test_cache_elastic hive_core)

add_executable(hive src/hive_main.cpp)
target_link_libraries(hive hive_core)

add_executable(hived src/hived.cpp)
target_link_libraries(hived hive_core)

# Content identity: the source tree may not be a git checkout (or its HEAD may not match the files), so the id is a hash of the sources.
file(GLOB_RECURSE HIVE_ID_FILES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu" "src/*.cuh" "include/*.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${HIVE_ID_FILES})
set(HIVE_SOURCE_INPUT "")
foreach(path IN LISTS HIVE_ID_FILES)
  file(SHA256 "${path}" file_hash)
  file(RELATIVE_PATH relative_path "${CMAKE_CURRENT_SOURCE_DIR}" "${path}")
  string(APPEND HIVE_SOURCE_INPUT "${relative_path}:${file_hash}\n")
endforeach()
string(SHA256 HIVE_SOURCE_SHA "${HIVE_SOURCE_INPUT}")
target_compile_definitions(hived PRIVATE HIVE_BUILD_ID="${HIVE_SOURCE_SHA}")
# hived for the GLM-5.3-Flash family (same daemon source, family types from hive/glm/hived_family.h)
add_executable(hived_glm src/hived.cpp)
target_link_libraries(hived_glm hive_glm)
target_compile_definitions(hived_glm PRIVATE HIVE_FAMILY_GLM HIVE_BUILD_ID="${HIVE_SOURCE_SHA}")

add_executable(st_inspect tools/st_inspect.cpp)
target_link_libraries(st_inspect hive_core)
