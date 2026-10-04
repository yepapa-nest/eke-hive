// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// E3 fused decode hc sinkhorn — switches · standalone parallel sinkhorn · lazy-load warm-up. Device code and the numeric contract are in the hc_decode_fused.cuh header comment.
//
// Background (nsys, service build, 128 decode steps): sinkhorn_kernel runs twice per layer (before attention and before ffn) on a
//   side stream at ≈17 µs (one thread = one token, 20 serial iterations on a 4×4 matrix — 16 divisions × 2 stages × 20 iterations
//   form one thread's dependency chain) · 1.4 ms/step. Two side-stream (hc_side_) fork/join event pairs per layer.
// HIVE_HC_DECODE_FUSED (env_on, default off): the **last mixing block** of the mixing-coefficient kernel (hc_mix_pre_norm /
//   hc_mix_pre_norm2 — following the HIVE_DECODE_GEMV2 choice) solves sinkhorn in parallel, M tokens × 16 lanes → one launch
//   fewer and no side stream or fork/join (the caller records the join event on st_, so the existing wait points still hold).
//   Cost: sinkhorn (parallel, estimated a few µs) sits on the main stream path (otherwise it overlaps attention and the shared expert).
// HIVE_HC_SINKHORN_PAR (env_on, default off · only when FUSED is off): keeps the side stream and only swaps that kernel for the
//   16-lane parallel version (overlap kept + lower SM occupancy).
//   Both are bit-identical (test_hc_decode) — which one pays off in step time is decided by a GPU A/B.
#include <cstdlib>
#include <cstring>

#include "hc_decode_fused.cuh"
#include "hive/gemv_decode.h"
#include "hive/model_kernels.h"

namespace hive::k {

namespace {

// Switch rule = same as env_on in runtime.cpp (unset, "" or "0" = off). Kept separate for the same reason as gd_env_on in gemv_decode.cu.
inline bool hcdf_env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }

template <int HC>
__global__ void sinkhorn_par_kernel(const float* mixes, const float* rsq, const float* __restrict__ scale, const float* __restrict__ base, int M, int iters,
                                    float eps, float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb) {
  hcdf::sinkhorn_block<HC>(mixes, rsq, scale, base, M, iters, eps, pre, post, comb);
}

}  // namespace

bool hc_decode_fused_on() { static const bool on = hcdf_env_on("HIVE_HC_DECODE_FUSED"); return on; }
bool hc_sinkhorn_par_on() { static const bool on = hcdf_env_on("HIVE_HC_SINKHORN_PAR"); return on; }

void hc_split_sinkhorn_par(const float* mixes, const float* rsq, const float* scale, const float* base, int M, int hc, int iters, float eps, float* pre,
                           float* post, float* comb, cudaStream_t st) {
  if (!(hc == 4 && M >= 1 && M <= 16)) { hc_split_sinkhorn(mixes, rsq, scale, base, M, hc, iters, eps, pre, post, comb, st); return; }  // fallback: default version
  sinkhorn_par_kernel<4><<<1, ((M * hcdf::SK_LANES + 31) / 32) * 32, 0, st>>>(mixes, rsq, scale, base, M, iters, eps, pre, post, comb);
}

void hc_decode_fused_preload() {
  cudaFuncAttributes fa;
  CUDA_CHECK(cudaFuncGetAttributes(&fa, sinkhorn_par_kernel<4>));
  hc_mix_pre_norm_sk_preload();
  hc_mix_pre_norm2_sk_preload();
}

}  // namespace hive::k
