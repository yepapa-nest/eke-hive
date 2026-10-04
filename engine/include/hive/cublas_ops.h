// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Thin cuBLAS wrapper — used where the reference computes with torch fp32/bf16 matmul (hc mixes, router, compressor, indexer
// projections, wo_a, vision). Semantics: fp32 accumulation, output dtype equals input dtype (bf16 in → bf16 out).
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>

#include "hive/common.h"

namespace hive {

#define CUBLAS_CHECK(expr)                                                                            \
  do {                                                                                                \
    cublasStatus_t _s = (expr);                                                                       \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                                                \
      fprintf(stderr, "cuBLAS error %d at %s:%d: %s (cuda: %s)\n", (int)_s, __FILE__, __LINE__, #expr,     \
              cudaGetErrorString(cudaGetLastError()));                                                   \
      abort();                                                                                        \
    }                                                                                                 \
  } while (0)

class Blas {
 public:
  explicit Blas(cudaStream_t st);
  ~Blas();
  void set_stream(cudaStream_t st);
  // C[M,N] (row-major) = A[M,K] · B[N,K]ᵀ   (same shape as torch F.linear)
  void gemm_f32(const float* A, const float* B, float* C, int M, int N, int K);
  void gemm_bf16(const bf16* A, const bf16* B, bf16* C, int M, int N, int K);
  // C = A·Bᵀ + C (beta=1, added in fp32 and rounded once) — a Linear with bias in one call, like torch
  void gemm_bf16_acc(const bf16* A, const bf16* B, bf16* C, int M, int N, int K);
  void gemm_bf16_f32out(const bf16* A, const bf16* B, float* C, int M, int N, int K);
  // Batched: C[b] = A[b] · B[b]ᵀ, each with its stride (wo_a grouped einsum, vision attention)
  void gemm_bf16_batched(const bf16* A, long strideA, const bf16* B, long strideB, bf16* C, long strideC, int M, int N, int K,
                         int batch);
  // As above with explicit leading dimensions: A[b] = A + b·strideA (row stride lda), B[b] (row stride ldb), C[b] (row stride ldc)
  void gemm_bf16_batched_ld(const bf16* A, int lda, long strideA, const bf16* B, int ldb, long strideB, bf16* C, int ldc, long strideC,
                            int M, int N, int K, int batch);
  // C[M,N] fp32 = A[M,K] · B[N,K]ᵀ (bf16 inputs, explicit leading dims) — vision QKᵀ scores (no bf16 rounding before softmax)
  void gemm_bf16_f32out_ld(const bf16* A, int lda, const bf16* B, int ldb, float* C, int ldc, int M, int N, int K);
  // C[M,N] = A[M,K] · B[K,N] (B row-major [K,N]) with explicit leading dims
  void gemm_bf16_nn_ld(const bf16* A, int lda, const bf16* B, int ldb, bf16* C, int ldc, int M, int N, int K);
  // C[M,N] = A[M,K] · B[K,N] (for row-major B [K,N]: vision P·V)
  void gemm_bf16_nn_batched(const bf16* A, long strideA, const bf16* B, long strideB, bf16* C, long strideC, int M, int N, int K,
                            int batch);

 private:
  cublasHandle_t h_ = nullptr;
  void* ws_ = nullptr;
  size_t ws_bytes_ = 64ull << 20;  // fixed cuBLAS workspace (avoids internal allocation failures when VRAM is tight)
};

}  // namespace hive
