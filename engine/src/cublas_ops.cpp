// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/cublas_ops.h"
#include "hive/devmem.h"

namespace hive {

Blas::Blas(cudaStream_t st) {
  CUBLAS_CHECK(cublasCreate(&h_));
  CUBLAS_CHECK(cublasSetStream(h_, st));
  // No TF32: the reference is an exact fp32 matmul
  CUBLAS_CHECK(cublasSetMathMode(h_, CUBLAS_DEFAULT_MATH));
  // With VMM the workspace is also restorable at the same VA (sleep level 3 — cuBLAS calls captured in graphs use this address)
  if (devmem::on()) { ws_ = devmem::alloc(ws_bytes_); HIVE_CHECK(ws_ != nullptr, "cuBLAS workspace (VMM) allocation failed"); }
  else CUDA_CHECK(cudaMalloc(&ws_, ws_bytes_));
  CUBLAS_CHECK(cublasSetWorkspace(h_, ws_, ws_bytes_));
}
Blas::~Blas() {
  if (h_) cublasDestroy(h_);
  if (ws_ && !devmem::free_if_owned(ws_)) cudaFree(ws_);
}
void Blas::set_stream(cudaStream_t st) { CUBLAS_CHECK(cublasSetStream(h_, st)); }

// row-major C[M,N] = A[M,K]·B[N,K]ᵀ  ⇔  col-major Cᵀ[N,M] = B[K,N]ᵀ… cuBLAS is col-major: C^T (N×M) = B (N×K; "row-major B[N,K]" is
// col-major K×N, hence op=T for N×K) · A^T (K×M; row-major A[M,K] = col-major K×M, op=N).
void Blas::gemm_f32(const float* A, const float* B, float* C, int M, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, K, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
void Blas::gemm_bf16(const bf16* A, const bf16* B, bf16* C, int M, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF,
                            N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
void Blas::gemm_bf16_acc(const bf16* A, const bf16* B, bf16* C, int M, int N, int K) {
  const float alpha = 1.f, beta = 1.f;
  CUBLAS_CHECK(cublasGemmEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF,
                            N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
void Blas::gemm_bf16_f32out(const bf16* A, const bf16* B, float* C, int M, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &beta, C, CUDA_R_32F,
                            N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
void Blas::gemm_bf16_batched(const bf16* A, long strideA, const bf16* B, long strideB, bf16* C, long strideC, int M, int N, int K,
                             int batch) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, K, strideB, A, CUDA_R_16BF, K,
                                          strideA, &beta, C, CUDA_R_16BF, N, strideC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
void Blas::gemm_bf16_batched_ld(const bf16* A, int lda, long strideA, const bf16* B, int ldb, long strideB, bf16* C, int ldc,
                                long strideC, int M, int N, int K, int batch) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, ldb, strideB, A, CUDA_R_16BF,
                                          lda, strideA, &beta, C, CUDA_R_16BF, ldc, strideC, batch, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
}
void Blas::gemm_bf16_f32out_ld(const bf16* A, int lda, const bf16* B, int ldb, float* C, int ldc, int M, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, ldb, A, CUDA_R_16BF, lda, &beta, C, CUDA_R_32F,
                            ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
void Blas::gemm_bf16_nn_ld(const bf16* A, int lda, const bf16* B, int ldb, bf16* C, int ldc, int M, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, ldb, A, CUDA_R_16BF, lda, &beta, C, CUDA_R_16BF,
                            ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
// C[M,N] = A[M,K]·B[K,N] (both row-major) ⇔ col-major: Cᵀ(N×M) = Bᵀ… B row-major [K,N] = col-major N×K (op N), A row-major [M,K] =
// col-major K×M (op N): Cᵀ = B_cm(N×K) · A_cm(K×M)
void Blas::gemm_bf16_nn_batched(const bf16* A, long strideA, const bf16* B, long strideB, bf16* C, long strideC, int M, int N, int K,
                                int batch) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(h_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, N, strideB, A, CUDA_R_16BF, K,
                                          strideA, &beta, C, CUDA_R_16BF, N, strideC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

}  // namespace hive
