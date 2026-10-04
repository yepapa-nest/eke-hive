// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Fake cuBLAS types for CPU-only tests (no cuBLAS call is executed).
#pragma once
typedef struct cublasContext* cublasHandle_t;
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
