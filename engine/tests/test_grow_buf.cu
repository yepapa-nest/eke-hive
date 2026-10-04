// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GrowBuf (hive/grow_buf.h) GPU test (run: scripts/hive-run.sh "build/test_grow_buf"): one address range, memory mapped in steps.
//   1) reserve 4 GiB, back 64 MiB, fill it with a kernel; 2) grow to 1 GiB in uneven steps, fill the new part — the first 64 MiB must be
//   unchanged and the address the same; free VRAM must drop by about the mapped size only; 3) trim back to 64 MiB — free VRAM comes back,
//   the first 64 MiB still unchanged; 4) grow again and fill. One PASS/FAIL line.
#include <cstdio>
#include <vector>

#include "hive/grow_buf.h"

using namespace hive;

__global__ void fill_k(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = (uint32_t)(i * 2654435761u) ^ seed;
}
__global__ void check_k(const uint32_t* p, size_t n, uint32_t seed, unsigned long long* bad) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    if (p[i] != ((uint32_t)(i * 2654435761u) ^ seed)) atomicAdd(bad, 1ull);
}
static size_t free_mib() { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return f >> 20; }
static unsigned long long check(const void* p, size_t bytes, uint32_t seed) {
  unsigned long long* d = nullptr; cudaMalloc(&d, 8); cudaMemset(d, 0, 8);
  check_k<<<512, 256>>>((const uint32_t*)p, bytes / 4, seed, d);
  unsigned long long h = 0; cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost); cudaFree(d);
  return h;
}

int main() {
  int fails = 0;
  auto expect = [&](bool ok, const char* what) { if (!ok) { ++fails; printf("FAIL: %s\n", what); } };
  const size_t MiB = 1 << 20;
  GrowBuf b;
  b.reserve(4096 * MiB);
  expect(b.vmm(), "VMM backend available");
  void* const base = b.p;
  const size_t f0 = free_mib();
  expect(b.ensure(64 * MiB) && b.mapped >= 64 * MiB, "ensure 64 MiB");
  fill_k<<<512, 256>>>((uint32_t*)b.p, 64 * MiB / 4, 7);
  const size_t f1 = free_mib();
  for (size_t want : {100 * MiB, 333 * MiB, 1024 * MiB}) expect(b.ensure(want), "grow");
  expect(b.p == base, "address unchanged while growing");
  fill_k<<<512, 256>>>((uint32_t*)((uint8_t*)b.p + 64 * MiB), (1024 - 64) * MiB / 4, 9);
  const size_t f2 = free_mib();
  expect(check(b.p, 64 * MiB, 7) == 0, "first 64 MiB unchanged after growing");
  expect(check((uint8_t*)b.p + 64 * MiB, (1024 - 64) * MiB, 9) == 0, "grown part holds what was written");
  expect(f1 > f2 && f1 - f2 >= 900 && f1 - f2 <= 1100, "free VRAM drops by the grown size only");
  const size_t got = b.trim(64 * MiB);
  cudaDeviceSynchronize();
  const size_t f3 = free_mib();
  expect(got >= 900 * MiB && b.mapped >= 64 * MiB && b.mapped < 128 * MiB, "trim gives the tail back");
  expect(f3 >= f2 + 900, "free VRAM back after trim");
  expect(check(b.p, 64 * MiB, 7) == 0, "first 64 MiB unchanged after trim");
  expect(b.ensure(512 * MiB) && b.p == base, "grow again at the same address");
  fill_k<<<512, 256>>>((uint32_t*)((uint8_t*)b.p + 64 * MiB), (512 - 64) * MiB / 4, 11);
  expect(check((uint8_t*)b.p + 64 * MiB, (512 - 64) * MiB, 11) == 0 && check(b.p, 64 * MiB, 7) == 0, "contents after growing again");
  expect(!b.ensure(8192 * MiB), "ensure beyond the reservation reports false");
  b.free();
  cudaDeviceSynchronize();
  const size_t f4 = free_mib();
  expect(cudaGetLastError() == cudaSuccess, "no CUDA error");
  printf("free VRAM MiB: start %zu · 64 MiB %zu · 1 GiB %zu · trimmed %zu · freed %zu\n", f0, f1, f2, f3, f4);
  printf("%s test_grow_buf (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
