// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Measures pinned RAM → VRAM DMA bandwidth per NUMA node — to diagnose prefill expert streaming (8K chunk in 21 s = 275 GB/21 s ≈ 13 GB/s, half of PCIe 4.0 x16).
//   H2D copy of 1 GiB in 64 MiB chunks from node 0/1 arenas (numa_alloc_onnode + cudaHostRegister, same as the expert store) · 1 or 2 streams.
// Usage: ./build/test_dma [GiB=1]
#include <numa.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hive/common.h"

using namespace hive;

int main(int argc, char** argv) {
  const size_t gib = argc > 1 ? (size_t)atoi(argv[1]) : 1;
  const size_t bytes = gib << 30, chunk = 64ull << 20;
  const int nodes = numa_available() >= 0 ? numa_max_node() + 1 : 1;
  void* dev;
  CUDA_CHECK(cudaMalloc(&dev, chunk * 4));
  cudaStream_t s[2];
  for (int i = 0; i < 2; ++i) CUDA_CHECK(cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  int dev_id = 0; cudaGetDevice(&dev_id);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev_id);
  printf("GPU %s · pciBus %02x · copy engines %d · NUMA nodes %d\n", p.name, p.pciBusID, p.asyncEngineCount, nodes);
  for (int node = 0; node < nodes; ++node) {
    uint8_t* base = (uint8_t*)numa_alloc_onnode(bytes, node);
    if (!base) { printf("node %d alloc failed\n", node); continue; }
    memset(base, 1, bytes);  // fault the pages in
    CUDA_CHECK(cudaHostRegister(base, bytes, cudaHostRegisterDefault));
    for (int nstreams = 1; nstreams <= 2; ++nstreams) {
      // warm-up
      CUDA_CHECK(cudaMemcpyAsync(dev, base, chunk, cudaMemcpyHostToDevice, s[0]));
      CUDA_CHECK(cudaStreamSynchronize(s[0]));
      CUDA_CHECK(cudaEventRecord(e0, s[0]));
      if (nstreams == 2) CUDA_CHECK(cudaStreamWaitEvent(s[1], e0, 0));
      size_t off = 0; int i = 0;
      while (off < bytes) {
        const size_t n = std::min(chunk, bytes - off);
        cudaStream_t st = s[i % nstreams];
        CUDA_CHECK(cudaMemcpyAsync((uint8_t*)dev + (size_t)(i % 4) * chunk, base + off, n, cudaMemcpyHostToDevice, st));
        off += n; ++i;
      }
      for (int k = 0; k < nstreams; ++k) CUDA_CHECK(cudaStreamSynchronize(s[k]));
      CUDA_CHECK(cudaEventRecord(e1, s[0]));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      printf("node %d → VRAM: %zu GiB in %.0f ms = %.1f GB/s (streams %d)\n", node, gib, ms, bytes / ms / 1e6, nstreams);
    }
    CUDA_CHECK(cudaHostUnregister(base));
    numa_free(base, bytes);
  }
  // both nodes at once (1 GiB each, 2 streams) — same shape as the expert store's even/odd split streaming
  if (nodes >= 2) {
    uint8_t* b[2];
    for (int n = 0; n < 2; ++n) { b[n] = (uint8_t*)numa_alloc_onnode(bytes, n); memset(b[n], 1, bytes); CUDA_CHECK(cudaHostRegister(b[n], bytes, cudaHostRegisterDefault)); }
    CUDA_CHECK(cudaEventRecord(e0, s[0])); CUDA_CHECK(cudaStreamWaitEvent(s[1], e0, 0));
    for (size_t off = 0; off < bytes; off += chunk)
      for (int n = 0; n < 2; ++n) CUDA_CHECK(cudaMemcpyAsync((uint8_t*)dev + (size_t)n * chunk, b[n] + off, std::min(chunk, bytes - off), cudaMemcpyHostToDevice, s[n]));
    for (int k = 0; k < 2; ++k) CUDA_CHECK(cudaStreamSynchronize(s[k]));
    CUDA_CHECK(cudaEventRecord(e1, s[0])); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    printf("node 0+1 concurrent: %zu GiB total in %.0f ms = %.1f GB/s\n", 2 * gib, ms, 2.0 * bytes / ms / 1e6);
    for (int n = 0; n < 2; ++n) { CUDA_CHECK(cudaHostUnregister(b[n])); numa_free(b[n], bytes); }
  }
  return 0;
}
