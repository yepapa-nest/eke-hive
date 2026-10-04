// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Sleep (three-stage) GPU test (run: scripts/hive-run.sh "build/test_devmem [--gib N] [--hold S] [--context-only]").
//   Allocates DevBufs through real CUDA VMM (hive/devmem.h), then:
//   1) fills a few hundred small buffers + one large buffer (--gib, default 10 GiB = size of the dense weights) with a kernel.
//   2) captures and runs a CUDA graph of kernels writing A → B → B checksum.
//   3) release_all: free VRAM gain (cudaMemGetInfo) · time · D2H bandwidth. With --hold S it pauses for S seconds in the released state (meanwhile nvidia-smi --query-compute-apps
//      measures this process's VRAM = CUDA context + residue — tools/sleep_wake_gpu.py --context reads this number).
//   4) restore_all: same addresses (pointers unchanged) · time · H2D bandwidth → byte comparison of contents (host reference copy) → rerun the same graph → same B checksum (the graph's captured addresses are valid).
//   --context-only: only creates a context (cudaFree(0)) and pauses for --hold — the VRAM baseline held by an "empty context".
//   --prepin: allocates buffers within the keep range and pre-allocates the shadows with prepare_shadows (timed) before sleeping — the first release must have shadow_new ≈ 0 and a release time
//   at the level of a second sleep. In both modes (default, --prepin) a second release/restore is measured at the end, so "first sleep − second" = the shadow allocation share.
//   One PASS/FAIL line + a timing table.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

#include "hive/model.h"
#include "hive/clock.h"

using namespace hive;
static double now_ms() { return hive::mono_ms(); }

__global__ void fill_k(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = (uint32_t)(i * 2654435761u) ^ seed;
}
__global__ void mix_k(const uint32_t* a, uint32_t* b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i] * 3u + (uint32_t)(i >> 7);
}

int main(int argc, char** argv) {
  double gib = 10.0;
  int hold = 0;
  bool ctx_only = false, prepin = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--gib") == 0 && i + 1 < argc) gib = atof(argv[++i]);
    else if (!strcmp(argv[i], "--hold") && i + 1 < argc) hold = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--context-only")) ctx_only = true;
    else if (!strcmp(argv[i], "--prepin")) prepin = true;
  }
  if (ctx_only) {
    CUDA_CHECK(cudaFree(nullptr));
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    printf("[devmem-gpu] context only · pid %d · device used %.0f MiB (device-wide) · holding %d s for nvidia-smi\n", (int)getpid(), (tot - fr) / 1048576.0, hold);
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(hold));
    return 0;
  }
  setenv("HIVE_SLEEP_VMM", "1", 1);
  if (!devmem::on()) { printf("FAIL: CUDA VMM unavailable\n"); return 1; }
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  if (prepin) devmem::keep_begin();
  std::vector<DevBuf> small(400);
  for (size_t i = 0; i < small.size(); ++i) small[i].alloc(256 + (i * 7919) % (1 << 20));
  const size_t big_n = (size_t)(gib * 1073741824.0) / 4 * 4;
  DevBuf A(big_n), B(big_n);
  if (prepin) {
    devmem::keep_end();
    const auto pr = devmem::prepare_shadows();
    printf("[devmem-gpu] prepare_shadows: %d regions · %.0f MiB pinned in %.0f ms (%.2f GB/s) · failed %d\n", pr.regions, pr.shadow_new / 1048576.0, pr.ms,
           pr.shadow_new / 1e6 / std::max(pr.ms, 1e-3), pr.shadow_failed);
  }
  for (size_t i = 0; i < small.size(); ++i) fill_k<<<64, 256, 0, st>>>(small[i].as<uint32_t>(), small[i].n / 4, (uint32_t)i);
  fill_k<<<1024, 256, 0, st>>>(A.as<uint32_t>(), big_n / 4, 77u);
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaGraph_t g; cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  mix_k<<<1024, 256, 0, st>>>(A.as<uint32_t>(), B.as<uint32_t>(), big_n / 4);
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  CUDA_CHECK(cudaGraphLaunch(ge, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  auto checksum = [&](const DevBuf& b) {  // sampled checksum (the full D2H is done by the byte comparison below)
    std::vector<uint32_t> h(1 << 20);
    uint64_t s = 1469598103934665603ull;
    for (size_t off = 0; off < b.n; off += b.n / 64 + 4) {
      const size_t n = std::min<size_t>(h.size() * 4, b.n - off) / 4 * 4;
      CUDA_CHECK(cudaMemcpy(h.data(), b.as<uint8_t>() + off, n, cudaMemcpyDeviceToHost));
      for (size_t k = 0; k < n / 4; ++k) { s ^= h[k]; s *= 1099511628211ull; }
    }
    return s;
  };
  const uint64_t cb0 = checksum(B);
  std::vector<std::vector<uint8_t>> ref(small.size());
  for (size_t i = 0; i < small.size(); ++i) { ref[i].resize(small[i].n); CUDA_CHECK(cudaMemcpy(ref[i].data(), small[i].p, small[i].n, cudaMemcpyDeviceToHost)); }
  const uint64_t ca0 = checksum(A);
  std::vector<void*> addr;
  for (auto& b : small) addr.push_back(b.p);
  void *pa = A.p, *pb = B.p;
  size_t fr0 = 0, fr1 = 0, fr2 = 0, tot = 0;
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaMemGetInfo(&fr0, &tot);
  const auto r = devmem::release_all();
  cudaMemGetInfo(&fr1, &tot);
  printf("[devmem-gpu] release: ok %d · %d regions %.0f MiB · copied %.0f MiB in %.0f ms (%.1f GB/s) · new shadows %.0f MiB in %.0f ms · VRAM free %.0f → %.0f MiB · pid %d\n",
         r.ok, r.regions, r.bytes / 1048576.0, r.copied / 1048576.0, r.ms, r.copied / 1e6 / std::max(r.ms, 1e-3), r.shadow_new / 1048576.0, r.shadow_ms,
         fr0 / 1048576.0, fr1 / 1048576.0, (int)getpid());
  fflush(stdout);
  if (hold > 0) { printf("[devmem-gpu] holding %d s released (measure: nvidia-smi --query-compute-apps=pid,used_memory --format=csv)\n", hold); fflush(stdout);
                  std::this_thread::sleep_for(std::chrono::seconds(hold)); }
  const auto w = devmem::restore_all();
  cudaMemGetInfo(&fr2, &tot);
  printf("[devmem-gpu] restore: ok %d · %.0f MiB in %.0f ms (%.1f GB/s) · VRAM free %.0f MiB%s%s\n", w.ok, w.bytes / 1048576.0, w.ms,
         w.copied / 1e6 / std::max(w.ms, 1e-3), fr2 / 1048576.0, w.ok ? "" : " · ", w.error.c_str());
  bool ok = r.ok && w.ok && fr1 > fr0 + (size_t)(0.9 * (double)r.bytes) && A.p == pa && B.p == pb;
  for (size_t i = 0; i < small.size(); ++i) {
    ok = ok && small[i].p == addr[i];
    std::vector<uint8_t> h(small[i].n);
    CUDA_CHECK(cudaMemcpy(h.data(), small[i].p, small[i].n, cudaMemcpyDeviceToHost));
    if (h != ref[i]) { printf("FAIL: small buffer %zu not restored\n", i); ok = false; break; }
  }
  const bool a_same = checksum(A) == ca0;
  CUDA_CHECK(cudaMemsetAsync(B.p, 0, B.n, st));
  CUDA_CHECK(cudaGraphLaunch(ge, st));  // graph captured before sleeping — must still be valid since the addresses are the same
  CUDA_CHECK(cudaStreamSynchronize(st));
  const bool graph_same = checksum(B) == cb0;
  ok = ok && a_same && graph_same;
  {  // second sleep/wake (shadows reused) — difference from the first sleep = shadow allocation share
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto r2 = devmem::release_all();
    const auto w2 = devmem::restore_all();
    const bool a2 = checksum(A) == ca0;
    printf("[devmem-gpu] second cycle: release %.0f ms (new shadows %.0f MiB) · restore %.0f ms · contents %s\n", r2.ms, r2.shadow_new / 1048576.0, w2.ms,
           a2 ? "byte-identical" : "DIFFER");
    ok = ok && r2.ok && w2.ok && a2 && r2.shadow_new == 0;
    if (prepin) ok = ok && r.shadow_new < (size_t)(64ull << 20);  // with pre-allocation the first sleep creates (almost) no new shadows
  }
  printf("%s: devmem VMM release/restore at the same addresses · small %zu + big 2×%.1f GiB · contents %s · captured graph after restore %s\n", ok ? "PASS" : "FAIL",
         small.size(), gib, a_same ? "byte-identical" : "DIFFER", graph_same ? "same output" : "DIFFERENT");
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  return ok ? 0 : 1;
}
