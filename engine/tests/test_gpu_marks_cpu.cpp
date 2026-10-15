// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive/gpu_marks.h on the fake CUDA runtime (tools/cpu_fake/include, FAKE_CUDA_ASYNC=1 = every stream is a worker thread):
//   * the reported fraction is what the stream has executed, not what the host issued — while the stream is held after the first
//     mark, later marks are not reported (the measured failure: FN progress jumped to 75 % of a chunk, then stood still for 8 s),
//   * after the stream drains, the last mark is reported; begin() starts over at 0 and reuses the events,
//   * synchronous streams (default fake mode): every mark is reached at once.
// Build: g++ -std=c++20 -O1 -Itools/cpu_fake/include -Iengine/include engine/tests/test_gpu_marks_cpu.cpp -o /tmp/t && /tmp/t
#include <cstdlib>
#include <cstdio>

int main_impl();
int main() {
  setenv("FAKE_CUDA_ASYNC", "1", 1);  // before the fake runtime reads it
  return main_impl();
}

#include <atomic>
#include <chrono>
#include <thread>

#include "fake_cuda.h"
#include "hive/gpu_marks.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static double settle(hive::ly::GpuMarks& m, double want) {  // the worker thread completes events asynchronously
  double r = m.reached();
  for (int i = 0; i < 400 && r != want; ++i) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); r = m.reached(); }
  return r;
}

int main_impl() {
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);
  hive::ly::GpuMarks m;
  m.begin();
  CHECK(m.reached() == 0.0, "before any mark: %g", m.reached());
  m.mark(s, 0.1);
  CHECK(settle(m, 0.1) == 0.1, "first mark reached");
  std::atomic<bool> release{false};
  fakecuda::enqueue(s, [&] { while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1)); });  // the GPU is busy
  m.mark(s, 0.5);
  m.mark(s, 0.9);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK(m.reached() == 0.1, "held stream: still 0.1, got %g (the host issued 0.9)", m.reached());
  release = true;
  cudaStreamSynchronize(s);
  CHECK(m.reached() == 0.9, "drained: 0.9, got %g", m.reached());
  m.begin();
  CHECK(m.reached() == 0.0, "begin starts over: %g", m.reached());
  m.mark(s, 0.3);
  cudaStreamSynchronize(s);
  CHECK(m.reached() == 0.3, "reused events: 0.3, got %g", m.reached());
  hive::ly::GpuMarks sync_marks;  // default-stream marks complete in the caller
  sync_marks.begin();
  sync_marks.mark(nullptr, 0.4);
  sync_marks.mark(nullptr, 0.8);
  CHECK(sync_marks.reached() == 0.8, "synchronous: 0.8, got %g", sync_marks.reached());
  cudaStreamDestroy(s);
  printf("gpu marks CPU: reported = executed (held stream not overtaken), drain, reuse, synchronous %s\n", fails ? "FAILED" : "OK");
  return fails ? 1 : 0;
}
