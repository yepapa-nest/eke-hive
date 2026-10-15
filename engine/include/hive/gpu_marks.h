// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Read progress the GPU has really reached inside a prefill forward (layer_yield.h Progress).
//
// Problem (measured 2026-10-15, Qwen3.8-Flash-Next 65,536-row chunks): the host issues a forward's layers ahead of the GPU, so the
//   layer index seen at a host-side layer boundary ran ahead — the reported progress jumped to 75 % of a chunk within a second and then
//   stood still for 8 s while the GPU caught up.
// Fix: record an event at each layer boundary (only in a forward that reports progress) and, when the period is due, report the
//   fraction of the last event the GPU has passed — cudaEventQuery does not wait. Cost: one event record per layer boundary
//   (microseconds) and a few queries per period; no synchronization.
#pragma once
#include <cuda_runtime.h>

#include <vector>

namespace hive::ly {

class GpuMarks {
 public:
  GpuMarks() = default;
  GpuMarks(const GpuMarks&) = delete;
  GpuMarks& operator=(const GpuMarks&) = delete;
  ~GpuMarks() {
    for (cudaEvent_t e : ev_) cudaEventDestroy(e);
  }
  // A new forward: earlier marks are done (the previous forward ended), their events are reused.
  void begin() { n_ = 0; done_ = -1; }
  // After the host issued work up to `frac` of the forward on stream s. A failure only skips this mark (display aid).
  void mark(cudaStream_t s, double frac) {
    if (n_ == (int)ev_.size()) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) { (void)cudaGetLastError(); return; }
      ev_.push_back(e);
      fr_.push_back(0.0);
    }
    if (cudaEventRecord(ev_[n_], s) != cudaSuccess) { (void)cudaGetLastError(); return; }
    fr_[n_] = frac;
    ++n_;
  }
  // Fraction of the last mark the GPU has passed (0 before the first) — never waits.
  double reached() {
    while (done_ + 1 < n_) {
      const cudaError_t r = cudaEventQuery(ev_[done_ + 1]);
      if (r != cudaSuccess) {
        if (r != cudaErrorNotReady) (void)cudaGetLastError();
        break;
      }
      ++done_;
    }
    return done_ >= 0 ? fr_[done_] : 0.0;
  }

 private:
  std::vector<cudaEvent_t> ev_;
  std::vector<double> fr_;
  int n_ = 0, done_ = -1;
};

}  // namespace hive::ly
