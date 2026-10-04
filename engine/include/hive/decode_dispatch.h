// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_STEP_GRAPH — CPU-side handshake thread (dispatcher). Host only (header only — the CPU test compiles it unchanged against fake CUDA).
//
// During one step (arm .. disarm) it watches post_seq in the mapped pinned mailbox; when a post arrives it:
//   ① reads the tables (header, DMA list, CPU jobs), ② issues the staging copies for the DMA share and then the signal (dma_signal — side stream, after the copies), ③ if there are CPU jobs,
//   unpacks the activations to fp32 (same values as the e4m3 table lookup the engine thread used before) and runs them on the ExpertStore pool (start_jobs/wait_jobs — same Job grouping and row order),
//   ④ done_seq ← seq (release). The GPU waits for ②'s signal before the DMA group and for ④ before the CPU accumulation.
// Ordering between posts: after a post with CPU jobs the GPU plans the next layer only once it has seen ④; after a DMA-only post only once it has seen ②
//   (the signal issued after the tables were read) — so rewriting the single mailbox slot every layer never overwrites a slot the dispatcher is still reading (③ reads the activations before ④).
// Waiting: pause-spin while armed (default — this thread takes over the spin the engine thread used to do with per-layer cuStreamSynchronize; the engine thread sleeps in the OS until the step ends).
//   With spin_us ≥ 0, once spin_us passes without a post it polls, sleeping poll_us each time (power vs wake-up latency trade-off — off by default). Between steps (disarmed) it sleeps on a cv.
// Exceptions (pool allocation failure etc.) are recorded in cpu_err and done is still written to release the GPU — the host checks after the step and raises (sequence broken).
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hive/clock.h"
#include "hive/decode_handshake.h"
#include "hive/expert_store.h"

namespace hive {
namespace hs {

class Dispatcher {
 public:
  struct Io {
    ExpertStore* store = nullptr;
    Ctrl* ctrl = nullptr;               // mailbox (host pointer)
    const PostHdr* hdr = nullptr;
    const JobItem* jobs = nullptr;
    const uint8_t* xq = nullptr;        // [Mb, dim] e4m3 · [Mb, dim/32] e8m0
    const uint8_t* xs = nullptr;
    float* a_f = nullptr;               // [Mb, dim] · [Mb, dim/32] (dispatcher-private work buffers)
    float* a_s = nullptr;
    float* scratch = nullptr;           // [Jcap·8 rows × job_scratch_floats(inter)]
    float* cpu_out = nullptr;           // [job_rows, dim] mapped pinned (read by the GPU)
    int dim = 0, inter = 0, Mb = 0, Jcap = 0, nL = 0;
    std::function<void()> thread_init;                      // once at thread start (cudaSetDevice etc.)
    std::function<void(int l, int e, int si)> dma_copy;     // issue the copy of expert (l, e) into staging slot si
    std::function<void(uint32_t seq)> dma_signal;           // GPU signal after this post's copies
    std::function<void(int l)> after_dma;                   // after the DMA issue + signal of layer l's post (after waking the pool if there are CPU jobs) — paces promotion chunks (not called if unset)
    int spin_us = -1, poll_us = 0;
  };
  struct LayerLog { int layer = -1; double cpu_ms = -1; int n_jobs = 0, n_dma = 0; };

  explicit Dispatcher(Io io) : io_(std::move(io)), log_((size_t)std::max(1, io_.nL)) {
    lut_.resize(256);
    for (int i = 0; i < 256; ++i) lut_[i] = e4m3_to_f32((uint8_t)i);
    th_ = std::thread([this] { loop(); });
  }
  ~Dispatcher() {
    { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
    stop_a_.store(true, std::memory_order_relaxed);
    armed_.store(false, std::memory_order_release);
    cv_.notify_all();
    if (th_.joinable()) th_.join();
  }
  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  // engine thread: before the step's GPU work can post
  void arm() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto& g : log_) g = LayerLog{};
      want_ = true; idle_ = false;
      armed_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
  }
  // engine thread: after the step's GPU work has finished (after stream sync). Waits until every post so far is handled and the thread is idle.
  void disarm() {
    armed_.store(false, std::memory_order_release);
    std::unique_lock<std::mutex> lk(mu_);
    cv_idle_.wait(lk, [&] { return idle_; });
  }
  const std::vector<LayerLog>& log() const { return log_; }       // only between disarm and the next arm
  uint32_t served() const { return served_.load(std::memory_order_acquire); }
  uint64_t posts() const { return posts_.load(std::memory_order_relaxed); }

 private:
  static double now_ms() { return hive::mono_ms(); }
  static uint32_t ld_acq(uint32_t* p) { return std::atomic_ref<uint32_t>(*p).load(std::memory_order_acquire); }
  static void st_rel(uint32_t* p, uint32_t v) { std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release); }

  void loop() {
    if (io_.thread_init) io_.thread_init();
    while (true) {
      {
        std::unique_lock lk(mu_);
        cv_.wait(lk, [this] { return want_ || stop_; });
        if (stop_) return;
      }
      double t_idle = now_ms();
      uint32_t done = served_.load(std::memory_order_relaxed);
      for (;;) {
        const bool a = armed_.load(std::memory_order_acquire);        // armed first, then post_seq (so the last post before disarm is not missed)
        const uint32_t ps = ld_acq(&io_.ctrl->post_seq);
        if (ps != done) {
          serve(ps);
          done = ps;
          served_.store(ps, std::memory_order_release);
          t_idle = now_ms();
          continue;
        }
        if (!a) break;
        if (stop_a_.load(std::memory_order_relaxed)) break;
        if (io_.spin_us >= 0 && io_.poll_us > 0 && now_ms() - t_idle > io_.spin_us / 1000.0)
          std::this_thread::sleep_for(std::chrono::microseconds(io_.poll_us));
        else {
#if defined(__x86_64__) || defined(__i386__)
          __builtin_ia32_pause();
#else
          std::this_thread::yield();
#endif
        }
      }
      {
        std::lock_guard<std::mutex> lk(mu_);
        want_ = false; idle_ = true;
      }
      cv_idle_.notify_all();
    }
  }

  void serve(uint32_t seq) {
    posts_.fetch_add(1, std::memory_order_relaxed);
    bool dma_owed = false;  // this post has DMA but the signal has not been sent yet (the GPU is released on the exception path too)
    try {
      const PostHdr h = *io_.hdr;  // after post_seq (acquire) — values the GPU finished writing before posting
      // The range checks below are not a rejection policy but a memory-safety boundary (mailbox values position the host buffers a_f, scratch, cpu_out, so out-of-range writes must be prevented).
      //   On the normal path hs_plan only writes within these ranges — a mismatch is a posting-protocol defect and the step fails via cpu_err (sequence broken).
      if (h.seq != seq || h.n_jobs < 0 || h.n_jobs > io_.Jcap || h.n_dma < 0 || h.n_dma > kMaxDma || h.M < 1 || h.M > io_.Mb || h.layer < 0 ||
          h.layer >= io_.nL)
        throw std::runtime_error("decode handshake: malformed post");
      dma_owed = h.n_dma > 0;
      jb_.assign(io_.jobs, io_.jobs + h.n_jobs);  // ② read the whole table before the signal (after a DMA-only post's signal the GPU may write the next layer's table)
      for (int i = 0; i < h.n_dma; ++i) io_.dma_copy(h.layer, h.dma[i].e, h.dma[i].si);
      if (h.n_dma > 0) { dma_owed = false; io_.dma_signal(seq); }
      LayerLog lg;
      lg.layer = h.layer; lg.n_jobs = h.n_jobs; lg.n_dma = h.n_dma;
      if (h.n_jobs > 0) {
        const int dim = io_.dim, nb = dim / 32;
        for (int m = 0; m < h.M; ++m) {  // same activation unpacking as moe_decode_experts (e4m3 table, e8m0_to_f32)
          const uint8_t* q = io_.xq + (size_t)m * dim;
          float* af = io_.a_f + (size_t)m * dim;
          for (int d = 0; d < dim; ++d) af[d] = lut_[q[d]];
          for (int b = 0; b < nb; ++b) io_.a_s[(size_t)m * nb + b] = e8m0_to_f32(io_.xs[(size_t)m * nb + b]);
        }
        const size_t scratch_n = ExpertStore::job_scratch_floats(io_.inter);
        jobs_.resize(jb_.size());
        for (size_t j = 0; j < jb_.size(); ++j) {
          const JobItem& it = jb_[j];
          if (it.R < 1 || it.R > ExpertStore::kMaxRows || it.row0 < 0 || it.row0 + it.R > io_.Jcap)
            throw std::runtime_error("decode handshake: malformed job");
          ExpertStore::Job& J = jobs_[j];
          J = ExpertStore::Job{};
          J.layer = h.layer; J.e = it.e; J.R = it.R;
          J.scratch = io_.scratch + (size_t)it.row0 * scratch_n;
          for (int r = 0; r < it.R; ++r) {
            const int m = it.m[r];
            if (m < 0 || m >= h.M) throw std::runtime_error("decode handshake: job row out of range");
            J.a_f[r] = io_.a_f + (size_t)m * dim;
            J.a_s[r] = io_.a_s + (size_t)m * nb;
            J.route_w[r] = it.rw[r];
            J.out[r] = io_.cpu_out + (size_t)(it.row0 + r) * dim;
          }
        }
        const double t0 = now_ms();
        io_.store->start_jobs(jobs_);
        bool pump_err = false;  // pace while the pool runs (leaving via an exception before it finishes would write done while the pool is still writing cpu_out — raise after the wait)
        if (io_.after_dma) { try { io_.after_dma(h.layer); } catch (...) { pump_err = true; } }
        io_.store->wait_jobs();
        if (pump_err) throw std::runtime_error("decode handshake: promotion pacing failed");
        lg.cpu_ms = std::max(0.0, io_.store->jobs_done_ms() - t0);
      } else if (io_.after_dma) {
        io_.after_dma(h.layer);  // post without CPU jobs (DMA only) — after the signal
      }
      log_[(size_t)h.layer] = lg;
    } catch (...) {
      st_rel(&io_.ctrl->cpu_err, 1u);
      if (dma_owed) { try { io_.dma_signal(seq); } catch (...) {} }  // exception before the signal: keep the GPU from waiting until timeout (the step fails via cpu_err)
    }
    st_rel(&io_.ctrl->done_seq, seq);  // ④ after all results are written (even on exception the GPU is released — the host sees cpu_err)
  }

  Io io_;
  std::vector<float> lut_;
  std::vector<LayerLog> log_;
  std::vector<JobItem> jb_;
  std::vector<ExpertStore::Job> jobs_;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_, cv_idle_;
  bool stop_ = false, want_ = false, idle_ = true;
  std::atomic<bool> armed_{false}, stop_a_{false};
  std::atomic<uint32_t> served_{0};
  std::atomic<uint64_t> posts_{0};
};

}  // namespace hs
}  // namespace hive
