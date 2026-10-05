// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/glm/glm_experts.h"

#include <pthread.h>

#include <sched.h>

#include <numa.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstring>

#include "hive/cublas_ops.h"
#include "hive/devmem.h"
#include "hive/grow_buf.h"
#include "hive/expert_cpu.h"
#include "hive/glm/glm_kernels.h"

namespace hive::glm {

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

namespace {
// Device scratch that must follow the engine into sleep level 3: with HIVE_SLEEP_VMM it is a VMM region (released and restored at the same
//   VA by devmem::release_all / restore_all, like DevBuf); otherwise plain cudaMalloc. The expert slots and the staging ring are NOT allocated
//   this way — sleep level 1 frees them (sleep_release) and wake allocates them again, so level 3 never copies the cache to the host.
void* dev_scratch(size_t bytes) {
  void* p = nullptr;
  if (devmem::on()) { p = devmem::alloc(bytes); HIVE_CHECK(p != nullptr, "GLM scratch (VMM) allocation failed"); }
  else CUDA_CHECK(cudaMalloc(&p, bytes));
  return p;
}
void dev_scratch_free(void* p) { if (p && !devmem::free_if_owned(p)) cudaFree(p); }
}  // namespace


GlmExperts::GlmExperts(GlmModel& m, size_t cache_bytes, int cpu_threads, int staging_slots) : m_(m), cpu_threads_(std::max(1, cpu_threads)) {
  const GlmConfig& c = m.cfg();
  H_ = c.hidden; I_ = c.moe_inter; E_ = c.n_routed; K_ = c.n_act; n_moe_ = c.n_moe; limit_ = c.swiglu_limit;
  L_ = ExpertRecLayout::make(H_, I_);
  host_bytes_ = (size_t)n_keys() * L_.total;
  // interleaved pinned host memory (both NUMA nodes feed the CPU pool and the PCIe copies)
  // HIVE_GLM_NUMA=local: plain mmap (placement follows the process policy, e.g. numactl --membind) — for NUMA A/B measurements
  const char* numa_env = getenv("HIVE_GLM_NUMA");
  numa_interleaved_ = !(numa_env && std::string(numa_env) == "local") && numa_available() >= 0;
  host_ = static_cast<uint8_t*>(numa_interleaved_ ? numa_alloc_interleaved(host_bytes_) : nullptr);
  if (!host_) host_ = static_cast<uint8_t*>(mmap(nullptr, host_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  HIVE_CHECK(host_ && host_ != MAP_FAILED, "GLM experts: host allocation failed");
  slot_of_.assign(n_keys(), -1);
  score_.assign(n_keys(), 0.f);
  CUDA_CHECK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
  n_staging_ = std::max(2, staging_slots);
  CUDA_CHECK(cudaMalloc(&staging_, (size_t)n_staging_ * L_.total));
  stage_ev_.resize(n_staging_);
  for (auto& e : stage_ev_) { CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); CUDA_CHECK(cudaEventRecord(e, side_)); }
  deq_gu_ = (bf16*)dev_scratch((size_t)2 * I_ * H_ * 2);
  deq_d_ = (bf16*)dev_scratch((size_t)H_ * I_ * 2);
  CUDA_CHECK(cudaMallocHost(&host_x_, (size_t)64 * H_ * 4));
  CUDA_CHECK(cudaMallocHost(&host_y_, (size_t)64 * H_ * 4));
  defer_on_ = getenv("HIVE_GLM_DEFER") && atoi(getenv("HIVE_GLM_DEFER")) > 0;
  if (defer_on_) {
    for (auto& b : host_yd_) CUDA_CHECK(cudaMallocHost(&b, (size_t)64 * H_ * 4));
    dthr_ = std::thread([this] {
      for (;;) {
        std::unique_lock<std::mutex> lk(dmu_);
        dcv_.wait(lk, [&] { return dgo_ || dstop_; });
        if (dstop_) return;
        dgo_ = false;
        lk.unlock();
        cpu_experts(dli_, djobs_, host_x_, host_yd_[dbuf_]);
        lk.lock();
        ddone_ = true;
        dcv_.notify_all();
      }
    });
    fprintf(stderr, "[glm] expert deferral on: CPU misses ranked 3rd or lower are added one MoE layer later\n");
  }
  blas_ = new Blas(nullptr);
  if (cache_bytes) alloc_cache(cache_bytes);
  {
    const char* sp = getenv("HIVE_GLM_NUMA_SPLIT");
    split_ = !(sp && std::string(sp) == "0") && numa_available() >= 0 && numa_max_node() >= 1;
    if (const char* su = getenv("HIVE_GLM_SPIN_US")) spin_us_ = std::max(0, atoi(su));
    // HIVE_GLM_PROMOTE=N (measurement): misses promoted into the VRAM cache per token of a decode step (default 8; "0"/unset = default)
    if (const char* pr = getenv("HIVE_GLM_PROMOTE")) { const int v = atoi(pr); if (v > 0) promote_per_step_ = v; }
  }
  next_[0] = 0; next_[1] = 0;  // generation 0 has no tasks
  // Workers are pinned to their node's cpuset only. Pinning each to one physical core spread over the node's L3 domains was measured
  //   (hived, c1 chat ×12, interleaved twice): c1 45.2 / 45.3 pinned vs 46.3 / 45.1 not, CPU phase 79-81 GB/s either way — not kept.
  for (int t = 0; t < cpu_threads_; ++t) {
    const int node = split_ ? (t < (cpu_threads_ + 1) / 2 ? 0 : 1) : -1;
    workers_.emplace_back([this, node] {
      if (node >= 0) {  // pin to the node's CPUs
        bitmask* bm = numa_allocate_cpumask();
        if (numa_node_to_cpus(node, bm) == 0) {
          cpu_set_t cs; CPU_ZERO(&cs);
          for (unsigned i = 0; i < bm->size && i < CPU_SETSIZE; ++i) if (numa_bitmask_isbitset(bm, i)) CPU_SET(i, &cs);
          pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
        }
        numa_free_cpumask(bm);
      }
      const int q = node >= 0 ? node : 0;
      uint64_t seen = 0;
      for (;;) {
        // spin briefly for the next batch (layers follow each other within ~1 ms during decode), then sleep
        uint64_t g = gen_.load(std::memory_order_acquire);
        if (g == seen) {
          const auto t0 = std::chrono::steady_clock::now();
          while ((g = gen_.load(std::memory_order_acquire)) == seen && !stop_ &&
                 std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < spin_us_)
            __builtin_ia32_pause();
          if (g == seen) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || gen_.load() != seen; });
            if (stop_) return;
            g = gen_.load();
          }
        }
        if (stop_) return;
        seen = g;
        const uint64_t gtag = g & 0xffffffffull;
        for (;;) {
          // word = [generation 32 | task count 16 | next index 16]: tag, count and index are read together
          uint64_t v = next_[q].load(std::memory_order_acquire);
          if ((v >> 32) != gtag) break;                                   // a newer batch has started: not ours
          const int i = (int)(v & 0xffffull), n = (int)((v >> 16) & 0xffffull);
          if (i >= n) break;
          if (!next_[q].compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) continue;
          task_(q, i);
          pending_tasks_.fetch_sub(1, std::memory_order_acq_rel);
        }
      }
    });
  }
}
GlmExperts::~GlmExperts() {
  if (dthr_.joinable()) { { std::lock_guard<std::mutex> lk(dmu_); dstop_ = true; } dcv_.notify_all(); dthr_.join(); }
  { std::lock_guard<std::mutex> lk(mu_); stop_ = true; gen_.fetch_add(1); }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
  delete static_cast<Blas*>(blas_);
  if (host_) {
    cudaHostUnregister(host_);
    if (numa_interleaved_) numa_free(host_, host_bytes_); else munmap(host_, host_bytes_);
  }
  for (int k = 0; k < 2; ++k) if (half_[k]) numa_free(half_[k], half_bytes_);
}

void GlmExperts::run_tasks(int n0, int n1, const std::function<void(int, int)>& f) {
  if (!split_) { n0 += n1; n1 = 0; }
  if (n0 + n1 <= 0) return;
  // publish: counts, task and pending total first, then the counters tagged with the new generation, then the generation itself
  const uint64_t ng = gen_.load() + 1;
  HIVE_CHECK(n0 < 65536 && n1 < 65536, "GLM CPU pool: too many tasks");
  task_ = f;
  pending_tasks_.store(n0 + n1, std::memory_order_release);
  next_[0].store(((ng & 0xffffffffull) << 32) | ((uint64_t)n0 << 16), std::memory_order_release);
  next_[1].store(((ng & 0xffffffffull) << 32) | ((uint64_t)n1 << 16), std::memory_order_release);
  { std::lock_guard<std::mutex> lk(mu_); gen_.store(ng, std::memory_order_release); }
  cv_.notify_all();
  while (pending_tasks_.load(std::memory_order_acquire) > 0) __builtin_ia32_pause();
}

// Half records per NUMA node (copies of the interleaved records; the GPU keeps using the full records): node k gets rows
//   [k·I/2, (k+1)·I/2) of w1/w3 (+ scales) and rows [k·H/2, (k+1)·H/2) of w2 (+ scales) — every row keeps its full K, so the CPU kernels
//   run unchanged on the halves. Copied by threads pinned to the target node (numa_alloc_onnode places the pages there anyway).
void GlmExperts::build_halves(int threads) {
  const double t0 = now_ms();
  const size_t Ih = I_ / 2, Hh = H_ / 2;
  auto al = [](size_t x) { return (x + 63) / 64 * 64; };
  HL_.w1 = 0; HL_.s1 = al(Ih * H_ / 2); HL_.w3 = HL_.s1 + al(Ih * H_ / 16); HL_.s3 = HL_.w3 + al(Ih * H_ / 2);
  HL_.w2 = HL_.s3 + al(Ih * H_ / 16); HL_.s2 = HL_.w2 + al(Hh * I_ / 2); HL_.g = HL_.s2 + al(Hh * I_ / 16); HL_.total = al(HL_.g + 16);
  half_bytes_ = (size_t)n_keys() * HL_.total;
  for (int k = 0; k < 2; ++k) {
    half_[k] = static_cast<uint8_t*>(numa_alloc_onnode(half_bytes_, k));
    HIVE_CHECK(half_[k], "GLM experts: half-record allocation failed");
  }
  std::vector<std::thread> th;
  const int per = std::max(1, threads / 2);
  for (int k = 0; k < 2; ++k)
    for (int t = 0; t < per; ++t)
      th.emplace_back([&, k, t] {
        numa_run_on_node(k);
        for (int key = t; key < n_keys(); key += per) {
          const uint8_t* r = host_rec(key);
          uint8_t* d = half_[k] + (size_t)key * HL_.total;
          std::memcpy(d + HL_.w1, r + L_.w1 + k * Ih * H_ / 2, Ih * H_ / 2);
          std::memcpy(d + HL_.s1, r + L_.s1 + k * Ih * H_ / 16, Ih * H_ / 16);
          std::memcpy(d + HL_.w3, r + L_.w3 + k * Ih * H_ / 2, Ih * H_ / 2);
          std::memcpy(d + HL_.s3, r + L_.s3 + k * Ih * H_ / 16, Ih * H_ / 16);
          std::memcpy(d + HL_.w2, r + L_.w2 + k * Hh * I_ / 2, Hh * I_ / 2);
          std::memcpy(d + HL_.s2, r + L_.s2 + k * Hh * I_ / 16, Hh * I_ / 16);
          std::memcpy(d + HL_.g, r + L_.g, 12);
        }
      });
  for (auto& t : th) t.join();
  fprintf(stderr, "[glm] experts: NUMA half records 2 × %.1f GiB built in %.1f s (CPU pool %d threads pinned per node)\n", half_bytes_ / 1073741824.0,
          (now_ms() - t0) / 1000, cpu_threads_);
}

void GlmExperts::load_all(int threads) {
  const GlmConfig& c = m_.cfg();
  Checkpoint& ck = m_.ckpt();
  const double t0 = now_ms();
  struct Src { const uint8_t* p[9]; size_t n[9]; };
  std::vector<Src> src(n_keys());
  const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int l = 0; l < c.n_layers; ++l) {
    if (!c.is_moe[l]) continue;
    const int li = c.moe_index[l];
    for (int e = 0; e < E_; ++e) {
      Src& s = src[(size_t)li * E_ + e];
      for (int j = 0; j < 3; ++j) {
        const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + proj[j] + ".";
        const TensorInfo& tw = ck.get(b + "weight");
        const TensorInfo& ts = ck.get(b + "weight_scale");
        const TensorInfo& tg = ck.get(b + "weight_scale_2");
        HIVE_CHECK(tw.dtype == "U8" && ts.dtype == "F8_E4M3" && tg.dtype == "F32", "GLM: expert must be NVFP4: " + b);
        s.p[j * 3] = tw.data; s.n[j * 3] = tw.nbytes; s.p[j * 3 + 1] = ts.data; s.n[j * 3 + 1] = ts.nbytes; s.p[j * 3 + 2] = tg.data; s.n[j * 3 + 2] = 4;
      }
      HIVE_CHECK(s.n[0] == L_.s1 - L_.w1 && s.n[1] == L_.w3 - L_.s1 && s.n[6] == L_.s2 - L_.w2, "GLM: expert tensor sizes vs record layout");
    }
  }
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int t = 0; t < std::max(1, threads); ++t)
    th.emplace_back([&] {
      for (int k; (k = next.fetch_add(1)) < n_keys();) {
        uint8_t* r = host_ + (size_t)k * L_.total;
        const Src& s = src[k];
        std::memcpy(r + L_.w1, s.p[0], s.n[0]); std::memcpy(r + L_.s1, s.p[1], s.n[1]);
        std::memcpy(r + L_.w3, s.p[3], s.n[3]); std::memcpy(r + L_.s3, s.p[4], s.n[4]);
        std::memcpy(r + L_.w2, s.p[6], s.n[6]); std::memcpy(r + L_.s2, s.p[7], s.n[7]);
        float g[3]; std::memcpy(&g[0], s.p[2], 4); std::memcpy(&g[1], s.p[5], 4); std::memcpy(&g[2], s.p[8], 4);
        std::memcpy(r + L_.g, g, 12);
      }
    });
  for (auto& t : th) t.join();
  const double t1 = now_ms();
  CUDA_CHECK(cudaHostRegister(host_, host_bytes_, cudaHostRegisterDefault));
  if (split_) build_halves(threads);
  fprintf(stderr, "[glm] experts: %d records × %.2f MiB = %.1f GiB loaded in %.1f s (pinned in %.1f s)\n", n_keys(), L_.total / 1048576.0,
          host_bytes_ / 1073741824.0, (t1 - t0) / 1000, (now_ms() - t1) / 1000);
}

// Slot area. With VMM (hive/grow_buf.h) the address range for every slot is reserved once and physical memory is mapped in fixed steps,
//   so the KV growth of a long sequence can take tail steps back (shrink_cache) and return them (regrow_cache) while slot addresses stay
//   where they are. HIVE_GLM_SLOTS_VMM=0 or no VMM: one cudaMalloc as before (the cache cannot shrink then).
// the first version mapped chunks of k slots with k × record bytes a multiple of the granularity; the record (13.5 MiB plus
//   alignment) has no such k ≤ 64, so it silently fell back to cudaMalloc and a 200K prompt found no memory — steps are now independent
//   of the record size.
int GlmExperts::alloc_cache(size_t bytes) {
  if (slots_) return n_slots_;
  int n = (int)(bytes / L_.total);
  devmem::Backend* be = getenv("HIVE_GLM_SLOTS_VMM") && strcmp(getenv("HIVE_GLM_SLOTS_VMM"), "0") == 0 ? nullptr : grow_backend();
  if (be && n > 0) {
    slot_step_ = devmem::round_up(4 * L_.total, be->gran);  // 56 MiB (4 records rounded up to the 2 MiB granularity): the unit the cache shrinks and grows by
    slot_buf_.reserve((size_t)n * L_.total);
    slots_ = slot_buf_.as<uint8_t>(); slots_vmm_ = true; max_slots_ = n;
    for (size_t b = slot_step_; ; b += slot_step_) {
      const size_t want = std::min(b, slot_buf_.n);
      if (!slot_buf_.ensure(want) || want >= (size_t)n * L_.total) break;
    }
    n = std::min(n, (int)(slot_buf_.mapped / L_.total));
  } else {
    while (n > 0 && cudaMalloc(&slots_, (size_t)n * L_.total) != cudaSuccess) { (void)cudaGetLastError(); n -= std::max(1, n / 64); }
    max_slots_ = n;
  }
  n_slots_ = n;
  key_of_slot_.assign(max_slots_, -1);
  pending_.assign(max_slots_, 0);
  fprintf(stderr, "[glm] expert cache: %d slots (%.1f GiB, %.1f %% of %d experts)%s\n", n, (double)n * L_.total / 1073741824.0, 100.0 * n / n_keys(), n_keys(),
          slots_vmm_ ? (" · mapped in " + std::to_string(slot_step_ >> 20) + " MiB steps (long contexts can borrow the tail)").c_str() : "");
  return n;
}

size_t GlmExperts::shrink_cache(size_t bytes) {
  if (!slots_vmm_ || lent_ || asleep_ || bytes == 0 || n_slots_ == 0) return 0;
  CUDA_CHECK(cudaStreamSynchronize(side_));  // promotions / prefetches in flight may target the tail
  commit_ready();
  const int before = n_slots_;
  const size_t have = slot_buf_.mapped;
  int n2 = n_slots_;
  // bytes still mapped if only slots < n stay: whole steps up to the one holding slot n−1 (the last step of the area can be shorter, so cap
  //   by what is mapped — without the cap the subtraction wrapped around and nothing was ever freed)
  auto kept = [&](int n) { return std::min(slot_bytes_mapped(n), have); };
  while (n2 > 0 && have - kept(n2) < bytes) --n2;
  for (int s2 = n2; s2 < n_slots_; ++s2) {
    if (key_of_slot_[s2] >= 0) { slot_of_[key_of_slot_[s2]] = -1; key_of_slot_[s2] = -1; ++stats_.evicted; }
    pending_[s2] = 0;
  }
  n_slots_ = n2;
  const size_t freed = slot_buf_.trim((size_t)n2 * L_.total);
  victims_.clear(); vpos_ = 0;
  if (n_slots_ != before) fprintf(stderr, "[glm] expert cache %d → %d slots (%.0f MiB lent to sequence KV)\n", before, n_slots_, freed / 1048576.0);
  return freed;
}

int GlmExperts::regrow_cache(size_t reserve) {
  if (!slots_vmm_ || lent_ || asleep_ || n_slots_ >= max_slots_) return 0;
  const int before = n_slots_;
  while (n_slots_ < max_slots_) {
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    const size_t next = std::min(slot_buf_.mapped + slot_step_, slot_buf_.n);
    if (next <= slot_buf_.mapped || fr < reserve + (next - slot_buf_.mapped) || !slot_buf_.ensure(next)) break;
    const int n2 = std::min(max_slots_, (int)(slot_buf_.mapped / L_.total));
    for (int s2 = n_slots_; s2 < n2; ++s2) { key_of_slot_[s2] = -1; pending_[s2] = 0; }
    n_slots_ = n2;
  }
  if (n_slots_ > before) {
    victims_.clear(); vpos_ = 0;
    fprintf(stderr, "[glm] expert cache %d → %d slots (sequence KV returned)\n", before, n_slots_);
  }
  return n_slots_ - before;
}

size_t GlmExperts::sleep_release() {
  if (asleep_) return 0;
  HIVE_CHECK(lent_ == 0, "GLM sleep: cache slots are lent to a prefill");
  CUDA_CHECK(cudaDeviceSynchronize());  // promotions and prefetches (side_), streamed prefill copies, every reader of a slot
  commit_ready();
  HIVE_CHECK(promos_.empty(), "GLM sleep: a promotion did not complete");
  const size_t freed = ((size_t)n_slots_ + (size_t)n_staging_) * L_.total;
  for (int k : key_of_slot_) if (k >= 0) slot_of_[k] = -1;
  key_of_slot_.clear(); pending_.clear(); victims_.clear(); vpos_ = 0; step_used_.clear();
  warm_left_ = 0;
  if (slots_vmm_) {
    slot_buf_.free();
    slots_vmm_ = false;
  } else if (slots_) {
    CUDA_CHECK(cudaFree(slots_));
  }
  if (staging_) CUDA_CHECK(cudaFree(staging_));
  slots_ = nullptr; staging_ = nullptr;
  slots_before_sleep_ = n_slots_;
  n_slots_ = 0; max_slots_ = 0;
  asleep_ = true;
  return freed;
}

int GlmExperts::wake_alloc(int slots, size_t reserve) {
  if (!asleep_) return n_slots_;
  if (cudaMalloc(&staging_, (size_t)n_staging_ * L_.total) != cudaSuccess) { (void)cudaGetLastError(); staging_ = nullptr; return -1; }
  for (auto& e : stage_ev_) CUDA_CHECK(cudaEventRecord(e, side_));
  size_t bytes = (size_t)std::max(0, slots) * L_.total;
  if (slots < 0) { size_t fr = 0, tot = 0; CUDA_CHECK(cudaMemGetInfo(&fr, &tot)); bytes = fr > reserve ? fr - reserve : 0; }
  asleep_ = false;
  return alloc_cache(bytes);
}

void GlmExperts::cpu_experts(int li, const std::vector<std::pair<int, std::vector<std::pair<int, float>>>>& jobs, const float* x_host, float* y_host) {
  // phase 1: gate/up rows (per node: that node's half of the rows) · phase 2: down rows (same split)
  // HIVE_GLM_CPU_BALANCE=1 (off by default — measured within noise): each node's rows of all jobs are laid end to end and cut into one equal
  //   range per worker of that node, instead of fixed 128-row chunks. A/B (hived, 32 threads, c1 chat ×12, interleaved twice): CPU
  //   expert phase per verify of 3 rows 18.65 / 19.98 ms on vs 19.44 / 19.73 ms off; c1 median 46.9 / 45.9 vs 47.2 / 44.3 tok/s.
  //   (Reusing the scratch below instead of allocating it per call is what moved this phase: 21-22 → 19-20 ms in the same A/B series.)
  static const bool balance = getenv("HIVE_GLM_CPU_BALANCE") && atoi(getenv("HIVE_GLM_CPU_BALANCE")) > 0;
  const int nodes = split_ ? 2 : 1;
  const int Ip = I_ / nodes, Hp = H_ / nodes;                       // rows per node
  const size_t nj = jobs.size();
  // scratch reused across calls (one row block per (job, row)): gate, up, swiglu output [Rt, I], down output [Rt, H]
  job_off_.resize(nj + 1);
  job_off_[0] = 0;
  for (size_t j = 0; j < nj; ++j) job_off_[j + 1] = job_off_[j] + jobs[j].second.size();
  const size_t Rt = job_off_[nj];
  if (scr_g_.size() < Rt * I_) { scr_g_.resize(Rt * I_); scr_u_.resize(Rt * I_); scr_y_.resize(Rt * I_); }
  if (scr_o_.size() < Rt * H_) scr_o_.resize(Rt * H_);
  float* const G = scr_g_.data(); float* const U = scr_u_.data(); float* const Y = scr_y_.data(); float* const O = scr_o_.data();
  // weights of node k for record key: split → the half record (rows local to the half); else the full record (k = 0)
  struct W { const uint8_t *w1, *s1, *w3, *s3, *w2, *s2; float g1, g3, g2; };
  auto wts = [&](int node, int key) {
    W w;
    if (split_) {
      const uint8_t* r = half_rec(node, key);
      w = {r + HL_.w1, r + HL_.s1, r + HL_.w3, r + HL_.s3, r + HL_.w2, r + HL_.s2, 0, 0, 0};
      std::memcpy(&w.g1, r + HL_.g, 4); std::memcpy(&w.g3, r + HL_.g + 4, 4); std::memcpy(&w.g2, r + HL_.g + 8, 4);
    } else {
      const uint8_t* r = host_rec(key);
      w = {r + L_.w1, r + L_.s1, r + L_.w3, r + L_.s3, r + L_.w2, r + L_.s2, 0, 0, 0};
      std::memcpy(&w.g1, r + L_.g, 4); std::memcpy(&w.g3, r + L_.g + 4, 4); std::memcpy(&w.g2, r + L_.g + 8, 4);
    }
    return w;
  };
  // run body(node, job, n0, n1) over every job's rows [0, rows) of each node
  const int P0 = split_ ? (cpu_threads_ + 1) / 2 : cpu_threads_, P1 = split_ ? cpu_threads_ - P0 : 0;
  auto phase = [&](int rows, const std::function<void(int, int, int, int)>& body) {
    if (!balance) {
      // HIVE_GLM_CPU_ADAPT (default on; "0" = fixed 128/256-row chunks): with few jobs a layer had fewer chunks per node than workers — measured
      //   (HIVE_GLM_PROF, real chat c1, after cache-aware routing and deferral): 1.5 jobs per CPU layer, gate/up workers busy 58 %, the CPU phase at
      //   70 GB/s. Chunks shrink (halving, not below 32 rows) until every worker of a node has one; an output row is one dot product, so the
      //   values do not depend on the chunking.
      static const bool adapt = !(getenv("HIVE_GLM_CPU_ADAPT") && strcmp(getenv("HIVE_GLM_CPU_ADAPT"), "0") == 0);
      int CH = split_ ? 128 : 256;
      const int per_node = split_ ? std::max(P0, P1) : cpu_threads_;
      if (adapt) while (CH > 32 && (long)nj * ((rows + CH - 1) / CH) < per_node) CH /= 2;
      const int chunks = (rows + CH - 1) / CH, n = (int)nj * chunks;
      run_tasks(n, split_ ? n : 0, [&](int node, int t) { const int j = t / chunks, c = t % chunks; body(node, j, c * CH, std::min(rows, (c + 1) * CH)); });
      return;
    }
    const long total = (long)nj * rows;
    run_tasks(P0, P1, [&](int node, int t) {
      const int P = node == 0 ? P0 : P1;
      long a = total * t / P;
      const long b = total * (t + 1) / P;
      while (a < b) {
        const int j = (int)(a / rows), r0 = (int)(a % rows), r1 = (int)std::min<long>(rows, r0 + (b - a));
        body(node, j, r0, r1);
        a += r1 - r0;
      }
    });
  };
  // worker busy time per phase (HIVE_GLM_PROF only — a clock read per task)
  static const bool busy_on = getenv("HIVE_GLM_PROF") && atoi(getenv("HIVE_GLM_PROF")) > 0;
  std::atomic<int64_t> busy_ns[2] = {0, 0};
  struct Busy { std::atomic<int64_t>& acc; std::chrono::steady_clock::time_point t0; bool on;
    ~Busy() { if (on) acc.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed); } };
  const auto Tc0 = std::chrono::steady_clock::now();
  phase(Ip, [&](int node, int j, int n0, int nn) {
    Busy bz{busy_ns[0], busy_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}, busy_on};
    const W w = wts(node, jobs[j].first);
    const auto& rows = jobs[j].second;
    const int R = (int)rows.size();
    const int off = node * Ip;                                      // global row offset of this node's half
    float* ag[64]; float* au[64]; const float* a[64];
    for (int i = 0; i < R; ++i) {
      const size_t r = job_off_[j] + i;
      a[i] = x_host + (size_t)rows[i].first * H_; ag[i] = G + r * I_ + off; au[i] = U + r * I_ + off;
    }
    for (int base = 0; base < R; base += 8) {
      const int r8 = std::min(8, R - base);
      cpu::gemv_nvfp4_rows_multi(w.w1, w.s1, w.g1, Ip, H_, a + base, r8, n0, nn, ag + base);
      cpu::gemv_nvfp4_rows_multi(w.w3, w.s3, w.g3, Ip, H_, a + base, r8, n0, nn, au + base);
    }
    for (int i = 0; i < R; ++i)
      for (int n = n0; n < nn; ++n) {
        float g = bf2f(f2bf(ag[i][n])), u = bf2f(f2bf(au[i][n]));
        if (limit_ > 0.f) { u = fminf(fmaxf(u, -limit_), limit_); g = fminf(g, limit_); }
        Y[(job_off_[j] + i) * I_ + off + n] = bf2f(f2bf(g / (1.f + expf(-g)) * u));
      }
  });
  const auto Tp = std::chrono::steady_clock::now();
  stats_.ms_p1 += std::chrono::duration<double, std::milli>(Tp - Tc0).count();
  phase(Hp, [&](int node, int j, int n0, int nn) {
    Busy bz{busy_ns[1], busy_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}, busy_on};
    const W w = wts(node, jobs[j].first);
    const int R = (int)jobs[j].second.size();
    const int off = node * Hp;
    const float* a[64]; float* y[64];
    for (int i = 0; i < R; ++i) { a[i] = Y + (job_off_[j] + i) * I_; y[i] = O + (job_off_[j] + i) * H_ + off; }
    for (int base = 0; base < R; base += 8)
      cpu::gemv_nvfp4_rows_multi(w.w2, w.s2, w.g2, Hp, I_, a + base, std::min(8, R - base), n0, nn, y + base);
  });
  stats_.ms_p2 += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - Tp).count();
  stats_.workers = cpu_threads_;
  stats_.ms_busy1 += busy_ns[0].load() / 1e6; stats_.ms_busy2 += busy_ns[1].load() / 1e6;
  // reduce into y_host in job order (deterministic)
  for (size_t j = 0; j < nj; ++j) {
    const auto& rows = jobs[j].second;
    for (size_t i = 0; i < rows.size(); ++i) {
      float* dst = y_host + (size_t)rows[i].first * H_;
      const float* o = O + (job_off_[j] + i) * H_;
      for (int n = 0; n < H_; ++n) dst[n] += bf2f(f2bf(o[n])) * rows[i].second;
    }
  }
  (void)li;
}

void GlmExperts::deferred_wait() {
  if (!dpending_) return;
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lk(dmu_);
  dcv_.wait(lk, [&] { return ddone_; });
  dpending_ = false;
  stats_.ms_defer_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void GlmExperts::decode_layer(int li, int M, const int32_t* ids, const float* w, const bf16* x_dev, float* out_dev, cudaStream_t st, const bf16* x_host,
                              bool allow_defer) {
  HIVE_CHECK(!dpending_, "GLM decode_layer: a deferred batch is still pending (deferred_wait first)");
  const bool defer = defer_on_ && allow_defer;
  djobs_.clear();
  HIVE_CHECK(M >= 1 && M <= 64, "GLM decode rows");
  const auto T0 = std::chrono::steady_clock::now();
  auto ms_since0 = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); };
  commit_ready();
  std::vector<MoePair> gpu;
  std::vector<std::pair<int, std::vector<std::pair<int, float>>>> cpu_jobs;  // key → rows
  // HIVE_GLM_SKIP_MISS=f (0/unset = off): a missed expert (not in VRAM) whose weight is below f of its row's routed weight and that is not
  //   among the row's first two choices is left out (its contribution dropped, the others keep their weights) — "skipping low-weight
  //   misses" (dynamic top-k on misses; e.g. ACL Findings 2026 "≥10 % throughput on DeepSeek-V3 without measurable loss"; idea only).
  static const float skip_frac = getenv("HIVE_GLM_SKIP_MISS") ? std::max(0.f, (float)atof(getenv("HIVE_GLM_SKIP_MISS"))) : 0.f;
  for (int m = 0; m < M; ++m) {
    float wsum = 0.f;
    if (skip_frac > 0.f) for (int k = 0; k < K_; ++k) wsum += w[m * K_ + k];
    for (int k = 0; k < K_; ++k) {
      const int key = li * E_ + ids[m * K_ + k];
      score_[key] += 1.f;
      ++stats_.routed;
      const int s = slot_of_[key];
      if (s >= 0) { gpu.push_back({m, dev_rec(s), w[m * K_ + k]}); ++stats_.hit; ++row_hit_[m]; }
      else if (skip_frac > 0.f && k >= 2 && w[m * K_ + k] < skip_frac * wsum) { ++stats_.skipped; step_used_.push_back(key); }
      else if (defer && k >= 2) {
        ++stats_.cpu; ++stats_.deferred; ++row_cpu_[m];
        step_used_.push_back(key);
        auto it = std::find_if(djobs_.begin(), djobs_.end(), [&](auto& j) { return j.first == key; });
        if (it == djobs_.end()) djobs_.push_back({key, {{m, w[m * K_ + k]}}});
        else it->second.push_back({m, w[m * K_ + k]});
      } else {
        ++stats_.cpu; ++row_cpu_[m];
        step_used_.push_back(key);
        auto it = std::find_if(cpu_jobs.begin(), cpu_jobs.end(), [&](auto& j) { return j.first == key; });
        if (it == cpu_jobs.end()) cpu_jobs.push_back({key, {{m, w[m * K_ + k]}}});
        else it->second.push_back({m, w[m * K_ + k]});
      }
    }
  }
  float* hx = host_x_;
  if (!cpu_jobs.empty() || !djobs_.empty()) {
    // x rows to host (bf16 → fp32 on the host)
    const bf16* src = x_host;
    static thread_local std::vector<bf16> xb;
    if (!src) {
      xb.resize((size_t)M * H_);
      CUDA_CHECK(cudaMemcpyAsync(xb.data(), x_dev, (size_t)M * H_ * 2, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaStreamSynchronize(st));
      src = xb.data();
    }
    for (size_t i = 0; i < (size_t)M * H_; ++i) hx[i] = bf2f(src[i]);
    stats_.ms_wait_x += ms_since0(T0);
  }
  if (marks_) CUDA_CHECK(cudaEventRecord(marks_->exp, st));
  if (!gpu.empty()) {
    if ((int)gpu.size() > dpairs_cap_) {
      dev_scratch_free(dpairs_);
      dpairs_cap_ = std::max<int>(gpu.size(), 512);
      dpairs_ = (MoePair*)dev_scratch(dpairs_cap_ * sizeof(MoePair));
    }
    const size_t wsb = moe_decode_ws_bytes((int)gpu.size(), H_, I_);
    if (wsb > moe_ws_bytes_) { dev_scratch_free(moe_ws_); moe_ws_bytes_ = std::max(wsb, moe_decode_ws_bytes(512, H_, I_)); moe_ws_ = dev_scratch(moe_ws_bytes_); }
    static thread_local std::vector<MoePair> staged; staged = gpu;
    CUDA_CHECK(cudaMemcpyAsync(dpairs_, staged.data(), gpu.size() * sizeof(MoePair), cudaMemcpyHostToDevice, st));
    moe_decode(L_, dpairs_, (int)gpu.size(), x_dev, H_, I_, limit_, out_dev, moe_ws_, st);
  }
  if (marks_) CUDA_CHECK(cudaEventRecord(marks_->gend, st));
  if (!cpu_jobs.empty()) {
    std::memset(host_y_, 0, (size_t)M * H_ * 4);
    const auto T1 = std::chrono::steady_clock::now();
    cpu_experts(li, cpu_jobs, hx, host_y_);
    stats_.ms_cpu += ms_since0(T1); ++stats_.cpu_layers;
    stats_.cpu_jobs += cpu_jobs.size(); stats_.cpu_bytes += cpu_jobs.size() * L_.total;
    // the add kernel reads the pinned host buffer directly (UVA): a cudaMemcpy here queued behind the bulk promotion / prefetch copies
    //   on the H2D copy engine (measured: up to ~10 ms per token right after a prefill, when warm promotions run)
    if (marks_) { CUDA_CHECK(cudaEventRecord(marks_->acc, st)); marks_->cpu = true; }
    add_f32(out_dev, host_y_, (size_t)M * H_, st);
    CUDA_CHECK(cudaStreamSynchronize(st));  // host_y_ reused by the next layer
  }
  if (!djobs_.empty()) {  // expert deferral: the rest of this layer's CPU misses run on the helper thread while the caller goes on
    dbuf_ ^= 1;
    std::memset(host_yd_[dbuf_], 0, (size_t)M * H_ * 4);
    { std::lock_guard<std::mutex> lk(dmu_); dli_ = li; drows_ = M; ddone_ = false; dgo_ = true; }
    dcv_.notify_all();
    dpending_ = true;
  }
  stats_.ms_layer += ms_since0(T0);
}

void GlmExperts::prefill_layer(int li, int T, const int32_t* ids, const float* w, const bf16* x_dev, float* out_dev, cudaStream_t st) {
  commit_ready();
  Blas& blas = *static_cast<Blas*>(blas_);
  blas.set_stream(st);
  // rows per expert
  std::vector<std::vector<std::pair<int, float>>> per(E_);
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < K_; ++k) { const int e = ids[t * K_ + k]; per[e].push_back({t, w[t * K_ + k]}); score_[li * E_ + e] += 1.f; }
  int maxR = 0; for (auto& v : per) maxR = std::max<int>(maxR, v.size());
  if (maxR > rows_cap_) {
    for (void* p : {(void*)rows_x_, (void*)rows_gu_, (void*)rows_y_, (void*)rows_o_, (void*)rows_idx_, (void*)rows_w_}) dev_scratch_free(p);
    rows_cap_ = maxR;
    rows_x_ = (bf16*)dev_scratch((size_t)maxR * H_ * 2); rows_gu_ = (bf16*)dev_scratch((size_t)maxR * 2 * I_ * 2);
    rows_y_ = (bf16*)dev_scratch((size_t)maxR * I_ * 2); rows_o_ = (bf16*)dev_scratch((size_t)maxR * H_ * 2);
    rows_idx_ = (int32_t*)dev_scratch((size_t)maxR * 4); rows_w_ = (float*)dev_scratch((size_t)maxR * 4);
  }
  // Pipelined: row indices / weights of every expert in one pinned host table read by the kernels (UVA — no per-expert H2D behind the
  //   staging copies); resident experts first, so the staging ring (n_staging_ slots) fills while they compute; no per-expert sync —
  //   copies on side_ run ahead of the compute on st, a slot is reused only after the compute that read it (stage_ev_).
  //   (The first version synchronized after every expert: copy and compute never overlapped — a 16K-token chunk spent 7.6 s for 3.7 s of
  //   transfers, hived log.)
  size_t total = 0;
  for (auto& v : per) total += v.size();
  if (total > pf_tab_cap_) {
    if (pf_idx_h_) cudaFreeHost(pf_idx_h_);
    if (pf_w_h_) cudaFreeHost(pf_w_h_);
    dev_scratch_free(pf_idx_d_);
    dev_scratch_free(pf_w_d_);
    pf_tab_cap_ = std::max<size_t>(total, 16384 * 8);
    CUDA_CHECK(cudaMallocHost(&pf_idx_h_, pf_tab_cap_ * 4)); CUDA_CHECK(cudaMallocHost(&pf_w_h_, pf_tab_cap_ * 4));
    pf_idx_d_ = (int32_t*)dev_scratch(pf_tab_cap_ * 4); pf_w_d_ = (float*)dev_scratch(pf_tab_cap_ * 4);
  }
  std::vector<int> order; order.reserve(E_);
  for (int e = 0; e < E_; ++e) if (!per[e].empty() && slot_of_[li * E_ + e] >= 0) order.push_back(e);
  for (int e = 0; e < E_; ++e) if (!per[e].empty() && slot_of_[li * E_ + e] < 0) order.push_back(e);
  std::vector<size_t> off(E_, 0);
  {
    size_t o = 0;
    for (int e : order) {
      off[e] = o;
      for (auto& [r, wt] : per[e]) { pf_idx_h_[o] = r; pf_w_h_[o] = wt; ++o; }
    }
  }
  // CPU share of the missed experts (HIVE_GLM_PREFILL_CPU, default on): an expert with few rows is cheaper on the CPU (one pass per 8 rows,
  //   ~0.17 ms with the whole pool) than over PCIe (0.507 ms per record, measured). Fewest-rows-first up to 24 rows, while the CPU
  //   estimate stays below the GPU's remaining transfer time; the CPU part runs on a helper thread while this thread issues the GPU part.
  static const bool pf_cpu = !(getenv("HIVE_GLM_PREFILL_CPU") && strcmp(getenv("HIVE_GLM_PREFILL_CPU"), "0") == 0);
  constexpr double kPcieMs = 0.507, kCpuPassMs = 0.17;
  std::vector<char> on_cpu(E_, 0);
  std::vector<std::pair<int, std::vector<std::pair<int, float>>>> cjobs;
  std::vector<int32_t> crow;  // compact row → chunk row
  if (pf_cpu) {
    std::vector<int> miss;
    for (int e = 0; e < E_; ++e) if (!per[e].empty() && slot_of_[li * E_ + e] < 0) miss.push_back(e);
    std::stable_sort(miss.begin(), miss.end(), [&](int a, int b) { return per[a].size() < per[b].size(); });
    double gpu_t = miss.size() * kPcieMs, cpu_t = 0;
    for (int e : miss) {
      const size_t R = per[e].size();
      if (R > 24) break;
      const double cc = ((R + 7) / 8) * kCpuPassMs;
      if (cpu_t + cc > gpu_t - kPcieMs) break;
      on_cpu[e] = 1; cpu_t += cc; gpu_t -= kPcieMs;
    }
    std::vector<int32_t> cmap(T, -1);
    for (int e = 0; e < E_; ++e) {
      if (!on_cpu[e]) continue;
      std::vector<std::pair<int, float>> rows;
      for (auto& [r, wt] : per[e]) {
        if (cmap[r] < 0) { cmap[r] = (int32_t)crow.size(); crow.push_back(r); }
        rows.push_back({cmap[r], wt});
      }
      cjobs.push_back({li * E_ + e, std::move(rows)});
    }
  }
  const int nc = (int)crow.size();
  std::thread cpu_thread;
  if (nc > 0) {
    if ((size_t)nc > pf_c_cap_) {
      for (void* p : {(void*)pf_crow_h_, (void*)pf_cx_h_, (void*)pf_cy_h_}) if (p) cudaFreeHost(p);
      dev_scratch_free(pf_cx_d_);
      pf_c_cap_ = std::max<size_t>(nc, 1024);
      CUDA_CHECK(cudaMallocHost(&pf_crow_h_, pf_c_cap_ * 4)); CUDA_CHECK(cudaMallocHost(&pf_cx_h_, pf_c_cap_ * H_ * 2));
      CUDA_CHECK(cudaMallocHost(&pf_cy_h_, pf_c_cap_ * H_ * 4)); pf_cx_d_ = (bf16*)dev_scratch(pf_c_cap_ * H_ * 2);
      dev_scratch_free(pf_crow_d_);
      pf_crow_d_ = (int32_t*)dev_scratch(pf_c_cap_ * 4);
    }
    std::memcpy(pf_crow_h_, crow.data(), (size_t)nc * 4);
    CUDA_CHECK(cudaMemcpyAsync(pf_crow_d_, pf_crow_h_, (size_t)nc * 4, cudaMemcpyHostToDevice, st));
    gather_rows(x_dev, pf_crow_d_, nc, H_, pf_cx_d_, st);           // the CPU rows, compacted on the GPU
    dcopy(pf_cx_h_, pf_cx_d_, (size_t)nc * H_ * 2, st);              // → pinned host (SM copy, UVA)
    CUDA_CHECK(cudaStreamSynchronize(st));
    cpu_thread = std::thread([this, li, nc, &cjobs] {
      pf_cxf_.resize((size_t)nc * H_);
      for (size_t i = 0; i < (size_t)nc * H_; ++i) pf_cxf_[i] = bf2f(pf_cx_h_[i]);
      std::memset(pf_cy_h_, 0, (size_t)nc * H_ * 4);
      cpu_experts(li, cjobs, pf_cxf_.data(), pf_cy_h_);
    });
    for (auto& j : cjobs) { stats_.routed += j.second.size(); stats_.cpu += j.second.size(); }
  }
  // one upload of the whole table at the start of the layer (the previous layer's staging copies have finished — synchronized at its end;
  //   kernels reading the table from pinned host memory instead re-read every index per element over PCIe: 15K prefill 8.0 → 11.2 s)
  CUDA_CHECK(cudaMemcpyAsync(pf_idx_d_, pf_idx_h_, total * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(pf_w_d_, pf_w_h_, total * 4, cudaMemcpyHostToDevice, st));
  int stage_next = 0;
  for (int e : order) {
    if (on_cpu[e]) continue;
    const auto& v = per[e];
    const int key = li * E_ + e;
    stats_.routed += v.size();
    const uint8_t* rec;
    int si = -1;
    if (slot_of_[key] >= 0) { rec = dev_rec(slot_of_[key]); stats_.hit += v.size(); }
    else {
      si = stage_next++ % n_staging_;
      uint8_t* dst = staging_ + (size_t)si * L_.total;
      CUDA_CHECK(cudaStreamWaitEvent(side_, stage_ev_[si], 0));  // the compute that used this staging slot has finished
      CUDA_CHECK(cudaMemcpyAsync(dst, host_rec(key), L_.total, cudaMemcpyHostToDevice, side_));
      if (copy_ev_.size() < (size_t)n_staging_) { copy_ev_.resize(n_staging_); for (auto& ev : copy_ev_) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)); }
      CUDA_CHECK(cudaEventRecord(copy_ev_[si], side_));
      CUDA_CHECK(cudaStreamWaitEvent(st, copy_ev_[si], 0));
      rec = dst; ++stats_.streamed;
    }
    const int R = (int)v.size();
    const int32_t* ri = pf_idx_d_ + off[e];
    const float* rwp = pf_w_d_ + off[e];
    expert_dequant(L_, rec, H_, I_, deq_gu_, deq_d_, st);
    gather_rows(x_dev, ri, R, H_, rows_x_, st);
    blas.gemm_bf16(rows_x_, deq_gu_, rows_gu_, R, 2 * I_, H_);
    swiglu_rows(rows_gu_, R, I_, limit_, rows_y_, st);
    blas.gemm_bf16(rows_y_, deq_d_, rows_o_, R, H_, I_);
    scatter_add_rows(rows_o_, ri, rwp, R, H_, out_dev, st);
    if (si >= 0) CUDA_CHECK(cudaEventRecord(stage_ev_[si], st));
  }
  if (cpu_thread.joinable()) {
    cpu_thread.join();
    scatter_add_f32_rows(pf_cy_h_, pf_crow_d_, nc, H_, out_dev, st);  // CPU results (route weights applied), read from pinned memory
  }
  CUDA_CHECK(cudaStreamSynchronize(st));  // the pinned index table is rewritten by the next layer
}

uint8_t* GlmExperts::lend_tail_slots(int n) {
  HIVE_CHECK(lent_ == 0 && n > 0 && n < n_slots_, "GLM experts: lend_tail_slots");
  CUDA_CHECK(cudaStreamSynchronize(side_));  // promotions / prefetches in flight may target these slots
  commit_ready();
  for (int s2 = n_slots_ - n; s2 < n_slots_; ++s2) {
    if (key_of_slot_[s2] >= 0) { slot_of_[key_of_slot_[s2]] = -1; key_of_slot_[s2] = -1; }
    pending_[s2] = 0;
  }
  n_slots_ -= n; lent_ = n;
  victims_.clear(); vpos_ = 0;
  return slots_ + (size_t)n_slots_ * L_.total;
}
void GlmExperts::return_tail_slots() {
  if (!lent_) return;
  n_slots_ += lent_; lent_ = 0;  // the returned slots are empty; promotions and the warm quota refill them
  victims_.clear(); vpos_ = 0;
}

std::vector<int32_t> GlmExperts::resident_keys_by_score() const {
  std::vector<int32_t> keys;
  for (int k : key_of_slot_) if (k >= 0) keys.push_back(k);
  std::stable_sort(keys.begin(), keys.end(), [&](int a, int b) { return score_[a] > score_[b]; });
  return keys;
}
int GlmExperts::warm_from_keys(const std::vector<int32_t>& keys) {
  int s = 0, n = 0;
  for (int32_t k : keys) {
    if (k < 0 || k >= n_keys() || slot_of_[k] >= 0) continue;
    while (s < n_slots_ && (key_of_slot_[s] >= 0 || pending_[s])) ++s;
    if (s >= n_slots_) break;
    CUDA_CHECK(cudaMemcpyAsync(slots_ + (size_t)s * L_.total, host_rec(k), L_.total, cudaMemcpyHostToDevice, side_));
    slot_of_[k] = s; key_of_slot_[s] = k; score_[k] = std::max(score_[k], 1.f);
    ++n;
  }
  CUDA_CHECK(cudaStreamSynchronize(side_));
  return n;
}
void GlmExperts::commit_ready() {
  while (!promos_.empty()) {
    Promo& p = promos_.front();
    const cudaError_t q = cudaEventQuery(p.ev);
    if (q == cudaErrorNotReady) break;
    CUDA_CHECK(q);
    pending_[p.slot] = 0;
    slot_of_[p.key] = p.slot; key_of_slot_[p.slot] = p.key;
    ev_pool_.push_back(p.ev);
    promos_.pop_front();
  }
}

void GlmExperts::promote_key(int key, const std::vector<int>& vic, size_t& vp) {
  if (slot_of_[key] >= 0 || n_slots_ == 0) return;
  for (auto& p : promos_) if (p.key == key) return;
  // victim: lowest-score non-pending slot (empty first, the lowest slot index among equal scores) — the next entry of the list after_step sorted
  //   once (a slot leaves the candidates only by being chosen here, so the next entry is what a fresh scan would find)
  if (vp >= vic.size()) return;
  const int victim = vic[vp];
  const float vs = key_of_slot_[victim] < 0 ? -1.f : score_[key_of_slot_[victim]];
  if (vs >= score_[key]) return;
  ++vp;
  if (key_of_slot_[victim] >= 0) { slot_of_[key_of_slot_[victim]] = -1; ++stats_.evicted; }
  key_of_slot_[victim] = -1;
  pending_[victim] = 1;
  CUDA_CHECK(cudaMemcpyAsync(slots_ + (size_t)victim * L_.total, host_rec(key), L_.total, cudaMemcpyHostToDevice, side_));
  cudaEvent_t ev;
  if (ev_pool_.empty()) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  else { ev = ev_pool_.back(); ev_pool_.pop_back(); }
  CUDA_CHECK(cudaEventRecord(ev, side_));
  promos_.push_back({victim, key, ev});
  ++stats_.promoted;
}

// Next prefetch victim: walk the per-step ascending-score slot list (built in after_step); skip slots in flight and experts of the
//   target layer. Falls back to a full scan when the list is exhausted (or not built yet).
int GlmExperts::next_victim(int li) {
  while (vpos_ < victims_.size()) {
    const int s2 = victims_[vpos_++];
    if (pending_[s2]) continue;
    const int k2 = key_of_slot_[s2];
    if (k2 >= 0 && k2 / E_ == li) continue;
    return s2;
  }
  int victim = -1; float vs = 1e30f;
  for (int s2 = 0; s2 < n_slots_; ++s2) {
    if (pending_[s2]) continue;
    const int k2 = key_of_slot_[s2];
    if (k2 >= 0 && k2 / E_ == li) continue;
    const float sc = k2 < 0 ? -1.f : score_[k2];
    if (sc < vs) { vs = sc; victim = s2; if (sc < 0) break; }
  }
  return victim;
}

int GlmExperts::prefetch(int li, const int32_t* ids, const float* w, int rows, int max_n, cudaStream_t compute) {
  if (max_n <= 0 || n_slots_ == 0) return 0;
  std::vector<std::pair<float, int>> cand;  // (summed predicted weight, key)
  for (int i = 0; i < rows * K_; ++i) {
    const int key = li * E_ + ids[i];
    if (slot_of_[key] >= 0) continue;
    bool inflight = false;
    for (auto& p : promos_) if (p.key == key) { inflight = true; break; }
    if (inflight) continue;
    auto it = std::find_if(cand.begin(), cand.end(), [&](auto& c) { return c.second == key; });
    if (it == cand.end()) cand.push_back({w[i], key}); else it->first += w[i];
  }
  if (cand.empty()) return 0;
  std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
  if (!pf_ev_) CUDA_CHECK(cudaEventCreateWithFlags(&pf_ev_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(pf_ev_, compute));
  CUDA_CHECK(cudaStreamWaitEvent(side_, pf_ev_, 0));
  int n = 0;
  for (auto& [wt, key] : cand) {
    if (n >= max_n) break;
    const int victim = next_victim(li);
    if (victim < 0) break;
    if (key_of_slot_[victim] >= 0) { slot_of_[key_of_slot_[victim]] = -1; ++stats_.evicted; }
    key_of_slot_[victim] = -1;
    pending_[victim] = 1;
    CUDA_CHECK(cudaMemcpyAsync(slots_ + (size_t)victim * L_.total, host_rec(key), L_.total, cudaMemcpyHostToDevice, side_));
    cudaEvent_t ev;
    if (ev_pool_.empty()) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    else { ev = ev_pool_.back(); ev_pool_.pop_back(); }
    CUDA_CHECK(cudaEventRecord(ev, side_));
    promos_.push_back({victim, key, ev});
    ++stats_.prefetched; ++n;
  }
  return n;
}

void GlmExperts::after_step(int tokens) {
  for (auto& s : score_) s *= 0.97f;
  commit_ready();
  // ⚠️ kernels of the step just finished read victim slots: the caller synchronizes its compute stream before calling after_step.
  std::sort(step_used_.begin(), step_used_.end());
  step_used_.erase(std::unique(step_used_.begin(), step_used_.end()), step_used_.end());
  std::sort(step_used_.begin(), step_used_.end(), [&](int a, int b) { return score_[a] > score_[b]; });
  int budget = promote_per_step_ * std::max(1, tokens) + (warm_left_ > 0 ? std::min(warm_left_, warm_per_step_) : 0);
  // Victim candidates sorted once per step: non-pending slots by (score, slot index), the first `budget` of them. promote_key used to scan every
  //   slot for each used key — measured (host timestamps, real chat c1): 1.88 ms per verify step with the GPU idle, ~5 % of the step. Same choices.
  vic_.clear();
  for (int s2 = 0; s2 < n_slots_; ++s2) if (!pending_[s2]) vic_.push_back(s2);
  auto vsc = [&](int s2) { const int k2 = key_of_slot_[s2]; return k2 < 0 ? -1.f : score_[k2]; };
  const size_t vk = std::min<size_t>(vic_.size(), (size_t)std::max(0, budget));
  std::partial_sort(vic_.begin(), vic_.begin() + vk, vic_.end(), [&](int a, int b) { const float x = vsc(a), y = vsc(b); return x < y || (x == y && a < b); });
  vic_.resize(vk);
  size_t vp = 0;
  int issued = 0;
  for (int k : step_used_) { if (issued >= budget) break; const uint64_t before = stats_.promoted; promote_key(k, vic_, vp); issued += stats_.promoted > before; }
  if (warm_left_ > 0 && issued < budget) {
    // warm: top scores among non-resident keys
    std::vector<int> cand;
    for (int k = 0; k < n_keys(); ++k) if (slot_of_[k] < 0 && score_[k] >= 1.f) cand.push_back(k);
    const int need = budget - issued;
    if ((int)cand.size() > need) std::partial_sort(cand.begin(), cand.begin() + need, cand.end(), [&](int a, int b) { return score_[a] > score_[b]; });
    int n = 0;
    for (int k : cand) { if (n >= need) break; const uint64_t before = stats_.promoted; promote_key(k, vic_, vp); n += stats_.promoted > before; }
    warm_left_ = n == 0 ? 0 : warm_left_ - n;
  }
  step_used_.clear();
  // prefetch victims for the next step: the 512 lowest-score slots, ascending
  victims_.resize(n_slots_);
  for (int s2 = 0; s2 < n_slots_; ++s2) victims_[s2] = s2;
  auto sc = [&](int s2) { const int k2 = key_of_slot_[s2]; return k2 < 0 ? -1.f : score_[k2]; };
  const size_t keep = std::min<size_t>(512, victims_.size());
  std::partial_sort(victims_.begin(), victims_.begin() + keep, victims_.end(), [&](int a, int b) { return sc(a) < sc(b); });
  victims_.resize(keep);
  vpos_ = 0;
}

}  // namespace hive::glm
