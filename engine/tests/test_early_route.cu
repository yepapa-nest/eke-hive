// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Verifies HIVE_DECODE_EARLY_ROUTE (GPU test).
//   Synthetic layer stack (NL layers, real shapes dim 5120, inter 2304, top-6, NE experts per layer in the real storage format e2m1+e8m0, ~60 % resident, real ExpertStore CPU pool):
//     front graph (one CUDA graph per layer — like production run_graph) = [(layer 0) x0 / (layer >= 1) bf16(acc) -> rmsnorm] -> act_quant_fp8 -> synthetic router
//       -> (NEW only) k::er_route_post (routing -> mapped host + post number) -> synthetic "shared expert" (bandwidth kernel reading WIN_MB — stands in for the shared-expert span after the router)
//       -> acc = 0 -> route_to_host (OLD = routing + activations, NEW = activations only)
//     experts (same order as production moe_decode_experts) = classify (hs::plan_host) -> issue DMA -> [OLD: unpack activations -> start pool] -> table H2D -> resident groups -> DMA wait -> DMA groups
//       -> [NEW: wait for front-end event -> unpack activations -> start pool] -> CPU wait -> CPU accumulation (fixed row order — for strict comparison).
//   (OLD) per layer front graph -> cudaStreamSynchronize -> experts (production baseline decode_layer).
//   (NEW) per layer Gate.begin -> front graph -> front-end event -> Gate.wait (mapped number, production hive/early_route.h as is) -> experts (activation unpack after the front end).
//   Verdict: per step the last layer's acc (fp32) NEW vs OLD **bitwise**, and zero Gate absorptions (resync/missing/untrusted) = PASS condition.
//   Timing: step wall-clock ms, engine thread CPU ms, "ahead" = fraction of layers where the host issued GPU experts before the front ended (= no GPU idle after the front on that layer).
//   Run: scripts/hive-run.sh "./build-dev/test_early_route [steps=24] [NL=8] [NE=48] [cpu_threads=16] [WIN_MB=48] [order]"
//   Batch regression reproduction: order = "cpufirst" runs the HIVE_DECODE_CPU_FIRST order (classify -> unpack activations, start pool -> table H2D -> resident groups -> **issue demand DMA**
//     -> DMA wait -> DMA groups) and measures three variants: OLD, NEW (table H2D on st after the front end) and NEWFIX (table H2D on a dedicated stream — the fix in runtime.cpp
//     moe_decode_experts). Hypothesis: in NEW the demand DMA grabs the copy engine first, so the table copy (-> resident groups) is pushed to the end of the DMA -> at M >= 2 (misses >= 3 -> DMA present)
//     it is slower than OLD, and NEWFIX is at least OLD. All three are bit-compared against OLD. No argument (default) = baseline order (demand DMA before the table — two variants OLD/NEW, output unchanged).
//   Needs: VRAM ~= 0.6*NL*NE x 18.9 MB + WIN_MB + 0.5 GB, pinned RAM ~= NL*NE x 18.9 MB.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/decode_handshake.h"
#include "hive/early_route.h"
#include "hive/expert_store.h"
#include "hive/kernels.h"
#include "hive/model.h"
#include "hive/model_kernels.h"
#include "hive/clock.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

constexpr int DIM = 5120, INTER = 2304, K = 6, MB = 8;
constexpr float LIMIT = 10.f;

double now_ms() { return hive::mono_ms(); }
double thread_cpu_ms() { timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

// Synthetic router (same formula as test_decode_step_graph): per row, hash of activation bytes -> K distinct experts (first 3 from a popular pool), rw
__global__ void router_synth(const uint8_t* __restrict__ xq, int M, int dim, int E, int layer, int32_t* __restrict__ ids, float* __restrict__ rw) {
  const int m = blockIdx.x;
  if (m >= M || threadIdx.x != 0) return;
  uint32_t h = 2166136261u ^ (uint32_t)(layer * 0x9e3779b9u);
  for (int d = 0; d < dim; d += 16) { h ^= xq[(size_t)m * dim + d]; h *= 16777619u; }
  int pick[K];
  for (int j = 0; j < K; ++j) {
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    int e = j < 3 ? (int)(h % (uint32_t)max(K, E / 4)) : (int)(h % (uint32_t)E);
    for (bool dup = true; dup;) {
      dup = false;
      for (int q = 0; q < j; ++q) if (pick[q] == e) { dup = true; e = (e + 1) % E; break; }
    }
    pick[j] = e;
    ids[m * K + j] = e;
    rw[m * K + j] = 0.1f + (float)((h >> 8) & 15) / 64.f;
  }
}
// Synthetic shared-expert span: reads the buffer to the end (result kept in one slot so it is not optimised away — unrelated to acc)
__global__ void window_read(const uint4* __restrict__ p, size_t n, uint32_t* __restrict__ out) {
  uint32_t s = 0;
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) { const uint4 v = p[i]; s ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (s == 0x9e3779b9u) out[0] = s;
}
__global__ void accum_f32_ordered(const float* __restrict__ src, const int32_t* __restrict__ rows, int n, int M, int dim, float* __restrict__ acc) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * dim || n <= 0) return;
  const int m = (int)(i / dim), d = (int)(i % dim);
  float s = acc[i];
  for (int r = 0; r < n; ++r)
    if (rows[r] == m) s += __ldcv(src + (size_t)r * dim + d);
  acc[i] = s;
}
__global__ void fill_bf16(bf16* p, size_t n, uint32_t seed) {
  const size_t i = hive::cu::global_tid();
  if (i >= n) return;
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
  p[i] = f2bf(((float)(h & 0xFFFF) / 32768.f - 1.f) * 2.f);
}
__global__ void fill_ones(bf16* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (n > i) p[i] = f2bf(1.f); }

template <class T> T* mapped(size_t n, T** dev) {
  T* p = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&p, n * sizeof(T), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)dev, p, 0));
  memset((void*)p, 0, n * sizeof(T));
  return p;
}

struct Rig {
  int NL, NE, S;
  size_t win_n = 0;  // number of uint4
  Config cfg;
  std::unique_ptr<ExpertStore> store;
  ExpertLayout lay;
  hs::RecOff rec{};
  cudaStream_t st = nullptr, side = nullptr;
  DevBuf x0, inb, tmpb, ones, xq, xs, ids, rw, y, yq, ys, eout, acc, tbl_d, cpu_rows_d, win, win_out;
  DevBuf ctr;
  int32_t* ids_h = nullptr; int32_t* ids_hd = nullptr; float* rw_h = nullptr; float* rw_hd = nullptr;
  uint8_t* xq_h = nullptr; uint8_t* xq_hd = nullptr; uint8_t* xs_h = nullptr; uint8_t* xs_hd = nullptr;
  float* cpu_out = nullptr; float* cpu_out_d = nullptr;
  uint32_t* flag_h = nullptr; uint32_t* flag_d = nullptr;
  uint8_t* tbl_h = nullptr;  // pinned (like production w.tbl_h — the H2D must be async to overlap the front end)
  std::vector<int32_t> share;
  std::vector<float> a_f, a_s, scratch, lut;
  std::vector<cudaEvent_t> copied, freed;
  std::vector<char> used;
  uint32_t stage_next = 0;
  std::unique_ptr<hs::HostPlan> hp = std::make_unique<hs::HostPlan>();
  std::vector<ExpertStore::Job> jobs;
  cudaEvent_t front_ev = nullptr;
  int order = 0;                      // 0 = baseline order, 1 = CPU_FIRST order (demand DMA after the resident-group launch)
  cudaStream_t tbl_st = nullptr;      // NEWFIX: dedicated stream for the table H2D
  cudaEvent_t tbl_ev = nullptr;
  er::Gate gate;
  std::vector<cudaGraphExec_t> exec;  // [early][M][l]
  long ahead = 0, layers_new = 0;

  Rig(int nl, int ne, int cpu_threads, int win_mb) : NL(nl), NE(ne) {
    cfg.dim = DIM; cfg.moe_inter = INTER; cfg.n_routed = NE; cfg.n_act = K; cfg.swiglu_limit = LIMIT; cfg.dspark_experts = std::min(NE, 128);
    lay = ExpertLayout::make(DIM, INTER);
    rec = hs::RecOff{lay.w1, lay.s1, lay.w3, lay.s3, lay.w2, lay.s2, lay.total};
    const int n_res = (int)(0.6 * NL * NE);
    store = std::make_unique<ExpertStore>(cfg, NL, (size_t)n_res * lay.total, cpu_threads, 0);
    S = store->staging_slots();
    const HalfLayout& hl = store->half_layout();
    std::mt19937_64 rng(20260930);
    for (int l = 0; l < NL; ++l)
      for (int e = 0; e < NE; ++e)
        for (int node = 0; node < 2; ++node) {
          uint8_t* h = const_cast<uint8_t*>(store->host_half(l, e, node));
          auto rnd = [&](size_t a, size_t b) { for (size_t i = a; i < b; i += 8) { const uint64_t v = rng(); memcpy(h + i, &v, std::min<size_t>(8, b - i)); } };
          auto scl = [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) h[i] = (uint8_t)(118 + rng() % 7); };
          rnd(hl.w1, hl.s1); scl(hl.s1, hl.w3); rnd(hl.w3, hl.s3); scl(hl.s3, hl.w2); rnd(hl.w2, hl.s2); scl(hl.s2, hl.total);
        }
    store->pin_all();
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    int placed = 0;
    for (int l = 0; l < NL; ++l)
      for (int e = 0; e < NE; ++e)
        if (placed < n_res && rng() % 100 < 60) { store->place(l, e, st); ++placed; }
    CUDA_CHECK(cudaStreamSynchronize(st));
    const int Mrows = MB * K;
    x0.alloc((size_t)MB * DIM * 2); inb.alloc((size_t)MB * DIM * 2); tmpb.alloc((size_t)MB * DIM * 2); ones.alloc((size_t)DIM * 2);
    fill_ones<<<(DIM + 255) / 256, 256, 0, st>>>(ones.as<bf16>(), DIM);
    xq.alloc((size_t)MB * DIM); xs.alloc((size_t)MB * DIM / 32); ids.alloc((size_t)MB * K * 4); rw.alloc((size_t)MB * K * 4);
    y.alloc((size_t)Mrows * INTER * 2); yq.alloc((size_t)Mrows * INTER); ys.alloc((size_t)Mrows * INTER / 32); eout.alloc((size_t)Mrows * DIM * 2);
    acc.alloc((size_t)MB * DIM * 4);
    const size_t tbl_bytes = (sizeof(k::GroupDesc) + 8) * (size_t)Mrows * 2;
    tbl_d.alloc(tbl_bytes); cpu_rows_d.alloc((size_t)Mrows * 4);
    CUDA_CHECK(cudaHostAlloc((void**)&tbl_h, tbl_bytes, cudaHostAllocDefault));
    win_n = std::max<size_t>(1, (size_t)win_mb * 1048576 / 16);
    win.alloc(win_n * 16); win_out.alloc(64);
    CUDA_CHECK(cudaMemset(win.p, 0x5a, win_n * 16));
    ctr.alloc(4); CUDA_CHECK(cudaMemset(ctr.p, 0, 4));
    ids_h = mapped<int32_t>((size_t)MB * K, &ids_hd); rw_h = mapped<float>((size_t)MB * K, &rw_hd);
    xq_h = mapped<uint8_t>((size_t)MB * DIM, &xq_hd); xs_h = mapped<uint8_t>((size_t)MB * DIM / 32, &xs_hd);
    cpu_out = mapped<float>((size_t)Mrows * DIM, &cpu_out_d);
    flag_h = mapped<uint32_t>(1, &flag_d);
    a_f.resize((size_t)MB * DIM); a_s.resize((size_t)MB * DIM / 32); scratch.resize((size_t)Mrows * ExpertStore::job_scratch_floats(INTER));
    copied.resize(S); freed.resize(S); used.assign(S, 0);
    for (int i = 0; i < S; ++i) { CUDA_CHECK(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&freed[i], cudaEventDisableTiming)); }
    CUDA_CHECK(cudaEventCreateWithFlags(&front_ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreateWithFlags(&tbl_st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&tbl_ev, cudaEventDisableTiming));
    lut.resize(256);
    for (int i = 0; i < 256; ++i) lut[i] = e4m3_to_f32((uint8_t)i);
    share.resize(hs::kMaxE + 1);  // runtime.cpp formula (frac 0.25, decode_gpu_share 1)
    const float frac = 0.25f; const int dma_cap = 8, gshare = 1;
    for (int n = 0; n <= hs::kMaxE; ++n) share[n] = n >= 3 ? std::min(std::min(dma_cap, S), std::max(gshare, (int)(frac * n + 0.5f))) : 0;
    exec.assign((size_t)2 * (MB + 1) * NL, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  ~Rig() {
    for (auto e : exec) if (e) cudaGraphExecDestroy(e);
    for (auto e : copied) cudaEventDestroy(e);
    for (auto e : freed) cudaEventDestroy(e);
    cudaEventDestroy(front_ev);
    cudaEventDestroy(tbl_ev); cudaStreamDestroy(tbl_st);
    for (void* p : {(void*)ids_h, (void*)rw_h, (void*)xq_h, (void*)xs_h, (void*)cpu_out, (void*)flag_h, (void*)tbl_h}) cudaFreeHost(p);
    cudaStreamDestroy(st); cudaStreamDestroy(side);
  }

  void front_body(int l, int M, bool early) {
    if (l == 0) CUDA_CHECK(cudaMemcpyAsync(inb.p, x0.p, (size_t)M * DIM * 2, cudaMemcpyDeviceToDevice, st));
    else {
      k::f32_to_bf16(acc.as<float>(), M * DIM, tmpb.as<bf16>(), st);
      k::rmsnorm(tmpb.as<bf16>(), ones.as<bf16>(), 1e-6f, M, DIM, inb.as<bf16>(), st);
    }
    k::act_quant_fp8(inb.as<bf16>(), M, DIM, xq.as<uint8_t>(), xs.as<uint8_t>(), st);
    router_synth<<<M, 32, 0, st>>>(xq.as<uint8_t>(), M, DIM, NE, l, ids.as<int32_t>(), rw.as<float>());
    if (early) k::er_route_post(ids.as<int32_t>(), rw.as<float>(), M * K, ids_hd, rw_hd, ctr.as<uint32_t>(), flag_d, st);
    window_read<<<1024, 256, 0, st>>>(win.as<uint4>(), win_n, win_out.as<uint32_t>());
    CUDA_CHECK(cudaMemsetAsync(acc.p, 0, (size_t)M * DIM * 4, st));
    k::route_to_host(ids.as<int32_t>(), rw.as<float>(), xq.as<uint8_t>(), xs.as<uint8_t>(), early ? 0 : M * K, M * DIM, ids_hd, rw_hd, xq_hd, xs_hd, st);
  }
  void front(int l, int M, bool early) {  // production run_graph: eager on first use, capture on the second
    cudaGraphExec_t& ex = exec[((size_t)(early ? 1 : 0) * (MB + 1) + M) * NL + l];
    if (!ex) {
      static std::vector<char> seen;
      if (seen.empty()) seen.assign(exec.size(), 0);
      char& s = seen[((size_t)(early ? 1 : 0) * (MB + 1) + M) * NL + l];
      if (!s) { s = 1; front_body(l, M, early); return; }
      cudaGraph_t g = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
      front_body(l, M, early);
      CUDA_CHECK(cudaStreamEndCapture(st, &g));
      CUDA_CHECK(cudaGraphInstantiate(&ex, g, 0));
      CUDA_CHECK(cudaGraphDestroy(g));
    }
    CUDA_CHECK(cudaGraphLaunch(ex, st));
  }
  void unpack(int M) {
    for (int m = 0; m < M; ++m) {
      for (int d = 0; d < DIM; ++d) a_f[(size_t)m * DIM + d] = lut[xq_h[(size_t)m * DIM + d]];
      for (int b = 0; b < DIM / 32; ++b) a_s[(size_t)m * (DIM / 32) + b] = e8m0_to_f32(xs_h[(size_t)m * (DIM / 32) + b]);
    }
  }
  // Production moe_decode_experts order. early = activation unpack and pool start after the front end (decode_layer early-route path)
  void experts(int l, int M, bool early, bool tbl_own = false) {
    hs::HostPlan& p = *hp;
    hs::plan_host(ids_h, rw_h, M, K, NE, store->slot_table() + (size_t)l * store->E(), share.data(), p, store->dev_rec(0), store->staging_rec(0), rec,
                  (int)(stage_next % (uint32_t)S), S);
    const hs::LayerCounts& c = p.counts;
    auto issue_copies = [&] {
      for (int i = 0; i < c.n_dma; ++i) {
        const int si = p.dma[i].si;
        if (used[si]) CUDA_CHECK(cudaStreamWaitEvent(side, freed[si], 0));
        store->copy_to_staging(si, l, p.dma[i].e, side);
        CUDA_CHECK(cudaEventRecord(copied[si], side));
      }
    };
    if (order == 0) issue_copies();  // baseline order (cpufirst = after the resident-group launch — production defer_copy)
    jobs.assign(c.n_jobs, ExpertStore::Job{});
    const size_t sn = ExpertStore::job_scratch_floats(INTER);
    for (int j = 0; j < c.n_jobs; ++j) {
      const hs::JobItem& it = p.jobs[j];
      ExpertStore::Job& J = jobs[j];
      J.layer = l; J.e = it.e; J.R = it.R; J.scratch = scratch.data() + (size_t)it.row0 * sn;
      for (int r = 0; r < it.R; ++r) {
        J.a_f[r] = a_f.data() + (size_t)it.m[r] * DIM; J.a_s[r] = a_s.data() + (size_t)it.m[r] * (DIM / 32); J.route_w[r] = it.rw[r];
        J.out[r] = cpu_out + (size_t)(it.row0 + r) * DIM;
      }
    }
    if (!early && c.n_jobs > 0) { unpack(M); store->start_jobs(jobs); }
    const int ng = c.ng_hit + c.ng_str, nr = c.R_hit + c.R_str;
    const size_t off_rows = sizeof(k::GroupDesc) * (size_t)ng, off_rw = off_rows + (size_t)nr * 4;
    if (ng > 0) {
      k::GroupDesc* gd = reinterpret_cast<k::GroupDesc*>(tbl_h);
      for (int g = 0; g < c.ng_hit; ++g) gd[g] = p.gd_hit[g];
      for (int g = 0; g < c.ng_str; ++g) gd[c.ng_hit + g] = p.gd_dma[g];
      memcpy(tbl_h + off_rows, p.g_rows, (size_t)nr * 4);
      memcpy(tbl_h + off_rw, p.g_rw, (size_t)nr * 4);
      if (early && tbl_own) {  // NEWFIX: the same three calls as the er_tbl branch of production moe_decode_experts (reads of the previous layer's table finished before the post number)
        CUDA_CHECK(cudaMemcpyAsync(tbl_d.p, tbl_h, off_rw + (size_t)nr * 4, cudaMemcpyHostToDevice, tbl_st));
        CUDA_CHECK(cudaEventRecord(tbl_ev, tbl_st));
        CUDA_CHECK(cudaStreamWaitEvent(st, tbl_ev, 0));
      } else
      CUDA_CHECK(cudaMemcpyAsync(tbl_d.p, tbl_h, off_rw + (size_t)nr * 4, cudaMemcpyHostToDevice, st));
    }
    const k::GroupDesc* gdev = reinterpret_cast<const k::GroupDesc*>(tbl_d.as<uint8_t>());
    const int32_t* rows = reinterpret_cast<const int32_t*>(tbl_d.as<uint8_t>() + off_rows);
    const float* rwd = reinterpret_cast<const float*>(tbl_d.as<uint8_t>() + off_rw);
    auto run_groups = [&](int g0, int n_groups, int r0, int n_rows) {
      if (n_groups <= 0) return;
      k::mx_grouped_w13(gdev + g0, n_groups, xq.as<uint8_t>(), xs.as<uint8_t>(), INTER, DIM, rwd, LIMIT, y.as<bf16>(), st, rows, yq.as<uint8_t>(), ys.as<uint8_t>());
      k::mx_grouped_w2(gdev + g0, n_groups, yq.as<uint8_t>(), ys.as<uint8_t>(), DIM, INTER, eout.as<bf16>(), st);
      k::accum_bf16_rows_seq(eout.as<bf16>() + (size_t)r0 * DIM, rows + r0, n_rows, M, DIM, acc.as<float>(), st);
    };
    run_groups(0, c.ng_hit, 0, c.R_hit);
    if (order == 1) issue_copies();  // cpufirst: where production defer_copy issues
    for (int i = 0; i < c.n_dma; ++i) CUDA_CHECK(cudaStreamWaitEvent(st, copied[p.dma[i].si], 0));
    run_groups(c.ng_hit, c.ng_str, c.R_hit, c.R_str);
    for (int i = 0; i < c.n_dma; ++i) { CUDA_CHECK(cudaEventRecord(freed[p.dma[i].si], st)); used[p.dma[i].si] = 1; }
    stage_next += (uint32_t)c.n_dma;
    if (early) {
      ++layers_new;
      const cudaError_t q = cudaEventQuery(front_ev);
      if (q == cudaErrorNotReady) ++ahead; else CUDA_CHECK(q);
      CUDA_CHECK(cudaEventSynchronize(front_ev));  // activations arrived (production wait_front)
      if (c.n_jobs > 0) { unpack(M); store->start_jobs(jobs); }
    }
    if (c.n_jobs > 0) {
      store->wait_jobs();
      CUDA_CHECK(cudaMemcpyAsync(cpu_rows_d.p, p.cpu_rows, (size_t)c.job_rows * 4, cudaMemcpyHostToDevice, st));
      accum_f32_ordered<<<(unsigned)(((size_t)M * DIM + 255) / 256), 256, 0, st>>>(cpu_out_d, cpu_rows_d.as<int32_t>(), c.job_rows, M, DIM, acc.as<float>());
    }
  }
  void old_step(int M) {
    for (int l = 0; l < NL; ++l) { front(l, M, false); CUDA_CHECK(cudaStreamSynchronize(st)); experts(l, M, false); }
    CUDA_CHECK(cudaStreamSynchronize(st));
  }
  void new_step(int M, bool tbl_own = false) {
    for (int l = 0; l < NL; ++l) {
      const bool trust = gate.begin();
      front(l, M, true);
      CUDA_CHECK(cudaEventRecord(front_ev, st));
      const int r = trust ? gate.wait(flag_h, [&] { const cudaError_t e = cudaEventQuery(front_ev); if (e != cudaErrorNotReady) CUDA_CHECK(e); return e == cudaSuccess; })
                          : (int)er::kMissing;
      if (r != er::kOk) {  // same absorption as production (must happen 0 times in the test to PASS)
        CUDA_CHECK(cudaEventSynchronize(front_ev));
        k::route_to_host(ids.as<int32_t>(), rw.as<float>(), nullptr, nullptr, M * K, 0, ids_hd, rw_hd, nullptr, nullptr, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        gate.resync(flag_h);
      }
      experts(l, M, true, tbl_own);
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
  }
};

}  // namespace

int main(int argc, char** argv) {
  const int steps = argc > 1 ? std::max(3, atoi(argv[1])) : 24;
  const int NL = argc > 2 ? std::max(1, atoi(argv[2])) : 8;
  const int NE = argc > 3 ? std::max(16, std::min(hs::kMaxE, atoi(argv[3]))) : 48;
  const int threads = argc > 4 ? std::max(1, atoi(argv[4])) : 16;
  const int win_mb = argc > 5 ? std::max(0, atoi(argv[5])) : 48;
  const bool cpufirst = argc > 6 && strcmp(argv[6], "cpufirst") == 0;  // batch-regression mode
  const int NV = cpufirst ? 3 : 2;  // variants: 0 = OLD, 1 = NEW (table on st), 2 = NEWFIX (dedicated table stream — cpufirst only)
  Rig rg(NL, NE, threads, win_mb);
  rg.order = cpufirst ? 1 : 0;
  printf("[early-route] layers %d · experts/layer %d · resident %d · staging %d · cpu threads/node %d · shared-expert window %d MB · order %s\n", NL, NE,
         (int)(0.6 * NL * NE), rg.S, threads, win_mb, cpufirst ? "cpufirst" : "legacy");
  bool pass = true;
  std::vector<float> ref((size_t)MB * DIM), got((size_t)MB * DIM);
  for (int M : {1, 2, 4, 8}) {
    long bad = 0, bad_fix = 0;
    double t_old = 0, t_new = 0, c_old = 0, c_new = 0, t_fix = 0, c_fix = 0;
    int n_t = 0;
    const long ahead0 = rg.ahead, lay0 = rg.layers_new;
    for (int s = 0; s < steps; ++s) {
      fill_bf16<<<(unsigned)(((size_t)M * DIM + 255) / 256), 256, 0, rg.st>>>(rg.x0.as<bf16>(), (size_t)M * DIM, 1000u * (uint32_t)M + (uint32_t)s);
      CUDA_CHECK(cudaStreamSynchronize(rg.st));
      double w[3] = {0, 0, 0}, cc[3] = {0, 0, 0};
      std::vector<float> fix((size_t)MB * DIM);
      for (int pass_i = 0; pass_i < NV; ++pass_i) {
        const int v = (pass_i + s) % NV;  // alternate (cancels order effects — with NV = 2 the same alternation as before)
        const double w0 = now_ms(), c0 = thread_cpu_ms();
        if (v == 0) rg.old_step(M); else rg.new_step(M, v == 2);
        w[v] = now_ms() - w0; cc[v] = thread_cpu_ms() - c0;
        CUDA_CHECK(cudaMemcpy((v == 0 ? ref : v == 1 ? got : fix).data(), rg.acc.p, (size_t)M * DIM * 4, cudaMemcpyDeviceToHost));
      }
      for (size_t i = 0; i < (size_t)M * DIM; ++i) bad += memcmp(&ref[i], &got[i], 4) != 0;
      if (NV == 3) for (size_t i = 0; i < (size_t)M * DIM; ++i) bad_fix += memcmp(&ref[i], &fix[i], 4) != 0;
      if (s >= 2) { t_old += w[0]; t_new += w[1]; c_old += cc[0]; c_new += cc[1]; t_fix += w[2]; c_fix += cc[2]; ++n_t; }  // first two steps = eager run and capture
    }
    pass = pass && bad == 0 && bad_fix == 0;
    const long nl = rg.layers_new - lay0;
    printf("  M=%d: %d steps · NEW vs OLD %s (%ld/%ld floats differ) · step %.3f → %.3f ms (×%.3f) · engine-thread CPU %.3f → %.3f ms · host ahead of front end %ld/%ld layers\n",
           M, steps, bad == 0 ? "bit-exact" : "MISMATCH", bad, (long)steps * M * DIM, t_old / std::max(1, n_t), t_new / std::max(1, n_t),
           t_new > 0 ? t_old / t_new : 0.0, c_old / std::max(1, n_t), c_new / std::max(1, n_t), rg.ahead - ahead0, nl);
    if (NV == 3)  // dedicated-table-stream variant (bit-compared against OLD) — if the hypothesis holds, at M >= 2 NEW < 1 and NEWFIX >= 1
      printf("  M=%d cpufirst: NEWFIX vs OLD %s (%ld floats differ) · step %.3f ms (×%.3f vs OLD · ×%.3f vs NEW) · engine-thread CPU %.3f ms\n", M,
             bad_fix == 0 ? "bit-exact" : "MISMATCH", bad_fix, t_fix / std::max(1, n_t), t_fix > 0 ? t_old / t_fix : 0.0, t_fix > 0 ? t_new / t_fix : 0.0,
             c_fix / std::max(1, n_t));
  }
  const er::Gate& g = rg.gate;
  printf("  gate: ok %ld · resync %ld · missing %ld · untrusted %ld\n", g.n_ok, g.n_resync, g.n_missing, g.n_untrusted);
  pass = pass && g.n_resync == 0 && g.n_missing == 0 && g.n_untrusted == 0 && g.n_ok > 0;
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
