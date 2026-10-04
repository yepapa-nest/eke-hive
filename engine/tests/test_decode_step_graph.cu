// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Verifies HIVE_DECODE_STEP_GRAPH (GPU test).
//   Synthetic layer stack (NL layers, real shapes dim 5120, inter 2304, top-6, swiglu_limit 10, NE experts per layer in the real storage format e2m1+e8m0, ~60 % resident):
//     layer = [front: (layer 0) input x0 / (layer >= 1) bf16(acc) -> rmsnorm] -> act_quant_fp8 -> synthetic router (hash of activation bytes -> distinct top-6 skewed to popular experts, rw)
//          -> acc = 0 -> experts (resident groups + DMA share + CPU misses).
//   (OLD) the same host-synchronised chain as moe_decode_experts: per layer cudaStreamSynchronize -> host plan -> table H2D -> mx_grouped_w13/w2 + accum_bf16_rows_seq
//         (resident -> staging DMA wait -> DMA groups) -> CPU jobs (real ExpertStore pool) -> cpu_rows H2D -> accumulate CPU results.
//         Two CPU accumulation variants: accum_f32_rows (atomic adds, as in production) and fixed row order (same order as the new path — for strict comparison).
//   (NEW) the whole step (NL layers) as one CUDA graph — k::hs_layer_experts (hs_plan + wait kernels) + hs::Dispatcher (real pool) — same order as runtime.cpp forward_batch_step
//         (parameter H2D -> arm -> launch -> one wait -> disarm -> error check).
//   Verdict: per step the last layer's acc (fp32), NEW vs OLD (fixed order), **bitwise** = PASS condition. Differences against OLD (atomic) are counted separately (they can only appear
//         where the addition order may change, i.e. rows with two or more CPU results — reference only). Device plan count logs == host plan (all layers and steps).
//   Timing: wall-clock ms per step, engine (main) thread CPU ms (CLOCK_THREAD_CPUTIME_ID — the baseline spin-syncs per layer).
//   NEW has two variants: the unfused chain and the fused one (HIVE_DECODE_FUSED — moe_decode_fused_dev). Both must match OLD (fixed order) bitwise to PASS.
//   Regression: the first step of a run (M=1 step 0) must finish without error (lazy module loading deadlock 0x2 — fixed by hs_preload).
//   Timeout: launch without waking the dispatcher -> the wait kernel leaves an error marker after the limit (50 ms) and the step finishes (no hang) -> after draining, the next step is bit-identical again.
//   Run: scripts/hive-run.sh "./build-dev/test_decode_step_graph [steps=24] [NL=6] [NE=48] [cpu_threads=16]"
//   Needs: VRAM ~= 0.6*NL*NE x 18.9 MB + 0.5 GB, pinned RAM ~= NL*NE x 18.9 MB.
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
#include "hive/decode_dispatch.h"
#include "hive/decode_handshake.h"
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

// Synthetic router: per row, hash of activation bytes -> K distinct experts (first 3 from a popular pool of E/4 — rows overlap so groups hold several rows), rw
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
// CPU accumulation in fixed row order (host counts) — same formula and order as hs_accum_cpu
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
  Config cfg;
  std::unique_ptr<ExpertStore> store;
  ExpertLayout lay;
  hs::RecOff rec{};
  cudaStream_t st = nullptr, side = nullptr;
  DevBuf x0, inb, tmpb, ones, xq, xs, ids, rw, y, yq, ys, eout, acc, gd_old, rows_old, rw_old, cpu_rows_old;
  // OLD host side
  std::vector<int32_t> ids_h, share;
  std::vector<float> rw_h, a_f, a_s, scratch;
  std::vector<uint8_t> xq_h, xs_h;
  float* cpu_out_old = nullptr; float* cpu_out_old_d = nullptr;
  std::vector<cudaEvent_t> copied, freed;
  std::vector<char> used;
  uint32_t stage_next = 0;
  std::unique_ptr<hs::HostPlan> hp = std::make_unique<hs::HostPlan>();
  std::vector<float> lut;
  // NEW
  hs::BoxLayout bl{};
  uint8_t* box_h = nullptr; uint8_t* box_d = nullptr;
  hs::Params* prm_h = nullptr;
  DevBuf prm_d, misc_d, cnt_d, gd_d, rows_d, rw_d, cpu_rows_d, slot_d;
  float* cpu_out_new = nullptr; float* cpu_out_new_d = nullptr;
  std::vector<float> a_f2, a_s2, scratch2;
  std::unique_ptr<hs::Dispatcher> disp;
  uint32_t seq_base = 0;
  cudaGraphExec_t exec[2][MB + 1] = {};  // [D1 fused or not][M]
  cudaEvent_t done_evt = nullptr;
  long plan_mismatch = 0;

  Rig(int nl, int ne, int cpu_threads) : NL(nl), NE(ne) {
    cfg.dim = DIM; cfg.moe_inter = INTER; cfg.n_routed = NE; cfg.n_act = K; cfg.swiglu_limit = LIMIT; cfg.dspark_experts = std::min(NE, 128);
    lay = ExpertLayout::make(DIM, INTER);
    rec = hs::RecOff{lay.w1, lay.s1, lay.w3, lay.s3, lay.w2, lay.s2, lay.total};
    const int n_res = (int)(0.6 * NL * NE);
    store = std::make_unique<ExpertStore>(cfg, NL, (size_t)n_res * lay.total, cpu_threads, 0);
    S = store->staging_slots();
    // Fill expert half-records (two nodes) randomly — weight nibbles arbitrary, scales 2^-9..2^-3
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
    // Buffers
    const int Mrows = MB * K;
    x0.alloc((size_t)MB * DIM * 2); inb.alloc((size_t)MB * DIM * 2); tmpb.alloc((size_t)MB * DIM * 2); ones.alloc((size_t)DIM * 2);
    fill_ones<<<(DIM + 255) / 256, 256, 0, st>>>(ones.as<bf16>(), DIM);
    xq.alloc((size_t)MB * DIM); xs.alloc((size_t)MB * DIM / 32); ids.alloc((size_t)MB * K * 4); rw.alloc((size_t)MB * K * 4);
    y.alloc((size_t)Mrows * INTER * 2); yq.alloc((size_t)Mrows * INTER); ys.alloc((size_t)Mrows * INTER / 32); eout.alloc((size_t)Mrows * DIM * 2);
    acc.alloc((size_t)MB * DIM * 4);
    gd_old.alloc(sizeof(k::GroupDesc) * 2 * Mrows); rows_old.alloc((size_t)Mrows * 4); rw_old.alloc((size_t)Mrows * 4); cpu_rows_old.alloc((size_t)Mrows * 4);
    ids_h.resize(MB * K); rw_h.resize(MB * K); xq_h.resize((size_t)MB * DIM); xs_h.resize((size_t)MB * DIM / 32);
    a_f.resize((size_t)MB * DIM); a_s.resize((size_t)MB * DIM / 32); scratch.resize((size_t)Mrows * ExpertStore::job_scratch_floats(INTER));
    cpu_out_old = mapped<float>((size_t)Mrows * DIM, &cpu_out_old_d);
    copied.resize(S); freed.resize(S); used.assign(S, 0);
    for (int i = 0; i < S; ++i) { CUDA_CHECK(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&freed[i], cudaEventDisableTiming)); }
    lut.resize(256);
    for (int i = 0; i < 256; ++i) lut[i] = e4m3_to_f32((uint8_t)i);
    // DMA share table: frac 0.25, decode_gpu_share 1 (same formula as P.share in runtime.cpp)
    share.resize(hs::kMaxE + 1);
    const float frac = 0.25f; const int dma_cap = 8, gshare = 1;
    for (int n = 0; n <= hs::kMaxE; ++n) { const int n_miss = n; share[n] = n_miss >= 3 ? std::min(std::min(dma_cap, S), std::max(gshare, (int)(frac * n_miss + 0.5f))) : 0; }
    // NEW
    bl = hs::BoxLayout::make(MB, K, DIM, NL);
    box_h = mapped<uint8_t>(bl.total, &box_d);
    CUDA_CHECK(cudaHostAlloc((void**)&prm_h, sizeof(hs::Params), cudaHostAllocDefault));
    memset((void*)prm_h, 0, sizeof(hs::Params));
    for (int n = 0; n <= hs::kMaxE; ++n) prm_h->share[n] = share[n];
    prm_d.alloc(sizeof(hs::Params)); misc_d.alloc(sizeof(hs::DevMisc)); CUDA_CHECK(cudaMemset(misc_d.p, 0, sizeof(hs::DevMisc)));
    cnt_d.alloc(sizeof(hs::LayerCounts));
    gd_d.alloc(sizeof(k::GroupDesc) * (size_t)(MB * K + hs::kMaxDma)); rows_d.alloc((size_t)Mrows * 4); rw_d.alloc((size_t)Mrows * 4); cpu_rows_d.alloc((size_t)Mrows * 4);
    slot_d.alloc(store->slot_table_size() * 4);
    CUDA_CHECK(cudaMemcpy(slot_d.p, store->slot_table(), store->slot_table_size() * 4, cudaMemcpyHostToDevice));
    cpu_out_new = mapped<float>((size_t)Mrows * DIM, &cpu_out_new_d);
    a_f2.resize((size_t)MB * DIM); a_s2.resize((size_t)MB * DIM / 32); scratch2.resize(scratch.size());
    hs::Dispatcher::Io io;
    io.store = store.get();
    io.ctrl = reinterpret_cast<hs::Ctrl*>(box_h + bl.ctrl); io.hdr = reinterpret_cast<const hs::PostHdr*>(box_h + bl.hdr);
    io.jobs = reinterpret_cast<const hs::JobItem*>(box_h + bl.jobs); io.xq = box_h + bl.xq; io.xs = box_h + bl.xs;
    io.a_f = a_f2.data(); io.a_s = a_s2.data(); io.scratch = scratch2.data(); io.cpu_out = cpu_out_new;
    io.dim = DIM; io.inter = INTER; io.Mb = MB; io.Jcap = bl.Jcap; io.nL = NL;
    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    io.thread_init = [dev] { cudaSetDevice(dev); };
    io.dma_copy = [this](int l, int e, int si) { store->copy_to_staging(si, l, e, side); };
    uint32_t* flag = &reinterpret_cast<hs::DevMisc*>(misc_d.p)->dma_flag;
    io.dma_signal = [this, flag](uint32_t seq) { k::hs_signal(flag, seq, side); };
    k::hs_preload(side, st);  // before the first step, as in runtime.cpp (fix for the lazy module loading deadlock — decode_handshake.h)
    disp = std::make_unique<hs::Dispatcher>(std::move(io));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_evt, cudaEventBlockingSync | cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamSynchronize(st));
  }
  ~Rig() {
    disp.reset();
    for (auto& row : exec) for (auto& e : row) if (e) cudaGraphExecDestroy(e);
    for (auto e : copied) cudaEventDestroy(e);
    for (auto e : freed) cudaEventDestroy(e);
    if (done_evt) cudaEventDestroy(done_evt);
    cudaFreeHost(cpu_out_old); cudaFreeHost(cpu_out_new); cudaFreeHost(box_h); cudaFreeHost(prm_h);
    cudaStreamDestroy(st); cudaStreamDestroy(side);
  }

  void front(int l, int M) {
    if (l == 0) CUDA_CHECK(cudaMemcpyAsync(inb.p, x0.p, (size_t)M * DIM * 2, cudaMemcpyDeviceToDevice, st));
    else {
      k::f32_to_bf16(acc.as<float>(), M * DIM, tmpb.as<bf16>(), st);
      k::rmsnorm(tmpb.as<bf16>(), ones.as<bf16>(), 1e-6f, M, DIM, inb.as<bf16>(), st);
    }
    k::act_quant_fp8(inb.as<bf16>(), M, DIM, xq.as<uint8_t>(), xs.as<uint8_t>(), st);
    router_synth<<<M, 32, 0, st>>>(xq.as<uint8_t>(), M, DIM, NE, l, ids.as<int32_t>(), rw.as<float>());
    CUDA_CHECK(cudaMemsetAsync(acc.p, 0, (size_t)M * DIM * 4, st));
  }
  // Baseline host-synchronised chain (same order as the GPU and CPU shares in moe_decode_experts). ordered = CPU accumulation in fixed row order
  void old_layer(int l, int M, bool ordered) {
    CUDA_CHECK(cudaStreamSynchronize(st));  // baseline per-layer sync
    CUDA_CHECK(cudaMemcpy(ids_h.data(), ids.p, (size_t)M * K * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rw_h.data(), rw.p, (size_t)M * K * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xq_h.data(), xq.p, (size_t)M * DIM, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xs_h.data(), xs.p, (size_t)M * DIM / 32, cudaMemcpyDeviceToHost));
    hs::HostPlan& p = *hp;
    hs::plan_host(ids_h.data(), rw_h.data(), M, K, NE, store->slot_table() + (size_t)l * store->E(), share.data(), p, store->dev_rec(0), store->staging_rec(0),
                  rec, (int)(stage_next % (uint32_t)S), S);
    const hs::LayerCounts& c = p.counts;
    // DMA first (side stream) — wait for slot release -> copy -> copy event
    for (int i = 0; i < c.n_dma; ++i) {
      const int si = p.dma[i].si;
      if (used[si]) CUDA_CHECK(cudaStreamWaitEvent(side, freed[si], 0));
      store->copy_to_staging(si, l, p.dma[i].e, side);
      CUDA_CHECK(cudaEventRecord(copied[si], side));
    }
    // Start CPU jobs
    std::vector<ExpertStore::Job> jobs(c.n_jobs);
    if (c.n_jobs > 0) {
      for (int m = 0; m < M; ++m) {
        for (int d = 0; d < DIM; ++d) a_f[(size_t)m * DIM + d] = lut[xq_h[(size_t)m * DIM + d]];
        for (int b = 0; b < DIM / 32; ++b) a_s[(size_t)m * (DIM / 32) + b] = e8m0_to_f32(xs_h[(size_t)m * (DIM / 32) + b]);
      }
      const size_t sn = ExpertStore::job_scratch_floats(INTER);
      for (int j = 0; j < c.n_jobs; ++j) {
        const hs::JobItem& it = p.jobs[j];
        ExpertStore::Job& J = jobs[j];
        J = ExpertStore::Job{}; J.layer = l; J.e = it.e; J.R = it.R; J.scratch = scratch.data() + (size_t)it.row0 * sn;
        for (int r = 0; r < it.R; ++r) {
          J.a_f[r] = a_f.data() + (size_t)it.m[r] * DIM; J.a_s[r] = a_s.data() + (size_t)it.m[r] * (DIM / 32); J.route_w[r] = it.rw[r];
          J.out[r] = cpu_out_old + (size_t)(it.row0 + r) * DIM;
        }
      }
      store->start_jobs(jobs);
    }
    // Table H2D (groups: resident, then DMA)
    const int ng = c.ng_hit + c.ng_str, nr = c.R_hit + c.R_str;
    std::vector<k::GroupDesc> gd(ng);
    for (int g = 0; g < c.ng_hit; ++g) gd[g] = p.gd_hit[g];
    for (int g = 0; g < c.ng_str; ++g) gd[c.ng_hit + g] = p.gd_dma[g];
    if (ng > 0) {
      CUDA_CHECK(cudaMemcpyAsync(gd_old.p, gd.data(), sizeof(k::GroupDesc) * ng, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(rows_old.p, p.g_rows, (size_t)nr * 4, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(rw_old.p, p.g_rw, (size_t)nr * 4, cudaMemcpyHostToDevice, st));
    }
    auto run_groups = [&](int g0, int n_groups, int r0, int n_rows) {
      if (n_groups <= 0) return;
      const k::GroupDesc* gdev = gd_old.as<k::GroupDesc>() + g0;
      k::mx_grouped_w13(gdev, n_groups, xq.as<uint8_t>(), xs.as<uint8_t>(), INTER, DIM, rw_old.as<float>(), LIMIT, y.as<bf16>(), st, rows_old.as<int32_t>(),
                        yq.as<uint8_t>(), ys.as<uint8_t>());
      k::mx_grouped_w2(gdev, n_groups, yq.as<uint8_t>(), ys.as<uint8_t>(), DIM, INTER, eout.as<bf16>(), st);
      k::accum_bf16_rows_seq(eout.as<bf16>() + (size_t)r0 * DIM, rows_old.as<int32_t>() + r0, n_rows, M, DIM, acc.as<float>(), st);
    };
    run_groups(0, c.ng_hit, 0, c.R_hit);
    for (int i = 0; i < c.n_dma; ++i) CUDA_CHECK(cudaStreamWaitEvent(st, copied[p.dma[i].si], 0));
    run_groups(c.ng_hit, c.ng_str, c.R_hit, c.R_str);
    for (int i = 0; i < c.n_dma; ++i) { CUDA_CHECK(cudaEventRecord(freed[p.dma[i].si], st)); used[p.dma[i].si] = 1; }
    stage_next += (uint32_t)c.n_dma;
    if (c.n_jobs > 0) {
      store->wait_jobs();
      CUDA_CHECK(cudaMemcpyAsync(cpu_rows_old.p, p.cpu_rows, (size_t)c.job_rows * 4, cudaMemcpyHostToDevice, st));
      if (ordered)
        accum_f32_ordered<<<(unsigned)(((size_t)M * DIM + 255) / 256), 256, 0, st>>>(cpu_out_old_d, cpu_rows_old.as<int32_t>(), c.job_rows, M, DIM, acc.as<float>());
      else k::accum_f32_rows(cpu_out_old_d, cpu_rows_old.as<int32_t>(), c.job_rows, DIM, acc.as<float>(), st);
    }
  }
  void old_step(int M, bool ordered) {
    for (int l = 0; l < NL; ++l) { front(l, M); old_layer(l, M, ordered); }
    CUDA_CHECK(cudaStreamSynchronize(st));
  }
  k::HsLayerArgs args(int l, int M, bool fused) {
    k::HsLayerArgs a{};
    k::HsPlanArgs& p = a.plan;
    p.ids = ids.as<int32_t>(); p.rw = rw.as<float>(); p.M = M; p.k = K; p.E = NE; p.l = l;
    p.slot_row = slot_d.as<int32_t>() + (size_t)l * store->E();
    p.slots_base = store->dev_rec(0); p.staging_base = store->staging_rec(0); p.rec = rec;
    p.xq = xq.as<uint8_t>(); p.xs = xs.as<uint8_t>(); p.dim = DIM;
    p.prm = reinterpret_cast<const hs::Params*>(prm_d.p); p.misc = reinterpret_cast<hs::DevMisc*>(misc_d.p);
    p.cnt = reinterpret_cast<hs::LayerCounts*>(cnt_d.p); p.gd = gd_d.as<k::GroupDesc>(); p.Gh = MB * K; p.Gd = hs::kMaxDma;
    p.g_rows = rows_d.as<int32_t>(); p.g_rw = rw_d.as<float>(); p.cpu_rows = cpu_rows_d.as<int32_t>();
    p.ctrl = reinterpret_cast<hs::Ctrl*>(box_d + bl.ctrl); p.hdr = reinterpret_cast<hs::PostHdr*>(box_d + bl.hdr);
    p.jobs = reinterpret_cast<hs::JobItem*>(box_d + bl.jobs); p.bxq = box_d + bl.xq; p.bxs = box_d + bl.xs;
    p.route_log_l = reinterpret_cast<int32_t*>(box_d + bl.route_log) + (size_t)l * MB * K;
    p.counts_log_l = reinterpret_cast<hs::LayerCounts*>(box_d + bl.counts_log) + l;
    p.time_log_l = reinterpret_cast<unsigned long long*>(box_d + bl.time_log) + 2 * (size_t)l;
    a.I = INTER; a.limit = LIMIT; a.y = y.as<bf16>(); a.yq = yq.as<uint8_t>(); a.ys = ys.as<uint8_t>(); a.eout = eout.as<bf16>(); a.acc = acc.as<float>();
    a.cpu_out_d = cpu_out_new_d;
    a.fused = fused;
    return a;
  }
  // One step of the new path. arm=false does not wake the dispatcher (timeout test). Returns Ctrl.err | cpu_err<<8
  uint32_t new_step(int M, bool arm, unsigned long long timeout_ns, bool fused = false) {
    hs::Ctrl* ctrl = reinterpret_cast<hs::Ctrl*>(box_h + bl.ctrl);
    seq_base += (uint32_t)NL + 1u;
    prm_h->seq_base = seq_base; prm_h->n_staging = S; prm_h->stage_base = (int)(stage_next % (uint32_t)S); prm_h->timeout_ns = timeout_ns;
    CUDA_CHECK(cudaMemcpyAsync(prm_d.p, prm_h, sizeof(hs::Params), cudaMemcpyHostToDevice, st));
    ctrl->err = 0; ctrl->err_seq = 0; ctrl->cpu_err = 0;
    for (int si = 0; si < S; ++si) if (used[si]) CUDA_CHECK(cudaStreamWaitEvent(side, freed[si], 0));
    if (arm) disp->arm();
    cudaGraphExec_t& ex = exec[fused ? 1 : 0][M];
    if (!ex) {
      cudaGraph_t g = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
      k::hs_step_reset(reinterpret_cast<hs::DevMisc*>(misc_d.p), st);
      for (int l = 0; l < NL; ++l) { front(l, M); k::hs_layer_experts(args(l, M, fused), st); }
      CUDA_CHECK(cudaStreamEndCapture(st, &g));
      CUDA_CHECK(cudaGraphInstantiate(&ex, g, 0));
      CUDA_CHECK(cudaGraphDestroy(g));
    }
    CUDA_CHECK(cudaGraphLaunch(ex, st));
    CUDA_CHECK(cudaEventRecord(done_evt, st));
    CUDA_CHECK(cudaEventSynchronize(done_evt));  // one blocking wait at step end, as in runtime.cpp
    if (arm) disp->disarm();
    const hs::LayerCounts* cl = reinterpret_cast<const hs::LayerCounts*>(box_h + bl.counts_log);
    uint32_t n_dma = 0;
    for (int l = 0; l < NL; ++l) n_dma += (uint32_t)cl[l].n_dma;
    for (uint32_t j = 0; j < n_dma; ++j) { const int si = (int)((stage_next + j) % (uint32_t)S); CUDA_CHECK(cudaEventRecord(freed[si], st)); used[si] = 1; }
    stage_next += n_dma;
    return ctrl->err | (ctrl->cpu_err << 8);
  }
  // Device plan of the new path == host plan (via logs)
  void check_plans(int M) {
    const int32_t* rl = reinterpret_cast<const int32_t*>(box_h + bl.route_log);
    const hs::LayerCounts* cl = reinterpret_cast<const hs::LayerCounts*>(box_h + bl.counts_log);
    for (int l = 0; l < NL; ++l) {
      hs::plan_host(rl + (size_t)l * MB * K, nullptr, M, K, NE, store->slot_table() + (size_t)l * store->E(), share.data(), *hp);
      plan_mismatch += memcmp(&hp->counts, &cl[l], sizeof(hs::LayerCounts)) != 0;
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  const int steps = argc > 1 ? std::max(2, atoi(argv[1])) : 24;
  const int NL = argc > 2 ? std::max(1, atoi(argv[2])) : 6;
  const int NE = argc > 3 ? std::max(16, std::min(hs::kMaxE, atoi(argv[3]))) : 48;
  const int threads = argc > 4 ? std::max(1, atoi(argv[4])) : 16;
  Rig rg(NL, NE, threads);
  printf("[step-graph] layers %d · experts/layer %d · resident %d · staging %d · cpu threads/node %d\n", NL, NE, (int)(0.6 * NL * NE), rg.S, threads);
  bool pass = true;
  std::vector<float> ref((size_t)MB * DIM), got((size_t)MB * DIM), atom((size_t)MB * DIM);
  const unsigned long long tmo = 10ull * 1000000000ull;
  for (int fz = 0; fz < 2; ++fz)
  for (int M : {1, 2, 4, 8}) {
    const bool fused = fz == 1;
    if (fused && !k::moe_decode_fused_ok(DIM, INTER, M, MB * K)) continue;
    long bits_bad = 0, atomic_diff = 0;
    double t_old = 0, t_new = 0, c_old = 0, c_new = 0;
    int n_t = 0;
    for (int s = 0; s < steps; ++s) {
      fill_bf16<<<(unsigned)(((size_t)M * DIM + 255) / 256), 256, 0, rg.st>>>(rg.x0.as<bf16>(), (size_t)M * DIM, 1000u * (uint32_t)M + (uint32_t)s);
      CUDA_CHECK(cudaStreamSynchronize(rg.st));
      // OLD (atomic), OLD (fixed order), NEW — same input, same residency
      rg.old_step(M, false);
      CUDA_CHECK(cudaMemcpy(atom.data(), rg.acc.p, (size_t)M * DIM * 4, cudaMemcpyDeviceToHost));
      double w0 = now_ms(), c0 = thread_cpu_ms();
      rg.old_step(M, true);
      const double w1 = now_ms(), c1 = thread_cpu_ms();
      CUDA_CHECK(cudaMemcpy(ref.data(), rg.acc.p, (size_t)M * DIM * 4, cudaMemcpyDeviceToHost));
      const double w2 = now_ms(), c2 = thread_cpu_ms();
      const uint32_t err = rg.new_step(M, true, tmo, fused);
      const double w3 = now_ms(), c3 = thread_cpu_ms();
      CUDA_CHECK(cudaMemcpy(got.data(), rg.acc.p, (size_t)M * DIM * 4, cudaMemcpyDeviceToHost));
      if (err) { printf("  %s M=%d step %d: handshake error 0x%x\n", fused ? "fused" : "chain", M, s, err); pass = false; }
      else rg.check_plans(M);  // error steps deliberately shorten the plans of later layers (excluded from comparison — the error itself is a FAIL)
      for (size_t i = 0; i < (size_t)M * DIM; ++i) {
        bits_bad += memcmp(&ref[i], &got[i], 4) != 0;
        atomic_diff += memcmp(&atom[i], &got[i], 4) != 0;
      }
      bool finite = true;
      for (size_t i = 0; i < (size_t)M * DIM; ++i) finite = finite && std::isfinite(got[i]);
      if (!finite) { printf("  M=%d step %d: non-finite output (fixture scale)\n", M, s); }
      if (s >= 2) { t_old += w1 - w0; t_new += w3 - w2; c_old += c1 - c0; c_new += c3 - c2; ++n_t; }
    }
    pass = pass && bits_bad == 0;
    printf("  %s M=%d: %d steps · NEW vs OLD(ordered CPU accum) %s (%ld/%ld floats differ) · vs OLD(atomic accum_f32_rows) %ld differ · "
           "step %.3f → %.3f ms · engine-thread CPU %.3f → %.3f ms\n",
           fused ? "NEW=D1 fused" : "NEW=chain", M, steps, bits_bad == 0 ? "bit-exact" : "MISMATCH", bits_bad, (long)steps * M * DIM, atomic_diff, t_old / std::max(1, n_t), t_new / std::max(1, n_t),
           c_old / std::max(1, n_t), c_new / std::max(1, n_t));
  }
  printf("  device plan counts == host plan: %s (%ld mismatching layers)\n", rg.plan_mismatch == 0 ? "yes" : "NO", rg.plan_mismatch);
  pass = pass && rg.plan_mismatch == 0;
  // Timeout: launch without waking the dispatcher -> the wait kernel sets an error marker after the limit; the step finishes
  {
    const int M = 8;
    fill_bf16<<<(unsigned)(((size_t)M * DIM + 255) / 256), 256, 0, rg.st>>>(rg.x0.as<bf16>(), (size_t)M * DIM, 777u);
    CUDA_CHECK(cudaStreamSynchronize(rg.st));
    const double t0 = now_ms();
    const uint32_t err = rg.new_step(M, false, 50ull * 1000000ull);
    const double dt = now_ms() - t0;
    rg.disp->arm(); rg.disp->disarm();  // drain remaining posts (no fallback path — in production the dispatcher is always awake and handles posts, even late)
    CUDA_CHECK(cudaStreamSynchronize(rg.side));
    const bool ok = (err & 3u) != 0 && dt < 2000.0;
    printf("  timeout: unarmed dispatcher → err 0x%x after %.1f ms (limit 50 ms/wait) %s\n", err, dt, ok ? "ok" : "FAIL");
    pass = pass && ok;
    rg.old_step(M, true);
    CUDA_CHECK(cudaMemcpy(ref.data(), rg.acc.p, (size_t)M * DIM * 4, cudaMemcpyDeviceToHost));
    const uint32_t err2 = rg.new_step(M, true, tmo);
    CUDA_CHECK(cudaMemcpy(got.data(), rg.acc.p, (size_t)M * DIM * 4, cudaMemcpyDeviceToHost));
    const bool ok2 = err2 == 0 && memcmp(ref.data(), got.data(), (size_t)M * DIM * 4) == 0;
    printf("  after timeout: next step %s\n", ok2 ? "bit-exact" : "FAIL");
    pass = pass && ok2;
  }
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
