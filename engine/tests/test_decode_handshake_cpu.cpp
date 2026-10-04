// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_STEP_GRAPH (G1) — CPU test (fake CUDA/NUMA, no GPU). Built and run by tools/test_decode_handshake_cpu.py.
//
//  1. sort   : hs::sort_desc_by_count (shared by the device plan and the host replay) == std::sort (same comparator) —
//              random lengths 0..512, many ties (> 16 = introsort path).
//  2. plan   : hs::plan_host (= plan_lists + build_tables — the same function hs_plan's thread 0 runs) == a transcription
//              of the legacy Runtime::moe_decode_experts classification (legacy_plan) — group table (weight pointers ·
//              row0 · n), g_rows/g_rw, CPU jobs (expert · rows · weights · row0), cpu_rows, DMA slot order, staging
//              advance. The Python side checks textually that the transcribed lines are still verbatim in runtime.cpp.
//  3. handshake: real ExpertStore CPU pool + hs::Dispatcher + fake GPU thread (same mailbox protocol as hs_plan and the
//              wait kernels: write tables and activations -> post_seq release -> wait for the DMA signal / done_seq
//              acquire). Per post: cpu_out rows == cpu::expert_forward (bitwise), DMA copy order == plan, repeated
//              arm/disarm, steps without a post, slow dispatcher -> fake GPU timeout -> disarm waits until the end,
//              callback exception -> cpu_err · done still written (GPU does not hang), sleep-polling mode.
//     Under TSAN (HIVE_TEST_SANITIZER=thread) every access to the mailbox, results and log is race-checked.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hive/decode_dispatch.h"
#include "hive/decode_handshake.h"
#include "hive/expert_cpu.h"
#include "hive/expert_store.h"
#include "hive/safetensors.h"
#include "hive/clock.h"

namespace hive { void fake_ckpt_put(const std::string&, const uint8_t*, size_t, const std::string& = "U8"); }
using namespace hive;
using MsDur = std::chrono::milliseconds;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
static double now_ms() { return hive::mono_ms(); }

// ---- 1. sort ----------------------------------------------------------------------------------------------------------------
static void test_sort(int iters) {
  std::mt19937 rng(7);
  long cases = 0, big = 0;
  std::vector<int32_t> cnt(hs::kMaxE);
  for (int it = 0; it < iters; ++it) {
    const int n = it % 7 == 0 ? (int)(rng() % 512) + 1 : (int)(rng() % 64);
    const int span = 1 + (int)(rng() % (it % 3 == 0 ? 3 : 9));  // row counts 1..span — many ties
    std::vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = i;                       // the legacy miss_e is in ascending expert order
    if (it % 5 == 0) std::shuffle(a.begin(), a.end(), rng);    // random order too (algorithmic equivalence)
    for (int i = 0; i < n; ++i) cnt[i] = 1 + (int)(rng() % span);
    if (it % 11 == 0) for (int i = 0; i < n; ++i) cnt[i] = 1 + (n - i) % span;  // descending/periodic patterns (worst case for partitioning)
    std::vector<int> ref = a, got = a;
    std::sort(ref.begin(), ref.end(), [&](int x, int y) { return cnt[x] > cnt[y]; });
    hs::sort_desc_by_count(got.data(), n, cnt.data());
    EXPECT(ref == got, "sort mismatch n=%d span=%d it=%d", n, span, it);
    ++cases; big += n > 16;
    if (fails > 5) break;
  }
  printf("handshake CPU: sort_desc_by_count == std::sort in %ld cases (%ld with n > 16)\n", cases, big);
}

// ---- 2. plan vs legacy transcription -----------------------------------------------------------------------------------------
// Verbatim copy of the classification/table part of the legacy Runtime::moe_decode_experts (engine/src/runtime.cpp),
// without statistics and CUDA launches. The Python side compares the original lines.
struct LegacyOut {
  std::vector<k::GroupDesc> gd; int ng_hit = 0;
  std::vector<int> g_rows; std::vector<float> g_rw;
  struct J { int e, R; int m[8]; float rw[8]; int row0; };
  std::vector<J> jobs; std::vector<int> cpu_rows, str_si, str_e, cpu_e, hit_e;
};
static void legacy_plan(const int32_t* route_ids, const float* rw_h, int M, int k, int E, const int32_t* slot_row, float frac, int decode_gpu_share,
                        int staging_slots, uint32_t& stage_next_, const uint8_t* slots_base, const uint8_t* staging_base, const ExpertLayout& lay, LegacyOut& o) {
  const int R = M * k;
  int cnt[512 + 1] = {0};
  for (int i = 0; i < R; ++i) ++cnt[route_ids[i] + 1];
  for (int e = 0; e < E; ++e) cnt[e + 1] += cnt[e];
  int fill[512];
  std::memcpy(fill, cnt, sizeof(int) * E);
  std::vector<int> rows_by_e(R);
  for (int i = 0; i < R; ++i) rows_by_e[fill[route_ids[i]]++] = i;
  auto n_of = [&](int e) { return cnt[e + 1] - cnt[e]; };
  int hit_e[512], str_e[512], cpu_e[512];
  int n_hit_e = 0, n_str_e = 0, n_cpu_e = 0;
  int n_miss = 0;
  for (int e = 0; e < E; ++e) if (n_of(e) > 0 && slot_row[e] < 0) ++n_miss;
  const int dma_cap = 8;
  int gpu_share_left = 0;
  if (n_miss >= 3) gpu_share_left = std::min(std::min(dma_cap, staging_slots), std::max(decode_gpu_share, (int)(frac * n_miss + 0.5f)));
  int miss_e[512];
  int n_miss_e = 0;
  for (int e = 0; e < E; ++e) {
    if (n_of(e) == 0) continue;
    if (slot_row[e] >= 0) hit_e[n_hit_e++] = e;
    else miss_e[n_miss_e++] = e;
  }
  std::sort(miss_e, miss_e + n_miss_e, [&](int a, int b) { return n_of(a) > n_of(b); });
  for (int i = 0; i < n_miss_e; ++i) {
    if (gpu_share_left > 0) { --gpu_share_left; str_e[n_str_e++] = miss_e[i]; }
    else cpu_e[n_cpu_e++] = miss_e[i];
  }
  int goff = 0;
  o.gd.clear(); o.g_rows.assign(R, -1); o.g_rw.assign(R, 0.f);
  auto add_group = [&](int e, const uint8_t* rec) {
    for (int i0 = cnt[e]; i0 < cnt[e + 1]; i0 += 8) {
      k::GroupDesc d{};
      d.w1 = rec + lay.w1; d.s1 = rec + lay.s1; d.w3 = rec + lay.w3; d.s3 = rec + lay.s3; d.w2 = rec + lay.w2; d.s2 = rec + lay.s2;
      d.row0 = goff; d.n = std::min(8, cnt[e + 1] - i0);
      for (int i = i0; i < i0 + d.n; ++i) { const int mj = rows_by_e[i]; o.g_rows[goff] = mj / k; o.g_rw[goff] = rw_h[mj]; ++goff; }
      o.gd.push_back(d);
    }
  };
  o.hit_e.assign(hit_e, hit_e + n_hit_e);
  for (int ei = 0; ei < n_hit_e; ++ei) add_group(hit_e[ei], slots_base + (size_t)slot_row[hit_e[ei]] * lay.total);
  o.ng_hit = (int)o.gd.size();
  o.str_si.clear(); o.str_e.assign(str_e, str_e + n_str_e); o.cpu_e.assign(cpu_e, cpu_e + n_cpu_e);
  for (int ei = 0; ei < n_str_e; ++ei) {
    const int si = stage_next_++ % staging_slots;
    o.str_si.push_back(si);
    add_group(str_e[ei], staging_base + (size_t)si * lay.total);
  }
  o.jobs.clear(); o.cpu_rows.clear();
  int job_rows = 0;
  for (int ei = 0; ei < n_cpu_e; ++ei) {
    const int e = cpu_e[ei];
    for (int i0 = cnt[e]; i0 < cnt[e + 1]; i0 += ExpertStore::kMaxRows) {
      LegacyOut::J jb{};
      jb.e = e; jb.R = 0; jb.row0 = job_rows;
      for (int i = i0; i < cnt[e + 1] && jb.R < ExpertStore::kMaxRows; ++i) {
        const int mj = rows_by_e[i], m = mj / k;
        jb.m[jb.R] = m; jb.rw[jb.R] = rw_h[mj];
        o.cpu_rows.push_back(m); ++job_rows;
        ++jb.R;
      }
      o.jobs.push_back(jb);
    }
  }
}
// DMA share table of forward_batch_step (the P.share line in runtime.cpp — the Python side compares the original)
static void make_share(float frac, int decode_gpu_share, int staging_slots, int32_t* share) {
  const int dma_cap = 8;
  for (int n = 0; n <= hs::kMaxE; ++n) {
    const int n_miss = n;
    share[n] = n_miss >= 3 ? std::min(std::min(dma_cap, staging_slots), std::max(decode_gpu_share, (int)(frac * n_miss + 0.5f))) : 0;
  }
}
static bool same_gd(const k::GroupDesc& a, const k::GroupDesc& b) {
  return a.w1 == b.w1 && a.s1 == b.s1 && a.w3 == b.w3 && a.s3 == b.s3 && a.w2 == b.w2 && a.s2 == b.s2 && a.row0 == b.row0 && a.n == b.n;
}
static void test_plan(int iters) {
  std::mt19937 rng(99);
  auto hp = std::make_unique<hs::HostPlan>();
  const ExpertLayout lay = ExpertLayout::make(5120, 2304);
  const hs::RecOff rec{lay.w1, lay.s1, lay.w3, lay.s3, lay.w2, lay.s2, lay.total};
  const uint8_t* slots_base = reinterpret_cast<const uint8_t*>(uintptr_t(1) << 40);
  const uint8_t* staging_base = reinterpret_cast<const uint8_t*>(uintptr_t(3) << 40);
  std::vector<int32_t> share(hs::kMaxE + 1);
  long cases = 0, with_dma = 0, with_cpu = 0, big_miss = 0;
  uint32_t stage_next = 5;
  for (int it = 0; it < iters; ++it) {
    const int E = it % 4 == 0 ? 384 : 8 + (int)(rng() % 120);
    const int k = std::min(E, 1 + (int)(rng() % 8));
    const int M = 1 + (int)(rng() % (it % 9 == 0 ? 64 : 8));
    if (M * k > hs::kMaxR) continue;
    std::vector<int32_t> ids(M * k), slot(E);
    std::vector<float> rw(M * k);
    const int pool = std::max(k, (int)(rng() % E) + 1);  // skew (drawing from a small pool raises per-expert row counts)
    for (int m = 0; m < M; ++m) {
      std::vector<int> pick;
      while ((int)pick.size() < k) { const int e = (int)(rng() % pool); if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
      for (int j = 0; j < k; ++j) { ids[m * k + j] = pick[j]; rw[m * k + j] = std::ldexp((float)(rng() % 1000 + 1), -10); }
    }
    const double pres = (rng() % 100) / 100.0;
    for (int e = 0; e < E; ++e) slot[e] = (rng() % 1000) / 1000.0 < pres ? (int)(rng() % 4000) : -1;
    const float frac = 0.1f + 0.05f * (float)(rng() % 15) + (it % 3 == 0 ? 1e-7f * (float)(rng() % 50) : 0.f);
    const int gshare = (int)(rng() % 3), S = 1 + (int)(rng() % 16);
    make_share(frac, gshare, S, share.data());
    LegacyOut L;
    uint32_t sn_legacy = stage_next;
    legacy_plan(ids.data(), rw.data(), M, k, E, slot.data(), frac, gshare, S, sn_legacy, slots_base, staging_base, lay, L);
    hs::plan_host(ids.data(), rw.data(), M, k, E, slot.data(), share.data(), *hp, slots_base, staging_base, rec, (int)(stage_next % (uint32_t)S), S);
    const hs::HostPlan& p = *hp;
    const hs::LayerCounts& c = p.counts;
    bool ok = c.ng_hit == L.ng_hit && c.ng_hit + c.ng_str == (int)L.gd.size() && c.n_jobs == (int)L.jobs.size() && c.job_rows == (int)L.cpu_rows.size() &&
              c.n_dma == (int)L.str_e.size() && p.n_cpu_e == (int)L.cpu_e.size() && p.n_hit_e == (int)L.hit_e.size();
    for (int g = 0; ok && g < c.ng_hit; ++g) ok = same_gd(p.gd_hit[g], L.gd[g]);
    for (int g = 0; ok && g < c.ng_str; ++g) ok = same_gd(p.gd_dma[g], L.gd[c.ng_hit + g]);
    for (int i = 0; ok && i < c.R_hit + c.R_str; ++i) ok = p.g_rows[i] == L.g_rows[i] && std::memcmp(&p.g_rw[i], &L.g_rw[i], 4) == 0;
    for (int j = 0; ok && j < c.n_jobs; ++j) {
      const hs::JobItem& a = p.jobs[j];
      const LegacyOut::J& b = L.jobs[j];
      ok = a.e == b.e && a.R == b.R && a.row0 == b.row0;
      for (int r = 0; ok && r < a.R; ++r) ok = a.m[r] == b.m[r] && std::memcmp(&a.rw[r], &b.rw[r], 4) == 0;
    }
    for (int i = 0; ok && i < c.job_rows; ++i) ok = p.cpu_rows[i] == L.cpu_rows[i];
    for (int i = 0; ok && i < c.n_dma; ++i) ok = p.dma[i].e == L.str_e[i] && p.dma[i].si == L.str_si[i] && p.str_e()[i] == L.str_e[i];
    for (int i = 0; ok && i < p.n_cpu_e; ++i) ok = p.cpu_e()[i] == L.cpu_e[i];
    for (int i = 0; ok && i < p.n_hit_e; ++i) ok = p.hit_e[i] == L.hit_e[i];
    ok = ok && (uint32_t)(stage_next + (uint32_t)p.stage_ctr) == sn_legacy;
    EXPECT(ok, "plan mismatch it=%d E=%d k=%d M=%d misses=%d dma=%d jobs=%d", it, E, k, M, p.n_miss, c.n_dma, c.n_jobs);
    stage_next = sn_legacy;
    ++cases; with_dma += c.n_dma > 0; with_cpu += c.n_jobs > 0; big_miss += p.n_miss > 16;
    if (fails > 5) break;
  }
  printf("handshake CPU: plan_host(build_tables) == legacy moe_decode_experts tables in %ld cases (dma %ld · cpu %ld · >16 misses %ld)\n", cases, with_dma,
         with_cpu, big_miss);
}

// ---- 3. handshake ---------------------------------------------------------------------------------------------------------
constexpr int L = 2, E = 8, DIM = 512, INTER = 512, MB = 8, K = 3;
struct World {
  Config cfg;
  std::vector<std::vector<uint8_t>> blobs;
  std::vector<cpu::ExpertDesc> desc;  // [l·E + e]
  World() {
    cfg.dim = DIM; cfg.moe_inter = INTER; cfg.n_routed = E; cfg.dspark_experts = 4; cfg.swiglu_limit = 10.f;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> byte(0, 255);
    const size_t wsz = (size_t)INTER * DIM / 2, ssz = (size_t)INTER * DIM / 32;
    blobs.reserve(L * E * 6);
    for (int l = 0; l < L; ++l)
      for (int e = 0; e < E; ++e)
        for (const char* m : {"w1", "w2", "w3"}) {
          blobs.emplace_back(wsz); for (auto& v : blobs.back()) v = (uint8_t)byte(rng);
          const std::string p = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + "." + m;
          fake_ckpt_put(p + ".weight", blobs.back().data(), wsz);
          blobs.emplace_back(ssz); for (auto& v : blobs.back()) v = (uint8_t)(127 - 10 + byte(rng) % 6);
          fake_ckpt_put(p + ".scale", blobs.back().data(), ssz);
        }
    for (int l = 0; l < L; ++l)
      for (int e = 0; e < E; ++e) {
        const size_t b = (size_t)(l * E + e) * 6;  // w1 w1s w2 w2s w3 w3s
        desc.push_back(cpu::ExpertDesc{cpu::ExpertFormat::E2M1_B32, DIM, INTER, blobs[b].data(), blobs[b + 1].data(), blobs[b + 2].data(), blobs[b + 3].data(),
                                       blobs[b + 4].data(), blobs[b + 5].data()});
      }
  }
};

struct Rig {
  hs::BoxLayout bl = hs::BoxLayout::make(MB, K, DIM, L);
  std::vector<uint8_t> box_mem;
  uint8_t* box = nullptr;
  std::vector<float> a_f, a_s, scratch, cpu_out;
  std::atomic<uint32_t> dma_flag{0};
  std::vector<std::array<int, 3>> dma_calls;  // written by the dispatcher, read by the fake GPU after the signal (acquire)
  std::atomic<int> slow_dma_ms{0}, throw_on_post{0};
  hs::Ctrl* ctrl() { return reinterpret_cast<hs::Ctrl*>(box + bl.ctrl); }
  hs::PostHdr* hdr() { return reinterpret_cast<hs::PostHdr*>(box + bl.hdr); }
  hs::JobItem* jobs() { return reinterpret_cast<hs::JobItem*>(box + bl.jobs); }
  Rig() {
    box_mem.assign(bl.total + 64, 0);
    box = box_mem.data() + (64 - (reinterpret_cast<uintptr_t>(box_mem.data()) & 63)) % 64;
    a_f.assign((size_t)MB * DIM, 0.f); a_s.assign((size_t)MB * DIM / 32, 0.f);
    scratch.assign((size_t)bl.Jcap * ExpertStore::job_scratch_floats(INTER), 0.f);
    cpu_out.assign((size_t)bl.Jcap * DIM, 0.f);
  }
  hs::Dispatcher::Io io(ExpertStore* st, int spin_us, int poll_us) {
    hs::Dispatcher::Io o;
    o.store = st; o.ctrl = ctrl(); o.hdr = hdr(); o.jobs = jobs(); o.xq = box + bl.xq; o.xs = box + bl.xs;
    o.a_f = a_f.data(); o.a_s = a_s.data(); o.scratch = scratch.data(); o.cpu_out = cpu_out.data();
    o.dim = DIM; o.inter = INTER; o.Mb = MB; o.Jcap = bl.Jcap; o.nL = L;
    o.dma_copy = [this](int l, int e, int si) {
      if (const int ms = slow_dma_ms.load()) std::this_thread::sleep_for(MsDur(ms));
      if (throw_on_post.load() && throw_on_post.load() == (int)hdr_seq_seen.load()) throw std::runtime_error("injected dma_copy failure");
      dma_calls.push_back({l, e, si});
    };
    o.dma_signal = [this](uint32_t seq) { dma_flag.store(seq, std::memory_order_release); };
    o.spin_us = spin_us; o.poll_us = poll_us;
    return o;
  }
  std::atomic<uint32_t> hdr_seq_seen{0};  // test fixture: identifies the post that should throw (the fake GPU writes it before posting)
};
static uint32_t ld_acq(uint32_t* p) { return std::atomic_ref<uint32_t>(*p).load(std::memory_order_acquire); }
static void st_rel(uint32_t* p, uint32_t v) { std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release); }
static bool reached(uint32_t have, uint32_t want) { return (int32_t)(have - want) >= 0; }

struct GpuResult { int posts = 0, rows_checked = 0, bad_rows = 0, dma_bad = 0, timeouts = 0; };
// fake GPU: one step (L layers). Writes the mailbox in the same order as hs_plan and waits on the same conditions as the wait kernels.
static GpuResult fake_gpu_step(World& w, Rig& rg, std::mt19937& rng, uint32_t seq_base, int M, double timeout_ms, uint32_t& stage_next, bool& dev_err) {
  GpuResult res;
  auto hp = std::make_unique<hs::HostPlan>();
  std::vector<int32_t> share(hs::kMaxE + 1);
  const int S = 8;
  for (int l = 0; l < L; ++l) {
    std::vector<int32_t> ids(M * K), slot(E);
    std::vector<float> rw(M * K);
    for (int m = 0; m < M; ++m) {
      std::vector<int> pick;
      while ((int)pick.size() < K) { const int e = (int)(rng() % E); if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
      for (int j = 0; j < K; ++j) { ids[m * K + j] = pick[j]; rw[m * K + j] = std::ldexp((float)(rng() % 200 + 1), -8); }
    }
    for (int e = 0; e < E; ++e) slot[e] = rng() % 3 == 0 ? e : -1;
    make_share(0.1f + 0.05f * (float)(rng() % 15), (int)(rng() % 2), S, share.data());
    hs::plan_host(ids.data(), rw.data(), M, K, E, slot.data(), share.data(), *hp, nullptr, nullptr, hs::RecOff{}, (int)(stage_next % S), S);
    if (dev_err) continue;  // hs_plan: no posts after a timeout on an earlier layer
    const hs::LayerCounts& c = hp->counts;
    stage_next += (uint32_t)hp->stage_ctr;
    if (c.n_jobs == 0 && c.n_dma == 0) continue;
    const uint32_t seq = seq_base + (uint32_t)l + 1u;
    // mailbox write (hs_plan: build_tables writes jobs and dma, then header -> activations -> post_seq)
    std::vector<uint8_t> xq((size_t)M * DIM), xs((size_t)M * DIM / 32);
    for (auto& v : xq) { do { v = (uint8_t)(rng() & 0xFF); } while ((v & 0x7F) == 0x7F || (v & 0x78) > 0x50); }
    for (auto& v : xs) v = (uint8_t)(127 - 6 + rng() % 4);
    for (int j = 0; j < c.n_jobs; ++j) rg.jobs()[j] = hp->jobs[j];
    hs::PostHdr* h = rg.hdr();
    for (int i = 0; i < c.n_dma; ++i) h->dma[i] = hp->dma[i];
    h->seq = seq; h->layer = l; h->M = M; h->n_jobs = c.n_jobs; h->n_dma = c.n_dma; h->job_rows = c.job_rows;
    if (c.n_jobs > 0) { std::memcpy(rg.box + rg.bl.xq, xq.data(), xq.size()); std::memcpy(rg.box + rg.bl.xs, xs.data(), xs.size()); }
    rg.hdr_seq_seen.store(seq);
    rg.dma_calls.clear();  // the previous post's list was already read after its signal (the dispatcher does not write until the next post)
    st_rel(&rg.ctrl()->post_seq, seq);
    ++res.posts;
    const double t0 = now_ms();
    if (c.n_dma > 0) {  // hs_wait_dma
      while (!reached(rg.dma_flag.load(std::memory_order_acquire), seq)) {
        if (now_ms() - t0 > timeout_ms) { dev_err = true; ++res.timeouts; st_rel(&rg.ctrl()->err, ld_acq(&rg.ctrl()->err) | 2u); break; }
        std::this_thread::yield();
      }
      if (!dev_err) {
        bool ok = (int)rg.dma_calls.size() == c.n_dma;
        for (int i = 0; ok && i < c.n_dma; ++i) ok = rg.dma_calls[i][0] == l && rg.dma_calls[i][1] == hp->dma[i].e && rg.dma_calls[i][2] == hp->dma[i].si;
        res.dma_bad += !ok;
      }
    }
    if (c.n_jobs > 0 && !dev_err) {  // hs_wait_cpu -> the results hs_accum_cpu reads
      while (!reached(ld_acq(&rg.ctrl()->done_seq), seq)) {
        if (now_ms() - t0 > timeout_ms) { dev_err = true; ++res.timeouts; st_rel(&rg.ctrl()->err, ld_acq(&rg.ctrl()->err) | 1u); break; }
        std::this_thread::yield();
      }
      if (!dev_err && ld_acq(&rg.ctrl()->cpu_err) == 0) {
        std::vector<float> ref(DIM), tmp(4 * INTER + 2 * DIM + 64);
        for (int j = 0; j < c.n_jobs; ++j) {
          const hs::JobItem& jb = hp->jobs[j];
          for (int r = 0; r < jb.R; ++r) {
            const int m = jb.m[r];
            cpu::expert_forward(w.desc[(size_t)l * E + jb.e], xq.data() + (size_t)m * DIM, xs.data() + (size_t)m * DIM / 32, jb.rw[r], w.cfg.swiglu_limit,
                                ref.data(), tmp.data());
            const float* got = rg.cpu_out.data() + (size_t)(jb.row0 + r) * DIM;
            res.bad_rows += std::memcmp(got, ref.data(), DIM * sizeof(float)) != 0;
            ++res.rows_checked;
          }
        }
      }
    }
  }
  return res;
}

static void test_handshake(int steps, int threads, int spin_us, int poll_us) {
  World w;
  Checkpoint ck("fake");
  auto store = std::make_unique<ExpertStore>(w.cfg, L, 0, threads, 0);
  for (int l = 0; l < L; ++l) store->load_layer_experts(ck, l, 2);
  Rig rg;
  auto disp = std::make_unique<hs::Dispatcher>(rg.io(store.get(), spin_us, poll_us));
  std::mt19937 rng(4321);
  uint32_t seq_base = 0, stage_next = 0;
  long posts = 0, rows = 0, bad = 0, dma_bad = 0, empty_steps = 0;
  for (int s = 0; s < steps; ++s) {
    seq_base += L + 1;
    disp->arm();
    if (s % 17 == 3) { disp->disarm(); ++empty_steps; continue; }  // step without a post
    bool dev_err = false;
    const GpuResult r = fake_gpu_step(w, rg, rng, seq_base, 1 + (int)(rng() % MB), 20000.0, stage_next, dev_err);
    disp->disarm();  // engine thread: after the end-of-step synchronization
    EXPECT(!dev_err && r.timeouts == 0, "step %d: unexpected timeout", s);
    EXPECT(ld_acq(&rg.ctrl()->cpu_err) == 0, "step %d: cpu_err", s);
    EXPECT(disp->served() == ld_acq(&rg.ctrl()->post_seq), "step %d: dispatcher did not serve the last post", s);
    for (int l = 0; l < L; ++l) {  // log: posted layers only; cpu_ms >= 0 when there were CPU jobs
      const auto& lg = disp->log()[(size_t)l];
      if (lg.layer == l && lg.n_jobs > 0) EXPECT(lg.cpu_ms >= 0, "step %d layer %d: cpu_ms %f", s, l, lg.cpu_ms);
    }
    posts += r.posts; rows += r.rows_checked; bad += r.bad_rows; dma_bad += r.dma_bad;
    if (fails > 5) break;
  }
  EXPECT(bad == 0 && dma_bad == 0, "%ld/%ld CPU rows differ from expert_forward · %ld DMA lists differ", bad, rows, dma_bad);
  printf("handshake CPU: %d steps (%ld empty) · %ld posts · %ld CPU rows exact vs expert_forward · DMA order exact (threads/node %d, %s)\n", steps,
         empty_steps, posts, rows, threads, poll_us > 0 ? "sleep-poll" : "spin");

  // slow dispatcher -> fake GPU timeout -> later layers are not posted -> disarm waits for the dispatcher to finish -> next step is clean
  {
    int timeouts = 0;
    for (int tries = 0; tries < 20 && timeouts == 0; ++tries) {
      rg.slow_dma_ms.store(60);
      seq_base += L + 1;
      disp->arm();
      bool dev_err = false;
      st_rel(&rg.ctrl()->err, 0);
      const GpuResult r = fake_gpu_step(w, rg, rng, seq_base, MB, 5.0, stage_next, dev_err);
      const double t0 = now_ms();
      disp->disarm();
      timeouts += r.timeouts;
      if (r.timeouts) EXPECT(ld_acq(&rg.ctrl()->err) != 0, "timeout not flagged");
      (void)t0;
      rg.slow_dma_ms.store(0);
    }
    EXPECT(timeouts > 0, "no DMA post in 20 tries (fixture)");
    seq_base += L + 1;
    disp->arm();
    bool dev_err = false;
    st_rel(&rg.ctrl()->err, 0);
    const GpuResult r = fake_gpu_step(w, rg, rng, seq_base, MB, 20000.0, stage_next, dev_err);
    disp->disarm();
    EXPECT(!dev_err && r.bad_rows == 0 && r.dma_bad == 0, "step after timeout not clean");
    printf("handshake CPU: slow dispatcher → GPU wait timeout flagged (%d), disarm waited, next step exact (%d rows)\n", timeouts, r.rows_checked);
  }
  // callback exception -> cpu_err, done still written (the GPU does not hang)
  {
    int hit = 0;
    for (int tries = 0; tries < 20 && !hit; ++tries) {
      seq_base += L + 1;
      rg.throw_on_post.store((int)(seq_base + 1));  // layer 0 post
      disp->arm();
      bool dev_err = false;
      st_rel(&rg.ctrl()->cpu_err, 0);
      (void)fake_gpu_step(w, rg, rng, seq_base, MB, 20000.0, stage_next, dev_err);
      disp->disarm();
      rg.throw_on_post.store(0);
      EXPECT(!dev_err, "exception in the dispatcher blocked the GPU");
      hit = ld_acq(&rg.ctrl()->cpu_err) != 0;
      st_rel(&rg.ctrl()->cpu_err, 0);
    }
    EXPECT(hit, "injected dispatcher exception never reached cpu_err");
    printf("handshake CPU: dispatcher exception → cpu_err set, done written (GPU released)\n");
  }
  disp.reset();   // stop the thread (idle)
  store.reset();  // stop the pool
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 20000;
  const int steps = argc > 2 ? atoi(argv[2]) : 300;
  const int threads = argc > 3 ? atoi(argv[3]) : 3;
  test_sort(iters);
  test_plan(iters);
  test_handshake(steps, threads, -1, 0);
  test_handshake(std::max(20, steps / 4), threads, 0, 50);  // sleep-polling mode
  if (fails) { fprintf(stderr, "handshake CPU: %d failure(s)\n", fails); return 1; }
  printf("handshake CPU suite OK\n");
  return 0;
}
