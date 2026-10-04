// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// O1 HIVE_DECODE_UBATCH GPU test — needs a GPU and nvcc.
//   (1) Per-row bitwise comparison: on a synthetic layer stack (L layers, random routing per layer, NE expert records in the real storage format, real layer shape dim 5120,
//       inter 2304, top-6) the GPU share of decode experts is run (A) once over the whole batch (same grouping as runtime moe_decode_experts: experts ascending, (m,j)
//       ascending within an expert, groups <= 8 rows) and (B) as two halves (A = [0, ceil(M/2)), B = the rest — like runtime ub_view, xq/xs/acc pointers offset by the
//       half's first row + half-local row numbers, grouped separately per half); the acc (fp32) accumulated across layers is compared bitwise per row. Versions: the
//       unfused chain (mx_grouped_w13 -> w2 -> accum_bf16_rows_seq), D1 fusion, F2 fusion (S2 4).
//       The CPU share is simulated per layer as one fp32 row on half of the rows (accum_f32_rows — atomic adds, so only one per row to stay deterministic).
//       Expected: everything bit-identical (GPU expert compute is row-independent — mma rows never mix, and accumulation per row follows group order (= experts
//       ascending)). Otherwise FAIL + the number of differing rows.
//       WARNING: only the GPU expert share is compared here. That the miss classification (DMA/CPU) can differ per half (see the O1 header comment in runtime.cpp)
//       and the M-dependent reductions of the front kernels (attention, router, cuBLAS) are outside this test — confirm on the real model by comparing
//       HIVE_DECODE_UBATCH=0/1 on the same prompt (greedy).
//   (2) Timing (model): runs the same ordering function as production (dov::run_ubatch — hive/decode_overlap.h) with real streams, events and CPU threads. Front =
//       globaltimer spin kernel (front_us + front_row_us x rows); GPU experts = the real kernel (D1 or the environment's version); CPU misses = host threads busy for
//       cpu_us + cpu_row_us x (CPU-share rows) and then writing fp32 rows. Compares step ms against the sequential order (per layer: front -> sync -> expert
//       launch + CPU start -> CPU wait -> accumulate) and also compares the acc of both runs bitwise (does the ordering function keep data dependencies with real
//       events and streams). Cost values are set on the command line — the verdict on the real model comes from A/B of hived's HIVE_PROFILE [decode-host] lines
//       (this model only shows the upper bound and trend of the overlap).
//   Run: scripts/hive-run.sh "./build-dev/test_decode_ubatch [layers=12] [NE=64] [reps=5] [front_us=250] [front_row_us=20] [cpu_us=150] [cpu_row_us=100]"
//   VRAM ≈ NE × 18.9 MB + 0.3 GB.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "hive/common.h"
#include "hive/decode_overlap.h"
#include "hive/expert_store.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/clock.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

constexpr int DIM = 5120, INTER = 2304, TOPK = 6;
constexpr float LIMIT = 10.f;

__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed, int mode) {
  const size_t i = hive::cu::global_tid();
  if (i >= n) return;
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  uint8_t v = (uint8_t)h;
  if (mode == 1) v = (uint8_t)((v & 0x80) | ((v & 0x7F) % 0x48));  // e4m3 activations (no NaN)
  else if (mode == 2) v = (uint8_t)(120 + (h >> 8) % 8);            // activation scales e8m0
  else if (mode == 3) v = (uint8_t)(118 + (h >> 8) % 6);            // weight scales e8m0
  p[i] = v;                                                         // mode 0: two e2m1 nibbles
}
void fill(void* p, size_t n, uint32_t seed, int mode) {
  fill_kernel<<<(unsigned)((n + 255) / 256), 256>>>((uint8_t*)p, n, seed, mode);
  CUDA_CHECK(cudaGetLastError());
}
__global__ void spin_ns_kernel(unsigned long long ns) {  // simulates the front (one warp — barely uses the SM: only the front's "time" is simulated)
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T))); return p; }
template <class T> T* hmalloc(size_t n) { T* p; CUDA_CHECK(cudaHostAlloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T), cudaHostAllocDefault)); return p; }
double now_ms() { return hive::mono_ms(); }
void busy_us(double us) { const double t_end = now_ms() + us / 1000.0; while (now_ms() < t_end) {} }

struct LayerRoute {
  std::vector<int> ids;       // [M·k]
  std::vector<float> rw;      // [M·k]
  std::vector<int> cpu_row;   // [M] 1 = this row has a CPU share (one fp32 row)
  std::vector<float> cpu_val; // [M·dim] CPU share values
};
// Same grouping as the runtime (rows [r0, r0+n) only, local row numbers)
struct Tables { std::vector<k::GroupDesc> gd; std::vector<int32_t> rows; std::vector<float> rw; };
Tables build(const LayerRoute& L, int r0, int n, const std::vector<uint8_t*>& rec, const ExpertLayout& lay, int NE) {
  Tables t;
  for (int e = 0; e < NE; ++e) {
    std::vector<int> mj;
    for (int i = r0 * TOPK; i < (r0 + n) * TOPK; ++i) if (L.ids[i] == e) mj.push_back(i);
    for (size_t i0 = 0; i0 < mj.size(); i0 += 8) {
      k::GroupDesc d{};
      const uint8_t* rp = rec[e];
      d.w1 = rp + lay.w1; d.s1 = rp + lay.s1; d.w3 = rp + lay.w3; d.s3 = rp + lay.s3; d.w2 = rp + lay.w2; d.s2 = rp + lay.s2;
      d.row0 = (int)t.rows.size(); d.n = (int)std::min<size_t>(8, mj.size() - i0);
      for (int i = (int)i0; i < (int)i0 + d.n; ++i) { t.rows.push_back(mj[i] / TOPK - r0); t.rw.push_back(L.rw[mj[i]]); }
      t.gd.push_back(d);
    }
  }
  return t;
}

struct Dev {  // mimics the decode buffers of runtime Work (rows = M, expert rows = M·k)
  int M = 0, L = 0;
  std::vector<uint8_t*> xq, xs;  // per layer [M, dim], [M, dim/32]
  uint8_t *yq = nullptr, *ys = nullptr; bf16 *y = nullptr, *eout = nullptr;
  float *acc = nullptr, *acc0 = nullptr;
  uint8_t* tbl_d = nullptr; uint8_t* tbl_h = nullptr; size_t tbl_half = 0;  // two half regions
  float* cpu_h = nullptr; float* cpu_d = nullptr; int32_t* cpu_rows_d = nullptr; int32_t* cpu_rows_h = nullptr;  // two half regions
};

// variant: 0 = unfused chain, 1 = D1, 2 = F2 (S2 4)
void run_groups(int variant, const k::GroupDesc* g, int ng, const int32_t* rows, const float* rw, int nrows, int M, const uint8_t* xq, const uint8_t* xs,
                Dev& d, float* acc, cudaStream_t st) {
  if (ng <= 0) return;
  if (variant == 0) {
    k::mx_grouped_w13(g, ng, xq, xs, INTER, DIM, rw, LIMIT, d.y, st, rows, d.yq, d.ys);
    k::mx_grouped_w2(g, ng, d.yq, d.ys, DIM, INTER, d.eout, st);
    k::accum_bf16_rows_seq(d.eout, rows, nrows, M, DIM, acc, st);
  } else {
    k::moe_decode_fused(g, ng, xq, xs, rows, rw, 0, nrows, M, DIM, INTER, LIMIT, d.y, d.yq, d.ys, d.eout, acc, st);
  }
}

// Layer experts for one half (or the whole batch): table H2D (half region) -> GPU groups -> (CPU values H2D -> accum_f32_rows)
struct Part { int r0, n, slot; };
size_t upload(const Tables& t, Dev& d, int slot, cudaStream_t st, const k::GroupDesc** g, const int32_t** rows, const float** rw) {
  const size_t off_rows = sizeof(k::GroupDesc) * t.gd.size(), off_rw = off_rows + t.rows.size() * 4, total = off_rw + t.rw.size() * 4;
  uint8_t* h = d.tbl_h + (size_t)slot * d.tbl_half;
  uint8_t* dv = d.tbl_d + (size_t)slot * d.tbl_half;
  std::memcpy(h, t.gd.data(), off_rows); std::memcpy(h + off_rows, t.rows.data(), t.rows.size() * 4); std::memcpy(h + off_rw, t.rw.data(), t.rw.size() * 4);
  if (total) CUDA_CHECK(cudaMemcpyAsync(dv, h, total, cudaMemcpyHostToDevice, st));
  *g = reinterpret_cast<const k::GroupDesc*>(dv); *rows = reinterpret_cast<const int32_t*>(dv + off_rows); *rw = reinterpret_cast<const float*>(dv + off_rw);
  return total;
}
int cpu_rows_of(const LayerRoute& L, const Part& p) { int n = 0; for (int m = p.r0; m < p.r0 + p.n; ++m) n += L.cpu_row[m]; return n; }
// CPU simulation: fp32 rows into the half region (value = L.cpu_val) + row numbers (half-local)
int cpu_fill(const LayerRoute& L, const Part& p, Dev& d) {
  float* out = d.cpu_h + (size_t)p.slot * d.M * DIM;
  int32_t* rr = d.cpu_rows_h + (size_t)p.slot * d.M;
  int n = 0;
  for (int m = p.r0; m < p.r0 + p.n; ++m)
    if (L.cpu_row[m]) { std::memcpy(out + (size_t)n * DIM, L.cpu_val.data() + (size_t)m * DIM, DIM * 4); rr[n++] = m - p.r0; }
  return n;
}
void cpu_accum(int n, const Part& p, Dev& d, cudaStream_t st) {
  if (n <= 0) return;
  float* h = d.cpu_h + (size_t)p.slot * d.M * DIM;
  float* dv = d.cpu_d + (size_t)p.slot * d.M * DIM;
  int32_t* rh = d.cpu_rows_h + (size_t)p.slot * d.M;
  int32_t* rd = d.cpu_rows_d + (size_t)p.slot * d.M;
  CUDA_CHECK(cudaMemcpyAsync(dv, h, (size_t)n * DIM * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(rd, rh, (size_t)n * 4, cudaMemcpyHostToDevice, st));
  k::accum_f32_rows(dv, rd, n, DIM, d.acc + (size_t)p.r0 * DIM, st);
}

// (1) Bitwise comparison: all layers in synchronous order (whole batch once vs two halves)
bool bit_check(int variant, const std::vector<LayerRoute>& R, const std::vector<uint8_t*>& rec, const ExpertLayout& lay, int NE, Dev& d, cudaStream_t st,
               std::vector<float>& out) {
  const int M = d.M;
  CUDA_CHECK(cudaMemcpyAsync(d.acc, d.acc0, (size_t)M * DIM * 4, cudaMemcpyDeviceToDevice, st));
  for (int l = 0; l < d.L; ++l) {
    const bool full = out.empty();  // first call (out empty) = whole batch, second = halves
    std::vector<Part> parts;
    if (full) parts.push_back(Part{0, M, 0});
    else for (int h = 0; h < 2; ++h) parts.push_back(Part{dov::half_row0(M, h), dov::half_rows(M, h), h});
    for (const Part& p : parts) {
      const Tables t = build(R[l], p.r0, p.n, rec, lay, NE);
      bool ok = variant == 0 || k::moe_decode_fused_ok(DIM, INTER, p.n, (int)t.gd.size());
      for (const auto& g : t.gd) ok = ok && (variant == 0 || k::moe_decode_desc_aligned(g));
      if (!ok) { printf("  fused path not applicable (shape/alignment)\n"); return false; }
      const k::GroupDesc* g; const int32_t* rows; const float* rw;
      upload(t, d, p.slot, st, &g, &rows, &rw);
      run_groups(variant, g, (int)t.gd.size(), rows, rw, (int)t.rows.size(), p.n, d.xq[l] + (size_t)p.r0 * DIM, d.xs[l] + (size_t)p.r0 * DIM / 32, d,
                 d.acc + (size_t)p.r0 * DIM, st);
      const int n = cpu_fill(R[l], p, d);
      cpu_accum(n, p, d, st);
      CUDA_CHECK(cudaStreamSynchronize(st));  // before the host region is reused (this comparison path is not timed)
    }
  }
  std::vector<float> got((size_t)M * DIM);
  CUDA_CHECK(cudaMemcpy(got.data(), d.acc, got.size() * 4, cudaMemcpyDeviceToHost));
  if (out.empty()) { out = got; return true; }
  int bad_rows = 0; size_t bad = 0; double max_abs = 0;
  for (int m = 0; m < M; ++m) {
    size_t b = 0;
    for (int x = 0; x < DIM; ++x) {
      const size_t i = (size_t)m * DIM + x;
      if (std::memcmp(&out[i], &got[i], 4) != 0) { ++b; max_abs = std::max(max_abs, (double)std::fabs(out[i] - got[i])); }
    }
    bad += b; bad_rows += b > 0;
  }
  if (bad) printf("  variant %d: %d/%d rows differ (%zu values, max abs %.3e)\n", variant, bad_rows, M, bad, max_abs);
  return bad == 0;
}

// (2) Timing model: sequential order vs dov::run_ubatch
struct Cost { double front_us, front_row_us, cpu_us, cpu_row_us; };
struct CpuThread {
  std::mutex mu; std::condition_variable cv; std::function<void()> job; bool has = false, busy = false, stop = false; std::thread th;
  CpuThread() { th = std::thread([this] { for (;;) { std::function<void()> f; { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return stop || has; }); if (!has) return; f = job; has = false; }
                                                 f(); { std::lock_guard<std::mutex> l(mu); busy = false; } cv.notify_all(); } }); }
  ~CpuThread() { { std::lock_guard<std::mutex> l(mu); stop = true; } cv.notify_all(); th.join(); }
  void start(std::function<void()> f) { std::lock_guard<std::mutex> l(mu); if (busy) { printf("FAIL: CPU pool got a second batch\n"); std::_Exit(4); } busy = has = true; job = std::move(f); cv.notify_all(); }
  void wait() { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return !busy; }); }
};
struct Pipe {
  const std::vector<LayerRoute>& R; const std::vector<uint8_t*>& rec; const ExpertLayout& lay; int NE; Dev& d; cudaStream_t st; Cost cost; int variant;
  CpuThread cpu;
  cudaEvent_t done[2];
  std::vector<std::vector<Tables>> tab;  // [layer][half] pre-grouped tables (host grouping time is excluded from this model — the runtime counts it in prep)
  Part part[2]; int ncpu[2] = {0, 0}; bool started[2] = {false, false};
  Pipe(const std::vector<LayerRoute>& r, const std::vector<uint8_t*>& rc, const ExpertLayout& ly, int ne, Dev& dv, cudaStream_t s, Cost c, int var, bool halves)
      : R(r), rec(rc), lay(ly), NE(ne), d(dv), st(s), cost(c), variant(var) {
    for (auto& e : done) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    for (int h = 0; h < 2; ++h) part[h] = halves ? Part{dov::half_row0(d.M, h), dov::half_rows(d.M, h), h} : Part{0, d.M, 0};
    tab.resize(d.L);
    for (int l = 0; l < d.L; ++l) for (int h = 0; h < (halves ? 2 : 1); ++h) tab[l].push_back(build(R[l], part[h].r0, part[h].n, rec, lay, NE));
  }
  ~Pipe() { for (auto e : done) cudaEventDestroy(e); }
  // Operations called by dov::run_ubatch (same role as runtime.cpp decode_layers_ubatch::Ops)
  void prep(int, int) {}
  void front(int h, int) {
    spin_ns_kernel<<<1, 32, 0, st>>>((unsigned long long)((cost.front_us + cost.front_row_us * part[h].n) * 1000.0));
    CUDA_CHECK(cudaEventRecord(done[h], st));
  }
  void wait_front(int h, int) { CUDA_CHECK(cudaEventSynchronize(done[h])); }
  void launch(int h, int l, bool go) {
    const Tables& t = tab[l][h];
    const k::GroupDesc* g; const int32_t* rows; const float* rw;
    upload(t, d, part[h].slot, st, &g, &rows, &rw);
    run_groups(variant, g, (int)t.gd.size(), rows, rw, (int)t.rows.size(), part[h].n, d.xq[l] + (size_t)part[h].r0 * DIM, d.xs[l] + (size_t)part[h].r0 * DIM / 32, d,
               d.acc + (size_t)part[h].r0 * DIM, st);
    ncpu[h] = cpu_rows_of(R[l], part[h]);
    started[h] = false;
    cur_l[h] = l;
    if (go) start_cpu(h, l);
  }
  int cur_l[2] = {0, 0};
  bool cpu_idle(int h, int) const { return ncpu[h] == 0; }
  void start_cpu(int h, int l) {
    if (ncpu[h] == 0 || started[h]) return;
    started[h] = true;
    const double us = cost.cpu_us + cost.cpu_row_us * ncpu[h];
    cpu.start([this, h, l, us] { busy_us(us); cpu_fill(R[l], part[h], d); });
  }
  void wait_cpu(int h, int) { if (ncpu[h]) cpu.wait(); }
  void accum(int h, int) { cpu_accum(ncpu[h], part[h], d, st); }
  // Sequential order (whole batch per layer: front -> sync -> experts + CPU start -> CPU wait -> accumulate)
  void run_serial() {
    for (int l = 0; l < d.L; ++l) { front(0, l); wait_front(0, l); launch(0, l, true); wait_cpu(0, l); accum(0, l); }
  }
};

double time_run(Pipe& P, bool ub, int reps, std::vector<float>& acc_out) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaMemcpyAsync(P.d.acc, P.d.acc0, (size_t)P.d.M * DIM * 4, cudaMemcpyDeviceToDevice, P.st));
    CUDA_CHECK(cudaStreamSynchronize(P.st));
    const double t0 = now_ms();
    if (ub) dov::run_ubatch(P.d.L, P);
    else P.run_serial();
    CUDA_CHECK(cudaStreamSynchronize(P.st));
    best = std::min(best, now_ms() - t0);
  }
  acc_out.resize((size_t)P.d.M * DIM);
  CUDA_CHECK(cudaMemcpy(acc_out.data(), P.d.acc, acc_out.size() * 4, cudaMemcpyDeviceToHost));
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const int L = argc > 1 ? std::max(1, atoi(argv[1])) : 12;
  int NE = argc > 2 ? std::max(12, atoi(argv[2])) : 64;
  const int reps = argc > 3 ? std::max(1, atoi(argv[3])) : 5;
  Cost cost{argc > 4 ? atof(argv[4]) : 250.0, argc > 5 ? atof(argv[5]) : 20.0, argc > 6 ? atof(argv[6]) : 150.0, argc > 7 ? atof(argv[7]) : 100.0};
  const ExpertLayout lay = ExpertLayout::make(DIM, INTER);
  size_t vfree = 0, vtot = 0;
  CUDA_CHECK(cudaMemGetInfo(&vfree, &vtot));
  const size_t reserve = (size_t)512 << 20;
  const int fit = vfree > reserve ? (int)((vfree - reserve) / lay.total) : 0;
  if (fit < NE) { printf("VRAM free %.0f MiB → experts %d → %d\n", vfree / 1048576.0, NE, fit); NE = fit; }
  if (NE < 12) { printf("RESULT: FAIL (not enough VRAM for 12 expert records)\n"); return 2; }
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("device %s · layers %d · experts %d × %.1f MB · cost: front %.0f+%.0f/row µs · cpu %.0f+%.0f/row µs (model)\n", prop.name, L, NE, lay.total / 1e6,
         cost.front_us, cost.front_row_us, cost.cpu_us, cost.cpu_row_us);
  std::vector<uint8_t*> rec(NE);
  for (int e = 0; e < NE; ++e) {
    rec[e] = dmalloc<uint8_t>(lay.total);
    fill(rec[e] + lay.w1, (size_t)INTER * DIM / 2, 1000 + e, 0); fill(rec[e] + lay.s1, (size_t)INTER * DIM / 32, 2000 + e, 3);
    fill(rec[e] + lay.w3, (size_t)INTER * DIM / 2, 3000 + e, 0); fill(rec[e] + lay.s3, (size_t)INTER * DIM / 32, 4000 + e, 3);
    fill(rec[e] + lay.w2, (size_t)DIM * INTER / 2, 5000 + e, 0); fill(rec[e] + lay.s2, (size_t)DIM * INTER / 32, 6000 + e, 3);
  }
  cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  bool ok = true;
  printf("%-3s %-8s %-28s %-12s %-12s %-8s %s\n", "M", "variant", "per-row acc (full vs halves)", "serial ms", "ubatch ms", "speedup", "pipeline acc");
  for (int M : {2, 4, 8}) {
    std::mt19937 rng(20260930u + (unsigned)M);
    std::vector<int> perm(NE); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<double> pw(NE); for (int r = 0; r < NE; ++r) pw[perm[r]] = 1.0 / (r + 4.0);  // popularity skew — so some experts overlap between the halves
    std::discrete_distribution<int> pick(pw.begin(), pw.end());
    std::uniform_real_distribution<float> urw(0.05f, 0.5f), uval(-0.5f, 0.5f);
    std::vector<LayerRoute> R(L);
    for (auto& lr : R) {
      lr.ids.resize((size_t)M * TOPK); lr.rw.resize((size_t)M * TOPK); lr.cpu_row.resize(M); lr.cpu_val.resize((size_t)M * DIM);
      for (int m = 0; m < M; ++m) {
        for (int j = 0; j < TOPK; ++j) {
          int e; bool dup;
          do { e = pick(rng); dup = false; for (int q = 0; q < j; ++q) dup |= lr.ids[m * TOPK + q] == e; } while (dup);
          lr.ids[m * TOPK + j] = e; lr.rw[m * TOPK + j] = urw(rng);
        }
        lr.cpu_row[m] = (int)(rng() & 1);
      }
      for (auto& v : lr.cpu_val) v = uval(rng);
    }
    Dev d; d.M = M; d.L = L;
    for (int l = 0; l < L; ++l) {
      d.xq.push_back(dmalloc<uint8_t>((size_t)M * DIM)); d.xs.push_back(dmalloc<uint8_t>((size_t)M * DIM / 32));
      fill(d.xq[l], (size_t)M * DIM, 77u * (unsigned)l + 5, 1); fill(d.xs[l], (size_t)M * DIM / 32, 91u * (unsigned)l + 3, 2);
    }
    const int R_ = M * TOPK;
    d.yq = dmalloc<uint8_t>((size_t)R_ * INTER); d.ys = dmalloc<uint8_t>((size_t)R_ * INTER / 32); d.y = dmalloc<bf16>((size_t)R_ * INTER);
    d.eout = dmalloc<bf16>((size_t)R_ * DIM);
    d.acc = dmalloc<float>((size_t)M * DIM); d.acc0 = dmalloc<float>((size_t)M * DIM);
    { std::vector<float> h((size_t)M * DIM); for (auto& v : h) v = uval(rng); CUDA_CHECK(cudaMemcpy(d.acc0, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
    d.tbl_half = (sizeof(k::GroupDesc) + 8) * (size_t)R_ + 256;
    d.tbl_d = dmalloc<uint8_t>(2 * d.tbl_half); d.tbl_h = hmalloc<uint8_t>(2 * d.tbl_half);
    d.cpu_h = hmalloc<float>((size_t)2 * M * DIM); d.cpu_d = dmalloc<float>((size_t)2 * M * DIM);
    d.cpu_rows_h = hmalloc<int32_t>((size_t)2 * M); d.cpu_rows_d = dmalloc<int32_t>((size_t)2 * M);
    CUDA_CHECK(cudaDeviceSynchronize());
    const char* vname[3] = {"old", "D1", "F2s4"};
    for (int variant = 0; variant < 3; ++variant) {
      k::moe_decode_fused_force(variant == 2 ? 4 : 0);
      printf("  [M=%d] launching %s\n", M, vname[variant]); fflush(stdout);
      std::vector<float> full;
      bool exact = bit_check(variant, R, rec, lay, NE, d, st, full);
      exact = exact && bit_check(variant, R, rec, lay, NE, d, st, full);
      // Timing model (real events, streams, CPU threads) — sequential order vs run_ubatch, also comparing the acc of both runs
      Pipe ser(R, rec, lay, NE, d, st, cost, variant, false), ub(R, rec, lay, NE, d, st, cost, variant, true);
      std::vector<float> a_ser, a_ub;
      const double t_ser = time_run(ser, false, reps, a_ser), t_ub = time_run(ub, true, reps, a_ub);
      const bool pipe_exact = std::memcmp(a_ser.data(), a_ub.data(), a_ser.size() * 4) == 0 && std::memcmp(a_ser.data(), full.data(), a_ser.size() * 4) == 0;
      printf("%-3d %-8s %-28s %-12.3f %-12.3f ×%-7.2f %s\n", M, vname[variant], exact ? "PASS (bit-exact)" : "FAIL", t_ser, t_ub, t_ser / std::max(1e-9, t_ub),
             pipe_exact ? "PASS (== full-batch)" : "FAIL");
      ok = ok && exact && pipe_exact;
    }
    k::moe_decode_fused_force(-1);
    for (auto p : d.xq) cudaFree(p);
    for (auto p : d.xs) cudaFree(p);
    cudaFree(d.yq); cudaFree(d.ys); cudaFree(d.y); cudaFree(d.eout); cudaFree(d.acc); cudaFree(d.acc0); cudaFree(d.tbl_d); cudaFreeHost(d.tbl_h);
    cudaFreeHost(d.cpu_h); cudaFree(d.cpu_d); cudaFreeHost(d.cpu_rows_h); cudaFree(d.cpu_rows_d);
  }
  for (auto* p : rec) cudaFree(p);
  cudaStreamDestroy(st);
  printf("RESULT: %s\n", ok ? "PASS (per-row bit-exact: full batch == halves == pipelined halves)" : "FAIL");
  return ok ? 0 : 1;
}
