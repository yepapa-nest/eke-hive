// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// R3 validation (GPU): new long-context decode indexer (hive/decode_longctx.h) vs the current production path (D2 dec_attention_front step (4), unchanged).
//   (1) Whole indexer stage (dec_indexer_stage — the same function as production): 4 layer kinds x context L = 4K, 32K, 80K, 100K x M = 1, 4, 8 x Tmax = graph bucket / exact
//      A  layers 2, 8, 14  (ratio 2, scores over all rows T = L/2 -> top-512)
//      B  layer 20         (ratio 1, scores over all rows T = L -> block max (8) -> block top-2048 = candidate table -> top-512)
//      C  layers 24..36    (candidate pool min(2048, ceil(T/8))*8 columns — bf16 scores (idx_mode 0), tc scores (idx_mode 1 = HIVE_IDX_TC, production))
//      Versions: 0 = previous, 1 = IDXSCORE2, 2 = TOPK2, 3 = both. Comparison = iscore (all columns), bmax, cand, idx (whole rows) **byte-identical** (no tolerance).
//      Time = µs/layer (after flushing the 128 MB L2 with a 512 MB memset, and back to back) -> step estimate = 3·A + B + 4·C(tc).
//      Graph: capture version 3 as a CUDA graph and replay it 3 times -> byte-identical to the previous output (checks counter reset to 0 and reuse of the union buffer).
//   (2) top-k alone: previous dec_topk_bf16/f32 vs dec_topk2_* — 6 distributions (random, tie storm, narrow exponent, half -inf + NaN, all equal, 98% -inf + negative NaN) x
//      T_in 4K..128K (bf16 k 512, per-row trows); block top-k (f32 k 2048, T_in 512..16K) — byte-identical + µs.
//   (3) Scores alone: KPT 1, 2, 4 each byte-identical to the previous kernel + µs (the basis for choosing HIVE_DECODE_IDXSCORE2_KPT).
//   Each launch prints "[run] ..." first (attributes abnormal exits). Verdict: everything identical = ALL PASS.
//   Run: scripts/hive-run.sh "./build-dev/test_decode_longctx [reps=20] [ctx list=4096,32768,81920,102400] [M list=1,4,8]"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "hive/attn_decode_fused.h"
#include "hive/common.h"
#include "hive/decode_longctx.h"
#include "hive/kv_pack.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

constexpr int HI = 32, DI = 128, ITOPK = 512, WIN = 128, BS = 8, CB = 2048, CAND_SRC = 20;

__device__ __forceinline__ uint32_t hsh(size_t i, uint32_t seed) {
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed ^ (uint32_t)(i >> 32) * 0x9E3779B9u;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  return h;
}
// Index key row (80 B): [e8m0 x4, 0 x12][nibbles x64]
__global__ void fill_idx_rows_kernel(uint8_t* p, size_t nrows, uint32_t seed) {
  const size_t n = nrows * kvp::IDX_ROW;
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) {
    const uint32_t h = hsh(i, seed);
    const int b = (int)(i % kvp::IDX_ROW);
    p[i] = b < kvp::IDX_SCALES ? (uint8_t)(122 + h % 8) : b < kvp::IDX_HDR ? (uint8_t)0 : (uint8_t)(h >> 8);
  }
}
__global__ void fill_bf16_kernel(bf16* p, size_t n, uint32_t seed, float a, float b) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size())
    p[i] = f2bf(a + b * ((float)(hsh(i, seed) >> 8) * (2.f / 16777216.f) - 1.f));
}
unsigned nblk(size_t n) { return (unsigned)std::min<size_t>(8192, (n + 255) / 256); }
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T) + 256)); CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T) + 256)); return p; }
std::vector<uint8_t> host(const void* d, size_t bytes) { std::vector<uint8_t> v(bytes); CUDA_CHECK(cudaMemcpy(v.data(), d, bytes, cudaMemcpyDeviceToHost)); return v; }
size_t diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) { size_t d = a.size() != b.size(); for (size_t i = 0; i < a.size(); i++) d += a[i] != b[i]; return d; }
int next_pow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }

void run_sync(const std::string& what, cudaStream_t st, const std::function<void()>& f) {
  printf("[run] %s\n", what.c_str());
  fflush(stdout);
  f();
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaGetLastError());
}
double time_us(const std::function<void()>& f, int reps, cudaStream_t st, uint8_t* flush, size_t flush_n) {
  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  double t = 0;
  for (int r = 0; r <= reps; ++r) {
    if (flush) CUDA_CHECK(cudaMemsetAsync(flush, r & 0xFF, flush_n, st));
    CUDA_CHECK(cudaEventRecord(e0, st)); f(); CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (r) t += ms;
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return t * 1000.0 / reps;
}

int g_fail = 0, g_cases = 0;
void check(bool ok, const std::string& what) { ++g_cases; if (!ok) { ++g_fail; printf("  FAIL %s\n", what.c_str()); fflush(stdout); } }

// ---- Indexer work buffers for one context and M ------------------------------------------------------------------------------------------------
struct Bufs {
  int M = 0, Tcap = 0;
  uint8_t *keys2 = nullptr, *keys1 = nullptr;  // ratio-2 source (layer 2), ratio-1 source (layer 20 — read by 24..36)
  bf16 *iq = nullptr, *iw = nullptr, *iscore = nullptr;
  uint8_t* iqp = nullptr;
  float* bmax = nullptr;
  int32_t *cand = nullptr, *cand_in = nullptr, *idx = nullptr;
  void* scratch = nullptr;
  size_t scratch_bytes = 0;
  int* attn_cnt = nullptr;
  k::DecRow* tab = nullptr;
  k::DecSoA* soa = nullptr;
};
void set_rows(Bufs& B, const uint8_t* kcache, const std::vector<int>& T_rows, cudaStream_t st) {
  std::vector<k::DecRow> h(B.M);
  k::DecSoA s{};
  for (int m = 0; m < B.M; ++m) {
    memset(&h[m], 0, sizeof(k::DecRow));
    h[m].kptr = kcache; h[m].trows = T_rows[m]; h[m].visible = T_rows[m];  // decode: visible = trows = (pos+1)/ratio
    s.kptr[m] = kcache; s.trows[m] = T_rows[m]; s.visible[m] = T_rows[m];
  }
  CUDA_CHECK(cudaMemcpyAsync(B.tab, h.data(), sizeof(k::DecRow) * B.M, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(B.soa, &s, sizeof(s), cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
}
enum Kind { KA = 0, KB = 1, KC = 2, KCT = 3 };
const char* kind_name(int k) { return k == KA ? "A L2(r2,full)" : k == KB ? "B L20(r1,cand src)" : k == KC ? "C L24(pool,bf16)" : "C L24(pool,tc)"; }
k::DecAttnArgs make_args(Bufs& B, int kind, int Tmax, int lctx) {
  k::DecAttnArgs a;
  a.M = B.M; a.Hi = HI; a.Di = DI; a.index_topk = ITOPK; a.win = WIN; a.idx_stride = WIN + ITOPK;
  a.ratio = kind == KA ? 2 : 1; a.layer = kind == KA ? 2 : kind == KB ? CAND_SRC : 24; a.cand_source_layer = CAND_SRC; a.cand_block = BS; a.cand_topk_blocks = CB;
  a.kv_source = kind <= KB; a.index_source = true; a.have_candidates = kind >= KC; a.Tmax = Tmax; a.idx_mode = kind == KCT ? 1 : 0;
  a.iq = B.iq; a.iw = B.iw; a.iqp = B.iqp; a.iscore = B.iscore; a.bmax = B.bmax; a.cand = B.cand; a.idx = B.idx;
  a.tab = B.tab; a.soa = B.soa; a.scratch = B.scratch; a.scratch_bytes = B.scratch_bytes; a.attn_cnt = B.attn_cnt; a.lctx = lctx;
  return a;
}
struct Out { std::vector<uint8_t> iscore, bmax, cand, idx, cnt; };
void reset_outputs(Bufs& B, int kind, int Tmax, cudaStream_t st) {
  CUDA_CHECK(cudaMemsetAsync(B.iscore, 0xA5, (size_t)B.M * (Tmax + 8) * 2, st));
  CUDA_CHECK(cudaMemsetAsync(B.bmax, 0x5A, (size_t)B.M * ((Tmax + BS - 1) / BS) * 4, st));
  CUDA_CHECK(cudaMemsetAsync(B.idx, 0x7B, (size_t)B.M * (WIN + ITOPK) * 4, st));
  if (kind >= KC) CUDA_CHECK(cudaMemcpyAsync(B.cand, B.cand_in, (size_t)B.M * CB * 4, cudaMemcpyDeviceToDevice, st));
  else CUDA_CHECK(cudaMemsetAsync(B.cand, 0x3C, (size_t)B.M * CB * 4, st));
}
Out snap(Bufs& B, int kind, int Tmax) {
  Out o;
  const int nblocks = (Tmax + BS - 1) / BS, ncols = kind >= KC ? std::min(CB, nblocks) * BS : Tmax;
  o.iscore = host(B.iscore, (size_t)B.M * ncols * 2);
  o.bmax = host(B.bmax, (size_t)B.M * nblocks * 4);
  o.cand = host(B.cand, (size_t)B.M * CB * 4);
  o.idx = host(B.idx, (size_t)B.M * (WIN + ITOPK) * 4);
  o.cnt = host(B.attn_cnt, (size_t)B.M * 64 * 4);
  return o;
}
bool same(const Out& a, const Out& b, std::string& why) {
  const size_t d0 = diff(a.iscore, b.iscore), d1 = diff(a.bmax, b.bmax), d2 = diff(a.cand, b.cand), d3 = diff(a.idx, b.idx), d4 = diff(a.cnt, b.cnt);
  char buf[256];
  snprintf(buf, sizeof(buf), "iscore %zu · bmax %zu · cand %zu · idx %zu · cnt %zu bytes differ", d0, d1, d2, d3, d4);
  why = buf;
  return d0 + d1 + d2 + d3 + d4 == 0;
}

// ---- (2) top-k alone -------------------------------------------------------------------------------------------------------------
template <class KeyT> KeyT mk(float x);
template <> float mk<float>(float x) { return x; }
template <> bf16 mk<bf16>(float x) { return f2bf(x); }
template <class KeyT>
void topk_unit(const char* tname, int M, int T_in, int k, int kind, bool use_trows, bool map_offset, int reps, cudaStream_t st, uint8_t* scratch, size_t scratch_n,
               int* cnt, k::DecRow* tab) {
  std::mt19937 rng(1234 + T_in * 7 + kind * 131 + M);
  std::uniform_real_distribution<float> U(0.f, 1.f);
  const int stride = T_in;
  std::vector<KeyT> hk((size_t)M * stride);
  for (auto& v : hk) {
    float x;
    switch (kind) {
      case 0: x = U(rng) * 20.f - 5.f; break;
      case 1: x = (float)(int)(U(rng) * 8.f); break;
      case 2: x = 2.f + U(rng) * 1.5f; break;
      case 3: { const float r = U(rng); x = r < 0.5f ? -INFINITY : r < 0.51f ? NAN : (r < 0.52f ? -NAN : U(rng) * 4.f); break; }
      case 4: x = 1.25f; break;
      default: { const float r = U(rng); x = r < 0.98f ? -INFINITY : r < 0.99f ? -NAN : U(rng) * 4.f; }
    }
    v = mk<KeyT>(x);
  }
  std::vector<k::DecRow> ht(M);
  for (int m = 0; m < M; ++m) { memset(&ht[m], 0, sizeof(k::DecRow)); ht[m].trows = use_trows ? std::max(0, T_in - m * 4099 - (m ? 1 : 0)) : T_in; ht[m].visible = ht[m].trows - m * 3; }
  KeyT* dk = dmalloc<KeyT>(hk.size());
  int32_t *o0 = dmalloc<int32_t>((size_t)M * (k + 4)), *o1 = dmalloc<int32_t>((size_t)M * (k + 4));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), hk.size() * sizeof(KeyT), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tab, ht.data(), M * sizeof(k::DecRow), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(o0, 0x11, (size_t)M * (k + 4) * 4)); CUDA_CHECK(cudaMemset(o1, 0x11, (size_t)M * (k + 4) * 4));
  auto old_f = [&](int32_t* o) {
    if constexpr (sizeof(KeyT) == 2) k::dec_topk_bf16(dk, M, T_in, use_trows, tab, k, stride, o, k + 4, nullptr, 0, 1, 133, map_offset, st);
    else k::dec_topk_f32(dk, M, T_in, use_trows, tab, k, stride, o, k + 4, nullptr, 0, 1, 133, map_offset, st);
  };
  bool took = true;
  auto new_f = [&](int32_t* o) {
    if constexpr (sizeof(KeyT) == 2) took = k::dec_topk2_bf16(dk, M, T_in, use_trows, tab, k, stride, o, k + 4, nullptr, 0, 1, 133, map_offset, scratch, scratch_n, cnt, st);
    else took = k::dec_topk2_f32(dk, M, T_in, use_trows, tab, k, stride, o, k + 4, nullptr, 0, 1, 133, map_offset, scratch, scratch_n, cnt, st);
  };
  char nm[160];
  snprintf(nm, sizeof(nm), "topk %s M %d T_in %d k %d kind %d trows %d", tname, M, T_in, k, kind, (int)use_trows);
  run_sync(std::string(nm) + " old", st, [&] { old_f(o0); });
  run_sync(std::string(nm) + " new", st, [&] { new_f(o1); });
  const std::vector<uint8_t> h0 = host(o0, (size_t)M * (k + 4) * 4), h1 = host(o1, (size_t)M * (k + 4) * 4);
  const size_t d = diff(h0, h1);
  const size_t dc = diff(host(cnt, (size_t)M * 4), std::vector<uint8_t>((size_t)M * 4, 0));
  check(took && d == 0 && dc == 0, std::string(nm) + (took ? "" : " (new path refused)") + " diff bytes " + std::to_string(d) + " cnt " + std::to_string(dc));
  if (d) {  // T1: first mismatch (row, position, previous/new value, key bits of that column) + per-row count of differing slots and of -1 — for attributing the cause
    const int32_t* a = reinterpret_cast<const int32_t*>(h0.data());
    const int32_t* b = reinterpret_cast<const int32_t*>(h1.data());
    auto kbits = [&](int m, int32_t p) -> unsigned {
      if (map_offset || p < 0 || p >= T_in) return 0;
      unsigned u = 0; memcpy(&u, &hk[(size_t)m * stride + p], sizeof(KeyT)); return u;
    };
    for (int m = 0; m < M; ++m) {
      int first = -1, nd = 0, neg0 = 0, neg1 = 0;
      for (int j = 0; j < k + 4; ++j) {
        const int32_t x = a[(size_t)m * (k + 4) + j], y = b[(size_t)m * (k + 4) + j];
        if (x != y) { ++nd; if (first < 0) first = j; }
        if (j < k) { neg0 += x == -1; neg1 += y == -1; }
      }
      if (first < 0) continue;
      const int32_t x = a[(size_t)m * (k + 4) + first], y = b[(size_t)m * (k + 4) + first];
      printf("    row %d: first diff pos %d · old %d (key 0x%x) · new %d (key 0x%x) · %d slots differ · -1 slots old %d new %d\n", m, first, x, kbits(m, x), y,
             kbits(m, y), nd, neg0, neg1);
    }
    fflush(stdout);
  }
  const double t0 = time_us([&] { old_f(o0); }, reps, st, nullptr, 0), t1 = time_us([&] { new_f(o1); }, reps, st, nullptr, 0);
  const k::Topk2Plan p = k::topk2_plan(T_in, k, M, sizeof(KeyT));
  printf("  %-52s CH %5d nc %2d : %s · old %7.1f µs · new %7.1f µs · x%.2f\n", nm, p.CH, p.ncmax, (took && d == 0 && dc == 0) ? "same" : "DIFF", t0, t1, t0 / t1);
  fflush(stdout);
  cudaFree(dk); cudaFree(o0); cudaFree(o1);
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::max(1, atoi(argv[1])) : 20;
  std::vector<int> ctxs = {4096, 32768, 81920, 102400}, Ms = {1, 4, 8};
  auto parse = [](const char* s) { std::vector<int> v; for (const char* p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } return v; };
  if (argc > 2) ctxs = parse(argv[2]);
  if (argc > 3) Ms = parse(argv[3]);
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("device %s · SMs %d · L2 %d MB · reps %d · (the test picks the version directly, independent of the switch — DecAttnArgs::lctx)\n", prop.name, prop.multiProcessorCount,
         prop.l2CacheSize >> 20, reps);
  cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const size_t flush_n = (size_t)512 << 20;
  uint8_t* flush = dmalloc<uint8_t>(flush_n);
  int Lmax = 0; for (int L : ctxs) Lmax = std::max(Lmax, L);
  const int Tcap = next_pow2(Lmax), Mmax = *std::max_element(Ms.begin(), Ms.end());
  Bufs B;
  B.Tcap = Tcap;
  B.keys2 = dmalloc<uint8_t>((size_t)(Tcap / 2 + 8) * kvp::IDX_ROW);
  B.keys1 = dmalloc<uint8_t>((size_t)(Tcap + 8) * kvp::IDX_ROW);
  fill_idx_rows_kernel<<<nblk((size_t)(Tcap / 2 + 8) * kvp::IDX_ROW), 256>>>(B.keys2, Tcap / 2 + 8, 11);
  fill_idx_rows_kernel<<<nblk((size_t)(Tcap + 8) * kvp::IDX_ROW), 256>>>(B.keys1, Tcap + 8, 12);
  B.iq = dmalloc<bf16>((size_t)Mmax * HI * DI); B.iw = dmalloc<bf16>((size_t)Mmax * HI);
  fill_bf16_kernel<<<nblk((size_t)Mmax * HI * DI), 256>>>(B.iq, (size_t)Mmax * HI * DI, 21, 0.f, 3.f);
  fill_bf16_kernel<<<1, 256>>>(B.iw, (size_t)Mmax * HI, 22, 0.02f, 0.2f);
  B.iqp = dmalloc<uint8_t>((size_t)Mmax * HI * kvp::IDX_ROW);
  fill_idx_rows_kernel<<<nblk((size_t)Mmax * HI * kvp::IDX_ROW), 256>>>(B.iqp, (size_t)Mmax * HI, 23);
  CUDA_CHECK(cudaGetLastError());
  B.iscore = dmalloc<bf16>((size_t)Mmax * (Tcap + 8));
  B.bmax = dmalloc<float>((size_t)Mmax * (Tcap / BS + 1));
  B.cand = dmalloc<int32_t>((size_t)Mmax * CB); B.cand_in = dmalloc<int32_t>((size_t)Mmax * CB);
  B.idx = dmalloc<int32_t>((size_t)Mmax * (WIN + ITOPK));
  B.scratch_bytes = (size_t)Mmax * (Tcap + 8) * 4;  // same size as production (Work::iscore_f = Msub·(Tcap+8)·4)
  B.scratch = dmalloc<uint8_t>(B.scratch_bytes);
  B.attn_cnt = dmalloc<int>((size_t)std::max(Mmax, 8) * 64);
  B.tab = dmalloc<k::DecRow>(std::max(Mmax, 8)); B.soa = dmalloc<k::DecSoA>(1);  // (2) needs up to M 8
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- (1) indexer stage ----
  printf("\n== ① indexer stage (dec_indexer_stage) — version 0 baseline · 1 IDXSCORE2 · 2 TOPK2 · 3 both · µs/layer (cold = L2 flushed · warm = back to back) ==\n");
  struct StepRow { int L, M, bucket; double old_c, new_c, old_w, new_w; };
  std::vector<StepRow> steps;
  for (int L : ctxs)
    for (int M : Ms)
      for (int bucket = 1; bucket >= 0; --bucket) {
        B.M = M;
        double tc[4][4] = {}, tw[4][4] = {};
        for (int kind = KA; kind <= KCT; ++kind) {
          const int T = kind == KA ? L / 2 : L;
          const int Tmax = bucket ? std::min(Tcap, std::max(1024, next_pow2(T))) : T;
          std::vector<int> Tr(M);
          for (int m = 0; m < M; ++m) Tr[m] = std::max(1, T - m * 1001);  // a different context per row (batched decode)
          set_rows(B, kind == KA ? B.keys2 : B.keys1, Tr, st);
          if (kind == KC) {  // candidate-pool layer input = the candidate table of layer 20's previous output (same key cache, same T)
            k::DecAttnArgs a20 = make_args(B, KB, Tmax, 0);
            reset_outputs(B, KB, Tmax, st);
            run_sync("cand table (L20 old)", st, [&] { k::dec_indexer_stage(a20, st); });
            CUDA_CHECK(cudaMemcpy(B.cand_in, B.cand, (size_t)M * CB * 4, cudaMemcpyDeviceToDevice));
          }
          Out ref;
          for (int v = 0; v < 4; ++v) {
            k::DecAttnArgs a = make_args(B, kind, Tmax, v);
            reset_outputs(B, kind, Tmax, st);
            char nm[160];
            snprintf(nm, sizeof(nm), "%s L %d M %d Tmax %d v%d", kind_name(kind), L, M, Tmax, v);
            run_sync(nm, st, [&] { k::dec_indexer_stage(a, st); });
            Out o = snap(B, kind, Tmax);
            if (v == 0) ref = std::move(o);
            else { std::string why; const bool ok = same(ref, o, why); check(ok, std::string(nm) + ": " + why); if (!ok) printf("  %s: %s\n", nm, why.c_str()); }
            tc[kind][v] = time_us([&] { k::dec_indexer_stage(a, st); }, reps, st, flush, flush_n);
            tw[kind][v] = time_us([&] { k::dec_indexer_stage(a, st); }, reps, st, nullptr, 0);
          }
          {  // graph: capture version 3, replay 3 times -> same as the previous output?
            k::DecAttnArgs a = make_args(B, kind, Tmax, 3);
            reset_outputs(B, kind, Tmax, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            k::dec_indexer_stage(a, st);
            CUDA_CHECK(cudaStreamEndCapture(st, &g));
            CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
            for (int r = 0; r < 3; ++r) CUDA_CHECK(cudaGraphLaunch(ge, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            Out o = snap(B, kind, Tmax);
            std::string why;
            char nm[160];
            snprintf(nm, sizeof(nm), "%s L %d M %d Tmax %d v3 graph×3", kind_name(kind), L, M, Tmax);
            const bool ok = same(ref, o, why);
            check(ok, std::string(nm) + ": " + why);
            if (!ok) printf("  %s: %s\n", nm, why.c_str());
            cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
          }
          printf("  %-20s L %6d M %d Tmax %6d%s | cold µs: v0 %7.1f v1 %7.1f v2 %7.1f v3 %7.1f (x%.2f) | warm µs: v0 %7.1f v3 %7.1f (x%.2f)\n", kind_name(kind), L, M,
                 Tmax, bucket ? "(bkt)" : "     ", tc[kind][0], tc[kind][1], tc[kind][2], tc[kind][3], tc[kind][0] / tc[kind][3], tw[kind][0], tw[kind][3],
                 tw[kind][0] / tw[kind][3]);
          fflush(stdout);
        }
        // Step estimate (production = HIVE_IDX_TC on): layers 2, 8, 14 (A) x3 + layer 20 (B) + layers 24, 28, 32, 36 (C tc) x4
        StepRow r{L, M, bucket, 3 * tc[KA][0] + tc[KB][0] + 4 * tc[KCT][0], 3 * tc[KA][3] + tc[KB][3] + 4 * tc[KCT][3],
                  3 * tw[KA][0] + tw[KB][0] + 4 * tw[KCT][0], 3 * tw[KA][3] + tw[KB][3] + 4 * tw[KCT][3]};
        steps.push_back(r);
      }
  printf("\n== step estimate (8 indexer layers = 3·A + B + 4·C(tc)) — µs/step ==\n");
  printf("  %7s %2s %6s | %9s %9s %9s | %9s %9s %9s\n", "ctx", "M", "Tmax", "cold old", "cold new", "saved", "warm old", "warm new", "saved");
  for (const auto& r : steps)
    printf("  %7d %2d %6s | %9.1f %9.1f %9.1f | %9.1f %9.1f %9.1f\n", r.L, r.M, r.bucket ? "bucket" : "exact", r.old_c, r.new_c, r.old_c - r.new_c, r.old_w, r.new_w,
           r.old_w - r.new_w);

  // ---- (2) top-k alone ----
  printf("\n== ② top-k alone (baseline dec_topk vs dec_topk2 · warm µs) ==\n");
  for (int T_in : {4096, 4097, 8192, 8193, 16384, 40960, 65536, 81920, 102400, 131072})
    for (int kind = 0; kind < 6; ++kind)
      topk_unit<bf16>("bf16", 1, T_in, ITOPK, kind, kind % 2 == 0, true, reps, st, (uint8_t*)B.scratch, B.scratch_bytes, B.attn_cnt, B.tab);
  for (int T_in : {16384, 81920, 131072}) topk_unit<bf16>("bf16", 8, T_in, ITOPK, 2, true, true, reps, st, (uint8_t*)B.scratch, B.scratch_bytes, B.attn_cnt, B.tab);
  for (int T_in : {512, 4096, 5120, 10240, 12800, 16384})
    for (int kind = 0; kind < 6; ++kind)
      topk_unit<float>("f32", kind == 2 ? 4 : 1, T_in, std::min(CB, T_in), kind, false, false, reps, st, (uint8_t*)B.scratch, B.scratch_bytes, B.attn_cnt, B.tab);

  // ---- (3) scores alone (KPT) ----
  printf("\n== ③ scores alone (baseline vs IDXSCORE2 KPT 1·2·4 · cold µs) — all rows (T = L) · candidate pool (16,384 columns) ==\n");
  for (int L : ctxs)
    for (int M : {1, 4}) {
      if (M > Mmax) continue;
      B.M = M;
      std::vector<int> Tr(M, L);
      set_rows(B, B.keys1, Tr, st);
      for (int cm = 0; cm < 2; ++cm) {
        const int ncols = cm ? std::min(CB, (L + BS - 1) / BS) * BS : std::min(Tcap, std::max(1024, next_pow2(L)));
        if (cm) {  // candidate table: distinct blocks (scattered positions)
          std::vector<int32_t> hc((size_t)M * CB);
          for (size_t i = 0; i < hc.size(); ++i) hc[i] = (int32_t)((i * 7919) % std::max(1, (L + BS - 1) / BS));
          CUDA_CHECK(cudaMemcpy(B.cand, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemsetAsync(B.iscore, 0xA5, (size_t)M * (ncols + 8) * 2, st));
        run_sync("scores ref", st, [&] { k::dec_idx_scores_ref(B.iq, B.iw, B.tab, M, ncols, cm != 0, B.cand, CB, BS, B.iscore, st); });
        const auto ref = host(B.iscore, (size_t)M * ncols * 2);
        const double t_old = time_us([&] { k::dec_idx_scores_ref(B.iq, B.iw, B.tab, M, ncols, cm != 0, B.cand, CB, BS, B.iscore, st); }, reps, st, flush, flush_n);
        printf("  %-4s L %6d M %d ncols %6d | old %7.1f µs |", cm ? "pool" : "full", L, M, ncols, t_old);
        for (int kpt : {1, 2, 4}) {
          CUDA_CHECK(cudaMemsetAsync(B.iscore, 0xA5, (size_t)M * (ncols + 8) * 2, st));
          bool took = false;
          run_sync("scores2 kpt " + std::to_string(kpt), st, [&] { took = k::dec_idx_scores2(B.iq, B.iw, B.tab, M, ncols, cm != 0, B.cand, CB, BS, B.iscore, kpt, st); });
          const size_t d = diff(ref, host(B.iscore, (size_t)M * ncols * 2));
          check(took && d == 0, std::string("scores2 ") + (cm ? "pool" : "full") + " L " + std::to_string(L) + " M " + std::to_string(M) + " kpt " + std::to_string(kpt) +
                                    " diff " + std::to_string(d));
          const double t = time_us([&] { k::dec_idx_scores2(B.iq, B.iw, B.tab, M, ncols, cm != 0, B.cand, CB, BS, B.iscore, kpt, st); }, reps, st, flush, flush_n);
          printf(" kpt %d %7.1f µs %s |", kpt, t, (took && d == 0) ? "same" : "DIFF");
        }
        printf("\n");
        fflush(stdout);
      }
    }
  printf("\n%d cases — %s\n", g_cases, g_fail ? "SOME FAIL" : "ALL PASS");
  return g_fail ? 1 : 0;
}
