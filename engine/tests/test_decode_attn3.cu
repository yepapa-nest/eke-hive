// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// A2 validation (GPU): the four new variants in decode_attn3.cu vs the current production kernels — real layer shapes
// (dim 5120 · H 64 · D 512 · q_lora 1280 · E 384 · top 6 · hc 4).
//   Per kernel (byte comparison — bit-identical by design, no tolerance · µs after flushing L2 (128 MB) with a 512 MB memset):
//     router   attn2_router (A1, production) · fused_router (older)  vs attn3_router (+ tied gate_w) · cost breakdown:
//              phase 1 (gate rows + xn load only) · 2 (+ dot products) · 0 (full), for both the xn-in-smem variant
//              (mode 1) and the register variant (mode 0) — side by side with production A1
//     hc       hc_mix_pre_norm (HIVE_HC_FUSE2, production)   vs hc_mix_pre_norm3
//     qkv_a    dec_qkv_a_kernel (D2, production)             vs attn3_qkv_a   (qr·qrn·kv·xf·ring slot·window idx·row table copy)
//     attn     dec_attn_kernel (D2, production)              vs attn3_sparse G = auto·1·2·4·8 (layer W: window only S=1 ·
//              C1: compressed ratio 1 · C2: ratio 2, S=4) × context 4K·32K·128K (compressed cache = context/ratio rows —
//              selected columns at scattered positions)
//   M = 1·4·8. Chain: the decode order of one layer hc_mix -> q_a‖kv -> q_b (A1) -> sparse attention -> wo_a -> wo_b (A1)
//     -> hc_mix -> router -> shared experts (A1) × NL weight copies (outside L2) — the 4 old / 4 new kernels (the rest are
//     the same production kernels), eager and CUDA-graph replay · output bit comparison + µs per layer.
//   Every launch prints "[run] name variant M" first and synchronizes (on an abnormal exit the variant is on record).
//   Verdict: all bit-identical = ALL PASS. Run: scripts/hive-run.sh "./build-dev/test_decode_attn3 [reps=20] [NL=4] [ctx list=4096,32768,131072]"
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
#include "hive/decode_attn2.h"
#include "hive/decode_attn3.h"
#include "hive/kernels.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

constexpr int DIM = 5120, H = 64, HD = 512, RD = 64, QL = 1280, OG = 8, OR = 1024, INTER = 2304, E = 384, TOPK = 6, HC = 4, MIX = (2 + HC) * HC;
constexpr int WIN = 128, ITOPK = 512, SPLITS = 4, SUB = H * HD / OG;
constexpr float LIMIT = 10.f, EPS = 1e-20f, HC_EPS = 1e-6f, ROUTE_SCALE = 1.5f;

__device__ __forceinline__ uint32_t hsh(size_t i, uint32_t seed) {
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed ^ (uint32_t)(i >> 32) * 0x9E3779B9u;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  return h;
}
__global__ void fill_u8_kernel(uint8_t* p, size_t n, uint32_t seed, int mode) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) {
    const uint32_t h = hsh(i, seed);
    uint8_t v = (uint8_t)h;
    if (mode == 1) v = (uint8_t)((v & 0x80) | ((v & 0x7F) % 0x7E));          // e4m3 weights (no NaN)
    else if (mode == 3) v = (uint8_t)(118 + (h >> 8) % 6);                    // e8m0 scales
    else if (mode == 4) v = (i % kvp::COMP_ROW) < kvp::COMP_SCALES ? (uint8_t)(0x20 + h % 24) : (uint8_t)h;  // packed compressed KV rows
    p[i] = v;
  }
}
__global__ void fill_bf16_kernel(bf16* p, size_t n, uint32_t seed, float a, float b, int spice) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) {
    const uint32_t h = hsh(i, seed);
    float v = a + b * ((float)(h >> 8) * (2.f / 16777216.f) - 1.f);
    if (spice) {
      const uint32_t r = h & 1023u;
      if (r == 1) v = 0.f; else if (r == 2) v = -0.f; else if (r == 3) v *= 300.f; else if (r == 4) v *= 1e-30f;
    }
    p[i] = f2bf(v);
  }
}
__global__ void fill_f32_kernel(float* p, size_t n, uint32_t seed, float a, float b, int dup) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) {
    size_t j = i;
    if (dup > 1) { const size_t row = i / DIM, col = i % DIM; j = (row / dup) * DIM + col; }
    p[i] = a + b * ((float)(hsh(j, seed) >> 8) * (2.f / 16777216.f) - 1.f);
  }
}
unsigned nblk(size_t n) { return (unsigned)std::min<size_t>(8192, (n + 255) / 256); }
void fill_u8(void* p, size_t n, uint32_t seed, int mode) { fill_u8_kernel<<<nblk(n), 256>>>((uint8_t*)p, n, seed, mode); CUDA_CHECK(cudaGetLastError()); }
void fill_bf16(bf16* p, size_t n, uint32_t seed, float a, float b, int spice = 1) { fill_bf16_kernel<<<nblk(n), 256>>>(p, n, seed, a, b, spice); CUDA_CHECK(cudaGetLastError()); }
void fill_f32(float* p, size_t n, uint32_t seed, float a, float b, int dup = 1) { fill_f32_kernel<<<nblk(n), 256>>>(p, n, seed, a, b, dup); CUDA_CHECK(cudaGetLastError()); }
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T) + 256)); CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T) + 256)); return p; }
template <class T> T* mapped(size_t n, T** dev) { T* p; CUDA_CHECK(cudaHostAlloc((void**)&p, n * sizeof(T), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void**)dev, p, 0)); memset(p, 0, n * sizeof(T)); return p; }
std::vector<uint8_t> host(const void* d, size_t bytes) { std::vector<uint8_t> v(bytes); CUDA_CHECK(cudaMemcpy(v.data(), d, bytes, cudaMemcpyDeviceToHost)); return v; }
size_t diff_bytes(const void* p, const void* q, size_t n) {
  const auto a = host(p, n), b = host(q, n);
  size_t d = 0;
  for (size_t i = 0; i < n; ++i) d += a[i] != b[i];
  return d;
}
std::vector<float2> rope_host(int npos, float theta) {
  std::vector<float2> fr((size_t)npos * (RD / 2));
  for (int p = 0; p < npos; ++p)
    for (int i = 0; i < RD / 2; ++i) {
      const float ang = fmodf((float)p * powf(theta, -2.f * i / RD), 6.2831853f);
      fr[(size_t)p * (RD / 2) + i] = make_float2(cosf(ang), sinf(ang));
    }
  return fr;
}

// launch then synchronize (prints the variant name first — attributes an abnormal exit)
void run_sync(const char* name, const char* variant, int M, cudaStream_t st, const std::function<void()>& f) {
  printf("[run] %s %s M=%d\n", name, variant, M);
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
double graph_us(const std::function<void()>& body, int reps, cudaStream_t st, int per) {
  cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
  cudaError_t e = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
  if (e == cudaSuccess) { body(); e = cudaStreamEndCapture(st, &g); }
  if (e == cudaSuccess) e = cudaGraphInstantiate(&ge, g, 0);
  double t = -1;
  if (e == cudaSuccess) {
    CUDA_CHECK(cudaGraphLaunch(ge, st)); CUDA_CHECK(cudaStreamSynchronize(st));
    t = time_us([&] { CUDA_CHECK(cudaGraphLaunch(ge, st)); }, reps, st, nullptr, 0) / per;
  } else {
    fprintf(stderr, "graph capture/instantiate failed: %s\n", cudaGetErrorString(e));
    (void)cudaGetLastError();
  }
  if (ge) cudaGraphExecDestroy(ge);
  if (g) cudaGraphDestroy(g);
  return t;
}
std::vector<int> parse_list(const char* s, std::vector<int> def) {
  if (!s || !*s) return def;
  std::vector<int> v;
  std::string t(s);
  size_t p = 0;
  while (p < t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size(); v.push_back(atoi(t.substr(p, q - p).c_str())); p = q + 1; }
  return v;
}

// ---- layer weights (one set) ----
struct Fp8 { uint8_t *w, *s; };
Fp8 make_fp8(size_t n, size_t k, uint32_t seed) {
  Fp8 f{dmalloc<uint8_t>(n * k), dmalloc<uint8_t>(n * (k / 32))};
  fill_u8(f.w, n * k, seed, 1); fill_u8(f.s, n * (k / 32), seed * 7919u + 1, 3);
  return f;
}
struct LayerW {
  Fp8 wqa, wkv, wqb, woa, wob, w1, w3, w2;
  float *gate, *bias, *bias_vl, *hc_attn, *hc_ffn, *sink;
  bf16 *q_norm, *kv_norm, *attn_norm, *ffn_norm;
};
LayerW make_layer(uint32_t seed) {
  LayerW L;
  L.wqa = make_fp8(QL, DIM, seed + 1); L.wkv = make_fp8(HD, DIM, seed + 2); L.wqb = make_fp8((size_t)H * HD, QL, seed + 3);
  L.woa = make_fp8((size_t)OG * OR, SUB, seed + 4); L.wob = make_fp8(DIM, (size_t)OG * OR, seed + 5);
  L.w1 = make_fp8(INTER, DIM, seed + 6); L.w3 = make_fp8(INTER, DIM, seed + 7); L.w2 = make_fp8(DIM, INTER, seed + 8);
  L.gate = dmalloc<float>((size_t)E * DIM); fill_f32(L.gate, (size_t)E * DIM, seed + 9, 0.f, 0.03f);
  L.bias = dmalloc<float>(E); fill_f32(L.bias, E, seed + 10, 0.f, 0.05f);
  L.bias_vl = dmalloc<float>(E); fill_f32(L.bias_vl, E, seed + 11, 0.f, 0.05f);
  L.hc_attn = dmalloc<float>((size_t)MIX * HC * DIM); fill_f32(L.hc_attn, (size_t)MIX * HC * DIM, seed + 12, 0.f, 0.01f);
  L.hc_ffn = dmalloc<float>((size_t)MIX * HC * DIM); fill_f32(L.hc_ffn, (size_t)MIX * HC * DIM, seed + 13, 0.f, 0.01f);
  L.sink = dmalloc<float>(H); fill_f32(L.sink, H, seed + 14, 0.2f, 0.7f);
  L.q_norm = dmalloc<bf16>(QL); fill_bf16(L.q_norm, QL, seed + 15, 1.f, 0.2f, 0);
  L.kv_norm = dmalloc<bf16>(HD); fill_bf16(L.kv_norm, HD, seed + 16, 1.f, 0.2f, 0);
  L.attn_norm = dmalloc<bf16>(DIM); fill_bf16(L.attn_norm, DIM, seed + 17, 1.f, 0.1f, 0);
  L.ffn_norm = dmalloc<bf16>(DIM); fill_bf16(L.ffn_norm, DIM, seed + 18, 1.f, 0.1f, 0);
  return L;
}

// ---- per-row (sequence) KV state + mapped pinned tables ----
struct Rows {
  int ctx = 0, ratio = 1, M = 1;
  std::vector<bf16*> ring;      // [WIN, HD] per row
  std::vector<uint8_t*> comp;   // [ctx/ratio, 288]
  k::KvRow* kv_h; k::KvRow* kv_d;
  int32_t *pos_h, *pos_d, *vis_h, *vis_d, *trows_h, *trows_d;
  bf16 **ringp_h, **ringp_d;
  std::vector<int> pos;
};
void make_rows(Rows& r, int ctx, int ratio, int M, uint32_t seed) {
  r.ctx = ctx; r.ratio = ratio; r.M = M;
  r.kv_h = mapped<k::KvRow>(8, &r.kv_d); r.pos_h = mapped<int32_t>(8, &r.pos_d); r.vis_h = mapped<int32_t>(8, &r.vis_d);
  r.trows_h = mapped<int32_t>(8, &r.trows_d); r.ringp_h = mapped<bf16*>(8, &r.ringp_d);
  for (int m = 0; m < 8; ++m) {
    bf16* ring = dmalloc<bf16>((size_t)WIN * HD); fill_bf16(ring, (size_t)WIN * HD, seed + 100 + m, 0.f, 2.f);
    r.ring.push_back(ring);
    const size_t T = ratio ? (size_t)ctx / ratio + 8 : 8;
    uint8_t* c = dmalloc<uint8_t>(T * kvp::COMP_ROW); fill_u8(c, T * kvp::COMP_ROW, seed + 200 + m, 4);
    r.comp.push_back(c);
    const int p = ctx - 1 - 3 * m;
    r.pos.push_back(p);
    r.pos_h[m] = p;
    const int comp_len = ratio ? (p + 1) / ratio : 0;
    r.vis_h[m] = comp_len; r.trows_h[m] = comp_len;
    r.kv_h[m].ring = ring; r.kv_h[m].comp = ratio ? c : nullptr; r.kv_h[m].comp_len = comp_len; r.kv_h[m].topk = ratio ? std::min(ITOPK, comp_len) : 0;
    r.ringp_h[m] = ring;
  }
}
void free_rows(Rows& r) {
  for (auto p : r.ring) cudaFree(p);
  for (auto p : r.comp) cudaFree(p);
  cudaFreeHost(r.kv_h); cudaFreeHost(r.pos_h); cudaFreeHost(r.vis_h); cudaFreeHost(r.trows_h); cudaFreeHost(r.ringp_h);
}
// idx (window + top-k columns): window = window_idxs_rows formula · compressed = scattered positions in ascending order
// (shape of the indexer top-k output), occasional -1
std::vector<int32_t> make_idx(const Rows& r, int M, uint32_t seed) {
  std::vector<int32_t> idx((size_t)8 * (WIN + ITOPK), -1);
  std::mt19937 rng(seed);
  for (int m = 0; m < M; ++m) {
    const int p = r.pos[m];
    int32_t* ir = idx.data() + (size_t)m * (WIN + ITOPK);
    for (int j = 0; j < WIN; ++j) { const long srcp = (long)p - (WIN - 1) + j; ir[j] = srcp < 0 ? -1 : (srcp < p ? (int32_t)(srcp % WIN) : (int32_t)(WIN + m)); }
    const int comp_len = r.kv_h[m].comp_len, topk = r.kv_h[m].topk;
    std::vector<int> sel;
    if (topk > 0) {
      std::uniform_int_distribution<int> u(0, comp_len - 1);
      std::vector<char> used(comp_len, 0);
      while ((int)sel.size() < topk) { const int x = u(rng); if (!used[x]) { used[x] = 1; sel.push_back(x); } }
      std::sort(sel.begin(), sel.end());
    }
    for (int j = 0; j < topk; ++j) ir[WIN + j] = (j % 97 == 13) ? -1 : sel[j] + WIN + M;
  }
  return idx;
}

// one set of activation/work buffers (one each for the old and new variants)
struct Acts {
  bf16 *h, *x, *xn, *qr, *qrn, *q, *kv, *o, *og, *attn_out, *y;
  float *xf, *pre, *mixes, *rsq, *scores, *rw, *acc, *pacc, *pm, *ps;
  uint8_t *qrq, *qrs, *xq, *xs;
  int32_t *ids, *idx;
  int *counters, *attn_cnt;
  int8_t* is_image;
  k::DecRow* tab;
  k::DecSoA* soa;
  std::vector<bf16*> ring;  // ring private to this variant (ring-slot writes are compared separately)
  bf16** ringp_h; bf16** ringp_d;
};
Acts make_acts(const Rows& r) {
  Acts a;
  a.h = dmalloc<bf16>((size_t)8 * HC * DIM); a.x = dmalloc<bf16>((size_t)8 * DIM); a.xn = dmalloc<bf16>((size_t)8 * DIM);
  a.qr = dmalloc<bf16>((size_t)8 * QL); a.qrn = dmalloc<bf16>((size_t)8 * QL); a.q = dmalloc<bf16>((size_t)8 * H * HD); a.kv = dmalloc<bf16>((size_t)8 * HD);
  a.o = dmalloc<bf16>((size_t)8 * H * HD); a.og = dmalloc<bf16>((size_t)8 * OG * OR); a.attn_out = dmalloc<bf16>((size_t)8 * DIM); a.y = dmalloc<bf16>((size_t)8 * INTER);
  a.xf = dmalloc<float>((size_t)8 * DIM); a.pre = dmalloc<float>(8 * HC); a.mixes = dmalloc<float>(8 * MIX); a.rsq = dmalloc<float>(8);
  a.scores = dmalloc<float>((size_t)8 * E); a.rw = dmalloc<float>(8 * TOPK); a.acc = dmalloc<float>((size_t)8 * DIM);
  a.pacc = dmalloc<float>((size_t)8 * H * 16 * HD); a.pm = dmalloc<float>((size_t)8 * H * 16); a.ps = dmalloc<float>((size_t)8 * H * 16);
  a.qrq = dmalloc<uint8_t>((size_t)8 * QL); a.qrs = dmalloc<uint8_t>((size_t)8 * QL / 32); a.xq = dmalloc<uint8_t>((size_t)8 * DIM); a.xs = dmalloc<uint8_t>((size_t)8 * DIM / 32);
  a.ids = dmalloc<int32_t>(8 * TOPK); a.idx = dmalloc<int32_t>((size_t)8 * (WIN + ITOPK));
  a.counters = dmalloc<int>(16); a.attn_cnt = dmalloc<int>((size_t)8 * H); a.is_image = dmalloc<int8_t>(8);
  a.tab = dmalloc<k::DecRow>(8); a.soa = dmalloc<k::DecSoA>(1);
  a.ringp_h = mapped<bf16*>(8, &a.ringp_d);
  for (int m = 0; m < 8; ++m) {
    bf16* rg = dmalloc<bf16>((size_t)WIN * HD);
    CUDA_CHECK(cudaMemcpy(rg, r.ring[m], (size_t)WIN * HD * 2, cudaMemcpyDeviceToDevice));
    a.ring.push_back(rg); a.ringp_h[m] = rg;
  }
  const int8_t img[8] = {0, 1, 0, 0, 1, 0, 1, 0};
  CUDA_CHECK(cudaMemcpy(a.is_image, img, 8, cudaMemcpyHostToDevice));
  return a;
}
void free_acts(Acts& a) {
  for (void* p : {(void*)a.h, (void*)a.x, (void*)a.xn, (void*)a.qr, (void*)a.qrn, (void*)a.q, (void*)a.kv, (void*)a.o, (void*)a.og, (void*)a.attn_out, (void*)a.y,
                  (void*)a.xf, (void*)a.pre, (void*)a.mixes, (void*)a.rsq, (void*)a.scores, (void*)a.rw, (void*)a.acc, (void*)a.pacc, (void*)a.pm, (void*)a.ps,
                  (void*)a.qrq, (void*)a.qrs, (void*)a.xq, (void*)a.xs, (void*)a.ids, (void*)a.idx, (void*)a.counters, (void*)a.attn_cnt, (void*)a.is_image,
                  (void*)a.tab, (void*)a.soa})
    cudaFree(p);
  for (auto p : a.ring) cudaFree(p);
  cudaFreeHost(a.ringp_h);
}
void seed_acts(Acts& a, const Rows& r, const std::vector<int32_t>& idx, uint32_t seed, int junk) {
  fill_bf16(a.h, (size_t)8 * HC * DIM, seed + 1, 0.f, 1.f);
  fill_bf16(a.xn, (size_t)8 * DIM, seed + 2, 0.f, 1.5f);
  fill_bf16(a.q, (size_t)8 * H * HD, seed + 3, 0.f, 0.6f);
  fill_bf16(a.kv, (size_t)8 * HD, seed + 4, 0.f, 2.f);
  fill_f32(a.pre, 8 * HC, seed + 5, 0.25f, 0.1f);
  const std::pair<void*, size_t> outs[] = {{a.x, (size_t)8 * DIM * 2}, {a.qr, (size_t)8 * QL * 2}, {a.qrn, (size_t)8 * QL * 2}, {a.o, (size_t)8 * H * HD * 2},
                                           {a.og, (size_t)8 * OG * OR * 2}, {a.attn_out, (size_t)8 * DIM * 2}, {a.y, (size_t)8 * INTER * 2},
                                           {a.xf, (size_t)8 * DIM * 4}, {a.mixes, (size_t)8 * MIX * 4}, {a.rsq, 32}, {a.scores, (size_t)8 * E * 4},
                                           {a.rw, (size_t)8 * TOPK * 4}, {a.acc, (size_t)8 * DIM * 4}, {a.qrq, (size_t)8 * QL}, {a.qrs, (size_t)8 * QL / 32},
                                           {a.xq, (size_t)8 * DIM}, {a.xs, (size_t)8 * DIM / 32}, {a.ids, (size_t)8 * TOPK * 4},
                                           {a.tab, 8 * sizeof(k::DecRow)}, {a.soa, sizeof(k::DecSoA)}};
  for (const auto& o : outs) CUDA_CHECK(cudaMemset(o.first, junk, o.second));
  CUDA_CHECK(cudaMemcpy(a.idx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
  for (int m = 0; m < 8; ++m) CUDA_CHECK(cudaMemcpy(a.ring[m], r.ring[m], (size_t)WIN * HD * 2, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
}
// D2 front-stage arguments (filled like runtime attention_decode_dev — only the fields needed)
k::DecAttnArgs front_args(const LayerW& L, Acts& a, const Rows& r, int M, int ratio, bool kv_source, const float2* freqs) {
  k::DecAttnArgs d;
  d.M = M; d.dim = DIM; d.H = H; d.D = HD; d.rd = RD; d.win = WIN; d.q_lora = QL; d.Hi = 32; d.Di = 128; d.index_topk = ITOPK; d.ratio = ratio;
  d.kv_source = kv_source; d.eps = EPS; d.splits = (WIN + (ratio ? ITOPK : 0)) > 256 ? SPLITS : 1;
  d.wqa = L.wqa.w; d.sqa = L.wqa.s; d.wqb = L.wqb.w; d.sqb = L.wqb.s; d.wkv = L.wkv.w; d.skv = L.wkv.s; d.q_norm = L.q_norm; d.kv_norm = L.kv_norm; d.sink = L.sink;
  d.freqs = freqs; d.freqs_idx = freqs;
  d.xn = a.xn; d.xf = a.xf; d.qr = a.qr; d.qrn = a.qrn; d.q = a.q; d.kv = a.kv; d.qrq = a.qrq; d.qrs = a.qrs; d.idx = a.idx; d.idx_stride = WIN + ITOPK;
  d.o = a.o; d.pacc = a.pacc; d.pm = a.pm; d.ps = a.ps; d.counters = a.counters; d.attn_cnt = a.attn_cnt; d.tab = a.tab; d.soa = a.soa;
  d.src = k::DecRowSrc{r.kv_d, nullptr, nullptr, nullptr, nullptr, nullptr, r.pos_d, nullptr, r.vis_d, r.trows_d, nullptr};
  d.ring_ptrs = a.ringp_d;
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::max(1, atoi(argv[1])) : 20;
  const int NL = argc > 2 ? std::max(1, atoi(argv[2])) : 4;
  const std::vector<int> ctxs = parse_list(argc > 3 ? argv[3] : nullptr, {4096, 32768, 131072});
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int rt = 0, drv = 0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
  printf("device %s · SMs %d · L2 %d MB · runtime %d · driver %d · reps %d · layer copies %d · (the test calls both versions directly, independent of the switch)\n", prop.name,
         prop.multiProcessorCount, prop.l2CacheSize >> 20, rt, drv, reps, NL);
  const size_t flush_n = (size_t)512 << 20;
  uint8_t* flush = dmalloc<uint8_t>(flush_n);
  cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const int max_ctx = *std::max_element(ctxs.begin(), ctxs.end());
  const int npos = max_ctx + 16;
  float2* freqs = dmalloc<float2>((size_t)npos * (RD / 2));
  { const auto fr = rope_host(npos, 10000.f); CUDA_CHECK(cudaMemcpy(freqs, fr.data(), fr.size() * sizeof(float2), cudaMemcpyHostToDevice)); }
  bool all_ok = true;
  const int Ms[] = {1, 4, 8};
  LayerW L0 = make_layer(20260930u);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ================= router =================
  {
    printf("\n== router (gate_w f32 [384,5120] = 7.9 MB · top 6) — µs L2 flushed · GB/s = gate bytes/time ==\n");
    float* gate_t = dmalloc<float>((size_t)E * DIM); fill_f32(gate_t, (size_t)E * DIM, 777u, 0.f, 0.03f, 4);  // every 4 rows share a gate -> tied scores
    float* zero = dmalloc<float>(E);
    Rows r; make_rows(r, 4096, 1, 8, 1);
    Acts A = make_acts(r), B = make_acts(r);
    const auto idx = make_idx(r, 8, 5);
    const double mb = (double)E * DIM * 4 / 1e6;
    printf("%-11s %2s %-24s | %8s %8s %8s | %8s %8s %8s %8s %8s | %s\n", "case", "M", "bit(new vs A1 · fused)", "A1 µs", "fused µs", "new µs", "ld(xs)", "ld+dot", "ld(reg)",
           "ld+dot", "xs/reg", "GB/s A1→new");
    for (int tie = 0; tie <= 1; ++tie)
      for (const int M : Ms) {
        const float* g = tie ? gate_t : L0.gate;
        const float* b0 = tie ? zero : L0.bias;
        const float* b1 = tie ? zero : L0.bias_vl;
        seed_acts(A, r, idx, 99u + M, 0xA5); seed_acts(B, r, idx, 99u + M, 0x5A);
        auto r_a1 = [&](Acts& a) { k::attn2_router(a.xn, DIM, g, M, E, b0, b1, a.is_image, TOPK, ROUTE_SCALE, a.scores, a.ids, a.rw, a.counters + 2, st); };
        auto r_fu = [&](Acts& a) { k::fused_router(a.xn, DIM, g, M, E, b0, b1, a.is_image, TOPK, ROUTE_SCALE, a.scores, a.ids, a.rw, a.counters + 2, st); };
        auto r_new = [&](Acts& a, int phase, int mode) {
          k::attn3_router_phase(a.xn, DIM, g, M, E, b0, b1, a.is_image, TOPK, ROUTE_SCALE, a.scores, a.ids, a.rw, a.counters + 2, phase, mode, st);
        };
        std::string detail;
        bool ok = true;
        auto cmp = [&](const char* tag) {
          const std::pair<const char*, std::pair<size_t, std::pair<const void*, const void*>>> outs[] = {
              {"scores", {(size_t)M * E * 4, {A.scores, B.scores}}}, {"ids", {(size_t)M * TOPK * 4, {A.ids, B.ids}}}, {"w", {(size_t)M * TOPK * 4, {A.rw, B.rw}}}};
          for (const auto& o : outs) {
            const size_t d = diff_bytes(o.second.second.first, o.second.second.second, o.second.first);
            if (d) { ok = false; detail += std::string(" ") + tag + o.first + "≠" + std::to_string(d); }
          }
        };
        run_sync("router", "A1(attn2)", M, st, [&] { r_a1(A); });
        for (int mode : {-1, 0, 1}) {
          if (mode == 1 && (size_t)DIM * 4 + (size_t)M * DIM * 2 > 98 * 1024) continue;
          const char* vn = mode < 0 ? "new(auto)" : mode ? "new(xs)" : "new(reg)";
          run_sync("router", vn, M, st, [&] { r_new(B, 0, mode); });
          cmp(mode < 0 ? "" : mode ? "xs:" : "reg:");
          run_sync("router", vn, M, st, [&] { r_new(B, 0, mode); });  // rerun (counters back to 0)
          cmp("rerun:");
        }
        run_sync("router", "fused", M, st, [&] { r_fu(A); });
        cmp("vsFused:");
        const double t_a1 = time_us([&] { r_a1(A); }, reps, st, flush, flush_n);
        const double t_fu = time_us([&] { r_fu(A); }, reps, st, flush, flush_n);
        const double t_new = time_us([&] { r_new(B, 0, -1); }, reps, st, flush, flush_n);
        const bool xs_fit = (size_t)DIM * 4 + (size_t)M * DIM * 2 <= 98 * 1024;
        const double p1x = xs_fit ? time_us([&] { r_new(B, 1, 1); }, reps, st, flush, flush_n) : -1;
        const double p2x = xs_fit ? time_us([&] { r_new(B, 2, 1); }, reps, st, flush, flush_n) : -1;
        const double p1r = time_us([&] { r_new(B, 1, 0); }, reps, st, flush, flush_n);
        const double p2r = time_us([&] { r_new(B, 2, 0); }, reps, st, flush, flush_n);
        const double fx = xs_fit ? time_us([&] { r_new(B, 0, 1); }, reps, st, flush, flush_n) : -1;
        const double fr = time_us([&] { r_new(B, 0, 0); }, reps, st, flush, flush_n);
        char xsreg[32]; snprintf(xsreg, sizeof xsreg, "%.1f/%.1f", fx, fr);
        printf("%-11s %2d %-24s | %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f %8.1f %8s | %.0f→%.0f (%.2fx)\n", tie ? "router(tie)" : "router", M,
               ok ? "identical" : ("DIFF" + detail).c_str(), t_a1, t_fu, t_new, p1x, p2x, p1r, p2r, xsreg, mb * 1e3 / t_a1, mb * 1e3 / t_new, t_a1 / t_new);
        fflush(stdout);
        all_ok = all_ok && ok;
      }
    printf("   (ld = gate row loads only (+xn for the xs version) · ld+dot = through the dot product (no tail) · tail = total − ld+dot · the xs version uses M·10KB smem, so a single wave breaks at M ≥ 2)\n");
    free_acts(A); free_acts(B); free_rows(r); cudaFree(gate_t); cudaFree(zero);
  }

  // ================= hc_mix_pre_norm =================
  {
    printf("\n== hc_mix_pre_norm (hc_fn f32 [24, 20480] = 1.97 MB + h) ==\n");
    Rows r; make_rows(r, 4096, 1, 8, 2);
    Acts A = make_acts(r), B = make_acts(r);
    const auto idx = make_idx(r, 8, 6);
    printf("%2s %-34s | %9s %9s %7s\n", "M", "bit(mixes·rsq·x·xn)", "old µs", "new µs", "x");
    for (const int M : Ms) {
      seed_acts(A, r, idx, 7u + M, 0xA5); seed_acts(B, r, idx, 7u + M, 0x5A);
      auto h_old = [&](Acts& a) { k::hc_mix_pre_norm(a.h, L0.hc_attn, M, HC * DIM, MIX, HC_EPS, a.mixes, a.rsq, a.pre, HC, DIM, L0.attn_norm, EPS, a.x, a.xn, st); };
      auto h_new = [&](Acts& a) { k::hc_mix_pre_norm3(a.h, L0.hc_attn, M, HC * DIM, MIX, HC_EPS, a.mixes, a.rsq, a.pre, HC, DIM, L0.attn_norm, EPS, a.x, a.xn, st); };
      run_sync("hc_mix", "old", M, st, [&] { h_old(A); });
      run_sync("hc_mix", "new", M, st, [&] { h_new(B); });
      std::string detail;
      bool ok = true;
      const std::pair<const char*, std::pair<size_t, std::pair<const void*, const void*>>> outs[] = {
          {"mixes", {(size_t)M * MIX * 4, {A.mixes, B.mixes}}}, {"rsq", {(size_t)M * 4, {A.rsq, B.rsq}}}, {"x", {(size_t)M * DIM * 2, {A.x, B.x}}},
          {"xn", {(size_t)M * DIM * 2, {A.xn, B.xn}}}};
      for (const auto& o : outs) {
        const size_t d = diff_bytes(o.second.second.first, o.second.second.second, o.second.first);
        if (d) { ok = false; detail += std::string(" ") + o.first + "≠" + std::to_string(d); }
      }
      const double t_old = time_us([&] { h_old(A); }, reps, st, flush, flush_n);
      const double t_new = time_us([&] { h_new(B); }, reps, st, flush, flush_n);
      printf("%2d %-34s | %9.1f %9.1f %6.2fx\n", M, ok ? "identical" : ("DIFF" + detail).c_str(), t_old, t_new, t_old / t_new);
      fflush(stdout);
      all_ok = all_ok && ok;
    }
    free_acts(A); free_acts(B); free_rows(r);
  }

  // ================= q_a ‖ kv =================
  {
    printf("\n== q_a‖kv (wq_a [1280,5120] + wkv [512,5120] e4m3 = 9.5 MB · tail: q rmsnorm · kv rmsnorm→RoPE→fp8 round trip→ring · row table · window idx) ==\n");
    printf("%2s %-4s %-44s | %9s %9s %7s\n", "M", "xf", "bit(qr·qrn·kv·xf·ring·idx·tab)", "old µs", "new µs", "x");
    Rows r; make_rows(r, 32768, 2, 8, 3);
    Acts A = make_acts(r), B = make_acts(r);
    const auto idx = make_idx(r, 8, 7);
    for (const int M : Ms)
      for (const int xf_on : {0, 1}) {
        seed_acts(A, r, idx, 11u + M, 0xA5); seed_acts(B, r, idx, 11u + M, 0x5A);
        k::DecAttnArgs da = front_args(L0, A, r, M, xf_on ? 2 : 1, true, freqs), db = front_args(L0, B, r, M, xf_on ? 2 : 1, true, freqs);
        bool took = true;
        run_sync("qkv_a", "old", M, st, [&] { k::dec_front_stage(da, 0, -1, st); });
        run_sync("qkv_a", "new", M, st, [&] { took = k::dec_front_stage(db, 0, 0, st); });
        std::string detail = took ? "" : " (new refused shape)";
        bool ok = took;
        const std::pair<const char*, std::pair<size_t, std::pair<const void*, const void*>>> outs[] = {
            {"qr", {(size_t)M * QL * 2, {A.qr, B.qr}}}, {"qrn", {(size_t)M * QL * 2, {A.qrn, B.qrn}}}, {"kv", {(size_t)M * HD * 2, {A.kv, B.kv}}},
            {"xf", {xf_on ? (size_t)M * DIM * 4 : 0, {A.xf, B.xf}}}};
        for (const auto& o : outs) {
          const size_t d = o.second.first ? diff_bytes(o.second.second.first, o.second.second.second, o.second.first) : 0;
          if (d) { ok = false; detail += std::string(" ") + o.first + "≠" + std::to_string(d); }
        }
        size_t dr = 0, di = 0, dt = 0;
        for (int m = 0; m < M; ++m) {
          dr += diff_bytes(A.ring[m] + (size_t)(r.pos[m] % WIN) * HD, B.ring[m] + (size_t)(r.pos[m] % WIN) * HD, HD * 2);
          dr += diff_bytes(A.ring[m], B.ring[m], (size_t)WIN * HD * 2);  // other slots must be untouched
          di += diff_bytes(A.idx + (size_t)m * (WIN + ITOPK), B.idx + (size_t)m * (WIN + ITOPK), (WIN + ITOPK) * 4);
          auto ta = host(A.tab + m, sizeof(k::DecRow)), tb = host(B.tab + m, sizeof(k::DecRow));
          k::DecRow ra, rb; memcpy(&ra, ta.data(), sizeof ra); memcpy(&rb, tb.data(), sizeof rb);
          ra.kv.ring = rb.kv.ring = nullptr;  // ring pointers are different buffers per variant (expected)
          dt += memcmp(&ra, &rb, sizeof ra) != 0;
        }
        {
          auto sa = host(A.soa, sizeof(k::DecSoA)), sb = host(B.soa, sizeof(k::DecSoA));
          k::DecSoA qa, qb; memcpy(&qa, sa.data(), sizeof qa); memcpy(&qb, sb.data(), sizeof qb);
          for (int m = 0; m < M; ++m) dt += (qa.trows[m] != qb.trows[m]) + (qa.visible[m] != qb.visible[m]) + (qa.kptr[m] != qb.kptr[m]);
        }
        const auto ca = host(A.counters, 8), cb = host(B.counters, 8);
        size_t dc = 0; for (int i = 0; i < 8; ++i) dc += ca[i] != 0 || cb[i] != 0;
        if (dr) { ok = false; detail += " ring≠" + std::to_string(dr); }
        if (di) { ok = false; detail += " idx≠" + std::to_string(di); }
        if (dt) { ok = false; detail += " tab≠" + std::to_string(dt); }
        if (dc) { ok = false; detail += " counters≠0"; }
        const double t_old = time_us([&] { k::dec_front_stage(da, 0, -1, st); }, reps, st, flush, flush_n);
        const double t_new = time_us([&] { k::dec_front_stage(db, 0, 0, st); }, reps, st, flush, flush_n);
        printf("%2d %-4s %-44s | %9.1f %9.1f %6.2fx\n", M, xf_on ? "yes" : "-", ok ? "identical" : ("DIFF" + detail).c_str(), t_old, t_new, t_old / t_new);
        fflush(stdout);
        all_ok = all_ok && ok;
      }
    free_acts(A); free_acts(B); free_rows(r);
  }

  // ================= sparse attention =================
  {
    printf("\n== sparse attention (dec_attn_kernel vs attn3_sparse · columns = window 128 + compressed top 512 · splits S) — µs L2 flushed ==\n");
    printf("%-4s %7s %2s %2s | %-16s | %8s | %8s %8s %8s %8s %8s | %s\n", "L", "ctx", "M", "S", "bit(o) all G", "old µs", "auto", "G=1", "G=2", "G=4", "G=8", "auto G · best");
    struct LC { const char* name; int ratio; };
    const LC lcs[] = {{"W", 0}, {"C1", 1}, {"C2", 2}};
    for (const int ctx : ctxs)
      for (const LC& lc : lcs) {
        if (lc.ratio == 0 && ctx != ctxs[0]) continue;  // window layers do not depend on context
        Rows r; make_rows(r, ctx, lc.ratio, 8, 10u + ctx / 1024 + lc.ratio);
        Acts A = make_acts(r), B = make_acts(r);
        for (const int M : Ms) {
          const auto idx = make_idx(r, M, 1234u + M);
          seed_acts(A, r, idx, 21u + M, 0xA5); seed_acts(B, r, idx, 21u + M, 0x5A);
          k::DecAttnArgs da = front_args(L0, A, r, M, lc.ratio, false, freqs), db = front_args(L0, B, r, M, lc.ratio, false, freqs);
          // fill the row table (normally done by q_a‖kv — here once with the old stage) · ring kv (chunk rows) keeps the seed values
          run_sync("stage_rows", "-", M, st, [&] { k::dec_stage_rows(da.src, M, A.tab, A.soa, st); k::dec_stage_rows(db.src, M, B.tab, B.soa, st); });
          // make both variants' inputs (q·kv·chunk·ring) equal: copy B's q·kv from A
          CUDA_CHECK(cudaMemcpy(B.q, A.q, (size_t)M * H * HD * 2, cudaMemcpyDeviceToDevice));
          CUDA_CHECK(cudaMemcpy(B.kv, A.kv, (size_t)M * HD * 2, cudaMemcpyDeviceToDevice));
          run_sync("attn", "old", M, st, [&] { k::dec_front_stage(da, 1, -1, st); });
          const int S = da.splits;
          std::string detail;
          bool ok = true;
          double tg[5] = {0, 0, 0, 0, 0};
          const int Gs[5] = {0, 1, 2, 4, 8};
          for (int gi = 0; gi < 5; ++gi) {
            CUDA_CHECK(cudaMemset(B.o, 0x5A, (size_t)8 * H * HD * 2));
            char vn[16]; snprintf(vn, sizeof vn, "new G=%d", Gs[gi]);
            bool took = true;
            run_sync("attn", vn, M, st, [&] { took = k::dec_front_stage(db, 1, Gs[gi], st); });
            if (!took) { ok = false; detail += std::string(" G") + std::to_string(Gs[gi]) + ":refused"; continue; }
            const size_t d = diff_bytes(A.o, B.o, (size_t)M * H * HD * 2);
            const auto cz = host(B.attn_cnt, (size_t)8 * H * 4);
            size_t nz = 0; for (size_t i = 0; i < cz.size(); ++i) nz += cz[i] != 0;
            if (d || nz) { ok = false; detail += std::string(" G") + std::to_string(Gs[gi]) + ":o≠" + std::to_string(d) + (nz ? "+cnt" : ""); }
            tg[gi] = time_us([&] { k::dec_front_stage(db, 1, Gs[gi], st); }, reps, st, flush, flush_n);
          }
          const double t_old = time_us([&] { k::dec_front_stage(da, 1, -1, st); }, reps, st, flush, flush_n);
          int best = 1; for (int gi = 2; gi < 5; ++gi) if (tg[gi] > 0 && tg[gi] < tg[best]) best = gi;
          printf("%-4s %7d %2d %2d | %-16s | %8.1f | %8.1f %8.1f %8.1f %8.1f %8.1f | auto G=%d %.2fx · best G=%d %.2fx\n", lc.name, ctx, M, S,
                 ok ? "identical" : ("DIFF" + detail).c_str(), t_old, tg[0], tg[1], tg[2], tg[3], tg[4], k::attn3_sparse_auto_g(M, H, S), t_old / tg[0], Gs[best],
                 t_old / tg[best]);
          fflush(stdout);
          all_ok = all_ok && ok;
        }
        free_acts(A); free_acts(B); free_rows(r);
      }
    printf("   (auto G = smallest G with M·H·S/G ≤ SM count · pin with HIVE_DECODE_SPARSE3_G=1/2/4/8)\n");
  }

  // ================= layer chain (eager · CUDA graph) =================
  {
    std::vector<LayerW> W;
    for (int l = 0; l < NL; ++l) W.push_back(l == 0 ? L0 : make_layer(1000u + 17u * l));
    const int ctx = ctxs.size() > 1 ? ctxs[1] : ctxs[0];
    printf("\nchain (layer C1 · context %d): hc_mix → q_a‖kv → q_b(A1) → sparse attention → wo_a → wo_b(A1) → hc_mix → router → shared(A1) · %d weight copies (outside L2)\n", ctx, NL);
    printf("   old = service path (hc_mix_pre_norm · dec_qkv_a · dec_attn · attn2_router) · new = the four new versions (HCMIX3 · QKV3 · SPARSE3 (auto G) · ROUTER3)\n");
    Rows r; make_rows(r, ctx, 1, 8, 77);
    Acts R = make_acts(r), N = make_acts(r);
    printf("%2s  %-40s %10s %10s %10s %10s\n", "M", "bit(new vs old chain)", "old µs/L", "new µs/L", "old graph", "new graph");
    for (const int M : Ms) {
      const auto idx = make_idx(r, M, 4321u + M);
      auto chain = [&](int which, Acts& a) {
        for (int l = 0; l < NL; ++l) {
          const LayerW& Lw = W[l];
          k::DecAttnArgs d = front_args(Lw, a, r, M, 1, false, freqs);
          d.v3 = which ? 3 : 0;
          (which ? k::hc_mix_pre_norm3 : k::hc_mix_pre_norm)(a.h, Lw.hc_attn, M, HC * DIM, MIX, HC_EPS, a.mixes, a.rsq, a.pre, HC, DIM, Lw.attn_norm, EPS, a.x, a.xn, st);
          k::dec_front_stage(d, 0, which ? 0 : -1, st);
          k::attn2_qb(a.qrn, QL, Lw.wqb.w, Lw.wqb.s, M, H * HD, HD, RD, freqs,
                      reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(a.tab) + offsetof(k::DecRow, pos)), (int)(sizeof(k::DecRow) / 4), a.q,
                      a.qrq, a.qrs, st);
          k::dec_front_stage(d, 1, which ? 0 : -1, st);
          k::gemv_bf16_fp8_grouped(a.o, H * HD, Lw.woa.w, Lw.woa.s, M, OG, OR, SUB, a.og, st);
          k::attn2_gemv_quantin(a.og, OG * OR, Lw.wob.w, Lw.wob.s, M, DIM, a.attn_out, st);
          (which ? k::hc_mix_pre_norm3 : k::hc_mix_pre_norm)(a.h, Lw.hc_ffn, M, HC * DIM, MIX, HC_EPS, a.mixes, a.rsq, a.pre, HC, DIM, Lw.ffn_norm, EPS, a.x, a.xn, st);
          (which ? k::attn3_router : k::attn2_router)(a.xn, DIM, Lw.gate, M, E, Lw.bias, Lw.bias_vl, a.is_image, TOPK, ROUTE_SCALE, a.scores, a.ids, a.rw,
                                                      a.counters + 2, st);
          k::attn2_shared_experts(a.xn, DIM, Lw.w1.w, Lw.w1.s, Lw.w3.w, Lw.w3.s, Lw.w2.w, Lw.w2.s, M, INTER, LIMIT, a.y, a.xq, a.xs, a.acc, st);
        }
      };
      auto compare = [&](std::string& detail) {
        bool ok = true;
        const std::pair<const char*, std::pair<size_t, std::pair<const void*, const void*>>> outs[] = {
            {"qr", {(size_t)M * QL * 2, {R.qr, N.qr}}}, {"qrn", {(size_t)M * QL * 2, {R.qrn, N.qrn}}}, {"kv", {(size_t)M * HD * 2, {R.kv, N.kv}}},
            {"q", {(size_t)M * H * HD * 2, {R.q, N.q}}}, {"o", {(size_t)M * H * HD * 2, {R.o, N.o}}}, {"og", {(size_t)M * OG * OR * 2, {R.og, N.og}}},
            {"attn_out", {(size_t)M * DIM * 2, {R.attn_out, N.attn_out}}}, {"x", {(size_t)M * DIM * 2, {R.x, N.x}}}, {"xn", {(size_t)M * DIM * 2, {R.xn, N.xn}}},
            {"mixes", {(size_t)M * MIX * 4, {R.mixes, N.mixes}}}, {"ids", {(size_t)M * TOPK * 4, {R.ids, N.ids}}}, {"w", {(size_t)M * TOPK * 4, {R.rw, N.rw}}},
            {"acc", {(size_t)M * DIM * 4, {R.acc, N.acc}}}, {"y", {(size_t)M * INTER * 2, {R.y, N.y}}}};
        for (const auto& o : outs) {
          const size_t d = diff_bytes(o.second.second.first, o.second.second.second, o.second.first);
          if (d) { ok = false; detail += std::string(" ") + o.first + "≠" + std::to_string(d); }
        }
        size_t dr = 0;
        for (int m = 0; m < M; ++m) dr += diff_bytes(R.ring[m], N.ring[m], (size_t)WIN * HD * 2);
        if (dr) { ok = false; detail += " ring≠" + std::to_string(dr); }
        return ok;
      };
      seed_acts(R, r, idx, 5u + M, 0xA5); seed_acts(N, r, idx, 5u + M, 0x5A);
      run_sync("chain", "old", M, st, [&] { chain(0, R); });
      run_sync("chain", "new", M, st, [&] { chain(1, N); });
      std::string detail;
      bool ok = compare(detail);
      const double t_old = time_us([&] { chain(0, R); }, reps, st, nullptr, 0) / NL;
      const double t_new = time_us([&] { chain(1, N); }, reps, st, nullptr, 0) / NL;
      seed_acts(R, r, idx, 5u + M, 0xA5); seed_acts(N, r, idx, 5u + M, 0x5A);
      printf("[run] chain graph old/new M=%d\n", M); fflush(stdout);
      const double g_old = graph_us([&] { chain(0, R); }, reps, st, NL);
      const double g_new = graph_us([&] { chain(1, N); }, reps, st, NL);
      if (g_old < 0 || g_new < 0) { ok = false; detail += " graph-capture-failed"; }
      std::string d2;
      if (!compare(d2)) { ok = false; detail += " (graph)" + d2; }
      printf("%2d  %-40s %10.1f %10.1f %10.1f %10.1f   eager %.2fx · graph %.2fx\n", M, ok ? "identical" : ("DIFF" + detail).c_str(), t_old, t_new, g_old, g_new,
             t_old / t_new, g_new > 0 ? g_old / g_new : 0.0);
      fflush(stdout);
      all_ok = all_ok && ok;
    }
    free_acts(R); free_acts(N); free_rows(r);
  }
  cudaFree(flush);
  cudaStreamDestroy(st);
  printf(all_ok ? "\nALL PASS\n" : "\nSOME FAIL\n");
  return all_ok ? 0 : 1;
}
