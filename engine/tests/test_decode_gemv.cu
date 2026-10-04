// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_GEMV2 / HIVE_DECODE_PDL validation: new decode dense GEMV, router and hc mix kernels (gemv_decode.cu) vs the previous functions.
//   Per kernel (real layer shapes, M = 1, 4, 8):
//     q_proj  fused_q_proj           vs gemv2_q_proj           — qr·qrn·q·qrq·qrs   (wq_a [1280,5120] + wq_b [32768,1280] e4m3)
//     wo_a    gemv_bf16_fp8_grouped  vs gemv2_bf16_fp8_grouped — og                 (8 groups × [1024,4096] e4m3, bf16 activations)
//     wo_b    fused_gemv_quantin     vs gemv2_quantin          — attn_out           ([5120,8192] e4m3)
//     shared  fused_shared_experts   vs gemv2_shared_experts   — y·xq·xs·acc        (w1‖w3 [2304,5120] · w2 [5120,2304] e4m3)
//     router  fused_router           vs gemv2_router           — scores·ids·w       (gate_w f32 [384,5120] · top-6 · rows mixed with the image bias)
//     router(tie) gate_w with each row replicated 4× + zero bias → many score ties (tie rule: larger value → smaller index)
//     hc_mix  hc_mix_pre_norm        vs hc_mix_pre_norm2       — mixes·rsq·x·xn     (hc_fn f32 [24,20480])
//   Each kernel: the new version runs twice, PDL off and on, both **byte-compared** with the previous one (bit-identical by design — no tolerance) · µs after flushing L2 (128MB) · effective GB/s (weight bytes / time).
//   The chain (decode order of one layer: hc_mix → q_proj → wo_a → wo_b → hc_mix → router → shared) × NL weight copies (outside L2) runs
//     previous · new (PDL off) · new (PDL on) · new (PDL on, captured CUDA graph replay) with output bit comparison + µs per layer.
//     The graph variant also checks empirically that PDL edges are captured and instantiated (CUDA 12.3+) — failure is reported as FAIL (serving uses graph replay).
//   Verdict: all bits match = PASS. Run: scripts/hive-run.sh "./build-dev/test_decode_gemv [reps=20] [NL=4]"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "hive/common.h"
#include "hive/gemv_decode.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

constexpr int DIM = 5120, H = 64, HD = 512, RD = 64, QL = 1280, OG = 8, OR = 1024, INTER = 2304, E = 384, TOPK = 6, HC = 4, MIX = (2 + HC) * HC;
constexpr int SUB = H * HD / OG;
constexpr float LIMIT = 10.f, EPS = 1e-6f, HC_EPS = 1e-6f, ROUTE_SCALE = 2.5f;
constexpr int MAXPOS = 4096;

__device__ __forceinline__ uint32_t hsh(size_t i, uint32_t seed) {
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed ^ (uint32_t)(i >> 32) * 0x9E3779B9u;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  return h;
}
__global__ void fill_u8_kernel(uint8_t* p, size_t n, uint32_t seed, int mode) {
  const size_t i = hive::cu::global_tid();
  if (i >= n) return;
  const uint32_t h = hsh(i, seed);
  uint8_t v = (uint8_t)h;
  if (mode == 1) v = (uint8_t)((v & 0x80) | ((v & 0x7F) % 0x7E));  // e4m3 weights (excluding NaN 0x7F/0xFF, including ±448)
  else if (mode == 3) v = (uint8_t)(118 + (h >> 8) % 6);            // weight scale e8m0
  p[i] = v;
}
__global__ void fill_bf16_kernel(bf16* p, size_t n, uint32_t seed, float a, float b) {  // a + b·u, u ∈ [-1,1)
  const size_t i = hive::cu::global_tid();
  if (i >= n) return;
  const float u = (float)(hsh(i, seed) >> 8) * (2.f / 16777216.f) - 1.f;
  p[i] = f2bf(a + b * u);
}
__global__ void fill_f32_kernel(float* p, size_t n, uint32_t seed, float a, float b, int dup) {  // dup > 1: same values for every dup rows (row length DIM)
  const size_t i = hive::cu::global_tid();
  if (i >= n) return;
  size_t j = i;
  if (dup > 1) { const size_t row = i / DIM, col = i % DIM; j = (row / dup) * DIM + col; }
  const float u = (float)(hsh(j, seed) >> 8) * (2.f / 16777216.f) - 1.f;
  p[i] = a + b * u;
}
unsigned nblk(size_t n) { return (unsigned)((n + 255) / 256); }
void fill_u8(void* p, size_t n, uint32_t seed, int mode) { fill_u8_kernel<<<nblk(n), 256>>>((uint8_t*)p, n, seed, mode); CUDA_CHECK(cudaGetLastError()); }
void fill_bf16(bf16* p, size_t n, uint32_t seed, float a, float b) { fill_bf16_kernel<<<nblk(n), 256>>>(p, n, seed, a, b); CUDA_CHECK(cudaGetLastError()); }
void fill_f32(float* p, size_t n, uint32_t seed, float a, float b, int dup = 1) { fill_f32_kernel<<<nblk(n), 256>>>(p, n, seed, a, b, dup); CUDA_CHECK(cudaGetLastError()); }
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T))); return p; }
std::vector<uint8_t> host(const void* d, size_t bytes) { std::vector<uint8_t> v(bytes); CUDA_CHECK(cudaMemcpy(v.data(), d, bytes, cudaMemcpyDeviceToHost)); return v; }

// one layer's weights (real storage format: e4m3 + [N, K/32] e8m0 · router/hc f32)
struct Weights {
  uint8_t *wqa, *sqa, *wqb, *sqb, *woa, *soa, *wob, *sob, *w1, *s1, *w3, *s3, *w2, *s2;
  float *gate, *bias, *bias_vl, *hc_fn;
  bf16 *q_norm, *attn_norm;
  size_t dense_bytes() const {
    auto fp8 = [](size_t n, size_t k) { return n * k + n * (k / 32); };
    return fp8(QL, DIM) + fp8((size_t)H * HD, QL) + fp8((size_t)OG * OR, SUB) + fp8(DIM, (size_t)OG * OR) + 2 * fp8(INTER, DIM) + fp8(DIM, INTER) +
           (size_t)E * DIM * 4 + 2 * (size_t)MIX * HC * DIM * 4;
  }
};
Weights make_weights(uint32_t seed, int gate_dup) {
  Weights w;
  auto fp8 = [&](uint8_t*& q, uint8_t*& s, size_t n, size_t k, uint32_t sd) {
    q = dmalloc<uint8_t>(n * k); s = dmalloc<uint8_t>(n * (k / 32));
    fill_u8(q, n * k, seed ^ sd, 1); fill_u8(s, n * (k / 32), seed ^ (sd * 7919u), 3);
  };
  fp8(w.wqa, w.sqa, QL, DIM, 11); fp8(w.wqb, w.sqb, (size_t)H * HD, QL, 12); fp8(w.woa, w.soa, (size_t)OG * OR, SUB, 13);
  fp8(w.wob, w.sob, DIM, (size_t)OG * OR, 14); fp8(w.w1, w.s1, INTER, DIM, 15); fp8(w.w3, w.s3, INTER, DIM, 16); fp8(w.w2, w.s2, DIM, INTER, 17);
  w.gate = dmalloc<float>((size_t)E * DIM); fill_f32(w.gate, (size_t)E * DIM, seed ^ 21, 0.f, 0.03f, gate_dup);
  w.bias = dmalloc<float>(E); w.bias_vl = dmalloc<float>(E);
  if (gate_dup > 1) { CUDA_CHECK(cudaMemset(w.bias, 0, E * 4)); CUDA_CHECK(cudaMemset(w.bias_vl, 0, E * 4)); }
  else { fill_f32(w.bias, E, seed ^ 22, 0.f, 0.05f); fill_f32(w.bias_vl, E, seed ^ 23, 0.f, 0.05f); }
  w.hc_fn = dmalloc<float>((size_t)MIX * HC * DIM); fill_f32(w.hc_fn, (size_t)MIX * HC * DIM, seed ^ 24, 0.f, 0.01f);
  w.q_norm = dmalloc<bf16>(QL); fill_bf16(w.q_norm, QL, seed ^ 25, 1.f, 0.1f);
  w.attn_norm = dmalloc<bf16>(DIM); fill_bf16(w.attn_norm, DIM, seed ^ 26, 1.f, 0.1f);
  return w;
}
void free_weights(Weights& w) {
  for (void* p : {(void*)w.wqa, (void*)w.sqa, (void*)w.wqb, (void*)w.sqb, (void*)w.woa, (void*)w.soa, (void*)w.wob, (void*)w.sob, (void*)w.w1, (void*)w.s1,
                  (void*)w.w3, (void*)w.s3, (void*)w.w2, (void*)w.s2, (void*)w.gate, (void*)w.bias, (void*)w.bias_vl, (void*)w.hc_fn, (void*)w.q_norm,
                  (void*)w.attn_norm})
    cudaFree(p);
}

// one set of activation and output buffers (one each for previous / new — for output comparison)
struct Acts {
  bf16 *h, *xn, *x, *o, *og_in, *qr, *qrn, *q, *og, *attn_out, *y;
  uint8_t *qrq, *qrs, *xq, *xs;
  float *pre, *mixes, *rsq, *scores, *rw, *acc;
  int32_t *ids, *pos;
  int8_t* is_image;
  int* counters;
  float2* freqs;
};
Acts make_acts() {
  Acts a;
  a.h = dmalloc<bf16>((size_t)8 * HC * DIM); a.xn = dmalloc<bf16>((size_t)8 * DIM); a.x = dmalloc<bf16>((size_t)8 * DIM);
  a.o = dmalloc<bf16>((size_t)8 * H * HD); a.og_in = dmalloc<bf16>((size_t)8 * OG * OR);
  a.qr = dmalloc<bf16>((size_t)8 * QL); a.qrn = dmalloc<bf16>((size_t)8 * QL); a.q = dmalloc<bf16>((size_t)8 * H * HD);
  a.og = dmalloc<bf16>((size_t)8 * OG * OR); a.attn_out = dmalloc<bf16>((size_t)8 * DIM); a.y = dmalloc<bf16>((size_t)8 * INTER);
  a.qrq = dmalloc<uint8_t>((size_t)8 * QL); a.qrs = dmalloc<uint8_t>((size_t)8 * QL / 32); a.xq = dmalloc<uint8_t>((size_t)8 * DIM);
  a.xs = dmalloc<uint8_t>((size_t)8 * DIM / 32);
  a.pre = dmalloc<float>(8 * HC); a.mixes = dmalloc<float>(8 * MIX); a.rsq = dmalloc<float>(8); a.scores = dmalloc<float>((size_t)8 * E);
  a.rw = dmalloc<float>(8 * TOPK); a.acc = dmalloc<float>((size_t)8 * DIM); a.ids = dmalloc<int32_t>(8 * TOPK); a.pos = dmalloc<int32_t>(8);
  a.is_image = dmalloc<int8_t>(8); a.counters = dmalloc<int>(16); a.freqs = dmalloc<float2>((size_t)MAXPOS * (RD / 2));
  CUDA_CHECK(cudaMemset(a.counters, 0, 16 * 4));
  return a;
}
void seed_inputs(const Acts& a, uint32_t seed) {  // same seed = same input (identical in the previous and new buffers)
  fill_bf16(a.h, (size_t)8 * HC * DIM, seed ^ 1, 0.f, 1.f); fill_bf16(a.xn, (size_t)8 * DIM, seed ^ 2, 0.f, 1.5f);
  fill_bf16(a.o, (size_t)8 * H * HD, seed ^ 3, 0.f, 0.8f); fill_bf16(a.og_in, (size_t)8 * OG * OR, seed ^ 4, 0.f, 0.8f);
  fill_f32(a.pre, 8 * HC, seed ^ 5, 0.25f, 0.1f);
  std::vector<float2> fr((size_t)MAXPOS * (RD / 2));
  for (int p = 0; p < MAXPOS; ++p)
    for (int i = 0; i < RD / 2; ++i) { const float ang = (float)p * powf(10000.f, -2.f * i / RD); fr[(size_t)p * (RD / 2) + i] = make_float2(cosf(ang), sinf(ang)); }
  CUDA_CHECK(cudaMemcpy(a.freqs, fr.data(), fr.size() * sizeof(float2), cudaMemcpyHostToDevice));
  const int32_t pos[8] = {100, 777, 1234, 5, 4000, 2048, 31, 999};
  CUDA_CHECK(cudaMemcpy(a.pos, pos, sizeof(pos), cudaMemcpyHostToDevice));
  const int8_t img[8] = {0, 1, 0, 0, 1, 0, 1, 0};
  CUDA_CHECK(cudaMemcpy(a.is_image, img, sizeof(img), cudaMemcpyHostToDevice));
}
void free_acts(Acts& a) {
  for (void* p : {(void*)a.h, (void*)a.xn, (void*)a.x, (void*)a.o, (void*)a.og_in, (void*)a.qr, (void*)a.qrn, (void*)a.q, (void*)a.og, (void*)a.attn_out,
                  (void*)a.y, (void*)a.qrq, (void*)a.qrs, (void*)a.xq, (void*)a.xs, (void*)a.pre, (void*)a.mixes, (void*)a.rsq, (void*)a.scores,
                  (void*)a.rw, (void*)a.acc, (void*)a.ids, (void*)a.pos, (void*)a.is_image, (void*)a.counters, (void*)a.freqs})
    cudaFree(p);
}

// kernel set — which: 0 = previous · 1 = new. Argument order and buffers match the runtime call sites.
void k_qproj(int which, const Weights& w, const Acts& a, int M, cudaStream_t st) {
  (which ? k::gemv2_q_proj : k::fused_q_proj)(a.xn, DIM, w.wqa, w.sqa, QL, w.q_norm, EPS, w.wqb, w.sqb, H, HD, RD, a.freqs, a.pos, M, a.qr, a.qrn, a.q, a.qrq,
                                               a.qrs, a.counters + 0, st);
}
void k_woa(int which, const Weights& w, const Acts& a, int M, cudaStream_t st) {
  (which ? k::gemv2_bf16_fp8_grouped : k::gemv_bf16_fp8_grouped)(a.o, H * HD, w.woa, w.soa, M, OG, OR, SUB, a.og, st);
}
void k_wob(int which, const Weights& w, const Acts& a, int M, cudaStream_t st, const bf16* in) {
  (which ? k::gemv2_quantin : k::fused_gemv_quantin)(in, OG * OR, w.wob, w.sob, M, DIM, a.attn_out, st);
}
void k_shared(int which, const Weights& w, const Acts& a, int M, cudaStream_t st) {
  (which ? k::gemv2_shared_experts : k::fused_shared_experts)(a.xn, DIM, w.w1, w.s1, w.w3, w.s3, w.w2, w.s2, M, INTER, LIMIT, a.y, a.xq, a.xs, a.acc, st);
}
void k_router(int which, const Weights& w, const Acts& a, int M, cudaStream_t st) {
  (which ? k::gemv2_router : k::fused_router)(a.xn, DIM, w.gate, M, E, w.bias, w.bias_vl, a.is_image, TOPK, ROUTE_SCALE, a.scores, a.ids, a.rw, a.counters + 2, st);
}
void k_hc(int which, const Weights& w, const Acts& a, int M, cudaStream_t st) {
  (which ? k::hc_mix_pre_norm2 : k::hc_mix_pre_norm)(a.h, w.hc_fn, M, HC * DIM, MIX, HC_EPS, a.mixes, a.rsq, a.pre, HC, DIM, w.attn_norm, EPS, a.x, a.xn, st);
}

struct Out { const char* name; size_t off, bytes; };
// comparison: byte comparison of the named output buffers (mismatch count)
size_t diff_bytes(const void* p, const void* q, size_t n) {
  const auto a = host(p, n), b = host(q, n);
  size_t d = 0;
  for (size_t i = 0; i < n; ++i) d += a[i] != b[i];
  return d;
}

double time_us(const std::function<void()>& f, int reps, cudaStream_t st, uint8_t* flush, size_t flush_n) {
  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  double t = 0;
  for (int r = 0; r <= reps; ++r) {
    if (flush) CUDA_CHECK(cudaMemsetAsync(flush, r & 0xFF, flush_n, st));  // flush L2 (so weights come from VRAM)
    CUDA_CHECK(cudaEventRecord(e0, st)); f(); CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (r) t += ms;  // first run = warm-up
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return t * 1000.0 / reps;
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::max(1, atoi(argv[1])) : 20;
  const int NL = argc > 2 ? std::max(1, atoi(argv[2])) : 4;
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int rt = 0, drv = 0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
  printf("device %s · SMs %d · L2 %d MB · runtime %d · driver %d · reps %d · layer copies %d\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, rt,
         drv, reps, NL);
  const size_t flush_n = (size_t)512 << 20;
  uint8_t* flush = dmalloc<uint8_t>(flush_n);
  cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  bool all_ok = true;

  // ---------------- per kernel ----------------
  {
    Weights w = make_weights(20260930u, 1), wt = make_weights(777u, 4);  // wt: router tie test (gate_w rows equal in groups of 4, zero bias)
    Acts A = make_acts(), B = make_acts();
    const auto mb = [](size_t n, size_t k) { return (double)(n * k + n * (k / 32)) / 1e6; };
    struct KCase {
      const char* name; double mbytes;
      std::function<void(int, const Acts&, int)> run;
      std::vector<std::pair<const char*, std::function<std::pair<const void*, size_t>(const Acts&, int)>>> outs;
    };
    auto B2 = [](const void* p, size_t n) { return std::make_pair(p, n); };
    std::vector<KCase> cases;
    cases.push_back({"q_proj", mb(QL, DIM) + mb((size_t)H * HD, QL), [&](int wh, const Acts& a, int M) { k_qproj(wh, w, a, M, st); },
                     {{"qr", [&](const Acts& a, int M) { return B2(a.qr, (size_t)M * QL * 2); }},
                      {"qrn", [&](const Acts& a, int M) { return B2(a.qrn, (size_t)M * QL * 2); }},
                      {"q", [&](const Acts& a, int M) { return B2(a.q, (size_t)M * H * HD * 2); }},
                      {"qrq", [&](const Acts& a, int M) { return B2(a.qrq, (size_t)M * QL); }},
                      {"qrs", [&](const Acts& a, int M) { return B2(a.qrs, (size_t)M * QL / 32); }}}});
    cases.push_back({"wo_a", mb((size_t)OG * OR, SUB), [&](int wh, const Acts& a, int M) { k_woa(wh, w, a, M, st); },
                     {{"og", [&](const Acts& a, int M) { return B2(a.og, (size_t)M * OG * OR * 2); }}}});
    cases.push_back({"wo_b", mb(DIM, (size_t)OG * OR), [&](int wh, const Acts& a, int M) { k_wob(wh, w, a, M, st, a.og_in); },
                     {{"attn_out", [&](const Acts& a, int M) { return B2(a.attn_out, (size_t)M * DIM * 2); }}}});
    cases.push_back({"shared", 2 * mb(INTER, DIM) + mb(DIM, INTER), [&](int wh, const Acts& a, int M) { k_shared(wh, w, a, M, st); },
                     {{"y", [&](const Acts& a, int M) { return B2(a.y, (size_t)M * INTER * 2); }},
                      {"xq", [&](const Acts& a, int M) { return B2(a.xq, (size_t)M * DIM); }},
                      {"xs", [&](const Acts& a, int M) { return B2(a.xs, (size_t)M * DIM / 32); }},
                      {"acc", [&](const Acts& a, int M) { return B2(a.acc, (size_t)M * DIM * 4); }}}});
    auto router_outs = std::vector<std::pair<const char*, std::function<std::pair<const void*, size_t>(const Acts&, int)>>>{
        {"scores", [&](const Acts& a, int M) { return B2(a.scores, (size_t)M * E * 4); }},
        {"ids", [&](const Acts& a, int M) { return B2(a.ids, (size_t)M * TOPK * 4); }},
        {"w", [&](const Acts& a, int M) { return B2(a.rw, (size_t)M * TOPK * 4); }}};
    cases.push_back({"router", (double)E * DIM * 4 / 1e6, [&](int wh, const Acts& a, int M) { k_router(wh, w, a, M, st); }, router_outs});
    cases.push_back({"router(tie)", (double)E * DIM * 4 / 1e6, [&](int wh, const Acts& a, int M) { k_router(wh, wt, a, M, st); }, router_outs});
    cases.push_back({"hc_mix", (double)MIX * HC * DIM * 4 / 1e6, [&](int wh, const Acts& a, int M) { k_hc(wh, w, a, M, st); },
                     {{"mixes", [&](const Acts& a, int M) { return B2(a.mixes, (size_t)M * MIX * 4); }},
                      {"rsq", [&](const Acts& a, int M) { return B2(a.rsq, (size_t)M * 4); }},
                      {"x", [&](const Acts& a, int M) { return B2(a.x, (size_t)M * DIM * 2); }},
                      {"xn", [&](const Acts& a, int M) { return B2(a.xn, (size_t)M * DIM * 2); }}}});
    printf("\n%-12s %2s  %-26s %9s %9s %9s  %8s %8s %8s\n", "kernel", "M", "bit-compare(old vs new)", "old µs", "new µs", "new+PDL", "old GB/s", "new GB/s", "speedup");
    for (const KCase& kc : cases) {
      for (const int M : {1, 4, 8}) {
        bool ok = true;
        std::string detail;
        for (int pdl = 0; pdl <= 1; ++pdl) {
          k::gemv2_force_pdl(pdl);
          seed_inputs(A, 99u + M); seed_inputs(B, 99u + M);
          CUDA_CHECK(cudaDeviceSynchronize());
          kc.run(0, A, M); kc.run(1, B, M);
          CUDA_CHECK(cudaStreamSynchronize(st)); CUDA_CHECK(cudaGetLastError());
          for (const auto& o : kc.outs) {
            const auto pa = o.second(A, M), pb = o.second(B, M);
            const size_t d = diff_bytes(pa.first, pb.first, pa.second);
            if (d) { ok = false; detail += std::string(" ") + o.first + (pdl ? "(pdl)" : "") + "≠" + std::to_string(d) + "B"; }
          }
        }
        k::gemv2_force_pdl(0);
        const double t_old = time_us([&] { kc.run(0, A, M); }, reps, st, flush, flush_n);
        const double t_new = time_us([&] { kc.run(1, B, M); }, reps, st, flush, flush_n);
        k::gemv2_force_pdl(1);
        const double t_pdl = time_us([&] { kc.run(1, B, M); }, reps, st, flush, flush_n);
        k::gemv2_force_pdl(0);
        printf("%-12s %2d  %-26s %9.1f %9.1f %9.1f  %8.0f %8.0f %7.2fx\n", kc.name, M, ok ? "identical" : ("DIFF" + detail).c_str(), t_old, t_new, t_pdl,
               kc.mbytes * 1e3 / t_old, kc.mbytes * 1e3 / t_new, t_old / t_new);
        all_ok = all_ok && ok;
      }
    }
    free_acts(A); free_acts(B); free_weights(w); free_weights(wt);
  }

  // ---------------- chain (layer order · NL weight copies · previous / new / new+PDL / new+PDL graph) ----------------
  {
    std::vector<Weights> W;
    for (int l = 0; l < NL; ++l) W.push_back(make_weights(1000u + 17u * l, 1));
    printf("\nchain: hc_mix → q_proj → wo_a → wo_b → hc_mix → router → shared · %d layer copies × %.1f MB dense\n", NL, W[0].dense_bytes() / 1e6);
    Acts R = make_acts(), N = make_acts();
    auto chain = [&](int which, const Acts& a, int M) {
      for (int l = 0; l < NL; ++l) {
        k_hc(which, W[l], a, M, st);
        k_qproj(which, W[l], a, M, st);
        k_woa(which, W[l], a, M, st);
        k_wob(which, W[l], a, M, st, a.og);  // reads the wo_a output (real RAW dependency)
        k_hc(which, W[l], a, M, st);         // rewrites xn (the buffer the earlier q_proj read — WAR)
        k_router(which, W[l], a, M, st);
        k_shared(which, W[l], a, M, st);
      }
    };
    auto compare = [&](int M, std::string& detail) {
      bool ok = true;
      const std::pair<const char*, std::pair<const void*, const void*>> bufs[] = {
          {"q", {R.q, N.q}}, {"og", {R.og, N.og}}, {"attn_out", {R.attn_out, N.attn_out}}, {"xn", {R.xn, N.xn}}, {"ids", {R.ids, N.ids}},
          {"w", {R.rw, N.rw}}, {"acc", {R.acc, N.acc}}, {"y", {R.y, N.y}}};
      const size_t sz[] = {(size_t)M * H * HD * 2, (size_t)M * OG * OR * 2, (size_t)M * DIM * 2, (size_t)M * DIM * 2, (size_t)M * TOPK * 4,
                           (size_t)M * TOPK * 4, (size_t)M * DIM * 4, (size_t)M * INTER * 2};
      for (int i = 0; i < 8; ++i) {
        const size_t d = diff_bytes(bufs[i].second.first, bufs[i].second.second, sz[i]);
        if (d) { ok = false; detail += std::string(" ") + bufs[i].first + "≠" + std::to_string(d) + "B"; }
      }
      return ok;
    };
    printf("%2s  %-12s %-34s %10s\n", "M", "variant", "bit-compare(vs old chain)", "µs/layer");
    for (const int M : {1, 4, 8}) {
      seed_inputs(R, 5u + M); CUDA_CHECK(cudaDeviceSynchronize());
      chain(0, R, M); CUDA_CHECK(cudaStreamSynchronize(st));
      const double t_old = time_us([&] { chain(0, R, M); }, reps, st, nullptr, 0) / NL;
      printf("%2d  %-12s %-34s %10.1f\n", M, "old", "(reference)", t_old);
      for (int variant = 1; variant <= 3; ++variant) {  // 1 = new · 2 = new + PDL · 3 = new + PDL, graph
        k::gemv2_force_pdl(variant >= 2 ? 1 : 0);
        seed_inputs(N, 5u + M); CUDA_CHECK(cudaDeviceSynchronize());
        double t = 0;
        std::string detail;
        bool ok = true;
        if (variant < 3) {
          chain(1, N, M); CUDA_CHECK(cudaStreamSynchronize(st)); CUDA_CHECK(cudaGetLastError());
          ok = compare(M, detail);
          t = time_us([&] { chain(1, N, M); }, reps, st, nullptr, 0) / NL;
        } else {
          cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
          cudaError_t e = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
          if (e == cudaSuccess) { chain(1, N, M); e = cudaStreamEndCapture(st, &g); }
          if (e == cudaSuccess) e = cudaGraphInstantiate(&ge, g, 0);
          size_t n_edges = 0, n_prog = 0;
          if (e == cudaSuccess) {  // how many captured edges are programmatic (did PDL make it into the graph)
            size_t ne = 0;
            if (cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &ne) == cudaSuccess && ne) {
              std::vector<cudaGraphNode_t> from(ne), to(ne); std::vector<cudaGraphEdgeData> ed(ne);
              if (cudaGraphGetEdges(g, from.data(), to.data(), ed.data(), &ne) == cudaSuccess)
                for (size_t i = 0; i < ne; ++i) n_prog += ed[i].type == cudaGraphDependencyTypeProgrammatic;
              n_edges = ne;
            }
          }
          if (e != cudaSuccess) {
            ok = false; detail = std::string(" graph capture/instantiate failed: ") + cudaGetErrorString(e);
            (void)cudaGetLastError();
          } else {
            seed_inputs(N, 5u + M); CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGraphLaunch(ge, st)); CUDA_CHECK(cudaStreamSynchronize(st)); CUDA_CHECK(cudaGetLastError());
            ok = compare(M, detail);
            t = time_us([&] { CUDA_CHECK(cudaGraphLaunch(ge, st)); }, reps, st, nullptr, 0) / NL;
            detail += " [edges " + std::to_string(n_edges) + ", programmatic " + std::to_string(n_prog) + "]";
          }
          if (ge) cudaGraphExecDestroy(ge);
          if (g) cudaGraphDestroy(g);
        }
        const char* names[] = {"", "new", "new+PDL", "new+PDL graph"};
        printf("%2d  %-12s %-34s %10.1f  (%.2fx)\n", M, names[variant], ((ok ? std::string("identical") : std::string("DIFF")) + detail).c_str(), t,
               t > 0 ? t_old / t : 0.0);
        all_ok = all_ok && ok;
      }
      k::gemv2_force_pdl(0);
      // graph replay of the previous chain (reference time in the same replay mode as serving)
      {
        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        chain(0, R, M);
        CUDA_CHECK(cudaStreamEndCapture(st, &g));
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        const double t = time_us([&] { CUDA_CHECK(cudaGraphLaunch(ge, st)); }, reps, st, nullptr, 0) / NL;
        printf("%2d  %-12s %-34s %10.1f\n", M, "old graph", "(reference timing)", t);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
      }
    }
    free_acts(R); free_acts(N);
    for (auto& w : W) free_weights(w);
  }
  cudaFree(flush);
  cudaStreamDestroy(st);
  printf(all_ok ? "\nALL PASS\n" : "\nSOME FAIL\n");
  return all_ok ? 0 : 1;
}
