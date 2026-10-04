// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// E3 validation of HIVE_HC_DECODE_FUSED / HIVE_HC_SINKHORN_PAR: decode hc mixing + sinkhorn.
//   Shapes (the checkpoint's config.json): hc_mult 4 · hidden 5120 · mix_hc 24 · hc_sinkhorn_iters 20 · hc_eps 1e-6 ·
//   rms_norm_eps 1e-20 · 40 layers (× 2 stages).
//   (1) single shot (M = 1, 4, 8 · mixing variant v1 = hc_mix_pre_norm · v2 = hc_mix_pre_norm2 with PDL off/on):
//        old = mixing kernel -> hc_split_sinkhorn (one thread = one token)  vs  new = *_sk (sinkhorn in the last block)
//        — byte comparison of mixes·rsq·x·xn·pre·post·comb. Running the new variant twice in a row (counter must return
//        to 0) and replaying a graph capture 3 times must give the same bytes.
//        hc_split_sinkhorn_par vs hc_split_sinkhorn (same mixes) — two sets, scale 1 and 5 (softmax saturation side).
//   (2) layer chain (NL layers × [attention pre-stage -> attention stand-in -> hc_post -> ffn pre-stage -> moe stand-in
//        -> hc_post_tail], weight copy per layer — outside L2):
//        A old (mix -> serial sinkhorn on a side stream · fork/join) · B side stream + parallel sinkhorn · C fused (no
//        side stream). The final h·pre_mix of all three must be byte-identical, plus CUDA-graph replay µs per layer.
//        Stand-in = one clock64-spinning block per SM (µs argument) — 0 is the worst case where sinkhorn is exposed; the
//        measured values (attention ≈60, moe ≈150 µs) are close to reality, where the old side stream overlaps and hides.
//   (3) sinkhorn alone, µs per launch (old serial vs parallel, M 1·4·8).
//   Verdict: all bit-identical = ALL PASS. Run: scripts/hive-run.sh "./build-dev/test_hc_decode [reps=50] [NL=40] [attn_us=60] [moe_us=150]"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "hive/common.h"
#include "hive/gemv_decode.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"

using namespace hive;

namespace {

constexpr int HC = 4, DIM = 5120, HCDIM = HC * DIM, MIX = (2 + HC) * HC, ITERS = 20;
constexpr float HC_EPS = 1e-6f, NORM_EPS = 1e-20f;

__device__ __forceinline__ uint32_t hsh(size_t i, uint32_t seed) {
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed ^ (uint32_t)(i >> 32) * 0x9E3779B9u;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  return h;
}
__global__ void fill_bf16_kernel(bf16* p, size_t n, uint32_t seed, float a, float b) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = f2bf(a + b * ((float)(hsh(i, seed) >> 8) * (2.f / 16777216.f) - 1.f));
}
__global__ void fill_f32_kernel(float* p, size_t n, uint32_t seed, float a, float b) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = a + b * ((float)(hsh(i, seed) >> 8) * (2.f / 16777216.f) - 1.f);
}
__global__ void spin_kernel(long long cycles) {  // stand-in: occupies each block for `cycles`
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}
unsigned nblk(size_t n) { return (unsigned)((n + 255) / 256); }
void fill_bf16(bf16* p, size_t n, uint32_t seed, float a, float b) { fill_bf16_kernel<<<nblk(n), 256>>>(p, n, seed, a, b); CUDA_CHECK(cudaGetLastError()); }
void fill_f32(float* p, size_t n, uint32_t seed, float a, float b) { fill_f32_kernel<<<nblk(n), 256>>>(p, n, seed, a, b); CUDA_CHECK(cudaGetLastError()); }
template <class T> T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T))); CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T))); return p; }
std::vector<uint8_t> host(const void* d, size_t bytes) { std::vector<uint8_t> v(bytes); CUDA_CHECK(cudaMemcpy(v.data(), d, bytes, cudaMemcpyDeviceToHost)); return v; }

struct Stage { float *fn, *scale, *base; bf16* norm; };  // weights of one stage (before attention or before ffn)
struct Out {  // outputs of one stage
  float *mixes, *rsq, *pre, *post, *comb; bf16 *x, *xn;
  void alloc(int M) {
    mixes = dmalloc<float>((size_t)M * MIX); rsq = dmalloc<float>(M); pre = dmalloc<float>((size_t)M * HC); post = dmalloc<float>((size_t)M * HC);
    comb = dmalloc<float>((size_t)M * HC * HC); x = dmalloc<bf16>((size_t)M * DIM); xn = dmalloc<bf16>((size_t)M * DIM);
  }
  void free_() { for (void* p : {(void*)mixes, (void*)rsq, (void*)pre, (void*)post, (void*)comb, (void*)x, (void*)xn}) cudaFree(p); }
  std::vector<uint8_t> bytes(int M) const {
    std::vector<uint8_t> v;
    auto add = [&](const void* d, size_t n) { auto b = host(d, n); v.insert(v.end(), b.begin(), b.end()); };
    add(mixes, (size_t)M * MIX * 4); add(rsq, (size_t)M * 4); add(pre, (size_t)M * HC * 4); add(post, (size_t)M * HC * 4); add(comb, (size_t)M * HC * HC * 4);
    add(x, (size_t)M * DIM * 2); add(xn, (size_t)M * DIM * 2);
    return v;
  }
};
Stage make_stage(uint32_t seed, float scale_mag) {
  Stage s;
  s.fn = dmalloc<float>((size_t)MIX * HCDIM); s.scale = dmalloc<float>(3); s.base = dmalloc<float>(MIX); s.norm = dmalloc<bf16>(DIM);
  fill_f32(s.fn, (size_t)MIX * HCDIM, seed, 0.f, 0.02f);
  fill_f32(s.scale, 3, seed + 1, scale_mag, 0.5f * scale_mag);
  fill_f32(s.base, MIX, seed + 2, 0.f, 1.f);
  fill_bf16(s.norm, DIM, seed + 3, 1.f, 0.1f);
  return s;
}

// v: 0 = v1 (hc_mix_pre_norm) · 1 = v2 (hc_mix_pre_norm2) — PDL is set outside via gemv2_force_pdl
void old_stage(int v, const Stage& S, const bf16* h, const float* pre_in, int M, Out& o, cudaStream_t mix_st, cudaStream_t sk_st, bool par) {
  (v ? k::hc_mix_pre_norm2 : k::hc_mix_pre_norm)(h, S.fn, M, HCDIM, MIX, NORM_EPS, o.mixes, o.rsq, pre_in, HC, DIM, S.norm, NORM_EPS, o.x, o.xn, mix_st);
  (void)sk_st;
  (par ? k::hc_split_sinkhorn_par : k::hc_split_sinkhorn)(o.mixes, o.rsq, S.scale, S.base, M, HC, ITERS, HC_EPS, o.pre, o.post, o.comb, mix_st);
}
void new_stage(int v, const Stage& S, const bf16* h, const float* pre_in, int M, Out& o, int* cnt, cudaStream_t st) {
  (v ? k::hc_mix_pre_norm2_sk : k::hc_mix_pre_norm_sk)(h, S.fn, M, HCDIM, MIX, NORM_EPS, o.mixes, o.rsq, pre_in, HC, DIM, S.norm, NORM_EPS, o.x, o.xn, S.scale,
                                                      S.base, ITERS, HC_EPS, o.pre, o.post, o.comb, cnt, st);
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::max(1, atoi(argv[1])) : 50;
  const int NL = argc > 2 ? std::max(1, atoi(argv[2])) : 40;
  const double attn_us = argc > 3 ? atof(argv[3]) : 60.0, moe_us = argc > 4 ? atof(argv[4]) : 150.0;
  int dev = 0, clk_khz = 0, nsm = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  k::hc_decode_fused_preload();
  cudaStream_t st, side;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
  int* cnt = dmalloc<int>(16);
  bool all_ok = true;
  auto verdict = [&](bool ok, const char* fmt, auto... a) { printf(fmt, a...); printf("  %s\n", ok ? "PASS" : "FAIL"); all_ok = all_ok && ok; };

  // (1) single-shot bit comparison
  for (const float smag : {1.f, 5.f}) {
    const Stage S = make_stage(100 + (uint32_t)smag, smag);
    for (const int M : {1, 4, 8}) {
      bf16* h = dmalloc<bf16>((size_t)M * HCDIM); float* pre_in = dmalloc<float>((size_t)M * HC);
      fill_bf16(h, (size_t)M * HCDIM, 7 + M, 0.f, 2.f); fill_f32(pre_in, (size_t)M * HC, 9 + M, 0.25f, 0.1f);
      for (int v = 0; v < 2; ++v)
        for (int pdl = 0; pdl < (v ? 2 : 1); ++pdl) {
          k::gemv2_force_pdl(pdl);
          Out a, b; a.alloc(M); b.alloc(M);
          old_stage(v, S, h, pre_in, M, a, st, side, false);
          new_stage(v, S, h, pre_in, M, b, cnt, st);
          CUDA_CHECK(cudaStreamSynchronize(st));
          const auto ra = a.bytes(M), rb = b.bytes(M);
          new_stage(v, S, h, pre_in, M, b, cnt, st);  // second run (checks the counter returned to 0)
          CUDA_CHECK(cudaStreamSynchronize(st));
          const auto rb2 = b.bytes(M);
          const int c0 = host(cnt, 16 * 4)[0];
          // graph capture replay
          cudaGraph_t g; cudaGraphExec_t ge;
          CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
          new_stage(v, S, h, pre_in, M, b, cnt, st);
          CUDA_CHECK(cudaStreamEndCapture(st, &g));
          CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
          bool gok = true;
          for (int r = 0; r < 3; ++r) { CUDA_CHECK(cudaMemset(b.comb, 0, (size_t)M * HC * HC * 4)); CUDA_CHECK(cudaGraphLaunch(ge, st)); CUDA_CHECK(cudaStreamSynchronize(st)); gok = gok && b.bytes(M) == ra; }
          cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
          verdict(ra == rb && rb2 == ra && c0 == 0 && gok, "① scale %.0f M=%d %s%s: run 1 %s · run 2 %s · cnt %d · graph %s", smag, M, v ? "v2" : "v1",
                  v ? (pdl ? "+PDL" : "   ") : "   ", ra == rb ? "=" : "≠", rb2 == ra ? "=" : "≠", c0, gok ? "=" : "≠");
          // parallel sinkhorn alone
          if (v == 0) {
            Out c; c.alloc(M);
            CUDA_CHECK(cudaMemcpy(c.mixes, a.mixes, (size_t)M * MIX * 4, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(c.rsq, a.rsq, (size_t)M * 4, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(c.x, a.x, (size_t)M * DIM * 2, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(c.xn, a.xn, (size_t)M * DIM * 2, cudaMemcpyDeviceToDevice));
            k::hc_split_sinkhorn_par(c.mixes, c.rsq, S.scale, S.base, M, HC, ITERS, HC_EPS, c.pre, c.post, c.comb, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            const bool ok = c.bytes(M) == ra;
            verdict(ok, "① scale %.0f M=%d sinkhorn_par vs sinkhorn: %s", smag, M, ok ? "=" : "≠");
            c.free_();
          }
          a.free_(); b.free_();
        }
      cudaFree(h); cudaFree(pre_in);
    }
  }
  k::gemv2_force_pdl(-1);

  // (2) layer chain
  std::vector<Stage> SA, SF;
  for (int l = 0; l < NL; ++l) { SA.push_back(make_stage(1000 + 10 * l, 1.f)); SF.push_back(make_stage(5000 + 10 * l, 1.f)); }
  cudaEvent_t fork, join, e0, e1;
  CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  for (const int M : {1, 4, 8}) {
    bf16 *h0 = dmalloc<bf16>((size_t)M * HCDIM), *h = dmalloc<bf16>((size_t)M * HCDIM), *attn_out = dmalloc<bf16>((size_t)M * DIM);
    float *pm0 = dmalloc<float>((size_t)M * HC), *pre_mix = dmalloc<float>((size_t)M * HC), *acc = dmalloc<float>((size_t)M * DIM);
    fill_bf16(h0, (size_t)M * HCDIM, 31 + M, 0.f, 1.f); fill_f32(pm0, (size_t)M * HC, 37 + M, 0.25f, 0.05f);
    fill_bf16(attn_out, (size_t)M * DIM, 41 + M, 0.f, 0.5f); fill_f32(acc, (size_t)M * DIM, 43 + M, 0.f, 0.5f);
    Out oa, of; oa.alloc(M); of.alloc(M);
    for (const double sc : {0.0, 1.0}) {
      const long long ca = (long long)(attn_us * sc * clk_khz / 1000.0), cm = (long long)(moe_us * sc * clk_khz / 1000.0);
      for (int v = 0; v < 2; ++v)
        for (int pdl = 0; pdl < (v ? 2 : 1); ++pdl) {
          k::gemv2_force_pdl(pdl);
          std::vector<uint8_t> ref;
          double base_us = 0;
          for (int mode = 0; mode < 3; ++mode) {  // 0 old · 1 side stream + parallel · 2 fused
            auto stage = [&](const Stage& S, const float* pre_in, Out& o, int ci) {
              if (mode == 2) { new_stage(v, S, h, pre_in, M, o, cnt + ci, st); return; }
              (v ? k::hc_mix_pre_norm2 : k::hc_mix_pre_norm)(h, S.fn, M, HCDIM, MIX, NORM_EPS, o.mixes, o.rsq, pre_in, HC, DIM, S.norm, NORM_EPS, o.x, o.xn, st);
              CUDA_CHECK(cudaEventRecord(fork, st)); CUDA_CHECK(cudaStreamWaitEvent(side, fork, 0));
              (mode ? k::hc_split_sinkhorn_par : k::hc_split_sinkhorn)(o.mixes, o.rsq, S.scale, S.base, M, HC, ITERS, HC_EPS, o.pre, o.post, o.comb, side);
              CUDA_CHECK(cudaEventRecord(join, side));
            };
            auto chain = [&] {
              CUDA_CHECK(cudaMemcpyAsync(h, h0, (size_t)M * HCDIM * 2, cudaMemcpyDeviceToDevice, st));
              CUDA_CHECK(cudaMemcpyAsync(pre_mix, pm0, (size_t)M * HC * 4, cudaMemcpyDeviceToDevice, st));
              for (int l = 0; l < NL; ++l) {
                stage(SA[l], pre_mix, oa, 3);
                if (ca) spin_kernel<<<nsm, 32, 0, st>>>(ca);
                if (mode != 2) CUDA_CHECK(cudaStreamWaitEvent(st, join, 0));
                k::hc_post(attn_out, oa.post, oa.comb, M, HC, DIM, h, st);
                stage(SF[l], oa.pre, of, 4);
                if (cm) spin_kernel<<<nsm, 32, 0, st>>>(cm);
                if (mode != 2) CUDA_CHECK(cudaStreamWaitEvent(st, join, 0));
                k::hc_post_tail(acc, of.post, of.comb, M, HC, DIM, h, of.pre, pre_mix, st);
              }
            };
            cudaGraph_t g; cudaGraphExec_t ge;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            chain();
            CUDA_CHECK(cudaStreamEndCapture(st, &g));
            CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
            CUDA_CHECK(cudaGraphLaunch(ge, st)); CUDA_CHECK(cudaStreamSynchronize(st));
            std::vector<uint8_t> out = host(h, (size_t)M * HCDIM * 2), pmb = host(pre_mix, (size_t)M * HC * 4);
            out.insert(out.end(), pmb.begin(), pmb.end());
            chain();  // eager run must give the same bytes
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::vector<uint8_t> out2 = host(h, (size_t)M * HCDIM * 2), pmb2 = host(pre_mix, (size_t)M * HC * 4);
            out2.insert(out2.end(), pmb2.begin(), pmb2.end());
            CUDA_CHECK(cudaEventRecord(e0, st));
            for (int r = 0; r < reps; ++r) CUDA_CHECK(cudaGraphLaunch(ge, st));
            CUDA_CHECK(cudaEventRecord(e1, st)); CUDA_CHECK(cudaEventSynchronize(e1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            const double us_layer = ms * 1000.0 / reps / NL;
            cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
            if (mode == 0) { ref = out; base_us = us_layer; }
            const bool ok = out == ref && out2 == ref;
            static const char* names[3] = {"A baseline (branch, serial)", "B branch+parallel          ", "C fused (no branch)        "};
            verdict(ok, "② M=%d %s%s band %s  %s: %.2f µs/layer (Δ vs A %+.2f µs · hc share without band is on the band-0 line)", M, v ? "v2" : "v1", v ? (pdl ? "+PDL" : "   ") : "   ",
                    sc ? "real" : "0   ", names[mode], us_layer, us_layer - base_us);
          }
        }
    }
    k::gemv2_force_pdl(-1);
    oa.free_(); of.free_();
    for (void* p : {(void*)h0, (void*)h, (void*)attn_out, (void*)pm0, (void*)pre_mix, (void*)acc}) cudaFree(p);
  }

  // (3) sinkhorn alone, µs
  {
    const Stage S = make_stage(77, 1.f);
    for (const int M : {1, 4, 8}) {
      Out o; o.alloc(M);
      fill_f32(o.mixes, (size_t)M * MIX, 5, 0.f, 2.f); fill_f32(o.rsq, M, 6, 1.f, 0.1f);
      for (int par = 0; par < 2; ++par) {
        cudaGraph_t g; cudaGraphExec_t ge;
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < 2 * NL; ++i)
          (par ? k::hc_split_sinkhorn_par : k::hc_split_sinkhorn)(o.mixes, o.rsq, S.scale, S.base, M, HC, ITERS, HC_EPS, o.pre, o.post, o.comb, st);
        CUDA_CHECK(cudaStreamEndCapture(st, &g));
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        CUDA_CHECK(cudaGraphLaunch(ge, st));
        CUDA_CHECK(cudaEventRecord(e0, st));
        for (int r = 0; r < reps; ++r) CUDA_CHECK(cudaGraphLaunch(ge, st));
        CUDA_CHECK(cudaEventRecord(e1, st)); CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        printf("③ sinkhorn %s M=%d: %.2f µs per launch (graph of %d in a row)\n", par ? "par16" : "serial", M, ms * 1000.0 / reps / (2 * NL), 2 * NL);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
      }
      o.free_();
    }
  }
  printf(all_ok ? "ALL PASS\n" : "SOME FAIL\n");
  return all_ok ? 0 : 1;
}
