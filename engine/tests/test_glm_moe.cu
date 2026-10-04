// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM MoE decode kernel == CPU expert_forward_nvfp4 (same bf16 rounding points) for several row/pair layouts (M=1 with 8 distinct experts,
//   M=4 with experts repeated across rows, groups larger than one chunk, unsorted pair order), run-to-run bitwise determinism, the old v0
//   kernel as a second reference, and a timing table (L2 flushed before every iteration) old vs new.
// Negative controls: build glm_moe.cu with -DHIVE_GLM_MOE_NEGCTRL (nibble order swapped) or -DHIVE_GLM_MOE_NEGCTRL_SCALE (scale pair
//   swapped) — the CPU comparison must then FAIL (exit 1); -DHIVE_GLM_MOE_NO_FALLBACK (no exact recompute of items whose f16 tensor-core
//   operands overflow) must FAIL exactly the two "x outside f16 range" cases.
// Timing: the L2 (128 MB) is flushed before every iteration by READING a 256 MB buffer (a memset flush leaves up to 128 MB of dirty lines
//   whose write-back would be charged to the kernel under test).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/expert_cpu.h"
#include "hive/glm/glm_moe.h"

namespace hive::glm {  // old kernel kept in glm_moe.cu for comparison (not part of the public header)
size_t moe_decode_v0_ws_bytes(int n_pairs, int H, int I);
void moe_decode_v0(const ExpertRecLayout& L, const MoePair* pairs, int n_pairs, const __nv_bfloat16* x, int H, int I, float limit, float* out,
                   void* ws, cudaStream_t st);
}  // namespace hive::glm

using namespace hive;
static int fails = 0;
__global__ void flush_read(const uint4* __restrict__ p, size_t n, unsigned* o) {
  unsigned a = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const uint4 v = p[i]; a ^= v.x ^ v.w; }
  if (a == 0x9E3779B9u) *o = a;  // never true for the memset pattern; keeps the loads alive
}
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static const int H = 4096, I = 2048, NR = 64;
static const float kLimit = 10.f;

struct Case { const char* name; int M; std::vector<int> row, rec; std::vector<float> w; float limit = kLimit; bool bigx = false; };

static uint64_t sm64(uint64_t& s) { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }

// M rows, each with 8 distinct records drawn from [0, pool) (pool ≥ 8); ascending row order like the engine.
static Case routed(const char* name, int M, int pool, int pool0, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ud(0.05f, 1.f);
  Case c{name, M, {}, {}, {}};
  for (int m = 0; m < M; ++m) {
    std::vector<int> ids(pool);
    for (int i = 0; i < pool; ++i) ids[i] = pool0 + i;
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int k = 0; k < 8; ++k) { c.row.push_back(m); c.rec.push_back(ids[k]); c.w.push_back(ud(rng)); }
  }
  return c;
}
static int unique_recs(const Case& c) { std::vector<int> u = c.rec; std::sort(u.begin(), u.end()); return (int)(std::unique(u.begin(), u.end()) - u.begin()); }

int main() {
  const glm::ExpertRecLayout L = glm::ExpertRecLayout::make(H, I);
  // ---- records: random nibbles, e4m3 scales in [0x28, 0x47] (0.25 .. 3.75), fixed global scales ----
  std::vector<std::vector<uint8_t>> recs(NR, std::vector<uint8_t>(L.total, 0));
  uint64_t seed = 12345;
  for (auto& r : recs) {
    for (size_t i = 0; i + 8 <= L.g; i += 8) { const uint64_t v = sm64(seed); memcpy(r.data() + i, &v, 8); }
    for (size_t off : {L.s1, L.s3}) for (size_t i = 0; i < (size_t)I * H / 16; ++i) r[off + i] = (uint8_t)(0x28 + sm64(seed) % 0x20);
    for (size_t i = 0; i < (size_t)H * I / 16; ++i) r[L.s2 + i] = (uint8_t)(0x28 + sm64(seed) % 0x20);
    float g[3] = {0.0031f, 0.0027f, 0.0042f}; memcpy(r.data() + L.g, g, 12);
  }
  std::vector<uint8_t*> drec(NR);
  for (int e = 0; e < NR; ++e) { CUDA_CHECK(cudaMalloc(&drec[e], L.total)); CUDA_CHECK(cudaMemcpy(drec[e], recs[e].data(), L.total, cudaMemcpyHostToDevice)); }
  const int MMAX = 64;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> x((size_t)MMAX * H); std::vector<float> xf(x.size());
  for (size_t i = 0; i < x.size(); ++i) { x[i] = f2bf(nd(rng)); xf[i] = bf2f(x[i]); }
  bf16* dx; CUDA_CHECK(cudaMalloc(&dx, x.size() * 2)); CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  // "big" activations: every 97th element of each row is ±3·10^5 (outside the f16 range → exercises the exact CUDA-core fallback)
  std::vector<bf16> xb = x; std::vector<float> xbf = xf;
  for (size_t i = 0; i < xb.size(); i += 97) { xb[i] = f2bf((i & 1 ? -3e5f : 3e5f)); xbf[i] = bf2f(xb[i]); }
  bf16* dxb; CUDA_CHECK(cudaMalloc(&dxb, xb.size() * 2)); CUDA_CHECK(cudaMemcpy(dxb, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
  std::vector<float> out0((size_t)MMAX * H);  // non-zero initial out (moe_decode accumulates)
  for (auto& v : out0) v = 0.01f * nd(rng);
  float* dout; CUDA_CHECK(cudaMalloc(&dout, out0.size() * 4));
  glm::MoePair* dp; CUDA_CHECK(cudaMalloc(&dp, 1024 * sizeof(glm::MoePair)));
  const size_t wsb = std::max(glm::moe_decode_ws_bytes(512, H, I), glm::moe_decode_v0_ws_bytes(512, H, I));
  void* ws; CUDA_CHECK(cudaMalloc(&ws, wsb));
  printf("workspace: moe_decode_ws_bytes(8) = %zu B, (64) = %zu B, (512) = %zu B  (v0: %zu / %zu / %zu B)\n", glm::moe_decode_ws_bytes(8, H, I),
         glm::moe_decode_ws_bytes(64, H, I), glm::moe_decode_ws_bytes(512, H, I), glm::moe_decode_v0_ws_bytes(8, H, I),
         glm::moe_decode_v0_ws_bytes(64, H, I), glm::moe_decode_v0_ws_bytes(512, H, I));

  auto upload = [&](const Case& c) {
    std::vector<glm::MoePair> pp;
    for (size_t i = 0; i < c.row.size(); ++i) pp.push_back({c.row[i], drec[c.rec[i]], c.w[i]});
    CUDA_CHECK(cudaMemcpy(dp, pp.data(), pp.size() * sizeof(glm::MoePair), cudaMemcpyHostToDevice));
    return (int)pp.size();
  };
  auto run = [&](const Case& c, bool v0) {
    const int n = upload(c);
    CUDA_CHECK(cudaMemcpy(dout, out0.data(), (size_t)c.M * H * 4, cudaMemcpyHostToDevice));
    const bf16* xx = c.bigx ? dxb : dx;
    if (v0) glm::moe_decode_v0(L, dp, n, xx, H, I, c.limit, dout, ws, 0);
    else glm::moe_decode(L, dp, n, xx, H, I, c.limit, dout, ws, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> o((size_t)c.M * H);
    CUDA_CHECK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
    return o;
  };
  auto rel = [&](const std::vector<float>& a, const std::vector<float>& ref) {  // relative error of the added part (initial out removed)
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double r = (double)ref[i] - out0[i]; num += std::pow((double)a[i] - ref[i], 2); den += r * r; }
    return std::sqrt(num / den);
  };

  // ---- correctness cases ----
  std::vector<Case> cases;
  cases.push_back({"M=3 6 pairs (e1 on rows 0 and 2)", 3, {0, 0, 1, 2, 2, 2}, {0, 1, 2, 1, 3, 4}, {0.3f, 0.7f, 1.1f, 0.2f, 0.5f, 0.9f}});
  cases.push_back(routed("M=1 8 distinct experts", 1, 8, 8, 11));
  cases.push_back(routed("M=4 32 pairs, pool 10 (experts repeated across rows)", 4, 10, 16, 12));
  cases.push_back(routed("M=8 64 pairs, pool 64", 8, 64, 0, 13));
  cases.push_back(routed("M=16 128 pairs, pool 9 (groups > 8 rows → chunk split)", 16, 9, 30, 14));
  {
    Case c = routed("M=4 32 pairs, pool 12, shuffled pair order", 4, 12, 40, 15);
    std::vector<int> perm(c.row.size()); for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int)i;
    std::shuffle(perm.begin(), perm.end(), std::mt19937(99));
    Case s{c.name, c.M, {}, {}, {}};
    for (int i : perm) { s.row.push_back(c.row[i]); s.rec.push_back(c.rec[i]); s.w.push_back(c.w[i]); }
    cases.push_back(s);
  }
  cases.push_back(routed("M=64 512 pairs, pool 64 (max decode batch)", 64, 64, 0, 16));
  { Case c = routed("M=2 x outside f16 range, limit 10 (gate/up fallback)", 2, 16, 0, 17); c.bigx = true; cases.push_back(c); }
  { Case c = routed("M=2 x outside f16 range, no limit (gu + down fallback)", 2, 16, 0, 18); c.bigx = true; c.limit = 0.f; cases.push_back(c); }
  std::vector<float> tmp(3 * I), o(H);
  for (const Case& c : cases) {
    std::vector<float> ref(out0.begin(), out0.begin() + (size_t)c.M * H);
    for (size_t p = 0; p < c.row.size(); ++p) {
      const auto& r = recs[c.rec[p]];
      cpu::ExpertDescNvfp4 d{H, I, r.data() + L.w1, r.data() + L.s1, r.data() + L.w3, r.data() + L.s3, r.data() + L.w2, r.data() + L.s2};
      memcpy(&d.g1, r.data() + L.g, 4); memcpy(&d.g3, r.data() + L.g + 4, 4); memcpy(&d.g2, r.data() + L.g + 8, 4);
      cpu::expert_forward_nvfp4(d, (c.bigx ? xbf : xf).data() + (size_t)c.row[p] * H, c.w[p], c.limit, o.data(), tmp.data());
      for (int n = 0; n < H; ++n) ref[(size_t)c.row[p] * H + n] += o[n];
    }
    const auto a = run(c, false), b = run(c, false), v0 = run(c, true);
    const double e_new = rel(a, ref), e_v0 = rel(v0, ref), e_nv = rel(a, v0);
    const bool det = memcmp(a.data(), b.data(), a.size() * 4) == 0;
    const bool fin = std::all_of(a.begin(), a.end(), [](float v) { return std::isfinite(v); });
    EXPECT(fin && e_new < 5e-3, "%s: moe_decode vs CPU rel %.3g%s", c.name, e_new, fin ? "" : " (non-finite output)");  // bf16 flips (fp32 reassociation)
    EXPECT(det, "%s: two runs differ bitwise", c.name);
    printf("%-58s uniq %2d  new vs CPU %.2e  v0 vs CPU %.2e  new vs v0 %.2e  deterministic %s  %s\n", c.name, unique_recs(c), e_new, e_v0, e_nv,
           det ? "yes" : "NO", fin && e_new < 5e-3 && det ? "PASS" : "FAIL");
  }

  // ---- timing: L2 flushed (256 MB read) before every iteration; median of 25 ----
  void* flush; const size_t flush_b = 256u << 20; CUDA_CHECK(cudaMalloc(&flush, flush_b)); CUDA_CHECK(cudaMemset(flush, 1, flush_b));
  cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb);
  auto time_case = [&](const Case& c, bool v0) {
    const int n = upload(c);
    std::vector<float> ms;
    for (int it = 0; it < 30; ++it) {
      flush_read<<<1024, 256>>>(reinterpret_cast<const uint4*>(flush), flush_b / 16, reinterpret_cast<unsigned*>(flush));
      cudaEventRecord(ea, 0);
      if (v0) glm::moe_decode_v0(L, dp, n, dx, H, I, kLimit, dout, ws, 0);
      else glm::moe_decode(L, dp, n, dx, H, I, kLimit, dout, ws, 0);
      cudaEventRecord(eb, 0);
      CUDA_CHECK(cudaEventSynchronize(eb));
      float t; cudaEventElapsedTime(&t, ea, eb);
      if (it >= 5) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
  };
  std::vector<Case> tcases;
  tcases.push_back(routed("M=1", 1, 8, 0, 21));
  tcases.push_back(routed("M=2", 2, 64, 0, 22));
  tcases.push_back(routed("M=4", 4, 64, 0, 23));
  tcases.push_back(routed("M=8", 8, 64, 0, 24));
  { Case c{"M=8 distinct", 8, {}, {}, {}}; for (int i = 0; i < 64; ++i) { c.row.push_back(i / 8); c.rec.push_back(i); c.w.push_back(0.1f); } tcases.push_back(c); }
  { Case c{"M=8 all rows same 8", 8, {}, {}, {}}; for (int i = 0; i < 64; ++i) { c.row.push_back(i / 8); c.rec.push_back(i % 8); c.w.push_back(0.1f); } tcases.push_back(c); }
  tcases.push_back(routed("M=64", 64, 64, 0, 25));
  printf("\n%-22s %5s %5s | %10s %10s %10s | %10s %10s %10s | %7s\n", "case", "pairs", "uniq", "v0 ms", "v0 GB/s", "v0 pairGB/s", "new ms",
         "new GB/s", "new pairGB/s", "speedup");
  const double recb = (double)L.g;  // bytes streamed per record (w/s/g, without the 4 KiB pad)
  for (const Case& c : tcases) {
    const double t0 = time_case(c, true), t1 = time_case(c, false);
    const int np = (int)c.row.size(), u = unique_recs(c);
    printf("%-22s %5d %5d | %10.3f %10.0f %10.0f | %10.3f %10.0f %10.0f | %6.2fx\n", c.name, np, u, t0, u * recb / (t0 * 1e6), np * recb / (t0 * 1e6), t1,
           u * recb / (t1 * 1e6), np * recb / (t1 * 1e6), t0 / t1);
  }
  printf("(GB/s = unique record bytes / time = DRAM traffic; pairGB/s = pairs × record bytes / time)\n");

  if (fails) { fprintf(stderr, "glm moe: %d failures\n", fails); return 1; }
  puts("glm moe: decode kernel == CPU reference, deterministic");
  return 0;
}
