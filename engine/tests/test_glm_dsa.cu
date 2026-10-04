// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash DSA kernels vs transformers 5.18 Glm5NextTextAttention / Glm5NextTextIndexer references.
// References: a transformers script (not included) → /tmp/dsa_ref/S<S>/ (inputs from the bf16 module, selected indices of the bf16
// module, attention output before o_proj of the fp32 module fed with the same bf16 inputs and the same indices).
//
// Usage: t_dsa [ref_root=/tmp/dsa_ref] [S list...]       — correctness (default S = 1500 2600 6000)
//        t_dsa bench                                     — timings (random data)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "hive/glm/dsa.h"

using namespace hive;
using namespace hive::glm;

namespace {

int g_fail = 0;
#define EXPECT(cond, ...)                 \
  do {                                    \
    if (!(cond)) {                        \
      printf("  FAIL: " __VA_ARGS__);     \
      printf("\n");                       \
      ++g_fail;                           \
    }                                     \
  } while (0)

float b2f(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
uint16_t f2b(float f) {  // RNE
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

template <class T>
std::vector<T> load(const std::string& path, size_t n) {
  std::vector<T> v(n);
  FILE* f = fopen(path.c_str(), "rb");
  HIVE_CHECK(f, "cannot open " + path);
  HIVE_CHECK(fread(v.data(), sizeof(T), n, f) == n, "short read " + path);
  fclose(f);
  return v;
}

template <class T>
T* dev(size_t n) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}
template <class T>
T* up(const std::vector<T>& v) {
  T* p = dev<T>(v.size());
  CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
template <class T>
std::vector<T> down(const T* p, size_t n) {
  std::vector<T> v(n);
  CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

struct Cache {
  DsaSeqCache c{};
  void alloc(int cap) {
    c.cap = cap;
    c.c = dev<bf16>((size_t)cap * kDsaLat);
    c.kI = dev<bf16>((size_t)cap * kDsaIdxDim);
    c.gs = dev<bf16>((size_t)cap * kDsaIdxDim);
    c.pooled = dev<bf16>((size_t)cap / 4 * kDsaIdxDim);
    CUDA_CHECK(cudaMemset(c.pooled, 0, (size_t)cap / 4 * kDsaIdxDim * 2));
  }
  void free_() {
    cudaFree(c.c); cudaFree(c.kI); cudaFree(c.gs); cudaFree(c.pooled);
  }
};

// relative L2 error over rows [r0, r1) of [*, width] (a: bf16 device output, ref: fp32 host)
double rel_err(const std::vector<uint16_t>& a, const std::vector<float>& ref, size_t width, int r0, int r1) {
  double num = 0, den = 0;
  for (size_t i = (size_t)r0 * width; i < (size_t)r1 * width; ++i) {
    const double d = (double)b2f(a[i]) - ref[i];
    num += d * d;
    den += (double)ref[i] * ref[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}
double rel_err_bb(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)b2f(a[i]) - b2f(b[i]);
    num += d * d;
    den += (double)b2f(b[i]) * b2f(b[i]);
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

std::set<int> row_set(const int32_t* r) {
  std::set<int> s;
  for (int e = 0; e < kDsaIdxWidth; ++e)
    if (r[e] >= 0) s.insert(r[e]);
  return s;
}

struct Ref {
  int S = 0;
  std::vector<uint16_t> kv_b, ape, q, c, qI, kI, gs;
  std::vector<float> wI, out_f32, out_bf16;
  std::vector<int32_t> idx;
};

Ref load_ref(const std::string& dir, int S) {
  Ref r;
  r.S = S;
  r.kv_b = load<uint16_t>(dir + "/kv_b.bin", (size_t)kDsaHeads * 512 * kDsaLat);
  r.ape = load<uint16_t>(dir + "/ape.bin", 4 * 128);
  r.q = load<uint16_t>(dir + "/q.bin", (size_t)S * kDsaHeads * kDsaQkDim);
  r.c = load<uint16_t>(dir + "/c.bin", (size_t)S * kDsaLat);
  r.qI = load<uint16_t>(dir + "/qI.bin", (size_t)S * kDsaIdxHeads * kDsaIdxDim);
  r.kI = load<uint16_t>(dir + "/kI.bin", (size_t)S * kDsaIdxDim);
  r.gs = load<uint16_t>(dir + "/gs.bin", (size_t)S * kDsaIdxDim);
  r.wI = load<float>(dir + "/wI.bin", (size_t)S * kDsaIdxHeads);
  r.idx = load<int32_t>(dir + "/idx.bin", (size_t)S * kDsaIdxWidth);
  r.out_f32 = load<float>(dir + "/out_f32.bin", (size_t)S * kDsaHeads * kDsaVDim);
  r.out_bf16 = load<float>(dir + "/out_bf16.bin", (size_t)S * kDsaHeads * kDsaVDim);
  return r;
}

// Host mirror of the pooled keys (same op order as the reference)
std::vector<uint16_t> host_pooled(const Ref& r, int npool, bool use_ape = true) {
  std::vector<uint16_t> P((size_t)npool * 128);
  for (int p = 0; p < npool; ++p)
    for (int ch = 0; ch < 128; ++ch) {
      float g[4], m = -INFINITY;
      for (int j = 0; j < 4; ++j) {
        g[j] = b2f(r.gs[(size_t)(4 * p + j) * 128 + ch]) + (use_ape ? b2f(r.ape[j * 128 + ch]) : 0.f);
        m = std::max(m, g[j]);
      }
      float s = 0;
      for (int j = 0; j < 4; ++j) { g[j] = expf(g[j] - m); s += g[j]; }
      float acc = 0;
      for (int j = 0; j < 4; ++j) {
        const float pr = b2f(f2b(g[j] / s));
        acc += b2f(f2b(pr * b2f(r.kI[(size_t)(4 * p + j) * 128 + ch])));
      }
      P[(size_t)p * 128 + ch] = f2b(acc);
    }
  return P;
}

// fp64 pool scores of query t (host), for tie analysis
std::vector<double> host_scores(const Ref& r, const std::vector<uint16_t>& P, int t) {
  const int n = (t + 1) / 4;
  std::vector<double> s(n, 0.0);
  for (int p = 0; p < n; ++p) {
    double acc = 0;
    for (int h = 0; h < 32; ++h) {
      double d = 0;
      for (int k = 0; k < 128; ++k)
        d += (double)b2f(r.qI[((size_t)t * 32 + h) * 128 + k]) * b2f(P[(size_t)p * 128 + k]);
      acc += (double)r.wI[(size_t)t * 32 + h] / std::sqrt(32.0) * std::max(d / std::sqrt(128.0), 0.0);
    }
    s[p] = acc;
  }
  return s;
}

struct Dev {
  bf16 *kv_b, *ape, *ape0, *q, *c, *qI, *kI, *gs;
  float* wI;
};

// Full one-shot run (append S, select, attention). Returns idx and out on host.
void run_oneshot(const Dev& d, const DsaLayerW& w, Cache& cache, int S, std::vector<int32_t>& idx_h,
                 std::vector<uint16_t>& out_h, int32_t* idx_override = nullptr, const bf16* q_override = nullptr) {
  int32_t* idx = dev<int32_t>((size_t)S * kDsaIdxWidth);
  bf16* out = dev<bf16>((size_t)S * kDsaHeads * kDsaVDim);
  const size_t sws = dsa_select_ws_bytes(S, S), aws = dsa_attention_ws_bytes(S);
  void* ws = dev<char>(std::max(sws, aws));
  dsa_append(w, cache.c, 0, S, d.c, d.kI, d.gs, 0);
  dsa_select(cache.c, 0, S, d.qI, d.wI, idx, ws, sws, 0);
  dsa_attention(w, cache.c, 0, S, q_override ? q_override : d.q, idx_override ? idx_override : idx, out, ws, aws, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  idx_h = down(idx, (size_t)S * kDsaIdxWidth);
  out_h = down(reinterpret_cast<uint16_t*>(out), (size_t)S * kDsaHeads * kDsaVDim);
  cudaFree(idx); cudaFree(out); cudaFree(ws);
}

void test_ref(const std::string& dir, int S) {
  printf("== reference S=%d (%s)\n", S, dir.c_str());
  Ref r = load_ref(dir, S);
  Dev d;
  d.kv_b = reinterpret_cast<bf16*>(up(r.kv_b));
  d.ape = reinterpret_cast<bf16*>(up(r.ape));
  d.ape0 = reinterpret_cast<bf16*>(up(std::vector<uint16_t>(512, 0)));
  d.q = reinterpret_cast<bf16*>(up(r.q));
  d.c = reinterpret_cast<bf16*>(up(r.c));
  d.qI = reinterpret_cast<bf16*>(up(r.qI));
  d.kI = reinterpret_cast<bf16*>(up(r.kI));
  d.gs = reinterpret_cast<bf16*>(up(r.gs));
  d.wI = up(r.wI);
  DsaLayerW w{d.kv_b, d.ape};
  const int cap = (S + 3) / 4 * 4;
  const size_t W = (size_t)kDsaHeads * kDsaVDim;
  const int t_sparse = std::min(S, kDsaIdxWidth);  // first row whose selection is sparse

  // ---- one-shot
  Cache c1;
  c1.alloc(cap);
  std::vector<int32_t> idx1;
  std::vector<uint16_t> out1;
  run_oneshot(d, w, c1, S, idx1, out1);

  // pooled keys vs host mirror
  {
    const int np = S / 4;
    auto P = down(reinterpret_cast<uint16_t*>(c1.c.pooled), (size_t)np * 128);
    auto H = host_pooled(r, np);
    size_t bad = 0;
    for (size_t i = 0; i < P.size(); ++i) bad += P[i] != H[i];
    printf("  pooled keys vs host mirror: %zu / %zu bf16 differ\n", bad, P.size());
    EXPECT(bad * 10000 <= P.size(), "pooled keys differ from the host mirror (%zu)", bad);
  }

  // (2) index sets vs HF
  int mism_rows = 0;
  size_t sym = 0;
  double worst_gap = 0;
  std::vector<uint16_t> Ph;
  for (int t = 0; t < S; ++t) {
    auto a = row_set(&idx1[(size_t)t * kDsaIdxWidth]), b = row_set(&r.idx[(size_t)t * kDsaIdxWidth]);
    if (a == b) continue;
    ++mism_rows;
    std::vector<int> diff;
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
    sym += diff.size();
    // tie analysis: distance of the differing pools' scores to the 512th score (relative to its magnitude)
    if (Ph.empty()) Ph = host_pooled(r, S / 4);
    auto s = host_scores(r, Ph, t);
    std::vector<double> srt = s;
    std::nth_element(srt.begin(), srt.begin() + 511, srt.end(), std::greater<double>());
    const double v512 = srt[511];
    for (int tok : diff) worst_gap = std::max(worst_gap, std::fabs(s[tok / 4] - v512) / std::max(std::fabs(v512), 1e-30));
    if (mism_rows <= 3) printf("  idx row %d differs: %zu tokens in symmetric difference\n", t, diff.size());
  }
  {  // documentation of the tail rule: rows whose own token is NOT in HF's selection (only possible when (t+1)%4 == 0)
    int own_missing = 0, own_missing_tail = 0;
    for (int t = 0; t < S; ++t) {
      auto b = row_set(&r.idx[(size_t)t * kDsaIdxWidth]);
      if (!b.count(t)) {
        ++own_missing;
        if ((t + 1) % 4 != 0) ++own_missing_tail;
      }
    }
    printf("  HF rows without their own token: %d (of which with an incomplete own pool: %d)\n", own_missing, own_missing_tail);
    EXPECT(own_missing_tail == 0, "tail rule: own token missing although its pool is incomplete");
  }
  printf("  (2) selected sets vs HF: %d / %d rows differ (sym-diff tokens %zu); max |s-s512|/|s512| of differing pools %.2e\n",
         mism_rows, S, sym, worst_gap);
  EXPECT(mism_rows == 0 || worst_gap < 1e-5, "index mismatches that are not boundary ties");

  // (1) attention vs HF fp32 (with HF's own indices, then with ours)
  {
    int32_t* idx_hf = up(r.idx);
    Cache c2;
    c2.alloc(cap);
    std::vector<int32_t> tmp;
    std::vector<uint16_t> out_hfidx;
    run_oneshot(d, w, c2, S, tmp, out_hfidx, idx_hf);
    c2.free_();
    cudaFree(idx_hf);
    const double e_all = rel_err(out_hfidx, r.out_f32, W, 0, S);
    const double e_dense = rel_err(out_hfidx, r.out_f32, W, 0, t_sparse);
    const double e_sparse = t_sparse < S ? rel_err(out_hfidx, r.out_f32, W, t_sparse, S) : 0.0;
    const double e_ours = rel_err(out1, r.out_f32, W, 0, S);
    // information: how far is HF's own bf16 eager path from its fp32 path
    double num = 0, den = 0;
    for (size_t i = 0; i < r.out_f32.size(); ++i) {
      const double dd = (double)r.out_bf16[i] - r.out_f32[i];
      num += dd * dd;
      den += (double)r.out_f32[i] * r.out_f32[i];
    }
    // floor: the fp32 reference itself rounded to bf16 (our output is bf16)
    std::vector<uint16_t> rb(r.out_f32.size());
    for (size_t i = 0; i < rb.size(); ++i) rb[i] = f2b(r.out_f32[i]);
    const double floor_ = rel_err(rb, r.out_f32, W, 0, S);
    printf("  (1) attention rel vs HF fp32: all %.3e  dense rows %.3e  sparse rows %.3e | with our idx %.3e | (bf16 rounding floor "
           "%.3e; HF bf16 eager vs HF fp32: %.3e)\n",
           e_all, e_dense, e_sparse, e_ours, floor_, std::sqrt(num / den));
    EXPECT(e_all < 3e-3 && e_dense < 3e-3 && e_sparse < 3e-3, "attention vs HF rel too large");
    EXPECT(e_ours < 3e-3, "attention with our indices vs HF rel too large");
  }

  // (3) incremental appends: pieces and token-by-token give the same caches, indices and outputs
  for (int mode = 0; mode < 2; ++mode) {
    Cache c3;
    c3.alloc(cap);
    int32_t* idx = dev<int32_t>((size_t)S * kDsaIdxWidth);
    bf16* out = dev<bf16>((size_t)S * W);
    const size_t sws = dsa_select_ws_bytes(S, S), aws = dsa_attention_ws_bytes(S);
    void* ws = dev<char>(std::max(sws, aws));
    const int pieces[] = {1, 3, 2, 7, 100, 513, 1, 1, 2047, 6, 999, 5};
    int pos = 0, k = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (pos < S) {
      int T = mode == 0 ? std::min(pieces[k++ % 12], S - pos) : 1;
      dsa_append(w, c3.c, pos, T, d.c + (size_t)pos * kDsaLat, d.kI + (size_t)pos * 128, d.gs + (size_t)pos * 128, 0);
      dsa_select(c3.c, pos, T, d.qI + (size_t)pos * 32 * 128, d.wI + (size_t)pos * 32, idx + (size_t)pos * kDsaIdxWidth, ws,
                 dsa_select_ws_bytes(T, pos + T), 0);
      dsa_attention(w, c3.c, pos, T, d.q + (size_t)pos * W, idx + (size_t)pos * kDsaIdxWidth, out + (size_t)pos * W, ws,
                    dsa_attention_ws_bytes(T), 0);
      pos += T;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto eq_bytes = [&](const bf16* a, const bf16* b, size_t n) {
      auto x = down(reinterpret_cast<const uint16_t*>(a), n), y = down(reinterpret_cast<const uint16_t*>(b), n);
      return x == y;
    };
    const bool caches_eq = eq_bytes(c3.c.c, c1.c.c, (size_t)S * kDsaLat) && eq_bytes(c3.c.kI, c1.c.kI, (size_t)S * 128) &&
                           eq_bytes(c3.c.gs, c1.c.gs, (size_t)S * 128) &&
                           eq_bytes(c3.c.pooled, c1.c.pooled, (size_t)S / 4 * 128);
    auto idx3 = down(idx, (size_t)S * kDsaIdxWidth);
    auto out3 = down(reinterpret_cast<uint16_t*>(out), (size_t)S * W);
    const bool idx_eq = idx3 == idx1;
    const double e = rel_err_bb(out3, out1, out3.size());
    printf("  (3) %s: caches %s, idx %s, out rel vs one-shot %.2e (%.1f s)\n", mode == 0 ? "pieces" : "token-by-token",
           caches_eq ? "identical" : "DIFFER", idx_eq ? "identical" : "DIFFER", e, secs);
    EXPECT(caches_eq && idx_eq && e < 1e-3, "incremental append mismatch");

    // (4) batched decode M=4 vs single rows (decode step of the last token of 4 prefixes of the same sequence)
    if (mode == 0 && S > 64) {
      const int L[4] = {S / 4 + 1, S / 2 + 2, S - 2, S};  // lengths after the decode step
      Cache cs[4];
      std::vector<DsaDecodeRow> rows_h(4);
      for (int i = 0; i < 4; ++i) {
        cs[i].alloc(cap);
        const int P = L[i] - 1;  // prefill P tokens, then decode token P
        dsa_append(w, cs[i].c, 0, P, d.c, d.kI, d.gs, 0);
        rows_h[i] = DsaDecodeRow{cs[i].c.c, cs[i].c.kI, cs[i].c.gs, cs[i].c.pooled, P};
      }
      DsaDecodeRow* rows = up(rows_h);
      // gather the 4 decode inputs
      bf16 *cn = dev<bf16>(4 * kDsaLat), *kn = dev<bf16>(4 * 128), *gn = dev<bf16>(4 * 128), *qIn = dev<bf16>(4 * 32 * 128),
           *qn = dev<bf16>(4 * W);
      float* wn = dev<float>(4 * 32);
      for (int i = 0; i < 4; ++i) {
        const int p = L[i] - 1;
        CUDA_CHECK(cudaMemcpy(cn + i * kDsaLat, d.c + (size_t)p * kDsaLat, kDsaLat * 2, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(kn + i * 128, d.kI + (size_t)p * 128, 256, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(gn + i * 128, d.gs + (size_t)p * 128, 256, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(qIn + i * 4096, d.qI + (size_t)p * 4096, 8192, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(wn + i * 32, d.wI + (size_t)p * 32, 128, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(qn + i * W, d.q + (size_t)p * W, W * 2, cudaMemcpyDeviceToDevice));
      }
      int32_t* didx = dev<int32_t>(4 * kDsaIdxWidth);
      bf16* dout = dev<bf16>(4 * W);
      const size_t ws1 = std::max(dsa_select_decode_ws_bytes(4, cap), dsa_attention_decode_ws_bytes(4));
      void* dws = dev<char>(ws1);
      dsa_append_decode(w, rows, 4, cn, kn, gn, 0);
      dsa_select_decode(rows, 4, cap, qIn, wn, didx, dws, ws1, 0);
      dsa_attention_decode(w, rows, 4, qn, didx, dout, dws, ws1, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      auto bidx = down(didx, 4 * kDsaIdxWidth);
      auto bout = down(reinterpret_cast<uint16_t*>(dout), 4 * W);
      // single-row decode of each (M=1) on the same caches (re-append is idempotent)
      bool idx_ok = true;
      double e_single = 0, e_prefill = 0;
      for (int i = 0; i < 4; ++i) {
        dsa_append_decode(w, rows + i, 1, cn + i * kDsaLat, kn + i * 128, gn + i * 128, 0);
        dsa_select_decode(rows + i, 1, cap, qIn + i * 4096, wn + i * 32, didx, dws, ws1, 0);
        dsa_attention_decode(w, rows + i, 1, qn + i * W, didx, dout, dws, ws1, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto sidx = down(didx, kDsaIdxWidth);
        auto sout = down(reinterpret_cast<uint16_t*>(dout), W);
        idx_ok = idx_ok && std::equal(sidx.begin(), sidx.end(), bidx.begin() + i * kDsaIdxWidth);
        idx_ok = idx_ok && std::equal(sidx.begin(), sidx.end(), idx1.begin() + (size_t)(L[i] - 1) * kDsaIdxWidth);
        std::vector<uint16_t> brow(bout.begin() + i * W, bout.begin() + (i + 1) * W);
        std::vector<uint16_t> prow(out1.begin() + (size_t)(L[i] - 1) * W, out1.begin() + (size_t)L[i] * W);
        e_single = std::max(e_single, rel_err_bb(brow, sout, W));
        e_prefill = std::max(e_prefill, rel_err_bb(brow, prow, W));
      }
      printf("  (4) decode M=4 (lengths %d %d %d %d): idx %s; out rel batched vs single %.2e, vs prefill row %.2e\n", L[0], L[1],
             L[2], L[3], idx_ok ? "identical (batched = single = prefill)" : "DIFFER", e_single, e_prefill);
      EXPECT(idx_ok && e_single < 1e-3 && e_prefill < 1e-3, "batched decode mismatch");
      for (auto& x : cs) x.free_();
      cudaFree(rows); cudaFree(cn); cudaFree(kn); cudaFree(gn); cudaFree(qIn); cudaFree(qn); cudaFree(wn);
      cudaFree(didx); cudaFree(dout); cudaFree(dws);
    }
    c3.free_();
    cudaFree(idx); cudaFree(out); cudaFree(ws);
  }

  // (5) negative controls — each must break the comparison
  {
    // wrong attention scaling: q·2 ≡ scaling 2/16 (exact in bf16)
    std::vector<uint16_t> q2(r.q.size());
    for (size_t i = 0; i < q2.size(); ++i) q2[i] = f2b(2.f * b2f(r.q[i]));
    bf16* dq2 = reinterpret_cast<bf16*>(up(q2));
    int32_t* idx_hf = up(r.idx);
    Cache cn;
    cn.alloc(cap);
    std::vector<int32_t> tmp;
    std::vector<uint16_t> o;
    run_oneshot(d, w, cn, S, tmp, o, idx_hf, dq2);
    const double e_scale = rel_err(o, r.out_f32, W, 0, S);
    cn.free_();
    cudaFree(dq2);
    // dropped tail rule: blank the 3 tail slots of HF's indices
    std::vector<int32_t> notail = r.idx;
    for (int t = 0; t < S; ++t)
      for (int e = kDsaTopk; e < kDsaIdxWidth; ++e) notail[(size_t)t * kDsaIdxWidth + e] = -1;
    // in HF's layout the tail sits right after the selected pools when fewer than 512 pools exist: remove tail tokens by value
    for (int t = 0; t < S; ++t) {
      const int tail0 = (t + 1) / 4 * 4;
      for (int e = 0; e < kDsaIdxWidth; ++e)
        if (notail[(size_t)t * kDsaIdxWidth + e] >= tail0) notail[(size_t)t * kDsaIdxWidth + e] = -1;
    }
    int32_t* idx_nt = up(notail);
    cn.alloc(cap);
    run_oneshot(d, w, cn, S, tmp, o, idx_nt);
    const double e_tail = rel_err(o, r.out_f32, W, 0, S);
    cn.free_();
    cudaFree(idx_nt);
    // ignored ape: pooled keys without the positional bias → selection
    DsaLayerW w0{d.kv_b, d.ape0};
    cn.alloc(cap);
    run_oneshot(d, w0, cn, S, tmp, o);
    int bad_rows = 0;
    for (int t = 0; t < S; ++t)
      if (row_set(&tmp[(size_t)t * kDsaIdxWidth]) != row_set(&r.idx[(size_t)t * kDsaIdxWidth])) ++bad_rows;
    const double e_ape = rel_err(o, r.out_f32, W, 0, S);
    cn.free_();
    cudaFree(idx_hf);
    printf("  (5) negative controls: wrong scaling rel %.3e | no tail rule rel %.3e | no ape: %d idx rows differ, rel %.3e\n",
           e_scale, e_tail, bad_rows, e_ape);
    EXPECT(e_scale > 3e-2, "wrong-scaling control did not fail");
    EXPECT(e_tail > 3e-3, "no-tail control did not fail");
    if (S > kDsaIdxWidth) EXPECT(bad_rows > (S - kDsaIdxWidth) / 2, "no-ape control did not change the selection");
  }
  c1.free_();
  cudaFree(d.kv_b); cudaFree(d.ape); cudaFree(d.ape0); cudaFree(d.q); cudaFree(d.c); cudaFree(d.qI); cudaFree(d.kI);
  cudaFree(d.gs); cudaFree(d.wI);
}

// ------------------------------------------------------------------ bench ------------------------------------------
__global__ void fill_rand(bf16* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = __float2bfloat16(((x & 0xFFFF) / 65536.f - 0.5f) * 3.4641f * scale);  // ~unit variance
  }
}
__global__ void fill_randf(float* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = ((x & 0xFFFF) / 65536.f - 0.5f) * 3.4641f;
  }
}
void rnd(bf16* p, size_t n, uint32_t seed, float scale = 1.f) { fill_rand<<<1024, 256>>>(p, n, seed, scale); }

template <class F>
float time_ms(F&& f, int reps) {
  f();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < reps; ++i) f();
  cudaEventRecord(b);
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / reps;
}

void bench() {
  printf("== bench (random data)\n");
  const size_t W = (size_t)kDsaHeads * kDsaVDim;
  bf16* kv_b = dev<bf16>((size_t)kDsaHeads * 512 * kDsaLat);
  bf16* ape = dev<bf16>(512);
  rnd(kv_b, (size_t)kDsaHeads * 512 * kDsaLat, 1, 0.044f);
  rnd(ape, 512, 2, 0.7f);
  DsaLayerW w{kv_b, ape};
  // ---- prefill: T new tokens after `ctx` cached tokens
  {
    const int Tmax = 16384, cap = 65536 + Tmax;
    Cache cc;
    cc.alloc(cap);
    rnd(cc.c.c, (size_t)cap * kDsaLat, 3);
    rnd(cc.c.kI, (size_t)cap * 128, 4);
    rnd(cc.c.gs, (size_t)cap * 128, 5);
    bf16 *cn = dev<bf16>((size_t)Tmax * kDsaLat), *kn = dev<bf16>((size_t)Tmax * 128), *gn = dev<bf16>((size_t)Tmax * 128);
    bf16* qI = dev<bf16>((size_t)Tmax * 4096);
    float* wI = dev<float>((size_t)Tmax * 32);
    bf16* q = dev<bf16>((size_t)Tmax * W);  // also the output (in place, see the chunk loop)
    int32_t* idx = dev<int32_t>((size_t)Tmax * kDsaIdxWidth);
    rnd(cn, (size_t)Tmax * kDsaLat, 6); rnd(kn, (size_t)Tmax * 128, 7); rnd(gn, (size_t)Tmax * 128, 8);
    rnd(qI, (size_t)Tmax * 4096, 9); rnd(q, (size_t)Tmax * W, 10, 2.f);
    fill_randf<<<256, 256>>>(wI, (size_t)Tmax * 32, 11);
    const size_t sws = dsa_select_ws_bytes(Tmax, cap), aws = dsa_attention_ws_bytes(Tmax);
    void* ws = dev<char>(std::max(sws, aws));
    // pooled keys of the whole buffer (pretend prefix)
    dsa_append(w, cc.c, 0, cap, cc.c.c, cc.c.kI, cc.c.gs, 0);  // self-copy: builds every pool
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t fr, tot;
    cudaMemGetInfo(&fr, &tot);
    printf("  prefill buffers allocated (free VRAM now %.2f GiB)\n", fr / 1073741824.0);
    printf("  %-7s %-7s %10s %10s %12s %10s\n", "T", "ctx0", "append ms", "select ms", "attention ms", "total ms");
    for (int T : {1, 2048, 16384})
      for (int ctx : {0, 4096, 16384, 65536}) {
        const int reps = T == 16384 ? 2 : (T == 2048 ? 5 : 50);
        float ta = time_ms([&] { dsa_append(w, cc.c, ctx, T, cn, kn, gn, 0); }, reps);
        float tsel = time_ms([&] { dsa_select(cc.c, ctx, T, qI, wI, idx, ws, sws, 0); }, reps);
        float tat = time_ms([&] { dsa_attention(w, cc.c, ctx, T, q, idx, q, ws, aws, 0); }, reps);
        printf("  %-7d %-7d %10.3f %10.3f %12.3f %10.3f\n", T, ctx, ta, tsel, tat, ta + tsel + tat);
      }
    cc.free_();
    cudaFree(cn); cudaFree(kn); cudaFree(gn); cudaFree(qI); cudaFree(wI); cudaFree(q); cudaFree(idx); cudaFree(ws);
  }
  // ---- decode: M rows, one token each, at context 32K / 128K
  for (int ctx : {32768, 131072}) {
    const int cap = ctx + 4;
    const size_t per = (size_t)cap * (kDsaLat + 256) * 2 + (size_t)cap / 4 * 256;
    const int ndistinct = (int)std::min<size_t>(8, std::max<size_t>(1, (700ull << 20) / per));
    std::vector<Cache> cs(ndistinct);
    for (int i = 0; i < ndistinct; ++i) {
      cs[i].alloc(cap);
      rnd(cs[i].c.c, (size_t)cap * kDsaLat, 20 + i);
      rnd(cs[i].c.kI, (size_t)cap * 128, 40 + i);
      rnd(cs[i].c.gs, (size_t)cap * 128, 60 + i);
      dsa_append(w, cs[i].c, 0, cap, cs[i].c.c, cs[i].c.kI, cs[i].c.gs, 0);
    }
    bf16 *cn = dev<bf16>(8 * kDsaLat), *kn = dev<bf16>(8 * 128), *gn = dev<bf16>(8 * 128), *qI = dev<bf16>(8 * 4096),
         *q = dev<bf16>(8 * W), *out = dev<bf16>(8 * W);
    float* wI = dev<float>(8 * 32);
    int32_t* idx = dev<int32_t>(8 * kDsaIdxWidth);
    rnd(cn, 8 * kDsaLat, 71); rnd(kn, 8 * 128, 72); rnd(gn, 8 * 128, 73); rnd(qI, 8 * 4096, 74); rnd(q, 8 * W, 75, 2.f);
    fill_randf<<<8, 256>>>(wI, 8 * 32, 76);
    const size_t ws_b = std::max(dsa_select_decode_ws_bytes(8, cap), dsa_attention_decode_ws_bytes(8));
    void* ws = dev<char>(ws_b);
    for (int M : {1, 4, 8}) {
      std::vector<DsaDecodeRow> rh(M);
      for (int i = 0; i < M; ++i) {
        auto& c = cs[i % ndistinct].c;
        rh[i] = DsaDecodeRow{c.c, c.kI, c.gs, c.pooled, ctx - 1};
      }
      DsaDecodeRow* rows = up(rh);
      const int reps = 200;
      float ta = time_ms([&] { dsa_append_decode(w, rows, M, cn, kn, gn, 0); }, reps);
      float ts = time_ms([&] { dsa_select_decode(rows, M, cap, qI, wI, idx, ws, ws_b, 0); }, reps);
      float tt = time_ms([&] { dsa_attention_decode(w, rows, M, q, idx, out, ws, ws_b, 0); }, reps);
      printf("  decode ctx %6d M=%d (%d distinct caches): append %.3f  select %.3f  attention %.3f  total %.3f ms\n", ctx, M,
             std::min(M, ndistinct), ta, ts, tt, ta + ts + tt);
      cudaFree(rows);
    }
    for (auto& c : cs) c.free_();
    cudaFree(cn); cudaFree(kn); cudaFree(gn); cudaFree(qI); cudaFree(q); cudaFree(out); cudaFree(wI); cudaFree(idx); cudaFree(ws);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") {
    bench();
    return 0;
  }
  std::string root = argc > 1 ? argv[1] : "/tmp/dsa_ref";
  std::vector<int> Ss;
  for (int i = 2; i < argc; ++i) Ss.push_back(atoi(argv[i]));
  if (Ss.empty()) Ss = {1500, 2600, 6000};
  for (int S : Ss) test_ref(root + "/S" + std::to_string(S), S);
  printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
