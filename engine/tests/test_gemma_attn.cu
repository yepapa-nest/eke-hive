// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma paged fp8 attention (hive/gemma/gemma_kernels.h): kv_write against a host e4m3 encoder, attn_paged against an fp64 host
//   reference over the same fp8 K / V, on page tables with shuffled pages, some pages in mapped pinned host memory, NaN-filled unwritten
//   slots and null (released) sliding pages; decode / verify / prefill rows mixed in one call; determinism (a sequence alone vs. inside the
//   mixed call, and a decode row vs. the same position inside a verify call → bitwise identical); image spans (AttnSeq::qlim: the rows of
//   a span see each other both ways) on the prefill and the split path; and timings at the model's shapes.
//   Run: scripts/hive-run.sh "./build-svc/test_gemma_attn"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/devbuf.h"
#include "hive/gemma/gemma_kernels.h"

using namespace hive;
using namespace hive::gemma::k;

static uint16_t h_f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
static float h_bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static double e4m3_calc(uint8_t v) {
  const int s = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
  const double r = e == 0 ? std::ldexp((double)m, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
  return s ? -r : r;
}
static const std::vector<double> kE4m3 = [] { std::vector<double> t(256); for (int i = 0; i < 256; ++i) t[i] = e4m3_calc((uint8_t)i); return t; }();
static inline double e4m3_val(uint8_t v) { return kE4m3[v]; }
// round-to-nearest-even, saturating to ±448 (codes 0x00..0x7E are the finite magnitudes, increasing)
static uint8_t e4m3_enc(float x) {
  const uint8_t sign = std::signbit(x) ? 0x80 : 0;
  const double a = std::fabs((double)x);
  if (a >= 448.0) return sign | 0x7E;
  int lo = 0, hi = 0x7E;
  while (hi - lo > 1) { const int mid = (lo + hi) / 2; if (e4m3_val((uint8_t)mid) <= a) lo = mid; else hi = mid; }
  const double dl = a - e4m3_val((uint8_t)lo), dh = e4m3_val((uint8_t)hi) - a;
  const int code = dl < dh ? lo : dh < dl ? hi : ((lo & 1) ? hi : lo);
  return sign | (uint8_t)code;
}

static bool check(bool ok, const char* what) { printf("%-100s %s\n", what, ok ? "PASS" : "FAIL"); return ok; }

// Page pool: device pages and mapped pinned host pages, handed out in shuffled order.
struct Pool {
  size_t page_bytes = 0;
  DevBuf dev;
  uint8_t* host = nullptr;
  std::vector<uint8_t*> free_pages;
  Pool(size_t pb, int n_dev, int n_host, std::mt19937& rng) : page_bytes(pb) {
    dev.alloc(pb * n_dev);
    CUDA_CHECK(cudaMemset(dev.p, 0x7F, pb * n_dev));   // e4m3 NaN everywhere nothing was written
    CUDA_CHECK(cudaHostAlloc((void**)&host, pb * n_host, cudaHostAllocMapped));
    memset(host, 0x7F, pb * n_host);
    uint8_t* hd = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&hd, host, 0));
    for (int i = 0; i < n_dev; ++i) free_pages.push_back(static_cast<uint8_t*>(dev.p) + pb * i);
    for (int i = 0; i < n_host; ++i) free_pages.push_back(hd + pb * i);
    std::shuffle(free_pages.begin(), free_pages.end(), rng);
  }
  ~Pool() { cudaFreeHost(host); }
  uint8_t* take() { HIVE_CHECK(!free_pages.empty(), "test pool exhausted"); uint8_t* p = free_pages.back(); free_pages.pop_back(); return p; }
};

struct HostSeq {
  int n_q, q_pos0, q_row0 = 0;
  std::vector<int32_t> lim;       // last visible key per row (empty = causal)
  DevBuf dlim;
  std::vector<uint8_t*> table;    // page pointers (nullptr = released)
  DevBuf dtable;
  std::vector<uint8_t> kc, vc;    // fp8 codes [pos][Hkv][D]
};

struct Case {
  const char* name;
  int D, Hkv, H, window, n_layers, layer;
};

// fp64 reference of one (row, head)
// out = Σ p v, mass = Σ p |v| (the scale of the rounding errors: P is rounded relative to the row max, the output to bf16)
static void reference(const Case& cs, const HostSeq& s, const std::vector<uint16_t>& q, int qi, int h, std::vector<double>& out,
                      std::vector<double>& mass) {
  const int D = cs.D, g = h / (cs.H / cs.Hkv), pos = s.q_pos0 + qi, hi = s.lim.empty() ? pos : s.lim[qi];
  const int j0 = cs.window > 0 ? std::max(0, pos - cs.window + 1) : 0;
  const uint16_t* qr = q.data() + ((size_t)(s.q_row0 + qi) * cs.H + h) * D;
  std::vector<double> qd(D), sc(hi - j0 + 1);
  for (int d = 0; d < D; ++d) qd[d] = h_bf2f(qr[d]);
  double mx = -1e300;
  for (int j = j0; j <= hi; ++j) {
    const uint8_t* kr = s.kc.data() + ((size_t)j * cs.Hkv + g) * D;
    double a = 0;
    for (int d = 0; d < D; ++d) a += qd[d] * e4m3_val(kr[d]);
    sc[j - j0] = a;
    mx = std::max(mx, a);
  }
  double L = 0;
  for (auto& v : sc) { v = std::exp(v - mx); L += v; }
  out.assign(D, 0.0);
  mass.assign(D, 0.0);
  for (int j = j0; j <= hi; ++j) {
    const uint8_t* vr = s.vc.data() + ((size_t)j * cs.Hkv + g) * D;
    const double p = sc[j - j0] / L;
    for (int d = 0; d < D; ++d) { const double v = e4m3_val(vr[d]); out[d] += p * v; mass[d] += p * std::fabs(v); }
  }
}

static bool run_case(const Case& cs, std::mt19937& rng) {
  bool all = true;
  char b[256];
  KvLayout L;
  L.n_layers = cs.n_layers; L.Hkv = cs.Hkv; L.D = cs.D;
  const int D = cs.D, Hkv = cs.Hkv, H = cs.H;
  // sequences: {n_q, q_pos0, image span rows [a, b)}; the last two share one KV (decode at p vs. verify ending at p)
  struct Spec { int n_q, q_pos0, a = 0, b = 0; };
  std::vector<Spec> specs = {{1, 0}, {1, 1500}, {4, 1101}, {16, 2047, 3, 12}, {300, 900, 18, 298}, {37, 0, 5, 14}, {2048, 2048}, {1, 1234}, {4, 1231}};
  const int n = (int)specs.size();
  size_t need = 0;
  for (auto& sp : specs) need += (sp.q_pos0 + sp.n_q + kPage - 1) / kPage;
  Pool pool(L.page_bytes(), (int)(need * 3 / 4) + 4, (int)(need / 4) + 4, rng);
  std::vector<HostSeq> seqs(n);
  std::normal_distribution<float> nd(0.f, 1.f);
  int rows = 0;
  for (int i = 0; i < n; ++i) {
    HostSeq& s = seqs[i];
    s.n_q = specs[i].n_q; s.q_pos0 = specs[i].q_pos0; s.q_row0 = rows; rows += s.n_q;
    if (specs[i].b > specs[i].a) {
      for (int t = 0; t < s.n_q; ++t) s.lim.push_back(s.q_pos0 + (t >= specs[i].a && t < specs[i].b ? specs[i].b - 1 : t));
      s.dlim.alloc(s.lim.size() * 4);
      CUDA_CHECK(cudaMemcpy(s.dlim.p, s.lim.data(), s.lim.size() * 4, cudaMemcpyHostToDevice));
    }
    const bool shared = i == n - 1;   // shares the previous sequence's pages and KV
    const int kv_len = s.q_pos0 + s.n_q;
    if (shared) { s.table = seqs[i - 1].table; s.kc = seqs[i - 1].kc; s.vc = seqs[i - 1].vc; }
    else {
      const int np = (kv_len + kPage - 1) / kPage;
      const int first_vis = cs.window > 0 ? std::max(0, s.q_pos0 - cs.window + 1) : 0;
      s.table.assign(np, nullptr);
      for (int p = 0; p < np; ++p) if ((p + 1) * kPage > first_vis) s.table[p] = pool.take();
      s.kc.resize((size_t)kv_len * Hkv * D);
      s.vc.resize((size_t)kv_len * Hkv * D);
      for (size_t x = 0; x < s.kc.size(); ++x) s.kc[x] = e4m3_enc(h_bf2f(h_f2bf(nd(rng))));
      for (size_t x = 0; x < s.vc.size(); ++x) s.vc[x] = e4m3_enc(h_bf2f(h_f2bf(nd(rng))));
      // prefix positions (< q_pos0) go straight into the pages; the new rows are written by kv_write below
      std::vector<uint8_t> img(L.page_bytes());
      for (int p = 0; p < np; ++p) {
        if (!s.table[p]) continue;
        memset(img.data(), 0x7F, img.size());
        for (int sl = 0; sl < kPage; ++sl) {
          const int pos = p * kPage + sl;
          if (pos >= s.q_pos0) break;
          memcpy(img.data() + L.k_off(sl, cs.layer), s.kc.data() + (size_t)pos * Hkv * D, (size_t)Hkv * D);
          memcpy(img.data() + L.v_off(sl, cs.layer), s.vc.data() + (size_t)pos * Hkv * D, (size_t)Hkv * D);
        }
        CUDA_CHECK(cudaMemcpy(s.table[p], img.data(), img.size(), cudaMemcpyDefault));
      }
    }
    s.dtable.alloc(s.table.size() * sizeof(uint8_t*));
    CUDA_CHECK(cudaMemcpy(s.dtable.p, s.table.data(), s.table.size() * sizeof(uint8_t*), cudaMemcpyHostToDevice));
  }
  std::vector<AttnSeq> as(n);
  for (int i = 0; i < n; ++i) {
    as[i].pages = seqs[i].dtable.as<const uint8_t*>(); as[i].q_row0 = seqs[i].q_row0; as[i].n_q = seqs[i].n_q; as[i].q_pos0 = seqs[i].q_pos0;
    if (!seqs[i].lim.empty()) { as[i].qlim = seqs[i].dlim.as<int32_t>(); as[i].kmax = *std::max_element(seqs[i].lim.begin(), seqs[i].lim.end()); }
  }
  DevBuf das(n * sizeof(AttnSeq));
  CUDA_CHECK(cudaMemcpy(das.p, as.data(), n * sizeof(AttnSeq), cudaMemcpyHostToDevice));

  // ---- kv_write of every new row (bf16 values whose e4m3 rounding are the codes above, plus exact bf16 → e4m3 rounding checks)
  {
    std::vector<uint16_t> kb, vb;
    std::vector<int32_t> so, po;
    std::vector<uint8_t> kexp, vexp;
    for (int i = 0; i < n - 1; ++i) {
      HostSeq& s = seqs[i];
      for (int t = 0; t < s.n_q; ++t) {
        const int pos = s.q_pos0 + t;
        so.push_back(i); po.push_back(pos);
        for (int x = 0; x < Hkv * D; ++x) {
          // random bf16 values (some beyond 448 to exercise saturation); the reference codes become their rounding
          const float scale = (x % 97 == 0) ? 300.f : 1.f;
          const uint16_t kv = h_f2bf(nd(rng) * scale), vv = h_f2bf(nd(rng) * scale);
          kb.push_back(kv); vb.push_back(vv);
          s.kc[(size_t)pos * Hkv * D + x] = e4m3_enc(h_bf2f(kv));
          s.vc[(size_t)pos * Hkv * D + x] = e4m3_enc(h_bf2f(vv));
        }
      }
    }
    seqs[n - 1].kc = seqs[n - 2].kc; seqs[n - 1].vc = seqs[n - 2].vc;
    const int wr = (int)so.size();
    DevBuf dk(kb.size() * 2), dv(vb.size() * 2), dso(wr * 4), dpo(wr * 4);
    CUDA_CHECK(cudaMemcpy(dk.p, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.p, vb.data(), vb.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dso.p, so.data(), wr * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpo.p, po.data(), wr * 4, cudaMemcpyHostToDevice));
    kv_write(dk.as<bf16>(), dv.as<bf16>(), wr, dso.as<int32_t>(), dpo.as<int32_t>(), das.as<AttnSeq>(), L, cs.layer, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    long bad = 0;
    std::vector<uint8_t> got((size_t)Hkv * D);
    for (int r = 0; r < wr; ++r) {
      const HostSeq& s = seqs[so[r]];
      const int pos = po[r];
      for (int kvsel = 0; kvsel < 2; ++kvsel) {
        const uint8_t* src = s.table[pos / kPage] + (kvsel ? L.v_off(pos % kPage, cs.layer) : L.k_off(pos % kPage, cs.layer));
        CUDA_CHECK(cudaMemcpy(got.data(), src, got.size(), cudaMemcpyDefault));
        const uint8_t* ex = (kvsel ? s.vc.data() : s.kc.data()) + (size_t)pos * Hkv * D;
        for (int x = 0; x < Hkv * D; ++x) bad += got[x] != ex[x];
      }
    }
    snprintf(b, sizeof b, "%s: kv_write == host e4m3 RNE satfinite (%d rows, layer %d of %d): %ld byte mismatches", cs.name, wr, cs.layer,
             cs.n_layers, bad);
    all &= check(bad == 0, b);
  }

  // ---- attention, mixed call
  std::vector<uint16_t> qh((size_t)rows * H * D);
  const float qs = 3.f / std::sqrt((float)D);   // score spread ~3 (natural-log units)
  for (auto& v : qh) v = h_f2bf(nd(rng) * qs);
  DevBuf dq(qh.size() * 2), dout(qh.size() * 2), dout1(qh.size() * 2);
  CUDA_CHECK(cudaMemcpy(dq.p, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dout.p, 0xFF, dout.n));
  int maxq = 0;
  for (auto& s : seqs) maxq = std::max(maxq, s.n_q);
  attn_paged(dq.as<bf16>(), dout.as<bf16>(), H, L, cs.layer, das.as<AttnSeq>(), n, maxq, cs.window, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> oh(qh.size());
  CUDA_CHECK(cudaMemcpy(oh.data(), dout.p, oh.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<double> ref, mass;
  for (int i = 0; i < n; ++i) {
    const HostSeq& s = seqs[i];
    const int stride = s.n_q <= 64 ? 1 : s.n_q / 40;
    double maxerr = 0, maxrel = 0;
    long bad = 0, checked = 0;
    for (int qi = 0; qi < s.n_q; qi += stride) {
      for (int pass = 0; pass < 2; ++pass) {
        const int row = pass ? s.n_q - 1 : qi;
        if (pass && qi + stride < s.n_q) break;   // also the last row
        for (int h = 0; h < H; ++h) {
          reference(cs, s, qh, row, h, ref, mass);
          const uint16_t* got = oh.data() + ((size_t)(s.q_row0 + row) * H + h) * D;
          for (int d = 0; d < D; ++d) {
            const double gv = h_bf2f(got[d]), e = std::fabs(gv - ref[d]);
            // bf16 output rounding ≤ 2^-9 |ref|, f16 P rounding ≤ 2^-11 Σ p |v| (+ fp32 noise)
            const double tol = 1e-5 + 2.5e-3 * std::fabs(ref[d]) + 2.5e-3 * mass[d];
            maxerr = std::max(maxerr, e);
            maxrel = std::max(maxrel, e / tol);
            bad += !(e <= tol);
            ++checked;
          }
        }
      }
    }
    snprintf(b, sizeof b, "%s: seq %d (n_q %4d, q_pos0 %4d%s): vs fp64 reference, max |err| %.2e, max err / tol %.2f, %ld / %ld out",
             cs.name, i, s.n_q, s.q_pos0, s.lim.empty() ? "" : ", image span", maxerr, maxrel, bad, checked);
    all &= check(bad == 0, b);
  }
  // ---- determinism: each sequence alone (rows at the same place) → bitwise identical rows
  {
    bool same = true;
    for (int i = 0; i < n; ++i) {
      CUDA_CHECK(cudaMemset(dout1.p, 0, dout1.n));
      attn_paged(dq.as<bf16>(), dout1.as<bf16>(), H, L, cs.layer, das.as<AttnSeq>() + i, 1, seqs[i].n_q, cs.window, nullptr);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<uint16_t> o1((size_t)seqs[i].n_q * H * D);
      CUDA_CHECK(cudaMemcpy(o1.data(), dout1.as<uint16_t>() + (size_t)seqs[i].q_row0 * H * D, o1.size() * 2, cudaMemcpyDeviceToHost));
      same &= memcmp(o1.data(), oh.data() + (size_t)seqs[i].q_row0 * H * D, o1.size() * 2) == 0;
    }
    snprintf(b, sizeof b, "%s: every sequence alone == the same sequence inside the mixed call (bitwise)", cs.name);
    all &= check(same, b);
    // decode at position 1234 (seq n-2, n_q 1) vs. the last verify row of seq n-1 (n_q 4 ending at 1234), same q
    CUDA_CHECK(cudaMemcpy(dq.as<uint16_t>() + (size_t)(seqs[n - 1].q_row0 + 3) * H * D, dq.as<uint16_t>() + (size_t)seqs[n - 2].q_row0 * H * D,
                          (size_t)H * D * 2, cudaMemcpyDeviceToDevice));
    attn_paged(dq.as<bf16>(), dout.as<bf16>(), H, L, cs.layer, das.as<AttnSeq>(), n, maxq, cs.window, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> a1((size_t)H * D), a4((size_t)H * D);
    CUDA_CHECK(cudaMemcpy(a1.data(), dout.as<uint16_t>() + (size_t)seqs[n - 2].q_row0 * H * D, a1.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a4.data(), dout.as<uint16_t>() + (size_t)(seqs[n - 1].q_row0 + 3) * H * D, a4.size() * 2, cudaMemcpyDeviceToHost));
    snprintf(b, sizeof b, "%s: decode row at position 1234 == verify (n_q 4) row at 1234 (bitwise)", cs.name);
    all &= check(memcmp(a1.data(), a4.data(), a1.size() * 2) == 0, b);
  }
  return all;
}

// ---- CUDA graph capture + replay: kv_write and attn_paged captured once (as the engine captures a decode / verify step), then positions,
//   page tables, q and K / V change in the same device buffers between replays; every replay is compared to the fp64 reference and to
//   a direct call (bitwise). Runs first in the process, so the kernels' first-use setup happens inside the capture.
static bool graph_case(const Case& cs, const std::vector<int>& nqs, std::mt19937& rng) {
  bool all = true;
  char b[256];
  KvLayout L;
  L.n_layers = cs.n_layers; L.Hkv = cs.Hkv; L.D = cs.D;
  const int D = cs.D, Hkv = cs.Hkv, H = cs.H, n = (int)nqs.size(), cap_pages = 48, cap = cap_pages * kPage;
  int maxq = 0, rows = 0;
  for (int v : nqs) { maxq = std::max(maxq, v); rows += v; }
  Pool pool(L.page_bytes(), n * cap_pages * 3 / 4 + 4, n * cap_pages / 4 + 4, rng);
  const std::vector<uint8_t*> all_pages = pool.free_pages;
  std::vector<HostSeq> seqs(n);
  std::vector<AttnSeq> as(n);
  for (int i = 0; i < n; ++i) {
    seqs[i].n_q = nqs[i];
    seqs[i].q_row0 = i ? seqs[i - 1].q_row0 + nqs[i - 1] : 0;
    seqs[i].dtable.alloc(cap_pages * sizeof(uint8_t*));
  }
  DevBuf das(n * sizeof(AttnSeq)), dq((size_t)rows * H * D * 2), dout(dq.n), dout2(dq.n), dk((size_t)rows * Hkv * D * 2), dv(dk.n),
      dso(rows * 4), dpo(rows * 4);
  std::normal_distribution<float> nd(0.f, 1.f);
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraphExec_t gexec = nullptr;
  cudaGraph_t graph = nullptr;
  std::vector<double> ref, mass;
  for (int round = 0; round < 4; ++round) {
    // new positions / page tables / contents in the same buffers
    std::vector<uint8_t*> pg = all_pages;
    std::shuffle(pg.begin(), pg.end(), rng);
    std::vector<int32_t> so, po;
    std::vector<uint16_t> kb, vb;
    for (int i = 0; i < n; ++i) {
      HostSeq& s = seqs[i];
      s.q_pos0 = (round == 0 && i == 0) ? 0 : (int)(rng() % (cap - s.n_q + 1));
      const int kv_len = s.q_pos0 + s.n_q, np = (kv_len + kPage - 1) / kPage;
      const int first_vis = cs.window > 0 ? std::max(0, s.q_pos0 - cs.window + 1) : 0;
      s.table.assign(cap_pages, nullptr);
      for (int p = 0; p < np; ++p) if ((p + 1) * kPage > first_vis) { s.table[p] = pg.back(); pg.pop_back(); }
      s.kc.assign((size_t)kv_len * Hkv * D, 0);
      s.vc.assign((size_t)kv_len * Hkv * D, 0);
      std::vector<uint8_t> img(L.page_bytes());
      for (int p = 0; p < np; ++p) {
        if (!s.table[p]) continue;
        memset(img.data(), 0x7F, img.size());
        for (int sl = 0; sl < kPage; ++sl) {
          const int pos = p * kPage + sl;
          if (pos >= s.q_pos0) break;
          for (int x = 0; x < Hkv * D; ++x) {
            s.kc[(size_t)pos * Hkv * D + x] = e4m3_enc(nd(rng));
            s.vc[(size_t)pos * Hkv * D + x] = e4m3_enc(nd(rng));
          }
          memcpy(img.data() + L.k_off(sl, cs.layer), s.kc.data() + (size_t)pos * Hkv * D, (size_t)Hkv * D);
          memcpy(img.data() + L.v_off(sl, cs.layer), s.vc.data() + (size_t)pos * Hkv * D, (size_t)Hkv * D);
        }
        CUDA_CHECK(cudaMemcpy(s.table[p], img.data(), img.size(), cudaMemcpyDefault));
      }
      for (int t = 0; t < s.n_q; ++t) {
        const int pos = s.q_pos0 + t;
        so.push_back(i); po.push_back(pos);
        for (int x = 0; x < Hkv * D; ++x) {
          const uint16_t kv = h_f2bf(nd(rng)), vv = h_f2bf(nd(rng));
          kb.push_back(kv); vb.push_back(vv);
          s.kc[(size_t)pos * Hkv * D + x] = e4m3_enc(h_bf2f(kv));
          s.vc[(size_t)pos * Hkv * D + x] = e4m3_enc(h_bf2f(vv));
        }
      }
      CUDA_CHECK(cudaMemcpy(s.dtable.p, s.table.data(), cap_pages * sizeof(uint8_t*), cudaMemcpyHostToDevice));
      as[i].pages = s.dtable.as<const uint8_t*>(); as[i].q_row0 = s.q_row0; as[i].n_q = s.n_q; as[i].q_pos0 = s.q_pos0;
    }
    std::vector<uint16_t> qh((size_t)rows * H * D);
    const float qs = 3.f / std::sqrt((float)D);
    for (auto& v : qh) v = h_f2bf(nd(rng) * qs);
    CUDA_CHECK(cudaMemcpy(das.p, as.data(), das.n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq.p, qh.data(), dq.n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk.p, kb.data(), dk.n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.p, vb.data(), dv.n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dso.p, so.data(), dso.n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpo.p, po.data(), dpo.n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout.p, 0xFF, dout.n));
    if (round == 0) {
      CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      kv_write(dk.as<bf16>(), dv.as<bf16>(), rows, dso.as<int32_t>(), dpo.as<int32_t>(), das.as<AttnSeq>(), L, cs.layer, st);
      attn_paged(dq.as<bf16>(), dout.as<bf16>(), H, L, cs.layer, das.as<AttnSeq>(), n, maxq, cs.window, st);
      CUDA_CHECK(cudaStreamEndCapture(st, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, 0));
    }
    CUDA_CHECK(cudaGraphLaunch(gexec, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<uint16_t> oh(qh.size()), oh2(qh.size());
    CUDA_CHECK(cudaMemcpy(oh.data(), dout.p, dout.n, cudaMemcpyDeviceToHost));
    attn_paged(dq.as<bf16>(), dout2.as<bf16>(), H, L, cs.layer, das.as<AttnSeq>(), n, maxq, cs.window, st);   // direct call
    CUDA_CHECK(cudaStreamSynchronize(st));
    CUDA_CHECK(cudaMemcpy(oh2.data(), dout2.p, dout2.n, cudaMemcpyDeviceToHost));
    long bad = 0, checked = 0;
    double maxrel = 0;
    for (int i = 0; i < n; ++i)
      for (int qi = 0; qi < seqs[i].n_q; ++qi)
        for (int h = 0; h < H; ++h) {
          reference(cs, seqs[i], qh, qi, h, ref, mass);
          const uint16_t* got = oh.data() + ((size_t)(seqs[i].q_row0 + qi) * H + h) * D;
          for (int d = 0; d < D; ++d) {
            const double e = std::fabs(h_bf2f(got[d]) - ref[d]), tol = 1e-5 + 2.5e-3 * std::fabs(ref[d]) + 2.5e-3 * mass[d];
            maxrel = std::max(maxrel, e / tol);
            bad += !(e <= tol);
            ++checked;
          }
        }
    snprintf(b, sizeof b, "%s graph (n_q %s, replay %d, positions %d/%d/%d): vs reference max err / tol %.2f, %ld / %ld out; == direct call",
             cs.name, maxq > 16 ? "prefill mix" : "1/4/3", round, seqs[0].q_pos0, seqs[1].q_pos0, seqs[2].q_pos0, maxrel, bad, checked);
    all &= check(bad == 0 && memcmp(oh.data(), oh2.data(), oh.size() * 2) == 0, b);
  }
  cudaGraphExecDestroy(gexec); cudaGraphDestroy(graph); cudaStreamDestroy(st);
  return all;
}

// ---- timing at the model's page layout (Gemma 4 12B: 40 sliding layers Hkv 8 D 256, 8 full layers Hkv 1 D 512)
static void timing(const Case& cs, int n_seqs, int n_q, int ctx, int reps) {
  KvLayout L;
  L.n_layers = cs.n_layers; L.Hkv = cs.Hkv; L.D = cs.D;
  const int q_pos0 = ctx - n_q, np = (ctx + kPage - 1) / kPage;
  const int first_vis = cs.window > 0 ? std::max(0, q_pos0 - cs.window + 1) : 0;
  int live = 0;
  for (int p = 0; p < np; ++p) live += (p + 1) * kPage > first_vis;
  DevBuf pages((size_t)n_seqs * live * L.page_bytes());
  CUDA_CHECK(cudaMemset(pages.p, 0x38, pages.n));   // e4m3 1.0
  std::vector<DevBuf> tables(n_seqs);
  std::vector<AttnSeq> as(n_seqs);
  int used = 0;
  for (int s = 0; s < n_seqs; ++s) {
    std::vector<const uint8_t*> t(np, nullptr);
    for (int p = 0; p < np; ++p)
      if ((p + 1) * kPage > first_vis) t[p] = static_cast<const uint8_t*>(pages.p) + (size_t)(used++) * L.page_bytes();
    tables[s].alloc(np * sizeof(void*));
    CUDA_CHECK(cudaMemcpy(tables[s].p, t.data(), np * sizeof(void*), cudaMemcpyHostToDevice));
    as[s].pages = tables[s].as<const uint8_t*>(); as[s].q_row0 = s * n_q; as[s].n_q = n_q; as[s].q_pos0 = q_pos0;
  }
  DevBuf das(n_seqs * sizeof(AttnSeq)), dq((size_t)n_seqs * n_q * cs.H * cs.D * 2), dout(dq.n);
  CUDA_CHECK(cudaMemcpy(das.p, as.data(), das.n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dq.p, 0x3C, dq.n));
  // every call takes the next layer of the type, so the KV streams from DRAM as in a decode step (one layer's KV alone stays in L2)
  for (int i = 0; i < 3; ++i) attn_paged(dq.as<bf16>(), dout.as<bf16>(), cs.H, L, i % L.n_layers, das.as<AttnSeq>(), n_seqs, n_q, cs.window, nullptr);
  // per-call intervals between back-to-back events; min ≈ the uncontended time (the GPU may be time-sliced with other processes)
  std::vector<cudaEvent_t> ev(reps + 1);
  for (auto& e : ev) cudaEventCreate(&e);
  cudaEventRecord(ev[0]);
  for (int i = 0; i < reps; ++i) {
    attn_paged(dq.as<bf16>(), dout.as<bf16>(), cs.H, L, i % L.n_layers, das.as<AttnSeq>(), n_seqs, n_q, cs.window, nullptr);
    cudaEventRecord(ev[i + 1]);
  }
  CUDA_CHECK(cudaEventSynchronize(ev[reps]));
  std::vector<double> t(reps);
  for (int i = 0; i < reps; ++i) { float ms = 0; cudaEventElapsedTime(&ms, ev[i], ev[i + 1]); t[i] = ms * 1e3; }
  for (auto& e : ev) cudaEventDestroy(e);
  std::sort(t.begin(), t.end());
  const double us_ev = t[0], med = t[reps / 2];
  // the same calls replayed from a CUDA graph (no host launch overhead — how the engine runs decode)
  cudaStream_t gs;
  CUDA_CHECK(cudaStreamCreateWithFlags(&gs, cudaStreamNonBlocking));
  const int G = 20;
  cudaGraph_t graph;
  cudaGraphExec_t gexec;
  CUDA_CHECK(cudaStreamBeginCapture(gs, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < G; ++i) attn_paged(dq.as<bf16>(), dout.as<bf16>(), cs.H, L, i % L.n_layers, das.as<AttnSeq>(), n_seqs, n_q, cs.window, gs);
  CUDA_CHECK(cudaStreamEndCapture(gs, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, 0));
  cudaEvent_t g0, g1;
  cudaEventCreate(&g0); cudaEventCreate(&g1);
  double us = 1e30;
  for (int r = 0; r < std::max(3, reps / G); ++r) {
    cudaEventRecord(g0, gs);
    CUDA_CHECK(cudaGraphLaunch(gexec, gs));
    cudaEventRecord(g1, gs);
    CUDA_CHECK(cudaEventSynchronize(g1));
    float ms = 0;
    cudaEventElapsedTime(&ms, g0, g1);
    us = std::min(us, (double)ms * 1e3 / G);
  }
  cudaEventDestroy(g0); cudaEventDestroy(g1);
  cudaGraphExecDestroy(gexec); cudaGraphDestroy(graph); cudaStreamDestroy(gs);
  double keys = 0;   // visible (row, key) pairs per head
  for (int qi = 0; qi < n_q; ++qi) { const int pos = q_pos0 + qi; keys += cs.window > 0 ? std::min(pos + 1, cs.window) : pos + 1; }
  const double flops = 4.0 * cs.D * cs.H * keys * n_seqs;
  const double kv_bytes = 2.0 * cs.Hkv * cs.D * n_seqs * (cs.window > 0 ? std::min(ctx, cs.window + n_q) : ctx);
  printf("  %-8s batch %d  n_q %4d  ctx %5d: graph %8.1f us/call | stream min %8.1f median %8.1f | %7.2f TFLOPS  %6.0f GB/s (KV read once)\n",
         cs.name, n_seqs, n_q, ctx, us, us_ev, med, flops / us / 1e6, kv_bytes / us / 1e3);
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  bool all = true;
  std::mt19937 rng(11);
  const Case sliding_t{"sliding", 256, 8, 16, 1024, 3, 1}, full_t{"full", 512, 1, 16, 0, 3, 2};
  if (!getenv("HIVE_TEST_TIMING_ONLY")) {
    all &= graph_case(sliding_t, {1, 4, 3}, rng);
    all &= graph_case(full_t, {1, 4, 3}, rng);
    all &= graph_case(sliding_t, {40, 1, 17}, rng);
    all &= graph_case(full_t, {40, 1, 17}, rng);
    all &= run_case(sliding_t, rng);
    all &= run_case(full_t, rng);
  }
  const cudaError_t e = cudaGetLastError();
  all &= check(e == cudaSuccess, cudaGetErrorString(e));
  printf("timings (real page layout, calls cycle over the layers so the KV comes from DRAM: sliding 40 layers, full 8 layers;\n"
         "  caveat: the GPU is time-shared with other processes; graph = per call when replayed from a CUDA graph)\n");
  const Case sliding{"sliding", 256, 8, 16, 1024, 40, 20}, full{"full", 512, 1, 16, 0, 8, 4};
  for (const Case* cs : {&sliding, &full})
    for (int ctx : {2048, 8192})
      for (int nq : {1, 4}) timing(*cs, 2, nq, ctx, 500);
  for (const Case* cs : {&sliding, &full}) timing(*cs, 1, 2048, 4096, 20);
  printf("%s\n", all ? "ALL PASS" : "FAIL");
  return all ? 0 : 1;
}
