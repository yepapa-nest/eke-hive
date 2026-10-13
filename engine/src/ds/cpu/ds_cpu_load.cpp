// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash CPU engine — loading (hive/ds/ds_cpu.h). Dense weights keep their checkpoint format (FP8 with 32×32 e8m0 block
//   scales, BF16, F32) in NUMA-split matrices. Routed experts (MXFP4 [N, K/2] + e8m0 [N, K/32] — the matmul kernels read that layout
//   directly) are stacked per layer like the other CPU families: gu = [every expert's gate rows | every expert's up rows] (gates on node 0,
//   ups on node 1) and d = [rows 0..dim/2 of every expert's down | rows dim/2..dim] — so every expert's work is split over both nodes'
//   memory bandwidth. HIVE_DS_CPU_EXPERTS=mmap aliases the checkpoint mapping instead (no copy, one node; page cache / SSD reads).
//   Engram tables: rows split in two halves over the nodes (RAM) or the checkpoint mapping (HIVE_DS_CPU_ENGRAM=mmap).
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "ds_cpu_internal.h"
#include "hive/clock.h"
#include "hive/common.h"

namespace hive::ds::cpu {

using ::hive::cpu::Pool;
using ::hive::cpu::SType;
using ::hive::cpu::WType;

std::vector<float> load_vec(Checkpoint& ck, const std::string& name) {
  const TensorInfo& t = ck.get(name);
  std::vector<float> v((size_t)t.numel());
  if (t.dtype == "BF16") { const uint16_t* p = (const uint16_t*)t.data; for (size_t i = 0; i < v.size(); ++i) v[i] = b2f(p[i]); }
  else if (t.dtype == "F32") memcpy(v.data(), t.data, v.size() * 4);
  else throw std::runtime_error("ds cpu: unexpected dtype " + t.dtype + " for " + name);
  return v;
}

Mat load_mat(Checkpoint& ck, const std::string& name) {
  const TensorInfo& t = ck.get(ck.has(name + ".weight") ? name + ".weight" : name);   // hc_*_fn are stored without the suffix
  if (t.shape.size() != 2) throw std::runtime_error("ds cpu: expected a matrix: " + name);
  const int N = (int)t.shape[0], K = (int)t.shape[1];
  if (t.dtype == "F8_E4M3") {
    const TensorInfo& s = ck.get(name + ".scale");
    if (s.dtype != "F8_E8M0" || s.shape.size() != 2 || s.shape[0] != (N + 31) / 32 || s.shape[1] != (K + 31) / 32)
      throw std::runtime_error("ds cpu: fp8 block scales of " + name);
    return ::hive::cpu::make_mat(WType::FP8, N, K, t.data, s.data, SType::E8M0, 1.f, 32, 32);
  }
  if (t.dtype == "BF16") return ::hive::cpu::make_mat(WType::BF16, N, K, t.data);
  if (t.dtype == "F32") return ::hive::cpu::make_mat(WType::F32, N, K, t.data);
  throw std::runtime_error("ds cpu: unexpected dtype " + t.dtype + " for " + name);
}

namespace {
// a matrix of n_rows rows whose node k holds rows [row0[k], row0[k+1]) — caller fills w / s
Mat alloc_mat(WType t, int N, int K, int split) {
  Mat m;
  m.t = t; m.N = N; m.K = K;
  m.nodes = Pool::get().nodes();
  if (m.nodes > 1) { m.row0[0] = 0; m.row0[1] = split; m.row0[2] = N; }
  else { m.row0[0] = 0; m.row0[1] = N; m.row0[2] = N; }
  for (int k = 0; k < m.nodes; ++k) {
    const size_t rows = (size_t)(m.row0[k + 1] - m.row0[k]);
    if (!rows) continue;
    m.w[k] = (uint8_t*)::hive::cpu::numa_alloc_node(rows * m.row_bytes(), k);
    m.s[k] = (uint8_t*)::hive::cpu::numa_alloc_node(rows * m.scale_row_bytes(), k);
    m.bytes += rows * (m.row_bytes() + m.scale_row_bytes());
  }
  return m;
}
uint8_t* row_w(Mat& m, int r) { const int k = (m.nodes > 1 && r >= m.row0[1]) ? 1 : 0; return m.w[k] + (size_t)(r - m.row0[k]) * m.row_bytes(); }
uint8_t* row_s(Mat& m, int r) { const int k = (m.nodes > 1 && r >= m.row0[1]) ? 1 : 0; return m.s[k] + (size_t)(r - m.row0[k]) * m.scale_row_bytes(); }
Mat alias_mat(WType t, int N, int K, const void* w, const void* s) {
  Mat m;
  m.t = t; m.N = N; m.K = K; m.nodes = 1;
  m.row0[0] = 0; m.row0[1] = N; m.row0[2] = N;
  m.w[0] = (uint8_t*)w; m.s[0] = (uint8_t*)s;
  return m;
}

// experts of another owner's half records (ExpertHalves): per expert three matrices whose rows are split over the two nodes exactly as the
//   half records are (node k = row half k) — no copy
void alias_experts(const ExpertHalves& X, int layer, int E, int dim, int I, MoeW& w) {
  w.ex.resize(E);
  w.alias = true;
  w.own.reserve(3 * (size_t)E);
  for (int e = 0; e < E; ++e) {
    const uint8_t* h0 = X.get(layer, e, 0);
    const uint8_t* h1 = X.get(layer, e, 1);
    HIVE_CHECK(h0 && h1, "ds cpu: expert half records missing");
    auto mk = [&](int N, int K, size_t wo, size_t so) {
      Mat m;
      m.t = WType::MXFP4; m.N = N; m.K = K; m.nodes = 2;
      m.row0[0] = 0; m.row0[1] = N / 2; m.row0[2] = N;
      m.w[0] = (uint8_t*)h0 + wo; m.s[0] = (uint8_t*)h0 + so;
      m.w[1] = (uint8_t*)h1 + wo; m.s[1] = (uint8_t*)h1 + so;
      return m;
    };
    w.own.push_back(mk(I, dim, X.w1, X.s1));
    w.own.push_back(mk(I, dim, X.w3, X.s3));
    w.own.push_back(mk(dim, I, X.w2, X.s2));
    const Mat* m = &w.own[w.own.size() - 3];
    w.ex[e].g = MatView{m, 0, I};
    w.ex[e].u = MatView{m + 1, 0, I};
    w.ex[e].d0 = MatView{m + 2, 0, dim};
  }
}

void load_experts(Checkpoint& ck, const std::string& p, int E, int dim, int I, bool mmap, MoeW& w) {
  struct T3 { const TensorInfo *w[3], *s[3]; };
  std::vector<T3> ts(E);
  const char* nm[3] = {"w1", "w3", "w2"};
  for (int e = 0; e < E; ++e)
    for (int j = 0; j < 3; ++j) {
      const std::string b = p + ".experts." + std::to_string(e) + "." + nm[j];
      ts[e].w[j] = &ck.get(b + ".weight");
      ts[e].s[j] = &ck.get(b + ".scale");
      const int N = j < 2 ? I : dim, K = j < 2 ? dim : I;
      if (ts[e].w[j]->nbytes != (size_t)N * K / 2 || ts[e].s[j]->nbytes != (size_t)N * K / 32 || ts[e].s[j]->dtype != "F8_E8M0")
        throw std::runtime_error("ds cpu: expert must be MXFP4 " + b);
    }
  w.ex.resize(E);
  w.alias = mmap;
  if (mmap) {
    w.own.reserve(3 * E);
    for (int e = 0; e < E; ++e) {
      w.own.push_back(alias_mat(WType::MXFP4, I, dim, ts[e].w[0]->data, ts[e].s[0]->data));
      w.own.push_back(alias_mat(WType::MXFP4, I, dim, ts[e].w[1]->data, ts[e].s[1]->data));
      w.own.push_back(alias_mat(WType::MXFP4, dim, I, ts[e].w[2]->data, ts[e].s[2]->data));
      const Mat* m = &w.own[w.own.size() - 3];
      w.ex[e].g = MatView{m, 0, I};
      w.ex[e].u = MatView{m + 1, 0, I};
      w.ex[e].d0 = MatView{m + 2, 0, dim};
    }
    return;
  }
  const int hh = dim / 2;
  w.own.resize(2);
  Mat& gu = w.own[0];
  Mat& d = w.own[1];
  gu = alloc_mat(WType::MXFP4, 2 * E * I, dim, E * I);
  d = alloc_mat(WType::MXFP4, E * dim, I, E * hh);
  // parallel copy: (expert, piece) tasks — gate, up, down half 0, down half 1
  Pool::get().run(4 * E, [&](int t) {
    const int e = t / 4, piece = t % 4;
    if (piece < 2) {
      const int r = piece * E * I + e * I;
      memcpy(row_w(gu, r), ts[e].w[piece]->data, (size_t)I * gu.row_bytes());
      memcpy(row_s(gu, r), ts[e].s[piece]->data, (size_t)I * gu.scale_row_bytes());
    } else {
      const int h = piece - 2, r = h * E * hh + e * hh;
      memcpy(row_w(d, r), ts[e].w[2]->data + (size_t)h * hh * d.row_bytes(), (size_t)hh * d.row_bytes());
      memcpy(row_s(d, r), ts[e].s[2]->data + (size_t)h * hh * d.scale_row_bytes(), (size_t)hh * d.scale_row_bytes());
    }
  });
  for (int e = 0; e < E; ++e) {
    w.ex[e].g = MatView{&gu, e * I, (e + 1) * I};
    w.ex[e].u = MatView{&gu, E * I + e * I, E * I + (e + 1) * I};
    w.ex[e].d0 = MatView{&d, e * hh, (e + 1) * hh};
    w.ex[e].d1 = MatView{&d, E * hh + e * hh, E * hh + (e + 1) * hh};
  }
}

void load_engram_table(Checkpoint& ck, const std::string& p, int hd, bool mmap, EngramTab& t) {
  const TensorInfo& tv = ck.get(p + ".embed.weight");
  const TensorInfo& ts = ck.get(p + ".embed.scale");
  if (tv.dtype != "F8_E4M3" || ts.dtype != "F8_E8M0" || tv.shape.size() != 2 || tv.shape[1] != hd || ts.shape[1] != hd / 32)
    throw std::runtime_error("ds cpu: engram table format " + p);
  t.rows = (uint64_t)tv.shape[0];
  const int ns = hd / 32;
  if (mmap) {
    t.split = t.rows; t.v[0] = tv.data; t.s[0] = ts.data; t.owned = false;
    ck.shard_for(p + ".embed.weight").advise_random();
    return;
  }
  Pool& P = Pool::get();
  const int nodes = P.nodes();
  t.split = nodes > 1 ? t.rows / 2 : t.rows;
  t.owned = true;
  uint8_t* v[2] = {nullptr, nullptr};
  uint8_t* s[2] = {nullptr, nullptr};
  for (int k = 0; k < nodes; ++k) {
    const uint64_t r0 = k ? t.split : 0, r1 = k ? t.rows : t.split;
    t.bytes[k] = (size_t)(r1 - r0) * (hd + ns);
    v[k] = (uint8_t*)::hive::cpu::numa_alloc_node((size_t)(r1 - r0) * hd, k);
    s[k] = (uint8_t*)::hive::cpu::numa_alloc_node((size_t)(r1 - r0) * ns, k);
  }
  ck.shard_for(p + ".embed.weight").advise_sequential();
  const uint64_t chunk = 1u << 20;  // rows per task
  std::vector<int> nt(2, 0);
  for (int k = 0; k < nodes; ++k) nt[k] = (int)(((k ? t.rows - t.split : t.split) + chunk - 1) / chunk);
  P.run_nodes(nt, [&](int k, int c) {
    const uint64_t base = k ? t.split : 0, end = k ? t.rows : t.split;
    const uint64_t a = base + (uint64_t)c * chunk, b = std::min(end, a + chunk);
    memcpy(v[k] + (a - base) * hd, tv.data + a * hd, (size_t)(b - a) * hd);
    memcpy(s[k] + (a - base) * ns, ts.data + a * ns, (size_t)(b - a) * ns);
  });
  for (int k = 0; k < 2; ++k) { t.v[k] = v[k]; t.s[k] = s[k]; }
  ck.shard_for(p + ".embed.weight").drop_cache();
  ck.shard_for(p + ".embed.scale").drop_cache();
}
}  // namespace

void DsCpuEngine::Impl::load_layer(LayerW& L, const std::string& p, int id, bool backbone, int E, int k) {
  Checkpoint& K = *ck;
  L.id = id;
  L.mtp = !backbone;
  L.ratio = backbone ? c.ratio(id) : 0;
  L.kv_src = backbone && c.is_kv_source(id);
  L.idx_src = backbone && c.is_index_source(id);
  if (L.ratio > 0) {
    for (size_t i = 0; i < kv_src.size(); ++i) if (kv_src[i] <= id) L.comp_ord = (int)i;
    if (L.comp_ord < 0) throw std::runtime_error("ds cpu: layer " + std::to_string(id) + " has no compressed-KV source before it");
  }
  if (L.kv_src) L.own_ord = (int)(std::find(kv_src.begin(), kv_src.end(), id) - kv_src.begin());
  const std::string a = p + ".attn";
  AttnW& A = L.at;
  A.wq_a = load_mat(K, a + ".wq_a"); A.wkv = load_mat(K, a + ".wkv"); A.wq_b = load_mat(K, a + ".wq_b");
  A.wo_a = load_mat(K, a + ".wo_a"); A.wo_b = load_mat(K, a + ".wo_b");
  A.q_norm = load_vec(K, a + ".q_norm.weight"); A.kv_norm = load_vec(K, a + ".kv_norm.weight"); A.sink = load_vec(K, a + ".attn_sink");
  if (L.kv_src) {
    A.comp_wkv = load_mat(K, a + ".compressor.wkv");
    if (L.ratio > 1) A.comp_wgate = load_mat(K, a + ".compressor.wgate");
    A.comp_norm = load_vec(K, a + ".compressor.norm.weight");
  }
  if (L.idx_src) {
    A.idx_wq_b = load_mat(K, a + ".indexer.wq_b");
    A.idx_wproj = load_mat(K, a + ".indexer.weights_proj");
    if (L.kv_src) { A.idx_wk = load_mat(K, a + ".indexer.wk"); A.idx_k_norm = load_vec(K, a + ".indexer.k_norm.weight"); }
  }
  L.attn_norm = load_vec(K, p + ".attn_norm.weight");
  L.ffn_norm = load_vec(K, p + ".ffn_norm.weight");
  L.hc_attn_fn = load_mat(K, p + ".hc_attn_fn"); L.hc_ffn_fn = load_mat(K, p + ".hc_ffn_fn");
  {
    const std::vector<float> s1 = load_vec(K, p + ".hc_attn_scale"), s2 = load_vec(K, p + ".hc_ffn_scale");
    for (int i = 0; i < 3; ++i) { L.hc_attn_scale[i] = s1[i]; L.hc_ffn_scale[i] = s2[i]; }
  }
  L.hc_attn_base = load_vec(K, p + ".hc_attn_base"); L.hc_ffn_base = load_vec(K, p + ".hc_ffn_base");
  MoeW& F = L.moe;
  F.E = E; F.k = k;
  F.gate = load_mat(K, p + ".ffn.gate");
  F.bias = load_vec(K, p + ".ffn.gate.bias");
  if (K.has(p + ".ffn.gate.bias_vl")) F.bias_vl = load_vec(K, p + ".ffn.gate.bias_vl");
  F.sh1 = load_mat(K, p + ".ffn.shared_experts.w1"); F.sh3 = load_mat(K, p + ".ffn.shared_experts.w3");
  F.sh2 = load_mat(K, p + ".ffn.shared_experts.w2");
  if (opt.ext_experts && backbone) alias_experts(*opt.ext_experts, id, E, c.dim, c.moe_inter, F);
  else load_experts(K, p + ".ffn", E, c.dim, c.moe_inter, opt.experts_mmap, F);
  if (backbone && std::find(c.engram_layer_ids.begin(), c.engram_layer_ids.end(), id) != c.engram_layer_ids.end()) {
    auto eg = std::make_unique<EngramW>();
    const std::string e = p + ".engram";
    eg->hash_index = (int)(std::find(c.engram_layer_ids.begin(), c.engram_layer_ids.end(), id) - c.engram_layer_ids.begin());
    eg->wkv = load_mat(K, e + ".wkv");
    const std::vector<float> q = load_vec(K, e + ".q_weight"), kk = load_vec(K, e + ".k_weight");
    eg->qk.resize(q.size());
    for (size_t i = 0; i < q.size(); ++i) eg->qk[i] = q[i] * kk[i];
    if (!opt.ext_engram) load_engram_table(K, e, c.engram_head_dim, opt.engram_mmap, eg->tab);
    L.engram = std::move(eg);
  }
}

void DsCpuEngine::Impl::load() {
  const double t0 = hive::mono_ms();
  c = DsConfig::load(dir);
  ck = std::make_unique<Checkpoint>(dir);
  kv_src = c.kv_source_layers;
  for (size_t i = 0; i < kv_src.size(); ++i) if (c.ratio(kv_src[i]) > 1) src2.push_back((int)i);
  tail_layer = kv_src.empty() ? -1 : *std::max_element(kv_src.begin(), kv_src.end());
  rope_win.init(c.rd, 0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow);
  rope_comp.init(c.rd, c.original_seq_len, c.compress_rope_theta, c.rope_factor, c.beta_fast, c.beta_slow);
  const bool block = opt.first_layer > 0;   // layer block mode: no embedding, no head, layers [first_layer, n_layers)
  if (!block) {
    const TensorInfo& t = ck->get("embed.weight");
    embed.resize((size_t)c.vocab * c.dim);
    memcpy(embed.data(), t.data, embed.size() * 2);
  }
  const int nl = opt.max_layer >= 0 ? std::min(opt.max_layer + 1, c.n_layers) : c.n_layers;
  layers.resize(nl);
  for (int l = std::max(0, opt.first_layer); l < nl; ++l) {
    const double tl = hive::mono_ms();
    load_layer(layers[l], "layers." + std::to_string(l), l, true, c.n_routed, c.n_act);
    fprintf(stderr, "[ds-cpu] layer %d loaded (%.1fs)\n", l, (hive::mono_ms() - tl) / 1000.0);
  }
  if (!block && nl == c.n_layers && ck->has("head.weight") && ck->has("norm.weight")) {
    head = load_mat(*ck, "head");
    norm = load_vec(*ck, "norm.weight");
    head_ok = true;
  }
  if (!block && ck->has("image_start")) {
    image_start = load_vec(*ck, "image_start"); image_end = load_vec(*ck, "image_end"); image_newline = load_vec(*ck, "image_newline");
  }
  size_t eb = 0, xb = 0;
  for (auto& L : layers) {
    for (auto& m : L.moe.own) xb += m.bytes;
    if (L.engram) eb += L.engram->tab.bytes[0] + L.engram->tab.bytes[1];
  }
  fprintf(stderr, "[ds-cpu] loaded %d layers%s in %.1fs · experts %.1f GB (%s) · engram %.1f GB (%s) · %d pool threads on %d NUMA nodes\n", nl,
          head_ok ? " + head" : " (no head)", (hive::mono_ms() - t0) / 1000.0, xb / 1e9, opt.experts_mmap ? "checkpoint mapping" : "host RAM", eb / 1e9,
          opt.engram_mmap ? "checkpoint mapping" : "host RAM", Pool::get().threads(), Pool::get().nodes());
}

void DsCpuEngine::Impl::load_mtp() {
  if (!mtp.empty() || (int)layers.size() != c.n_layers || !head_ok) return;
  const double t0 = hive::mono_ms();
  for (int s = 0; s < 64; ++s) {
    const std::string p = "mtp." + std::to_string(s);
    if (!ck->has(p + ".attn_norm.weight")) break;
    auto st_ = std::make_unique<MtpStage>();
    load_layer(st_->L, p, c.n_layers + s, false, c.dspark_experts, c.dspark_act);
    if (ck->has(p + ".main_proj.weight")) { st_->main_proj = load_mat(*ck, p + ".main_proj"); st_->main_norm = load_vec(*ck, p + ".main_norm.weight"); }
    if (ck->has(p + ".norm.weight")) st_->norm = load_vec(*ck, p + ".norm.weight");
    if (ck->has(p + ".markov_head.embed.weight")) {
      const TensorInfo& t = ck->get(p + ".markov_head.embed.weight");
      st_->markov_embed.resize((size_t)t.numel());
      memcpy(st_->markov_embed.data(), t.data, t.nbytes);
      st_->markov_head = load_mat(*ck, p + ".markov_head.head");
      st_->conf = load_vec(*ck, p + ".confidence_head.proj.weight");
    }
    mtp.push_back(std::move(st_));
  }
  if (!mtp.empty() && (mtp[0]->main_proj.N == 0 || mtp.back()->norm.empty() || mtp.back()->markov_embed.empty())) {
    fprintf(stderr, "[ds-cpu] DSpark stages incomplete in the checkpoint — drafting off\n");
    mtp.clear();
  }
  if (!mtp.empty()) fprintf(stderr, "[ds-cpu] DSpark: %d stages loaded (%.1fs)\n", (int)mtp.size(), (hive::mono_ms() - t0) / 1000.0);
}

DsCpuEngine::Impl::~Impl() {
  auto free_layer = [](LayerW& L) {
    AttnW& A = L.at;
    for (Mat* m : {&A.wq_a, &A.wkv, &A.wq_b, &A.wo_a, &A.wo_b, &A.comp_wkv, &A.comp_wgate, &A.idx_wq_b, &A.idx_wproj, &A.idx_wk, &L.hc_attn_fn,
                   &L.hc_ffn_fn, &L.moe.gate, &L.moe.sh1, &L.moe.sh3, &L.moe.sh2})
      ::hive::cpu::free_mat(*m);
    if (!L.moe.alias) for (Mat& m : L.moe.own) ::hive::cpu::free_mat(m);
    if (L.engram) {
      ::hive::cpu::free_mat(L.engram->wkv);
      EngramTab& t = L.engram->tab;
      if (t.owned)
        for (int k = 0; k < 2; ++k) {
          const uint64_t n = k ? t.rows - t.split : t.split;
          if (t.v[k]) ::hive::cpu::numa_free_node((void*)t.v[k], (size_t)n * 256);
          if (t.s[k]) ::hive::cpu::numa_free_node((void*)t.s[k], (size_t)n * 8);
        }
    }
  };
  for (LayerW& L : layers) free_layer(L);
  for (auto& m : mtp) { free_layer(m->L); ::hive::cpu::free_mat(m->main_proj); ::hive::cpu::free_mat(m->markov_head); }
  ::hive::cpu::free_mat(head);
}

}  // namespace hive::ds::cpu
