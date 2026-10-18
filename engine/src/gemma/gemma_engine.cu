// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gemma 4 engine (hive/gemma/gemma_engine.h): loading, the paged KV pool, the forward over rows of several sequences (prefill chunks, decode,
//   verify), the Frozen-KV MTP assistant. Rows ≤ 16 take the int4 decode GEMV; larger chunks dequantize each matrix into one bf16 scratch
//   and run a cuBLAS GEMM. Activations are bf16 between ops (the points the reference keeps in bf16), arithmetic fp32.
#include "hive/gemma/gemma_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

#include "hive/common.h"
#include "hive/cublas_ops.h"
#include "hive/model_kernels.h"
#include "hive/safetensors.h"
#include "nlohmann/json.hpp"

namespace hive::gemma {

using json = nlohmann::json;
using namespace hive::gemma::k;

GemmaConfig GemmaConfig::load(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  HIVE_CHECK(f.good(), "gemma: config.json not found");
  json j = json::parse(f);
  const json& t = j.contains("text_config") ? j["text_config"] : j;
  GemmaConfig c;
  c.n_layers = t.at("num_hidden_layers");
  c.hidden = t.at("hidden_size");
  c.inter = t.at("intermediate_size");
  c.heads = t.at("num_attention_heads");
  c.vocab = t.at("vocab_size");
  c.kv_heads = t.at("num_key_value_heads");
  c.head_dim = t.at("head_dim");
  c.g_head_dim = t.value("global_head_dim", c.head_dim);
  c.k_eq_v = t.value("attention_k_eq_v", false);
  c.g_kv_heads = c.k_eq_v ? t.value("num_global_key_value_heads", c.kv_heads) : c.kv_heads;
  c.window = t.value("sliding_window", 0);
  c.eps = t.value("rms_norm_eps", 1e-6);
  c.softcap = t.contains("final_logit_softcapping") && !t["final_logit_softcapping"].is_null() ? (float)t["final_logit_softcapping"] : 0.f;
  // the bf16 rounding of sqrt(hidden) (the reference multiplies in the embedding's dtype)
  c.embed_scale = __bfloat162float(__float2bfloat16(std::sqrt((float)c.hidden)));
  const json& rp = t.at("rope_parameters");
  c.theta_s = rp.at("sliding_attention").value("rope_theta", 1e4);
  c.theta_f = rp.at("full_attention").value("rope_theta", 1e6);
  c.partial_f = rp.at("full_attention").value("partial_rotary_factor", 1.0);
  for (const auto& lt : t.at("layer_types")) {
    const bool s = lt.get<std::string>() == "sliding_attention";
    c.type_index.push_back(s ? c.n_sliding : c.n_full);
    (s ? c.n_sliding : c.n_full)++;
    c.sliding.push_back(s);
  }
  HIVE_CHECK((int)c.sliding.size() == c.n_layers, "gemma: layer_types");
  HIVE_CHECK(t.value("num_kv_shared_layers", 0) == 0 && t.value("hidden_size_per_layer_input", 0) == 0, "gemma: KV-shared / per-layer-input target not supported");
  std::ifstream g(dir + "/generation_config.json");
  if (g.good()) {
    json gj = json::parse(g);
    if (gj.contains("eos_token_id")) {
      if (gj["eos_token_id"].is_array()) for (auto& e : gj["eos_token_id"]) c.eos.push_back(e);
      else c.eos.push_back(gj["eos_token_id"]);
    }
  }
  if (c.eos.empty()) c.eos = {1, 106};
  return c;
}

namespace {

struct LayerW {
  W4Mat q, k, v, o, gate, up, down;
  DevBuf in_ln, post_attn, pre_ff, post_ff, q_norm, k_norm, scalar;
  bool has_v = false;
};
struct DraftLayer {
  DevBuf q, o, gate, up, down, in_ln, post_attn, pre_ff, post_ff, q_norm, scalar;
  int D = 0;
  bool sliding = true;
};

DevBuf upload(const TensorInfo* t) {
  HIVE_CHECK(t != nullptr, "gemma: tensor missing");
  DevBuf b(t->nbytes);
  CUDA_CHECK(cudaMemcpy(b.p, t->data, t->nbytes, cudaMemcpyHostToDevice));
  return b;
}

}  // namespace

struct GemmaEngine::Impl {
  GemmaConfig c;
  Placement place;
  cudaStream_t st = nullptr;
  std::unique_ptr<Blas> blasp;   // created on the engine stream
  std::vector<std::unique_ptr<SafetensorsFile>> files;
  std::vector<LayerW> L;
  std::vector<DevBuf> wbufs;   // packed weights / scales (owned)
  DevBuf embed, norm;
  bf16* embed_host = nullptr;   // HIVE_GEMMA_EMBED_HOST=1: the embedding / lm_head in mapped pinned host memory (read over the bus)
  const bf16* E() const { return embed_host ? embed_host : embed.as<bf16>(); }
  DevBuf inv_s, inv_f;
  // assistant
  bool mtp = false;
  int mtp_k = 3;
  int d_hidden = 0, d_inter = 0;
  std::vector<DraftLayer> DL;
  DevBuf d_pre, d_post, d_norm, d_embed;
  // KV pool
  KvLayout Ls, Lf;
  DevBuf pool_s, pool_f;
  std::vector<uint8_t*> free_s, free_f;
  std::set<uint8_t*> host_pages;
  std::vector<uint8_t*> host_free_s, host_free_f;
  size_t host_limit = 0, host_used = 0;
  int n_pages_s = 0, n_pages_f = 0;
  int64_t max_ctx = 0;
  int swa_keep = 2048;
  // work
  int max_rows = 0, max_batch = 0;
  DevBuf x, hbuf, qraw, kraw, vraw, qb, kb, vb, ob, yb, gb, ub, act, deq, ws, hn;
  DevBuf ids_d, pos_d, seqof_d, as_s, as_f;
  DevBuf logits_d;
  float* logits_h = nullptr;
  bool host_logits = true;
  size_t weights_bytes = 0, work_bytes = 0;
  // drafter work
  DevBuf dcat, dx, dh, dq, dob, dy, dg, du, dact, dzn, dlog, dhn, darg;
  std::function<void(int)> layer_boundary;
  std::function<bool()> layer_boundary_due;
  std::vector<GemmaSeq*> last_verify;
  std::vector<int> last_verify_rows;

  const TensorInfo* find(const std::string& name) const {
    for (const auto& f : files) if (auto* t = f->find(name)) return t;
    return nullptr;
  }
  void open_files(const std::string& dir) {
    files.clear();
    std::ifstream idx(dir + "/model.safetensors.index.json");
    if (idx.good()) {
      json j = json::parse(idx);
      std::set<std::string> names;
      for (auto& [k, v] : j["weight_map"].items()) names.insert(v.get<std::string>());
      for (const auto& n : names) files.push_back(std::make_unique<SafetensorsFile>(dir + "/" + n));
    } else {
      files.push_back(std::make_unique<SafetensorsFile>(dir + "/model.safetensors"));
    }
  }
  // an int4 matrix from the checkpoint, repacked on the GPU (checkpoint-order copies freed right after)
  W4Mat load_w4(const std::string& prefix) {
    const TensorInfo* q = find(prefix + ".weight_packed");
    const TensorInfo* s = find(prefix + ".weight_scale");
    HIVE_CHECK(q && s && q->shape.size() == 2 && s->shape.size() == 2, "gemma: int4 matrix missing");
    const int N = (int)q->shape[0], K = (int)q->shape[1] * 8;
    HIVE_CHECK(s->shape[1] * 32 == K, "gemma: group size is not 32");
    DevBuf qc = upload(q), sc = upload(s);
    DevBuf qr((size_t)N * K / 2), sr((size_t)N * K / 32 * 2);
    w4_repack(qc.as<uint32_t>(), sc.as<bf16>(), N, K, qr.as<uint32_t>(), sr.as<uint32_t>(), st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    W4Mat w;
    w.q = qr.as<uint32_t>(); w.s = sr.as<uint32_t>(); w.N = N; w.K = K; w.split = w4_split_for(N, K);
    weights_bytes += qr.n + sr.n;
    wbufs.push_back(std::move(qr)); wbufs.push_back(std::move(sr));
    return w;
  }
  DevBuf vec(const std::string& name) { DevBuf b = upload(find(name)); weights_bytes += b.n; return b; }

  void load_target(const std::string& dir) {
    open_files(dir);
    const std::string P = "model.language_model.";
    if (getenv("HIVE_GEMMA_EMBED_HOST") && strcmp(getenv("HIVE_GEMMA_EMBED_HOST"), "0") != 0) {
      const TensorInfo* t = find(P + "embed_tokens.weight");
      HIVE_CHECK(t != nullptr, "gemma: embedding missing");
      void* h = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h, t->nbytes, cudaHostAllocMapped));
      memcpy(h, t->data, t->nbytes);
      embed_host = (bf16*)h;
    } else {
      embed = vec(P + "embed_tokens.weight");   // also the lm_head (the checkpoint's lm_head.weight is the same bytes — tied)
    }
    norm = vec(P + "norm.weight");
    // HIVE_GEMMA_LAYERS=N (validation only): load the first N layers — the output is then the residual after layer N - 1
    const int nl = getenv("HIVE_GEMMA_LAYERS") ? std::max(1, std::min(c.n_layers, atoi(getenv("HIVE_GEMMA_LAYERS")))) : c.n_layers;
    L.resize(nl);
    for (int l = 0; l < nl; ++l) {
      const std::string p = P + "layers." + std::to_string(l) + ".";
      LayerW& w = L[l];
      w.q = load_w4(p + "self_attn.q_proj");
      w.k = load_w4(p + "self_attn.k_proj");
      w.has_v = find(p + "self_attn.v_proj.weight_packed") != nullptr;
      if (w.has_v) w.v = load_w4(p + "self_attn.v_proj");
      w.o = load_w4(p + "self_attn.o_proj");
      w.gate = load_w4(p + "mlp.gate_proj");
      w.up = load_w4(p + "mlp.up_proj");
      w.down = load_w4(p + "mlp.down_proj");
      w.in_ln = vec(p + "input_layernorm.weight");
      w.post_attn = vec(p + "post_attention_layernorm.weight");
      w.pre_ff = vec(p + "pre_feedforward_layernorm.weight");
      w.post_ff = vec(p + "post_feedforward_layernorm.weight");
      w.q_norm = vec(p + "self_attn.q_norm.weight");
      w.k_norm = vec(p + "self_attn.k_norm.weight");
      w.scalar = vec(p + "layer_scalar");
    }
    files.clear();
  }

  void load_assistant(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f.good()) return;
    json j = json::parse(f);
    const json& t = j.contains("text_config") ? j["text_config"] : j;
    HIVE_CHECK(!j.value("use_ordered_embeddings", false), "gemma: centroid-head assistants are not supported");
    open_files(dir);
    d_hidden = t.at("hidden_size");
    d_inter = t.at("intermediate_size");
    const std::string P = find("model.language_model.norm.weight") ? "model.language_model." : "model.";
    d_pre = vec("pre_projection.weight");
    d_post = vec("post_projection.weight");
    d_norm = vec(P + "norm.weight");
    d_embed = vec(find("lm_head.weight") ? "lm_head.weight" : P + "embed_tokens.weight");
    const auto& types = t.at("layer_types");
    for (int i = 0; i < (int)types.size(); ++i) {
      const std::string p = P + "layers." + std::to_string(i) + ".";
      DraftLayer d;
      d.sliding = types[i].get<std::string>() == "sliding_attention";
      d.D = d.sliding ? c.head_dim : c.g_head_dim;
      d.q = vec(p + "self_attn.q_proj.weight");
      d.o = vec(p + "self_attn.o_proj.weight");
      d.gate = vec(p + "mlp.gate_proj.weight");
      d.up = vec(p + "mlp.up_proj.weight");
      d.down = vec(p + "mlp.down_proj.weight");
      d.in_ln = vec(p + "input_layernorm.weight");
      d.post_attn = vec(p + "post_attention_layernorm.weight");
      d.pre_ff = vec(p + "pre_feedforward_layernorm.weight");
      d.post_ff = vec(p + "post_feedforward_layernorm.weight");
      d.q_norm = vec(p + "self_attn.q_norm.weight");
      if (find(p + "layer_scalar")) d.scalar = vec(p + "layer_scalar");
      DL.push_back(std::move(d));
    }
    files.clear();
    mtp = true;
  }

  // ---- KV pages ----
  uint8_t* alloc_page(bool sliding) {
    auto& fl = sliding ? free_s : free_f;
    for (int tries = 0; fl.empty() && kv_reclaim_ && tries < 64; ++tries)
      if (!kv_reclaim_()) break;
    if (!fl.empty()) { uint8_t* p = fl.back(); fl.pop_back(); return p; }
    // the pool is full and nothing idle can be evicted: the page goes to mapped pinned host memory (read over the bus — slower, never
    //   an out-of-memory)
    auto& hf = sliding ? host_free_s : host_free_f;
    if (!hf.empty()) { uint8_t* p = hf.back(); hf.pop_back(); return p; }
    const size_t pb = (sliding ? Ls : Lf).page_bytes();
    if (host_used + pb > host_limit) throw DeviceOutOfMemory("gemma: KV pool and host KV budget (HIVE_GEMMA_HOST_KV_MB) exhausted");
    void* hp = nullptr;
    CUDA_CHECK(cudaHostAlloc(&hp, pb, cudaHostAllocMapped));
    host_used += pb;
    host_pages.insert((uint8_t*)hp);
    static int warned = 0;
    if (warned++ < 4) fprintf(stderr, "[gemma] KV pool full: %s page on the host (%.0f MiB of host KV in use)\n", sliding ? "sliding" : "full", host_used / 1048576.0);
    return (uint8_t*)hp;
  }
  void free_page(uint8_t* p, bool sliding) {
    if (!p) return;
    if (host_pages.count(p)) (sliding ? host_free_s : host_free_f).push_back(p);
    else (sliding ? free_s : free_f).push_back(p);
  }
  std::function<bool()> kv_reclaim_;

  void push_tables(GemmaSeq& s, int from_s, int from_f) {
    if ((int)s.pages_s.size() > from_s)
      CUDA_CHECK(cudaMemcpyAsync(s.tab_s.as<uint8_t*>() + from_s, s.pages_s.data() + from_s, (s.pages_s.size() - from_s) * sizeof(void*), cudaMemcpyHostToDevice, st));
    if ((int)s.pages_f.size() > from_f)
      CUDA_CHECK(cudaMemcpyAsync(s.tab_f.as<uint8_t*>() + from_f, s.pages_f.data() + from_f, (s.pages_f.size() - from_f) * sizeof(void*), cudaMemcpyHostToDevice, st));
  }
  // pages for positions [s.pos, s.pos + n)
  void reserve(GemmaSeq& s, int n) {
    const int64_t end = s.pos + n;
    HIVE_CHECK(end <= s.max_ctx, "gemma: context exceeds the sequence capacity");
    const int need = (int)((end + kPage - 1) / kPage);
    const int f0 = (int)s.pages_f.size(), s0 = (int)s.pages_s.size();
    while ((int)s.pages_f.size() < need) s.pages_f.push_back(alloc_page(false));
    while ((int)s.pages_s.size() < need) s.pages_s.push_back(alloc_page(true));
    push_tables(s, s0, f0);
  }
  // sliding pages no query can read again: below pos - window - swa_keep (the keep margin lets a verify roll back and a session image
  //   cover the recent window)
  void trim_sliding(GemmaSeq& s) {
    const int64_t lo = s.pos - c.window - swa_keep;
    if (lo <= 0) return;
    const int first_keep = std::min((int)(lo / kPage), (int)s.pages_s.size());
    const int old = (int)(s.swa_lo / kPage);
    if (first_keep <= old) return;
    for (int i = old; i < first_keep; ++i) { free_page(s.pages_s[i], true); s.pages_s[i] = nullptr; }
    s.swa_lo = (int64_t)first_keep * kPage;
    CUDA_CHECK(cudaMemcpyAsync(s.tab_s.as<uint8_t*>() + old, s.pages_s.data() + old, (size_t)(first_keep - old) * sizeof(void*), cudaMemcpyHostToDevice, st));
  }
  void release(GemmaSeq& s) {
    for (auto* p : s.pages_s) free_page(p, true);
    for (auto* p : s.pages_f) free_page(p, false);
    s.pages_s.clear(); s.pages_f.clear();
    s.swa_lo = 0;
  }

  // up to gemv_rows rows: the int4 GEMV in passes of 16 rows (each pass streams the weights once, ~6.5 ms per layer stack) — cheaper
  //   than dequantizing every matrix to bf16 for cuBLAS (~45 ms for a 20-token prompt; 66-75 ms for the ~70-row turn tails of a long
  //   conversation, against 7-11 ms per 16-row pass — measured 2026-10-18 in the service log). HIVE_GEMMA_GEMV_ROWS (default 128).
  int gemv_rows = 128;
  void linear(const W4Mat& W, const bf16* x, int R, bf16* y) {
    if (R <= gemv_rows) {
      for (int r0 = 0; r0 < R; r0 += 16)
        w4_gemv(W, x + (size_t)r0 * W.K, W.K, std::min(16, R - r0), y + (size_t)r0 * W.N, nullptr, W.N, ws.as<float>(), st);
      return;
    }
    w4_dequant(W, deq.as<bf16>(), st);
    blasp->gemm_bf16(x, deq.as<bf16>(), y, R, W.N, W.K);
  }
  void mlp_front(const LayerW& w, const bf16* h, int R) {
    if (R <= gemv_rows) {
      for (int r0 = 0; r0 < R; r0 += 16)
        w4_gemv_gu(w.gate, w.up, h + (size_t)r0 * c.hidden, c.hidden, std::min(16, R - r0), act.as<bf16>() + (size_t)r0 * c.inter, c.inter,
                   ws.as<float>(), st);
      return;
    }
    linear(w.gate, h, R, gb.as<bf16>());
    linear(w.up, h, R, ub.as<bf16>());
    gelu_mul(gb.as<bf16>(), c.inter, ub.as<bf16>(), c.inter, R, c.inter, act.as<bf16>(), c.inter, st);
  }

  struct Part { GemmaSeq* s; const int32_t* ids; int n; };
  void compute(int R, int S, int maxq, bool all_rows, const std::vector<int>& rows_last, bool want_logits, bool prefill);
  void draft_chain(int S, int k);
  DevBuf dsum, dmax, dones, dpart;
  // pinned staging of the per-forward inputs (ids / pos / seq_of: 3 · max_rows int32; 2 · 64 AttnSeq)
  struct Pinned { int32_t* p = nullptr; int32_t* as_host() { return p; } } stage_i;
  void* stage_seq = nullptr;
  bool use_graphs = true;
  std::map<std::string, cudaGraphExec_t> graphs;
  long n_captures = 0;
  // one forward over the parts' rows; logits rows: every row (decode / verify) or the last row of each part (prefill)
  void run(const std::vector<Part>& parts, bool all_rows, float* logits_out, bool keep_rows_hidden);
  // decode / verify graphs are keyed by the forward's shape only — total rows, sequences, the most rows of one sequence, logits on/off:
  //   every kernel reads the rows of each sequence from the device inputs (AttnSeq, seq_of, pos), so one graph serves every split of R
  //   rows over S sequences (keying by the split made up to 4^4 graphs per S and they appeared one by one while serving)
  cudaGraphExec_t fwd_graph(int R, int S, int maxq, bool want_logits);
  cudaGraphExec_t draft_graph(int S, int k);
};


// The kernels of one forward over R rows of S sequences (inputs already in ids_d / pos_d / seqof_d / as_s / as_f): everything a CUDA
//   graph can hold — no host copies, no allocation. Logit rows: all R rows (decode / verify) or the last row of each part (rows_last).
void GemmaEngine::Impl::compute(int R, int S, int maxq, bool all_rows, const std::vector<int>& rows_last, bool want_logits, bool prefill) {
  const int H = c.hidden;
  bf16* X = x.as<bf16>();
  k::embed(E(), ids_d.as<int32_t>(), R, H, c.embed_scale, X, H, st);
  for (int l = 0; l < (int)L.size(); ++l) {
    const LayerW& w = L[l];
    const bool sl = c.sliding[l];
    const int D = sl ? c.head_dim : c.g_head_dim, Hkv = sl ? c.kv_heads : c.g_kv_heads;
    const float* inv = sl ? inv_s.as<float>() : inv_f.as<float>();
    if (l == 0) rmsnorm(X, H, w.in_ln.as<bf16>(), R, H, c.eps, hbuf.as<bf16>(), H, st);   // later layers: fused into the previous add
    if (R <= gemv_rows) {   // q / k (/ v) in one GEMV launch per 16 rows
      for (int r0 = 0; r0 < R; r0 += 16) {
        W4Multi J;
        J.n = w.has_v ? 3 : 2;
        J.W[0] = w.q; J.W[1] = w.k; J.ldy[0] = w.q.N; J.ldy[1] = w.k.N;
        J.yb[0] = qraw.as<bf16>() + (size_t)r0 * w.q.N; J.yb[1] = kraw.as<bf16>() + (size_t)r0 * w.k.N;
        if (w.has_v) { J.W[2] = w.v; J.yb[2] = vraw.as<bf16>() + (size_t)r0 * w.v.N; J.ldy[2] = w.v.N; }
        w4_gemv_multi(J, hbuf.as<bf16>() + (size_t)r0 * H, H, std::min(16, R - r0), st);
      }
    } else {
      linear(w.q, hbuf.as<bf16>(), R, qraw.as<bf16>());
      linear(w.k, hbuf.as<bf16>(), R, kraw.as<bf16>());
      if (w.has_v) linear(w.v, hbuf.as<bf16>(), R, vraw.as<bf16>());
    }
    HnrParts hp;
    hp.n = 3;
    hp.p[0] = {qraw.as<bf16>(), c.heads * D, c.heads, w.q_norm.as<bf16>(), inv, qb.as<bf16>(), c.heads * D};
    hp.p[1] = {kraw.as<bf16>(), Hkv * D, Hkv, w.k_norm.as<bf16>(), inv, kb.as<bf16>(), Hkv * D};
    hp.p[2] = {w.has_v ? vraw.as<bf16>() : kraw.as<bf16>(), Hkv * D, Hkv, nullptr, nullptr, vb.as<bf16>(), Hkv * D};
    head_norm_rope(hp, R, D, c.eps, pos_d.as<int32_t>(), st);
    const KvLayout& LT = sl ? Ls : Lf;
    const AttnSeq* A = sl ? as_s.as<AttnSeq>() : as_f.as<AttnSeq>();
    kv_write(kb.as<bf16>(), vb.as<bf16>(), R, seqof_d.as<int32_t>(), pos_d.as<int32_t>(), A, LT, c.type_index[l], st);
    attn_paged(qb.as<bf16>(), ob.as<bf16>(), c.heads, LT, c.type_index[l], A, S, maxq, sl ? c.window : 0, st);
    linear(w.o, ob.as<bf16>(), R, yb.as<bf16>());
    add_postnorm(X, yb.as<bf16>(), H, w.post_attn.as<bf16>(), R, H, c.eps, nullptr, st, w.pre_ff.as<bf16>(), hbuf.as<bf16>(), H);
    mlp_front(w, hbuf.as<bf16>(), R);
    linear(w.down, act.as<bf16>(), R, yb.as<bf16>());
    const bool more = l + 1 < (int)L.size();
    add_postnorm(X, yb.as<bf16>(), H, w.post_ff.as<bf16>(), R, H, c.eps, w.scalar.as<bf16>(), st, more ? L[l + 1].in_ln.as<bf16>() : nullptr,
                 more ? hbuf.as<bf16>() : nullptr, H);
    if (prefill && layer_boundary && (!layer_boundary_due || layer_boundary_due())) {
      CUDA_CHECK(cudaStreamSynchronize(st));
      layer_boundary(l);
    }
  }
  bf16* HN = hn.as<bf16>();
  const int NR = all_rows ? R : (int)rows_last.size();
  if (all_rows) rmsnorm(X, H, norm.as<bf16>(), R, H, c.eps, HN, H, st);
  else for (int j = 0; j < NR; ++j) rmsnorm(X + (size_t)rows_last[j] * H, H, norm.as<bf16>(), 1, H, c.eps, HN + (size_t)j * H, H, st);
  if (want_logits) {
    blasp->gemm_bf16_f32out(HN, E(), logits_d.as<float>(), NR, c.vocab, H);
    if (c.softcap > 0) softcap(logits_d.as<float>(), (size_t)NR * c.vocab, c.softcap, st);
  }
}

void GemmaEngine::Impl::run(const std::vector<Part>& parts, bool all_rows, float* logits_out, bool keep_rows_hidden) {
  const int S = (int)parts.size(), H = c.hidden;
  int R = 0, maxq = 0;
  for (const Part& p : parts) { R += p.n; maxq = std::max(maxq, p.n); }
  HIVE_CHECK(R >= 1 && R <= max_rows && S <= 64, "gemma: rows of one forward");
  for (const Part& p : parts) reserve(*p.s, p.n);
  // inputs through one pinned staging block (ids · pos · seq_of · sliding AttnSeq · full AttnSeq)
  int32_t* ids = stage_i.as_host();
  int32_t* pos = ids + R;
  int32_t* seqof = pos + R;
  AttnSeq* as = reinterpret_cast<AttnSeq*>(stage_seq);
  AttnSeq* af = as + S;
  int r = 0;
  std::vector<int> rows_last;
  for (int i = 0; i < S; ++i) {
    const Part& p = parts[i];
    as[i].pages = p.s->tab_s.as<const uint8_t*>(); af[i].pages = p.s->tab_f.as<const uint8_t*>();
    as[i].q_row0 = af[i].q_row0 = r; as[i].n_q = af[i].n_q = p.n; as[i].q_pos0 = af[i].q_pos0 = (int)p.s->pos;
    for (int j = 0; j < p.n; ++j, ++r) { ids[r] = p.ids[j]; pos[r] = (int32_t)(p.s->pos + j); seqof[r] = i; }
    rows_last.push_back(r - 1);
  }
  CUDA_CHECK(cudaMemcpyAsync(ids_d.p, ids, (size_t)R * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(pos_d.p, pos, (size_t)R * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(seqof_d.p, seqof, (size_t)R * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(as_s.p, as, (size_t)S * sizeof(AttnSeq), cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(as_f.p, af, (size_t)S * sizeof(AttnSeq), cudaMemcpyHostToDevice, st));
  const bool want_logits = logits_out || !host_logits;
  // decode / verify (≤ 16 rows): one CUDA graph per row shape (rows per sequence, in order)
  if (all_rows && R <= 16 && use_graphs) {
    CUDA_CHECK(cudaGraphLaunch(fwd_graph(R, S, maxq, want_logits), st));
  } else {
    compute(R, S, maxq, all_rows, rows_last, want_logits, !all_rows);
  }
  if (const char* dump = getenv("HIVE_GEMMA_DUMP"); dump) {   // validation: the residual of the forward's last row (fp32)
    std::vector<uint16_t> b(H);
    CUDA_CHECK(cudaMemcpyAsync(b.data(), x.as<bf16>() + (size_t)(R - 1) * H, (size_t)H * 2, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<float> f(H);
    for (int i = 0; i < H; ++i) { uint32_t u = (uint32_t)b[i] << 16; memcpy(&f[i], &u, 4); }
    if (FILE* fp = fopen(dump, "wb")) { fwrite(f.data(), 4, H, fp); fclose(fp); }
  }
  const int NR = all_rows ? R : S;
  if (logits_out && host_logits) CUDA_CHECK(cudaMemcpyAsync(logits_out, logits_d.p, (size_t)NR * c.vocab * 4, cudaMemcpyDeviceToHost, st));
  // MTP seeds: the final-normed hidden of each part's last row (and every verify row, for rollback)
  bf16* HN = hn.as<bf16>();
  int base = 0, j = 0;
  for (const Part& p : parts) {
    const int last = all_rows ? base + p.n - 1 : j;
    CUDA_CHECK(cudaMemcpyAsync(p.s->h_last.p, HN + (size_t)last * H, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
    if (keep_rows_hidden && all_rows) CUDA_CHECK(cudaMemcpyAsync(p.s->h_rows.p, HN + (size_t)base * H, (size_t)p.n * H * 2, cudaMemcpyDeviceToDevice, st));
    p.s->h_last_valid = true;
    p.s->pos += p.n;
    p.s->tokens.insert(p.s->tokens.end(), p.ids, p.ids + p.n);
    base += p.n; ++j;
  }
  for (const Part& p : parts) trim_sliding(*p.s);
  CUDA_CHECK(cudaStreamSynchronize(st));
}

cudaGraphExec_t GemmaEngine::Impl::fwd_graph(int R, int S, int maxq, bool want_logits) {
  const std::string key = "R" + std::to_string(R) + "S" + std::to_string(S) + "q" + std::to_string(maxq) + (want_logits ? "L" : "-");
  auto it = graphs.find(key);
  if (it != graphs.end()) return it->second;
  cudaGraph_t g = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  compute(R, S, maxq, true, {}, want_logits, false);
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  cudaGraphExec_t ge = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  CUDA_CHECK(cudaGraphDestroy(g));
  CUDA_CHECK(cudaGraphUpload(ge, st));
  graphs.emplace(key, ge);
  ++n_captures;
  return ge;
}

cudaGraphExec_t GemmaEngine::Impl::draft_graph(int S, int k) {
  const std::string key = "D" + std::to_string(S) + "k" + std::to_string(k);
  auto it = graphs.find(key);
  if (it != graphs.end()) return it->second;
  cudaGraph_t g = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  draft_chain(S, k);
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  cudaGraphExec_t ge = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  CUDA_CHECK(cudaGraphDestroy(g));
  CUDA_CHECK(cudaGraphUpload(ge, st));
  graphs.emplace(key, ge);
  ++n_captures;
  return ge;
}

// Frozen-KV MTP chain of k steps for S sequences, all on the device (graph-capturable): step 0 embeds ids_d (the seeds), step j > 0 the
//   argmax of step j - 1 (darg row j - 1); the hidden half of the input is dcat's second half (h_last for step 0, then post_projection).
void GemmaEngine::Impl::draft_chain(int S, int k) {
  const int H = c.hidden, DH = d_hidden;
  for (int step = 0; step < k; ++step) {
    const int32_t* in = step == 0 ? ids_d.as<int32_t>() : darg.as<int32_t>() + (size_t)(step - 1) * 16;
    k::embed(E(), in, S, H, c.embed_scale, dcat.as<bf16>(), 2 * H, st);
    blasp->gemm_bf16(dcat.as<bf16>(), d_pre.as<bf16>(), dx.as<bf16>(), S, DH, 2 * H);
    for (size_t li = 0; li < DL.size(); ++li) {
      const DraftLayer& d = DL[li];
      const int D = d.D;
      const float* inv = d.sliding ? inv_s.as<float>() : inv_f.as<float>();
      if (li == 0) rmsnorm(dx.as<bf16>(), DH, d.in_ln.as<bf16>(), S, DH, c.eps, dh.as<bf16>(), DH, st);   // later: fused into the add
      blasp->gemm_bf16(dh.as<bf16>(), d.q.as<bf16>(), dq.as<bf16>(), S, c.heads * D, DH);
      head_norm_rope(dq.as<bf16>(), c.heads * D, S, c.heads, D, d.q_norm.as<bf16>(), c.eps, inv, pos_d.as<int32_t>(), dq.as<bf16>(), c.heads * D, st);
      const KvLayout& LT = d.sliding ? Ls : Lf;
      attn_paged(dq.as<bf16>(), dob.as<bf16>(), c.heads, LT, d.sliding ? c.n_sliding - 1 : c.n_full - 1,
                 d.sliding ? as_s.as<AttnSeq>() : as_f.as<AttnSeq>(), S, 1, d.sliding ? c.window : 0, st);
      blasp->gemm_bf16(dob.as<bf16>(), d.o.as<bf16>(), dy.as<bf16>(), S, DH, c.heads * D);
      add_postnorm(dx.as<bf16>(), dy.as<bf16>(), DH, d.post_attn.as<bf16>(), S, DH, c.eps, nullptr, st, d.pre_ff.as<bf16>(), dh.as<bf16>(), DH);
      blasp->gemm_bf16(dh.as<bf16>(), d.gate.as<bf16>(), dg.as<bf16>(), S, d_inter, DH);
      blasp->gemm_bf16(dh.as<bf16>(), d.up.as<bf16>(), du.as<bf16>(), S, d_inter, DH);
      gelu_mul(dg.as<bf16>(), d_inter, du.as<bf16>(), d_inter, S, d_inter, dact.as<bf16>(), d_inter, st);
      blasp->gemm_bf16(dact.as<bf16>(), d.down.as<bf16>(), dy.as<bf16>(), S, DH, d_inter);
      const bool last = li + 1 == DL.size();   // fused: the next layer's input norm, or the drafter's final norm
      add_postnorm(dx.as<bf16>(), dy.as<bf16>(), DH, d.post_ff.as<bf16>(), S, DH, c.eps, d.scalar.p ? d.scalar.as<bf16>() : nullptr, st,
                   last ? d_norm.as<bf16>() : DL[li + 1].in_ln.as<bf16>(), last ? dzn.as<bf16>() : dh.as<bf16>(), DH);
    }
    blasp->gemm_bf16_f32out(dzn.as<bf16>(), d_embed.as<bf16>(), dlog.as<float>(), S, c.vocab, DH);
    blasp->gemm_bf16(dzn.as<bf16>(), d_post.as<bf16>(), dhn.as<bf16>(), S, H, DH);
    argmax_sumexp(dlog.as<float>(), S, c.vocab, dpart.as<float>(), darg.as<int32_t>() + (size_t)step * 16, dsum.as<float>() + (size_t)step * 16, st);
    CUDA_CHECK(cudaMemcpy2DAsync(dcat.as<uint8_t>() + (size_t)H * 2, (size_t)2 * H * 2, dhn.p, (size_t)H * 2, (size_t)H * 2, S,
                                 cudaMemcpyDeviceToDevice, st));
  }
}

GemmaEngine::GemmaEngine(const std::string& ckpt_dir, const std::string& assistant_dir, int64_t max_ctx, int max_batch) : p_(std::make_unique<Impl>()) {
  Impl& I = *p_;
  I.c = GemmaConfig::load(ckpt_dir);
  const GemmaConfig& c = I.c;
  CUDA_CHECK(cudaStreamCreateWithFlags(&I.st, cudaStreamNonBlocking));
  I.blasp = std::make_unique<Blas>(I.st);
  I.max_ctx = max_ctx;
  I.max_batch = std::max(1, max_batch);
  const char* ch = getenv("HIVE_GEMMA_CHUNK");
  I.max_rows = std::max(64, ch ? atoi(ch) : 1024);
  if (const char* k = getenv("HIVE_GEMMA_SWA_KEEP")) I.swa_keep = std::max(0, atoi(k));
  if (const char* k = getenv("HIVE_GEMMA_GEMV_ROWS")) I.gemv_rows = std::max(16, atoi(k));
  I.load_target(ckpt_dir);
  if (!assistant_dir.empty()) I.load_assistant(assistant_dir);
  if (const char* k = getenv("HIVE_GEMMA_MTP_K")) I.mtp_k = std::max(0, std::min(7, atoi(k)));
  if (I.mtp_k == 0) I.mtp = false;
  // RoPE tables: sliding = default over head_dim; full = proportional (zero frequencies past the rotated fraction)
  {
    std::vector<float> a(c.head_dim / 2), b(c.g_head_dim / 2, 0.f);
    for (int i = 0; i < c.head_dim / 2; ++i) a[i] = 1.f / std::pow(c.theta_s, (2.f * i) / c.head_dim);
    const int nr = (int)(c.partial_f * c.g_head_dim / 2);
    for (int i = 0; i < nr; ++i) b[i] = 1.f / std::pow(c.theta_f, (2.f * i) / c.g_head_dim);
    I.inv_s.alloc(a.size() * 4); I.inv_f.alloc(b.size() * 4);
    CUDA_CHECK(cudaMemcpy(I.inv_s.p, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(I.inv_f.p, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  }
  // work buffers
  const int R = I.max_rows, H = c.hidden, QW = c.heads * std::max(c.head_dim, c.g_head_dim);
  const int KW = std::max(c.kv_heads * c.head_dim, c.g_kv_heads * c.g_head_dim);
  auto A = [&](DevBuf& b, size_t n) { b.alloc(n); I.work_bytes += n; };
  A(I.x, (size_t)R * H * 2); A(I.hbuf, (size_t)R * H * 2); A(I.yb, (size_t)R * H * 2); A(I.hn, (size_t)R * H * 2);
  A(I.qraw, (size_t)R * QW * 2); A(I.qb, (size_t)R * QW * 2); A(I.ob, (size_t)R * QW * 2);
  A(I.kraw, (size_t)R * KW * 2); A(I.vraw, (size_t)R * KW * 2); A(I.kb, (size_t)R * KW * 2); A(I.vb, (size_t)R * KW * 2);
  A(I.gb, (size_t)R * c.inter * 2); A(I.ub, (size_t)R * c.inter * 2); A(I.act, (size_t)R * c.inter * 2);
  size_t maxw = 0;
  for (const LayerW& w : I.L) for (const W4Mat* m : {&w.q, &w.k, &w.v, &w.o, &w.gate, &w.up, &w.down}) maxw = std::max(maxw, (size_t)m->N * m->K);
  A(I.deq, maxw * 2);
  A(I.ws, (size_t)16 << 20);
  A(I.ids_d, (size_t)R * 4); A(I.pos_d, (size_t)R * 4); A(I.seqof_d, (size_t)R * 4);
  A(I.as_s, (size_t)64 * sizeof(AttnSeq)); A(I.as_f, (size_t)64 * sizeof(AttnSeq));
  A(I.logits_d, (size_t)32 * c.vocab * 4);
  CUDA_CHECK(cudaMallocHost(&I.logits_h, (size_t)32 * c.vocab * 4));
  CUDA_CHECK(cudaMallocHost(&I.stage_i.p, (size_t)3 * I.max_rows * 4));
  CUDA_CHECK(cudaMallocHost(&I.stage_seq, (size_t)2 * 64 * sizeof(AttnSeq)));
  I.use_graphs = !(getenv("HIVE_GEMMA_GRAPHS") && strcmp(getenv("HIVE_GEMMA_GRAPHS"), "0") == 0);
  if (I.mtp) {
    const int DH = I.d_hidden, S = 16;
    A(I.dcat, (size_t)S * 2 * H * 2); A(I.dx, (size_t)S * DH * 2); A(I.dh, (size_t)S * DH * 2); A(I.dq, (size_t)S * QW * 2);
    A(I.dob, (size_t)S * QW * 2); A(I.dy, (size_t)S * DH * 2); A(I.dg, (size_t)S * I.d_inter * 2); A(I.du, (size_t)S * I.d_inter * 2);
    A(I.dact, (size_t)S * I.d_inter * 2); A(I.dzn, (size_t)S * DH * 2); A(I.dlog, (size_t)S * c.vocab * 4); A(I.dhn, (size_t)S * H * 2);
    A(I.darg, (size_t)7 * 16 * 4); A(I.dsum, (size_t)7 * 16 * 4); A(I.dmax, (size_t)16 * 4); A(I.dones, (size_t)16 * 4); A(I.dpart, argmax_sumexp_part_floats(16) * 4);
    { std::vector<float> one(16, 1.f); CUDA_CHECK(cudaMemcpy(I.dones.p, one.data(), 64, cudaMemcpyHostToDevice)); }
  }
  // KV pools: HIVE_GEMMA_KV_MB (default: what is free now minus 512 MiB, split by token cost) — the VRAM limit has already taken the rest
  I.Ls = KvLayout{c.n_sliding, c.kv_heads, c.head_dim};
  I.Lf = KvLayout{c.n_full, c.g_kv_heads, c.g_head_dim};
  size_t fr = 0, tot = 0;
  CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
  // under HIVE_VRAM_LIMIT_MB the engine is built before hived's ballast: the pool is sized from the limit (limit − weights − work − CUDA
  //   context / cuBLAS / graphs estimate − the runtime reserve HIVE_CACHE_RESERVE_MB), never from the whole card's free memory
  const size_t margin = ((size_t)(getenv("HIVE_CACHE_RESERVE_MB") ? atof(getenv("HIVE_CACHE_RESERVE_MB")) : 512.0) << 20) + ((size_t)1200 << 20);
  size_t kv = fr > margin ? fr - margin : 0;
  if (const char* lim = getenv("HIVE_VRAM_LIMIT_MB")) {
    const size_t L = (size_t)(atof(lim) * 1048576), used = I.weights_bytes + I.work_bytes + margin;
    kv = std::min(kv, L > used ? L - used : 0);
  }
  if (const char* m = getenv("HIVE_GEMMA_KV_MB")) kv = (size_t)(atof(m) * 1048576);
  HIVE_CHECK(kv >= ((size_t)64 << 20), "gemma: no VRAM left for the KV pool under the limit");
  // full pages: every position of every sequence · sliding pages: at most window + keep per sequence — size the two pools so a context of
  //   max_ctx fits with its sliding window (the rest of the budget goes to more sequences in the same proportion)
  const double ps = (double)I.Ls.page_bytes(), pf = (double)I.Lf.page_bytes();
  const double per_seq_s = std::ceil((double)(c.window + I.swa_keep + kPage) / kPage) + 1, per_seq_f = std::ceil((double)max_ctx / kPage);
  const double unit = per_seq_s * ps + per_seq_f * pf;
  const double seqs = kv / unit;   // may be below one sequence's full need: the pool then holds what the budget allows
  I.n_pages_s = (int)std::max(4.0, std::floor(seqs * per_seq_s));
  I.n_pages_f = (int)std::max(4.0, std::floor((kv - I.n_pages_s * ps) / pf));
  I.pool_s.alloc((size_t)I.n_pages_s * I.Ls.page_bytes());
  I.pool_f.alloc((size_t)I.n_pages_f * I.Lf.page_bytes());
  for (int i = I.n_pages_s - 1; i >= 0; --i) I.free_s.push_back(I.pool_s.as<uint8_t>() + (size_t)i * I.Ls.page_bytes());
  for (int i = I.n_pages_f - 1; i >= 0; --i) I.free_f.push_back(I.pool_f.as<uint8_t>() + (size_t)i * I.Lf.page_bytes());
  I.host_limit = (size_t)(getenv("HIVE_GEMMA_HOST_KV_MB") ? atof(getenv("HIVE_GEMMA_HOST_KV_MB")) : 16384.0) * 1048576;
  fprintf(stderr, "[gemma] %d layers (%d sliding / %d full) · weights %.0f MiB · work %.0f MiB · KV pool %d sliding pages (%.1f MiB each) + %d full "
          "pages (%.2f MiB each) = %.0f MiB · chunk %d · MTP %s (k %d)\n", c.n_layers, c.n_sliding, c.n_full, I.weights_bytes / 1048576.0,
          I.work_bytes / 1048576.0, I.n_pages_s, I.Ls.page_bytes() / 1048576.0, I.n_pages_f, I.Lf.page_bytes() / 1048576.0,
          (I.pool_s.n + I.pool_f.n) / 1048576.0, I.max_rows, I.mtp ? "on" : "off", I.mtp_k);
}

GemmaEngine::~GemmaEngine() {
  if (p_ && p_->logits_h) cudaFreeHost(p_->logits_h);
  if (p_ && p_->embed_host) cudaFreeHost(p_->embed_host);
  if (p_ && p_->stage_i.p) cudaFreeHost(p_->stage_i.p);
  if (p_ && p_->stage_seq) cudaFreeHost(p_->stage_seq);
  if (p_) for (auto& [k, g] : p_->graphs) cudaGraphExecDestroy(g);
  if (p_) for (uint8_t* h : p_->host_pages) cudaFreeHost(h);
}

const GemmaConfig& GemmaEngine::cfg() const { return p_->c; }
const Placement& GemmaEngine::placement() const { return p_->place; }
cudaStream_t GemmaEngine::stream() const { return p_->st; }
const float* GemmaEngine::logits_dev() const { return p_->logits_d.as<float>(); }
float* GemmaEngine::logits_host() { return p_->logits_h; }
void GemmaEngine::set_host_logits(bool on) { p_->host_logits = on; }
long GemmaEngine::graph_captures() const { return p_->n_captures; }
bool GemmaEngine::mtp_on() const { return p_->mtp; }
int GemmaEngine::mtp_k() const { return p_->mtp ? p_->mtp_k : 0; }
// Every decode / verify / draft graph the daemon can ask for, captured and uploaded before serving: the start-up VRAM plan (and the
//   daemon's limit reserve, rebased at ready) then already contains their memory. Captured lazily they appeared while serving, outside
//   the limit's accounting (measured 2026-10-18 beside DeepSeek at 72 GB: the process grew from 12,982 to 13,518 MiB under a 13,000 MiB
//   limit during the live harnesses). Shapes: S ≤ max_batch sequences of 1 … max_k + 1 rows, R ≤ 16 rows in all; logits on and off.
void GemmaEngine::warmup(int max_batch, int max_k) {
  Impl& I = *p_;
  if (!I.use_graphs) return;
  const int kr = std::max(1, max_k + 1);
  for (int S = 1; S <= std::min(max_batch, 16); ++S)
    for (int q = 1; q <= kr; ++q)
      for (int R = S + q - 1; R <= std::min(S * q, 16); ++R)
        for (int L = 0; L < 2; ++L) I.fwd_graph(R, S, q, L != 0);
  if (I.mtp)
    for (int S = 1; S <= std::min(max_batch, 16); ++S)
      for (int k = 1; k <= std::min(max_k, 7); ++k) I.draft_graph(S, k);
  CUDA_CHECK(cudaStreamSynchronize(I.st));
}

GemmaEngine::MemStats GemmaEngine::mem() const {
  const Impl& I = *p_;
  MemStats m;
  m.weights = I.weights_bytes; m.kv_pool = I.pool_s.n + I.pool_f.n; m.work = I.work_bytes;
  m.pages_s = I.n_pages_s; m.pages_f = I.n_pages_f; m.free_s = (int)I.free_s.size(); m.free_f = (int)I.free_f.size();
  m.host_pages = (int)I.host_pages.size();
  return m;
}

void GemmaEngine::init_seq(GemmaSeq& s, int64_t cap) {
  Impl& I = *p_;
  s.max_ctx = std::min<int64_t>(cap, I.max_ctx);
  const size_t entries = (size_t)((s.max_ctx + kPage - 1) / kPage + 1);
  s.tab_s.alloc(entries * sizeof(void*)); s.tab_f.alloc(entries * sizeof(void*));
  CUDA_CHECK(cudaMemsetAsync(s.tab_s.p, 0, s.tab_s.n, I.st)); CUDA_CHECK(cudaMemsetAsync(s.tab_f.p, 0, s.tab_f.n, I.st));
  s.h_last.alloc((size_t)I.c.hidden * 2); s.h_rows.alloc((size_t)16 * I.c.hidden * 2);
  s.tb_s = I.Ls.token_bytes(); s.tb_f = I.Lf.token_bytes(); s.h_bytes = (size_t)I.c.hidden * 2;
  reset_seq(s);
}
void GemmaEngine::reset_seq(GemmaSeq& s) {
  release_kv(s);
  s.pos = 0; s.tokens.clear(); s.h_last_valid = false; s.broken = false;
}
bool GemmaEngine::rewind(GemmaSeq& s, int64_t n) {
  Impl& I = *p_;
  if (n <= 0 || n >= s.pos || s.broken) return false;
  if (s.swa_lo > std::max<int64_t>(0, n - I.c.window)) return false;   // a sliding page the rows after n would read is already released
  const int keep = (int)((n + kPage - 1) / kPage);   // pages [0, keep) hold the positions below n (the last one partly — later writes overwrite it)
  for (int i = keep; i < (int)s.pages_f.size(); ++i) I.free_page(s.pages_f[i], false);
  for (int i = keep; i < (int)s.pages_s.size(); ++i) I.free_page(s.pages_s[i], true);
  if ((int)s.pages_f.size() > keep) s.pages_f.resize(keep);
  if ((int)s.pages_s.size() > keep) s.pages_s.resize(keep);
  s.pos = n;
  s.tokens.resize((size_t)n);
  s.h_last_valid = false;   // the drafter's seed was the old last row
  return true;
}
void GemmaEngine::release_kv(GemmaSeq& s) {
  Impl& I = *p_;
  I.release(s);
  if (s.tab_s.p) CUDA_CHECK(cudaMemsetAsync(s.tab_s.p, 0, s.tab_s.n, I.st));
  if (s.tab_f.p) CUDA_CHECK(cudaMemsetAsync(s.tab_f.p, 0, s.tab_f.n, I.st));
  s.pos = 0; s.tokens.clear(); s.h_last_valid = false;
}

void GemmaEngine::prefill(GemmaSeq& s, const int32_t* ids, int T, float* logits) {
  Impl& I = *p_;
  I.kv_reclaim_ = kv_reclaim;
  I.layer_boundary = layer_boundary; I.layer_boundary_due = layer_boundary_due;
  for (int off = 0; off < T; off += I.max_rows) {
    const int n = std::min(I.max_rows, T - off);
    const bool last = off + n >= T;
    I.run({{&s, ids + off, n}}, false, last ? logits : nullptr, false);
  }
}
void GemmaEngine::decode(GemmaSeq* const* seqs, const int32_t* ids, int M, float* logits) {
  Impl& I = *p_;
  I.kv_reclaim_ = kv_reclaim;
  std::vector<Impl::Part> parts;
  for (int i = 0; i < M; ++i) parts.push_back({seqs[i], ids + i, 1});
  I.run(parts, true, logits, false);
}
void GemmaEngine::verify_batch(GemmaSeq* const* seqs, const int* M, int S, const int32_t* ids, float* logits) {
  Impl& I = *p_;
  I.kv_reclaim_ = kv_reclaim;
  std::vector<Impl::Part> parts;
  int off = 0;
  I.last_verify.clear(); I.last_verify_rows.clear();
  for (int i = 0; i < S; ++i) {
    HIVE_CHECK(M[i] >= 1 && M[i] <= 16, "gemma: verify rows per sequence");
    parts.push_back({seqs[i], ids + off, M[i]}); off += M[i];
    I.last_verify.push_back(seqs[i]); I.last_verify_rows.push_back(M[i]);
  }
  I.run(parts, true, logits, true);
}
void GemmaEngine::rollback_batch(const int* n_keep, int S) {
  Impl& I = *p_;
  HIVE_CHECK(S == (int)I.last_verify.size(), "gemma: rollback without its verify");
  for (int i = 0; i < S; ++i) {
    GemmaSeq& s = *I.last_verify[i];
    const int m = I.last_verify_rows[i], keep = n_keep[i];
    HIVE_CHECK(keep >= 1 && keep <= m, "gemma: rollback rows");
    if (keep < m) {
      s.pos -= m - keep;
      s.tokens.resize(s.tokens.size() - (m - keep));
      CUDA_CHECK(cudaMemcpyAsync(s.h_last.p, s.h_rows.as<uint8_t>() + (size_t)(keep - 1) * I.c.hidden * 2, (size_t)I.c.hidden * 2,
                                 cudaMemcpyDeviceToDevice, I.st));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  I.last_verify.clear(); I.last_verify_rows.clear();
}

// Frozen-KV MTP: seed (token not yet in the KV, final-normed hidden of the last processed row) at rope position pos - 1; every step reads the
//   target's last sliding / full layer KV (keys 0 .. pos - 1), the chain feeds back post_projection(norm(z)).
void GemmaEngine::mtp_draft_batch(GemmaSeq* const* seqs, const int32_t* toks, int S, int k, int32_t* drafts, float* conf, int* n_out) {
  Impl& I = *p_;
  const GemmaConfig& c = I.c;
  HIVE_CHECK(I.mtp && S >= 1 && S <= 16 && k >= 1 && k <= 7, "gemma: draft batch");
  const int H = c.hidden;
  cudaStream_t st = I.st;
  AttnSeq* as = reinterpret_cast<AttnSeq*>(I.stage_seq);
  AttnSeq* af = as + S;
  int32_t* ids = I.stage_i.as_host();
  int32_t* pos = ids + S;
  for (int i = 0; i < S; ++i) {
    HIVE_CHECK(seqs[i]->h_last_valid && seqs[i]->pos >= 1, "gemma: draft without a seed");
    as[i].pages = seqs[i]->tab_s.as<const uint8_t*>(); af[i].pages = seqs[i]->tab_f.as<const uint8_t*>();
    as[i].q_row0 = af[i].q_row0 = i; as[i].n_q = af[i].n_q = 1; as[i].q_pos0 = af[i].q_pos0 = (int)(seqs[i]->pos - 1);
    pos[i] = (int32_t)(seqs[i]->pos - 1);
    ids[i] = toks[i];
    CUDA_CHECK(cudaMemcpyAsync(I.dcat.as<uint8_t>() + ((size_t)i * 2 * H + H) * 2, seqs[i]->h_last.p, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
  }
  CUDA_CHECK(cudaMemcpyAsync(I.ids_d.p, ids, (size_t)S * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(I.pos_d.p, pos, (size_t)S * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(I.as_s.p, as, (size_t)S * sizeof(AttnSeq), cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(I.as_f.p, af, (size_t)S * sizeof(AttnSeq), cudaMemcpyHostToDevice, st));
  if (I.use_graphs) CUDA_CHECK(cudaGraphLaunch(I.draft_graph(S, k), st));
  else I.draft_chain(S, k);
  std::vector<int32_t> arg((size_t)k * 16);
  std::vector<float> sm((size_t)k * 16);
  CUDA_CHECK(cudaMemcpyAsync(arg.data(), I.darg.p, (size_t)k * 16 * 4, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaMemcpyAsync(sm.data(), I.dsum.p, (size_t)k * 16 * 4, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  for (int i = 0; i < S; ++i) {
    for (int step = 0; step < k; ++step) {
      drafts[(size_t)i * 7 + step] = arg[(size_t)step * 16 + i];
      const float z = sm[(size_t)step * 16 + i];
      conf[(size_t)i * 7 + step] = z > 0 ? 1.f / z : 0.f;   // p(argmax) = 1 / Σ exp(l - max)
    }
    n_out[i] = k;
  }
}

int GemmaEngine::mtp_draft(GemmaSeq& s, int32_t tok, int k, int32_t* drafts, float* conf) {
  GemmaSeq* q[1] = {&s};
  int32_t d[7]; float cf[7]; int n = 0;
  mtp_draft_batch(q, &tok, 1, k, d, cf, &n);
  for (int i = 0; i < n; ++i) { drafts[i] = d[i]; conf[i] = cf[i]; }
  return n;
}

// ---- session images ----
// pieces: [header (host)] [h_last] [full pages 0 .. ] [sliding pages from swa_lo] — page pieces hold the filled prefix (tokens × token bytes)
void GemmaSeq::save_plan(std::vector<Piece>& out, std::vector<std::vector<uint8_t>>& hold) const {
  out.clear();
  std::vector<uint8_t> hdr(8 * 4);
  const int64_t v[4] = {pos, swa_lo, (int64_t)pages_f.size(), (int64_t)pages_s.size()};
  memcpy(hdr.data(), v, sizeof v);
  hold.push_back(std::move(hdr));
  out.push_back({hold.back().data(), nullptr, hold.back().size(), false});
  out.push_back({nullptr, h_last.p, h_bytes, false});
  auto filled = [&](int i) { return (size_t)std::max<int64_t>(0, std::min<int64_t>(kPage, pos - (int64_t)i * kPage)); };
  for (int i = 0; i < (int)pages_f.size(); ++i) out.push_back({nullptr, pages_f[i], filled(i) * tb_f, true});
  for (int i = (int)(swa_lo / kPage); i < (int)pages_s.size(); ++i) out.push_back({nullptr, pages_s[i], filled(i) * tb_s, true});
}

void GemmaEngine::load_views(GemmaSeq& s, const std::vector<std::pair<const uint8_t*, size_t>>& v,
                             const std::function<void(void*, const uint8_t*, size_t)>& h2d) {
  Impl& I = *p_;
  I.kv_reclaim_ = kv_reclaim;
  HIVE_CHECK(v.size() >= 2 && v[0].second == 32, "gemma: image header");
  int64_t hv[4];
  memcpy(hv, v[0].first, sizeof hv);
  release_kv(s);
  s.pos = hv[0]; s.swa_lo = hv[1];
  const int nf = (int)hv[2], ns = (int)hv[3];
  const int lo = (int)(s.swa_lo / kPage);
  HIVE_CHECK((int)v.size() == 2 + nf + (ns - lo), "gemma: image pieces");
  auto copy = [&](void* d, const uint8_t* h, size_t n) { if (!n) return; if (h2d) h2d(d, h, n); else CUDA_CHECK(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, I.st)); };
  copy(s.h_last.p, v[1].first, v[1].second);
  for (int i = 0; i < nf; ++i) { s.pages_f.push_back(I.alloc_page(false)); copy(s.pages_f.back(), v[2 + i].first, v[2 + i].second); }
  s.pages_s.assign(ns, nullptr);
  for (int i = lo; i < ns; ++i) { s.pages_s[i] = I.alloc_page(true); copy(s.pages_s[i], v[2 + nf + i - lo].first, v[2 + nf + i - lo].second); }
  I.push_tables(s, 0, 0);
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  s.h_last_valid = s.pos > 0;
}

}  // namespace hive::gemma
