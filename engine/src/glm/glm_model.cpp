// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/glm/glm_model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "hive/vram_limit.h"
#include "nlohmann/json.hpp"

namespace hive::glm {

using json = nlohmann::json;

GlmConfig GlmConfig::load(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) throw std::runtime_error("no config.json in " + dir);
  json all = json::parse(f);
  const json& t = all.contains("text_config") ? all["text_config"] : all;
  GlmConfig c;
  auto gi = [&](const char* k, int& v) { if (t.contains(k) && t[k].is_number()) v = t[k].get<int>(); };
  auto gf = [&](const char* k, float& v) { if (t.contains(k) && t[k].is_number()) v = t[k].get<float>(); };
  gi("hidden_size", c.hidden); gi("num_hidden_layers", c.n_layers); gi("vocab_size", c.vocab); gi("num_attention_heads", c.n_heads);
  gi("q_lora_rank", c.q_lora); gi("kv_lora_rank", c.kv_lora); gi("qk_head_dim", c.qk_head); gi("v_head_dim", c.v_head);
  gi("index_n_heads", c.index_heads); gi("index_head_dim", c.index_dim); gi("index_topk", c.index_topk); gi("index_kpool", c.index_kpool);
  gi("intermediate_size", c.dense_inter); gi("moe_intermediate_size", c.moe_inter); gi("n_routed_experts", c.n_routed);
  gi("num_experts_per_tok", c.n_act); gi("n_shared_experts", c.n_shared); gi("hc_mult", c.hc); gi("hc_sinkhorn_iters", c.sinkhorn_iters);
  gi("num_nextn_predict_layers", c.n_mtp);
  gf("routed_scaling_factor", c.routed_scale); gf("swiglu_limit", c.swiglu_limit); gf("rms_norm_eps", c.rms_eps); gf("hc_eps", c.hc_eps);
  if (t.contains("eos_token_id") && t["eos_token_id"].is_array()) c.eos_ids = t["eos_token_id"].get<std::vector<int>>();
  if (t.contains("linear_attn_config")) {
    const json& la = t["linear_attn_config"];
    if (la.contains("num_heads")) c.kda_heads = la["num_heads"].get<int>();
    if (la.contains("head_dim")) c.kda_dim = la["head_dim"].get<int>();
    if (la.contains("short_conv_kernel_size")) c.conv_k = la["short_conv_kernel_size"].get<int>();
    if (la.contains("gate_lower_bound")) c.kda_lower_bound = la["gate_lower_bound"].get<float>();
    c.is_kda.assign(c.n_layers, false);
    for (int l : la["kda_layers"].get<std::vector<int>>()) if (l < c.n_layers) c.is_kda[l] = true;
  } else {
    throw std::runtime_error("config: linear_attn_config missing (not a GLM-5.3-Flash checkpoint?)");
  }
  // mlp_layer_types lists only the first entries ([dense, sparse, sparse]); first_k_dense_replace is authoritative
  int first_dense = 3;
  if (t.contains("first_k_dense_replace")) first_dense = t["first_k_dense_replace"].get<int>();
  c.is_moe.assign(c.n_layers, false); c.moe_index.assign(c.n_layers, -1);
  for (int l = first_dense; l < c.n_layers; ++l) { c.is_moe[l] = true; c.moe_index[l] = c.n_moe++; }
  if (all.contains("vision_config")) {
    const json& v = all["vision_config"];
    auto vi = [&](const char* k, int& x) { if (v.contains(k)) x = v[k].get<int>(); };
    vi("depth", c.vision_layers); vi("hidden_size", c.vision_dim); vi("num_heads", c.vision_heads); vi("intermediate_size", c.vision_inter);
    vi("patch_size", c.vision_patch); vi("spatial_merge_size", c.vision_merge); vi("temporal_patch_size", c.vision_temporal);
    vi("out_hidden_size", c.vision_out); vi("projection_intermediate_size", c.vision_proj_inter);
  }
  auto ai = [&](const char* k, int& x) { if (all.contains(k)) x = all[k].get<int>(); };
  ai("image_token_id", c.image_token_id); ai("video_token_id", c.video_token_id); ai("image_start_token_id", c.image_start_id);
  ai("image_end_token_id", c.image_end_id); ai("video_start_token_id", c.video_start_id); ai("video_end_token_id", c.video_end_id);
  HIVE_CHECK(c.hc == 4, "GLM: hc_mult must be 4 (hyper-connection kernels)");
  HIVE_CHECK(c.moe_inter % 32 == 0 && c.hidden % 32 == 0, "GLM: dims must be multiples of 32");
  return c;
}

namespace {
const std::string P = "model.language_model.";
std::vector<uint8_t> host_copy(const TensorInfo& t) { return std::vector<uint8_t>(t.data, t.data + t.nbytes); }
}  // namespace

Mat GlmModel::mat(const std::string& name) {
  const TensorInfo& t = ckpt_.get(name);
  HIVE_CHECK(t.dtype == "BF16" && t.shape.size() == 2, "GLM: expected a BF16 matrix: " + name + " (" + t.dtype + ")");
  Mat m; m.out = (int)t.shape[0]; m.in = (int)t.shape[1];
  m.buf.alloc(t.nbytes);
  CUDA_CHECK(cudaMemcpy(m.buf.p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  vram_bytes_ += t.nbytes;
  return m;
}
DevBuf GlmModel::vec_bf16(const std::string& name) {
  const TensorInfo& t = ckpt_.get(name);
  HIVE_CHECK(t.dtype == "BF16", "GLM: expected BF16: " + name);
  DevBuf b(t.nbytes);
  CUDA_CHECK(cudaMemcpy(b.p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  vram_bytes_ += t.nbytes;
  return b;
}
DevBuf GlmModel::vec_f32(const std::string& name) {
  const TensorInfo& t = ckpt_.get(name);
  std::vector<float> h((size_t)t.numel());
  if (t.dtype == "F32") std::memcpy(h.data(), t.data, t.nbytes);
  else if (t.dtype == "BF16") { const bf16* s = reinterpret_cast<const bf16*>(t.data); for (size_t i = 0; i < h.size(); ++i) h[i] = bf2f(s[i]); }
  else throw std::runtime_error("GLM: unexpected dtype " + t.dtype + " for " + name);
  DevBuf b(h.size() * 4);
  CUDA_CHECK(cudaMemcpy(b.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  vram_bytes_ += h.size() * 4;
  return b;
}

void GlmModel::load_hc(const std::string& p, const char* which, HcW& w) {
  w.fn = vec_f32(p + "hc_" + which + "_fn");
  w.base = vec_f32(p + "hc_" + which + "_base");
  w.scale = vec_f32(p + "hc_" + which + "_scale");
}

void GlmModel::load_kda(const std::string& p, KdaW& w) {
  const std::string a = p + "self_attn.";
  w.q = mat(a + "q_proj.weight"); w.k = mat(a + "k_proj.weight"); w.v = mat(a + "v_proj.weight");
  w.f_a = mat(a + "f_a_proj.weight"); w.f_b = mat(a + "f_b_proj.weight");
  w.g_a = mat(a + "g_a_proj.weight"); w.g_b = mat(a + "g_b_proj.weight");
  w.b = mat(a + "b_proj.weight"); w.o = mat(a + "o_proj.weight");
  // conv weights q, k, v (fp32 [8192, 1, 4] each) stacked → [24576, 4]
  std::vector<float> cw;
  for (const char* n : {"q_conv1d", "k_conv1d", "v_conv1d"}) {
    const TensorInfo& t = ckpt_.get(a + n + ".weight");
    HIVE_CHECK(t.dtype == "F32", "GLM: conv1d weights F32");
    const float* s = reinterpret_cast<const float*>(t.data);
    cw.insert(cw.end(), s, s + t.numel());
  }
  w.conv_w.alloc(cw.size() * 4);
  CUDA_CHECK(cudaMemcpy(w.conv_w.p, cw.data(), cw.size() * 4, cudaMemcpyHostToDevice));
  vram_bytes_ += cw.size() * 4;
  w.A_log = vec_f32(a + "A_log"); w.dt_bias = vec_f32(a + "dt_bias");
  w.o_norm = vec_bf16(a + "o_norm.weight");
}

Mat GlmModel::mat_fp8ok(const std::string& name) {
  if (!fp8_ || !fp8_->has(name) || !fp8_->has(name + "_scale_inv")) return mat(name);
  const TensorInfo& t = fp8_->get(name);
  const TensorInfo& sc = fp8_->get(name + "_scale_inv");
  HIVE_CHECK(t.dtype == "F8_E4M3" && t.shape.size() == 2 && sc.dtype == "F32", "GLM fp8: unexpected tensor " + name);
  Mat m; m.fp8 = true; m.out = (int)t.shape[0]; m.in = (int)t.shape[1];
  HIVE_CHECK((int64_t)sc.shape[0] == (m.out + 127) / 128 && (int64_t)sc.shape[1] == (m.in + 127) / 128, "GLM fp8: scale shape " + name);
  m.f8w.alloc(t.nbytes); CUDA_CHECK(cudaMemcpy(m.f8w.p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  m.f8s.alloc(sc.nbytes); CUDA_CHECK(cudaMemcpy(m.f8s.p, sc.data, sc.nbytes, cudaMemcpyHostToDevice));
  // shape must match the BF16 tensor it replaces
  if (ckpt_.has(name)) { const TensorInfo& b = ckpt_.get(name); HIVE_CHECK(b.shape == t.shape, "GLM fp8: shape differs from base " + name); }
  vram_bytes_ += t.nbytes + sc.nbytes; ++n_fp8_;
  return m;
}

void GlmModel::load_dsa(const std::string& p, DsaW& w) {
  const std::string a = p + "self_attn.";
  w.q_a = mat_fp8ok(a + "q_a_proj.weight"); w.q_b = mat_fp8ok(a + "q_b_proj.weight");
  w.kv_a = mat_fp8ok(a + "kv_a_proj_with_mqa.weight"); w.kv_b = mat(a + "kv_b_proj.weight"); w.o = mat_fp8ok(a + "o_proj.weight");
  w.q_a_norm = vec_bf16(a + "q_a_layernorm.weight"); w.kv_a_norm = vec_bf16(a + "kv_a_layernorm.weight");
  const std::string i = a + "indexer.";
  w.idx_wq_b = mat(i + "wq_b.weight"); w.idx_wk = mat(i + "wk.weight"); w.idx_weights = mat(i + "weights_proj.weight");
  w.idx_gate = mat(i + "index_kpool_compress_gate");
  w.idx_k_norm_w = vec_bf16(i + "k_norm.weight"); w.idx_k_norm_b = vec_bf16(i + "k_norm.bias"); w.idx_ape = vec_bf16(i + "index_kpool_compress_ape");
}

void GlmModel::load_moe(const std::string& p, MoeW& w) {
  w.gate = mat(p + "mlp.gate.weight"); w.bias = vec_f32(p + "mlp.gate.e_score_correction_bias");
  w.sh_gate = mat_fp8ok(p + "mlp.shared_experts.gate_proj.weight"); w.sh_up = mat_fp8ok(p + "mlp.shared_experts.up_proj.weight");
  w.sh_down = mat_fp8ok(p + "mlp.shared_experts.down_proj.weight");
}

// VRAM a GPU-placed layer takes, from the checkpoint headers (the tensors GlmModel loads for it: FP8 where the Z.ai copy has one)
size_t GlmModel::layer_bytes(int l) {
  const std::string p = P + "layers." + std::to_string(l) + ".";
  size_t b = 0;
  for (const std::string& n : ckpt_.names_with_prefix(p)) {
    if (n.find(".mlp.experts.") != std::string::npos || n.find("input_scale") != std::string::npos) continue;
    if (fp8_ && fp8_->has(n) && fp8_->has(n + "_scale_inv")) { b += fp8_->get(n).nbytes + fp8_->get(n + "_scale_inv").nbytes; continue; }
    const TensorInfo& t = ckpt_.get(n);
    b += (t.dtype == "BF16" && (n.find("hc_") != std::string::npos || n.find("A_log") != std::string::npos || n.find("dt_bias") != std::string::npos))
             ? t.nbytes * 2 : t.nbytes;  // vec_f32 widens these to fp32
  }
  return b;
}

// HIVE_GLM_GPU_LAYERS=N (0 ≤ N ≤ n_layers): layers N.. run on the CPU. Unset, with HIVE_VRAM_LIMIT_MB: as many leading layers as fit beside
//   the parts that stay on the GPU (lm_head, the NextN layer when loaded) and an estimate of the rest of the startup (hive/vram_limit.h
//   fixed_bytes at the lean knobs, the cache reserve, a minimum expert cache of HIVE_GLM_MIN_SLOTS slots, default 64) — an estimate: hived's
//   own plan after the load works from the measured weights. Otherwise everything on the GPU (unchanged behaviour).
int GlmModel::gpu_layers_for_limit(bool load_mtp) {
  const GlmConfig& c = cfg_;
  if (const char* v = getenv("HIVE_GLM_GPU_LAYERS"); v && *v) return std::max(0, std::min(c.n_layers, atoi(v)));
  const size_t L = vram_limit::limit_bytes();
  if (L == 0) return c.n_layers;
  const bool mtp = load_mtp && !(getenv("HIVE_GLM_MTP_K") && strcmp(getenv("HIVE_GLM_MTP_K"), "0") == 0);
  vram_limit::Knobs k;
  k.mtp = mtp; k.max_batch = 1; k.max_sessions = 2; k.max_chunk = 4096; k.max_ctx = 32768;
  const double reserve = (getenv("HIVE_CACHE_RESERVE_MB") ? atof(getenv("HIVE_CACHE_RESERVE_MB")) : 2400.0) * vram_limit::kMiB;
  const int min_slots = getenv("HIVE_GLM_MIN_SLOTS") ? std::max(0, atoi(getenv("HIVE_GLM_MIN_SLOTS"))) : 64;
  double room = (double)L - (double)vram_limit::kCtxFallback - vram_limit::fixed_bytes(vram_limit::Family::Glm, k) - reserve - min_slots * 13.5 * vram_limit::kMiB;
  room -= (double)ckpt_.get("lm_head.weight").nbytes + 64.0 * vram_limit::kMiB;
  if (mtp) room -= 7.2 * 1024 * vram_limit::kMiB;  // NextN layer with its FP8 experts (6.95 GiB measured, docs/vram-footprint.md §3.2)
  // prefill streaming (default on): one staging copy per layer kind in VRAM — counted as the two largest layers
  if (!(getenv("HIVE_GLM_STREAM_PREFILL") && strcmp(getenv("HIVE_GLM_STREAM_PREFILL"), "0") == 0)) {
    size_t big[2] = {0, 0};
    for (int l = 0; l < c.n_layers; ++l) { const size_t b = layer_bytes(l); if (b > big[0]) { big[1] = big[0]; big[0] = b; } else if (b > big[1]) big[1] = b; }
    room -= (double)(big[0] + big[1]);
  }
  int n = 0;
  for (; n < c.n_layers; ++n) {
    const double b = (double)layer_bytes(n);
    if (b > room) break;
    room -= b;
  }
  fprintf(stderr, "[glm] VRAM limit %.0f MiB: %d of %d layers on the GPU (estimate from the checkpoint headers; HIVE_GLM_GPU_LAYERS overrides)\n",
          L / 1048576.0, n, c.n_layers);
  return n;
}

void GlmModel::load_layer(int l, LayerW& L) {
  const GlmConfig& c = cfg_;
  const std::string p = P + "layers." + std::to_string(l) + ".";
  L.kda = c.is_kda[l]; L.moe = c.is_moe[l];
  L.in_norm = vec_bf16(p + "input_layernorm.weight"); L.post_norm = vec_bf16(p + "post_attention_layernorm.weight");
  load_hc(p, "attn", L.hc_attn); load_hc(p, "ffn", L.hc_ffn);
  if (L.kda) load_kda(p, L.kda_w); else load_dsa(p, L.dsa_w);
  if (L.moe) load_moe(p, L.moe_w);
  else {
    const char* names[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int j = 0; j < 3; ++j) {
      const std::string b = p + "mlp." + names[j] + ".";
      const TensorInfo& tw = ckpt_.get(b + "weight");
      const TensorInfo& ts = ckpt_.get(b + "weight_scale");
      const TensorInfo& tg = ckpt_.get(b + "weight_scale_2");
      HIVE_CHECK(tw.dtype == "U8" && ts.dtype == "F8_E4M3" && tg.dtype == "F32", "GLM: dense MLP must be NVFP4: " + b);
      L.dense.w[j].alloc(tw.nbytes); CUDA_CHECK(cudaMemcpy(L.dense.w[j].p, tw.data, tw.nbytes, cudaMemcpyHostToDevice));
      L.dense.s[j].alloc(ts.nbytes); CUDA_CHECK(cudaMemcpy(L.dense.s[j].p, ts.data, ts.nbytes, cudaMemcpyHostToDevice));
      std::memcpy(&L.dense.g[j], tg.data, 4);
      vram_bytes_ += tw.nbytes + ts.nbytes;
    }
  }
}

// Every device buffer of a layer, in a fixed order (the staging copy of a kind has the same buffers with the same sizes)
void GlmModel::bufs_of(LayerW& L, std::vector<DevBuf*>& o) {
  auto mat = [&](Mat& m) { o.push_back(&m.buf); o.push_back(&m.f8w); o.push_back(&m.f8s); };
  for (DevBuf* b : {&L.in_norm, &L.post_norm, &L.hc_attn.fn, &L.hc_attn.base, &L.hc_attn.scale, &L.hc_ffn.fn, &L.hc_ffn.base, &L.hc_ffn.scale}) o.push_back(b);
  KdaW& k = L.kda_w;
  for (Mat* m : {&k.q, &k.k, &k.v, &k.f_a, &k.f_b, &k.g_a, &k.g_b, &k.b, &k.o}) mat(*m);
  for (DevBuf* b : {&k.conv_w, &k.A_log, &k.dt_bias, &k.o_norm}) o.push_back(b);
  DsaW& d = L.dsa_w;
  for (Mat* m : {&d.q_a, &d.q_b, &d.kv_a, &d.kv_b, &d.o, &d.idx_wq_b, &d.idx_wk, &d.idx_weights, &d.idx_gate}) mat(*m);
  for (DevBuf* b : {&d.q_a_norm, &d.kv_a_norm, &d.idx_k_norm_w, &d.idx_k_norm_b, &d.idx_ape}) o.push_back(b);
  for (int j = 0; j < 3; ++j) { o.push_back(&L.dense.w[j]); o.push_back(&L.dense.s[j]); }
  MoeW& m = L.moe_w;
  for (Mat* x : {&m.gate, &m.sh_gate, &m.sh_up, &m.sh_down}) mat(*x);
  o.push_back(&m.bias);
}

// Load layer l the usual way (temporary VRAM), keep its buffers in pinned RAM; the first layer of each kind stays as that kind's staging copy.
void GlmModel::keep_streamed(int l) {
  const size_t v0 = vram_bytes_;
  auto tmp = std::make_unique<LayerW>();
  load_layer(l, *tmp);
  std::vector<DevBuf*> bufs;
  bufs_of(*tmp, bufs);
  Streamed& S = streamed_[l];
  for (DevBuf* b : bufs) {
    uint8_t* h = nullptr;
    if (b->n) { CUDA_CHECK(cudaMallocHost(&h, b->n)); CUDA_CHECK(cudaMemcpy(h, b->p, b->n, cudaMemcpyDeviceToHost)); }
    S.blobs.push_back({h, b->n});
  }
  for (int j = 0; j < 3; ++j) S.g[j] = tmp->dense.g[j];
  auto& st = staging_[tmp->kda][tmp->moe];
  if (!st) st = std::move(tmp);              // stays in VRAM as the staging copy of this kind
  else vram_bytes_ = v0;                     // freed again (tmp goes out of scope)
}

GlmModel::~GlmModel() {
  for (auto& S : streamed_) for (auto& b : S.blobs) if (b.first) cudaFreeHost(b.first);
}

size_t GlmModel::stream_bytes(int l) const {
  size_t b = 0;
  for (const auto& x : streamed_[l].blobs) b += x.second;
  return b;
}

void GlmModel::stream_in(int l, cudaStream_t st) {
  HIVE_CHECK(l >= n_gpu_ && stream_on_, "GLM stream_in: layer is not streamed");
  const GlmConfig& c = cfg_;
  LayerW& S = *staging_[c.is_kda[l]][c.is_moe[l]];
  std::vector<DevBuf*> bufs;
  bufs_of(S, bufs);
  const Streamed& src = streamed_[l];
  HIVE_CHECK(bufs.size() == src.blobs.size(), "GLM stream_in: layout");
  for (size_t i = 0; i < bufs.size(); ++i) {
    HIVE_CHECK(bufs[i]->n == src.blobs[i].second, "GLM stream_in: buffer size differs from the staging copy");
    if (src.blobs[i].second) CUDA_CHECK(cudaMemcpyAsync(bufs[i]->p, src.blobs[i].first, src.blobs[i].second, cudaMemcpyHostToDevice, st));
  }
  for (int j = 0; j < 3; ++j) S.dense.g[j] = src.g[j];
  for (int x = n_gpu_; x < c.n_layers; ++x) if (active_[x] == &S) active_[x] = nullptr;
  active_[l] = &S;
}

GlmModel::GlmModel(const std::string& dir, bool load_mtp) : cfg_(GlmConfig::load(dir)), dir_(dir), ckpt_(dir) {
  const GlmConfig& c = cfg_;
  // Z.ai FP8 tensors (DSA MLA projections, shared experts): HIVE_GLM_FP8_DIR = a directory with model.safetensors.index.json; "0" = off
  if (const char* fd = getenv("HIVE_GLM_FP8_DIR"); fd && *fd && std::string(fd) != "0") {
    fp8_ = std::make_unique<Checkpoint>(fd);
    fprintf(stderr, "[glm] fp8 tensors from %s\n", fd);
  }
  n_gpu_ = gpu_layers_for_limit(load_mtp);
  stream_on_ = !(getenv("HIVE_GLM_STREAM_PREFILL") && strcmp(getenv("HIVE_GLM_STREAM_PREFILL"), "0") == 0);
  active_.assign(c.n_layers, nullptr);
  streamed_.resize(c.n_layers);
  if (n_gpu_ < c.n_layers) fprintf(stderr, "[glm] layers %d..%d placed on the CPU (hybrid path)\n", n_gpu_, c.n_layers - 1);
  layers_.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; ++l) {
    LayerW& L = layers_[l];
    const std::string p = P + "layers." + std::to_string(l) + ".";
    L.kda = c.is_kda[l]; L.moe = c.is_moe[l];
    if (l >= n_gpu_) {  // CPU-placed: the hybrid CPU engine loads it; with prefill streaming its GPU form is kept in pinned RAM
      if (stream_on_) keep_streamed(l);
      continue;
    }
    load_layer(l, L);
    if ((l + 1) % 9 == 0 || l == c.n_layers - 1)
      fprintf(stderr, "[glm] layers 0..%d loaded · VRAM %.1f GiB\n", l, vram_bytes_ / 1073741824.0);
  }
  {
    const TensorInfo& e = ckpt_.get(P + "embed_tokens.weight");
    HIVE_CHECK(e.dtype == "BF16" && e.shape[0] == c.vocab && e.shape[1] == c.hidden, "GLM: embed_tokens shape");
    embed_host_.resize((size_t)e.numel());
    std::memcpy(embed_host_.data(), e.data, e.nbytes);
  }
  lm_head_ = mat("lm_head.weight");
  final_norm_ = vec_bf16(P + "norm.weight");
  if (load_mtp && c.n_mtp > 0) {
    mtp_ = std::make_unique<MtpW>();
    const std::string p = P + "layers." + std::to_string(c.n_layers) + ".";
    if (ckpt_.has(p + "eh_proj.weight")) {
      mtp_->enorm = vec_bf16(p + "enorm.weight"); mtp_->hnorm = vec_bf16(p + "hnorm.weight");
      mtp_->head_norm = vec_bf16(p + "shared_head.norm.weight");
      mtp_->in_norm = vec_bf16(p + "input_layernorm.weight"); mtp_->post_norm = vec_bf16(p + "post_attention_layernorm.weight");
      mtp_->eh_proj = mat(p + "eh_proj.weight");
      load_dsa(p, mtp_->dsa_w); load_moe(p, mtp_->moe_w);
      // routed experts: only the Z.ai FP8 copy is used (the NVFP4 checkpoint ships them in BF16 — 50 MB each); without it MTP is off
      const std::string e0 = p + "mlp.experts.0.gate_proj.weight";
      if (fp8_ && fp8_->has(e0) && fp8_->has(e0 + "_scale_inv")) {
        const int E = c.n_routed, I = c.moe_inter, H = c.hidden;
        const size_t wgu = (size_t)I * H, sgu = (size_t)((I + 127) / 128) * ((H + 127) / 128) * 4;
        const size_t wd = (size_t)H * I, sd = (size_t)((H + 127) / 128) * ((I + 127) / 128) * 4;
        const size_t per = wgu * 2 + wd + sgu * 2 + sd;
        mtp_->expert_arena.alloc((size_t)E * per);
        uint8_t* base = mtp_->expert_arena.as<uint8_t>();
        const char* names[3] = {"gate_proj", "up_proj", "down_proj"};
        for (int e = 0; e < E; ++e) {
          uint8_t* q = base + (size_t)e * per;
          std::array<Fp8BMat, 3> m{};
          for (int j = 0; j < 3; ++j) {
            const std::string b = p + "mlp.experts." + std::to_string(e) + "." + names[j] + ".weight";
            const TensorInfo& tw = fp8_->get(b);
            const TensorInfo& ts = fp8_->get(b + "_scale_inv");
            HIVE_CHECK(tw.dtype == "F8_E4M3" && ts.dtype == "F32", "GLM MTP expert must be FP8: " + b);
            const int N = (int)tw.shape[0], K = (int)tw.shape[1];
            HIVE_CHECK((j < 2 && N == I && K == H) || (j == 2 && N == H && K == I), "GLM MTP expert shape: " + b);
            CUDA_CHECK(cudaMemcpy(q, tw.data, tw.nbytes, cudaMemcpyHostToDevice));
            uint8_t* sq = q + tw.nbytes;
            CUDA_CHECK(cudaMemcpy(sq, ts.data, ts.nbytes, cudaMemcpyHostToDevice));
            m[j] = Fp8BMat{q, reinterpret_cast<const float*>(sq), N, K};
            q = sq + ts.nbytes;
          }
          mtp_->experts.push_back(m);
        }
        vram_bytes_ += (size_t)E * per;
        fprintf(stderr, "[glm] MTP: NextN layer loaded · %d routed experts FP8 in VRAM (%.2f GiB)\n", E, (double)E * per / 1073741824.0);
      } else {
        mtp_.reset();
        fprintf(stderr, "[glm] MTP off: the NextN layer's routed experts need the Z.ai FP8 copy (scripts/glm-fetch-fp8.py --mtp-experts)\n");
      }
    } else {
      mtp_.reset();
      fprintf(stderr, "[glm] no NextN layer in the checkpoint — speculative decoding off\n");
    }
  }
  fprintf(stderr, "[glm] model loaded: %d layers (%d KDA · %d DSA · %d MoE) · non-expert VRAM %.2f GiB · MTP %s · fp8 matrices %d\n", c.n_layers,
          (int)std::count(c.is_kda.begin(), c.is_kda.end(), true), c.n_layers - (int)std::count(c.is_kda.begin(), c.is_kda.end(), true),
          c.n_moe, vram_bytes_ / 1073741824.0, mtp_ ? "on" : "off", n_fp8_);
}

Nvfp4Mat GlmModel::dense_mat(int l, int which) const {
  const DenseMlpW& d = layer(l).dense;
  const int N = which == 2 ? cfg_.hidden : cfg_.dense_inter, K = which == 2 ? cfg_.dense_inter : cfg_.hidden;
  return Nvfp4Mat{d.w[which].as<uint8_t>(), d.s[which].as<uint8_t>(), d.g[which], N, K};
}

}  // namespace hive::glm
