// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next forward pass — see hive/qwen/qwen_engine.h. Layout follows GlmEngine (hive/glm/glm_engine.cpp): the same routed-expert
//   store, sequence-state handling, KV growth, layer yields and MTP contract; the layer body is this model's (hyper-connection gated
//   residual, Gated DeltaNet / QSA attention, PLE, softmax-routed MoE with a gated shared expert).
#include "hive/qwen/qwen_engine.h"
#include "hive/qwen/fn_gemv.h"
#include "hive/qwen/fn_moe.h"
#include "hive/glm/glm_moe_fp8.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "hive/cache_reserve.h"
#include "hive/clock.h"
#include "hive/glm/glm_decode.h"
#include "hive/glm/glm_kernels.h"
#include "hive/glm/glm_moe.h"
#include "hive/qwen/qwen_kernels.h"

namespace hive::qwen {

namespace {
bool env_on(const char* n) { const char* v = getenv(n); return v && *v && strcmp(v, "0") != 0; }
bool env_off(const char* n) { const char* v = getenv(n); return v && strcmp(v, "0") == 0; }
template <class T> T* pinned(size_t n) { void* p = nullptr; CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(n, 1) * sizeof(T))); return (T*)p; }
template <class T> T* mapped(size_t n) { void* p = nullptr; CUDA_CHECK(cudaHostAlloc(&p, std::max<size_t>(n, 1) * sizeof(T), cudaHostAllocMapped)); return (T*)p; }
}  // namespace

QwenEngine::QwenEngine(QwenModel& m, GlmExperts& x, int max_chunk) : m_(m), ex_(x), c_(m.cfg()), blas_(nullptr), max_chunk_(max_chunk) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  blas_.set_stream(stream_);
  const QwenConfig& c = c_;
  lin_index_.assign(c.n_layers, -1); attn_index_.assign(c.n_layers, -1);
  // layer indices of the GPU layers (all, or the prefix [0, gpu_layers) under hybrid placement — the tail engine counts its own)
  for (int l = 0; l < m.gpu_layers(); ++l) { if (c.is_attn[l]) attn_index_[l] = n_attn_++; else lin_index_[l] = n_lin_++; }
  kv_base_ = max_chunk;
  if (const char* v = getenv("HIVE_QWEN_KV_BASE")) { const long long a = atoll(v); if (a >= 0 && *v) kv_base_ = a; }
  if (const char* v = getenv("HIVE_QWEN_KV_STEP")) { const long long a = atoll(v); if (a > 0) kv_step_ = a; }
  kv_headroom_ = cache_reserve_bytes();
  free0_ = m.free_at_load();
  if (const char* v = getenv("HIVE_QWEN_VRAM_MB"); v && *v && atof(v) > 0) budget_ = (size_t)(atof(v) * 1048576.0);
  device_route_ = !env_off("HIVE_QWEN_DEVICE_ROUTE");
  // HIVE_QWEN_MAPPED_ROUTE (default on): when the expert cache cannot hold every expert (a VRAM limit) but holds most of them
  //   (HIVE_QWEN_MAPPED_MIN, default 0.6), decode / verify still route on the device — a non-resident expert's record is read by the expert
  //   kernels from the pinned host copy over PCIe (GlmExperts::load_all registers it) — instead of the host route (48 stream syncs per
  //   step, no CUDA graph). The cache bookkeeping follows the step (GlmExperts::note_used, after_step promotes as on the host route).
  //   Measured at a 72 GB cap (warm c1, 2026-10-12): 74.5 % of the experts resident, 98 % of the routed experts hit, c1 86.0 tok/s vs 222.3
  //   at 96 GB — the host route itself was the cost (CPU misses skipped: still 80.8). Mapped: 156.4. Below the threshold (48 GB: 37 %
  //   resident, 35 % misses) whole records over PCIe cost more than the host route's CPU work (measured 31.5 vs 81.1) — the host route stays.
  {
    int can = 0;
    cudaDeviceGetAttribute(&can, cudaDevAttrCanUseHostPointerForRegisteredMem, 0);
    mapped_route_ = device_route_ && can && !env_off("HIVE_QWEN_MAPPED_ROUTE");
    if (const char* v = getenv("HIVE_QWEN_MAPPED_MIN"); v && *v) mapped_min_ = atof(v);
  }
  if (budget_ && !grow_backend())  // without VMM the expert cache cannot give memory back: KV growth would pass the budget unnoticed
    fprintf(stderr, "[qwen] ⚠️HIVE_QWEN_VRAM_MB without CUDA VMM: the expert cache cannot shrink for KV growth, so the budget holds only at startup\n");
  tiles_ = std::max(1, getenv("HIVE_QWEN_PREFILL_TILES") ? atoi(getenv("HIVE_QWEN_PREFILL_TILES")) : 4);
  const size_t R = max_chunk, H = c.hidden, HC = c.hcw();
  auto al = [](DevBuf& b, size_t bytes) { b.alloc(std::max<size_t>(bytes, 256)); };
  const size_t projw = std::max<size_t>(kGdnProjWidth, kQsaProjWidth);
  al(h_, R * HC * 2); al(x_, R * H * 2); al(y_, R * H * 2); al(hn_, R * HC * 2); al(dl_, R * (c.hc_rank + c.hc) * 2);
  al(hs_, R * c.hc_rank * 2); al(hu_, R * HC * 2); al(proj_, R * projw * 2);
  al(q_, R * c.q_width() * 2); al(iq_, R * c.idx_heads * c.idx_dim * 2); al(idx_, R * kQsaSel * 4);
  al(attn_out_, R * c.q_width() * 2); al(gdn_out_, R * c.lin_v_width() * 2);
  al(mlp_out_f32_, R * H * 4); al(shared_out_, R * H * 2); al(sh_gu_, R * 2 * c.shared_inter * 2); al(sh_y_, R * c.shared_inter * 2);
  al(router_, R * (c.n_experts + 1) * 4); al(rids_, R * c.n_act * 4); al(rw_, R * c.n_act * 4);
  ws_bytes_ = std::max(gdn_workspace_bytes(max_chunk), qsa_select_ws_bytes(64, (int)std::min<int64_t>(c.max_position, 1 << 20)));
  al(ws_, ws_bytes_);
  al(logits_dev_, (size_t)64 * c.vocab * 4); al(xfin_, 64 * H * 2);
  al(gemv_ws_, std::max({fn_gemv_ws_bytes(c.hc_rank + c.hc, (int)HC, kDecRows), fn_gemv_ws_bytes(c.n_experts + 1 + 2 * c.shared_inter, (int)H, kDecRows),
                         (size_t)32 << 20}));   // + the batched-row bf16 GEMVs (lin)   // fn_gemv partials
  emb_pin_ = pinned<bf16>(R * H);
  ple_rows_pin_ = pinned<uint8_t>(R * kPleRow);
  rope_pin_ = pinned<int32_t>(R * 3);
  rope_dst_pin_ = pinned<int32_t*>(R);
  qrows_cap_ = (int)R * (n_attn_ + 2);
  qrows_pin_ = pinned<QsaRow>(qrows_cap_);
  al(qrows_dev_, (size_t)qrows_cap_ * sizeof(QsaRow));   // device copy the kernels read (attn_layer / mtp_kv upload it)
  gdn_ptr_pin_ = pinned<float*>((size_t)n_lin_ * 2 * kDecRows);   // per linear layer: kDecRows conv then kDecRows state pointers
  ple_ptr_pin_ = pinned<float*>(kDecRows);
  ptr_gdn_n_ = (size_t)n_lin_ * 2 * kDecRows;
  al(ptr_dev_, (ptr_gdn_n_ + kDecRows + (size_t)(kDecRows / 2) * (n_lin_ * 16 + 8)) * sizeof(float*));
  rids_h_ = mapped<int32_t>((size_t)64 * c.n_act);
  rw_h_ = mapped<float>((size_t)64 * c.n_act);
  rope_inv_freq_.alloc(64 * 4); rope_axis_.alloc(64 * 4);
  qsa_rope_tables(c.rope_theta, c.mrope_section.data(), c.mrope_interleaved, rope_inv_freq_.as<float>(), rope_axis_.as<int32_t>(), stream_);
  if (m_.ple_layer() >= 0) {
    ple_hash_.ngram = c.ngram; ple_hash_.heads_per = c.heads_per_ngram; ple_hash_.eos = c.eos_id; ple_hash_.vocab = c.vocab;
    ple_ = ple_open(m_.ckpt(), "model.language_model.layers." + std::to_string(m_.ple_layer()) + ".ple.", ple_hash_, ple_scale_);
  }
  // device routing table and the decode MoE workspace (glm::moe_decode's v0 path for these shapes: any H, I multiple of 32)
  rec_tab_.alloc((size_t)ex_.n_moe_layers() * ex_.n_experts() * sizeof(void*));
  pairs_.alloc((size_t)64 * c.n_act * sizeof(MoePairDev));
  // FN's own decode kernels (fn_moe.h) unless HIVE_QWEN_FN_MOE=0 (the GLM reference path, for A/B)
  fn_moe_ = !env_off("HIVE_QWEN_FN_MOE") && fn_moe_decode_supported(c.hidden, c.moe_inter);
  // HIVE_QWEN_FUSE (default on): fused decode launches — router + shared gate|up in one split-K GEMV, the routed MoE straight from the
  //   router's device output with the final combine, the hyper-connection down projection with its activation (fn_gemv_epi)
  fuse_ = fn_moe_ && !env_off("HIVE_QWEN_FUSE");
  // HIVE_QWEN_HYBRID_ROUTE (default on; qwen_engine.h): below the mapped threshold the step keeps the device path and a CUDA graph, the
  //   CPU computing the non-resident experts in host nodes — instead of the host route (per MoE layer: a stream sync for the routing, a
  //   D2H + sync for the input, the CPU experts, a sync after their add; every launch eager). Needs the fused decode MoE (null records are
  //   skipped by fn_moe_decode_routed) and every layer on the GPU.
  hybrid_route_ = device_route_ && fuse_ && m.gpu_layers() == c.n_layers && !env_off("HIVE_QWEN_HYBRID_ROUTE");
  if (hybrid_route_) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&hyb_st_, cudaStreamNonBlocking));
    hyb_ev_.resize((size_t)2 * c.n_layers);
    for (auto& e : hyb_ev_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    hyb_arg_.resize(c.n_layers);
    for (int l = 0; l < c.n_layers; ++l) hyb_arg_[l] = HybArg{this, l};
    hyb_x_ = pinned<bf16>((size_t)kDecRows * c.hidden);
    hyb_y_ = pinned<float>((size_t)kDecRows * c.hidden);
    hyb_xf_.resize((size_t)kDecRows * c.hidden);
    hyb_ids_.resize((size_t)kDecRows * c.n_act);
    hyb_mrank_.resize(c.n_experts);
    if (const char* v = getenv("HIVE_QWEN_HYB_DIAG"); v && strcmp(v, "nocpu") == 0) hyb_diag_nocpu_ = true;
    if (const char* v = getenv("HIVE_QWEN_HYB_PCIE"); v && *v) hyb_pcie_div_ = std::max(0, atoi(v));
    if (const char* v = getenv("HIVE_QWEN_HYB_ROWS"); v && *v) hyb_max_rows_ = std::max(0, atoi(v));
    int can = 0;
    cudaDeviceGetAttribute(&can, cudaDevAttrCanUseHostPointerForRegisteredMem, 0);
    if (!can) hyb_pcie_div_ = 0;
  }
  // HIVE_QWEN_DENSE8 with the bf16 originals freed: the expansion scratch of the prefill GEMMs, and the shared gate|up rows of the fused
  //   router GEMV (rs_) expanded from the 8-bit copy
  if (m.dense8() && m.dense8_max()) al(dq_, m.dense8_max() * 2);
  auto sh_gu_bf16 = [&](const QMat& W, uint8_t* dst) {
    if (W.p()) CUDA_CHECK(cudaMemcpy(dst, W.p(), (size_t)W.out * W.in * 2, cudaMemcpyDeviceToDevice));
    else { fn_dequant8(W.q8.as<uint8_t>(), W.s8.as<float>(), W.fmt8, W.gs8, W.out, W.in, reinterpret_cast<bf16*>(dst), stream_); CUDA_CHECK(cudaStreamSynchronize(stream_)); }
  };
  if (fuse_ && m.mtp()) {   // the MTP layer: the same fused router + shared GEMV and FN's FP8 expert kernels (graph-capturable drafts)
    const MoeW& w = m.mtp()->layer.moe;
    const size_t a = (size_t)w.router.out * w.router.in * 2, b = (size_t)w.sh_gu.out * w.sh_gu.in * 2;
    mtp_rs_.alloc(a + b);
    CUDA_CHECK(cudaMemcpy(mtp_rs_.p, w.router.p(), a, cudaMemcpyDeviceToDevice));
    sh_gu_bf16(w.sh_gu, mtp_rs_.as<uint8_t>() + a);
    mtp_tab_ = glm::glm_moe_fp8_table_create(m.mtp()->experts.data()->data(), (int)m.mtp()->experts.size(), c.hidden, c.moe_inter);
  }
  if (fuse_) {
    rs_.resize(m.gpu_layers());
    for (int l = 0; l < m.gpu_layers(); ++l) {
      const MoeW& w = m.layer(l).moe;
      const size_t a = (size_t)w.router.out * w.router.in * 2, b = (size_t)w.sh_gu.out * w.sh_gu.in * 2;
      rs_[l].alloc(a + b);
      CUDA_CHECK(cudaMemcpy(rs_[l].p, w.router.p(), a, cudaMemcpyDeviceToDevice));
      sh_gu_bf16(w.sh_gu, rs_[l].as<uint8_t>() + a);
    }
  }
  // HIVE_QWEN_FN_PREFILL (default on): prefill chunks whose experts are all resident run FN's grouped W4A16 GEMMs (fn_moe_prefill) on the
  //   router's device output; HIVE_QWEN_MOE_PF_ROWS (default 8192) rows per sub-chunk size its per-pair buffers (≈ 64 KB per row; 4096 → 8192: 16K prefill 6,614 → 7,031 tok/s, 64K 5,876 → 6,176 on the reference machine 2026-10-10)
  fn_pf_ = fn_moe_ && device_route_ && !env_off("HIVE_QWEN_FN_PREFILL") && fn_moe_prefill_supported(c.hidden, c.moe_inter);
  if (fn_pf_) {
    pf_rows_ = std::max(64, std::min(max_chunk, getenv("HIVE_QWEN_MOE_PF_ROWS") ? atoi(getenv("HIVE_QWEN_MOE_PF_ROWS")) : 8192));
    al(pf_ws_, fn_moe_prefill_ws_bytes(pf_rows_, c.n_act, c.n_experts, c.hidden, c.moe_inter));
  }
  // HIVE_QWEN_QSA_SPLIT (default 16; 0/1 = the CUDA-core kernel): decode / verify attention over the ≤ 2,051 selected positions split
  //   across that many blocks per (row, kv head). hived c1 (1K context): verify 15.3 ms against 12.2 in qwen_check at ~30 tokens — the
  //   CUDA-core kernel ran a 4-row verify on 8 blocks
  qsa_split_ = getenv("HIVE_QWEN_QSA_SPLIT") ? std::max(0, std::min(32, atoi(getenv("HIVE_QWEN_QSA_SPLIT")))) : 16;
  if (qsa_split_ > 1) al(qsa_split_ws_, qsa_attention_split_ws_bytes(kDecRows, qsa_split_));
  moe_ws_bytes_ = std::max(glm::moe_decode_ws_bytes(64 * c.n_act, c.hidden, c.moe_inter), fn_moe_decode_ws_bytes(64 * c.n_act, c.hidden, c.moe_inter));
  al(moe_ws_, moe_ws_bytes_);
  // MTP (HIVE_QWEN_MTP_K drafts per step, default 3; 0 = off)
  // default 4 drafts (hived's gate still picks k ≤ 4 per step): c1 600 W hived, 48 requests — k 3 TPOT 4.66 ms / accept 2.50, k 4 4.51 / 2.68,
  //   k 5 4.54 / 2.77 (2026-10-10). Drafts are verified by the full model: outputs do not depend on k
  mtp_k_ = m_.mtp() ? (getenv("HIVE_QWEN_MTP_K") ? std::max(0, std::min(7, atoi(getenv("HIVE_QWEN_MTP_K")))) : 4) : 0;
  if (m_.mtp()) {
    al(fh_, R * HC * 2);
    mtp_emb_pin_ = pinned<bf16>(R * H);
    al(mtp_emb_, R * H * 2); al(mtp_en_, R * H * 2); al(mtp_fc_, R * H * 2); al(mtp_hn_, R * HC * 2); al(mtp_h_, R * HC * 2);
    al(mtp_hout_, 2 * HC * 2); al(mtp_ids_d_, 16 * 4); al(mtp_conf_d_, 32 * 4);
    al(mtpb_h_, (size_t)3 * kDraftRows * HC * 2); al(mtpb_ids_d_, (size_t)8 * kDraftRows * 4); al(mtpb_conf_d_, (size_t)16 * kDraftRows * 4);
    one_f_.alloc(16); { const float one = 1.f; CUDA_CHECK(cudaMemcpy(one_f_.p, &one, 4, cudaMemcpyHostToDevice)); }
    // the draft chain gathers the embedding of its previous draft on the device: pin + map the host table
    CUDA_CHECK(cudaHostRegister(const_cast<bf16*>(m_.embed_host()), (size_t)c.vocab * c.hidden * 2, cudaHostRegisterMapped));
    alloc_snapshots(1);
    // HIVE_QWEN_DRAFT_VOCAB=FILE (int32 token ids, tools/build_draft_vocab.py): the draft head scores only these lm_head rows (drafts are
    //   proposals; the verify step scores them with the full head, so outputs do not change) — the GLM switch of the same name and reason.
    if (const char* dv = getenv("HIVE_QWEN_DRAFT_VOCAB"); dv && *dv && strcmp(dv, "0") != 0) {
      std::vector<int32_t> rows;
      if (FILE* f = fopen(dv, "rb")) { int32_t v; while (fread(&v, 4, 1, f) == 1) if (v >= 0 && v < c.vocab) rows.push_back(v); fclose(f); }
      std::sort(rows.begin(), rows.end());
      rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
      if (!rows.empty() && (int)rows.size() < c.vocab) {
        al(draft_rows_, rows.size() * 4);
        CUDA_CHECK(cudaMemcpy(draft_rows_.p, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
        al(draft_idx_, 16 * 4);
        draft_n_ = (int)rows.size();
        fprintf(stderr, "[qwen] MTP draft head: %d of %d tokens (%s)\n", draft_n_, c.vocab, dv);
      } else {
        fprintf(stderr, "[qwen] ⚠️HIVE_QWEN_DRAFT_VOCAB=%s: no usable token ids — the draft head scores the whole vocabulary\n", dv);
      }
    }
    // HIVE_QWEN_DRAFT_FP8 (default on): the draft head reads an FP8 block-scaled copy of lm_head (half the bytes of the three draft
    //   heads per step; outputs unchanged — the verify step scores the drafts with the bf16 head). With HIVE_QWEN_DRAFT_VOCAB the copy
    //   holds only the listed rows (draft_n_ × hidden, ascending token id).
    if (!env_off("HIVE_QWEN_DRAFT_FP8")) {
      const int V = draft_n_ > 0 ? draft_n_ : c.vocab, Hd = c.hidden;
      al(draft_q_, (size_t)V * Hd); al(draft_s_, (size_t)((V + 127) / 128) * ((Hd + 127) / 128) * 4); al(draft_lg_, (size_t)V * 2 * kDraftRows); al(draft_part_, (size_t)fn_draft_pick_parts(V) * 12 * kDraftRows);
      if (draft_n_ > 0) {
        DevBuf tmp; al(tmp, (size_t)V * Hd * 2);
        fn_gather_rows(m_.lm_head().p(), draft_rows_.as<int32_t>(), V, Hd, tmp.as<bf16>(), stream_);
        fn_quant_fp8b(tmp.as<bf16>(), V, Hd, draft_q_.as<uint8_t>(), draft_s_.as<float>(), stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
      } else {
        fn_quant_fp8b(m_.lm_head().p(), V, Hd, draft_q_.as<uint8_t>(), draft_s_.as<float>(), stream_);
      }
    }
  }
  // hybrid placement: the CPU engine for layers [gpu_layers, n_layers) (its own weights and experts for those layers, in host memory)
  if (m_.gpu_layers() < c.n_layers) {
    cpu::QwenCpuEngine::Options o;
    o.load_mtp = false; o.first_layer = m_.gpu_layers(); o.head = false;
    tail_ = std::make_unique<cpu::QwenCpuEngine>(m_.ckpt().dir(), o);
    tail_pin_ = pinned<bf16>(R * HC);
    fprintf(stderr, "[qwen] hybrid placement: layers 0..%d on the GPU, %d..%d on the CPU (%.1f GiB host weights)\n", m_.gpu_layers() - 1,
            m_.gpu_layers(), c.n_layers - 1, tail_->host_bytes() / 1073741824.0);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  fprintf(stderr, "[qwen] engine: chunk %d · %d linear + %d attention layers · workspace %.0f MiB · PLE %s · device routing %s · VRAM budget %s · MTP k %d\n",
          max_chunk, n_lin_, n_attn_, ws_bytes_ / 1048576.0, ple_ ? ple_->kind() : "none", device_route_ ? "on" : "off",
          budget_ ? (std::to_string(budget_ >> 20) + " MiB").c_str() : "none (whole card)", mtp_k_);
}

QwenEngine::~QwenEngine() {
  clear_graphs();
  if (mtp_tab_) glm::glm_moe_fp8_table_free(mtp_tab_);
  for (void* p : {(void*)emb_pin_, (void*)ple_rows_pin_, (void*)rope_pin_, (void*)rope_dst_pin_, (void*)qrows_pin_, (void*)gdn_ptr_pin_,
                  (void*)ple_ptr_pin_, (void*)rids_h_, (void*)rw_h_, (void*)snap_ptr_pin_, (void*)mtp_emb_pin_, (void*)tail_pin_, (void*)hyb_x_, (void*)hyb_y_})
    if (p) cudaFreeHost(p);
  for (auto e : hyb_ev_) cudaEventDestroy(e);
  if (hyb_st_) cudaStreamDestroy(hyb_st_);
}

// ---- memory ----------------------------------------------------------------------------------------------------------------------------
size_t QwenEngine::vram_avail() const {
  size_t fr = 0, tot = 0;
  CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
  if (!budget_) return fr;
  const size_t used = free0_ > fr ? free0_ - fr : 0;
  return std::min(fr, budget_ > used ? budget_ - used : 0);
}

void QwenEngine::make_room(size_t bytes) {
  const size_t av = vram_avail();
  if (av >= bytes + kv_headroom_) return;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  ex_.shrink_cache(bytes + kv_headroom_ - av);
}

void QwenEngine::give_back_room() {
  if (no_regrow_) return;   // inside kv_reserve's reclaim: the freed bytes are for the growing sequence
  // regrow_cache keeps `reserve` bytes free; with a budget, the part of the free VRAM above the budget must stay free too
  size_t fr = 0, tot = 0;
  CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
  const size_t av = vram_avail();
  ex_.regrow_cache(kv_headroom_ + (fr > av ? fr - av : 0));
}

void QwenEngine::init_seq(QwenSeq& s, int64_t cap) {
  const QwenConfig& c = c_;
  s.cap = cap; s.device_bytes = 0; s.kv_tokens = 0;
  s.conv.resize(n_lin_); s.state.resize(n_lin_);
  for (int i = 0; i < n_lin_; ++i) {
    s.conv[i].alloc(kGdnConvStateFloats * 4); s.state[i].alloc(kGdnStateFloats * 4);
    s.device_bytes += s.conv[i].n + s.state[i].n;
  }
  auto res = [&](AttnCache& a) {
    a.k.reserve((size_t)cap * c.kv_width() * 2); a.v.reserve((size_t)cap * c.kv_width() * 2);
    a.ik.reserve((size_t)cap * c.idx_dim * 2); a.gk.reserve((size_t)(cap / c.idx_ratio + 1) * c.idx_dim * 2);
  };
  s.attn.resize(n_attn_);
  for (auto& a : s.attn) res(a);
  s.rope.reserve((size_t)cap * 12);
  if (m_.ple_layer() >= 0) { s.ple_state.alloc((size_t)kPleConvState * c.hcw() * 4); s.device_bytes += s.ple_state.n; }
  if (m_.mtp()) { res(s.mtp); s.h_last.alloc((size_t)c.hcw() * 2); s.device_bytes += s.h_last.n; }
  kv_reserve(s, std::min<int64_t>(kv_base_, cap), false);
  if (tail_) { s.tail = std::make_unique<cpu::CpuSeq>(); tail_->init_seq(*s.tail, cap); }
  reset_seq(s);
}

size_t QwenEngine::kv_bytes(const QwenSeq& s, int64_t t) const {
  const QwenConfig& c = c_;
  const size_t T = (size_t)t;
  auto one = [&](const AttnCache& a) {
    return a.k.grow_cost(T * c.kv_width() * 2) + a.v.grow_cost(T * c.kv_width() * 2) + a.ik.grow_cost(T * c.idx_dim * 2) +
           a.gk.grow_cost((T / c.idx_ratio + 1) * c.idx_dim * 2);
  };
  size_t n = s.rope.grow_cost(T * 12);
  for (const auto& a : s.attn) n += one(a);
  if (s.h_last.p) n += one(s.mtp);
  return n;
}

void QwenEngine::kv_reserve(QwenSeq& s, int64_t tokens, bool reclaim) {
  if (tokens <= s.kv_tokens) return;
  HIVE_CHECK(tokens <= s.cap, "qwen: sequence capacity exceeded");
  const QwenConfig& c = c_;
  const int64_t t = std::min<int64_t>(s.cap, (tokens + kv_step_ - 1) / kv_step_ * kv_step_);
  const size_t need = kv_bytes(s, t);
  if (need > 0) {
    size_t av = vram_avail();
    // idle sessions first (kv_reclaim_, hived: evicts the least recently used idle session to a host image and trims its sequence), then the
    //   expert cache — lending expert slots to idle sessions' KV evicted experts for good (fn_final600b, 2026-10-10: 24576 → 13415 slots)
    if (av < need + kv_headroom_ && kv_reclaim_ && reclaim) {   // (init_seq passes false: hived may hold its session lock there)
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      no_regrow_ = true;
      while (av < need + kv_headroom_ && kv_reclaim_()) av = vram_avail();
      no_regrow_ = false;
    }
    if (av < need + kv_headroom_) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));  // kernels of the last step may still read tail slots
      fprintf(stderr, "[qwen] KV growth to %lld tokens lends the expert cache %.0f MiB (free %.0f MiB, need %.0f + headroom %.0f MiB, reclaim %s)\n",
              (long long)t, (need + kv_headroom_ - av) / 1048576.0, av / 1048576.0, need / 1048576.0, kv_headroom_ / 1048576.0,
              kv_reclaim_ ? (reclaim ? "tried" : "not allowed here") : "none");
      ex_.shrink_cache(need + kv_headroom_ - av);
    }
  }
  const size_t T = (size_t)t;
  bool ok = true;
  auto grow = [&](GrowBuf& b, size_t bytes) { ok = ok && b.ensure(bytes); };
  auto one = [&](AttnCache& a) {
    grow(a.k, T * c.kv_width() * 2); grow(a.v, T * c.kv_width() * 2); grow(a.ik, T * c.idx_dim * 2); grow(a.gk, (T / c.idx_ratio + 1) * c.idx_dim * 2);
  };
  grow(s.rope, T * 12);
  for (auto& a : s.attn) one(a);
  if (s.h_last.p) one(s.mtp);
  if (!ok) {
    kv_trim(s, s.kv_tokens);
    throw std::runtime_error("qwen: no VRAM for the KV cache of " + std::to_string((long long)tokens) + " positions (available " +
                             std::to_string((long long)(vram_avail() >> 20)) + " MiB, expert cache " + std::to_string(ex_.n_slots()) + " slots)");
  }
  s.kv_tokens = t;
}

void QwenEngine::kv_trim(QwenSeq& s, int64_t tokens) {
  const QwenConfig& c = c_;
  const size_t T = (size_t)std::max<int64_t>(tokens, 0);
  size_t freed = s.rope.trim(T * 12);
  auto one = [&](AttnCache& a) {
    return a.k.trim(T * c.kv_width() * 2) + a.v.trim(T * c.kv_width() * 2) + a.ik.trim(T * c.idx_dim * 2) + a.gk.trim((T / c.idx_ratio + 1) * c.idx_dim * 2);
  };
  for (auto& a : s.attn) freed += one(a);
  if (s.h_last.p) freed += one(s.mtp);
  s.kv_tokens = tokens;
  if (freed) give_back_room();
}

void QwenEngine::reset_seq(QwenSeq& s) {
  for (auto& b : s.conv) CUDA_CHECK(cudaMemsetAsync(b.p, 0, b.n, stream_));
  for (auto& b : s.state) CUDA_CHECK(cudaMemsetAsync(b.p, 0, b.n, stream_));
  if (s.ple_state.p) CUDA_CHECK(cudaMemsetAsync(s.ple_state.p, 0, s.ple_state.n, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  kv_trim(s, std::min<int64_t>(kv_base_, s.cap));
  s.pos = 0; s.tokens.clear(); s.broken = false; s.rope_delta = 0;
  s.h_last_valid = false; s.mtp_written = 0;
  if (tail_ && s.tail) tail_->reset_seq(*s.tail);
}

// ---- building blocks -------------------------------------------------------------------------------------------------------------------
QsaRow* QwenEngine::qrows_alloc(int n) {
  HIVE_CHECK(n <= qrows_cap_, "qwen: attention row table too small");
  if (qrows_off_ + (size_t)n > (size_t)qrows_cap_) { CUDA_CHECK(cudaStreamSynchronize(stream_)); qrows_off_ = 0; }  // earlier tables are no longer read
  QsaRow* p = qrows_pin_ + qrows_off_;
  qrows_off_ += n;
  return p;
}

void QwenEngine::lin(const QMat& W, const bf16* A, int lda, int rows, bf16* C, int ldc) {
  if (rows <= 0) return;
  // decode rows: one tensor-core GEMV launch (glm_decode.h) — the bf16 cuBLAS GEMM is not a single rounding at these shapes either
  // HIVE_QWEN_DENSE8: the 8-bit copy (fn_gemv8)
  if (rows <= kDecRows && W.fmt8 && lda % 8 == 0 && ((uintptr_t)A & 15) == 0 && fn_gemv8_ok(W.out, W.in, W.gs8, rows)) {
    fn_gemv8(W.q8.as<uint8_t>(), W.s8.as<float>(), W.fmt8, W.gs8, W.out, W.in, A, lda, rows, C, ldc, stream_);
    return;
  }
  const bf16* Wp = W.p();
  if (!Wp) {   // 8-bit only (the bf16 original was freed): expand it for this GEMM
    HIVE_CHECK(W.fmt8 && (size_t)W.out * W.in * 2 <= dq_.n, "qwen: an 8-bit-only matrix without the expansion scratch");
    fn_dequant8(W.q8.as<uint8_t>(), W.s8.as<float>(), W.fmt8, W.gs8, W.out, W.in, dq_.as<bf16>(), stream_);
    Wp = dq_.as<bf16>();
  }
  // few 16-row tiles with a long K (the hyper-connection down projection, 324 × 10240): K split across blocks too (fn_gemv.h) — measured
  //   14.7 → 10.2 µs (bench_fn_gemv); every other shape is faster on the GLM kernel
  if (rows <= kDecRows && (W.out + 15) / 16 < 64 && W.in >= 4096 && lda % 8 == 0 && ((uintptr_t)A & 15) == 0 && fn_gemv_ok(W.out, W.in, rows) &&
      fn_gemv_ws_bytes(W.out, W.in, rows) <= gemv_ws_.n) {
    fn_gemv(Wp, W.out, W.in, A, lda, rows, C, ldc, nullptr, gemv_ws_.as<float>(), stream_);
    return;
  }
  // 9..64 decode / verify rows on a bf16 matrix (lossless default, batched sequences): fn_gemv_tc — the GLM kernel's schedule with up to 8
  //   activation tiles per weight read, so a row's result equals its ≤ 8-row result (cuBLAS / split-K made batched decode diverge)
  if (rows > 8 && rows <= kDecRows && W.in % 32 == 0 && lda % 8 == 0 && ((uintptr_t)A & 15) == 0) {   // same per-row result as ≤ 8 rows
    fn_gemv_tc(Wp, W.out, W.in, A, lda, rows, C, ldc, nullptr, stream_);
    return;
  }
  if (rows <= 8 && W.in % 32 == 0 && lda % 8 == 0 && ((uintptr_t)A & 15) == 0) {
    const glm::GemvSeg sg{Wp, A, C, W.out, W.in, lda, ldc};
    glm::glm_gemv_bf16_multi(&sg, 1, rows, stream_);
    return;
  }
  blas_.gemm_bf16_batched_ld(A, lda, 0, Wp, W.in, 0, C, ldc, 0, rows, W.out, W.in, 1);
}

void QwenEngine::hc_mix(const HcW& w, const bf16* h, int rows, bf16* x, bool normed) {
  const int H = c_.hidden, HC = c_.hcw(), rk = c_.hc_rank, nd = w.down_inj.out;
  if (!normed) gemma_rmsnorm(h, HC, w.norm.as<bf16>(), rows, HC, H, c_.rms_eps, hn_.as<bf16>(), HC, stream_);   // (normed: hc_combine_norm did it)
  if (fuse_ && rows <= kDecRows && w.down_inj.fmt8 && (size_t)fn_gemv8_split_parts(nd, HC, w.down_inj.gs8) * rows * nd * 4 <= gemv_ws_.n) {
    FnEpi e; e.mode = 2; e.y = dl_.as<bf16>(); e.ldy = nd; e.s = hs_.as<bf16>(); e.rank = rk; e.inv_hc = 1.f / c_.hc;
    fn_gemv8_epi(w.down_inj.q8.as<uint8_t>(), w.down_inj.s8.as<float>(), w.down_inj.fmt8, w.down_inj.gs8, nd, HC, hn_.as<bf16>(), HC, rows, e,
                 gemv_ws_.as<float>(), stream_);
  } else if (fuse_ && rows <= kDecRows && !w.down_inj.fmt8 && fn_gemv_ok(nd, HC, rows) && fn_gemv_ws_bytes(nd, HC, rows) <= gemv_ws_.n) {
    FnEpi e; e.mode = 2; e.y = dl_.as<bf16>(); e.ldy = nd; e.s = hs_.as<bf16>(); e.rank = rk; e.inv_hc = 1.f / c_.hc;
    fn_gemv_epi(w.down_inj.p(), nd, HC, hn_.as<bf16>(), HC, rows, e, gemv_ws_.as<float>(), stream_);
  } else {
    lin(w.down_inj, hn_.as<bf16>(), HC, rows, dl_.as<bf16>(), nd);
    hc_mix_act(dl_.as<bf16>(), nd, rows, rk, c_.hc, hs_.as<bf16>(), stream_);
  }
  lin(w.up, hs_.as<bf16>(), rk, rows, hu_.as<bf16>(), HC);
  hc_mix_out(hu_.as<bf16>(), hn_.as<bf16>(), rows, c_.hc, H, x, stream_);
}

bool QwenEngine::hc_fuse() { static const bool on = !env_off("HIVE_QWEN_HC_FUSE"); return on; }

// layer l's MLP-side combine; with the next layer's attention-side input norm in the same launch when nothing runs in between (no PLE in that
//   layer, no validation hook, no layer yield)
void QwenEngine::combine_mlp(int l, bf16* h, int rows, bool may_fuse) {
  const HcW& w = m_.layer(l).hc_mlp;
  if (hc_fuse() && may_fuse && l + 1 < m_.gpu_layers() && !m_.layer(l + 1).ple) {
    hc_combine_norm(h, y_.as<bf16>(), c_.hidden, dl_.as<bf16>(), w.down_inj.out, c_.hc_rank, rows, c_.hc, c_.hidden,
                    m_.layer(l + 1).hc_attn.norm.as<bf16>(), c_.rms_eps, hn_.as<bf16>(), stream_);
    pre_normed_ = true;
  } else {
    hc_combine_(w, h, y_.as<bf16>(), rows, dl_.as<bf16>());
  }
}

void QwenEngine::hc_combine_(const HcW& w, bf16* h, const bf16* y, int rows, const bf16* dl) {
  hc_combine(h, y, c_.hidden, dl, w.down_inj.out, c_.hc_rank, rows, c_.hc, c_.hidden, stream_);
}

void QwenEngine::apply_embed_override(bf16* x_dev, int64_t base, int rows) {
  const size_t H = c_.hidden;
  for (const EmbedOverride& o : emb_over_) {
    const int64_t a = std::max<int64_t>(o.row, base), b = std::min<int64_t>(o.row + o.n, base + rows);
    if (a >= b) continue;
    CUDA_CHECK(cudaMemcpyAsync(x_dev + (size_t)(a - base) * H, o.dev + (size_t)(a - o.row) * H, (size_t)(b - a) * H * 2, cudaMemcpyDeviceToDevice, stream_));
  }
}

void QwenEngine::embed(const std::vector<RowInfo>& rows, const int32_t* ids, int64_t base, bf16* h) {
  const int R = (int)rows.size(), H = c_.hidden;
  if (gphase_ != 2) for (int r = 0; r < R; ++r) std::memcpy(emb_pin_ + (size_t)r * H, m_.embed_host() + (size_t)ids[r] * H, (size_t)H * 2);
  if (gphase_ == 1) return;
  bool over = false;
  for (const EmbedOverride& o : emb_over_) over = over || (o.row < base + R && o.row + o.n > base);
  // decode rows: the pinned rows read directly (UVA, graph-capturable); prefill rows: one DMA first — hc_expand reads every row hc times,
  //   over PCIe that was 3.7 ms of a 1K-token prefill (nsys 2026-10-10)
  if (!over && R <= kDecRows) { hc_expand(emb_pin_, R, c_.hc, H, h, stream_); return; }
  CUDA_CHECK(cudaMemcpyAsync(x_.p, emb_pin_, (size_t)R * H * 2, cudaMemcpyHostToDevice, stream_));
  apply_embed_override(x_.as<bf16>(), base, R);
  hc_expand(x_.as<bf16>(), R, c_.hc, H, h, stream_);
}

// Rope coordinates of the rows (see qwen_engine.h): text = (p + d)·3; an image span [row, row + gh·gw) of the prefill call starting at
//   rope coordinate st = (position of its first token) + d takes (st, st + i / gw, st + i % gw) and moves d by max(gh, gw) − gh·gw after
//   its last token (the next text coordinate is st + max(gh, gw)).
void QwenEngine::set_rope(const std::vector<RowInfo>& rows, int64_t base, bool prefill) {
  const int R = (int)rows.size();
  if (gphase_ == 2) { write_rope(rope_pin_, rope_dst_pin_, R, stream_); return; }   // graph capture: host part done by host_prep
  for (int r = 0; r < R; ++r) {
    QwenSeq& s = *rows[r].s;
    const int64_t p = rows[r].pos, gi = base + r;
    int32_t* o = rope_pin_ + (size_t)r * 3;
    const EmbedOverride* sp = nullptr;
    if (prefill)
      for (const EmbedOverride& e : emb_over_) if (e.gw > 0 && gi >= e.row && gi < e.row + e.n) { sp = &e; break; }
    if (!sp) {
      const int32_t v = (int32_t)(p + s.rope_delta);
      o[0] = o[1] = o[2] = v;
    } else {
      const int64_t k = gi - sp->row, st = (p - k) + s.rope_delta;
      o[0] = (int32_t)st; o[1] = (int32_t)(st + k / sp->gw); o[2] = (int32_t)(st + k % sp->gw);
      if (k == sp->n - 1) s.rope_delta += std::max(sp->gh, sp->gw) - sp->n;
    }
    rope_dst_pin_[r] = s.rope.as<int32_t>() + (size_t)p * 3;
  }
  if (gphase_ != 1) write_rope(rope_pin_, rope_dst_pin_, R, stream_);
}

void QwenEngine::ple_block(const std::vector<RowInfo>& rows, const int32_t* ids, bf16* h, bool prefill) {
  const QwenConfig& c = c_;
  const int R = (int)rows.size(), H = c.hidden, HC = c.hcw();
  const PleW& w = m_.layer(m_.ple_layer()).plew;
  ple_ids_.resize((size_t)R * kPleHeads);
  // n-gram rows: consecutive rows of one sequence share their history; each starts from its sequence's processed tokens
  for (int r = 0; r < R && gphase_ != 2;) {
    QwenSeq& s = *rows[r].s;
    int n = 1;
    while (r + n < R && rows[r + n].s == &s && rows[r + n].pos == rows[r].pos + n) ++n;
    const int64_t p0 = rows[r].pos;
    HIVE_CHECK((int64_t)s.tokens.size() >= p0, "qwen PLE: token history shorter than the position");
    const int32_t prev1 = p0 >= 1 ? s.tokens[p0 - 1] : -1, prev2 = p0 >= 2 ? s.tokens[p0 - 2] : -1;
    ple_hash_.rows_of(ids + r, n, prev2, prev1, ple_ids_.data() + (size_t)r * kPleHeads);
    r += n;
  }
  if (gphase_ != 2) ple_->gather(ple_ids_.data(), R, ple_rows_pin_);
  if (gphase_ != 2 && !prefill) {   // the decode rows' / verify parts' state pointers (read by the kernels through ptr_dev_)
    if (verify_) for (size_t pi = 0; pi < vparts_.size(); ++pi) ple_ptr_pin_[pi] = vparts_[pi].s->ple_state.as<float>();
    else for (int r = 0; r < R; ++r) ple_ptr_pin_[r] = rows[r].s->ple_state.as<float>();
    if (gphase_ == 0) upload_ptr_tables();   // eager: the table just changed (graph phases upload at the graph start)
  }
  if (gphase_ == 1) return;   // host part only (graph replay)
  bf16* e = x_.as<bf16>();
  bf16* key = hu_.as<bf16>();
  bf16* val = y_.as<bf16>();
  bf16* gv = hn_.as<bf16>();
  bf16* gvn = proj_.as<bf16>();
  ple_dequant(ple_rows_pin_, R, ple_scale_, e, stream_);   // pinned rows read directly (UVA)
  lin(w.key, e, c.ple_dim, R, key, HC);
  lin(w.value, e, c.ple_dim, R, val, H);
  ple_gate(key, h, w.norm_key.as<bf16>(), w.norm_query.as<bf16>(), val, R, c.hc, H, c.rms_eps, gv, stream_);
  gemma_rmsnorm(gv, HC, w.norm_conv.as<bf16>(), R, HC, H, c.rms_eps, gvn, HC, stream_);
  if (prefill) {
    ple_conv_seq(gv, gvn, w.conv_w.as<float>(), R, HC, rows[0].s->ple_state.as<float>(), h, stream_);
  } else if (verify_) {
    const int kmax = std::max(1, mtp_k_);
    for (size_t pi = 0; pi < vparts_.size(); ++pi) {
      const VPart& vp = vparts_[pi];
      float** sp = dev_ptr(snap_ptr_pin_ + (size_t)snap_parts_ * n_lin_ * 16 + pi * 8);
      ple_conv_rows(gv + (size_t)vp.r0 * HC, gvn + (size_t)vp.r0 * HC, w.conv_w.as<float>(), vp.M, HC, dev_ptr(ple_ptr_pin_ + pi),
                    vp.M > 1 ? sp : nullptr, h + (size_t)vp.r0 * HC, stream_);
      (void)kmax;
    }
  } else {
    ple_conv_batch(gv, gvn, w.conv_w.as<float>(), R, HC, dev_ptr(ple_ptr_pin_), h, stream_);
  }
}

void QwenEngine::lin_layer(int l, const std::vector<RowInfo>& rows, bool prefill, const bf16* x, bf16* out) {
  const LinW& w = m_.layer(l).lin;
  const int R = (int)rows.size(), H = c_.hidden, li = lin_index_[l];
  bf16* proj = proj_.as<bf16>();
  lin(w.in, x, H, R, proj, kGdnProjWidth);
  const GdnParams p{w.conv_w.as<float>(), w.A_log.as<float>(), w.dt_bias.as<float>(), w.norm.as<bf16>(), c_.rms_eps};
  bf16* o = gdn_out_.as<bf16>();
  if (prefill) {
    QwenSeq& s = *rows[0].s;
    gdn_forward_seq(p, proj, kGdnProjWidth, R, s.conv[li].as<float>(), s.state[li].as<float>(), o, ws_.p, ws_bytes_, stream_);
  } else if (verify_) {
    if (snap_replay_ && vparts_.size() > 1)   // the projections of every row (the replay rollback of parts ≥ 1 runs the kept rows again)
      CUDA_CHECK(cudaMemcpyAsync(proj_save_.as<bf16>() + (size_t)li * kDecRows * kGdnProjWidth, proj, (size_t)R * kGdnProjWidth * 2,
                                 cudaMemcpyDeviceToDevice, stream_));
    float** dptr = dev_ptr(gdn_ptr_pin_ + (size_t)li * 2 * kDecRows);   // row r → its part's sequence
    GdnRowsPart gparts[kDecRows];
    for (size_t pi = 0; pi < vparts_.size(); ++pi) {
      const VPart& vp = vparts_[pi];
      float** sp = dev_ptr(snap_ptr_pin_ + ((size_t)pi * n_lin_ + li) * 16);
      const bool rp = replay_part((int)pi);
      gparts[pi] = GdnRowsPart{proj + (size_t)vp.r0 * kGdnProjWidth, vp.M, dptr + vp.r0, dptr + kDecRows + vp.r0, sp, sp + 8,
                               o + (size_t)vp.r0 * kGdnVWidth, rp ? 1 : 0, rp ? sp + 7 : nullptr};
    }
    gdn_decode_rows_multi(p, kGdnProjWidth, (int)vparts_.size(), gparts, stream_);   // every part in one launch pair
  } else {
    float** dptr = dev_ptr(gdn_ptr_pin_ + (size_t)li * 2 * kDecRows);
    gdn_decode_batch(p, proj, kGdnProjWidth, R, dptr, dptr + kDecRows, o, stream_);
  }
  lin(w.out, o, kGdnVWidth, R, out, H);
}

void QwenEngine::attn_layer(int l, const std::vector<RowInfo>& rows, const bf16* x, bf16* out, bool mtp_layer) {
  const AttnW& w = mtp_layer ? m_.mtp()->layer.att : m_.layer(l).att;
  const int R = (int)rows.size(), H = c_.hidden, ai = mtp_layer ? -1 : attn_index_[l];
  bf16* proj = proj_.as<bf16>();
  // graph phases: the row table of attention layer ai has a fixed slot (the ring would move it every step)
  QsaRow* qr = gphase_ ? qrows_pin_ + (size_t)(mtp_layer ? n_attn_ + mtp_slot_ : ai) * R : qrows_alloc(R);
  int max_pos = 0;
  for (int r = 0; r < R; ++r) {
    QwenSeq& s = *rows[r].s;
    const AttnCache& a = mtp_layer ? s.mtp : s.attn[ai];
    if (gphase_ != 2) qr[r] = QsaRow{a.view(s.rope.as<int32_t>()), (int)rows[r].pos};
    max_pos = std::max(max_pos, (int)rows[r].pos);
  }
  if (gphase_ == 1) return;
  lin(w.in, x, H, R, proj, kQsaProjWidth);
  const QsaWeights qw{w.q_norm.as<bf16>(), w.k_norm.as<bf16>(), w.iq_norm.as<bf16>(), w.ik_norm.as<bf16>(), c_.rms_eps};
  const QsaRope rope{rope_inv_freq_.as<float>(), rope_axis_.as<int32_t>()};
  // the row table goes to device memory first: the QSA kernels read it per thread / per block, and reading pinned host memory over PCIe
  //   cost ~130 µs per scoring call at 64K (nsys 2026-10-10: the decode scorer at 2.1 ms per MTP step)
  QsaRow* qd = qrows_dev_.as<QsaRow>() + (qr - qrows_pin_);
  CUDA_CHECK(cudaMemcpyAsync(qd, qr, (size_t)R * sizeof(QsaRow), cudaMemcpyHostToDevice, stream_));
  qr = qd;
  qsa_prep(qw, rope, qr, R, proj, kQsaProjWidth, q_.as<bf16>(), iq_.as<bf16>(), stream_);
  qsa_select(qr, R, max_pos, iq_.as<bf16>(), idx_.as<int32_t>(), ws_.p, ws_bytes_, stream_, gphase_ ? graph_ctx_ : 0,
             [&] { for (const RowInfo& r : rows) if (r.s != rows[0].s) return false; return true; }());
  if (qsa_split_ > 1 && R <= kDecRows)   // decode / verify rows: split-KV tensor-core attention (HIVE_QWEN_QSA_SPLIT)
    qsa_attention_split(qr, R, q_.as<bf16>(), proj, kQsaProjWidth, idx_.as<int32_t>(), attn_out_.as<bf16>(), qsa_split_,
                        qsa_split_ws_.p, stream_);   // the same split for every row count: a row's output does not depend on the batch
  else
    qsa_attention(qr, R, q_.as<bf16>(), proj, kQsaProjWidth, idx_.as<int32_t>(), attn_out_.as<bf16>(), stream_);
  lin(w.out, attn_out_.as<bf16>(), c_.q_width(), R, out, H);
}

void QwenEngine::moe_shared(const MoeW& w, const bf16* x, int rows, bf16* shared_out) {
  const int I = c_.shared_inter;
  lin(w.sh_gu, x, c_.hidden, rows, sh_gu_.as<bf16>(), 2 * I);
  glm::swiglu_rows(sh_gu_.as<bf16>(), rows, I, 0.f, sh_y_.as<bf16>(), stream_);
  lin(w.sh_down, sh_y_.as<bf16>(), I, rows, shared_out, c_.hidden);
}

// Device routing: the record pointer of every (MoE layer, expert) from the cache's slot map. Rebuilt on the host for each step that uses it
//   (24,576 pointer reads) and uploaded only when it differs from the last upload — residency changes through promotions, evictions, KV
//   growth, sleep / wake and warm starts, and a version counter in the store would have to be kept in step with every one of them.
bool QwenEngine::refresh_rec_table(bool allow_host, bool allow_null) {
  const int nk = ex_.n_keys();
  rec_host_.resize(nk);
  int nonres = 0;
  for (int k = 0; k < nk; ++k) {
    const int s = ex_.slot_of(k);
    if (s < 0) {
      if (!allow_host && !allow_null) return false;
      rec_host_[k] = allow_host ? ex_.host_rec(k) : nullptr;   // pinned host record (host-registered: the same address on the device) | CPU
      ++nonres;
      continue;
    }
    rec_host_[k] = ex_.dev_rec(s);
  }
  nonres_now_ = nonres;
  if (nonres && (mapped_nonres_ < 0 || std::abs(nonres - mapped_nonres_) * 10 > std::max(1, mapped_nonres_))) {
    if (allow_host)
      fprintf(stderr, "[qwen] device routing with %d of %d experts resident — the other %d are read from pinned host memory over PCIe\n", nk - nonres, nk,
              nonres);
    else
      fprintf(stderr, "[qwen] hybrid routing with %d of %d experts resident — the CPU computes the other %d (host nodes of the step)\n", nk - nonres, nk,
              nonres);
    mapped_nonres_ = nonres;
  }
  if (rec_up_ == rec_host_) return true;
  CUDA_CHECK(cudaMemcpyAsync(rec_tab_.p, rec_host_.data(), (size_t)nk * sizeof(void*), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  rec_up_ = rec_host_;
  return true;
}

// Device mirror of the pinned state-pointer tables (gdn rows, PLE states, verify snapshots): the kernels read pointers per thread, and
//   reading them from pinned host memory over PCIe cost 66 µs per gdn_conv_tail call at 8 decode rows (nsys 2026-10-10)
void QwenEngine::upload_ptr_tables() {
  if (!ptr_dev_.p) return;
  char* d = ptr_dev_.as<char>();
  CUDA_CHECK(cudaMemcpyAsync(d, gdn_ptr_pin_, ptr_gdn_n_ * sizeof(float*), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d + ptr_gdn_n_ * sizeof(float*), ple_ptr_pin_, kDecRows * sizeof(float*), cudaMemcpyHostToDevice, stream_));
  if (snap_ptr_pin_ && ptr_snap_n_)
    CUDA_CHECK(cudaMemcpyAsync(d + (ptr_gdn_n_ + kDecRows) * sizeof(float*), snap_ptr_pin_, ptr_snap_n_ * sizeof(float*), cudaMemcpyHostToDevice, stream_));
}
float** QwenEngine::dev_ptr(float** pin) const {
  float** d = ptr_dev_.as<float*>();
  if (pin >= gdn_ptr_pin_ && pin < gdn_ptr_pin_ + ptr_gdn_n_) return d + (pin - gdn_ptr_pin_);
  if (pin >= ple_ptr_pin_ && pin < ple_ptr_pin_ + kDecRows) return d + ptr_gdn_n_ + (pin - ple_ptr_pin_);
  if (snap_ptr_pin_ && pin >= snap_ptr_pin_ && pin < snap_ptr_pin_ + ptr_snap_n_) return d + ptr_gdn_n_ + kDecRows + (pin - snap_ptr_pin_);
  HIVE_CHECK(false, "qwen: pointer outside the mirrored tables");
  return nullptr;
}

bool QwenEngine::device_route_ready() const {
  return device_route_ && ex_.n_resident() == ex_.n_keys();
}

// mapped forward: this layer's routing → pinned host (a small D2H, a node of the decode / verify graph)
void QwenEngine::route_note(int li, int rows) {
  if (!map_now_) return;
  if (!route_h_) CUDA_CHECK(cudaMallocHost(&route_h_, (size_t)kDecRows * c_.n_act * c_.n_layers * 4));
  route_rows_ = rows;
  CUDA_CHECK(cudaMemcpyAsync(route_h_ + (size_t)li * kDecRows * c_.n_act, rids_.p, (size_t)rows * c_.n_act * 4, cudaMemcpyDeviceToHost, stream_));
}
// after a mapped forward (stream synchronized): the routing to the cache's bookkeeping; the caller's after_step then promotes
void QwenEngine::route_bookkeeping() {
  if (!map_now_ || !route_h_) return;
  for (int li = 0; li < ex_.n_moe_layers(); ++li) ex_.note_used(li, route_h_ + (size_t)li * kDecRows * c_.n_act, route_rows_ * c_.n_act);
  if (hyb_cpu_pairs_) {   // note_used counted the CPU's pairs as routed misses; cpu_rows counted them (routed + cpu) as well
    GlmCacheStats& st = ex_.stats();
    st.routed -= (uint64_t)hyb_cpu_pairs_; st.streamed -= std::min<uint64_t>(st.streamed, (uint64_t)hyb_cpu_pairs_);
    hyb_cpu_pairs_ = 0;
  }
}

// ---- hybrid route ------------------------------------------------------------------------------------------------------------------------
// layer li: the MoE input rows → pinned host (D2H), then the CPU share on the side stream as a host node; the reduce waits for hyb_ev_[2·li + 1]
void QwenEngine::hyb_fork(int li, int rows, const bf16* x) {
  (void)rows;
  CUDA_CHECK(cudaMemcpyAsync(hyb_x_, x, (size_t)hyb_rows_ * c_.hidden * 2, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaEventRecord(hyb_ev_[2 * li], stream_));
  CUDA_CHECK(cudaStreamWaitEvent(hyb_st_, hyb_ev_[2 * li], 0));
  CUDA_CHECK(cudaLaunchHostFunc(hyb_st_, &QwenEngine::hyb_host, &hyb_arg_[li]));
  CUDA_CHECK(cudaEventRecord(hyb_ev_[2 * li + 1], hyb_st_));
}
void CUDART_CB QwenEngine::hyb_host(void* arg) { auto* a = static_cast<HybArg*>(arg); a->e->hyb_cpu(a->li); }
// host node (no CUDA calls): the routing (rids_h_ / rw_h_, written by the top-k kernel) against this forward's record table — a pair whose
//   expert has no record runs on the CPU pool; its weighted result goes to hyb_y_ (zero rows when the layer has no CPU pair)
void QwenEngine::hyb_cpu(int li) {
  const int R = hyb_rows_, K = c_.n_act, E = c_.n_experts, H = c_.hidden, D = hyb_pcie_div_;
  if (D > 0) {   // the PCIe share's rule (group_ids2_kernel): non-resident experts of the layer ranked by id, rank % D == D − 1 → GPU
    std::fill(hyb_mrank_.begin(), hyb_mrank_.end(), 0);
    for (int p = 0; p < R * K; ++p) { const int e = rids_h_[p]; if (e >= 0 && e < E && !rec_host_[(size_t)li * E + e]) hyb_mrank_[e] = 1; }
    for (int e = 0, r = 0; e < E; ++e) { const int f = hyb_mrank_[e]; hyb_mrank_[e] = f ? r : -1; r += f; }
  }
  int n = 0;
  for (int p = 0; p < R * K; ++p) {
    const int e = rids_h_[p];
    bool cpu = e >= 0 && e < E && !rec_host_[(size_t)li * E + e];
    if (cpu && D > 0 && hyb_mrank_[e] % D == D - 1) cpu = false;
    hyb_ids_[p] = cpu ? e : -1;
    n += cpu;
  }
  std::memset(hyb_y_, 0, (size_t)R * H * 4);
  if (!n || hyb_diag_nocpu_) return;
  for (size_t i = 0; i < (size_t)R * H; ++i) hyb_xf_[i] = bf2f(hyb_x_[i]);
  ex_.cpu_rows(li, R, hyb_ids_.data(), rw_h_, hyb_xf_.data(), hyb_y_);
  hyb_cpu_pairs_ += n;
}

void QwenEngine::moe_device(int li, int rows, const bf16* x, float* out) {
  const int K = c_.n_act;
  route_note(li, rows);
  build_moe_pairs(rids_.as<int32_t>(), rw_.as<float>(), rows, K, reinterpret_cast<const uint8_t* const*>(rec_tab_.p), li, ex_.n_experts(),
                  pairs_.as<MoePairDev>(), stream_);
  if (fn_moe_)
    fn_moe_decode(ex_.layout(), reinterpret_cast<const glm::MoePair*>(pairs_.p), rows * K, x, c_.hidden, c_.moe_inter, out, moe_ws_.p, stream_);
  else
    glm::moe_decode(ex_.layout(), reinterpret_cast<const glm::MoePair*>(pairs_.p), rows * K, x, c_.hidden, c_.moe_inter, 0.f, out, moe_ws_.p, stream_);
  GlmCacheStats& st = ex_.stats();
  st.routed += (uint64_t)rows * K; st.hit += (uint64_t)rows * K;
}

void QwenEngine::moe_layer(int l, int rows, bool prefill, const bf16* x, bf16* out) {
  const MoeW& w = m_.layer(l).moe;
  const int H = c_.hidden, E = c_.n_experts, K = c_.n_act, ld = E + 1;
  float* lg = router_.as<float>();
  float* acc = mlp_out_f32_.as<float>();
  router_logits(w.router, x, rows, lg, ld);
  const int li = l;  // every layer is a MoE layer (GlmExperts' MoE index = layer)
  const bool dev = dr_now_ && !prefill && rows <= 64;
  if (dev && fuse_ && rows <= kDecRows) {   // fused decode MoE (see the constructor): 5 launches
    const int SI = c_.shared_inter;
    FnEpi e; e.mode = 3; e.yf = lg; e.ldyf = ld; e.nr = ld; e.I = SI; e.ys = sh_y_.as<bf16>();
    fn_gemv_epi(rs_[l].as<bf16>(), ld + 2 * SI, H, x, H, rows, e, gemv_ws_.as<float>(), stream_);
    router_softmax_topk(lg, ld, rows, E, K, c_.norm_topk, rids_.as<int32_t>(), rw_.as<float>(), hyb_now_ ? rids_h_ : nullptr, hyb_now_ ? rw_h_ : nullptr,
                        stream_);
    route_note(li, rows);
    if (hyb_now_) hyb_fork(li, rows, x);   // the CPU share runs while the shared and the resident experts run here
    lin(w.sh_down, sh_y_.as<bf16>(), SI, rows, shared_out_.as<bf16>(), H);
    fn_moe_decode_routed(ex_.layout(), rids_.as<int32_t>(), rw_.as<float>(), rows, K, reinterpret_cast<const uint8_t* const*>(rec_tab_.p), li,
                         ex_.n_experts(), x, H, c_.moe_inter, shared_out_.as<bf16>(), lg, ld, E, out, moe_ws_.p, stream_,
                         hyb_now_ ? hyb_y_ : nullptr, hyb_now_ ? hyb_ev_[2 * li + 1] : nullptr,
                         hyb_now_ && hyb_pcie_div_ > 0 ? reinterpret_cast<const uint8_t* const*>(host_tab_.p) : nullptr, hyb_now_ ? hyb_pcie_div_ : 0);
    if (!hyb_now_) {   // (hybrid: route_bookkeeping counts, after the step — the host nodes update the store's counters meanwhile)
      GlmCacheStats& cs = ex_.stats();
      cs.routed += (uint64_t)rows * K; cs.hit += (uint64_t)rows * K;
    }
    return;
  }
  if (prefill && pf_dev_) {   // FN grouped prefill: routing stays on the device, out = Σ_k part·w (fn_moe_pf.cu)
    router_softmax_topk(lg, ld, rows, E, K, c_.norm_topk, rids_.as<int32_t>(), rw_.as<float>(), nullptr, nullptr, stream_);
    moe_shared(w, x, rows, shared_out_.as<bf16>());
    fn_moe_prefill(ex_.layout(), reinterpret_cast<const uint8_t* const*>(rec_tab_.p), li, E, rids_.as<int32_t>(), rw_.as<float>(), rows, K, x, H,
                   c_.moe_inter, acc, pf_ws_.p, pf_rows_, stream_);
    GlmCacheStats& cs = ex_.stats();
    cs.routed += (uint64_t)rows * K; cs.hit += (uint64_t)rows * K;
    moe_combine(acc, shared_out_.as<bf16>(), lg, ld, E, rows, H, out, stream_);
    return;
  }
  CUDA_CHECK(cudaMemsetAsync(acc, 0, (size_t)rows * H * 4, stream_));
  if (dev) {
    router_softmax_topk(lg, ld, rows, E, K, c_.norm_topk, rids_.as<int32_t>(), rw_.as<float>(), nullptr, nullptr, stream_);
    moe_shared(w, x, rows, shared_out_.as<bf16>());
    moe_device(li, rows, x, acc);
  } else {
    const bool mapped_out = rows <= 64;
    router_softmax_topk(lg, ld, rows, E, K, c_.norm_topk, rids_.as<int32_t>(), rw_.as<float>(), mapped_out ? rids_h_ : nullptr, mapped_out ? rw_h_ : nullptr,
                        stream_);
    moe_shared(w, x, rows, shared_out_.as<bf16>());
    h_ids_.resize((size_t)rows * K); h_w_.resize((size_t)rows * K);
    if (!mapped_out) {
      CUDA_CHECK(cudaMemcpyAsync(h_ids_.data(), rids_.p, (size_t)rows * K * 4, cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaMemcpyAsync(h_w_.data(), rw_.p, (size_t)rows * K * 4, cudaMemcpyDeviceToHost, stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));  // routing on the host
    if (mapped_out) { std::memcpy(h_ids_.data(), rids_h_, (size_t)rows * K * 4); std::memcpy(h_w_.data(), rw_h_, (size_t)rows * K * 4); }
    const double t0 = hive::mono_ms();
    if (prefill && rows > 64) ex_.prefill_layer(li, rows, h_ids_.data(), h_w_.data(), x, acc, stream_);
    else ex_.decode_layer(li, rows, h_ids_.data(), h_w_.data(), x, acc, stream_);
    st_.ms_moe += hive::mono_ms() - t0;
  }
  moe_combine(acc, shared_out_.as<bf16>(), lg, ld, E, rows, H, out, stream_);
}

// router logits (fp32): decode rows on fn_gemv (measured ≈ 16 µs per call on cuBLAS's GEMV at M ≤ 4), else cuBLAS
void QwenEngine::router_logits(const QMat& W, const bf16* x, int rows, float* lg, int ld) {
  if (rows <= kDecRows && fn_gemv_ok(W.out, W.in, rows) && fn_gemv_ws_bytes(W.out, W.in, rows) <= gemv_ws_.n && ((uintptr_t)x & 15) == 0) {
    fn_gemv(W.p(), W.out, W.in, x, W.in, rows, nullptr, ld, lg, gemv_ws_.as<float>(), stream_);
    return;
  }
  blas_.gemm_bf16_f32out(x, W.p(), lg, rows, ld, W.in);
}

void QwenEngine::head(const bf16* h, int rows, float* logits_host) {
  HIVE_CHECK(rows >= 1 && rows <= 64, "qwen: head rows");
  hc_mix(m_.mixer(), h, rows, xfin_.as<bf16>());
  const QMat& W = m_.lm_head();
  if (W.fmt8 && rows <= kDecRows && fn_gemv8_ok(W.out, W.in, W.gs8, rows))   // HIVE_QWEN_DENSE8_HEAD: the 8-bit copy, fp32 logits
    fn_gemv8(W.q8.as<uint8_t>(), W.s8.as<float>(), W.fmt8, W.gs8, W.out, W.in, xfin_.as<bf16>(), c_.hidden, rows, nullptr, c_.vocab, stream_,
             logits_dev_.as<float>());
  else if (rows <= kDecRows && c_.hidden % 32 == 0)   // bf16 head: batch-invariant GEMV (cuBLAS picked other algorithms per row count)
    fn_gemv_tc(W.p(), c_.vocab, c_.hidden, xfin_.as<bf16>(), c_.hidden, rows, nullptr, c_.vocab, logits_dev_.as<float>(), stream_);
  else
    blas_.gemm_bf16_f32out(xfin_.as<bf16>(), W.p(), logits_dev_.as<float>(), rows, c_.vocab, c_.hidden);
  if (logits_host) CUDA_CHECK(cudaMemcpyAsync(logits_host, logits_dev_.p, (size_t)rows * c_.vocab * 4, cudaMemcpyDeviceToHost, stream_));
}

// ---- forward ----------------------------------------------------------------------------------------------------------------------------
void QwenEngine::layer_front(int l, const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, bf16* h) {
  const LayerW& L = m_.layer(l);
  const int R = (int)rows.size();
  if (L.ple) ple_block(rows, ids, h, prefill);
  hc_mix(L.hc_attn, h, R, x_.as<bf16>(), pre_normed_);   // pre_normed_: the previous layer's combine_mlp produced hn_
  pre_normed_ = false;
  if (L.attn) attn_layer(l, rows, x_.as<bf16>(), y_.as<bf16>());
  else lin_layer(l, rows, prefill, x_.as<bf16>(), y_.as<bf16>());
  // the attention side's combine and the MLP side's input norm in one launch (bit-identical; HIVE_QWEN_HC_FUSE=0: two launches)
  if (hc_fuse()) {
    hc_combine_norm(h, y_.as<bf16>(), c_.hidden, dl_.as<bf16>(), L.hc_attn.down_inj.out, c_.hc_rank, R, c_.hc, c_.hidden, L.hc_mlp.norm.as<bf16>(),
                    c_.rms_eps, hn_.as<bf16>(), stream_);
    hc_mix(L.hc_mlp, h, R, x_.as<bf16>(), true);
  } else {
    hc_combine_(L.hc_attn, h, y_.as<bf16>(), R, dl_.as<bf16>());
    hc_mix(L.hc_mlp, h, R, x_.as<bf16>());   // x_ = MoE input, dl_ = the MLP side's inject logits
  }
}

void QwenEngine::forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, float* logits_host, bool all_logits) {
  const int R = (int)rows.size(), HC = c_.hcw();
  HIVE_CHECK(R >= 1 && R <= max_chunk_, "qwen: forward rows");
  const double t0 = hive::mono_ms();
  if (!prefill) {  // per-step state pointer tables (pinned, read by the kernels through UVA)
    HIVE_CHECK(R <= kDecRows, "qwen: decode batch > kDecRows");
    for (int li = 0; li < n_lin_; ++li)
      for (int r = 0; r < R; ++r) {
        gdn_ptr_pin_[(size_t)li * 2 * kDecRows + r] = rows[r].s->conv[li].as<float>();
        gdn_ptr_pin_[(size_t)li * 2 * kDecRows + kDecRows + r] = rows[r].s->state[li].as<float>();
      }
  }
  // decided once per forward: residency changes only between forwards (after_step, kv_reserve and make_room run outside forward())
  const bool mapped_ok = mapped_route_ && ex_.n_slots() >= mapped_min_ * ex_.n_keys();
  dr_now_ = !prefill && device_route_ && (device_route_ready() || mapped_ok) && refresh_rec_table(mapped_ok);
  hyb_now_ = false;
  if (!prefill && !dr_now_ && hybrid_route_ && R <= kDecRows && (!hyb_max_rows_ || R <= hyb_max_rows_)) dr_now_ = hyb_now_ = refresh_rec_table(false, true);
  if (hyb_now_ && hyb_pcie_div_ > 0 && !host_tab_.p) {   // host records never move: built once
    std::vector<const uint8_t*> t(ex_.n_keys());
    for (int k = 0; k < ex_.n_keys(); ++k) t[k] = ex_.host_rec(k);
    host_tab_.alloc(t.size() * sizeof(void*));
    CUDA_CHECK(cudaMemcpy(host_tab_.p, t.data(), t.size() * sizeof(void*), cudaMemcpyHostToDevice));
  }
  hyb_rows_ = R;
  map_now_ = dr_now_ && nonres_now_ > 0;   // (hybrid too: route_note / route_bookkeeping feed the cache's scores and promotions)
  pf_dev_ = prefill && fn_pf_ && device_route_ready() && refresh_rec_table();
  if (!prefill) { if (dr_now_) ++st_.device_route_steps; else ++st_.host_route_steps; if (hyb_now_) ++st_.hybrid_route_steps; }
  if (!prefill && graph_forward(rows, ids, logits_host, all_logits)) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    route_bookkeeping();
    ++st_.steps;
    if (mtp_on()) mtp_after_forward(rows, ids, prefill);
    for (int r = 0; r < R; ++r) { QwenSeq& s = *rows[r].s; s.pos = rows[r].pos + 1; s.tokens.push_back(ids[r]); }
    st_.ms_total += hive::mono_ms() - t0;
    return;
  }
  if (!prefill) upload_ptr_tables();
  set_rope(rows, emb_base_, prefill);
  std::vector<int32_t> tail_coords;   // the rows' rope coordinates for the CPU tail (rope_pin_ is reused by forwards run in a layer yield)
  if (tail_) tail_coords.assign(rope_pin_, rope_pin_ + (size_t)R * 3);
  bf16* h = hb();
  embed(rows, ids, emb_base_, h);
  for (int l = 0; l < m_.gpu_layers(); ++l) {
    if (l > 0 && layer_inject) { CUDA_CHECK(cudaStreamSynchronize(stream_)); layer_inject(l, h, R); }
    layer_front(l, rows, ids, prefill, h);
    moe_layer(l, R, prefill, x_.as<bf16>(), y_.as<bf16>());
    combine_mlp(l, h, R, !layer_hook && !layer_inject && !(prefill && layer_boundary && layer_boundary_yields));
    if (layer_hook) { CUDA_CHECK(cudaStreamSynchronize(stream_)); layer_hook(l, h, R); }
    if (prefill && layer_boundary && l + 1 < c_.n_layers) layer_boundary(l);
  }
  if (tail_) run_tail(rows, ids, prefill, tail_coords.data(), h);
  if (mtp_on()) CUDA_CHECK(cudaMemcpyAsync(fh_.p, h, (size_t)R * HC * 2, cudaMemcpyDeviceToDevice, stream_));
  const int r0 = all_logits ? 0 : R - 1, nr = all_logits ? R : 1;
  head(h + (size_t)r0 * HC, nr, logits_host);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  route_bookkeeping();
  if (!prefill) ++st_.steps;
  if (mtp_on()) mtp_after_forward(rows, ids, prefill);
  for (int r = 0; r < R; ++r) { QwenSeq& s = *rows[r].s; s.pos = rows[r].pos + 1; s.tokens.push_back(ids[r]); }
  st_.ms_total += hive::mono_ms() - t0;
}

void QwenEngine::prefill(QwenSeq& s, const int32_t* ids, int T, float* logits_last) {
  HIVE_CHECK(T >= 1 && s.pos + T <= s.cap, "qwen: sequence capacity exceeded");
  HIVE_CHECK((int64_t)s.tokens.size() == s.pos, "qwen: sequence tokens out of step with its position");
  // boundary_base/span (qwen_engine.h): restored on exit — a short prefill admitted inside a layer yield of this one runs this function too
  struct BoundaryKeep { QwenEngine& e; double base, span; ~BoundaryKeep() { e.boundary_base = base; e.boundary_span = span; } } boundary_keep{*this, boundary_base, boundary_span};
  kv_reserve(s, std::min<int64_t>(s.cap, s.pos + T + kKvSlack));
  if (ple_ && strcmp(ple_->kind(), "ram") != 0) {  // SSD: read the whole prompt's rows ahead (they depend on the token ids only)
    std::vector<int64_t> pr((size_t)T * kPleHeads);
    const int32_t prev1 = s.pos >= 1 ? s.tokens[s.pos - 1] : -1, prev2 = s.pos >= 2 ? s.tokens[s.pos - 2] : -1;
    ple_hash_.rows_of(ids, T, prev2, prev1, pr.data());
    ple_->prefetch(pr.data(), T);
  }
  // layer-major over several chunks only while some expert is not resident (it exists to stream each missing expert once per layer), and only
  //   over as many rows as half of the expert cache can lend (prefill_layer_major borrows its whole-prompt buffers from the tail slots — a
  //   small budget's cache cannot lend 64K rows' worth, ~2.5 GB)
  int span = tiles_ * max_chunk_;
  if (tiles_ > 1 && T > max_chunk_ && !device_route_ready()) {
    const size_t per_row = (size_t)c_.hcw() * 2 + (size_t)c_.hidden * 6 + (size_t)(c_.hc_rank + c_.hc) * 2 + (size_t)(c_.n_experts + 1) * 4 +
                           (size_t)c_.n_act * 8 + 64;
    const size_t lendable = (size_t)std::max(0, ex_.n_slots() / 2) * ex_.layout().total;
    span = (int)std::min<size_t>(span, lendable / per_row / max_chunk_ * max_chunk_);
  }
  if (tiles_ > 1 && T > max_chunk_ && !device_route_ready() && span > max_chunk_ && !tail_) {   // (layer-major: GPU layers only)
    for (int off = 0; off < T; off += span) {
      const int n = std::min(span, T - off);
      emb_base_ = off;
      boundary_base = (double)off / T; boundary_span = (double)n / T;
      if (n > max_chunk_) prefill_layer_major(s, ids + off, n, off + n == T ? logits_last : nullptr);
      else {
        std::vector<RowInfo> rows(n);
        for (int i = 0; i < n; ++i) rows[i] = {&s, s.pos + i};
        forward(rows, ids + off, true, off + n == T ? logits_last : nullptr, false);
      }
    }
  } else {
    for (int off = 0; off < T; off += max_chunk_) {
      const int n = std::min(max_chunk_, T - off);
      emb_base_ = off;
      boundary_base = (double)off / T; boundary_span = (double)n / T;
      std::vector<RowInfo> rows(n);
      for (int i = 0; i < n; ++i) rows[i] = {&s, s.pos + i};
      forward(rows, ids + off, true, off + n == T ? logits_last : nullptr, false);
    }
  }
  emb_base_ = 0;
  ex_.warm_quota(2048);
}

void QwenEngine::decode(QwenSeq* const* seqs, const int32_t* ids, int M, float* logits) {
  std::vector<RowInfo> rows(M);
  for (int m = 0; m < M; ++m) {
    HIVE_CHECK(seqs[m]->pos + 1 <= seqs[m]->cap, "qwen: sequence capacity exceeded");
    for (int q = 0; q < m; ++q) HIVE_CHECK(seqs[q] != seqs[m], "qwen decode: a sequence appears twice");
    kv_reserve(*seqs[m], std::min<int64_t>(seqs[m]->cap, seqs[m]->pos + 1 + kKvSlack));
    rows[m] = {seqs[m], seqs[m]->pos};
  }
  forward(rows, ids, false, logits, true);
  ex_.after_step();
}

// ---- CUDA graphs of decode / verify forwards (HIVE_QWEN_GRAPH, default on) ---------------------------------------------------------------
// The device part of a decode or verify forward (embedding through the logits copy) is captured once per shape and replayed: the host part —
//   rope coordinates, embedding rows, PLE rows, attention row tables (fixed slots), GDN / PLE state pointers — is redone before every replay
//   into the same pinned buffers the graph reads. Only while every expert is resident (device routing: no host round trip inside) and
//   without a CPU tail. Key = rows, verify parts, scoring on/off, context bucket (the attention selection's score layout), logits target,
//   stream buffer. Measured: see docs/qwen.md.
bool QwenEngine::graph_forward(const std::vector<RowInfo>& rows, const int32_t* ids, float* logits_host, bool all_logits) {
  static const bool on = !env_off("HIVE_QWEN_GRAPH");
  if (!on || !dr_now_ || tail_ || !emb_over_.empty() || layer_hook || layer_inject) return false;   // (validation hooks: eager)
  const int R = (int)rows.size(), HC = c_.hcw();
  if (R > kDecRows) return false;
  // HIVE_QWEN_GRAPH_ROWS (default 0 = any): forwards with more rows run eager. Each captured graph holds ~10 MB of device memory, which a
  //   VRAM budget takes from the expert cache; batched verify steps have many shapes (rows × part layout). Measured at 32 GB (11 % of the
  //   experts resident, hybrid route, short pass 2026-10-13): graphs on c1 / c4 / c8 89.5 / 98.4 / 116.9, all eager 87.7 / 105.9 / 125.9
  //   (host route 76.6 / 107.6 / 128.1) — the profiles of the small cards set it.
  static const int graph_rows = getenv("HIVE_QWEN_GRAPH_ROWS") ? std::max(0, atoi(getenv("HIVE_QWEN_GRAPH_ROWS"))) : 0;
  if (graph_rows && R > graph_rows) return false;
  if (logits_host) {   // the logits copy is a graph node: its host buffer must be pinned
    cudaPointerAttributes pa{};
    if (cudaPointerGetAttributes(&pa, logits_host) != cudaSuccess || pa.type != cudaMemoryTypeHost) { cudaGetLastError(); return false; }
  }
  int max_pos = 0;
  for (const RowInfo& ri : rows) max_pos = std::max(max_pos, (int)ri.pos);
  const bool scoring = (max_pos + 1) / kQsaRatio > kQsaTopBlocks;
  const int ctx = scoring ? (max_pos + 1 + 4095) / 4096 * 4096 : 0;
  std::string key = std::to_string(R) + (verify_ ? "v" : "d") + std::to_string(scoring) + ":" + std::to_string(ctx) + ":" + std::to_string(all_logits) + ":" +
                    std::to_string((uintptr_t)logits_host) + ":" + std::to_string((uintptr_t)hb());
  if (verify_) for (const VPart& vp : vparts_) key += "," + std::to_string(vp.M) + "@" + std::to_string(vp.r0);
  if (map_now_) key += ":map";   // with the routing D2H nodes
  if (hyb_now_) key += ":hyb";   // with the CPU host nodes
  graph_ctx_ = ctx;
  // host part: the same pinned buffers the captured kernels read (earlier work on the stream may still read them — wait for it first)
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  gphase_ = 1;
  set_rope(rows, emb_base_, false);
  embed(rows, ids, emb_base_, hb());
  for (int l = 0; l < c_.n_layers; ++l) {
    const LayerW& L = m_.layer(l);
    if (L.ple) ple_block(rows, ids, hb(), false);
    if (L.attn) attn_layer(l, rows, nullptr, nullptr);
  }
  gphase_ = 0;
  auto it = graphs_.find(key);
  if (it == graphs_.end()) {
    // first use of a shape: eager (lazy initialisation, attribute calls), second: capture
    if (graph_seen_[key]++ < 1) return false;
    gphase_ = 2;
    cudaGraph_t g = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    try {
      graph_body(rows, ids, logits_host, all_logits);
    } catch (...) {
      cudaGraph_t bad = nullptr;
      cudaStreamEndCapture(stream_, &bad);
      if (bad) cudaGraphDestroy(bad);
      cudaGetLastError();
      gphase_ = 0;
      throw;
    }
    gphase_ = 0;
    CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
    cudaGraphExec_t ex = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&ex, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    it = graph_put(key, ex);
    (void)HC;
  }
  graph_used(key);
  CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  return true;
}

// the device part of forward() for decode / verify rows (gphase_ = 2: the host parts were done by graph_forward)
void QwenEngine::graph_body(const std::vector<RowInfo>& rows, const int32_t* ids, float* logits_host, bool all_logits) {
  const int R = (int)rows.size(), HC = c_.hcw();
  upload_ptr_tables();   // a memcpy node: every replay copies the pinned tables the host part just filled
  set_rope(rows, emb_base_, false);
  bf16* h = hb();
  embed(rows, ids, emb_base_, h);
  for (int l = 0; l < c_.n_layers; ++l) {
    layer_front(l, rows, ids, false, h);
    moe_layer(l, R, false, x_.as<bf16>(), y_.as<bf16>());
    combine_mlp(l, h, R, true);
  }
  if (mtp_on()) CUDA_CHECK(cudaMemcpyAsync(fh_.p, h, (size_t)R * HC * 2, cudaMemcpyDeviceToDevice, stream_));
  const int r0 = all_logits ? 0 : R - 1, nr = all_logits ? R : 1;
  head(h + (size_t)r0 * HC, nr, logits_host);
}

std::map<std::string, cudaGraphExec_t>::iterator QwenEngine::graph_put(const std::string& key, cudaGraphExec_t ex) {
  if (!graph_cap_set_) {   // decided at the first capture: the expert cache is sized by then (HIVE_CACHE_FIT runs after the engine exists)
    graph_cap_set_ = true;
    if (const char* v = getenv("HIVE_QWEN_GRAPH_CAP"); v && *v) graph_cap_ = (size_t)std::max(0, atoi(v));
    else if (ex_.max_slots() < ex_.n_keys()) graph_cap_ = 32;
    if (graph_cap_) fprintf(stderr, "[qwen] CUDA graphs: at most %zu kept (least recently used dropped first)\n", graph_cap_);
  }
  if (graph_cap_ && graphs_.size() >= graph_cap_) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));   // (no graph of this engine is in flight while one is destroyed)
    while (graphs_.size() >= graph_cap_) {
      auto lru = graphs_.begin();
      uint64_t best = UINT64_MAX;
      for (auto g = graphs_.begin(); g != graphs_.end(); ++g) {
        const auto t = graph_tick_.find(g->first);
        const uint64_t v = t == graph_tick_.end() ? 0 : t->second;
        if (v < best) { best = v; lru = g; }
      }
      cudaGraphExecDestroy(lru->second);
      graph_tick_.erase(lru->first);
      graphs_.erase(lru);
    }
  }
  return graphs_.emplace(key, ex).first;
}

void QwenEngine::clear_graphs() {
  for (auto& [k, ex] : graphs_) cudaGraphExecDestroy(ex);
  graphs_.clear();
  graph_tick_.clear();
  graph_seen_.clear();
}

// ---- hybrid placement ------------------------------------------------------------------------------------------------------------------------
// The GPU layers' streams go to the CPU engine and back (bf16 values both ways: exact), with the rows' rope coordinates (set_rope's pinned
//   table) and token ids. The tail runs its layers' MoE on its own expert copies; nothing of it touches the GPU expert cache.
void QwenEngine::run_tail(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill, const int32_t* coords, bf16* h) {
  const int R = (int)rows.size(), HC = c_.hcw();
  CUDA_CHECK(cudaMemcpyAsync(tail_pin_, h, (size_t)R * HC * 2, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  tail_h_.resize((size_t)R * HC);
  for (size_t i = 0; i < tail_h_.size(); ++i) tail_h_[i] = bf2f(tail_pin_[i]);
  std::vector<cpu::CpuSeq*> seqs(R);
  std::vector<int64_t> pos(R);
  for (int r = 0; r < R; ++r) {
    HIVE_CHECK(rows[r].s->tail, "qwen: sequence without CPU-tail state");
    seqs[r] = rows[r].s->tail.get(); pos[r] = rows[r].pos;
  }
  const double t0 = hive::mono_ms();
  tail_->run_layers(seqs.data(), pos.data(), R, ids, prefill, coords, tail_h_.data());
  st_.ms_tail += hive::mono_ms() - t0;
  for (size_t i = 0; i < tail_h_.size(); ++i) tail_pin_[i] = __float2bfloat16(tail_h_[i]);   // bf16 values: exact
  CUDA_CHECK(cudaMemcpyAsync(h, tail_pin_, (size_t)R * HC * 2, cudaMemcpyHostToDevice, stream_));
}

// ---- layer yields -------------------------------------------------------------------------------------------------------------------------
void QwenEngine::alloc_yield_rows(int rows) {
  if (rows <= hy_rows_) return;
  hy_.alloc((size_t)rows * c_.hcw() * 2);
  hy_rows_ = rows;
}

bool QwenEngine::yield_enter(YieldSave& sv) {
  if (hcur_ && hcur_ == hy_.as<bf16>()) return false;
  if (!hcur_ && !hy_.p) return false;
  sv.hcur = hcur_; sv.emb = std::move(emb_over_); sv.emb_base = emb_base_; sv.on = true;
  emb_over_.clear(); emb_base_ = 0;
  hcur_ = sv.hcur ? nullptr : hy_.as<bf16>();
  return true;
}

void QwenEngine::yield_exit(YieldSave& sv) {
  if (!sv.on) return;
  hcur_ = sv.hcur; emb_over_ = std::move(sv.emb); emb_base_ = sv.emb_base; sv.on = false;
}

}  // namespace hive::qwen
