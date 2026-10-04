// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Verifies HIVE_DECODE_ATTN_FUSED (k::dec_attention_front) against the baseline fuse_ decode attention front (legacy_front, a verbatim port of the
//   `if (fuse_)` chain in runtime attention_decode_dev) — real layer shapes (dim 5120, H 64, D 512, q_lora 1280, indexer 32x128, top 512, win 128),
//   synthetic KV state (per-row ring, packed compressed KV, packed index keys, compressor state), context 4K/32K/128K/262K x M 1/4/8 x 5 layer kinds:
//     W   = window layer (ratio 0, layers 0/1)            C   = compressed consumer layer (ratio 1, 21-23 ..., index from the previous index layer)
//     K2  = kv+index source ratio 2 (layers 2/8/14)       K20 = kv+index source ratio 1, candidate source (layer 20)   U = candidate-pool index layer (24/28/32/36)
//     Utc = U with the HIVE_IDX_TC rule (on in production) — packed query (iqp) + indexer_scores_cand_tc (same kernel as before)
//   Compared (bitwise): qr, qrn, q, qrq, qrs, kv, iq, iw, iqp, idx (all window + top-k columns = selected set and order), o, ring slots, compressed cache rows, index-key rows, compressor state, valid, cand.
//   Also: a "ties" case at 32K where index keys use only 4 patterns (thousands of ties — also exercises the baseline kernel's tie fallback path).
//   Timing: one layer's front (q, kv, compressor, indexer, attention — wo_a/wo_b are common to both and excluded), eager and CUDA graph replay, 20-run averages each (us).
// Run (when the GPU is free): scripts/hive-run.sh "./build-dev/test_decode_attn [ctx,ctx,...] [M,M,...]"   (default 4096,32768,131072,262144 and 1,4,8)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hive/attn_decode_fused.h"
#include "hive/cublas_ops.h"
#include "hive/kernels.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

using namespace hive;

namespace {

constexpr int DIM = 5120, H = 64, D = 512, RD = 64, WIN = 128, QL = 1280, HI = 32, DI = 128, TOPK = 512, BS = 8, CB = 2048, SPLITS = 4;
constexpr float EPS = 1e-20f;
constexpr int CAND_SRC = 20;

__device__ __forceinline__ uint32_t hsh(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
__device__ __forceinline__ float u01(uint32_t x) { return (hsh(x) >> 8) * (1.0f / 16777216.0f); }
// e4m3 weight codes (no NaN): sign, exponent 4..9, mantissa
__global__ void fill_e4m3(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) {
    const uint32_t h = hsh((uint32_t)i * 2654435761u ^ seed);
    p[i] = (uint8_t)(((h >> 8) & 0x80u) | ((4u + (h % 6u)) << 3) | ((h >> 4) & 7u));
  }
}
__global__ void fill_u8c(uint8_t* p, size_t n, uint8_t lo, uint8_t span, uint32_t seed) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size())
    p[i] = (uint8_t)(lo + hsh((uint32_t)i ^ seed) % span);
}
__global__ void fill_f32(float* p, size_t n, float a, float b, uint32_t seed) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) p[i] = a + (b - a) * u01((uint32_t)i ^ seed);
}
__global__ void fill_bf16(bf16* p, size_t n, float a, float b, uint32_t seed) {
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) p[i] = f2bf(a + (b - a) * u01((uint32_t)i ^ seed));
}
// Packed compressed KV rows [rows, 288]: e4m3 scales (around 0.03-0.5), random nibbles
__global__ void fill_comp(uint8_t* p, size_t rows, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * kvp::COMP_ROW; i += (size_t)gridDim.x * blockDim.x) {
    const int j = (int)(i % kvp::COMP_ROW);
    const uint32_t h = hsh((uint32_t)i ^ seed);
    p[i] = j < kvp::COMP_SCALES ? (uint8_t)(0x20 + h % 24) : (uint8_t)h;
  }
}
// Packed index-key rows [rows, 80]: e8m0 scales 120..126, zero padding, random nibbles (ties = 4 row patterns)
__global__ void fill_idxk(uint8_t* p, size_t rows, uint32_t seed, int ties) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * kvp::IDX_ROW; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / kvp::IDX_ROW;
    const int j = (int)(i % kvp::IDX_ROW);
    const uint32_t key = ties ? (uint32_t)((r % 4) * kvp::IDX_ROW + j) : (uint32_t)i;
    const uint32_t h = hsh(key ^ seed);
    p[i] = j < kvp::IDX_SCALES ? (uint8_t)(120 + h % 7) : j < kvp::IDX_HDR ? (uint8_t)0 : (uint8_t)h;
  }
}
__global__ void fill_rope(float2* f, int npos, int half, float theta) {
  const size_t n = (size_t)npos * half;
  for (size_t i = hive::cu::global_tid(); i < n; i += hive::cu::grid_size()) {
    const int p = (int)(i / half), k = (int)(i % half);
    const float ang = (float)p * powf(theta, -2.0f * k / (2.0f * half));
    float s, c;
    sincosf(fmodf(ang, 6.2831853f), &s, &c);
    f[i] = make_float2(c, s);
  }
}
void grid_fill(size_t n, int& g) { g = (int)std::min<size_t>(4096, (n + 255) / 256); if (g < 1) g = 1; }

template <class T>
T* dalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc(&p, n * sizeof(T) + 64)); CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T) + 64)); return p; }
template <class T>
T* mapped(size_t n, T** dev) { T* p; CUDA_CHECK(cudaHostAlloc((void**)&p, n * sizeof(T), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void**)dev, p, 0)); memset(p, 0, n * sizeof(T)); return p; }
template <class T>
std::vector<uint8_t> snap(const T* d, size_t n) { std::vector<uint8_t> v(n * sizeof(T)); CUDA_CHECK(cudaMemcpy(v.data(), d, v.size(), cudaMemcpyDeviceToHost)); return v; }
void put(void* d, const std::vector<uint8_t>& v) { CUDA_CHECK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice)); }

struct Fp8 { uint8_t* w; uint8_t* s; int N, K; };
Fp8 make_fp8(int N, int K, uint32_t seed) {
  Fp8 f{dalloc<uint8_t>((size_t)N * K), dalloc<uint8_t>((size_t)N * (K / 32)), N, K};
  int g; grid_fill((size_t)N * K, g);
  fill_e4m3<<<g, 256>>>(f.w, (size_t)N * K, seed);
  fill_u8c<<<g, 256>>>(f.s, (size_t)N * (K / 32), 117, 4, seed * 7 + 1);
  return f;
}
bf16* make_bf16(size_t n, float a, float b, uint32_t seed) { bf16* p = dalloc<bf16>(n); int g; grid_fill(n, g); fill_bf16<<<g, 256>>>(p, n, a, b, seed); return p; }
float* make_f32(size_t n, float a, float b, uint32_t seed) { float* p = dalloc<float>(n); int g; grid_fill(n, g); fill_f32<<<g, 256>>>(p, n, a, b, seed); return p; }

struct LayerW {
  Fp8 wqa, wqb, wkv, idx_wqb;
  bf16 *q_norm, *kv_norm, *comp_norm, *idx_wproj, *idx_wk, *idx_k_norm, *comp_wkv_bf;
  float *sink, *comp_wkv_f, *comp_wgate;
};
LayerW make_layer(uint32_t seed) {
  LayerW L;
  L.wqa = make_fp8(QL, DIM, seed + 1); L.wqb = make_fp8(H * D, QL, seed + 2); L.wkv = make_fp8(D, DIM, seed + 3); L.idx_wqb = make_fp8(HI * DI, QL, seed + 4);
  L.q_norm = make_bf16(QL, 0.8f, 1.2f, seed + 5); L.kv_norm = make_bf16(D, 0.8f, 1.2f, seed + 6); L.comp_norm = make_bf16(D, 0.8f, 1.2f, seed + 7);
  L.idx_wproj = make_bf16((size_t)HI * DIM, -0.03f, 0.03f, seed + 8); L.idx_wk = make_bf16((size_t)DI * D, -0.08f, 0.08f, seed + 9);
  L.idx_k_norm = make_bf16(DI, 0.8f, 1.2f, seed + 10); L.comp_wkv_bf = make_bf16((size_t)D * DIM, -0.02f, 0.02f, seed + 11);
  L.sink = make_f32(H, -0.5f, 1.0f, seed + 12); L.comp_wkv_f = make_f32((size_t)D * DIM, -0.02f, 0.02f, seed + 13);
  L.comp_wgate = make_f32((size_t)D * DIM, -0.02f, 0.02f, seed + 14);
  return L;
}

// Row (sequence) state
struct RowState { bf16* ring; uint8_t* comp; uint8_t* idxk; float* skv; float* ssc; };

// Work buffers (same names and sizing rules as the runtime Work)
struct Work {
  int Mmax = 8, Tcap = 0;
  bf16 *xn, *qr, *qrn, *q, *kv, *latent, *ik, *iq, *iw, *iscore, *o;
  float *xf, *ckv, *cscore, *cout, *iscore_f, *bmax, *pacc, *pm, *ps;
  uint8_t *qrq, *qrs, *valid, *iqp;
  int32_t *idx, *topk_pos, *cand;
  int *counters, *attn_cnt;
  k::DecRow* tab;
  k::DecSoA* soa;
  // Mapped pinned row tables (same as filled by the runtime's attention_decode_host)
  k::KvRow* kvrows_h; k::KvRow* kvrows_d;
  const uint8_t** kptrs_h; const uint8_t** kptrs_d;
  int32_t *trows_h, *trows_d, *dsti_h, *dsti_d, *pos_h, *pos_d, *gpos_h, *gpos_d, *visible_h, *visible_d;
  bf16 **ringp_h, **ringp_d;
  uint8_t **dstp_h, **dstp_d, **dstk_h, **dstk_d;
  float **skv_h, **skv_d, **ssc_h, **ssc_d;
};
void make_work(Work& w, int Tcap) {
  const int M = w.Mmax;
  w.Tcap = Tcap;
  w.xn = dalloc<bf16>((size_t)M * DIM); w.qr = dalloc<bf16>((size_t)M * QL); w.qrn = dalloc<bf16>((size_t)M * QL); w.q = dalloc<bf16>((size_t)M * H * D);
  w.kv = dalloc<bf16>((size_t)M * D); w.latent = dalloc<bf16>((size_t)M * D); w.ik = dalloc<bf16>((size_t)M * DI); w.iq = dalloc<bf16>((size_t)M * HI * DI);
  w.iw = dalloc<bf16>((size_t)M * HI); w.iscore = dalloc<bf16>((size_t)M * (Tcap + 8)); w.o = dalloc<bf16>((size_t)M * H * D);
  w.xf = dalloc<float>((size_t)M * DIM); w.ckv = dalloc<float>((size_t)M * D); w.cscore = dalloc<float>((size_t)M * D); w.cout = dalloc<float>((size_t)M * D);
  w.iscore_f = dalloc<float>((size_t)M * (Tcap + 8)); w.bmax = dalloc<float>((size_t)M * ((Tcap + BS - 1) / BS));
  w.pacc = dalloc<float>((size_t)M * H * 16 * D); w.pm = dalloc<float>((size_t)M * H * 16); w.ps = dalloc<float>((size_t)M * H * 16);
  w.qrq = dalloc<uint8_t>((size_t)M * QL); w.qrs = dalloc<uint8_t>((size_t)M * QL / 32); w.valid = dalloc<uint8_t>(M);
  w.idx = dalloc<int32_t>((size_t)M * (WIN + TOPK)); w.topk_pos = dalloc<int32_t>((size_t)M * std::max(TOPK, CB)); w.cand = dalloc<int32_t>((size_t)M * CB);
  w.counters = dalloc<int>(16); w.attn_cnt = dalloc<int>((size_t)M * H); w.tab = dalloc<k::DecRow>(M); w.soa = dalloc<k::DecSoA>(1);
  w.iqp = dalloc<uint8_t>((size_t)M * HI * kvp::IDX_ROW);
  w.kvrows_h = mapped<k::KvRow>(M, &w.kvrows_d); w.kptrs_h = mapped<const uint8_t*>(M, &w.kptrs_d);
  w.trows_h = mapped<int32_t>(M, &w.trows_d); w.dsti_h = mapped<int32_t>(M, &w.dsti_d); w.pos_h = mapped<int32_t>(M, &w.pos_d);
  w.gpos_h = mapped<int32_t>(M, &w.gpos_d); w.visible_h = mapped<int32_t>(M, &w.visible_d); w.ringp_h = mapped<bf16*>(M, &w.ringp_d);
  w.dstp_h = mapped<uint8_t*>(M, &w.dstp_d); w.dstk_h = mapped<uint8_t*>(M, &w.dstk_d); w.skv_h = mapped<float*>(M, &w.skv_d); w.ssc_h = mapped<float*>(M, &w.ssc_d);
}

struct Case {
  const char* name;
  int layer, ratio;
  bool kv_source, index_source;
  bool idx_tc;  // HIVE_IDX_TC (on in production): candidate-pool layers use packed queries + indexer_scores_cand_tc
};

// Fill the tables like runtime attention_decode_host. Returns Tmax (index layers); the topk row count is shared (non-index layers use the previous index layer's value)
int fill_tables(Work& w, const Case& cs, const std::vector<RowState>& rows, const std::vector<int>& pos, int M, std::vector<int>& shared_topk) {
  for (int m = 0; m < M; ++m) {
    w.kvrows_h[m].ring = rows[m].ring; w.kvrows_h[m].comp = nullptr; w.kvrows_h[m].comp_len = 0; w.kvrows_h[m].topk = 0;
    w.ringp_h[m] = rows[m].ring;
    w.pos_h[m] = pos[m];
  }
  if (!cs.ratio) return 0;
  const int ratio = cs.ratio;
  for (int m = 0; m < M; ++m) {
    const int p = pos[m];
    w.kvrows_h[m].comp = rows[m].comp;
    w.kvrows_h[m].comp_len = (p + 1) / ratio;
    w.kptrs_h[m] = rows[m].idxk;
    w.trows_h[m] = w.kvrows_h[m].comp_len;
    w.visible_h[m] = (p + 1) / ratio;
  }
  if (cs.kv_source) {
    for (int m = 0; m < M; ++m) {
      const int p = pos[m];
      w.gpos_h[m] = p + 1 - ratio; w.dsti_h[m] = p / ratio; w.dstp_h[m] = rows[m].comp;
      if (ratio > 1) { w.skv_h[m] = rows[m].skv; w.ssc_h[m] = rows[m].ssc; }
      if (cs.index_source) w.dstk_h[m] = rows[m].idxk;
    }
  }
  int Tmax = 0;
  if (cs.index_source) {
    for (int m = 0; m < M; ++m) Tmax = std::max(Tmax, w.trows_h[m]);
    for (int m = 0; m < M; ++m) shared_topk[m] = std::min(TOPK, w.trows_h[m]);
  }
  for (int m = 0; m < M; ++m) w.kvrows_h[m].topk = shared_topk[m];
  return Tmax;
}

// ---- Baseline path: the fuse_ front of runtime attention_decode_dev (HIVE_ATTN_TC, IDX_TC, IDX_F32 off), verbatim --------------------------------
void legacy_front(const Case& cs, const LayerW& A, Work& w, Blas& blas, int M, int Tmax, bool& have_cand, const float2* fr_win, const float2* fr_cmp, cudaStream_t st) {
  const bool pack_dec = cs.idx_tc && CAND_SRC >= 0 && CAND_SRC < cs.layer;
  const int dim = DIM, win = WIN, ratio = cs.ratio, rd = RD;
  const float2* freqs = ratio ? fr_cmp : fr_win;
  k::fused_q_proj(w.xn, dim, A.wqa.w, A.wqa.s, QL, A.q_norm, EPS, A.wqb.w, A.wqb.s, H, D, rd, freqs, w.pos_d, M, w.qr, w.qrn, w.q, w.qrq, w.qrs, w.counters + 0, st);
  k::fused_kv_proj(w.xn, dim, A.wkv.w, A.wkv.s, D, A.kv_norm, EPS, rd, freqs, w.pos_d, M, w.kv, w.ringp_d, win, w.idx, win + TOPK, w.counters + 1, st);
  const int idx_stride = win + TOPK;
  const int topk_max_cols = ratio ? TOPK : 0;
  if (ratio) {
    if (cs.kv_source) {
      if (ratio > 1) {
        k::bf16_to_f32(w.xn, M * dim, w.xf, st);
        blas.gemm_f32(w.xf, A.comp_wkv_f, w.ckv, M, D, dim);
        blas.gemm_f32(w.xf, A.comp_wgate, w.cscore, M, D, dim);
        k::compressor_step_rows(w.ckv, w.cscore, M, D, ratio, w.pos_d, w.skv_d, w.ssc_d, w.cout, w.valid, st);
        k::f32_to_bf16(w.cout, M * D, w.latent, st);
      } else {
        blas.gemm_bf16(w.xn, A.comp_wkv_bf, w.latent, M, D, dim);
        CUDA_CHECK(cudaMemsetAsync(w.valid, 1, (size_t)M, st));
      }
      k::rmsnorm(w.latent, A.comp_norm, EPS, M, D, w.latent, st);
      if (cs.index_source) {
        blas.gemm_bf16(w.latent, A.idx_wk, w.ik, M, DI, D);
        k::rmsnorm(w.ik, A.idx_k_norm, EPS, M, DI, w.ik, st);
        k::rope_last(w.ik, M, 1, DI, rd, freqs, w.gpos_d, false, st);
        k::fp4_pack_rows(w.ik, M, DI, 32, false, w.dstk_d, w.dsti_d, w.valid, kvp::IDX_ROW, st);
      }
      k::rope_last(w.latent, M, 1, D, rd, freqs, w.gpos_d, false, st);
      k::fp4_pack_rows(w.latent, M, D, 16, true, w.dstp_d, w.dsti_d, w.valid, kvp::COMP_ROW, st);
    }
    if (cs.index_source && Tmax > 0) {
      k::gemm_bs(w.qrq, w.qrs, A.idx_wqb.w, A.idx_wqb.s, false, M, HI * DI, QL, w.iq, nullptr, st);
      k::rope_last(w.iq, M, HI, DI, rd, fr_cmp, w.pos_d, false, st);
      if (pack_dec) k::fp4_pack(w.iq, M * HI, DI, 32, false, w.iqp, kvp::IDX_ROW, st);
      k::fp4_quant_roundtrip(w.iq, M, HI * DI, 32, false, st);
      blas.gemm_bf16(w.xn, A.idx_wproj, w.iw, M, HI, dim);
      k::scale_bf16(w.iw, M * HI, (1.0f / sqrtf((float)DI)) * (1.0f / sqrtf((float)HI)), st);
      const bool cand_src = (cs.layer == CAND_SRC);
      const bool uses_cand = (CAND_SRC >= 0 && CAND_SRC < cs.layer);
      const int nblocks = (Tmax + BS - 1) / BS;
      const int topk = std::min(TOPK, Tmax);
      if (uses_cand) {
        const int kbc = std::min(CB, nblocks);
        HIVE_CHECK(have_cand && kbc > 0, "candidates missing");
        const int ncand = kbc * BS;
        if (pack_dec) k::indexer_scores_cand_tc(w.iqp, nullptr, w.kptrs_d, w.trows_d, w.iw, M, HI, w.cand, CB, kbc, BS, w.visible_d, w.iscore, st);
        else k::indexer_scores_cand(w.iq, nullptr, w.kptrs_d, w.trows_d, w.iw, M, HI, DI, w.cand, CB, kbc, BS, w.visible_d, w.iscore, st);
        k::bf16_rows_to_f32(w.iscore, M * ncand, w.iscore_f, st);
        k::topk_select_rows(w.iscore_f, M, ncand, topk, ncand, w.topk_pos, topk, st);
        k::cand_to_pos(w.topk_pos, M, topk, w.cand, CB, BS, st);
      } else {
        k::indexer_scores_rows(w.iq, w.kptrs_d, w.trows_d, Tmax, w.iw, M, HI, DI, w.visible_d, w.iscore, st);
        if (cand_src) {
          k::block_max(w.iscore, M, Tmax, BS, w.visible_d, w.bmax, st);
          const int kb = std::min(CB, nblocks);
          k::topk_select_rows(w.bmax, M, nblocks, kb, nblocks, w.topk_pos, kb, st);
          CUDA_CHECK(cudaMemcpy2DAsync(w.cand, (size_t)CB * 4, w.topk_pos, (size_t)kb * 4, (size_t)kb * 4, M, cudaMemcpyDeviceToDevice, st));
          have_cand = true;
        }
        k::bf16_rows_to_f32(w.iscore, M * Tmax, w.iscore_f, st);
        k::topk_select_rows_t(w.iscore_f, M, w.trows_d, Tmax, topk, Tmax, w.topk_pos, topk, st);
      }
      k::offset_idxs(w.topk_pos, M, topk, w.visible_d, win + M, w.idx + win, idx_stride, st);
    }
  }
  k::sparse_attn_decode_rows(w.q, M, H, D, w.kvrows_d, w.kv, M, win, w.idx, idx_stride, win + topk_max_cols, A.sink, 1.0f / sqrtf((float)D), w.o,
                             (win + topk_max_cols) > 256 ? SPLITS : 1, w.pacc, w.pm, w.ps, st, freqs, w.pos_d, rd);
}

void fused_front(const Case& cs, const LayerW& A, Work& w, Blas& blas, int M, int Tmax, bool& have_cand, const float2* fr_win, const float2* fr_cmp, cudaStream_t st) {
  k::DecAttnArgs a;
  a.M = M; a.dim = DIM; a.H = H; a.D = D; a.rd = RD; a.win = WIN; a.q_lora = QL; a.Hi = HI; a.Di = DI; a.index_topk = TOPK; a.ratio = cs.ratio; a.layer = cs.layer;
  a.cand_source_layer = CAND_SRC; a.cand_block = BS; a.cand_topk_blocks = CB; a.kv_source = cs.kv_source; a.index_source = cs.index_source;
  a.have_candidates = have_cand; a.Tmax = Tmax; a.eps = EPS; a.splits = (WIN + (cs.ratio ? TOPK : 0)) > 256 ? SPLITS : 1;
  a.wqa = A.wqa.w; a.sqa = A.wqa.s; a.wqb = A.wqb.w; a.sqb = A.wqb.s; a.wkv = A.wkv.w; a.skv = A.wkv.s; a.q_norm = A.q_norm; a.kv_norm = A.kv_norm; a.sink = A.sink;
  a.comp_wkv = cs.ratio > 1 ? (const void*)A.comp_wkv_f : (const void*)A.comp_wkv_bf; a.comp_wgate = A.comp_wgate; a.comp_norm = A.comp_norm;
  a.idx_wqb = A.idx_wqb.w; a.idx_sqb = A.idx_wqb.s; a.idx_wproj = A.idx_wproj; a.idx_wk = A.idx_wk; a.idx_k_norm = A.idx_k_norm;
  a.freqs = cs.ratio ? fr_cmp : fr_win; a.freqs_idx = fr_cmp;
  a.xn = w.xn; a.xf = w.xf; a.qr = w.qr; a.qrn = w.qrn; a.q = w.q; a.kv = w.kv; a.qrq = w.qrq; a.qrs = w.qrs; a.ckv = w.ckv; a.cscore = w.cscore; a.cout = w.cout;
  a.latent = w.latent; a.valid = w.valid; a.ik = w.ik; a.iq = w.iq; a.iw = w.iw; a.iscore = w.iscore; a.bmax = w.bmax; a.cand = w.cand; a.idx = w.idx;
  a.idx_stride = WIN + TOPK; a.o = w.o; a.pacc = w.pacc; a.pm = w.pm; a.ps = w.ps; a.counters = w.counters + 12; a.attn_cnt = w.attn_cnt; a.tab = w.tab; a.soa = w.soa;
  a.idx_mode = cs.idx_tc ? 1 : 0; a.iqp = w.iqp;
  a.src = k::DecRowSrc{w.kvrows_d, w.kptrs_d, w.dstp_d, w.dstk_d, w.skv_d, w.ssc_d, w.pos_d, w.gpos_d, w.visible_d, w.trows_d, w.dsti_d};
  a.ring_ptrs = w.ringp_d;
  if (k::dec_attention_front(a, &blas, st)) have_cand = true;
}

// What is compared after one run
struct Outs { std::vector<std::vector<uint8_t>> v; };
const char* kOutNames[] = {"qr", "qrn", "q", "qrq", "qrs", "kv", "iq", "idx", "o", "ring", "comp_row", "idxk_row", "state", "valid", "cand", "iqp", "iw"};
Outs grab(const Case& cs, Work& w, const std::vector<RowState>& rows, const std::vector<int>& pos, int M) {
  Outs o;
  o.v.push_back(snap(w.qr, (size_t)M * QL)); o.v.push_back(snap(w.qrn, (size_t)M * QL)); o.v.push_back(snap(w.q, (size_t)M * H * D));
  o.v.push_back(snap(w.qrq, (size_t)M * QL)); o.v.push_back(snap(w.qrs, (size_t)M * QL / 32)); o.v.push_back(snap(w.kv, (size_t)M * D));
  o.v.push_back(cs.index_source && cs.ratio ? snap(w.iq, (size_t)M * HI * DI) : std::vector<uint8_t>());
  o.v.push_back(snap(w.idx, (size_t)M * (WIN + TOPK))); o.v.push_back(snap(w.o, (size_t)M * H * D));
  std::vector<uint8_t> ring, comp, idxk, state, valid;
  for (int m = 0; m < M; ++m) {
    auto r = snap(rows[m].ring + (size_t)(pos[m] % WIN) * D, D); ring.insert(ring.end(), r.begin(), r.end());
    if (cs.ratio && cs.kv_source) {
      const int di = pos[m] / cs.ratio;
      auto c = snap(rows[m].comp + (size_t)di * kvp::COMP_ROW, kvp::COMP_ROW); comp.insert(comp.end(), c.begin(), c.end());
      if (cs.index_source) { auto x = snap(rows[m].idxk + (size_t)di * kvp::IDX_ROW, kvp::IDX_ROW); idxk.insert(idxk.end(), x.begin(), x.end()); }
      if (cs.ratio > 1) {
        auto a = snap(rows[m].skv, (size_t)cs.ratio * D), b = snap(rows[m].ssc, (size_t)cs.ratio * D);
        state.insert(state.end(), a.begin(), a.end()); state.insert(state.end(), b.begin(), b.end());
      }
    }
  }
  if (cs.ratio && cs.kv_source) valid = snap(w.valid, M);
  o.v.push_back(ring); o.v.push_back(comp); o.v.push_back(idxk); o.v.push_back(state); o.v.push_back(valid);
  o.v.push_back(cs.layer == CAND_SRC ? snap(w.cand, (size_t)M * CB) : std::vector<uint8_t>());
  o.v.push_back(cs.idx_tc ? snap(w.iqp, (size_t)M * HI * kvp::IDX_ROW) : std::vector<uint8_t>());
  o.v.push_back(cs.index_source && cs.ratio ? snap(w.iw, (size_t)M * HI) : std::vector<uint8_t>());  // weights after scaling
  return o;
}
// Restore the mutable state (per-row ring slots, compressed/index-key rows, compressor state, idx, cand) to the reference values
struct Saved { std::vector<std::vector<uint8_t>> ring, comp, idxk, skv, ssc; std::vector<uint8_t> idx, cand; };
Saved save_state(const Case& cs, Work& w, const std::vector<RowState>& rows, const std::vector<int>& pos, int M) {
  Saved s;
  for (int m = 0; m < M; ++m) {
    s.ring.push_back(snap(rows[m].ring + (size_t)(pos[m] % WIN) * D, D));
    const int di = cs.ratio ? pos[m] / cs.ratio : 0;
    s.comp.push_back(cs.ratio ? snap(rows[m].comp + (size_t)di * kvp::COMP_ROW, kvp::COMP_ROW) : std::vector<uint8_t>());
    s.idxk.push_back(cs.ratio ? snap(rows[m].idxk + (size_t)di * kvp::IDX_ROW, kvp::IDX_ROW) : std::vector<uint8_t>());
    s.skv.push_back(snap(rows[m].skv, (size_t)2 * D)); s.ssc.push_back(snap(rows[m].ssc, (size_t)2 * D));
  }
  s.idx = snap(w.idx, (size_t)M * (WIN + TOPK)); s.cand = snap(w.cand, (size_t)M * CB);
  return s;
}
void restore_state(const Case& cs, Work& w, const std::vector<RowState>& rows, const std::vector<int>& pos, int M, const Saved& s) {
  for (int m = 0; m < M; ++m) {
    put(rows[m].ring + (size_t)(pos[m] % WIN) * D, s.ring[m]);
    const int di = cs.ratio ? pos[m] / cs.ratio : 0;
    if (cs.ratio) { put(rows[m].comp + (size_t)di * kvp::COMP_ROW, s.comp[m]); put(rows[m].idxk + (size_t)di * kvp::IDX_ROW, s.idxk[m]); }
    put(rows[m].skv, s.skv[m]); put(rows[m].ssc, s.ssc[m]);
  }
  put(w.idx, s.idx); put(w.cand, s.cand);
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <class F>
float time_eager(F&& f, cudaStream_t st, int iters) {
  for (int i = 0; i < 3; ++i) f();
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0, st));
  for (int i = 0; i < iters; ++i) f();
  CUDA_CHECK(cudaEventRecord(e1, st));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms * 1000.f / iters;
}
template <class F>
float time_graph(F&& f, cudaStream_t st, int iters) {
  f();  // first launch is eager (attribute setup, cuBLAS init — the runtime's run_graph also runs the first use eagerly)
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaGraph_t g; cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  f();
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaGraphLaunch(ge, st));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0, st));
  for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(ge, st));
  CUDA_CHECK(cudaEventRecord(e1, st));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  return ms * 1000.f / iters;
}

std::vector<int> parse_list(const char* s, std::vector<int> def) {
  if (!s || !*s) return def;
  std::vector<int> v;
  std::string t(s);
  size_t p = 0;
  while (p < t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size(); v.push_back(atoi(t.substr(p, q - p).c_str())); p = q + 1; }
  return v;
}
int next_pow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }

}  // namespace

int main(int argc, char** argv) {
  const std::vector<int> ctxs = parse_list(argc > 1 ? argv[1] : nullptr, {4096, 32768, 131072, 262144});
  const std::vector<int> Ms = parse_list(argc > 2 ? argv[2] : nullptr, {1, 4, 8});
  const int max_ctx = *std::max_element(ctxs.begin(), ctxs.end());
  const int Mmax = 8;
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));  // blocking stream: ordered with default-stream cudaMemcpy (put/snap)
  Blas blas(st);
  // RoPE tables (positions max_ctx + margin)
  const int npos = max_ctx + 16;
  float2 *fr_win = dalloc<float2>((size_t)npos * (RD / 2)), *fr_cmp = dalloc<float2>((size_t)npos * (RD / 2));
  fill_rope<<<4096, 256>>>(fr_win, npos, RD / 2, 10000.f);
  fill_rope<<<4096, 256>>>(fr_cmp, npos, RD / 2, 160000.f);
  // Row state (separate per row — the M rows are different sequences)
  std::vector<RowState> rows(Mmax);
  const size_t crow = (size_t)max_ctx + 16;
  for (int m = 0; m < Mmax; ++m) {
    rows[m].ring = make_bf16((size_t)WIN * D, -2.f, 2.f, 1000 + m);
    rows[m].comp = dalloc<uint8_t>(crow * kvp::COMP_ROW);
    rows[m].idxk = dalloc<uint8_t>(crow * kvp::IDX_ROW);
    int g; grid_fill(crow * kvp::COMP_ROW, g);
    fill_comp<<<g, 256>>>(rows[m].comp, crow, 2000 + m);
    fill_idxk<<<g, 256>>>(rows[m].idxk, crow, 3000 + m, 0);
    rows[m].skv = make_f32((size_t)2 * D, -1.f, 1.f, 4000 + m);
    rows[m].ssc = make_f32((size_t)2 * D, -2.f, 2.f, 5000 + m);
  }
  Work w;
  make_work(w, max_ctx + 8);
  CUDA_CHECK(cudaMemset(w.cand, 0xFF, (size_t)Mmax * CB * 4));
  const Case cases[] = {{"W", 0, 0, false, false, false}, {"C", 22, 1, false, false, false}, {"K2", 2, 2, true, true, false}, {"K20", 20, 1, true, true, false},
                        {"U", 24, 1, false, true, false}, {"Utc", 28, 1, false, true, true}};
  std::vector<LayerW> LW;
  for (int i = 0; i < 6; ++i) LW.push_back(make_layer(100 * (i + 1)));
  CUDA_CHECK(cudaDeviceSynchronize());
  bool all_ok = true;
  int n_cases = 0;
  printf("%-4s %7s %2s %-5s | %-40s | %9s %9s %6s | %9s %9s %6s\n", "L", "ctx", "M", "ties", "bit match", "old µs", "new µs", "x", "oldG µs", "newG µs", "x");
  for (const int ctx : ctxs) {
    for (const int ties : {0, 1}) {
      if (ties && ctx != 32768) continue;
      for (int m = 0; m < Mmax; ++m) { int g; grid_fill(crow * kvp::IDX_ROW, g); fill_idxk<<<g, 256>>>(rows[m].idxk, crow, 3000 + m, ties); }
      for (const int M : Ms) {
        if (M < 1 || M > Mmax) continue;
        std::vector<int> pos(M);
        for (int m = 0; m < M; ++m) pos[m] = ctx - 1 - 3 * m;  // ratio 2: mixes even and odd rows (completed/incomplete groups)
        std::vector<int> shared_topk(M, 0);
        bool have_cand = false;
        for (const Case& cs : cases) {
          const LayerW& A = LW[&cs - cases];
          { int g; grid_fill((size_t)M * DIM, g); fill_bf16<<<g, 256>>>(w.xn, (size_t)M * DIM, -1.7f, 1.7f, 77 + cs.layer * 13 + ctx + M); }
          // idx reference values: window columns are written by the kv kernel; top-k columns look like values left by the previous index layer (read by non-index layers) — valid compressed positions ascending
          {
            std::vector<int32_t> idx0((size_t)M * (WIN + TOPK), -1);
            for (int m = 0; m < M; ++m)
              for (int j = 0; j < TOPK; ++j) {
                const int T = cs.ratio ? (pos[m] + 1) / cs.ratio : 0;
                const int stride = std::max(1, T / TOPK);
                const int p = j * stride;
                idx0[(size_t)m * (WIN + TOPK) + WIN + j] = (T > 0 && p < T) ? p + WIN + M : -1;
              }
            CUDA_CHECK(cudaMemcpy(w.idx, idx0.data(), idx0.size() * 4, cudaMemcpyHostToDevice));
          }
          const int Tact = fill_tables(w, cs, rows, pos, M, shared_topk);
          if (!cs.index_source && cs.ratio) for (int m = 0; m < M; ++m) { shared_topk[m] = std::min(TOPK, w.trows_h[m]); w.kvrows_h[m].topk = shared_topk[m]; }
          const int Tb = cs.index_source && cs.ratio ? std::min(w.Tcap - 8, std::max(1024, next_pow2(Tact))) : 0;  // graph bucket (runtime decode_layer)
          CUDA_CHECK(cudaDeviceSynchronize());
          const Saved base = save_state(cs, w, rows, pos, M);
          bool hc_old = have_cand, hc_new = have_cand;
          legacy_front(cs, A, w, blas, M, Tb, hc_old, fr_win, fr_cmp, st);
          CUDA_CHECK(cudaStreamSynchronize(st));
          const Outs o_old = grab(cs, w, rows, pos, M);
          const std::vector<uint8_t> cand_old = snap(w.cand, (size_t)M * CB);
          restore_state(cs, w, rows, pos, M, base);
          fused_front(cs, A, w, blas, M, Tb, hc_new, fr_win, fr_cmp, st);
          CUDA_CHECK(cudaStreamSynchronize(st));
          CUDA_CHECK(cudaGetLastError());
          const Outs o_new = grab(cs, w, rows, pos, M);
          std::string bad;
          for (size_t i = 0; i < o_old.v.size(); ++i)
            if (o_old.v[i] != o_new.v[i]) {
              size_t first = 0, n = 0;
              for (size_t b = 0; b < std::min(o_old.v[i].size(), o_new.v[i].size()); ++b) if (o_old.v[i][b] != o_new.v[i][b]) { if (!n) first = b; ++n; }
              char buf[96]; snprintf(buf, sizeof buf, "%s≠(%zu B, first %zu) ", kOutNames[i], n, first); bad += buf;
            }
          if (hc_old != hc_new) bad += "have_cand≠ ";
          const bool ok = bad.empty();
          // Timing (eager, graph) — repetitions rewrite the same values, so the state does not change (ring slots, compressor slots, cache rows all identical)
          restore_state(cs, w, rows, pos, M, base);
          bool hco = have_cand, hcn = have_cand;
          const float t_old = time_eager([&] { legacy_front(cs, A, w, blas, M, Tb, hco, fr_win, fr_cmp, st); }, st, 20);
          restore_state(cs, w, rows, pos, M, base);
          const float t_new = time_eager([&] { fused_front(cs, A, w, blas, M, Tb, hcn, fr_win, fr_cmp, st); }, st, 20);
          restore_state(cs, w, rows, pos, M, base);
          const float g_old = time_graph([&] { legacy_front(cs, A, w, blas, M, Tb, hco, fr_win, fr_cmp, st); }, st, 20);
          restore_state(cs, w, rows, pos, M, base);
          const float g_new = time_graph([&] { fused_front(cs, A, w, blas, M, Tb, hcn, fr_win, fr_cmp, st); }, st, 20);
          // Candidates for the next layer (U): the baseline result (equal to the new result when the comparison passed)
          restore_state(cs, w, rows, pos, M, base);
          put(w.cand, cand_old);
          have_cand = hc_old;
          CUDA_CHECK(cudaDeviceSynchronize());
          printf("%-4s %7d %2d %-5s | %-40s | %9.1f %9.1f %6.2f | %9.1f %9.1f %6.2f  %s\n", cs.name, ctx, M, ties ? "ties" : "-", ok ? "all bit-identical" : bad.c_str(), t_old,
                 t_new, t_old / t_new, g_old, g_new, g_old / g_new, ok ? "PASS" : "FAIL");
          fflush(stdout);
          all_ok = all_ok && ok;
          ++n_cases;
        }
      }
    }
  }
  printf("%d cases — %s\n", n_cases, all_ok ? "ALL PASS" : "SOME FAIL");
  return all_ok ? 0 : 1;
}
