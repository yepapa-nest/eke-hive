// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// R3 long-context decode — switches · launches (device code in decode_longctx.cuh, contract in the hive/decode_longctx.h header comment).
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "decode_longctx.cuh"
#include "hive/decode_longctx.h"

namespace hive::k {

namespace {
inline bool lctx_env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }
inline void lctx_launch_check(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) return;
  throw std::runtime_error(std::string("decode longctx launch failed: ").append(what).append(": ").append(cudaGetErrorString(e)));
}

template <class KeyT, int NBITS>
bool topk2_launch(const KeyT* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                  const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, void* scratch, size_t scratch_bytes, int* cnt,
                  cudaStream_t st) {
  // only shapes the default topk_launch accepts (k 1..2048 · M ≥ 1 · row table) — anything else returns false (the caller uses the default launch → same HIVE_CHECK as before)
  if (k < 1 || k > 2048 || M < 1 || T_in < 0 || M > 65535) return false;
  if ((use_trows || map_offset) && !tab) return false;
  const Topk2Plan p = topk2_plan(T_in, k, M, sizeof(KeyT));
  KeyT* ukeys = nullptr;
  int32_t* uidx = nullptr;
  if (p.ncmax > 1) {
    if (!scratch || !cnt || p.bytes > scratch_bytes || (reinterpret_cast<uintptr_t>(scratch) & 15)) return false;
    ukeys = static_cast<KeyT*>(scratch);
    uidx = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(scratch) + p.idx_off);
  }
  lctx::topk2_kernel<KeyT, NBITS><<<dim3(p.ncmax, M), lctx::TPK2, 0, st>>>(keys, T_in, use_trows, tab, k, row_stride, out, out_stride, cand, cand_stride, bs,
                                                                          offset, map_offset, p.CH, ukeys, uidx, p.ustride, cnt);
  lctx_launch_check("topk2");
  return true;
}

template <bool CAND>
void scores2_launch(int kpt, const bf16* q, const bf16* w, const DecRow* tab, int M, int ncols, const int32_t* cand, int cand_stride, int bs, bf16* score,
                    cudaStream_t st) {
  const int per = lctx::IDX2_T * kpt;
  const dim3 grid((ncols + per - 1) / per, M);
  if (kpt == 4) lctx::idx_scores2_kernel<CAND, 4><<<grid, lctx::IDX2_T, 0, st>>>(q, w, tab, ncols, cand, cand_stride, bs, score);
  else if (kpt == 1) lctx::idx_scores2_kernel<CAND, 1><<<grid, lctx::IDX2_T, 0, st>>>(q, w, tab, ncols, cand, cand_stride, bs, score);
  else lctx::idx_scores2_kernel<CAND, 2><<<grid, lctx::IDX2_T, 0, st>>>(q, w, tab, ncols, cand, cand_stride, bs, score);
  lctx_launch_check("idx_scores2");
}
}  // namespace

bool decode_topk2_on() { static const bool on = lctx_env_on("HIVE_DECODE_TOPK2"); return on; }
bool decode_idxscore2_on() { static const bool on = lctx_env_on("HIVE_DECODE_IDXSCORE2"); return on; }
int decode_idxscore2_kpt() {
  static const int kpt = [] {
    const char* v = getenv("HIVE_DECODE_IDXSCORE2_KPT");
    const int x = v && *v ? atoi(v) : 2;
    return (x == 1 || x == 2 || x == 4) ? x : 2;  // tuning value (performance only) — other values are absorbed into the default 2
  }();
  return kpt;
}

bool dec_topk2_bf16(const bf16* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                    const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, void* scratch, size_t scratch_bytes, int* cnt,
                    cudaStream_t st) {
  return topk2_launch<bf16, 16>(keys, M, T_in, use_trows, tab, k, row_stride, out, out_stride, cand, cand_stride, bs, offset, map_offset, scratch, scratch_bytes,
                                cnt, st);
}
bool dec_topk2_f32(const float* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                   const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, void* scratch, size_t scratch_bytes, int* cnt,
                   cudaStream_t st) {
  return topk2_launch<float, 32>(keys, M, T_in, use_trows, tab, k, row_stride, out, out_stride, cand, cand_stride, bs, offset, map_offset, scratch,
                                 scratch_bytes, cnt, st);
}

bool dec_idx_scores2(const bf16* q, const bf16* w, const DecRow* tab, int M, int ncols, bool cand_mode, const int32_t* cand, int cand_stride, int bs,
                     bf16* score, int kpt, cudaStream_t st) {
  if (M < 1 || M > 65535 || ncols < 0 || !q || !w || !tab || !score || (reinterpret_cast<uintptr_t>(q) & 15)) return false;  // q is read as uint4
  if (cand_mode && (!cand || bs < 1)) return false;
  if (ncols == 0) return true;
  if (kpt <= 0) kpt = decode_idxscore2_kpt();
  if (cand_mode) scores2_launch<true>(kpt, q, w, tab, M, ncols, cand, cand_stride, bs, score, st);
  else scores2_launch<false>(kpt, q, w, tab, M, ncols, nullptr, 0, 1, score, st);
  return true;
}

}  // namespace hive::k
