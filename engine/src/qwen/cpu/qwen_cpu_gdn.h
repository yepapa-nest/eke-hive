// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Gated DeltaNet recurrence of one value head over n tokens (oracle/qwen4_oracle.py Gdn): per token t, with the state S [k][v] (D × D fp32),
//   S ← S · e_t;  kv = Σ_k S[k] · k_t[k];  δ = (v_t − kv) · β_t;  S[k] ← S[k] + k_t[k] · δ;  o_t = Σ_k S[k] · q_t[k]  (o rounded to bf16).
// Every column v is independent, so a call may take a column range [j0, j1). Two forms, bit-identical per element:
//   gdn_scan_ref  — two passes over S per token (decay + kv, then update + o), S in place (row stride D).
//   gdn_scan_slab — the column range copied into a contiguous slab (≤ 32 columns: 16 KiB with D = 128, inside L1 — the strided rows of S
//                   alias to a few L1 sets), and one pass per token: token t's update + o fused with token t+1's decay + kv on the same rows.
//                   The operations on each element and their order are those of the reference (S' = fma(k, δ, S), o += S'·q; S'' = S'·e,
//                   kv += S''·k), only interleaved across the two tokens.
#pragma once
#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "qwen_cpu_simd.h"

namespace hive::qwen::cpu::gdn {

struct ScanIn {
  const float* k; size_t ks;    // normalised key of token t at k + t·ks [D]
  const float* q; size_t qs;    // scaled normalised query [D]
  const float* v; size_t vs;    // value [D] (this head)
  const float* e; size_t es;    // decay exp(g) of token t at e[t·es]
  const float* b; size_t bs;    // step size β
  float* o; size_t os;          // output [D] (bf16-rounded values)
};

inline void gdn_scan_ref(float* S, int D, int n, const ScanIn& in, int j0, int j1) {
  alignas(32) float kv[128], dl[128], oo[128];
  for (int t = 0; t < n; ++t) {
    const float* kk = in.k + (size_t)t * in.ks;
    const float* qq = in.q + (size_t)t * in.qs;
    const float* vv = in.v + (size_t)t * in.vs;
    const float e = in.e[(size_t)t * in.es], bt = in.b[(size_t)t * in.bs];
    const __m256 ev = _mm256_set1_ps(e);
    for (int j = j0; j < j1; j += 8) _mm256_store_ps(kv + j, _mm256_setzero_ps());
    for (int k = 0; k < D; ++k) {
      float* Sr = S + (size_t)k * D;
      const __m256 kb = _mm256_set1_ps(kk[k]);
      for (int j = j0; j < j1; j += 8) {
        const __m256 sv = _mm256_mul_ps(_mm256_loadu_ps(Sr + j), ev);
        _mm256_storeu_ps(Sr + j, sv);
        _mm256_store_ps(kv + j, _mm256_fmadd_ps(sv, kb, _mm256_load_ps(kv + j)));
      }
    }
    for (int j = j0; j < j1; ++j) { dl[j] = (vv[j] - kv[j]) * bt; oo[j] = 0.f; }
    for (int k = 0; k < D; ++k) {
      float* Sr = S + (size_t)k * D;
      const __m256 kb = _mm256_set1_ps(kk[k]), qb = _mm256_set1_ps(qq[k]);
      for (int j = j0; j < j1; j += 8) {
        const __m256 sv = _mm256_fmadd_ps(kb, _mm256_load_ps(dl + j), _mm256_loadu_ps(Sr + j));
        _mm256_storeu_ps(Sr + j, sv);
        _mm256_store_ps(oo + j, _mm256_fmadd_ps(sv, qb, _mm256_load_ps(oo + j)));
      }
    }
    float* orow = in.o + (size_t)t * in.os;
    for (int j = j0; j < j1; ++j) orow[j] = simd::bf16r_s(oo[j]);
  }
}

// W = j1 − j0 columns, a multiple of 8, at most 32
template <int W>
inline void gdn_scan_slab_w(float* S, int D, int n, const ScanIn& in, int j0) {
  constexpr int V = W / 8;
  alignas(64) float sl[128 * W];
  for (int k = 0; k < D; ++k) std::memcpy(sl + (size_t)k * W, S + (size_t)k * D + j0, W * 4);
  __m256 kv[V];
  // token 0: decay + kv
  {
    const float* kk = in.k;
    const __m256 ev = _mm256_set1_ps(in.e[0]);
    for (int c = 0; c < V; ++c) kv[c] = _mm256_setzero_ps();
    for (int k = 0; k < D; ++k) {
      const __m256 kb = _mm256_set1_ps(kk[k]);
      float* r = sl + (size_t)k * W;
      for (int c = 0; c < V; ++c) {
        const __m256 sv = _mm256_mul_ps(_mm256_load_ps(r + 8 * c), ev);
        _mm256_store_ps(r + 8 * c, sv);
        kv[c] = _mm256_fmadd_ps(sv, kb, kv[c]);
      }
    }
  }
  alignas(32) float kvs[W], dl[W];
  for (int t = 0; t < n; ++t) {
    const float* kk = in.k + (size_t)t * in.ks;
    const float* qq = in.q + (size_t)t * in.qs;
    const float* vv = in.v + (size_t)t * in.vs + j0;
    const float bt = in.b[(size_t)t * in.bs];
    for (int c = 0; c < V; ++c) _mm256_store_ps(kvs + 8 * c, kv[c]);
    for (int j = 0; j < W; ++j) dl[j] = (vv[j] - kvs[j]) * bt;
    __m256 dv[V], oo[V], kn[V];
    for (int c = 0; c < V; ++c) { dv[c] = _mm256_load_ps(dl + 8 * c); oo[c] = _mm256_setzero_ps(); kn[c] = _mm256_setzero_ps(); }
    if (t + 1 < n) {
      const float* kk1 = in.k + (size_t)(t + 1) * in.ks;
      const __m256 ev = _mm256_set1_ps(in.e[(size_t)(t + 1) * in.es]);
      for (int k = 0; k < D; ++k) {
        const __m256 kb = _mm256_set1_ps(kk[k]), qb = _mm256_set1_ps(qq[k]), kb1 = _mm256_set1_ps(kk1[k]);
        float* r = sl + (size_t)k * W;
        for (int c = 0; c < V; ++c) {
          __m256 sv = _mm256_fmadd_ps(kb, dv[c], _mm256_load_ps(r + 8 * c));
          oo[c] = _mm256_fmadd_ps(sv, qb, oo[c]);
          sv = _mm256_mul_ps(sv, ev);
          _mm256_store_ps(r + 8 * c, sv);
          kn[c] = _mm256_fmadd_ps(sv, kb1, kn[c]);
        }
      }
    } else {
      for (int k = 0; k < D; ++k) {
        const __m256 kb = _mm256_set1_ps(kk[k]), qb = _mm256_set1_ps(qq[k]);
        float* r = sl + (size_t)k * W;
        for (int c = 0; c < V; ++c) {
          const __m256 sv = _mm256_fmadd_ps(kb, dv[c], _mm256_load_ps(r + 8 * c));
          _mm256_store_ps(r + 8 * c, sv);
          oo[c] = _mm256_fmadd_ps(sv, qb, oo[c]);
        }
      }
    }
    float* orow = in.o + (size_t)t * in.os + j0;
    for (int c = 0; c < V; ++c) _mm256_storeu_ps(orow + 8 * c, simd::bf16r8(oo[c]));
    for (int c = 0; c < V; ++c) kv[c] = kn[c];
  }
  for (int k = 0; k < D; ++k) std::memcpy(S + (size_t)k * D + j0, sl + (size_t)k * W, W * 4);
}

inline void gdn_scan_slab(float* S, int D, int n, const ScanIn& in, int j0, int j1) {
  if (n <= 0) return;
  switch (j1 - j0) {
    case 8: gdn_scan_slab_w<8>(S, D, n, in, j0); return;
    case 16: gdn_scan_slab_w<16>(S, D, n, in, j0); return;
    case 24: gdn_scan_slab_w<24>(S, D, n, in, j0); return;
    case 32: gdn_scan_slab_w<32>(S, D, n, in, j0); return;
    default: gdn_scan_ref(S, D, n, in, j0, j1); return;
  }
}

}  // namespace hive::qwen::cpu::gdn
