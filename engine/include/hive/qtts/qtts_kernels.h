// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// qtts kernels: decode GEMV with fused prologues / epilogues, attention, RoPE, sampling, and the codec decoder's element-wise and conv
//   helpers. Activations are fp32 unless noted; weights bf16 (GEMM / GEMV operands) or fp32 (norms, biases, scales, small tables).
//   Every launcher takes the stream; none allocates, so all of them can be captured in CUDA graphs.
#pragma once
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "hive/common.h"

namespace hive::qtts {

// Device-side generation state, read and advanced by the captured kernels (the graphs are position independent).
struct Ctl {
  int pos;          // talker: KV position of the next row
  int frame;        // frames completed (rows written to the code buffer)
  int done;         // 1 once EOS was sampled or a limit was reached
  int max_frames;   // frame limit for this generation
  int max_pos;      // talker KV capacity
  int forced_n;     // teacher forcing: frames < forced_n take their codes from the forced buffer (then EOS)
  int dump;         // 1 = copy every sampled logit row to the dump buffer
  int dump_frames;  // dump buffer capacity in frames
  unsigned long long seed;
  unsigned int rng; // sampling calls so far (counter of the counter-based RNG)
  int codec_frame;  // first frame of the next codec chunk
  int eos_frame;    // frame count when EOS was sampled (-1 = not yet)
  // per-generation sampling settings, read at run time by sample() with SampleSpec::from_ctl (a captured graph keeps its kernel
  //   arguments, so these cannot be launch parameters): index 0 = first codebook, 1 = the other codebooks
  float temperature[2];
  int top_k[2];
  float rep_penalty;
  int min_frames;
};

// ---- GEMV: y[m, n] = epilogue(sum_k W[n, k] * x'[m, k]) for M <= 4 rows -----------------------------------------------------------------
enum GemvMode { kGemvStore = 0, kGemvAccum = 1, kGemvSiluMul = 2 };
struct GemvParams {
  const bf16* W = nullptr;  // [rows, K] row-major (kGemvSiluMul: 2N rows, gate rows then up rows)
  int N = 0, K = 0, M = 1;
  const float* x = nullptr;  // fp32 input rows (row stride ldx) ...
  int ldx = 0;
  const bf16* table = nullptr;  // ... or one bf16 row of `table` selected on the device by *idx (M = 1)
  const int* idx = nullptr;
  const float* norm_w = nullptr;  // optional RMS norm of each input row (x' = x * rsqrt(mean(x^2) + eps) * norm_w)
  float eps = 1e-6f;
  const float* bias = nullptr;    // kGemvStore: + bias[n]
  const float* scale = nullptr;   // kGemvAccum: y += scale[n] * acc (scale null = 1)
  float* y = nullptr;
  int ldy = 0;
  int mode = kGemvStore;
};
void gemv(const GemvParams& p, cudaStream_t st);

// ---- talker / predictor attention ---------------------------------------------------------------------------------------------------------
// rows m < M of qkv [M, (H + 2 KV) * D] at positions pos0 + m (pos0 = *pos_dev + pos_off when pos_dev is set, else pos_off):
//   q heads -> RMS norm (qn) -> RoPE -> q_out [M, H * D]; k heads -> RMS norm (kn) -> RoPE -> kc; v -> vc. kc / vc are bf16
//   [KV][cap][D] (one layer). cos / sin tables are [positions][D / 2].
void qk_norm_rope_store(const float* qkv, int M, int H, int KV, int D, const float* qn, const float* kn, float eps, const float* cos_t,
                        const float* sin_t, const int* pos_dev, int pos_off, float* q_out, bf16* kc, bf16* vc, int cap, cudaStream_t st);
// Decode attention of one row (keys 0 .. *pos_dev + pos_off, inclusive) split over `splits` key ranges; with splits == 1 the result is
//   written to out [H * D] directly, otherwise partial (m, l, acc) to part and attn_combine merges them.
void attn_decode(const float* q, const bf16* kc, const bf16* vc, int H, int KV, int D, int cap, const int* pos_dev, int pos_off, int splits,
                 float* part, float* out, cudaStream_t st);
void attn_combine(const float* part, int H, int D, int splits, float* out, cudaStream_t st);
// Causal attention of M prompt rows at positions 0 .. M-1 (all keys in the cache) -> out bf16 [M, H * D]
void attn_prefill(const float* q, const bf16* kc, const bf16* vc, int M, int H, int KV, int D, int cap, bf16* out, cudaStream_t st);

// ---- prefill helpers ----------------------------------------------------------------------------------------------------------------------
void rms_norm_bf16(const float* x, const float* w, float eps, int M, int K, bf16* out, cudaStream_t st);  // [M, K] -> bf16
void silu_mul_bf16(const float* gu, int M, int N, bf16* out, cudaStream_t st);  // gu [M, 2N] (gate | up) -> silu(gate) * up
void f32_to_bf16(const float* x, size_t n, bf16* out, cudaStream_t st);
void bias_silu_bf16(const float* x, const float* bias, int M, int N, bf16* out, cudaStream_t st);  // silu(x + b) -> bf16
void bias_add(float* x, const float* bias, int M, int N, cudaStream_t st);                         // x += b (row broadcast)
// Talker prompt rows: E[r] = (tslot[r] >= 0 ? P[tslot[r]] : 0) + (cid[r] >= 0 ? codec_emb[cid[r]] : 0)
void assemble_rows(const float* P, const bf16* codec_emb, const int* tslot, const int* cid, int R, int K, float* E, cudaStream_t st);

// ---- frame bookkeeping ---------------------------------------------------------------------------------------------------------------------
// h = codec_emb[tok[0]] + sum_i pred_emb[i][tok[i + 1]] + pad; codes[frame] = tok; hist[tok[0]] = 1; ctl->frame++.
//   pred_emb holds the 15 tables back to back ([15][vocab][K]).
void frame_input(const bf16* codec_emb, const bf16* pred_emb, int pred_vocab, const float* pad, const int* tok, int K, float* h, int* codes,
                 uint8_t* hist, Ctl* ctl, cudaStream_t st);

// ---- sampling --------------------------------------------------------------------------------------------------------------------------------
struct SampleSpec {
  int V = 0;                         // logits in this head
  float temperature = 0.9f;          // <= 0: greedy (argmax)
  int top_k = 50;
  float rep_penalty = 1.f;           // over ids flagged in hist (first codebook)
  int sup_lo = 0, sup_hi = 0;        // ids in [sup_lo, sup_hi) are suppressed ...
  int keep_id = -1;                  // ... except this one (EOS)
  int eos_id = -1, min_frames = 0;   // EOS is suppressed while ctl->frame < min_frames
  int cb = 0;                        // codebook (0 = talker; its token is the first code of frame ctl->frame)
  int first = 0;                     // talker: 1 = checks EOS / limits and sets ctl->done
  int advance = 0;                   // talker frame step: ctl->pos += 1 (the row just written to the KV cache)
  int from_ctl = 0;                  // 1 = temperature / top_k (/ rep_penalty / min_frames for cb 0) come from ctl, not from here
};
// One block: penalty, suppression, top-k (ties at the k-th value kept), temperature, Gumbel-max with a counter-based RNG
//   (seed, call counter, id). Writes *tok_out; teacher forcing and logit dumps follow ctl.
void sample(const float* logits, const SampleSpec& sp, const uint8_t* hist, Ctl* ctl, int* tok_out, const int* forced, float* dump,
            cudaStream_t st);

// ---- codec decoder ---------------------------------------------------------------------------------------------------------------------------
// RVQ: codes rows [ctl->codec_frame + t] (t < T) -> x [T, 512] = Pf * cb0[c0] + Pr * sum_{q>=1} cbq[cq]. cb [16][2048][256] fp32,
//   Pf / Pr [512][256] fp32.
void codec_rvq(const int* codes, const Ctl* ctl, int T, const float* cb, int cb_size, int cb_dim, const float* Pf, const float* Pr, int out_dim,
               float* x, cudaStream_t st);
// Causal im2col: A[t, (j, c)] = act(xx[t + j * dil, c]) with xx = [state (H rows); x (T rows)], H = (k - 1) * dil; act = optional
//   SnakeBeta (x + sin^2(x * ea[c]) * ib[c]). A is bf16 with row stride lda (>= k * C, padding columns untouched).
void im2col_causal(const float* x, const float* state, int T, int C, int k, int dil, const float* ea, const float* ib, bf16* A, int lda,
                   cudaStream_t st);
// state <- the last H rows of x [T, C] (T >= H)
void save_state(const float* x, int T, int C, int H, float* state, cudaStream_t st);
// y[t, n] = bias[n % nb]  (prefill of a GEMM output that is accumulated with beta = 1)
void fill_bias(float* y, int M, int N, const float* bias, int nb, cudaStream_t st);
// a = bf16(snake(v + b1)); r += b2   (residual unit middle)
void ru_mid(const float* v, const float* b1, const float* ea, const float* ib, const float* b2, int T, int C, bf16* a, float* r,
            cudaStream_t st);
// ConvNeXt front: depthwise causal conv k7 (state 6 rows) + bias -> LayerNorm (eps 1e-6) -> bf16 out; state updated.
void convnext_dw_ln(const float* x, float* state, int T, int C, const float* dw_w, const float* dw_b, const float* ln_w, const float* ln_b,
                    bf16* out, cudaStream_t st);
void bias_gelu_bf16(const float* x, const float* b, int M, int N, bf16* out, cudaStream_t st);  // exact (erf) GELU
// x += gamma * (y + b)
void scale_bias_residual(float* x, const float* y, const float* b, const float* gamma, int M, int N, cudaStream_t st);
// Codec transformer attention: rows t < T at positions p = ctl->codec_frame + t; qkv [T, 3 * H * D] (no q/k norms) -> RoPE; K / V written
//   to a ring [H][ring][D] fp32 (slot = position % ring); each row attends to positions (p - window, p] -> out [T, H * D].
void codec_attn(const float* qkv, const Ctl* ctl, int T, int H, int D, int window, int ring, const float* cos_t, const float* sin_t, float* kr,
                float* vr, float* q_tmp, float* out, cudaStream_t st);
// Last layer: SnakeBeta, causal conv C -> 1 (k7, state 6 rows), bias, clamp(-1, 1), pcm = trunc(y * 32767); state updated.
void codec_out(const float* x, float* state, int T, int C, const float* ea, const float* ib, const float* w /*[7][C]*/, float b,
               int16_t* pcm, cudaStream_t st);
void ctl_advance_codec(Ctl* ctl, int frames, cudaStream_t st);

// ---- GEMM (cuBLAS): C[M, N] fp32 (row stride ldc) = beta * C + A[M, K] (bf16, lda) * B[N, K]^T (bf16, ldb) -------------------------------
void gemm(cublasHandle_t h, const bf16* A, int lda, const bf16* B, int ldb, float* C, int ldc, int M, int N, int K, float beta);

}  // namespace hive::qtts
