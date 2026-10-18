// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// libhive_qasr — Qwen3-ASR (config model_type "qwen3_asr") speech recognition engine, C API.
//
// One engine = one GPU, one worker thread that owns the CUDA stream. transcribe() may be called from several threads at once (the calls
//   block; ctypes drops the GIL around them): up to `max_seqs` requests decode together in one batched step (CUDA graph per batch size),
//   further ones wait in a FIFO. A new request's audio encoder + prompt prefill runs between two decode steps of the others.
// Memory: everything (weights, KV cache for max_seqs x max_ctx positions, work buffers, cuBLAS workspace, CUDA graphs) is allocated in
//   qasr_create, which also runs a warm-up so that lazily loaded kernels are resident; no device allocation happens per request.
//   vram_limit_mb (or HIVE_VRAM_LIMIT_MB when 0) caps the startup plan; qasr_create fails with the reason if the plan does not fit.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qasr_engine qasr_engine;

typedef struct {
  int device;           // CUDA device ordinal (default 0)
  int max_seqs;         // concurrent sequences (KV slots), 1..4 (default 2)
  int max_ctx;          // positions per sequence: prompt (incl. audio tokens) + generated (default 2048)
  int vram_limit_mb;    // 0 = HIVE_VRAM_LIMIT_MB from the environment (unset = no limit)
  int prefill_chunk;    // prompt tokens per prefill pass (default 512; output-neutral up to GEMM rounding)
  int encoder_windows;  // audio encoder attention windows (104 tokens = 8 s each) per pass (default 8)
  int warmup;           // run the start-up warm-up (default 1)
} qasr_options;

typedef struct {
  int n_prompt;        // prompt tokens incl. audio tokens
  int n_audio;         // audio tokens
  int n_generated;     // generated tokens (stop token not counted)
  int finish;          // 0 = stop token, 1 = max_new_tokens / context full, 2 = cancelled
  int stop_token;      // the stop token that ended generation (finish 0), else -1
  double queue_ms, encode_ms, prefill_ms, decode_ms, total_ms;
} qasr_stats;

typedef struct {
  double weights_mib, kv_mib, work_mib, context_mib, plan_mib;  // startup plan (context: measured when NVML is available)
  double used_mib;     // process usage after warm-up (NVML free before the context - free now; 0 if not measurable)
  int context_measured;
  int max_seqs, max_ctx, vocab, hidden, n_mels;
} qasr_info;

void qasr_default_options(qasr_options* o);
// Returns NULL on failure (reason in err).
qasr_engine* qasr_create(const char* ckpt_dir, const qasr_options* opt, char* err, int err_len);
void qasr_destroy(qasr_engine* e);
void qasr_get_info(const qasr_engine* e, qasr_info* out);
// Number of audio tokens the encoder produces for n_frames mel frames.
int qasr_audio_tokens(int n_frames);

// Transcribe one request. mel: fp32 [n_mels, n_frames] (row = mel bin), may be NULL with n_frames = 0 (text-only prompt). The prompt is
//   pre_ids + <audio tokens> + post_ids. temperature <= 0 = greedy; top_k <= 0 = off; top_p >= 1 = off. stop_ids end generation (not
//   emitted). *cancel != 0 (checked before every step) stops the request. Returns the number of tokens written to out_ids (<= out_cap),
//   or -1 with the reason in err.
int qasr_transcribe(qasr_engine* e, const float* mel, int n_mels, int n_frames, const int32_t* pre_ids, int n_pre, const int32_t* post_ids,
                    int n_post, float temperature, float top_p, int top_k, uint64_t seed, int max_new_tokens, const int32_t* stop_ids,
                    int n_stop, const volatile int32_t* cancel, int32_t* out_ids, int out_cap, qasr_stats* stats, char* err, int err_len);

// Validation hooks (same worker path): audio encoder output fp32 [n_audio, hidden] / logits fp32 [vocab] after the full prompt.
int qasr_debug_encode(qasr_engine* e, const float* mel, int n_mels, int n_frames, float* out, int out_rows, char* err, int err_len);
int qasr_debug_prompt_logits(qasr_engine* e, const float* mel, int n_mels, int n_frames, const int32_t* pre_ids, int n_pre,
                             const int32_t* post_ids, int n_post, float* logits, char* err, int err_len);

#ifdef __cplusplus
}
#endif
