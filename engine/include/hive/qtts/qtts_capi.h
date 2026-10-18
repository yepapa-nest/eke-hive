/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Eke Hive Authors
 * libhive_qtts C API — Qwen3-TTS 12Hz CustomVoice streaming synthesis (hive/qtts/qtts_engine.h), for ctypes / FFI callers.
 * Every call returns >= 0 on success and a negative code on failure (qtts_error() then holds the message); no call throws or aborts on
 * bad input. Calls on one handle must not overlap (one generation at a time); they block the calling thread only (no global lock).
 */
#ifndef HIVE_QTTS_CAPI_H
#define HIVE_QTTS_CAPI_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qtts_engine qtts_engine;

/* flags for qtts_create */
#define QTTS_NO_GRAPHS 1 /* launch kernels eagerly (tests) */

typedef struct qtts_sampling {
  float temperature;     /* first codebook (<= 0: greedy); default 0.9 */
  int32_t top_k;         /* default 50 */
  float sub_temperature; /* other codebooks (<= 0: greedy); default 0.9 */
  int32_t sub_top_k;     /* default 50 */
  float rep_penalty;     /* default 1.05 */
  int32_t min_frames;    /* default 2 */
  int32_t max_frames;    /* 0 = until the KV cache is full */
} qtts_sampling;

/* model_dir: checkpoint directory (config.json, model.safetensors, speech_tokenizer/). vram_limit_mb: 0 = HIVE_VRAM_LIMIT_MB or none.
 * max_seq: talker positions (0 = 2048). dump_frames: > 0 keeps sampled logits for tests. Returns NULL on failure (message in err). */
qtts_engine* qtts_create(const char* model_dir, int32_t vram_limit_mb, int32_t max_seq, int32_t dump_frames, int32_t flags, char* err,
                         int32_t err_len);
void qtts_destroy(qtts_engine* e);
const char* qtts_error(qtts_engine* e);
/* JSON: dimensions, speakers, languages, VRAM figures. Returns the length written (truncated to len - 1). */
int32_t qtts_info(qtts_engine* e, char* buf, int32_t len);

/* ids: tokens of the assistant template; instruct: tokens of the user instruct template (n_instruct = 0: none); speaker_id /
 * language_id: codec ids (-1 = none / auto). sp may be NULL (defaults). forced (tests): n_forced frames x 16 codes. */
int32_t qtts_start(qtts_engine* e, const int32_t* ids, int32_t n_ids, const int32_t* instruct, int32_t n_instruct, int32_t speaker_id,
                   int32_t language_id, uint64_t seed, const qtts_sampling* sp, const int32_t* forced, int32_t n_forced);
/* Up to 4 frames (7680 samples): returns samples written, 0 = finished, -2 = cancelled (*cancel != 0 before a frame), -1 = error. */
int32_t qtts_next_chunk(qtts_engine* e, int16_t* pcm, int32_t max_samples, const volatile int32_t* cancel);

int32_t qtts_frames(qtts_engine* e);
int32_t qtts_prompt_rows(qtts_engine* e);
int32_t qtts_get_codes(qtts_engine* e, int32_t* out, int32_t max_frames);      /* frames copied */
int32_t qtts_get_logits(qtts_engine* e, float* out, int32_t max_frames);       /* rows of [16][3072] copied */
int32_t qtts_decode_codes(qtts_engine* e, const int32_t* codes, int32_t n_frames, int16_t* pcm, int32_t max_samples);

#ifdef __cplusplus
}
#endif
#endif
