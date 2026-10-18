// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// qtts engine: Qwen3-TTS 12Hz CustomVoice text-to-speech on one GPU (sm_120a), streaming 24 kHz PCM16 in chunks of 4 codec frames.
//
// Contract (oracle/qwen3_tts_oracle.py defines the math):
//   start()      builds the talker prompt (non-streaming text mode) from the tokenized assistant template, an optional instruct template, a
//                speaker codec id and a language codec id, runs the prompt (cuBLAS GEMMs) and samples the first code of frame 0.
//   next_chunk() runs up to 4 frames (each = one CUDA graph: the 16-row code predictor, the frame embedding, one talker decode step and the
//                sampling of the next frame's first code), then the codec decoder on those frames (a second CUDA graph that carries the
//                causal conv histories and a 72-frame attention window from chunk to chunk, so the stream equals a whole-sequence decode).
//                Returns the number of samples (frames x 1920); 0 = finished; -2 = cancelled (checked before every frame).
//   Everything is allocated in the constructor (weights bf16 / fp32, KV cache for max_seq talker positions, work buffers, graphs); a
//   VRAM limit (EngineOptions::vram_limit_mb, else HIVE_VRAM_LIMIT_MB) lowers max_seq to fit or fails with the shortfall.
//   One generation at a time per engine.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hive/qtts/qtts_config.h"

namespace hive::qtts {

struct EngineOptions {
  int max_seq = 2048;      // talker positions (prompt + frames)
  int max_prompt = 512;    // prompt rows (instruct + template + 11 control rows)
  size_t vram_limit_mb = 0;  // 0 = HIVE_VRAM_LIMIT_MB or no limit
  bool graphs = true;      // false = launch the same kernels eagerly (tests)
  int dump_frames = 0;     // > 0: keep every sampled logit row of the first dump_frames frames (tests)
  int attn_splits = 16;    // talker decode attention key splits
  bool codec_only = false; // tests: load only the codec decoder (decode_codes works, start does not)
};

struct SamplingParams {
  float temperature = 0.9f;      // first codebook; <= 0 greedy
  int top_k = 50;
  float sub_temperature = 0.9f;  // the other 15 codebooks; <= 0 greedy
  int sub_top_k = 50;
  float rep_penalty = 1.05f;     // first codebook
  int min_frames = 2;            // EOS suppressed before this many frames
  int max_frames = 0;            // 0 = until the KV cache is full
};

struct EngineInfo {
  size_t weight_bytes = 0, buffer_bytes = 0, context_bytes = 0, process_bytes = 0, limit_bytes = 0;
  int max_seq = 0, max_prompt = 0;
  bool context_measured = false;
};

class Engine {
 public:
  Engine(const std::string& model_dir, const EngineOptions& opt);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const QttsConfig& config() const;
  const EngineInfo& info() const;

  // ids = tokens of "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n" (at least 8);
  // instruct = tokens of "<|im_start|>user\n{instruct}<|im_end|>\n" or empty; speaker_id / language_id = codec ids or -1.
  // forced (tests): n_forced frames of 16 codes replayed instead of sampling (EOS after the last one).
  void start(const int32_t* ids, int n_ids, const int32_t* instruct, int n_instruct, int speaker_id, int language_id, uint64_t seed,
             const SamplingParams& sp, const int32_t* forced = nullptr, int n_forced = 0);
  int next_chunk(int16_t* pcm, int max_samples, const volatile int32_t* cancel);

  int frames() const;                                      // frames generated in the current / last generation
  int prompt_rows() const;
  std::vector<int32_t> codes() const;                      // [frames][16]
  std::vector<float> logits() const;                       // dump: [min(frames + 1, dump_frames)][16][3072]
  // codec only: decode n_frames codes (a fresh stream) -> n_frames * 1920 samples
  int decode_codes(const int32_t* codes, int n_frames, int16_t* pcm, int max_samples);
  // tests: run the captured graphs (default when built with graphs) or the same kernels eagerly
  void use_graphs(bool on);

  struct Impl;

 private:
  std::unique_ptr<Impl> d_;
};

}  // namespace hive::qtts
