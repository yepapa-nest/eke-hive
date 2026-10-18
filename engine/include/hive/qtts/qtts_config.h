// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3-TTS 12Hz (CustomVoice) model dimensions and special ids, read from the checkpoint's config.json and speech_tokenizer/config.json.
//   The math contract is oracle/qwen3_tts_oracle.py.
#pragma once
#include <map>
#include <string>
#include <vector>

namespace hive::qtts {

struct QttsConfig {
  // talker (Qwen3-style decoder over text / codec embeddings)
  int hidden = 2048, layers = 28, heads = 16, kv_heads = 8, head_dim = 128, inter = 6144;
  int vocab = 3072, text_vocab = 151936, text_hidden = 2048, groups = 16;
  float eps = 1e-6f;
  double theta = 1e6;
  // code predictor (the other 15 codebooks)
  int p_hidden = 1024, p_layers = 5, p_heads = 16, p_kv_heads = 8, p_head_dim = 128, p_inter = 3072, p_vocab = 2048;
  float p_eps = 1e-6f;
  double p_theta = 1e6;
  // special ids (text vocabulary)
  int tts_bos = 151672, tts_eos = 151673, tts_pad = 151671;
  // special ids (codec vocabulary)
  int codec_bos = 2149, codec_eos = 2150, codec_pad = 2148, codec_think = 2154, codec_nothink = 2155, think_bos = 2156, think_eos = 2157;
  std::map<std::string, int> speakers;        // name -> codec id
  std::vector<std::string> speaker_order;     // as listed in config.json
  std::map<std::string, int> languages;       // name -> codec id
  std::map<std::string, std::string> dialect; // speaker -> dialect language name (empty = none)
  // codec decoder (speech_tokenizer/config.json "decoder_config")
  int c_quantizers = 16, c_codebook_size = 2048, c_codebook_dim = 512;  // the RVQ entries are c_codebook_dim / 2 wide
  int c_latent = 1024, c_hidden = 512, c_inter = 1024, c_heads = 16, c_head_dim = 64, c_layers = 8, c_window = 72, c_decoder_dim = 1536;
  float c_eps = 1e-5f;
  double c_theta = 1e4;
  std::vector<int> c_upsample_rates{8, 5, 4, 3};
  std::vector<int> c_upsampling_ratios{2, 2};
  int frame_samples() const {
    int s = 1;
    for (int r : c_upsample_rates) s *= r;
    for (int r : c_upsampling_ratios) s *= r;
    return s;
  }
  static QttsConfig load(const std::string& model_dir);
};

}  // namespace hive::qtts
