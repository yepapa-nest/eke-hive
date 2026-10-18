// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// libhive_qtts C API (hive/qtts/qtts_capi.h) over hive::qtts::Engine: exceptions become negative return codes and a stored message.
#include "hive/qtts/qtts_capi.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include "hive/qtts/qtts_engine.h"
#include "nlohmann/json.hpp"

struct qtts_engine {
  std::unique_ptr<hive::qtts::Engine> eng;
  std::string err;
};

namespace {

template <class F>
int32_t guard(qtts_engine* e, F&& f) {
  if (!e || !e->eng) return -1;
  try {
    return f();
  } catch (const std::exception& x) {
    e->err = x.what();
  } catch (...) {
    e->err = "unknown error";
  }
  return -1;
}

}  // namespace

extern "C" {

qtts_engine* qtts_create(const char* model_dir, int32_t vram_limit_mb, int32_t max_seq, int32_t dump_frames, int32_t flags, char* err,
                         int32_t err_len) {
  try {
    hive::qtts::EngineOptions o;
    if (max_seq > 0) o.max_seq = max_seq;
    o.vram_limit_mb = vram_limit_mb > 0 ? (size_t)vram_limit_mb : 0;
    o.dump_frames = std::max(0, dump_frames);
    o.graphs = (flags & QTTS_NO_GRAPHS) == 0;
    auto* e = new qtts_engine;
    try {
      e->eng = std::make_unique<hive::qtts::Engine>(model_dir ? model_dir : "", o);
    } catch (...) {
      delete e;
      throw;
    }
    return e;
  } catch (const std::exception& x) {
    if (err && err_len > 0) {
      strncpy(err, x.what(), (size_t)err_len - 1);
      err[err_len - 1] = 0;
    }
  }
  return nullptr;
}

void qtts_destroy(qtts_engine* e) { delete e; }

const char* qtts_error(qtts_engine* e) { return e ? e->err.c_str() : "null engine"; }

int32_t qtts_info(qtts_engine* e, char* buf, int32_t len) {
  return guard(e, [&]() -> int32_t {
    const auto& c = e->eng->config();
    const auto& i = e->eng->info();
    nlohmann::json j;
    j["speakers"] = c.speaker_order;
    j["speaker_ids"] = c.speakers;
    j["languages"] = c.languages;
    j["dialect"] = c.dialect;
    j["sample_rate"] = 24000;
    j["frame_samples"] = c.frame_samples();
    j["chunk_frames"] = 4;
    j["max_seq"] = i.max_seq;
    j["max_prompt"] = i.max_prompt;
    j["weight_mb"] = i.weight_bytes / 1048576.0;
    j["buffer_mb"] = i.buffer_bytes / 1048576.0;
    j["context_mb"] = i.context_bytes / 1048576.0;
    j["context_measured"] = i.context_measured;
    j["process_mb"] = i.process_bytes / 1048576.0;
    j["limit_mb"] = i.limit_bytes / 1048576.0;
    const std::string s = j.dump();
    if (!buf || len <= 0) return (int32_t)s.size();
    const size_t n = std::min(s.size(), (size_t)len - 1);
    memcpy(buf, s.data(), n);
    buf[n] = 0;
    return (int32_t)n;
  });
}

int32_t qtts_start(qtts_engine* e, const int32_t* ids, int32_t n_ids, const int32_t* instruct, int32_t n_instruct, int32_t speaker_id,
                   int32_t language_id, uint64_t seed, const qtts_sampling* sp, const int32_t* forced, int32_t n_forced) {
  return guard(e, [&]() -> int32_t {
    hive::qtts::SamplingParams p;
    if (sp) {
      p.temperature = sp->temperature;
      p.top_k = sp->top_k;
      p.sub_temperature = sp->sub_temperature;
      p.sub_top_k = sp->sub_top_k;
      p.rep_penalty = sp->rep_penalty;
      p.min_frames = sp->min_frames;
      p.max_frames = sp->max_frames;
    }
    if (!ids) throw std::invalid_argument("ids is NULL");
    e->eng->start(ids, n_ids, instruct, n_instruct, speaker_id, language_id, seed, p, forced, n_forced);
    return e->eng->prompt_rows();
  });
}

int32_t qtts_next_chunk(qtts_engine* e, int16_t* pcm, int32_t max_samples, const volatile int32_t* cancel) {
  return guard(e, [&]() -> int32_t { return e->eng->next_chunk(pcm, max_samples, cancel); });
}

int32_t qtts_frames(qtts_engine* e) { return guard(e, [&]() -> int32_t { return e->eng->frames(); }); }
int32_t qtts_prompt_rows(qtts_engine* e) { return guard(e, [&]() -> int32_t { return e->eng->prompt_rows(); }); }

int32_t qtts_get_codes(qtts_engine* e, int32_t* out, int32_t max_frames) {
  return guard(e, [&]() -> int32_t {
    const auto v = e->eng->codes();
    const int32_t n = std::min<int32_t>((int32_t)(v.size() / 16), max_frames);
    memcpy(out, v.data(), (size_t)n * 16 * sizeof(int32_t));
    return n;
  });
}

int32_t qtts_get_logits(qtts_engine* e, float* out, int32_t max_frames) {
  return guard(e, [&]() -> int32_t {
    const auto v = e->eng->logits();
    const int32_t n = std::min<int32_t>((int32_t)(v.size() / (16 * 3072)), max_frames);
    memcpy(out, v.data(), (size_t)n * 16 * 3072 * sizeof(float));
    return n;
  });
}

int32_t qtts_decode_codes(qtts_engine* e, const int32_t* codes, int32_t n_frames, int16_t* pcm, int32_t max_samples) {
  return guard(e, [&]() -> int32_t { return e->eng->decode_codes(codes, n_frames, pcm, max_samples); });
}

}  // extern "C"
