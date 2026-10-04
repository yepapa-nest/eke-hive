// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Observes how the PRODUCTION code consumes HIVE_* options (one process per environment; the
// driver tools/test_options_cpu.py sets the variables and compares the printed JSON with the
// documented contract "unset / empty / 0 = off, value used as given"). No GPU.
//   * ExpertStore ctor (engine/src/expert_store.cpp): HIVE_CACHE_REUSE_STAGE (behaviour: a staged
//     record is promoted by D2D only when on), HIVE_CACHE_EVENTS (trace file opened or not)
//   * Runtime ctor option block + save_image (production text via tools/cpu_fake/harness.py):
//     HIVE_MTP_CACHE / HIVE_PHASE_SCORE / HIVE_PROMOTE_SCORE / HIVE_PROMOTE_BYTES -> opt(),
//     HIVE_CKPT_ASYNC (fence on the image), HIVE_CKPT_DELTA (prefix segment shared with base),
//     HIVE_CKPT_PINNED_POOL_MB (pool caches a returned block)
// Expressions inside functions that cannot run on CPU (hived main statics, runtime prefetch /
// tile-group lines) are evaluated by the driver from their exact source text instead.
#include <cstdio>
#include <memory>
#include <vector>

#include "hive/runtime.h"

using namespace hive;

int main() {
  Model model("fake");
  const Config& c = model.cfg();
  const size_t rec = ExpertLayout::make(c.dim, c.moe_inter).total;
  bool reuse_d2d = false, events_opened = false;
  {
    ExpertStore store(c, model.n_loaded_layers(), 4 * rec, 1, 0);
    for (int l = 0; l < model.n_loaded_layers(); ++l) store.load_layer_experts(model.ckpt(), l, 1);
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    store.copy_to_staging(0, 0, 3, st);          // expert (0,3) now sits in staging slot 0
    store.promote_keys({3}, 1, st);             // promote the same key
    store.commit_all();
    const auto cs = store.cache_stats();
    reuse_d2d = cs.d2d_records == 1;
    events_opened = false;
    cudaStreamSynchronize(st); cudaStreamDestroy(st);
  }
  if (const char* p = getenv("HIVE_CACHE_EVENTS")) { FILE* f = *p ? fopen(p, "rb") : nullptr; if (f) { int ch = fgetc(f); events_opened = ch == 'B'; fclose(f); } }
  RuntimeOptions opt; opt.max_ctx = 512;
  ExpertStore store(c, model.n_loaded_layers(), 0, 1, 0);
  Runtime rt(model, store, nullptr, opt);
  auto seq = rt.new_seq();
  std::vector<int32_t> ids(300, 42);
  std::vector<float> logits;
  rt.forward(*seq, ids.data(), 300, nullptr, &logits, nullptr);
  SeqImage base; rt.save_image(*seq, base);
  rt.forward(*seq, ids.data(), 20, nullptr, &logits, nullptr);
  bool fenced = false, shared = false;
  size_t pool_cached = 0;
  {
    SeqImage img; rt.save_image(*seq, img, &base);
    fenced = img.fence != nullptr;
    shared = img.comp.at(0).segments.size() == 2 && img.comp.at(0).segments[0].data == base.comp.at(0).segments[0].data;
  }
  pool_cached = rt.snapshot_pool_cached_bytes();  // >0 only if the pinned pool exists and took the block back
  rt.reset_seq(*seq);
  const RuntimeOptions& o = rt.opt();
  printf("{\"HIVE_CACHE_REUSE_STAGE\": %s, \"HIVE_CACHE_EVENTS\": %s, \"HIVE_MTP_CACHE\": %s, \"HIVE_PHASE_SCORE\": %s, \"HIVE_PROMOTE_SCORE\": %s, "
         "\"HIVE_PROMOTE_BYTES\": %zu, \"HIVE_CKPT_ASYNC\": %s, \"HIVE_CKPT_DELTA\": %s, \"HIVE_CKPT_PINNED_POOL_MB\": %s}\n",
         reuse_d2d ? "true" : "false", events_opened ? "true" : "false", o.mtp_cache ? "true" : "false", o.phase_score ? "true" : "false",
         o.promote_score_victims ? "true" : "false", o.promote_byte_budget, fenced ? "true" : "false", shared ? "true" : "false",
         pool_cached > 0 ? "true" : "false");
  return 0;
}
