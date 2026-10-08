// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hived — engine daemon. Accepts JSON-lines requests on a Unix socket and streams token ids. Tokenization, chat templates and
// image preprocessing belong to the Python server (server/hive_server.py); this process only maps id sequences to id sequences.
//
// Requests (one JSON object per line):
//   {"op":"generate","rid":"<client request id>","session":"s1","ids":[...],"max_tokens":256 (omitted or 0 = the rest of the context),"temperature":1.0,"top_p":0.95,"top_k":0,"min_p":0,
//    "stop_ids":[1],"seed":0, "images":[{"start":12,"n_vit_h":..,"n_vit_w":..,"nbytes":..,"types":[...]}], "bin":<byte count of the bf16 image patches>,
//    optional thinking cap: "think_cap":N,"think_end_id":id,"think_start_id":id,"think_open":true,"think_exit_ids":[...]; optional "prefix_extra":1;
//    optional "stages":true (adds {"admitted":true,"cached":c,"total":n} once the reuse is decided — sent only to requests that ask)}
//   {"op":"cancel","rid":...} · {"op":"flush"} · {"op":"sleep","level":1|2|3} · {"op":"wake"} · {"op":"stats"}
//   -> response lines: [{"admitted":...} with "stages"] {"progress":i,"total":n} after each prefill chunk · {"id":123} ... {"done":true,"n":k,"finish":"stop|length","prefill_ms":..,"decode_ms":..,"cached_prefix":p}
//   {"op":"cancel","session":"s1"}  -> {"ok":true}
//   {"op":"cancel","session":"s1","rid":"r1"}  -> {"ok":true}  (with rid: only the request that passed the same rid to generate)
//   {"op":"stats"} -> cache and routing statistics ("state": ready|draining|sleeping|waking, "vram", the last "sleep"/"wake" report)
//   {"op":"sleep"[, "timeout_s":N]} -> releases VRAM once in-flight requests finish {"ok":true,"state":"sleeping","sleep":{...}} · {"op":"wake"} -> {"ok":true,"state":"ready",...}
//     (idempotent; failure/timeout responses and the queueing policy are described in the header comment of the "sleep / wake" block below)
//   (optional) generate's "boundaries":[offset...] — read only with HIVE_PREFIX_SHARE (boundary snapshots). Otherwise ignored like any unknown field.
//   {"op":"flush"} — drop every reusable prompt state (session tokens/checkpoints/history, archived sessions, shared boundary snapshots)
//     when no request is decoding; replies {"ok":true,...} or {"error":"not idle",...}. The expert VRAM cache (weights) is kept.
//   (optional) generate's "prefix_extra":N — per-request extra-chunk budget for boundary cuts (overrides HIVE_PREFIX_EXTRA_CHUNKS; HIVE_PREFIX_SHARE only).
// Sessions: if a session's previous token sequence is a prefix of the new request, only the remainder is prefilled (conversation cache).
// If the prefix breaks, resume from whichever of the prompt-end checkpoint and the RAM-held states reaches furthest.
#include <climits>
#include <fcntl.h>
#include <csignal>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hive/cache_reserve.h"
#include "hive/clock.h"
#include "hive/engram_hash.h"
#include "hive/mtp_gate.h"
#include "hive/sampler.h"
#include "hive/socket_writer.h"
#include "nlohmann/json.hpp"
// Model family: the DeepSeek build (default) and the GLM build (-DHIVE_FAMILY_GLM, target hived_glm) share this file; the family header
//   provides Config, Model, ExpertStore, RuntimeOptions, ForwardStats, ImageInput, PrefillPart, Seq, SeqImage, Runtime and the load helpers.
#ifdef HIVE_FAMILY_GLM
#include "hive/glm/hived_family.h"
#else
#include "hive/bulk_load.h"
#include "hive/expert_store.h"
#include "hive/model.h"
#include "hive/runtime.h"
#endif

using json = nlohmann::json;
using namespace hive;
using namespace hive::sampling;
#ifdef HIVE_FAMILY_GLM
using namespace hive::glm::fam;
#endif
#ifndef HIVE_BUILD_ID
#define HIVE_BUILD_ID "unknown"
#endif

namespace {

double now_ms() { return hive::mono_ms(); }
// Project convention (same as env_on in runtime.cpp): unset / "" / "0" = off
bool env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }

// Graceful shutdown (installed only when HIVE_GRACEFUL_STOP_S>0 — by default SIGTERM keeps its default action = immediate exit).
//   The handler only writes one byte to a self-pipe (async-signal-safe). A second signal restores the default action and dies at once (forced exit while draining).
int g_stop_pipe_w = -1;
volatile sig_atomic_t g_stop_seen = 0;
void on_stop_signal(int sig) {
  if (g_stop_seen) { signal(sig, SIG_DFL); raise(sig); return; }
  g_stop_seen = 1;
  const char b = 1;
  const ssize_t r = ::write(g_stop_pipe_w, &b, 1);
  (void)r;
}

struct Session {
  std::unique_ptr<Seq> seq;
  std::vector<int32_t> tokens;  // tokens processed by seq, in order
  double last_used = 0;
  // Conversation KV reuse. The reference encoder defaults to drop_thinking=True, which strips the previous answer's reasoning from the
  //   next turn's prompt, so the session tokens (prompt + generation) and the new request diverge at the start of the previous answer.
  //   Without a checkpoint that divergence forces a full re-prefill of the conversation; keeping the prompt-end (pre-generation) state
  //   means everything up to that point can always be reused.
  std::shared_ptr<SeqImage> ckpt;        // state at the end of the last prompt (right after prefill)
  std::vector<float> ckpt_logits;        // logits at that position — re-requesting the same prompt (regenerate) needs zero prefill
  std::shared_ptr<SeqImage> live;        // last state (including generation) captured when evicted from the pool — for a next request that continues exactly
  // Signatures of the images held by the live state (span start, end, hash of patch bytes) — two different pictures of the same size
    // produce identical placeholder tokens, so token comparison alone cannot tell them apart
  struct ImgSig { int start, end; uint64_t hash; };
  std::vector<ImgSig> img_sig;
  std::vector<ImgSig> ckpt_sig, live_sig;
  struct Frame { std::shared_ptr<SeqImage> image; std::vector<float> logits; std::vector<ImgSig> sig; };
  std::deque<Frame> history;
  size_t host_bytes(std::set<const void*>& seen) const {
    size_t n = tokens.capacity() * sizeof(int32_t) + ckpt_logits.capacity() * sizeof(float);
    if (seq) n += seq->tokens.capacity()*sizeof(int32_t) + seq->engram_history.capacity()*sizeof(int64_t);
    n += (img_sig.capacity() + ckpt_sig.capacity() + live_sig.capacity()) * sizeof(ImgSig);
    if (ckpt) n += ckpt->allocated_bytes(seen);
    if (live) n += live->allocated_bytes(seen);
    for (const auto& f : history) {
      n += sizeof(Frame) + f.logits.capacity() * sizeof(float) + f.sig.capacity() * sizeof(ImgSig);
      if (f.image) n += f.image->allocated_bytes(seen);
    }
    return n;
  }
};

// Boundary snapshots (HIVE_PREFIX_SHARE): states captured at boundaries supplied by the server (end of system/tools, end of a completed
//   turn). Any session may reuse one if its token prefix matches exactly (image signatures included) and the C1 condition holds.
//   upper_skipped = the chunk that ended here skipped the tail layers (C1); that ring holds stale values and is valid only if the next
//   chunk is also committed in tail mode (the same condition as the original path).
struct SharedSnap {
  std::shared_ptr<SeqImage> image;
  std::vector<Session::ImgSig> sig;
  bool upper_skipped = false;
  double last_used = 0;
};
uint64_t prefix_key(const int32_t* t, size_t n, const std::vector<Session::ImgSig>& sig) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](const void* p, size_t k) { for (size_t b = 0; b < k; ++b) { h ^= static_cast<const uint8_t*>(p)[b]; h *= 1099511628211ull; } };
  mix(t, n * sizeof(int32_t));
  mix(&n, sizeof n);
  for (const auto& s : sig) { mix(&s.start, sizeof s.start); mix(&s.end, sizeof s.end); mix(&s.hash, sizeof s.hash); }
  return h;
}

struct Request {
  int fd;
  json req;
  std::vector<uint8_t> bin;
  std::shared_ptr<std::atomic<bool>> cancel;  // each request has its own cancel flag (a per-session shared flag let a re-request clear the previous request's cancel)
  bool solo = false;  // a request put back after a batched-prefill exception — run alone (single path) only
  double t_recv = 0;  // time the receiver thread queued it (now_ms) — queue_ms in the request's final log line (receive -> prefill start)
};

SocketWriter* output = nullptr;
bool send_json(int fd, const json& j) { return output->send(fd, j.dump() + "\n"); }
void finish_socket(int fd) { output->finish(fd); }

// The sampler (sample, Cands, sample_cands) lives in hive/sampler.h (unchanged code — the CPU distribution test tests/test_sampler_cpu.cpp calls the same functions).
void snapshot_cands(Runtime& rt, int row, Cands& c) {
  const int NC = rt.n_cands();
  if (NC <= 0) { c.valid = false; return; }
  c.idx.assign(rt.cand_idx(row), rt.cand_idx(row) + NC);
  c.val.assign(rt.cand_val(row), rt.cand_val(row) + NC);
  c.mx = rt.row_max(row); c.sum = rt.row_sumexp(row); c.argmax = rt.row_argmax(row);
  c.valid = true;
}

// Host memory of this process (/proc/self/status — VmRSS, VmHWM, VmLck, VmPin, MiB). Empty object if it cannot be read (absorbed).
json proc_memory_mib() {
  json out = json::object();
  FILE* f = fopen("/proc/self/status", "r");
  if (!f) return out;
  char line[256];
  while (fgets(line, sizeof line, f)) {
    long kb = 0;
    for (const char* k : {"VmRSS", "VmHWM", "VmLck", "VmPin"}) {
      const size_t n = strlen(k);
      if (!strncmp(line, k, n) && line[n] == ':' && sscanf(line + n + 1, "%ld", &kb) == 1) out[std::string(k) + "_mib"] = kb / 1024.0;
    }
  }
  fclose(f);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = getenv("HIVE_CKPT") ? getenv("HIVE_CKPT") : "", engram_dir, sock_path = "/tmp/hive.sock";
  int max_layer = -1, cpu_threads = 16, max_chunk = 4096, max_sessions = 2;
  int prefill_tile = 1;  // --prefill-tile N (max_chunk x N tokens per tile, experts loaded once per encoder layer)
  double host_session_mb = 32768;  // budget for session images kept outside the pool (RAM) — oldest are dropped beyond it
  int ckpt_min_tokens = 256;       // prompts shorter than this get no checkpoint (re-reading is cheaper)
  double decode_share = 0.34;  // wall-clock share given to running decoders between prefill chunks (0 = one decode step per chunk)
  int prefill_threshold = -1;  // --prefill-threshold N (RuntimeOptions.prefill_threshold, default 1024 unchanged) — no rebuild needed for A/B
  int busy_chunk = 0;  // prefill chunk while others are decoding (0 = no split: max_chunk x prefill_tile as is)
  const double prefill_budget_ms = getenv("HIVE_PREFILL_BUDGET_MS") ? std::max(0.0, atof(getenv("HIVE_PREFILL_BUDGET_MS"))) : 0;
  double prefill_ms_per_token = 0;  // observed chunk EMA, not a hard latency bound
  bool fake_head = false;  // exercise only the protocol on a partial checkpoint: fake (deterministic random) logits
  bool no_vision = false, vision_eager = false;
  int vision_max_patches = 0, promote = -1, gpu_share = -1, decoder_tail = -1, max_batch_arg = -1;
  bool decoder_replay = false;
  int promote_misses = -1;
  float promote_miss_ratio = 0.f;
  std::string cache_state = "/out/cache-state.bin";  // warm-start file (resident expert keys, by score) — refreshed every 60 s, loaded first at startup. "" = off
  bool no_mtp = false;      // disable DSpark speculative decoding (do not load the mtp stage even if the model has one)
  // Start asleep (--start-asleep or HIVE_START_ASLEEP via env_on; default off = normal startup): experts and engram go to pinned RAM, dense
    //   weights and Work are placed with VMM (hive/devmem.h) and then dropped to host copies (sleep level 3); becomes ready as 'sleeping'
    //   without the slot region or session pool. The first wake claims VRAM and warms up.
  bool start_asleep = env_on("HIVE_START_ASLEEP");
  float mtp_conf = -1.0f;   // draft submission confidence threshold (logit): drafts are verified from the front while conf >= threshold (calibrate with HIVE_TRACE_MTP logs)
  int mtp_max = 0;          // cap on drafts verified per step (0 = the whole block)
  double cache_mb = 70000;
  int64_t max_ctx = 262144;
  for (int i = 1; i < argc; i++) {
    const std::string a(argv[i]);
    auto next = [&]() -> std::string { return argv[++i]; };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--engram") engram_dir = next();
    else if (a == "--sock") sock_path = next();
    else if (a == "--max-layer") max_layer = atoi(next().c_str());
    else if (a == "--cpu-threads") cpu_threads = atoi(next().c_str());
    else if (a == "--vram-cache-mb") cache_mb = atof(next().c_str());
    else if (a == "--max-ctx") max_ctx = atoll(next().c_str());
    else if (a == "--max-chunk") max_chunk = atoi(next().c_str());
    else if (a == "--prefill-tile") prefill_tile = std::max(1, atoi(next().c_str()));
    else if (a == "--decode-share") decode_share = atof(next().c_str());
    else if (a == "--busy-chunk") busy_chunk = atoi(next().c_str());
    else if (a == "--prefill-threshold") prefill_threshold = atoi(next().c_str());
    else if (a == "--max-sessions") max_sessions = atoi(next().c_str());
    else if (a == "--host-session-mb") host_session_mb = atof(next().c_str());
    else if (a == "--ckpt-min-tokens") ckpt_min_tokens = atoi(next().c_str());
    else if (a == "--fake-head") fake_head = true;
    else if (a == "--no-vision") no_vision = true;
    else if (a == "--vision-eager") vision_eager = true;  // load the vision encoder at startup (default: lazy load on the first image, released when idle)
    else if (a == "--vision-max-patches") vision_max_patches = atoi(next().c_str());
    else if (a == "--promote") promote = atoi(next().c_str());
    else if (a == "--promote-misses") {  // N | auto | auto:cap
      const std::string v = next();
      if (v == "auto") { promote_misses = 32; promote_miss_ratio = 0.25f; }
      else if (v.rfind("auto:", 0) == 0 && atoi(v.c_str() + 5) > 0) { promote_misses = atoi(v.c_str() + 5); promote_miss_ratio = 0.25f; }
      else if (!v.empty() && v.find_first_not_of("0123456789") == std::string::npos) { promote_misses = atoi(v.c_str()); promote_miss_ratio = 0.f; }
      else { fprintf(stderr, "--promote-misses: must be N | auto | auto:N (N>0): %s\n", v.c_str()); return 1; }
    }  // miss promotion (N per step -> LRU slots)
    else if (a == "--gpu-share") gpu_share = atoi(next().c_str());
    else if (a == "--decoder-tail") decoder_tail = atoi(next().c_str());
    else if (a == "--decoder-replay") decoder_replay = true;  // shrink the decoder tail to the window length (128) and look only at the replay span (deployment recipe, approximate)
    else if (a == "--max-batch") max_batch_arg = atoi(next().c_str());
    else if (a == "--no-mtp") no_mtp = true;
    else if (a == "--start-asleep") start_asleep = true;
    else if (a == "--cache-state") cache_state = next();
    else if (a == "--mtp-conf") mtp_conf = (float)atof(next().c_str());
    else if (a == "--mtp-max") mtp_max = atoi(next().c_str());
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
  }
  if (ckpt.empty()) { fprintf(stderr, "--ckpt DIR (or HIVE_CKPT) is required\n"); return 1; }
  const double t_load = now_ms();
  // Faster cold load, HIVE_LOAD_PAR (unset/""/"0" = off; measurements and design in the header of hive/bulk_load.h). When on: (1) start a dense-tensor
    //   prefetch thread before constructing Model (page cache — so Model's mmap reads do not wait on disk), (2) read the expert and engram tables
    //   with one parallel O_DIRECT pass, overlapping pinned registration. The resulting bytes are identical (tools/test_load_par_cpu.py).
    //   If the bulk load fails, the store reloads via the regular path.
  const int load_par = load_par_threads();
  DensePrefetch dense_pf;
  if (load_par > 0) dense_pf.start(ckpt, max_layer, 4);
  // HIVE_LOAD_PREFAULT (unset/""/"0" = off): claim the expert arena and engram table memory now so first touch (including THP compaction) overlaps the dense load — see HostPrefault in expert_store.h
  const int load_prefault = load_prefault_threads();
  if (load_prefault > 0) HostPrefault::start(HostPrefault::predict(ckpt, max_layer, no_mtp), load_prefault);
  // Logging: steady now_ms -> epoch ms offset (computed once) — wall t0/t1/t2 in the request's final log line
  const double wall_off = hive::Millis(std::chrono::system_clock::now().time_since_epoch()).count() - now_ms();
  if (start_asleep) {  // VMM is needed to restore dense weights at the same addresses — without it, absorb into a normal startup (ready and awake)
    devmem::force_on();
    if (!devmem::on()) { start_asleep = false; fprintf(stderr, "[hived] ⚠️--start-asleep needs CUDA VMM — starting awake\n"); }
  }
  // Faster first sleep (hive/devmem.h prepare_shadows): the VMM regions claimed here (dense weights, MTP stage, and below the Runtime
    //   constructor's Work, cuBLAS, sampler) are the "long-lived" regions that survive down to sleep level 3 — their pinned copies are claimed
    //   during startup. With VMM off, keep_* does nothing.
  devmem::keep_begin();
  {  // Fitness check before the long load: the kernels are built for sm_120a (Blackwell, compute capability 12.x) and the CPU expert pool for
     //   AVX2/FMA/F16C — on other hardware the first kernel would fail with a bare CUDA error and the first CPU expert with SIGILL.
    cudaDeviceProp prop{};
    const cudaError_t st = cudaGetDeviceProperties(&prop, 0);
    if (st != cudaSuccess || prop.major != 12) {
      fprintf(stderr, "[hived] this build runs on compute capability 12.x GPUs (Blackwell, sm_120a); found %s (%d.%d, %s) — see docs/requirements.md\n",
              st == cudaSuccess ? prop.name : "no CUDA device", prop.major, prop.minor, st == cudaSuccess ? "unsupported" : cudaGetErrorString(st));
      return 2;
    }
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c")) {
      fprintf(stderr, "[hived] this build needs a CPU with AVX2, FMA and F16C — see docs/requirements.md\n");
      return 2;
    }
  }
  Model model(ckpt, max_layer, max_ctx);
  size_t vram_model_used = 0;  // sleep VRAM table: weights (dense + MTP) + CUDA context = usage right after model load (measured)
  { size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot); vram_model_used = tot > fr ? tot - fr : 0; }
  const Config& c = model.cfg();
  const int nl = model.n_loaded_layers();
  if (!no_mtp && nl == c.n_layers && model.has_head()) model.load_mtp();
  const int n_mtp = model.n_mtp();
  devmem::keep_end();
  dense_pf.join();
  const double t_model = now_ms();  // cold-load phase timestamps (the [hived] load phases line below)
  // HIVE_CACHE_FIT (optional; unset/""/"0" = off: size the slots from --vram-cache-mb here). When on, the slot region is sized **after** the
    //   session pool, as "free VRAM at that point - HIVE_CACHE_RESERVE_MB (unset, empty, non-numeric or negative = 2400 MiB)". Headroom a
    //   hand-tuned --vram-cache-mb used to leave beyond the operating rule (>= 2,300 MiB: lazily loaded vision encoder ~1.7 GB + image
    //   patches + graph growth), and VRAM reclaimed elsewhere (e.g. runtime HIVE_ENGRAM_ALIAS), become slots automatically. The warm start
    //   also moves after that point (once slots exist). --vram-cache-mb is ignored in this mode (logged only).
    //   Lossless: only the number of slots changes (header comment of alloc_cache in expert_store.cpp).
  static const bool cache_fit = env_on("HIVE_CACHE_FIT");
  const size_t cache_reserve = cache_reserve_bytes();  // hive/cache_reserve.h (the GLM KV growth keeps the same headroom)
  ExpertStore store(c, nl, (size_t)(cache_mb * 1048576.0), cpu_threads, n_mtp, cache_fit || start_asleep);  // started asleep: slots are allocated at the first wake
  const double t_store = now_ms();
  if (load_par > 0) {  // HIVE_LOAD_PAR: all expert layers + engram tables at once (overlapping registration) — load_bulk falls back to the regular path on failure
    std::vector<std::pair<int, int>> engram_tables;
    for (size_t i = 0; i < c.engram_layer_ids.size(); ++i) {
      const int el = c.engram_layer_ids[i];
      if (el >= nl) continue;
      HIVE_CHECK(!engram_dir.empty(), "--engram DIR is required");
      engram_tables.push_back({el, (int)i});
    }
    fprintf(stderr, "[hived] load par: dense prefetch %.1f GB in %.1fs (model+mtp %.1fs)\n", dense_pf.bytes.load() / 1e9, dense_pf.sec, (t_model - t_load) / 1000.0);
    store.load_bulk(model.ckpt(), nl + n_mtp, engram_tables, load_opts_from_env(load_par), true);
  } else {
  for (int l = 0; l < nl + n_mtp; ++l) store.load_layer_experts(model.ckpt(), l);
  store.pin_all();
  }
  EngramHash eh;
  bool have_eh = false;
  for (size_t i = 0; i < c.engram_layer_ids.size(); ++i) {
    int el = c.engram_layer_ids[i];
    if (el >= nl) continue;
    HIVE_CHECK(!engram_dir.empty(), "--engram DIR is required");
    if (!have_eh) { eh.load(engram_dir + "/engram_hash.json", engram_dir + "/token_map.bin"); have_eh = true; }
    if (load_par <= 0) store.load_engram(model.ckpt(), el, (int)i);
  }
  store.engram_ssd_finish(model.ckpt());  // HIVE_ENGRAM_SSD: row cache + the `[engram-ssd]` startup line (no-op when off)
  HostPrefault::finish();  // does nothing unless enabled
  if (load_checksum_on()) {  // verification log (default off): compare host-copy bits between the regular and the new load path
    const double tc = now_ms();
    const std::vector<uint64_t> cs = store.host_checksums(32);
    std::string line;
    for (uint64_t x : cs) { char b[24]; snprintf(b, sizeof b, " %016llx", (unsigned long long)x); line += b; }
    fprintf(stderr, "[hived] host checksum (experts node0 node1 · engram vals/scales per table):%s · %.1fs\n", line.c_str(), (now_ms() - tc) / 1000.0);
  }
  const double t_weights = now_ms();
  RuntimeOptions opt;
  opt.max_chunk = max_chunk;
  opt.prefill_tile = prefill_tile;
  opt.max_ctx = max_ctx;
  opt.vision = !no_vision;
  opt.vision_lazy = !vision_eager;
  opt.vision_max_patches = vision_max_patches;
  if (promote >= 0) opt.promote_per_token = promote;
  if (promote_misses >= 0) opt.promote_misses = promote_misses;
  if (promote_miss_ratio > 0.f) opt.promote_miss_ratio = promote_miss_ratio;
  if (gpu_share >= 0) opt.decode_gpu_share = gpu_share;
  if (decoder_tail >= 0) opt.decoder_tail = decoder_tail;
  if (decoder_replay) { opt.decoder_replay = true; if (decoder_tail < 0) opt.decoder_tail = model.cfg().window; }
  if (max_batch_arg > 0) opt.max_batch = max_batch_arg;
  if (prefill_threshold > 0) opt.prefill_threshold = prefill_threshold;
  opt.mtp = !no_mtp;
  max_sessions = std::max(max_sessions, opt.max_batch);
  devmem::keep_begin();
  Runtime rt(model, store, have_eh ? &eh : nullptr, opt);
  devmem::keep_end();
  const double t_runtime = now_ms();
  // Faster first sleep: claim the sleep copies of the keep regions (pinned, ~10 GB) in the background — joined before ready (overlaps the
    //   session pool, slots and warm start). Measurements and mechanism in the header of hive/devmem.h. HIVE_SLEEP_PREPIN=0 = claim them at the
    //   first level-3 sleep instead. Not needed when started asleep (release_all claims them right away).
  std::thread shadow_prep;
  if (devmem::on() && !start_asleep && !(getenv("HIVE_SLEEP_PREPIN") && !strcmp(getenv("HIVE_SLEEP_PREPIN"), "0")))
    shadow_prep = std::thread([] {
      const devmem::Report r = devmem::prepare_shadows();
      fprintf(stderr, "[hived] sleep shadows pinned ahead: %.0f MiB in %d regions in %.0f ms%s (the first level-3 sleep reuses them)\n", r.shadow_new / 1048576.0, r.regions,
              r.ms, r.shadow_failed ? (" · " + std::to_string(r.shadow_failed) + " regions not pinned (absorbed — allocated at sleep as before)").c_str() : "");
    });
  {  // release the checkpoint page cache (load complete — shards are not read again)
    auto mem_avail = [] { FILE* f = fopen("/proc/meminfo", "r"); long kb = 0; char line[256]; while (f && fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break; if (f) fclose(f); return kb / 1024 / 1024; };
    const long before = mem_avail();
    const size_t dropped = model.ckpt().release_all();
    fprintf(stderr, "[hived] checkpoint page cache released: %.0f GB (MemAvailable %ld → %ld GB)\n", dropped / 1e9, before, mem_avail());
  }
  // Warm start: load the previous instance's resident list (pinned RAM -> VRAM, ~3 s for all). Ignored if the header (layer count, E) differs.
  auto save_cache_state = [&]() {
    if (cache_state.empty()) return;
    std::vector<int32_t> keys = store.resident_keys_by_score();
    const std::string tmp = cache_state + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    const int32_t hdr[4] = {0x48564353, store.n_layers(), store.E(), (int32_t)keys.size()};
    fwrite(hdr, 4, 4, f);
    fwrite(keys.data(), 4, keys.size(), f);
    fclose(f);
    rename(tmp.c_str(), cache_state.c_str());
  };
  std::vector<int32_t> start_keys;  // started asleep: only read the warm-start list; the first wake loads it
  auto warm_start = [&]() {  // with HIVE_CACHE_FIT, called after the slots are allocated (below the session pool) — otherwise right here
  if (!cache_state.empty()) {
    FILE* f = fopen(cache_state.c_str(), "rb");
    if (f) {
      int32_t hdr[4] = {0, 0, 0, 0};
      std::vector<int32_t> keys;
      if (fread(hdr, 4, 4, f) == 4 && hdr[0] == 0x48564353 && hdr[1] == store.n_layers() && hdr[2] == store.E() && hdr[3] > 0 && hdr[3] < 100000) {
        keys.resize(hdr[3]);
        if (fread(keys.data(), 4, keys.size(), f) != keys.size()) keys.clear();
      }
      fclose(f);
      if (!keys.empty() && start_asleep) {
        start_keys = keys;
        fprintf(stderr, "[hived] warm start: %zu experts from %s kept for the first wake (started asleep)\n", keys.size(), cache_state.c_str());
      } else if (!keys.empty()) {
        const double tw = now_ms();
        const int n = rt.warm_from_keys(keys);
        fprintf(stderr, "[hived] warm start: %d/%zu experts from %s in %.0f ms\n", n, keys.size(), cache_state.c_str(), now_ms() - tw);
      } else {
        fprintf(stderr, "[hived] warm start: %s ignored (header mismatch)\n", cache_state.c_str());
      }
    }
  }
  };
  if (!cache_fit) warm_start();
  double last_state_save = now_ms();
  const int mtp_B = rt.mtp_enabled() ? rt.mtp_block() : 0;
  if (mtp_max <= 0 || mtp_max > mtp_B) mtp_max = mtp_B;
  static const bool trace_mtp = getenv("HIVE_TRACE_MTP") && strcmp(getenv("HIVE_TRACE_MTP"), "0") != 0;
  // Tail-aware chunk split (optional, default off). With N = 16384 + r (small r) the last chunk cannot be in tail mode (<= decoder_tail), so
    //   the previous chunk loses C1, and if r < prefill_threshold that chunk takes the short path with a large CPU share -> move the boundary
    //   between the last two chunks so the last chunk is at least the minimum tail-mode size.
    //   Lossless: chunk boundaries only process the same tokens at the same positions in the same causal order, so the result is identical
    //   up to fp32 accumulation order (the same property as choosing a chunk size).
  static const bool tail_split = getenv("HIVE_TAIL_SPLIT") && *getenv("HIVE_TAIL_SPLIT") && strcmp(getenv("HIVE_TAIL_SPLIT"), "0") != 0;
  int tail_need = 0;  // smallest chunk that is in tail mode (0 = none in this configuration) — decided by rt.tail_mode_for itself (monotone in M)
  if (tail_split)
    for (int m = 1; m <= max_chunk * rt.prefill_slots(); ++m) if (rt.tail_mode_for(m)) { tail_need = m; break; }
  if (tail_split) fprintf(stderr, "[hived] tail split on: final prefill chunk >= %d rows\n", tail_need);
  // HIVE_PREFILL_HOST_TILES, HIVE_BATCH_PREFILL — read by the runtime constructor. Text chunk cap = max_chunk x prefill_slots
  if (rt.host_slots() > 0 || rt.batch_prefill())
    fprintf(stderr, "[hived] prefill slots %d (tile %d + host %d) · text chunk cap %d · batch prefill %s\n", rt.prefill_slots(), rt.prefill_tile(), rt.host_slots(),
            max_chunk * rt.prefill_slots(), rt.batch_prefill() ? (rt.prefill_slots() >= 2 ? "on" : "on but inactive (needs >= 2 slots)") : "off");
  // Boundary snapshots and cross-session prefix sharing (optional, default off). Extra-chunk cap: a boundary that does not increase the
    //   chunk count is cut for free; one that does (usually inside the last chunk) is cut at most this many times per request (default 1 —
    //   the one split-off tail piece is the whole extra cost, paid so the next request with the same prefix can skip that span entirely).
  static const bool prefix_share = env_on("HIVE_PREFIX_SHARE");
  static const int prefix_extra = !prefix_share ? 0 : (getenv("HIVE_PREFIX_EXTRA_CHUNKS") && *getenv("HIVE_PREFIX_EXTRA_CHUNKS")) ? std::max(0, atoi(getenv("HIVE_PREFIX_EXTRA_CHUNKS"))) : 1;
  // HIVE_DEFER_CKPT (optional, default off): take the prompt checkpoint and warm_cache after the first token is sent — right after the first
    // decode step's send and before its forward (the state is exactly the prompt end)
  static const bool defer_ckpt = env_on("HIVE_DEFER_CKPT");
  // Graceful shutdown (optional, 0 = off = SIGTERM exits immediately): finish in-flight requests for up to N seconds, then cancel them (shutdown error)
  const double graceful_stop_s = getenv("HIVE_GRACEFUL_STOP_S") && *getenv("HIVE_GRACEFUL_STOP_S") ? std::max(0.0, atof(getenv("HIVE_GRACEFUL_STOP_S"))) : 0.0;
  if (prefix_share) fprintf(stderr, "[hived] prefix share on: boundary snapshots, extra chunks per request %d\n", prefix_extra);
  if (defer_ckpt) fprintf(stderr, "[hived] deferred prompt checkpoint on (after the first token)\n");
  // HIVE_WARM_BUSY_CAP (optional; unset / "" / "0" = off): while other requests are decoding, the synchronous cache warm after a prefill is
  //   capped at N experts (1 = effectively skipped — the per-step paced promotions fill the cache instead). Measured (chat shape,
  //   4 decoders): a follow-up turn's 699-row prefill (1.03 s) was followed by "warm cache: 1065 experts in 693 ms", during which every
  //   running decoder stalled (one such warm per prefill that leaves the cache below 90 %). With nothing else running the full warm_cap applies as before.
  //   Rejected (same shape, cap 1): the decoders' longest gap fell 1.44 → 1.04 s but the cache went stale — 17.0 → 10.3 tok/s per decoder
  //   (the warm is the general top-score refill after a prefill, not only the new request's experts). Kept as a documented switch; HIVE_WARM_DEFER below
  //   keeps the refill and removes the stall.
  static const int warm_busy_cap = [] { const char* v = getenv("HIVE_WARM_BUSY_CAP"); char* e = nullptr; const long x = v && *v ? strtol(v, &e, 10) : 0;
                                        return v && *v && e && *e == 0 && x > 0 ? (int)std::min<long>(x, 1 << 20) : 0; }();
  if (warm_busy_cap) fprintf(stderr, "[hived] warm cache busy cap: %d experts while other requests decode\n", warm_busy_cap);
  // HIVE_WARM_DEFER (optional; unset / "" / "0" = off): the cache warm after a prefill is not uploaded synchronously but handed to the following decode steps,
  //   at most N experts per step (Runtime::warm_defer — the same top-score promotions, paced). Measured (chat shape, 4 decoders + cached
  //   follow-up turns, off ×3 vs 16 ×2 vs 32 ×2): decoder 17.0 → 19.0–19.2 → 21.5–21.6 tok/s, follow-up turns 3.1–3.3/2.3/2.3 → 1.9–2.1/1.7–1.9/1.7 s,
  //   the decoders' longest gap 1.43–1.45 → 1.05–1.07 s (docs/performance.md step 19). Takes precedence over HIVE_WARM_BUSY_CAP. Only the score-based step promotion carries the quota, so with
  //   --promote-misses (promote_misses > 0) or --promote 0 the warm stays synchronous and the switch is ignored (banner).
  int warm_defer = [] { const char* v = getenv("HIVE_WARM_DEFER"); char* e = nullptr; const long x = v && *v ? strtol(v, &e, 10) : 0;
                        return v && *v && e && *e == 0 && x > 0 ? (int)std::min<long>(x, 1 << 20) : 0; }();
  if (warm_defer && opt.promote_misses > 0) { fprintf(stderr, "[hived] warm cache deferred: ignored with promote-misses (synchronous warm)\n"); warm_defer = 0; }
  if (warm_defer && opt.promote_per_token <= 0) { fprintf(stderr, "[hived] warm cache deferred: ignored with --promote 0 (no step promotion to carry it; synchronous warm)\n"); warm_defer = 0; }
  if (warm_defer) fprintf(stderr, "[hived] warm cache deferred: <= %d experts per decode step\n", warm_defer);
  // Prefill scheduling (optional, default off — meaningful only with HIVE_BATCH_PREFILL). All of these change only the schedule (who runs how
    //   many rows in which forward); per-request math is unchanged (same tokens, positions and causal order). What may differ is the same as
    //   for batched prefill: chunk boundaries and the set of rows sharing a forward change, so only the fp32 accumulation order from expert
    //   GEMM row grouping and the CPU/DMA split for missed experts (which depends on the row count) can differ.
    //   HIVE_BATCH_WINDOW: before the first round of a batched prefill, wait up to HIVE_BATCH_WINDOW_MS (unset/empty = 50, <= 0 or non-numeric =
    //     no window, capped at 10 s) for following requests and put them in the same forward. Waits only if the head request is streaming-sized
    //     (>= prefill_threshold) and slots and seats (max_batch) remain; ends at once when slots fill up or the queue head cannot be batched
    //     (image, same session). Running decoders keep stepping while it waits (they do not stall).
    //     Measured: 16K x3 concurrent = 17.7 -> 14.9 s — without the window the first request arrived first and ran alone (only the other two were batched).
    //   HIVE_BATCH_SJF: slot allocation per round — first to requests that can finish this round (not shortest-first but FIFO among "the rest
    //     fits"), then the remaining slots FIFO to the others. Short requests (< prefill_threshold rows = their own forward) cost no slots. If the
    //     oldest request cannot finish this round it always keeps one slot (no starvation). Without it the leading request took every slot and a
    //     request joining behind a long prefill got none until that prefill finished.
    //     Part order (forward_multi) remains request order.
    //   HIVE_PREFILL_FAIR: when one request's (or batch's) prefill ends and another prefill is queued, give running decoders the same
    //     decode_share as between chunks before taking the next prefill. Without it, back-to-back single-chunk prefills left decoders only one
    //     step in between (the between-chunk share applies only within one request).
  static const bool batch_window = env_on("HIVE_BATCH_WINDOW");
  static const double batch_window_ms = !batch_window ? 0.0 : [] {
    const char* v = getenv("HIVE_BATCH_WINDOW_MS");
    const double x = v && *v ? atof(v) : 50.0;  // non-numeric = atof 0 = off (treated like "0")
    return std::isfinite(x) && x > 0 ? std::min(x, 10000.0) : 0.0;
  }();
  static const bool batch_sjf = env_on("HIVE_BATCH_SJF");
  static const bool prefill_fair = env_on("HIVE_PREFILL_FAIR");
  if (batch_window || batch_sjf || prefill_fair)
    fprintf(stderr, "[hived] prefill scheduling: batch window %.0f ms · slot share %s · fair gap %s%s\n", batch_window_ms, batch_sjf ? "on" : "off",
            prefill_fair ? "on" : "off", rt.batch_prefill() ? "" : " (window/slot share need HIVE_BATCH_PREFILL)");
  // HIVE_PREFILL_YIELD (optional, env_on, default off): at each **existing** chunk/round boundary of a long prefill (before the next chunk),
    //   admit queued short requests (prompt id count <= HIVE_PREFILL_YIELD_MAX — unset, empty, non-numeric or <= 0 = prefill_threshold - 1, i.e.
    //   the short/SMALL path below the streaming path) via the single path (admit) into active, and emit their first token before the next
    //   chunk. Measured (quality suite, service configuration): a short request sent during a 100K prefill had TTFT 15.5 s (the long request
    //   15.3 s — it waited for the whole long prefill; 0.22 s when run sequentially). Without it, the single path did not look at the queue
    //   between chunks, and the batched path joined the short request at the next round, so its first token left only after that round's long
    //   chunk forward (with SJF too, solo runs after multi).
    //   - Schedule only: no new chunk boundaries for the long request (chunk size rules, C1 and tail split unchanged — splitting one 98K forward
    //     would change the expert CPU/DMA split and fp32 accumulation order and thus the output bits, so it is not split). A request arriving
    //     during a forward therefore waits until that forward ends — little effect for a one-chunk prompt (98,304 rows ~ 14 s); yielding at
    //     layer boundaries is the runtime's job (HIVE_LAYER_YIELD below).
    //   - The short request's math equals a standalone request (single-path admit as is). If active was empty, the long request waits for just
    //     one extra decode_step for that first token (the decode_share applies only when decoders already exist, as before). Admitted short
    //     requests then get decode_share between later chunks like any decoder.
    //   - C1: active can grow between chunks, so the single path also sizes the next chunk's lower bound with c1_floor_j (same formula as the
    //     batched path; with service values it stays chunk_cap).
    //   - Same-session requests and sessions active or prefilling are skipped (stay queued); seats (max_batch) = active + prefilling + 1; no
    //     recursion while yielding.
  static const bool prefill_yield = env_on("HIVE_PREFILL_YIELD");
  static const size_t prefill_yield_max_env = !prefill_yield ? 0 : [] {
    const char* v = getenv("HIVE_PREFILL_YIELD_MAX");
    char* end = nullptr;
    const long long x = v && *v ? strtoll(v, &end, 10) : 0;
    return v && *v && end && *end == 0 && x > 0 ? (size_t)std::min<long long>(x, 1LL << 30) : (size_t)0;  // only if the whole string is numeric and > 0
  }();
  // HIVE_LAYER_YIELD (optional; unset/""/"0" = off): yield at layer boundaries **inside** a long prefill forward (runtime.cpp, hive/layer_yield.h).
    //   Measured: short-request TTFT 15.5 s during a 100K prefill; a running decode stalled 14 s during a long prefill; in service, up to 4.9 s
    //   five times in 70 minutes. HIVE_PREFILL_YIELD only sees boundaries between forwards (chunk/round), so a one-forward prompt (98,304 rows)
    //   offered no boundary to cut in at.
    //   Values: period in ms (>= 50; "1" or non-numeric = 500) · HIVE_LAYER_YIELD_SHARE (decode share, default 0.05) · HIVE_LAYER_YIELD_STEPS
    //   (step cap per yield, default 4) · HIVE_LAYER_YIELD_MAX (prompt id cap for short requests admitted by yielding — unset, non-numeric or
    //   <= 0 = prefill_threshold - 1, same convention as HIVE_PREFILL_YIELD).
    //   Yield body (ly_run below): admit eligible short text requests via the single path (admit) up to their first token, and give running
    //   decoders steps (>= 1, up to the budget and cap).
    //   Schedule only: chunk boundaries, chunk sizes and forward grouping are unchanged (keeps the "not split" rule above).
  static const double ly_period = ly::parse_period(getenv("HIVE_LAYER_YIELD"));
  static const double ly_share = ly::parse_share(getenv("HIVE_LAYER_YIELD_SHARE"));
  static const int ly_steps = ly::parse_steps(getenv("HIVE_LAYER_YIELD_STEPS"));
  static const size_t ly_max_env = ly_period <= 0 ? 0 : [] {
    const char* v = getenv("HIVE_LAYER_YIELD_MAX");
    char* end = nullptr;
    const long long x = v && *v ? strtoll(v, &end, 10) : 0;
    return v && *v && end && *end == 0 && x > 0 ? (size_t)std::min<long long>(x, 1LL << 30) : (size_t)0;
  }();
  const bool layer_yield = ly_period > 0;
  // HIVE_LAYER_YIELD_MID (optional; unset/""/"0"/non-numeric = off; N = row cap): at a layer yield also admit a request of more than
    //   HIVE_LAYER_YIELD_MAX rows (up to N, and up to the rows the runtime parked) when it is much smaller than what the paused prefill still has
    //   to do — shortest remaining work first. Measured on the service log (10-01..10-05, benchmarks excluded): 114 requests of >= 4K rows waited
    //   > 2 s behind another prefill of >= 4K rows, e.g. three 22K-row requests 5.3 s behind a 20K one and an 11K one 5.4 s behind a 109K one;
    //   the layer yield admitted only <= 1,023-row requests, so they waited for the whole forward. Rule: rows x 2 <= the paused forward's
    //   remaining rows (its rows x (1 - progress at this layer) + its later chunks), and the rows admitted this way during one forward stay <=
    //   that forward's rows (the paused prompt is delayed by at most about its own prefill time). The factor 2 covers the admitted request's
    //   own expert pass, which it does not share.
  static const size_t ly_mid = ly_period <= 0 ? 0 : [] {
    const char* v = getenv("HIVE_LAYER_YIELD_MID");
    char* end = nullptr;
    const long long x = v && *v ? strtoll(v, &end, 10) : 0;
    return v && *v && end && *end == 0 && x > 0 ? (size_t)std::min<long long>(x, 1LL << 30) : (size_t)0;
  }();
  // Arrival coalescing (HIVE_BATCH_COALESCE_MS, HIVE_BATCH_PREFILL path): before the first round of a batched prefill, if the head request is
    //   streaming-sized and seats/slots remain, wait until following arrivals pause (no new arrival for this many ms after the last one; cap =
    //   this many ms x prefill slots) and put them in the same forward.
    //   Measured (bench_conc_pf 2.5K x 4 concurrent, hived log wall t0/queue_ms): the server hands a burst to hived ~10 ms apart (arrival gaps
    //   8/10/8 ms, 11/9/10 ms). A single pull right after adding the first request made batching a race against the time that add took — in one
    //   run the first add took ~20 ms (evicting and saving a 27K session) and all 4 were batched (5.16 s); in another the add was short, the
    //   first ran alone and the other 3 ran after its forward (7.48 s). The same race gave 2.5K x 2 rep0 [2.38, 5.21].
    //   Layer yielding cannot take these 3 (> 1023 ids) inside a forward (the rows it can hand over are only the leading Work rows) — hence
    //   they are gathered before the forward starts.
    //   Values: unset = 25 when layer yield is on (a bit over twice the largest observed gap of 11 ms — +25 ms for a lone streaming prefill
    //   (>= 1.5 s)), else 0 (off) · "0" = off · number >= 0 = that many ms (cap 1000) · non-numeric = same as unset. Together with
    //   HIVE_BATCH_WINDOW, whichever is reached first ends the wait.
  static const double batch_quiet_ms = [&] {
    const char* v = getenv("HIVE_BATCH_COALESCE_MS");
    char* end = nullptr;
    const double x = v && *v ? strtod(v, &end) : -1.0;
    const bool num = v && *v && end != v && end && *end == 0 && std::isfinite(x) && x >= 0;
    return num ? std::min(x, 1000.0) : (ly_period > 0 ? 25.0 : 0.0);
  }();
  if (batch_quiet_ms > 0) fprintf(stderr, "[hived] batch coalesce on: first prefill round waits until arrivals pause %.0f ms\n", batch_quiet_ms);

  // ---- socket ----
  ::unlink(sock_path.c_str());
  int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  HIVE_CHECK(lfd >= 0, "socket");
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  HIVE_CHECK(sock_path.size() < sizeof(addr.sun_path), "socket path too long: " + sock_path);  // strncpy would bind a silently truncated name
  strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
  HIVE_CHECK(::bind(lfd, (sockaddr*)&addr, sizeof(addr)) == 0, "bind " + sock_path);
  HIVE_CHECK(::listen(lfd, SOMAXCONN) == 0, "listen");  // SOMAXCONN: a backlog of 16 gave connect EAGAIN under connection bursts (lost cancel requests) — test_daemon_cpu accept burst
  chmod(sock_path.c_str(), 0666);
  fprintf(stderr, "[hived] listening on %s\n", sock_path.c_str());
  const size_t kMaxHeaderBytes = (size_t)64 << 20;  // request line (ids, sampling, image geometry) — a full-context id list is a few MiB
  const size_t kMaxBinBytes = (size_t)1 << 30;      // binary payload (bf16 image patches of all images of one request); per-image geometry is checked at admission

  std::mutex mu;
  std::condition_variable cv;
  std::deque<Request> queue;
  // Control requests (sleep/wake) — queued by the receiver thread, handled by the engine thread, which answers on that fd. deadline = the
    // sleep's timeout_s (0 = wait until done)
  struct Ctl { int fd; bool sleep; double deadline; int level; };
  std::deque<Ctl> ctl_q;
  std::vector<int> flush_q;  // {"op":"flush"} connections waiting for the engine thread (handled at the top of the loop, never mid-request)
  std::atomic<long> tot_routed{0}, tot_hit{0}, tot_cpu{0}, tot_streamed{0};  // decode totals (stats op)
  long draft_routed = 0, draft_hit = 0, draft_cpu = 0, draft_dma_rows = 0, decode_dma_rows = 0;
  double draft_ms = 0;
  std::atomic<long> tot_mtp_steps{0}, tot_mtp_drafted{0}, tot_mtp_accepted{0}, tot_mtp_tokens{0};  // speculation totals: steps, submitted drafts, accepted, tokens produced by those steps
  std::map<std::string, std::shared_ptr<std::atomic<bool>>> cancel_flags;  // session -> cancel flag of its latest request (a cancel reaches only that request)
  // Request id (rid) -> that request's flag. Hitting only the session's latest flag let a late cancel for a dropped request kill the next
    //   request in the same session (reproduced on CPU hived: with a 100 ms cancel delay, B was cut at 8 tokens). weak — invalid by itself once
    //   the request ends; expired entries are swept on insert.
  std::map<std::string, std::weak_ptr<std::atomic<bool>>> rid_flags;
  std::map<std::string, Session> sessions;
  std::map<std::string, Session> archived;  // sessions evicted from the pool (no seq, image in RAM only) — LRU within the host_session_mb budget
  std::map<uint64_t, SharedSnap> shared_snaps;  // prefix hash -> boundary snapshot (shared by all sessions; reclaimed under the same budget in the same recency order)
  long shared_hits = 0, shared_saves = 0;
  size_t host_peak_bytes = 0;
  auto host_total = [&](const SeqImage* transient = nullptr) {
    std::set<const void*> seen;
    size_t n = rt.snapshot_pool_cached_bytes();
    for (const auto& [id, s] : sessions) n += s.host_bytes(seen);
    for (const auto& [id, s] : archived) n += s.host_bytes(seen);
    for (const auto& [key, e] : shared_snaps) n += sizeof(e) + e.sig.capacity() * sizeof(Session::ImgSig) + (e.image ? e.image->allocated_bytes(seen) : 0);
    if (transient) n += transient->allocated_bytes(seen);
    host_peak_bytes = std::max(host_peak_bytes, n);
    return n;
  };
  // Reclaim archived sessions (whole) and the reusable images of resident sessions (history frames -> checkpoint and last state) in **one
    //   recency order**. keep = the session being admitted/revived now — its state is never reclaimed (the budget is soft: if it cannot
    //   shrink, it stays over). Budget accounting (host_total) is unchanged.
  auto archive_trim = [&](const std::string& keep) {
    const size_t budget = (size_t)(std::max(0.0, host_session_mb) * 1048576.0);
    if(host_total()>budget) rt.trim_snapshot_pool();
    // Resident GPU state remains valid: reclaim optional host reuse, never reject
    // a conversation for exceeding the checkpoint budget.
    while (host_total() > budget) {
      auto arch = archived.end();
      Session* victim = nullptr;
      double oldest = INFINITY;
      for (auto it = archived.begin(); it != archived.end(); ++it)
        if (it->first != keep && it->second.last_used < oldest) { oldest = it->second.last_used; arch = it; }
      for (auto& [id, s] : sessions)
        if (id != keep && (s.ckpt || s.live || !s.history.empty()) && s.last_used < oldest) { oldest = s.last_used; victim = &s; arch = archived.end(); }
      auto sh = shared_snaps.end();  // shared boundary snapshots also join the same recency order (no effect when empty)
      for (auto it = shared_snaps.begin(); it != shared_snaps.end(); ++it)
        if (it->second.last_used < oldest) { oldest = it->second.last_used; sh = it; arch = archived.end(); victim = nullptr; }
      if (sh != shared_snaps.end()) { shared_snaps.erase(sh); rt.trim_snapshot_pool(); continue; }
      if (arch != archived.end()) { archived.erase(arch); rt.trim_snapshot_pool(); continue; }
      if (!victim) break;  // mandatory token history may exceed the soft budget
      if (!victim->history.empty()) victim->history.pop_front();
      else { victim->ckpt.reset(); victim->live.reset();
        std::vector<float>().swap(victim->ckpt_logits);
        std::vector<Session::ImgSig>().swap(victim->ckpt_sig);
        std::vector<Session::ImgSig>().swap(victim->live_sig);
      }
      rt.trim_snapshot_pool();
    }
  };
  // Session pool: reserve max_sessions at startup — no cudaMalloc while running, so peak VRAM = right after startup (the "free" in the log
    // below). Buffers of evicted sessions return to the pool.
  std::vector<std::unique_ptr<Seq>> seq_pool;
  size_t session_pool_bytes = 0;
  {
    size_t fr0 = 0, fr1 = 0, tot = 0;
    cudaMemGetInfo(&fr0, &tot);
    for (int i = 0; i < (start_asleep ? 0 : max_sessions); ++i) seq_pool.push_back(rt.new_seq());  // started asleep: allocated at the first wake
    cudaMemGetInfo(&fr1, &tot);
    session_pool_bytes = fr0 > fr1 ? fr0 - fr1 : 0;  // sleep VRAM table (session KV — kept while asleep)
    fprintf(stderr, "[hived] session pool %d × %.0f MiB reserved · VRAM free %.0f MiB (= operating headroom: the only runtime allocation is the image patch buffer)\n", max_sessions,
            max_sessions ? (double)(fr0 - fr1) / 1048576.0 / max_sessions : 0.0, fr1 / 1048576.0);
  }
  if (cache_fit && !start_asleep) {  // HIVE_CACHE_FIT: slot region (one contiguous block) from the headroom after the session pool -> warm start
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    const size_t rec = store.layout().total;
    const int got = store.alloc_cache(ExpertStore::fit_slots(fr, cache_reserve, rec));
    const size_t fr0 = fr;
    cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "[hived] cache fit: %d slots (free %.0f MiB − reserve %.0f MiB · --vram-cache-mb %.0f = %d slots ignored) · VRAM free %.0f MiB\n", got,
            fr0 / 1048576.0, cache_reserve / 1048576.0, cache_mb, (int)(cache_mb * 1048576.0 / (double)rec), fr / 1048576.0);
    warm_start();
  }
  if (shadow_prep.joinable()) shadow_prep.join();  // faster first sleep: become ready only after the shadow pre-pinning ends (so it never overlaps later sleeps or requests)
  {  // cold-load phase breakdown (without timestamps the ready time could not be split into phases)
    const double t_ready = now_ms();
    fprintf(stderr, "[hived] load phases: model+dense+mtp %.1fs · store init %.1fs · experts+engram+pin %.1fs · runtime %.1fs · release+pool+cache+warm %.1fs (load par %d · prefault %d)\n",
            (t_model - t_load) / 1000.0, (t_store - t_model) / 1000.0, (t_weights - t_store) / 1000.0, (t_runtime - t_weights) / 1000.0,
            (t_ready - t_runtime) / 1000.0, load_par, load_prefault);
  }
  // ready only after the session pool is reserved — launchers treat this line as startup complete (a pool OOM must not happen after ready)
  fprintf(stderr, "[hived] ready in %.1fs · layers %d · ctx %lld · chunk %d · dspark %s(block %d · conf ≥ %.2f · max %d)\n", (now_ms() - t_load) / 1000.0, nl,
          (long long)max_ctx, max_chunk, rt.mtp_enabled() ? "on" : "off", mtp_B, mtp_conf, mtp_max);

  auto take_seq = [&]() -> std::unique_ptr<Seq> {
    if (!seq_pool.empty()) { auto s = std::move(seq_pool.back()); seq_pool.pop_back(); rt.reset_seq(*s); return s; }
    fprintf(stderr, "[hived] session pool empty — allocating (VRAM peak moves)\n");
    return rt.new_seq();
  };

  // Sleep/wake state — changed only by the engine thread (power is also read by the receiver thread). 0 ready · 1 draining (sleep requested:
    //   new requests stay queued, in-flight requests run to completion) · 2 sleeping (VRAM released) · 3 waking (reclaiming and warming up).
    //   sleep_asked = a sleep request arrived (set by the receiver thread — so the extra intake points inside a prefill (batch pull, window,
    //   yield) stop even before the engine thread handles the request).
  enum { kPowerReady = 0, kPowerDraining = 1, kPowerSleeping = 2, kPowerWaking = 3 };
  std::atomic<int> power{kPowerReady};
  std::atomic<bool> sleep_asked{false};
  json sleep_info = json::object(), wake_info = json::object();  // last sleep/wake reports (stats "sleep", "wake")
  std::vector<int32_t> sleep_keys;  // resident keys (by score) at sleep time — reloaded in this order on wake
  int drain_running = 0;
  int sleep_level = 0;          // sleep depth: 1 = slots, staging, elastic span, vision · 2 = + session KV (session pool to RAM images — offload_sessions) · 3 = + VMM regions (devmem)
  int sessions_offloaded = 0;   // sessions moved to RAM images at level 2
  bool pool_released = false;   // level 2: the session pool's Seqs were freed (wake re-allocates as many as at startup)
  bool start_asleep_pending = false;  // started asleep, before the first wake (the slot region must be allocated for the first time)
  json vmm_info = json::object();  // level 3 (VMM release) report — released bytes, shadow copies, time, errors
  json vram_json = json::object(), host_mem_json = json::object();
  double vram_ms = -1e18;
  auto power_name = [](int p) { return p == kPowerDraining ? "draining" : p == kPowerSleeping ? "sleeping" : p == kPowerWaking ? "waking" : "ready"; };
  // Mutable engine state never crosses threads. Stats readers only copy this snapshot.
  std::mutex stats_mu;
  json stats_snapshot;
  double last_stats = 0;
  // force = at each request end and prefill chunk (publishing only at the top of the loop dropped the last second of a finished request for
    //   the 30 s idle cv wait, and froze during long prefills). Otherwise throttled to once per second.
  auto publish_stats = [&](bool force = false) {
    if (!force && now_ms() - last_stats < 1000) return;
    last_stats = now_ms();
    const auto cs = store.cache_stats();
    json s = {{"build_id", HIVE_BUILD_ID}, {"snapshot_ms", last_stats}, {"slots", store.n_slots()},
              {"resident", cs.resident}, {"unique_resident", cs.unique}, {"pending", cs.pending},
              {"duplicates", cs.duplicates}, {"invalid_mappings", cs.invalid_mappings},
              {"promotions", cs.promotions}, {"commits", cs.commits}, {"evictions", cs.evictions},
              {"duplicate_skips", cs.duplicate_skips}, {"h2d_records", cs.h2d_records}, {"d2d_records", cs.d2d_records},
              {"expert_record_bytes", store.layout().total}, {"sessions", sessions.size()}, {"archived", archived.size()},
              {"decode_routed", tot_routed.load()}, {"decode_hit", tot_hit.load()}, {"decode_cpu", tot_cpu.load()},
              {"decode_streamed", tot_streamed.load()}, {"expert_uses", store.total_uses()},
              {"draft", {{"routed", draft_routed}, {"hit", draft_hit}, {"cpu", draft_cpu}, {"dma_rows", draft_dma_rows}, {"ms", draft_ms}}},
              {"mtp", {{"steps", tot_mtp_steps.load()}, {"drafted", tot_mtp_drafted.load()}, {"accepted", tot_mtp_accepted.load()}, {"tokens", tot_mtp_tokens.load()}}}};
    s["host_session_bytes"] = host_total();
    s["host_session_peak_bytes"] = host_peak_bytes;
    s["snapshot_pool_cached_bytes"] = rt.snapshot_pool_cached_bytes();
    s["decode_dma_rows"] = decode_dma_rows;
    s["decode_streamed_records"] = tot_streamed.load();
    s["host_accounting_scope"] = "reuse state/capacities/pinned pool, approximate metadata; not process RSS";
    if (prefix_share) s["prefix_share"] = {{"entries", shared_snaps.size()}, {"hits", shared_hits}, {"saves", shared_saves}};
    s["state"] = power_name(power.load());  // sleep/wake state
    s["max_ctx"] = max_ctx;  // the server uses it to reject over-window requests with 400 before the stream starts
    if (power.load() != kPowerReady || last_stats - vram_ms >= 1000.0) {  // even on forced publishes (request end, every chunk) query the device at most once per second (every time during transitions)
      size_t fr = 0, tot = 0;
      cudaMemGetInfo(&fr, &tot);
      vram_json = {{"free_mib", fr / 1048576.0}, {"used_mib", (double)(tot - fr) / 1048576.0}, {"total_mib", tot / 1048576.0}};
      host_mem_json = proc_memory_mib();
      host_mem_json["experts_pinned_mib"] = store.experts_host_bytes() / 1048576.0;
      host_mem_json["engram_mib"] = store.engram_host_bytes() / 1048576.0;
      vram_ms = last_stats;
    }
    s["vram"] = vram_json;
    s["host_memory"] = host_mem_json;
    if (power.load() == kPowerDraining) s["draining_running"] = drain_running;
    if (!sleep_info.empty()) s["sleep"] = sleep_info;
    if (!wake_info.empty()) s["wake"] = wake_info;
    const auto pf = rt.prefetch_stats();
    s["prefetch"] = {{"issued_records", pf[0]}, {"used_records", pf[1]}, {"wasted_records", pf[2]},
                      {"issued_bytes", pf[0] * store.layout().total}, {"used_bytes", pf[1] * store.layout().total},
                      {"wasted_bytes", pf[2] * store.layout().total}};
    s["prefetch"]["completed_copy_ms_sampled"] = rt.prefetch_copy_ms();
    s["prefetch"]["timing_scope"] = "side_stream_overlap_not_additive; last/inflight sample may be absent";
    s["coverage"] = json::array();
    for (auto [n, value] : store.coverage({1000,2000,3000,4000,5000})) s["coverage"].push_back({{"slots",n},{"share",value}});
    std::lock_guard<std::mutex> lk(stats_mu);
    stats_snapshot = std::move(s);
  };
  if (start_asleep) {  // started asleep, finishing up: drop everything placed via VMM so far (dense weights, Work, cuBLAS, staging) to host copies (level 3)
    if (cache_fit) warm_start();  // reads the list only (with HIVE_CACHE_FIT the slot block above was skipped, so it has not been read yet)
    size_t fr0 = 0, fr1 = 0, tot = 0;
    cudaMemGetInfo(&fr0, &tot);
    CUDA_CHECK(cudaDeviceSynchronize());
    const devmem::Report r = devmem::release_all();
    cudaMemGetInfo(&fr1, &tot);
    sleep_level = r.ok ? 3 : 2;  // if it cannot be dropped, stay asleep at level 2 (no slots or session pool — dense weights stay in VRAM)
    vmm_info = {{"released_mib", r.bytes / 1048576.0}, {"copied_mib", r.copied / 1048576.0}, {"regions", r.regions}, {"ms", r.ms}};
    if (!r.ok) vmm_info["error"] = r.error;
    pool_released = true;
    start_asleep_pending = true;
    sleep_keys = start_keys;
    power.store(kPowerSleeping);
    sleep_asked.store(true);
    json pm = proc_memory_mib();
    pm["experts_pinned_mib"] = store.experts_host_bytes() / 1048576.0;
    pm["engram_mib"] = store.engram_host_bytes() / 1048576.0;
    pm["dense_host_copy_mib"] = devmem::shadow_bytes() / 1048576.0;
    sleep_info = {{"level", sleep_level}, {"started_asleep", true}, {"ms", r.ms}, {"resident_keys", sleep_keys.size()}, {"slots_released", 0},
                  {"sessions_offloaded", 0}, {"freed_mib", fr1 > fr0 ? (fr1 - fr0) / 1048576.0 : 0.0}, {"vram_used_mib", (double)(tot - fr1) / 1048576.0},
                  {"vram_scope", "device-wide cudaMemGetInfo — exact for hived only while no other process holds VRAM (per-process: nvidia-smi --query-compute-apps)"},
                  {"vmm", vmm_info}, {"host_memory", pm}};
    fprintf(stderr, "[hived] started asleep (level %d): dense/work %.0f MiB to RAM in %.0f ms · VRAM used %.0f MiB · host RSS %.0f MiB (experts pinned %.0f · engram %.0f · "
            "dense copy %.0f MiB) · first wake allocates the cache and warms %zu experts%s\n", sleep_level, r.bytes / 1048576.0, r.ms, (double)(tot - fr1) / 1048576.0,
            pm.value("VmRSS_mib", 0.0), pm.value("experts_pinned_mib", 0.0), pm.value("engram_mib", 0.0), pm.value("dense_host_copy_mib", 0.0), sleep_keys.size(),
            r.ok ? "" : (" · ⚠️release failed: " + r.error).c_str());
  }
  publish_stats();

  SocketWriter writer;
  output = &writer;
  // Graceful shutdown state: 0 serving · 1 draining (new generate refused, in-flight requests continue) · 2 deadline passed (cancel all
    // in-flight -> shutdown error) · 3 receiver thread finished
  std::atomic<int> stop_state{0};
  int stop_pipe[2] = {-1, -1};
  if (graceful_stop_s > 0) {
    HIVE_CHECK(::pipe2(stop_pipe, O_CLOEXEC | O_NONBLOCK) == 0, "pipe2");
    g_stop_pipe_w = stop_pipe[1];
    struct sigaction sa{};
    sa.sa_handler = on_stop_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    fprintf(stderr, "[hived] graceful stop on: SIGTERM/SIGINT drain up to %.1f s, then cancel (second signal = immediate)\n", graceful_stop_s);
  }
  // Multiplex partial headers/images: a slow upload cannot block cancel/stats.
  std::thread acceptor([&] {
    ::fcntl(lfd, F_SETFL, ::fcntl(lfd, F_GETFL) | O_NONBLOCK);  // non-blocking so accept can loop until EAGAIN (accepted fds do not inherit it — recv/send use MSG_DONTWAIT)
    struct Incoming { std::string bytes; json req; size_t bin=0; bool header=false; };
    std::map<int, Incoming> incoming;
    const size_t nfixed = stop_pipe[0] >= 0 ? 2 : 1;  // [lfd, (graceful-stop pipe)] then the accepted connections
    double stop_deadline = 0;
    for (;;) {
      std::vector<pollfd> fds{{lfd, POLLIN, 0}};
      if (nfixed == 2) fds.push_back({stop_pipe[0], POLLIN, 0});
      for (const auto& [fd, in] : incoming) fds.push_back({fd, POLLIN, 0});
      int timeout = -1;
      if (stop_state.load() == 1) timeout = std::max(0, (int)std::ceil(stop_deadline - now_ms()));
      const int pr = ::poll(fds.data(), fds.size(), timeout);
      if (nfixed == 2) {
        if (pr > 0 && (fds[1].revents & POLLIN)) {
          char b[64];
          while (::read(stop_pipe[0], b, sizeof b) > 0) {}
          if (stop_state.load() == 0) {
            stop_deadline = now_ms() + graceful_stop_s * 1000.0;
            fprintf(stderr, "[hived] graceful stop: signal received — refusing new requests, draining up to %.1f s\n", graceful_stop_s);
            { std::lock_guard<std::mutex> lk(mu); stop_state.store(1); }
            cv.notify_all();
          }
        }
        if (stop_state.load() == 1 && now_ms() >= stop_deadline) {
          fprintf(stderr, "[hived] graceful stop: drain deadline reached — cancelling in-flight requests\n");
          { std::lock_guard<std::mutex> lk(mu); stop_state.store(2); }
          cv.notify_all();
        }
        if (stop_state.load() == 3) {  // the engine is done: close the remaining half-open connections and leave
          for (const auto& [fd, in] : incoming) finish_socket(fd);
          return;
        }
      }
      if (pr < 0) continue;
      if (fds[0].revents & POLLIN) {  // drain the accept queue on every wakeup (one accept per poll let the backlog fill under bursts). lfd is non-blocking
        for (;;) {
          const int fd = ::accept(lfd, nullptr, nullptr);
          if (fd < 0) { if (errno == EINTR) continue; break; }  // EAGAIN = all accepted · anything else (EMFILE etc.) retried at the next poll
          writer.add(fd); incoming.emplace(fd, Incoming{});
        }
      }
      char buf[65536];
      for (size_t fi=nfixed; fi<fds.size(); ++fi) {
        if (!fds[fi].revents) continue;
        const int fd=fds[fi].fd;
        auto& in=incoming.at(fd);
        const ssize_t n=::recv(fd, buf, sizeof(buf), MSG_DONTWAIT);
        if (n<0 && (errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR)) continue;
        if (n<=0) { finish_socket(fd); incoming.erase(fd); continue; }
        in.bytes.append(buf, (size_t)n);
        if (!in.header) {
          const size_t end=in.bytes.find('\n');
          // A header line that never ends (or a declared binary payload beyond what the vision budget can use) must not grow the buffer
          //   without bound: 64 MiB covers a full-context id list; the patch payload is at most vision_max_patches patches of 3·p·p bf16.
          if (end==std::string::npos) { if (in.bytes.size() > kMaxHeaderBytes) { send_json(fd, {{"error","request too large"}}); finish_socket(fd); incoming.erase(fd); } continue; }
          if (end > kMaxHeaderBytes) { send_json(fd, {{"error","request too large"}}); finish_socket(fd); incoming.erase(fd); continue; }
          try { in.req=json::parse(in.bytes.substr(0,end)); }
          catch (const json::exception&) { send_json(fd, {{"error","bad json"}}); finish_socket(fd); incoming.erase(fd); continue; }
          // Optional envelope fields are normalized instead of terminating the
          // acceptor via json::type_error. Model payload errors remain per-request.
          if (!in.req.is_object()) in.req=json::object();
          for (auto key : {"op", "session", "rid"}) if (in.req.contains(key) && !in.req[key].is_string()) in.req.erase(key);
          in.bin = in.req.contains("bin") && in.req["bin"].is_number_unsigned() ? in.req["bin"].get<size_t>() : 0;
          if (in.bin > kMaxBinBytes) { send_json(fd, {{"error","request too large"}}); finish_socket(fd); incoming.erase(fd); continue; }
          in.bytes.erase(0,end+1); in.header=true;
        }
        json& j=in.req;
        std::string op=j.value("op", "generate");
      if (op == "cancel") {
        std::lock_guard<std::mutex> lk(mu);
        const std::string rid = j.value("rid", "");
        if (!rid.empty()) {  // rid: only that request — nothing is hit if it already finished
          auto it = rid_flags.find(rid);
          if (it != rid_flags.end()) if (auto f = it->second.lock()) f->store(true);
        } else {
          auto it = cancel_flags.find(j.value("session", ""));
          if (it != cancel_flags.end() && it->second) it->second->store(true);
        }
        send_json(fd, {{"ok", true}});
        finish_socket(fd); incoming.erase(fd);
        continue;
      }
      if (op == "stats") {
        json s;
        { std::lock_guard<std::mutex> lk(stats_mu); s = stats_snapshot; }
        { std::lock_guard<std::mutex> lk2(mu); s["pending_requests"] = queue.size(); }
        send_json(fd, s);
        finish_socket(fd); incoming.erase(fd);
        continue;
      }
      if (op == "flush") {
        { std::lock_guard<std::mutex> lk(mu); flush_q.push_back(fd); }
        incoming.erase(fd);
        cv.notify_all();
        continue;
      }
      if (stop_state.load() != 0) {  // draining for graceful shutdown: no new generate (cancel and stats are still accepted)
        send_json(fd, {{"error", "daemon shutting down"}});
        finish_socket(fd); incoming.erase(fd);
        continue;
      }
      // Sleep/wake: {"op":"sleep"[, "timeout_s":N]} · {"op":"wake"} — handled by the engine thread, which answers one line on this connection
            //   (the receiver thread never blocks). Sleep releases VRAM after in-flight requests finish (it does not cut them off). If timeout_s
            //   (optional, only numbers > 0 — anything else normalizes to absent) passes first, it answers {"error":"not idle"} and withdraws the
            //   sleep (the same shape as SGLang's release refusal). Response format: handle_ctl below.
      if (op == "sleep" || op == "wake") {
        double timeout_s = 0;
        if (j.contains("timeout_s") && j["timeout_s"].is_number()) { const double v = j["timeout_s"].get<double>(); if (std::isfinite(v) && v > 0) timeout_s = std::min(v, 1e7); }
        // level (optional): 2 = session KV to RAM as well (releases the session pool's VRAM, for long sleeps) · anything else (absent, 1,
                //   non-numeric) = 1 (session KV stays in VRAM).
                //   Level 3 = + VMM regions (dense weights, small Work slabs, cuBLAS) -> pinned host copies (same VA reservation — only when started
                //   with HIVE_SLEEP_VMM, otherwise absorbed into 2)
        const double lv = j.contains("level") && j["level"].is_number() ? j["level"].get<double>() : 1.0;
        const int level = lv >= 3 ? 3 : lv >= 2 ? 2 : 1;
        {
          std::lock_guard<std::mutex> lk(mu);
          ctl_q.push_back({fd, op == "sleep", timeout_s > 0 ? now_ms() + timeout_s * 1000.0 : 0.0, level});
          if (op == "sleep") sleep_asked.store(true);
        }
        incoming.erase(fd);
        cv.notify_all();
        continue;
      }
      if (in.bytes.size() < in.bin) continue;
      j["session"]=j.value("session","default");
      Request r;
      r.fd = fd;
      r.req = j;
      r.bin.assign(in.bytes.begin(), in.bytes.begin()+in.bin);
      {
        std::lock_guard<std::mutex> lk(mu);
        r.cancel = std::make_shared<std::atomic<bool>>(false);
        cancel_flags[j.value("session", "")] = r.cancel;
        if (const std::string rid = j.value("rid", ""); !rid.empty()) {
          for (auto it = rid_flags.begin(); it != rid_flags.end();) it = it->second.expired() ? rid_flags.erase(it) : std::next(it);
          rid_flags[rid] = r.cancel;
        }
        r.t_recv = now_ms();
        queue.push_back(std::move(r));
      }
      incoming.erase(fd);
      cv.notify_one();
      }
    }
  });

  // ---- processing loop (engine thread) — scheduler ----
  // Each new request is prefilled on its own (in chunks; running decoders pause briefly) and then joins active; all of active is decoded
    // together one step at a time with forward_batch (row = request, <= max_batch). Sampling, stopping and cancellation are per row. Active
    // sessions are excluded from LRU eviction.
  struct Active {
    Request r;
    std::string sid;
    Session* S = nullptr;
    std::atomic<bool>* cancel = nullptr;  // = r.cancel.get() (the Request owns its lifetime)
    std::vector<float> logits;
    std::mt19937_64 rng;
    float temperature = 1.f, top_p = 1.f, min_p = 0.f;
    int top_k = 0, max_tokens = 256, n = 0;
    int32_t pending_tok = -1;   // next token already drawn by speculative verification (correction sample at the rejected position, or the bonus) — emitted instead of sampling
    int mtp_skip = 0;           // remaining steps to skip speculation for lack of expected gain (retried every 8 steps)
    mtpg::Backoff g2;           // HIVE_MTP_GATE2: per-request backoff (net-gain EMA) — used only when enabled
    Cands cands;                // GPU sampler candidates of the last forward (row snapshot)
    // Thinking cap (only when the request's think_cap > 0; default 0 = off): once the tokens emitted inside the thinking span (in_think) reach
        //   think_cap, the next token is forced to think_end (</think>). think_n = tokens emitted in the thinking span (excluding </think>) ·
        //   think_forced = number of times it was forced.
    int think_cap = 0, think_n = 0, think_forced = 0;
    int prio = 0;            // the request's "priority" (lower runs first, default 0) — decode_step steps only the best priority present
    long held_steps = 0;     // decode steps this request sat out for a better-priority request
    int32_t think_end = -1, think_start = -1;
    // Exit phrase (think_exit_ids, supplied by the server): when the cap is reached, force this token sequence one token at a time before </think>.
        //   Measured with vLLM: forcing only </think> without an exit phrase let 3.7% (6/162) continue reasoning in the answer (leaked); with a
        //   transition phrase, 0/162. Empty = force </think> immediately.
    std::vector<int32_t> think_exit;
    size_t think_exit_pos = 0;
    bool in_think = false;
    long mtp_steps = 0, mtp_drafted = 0, mtp_accepted = 0;
    std::vector<int32_t> stop_ids;
    std::string finish = "length";
    bool client_ok = true;
    bool defer_ckpt = false, defer_warm = false;  // HIVE_DEFER_CKPT: prompt checkpoint and warm_after_prefill deferred until after the first token is sent
    double t0 = 0, t1 = 0;
    size_t common = 0, prompt_len = 0;
    ForwardStats ds{};
  };
  std::vector<std::unique_ptr<Active>> active;
  active.reserve(rt.max_batch()); // no allocation between active_sids insert and publishing Active
  std::set<std::string> active_sids;
  auto fake_logits = [&](std::vector<float>& lg, uint64_t salt) {
    if (!fake_head) return;
    lg.assign(c.vocab, 0.f);
    uint64_t x = 0x9E3779B97F4A7C15ull ^ salt;
    for (int i = 0; i < 64; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; lg[(size_t)(x % 20000) + 300] = 5.f + (float)(i % 7); }
  };
  // Prompt-end checkpoint — shared by the immediate path in admit and the deferred (HIVE_DEFER_CKPT) path
  auto save_prompt_ckpt = [&](Active& X, const std::vector<float>& logits) {
    Session* S = X.S;
    auto img = std::make_shared<SeqImage>();
    rt.save_image(*S->seq, *img, S->ckpt.get());
    host_total(img.get());  // replacement peak, shared segments counted once
    static const int history_limit = getenv("HIVE_CKPT_HISTORY") ? std::max(1, atoi(getenv("HIVE_CKPT_HISTORY"))) : 1;
    if (S->ckpt && history_limit > 1)
      S->history.push_back({S->ckpt, std::move(S->ckpt_logits), std::move(S->ckpt_sig)});
    while ((int)S->history.size() >= history_limit) S->history.pop_front();
    S->ckpt = std::move(img);
    S->ckpt_logits = logits;
    S->ckpt_sig = S->img_sig;
    host_total();
    archive_trim(X.sid);
  };
  auto warm_after_prefill = [&](size_t self_active) {  // self_active = 1 when the warming request is already counted in active (deferred warm)
    // Inside a layer yield (a long prefill is paused between layers) do not warm: a synchronous upload (~1.3 s per request, measured) would turn
    //   directly into a stall of the long prefill, and a deferred quota would only be consumed by the yield steps while that prefill runs.
    //   Warming only fills the cache (lossless) — step promotion fills it afterwards.
    if (rt.in_layer_yield()) return;
    if (warm_defer > 0) {  // HIVE_WARM_DEFER: no upload here — the following decode steps promote up to warm_defer experts each until warm_cap is spent
      rt.warm_defer(warm_defer);
      fprintf(stderr, "[hived] warm cache: deferred, <= %d experts per step (%zu other active)\n", warm_defer, active.size() - self_active);
      return;
    }
    const bool busy = warm_busy_cap > 0 && active.size() > self_active;  // other requests are decoding: they would stall for the whole upload
    const int full_cap = rt.warm_cap();
    if (busy) rt.set_warm_cap(std::min(full_cap, warm_busy_cap));
    const double tw = now_ms();
    int nw = rt.warm_cache();
    if (busy) rt.set_warm_cap(full_cap);
    if (busy) fprintf(stderr, "[hived] warm cache: busy cap %d -> %d experts in %.0f ms (%zu other active)\n", warm_busy_cap, nw, now_ms() - tw, active.size() - self_active);
    else if (nw) fprintf(stderr, "[hived] warm cache: %d experts in %.0f ms\n", nw, now_ms() - tw);
  };
  // HIVE_DEFER_CKPT: the deferred checkpoint is taken right after the first token is sent and before that session's forward — at that point
    //   seq is exactly at the prompt end (pos == prompt length) and X.logits are still the prefill logits (the sampler only reads them). If pos
    //   differs (an impossible order) it is dropped rather than capturing a wrong state. The state is only for reuse, so host exceptions are
    //   absorbed (they must not kill the request or other users).
  auto run_deferred = [&](Active& X) {
    if (X.defer_ckpt) {
      X.defer_ckpt = false;
      if (X.S && X.S->seq && !X.S->seq->broken && X.S->seq->pos == (int64_t)X.prompt_len) {
        try { save_prompt_ckpt(X, X.logits); }
        catch (const std::exception& e) { fprintf(stderr, "[hived] %s: deferred prompt checkpoint dropped: %s\n", X.sid.c_str(), e.what()); }
      } else fprintf(stderr, "[hived] %s: deferred prompt checkpoint dropped (sequence moved)\n", X.sid.c_str());
    }
    if (X.defer_warm) { X.defer_warm = false; warm_after_prefill(1); }
  };
  auto finish_active = [&](Active& A) {
    if (A.defer_ckpt || A.defer_warm) run_deferred(A);  // HIVE_DEFER_CKPT: paths that finish before the first step (e.g. cleaning up a busy session) still keep the checkpoint
    const double t2 = now_ms();
    json done = {{"done", true}, {"n", A.n}, {"finish", A.finish}, {"prefill_ms", A.t1 - A.t0}, {"decode_ms", t2 - A.t1},
                 {"cached_prefix", (int)A.common}, {"prefill_tokens", (int)(A.prompt_len - A.common)},
                 {"decode_hit", A.ds.n_hit}, {"decode_cpu", A.ds.n_cpu}, {"cpu_wait_ms", A.ds.ms_cpu_wait}, {"batch_rows", (int)active.size()}, {"held_steps", A.held_steps},
                 {"decode_dma_rows", A.ds.n_dma_rows}, {"cpu_span_ms", A.ds.ms_cpu_span}, {"timing_scope", "shared_batch_spans_not_additive"},
                 {"mtp_steps", A.mtp_steps}, {"mtp_drafted", A.mtp_drafted}, {"mtp_accepted", A.mtp_accepted}};
    if (A.think_cap > 0) {  // only for requests that received a thinking cap (off = done message shape unchanged)
      done["think_cap"] = A.think_cap;
      done["think_tokens"] = A.think_n;
      done["think_forced"] = A.think_forced > 0;
    }
    publish_stats(true);  // done — publish before sending, so a client asking for stats right after done already sees this request
    // a request cancelled because the graceful-stop deadline passed: report a shutdown error so a truncated answer does not look like a normal finish
        // (the server turns the error into 502 / a stream error)
    if (stop_state.load() >= 2 && A.finish == "cancel") send_json(A.r.fd, {{"error", "daemon shutting down: generation cancelled"}});
    else send_json(A.r.fd, done);
    finish_socket(A.r.fd);
    // Logging: the request line ends with " · wall t0= t1= t2= queue_ms= rid=" — t0 admission (prefill start), t1 prefill end, t2 finish, in
        //   epoch ms (wall_off taken at startup = system clock - now_ms) · queue_ms = time queued by the receiver thread -> t0 · "-" without rid
    const std::string rid_s = A.r.req.value("rid", "");
    fprintf(stderr, "[hived] %s: prefill %zu tok %.0f ms · decode %d tok %.0f ms (%.1f tok/s, batch %zu) · hit %d cpu %d · resident %d/%d · mtp %ld/%ld in %ld steps · %s"
            " · wall t0=%.0f t1=%.0f t2=%.0f queue_ms=%.0f rid=%s\n",
            A.sid.c_str(), A.prompt_len - A.common, A.t1 - A.t0, A.n, t2 - A.t1, A.n > 1 ? (A.n - 1) * 1000.0 / (t2 - A.t1) : 0.0, active.size(),
            A.ds.n_hit, A.ds.n_cpu, store.n_resident(), store.n_slots(), A.mtp_accepted, A.mtp_drafted, A.mtp_steps, A.finish.c_str(),
            A.t0 + wall_off, A.t1 + wall_off, t2 + wall_off, A.r.t_recv > 0 ? std::max(0.0, A.t0 - A.r.t_recv) : 0.0, rid_s.empty() ? "-" : rid_s.c_str());
    active_sids.erase(A.sid);
    {  // if this was the session's last request, remove its flag entry too (otherwise the map grew per session)
      std::lock_guard<std::mutex> lk(mu);
      auto it = cancel_flags.find(A.sid);
      if (it != cancel_flags.end() && it->second.get() == A.cancel) cancel_flags.erase(it);
    }
  };
  // Admit one request: prefill it and put it in active (on failure or cancel: respond and drop it)
  std::function<void()> decode_step;  // defined below (declared first because it is called between prefill chunks)
  long prefill_epoch = 0;  // [step-host]: bumped before every prefill forward and at every layer-yield run — a decode-step gap that saw a change is not counted
  double last_prefill_tc = 0;  // HIVE_PREFILL_FAIR: wall clock of this admission's (admit or admit_batch) last prefill forward (a round for batches) — read only by HIVE_PREFILL_FAIR
  // values shared by the admit stages
  static const bool image_ckpt = getenv("HIVE_IMAGE_CKPT") && *getenv("HIVE_IMAGE_CKPT") && strcmp(getenv("HIVE_IMAGE_CKPT"), "0") != 0;  // env_on convention (unset/""/"0" = off — atoi would read "true" as off)
  const size_t stream_floor = (size_t)std::max(1, rt.opt().prefill_threshold);  // lower bound for budget splitting = the streaming path (default 1024)
  const size_t yield_max = !prefill_yield ? 0 : prefill_yield_max_env ? prefill_yield_max_env : stream_floor - 1;  // HIVE_PREFILL_YIELD: prompt id cap for requests admitted by yielding
  if (prefill_yield) fprintf(stderr, "[hived] prefill yield on: requests with <= %zu rows to prefill (ids minus the reusable prefix) are admitted at chunk/round boundaries of a longer prefill\n", yield_max);
  const size_t ly_max = !layer_yield ? 0 : ly_max_env ? ly_max_env : stream_floor - 1;  // HIVE_LAYER_YIELD: prompt id cap for requests admitted by layer yielding (further limited by R)
  if (layer_yield)
    fprintf(stderr, "[hived] layer yield on: every %.0f ms at layer boundaries of a long prefill · decode share %.2f · <= %d steps · admits requests with <= %zu rows to prefill (ids minus the reusable prefix)%s\n",
            ly_period, ly_share, ly_steps, ly_max,
            ly_mid > ly_max ? (" · up to " + std::to_string(ly_mid) + " rows when <= half of the paused forward's remaining rows (HIVE_LAYER_YIELD_MID)").c_str() : "");
  // HIVE_LAYER_YIELD_MID: the forward a layer yield pauses — its rows, the rows of the request's later chunks, and the rows admitted by the mid rule
  //   during it (set by the outer admission right before rt.forward / rt.forward_multi; never from inside a yield)
  size_t ly_fwd_rows = 0, ly_rest_rows = 0, ly_mid_used = 0;
  // Rows a text request will actually prefill = its ids minus the prefix the daemon can resume from. This is a lower bound of what admit
  //   chooses (the session's live state, prompt checkpoint, history frames or archived last state, with the same token and image checks;
  //   shared boundary snapshots are not counted, so the real reuse can only be larger and the forward only shorter). Used by the yield
  //   admissions below so that follow-up turns of long conversations get in while another prompt is being prefilled: their prompt ids are
  //   many but only the new tokens are prefilled. Measured (interactive traffic on the reference machine): of the requests that waited >= 1 s behind a long
  //   prefill (2.5 % of all, median 2.4 s, up to 12.5 s), 37 of 82 were such turns (< 1024 new tokens on a >= 1024-id prompt).
  //   Cost: one token comparison over the matching prefix (json ids) per queued text request per yield point.
  auto reuse_lower_bound = [&](const std::string& sid, const json& q) -> size_t {
    const auto ids_it = q.find("ids");
    if (ids_it == q.end() || !ids_it->is_array() || ids_it->empty()) return 0;
    const json& ids = *ids_it;
    const Session* S = nullptr;
    if (auto it = sessions.find(sid); it != sessions.end()) S = &it->second;
    else if (auto ar = archived.find(sid); ar != archived.end()) S = &ar->second;
    if (!S) return 0;
    auto lcp = [&](const std::vector<int32_t>& a) {
      size_t k = 0; const size_t n = std::min(a.size(), ids.size());
      while (k < n && ids[k].is_number_integer() && (int64_t)a[k] == ids[k].get<int64_t>()) ++k;
      return k;
    };
    auto clear_upto = [&](const std::vector<Session::ImgSig>& sig, size_t pos) {  // text request: no saved image may end at or straddle pos
      for (const auto& s : sig) if ((size_t)s.end <= pos || ((size_t)s.start < pos && (size_t)s.end > pos)) return false;
      return true;
    };
    size_t best = 0;
    if (S->seq && !S->seq->broken && S->seq->pos == (int64_t)S->tokens.size()) {
      const size_t k = lcp(S->tokens);
      if (k == S->tokens.size() && ids.size() > k && clear_upto(S->img_sig, k)) best = k;
    }
    if (S->ckpt) { const size_t k = lcp(S->ckpt->tokens); if (k == (size_t)S->ckpt->pos && ids.size() > k && clear_upto(S->ckpt_sig, k)) best = std::max(best, k); }
    for (const auto& f : S->history)
      if (f.image) { const size_t k = lcp(f.image->tokens); if (k == (size_t)f.image->pos && ids.size() > k && clear_upto(f.sig, k)) best = std::max(best, k); }
    if (S->live) { const size_t k = lcp(S->live->tokens); if (k == (size_t)S->live->pos && ids.size() > k && clear_upto(S->live_sig, k)) best = std::max(best, k); }
    return best;
  };
  auto rows_to_prefill = [&](const std::string& sid, const json& q) -> size_t {
    const auto ids_it = q.find("ids");
    const size_t rows = ids_it != q.end() && ids_it->is_array() ? ids_it->size() : 0;
    return rows ? rows - std::min(rows, reuse_lower_bound(sid, q)) : 0;
  };
  std::set<std::string> prefilling_sids;  // sessions in a batched prefill (not active yet) — excluded from LRU eviction
  // HIVE_LAYER_YIELD: prompt id count of sessions being prefilled (the "sid:rows" layer-yield header line — lets a monitor tell >= 16K prefill
    // gaps apart). Only when enabled; entries no longer prefilling are swept on insert.
  std::map<std::string, size_t> ly_rows;
  auto ly_mark = [&](const std::string& sid, size_t rows) {
    if (ly_period <= 0) return;
    for (auto it = ly_rows.begin(); it != ly_rows.end();) it = prefilling_sids.count(it->first) ? std::next(it) : ly_rows.erase(it);
    ly_rows[sid] = rows;
  };
  int prefill_pending_n = 0;               // the multi-chunk marker means "at least one request is prefilling" — 0/1 on the single path
  // Tell the runtime while a multi-chunk prefill runs (before the first chunk until after the last). RAII — cleared on error, cancel, exception
    //   and early return alike. The effect (pausing promotion in decode steps between chunks) is decided by the runtime's
    //   HIVE_PREFILL_PAUSE_PROMOTE (when off only the marker changes).
  struct PrefillPending {
    Runtime& rt; const std::string& sid; int& n; bool on = false;
    void set() { if (!on) { ++n; rt.set_prefill_pending(true); on = true; fprintf(stderr, "[hived] %s: prefill_pending=%d (multi-chunk)\n", sid.c_str(), (int)rt.prefill_pending()); } }
    void clear() { if (on) { --n; rt.set_prefill_pending(n > 0); on = false; fprintf(stderr, "[hived] %s: prefill_pending=%d\n", sid.c_str(), (int)rt.prefill_pending()); } }
    ~PrefillPending() { clear(); }
  };
  // Prefill state of one request — lives across stages (on the heap so references do not move)
  struct PJob {
    std::unique_ptr<Active> A;
    std::vector<int32_t> ids;
    Session* S = nullptr;
    std::vector<Session::ImgSig> request_sig, req_sig;
    std::vector<ImageInput> all_images;
    std::vector<DevBuf> patch_bufs;
    size_t common = 0, i = 0, chunk_cap = 0, busy_cap = 0;
    int pick = 0, extra_left = 0, n_chunks = 0;
    bool has_images = false, last_upper_skipped = false;
    std::vector<size_t> hints;
    std::shared_ptr<SeqImage> last_boundary;
    ForwardStats st{};
    std::unique_ptr<PrefillPending> pending;
    // current chunk (chunk_begin -> chunk_end)
    int M = 0; bool upper_needed = true;
    std::vector<ImageInput> chunk_images;
    int hit0 = 0, str0 = 0, cpu0 = 0, dma0 = 0; double wait0 = 0, span0 = 0;
    long ly_n = 0; double ly_last = 0;  // HIVE_LAYER_YIELD: layer yields inside this forward · last resume -> forward end (ms) — tail of the chunk log line (monitor gaps)
  };
  auto plan_chunk_j = [&](PJob& J, size_t at, size_t cap) -> size_t {
    auto& ids = J.ids; auto& all_images = J.all_images;
    const size_t rem = ids.size() - at;
    if (rem <= cap) return rem;
    // HIVE_TAIL_SPLIT: one more cut would leave the last piece below the tail-mode minimum (tail_need) -> pull the boundary in so the last is
        // tail_need. Only if this chunk is also in tail mode (never create a short-path chunk).
    if (tail_split && tail_need > 0 && all_images.empty() && rem - cap < (size_t)tail_need && rt.tail_mode_for((int)(rem - (size_t)tail_need)))
      return rem - (size_t)tail_need;  // < cap (rem - cap < tail_need)
    return cap;
  };
  // this chunk's cap (busy cap / budget when decoders exist) — the loop and the boundary-snapshot restore check (C1) use the same formula
  auto cap_now_j = [&](PJob& J) -> size_t {
    auto& all_images = J.all_images; const size_t chunk_cap = J.chunk_cap, busy_cap = J.busy_cap;
    size_t chunk_now = active.empty() ? chunk_cap : busy_cap;
    if (!active.empty() && prefill_budget_ms>0 && all_images.empty()) {
      const size_t predicted = prefill_ms_per_token>0 ? (size_t)std::min<double>(chunk_now,prefill_budget_ms/prefill_ms_per_token) : stream_floor;
      chunk_now=std::min(chunk_now,std::max<size_t>(stream_floor,predicted));
    }
    return chunk_now;
  };
  auto sig_upto_j = [&](PJob& J, size_t h) { std::vector<Session::ImgSig> v; for (const auto& r : J.req_sig) if ((size_t)r.end <= h) v.push_back(r); return v; };
  // C1 lower-bound cap for the batched path — the formula used when decoders exist (busy; stream_floor with a budget). In batches, requests
    //   whose prefill finished join active between rounds (so "active only shrinks inside the prefill loop" no longer holds) -> always use this
    //   lower bound when sizing the next chunk.
  auto c1_floor_j = [&](PJob& J) -> size_t { size_t n = J.busy_cap; if (prefill_budget_ms>0 && J.all_images.empty()) n = std::min(n, stream_floor); return n; };
  auto shared_has_j = [&](PJob& J, size_t h) {
    auto& ids = J.ids;
    auto sig_upto = [&](size_t hh) { return sig_upto_j(J, hh); };
    const auto sg = sig_upto(h);
    auto it = shared_snaps.find(prefix_key(ids.data(), h, sg));
    return it != shared_snaps.end() && it->second.image && (size_t)it->second.image->pos == h &&
           std::equal(it->second.image->tokens.begin(), it->second.image->tokens.end(), ids.begin());
  };
  auto hint_ok_j = [&](PJob& J, size_t h) {  // a boundary that may be cut: not inside an image span and no shared snapshot yet
    auto& all_images = J.all_images;
    auto shared_has = [&](size_t hh) { return shared_has_j(J, hh); };
    for (const auto& x : all_images) if ((size_t)x.start < h && (size_t)x.start + x.types.size() > h) return false;
    return !shared_has(h);
  };
  // Boundary cutting: cut at a boundary inside this chunk [i, i+M). (a) free — the deepest boundary that does not increase cost in the
    //   comparison below, (b) extra chunk — if the deepest remaining boundary is inside this chunk, up to extra_left times per request. C1: if
    //   the previous chunk skipped the tail layers (or was restored from such a snapshot), this chunk must be in tail mode — cut only at sizes
    //   that are in tail mode.
    //   "Free" is decided by expanding the remaining plan (same cap, plan_chunk) to the end: only if the chunk count and the number of
    //   streaming chunks (>= prefill_threshold, each sends all non-resident experts) do not grow and the number of C1 skips does not shrink —
    //   even with the same chunk count, a short tail growing into a streaming chunk or the previous chunk losing C1 increases transfers.
    //   (The actual cap can change with whether decoders exist, so this is an estimate — if wrong, the result is the same, only the cost differs.)
  auto plan_cost_j = [&](PJob& J, size_t first, size_t from, size_t cap) {
    auto& ids = J.ids; auto& all_images = J.all_images;
    auto plan_chunk = [&](size_t at, size_t c) { return plan_chunk_j(J, at, c); };
    std::vector<size_t> p{first};
    for (size_t x = from; x < ids.size();) { const size_t m = plan_chunk(x, cap); p.push_back(m); x += m; }
    std::array<size_t, 3> r{p.size(), 0, 0};  // chunks · streaming chunks · C1 skips
    for (size_t j = 0; j < p.size(); ++j) {
      if (p[j] >= stream_floor) ++r[1];
      if (j + 1 < p.size() && all_images.empty() && rt.tail_mode_for((int)p[j]) && rt.tail_mode_for((int)p[j + 1])) ++r[2];
    }
    return r;
  };
  auto hint_cut_j = [&](PJob& J, size_t i, int M, size_t cap) -> int {
    auto& hints = J.hints; const bool last_upper_skipped = J.last_upper_skipped; int& extra_left = J.extra_left; auto& ids = J.ids;
    auto hint_ok = [&](size_t h) { return hint_ok_j(J, h); };
    auto plan_cost = [&](size_t a, size_t b, size_t c) { return plan_cost_j(J, a, b, c); };
    if (hints.empty()) return M;
    size_t deepest = 0;
    for (size_t h : hints) if (h > i && hint_ok(h) && ids.size() - h > 16) deepest = h;  // the deepest conversation boundary (not the generation-suffix hint below)
    size_t free_cut = 0, extra_cut = 0, gen_cut = 0;
    std::array<size_t, 3> nat{};
    bool have_nat = false;
    for (size_t h : hints) {
      if (h <= i || h >= i + (size_t)M || !hint_ok(h)) continue;
      if (last_upper_skipped && !rt.tail_mode_for((int)(h - i))) continue;
      // A generation-suffix hint (the server sends it at most GEN_TAIL_MAX = 16 tokens before the end — hive_server.boundary_hints):
      //   the piece after it is a few tokens on the short path (decode policy, about one decode step), and the chunk before it keeps
      //   its tail mode with the upper layers on (the next chunk is not a tail-mode chunk), so the snapshot there is complete. It is
      //   where the next turn of the conversation continues (that turn renders this answer without its reasoning) — cut it for free.
      //   Taken only when no other boundary is cut in this chunk; a conversation boundary cut first leaves this one to the next chunk.
      if (ids.size() - h <= 16) { gen_cut = h; continue; }
      if (!have_nat) { nat = plan_cost((size_t)M, i + (size_t)M, cap); have_nat = true; }
      const auto cut = plan_cost(h - i, h, cap);
      if (cut[0] <= nat[0] && cut[1] <= nat[1] && cut[2] >= nat[2]) free_cut = h;
      else if (extra_left > 0 && h == deepest) extra_cut = h;
    }
    if (extra_cut > free_cut) { --extra_left; return (int)(extra_cut - i); }
    if (free_cut) return (int)(free_cut - i);
    return gen_cut ? (int)(gen_cut - i) : M;
  };
  // boundary snapshot of the same request's previous boundary (PJob::last_boundary — HIVE_CKPT_DELTA base; save_image validates lineage and prefix)
  auto snapshot_boundary_j = [&](PJob& J, size_t at, bool skipped) {
    auto& ids = J.ids; auto& S = J.S; auto& A = J.A; auto& last_boundary = J.last_boundary;
    auto sig_upto = [&](size_t h) { return sig_upto_j(J, h); };
    const auto sg = sig_upto(at);
    const uint64_t key = prefix_key(ids.data(), at, sg);
    auto it = shared_snaps.find(key);
    if (it != shared_snaps.end()) {
      const bool same = it->second.image && (size_t)it->second.image->pos == at && std::equal(it->second.image->tokens.begin(), it->second.image->tokens.end(), ids.begin());
      if (!same || !(it->second.upper_skipped && !skipped)) { if (same) it->second.last_used = now_ms(); return; }  // already present (or a hash collision — keep the old one)
    }
    auto img = std::make_shared<SeqImage>();
    rt.save_image(*S->seq, *img, last_boundary ? last_boundary.get() : S->ckpt.get());
    host_total(img.get());
    shared_snaps[key] = SharedSnap{img, sg, skipped, now_ms()};
    last_boundary = img;
    ++shared_saves;
    fprintf(stderr, "[hived] %s: prefix snapshot at %zu (upper %s · %zu entries)\n", A->sid.c_str(), at, skipped ? "skip" : "on", shared_snaps.size());
    host_total();
    archive_trim(A->sid);
  };
  // admit is split into stages (per-request state = PJob, on the heap): setup -> [chunk_begin -> forward -> chunk_end]* -> finish.
    //   A single request (admit) runs them in order with the same calls. batched = called from the HIVE_BATCH_PREFILL batched path (admit_batch).
  auto admit_setup = [&](Request&& r, bool batched) -> std::unique_ptr<PJob> {
    auto Jp = std::make_unique<PJob>();
    PJob& J = *Jp;
    J.A = std::make_unique<Active>();
    auto& A = J.A;
    A->r = std::move(r);
    const json& q = A->r.req;
    A->sid = q.value("session", "default");
    auto& ids = J.ids;
    ids = q.value("ids", std::vector<int32_t>{});
    A->temperature = q.value("temperature", 1.0f); A->top_p = q.value("top_p", 1.0f); A->min_p = q.value("min_p", 0.f);
    A->top_k = q.value("top_k", 0);
    A->stop_ids = q.value("stop_ids", std::vector<int32_t>{1});
    {  // Priority (the server's request_priority: vLLM "priority" / OpenAI "service_tier") — a number, clamped to ±1000; anything else = 0 (absorbed)
      const auto ip = q.find("priority");
      if (ip != q.end() && ip->is_number()) { const double v = ip->get<double>(); if (std::isfinite(v)) A->prio = (int)std::max(-1000.0, std::min(1000.0, v)); }
    }
    {  // Thinking cap (think_cap, think_end_id, think_start_id, think_open — the server sends them only for thinking-mode requests carrying a
       //   max_thinking_tokens-style field). Absorption rules (not guards): if think_cap is not a positive integer or think_end_id is outside
       //   the vocabulary, the feature is off · think_open absent = true (a thinking-mode prompt ends with <think> — DS encoding.py) ·
       //   think_start_id is optional (a <think> emitted later in the body re-enters the thinking span — the cap is cumulative).
      const auto ic = q.find("think_cap"), ie = q.find("think_end_id"), is = q.find("think_start_id"), io = q.find("think_open");
      const double cap = ic != q.end() && ic->is_number() ? ic->get<double>() : 0.0;
      const double end = ie != q.end() && ie->is_number() ? ie->get<double>() : -1.0;
      if (cap >= 1.0 && end >= 0.0 && end < (double)c.vocab) {
        A->think_cap = (int)std::min(cap, 1e9);
        A->think_end = (int32_t)end;
        A->think_start = is != q.end() && is->is_number() && is->get<double>() >= 0.0 && is->get<double>() < (double)c.vocab ? (int32_t)is->get<double>() : -1;
        A->in_think = io == q.end() || !io->is_boolean() || io->get<bool>();
        // exit phrase: only an array of in-vocabulary integer ids (absorption rule — if it is not an array or any id is out of vocabulary, no phrase)
        const auto ix = q.find("think_exit_ids");
        if (ix != q.end() && ix->is_array()) {
          std::vector<int32_t> ex;
          bool ok = true;
          for (const auto& v : *ix) {
            if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() >= (int64_t)c.vocab) { ok = false; break; }
            ex.push_back((int32_t)v.get<int64_t>());
          }
          if (ok) A->think_exit = std::move(ex);
        }
      }
    }
    {  // seed: 0, absent or non-numeric = a different seed per request (request_seed in hive/sampler.h — seeding from now_ms gave requests in the
      // same ms the same seed) · integer = a deterministic random sequence
      const auto sd = q.find("seed");
      const int64_t seed = sd != q.end() && sd->is_number_integer() ? sd->get<int64_t>()
                           : sd != q.end() && sd->is_number_float() && std::fabs(sd->get<double>()) < 9.0e18 ? (int64_t)sd->get<double>() : 0;
      A->rng.seed(request_seed(seed));
    }
    auto fail = [&](const char* msg) { send_json(A->r.fd, {{"error", msg}}); finish_socket(A->r.fd); };
    if (ids.empty()) { fail("ids empty"); return nullptr; }
    // Every id indexes the embedding table (runtime prologue_rows memcpy) and the token history — a value outside [0, vocab) would read
    //   outside the table (negative ids wrap to ~2^64). The API server tokenizes, so this only triggers on a raw socket client or a
    //   tokenizer/checkpoint mismatch; it is a request error, not a daemon exit.
    for (const int32_t id : ids) if (id < 0 || id >= c.vocab) { fail("id out of vocabulary"); return nullptr; }
    if (!fake_head && (!model.has_head() || nl < c.n_layers)) { fail("model incomplete (partial checkpoint)"); return nullptr; }
    if ((int64_t)ids.size() + 1 > max_ctx) {  // keep the reason string as is, plus the numbers the server needs to build an OpenAI-style 400 (context_length_exceeded)
      send_json(A->r.fd, {{"error", "context overflow"}, {"max_ctx", max_ctx}, {"prompt_tokens", ids.size()}});
      finish_socket(A->r.fd);
      return nullptr;
    }
    {  // max_tokens defaults to the full context: unspecified (absent, <= 0, non-numeric) = the rest of the context (max_ctx - prompt).
       //   A value larger than the remaining context is clamped to it rather than rejected.
      const int64_t remaining = max_ctx - (int64_t)ids.size();
      const auto mt_it = q.find("max_tokens");
      const double mtd = mt_it != q.end() && mt_it->is_number() ? mt_it->get<double>() : 0.0;
      const int64_t mt = std::isfinite(mtd) ? (int64_t)std::clamp(mtd, 0.0, 9.0e18) : 0;  // non-finite/huge = unspecified (the cast alone would be UB)
      A->max_tokens = (int)(mt <= 0 || mt > remaining ? remaining : mt);
    }
    if (active_sids.count(A->sid)) {
      // regenerate right after stop — if the previous request is already cancelled, finish it here and accept (otherwise regenerate fails with "session busy")
      for (size_t i = 0; i < active.size(); ++i)
        if (active[i]->sid == A->sid && active[i]->cancel->load()) { active[i]->finish = "cancel"; finish_active(*active[i]); active.erase(active.begin() + i); break; }
      if (active_sids.count(A->sid)) { fail("session busy"); return nullptr; }
    }
    // session and prefix
    Session*& S = J.S;
    {
      std::lock_guard<std::mutex> lk(mu);
      auto it = sessions.find(A->sid);
      if (it == sessions.end()) {
        bool evicted = false;
        if ((int)sessions.size() >= max_sessions) {  // LRU eviction (active excluded)
          auto victim = sessions.end();
          for (auto jt = sessions.begin(); jt != sessions.end(); ++jt)
            if (!active_sids.count(jt->first) && !prefilling_sids.count(jt->first) && (victim == sessions.end() || jt->second.last_used < victim->second.last_used)) victim = jt;
          if (victim != sessions.end()) {
            Session& V = victim->second;
            if (V.seq) {
              // capture the last state (for a next request that continues exactly) and return the buffers to the pool. The checkpoint moves as is.
              if (!V.tokens.empty() && !V.seq->broken && V.seq->pos == (int64_t)V.tokens.size()) {
                V.live = std::make_shared<SeqImage>();
                rt.save_image(*V.seq, *V.live, V.ckpt.get());
                V.live_sig = V.img_sig;
              }
              seq_pool.push_back(std::move(V.seq));
            }
            archived[victim->first] = std::move(V);
            sessions.erase(victim);
            evicted = true;
          }
        }
        // Reclaim only **after** the session to revive has been taken out of archived, and excluding it (trimming right after the eviction would
                //   delete the oldest state first — the one about to be revived). The revived state is in sessions, so it still counts toward the budget.
        auto ar = archived.find(A->sid);
        if (ar != archived.end()) { it = sessions.emplace(A->sid, std::move(ar->second)).first; archived.erase(ar); }
        else it = sessions.emplace(A->sid, Session{}).first;
        if (evicted) archive_trim(A->sid);
      }
      S = &it->second;
    }
    A->S = S;
    auto& request_sig = J.request_sig;
    if (q.contains("images") && q["images"].is_array()) {
      size_t off=0;
      for (const auto& im : q["images"]) {
        const size_t nb=im.at("nbytes").get<size_t>();
        if (nb>A->r.bin.size()-off) { fail("image bytes short"); return nullptr; }
        uint64_t hash=1469598103934665603ull;
        auto mix=[&](uint8_t b){hash^=b;hash*=1099511628211ull;};
        for (size_t b=0;b<nb;++b) mix(A->r.bin[off+b]);
        for (const auto& t: im.at("types")) mix((uint8_t)t.get<int>());
        for (auto key : {"n_vit_h", "n_vit_w"}) { uint32_t n=im.at(key).get<uint32_t>(); for(int b=0;b<4;++b) mix((n>>(8*b))&255); }
        int start=im.at("start").get<int>();
        request_sig.push_back({start,start+(int)im.at("types").size(),hash}); off+=nb;
      }
    }
    auto images_match = [&](const std::vector<Session::ImgSig>& saved, size_t pos) {
      std::vector<std::tuple<int,int,uint64_t>> a,b;
      for (const auto& r:saved) if ((size_t)r.end<=pos) a.emplace_back(r.start,r.end,r.hash);
      for (const auto& r:request_sig) {
        if ((size_t)r.start<pos && (size_t)r.end>pos) return false;
        if ((size_t)r.end<=pos) b.emplace_back(r.start,r.end,r.hash);
      }
      std::ranges::sort(a);std::ranges::sort(b);return a==b;
    };
    // Pick where to continue writing: (1) the live state (all session tokens are a prefix, the new request is longer), (2) the prompt-end checkpoint
        //   (its token sequence is a prefix), (3) the last state of a session returning from outside the pool — whichever reaches furthest. Re-requesting
        //   the same prompt needs no prefill when checkpoint logits exist.
    auto lcp = [&](const std::vector<int32_t>& a) { size_t k = 0; while (k < a.size() && k < ids.size() && a[k] == ids[k]) ++k; return k; };
    const size_t live_lcp = S->seq ? lcp(S->tokens) : 0;
    size_t& common = J.common;
    int& pick = J.pick;  // 0 reset · 1 live state · 2 checkpoint · 3 archived last state
    // The live state is valid only if its KV position equals the session token count — after an exception during prefill or an "image span"
        //   failure, tokens are a prefix but seq.pos is ahead by the chunks already processed; continuing would silently misalign RoPE positions,
        //   ring cells and compressed rows.
    const bool live_ok = S->seq && !S->seq->broken && S->seq->pos == (int64_t)S->tokens.size();  // broken = exception during a forward
    if (live_ok && live_lcp == S->tokens.size() && ids.size() > live_lcp && images_match(S->img_sig,live_lcp)) { common = live_lcp; pick = 1; }
    // Image requests do not use the checkpoint or archived state by default — different pictures of the same size have identical placeholder ids,
        //   so token comparison cannot tell them apart (an exact re-request would return the old picture's logits). Checkpoints are also taken only for
        //   requests without images (below).
    J.has_images = q.contains("images") && q["images"].is_array() && !q["images"].empty();
    const bool has_images = J.has_images;
    // History/image reuse are opt-in; token AND image prefixes must match.
    if (!has_images || image_ckpt) {
      size_t best = S->ckpt ? lcp(S->ckpt->tokens) : 0;
      if (S->ckpt && (best != (size_t)S->ckpt->pos || !images_match(S->ckpt_sig,best))) best = 0;
      for (auto& f : S->history) {
        if (!f.image) continue;
        const size_t k = lcp(f.image->tokens);
        if (k <= best || k != (size_t)f.image->pos || (k == ids.size() && f.logits.empty()) || !images_match(f.sig,k)) continue;
        std::swap(S->ckpt, f.image); std::swap(S->ckpt_logits, f.logits); std::swap(S->ckpt_sig, f.sig);
        best = k;
      }
    }
    if ((!has_images || image_ckpt) && S->ckpt && (size_t)S->ckpt->pos > common && images_match(S->ckpt_sig,S->ckpt->pos)) {
      const size_t k = lcp(S->ckpt->tokens);
      const bool exact = k == ids.size() && k == (size_t)S->ckpt->pos && !S->ckpt_logits.empty();
      if (k == (size_t)S->ckpt->pos && (ids.size() > k || exact)) { common = k; pick = 2; }
    }
    if ((!has_images || image_ckpt) && S->live && (size_t)S->live->pos > common && images_match(S->live_sig,S->live->pos)) {
      const size_t k = lcp(S->live->tokens);
      if (k == (size_t)S->live->pos && ids.size() > k) { common = k; pick = 3; }
    }
    if (!S->seq) S->seq = take_seq();
    // the session's token count before the decision below rewrites it — the log line used to print the count after a reset (always 0), which
    //   read as "the session lost its state" when the client had in fact rewritten an earlier part of the prompt (2026-10-06 audit of 145 such
    //   lines: 88 rewritten prompts, 50 identical re-sends, 4 changed outputs, 3 after a cancel — none was a lost session)
    const size_t session_tokens = S->tokens.size();
    if (pick == 0 || pick == 2 || pick == 3) {
      if (pick == 0) { rt.reset_seq(*S->seq); S->img_sig.clear(); }
      else { rt.load_image(*S->seq, pick == 2 ? *S->ckpt : *S->live); S->img_sig = pick == 2 ? S->ckpt_sig : S->live_sig; }
      S->tokens.assign(ids.begin(), ids.begin() + common);
    }
    S->live.reset();  // an archived last state is used once (or dropped on mismatch) — the session is live again
    if (!S->tokens.empty() || live_lcp)
      fprintf(stderr, "[hived] %s: reuse %zu/%zu via %s (session tokens %zu · common with live %zu)\n", A->sid.c_str(), common, ids.size(),
              pick == 1 ? "live" : pick == 2 ? "prompt-ckpt" : pick == 3 ? "archived" : "reset", session_tokens, live_lcp);
    S->last_used = now_ms();
    A->cancel = A->r.cancel.get();
    // Images: metadata + bin (concatenated bf16 patches). A span must fit entirely in one prefill chunk (chunk boundaries are cut before the span).
    auto& all_images = J.all_images;
    auto& patch_bufs = J.patch_bufs;
    auto& req_sig = J.req_sig;
    if (q.contains("images") && q["images"].is_array()) {
      size_t off = 0;
      for (auto& im : q["images"]) {
        ImageInput x;
        x.start = im["start"].get<int>();
        x.n_vit_h = im["n_vit_h"].get<int>();
        x.n_vit_w = im["n_vit_w"].get<int>();
        for (auto& t : im["types"]) x.types.push_back((int8_t)t.get<int>());
        size_t nb = im["nbytes"].get<size_t>();
        // Geometry vs. bytes: the patch tensor is n_vit_h x n_vit_w patches of 3 x patch x patch bf16; the vision kernels index by that product,
        //   so a mismatch (or a product beyond the patch budget) would read outside the buffer on the device. Rejected here as a request error.
        const int64_t npatch = (int64_t)x.n_vit_h * (int64_t)x.n_vit_w;
        const int64_t pbytes = rt.image_patch_bytes();
        const int64_t max_patches = vision_max_patches > 0 ? vision_max_patches : (int64_t)c.vision_max_tokens * c.vision_downsample * c.vision_downsample;  // as the runtime
        if (x.n_vit_h <= 0 || x.n_vit_w <= 0 || npatch > max_patches || (int64_t)nb != npatch * pbytes ||
            x.start < 0 || (size_t)x.start >= ids.size() || x.types.empty()) { fail("image geometry invalid"); return nullptr; }
        if (off + nb > A->r.bin.size()) { fail("image bytes short"); return nullptr; }
        req_sig.push_back(request_sig[req_sig.size()]);
        DevBuf d(nb);
        CUDA_CHECK(cudaMemcpy(d.p, A->r.bin.data() + off, nb, cudaMemcpyHostToDevice));
        off += nb;
        x.patches_dev = d.as<bf16>();
        patch_bufs.push_back(std::move(d));
        all_images.push_back(std::move(x));
      }
      if (!all_images.empty() && !rt.has_vision()) { fail("vision encoder not loaded"); return nullptr; }
      if (!all_images.empty() && common > 0) {  // image requests run whole without prefix reuse (never continue inside a span)
        bool span_after = false;
        for (auto& x : all_images) if ((size_t)x.start + x.types.size() > common) span_after = true;
        // images inside the prefix must have been produced from the same picture by the live state (signature check)
        bool sig_ok = true;
        for (const auto& r : req_sig) {
          if ((size_t)r.end > common) continue;
          bool found = false;
          for (const auto& o : S->img_sig) if (o.start == r.start && o.end == r.end && o.hash == r.hash) { found = true; break; }
          if (!found) sig_ok = false;
        }
        if ((!image_ckpt && span_after) || !sig_ok) { rt.reset_seq(*S->seq); S->tokens.clear(); S->img_sig.clear(); common = 0; pick = 0; }
      }
    }
    // prefill (chunks)
    A->t0 = now_ms();
    bool& last_upper_skipped = J.last_upper_skipped;  // did the last chunk run skip the tail layers — if it is cut after that, the tail-layer ring holds stale values and must not be continued
    // Chunking while others decode: chunk sizes trade transfer cost against decoder stalls. Measured: 1023 rows take the grouped path (<1024,
        //   85% CPU experts) at 7.4 s per chunk, slower than a streaming 1K chunk (4.7-6.3 s), so busy chunks use the streaming path. Each prefill
        //   chunk re-streams the non-resident experts (~207 GB) over PCIe, so larger chunks are cheaper: splitting a 9.7K prompt into 2048-token
        //   chunks = 5 chunks = 5x the transfer (measured 61.6 s vs ~16 s unsplit). The default does not split — other decoders run only via
        //   decode_share between chunks (with a single chunk they pause for that chunk).
        // Tile = max_chunk x prefill_tile tokens in one forward (experts transferred once per encoder layer). Image requests use max_chunk per the
        // single-chunk rule.
    size_t& chunk_cap = J.chunk_cap;  // text = max_chunk x prefill_slots (host tiles included — equals prefill_tile when off)
    chunk_cap = all_images.empty() ? (size_t)max_chunk * (size_t)rt.prefill_slots() : (size_t)max_chunk;
    size_t& busy_cap = J.busy_cap;
    busy_cap = busy_chunk <= 0 ? chunk_cap : std::min<size_t>(chunk_cap, (size_t)busy_chunk);
    // One chunk-size rule: the chunk to cut from at within cap. This chunk and the "next chunk" used for the C1 decision share the same function.
    auto plan_chunk = [&](size_t at, size_t cap) { return plan_chunk_j(J, at, cap); };
    auto cap_now = [&]() { return cap_now_j(J); };
    // Boundary snapshots: the longest shared boundary snapshot exactly matching this request's token prefix (image signatures included) — only
        //   if longer than the session's own state (pick above). It holds no logits, so at least one token must remain to prefill. C1: a snapshot
        //   taken after a chunk that skipped the tail layers is used only if the first continuing chunk is committed in tail mode (text request ·
        //   first chunk = plan_chunk(k, cap_now()) as the loop will actually cut it — the state does not change in between), and after restoring,
        //   last_upper_skipped is set to that value so the first chunk's boundary cut also respects tail mode (hint_cut below).
    if (prefix_share && (!has_images || image_ckpt) && !shared_snaps.empty()) {
      SharedSnap* best = nullptr;
      // in a batched prefill the first chunk may shrink this round to one slot (max_chunk) or the decoders-present size (busy/budget) — measure
            // with that lower bound (tail_mode_for is monotone)
      const size_t restore_cap = batched ? std::min({cap_now(), c1_floor_j(J), (size_t)max_chunk}) : cap_now();
      size_t best_len = common;
      for (auto& [key, e] : shared_snaps) {
        const size_t k = e.image ? (size_t)e.image->pos : 0;
        if (k <= best_len || k >= ids.size() || e.image->tokens.size() != k || !std::equal(e.image->tokens.begin(), e.image->tokens.end(), ids.begin())) continue;
        if (!images_match(e.sig, k)) continue;
        if (e.upper_skipped && !(all_images.empty() && rt.tail_mode_for((int)plan_chunk(k, restore_cap)))) continue;  // C1 validity
        best = &e; best_len = k;
      }
      if (best) {
        rt.load_image(*S->seq, *best->image);
        S->tokens.assign(ids.begin(), ids.begin() + best_len);
        S->img_sig = best->sig;
        best->last_used = now_ms();
        ++shared_hits;
        last_upper_skipped = best->upper_skipped;
        fprintf(stderr, "[hived] %s: reuse %zu/%zu via shared-prefix (was %s %zu · upper %s · %zu entries)\n", A->sid.c_str(), best_len, ids.size(),
                pick == 1 ? "live" : pick == 2 ? "prompt-ckpt" : pick == 3 ? "archived" : "reset", common, best->upper_skipped ? "skip" : "on", shared_snaps.size());
        common = best_len; pick = 4;
      }
    }
    // HIVE_IMAGE_CKPT: images entirely inside the reused prefix are already processed (not fed to chunks again — the loop only covers i >= common).
        //   Removing them from the chunk rules means a new span without images uses the same chunk rules as a text request (tiles, C1, tail split,
        //   budget). Images in the new span keep the rule one span = one chunk. Lossless: chunk boundaries do not change the result (same property
        //   as the tail split), and the prefix is the same state down to the image signatures.
    if (image_ckpt && common > 0 && !all_images.empty()) {
      std::vector<ImageInput> suffix;
      for (auto& x : all_images) if ((size_t)x.start + x.types.size() > common) suffix.push_back(x);
      if (suffix.size() != all_images.size()) {
        all_images = std::move(suffix);
        chunk_cap = all_images.empty() ? (size_t)max_chunk * (size_t)rt.prefill_slots() : (size_t)max_chunk;
        busy_cap = busy_chunk <= 0 ? chunk_cap : std::min<size_t>(chunk_cap, (size_t)busy_chunk);
      }
    }
    A->common = common;
    A->prompt_len = ids.size();
    // "stages": true (the API server's HIVE_STAGE_STATUS — the engine status of a request): one line once the reuse is decided, before the
    //   prefill — how much of the prompt is reused and how much will be read. Requests without it get no new line.
    if (q.contains("stages") && q["stages"].is_boolean() && q["stages"].get<bool>() &&
        !send_json(A->r.fd, {{"admitted", true}, {"cached", (int)common}, {"total", (int)ids.size()}}))
      A->client_ok = false;
    // Boundaries supplied by the server (the request's "boundaries") — integers only, inside (0, length), sorted and deduplicated. Unknown formats are dropped (absorbed).
    auto& hints = J.hints;
    if (prefix_share && !fake_head && (!has_images || image_ckpt) && q.contains("boundaries") && q["boundaries"].is_array()) {
      for (const auto& b : q["boundaries"])
        if (b.is_number_integer() && b.get<int64_t>() > (int64_t)common && b.get<int64_t>() < (int64_t)ids.size() && b.get<int64_t>() >= ckpt_min_tokens)
          hints.push_back((size_t)b.get<int64_t>());
      std::sort(hints.begin(), hints.end());
      hints.erase(std::unique(hints.begin(), hints.end()), hints.end());
    }
    // Per-request extra-chunk budget (the request's optional "prefix_extra", a non-negative integer) overrides HIVE_PREFIX_EXTRA_CHUNKS for
    //   this request only. The server sends it when a conversation is not append-only (its tail changes between turns), so the boundary
    //   before the changing tail gets a snapshot that the next turn can resume from; append-only conversations keep the global budget.
    //   Anything else (missing, negative, non-integer) is absorbed as "use the global budget".
    J.extra_left = q.contains("prefix_extra") && q["prefix_extra"].is_number_integer() && q["prefix_extra"].get<int64_t>() >= 0
                       ? (int)std::min<int64_t>(q["prefix_extra"].get<int64_t>(), std::numeric_limits<int>::max()) : prefix_extra;
    J.pending = std::make_unique<PrefillPending>(PrefillPending{rt, A->sid, prefill_pending_n});
    J.i = common;
    return Jp;
  };
  // Plan this chunk: size, boundary cut, image spans, multi-chunk marker, C1 (upper_needed). false = the request ended with a failure (response sent).
  auto chunk_begin = [&](PJob& J, size_t lim, bool batched) -> bool {
    auto& A = J.A; auto& ids = J.ids; auto& all_images = J.all_images; auto& st = J.st; auto& pending = *J.pending;
    const size_t i = J.i;
    const size_t chunk_cap = J.chunk_cap, busy_cap = J.busy_cap;
    auto fail = [&](const char* msg) { send_json(A->r.fd, {{"error", msg}}); finish_socket(A->r.fd); };
    auto cap_now = [&]() { return cap_now_j(J); };
    auto plan_chunk = [&](size_t at, size_t cap) { return plan_chunk_j(J, at, cap); };
    auto hint_cut = [&](size_t at, int m, size_t cap) { return hint_cut_j(J, at, m, cap); };
    const size_t chunk_now = std::min(cap_now(), lim);  // lim = this round's slots x max_chunk (unlimited on the single path)
    int M = (int)plan_chunk(i, chunk_now);
    M = hint_cut(i, M, chunk_now);
    for (auto& x : all_images) {  // if the chunk splits a span, cut before the span (error if the span is longer than a chunk)
      size_t s0 = (size_t)x.start, s1 = s0 + x.types.size();
      if (s1 <= i || s0 >= i + M) continue;
      if (s0 >= i && s1 <= i + M) continue;
      if (s0 > i) { M = (int)(s0 - i); break; }
      fail("image span longer than chunk");
      return false;
    }
    if (i + (size_t)M < ids.size()) pending.set();
    auto& chunk_images = J.chunk_images;
    chunk_images.clear();
    for (auto& x : all_images) {
      size_t s0 = (size_t)x.start, s1 = s0 + x.types.size();
      if (s0 >= i && s1 <= i + M) { ImageInput y = x; y.start = (int)(s0 - i); chunk_images.push_back(std::move(y)); }
    }
    // C1: if the next chunk is certainly in decoder-tail mode (even the smallest next chunk the remaining tokens can form is in tail mode), this
        //   chunk skips the tail layers — removing the per-chunk re-streaming of non-resident experts for layers 20-39 (half of the expert transfer).
        //   Image requests are excluded.
        // The next chunk's lower bound follows the same rule. On the single path without yielding, active only shrinks inside the prefill loop
        //   (admit is called only from the main loop; decode_step between chunks only removes finished requests or clears on exceptions and never
        //   touches the queue) -> if it is empty now the next cap is chunk_cap; with decoders it is busy_cap (lower bound stream_floor with a budget)
        //   or chunk_cap once they finish. Using the budget/busy size without decoders would turn C1 off needlessly.
        //   tail_mode_for is monotone in M and plan_chunk's pulled boundary is also in tail mode, so if the next chunk measured with the smallest cap
        //   is in tail mode, every case is.
        //   Boundary cuts (hint_cut) only cut at tail-mode sizes after a skipped chunk, so they do not break this invariant.
    const size_t rem_after = ids.size() - (i + (size_t)M);
    size_t next_cap = chunk_cap;
    if (!active.empty()) { next_cap = busy_cap; if (prefill_budget_ms>0 && all_images.empty()) next_cap = std::min(next_cap, stream_floor); }
    if (batched) next_cap = std::min(c1_floor_j(J), (size_t)max_chunk);  // batched: the next round may have one slot and decoders present (c1_floor_j)
    else if (prefill_yield || layer_yield) next_cap = std::min(next_cap, c1_floor_j(J));  // HIVE_PREFILL_YIELD / HIVE_LAYER_YIELD: active can grow between chunks (inside the forward for layer yield) — use the decoders-present lower bound
    const size_t next_min = rem_after > 0 ? plan_chunk(i + (size_t)M, next_cap) : 0;
    J.upper_needed = !(rem_after > 0 && all_images.empty() && rt.tail_mode_for(M) && rt.tail_mode_for((int)next_min));
    J.hit0 = st.n_hit; J.str0 = st.n_streamed; J.cpu0 = st.n_cpu; J.dma0 = st.n_dma_rows;
    J.wait0 = st.ms_cpu_wait; J.span0 = st.ms_cpu_span;
    st.tail_rows = 0;  // the runtime overwrites it only for tail-mode chunks — clear it so it reads as a per-chunk value
    J.M = M;
    return true;
  };
  // After a chunk: C1 record, EMA, chunk log, boundary snapshot, stats, cancel, progress notice. false = end the loop (cancel, disconnect).
  auto chunk_end = [&](PJob& J, double tc, size_t rows) -> bool {
    auto& A = J.A; auto& ids = J.ids; auto& st = J.st; auto& hints = J.hints; size_t& i = J.i; auto& n_chunks = J.n_chunks;
    const int M = J.M; const bool upper_needed = J.upper_needed; auto& last_upper_skipped = J.last_upper_skipped;
    const int hit0 = J.hit0, str0 = J.str0, cpu0 = J.cpu0, dma0 = J.dma0; const double wait0 = J.wait0, span0 = J.span0;
    auto snapshot_boundary = [&](size_t at, bool skipped) { snapshot_boundary_j(J, at, skipped); };
    last_upper_skipped = !upper_needed;
    const double sample_ms_per_token=tc/(double)std::max<size_t>(1,rows);  // rows = all rows of this forward (single = M · batched = the sum)
    prefill_ms_per_token=prefill_ms_per_token>0 ? .8*prefill_ms_per_token+.2*sample_ms_per_token : sample_ms_per_token;
    // per-chunk instrumentation: one line with the chunk's ForwardStats deltas
    ++n_chunks;
    // HIVE_LAYER_YIELD: if layer yields happened inside this forward, append " · layer yields N last X ms" (X = last resume -> forward end —
        // used by monitors as a gap). tc excludes the yield time.
    char ly_tail[64] = "";
    if (J.ly_n > 0) snprintf(ly_tail, sizeof ly_tail, " · layer yields %ld last %.0f ms", J.ly_n, J.ly_last);
    fprintf(stderr, "[hived] %s: prefill chunk %d M %d · %zu→%zu/%zu · upper %s · tail rows %d · hit %d streamed %d cpu %d dma_rows %d · cpu wait %.0f ms span %.0f ms · %.0f ms%s\n",
            A->sid.c_str(), n_chunks, M, i, i + (size_t)M, ids.size(), upper_needed ? "on" : "skip", st.tail_rows, st.n_hit - hit0,
            st.n_streamed - str0, st.n_cpu - cpu0, st.n_dma_rows - dma0, st.ms_cpu_wait - wait0, st.ms_cpu_span - span0, tc, ly_tail);
    J.ly_n = 0; J.ly_last = 0;
    i += M;
    if (!hints.empty() && i < ids.size() && std::binary_search(hints.begin(), hints.end(), i)) snapshot_boundary(i, !upper_needed);
    publish_stats(true);  // keep stats moving during long prefills (one chunk = seconds)
    if (stop_state.load() >= 2) A->cancel->store(true);  // graceful-stop deadline passed: stop before the next chunk (finished afterwards with a shutdown error)
    if (A->cancel->load()) return false;
    if (!send_json(A->r.fd, {{"progress", (int)i}, {"total", (int)ids.size()}})) { A->client_ok = false; return false; }
    return true;
  };
  // Between chunks: give running decoders their decode_share (tc = wall clock of the forward just run)
  auto decode_between = [&](double tc) {
      // Give running decoders decode_share of the wall clock between chunks (measured: 8 concurrent 8K prompts with one step per chunk starved
            //   decoding at 0.2 tok/s). share 0.34 = ~2.9 s of decode per 5.6 s prefill chunk (steps of ~40-300 ms, several times). 0 = one step.
      const double budget = decode_share > 0 && decode_share < 1 ? tc * decode_share / (1.0 - decode_share) : 0;
      const double td0 = now_ms();
      do { decode_step(); } while (!active.empty() && now_ms() - td0 < budget);
  };
  // After prefill: session tokens, image signatures, checkpoint, warm -> active
  auto admit_finish = [&](PJob& J) {
    auto& A = J.A; auto& ids = J.ids; auto& S = J.S; auto& st = J.st; const size_t common = J.common; const int pick = J.pick;
    const int n_chunks = J.n_chunks; const bool last_upper_skipped = J.last_upper_skipped; auto& all_images = J.all_images; auto& req_sig = J.req_sig;
    prefilling_sids.erase(A->sid);
    J.pending->clear();
    if (n_chunks)
      fprintf(stderr, "[hived] %s: prefill total %zu rows in %d chunks · hit %d streamed %d cpu %d dma_rows %d · cpu wait %.0f ms span %.0f ms · %.0f ms\n",
              A->sid.c_str(), (size_t)S->seq->pos - common, n_chunks, st.n_hit, st.n_streamed, st.n_cpu, st.n_dma_rows, st.ms_cpu_wait, st.ms_cpu_span, now_ms() - A->t0);
    // re-request of the same prompt (checkpoint logits) — the prefill loop did not run
    const bool reused_exact = pick == 2 && common == ids.size();
    if (reused_exact) A->logits = S->ckpt_logits;
    // If the loop exits early (cancel, send failure), keep only the processed tokens in the session (recording all ids would let the next request
        //   treat unprocessed tokens as processed). Checkpoints are taken only on full completion.
    const size_t done_tokens = (size_t)S->seq->pos;
    const bool prefill_complete = done_tokens == ids.size();
    if (!prefill_complete && last_upper_skipped) { rt.reset_seq(*S->seq); S->tokens.clear(); }  // cut off after a C1 chunk = a state that cannot be continued
    else S->tokens.assign(ids.begin(), ids.begin() + std::min(done_tokens, ids.size()));
    {  // image signatures: only those inside the processed tokens (this request's first; otherwise the previous ones inside the remaining prefix)
      std::vector<Session::ImgSig> keep;
      const auto& src = req_sig.empty() ? S->img_sig : req_sig;
      for (const auto& r : src) if ((size_t)r.end <= S->tokens.size()) keep.push_back(r);
      S->img_sig = std::move(keep);
    }
    if (prefill_complete && !reused_exact && (all_images.empty() || image_ckpt) && (int)ids.size() >= ckpt_min_tokens && !fake_head) {
      if (defer_ckpt) A->defer_ckpt = true;  // HIVE_DEFER_CKPT: after the first token (decode_step)
      else save_prompt_ckpt(*A, A->logits);
    }
    fake_logits(A->logits, ids.size());
    // warm the cache after a prefill of >= 64 tokens (synchronous, or handed to the decode steps with HIVE_WARM_DEFER) — skipped when the cache
        // is already >= 90% full (measured: a synchronous 1.3 s per request = 14% of c1 TTFT; per-token promotion continues the job)
    if (ids.size() - common >= 64 && store.n_resident() < store.n_slots() * 9 / 10) {
      if (defer_ckpt) A->defer_warm = true;
      else warm_after_prefill(0);
    }
    A->t1 = now_ms();
    active_sids.insert(A->sid);
    active.push_back(std::move(A));
  };
  // Recover host exceptions per request. CUDA_CHECK/HIVE_CHECK aborts are fatal,
  // not C++ exceptions; never present a broken CUDA context as recovered.
  auto admit_error = [&](int fd, const std::exception& e, const std::string& sid) {
    fprintf(stderr, "[hived] %s: request failed: %s\n", sid.c_str(), e.what());  // logging: include the session id (so failures can be attributed)
    rt.quiesce_after_host_error();
    send_json(fd, {{"error", std::string("request failed: ") + e.what()}});
    finish_socket(fd);
  };
  auto admit_epilogue = [&](const std::shared_ptr<std::atomic<bool>>& flag, const std::string& sid) {
    const bool admitted=std::any_of(active.begin(),active.end(),[&](const auto& a){return a->r.cancel==flag;});
    if(!admitted) {
      std::lock_guard<std::mutex> lock(mu);
      auto it=cancel_flags.find(sid);
      if(it!=cancel_flags.end() && it->second==flag) cancel_flags.erase(it);
      for(const auto& a:active) if(a->sid==sid && !cancel_flags.count(sid)) cancel_flags[sid]=a->r.cancel;
    }
  };
  // HIVE_PREFILL_YIELD yielding between chunks/rounds of a long prefill. It calls admit below, so only declared here (defined after admit).
  std::function<void(double, size_t)> yield_after_chunk;
  // Admit one request: prefill it and put it in active (on failure or cancel: respond and drop it)
  auto admit_inner = [&](Request&& r) {
    auto Jp = admit_setup(std::move(r), false);
    if (!Jp) return;
    PJob& J = *Jp;
    // HIVE_PREFILL_YIELD: exclude this request's session from LRU eviction and session selection for requests admitted by yielding (same meaning
        //   as prefilling_sids on the batched path). admit_finish removes it; on early return (chunk planning failure) or exception this guard does.
        //   Not inserted when off.
        //   HIVE_LAYER_YIELD inserts it for the same reason — a request admitted by a layer yield must not evict or reuse this session's sequence
        //   (the forward is paused between layers).
    struct YieldSid { std::set<std::string>* s; std::string sid; ~YieldSid() { if (s) s->erase(sid); } } yield_sid{prefill_yield || layer_yield ? &prefilling_sids : nullptr, J.A->sid};
    if (prefill_yield || layer_yield) prefilling_sids.insert(J.A->sid);
    ly_mark(J.A->sid, J.ids.size());
    while (J.i < J.ids.size()) {
      if (!chunk_begin(J, SIZE_MAX, false)) return;
      const double tc0 = now_ms();
      const ly::Stats ly0 = rt.layer_yield_stats();
      if (!rt.in_layer_yield()) { ly_fwd_rows = (size_t)J.M; ly_rest_rows = J.ids.size() - J.i - (size_t)J.M; ly_mid_used = 0; }  // HIVE_LAYER_YIELD_MID
      ++prefill_epoch;
      rt.forward(*J.S->seq, J.ids.data() + J.i, J.M, J.chunk_images.empty() ? nullptr : &J.chunk_images, &J.A->logits, &J.st, J.upper_needed);
      const double tend = now_ms();
      // HIVE_LAYER_YIELD: tc = the prefill share (minus layer-yield time — so the EMA and the decode_between budget do not count yield time as prefill)
      J.ly_n = rt.layer_yield_stats().yields - ly0.yields;
      J.ly_last = J.ly_n > 0 ? tend - rt.layer_yield_resume_ms() : 0;
      const double tc = tend - tc0 - (rt.layer_yield_stats().inner_ms - ly0.inner_ms);
      last_prefill_tc = tc;
      if (!chunk_end(J, tc, (size_t)J.M)) break;
      if (J.i < J.ids.size() && prefill_yield) yield_after_chunk(tc, 1);
      else if (J.i < J.ids.size() && !active.empty()) decode_between(tc);
    }
    admit_finish(J);
  };
  auto admit = [&](Request&& r) {
    const int fd = r.fd;
    const auto flag=r.cancel;
    const std::string sid=r.req.value("session","default");
    try {
      admit_inner(std::move(r));
    } catch (const std::exception& e) {
      admit_error(fd, e, sid);
    }
    admit_epilogue(flag, sid);
  };
  // HIVE_PREFILL_YIELD: busy = number of requests prefilling now (1 on the single path, the job count on the batched path) — for the seat
    //   (max_batch) computation. Not called when off.
    //   Admits every short request it can take from the queue in FIFO order via the single path (requests that are not short or whose session
    //   is busy are skipped and keep their place).
  bool in_yield = false;
  yield_after_chunk = [&](double tc, size_t busy) {
    const bool had_active = !active.empty();
    int n = 0;
    if (!in_yield) {
      in_yield = true;
      const double saved_tc = last_prefill_tc;  // HIVE_PREFILL_FAIR looks at the outer admission's last prefill forward (not the short request's)
      const double t0 = now_ms();
      for (;;) {
        if ((int)(active.size() + busy) + 1 > rt.max_batch()) break;
        Request r;
        bool got = false;
        {
          std::lock_guard<std::mutex> lk(mu);
          if (stop_state.load() != 0 || sleep_asked.load()) break;  // stop taking requests once a sleep is requested
          for (auto it = queue.begin(); it != queue.end(); ++it) {
            const json& q = it->req;
            const std::string sid = q.value("session", "default");
            const size_t rows = rows_to_prefill(sid, q);  // ids minus the reusable prefix (lower bound) — a follow-up turn counts only its new tokens
            if (rows == 0 || rows > yield_max || active_sids.count(sid) || prefilling_sids.count(sid)) continue;
            r = std::move(*it); queue.erase(it); got = true;
            break;
          }
        }
        if (!got) break;
        admit(std::move(r));
        ++n;
      }
      last_prefill_tc = saved_tc;
      in_yield = false;
      if (n) fprintf(stderr, "[hived] prefill yield: %d short request%s admitted in %.0f ms (active %zu · prefilling %zu)\n", n, n > 1 ? "s" : "", now_ms() - t0,
                     active.size(), busy);
    }
    if (had_active) decode_between(tc);        // decoders were already running: their decode_share as usual
    else if (n && !active.empty()) decode_step();  // only the first token of the newly admitted short requests — no new decode_share on top of the long prefill
  };
  // HIVE_BATCH_PREFILL: run several queued text prefills per round in one rt.forward_multi — sharing the expert transfer per layer.
    //   Per-request semantics match the single path: setup (reuse selection), chunk rules, C1, boundary snapshots, checkpoints, cancellation
    //   and errors are handled per request by the same functions.
    //   - Round: take more batchable requests from the queue head (FIFO — stop when the head cannot join), then size each request's chunk in
    //     request order within the remaining slots x max_chunk (the first request gets all slots = the same cap as the single path). Two or more
    //     streaming-sized chunks (>= prefill_threshold) go to forward_multi, otherwise each runs its own forward.
    //   - A request whose prefill finishes goes straight to active — its first token leaves during the decode_share between rounds.
    //     Cancel/disconnect ends only that request.
    //   - If forward_multi throws, the culprit is unknown (every seq is broken): the remaining requests go back to the queue head and are
    //     re-admitted one at a time (solo) — a failing request ends with the same error on the single path, the others start over (from reuse
    //     selection).
  auto batch_eligible = [&](const Request& r, bool head) {
    if (!rt.batch_prefill() || r.solo || rt.prefill_slots() < 2) return false;
    const json& q = r.req;
    if (q.contains("images") && q["images"].is_array() && !q["images"].empty()) return false;
    if (head) return true;
    const std::string sid = q.value("session", "default");
    return !active_sids.count(sid) && !prefilling_sids.count(sid);
  };
  auto admit_batch = [&](Request&& first) {
    std::vector<std::unique_ptr<PJob>> jobs;
    struct Tag { std::shared_ptr<std::atomic<bool>> flag; std::string sid; };
    auto add = [&](Request&& r) {
      const int fd = r.fd;
      const Tag tag{r.cancel, r.req.value("session", "default")};
      try {
        auto Jp = admit_setup(std::move(r), true);
        if (Jp && Jp->i < Jp->ids.size()) { prefilling_sids.insert(Jp->A->sid); ly_mark(Jp->A->sid, Jp->ids.size()); jobs.push_back(std::move(Jp)); return; }
        if (Jp) admit_finish(*Jp);  // nothing to prefill (regenerating the same prompt)
      } catch (const std::exception& e) {
        admit_error(fd, e, tag.sid);
      }
      admit_epilogue(tag.flag, tag.sid);
    };
    auto end_job = [&](size_t j, bool finish) {  // finish = admit_finish (into active) · otherwise a response was already sent (failure). The slot is nulled and swept at the end of the round
      PJob& J = *jobs[j];
      const Tag tag{J.A->r.cancel, J.A->sid};
      const int fd = J.A->r.fd;
      prefilling_sids.erase(J.A->sid);
      if (finish) {
        try { admit_finish(J); } catch (const std::exception& e) { admit_error(fd, e, tag.sid); }
      }
      jobs[j].reset();
      admit_epilogue(tag.flag, tag.sid);
    };
    auto pull = [&] {
      for (;;) {  // take more: only while seats (max_batch) and slots (at least one per request) remain and the queue head is batchable
        if ((int)(active.size() + jobs.size()) >= rt.max_batch() || (int)jobs.size() >= rt.prefill_slots()) break;
        Request r;
        bool got = false;
        {
          std::lock_guard<std::mutex> lk(mu);
          if (!queue.empty() && stop_state.load() == 0 && !sleep_asked.load() && batch_eligible(queue.front(), false)) { r = std::move(queue.front()); queue.pop_front(); got = true; }
        }
        if (!got) break;
        add(std::move(r));
      }
    };
    add(std::move(first));
    int round = 0;
    bool window_done = !(batch_window_ms > 0 || batch_quiet_ms > 0);  // arrival coalescing uses the same window loop
    while (!jobs.empty()) {
      pull();
      if (jobs.empty()) break;
      // HIVE_BATCH_WINDOW: once before the first round — if the head request is streaming-sized and slots/seats remain, wait for following
            //   requests within the window. Remaining slots = total minus slots_for of each request's chunk this round (min of remaining rows and cap)
            //   — an estimate before planning; the plan below is redone with the regular rules.
      if (!window_done) {
        window_done = true;
        auto slots_left = [&] {
          int s = rt.prefill_slots();
          for (auto& Jp : jobs) if (Jp) s -= rt.slots_for((int)std::min(Jp->ids.size() - Jp->i, cap_now_j(*Jp)));
          return s;
        };
        auto room = [&] { return !jobs.empty() && (int)(active.size() + jobs.size()) < rt.max_batch() && (int)jobs.size() < rt.prefill_slots() && slots_left() > 0; };
        if (jobs.size() == 1 && jobs[0]->ids.size() - jobs[0]->i >= stream_floor && room()) {
          const double tw0 = now_ms(), until = batch_window_ms > 0 ? tw0 + batch_window_ms : INFINITY;
          // arrival coalescing: pause detection (last arrival + batch_quiet_ms) and cap (batch_quiet_ms x slots) — when off (0) both are infinite = plain window
          const double cap_until = batch_quiet_ms > 0 ? tw0 + batch_quiet_ms * rt.prefill_slots() : INFINITY;
          double quiet_until = batch_quiet_ms > 0 ? tw0 + batch_quiet_ms : INFINITY;
          const size_t n0 = jobs.size();
          const char* why = "window";
          while (room()) {
            const double tn = now_ms();
            if (tn >= until) break;
            if (tn >= quiet_until || tn >= cap_until) { why = "quiet"; break; }
            const double wake_at = std::min({until, quiet_until, cap_until});
            {
              std::unique_lock<std::mutex> lk(mu);
              if (stop_state.load() != 0 || sleep_asked.load()) { why = "stop"; break; }
              if (!queue.empty() && !batch_eligible(queue.front(), false)) { why = "blocked"; break; }  // FIFO: if the head cannot join, nothing behind it can
              if (queue.empty() && active.empty()) {  // without decoders, sleep until an arrival (or the end of the window)
                const auto left = std::chrono::microseconds((long long)std::max(0.0, (wake_at - now_ms()) * 1000.0));
                cv.wait_for(lk, left, [&] { return !queue.empty() || stop_state.load() != 0 || sleep_asked.load(); });
              }
            }
            if (!active.empty()) decode_step();  // when there are decoders, keep stepping while waiting
            const size_t before = jobs.size();
            pull();
            if (jobs.size() > before && batch_quiet_ms > 0) quiet_until = now_ms() + batch_quiet_ms;  // arrival coalescing: a new arrival postpones the pause detection
          }
          if (!room() && !strcmp(why, "window")) why = "full";
          fprintf(stderr, "[hived] batch window: +%zu requests in %.0f ms (%s · %zu jobs · slots left %d)\n", jobs.size() - n0, now_ms() - tw0, why,
                  jobs.size(), jobs.empty() ? 0 : slots_left());
          if (jobs.empty()) break;
        }
      }
      // plan: in request order within the remaining slots
      int slots = rt.prefill_slots();
      std::vector<size_t> multi, solo;
      std::vector<size_t> dead;
      const bool sjf_round = batch_sjf && jobs.size() > 1;  // HIVE_BATCH_SJF: when on with two or more jobs, use the allocation below (this loop does not run)
      for (size_t j = 0; !sjf_round && j < jobs.size() && slots > 0; ++j) {
        PJob& J = *jobs[j];
        bool ok = false;
        try { ok = chunk_begin(J, (size_t)slots * (size_t)max_chunk, true); }
        catch (const std::exception& e) { admit_error(J.A->r.fd, e, J.A->sid); }
        if (!ok) { dead.push_back(j); continue; }
        slots -= rt.slots_for(J.M);
        ((size_t)J.M >= stream_floor && J.chunk_images.empty() ? multi : solo).push_back(j);
      }
      if (sjf_round) {
        // HIVE_BATCH_SJF allocation: (1) short requests (remaining rows < stream_floor — their own forward, so they use no batch slots), (2) requests
                //   that can finish this round (remaining rows <= their cap) FIFO, skipped if slots run short, (3) the rest (including those skipped in
                //   (2)) FIFO within the remaining slots.
                //   If the oldest request (jobs[0]) is in (3), (2) leaves one slot for it (it gets at least one slot per round = no starvation).
                //   Chunk sizes follow the regular chunk_begin rules (lim = remaining slots x max_chunk). Only chunks going into the batch (multi) cost slots.
                //   C1: the next-chunk lower bound already assumes chunk_begin's batched rule (one slot) — this allocation never shrinks an admitted chunk
                //   below one slot (or all that remain).
        std::vector<size_t> shorts, fins, rest;
        for (size_t j = 0; j < jobs.size(); ++j) {
          PJob& J = *jobs[j];
          const size_t rem = J.ids.size() - J.i;
          (rem < stream_floor ? shorts : rem <= cap_now_j(J) ? fins : rest).push_back(j);
        }
        const int reserve = !rest.empty() && rest.front() == 0 ? 1 : 0;
        auto plan_one = [&](size_t j, size_t lim) {
          PJob& J = *jobs[j];
          bool ok = false;
          try { ok = chunk_begin(J, lim, true); }
          catch (const std::exception& e) { admit_error(J.A->r.fd, e, J.A->sid); }
          if (!ok) { dead.push_back(j); return; }
          const bool to_multi = (size_t)J.M >= stream_floor && J.chunk_images.empty();
          slots -= to_multi ? rt.slots_for(J.M) : 0;
          (to_multi ? multi : solo).push_back(j);
        };
        for (size_t j : shorts) plan_one(j, (size_t)rt.prefill_slots() * (size_t)max_chunk);
        for (size_t j : fins) {
          const int need = rt.slots_for((int)(jobs[j]->ids.size() - jobs[j]->i));
          if (need <= slots - reserve) plan_one(j, (size_t)(slots - reserve) * (size_t)max_chunk);
          else rest.push_back(j);
        }
        std::sort(rest.begin(), rest.end());
        for (size_t j : rest) if (slots > 0) plan_one(j, (size_t)slots * (size_t)max_chunk);
        std::sort(multi.begin(), multi.end());  // forward_multi part order and solo order = request order
        std::sort(solo.begin(), solo.end());
        std::sort(dead.begin(), dead.end());
      }
      if (multi.size() == 1) { solo.insert(solo.begin(), multi[0]); multi.clear(); }
      std::vector<std::pair<size_t, bool>> ended;  // (job, finish)
      for (size_t j : dead) ended.push_back({j, false});
      double tc_round = 0;
      if (!multi.empty()) {
        std::vector<PrefillPart> ps(multi.size());
        size_t rows = 0;
        for (size_t m = 0; m < multi.size(); ++m) {
          PJob& J = *jobs[multi[m]];
          ps[m].seq = J.S->seq.get(); ps[m].ids = J.ids.data() + J.i; ps[m].M = J.M; ps[m].logits_out = &J.A->logits; ps[m].upper_needed = J.upper_needed;
          rows += (size_t)J.M;
        }
        ForwardStats rs{};
        const double tc0 = now_ms();
        const ly::Stats ly0 = rt.layer_yield_stats();
        {  // HIVE_LAYER_YIELD_MID: this round's rows and what every job of the batch still has after it
          ly_fwd_rows = rows; ly_rest_rows = 0; ly_mid_used = 0;
          for (auto& Jp : jobs) if (Jp) ly_rest_rows += Jp->ids.size() - Jp->i;
          ly_rest_rows -= std::min(ly_rest_rows, rows);
        }
        try {
          ++prefill_epoch;
          rt.forward_multi(ps, &rs);
        } catch (const std::exception& e) {
          fprintf(stderr, "[hived] batch prefill failed: %s — %zu requests retried one by one\n", e.what(), jobs.size());
          rt.quiesce_after_host_error();
          std::deque<Request> back;
          for (auto& Jp : jobs) { prefilling_sids.erase(Jp->A->sid); Request r = std::move(Jp->A->r); r.solo = true; back.push_back(std::move(r)); }
          jobs.clear();
          std::lock_guard<std::mutex> lk(mu);
          for (auto it = back.rbegin(); it != back.rend(); ++it) queue.push_front(std::move(*it));
          return;
        }
        const double tend = now_ms();
        const long ly_n = rt.layer_yield_stats().yields - ly0.yields;
        const double tc = tend - tc0 - (rt.layer_yield_stats().inner_ms - ly0.inner_ms);  // HIVE_LAYER_YIELD: prefill share excluding layer-yield time
        tc_round += tc;
        ++round;
        std::string who;
        for (size_t m = 0; m < multi.size(); ++m) who += (m ? "," : "") + jobs[multi[m]]->A->sid + ":" + std::to_string(ps[m].M);
        char ly_tail[64] = "";
        if (ly_n > 0) snprintf(ly_tail, sizeof ly_tail, " · layer yields %ld last %.0f ms", ly_n, tend - rt.layer_yield_resume_ms());
        fprintf(stderr, "[hived] batch prefill round %d: %zu sequences (%s) · %zu rows · slots %d/%d · hit %d streamed %d cpu %d dma_rows %d · %.0f ms%s\n", round,
                multi.size(), who.c_str(), rows, rt.prefill_slots() - slots, rt.prefill_slots(), rs.n_hit, rs.n_streamed, rs.n_cpu, rs.n_dma_rows, tc, ly_tail);
        for (size_t m = 0; m < multi.size(); ++m) {
          PJob& J = *jobs[multi[m]];
          ForwardStats& st = J.st;  // expert counters are for the whole round (cannot be split per request — log only)
          st.n_routed += rs.n_routed; st.n_hit += rs.n_hit; st.n_cpu += rs.n_cpu; st.n_streamed += rs.n_streamed; st.n_dma_rows += rs.n_dma_rows;
          st.ms_cpu_wait += rs.ms_cpu_wait; st.ms_cpu_span += rs.ms_cpu_span; st.ms_total += rs.ms_total;
          st.tail_rows = ps[m].tail_rows;
          bool cont = false;
          try { cont = chunk_end(J, tc, rows); } catch (const std::exception& e) { admit_error(J.A->r.fd, e, J.A->sid); ended.push_back({multi[m], false}); continue; }
          if (!cont || J.i >= J.ids.size()) ended.push_back({multi[m], true});
        }
      }
      for (size_t j : solo) {
        PJob& J = *jobs[j];
        const double tc0 = now_ms();
        bool cont = false;
        try {
          const ly::Stats ly0 = rt.layer_yield_stats();
          ly_fwd_rows = (size_t)J.M; ly_rest_rows = J.ids.size() - J.i - (size_t)J.M; ly_mid_used = 0;  // HIVE_LAYER_YIELD_MID (solo forward of a batch round)
          ++prefill_epoch;
          rt.forward(*J.S->seq, J.ids.data() + J.i, J.M, J.chunk_images.empty() ? nullptr : &J.chunk_images, &J.A->logits, &J.st, J.upper_needed);
          const double tend = now_ms();
          J.ly_n = rt.layer_yield_stats().yields - ly0.yields;  // HIVE_LAYER_YIELD (same as the single path)
          J.ly_last = J.ly_n > 0 ? tend - rt.layer_yield_resume_ms() : 0;
          const double tc = tend - tc0 - (rt.layer_yield_stats().inner_ms - ly0.inner_ms);
          tc_round += tc;
          cont = chunk_end(J, tc, (size_t)J.M);
        } catch (const std::exception& e) { admit_error(J.A->r.fd, e, J.A->sid); ended.push_back({j, false}); continue; }
        if (!cont || J.i >= J.ids.size()) ended.push_back({j, true});
      }
      last_prefill_tc = tc_round;
      std::sort(ended.begin(), ended.end());  // put into active in request order
      for (auto [j, fin] : ended) end_job(j, fin);
      jobs.erase(std::remove(jobs.begin(), jobs.end(), nullptr), jobs.end());
      if (!jobs.empty() && prefill_yield) yield_after_chunk(tc_round, jobs.size());  // HIVE_PREFILL_YIELD: admit short requests separately before the next round (pull)
      else if (!jobs.empty() && !active.empty()) decode_between(tc_round);
    }
  };

  const int max_batch = rt.max_batch();
  std::vector<Seq*> step_seqs;
  std::vector<int32_t> step_ids, step_next;
  std::vector<std::vector<float>> step_logits;
  // One decode step: sample per row -> send -> stop check -> forward_batch -> clean up finished requests. Also called between prefill chunks so
    // other users do not stall.
    // Emit one token (send, stop and limit checks). Returns whether decoding can continue.
  auto emit = [&](Active& A, int32_t tok) -> bool {
    ++A.n;
    if (A.think_cap > 0) {  // thinking cap: count tokens in the thinking span (closed by </think>, reopened by <think>)
      if (A.in_think) { if (tok == A.think_end) A.in_think = false; else ++A.think_n; }
      else if (tok == A.think_start) A.in_think = true;
    }
    const bool stop = std::find(A.stop_ids.begin(), A.stop_ids.end(), tok) != A.stop_ids.end();
    if (!send_json(A.r.fd, {{"id", tok}})) { A.client_ok = false; return false; }
    if (stop) { A.finish = "stop"; return false; }
    return A.n < A.max_tokens;
  };
  // HIVE_PROFILE=N: host time of decode steps, one [step-host <kind>] line per N steps of a kind (tools/hive_monitor.py). Kinds: spec (one
  //   request, MTP verify) · spec-batch (HIVE_MTP_BATCH verify) · plain (forward_batch, including a declined draft). Per step:
  //   gap   = previous step's end → this step's start, counted only while requests stayed active and no prefill forward or layer-yield run
  //           happened in between (prefill_epoch) — the main loop's host work between steps (n = counted gaps)
  //   pre   = step start → the forward, minus draft time (sampling the pending token, emit/send, deferred checkpoints, gate decisions)
  //   draft = MTP draft calls · fwd = the runtime's forward call (verify / batch — split further by [call-host])
  //   post  = forward end → step end (row sampling and acceptance, rollback, bookkeeping, [mtp] lines, finished requests)
  struct StepHost { long n = 0, gap_n = 0; double gap = 0, pre = 0, draft = 0, fwd = 0, post = 0; };
  static const int step_host_every = getenv("HIVE_PROFILE") ? std::max(0, atoi(getenv("HIVE_PROFILE"))) : 0;
  StepHost step_host[3];
  double sh_draft = 0, sh_f0 = -1, sh_f1 = -1, sh_prev_end = 0;
  int sh_kind = -1, sh_depth = 0;  // sh_depth: only the outermost decode step is accounted
  bool sh_nested = false;  // a nested decode step ran inside the outer one and overwrote its marks — the outer step is not accounted
  long sh_prev_epoch = -1;
  decode_step = [&] {
    // refresh the warm-start file (every 60 s) — at the very top because the speculative path returns from the middle of this function
        // (placed at the end, it was never saved with a single stream)
    if (now_ms() - last_state_save > 60000.0) { save_cache_state(); last_state_save = now_ms(); }
    step_seqs.clear(); step_ids.clear();
    std::vector<Active*> stepping;
    // Priority: only the best priority among the requests that can step runs this step; the others keep their pending token and logits
    //   and resume when it finishes. Measured 2026-10-08 (DeepSeek): one stream 103 tok/s, two overlapping 45.6 tok/s each (a batch of two
    //   runs without speculation) — a background request halved a conversation. Requests without the field are all 0: unchanged.
    int best_prio = INT_MAX;
    for (auto& A : active) if (A->client_ok && !A->cancel->load() && A->n < A->max_tokens) best_prio = std::min(best_prio, A->prio);
    for (auto& A : active) {
      if (!A->client_ok || A->cancel->load() || A->n >= A->max_tokens) { if (A->cancel->load()) A->finish = "cancel"; continue; }
      if (A->prio > best_prio) { ++A->held_steps; continue; }
      int32_t tok = A->pending_tok >= 0 ? A->pending_tok : (fake_head ? sample(A->logits, A->temperature, A->top_p, A->top_k, A->min_p, A->rng)
                                                                    : sample_cands(A->cands, A->logits, A->temperature, A->top_p, A->top_k, A->min_p, A->rng));
      A->pending_tok = -1;
      A->cands.valid = false;
      // Thinking cap: if the thinking-span tokens reached think_cap, turn this token (whether sampled or a pending token left by speculation — not
            //   yet emitted or forwarded) into </think>. This token becomes this step's input, so the KV, the MTP hidden state and the session token
            //   sequence all match the forced token. Drafts emitted inside speculative verification are kept under the cap by the k limit below.
      if (A->think_cap > 0 && A->in_think && A->think_n >= A->think_cap && tok != A->think_end) {
        if (!A->think_forced) fprintf(stderr, "[hived] %s: thinking capped at %d tokens — </think> forced\n", A->sid.c_str(), A->think_cap);
        // With an exit phrase, emit its tokens in order, then </think>. Phrase tokens are thinking-span tokens too, so think_n stays >= cap and the
                //   next step comes here again (the MTP draft count is cap - think_n <= 0, i.e. 0 — the two limits below block it). If the model emits
                //   </think> itself in the middle of the phrase, that closes the span.
        tok = A->think_exit_pos < A->think_exit.size() ? A->think_exit[A->think_exit_pos++] : A->think_end;
        ++A->think_forced;
      }
      if (!emit(*A, tok)) continue;
      stepping.push_back(A.get());
      step_seqs.push_back(A->S->seq.get());
      step_ids.push_back(tok);
    }
    // HIVE_DEFER_CKPT: deferred prompt checkpoint — the first token was already sent above, and before this step's forward (drafts included) seq is still at the prompt end
    if (defer_ckpt) for (auto& A : active) if (A->defer_ckpt || A->defer_warm) run_deferred(*A);
    // ---- speculation expected-value gate (measured: on random-token prompts, 31% acceptance and 2.25 tokens per step, but a ~118 ms verify step gave TPOT 52 ms, slower than the 36 ms baseline) ----
  //   confidence -> acceptance rate (measured over 3,727 steps: [-1,0) 6% · [0,1) 19% · [1,2) 36% · [2,3) 55% · [3,4) 65% · [4,5) 75% · [5,7) 85% · >=7 90%),
  //   cost = draft D + verify V(rows) (online EMA per row count), single step S (EMA of non-speculative steps). Expected tokens E(k) = 1 + sum_i prod_{j<=i} p(c_j).
  //   Speculate only if k = argmax E(k)/(D + V(k+1)) beats 1/S; otherwise rest 8 steps (also saving the 7 ms draft cost).
  auto p_accept = [](float c) -> float {
    return c < 0.f ? 0.06f : c < 1.f ? 0.19f : c < 2.f ? 0.36f : c < 3.f ? 0.55f : c < 4.f ? 0.65f : c < 5.f ? 0.75f : c < 7.f ? 0.85f : 0.90f;
  };
  static double ema_single_ms = 38.0, ema_draft_ms = 7.5;          // non-speculative step and draft cost (EMA)
  static double ema_verify_ms[9] = {0, 0, 66, 81, 93, 108, 124, 138, 152};  // verify cost per row count (EMA, initial values = measured at c1)
  auto ema = [](double& v, double x, double a = 0.1) { v = v <= 0 ? x : v * (1 - a) + x * a; };
  // HIVE_MTP_GATE2 (optional, default off, env_on): instead of the table above, hive/mtp_gate.h — a step-cost table per row count measured
    //   **now** (unmeasured cells interpolated/extrapolated from measured ones) · online acceptance rates · after drafting, only
    //   E(k)/T(k+1) > 1/T(1) is checked (the draft cost is already spent) · per-request net-gain backoff. The gate only decides the number of
    //   verify rows — the acceptance and sampling code below is unchanged (lossless).
    //   Batches (active >= 2) do not speculate here (forward_verify/rollback handle one sequence) — with HIVE_MTP_BATCH the batch block below
    //   uses forward_verify_batch.
  static const bool mtp_gate2 = env_on("HIVE_MTP_GATE2");
  // HIVE_MTP_GATE3 (env_on, default off; layered on GATE2 — no effect without it): if a CUDA graph was captured during the step
    //   (rt.graph_captures() increased), that step's cost sample is not fed to the gate table. Reason (from the code): run_graph captures and
    //   instantiates all 41 layers on the second use of each (layer, rows, Tb) key — the first two verifies with k+1 rows (and again whenever the
    //   context crosses a Tb bucket) carry that cost, while gate 2 takes the first sample as the cell value (n == 0) and does not replace it
    //   during kStale decisions unless the cell is chosen -> one inflated cell blocks that k for a long time (HIVE_MTP_VERIFY2 makes the verify
    //   front end a graph too, so this happens; the size of the inflation is not measured on GPU). Decision rules and table semantics are
    //   those of GATE2 (only samples are filtered).
  static const bool mtp_gate3 = env_on("HIVE_MTP_GATE3");
  static mtpg::Gate gate2;
  if (mtp_gate2 && stepping.size() == 1) gate2.tick();
  // ---- DSpark speculation (one active sequence that has the target-layer hidden state) ----
    //   Of B drafts, the leading k above the confidence threshold are verified as [tok, d1..dk] -> a sample t' is drawn from row i's logits and
        //   accepted if it equals d_{i+1} (t' = d); otherwise t' is the next token (correction) — either way emitted tokens follow the target
        //   distribution p(.|prefix) (independent of the draft distribution, lossless). If all are accepted, row k's sample is a bonus.
    if (stepping.size() == 1 && rt.mtp_enabled() && mtp_max > 0 && stepping[0]->S->seq->mtp_hidden_valid && !fake_head &&
        (mtp_gate2 ? gate2.known() && stepping[0]->g2.want() && (gate2.can_profit(stepping[0]->temperature > 0.f, mtp_max) || gate2.probe_due())
                   : stepping[0]->mtp_skip <= 0)) {  // gate 2: a regular step must be measured once (T(1)) before starting
      // If the table says "no prospect" (can_profit false — even the best learned acceptance gives E(k)/T(k+1) <= 1/T(1)) and it is not a
            //   re-measurement window, do not draft — the post-draft decision (choose_k) would be 0 anyway and only the draft cost D would be paid
            //   (with backoff, D every 32 steps; D 3 ms and T1 20 ms = ~0.45% of decode · tests/test_mtp_gate_cpu (8)).
            //   When a prospect appears (T(1) rises or a verify cell is re-measured cheaper), the same check revives drafting at once. Acceptance and
            //   sampling are unchanged (lossless).
      Active& A = *stepping[0];
      Seq& seq = *A.S->seq;
      const int32_t tok = step_ids[0];
      const double td = now_ms();
      std::vector<int32_t> drafts; std::vector<float> conf;
      ForwardStats dst{};
      rt.mtp_draft(seq, tok, drafts, conf, &dst);
      const double tv = now_ms();
      sh_draft += tv - td;
      draft_routed += dst.n_routed; draft_hit += dst.n_hit; draft_cpu += dst.n_cpu; draft_dma_rows += dst.n_dma_rows; draft_ms += tv - td;
      ema(ema_draft_ms, tv - td);
      if (mtp_gate2) gate2.observe_draft(tv - td);
      // choose k by expected value
      int k = 0;
      double g2_e = 0, g2_ms = 0;
      if (mtp_gate2) k = gate2.choose_k(A.temperature > 0.f, conf.data(), (int)std::min(drafts.size(), conf.size()), mtp_max, mtp_conf, A.max_tokens - A.n, &g2_e, &g2_ms);
      else {
        double best_rate = 1.0 / ema_single_ms, chain = 1.0, expect = 1.0;
        for (int i = 0; i < (int)drafts.size() && i < mtp_max && A.n + i < A.max_tokens; ++i) {
          if (conf[i] < mtp_conf) break;
          chain *= p_accept(conf[i]);
          expect += chain;
          const double rate = expect / (ema_draft_ms + ema_verify_ms[i + 2]);
          if (rate > best_rate) { best_rate = rate; k = i + 1; }
        }
      }
      if (A.think_cap > 0 && A.in_think) k = std::min(k, std::max(0, A.think_cap - A.think_n));  // thinking cap: accepted drafts must not exceed the cap (the next token is forced above)
      if (k == 0) {  // no expected gain -> regular decode (batch path), rest 8 steps
        if (mtp_gate2) A.g2.on_decline(tv - td, !gate2.can_profit(A.temperature > 0.f, mtp_max)); else A.mtp_skip = 8;
        if (trace_mtp && mtp_gate2) fprintf(stderr, "[mtp] pos %lld draft %.1f ms · gate2 no draft (conf0 %.2f · T1 %.1f T2 %.1f T6 %.1f ms · skip %d)\n", (long long)seq.pos, tv - td,
                                            conf.empty() ? 0.f : conf[0], gate2.cost(1), gate2.vcost(2), gate2.vcost(6), A.g2.skip);  // the verify cost (vcost) the decision uses
        else if (trace_mtp) fprintf(stderr, "[mtp] pos %lld draft %.1f ms · no draft (conf0 %.2f, single %.0f ms)\n", (long long)seq.pos, tv - td, conf.empty() ? 0.f : conf[0], ema_single_ms);
      } else {
        std::vector<int32_t> ids(1, tok);
        ids.insert(ids.end(), drafts.begin(), drafts.begin() + k);
        static thread_local std::vector<float> rows;  // reused: a fresh vector per verify step page-faulted while the runtime copied rows × vocab logits into it
        ForwardStats ds{};
        for (int i = 0; i <= k; ++i) rt.row_inv_temp()[i] = A.temperature > 0.f ? 1.f / A.temperature : 1.f;
        const long cap_v = rt.graph_captures();  // HIVE_MTP_GATE3
        sh_f0 = now_ms();
        rt.forward_verify(seq, ids.data(), (int)ids.size(), rows, &ds);
        sh_f1 = now_ms(); sh_kind = 0;
        ema(ema_verify_ms[std::min(8, k + 1)], now_ms() - tv);
        const int V = c.vocab;
        int n_keep = 1, n_cmp = 0;  // n_cmp: drafts compared (for gate 2 acceptance learning)
        int32_t nxt = -1;
        bool cont = true;
        Cands rc;
        auto sample_row = [&](int i) {
          std::vector<float> row(rows.begin() + (size_t)i * V, rows.begin() + (size_t)(i + 1) * V);
          snapshot_cands(rt, i, rc);
          return sample_cands(rc, row, A.temperature, A.top_p, A.top_k, A.min_p, A.rng);
        };
        for (int i = 0; i < k; ++i) {
          const int32_t t = sample_row(i);
          ++n_cmp;
          if (t != drafts[i]) { nxt = t; break; }
          ++n_keep;
          cont = emit(A, t);
          if (!cont) break;
        }
        if (nxt < 0 && cont && n_keep == k + 1) nxt = sample_row(k);  // all accepted: row k's sample is the bonus
        rt.rollback(seq, n_keep);
        if (mtp_gate2) {  // verify step cost = forward_verify + row samples + rollback (regular step T(1) is forward_batch only — its sample is one row at the start of the next step)
          const double tvr = now_ms() - tv;
          if (!mtp_gate3 || rt.graph_captures() == cap_v) gate2.observe_step(k + 1, tvr);  // HIVE_MTP_GATE3: drop samples that included a capture
          gate2.observe_accept(A.temperature > 0.f, conf.data(), n_cmp, n_keep - 1);
          A.g2.on_spec(g2_e, g2_ms, tv - td, gate2.cost(1));
          if (trace_mtp) fprintf(stderr, "[mtp] gate2 k %d · expect %.2f tok / %.1f ms (T1 %.1f) · got %d tok / %.1f ms · net %.1f ms\n", k, g2_e, g2_ms, gate2.cost(1), n_keep, tvr, A.g2.net);
        }
        for (int i = 0; i < n_keep; ++i) A.S->tokens.push_back(ids[i]);
        A.S->last_used = now_ms();
        A.pending_tok = cont ? nxt : -1;
        A.logits.clear();
        ++A.mtp_steps; A.mtp_drafted += k; A.mtp_accepted += n_keep - 1;
        ++tot_mtp_steps; tot_mtp_drafted += k; tot_mtp_accepted += n_keep - 1; tot_mtp_tokens += n_keep;
        A.ds.n_hit += ds.n_hit; A.ds.n_cpu += ds.n_cpu; A.ds.n_dma_rows += ds.n_dma_rows;
        A.ds.ms_cpu_wait += ds.ms_cpu_wait; A.ds.ms_cpu_span += ds.ms_cpu_span; A.ds.ms_total += ds.ms_total;
        tot_routed += ds.n_routed; tot_hit += ds.n_hit; tot_cpu += ds.n_cpu; tot_streamed += ds.n_streamed;
        decode_dma_rows += ds.n_dma_rows;
        if (trace_mtp) {
          std::string cs;
          for (int i = 0; i < (int)conf.size(); ++i) { char b[16]; snprintf(b, sizeof b, "%s%.2f", i ? "," : "", conf[i]); cs += b; }
          fprintf(stderr, "[mtp] pos %lld draft %.1f ms · verify %d rows %.1f ms · accepted %d/%d · conf [%s] · hit %d cpu %d\n", (long long)seq.pos, tv - td,
                  k + 1, now_ms() - tv, n_keep - 1, k, cs.c_str(), ds.n_hit, ds.n_cpu);
        }
        for (size_t i = 0; i < active.size();) {
          Active& X = *active[i];
          const bool done = !X.client_ok || X.cancel->load() || X.finish == "stop" || X.n >= X.max_tokens;
          if (done) { if (X.cancel->load()) X.finish = "cancel"; finish_active(X); active.erase(active.begin() + i); }
          else ++i;
        }
        return;
      }
    }
    // ---- HIVE_MTP_BATCH (env_on, default off): DSpark speculation for batches of >= 2 active requests ----
    //   Part = one active request. Draft for each request with a hidden state (mtp_draft — one sequence at a time; a draft block is B rows, so
        //   several requests cannot share a launch: B*S > 8 = the decode kernel row limit) -> the batch gate (mtpg::choose_batch — measured cost T(M)
        //   per total row count, learned acceptance) picks k per part -> forward_verify_batch (rows = sum (k_s+1) <= rt.mtp_batch_rows(); a request
        //   with k_s = 0 is one row = a regular decode row) -> per part the same acceptance rule as the single path (accept if row i's sample ==
        //   draft d_{i+1}, otherwise that sample is the correction token; if all are accepted row k's sample is a bonus; with k = 0 row 0's sample
        //   is the next token) -> rollback_batch. If the gate picks K = 0, only the draft cost is spent and the regular batch step below runs.
        //   Because of the row limit, c4 gets k <= 1 per request and c8 has no room (the block is skipped when S >= mtp_batch_rows()) — whether
        //   it pays is decided by the gate from measured T(M).
        //   Two details keep the gate from rejecting the block forever after one bad sample:
        //   (1) Sample filtering covers captures **and eager runs** (rt.graph_eager_runs — the first use of a graph key). A first 5-row verify
        //       measured 47.5 ms (T(4) 33 ms) because it was a first use, not a capture, and was not filtered.
        //   (2) Re-measurement (mtpg::batch_probe_due, choose_batch) + a pre-draft prospect check (can_profit_batch — the same place as the single
        //       path's check): while there is no prospect, do not draft (~3.6 ms per part, 25% of a 29 ms c2 step), and once every verify cell is
        //       stale, re-measure once per window. Without it, every "no draft · skip 32" interval still drafted 4 parts (14-17 ms).
        //   Row limit (rt.mtp_batch_rows() = 8 — decode kernel M <= 8): sum(k_s+1) <= 8, so c2 gets a k sum <= 6, c4 <= 4, c8 none -> then a
        //   regular batch step without speculation (absorbed, no rejection).
    static const bool mtp_batch = env_on("HIVE_MTP_BATCH");
    static mtpg::Gate gate_b;
    static mtpg::Backoff back_b;
    if (mtp_batch && stepping.size() >= 2) gate_b.tick();
    bool b_any_sampled = false;
    for (Active* X : stepping) b_any_sampled |= X->temperature > 0.f;
    if (mtp_batch && stepping.size() >= 2 && rt.mtp_batch_enabled() && mtp_max > 0 && !fake_head && (int)stepping.size() < rt.mtp_batch_rows() &&
        (int)stepping.size() <= mtpg::kMaxRows && gate_b.cost((int)stepping.size()) > 0 &&  // T(S) must be measured once before starting
        (mtpg::can_profit_batch(gate_b, (int)stepping.size(), b_any_sampled, mtp_max, rt.mtp_batch_rows()) ||
         mtpg::batch_probe_due(gate_b, (int)stepping.size(), rt.mtp_batch_rows())) &&  // pre-draft prospect / re-measurement window
        back_b.want()) {
      const int S = (int)stepping.size();
      std::vector<std::vector<int32_t>> bdr(S);
      std::vector<std::vector<float>> bcf(S);
      const double td = now_ms();
      int n_drafts_made = 0;
      const bool any_sampled = b_any_sampled;
      for (int s = 0; s < S; ++s) {
        Seq& q = *stepping[s]->S->seq;
        if (!q.mtp_hidden_valid) continue;  // no hidden state -> one row without drafts
        ForwardStats dst{};
        rt.mtp_draft(q, step_ids[s], bdr[s], bcf[s], &dst);
        draft_routed += dst.n_routed; draft_hit += dst.n_hit; draft_cpu += dst.n_cpu; draft_dma_rows += dst.n_dma_rows;
        ++n_drafts_made;
      }
      const double tv = now_ms();
      sh_draft += tv - td;
      draft_ms += tv - td;
      if (n_drafts_made > 0) gate_b.observe_draft((tv - td) / n_drafts_made);
      bool smp[mtpg::kMaxRows];
      const float* cfp[mtpg::kMaxRows];
      int nd[mtpg::kMaxRows], rem[mtpg::kMaxRows];
      for (int s = 0; s < S; ++s) {
        smp[s] = stepping[s]->temperature > 0.f;
        cfp[s] = bcf[s].data();
        nd[s] = (int)std::min(bdr[s].size(), bcf[s].size());
        rem[s] = stepping[s]->max_tokens - stepping[s]->n;
      }
      for (int s = 0; s < S; ++s)  // thinking cap: per-request draft cap (accepted drafts must not exceed the cap — the next token is forced above)
        if (stepping[s]->think_cap > 0 && stepping[s]->in_think) rem[s] = std::min(rem[s], std::max(0, stepping[s]->think_cap - stepping[s]->think_n));
      const mtpg::BatchChoice ch = mtpg::choose_batch(gate_b, S, smp, cfp, nd, mtp_max, mtp_conf, rem, rt.mtp_batch_rows());
      if (ch.K == 0) {
        back_b.on_decline(tv - td, !mtpg::can_profit_batch(gate_b, S, any_sampled, mtp_max, rt.mtp_batch_rows()));
        if (trace_mtp) fprintf(stderr, "[mtp] batch S %d draft %.1f ms · no draft (T(S) %.1f · T(S+1) %.1f ms · skip %d)\n", S, tv - td, gate_b.cost(S), gate_b.cost(S + 1), back_b.skip);
      } else {
        std::vector<std::vector<int32_t>> bids(S);
        std::vector<Runtime::VerifyPart> parts(S);
        std::vector<int> brow0(S), bkeep(S, 1);
        int row = 0;
        for (int s = 0; s < S; ++s) {
          bids[s].assign(1, step_ids[s]);
          bids[s].insert(bids[s].end(), bdr[s].begin(), bdr[s].begin() + ch.k[s]);
          parts[s] = Runtime::VerifyPart{stepping[s]->S->seq.get(), bids[s].data(), (int)bids[s].size()};
          brow0[s] = row;
          for (int i = 0; i <= ch.k[s]; ++i) rt.row_inv_temp()[row++] = stepping[s]->temperature > 0.f ? 1.f / stepping[s]->temperature : 1.f;
        }
        std::vector<float> brows;
        ForwardStats ds{};
        const long cap_v = rt.graph_captures(), eag_v = rt.graph_eager_runs();  // filter captures and first uses
        sh_f0 = now_ms();
        rt.forward_verify_batch(parts, brows, &ds);
        sh_f1 = now_ms(); sh_kind = 1;
        const int V = c.vocab;
        std::vector<int32_t> bnxt(S, -1);
        std::vector<char> bcont(S, 1);
        std::vector<int> bcmp(S, 0);
        for (int s = 0; s < S; ++s) {
          Active& A = *stepping[s];
          Cands rc;
          auto sample_at = [&](int r) {
            std::vector<float> rowv(brows.begin() + (size_t)r * V, brows.begin() + (size_t)(r + 1) * V);
            snapshot_cands(rt, r, rc);
            return sample_cands(rc, rowv, A.temperature, A.top_p, A.top_k, A.min_p, A.rng);
          };
          for (int i = 0; i < ch.k[s]; ++i) {  // same acceptance rule as the single path
            const int32_t t = sample_at(brow0[s] + i);
            ++bcmp[s];
            if (t != bdr[s][i]) { bnxt[s] = t; break; }
            ++bkeep[s];
            bcont[s] = emit(A, t);
            if (!bcont[s]) break;
          }
          if (bnxt[s] < 0 && bcont[s] && bkeep[s] == ch.k[s] + 1) bnxt[s] = sample_at(brow0[s] + ch.k[s]);  // all accepted (or k = 0): sample of the last row
        }
        rt.rollback_batch(bkeep);
        const double tvr = now_ms() - tv;
        if (rt.graph_captures() == cap_v && rt.graph_eager_runs() == eag_v) gate_b.observe_step(row, tvr);  // cell for the total row count (samples that included a capture or first use are dropped)
        int got = 0;
        for (int s = 0; s < S; ++s) { gate_b.observe_accept(smp[s], cfp[s], bcmp[s], bkeep[s] - 1); got += bkeep[s]; }
        back_b.on_spec(ch.e, ch.ms, tv - td, gate_b.cost(S) / S);
        if (trace_mtp) {
          std::string ks;
          for (int s = 0; s < S; ++s) { char b[16]; snprintf(b, sizeof b, "%s%d/%d", s ? "," : "", bkeep[s] - 1, ch.k[s]); ks += b; }
          fprintf(stderr, "[mtp] batch S %d rows %d · accepted [%s] · expect %.2f tok / %.1f ms (T(S) %.1f) · got %d tok / %.1f ms · draft %.1f ms · net %.1f ms\n", S, row, ks.c_str(),
                  ch.e, ch.ms, gate_b.cost(S), got, tvr, tv - td, back_b.net);
        }
        for (int s = 0; s < S; ++s) {
          Active& A = *stepping[s];
          for (int i = 0; i < bkeep[s]; ++i) A.S->tokens.push_back(bids[s][i]);
          A.S->last_used = now_ms();
          A.pending_tok = bcont[s] ? bnxt[s] : -1;
          A.logits.clear();
          if (ch.k[s] > 0) {
            ++A.mtp_steps; A.mtp_drafted += ch.k[s]; A.mtp_accepted += bkeep[s] - 1;
            ++tot_mtp_steps; tot_mtp_drafted += ch.k[s]; tot_mtp_accepted += bkeep[s] - 1; tot_mtp_tokens += bkeep[s];
          }
          for (int r = brow0[s]; r < brow0[s] + ch.k[s] + 1; ++r) {
            A.ds.n_hit += r < (int)ds.row_hit.size() ? ds.row_hit[r] : 0; A.ds.n_cpu += r < (int)ds.row_cpu.size() ? ds.row_cpu[r] : 0;
            A.ds.n_dma_rows += r < (int)ds.row_dma.size() ? ds.row_dma[r] : 0;
          }
          A.ds.ms_cpu_wait += ds.ms_cpu_wait; A.ds.ms_cpu_span += ds.ms_cpu_span; A.ds.ms_total += ds.ms_total;
        }
        tot_routed += ds.n_routed; tot_hit += ds.n_hit; tot_cpu += ds.n_cpu; tot_streamed += ds.n_streamed;
        decode_dma_rows += ds.n_dma_rows;
        for (size_t i = 0; i < active.size();) {
          Active& X = *active[i];
          const bool done = !X.client_ok || X.cancel->load() || X.finish == "stop" || X.n >= X.max_tokens;
          if (done) { if (X.cancel->load()) X.finish = "cancel"; finish_active(X); active.erase(active.begin() + i); }
          else ++i;
        }
        return;
      }
    }
    if (!stepping.empty()) {
      ForwardStats ds{};
      for (size_t i = 0; i < stepping.size(); ++i) rt.row_inv_temp()[i] = stepping[i]->temperature > 0.f ? 1.f / stepping[i]->temperature : 1.f;
      const double tb = now_ms();
      const long cap_b = rt.graph_captures(), eag_b = rt.graph_eager_runs();  // HIVE_MTP_GATE3 · HIVE_MTP_BATCH (first uses as well)
      sh_f0 = tb;
      rt.forward_batch(step_seqs, step_ids.data(), step_next, &step_logits, &ds);
      sh_f1 = now_ms(); sh_kind = 2;
      const bool cap_clean = rt.graph_captures() == cap_b;
      const bool eag_clean = rt.graph_eager_runs() == eag_b;
      if (stepping.size() == 1) { ema(ema_single_ms, now_ms() - tb); if (stepping[0]->mtp_skip > 0) --stepping[0]->mtp_skip; if (mtp_gate2 && (!mtp_gate3 || cap_clean)) gate2.observe_step(1, now_ms() - tb); }
      else if (mtp_batch && cap_clean && eag_clean) gate_b.observe_step((int)stepping.size(), now_ms() - tb);  // batch gate T(S) (samples that included a capture or first use are dropped)
      for (size_t i = 0; i < stepping.size(); ++i) {
        Active& A = *stepping[i];
        if (i < step_logits.size()) A.logits = std::move(step_logits[i]); else A.logits.clear();
        if (!fake_head) snapshot_cands(rt, (int)i, A.cands);
        fake_logits(A.logits, (uint64_t)A.S->tokens.size() * 7919 + (uint64_t)step_ids[i]);
        A.S->tokens.push_back(step_ids[i]);
        A.S->last_used = now_ms();
        A.ds.n_hit += ds.row_hit.at(i); A.ds.n_cpu += ds.row_cpu.at(i); A.ds.n_dma_rows += ds.row_dma.at(i);
        A.ds.ms_cpu_wait += ds.ms_cpu_wait; A.ds.ms_cpu_span += ds.ms_cpu_span; A.ds.ms_total += ds.ms_total;
      }
      tot_routed += ds.n_routed; tot_hit += ds.n_hit; tot_cpu += ds.n_cpu; tot_streamed += ds.n_streamed;
      decode_dma_rows += ds.n_dma_rows;
    }
    for (size_t i = 0; i < active.size();) {
      Active& A = *active[i];
      const bool done = !A.client_ok || A.cancel->load() || A.finish == "stop" || A.n >= A.max_tokens;
      if (done) { if (A.cancel->load()) A.finish = "cancel"; finish_active(A); active.erase(active.begin() + i); }
      else ++i;
    }
  };
  // Wrap the callable itself, so both the main loop and interleaved prefill decode
  // use the same failure boundary. CUDA_CHECK remains fatal for a broken context.
  auto decode_inner = std::move(decode_step);
  auto step_host_end = [&](double t_in, double gap) {  // [step-host] accounting of the step that just returned (see StepHost)
    const double t_out = now_ms();
    if (sh_kind >= 0) {
      StepHost& h = step_host[sh_kind];
      ++h.n; h.pre += sh_f0 - t_in - sh_draft; h.draft += sh_draft; h.fwd += sh_f1 - sh_f0; h.post += t_out - sh_f1;
      if (gap >= 0) { ++h.gap_n; h.gap += gap; }
      if (h.n >= step_host_every) {
        static const char* const names[3] = {"spec", "spec-batch", "plain"};
        const double n = (double)h.n;
        fprintf(stderr, "[step-host %s] steps %ld · gap %.3f (n %ld) · pre %.3f · draft %.3f · fwd %.3f · post %.3f ms/step\n", names[sh_kind], h.n,
                h.gap_n ? h.gap / h.gap_n : 0.0, h.gap_n, h.pre / n, h.draft / n, h.fwd / n, h.post / n);
        h = StepHost{};
      }
    }
    sh_prev_end = active.empty() ? 0 : t_out;
    sh_prev_epoch = prefill_epoch;
  };
  decode_step = [&] {
    struct Depth { int& d; ~Depth() { --d; } } depth{++sh_depth};
    const bool sh_on = step_host_every > 0 && sh_depth == 1;
    double t_in = 0, gap = -1;
    if (sh_on) {
      t_in = now_ms();
      if (sh_prev_end > 0 && sh_prev_epoch == prefill_epoch) gap = t_in - sh_prev_end;
      sh_draft = 0; sh_f0 = sh_f1 = -1; sh_kind = -1; sh_nested = false;
    } else if (step_host_every > 0) sh_nested = true;
    try {
      decode_inner();
      if (sh_on && !sh_nested) step_host_end(t_in, gap);
      else if (sh_on) { sh_prev_end = active.empty() ? 0 : now_ms(); sh_prev_epoch = prefill_epoch; }
    }
    catch (const std::exception& e) {
      fprintf(stderr, "[hived] decode failed: %s\n", e.what());
      rt.quiesce_after_host_error();
      for (auto& a : active) {
        if (a->S && a->S->seq) a->S->seq->broken = true;
        send_json(a->r.fd, {{"error", std::string("decode failed: ") + e.what()}});
        a->client_ok = false;
        finish_socket(a->r.fd);
        active_sids.erase(a->sid);
        std::lock_guard<std::mutex> lk(mu);
        auto it = cancel_flags.find(a->sid);
        if (it != cancel_flags.end() && it->second.get() == a->cancel) cancel_flags.erase(it);
      }
      active.clear();
    }
  };
  // ---- sleep / wake (engine thread) ----------------------------------------------------------------------------------------------------------
  // Flow (the caller typically holds its own request proxy while calling): sleep -> draining (new requests stay queued, in-flight requests run
    //   to completion — nothing is cut off or cancelled; waiting has no upper bound unless timeout_s is given) -> once nothing is running, save the
    //   resident keys (+ the warm-start file) -> rt.sleep_release (release VRAM) -> sleeping. wake -> waking (re-allocate slots and staging ->
    //   re-bind the elastic span -> reload the saved resident keys) -> ready -> take the queue.
  // Queue policy (absorb instead of reject): generate requests arriving while asleep are not refused but queued (taken as-is on wake — the same
    //   meaning as a proxy hold; refusing would surface a failure to the client). Cancelled queued requests finish within 1 s even while asleep
    //   (finish "cancel"). No automatic wake — the sleep window belongs to another process using the GPU.
  // Responses (one JSON line): sleep -> {"ok":true,"state":"sleeping","sleep":{...}} · already asleep: same shape (idempotent) · timeout_s exceeded
    //   -> {"error":"not idle","state":...,"running":n,"retryable":true} (the sleep is withdrawn — back to ready if no other sleep waiter) · a wake
    //   withdrawing a sleep -> {"error":"sleep cancelled by a wake request"}.
    //   wake -> {"ok":true,"state":"ready","wake":{...}} · already awake: same shape (idempotent) · VRAM shortage -> {"error":"wake failed: ...","state":"sleeping","retryable":true}.
  // ---- HIVE_LAYER_YIELD yield body (called by the runtime at layer boundaries of a long prefill forward — runtime.cpp, hive/layer_yield.h) ----
  //   want(): is there work — 1 if decoders are running · the prompt id count (largest) of admissible short text requests · otherwise 0 (the runtime then does not hand over).
  //   run(R, gap): (1) admit — from the queue head (FIFO; inadmissible requests are skipped and keep their place) requests with prompt id count
    //     <= min(ly_max, R), no images, session not running/prefilling, a free seat (active + prefilling + 1 <= max_batch), no stop/sleep request
    //     -> single-path admit (prefill = forward rows <= R — the runtime lent only the first R Work rows of the outer prefill, so no larger forward
    //     is called). (2) decode — if decoders are running, decode_step at least once, up to the budget (preceding prefill gap x share/(1-share))
    //     and the cap (ly_steps). The first token of newly admitted requests also leaves here.
    //   No recursion: in_yield (same flag as HIVE_PREFILL_YIELD) — the runtime does not call again from a forward inside a yield (ly_.depth).
    //   Warming (warm_after_prefill) is skipped.
    //   Logs: header line "[hived] layer yield: <prefilling sid> · prefill X ms since resume ..." (counted as a gap by monitors) -> admit line
    //   (same format as HIVE_PREFILL_YIELD + " · layer") -> done line.
  if (layer_yield) {
    auto h_nested_ok = [&] { return ly_mid > ly_max && rt.in_layer_yield(); };  // in_yield from a layer yield (not HIVE_PREFILL_YIELD) with level 2 on
    auto ly_mid_ok = [&](size_t rows) {  // HIVE_LAYER_YIELD_MID (see its comment): much smaller than the paused forward's remaining rows
      if (ly_mid <= ly_max || rows <= ly_max || rows > ly_mid) return false;
      const double p = std::min(1.0, std::max(0.0, rt.layer_yield_progress()));
      const double rem = (double)ly_fwd_rows * (1.0 - p) + (double)ly_rest_rows;
      return 2.0 * (double)rows <= rem && ly_mid_used + rows <= ly_fwd_rows;
    };
    auto ly_rows_of = [&, ly_mid_ok](const Request& r, size_t cap) -> size_t {
      const json& q = r.req;
      if (q.contains("images") && q["images"].is_array() && !q["images"].empty()) return 0;  // do not lazily load the vision encoder (VRAM) in the middle of a long prefill
      const std::string sid = q.value("session", "default");
      if (active_sids.count(sid) || prefilling_sids.count(sid)) return 0;
      const size_t rows = rows_to_prefill(sid, q);  // ids minus the reusable prefix (lower bound): what the forward will really hold
      if (rows == 0 || rows > cap) return 0;
      if (rows > ly_max && !ly_mid_ok(rows)) return 0;
      return rows;
    };
    auto ly_seat = [&] { return (int)(active.size() + prefilling_sids.size()) + 1 <= rt.max_batch(); };
    ly::Hooks h;
    h.period_ms = ly_period;
    // HIVE_LAYER_YIELD_MID: a mid-size prompt admitted at a yield prefills for seconds — its own forward yields once more (level 2), for decode
    //   steps only (measured without it: the running decoder's longest stall 1.83 → 5.13 s while a 12K-row prompt was admitted into an 85K prefill)
    h.max_depth = ly_mid > ly_max ? 2 : 1;
    h.want = [&, ly_rows_of, ly_seat, h_nested_ok]() -> int {
      if (in_yield) return h_nested_ok() && !active.empty() ? 1 : 0;
      int need = active.empty() ? 0 : 1;
      if (ly_max > 0 && ly_seat()) {
        std::lock_guard<std::mutex> lk(mu);
        if (stop_state.load() == 0 && !sleep_asked.load())
          for (const auto& r : queue) need = std::max(need, (int)std::min<size_t>(ly_rows_of(r, std::max(ly_max, ly_mid)), (size_t)INT32_MAX));
      }
      return need;
    };
    h.run = [&, ly_rows_of, ly_seat](int R, double gap_ms) {
      const bool nested = in_yield;  // level 2 (inside a forward admitted by a yield): decode steps only
      struct InYield { bool& f; bool was; ~InYield() { f = was; } } in_yield_guard{in_yield, in_yield};
      in_yield = true;
      ++prefill_epoch;  // prefill layers ran since the last decode step
      const double t0 = now_ms();
      const double saved_tc = last_prefill_tc;  // HIVE_PREFILL_FAIR looks at the outer admission's last prefill forward
      std::string who;
      for (const auto& sid : prefilling_sids) {
        const auto it = ly_rows.find(sid);
        who += (who.empty() ? "" : ",") + sid + ":" + (it != ly_rows.end() ? std::to_string(it->second) : std::string("?"));
      }
      fprintf(stderr, "[hived] layer yield: %s · prefill %.0f ms since resume · rows %d · active %zu%s\n", who.empty() ? "-" : who.c_str(), gap_ms, R, active.size(),
              nested ? " · level 2 (decode only)" : "");
      const size_t cap = nested ? 0 : std::min(std::max(ly_max, ly_mid), (size_t)std::max(0, R));
      int n = 0, n_mid = 0;
      while (cap > 0 && ly_seat()) {
        Request r;
        bool got = false;
        size_t rows = 0;
        {
          std::lock_guard<std::mutex> lk(mu);
          if (stop_state.load() != 0 || sleep_asked.load()) break;
          for (auto it = queue.begin(); it != queue.end(); ++it)
            if ((rows = ly_rows_of(*it, cap)) > 0) { r = std::move(*it); queue.erase(it); got = true; break; }
        }
        if (!got) break;
        if (rows > ly_max) { ly_mid_used += rows; ++n_mid; }  // HIVE_LAYER_YIELD_MID
        admit(std::move(r));
        ++n;
      }
      const double ta = now_ms();
      if (n) fprintf(stderr, "[hived] prefill yield: %d short request%s admitted in %.0f ms (active %zu · prefilling %zu) · layer%s\n", n, n > 1 ? "s" : "", ta - t0,
                     active.size(), prefilling_sids.size(), n_mid ? (" · " + std::to_string(n_mid) + " by the mid rule").c_str() : "");
      int steps = 0;
      if (!active.empty()) {
        const double budget = ly::decode_budget_ms(gap_ms, ly_share);
        do { decode_step(); ++steps; } while (!active.empty() && steps < ly_steps && now_ms() - ta < budget);
      }
      last_prefill_tc = saved_tc;
      fprintf(stderr, "[hived] layer yield done: admitted %d · decode steps %d · %.0f ms\n", n, steps, now_ms() - t0);
    };
    rt.set_layer_yield(std::move(h));
  }
  struct SleepWaiter { int fd; double deadline; int level; };
  std::vector<SleepWaiter> sleep_waiters;
  double drain_t0 = 0;
  std::string wake_why;
  auto ctl_reply = [&](int fd, const json& j) { send_json(fd, j); finish_socket(fd); };
  auto recompute_sleep_asked = [&] {
    std::lock_guard<std::mutex> lk(mu);
    bool pending = false;
    for (const auto& c : ctl_q) pending = pending || c.sleep;
    sleep_asked.store(power.load() != kPowerReady || pending);
  };
  // level 2: session KV to RAM images — the same code as pool eviction (capture the last state as a live image, Seq back to the pool), then
    //   free every Seq in the pool. The next request continues exactly via the "session returning from outside the pool" path (load_image —
    //   prompt-end checkpoint, last state) (test_daemon_cpu archive restore).
  auto offload_sessions = [&]() -> int {
    int n = 0;
    {
      std::lock_guard lk(mu);
      for (auto& [id, V] : sessions) {
        if (!V.seq) continue;
        if (!V.tokens.empty() && !V.seq->broken && V.seq->pos == (int64_t)V.tokens.size()) {
          V.live = std::make_shared<SeqImage>();
          rt.save_image(*V.seq, *V.live, V.ckpt.get());
          V.live_sig = V.img_sig;
        }
        seq_pool.push_back(std::move(V.seq));
        ++n;
      }
      archive_trim("");
    }
    seq_pool.clear();  // the Seq destructor waits on the snapshot fence before freeing device buffers
    pool_released = true;
    sessions_offloaded = n;
    return n;
  };
  auto sleep_report = [&](double t0, size_t fr0, size_t freed, bool deepen) {
    size_t fr1 = 0, tot = 0;
    cudaMemGetInfo(&fr1, &tot);
    const double MiB = 1048576.0, used = (double)(tot - fr1);
    const double pool = sleep_level >= 2 ? 0.0 : (double)session_pool_bytes;
    const double model = sleep_level >= 3 ? 0.0 : (double)vram_model_used;  // level 3: dense weights go to host copies too (what remains is the CUDA context and residue)
    const double other = std::max(0.0, used - model - pool);
    json pm = proc_memory_mib();
    pm["experts_pinned_mib"] = store.experts_host_bytes() / MiB;
    pm["engram_mib"] = store.engram_host_bytes() / MiB;
    pm["dense_host_copy_mib"] = devmem::shadow_bytes() / MiB;
    sleep_info = {{"level", sleep_level}, {"ms", now_ms() - t0}, {"drain_ms", t0 - drain_t0}, {"resident_keys", sleep_keys.size()},
                  {"slots_released", store.sleep_phys_slots()}, {"sessions_offloaded", sleep_level >= 2 ? sessions_offloaded : 0},
                  {"freed_mib", fr1 > fr0 ? (fr1 - fr0) / MiB : 0.0}, {"store_freed_mib", freed / MiB}, {"vram_used_mib", used / MiB},
                  {"vram_scope", "device-wide cudaMemGetInfo — exact for hived only while no other process holds VRAM (per-process: nvidia-smi --query-compute-apps)"},
                  {"vram_kept", {{"model_weights_and_cuda_context_mib", model / MiB}, {"session_pool_mib", pool / MiB},
                                 {"runtime_decode_buffers_and_rest_mib", other / MiB}}},
                  {"host_memory", pm}};
    if (!vmm_info.empty()) sleep_info["vmm"] = vmm_info;
    // Line format (contract with the "sleep" regex in tools/hive_monitor.py): "[hived] sleep: <ms> ms (drain <ms> ms) · freed <MiB> MiB (...)" —
        //   deepening while already asleep is not a new sleep, so it logs "[hived] sleep deepened: ..." (not starting with sleep:)
    const std::string tail = (sleep_level >= 2 ? " · " + std::to_string(sessions_offloaded) + " sessions to RAM" : std::string()) +
                             (sleep_level >= 3 ? " · dense/work " + std::to_string((long long)(vmm_info.value("released_mib", 0.0))) + " MiB to RAM" : std::string());
    if (deepen)
      fprintf(stderr, "[hived] sleep deepened (level %d): %.0f ms · freed %.0f MiB%s · VRAM used %.0f MiB = model+ctx %.0f + sessions %.0f + runtime/rest %.0f\n",
              sleep_level, now_ms() - t0, fr1 > fr0 ? (fr1 - fr0) / MiB : 0.0, tail.c_str(), used / MiB, model / MiB, pool / MiB, other / MiB);
    else
      fprintf(stderr, "[hived] sleep: %.0f ms (drain %.0f ms) · freed %.0f MiB (level %d · store %.0f MiB · %d slots · %zu resident keys saved%s) · VRAM used %.0f MiB = "
              "model+ctx %.0f + sessions %.0f + runtime/rest %.0f · host RSS %.0f MiB (experts pinned %.0f · engram %.0f · dense copy %.0f MiB)\n", now_ms() - t0,
              t0 - drain_t0, fr1 > fr0 ? (fr1 - fr0) / MiB : 0.0, sleep_level, freed / MiB, store.sleep_phys_slots(), sleep_keys.size(), tail.c_str(), used / MiB,
              model / MiB, pool / MiB, other / MiB, pm.value("VmRSS_mib", 0.0), pm.value("experts_pinned_mib", 0.0), pm.value("engram_mib", 0.0),
              pm.value("dense_host_copy_mib", 0.0));
  };
  // Deepen the sleep up to level: 2 = session KV -> RAM images · 3 = + all VMM regions (dense weights, small Work slabs, cuBLAS workspace, runtime
    //   residue) -> pinned host copies, keeping the same VA reserved (hive/devmem.h). Returns the absorbed reason ("" = reached the requested depth)
    //   — if level 3 is not possible it stays asleep at level 2 and only reports why (not a refusal).
  auto deepen = [&](int level) -> std::string {
    if (level >= 2 && sleep_level < 2) { offload_sessions(); sleep_level = 2; }
    if (level >= 3 && sleep_level < 3) {
      if (!devmem::on()) return "level 3 needs VMM device memory (start hived with HIVE_SLEEP_VMM=1) — asleep at level 2";
      CUDA_CHECK(cudaDeviceSynchronize());
      const devmem::Report r = devmem::release_all();
      vmm_info = {{"released_mib", r.bytes / 1048576.0}, {"copied_mib", r.copied / 1048576.0}, {"regions", r.regions}, {"ms", r.ms},
                  {"shadow_new_mib", r.shadow_new / 1048576.0}, {"shadow_alloc_ms", r.shadow_ms}};  // faster first sleep: shadow copies allocated this time (~0 if pre-pinned)
      if (!r.ok) { vmm_info["error"] = r.error; return "level 3 release failed: " + r.error + " — asleep at level 2"; }
      sleep_level = 3;
    }
    return "";
  };
  auto do_sleep = [&] {
    const double t0 = now_ms();
    size_t fr0 = 0, tot = 0;
    cudaMemGetInfo(&fr0, &tot);
    int level = 1;
    for (auto& w : sleep_waiters) level = std::max(level, w.level);
    try {
      sleep_keys = store.resident_keys_by_score();
      save_cache_state();  // also the warm-start file (a restart while asleep comes up with the same resident list)
      last_state_save = now_ms();
      const size_t freed = rt.sleep_release();
      power.store(kPowerSleeping);
      sleep_level = 1;
      vmm_info = json::object();
      const std::string note = deepen(level);
      sleep_report(t0, fr0, freed, false);
      if (!note.empty()) { sleep_info["note"] = note; fprintf(stderr, "[hived] sleep: %s\n", note.c_str()); }
      publish_stats(true);
      for (auto& w : sleep_waiters) ctl_reply(w.fd, {{"ok", true}, {"state", "sleeping"}, {"sleep", sleep_info}});
    } catch (const std::exception& e) {
      // If the pre-sleep invariant holds (head of runtime sleep_release — nothing was freed), stay awake. An exception while offloading sessions
            //   (host memory etc.) means it is already asleep — keep it asleep (wake re-allocates the session pool) and tell the waiters.
      const bool asleep = rt.asleep();
      fprintf(stderr, "[hived] sleep failed: %s — %s\n", e.what(), asleep ? "asleep" : "staying awake");
      if (asleep) { if (!seq_pool.empty() && level >= 2) { seq_pool.clear(); pool_released = true; } power.store(kPowerSleeping); }
      else power.store(kPowerReady);
      for (auto& w : sleep_waiters) ctl_reply(w.fd, {{"error", std::string("sleep failed: ") + e.what()}, {"state", power_name(power.load())}});
    }
    sleep_waiters.clear();
    recompute_sleep_asked();
  };
  auto do_wake = [&]() -> bool {
    const double t0 = now_ms();
    power.store(kPowerWaking);
    wake_info = {{"phase", "alloc"}};
    publish_stats(true);
    int n = -1;
    wake_why.clear();
    bool vmm_restored = false;
    double vmm_ms = 0, vmm_mib = 0;
    if (sleep_level >= 3) {  // level 3: dense weights, Work and cuBLAS back at the same VA first (everything else uses them)
      const devmem::Report r = devmem::restore_all();
      if (!r.ok) wake_why = r.error;
      else { vmm_restored = true; vmm_ms = r.ms; vmm_mib = r.bytes / 1048576.0; }
    }
    if (wake_why.empty() && pool_released) {  // level 2: re-create the session pool at its startup size (first — same order as startup: session pool -> slots). DevBuf aborts on failure, so check the headroom first
      size_t fr = 0, tot = 0;
      cudaMemGetInfo(&fr, &tot);
      const size_t need = session_pool_bytes + (size_t)(store.staging_slots() + store.elastic_slots()) * store.layout().total;  // pool + mandatory regions (staging, elastic)
      if (session_pool_bytes > 0 && fr < need) {
        wake_why = "not enough free VRAM for the session pool + staging/prefill work region (free " + std::to_string((long long)(fr >> 20)) + " MiB, need " +
                   std::to_string((long long)(need >> 20)) + " MiB)";
      } else {
        for (int i = 0; i < max_sessions; ++i) seq_pool.push_back(rt.new_seq());
        size_t fr1 = 0;
        cudaMemGetInfo(&fr1, &tot);
        if (session_pool_bytes == 0 && fr > fr1) session_pool_bytes = fr - fr1;  // started asleep: measured at the first wake
      }
    }
    if (wake_why.empty()) {
      if (start_asleep_pending) {  // first wake after starting asleep: allocate the slot region for the first time (startup order — after the session pool; with HIVE_CACHE_FIT from the headroom at that point)
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        const size_t rec = store.layout().total;
        n = store.alloc_cache(cache_fit ? ExpertStore::fit_slots(fr, cache_reserve, rec) : (int)(cache_mb * 1048576.0 / (double)rec));
        start_asleep_pending = false;
      } else {
        try { n = rt.wake_restore(cache_fit ? cache_reserve : 0, &wake_why); }
        catch (const std::exception& e) { wake_why = e.what(); n = -1; }
      }
      if (n < 0 && pool_released) seq_pool.clear();  // go back to sleep (level 2 VRAM unchanged)
    }
    if (n < 0) {
      if (vmm_restored) {  // return to level 3 (the next wake starts over — the shadow copies are reused)
        CUDA_CHECK(cudaDeviceSynchronize());
        const devmem::Report r = devmem::release_all();
        if (!r.ok) { sleep_level = 2; fprintf(stderr, "[hived] wake: could not return to level 3 (%s) — asleep at level 2\n", r.error.c_str()); }
      }
      power.store(kPowerSleeping);
      wake_info = {{"phase", "failed"}, {"error", wake_why}, {"ms", now_ms() - t0}};
      fprintf(stderr, "[hived] wake failed: %s — still sleeping (retry wake when the VRAM is free)\n", wake_why.c_str());
      publish_stats(true);
      return false;
    }
    const double t1 = now_ms();
    wake_info = {{"phase", "warm"}, {"alloc_ms", t1 - t0}, {"slots", n}, {"slots_before_sleep", store.sleep_phys_slots()}, {"warm_done", 0},
                 {"warm_total", sleep_keys.size()}};
    publish_stats(true);
    int warmed = 0;
    try {
      warmed = rt.wake_warm(sleep_keys, [&](int done, int total) { wake_info["warm_done"] = done; wake_info["warm_total"] = total; publish_stats(true); });
    } catch (const std::exception& e) {  // warming only fills the cache (lossless) — on failure the engine still comes up awake (promotion fills the empty slots)
      fprintf(stderr, "[hived] wake: warm stopped: %s (serving with a partly filled cache)\n", e.what());
    }
    const int level_was = sleep_level;
    power.store(kPowerReady);
    sleep_level = 0;
    sessions_offloaded = 0;
    pool_released = false;
    const double t2 = now_ms();
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    wake_info = {{"phase", "done"}, {"level", level_was}, {"ms", t2 - t0}, {"alloc_ms", t1 - t0}, {"warm_ms", t2 - t1}, {"slots", n},
                 {"slots_before_sleep", store.sleep_phys_slots()}, {"warmed", warmed}, {"resident_keys", sleep_keys.size()},
                 {"vram_used_mib", (double)(tot - fr) / 1048576.0}};
    if (vmm_restored) wake_info["vmm"] = {{"restored_mib", vmm_mib}, {"ms", vmm_ms}};
    fprintf(stderr, "[hived] wake: %.0f ms (alloc %.0f · warm %.0f) · slots %d (before sleep %d) · warmed %d/%zu experts · VRAM used %.0f MiB · level %d%s\n", t2 - t0,
            t1 - t0, t2 - t1, n, store.sleep_phys_slots(), warmed, sleep_keys.size(), (double)(tot - fr) / 1048576.0, level_was,
            vmm_restored ? (" · dense/work " + std::to_string((long long)vmm_mib) + " MiB back in " + std::to_string((long long)vmm_ms) + " ms").c_str() : "");
    publish_stats(true);
    return true;
  };
  // {"op":"flush"}: at the top of the engine loop no prefill is in progress (admit runs a request's chunks to the end), so only decoding
  //   requests can hold session state. With any of them running the request is refused ("not idle", like SGLang's /flush_cache refusal)
  //   instead of waiting — benchmark clients flush between runs, when nothing runs. Sessions keep their pooled device buffers; clearing
  //   their tokens makes the next request start from a reset (pick 0), exactly as a never-seen conversation.
  auto handle_flush = [&] {
    std::vector<int> fds;
    { std::lock_guard<std::mutex> lk(mu); fds.swap(flush_q); }
    if (fds.empty()) return;
    if (!active.empty()) {
      for (int fd : fds) ctl_reply(fd, {{"error", "not idle"}, {"running", active.size()}, {"retryable", true}});
      return;
    }
    size_t n_sess = 0;
    const size_t n_arch = archived.size(), n_snap = shared_snaps.size();
    {
      std::lock_guard<std::mutex> lk(mu);
      for (auto& [id, V] : sessions) {
        if (V.tokens.empty() && !V.ckpt && !V.live && V.history.empty()) continue;
        V.tokens.clear(); V.history.clear(); V.ckpt.reset(); V.live.reset();
        std::vector<float>().swap(V.ckpt_logits);
        V.img_sig.clear(); V.ckpt_sig.clear(); V.live_sig.clear();
        ++n_sess;
      }
      archived.clear();
      shared_snaps.clear();
    }
    rt.trim_snapshot_pool();
    host_total();
    fprintf(stderr, "[hived] flush: cleared %zu session(s) · %zu archived · %zu shared snapshot(s) (expert cache kept)\n", n_sess, n_arch, n_snap);
    for (int fd : fds) ctl_reply(fd, {{"ok", true}, {"sessions", n_sess}, {"archived", n_arch}, {"snapshots", n_snap}});
    publish_stats(true);
  };
  auto handle_ctl = [&] {
    std::deque<Ctl> todo;
    { std::lock_guard<std::mutex> lk(mu); todo.swap(ctl_q); }
    if (todo.empty()) return;
    for (auto& c : todo) {
      if (!c.sleep) {
        if (power.load() == kPowerDraining) {  // a wake withdraws a sleep in progress (in-flight requests continue)
          for (auto& w : sleep_waiters) ctl_reply(w.fd, {{"error", "sleep cancelled by a wake request"}, {"state", "ready"}});
          sleep_waiters.clear();
          power.store(kPowerReady);
          fprintf(stderr, "[hived] sleep withdrawn by a wake request\n");
        }
        if (power.load() == kPowerSleeping && !do_wake()) {
          ctl_reply(c.fd, {{"error", "wake failed: " + wake_why}, {"state", "sleeping"}, {"retryable", true}});
          continue;
        }
        ctl_reply(c.fd, {{"ok", true}, {"state", power_name(power.load())}, {"wake", wake_info}});
      } else {
        if (power.load() == kPowerSleeping) {
          if (c.level > sleep_level) {  // deeper sleep requested while asleep: run only the missing stages
            const double t0 = now_ms();
            size_t fr0 = 0, tot = 0;
            cudaMemGetInfo(&fr0, &tot);
            std::string note;
            try { note = deepen(c.level); sleep_report(t0, fr0, 0, true); }
            catch (const std::exception& e) {
              if (!seq_pool.empty()) { seq_pool.clear(); pool_released = true; }
              ctl_reply(c.fd, {{"error", std::string("sleep failed: ") + e.what()}, {"state", "sleeping"}});
              continue;
            }
            if (!note.empty()) sleep_info["note"] = note;
          }
          ctl_reply(c.fd, {{"ok", true}, {"state", "sleeping"}, {"sleep", sleep_info}});
          continue;
        }
        sleep_waiters.push_back({c.fd, c.deadline, c.level});
        if (power.load() == kPowerReady) {
          power.store(kPowerDraining);
          drain_t0 = now_ms();
          size_t qn = 0;
          { std::lock_guard<std::mutex> lk(mu); qn = queue.size(); }
          fprintf(stderr, "[hived] sleep requested: draining %zu in-flight request(s) · %zu queued stay queued until wake\n", active.size(), qn);
        }
      }
    }
    recompute_sleep_asked();
    publish_stats(true);
  };
  // Queued requests cancelled while draining or asleep finish immediately (do not hold the connection until wake) — never admitted, so zero prefill/decode
  auto sweep_cancelled_queue = [&] {
    std::deque<Request> dead;
    {
      std::lock_guard<std::mutex> lk(mu);
      for (auto it = queue.begin(); it != queue.end();) {
        if (it->cancel && it->cancel->load()) {
          auto f = cancel_flags.find(it->req.value("session", "default"));
          if (f != cancel_flags.end() && f->second == it->cancel) cancel_flags.erase(f);
          dead.push_back(std::move(*it));
          it = queue.erase(it);
        } else ++it;
      }
    }
    for (auto& r : dead) {
      send_json(r.fd, {{"done", true}, {"n", 0}, {"finish", "cancel"}, {"prefill_ms", 0.0}, {"decode_ms", 0.0}, {"cached_prefix", 0}, {"prefill_tokens", 0}});
      finish_socket(r.fd);
    }
  };
  for (;;) {
    publish_stats();
    if (stop_state.load() != 0) {  // graceful shutdown (enabled only with HIVE_GRACEFUL_STOP_S>0): refuse the queue, run in-flight requests to completion or cancel at the deadline
      {  // sleep/wake waiters also get the shutdown error (while draining, in-flight requests finish under the rule below)
        std::deque<Ctl> ctl;
        { std::lock_guard<std::mutex> lk(mu); ctl.swap(ctl_q); }
        for (auto& c : ctl) ctl_reply(c.fd, {{"error", "daemon shutting down"}, {"state", power_name(power.load())}});
        for (auto& w : sleep_waiters) ctl_reply(w.fd, {{"error", "daemon shutting down"}, {"state", power_name(power.load())}});
        sleep_waiters.clear();
        std::vector<int> fl;
        { std::lock_guard<std::mutex> lk(mu); fl.swap(flush_q); }
        for (int fd : fl) ctl_reply(fd, {{"error", "daemon shutting down"}});
      }
      std::deque<Request> rejected;
      { std::lock_guard<std::mutex> lk(mu); rejected.swap(queue); }
      for (auto& r : rejected) { send_json(r.fd, {{"error", "daemon shutting down"}}); finish_socket(r.fd); }
      if (stop_state.load() >= 2) for (auto& a : active) a->cancel->store(true);
      if (active.empty()) break;
      decode_step();
      continue;
    }
    // sleep/wake (without control requests power is always ready — the two branches below are never taken)
    handle_ctl();
    handle_flush();  // {"op":"flush"}: refused while a request decodes; while asleep it clears host-side state only
    if (power.load() == kPowerDraining) {
      const double tn = now_ms();
      std::vector<int> late;
      for (auto it = sleep_waiters.begin(); it != sleep_waiters.end();) {
        if (it->deadline > 0 && tn >= it->deadline) { late.push_back(it->fd); it = sleep_waiters.erase(it); }
        else ++it;
      }
      if (sleep_waiters.empty()) {  // every pending sleep passed its deadline: withdraw the sleep and reopen intake
        power.store(kPowerReady);
        recompute_sleep_asked();
        fprintf(stderr, "[hived] sleep withdrawn: still %zu in-flight request(s) at the caller's timeout\n", active.size());
      }
      for (int fd : late) ctl_reply(fd, {{"error", "not idle"}, {"state", power_name(power.load())}, {"running", active.size()}, {"retryable", true}});
      if (!late.empty()) publish_stats(true);
      if (power.load() == kPowerDraining) {
        if (active.empty()) { do_sleep(); continue; }
        drain_running = (int)active.size();
        sweep_cancelled_queue();  // requests cancelled in the queue while waiting to sleep also finish at once (not held until wake)
        decode_step();
        rt.release_vision_if_idle();
        continue;
      }
    }
    if (power.load() == kPowerSleeping) {
      sweep_cancelled_queue();
      std::unique_lock<std::mutex> lk(mu);
      cv.wait_for(lk, std::chrono::seconds(1), [&] { return !ctl_q.empty() || !flush_q.empty() || stop_state.load() != 0; });  // sweep cancelled queued requests every second
      continue;
    }
    // 1) take a new request: wait if active is empty, otherwise take one and prefill it only if a seat (max_batch) is free
    {
      Request r;
      bool got = false;
      std::unique_lock<std::mutex> lk(mu);
      if (active.empty()) cv.wait_for(lk, std::chrono::seconds(30), [&] { return !queue.empty() || stop_state.load() != 0 || !ctl_q.empty() || !flush_q.empty(); });  // wake every 30 s for idle cleanup (vision encoder release)
      if (!queue.empty() && (int)active.size() < max_batch && !sleep_asked.load()) {
        // Priority: the best-priority queued request first (FIFO among equals), and none below a running request's priority — its
        //   prefill would interrupt that request's decode (the decode_step rule holds it anyway). Without the field everything is 0: FIFO as before.
        auto prio_of = [](const Request& x) {
          const auto it = x.req.find("priority");
          if (it == x.req.end() || !it->is_number()) return 0;
          const double v = it->get<double>();
          return std::isfinite(v) ? (int)std::max(-1000.0, std::min(1000.0, v)) : 0;
        };
        int best_active = INT_MAX;
        for (auto& A : active) best_active = std::min(best_active, A->prio);
        auto pick = queue.begin();
        int pick_prio = prio_of(*pick);
        for (auto it = std::next(queue.begin()); it != queue.end(); ++it) { const int p = prio_of(*it); if (p < pick_prio) { pick = it; pick_prio = p; } }
        if (pick_prio <= best_active) { r = std::move(*pick); queue.erase(pick); got = true; }
      }
      lk.unlock();
      if (got) {
        last_prefill_tc = 0;
        if (batch_eligible(r, true)) admit_batch(std::move(r));  // HIVE_BATCH_PREFILL — when off, always the regular admit
        else admit(std::move(r));
        // HIVE_PREFILL_FAIR: if another prefill is waiting (and a seat is free), give decoders the same share as between chunks first. With an empty queue the loop below decodes anyway.
        if (prefill_fair && last_prefill_tc > 0 && !active.empty()) {
          bool more = false;
          { std::lock_guard<std::mutex> lk2(mu); more = !queue.empty() && (int)active.size() < max_batch && stop_state.load() == 0 && !sleep_asked.load(); }
          if (more) decode_between(last_prefill_tc);
        }
      }
    }
    if (active.empty()) { rt.release_vision_if_idle(); continue; }
    decode_step();
    rt.release_vision_if_idle();  // release the lazily loaded vision encoder between steps when idle (reclaims VRAM)
  }
  // Graceful shutdown: save the warm-start file -> stop the receiver thread (close remaining half-open connections) -> wait until the shutdown
    //   messages are flushed (up to 2 s) -> remove the socket file -> exit 0.
    //   Runtime and store destructors are not run (_exit) — that path is not exercised in operation; the OS reclaims process memory and the device.
  if (power.load() != kPowerSleeping) save_cache_state();  // asleep means nothing is resident — keep the file written at sleep time
  stop_state.store(3);
  { const char b = 1; const ssize_t w = ::write(stop_pipe[1], &b, 1); (void)w; }
  acceptor.join();
  ::close(lfd);
  const bool flushed = writer.wait_idle(2000);
  ::unlink(sock_path.c_str());
  fprintf(stderr, "[hived] graceful stop complete (%s) — exit 0\n", flushed ? "all replies flushed" : "reply flush timed out");
  fflush(stderr);
  _exit(0);
}
