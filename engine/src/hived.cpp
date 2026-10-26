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
//   -> response lines: [{"admitted":...} with "stages"] {"progress":i,"total":n} after each prefill chunk · {"id":123} ... {"done":true,"n":k,"finish":"stop|length","prefill_ms":..,"decode_ms":..,"cached_prefix":p,"restore_ms":..} (restore_ms = host image restores, part of prefill_ms)
//   {"op":"cancel","session":"s1"}  -> {"ok":true}
//   {"op":"cancel","session":"s1","rid":"r1"}  -> {"ok":true}  (with rid: only the request that passed the same rid to generate)
//   {"op":"stats"} -> cache and routing statistics ("state": ready|draining|sleeping|waking, "vram", the last "sleep"/"wake" report)
//   {"op":"sleep"[, "timeout_s":N]} -> releases VRAM once in-flight requests finish {"ok":true,"state":"sleeping","sleep":{...}} · {"op":"wake"} -> {"ok":true,"state":"ready",...}
//     (idempotent; failure/timeout responses and the queueing policy are described in the header comment of the "sleep / wake" block below)
//   (optional) generate's "boundaries":[offset...] — read only with HIVE_PREFIX_SHARE (boundary snapshots). Otherwise ignored like any unknown field.
//   {"op":"set","mtp_batch":1|0|-1} -> {"ok":true,"mtp_batch":v} — run-time HIVE_MTP_BATCH override (-1 = startup value; needs the verify decode path allocated at startup)
//   {"op":"flush"} — drop every reusable prompt state (session tokens/checkpoints/history, archived sessions, shared boundary snapshots)
//     when no request is decoding; replies {"ok":true,...} or {"error":"not idle",...}. The expert VRAM cache (weights) is kept.
//   {"op":"where","sessions":["s1",...]} -> {"ok":true,"sessions":{"s1":{"where":"vram|ram|ssd|none","tokens":N,"bytes":B,"tiers":{...}[,"loading":true]}}}
//     where a session's reusable state is, fastest tier first (answered on the receiver thread from a snapshot the engine thread publishes —
//     never waits for a request; "tokens" / "bytes" / the tiers are defined at the "where / load" block below and in docs/architecture.md)
//   {"op":"load","session":"s1"} -> {"ok":true,"session":"s1","loaded":true|false,"where":...,"tokens":N,"bytes":B[,"reason":...]}
//     reads the session's stored prompt checkpoint (HIVE_SESSION_STORE) on a loader thread and attaches it as an archived (RAM) session; a
//     session already in VRAM or RAM is left as it is and its place returned
//   (optional) generate's "history":true — the conversation already has a turn (enables the long re-read boundary snapshot; HIVE_PREFIX_SHARE only).
//   (optional) generate's "prefix_extra":N — per-request extra-chunk budget for boundary cuts (overrides HIVE_PREFIX_EXTRA_CHUNKS; HIVE_PREFIX_SHARE only).
// Sessions: if a session's previous token sequence is a prefix of the new request, only the remainder is prefilled (conversation cache).
// If the prefix breaks, resume from whichever of the prompt-end checkpoint and the RAM-held states reaches furthest.
#include <climits>
#include <fcntl.h>
#include <csignal>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
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
#include "hive/host_ledger.h"
#include "hive/mtp_gate.h"
#include "hive/sampler.h"
#include "hive/socket_writer.h"
#include "hive/vram_limit.h"
#include "nlohmann/json.hpp"
#include "hive/session_store.h"
// Model family: the DeepSeek build (default), the GLM build (-DHIVE_FAMILY_GLM, target hived_glm) and the Qwen3.8-Flash-Next build
//   (-DHIVE_FAMILY_QWEN, target hived_qwen; -DHIVE_FAMILY_QWEN_CPU, target hived_qwen_cpu: CPU only, built against engine/cpu_shim), the CPU-only
//   GLM / DeepSeek builds (-DHIVE_FAMILY_GLM_CPU / -DHIVE_FAMILY_DS_CPU), the Qwen3.8-27B builds (-DHIVE_FAMILY_Q27 / -DHIVE_FAMILY_Q27_CPU) and
//   the Gemma 4 build (-DHIVE_FAMILY_GEMMA, target hived_gemma) share
//   this file; the family header
//   provides Config, Model, ExpertStore, RuntimeOptions, ForwardStats, ImageInput, PrefillPart, Seq, SeqImage, Runtime and the load helpers.
#if defined(HIVE_FAMILY_GLM)
#include "hive/glm/hived_family.h"
#elif defined(HIVE_FAMILY_QWEN)
#include "hive/qwen/hived_family.h"
#elif defined(HIVE_FAMILY_QWEN_CPU)
#include "hive/qwen/hived_family_cpu.h"
#elif defined(HIVE_FAMILY_GLM_CPU)
#include "hive/glm/hived_family_cpu.h"
#elif defined(HIVE_FAMILY_DS_CPU)
#include "hive/ds/hived_family_cpu.h"
#elif defined(HIVE_FAMILY_Q27)
#include "hive/q27/hived_family.h"
#elif defined(HIVE_FAMILY_Q27_CPU)
#include "hive/q27/hived_family_cpu.h"
#elif defined(HIVE_FAMILY_GEMMA)
#include "hive/gemma/hived_family.h"
#else
#include "hive/bulk_load.h"
#include "hive/expert_store.h"
#include "hive/model.h"
#include "hive/runtime.h"
#endif

using json = nlohmann::json;
using namespace hive;
using namespace hive::sampling;
#if defined(HIVE_FAMILY_GLM)
using namespace hive::glm::fam;
#elif defined(HIVE_FAMILY_QWEN)
using namespace hive::qwen::fam;
#elif defined(HIVE_FAMILY_QWEN_CPU)
using namespace hive::qwen::cpufam;
#elif defined(HIVE_FAMILY_GLM_CPU)
using namespace hive::glm::cpufam;
#elif defined(HIVE_FAMILY_DS_CPU)
using namespace hive::ds::cpufam;
#elif defined(HIVE_FAMILY_Q27)
using namespace hive::q27::fam;
#elif defined(HIVE_FAMILY_Q27_CPU)
using namespace hive::q27::cpufam;
#elif defined(HIVE_FAMILY_GEMMA)
using namespace hive::gemma::fam;
#endif
// HIVE_VRAM_LIMIT_MB (hive/vram_limit.h): only the GPU families take part; the CPU-only families have no device memory to limit.
#if defined(HIVE_FAMILY_QWEN_CPU) || defined(HIVE_FAMILY_GLM_CPU) || defined(HIVE_FAMILY_DS_CPU) || defined(HIVE_FAMILY_Q27_CPU) || defined(HIVE_CPU_ONLY)
#define HIVE_VRAM_LIMIT_ON 0
constexpr bool kVramReserve = false;
#else
#define HIVE_VRAM_LIMIT_ON 1
#if defined(HIVE_FAMILY_GLM)
constexpr hive::vram_limit::Family kVramFamily = hive::vram_limit::Family::Glm;
#elif defined(HIVE_FAMILY_QWEN)
constexpr hive::vram_limit::Family kVramFamily = hive::vram_limit::Family::Qwen;
#elif defined(HIVE_FAMILY_Q27) || defined(HIVE_FAMILY_GEMMA)
constexpr hive::vram_limit::Family kVramFamily = hive::vram_limit::Family::None;  // no cost table yet: ballast only, no startup plan
#else
constexpr hive::vram_limit::Family kVramFamily = hive::vram_limit::Family::DeepSeek;
#endif
// The ballast is swapped for a runtime reserve at ready only where runtime sizing does not read the device's free memory (see the ready block).
#if defined(HIVE_FAMILY_GEMMA)
constexpr bool kVramReserve = true;   // Gemma sizes everything at load (KV page pool from the limit; a full pool spills pages to host memory)
#else
constexpr bool kVramReserve = kVramFamily == hive::vram_limit::Family::DeepSeek;
#endif
#endif
#ifndef HIVE_BUILD_ID
#define HIVE_BUILD_ID "unknown"
#endif
// Session store format (HIVE_SESSION_STORE layout key): raise it when the bytes of a saved image change meaning (SeqImage fields, buffer
//   layouts, KV numerics a canary would not catch). Rebuilds keep the store otherwise — the canary check at start-up decides (store_canary).
constexpr int kStoreFormat = 1;

namespace {

// Optional Runtime members, probed per family. A requires-expression on a concrete object outside a template is a hard error when the
//   member is missing (no SFINAE) — these concepts keep the probe dependent so families without the member simply skip the branch.
template <class R, class S> concept RtHasTrimSeq = requires(R& r, S& q) { r.trim_seq(q); };
template <class R> concept RtHasKvReclaim = requires(R& r, std::function<bool()> f) { r.set_kv_reclaim(f); };
template <class R, class S> concept RtHasKvReclaimable = requires(const R& r, const S& q) { { r.kv_reclaimable(q) } -> std::convertible_to<bool>; };
template <class R, class S, class F> concept RtHasDraftBatch =
    requires(R& r, std::vector<S*>& v, std::vector<int32_t>& t, std::vector<std::vector<int32_t>>& d, std::vector<std::vector<float>>& c, F* f) {
      r.mtp_draft_batch(v, t, d, c, f);
    };
template <class R> concept RtHasVerifyLogits = requires(R& r) { r.set_verify_logits(true); };
template <class R> void rt_set_verify_logits(R& r, bool on) { if constexpr (RtHasVerifyLogits<R>) r.set_verify_logits(on); }
// The calls themselves go through these templates as well: a discarded if-constexpr branch in a non-template function is still checked.
template <class R, class S> void rt_trim_seq(R& r, S& q) { if constexpr (RtHasTrimSeq<R, S>) r.trim_seq(q); }
// A family whose KV can be cut back to any earlier position it still covers (Gemma: paged KV) continues the live state from the common
//   prefix of a request that shares only part of it, instead of restoring a snapshot, and needs no per-turn boundary snapshots.
template <class R, class S> concept RtHasRewind = requires(const R& r, S& q) { { r.rewind(q, size_t{}) } -> std::convertible_to<bool>; };
template <class R, class S> bool rt_rewind(const R& r, S& q, size_t n) { if constexpr (RtHasRewind<R, S>) return r.rewind(q, n); else return false; }
template <class R> void rt_set_kv_reclaim(R& r, std::function<bool()> f) { if constexpr (RtHasKvReclaim<R>) r.set_kv_reclaim(std::move(f)); }
// false = trimming this sequence frees no device KV (it holds only what every pooled sequence keeps): evicting its session would cost a
//   restore later for nothing. Families without the member: every idle sequence is a candidate.
template <class R, class S> bool rt_kv_reclaimable(const R& r, const S& q) { if constexpr (RtHasKvReclaimable<R, S>) return r.kv_reclaimable(q); else return true; }
template <class R, class S, class F>
void rt_draft_batch(R& r, std::vector<S*>& v, std::vector<int32_t>& t, std::vector<std::vector<int32_t>>& d, std::vector<std::vector<float>>& c, F* f) {
  if constexpr (RtHasDraftBatch<R, S, F>) r.mtp_draft_batch(v, t, d, c, f);
}

double now_ms() { return hive::mono_ms(); }
// Project convention (same as env_on in runtime.cpp): unset / "" / "0" = off
bool env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }
// HIVE_VRAM_LIMIT_MB ballast held for the life of the process (0 = no limit). Subtracted from the "used" values hived publishes, so they read
//   like the smaller card.
size_t g_vram_ballast = 0;  // the reserve at startup (vram_reserve pieces); stats use vram_reserve::held() — runtime allocations draw from it
size_t g_vram_limit_armed = 0;  // HIVE_VRAM_LIMIT_MB held for the process lifetime (vram_reserve armed at ready): the process occupies exactly this
void* g_vram_ballast_ptr = nullptr;  // kept so HIVE_VRAM_BALLAST_RELEASE can free it before ready
double vram_used_mib(size_t tot, size_t fr) { const size_t u = tot > fr ? tot - fr : 0, b = vram_reserve::held(); return (double)(u > b ? u - b : 0) / 1048576.0; }

// Graceful shutdown (installed only when HIVE_GRACEFUL_STOP_S>0 — by default SIGTERM keeps its default action = immediate exit).
//   The handler only writes one byte to a self-pipe (async-signal-safe). A second signal restores the default action and dies at once (forced exit while draining).
// {"op":"set","mtp_batch":0|1|-1}: run-time override of HIVE_MTP_BATCH (-1 = the startup value). Written by the receiver thread, read by the engine
//   thread at each batched step (relaxed — a step or two of delay is fine). Added 2026-10-09 so the batched-speculation A/B can alternate inside one
//   running service instead of restarting it (no spare GPU window — the service carries live traffic).
std::atomic<int> g_mtp_batch_override{-1};
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
  // Host bytes outside the images (the images themselves are counted once each by HostLedger — see host_total)
  size_t host_bytes() const {
    size_t n = tokens.capacity() * sizeof(int32_t) + ckpt_logits.capacity() * sizeof(float);
    if (seq) n += seq->tokens.capacity()*sizeof(int32_t) + seq->engram_history.capacity()*sizeof(int64_t);
    n += (img_sig.capacity() + ckpt_sig.capacity() + live_sig.capacity()) * sizeof(ImgSig);
    for (const auto& f : history) n += sizeof(Frame) + f.logits.capacity() * sizeof(float) + f.sig.capacity() * sizeof(ImgSig);
    return n;
  }
  template <class F> void each_image(F&& f) const {
    f(ckpt);
    f(live);
    for (const auto& h : history) f(h.image);
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
  // Image signatures (span start, end, hash of the patch bytes, token types and grid) — computed once (request_img_sig): admission and the
  //   layer-yield rules both need them, and the hash walks every patch byte (a long agent conversation with screenshots sends 118 MB of patches per turn).
  struct ImgSig { int start, end; uint64_t hash; };
  std::vector<ImgSig> img_sig;
  int img_sig_state = 0;  // 0 = not computed · 1 = computed · -1 = the payload is shorter than the images declare
};
// The request's image signatures (computed on the first call). nullptr = its payload is shorter than its images declare. A malformed image
//   entry throws (json) — every call, since nothing is cached then.
const std::vector<Request::ImgSig>* request_img_sig(Request& r) {
  if (r.img_sig_state == 0) {
    std::vector<Request::ImgSig> sig;
    int state = 1;
    const json& q = r.req;
    if (q.contains("images") && q["images"].is_array()) {
      size_t off = 0;
      for (const auto& im : q["images"]) {
        const size_t nb = im.at("nbytes").get<size_t>();
        if (nb > r.bin.size() - off) { state = -1; sig.clear(); break; }
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&](uint8_t b) { hash ^= b; hash *= 1099511628211ull; };
        for (size_t b = 0; b < nb; ++b) mix(r.bin[off + b]);
        for (const auto& t : im.at("types")) mix((uint8_t)t.get<int>());
        for (auto key : {"n_vit_h", "n_vit_w"}) { uint32_t n = im.at(key).get<uint32_t>(); for (int b = 0; b < 4; ++b) mix((n >> (8 * b)) & 255); }
        const int start = im.at("start").get<int>();
        sig.push_back({start, start + (int)im.at("types").size(), hash});
        off += nb;
      }
    }
    r.img_sig = std::move(sig);
    r.img_sig_state = state;
  }
  return r.img_sig_state > 0 ? &r.img_sig : nullptr;
}

// An image's patches at admission. Families whose ImageInput carries host fields (DeepSeek: hive/runtime.h, its CPU daemon) get the request's
//   payload bytes — merge_images uploads only the images a chunk encodes, into the encoder's own buffer, so admission allocates no device memory
//   (measured 2026-10-25, DeepSeek at VRAM 100 %: the per-image copy below failed every turn of a browser conversation, ImageInput). The other
//   families (GLM, Qwen, Gemma, Q27) keep the device copy, held by the job until it ends. A template, so the branch a family lacks is discarded.
template <class Img> constexpr bool kHostPatches = requires(Img& x) { x.patches_host; x.patch_bytes; };  // admission allocates no device memory for images
template <class Img>
void set_image_patches(Img& x, const uint8_t* bytes, size_t nb, std::vector<DevBuf>& keep) {
  if constexpr (requires { x.patches_host; x.patch_bytes; }) {
    x.patches_host = bytes;
    x.patch_bytes = nb;
  } else {
    DevBuf d(nb);
    CUDA_CHECK(cudaMemcpy(d.p, bytes, nb, cudaMemcpyHostToDevice));
    x.patches_dev = d.as<bf16>();
    keep.push_back(std::move(d));
  }
}

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

// All CUDA pinned host memory of this process: cudaHostAlloc maps it as shared /dev/zero, and pinned pages are always resident, so the sum of
//   those mappings is their RSS. It holds the session images (counted in host_session_bytes), the pinned copies of the dense weights kept for
//   level-3 sleep (dense_host_copy_mib, hive/devmem.h prepare_shadows), the embedding table, the prefill host tiles and the work buffers.
//   Measured 2026-10-21 (production deployment, 8.5 h after a restart): 22,127 MiB in 3,429 mappings, of which 10,256 MiB were sleep shadows ("sleep shadows
//   pinned ahead" at startup), 1,262 MiB the embedding table, 1,920 MiB the host tiles — host_session_bytes was 7,742 MiB, so without these two
//   fields ~14 GB of RSS looked unaccounted. Reads /proc/self/maps (4,514 lines: ~1-2 ms), so publish_stats refreshes it at most every 10 s.
size_t cuda_pinned_bytes() {
  FILE* f = fopen("/proc/self/maps", "r");
  if (!f) return 0;
  size_t n = 0;
  char line[512];
  while (fgets(line, sizeof line, f)) {
    if (!strstr(line, " /dev/zero")) continue;
    unsigned long a = 0, b = 0;
    if (sscanf(line, "%lx-%lx", &a, &b) == 2 && b > a) n += b - a;
  }
  fclose(f);
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = getenv("HIVE_CKPT") ? getenv("HIVE_CKPT") : "", engram_dir, sock_path = "/tmp/hive.sock";
  int max_layer = -1, cpu_threads = 16, max_chunk = 4096, max_sessions = 2;
  int prefill_tile = 1;  // --prefill-tile N (max_chunk x N tokens per tile, experts loaded once per encoder layer)
  double host_session_mb = 32768;  // budget for session images kept outside the pool (RAM) — oldest are dropped beyond it
  int ckpt_min_tokens = 256;       // prompts shorter than this get no checkpoint (re-reading is cheaper)
  // boundary hints (HIVE_PREFIX_SHARE) shorter than this are not cut (-1 = ckpt_min_tokens). Separate because the two pay off at different
  //   sizes: a short shared system block (Gemma turn judgments: ~200 tokens before a one-line question, ~10 per second) is worth a shared
  //   snapshot, while a prompt checkpoint of every such one-token request kept the image writer busy until the engine thread waited on it
  //   (measured 2026-10-18: judgments of 252-267 tokens over the default 256 — the engine stalled 2.4 s with nothing in the log).
  int hint_min_tokens = -1;
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
    else if (a == "--hint-min-tokens") hint_min_tokens = atoi(next().c_str());
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
  // HIVE_VRAM_LIMIT_MB state (GPU families; 0 = off): limit, CUDA context, free VRAM right after the context was created
  const size_t vram_limit_bytes = HIVE_VRAM_LIMIT_ON ? vram_limit::limit_bytes() : 0;
  size_t vram_ctx_bytes = 0, vram_free_ctx = 0;
  bool vram_ctx_measured = false;
  (void)vram_ctx_measured; (void)vram_ctx_bytes; (void)vram_free_ctx;
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
#if HIVE_VRAM_LIMIT_ON
    // HIVE_VRAM_LIMIT_MB (hive/vram_limit.h): measure the CUDA context now — NVML's free memory before the context exists, cudaMemGetInfo's
    //   right after cudaFree(0) creates it. Unset = nothing here runs (no context is created early, startup unchanged).
    if (vram_limit_bytes > 0) {
      char pci[32];
      snprintf(pci, sizeof pci, "%08x:%02x:%02x.0", (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
      size_t f0 = 0, tot = 0;
      const bool nvml = vram_limit::nvml_free(pci, &f0);
      CUDA_CHECK(cudaFree(0));
      CUDA_CHECK(cudaMemGetInfo(&vram_free_ctx, &tot));
      // NVML free below the CUDA free (another process freed memory in between) or a context under 64 MiB (it already existed): not a
      //   measurement — fall back to the estimate and say so
      vram_ctx_measured = nvml && f0 > vram_free_ctx && f0 - vram_free_ctx >= (64ull << 20);
      vram_ctx_bytes = vram_ctx_measured ? f0 - vram_free_ctx : vram_limit::kCtxFallback;
      fprintf(stderr, "[hived] VRAM limit %.0f MiB on a %.0f MiB card: CUDA context %.0f MiB (%s)\n", vram_limit_bytes / 1048576.0, tot / 1048576.0,
              vram_ctx_bytes / 1048576.0, vram_ctx_measured ? "measured: NVML free before − CUDA free after" : nvml ? "estimate — NVML reading not usable" : "estimate — NVML not available");
    }
#endif
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
#if HIVE_VRAM_LIMIT_ON
  // HIVE_VRAM_LIMIT_MB, after the weights (measured, not estimated): startup plan, then the ballast (hive/vram_limit.h, docs/vram-footprint.md §4.2)
  if (vram_limit_bytes > 0) {
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    const size_t weights = vram_free_ctx > fr ? vram_free_ctx - fr : 0;  // dense weights, MTP, model-side buffers
    const double MiB = 1048576.0;
    const double after_weights = (double)vram_limit_bytes - (double)vram_ctx_bytes - (double)weights;  // what the limit leaves for the rest
    const double per_layer = (double)weights / std::max(1, nl);
    // the MTP weights are the one block that leaves the GPU without moving a layer
#if defined(HIVE_FAMILY_GLM)
    const bool mtp_on = !(getenv("HIVE_GLM_MTP_K") && strcmp(getenv("HIVE_GLM_MTP_K"), "0") == 0);
    const char* mtp_knob = "HIVE_GLM_MTP_K=0";
#elif defined(HIVE_FAMILY_QWEN)
    const bool mtp_on = !(getenv("HIVE_QWEN_MTP_K") && strcmp(getenv("HIVE_QWEN_MTP_K"), "0") == 0);
    const char* mtp_knob = "HIVE_QWEN_MTP_K=0";
#else
    const bool mtp_on = n_mtp > 0;
    const char* mtp_knob = "--no-mtp";
#endif
    auto give_up = [&](double short_by, const char* what) {
#if defined(HIVE_FAMILY_QWEN)
      // FN places layers on the CPU itself (fam::Model, HIVE_QWEN_GPU_LAYERS / the limit): reaching this point means even its choice did not fit
      const char* placement = " — set a lower HIVE_QWEN_GPU_LAYERS (or unset it to let the family choose; docs/qwen.md)";
#elif defined(HIVE_FAMILY_GLM)
      // GLM places layers on the CPU itself (GlmModel, HIVE_GLM_GPU_LAYERS / the limit)
      const char* placement = " — set a lower HIVE_GLM_GPU_LAYERS (or unset it to let the family choose from the limit)";
#elif !defined(HIVE_FAMILY_Q27) && !defined(HIVE_FAMILY_GEMMA)
      // DeepSeek places layers on the CPU itself (Model::plan_gpu_layers: HIVE_DS_GPU_LAYERS / the limit)
      const char* placement = " — set a lower HIVE_DS_GPU_LAYERS (or unset it to let the model choose from the limit; docs/vram-footprint.md §3.2)";
#else
      const char* placement = ". Layer placement on the CPU does not exist yet for this family — see docs/vram-footprint.md §3.2";
#endif
      fprintf(stderr, "[hived] VRAM limit %.0f MiB: %s — short by %.0f MiB (context %.0f + weights %.0f MiB, %d layers ≈ %.0f MiB each): about %d layer(s) "
              "would have to leave the GPU%s%s\n",
              vram_limit_bytes / MiB, what, short_by / MiB, vram_ctx_bytes / MiB, weights / MiB, nl, per_layer / MiB,
              (int)std::ceil(short_by / std::max(1.0, per_layer)), mtp_on ? (std::string(" (first: ") + mtp_knob + " drops the MTP weights)").c_str() : "",
              placement);
      fflush(stderr);
      _exit(2);
    };
    if (after_weights <= 0) give_up(-after_weights, "the weights alone do not fit");
    if (kVramFamily != vram_limit::Family::None) {
      vram_limit::Knobs k;
      k.max_chunk = max_chunk; k.prefill_tile = prefill_tile; k.max_ctx = max_ctx; k.max_sessions = max_sessions;
      k.max_batch = max_batch_arg > 0 ? max_batch_arg : 8;
      // runtime batch caps: GLM 8 · Qwen FN 16 sequences (64 verify rows, 2026-10-10 batched decode) · DeepSeek uncapped here
      if (kVramFamily == vram_limit::Family::Glm) k.max_batch = std::min(k.max_batch, 8);
      else if (kVramFamily == vram_limit::Family::Qwen) k.max_batch = std::min(k.max_batch, 16);
      k.mtp = mtp_on; k.mtp_batch = env_on("HIVE_MTP_BATCH");
      if (const char* v = getenv("HIVE_STAGING")) k.staging = std::max(8, std::min(256, atoi(v)));
      if (kVramFamily == vram_limit::Family::DeepSeek && env_on("HIVE_PREFILL_HOST_TILES")) {  // the runtime's host_tile_count rule (runtime.cpp)
        int n = 2;
        if (const char* mb = getenv("HIVE_PREFILL_HOST_MB"); mb && *mb) {
          const double v = atof(mb), hb = (double)max_chunk * 4 * 5120 * 2;
          n = std::isfinite(v) && v > 0 ? (int)std::min(1e6, v * MiB / hb) : 0;
        }
        k.host_tiles = (int)std::max<int64_t>(0, std::min<int64_t>(n, (max_ctx + max_chunk - 1) / std::max(1, max_chunk) - prefill_tile));
      }
      const char* kvb = getenv(kVramFamily == vram_limit::Family::Glm ? "HIVE_GLM_KV_BASE" : "HIVE_QWEN_KV_BASE");
      if (kvb && *kvb && atoll(kvb) >= 0) k.kv_base = atoll(kvb);
      const double room = after_weights - (double)cache_reserve_bytes();
      const double before = vram_limit::fixed_bytes(kVramFamily, k);
      bool fits = false;
      const vram_limit::Knobs k0 = k;
      const auto steps = vram_limit::plan(kVramFamily, k, room, &fits);
      fprintf(stderr, "[hived] VRAM plan (%s, estimate): limit %.0f − context %.0f − weights %.0f − reserve %.0f = %.0f MiB for work buffers, sessions and "
              "staging; they need %.0f MiB at the configured settings%s\n", vram_limit::family_name(kVramFamily), vram_limit_bytes / MiB, vram_ctx_bytes / MiB,
              weights / MiB, cache_reserve_bytes() / MiB, room / MiB, before / MiB, steps.empty() && fits ? " — fits, nothing changed" : "");
      for (const auto& s2 : steps) fprintf(stderr, "[hived] VRAM plan: %s (%.0f → %.0f MiB)\n", s2.what.c_str(), s2.before / MiB, s2.after / MiB);
      if (!fits) give_up(vram_limit::fixed_bytes(kVramFamily, k) - room, "work buffers and sessions do not fit even at the lowest settings");
      // apply: the store, runtime and session pool below read these
      max_chunk = k.max_chunk; prefill_tile = k.prefill_tile; max_sessions = k.max_sessions;
      if (k.max_batch != k0.max_batch) max_batch_arg = k.max_batch;
      if (k.host_tiles != k0.host_tiles) setenv("HIVE_PREFILL_HOST_TILES", "0", 1);
      if (k.staging != k0.staging) setenv("HIVE_STAGING", std::to_string(k.staging).c_str(), 1);
      if (k.mtp_batch != k0.mtp_batch) setenv("HIVE_MTP_BATCH", "0", 1);
      if (!steps.empty())
        fprintf(stderr, "[hived] VRAM plan: now --max-chunk %d · --prefill-tile %d · --max-batch %d · --max-sessions %d · fixed part %.0f MiB, %.0f MiB left for the expert cache\n",
                max_chunk, prefill_tile, k.max_batch, max_sessions, vram_limit::fixed_bytes(kVramFamily, k) / MiB, (room - vram_limit::fixed_bytes(kVramFamily, k)) / MiB);
    } else {
      fprintf(stderr, "[hived] VRAM plan: no cost table for this family — ballast only\n");
    }
    // ballast: the process keeps exactly (limit − context − weights) free from here on
    const size_t target = (size_t)after_weights;
    if (fr > target) {
      // In vram_reserve pieces (hive/vram_reserve.h), so that after ready a runtime allocation can take its room from this reserve
      //   instead of from memory another process on the card freed (2026-10-18 OOM beside a 25.6 GB neighbour).
      g_vram_ballast = vram_reserve::fill(fr - target);
      g_vram_ballast_ptr = g_vram_ballast ? (void*)1 : nullptr;  // marker only: the pieces live in vram_reserve
      size_t fr2 = 0;
      cudaMemGetInfo(&fr2, &tot);
      fprintf(stderr, "[hived] VRAM limit %.0f MiB: ballast %.0f MiB%s — VRAM free %.0f MiB (the process now sees a %.0f MiB card)\n", vram_limit_bytes / MiB,
              g_vram_ballast / MiB, g_vram_ballast + vram_reserve::kChunk < fr - target ? " (⚠️less than wanted: allocation refused)" : "", fr2 / MiB,
              (double)(vram_ctx_bytes + weights + fr2) / MiB);
    } else {
      fprintf(stderr, "[hived] VRAM limit %.0f MiB: no ballast — free VRAM %.0f MiB is already below what the limit leaves (%.0f MiB)\n", vram_limit_bytes / MiB,
              fr / MiB, after_weights / MiB);
    }
  }
#endif
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
  // HIVE_PREFIX_INTERVAL (with HIVE_PREFIX_SHARE; default 32768 tokens, "0" = off): a long prefill also keeps a snapshot at a chunk end
  //   once it has read this many tokens past its last snapshot (or past where it resumed). Turn-end snapshots only cover the turns a
  //   conversation was read in, so a long re-read left nothing between its head and its last turns. Measured 2026-10-19 (DeepSeek, a
  //   chat client's 280K conversation): re-read from 0 at 09:01 (snapshots at 276,464 and later), then at 09:36 the client changed the
  //   history at 167,322 and the request re-read all 297,628 tokens from 0 (~100 s) — "snapshot at 280573 shares 167322 tokens (prompt
  //   differs)". A chunk end costs no extra chunk, only the image copy (a delta on the request's previous snapshot: new rows plus the
  //   fixed window state), and the snapshots go to the session store with the others.
  static const size_t prefix_interval = !prefix_share ? 0 : (getenv("HIVE_PREFIX_INTERVAL") && *getenv("HIVE_PREFIX_INTERVAL")) ? (size_t)std::max(0L, atol(getenv("HIVE_PREFIX_INTERVAL"))) : 32768;
  if (prefix_share) fprintf(stderr, "[hived] prefix share on: boundary snapshots, extra chunks per request %d · interval %zu\n", prefix_extra, prefix_interval);
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
  // HIVE_LAYER_YIELD_SLICE (with HIVE_LAYER_YIELD; unset / "" / "0" / non-numeric = off · N = rows per slice, capped at max_chunk): a queued
    //   request that no yield rule admits (too long for the short and mid rules) still starts at a layer yield — as a sliced prefill: its
    //   chunks of at most N rows (and at most the rows the runtime parked) run at the following yields, alternating with the paused prefill,
    //   until either finishes; the one left over continues as the main prefill. Requests of the same priority class share the time about
    //   equally (a slice runs once the paused prefill has run as long as the last slice took); a better class gets a slice at every yield; a
    //   worse class is not sliced into a better one. Measured 2026-10-25 20:50 (production log, DeepSeek): a 174,310-row read-ahead (priority 1)
    //   prefilled for 218 s while a conversation turn of 24,639 new rows waited 156 s in the queue behind it, and two conversations of the
    //   same priority read their prompts one after the other (a 298K re-read held every other conversation for 80 s) — a request waited for the
    //   whole prefill in front of it unless it passed the short or mid rule. Schedule only: a slice is an ordinary chunk of that request
    //   (its own tokens, positions and causal order; the tail-layer skip (C1) is not used before a slice, since its next chunk may be small).
  const size_t ly_slice = ly_period <= 0 ? 0 : [&] {
    const char* v = getenv("HIVE_LAYER_YIELD_SLICE");
    char* end = nullptr;
    const long long x = v && *v ? strtoll(v, &end, 10) : 0;
    return v && *v && end && *end == 0 && x > 0 ? (size_t)std::min<long long>(x, (long long)max_chunk) : (size_t)0;
  }();
  // HIVE_PRIORITY_SHARED (optional; unset / "" / non-numeric = off): requests whose priority is <= N form one class — they decode together and
    //   none holds another's decode, prefill or admission (the priority rules below compare classes). Priorities above N keep the strict order
    //   (a background request still waits for the better ones). Measured 2026-10-25/26 (production log): interactive turns arrive with -1 or 0
    //   by surface; under the strict order a 0 turn's decode and its admission were held for the whole of every -1 turn (an agent
    //   conversation running turn after turn kept the other conversations waiting), which a single user saw as "one turn at a time".
  static const int prio_shared = [] {
    const char* v = getenv("HIVE_PRIORITY_SHARED");
    char* end = nullptr;
    const long x = v && *v ? strtol(v, &end, 10) : 0;
    return v && *v && end && *end == 0 ? (int)std::max(-1000L, std::min(1000L, x)) : INT_MIN;
  }();
  auto prio_class = [](int p) { return prio_shared != INT_MIN && p <= prio_shared ? prio_shared : p; };
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
  // {"op":"load"} (the "where / load" block below): connections queued for the loader thread · images it read, waiting for the engine thread
  //   to attach them (top of the loop, like flush) · sessions being loaded -> every connection waiting for that load (all under mu)
  std::deque<std::pair<int, std::string>> load_q;
  std::condition_variable load_cv;
  struct LoadDone { std::string sid; std::shared_ptr<hive::store::Loaded<SeqImage>> img; uint64_t flush_gen; };
  std::deque<LoadDone> attach_q;
  std::map<std::string, std::vector<int>> load_waiting;
  std::atomic<uint64_t> flush_gen{0};  // raised by every flush — a load read before it is not attached after it
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
  // HIVE_SESSION_STORE (hive/session_store.h): prompt checkpoints and boundary snapshots also on disk — set up once the session pool exists
  //   (store_init below), so conversations survive an engine restart or a model switch
  std::unique_ptr<hive::store::SessionStore<SeqImage>> sstore;
  std::atomic<hive::store::SessionStore<SeqImage>*> sstore_live{nullptr};  // the same store for the stats op (socket thread): its counters live
  auto to_store_sig = [](const std::vector<Session::ImgSig>& v) { std::vector<hive::store::Sig> o; for (const auto& x : v) o.push_back({x.start, x.end, x.hash}); return o; };
  auto from_store_sig = [](const std::vector<hive::store::Sig>& v) { std::vector<Session::ImgSig> o; for (const auto& x : v) o.push_back({x.start, x.end, x.hash}); return o; };
  std::map<uint64_t, SharedSnap> shared_snaps;  // prefix hash -> boundary snapshot (shared by all sessions; reclaimed under the same budget in the same recency order)
  long shared_hits = 0, shared_saves = 0;
  size_t host_peak_bytes = 0;
  // Images (and the segments they share) counted incrementally — the full walk this replaced grew with every kept image and ran on
  //   this thread once a second while decoding (measurements in hive/host_ledger.h)
  hive::HostLedger<SeqImage> host_ledger;
  auto host_total = [&](const SeqImage* transient = nullptr) {
    size_t n = rt.snapshot_pool_cached_bytes();
    host_ledger.begin();
    const auto visit = [&](const std::shared_ptr<SeqImage>& img) { host_ledger.visit(img); };
    for (const auto& [id, s] : sessions) { n += s.host_bytes(); s.each_image(visit); }
    for (const auto& [id, s] : archived) { n += s.host_bytes(); s.each_image(visit); }
    for (const auto& [key, e] : shared_snaps) { n += sizeof(e) + e.sig.capacity() * sizeof(Session::ImgSig); visit(e.image); }
    n += host_ledger.end();
    if (transient) n += host_ledger.extra(transient);
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
  // HIVE_SESSION_STORE=<dir> (unset or empty = off: nothing is written or read, responses and logs unchanged) · HIVE_SESSION_STORE_GB (default
  //   64) = disk budget, oldest entries dropped first. The layout key = this daemon + kStoreFormat + the checkpoint's config.json + the shape
  //   of an image (buffer counts, layer keys) of a fresh sequence: a store written for another model or layout is never read. The build is
  //   not in the key — a rebuild resumes the store once its canary agrees (store_canary, before ready).
  //   Set up here and, when the daemon starts asleep, at the first wake (the shape needs a sequence from the pool).
  const std::string store_root = getenv("HIVE_SESSION_STORE") && strcmp(getenv("HIVE_SESSION_STORE"), "0") != 0 ? getenv("HIVE_SESSION_STORE") : "";  // "" / "0" = off
  const double store_gb = getenv("HIVE_SESSION_STORE_GB") && *getenv("HIVE_SESSION_STORE_GB") ? std::max(0.000001, atof(getenv("HIVE_SESSION_STORE_GB"))) : 64.0;
  auto store_init = [&] {
    if (sstore || store_root.empty() || seq_pool.empty()) return;
    SeqImage probe;
    rt.save_image(*seq_pool.back(), probe);
    probe.wait_ready();
    hive::store::ShapeAr shape;
    probe.fields(shape);
    std::string cfg;
    if (FILE* f = fopen((ckpt + "/config.json").c_str(), "rb")) { char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) cfg.append(b, n); fclose(f); }
    std::string daemon = argv[0];
    if (auto sl = daemon.rfind('/'); sl != std::string::npos) daemon = daemon.substr(sl + 1);
    const std::string id = "store-format " + std::to_string(kStoreFormat) + "\n" + cfg + "\n" + shape.s;
    const std::string key = daemon + "-" + hive::store::hex16(hive::store::hash_bytes(reinterpret_cast<const uint8_t*>(id.data()), id.size()));
    try {
      sstore = std::make_unique<hive::store::SessionStore<SeqImage>>(store_root, key, (size_t)(store_gb * 1073741824.0));
    } catch (const std::exception& e) {
      fprintf(stderr, "[hived] session store off: %s\n", e.what());
      return;
    }
    sstore_live.store(sstore.get());
    const json st = sstore->stats();
    fprintf(stderr, "[hived] session store on: %s · %zu checkpoint(s) · %zu snapshot(s) · %.2f GB (budget %.0f GB)\n", sstore->dir().c_str(),
            st["ckpts"].get<size_t>(), st["snaps"].get<size_t>(), st["bytes"].get<size_t>() / 1073741824.0, store_gb);
  };
  store_init();
  // The store's canary — whether this build reads the images in the store the way the build that wrote them meant (the layout key no longer
  //   names the build). Once per process, on a pooled sequence, before the first request (before ready; at the first wake when started
  //   asleep): a fixed token sequence (a block of 64 ordinary tokens repeated — its continuation is predictable, so the logits are peaked)
  //   of HIVE_STORE_CANARY_TOKENS (default 2048, at most half the context) and HIVE_STORE_CANARY_TAIL (16) more. Runs of one build are not
  //   bit-identical (CPU-computed experts are fp32, the DMA/CPU split follows timing): measured 2026-10-19 on DeepSeek, two starts of the same
  //   build differed by relative L2 0.18 on random-token logits, so a fixed tolerance dropped a compatible store. The noise is therefore
  //   measured here: the prompt is prefilled twice, and the stored canary must agree with this run as closely as the two runs agree with
  //   each other — the same argmax and KL(this run || stored) ≤ HIVE_STORE_CANARY_TOL (default 4) × this run's KL between its two
  //   prefills (clamped to 1e-3 … 0.25 nats), for the prompt logits and for the tail run from the resumed stored image (a changed image layout breaks the
  //   resumed tail, changed numerics the prompt logits). A failure drops the whole directory (and writes this build's canary), as does a
  //   directory with entries but no canary; without a canary the directory is new and this build's canary is written. HIVE_STORE_CANARY=0
  //   = no check (the store is trusted as is). A fake head has no logits: only the resume itself is checked.
  bool canary_done = false;
  auto store_canary = [&] {
    if (canary_done || !sstore || seq_pool.empty()) return;
    canary_done = true;
    if (getenv("HIVE_STORE_CANARY") && strcmp(getenv("HIVE_STORE_CANARY"), "0") == 0) return;
    const double t0 = now_ms();
    const int64_t P = std::max<int64_t>(16, std::min<int64_t>(getenv("HIVE_STORE_CANARY_TOKENS") ? atoll(getenv("HIVE_STORE_CANARY_TOKENS")) : 2048, max_ctx / 2));
    const int64_t Q = std::max<int64_t>(1, std::min<int64_t>(getenv("HIVE_STORE_CANARY_TAIL") ? atoll(getenv("HIVE_STORE_CANARY_TAIL")) : 16, max_chunk));
    const double scale = getenv("HIVE_STORE_CANARY_TOL") ? atof(getenv("HIVE_STORE_CANARY_TOL")) : 4.0;
    std::vector<int32_t> ids((size_t)(P + Q));
    const int64_t lo = std::min<int64_t>(1000, c.vocab / 4), span = std::max<int64_t>(1, c.vocab / 2);  // ordinary tokens, no specials
    uint64_t x = 0x243F6A8885A308D3ull;
    for (size_t k = 0; k < ids.size(); ++k) {
      if (k < 64) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ids[k] = (int32_t)(lo + (int64_t)(x % (uint64_t)span)); }
      else ids[k] = ids[k - 64];
    }
    Seq& sq = *seq_pool.back();
    ForwardStats fs{};
    auto run = [&](int64_t from, int64_t to, std::vector<float>& lg) {
      for (int64_t i = from; i < to;) {
        const int m = (int)std::min<int64_t>(to - i, max_chunk);
        rt.forward(sq, ids.data() + i, m, nullptr, &lg, &fs, true);
        i += m;
      }
    };
    // KL(a || b) of the two softmax distributions (nats); INFINITY when the argmax differs or the sizes do — 0 for two empty vectors
    auto kl = [](const std::vector<float>& a, const std::vector<float>& b) -> double {
      if (a.size() != b.size()) return INFINITY;
      if (a.empty()) return 0.0;
      if (std::max_element(a.begin(), a.end()) - a.begin() != std::max_element(b.begin(), b.end()) - b.begin()) return INFINITY;
      const double ma = *std::max_element(a.begin(), a.end()), mb = *std::max_element(b.begin(), b.end());
      double za = 0, zb = 0;
      for (size_t k = 0; k < a.size(); ++k) { za += std::exp((double)a[k] - ma); zb += std::exp((double)b[k] - mb); }
      const double la = std::log(za) + ma, lb = std::log(zb) + mb;
      double d = 0;
      for (size_t k = 0; k < a.size(); ++k) { const double pa = std::exp((double)a[k] - la); if (pa > 0) d += pa * (((double)a[k] - la) - ((double)b[k] - lb)); }
      return std::max(0.0, d);
    };
    std::vector<float> lg_prompt, lg_tail, lg_prompt2, lg_tail2, lg_tail_old;
    rt.reset_seq(sq);
    run(0, P, lg_prompt);
    auto img = std::make_shared<SeqImage>();
    rt.save_image(sq, *img);
    img->wait_ready();
    run(P, P + Q, lg_tail);
    rt.reset_seq(sq);  // the second run: this build's own noise
    run(0, P, lg_prompt2);
    run(P, P + Q, lg_tail2);
    // this run's noise, between 1e-3 and 0.25 nats (two prefills that disagree on the argmax would otherwise make the check vacuous)
    const double n_prompt = std::clamp(kl(lg_prompt, lg_prompt2), 1e-3, 0.25), n_tail = std::clamp(kl(lg_tail, lg_tail2), 1e-3, 0.25);
    const size_t had = sstore->entries();
    std::string verdict;
    double k_prompt = -1, k_tail = -1;
    if (sstore->has_canary()) {
      auto L = sstore->load_canary();
      const bool same_tokens = L && L->image && L->image->pos == P && L->image->tokens.size() == (size_t)P && std::equal(L->image->tokens.begin(), L->image->tokens.end(), ids.begin());
      if (same_tokens) {
        rt.reset_seq(sq);
        rt.load_image(sq, *L->image);
        run(P, P + Q, lg_tail_old);
        k_prompt = kl(lg_prompt, L->logits);
        k_tail = kl(lg_tail, lg_tail_old);
      }
      if (same_tokens && k_prompt <= scale * n_prompt && k_tail <= scale * n_tail) verdict = "kept";
      else {
        const size_t n = sstore->clear(true);
        sstore->put_canary(img, lg_prompt);
        verdict = same_tokens ? "differs — " + std::to_string(n) + " entries of an incompatible build dropped"
                              : "unreadable or another canary — " + std::to_string(n) + " entries dropped";
      }
    } else if (had) {
      const size_t n = sstore->clear(true);
      sstore->put_canary(img, lg_prompt);
      verdict = "missing — " + std::to_string(n) + " unverifiable entries dropped";
    } else {
      sstore->put_canary(img, lg_prompt);
      verdict = "written (new store)";
    }
    if constexpr (RtHasTrimSeq<std::remove_reference_t<decltype(rt)>, Seq>) rt_trim_seq(rt, sq);
    rt.reset_seq(sq);
    fprintf(stderr, "[hived] session store canary: %lld+%lld tokens · KL vs stored: prompt %.2e · resumed tail %.2e · this run's noise %.2e / %.2e "
            "(keep ≤ %.3g×) · %s · %.0f ms\n", (long long)P, (long long)Q, k_prompt, k_tail, n_prompt, n_tail, scale, verdict.c_str(), now_ms() - t0);
  };
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
#if HIVE_VRAM_LIMIT_ON
  // HIVE_VRAM_BALLAST_RELEASE=1 (opt-in, for two daemons on one card; unset = unchanged behaviour): every startup sizing
  //   decision above (layer placement, VRAM plan, session pool, cache fit) ran with the HIVE_VRAM_LIMIT_MB ballast held, exactly as in a
  //   solo start; the ballast is freed here, just before ready, so a second daemon can start beside this one. Runtime growth (KV, MTP lend,
  //   image buffers) then reads the device-wide free memory instead of the emulated card.
  if (g_vram_ballast_ptr && env_on("HIVE_VRAM_BALLAST_RELEASE")) {
    const size_t b = vram_reserve::release_all();
    g_vram_ballast_ptr = nullptr; g_vram_ballast = 0;
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "[hived] VRAM limit: ballast %.0f MiB released before ready (HIVE_VRAM_BALLAST_RELEASE) — device VRAM free %.0f MiB\n", b / 1048576.0, fr / 1048576.0);
  }
#endif
  // HIVE_VRAM_LIMIT_MB kept (ballast not released): the ballast (outside the limit) goes back to the card and the headroom inside the limit
  //   becomes the reserve — from here on runtime allocations take their room from it, so the process stays at the limit (vram_reserve.h).
  //   DeepSeek only: after ready its sizes are fixed (session pool, slots; wake restores at most what sleep released) and runtime buffers
  //   are DevBuf / GrowBuf. GLM, Qwen and Q27 size KV and cache growth at runtime from the device's free memory (kv_reserve, make_room,
  //   regrow_cache, vram_avail) — with the ballast returned they would read the rest of the card as theirs, so they keep the ballast
  //   (the limit as before; a refused allocation still fails only the request).
  if (vram_reserve::held() && kVramReserve) {
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);  // the ballast is still held: what is free is this process's headroom inside the limit
    const size_t ballast = vram_reserve::held();
    const size_t kept = vram_reserve::rebase(fr);
    g_vram_ballast = kept;
    vram_reserve::arm();
    g_vram_limit_armed = vram_limit_bytes;
    size_t fr2 = 0;
    cudaMemGetInfo(&fr2, &tot);
    fprintf(stderr, "[hived] VRAM limit: ballast %.0f MiB returned to the card · headroom %.0f MiB inside the limit kept as the reserve for runtime allocations "
            "(held for the process lifetime) — device VRAM free %.0f MiB\n", ballast / 1048576.0, kept / 1048576.0, fr2 / 1048576.0);
  }
  if (!start_asleep) store_canary();  // before ready: no request may resume a stored image this build has not checked
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
  double pinned_ms = -1e18, pinned_mib = 0;  // cuda_pinned_mib in host_memory (publish_stats)
  int vram_power = -1;  // the power state of the last device query — a new state queries again (publish_stats)
  auto power_name = [](int p) { return p == kPowerDraining ? "draining" : p == kPowerSleeping ? "sleeping" : p == kPowerWaking ? "waking" : "ready"; };
  // Mutable engine state never crosses threads. Stats readers only copy this snapshot.
  std::mutex stats_mu;
  json stats_snapshot;
  double last_stats = 0;
  // ---- where / load: where each session's reusable state is (2026-10-22: a per-conversation cache-location column with a manual reload in
  //   the operator's console — a session in VRAM, in RAM or only on disk costs the next turn nothing, a short restore, or a disk read + restore).
  //   Tiers, fastest first:
  //     vram — the session holds a pooled sequence whose KV is its token history (seq present, not broken, pos == tokens): a request that
  //            extends those tokens prefills only the rest. tokens = that history's length. bytes = 0: the device KV comes from the session
  //            pool reserved at start-up and is not attributed per session.
  //     ram  — host images of the session (prompt-end checkpoint, the last state captured at eviction, history frames): a resident session
  //            whose sequence was offloaded (sleep level 2) or an archived one. tokens = the furthest image position. bytes = those images'
  //            host bytes, shared segments counted once.
  //     ssd  — the session's prompt checkpoint in the session store. tokens = its position. bytes = the byte ranges a load reads.
  //   tokens is the most a next request can reuse from that tier: it reuses the common prefix of its prompt with that state (the whole state
  //   only when the prompt extends it exactly). A session in several tiers reports the fastest one at the top and every tier under "tiers".
  //   The engine thread publishes the vram/ram view with the stats (at most a second old — at once at every request end, prefill chunk,
  //   flush, sleep/wake and load); the receiver thread answers {"op":"where"} from it and the store's index, so a long prefill never delays it.
  struct Loc { int64_t vram_tokens = -1, ram_tokens = -1; size_t ram_bytes = 0; };
  std::mutex loc_mu;
  std::shared_ptr<const std::map<std::string, Loc>> loc_snapshot = std::make_shared<const std::map<std::string, Loc>>();
  // per session: the images its ram bytes were computed for (weak — never keeps an image alive) — recomputed only when they change
  struct LocBytes { std::vector<std::weak_ptr<SeqImage>> imgs; size_t bytes = 0; };
  std::map<std::string, LocBytes> loc_bytes;
  auto publish_locations = [&] {
    auto m = std::make_shared<std::map<std::string, Loc>>();
    auto ram_of = [&](const std::string& id, const Session& s, Loc& L) {
      std::vector<std::shared_ptr<SeqImage>> v;
      s.each_image([&](const std::shared_ptr<SeqImage>& p) { if (p) v.push_back(p); });
      if (v.empty()) return;
      auto& e = loc_bytes[id];
      bool same = e.imgs.size() == v.size();
      for (size_t k = 0; same && k < v.size(); ++k) same = !e.imgs[k].owner_before(v[k]) && !v[k].owner_before(e.imgs[k]);
      if (!same) {
        std::set<const void*> seen;
        size_t b = 0;
        for (const auto& p : v) b += p->allocated_bytes(seen);
        e.imgs.assign(v.begin(), v.end());
        e.bytes = b;
      }
      for (const auto& p : v) L.ram_tokens = std::max<int64_t>(L.ram_tokens, p->pos);
      L.ram_bytes = e.bytes;
    };
    for (const auto& [id, s] : sessions) {
      Loc L;
      if (s.seq && !s.seq->broken && !s.tokens.empty() && s.seq->pos == (int64_t)s.tokens.size()) L.vram_tokens = (int64_t)s.tokens.size();
      ram_of(id, s, L);
      if (L.vram_tokens >= 0 || L.ram_tokens >= 0) m->emplace(id, L);
    }
    for (const auto& [id, s] : archived) {
      Loc L;
      ram_of(id, s, L);
      if (L.ram_tokens >= 0) m->emplace(id, L);
    }
    for (auto it = loc_bytes.begin(); it != loc_bytes.end();) it = sessions.count(it->first) || archived.count(it->first) ? std::next(it) : loc_bytes.erase(it);
    std::lock_guard<std::mutex> lk(loc_mu);
    loc_snapshot = std::move(m);
  };
  // One session's place (any thread): the published vram/ram view, the store's index, and whether a load is running.
  auto where_json = [&](const std::string& sid) -> json {
    std::shared_ptr<const std::map<std::string, Loc>> snap;
    { std::lock_guard<std::mutex> lk(loc_mu); snap = loc_snapshot; }
    json tiers = json::object();
    if (auto it = snap->find(sid); it != snap->end()) {
      if (it->second.vram_tokens >= 0) tiers["vram"] = {{"tokens", it->second.vram_tokens}, {"bytes", 0}};
      if (it->second.ram_tokens >= 0) tiers["ram"] = {{"tokens", it->second.ram_tokens}, {"bytes", it->second.ram_bytes}};
    }
    if (auto* ss = sstore_live.load()) {
      const auto ci = ss->ckpt_info(sid);
      if (ci.pos >= 0) tiers["ssd"] = {{"tokens", ci.pos}, {"bytes", ci.bytes}};
    }
    json out = {{"where", "none"}, {"tokens", 0}, {"bytes", 0}};
    for (const char* t : {"vram", "ram", "ssd"})
      if (tiers.contains(t)) { out = {{"where", t}, {"tokens", tiers[t]["tokens"]}, {"bytes", tiers[t]["bytes"]}}; break; }
    out["tiers"] = std::move(tiers);
    bool loading;
    { std::lock_guard<std::mutex> lk(mu); loading = load_waiting.count(sid) > 0; }
    if (loading) out["loading"] = true;
    return out;
  };
  // force = at each request end and prefill chunk (publishing only at the top of the loop dropped the last second of a finished request for
    //   the 30 s idle cv wait, and froze during long prefills). Otherwise throttled to once per second.
  auto publish_stats = [&](bool force = false) {
    if (!force && now_ms() - last_stats < 1000) return;
    last_stats = now_ms();
    publish_locations();
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
    // the ledger's own counts (images and distinct segments behind host_session_bytes) — bytes per image and per segment can then be compared
    //   with the pinned mappings (host_memory.cuda_pinned_mib); without them the 2 MiB-chunk share of the pinned memory could not be attributed
    s["host_images"] = host_ledger.images();
    s["host_segments"] = host_ledger.segments();
    s["decode_dma_rows"] = decode_dma_rows;
    s["decode_streamed_records"] = tot_streamed.load();
    s["host_accounting_scope"] = "reuse state/capacities/pinned pool, approximate metadata; not process RSS";
    if (prefix_share) s["prefix_share"] = {{"entries", shared_snaps.size()}, {"hits", shared_hits}, {"saves", shared_saves}};
    if (sstore) s["session_store"] = sstore->stats();
    s["state"] = power_name(power.load());  // sleep/wake state
    s["max_ctx"] = max_ctx;  // the server uses it to reject over-window requests with 400 before the stream starts
    // even on forced publishes (request end, every chunk) query the device at most once per second (every time during transitions, and once
    //   more on the publish that ends one: measured 2026-10-19, a wake's last publish came < 1 s after its previous query, so /health showed
    //   the mid-wake reserve (672 MiB, really 2,432) until the idle loop's next sweep — seconds to minutes)
    if (power.load() != kPowerReady || last_stats - vram_ms >= 1000.0 || vram_power != power.load()) {
      size_t fr = 0, tot = 0;
      cudaMemGetInfo(&fr, &tot);
      const size_t held = vram_reserve::held();
      if (g_vram_limit_armed) {  // the process is the limit: its reserve is its free part (the device-wide numbers include other processes on the card)
        vram_json = {{"free_mib", held / 1048576.0}, {"used_mib", (double)(g_vram_limit_armed - std::min(held, g_vram_limit_armed)) / 1048576.0},
                     {"total_mib", g_vram_limit_armed / 1048576.0}, {"device_free_mib", fr / 1048576.0}};
        vram_json["limit_reserve_mib"] = held / 1048576.0;
      } else {
        vram_json = {{"free_mib", fr / 1048576.0}, {"used_mib", vram_used_mib(tot, fr)}, {"total_mib", (double)(tot - held) / 1048576.0}};
        if (held) vram_json["limit_ballast_mib"] = held / 1048576.0;  // HIVE_VRAM_LIMIT_MB: used / total exclude it
      }
      host_mem_json = proc_memory_mib();
      host_mem_json["experts_pinned_mib"] = store.experts_host_bytes() / 1048576.0;
      host_mem_json["engram_mib"] = store.engram_host_bytes() / 1048576.0;
      host_mem_json["dense_host_copy_mib"] = devmem::shadow_bytes() / 1048576.0;  // sleep shadows (pinned even while awake)
      if (last_stats - pinned_ms >= 10000.0) { pinned_mib = cuda_pinned_bytes() / 1048576.0; pinned_ms = last_stats; }
      host_mem_json["cuda_pinned_mib"] = pinned_mib;  // every cudaHostAlloc mapping (see cuda_pinned_bytes), refreshed at most every 10 s
      vram_ms = last_stats;
      vram_power = power.load();
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
                  {"sessions_offloaded", 0}, {"freed_mib", fr1 > fr0 ? (fr1 - fr0) / 1048576.0 : 0.0}, {"vram_used_mib", vram_used_mib(tot, fr1)},
                  {"vram_scope", "device-wide cudaMemGetInfo — exact for hived only while no other process holds VRAM (per-process: nvidia-smi --query-compute-apps)"},
                  {"vmm", vmm_info}, {"host_memory", pm}};
    fprintf(stderr, "[hived] started asleep (level %d): dense/work %.0f MiB to RAM in %.0f ms · VRAM used %.0f MiB · host RSS %.0f MiB (experts pinned %.0f · engram %.0f · "
            "dense copy %.0f MiB) · first wake allocates the cache and warms %zu experts%s\n", sleep_level, r.bytes / 1048576.0, r.ms, vram_used_mib(tot, fr1),
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
        if (auto* ss = sstore_live.load()) s["session_store"] = ss->stats();  // live (writes finish on the store's thread after a publish)
        send_json(fd, s);
        finish_socket(fd); incoming.erase(fd);
        continue;
      }
      if (op == "where") {  // {"sessions":[sid,...]} (or "session":sid) — answered here from the published view, never queued behind a request
        json out = {{"ok", true}, {"sessions", json::object()}};
        std::vector<std::string> ids;
        if (j.contains("sessions") && j["sessions"].is_array())
          for (const auto& v : j["sessions"]) if (v.is_string() && ids.size() < 4096) ids.push_back(v.get<std::string>());
        if (j.contains("session") && j["session"].is_string()) ids.push_back(j["session"].get<std::string>());
        for (const auto& id : ids) out["sessions"][id] = where_json(id);
        send_json(fd, out);
        finish_socket(fd); incoming.erase(fd);
        continue;
      }
      if (op == "set") {  // run-time switches (only the keys below; others are ignored, values absorbed: number → 1 if > 0, 0 if == 0, -1 if < 0 = startup value)
        json out = {{"ok", true}};
        if (j.contains("mtp_batch") && j["mtp_batch"].is_number()) {
          const double v = j["mtp_batch"].get<double>();
          g_mtp_batch_override.store(!std::isfinite(v) || v < 0 ? -1 : v > 0 ? 1 : 0);
        }
        out["mtp_batch"] = g_mtp_batch_override.load();
        send_json(fd, out);
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
      if (op == "load") {  // {"session":sid} — read on the loader thread, attached by the engine thread; one reply line when done
        const std::string sid = j.value("session", "");
        if (sid.empty()) { send_json(fd, {{"error", "load needs a session"}}); finish_socket(fd); incoming.erase(fd); continue; }
        { std::lock_guard<std::mutex> lk(mu); load_q.emplace_back(fd, sid); }
        incoming.erase(fd);
        load_cv.notify_one();
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

  // {"op":"load"} loader thread: a stored prompt checkpoint is read here — off the engine thread and outside every lock (GBs for a long
  //   conversation: the store reads ~6.5 GB/s on the reference disk, so a 7.5 GB image is over a second) — with the same reader the request path
  //   uses (SessionStore::load_ckpt_detached = load_ckpt with its own pack descriptors), then handed to the engine thread, which attaches it as
  //   an archived session at the top of its loop (handle_attach) exactly as an evicted session is kept: the next request of that session
  //   revives it like any archived one (no disk read on the request path). Rules:
  //   · a session already in VRAM or RAM is left as it is (its place is the answer) — also when the engine took it in while this read ran;
  //   · two loads of one session read once (the second connection waits for the first's answer);
  //   · an image larger than the whole host_session_mb budget is not read (attaching it would evict every other session and still not fit);
  //     otherwise the budget is enforced by archive_trim's least-recently-used rule, keeping the loaded session (it is the newest);
  //   · a flush during the read wins — the image is dropped (flush promises that nothing reusable is left);
  //   · asleep (any level) the attach still happens: archived state is host memory only, the same as a flush or the level-2 offload does
  //     while asleep, and wake does not touch it;
  //   · the thread runs at nice +10 (its read threads inherit it): the read and its hash checks run beside the CPU expert pool.
  std::thread loader([&] {
    if (setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10) != 0) {}  // best effort: a refused priority change keeps the default
    using CkptInfo = hive::store::SessionStore<SeqImage>::CkptInfo;
    for (;;) {
      std::pair<int, std::string> job;
      {
        std::unique_lock<std::mutex> lk(mu);
        load_cv.wait(lk, [&] { return !load_q.empty() || stop_state.load() == 3; });
        if (load_q.empty()) return;
        job = std::move(load_q.front());
        load_q.pop_front();
      }
      const int fd = job.first;
      const std::string& sid = job.second;
      auto answer = [&](bool loaded, const char* reason) {
        json o = {{"ok", true}, {"session", sid}, {"loaded", loaded}};
        o.update(where_json(sid));
        if (reason) o["reason"] = reason;
        return o;
      };
      auto reply = [&](const std::vector<int>& fds, const json& o) { for (int f : fds) { send_json(f, o); finish_socket(f); } };
      if (stop_state.load() != 0) { reply({fd}, {{"error", "daemon shutting down"}}); continue; }
      {
        std::lock_guard<std::mutex> lk(mu);
        if (auto w = load_waiting.find(sid); w != load_waiting.end()) { w->second.push_back(fd); continue; }
      }
      const std::string at = where_json(sid)["where"].get<std::string>();
      if (at == "vram" || at == "ram") { reply({fd}, answer(false, "already in memory")); continue; }
      auto* ss = sstore_live.load();
      const CkptInfo ci = ss ? ss->ckpt_info(sid) : CkptInfo{};
      if (ci.pos < 0) { reply({fd}, answer(false, ss ? "no stored checkpoint" : "no session store")); continue; }
      if ((double)ci.bytes > std::max(0.0, host_session_mb) * 1048576.0) { reply({fd}, answer(false, "larger than the host session budget")); continue; }
      { std::lock_guard<std::mutex> lk(mu); load_waiting[sid].push_back(fd); }
      const uint64_t gen = flush_gen.load();
      std::shared_ptr<hive::store::Loaded<SeqImage>> img(ss->load_ckpt_detached(sid));
      if (!img) {
        std::vector<int> fds;
        { std::lock_guard<std::mutex> lk(mu); fds = std::move(load_waiting[sid]); load_waiting.erase(sid); }
        fprintf(stderr, "[hived] load %.16s: the stored checkpoint could not be read (pos %lld)\n", sid.c_str(), (long long)ci.pos);
        reply(fds, answer(false, "read failed"));
        continue;
      }
      {
        std::lock_guard<std::mutex> lk(mu);
        if (stop_state.load() == 0) attach_q.push_back({sid, std::move(img), gen});
      }
      if (img) {  // shutting down: the engine loop may be gone — answer here
        std::vector<int> fds;
        { std::lock_guard<std::mutex> lk(mu); fds = std::move(load_waiting[sid]); load_waiting.erase(sid); }
        reply(fds, {{"error", "daemon shutting down"}});
        continue;
      }
      cv.notify_all();
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
    std::shared_ptr<SeqImage> ckpt_base;  // this request's deepest boundary snapshot — the prompt checkpoint's delta base (shares its rows)
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
    bool stages = false;  // the request asked for stage lines ("stages": true) — also gets read progress inside a forward (pg_targets)
    bool defer_ckpt = false, defer_warm = false;  // HIVE_DEFER_CKPT: prompt checkpoint and warm_after_prefill deferred until after the first token is sent
    double t0 = 0, t1 = 0;
    double restore_ms = 0;  // host image restores of this request (load_image of a checkpoint / archived / parent / shared snapshot, plus session-store reads) — part of prefill_ms
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
    // Base = this request's deepest boundary snapshot when it took one, else the session's previous checkpoint. After a re-read the old
    //   checkpoint belongs to an earlier generation (nothing shared), so the checkpoint copied the whole prompt again right next to a boundary
    //   snapshot holding the same rows: measured 2026-10-14 27B (192K tokens), 12 GB + 12 GB over its 16 GB host budget — the snapshot was
    //   reclaimed before the next turn could resume from it. save_image checks lineage and prefix and copies everything when they differ.
    std::shared_ptr<SeqImage> base = std::move(X.ckpt_base);
    rt.save_image(*S->seq, *img, base ? base.get() : S->ckpt.get());
    host_total(img.get());  // replacement peak, shared segments counted once
    static const int history_limit = getenv("HIVE_CKPT_HISTORY") ? std::max(1, atoi(getenv("HIVE_CKPT_HISTORY"))) : 1;
    if (S->ckpt && history_limit > 1)
      S->history.push_back({S->ckpt, std::move(S->ckpt_logits), std::move(S->ckpt_sig)});
    while ((int)S->history.size() >= history_limit) S->history.pop_front();
    S->ckpt = std::move(img);
    S->ckpt_logits = logits;
    S->ckpt_sig = S->img_sig;
    if (sstore) sstore->put_ckpt(X.sid, S->ckpt, S->ckpt_logits, to_store_sig(S->ckpt_sig));  // written on the store's thread
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
                 {"mtp_steps", A.mtp_steps}, {"mtp_drafted", A.mtp_drafted}, {"mtp_accepted", A.mtp_accepted}, {"restore_ms", A.restore_ms}};
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
        //   epoch ms (wall_off taken at startup = system clock - now_ms) · queue_ms = time queued by the receiver thread -> t0 (before any restore) ·
        //   restore_ms = host image restores inside prefill_ms · "-" without rid
    const std::string rid_s = A.r.req.value("rid", "");
    fprintf(stderr, "[hived] %s: prefill %zu tok %.0f ms · decode %d tok %.0f ms (%.1f tok/s, batch %zu) · hit %d cpu %d · resident %d/%d · mtp %ld/%ld in %ld steps · %s"
            " · wall t0=%.0f t1=%.0f t2=%.0f queue_ms=%.0f restore_ms=%.0f rid=%s\n",
            A.sid.c_str(), A.prompt_len - A.common, A.t1 - A.t0, A.n, t2 - A.t1, A.n > 1 ? (A.n - 1) * 1000.0 / (t2 - A.t1) : 0.0, active.size(),
            A.ds.n_hit, A.ds.n_cpu, store.n_resident(), store.n_slots(), A.mtp_accepted, A.mtp_drafted, A.mtp_steps, A.finish.c_str(),
            A.t0 + wall_off, A.t1 + wall_off, t2 + wall_off, A.r.t_recv > 0 ? std::max(0.0, A.t0 - A.r.t_recv) : 0.0, A.restore_ms, rid_s.empty() ? "-" : rid_s.c_str());
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
  if (layer_yield && ly_slice > 0) fprintf(stderr, "[hived] layer yield slices on: other prefills start at a yield and run in chunks of <= %zu rows (HIVE_LAYER_YIELD_SLICE)\n", ly_slice);
  if (prio_shared != INT_MIN) fprintf(stderr, "[hived] priority shared: priorities <= %d are one class (HIVE_PRIORITY_SHARED)\n", prio_shared);
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
  //   rsig: the request's image signatures (request_img_sig) — an image request resumes only where the saved images up to that point are its
  //   own (the same rule as admit_setup's images_match); nullptr = a text request.
  auto reuse_lower_bound = [&](const std::string& sid, const json& q, const std::vector<Request::ImgSig>* rsig = nullptr) -> size_t {
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
    auto clear_upto = [&](const std::vector<Session::ImgSig>& sig, size_t pos) {  // no saved image may straddle pos; up to pos the saved images are the request's
      std::vector<std::tuple<int, int, uint64_t>> a, b;
      for (const auto& s : sig) {
        if ((size_t)s.start < pos && (size_t)s.end > pos) return false;
        if ((size_t)s.end <= pos) a.emplace_back(s.start, s.end, s.hash);
      }
      if (rsig)
        for (const auto& s : *rsig) {
          if ((size_t)s.start < pos && (size_t)s.end > pos) return false;
          if ((size_t)s.end <= pos) b.emplace_back(s.start, s.end, s.hash);
        }
      std::ranges::sort(a); std::ranges::sort(b);
      return a == b;
    };
    size_t best = 0;
    if (S->seq && !S->seq->broken && S->seq->pos == (int64_t)S->tokens.size()) {
      const size_t k = lcp(S->tokens);
      if (k == S->tokens.size() && ids.size() > k && clear_upto(S->img_sig, k)) best = k;
    }
    if (rsig && !rsig->empty() && !image_ckpt) return best;  // admit_setup resumes an image request from its live state only (HIVE_IMAGE_CKPT off)
    if (S->ckpt) { const size_t k = lcp(S->ckpt->tokens); if (k == (size_t)S->ckpt->pos && ids.size() > k && clear_upto(S->ckpt_sig, k)) best = std::max(best, k); }
    for (const auto& f : S->history)
      if (f.image) { const size_t k = lcp(f.image->tokens); if (k == (size_t)f.image->pos && ids.size() > k && clear_upto(f.sig, k)) best = std::max(best, k); }
    if (S->live) { const size_t k = lcp(S->live->tokens); if (k == (size_t)S->live->pos && ids.size() > k && clear_upto(S->live_sig, k)) best = std::max(best, k); }
    return best;
  };
  auto rows_to_prefill = [&](const std::string& sid, const json& q, const std::vector<Request::ImgSig>* rsig = nullptr) -> size_t {
    const auto ids_it = q.find("ids");
    const size_t rows = ids_it != q.end() && ids_it->is_array() ? ids_it->size() : 0;
    return rows ? rows - std::min(rows, reuse_lower_bound(sid, q, rsig)) : 0;
  };
  std::set<std::string> prefilling_sids;  // sessions in a batched prefill (not active yet) — excluded from LRU eviction
  std::set<std::string> admitting_sids;   // sessions whose prefill runs now (excluded from the KV reclaim below)
  // KV growth under VRAM pressure (the Qwen family): before the expert cache lends slots to sequence KV, evict the least recently used IDLE
  //   session to a host image and trim its sequence — one per call; false when no idle session holds a sequence. Lending to idle sessions'
  //   KV evicted experts for good (a 600 W FN run, 2026-10-10: 24576 → 13415 resident slots, device routing off from then on).
  if constexpr (RtHasKvReclaim<std::remove_reference_t<decltype(rt)>>) {
    rt_set_kv_reclaim(rt, [&]() -> bool {
      std::unique_lock<std::mutex> lk(mu, std::try_to_lock);
      if (!lk.owns_lock()) return false;
      auto victim = sessions.end();
      for (auto jt = sessions.begin(); jt != sessions.end(); ++jt)
        if (jt->second.seq && !active_sids.count(jt->first) && !prefilling_sids.count(jt->first) && !admitting_sids.count(jt->first) &&
            rt_kv_reclaimable(rt, *jt->second.seq) && (victim == sessions.end() || jt->second.last_used < victim->second.last_used)) victim = jt;
      if (victim == sessions.end()) return false;
      Session& V = victim->second;
      if (!V.tokens.empty() && !V.seq->broken && V.seq->pos == (int64_t)V.tokens.size()) {
        V.live = std::make_shared<SeqImage>();
        rt.save_image(*V.seq, *V.live, V.ckpt.get());
        V.live_sig = V.img_sig;
      }
      rt_trim_seq(rt, *V.seq);
      seq_pool.push_back(std::move(V.seq));
      const std::string id = victim->first;
      archived[id] = std::move(V);
      sessions.erase(victim);
      archive_trim("");
      fprintf(stderr, "[hived] KV reclaim: idle session %.16s moved to a host image (sequence trimmed)\n", id.c_str());
      return true;
    });
  }
  // HIVE_LAYER_YIELD: prompt id count of sessions being prefilled (the "sid:rows" layer-yield header line — lets a monitor tell >= 16K prefill
    // gaps apart). Only when enabled; entries no longer prefilling are swept on insert.
  std::map<std::string, size_t> ly_rows;
  std::map<std::string, int> ly_prio;  // priority of the request being prefilled per session (the layer-yield priority rule below)
  auto ly_mark = [&](const std::string& sid, size_t rows, int prio) {
    if (ly_period <= 0) return;
    for (auto it = ly_rows.begin(); it != ly_rows.end();) it = prefilling_sids.count(it->first) ? std::next(it) : ly_rows.erase(it);
    for (auto it = ly_prio.begin(); it != ly_prio.end();) it = prefilling_sids.count(it->first) ? std::next(it) : ly_prio.erase(it);
    ly_rows[sid] = rows;
    ly_prio[sid] = prio;
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
    std::vector<DevBuf> patch_bufs;  // families without host patch fields (see the image loop at admission)
    size_t common = 0, i = 0, chunk_cap = 0, busy_cap = 0;
    int pick = 0, extra_left = 0, n_chunks = 0;
    bool has_images = false, last_upper_skipped = false;
    std::vector<size_t> hints;
    std::shared_ptr<SeqImage> last_boundary;
    bool last_boundary_full = false;  // last_boundary was taken with the upper layers on (a complete state — usable as the checkpoint's base)
    size_t last_snap_at = 0;  // HIVE_PREFIX_INTERVAL: position of this request's last snapshot (or where it resumed) · 0 interval = none for this request
    bool interval_ok = false;
    ForwardStats st{};
    std::unique_ptr<PrefillPending> pending;
    // current chunk (chunk_begin -> chunk_end)
    int M = 0; bool upper_needed = true;
    std::vector<ImageInput> chunk_images;
    int hit0 = 0, str0 = 0, cpu0 = 0, dma0 = 0; double wait0 = 0, span0 = 0;
    long ly_n = 0; double ly_last = 0;  // HIVE_LAYER_YIELD: layer yields inside this forward · last resume -> forward end (ms) — tail of the chunk log line (monitor gaps)
    // the session's entry in admitting_sids (excluded from the KV reclaim) for the job's whole life: set in admit_setup, erased when the job ends
    struct SidHold {
      std::set<std::string>* s = nullptr; std::string sid;
      SidHold() = default; SidHold(const SidHold&) = delete; SidHold& operator=(const SidHold&) = delete;
      ~SidHold() { if (s) s->erase(sid); }
    } admit_hold;
  };
  // HIVE_LAYER_YIELD_SLICE: prefills started at a layer yield that run chunk by chunk at the following yields (their sessions are in
  //   prefilling_sids). The engine loop continues the one left when no other prefill runs; drain, stop and flush wait for them like for active.
  std::vector<std::unique_ptr<PJob>> sliced;
  double slice_credit = 0;  // see the layer-yield hooks: the paused prefill's time since the last slice minus that slice's time (ms)
  // The first image span that ends after `at` (null = the rest of the prompt is text). Chunk rules depend on the images of the chunk being cut,
  //   not on whether the request has an image somewhere: the runtime's limits are per forward (runtime.cpp — host/tiled passes and the C1 skip
  //   take only forwards whose own chunk has no image).
  auto next_span_j = [&](const PJob& J, size_t at) -> const ImageInput* {
    const ImageInput* best = nullptr;
    for (const auto& x : J.all_images)
      if ((size_t)x.start + x.types.size() > at && (!best || x.start < best->start)) best = &x;
    return best;
  };
  auto plan_chunk_j = [&](PJob& J, size_t at, size_t cap) -> size_t {
    auto& ids = J.ids;
    const size_t rem = ids.size() - at;
    if (rem <= cap) return rem;
    // HIVE_TAIL_SPLIT: one more cut would leave the last piece below the tail-mode minimum (tail_need) -> pull the boundary in so the last is
        // tail_need. Only if this chunk is also in tail mode (never create a short-path chunk).
    if (tail_split && tail_need > 0 && !next_span_j(J, at) && rem - cap < (size_t)tail_need && rt.tail_mode_for((int)(rem - (size_t)tail_need)))
      return rem - (size_t)tail_need;  // < cap (rem - cap < tail_need)
    return cap;
  };
  // Rows of the chunk starting at `at` under `cap`, image spans applied. One span = one chunk of at most max_chunk rows (the vision path runs in
  //   a single forward, as before): when the next span ends within max_chunk of `at`, this is that span's chunk (it also takes the text around
  //   the span up to max_chunk); otherwise the chunk is text with the full cap and ends at the span's start if it would reach it. Every chunk
  //   re-streams the non-resident experts, so neither side leaves a piece shorter than max_chunk minus the span. Measured 2026-10-21 (DS
  //   service): a 572,678-token cold prefill with one image near the end ran as 35 chunks of 16,384 rows, every one "upper on" (226 s), because
  //   the cap and the C1 skip were decided for the whole request; text requests of the same day ran 98,304-row passes with the tail layers
  //   skipped (19-24 s each).
  auto chunk_rows_j = [&](PJob& J, size_t at, size_t cap) -> size_t {
    const ImageInput* x = next_span_j(J, at);
    if (!x) return plan_chunk_j(J, at, cap);
    const size_t s0 = (size_t)x->start, s1 = s0 + x->types.size();
    if (s0 <= at || s1 - at <= (size_t)max_chunk) return plan_chunk_j(J, at, std::min(cap, (size_t)max_chunk));
    return std::min(plan_chunk_j(J, at, cap), s0 - at);
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
    for (size_t x = from; x < ids.size();) { const size_t m = chunk_rows_j(J, x, cap); p.push_back(m); x += m; }
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
      //   With several (the end of the last message and the reasoning-marker point), the earliest is cut here and the next chunk cuts the next.
      if (ids.size() - h <= 16) { if (!gen_cut) gen_cut = h; continue; }
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
    if (sstore) sstore->put_snap(key, img, to_store_sig(sg), skipped);  // written on the store's thread
    last_boundary = img;
    J.last_boundary_full = !skipped;
    J.last_snap_at = at;
    ++shared_saves;
    fprintf(stderr, "[hived] %s: prefix snapshot at %zu (upper %s · %zu entries)\n", A->sid.c_str(), at, skipped ? "skip" : "on", shared_snaps.size());
    host_total();
    archive_trim(A->sid);
  };
  // admit is split into stages (per-request state = PJob, on the heap): setup -> [chunk_begin -> forward -> chunk_end]* -> finish.
    //   A single request (admit) runs them in order with the same calls. batched = called from the HIVE_BATCH_PREFILL batched path (admit_batch).
  // first_cap: an upper bound of the first chunk's rows (HIVE_LAYER_YIELD_SLICE — a sliced prefill's first chunk is a slice): a boundary
  //   snapshot that skipped the tail layers is resumed only when that first chunk is in tail mode (C1 validity)
  auto admit_setup = [&](Request&& r, bool batched, size_t first_cap = SIZE_MAX) -> std::unique_ptr<PJob> {
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
    // HIVE_SESSION_STORE: a session this daemon does not hold (a restart, a model switch, or long evicted) continues from its checkpoint on
    //   disk — read here on the engine thread, outside the lock (GBs for a long conversation), and attached below like an archived session
    std::unique_ptr<hive::store::Loaded<SeqImage>> disk_ckpt;
    if (sstore) {
      bool held;
      { std::lock_guard<std::mutex> lk(mu); held = sessions.count(A->sid) || archived.count(A->sid); }
      // a boundary snapshot deeper than the checkpoint that matches this prompt (on disk or in memory) serves the request below — then the
      //   checkpoint would be read for nothing (measured 2026-10-15 GLM after a restart: both read, 2.74 GB each, 2.7 s + 2.6 s)
      auto [cpos, ctok] = held ? std::pair<int64_t, int32_t>{-1, -1} : sstore->ckpt_at(A->sid);
      // its last token must match too: a Qwen prompt ends "<think>\n" and the next turn merges it into another token, so a checkpoint at the
      //   prompt end is often no prefix of the next turn (measured 2026-10-15 FN after a restart: 4.94 GB read for nothing, 4.7 s)
      if (cpos > 0 && ((size_t)cpos >= J.ids.size() || J.ids[(size_t)cpos - 1] != ctok)) cpos = -1;
      bool deeper = false;
      const bool text_only = !(q.contains("images") && q["images"].is_array() && !q["images"].empty());  // (J.request_sig is filled below)
      if (cpos >= 0 && prefix_share && text_only) {
        const auto& pids = J.ids;
        for (const auto& [key, pos] : sstore->snap_candidates(cpos, (int64_t)pids.size(), [&](int64_t p) { return pids[(size_t)p]; }))
          if (prefix_key(pids.data(), (size_t)pos, {}) == key) { deeper = true; break; }
        if (!deeper) {
          std::lock_guard<std::mutex> lk(mu);
          for (const auto& [key, e] : shared_snaps)
            if (e.image && e.image->pos > cpos && (size_t)e.image->pos < pids.size() && e.image->tokens.size() == (size_t)e.image->pos &&
                std::equal(e.image->tokens.begin(), e.image->tokens.end(), pids.begin())) { deeper = true; break; }
        }
      }
      if (cpos >= 0 && !deeper) disk_ckpt = sstore->load_ckpt(A->sid);
    }
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
              // the contents now live in the image: trim the sequence (its KV back to the base) — a pooled sequence kept the KV it grew
              if constexpr (RtHasTrimSeq<std::remove_reference_t<decltype(rt)>, Seq>) rt_trim_seq(rt, *V.seq);
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
        else {
          Session fresh;
          if (disk_ckpt) {
            fresh.ckpt = disk_ckpt->image; fresh.ckpt_logits = std::move(disk_ckpt->logits); fresh.ckpt_sig = from_store_sig(disk_ckpt->sig);
            fresh.last_used = now_ms();
          }
          it = sessions.emplace(A->sid, std::move(fresh)).first;
        }
        if (evicted) archive_trim(A->sid);
      }
      S = &it->second;
    }
    if (disk_ckpt && S->ckpt == disk_ckpt->image)
      fprintf(stderr, "[hived] %s: checkpoint restored from the session store (pos %lld · %.2f GB in %.0f ms)\n", A->sid.c_str(),
              (long long)disk_ckpt->image->pos, disk_ckpt->bytes / 1073741824.0, disk_ckpt->ms);
    A->S = S;
    // excluded from the KV reclaim from here on: before its sequence is taken from the pool and its image (checkpoint / archived state) is
    //   loaded. The load allocates KV, under VRAM pressure that allocation runs the reclaim, and the reclaim took the least recently used idle
    //   session holding a sequence: this one (last_used still from its previous request). Measured 2026-10-13, hived_q27 at a 32 GB cap (KV tier),
    //   the same 34.8K-token prompt sent twice: "KV reclaim: idle session … moved to a host image" for the session being
    //   restored, then "q27 image position mismatch" (trimmed to pos 0 under the load) or CUDA "invalid argument" in staged_h2d (KV freed under it).
    //   Held by the job (PJob::admit_hold) until it ends: also through the prefill chunks of the single path, which before ran unprotected when
    //   no prefill / layer yield put the session in prefilling_sids.
    admitting_sids.insert(A->sid);
    J.admit_hold.s = &admitting_sids; J.admit_hold.sid = A->sid;
    auto& request_sig = J.request_sig;
    if (q.contains("images") && q["images"].is_array()) {
      const auto* sig = request_img_sig(A->r);
      if (!sig) { fail("image bytes short"); return nullptr; }
      for (const auto& s : *sig) request_sig.push_back({s.start, s.end, s.hash});
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
    int& pick = J.pick;  // 0 reset · 1 live state · 2 checkpoint · 3 archived last state · 4 shared snapshot · 5 parent conversation's checkpoint
    // The live state is valid only if its KV position equals the session token count — after an exception during prefill or an "image span"
        //   failure, tokens are a prefix but seq.pos is ahead by the chunks already processed; continuing would silently misalign RoPE positions,
        //   ring cells and compressed rows.
    const bool live_ok = S->seq && !S->seq->broken && S->seq->pos == (int64_t)S->tokens.size();  // broken = exception during a forward
    if (live_ok && live_lcp == S->tokens.size() && ids.size() > live_lcp && images_match(S->img_sig,live_lcp)) { common = live_lcp; pick = 1; }
    // Rewind (RtHasRewind): the request shares only a prefix of the live state — typically a chat template that renders the last answer
    //   differently from how it was generated (Gemma: the empty thought channel of the generation prompt is not in the history, so every
    //   turn diverges where the previous answer began). Measured 2026-10-18, a 24-turn append-only conversation: each turn restored a
    //   boundary snapshot and took two new ones (~0.5 GB of sliding-window KV each) — first token 46 → 160 ms over the turns, spikes 220-
    //   280 ms. Cutting the live KV back costs nothing.
    bool rewound = false;
    //   An exact re-request (the whole prompt is a prefix of the live state) rewinds one token short of it, so the last row is read again
    //   for its logits — these families take no prompt checkpoint (below).
    const size_t rw_to = ids.empty() ? 0 : std::min(live_lcp, ids.size() - 1);
    if (pick == 0 && live_ok && rw_to > 0 && rw_to < S->tokens.size() && images_match(S->img_sig, rw_to) && rt_rewind(rt, *S->seq, rw_to)) {
      S->tokens.resize(rw_to);
      common = rw_to; pick = 1; rewound = true;
    }
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
    // A rewinding family resumes an archived last state the request shares only a prefix of, and cuts it back after the load (as the live
    //   state above). Measured 2026-10-19 (Gemma, two voice conversations taking turns): a session moved to a host image by the KV reclaim
    //   came back with 1,468 of its 1,488 tokens shared (the previous answer re-rendered) and was read from the shared head instead (684).
    size_t archived_rw = 0;
    if constexpr (RtHasRewind<std::remove_cvref_t<decltype(rt)>, std::remove_cvref_t<decltype(*S->seq)>>) {
      if (pick == 0 && (!has_images || image_ckpt) && S->live && !ids.empty()) {
        const size_t k = std::min(lcp(S->live->tokens), ids.size() - 1);
        if (k > common && k < (size_t)S->live->pos && images_match(S->live_sig, k)) { common = k; pick = 3; archived_rw = k; }
      }
    }
    // An auxiliary session of a conversation ("<sid>~next" = the server's read-ahead of the next turn, "<sid>~k" = a concurrent request slot)
    //   also continues from the conversation's own prompt checkpoints. Measured 2026-10-16 (DeepSeek, a chat client): a turn's first prompt ends
    //   "<｜Assistant｜><think>" and the next turn renders that step as "<｜Assistant｜><think></think>…" (GLM the same), so the prompt checkpoint
    //   at 13,468 was a prefix of the read-ahead, but the read-ahead looked only at its own session and the shared snapshots and prefilled
    //   all 19,228 tokens from 0 (8.5 s); a 400K conversation's read-ahead re-read 63,835 tokens from an older snapshot the same way.
    //   The parent's state is only copied (load_image) — its session is not touched.
    std::shared_ptr<SeqImage> parent_img;
    std::vector<Session::ImgSig> parent_sig;
    if (!has_images || image_ckpt) {
      if (const auto tilde = A->sid.find('~'); tilde != std::string::npos && tilde > 0) {
        const std::string parent = A->sid.substr(0, tilde);
        const Session* P = nullptr;
        if (auto it = sessions.find(parent); it != sessions.end()) P = &it->second;
        else if (auto ar = archived.find(parent); ar != archived.end()) P = &ar->second;
        auto consider = [&](const std::shared_ptr<SeqImage>& img, const std::vector<Session::ImgSig>& sig) {
          if (!img || (size_t)img->pos <= common) return;
          const size_t k = lcp(img->tokens);
          if (k != (size_t)img->pos || ids.size() <= k || !images_match(sig, k)) return;
          common = k; pick = 5; parent_img = img; parent_sig = sig;
        };
        if (P) {
          consider(P->ckpt, P->ckpt_sig);
          for (const auto& f : P->history) consider(f.image, f.sig);
        }
      }
    }
    if (!S->seq) S->seq = take_seq();
    // the session's token count before the decision below rewrites it — the log line used to print the count after a reset (always 0), which
    //   read as "the session lost its state" when the client had in fact rewritten an earlier part of the prompt (2026-10-06 audit of 145 such
    //   lines: 88 rewritten prompts, 50 identical re-sends, 4 changed outputs, 3 after a cancel — none was a lost session)
    const size_t session_tokens = S->tokens.size();
    // A reset means no saved state of this session is a prefix of the request (the history frames were tried above): they are states the
    //   conversation moved away from. Release them now — kept as the admitted session's state they outranked this request's own boundary
    //   snapshots in the host budget. Measured 2026-10-14, 27B with a 16 GB budget, a 192K conversation re-read from the top: its old 12 GB
    //   checkpoint stayed and the boundary snapshot of the re-read was reclaimed, so the next turn re-read everything again.
    if (pick == 0 && (S->ckpt || !S->history.empty())) {
      S->ckpt.reset(); std::vector<float>().swap(S->ckpt_logits); std::vector<Session::ImgSig>().swap(S->ckpt_sig);
      S->history.clear();
    }
    // Restores from host images are timed into A->restore_ms; t0 (prefill start) is taken before them, so queue_ms is the wait for admission only
    //   and the restore is part of prefill_ms. Before, a checkpoint / archived / parent restore ran before t0 and was reported as queue time
    //   (service log 2026-10-21: prompt-ckpt reuse at 256K+ had a queue_ms median of 83.5 ms against 14 ms for live reuse).
    A->t0 = now_ms();
    auto timed_load = [&](const SeqImage& img) {
      const double tl = now_ms();
      rt.load_image(*S->seq, img);
      A->restore_ms += now_ms() - tl;
      fprintf(stderr, "[hived] %s: restored host image at %lld (%.0f ms)\n", A->sid.c_str(), (long long)img.pos, now_ms() - tl);
    };
    // The session's own state (checkpoint / archived / parent checkpoint) is loaded once the restore point is final — after the shared boundary
    //   snapshots below were considered. Loading it first and then a deeper shared snapshot copied two full images (measured 2026-10-21, DS
    //   service: a 425K session image (~400 MB) loaded and then replaced by a snapshot 2 tokens deeper, 364 ms). A pending load is completed on
    //   every path out of setup (early returns, exceptions), so the session's tokens and its sequence never disagree.
    struct DeferredLoad {
      std::function<void(const SeqImage&)> load; Seq* seq = nullptr;
      std::shared_ptr<SeqImage> img;
      void flush() { if (auto i = std::move(img)) load(*i); }
      ~DeferredLoad() { try { flush(); } catch (...) { if (seq) seq->broken = true; } }  // broken: the next request cannot take the half-restored state as live
    } deferred{timed_load, S->seq.get(), nullptr};
    if (pick == 0 || pick == 2 || pick == 3) {
      if (pick == 0) { rt.reset_seq(*S->seq); S->img_sig.clear(); }
      else {
        const auto& img = pick == 2 ? S->ckpt : S->live;
        if (archived_rw) timed_load(*img);  // a rewinding family cuts the loaded state back right away (its result decides common below)
        else deferred.img = img;
        S->img_sig = pick == 2 ? S->ckpt_sig : S->live_sig;
      }
      if (archived_rw && !rt_rewind(rt, *S->seq, archived_rw)) { rt.reset_seq(*S->seq); S->img_sig.clear(); common = 0; pick = 0; }  // its window is gone
      S->tokens.assign(ids.begin(), ids.begin() + common);
    }
    if (pick == 5) {  // the parent conversation's prompt checkpoint (above) — a complete prompt-end state, so also this request's delta base
      deferred.img = parent_img;
      S->img_sig = parent_sig;
      S->tokens.assign(ids.begin(), ids.begin() + common);
      J.last_boundary = parent_img; J.last_boundary_full = true;
    }
    const size_t archived_lcp = S->live ? lcp(S->live->tokens) : 0;  // where a revived conversation left its archived state (log only)
    S->live.reset();  // an archived last state is used once (or dropped on mismatch) — the session is live again
    // session_tokens too: a session revived from the archive (no sequence, so live_lcp = 0) that resets left no line at all, which read
    //   as a lost session (2026-10-14 service log: a 188K conversation re-read after 7 h with no reuse line)
    if (!S->tokens.empty() || live_lcp || session_tokens)
      fprintf(stderr, "[hived] %s: reuse %zu/%zu via %s (session tokens %zu · common with live %zu · with archived %zu)\n", A->sid.c_str(), common,
              ids.size(), pick == 1 ? (rewound ? "live (rewound)" : "live") : pick == 2 ? "prompt-ckpt" : pick == 3 ? (archived_rw ? "archived (rewound)" : "archived") : pick == 5 ? "parent-ckpt" : "reset", session_tokens, live_lcp, archived_lcp);
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
        // DeepSeek: the patches stay in the request's payload (A->r.bin lives until the request ends) — set_image_patches
        set_image_patches(x, A->r.bin.data() + off, nb, patch_bufs);
        off += nb;
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
        if ((!image_ckpt && span_after) || !sig_ok) { deferred.img.reset(); rt.reset_seq(*S->seq); S->tokens.clear(); S->img_sig.clear(); common = 0; pick = 0; }
      }
    }
    // prefill (chunks) — A->t0 was taken before the restores above
    bool& last_upper_skipped = J.last_upper_skipped;  // did the last chunk run skip the tail layers — if it is cut after that, the tail-layer ring holds stale values and must not be continued
    // Chunking while others decode: chunk sizes trade transfer cost against decoder stalls. Measured: 1023 rows take the grouped path (<1024,
        //   85% CPU experts) at 7.4 s per chunk, slower than a streaming 1K chunk (4.7-6.3 s), so busy chunks use the streaming path. Each prefill
        //   chunk re-streams the non-resident experts (~207 GB) over PCIe, so larger chunks are cheaper: splitting a 9.7K prompt into 2048-token
        //   chunks = 5 chunks = 5x the transfer (measured 61.6 s vs ~16 s unsplit). The default does not split — other decoders run only via
        //   decode_share between chunks (with a single chunk they pause for that chunk).
        // Tile = max_chunk x prefill_tile tokens in one forward (experts transferred once per encoder layer). A chunk holding an image span uses
        // max_chunk per the single-chunk rule (chunk_rows_j); the text of an image request takes the text cap.
    size_t& chunk_cap = J.chunk_cap;  // max_chunk x prefill_slots (host tiles included — equals prefill_tile when off)
    chunk_cap = (size_t)max_chunk * (size_t)rt.prefill_slots();
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
    // HIVE_SESSION_STORE: the deepest boundary snapshot on disk that matches this prompt and lies beyond the session's own state joins the
    //   in-memory snapshots first (the selection below then takes it like any other) — a new conversation after a restart still resumes
    //   from the system/tools block another conversation saved
    if (prefix_share && sstore && (!has_images || image_ckpt)) {
      const auto& pids = J.ids;
      auto cands = sstore->snap_candidates((int64_t)common, (int64_t)pids.size(), [&](int64_t p) { return pids[(size_t)p]; });
      for (const auto& [key, pos] : cands) {
        if (shared_snaps.count(key)) break;  // already in memory
        if (prefix_key(pids.data(), (size_t)pos, sig_upto_j(J, (size_t)pos)) != key) continue;
        if (auto L = sstore->load_snap(key)) {
          shared_snaps[key] = SharedSnap{L->image, from_store_sig(L->sig), L->upper_skipped, now_ms()};
          A->restore_ms += L->ms;
          fprintf(stderr, "[hived] %s: boundary snapshot at %lld restored from the session store (%.2f GB in %.0f ms)\n", A->sid.c_str(), (long long)pos,
                  L->bytes / 1073741824.0, L->ms);
          host_total();
        }
        break;
      }
    }
    if (prefix_share && (!has_images || image_ckpt) && !shared_snaps.empty()) {
      SharedSnap* best = nullptr;
      // in a batched prefill the first chunk may shrink this round to one slot (max_chunk) or the decoders-present size (busy/budget) — measure
            // with that lower bound (tail_mode_for is monotone)
      const size_t restore_cap = std::min(first_cap, batched ? std::min({cap_now(), c1_floor_j(J), (size_t)max_chunk}) : cap_now());
      size_t best_len = common;
      for (auto& [key, e] : shared_snaps) {
        const size_t k = e.image ? (size_t)e.image->pos : 0;
        if (k <= best_len || k >= ids.size() || e.image->tokens.size() != k || !std::equal(e.image->tokens.begin(), e.image->tokens.end(), ids.begin())) continue;
        if (!images_match(e.sig, k)) continue;
        if (e.upper_skipped && !rt.tail_mode_for((int)chunk_rows_j(J, k, restore_cap))) continue;  // C1 validity (the first chunk from k, image spans applied)
        best = &e; best_len = k;
      }
      if (!best && common == 0) {  // log why the deepest snapshot sharing this prompt's head was not taken (2026-10-14: a new conversation
        //   prefilled 8,915 tokens from 0 with a 6,922-token snapshot of the same system block present, and the log could not say why)
        size_t near_k = 0, near_m = 0; const char* why = "";
        for (auto& [key, e] : shared_snaps) {
          if (!e.image) continue;
          const size_t k = (size_t)e.image->pos, n = std::min(e.image->tokens.size(), ids.size());
          size_t m = 0; while (m < n && e.image->tokens[m] == ids[m]) ++m;
          if (m < 256 || m <= near_m) continue;
          near_k = k; near_m = m;
          why = e.image->tokens.size() != k ? "image tokens != pos" : m < k ? "prompt differs" : k >= ids.size() ? "covers the whole prompt"
              : !images_match(e.sig, k) ? "images" : "C1 tail mode";
        }
        if (near_m) fprintf(stderr, "[hived] %s: shared-prefix not used — snapshot at %zu shares %zu tokens (%s)\n", A->sid.c_str(), near_k, near_m, why);
      }
      if (best) {
        deferred.img.reset();  // the deeper shared snapshot replaces the session's own state — one load instead of two
        timed_load(*best->image);
        // the restored rows are exact: this request's later boundary snapshots and its prompt checkpoint share them (HIVE_CKPT_DELTA) instead of
        //   copying the whole prompt again (measured 2026-10-15 GLM 145K: 2.8 GB per checkpoint after a shared-prefix restore, 147 MB with a base)
        if (!best->upper_skipped) { J.last_boundary = best->image; J.last_boundary_full = true; }
        S->tokens.assign(ids.begin(), ids.begin() + best_len);
        S->img_sig = best->sig;
        best->last_used = now_ms();
        ++shared_hits;
        last_upper_skipped = best->upper_skipped;
        fprintf(stderr, "[hived] %s: reuse %zu/%zu via shared-prefix (was %s %zu · upper %s · %zu entries)\n", A->sid.c_str(), best_len, ids.size(),
                pick == 1 ? "live" : pick == 2 ? "prompt-ckpt" : pick == 3 ? "archived" : pick == 5 ? "parent-ckpt" : "reset", common, best->upper_skipped ? "skip" : "on", shared_snaps.size());
        common = best_len; pick = 4;
      }
    }
    deferred.flush();  // the session's own state, when no deeper shared snapshot was taken
    // HIVE_IMAGE_CKPT: images entirely inside the reused prefix are already processed (not fed to chunks again — the loop only covers i >= common).
        //   Removing them from the chunk rules means a new span without images uses the same chunk rules as a text request (tiles, C1, tail split,
        //   budget). Images in the new span keep the rule one span = one chunk. Lossless: chunk boundaries do not change the result (same property
        //   as the tail split), and the prefix is the same state down to the image signatures.
    if (image_ckpt && common > 0 && !all_images.empty()) {
      std::vector<ImageInput> suffix;
      for (auto& x : all_images) if ((size_t)x.start + x.types.size() > common) suffix.push_back(x);
      if (suffix.size() != all_images.size()) {
        all_images = std::move(suffix);
      }
    }
    A->common = common;
    A->prompt_len = ids.size();
    // "stages": true (the API server's HIVE_STAGE_STATUS — the engine status of a request): one line once the reuse is decided, before the
    //   prefill — how much of the prompt is reused and how much will be read. Requests without it get no new line.
    A->stages = q.contains("stages") && q["stages"].is_boolean() && q["stages"].get<bool>();
    if (A->stages && !send_json(A->r.fd, {{"admitted", true}, {"cached", (int)common}, {"total", (int)ids.size()}}))
      A->client_ok = false;
    // Boundaries supplied by the server (the request's "boundaries") — integers only, inside (0, length), sorted and deduplicated. Unknown formats are dropped (absorbed).
    auto& hints = J.hints;
    if (prefix_share && !fake_head && (!has_images || image_ckpt) && q.contains("boundaries") && q["boundaries"].is_array()) {
      for (const auto& b : q["boundaries"])
        if (b.is_number_integer() && b.get<int64_t>() > (int64_t)common && b.get<int64_t>() < (int64_t)ids.size() && b.get<int64_t>() >= (hint_min_tokens >= 0 ? hint_min_tokens : ckpt_min_tokens))
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
    // A long re-read saves its deepest conversation boundary on the engine's own evidence, whatever the server or client knew. When the rows
    //   between the reused prefix and that boundary fill at least one prefill chunk, the cut adds at most one chunk to a run of two or more,
    //   while a next turn that changes the tail (reasoning templates drop the last turn's thinking — GLM clear_thinking, Qwen
    //   last_query_index, DeepSeek) would otherwise re-read at least that chunk again. Measured 2026-10-14 GLM (96 GB, 300 W): a 378K
    //   conversation re-read from the top (its head changed) saved no snapshot because the server saw no tail change → the next turn re-read
    //   379K again (130 s, then 139 s). The server's tracker only knows its own history; a session evicted or reset inside the daemon is
    //   invisible to it.
    //   Only for a conversation that already has a turn (the request's "history": true, from the API server) — a first turn keeps the
    //   server's first-turn policy (HIVE_PREFIX_FIRST_TURN: a system block nobody else sent is not cut).
    const bool history = q.contains("history") && q["history"].is_boolean() && q["history"].get<bool>();
    // A family that rewinds resumes a changed tail from its live state: per-turn boundary snapshots of a conversation would only cost copies
    //   (the first turn's head snapshot, shared by new conversations, is unaffected).
    //   That includes the generation-suffix cut (its point is the next turn re-rendering this prompt's end, which a rewind absorbs).
    if constexpr (RtHasRewind<std::remove_cvref_t<decltype(rt)>, std::remove_cvref_t<decltype(*S->seq)>>) if (history) { J.extra_left = 0; hints.clear(); }
    // HIVE_PREFIX_INTERVAL: the same requests that may take boundary snapshots (a rewinding family continues a changed history from its
    //   live or archived state instead)
    J.interval_ok = prefix_interval > 0 && prefix_share && !fake_head && (!has_images || image_ckpt) &&
                    !RtHasRewind<std::remove_cvref_t<decltype(rt)>, std::remove_cvref_t<decltype(*S->seq)>>;
    J.last_snap_at = common;
    if (history && J.extra_left <= 0 && !hints.empty() && !RtHasRewind<std::remove_cvref_t<decltype(rt)>, std::remove_cvref_t<decltype(*S->seq)>>) {
      size_t deepest = 0;
      for (size_t h : hints) if (ids.size() - h > 16) deepest = h;  // a conversation boundary, not the generation-suffix hint
      if (deepest > common && deepest - common >= J.chunk_cap) {
        J.extra_left = 1;
        fprintf(stderr, "[hived] %s: long re-read (%zu → boundary %zu) — boundary snapshot kept for the next turn\n", A->sid.c_str(), common, deepest);
      }
    }
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
    int M = (int)chunk_rows_j(J, i, chunk_now);
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
    const size_t next_min = rem_after > 0 ? chunk_rows_j(J, i + (size_t)M, next_cap) : 0;
    // the C1 skip needs this chunk free of images (the runtime skips the tail layers only then); the next chunk only has to be in tail mode,
    //   which depends on its rows alone — an image chunk of tail-mode size runs the tail like any other
    J.upper_needed = !(rem_after > 0 && chunk_images.empty() && rt.tail_mode_for(M) && rt.tail_mode_for((int)next_min));
    J.hit0 = st.n_hit; J.str0 = st.n_streamed; J.cpu0 = st.n_cpu; J.dma0 = st.n_dma_rows;
    J.wait0 = st.ms_cpu_wait; J.span0 = st.ms_cpu_span;
    st.tail_rows = 0;  // the runtime overwrites it only for tail-mode chunks — clear it so it reads as a per-chunk value
    J.M = M;
    return true;
  };
  // Read progress inside a forward (layer_yield.h Progress, installed below): the requests of the running prefill forward that asked for stages.
  //   A scope sets them around each forward and restores the previous list (forwards admitted inside a layer yield never report: depth > 0).
  struct PgTarget { int fd; size_t start, M, total; };
  std::vector<PgTarget> pg_targets;
  struct PgScope {
    std::vector<PgTarget>& v; std::vector<PgTarget> old;
    PgScope(std::vector<PgTarget>& v_, std::vector<PgTarget> n) : v(v_), old(std::move(v_)) { v = std::move(n); }
    ~PgScope() { v = std::move(old); }
  };
  auto pg_of = [](PJob& J) { return J.A->stages ? std::vector<PgTarget>{{J.A->r.fd, J.i, (size_t)J.M, J.ids.size()}} : std::vector<PgTarget>{}; };
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
    // HIVE_PREFIX_INTERVAL: at least the interval past the last snapshot, not in the generation tail. A chunk that skipped the tail layers
    //   (C1) is snapshotted too, flagged like a boundary snapshot (restored only when the continuing chunk is in tail mode — the rest of
    //   a long prompt is): on DeepSeek every full chunk of a long prefill skips them (2026-10-19, 98,304-row chunks of a 199,508-token
    //   prompt all "upper skip"), so requiring the upper layers kept nothing.
    if (J.interval_ok && i < ids.size() && ids.size() - i > 16 && i - J.last_snap_at >= prefix_interval && hint_ok_j(J, i))
      snapshot_boundary(i, !upper_needed);
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
    // A family that rewinds keeps no prompt checkpoints: its live state is the conversation's state (cut back when the next request diverges,
    //   moved to a host image when evicted). Measured 2026-10-18, Gemma: each turn's checkpoint copied ~0.5 GB of sliding-window KV, and when
    //   the scratch slot did not fit under the VRAM limit the copy ran on the engine thread — the first token of a 60-turn conversation grew
    //   from 56 to ~230 ms; one-token turn judgments over 256 tokens checkpointed too and stalled the engine 2.4 s.
    constexpr bool kRewinds = RtHasRewind<std::remove_cvref_t<decltype(rt)>, std::remove_cvref_t<decltype(*S->seq)>>;
    if (!kRewinds && prefill_complete && !reused_exact && (all_images.empty() || image_ckpt) && (int)ids.size() >= ckpt_min_tokens && !fake_head) {
      if (J.last_boundary_full) A->ckpt_base = J.last_boundary;  // a snapshot with skipped upper layers holds unfinished rows — never a base
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
  // The chunk loop of one prefill (the single path): from J.i to the end of the prompt, or until a cancel / disconnect / error
  //   false = chunk planning failed (chunk_begin answered the error) — the job ends without admit_finish
  std::function<bool(PJob&)> run_chunks;  // defined after continue_sliced (both share it)
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
    ly_mark(J.A->sid, J.ids.size(), J.A->prio);
    if (run_chunks(J)) admit_finish(J);
  };
  // HIVE_LAYER_YIELD_SLICE: a sliced prefill no other prefill is running — continued here as the main prefill (the regular chunk rules; its
  //   own forwards yield, and the other sliced prefills get their slices there). Its session stays in prefilling_sids until it ends.
  auto continue_sliced = [&](size_t k) {
    std::unique_ptr<PJob> Jp = std::move(sliced[k]);
    sliced.erase(sliced.begin() + (long)k);
    PJob& J = *Jp;
    const int fd = J.A->r.fd;
    const auto flag = J.A->r.cancel;
    const std::string sid = J.A->sid;
    struct Erase { std::set<std::string>& s; std::string sid; ~Erase() { s.erase(sid); } } erase{prefilling_sids, sid};
    fprintf(stderr, "[hived] %s: sliced prefill continues as the main prefill · %zu rows left (%zu → %zu)\n", sid.c_str(), J.ids.size() - J.i, J.i, J.ids.size());
    try {
      if (run_chunks(J)) admit_finish(J);
    } catch (const std::exception& e) {
      admit_error(fd, e, sid);
    }
    admit_epilogue(flag, sid);
  };
  run_chunks = [&](PJob& J) -> bool {
    while (J.i < J.ids.size()) {
      if (!chunk_begin(J, SIZE_MAX, false)) return false;
      const double tc0 = now_ms();
      const ly::Stats ly0 = rt.layer_yield_stats();
      if (!rt.in_layer_yield()) { ly_fwd_rows = (size_t)J.M; ly_rest_rows = J.ids.size() - J.i - (size_t)J.M; ly_mid_used = 0; }  // HIVE_LAYER_YIELD_MID
      ++prefill_epoch;
      {
        PgScope pg{pg_targets, pg_of(J)};
        rt.forward(*J.S->seq, J.ids.data() + J.i, J.M, J.chunk_images.empty() ? nullptr : &J.chunk_images, &J.A->logits, &J.st, J.upper_needed);
      }
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
    return true;
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
        if ((int)(active.size() + sliced.size() + busy) + 1 > rt.max_batch()) break;
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
        if (Jp && Jp->i < Jp->ids.size()) { prefilling_sids.insert(Jp->A->sid); ly_mark(Jp->A->sid, Jp->ids.size(), Jp->A->prio); jobs.push_back(std::move(Jp)); return; }
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
        if ((int)(active.size() + sliced.size() + jobs.size()) >= rt.max_batch() || (int)jobs.size() >= rt.prefill_slots()) break;
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
        auto room = [&] { return !jobs.empty() && (int)(active.size() + sliced.size() + jobs.size()) < rt.max_batch() && (int)jobs.size() < rt.prefill_slots() && slots_left() > 0; };
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
        std::vector<PgTarget> pgv;
        for (size_t m = 0; m < multi.size(); ++m) for (auto& t : pg_of(*jobs[multi[m]])) pgv.push_back(t);
        PgScope pg{pg_targets, std::move(pgv)};
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
          {
            PgScope pg{pg_targets, pg_of(J)};
            rt.forward(*J.S->seq, J.ids.data() + J.i, J.M, J.chunk_images.empty() ? nullptr : &J.chunk_images, &J.A->logits, &J.st, J.upper_needed);
          }
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
    //   HIVE_PRIORITY_SHARED: priorities compare by class (prio_class) — the requests of one class all step.
    int best_prio = INT_MAX;
    for (auto& A : active) if (A->client_ok && !A->cancel->load() && A->n < A->max_tokens) best_prio = std::min(best_prio, prio_class(A->prio));
    for (auto& A : active) {
      if (!A->client_ok || A->cancel->load() || A->n >= A->max_tokens) { if (A->cancel->load()) A->finish = "cancel"; continue; }
      if (prio_class(A->prio) > best_prio) { ++A->held_steps; continue; }
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
  static const bool gate2_prior = mtpg::parse_prior(getenv("HIVE_MTP_PRIOR"), gate2.prior);  // HIVE_MTP_PRIOR: the family's acceptance prior (hive/mtp_gate.h)
  (void)gate2_prior;
  static long gate2_epoch = -1;   // prefill_epoch the gate last saw — a change starts its warm-up (Gate::mark_cold)
  if (mtp_gate2 && gate2_epoch != prefill_epoch) { gate2_epoch = prefill_epoch; gate2.mark_cold(); }
  if (mtp_gate2 && stepping.size() == 1) gate2.tick();
  // ---- DSpark speculation (one active sequence that has the target-layer hidden state) ----
    //   Of B drafts, the leading k above the confidence threshold are verified as [tok, d1..dk] -> a sample t' is drawn from row i's logits and
        //   accepted if it equals d_{i+1} (t' = d); otherwise t' is the next token (correction) — either way emitted tokens follow the target
        //   distribution p(.|prefix) (independent of the draft distribution, lossless). If all are accepted, row k's sample is a bonus.
    if (stepping.size() == 1 && rt.mtp_enabled() && mtp_max > 0 && stepping[0]->S->seq->mtp_hidden_valid && !fake_head &&
        (mtp_gate2 ? gate2.known() && (gate2.follow_due() || (stepping[0]->g2.want() && (gate2.can_profit(stepping[0]->temperature > 0.f, mtp_max) || gate2.probe_due() || gate2.small_probe_due())))
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
        // greedy request: the device candidates carry the argmax — the runtime skips the rows × vocab logit copy (Qwen: ~0.5 ms per step with
        //   the per-row copies below); the full rows are only a fallback of the sampled path
        rt_set_verify_logits(rt, A.temperature > 0.f);
        rt.forward_verify(seq, ids.data(), (int)ids.size(), rows, &ds);
        rt_set_verify_logits(rt, true);
        sh_f1 = now_ms(); sh_kind = 0;
        ema(ema_verify_ms[std::min(8, k + 1)], now_ms() - tv);
        const int V = c.vocab;
        int n_keep = 1, n_cmp = 0;  // n_cmp: drafts compared (for gate 2 acceptance learning)
        int32_t nxt = -1;
        bool cont = true;
        Cands rc;
        auto sample_row = [&](int i) {
          snapshot_cands(rt, i, rc);
          if (rows.empty() && rc.valid && A.temperature <= 0.f) return rc.argmax;   // greedy without the full rows (see above)
          std::vector<float> row(rows.begin() + (size_t)i * V, rows.begin() + (size_t)(i + 1) * V);
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
    static const bool mtp_batch_env = env_on("HIVE_MTP_BATCH");
    const int mtp_batch_ov = g_mtp_batch_override.load(std::memory_order_relaxed);
    const bool mtp_batch = mtp_batch_ov < 0 ? mtp_batch_env : mtp_batch_ov > 0;
    rt.set_mtp_batch(mtp_batch);  // a plain flag store each step (engine thread) — mtp_batch_enabled() still requires the startup allocation
    static mtpg::Gate gate_b;
    static const bool gate_b_prior = mtpg::parse_prior(getenv("HIVE_MTP_PRIOR"), gate_b.prior);  // same prior as gate2
    (void)gate_b_prior;
    static long gate_b_epoch = -1;  // same warm-up rule as gate2 (Gate::mark_cold)
    if (mtp_batch && gate_b_epoch != prefill_epoch) { gate_b_epoch = prefill_epoch; gate_b.mark_cold(); }
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
      if constexpr (RtHasDraftBatch<std::remove_reference_t<decltype(rt)>, Seq, ForwardStats>) {
        // one MTP chain of S rows (the Qwen family): the per-sequence chains were S × the draft cost at c4 / c8
        std::vector<Seq*> qv;
        std::vector<int32_t> tv2;
        std::vector<int> at;
        for (int s = 0; s < S; ++s) {
          Seq& q = *stepping[s]->S->seq;
          if (!q.mtp_hidden_valid) continue;  // no hidden state -> one row without drafts
          qv.push_back(&q); tv2.push_back(step_ids[s]); at.push_back(s);
        }
        if (!qv.empty()) {
          std::vector<std::vector<int32_t>> dv;
          std::vector<std::vector<float>> cv;
          ForwardStats dst{};
          rt_draft_batch(rt, qv, tv2, dv, cv, &dst);
          for (size_t j = 0; j < at.size(); ++j) { bdr[at[j]] = std::move(dv[j]); bcf[at[j]] = std::move(cv[j]); }
          draft_routed += dst.n_routed; draft_hit += dst.n_hit; draft_cpu += dst.n_cpu; draft_dma_rows += dst.n_dma_rows;
          n_drafts_made += (int)qv.size();
        }
      } else {
      for (int s = 0; s < S; ++s) {
        Seq& q = *stepping[s]->S->seq;
        if (!q.mtp_hidden_valid) continue;  // no hidden state -> one row without drafts
        ForwardStats dst{};
        rt.mtp_draft(q, step_ids[s], bdr[s], bcf[s], &dst);
        draft_routed += dst.n_routed; draft_hit += dst.n_hit; draft_cpu += dst.n_cpu; draft_dma_rows += dst.n_dma_rows;
        ++n_drafts_made;
      }
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
        {
          bool any_sampled_v = false;
          for (Active* X : stepping) any_sampled_v |= X->temperature > 0.f;
          rt_set_verify_logits(rt, any_sampled_v);
        }
        rt.forward_verify_batch(parts, brows, &ds);
        rt_set_verify_logits(rt, true);
        sh_f1 = now_ms(); sh_kind = 1;
        const int V = c.vocab;
        std::vector<int32_t> bnxt(S, -1);
        std::vector<char> bcont(S, 1);
        std::vector<int> bcmp(S, 0);
        for (int s = 0; s < S; ++s) {
          Active& A = *stepping[s];
          Cands rc;
          auto sample_at = [&](int r) {
            snapshot_cands(rt, r, rc);
            if (brows.empty() && rc.valid && A.temperature <= 0.f) return rc.argmax;
            std::vector<float> rowv(brows.begin() + (size_t)r * V, brows.begin() + (size_t)(r + 1) * V);
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
      bool any_sampled_d = false;
      for (Active* X : stepping) any_sampled_d |= X->temperature > 0.f;
      // greedy batch: no rows × vocab copy (16 rows = 16 MB per step) — the device candidates carry the argmax. Only where the runtime has
      //   candidates (n_cands > 0): otherwise sample_cands falls back to the full rows, which must then exist.
      const bool skip_logits = !any_sampled_d && !fake_head && rt.n_cands() > 0;
      if (skip_logits) step_logits.clear();
      rt.forward_batch(step_seqs, step_ids.data(), step_next, skip_logits ? nullptr : &step_logits, &ds);
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
  // A queued request's priority — same reading as the Active's prio (a number clamped to ±1000, anything else 0)
  auto req_prio = [](const Request& x) {
    const auto it = x.req.find("priority");
    if (it == x.req.end() || !it->is_number()) return 0;
    const double v = it->get<double>();
    return std::isfinite(v) ? (int)std::max(-1000.0, std::min(1000.0, v)) : 0;
  };
  // HIVE_LAYER_YIELD priority rule: a queued request whose priority beats every request being prefilled ("urgent") is admitted at a layer
    //   yield up to the rows the runtime can lend (<= max(HIVE_LAYER_YIELD_MAX, HIVE_LAYER_YIELD_MID) and <= R) without the mid rule's
    //   "much smaller than what is left" test, best priority first; and while such a request decodes, the yield keeps stepping it (no
    //   share/step budget) — the paused prefill resumes when it finishes. Measured 2026-10-18 (DeepSeek, a background-prefill probe — numbers in docs/performance.md step 34):
    //   decode was held by priority since 10-08 but prefill was not — a conversation turn queued behind a background prefill waited for the
    //   paused forward unless it passed the mid rule, and once admitted it decoded only a few steps per 150 ms yield. Requests without the
    //   field are all 0: no request is urgent and nothing changes.
  if (layer_yield) {
    // HIVE_LAYER_YIELD_SLICE: a sliced prefill runs inside the yields of the paused one — it is not the prefill a yield pauses
    auto is_sliced = [&](const std::string& sid) {
      for (const auto& Jp : sliced) if (Jp && Jp->A->sid == sid) return true;
      return false;
    };
    // the priority class (prio_class) of the prefill a yield pauses
    auto paused_prio = [&, is_sliced] {
      int p = INT_MAX;
      for (const auto& sid : prefilling_sids) {
        if (is_sliced(sid)) continue;
        const auto it = ly_prio.find(sid);
        p = std::min(p, prio_class(it != ly_prio.end() ? it->second : 0));
      }
      return p;
    };
    auto urgent_active = [&, paused_prio] {  // paused_prio by value: this block ends before the hooks run
      const int pp = paused_prio();
      for (auto& A : active) if (prio_class(A->prio) < pp && A->client_ok && !A->cancel->load() && A->n < A->max_tokens) return true;
      return false;
    };
    auto h_nested_ok = [&] { return ly_mid > ly_max && rt.in_layer_yield(); };  // in_yield from a layer yield (not HIVE_PREFILL_YIELD) with level 2 on
    auto ly_mid_ok = [&](size_t rows) {  // HIVE_LAYER_YIELD_MID (see its comment): much smaller than the paused forward's remaining rows
      if (ly_mid <= ly_max || rows <= ly_max || rows > ly_mid) return false;
      const double p = std::min(1.0, std::max(0.0, rt.layer_yield_progress()));
      const double rem = (double)ly_fwd_rows * (1.0 - p) + (double)ly_rest_rows;
      return 2.0 * (double)rows <= rem && ly_mid_used + rows <= ly_fwd_rows;
    };
    // Rows a queued request would prefill inside a yield (ids minus the reusable prefix — a lower bound of what the forward will hold), or 0 =
    //   it cannot run there: its session is running or prefilling, or it has an image still to encode (the vision encoder may have to be
    //   loaded — VRAM — in the middle of a long prefill). Images inside the reusable prefix are not encoded again, so such a request runs like
    //   a text one on families that keep the patches on the host until they are encoded (kHostPatches — the others copy every image to the
    //   device at admission). Measured 2026-10-25 20:51 (production log, DeepSeek): an agent conversation's turn — 12 images, all inside its
    //   first 165K tokens, 24,639 new rows — waited 156 s behind a read-ahead prefill because every image request was refused here.
    auto yield_rows = [&](Request& r) -> size_t {
      const json& q = r.req;
      const std::string sid = q.value("session", "default");
      if (active_sids.count(sid) || prefilling_sids.count(sid)) return 0;
      const bool has_img = q.contains("images") && q["images"].is_array() && !q["images"].empty();
      const std::vector<Request::ImgSig>* rsig = nullptr;
      if (has_img) {
        if constexpr (!kHostPatches<ImageInput>) return 0;
        try { rsig = request_img_sig(r); } catch (const std::exception&) { return 0; }  // a malformed image entry: admission answers it later
        if (!rsig) return 0;
      }
      const size_t rows = rows_to_prefill(sid, q, rsig);
      if (rsig) {
        const size_t from = q["ids"].size() - rows;  // every image must end inside the reusable prefix
        for (const auto& s : *rsig) if ((size_t)s.end > from) return 0;
      }
      return rows;
    };
    auto ly_rows_of = [&, ly_mid_ok, paused_prio, yield_rows](Request& r, size_t cap) -> size_t {
      const size_t rows = yield_rows(r);
      if (rows == 0 || rows > cap) return 0;
      if (rows > ly_max && !ly_mid_ok(rows) && !(prio_class(req_prio(r)) < paused_prio())) return 0;
      return rows;
    };
    auto ly_seat = [&] { return (int)(active.size() + prefilling_sids.size()) + 1 <= rt.max_batch(); };
    // HIVE_LAYER_YIELD_SLICE (see its comment). slice_credit = the paused prefill's time since the last slice minus that slice's time (ms):
    //   a slice of the paused prefill's own class is due once it is >= 0 (the two take turns of about equal length).
    auto slice_pick = [&]() -> int {  // the sliced prefill to run next: best class, then the one whose last slice is oldest (front)
      int k = -1;
      for (size_t i = 0; i < sliced.size(); ++i)
        if (k < 0 || prio_class(sliced[i]->A->prio) < prio_class(sliced[(size_t)k]->A->prio)) k = (int)i;
      return k;
    };
    auto slice_due = [&, paused_prio](int k, double gap_now) {
      if (k < 0) return false;
      const int c = prio_class(sliced[(size_t)k]->A->prio), pc = paused_prio();
      if (c < pc) return true;    // a better class: a slice at every yield
      if (c > pc) return false;   // a worse class waits for the paused prefill
      return slice_credit + gap_now >= 0;
    };
    // a queued request that would start as a sliced prefill: no rule above takes it whole, and its class is not worse than the paused one's
    auto slice_rows_of = [&, yield_rows, ly_rows_of, paused_prio](Request& r) -> size_t {
      if (ly_slice == 0 || prio_class(req_prio(r)) > paused_prio()) return 0;
      if (ly_rows_of(r, std::max(ly_max, ly_mid)) > 0) return 0;
      return yield_rows(r);
    };
    ly::Hooks h;
    h.period_ms = ly_period;
    // HIVE_LAYER_YIELD_MID: a mid-size prompt admitted at a yield prefills for seconds — its own forward yields once more (level 2), for decode
    //   steps only (measured without it: the running decoder's longest stall 1.83 → 5.13 s while a 12K-row prompt was admitted into an 85K prefill).
    //   A slice (HIVE_LAYER_YIELD_SLICE) yields the same way.
    h.max_depth = ly_mid > ly_max || ly_slice > 0 ? 2 : 1;
    h.want = [&, ly_rows_of, ly_seat, h_nested_ok, slice_pick, slice_due, slice_rows_of]() -> int {
      if (in_yield) return (h_nested_ok() || (ly_slice > 0 && rt.in_layer_yield())) && !active.empty() ? 1 : 0;
      // HIVE_LAYER_YIELD_INTRA: a point inside a layer runs decode steps only — the runtime parks only the decode floor of rows there and the
      //   paused layer's staging records are held (runtime.cpp "T11b"), so nothing is admitted (requests wait for the next layer boundary)
      if (rt.layer_yield_kind() != ly::kLayer) return active.empty() ? 0 : 1;
      int need = active.empty() ? 0 : 1;
      if (ly_slice > 0) {
        const int k = slice_pick();
        if (slice_due(k, now_ms() - rt.layer_yield_resume_ms()))
          need = std::max(need, (int)std::min(sliced[(size_t)k]->ids.size() - sliced[(size_t)k]->i, ly_slice));
      }
      if ((ly_max > 0 || ly_slice > 0) && ly_seat()) {
        std::lock_guard<std::mutex> lk(mu);
        if (stop_state.load() == 0 && !sleep_asked.load())
          for (auto& r : queue) {
            if (ly_max > 0) need = std::max(need, (int)std::min<size_t>(ly_rows_of(r, std::max(ly_max, ly_mid)), (size_t)INT32_MAX));
            if (const size_t x = slice_rows_of(r)) need = std::max(need, (int)std::min(x, ly_slice));
          }
      }
      return need;
    };
    h.run = [&, ly_rows_of, ly_seat, paused_prio, urgent_active, slice_pick, slice_due, slice_rows_of](int R, double gap_ms) {
      const bool nested = in_yield;  // level 2 (inside a forward admitted by a yield): decode steps only
      const int kind = rt.layer_yield_kind();  // HIVE_LAYER_YIELD_INTRA: ≠ kLayer = a point inside a layer — decode steps only (no admission)
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
      fprintf(stderr, "[hived] layer yield: %s · prefill %.0f ms since resume · rows %d · active %zu%s%s%s\n", who.empty() ? "-" : who.c_str(), gap_ms, R, active.size(),
              nested ? " · level 2 (decode only)" : "", kind != ly::kLayer ? " · intra " : "", kind != ly::kLayer ? ly::kind_name(kind) : "");
      const size_t cap = nested || kind != ly::kLayer ? 0 : std::min(std::max(ly_max, ly_mid), (size_t)std::max(0, R));
      const bool slicing = ly_slice > 0 && !nested && kind == ly::kLayer;  // slices start and run at layer points of the main prefill only
      if (slicing && !sliced.empty()) slice_credit += gap_ms;
      int n = 0, n_mid = 0, n_urgent = 0;
      // Admit the best-priority admissible request first (FIFO among equals) — with no priorities this is the queue order as before
      auto admit_some = [&] {
        while (cap > 0 && ly_seat()) {
          Request r;
          bool got = false;
          size_t rows = 0;
          int prio = 0;
          {
            std::lock_guard<std::mutex> lk(mu);
            if (stop_state.load() != 0 || sleep_asked.load()) break;
            auto pick = queue.end();
            for (auto it = queue.begin(); it != queue.end(); ++it) {
              const size_t x = ly_rows_of(*it, cap);
              if (x > 0 && (pick == queue.end() || req_prio(*it) < prio)) { pick = it; rows = x; prio = req_prio(*it); }
            }
            if (pick != queue.end()) { r = std::move(*pick); queue.erase(pick); got = true; }
          }
          if (!got) break;
          if (prio_class(prio) < paused_prio()) ++n_urgent;
          else if (rows > ly_max) { ly_mid_used += rows; ++n_mid; }  // HIVE_LAYER_YIELD_MID
          admit(std::move(r));
          ++n;
        }
      };
      // HIVE_LAYER_YIELD_SLICE: end a sliced prefill (finish = into active, its first token at the next decode step · otherwise it already answered)
      auto end_sliced = [&](size_t k, bool finish) {
        std::unique_ptr<PJob> Jp = std::move(sliced[k]);
        sliced.erase(sliced.begin() + (long)k);
        PJob& J = *Jp;
        const int fd = J.A->r.fd;
        const auto flag = J.A->r.cancel;
        const std::string sid = J.A->sid;
        prefilling_sids.erase(sid);
        if (finish) {
          try { admit_finish(J); } catch (const std::exception& e) { admit_error(fd, e, sid); }
        }
        admit_epilogue(flag, sid);
        if (sliced.empty()) slice_credit = 0;
      };
      // start the queued requests no rule above took (best class first, FIFO among equals), then run one slice when it is due. true = a slice ran
      auto slice_turn = [&]() -> bool {
        if (!slicing) return false;
        bool finished = false;
        for (size_t i = 0; i < sliced.size();) {  // cancelled or disconnected while waiting for a slice: end now (into active, which answers the cancel)
          if (sliced[i]->A->cancel->load() || !sliced[i]->A->client_ok) { end_sliced(i, true); finished = true; }
          else ++i;
        }
        while (ly_seat()) {
          Request r;
          bool got = false;
          {
            std::lock_guard<std::mutex> lk(mu);
            if (stop_state.load() != 0 || sleep_asked.load()) break;
            auto pick = queue.end();
            int prio = 0;
            for (auto it = queue.begin(); it != queue.end(); ++it)
              if ((pick == queue.end() || prio_class(req_prio(*it)) < prio) && slice_rows_of(*it) > 0) { pick = it; prio = prio_class(req_prio(*it)); }
            if (pick != queue.end()) { r = std::move(*pick); queue.erase(pick); got = true; }
          }
          if (!got) break;
          const int fd = r.fd;
          const auto flag = r.cancel;
          const std::string sid = r.req.value("session", "default");
          try {
            auto Jp = admit_setup(std::move(r), false, ly_slice);
            if (Jp && Jp->i < Jp->ids.size()) {
              prefilling_sids.insert(Jp->A->sid);
              ly_mark(Jp->A->sid, Jp->ids.size(), Jp->A->prio);
              if (sliced.empty()) slice_credit = 0;  // the first slice runs at once
              fprintf(stderr, "[hived] %s: sliced prefill started at a layer yield · %zu rows to prefill (%zu → %zu) · priority %d\n", sid.c_str(),
                      Jp->ids.size() - Jp->i, Jp->i, Jp->ids.size(), Jp->A->prio);
              sliced.push_back(std::move(Jp));
              continue;
            }
            if (Jp) { admit_finish(*Jp); finished = true; }  // nothing to prefill (the same prompt again)
          } catch (const std::exception& e) {
            admit_error(fd, e, sid);
          }
          admit_epilogue(flag, sid);
        }
        bool ran = false;
        const int k = slice_pick();
        if (slice_due(k, 0)) {
          PJob& J = *sliced[(size_t)k];
          const size_t rem = J.ids.size() - J.i, lim = std::min((size_t)std::max(0, R), ly_slice);
          // a slice smaller than the streaming path is not worth a forward (the runtime parked fewer rows than asked — the next yield asks again);
          //   after a restored snapshot that skipped the tail layers the first chunk must be in tail mode (C1 validity)
          const bool fits = lim >= std::min(rem, std::min(ly_slice, stream_floor)) &&
                            (!J.last_upper_skipped || rt.tail_mode_for((int)chunk_rows_j(J, J.i, std::min(cap_now_j(J), lim))));
          if (fits) {
            bool done = false, failed = false;
            const double ts = now_ms();
            try {
              if (!chunk_begin(J, lim, false)) failed = true;  // chunk_begin answered the error
              else {
                J.upper_needed = true;  // its next chunk may be a slice of any size: never skip the tail layers before it (C1)
                ++prefill_epoch;
                const ly::Stats ly0 = rt.layer_yield_stats();
                rt.forward(*J.S->seq, J.ids.data() + J.i, J.M, J.chunk_images.empty() ? nullptr : &J.chunk_images, &J.A->logits, &J.st, true);
                J.ly_n = rt.layer_yield_stats().yields - ly0.yields;
                J.ly_last = 0;
                const double tc = now_ms() - ts - (rt.layer_yield_stats().inner_ms - ly0.inner_ms);
                done = !chunk_end(J, tc, (size_t)J.M) || J.i >= J.ids.size();
              }
            } catch (const std::exception& e) {
              admit_error(J.A->r.fd, e, J.A->sid);
              failed = true;
            }
            slice_credit -= now_ms() - ts;
            ran = true;
            if (failed) end_sliced((size_t)k, false);
            else if (done) { end_sliced((size_t)k, true); finished = true; }
            else std::rotate(sliced.begin() + k, sliced.begin() + k + 1, sliced.end());  // round robin within a class
          }
        }
        if (finished && !active.empty()) decode_step();  // the first token of the prefills that just finished
        return ran;
      };
      admit_some();
      const double ta = now_ms();
      if (n) fprintf(stderr, "[hived] prefill yield: %d short request%s admitted in %.0f ms (active %zu · prefilling %zu) · layer%s%s\n", n, n > 1 ? "s" : "", ta - t0,
                     active.size(), prefilling_sids.size(), n_mid ? (" · " + std::to_string(n_mid) + " by the mid rule").c_str() : "",
                     n_urgent ? (" · " + std::to_string(n_urgent) + " by priority").c_str() : "");
      int steps = 0;
      if (!active.empty()) {
        const double budget = ly::decode_budget_ms(gap_ms, ly_share);
        do { decode_step(); ++steps; } while (!active.empty() && steps < ly_steps && now_ms() - ta < budget);
      }
      int slices = slice_turn() ? 1 : 0;
      // Priority hold: while a better-priority request decodes, the yield keeps stepping it until it finishes (more urgent arrivals are
      //   admitted between steps at a layer point, and sliced prefills of a better class get their slices); a stop or sleep request ends the hold
      if (!nested && urgent_active()) {
        const double th = now_ms();
        const int s0 = steps;
        while (urgent_active() && stop_state.load() == 0 && !sleep_asked.load()) {
          if (kind == ly::kLayer) admit_some();
          const int k = slice_pick();
          if (k >= 0 && prio_class(sliced[(size_t)k]->A->prio) < paused_prio() && slice_turn()) { ++slices; continue; }
          decode_step();
          ++steps;
        }
        fprintf(stderr, "[hived] layer yield: priority hold %d steps · %.0f ms (prefilling priority %d)\n", steps - s0, now_ms() - th, paused_prio());
      }
      last_prefill_tc = saved_tc;
      fprintf(stderr, "[hived] layer yield done: admitted %d · decode steps %d%s · %.0f ms\n", n, steps,
              slices || !sliced.empty() ? (" · slices " + std::to_string(slices) + " (sliced " + std::to_string(sliced.size()) + ")").c_str() : "", now_ms() - t0);
    };
    rt.set_layer_yield(std::move(h));
  }
  // HIVE_STAGE_PROGRESS_MS (default 1000; 0 = off): read progress inside a prefill forward for requests that asked for stages — the chunk
  //   notices alone left a long prompt at 0 % for a whole chunk (2026-10-15: GLM ~25 s, DS ~20 s, FN ~9 s). {"progress":i,"total":n} with
  //   i = chunk start + rows × the fraction of layers done (below the chunk end; the chunk notice follows as before) and "partial": true —
//   a layer fraction is not a position, so the API server shows it but fits its time-left cost to whole chunks only. Cost: one clock read per
  //   layer boundary and one short line per period (layer_yield.h Progress) — no GPU sync.
  {
    const char* v = getenv("HIVE_STAGE_PROGRESS_MS");
    const double pg_ms = v && *v ? std::max(0.0, atof(v)) : 1000.0;
    if (pg_ms > 0) {
      ly::Progress p;
      p.period_ms = pg_ms;
      p.fn = [&pg_targets](double f) {
        for (const auto& t : pg_targets) {
          if (t.M == 0) continue;
          const size_t at = t.start + std::min(t.M - 1, (size_t)((double)t.M * f));
          send_json(t.fd, {{"progress", (int)at}, {"total", (int)t.total}, {"partial", true}});  // a closed client is noticed at the chunk notice
        }
      };
      rt.set_progress(std::move(p));
    }
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
    const double MiB = 1048576.0, used = vram_used_mib(tot, fr1) * MiB;
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
    vram_reserve::set_cap(false);  // what sleep frees is for the media engines — not refilled into the reserve
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
        store_init();  // started asleep: the session store needs a sequence for its layout key
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
                 {"vram_used_mib", vram_used_mib(tot, fr)}};
    if (vmm_restored) wake_info["vmm"] = {{"restored_mib", vmm_mib}, {"ms", vmm_ms}};
    // serving again: runtime allocations draw from the reserve (wake itself took back what sleep gave the card) · resume also takes back the
    //   room of the counted buffers sleep freed (the vision encoder) — without it each sleep with the encoder loaded shrank the reserve for good
    if (g_vram_limit_armed) vram_reserve::resume();
    store_canary();  // started asleep: the first wake checks the store before the first request (later wakes: done already)
    fprintf(stderr, "[hived] wake: %.0f ms (alloc %.0f · warm %.0f) · slots %d (before sleep %d) · warmed %d/%zu experts · VRAM used %.0f MiB · level %d%s\n", t2 - t0,
            t1 - t0, t2 - t1, n, store.sleep_phys_slots(), warmed, sleep_keys.size(), vram_used_mib(tot, fr), level_was,
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
    if (!active.empty() || !sliced.empty()) {  // a sliced prefill (HIVE_LAYER_YIELD_SLICE) holds a session like a running request
      for (int fd : fds) ctl_reply(fd, {{"error", "not idle"}, {"running", active.size() + sliced.size()}, {"retryable", true}});
      return;
    }
    size_t n_sess = 0;
    const size_t n_arch = archived.size(), n_snap = shared_snaps.size();
    {
      std::lock_guard<std::mutex> lk(mu);
      flush_gen.fetch_add(1);  // a load still reading is dropped when it arrives (handle_attach)
      for (auto& [id, V] : sessions) {
        if (V.tokens.empty() && !V.ckpt && !V.live && V.history.empty()) continue;
        V.tokens.clear(); V.history.clear(); V.ckpt.reset(); V.live.reset();
        std::vector<float>().swap(V.ckpt_logits);
        V.img_sig.clear(); V.ckpt_sig.clear(); V.live_sig.clear();
        if constexpr (RtHasTrimSeq<std::remove_reference_t<decltype(rt)>, Seq>) { if (V.seq) rt_trim_seq(rt, *V.seq); }   // KV back to the base (cache untouched)
        ++n_sess;
      }
      archived.clear();
      shared_snaps.clear();
    }
    const size_t n_disk = sstore ? sstore->clear() : 0;  // the session store's copies are reusable state too
    rt.trim_snapshot_pool();
    host_total();
    fprintf(stderr, "[hived] flush: cleared %zu session(s) · %zu archived · %zu shared snapshot(s) · %zu stored (expert cache kept)\n", n_sess, n_arch, n_snap, n_disk);
    json fr = {{"ok", true}, {"sessions", n_sess}, {"archived", n_arch}, {"snapshots", n_snap}};
    if (sstore) fr["stored"] = n_disk;
    for (int fd : fds) ctl_reply(fd, fr);
    publish_stats(true);
  };
  // {"op":"load"}: attach the images the loader thread read (rules at the loader). Host-side only, so it runs in every power state.
  auto handle_attach = [&] {
    std::deque<LoadDone> todo;
    { std::lock_guard<std::mutex> lk(mu); todo.swap(attach_q); }
    for (auto& d : todo) {
      const char* reason = nullptr;
      std::vector<int> fds;
      {
        std::lock_guard<std::mutex> lk(mu);
        if (d.flush_gen != flush_gen.load()) reason = "flushed while reading";
        else if (sessions.count(d.sid) || archived.count(d.sid)) reason = "already in memory";
        else {
          Session fresh;
          fresh.ckpt = d.img->image;
          fresh.ckpt_logits = std::move(d.img->logits);
          fresh.ckpt_sig = from_store_sig(d.img->sig);
          fresh.last_used = now_ms();
          archived.emplace(d.sid, std::move(fresh));
          archive_trim(d.sid);
          publish_locations();  // before the session leaves load_waiting: a load queued behind this one must see it in RAM, not read it again
        }
        fds = std::move(load_waiting[d.sid]);
        load_waiting.erase(d.sid);
      }
      const double read_ms = d.img->ms;
      const size_t read_bytes = d.img->bytes;
      fprintf(stderr, "[hived] load %.16s: stored checkpoint (pos %lld · %.2f GB read in %.0f ms) %s\n", d.sid.c_str(), (long long)d.img->image->pos,
              read_bytes / 1073741824.0, read_ms, reason ? reason : "attached in RAM");
      d.img.reset();  // not attached: dropped here (the image reaper frees large pieces off this thread)
      publish_stats(true);  // the reply below reads the published view
      json o = {{"ok", true}, {"session", d.sid}, {"loaded", reason == nullptr}, {"read_ms", read_ms}, {"read_bytes", read_bytes}};
      o.update(where_json(d.sid));
      if (reason) o["reason"] = reason;
      for (int f : fds) ctl_reply(f, o);
    }
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
        std::vector<int> lw;  // loads read but not attached: their waiters are told (the loader answers the jobs still queued)
        {
          std::lock_guard<std::mutex> lk(mu);
          for (auto& d : attach_q) { for (int f : load_waiting[d.sid]) lw.push_back(f); load_waiting.erase(d.sid); }
          attach_q.clear();
        }
        for (int fd : lw) ctl_reply(fd, {{"error", "daemon shutting down"}});
      }
      std::deque<Request> rejected;
      { std::lock_guard<std::mutex> lk(mu); rejected.swap(queue); }
      for (auto& r : rejected) { send_json(r.fd, {{"error", "daemon shutting down"}}); finish_socket(r.fd); }
      if (stop_state.load() >= 2) for (auto& a : active) a->cancel->store(true);
      if (stop_state.load() >= 2) for (auto& Jp : sliced) Jp->A->cancel->store(true);
      if (!sliced.empty()) { continue_sliced(0); continue; }  // in-flight like active: run to completion, or cut at the next chunk once cancelled
      if (active.empty()) break;
      decode_step();
      continue;
    }
    // sleep/wake (without control requests power is always ready — the two branches below are never taken)
    handle_ctl();
    handle_flush();  // {"op":"flush"}: refused while a request decodes; while asleep it clears host-side state only
    handle_attach();  // {"op":"load"}: images the loader thread read — attached in every power state (host memory only)
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
        if (!sliced.empty()) { continue_sliced(0); continue; }  // in-flight: a sliced prefill finishes before the sleep (it holds a session's KV)
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
      cv.wait_for(lk, std::chrono::seconds(1), [&] { return !ctl_q.empty() || !flush_q.empty() || !attach_q.empty() || stop_state.load() != 0; });  // sweep cancelled queued requests every second
      continue;
    }
    // 1) take a new request: wait if active is empty, otherwise take one and prefill it only if a seat (max_batch) is free
    {
      Request r;
      bool got = false;
      int cont = -1;  // HIVE_LAYER_YIELD_SLICE: the sliced prefill to continue as the main prefill
      std::unique_lock<std::mutex> lk(mu);
      if (active.empty() && sliced.empty()) cv.wait_for(lk, std::chrono::seconds(30), [&] { return !queue.empty() || stop_state.load() != 0 || !ctl_q.empty() || !flush_q.empty() || !attach_q.empty(); });  // wake every 30 s for idle cleanup (vision encoder release)
      // Priority: the best-priority queued request first (FIFO among equals), and none below a running request's priority — its
      //   prefill would interrupt that request's decode (the decode_step rule holds it anyway). Without the field everything is 0: FIFO as before.
      //   Priorities compare by class (HIVE_PRIORITY_SHARED). A sliced prefill of the best class (FIFO) goes on before a queued request of the
      //   same class — the queued one then starts at its yields — and a sliced prefill takes a seat like a running request.
      int best_active = INT_MAX;
      for (auto& A : active) best_active = std::min(best_active, prio_class(A->prio));
      int sk = -1;
      for (size_t i = 0; i < sliced.size(); ++i) if (sk < 0 || prio_class(sliced[i]->A->prio) < prio_class(sliced[(size_t)sk]->A->prio)) sk = (int)i;
      const int sk_prio = sk >= 0 ? prio_class(sliced[(size_t)sk]->A->prio) : INT_MAX;
      if (!queue.empty() && (int)(active.size() + sliced.size()) < max_batch && !sleep_asked.load()) {
        auto pick = queue.begin();
        int pick_prio = req_prio(*pick);
        for (auto it = std::next(queue.begin()); it != queue.end(); ++it) { const int p = req_prio(*it); if (p < pick_prio) { pick = it; pick_prio = p; } }
        if (prio_class(pick_prio) <= best_active && prio_class(pick_prio) < sk_prio) { r = std::move(*pick); queue.erase(pick); got = true; }
      }
      if (!got && sk >= 0 && sk_prio <= best_active) cont = sk;
      lk.unlock();
      if (!got && cont >= 0) { last_prefill_tc = 0; continue_sliced((size_t)cont); }
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
  { std::lock_guard<std::mutex> lk(mu); stop_state.store(3); }
  load_cv.notify_all();
  { const char b = 1; const ssize_t w = ::write(stop_pipe[1], &b, 1); (void)w; }
  acceptor.join();
  loader.join();  // a read in progress finishes first (its waiters are answered by the loader)
  ::close(lfd);
  const bool flushed = writer.wait_idle(2000);
  ::unlink(sock_path.c_str());
  fprintf(stderr, "[hived] graceful stop complete (%s) — exit 0\n", flushed ? "all replies flushed" : "reply flush timed out");
  fflush(stderr);
  _exit(0);
}
