// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B on the GPU (hived_q27) with layer-level placement under a VRAM cap. The math contract is hive/q27/q27_model.h; the API is the
//   one of the CPU engine (hive/q27/q27_cpu.h), so the family surface and the checks are shared.
//
// Placement (startup only). Cap = HIVE_Q27_VRAM_MB, else HIVE_VRAM_LIMIT_MB (hived's whole-process limit), else the free VRAM. From the cap
//   the engine subtracts the CUDA context estimate (HIVE_Q27_CTX_MB, default 1000) and the runtime reserve (HIVE_Q27_RESERVE_MB, default:
//   work buffers for HIVE_Q27_CHUNK rows + the dequantization scratch + per-sequence states and a KV allowance for HIVE_Q27_SESSIONS
//   sequences + MTP verify snapshots), then places in this order while the bytes fit: decoder layers from layer 0 upward (each layer whole:
//   its projections, MLP and norms), the lm_head, the MTP block. Every layer that does not fit runs on the CPU (Q27CpuEngine::run_layer,
//   NUMA-split host memory), as does the lm_head when it does not fit; the MTP block is only used when it fits on the GPU.
//   HIVE_Q27_GPU_LAYERS=N forces the first N layers on the GPU (the rest on the CPU) for tests.
//   Layers on the GPU form a prefix, so a forward crosses PCIe once per direction ([M, 5120] bf16 rows) when some layers are on the CPU.
// Weights stay in their checkpoint format in VRAM (FP8 per-row, NVFP4, BF16): decode reads them with GEMV kernels (q27_kernels.h); prefill
//   dequantizes one matrix at a time to its exact bf16 pre-scale values and runs a cuBLAS bf16 GEMM with fp32 output and the scale epilogue.
// The vision encoder runs on the CPU engine (the image embeddings are uploaded as override rows).
#pragma once
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hive/devbuf.h"
#include "hive/q27/q27_cpu.h"
#include "hive/q27/q27_model.h"

namespace hive::q27 {

// device bytes held by VRAM-tier KV buffers (all sequences), for the host-tier decision of HIVE_Q27_KV_TIER
inline std::atomic<size_t> g_kv_dev_bytes{0};
// growable host-tier range (KvBuf, q27_engine.cpp alloc_kv / grow_kv): a reserved address range whose pages are registered (pinned + mapped)
//   piece by piece as a cache grows — the rows stay where they are (no new buffer and copy per growth step) and with UVA the device pointer of
//   every piece is its host pointer, so the range reads as one buffer on the device. Released ranges go back to a pool with their registration
//   (g_kv_host_pool, up to g_kv_host_pool_cap registered bytes; the engine also fills it ahead of use from a background thread, so a long
//   prompt's spill does not pay the pinning: measured 2026-10-12, 32 GB, ~2 GB/s for cudaHostRegister / ~1.3 GB/s for cudaHostAlloc).
// the registered pieces of every growable host range (start → bytes): a cudaMemcpy must not span two registrations (measured 2026-10-13, a
//   staging copy across the boundary of a range extended past its first registration: "invalid argument"); device kernels read across them
inline std::mutex g_kv_regs_mu;
inline std::map<uintptr_t, size_t> g_kv_regs;
// cudaMemcpyAsync split at the registration boundaries of either side (KV copies where a side may be a growable host-tier range)
inline cudaError_t kv_memcpy_async(void* dst, const void* src, size_t n, cudaStream_t st) {
  auto room = [](uintptr_t a) -> size_t {   // bytes to the end of the registered piece holding a (SIZE_MAX: not in a piece)
    auto it = g_kv_regs.upper_bound(a);
    if (it == g_kv_regs.begin()) return SIZE_MAX;
    --it;
    return a < it->first + it->second ? it->first + it->second - a : SIZE_MAX;
  };
  uint8_t* d = static_cast<uint8_t*>(dst);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  while (n) {
    size_t m = n;
    {
      std::lock_guard<std::mutex> lk(g_kv_regs_mu);
      if (!g_kv_regs.empty()) m = std::min({m, room((uintptr_t)s), room((uintptr_t)d)});
    }
    const cudaError_t e = cudaMemcpyAsync(d, s, m, cudaMemcpyDefault, st);
    if (e != cudaSuccess) return e;
    d += m; s += m; n -= m;
  }
  return cudaSuccess;
}
struct KvHostSlot {
  void* base = nullptr;
  size_t res = 0, reg = 0;   // reserved bytes · registered bytes from base
  std::vector<std::pair<void*, size_t>> regs;
  static KvHostSlot* make(size_t reserve) {
    void* a = mmap(nullptr, reserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (a == MAP_FAILED) return nullptr;
    KvHostSlot* s = new KvHostSlot;
    s->base = a; s->res = reserve;
    return s;
  }
  // the bytes [reg, bytes) registered in place (rounded up to 2 MiB); false = not possible (reg unchanged)
  bool extend_to(size_t bytes) {
    if (bytes <= reg) return true;
    bytes = std::min(res, (bytes + ((size_t)2 << 20) - 1) / ((size_t)2 << 20) * ((size_t)2 << 20));
    if (bytes <= reg) return false;
    uint8_t* at = static_cast<uint8_t*>(base) + reg;
    const size_t add = bytes - reg;
    if (mprotect(at, add, PROT_READ | PROT_WRITE) != 0) return false;
    if (cudaHostRegister(at, add, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
      cudaGetLastError(); mprotect(at, add, PROT_NONE); return false;
    }
    void* dp = nullptr;
    if (cudaHostGetDevicePointer(&dp, at, 0) != cudaSuccess || dp != at) {   // the in-place layout needs device pointer == host pointer (UVA)
      cudaGetLastError(); cudaHostUnregister(at); mprotect(at, add, PROT_NONE); return false;
    }
    regs.emplace_back(at, add);
    { std::lock_guard<std::mutex> lk(g_kv_regs_mu); g_kv_regs[(uintptr_t)at] = add; }
    reg = bytes;
    return true;
  }
  void destroy() {
    { std::lock_guard<std::mutex> lk(g_kv_regs_mu); for (auto& r : regs) g_kv_regs.erase((uintptr_t)r.first); }
    for (auto& r : regs) cudaHostUnregister(r.first);
    munmap(base, res); delete this;
  }
};
inline std::mutex g_kv_host_mu;
inline std::vector<KvHostSlot*> g_kv_host_pool;   // free ranges (registered prefix kept)
inline size_t g_kv_host_pool_bytes = 0, g_kv_host_pool_cap = 0;
inline KvHostSlot* kv_host_take(size_t reserve, size_t bytes) {   // a pooled range of this reserve with the largest registered prefix, else new
  {
    std::lock_guard<std::mutex> lk(g_kv_host_mu);
    int best = -1;
    for (int i = 0; i < (int)g_kv_host_pool.size(); ++i)
      if (g_kv_host_pool[i]->res >= std::max(reserve, bytes) && (best < 0 || g_kv_host_pool[i]->reg > g_kv_host_pool[best]->reg)) best = i;
    if (best >= 0) {
      KvHostSlot* s = g_kv_host_pool[best];
      g_kv_host_pool.erase(g_kv_host_pool.begin() + best);
      g_kv_host_pool_bytes -= s->reg;
      return s;
    }
  }
  return KvHostSlot::make(std::max(reserve, bytes));
}
inline void kv_host_give(KvHostSlot* s) {
  std::lock_guard<std::mutex> lk(g_kv_host_mu);
  if (g_kv_host_pool_bytes + s->reg <= g_kv_host_pool_cap) { g_kv_host_pool.push_back(s); g_kv_host_pool_bytes += s->reg; return; }
  s->destroy();
}

// KV rows of one (sequence, attention layer): VRAM (a DevBuf) or — HIVE_Q27_KV_TIER, when the VRAM KV budget is spent — pinned host memory
//   mapped into the device address space (UVA: the device pointer is the host pointer). The attention kernels read either through the same
//   pointer with the same arithmetic, so the values do not depend on the tier.
struct KvBuf {
  DevBuf d;
  void* h = nullptr;
  void* p = nullptr;
  size_t n = 0;
  KvHostSlot* slot = nullptr;   // growable host tier (null: h came from cudaHostAlloc)
  KvBuf() = default;
  ~KvBuf() { reset(); }
  KvBuf(const KvBuf&) = delete;
  KvBuf& operator=(const KvBuf&) = delete;
  KvBuf(KvBuf&& o) noexcept { *this = std::move(o); }
  KvBuf& operator=(KvBuf&& o) noexcept {
    if (this != &o) {
      reset(); d = std::move(o.d); h = o.h; p = o.p; n = o.n; slot = o.slot;
      o.h = nullptr; o.p = nullptr; o.n = 0; o.slot = nullptr;
    }
    return *this;
  }
  bool host() const { return h != nullptr; }
  bool growable() const { return slot != nullptr; }
  void set_dev(DevBuf&& b) { reset(); d = std::move(b); p = d.p; n = d.n; g_kv_dev_bytes += n; }
  void set_host(void* hp, size_t bytes) { reset(); h = hp; p = hp; n = bytes; }
  DevBuf take_dev() { DevBuf b = std::move(d); if (b.p) g_kv_dev_bytes -= b.n; p = nullptr; n = 0; return b; }   // a VRAM buffer back to the caller
  // growable host tier: a range reserving `reserve` bytes with at least `bytes` registered (n = the registered bytes, may be more); false = not possible
  bool set_host_growable(size_t bytes, size_t reserve) {
    reset();
    KvHostSlot* s = kv_host_take(reserve, bytes);
    if (!s) return false;
    if (!s->extend_to(bytes)) { kv_host_give(s); return false; }
    slot = s; h = p = s->base; n = s->reg;
    return true;
  }
  // growable host tier: at least `bytes` registered in place; false = not possible (n unchanged)
  bool extend(size_t bytes) {
    if (!slot) return false;
    if (bytes > slot->res || !slot->extend_to(bytes)) return false;
    n = slot->reg;
    return true;
  }
  void reset() {
    if (h) {
      cudaDeviceSynchronize();   // a kernel may still read it (rare: host-tier buffers only)
      if (slot) kv_host_give(slot); else cudaFreeHost(h);
    }
    if (d.p) g_kv_dev_bytes -= d.n;
    h = nullptr; d.free(); p = nullptr; n = 0; slot = nullptr;
  }
  template <class T> T* as() const { return reinterpret_cast<T*>(p); }
};
// allocation of a KV buffer of `bytes` (VRAM, or host tier) — set by the engine; session image loads use it
inline std::function<void(KvBuf&, size_t)> g_kv_alloc;

struct Q27Seq {
  int64_t pos = 0, cap = 0;
  std::vector<int32_t> tokens;
  int64_t rope_delta = 0;
  std::vector<int32_t> rope;                         // host [pos, 3]
  std::vector<DevBuf> conv, state;                   // per linear layer index (allocated for GPU layers): fp32 [3·10240], [48·128·128]
  struct Kv { KvBuf k, v; int64_t tokens = 0; };    // bf16 [tokens, 4·256] each (VRAM or host tier, KvBuf)
  std::vector<Kv> kv;                                // per attention layer index (GPU layers)
  Kv mtp;
  DevBuf h_last;                                     // bf16 [5120] residual stream of the last position (MTP input)
  bool h_last_valid = false;
  int64_t mtp_written = 0;
  DevBuf df_k, df_v;                                 // block drafter context ring (HIVE_Q27_DFLASH): bf16 [5 layers][df_slots][8·128] each
  int df_slots = 0;
  bool broken = false;
  cpu::CpuSeq cs;                                    // state of the CPU-placed layers
  std::vector<std::vector<uint8_t>> save_bytes() const;
  void load_bytes(const std::vector<std::vector<uint8_t>>& b);
  // session images without intermediate copies: the pieces of an image in order (host bytes kept in `hold`, device ranges copied by the
  //   caller straight into its buffers; `append` = a prefix of the piece is unchanged while the sequence only grows: KV rows) and the load from
  //   contiguous views of those pieces (device copies on `st`, synchronized). save_bytes / load_bytes use the same layout.
  struct Piece { const void* host; const void* dev; size_t n; bool append; };
  void save_plan(std::vector<Piece>& out, std::vector<std::vector<uint8_t>>& hold) const;
  void load_views(const std::vector<std::pair<const uint8_t*, size_t>>& v, cudaStream_t st,
                  const std::function<void(void*, const uint8_t*, size_t)>& h2d = {});   // h2d: the caller's host → device copy (default cudaMemcpyAsync)
};

struct GpuStats {
  double ms_total = 0; long steps = 0; double ms_gpu_layers = 0, ms_cpu_layers = 0, ms_xfer = 0, ms_head = 0, ms_mtp = 0;
};

struct Placement {
  int gpu_layers = 0;            // layers [0, gpu_layers) on the GPU
  bool head_gpu = false, mtp_gpu = false;
  size_t cap = 0, weights = 0, reserve = 0;
};

class Q27Engine {
 public:
  explicit Q27Engine(const std::string& ckpt_dir, bool load_mtp = true);
  // the context the placement plans KV for (KV plan, HIVE_Q27_KV_PLAN; hived's --max-ctx) — call before constructing an engine
  static void set_plan_ctx(int64_t max_ctx);
  ~Q27Engine();
  const Q27Config& cfg() const;
  const Placement& placement() const;
  void init_seq(Q27Seq& s, int64_t cap);
  void reset_seq(Q27Seq& s);
  void release_kv(Q27Seq& s);   // free the sequence's KV buffers (after reset; they grow again on demand)
  // serving-path allocation failure: evict one idle session and return true to retry (hived's KV reclaim), false = nothing left
  std::function<bool()> kv_reclaim;
  // capture the decode / verify CUDA graphs and run one draft before serving (HIVE_Q27_WARMUP=0: off)
  void warmup(int max_batch, int max_k);
  using EmbedOverride = cpu::Q27CpuEngine::EmbedOverride;
  void set_embed_override(std::vector<EmbedOverride> v);
  bool has_vision() const;
  void encode_image(const uint16_t* patches, int gh, int gw, std::vector<float>& out);
  void prefill(Q27Seq& s, const int32_t* ids, int T, float* logits);
  void decode(Q27Seq* const* seqs, const int32_t* ids, int M, float* logits);
  bool mtp_on() const;
  int mtp_k() const;
  int mtp_draft(Q27Seq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  // drafts of S sequences: drafts / conf [S, 7] (row stride 7), n_out[s] drafts each (DFlash2: one stacked pass; MTP: one chain each)
  void mtp_draft_batch(Q27Seq* const* seqs, const int32_t* toks, int S, int k, int32_t* drafts, float* conf, int* n_out);
  void verify_batch(Q27Seq* const* seqs, const int* M, int S, const int32_t* ids, float* logits);
  void rollback_batch(const int* n_keep, int S);
  std::function<void(int layer, const float* h, int rows)> layer_hook;     // host fp32 copy of the residual stream after each layer
  std::function<void(int layer, float* h, int rows)> layer_inject;         // prefill: overwrite the stream in front of layer l ≥ 1
  std::function<void(int layer)> layer_boundary;
  // optional: whether a layer-boundary yield could fire now (period elapsed and work waiting) — when false the boundary is skipped without the
  //   stream sync it needs (measured 2026-10-11 nsys, cold 8K prefill: one sync per layer = 218 gaps, 28 ms of GPU idle with nothing to yield to)
  std::function<bool()> layer_boundary_due;
  GpuStats& stats();
  // logits of the last forward / verify on the device (rows from 0; valid when the lm_head is on the GPU) and the engine's stream
  const float* logits_dev() const;
  // pinned host buffer [16, vocab] the engine stages logits in: passing it as the logits argument of decode / verify_batch skips one copy
  float* logits_host();
  // pin a caller's logits buffer (it must stay at this address for the process lifetime): decode / verify graphs then copy the logits
  //   straight into it. false = not possible (the caller copies from logits_host())
  bool register_host_logits(void* p, size_t bytes);
  // false: decode / verify graphs leave the logits on the device (no rows × vocab D2H; the runtime takes argmax / candidates from
  //   logits_dev()). The caller's logits pointer must still be non-null (it selects the graph path) but is not written.
  void set_host_logits(bool on);
  long graph_captures() const;   // CUDA graphs captured so far (decode / verify steps and draft passes)
  cudaStream_t stream() const;
  cpu::CpuStats& cpu_stats();
  struct Impl;
 private:
  std::unique_ptr<Impl> p_;
};

}  // namespace hive::q27
