// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3-ASR engine (C API in include/hive/qasr/qasr.h). Model description: oracle/qwen3_asr_oracle.py (the fp32 golden model this
//   engine is checked against with tools/test_qasr_engine.py).
//
// Numerics: bf16 weights; GEMM inputs bf16, GEMM outputs and both residual streams fp32; norms, softmax and RoPE in fp32; KV cache bf16.
// Phases (one worker thread, one stream):
//   encode  — audio encoder in passes of `encoder_windows` attention windows (the windows are independent through every layer, so a pass
//             is exactly the reference computation for its tokens); the convolutions run on groups of 4 chunks (im2col + cuBLAS).
//   prefill — the prompt (text embeddings, audio rows written over the <|audio_pad|> positions) in chunks of `prefill_chunk` tokens
//             with cuBLAS GEMMs; causal attention reads the slot's KV cache.
//   decode  — one token for every active sequence (1..max_seqs) per step: a CUDA graph per batch size, GEMVs streaming the weights once
//             for the whole batch; the step reads token / position / slot from a small device array written before each launch.
#include "hive/qasr/qasr.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hive/common.h"
#include "hive/devbuf.h"
#include "hive/qasr/qasr_kernels.h"
#include "hive/safetensors.h"
#include "hive/vram_limit.h"
#include "nlohmann/json.hpp"

namespace hive::qasr {
namespace {

constexpr double kMiB = 1048576.0;

#define QASR_CUBLAS(expr)                                                                                    \
  do {                                                                                                       \
    cublasStatus_t _s = (expr);                                                                              \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                                                       \
      fprintf(stderr, "[qasr] cuBLAS error %d at %s:%d: %s\n", (int)_s, __FILE__, __LINE__, #expr);         \
      abort();                                                                                               \
    }                                                                                                        \
  } while (0)

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Cfg {
  // audio encoder
  int n_mels = 128, d = 1024, enc_layers = 24, enc_heads = 16, ffn = 4096, conv_ch = 480, n_window = 50, n_window_infer = 800, out_dim = 2048;
  // decoder
  int hidden = 2048, layers = 28, H = 16, KV = 8, hd = 128, inter = 6144, vocab = 151936;
  float eps = 1e-6f, theta = 1e6f;
  int audio_token = 151676;

  static Cfg load(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f) throw std::runtime_error("no config.json in " + dir);
    const nlohmann::json all = nlohmann::json::parse(f);
    const nlohmann::json& th = all.contains("thinker_config") ? all["thinker_config"] : all;
    if (!th.contains("audio_config") || !th.contains("text_config")) throw std::runtime_error("config.json is not a qwen3_asr config");
    const auto& a = th["audio_config"];
    const auto& t = th["text_config"];
    Cfg c;
    auto gi = [](const nlohmann::json& j, const char* k, int& v) { if (j.contains(k) && j[k].is_number()) v = j[k].get<int>(); };
    auto gf = [](const nlohmann::json& j, const char* k, float& v) { if (j.contains(k) && j[k].is_number()) v = j[k].get<float>(); };
    gi(a, "num_mel_bins", c.n_mels); gi(a, "d_model", c.d); gi(a, "encoder_layers", c.enc_layers); gi(a, "encoder_attention_heads", c.enc_heads);
    gi(a, "encoder_ffn_dim", c.ffn); gi(a, "downsample_hidden_size", c.conv_ch); gi(a, "n_window", c.n_window);
    gi(a, "n_window_infer", c.n_window_infer); gi(a, "output_dim", c.out_dim);
    gi(t, "hidden_size", c.hidden); gi(t, "num_hidden_layers", c.layers); gi(t, "num_attention_heads", c.H); gi(t, "num_key_value_heads", c.KV);
    gi(t, "head_dim", c.hd); gi(t, "intermediate_size", c.inter); gi(t, "vocab_size", c.vocab);
    gf(t, "rms_norm_eps", c.eps); gf(t, "rope_theta", c.theta);
    gi(th, "audio_token_id", c.audio_token);
    // Validation only: HIVE_QASR_DEBUG_LAYERS=N keeps the first N decoder layers (compare with qwen3_asr_oracle.py --layers N) so the whole
    //   pipeline can be checked on a GPU without room for the full model. Never set it for serving.
    if (const char* dbg = getenv("HIVE_QASR_DEBUG_LAYERS"); dbg && atoi(dbg) > 0 && atoi(dbg) < c.layers) {
      c.layers = atoi(dbg);
      fprintf(stderr, "[qasr] HIVE_QASR_DEBUG_LAYERS: only the first %d decoder layers are loaded (validation mode)\n", c.layers);
    }
    // The kernels are built for this geometry (Qwen3-ASR 1.7B; the 0.6B checkpoint differs in widths the GEMV templates cover too).
    auto need = [](bool ok, const std::string& m) { if (!ok) throw std::runtime_error("unsupported qwen3_asr geometry: " + m); };
    need(c.hd == 128, "decoder head_dim must be 128");
    need(c.H == 2 * c.KV, "decoder needs 2 query heads per kv head");
    need(c.hidden % 256 == 0 && (c.hidden == 1024 || c.hidden == 2048), "decoder width must be 1024 or 2048");
    need(c.inter == 6144 || c.inter == 3072, "decoder MLP width must be 3072 or 6144");
    need(c.H * c.hd == 2048, "decoder attention width must be 2048");
    need(c.d / c.enc_heads == 64, "encoder head dim must be 64");
    need(c.n_mels == 128 && c.conv_ch % 8 == 0, "encoder front (128 mel bins, conv channels multiple of 8)");
    need(c.out_dim == c.hidden, "encoder output width must equal the decoder width");
    need(c.n_window_infer % (2 * c.n_window) == 0 && 2 * c.n_window == 100, "encoder chunking (100-frame chunks)");
    return c;
  }
};

int conv_len(int n) {
  for (int i = 0; i < 3; ++i) n = (n - 1) / 2 + 1;
  return n;
}

// ---- cuBLAS: C[M,N] (row-major, fp32 or bf16) = A[M,K] · W[N,K]^T (+ beta·C), bf16 inputs, fp32 accumulation ----
struct Gemm {
  cublasHandle_t h = nullptr;
  DevBuf ws;
  void init(cudaStream_t st, size_t ws_bytes) {
    QASR_CUBLAS(cublasCreate(&h));
    QASR_CUBLAS(cublasSetStream(h, st));
    ws.alloc(ws_bytes);
    QASR_CUBLAS(cublasSetWorkspace(h, ws.p, ws_bytes));
  }
  ~Gemm() { if (h) cublasDestroy(h); }
  // Row count actually handed to cuBLAS: M rounded up to a bucket (16, 32, 64, then multiples of 64, at most `cap` = the rows the A and
  //   C buffers hold). cuBLAS picks its kernel by shape and its kernels are loaded lazily on first use — measured 2026-10-18: a first-seen
  //   M stalled a request by ~80 ms. With buckets the set of shapes is finite and the start-up warm-up runs every one of them. The extra
  //   rows read stale buffer contents and write rows nobody reads (every GEMM row is independent).
  static int bucket(int M, int cap) {
    if (cap <= 0) return M;
    int b = 16;
    while (b < M && b < 64) b <<= 1;
    if (M > 64) b = (M + 63) / 64 * 64;
    return std::max(M, std::min(b, cap));
  }
  void run(const bf16* A, int lda, const bf16* W, int ldw, float* C, int ldc, int M, int N, int K, float beta, int cap) {
    if (M <= 0) return;
    const int Mb = bucket(M, cap);
    const float alpha = 1.f;
    QASR_CUBLAS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, Mb, K, &alpha, W, CUDA_R_16BF, ldw, A, CUDA_R_16BF, lda, &beta, C, CUDA_R_32F, ldc,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
};

// ---- device arena: one allocation, 256-byte aligned pieces ----
struct Arena {
  size_t used = 0;
  DevBuf buf;
  size_t take(size_t bytes) { const size_t o = used; used += (bytes + 255) / 256 * 256; return o; }
  template <class T> T* at(size_t off) const { return reinterpret_cast<T*>(static_cast<char*>(buf.p) + off); }
};

struct EncLayer { size_t wqkv, bqkv, wo, bo, w1, b1, w2, b2, ln1w, ln1b, ln2w, ln2b; };
struct DecLayer { size_t wqkv, wo, wgu, wd, ln1, ln2, qn, kn; };

}  // namespace

enum class Kind { kTranscribe, kEncode, kLogits };

struct Job {
  Kind kind = Kind::kTranscribe;
  const float* mel = nullptr;
  int n_frames = 0;
  std::vector<int32_t> ids;  // full prompt (audio positions hold the audio token id)
  int a0 = 0, na = 0;        // audio rows [a0, a0+na)
  float temperature = 0, top_p = 1;
  int top_k = 0, max_new = 0;
  std::vector<int32_t> stop;
  const volatile int32_t* cancel = nullptr;
  std::mt19937_64 rng;
  // results
  std::vector<int32_t> out;
  float* fout = nullptr;  // debug outputs
  qasr_stats st{};
  std::string err;
  // state
  int slot = -1, n_ctx = 0, next_tok = 0;
  double t_submit = 0, t_admit = 0, t_dec0 = 0;
  bool done = false;
  std::condition_variable cv;
};

struct Engine {
  Cfg c;
  qasr_options opt{};
  int dev = 0;
  cudaStream_t st = nullptr;
  Gemm gemm;
  Arena wbf, wf32, work, persist;
  size_t emb = 0, lm = 0, final_norm = 0;
  size_t c1w, c1b, c2w, c2b, c3w, c3b, cow, pos_emb, lnpw, lnpb, p1w, p1b, p2w, p2b;
  std::vector<EncLayer> el;
  std::vector<DecLayer> dl;
  // work (phase union) offsets
  size_t x, xn, qkv, q, att, gu, act;                                 // decoder
  size_t eh, ey, ebig, eqkv, eao, emid, c1, col, cf, c2, c3;          // encoder
  // persistent
  size_t audio, mel, logits, rcos, rsin, pf_io, dec_io, dec_out, kc, vc;
  long slot_stride = 0;
  size_t layer_kv = 0;  // elements per layer of one of K/V
  int P = 512, EW = 8, CG = 4, EA = 0, max_frames = 0;
  int T1max = 50, T2max = 25, T3max = 13;
  // pinned host
  int32_t* h_pf_io = nullptr;   // [3 * P]
  int32_t* h_dec_io = nullptr;  // [3 * 4]
  int32_t* h_dec_out = nullptr; // [4]
  float* h_logits = nullptr;    // [max_seqs, vocab]
  std::vector<cudaGraphExec_t> graphs;  // index m-1
  qasr_info info{};
  // scheduling
  std::mutex mu;
  std::condition_variable cv_work;
  std::deque<Job*> pending;
  std::vector<Job*> active;  // by slot
  bool stop = false;
  std::thread worker;

  ~Engine() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv_work.notify_all();
    if (worker.joinable()) worker.join();
    for (auto g : graphs) if (g) cudaGraphExecDestroy(g);
    if (h_pf_io) cudaFreeHost(h_pf_io);
    if (h_dec_io) cudaFreeHost(h_dec_io);
    if (h_dec_out) cudaFreeHost(h_dec_out);
    if (h_logits) cudaFreeHost(h_logits);
    if (st) cudaStreamDestroy(st);
  }

  template <class T> T* W(size_t o) const { return wbf.at<T>(o); }
  float* F(size_t o) const { return wf32.at<float>(o); }
  template <class T> T* Wk(size_t o) const { return work.at<T>(o); }
  template <class T> T* Ps(size_t o) const { return persist.at<T>(o); }
  bf16* kcache(int l) const { return Ps<bf16>(kc) + (size_t)l * layer_kv; }
  bf16* vcache(int l) const { return Ps<bf16>(vc) + (size_t)l * layer_kv; }

  // ---------------------------------------------------------------- loading
  void load_weights(const std::string& dir) {
    Checkpoint ck(dir);
    const std::string pre = "thinker.";
    struct Up { size_t off; bool f32; size_t bytes; std::function<void(void*)> fill; };
    std::vector<Up> ups;
    auto tensor = [&](const std::string& n) -> const TensorInfo& {
      if (!ck.has(pre + n)) throw std::runtime_error("checkpoint is missing " + pre + n);
      return ck.get(pre + n);
    };
    auto read_f32 = [&](const TensorInfo& t, float* dst) {
      const int64_t n = t.numel();
      if (t.dtype == "BF16") {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(t.data);
        for (int64_t i = 0; i < n; ++i) { const uint32_t w = (uint32_t)s[i] << 16; std::memcpy(dst + i, &w, 4); }
      } else if (t.dtype == "F32") {
        std::memcpy(dst, t.data, n * 4);
      } else if (t.dtype == "F16") {
        const __half* s = reinterpret_cast<const __half*>(t.data);
        for (int64_t i = 0; i < n; ++i) dst[i] = __half2float(s[i]);
      } else {
        throw std::runtime_error("unsupported dtype " + t.dtype + " for " + t.name);
      }
    };
    auto read_bf16 = [&](const TensorInfo& t, uint16_t* dst) {
      if (t.dtype == "BF16") { std::memcpy(dst, t.data, t.numel() * 2); return; }
      std::vector<float> f(t.numel());
      read_f32(t, f.data());
      for (size_t i = 0; i < f.size(); ++i) {  // round to nearest even
        uint32_t u; std::memcpy(&u, &f[i], 4);
        dst[i] = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
      }
    };
    // bf16 matrix made of row blocks (concatenation along the output dim)
    auto mat = [&](std::vector<std::string> names) -> size_t {
      size_t rows = 0, cols = 0;
      for (auto& n : names) {
        const auto& t = tensor(n);
        if (t.shape.size() != 2) throw std::runtime_error(n + " is not a matrix");
        if (cols && (size_t)t.shape[1] != cols) throw std::runtime_error(n + ": column mismatch");
        cols = t.shape[1]; rows += t.shape[0];
      }
      const size_t bytes = rows * cols * 2, off = wbf.take(bytes);
      ups.push_back({off, false, bytes, [&, names, cols](void* dst) {
        uint16_t* d = static_cast<uint16_t*>(dst);
        for (auto& n : names) { const auto& t = tensor(n); read_bf16(t, d); d += t.shape[0] * cols; }
      }});
      return off;
    };
    auto vec = [&](std::vector<std::string> names) -> size_t {  // fp32 vector(s), concatenated
      size_t n = 0;
      for (auto& s : names) n += tensor(s).numel();
      const size_t off = wf32.take(n * 4);
      ups.push_back({off, true, n * 4, [&, names](void* dst) {
        float* d = static_cast<float*>(dst);
        for (auto& s : names) { const auto& t = tensor(s); read_f32(t, d); d += t.numel(); }
      }});
      return off;
    };
    const std::string A = "audio_tower.";
    // conv1 [C,1,3,3] -> fp32 [C][9]
    c1w = vec({A + "conv2d1.weight"}); c1b = vec({A + "conv2d1.bias"});
    // conv2/3 [C,C,3,3] -> bf16 [C][(kh*3+kw)*C + i] (taps outer, channels inner — the im2col column order)
    auto conv_w = [&](const std::string& n) -> size_t {
      const auto& t = tensor(n);
      const int C = (int)t.shape[0], Ci = (int)t.shape[1];
      const size_t bytes = (size_t)C * Ci * 9 * 2, off = wbf.take(bytes);
      ups.push_back({off, false, bytes, [&, n, C, Ci](void* dst) {
        const auto& tt = tensor(n);
        std::vector<uint16_t> raw(tt.numel());
        read_bf16(tt, raw.data());
        uint16_t* d = static_cast<uint16_t*>(dst);
        for (int o = 0; o < C; ++o)
          for (int i = 0; i < Ci; ++i)
            for (int k = 0; k < 9; ++k) d[(size_t)o * 9 * Ci + k * Ci + i] = raw[((size_t)o * Ci + i) * 9 + k];
      }});
      return off;
    };
    c2w = conv_w(A + "conv2d2.weight"); c2b = vec({A + "conv2d2.bias"});
    c3w = conv_w(A + "conv2d3.weight"); c3b = vec({A + "conv2d3.bias"});
    {  // conv_out [d, C*F] with column c*F + f  ->  column f*C + c (the engine's frame layout is [f][c])
      const auto& t = tensor(A + "conv_out.weight");
      const int dd = (int)t.shape[0], C = c.conv_ch, Fq = (int)t.shape[1] / C;
      if ((int)t.shape[1] != C * Fq || Fq != 16) throw std::runtime_error("conv_out shape");
      const size_t bytes = (size_t)dd * C * Fq * 2;
      cow = wbf.take(bytes);
      ups.push_back({cow, false, bytes, [&, dd, C, Fq, A](void* dst) {
        const auto& tt = tensor(A + "conv_out.weight");
        std::vector<uint16_t> raw(tt.numel());
        read_bf16(tt, raw.data());
        uint16_t* d = static_cast<uint16_t*>(dst);
        for (int o = 0; o < dd; ++o)
          for (int ch = 0; ch < C; ++ch)
            for (int f = 0; f < Fq; ++f) d[(size_t)o * C * Fq + f * C + ch] = raw[(size_t)o * C * Fq + ch * Fq + f];
      }});
    }
    {  // sinusoidal positions [T3max, d]: [sin(t * w_i), cos(t * w_i)], w_i = 10000^(-i/(d/2-1))
      const int half = c.d / 2;
      pos_emb = wf32.take((size_t)16 * c.d * 4);
      ups.push_back({pos_emb, true, (size_t)16 * c.d * 4, [this, half](void* dst) {
        float* d = static_cast<float*>(dst);
        const double inc = std::log(10000.0) / (half - 1);
        for (int t = 0; t < 16; ++t)
          for (int i = 0; i < half; ++i) {
            // the reference builds the table in fp32: inv = exp(-inc * i) (fp32), angle = t * inv (fp32)
            const float inv = (float)std::exp(-(double)(float)inc * i);
            const float ang = (float)t * inv;
            d[(size_t)t * c.d + i] = (float)std::sin((double)ang);
            d[(size_t)t * c.d + half + i] = (float)std::cos((double)ang);
          }
      }});
    }
    for (int l = 0; l < c.enc_layers; ++l) {
      const std::string p = A + "layers." + std::to_string(l) + ".";
      EncLayer e;
      e.wqkv = mat({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
      e.bqkv = vec({p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias", p + "self_attn.v_proj.bias"});
      e.wo = mat({p + "self_attn.out_proj.weight"}); e.bo = vec({p + "self_attn.out_proj.bias"});
      e.w1 = mat({p + "fc1.weight"}); e.b1 = vec({p + "fc1.bias"});
      e.w2 = mat({p + "fc2.weight"}); e.b2 = vec({p + "fc2.bias"});
      e.ln1w = vec({p + "self_attn_layer_norm.weight"}); e.ln1b = vec({p + "self_attn_layer_norm.bias"});
      e.ln2w = vec({p + "final_layer_norm.weight"}); e.ln2b = vec({p + "final_layer_norm.bias"});
      el.push_back(e);
    }
    lnpw = vec({A + "ln_post.weight"}); lnpb = vec({A + "ln_post.bias"});
    p1w = mat({A + "proj1.weight"}); p1b = vec({A + "proj1.bias"});
    p2w = mat({A + "proj2.weight"}); p2b = vec({A + "proj2.bias"});
    for (int l = 0; l < c.layers; ++l) {
      const std::string p = "model.layers." + std::to_string(l) + ".";
      DecLayer d;
      d.wqkv = mat({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
      d.wo = mat({p + "self_attn.o_proj.weight"});
      d.wgu = mat({p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"});
      d.wd = mat({p + "mlp.down_proj.weight"});
      d.ln1 = vec({p + "input_layernorm.weight"}); d.ln2 = vec({p + "post_attention_layernorm.weight"});
      d.qn = vec({p + "self_attn.q_norm.weight"}); d.kn = vec({p + "self_attn.k_norm.weight"});
      dl.push_back(d);
    }
    final_norm = vec({"model.norm.weight"});
    emb = mat({"model.embed_tokens.weight"});
    bool tied = !ck.has(pre + "lm_head.weight");
    if (!tied) {  // stored separately; share the table when the bytes are identical (they are in the released checkpoints)
      const auto& a = tensor("lm_head.weight");
      const auto& b = tensor("model.embed_tokens.weight");
      tied = a.dtype == b.dtype && a.shape == b.shape && std::memcmp(a.data, b.data, a.nbytes) == 0;
    }
    lm = tied ? emb : mat({"lm_head.weight"});
    if ((int)tensor("model.embed_tokens.weight").shape[0] != c.vocab) throw std::runtime_error("embedding rows != vocab_size");
    info.weights_mib = (wbf.used + wf32.used) / kMiB;
    // the plan is checked before the weights go up (create() calls check_plan with these sizes), then upload
    plan_check_hook();
    wbf.buf.alloc(wbf.used);
    wf32.buf.alloc(wf32.used);
    std::vector<char> stage;
    for (auto& u : ups) {
      stage.resize(u.bytes);
      u.fill(stage.data());
      CUDA_CHECK(cudaMemcpy((u.f32 ? wf32 : wbf).at<char>(u.off), stage.data(), u.bytes, cudaMemcpyHostToDevice));
    }
    ck.release_all();
  }

  std::function<void()> plan_check_hook = [] {};

  // ---------------------------------------------------------------- buffers
  size_t plan_work(Arena& w, Arena& p) {
    const int h = c.hidden, nq = c.H * c.hd, nkv = c.KV * c.hd;
    // decoder phase
    Arena d;
    x = d.take((size_t)P * h * 4); xn = d.take((size_t)P * h * 2); qkv = d.take((size_t)P * (nq + 2 * nkv) * 4);
    q = d.take((size_t)P * nq * 2); att = d.take((size_t)P * nq * 2); gu = d.take((size_t)P * 2 * c.inter * 4);
    act = d.take((size_t)P * c.inter * 2);
    // encoder phase
    Arena e;
    EA = EW * (c.n_window_infer / (2 * c.n_window)) * T3max;
    const int dd = c.d, C = c.conv_ch;
    const int F1 = (c.n_mels + 1) / 2, F2 = (F1 + 1) / 2, F3 = (F2 + 1) / 2;
    eh = e.take((size_t)EA * dd * 4); ey = e.take((size_t)EA * dd * 2); ebig = e.take((size_t)EA * std::max(3 * dd, c.ffn) * 4);
    eqkv = e.take((size_t)EA * 3 * dd * 2); eao = e.take((size_t)EA * dd * 2); emid = e.take((size_t)EA * c.ffn * 2);
    c1 = e.take((size_t)CG * F1 * T1max * C * 2);
    col = e.take((size_t)CG * std::max(F2 * T2max, F3 * T3max) * 9 * C * 2);
    cf = e.take((size_t)CG * std::max((size_t)F2 * T2max * C, (size_t)T3max * dd) * 4);
    c2 = e.take((size_t)CG * F2 * T2max * C * 2); c3 = e.take((size_t)CG * T3max * F3 * C * 2);
    w.used = std::max(d.used, e.used);
    // persistent
    const int mc = opt.max_ctx, ms = opt.max_seqs;
    max_frames = (mc / T3max + 1) * 100;
    audio = p.take((size_t)mc * h * 4);
    mel = p.take((size_t)c.n_mels * max_frames * 4);
    logits = p.take((size_t)ms * c.vocab * 4);
    rcos = p.take((size_t)mc * (c.hd / 2) * 4); rsin = p.take((size_t)mc * (c.hd / 2) * 4);
    pf_io = p.take((size_t)3 * P * 4); dec_io = p.take(3 * 4 * 4); dec_out = p.take(4 * 4);
    layer_kv = (size_t)ms * mc * nkv;
    slot_stride = (long)mc * nkv;
    const size_t kv_bytes = (size_t)c.layers * layer_kv * 2;
    kc = p.take(kv_bytes); vc = p.take(kv_bytes);
    info.kv_mib = 2.0 * kv_bytes / kMiB;
    info.work_mib = (w.used + p.used) / kMiB - info.kv_mib;
    return w.used + p.used;
  }

  // rows the conv GEMM buffers hold (CG chunks of the largest padded chunk)
  int F2() const { return ((c.n_mels + 1) / 2 + 1) / 2; }
  int F3() const { return (F2() + 1) / 2; }
  int cap_conv2() const { return CG * F2() * T2max; }
  int cap_conv3() const { return CG * T3max * F3(); }
  int cap_convo() const { return CG * T3max; }

  // Every GEMM shape the request paths can produce, once (see Gemm::bucket). Inputs are whatever the buffers hold; outputs are scratch.
  void warm_gemms() {
    const int dd = c.d, C = c.conv_ch, h = c.hidden, nq = c.H * c.hd, nkv = c.KV * c.hd;
    auto buckets = [](std::vector<int> ms, int cap) {
      std::vector<int> b;
      for (int m : ms) if (m >= 1 && m <= cap) b.push_back(Gemm::bucket(m, cap));
      std::sort(b.begin(), b.end());
      b.erase(std::unique(b.begin(), b.end()), b.end());
      return b;
    };
    auto upto = [](int cap) { std::vector<int> v(cap); std::iota(v.begin(), v.end(), 1); return v; };
    std::vector<int> m2, m3, mo;  // reachable conv row counts: groups of nc full chunks, or one short chunk of lp frames
    for (int nc = 1; nc <= CG; ++nc) { m2.push_back(nc * F2() * T2max); m3.push_back(nc * T3max * F3()); mo.push_back(nc * T3max); }
    for (int lp = 1; lp < 100; ++lp) {
      const int T1 = (lp + 1) / 2, T2 = (T1 + 1) / 2, T3 = conv_len(lp);
      m2.push_back(F2() * T2); m3.push_back(T3 * F3()); mo.push_back(T3);
    }
    for (int b : buckets(m2, cap_conv2())) gemm.run(Wk<bf16>(col), 9 * C, W<bf16>(c2w), 9 * C, Wk<float>(cf), C, b, C, 9 * C, 0.f, cap_conv2());
    for (int b : buckets(m3, cap_conv3())) gemm.run(Wk<bf16>(col), 9 * C, W<bf16>(c3w), 9 * C, Wk<float>(cf), C, b, C, 9 * C, 0.f, cap_conv3());
    for (int b : buckets(mo, cap_convo())) gemm.run(Wk<bf16>(c3), F3() * C, W<bf16>(cow), F3() * C, Wk<float>(cf), dd, b, dd, F3() * C, 0.f, cap_convo());
    const EncLayer& e = el[0];
    for (int b : buckets(upto(EA), EA)) {
      gemm.run(Wk<bf16>(ey), dd, W<bf16>(e.wqkv), dd, Wk<float>(ebig), 3 * dd, b, 3 * dd, dd, 0.f, EA);
      gemm.run(Wk<bf16>(eao), dd, W<bf16>(e.wo), dd, Wk<float>(ebig), dd, b, dd, dd, 0.f, EA);
      gemm.run(Wk<bf16>(ey), dd, W<bf16>(e.w1), dd, Wk<float>(ebig), c.ffn, b, c.ffn, dd, 0.f, EA);
      gemm.run(Wk<bf16>(emid), c.ffn, W<bf16>(e.w2), c.ffn, Wk<float>(ebig), dd, b, dd, c.ffn, 0.f, EA);
      gemm.run(Wk<bf16>(eao), dd, W<bf16>(p2w), dd, Wk<float>(ebig), c.hidden, b, c.hidden, dd, 0.f, EA);
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    const DecLayer& L = dl[0];
    for (int b : buckets(upto(P), P)) {
      gemm.run(Wk<bf16>(xn), h, W<bf16>(L.wqkv), h, Wk<float>(qkv), nq + 2 * nkv, b, nq + 2 * nkv, h, 0.f, P);
      gemm.run(Wk<bf16>(att), nq, W<bf16>(L.wo), nq, Wk<float>(x), h, b, h, nq, 1.f, P);
      gemm.run(Wk<bf16>(xn), h, W<bf16>(L.wgu), h, Wk<float>(gu), 2 * c.inter, b, 2 * c.inter, h, 0.f, P);
      gemm.run(Wk<bf16>(act), c.inter, W<bf16>(L.wd), c.inter, Wk<float>(x), h, b, h, c.inter, 1.f, P);
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
  }

  void init_rope() {
    const int half = c.hd / 2, mc = opt.max_ctx;
    std::vector<float> cs((size_t)mc * half), sn((size_t)mc * half);
    for (int i = 0; i < half; ++i) {
      // inv_freq_i = theta^(-2i/hd) rounded to fp32, angle = pos * inv_freq in fp32 (the reference's fp32 table), cos/sin accurate
      const float inv = (float)(1.0 / std::pow((double)c.theta, (double)(2 * i) / c.hd));
      for (int p = 0; p < mc; ++p) {
        const float ang = (float)p * inv;
        cs[(size_t)p * half + i] = (float)std::cos((double)ang);
        sn[(size_t)p * half + i] = (float)std::sin((double)ang);
      }
    }
    CUDA_CHECK(cudaMemcpy(Ps<float>(rcos), cs.data(), cs.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Ps<float>(rsin), sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
  }

  // ---------------------------------------------------------------- audio encoder
  // Returns the number of audio tokens written to `audio` (fp32 [Ta, hidden]).
  int encode(const float* mel_host, int nfr) {
    const int C = c.conv_ch, dd = c.d;
    const int nch = (nfr + 99) / 100, lp = nch > 1 ? 100 : nfr;
    const int T1 = (lp + 1) / 2, T2 = (T1 + 1) / 2, T3 = conv_len(lp);
    const int last_valid = conv_len(nfr - (nch - 1) * 100);
    const int Ta = (nch - 1) * T3 + last_valid;
    const int cpw = c.n_window_infer / (2 * c.n_window), win = T3 * cpw;
    const int F1 = (c.n_mels + 1) / 2, F2 = (F1 + 1) / 2, F3 = (F2 + 1) / 2;
    CUDA_CHECK(cudaMemcpyAsync(Ps<float>(mel), mel_host, (size_t)c.n_mels * nfr * 4, cudaMemcpyHostToDevice, st));
    float* h = Wk<float>(eh);
    for (int cb = 0; cb < nch; cb += EW * cpw) {
      const int ce = std::min(nch, cb + EW * cpw);
      const int tok0 = cb * T3, ntok = (ce == nch ? Ta : ce * T3) - tok0;
      for (int g = cb; g < ce; g += CG) {
        const int nc = std::min(ce, g + CG) - g;
        conv1_gelu(Ps<float>(mel), c.n_mels, nfr, g, nc, lp, F(c1w), F(c1b), C, Wk<bf16>(c1), st);
        im2col_s2(Wk<bf16>(c1), nc, F1, T1, C, false, Wk<bf16>(col), st);
        gemm.run(Wk<bf16>(col), 9 * C, W<bf16>(c2w), 9 * C, Wk<float>(cf), C, nc * F2 * T2, C, 9 * C, 0.f, cap_conv2());
        bias_epilogue(Wk<float>(cf), F(c2b), nc * F2 * T2, C, Epi::kGeluBf16, Wk<bf16>(c2), nullptr, st);
        im2col_s2(Wk<bf16>(c2), nc, F2, T2, C, true, Wk<bf16>(col), st);
        gemm.run(Wk<bf16>(col), 9 * C, W<bf16>(c3w), 9 * C, Wk<float>(cf), C, nc * T3 * F3, C, 9 * C, 0.f, cap_conv3());
        bias_epilogue(Wk<float>(cf), F(c3b), nc * T3 * F3, C, Epi::kGeluBf16, Wk<bf16>(c3), nullptr, st);
        gemm.run(Wk<bf16>(c3), F3 * C, W<bf16>(cow), F3 * C, Wk<float>(cf), dd, nc * T3, dd, F3 * C, 0.f, cap_convo());
        const int lv = (g + nc == nch) ? last_valid : T3;
        add_pos_gather(Wk<float>(cf), F(pos_emb), nc, T3, lv, dd, h + (size_t)(g - cb) * T3 * dd, st);
      }
      for (const EncLayer& e : el) {
        layernorm(h, F(e.ln1w), F(e.ln1b), Wk<bf16>(ey), ntok, dd, 1e-5f, st);
        gemm.run(Wk<bf16>(ey), dd, W<bf16>(e.wqkv), dd, Wk<float>(ebig), 3 * dd, ntok, 3 * dd, dd, 0.f, EA);
        bias_epilogue(Wk<float>(ebig), F(e.bqkv), ntok, 3 * dd, Epi::kBf16, Wk<bf16>(eqkv), nullptr, st);
        attn_encoder(Wk<bf16>(eqkv), ntok, c.enc_heads, win, Wk<bf16>(eao), st);
        gemm.run(Wk<bf16>(eao), dd, W<bf16>(e.wo), dd, Wk<float>(ebig), dd, ntok, dd, dd, 0.f, EA);
        bias_epilogue(Wk<float>(ebig), F(e.bo), ntok, dd, Epi::kAddF32, nullptr, h, st);
        layernorm(h, F(e.ln2w), F(e.ln2b), Wk<bf16>(ey), ntok, dd, 1e-5f, st);
        gemm.run(Wk<bf16>(ey), dd, W<bf16>(e.w1), dd, Wk<float>(ebig), c.ffn, ntok, c.ffn, dd, 0.f, EA);
        bias_epilogue(Wk<float>(ebig), F(e.b1), ntok, c.ffn, Epi::kGeluBf16, Wk<bf16>(emid), nullptr, st);
        gemm.run(Wk<bf16>(emid), c.ffn, W<bf16>(e.w2), c.ffn, Wk<float>(ebig), dd, ntok, dd, c.ffn, 0.f, EA);
        bias_epilogue(Wk<float>(ebig), F(e.b2), ntok, dd, Epi::kAddF32, nullptr, h, st);
      }
      layernorm(h, F(lnpw), F(lnpb), Wk<bf16>(ey), ntok, dd, 1e-5f, st);
      gemm.run(Wk<bf16>(ey), dd, W<bf16>(p1w), dd, Wk<float>(ebig), dd, ntok, dd, dd, 0.f, EA);
      bias_epilogue(Wk<float>(ebig), F(p1b), ntok, dd, Epi::kGeluBf16, Wk<bf16>(eao), nullptr, st);
      float* dst = Ps<float>(audio) + (size_t)tok0 * c.hidden;
      gemm.run(Wk<bf16>(eao), dd, W<bf16>(p2w), dd, Wk<float>(ebig), c.hidden, ntok, c.hidden, dd, 0.f, EA);
      CUDA_CHECK(cudaMemsetAsync(dst, 0, (size_t)ntok * c.hidden * 4, st));
      bias_epilogue(Wk<float>(ebig), F(p2b), ntok, c.hidden, Epi::kAddF32, nullptr, dst, st);
    }
    return Ta;
  }

  // ---------------------------------------------------------------- prefill (one sequence) -> logits row `slot`, argmax in dec_out[0]
  void prefill(const Job& j, int slot) {
    const int n = (int)j.ids.size(), h = c.hidden, nq = c.H * c.hd, nkv = c.KV * c.hd;
    for (int s = 0; s < n; s += P) {
      const int e = std::min(n, s + P), T = e - s;
      for (int t = 0; t < T; ++t) { h_pf_io[t] = j.ids[s + t]; h_pf_io[P + t] = s + t; h_pf_io[2 * P + t] = slot; }
      int32_t* io = Ps<int32_t>(pf_io);
      CUDA_CHECK(cudaMemcpyAsync(io, h_pf_io, (size_t)3 * P * 4, cudaMemcpyHostToDevice, st));
      float* X = Wk<float>(x);
      embed_rows(W<bf16>(emb), io, X, T, h, st);
      const int lo = std::max(s, j.a0), hi = std::min(e, j.a0 + j.na);
      if (hi > lo)
        CUDA_CHECK(cudaMemcpyAsync(X + (size_t)(lo - s) * h, Ps<float>(audio) + (size_t)(lo - j.a0) * h, (size_t)(hi - lo) * h * 4,
                                   cudaMemcpyDeviceToDevice, st));
      for (int l = 0; l < c.layers; ++l) {
        const DecLayer& L = dl[l];
        rmsnorm(X, F(L.ln1), Wk<bf16>(xn), T, h, c.eps, st);
        gemm.run(Wk<bf16>(xn), h, W<bf16>(L.wqkv), h, Wk<float>(qkv), nq + 2 * nkv, T, nq + 2 * nkv, h, 0.f, P);
        qk_norm_rope_store(Wk<float>(qkv), F(L.qn), F(L.kn), Ps<float>(rcos), Ps<float>(rsin), io + P, io + 2 * P, T, c.H, c.KV, c.eps,
                           Wk<bf16>(q), kcache(l), vcache(l), slot_stride, st);
        attn_prefill_causal(Wk<bf16>(q), kcache(l) + (size_t)slot * slot_stride, vcache(l) + (size_t)slot * slot_stride, T, s, c.H, c.KV,
                            Wk<bf16>(att), st);
        gemm.run(Wk<bf16>(att), nq, W<bf16>(L.wo), nq, X, h, T, h, nq, 1.f, P);
        rmsnorm(X, F(L.ln2), Wk<bf16>(xn), T, h, c.eps, st);
        gemm.run(Wk<bf16>(xn), h, W<bf16>(L.wgu), h, Wk<float>(gu), 2 * c.inter, T, 2 * c.inter, h, 0.f, P);
        swiglu(Wk<float>(gu), Wk<bf16>(act), T, c.inter, st);
        gemm.run(Wk<bf16>(act), c.inter, W<bf16>(L.wd), c.inter, X, h, T, h, c.inter, 1.f, P);
      }
      if (e == n) {
        rmsnorm(X + (size_t)(T - 1) * h, F(final_norm), Wk<bf16>(xn), 1, h, c.eps, st);
        gemv(W<bf16>(lm), Wk<bf16>(xn), 1, c.vocab, h, Ps<float>(logits) + (size_t)slot * c.vocab, st);
        argmax_rows(Ps<float>(logits) + (size_t)slot * c.vocab, 1, c.vocab, Ps<int32_t>(dec_out), st);
      }
    }
  }

  // ---------------------------------------------------------------- decode step body (captured per batch size m)
  void decode_body(int m) {
    const int h = c.hidden, nq = c.H * c.hd, nkv = c.KV * c.hd;
    int32_t* io = Ps<int32_t>(dec_io);  // tok[4], pos[4], slot[4]
    float* X = Wk<float>(x);
    embed_rows(W<bf16>(emb), io, X, m, h, st);
    for (int l = 0; l < c.layers; ++l) {
      const DecLayer& L = dl[l];
      rmsnorm(X, F(L.ln1), Wk<bf16>(xn), m, h, c.eps, st);
      gemv(W<bf16>(L.wqkv), Wk<bf16>(xn), m, nq + 2 * nkv, h, Wk<float>(qkv), st);
      qk_norm_rope_store(Wk<float>(qkv), F(L.qn), F(L.kn), Ps<float>(rcos), Ps<float>(rsin), io + 4, io + 8, m, c.H, c.KV, c.eps, Wk<bf16>(q),
                         kcache(l), vcache(l), slot_stride, st);
      attn_decode(Wk<bf16>(q), kcache(l), vcache(l), io + 4, io + 8, m, c.H, c.KV, slot_stride, Wk<bf16>(att), st);
      gemv_add(W<bf16>(L.wo), Wk<bf16>(att), m, h, nq, X, st);
      rmsnorm(X, F(L.ln2), Wk<bf16>(xn), m, h, c.eps, st);
      gemv_swiglu(W<bf16>(L.wgu), Wk<bf16>(xn), m, c.inter, h, Wk<bf16>(act), st);
      gemv_add(W<bf16>(L.wd), Wk<bf16>(act), m, h, c.inter, X, st);
    }
    rmsnorm(X, F(final_norm), Wk<bf16>(xn), m, h, c.eps, st);
    gemv(W<bf16>(lm), Wk<bf16>(xn), m, c.vocab, h, Ps<float>(logits), st);
    argmax_rows(Ps<float>(logits), m, c.vocab, Ps<int32_t>(dec_out), st);
  }

  void capture_graphs() {
    graphs.assign(opt.max_seqs, nullptr);
    for (int m = 1; m <= opt.max_seqs; ++m) {
      cudaGraph_t g;
      CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
      decode_body(m);
      CUDA_CHECK(cudaStreamEndCapture(st, &g));
      CUDA_CHECK(cudaGraphInstantiate(&graphs[m - 1], g, 0));
      CUDA_CHECK(cudaGraphDestroy(g));
    }
  }

  // ---------------------------------------------------------------- sampling
  int sample(Job& j, const float* lg) {
    const int V = c.vocab;
    float mx = -INFINITY;
    for (int i = 0; i < V; ++i) mx = std::max(mx, lg[i]);
    const float inv_t = 1.f / j.temperature;
    std::vector<std::pair<float, int>> cand;
    cand.reserve(V);
    for (int i = 0; i < V; ++i) cand.emplace_back(std::exp((lg[i] - mx) * inv_t), i);
    auto desc = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
    if (j.top_k > 0 && j.top_k < V) {
      std::partial_sort(cand.begin(), cand.begin() + j.top_k, cand.end(), desc);
      cand.resize(j.top_k);
    } else if (j.top_p < 1.f) {
      std::sort(cand.begin(), cand.end(), desc);
    }
    if (j.top_p < 1.f) {
      if (j.top_k > 0) std::sort(cand.begin(), cand.end(), desc);
      double tot = 0;
      for (auto& p : cand) tot += p.first;
      double cum = 0;
      size_t keep = 0;
      while (keep < cand.size()) { cum += cand[keep].first; ++keep; if (cum >= j.top_p * tot) break; }
      cand.resize(std::max<size_t>(1, keep));
    }
    double tot = 0;
    for (auto& p : cand) tot += p.first;
    double r = std::uniform_real_distribution<double>(0.0, tot)(j.rng), acc = 0;
    for (auto& p : cand) { acc += p.first; if (r < acc) return p.second; }
    return cand.back().second;
  }

  // ---------------------------------------------------------------- scheduling
  void finish(Job* j, int reason) {
    j->st.finish = reason;
    j->st.n_generated = (int)j->out.size();
    const double t = now_ms();
    j->st.decode_ms = j->t_dec0 > 0 ? t - j->t_dec0 : 0;
    j->st.total_ms = t - j->t_submit;
    std::lock_guard<std::mutex> lk(mu);
    if (j->slot >= 0) active[j->slot] = nullptr;
    j->done = true;
    j->cv.notify_all();
  }

  bool cancelled(const Job* j) const { return j->cancel && *j->cancel != 0; }

  // Takes the newly sampled token for job j; returns true when the job ended.
  bool accept_token(Job* j, int tok) {
    if (std::find(j->stop.begin(), j->stop.end(), tok) != j->stop.end()) { j->st.stop_token = tok; finish(j, 0); return true; }
    j->out.push_back(tok);
    j->next_tok = tok;
    if ((int)j->out.size() >= j->max_new || j->n_ctx >= opt.max_ctx) { finish(j, 1); return true; }
    return false;
  }

  void admit(Job* j, int slot) {
    j->t_admit = now_ms();
    j->st.queue_ms = j->t_admit - j->t_submit;
    if (cancelled(j)) { finish(j, 2); return; }
    if (j->na > 0) encode(j->mel, j->n_frames);
    if (j->kind == Kind::kEncode) {
      CUDA_CHECK(cudaMemcpyAsync(j->fout, Ps<float>(audio), (size_t)j->na * c.hidden * 4, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaStreamSynchronize(st));
      finish(j, 0);
      return;
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    const double t_enc = now_ms();
    j->st.encode_ms = t_enc - j->t_admit;
    {
      std::lock_guard<std::mutex> lk(mu);
      j->slot = slot;
      active[slot] = j;
    }
    prefill(*j, slot);
    j->n_ctx = (int)j->ids.size();
    if (j->kind == Kind::kLogits) {
      CUDA_CHECK(cudaMemcpyAsync(j->fout, Ps<float>(logits) + (size_t)slot * c.vocab, (size_t)c.vocab * 4, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaStreamSynchronize(st));
      finish(j, 0);
      return;
    }
    int tok;
    if (j->temperature > 0) {
      CUDA_CHECK(cudaMemcpyAsync(h_logits, Ps<float>(logits) + (size_t)slot * c.vocab, (size_t)c.vocab * 4, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaStreamSynchronize(st));
      tok = sample(*j, h_logits);
    } else {
      CUDA_CHECK(cudaMemcpyAsync(h_dec_out, Ps<int32_t>(dec_out), 4, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaStreamSynchronize(st));
      tok = h_dec_out[0];
    }
    const double t_pf = now_ms();
    j->st.prefill_ms = t_pf - t_enc;
    j->t_dec0 = t_pf;
    if (j->max_new <= 0) { finish(j, 1); return; }
    accept_token(j, tok);
  }

  void decode_step() {
    std::vector<Job*> b;
    for (int s = 0; s < opt.max_seqs; ++s) {
      Job* j = active[s];
      if (!j) continue;
      if (cancelled(j)) { finish(j, 2); continue; }
      b.push_back(j);
    }
    const int m = (int)b.size();
    if (m == 0) return;
    for (int i = 0; i < m; ++i) {
      h_dec_io[i] = b[i]->next_tok;
      h_dec_io[4 + i] = b[i]->n_ctx;  // position of the token being fed
      h_dec_io[8 + i] = b[i]->slot;
    }
    CUDA_CHECK(cudaMemcpyAsync(Ps<int32_t>(dec_io), h_dec_io, 12 * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaGraphLaunch(graphs[m - 1], st));
    CUDA_CHECK(cudaMemcpyAsync(h_dec_out, Ps<int32_t>(dec_out), 4 * 4, cudaMemcpyDeviceToHost, st));
    bool any_sample = false;
    for (auto* j : b) any_sample |= j->temperature > 0;
    if (any_sample)
      CUDA_CHECK(cudaMemcpyAsync(h_logits, Ps<float>(logits), (size_t)m * c.vocab * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    for (int i = 0; i < m; ++i) {
      Job* j = b[i];
      j->n_ctx += 1;
      const int tok = j->temperature > 0 ? sample(*j, h_logits + (size_t)i * c.vocab) : h_dec_out[i];
      accept_token(j, tok);
    }
  }

  void run() {
    CUDA_CHECK(cudaSetDevice(dev));
    for (;;) {
      Job* j = nullptr;
      int slot = -1;
      bool any_active = false;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv_work.wait(lk, [&] {
          if (stop || !pending.empty()) return true;
          for (auto* a : active) if (a) return true;
          return false;
        });
        if (stop) {
          for (auto* p : pending) { p->err = "engine shut down"; p->done = true; p->cv.notify_all(); }
          pending.clear();
          for (auto*& a : active) if (a) { a->err = "engine shut down"; a->done = true; a->cv.notify_all(); a = nullptr; }
          return;
        }
        for (int s = 0; s < opt.max_seqs; ++s) if (!active[s]) { slot = s; break; }
        if (!pending.empty() && slot >= 0) { j = pending.front(); pending.pop_front(); }
        for (auto* a : active) any_active |= a != nullptr;
      }
      if (j) { admit(j, slot); continue; }
      if (any_active) decode_step();
    }
  }

  void submit(Job* j) {
    j->t_submit = now_ms();
    std::unique_lock<std::mutex> lk(mu);
    pending.push_back(j);
    cv_work.notify_one();
    j->cv.wait(lk, [&] { return j->done; });
  }

  // ---------------------------------------------------------------- warm-up: every kernel and GEMM shape family once
  void warmup() {
    warm_gemms();
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 0.5f);
    for (int frames : {1730, 370, 95}) {  // several windows over two passes when EW is small, a padded tail chunk, a single short chunk
      if (qasr_audio_tokens(frames) + 40 > opt.max_ctx) continue;
      std::vector<float> m((size_t)c.n_mels * frames);
      for (auto& v : m) v = nd(rng);
      Job j;
      j.mel = m.data(); j.n_frames = frames; j.na = qasr_audio_tokens(frames);
      j.ids.assign(9, 151644); j.a0 = 9;
      j.ids.insert(j.ids.end(), j.na, c.audio_token);
      j.ids.insert(j.ids.end(), 12, 151644);
      j.max_new = 1;
      j.t_submit = now_ms();
      active.assign(opt.max_seqs, nullptr);
      admit(&j, 0);
    }
    // one decode step per batch size
    active.assign(opt.max_seqs, nullptr);
    for (int m = 1; m <= opt.max_seqs; ++m) {
      for (int i = 0; i < m; ++i) { h_dec_io[i] = 151644; h_dec_io[4 + i] = 50 + i; h_dec_io[8 + i] = i; }
      CUDA_CHECK(cudaMemcpyAsync(Ps<int32_t>(dec_io), h_dec_io, 12 * 4, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaGraphLaunch(graphs[m - 1], st));
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    active.assign(opt.max_seqs, nullptr);
  }
};

}  // namespace hive::qasr

using hive::qasr::Engine;
using hive::qasr::Job;
using hive::qasr::Kind;

struct qasr_engine {
  std::unique_ptr<Engine> e;
};

namespace {
void set_err(char* err, int len, const std::string& m) {
  if (err && len > 0) { snprintf(err, (size_t)len, "%s", m.c_str()); }
}
}  // namespace

extern "C" {

void qasr_default_options(qasr_options* o) {
  o->device = 0; o->max_seqs = 2; o->max_ctx = 2048; o->vram_limit_mb = 0; o->prefill_chunk = 512; o->encoder_windows = 8; o->warmup = 1;
}

int qasr_audio_tokens(int n_frames) {
  if (n_frames <= 0) return 0;
  const int full = n_frames / 100, r = n_frames % 100;
  return 13 * full + (r ? hive::qasr::conv_len(r) : 0);
}

qasr_engine* qasr_create(const char* ckpt_dir, const qasr_options* opt_in, char* err, int err_len) {
  qasr_options opt;
  qasr_default_options(&opt);
  if (opt_in) opt = *opt_in;
  try {
    if (opt.max_seqs < 1 || opt.max_seqs > 4) throw std::runtime_error("max_seqs must be 1..4");
    if (opt.max_ctx < 64) throw std::runtime_error("max_ctx must be >= 64");
    if (opt.prefill_chunk < 16) opt.prefill_chunk = 16;
    if (opt.encoder_windows < 1) opt.encoder_windows = 1;
    auto e = std::make_unique<Engine>();
    e->opt = opt;
    e->dev = opt.device;
    e->c = hive::qasr::Cfg::load(ckpt_dir);
    e->P = opt.prefill_chunk;
    e->EW = opt.encoder_windows;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, e->dev));
    if (prop.major != 12) throw std::runtime_error(std::string("this build runs on compute capability 12.x GPUs; found ") + prop.name);
    // CUDA context: NVML free before it exists - CUDA free right after (same measurement as the daemons, hive/vram_limit.h). The NVML
    //   reading comes before cudaSetDevice, which creates the primary context since CUDA 12.
    size_t limit = opt.vram_limit_mb > 0 ? (size_t)opt.vram_limit_mb << 20 : hive::vram_limit::limit_bytes();
    char pci[32];
    snprintf(pci, sizeof pci, "%08x:%02x:%02x.0", (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
    size_t f0 = 0, f1 = 0, tot = 0;
    const bool nvml = hive::vram_limit::nvml_free(pci, &f0);
    CUDA_CHECK(cudaSetDevice(e->dev));
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaMemGetInfo(&f1, &tot));
    const bool ctx_measured = nvml && f0 > f1 && f0 - f1 >= (64ull << 20);
    const size_t ctx = ctx_measured ? f0 - f1 : hive::vram_limit::kCtxFallback;
    e->info.context_mib = ctx / hive::qasr::kMiB;
    e->info.context_measured = ctx_measured;
    CUDA_CHECK(cudaStreamCreateWithFlags(&e->st, cudaStreamNonBlocking));
    Engine* E = e.get();
    const size_t cublas_ws = 32ull << 20;
    // The plan is checked once the weight sizes are known and before anything big is allocated.
    E->plan_check_hook = [E, limit, ctx, cublas_ws, f1]() {
      const size_t buffers = E->plan_work(E->work, E->persist);
      const double graphs_and_modules = 96.0 * hive::qasr::kMiB;  // CUDA graphs, lazily loaded kernels (warm-up), allocator rounding
      const double plan = ctx + E->wbf.used + E->wf32.used + buffers + cublas_ws + graphs_and_modules;
      E->info.plan_mib = plan / hive::qasr::kMiB;
      fprintf(stderr, "[qasr] VRAM plan %.0f MiB = context %.0f + weights %.0f + KV %.0f + work %.0f + cuBLAS %.0f + graphs/modules %.0f%s\n",
              plan / hive::qasr::kMiB, ctx / hive::qasr::kMiB, E->info.weights_mib, E->info.kv_mib, E->info.work_mib, cublas_ws / hive::qasr::kMiB,
              graphs_and_modules / hive::qasr::kMiB, limit ? (" (limit " + std::to_string(limit >> 20) + " MiB)").c_str() : "");
      if (limit && plan > (double)limit)
        throw std::runtime_error("VRAM plan " + std::to_string((long)(plan / hive::qasr::kMiB)) + " MiB exceeds the limit " +
                                 std::to_string(limit >> 20) + " MiB (lower max_ctx / max_seqs / prefill_chunk / encoder_windows)");
      if (plan - ctx > (double)f1)
        throw std::runtime_error("not enough free VRAM: need " + std::to_string((long)((plan - ctx) / hive::qasr::kMiB)) + " MiB after the context, " +
                                 std::to_string(f1 >> 20) + " MiB free");
    };
    E->load_weights(ckpt_dir);
    E->work.buf.alloc(E->work.used);
    E->persist.buf.alloc(E->persist.used);
    CUDA_CHECK(cudaMemset(E->persist.buf.p, 0, E->persist.used));
    E->gemm.init(E->st, cublas_ws);
    CUDA_CHECK(cudaMallocHost(&E->h_pf_io, (size_t)3 * E->P * 4));
    CUDA_CHECK(cudaMallocHost(&E->h_dec_io, 12 * 4));
    CUDA_CHECK(cudaMallocHost(&E->h_dec_out, 4 * 4));
    CUDA_CHECK(cudaMallocHost(&E->h_logits, (size_t)opt.max_seqs * E->c.vocab * 4));
    E->init_rope();
    E->active.assign(opt.max_seqs, nullptr);
    E->capture_graphs();
    if (opt.warmup) E->warmup();
    CUDA_CHECK(cudaStreamSynchronize(E->st));
    size_t f2 = 0;
    CUDA_CHECK(cudaMemGetInfo(&f2, &tot));
    E->info.used_mib = nvml && f0 > f2 ? (f0 - f2) / hive::qasr::kMiB : 0;
    E->info.max_seqs = opt.max_seqs; E->info.max_ctx = opt.max_ctx; E->info.vocab = E->c.vocab; E->info.hidden = E->c.hidden;
    E->info.n_mels = E->c.n_mels;
    fprintf(stderr, "[qasr] ready: %d slots x %d positions, process VRAM %.0f MiB%s\n", opt.max_seqs, opt.max_ctx,
            E->info.used_mib > 0 ? E->info.used_mib : E->info.plan_mib, E->info.used_mib > 0 ? " (measured)" : " (plan)");
    E->worker = std::thread([E] { E->run(); });
    auto* h = new qasr_engine;
    h->e = std::move(e);
    return h;
  } catch (const std::exception& ex) {
    set_err(err, err_len, ex.what());
    return nullptr;
  }
}

void qasr_destroy(qasr_engine* h) { delete h; }

void qasr_get_info(const qasr_engine* h, qasr_info* out) {
  if (h && out) *out = h->e->info;
}

namespace {
bool fill_prompt(Engine& E, Job& j, const float* mel, int n_mels, int n_frames, const int32_t* pre, int n_pre, const int32_t* post, int n_post,
                 std::string& err) {
  if (n_frames < 0 || n_pre < 0 || n_post < 0) { err = "negative length"; return false; }
  if (n_frames > 0 && (!mel || n_mels != E.c.n_mels)) { err = "mel features must be [" + std::to_string(E.c.n_mels) + ", frames]"; return false; }
  if (n_frames > E.max_frames) { err = "audio too long (" + std::to_string(n_frames / 100.0) + " s)"; return false; }
  j.mel = mel;
  j.n_frames = n_frames;
  j.na = qasr_audio_tokens(n_frames);
  j.a0 = n_pre;
  j.ids.assign(pre, pre + n_pre);
  j.ids.insert(j.ids.end(), j.na, E.c.audio_token);
  j.ids.insert(j.ids.end(), post, post + n_post);
  for (int32_t t : j.ids)
    if (t < 0 || t >= E.c.vocab) { err = "token id out of range"; return false; }
  if (j.ids.empty()) { err = "empty prompt"; return false; }
  if ((int)j.ids.size() >= E.opt.max_ctx) {
    err = "prompt of " + std::to_string(j.ids.size()) + " tokens (" + std::to_string(j.na) + " audio) does not fit the context of " +
          std::to_string(E.opt.max_ctx);
    return false;
  }
  return true;
}
}  // namespace

int qasr_transcribe(qasr_engine* h, const float* mel, int n_mels, int n_frames, const int32_t* pre_ids, int n_pre, const int32_t* post_ids,
                    int n_post, float temperature, float top_p, int top_k, uint64_t seed, int max_new_tokens, const int32_t* stop_ids,
                    int n_stop, const volatile int32_t* cancel, int32_t* out_ids, int out_cap, qasr_stats* stats, char* err, int err_len) {
  if (!h) { set_err(err, err_len, "no engine"); return -1; }
  Engine& E = *h->e;
  Job j;
  std::string e;
  if (!fill_prompt(E, j, mel, n_mels, n_frames, pre_ids, n_pre, post_ids, n_post, e)) { set_err(err, err_len, e); return -1; }
  j.temperature = std::isfinite(temperature) && temperature > 0 ? temperature : 0.f;
  j.top_p = top_p > 0 && top_p < 1 ? top_p : 1.f;
  j.top_k = top_k;
  j.max_new = std::min(std::max(0, max_new_tokens), E.opt.max_ctx - (int)j.ids.size());
  j.max_new = std::min(j.max_new, out_cap);
  if (stop_ids && n_stop > 0) j.stop.assign(stop_ids, stop_ids + n_stop);
  j.cancel = cancel;
  j.rng.seed(seed ? seed : std::random_device{}());
  j.st.n_prompt = (int)j.ids.size();
  j.st.n_audio = j.na;
  j.st.stop_token = -1;
  E.submit(&j);
  if (!j.err.empty()) { set_err(err, err_len, j.err); return -1; }
  const int n = std::min((int)j.out.size(), out_cap);
  if (n > 0) std::memcpy(out_ids, j.out.data(), (size_t)n * 4);
  if (stats) *stats = j.st;
  return n;
}

int qasr_debug_encode(qasr_engine* h, const float* mel, int n_mels, int n_frames, float* out, int out_rows, char* err, int err_len) {
  if (!h) { set_err(err, err_len, "no engine"); return -1; }
  Engine& E = *h->e;
  Job j;
  std::string e;
  if (!fill_prompt(E, j, mel, n_mels, n_frames, nullptr, 0, nullptr, 0, e)) { set_err(err, err_len, e); return -1; }
  if (out_rows < j.na) { set_err(err, err_len, "output too small"); return -1; }
  j.kind = Kind::kEncode;
  j.fout = out;
  E.submit(&j);
  if (!j.err.empty()) { set_err(err, err_len, j.err); return -1; }
  return j.na;
}

int qasr_debug_prompt_logits(qasr_engine* h, const float* mel, int n_mels, int n_frames, const int32_t* pre_ids, int n_pre,
                             const int32_t* post_ids, int n_post, float* logits, char* err, int err_len) {
  if (!h) { set_err(err, err_len, "no engine"); return -1; }
  Engine& E = *h->e;
  Job j;
  std::string e;
  if (!fill_prompt(E, j, mel, n_mels, n_frames, pre_ids, n_pre, post_ids, n_post, e)) { set_err(err, err_len, e); return -1; }
  j.kind = Kind::kLogits;
  j.fout = logits;
  E.submit(&j);
  if (!j.err.empty()) { set_err(err, err_len, j.err); return -1; }
  return E.c.vocab;
}

}  // extern "C"
