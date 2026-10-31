// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next vision encoder — see hive/qwen/qwen_vision.h. Structure adapted from hive/glm/glm_vision.cu (attention in query tiles of
//   1024 rows with fp32 scores, so N×N is never held).
#include "hive/qwen/qwen_vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "hive/ckpt_quant.h"
#include "hive/glm/glm_kernels.h"

namespace hive::qwen {

namespace {

__global__ void add_bias_k(bf16* x, const bf16* b, int M, int N) {
  const size_t total = (size_t)M * N;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x)
    x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(b[i % N]));
}
void add_bias(bf16* x, const bf16* b, int M, int N, cudaStream_t st) {
  const size_t total = (size_t)M * N;
  add_bias_k<<<(unsigned)std::min<size_t>((total + 255) / 256, 65535), 256, 0, st>>>(x, b, M, N);
}

// x[r, :] = bf16(x[r, :] + bf16(Σ_i w[r, i] · pos[idx[r, i], :]))  — the bilinear position embedding of patch row r (4 corners)
__global__ void add_pos_k(bf16* x, const bf16* pos, const int32_t* idx, const float* w, int N, int D) {
  const int r = blockIdx.x;
  const int32_t* ir = idx + (size_t)r * 4;
  const float* wr = w + (size_t)r * 4;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) s = fmaf(wr[i], __bfloat162float(pos[(size_t)ir[i] * D + d]), s);
    const float p = __bfloat162float(__float2bfloat16(s));
    x[(size_t)r * D + d] = __float2bfloat16(__bfloat162float(x[(size_t)r * D + d]) + p);
  }
}

// q and k of one (row, head): the axial 2D rotary embedding in place in qkv [M, 3·heads·72]. pos [M, 2] (row, column), inv [18].
__global__ void qk_rope_k(bf16* qkv, int heads, const int32_t* pos, const float* inv) {
  constexpr int HD = 72, HALF = 36, NF = 18;
  const int row = blockIdx.x, head = blockIdx.y, which = blockIdx.z;  // which 0 = q, 1 = k
  const int d = threadIdx.x;                                          // 72 threads
  bf16* p = qkv + (size_t)row * 3 * heads * HD + (size_t)which * heads * HD + (size_t)head * HD;
  __shared__ float v[HD];
  v[d] = __bfloat162float(p[d]);
  __syncthreads();
  const int j = d < HALF ? d : d - HALF;  // dims [0,36) and [36,72) carry the same angles
  const float ang = (float)(j < NF ? pos[row * 2] : pos[row * 2 + 1]) * inv[j < NF ? j : j - NF];
  float sn, cs;
  sincosf(ang, &sn, &cs);
  const float rot = d < HALF ? -v[d + HALF] : v[d - HALF];  // rotate-half
  p[d] = __float2bfloat16(v[d] * cs + rot * sn);
}

// Row softmax of fp32 scores [R, C] (scaled) → bf16 probabilities.
__global__ void softmax_rows_k(const float* S, int C, float scale, bf16* P) {
  const float* s = S + (size_t)blockIdx.x * C;
  bf16* p = P + (size_t)blockIdx.x * C;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int c = threadIdx.x; c < C; c += blockDim.x) mx = fmaxf(mx, s[c] * scale);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffff, t, o));
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  mx = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) sum += __expf(s[c] * scale - mx);
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float inv = 1.f / red[0];
  for (int c = threadIdx.x; c < C; c += blockDim.x) p[c] = __float2bfloat16(__expf(s[c] * scale - mx) * inv);
}

// GELU in place: tanh approximation (blocks) or exact erf (merger)
__global__ void gelu_k(bf16* x, size_t n, bool tanh_form) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float v = __bfloat162float(x[i]);
    float y;
    if (tanh_form) y = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    else y = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    x[i] = __float2bfloat16(y);
  }
}
void gelu(bf16* x, size_t n, bool tanh_form, cudaStream_t st) {
  gelu_k<<<(unsigned)std::min<size_t>((n + 255) / 256, 65535), 256, 0, st>>>(x, n, tanh_form);
}

constexpr int kQTile = 1024;

}  // namespace

// BF16 tensors are copied. The FP4-Local checkpoint quantizes the linear weights (attention qkv / proj MXFP8, MLP and merger NVFP4 W4A16):
//   they are expanded to bf16 here (MXFP8 exactly, NVFP4 with one rounding — hive/ckpt_quant.h), so the encoder's kernels are unchanged.
DevBuf QwenVision::load(QwenModel& m, const std::string& name, size_t expect_elems) {
  const TensorInfo& t = m.ckpt().get(name);
  if (t.dtype != "BF16" && name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) {
    const std::string base = name.substr(0, name.size() - 6);
    const ckq::Dense f = ckq::dense_format(m.ckpt(), base);
    if (f == ckq::Dense::MXFP8 || f == ckq::Dense::NVFP4) {
      const std::vector<uint16_t> h = ckq::expand_bf16(m.ckpt(), base, 8);
      HIVE_CHECK(h.size() == expect_elems, "Qwen vision: unexpected tensor size " + name);
      DevBuf b(h.size() * 2);
      CUDA_CHECK(cudaMemcpy(b.p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
      bytes_ += h.size() * 2;
      return b;
    }
  }
  HIVE_CHECK(t.dtype == "BF16" && (size_t)t.numel() == expect_elems, "Qwen vision: unexpected tensor " + name + " (" + t.dtype + ")");
  DevBuf b(t.nbytes);
  CUDA_CHECK(cudaMemcpy(b.p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  bytes_ += t.nbytes;
  return b;
}

QwenVision::QwenVision(QwenModel& m) : blas_(nullptr) {
  const QwenConfig& c = m.cfg();
  dim_ = c.vision_dim; heads_ = c.vision_heads; inter_ = c.vision_inter; out_ = c.vision_out; depth_ = c.vision_layers;
  side_ = (int)std::lround(std::sqrt((double)c.vision_pos));
  HIVE_CHECK(dim_ / heads_ == 72 && dim_ % heads_ == 0 && c.vision_patch == 16 && c.vision_temporal == 2 && c.vision_merge == 2 &&
                 side_ * side_ == c.vision_pos,
             "Qwen vision: unsupported geometry");
  const std::string p = "model.visual.";
  patch_w_ = load(m, p + "patch_embed.proj.weight", (size_t)dim_ * kPatchDim);  // conv3d [dim, 3, 2, 16, 16] = linear [dim, 1536]
  patch_b_ = load(m, p + "patch_embed.proj.bias", dim_);
  pos_ = load(m, p + "pos_embed.weight", (size_t)c.vision_pos * dim_);
  blocks_.resize(depth_);
  for (int l = 0; l < depth_; ++l) {
    const std::string b = p + "blocks." + std::to_string(l) + ".";
    Block& B = blocks_[l];
    B.n1w = load(m, b + "norm1.weight", dim_); B.n1b = load(m, b + "norm1.bias", dim_);
    B.n2w = load(m, b + "norm2.weight", dim_); B.n2b = load(m, b + "norm2.bias", dim_);
    B.qkv_w = load(m, b + "attn.qkv.weight", (size_t)3 * dim_ * dim_); B.qkv_b = load(m, b + "attn.qkv.bias", 3 * dim_);
    B.proj_w = load(m, b + "attn.proj.weight", (size_t)dim_ * dim_); B.proj_b = load(m, b + "attn.proj.bias", dim_);
    B.fc1_w = load(m, b + "mlp.linear_fc1.weight", (size_t)inter_ * dim_); B.fc1_b = load(m, b + "mlp.linear_fc1.bias", inter_);
    B.fc2_w = load(m, b + "mlp.linear_fc2.weight", (size_t)dim_ * inter_); B.fc2_b = load(m, b + "mlp.linear_fc2.bias", dim_);
  }
  const size_t md = (size_t)4 * dim_;
  mn_w_ = load(m, p + "merger.norm.weight", dim_); mn_b_ = load(m, p + "merger.norm.bias", dim_);
  m1_w_ = load(m, p + "merger.linear_fc1.weight", md * md); m1_b_ = load(m, p + "merger.linear_fc1.bias", md);
  m2_w_ = load(m, p + "merger.linear_fc2.weight", (size_t)out_ * md); m2_b_ = load(m, p + "merger.linear_fc2.bias", out_);
  float inv[18];
  for (int j = 0; j < 18; ++j) inv[j] = 1.f / std::pow(10000.f, (2.f * j) / 36.f);  // rotary dim = head_dim / 2 = 36
  inv_freq_.alloc(sizeof inv);
  CUDA_CHECK(cudaMemcpy(inv_freq_.p, inv, sizeof inv, cudaMemcpyHostToDevice));
}

size_t QwenVision::scratch_bytes(int n) {
  const size_t N = (size_t)n, D = 1152, I = 4304, MD = 4 * 1152;
  // x, xn, att [N,D] · qkv [N,3D] · fc1 [N,I] · one score tile fp32 + bf16 [1024,N] · positions (pos 8 + interp 32 B per row) ·
  //   merger [N/4,4D] · allocator slack
  return N * D * 2 * 3 + N * 3 * D * 2 + N * I * 2 + (size_t)std::min(kQTile, n) * N * (4 + 2) + N * 40 + (N / 4) * MD * 2 + (64u << 20);
}

void QwenVision::encode(const bf16* patches, int gh, int gw, bf16* out, cudaStream_t st) {
  HIVE_CHECK(gh > 0 && gw > 0 && gh % 2 == 0 && gw % 2 == 0, "Qwen vision: grid must be even");
  const int N = gh * gw, D = dim_, NB = N / 4, HD = dim_ / heads_;
  blas_.set_stream(st);
  DevBuf x((size_t)N * D * 2), xn((size_t)N * D * 2), att((size_t)N * D * 2), qkv((size_t)N * 3 * D * 2), f1((size_t)N * inter_ * 2);
  const int qt = std::min(kQTile, N);
  DevBuf S((size_t)qt * N * 4), P((size_t)qt * N * 2);
  // per patch row (merge-block order): grid position and the 4 bilinear corners of the position embedding
  std::vector<int32_t> pos((size_t)N * 2), cidx((size_t)N * 4);
  std::vector<float> cw((size_t)N * 4);
  {
    auto samples = [&](int n, std::vector<int>& f, std::vector<int>& cl, std::vector<float>& fr) {
      f.resize(n); cl.resize(n); fr.resize(n);
      for (int i = 0; i < n; ++i) {
        const float t = n == 1 ? 0.f : (float)i * (float)(side_ - 1) / (float)(n - 1);  // linspace(0, side − 1, n)
        const int fl = std::min((int)t, side_ - 1);
        f[i] = fl; cl[i] = std::min(fl + 1, side_ - 1); fr[i] = t - (float)fl;
      }
    };
    std::vector<int> hf, hc, wf, wc;
    std::vector<float> dh, dw;
    samples(gh, hf, hc, dh);
    samples(gw, wf, wc, dw);
    size_t r = 0;
    for (int bh = 0; bh < gh / 2; ++bh)
      for (int bw = 0; bw < gw / 2; ++bw)
        for (int mh = 0; mh < 2; ++mh)
          for (int mw = 0; mw < 2; ++mw) {
            const int y = bh * 2 + mh, xx = bw * 2 + mw;
            pos[r * 2] = y; pos[r * 2 + 1] = xx;
            const float a = dh[y], b = dw[xx];
            cidx[r * 4 + 0] = hf[y] * side_ + wf[xx]; cw[r * 4 + 0] = (1.f - a) * (1.f - b);
            cidx[r * 4 + 1] = hf[y] * side_ + wc[xx]; cw[r * 4 + 1] = (1.f - a) * b;
            cidx[r * 4 + 2] = hc[y] * side_ + wf[xx]; cw[r * 4 + 2] = a * (1.f - b);
            cidx[r * 4 + 3] = hc[y] * side_ + wc[xx]; cw[r * 4 + 3] = a * b;
            ++r;
          }
  }
  DevBuf posd(pos.size() * 4), cid(cidx.size() * 4), cwd(cw.size() * 4);
  CUDA_CHECK(cudaMemcpyAsync(posd.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(cid.p, cidx.data(), cidx.size() * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(cwd.p, cw.data(), cw.size() * 4, cudaMemcpyHostToDevice, st));
  bf16* X = x.as<bf16>(); bf16* XN = xn.as<bf16>(); bf16* QKV = qkv.as<bf16>(); bf16* ATT = att.as<bf16>();
  // patch embedding + position embedding
  blas_.gemm_bf16(patches, patch_w_.as<bf16>(), X, N, D, kPatchDim);
  add_bias(X, patch_b_.as<bf16>(), N, D, st);
  add_pos_k<<<N, 256, 0, st>>>(X, pos_.as<bf16>(), cid.as<int32_t>(), cwd.as<float>(), N, D);
  const float scale = 1.f / std::sqrt((float)HD);
  for (const Block& B : blocks_) {
    glm::layernorm(X, B.n1w.as<bf16>(), B.n1b.as<bf16>(), eps_, N, D, XN, st);
    blas_.gemm_bf16(XN, B.qkv_w.as<bf16>(), QKV, N, 3 * D, D);
    add_bias(QKV, B.qkv_b.as<bf16>(), N, 3 * D, st);
    qk_rope_k<<<dim3(N, heads_, 2), HD, 0, st>>>(QKV, heads_, posd.as<int32_t>(), inv_freq_.as<float>());
    for (int h = 0; h < heads_; ++h) {
      const bf16* Q = QKV + (size_t)h * HD;
      const bf16* K = QKV + (size_t)D + (size_t)h * HD;
      const bf16* V = QKV + (size_t)2 * D + (size_t)h * HD;
      for (int q0 = 0; q0 < N; q0 += qt) {
        const int rows = std::min(qt, N - q0);
        blas_.gemm_bf16_f32out_ld(Q + (size_t)q0 * 3 * D, 3 * D, K, 3 * D, S.as<float>(), N, rows, N, HD);
        softmax_rows_k<<<rows, 256, 0, st>>>(S.as<float>(), N, scale, P.as<bf16>());
        blas_.gemm_bf16_nn_ld(P.as<bf16>(), N, V, 3 * D, ATT + (size_t)q0 * D + (size_t)h * HD, D, rows, HD, N);
      }
    }
    blas_.gemm_bf16_acc(ATT, B.proj_w.as<bf16>(), X, N, D, D);  // x += proj(att)
    add_bias(X, B.proj_b.as<bf16>(), N, D, st);
    glm::layernorm(X, B.n2w.as<bf16>(), B.n2b.as<bf16>(), eps_, N, D, XN, st);
    blas_.gemm_bf16(XN, B.fc1_w.as<bf16>(), f1.as<bf16>(), N, inter_, D);
    add_bias(f1.as<bf16>(), B.fc1_b.as<bf16>(), N, inter_, st);
    gelu(f1.as<bf16>(), (size_t)N * inter_, true, st);
    blas_.gemm_bf16_acc(f1.as<bf16>(), B.fc2_w.as<bf16>(), X, N, D, inter_);  // x += fc2(…)
    add_bias(X, B.fc2_b.as<bf16>(), N, D, st);
  }
  // merger: LayerNorm per patch, then each group of 4 consecutive rows (one merge block) is one [4·D] row
  glm::layernorm(X, mn_w_.as<bf16>(), mn_b_.as<bf16>(), eps_, N, D, XN, st);
  const int MD = 4 * D;
  DevBuf mh((size_t)NB * MD * 2);
  blas_.gemm_bf16(XN, m1_w_.as<bf16>(), mh.as<bf16>(), NB, MD, MD);
  add_bias(mh.as<bf16>(), m1_b_.as<bf16>(), NB, MD, st);
  gelu(mh.as<bf16>(), (size_t)NB * MD, false, st);
  blas_.gemm_bf16(mh.as<bf16>(), m2_w_.as<bf16>(), out, NB, out_, MD);
  add_bias(out, m2_b_.as<bf16>(), NB, out_, st);
  CUDA_CHECK(cudaStreamSynchronize(st));  // the scratch buffers above are freed on return
}

}  // namespace hive::qwen
