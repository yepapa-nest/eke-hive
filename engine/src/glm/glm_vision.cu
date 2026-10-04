// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash vision encoder — see hive/glm/glm_vision.h.
#include "hive/glm/glm_vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "hive/glm/glm_kernels.h"
#include "hive/kernels.h"

namespace hive::glm {

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

// q and k of one (row, head): RMSNorm over the 64 dims (weight per dim) then the axial 2D rotary embedding, in place in qkv [M, 3·heads·64].
//   pos: [M, 2] (row index, column index of the patch). inv: [16] inverse frequencies.
__global__ void qk_norm_rope_k(bf16* qkv, int M, int heads, const bf16* qn, const bf16* kn, float eps, const int32_t* pos,
                               const float* inv) {
  const int row = blockIdx.x, head = blockIdx.y, which = blockIdx.z;  // which 0 = q, 1 = k
  const int d = threadIdx.x;                                          // 64 threads
  bf16* p = qkv + (size_t)row * 3 * heads * 64 + (size_t)which * heads * 64 + (size_t)head * 64;
  __shared__ float v[64];
  __shared__ float red[2];
  const float x = __bfloat162float(p[d]);
  float s = x * x;
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  if ((d & 31) == 0) red[d >> 5] = s;
  __syncthreads();
  const float r = rsqrtf((red[0] + red[1]) / 64.f + eps);
  const bf16* w = which ? kn : qn;
  // the reference rounds the normed value to bf16 (its RMSNorm returns the input dtype) before the rotary math in fp32
  v[d] = __bfloat162float(__float2bfloat16(x * r)) * __bfloat162float(w[d]);
  v[d] = __bfloat162float(__float2bfloat16(v[d]));
  __syncthreads();
  const int j = d & 31;                                // dims [0,32) and [32,64) carry the same angles
  const float ang = (float)(j < 16 ? pos[row * 2] : pos[row * 2 + 1]) * inv[j & 15];
  float sn, cs;
  sincosf(ang, &sn, &cs);
  const float rot = d < 32 ? -v[d + 32] : v[d - 32];  // rotate-half
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

// out = silu(min(gate, lim)) · clamp(up, −lim, lim) (gate/up already include their bias), bf16 [M, N]
__global__ void swiglu_clamp_k(const bf16* g, const bf16* u, bf16* out, size_t n, float lim) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float a = fminf(__bfloat162float(g[i]), lim);
    const float b = fminf(fmaxf(__bfloat162float(u[i]), -lim), lim);
    const float sa = __bfloat162float(__float2bfloat16(a / (1.f + __expf(-a))));
    out[i] = __float2bfloat16(sa * b);
  }
}
void swiglu_clamp(const bf16* g, const bf16* u, bf16* out, size_t n, float lim, cudaStream_t st) {
  swiglu_clamp_k<<<(unsigned)std::min<size_t>((n + 255) / 256, 65535), 256, 0, st>>>(g, u, out, n, lim);
}

// exact (erf) GELU in place
__global__ void gelu_k(bf16* x, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
  }
}

}  // namespace

DevBuf GlmVision::load(GlmModel& m, const std::string& name, size_t expect_elems) {
  const TensorInfo& t = m.ckpt().get(name);
  HIVE_CHECK(t.dtype == "BF16" && (size_t)t.numel() == expect_elems, "GLM vision: unexpected tensor " + name + " (" + t.dtype + ")");
  DevBuf b(t.nbytes);
  CUDA_CHECK(cudaMemcpy(b.p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  bytes_ += t.nbytes;
  return b;
}

GlmVision::GlmVision(GlmModel& m) : blas_(nullptr) {
  const GlmConfig& c = m.cfg();
  dim_ = c.vision_dim; heads_ = c.vision_heads; inter_ = c.vision_inter; out_ = c.vision_out; pinter_ = c.vision_proj_inter; depth_ = c.vision_layers;
  HIVE_CHECK(dim_ / heads_ == 64 && c.vision_patch == 14 && c.vision_temporal == 2 && c.vision_merge == 2, "GLM vision: unsupported geometry");
  const std::string p = "model.visual.";
  patch_w_ = load(m, p + "patch_embed.proj.weight", (size_t)dim_ * kPatchDim);  // conv3d [dim, 3, 2, 14, 14] = linear [dim, 1176] (same flatten order as the patches)
  patch_b_ = load(m, p + "patch_embed.proj.bias", dim_);
  blocks_.resize(depth_);
  for (int l = 0; l < depth_; ++l) {
    const std::string b = p + "blocks." + std::to_string(l) + ".";
    Block& B = blocks_[l];
    B.norm1 = load(m, b + "norm1.weight", dim_); B.norm2 = load(m, b + "norm2.weight", dim_);
    B.qkv_w = load(m, b + "attn.qkv.weight", (size_t)3 * dim_ * dim_); B.qkv_b = load(m, b + "attn.qkv.bias", 3 * dim_);
    B.q_norm = load(m, b + "attn.q_norm.weight", 64); B.k_norm = load(m, b + "attn.k_norm.weight", 64);
    B.proj_w = load(m, b + "attn.proj.weight", (size_t)dim_ * dim_); B.proj_b = load(m, b + "attn.proj.bias", dim_);
    B.gate_w = load(m, b + "mlp.gate_proj.weight", (size_t)inter_ * dim_); B.gate_b = load(m, b + "mlp.gate_proj.bias", inter_);
    B.up_w = load(m, b + "mlp.up_proj.weight", (size_t)inter_ * dim_); B.up_b = load(m, b + "mlp.up_proj.bias", inter_);
    B.down_w = load(m, b + "mlp.down_proj.weight", (size_t)dim_ * inter_); B.down_b = load(m, b + "mlp.down_proj.bias", dim_);
  }
  post_norm_ = load(m, p + "post_layernorm.weight", dim_);
  {  // downsample conv2d [out, dim, 2, 2] → linear [out, 4·dim] with input order (i·2 + j)·dim + c (= 4 consecutive patches of a block)
    const TensorInfo& t = m.ckpt().get(p + "downsample.weight");
    HIVE_CHECK(t.dtype == "BF16" && (size_t)t.numel() == (size_t)out_ * dim_ * 4, "GLM vision: downsample.weight");
    std::vector<uint16_t> src((size_t)t.numel()), dst(src.size());
    std::memcpy(src.data(), t.data, t.nbytes);
    for (int o = 0; o < out_; ++o)
      for (int ch = 0; ch < dim_; ++ch)
        for (int ij = 0; ij < 4; ++ij) dst[(size_t)o * 4 * dim_ + (size_t)ij * dim_ + ch] = src[((size_t)o * dim_ + ch) * 4 + ij];
    down_w_.alloc(t.nbytes);
    CUDA_CHECK(cudaMemcpy(down_w_.p, dst.data(), t.nbytes, cudaMemcpyHostToDevice));
    bytes_ += t.nbytes;
  }
  down_b_ = load(m, p + "downsample.bias", out_);
  m_proj_ = load(m, p + "merger.proj.weight", (size_t)out_ * out_);
  m_ln_w_ = load(m, p + "merger.post_projection_norm.weight", out_);
  m_ln_b_ = load(m, p + "merger.post_projection_norm.bias", out_);
  m_gate_ = load(m, p + "merger.gate_proj.weight", (size_t)pinter_ * out_);
  m_up_ = load(m, p + "merger.up_proj.weight", (size_t)pinter_ * out_);
  m_down_ = load(m, p + "merger.down_proj.weight", (size_t)out_ * pinter_);
  float inv[16];
  for (int i = 0; i < 16; ++i) inv[i] = 1.f / std::pow(10000.f, (2.f * i) / 32.f);  // spatial dim = head_dim / 2 = 32, frequencies at even steps
  inv_freq_.alloc(sizeof inv);
  CUDA_CHECK(cudaMemcpy(inv_freq_.p, inv, sizeof inv, cudaMemcpyHostToDevice));
}

static constexpr int kQTile = 1024;

size_t GlmVision::scratch_bytes(int n) {
  const size_t N = (size_t)n;
  // x, xn, att [N,1024] · qkv [N,3072] · gate, up, act [N,4096] · one score tile fp32 + bf16 [1024,N] · positions ·
  //   downsample/merger [N/4,4096]×2 + [N/4,10240]×3 · allocator slack
  return N * 1024 * 2 * 3 + N * 3072 * 2 + N * 4096 * 2 * 3 + (size_t)std::min(kQTile, n) * N * (4 + 2) + N * 8 + (N / 4) * 4096 * 2 * 2 +
         (N / 4) * 10240 * 2 * 3 + (64u << 20);
}

void GlmVision::encode(const bf16* patches, int gh, int gw, bf16* out, cudaStream_t st) {
  HIVE_CHECK(gh > 0 && gw > 0 && gh % 2 == 0 && gw % 2 == 0, "GLM vision: grid must be even");
  const int N = gh * gw, D = dim_, NB = N / 4;
  blas_.set_stream(st);
  DevBuf x((size_t)N * D * 2), xn((size_t)N * D * 2), att((size_t)N * D * 2), qkv((size_t)N * 3 * D * 2);
  DevBuf g((size_t)N * inter_ * 2), u((size_t)N * inter_ * 2), a((size_t)N * inter_ * 2);
  const int qt = std::min(kQTile, N);
  DevBuf S((size_t)qt * N * 4), P((size_t)qt * N * 2);
  // patch positions in merge-block order (the order of the rows in `patches`)
  std::vector<int32_t> pos((size_t)N * 2);
  {
    size_t r = 0;
    for (int bh = 0; bh < gh / 2; ++bh)
      for (int bw = 0; bw < gw / 2; ++bw)
        for (int mh = 0; mh < 2; ++mh)
          for (int mw = 0; mw < 2; ++mw) { pos[r * 2] = bh * 2 + mh; pos[r * 2 + 1] = bw * 2 + mw; ++r; }
  }
  DevBuf posd(pos.size() * 4);
  CUDA_CHECK(cudaMemcpyAsync(posd.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, st));
  bf16* X = x.as<bf16>(); bf16* XN = xn.as<bf16>(); bf16* QKV = qkv.as<bf16>(); bf16* ATT = att.as<bf16>();
  // patch embedding
  blas_.gemm_bf16(patches, patch_w_.as<bf16>(), X, N, D, kPatchDim);
  add_bias(X, patch_b_.as<bf16>(), N, D, st);
  const float scale = 1.f / 8.f;  // head_dim^-0.5
  for (const Block& B : blocks_) {
    k::rmsnorm(X, B.norm1.as<bf16>(), eps_, N, D, XN, st);
    blas_.gemm_bf16(XN, B.qkv_w.as<bf16>(), QKV, N, 3 * D, D);
    add_bias(QKV, B.qkv_b.as<bf16>(), N, 3 * D, st);
    qk_norm_rope_k<<<dim3(N, heads_, 2), 64, 0, st>>>(QKV, N, heads_, B.q_norm.as<bf16>(), B.k_norm.as<bf16>(), eps_, posd.as<int32_t>(),
                                                     inv_freq_.as<float>());
    for (int h = 0; h < heads_; ++h) {
      const bf16* Q = QKV + (size_t)h * 64;
      const bf16* K = QKV + (size_t)D + (size_t)h * 64;
      const bf16* V = QKV + (size_t)2 * D + (size_t)h * 64;
      for (int q0 = 0; q0 < N; q0 += qt) {
        const int rows = std::min(qt, N - q0);
        blas_.gemm_bf16_f32out_ld(Q + (size_t)q0 * 3 * D, 3 * D, K, 3 * D, S.as<float>(), N, rows, N, 64);
        softmax_rows_k<<<rows, 256, 0, st>>>(S.as<float>(), N, scale, P.as<bf16>());
        blas_.gemm_bf16_nn_ld(P.as<bf16>(), N, V, 3 * D, ATT + (size_t)q0 * D + (size_t)h * 64, D, rows, 64, N);
      }
    }
    blas_.gemm_bf16_acc(ATT, B.proj_w.as<bf16>(), X, N, D, D);  // x += proj(att)
    add_bias(X, B.proj_b.as<bf16>(), N, D, st);
    k::rmsnorm(X, B.norm2.as<bf16>(), eps_, N, D, XN, st);
    blas_.gemm_bf16(XN, B.gate_w.as<bf16>(), g.as<bf16>(), N, inter_, D);
    add_bias(g.as<bf16>(), B.gate_b.as<bf16>(), N, inter_, st);
    blas_.gemm_bf16(XN, B.up_w.as<bf16>(), u.as<bf16>(), N, inter_, D);
    add_bias(u.as<bf16>(), B.up_b.as<bf16>(), N, inter_, st);
    swiglu_clamp(g.as<bf16>(), u.as<bf16>(), a.as<bf16>(), (size_t)N * inter_, limit_, st);
    blas_.gemm_bf16_acc(a.as<bf16>(), B.down_w.as<bf16>(), X, N, D, inter_);
    add_bias(X, B.down_b.as<bf16>(), N, D, st);
  }
  k::rmsnorm(X, post_norm_.as<bf16>(), eps_, N, D, XN, st);
  // 2×2 downsample: each group of 4 consecutive rows is one block → [NB, 4·D] · W[out, 4·D]ᵀ
  DevBuf ds((size_t)NB * out_ * 2), mt((size_t)NB * out_ * 2), mg((size_t)NB * pinter_ * 2), mu((size_t)NB * pinter_ * 2), ma((size_t)NB * pinter_ * 2);
  blas_.gemm_bf16(XN, down_w_.as<bf16>(), ds.as<bf16>(), NB, out_, 4 * D);
  add_bias(ds.as<bf16>(), down_b_.as<bf16>(), NB, out_, st);
  // merger
  blas_.gemm_bf16(ds.as<bf16>(), m_proj_.as<bf16>(), mt.as<bf16>(), NB, out_, out_);
  layernorm(mt.as<bf16>(), m_ln_w_.as<bf16>(), m_ln_b_.as<bf16>(), 1e-5f, NB, out_, ds.as<bf16>(), st);  // ds is free again: LayerNorm out of place
  gelu_k<<<(unsigned)std::min<size_t>(((size_t)NB * out_ + 255) / 256, 65535), 256, 0, st>>>(ds.as<bf16>(), (size_t)NB * out_);
  blas_.gemm_bf16(ds.as<bf16>(), m_gate_.as<bf16>(), mg.as<bf16>(), NB, pinter_, out_);
  blas_.gemm_bf16(ds.as<bf16>(), m_up_.as<bf16>(), mu.as<bf16>(), NB, pinter_, out_);
  swiglu_clamp(mg.as<bf16>(), mu.as<bf16>(), ma.as<bf16>(), (size_t)NB * pinter_, limit_, st);
  blas_.gemm_bf16(ma.as<bf16>(), m_down_.as<bf16>(), out, NB, out_, pinter_);
  CUDA_CHECK(cudaStreamSynchronize(st));  // the scratch buffers above are freed on return
}

}  // namespace hive::glm
