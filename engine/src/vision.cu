// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Vision encoder implementation — follows the reference vision.py exactly (bf16 compute, fp32 accumulation). One image at a time.
#include "hive/vision.h"

#include <cmath>
#include <vector>

#include <cstdlib>
#include <fstream>
#include <cstring>
#include <string>

#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

namespace hive {

static void vdump(cudaStream_t st, const char* name, const void* dev, size_t bytes) {
  const char* dir = getenv("HIVE_VISION_DUMP");
  if (!dir || !*dir || strcmp(dir, "0") == 0) return;  // unset/""/"0" = off
  std::vector<uint8_t> buf(bytes);
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaMemcpy(buf.data(), dev, bytes, cudaMemcpyDeviceToHost));
  std::ofstream f(std::string(dir) + "/" + name, std::ios::binary);
  f.write((const char*)buf.data(), bytes);
}

namespace k {
// ---- Vision-only kernels ----------------------------------------------------------------------------------------------
namespace {
__global__ void vrms_kernel(const bf16* __restrict__ x, const float* __restrict__ w, float eps, int D, bf16* __restrict__ out) {
  __shared__ float red[32];
  const int n = blockIdx.x;
  const bf16* p = x + (size_t)n * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += 256) { float v = bf2f(p[i]); ss += v * v; }
  ss = cu::warp_allreduce<cu::Add>(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    t = cu::warp_allreduce<cu::Add>(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  float rs = rsqrtf(red[0] / D + eps);
  for (int i = threadIdx.x; i < D; i += 256) out[(size_t)n * D + i] = f2bf(w[i] * (bf2f(p[i]) * rs));
}
// Fill C with bias rows (so a GEMM with beta=1 adds them in one go)
__global__ void fill_bias_kernel(bf16* __restrict__ x, const bf16* __restrict__ b, int N, int D) {
  size_t i = cu::global_tid();
  if (i >= (size_t)N * D) return;
  x[i] = b[i % D];
}
// qkv [N, 3*H*hd] -> q,k,v each [H, N, hd] (head-major) + 2D rope (q,k): reference apply_rotary: x1,x2 = halves, cos/sin [N, hd/2]
__global__ void split_rope_kernel(const bf16* __restrict__ qkv, int N, int H, int hd, const float* __restrict__ cs,
                                  const float* __restrict__ sn, bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v) {
  size_t i = cu::global_tid();
  const int half = hd / 2;
  if (i >= (size_t)N * H * half) return;
  int n = i / ((size_t)H * half), rem = i % ((size_t)H * half), h = rem / half, d = rem % half;
  const bf16* row = qkv + (size_t)n * 3 * H * hd;
  float c = cs[(size_t)n * half + d], s = sn[(size_t)n * half + d];
  // q
  {
    float x1 = bf2f(row[h * hd + d]), x2 = bf2f(row[h * hd + half + d]);
    q[((size_t)h * N + n) * hd + d] = f2bf(x1 * c - x2 * s);
    q[((size_t)h * N + n) * hd + half + d] = f2bf(x2 * c + x1 * s);
  }
  {
    float x1 = bf2f(row[H * hd + h * hd + d]), x2 = bf2f(row[H * hd + h * hd + half + d]);
    k[((size_t)h * N + n) * hd + d] = f2bf(x1 * c - x2 * s);
    k[((size_t)h * N + n) * hd + half + d] = f2bf(x2 * c + x1 * s);
  }
  v[((size_t)h * N + n) * hd + d] = row[2 * H * hd + h * hd + d];
  v[((size_t)h * N + n) * hd + half + d] = row[2 * H * hd + h * hd + half + d];
}
// softmax over last dim (fp32 in, bf16 out), with scale. rows = H*N, cols = N
__global__ void softmax_rows_kernel(const float* __restrict__ s, int cols, int ld, float scale, bf16* __restrict__ P) {
  __shared__ float red[32];
  const size_t r = blockIdx.x;
  const float* p = s + r * ld;
  bf16* q = P + r * ld;
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += 256) mx = fmaxf(mx, p[i] * scale);
  mx = cu::warp_allreduce<cu::Max>(mx);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : -INFINITY;
    t = cu::warp_allreduce<cu::Max>(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  mx = *red;
  __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < cols; i += 256) sum += expf(p[i] * scale - mx);
  sum = cu::warp_allreduce<cu::Add>(sum);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    t = cu::warp_allreduce<cu::Add>(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  sum = red[0];
  for (int i = threadIdx.x; i < cols; i += 256) q[i] = f2bf(expf(p[i] * scale - mx) / sum);
}
// o [H, N, hd] → [N, H·hd]
__global__ void merge_heads_kernel(const bf16* __restrict__ o, int N, int H, int hd, bf16* __restrict__ out) {
  size_t i = cu::global_tid();
  if (i >= (size_t)N * H * hd) return;
  int n = i / ((size_t)H * hd), rem = i % ((size_t)H * hd), h = rem / hd, d = rem % hd;
  out[i] = o[((size_t)h * N + n) * hd + d];
}
__global__ void residual_add_kernel(bf16* __restrict__ x, const bf16* __restrict__ y, size_t n) {
  size_t i = cu::global_tid();
  if (i < n) x[i] = f2bf(bf2f(x[i]) + bf2f(y[i]));
}
// mlp: w1 output [N, 2*I] = (gate|up) -> silu(gate)*up [N, I]
__global__ void silu_gate_kernel(const bf16* __restrict__ gu, int N, int I, bf16* __restrict__ out) {
  size_t i = cu::global_tid();
  if (i >= (size_t)N * I) return;
  int n = i / I, d = i % I;
  float g = bf2f(gu[(size_t)n * 2 * I + d]), u = bf2f(gu[(size_t)n * 2 * I + I + d]);
  float sg = bf2f(f2bf(g / (1.f + expf(-g))));  // torch: the silu output is rounded to bf16 before the multiply
  out[i] = f2bf(sg * u);
}
__global__ void gelu_kernel(bf16* __restrict__ x, size_t n) {
  size_t i = cu::global_tid();
  if (i >= n) return;
  float v = bf2f(x[i]);
  x[i] = f2bf(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));  // torch F.gelu default (erf)
}
// aligner unfold: x [n_h*n_w, vdim] (row-major grid) -> out [ceil(n_h/r)*ceil(n_w/r), vdim*r*r]. F.unfold channel order = (c, kh, kw).
__global__ void unfold_kernel(const bf16* __restrict__ x, int n_h, int n_w, int vdim, int r, int oh, int ow, bf16* __restrict__ out) {
  size_t i = cu::global_tid();
  const size_t cols = (size_t)vdim * r * r;
  if (i >= (size_t)oh * ow * cols) return;
  size_t o = i / cols;
  int col = (int)(i % cols);
  int oy = (int)(o / ow), ox = (int)(o % ow);
  int c = col / (r * r), kh = (col / r) % r, kw = col % r;
  int y = oy * r + kh, xx = ox * r + kw;
  out[i] = (y < n_h && xx < n_w) ? x[((size_t)y * n_w + xx) * vdim + c] : f2bf(0.f);
}
inline int g1(size_t n) { return (int)((n + 255) / 256); }
}  // namespace
}  // namespace k

Vision::Vision(Model& model, Blas& blas, cudaStream_t st, int max_patches) : model_(model), blas_(blas), st_(st), max_patches_(max_patches) {
  load();
  const Config& c = model_.cfg();
  const int vd = c.vision_dim, N = max_patches, I = c.vision_inter, H = c.vision_heads;
  x.alloc((size_t)N * vd * 2); xn.alloc((size_t)N * vd * 2); qkv.alloc((size_t)N * 3 * vd * 2);
  q.alloc((size_t)N * vd * 2); k.alloc((size_t)N * vd * 2); v.alloc((size_t)N * vd * 2);
  attn.alloc((size_t)N * vd * 2); o.alloc((size_t)N * vd * 2);
  mlp.alloc((size_t)N * 2 * I * 2);
  act.alloc((size_t)N * I * 2);
  cos_.alloc((size_t)N * (vd / H / 2) * 4); sin_.alloc((size_t)N * (vd / H / 2) * 4);
  scores.alloc((size_t)N * align_up(N, 8) * 4);  // fp32 scores, one head at a time (all heads at once would be 9216^2 x 16 x 4 B = 5.4 GB), row stride a multiple of 8
  probs.alloc((size_t)N * align_up(N, 8) * 2);
  const int r = c.vision_downsample;
  const size_t max_rows = (size_t)N / (r * r) + 2 * (size_t)(sqrt((double)N) + 1) / r + 4;
  unfold.alloc(max_rows * vd * r * r * 2);
  al_hidden.alloc(max_rows * c.dim * 2);
}

void Vision::load() {
  Checkpoint& ck = model_.ckpt();
  const Config& c = model_.cfg();
  auto raw = [&](const std::string& n) {
    const TensorInfo& t = ck.get(n);
    DevBuf b(t.nbytes);
    CUDA_CHECK(cudaMemcpy(b.p, t.data, t.nbytes, cudaMemcpyHostToDevice));
    return b;
  };
  auto f32 = [&](const std::string& n) {
    const TensorInfo& t = ck.get(n);
    std::vector<float> tmp(t.numel());
    if (t.dtype == "F32") memcpy(tmp.data(), t.data, t.nbytes);
    else {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(t.data);
      for (size_t i = 0; i < tmp.size(); ++i) { uint32_t u = (uint32_t)p[i] << 16; memcpy(&tmp[i], &u, 4); }
    }
    DevBuf b(tmp.size() * 4);
    CUDA_CHECK(cudaMemcpy(b.p, tmp.data(), tmp.size() * 4, cudaMemcpyHostToDevice));
    return b;
  };
  w_.patch_w = raw("vision.patch_embed.proj.weight");
  w_.patch_b = raw("vision.patch_embed.proj.bias");
  w_.norm = f32("vision.norm.weight");
  w_.layers.resize(c.vision_layers);
  for (int l = 0; l < c.vision_layers; ++l) {
    std::string p = "vision.blocks." + std::to_string(l);
    auto& L = w_.layers[l];
    L.norm1 = f32(p + ".norm1.weight");
    L.norm2 = f32(p + ".norm2.weight");
    L.wqkv = raw(p + ".attn.wqkv.weight"); L.bqkv = raw(p + ".attn.wqkv.bias");
    L.wo = raw(p + ".attn.wo.weight"); L.bo = raw(p + ".attn.wo.bias");
    L.w1 = raw(p + ".mlp.w1.weight"); L.w2 = raw(p + ".mlp.w2.weight");
  }
  w_.al_w1 = raw("aligner.w1.weight"); w_.al_b1 = raw("aligner.w1.bias");
  w_.al_w2 = raw("aligner.w2.weight"); w_.al_b2 = raw("aligner.w2.bias");
  w_.image_start = raw("image_start"); w_.image_end = raw("image_end"); w_.image_newline = raw("image_newline");
}

void Vision::encode(const bf16* patches, int n_vit_h, int n_vit_w, bf16* out) {
  const Config& c = model_.cfg();
  const int N = n_vit_h * n_vit_w, vd = c.vision_dim, H = c.vision_heads, hd = vd / H, I = c.vision_inter;
  HIVE_CHECK(N <= max_patches_, "too many patches");
  const int rope_dim = hd / 2;  // reference: rope_dim = vision_dim // n_heads // 2 (= hd/2), cos/sin length = rope_dim (2D: rope_dim/2 each for h and w)
  // cos/sin [N, rope_dim]: freqs = stack(hpos, wpos)[..., None] * inv_freq(rope_dim/2) → flatten → [N, rope_dim]
  {
    std::vector<float> cs((size_t)N * rope_dim), sn((size_t)N * rope_dim);
    const int half = rope_dim / 2;
    std::vector<float> inv(half);
    for (int i = 0; i < half; ++i) inv[i] = 1.0f / powf(c.vision_rope_theta, (float)(2 * i) / (float)rope_dim);
    for (int y = 0; y < n_vit_h; ++y)
      for (int xx = 0; xx < n_vit_w; ++xx) {
        int n = y * n_vit_w + xx;
        for (int i = 0; i < half; ++i) {
          float fh = (float)y * inv[i], fw = (float)xx * inv[i];
          cs[(size_t)n * rope_dim + i] = cosf(fh); sn[(size_t)n * rope_dim + i] = sinf(fh);
          cs[(size_t)n * rope_dim + half + i] = cosf(fw); sn[(size_t)n * rope_dim + half + i] = sinf(fw);
        }
      }
    CUDA_CHECK(cudaMemcpyAsync(cos_.p, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(sin_.p, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice, st_));
  }
  const int pp = 3 * c.vision_patch * c.vision_patch;
  k::fill_bias_kernel<<<k::g1((size_t)N * vd), 256, 0, st_>>>(x.as<bf16>(), w_.patch_b.as<bf16>(), N, vd);
  blas_.gemm_bf16_acc(patches, w_.patch_w.as<bf16>(), x.as<bf16>(), N, vd, pp);
  vdump(st_, "vit_patch.bf16", x.p, (size_t)N * vd * 2);
  const float scale = 1.0f / sqrtf((float)hd);
  for (int l = 0; l < c.vision_layers; ++l) {
    auto& L = w_.layers[l];
    k::vrms_kernel<<<N, 256, 0, st_>>>(x.as<bf16>(), L.norm1.as<float>(), 1e-6f, vd, xn.as<bf16>());
    k::fill_bias_kernel<<<k::g1((size_t)N * 3 * vd), 256, 0, st_>>>(qkv.as<bf16>(), L.bqkv.as<bf16>(), N, 3 * vd);
    blas_.gemm_bf16_acc(xn.as<bf16>(), L.wqkv.as<bf16>(), qkv.as<bf16>(), N, 3 * vd, vd);
    k::split_rope_kernel<<<k::g1((size_t)N * H * (hd / 2)), 256, 0, st_>>>(qkv.as<bf16>(), N, H, hd, cos_.as<float>(), sin_.as<float>(),
                                                                          q.as<bf16>(), k.as<bf16>(), v.as<bf16>());
    const int Np = (int)align_up(N, 8);  // cuBLAS bf16 leading-dimension alignment
    for (int h = 0; h < H; ++h) {  // per head: scores = q[h]*k[h]^T [N,N] -> softmax -> o[h] = P*v[h]
      blas_.gemm_bf16_f32out_ld(q.as<bf16>() + (size_t)h * N * hd, hd, k.as<bf16>() + (size_t)h * N * hd, hd, scores.as<float>(), Np, N, N, hd);
      k::softmax_rows_kernel<<<N, 256, 0, st_>>>(scores.as<float>(), N, Np, scale, probs.as<bf16>());
      blas_.gemm_bf16_nn_ld(probs.as<bf16>(), Np, v.as<bf16>() + (size_t)h * N * hd, hd, o.as<bf16>() + (size_t)h * N * hd, hd, N, hd, N);
    }
    k::merge_heads_kernel<<<k::g1((size_t)N * vd), 256, 0, st_>>>(o.as<bf16>(), N, H, hd, attn.as<bf16>());
    k::fill_bias_kernel<<<k::g1((size_t)N * vd), 256, 0, st_>>>(xn.as<bf16>(), L.bo.as<bf16>(), N, vd);
    blas_.gemm_bf16_acc(attn.as<bf16>(), L.wo.as<bf16>(), xn.as<bf16>(), N, vd, vd);
    k::residual_add_kernel<<<k::g1((size_t)N * vd), 256, 0, st_>>>(x.as<bf16>(), xn.as<bf16>(), (size_t)N * vd);
    k::vrms_kernel<<<N, 256, 0, st_>>>(x.as<bf16>(), L.norm2.as<float>(), 1e-6f, vd, xn.as<bf16>());
    blas_.gemm_bf16(xn.as<bf16>(), L.w1.as<bf16>(), mlp.as<bf16>(), N, 2 * I, vd);
    k::silu_gate_kernel<<<k::g1((size_t)N * I), 256, 0, st_>>>(mlp.as<bf16>(), N, I, act.as<bf16>());
    blas_.gemm_bf16(act.as<bf16>(), L.w2.as<bf16>(), xn.as<bf16>(), N, vd, I);
    k::residual_add_kernel<<<k::g1((size_t)N * vd), 256, 0, st_>>>(x.as<bf16>(), xn.as<bf16>(), (size_t)N * vd);
    if (l == 0 || l == 1 || l == 7 || l == 15 || l == 31) vdump(st_, ("vit_blk" + std::to_string(l) + ".bf16").c_str(), x.p, (size_t)N * vd * 2);
  }
  k::vrms_kernel<<<N, 256, 0, st_>>>(x.as<bf16>(), w_.norm.as<float>(), 1e-6f, vd, xn.as<bf16>());
  vdump(st_, "vit_norm.bf16", xn.p, (size_t)N * vd * 2);
  // aligner
  const int r = c.vision_downsample;
  const int oh = (n_vit_h + r - 1) / r, ow = (n_vit_w + r - 1) / r;
  const int cols = vd * r * r;
  k::unfold_kernel<<<k::g1((size_t)oh * ow * cols), 256, 0, st_>>>(xn.as<bf16>(), n_vit_h, n_vit_w, vd, r, oh, ow, unfold.as<bf16>());
  k::fill_bias_kernel<<<k::g1((size_t)oh * ow * c.dim), 256, 0, st_>>>(al_hidden.as<bf16>(), w_.al_b1.as<bf16>(), oh * ow, c.dim);
  blas_.gemm_bf16_acc(unfold.as<bf16>(), w_.al_w1.as<bf16>(), al_hidden.as<bf16>(), oh * ow, c.dim, cols);
  k::gelu_kernel<<<k::g1((size_t)oh * ow * c.dim), 256, 0, st_>>>(al_hidden.as<bf16>(), (size_t)oh * ow * c.dim);
  k::fill_bias_kernel<<<k::g1((size_t)oh * ow * c.dim), 256, 0, st_>>>(out, w_.al_b2.as<bf16>(), oh * ow, c.dim);
  blas_.gemm_bf16_acc(al_hidden.as<bf16>(), w_.al_w2.as<bf16>(), out, oh * ow, c.dim, c.dim);
}

}  // namespace hive
