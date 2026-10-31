// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Checkpoint weight formats — see hive/ckpt_quant.h.
#include "hive/ckpt_quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace hive::ckq {

float e4m3_f32(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  if (e == 15 && m == 7) return NAN;
  const float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.f + m / 8.f, e - 7);
  return s ? -v : v;
}
float e2m1_f32(uint8_t n) {
  static const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (n & 8) ? -t[n & 7] : t[n & 7];
}
float ue8m0_f32(uint8_t b) { return b == 255 ? NAN : std::ldexp(1.f, (int)b - 127); }
uint16_t f32_bf16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

bool csf_decode(const uint8_t* fixed, int64_t fixed_rows, int64_t fixed_cols, const uint32_t* exc, size_t n_exc, int R, int C, uint8_t* out,
                std::string* err) {
  auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
  if (R <= 0 || C <= 0 || R % 16) return fail("CSF: rows not a multiple of 16");
  if (fixed_rows != R / 16 || fixed_cols != 16 + 8 * (int64_t)C)
    return fail("CSF: fixed shape [" + std::to_string(fixed_rows) + ", " + std::to_string(fixed_cols) + "] for scales [" + std::to_string(R) + ", " +
                std::to_string(C) + "]");
  const int64_t RC = (int64_t)R * C;
  if (RC > (int64_t)1 << 24) return fail("CSF: more scales than a 24-bit index addresses");
  for (int g = 0; g < R / 16; ++g) {
    const uint8_t* f = fixed + (size_t)g * fixed_cols;
    const uint8_t* body = f + 16;
    for (int r = 0; r < 16; ++r) {
      const int base = f[r];
      uint8_t* o = out + (size_t)(g * 16 + r) * C;
      const size_t n0 = (size_t)r * C;   // nibble position of (r, 0) inside the group
      for (int c = 0; c < C; ++c) {
        const size_t n = n0 + c;
        const int d = (body[n >> 1] >> ((n & 1) * 4)) & 15;
        o[c] = (uint8_t)(base + d);   // checked below (after the exceptions)
        if (base + d > 0x7E) o[c] = 0xFF;   // marked: invalid unless an exception replaces it
      }
    }
  }
  for (size_t i = 0; i < n_exc; ++i) {
    const uint32_t w = exc[i];
    const int64_t idx = w & 0xFFFFFF;
    const uint8_t v = (uint8_t)(w >> 24);
    if (idx >= RC) return fail("CSF: exception index " + std::to_string(idx) + " outside " + std::to_string(RC));
    const int r = (int)(idx / C), c = (int)(idx % C), g = r / 16;
    const size_t n = (size_t)(r % 16) * C + c;
    const int d = (fixed[(size_t)g * fixed_cols + 16 + (n >> 1)] >> ((n & 1) * 4)) & 15;
    if (d != 0) return fail("CSF: non-zero delta at exception index " + std::to_string(idx));
    if (v > 0x7E) return fail("CSF: exception value " + std::to_string(v) + " is not a finite non-negative E4M3 scale");
    out[idx] = v;
  }
  for (int64_t i = 0; i < RC; ++i)
    if (out[i] > 0x7E) return fail("CSF: scale " + std::to_string(i) + " (base + delta) is not a finite non-negative E4M3 value");
  return true;
}

bool has_csf(Checkpoint& ck, const std::string& base) {
  return !ck.has(base + "weight_scale") && ck.has(base + "weight_scale.nvfp4_csf_fixed") && ck.has(base + "weight_scale.nvfp4_csf_exceptions");
}

void nvfp4_scales(Checkpoint& ck, const std::string& base, int R, int C, uint8_t* out) {
  if (ck.has(base + "weight_scale")) {
    const TensorInfo& t = ck.get(base + "weight_scale");
    if (t.dtype != "F8_E4M3" || t.numel() != (int64_t)R * C) throw std::runtime_error("NVFP4 scale shape / dtype: " + base);
    std::memcpy(out, t.data, t.nbytes);
    return;
  }
  const TensorInfo& f = ck.get(base + "weight_scale.nvfp4_csf_fixed");
  const TensorInfo& e = ck.get(base + "weight_scale.nvfp4_csf_exceptions");
  if (f.dtype != "U8" || f.shape.size() != 2 || e.dtype != "U32" || e.shape.size() != 1) throw std::runtime_error("CSF dtype / rank: " + base);
  // the U32 words are little-endian in the file and unaligned in the mmap (8-byte header + JSON): copy them out
  std::vector<uint32_t> ex((size_t)e.numel());
  std::memcpy(ex.data(), e.data, e.nbytes);
  std::string err;
  if (!csf_decode(f.data, f.shape[0], f.shape[1], ex.data(), ex.size(), R, C, out, &err)) throw std::runtime_error(err + " (" + base + ")");
}

Dense dense_format(Checkpoint& ck, const std::string& base) {
  const TensorInfo& w = ck.get(base + "weight");
  if (w.dtype == "BF16") return Dense::BF16;
  if (w.dtype == "F8_E4M3" && ck.has(base + "weight_scale") && ck.get(base + "weight_scale").dtype == "U8") return Dense::MXFP8;
  if (w.dtype == "U8" && ck.has(base + "weight_scale") && ck.get(base + "weight_scale").dtype == "F8_E4M3" && ck.has(base + "weight_scale_2"))
    return Dense::NVFP4;
  return Dense::Other;
}

void dense_shape(Checkpoint& ck, const std::string& base, int64_t& N, int64_t& K) {
  const TensorInfo& w = ck.get(base + "weight");
  if (w.shape.size() != 2) throw std::runtime_error("dense weight must be 2-D: " + base);
  N = w.shape[0];
  K = dense_format(ck, base) == Dense::NVFP4 ? w.shape[1] * 2 : w.shape[1];
}

void mxfp8_parts(Checkpoint& ck, const std::string& base, const uint8_t*& q, const uint8_t*& s, int64_t& N, int64_t& K) {
  const TensorInfo& w = ck.get(base + "weight");
  const TensorInfo& sc = ck.get(base + "weight_scale");
  if (w.dtype != "F8_E4M3" || sc.dtype != "U8" || w.shape.size() != 2 || sc.shape.size() != 2 || w.shape[1] % 32 ||
      sc.shape[0] != w.shape[0] || sc.shape[1] != w.shape[1] / 32)
    throw std::runtime_error("MXFP8: expected F8_E4M3 [N, K] + U8 [N, K/32]: " + base);
  q = w.data; s = sc.data; N = w.shape[0]; K = w.shape[1];
}

std::vector<uint16_t> expand_bf16(Checkpoint& ck, const std::string& base, int threads) {
  const Dense f = dense_format(ck, base);
  const TensorInfo& w = ck.get(base + "weight");
  int64_t N = 0, K = 0;
  dense_shape(ck, base, N, K);
  std::vector<uint16_t> out((size_t)N * K);
  if (f == Dense::BF16) { std::memcpy(out.data(), w.data, w.nbytes); return out; }
  if (f == Dense::Other) throw std::runtime_error("unsupported weight format " + w.dtype + ": " + base);
  float lut8[256];
  for (int b = 0; b < 256; ++b) lut8[b] = e4m3_f32((uint8_t)b);
  auto run = [&](int64_t r0, int64_t r1) {
    if (f == Dense::MXFP8) {
      const TensorInfo& s = ck.get(base + "weight_scale");
      if (s.shape.size() != 2 || s.shape[0] != N || s.shape[1] * 32 != K) throw std::runtime_error("MXFP8 scale shape: " + base);
      for (int64_t r = r0; r < r1; ++r)
        for (int64_t k = 0; k < K; ++k)
          out[(size_t)r * K + k] = f32_bf16(lut8[w.data[(size_t)r * K + k]] * ue8m0_f32(s.data[(size_t)r * (K / 32) + k / 32]));
    } else {
      const TensorInfo& s = ck.get(base + "weight_scale");
      const TensorInfo& g = ck.get(base + "weight_scale_2");
      if (s.shape.size() != 2 || s.shape[0] != N || s.shape[1] * 16 != K || g.dtype != "F32" || g.numel() != 1)
        throw std::runtime_error("NVFP4 scale shape: " + base);
      float gs;
      std::memcpy(&gs, g.data, 4);
      for (int64_t r = r0; r < r1; ++r)
        for (int64_t k = 0; k < K; ++k) {
          const uint8_t byte = w.data[(size_t)r * (K / 2) + k / 2];
          const float v = e2m1_f32((k & 1) ? byte >> 4 : byte & 15);
          out[(size_t)r * K + k] = f32_bf16(v * (lut8[s.data[(size_t)r * (K / 16) + k / 16]] * gs));
        }
    }
  };
  const int nt = (int)std::max<int64_t>(1, std::min<int64_t>(threads, N / 64));
  if (nt == 1) { run(0, N); return out; }
  std::vector<std::thread> th;
  std::vector<std::string> errs(nt);
  for (int t = 0; t < nt; ++t)
    th.emplace_back([&, t] { try { run(N * t / nt, N * (t + 1) / nt); } catch (const std::exception& e) { errs[t] = e.what(); } });
  for (auto& x : th) x.join();
  for (auto& e : errs) if (!e.empty()) throw std::runtime_error(e);
  return out;
}

std::vector<float> vec_f32(Checkpoint& ck, const std::string& name) {
  const TensorInfo& t = ck.get(name);
  std::vector<float> v((size_t)t.numel());
  if (t.dtype == "F32") std::memcpy(v.data(), t.data, t.nbytes);
  else if (t.dtype == "BF16") {
    for (size_t i = 0; i < v.size(); ++i) {
      uint16_t h;
      std::memcpy(&h, t.data + 2 * i, 2);
      const uint32_t u = (uint32_t)h << 16;
      std::memcpy(&v[i], &u, 4);
    }
  } else throw std::runtime_error("vector must be F32 or BF16: " + name + " (" + t.dtype + ")");
  return v;
}

}  // namespace hive::ckq
