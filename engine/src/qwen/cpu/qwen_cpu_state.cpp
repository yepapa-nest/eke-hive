// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next CPU engine — CpuSeq serialization for hived's session images. Order: header, tokens, rope, then per linear layer conv and
//   state, the PLE state, per attention layer k / v / ik / gk (rows below pos), the MTP caches (rows below mtp_written), h_last.
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "hive/qwen/qwen_cpu.h"

namespace hive::qwen::cpu {

namespace {
template <class T>
std::vector<uint8_t> bytes_of(const T* p, size_t n) {
  std::vector<uint8_t> b(n * sizeof(T));
  if (n) memcpy(b.data(), p, b.size());
  return b;
}
template <class T>
void restore(std::vector<T>& v, const std::vector<uint8_t>& b, size_t min_n = 0) {
  if (b.size() % sizeof(T)) throw std::runtime_error("qwen cpu: bad state image");
  v.resize(std::max(min_n, b.size() / sizeof(T)));
  if (!b.empty()) memcpy(v.data(), b.data(), b.size());
}
size_t rows_n(const std::vector<uint16_t>& v, int64_t rows, size_t width) { return std::min(v.size(), (size_t)std::max<int64_t>(0, rows) * width); }
}  // namespace

std::vector<std::vector<uint8_t>> CpuSeq::save_bytes() const {
  std::vector<std::vector<uint8_t>> out;
  const int64_t hdr[8] = {pos, cap, rope_delta, h_last_valid ? 1 : 0, mtp_written, broken ? 1 : 0, (int64_t)conv.size(), (int64_t)attn.size()};
  out.push_back(bytes_of(hdr, 8));
  out.push_back(bytes_of(tokens.data(), tokens.size()));
  out.push_back(bytes_of(rope.data(), std::min(rope.size(), (size_t)pos * 3)));
  for (size_t i = 0; i < conv.size(); ++i) { out.push_back(bytes_of(conv[i].data(), conv[i].size())); out.push_back(bytes_of(state[i].data(), state[i].size())); }
  out.push_back(bytes_of(ple_state.data(), ple_state.size()));
  // cache widths per row (FN): k / v 2·256, ik 128, gk 128
  constexpr size_t kvw = 512, idw = 128;
  auto put_attn = [&](const Attn& a, int64_t n) {
    out.push_back(bytes_of(a.k.data(), rows_n(a.k, n, kvw)));
    out.push_back(bytes_of(a.v.data(), rows_n(a.v, n, kvw)));
    out.push_back(bytes_of(a.ik.data(), rows_n(a.ik, n, idw)));
    out.push_back(bytes_of(a.gk.data(), rows_n(a.gk, n / 4 + 1, idw)));
  };
  for (const Attn& a : attn) put_attn(a, pos);
  put_attn(mtp, mtp_written);
  out.push_back(bytes_of(h_last.data(), h_last.size()));
  return out;
}

void CpuSeq::load_bytes(const std::vector<std::vector<uint8_t>>& b) {
  if (b.empty() || b[0].size() != 64) throw std::runtime_error("qwen cpu: bad state image");
  int64_t hdr[8];
  memcpy(hdr, b[0].data(), 64);
  const size_t nl = (size_t)hdr[6], na = (size_t)hdr[7];
  if (b.size() != 3 + 2 * nl + 1 + 4 * na + 4 + 1) throw std::runtime_error("qwen cpu: state image does not match the model");
  pos = hdr[0]; cap = hdr[1]; rope_delta = hdr[2]; h_last_valid = hdr[3] != 0; mtp_written = hdr[4]; broken = hdr[5] != 0;
  size_t k = 1;
  restore(tokens, b[k++]);
  restore(rope, b[k++]);
  conv.resize(nl); state.resize(nl);
  for (size_t i = 0; i < nl; ++i) { restore(conv[i], b[k++]); restore(state[i], b[k++]); }
  restore(ple_state, b[k++]);
  attn.resize(na);
  auto get_attn = [&](Attn& a) { restore(a.k, b[k++]); restore(a.v, b[k++]); restore(a.ik, b[k++]); restore(a.gk, b[k++]); a.gkt_n = 0; };
  for (Attn& a : attn) get_attn(a);
  get_attn(mtp);
  restore(h_last, b[k++]);
}

}  // namespace hive::qwen::cpu
