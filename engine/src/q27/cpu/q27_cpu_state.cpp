// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B CPU engine — CpuSeq serialization for hived's session images. Order: header, tokens, rope, per linear layer conv and state
//   (empty for layers not on the CPU), per attention layer k / v (rows below pos), the MTP caches (rows below mtp_written), h_last.
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "hive/q27/q27_cpu.h"

namespace hive::q27::cpu {

namespace {
template <class T>
std::vector<uint8_t> bytes_of(const T* p, size_t n) {
  std::vector<uint8_t> b(n * sizeof(T));
  if (n) memcpy(b.data(), p, b.size());
  return b;
}
template <class T>
void restore(std::vector<T>& v, const std::vector<uint8_t>& b) {
  if (b.size() % sizeof(T)) throw std::runtime_error("q27 cpu: bad state image");
  v.resize(b.size() / sizeof(T));
  if (!b.empty()) memcpy(v.data(), b.data(), b.size());
}
}  // namespace

std::vector<std::vector<uint8_t>> CpuSeq::save_bytes() const {
  std::vector<std::vector<uint8_t>> out;
  // row width of the K / V caches: taken from the caches themselves (size / rows is not known here), stored in the header
  const int64_t kvw = 1024;
  const int64_t hdr[9] = {pos, cap, rope_delta, h_last_valid ? 1 : 0, mtp_written, broken ? 1 : 0, (int64_t)conv.size(), (int64_t)attn.size(), kvw};
  out.push_back(bytes_of(hdr, 9));
  out.push_back(bytes_of(tokens.data(), tokens.size()));
  out.push_back(bytes_of(rope.data(), std::min(rope.size(), (size_t)pos * 3)));
  for (size_t i = 0; i < conv.size(); ++i) { out.push_back(bytes_of(conv[i].data(), conv[i].size())); out.push_back(bytes_of(state[i].data(), state[i].size())); }
  auto rows = [&](const std::vector<uint16_t>& v, int64_t n) { return std::min(v.size(), (size_t)std::max<int64_t>(0, n) * kvw); };
  for (const Attn& a : attn) { out.push_back(bytes_of(a.k.data(), rows(a.k, pos))); out.push_back(bytes_of(a.v.data(), rows(a.v, pos))); }
  out.push_back(bytes_of(mtp.k.data(), rows(mtp.k, mtp_written)));
  out.push_back(bytes_of(mtp.v.data(), rows(mtp.v, mtp_written)));
  out.push_back(bytes_of(h_last.data(), h_last.size()));
  return out;
}

void CpuSeq::load_bytes(const std::vector<std::vector<uint8_t>>& b) {
  if (b.empty() || b[0].size() != 72) throw std::runtime_error("q27 cpu: bad state image");
  int64_t hdr[9];
  memcpy(hdr, b[0].data(), 72);
  const size_t nl = (size_t)hdr[6], na = (size_t)hdr[7];
  if (b.size() != 3 + 2 * nl + 2 * na + 2 + 1) throw std::runtime_error("q27 cpu: state image does not match the model");
  pos = hdr[0]; cap = hdr[1]; rope_delta = hdr[2]; h_last_valid = hdr[3] != 0; mtp_written = hdr[4]; broken = hdr[5] != 0;
  size_t k = 1;
  restore(tokens, b[k++]);
  restore(rope, b[k++]);
  conv.resize(nl); state.resize(nl);
  for (size_t i = 0; i < nl; ++i) { restore(conv[i], b[k++]); restore(state[i], b[k++]); }
  attn.resize(na);
  for (Attn& a : attn) { restore(a.k, b[k++]); restore(a.v, b[k++]); }
  restore(mtp.k, b[k++]); restore(mtp.v, b[k++]);
  restore(h_last, b[k++]);
}

}  // namespace hive::q27::cpu
