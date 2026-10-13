// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash CPU engine — CpuSeq serialization for hived's session images. Order: header, tokens, per KDA layer conv and state, per DSA layer
//   c / kI / gs (rows below pos) and pooled (pools below pos/4 + 1), the NextN cache (rows below mtp_written), h_last.
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "hive/glm/glm_cpu.h"

namespace hive::glm::cpu {

namespace {
template <class T>
std::vector<uint8_t> bytes_of(const T* p, size_t n) {
  std::vector<uint8_t> b(n * sizeof(T));
  if (n) std::memcpy(b.data(), p, b.size());
  return b;
}
template <class T>
void restore(std::vector<T>& v, const std::vector<uint8_t>& b) {
  if (b.size() % sizeof(T)) throw std::runtime_error("glm cpu: bad state image");
  v.resize(b.size() / sizeof(T));
  if (!b.empty()) std::memcpy(v.data(), b.data(), b.size());
}
size_t rows_n(const std::vector<uint16_t>& v, int64_t rows, size_t width) { return std::min(v.size(), (size_t)std::max<int64_t>(0, rows) * width); }
}  // namespace

std::vector<std::vector<uint8_t>> CpuSeq::save_bytes() const {
  std::vector<std::vector<uint8_t>> out;
  const int64_t hdr[8] = {pos, cap, h_last_valid ? 1 : 0, mtp_written, broken ? 1 : 0, (int64_t)conv.size(), (int64_t)dsa.size(), 0};
  out.push_back(bytes_of(hdr, 8));
  out.push_back(bytes_of(tokens.data(), tokens.size()));
  for (size_t i = 0; i < conv.size(); ++i) { out.push_back(bytes_of(conv[i].data(), conv[i].size())); out.push_back(bytes_of(state[i].data(), state[i].size())); }
  auto put = [&](const Dsa& a, int64_t n) {
    out.push_back(bytes_of(a.c.data(), rows_n(a.c, n, 512)));
    out.push_back(bytes_of(a.kI.data(), rows_n(a.kI, n, 128)));
    out.push_back(bytes_of(a.gs.data(), rows_n(a.gs, n, 128)));
    out.push_back(bytes_of(a.pooled.data(), rows_n(a.pooled, n / 4 + 1, 128)));
  };
  for (const Dsa& a : dsa) put(a, pos);
  put(mtp, mtp_written);
  out.push_back(bytes_of(h_last.data(), h_last.size()));
  return out;
}

void CpuSeq::load_bytes(const std::vector<std::vector<uint8_t>>& b) {
  if (b.empty() || b[0].size() != 64) throw std::runtime_error("glm cpu: bad state image");
  int64_t hdr[8];
  std::memcpy(hdr, b[0].data(), 64);
  const size_t nk = (size_t)hdr[5], nd = (size_t)hdr[6];
  if (b.size() != 2 + 2 * nk + 4 * nd + 4 + 1) throw std::runtime_error("glm cpu: state image does not match the model");
  pos = hdr[0]; cap = hdr[1]; h_last_valid = hdr[2] != 0; mtp_written = hdr[3]; broken = hdr[4] != 0;
  size_t k = 1;
  restore(tokens, b[k++]);
  conv.resize(nk); state.resize(nk);
  for (size_t i = 0; i < nk; ++i) { restore(conv[i], b[k++]); restore(state[i], b[k++]); }
  dsa.resize(nd);
  auto get = [&](Dsa& a) { restore(a.c, b[k++]); restore(a.kI, b[k++]); restore(a.gs, b[k++]); restore(a.pooled, b[k++]); };
  for (Dsa& a : dsa) get(a);
  get(mtp);
  restore(h_last, b[k++]);
}

}  // namespace hive::glm::cpu
