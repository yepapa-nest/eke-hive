// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Persistent session store (HIVE_SESSION_STORE): prompt checkpoints and boundary snapshots written to disk as they are made, so a
//   conversation survives an engine restart or a model switch.
//
// Problem (measured 2026-10-15): every restart or main-model switch dropped all conversation state (it lived in RAM only), so the first
//   request of a long room read the whole prompt again — a 379,914-token GLM room took 138 s. Its state is ~7.5 GB; the reference
//   machine's NVMe writes 3.6 GB/s and reads 6.5 GB/s. The engine is stopped with `docker rm -f` (no time to save at stop), so images are
//   written when they are made (write-through, on a writer thread) and read back only when a request needs them (lazy).
//
// Format (one directory per layout key = family + store format + checkpoint config + image shape; anything else is ignored):
//   packs/<id>.pack   the bytes written by one image save — only segments not already on disk (images share the unchanged prefix of the
//                     previous image: HostImageBuffer segments, HIVE_CKPT_DELTA), so a follow-up turn writes its increment only
//   metas/<name>.json one image: kind (ckpt | snap), session id or prefix key, position, recency, image signatures, upper_skipped, the
//                     image's fields in SeqImage::fields order — numbers inline, byte ranges as [pack, offset, length, hash]
//   metas/canary-*.json + its pack: the canary (put_canary) — the state of a fixed token sequence and its logits, written by the build that
//                     created the directory, never evicted. The layout key has no build id (measured 2026-10-18/19: five DeepSeek rebuilds in
//                     a day each dropped every stored conversation, so every room read its whole prompt once more after each deploy — up to
//                     224K tokens, 40-50 s); a new build instead resumes the canary and compares its logits with its own (hived store_canary)
//                     and drops the directory when they differ.
//   Every file is written to a temporary name, fsync'ed and renamed; a range is read back only if its length and hash match.
// Images are immutable once saved (segments are never rewritten), so the writer holds references and copies nothing.
#pragma once
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "hive/host_image.h"
#include "hive/image_reaper.h"
#include "nlohmann/json.hpp"

namespace hive::store {

using json = nlohmann::json;

inline uint64_t hash_bytes(const uint8_t* p, size_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ n;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w; std::memcpy(&w, p + i, 8);
    h ^= w; h *= 0xff51afd7ed558ccdull; h ^= h >> 32;
  }
  for (; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
  h ^= h >> 29; h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 32;
  return h;
}
inline std::string hex16(uint64_t v) { char b[17]; snprintf(b, sizeof b, "%016llx", (unsigned long long)v); return b; }
inline double wall_ms() { return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count(); }
inline bool write_all(int fd, const uint8_t* p, size_t n) {
  while (n) { const ssize_t w = ::write(fd, p, n); if (w <= 0) return false; p += w; n -= (size_t)w; }
  return true;
}
inline bool read_at(int fd, uint8_t* p, size_t n, off_t off) {
  while (n) { const ssize_t r = ::pread(fd, p, n, off); if (r <= 0) return false; p += r; n -= (size_t)r; off += r; }
  return true;
}
inline bool write_file_atomic(const std::string& path, const std::string& text) {
  const std::string tmp = path + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return false;
  const bool ok = write_all(fd, reinterpret_cast<const uint8_t*>(text.data()), text.size()) && ::fsync(fd) == 0;
  ::close(fd);
  if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) { ::unlink(tmp.c_str()); return false; }
  return true;
}
inline std::vector<std::string> list_dir(const std::string& dir) {
  std::vector<std::string> v;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (dirent* e = ::readdir(d)) if (e->d_name[0] != '.') v.push_back(e->d_name);
    ::closedir(d);
  }
  return v;
}
inline long long file_size(const std::string& path) { struct stat st; return ::stat(path.c_str(), &st) == 0 ? (long long)st.st_size : -1; }
// A store directory: <dir>/packs/*, <dir>/metas/* and the two subdirectories (nothing deeper is ever written there)
inline void remove_tree(const std::string& dir) {
  for (const auto& sub : {"packs", "metas"}) {
    const std::string d = dir + "/" + sub;
    for (const auto& f : list_dir(d)) ::unlink((d + "/" + f).c_str());
    ::rmdir(d.c_str());
  }
  ::rmdir(dir.c_str());
}

// ---- Field archives (SeqImage::fields visits every member in a fixed order) --------------------------------------------------------------
// Shape: the structure only (counts, map keys) — part of the layout key, so an image is read back only into the same structure.
struct ShapeAr {
  std::string s;
  void i64(int64_t&) { s += 'q'; }
  void u32(uint32_t&) { s += 'u'; }
  void flag(bool&) { s += 'f'; }
  void vi32(std::vector<int32_t>&) { s += 't'; }
  void vi64(std::vector<int64_t>&) { s += 'e'; }
  void buf(HostImageBuffer&) { s += 'b'; }
  void bufs(std::vector<HostImageBuffer>& v) { s += 'B' + std::to_string(v.size()) + ';'; }
  void bufmap(std::map<int, HostImageBuffer>& m) { s += 'M'; for (auto& [k, v] : m) s += std::to_string(k) + ','; s += ';'; }
  void bytes(std::vector<uint8_t>&) { s += 'y'; }
  void bytesv(std::vector<std::vector<uint8_t>>& v) { s += 'Y' + std::to_string(v.size()) + ';'; }
};

struct Ref { uint64_t pack = 0, off = 0, len = 0, hash = 0; };
inline json ref_json(const Ref& r) { return json::array({hex16(r.pack), r.off, r.len, hex16(r.hash)}); }
inline bool ref_from(const json& j, Ref& r) {
  if (!j.is_array() || j.size() != 4 || !j[0].is_string() || !j[3].is_string()) return false;
  r.pack = std::stoull(j[0].get<std::string>(), nullptr, 16); r.off = j[1].get<uint64_t>(); r.len = j[2].get<uint64_t>();
  r.hash = std::stoull(j[3].get<std::string>(), nullptr, 16);
  return true;
}

// Writes the bytes not yet on disk into one pack and the fields as json. `known` maps a segment allocation to where it already is.
struct WriteAr {
  struct Known { std::weak_ptr<uint8_t> owner; Ref ref; };
  std::function<bool(const uint8_t*, Ref&)> find;  // where an allocation already is on disk (locks the store index per lookup)
  int fd; uint64_t pack; uint64_t off = 0; size_t fresh = 0; bool ok = true;
  json out = json::array();
  std::vector<std::pair<std::shared_ptr<uint8_t>, Ref>> made;  // segments written now (registered as known once the pack is renamed)
  Ref put(const uint8_t* p, size_t n) {
    Ref r{pack, off, n, hash_bytes(p, n)};
    if (n && !write_all(fd, p, n)) ok = false;
    off += n; fresh += n;
    return r;
  }
  Ref seg(const HostImageBuffer::Segment& sg) {
    Ref at;
    if (find && find(sg.data.get(), at) && at.len >= sg.n) {
      if (sg.n != at.len) { at.len = sg.n; at.hash = hash_bytes(sg.data.get(), sg.n); }  // a leading part of a segment on disk
      return at;
    }
    Ref r = put(sg.data.get(), sg.n);
    made.push_back({sg.data, r});
    return r;
  }
  json buf_json(const HostImageBuffer& b) { json a = json::array(); for (auto& sg : b.segments) a.push_back(ref_json(seg(sg))); return a; }
  void i64(int64_t& v) { out.push_back(v); }
  void u32(uint32_t& v) { out.push_back(v); }
  void flag(bool& v) { out.push_back(v); }
  void vi32(std::vector<int32_t>& v) { out.push_back(ref_json(put(reinterpret_cast<const uint8_t*>(v.data()), v.size() * 4))); }
  void vi64(std::vector<int64_t>& v) { out.push_back(ref_json(put(reinterpret_cast<const uint8_t*>(v.data()), v.size() * 8))); }
  void buf(HostImageBuffer& b) { out.push_back(buf_json(b)); }
  void bufs(std::vector<HostImageBuffer>& v) { json a = json::array(); for (auto& b : v) a.push_back(buf_json(b)); out.push_back(a); }
  void bufmap(std::map<int, HostImageBuffer>& m) { json a = json::array(); for (auto& [k, b] : m) a.push_back(json::array({k, buf_json(b)})); out.push_back(a); }
  void bytes(std::vector<uint8_t>& v) { out.push_back(ref_json(put(v.data(), v.size()))); }
  void bytesv(std::vector<std::vector<uint8_t>>& v) { json a = json::array(); for (auto& b : v) a.push_back(ref_json(put(b.data(), b.size()))); out.push_back(a); }
};

// Reads the fields back. Byte ranges come from the packs (opened once each), verified by length and hash.
struct ReadAr {
  const json& in; size_t i = 0; bool ok = true;
  std::function<bool(const Ref&, uint8_t*)> read;
  std::vector<std::pair<std::shared_ptr<uint8_t>, Ref>> loaded;  // allocations made (registered as known so a later save does not rewrite them)
  size_t bytes_read = 0;
  const json& next() { static const json null; if (i >= in.size()) { ok = false; return null; } return in[i++]; }
  template <class T> void vec(std::vector<T>& v) {
    Ref r; if (!ref_from(next(), r) || r.len % sizeof(T)) { ok = false; return; }
    v.resize(r.len / sizeof(T));
    if (r.len && !read(r, reinterpret_cast<uint8_t*>(v.data()))) ok = false;
    bytes_read += r.len;
  }
  bool buf_from(const json& a, HostImageBuffer& b) {
    if (!a.is_array()) return false;
    b.segments.clear();
    for (const auto& rj : a) {
      Ref r; if (!ref_from(rj, r)) return false;
      std::shared_ptr<uint8_t> p = hive::reaped_bytes(std::max<size_t>(r.len, 1));
      if (r.len && !read(r, p.get())) return false;
      bytes_read += r.len;
      loaded.push_back({p, r});
      b.segments.push_back({std::move(p), (size_t)r.len});
    }
    return true;
  }
  void i64(int64_t& v) { const json& j = next(); if (!j.is_number_integer()) ok = false; else v = j.get<int64_t>(); }
  void u32(uint32_t& v) { const json& j = next(); if (!j.is_number_integer()) ok = false; else v = j.get<uint32_t>(); }
  void flag(bool& v) { const json& j = next(); if (!j.is_boolean()) ok = false; else v = j.get<bool>(); }
  void vi32(std::vector<int32_t>& v) { vec(v); }
  void vi64(std::vector<int64_t>& v) { vec(v); }
  void buf(HostImageBuffer& b) { if (!buf_from(next(), b)) ok = false; }
  void bufs(std::vector<HostImageBuffer>& v) {
    const json& a = next(); if (!a.is_array()) { ok = false; return; }
    v.assign(a.size(), HostImageBuffer{});
    for (size_t k = 0; k < a.size() && ok; ++k) if (!buf_from(a[k], v[k])) ok = false;
  }
  void bufmap(std::map<int, HostImageBuffer>& m) {
    const json& a = next(); if (!a.is_array()) { ok = false; return; }
    m.clear();
    for (const auto& e : a) { if (!e.is_array() || e.size() != 2 || !buf_from(e[1], m[e[0].get<int>()])) { ok = false; return; } }
  }
  void bytes(std::vector<uint8_t>& v) { vec(v); }
  void bytesv(std::vector<std::vector<uint8_t>>& v) {
    const json& a = next(); if (!a.is_array()) { ok = false; return; }
    v.assign(a.size(), {});
    for (size_t k = 0; k < a.size() && ok; ++k) {
      Ref r; if (!ref_from(a[k], r)) { ok = false; return; }
      v[k].resize(r.len);
      if (r.len && !read(r, v[k].data())) ok = false;
      bytes_read += r.len;
    }
  }
};

// One stored image as the engine needs it back.
struct Sig { int start, end; uint64_t hash; };
template <class Image> struct Loaded {
  std::shared_ptr<Image> image;
  std::vector<float> logits;  // ckpt only
  std::vector<Sig> sig;
  bool upper_skipped = false;
  double ms = 0; size_t bytes = 0;
};

template <class Image>
class SessionStore {
 public:
  struct Stats { long writes = 0, write_fail = 0, loads = 0, load_fail = 0, evicted = 0; double write_ms = 0; size_t written = 0; };
  // dir = HIVE_SESSION_STORE · key = layout key (family, store format, checkpoint config, image shape) · budget in bytes
  SessionStore(std::string dir, std::string key, size_t budget) : root_(std::move(dir)), budget_(budget) {
    dir_ = root_ + "/" + key;
    ::mkdir(root_.c_str(), 0755);
    // the same daemon's directories of another layout (store format, checkpoint, image shape — or the build-keyed directories before the
    //   canary) can never be read again — drop them, or every change would leave its store behind (measured 2026-10-15: 19 GB after one
    //   rebuild); other daemons' directories are left alone
    if (const size_t dash = key.rfind('-'); dash != std::string::npos)
      for (const auto& d : list_dir(root_))
        if (d != key && d.compare(0, dash + 1, key, 0, dash + 1) == 0 && d.size() == key.size()) remove_tree(root_ + "/" + d);
    ::mkdir(dir_.c_str(), 0755);
    ::mkdir((dir_ + "/packs").c_str(), 0755); ::mkdir((dir_ + "/metas").c_str(), 0755);
    scan();
    th_ = std::thread([this] { run(); });
  }
  ~SessionStore() {
    { std::lock_guard<std::mutex> lk(qm_); stop_ = true; } qcv_.notify_all();
    if (th_.joinable()) th_.join();
    for (auto& [id, fd] : fds_) ::close(fd);
  }
  const std::string& dir() const { return dir_; }
  // ---- engine thread: enqueue (never waits for the disk) ----
  void put_ckpt(const std::string& sid, std::shared_ptr<Image> img, const std::vector<float>& logits, std::vector<Sig> sig) {
    Job j; j.kind = "ckpt"; j.name = sid; j.image = std::move(img); j.logits = logits; j.sig = std::move(sig);
    enqueue(std::move(j));
  }
  void put_snap(uint64_t key, std::shared_ptr<Image> img, std::vector<Sig> sig, bool upper_skipped) {
    Job j; j.kind = "snap"; j.name = hex16(key); j.key = key; j.image = std::move(img); j.sig = std::move(sig); j.upper_skipped = upper_skipped;
    enqueue(std::move(j));
  }
  // the canary (hived store_canary): kept outside the session index — no session id can name it, the budget never drops it
  void put_canary(std::shared_ptr<Image> img, const std::vector<float>& logits) {
    Job j; j.kind = "canary"; j.name = "canary"; j.image = std::move(img); j.logits = logits;
    enqueue(std::move(j));
  }
  bool has_canary() { std::lock_guard<std::mutex> lk(im_); return !canary_.file.empty(); }
  std::unique_ptr<Loaded<Image>> load_canary() { std::string f; { std::lock_guard<std::mutex> lk(im_); f = canary_.file; } return load(f); }
  size_t entries() { std::lock_guard<std::mutex> lk(im_); return ckpt_.size() + snaps_.size(); }
  bool has_ckpt(const std::string& sid) { std::lock_guard<std::mutex> lk(im_); return ckpt_.count(sid) > 0; }
  // position and last token of a session's checkpoint on disk (-1 = none) — the caller checks the token before reading GBs
  std::pair<int64_t, int32_t> ckpt_at(const std::string& sid) {
    std::lock_guard<std::mutex> lk(im_); auto it = ckpt_.find(sid);
    return it == ckpt_.end() ? std::pair<int64_t, int32_t>{-1, -1} : std::pair<int64_t, int32_t>{it->second.pos, it->second.last_tok};
  }
  // Snapshots on disk whose position lies in (lo, hi) and whose token at pos-1 matches — the caller confirms with its prefix key.
  std::vector<std::pair<uint64_t, int64_t>> snap_candidates(int64_t lo, int64_t hi, const std::function<int32_t(int64_t)>& tok_at) {
    std::vector<std::pair<uint64_t, int64_t>> v;
    std::lock_guard<std::mutex> lk(im_);
    for (const auto& [key, m] : snaps_) if (m.pos > lo && m.pos < hi && tok_at(m.pos - 1) == m.last_tok) v.push_back({key, m.pos});
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
    return v;
  }
  // ---- engine thread: load (reads the disk) ----
  std::unique_ptr<Loaded<Image>> load_ckpt(const std::string& sid) { return load(meta_of("ckpt", sid)); }
  std::unique_ptr<Loaded<Image>> load_snap(uint64_t key) { return load(meta_of("snap", hex16(key))); }
  json stats() {
    std::lock_guard<std::mutex> lk(im_);
    return {{"dir", dir_}, {"ckpts", ckpt_.size()}, {"snaps", snaps_.size()}, {"bytes", disk_bytes_}, {"budget", budget_},
            {"writes", st_.writes}, {"write_fail", st_.write_fail}, {"loads", st_.loads}, {"load_fail", st_.load_fail},
            {"evicted", st_.evicted}, {"written", st_.written}, {"queue", queued_.load()}, {"canary", !canary_.file.empty()}};
  }
  // Waits until the writer is idle (tests and the flush op).
  void drain() { std::unique_lock<std::mutex> lk(qm_); idle_cv_.wait(lk, [&] { return q_.empty() && !busy_; }); }
  // The flush op ("drop every reusable prompt state"): the disk copies go too (the canary stays — it is no prompt state; with_canary drops
  //   it as well, for a directory a new build cannot read). Returns the number of entries dropped.
  size_t clear(bool with_canary = false) {
    { std::lock_guard<std::mutex> lk(qm_); queued_ -= (long)q_.size(); q_.clear(); }
    drain();
    for (auto& [id, fd] : fds_) ::close(fd);
    fds_.clear();
    std::lock_guard<std::mutex> lk(im_);
    const size_t n = ckpt_.size() + snaps_.size();
    for (auto& [k, m] : ckpt_) ::unlink(m.file.c_str());
    for (auto& [k, m] : snaps_) ::unlink(m.file.c_str());
    if (with_canary && !canary_.file.empty()) { ::unlink(canary_.file.c_str()); canary_ = Meta{}; }
    ckpt_.clear(); snaps_.clear(); known_.clear();
    gc_packs();
    return n;
  }

 private:
  struct Job { std::string kind, name; uint64_t key = 0; std::shared_ptr<Image> image; std::vector<float> logits; std::vector<Sig> sig; bool upper_skipped = false; };
  struct Meta { std::string file; int64_t pos = 0; int32_t last_tok = -1; double last_used = 0; std::set<uint64_t> packs; };
  std::string root_, dir_;
  size_t budget_;
  std::thread th_;
  std::mutex qm_; std::condition_variable qcv_, idle_cv_; std::deque<Job> q_; bool stop_ = false, busy_ = false;
  std::atomic<long> queued_{0};
  std::mutex im_;  // index + stats + known
  std::map<std::string, Meta> ckpt_;            // session id -> its latest checkpoint
  std::map<uint64_t, Meta> snaps_;              // prefix key -> boundary snapshot
  Meta canary_;                                 // the canary (file empty = none)
  std::map<uint64_t, long long> pack_bytes_;    // pack id -> size
  size_t disk_bytes_ = 0;
  std::map<const uint8_t*, typename WriteAr::Known> known_;
  std::map<uint64_t, int> fds_;                 // engine thread only (load)
  uint64_t next_pack_ = 0;
  Stats st_;

  std::string pack_path(uint64_t id) const { return dir_ + "/packs/" + hex16(id) + ".pack"; }
  // file name = kind + a hash of the session id or the prefix key (session ids are client strings)
  std::string meta_path(const std::string& kind, const std::string& name) const {
    return dir_ + "/metas/" + kind + "-" + hex16(hash_bytes(reinterpret_cast<const uint8_t*>(name.data()), name.size())) + ".json";
  }
  std::string meta_of(const std::string& kind, const std::string& name) {
    std::lock_guard<std::mutex> lk(im_);
    if (kind == "ckpt") { auto it = ckpt_.find(name); return it == ckpt_.end() ? "" : it->second.file; }
    auto it = snaps_.find(std::stoull(name, nullptr, 16)); return it == snaps_.end() ? "" : it->second.file;
  }
  static void collect_packs(const json& j, std::set<uint64_t>& out) {
    if (j.is_array() && j.size() == 4 && j[0].is_string() && j[3].is_string()) { out.insert(std::stoull(j[0].get<std::string>(), nullptr, 16)); return; }
    if (j.is_array() || j.is_object()) for (const auto& e : j) collect_packs(e, out);
  }
  // Start-up: drop temporary files, read the metas whose packs are all present, then drop packs nothing refers to.
  void scan() {
    for (const auto& sub : {"packs", "metas"})
      for (const auto& f : list_dir(dir_ + "/" + sub)) if (f.size() > 4 && f.compare(f.size() - 4, 4, ".tmp") == 0) ::unlink((dir_ + "/" + sub + "/" + f).c_str());
    for (const auto& f : list_dir(dir_ + "/packs")) {
      const uint64_t id = std::stoull(f.substr(0, 16), nullptr, 16);
      pack_bytes_[id] = file_size(dir_ + "/packs/" + f);
      next_pack_ = std::max(next_pack_, id + 1);
    }
    for (const auto& f : list_dir(dir_ + "/metas")) {
      const std::string path = dir_ + "/metas/" + f;
      json m;
      try { FILE* fp = fopen(path.c_str(), "r"); if (!fp) continue; m = json::parse(fp); fclose(fp); } catch (...) { ::unlink(path.c_str()); continue; }
      Meta e; e.file = path;
      try { e.pos = m.at("pos").get<int64_t>(); e.last_tok = m.at("last_tok").get<int32_t>(); e.last_used = m.at("last_used").get<double>(); } catch (...) { ::unlink(path.c_str()); continue; }
      collect_packs(m.at("fields"), e.packs); if (m.contains("logits")) collect_packs(m["logits"], e.packs);
      bool ok = true;
      for (uint64_t p : e.packs) if (!pack_bytes_.count(p)) ok = false;
      if (!ok) { ::unlink(path.c_str()); continue; }
      const std::string kind = m.value("kind", "");
      if (kind == "ckpt") ckpt_[m.value("sid", "")] = e;
      else if (kind == "snap") snaps_[std::stoull(m.value("key", "0"), nullptr, 16)] = e;
      else if (kind == "canary") canary_ = e;
    }
    gc_packs();
    for (auto& [id, n] : pack_bytes_) disk_bytes_ += (size_t)std::max(0LL, n);
  }
  void gc_packs() {  // under im_ (or before the writer starts)
    std::set<uint64_t> used;
    for (auto& [k, m] : ckpt_) used.insert(m.packs.begin(), m.packs.end());
    for (auto& [k, m] : snaps_) used.insert(m.packs.begin(), m.packs.end());
    used.insert(canary_.packs.begin(), canary_.packs.end());
    for (auto it = pack_bytes_.begin(); it != pack_bytes_.end();) {
      if (used.count(it->first)) { ++it; continue; }
      ::unlink(pack_path(it->first).c_str());
      disk_bytes_ -= std::min(disk_bytes_, (size_t)std::max(0LL, it->second));
      for (auto k = known_.begin(); k != known_.end();) k = k->second.ref.pack == it->first ? known_.erase(k) : std::next(k);
      it = pack_bytes_.erase(it);
    }
  }
  void enqueue(Job j) {
    { std::lock_guard<std::mutex> lk(qm_); q_.push_back(std::move(j)); ++queued_; }
    qcv_.notify_one();
  }
  void run() {
    for (;;) {
      Job j;
      {
        std::unique_lock<std::mutex> lk(qm_);
        qcv_.wait(lk, [&] { return stop_ || !q_.empty(); });
        if (q_.empty()) return;  // stop with nothing queued
        j = std::move(q_.front()); q_.pop_front(); --queued_; busy_ = true;
      }
      try { write(j); } catch (...) { std::lock_guard<std::mutex> lk(im_); ++st_.write_fail; }
      j.image.reset();
      { std::lock_guard<std::mutex> lk(qm_); busy_ = false; }
      idle_cv_.notify_all();
    }
  }
  void write(Job& j) {
    const double t0 = wall_ms();
    j.image->wait_ready();  // device-to-host copies of the image still in flight
    uint64_t pack;
    { std::lock_guard<std::mutex> lk(im_); pack = next_pack_++; }
    const std::string ppath = pack_path(pack), ptmp = ppath + ".tmp";
    const int fd = ::open(ptmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { std::lock_guard<std::mutex> lk(im_); ++st_.write_fail; return; }
    WriteAr ar;
    ar.fd = fd; ar.pack = pack;
    ar.find = [this](const uint8_t* p, Ref& r) {  // the index lock only per lookup — the engine thread reads the index while this writes GBs
      std::lock_guard<std::mutex> lk(im_);
      auto it = known_.find(p);
      if (it == known_.end() || it->second.owner.expired()) return false;
      r = it->second.ref;
      return true;
    };
    j.image->fields(ar);
    std::vector<Ref> logit_ref;
    if (!j.logits.empty()) logit_ref.push_back(ar.put(reinterpret_cast<const uint8_t*>(j.logits.data()), j.logits.size() * 4));
    const bool synced = ar.ok && ::fsync(fd) == 0;
    ::close(fd);
    if (!synced || (ar.fresh ? ::rename(ptmp.c_str(), ppath.c_str()) != 0 : (::unlink(ptmp.c_str()), false))) {
      ::unlink(ptmp.c_str());
      std::lock_guard<std::mutex> lk(im_); ++st_.write_fail; return;
    }
    json m = {{"v", 1}, {"kind", j.kind}, {"pos", j.image->pos}, {"last_used", wall_ms()},
              {"last_tok", j.image->tokens.empty() ? -1 : j.image->tokens.back()}, {"fields", ar.out}};
    if (j.kind == "ckpt") m["sid"] = j.name; else if (j.kind == "snap") { m["key"] = j.name; m["upper_skipped"] = j.upper_skipped; }
    json sg = json::array(); for (const auto& s : j.sig) sg.push_back(json::array({s.start, s.end, hex16(s.hash)})); m["sig"] = sg;
    if (!logit_ref.empty()) m["logits"] = ref_json(logit_ref[0]);
    const std::string mpath = meta_path(j.kind, j.name);
    if (!write_file_atomic(mpath, m.dump())) { std::lock_guard<std::mutex> lk(im_); ++st_.write_fail; return; }
    std::lock_guard<std::mutex> lk(im_);
    if (ar.fresh) { pack_bytes_[pack] = (long long)ar.off; disk_bytes_ += ar.off; }
    for (auto& [owner, r] : ar.made) known_[owner.get()] = {owner, r};
    Meta e; e.file = mpath; e.pos = j.image->pos; e.last_tok = j.image->tokens.empty() ? -1 : j.image->tokens.back(); e.last_used = m["last_used"];
    collect_packs(ar.out, e.packs); if (!logit_ref.empty()) e.packs.insert(pack);
    if (j.kind == "ckpt") ckpt_[j.name] = e; else if (j.kind == "snap") snaps_[j.key] = e; else canary_ = e;
    gc_packs();  // a replaced checkpoint's packs that no entry refers to any more
    if (j.kind != "canary") { ++st_.writes; st_.written += ar.fresh; st_.write_ms += wall_ms() - t0; }  // the counters are conversation state
    evict();
    fprintf(stderr, "[store] %s %.16s pos %lld · %.1f MB new · %.0f ms · disk %.2f GB / %.0f GB\n", j.kind.c_str(), j.name.c_str(),
            (long long)j.image->pos, ar.fresh / 1048576.0, wall_ms() - t0, disk_bytes_ / 1073741824.0, budget_ / 1073741824.0);
  }
  void evict() {  // under im_: oldest entries first until the packs fit the budget (the newest entry always stays)
    while (disk_bytes_ > budget_ && ckpt_.size() + snaps_.size() > 1) {
      double oldest = 1e300; std::string ck; uint64_t sk = 0; bool is_snap = false;
      for (auto& [k, m] : ckpt_) if (m.last_used < oldest) { oldest = m.last_used; ck = k; is_snap = false; }
      for (auto& [k, m] : snaps_) if (m.last_used < oldest) { oldest = m.last_used; sk = k; is_snap = true; }
      if (is_snap) { ::unlink(snaps_[sk].file.c_str()); snaps_.erase(sk); } else { ::unlink(ckpt_[ck].file.c_str()); ckpt_.erase(ck); }
      ++st_.evicted;
      gc_packs();
    }
  }
  int pack_fd(uint64_t id) {
    auto it = fds_.find(id);
    if (it != fds_.end()) return it->second;
    const int fd = ::open(pack_path(id).c_str(), O_RDONLY);
    if (fd >= 0) fds_[id] = fd;
    return fd;
  }
  std::unique_ptr<Loaded<Image>> load(const std::string& path) {
    if (path.empty()) return nullptr;
    try { return load_file(path); } catch (...) { std::lock_guard<std::mutex> lk(im_); ++st_.load_fail; return nullptr; }
  }
  std::unique_ptr<Loaded<Image>> load_file(const std::string& path) {
    if (fds_.size() > 256) { for (auto& [id, fd] : fds_) ::close(fd); fds_.clear(); }
    const double t0 = wall_ms();
    auto fail = [&]() -> std::unique_ptr<Loaded<Image>> { std::lock_guard<std::mutex> lk(im_); ++st_.load_fail; return nullptr; };
    json m;
    try { FILE* fp = fopen(path.c_str(), "r"); if (!fp) return fail(); m = json::parse(fp); fclose(fp); } catch (...) { return fail(); }
    auto out = std::make_unique<Loaded<Image>>();
    out->image = std::make_shared<Image>();
    // The field walk only allocates and records what to read; the reads and hash checks then run on several threads (measured 2026-10-15:
    //   one by one, 4.94 GB took 4.7 s — about 1 GB/s on a disk that reads 6.5 GB/s; an image is thousands of segments)
    struct Job { Ref r; uint8_t* dst; int fd; };
    std::vector<Job> jobs;
    bool fd_ok = true;
    auto read = [&](const Ref& r, uint8_t* dst) {
      const int fd = pack_fd(r.pack);
      if (fd < 0) { fd_ok = false; return false; }
      jobs.push_back({r, dst, fd});
      return true;
    };
    ReadAr ar{m.at("fields")};
    ar.read = read;
    out->image->fields(ar);
    if (!ar.ok || !fd_ok || ar.i != m["fields"].size()) return fail();
    if (m.contains("logits")) {
      Ref r; if (!ref_from(m["logits"], r) || r.len % 4) return fail();
      out->logits.resize(r.len / 4);
      if (r.len && !read(r, reinterpret_cast<uint8_t*>(out->logits.data()))) return fail();
    }
    {
      const unsigned nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
      std::atomic<size_t> next{0};
      std::atomic<bool> bad{false};
      auto work = [&] {
        for (size_t k; (k = next.fetch_add(1)) < jobs.size() && !bad.load();) {
          const Job& j = jobs[k];
          if (!read_at(j.fd, j.dst, j.r.len, (off_t)j.r.off) || hash_bytes(j.dst, j.r.len) != j.r.hash) bad = true;
        }
      };
      std::vector<std::thread> ths;
      for (unsigned t = 1; t < nt; ++t) ths.emplace_back(work);
      work();
      for (auto& th : ths) th.join();
      if (bad) return fail();
    }
    for (const auto& s : m.value("sig", json::array())) out->sig.push_back({s[0].get<int>(), s[1].get<int>(), std::stoull(s[2].get<std::string>(), nullptr, 16)});
    out->upper_skipped = m.value("upper_skipped", false);
    // A fresh lineage: the generation decides which earlier image a later save may share rows with (save_image) — a value from another
    //   process could equal a live sequence's
    static std::atomic<uint32_t> gen{0};
    out->image->generation = 0xC0000000u | (++gen & 0x3FFFFFFFu);
    out->bytes = ar.bytes_read; out->ms = wall_ms() - t0;
    std::lock_guard<std::mutex> lk(im_);
    for (auto& [p, r] : ar.loaded) known_[p.get()] = {p, r};
    if (m.value("kind", "") != "canary") ++st_.loads;
    return out;
  }
};

}  // namespace hive::store
