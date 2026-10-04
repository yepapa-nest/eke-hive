// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Cold load HIVE_LOAD_PAR — CPU test (fake CUDA/NUMA, no GPU). Built and run by tools/test_load_par_cpu.py.
//
// Real safetensors.cpp (real files, mmap) + real ExpertStore (engine/src/expert_store.cpp) + hive/bulk_load.h:
//  1. bulk     : random files and random pieces (unaligned offsets, zero length, overlapping file ranges, up to end of file) read with bulk_load == pread reference.
//                O_DIRECT / buffered · 1·3·8 threads · chunks 4 KiB·64 KiB·32 MiB · merge_gap 0·1 MiB. A piece past end of file = false (truncated file) ·
//                missing file = false (the caller falls back to the per-tensor path).
//  2. store    : a fake checkpoint (real safetensors shards — odd header length so data is not 4 KiB aligned · dense tensors between expert tensors · one layer spanning two
//                shards · MTP layer · 2 engram tables) loaded as A = per-tensor path (load_layer_experts + pin_all + load_engram) and B = load_bulk (several options);
//                all half records (tail alignment padding included), all engram tables, table metadata (rows, hd, layer) and loaded are byte-identical · host_checksums (HIVE_LOAD_CHECKSUM log)
//                match too (independent of thread count) and change when a single byte changes.
//  3. fallback : after Checkpoint has opened a shard it is renamed (cannot be reopened by path) → load_bulk returns false → the store reloads via the per-tensor path, same as A.
//  5. prefault : HostPrefault::predict == the actual arena and engram allocations (all taken, nothing left) · bulk/per-layer loads into the pre-reserved arena == reference ·
//                a wrong prediction = the pool is dropped and normal allocation used · value-preserving touches do not erase concurrent writes (8 touch threads + 4 writer threads, 192 MiB × 3).
//  4. dense    : DensePrefetch selects only dense tensors (excluding experts, engram tables, vision) and reads ≥ the dense total · a missing directory ends quietly.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hive/bulk_load.h"
#include "hive/expert_store.h"
#include "hive/safetensors.h"
#include "nlohmann/json.hpp"

using namespace hive;
namespace fs = std::filesystem;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { if (fails < 30) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } ++fails; } } while (0)

namespace {

bool expect_direct() { const char* v = getenv("HIVE_TEST_EXPECT_DIRECT"); return v && strcmp(v, "1") == 0; }
constexpr int DIM = 256, INTER = 256, E = 6, L = 3, N_MTP = 1, MTP_E = 4;

std::vector<uint8_t> rand_bytes(std::mt19937_64& rng, size_t n) {
  std::vector<uint8_t> v(n);
  for (auto& b : v) b = (uint8_t)rng();
  return v;
}

// ---- 1. bulk_load unit ----
void test_bulk(const std::string& dir, std::mt19937_64& rng) {
  std::vector<std::string> paths;
  std::vector<std::vector<uint8_t>> data;
  for (int f = 0; f < 3; ++f) {
    const size_t n = 3 * 1048576 + 12345 * (f + 1);  // not a multiple of 4 KiB (short read at end of file)
    data.push_back(rand_bytes(rng, n));
    paths.push_back(dir + "/bulk" + std::to_string(f) + ".bin");
    std::ofstream(paths.back(), std::ios::binary).write((const char*)data.back().data(), (std::streamsize)n);
  }
  for (int round = 0; round < 24; ++round) {
    std::vector<LoadSeg> segs;
    std::vector<std::vector<uint8_t>> dst;
    const int nseg = 1 + (int)(rng() % 300);
    dst.reserve(nseg + 4);
    for (int i = 0; i < nseg; ++i) {
      const int f = (int)(rng() % 3);
      const size_t n = data[f].size();
      size_t off = rng() % n, len = (rng() % 5 == 0) ? 0 : 1 + rng() % std::min<size_t>(n - off, (rng() % 2) ? 5000 : 400000);
      if (i == 0) { off = 0; len = n; }                       // whole file
      if (i == 1) { off = n - 7; len = 7; }                   // tail at end of file
      len = std::min(len, n - off);
      dst.emplace_back(len + 1, 0xEE);
      segs.push_back({f, off, len, dst.back().data()});
    }
    BulkLoadOpts o;
    const int threads[] = {1, 3, 8};
    const size_t chunks[] = {4096, 65536, 32ull << 20};
    o.threads = threads[round % 3];
    o.chunk = chunks[(round / 3) % 3];
    o.direct = (round % 2) == 0;
    o.merge_gap = (round % 4 < 2) ? 0 : (1u << 20);
    BulkLoadStats st;
    const bool ok = bulk_load(paths, segs, o, &st);
    EXPECT(ok, "bulk round %d: failed: %s", round, st.err.c_str());
    if (o.direct && expect_direct()) EXPECT(st.direct_files == 3, "bulk round %d: O_DIRECT files %d/3 (test dir must be on a disk filesystem)", round, st.direct_files);
    int bad = 0;
    for (size_t i = 0; i < segs.size(); ++i) {
      const auto& s = segs[i];
      bad += s.len && memcmp(dst[i].data(), data[s.file].data() + s.off, s.len) != 0;
      bad += dst[i][s.len] != 0xEE;  // nothing is written outside the piece
    }
    EXPECT(bad == 0, "bulk round %d (threads %d chunk %zu direct %d gap %zu): %d segment(s) differ", round, o.threads, o.chunk, o.direct, o.merge_gap, bad);
  }
  {  // piece past end of file = failure (truncated file)
    std::vector<uint8_t> d(100);
    BulkLoadStats st;
    const bool ok = bulk_load(paths, {{0, data[0].size() - 50, 100, d.data()}}, BulkLoadOpts{}, &st);
    EXPECT(!ok && st.err.find("short read") != std::string::npos, "bulk: past-EOF segment not reported (ok %d err %s)", ok, st.err.c_str());
  }
  {  // missing file = failure
    std::vector<uint8_t> d(10);
    BulkLoadStats st;
    const bool ok = bulk_load({dir + "/nope.bin"}, {{0, 0, 10, d.data()}}, BulkLoadOpts{}, &st);
    EXPECT(!ok && !st.err.empty(), "bulk: missing file not reported");
  }
}

// ---- the touch threads must not erase content in regions already handed over (value-preserving atomic touch) ----
void test_prefault_race(std::mt19937_64& rng) {
  const size_t n = 192ull << 20;
  for (int round = 0; round < 3; ++round) {
    HostPrefault::start({{n, -1}}, 8);
    uint8_t* p = HostPrefault::take(n, -1, true);
    EXPECT(p != nullptr, "prefault race: take failed");
    if (!p) { HostPrefault::finish(); continue; }
    const uint64_t seed = rng();
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t)  // write concurrently with the touches (emulating loader threads) — includes the touch position (first 8 bytes) of every 4 KiB
      th.emplace_back([=] { for (size_t off = (size_t)t * 8; off < n; off += 32) { uint64_t v = seed ^ (off * 0x9E3779B97F4A7C15ull); memcpy(p + off, &v, 8); } });
    for (auto& x : th) x.join();
    HostPrefault::finish();
    size_t bad = 0;
    for (size_t off = 0; off < n; off += 8) { uint64_t v; memcpy(&v, p + off, 8); bad += v != (seed ^ (off * 0x9E3779B97F4A7C15ull)); }
    EXPECT(bad == 0, "prefault race round %d: %zu words clobbered", round, bad);
    munmap(p, n);  // fake NUMA: node -1 = mmap
  }
}

// ---- fake checkpoint (real safetensors shards) ----
struct FakeTensor { std::string name, dtype; std::vector<int64_t> shape; std::vector<uint8_t> bytes; };
void write_shard(const std::string& path, const std::vector<FakeTensor>& ts, int pad) {
  nlohmann::json h = nlohmann::json::object();
  size_t off = 0;
  for (const auto& t : ts) {
    h[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {off, off + t.bytes.size()}}};
    off += t.bytes.size();
  }
  h["__metadata__"] = {{"pad", std::string((size_t)pad, 'x')}};  // varies the header length (= data start)
  const std::string hs = h.dump();
  std::ofstream f(path, std::ios::binary);
  const uint64_t n = hs.size();
  f.write((const char*)&n, 8);
  f.write(hs.data(), (std::streamsize)n);
  for (const auto& t : ts) f.write((const char*)t.bytes.data(), (std::streamsize)t.bytes.size());
}

struct FakeCkpt {
  std::string dir;
  size_t dense_bytes = 0;
  std::vector<std::pair<int, int>> engram = {{1, 0}, {2, 1}};  // (layer, hash_index)
  void build(std::mt19937_64& rng) {
    fs::create_directories(dir);
    const size_t wsz = (size_t)INTER * DIM / 2, ssz = (size_t)INTER * DIM / 32;
    std::map<std::string, std::vector<FakeTensor>> shards;
    nlohmann::json wm = nlohmann::json::object();
    auto put = [&](const std::string& shard, FakeTensor t) { wm[t.name] = shard; shards[shard].push_back(std::move(t)); };
    auto dense = [&](const std::string& shard, const std::string& name, size_t n) {
      if (name.rfind("vision.", 0) != 0) dense_bytes += n;
      put(shard, {name, "U8", {(int64_t)n}, rand_bytes(rng, n)});
    };
    dense("s-embed.safetensors", "embed.weight", 50001);
    dense("s-embed.safetensors", "vision.blocks.0.w", 7777);
    dense("s-head.safetensors", "head.weight", 3000);
    dense("s-head.safetensors", "norm.weight", 512);
    dense("s-3.safetensors", "mtp.0.attn_norm.weight", 512);
    for (int l = 0; l < L + N_MTP; ++l) {
      const std::string pre = l < L ? "layers." + std::to_string(l) : "mtp." + std::to_string(l - L);
      const int El = l < L ? E : MTP_E;
      std::vector<FakeTensor> ex;
      for (int e = 0; e < El; ++e)
        for (const char* m : {"w1", "w2", "w3"}) {
          const std::string p = pre + ".ffn.experts." + std::to_string(e) + "." + m;
          ex.push_back({p + ".weight", "U8", {(int64_t)wsz}, rand_bytes(rng, wsz)});
          ex.push_back({p + ".scale", "U8", {(int64_t)ssz}, rand_bytes(rng, ssz)});
        }
      std::shuffle(ex.begin(), ex.end(), rng);  // order in the file ≠ load order
      const std::string sh = "s-" + std::to_string(l) + ".safetensors", sh2 = "s-" + std::to_string(l) + "b.safetensors";
      dense(sh, pre + ".attn.wq_a.weight", 3001 + l);
      for (size_t i = 0; i < ex.size(); ++i) {
        // layer 1 puts its second half in another shard · odd-sized dense tensors between experts
        const std::string& target = (l == 1 && i >= ex.size() / 2) ? sh2 : sh;
        put(target, std::move(ex[i]));
        if (i % 7 == 3) dense(target, pre + ".dense_pad." + std::to_string(i), 1 + rng() % 9000);
      }
      dense(sh, pre + ".ffn.gate.weight", 4097);
    }
    for (auto [layer, hi] : engram) {
      const int64_t rows = 1000 + 333 * hi, hd = 256;
      const std::string p = "layers." + std::to_string(layer) + ".engram.embed.";
      const std::string sh = "s-engram" + std::to_string(hi) + ".safetensors";
      dense(sh, "layers." + std::to_string(layer) + ".engram.q_weight", 2049);
      put(sh, {p + "weight", "F8_E4M3", {rows, hd}, rand_bytes(rng, (size_t)(rows * hd))});
      put(sh, {p + "scale", "F8_E8M0", {rows, hd / 32}, rand_bytes(rng, (size_t)(rows * hd / 32))});
    }
    int pad = 1;
    for (auto& [name, ts] : shards) write_shard(dir + "/" + name, ts, pad += 37);
    std::ofstream(dir + "/model.safetensors.index.json") << nlohmann::json{{"weight_map", wm}}.dump();
    std::ofstream(dir + "/config.json") << nlohmann::json{{"text_config", {{"hidden_size", DIM}, {"moe_intermediate_size", INTER}, {"n_routed_experts", E},
        {"num_hidden_layers", L}, {"dspark_n_routed_experts", MTP_E}, {"engram_layer_ids", {1, 2}}}}}.dump();
  }
};

Config small_cfg() {
  Config c;
  c.dim = DIM; c.moe_inter = INTER; c.n_routed = E; c.dspark_experts = MTP_E; c.swiglu_limit = 10.f;
  return c;
}

void load_reference(ExpertStore& A, Checkpoint& ck, const FakeCkpt& fc) {
  for (int l = 0; l < L + N_MTP; ++l) A.load_layer_experts(ck, l, 2);
  A.pin_all();
  for (auto [layer, hi] : fc.engram) A.load_engram(ck, layer, hi);
}

int compare(ExpertStore& A, ExpertStore& B, const FakeCkpt& fc, const char* what) {
  int bad = 0;
  const size_t ht = A.half_layout().total;
  for (int l = 0; l < L + N_MTP; ++l) {
    bad += A.layer_loaded(l) != B.layer_loaded(l);
    for (int e = 0; e < A.E_of(l); ++e)
      for (int n = 0; n < 2; ++n) bad += memcmp(A.host_half(l, e, n), B.host_half(l, e, n), ht) != 0;
  }
  for (auto [layer, hi] : fc.engram) {
    const EngramTable* a = A.engram(hi);
    const EngramTable* b = B.engram(hi);
    if (!a || !b) { ++bad; continue; }
    bad += a->layer != b->layer || a->rows != b->rows || a->hd != b->hd || a->vals.bytes != b->vals.bytes || a->scales.bytes != b->scales.bytes;
    bad += a->vals.node != b->vals.node || a->scales.node != b->scales.node;
    bad += memcmp(a->vals.base, b->vals.base, a->vals.bytes) != 0;
    bad += memcmp(a->scales.base, b->scales.base, a->scales.bytes) != 0;
  }
  bad += A.host_checksums(3) != B.host_checksums(1);  // HIVE_LOAD_CHECKSUM log values (independent of thread count)
  EXPECT(bad == 0, "%s: %d difference(s) against the per-layer reference", what, bad);
  return bad;
}

// Does the reference record equal the source tensors (checks the definition of the per-tensor path itself — half record n = the n-th half of each tensor)
void check_reference_definition(ExpertStore& A, Checkpoint& ck) {
  const HalfLayout& h = A.half_layout();
  const size_t w13 = h.s1 - h.w1, s13 = h.w3 - h.s1, w2 = h.s2 - h.w2, s2 = h.w2_rows * INTER / 32;
  int bad = 0;
  for (int l = 0; l < L + N_MTP; ++l) {
    const std::string pre = (l < L ? "layers." + std::to_string(l) : "mtp." + std::to_string(l - L)) + ".ffn.experts.";
    for (int e = 0; e < A.E_of(l); ++e)
      for (int n = 0; n < 2; ++n) {
        const uint8_t* r = A.host_half(l, e, n);
        auto t = [&](const char* m, const char* k) { return ck.get(pre + std::to_string(e) + "." + m + "." + k).data; };
        bad += memcmp(r + h.w1, t("w1", "weight") + n * w13, w13) != 0;
        bad += memcmp(r + h.s1, t("w1", "scale") + n * s13, s13) != 0;
        bad += memcmp(r + h.w3, t("w3", "weight") + n * w13, w13) != 0;
        bad += memcmp(r + h.s3, t("w3", "scale") + n * s13, s13) != 0;
        bad += memcmp(r + h.w2, t("w2", "weight") + n * w2, w2) != 0;
        bad += memcmp(r + h.s2, t("w2", "scale") + n * s2, s2) != 0;
      }
  }
  EXPECT(bad == 0, "reference: %d piece(s) differ from the source tensors", bad);
}

void test_store(const std::string& dir, std::mt19937_64& rng) {
  FakeCkpt fc{dir + "/ckpt"};
  fc.build(rng);
  const Config cfg = small_cfg();
  Checkpoint ckA(fc.dir);
  ExpertStore A(cfg, L, 0, 1, N_MTP);
  load_reference(A, ckA, fc);
  check_reference_definition(A, ckA);
  {  // the checksum changes on a single byte (the verification log really compares)
    const auto c0 = A.host_checksums(2);
    uint8_t* p = const_cast<uint8_t*>(A.host_half(L - 1, E - 1, 1)) + 77;
    *p ^= 1;
    const auto c1 = A.host_checksums(2);
    *p ^= 1;
    EXPECT(c0.size() == 6 && c0 != c1 && c0 == A.host_checksums(4), "host_checksums: size %zu · flips a byte %d · thread-independent %d", c0.size(), c0 != c1,
           c0 == A.host_checksums(4));
  }
  struct V { int threads; size_t chunk; bool direct; size_t gap; bool pin; };
  const V vs[] = {{8, 32ull << 20, true, 1u << 20, true}, {3, 4096, true, 0, true}, {1, 65536, false, 1u << 20, false}, {5, 12288, true, 1u << 20, true},
                  {2, 1ull << 20, false, 0, true}};
  int vi = 0;
  for (const V& v : vs) {
    Checkpoint ck(fc.dir);
    ExpertStore B(cfg, L, 0, 1, N_MTP);
    BulkLoadOpts o; o.threads = v.threads; o.chunk = v.chunk; o.direct = v.direct; o.merge_gap = v.gap;
    BulkLoadStats st;
    const bool ok = B.load_bulk(ck, L + N_MTP, fc.engram, o, v.pin, &st);
    if (!v.pin) B.pin_all();
    char what[128];
    snprintf(what, sizeof what, "store variant %d (threads %d chunk %zu direct %d gap %zu pin %d)", vi++, v.threads, v.chunk, v.direct, v.gap, v.pin);
    EXPECT(ok, "%s: bulk load failed: %s", what, st.err.c_str());
    if (v.direct && expect_direct()) EXPECT(st.direct_files == st.files && st.files > 0, "%s: O_DIRECT %d/%d files", what, st.direct_files, st.files);
    compare(A, B, fc, what);
  }
  {  // 3. bulk load failure → absorbed by the per-tensor path
    Checkpoint ck(fc.dir);
    ExpertStore B(cfg, L, 0, 1, N_MTP);
    // Checkpoint opens the shards of layer 1 first (mmap — the per-tensor path reads through this mapping). Then one is moved away from its path, so bulk_load fails reopening it by path.
    for (int e = 0; e < E; ++e) for (const char* m : {"w1", "w2", "w3"}) for (const char* k : {"weight", "scale"}) (void)ck.get("layers.1.ffn.experts." + std::to_string(e) + "." + m + "." + k);
    fs::rename(fc.dir + "/s-1.safetensors", fc.dir + "/s-1.safetensors.moved");
    BulkLoadStats st;
    const bool ok = B.load_bulk(ck, L + N_MTP, fc.engram, BulkLoadOpts{}, true, &st);
    fs::rename(fc.dir + "/s-1.safetensors.moved", fc.dir + "/s-1.safetensors");
    EXPECT(!ok && !st.err.empty(), "fallback: bulk load did not fail (ok %d)", ok);
    compare(A, B, fc, "fallback (per-layer reload after a failed bulk load)");
  }
  {  // 5. HIVE_LOAD_PREFAULT: prediction == actual allocation (all taken, nothing freed) · bulk load into that arena == reference
    const auto regions = HostPrefault::predict(fc.dir, -1, false);
    EXPECT(regions.size() == 6, "prefault predict: %zu regions (want 2 arenas + 2 engram tables × 2)", regions.size());
    for (int round = 0; round < 2; ++round) {
      HostPrefault::start(regions, 3);
      Checkpoint ck(fc.dir);
      ExpertStore B(cfg, L, 0, 1, N_MTP);
      BulkLoadOpts o; o.threads = 4; o.chunk = 8192;
      bool ok = true;
      if (round == 0) ok = B.load_bulk(ck, L + N_MTP, fc.engram, o, true);
      else { load_reference(B, ck, fc); }  // the per-tensor path also uses the pre-reserved arena
      const auto [claimed, freed] = HostPrefault::finish();
      size_t want = 0;
      for (auto [n, node] : regions) want += (n + (2u << 20) - 1) / (2u << 20) * (2u << 20);
      EXPECT(ok && claimed == want && freed == 0, "prefault round %d: ok %d claimed %zu (want %zu) released %zu", round, ok, claimed, want, freed);
      compare(A, B, fc, round == 0 ? "prefault + bulk load" : "prefault + per-layer load");
    }
    {  // wrong prediction (+4 MiB per size) → the first take drops the pool and uses normal allocation · same result
      auto wrong = regions;
      for (auto& r : wrong) r.first += 4u << 20;
      HostPrefault::start(wrong, 2);
      Checkpoint ck(fc.dir);
      ExpertStore B(cfg, L, 0, 1, N_MTP);
      const bool ok = B.load_bulk(ck, L + N_MTP, fc.engram, BulkLoadOpts{}, true);
      const auto [claimed, freed] = HostPrefault::finish();
      EXPECT(ok && claimed == 0 && freed == 0, "prefault mispredict: ok %d claimed %zu released at finish %zu (released at the first miss)", ok, claimed, freed);
      compare(A, B, fc, "prefault mispredicted");
    }
    EXPECT(HostPrefault::predict(dir + "/no-such-ckpt", -1, false).empty(), "prefault predict on a missing dir");
  }
  {  // 4. dense prefetch
    EXPECT(dense_tensor_name("layers.3.attn.wq_a.weight", -1) && !dense_tensor_name("layers.3.ffn.experts.1.w1.weight", -1) &&
               !dense_tensor_name("layers.1.engram.embed.weight", -1) && dense_tensor_name("layers.1.engram.q_weight", -1) &&
               !dense_tensor_name("vision.blocks.0.w", -1) && !dense_tensor_name("aligner.w1.bias", -1) && dense_tensor_name("mtp.0.attn.x", -1) &&
               dense_tensor_name("head.weight", -1) && dense_tensor_name("embed.weight", -1) && dense_tensor_name("norm.weight", -1) &&
               !dense_tensor_name("layers.12.attn.x", 9) && dense_tensor_name("layers.9.attn.x", 9),
           "dense_tensor_name filter");
    DensePrefetch pf;
    pf.start(fc.dir, -1, 3);
    pf.join();
    EXPECT(pf.bytes.load() >= fc.dense_bytes, "dense prefetch read %llu < dense %zu", (unsigned long long)pf.bytes.load(), fc.dense_bytes);
    DensePrefetch none;
    none.start(dir + "/no-such-ckpt", -1, 2);
    none.join();
    EXPECT(none.bytes.load() == 0, "dense prefetch on a missing dir read bytes");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir(argc < 2 ? "/tmp" : argv[1]);
  std::mt19937_64 rng(20261001);
  test_bulk(dir, rng);
  test_store(dir, rng);
  test_prefault_race(rng);
  // switch parsing (unset / "" / "0" = off · 1 / other "on" values = 8 · N = N · cap 64)
  struct C { const char* v; int want; };
  for (C c : {C{nullptr, 0}, C{"", 0}, C{"0", 0}, C{"1", 8}, C{"on", 8}, C{"12", 12}, C{"999", 64}}) {
    if (c.v) setenv("HIVE_LOAD_PAR", c.v, 1); else unsetenv("HIVE_LOAD_PAR");
    EXPECT(load_par_threads() == c.want, "HIVE_LOAD_PAR=%s → %d (want %d)", c.v ? c.v : "(unset)", load_par_threads(), c.want);
  }
  unsetenv("HIVE_LOAD_CHUNK_MB"); EXPECT(load_chunk_bytes() == (32ull << 20), "chunk default");
  setenv("HIVE_LOAD_CHUNK_MB", "8", 1); EXPECT(load_chunk_bytes() == (8ull << 20), "chunk 8");
  setenv("HIVE_LOAD_CHUNK_MB", "x", 1); EXPECT(load_chunk_bytes() == (32ull << 20), "chunk non-numeric");
  unsetenv("HIVE_LOAD_BUFFERED"); EXPECT(load_opts_from_env(8).direct, "direct default");
  setenv("HIVE_LOAD_BUFFERED", "1", 1); EXPECT(!load_opts_from_env(8).direct, "buffered switch");
  if (fails) { fprintf(stderr, "test_load_par_cpu: %d failure(s)\n", fails); return 1; }
  printf("test_load_par_cpu: bulk_load == pread · load_bulk == per-layer load (5 variants + fallback) · dense prefetch · switches OK\n");
  return 0;
}
