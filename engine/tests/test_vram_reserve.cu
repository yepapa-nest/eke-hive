// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// vram_reserve (hive/vram_reserve.h) + DevBuf / GrowBuf with HIVE_VRAM_LIMIT_MB held for the process lifetime:
//   ① cap on: allocations and frees keep the process footprint (its buffers + the reserve) constant, small buffers included (credit)
//   ② a refused allocation throws DeviceOutOfMemory instead of aborting, and the reserve is used before giving up
//   ③ cap off (sleep): frees are not refilled
//   ④ a refused request never makes the reserve grow past its share (first cut credited a refused 1 TiB and took the whole card)
//   Run: scripts/hive-run.sh "./build-svc/test_vram_reserve"   (needs ~1 GiB of free VRAM)
#include <cstdio>
#include <vector>

#include "hive/devbuf.h"
#include "hive/grow_buf.h"
#include "hive/vram_reserve.h"

using namespace hive;
namespace vr = hive::vram_reserve;

static size_t dev_free() { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return f; }
static bool check(bool ok, const char* what) { printf("%-72s %s\n", what, ok ? "PASS" : "FAIL"); return ok; }

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  bool all = true;
  cudaFree(nullptr);  // context
  const size_t reserve = 16 * vr::kChunk;  // 512 MiB
  const size_t got = vr::fill(reserve);
  all &= check(got == reserve, "reserve filled in pieces");
  vr::arm();
  // ① footprint constant: allocate 3 x 100 MiB + 50 small 2 MiB buffers, then free them
  const size_t tol = 2 * vr::kChunk;  // a piece rounds up; device granularity
  const size_t f0 = dev_free();
  {
    std::vector<DevBuf> big(3), small(50);
    for (auto& b : big) b.alloc(100ull << 20);
    for (auto& b : small) b.alloc(2ull << 20);
    const size_t f1 = dev_free();
    all &= check(f1 + tol >= f0 && f1 <= f0 + tol, "cap on: 400 MiB of buffers come out of the reserve (device free unchanged)");
    all &= check(vr::held() < reserve, "  … the reserve shrank");
  }
  const size_t f2 = dev_free();
  all &= check(f2 + tol >= f0 && f2 <= f0 + tol, "cap on: freeing refills the reserve (device free unchanged)");
  all &= check(vr::held() + tol >= reserve, "  … the reserve is back");
  // small-buffer churn must not drain the reserve (credit)
  const size_t h_churn = vr::held();
  for (int i = 0; i < 200; ++i) { DevBuf b(2ull << 20); }
  all &= check(vr::held() + tol >= reserve, "200 x (alloc 2 MiB, free) keeps the reserve");
  all &= check(vr::held() == h_churn, "  … inside the one-piece slack (no piece freed and re-taken)");
  // GrowBuf growth also comes from the reserve
  {
    GrowBuf g;
    g.reserve(256ull << 20);
    const size_t fa = dev_free();
    g.ensure(128ull << 20);
    const size_t fb = dev_free();
    all &= check(fb + tol >= fa, "GrowBuf growth comes out of the reserve");
  }
  // ② refused allocation: throws, does not abort
  bool threw = false;
  const size_t fh = dev_free(), hh = vr::held();
  try { DevBuf huge((size_t)1 << 40); } catch (const DeviceOutOfMemory& e) { threw = true; printf("  (%s)\n", e.what()); }
  all &= check(threw, "a refused allocation throws DeviceOutOfMemory (daemon survives)");
  all &= check(dev_free() + tol >= fh && vr::held() <= hh + tol, "④ the refusal leaves the card as it was (reserve not grown past its share)");
  { DevBuf after(64ull << 20); all &= check(after.p != nullptr, "allocations still work after the refusal"); }
  // past the reserve's share (but well within what the card has free): refused without touching the device
  threw = false;
  const size_t fo = dev_free();
  try { DevBuf over(reserve + 4 * vr::kChunk); } catch (const DeviceOutOfMemory& e) { threw = true; printf("  (%s)\n", e.what()); }
  all &= check(threw && dev_free() + tol >= fo, "past the limit: refused although the card has room (the room is a neighbour's)");
  { GrowBuf g; g.reserve(reserve + 8 * vr::kChunk); all &= check(!g.ensure(reserve + 4 * vr::kChunk), "  … GrowBuf growth past the limit refused too"); }
  // ③ cap off: frees are not refilled
  vr::set_cap(false);
  const size_t h0 = vr::held();
  { DevBuf b(128ull << 20); }
  all &= check(vr::held() == h0, "cap off (sleep): a free is not refilled");
  vr::release_all();
  all &= check(vr::held() == 0, "release_all frees every piece");
  // ⑤ ready: the ballast (outside the limit) goes back to the card, the headroom inside the limit becomes the reserve
  const size_t ballast = 12 * vr::kChunk, headroom = 4 * vr::kChunk;
  all &= check(vr::fill(ballast) == ballast, "⑤ startup ballast taken");
  const size_t fb0 = dev_free();
  all &= check(vr::rebase(headroom) == headroom, "  rebase keeps exactly the headroom");
  const size_t fb1 = dev_free();
  all &= check(fb1 + tol >= fb0 + (ballast - headroom) && fb1 <= fb0 + (ballast - headroom) + tol, "  … and the card gets the ballast back (net ballast − headroom)");
  vr::arm();
  { DevBuf b(3 * vr::kChunk); all &= check(dev_free() + tol >= fb1, "  a runtime buffer inside the headroom does not touch the card"); }
  vr::release_all();
  printf("%s\n", all ? "ALL PASS" : "FAIL");
  return all ? 0 : 1;
}
