// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Sparse attention v3 — flash prefill version (P2). Same semantics as v2 (attn_tc.cu):
//   S = bf16 q · bf16 kv (fp32 accumulation) x scale, index -1 is -inf; P = bf16(exp(S - max)); O = sum P·V (fp32 accumulation)
//   · denominator = fp32 sum of exp + exp(sink - max); o = bf16(O / denominator).
// v2 used three kernels that round-trip S[M,H,topk] fp32 through global memory, re-read 32 KB of Q for every block = (query, 32 heads, 32 keys), and gathered
//   keys element by element (kv_elem2: branch + fp4 dequantization) — in an M=2048 profile, 40 layers took 843 ms = 8 TFLOPS (~2% of tensor peak, see docs/performance.md). Here:
//   · block = (query, 32 heads); 8 warps = (16-head row group hq 0..1) x (dimension quarter dq 0..3, 128 dims each)
//   · each warp loads its own (16 heads x 128 dims) slice of Q into registers once (A fragments, 8 k-steps x 4 = 32 registers) — no smem
//   · 16-key blocks are double-buffered with cp.async (bf16 rows as 16B chunks; compressed rows arrive packed, 288B, and are expanded in smem)
//   · S as mma.sync m16n8k16 bf16 fragments (registers) -> online softmax (row max, fp32 sum, rescale accumulated O) -> P fragments used directly as A of P·V.
//   For QK the four dq warps each produce a partial sum over a quarter of the dimensions and exchange them via smem, adding them **in the same order** (all four hold bit-identical S).
// smem 46,080B — WARNING: on sm_120 the per-block dynamic smem limit is 99 KB (measured; not the 227 KB of sm_100), so a 64-head / 32-key / Q-in-smem version (169 KB)
//   failed to launch. SM smem is 100 KB (1 KB reserved per block), so two blocks fit only at <= 50,176B — packed compressed rows are not stored separately but land in
//   the tile row tail (736..1023B) and are expanded in place in two steps (via registers), saving 9 KB.
// Differences from v2 = P is rounded to bf16 against the running max at that point rather than the final max (as in FlashAttention-2), and the fp32 summation order. Compared within tolerance by test_attn_flash.
// Fragment layout (m16n8k16.row.col, lane = 4r+q): A a0=(r, 2q..) a1=(r+8, 2q..) a2=(r, 2q+8..) a3=(r+8, 2q+8..); B b0=(k 2q.., n r) b1=(k 2q+8.., n r)
//   · C c0,c1=(r, 2q..2q+1) c2,c3=(r+8, ...). ldmatrix (non-transposed) yields K^T fragments from an [n][k] tile; ldmatrix.trans yields V fragments from a [k][n] tile.
// v4 (attn_flash64_kernel below): block = (query, 64 heads) — the default of sparse_attn_flash. v3 remains available via HIVE_ATTN_FLASH32=1.
#include <cstdlib>

#include "hive/model_kernels.h"

namespace hive::k {

namespace {
constexpr int D = 512;
constexpr int HG = 32;          // heads per block
constexpr int KB = 16;          // key block
constexpr int THREADS = 256;
constexpr int TS = D + 8;       // smem row stride (elements) — 1040B: the 8 rows of an ldmatrix fall into distinct bank groups (1024B would be an 8-way conflict)
constexpr int PKB = kvp::COMP_ROW;  // 288
constexpr int NW = THREADS / 32;    // 8

struct RowDesc { const void* p; int type; int pad; };  // type 0 none (zero fill, -inf), 1 bf16 row (ring, chunk), 2 packed fp4 compressed row

constexpr int PK_OFF = 736;     // byte offset of the packed row (288B) within a tile row (736 + 288 = 1024 <= 1040, 16B aligned)
constexpr size_t SMEM_KT = (size_t)2 * KB * TS * 2;           // 33,280
constexpr size_t SMEM_XCH = (size_t)NW * 8 * 32 * 4;          // 8,192 (warps x 8 S fragments x lanes)
constexpr size_t SMEM_DESC = (size_t)2 * KB * sizeof(RowDesc);  // 512
constexpr int MAXK = 1024;                                     // columns per query, upper bound (win 128 + index_topk 512 = 640)
constexpr size_t SMEM_IDX = (size_t)MAXK * 4;                  // 4,096 — the index row is loaded into smem at once (a global round trip every 16 keys was on the critical path)
constexpr size_t SMEM_FLASH = SMEM_KT + SMEM_XCH + SMEM_DESC + SMEM_IDX;  // 46,080
static_assert(SMEM_FLASH + 1024 <= 100 * 1024 / 2, "two blocks per SM on sm_120 (100KB SM smem, 1KB reserved per block)");
static_assert(PK_OFF % 16 == 0 && PK_OFF + PKB <= TS * 2, "packed row must fit in the tile row tail");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  const uint32_t ar0 = a[0], ar1 = a[1], ar2 = a[2], ar3 = a[3], br0 = b[0], br1 = b[1];
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(ar0), "r"(ar1), "r"(ar2), "r"(ar3), "r"(br0), "r"(br1));
}
__device__ __forceinline__ uint32_t pack2bf(float a, float b) {  // lower index goes to the low 16 bits
  return (uint32_t)__bfloat16_as_ushort(f2bf(a)) | ((uint32_t)__bfloat16_as_ushort(f2bf(b)) << 16);
}

__global__ void __launch_bounds__(THREADS, 2) attn_flash_kernel(const bf16* __restrict__ q, int H, KvSources kv, const int32_t* __restrict__ idx,
                                                                 int idx_stride, int topk, const float* __restrict__ sink, float scale,
                                                                 bf16* __restrict__ o) {
  extern __shared__ __align__(128) uint8_t smem[];
  bf16* kt = reinterpret_cast<bf16*>(smem);                        // [2][16][TS] — compressed rows first arrive packed in the tail of each row (PK_OFF..)
  float* xch = reinterpret_cast<float*>(kt + 2 * KB * TS);         // [8 warps][8][32]
  RowDesc* desc = reinterpret_cast<RowDesc*>(xch + NW * 8 * 32);   // [2][16]
  int32_t* sidx = reinterpret_cast<int32_t*>(desc + 2 * KB);       // [topk] index row of this query
  const int ngroups = H / HG;
  const int m = blockIdx.x / ngroups, hg = blockIdx.x % ngroups;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int hq = warp & 1, dq = warp >> 1;
  const int q4 = lane & 3, r = lane >> 2;
  const int h0 = hg * HG + hq * 16;         // this warp's 16 head rows
  const int d0 = dq * 128;                  // this warp's dimension quarter
  const int32_t* irow = idx + (size_t)m * idx_stride;
  // Q fragments (registers): 8 k-steps x 4 A fragments — rows r / r+8, columns 2q.. / 2q+8..
  uint32_t qa[8][4];
  {
    const bf16* qr0 = q + ((size_t)m * H + h0 + r) * D + d0 + 2 * q4;
    const bf16* qr8 = qr0 + (size_t)8 * D;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      qa[ks][0] = *reinterpret_cast<const uint32_t*>(qr0 + ks * 16);
      qa[ks][1] = *reinterpret_cast<const uint32_t*>(qr8 + ks * 16);
      qa[ks][2] = *reinterpret_cast<const uint32_t*>(qr0 + ks * 16 + 8);
      qa[ks][3] = *reinterpret_cast<const uint32_t*>(qr8 + ks * 16 + 8);
    }
  }
  const int nkb = (topk + KB - 1) / KB;
  for (int t = tid; t < topk; t += THREADS) sidx[t] = irow[t];
  __syncthreads();
  // Resolve the row sources of key block j (16 threads) and launch the loads (all threads)
  auto stage = [&](int j, int buf) {
    if (tid < KB) {
      const int t = j * KB + tid;
      const int ix = t < topk ? sidx[t] : -1;
      RowDesc d; d.type = 0; d.p = nullptr; d.pad = 0;
      if (ix >= 0) {
        if (ix < kv.win) { d.type = 1; d.p = kv.ring + (size_t)ix * D; }
        else if (ix - kv.win < kv.chunk_len) { d.type = 1; d.p = kv.chunk + (size_t)(ix - kv.win) * D; }
        else { d.type = 2; d.p = kv.comp + (size_t)(ix - kv.win - kv.chunk_len) * PKB; }
      }
      desc[buf * KB + tid] = d;
    }
    __syncthreads();
    bf16* tile = kt + buf * KB * TS;
    for (int i = tid; i < KB * (D / 8); i += THREADS) {
      const int rr = i / (D / 8), c = i % (D / 8);
      const RowDesc d = desc[buf * KB + rr];
      uint8_t* rowb = reinterpret_cast<uint8_t*>(tile + rr * TS);
      if (d.type == 1) cp_async16(rowb + c * 16, reinterpret_cast<const bf16*>(d.p) + c * 8);
      else if (d.type == 2) { if (c < PKB / 16) cp_async16(rowb + PK_OFF + c * 16, reinterpret_cast<const uint8_t*>(d.p) + c * 16); }
      else *reinterpret_cast<uint4*>(rowb + c * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    cp_async_commit();
  };
  // Packed fp4 compressed rows (row tail PK_OFF..) -> in-place expansion into the bf16 tile (same values as the bf16 stored form, kv_pack.h). Threads = 2 (row, 16-dim block) pairs.
  //   (1) load the 8B of nibbles and 1B scale of my two blocks into registers -> sync -> (2) unpack and write: two steps are needed because the outputs of later blocks (b >= 23) overwrite the packed area.
  auto decode = [&](int buf) {
    bf16* tile = kt + buf * KB * TS;
    uint2 nib[2]; uint8_t sc[2]; int rr_[2], b_[2]; bool act[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int idx = tid + i * THREADS;
      rr_[i] = idx / (D / 16); b_[i] = idx % (D / 16);
      act[i] = desc[buf * KB + rr_[i]].type == 2;
      nib[i] = make_uint2(0u, 0u); sc[i] = 0;
      if (act[i]) {
        const uint8_t* pkr = reinterpret_cast<const uint8_t*>(tile + rr_[i] * TS) + PK_OFF;
        nib[i] = *reinterpret_cast<const uint2*>(pkr + kvp::COMP_SCALES + 8 * b_[i]);
        sc[i] = pkr[b_[i]];
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (!act[i]) continue;
      const float s = e4m3_to_f32(sc[i]);
      const uint32_t x[2] = {nib[i].x, nib[i].y};
      float v[16];
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int j = 0; j < 8; ++j) v[8 * k + j] = kvp::e2m1_val((x[k] >> (4 * j)) & 0xF) * s;
      uint4* dst = reinterpret_cast<uint4*>(tile + rr_[i] * TS + b_[i] * 16);
      dst[0] = make_uint4(pack2bf(v[0], v[1]), pack2bf(v[2], v[3]), pack2bf(v[4], v[5]), pack2bf(v[6], v[7]));
      dst[1] = make_uint4(pack2bf(v[8], v[9]), pack2bf(v[10], v[11]), pack2bf(v[12], v[13]), pack2bf(v[14], v[15]));
    }
  };

  float O[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) O[i][0] = O[i][1] = O[i][2] = O[i][3] = 0.f;
  float m_r[2] = {-1e30f, -1e30f}, l_r[2] = {0.f, 0.f};  // rows r and r+8 of this warp's head tile

  if (nkb > 0) stage(0, 0);
  for (int j = 0; j < nkb; ++j) {
    const int buf = j & 1;
    if (j + 1 < nkb) { stage(j + 1, buf ^ 1); cp_async_wait<1>(); }
    else cp_async_wait<0>();
    __syncthreads();
    bool any_comp = false;  // does this block contain compressed rows (desc is in smem — all threads get the same answer, so the __syncthreads inside is uniform)
#pragma unroll
    for (int i = 0; i < KB; ++i) any_comp |= desc[buf * KB + i].type == 2;
    if (any_comp) { decode(buf); __syncthreads(); }  // blocks with only window keys (the first 8 blocks of each query) skip the expansion and its two syncs
    const bf16* tile = kt + buf * KB * TS;
    // (1) S partial sums (dims d0 .. d0+128): 16 heads x 16 keys = 2 n-tiles
    float S[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) S[i][0] = S[i][1] = S[i][2] = S[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int k0 = d0 + ks * 16;
      uint32_t bb[4];
      ldmatrix_x4(bb, tile + ((lane & 7) + 8 * (lane >> 4)) * TS + k0 + 8 * ((lane >> 3) & 1));
      const uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
      mma_bf16(S[0], qa[ks], b0);
      mma_bf16(S[1], qa[ks], b1);
    }
    // (2) combine the partial sums of the four dq warps in the same order (dq 0->3) — all four get bit-identical S
    {
      float* mine = xch + (size_t)warp * 8 * 32;
#pragma unroll
      for (int i = 0; i < 8; ++i) mine[i * 32 + lane] = S[i >> 2][i & 3];
      __syncthreads();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float acc = xch[(size_t)(hq + 0) * 8 * 32 + i * 32 + lane];
        acc += xch[(size_t)(hq + 2) * 8 * 32 + i * 32 + lane];
        acc += xch[(size_t)(hq + 4) * 8 * 32 + i * 32 + lane];
        acc += xch[(size_t)(hq + 6) * 8 * 32 + i * 32 + lane];
        S[i >> 2][i & 3] = acc;
      }
    }
    // (3) scale; missing keys -inf
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int k0 = nt * 8 + 2 * q4;
      const bool v0 = desc[buf * KB + k0].type != 0, v1 = desc[buf * KB + k0 + 1].type != 0;
      S[nt][0] = v0 ? S[nt][0] * scale : -INFINITY;
      S[nt][1] = v1 ? S[nt][1] * scale : -INFINITY;
      S[nt][2] = v0 ? S[nt][2] * scale : -INFINITY;
      S[nt][3] = v1 ? S[nt][3] * scale : -INFINITY;
    }
    // (4) online softmax (rows r, r+8): max -> rescale factor -> p (fp32 sum) -> P bf16 fragments (one k-step = 16 keys)
    float mx0 = fmaxf(fmaxf(S[0][0], S[0][1]), fmaxf(S[1][0], S[1][1]));
    float mx1 = fmaxf(fmaxf(S[0][2], S[0][3]), fmaxf(S[1][2], S[1][3]));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float mn0 = fmaxf(m_r[0], mx0), mn1 = fmaxf(m_r[1], mx1);
    const float al0 = expf(m_r[0] - mn0), al1 = expf(m_r[1] - mn1);
    const float p00 = expf(S[0][0] - mn0), p01 = expf(S[0][1] - mn0), p02 = expf(S[0][2] - mn1), p03 = expf(S[0][3] - mn1);
    const float p10 = expf(S[1][0] - mn0), p11 = expf(S[1][1] - mn0), p12 = expf(S[1][2] - mn1), p13 = expf(S[1][3] - mn1);
    float sum0 = p00 + p01 + p10 + p11, sum1 = p02 + p03 + p12 + p13;
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1); sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1); sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
    l_r[0] = l_r[0] * al0 + sum0; l_r[1] = l_r[1] * al1 + sum1;
    m_r[0] = mn0; m_r[1] = mn1;
    const uint32_t P[4] = {pack2bf(p00, p01), pack2bf(p02, p03), pack2bf(p10, p11), pack2bf(p12, p13)};  // a0(r,k0..) a1(r+8) a2(r,k8..) a3(r+8)
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) { O[nt][0] *= al0; O[nt][1] *= al0; O[nt][2] *= al1; O[nt][3] *= al1; }
    // (5) O (16 heads x 128 dims) += P (16x16) · V (16x128): 8 pairs of 16-dim columns
#pragma unroll
    for (int np = 0; np < 8; ++np) {
      const int n0 = d0 + np * 16;
      uint32_t bb[4];
      ldmatrix_x4_trans(bb, tile + (lane & 15) * TS + n0 + 8 * (lane >> 4));
      const uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
      mma_bf16(O[np * 2], P, b0);
      mma_bf16(O[np * 2 + 1], P, b1);
    }
    __syncthreads();  // before the next stage overwrites the tile (buf) and xch
  }
  // Epilogue: denominator = fp32 sum + exp(sink - max), bf16 output
  const float den0 = l_r[0] + expf(sink[h0 + r] - m_r[0]);
  const float den1 = l_r[1] + expf(sink[h0 + r + 8] - m_r[1]);
  bf16* om = o + ((size_t)m * H + h0 + r) * D + d0 + 2 * q4;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    *reinterpret_cast<uint32_t*>(om + nt * 8) = pack2bf(O[nt][0] / den0, O[nt][1] / den0);
    *reinterpret_cast<uint32_t*>(om + (size_t)8 * D + nt * 8) = pack2bf(O[nt][2] / den1, O[nt][3] / den1);
  }
}

// ---- v4: block = (query, 64 heads); 16 warps = (16-head row group hq 0..3) x (dimension quarter dq 0..3); 512 threads ----------------------
// In v3 the two 32-head blocks of the same query each gathered (cp.async) and expanded (fp4) the key tiles **separately** — twice the loads, expansions and syncs per query (estimated 1.5–2x).
//   · The work of one warp (16 heads x 128 dims; per 16 keys, 16 QK mma + 16 PV mma) is the same as in v3 -> the register shape (Q 32 + O 64) is the same too.
//     512 threads x 2 blocks would leave 64 registers per thread, too few for Q+O (96) -> 1 block/SM (<= 128 registers). 16 warps per SM equals v3 (8 x 2 blocks).
//   · Since two blocks no longer hide each other's syncs, the pipeline is deeper: a 3-stage buffer of 16-key tiles (loads run 2 tiles ahead) +
//     compressed-row expansion **one tile ahead** (expand j+1 right after the PV of j). Packed rows go to separate staging (3 x 4,608B) instead of the tile tail,
//     which removes the two-step sync of in-place expansion. Row sources (ring/chunk/comp) are computed per thread directly from the smem index row — the desc table and its sync are gone.
//   · Syncs: per 16 keys, v3's 4–6 (desc, wait, expand x2, exchange, end) -> 2 (exchange, end); key loads and expansions per query are halved.
//   · KB=32 does not fit: tiles 33,280B x 2 stages + S exchange 32 KB (16 warps x 16 fragments x lanes) + 4 KB index > 99 KB (sm_120 block limit).
//   · smem 84,480B (<= 99 KB opt-in) — registers already limit this to 1 block/SM, so there is no reason to squeeze smem below 50 KB.
// Numerics: per head row, the QK partial-sum order (dq 0->3), online softmax, P rounding and PV order are the same as v3 -> **bit-identical** to v3 (test_attn_flash compares bitwise).
namespace f64 {
constexpr int HG = 64, THREADS = 512, NW = THREADS / 32, NST = 3;
constexpr size_t TILE_B = (size_t)KB * TS * 2;                 // 16,640
constexpr size_t PK_B = (size_t)KB * PKB;                      // 4,608
constexpr size_t SMEM_KT = (size_t)NST * TILE_B;               // 49,920
constexpr size_t SMEM_PK = (size_t)NST * PK_B;                 // 13,824
constexpr size_t SMEM_XCH = (size_t)NW * 8 * 32 * 4;           // 16,384
constexpr int MAXKB = MAXK / KB;                               // 64
constexpr size_t SMEM = SMEM_KT + SMEM_PK + SMEM_XCH + (size_t)MAXK * 4 + (size_t)MAXKB * 4;  // 84,480
static_assert(SMEM <= 99 * 1024, "sm_120 opt-in dynamic smem per block is 99KB");
static_assert(KB * (D / 8) == 2 * THREADS && KB * (D / 16) == THREADS, "stage = 2 chunks per thread, decode = 1 block per thread");
}  // namespace f64

__global__ void __launch_bounds__(f64::THREADS, 1) attn_flash64_kernel(const bf16* __restrict__ q, int H, KvSources kv, const int32_t* __restrict__ idx,
                                                                         int idx_stride, int topk, const float* __restrict__ sink, float scale,
                                                                         bf16* __restrict__ o) {
  // shadow v3's constants of the same names (block-scoped declarations)
  constexpr int HG = f64::HG, THREADS = f64::THREADS, NW = f64::NW, NST = f64::NST;
  constexpr size_t SMEM_KT = f64::SMEM_KT, SMEM_PK = f64::SMEM_PK, PK_B = f64::PK_B;
  extern __shared__ __align__(128) uint8_t smem[];
  bf16* kt = reinterpret_cast<bf16*>(smem);                                   // [3][16][TS] bf16 tiles
  uint8_t* pk = smem + SMEM_KT;                                               // [3][16][288] packed compressed-row staging
  float* xch = reinterpret_cast<float*>(pk + SMEM_PK);                        // [16 warps][8][32]
  int32_t* sidx = reinterpret_cast<int32_t*>(xch + NW * 8 * 32);              // [MAXK] index row of this query
  int32_t* tflag = sidx + MAXK;                                               // [MAXKB] does the tile contain compressed rows
  const int ngroups = H / HG;
  const int m = blockIdx.x / ngroups, hg = blockIdx.x % ngroups;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int hq = warp & 3, dq = warp >> 2;
  const int q4 = lane & 3, r = lane >> 2;
  const int h0 = hg * HG + hq * 16;
  const int d0 = dq * 128;
  const int wc = kv.win + kv.chunk_len;  // compressed rows start at this global index
  const int32_t* irow = idx + (size_t)m * idx_stride;
  uint32_t qa[8][4];
  {
    const bf16* qr0 = q + ((size_t)m * H + h0 + r) * D + d0 + 2 * q4;
    const bf16* qr8 = qr0 + (size_t)8 * D;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      qa[ks][0] = *reinterpret_cast<const uint32_t*>(qr0 + ks * 16);
      qa[ks][1] = *reinterpret_cast<const uint32_t*>(qr8 + ks * 16);
      qa[ks][2] = *reinterpret_cast<const uint32_t*>(qr0 + ks * 16 + 8);
      qa[ks][3] = *reinterpret_cast<const uint32_t*>(qr8 + ks * 16 + 8);
    }
  }
  const int nkb = (topk + KB - 1) / KB;
  for (int t = tid; t < topk; t += THREADS) sidx[t] = irow[t];
  __syncthreads();
  if (tid < nkb) {  // per-tile compressed-row flags — visible at the first expansion after the prologue sync
    int f = 0;
#pragma unroll
    for (int i = 0; i < KB; ++i) { const int t = tid * KB + i; f |= (t < topk && sidx[t] >= wc) ? 1 : 0; }
    tflag[tid] = f;
  }
  auto key_ix = [&](int t) { return t < topk ? sidx[t] : -1; };
  // Launch the loads of tile j (for j >= nkb only an empty group is committed — keeps the wait count uniform). Threads = (row i/64, 16B chunk i%64) x 2
  auto stage = [&](int j, int buf) {
    if (j < nkb) {
      uint8_t* tile = reinterpret_cast<uint8_t*>(kt + (size_t)buf * KB * TS);
      uint8_t* pkb = pk + (size_t)buf * PK_B;
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int i = tid + s * THREADS, rr = i >> 6, c = i & 63;
        const int ix = key_ix(j * KB + rr);
        uint8_t* rowb = tile + (size_t)rr * TS * 2;
        if (ix < 0) *reinterpret_cast<uint4*>(rowb + c * 16) = make_uint4(0u, 0u, 0u, 0u);  // missing key = zero row (even with P = 0, garbage V would give NaN)
        else if (ix < kv.win) cp_async16(rowb + c * 16, kv.ring + (size_t)ix * D + c * 8);
        else if (ix < wc) cp_async16(rowb + c * 16, kv.chunk + (size_t)(ix - kv.win) * D + c * 8);
        else if (c < PKB / 16) cp_async16(pkb + rr * PKB + c * 16, kv.comp + (size_t)(ix - wc) * PKB + c * 16);
      }
    }
    cp_async_commit();
  };
  // Expand compressed rows of tile j (staging -> bf16 tile rows): threads = (row tid/32, 16-dim block tid%32). Same formula as v3 decode.
  auto decode = [&](int j, int buf) {
    if (!tflag[j]) return;  // tiles with only window keys (the first 8 of each query) are skipped — there is no sync inside, so per-thread divergence is safe
    const int rr = tid >> 5, b = tid & 31;
    if (key_ix(j * KB + rr) < wc) return;
    const uint8_t* pkr = pk + (size_t)buf * PK_B + rr * PKB;
    const uint2 nb = *reinterpret_cast<const uint2*>(pkr + kvp::COMP_SCALES + 8 * b);
    const float s = e4m3_to_f32(pkr[b]);
    const uint32_t x[2] = {nb.x, nb.y};
    float v[16];
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) v[8 * k + jj] = kvp::e2m1_val((x[k] >> (4 * jj)) & 0xF) * s;
    uint4* dst = reinterpret_cast<uint4*>(kt + (size_t)buf * KB * TS + rr * TS + b * 16);
    dst[0] = make_uint4(pack2bf(v[0], v[1]), pack2bf(v[2], v[3]), pack2bf(v[4], v[5]), pack2bf(v[6], v[7]));
    dst[1] = make_uint4(pack2bf(v[8], v[9]), pack2bf(v[10], v[11]), pack2bf(v[12], v[13]), pack2bf(v[14], v[15]));
  };

  float O[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) O[i][0] = O[i][1] = O[i][2] = O[i][3] = 0.f;
  float m_r[2] = {-1e30f, -1e30f}, l_r[2] = {0.f, 0.f};

  // Prologue: load tiles 0 and 1 -> tile 0 arrives and is expanded
  stage(0, 0);
  stage(1, 1);
  cp_async_wait<1>();
  __syncthreads();
  if (nkb > 0) decode(0, 0);
  __syncthreads();
  int buf = 0;
  for (int j = 0; j < nkb; ++j) {
    const int nbuf = buf == NST - 1 ? 0 : buf + 1, fbuf = nbuf == NST - 1 ? 0 : nbuf + 1;
    stage(j + 2, fbuf);  // fbuf = buffer of tile j-1 — after the sync at the end of the previous iteration nobody reads it
    const bf16* tile = kt + (size_t)buf * KB * TS;
    // (1) S partial sums (dims d0 .. d0+128): 16 heads x 16 keys
    float S[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) S[i][0] = S[i][1] = S[i][2] = S[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int k0 = d0 + ks * 16;
      uint32_t bb[4];
      ldmatrix_x4(bb, tile + ((lane & 7) + 8 * (lane >> 4)) * TS + k0 + 8 * ((lane >> 3) & 1));
      const uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
      mma_bf16(S[0], qa[ks], b0);
      mma_bf16(S[1], qa[ks], b1);
    }
    {
      float* mine = xch + (size_t)warp * 8 * 32;
#pragma unroll
      for (int i = 0; i < 8; ++i) mine[i * 32 + lane] = S[i >> 2][i & 3];
    }
    cp_async_wait<1>();  // tile j+1 arrived (my share) — after the sync below everyone's share is visible
    __syncthreads();
    // (2) partial sums of the four dq warps in order dq 0->3 (same order as v3 — bit-identical)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float acc = xch[(size_t)(hq + 0) * 8 * 32 + i * 32 + lane];
      acc += xch[(size_t)(hq + 4) * 8 * 32 + i * 32 + lane];
      acc += xch[(size_t)(hq + 8) * 8 * 32 + i * 32 + lane];
      acc += xch[(size_t)(hq + 12) * 8 * 32 + i * 32 + lane];
      S[i >> 2][i & 3] = acc;
    }
    // (3) scale; missing keys -inf
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int k0 = j * KB + nt * 8 + 2 * q4;
      const bool v0 = key_ix(k0) >= 0, v1 = key_ix(k0 + 1) >= 0;
      S[nt][0] = v0 ? S[nt][0] * scale : -INFINITY;
      S[nt][1] = v1 ? S[nt][1] * scale : -INFINITY;
      S[nt][2] = v0 ? S[nt][2] * scale : -INFINITY;
      S[nt][3] = v1 ? S[nt][3] * scale : -INFINITY;
    }
    // (4) online softmax (same formula as v3)
    float mx0 = fmaxf(fmaxf(S[0][0], S[0][1]), fmaxf(S[1][0], S[1][1]));
    float mx1 = fmaxf(fmaxf(S[0][2], S[0][3]), fmaxf(S[1][2], S[1][3]));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float mn0 = fmaxf(m_r[0], mx0), mn1 = fmaxf(m_r[1], mx1);
    const float al0 = expf(m_r[0] - mn0), al1 = expf(m_r[1] - mn1);
    const float p00 = expf(S[0][0] - mn0), p01 = expf(S[0][1] - mn0), p02 = expf(S[0][2] - mn1), p03 = expf(S[0][3] - mn1);
    const float p10 = expf(S[1][0] - mn0), p11 = expf(S[1][1] - mn0), p12 = expf(S[1][2] - mn1), p13 = expf(S[1][3] - mn1);
    float sum0 = p00 + p01 + p10 + p11, sum1 = p02 + p03 + p12 + p13;
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1); sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1); sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
    l_r[0] = l_r[0] * al0 + sum0; l_r[1] = l_r[1] * al1 + sum1;
    m_r[0] = mn0; m_r[1] = mn1;
    const uint32_t P[4] = {pack2bf(p00, p01), pack2bf(p02, p03), pack2bf(p10, p11), pack2bf(p12, p13)};
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) { O[nt][0] *= al0; O[nt][1] *= al0; O[nt][2] *= al1; O[nt][3] *= al1; }
    // (5) O (16 heads x 128 dims) += P · V
#pragma unroll
    for (int np = 0; np < 8; ++np) {
      const int n0 = d0 + np * 16;
      uint32_t bb[4];
      ldmatrix_x4_trans(bb, tile + (lane & 15) * TS + n0 + 8 * (lane >> 4));
      const uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
      mma_bf16(O[np * 2], P, b0);
      mma_bf16(O[np * 2 + 1], P, b1);
    }
    // (6) expansion one tile ahead: compressed rows of tile j+1 (nbuf) — nbuf is read from the next iteration on
    if (j + 1 < nkb) decode(j + 1, nbuf);
    __syncthreads();  // publish the expansion results; before xch and buf are reused
    buf = nbuf;
  }
  cp_async_wait<0>();  // drain the remaining empty groups (exit with no pending cp.async)
  const float den0 = l_r[0] + expf(sink[h0 + r] - m_r[0]);
  const float den1 = l_r[1] + expf(sink[h0 + r + 8] - m_r[1]);
  bf16* om = o + ((size_t)m * H + h0 + r) * D + d0 + 2 * q4;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    *reinterpret_cast<uint32_t*>(om + nt * 8) = pack2bf(O[nt][0] / den0, O[nt][1] / den0);
    *reinterpret_cast<uint32_t*>(om + (size_t)8 * D + nt * 8) = pack2bf(O[nt][2] / den1, O[nt][3] / den1);
  }
}
}  // namespace

// v3 (32-head blocks) — kept for A/B via HIVE_ATTN_FLASH32=1
void sparse_attn_flash_v3(const bf16* q, int M, int H, int Dd, const KvSources& kv, const int32_t* idx, int idx_stride, int topk, const float* sink,
                          float scale, bf16* o, cudaStream_t st) {
  HIVE_CHECK(Dd == D && H % HG == 0 && topk <= MAXK, "sparse_attn_flash shape (D 512 · H % 32 · topk ≤ 1024)");
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(attn_flash_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_FLASH));
    configured = true;
  }
  if (M <= 0) return;
  attn_flash_kernel<<<M * (H / HG), THREADS, SMEM_FLASH, st>>>(q, H, kv, idx, idx_stride, topk, sink, scale, o);
}

void sparse_attn_flash64(const bf16* q, int M, int H, int Dd, const KvSources& kv, const int32_t* idx, int idx_stride, int topk, const float* sink,
                         float scale, bf16* o, cudaStream_t st) {
  HIVE_CHECK(Dd == D && H % f64::HG == 0 && topk <= MAXK, "sparse_attn_flash64 shape (D 512 · H % 64 · topk ≤ 1024)");
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(attn_flash64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f64::SMEM));
    configured = true;
  }
  if (M <= 0) return;
  attn_flash64_kernel<<<M * (H / f64::HG), f64::THREADS, f64::SMEM, st>>>(q, H, kv, idx, idx_stride, topk, sink, scale, o);
}

// Default = v4 (64-head blocks, H % 64 == 0); HIVE_ATTN_FLASH32=1 selects v3. The two must be bit-identical (test_attn_flash).
//   v4 is expected not to lose at small M either: the work per warp is the same, so one block takes about as long as one v3 block, with half as many blocks (not measured — confirm by A/B).
void sparse_attn_flash(const bf16* q, int M, int H, int Dd, const KvSources& kv, const int32_t* idx, int idx_stride, int topk, const float* sink,
                       float scale, bf16* o, cudaStream_t st) {
  static const bool force32 = getenv("HIVE_ATTN_FLASH32") && atoi(getenv("HIVE_ATTN_FLASH32")) != 0;
  if (!force32 && H % f64::HG == 0) sparse_attn_flash64(q, M, H, Dd, kv, idx, idx_stride, topk, sink, scale, o, st);
  else sparse_attn_flash_v3(q, M, H, Dd, kv, idx, idx_stride, topk, sink, scale, o, st);
}

}  // namespace hive::k
