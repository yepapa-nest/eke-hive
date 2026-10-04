// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM routed MoE with FP8 (128×128 block-scale) experts for decode rows M ≤ 4 — contract in include/hive/glm/glm_moe_fp8.h, validation and
//   timings in tests/test_glm_moe_fp8.cu.
// Why: the MTP draft step ran the 8 routed experts as a host round trip (router → stream sync → per expert glm_shared_fp8_decode (2
//   launches) + accum_scaled) = 1 sync + 24 launches per row, each expert's 25.2 MB streamed by a grid that drains before the next starts.
//   Here: launch 1 streams gate/up of every chosen expert in one grid, launch 2 streams every down matrix and accumulates in fixed order.
// Provenance: original code for this repository. The FP8 helpers (ld_stream, e4m3x16, bf16x8_to_f, fp8_chunk_dot, fp8_scale, xor_sum,
//   swiglu_bf), launch_pdl and the per-warp gate/up and down bodies are copied from engine/src/glm/glm_decode.cu (this repository, MIT) so
//   that the arithmetic is bit-identical to glm_shared_fp8_decode. No external code.
// Negative controls (each must make tests/test_glm_moe_fp8.cu FAIL): -DHIVE_GLM_MOE_FP8_NEG_SCALE (wrong block-scale column) ·
//   -DHIVE_GLM_MOE_FP8_NEG_ID (expert id + 1 looked up) · -DHIVE_GLM_MOE_FP8_NEG_CLAMP (SwiGLU clamp skipped).
#include "hive/devmem.h"
#include "hive/glm/glm_moe_fp8.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdlib>
#include <utility>

#include "hive/common.h"

namespace hive::glm {

namespace {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_stream(const void* p) {  // weights: read once, no L1 allocation
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

template <class... KArgs, class... Args>
void launch_pdl(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  static const bool no_pdl = [] { const char* v = getenv("HIVE_GLM_DEC_NO_PDL"); return v && *v && *v != '0'; }();  // A/B switch
  cfg.attrs = at; cfg.numAttrs = no_pdl ? 0 : 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, std::forward<Args>(args)...));
}

// ---- FP8 helpers (copied from glm_decode.cu) ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void e4m3x16(const uint4 q, float (&w)[16]) {
  const uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t h0, h1;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"((uint16_t)(words[j] & 0xFFFFu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"((uint16_t)(words[j] >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&h0)), b = __half22float2(*reinterpret_cast<const __half2*>(&h1));
    w[4 * j] = a.x; w[4 * j + 1] = a.y; w[4 * j + 2] = b.x; w[4 * j + 3] = b.y;
  }
}
__device__ __forceinline__ void bf16x8_to_f(const uint4 v, float* f) {  // f[0..7] = 8 bf16 (exact widening)
  const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) { f[2 * k] = __uint_as_float(u[k] << 16); f[2 * k + 1] = __uint_as_float(u[k] & 0xFFFF0000u); }
}
// fp8b_gemv's per-chunk dot: d = Σ_{i<16} w[i]·x[i] (fmaf in column order) for 16 bf16 activations at xp
__device__ __forceinline__ float fp8_chunk_dot(const float (&w)[16], const bf16* xp) {
  const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(xp)), a1 = __ldg(reinterpret_cast<const uint4*>(xp) + 1);
  const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
  const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
  float d = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 f = __bfloat1622float2(p0[i]); d = fmaf(w[2 * i], f.x, d); d = fmaf(w[2 * i + 1], f.y, d); }
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 f = __bfloat1622float2(p1[i]); d = fmaf(w[8 + 2 * i], f.x, d); d = fmaf(w[8 + 2 * i + 1], f.y, d); }
  return d;
}
__device__ __forceinline__ float fp8_scale(const float* srow, int c) {  // chunk c = 16 columns → block column c / 8
#ifdef HIVE_GLM_MOE_FP8_NEG_SCALE
  return __ldg(srow + (c >> 4));  // negative control: wrong block-scale column
#else
  return __ldg(srow + (c >> 3));
#endif
}
__device__ __forceinline__ float xor_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
// == swiglu_rows_kernel on bf16-rounded gate/up values
__device__ __forceinline__ float swiglu_bf(float g, float u, float limit) {
  g = bf2f(f2bf(g)); u = bf2f(f2bf(u));
#ifndef HIVE_GLM_MOE_FP8_NEG_CLAMP
  if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
#endif
  return g / (1.f + __expf(-g)) * u;
}

// ---- slot table (warp-wide, MK ≤ 32 slots: lane t ↔ slot t = m·K + k) -------------------------------------------------------------------------
// key: the expert id for a valid slot, a value no id can take for an invalid one (unique per lane, so invalid slots never share).
// share: lanes (slots) with the same key; the LEADER of a group = its lowest slot — it alone streams that expert's weights.
struct Slots {
  int key;          // this lane's key
  unsigned share;   // this lane's group
  unsigned valid;   // slots with an id in [0, E)
  unsigned leaders; // valid group leaders
};
__device__ __forceinline__ Slots read_slots(const int32_t* ids, int MK, int E) {
  const int lane = threadIdx.x & 31;
  const int id = lane < MK ? ids[lane] : -1;
  const bool ok = lane < MK && id >= 0 && id < E;
  Slots s;
  s.key = ok ? id : (int)(0x40000000u | (unsigned)lane);
  s.share = __match_any_sync(0xffffffffu, s.key);
  s.valid = __ballot_sync(0xffffffffu, ok);
  s.leaders = __ballot_sync(0xffffffffu, ok && (__ffs(s.share) - 1) == lane);
  return s;
}
__device__ __forceinline__ int table_id(int e, int E) {
#ifdef HIVE_GLM_MOE_FP8_NEG_ID
  return (e + 1) % E;  // negative control: off-by-one expert lookup
#else
  (void)E;
  return e;
#endif
}

// ---- launch 1: gate/up + SwiGLU. block = (8 rows of I, slot p); warp = row i; lane chunks c = lane + 32·ii (16 columns each) ---------------
// Only the leader slot's blocks work; they compute every row m whose slot shares the expert and write y[s, i] for each such slot s.
#ifndef HIVE_GLM_MOE_FP8_GU_MIN
#define HIVE_GLM_MOE_FP8_GU_MIN 2
#endif
template <int MM, int CPL>
__global__ void __launch_bounds__(256, HIVE_GLM_MOE_FP8_GU_MIN) moe_gu_kernel(const Fp8BMat* __restrict__ tab, int E, const int32_t* __restrict__ ids, int MK,
                                                                              int K, const bf16* __restrict__ x, int ldx, float limit, bf16* __restrict__ y) {
  pdl_wait();     // ids / x come from the predecessor (the router's tail triggers before it writes them)
  pdl_trigger();  // after the wait: launch 2 may read ids before its own wait (every block of this grid has seen the predecessor complete)
  const int p = blockIdx.y;
  const Slots S = read_slots(ids, MK, E);
  const unsigned grp = __shfl_sync(0xffffffffu, S.share, p);
  if (!((S.leaders >> p) & 1u)) return;  // block-uniform
  const int e = table_id(__shfl_sync(0xffffffffu, S.key, p), E);
  unsigned rows = 0;
  for (unsigned g = grp; g; g &= g - 1) rows |= 1u << ((__ffs(g) - 1) / K);

  const int lane = threadIdx.x & 31;
  const Fp8BMat G = tab[3 * e], U = tab[3 * e + 1];
  const int i = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (i >= G.N) return;
  const int Kd = G.K, kb_n = (Kd + 127) / 128;
  const uint4* wg = reinterpret_cast<const uint4*>(G.w + (size_t)i * Kd);
  const uint4* wu = reinterpret_cast<const uint4*>(U.w + (size_t)i * Kd);
  const float* sgr = G.s + (size_t)(i / 128) * kb_n;
  const float* sur = U.s + (size_t)(i / 128) * kb_n;
  uint4 qg[CPL], qu[CPL];
  float sg[CPL], su[CPL];
#pragma unroll
  for (int ii = 0; ii < CPL; ++ii) {
    const int c = lane + 32 * ii;
    qg[ii] = ld_stream(wg + c); qu[ii] = ld_stream(wu + c);
    sg[ii] = fp8_scale(sgr, c); su[ii] = fp8_scale(sur, c);
  }
  float ag[MM], au[MM];
#pragma unroll
  for (int m = 0; m < MM; ++m) { ag[m] = 0.f; au[m] = 0.f; }
#pragma unroll
  for (int ii = 0; ii < CPL; ++ii) {
    const int c = lane + 32 * ii;
    float fg[16], fu[16];
    e4m3x16(qg[ii], fg);
    e4m3x16(qu[ii], fu);
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      if (!((rows >> m) & 1u)) continue;  // warp-uniform
      const bf16* xp = x + (size_t)m * ldx + c * 16;
      const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(xp)), a1 = __ldg(reinterpret_cast<const uint4*>(xp) + 1);
      float xf[16];
      bf16x8_to_f(a0, xf); bf16x8_to_f(a1, xf + 8);
      float dg = 0.f, du = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) { dg = fmaf(fg[j], xf[j], dg); du = fmaf(fu[j], xf[j], du); }
      ag[m] = fmaf(dg, sg[ii], ag[m]);
      au[m] = fmaf(du, su[ii], au[m]);
    }
  }
  const int I = G.N;
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    if (!((rows >> m) & 1u)) continue;
    const float gv = xor_sum(ag[m]), uv = xor_sum(au[m]);
    if (lane == 0) {
      const bf16 v = f2bf(swiglu_bf(gv, uv, limit));
      for (unsigned g = grp; g; g &= g - 1) {
        const int s = __ffs(g) - 1;
        if (s / K == m) y[(size_t)s * I + i] = v;
      }
    }
  }
}

// ---- launch 2: down + weighted accumulate. block = R output rows n0..n0+R-1; the j-th distinct expert (slot order) goes to warp j % 8,
//   which streams its R down rows (prefetched before the PDL wait) and dots them with y of every slot in the group → z[s][r] (bf16-rounded,
//   = glm_shared_fp8_decode's output). Then thread (m, r) does out[m, n0+r] = fmaf(w[m,k], z[m·K+k][r], ·) for k = 0..K-1 in order.
#ifndef HIVE_GLM_MOE_FP8_DN_MIN
#define HIVE_GLM_MOE_FP8_DN_MIN 1  // measured M=1 (whole op): 1 → 133-135 µs (no spill), 2 → 143-144 µs (128 regs, 224 B spill)
#endif
template <int MM, int CPL, int R>
__global__ void __launch_bounds__(256, HIVE_GLM_MOE_FP8_DN_MIN) moe_down_kernel(const Fp8BMat* __restrict__ tab, int E, const int32_t* __restrict__ ids,
                                                                                const float* __restrict__ wts, int MK, int K, const bf16* __restrict__ y,
                                                                                float* __restrict__ out, int H) {
  __shared__ float z[32][R];
  pdl_trigger();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n0 = blockIdx.x * R;
  // ids are complete here: launch 1 triggers only after its own wait on their producer (see moe_gu_kernel).
  const Slots S = read_slots(ids, MK, E);
  unsigned L = S.leaders;
  for (int j = 0; j < warp && L; ++j) L &= L - 1;
  uint4 q[R][CPL];
  float sc[R][CPL];
  int Kd = 0;
  auto load = [&](int p) {
    const Fp8BMat D = tab[3 * table_id(__shfl_sync(0xffffffffu, S.key, p), E) + 2];
    Kd = D.K;
    const int kb_n = (Kd + 127) / 128;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = n0 + r;
      const uint4* wr = reinterpret_cast<const uint4*>(D.w + (size_t)n * Kd);
      const float* sr = D.s + (size_t)(n / 128) * kb_n;
#pragma unroll
      for (int ii = 0; ii < CPL; ++ii) { const int c = lane + 32 * ii; q[r][ii] = ld_stream(wr + c); sc[r][ii] = fp8_scale(sr, c); }
    }
  };
  int p = L ? __ffs(L) - 1 : -1;
  if (p >= 0) load(p);
  pdl_wait();  // y (launch 1) and out (earlier kernels) complete after this
  while (p >= 0) {
    const unsigned grp = __shfl_sync(0xffffffffu, S.share, p);
    for (unsigned g = grp; g; g &= g - 1) {
      const int s = __ffs(g) - 1;
      const bf16* ys = y + (size_t)s * Kd;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float acc = 0.f;
#pragma unroll
        for (int ii = 0; ii < CPL; ++ii) {
          const int c = lane + 32 * ii;
          float wf[16];
          e4m3x16(q[r][ii], wf);
          acc = fmaf(fp8_chunk_dot(wf, ys + c * 16), sc[r][ii], acc);
        }
        const float v = xor_sum(acc);
        if (lane == 0) z[s][r] = bf2f(f2bf(v));
      }
    }
    for (int j = 0; j < 8 && L; ++j) L &= L - 1;
    p = L ? __ffs(L) - 1 : -1;
    if (p >= 0) load(p);
  }
  __syncthreads();
  if (threadIdx.x < MM * R) {
    const int m = threadIdx.x / R, r = threadIdx.x % R;
    float* o = out + (size_t)m * H + n0 + r;
    float a = *o;
    for (int k = 0; k < K; ++k) {
      const int s = m * K + k;
      if ((S.valid >> s) & 1u) a = fmaf(wts[s], z[s][r], a);  // == accum_scaled: acc += w · bf16(y) (contracted to one fma)
    }
    *o = a;
  }
}

#define HIVE_MOE_DISPATCH_M(M, CALL) \
  switch (M) {                       \
    case 1: CALL(1); break;          \
    case 2: CALL(2); break;          \
    case 3: CALL(3); break;          \
    default: CALL(4); break;         \
  }

#ifndef HIVE_GLM_MOE_FP8_DN_ROWS
#define HIVE_GLM_MOE_FP8_DN_ROWS 4  // measured M=1/2/4 (µs): 4 → 133/184/262 · 2 → 135/188/266 · 8 → 135/186/268
#endif
constexpr int kDownRows = HIVE_GLM_MOE_FP8_DN_ROWS;

}  // namespace

Fp8BMat* glm_moe_fp8_table_create(const Fp8BMat* host_triples, int E, int H, int I) {
  HIVE_CHECK(E > 0 && host_triples, "glm_moe_fp8_table_create: empty table");
  for (int e = 0; e < E; ++e) {
    const Fp8BMat* t = host_triples + 3 * e;
    HIVE_CHECK(t[0].w && t[0].s && t[1].w && t[1].s && t[2].w && t[2].s, "glm_moe_fp8_table_create: null matrix");
    HIVE_CHECK(t[0].N == I && t[0].K == H && t[1].N == I && t[1].K == H && t[2].N == H && t[2].K == I, "glm_moe_fp8_table_create: shape");
  }
  // with HIVE_SLEEP_VMM a VMM region (sleep level 3 restores it at the same VA, like the weights its pointers name)
  Fp8BMat* d = nullptr;
  if (devmem::on()) { d = (Fp8BMat*)devmem::alloc(sizeof(Fp8BMat) * 3 * (size_t)E); HIVE_CHECK(d != nullptr, "glm_moe_fp8_table_create: VMM allocation"); }
  else CUDA_CHECK(cudaMalloc(&d, sizeof(Fp8BMat) * 3 * (size_t)E));
  CUDA_CHECK(cudaMemcpy(d, host_triples, sizeof(Fp8BMat) * 3 * (size_t)E, cudaMemcpyHostToDevice));
  return d;
}
void glm_moe_fp8_table_free(Fp8BMat* table) {
  if (table && !devmem::free_if_owned(table)) CUDA_CHECK(cudaFree(table));
}

size_t glm_moe_fp8_ws_bytes(int M, int K, int I) { return ((size_t)M * K * I * sizeof(bf16) + 255) & ~(size_t)255; }

void glm_moe_fp8_decode(const Fp8BMat* table, int E, const int32_t* ids, const float* w, const bf16* x, int M, int K, int H, int I, float limit,
                        float* out, void* ws, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 4 && K >= 1 && M * K <= 32 && E >= 1, "glm_moe_fp8_decode: 1 <= M <= 4, M*K <= 32");
  HIVE_CHECK(H == 4096 && I == 2048, "glm_moe_fp8_decode: instantiated for H 4096 · I 2048");
  HIVE_CHECK(((uintptr_t)x % 16) == 0 && ((uintptr_t)ws % 16) == 0, "glm_moe_fp8_decode: alignment");
  bf16* y = static_cast<bf16*>(ws);
  const int MK = M * K;
#define HIVE_MOE_GU(MM) launch_pdl(moe_gu_kernel<MM, 8>, dim3(I / 8, MK), dim3(256), 0, st, table, E, ids, MK, K, x, H, limit, y)
  HIVE_MOE_DISPATCH_M(M, HIVE_MOE_GU)
#undef HIVE_MOE_GU
#define HIVE_MOE_DN(MM) \
  launch_pdl(moe_down_kernel<MM, 4, kDownRows>, dim3(H / kDownRows), dim3(256), 0, st, table, E, ids, w, MK, K, (const bf16*)y, out, H)
  HIVE_MOE_DISPATCH_M(M, HIVE_MOE_DN)
#undef HIVE_MOE_DN
}

}  // namespace hive::glm
