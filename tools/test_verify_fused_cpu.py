#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Q2 HIVE_MTP_VERIFY2_FUSED — CPU checks (no GPU).

The R1 verify path (runtime.cpp attention_verify_dev) can run its attention front through the D2 fused front
(k::dec_attention_front) in a verify mode (DecAttnArgs::vgrp). GPU bit comparison lives in engine/tests/test_verify_decode.cu ⑤.
Here, on the PRODUCTION source text:
  1. switch parsing: HIVE_MTP_VERIFY2_FUSED is env_on (unset/""/"0" = off), read once;
  2. the verify branch fills DecAttnArgs with exactly the same lines as attention_decode_dev's D2 branch (verbatim),
     under the same condition (only the D2 switch replaced by this switch), and the normal decode never sets vgrp/vsave;
  3. verify-mode window indices: the code inside dec_qkv_a_kernel (D2) and dec_qkv3_kernel (A2 QKV3) is compiled on the CPU
     (pasted verbatim, threadIdx emulated) and compared with window_idxs_verify_kernel (R1) over part layouts × positions;
     single-row parts must also equal the default (window_idxs_rows) formula, and the default branch text is unchanged;
  4. front wiring: ring pointers are withheld from the q_a‖kv kernels in verify mode (all launch sites), the ratio>1
     compressor in verify mode is R1's compressor_step_seq and the compress tail is told so, the ring is written after
     the sparse attention, and runtime does not write the ring a second time.
Negative controls: mutated index formulas must fail check 3.
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / 'engine/src/runtime.cpp'
FUSED = ROOT / 'engine/src/kernels/attn_decode_fused.cu'
A3 = ROOT / 'engine/src/kernels/decode_attn3.cu'
VD = ROOT / 'engine/src/kernels/verify_decode.cu'


def fail(msg):
    print('verify-fused CPU: FAIL', msg)
    sys.exit(1)


def func_body(text, signature):
    i = text.find(signature)
    if i < 0:
        fail(f'missing {signature!r}')
    j = text.find('{', i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    fail(f'unbalanced {signature!r}')


def block_after(text, marker):
    """Text of the brace block that opens on the line containing marker (without the braces)."""
    i = text.find(marker)
    if i < 0:
        fail(f'missing marker {marker!r}')
    line_start = text.rfind('\n', 0, i) + 1
    j = text.find('{', line_start)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    fail(f'unbalanced block at {marker!r}')


def args_block(body):
    a = body.find('k::DecAttnArgs a;')
    b = body.find('a.scratch = w.iscore_f.p; a.scratch_bytes = w.iscore_f.n;')
    if a < 0 or b < 0:
        fail('DecAttnArgs block not found')
    return body[a:b]


def check_runtime():
    rt = RT.read_text()
    m = re.findall(r'static const bool v2_fused = (env_on\("HIVE_MTP_VERIFY2_FUSED"\));', rt)
    if len(m) != 1:
        fail(f'HIVE_MTP_VERIFY2_FUSED parse expression count {len(m)}')
    dec = func_body(rt, 'void Runtime::attention_decode_dev(')
    ver = func_body(rt, 'void Runtime::attention_verify_dev(')
    if args_block(dec) != args_block(ver):
        fail('DecAttnArgs fill differs between attention_decode_dev and attention_verify_dev')
    cd = re.search(r'if \(dec_attn_fused && (fuse_ && M <= 8 [^{]*)\{', dec)
    cv = re.search(r'if \(v2_fused && (fuse_ && M <= 8 [^{]*)\{', ver)
    if not cd or not cv or cd.group(1) != cv.group(1):
        fail('verify fused condition != D2 condition')
    for line in ('static const bool dec_attn_tc = env_on("HIVE_ATTN_TC");',
                 'static const bool dec_idx_tc = getenv("HIVE_IDX_TC") && atoi(getenv("HIVE_IDX_TC")) != 0;',
                 'static const bool dec_idx_f32 = getenv("HIVE_IDX_F32") && atoi(getenv("HIVE_IDX_F32")) != 0;'):
        if line not in dec or line not in ver:
            fail(f'switch line differs: {line}')
    if 'a.vgrp' in dec or 'a.vsave' in dec:
        fail('normal decode sets verify-mode fields')
    if 'a.vgrp = w.vgrp_d;' not in ver:
        fail('verify branch does not set vgrp')
    if 'if (!front_done) k::ring_write_rows(w.kv.as<bf16>(), M, D, win, w.pos_d, w.ringp_d, st_);' not in ver:
        fail('runtime ring write not skipped after the fused front')
    if ver.count('k::window_idxs_verify(') != 1 or ver.find('if (!front_done) {') < 0 or ver.find('if (!front_done) {') > ver.find('k::window_idxs_verify('):
        fail('R1 chain is not inside !front_done')
    print('verify-fused CPU: runtime.cpp contracts OK (env_on parse · same DecAttnArgs lines · same condition · default decode untouched · one ring write)')


def check_front():
    f = FUSED.read_text()
    a3 = A3.read_text()
    front = func_body(f, 'bool dec_attention_front(const DecAttnArgs& a, Blas* blas, cudaStream_t st)')
    stage = func_body(f, 'bool dec_front_stage(')
    q3 = func_body(a3, 'bool attn3_qkv_a(')
    for name, body in (('front', front), ('stage', stage), ('qkv3', q3)):
        if 'a.vgrp ? nullptr : a.ring_ptrs' not in body or 'a.counters, a.vgrp)' not in body:
            fail(f'{name}: q_a‖kv launch does not withhold the ring / pass vgrp in verify mode')
    if 'if (vstep) compressor_step_seq(a.ckv, a.cscore, M, D, a.ratio, a.src.pos, a.src.skv, a.src.ssc, a.cout, a.valid, st);' not in front:
        fail('verify compressor is not compressor_step_seq')
    if not re.search(r'dec_compress_kernel<<<[^;]*a\.valid, vstep\);', front):
        fail('compress tail not told about the stepped compressor')
    ia, ir = front.find('dec_attn_kernel<<<'), front.find('ring_write_rows(a.kv, M, D, a.win, a.src.pos, a.ring_ptrs, st);')
    if ia < 0 or ir < ia or 'if (a.vgrp) {' not in front[ia:ir + 1]:
        fail('verify ring write is not after the sparse attention under a.vgrp')
    print('verify-fused CPU: front wiring OK (ring withheld at 3 launch sites · compressor_step_seq · stepped tail · ring after attention)')


PROG = r'''
#include <cstdint>
#include <cstdio>
#include <vector>
#include "hive/verify_rows.h"  // R1: the window_col formula moved into the header
using namespace hive;
#define __restrict__
struct U { unsigned x; };
static void ref_kernel(unsigned tid, int M, int win, const int32_t* pos, const int32_t* grp, int32_t* out, int out_stride) {
  U blockIdx{0}, blockDim{1}, threadIdx{tid};
  @@REF@@
}
static void d2_block(int M, int win, const int* spos, const int* sgrp, int32_t* idx_out, int idx_stride) {
  U threadIdx{0}; const int FT = 1;
  @@D2@@
}
static void q3_block(int M, int win, const int* spos, const int* sgrp, int32_t* idx_out, int idx_stride) {
  U threadIdx{0}; const int Q3T = 1;
  @@Q3@@
}
static void rows_default(int M, int win, const int* spos, int32_t* idx_out, int idx_stride) {
  U threadIdx{0}; const int FT = 1;
  @@DEF@@
}
int main() {
  const int win = 128, stride = 640;
  const long starts[] = {0, 1, 2, 5, 126, 127, 128, 129, 130, 254, 255, 256, 4095, 4096, 100000, 1048575};
  const std::vector<std::vector<int>> lays = {{1}, {2}, {3}, {6}, {8}, {1, 1}, {2, 3}, {4, 4}, {1, 5}, {3, 1, 2}, {2, 2, 2, 2}, {1, 1, 1, 1}, {1, 2, 1, 4}};
  long n = 0, bad = 0;
  for (const auto& lay : lays)
    for (int si = 0; si < 16; ++si) {
      std::vector<int32_t> pos, grp;
      int r = 0;
      for (size_t p = 0; p < lay.size(); ++p) {
        const long p0 = starts[(si + 5 * p) % 16];
        for (int i = 0; i < lay[p]; ++i) { pos.push_back((int32_t)(p0 + i)); grp.push_back(r); }
        r += lay[p];
      }
      const int M = r;
      std::vector<int32_t> a((size_t)M * stride, 7777), b(a), c(a), d(a);
      for (unsigned t = 0; t < (unsigned)(M * win) + 3; ++t) ref_kernel(t, M, win, pos.data(), grp.data(), a.data(), stride);
      d2_block(M, win, pos.data(), grp.data(), b.data(), stride);
      q3_block(M, win, pos.data(), grp.data(), c.data(), stride);
      rows_default(M, win, pos.data(), d.data(), stride);
      for (int m = 0; m < M; ++m) {
        const bool single = (m == grp[m]) && (m + 1 == M || grp[m + 1] != grp[m]);
        for (int j = 0; j < stride; ++j) {
          const size_t i = (size_t)m * stride + j;
          ++n;
          if (a[i] != b[i] || a[i] != c[i] || (single && a[i] != d[i])) ++bad;
        }
      }
    }
  printf("%ld cells, %ld mismatches\n", n, bad);
  return bad ? 1 : 0;
}
'''


def build_prog(f, a3, vd):
    ref = func_body(vd, '__global__ void window_idxs_verify_kernel(')
    d2 = block_after(f, 'if (idx_out && vgrp) {')
    q3 = block_after(a3, '    if (vgrp) {  // verify version: same formula as window_idxs_verify_kernel')
    dflt = block_after(f, '} else if (idx_out) {  // same formula as window_idxs_rows')
    return PROG.replace('@@REF@@', ref).replace('@@D2@@', d2).replace('@@Q3@@', q3).replace('@@DEF@@', dflt)


def run_prog(src, tmp, name):
    c = Path(tmp) / f'{name}.cpp'
    exe = Path(tmp) / name
    c.write_text(src)
    r = subprocess.run(['g++', '-std=c++20', '-O1', '-I' + str(ROOT / 'engine/include'), str(c), '-o', str(exe)], capture_output=True, text=True)
    if r.returncode:
        fail(f'{name} compile:\n{r.stderr[-3000:]}')
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def check_index_formula():
    f, a3, vd = FUSED.read_text(), A3.read_text(), VD.read_text()
    # the default (window_idxs_rows formula) statement is still present unchanged in both kernels
    base_line = 'idx_out[(size_t)m * idx_stride + j] = srcp < 0 ? -1 : (srcp < p ? (int32_t)(srcp % win) : (int32_t)(win + m));'
    if f.count(base_line) != 1 or a3.count(base_line) != 1:
        fail('default window-index line changed')
    with tempfile.TemporaryDirectory() as tmp:
        rc, out = run_prog(build_prog(f, a3, vd), tmp, 'vidx')
        if rc:
            fail(f'verify window indices differ from window_idxs_verify: {out}')
        print(f'verify-fused CPU: verify window indices (D2 · QKV3 kernel text) == window_idxs_verify, single-row parts == window_idxs_rows: {out}')
        mutants = [
            ('chunk row = own row only', 'win + g + (srcp - p0)', 'win + m'),
            ('ring for in-part positions', 'srcp < p0 ?', 'srcp < p ?'),
            ('part start from own row', 'p0 = spos[g]', 'p0 = spos[m]'),
        ]
        for name, old, new in mutants:
            if f.count(old) < 1:
                fail(f'mutant anchor missing: {old}')
            rc, out = run_prog(build_prog(f.replace(old, new, 1), a3, vd), tmp, 'mut')
            if rc == 0:
                fail(f'negative control survived: {name}')
            rc, out = run_prog(build_prog(f, a3.replace(old, new, 1), vd), tmp, 'mut3')
            if rc == 0:
                fail(f'negative control survived (qkv3): {name}')
        print(f'verify-fused CPU: {len(mutants)} negative controls × 2 kernels fail as expected')


if __name__ == '__main__':
    check_runtime()
    check_front()
    check_index_formula()
    print('verify-fused CPU: ALL PASS')
