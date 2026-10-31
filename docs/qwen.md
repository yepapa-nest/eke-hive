# Qwen3.8-Flash-Next in Eke Hive

> **Status: measured.** The family runs on the GPU at 96 / 72 / 48 / 32 GB — numbers in the [README](../README.md#performance-by-vram-size)
> and under [Measured](#measured-rtx-pro-6000-96-gb-600-w-2026-10-10) below; the [validation plan](#validation-plan) is the
> order it was brought up in.

Eke Hive's third model family is **Qwen3.8-Flash-Next** (config.json `model_type` `qwen4_exp`), from the NVIDIA ModelOpt NVFP4
checkpoint. The daemon is the same source (`engine/src/hived.cpp`) built a third time as `hived_qwen`; the launcher picks it from the
checkpoint's `config.json`, so the same scripts, API server, sessions, prompt-prefix reuse and sampling serve all three families.

## Quick start

```bash
# 1. Checkpoint (NVFP4 experts, FP8 PLE table and MTP experts, BF16 rest — ~124 GB)
huggingface-cli download nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir /models/nvidia-Qwen3.8-Flash-Next-NVFP4
# 2. Build (builds hived, hived_glm and hived_qwen)
scripts/build.sh
# 3. Start (the launcher sees a qwen4_exp checkpoint, starts hived_qwen and reads config/qwen.env)
HIVE_CKPT=/models/nvidia-Qwen3.8-Flash-Next-NVFP4 scripts/hive-start.sh
```

## Model and where each part lives

| Part | Format | Where it lives |
|---|---|---|
| 48 MoE layers × 512 routed experts (top-10, intermediate 640) | NVFP4 (checkpoint), 2.64 MiB per record | pinned RAM (63.4 GiB) + VRAM cache — **all of them in VRAM with the whole card**; misses computed by the CPU |
| 36 Gated DeltaNet layers (16 key / 48 value heads × 128) | BF16 | VRAM |
| 12 full-attention layers (24 / 2 heads × 256, output gate) + QSA indexer (4 heads × 128) | BF16 | VRAM |
| hyper-connections (4 copies of the 2560 hidden, low-rank 320 gated mix and combine), shared experts, routers | BF16 | VRAM |
| PLE n-gram table (320,001,536 rows × 160, layer 1) | FP8 + one BF16 scale | pinned RAM (47.7 GiB) or on the SSD (`HIVE_QWEN_PLE=ssd`) |
| MTP layer (one full-attention layer, 512 routed experts) | experts FP8 128×128 blocks, rest BF16 | VRAM (experts 2.4 GiB) |
| token embedding | BF16 | host RAM (rows gathered per token) |
| vision encoder (27 blocks × 1152, patch 16, 2×2 merge) | BF16 | VRAM, loaded on the first image |

Code: `engine/include/hive/qwen/` and `engine/src/qwen/`. The routed experts use the GLM family's expert store
(`hive/glm/glm_experts.h` — the same ModelOpt NVFP4 record format, slot cache, promotion policy, NUMA-split CPU pool and prefill
streaming), constructed from a model-independent `ExpertSpec`.

### The layer

Each layer works on a hidden stream of four 2560-wide copies:

1. (layer 1 only) **PLE**: 16 hashed 2- and 3-gram rows of the token, gated against the stream, through a dilated causal conv, added
   to the stream (`hive/qwen/ple.h`).
2. **Mix**: a (1 + w) RMS norm per copy, a low-rank (320) sigmoid weight per element, the mean of the weighted copies → the block input.
3. **Attention**: Gated DeltaNet (`hive/qwen/gdn.h`, adapted from the GLM family's KDA kernels) or full attention with the QSA indexer
   (`hive/qwen/qsa.h`): 4-token groups of indexer keys are mean-pooled and normed; each query keeps the 512 best-scored groups plus
   its own open group — dense attention while at most 512 groups are visible (2,048 tokens).
4. **Combine**: the block output is added to each copy with its own weight 2·sigmoid(·).
5. The same mix → **MoE** (softmax router, top 10 renormalised; shared expert gated by sigmoid) → combine.

After the last layer a final mix (no combine) feeds `lm_head`. Positions use interleaved M-RoPE (θ 1e7, sections 11/11/10) on the
first 64 of 256 dims; image tokens take (t, h, w) coordinates and move the text after them (`QwenSeq::rope_delta`).

## What the three goals are in the code

### 1. VRAM budget — startup only

`HIVE_QWEN_VRAM_MB` (unset = the whole card) is the most device memory the process may use. Everything that sizes itself from free
VRAM counts against `min(free VRAM, budget − used)`: the expert cache at startup (`fam::ExpertStore::alloc_cache`), KV growth
(`QwenEngine::kv_reserve`) and the vision encoder (`make_room`). Experts that do not fit stay in pinned RAM and are computed by the
CPU when routed (the GLM family's path, with promotion of the most used ones into free slots).

With the whole card every routed expert fits (24,576 records), the runtime loads them all at startup, and decode / verify steps
**route on the device**: the router's top-10 goes straight into a device table of record pointers and the experts run without a host
round trip (`QwenEngine::moe_device`; `HIVE_QWEN_DEVICE_ROUTE=0` turns it off). The host path of `GlmExperts::decode_layer` reads the
routing on the host once per MoE layer — 48 synchronizations per step, every launch eager.

When the cache cannot hold every expert (a VRAM budget, or a long context took cache slots for its KV), decode / verify keep a device
path as long as one of these applies:

- **Mapped route** (`HIVE_QWEN_MAPPED_ROUTE`, default on) while the cache holds at least `HIVE_QWEN_MAPPED_MIN` (default 0.6) of the
  experts: the record table points a non-resident expert at its pinned host copy, and the expert kernels read it over PCIe; the routing
  of the step is copied back for the cache bookkeeping (`GlmExperts::note_used`, promotions as on the host path). At startup the free
  slots are filled so the cache starts full. Measured at a 72 GB budget (74.5 % of the experts resident, 98 % of the routed experts hit):
  c1 86.0 tok/s on the host path → 156.4 mapped. Below the threshold whole records over PCIe cost more than the CPU (48 GB: 31.5 vs
  81.1 tok/s).
- **Hybrid route** (`HIVE_QWEN_HYBRID_ROUTE`, default on) below that threshold: the step stays on the device as a CUDA graph; per MoE
  layer the routing and the MoE input go to pinned host memory, a host node on a side stream runs the non-resident experts on the CPU
  pool while the resident ones run on the GPU, and the reduce adds the CPU share. `HIVE_QWEN_HYB_PCIE=N` sends every N-th non-resident
  expert of a layer over PCIe to the GPU instead (`config/profiles/fn-48.env`: 4 — c1 / c4 / c8 125.9 / 172.9 / 220.1 → 129.7 / 179.8 /
  227.2 tok/s in a short pass); `HIVE_QWEN_HYB_ROWS=N` sends steps with more rows to the host path. Needs the fused decode MoE
  (`HIVE_QWEN_FUSE`) and every layer on the GPU.
- **CUDA graphs under a budget.** Each captured graph holds ~10 MB of device memory, and under a budget that memory comes out of the
  expert cache. `HIVE_QWEN_GRAPH_CAP` (default 32 while the cache cannot hold every expert, else unlimited) keeps the most recently
  used graphs (48 GB, cap 128 → 32: c1 / c4 / c8 129.8 / 181.9 / 230.0 → 131.4 / 186.5 / 248.4 tok/s); `HIVE_QWEN_GRAPH_ROWS=N` runs
  forwards with more than N rows eagerly (`config/profiles/fn-48.env` and `fn-32.env`: 8 — at 32 GB, 11 % resident, c4 / c8 98.4 /
  116.9 with graphs for every shape → 109.0 / 128.9).

Only when neither applies (or with `HIVE_QWEN_DEVICE_ROUTE=0`) the host path is used.

Layer-major prefill (`HIVE_QWEN_PREFILL_TILES`, default 4) runs only while some expert is not resident: it exists to stream each
missing expert over PCIe once per layer for the whole prompt.

### 1b. Layer placement on the CPU (cards smaller than the dense weights)

`HIVE_QWEN_GPU_LAYERS=N` (0 < N < 48) keeps layers 0..N−1 on the GPU and runs layers N..47 in the FN CPU engine
(`hive/qwen/qwen_cpu.h`, the engine of `hived_qwen_cpu`) inside the same process: after the GPU layers the hidden streams
(bf16, 20 KiB per row) go to host memory, the CPU engine runs its layers (attention / Gated DeltaNet, MoE with its own copy of those
layers' experts, PLE when layer 1 is among them) and the streams come back for the final mixer and `lm_head` on the GPU — one PCIe
crossing each way per forward, exact (the values are bf16 both ways). The GPU expert store (`GlmExperts`) covers only the GPU layers.

Unset, under `HIVE_VRAM_LIMIT_MB` (docs/vram-footprint.md) the family chooses N itself before loading: the most layers whose dense
weights (from the checkpoint headers) fit in the limit minus the CUDA context estimate, `lm_head` + final mixer, the lowest-setting
work buffers of hived's plan, `HIVE_CACHE_RESERVE_MB` and a minimum expert cache (`HIVE_QWEN_PLAN_CACHE_MB`, default 256). hived's
plan then lowers the knobs as usual. Without a limit and without the switch nothing changes.

With a CPU tail: no MTP (the switch is forced to 0 — the MTP block reads every layer's final stream), no layer-major prefill (the
GPU layers' chunks run one at a time), session images carry the tail's state. Validated against the oracle with N = 24 and N = 4:
per-layer injection worst layer cos 0.99663, last logits cos 0.999995, 16 / 16 decode tokens identical (2026-10-09).

### 2. PLE table in RAM or on the SSD

`HIVE_QWEN_PLE=ram` (default): the 128 table shards are copied once into pinned memory (47.7 GiB) and each token's 16 rows are
copied from there. `HIVE_QWEN_PLE=ssd`: the table stays in the checkpoint file; rows are read with O_DIRECT `pread` by a thread pool
(`HIVE_QWEN_PLE_SSD_THREADS`, default 64) into a row cache (`HIVE_QWEN_PLE_SSD_CACHE_MB`, default 2048, CLOCK — the engram SSD cache of
the DeepSeek family), and a prompt's rows are read ahead as soon as its tokens are known (rows depend on token ids only). Both modes
give the same bytes. A token needs 16 rows of 160 B: 16 reads per token, against 48 per token for DeepSeek's engram SSD mode.

The n-gram rule (`PleHash::rows_of`) is checked against an independent implementation (`oracle/qwen4_ple_vectors.py`) by the CPU
test `test_qwen_ple_cpu`.

### 3. Kernels for the RTX PRO 6000

- decode GEMVs on the tensor cores (`glm_gemv_bf16_multi`, shared with GLM) for every BF16 projection at ≤ 8 rows;
- device routing with every expert resident (above): no host synchronization inside a decode step;
- Gated DeltaNet recurrence with the state in registers (from the KDA kernels), verify rows with per-row state snapshots in one launch;
- QSA prep / group compression / scoring / sparse GQA attention driven by a row table, so prefill chunks, verify rows and decode rows of
  several sequences take the same kernels.

Decode / verify (rows ≤ 64, several sequences per call):
- dense projections: 8-bit copies (`HIVE_QWEN_DENSE8=int8`, group 128, bf16 originals freed) on the tensor cores (`fn_gemv8`; 9–32 rows use
  the 2-blocks-per-SM instances), the verify head int8 with fp32 logits (`HIVE_QWEN_DENSE8_HEAD=1`); remaining bf16 projections take
  `fn_gemv_tc`, whose per-row arithmetic does not depend on the row count (batched decode == single-sequence decode, token for token);
- routed experts: `fn_moe` (records grouped by expert id, gate|up and down kernels at ~1.5 TB/s of record traffic at 20–40 rows);
- Gated DeltaNet verify rows of all sequences in one launch (`gdn_decode_rows_multi`); replay rollback for parts ≥ 1, batched across parts;
- QSA: split-KV tensor-core decode attention (`HIVE_QWEN_QSA_SPLIT`, default 16), a dedicated scorer for ≤ 8 rows of one sequence,
  row tables mirrored to device memory;
- MTP k = 4 with a 64K-token FP8 draft head (`HIVE_QWEN_DRAFT_VOCAB`), drafts of all sequences in one chain, the accepted rows' MTP
  entries in one pass per step.

Prefill: grouped W4A16 expert GEMMs (`fn_moe_pf`, 32- or 128-row tiles), tensor-core QSA attention and scoring, HC vector kernels.
Checkpoints: the host copy runs on a worker thread from a device snapshot (`HIVE_QWEN_CKPT_ASYNC`, default on); images up to
`HIVE_QWEN_CKPT_SNAP_MB` (192) are snapshotted whole, so a pool eviction never blocks the next admission.

### Measured (RTX PRO 6000 96 GB, 600 W, 2026-10-10)

Official `sglang.benchmark.serving`, random 1024 in / 256 out (prefill axes: 8K / 32K / 64K in, 128 out), temperature 0, no thinking,
100 prompts per concurrency axis, 20 per prefill axis. hived_qwen, `config/qwen.env` defaults (int8 dense + int8
head, 64K draft vocab, MTP k 4, `HIVE_MAX_BATCH=16`, `HIVE_QWEN_KV_BASE=2048`). SGLang column: perf-core run of 2026-09-18 (its
`--max-running-requests 8`, NEXTN 3 steps / 4 draft tokens).

| Axis | FN (hived_qwen) | SGLang | |
|---|---|---|---|
| c1 output tok/s · median TPOT · 1/TPOT | 189.4 · 4.22 ms · 237 | — · 4.45 ms · 224.7 (300 W) / 228.8 (600 W) | ✅ |
| c1 median TTFT | 162 ms | 180 ms | ✅ |
| c4 output tok/s (TTFT) | 370.5 (177 ms) | 377.3 (209 ms) | 98 % |
| c8 output tok/s (TTFT) | 455.1 (184 ms) | 445.7 (233 ms) | ✅ |
| c16 output tok/s (TTFT) | 533.5 (206 ms) | 438 | ✅ |
| prefill 8K / 32K / 64K TTFT | 792 / 3194 / 6462 ms | 1097 / 4247 / 8668 ms | ✅ |
| long-context TPOT 8K / 32K / 64K | 4.51 / 4.61 / 4.51 ms | ~4.4–4.8 ms | ✅ |
| quality (ruler / qa / code / tool / session) | 33/35 · 72/80 · 20/20 · 22/22 · 13/13 = 160/170 | bf16 baseline 160/170 | = |

MTP accept length 2.78 tokens per step. Remaining c4 lever: the 1K-token prefill (129 ms GPU, of which the expert GEMMs 70 ms at
~0.9–1.2 TB/s) stalls the other decoders on every arrival.

### Against the FP4-Local vLLM fork (2026-10-30/31)

Target: hyudryu/Qwen3.8-Flash-Next-FP4-1xRTXPro6000 (vLLM + b12x, 450 W): coding bench C1 / C4 / C8 287.0 / 662.2 / 965.6 tok/s, prefill
4K / 16K / 64K / 128K 14.6K / 14.4K / 14.0K / 13.5K tok/s. That release serves **nvidia/Qwen3.8-Flash-Next-FP4-Local**, not the checkpoint
served here: its config.json (read 2026-10-30) keeps the experts NVFP4 *with FP4 activations* and every dense projection (GDN in/out,
attention q/k/v/o and indexer, shared expert) in **MXFP8**, the PLE table in NVFP4. Here the dense part is BF16 (INT8 copies, group 128)
and the experts run W4A16.

Where a 16K-token prefill went before (nsys, 600 W, 1.51 s of kernels in a 1.73 s request): routed experts (W4A16 grouped GEMM) 512 ms ·
dense projections (cuBLAS bf16 on the expanded INT8 copies, ~357 of the card's 551 TFLOP/s) 347 · QSA attention 277 · GDN recurrence 210 ·
hyper-connection mixes and norms 144 · idle ~220. mma.sync peaks on this card (register-only loops): FP4 2184, FP8 1095, INT8 1100, BF16 551
TFLOP/s. C8 coding: verify steps of ~33 rows 26.8 ms (expert records ≈ 210 distinct per layer × 2.77 MB: the bandwidth floor is ~18 ms)
plus 2.8 ms of drafts; **only 8 % of the verify forwards replayed a captured graph** (the per-split keys never repeated).

**Defaults (bit-identical — every quality-suite output equal to the 2026-10-30 build except the 20 tool items that differ between two
baseline runs as well; serving-path `qwen_check` of a 6,012-token prompt with every expert resident: all 48 layer streams, the logits and the
decoded ids equal):**
- W4A16 grouped experts v2 (`HIVE_QWEN_MOE_PF_V2`): the NVFP4 bytes and scales are staged as they are and each lane converts its own
  B fragment right before the mma (the first kernel expanded every weight tile to bf16 in shared memory — ncu: stalls wait / mio_throttle /
  barrier at 8 warps per SM). 8,192 rows per layer 5.57 → 4.09 ms on the 128-row tiles (short prompts' 32-row tiles are weight-bandwidth
  bound and keep the first kernel); the reduce writes the combined MoE output.
- GDN recurrence with 4 warps per block (192 blocks instead of 96): 16K prefill 166 → 154 ms.
- the combine + next-norm fusion in prefills unless a layer yield is actually due (it was off for every yielding prefill).
- verify part splits read from device memory (`HIVE_QWEN_VERIFY_DYN`): one graph per (rows, parts) serves every split — 94–99 % of the
  verify forwards replay a graph; C8 coding 786 → 853 tok/s at 600 W in an A/B on the same build.
- the batched MTP draft chain as a CUDA graph (`HIVE_QWEN_DRAFT_GRAPH`).

**Switches that did not pass the zero-loss gate** (quality_eval non-thinking, seed 20261001; baseline twice: ruler 35/35 · qa 72/80 ·
code 20/20 · tool 22/22 · ops 19/20 = 168; pass = every category within one binomial standard error of the baseline and total ≥ 168):

| Switch | What | Gate | Speed |
|---|---|---|---|
| `HIVE_QWEN_QSA_MR=1` | 4 consecutive prefill rows attend over the union of their selected groups (48 mma rows per block, K/V once); same math, another fp32 order | ruler 32/35 (vartrack 4K/16K/100K answer with a derivation), qa 73/80 → FAIL; `=0` on the same build passes every vartrack item | 16K attention 246 → 131 ms |
| `HIVE_QWEN_MOE_A4=1` | W4A4 experts: activations NVFP4 per row (dynamic global), `mma.sync kind::mxf4nvf4` m16n8k64 — equals a host emulation of the same conversions | ruler 34/35 · qa 70/80 · tool 21/22 = 164 → FAIL | 8,192 rows per layer 5.57 → 2.27 ms |
| `HIVE_QWEN_DENSE_MX=1` | MXFP8 dense prefill GEMMs (e4m3 + ue8m0 per 32, cuBLASLt VEC32_UE8M0 — 886 TFLOP/s at 16384 × 16480 × 2560) | ruler 32/35 · qa 73/80 = 166 → FAIL | see the table |
| all three | the FP4-Local precision set | ruler 34/35 · qa 70/80 = 165 → FAIL | see the table |

A chunked GDN prefill (in-chunk triangular solve; chunk recurrence on CUDA cores or 3xTF32 tensor cores) was tried and dropped: 8,192
tokens 3.5–4.2 ms against 2.70 ms for the token-wise kernel (its extra prep kernels alone cost 1.3 ms).

Results — official harness (tools/bench_coding.py C1 / C4 / C8 × 3 rounds, median; tools/bench_chat.py long prompts, fresh salt):

| 450 W | C1 | C4 | C8 | 4K | 16K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP4-Local vLLM fork (its README) | 287.0 | 662.2 | 965.6 | 14.6K | 14.4K | 14.0K | 13.5K |
| before (2026-10-30 service build) | 265.9 | 627.7 | 769.1 | 7.8K | 8.7K | 8.8K | – |
| defaults now (zero-loss) | 261.8 | 602.1 | 808.4 | 7.8K | 9.3K | 9.4K | 9.6K |
| + `MOE_A4` `DENSE_MX` `QSA_MR` (lossy, gate FAIL) | 265.1 | 641.4 | 826.2 | 11.7K | 13.5K | 13.4K | 13.1K |

| 600 W | C1 | C4 | C8 | 4K | 16K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 276.9 | 631.6 | 770.1 | 8.3K | 9.6K | 9.9K | – |
| defaults now | 266.5 | 651.7 | 864.0 | 8.3K | 10.5K | 10.7K | 10.5K |
| + the three lossy switches | 279.4 | 650.4 | 860.9 | 12.5K | 14.7K | 14.7K | 13.8K |

C1 / C4 rounds spread ±5–10 % (MTP acceptance follows the generated text); the C8 gain is the verify graphs. 128K was measured with a salt of
its own: tools/bench_chat.py sends all lengths with one salt, and the 128K prompt then reuses the 64K prompt's first 65,536 tokens through the
shared prefix ("reuse 65536/145259 via shared-prefix") — 14.7K / 16.7K tok/s in that run are not prefill speeds. Prefill tok/s = prompt
tokens / time to the first token (bench_chat), the same definition as the reference.

What is left with the zero-loss defaults (16K prefill, 600 W, ~1.5 s): the dense projections stay BF16 (~345 ms at ~65 % of the BF16 peak;
the reference runs them MXFP8), the experts W4A16 (~400 ms; the reference W4A4), GDN ~200 ms (token-wise recurrence, latency-bound at
4 warps per SM), QSA attention ~280 ms (the per-row kernel: 12 of 16 mma rows used, K/V gathered per row). The lossy switches close
most of the prefill gap (450 W: 16K −6 %, 64K −4 %, 128K −3 %, 4K −20 % — short prompts carry a fixed ~15 ms tail chunk at the prefix
snapshot and the expert weights are read once per layer whatever the prompt length), and none passes the gate. C8: a verify step of
~33 rows reads ~210 distinct expert records per layer (~18 ms at the card's bandwidth of a 25.6 ms step) plus 2.8 ms of drafts — the
remaining 15 % to the reference is in that step, not in launches any more (98 % of the forwards replay graphs).

`HIVE_QWEN_DENSE_MX` keeps the MTP layer's prompt entries on the bf16 GEMMs: with MXFP8 there the drafts after a 16K prompt with decode
were accepted 455/988 for every later request (C1 135 tok/s) against 1094/1611 (275 tok/s) with the guard — measured A/B 2026-10-31.

## NVIDIA FP4-Local checkpoint (2026-10-31)

`nvidia/Qwen3.8-Flash-Next-FP4-Local` (rev b9ce18a, ~96 GB) is a quantization-aware-distilled release of the same model. The family loads it
with the same `scripts/hive-start.sh` (`HIVE_CKPT` pointing at it); nothing else changes in the setup.

| Part | FP4-Local storage | How the engine takes it |
|---|---|---|
| routed experts (48 × 512) | NVFP4, block scales **compressed (CSF)**: `weight_scale.nvfp4_csf_fixed` U8 [R/16, 16 + 8·C] + `…_exceptions` U32 [n] | decoded into the usual record (`GlmExperts::load_all`, `hive/ckpt_quant.h`) — the records, cache and kernels are the NVFP4 checkpoint's |
| GDN in/out, attention q/k/v/o + indexer, shared experts (438 matrices) | MXFP8: e4m3 [N, K] + ue8m0 U8 [N, K/32] | kept as stored (QMat `fmt8` 3, 2.94 GB): decode GEMV folds the power-of-two scale into the bf16 operand (exact), prefill expands it exactly to bf16 or, with `HIVE_QWEN_DENSE_MX=1`, runs the block-scaled FP8 GEMM on the stored bytes (only the scales are re-tiled) |
| hyper-connection norms (97), PLE norms (3) | fp32 | kept fp32 (`NormW`; `HIVE_QWEN_NORM_BF16=1` rounds them as before) |
| PLE n-gram table | NVFP4 rows (U8 [rows, 80] + e4m3 [rows, 10] + one fp32 scale_2) | pinned RAM, 26.8 GiB (90 B per row; SSD mode not available for it — falls back to RAM with a warning) |
| MTP experts | NVFP4 W4A16, plain scales | records of the main layers' format in VRAM (1.32 GiB), run by `fn_moe_decode_routed` |
| vision qkv / proj · MLP / merger | MXFP8 · NVFP4 W4A16 | expanded to bf16 at load (MXFP8 exact, NVFP4 one rounding) |
| embeddings, lm_head, routers, hc mix weights, PLE key / value, MTP dense | BF16 | unchanged |

**CSF.** The scale layout was read from the tensor shapes and bytes only (no third-party code): per 16 rows, 16 base E4M3 code bytes (one per
row), then 16·C 4-bit deltas row-major (even position = low nibble), code = base + delta; each U32 exception = (code << 24) | flat index and its
delta is 0. Checks (`tools/fp4l_check.py`, 2026-10-31): all 73,728 projections decode with no invariant violation (exception index range, zero
delta under every exception, every code a finite non-negative E4M3 byte 0x40–0x7E); 24.2 M exceptions (59–13,793 per projection); the engine
decoder equals an independent numpy decoder byte for byte on 256 sampled projections; dequantized experts match the NVFP4 checkpoint's same
experts at cos 0.994 with norm ratio 1.003 (an offset of ±1 code: norm ratio 0.92 / 1.10, decoding without the exceptions: cos 0.989). The
loader refuses a projection that violates an invariant (a damaged tensor — data integrity). The PLE NVFP4 rows match the NVFP4 checkpoint's FP8
rows at cos 0.993–0.996 with the low nibble first (≈ 0 with the high nibble first).

**Oracle.** `oracle/qwen4_oracle.py` reads the FP4-Local formats with its own decoders. 26-token prompt + 8 decode steps, engine (`qwen_check`)
against it: bf16 for the remaining BF16 matrices — lowest layer cosine 0.976 (layer 36), top-1 9/9, top-20 overlap 0.939; with the INT8 copies
(`HIVE_QWEN_DENSE8=int8`, head INT8) 0.961 / 9/9 / 0.889 (the NVFP4 checkpoint against its oracle: 0.924 / 9/9 / 0.911).

**Quality** (tools/quality_eval.py non-thinking, seed 20261001; baseline = the NVFP4 service build 31a, 168 = ruler 35 · qa 72 · code 20 ·
tool 22 · ops 19, twice; pass = every category within one binomial standard error (≥ 1 item) and total ≥ 168, two runs):
| Run | Settings (FP4-Local; dense MXFP8 as stored; decode experts W4A16) | ruler · qa · code · tool · ops | Total (run 1 / 2) | Gate |
|---|---|---|---|---|
| L0 | INT8 copies of every bf16 matrix (`HIVE_QWEN_DENSE8=int8`, head INT8 — the service's `qwen.env`), norms rounded to bf16 | 35 · 67 · 20 · 22 · 20 | 164 | FAIL (qa) |
| L1 | no INT8 copies (`DENSE8=0`), norms rounded to bf16 | 35 · 72 · 20 · 22 · 20 | 169 | pass (one run) |
| L2 | INT8 copies, bf16 head | 35 · 69 · 20 · 22 · 20 | 166 | FAIL |
| L3 | `DENSE8=0` with `DENSE8_HEAD=1` (the head switch needs `DENSE8`, so = L1) | 35 · 72 · 20 · 22 · 20 | 169 | = L1 |
| L4 | `DENSE8=0`, fp32 norms | 35 · 72 · 20 · 22 · 20 | 169 / 169 | **pass** |
| L5 | INT8 copy of lm_head only (`DENSE8=int8 DENSE8_SCOPE=head DENSE8_HEAD=1`), fp32 norms | 35 · 72 · 20 · 22 · 20 | 169 / 169 | **pass** |
| N1 | L4 + `HIVE_QWEN_MOE_A4=1` + `HIVE_QWEN_DENSE_MX=1` (the checkpoint's own precision set) | 35 · 70 · 20 · 22 · 20 | 167 / 167 | FAIL (total) |
| N2 | L4 + `MOE_A4=1` | 35 · 70 · 20 · 22 · 20 | 167 | FAIL (total) |
| N3 | L4 + `DENSE_MX=1` | 35 · 72 · 20 · 22 · 20 | 169 / 169 | **pass** |
| N4 | L5 + `DENSE_MX=1` | 35 · 72 · 20 · 22 · 20 | 169 / 169 | **pass — chosen** |

Greedy decoding makes repeated runs identical (every second run equalled the first). The QAD checkpoint takes its MXFP8 prefill GEMMs
(activations e4m3 / ue8m0 per 32 — the precision it was trained with) without loss on this suite, unlike the NVFP4 checkpoint (166); its
W4A4 expert prefill still loses two qa items (167, as on the NVFP4 checkpoint: 164). INT8 copies of its remaining bf16 matrices (hyper-connection
up, PLE key / value, the MTP layer) fail where they passed on the NVFP4 checkpoint; the INT8 head alone passes.

**Speed** (official harness: tools/bench_coding.py C1 / C4 / C8 × 3 rounds median, tools/bench_chat.py long prompts with a fresh salt):
| | 450 W C1 | C4 | C8 | 4K | 16K | 64K | 600 W C1 | C4 | C8 | 4K | 16K | 64K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NVFP4 service build 31a | 277.1 | 615.2 | 798.8 | 7.7K | 9.2K | 9.4K | 260.6 | 618.9 | 850.1 | 8.4K | 10.5K | 10.7K |
| FP4-Local L4 | 267.8 | 622.0 | 792.7 | 7.6K | 9.3K | 9.4K | 265.1 | 639.6 | 852.9 | 8.3K | 10.5K | 10.8K |
| FP4-Local L5 | 287.0 | 623.0 | 807.0 | 7.7K | 9.3K | 9.4K | 263.7 | 632.6 | 862.3 | 8.2K | 10.5K | 10.8K |
| FP4-Local N3 | 264.1 | 609.0 | 808.5 | 8.1K | 10.0K | 10.1K | 273.1 | 617.8 | 837.1 | 8.7K | 11.2K | 11.5K |
| **FP4-Local N4 (chosen)** | 276.6 | 603.4 | 804.1 | 8.1K | 10.0K | 10.1K | 282.8 | 642.3 | 851.7 | 8.8K | 11.2K | 11.5K |
| FP4-Local N1 (gate FAIL) | 279.6 | 626.0 | 815.3 | 10.9K | 12.2K | 12.3K | 269.0 | 627.8 | 844.8 | 11.5K | 13.2K | 13.4K |
| vLLM + b12x (its README, 450 W) | 287.0 | 662.2 | 965.6 | 14.6K | 14.4K | 14.0K | | | | | | |

C1 / C4 rounds spread ±5–10 % (MTP acceptance follows the text); decode is the same code for every row of the table except the head (L5 / N4:
INT8) — the differences there are within that spread. MTP acceptance on the coding bench: 0.817–0.826 (3.57–3.64 tokens per step) against
0.826 (3.52) for 31a. The prefill gains come only from the MXFP8 dense GEMMs (+7 % at 16K / 64K); W4A4 experts would add another ~20 % but fail
the gate. Memory (N4, whole card): VRAM 87.7 GB at idle (non-expert weights 7.49 GiB, all 24,576 expert records resident), host RSS 172 GiB
(the PLE table 26.8 GiB instead of 47.7). Regression checks on N4 (2026-10-31): image-prefix reuse (`IMAGE REUSE OK`, 5.3K of 5.8K tokens
cached), thinking cap (64 + the exit phrase = 81 thinking tokens, `</think>` forced), a system message in mid-conversation with a new tool
(tool call, no 400), one-image text reading ("HIVE 4271").

Service settings for N4 (beside `config/qwen.env`): `HIVE_CKPT=<FP4-Local>`, `HIVE_QWEN_DENSE8=int8`, `HIVE_QWEN_DENSE8_SCOPE=head`,
`HIVE_QWEN_DENSE8_HEAD=1`, `HIVE_QWEN_DENSE_MX=1` (`MOE_A4` stays off).

### FP4-Local in 68 GiB beside another process (2026-10-31)

Goal: N4 with ~28 GB of the card left free; the experts that do not fit stay in pinned RAM (mapped / hybrid routing, §1). `HIVE_VRAM_LIMIT_MB=69632`
(the DS service value) is the wrong switch for that: this family keeps the limit's ballast for its life (the reserve that hands it back exists
for DeepSeek only — docs/vram-footprint.md §4.2), so the process held 94.6 GB. `HIVE_QWEN_VRAM_MB=68600` is the family's own budget
(context not counted) and the card stays below 68 GiB. Measured with the official harness; VRAM = the run's peak in `nvidia-smi` (1 s samples);
host RSS 265–267 GiB in every row (the 96 GB service: 267 GiB); C1 / C4 / C8 rounds spread ± 10 %:

| 68 GiB | 450 W C1 · C4 · C8 tok/s | coding TTFT C1 · C4 · C8 | 4K · 16K · 64K | 600 W C1 · C4 · C8 | 4K · 16K · 64K | VRAM peak | experts resident | quality |
|---|---|---|---|---|---|---:|---:|---|
| H0 `HIVE_VRAM_LIMIT_MB=69632` | 206 · 388 · 514 | 591 · 2194 · 4364 ms | 2.4K · 4.0K · 7.5K | 187 · 370 · 486 | 2.4K · 4.4K · 7.7K | 94,630 MiB | 68.9 % | 165 |
| H1 `HIVE_QWEN_VRAM_MB=68600` | 193 · 380 · 492 | 575 · 2165 · 4173 ms | 2.3K · 3.9K · 7.4K | 207 · 362 · 476 | 2.6K · 4.4K · 7.7K | 66,556 MiB | 68.8 % | 167 |
| H2 H1 + `MAPPED_MIN=0.3` `CACHE_RESERVE_MB=1500` | 207 · 424 · 531 | 589 · 2166 · 4189 ms | 2.4K · 4.0K · 7.5K | 190 · 383 · 492 | 2.6K · 4.2K · 7.7K | 68,314 MiB | 71.1 % | 166 |
| **H3** H2 + `MAX_BATCH=8` `MAX_CHUNK=8192` | **219 · 465 · 595** | 591 · 2219 · 4307 ms | 2.3K · 4.4K · 6.7K | **216 · 453 · 578** | 2.5K · 5.3K · 6.9K | 67,882 MiB | **78.0 %** | **168** |
| H4 H3 + `HIVE_GLM_PREFILL_CPU=0` `HYBRID_ROUTE=0` | 216 · 429 · 550 | 590 · 2153 · 4321 ms | 2.0K · 4.5K · 6.0K | 216 · 441 · 542 | 2.1K · 4.5K · 6.0K | 67,874 MiB | 78.0 % | 167 |

What moved it: the slots, not the routing. At start the cache held 16,899 slots (68.8 %) but a long prompt lends tail slots to its KV and the
cache fell below the 0.6 mapped threshold (13,809 resident) — the decode then took the hybrid route (CPU experts); `MAPPED_MIN=0.3` keeps the
device route. The session pool (16 × 308 MiB) and the 16K-row work buffers held ~7 GiB: 8 sessions and 8K-row chunks give the cache 2,260 more
slots (78 %), +15 % C4 / +12 % C8; the 64K prefill pays ~10 % for the smaller chunk. Short-prompt TTFT stays ~0.6 s: a prefill streams every
missing expert once per layer (`HIVE_QWEN_PREFILL_TILES`) — the gap to the whole card (84 ms) is that traffic.
Quality: the 96 GB path is bit-exact to the gate's baseline (169); here no row is — the partial-cache prefill runs the expert store's streaming
GEMMs (not FN's grouped kernels, which need every expert resident) and the missed experts' CPU share computes in fp32. H3 passes the gate
(qa 71 within one standard error, total 168); H4 shows the CPU share is not the only cause (still 167). Settings for 68 GiB (beside the N4
lines): `HIVE_QWEN_VRAM_MB=68600 HIVE_QWEN_MAPPED_MIN=0.3 HIVE_CACHE_RESERVE_MB=1500 HIVE_MAX_BATCH=8 HIVE_MAX_CHUNK=8192` and no
`HIVE_VRAM_LIMIT_MB`.

## Han / kana in Korean answers

Qwen3.8-Flash-Next now and then writes a Chinese word (上空, 精准, 面临的) or kana inside a Korean answer. The service keeps the model's own
distribution — the default is a sampling setting, not a filter (measured below); two tools exist for clients that want more:

- **`presence_penalty`** (OpenAI semantics; this family only — other families keep ignoring the field): every token already in the response
  has its logit lowered by the penalty (−2..2) before the request's temperature / top_k / top_p / min_p. hived applies it on the host at
  every row it samples — the step's row and each speculative verify row (accepted drafts are emitted before the next row is sampled, so they
  count) — on the GPU's candidate snapshot with the row's Σexp corrected from the shifted tokens' own logits, and on the full row when the
  request has one; the paths that could need a token outside the candidates take the full row (`hive/presence.h`,
  `tests/test_presence_cpu.cpp`: equal to the rule over the fully penalized row for greedy, top-k / top-p, min_p and the order-free path,
  positive and negative penalties).
- **Han / kana filter** (opt-in): `"hanja_filter": true` (or `x-hanja-filter: 1`) lowers the logits of a fixed token list in the answer span
  (after `</think>`; the whole output of a non-thinking request) by `"hanja_penalty"` (default ∞ — never sampled). The list: every token whose
  text is two or more Han ideographs and nothing else, or contains kana — single ideographs stay (개항(開港)); built once with
  `tools/build_cjk_mask.py <ckpt> config/fn-cjk-mask.bin` (53,479 of 248,077 tokens) and loaded with `HIVE_QWEN_TOKEN_MASK`. The shift is
  applied to the logits on the device before the candidate selection (decode and verify heads, captured graphs included) and on the host for
  the first token after a prefill; with MTP the verify rows are shifted the same way, so the output distribution does not depend on
  speculation (`hive/token_mask.h`, `tests/test_token_mask_cpu.cpp`). A request that asks for Japanese or Chinese text cannot get it while the
  filter is on — that is why it is not a default.

Measured (FP4-Local N4, 2026-10-31; `tools/hanja_eval.py`: 400 Korean free-response prompts, non-thinking, two seed sets = 800 answers per
row, temperature 0.6 fixed; "Han word" = an answer with two or more ideographs in a row outside parentheses — 上空, 精准, 面临的 — kana counted
separately; `tools/hanja_intent.py`: 10 requests each for a Japanese answer, a Chinese answer and Korean with Han in parentheses, judged by
script (≥ 5 kana · ≥ 10 ideographs and ≤ 2 Hangul · ≥ 2 parenthesized groups with an ideograph); sampled quality = `tools/quality_eval.py
--sampling` with the row's settings, seeds 20261001 / 20261002 — greedy quality of every row is the same 169):

| Sampling (temperature 0.6) | Han-word answers / 800 | Han characters | answers with kana | Japanese · Chinese · Han in parentheses | sampled quality |
|---|---:|---:|---:|---|---|
| S0 top_p 0.95 · top_k 20 (service) | 20 (7 + 13) | 1,048 | 0 | 8 · 4 · 10 / 10 | 166 · 168 |
| S1 S0 + min_p 0.05 | 29 (11 + 18) | 1,023 | 0 | 8 · 4 · 10 | — |
| S2 S0 + min_p 0.1 | 18 (7 + 11) | 1,112 | 0 | 8 · 4 · 9 | — |
| S3 top_p 0.8 · top_k 20 | 16 (8 + 8) | 1,156 | 1 | 8 · 4 · 10 | 162 · 170 |
| S4 S3 + presence_penalty 1.5 (Qwen's non-thinking recommendation) | 27 (16 + 11) | 936 | 1 | 8 · 4 · 10 | 163 · 170 |
| S0 + filter Δ = ∞ (opt-in; 400 answers, a quarter thinking) | 1 of 400 (7 without) | — | 0 | the filter blocks Japanese / Chinese answers | greedy 169 |

The leak rate is ~2 % of answers whatever the setting: the differences between S0–S4 are within the Poisson spread of ~20 events (± 4–5),
and the sampled quality runs spread ± 4 between seeds. presence_penalty 1.5 does not help (27): it pushes the model off tokens it already used,
toward rarer ones. Thinking mode leaks more in the answer than non-thinking (S0, 120 prompts each: `reasoning_effort` low 17, medium 6 Han-word
answers). Recommendation: keep S0; a client that must have no Chinese words in Korean answers can opt into the filter (it costs the ability to
answer in Japanese / Chinese in that request).

## Images and video

The server (`server/families/qwen.py`) resizes as the checkpoint's processor config describes (patch 16, 2×2 merge, factor 32, mean
and std 0.5, BICUBIC; image pixels between 65,536 and `HIVE_QWEN_IMAGE_MAX_PIXELS`, default 8,388,608, and at most 8,000 merged tokens
per span), writes patches in merge-block order (3 × 2 × 16 × 16 values per row), and expands the template's
`<|vision_start|><|image_pad|><|vision_end|>` to one pad token per merged patch. A video is sampled at 2 fps; each pair of frames is
one span written as `<t seconds><|vision_start|>` + pad tokens + `<|vision_end|>`, t the mean of the pair's timestamps. hived runs the
encoder (`engine/src/qwen/qwen_vision.cu`) on each span and the prefill takes those rows' embeddings from it.

## Thinking and tools

The chat template takes `reasoning_effort` low / medium / xhigh (default xhigh) and `enable_thinking`. Requests map as: no thinking
(`enable_thinking: false`, effort none / minimal) → the template's empty think block; low → low; medium → medium; high, xhigh, max → xhigh;
numbers 0–33 / 34–66 / 67+ → low / medium / xhigh; unset → xhigh. Tool calls are the template's
`<tool_call><function=NAME><parameter=KEY>value</parameter>…</function></tool_call>`, typed from the request's tool schemas.

## Explored and not adopted (2026-10-31, FP4-Local, one RTX PRO 6000 at 450 W)

Public-source review of the family (Qwen model card · NVIDIA FP4-Local card and discussions · SGLang day-0 blog and cookbook · the vLLM
fork's V3 notes · Local Inference Lab's CSF card · unsloth). What hive already had: the CSF decoding (the designer's codec name
`byte-window4-fixed-stream-u24-exceptions/1` matches the layout this reader recovered from the bytes), a replay rollback of the GDN state
for speculative verify (SGLang's ReplaySSM), a pruned MTP draft vocabulary (64K, every Hangul token in it) and MTP depth 4. A fused
hyper-connection kernel was measured by the vLLM fork on the same GPU with no serving gain (0.996) and is not pursued.

| Candidate | Measured | Verdict |
| --- | --- | --- |
| **IndexShare MTP** (SGLang): draft steps ≥ 1 of an MTP chain reuse step 0's QSA group selection instead of running the indexer — `qsa_select_reuse`, the saved top groups plus a tail from the group open at step 0 to the drafted position (kQsaSel 2051 → 2056) | greedy 24K output identical (lossless, as expected). 768-token decodes, C1: 172K 213/118 → 138/134 tok/s · 64K 124 → 201 · 24K 265 → 221 · 1K 227 → 259 — run-to-run swings larger than any difference (the `ignore_eos` tail text is degenerate and the acceptance with it is erratic). Mean accepted drafts per step over the same mixed run 2.12 → 1.94 | not adopted (code reverted). The draft-side indexer is a small part of a step here (a long-context C1 step at 172K is ~20 % slower than at 1K in all, 194 → ~130-160 tok/s with 70-90-token answers); a stale selection costs acceptance |
| **Dynamic request-context draft vocabulary** (vLLM fork V3: a 1,024-slot ring of prompt tokens outside the static list) | coverage of the 64K static list on 800 Korean answers (the hanja evaluation, thinking on and off): 99.5 % of answer tokens, 99.0 % of thinking tokens; the misses are bold-quote fragments and rare Han characters | not worth a kernel: ≤ 0.5-1 % of draft tokens are outside the list already |
| **MXFP8 dense GEMM for batched verify rows** (`HIVE_QWEN_DENSE_MX_ROWS=16`: lin() takes the block-scaled GEMM from 16 rows instead of prefill chunks only) — from the C8 profile: routed experts 45 %, dense projections 32 % (fn_gemv + gemv8) of the step; `bench_fn_gemv32` (tests/): at 32 rows gemv8 streams the MXFP8 matrices at 790-980 GB/s against 1,320 at ≤ 16 rows, and the small shapes (hc 324×10240 / 10240×320, shared 1280×2560 / 2560×640) are latency-bound at 12-27 µs whatever the kernel; the launch knobs (GEMV8_MINW 1/2, NWB 16, LEAN 0) make nothing faster | quality 169 (= N4) · coding bench 450 W, same tree: C1 284.9 vs 285.2 · C4 610.7 vs 618.1 · C8 803.6 vs 794.5 tok/s | not adopted (the knob stays at its default, prefill only): the per-call scale re-layout and activation quantization eat what the GEMM saves at 32 rows. The remaining C8 lever is kernel work — a gemv8 schedule that keeps its bandwidth at 32-48 rows, and fewer / fused launches for the latency-bound small projections |
| **Every draft verified at batch** (`HIVE_MTP_BATCH_FULL=1`: the batched chain's 4 drafts per sequence go to the verify instead of the batch gate's choice — the C8 profile showed 31.7 of 40 rows and ~2.1 accepted per sequence against 2.84 at C1) | coding bench 450 W: C1 293.2 (unchanged, single-sequence path) · C4 598.3 vs 618.1 · **C8 567.0 vs 813.3** | the gate is right: the extra verify rows cost more than the tokens they return; the knob stays off. The C8 gap to the vLLM fork is not in the draft count |
| **FP8 KV cache and FP8 indexer** (`--kv-cache-dtype fp8_e4m3 --qsa-indexer-dtype fp8_e4m3` in the SGLang / vLLM FP4-Local recipes; "validated" by the NVIDIA card) | not measured this round | open: halves KV memory (the 68 GiB run) and long-context reads; a lossy change that has to pass the quality gate (169) first |
| **Official sampling** — thinking 1.0 / 0.95 / 20 · instruct 0.7 / 0.8 / 20 + presence 1.5 (model card) | Han / kana mixing on 400 Korean prompts × 2 seeds was flat across settings; the quality harness is greedy | adopted in the serving engine's per-mode presets (not a hive change) |

Long-context C1 decode of this build (summaries, thinking off, 70-90 answer tokens): 1K 194 · 24K 158 / 142 · 64K 119 · 172K 137 / 128 tok/s;
the SGLang and vLLM FP4-Local recipes on the same GPU report 92 / 107 at 24K and 22 / 25 at 192K (HF discussion #10).

Known stale test: `engine/tests/test_qwen_qsa` reports `FAIL (4)` — the q / k / iq projections "differ from the reference beyond one bf16
rounding" — on this tree before any of today's changes (checked on the 31d sources); the selection and attention checks pass. The
reference predates a `qsa_prep` change; it needs regenerating, not the kernels.

## Switches of this family

| Switch | Default | Meaning |
|---|---|---|
| `HIVE_QWEN_VRAM_MB` | unset (whole card) | device memory budget of the process (startup) |
| `HIVE_QWEN_GPU_LAYERS` | unset (all, or chosen under `HIVE_VRAM_LIMIT_MB`) | layers 0..N−1 on the GPU, the rest on the CPU (§1b) |
| `HIVE_QWEN_PLAN_CACHE_MB` | 256 | expert cache the automatic layer choice leaves room for (§1b) |
| `HIVE_QWEN_DEVICE_ROUTE` | on | route decode / verify on the device while every expert is resident; `0` = always the host path |
| `HIVE_QWEN_MAPPED_ROUTE` | on | partial cache: device route, non-resident experts read from pinned host memory over PCIe (§1) |
| `HIVE_QWEN_MAPPED_MIN` | 0.6 | share of the experts the cache must hold for the mapped route |
| `HIVE_QWEN_HYBRID_ROUTE` | on | partial cache below the mapped threshold: device route with the non-resident experts on the CPU in graph host nodes (§1) |
| `HIVE_QWEN_HYB_PCIE` | 0 (`fn-48.env`: 4) | hybrid route: every N-th non-resident expert of a layer read over PCIe by the GPU (0 = all on the CPU) |
| `HIVE_QWEN_HYB_ROWS` | 0 | hybrid route: steps with more rows take the host path (0 = no limit) |
| `HIVE_QWEN_HYB_DIAG` | unset | `nocpu` = measurement only, the host nodes skip the CPU experts (wrong outputs) |
| `HIVE_QWEN_GRAPH_CAP` | 32 with a partial cache, else no cap | captured decode / verify graphs kept (least recently used dropped) |
| `HIVE_QWEN_GRAPH_ROWS` | 0 = any (`fn-48.env`, `fn-32.env`: 8) | forwards with more rows run eagerly |
| `HIVE_QWEN_PLE` | `ram` | PLE table placement: `ram` or `ssd` |
| `HIVE_QWEN_PLE_SSD_CACHE_MB` | 2048 | SSD mode row cache |
| `HIVE_QWEN_PLE_SSD_THREADS` | 64 | SSD mode readers (= queue depth) |
| `HIVE_QWEN_PREFILL_TILES` | 4 | max_chunk blocks per layer-major prefill call (used only while some expert is not resident) |
| `HIVE_QWEN_MTP_K` | 4 | MTP drafts per step (0 = off) |
| `HIVE_QWEN_DRAFT_VOCAB` | qwen.env: `config/q27-draft-vocab-64k.bin` | draft head over a token list (as `HIVE_GLM_DRAFT_VOCAB`) |
| `HIVE_QWEN_KV_BASE` / `HIVE_QWEN_KV_STEP` | max_chunk / 4096 | KV positions every sequence keeps / growth step |
| `HIVE_QWEN_PROF` | off | per-call-kind timing lines `[qwen-prof]` |
| `HIVE_QWEN_DENSE8` | qwen.env: `int8` | 8-bit copies of the dense projections (`0` = bf16) |
| `HIVE_QWEN_DENSE8_HEAD` | qwen.env: `1` | int8 verify head (fp32 logits) |
| `HIVE_QWEN_QSA_SPLIT` | 16 | KV splits of the decode attention |
| `HIVE_QWEN_CKPT_ASYNC` | on | checkpoint host copy on a worker thread |
| `HIVE_QWEN_CKPT_SNAP_MB` | 192 | images up to this size are snapshotted whole on the device (no wait in reset / load) |
| `HIVE_QWEN_MOE_PF_ROWS` | 8192 | rows per grouped prefill expert pass |
| `HIVE_QWEN_GEMV8_LEAN` | on | 2-blocks-per-SM 8-bit GEMV instances for 9–32 rows (`0` = the 1-block ones) |
| `HIVE_QWEN_GEMV8_MINW` | 4 | warps-per-block floor of the 8-bit GEMV |
| `HIVE_QWEN_MTPKV_SERIAL` | off | per-sequence MTP entry passes in the batched rollback (A/B only) |
| `HIVE_QWEN_GDN_ROWS_LEGACY` | off | previous GDN verify-row kernels, one launch per sequence (A/B only) |
| `HIVE_QWEN_TRACE_HOST` | off | `[qwen-host]` timing of save_image / load_image / reset_seq |
| `HIVE_QWEN_MOE_PF_V2` | on | W4A16 v2 grouped prefill expert kernels for the 128-row tiles (bit-identical; `0` = the first ones) |
| `HIVE_QWEN_MOE_A4` | off | **lossy**: W4A4 grouped prefill experts (NVFP4 activations, FP4 tensor cores) — quality gate FAIL, see below |
| `HIVE_QWEN_MOE_A4V2` | on (with `MOE_A4`) | the W4A4 v2 kernels (bit-identical to v1) |
| `HIVE_QWEN_DENSE_MX` | off | prefill dense GEMMs on the cuBLASLt block-scaled FP8 path, reading the checkpoint's MXFP8 matrices as stored (FP4-Local: lossless against the quality gate, 169 — the N4 production mix). No effect on a checkpoint without MXFP8 matrices; the former INT8-copy re-quantization path (lossy, 166) was removed 2026-10-31 |
| `HIVE_QWEN_QSA_MR` | off | multi-row prefill attention over the union of 4 rows' selections (same math, other fp32 order — gate FAIL, see below) |
| `HIVE_QWEN_GDN_WARPS` | 4 | warps per GDN prefill recurrence block (2 / 4 / 8, bit-identical) |
| `HIVE_QWEN_VERIFY_DYN` | on | verify part splits read from device memory: one verify graph per (rows, parts) serves any split (bit-identical) |
| `HIVE_QWEN_DRAFT_GRAPH` | on | batched MTP draft chain captured as a CUDA graph |
| `HIVE_QWEN_GRAPH_TRACE` | off | `[qwen-graph]` share of decode / verify forwards that replayed a captured graph |
| `HIVE_QWEN_DENSE8_SCOPE` | all | `head`: only lm_head gets the 8-bit copy (FP4-Local: INT8 copies of its other bf16 matrices failed the gate) |
| `HIVE_QWEN_MX_EXPAND` | off | FP4-Local: expand the MXFP8 dense matrices to bf16 at load instead of keeping them as stored (A/B) |
| `HIVE_QWEN_NORM_BF16` | off | round fp32 norm weights (FP4-Local) to bf16 as the NVFP4 checkpoint stores them (A/B) |
| `HIVE_QWEN_TOKEN_MASK` | unset | token list of the opt-in answer-span Han / kana filter (`config/fn-cjk-mask.bin`; requests ask with `hanja_filter`) |

`HIVE_GLM_NUMA_SPLIT`, `HIVE_GLM_SPIN_US`, `HIVE_GLM_PROMOTE` and the prefill switches of `GlmExperts` (`HIVE_GLM_PREFILL_CPU`,
`HIVE_GLM_PREFILL_ADAPT`) apply to this family's expert store as well. `HIVE_GLM_DEFER` and `HIVE_GLM_PREFETCH` do not (the engine
does not use expert deferral or prediction prefetch).

## Validation plan

The bring-up plan, in order (the GPU results are above and in the README's VRAM-size table):

1. **CPU** (done): `tools/test_cpu.sh` — PLE hash vs the oracle's vectors, the server family module (effort mapping, tool calls,
   preprocessing, spans), and the whole existing suite.
2. **Kernels** (GPU, no model): `test_qwen_gdn` (against an fp64 reference, chunked = one-shot, decode = T 1, verify rows and
   snapshots) and `test_qwen_qsa` (prep, group keys across calls, selection, attention against fp64).
3. **Forward** (GPU + checkpoint): `oracle/qwen4_oracle.py` (CPU torch reference written for this project) dumps per-layer streams and
   logits for a prompt; `qwen_check` dumps the engine's; `tools/compare_qwen.py` compares them. `QWEN_CHECK_DECODE_PATH=1` checks the
   decode path against the prefill path.
4. **Service, whole card** (`HIVE_QWEN_VRAM_MB` unset): quality suite and needle tests against the current SGLang service, then
   `tools/bench_chat.py` c1–c32 and long-prompt first-token times; fixes; then the kernel work listed above.
5. **50 % budget** (`HIVE_QWEN_VRAM_MB` ≈ 48000): the same measurements, CPU-miss and promotion behaviour, fixes.
6. **20 % budget** (≈ 19000): the same.
7. PLE on the SSD at each step where it matters (first-token time of long prompts).

## Correctness notes to verify against the reference

- The PLE takes the real token ids of image / video pad tokens; SGLang replaces those ids by per-image pad values before the PLE sees
  them. The engine keeps the model's own token ids (to be compared on image prompts).
- The MTP layer's input at an image or video pad position takes the vision encoder's row (as the main model's embedding does) for the
  entries written during the prompt's prefill.
- The MTP entry j combines h_j and x_{j+1} at rope position j (the speculative worker's convention of reusing the target positions).
- Activations of the NVFP4 experts stay BF16 (weights dequantized on the fly), as in the GLM family; SGLang's FlashInfer path
  quantizes them to FP4 with the checkpoint's input scales. Outputs therefore differ slightly from SGLang's, toward the BF16 reference.
