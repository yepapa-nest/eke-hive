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
