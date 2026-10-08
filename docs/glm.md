# GLM-5.3-Flash in Eke Hive

Eke Hive runs a second model family next to DeepSeek-V4.1-Flash: **GLM-5.3-Flash** (Z.ai, MIT license), from the NVIDIA ModelOpt
NVFP4 checkpoint plus the FP8 tensors of the original Z.ai release. The daemon is the same source (`engine/src/hived.cpp`) built a
second time as `hived_glm`; the launcher picks it from the checkpoint's `config.json` (`model_type: glm5_next`), so the same scripts,
API server, sessions, prompt-prefix reuse and sampling serve both families.

## Quick start

```bash
# 1. Checkpoint (NVFP4, ~191 GB) and the Z.ai FP8 tensors (~2.1 GiB; --mtp-experts adds the NextN layer's experts, ~7 GiB)
huggingface-cli download nvidia/GLM-5.3-Flash-NVFP4 --local-dir /models/GLM-5.3-Flash-NVFP4
python3 scripts/glm-fetch-fp8.py /models/GLM-5.3-Flash-zai-fp8 --mtp-experts
# 2. Build (builds hived and hived_glm)
scripts/build.sh
# 3. Start (the launcher sees a GLM checkpoint, starts hived_glm and reads config/glm.env; no scripts/prepare.sh)
HIVE_CKPT=/models/GLM-5.3-Flash-NVFP4 scripts/hive-start.sh
```

`config/glm.env` holds the measured defaults (below); `HIVE_CONFIG=config/glm.env` selects it explicitly. The DeepSeek engine
switches of `config/hive.env` (`HIVE_DECODE_*`, `HIVE_PREFILL_*`, Engram, ...) do not apply to this family; the daemon-level
switches do (scheduler, sessions, prefix sharing, MTP gate, cache fit, sleep — see [configuration.md](configuration.md)).
Cold start to ready: 95–97 s; resident host memory right after start ~385 GB (358 GiB), growing with open conversations.

## Model and what the engine does with it

| Part | Format used | Where it lives |
|---|---|---|
| 42 MoE layers × 288 routed experts (top-8) | NVFP4 (checkpoint) | pinned RAM (159.5 GiB) + VRAM cache; misses computed by the CPU |
| KDA linear attention (34 layers) | BF16 (checkpoint; Z.ai ships them in BF16 too) | VRAM |
| DSA attention projections (11 layers), shared experts | FP8 128×128 blocks (Z.ai release) | VRAM |
| dense MLP of layers 0-2 | NVFP4 | VRAM |
| NextN (MTP) layer | its routed experts FP8 (Z.ai), rest as above | VRAM |
| vision encoder (24 blocks, 2×2 merge, merger to 4,096) | BF16 (checkpoint — not quantized there) | VRAM, ~1.1 GB, loaded on the first image and kept until the next sleep |

Precision rule: where Z.ai ships FP8 the engine uses that FP8; what Z.ai keeps in BF16 stays BF16.

## Images and video

`image_url` / `video_url` parts (OpenAI) and Anthropic image blocks are accepted. The server (`server/families/glm.py`) prepares
them as the model's `processor_config.json` describes: aspect-preserving resize into the token budget (images 16–8,000 tokens,
video up to 240,000), black padding to multiples of 28, CLIP mean/std, patches in 2×2 merge-block order; an image is one temporal
patch (the frame twice), a video is sampled at 2 fps and each pair of frames is one span written into the prompt as
`<|begin_of_image|>` + its image tokens + `<|end_of_image|>` + its start time (`1.0 seconds`). hived runs the encoder
(`engine/src/glm/glm_vision.cu`: patch embedding, 24 blocks with per-head RMSNorm and axial 2D rotary positions, clamped SwiGLU,
2×2 downsample, merger) on each span and the prefill takes those rows' embeddings from it. The encoder's memory comes from the
expert-cache tail like long-context KV: its weights (~1.1 GB, loaded at the first image) stay until the next sleep, its per-request
working memory is given back after each request. Video needs PyAV in the server image (`requirements.txt`).

A functional check (not a benchmark): text in an image is read correctly, a photo is described correctly, and a short clip's
order of events and colors are right (small text in video can be misread at ~280 tokens per pair of frames).

## Correctness

`glm_check` (`engine/tests/glm_check.cpp`, built as `engine/build/glm_check`) compares the engine with the transformers 5.18 implementation (`Glm5NextTextModel`, run on the CPU with the
same NVFP4 weights dequantized to fp32) layer by layer on three chat prompts:

| | BF16 path | Z.ai FP8 path |
|---|---|---|
| next-token top-1 | 3/3 match | 3/3 match |
| top-20 overlap | 18-19/20 | 19-20/20 |
| lowest hidden-state cosine over 45 layers | 0.99996 | 0.99996 |

The KDA and DSA kernels have their own tests against the transformers modules (`test_glm_kda`: relative error 8.5e-5, chunked
prefill and batched decode bit-identical to one-shot; `test_glm_dsa`: selected token sets identical to transformers at 64-6000 tokens). These three read reference dumps produced with transformers;
the scripts that generate the dumps are not included in this repository.

## Thinking

GLM-5.3-Flash cannot turn thinking off: Z.ai documents `thinking.type` as `enabled` only and `reasoning_effort` as `low` / `high` /
`max` (default `max`); the chat template always opens `<think>`. A request for no thinking (`reasoning_effort: none`,
`enable_thinking: false`) is mapped to the official minimum `low` (`HIVE_GLM_NOTHINK=low`, default). `HIVE_GLM_NOTHINK=empty` instead
prefills an empty thinking block — not an official mode. Measured against `low` on the same server under the same load (quality suite without the long-context
items, 144 items each): `empty` scored 136 vs 142 — QA 73 vs 78 / 80 (four of the five misses are right answers without the
requested "Answer:" line), code 19 vs 20 — and took 1,030 s vs 574 s in all, because its answers are longer (a translation request
came back as 107 tokens with a preamble instead of 23). At `low` the model already writes little or no reasoning for such requests,
so `low` stays the default.
A request without an effort (or with a value the family does not know) runs at `high`, not the template's default `max`:
community measurements on the official checkpoints (NVIDIA developer forum) put `high` at 36-45 % less time than `max`, at a
cost of 1-4 benchmark points. Mapping: `none`/`minimal` → `low`, `low` → `low`, `medium`/`high` → `high`,
`xhigh`/`max` → `max`, numbers 0-49 / 50-89 / 90+ → `low` / `high` / `max`.

## Performance

Workstation: RTX PRO 6000 96 GB (PCIe 4.0 x16, 27.9 GB/s measured H2D), Threadripper PRO 3975WX (2 NUMA nodes, AVX2), 1 TB DDR4.
Expert cache: about 4,690 slots (38.8 % of the 12,096 experts) with `config/glm.env`, fewer while a long conversation holds KV
(see [Context and KV memory](#context-and-kv-memory)).

### Through hived (the API), non-thinking requests — current defaults (`config/glm.env`)

Measured in two runs (2026-10-07), every adopted change on (262,144 context with KV memory that follows use, 32 CPU threads, cache-aware
routing λ 0.1, expert deferral, KDA verify rows, MTP k ≤ 3 with the 64K-token draft head, prefix sharing, 8 seats with batched MTP verify,
layer yields with mid-size admission, sorted promotion victims), with the real-chat benchmark `tools/bench_chat.py` — the headline harness for both families: twelve different
chat prompts (5 Korean, 3 English, 2 code, 2 creative), answers up to 600 tokens, `reasoning_effort: none`, decode speed after the first token:

| Streams | Total | Per stream | Time to first token (median / max) |
|---|---|---|---|
| 1 | **70.8 tok/s** (Korean 75.5 · English 67.9 · code 73.9 · creative 54.4) | 71.2 | 0.51 / 0.96 s |
| 2 | 87.8 | 47.2 | 0.54 / 0.92 s |
| 4 | 103.9 | 28.6 | 0.53 / 1.73 s |
| 8 | 130.6 | 17.5 | 0.63 / 3.40 s |
| 16 | 130.3 | 17.4 | 12.7 / 39.2 s (`HIVE_MAX_BATCH` 8 — the rest wait) |
| 32 | 130.9 | 16.9 | 45.1 / 107.8 s |

2026-10-05 (before the draft head of 2026-10-07): 70.3 / 85.2 / 97.9 / 129.1 / 128.3 / 129.6 tok/s. Before 2026-10-05 (4 seats, no batched
verify, one run): 65.2 / 66.6 / 89.3 / 88.8 / 91.6 / 93.3 tok/s. With two streams both drafts
are verified in one step; with more each step decodes several sequences without drafts.

2026-10-09 (a newer build, the shipped OpenAssistant / NSMC draft list, one run each): 68.1 / 86.5 / 103.5 / 129.0 / 127.4 / 128.9 tok/s at 600 W and 69.4 / 85.6 / 102.5 / 129.3 / 129.8 / 130.5 tok/s with the GPU capped at 300 W; first token for 16.9K / 42K / 55K-token prompts 7.21 / 11.31 / 13.74 s (300 W: 8.17 / 15.25 / 18.90 s).

| Prompt | First token | Prefill | Decode after it |
|---|---|---|---|
| 17K tokens | 7.4 s | 2,307 tok/s | 54.9 tok/s |
| 42K tokens | 11.7 s | 3,639 tok/s | 49.6 tok/s |
| 54K tokens | 14.0 s | 3,925 tok/s | 45.1 tok/s |
| 100K tokens | 26.4 s | 3,833 tok/s | 45.1 tok/s |
| 200K tokens | 53.7 s | 3,787 tok/s | 43.4 tok/s |
| 250K tokens | 66.3 s | 3,837 tok/s | 40.0 tok/s |

The 100K–250K rows are from the previous build (one request's prefill — that path did not change). Decode after a long prompt slows with its length: the DSA indexer scores every earlier position at each step, and the expert
cache is smaller while that conversation holds its KV (below).

Context limit: 262,144 tokens (`HIVE_MAX_CTX`). Quality suite (163 items, `--effort low`): 158 / 163; needle retrieval at 64K,
128K and 250K: 21 / 21.

### Engine only (`glm_check`, greedy, 600 tokens, one request)

| Change (cumulative) | tok/s |
|---|---|
| first end-to-end run (CPU shared with another job) | 14.6 |
| quiet machine + tensor-core MoE decode kernel (1.45 TB/s) | 21.6 |
| spin-waiting CPU expert pool | 24.3 |
| NUMA half records (node-pinned workers read local memory only) | 25.8 |
| 24 CPU threads | 27.7 |
| fused decode kernels + no H2D copies on the critical path | 33.5 |
| next-layer expert prefetch (1 per layer) | 34.1–36.2 (two runs) |
| MTP k = 3 (77-83 % of drafts accepted, 3.3-3.5 tokens per verify step) | 50-53 |

### Prefill (hived, cumulative)

| Change | 15K tokens | 42K tokens | 0.8K tokens |
|---|---|---|---|
| 4K blocks, every used expert streamed per block | 22.7 s | 61.4 s | 5.2 s |
| 16K blocks | 8.0 s | 23.1 s | 5.4 s |
| copies overlapped with compute + CPU share of few-row experts | 6.9 s | 19.5 s | 3.5 s |
| layer-major over up to 64K rows (buffers borrowed from the cache) | 6.9 s | 11.0 s | 3.5 s |

The limit is PCIe 4.0: an expert that is not resident has to cross it (13.5 MiB, 0.507 ms). Layer-major prefill sends each such
expert once per layer for the whole prompt instead of once per 16K block; experts used by only a few rows (≤ 24) are computed by the
CPU instead.

### Where a c1 step goes (hived, `HIVE_GLM_PROF=2`, real chat)

Measured before cache-aware routing, with 24 CPU threads (history: the hit rates are those of λ 0). A single-stream step is a draft (4.1 ms) plus a verify of 2-4 rows. Per verify call:

| | 2 rows | 3 rows | 4 rows |
|---|---|---|---|
| routed experts (of which the CPU computing the VRAM misses) | 17.3 (14.0) | 26.2 (23.2) | 34.6 (31.7) |
| KDA, 34 layers (projections · recurrence) | 9.1 (5.1 · 2.2) | 10.0 (5.1 · 3.1) | 10.5 (5.0 · 3.7) |
| DSA · hyper-connections · shared expert · router · head | 3.1 · 2.4 · 1.7 · 1.1 · 1.0 | 3.2 · 2.3 · 1.8 · 1.1 · 1.0 | 3.0 · 2.1 · 1.8 · 1.1 · 1.1 |
| wall | 37.8 ms | 47.8 ms | 56.1 ms |
| expert-cache hit rate | 88.0 % | 85.8 % | 84.7 % |

Draft acceptance in real chat: 72.7 %, 2.5 tokens per step. The CPU phase reads the missed experts at 77-91 GB/s; the same
kernel reaches 105 GB/s (12 threads per node) and 118 GB/s (16 per node) in a node-local stream benchmark against a measured
128 GB/s read ceiling (one CCD of the 3975WX tops out near 37 GB/s).

Changes from that breakdown (A/B, interleaved, restart per configuration, 12 chat requests each):

| Change | Measured |
|---|---|
| 32 CPU threads instead of 24 (`config/glm.env`) | CPU phase 79 → 86 GB/s; c1 median 45.6 → 47.0 tok/s (two pairs) |
| CPU-expert scratch reused instead of allocated per layer | CPU phase per 3-row verify 21-22 → 19-20 ms |
| KDA verify rows in one launch (`HIVE_GLM_KDA_ROWS`, bit-identical) | one layer, 4 rows: 36.9 → 16.4 µs; 3-row verify call 44.0 / 44.5 → 43.1 / 43.4 ms (outside the CPU phase 23.1-23.3 → 22.2-22.4 ms) |
| CPU-expert chunks sized to the worker count (`HIVE_GLM_CPU_ADAPT`, bit-identical) — after cache-aware routing and deferral a CPU layer had 1.5 jobs, fewer 128-row chunks per node than workers | gate/up workers busy 58 → 80 %, CPU expert phase 70.0 → 81.2 GB/s (0.203 → 0.174 ms per expert, 4-row verify); real-chat c1/c4/c8 unchanged (68.6 / 89.0 / 93.3 vs 68.2 / 88.5 / 93.2 tok/s — the CPU phase mostly overlaps the GPU at these loads) |
| Decode/verify logits into a pinned buffer, hived's verify row vector reused (values unchanged) | removes a pageable copy into freshly allocated memory each verify step (rows × 154,880 × 4 B); real-chat c1 69.4 / 67.8 vs 68.2 tok/s — kept for the host work it saves |
| Early routing on the fast decode path (`HIVE_GLM_EARLY_ROUTE`, the DeepSeek post-and-gate code): routing and the CPU input rows reach the host right after the router, the experts are launched while the shared expert runs | output identical to off (300-token greedy streams; off vs off diverged at token 2); real-chat, interleaved ×2: c1 69.7 → 70.0, c2 86.7 → 85.2, c4 95.3 → 99.4, c8 128.3 → 128.2 tok/s |
| Promotion victims sorted once per step instead of a scan of every slot for each used expert (same choices) — host timestamps showed `after_step` holding the GPU idle 1.88 ms per verify step | real-chat, interleaved ×2 at 8 seats + batched MTP: c1 67.3 → 70.4, c2 78.6 → 86.8, c4 88.1 → 96.6, c8 126.0 → 129.9 tok/s; promotions per verify unchanged (28 vs 25–29) |
| Session snapshots copy only the new rows (`HIVE_CKPT_DELTA`), prompt checkpoint after the first token (`HIVE_DEFER_CKPT`) | service log, before / after: prefill end → first token, mean 746 ms (1,097 ms at 64K+) → 0; shared-prefix checks (`HIVE_CKPT_VERIFY`) all ok |
| Prefill CPU / streaming split from measured costs (`HIVE_GLM_PREFILL_ADAPT`) | service log, before / after: 65–256-row prefill 1,115 → 862 ms, 257–1,024 rows 2.99 → 2.85 ms/row, larger prompts unchanged |
| MTP draft head over 65,536 frequent tokens (`HIVE_GLM_DRAFT_VOCAB`) — three drafts read the whole bf16 `lm_head` (1.27 GB) three times per step | outputs unchanged (full-head verify); real-chat, interleaved ×2: c1 69.4 → 71.4, c4 98.1 → 104.6 tok/s, c2 / c8 unchanged; draft acceptance 0.68 / 0.69 (measured with a ShareGPT-based list; the shipped list covers ~1 point fewer chat tokens) |
| Mid-size requests admitted at layer yields (`HIVE_LAYER_YIELD_MID=16384`, DeepSeek step 23) | 96K-token prefill with a 2K request arriving 8 s in: its first token 18.9 → 2.9 s (×2); the long prompt waits for it (22.7 → 31.0 s with a 12K request served too) |
| RAM budget for evicted conversations 96 GB (`HIVE_HOST_SESSION_MB=98304`) | twelve 145K-token conversations, then the first one again: 37.0 s → 1.6 s to the first token |

### Lossy changes measured with quality (on top of each other)

| Change | Speed | Quality (163 items) | Verdict |
|---|---|---|---|
| Cache-aware routing (`HIVE_CACHE_PRIOR`, Skliar et al. TMLR 2025) λ 0.1 | c1 47.8 → 61.3 tok/s, cache hits 86.7 → 96.3 % | 158 vs 160 / 158 for λ 0 run twice | adopted |
| same, λ 0.2 | 63.5 tok/s, hits 97.9 % | 157 (a long-context item flipped) | not adopted |
| Expert deferral (`HIVE_GLM_DEFER`, after KTransformers SOSP'25) | 60.6 → 63.6 tok/s | 157 / 157 | adopted |
| Skip low-weight misses (`HIVE_GLM_SKIP_MISS`) 0.05 / 0.1 | +0 / +2 % (noise) | 158 / 160 | kept off |
| Batched MTP verify across sequences (`HIVE_MTP_BATCH`) | c4 82.4 / 82.5 → 84.4 / 82.8 (4 seats); with 8 seats c2 66.6 → 79.5 tok/s (two runs) | (lossless) | adopted with `HIVE_MAX_BATCH=8` |
| 8 sequences per step instead of 4 (`HIVE_MAX_BATCH=8`) | c8 93.2 → 127.8 tok/s, c8 first token 23 → 0.7–0.9 s; c1–c4 within the base spread | (lossless) | adopted |
| MTP k 4 / 5 instead of 3 | 46.7 / 45.8 vs 48.1 tok/s | | kept at 3 |

### What did not help (measured, kept off)

- W4A8 CPU kernel (llama.cpp's nvfp4·q8_0 method): one NUMA node saturates at about 50 GB/s with either kernel, so one row is not faster
  (49.4 → 44.8 GB/s with 12 threads) and accuracy drops (7e-3 relative); faster only with 4+ rows.
- More than one predicted expert copied ahead per layer: 2 → 30.7 tok/s, 3 → worse (the copies compete with the CPU for DRAM bandwidth
  and evict useful experts); the same for verify steps (`HIVE_GLM_PREFETCH_ROWS2` 2/3/4: 44.5/42.0/36.3 vs 53.1 tok/s).
- Layer-major buffers reserved permanently: 3.6 GiB less cache → c1 −6 %; borrowing cache slots only while a long prompt runs keeps both.
- Pinning each CPU worker to its own physical core, spread over the L3 domains: c1 45.2 / 45.3 vs 46.3 / 45.1 tok/s unpinned, same GB/s.
- Four independent accumulators in the 1-row NVFP4 CPU kernel: +2 % in isolation (105.5 → 107.4 GB/s) — the kernel is not latency-bound.
- Equal row ranges per CPU worker instead of 128-row chunks (`HIVE_GLM_CPU_BALANCE=1`): 18.65 / 19.98 vs 19.44 / 19.73 ms per 3-row verify.

## Tool calls

OpenAI `tools` / Anthropic tool definitions go through the checkpoint's own chat template. The model writes a call as
`<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value>…</tool_call>`; the server (`server/families/glm.py`) turns it
into a standard tool call, taking each value as JSON when it parses as JSON (the template writes non-string arguments that way)
and as a string otherwise. Tool calls are parsed when the call is complete and sent in one piece — the incremental tool-call
streaming of the DeepSeek path (`HIVE_TOOL_STREAM`) does not apply to GLM.

## Prompt-prefix reuse

The same as DeepSeek: a conversation continues from its own state, and with `HIVE_PREFIX_SHARE` + `HIVE_PREFIX_ADAPTIVE` (on in
`config/glm.env`, with `HIVE_PREFIX_EXTRA_CHUNKS=0` and `HIVE_PREFIX_FIRST_TURN=seen` since 2026-10-07 — a first turn is cut at its
system block only once another conversation sent the same block; [performance.md](performance.md) step 29) conversations share boundary snapshots — the KDA recurrent state is part of each snapshot. Measured with an
8.6K-token system prompt, answers correct in every case (prefill times taken while another CPU job ran — compare within a column, not with the tables above):

| | sharing off | sharing on |
|---|---|---|
| new conversation | 0 cached · 6.2 s | 0 cached · 6.4 s |
| new conversation, same system prompt | 0 cached · 5.8 s | **8,625 cached · 0.52 s** |
| second turn of the first conversation | 8,648 cached · 0.40 s | 8,640 cached · 0.66 s |
| system prompt changed at its start (control) | 0 cached · 5.6 s | 0 cached · 6.0 s |

Responses report the reused tokens as `usage.prompt_tokens_details.cached_tokens` (OpenAI) and `cache_read_input_tokens` (Anthropic).

Snapshots (prompt checkpoints, boundary snapshots, archived sessions) copy the KDA conv / state whole and the DSA latent / indexer and MTP
rows up to the position. With `HIVE_CKPT_DELTA` (on in `config/glm.env`) a snapshot shares the rows the previous snapshot of the same
sequence already holds — rows below a committed position are never rewritten; MTP rows only below that snapshot's exact MTP position —
and copies only the new ones; `HIVE_DEFER_CKPT` (on) takes the prompt checkpoint after the first token is sent. Motivation (service
traffic, 2026-10-05, before both): the synchronous full copy came before the first token and added 746 ms to the mean TTFT, 1,097 ms
for prompts of 64K tokens and more. `HIVE_CKPT_VERIFY=N` compares every N-th shared prefix with the device rows (`[glm-ckpt] verify`).

## Context and KV memory

`HIVE_MAX_CTX=262144` only reserves address space. Each sequence's DSA and MTP caches (about 19 KB per position: 11 DSA layers
plus the NextN layer, a 512-wide compressed KV and the indexer keys) are growable buffers (`engine/include/hive/grow_buf.h`, CUDA VMM): the
first `HIVE_GLM_KV_BASE` positions (default: one 16K prefill block) are mapped when the sequence is created, and more is mapped in
steps of `HIVE_GLM_KV_STEP` positions (4,096) as a conversation grows. The memory comes from the free VRAM above
`HIVE_CACHE_RESERVE_MB`; when that is not enough, the expert cache gives up tail slots (its slot area is mapped in 56 MiB steps
for this), and when the sequence is reused for a new conversation its KV is trimmed back and the slots are taken back.

Measured against the old fixed allocation of 65,536 positions: session pool 4 × 1,370 → 4 × 458 MiB, expert cache
4,420 → 4,693 slots, the same chat speed (above), and prompts up to 250K tokens. A long conversation takes the expert-cache slots its KV needs for
as long as it keeps it; after the 250K needle runs, with several long conversations still held, the cache stood at ~3,600 slots until their sequences
were reused.
A failed growth gives back what it mapped before the error is returned.

## Sleep and wake

The GLM build supports the same sleep levels as DeepSeek (`scripts/hive-sleep.sh`, `HIVE_SLEEP_LEVEL`); level 3 (`HIVE_SLEEP_LEVEL=3`) needs
`HIVE_SLEEP_VMM=1` at start, which `config/glm.env` sets. Measured with `HIVE_MAX_CTX=262144`:

| | sleep | VRAM while asleep | wake (allocation + cache refill) |
|---|---|---|---|
| level 1 (expert cache and staging freed), right after the 100K–250K prompts | 0.19 s | 42.9 GB (those sessions keep their KV) | 2.0 s (0.07 + 1.9) |
| level 3 (+ weights, sessions, scratch to RAM), same state | 9.0 s (the long sessions' KV moves to RAM) | 984 MiB | 3.2 s (1.3 + 1.9) |
| level 3 with one 120K-token conversation open | 3.8 s | 984 MiB | 3.0 s (1.3 + 1.6) |

The time to sleep at level 3 grows with the KV of the open conversations. After the 120K run the next turn of that conversation
answered correctly with its first token after 2.6 s (118,169 of 118,184 prompt tokens reused), and a new short chat after 0.58 s.

Every resident expert is restored on wake (3,738 of 3,738 and 3,226 of 3,226 in the two level-3 runs). Answers after a wake are correct but not word-for-word the same as
before — three identical requests without any sleep also differ, because residency (CPU or GPU for a missed expert) changes
with timing (see the numerics note under [Speculative decoding](#speculative-decoding-nextn--mtp)).

## Speculative decoding (NextN / MTP)

The checkpoint's NextN layer drafts tokens; hived's existing MTP gate decides per step how many drafts to verify (the same gate as the
DeepSeek path). GLM needs the gate's measured cost tables — `HIVE_MTP_GATE2=1` and `HIVE_MTP_GATE3=1`, both in `config/glm.env` —
because the default gate's fixed cost table is DeepSeek's; `HIVE_MTP_GATE3` also leaves out the step costs of the decode steps
right after a prefill, while the expert cache is still warming. The engine side (`engine/src/glm/glm_mtp.cpp`):

- **Draft**: the NextN layer takes `[enorm(embedding of the next token) | hnorm(main model's final hidden)]`, runs one DSA attention
  layer with its own cache and a MoE whose 288 experts are the Z.ai FP8 copy held in VRAM (6.75 GiB), then the main `lm_head`.
- **Verify**: the main model runs `[token, drafts…]` as one decode step of up to 8 rows of the same sequence. The KDA recurrence is
  processed row by row and its state after each row is kept, so a rejected tail is undone by restoring that snapshot; DSA caches are
  position-indexed and only the position moves back.
- **NextN cache**: entry j pairs the main hidden of position j with token j+1; the main path writes each entry as soon as both are
  known, so every entry the drafts read is exact (draft steps beyond the accepted point write scratch entries that are overwritten).

Measured (greedy, `glm_check` with `GLM_CHECK_MTP`): with expert prefetch off the tokens generated with MTP k=3 were identical to plain
greedy decoding (32 and 120 tokens); 77-83 % of the drafts are accepted (3.3-3.5 tokens per verify step). The draft chain runs on the GPU
without host synchronization (4.1 ms for 3 drafts); the verify step of 4 rows (62 ms on that early build) is bound by the CPU computing the missed experts
of the union of the rows' choices.

Numerics note: which experts are resident changes with timing (asynchronous promotions and prefetches), and a missed expert is computed
by the CPU (fp32) instead of the GPU (tensor cores), so runs are not bit-identical; near-ties can resolve differently after many tokens.
The reference comparison (next-token top-1 and hidden states) passes on every path (BF16, FP8, short/streamed/layer-major prefill).

## Switches of this family

| Variable | Default | Meaning |
|---|---|---|
| `HIVE_GLM_FP8_DIR` | unset | directory of the Z.ai FP8 tensors (`scripts/glm-fetch-fp8.py`); unset/`0` = BF16 attention projections and shared experts, no MTP |
| `HIVE_GLM_MTP_K` | 3 | drafts per step for the NextN layer; `0` = do not load it |
| `HIVE_GLM_PREFETCH` | 0 (`config/glm.env`: 1) | decode only: experts of the next MoE layer copied ahead per layer, predicted by applying that layer's router to the current FFN input |
| `HIVE_GLM_NUMA_SPLIT` | on | CPU experts as NUMA-local half records with node-pinned workers; `0` = one interleaved copy |
| `HIVE_GLM_SPIN_US` | 3000 | CPU workers spin this long for the next batch before sleeping |
| `HIVE_GLM_DECODE_FUSED` | on | fused decode kernels (`glm_decode.h`); `0` = the cuBLAS/unfused path |
| `HIVE_GLM_SHORT_PREFILL` | 64 | prompt chunks up to this many rows (capped at 64) use the decode expert path (VRAM hits + CPU misses) instead of streaming experts |
| `HIVE_GLM_PREFILL_TILES` | 4 | blocks of `max_chunk` rows one prefill call runs layer-major (1 = block by block) |
| `HIVE_GLM_PREFILL_ELASTIC` | on | layer-major buffers borrowed from the expert-cache tail while a long prompt runs; `0` = reserved at start |
| `HIVE_GLM_PREFILL_CPU` | on | prefill experts computed by the CPU next to the GPU's streamed ones (fewest rows first, while the CPU share finishes before the copies) |
| `HIVE_GLM_PREFILL_ADAPT` | on | the CPU/streaming split uses the measured cost of a streamed record and of an 8-row CPU pass (moving averages over the previous prefill layers, shown on `[glm-prefill]` as `est copy … pass …`) instead of fixed 0.507 / 0.17 ms; an expert goes to the CPU while its CPU cost is below one copy. Only the split changes; the CPU (fp32) and GPU (bf16) results of an expert can differ by rounding, as they already did when the cache state moved the split. `0` = the fixed estimates and the 24-row limit. Motivation: a record measured 0.65–0.85 ms, the CPU finished its share early and the GPU streamed 1,257 records per 65–256-row prefill (1,072 of 1,135 ms) |
| `HIVE_GLM_PREFETCH_ROWS2` | 0 | prefetch budget for steps with ≥ 2 rows (0 = same as `HIVE_GLM_PREFETCH`) |
| `HIVE_CACHE_PRIOR` | 0 (`config/glm.env`: 0.1) | decode routing biased toward VRAM-resident experts by λ × the layer's score range; top-2 always kept (`HIVE_CACHE_PRIOR_TOPJ`) |
| `HIVE_GLM_DEFER` | 0 (`config/glm.env`: 1) | CPU misses ranked 3rd or lower computed while the GPU goes on, added one MoE layer later |
| `HIVE_GLM_SKIP_MISS` | 0 | skip missed experts ranked 3rd or lower whose weight is below this fraction of the row (measured within noise) |
| `HIVE_GLM_DRAFT_VOCAB` | unset (`config/glm.env`: `/hive/config/glm-draft-vocab-64k.bin`) | file of int32 token ids (`tools/build_draft_vocab.py`): the MTP draft head scores only these rows of `lm_head`, read in place (no copy, no VRAM taken). The verify step scores against the full `lm_head`, so outputs are unchanged; a token outside the list can only be missed as a draft. Shipped list: 65,536 tokens ranked by frequency in OpenAssistant oasst2 English assistant messages and the NSMC Korean corpus, equally weighted (`--balance --lang en`; [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES) §5); it covers 98.5 % / 98.4 % of the tokens of English / Korean chat answers (ShareGPT, 200K answers each). The speed numbers were measured with an earlier list built from ShareGPT answers (99.4 % / 99.8 %) |
| `HIVE_GLM_PROMOTE` | 8 | misses promoted per token of a step (4 / 16 measured equal / slower) |
| `HIVE_MTP_BATCH` | off (`config/glm.env`: 1) | verify the drafts of several sequences in one step: c2 66.6 → 79.5 tok/s with `HIVE_MAX_BATCH=8` (earlier, at 4 seats, c4 within noise) |
| `HIVE_GLM_LAUNCH_PROBE` | 0 | measurement: every N-th decode step, GPU time vs host enqueue time of each layer's front (totals every 50 probed steps) |
| `HIVE_GLM_NOTHINK` | `low` | API server setting: what a no-thinking request becomes — `low` (official minimum effort) or `empty` (prefilled empty thinking block) |
| `HIVE_PROFILE` / `HIVE_TRACE_CACHE` / `HIVE_TRACE_MTP` | off | monitoring lines in the DeepSeek formats, read by `tools/hive_monitor.py`: `[cache]` per decode / verify step; every N-th step (`HIVE_PROFILE=N`) `[profile]` section times, `[decode-host]` (where the decode stream sat empty: routing to the host, waiting for CPU experts, between layers; plus the host's deferral wait) and `[early-route]`, from CUDA events — no host synchronization added; `[mtp]` from hived; host time: `[step-host]` (hived), `[call-host]`, `[fwd-host]` and one `[glm-prefill]` line per prefill call (see configuration.md). Ignored while `HIVE_GLM_PROF` is set (`[glm-prefill]` and `[step-host]` / `[call-host]` are not) |
| `HIVE_GLM_PROF` | 0 | 1 = host-synchronized phase timings, 2 = CUDA-event phase timings; hived prints a per-call breakdown (phases, cache hits, CPU expert GB/s) every 400 calls of a kind (measurement only) |
| `HIVE_GLM_KDA_ROWS` | on | KDA recurrence of a verify step's rows in one launch (bit-identical); `0` = one launch per row plus snapshot copies |
| `HIVE_GLM_EARLY_ROUTE` | on | fast decode path: routing posted right after the router, experts launched while the shared expert runs; `0` = stream synchronization after the shared expert |
| `HIVE_GLM_CPU_ADAPT` | on | CPU-expert row chunks halve (not below 32 rows) until every worker of a node has one; `0` = fixed 128-row chunks (bit-identical either way) |
| `HIVE_GLM_CPU_BALANCE` | 0 | `1` = equal row ranges per CPU worker instead of 128-row chunks (measured within noise) |
| `HIVE_GLM_PREDICT_EVAL` | 0 | measure next-layer expert prediction recall (outputs unchanged) |
| `HIVE_GLM_KV_BASE` | `max_chunk` (16384) | positions of KV memory every sequence keeps (mapped at creation, kept on reset) |
| `HIVE_GLM_KV_STEP` | 4096 | positions mapped per KV growth step |
| `HIVE_GLM_NUMA` | (split) | measurement: `local` = plain mapping that follows the process NUMA policy (e.g. `numactl --membind`) |
| `HIVE_GLM_DEC_NO_PDL` | off | measurement: decode kernels without programmatic dependent launch |
| `HIVE_GLM_HC_CUBLAS` | 0 | measurement: hyper-connection mixing through the cuBLAS reference path |
| `HIVE_GLM_DEBUG` | 0 | prefill debug output |
| `HIVE_GLM_SLOTS_VMM` | on | expert-cache slot area mapped in steps so long contexts can borrow its tail; `0` = one `cudaMalloc` (KV can then only grow into the free headroom) |
