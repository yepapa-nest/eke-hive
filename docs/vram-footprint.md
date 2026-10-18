# VRAM footprint of the four hive models, and a single VRAM cap

Status: analysis and design (2026-10-09). Nothing here was run on a GPU for this document. Every number is tagged:

- **M** = measured, with its source in brackets;
- **E** = estimate (computed from checkpoint headers and the allocation code, or extrapolated from a measured number).

Sources used for **M** numbers:

| Tag | Source |
|---|---|
| [ds-log] | DeepSeek service log on the reference machine, `hived.log` (2026-10-09 start, `config/hive.env`, RTX PRO 6000 96 GB) |
| [glm-log] | GLM service log on the reference machine, `hived.log` (2026-10-09 starts, `config/glm.env`) |
| [hdr] | safetensors headers of the checkpoints on the reference machine (header-only scan, byte counts per tensor) |
| [glm.md] | `docs/glm.md` (sleep table, KV section) |
| [trace] | DeepSeek decode expert trace `logs/expert-trace-20260929-044244.analysis/counts_decode.npy` (2.6 M records, 63K decode steps) |

The reference card reports `memory.total` = 97,887 MiB **M** (nvidia-smi on the reference machine). All MiB below are 2^20 bytes.

## 1. What is in VRAM, per model

### 1.1 Checkpoint bytes by role [hdr] **M**

| | DeepSeek-V4.1-Flash | GLM-5.3-Flash NVFP4 | Qwen3.8-Flash-Next (FN) NVFP4 | Qwen3.8-27B NVFP4 |
|---|---|---|---|---|
| routed experts (main) | 268.9 GiB (MXFP4, 15,360 + 384 DSpark) | 173.0 GiB (NVFP4, 12,096) | 63.3 GiB (NVFP4, 24,576) | — (dense) |
| MTP / DSpark experts | 6.7 GiB (in the expert store, RAM) | Z.ai FP8 copy 6.75 GiB (VRAM) | 2.34 GiB FP8 (VRAM) | — |
| Engram / PLE table | 188.8 GiB (RAM or SSD) | — | 47.7 GiB (RAM or SSD) | — |
| dense per-layer weights | 6.71 GiB (FP8, 40 layers) | 14.0 GiB BF16 in the NVFP4 ckpt (45 + NextN); VRAM uses FP8 for DSA attention and shared experts | 6.84 GiB BF16 (48 layers) | 16.61 GiB (FP8 attention, NVFP4 MLP × 56, FP8 MLP × 8; 64 layers) |
| embedding | 1.23 GiB BF16 → **host** | 1.18 GiB BF16 → **host** | 1.18 GiB BF16 → **host** | 2.37 GiB BF16 (host in an FN-like family) |
| lm_head | 1.23 GiB BF16 (VRAM) | 1.18 GiB BF16 (VRAM) | 1.18 GiB BF16 (VRAM) | 1.18 GiB FP8 |
| MTP dense | 0.66 GiB | (in the NextN layer) | 0.17 GiB | 0.79 GiB BF16 |
| vision encoder | 0.90 GiB (lazy) | 1.05 GiB (lazy) | 0.84 GiB (lazy) | 0.86 GiB |

Expert record sizes (one routed expert = one cache slot): DeepSeek 17.93 MiB **M** [ds-log "record 17.9 MiB"], GLM 13.50 MiB **M**
[glm-log], FN 2.64 MiB **E** (3 × 640 × 2560 / 2 + 1/16 scales; matches docs/qwen.md).

### 1.2 Component table (GB = GiB), shipped configuration, 96 GB card

| Component | DeepSeek | GLM | FN | Scales with |
|---|---|---|---|---|
| dense weights + MTP + lm_head | 9.83 **E** [hdr] | 19.86 **M** [glm-log "non-expert VRAM 19.86 GiB"; of which MTP FP8 experts 6.75] | 10.54 **E** [hdr; MTP experts 2.34] | fixed (MTP part removable: DS `--no-mtp`, GLM `HIVE_GLM_MTP_K=0`, FN `HIVE_QWEN_MTP_K=0`) |
| CUDA context + process residual | 0.77 **M** (10,857 MiB used after model load [ds-log] − checkpoint bytes) | 1.78 **M** (residual of [glm-log] after the itemised parts below) | 1.8 **E** (GLM's) | fixed; level-3 sleep floor 0.96 GiB **M** [glm.md] |
| runtime small buffers, cuBLAS (2 × 64 MiB), indexer scores, logits, sampler | 1.06 **M** [ds-log: 84,226 → 87,030 free minus staging] | 0.43 **E** (deq 128 + attention ws 144 + logits 38 + cuBLAS 128 MiB) | 0.28 **E** | ~fixed (DS indexer is capped by `index_budget` 32 M elements) |
| expert staging ring | 1.68 **M** (96 × 17.93 MiB, `HIVE_STAGING=96`) | 0.11 **E** (8 records) | 0.02 **E** | `HIVE_STAGING` (DS, ≥ 8); fixed 8 in GLM/FN |
| prefill work buffers | 10.09 **M** [ds-log "prefill-only 10335 MiB"], lent to the cache when idle (`HIVE_CACHE_ELASTIC`) | 6.50 **E** (416 KiB/row × 16,384) | 3.83 **E** (245 KiB/row × 16,384) | linear in `--max-chunk` (below) |
| layer-major buffers (prefill of 2–4 chunks) | (inside the 10.09 above: 2 VRAM tiles + 3 host-tile device parts + 2 staging) | 3.58 **M** [glm.md "3.6 GiB less cache"], borrowed from the cache tail while running | 2.36 **E**, borrowed, only while an expert is missing | `--max-chunk` × tiles |
| MTP verify snapshots | small (ring/compressor rows) | 3.41 **M** [glm-log "8 part(s) 3494 MiB"] = 437 MiB per part | 2.63 **E** = 337 MiB per part | parts = `--max-batch` with `HIVE_MTP_BATCH=1`, else 1 |
| session pool (preallocated) | 8 × 250 MiB **M** [ds-log] | 8 × 456 MiB **M** [glm-log] | 8 × 594 MiB **E** | `max(--max-sessions, --max-batch)` × per-sequence size |
| reserve kept free | 2,400 MiB (default) | 3,000 MiB (`config/glm.env`) | 3,000 MiB | `HIVE_CACHE_RESERVE_MB` |
| expert cache | 4,447 slots = 77.9 GiB **M** (577 of them are the lent prefill buffers) | 4,323 slots = 57.0 GiB **M** | all 24,576 = 63.4 GiB **E** (fits with ~6 GB to spare) | everything left |

Cross-check of the model in §1.3 against the two measured startups: predicted slots DS 4,452 vs 4,447 **M**, GLM 4,330 vs 4,323 **M**.

### 1.3 Scaling rules

Per prefill row (`--max-chunk` M), from the allocation code:

| | per row | at M = 16,384 | at M = 4,096 | Code |
|---|---|---|---|---|
| DeepSeek `Work` | 344 KiB **E** (sum of `Work` allocations; the formula reproduces the measured 10,335 MiB total within 0.5 %) | 5.4 GiB | 1.34 GiB | `Runtime::Runtime`, runtime.cpp ~918–1010 |
| DeepSeek extra VRAM tile (`--prefill-tile` N: N − 1 slots) | 67.8 KiB/row/slot **E** | 1.06 GiB each | | runtime.cpp ~1140 |
| DeepSeek host tile, device part (`HIVE_PREFILL_HOST_TILES`) | 27.8 KiB/row/tile + 40 KiB/row per staging buffer (≤ 2) | 0.43 + 0.63 GiB | | runtime.cpp ~1156 |
| GLM `GlmEngine` work | 416 KiB **E** (352 without MTP) | 6.5 GiB | 1.63 GiB | glm_engine.cpp 130–180 |
| GLM layer-major | 57.3 KiB/row × tiles × M (borrowed) | 3.58 GiB at 4 tiles | | glm_engine.cpp 160 |
| FN `QwenEngine` work | 245 KiB **E** (170 without MTP) | 3.83 GiB | 0.96 GiB | qwen_engine.cpp 44–100 |
| FN layer-major | 37.8 KiB/row × tiles × M (borrowed, at most half the cache) | 2.36 GiB at 4 tiles | | qwen_engine.cpp 533 |
| Qwen3.8-27B (no family) | ~210 KiB **E** (H 5120, MLP 17,408) | 3.3 GiB | 0.84 GiB | — |

Per sequence (context):

| | fixed per sequence | per position | allocation |
|---|---|---|---|
| DeepSeek | 5 MiB (window rings) | 979 B **E** (from 250 MiB **M** at 262,144) | **preallocated at `--max-ctx`** for every pool sequence (`Runtime::new_seq`) |
| GLM | 146 MiB (34 KDA states + conv) **E** | 18.75 KiB (11 DSA + NextN × 1,600 B) **E**; 456 MiB at the 16K base **M** | address space for `--max-ctx`; memory mapped for `max_chunk` positions at creation, then in 4,096-position steps (`HIVE_GLM_KV_BASE/STEP`), taken from the cache tail when free VRAM < reserve |
| FN | 112 MiB (36 GDN states + conv) **E** | 30.1 KiB (12 full-attention + MTP layer, BF16 K/V 2 KiB + indexer 320 B, + 12 B rope) **E** | same growth scheme as GLM (`HIVE_QWEN_KV_BASE/STEP`) — **7.7 GiB for one 262K conversation** |
| Qwen3.8-27B | 150 MiB (48 GDN states) **E** | 68 KiB (16 attention layers × 4 KV heads × 256 × 2 × BF16, + MTP) **E** | — (17 GiB at 262K) |

## 2. Minimum VRAM: dense on the GPU, experts from cache / CPU

"Lean" = what today's knobs allow without code changes: `--max-chunk 4096`, `--max-batch 1` (pool = 2 sequences, hived's
`--max-sessions` default), `HIVE_MTP_BATCH=0` (1 snapshot part), DS `--prefill-tile 1`, `HIVE_PREFILL_HOST_TILES=0`, `HIVE_STAGING=8`,
`HIVE_CACHE_ELASTIC=0`, `--max-ctx 32768`; GLM/FN `HIVE_*_PREFILL_TILES=1`; reserve 1,700 MiB with images (DS vision ~1.7 GB, GLM ~1.1 GB,
FN ~0.9 GB + scratch), 500 MiB with `--no-vision` / no images. All **E** (built on the measured residuals above).

| VRAM needed before the first expert slot | lean | lean, no MTP | lean, no MTP, no images |
|---|---|---|---|
| DeepSeek | 14.8 GiB | 14.1 GiB | 13.0 GiB |
| GLM | 27.3 GiB | 19.7 GiB | 18.5 GiB |
| FN | 17.7 GiB | 14.5 GiB | 13.4 GiB |
| Qwen3.8-27B (all on GPU, 1 sequence × 8K) | 23.7 GiB | 22.9 GiB | 21.7 GiB |

Each `--max-chunk` halving saves DS 0.67 / GLM 0.81 / FN 0.48 GiB at 4,096 → 2,048 (§1.3).

What one more GiB buys:

| | expert slots per GiB | share of all experts per GiB | or KV positions per GiB |
|---|---|---|---|
| DeepSeek | 57 | 0.36 % of 15,744 | 1.10 M positions (= 4 more pool sequences at 262K) |
| GLM | 76 | 0.63 % of 12,096 | 55.9 K |
| FN | 388 | 1.58 % of 24,576 | 34.9 K |
| Qwen3.8-27B | — | — | 15.4 K |

Hit rate is far better than the slot share: measured DS 4,447 slots (28 %) → 94 % of decode routings hit (one request in [ds-log]: hit 14,735
/ cpu 965); GLM 4,323 slots (36 %) → 96.3 % with cache-aware routing **M** [docs/glm.md]. A **static** cache of the globally most-used DS experts
covers only (**M** [trace]): 121 slots 11.8 %, 500 slots 23.7 %, 1,000 32.5 %, 2,000 44.5 %, 4,447 63.9 %, 8,000 82.3 % — a lower bound;
the live cache follows each conversation's experts and the routing prior pulls toward residents. The hit rate of small caches (≤ 1,000 slots)
has **not been measured**; it is the main unknown of the small-card cases.

## 3. Per card

Card totals: 96 GB 97,887 MiB **M**; the others **E** (nominal × 0.9958, the ratio measured on the 96 GB card): 72 GB 73,414 · 48 GB
48,943 · 32 GB 32,607 · 24 GB 24,467 · 16 GB 16,303 · 12 GB 12,227 · 8 GB 8,151 MiB. A desktop on the same card takes more.
`scripts/hive-start.sh` refuses to start below `12000 + reserve` MiB free (line 54) — on 12 GB and 8 GB cards it must change first.

### 3.1 Result with today's code (no layer placement exists today)

Cells: expert slots left for the cache (share of all experts), or the shortfall. Shipped = the family's `config/*.env`; lean = §2.
**All E**, except the 96 GB shipped DS/GLM cells (**M**).

| Card | DeepSeek shipped | DeepSeek lean (no MTP / no images) | GLM shipped | GLM lean / no MTP / + no images | FN shipped | FN lean / no MTP / + no images | Qwen3.8-27B |
|---|---|---|---|---|---|---|---|
| 96 GB | 4,447 (28 %) **M** | 4,614 / 4,719 | 4,323 (36 %) **M** | 5,176 / 5,758 / 5,847 | all 24,576 → device routing | all | fits, ~70 GB for KV |
| 72 GB | 3,087 (20 %) | 3,249 / 3,354 | 2,517 (21 %) | 3,364 / 3,945 / 4,034 | 17,469 (71 %) | 20,953 / 22,169 / 22,623 | fits |
| 48 GB | 1,722 (11 %) | 1,884 / 1,989 | 704 (6 %) | 1,551 / 2,133 / 2,222 | 8,200 (33 %) | 11,684 / 12,899 / 13,354 | fits, ~24 GB KV |
| 32 GB | 811 (5 %) | 973 / 1,078 | **no** (−6.7 GB) | 341 / 923 / 1,012 | 2,012 (8 %) | 5,496 / 6,712 / 7,166 | lean only (~8 GB KV ≈ 125K positions) |
| 24 GB | 357 (2 %) | 519 / 624 | **no** | **no** (−3.5 GB) / 320 / 409 | **no** (−2.8 GB) | 2,412 / 3,628 / 4,083 | lean only, ≤ 15–33K context |
| 16 GB | **no** (−1.7 GB) | 64 / 169 | **no** | **no**: −11.4 / −3.8 / −2.6 GB | **no** | **no** (−1.8 GB) / 536 / 990 | **no** (−5.8 GB min) |
| 12 GB | **no** | **no**: −2.9 / −1.0 GB | **no** | **no**: −15.4 / −7.7 / −6.6 GB | **no** | **no**: −5.7 / −2.6 / −1.4 GB | **no** (−9.8 GB) |
| 8 GB | **no** | **no**: −6.8 / −5.0 GB | **no** | **no**: −19.4 / −11.7 / −10.5 GB | **no** | **no**: −9.7 / −6.6 / −5.4 GB | **no** (−13.7 GB) |
| CPU only | GPU build: no. CPU families (`HIVE_FAMILY_*_CPU`) in progress; FN CPU family exists | | | | | | |

Notes:
- "Runnable" with tens of slots means the CPU computes nearly every expert: decode is bound by RAM bandwidth (§3 of
  [requirements.md](requirements.md)), and prefill streams every expert over PCIe once per chunk (a smaller chunk means more passes).
- FN routes on the device only when every expert is resident: only the 96 GB card qualifies; every other cell runs the host routing path.
- Qwen3.8-27B has no hive family yet; its column assumes one built like FN (dense weights on the GPU, embedding in host RAM).
- The DS shipped column keeps `HIVE_CACHE_ELASTIC=1`: the prefill buffers (10.1 GB at 16K) are lent from the cache, so the card must hold at
  least that many slots; below ~24 GB use the lean set.

### 3.2 Layer-level placement for the cards that do not fit

No family can place a dense layer on the CPU today; this is new code. Cost per layer (VRAM **E**, from [hdr]; GLM per-layer average **M**
from the load log: 11.7 GiB for 45 layers):

| | per layer (VRAM) | parts of one layer | other movable blocks |
|---|---|---|---|
| DeepSeek | 162–178 MiB (layers 1, 14: +150 MiB Engram projections) | attention 123–138 (FP8 + per-row scales), shared expert 34, router 4, HC 4 | lm_head 1.23 GiB, DSpark 0.66 GiB (`--no-mtp`), vision |
| GLM | avg 266 **M**; KDA layer ≈ 291, DSA layer ≈ 150, dense-MLP layers 0–2 ≈ 346 | KDA attention 264 (BF16), DSA attention ≈ 120 (FP8), shared 24 (FP8), router 2 | lm_head 1.18 GiB, NextN + FP8 experts 6.95 GiB (`HIVE_GLM_MTP_K=0`) |
| FN | 148 (GDN layer), 135 (attention layer), 210 (layer 1 with PLE) | attention / GDN 111–123, shared 9, HC 13, router 3 | lm_head 1.18 GiB, MTP 2.5 GiB (`HIVE_QWEN_MTP_K=0`) |
| Qwen3.8-27B | 254 (GDN + NVFP4 MLP), 243 (attention + NVFP4 MLP), 355–366 (layers 56–63, FP8 MLP) | MLP 143 (NVFP4) / 255 (FP8), attention 100–111 | lm_head 1.18 GiB, MTP 0.79 GiB |

Layers to move (shortfall ÷ average layer, lean set; first drop MTP and images, which cost no compute):

| Card | DeepSeek | GLM | FN | Qwen3.8-27B |
|---|---|---|---|---|
| 16 GB | fits lean (no MTP) | 10 of 45 layers (or lm_head + 6) | fits lean (no MTP) | 23 of 64 |
| 12 GB | 6 of 40 (or lm_head) | 25 of 45 | 10 of 48 (or lm_head + 2) | 39 of 64 |
| 8 GB | 30 of 40 | 41 of 45 | 38 of 48 | 55 of 64 |

At 8 GB (and GLM at 12 GB) almost every layer moves: that is the CPU family with a GPU helper, not a GPU family with CPU layers.

Cost of a CPU-placed layer **E**: decode reads its weights once per step on the CPU at the measured 77–118 GB/s of the expert kernels
([glm.md]) — DS 1.7 ms, GLM 2.7 ms, FN 1.5 ms, Qwen3.8-27B 2.6 ms per layer per step (vs ~0.1–0.2 ms on the GPU). For prefill, streaming
the layer's weights to the GPU over PCIe (27.9 GB/s **M**) costs 6–10 ms per layer per chunk, far below CPU GEMM time for a 4K-row chunk —
the same split hive already uses for experts (decode: CPU in place; prefill: stream). The layer's KV / recurrent state must then live in
RAM for decode (DS ≈ 25 B/position/layer; GLM 4.3 MiB per KDA layer per sequence; FN 3.1 MiB per GDN layer; Qwen3.8-27B 4 KiB/position per
attention layer).

## 4. Knobs today, and one total cap

### 4.1 Existing knobs

| Knob | Families | What it controls |
|---|---|---|
| `HIVE_CACHE_MB` → `--vram-cache-mb` | all | slot bytes when `HIVE_CACHE_FIT` is off (DS: allocated in the `ExpertStore` constructor; GLM/FN: in the store constructor unless deferred) |
| `HIVE_CACHE_FIT=1` | all | slots = (free VRAM after the session pool − reserve − 64 MiB) / record (`ExpertStore::fit_slots`, hived.cpp 703); also at wake |
| `HIVE_CACHE_RESERVE_MB` | all (default 2,400) | VRAM left free beside the cache; GLM/FN KV growth and vision `make_room` keep it by shrinking the cache tail in 56 MiB steps |
| `HIVE_QWEN_VRAM_MB` | FN only | budget = most device memory the process may use, counted from `free_at_load` (after the CUDA context, so the context is **not** counted); clamps slots at `alloc_cache` and wake, and `kv_reserve` / `make_room` / `regrow_cache` through `QwenEngine::vram_avail`. It does not size the work buffers, pool or snapshots (they are only counted as "used") |
| `--max-ctx` | all | DS: preallocated per pool sequence (979 B/position); GLM/FN: address space only |
| `--max-chunk` | all | all work buffers linearly (§1.3) |
| `--max-batch`, `--max-sessions` | all | pool = max(both) sequences; with `HIVE_MTP_BATCH=1` GLM/FN snapshot parts = max-batch (437 / 337 MiB each); GLM/FN cap max-batch at 8 |
| `--prefill-tile`, `HIVE_PREFILL_HOST_TILES`, `HIVE_CACHE_ELASTIC`, `HIVE_STAGING`, `HIVE_ENGRAM_ALIAS` | DS | tile slots, host tiles, lending prefill buffers to the cache, staging ring, Engram buffers inside q/o (saves 1.19 GB **M**) |
| `HIVE_GLM_PREFILL_TILES`, `HIVE_GLM_PREFILL_ELASTIC`, `HIVE_QWEN_PREFILL_TILES` | GLM, FN | layer-major buffers (borrowed by default) |
| `--no-mtp` (DS), `HIVE_GLM_MTP_K=0`, `HIVE_QWEN_MTP_K=0` | | MTP weights and buffers not loaded (GLM/FN read the env in the model constructor; `--no-mtp` does not reach them) |
| `--no-vision`, `--vision-eager` | all | vision encoder (lazy by default, taken from the reserve) |
| sleep levels 1 / 2 / 3 | all | 1: slots + staging (+ lent buffers); 2: + session KV to RAM; 3 (`HIVE_SLEEP_VMM=1`): + weights, work, cuBLAS — 984 MiB left **M** [glm.md] |

### 4.2 `HIVE_VRAM_LIMIT_MB`: proposed implementation

Goal: one number, the card size, counting everything — context, weights, work buffers, KV, graphs, cuBLAS, allocator granularity.

> **Implemented 2026-10-09 (steps 1 and 3; since measured at 96 / 72 / 48 / 32 GB for all four families):** `engine/include/hive/vram_limit.h`, `engine/src/hived.cpp`
> (GPU daemons only; the CPU-only daemons skip it), `scripts/hive-start.sh` (requires exactly the limit free), CPU test
> `engine/tests/test_vram_plan_cpu.cpp`. Difference from the sketch below: the context is measured at the device check, but the ballast is
> allocated **after the weights have loaded** — the weights are then a measured number, so "the weights alone do not fit" is exact, not an
> estimate from the headers.

**Step 1 — ballast (hard cap, all families, one place).** In `hived.cpp`, right after the device check (before `Model`), and nowhere else:

1. Read the device's free memory with NVML **before** the CUDA context exists (`nvmlDeviceGetMemoryInfo`, `dlopen("libnvidia-ml.so.1")` —
   hive links no NVML today) → F0; create the context (`cudaFree(0)`), read `cudaMemGetInfo` → F1. Context C = F0 − F1 (exact if no other
   process allocates in between). Without NVML: C = the 984 MiB level-3 floor as an estimate, logged as such.
2. Ballast B = F1 − (L − C), where L = `HIVE_VRAM_LIMIT_MB`. `cudaMalloc(B)` once (plain, not VMM: it never moves, never sleeps).
3. Log `[hived] VRAM limit L MiB: context C MiB, ballast B MiB — the process sees an L MiB card`. Subtract B from the "used" values hived
   publishes (`vram` stats, sleep reports) so they read like the small card.

Every existing sizing path then sees the small card with no family change: `HIVE_CACHE_FIT`, the reserve, DS host-tile fitting
(runtime.cpp 1163), DS/GLM/FN wake (`fit_slots`, `wake_alloc`), GLM/FN `kv_reserve` / `make_room` / `regrow_cache`, the vision encoder,
cuBLAS internals, CUDA graph growth and VMM granularity — all are counted because they come out of the same physical pool. This is also how
to measure every row of §3 on the 96 GB card. `HIVE_QWEN_VRAM_MB` becomes redundant (keep it as an alias: L = its value + C).

> **Implemented 2026-10-18 for DeepSeek (a narrower form of step 2):** `engine/include/hive/vram_reserve.h`. The ballast is held in 32 MiB
> pieces; at ready the ballast goes back to the card and the headroom left inside the limit (what is free while the ballast is held) is
> taken as the reserve, new pieces first so nothing can take it in between. While serving, `DevBuf::alloc` and `GrowBuf::ensure` take their
> room out of the reserve and a free refills it — never past the share it was armed with — and an allocation past the reserve is refused
> without asking the device. Sleep turns the cap off (what it frees goes to the card); a buffer that came out of the reserve and is freed
> asleep (the vision encoder) still credits the reserve's share, and wake refills it. A refusal throws `DeviceOutOfMemory` (the request paths fail only that request) instead of aborting. Why:
> with `HIVE_VRAM_BALLAST_RELEASE=1` beside another process, a 2 MiB runtime buffer found 26 MiB free on the card and the abort took every
> open conversation down; with the ballast kept, the other process had no room to grow. DeepSeek's sizes are fixed after ready (session
> pool, slots; wake restores at most what sleep released), so the reserve covers it; GLM, Qwen and Q27 still read the device's free
> memory at runtime (the table below) and keep the ballast until they read `avail()` instead. GPU test: `engine/tests/test_vram_reserve.cu`.

**Step 2 — accounting cap (for a card shared by several processes; optional).** A header `hive/vram_budget.h` with
`avail() = min(cudaMemGetInfo free, L − used_self)`, where `used_self` = C + bytes counted at the allocation points — `DevBuf::alloc/free`,
`devmem::alloc/free`, `GrowBuf::ensure/trim`, `ExpertStore::alloc_cache` / `elastic_reserve` / staging, `GlmExperts` slots (`slot_buf_`) and
`staging_`, the vision encoder, `Blas` workspace — plus a residual term (graphs, cuBLAS internals) measured once after startup as
(F1 − free) − counted. Replace the free-VRAM reads that size things with `avail()`:

| Where | Family |
|---|---|
| hived.cpp 705 (cache fit at start), 2796 / 2804 (wake: pool + first slot allocation) | all |
| expert_store.cpp 802 (`wake_alloc` fit), runtime.cpp 1163 (host tiles) | DS |
| glm_engine.cpp 266 (`kv_reserve`), 305 (`make_room`); glm_experts.cpp 316 (`regrow_cache`), 359 (`wake_alloc`); glm `hived_family.h` 241 | GLM |
| `QwenEngine::vram_avail`, `ExpertStore::clamp_slots`, qwen_runtime.cpp 87 (wake) | FN (replaces `HIVE_QWEN_VRAM_MB`) |

**Step 3 — startup plan (both steps).** Before the first large allocation, compute the fixed part from the formulas of §1.3 (weights from
the safetensors headers, work = per-row × max-chunk, pool, snapshots, staging, reserve) and print it next to L. If the fixed part does
not fit, absorb rather than fail, in this order (each is lossless or output-preserving and is logged): DS host tiles → prefill tiles →
`HIVE_STAGING` → MTP batch parts → `--max-chunk` halving (≥ 1,024) → `--max-batch`. Only when the weights alone exceed L is there nothing to
absorb: the run would end at the same point in a `DevBuf` abort, so print the shortfall and the layers/blocks of §3.2 that would have to move,
and exit with code 2 like the existing sm_120 check (no new class of rejection — the allocation would fail anyway). Layer placement (§3.2)
then becomes the next item that reads this plan.

Files: `hived.cpp` is shared (main applies the diff); the families' own files for step 2.

## 5. What to measure first (when the GPU is free)

1. DS, GLM, FN with the ballast at each card size in §3 (shipped and lean sets): the `[hived] session pool / cache fit / ready` lines give
   every component of §1.2 for that card; compare with the **E** cells.
2. CUDA context alone (NVML before/after `cudaFree(0)`), to split the 0.77–1.78 GiB residuals.
3. Decode hit rate and tok/s against slot count (500 / 1,000 / 2,000 slots) — the unknown that decides whether 16–32 GB cards are useful.
