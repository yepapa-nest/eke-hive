# Configuration

This page lists every setting Eke Hive reads: the launcher keys, the `hived` command line, the engine
environment switches and the HTTP request fields. For each engine switch it gives the default in the code, the
value in the recommended configuration ([`config/hive.env`](../config/hive.env)) and, where one exists, the
measurement that decided it. The measurements themselves are described in [performance.md](performance.md).

Internal identifiers keep the short name: the daemons are `hived` (DeepSeek-V4.1-Flash) and `hived_glm` (GLM-5.3-Flash, the
same source built with the GLM family types), the CLI is `hive`, variables start with `HIVE_`.

## Model families

The launcher reads the checkpoint's `config.json`: `model_type` `glm5_next` selects GLM-5.3-Flash (`hived_glm`), anything else
DeepSeek-V4.1-Flash (`hived`); `HIVE_MODEL_FAMILY=deepseek|glm` overrides it. The configuration file follows the family:
[`config/hive.env`](../config/hive.env) for DeepSeek, [`config/glm.env`](../config/glm.env) for GLM, unless `HIVE_CONFIG` names one.

- **Launcher keys, the API server settings and the HTTP request fields** apply to both families (differences are noted).
- **Daemon-level switches** (scheduler, sessions, prefix sharing, the speculative-decoding gate, `HIVE_CACHE_FIT`,
  `HIVE_CACHE_RESERVE_MB`, `HIVE_CACHE_PRIOR`, sleep) are read by the shared `hived.cpp` and apply to both:
  `HIVE_CACHE_FIT`, `HIVE_CACHE_RESERVE_MB`, `HIVE_CACHE_PRIOR`(`_TOPJ`), `HIVE_PREFIX_SHARE`, `HIVE_PREFIX_ADAPTIVE`,
  `HIVE_PREFIX_EXTRA_CHUNKS`, `HIVE_MTP_GATE2`, `HIVE_MTP_GATE3`, `HIVE_MTP_BATCH`, `HIVE_WARM_DEFER`, `HIVE_DEFER_CKPT`,
  `HIVE_CKPT_HISTORY`, `HIVE_IMAGE_CKPT`, `HIVE_BATCH_*`, `HIVE_START_ASLEEP`, `HIVE_SLEEP_VMM`, `HIVE_SLEEP_PREPIN`,
  `HIVE_GRACEFUL_STOP_S`, `HIVE_TRACE_MTP`.
- **Engine switches of the DeepSeek runtime** (`HIVE_DECODE_*`, `HIVE_PREFILL_*`, `HIVE_CPU_*`, `HIVE_ENGRAM_*`, `HIVE_STAGING`,
  `HIVE_PREFETCH*`, the kernel switches, and the `hived` flags `--gpu-share`, `--promote`, `--promote-misses`, `--prefill-tile`,
  `--decoder-tail`, `--decoder-replay`, `--engram`) apply to DeepSeek only.
- **GLM's own switches** (`HIVE_GLM_*`) are listed in [glm.md](glm.md#switches-of-this-family).

## How configuration works

All scripts in `scripts/` load their settings through `scripts/lib.sh`:

1. Variables already set in the **environment** win.
2. Then the file named by **`HIVE_LOCAL_ENV`** (your own settings).
3. Then the **configuration file**: `HIVE_CONFIG` if set (a relative path that does not exist from the current directory is
   taken from the repository root), otherwise `config/glm.env` for a GLM checkpoint and `config/hive.env` for DeepSeek — the
   configurations the published numbers were measured with. Do not edit them; override them. A missing file is an error.

Both files are `KEY=value` lines, one per line, taken literally (no quotes, no shell expansion, `#` starts a
comment line). `scripts/hive-start.sh` then passes **every** `HIVE_*` variable into the container by value, so
the daemon and the API server see exactly what you set. Command-line flags of `hive-start.sh` override the
corresponding keys, and any flag it does not know is passed on to `hived`. See [install.md](install.md) for the
step-by-step setup.

Most engine switches are read once when the daemon starts; change them and restart.

### Value conventions

- **On/off switches** (the large majority): unset, empty or `0` means off; any other value means on. Off is
  always the code path that existed before the switch was added.
- **Numeric switches** are parsed leniently: a value that is out of range or not a number falls back to the
  documented default instead of failing the start.
- A few older switches are parsed as integers (`atoi`): they are on only for a non-zero number, so `true` turns
  them **off**. These are marked *integer* below (`HIVE_IDX_TC`, `HIVE_IDX_F32`, `HIVE_HC_FUSE2`,
  `HIVE_MX_GEMM`, `HIVE_MX_PREFILL`, `HIVE_HC_MIX_ROWS4`, `HIVE_ATTN_FLASH32`, `HIVE_FUSE`, `HIVE_MX`; on GLM also
  `HIVE_GLM_DEFER`, `HIVE_GLM_PREFETCH`, `HIVE_GLM_PROF`, `HIVE_GLM_CPU_BALANCE`, `HIVE_GLM_HC_CUBLAS`, `HIVE_GLM_DEBUG`).
- Engine switches read by hived that are **on by default**: `HIVE_FUSE`, `HIVE_MX`, `HIVE_HEAD_ROWS` (set to `0` to turn off) and
  `HIVE_SLEEP_PREPIN` (only the exact value `0` turns it off). The server's `HIVE_TOOL_STREAM` and the layer-yield
  `HIVE_BATCH_COALESCE_MS` default are on as well (their own tables below).

Verdict column wording: *adopted* = measured and switched on in `config/hive.env`; *adopted, no separate A/B* =
switched on as part of a group (most of them together in step 1, "opt-in features + tuning") without a
standalone measurement on record; *rejected* = measured and
left off; *diagnostic* / *comparison* = exists for measurement or for testing a kernel against its predecessor,
not for production; *experimental* = implemented but never measured as a standalone change. "Step N" refers to
the step table in [performance.md](performance.md#steps-in-order). "c1/c4/c8" are decode throughput at 1, 4
and 8 concurrent streams.

## Launcher keys

Read by the scripts in `scripts/` (and forwarded into the container). The second column is the value in `config/hive.env` /
`config/glm.env`; keys without a value there have no default — `hive-start.sh` stops if they are unset.

| Key | Shipped (DeepSeek / GLM) | Meaning |
| --- | --- | --- |
| `HIVE_CKPT` | (required) | Checkpoint directory on the host (`config.json`, `*.safetensors`, `tokenizer.json`, `encoding/`, `inference/`) |
| `HIVE_CKPT_MOUNT` | `/model` | Where the checkpoint is mounted inside the container (read-only) |
| `HIVE_STATE_DIR` | `<repo>/run` | Host directory mounted at `/out`: logs, daemon socket, Engram constants, warm-start file |
| `HIVE_LOCAL_ENV` | — | Your settings file (see above) |
| `HIVE_CONFIG` | by family | Base configuration file (see [Model families](#model-families)) |
| `HIVE_MODEL_FAMILY` | from `config.json` | `deepseek` or `glm` — overrides the family detected from the checkpoint |
| `HIVE_IMAGE` | `eke-hive:latest` | Docker image |
| `HIVE_CONTAINER` | `hived` | Container name |
| `HIVE_GPU` | `0` | GPU index |
| `HIVE_PORT` | `8430` | HTTP port of the API server (host network) |
| `HIVE_CACHE_MB` | `68000` / `60000` | VRAM expert cache in MiB → `hived --vram-cache-mb`; ignored while `HIVE_CACHE_FIT=1` (both shipped configurations) |
| `HIVE_MAX_CTX` | `262144` / `262144` | Maximum context length → `--max-ctx` |
| `HIVE_MAX_CHUNK` | `16384` / `16384` | Prefill chunk (rows per sub-chunk) → `--max-chunk` |
| `HIVE_CPU_THREADS` | `16` / `32` | CPU expert worker threads, split over the NUMA nodes → `--cpu-threads` |
| `HIVE_MAX_BATCH` | `8` / `8` | Sequences in one decode batch → `--max-batch` |
| `HIVE_HOST_SESSION_MB` | — / `98304` | RAM budget for conversations evicted from the VRAM session pool → `--host-session-mb` (hived default 32768; oldest dropped first). A 145K-token GLM conversation keeps ~3 GB: twelve of them, then the first one again, 37.0 s → 1.6 s to the first token ([performance.md](performance.md#steps-in-order-1) GLM step 27) |
| `HIVE_DAEMON_ARGS` | `--gpu-share 0 --promote 8 --prefill-tile 3 --decoder-replay` / — | Extra `hived` arguments (empty when unset) |
| `HIVE_RAM_WAIT_GB` | `480` / `390` | `hive-start.sh` waits (up to 10 min) until the previous `hived` has exited and this much RAM is available; values above the installed RAM are clamped to MemTotal − 8 GiB. Use ~360 in SSD mode |
| `HIVE_BUILD` | `build` | Build directory under `engine/`. `hive-start.sh` ignores a value from the shell on purpose and uses only `--build DIR`; `build.sh` and the test scripts use it |
| `HIVE_START_ASLEEP` | off | Start in the sleeping state (see [Sleep and wake](#sleep-and-wake)); `hive-start.sh` then only requires 14,000 MiB of free VRAM instead of cache + 12,000 MiB (with `HIVE_CACHE_FIT` on: 12,000 MiB + `HIVE_CACHE_RESERVE_MB`) |
| `HIVE_SLEEP_LEVEL` | `1` | `hive-sleep.sh sleep`: 1 = free the expert cache · 2 = also move session KV to RAM · 3 = also move dense weights, work buffers and cuBLAS state to host (needs `HIVE_SLEEP_VMM=1` or `HIVE_START_ASLEEP=1`, otherwise the daemon uses 2) |
| `HIVE_SLEEP_TIMEOUT_S` | wait forever | `hive-sleep.sh`: give up sleeping if running requests take longer |
| `HIVE_SLEEP_LOG` | — | `hive-sleep.sh` log file |
| `HIVE_HOOK_BEFORE_SLEEP`, `HIVE_HOOK_SLEEP_FAILED`, `HIVE_HOOK_AFTER_SLEEP`, `HIVE_HOOK_AFTER_WAKE` | — | Shell commands run by `hive-sleep.sh` around sleep/wake (e.g. to stop and resume routing traffic in a front proxy) |

`scripts/hive-start.sh` flags: `--build DIR`, `--vram-cache-mb N`, `--max-ctx N`, `--max-chunk N`, `--port P`,
`--cpu-threads N`, `--max-batch N`; anything else is appended to the `hived` command line.

### Adapting to your machine

The values in `config/hive.env` and `config/glm.env` are the reference machine's (RTX PRO 6000 96 GB, 32-core Threadripper, 1 TB RAM).
The ones that depend on the hardware, and how to set them elsewhere:

| Key | Reference | How to choose |
| --- | --- | --- |
| `HIVE_CPU_THREADS` | 16 / 32 | DeepSeek: physical cores of the NUMA node that holds the GPU; the CPU expert pool is DRAM-bandwidth bound, so more threads than that do not help (8/12/24 measured: no gain). GLM: all physical cores of both nodes (its CPU phase is larger; 24 → 32 measured +3 % c1, 79 → 86 GB/s) |
| `HIVE_CACHE_MB` / `HIVE_CACHE_FIT` | 68000 / on | With `HIVE_CACHE_FIT=1` the cache takes the VRAM left after the dense weights, sessions and `HIVE_CACHE_RESERVE_MB`; `HIVE_CACHE_MB` only matters when FIT is off (VRAM − ~12 GB − reserve). A smaller card means a smaller cache and a higher CPU share, not a failure |
| `HIVE_RAM_WAIT_GB` | 480 / 390 | Below the installed RAM (the start script clamps to MemTotal − 8 GiB); ~360 in SSD mode |
| `HIVE_ENGRAM_SSD` | off | Turn on with less than ~512 GB of RAM (295 GB resident, needs a fast NVMe: ~48 random row reads per token) |
| `HIVE_PREFILL_HOST_MB`, `HIVE_CKPT_PINNED_POOL_MB` | 1920, 4096 | Pinned host buffers for prefill tiles and prompt checkpoints; lower them when RAM is tight (checkpoints then fall back to pageable copies) |
| `HIVE_MAX_CTX`, `HIVE_MAX_BATCH` | 262144, 8 / 262144, 4 | DeepSeek reserves session KV per slot in VRAM (`session pool` line in the log); fewer or shorter sessions free VRAM for the cache. GLM maps KV as conversations grow ([glm.md](glm.md#context-and-kv-memory)) |

## `hived` command line

Flags marked DeepSeek in [Model families](#model-families) do nothing in `hived_glm` (GLM uses `HIVE_GLM_PROMOTE` and
`HIVE_GLM_PREFILL_TILES` instead). The launcher always passes `--ckpt`, `--engram`, `--sock`, `--vram-cache-mb`, `--cpu-threads`, `--max-ctx`,
`--max-chunk` and `--max-batch`, plus `HIVE_DAEMON_ARGS`.

| Flag | Default in `hived` | Launcher value | Meaning |
| --- | --- | --- | --- |
| `--ckpt DIR` | `$HIVE_CKPT` (error if neither is given) | `/model` | Checkpoint directory |
| `--engram DIR` | — (required when Engram layers are loaded) | `/out/engram` | Engram hash constants written by `scripts/prepare.sh` |
| `--sock PATH` | `/tmp/hive.sock` | `/out/hive.sock` | Unix socket the API server talks to |
| `--vram-cache-mb N` | 70000 | 68000 | VRAM expert cache size; ignored when `HIVE_CACHE_FIT` is on |
| `--max-ctx N` | 262144 | 262144 | Maximum sequence length |
| `--max-chunk N` | 4096 | 16384 | Rows per prefill sub-chunk (and size of the work buffers) |
| `--prefill-tile N` | 1 | 3 | Sub-chunks processed layer-first in one pass, so each layer's experts stream once per tile |
| `--prefill-threshold N` | 1024 | — | Chunks with at least this many rows use the streaming prefill path; shorter ones use the decode-style path. Measured at 512: −11 % follow-up TTFT alone, no gain together with `HIVE_PREFIX_EXTRA_CHUNKS=0` — not adopted |
| `--max-batch N` | 8 | 8 | Sequences in one decode batch |
| `--max-sessions N` | 2 (raised to at least `--max-batch`) | — | Sessions whose KV is pre-allocated in VRAM at start |
| `--host-session-mb MB` | 32768 | — | RAM budget for sessions evicted from the VRAM pool (oldest dropped first) |
| `--ckpt-min-tokens N` | 256 | — | Prompts shorter than this get no prompt checkpoint |
| `--cpu-threads N` | 16 | 16 | CPU expert workers. 8/12/24 measured: no gain |
| `--promote N` | 4 | 8 | Experts promoted into the VRAM cache per token (side stream). 4/16 measured: no gain |
| `--promote-misses N\|auto\|auto:N` | off | — | Instead of score-based promotion, promote up to N of this step's misses (`auto` = up to 32, at most a quarter of the unique misses). Measured: with it on, c4 −14 % (performance.md) — kept off |
| `--gpu-share N` | 1 | 0 | Decode misses per layer copied to the GPU through staging instead of computed on the CPU (the adaptive split below still applies) |
| `--decoder-tail N` | 2688 (= 128 × 21) | — | Prefill computes decoder layers (20+) only for the last N rows; 0 = off |
| `--decoder-replay` | off | on | Decoder tail = attention window (128) and the tail rows do not see window keys before the replayed span — the "bounded replay" deployment recipe from the model's technical report. An approximation by design: not bit-identical to the reference computation |
| `--no-mtp` | MTP on | — | Do not load the MTP (DSpark) draft head; no speculative decoding |
| `--mtp-max K` | 0 = whole draft block | — | Upper bound on drafts verified per step. 2/3 measured: no gain |
| `--mtp-conf F` | −1 | — | Confidence threshold (logit) a draft must reach to be submitted. 0.3 measured: no gain |
| `--cache-state PATH` | `/out/cache-state.bin` | — | Warm-start file: resident expert keys are saved every 60 s and loaded first at start; `""` = off |
| `--no-vision` | vision on | — | Do not load the vision encoder |
| `--vision-eager` | lazy | — | Load the vision encoder at start instead of at the first image (lazy mode also unloads it after 120 s idle) |
| `--vision-max-patches N` | 0 = model maximum | — | Cap on image patches (tests) |
| `--busy-chunk N` | 0 = no split | — | Prefill chunk size while other sequences are decoding |
| `--decode-share F` | 0.34 | — | Wall-clock share given to running decoders between prefill chunks |
| `--start-asleep` | off | — | Same as `HIVE_START_ASLEEP=1` |
| `--max-layer N` | −1 = all | — | Load only layers 0..N (partial checkpoints, tests) |
| `--fake-head` | off | — | Deterministic fake logits; for protocol tests on partial checkpoints |

## Engine switches

Default = behaviour when the variable is unset. "hive.env" = value in `config/hive.env` (— = not set there). The tables below
are the DeepSeek runtime's switches plus the shared daemon-level ones listed in [Model families](#model-families) (those are
read by `hived_glm` too; `config/glm.env` sets `HIVE_CACHE_FIT`, `HIVE_CACHE_RESERVE_MB`, `HIVE_CACHE_PRIOR`, `HIVE_PREFIX_SHARE`,
`HIVE_PREFIX_ADAPTIVE`, `HIVE_PREFIX_EXTRA_CHUNKS`, `HIVE_PREFIX_FIRST_TURN`, `HIVE_MTP_GATE2`, `HIVE_MTP_GATE3` and `HIVE_SLEEP_VMM`). GLM's own switches:
[glm.md](glm.md#switches-of-this-family).

### Loading and start-up

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_LOAD_PAR` | off | `1` | Read all expert layers and both Engram tables in one parallel O_DIRECT pass (aligned chunks → bounce buffer → final place), overlap pinning, pre-read dense tensors. `1`/non-number = 8 threads, N = N (≤ 64). On failure falls back to the per-layer loader | adopted (step 13, with `HIVE_LOAD_PREFAULT`): disk 1.0 → 7.3 GB/s, cold start 484 → 88–90 s, host copies checksum-identical |
| `HIVE_LOAD_PREFAULT` | off | `1` | Allocate the expert arena and Engram memory before the dense load and fault the pages in on N threads (`1` = 16, ≤ 64) while dense weights load | adopted (step 13) |
| `HIVE_LOAD_CHUNK_MB` | 32 | — | Read size for `HIVE_LOAD_PAR` (1–1024 MiB) | — |
| `HIVE_LOAD_BUFFERED` | off | — | Buffered `pread` instead of O_DIRECT (comparison / fallback) | comparison |
| `HIVE_ENGINE_CPU` | GPU's NUMA node | — | Pin the engine thread to this CPU number (default: the CPUs of the GPU's NUMA node) | — |
| `HIVE_GRACEFUL_STOP_S` | 0 = exit at once | — | On SIGTERM/SIGINT finish running requests for up to N s, then cancel them and exit 0. A chunk already running is not interrupted | experimental |

### VRAM expert cache and promotion

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_CACHE_FIT` | off | `1` | Size the expert cache after the session pool is allocated: free VRAM minus `HIVE_CACHE_RESERVE_MB`. `--vram-cache-mb` is then ignored | adopted (step 10): +81 slots |
| `HIVE_CACHE_RESERVE_MB` | 2400 | — | VRAM left free by `HIVE_CACHE_FIT` (lazy vision encoder, image patches, graph growth) | 1800 rejected: image requests ran out of VRAM |
| `HIVE_CACHE_ELASTIC` | off | `1` | Lend the prefill-only work buffers to the expert cache as extra slots while no prefill runs; a chunk larger than S rows takes them back by a table update (no copy). Value S = small-buffer rows (`1` = decode minimum) | adopted (step 10): +638 slots, reclaimed in 3.9 ms; joint A/B c4 +12 %, c8 +13 % |
| `HIVE_CACHE_POLICY` | score policy | `seq` | `seq` = request-aware residency by expected use count. Tuning form `seq:alpha=8:pre=8:min=0.025:gd=0.999:od=0.99:idle=8`; unknown names fall back to the default policy | adopted (step 3): real-trace replay −5 to −8 % misses on natural language |
| `HIVE_PROMOTE_SCORE` | off | `1` | Score-based victim choice for promoted misses | adopted, no separate A/B |
| `HIVE_PHASE_SCORE` | off | `1` | Prefill observations weigh 1/M in the expert score; decode/verify keep full weight | adopted, no separate A/B |
| `HIVE_MTP_CACHE` | off | `1` | MTP draft misses join the promotion candidates | adopted, no separate A/B |
| `HIVE_PROMOTE_BYTES` | 0 = no cap | — | Byte cap on permanent promotions per step | experimental |
| `HIVE_CACHE_REUSE_STAGE` | off | `1` | Reuse a record already in a staging slot when promoting it (device-to-device copy), with an event so the slot is not overwritten early | adopted, no separate A/B |
| `HIVE_STAGING` | 8 | `96` | Staging slots (18.8 MB each) for DMA'd experts, clamped to 8–256 | adopted (step 1): 96 slots, 42K prefill −6 % |
| `HIVE_DECODE_COPY_PRIO` | off | `1` | Issue promotion copies after each layer's demand DMA (paced, per-layer byte budget) and give demand copies the higher stream priority | adopted (step 3): c8 +5 % |
| `HIVE_PREFILL_PAUSE_PROMOTE` | off | `1` | No promotions from decode steps between the chunks of a multi-chunk prefill (score decay continues) | adopted, no separate A/B |
| `HIVE_ENGRAM_ALIAS` | off | `1` | Place the Engram work buffers inside the attention q/o buffers (1,190 MiB reclaimed, bit-identical) | adopted, no separate A/B |

### Prefill

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_GROUPED_PREFILL` | off | `1` | Prefill MoE as grouped GEMMs (w1‖w3 → SwiGLU + quantisation → w2 → fixed-order accumulation); same kernel bodies, bit-identical per layer | adopted (step 2): MoE layers ×2.8–3.0 |
| `HIVE_TILE_GROUP_GEMM` | off | `1` | Merge the rows of several sub-chunks of a tile into one GEMM per expert | adopted, no separate A/B |
| `HIVE_PREFILL_HOST_TILES` | off | `1` | Sub-chunks beyond the VRAM-resident tiles keep their hidden state in pinned host memory, moved up and down per layer, so a long prompt is one forward pass (experts sent once per layer) | adopted (step 2) |
| `HIVE_PREFILL_HOST_MB` | 2 tiles | `1920` | Pinned budget for host tiles (1920 MiB = 3 tiles at chunk 16384 → 98,304 rows per pass) | adopted (step 6): 100K 19.65 → 17.77 s; 4 tiles left too little VRAM for vision |
| `HIVE_BATCH_PREFILL` | off | `1` | Prefill several waiting requests in one forward, layer-first, sharing the expert transfers; reuse, cancellation and errors stay per request | adopted (step 2): 3 × 16K concurrent 17.7 → 14.9 s |
| `HIVE_DMA_FRAC_PREFILL` | adaptive | `0.7` | Fixed share of streaming-prefill misses copied to the GPU (0 < F ≤ 1); decode is not affected | adopted (step 2): 42K −24 %. 0.9 within noise |
| `HIVE_DMA_FRAC` | adaptive | — | Fixed DMA share for **all** phases including decode | rejected: 0.9 gave decode −28 to −41 % |
| `HIVE_PREFILL_SPLIT` | fixed 0.7 | — | `balance` = per-layer-type cost model for the prefill CPU/DMA split | rejected: prefill 13–37 % slower |
| `HIVE_PREFETCH` | 0 | `96` | Records copied ahead of use during streaming prefill (bounded by the staging slots) | adopted (step 1) |
| `HIVE_PREFETCH_BYTES` | depth limit only | — | Byte cap on prefetch; `0` = no prefetch | experimental |
| `HIVE_PREFETCH_MIN_ROWS` | 32 | — | Only experts with at least this many rows are prefetched | — |
| `HIVE_PREFETCH_SCALE` | off | `1` | Scale the predicted rows by this chunk's size relative to the previous chunk (a short turn after a long one no longer prefetches tiny experts) | in the step-11 configuration; no effect in its A/B (the benchmark does not produce the case) |
| `HIVE_EARLY_STREAM` | off | `1` | Issue the first staging ring of H2D copies before the CPU share is prepared; same slots and order (bit-identical) | adopted, no separate A/B |
| `HIVE_ENGRAM_PAR` | off | `1` | Parallel Engram lookups (layers 1 and 14); sequential below 64 rows | adopted, no separate A/B |
| `HIVE_IDX_MSUB_ACTUAL` | off | `1` | Indexer sub-chunk rows sized from the actual sequence length (within the same scratch) | adopted, no separate A/B |
| `HIVE_TAIL_SPLIT` | off | `1` | Move the last chunk boundary so the final chunk is large enough for decoder-tail mode (not for image requests) | adopted, no separate A/B |
| `HIVE_PREFILL_TAIL_SHORT` | off | `1` | Decoder-tail layers (rows < threshold) use the short-prefill expert path (DMA cap + CPU) | adopted (step 10): 4K 4.20 → 3.65 s, 15K 5.54 → 5.06 s |
| `HIVE_PREFILL_MULTI_TAIL_SHORT` | off | `1` | Same for the upper layers of multi-unit forwards (host tiles, batched prefill) | adopted (step 10): 80K 14.74 → 14.14 s |
| `HIVE_PREFILL_SHORT_ADAPT` | off | `1` | Streaming layers with ≤ N rows use the adaptive DMA share instead of the fixed prefill share (`1` = 4096) | adopted (step 10): 15K 5.54 → 4.87 s |
| `HIVE_PREFILL_SMALL` | off | `256` | Chunks from N rows up to the prefill threshold (prompts under 1K) use the streaming expert path (ring DMA, prefetch, CPU); floor max(N, max_batch + 1, 9) | adopted (steps 10–11): 1K prompt 6.6 → 3.1 s |
| `HIVE_SHORT_DMA_CAP` | 8 | `32` | DMA records per layer for short prefill (64 rows up to the threshold); decode stays at 8 | adopted (step 12); 64/96 within noise |
| `HIVE_PREFILL_BUDGET_MS` | 0 = off | — | Choose the next chunk length from the observed time of previous chunks (EMA, ≥ 1024 rows, safe boundaries only; not a latency guarantee) | experimental |
| `HIVE_MX_GEMM` | off (*integer*) | `1` | Tiled block-scaled tensor-core GEMM for prefill (fp8 and fp4 weights, K % 64 = 0) | adopted, no separate A/B |
| `HIVE_MX_PREFILL` | off (*integer*) | — | Earlier row-wise block-scaled tensor-core GEMM for fp4 prefill weights (superseded by `HIVE_MX_GEMM`) | comparison |
| `HIVE_IDX_TC` | off (*integer*) | `1` | Indexer scores on e2m1 × e2m1 tensor cores (prefill and decode) | adopted, no separate A/B |
| `HIVE_IDX_F32` | off (*integer*) | — | Indexer scores in fp32 without per-head bf16 rounding; exclusive with `HIVE_IDX_TC` | comparison |

### Decode kernels

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_FUSE` | **on** (*integer*) | — | Fused decode chain kernels (projections, norms, RoPE, router, shared experts, HC tail); bit-identical to the unfused chain. `0` = unfused | default |
| `HIVE_MX` | **on** (*integer*) | — | Block-scaled tensor-core grouped expert kernels; `0` = CUDA-core kernels | default |
| `HIVE_NO_GRAPH` | graphs on | — | Run eagerly instead of replaying per-layer CUDA graphs | diagnostic |
| `HIVE_DECODE_FUSED` | off | `1` | One launch per layer for resident experts: w1‖w3 → SwiGLU → quantise → w2 → accumulate | adopted (step 3): expert launches ×1.25–1.58 |
| `HIVE_DECODE_FUSED2` | off | `1` | Second expert kernel with a multi-stage `cp.async` pipeline | adopted (step 4): ×1.2–1.3 over step-3 kernels |
| `HIVE_DECODE_FUSED2_STAGES` | 4 | `4` | Pipeline depth 3, 4 or 6 | — |
| `HIVE_DECODE_FUSED3` | off | `1` | Third expert kernel (warp-level work, 128-byte contiguous stages per row); takes precedence over F2 | adopted (step 10): kernel ×1.1–1.2, bit-identical |
| `HIVE_DECODE_FUSED3_STAGES` | 4 | — | 3, 4 or 5 stages | 3 rejected |
| `HIVE_HEAD_ROWS` | on | — | Head kernel reads each vocabulary row once for all rows of the step (≤ 8; verify rows, MTP draft block); `0` = one pass per row. Bit-identical | adopted (step 22): 4 rows 1,169 → 911 µs, 8 rows 2,170 → 950 µs |
| `HIVE_DECODE_ATTN_FUSED` | off | `1` | Fused decode attention front (q, kv, compressor, indexer, top-k, sparse attention) with fewer launches; same top-k set | adopted (step 3): ×1.12–1.94 |
| `HIVE_DECODE_ATTN2` | off | `1` | Router, shared experts, q_a/q_b and wo_b GEMVs in one wave with a `cp.async` ring and warp-parallel top-k (bit-identical) | adopted (step 7): layer chain ×1.14–1.50; c1 +4 %, c4 +7 %, c8 +12 % |
| `HIVE_DECODE_ROUTER3` | off | `1` | Router v3 (`cp.async` input, parallel tail) | adopted (step 8) |
| `HIVE_DECODE_SPARSE3` | off | `1` | Sparse attention where G heads share key tiles (split order kept) | adopted (step 8) |
| `HIVE_DECODE_SPARSE3_G` | auto | — | G = 1, 2, 4 or 8; other values = automatic | 2 rejected |
| `HIVE_DECODE_QKV3` | off | `1` | q_a/kv weights issued early, per-row warp tail | adopted (step 8) |
| `HIVE_DECODE_HCMIX3` | off | `1` | HC-mix loads issued early | adopted (step 8) — the four v3 kernels together: layer chain ×1.05–1.14, bit-identical |
| `HIVE_HC_FUSE2` | off (*integer*) | `1` | `hc_mix` + `hc_pre_norm` in one launch (bit-identical) | adopted, no separate A/B |
| `HIVE_HC_DECODE_FUSED` | off | — | HC mix kernel also runs the Sinkhorn iterations in its last block (no side stream) | rejected: kernel ×5.5 but hidden behind another stream; c4 −5 % |
| `HIVE_HC_SINKHORN_PAR` | off | — | 16-lane Sinkhorn on the side stream only | rejected (no effect) |
| `HIVE_DECODE_GEMV2` | off | — | Decode dense GEMV / router / HC-mix variants with register preloads (bit-identical) | rejected: 0.5–0.97× |
| `HIVE_DECODE_PDL` | off | — | Programmatic dependent launch for those kernels | rejected (with `HIVE_DECODE_GEMV2`) |
| `HIVE_DECODE_TOPK2` | off | — | Decode indexer top-k in per-segment CTAs + merge | rejected: no speed effect |
| `HIVE_DECODE_IDXSCORE2` | off | — | Indexer score kernel v2 (no bank conflicts, skips invisible blocks; bit-identical) | rejected: kernel ×1.3–1.6 but ~0.1 ms per step |
| `HIVE_DECODE_IDXSCORE2_KPT` | 2 | — | Keys per thread (1, 2, 4) for that kernel | — |
| `HIVE_ATTN_SPLITS` | 4 | — | Split-K factor of decode sparse attention when window + top-k > 256 | — |
| `HIVE_DECODE_HOST_FAST` | off | `1` | Reordered host work per decode layer (launch first, only experts in use; same device work) | adopted (step 5): c4 +6 % |
| `HIVE_DECODE_EARLY_ROUTE` | off | `1` | Publish routing right after the router, so host classification and GPU expert launches overlap the shared experts | adopted (step 12, fixed version): 2–8 concurrent ×1.07–1.08, bit-identical. The first version lost c4 −35 % / c8 −38 % |
| `HIVE_DECODE_UBATCH` | off | — | Two half-batch layer pipelines in batched decode | rejected: c8 +2 % (noise) |
| `HIVE_DECODE_STEP_GRAPH` | off | — | Whole decode step (40 layers + head) as one graph; a dispatcher thread serves CPU misses | rejected: engine thread CPU ~0 and −11 to −17 W, but c1 −9 % |
| `HIVE_DECODE_STEP_POLL_US` | 0 = spin | — | Step-graph dispatcher sleep between polls | (with step graph) |
| `HIVE_DECODE_STEP_SPIN_US` | 0 | — | Spin before sleeping when polling is on | (with step graph) |
| `HIVE_DECODE_STEP_TIMEOUT_MS` | 10000 | — | Upper bound for the GPU waiting on CPU results (safety value) | (with step graph) |
| `HIVE_DECODE_BLOCKING_SYNC` | off (spin) | — | Blocking wait instead of spinning on the per-layer routing sync | rejected: −7 W but c4 −6 % |
| `HIVE_DECODE_SYNC_SPIN_US` | 0 | — | Polling before the blocking wait | (with blocking sync) |

### CPU experts and the CPU/DMA split

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_CPU_GEMV2` | off | `1` | CPU fp4 GEMV v2: e2m1 straight to fp32, vectorised scales, prefetch (bit-identical) | adopted (step 7): c1 +3 %, 6–9 W lower |
| `HIVE_CPU_UNPACK2` | off | `1` | Vectorised unpacking of the CPU input activations | adopted (step 9): c8 +6.4 % |
| `HIVE_CPU_MULTIROW2` | off | `1` | K-tiled multi-row kernel for jobs with ≥ N rows (2–8; `1` = 5) | adopted (step 9): 1K prefill −7.8 %, c4 +2.8 % |
| `HIVE_CPU_FINE` | off | — | Finer work items for multi-row jobs | rejected (no effect on the step-9 base) |
| `HIVE_CPU_SPLIT13` | off | — | Split CPU phase 1 into separate w1 and w3 items | rejected: within noise, c1 −3.2 % |
| `HIVE_CPU_P2_PREFETCH` | off | — | Prefetch the w2 rows into L2 while waiting for phase 2 | rejected: CPU benchmark 8–15 % slower |
| `HIVE_CPU_P1`, `HIVE_CPU_P2` | 32, 32 | — | Rows per CPU work item in phase 1 (must be a multiple of 32) and phase 2 | — |
| `HIVE_CPU_SPIN_US` | 0 | — | Worker spin before sleeping | not measured |
| `HIVE_DECODE_CPU_FIRST` | off | `1` | In batched decode, issue demand DMA after the CPU jobs have started | adopted (step 8, with the v3 kernels): c1 +2 %, c8 +6 % |
| `HIVE_DECODE_STAGE_HIT` | off | — | Reuse records still in the staging ring | experimental |
| `HIVE_DECODE_PREFETCH` | off | — | Copy up to N (≤ 4) of the previous step's CPU misses per layer while the link is idle (implies stage hit) | rejected: 2 within noise |
| `HIVE_STAGE_COPY2D` | off | — | 8 instead of 12 H2D calls per record (2D copies, same bytes) | rejected: within noise |
| `HIVE_DECODE_SPLIT` | rule-based | — | `balance` = per-layer cost table for the decode DMA/CPU split | rejected: c4 −3 %, c8 −5 % |
| `HIVE_DECODE_PREGATE` | off | — | Predict each decode layer's experts before attention and prefetch the non-resident ones into the CPU workers' caches during attention (`1` = top 8, ≤ 16) | rejected: c4 −24 %, c8 −29 % (precision 0.20, recall 0.69 in real decode); a score threshold reduced but did not remove the loss |
| `HIVE_DECODE_PREGATE_SPIN_US` | 1000 | — | Worker spin after prefetching (1–100000 µs) | (with pregate) |
| `HIVE_DECODE_PREDICT_EVAL` | off | — | Measurement only: run layer l+1's router on layer l's FFN input and log how often it names the real experts (top k / 2k, among non-resident ones, precision of the first 1/2/4 non-resident candidates) | measurement: recall top-6 72.5 %, first candidate 63.1 % (using it for pre-copies was rejected — performance.md) |
| `HIVE_CACHE_PRIOR` | 0 | `0.1` | Cache-aware routing (decode only): each router's selection scores get λ × the layer's running score range added for experts already in VRAM; the top 2 by the original scores are always kept (`HIVE_CACHE_PRIOR_TOPJ`), and the routing weights stay the model's own. Idea from Skliar et al., TMLR 2025 (arXiv 2412.00099); no code taken | adopted with the deferral below (real-chat benchmark, `tools/bench_chat.py`): c1 53.8 → 76.9 tok/s, c4 81.1 → 124.0, c8 100.2 → 173.0; quality 172 → 175 / 179, long-context items unchanged |
| `HIVE_CACHE_PRIOR_TOPJ` | 2 | — | With `HIVE_CACHE_PRIOR`: how many experts by the original scores are always kept (0–4) | default kept |
| `HIVE_DECODE_DEFER` | off | `1` | Expert deferral (decode, early-route path): CPU misses ranked 3rd or lower in their row run while the next layer starts and are added to the hidden streams one layer later (never on the last layer or for verify rows). Idea from KTransformers (SOSP 2025); no code taken | adopted (same measurement) |
| `HIVE_DECODE_SKIP_MISS` | 0 | — | Skip CPU misses ranked 3rd or lower whose routing weight is below f × the row's weight sum | rejected at f 0.1: c1 +8 %, c2–c32 within ±1 %, decode after 17–54K prompts 4–12 % lower, quality 176 / 179 (unchanged, p = 1) |

### Speculative decoding (DSpark / MTP)

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_MTP_GATE2` | off | `1` | Draft only when a measured per-row-count cost table and the online acceptance rate say it pays | adopted (step 9): c1 +2.6 %, c4 +2.4 %, c8 +2.7 % |
| `HIVE_MTP_GATE3` | off | `1` | Drop cost samples taken while a CUDA graph was being captured | adopted (step 12) |
| `HIVE_MTP_VERIFY2` | off | `1` | Run verify rows through the decode path (decode kernels, grouped experts) | adopted (step 12, with `_FUSED`): two-row verify at 80K 62.9 → 32.0 ms; decode after 80K +6.2 % |
| `HIVE_MTP_VERIFY2_FUSED` | off | `1` | Use the fused attention front for verify rows (needs `HIVE_DECODE_ATTN_FUSED=1`) | adopted (step 12) |
| `HIVE_MTP_VERIFY_ROWIND` | off | — | Make verify rows independent (all misses on CPU, fixed accumulation order) | rejected: same results, row independence not achieved |
| `HIVE_MTP_BATCH` | off | — | Speculative decoding with ≥ 2 active sequences (total rows ≤ 8) | rejected: lossless, but c2 +0 %, c4 up to −8 % on this machine; re-measured on the current verify path (`HIVE_MTP_VERIFY2`, `HIVE_MTP_GATE3`, step 19 build): c2 37–38 vs 37–41 off, c4/c8 unchanged — no gain, kept off |

### Prompt checkpoints and prefix reuse

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_CKPT_ASYNC` | off | `1` | Prompt checkpoints copied to pinned memory asynchronously with completion fences | adopted, no separate A/B |
| `HIVE_CKPT_PINNED_POOL_MB` | 0 | `4096` | Pool of reusable pinned buffers for async checkpoints (only with `HIVE_CKPT_ASYNC`) | adopted, no separate A/B |
| `HIVE_CKPT_DELTA` | off | `1` | Share the completed compressed-KV/index prefix between checkpoints of the same generation; copy only the suffix (GLM: the DSA latent / indexer and exact MTP rows — glm.md) | adopted; GLM measured together: [performance.md](performance.md#steps-in-order-1) GLM step 23 |
| `HIVE_CKPT_VERIFY` | off | (monitoring) | GLM: every N-th snapshot with a shared prefix compares that prefix with the device rows and prints `[glm-ckpt] verify … ok` or a mismatch (the monitor reports it as an error) | check, not a behaviour change |
| `HIVE_CKPT_HISTORY` | 1 | `4` | Prompt checkpoints kept per session (≥ 1) | adopted, no separate A/B |
| `HIVE_IMAGE_CKPT` | off | `1` | Reuse checkpoints of image prompts when tokens and the image signature (bytes, types, grid) match | adopted, no separate A/B |
| `HIVE_DEFER_CKPT` | off | `1` | Take the prompt checkpoint and warm the cache after the first token is sent | adopted; GLM measured together: [performance.md](performance.md#steps-in-order-1) GLM step 23 |
| `HIVE_PREFIX_SHARE` | off | `1` | Snapshots at boundaries hinted by the server (end of system/tools, end of recent assistant turns), shared across sessions by exact token prefix; the server computes the hints only when this is on | adopted, no separate A/B |
| `HIVE_PREFIX_EXTRA_CHUNKS` | 1 (with `PREFIX_SHARE`) | `0` | Extra chunks per request a boundary cut may add — a request can raise it for itself (`prefix_extra` in the daemon request, sent by the server with `HIVE_PREFIX_ADAPTIVE`) | adopted (step 16): follow-up turns −25 % TTFT, 80K conversation + 1000 tokens 3.00 → 1.98–2.01 s; GLM (2026-10-07, `config/glm.env` had kept the default 1): 22K-token first turn with a new system block 8.8–9.2 → 7.3–8.1 s. Pair it with `HIVE_PREFIX_FIRST_TURN=seen` so new conversations still share a repeated system block |
| `HIVE_VIT_CACHE_MB` | 0 = off | `2048` | Host LRU cache of vision-encoder outputs (hit only on identical patch bytes, types and grid) | adopted, no separate A/B |

### Engram tables on SSD

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_ENGRAM_SSD` | off (RAM) | — (optional) | Keep the Engram value tables in the checkpoint on disk and read rows with O_DIRECT; scales stay in RAM. `2`/`all` = scales on disk too | optional: resident 478 → 295 GB, cold start −34 %, throughput −1 to −6 %, byte-identical lookups ([performance.md](performance.md#the-engram-tables-on-ssd)) |
| `HIVE_ENGRAM_SSD_CACHE_MB` | 2048 | — | Row cache (CLOCK); `0` = no cache | — |
| `HIVE_ENGRAM_SSD_THREADS` | 64 | — | Reader threads = queue depth (1–256) | — |
| `HIVE_ENGRAM_SSD_NO_PREFETCH` | prefetch on | — | Disable reading rows ahead from token ids | comparison |

A read that fails is retried three times; then only that request fails.

### Sleep and wake

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_SLEEP_VMM` | off | — (optional) | Allocate device memory through CUDA virtual memory management so level-3 sleep can release dense weights and restore them at the same addresses | optional |
| `HIVE_SLEEP_PREPIN` | **on** with VMM | — | Allocate the pinned host shadows for level-3 sleep in the background at start; `0` = allocate at the first sleep | adopted (step 15): first sleep 6.8 → 0.6 s, byte-identical restore |
| `HIVE_START_ASLEEP` | off | — | Load to RAM, then report `sleeping` with no expert cache or session pool; the first wake allocates and warms them | optional |

### Scheduling and yielding

| Switch | Default | hive.env | Meaning | Verdict |
| --- | --- | --- | --- | --- |
| `HIVE_LAYER_YIELD` | off | `1` | Inside a long prefill forward, at layer boundaries every period (ms ≥ 50; `1` = 500), park the prefill state in pinned memory, admit waiting short text requests up to their first token and run decode steps | adopted (step 14): short request behind an 85K prefill 14.8 → 1.1 s; running decode stall 13.4 → 2.3 s. Periods 1000/2000 ms were worse |
| `HIVE_LAYER_YIELD_SHARE` | 0.05 | — | Decode budget per yield as a share of the preceding prefill time | — |
| `HIVE_LAYER_YIELD_STEPS` | 4 | — | Decode steps per yield (≤ 64) | — |
| `HIVE_LAYER_YIELD_MAX` | threshold − 1 | — | Largest request admitted during a yield, counted in rows to prefill (its ids minus the prefix the daemon can resume from — the live state, prompt checkpoint, history frames or archived state; boundary snapshots are not counted). A follow-up turn of a long conversation therefore counts only its new tokens | — |
| `HIVE_WARM_BUSY_CAP` | off | — | While other requests are decoding, cap the synchronous cache warm that follows a prefill at N experts (`1` = effectively skip; the per-step promotions fill the cache instead). Measured: after a follow-up turn's 699-row prefill the warm uploaded 1,065 experts in 693 ms, stalling every running decoder | rejected (performance.md step 19): with `1` the decoders' longest gap fell 1.44 → 1.04 s, but the cache went stale — 17.0 → 10.3 tok/s per decoder. Superseded by `HIVE_WARM_DEFER` |
| `HIVE_WARM_DEFER` | off | `32` | The cache warm after a prefill is not uploaded synchronously but handed to the following decode steps: each step's promotion runs with the warm's cap (N experts) and threshold until `warm_cap` experts were issued or no candidate is left (`Runtime::warm_defer`). Same promotions, paced on the side stream; takes precedence over `HIVE_WARM_BUSY_CAP`; ignored with `--promote-misses` | adopted (step 19 default `32`): decoder throughput with 4 streams 17.0 → 21.5–21.6 tok/s, cached follow-up turns −0.5 to −1.3 s, the decoders' longest stall 1.44 → 1.07 s; `64` gives no further gain (21.5) |
| `HIVE_LAYER_YIELD_MID` | off | `16384` | Also admit a request of more than `HIVE_LAYER_YIELD_MAX` rows (up to this many) at a yield when its rows × 2 ≤ the paused forward's remaining rows, at most that forward's rows per forward; the admitted request's forward yields once more for decode steps only | adopted (step 23): 12K request behind an 85K prefill 14.6 → 4.4 s to the first token, the long prompt 13.9 → 18.2 s, decoder stall 1.83 → 2.05 s |
| `HIVE_LAYER_YIELD_SMALL` | off | `1` | Layer-boundary yields also inside `HIVE_PREFILL_SMALL` forwards (prompt chunks below the prefill threshold — a few hundred new tokens of a follow-up turn take 0.5–1.5 s, during which no decode step ran) | adopted (step 20, on top of step 19): with the post-prefill warm gone, the decoders' longest stall is that turn's own small prefill — 1.05–1.07 → 0.59–0.60 s; cost 4-stream decode 21.5 → 21.2 tok/s (−1.5 %) and follow-up turns +0.05–0.1 s. Measured before step 19, with the synchronous warm still in place, it showed no gain (the warm stall dominated) |
| `HIVE_BATCH_COALESCE_MS` | 25 with layer yield, else 0 | — | Before the first batched-prefill round, wait until arrivals pause this long (cap: ms × slots); `0` = off | adopted (step 14 default): 2.5K × 4 simultaneous requests run together again (5.1–5.2 s instead of one alone, then three) |
| `HIVE_PREFILL_YIELD` | off | — | Admit short requests at existing chunk boundaries of a long prefill | rejected: an 85K prompt is one chunk, no boundary to yield at |
| `HIVE_PREFILL_YIELD_MAX` | threshold − 1 | — | Size limit for that | (with prefill yield) |
| `HIVE_BATCH_WINDOW` | off | — | Wait up to `HIVE_BATCH_WINDOW_MS` (50) for followers before the first batched round | rejected (with SJF/fair): short-request TTFT during a 30K prefill 3.6 → 6.3 s |
| `HIVE_BATCH_WINDOW_MS` | 50 | — | Window length (≤ 10 s) | — |
| `HIVE_BATCH_SJF` | off | — | Give batch slots first to requests that can finish this round | rejected |
| `HIVE_PREFILL_FAIR` | off | — | Give running decoders their share between consecutive prefills | rejected |

### Diagnostics, tracing and comparison paths

None of these is needed in production. Tracing has overhead; keep it off when measuring speed.

| Switch | Default | hive.env | Meaning |
| --- | --- | --- | --- |
| `HIVE_PROFILE` | off | (monitoring: `64`) | Per-phase timings; N = profile every N steps. The `[decode-host]` line also gives the step-head promotion commit wait (`promo wait … ms/step`, mean over every step since the previous line). GLM: `[profile]`, `[decode-host]` and `[early-route]` in the same formats, from CUDA events on the sample step (no host synchronization added); its `[decode-host]` adds `defer wait` (host time blocked on deferred CPU experts). Host time around the forward: hived prints `[step-host <kind>]` every N decode steps of a kind (both models — gap between steps, pre, draft, forward call, post); GLM adds `[call-host <kind>]` (engine call, logits copy, sampler candidates), `[fwd-host]` on the sample step (host wall against the GPU entry / window / head spans) and one `[glm-prefill]` line per prefill call (yield, MoE prep / GPU / CPU join, KDA / DSA attention, other, streamed GB/s). The daily report turns them into the decode host-time, prefill and TTFT breakdown tables |
| `HIVE_TRACE_CACHE` | off | (monitoring: `1`) | Per-token cache hit/resident lines and snapshot lines (GLM: the `[cache]` line of every decode / verify step) |
| `HIVE_TRACE_MTP` | off | (monitoring: `1`) | Per-step draft/accept/confidence log |
| `HIVE_PREFILL_PROF` | off | (monitoring: `1`) | Per-chunk `[prefill-prof]` phase breakdown (DeepSeek only) |
| `HIVE_EXPERT_TRACE` | off | (monitoring: a path) | Append a binary expert-routing trace to this file (input for cache replay) |
| `HIVE_CACHE_EVENTS` | off | — | Append cache transitions (use/evict/issue/commit/place) as CSV to this file; checked by `tools/check_cache_events.py` |
| `HIVE_LOAD_CHECKSUM` | off | — | After loading, print a 64-bit hash of every host copy (expert arenas, Engram tables) to compare loaders |
| `HIVE_ENGRAM_DIGEST` | off | — | Per-layer hash of every Engram lookup (compare RAM and SSD modes) |
| `HIVE_GRAPH_DUMP` | off | — | Keep CUDA graphs on while dumping (`hive --dump` otherwise runs eagerly); final state only |
| `HIVE_VISION_DUMP` | off | — | Directory for vision-encoder stage dumps (`tools/compare_vision.py`) |
| `HIVE_ATTN_V1` | off | — | Prefill sparse attention v1 kernel (comparison) |
| `HIVE_ATTN_V2` | off | — | Prefill sparse attention v2 kernel instead of the flash kernel (comparison; allocates full scratch) |
| `HIVE_ATTN_TC` | off | — | Three-kernel tensor-core decode attention instead of the fused one (comparison) |
| `HIVE_ATTN_FLASH32` | off (*integer*) | — | Flash attention v3 (32-head blocks) instead of v4 (64); bit-identical (comparison) |
| `HIVE_WOA_TC` | off | — | Older bf16 × fp8 tensor-core kernel for prefill `wo_a` instead of dequantise + cuBLAS (comparison) |
| `HIVE_HC_MIX_ROWS4` | off (*integer*) | — | Older 4-row HC-mix kernel (comparison) |
| `HIVE_HC_CUBLAS` | off | — | HC mixing through cuBLAS fp32 GEMM (the original path, comparison) |

## API server settings

Read by `server/hive_server.py` (the container runs it with `--ckpt $HIVE_CKPT_MOUNT --sock /out/hive.sock
--host $HIVE_BIND --port $HIVE_PORT`). See [install.md](install.md#exposure-and-authentication) before exposing the port.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HIVE_GLM_NOTHINK` | `low` | GLM only: what a no-thinking request becomes — `low` (the official minimum effort) or `empty` (an empty thinking block is prefilled; not an official mode) |
| `HIVE_THINK_EXIT_TEXT` | "I have enough reasoning. I will now stop thinking and write the response." | Phrase forced before `</think>` when `max_thinking_tokens` is reached; empty = only `</think>` |
| `HIVE_BIND` | `127.0.0.1` | Address the API server binds (entrypoint); `0.0.0.0` exposes it on every interface of the host (host network) |
| `HIVE_API_KEY` | unset | When set, every endpoint except `/health` requires `Authorization: Bearer <key>` or `x-api-key: <key>` (401 otherwise) |
| `HIVE_MAX_BODY_MB` | `64` | Request bodies above this size are rejected with 413 before parsing |
| `HIVE_IMAGE_FETCH` | on | Image inputs (and GLM video inputs) may be `http(s)` URLs fetched by the server; `0` or empty = `data:` URLs only. Filesystem paths are always rejected |
| `HIVE_REQUEST_LOG` | `/out/logs/requests-server.jsonl` if `/out/logs` exists | One JSON line per request (timings, outcome, client tags — no prompt text); empty = off |
| `HIVE_CLIENT_TAG_PREFIXES` | `x-hive,x-client` | Comma-separated header-name prefixes (case-insensitive) recorded as client tags in the request log, besides `user-agent`, the body `user` and `metadata`; add your gateway's header prefix here; empty = default. Header names that look like credentials (`authorization`, `cookie`, `x-api-key`, anything containing key/token/secret/auth) are never recorded |
| `HIVE_PREFIX_SHARE` | off | Compute the boundary hints sent to the daemon (same switch as the engine) |
| `HIVE_PREFIX_ADAPTIVE` | off | With `HIVE_PREFIX_SHARE`: ask the daemon for one extra boundary chunk (`prefix_extra`) on a request whose conversation is not append-only — the previous request was not a prefix of this one but shared one of its boundary hints (a block near the end is replaced each turn). The deepest boundary of that request is saved, so the next turn resumes from it instead of prefilling the whole prompt. Append-only conversations keep `HIVE_PREFIX_EXTRA_CHUNKS`. The server keeps only hashes, for 4,096 conversations (LRU). Measured: [performance.md](performance.md#steps-in-order) step 17 |
| `HIVE_PREFIX_FIRST_TURN` | off (`seen` in both shipped configs) | With `HIVE_PREFIX_SHARE`: also ask for the extra chunk on the first turn of a conversation (no assistant message yet), so the end of the system/tools block is saved as a shared snapshot that later new conversations resume from. `1` = on every first turn. `seen` = only when another conversation already sent exactly that block (the server keeps a 16-byte hash per boundary for 4,096 blocks, LRU): a block that never repeats (a per-document date, per-conversation memory) pays no cut, the second conversation with a block pays it, the third and later resume. With `HIVE_PREFIX_EXTRA_CHUNKS=0` and this off, new conversations never share a block — `HIVE_PREFIX_ADAPTIVE` tracks one conversation and does not see a new one. Measured 2026-10-07 (21.5K-token block, ~0.6K-token question, 3 runs each): DeepSeek 3rd and later conversations 4.34–4.37 → 0.84–0.92 s, 2nd 4.33–4.35 → 5.13–5.35 s, a block seen once unchanged (4.3–4.7 s); GLM 3rd and later 7.3–7.7 → 1.6–2.1 s, 2nd 9.0–9.3 s. Earlier (DeepSeek, 9.9K-token block, `1`): first conversation 3.8–4.5 → 6.1–6.5 s, second 6.2 → 1.4 s |
| `HIVE_TOOL_STREAM` | on | DeepSeek only (GLM tool calls are always sent when complete): stream DSML tool calls as `tool_calls` deltas while they are generated (id and name first, then argument pieces); `0` = collect the block and send the parsed calls at the end. Either way the arguments equal the reference parser's; the request log records `tool_stream` (match / incomplete / mismatch) for streamed calls. Both families: when a call's arguments are not a JSON object a client can read (not JSON, NaN / Infinity, a non-finite number — for example a string the model marked `string="false"`), the request log records `tool_args_invalid` (the tool names); the call is sent unchanged |
| `HIVE_TOKEN_CACHE` | on | Tokenize only the changed tail of a conversation's prompt: per conversation the last prompt, its token ids and offsets are kept and a new prompt is cut at the last role marker (an added special token) inside the common prefix — a fast tokenizer splits those out first, so the result equals a whole-prompt tokenization. Tokenizers whose template appends tokens at the end are not cached. 159K-token conversation, next turn: prepare 554 → 27 ms (GLM), 590 → 25 ms (DeepSeek); first request 562 → 314 ms (one tokenization with offsets instead of two). `0` = off |
| `HIVE_TOKEN_CACHE_SESSIONS` | `64` | Conversations kept by the token cache (LRU; ~2 MB each at 150K tokens) |
| `HIVE_TOKEN_CACHE_VERIFY` | `32` | Every N-th cached tokenization is compared with a whole-prompt tokenization on a background thread; a mismatch turns the cache off for the process and is logged (`[server] token cache MISMATCH`). `0` = no check |
| `HIVE_STAGE_STATUS` | off | Engine-side stage of each request at `GET /v1/hive/requests/{id}` (`id` = the response id of the first stream chunk): `queued` (waiting for admission), `prefill` (with the reused and to-be-read token counts and the tokens read so far), then `thinking` / `answer` / `tool_call` (with the tool name once written) and `done` / `error`; entries are kept 120 s after the end. Off = the route answers 404 and responses are byte-identical — the status is a separate channel for clients that want to show what a response is waiting on |
| `HIVE_REPORT_CLIENTS` | unset (all traffic) | Read by `tools/hive_daily_report.py`, not by the server: comma-separated user-agent prefixes whose requests count as real use; requests carrying an `x-client-test` header (the bundled benchmarks set it) are left out of the per-request sections |

Variables used only by the test harnesses (`HIVE_TEST_SANITIZER`, `HIVE_DAEMON_NEGATIVE`, ...) are listed in
[validation.md](validation.md#harness-variables).

## HTTP request fields

### `POST /v1/chat/completions`

| Field | Default | Meaning |
| --- | --- | --- |
| `messages` | — | Chat messages; image parts (and `video_url` parts on GLM) and tool calls/results are rendered by the model's own encoding (DeepSeek: `encoding/`; GLM: `chat_template.jinja`) |
| `tools` | — | Function definitions (rendered into the system message) |
| `stream` | false | Server-sent events |
| `max_tokens`, `max_completion_tokens` | rest of the context | 0 or absent = up to `max_ctx` minus the prompt |
| `temperature` | 1.0 | Sampling temperature (0 = greedy) |
| `top_p` | 0.95 | Nucleus sampling |
| `top_k` | 0 = off | Top-k |
| `min_p` | 0 | Min-p |
| `seed` | 0 | Random seed |
| `stop` | — | Stop strings, matched on the visible answer text only (not in reasoning) |
| `ignore_eos` | false | Do not stop at EOS (fixed-length benchmarks) |
| `reasoning_effort`, `reasoning.effort`, `chat_template_kwargs.reasoning_effort` | 75 (GLM: `high`) | DeepSeek: `none` (thinking off), `minimal` 25, `low` 50, `medium`/`high` 75, `xhigh`/`max` 100, or a number 1–100. GLM: `low`, `high`, `max`; `none`/`minimal` → `low` (or an empty thinking block with `HIVE_GLM_NOTHINK=empty`), `medium` → `high`, `xhigh` → `max`, numbers < 50 / < 90 / ≥ 90 → `low` / `high` / `max` |
| `chat_template_kwargs.thinking` (or `enable_thinking`) | true | `false` = no reasoning on DeepSeek; on GLM the same as `none` above |
| `max_thinking_tokens` (alias `thinking_token_budget`; top level or in `chat_template_kwargs`) | no cap | Reasoning cap: at N reasoning tokens the daemon forces `HIVE_THINK_EXIT_TEXT` + `</think>` and the model writes its answer |
| `user` | — | Mixed into the session key |
| `hive_session_id` | — | Explicit session identifier for conversation reuse |
| `metadata` | — | Logged only |

The session key is otherwise a hash of the messages up to the first user message. Concurrent requests of the
same conversation get auxiliary sessions (`~1`, `~2`, ...). A prompt longer than the context returns HTTP 400
with code `context_length_exceeded`. Responses carry the standard `usage` and a `hive` object (prefill/decode
ms, cached prefix length, cache hits, CPU experts, DMA rows, CPU wait, batch rows, MTP steps/drafted/accepted,
and the reasoning-cap counters when a cap was requested).

### `POST /v1/messages`

| Field | Meaning |
| --- | --- |
| `system`, `messages` | Text, `image`, `tool_use` and `tool_result` blocks |
| `tools` | `name`, `description`, `input_schema` |
| `max_tokens`, `temperature` (1.0), `top_p` (0.95), `stream`, `stop_sequences` | As above |
| `metadata.user_id` | Mixed into the session key |
| `hive_session_id` | Explicit session identifier |
| `thinking` | `{"type": "disabled"}` = no reasoning; `budget_tokens` selects the effort: < 2000 low, < 8000 medium, otherwise high (it is not a token cap) |
| `output_config.effort` | Effort level, as `reasoning_effort` |

`top_k`, `min_p`, `seed` and `max_thinking_tokens` are not forwarded on the messages endpoint.

### Control endpoints

| Endpoint | Body | Meaning |
| --- | --- | --- |
| `POST /release_memory_occupation` (alias `/admin/sleep`) | optional `{"level": 2\|3, "timeout_s": N}` | Sleep after running requests finish. HTTP 400 `not idle` if `timeout_s` passes first |
| `POST /resume_memory_occupation` (alias `/admin/wake`) | `{}` | Wake; HTTP 503 if VRAM is still taken (stays asleep) |
| `POST /flush_cache` | — | Drop every reusable prompt state (session checkpoints and history, archived sessions, boundary snapshots) so the next request of any conversation is prefilled from scratch; the expert cache is kept. HTTP 400 `not idle` while a request is decoding (retry when idle). For benchmarks between runs |
| `GET /health` | — | `loading` (503), `ready`, `draining`, `sleeping`, `waking`, plus VRAM use |
| `GET /live` | — | Liveness of the API server |
| `GET /v1/models` | — | One model, `hive` |
| `GET /v1/hive/requests/{id}` | — | With `HIVE_STAGE_STATUS`: the engine-side stage of a request (`id` = the response id of its first stream chunk); 404 when the option is off or the id is unknown (an entry is kept at least 120 s after the request ends, then dropped when a later request starts) |
