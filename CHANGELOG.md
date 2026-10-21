# Changelog

## Unreleased

- **Session RAM accounting no longer slows decoding as conversations pile up**: the host budget total (`host_total`, also the stats'
  `host_session_bytes`) walked every kept image and put every segment into one set to count shared prefixes once. An image is
  thousands of segments (median 5,994, p90 8,466 on the DeepSeek service), and the walk ran on the engine thread once a second while
  decoding and several times per request. After 39 h (1,649 boundary snapshots, 111 evicted sessions) it took 3.1 s of a 13 s profile:
  ~15 ms of idle engine time per decode step, 13–175-token follow-up prefills taking 3–4 s, single-stream decode down to ~24 tok/s.
  A ledger (`hive/host_ledger.h`) now keeps each image's metadata and each segment's holder count and only reads the images that
  appeared or went away since the last pass — the same total (`tests/test_host_ledger_cpu.cpp` checks it against the full walk), no
  image kept alive longer. 2,000 images / 400K segments: 139 ms per walk → 0.3 ms per pass. Its own RAM follows the live segments
  (~117 MB for 3.85M). After the restart the same short request decoded at 76–81 tok/s instead of 42.6, and a 22-token prefill took
  ~200 ms instead of 666.

- **No absurd time left while a prompt is read** (`HIVE_STAGE_STATUS`): a new 92,706-token read that took 24.7 s was shown about
  222 minutes left at 8,037 tokens (DeepSeek service, 2026-10-19). The kept fit had a negative cost at position 0 (a = -0.00037 s,
  b = 1.37e-8): the old check only looked at the chunks' mid positions, and the slope had been bent by 1–8-row tail chunks (4–32 ms
  per token against 0.2–0.3 for whole chunks), by a slower tail-layer chunk after a long one and by reads resuming 98K–300K tokens deep.
  Now a fit is used and kept only if a > 0, b >= 0 and it predicts its chunks' total time within ±50 % (else the read's own average
  speed with the kept slope); chunks under 1,024 tokens stay out of the fit; a read whose reused prefix is over a quarter of its end
  position does not replace the kept fit; `prefill-rate.json` stores the chunks it came from and is ignored at start unless it passes
  the same check (the old file has no chunks and is dropped); and a time left implying a speed 8x below the slowest (or above the
  fastest) whole chunk measured is reported as null, which clients hide. Replaying that day's reads: 23.2 s left at 8,037 tokens
  (22.3 s really left; the old code gave 12,891 s).

- **Gemma prefill GEMM, K split across a thread-block cluster**: the 17–192-row int4 GEMM was bound by L2 traffic (weights re-read per
  16-token block, activations per 64-row block); the default kernel now gives each of the 8 K-split ranks its own block of a cluster, so
  one staged activation tile feeds 128 rows and the weights are read once, and sums the ranks' partials in rank order — outputs
  unchanged, bit for bit (every variant tested against the decode GEMV at 1–200 rows and the model's shapes). Per sliding layer, cold
  L2: 70 rows 385 → 250 µs, 128 rows 520 → 387, 17 rows 209 → 125, 192 rows 711 → 553 (`tests/test_gemma_w4.cu`, 2026-10-19). The
  former kernel stays selectable (`HIVE_GEMMA_GEMM_V=0 … 5`). On the model (`tests/gemma_tail.cpp`, a conversation turn's tail after
  3,000 tokens, medians of 20): 70 rows 26.7 → 19.4 ms, 128 rows 34.9 → 29.4 ms; with the attention change above, the 70-row tail went
  from 28.9 to 19.4 ms. Two served conversations taking turns: first token 27 → 20 ms.

- **Fewer conversations read again** (measured on the DeepSeek service's request log, 2026-10-16 … 19):
  - A long prefill keeps a snapshot at a chunk end every 32,768 tokens (`HIVE_PREFIX_INTERVAL`, `0` = off): turn-end snapshots only
    cover the turns a request was read in, so after a long re-read a change deep in the history re-read everything again (a 280K
    conversation re-read from 0, then a change at 167,322 re-read all 297,628 tokens, ~100 s). No extra chunk, only the copy.
    After a chunk that skipped the tail layers (every full chunk of a long DeepSeek prefill) the snapshot is flagged like a boundary
    snapshot. Measured on the DeepSeek service (72 GB, 98,304-row chunks), a 199,508-token conversation read from the top and then sent
    with a message at ~164K changed: 0 → 98,304 tokens reused, 43.5 → 23.3 s (twice). The session store also kept its 650 conversations
    across both rebuilds of the day (canary KL 5.7e-6).
  - A client's fixed first user message followed by per-room system notices and then the conversation gets boundaries at the end of
    the block and at the end of the notices (only "followed by another user message" counted before). Measured with the DeepSeek
    template and tokenizer on a chat client's first turns: rooms of one person share 19,827–22,778 tokens and the deepest boundary
    went from the system/tools block (9,224–15,720) to the whole shared head. On the service, three rooms of one person in a row:
    the second and third resume 19,806 of 19,813 and 22,757 of 22,764 tokens (6.0 → 0.4–0.5 s).
  - The session store outlives rebuilds: its layout key no longer names the build; a canary state checked at start-up decides whether
    a new build reads the stored images (five rebuilds in one day had each dropped every stored conversation). Runs of one build are
    not bit-identical (CPU-computed experts), so the canary is prefilled twice and the stored one must agree within four times that
    run's own KL noise — measured across two builds: KL 1.35e-6 against a 1e-3 floor, stored conversations kept.
  - Every text request of a family that cannot rewind (DeepSeek) gets a generation-suffix hint at the end of its last message — one
    token earlier when the further message's separator merges with that token — and the daemon cuts every such hint: a request that
    repeats a prompt with a further message (no answer in between) continues there. A background job's second requests re-read
    23–44K tokens each before (1.63M tokens in three days); measured after: 26,120 of 26,138 tokens reused, 8.0 → 0.8 s. Gemma
    rewinds its live state instead and gets no suffix hints (they only left snapshots).
  - Gemma: the KV pool holds four conversations' sliding windows (it was sized for one max-length context, so two voice conversations
    taking turns evicted each other) and the keep margin follows the measured rewind depth (768; p99.5 600 tokens). A session brought
    back from a host image is rewound like a live one. Short prefills (≤ 128 rows) take the split or 16-row-tile attention: a 70-row
    turn tail 49.6 → 24.2 µs per sliding layer, 156.5 → 95.5 µs per full layer.
  - `/health` reads the device again on the publish that ends a sleep or wake (it showed the mid-wake reserve until the idle sweep).

- **The VRAM reserve survives sleep** (DeepSeek under `HIVE_VRAM_LIMIT_MB`): a buffer taken out of the reserve while serving gives its
  room back even when it is freed asleep, and wake refills the reserve. Before, sleep freed the lazily loaded vision encoder without
  that credit, so one sleep with it loaded shrank the reserve from 2,464 to 640 MiB for good and every later image request was refused
  while the encoder loaded. Measured on the service after the fix: two level-3 sleep/wake cycles with the encoder loaded, and the
  encoder (1,792 MiB) loaded again after each wake.
- **Priority reaches prefill**: a request whose priority beats every request being prefilled is admitted at a layer yield of that
  prefill and decodes to the end before it resumes (DeepSeek: a 6K-token conversation turn 3 s into a 60K-token background prefill
  decoded at 42–45 instead of 13–16 tok/s). With `HIVE_PREFIX_FIRST_TURN=seen` the first-turn cut goes to the deepest boundary another
  conversation already sent, so rooms that share a system block but differ after it share that block.
- **Gemma prefill GEMM**: forwards of 17 to 192 rows (a conversation turn's tail, short prompts) run an int4 tensor-core GEMM that keeps
  the decode GEMV's arithmetic bit for bit but stages the activations in shared memory: a 70-row turn tail after 3,000 tokens 42.7 →
  28.9 ms, 128 rows 70.4 → 36.6 ms; outputs unchanged. Qwen3-ASR lowers the per-thread stack limit after its context is created, as
  Qwen3-TTS does (its kernels use no stack; ~210 MiB of VRAM).

- **Gemma 4 12B reads images** ([docs/gemma.md](docs/gemma.md)): the checkpoint's encoder-free vision embedder (~95 MiB) turns 48 × 48
  pixel blocks straight into soft tokens, and the soft tokens of one image attend to each other both ways (`use_bidirectional_attention:
  "vision"`). The server's preprocessing gives the Hugging Face processor's token ids and positions; teacher-forced against the fp32
  oracle on three images 58 / 59 (one near-tie). Text output is bit-identical to before, apart from the end-of-image / end-of-audio
  markers, which are now suppressed as the reference generation does (`suppress_tokens`). `HIVE_GEMMA_VISION=0` leaves the embedder out.

Four models at four VRAM sizes. Every family is measured with the same harness at 96 / 72 / 48 / 32 GB (`HIVE_MODEL`, `HIVE_VRAM_GB`;
smaller cards reproduced with the engine's VRAM limit) — the table is in the [README](README.md#performance-by-vram-size).

- **Three new families for a real-time voice assistant**, all measured against the servers they replace with those servers' own client
  harnesses on the same card:
  - **Gemma 4 12B** (`hived_gemma`, [docs/gemma.md](docs/gemma.md)) — int4 QAT weights, paged fp8 KV whose sliding-window pages are
    released behind the window (and moved to mapped host memory when the pool is full), the Frozen-KV MTP assistant as the drafter,
    tool calls streamed as they are generated. First token p50 42 ms, first sentence 96 ms, 303 tok/s (the reference server: 69 ms,
    179 ms, 181 tok/s), same tool-call and delegation accuracy.
  - **Qwen3-ASR 1.7B** (`libhive_qasr.so` + `server/speech/asr_server.py`, [docs/qasr.md](docs/qasr.md)) — `/generate` with forced or
    auto-detected language: p50 72 ms, CER 0.0054 (the reference server: 152 ms, 0.0067), 4.8 GB.
  - **Qwen3-TTS 12 Hz 1.7B CustomVoice** (`libhive_qtts.so` + `server/speech/tts_server.py`, [docs/qtts.md](docs/qtts.md)) — streaming
    WebSocket speech: first audio chunk p50 25 ms, real-time factor 0.069 (the reference server: 163 ms, 0.33), 4.0 GB.
  The three run beside DeepSeek at 72 GB on one 96 GB card, each under its own `HIVE_VRAM_LIMIT_MB`.
- **The VRAM limit holds while serving (DeepSeek)**: with `HIVE_VRAM_LIMIT_MB`, the ballast that emulates the smaller card goes back
  to the card at ready and the headroom left inside the limit becomes a reserve; runtime allocations come out of it and are refused past
  it, so the daemon stays at its limit beside another process instead of taking the memory that process frees, and the other process
  keeps room to grow. GLM, Qwen and Q27 keep the ballast as before (they size growth from the device's free memory).
- Fixed: a refused device allocation aborted the daemon and every open conversation with it; it now fails only its request
  (`DeviceOutOfMemory`), for all four families.
- **A thinking cut no longer spreads** (`HIVE_CUT_LEAK_STRIP`, off by default): when the thinking cap cuts a response, the model often
  writes its unfinished reasoning as text in front of the tool call, and once that text is in the history the next answers copy it.
  How the thinking is closed made no difference (exit phrase or none, other phrases, closing at a newline or sentence end, a wrap-up
  phrase before the cap: 47-58 % of the answers began with such a preamble in a Korean DeepSeek-V4.1-Flash agent room); leaving those
  preambles out of what the model sees did (49 % → 2 %). The server keeps a fingerprint of each such content (on disk) and renders it
  as empty on later requests; the client's history is unchanged. The fingerprint ignores whitespace and think tags (a model that
  closes its thinking again in the content leaves a stray `</think>` that clients drop before sending the history back).
- **Conversations survive a restart** (`HIVE_SESSION_STORE`): prompt checkpoints and boundary snapshots are written to disk as they are
  made (increments only, on a writer thread, atomic files with length and hash checks) and read back when a request needs a conversation
  the daemon no longer holds — an engine restart or a model switch no longer means re-reading every long conversation once.
- **The next turn starts where the user writes** (`HIVE_TURN_PRERENDER`): after an answer, the server reads the conversation ahead in the
  form its next request will have (this turn's reasoning removed), so a new user message no longer re-reads the whole last turn.
- **Re-read per conversation** (`tools/reread_report.py`, a section of the daily report): the server records, for each follow-up
  request, the length of the conversation's previous prompt and how much of it the new prompt keeps (`prev_prompt_tokens`,
  `diverge_at`); with the engine's reused prefix this splits every token read a second time into a client rewrite (an earlier
  message changed) or an engine loss (the conversation's state was gone, or no snapshot close to where it changed), per conversation,
  next to what reading ahead cost, how many read-aheads failed and which next turns did not match their read-ahead (and where).
- Fixed: read-ahead tasks were only weakly referenced and about one in five was garbage-collected before it finished (logged as
  `Task was destroyed but it is pending` / `aclose(): asynchronous generator is already running`, no snapshot made); with a client
  filter (`HIVE_REPORT_CLIENTS`) the daily report dropped every read-ahead and showed 0; a turn with both a client rewrite and an
  engine part was counted whole under the engine.
- **GLM keeps earlier reasoning** a client sends back (the official template's `clear_thinking` default; `HIVE_GLM_CLEAR_THINKING=1` = before).
- **Reading progress inside a chunk** (`HIVE_STAGE_PROGRESS_MS`, default 1000): requests that asked for stages now get how much is read
  about once a second while a prefill forward runs its layers, not only after each chunk — a long prompt no longer sits at 0 % for a
  whole chunk (about 25 s on GLM, 20 s on DeepSeek, 9 s on Qwen3.8-Flash-Next). Taken at the layer boundaries the engine already
  visits — on the GPU as what the GPU has executed (an event per layer boundary, queried without waiting), and on Qwen3.8-Flash-Next
  across the sub-forwards a chunk is split into (one clock read and one event record per layer, no GPU synchronization; independent
  of `HIVE_LAYER_YIELD`). Measured on all four models: updates every 1-1.5 s (the longest gap 3.5-5.5 s, at the first layer of a
  chunk), the same read time as before, and the predicted end within -8 % to +5 % once a request has read a few chunks (in-chunk
  points only move the shown progress; the time left is fitted to whole chunks and keeps the last long read's fit until a request's
  own chunks span a quarter of the read).
- **Reading progress with a time left** (`HIVE_STAGE_STATUS`): while a prompt is read, the request stage also gives the measured read
  rate, the seconds left and whether the server already served the conversation, so a client can show "43% · about 1 min 23 s left"
  instead of a bare spinner during a long first read (380K tokens took 138 s on GLM). The estimate is the engine's own: a cost per token that grows with
  the position, fitted to the chunks read (measured on all four models, the predicted end at half read is within 2-8 % of the real one;
  the last chunk's rate alone was 20 % short on the 27B model, whose speed falls ~4x over 227K tokens).
- **The turn after a long re-read resumes** (all families, [docs/performance.md](docs/performance.md) step 33): a conversation re-read
  from the top (its head changed, or the daemon lost the state) now keeps a boundary snapshot — decided in the daemon on its own evidence
  (`"history"` from the server, at least one prefill chunk before the deepest boundary) and flagged by the server's tail tracker — so the
  next user message (reasoning templates drop the last turn's thinking) resumes instead of re-reading everything again. The prompt
  checkpoint shares the snapshot's rows, a reset releases checkpoints the conversation moved away from, Qwen's generation-suffix hint sits
  one token before the end (`<think>\n` merges into `\n\n`), and segmented 27B images load without joining. Next turn after a re-read:
  GLM 54 → 4.8 s, Flash-Next 29 → 1.0 s, 27B 81 → 6.0 s (and 27B after a 62K turn 15.4 → 1.2 s). `tools/reread_check.py` reproduces it.
- **GLM-5.3-Flash at 72 / 48 / 32 GB** ([docs/glm.md](docs/glm.md#smaller-vram-budgets)): the layer-major prefill span is lowered to what
  the expert-cache tail can lend (`GlmEngine::layer_major_span`) instead of failing; `HIVE_GLM_PREFILL_ROWBLOCK` bounds the per-expert
  prefill scratch, so the 64K prompt runs in one pass (64K first token 20.2 / 20.9 / 22.0 / 28.2 → 16.2 / 16.3 / 16.4 / 22.4 s at
  96 / 72 / 48 / 32 GB); `HIVE_GLM_MTP_LEND` lends the NextN layer's 6.75 GiB of experts to the expert cache while several streams decode;
  `HIVE_GLM_CPU_CHUNKS` balances the CPU expert workers. Profiles: `--max-chunk 8192`, `HIVE_GLM_KV_BASE=4096`, reserve 1500 MiB,
  single-stream MTP. c1 / c8 at 48 GB 48.1 / 57.2 → 53.3 / 77.8 tok/s, at 32 GB 34.5 / 36.2 → 34.9 / 50.9.
- **Qwen3.8-Flash-Next with a partial expert cache** ([docs/qwen.md](docs/qwen.md)): decode / verify stay on the device — non-resident
  experts read over PCIe from pinned memory while at least 60 % are resident (`HIVE_QWEN_MAPPED_ROUTE`), below that computed by the CPU
  in CUDA-graph host nodes (`HIVE_QWEN_HYBRID_ROUTE`, `HIVE_QWEN_HYB_PCIE`); captured graphs capped under a budget
  (`HIVE_QWEN_GRAPH_CAP`, `HIVE_QWEN_GRAPH_ROWS`). c1 at 72 / 48 / 32 GB 153.6 / 80.5 / 79.0 → 163.6 / 132.7 / 87.8 tok/s.
- **Qwen3.8-27B under a budget** ([docs/q27.md](docs/q27.md#vram-budget-and-the-kv-tier)): placement counts every device allocation,
  the drafter is placed first, and the KV tier keeps KV in VRAM while the budget lasts and in pinned host memory beyond it (same kernels,
  same values), so 262K context works at 32 GB at the all-GPU decode speed: growable host ranges, pinned ahead of use, and a prefetched
  staging copy for prefill attention (32 GB: 32K / 64K first token 7.2 / 21.0 → 4.6 / 11.8 s). New switches `HIVE_Q27_KV_TIER`,
  `HIVE_Q27_KV_PLAN`, `HIVE_Q27_KV_HOST_*`, `HIVE_Q27_STAGE_PREFETCH`, `HIVE_Q27_DRAFTER_FIRST`, `HIVE_Q27_HEAD_FIRST`,
  `HIVE_Q27_EMB_MAPPED`, `HIVE_Q27_W4_PARTIAL` (off), `HIVE_Q27_KV_TIER_MIN_MB`, `HIVE_Q27_LOAD_SLACK_MB` and diagnostics.
- **CPU expert kernels** (bit-identical): NVFP4 v2 (`HIVE_CPU_NVFP4_V2`, GLM / Flash-Next misses and CPU daemons) and register
  accumulators for the MXFP4 multi-row kernel (`HIVE_CPU_E2M1_REG`, DeepSeek); pool microbenchmarks in `engine/tests/bench_*_pool_cpu.cpp`.
- **Session images** of every family are freed off the engine thread (`hive/image_reaper.h`); `HIVE_HOST_SESSION_MB` is set for
  DeepSeek and the 27B as well and scaled down on hosts with less RAM.
- **1M-token context measured** for DeepSeek-V4.1-Flash and GLM-5.3-Flash at 96 and 48 GB (`HIVE_MAX_CTX=1048576`, no other change):
  short chat unchanged, ~1M-token prompt 335–470 s to the first token, the next turn of a 1M conversation about 4 s with the session
  kept ([README](README.md#reference-1m-token-context-glm-53-flash-deepseek-v41-flash)).
- **Two models on one card**: `HIVE_VRAM_BALLAST_RELEASE=1` frees the VRAM-limit ballast just before ready (startup sizing unchanged), and
  `HIVE_START_WAIT_OTHERS=0` lets `hive-start.sh` start beside another running daemon ([README](README.md#reference-two-models-on-one-card)).
- **Fix (27B):** a repeated long prompt on a reduced-VRAM 27B could crash during session restore — the KV reclaim could pick the session
  being restored. A session is now protected from the reclaim from the moment its sequence is taken until its admission job ends.
- **Fix (GLM):** long conversations could drain the expert cache to 0 slots and then fail a new request with `no VRAM for the KV
  cache` (48 GB, 1M context: 256K, 512K and 128K prompts in distinct conversations). Idle conversations kept the KV their sequence had
  grown (the GLM runtime had neither `trim_seq` nor hived's KV reclaim hook). KV growth now moves the least recently used idle
  conversation that holds KV to a host image before any cache slot is lent, LRU evictions and `/flush_cache` trim GLM sequences, and an
  evicted conversation continues from its image ([docs/glm.md](docs/glm.md#context-and-kv-memory)). hived skips idle sessions whose
  trim would free nothing (`kv_reclaimable`); `tools/test_kv_reclaim_cpu.py` covers the hook on the real hived.cpp.
- **Shared prefix for a leading user block**: clients that send fixed instructions as the first user message, followed by the user's
  turn, now get a boundary at the end of that block too (`boundary_hints`), so new conversations resume the whole shared head instead
  of only the system/tools block. hived logs a reset of a conversation revived from the host archive (it printed nothing) with its
  common prefix, and says why the deepest shared snapshot matching a reset prompt's head was not used.
- **Faster GLM prompt reads** (`HIVE_GLM_PREFILL_BALANCE`, on in `config/glm.env`): the CPU takes prefill experts until its share meets
  the GPU's transfer time instead of only those cheaper than one copy, so fewer experts cross PCIe (−20…−30 %). Time to first token at
  96 GB: 1.9K 3.90 → 3.17 s, 3.6K 4.84 → 4.11, 7.5K 5.98 → 5.37, 16K 7.44 → 6.98 s; decode unchanged.
- **Fix (launcher):** with `HIVE_MODEL` the launcher wrote the size profile's name (e.g. `glm-96`) into `HIVE_PROFILE`, the engines' timing
  switch, so `HIVE_PROFILE=N` never reached a daemon started that way. The name is now `HIVE_PROFILE_NAME`.
- **Fix (DeepSeek CPU daemon):** an empty `HIVE_DS_CPU_TAIL` skipped the daemon's decoder-tail default while the engine ignored the
  empty value; empty now means unset in both places. The CPU suite's option audit is clean again (the GLM / Flash-Next CPU daemons'
  `HIVE_*_MTP_K` test and the 27B `HIVE_Q27_TIMING` interval parse were flagged only by the audit's line matching).
- `tools/bench_chat.py --long-timeout` (cap a long prompt's first token; the VRAM-size table uses 600 s).

New families ([docs/qwen.md](docs/qwen.md), [docs/q27.md](docs/q27.md)): **Qwen3.8-Flash-Next** (`model_type` `qwen4_exp`, NVIDIA NVFP4
checkpoint) as `hived_qwen` — Gated DeltaNet and QSA sparse-attention kernels, gated-residual hyper-connections, the PLE n-gram table in
pinned RAM or read from the SSD (`HIVE_QWEN_PLE`), MTP drafting, the vision encoder, a startup VRAM budget (`HIVE_QWEN_VRAM_MB`), and
device-side expert routing; **Qwen3.8-27B** as `hived_q27` with the DFlash2 drafter; CPU-only daemons for all four families. The GLM
expert store is now built from a model-independent `ExpertSpec` (GLM behaviour unchanged). Server: `server/families/qwen.py`, `server/families/q27.py`; oracles:
`oracle/qwen4_oracle.py`, `oracle/q27_oracle.py`.

Real-chat benchmark (`tools/bench_chat.py`, 2026-10-09, one run each, GPU at 600 W and capped at 300 W; [docs/benchmarks.md](docs/benchmarks.md)):

| | DeepSeek-V4.1-Flash | GLM-5.3-Flash |
| --- | --- | --- |
| Decode, 1 / 2 / 4 / 8 / 16 / 32 streams (total), 600 W | 80.7 / 106.4 / 125.0 / 172.8 / 171.6 / 176.9 tok/s | 68.1 / 86.5 / 103.5 / 129.0 / 127.4 / 128.9 tok/s |
| The same, 300 W cap | 82.9 / 99.9 / 115.8 / 157.9 / 160.8 / 163.1 tok/s | 69.4 / 85.6 / 102.5 / 129.3 / 129.8 / 130.5 tok/s |
| Time to first token, 17K / 42K / 54K prompt, 600 W | 4.90 / 7.49 / 9.59 s | 7.21 / 11.31 / 13.74 s |
| The same, 300 W cap | 6.56 / 11.00 / 13.66 s | 8.17 / 15.25 / 18.90 s |

DeepSeek ([docs/performance.md](docs/performance.md), steps 29–32):
- Intra-layer yields (`HIVE_LAYER_YIELD_INTRA`, on in `config/hive.env`): decode-only yield points inside a layer of a long
  prefill (embedding units, unit / tile fronts, before the expert pass, indexer row batches), with the paused prefill's staging
  records held and its shared buffers parked. Bit-identical prompt logits and state on the GPU check. 85K prompt with a decoder:
  decoder tokens during the prompt 57 → 106–132, longest gap 1.86 → 1.16–1.49 s; the long prompt's first token +8–10 %.
- Batched speculation (`HIVE_MTP_BATCH`) re-measured on the current build and turned on: c2 88 → 100 tok/s (+13 %), first token
  median c2 0.5 → 0.25 s and c4 0.72 → 0.27 s, c4 / c8 throughput unchanged. The daemon op `{"op":"set","mtp_batch":N}` switches it
  at run time (used for the interleaved A/B).
- Short-prefill DMA band (`HIVE_DMA_BAND_SHORT=1.05`): the adaptive CPU / DMA share of short prefills now settles within 5 % instead of
  20 %; `[prefill-prof]` logs the experts sent to the CPU / DMA and the pre-copies used / wasted.
- Fix: a decode step that failed inside a layer yield could leave a deferred expert batch that the next short-path layer of the
  paused prefill would have added to its hidden state; deferred work is flushed before the prefill resumes and dropped after a host error.

Server: boundary-hint probe renders are kept per conversation (`HIVE_HINT_CACHE`) — a turn renders the template 4 times instead of
about 12; 156K-token conversation 29.1 → 12.2 ms per turn (CPU), hints identical.

DeepSeek (steps 22–23):
- Decode kernels: the fused expert kernel loads its activations a stage ahead, and the head reads each vocabulary row once for all
  rows of a step (`HIVE_HEAD_ROWS`) — both bit-identical; interleaved with the previous build c1 83.9 → 86.0 tok/s, c8 171.4 → 173.7.
- Layer yields also admit a request of up to 16K rows when it is at most half of what the paused prefill still has to do
  (`HIVE_LAYER_YIELD_MID`); its own prefill yields once more for decode steps. A 12K request behind an 85K prefill: 14.6 → 4.4 s to
  the first token.
- Rejected after measurement: a lower promotion threshold for the `seq` cache policy (c1 −2.5 to −4 %), and committing paced
  promotions one step head later (c4 and decode after a 54K prompt lower in both runs).
- `[decode-host]` gives the step-head promotion commit wait (`promo wait … ms/step`).

DeepSeek configuration: layer yields every 150 ms with a 0.2 decode share (were 500 ms / 0.05) — a streaming answer during a
background prefill stalled at most 0.42 s instead of 0.74 s; the overlapped prefill pays 349 → 469 ms per 1K tokens.

Server and daemon (both families):
- Fix: with reasoning on, every follow-up DeepSeek turn re-read most of the conversation — the prompt ends in `<think>`
  while the next turn renders the answer as `</think>…`, so the prompt-end checkpoint never matched. The server now sends
  a generation-suffix hint (found by rendering, no template knowledge) and the daemon cuts that tail of a few tokens for
  free. 35–45K-token conversation, effort low: 300-token turns 2.98 → 1.43 s, 1000-token turns 3.57 → 1.96 s.
- Conversation keys: `prompt_cache_key`, `session_id`, `X-Session-ID` or `user` are mixed into the session key, so
  conversations with an identical head (fixed instructions as the first user message) no longer share one session.
- Request priority: vLLM `priority` / OpenAI `service_tier`. Only the best priority among the running requests steps;
  the queue admits by priority. A conversation overlapping a background answer: 8.64 → 5.11 s.
- The token cache reuses the kept prompt with the longest common prefix when a conversation's key is new (154K tokens:
  285 → 8 ms).

- Fix: a boundary hint stopped one token too far when the next message began with the same character as the template's next
  text (for example a context block `<info-msg>` placed before the user's message, against `<｜Assistant｜>`), so the saved
  snapshot never matched the next turn and every follow-up turn prefilled the whole conversation. Hints now stop where the next
  content starts. Measured on DeepSeek (26K-token conversation with such a block): the next turn's first request 0 → 23,287
  tokens reused, prefill 6.9 → 2.2 s.
- `HIVE_PREFIX_ADAPTIVE` also saves a boundary for a conversation with earlier turns seen for the first time (after a restart),
  so its second turn there resumes instead of prefilling everything.

GLM ([docs/glm.md](docs/glm.md), steps 17–21):
- Fix: tool-call arguments typed `string` in the request's schema are kept as written. The parser decoded every argument as
  JSON, so a job id `3e382151` became `inf` (and the arguments `{"job_id": Infinity}`, which is not JSON), `0123` became 123.
  Untyped arguments still decode as JSON, but NaN / Infinity and non-finite numbers stay text; the arguments are always valid JSON.
- The request log records `tool_args_invalid` when a tool call's arguments are not readable JSON (both families; the call is
  sent unchanged) — DeepSeek copies values marked `string="false"` as written.
- Layer yields inside a prefill (short requests served, decoders keep stepping): short request 14.0 → 0.65 s, decoder stall 14.4 → 1.6 s.
- 8 sequences per decode step and batched MTP verify by default: c2 66.6 → 79.5, c8 93.2 → 125.7 tok/s.
- Promotion victims sorted once per step instead of a scan of every slot per used expert (the host held the GPU idle 1.88 ms per
  verify step): c1 67.3 → 70.4, c4 88.1 → 96.6 tok/s.
- CPU-expert chunks sized to the worker count (`HIVE_GLM_CPU_ADAPT`, CPU phase 70 → 81 GB/s); logits into a pinned buffer.
- Early routing on the fast decode path (`HIVE_GLM_EARLY_ROUTE`, the DeepSeek post-and-gate code): output identical; c4 95.3 → 99.4,
  c2 86.7 → 85.2 tok/s.
- Monitoring lines in the DeepSeek formats (`HIVE_TRACE_CACHE`, `HIVE_PROFILE`): `[cache]`, `[profile]`, `[decode-host]` (GPU idle
  by cause, deferral wait) and `[early-route]`, read by `tools/hive_monitor.py`.

Tools: `tests/bench_moe_lowm.cu` (low-row expert launch timing), `tests/test_head_rows.cu` (head kernel bit comparison).

## v0.1.0 (first public release)

Eke Hive runs two mixture-of-experts models on one 96 GB Blackwell GPU, with the routed experts in host RAM:

- **DeepSeek-V4.1-Flash** (original MXFP4/FP8 checkpoint, text and images; optionally the Engram tables on NVMe) — the main model;
- **GLM-5.3-Flash** (NVIDIA NVFP4 checkpoint plus Z.ai's FP8 attention and shared-expert tensors; text, images and video).

One daemon serves one of the two at a time; switching restarts it with the other configuration (`config/hive.env`,
`config/glm.env`). HTTP API (OpenAI chat and Anthropic messages, streaming, tool calls, images, reasoning effort and budget),
sleep/wake, prompt-state reuse across turns, 262,144-token context on both.

Measured on the reference machine with the real-chat benchmark (`tools/bench_chat.py`; [docs/benchmarks.md](docs/benchmarks.md)):

| | DeepSeek-V4.1-Flash | GLM-5.3-Flash |
| --- | --- | --- |
| Decode, 1 / 4 / 8 streams (total) | 76.9 / 124.0 / 173.0 tok/s | 65.2 / 89.3 / 88.8 tok/s |
| Time to first token, 17K / 42K / 54K prompt | 4.9 / 7.5 / 9.6 s | 7.2 / 11.6 / 13.7 s |
| Quality suite (bundled) | 175 / 179 | 158 / 163 |
| Sleep / wake (level 3, a 120K conversation open) | 0.7 s / 3.2 s | 3.8 s / 3.0 s |
| Cold start to ready | 88 s | 95 s |

What is in this release (the DeepSeek optimisation steps are numbered in [docs/performance.md](docs/performance.md); the GLM
path is described in [docs/glm.md](docs/glm.md)):

- Expert placement: VRAM cache of the hottest experts with score-based promotion, CPU experts computed in place,
  PCIe streaming for prefill, elastic cache slots lent to long prefills.
- Routing that knows the cache: cache-aware routing (`HIVE_CACHE_PRIOR`) and expert deferral on both models.
- Decode path: fused kernels, early routing, speculative decoding with each checkpoint's own draft layer (lossless),
  batched decode (8 streams on DeepSeek, 4 on GLM).
- Prefill path: chunked and batched prefill, prompt checkpoints, prefix sharing with boundary snapshots, layer
  yields so that short requests and running decoders are not blocked by a long prefill.
- GLM: KV memory and expert-cache slots that grow with use up to the 262K context, and the checkpoint's vision encoder on the GPU
  for images and video.
- Operations: `/flush_cache`, `/release_memory_occupation` and `/resume_memory_occupation` (sleep/wake), optional API
  key and request-size limit, a log monitor and a daily report, the benchmark drivers, a CPU-only test suite that runs
  without the GPU, and a LiteLLM guide ([docs/litellm.md](docs/litellm.md)).

Known limits are listed in [docs/limitations.md](docs/limitations.md); other models and other GPU families are not supported.
