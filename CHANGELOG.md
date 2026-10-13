# Changelog

## Unreleased

Four models at four VRAM sizes. Every family is measured with the same harness at 96 / 72 / 48 / 32 GB (`HIVE_MODEL`, `HIVE_VRAM_GB`;
smaller cards reproduced with the engine's VRAM limit) — the table is in the [README](README.md#performance-by-vram-size).

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
