# Eke Hive

**A from-scratch inference engine for four popular open models — Qwen3.8-27B, Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash and
GLM-5.3-Flash — that runs them on whatever GPU memory you give it, with the CPU and RAM computing the part that does not fit.**

> *eke out* — to make a small supply go a long way. In beekeeping, an *eke* is the extra ring added to a hive
> when it runs out of room. Eke Hive does both: it adds RAM and SSD tiers under a single GPU and squeezes the
> most out of each.

## Concept

1. **GPU + CPU + RAM, computing together.** A model that does not fit in VRAM still runs: the CPU computes the layers and experts
   that live in RAM instead of only shipping weights over PCIe.
2. **You choose the VRAM size.** Each model takes a VRAM budget (`HIVE_VRAM_GB`: 96 / 72 / 48 / 32 GB measured), so one GPU can run
   two or more models at once.
3. **No compromise with the whole model in VRAM.** The bar: with enough VRAM, each model matches or beats the best dedicated
   engine for it (Qwen3.8-Flash-Next: SGLang · Qwen3.8-27B: NInfer) — see the benchmark docs for where each axis stands.
4. **Eke out every byte.** Down to 3- and 2-bit quantized builds, each configuration is tuned for the best speed it can reach.
5. **Four models, not all models.** Eke Hive supports the most popular and useful open models only — Qwen3.8-27B,
   Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash and GLM-5.3-Flash — and follows new versions of these families. Beside them, a
   real-time voice set that shares the card: Gemma 4 12B (chat), Qwen3-ASR 1.7B (speech recognition) and Qwen3-TTS 1.7B
   (speech synthesis).
6. **We pick and optimize; we do not fine-tune.** Model quality scores are the model authors'; we run a quality suite only to
   prove a quantization or optimization did not regress. Performance is measured the same way, with one harness, at every
   VRAM size (96 / 72 / 48 / 32 GB on sm_120 cards).

This is not a general-purpose engine. It is built for exactly these four model families and one kind of machine, and it was
pieced together from what we had: one RTX PRO 6000 Blackwell (96 GB), an older 32-core Threadripper PRO 3975WX (Zen 2),
1 TB of 8-channel **DDR4-2667** and **PCIe 4.0**. No datacenter, no NVLink, no second GPU. Every number below comes from that
box — on DDR5 and PCIe 5.0 the CPU-side and transfer-side work would get faster (see the hardware note).

How the work is split:

- dense weights, attention state and a cache of the hottest experts live in **VRAM** (the cache fills whatever is left of the budget);
- all routed experts (and DeepSeek's Engram tables) live in pinned **RAM**, split across NUMA nodes;
- experts that miss the VRAM cache are computed by the **CPU** in place, or copied over PCIe to the GPU;
- optionally, DeepSeek's ~195 GB Engram value tables stay on **NVMe** and are read row by row.

## Performance by VRAM size

One GPU (RTX PRO 6000 Blackwell 96 GB; smaller cards reproduced with the engine's VRAM limit `HIVE_VRAM_GB`), 600 W,
context 262,144, vision on, Threadripper PRO 3975WX + 8-channel DDR4-2667 + PCIe 4.0. Benchmark: `tools/bench_chat.py`
(12 real chat prompts, answers up to 600 tokens, temperature 0, thinking off) after one discarded warm pass
(1 / 4 / 8 streams), then 1 / 4 / 8 / 16 streams and one 8K / 32K / 64K-token prompt.

- **VRAM** = the budget (GB actually used) · **RAM** = host RAM of the daemon (GB)
- **c1** = decode tok/s, one stream · **TTFT** = time to first token, one stream (s)
- **c4 / c8 / c16** = total / per-stream decode tok/s (DeepSeek and GLM run 8 at a time; c16 queues)
- **8K / 32K / 64K** = time to first token for that prompt length (s)

| Model | VRAM | RAM | c1 | TTFT | c4 | c8 | c16 | 8K / 32K / 64K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | 96 (92) | 479 | 85.3 | 0.22 | 123.4 / 33.6 | 175.3 / 23.8 | 176.2 / 23.6 | 3.8 / 6.4 / 11.7 |
| | 72 (69) | 479 | 78.7 | 0.26 | 108.8 / 29.3 | 149.2 / 19.7 | 153.8 / 19.7 | 4.1 / 6.9 / 12.0 |
| | 48 (45) | 479 | 67.3 | 0.31 | 87.7 / 22.5 | 105.5 / 14.3 | 111.2 / 14.3 | 4.4 / 7.2 / 12.8 |
| | 32 (29) | 479 | 56.6 | 0.33 | 64.8 / 16.6 | 72.5 / 9.7 | 76.0 / 9.5 | 4.7 / 7.6 / 13.5 |
| GLM-5.3-Flash | 96 (92) | 366 | 71.8 | 0.47 | 103.7 / 29.8 | 130.0 / 17.3 | 133.4 / 17.5 | 5.4 / 10.1 / 16.2 |
| | 72 (70) | 360 | 64.5 | 0.54 | 86.8 / 23.7 | 117.7 / 16.2 | 116.1 / 16.2 | 6.2 / 9.8 / 16.3 |
| | 48 (46) | 360 | 53.3 | 0.64 | 67.1 / 17.9 | 77.8 / 10.4 | 80.5 / 10.6 | 6.9 / 10.3 / 16.4 |
| | 32 (30) | 358 | 34.9 | 0.75 | 46.2 / 12.2 | 50.9 / 6.6 | 52.7 / 6.8 | 7.4 / 10.5 / 22.4 |
| Qwen3.8-Flash-Next | 96 (92) | 199 | 220.7 | 0.04 | 436.5 / 127.7 | 639.9 / 90.4 | 806.4 / 58.2 | 0.83 / 3.47 / 7.20 |
| | 72 (69) | 199 | 163.6 | 0.13 | 254.4 / 72.7 | 378.6 / 51.3 | 457.0 / 34.8 | 2.8 / 4.9 / 9.6 |
| | 48 (45) | 199 | 132.7 | 0.13 | 192.2 / 50.8 | 246.9 / 32.3 | 298.3 / 21.5 | 3.3 / 5.5 / 11.1 |
| | 32 (29) | 199 | 87.8 | 0.17 | 115.6 / 30.3 | 142.7 / 19.1 | 154.6 / 10.9 | 3.7 / 7.1 / 19.2 |
| Qwen3.8-27B | 96 (55) | 17 | 190.7 | 0.02 | 508.5 / 140.2 | 803.9 / 125.3 | 900.8 / 64.3 | 1.0 / 4.2 / 11.1 |
| | 72 (47) | 17 | 190.5 | 0.02 | 507.9 / 137.7 | 802.1 / 125.0 | 888.9 / 63.5 | 0.97 / 4.2 / 11.1 |
| | 48 (45) | 17 | 190.0 | 0.02 | 510.2 / 138.2 | 800.0 / 124.2 | 881.6 / 61.8 | 0.97 / 4.2 / 11.1 |
| | 32 (32) | 27 | 158.8 | 0.03 | 444.2 / 122.5 | 745.2 / 112.1 | 825.6 / 59.6 | 0.99 / 4.6 / 11.8 |

DeepSeek's 72 / 48 / 32 GB rows were measured with a 1-stream warm pass (the other rows: 1 + 4 + 8 streams).

How speed falls with VRAM depends on the model. For the three MoE models the speed is set by how many of the needed experts are
in the VRAM cache; the misses are computed by the CPU from RAM.

- **DeepSeek-V4.1-Flash** falls gently: one stream 85.3 → 56.6 tok/s from 96 to 32 GB, long prompts within 2 s of the whole card.
- **GLM-5.3-Flash** keeps three quarters of its single-stream speed at 48 GB and the 64K prompt at ~16 s down to 48 GB: the profiles
  give the expert cache everything else, run each long prompt in one layer-major pass sized to the free cache tail, and lend the
  draft layer's 6.75 GiB of experts to the cache while several streams decode.
- **Qwen3.8-Flash-Next** keeps its decode step on the GPU at every size: at 72 GB the experts that are not resident are read over
  PCIe, below 60 % residency the CPU computes them inside the same CUDA graph — 132.7 tok/s at 48 GB and 87.8 at 32 GB for one stream.
- **Qwen3.8-27B** is dense: it runs at full speed while the whole model fits (96 / 72 / 48 GB are the same). At 32 GB every layer
  and the drafter still fit; decode reads the official FP8 weights instead of the 4-bit decode copies (158.8 vs ~190 tok/s), and KV
  beyond ~100K positions moves to pinned RAM — the full 262K context stays available and a 64K prompt starts in 11.8 s (11.1 s
  on the whole card).

### How these numbers were measured

Every number in the table above is **measured on the machine described here, not projected** from a drafter card or a
speed-up ratio. All cells use the same harness and the same settings, so they can be compared with each other.

- **Harness:** `tools/bench_chat.py` (in this repository). Twelve real chat prompts: 5 Korean, 3 English, 2 code,
  2 creative. Answers up to 600 tokens, `temperature 0`, thinking off (`reasoning_effort: none`, or the model's lowest
  effort where thinking cannot be turned off, as on GLM-5.3-Flash).
- **Order per cell:** one discarded warm pass (1, 4 and 8 streams), then 1 / 4 / 8 / 16 streams, then one prompt each of
  ~8K, ~32K and ~64K tokens (documentation text). A long prompt that would take more than 10 minutes is cut
  (`--long-timeout 600`).
- **Decode tok/s:** tokens after the first token divided by the time they took. c1 runs the twelve prompts one after
  another; cN starts N streams at once and reports total and per-stream (median) throughput, including queueing when
  N is above the model's batch limit (DeepSeek and GLM run 8 at a time).
- **TTFT:** time to first token as the client sees it, median over the requests.
- **Speculative decoding** (MTP / DFlash2) is on wherever the model ships a drafter, as in normal serving. Its speed-up
  depends heavily on the text: math and code drafts are accepted far more often than Korean or creative prose. That is
  why we measure on mixed real chat rather than on a math benchmark such as GSM8K, and why our numbers can be lower than
  figures quoted from drafter cards.
- **Conditions:** one RTX PRO 6000 Blackwell (96 GB) at 600 W; smaller cards are reproduced with the engine's own VRAM
  limit (`HIVE_VRAM_GB`), so the RAM, CPU and PCIe side is the same in every row. Context 262,144 tokens and vision on
  in every cell. Threadripper PRO 3975WX (Zen 2, 32 cores), 8 × 128 GB DDR4-2667, PCIe 4.0 x16.
- **Exclusive GPU:** each cell runs alone on the GPU. Running two models side by side (see
  [Reference: two models on one card](#reference-two-models-on-one-card)) shares GPU compute, PCIe and the CPU pool, so each
  will be slower while both are busy.
- **Builds:** the cells the latest changes do not touch (DeepSeek 72 / 48 / 32 GB, Qwen3.8-Flash-Next 96 GB, Qwen3.8-27B
  96 / 72 / 48 GB) were kept from the previous build, whose code path for them is the same; every other cell was measured on
  this source tree. The per-size settings are the files in `config/profiles/`, each with its measurement.

### DeepSeek-V4.1-Flash and GLM-5.3-Flash on the whole card, in detail

DeepSeek-V4.1-Flash is the main model; GLM-5.3-Flash is the second option — slower here, but a good model to have
on the same box. The details below are from an earlier build (2026-10-09; the table above is current), both measured with the same real-chat benchmark (`tools/bench_chat.py`: twelve different
chat prompts — 5 Korean, 3 English, 2 code, 2 creative — answers up to 600 tokens, thinking off, decode speed after
the first token; [docs/benchmarks.md](docs/benchmarks.md#real-chat-benchmark-headline)):

| | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [GLM-5.3-Flash](docs/glm.md) |
| --- | --- | --- |
| Checkpoint used | original (MXFP4 experts, FP8 dense, Engram) | NVIDIA NVFP4 + Z.ai's FP8 attention / shared experts |
| Inputs | text and images | text, images and video |
| Thinking off | yes (`reasoning_effort: none`) | no — the model always thinks; "off" maps to its lowest effort |
| Context | 262,144 | 262,144 |
| Decode, 1 stream | **80.7 tok/s** (Korean 87.1 · English 82.0 · code 78.2 · creative 62.0) | 68.1 tok/s (72.0 · 67.3 · 67.1 · 56.8) |
| Decode, 2 / 4 streams (total) | **106.4 / 125.0 tok/s** | 86.5 / 103.5 tok/s |
| Decode, 8 / 16 / 32 streams (total) | **172.8 / 171.6 / 176.9 tok/s** (8 at a time) | 129.0 / 127.4 / 128.9 tok/s (8 at a time) |
| Time to first token, 1 stream | **0.26 s** | 0.52 s |
| Time to first token, 17K / 42K / 54K prompt | **4.90 / 7.49 / 9.59 s** (17.7K / 44K / 57K tokens) | 7.2 / 11.3 / 13.7 s (16.9K / 42K / 55K tokens) |
| The same with the GPU capped at 300 W (`nvidia-smi -pl 300`): decode 1 / 4 / 8 streams · first token, 17K / 42K / 54K | 82.9 / 115.8 / 157.9 tok/s · 6.6 / 11.0 / 13.7 s | 69.4 / 102.5 / 129.3 tok/s · 8.2 / 15.3 / 18.9 s |
| Time to first token, 100K / 200K / 250K prompt | — | 26.4 / 53.7 / 66.3 s |
| Quality suite (bundled) | 175 / 179 | 158 / 163 (thinking at its lowest level) |
| Sleep / wake (level 3, a 120K conversation open, GPU memory back to ~1 GB) | 0.7 s / 3.2 s | 3.8 s / 3.0 s |
| Cold start to ready | 88 s | 95–97 s |
| Host RAM in use | ~480 GB (RAM mode) · ~300 GB (Engram on SSD) | ~385 GB (358 GiB) |

### Why DeepSeek runs faster than GLM here

Both models get the same treatment, but on this machine DeepSeek comes out ahead, for measured reasons:

- **Maturity.** The DeepSeek path went through about twenty measured optimisation steps (fused decode kernels, early routing,
  CUDA graphs, a CPU/PCIe split for cache misses, tuned batching — [docs/performance.md](docs/performance.md)). The GLM path is new.
- **Where the time goes.** Only about 39 % of GLM's experts fit in VRAM. Before cache-aware routing about half of a GLM decode
  step was the CPU computing the missed experts (at 77–91 GB/s, bound by this DDR4); with it the CPU share is
  under a quarter, and what remains is mostly GPU work outside the experts — 34 linear-attention (KDA) layers with BF16
  projections, and the hyper-connection mixing ([docs/glm.md](docs/glm.md)).
- **PCIe is already busy.** GLM's cache promotions and next-layer prefetches already send expert records (13.5 MiB each)
  over PCIe 4.0 on every decode step, so its misses are not also streamed to the GPU the way DeepSeek's are.
- **The same two routing changes paid off more on DeepSeek.** Cache-aware routing and expert deferral
  ([docs/configuration.md](docs/configuration.md)) took DeepSeek from 53.8 to 76.9 tok/s for one stream and from 100.2 to
  173.0 for eight (the same benchmark, before the later decode-kernel step).
- **Speculative decoding helps one or two streams.** GLM's draft layer gives 2.5 tokens per step for a single user, and with two
  users both drafts are verified in one step (`HIVE_MTP_BATCH`); with more users at once each step is plain decoding bound by the CPU.
- **Thinking cannot be switched off** on GLM-5.3-Flash, so a "no thinking" request still produces some reasoning tokens.

If you can only run one, run DeepSeek. GLM is there because it is a good model and the engine can serve it.

### Sharing the GPU: keep the model in RAM, hand the card over

The weights never leave host RAM, so the GPU can be lent out and taken back in seconds. `HIVE_SLEEP_LEVEL=3 scripts/hive-sleep.sh sleep`
drains running requests, moves the dense weights, sessions and work buffers to RAM and frees the expert cache — about 1 GB
of VRAM stays in use (the script's default, level 1, frees only the expert cache). `wake` brings it back and refills the expert cache in ~3 s; open conversations
continue where they were (measured with a 120K-token conversation open: the next turn after the wake answered correctly
with its first token after 0.6 s on DeepSeek and 2.6 s on GLM, the whole history reused). We use this every day to alternate between one of the two big models and a **media stack plus a
smaller model (for example a 27B model next to image and video generation)** on the same card: put Eke Hive to sleep,
run the media job, wake it up — no reload from disk (a cold start takes 60–100 s). Details: [docs/install.md](docs/install.md#sleep-and-wake-sharing-the-gpu).

> **Hardware note.** Everything here was measured on 8-channel DDR4 (Zen 2) with PCIe 4.0 — what we had, not what we would
> pick. Experts that miss the VRAM cache are read from host RAM and prefill streams experts over PCIe, so a DDR5 / PCIe 5.0
> platform should do better — an estimate, not a measurement: roughly +30–45 % multi-stream decode, ~2× faster long-prompt
> prefill, and about +10 % single-stream decode for DeepSeek (bound mostly by per-layer GPU work). GLM still spends about a
> quarter of each decode step reading experts from RAM, so it should gain more from faster memory than DeepSeek does.

How each of these numbers was reached — including the ideas that were measured and dropped — is in
[docs/performance.md](docs/performance.md); the benchmark itself is described in [docs/benchmarks.md](docs/benchmarks.md).

## Use case: one 96 GB card for chat, images, video and music

Eke Hive exists to get the most out of a single RTX PRO 6000 (96 GB). Our own box runs the same idea one level up: the GPU
serves one chat model at full speed, while the image, video and music engines — and a second LLM for media prompts — wait
in host RAM and swap onto the GPU only when a request needs them. Nothing is cold-started for a media request.

**What stays where**

| Component | Engine | While idle | How it wakes |
|---|---|---|---|
| Main chat model (DeepSeek-V4.1-Flash, GLM-5.3-Flash or Qwen3.8-Flash-Next) | Eke Hive | on the GPU, serving | — |
| Media-mode LLM (Qwen3.8-27B) | Eke Hive | asleep in RAM (hive sleep level 3: weights and sessions kept in pinned RAM, ~0.8 GB VRAM) | hive wake, ~1.3 s |
| Image (Qwen-Image 2.1), video (MiniMax H3, 4 engines), music (YuE2) | SGLang Diffusion / own worker | GPU state saved to RAM with NVIDIA `cuda-checkpoint`, container paused (`docker pause`) | `cuda-checkpoint` restore + unpause, ~1.4 s for the image engine |

**A media request, step by step** (measured on this box, one image request)

| Step | Time |
|---|---|
| Main chat model goes to sleep (hive level 3, VRAM released, sessions kept) | 0.97 s |
| Image engine restored from RAM | 1.4 s |
| Media-mode 27B woken from RAM | 1.3 s |
| **Swap in, total** (steps overlap) | **2.4 s** |
| Image generation (first after load) | 65.7 s |
| 27B back to sleep, main model woken | 3.3 s + 3.2 s |
| **Swap out, total** | **7.2 s** |

After the swap the main model comes back with its expert cache intact (4,396 of 4,400 slots) and answers immediately.

**What makes it work**
- **Eke Hive sleep / wake.** The engine can hand its VRAM back while keeping weights, KV sessions and caches in pinned
  host RAM, and take the GPU again in about a second. This is an engine feature, not a restart.
- **`cuda-checkpoint`** (NVIDIA's tool shipped with the driver) for engines that have no sleep mode of their own: it moves
  a running process's whole GPU state to host memory and back.
- **A small orchestration script** that runs the steps in order: wait until no request is in flight, put the main model
  to sleep, wake what the request needs, then reverse. It also checks the RAM budget before keeping everything resident.

**The RAM budget it needs** (1 TB installed; measured)

| Item | Host RAM |
|---|---|
| DeepSeek-V4.1-Flash, awake | 479 GB |
| Qwen3.8-27B, asleep | 26 GB |
| Video engines (4) | 252 GB |
| Image engine | 45 GB |
| Music engine | 14 GB |

With GLM-5.3-Flash or Qwen3.8-Flash-Next as the main model the budget is roughly 90 GB or 280 GB smaller. The idea
scales down: on a box with less RAM, keep only the engines you use most resident and cold-start the rest.

## Reference: two models on one card

GLM-5.3-Flash with a 48 GB budget and Qwen3.8-27B with a 32 GB budget, both running at the same time on the one
RTX PRO 6000 (96 GB). Each daemon sizes itself at start exactly as it does alone; after start they share the card's free
memory (`HIVE_VRAM_BALLAST_RELEASE=1` frees the VRAM-limit ballast just before ready and leaves growth after start uncapped, `HIVE_START_WAIT_OTHERS=0` lets the second
start beside the first — [docs/configuration.md](docs/configuration.md)). Same harness as above (`tools/bench_chat.py`, warm pass discarded); total tok/s, long prompts as time to first
token.

| | GLM c1 / c4 / c8 | GLM 8K / 32K | 27B c1 / c4 / c8 | 27B 8K / 32K |
| --- | --- | --- | --- | --- |
| Alone on the card (table above) | 53.3 / 67.1 / 77.8 | 6.9 / 10.3 s | 158.8 / 444.2 / 745.2 | 0.99 / 4.6 s |
| Both loaded, the other idle | 57.4 / 67.6 / 78.1 | 7.2 / 10.7 s | 158.7 / 449.1 / 752.5 | 0.97 / 5.0 s |
| Both loaded, the other busy (steady 8 streams) | 19.2 / 23.3 / 37.2 | 20.7 / 28.3 s | 118.0 / 332.5 / 587.0 | 1.4 / 7.1 s |

- **Memory:** 75.6 GB on the card after start, at most 88.0 GB under load (GLM 54.8 + 27B 32.8 GB); host RAM about
  366 GB (GLM) + 7–22 GB (27B). No out-of-memory events.
- **Loaded but idle costs nothing:** each model runs at its solo speed while the other only waits.
- **Busy together, they share the GPU and the CPU:** the card sits at 100 % and the 600 W limit, and both daemons use the
  same 32 CPU cores. The 27B, which runs fully on the GPU, keeps about 75–80 % of its speed; GLM, which computes its
  missing experts on the CPU, drops to a third to a half.

## Reference: 1M-token context (GLM-5.3-Flash, DeepSeek-V4.1-Flash)

Both checkpoints declare 1,048,576 positions (DeepSeek-V4.1-Flash through its built-in YaRN ×16 over 65,536 original
positions, GLM-5.3-Flash natively), so a 1M context needs no model change — only `HIVE_MAX_CTX=1048576`. Same machine and
600 W as the tables above; the prompts are real documentation text (no repeated filler), one request at a time.

**How the KV memory is found**

- **DeepSeek-V4.1-Flash** reserves each session's KV in VRAM at start. At 1M the session pool grows from 8 × 252 MiB to
  8 × 942 MiB, so the expert cache starts smaller: 4,447 → 4,118 slots at 96 GB, 1,741 → 1,411 slots at 48 GB.
- **GLM-5.3-Flash** maps KV as a conversation grows (about 19 KB per position: 11 DSA layers plus the MTP layer) and
  takes the room from the tail of the expert cache. When VRAM runs short it first moves the least recently used idle
  conversation to host RAM, then lends cache slots ([glm.md](docs/glm.md#context-and-kv-memory)). A 1M conversation needs
  about 20 GB, which at 48 GB is nearly the whole expert cache.

**Short chat at 1M** (bench_chat c1 / c4 / c8 total tok/s; the official 262,144-token numbers in brackets)

| Model | VRAM | c1 | c4 | c8 | Start | Host RAM |
| --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash | 96 GB | 70.7 (71.8) | 103.7 (103.7) | 133.0 (130.0) | 107 s | 405 GB |
| GLM-5.3-Flash | 48 GB | 52.9 (53.3) | 66.4 (67.1) | 78.8 (77.8) | 112 s | 399 GB |
| DeepSeek-V4.1-Flash | 96 GB | 85.3 (85.3) | 121.1 (123.4) | 172.0 (175.3) | 89 s | 492 GB |
| DeepSeek-V4.1-Flash | 48 GB | 64.3 (67.3) | 81.2 (87.7) | 93.4 (105.5) | 91 s | 492 GB |

A 1M limit costs short chats nothing on GLM. On DeepSeek it costs 0–2 % at 96 GB and 4–11 % at 48 GB, where the reserved KV takes a larger share of a smaller expert cache. Host RAM is the peak of the daemon process during the run.

**Long prompts** (time to first token, then decode tok/s right after it)

| Prompt | GLM 96 GB | DeepSeek 96 GB | GLM 48 GB | DeepSeek 48 GB |
| --- | --- | --- | --- | --- |
| 128K | 31.4 s · 47.4 | 27.7 s · 55.1 | 32.0 s · 28.8 | 34.1 s · 39.5 |
| 256K | 69.7 s · 40.6 | 76.1 s · 33.9 | 69.9 s · 27.5 | — |
| 512K | 144.8 s · 36.8 | 151.0 s · 45.1 | 152.9 s · 23.9 | — |
| ~1M (1,039,904 tokens) | 335.0 s · 10.0 | 469.5 s · 23.6 | 386.8 s · 8.1 | 459.1 s · 26.1 |

Prefill stays at 2,200–4,700 tokens/s all the way to 1M (repeat runs vary by about 10 %). Decode slows as the context
grows; GLM drops hardest near 1M because its KV then occupies most of the expert cache (about 2,200 of 4,323 slots left
at 96 GB, none at 48 GB), so more experts are computed on the CPU. DeepSeek keeps its reserved KV apart from the cache and
holds 23–26 tok/s. The GLM 48 GB column was measured with one long prompt per daemon start; with the idle-session reclaim
in this release the same sequence runs in one daemon at the same times (128K 34.0 s, 512K 152.7 s, 1M 386.2 s).

**The next turn of a long conversation** (time to first token for a short follow-up)

| Conversation | GLM 96 GB | DeepSeek 96 GB | GLM 48 GB | DeepSeek 48 GB |
| --- | --- | --- | --- | --- |
| 512K, session kept | 1.07 s | 2.29 s | 1.26 s | — |
| 512K, cache flushed | 146.8 s | 135.7 s | 146.5 s | — |
| 1M, session kept | 3.88 s | 4.11 s | 4.02 s | 4.06 s |
| 1M, cache flushed | 361.6 s | 461.0 s | 400.6 s | — |

With the session kept, only the new message is processed, so a 1M conversation answers its next turn in about 4 s instead
of 6–8 minutes — the context never has to be summarised to stay usable. Sessions live in the VRAM session pool and, when
it is full, in host RAM (`HIVE_HOST_SESSION_MB`, 96 GB in the shipped configs); the least recently used one is dropped
first, with no time-based expiry. A sleeping engine (level 3) keeps them; a restart does not.

**Needle retrieval** (a made-up name and a 6-digit code placed at a depth of the prompt; greedy; the model must return the code; not run for DeepSeek at 48 GB)

| Prompt · depth | GLM 96 GB | DeepSeek 96 GB | GLM 48 GB |
| --- | --- | --- | --- |
| 128K · 10 / 50 / 90 % | ✗ ✓ ✓ | ✓ ✓ ✓ | ✓ ✓ ✓ |
| 512K · 10 / 50 / 90 % | ✓ ✓ ✗ | ✓ ✓ ✓ | ✓ ✓ ✓ |
| 1M · 10 / 50 / 90 % | ✗ ✓ ✓ | ✓ ✓ ✓ | ✓ ✓ ✓ |
| Total | 6 / 9 | 9 / 9 | 9 / 9 |

At 512K and 1M the 50 % code sits in the first turn of a two-turn conversation and is asked as the follow-up question. DeepSeek found
every code. GLM missed three at 96 GB but none of the same positions at 48 GB; its output varies between runs, so read its
misses as model behaviour at this length rather than an effect of the VRAM size.

## Model sources

Eke Hive runs the models' published weights as they are — nothing is re-trained, distilled or re-quantized here. Download them
yourself; they are not part of this repository. Licenses as stated in each release's own license file or model card.

| Model | Maker (official page) | Weights loaded | Format | License |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | DeepSeek ([`deepseek-ai`](https://huggingface.co/deepseek-ai)) | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) — everything: experts, dense weights, Engram tables, MTP layer, vision encoder, tokenizer and chat encoding | original: MXFP4 routed experts, FP8 dense, Engram tables | MIT |
| GLM-5.3-Flash | Z.ai ([`zai-org`](https://huggingface.co/zai-org)) | [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (NVIDIA ModelOpt quantization of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)) + Z.ai's own FP8 tensors from `zai-org/GLM-5.3-Flash` (DSA attention projections, shared experts, MTP experts — `scripts/glm-fetch-fp8.py`) | NVFP4 routed experts, BF16 vision encoder · FP8 (Z.ai tensors) | NVFP4 release: MIT · Z.ai release: see its model card |
| Qwen3.8-Flash-Next | Qwen ([`Qwen`](https://huggingface.co/Qwen)) | [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (NVIDIA ModelOpt quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) | NVFP4 | NVIDIA Open Model License; the base model's Qwen Community License 1.0 (both named on the model card) |
| Qwen3.8-27B | Qwen ([`Qwen`](https://huggingface.co/Qwen)) | [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (base [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)) + the DFlash2 drafter [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | NVFP4 (compressed-tensors; FP8 `lm_head`) · drafter BF16 | Apache-2.0 (NVFP4 release, base model and drafter) |

| Gemma 4 12B (voice chat) | Google ([`google`](https://huggingface.co/google)) | `google/gemma-4-12B-it-qat-w4a16-ct` + the MTP assistant `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` (text and images; the audio projection is not loaded) | int4 group 32 (compressed-tensors, quantization-aware trained) · assistant BF16 | Apache-2.0 |
| Qwen3-ASR 1.7B | Qwen ([`Qwen`](https://huggingface.co/Qwen)) | `Qwen/Qwen3-ASR-1.7B` | BF16 | Apache-2.0 |
| Qwen3-TTS 12 Hz 1.7B CustomVoice | Qwen ([`Qwen`](https://huggingface.co/Qwen)) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (speech tokenizer included) | BF16 | Apache-2.0 |

## Requirements (short version)

- NVIDIA Blackwell GPU with `sm_120a` (RTX PRO 6000 Blackwell, RTX 50 series); 96 GB was used for development.
- x86-64 CPU with AVX2/FMA/F16C, ideally with many memory channels (CPU experts are memory-bandwidth bound).
- RAM: DeepSeek ~480 GB resident in RAM mode (512 GB installed is comfortable), ~300 GB in SSD mode (384 GB installed);
  GLM ~385 GB (358 GiB) resident after start, growing with open conversations; Qwen3.8-Flash-Next ~200 GB; Qwen3.8-27B 17–27 GB;
  plus headroom.
- Fast local NVMe for the checkpoint: ~510 GB (DeepSeek), ~200 GB (GLM: ~191 GB NVFP4 checkpoint + Z.ai FP8 tensors).
- Linux, Docker and the NVIDIA Container Toolkit.

Details and what is measured vs. estimated: [docs/requirements.md](docs/requirements.md).

## Quick start

```bash
git clone https://github.com/yepapa-nest/eke-hive.git && cd eke-hive
# 1. Get the checkpoint (~510 GB)
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /models/DeepSeek-V4.1-Flash
export HIVE_CKPT=/models/DeepSeek-V4.1-Flash
# 2. Build the image and the engine (no GPU needed for this step)
docker build -t eke-hive:latest -f docker/Dockerfile .
scripts/build.sh
# 3. One-time preparation (exports the Engram hash constants)
scripts/prepare.sh
# 4. Start (daemon + API server on :8430); ready when the log prints "[hived] ready in"
scripts/hive-start.sh
tail -f run/logs/hived.log
# 5. Use it
curl -s localhost:8430/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hive","messages":[{"role":"user","content":"Hello!"}]}'
```

For GLM-5.3-Flash (same image and build — `scripts/build.sh` builds both `hived` and `hived_glm`; no `prepare.sh`):

```bash
huggingface-cli download nvidia/GLM-5.3-Flash-NVFP4 --local-dir /models/GLM-5.3-Flash-NVFP4
python3 scripts/glm-fetch-fp8.py /models/GLM-5.3-Flash-zai-fp8 --mtp-experts   # Z.ai FP8 tensors (HIVE_GLM_FP8_DIR)
HIVE_CKPT=/models/GLM-5.3-Flash-NVFP4 scripts/hive-start.sh
```

The launcher sees a GLM checkpoint, starts `hived_glm` and uses `config/glm.env` (`HIVE_CONFIG=config/glm.env` also works);
see [docs/glm.md](docs/glm.md#quick-start).

Full instructions, smaller-RAM (SSD) mode and running as a service: [docs/install.md](docs/install.md).

## Reasoning levels and budget

DeepSeek-V4.1-Flash thinks before it answers. Two request fields control it:

- `reasoning_effort` (or `reasoning.effort`): `none` turns thinking off; `minimal` 25,
  `low` 50, `medium`/`high` 75 (the default, the model's documented setting), `xhigh`/`max` 100, or any number 1–100.
  The number is the model's own effort scale, not a token count.
- `max_thinking_tokens` (alias `thinking_token_budget`): a hard cap. When the reasoning reaches N tokens the daemon
  inserts a short exit phrase (`HIVE_THINK_EXIT_TEXT`) and closes the thinking block, and the model writes its
  answer from what it has. Uncapped by default; a cap trades answer quality on hard questions for a bounded response
  time, so pick it per use (chat vs. batch jobs).
  After a cut a model may write the rest of its reasoning in front of a tool call; with `HIVE_CUT_LEAK_STRIP=1` such
  text is left out of what the model sees on later requests, so it is not copied into the next answers.
- Both are per request; nothing is restarted. Thinking text is returned as `reasoning_content` (streamed as it is
  generated), separate from the visible answer.

Details: [docs/configuration.md](docs/configuration.md#http-request-fields). GLM-5.3-Flash always thinks and has three levels
(low, high, max — unset means high): [docs/glm.md](docs/glm.md#thinking).

## Documentation

| | |
| --- | --- |
| [docs/install.md](docs/install.md) | Install, build, start, stop, sleep/wake, SSD mode, monitoring |
| [docs/build.md](docs/build.md) | Building from source, tests and the CPU-only test suite |
| [docs/configuration.md](docs/configuration.md) | Every option and environment switch, with the measurement behind its default |
| [docs/litellm.md](docs/litellm.md) | Running Eke Hive behind LiteLLM (or another OpenAI-compatible gateway): config, thinking fields, timeouts, sleep |
| [docs/architecture.md](docs/architecture.md) | How the engine is put together |
| [docs/performance.md](docs/performance.md) | The optimisation history of both models, step by step, including what was measured and dropped |
| [docs/benchmarks.md](docs/benchmarks.md) | The real-chat benchmark behind the headline numbers, both models |
| [docs/validation.md](docs/validation.md) | How correctness is checked (oracle, kernel bit-exact tests, quality suite) |
| [docs/limitations.md](docs/limitations.md) | Known limitations |
| [docs/glm.md](docs/glm.md) | The GLM-5.3-Flash family: setup, correctness, performance and its switches |
| [docs/qwen.md](docs/qwen.md) | The Qwen3.8-Flash-Next family: model layout, routing with a partial expert cache, PLE in RAM or on the SSD, its switches |
| [docs/q27.md](docs/q27.md) | The Qwen3.8-27B family: CPU and GPU daemons, DFlash2 speculation, placement and the KV tier under a VRAM budget, its switches |
| [docs/vram-footprint.md](docs/vram-footprint.md) | Where the VRAM goes per family and card size |
| [docs/gemma.md](docs/gemma.md) | Gemma 4 12B for voice chat: int4 GEMV, paged fp8 KV with sliding-window release, Frozen-KV MTP, measurements |
| [docs/qasr.md](docs/qasr.md) · [validation](docs/qasr-validation-20261018.md) | Qwen3-ASR 1.7B: `/generate` server, engine library, accuracy and latency against the reference server |
| [docs/qtts.md](docs/qtts.md) · [validation](docs/qtts-validation-20261018.md) | Qwen3-TTS 1.7B: streaming `/ws/speak` server, engine library, first-chunk latency and quality |

## Status

Eke Hive is a dedicated engine for four models (DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8-27B) and a voice set (Gemma 4 12B,
Qwen3-ASR, Qwen3-TTS) on one machine type that we own, put
together from the parts we had. It is shared as-is: it works for us every day, and the documentation says exactly what was
and was not measured. Other models and other GPUs are not planned; faster memory and PCIe would help most.

## License

MIT — see [LICENSE](LICENSE). Third-party material (DeepSeek reference code, GLM-5.3-Flash, nlohmann/json, the llama.cpp
nvfp4 dot-product technique) and its licenses are listed in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Model weights are not part
of this repository and are distributed by their publishers (DeepSeek, Z.ai, NVIDIA, Qwen, Unsloth, Z Lab) under their own licenses —
see [Model sources](#model-sources).
