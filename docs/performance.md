# Performance and how we got there

This page records how Eke Hive went from a correct-but-slow first version to its current speed on both models — DeepSeek-V4.1-Flash
first, then GLM-5.3-Flash — one measured step at a time. Every number here was measured on the reference machine below; each table
names the harness it used, and nothing is extrapolated. Steps that were tried and **rejected** are listed too, because they explain
the shape of the final design as much as the accepted ones.

The headline numbers (real chat prompts, both models) are in [benchmarks.md](benchmarks.md).

## Reference machine

| Part | Value |
| --- | --- |
| GPU | 1 × NVIDIA RTX PRO 6000 Blackwell, 96 GB, power limit 600 W, PCIe 4.0 x16 |
| CPU | AMD Threadripper PRO 3975WX (32 cores, Zen 2, AVX2, no AVX-512) |
| RAM | 1 TB DDR4, 8 channels, 2 NUMA nodes (measured ~61 GB/s per node) |
| Storage | weights on one PCIe 4.0 x4 NVMe (measured 7.1–7.3 GB/s with parallel O_DIRECT reads) |
| PCIe | measured 27.9 GB/s host → GPU |
| Models | DeepSeek-V4.1-Flash, original MXFP4 checkpoint (`config/hive.env`) · GLM-5.3-Flash, NVIDIA NVFP4 checkpoint + Z.ai FP8 tensors (`config/glm.env`) |

## How it is measured

- **Headline and the latest steps of both models:** the real-chat benchmark `tools/bench_chat.py`
  ([benchmarks.md](benchmarks.md#real-chat-benchmark-headline)).
- **DeepSeek steps 1–20:** the development benchmark `tools/bench_tune.py` (historic — not re-run after step 21). Restart the
  server for every configuration, warm up (3 requests × 300 tokens), check 3 known answers; decode with 1 stream ×2, 4 and 8
  concurrent streams on fixed story prompts, 256 tokens each, throughput summed over streams; prefill: time to first token for
  prompts of 1K, 4K, 15K, 42K and 80K tokens (100K with `BENCH_PF100K=1`) cut from a frozen text snapshot of this repository,
  plus decode speed (192 tokens) after the 80K prompt. Steps 13–20 name their own scenario (`tools/bench_turns.py` for the
  chat shapes with several users).
- **GLM steps:** first the engine alone (`glm_check`, greedy, one request), then the daemon with A/B runs of 12 chat requests
  per configuration, interleaved, restarted per configuration.
- Noise floor from repeated identical runs: c1 ±1 %, c4 ±3 %, c8 ±1–2 %, prefill ±2 %. Changes inside the
  noise floor were not adopted.
- Correctness gates for every adopted change: kernel tests that compare against the previous kernel bit for
  bit (or within the documented tolerance), layer-by-layer comparison with the reference model (DeepSeek's "oracle", the
  transformers implementation for GLM), and the self-contained quality suite (long context, Korean/English QA, code, tool
  calls, operations — 179 items on DeepSeek, 163 on GLM; see [validation.md](validation.md#quality-suite)).

## DeepSeek-V4.1-Flash

### Result

Headline (the real-chat benchmark, `tools/bench_chat.py` — [benchmarks.md](benchmarks.md#real-chat-benchmark-headline)),
before and after step 21, and with steps 22–23 (two runs; the step-21 column is one run — one-stream runs of the step-21 build
measured 82.2–84.9 tok/s on 2026-10-05, so the step-22 gain is the interleaved 83.9 → 86.0 of its row, not the difference of the columns):

| Streams | Steps 1–20 | Step 21 | Steps 22–23 (current) |
| --- | --- | --- | --- |
| 1 | 53.8 tok/s | 76.9 tok/s | **85.3 tok/s** |
| 2 / 4 | 68.4 / 81.1 | 92.6 / 124.0 | **92.9 / 124.6** |
| 8 / 16 / 32 | 100.2 / 98.4 / 105.1 | 173.0 / 172.7 / 174.3 | **175.0 / 178.1 / 181.5** |
| Decode after a 17K / 42K / 54K prompt | 36.9 / 43.8 / 44.5 | 64.1 / 72.6 / 77.4 | 63.4 / 79.1 / 75.8 |
| Time to first token, 17K / 42K / 54K | 4.94 / 7.38 / 9.63 s | 4.95 / 7.48 / 9.63 s | 5.05 / 7.53 / 9.69 s |
| Quality suite | 172 / 179 | 175 / 179 | (steps 22–23 do not change outputs) |

The development history below uses the development benchmark `tools/bench_tune.py` ([How it is measured](#how-it-is-measured);
it reads lower than the chat benchmark and was not re-run after step 21):

| Metric | First working version | Steps 1–20 | Change |
| --- | --- | --- | --- |
| Decode, 1 stream (tok/s) | 29.6 | 50–60 (two runs) | ×1.9 |
| Decode, 4 streams total (tok/s) | 55.3 | 108–111 | ×2.0 |
| Decode, 8 streams total (tok/s) | 71.1 | 136–141 | ×2.0 |
| Prefill 15K (s) | 16.4 | 4.83 | ×3.4 |
| Prefill 42K (s) | 37.4 | 7.43 | ×5.0 |
| Prefill 80K (s) | 69.7 | 13.98 | ×5.0 |
| Prefill 100K (s) | — | 16.39 | |
| Prefill 1K / 4K (s) | 7.31 / 4.19 (first measured at step 9) | 2.58 / 3.60 | |
| Decode after an 80K prompt (tok/s) | 36.5 (first measured at step 9) | 42–48 (lower when the post-prefill refill of step 19 rides on those steps) | |
| Follow-up turn in an 80K conversation, 1000 new tokens (s) | 3.00 | 1.98–2.01 | −34 % |
| Cold start to ready (s) | 484 | 88–90 | ×5.4 |
| Short request arriving during a long 85K prefill, first token (s) | 14.8 | 1.1–1.4 | |
| Longest stall of a running decode during that prefill (s) | 13.4 | 2.3–2.4 | |
| Release GPU memory ("sleep", first time) | 6.8 s | 0.6 s | |

Quality on the 179-question suite: 174/179 with every optimisation switched off, 177/179 after step 12 and
175/179 after step 20 (McNemar p = 1.0 against step 12 — the flips are questions that also flip between
two runs of the same build). Step 21 was measured against a re-run of the step-20 build, which gave 172/179 (run-to-run
spread; the long-context items gave identical outputs), and reached 175/179.

### Where the time goes (the model of the problem)

DeepSeek-V4.1-Flash is a large mixture-of-experts model: the routed experts alone are ~296 GB in the checkpoint
and the Engram tables another ~203 GB. Neither fits in 96 GB of VRAM, so Eke Hive splits the work:

- Dense weights, attention state and a cache of hot experts live in VRAM.
- All routed experts live in pinned host RAM, split by rows across the two NUMA nodes.
- An expert that is not in the VRAM cache (a "miss") is computed **in place by the CPU** from RAM, or a few of
  them are copied over PCIe and computed on the GPU. Both paths compute the same function.
- During prefill, every expert is needed for every chunk, so experts are streamed over PCIe (~27 GB/s) while
  the GPU computes; bigger chunks amortise the stream.

Three limits shape everything that follows: the GPU for attention and cached experts, **PCIe bandwidth** for
prefill, and **DRAM bandwidth** for CPU experts during decode (one expert record is 17.9 MiB (18.8 MB); the CPU spends
~0.185 ms per record, i.e. ~100 GB/s — most of the ~122 GB/s the two memory nodes measured).

### Steps, in order

Numbers are decode c1 / c4 / c8 tok/s and prefill 15K / 42K / 80K seconds after each step (steps 0–12, `tools/bench_tune.py`);
steps 13–20 give the metric named in the row, step 21 the real-chat benchmark.

| # | Step | What changed | After |
| --- | --- | --- | --- |
| 0 | First correct version | Bit-exact against the reference math; experts computed per token | 29.6 / 55.3 / 71.1 · 16.4 / 37.4 / 69.7 |
| 1 | Opt-in features + tuning | Prefill tiles, early streaming, async prompt checkpoints; stop promoting misses into the cache during prefill; 96 staging buffers | 28.9 / 59.0 / 75.7 · 6.67 / 14.05 / 24.69 |
| 2 | Grouped prefill GEMM, host tiles, batched prefill, prefill DMA share 0.7 | All cached experts of a layer in one or two launches (MoE layers ×2.8–3.0); several prompts prefilled together; 70 % of misses copied to GPU during prefill (42K −24 %) | 29.7 / 59.2 / 76.9 · 5.60 / 7.90 / 14.88 |
| 3 | One expert launch per decode layer, fused decode attention front, cache policy, paced promotions | Decode expert launches ×1.25–1.58, attention front ×1.12–1.94, promotion copies no longer collide with decode (c8 +5 %) | 41.6 / 79.8 / 96.8 |
| 4 | Multi-stage expert kernel (cp.async) | ×1.2–1.3 over step 3 kernels | 43.1 / 81.8 / 96.5 |
| 5 | Host work reordering per decode layer | c4 +6 % | 43.7 / 85.2 / 96.6 |
| 6 | Three prefill host tiles | 98,304 rows per pass with 2.7 GB VRAM headroom kept for vision (100K 19.65 → 17.77 s) | 43.2 / 80.5 / 96.5 · 100K 17.77 |
| 7 | Fused router / shared-expert / projection GEMVs; CPU fp4 GEMV v2 | Layer chain ×1.14–1.50; CPU kernel bit-identical and 6–9 W lower | 47.1 / 91.7 / 104.3 |
| 8 | Router, sparse attention, QKV and HC-mix kernels v3; CPU-first misses in batches | Layer chain ×1.05–1.14 | 48.0 / 91.1–92.6 / 111.0 |
| 9 | Speculative-decoding gate from measured costs; CPU multi-row K tiles; vectorised CPU unpack | MTP drafts only when the measured cost table says they pay | 49.2 / 96.5 / 117.7 |
| 10 | Elastic VRAM cache, deferred slot allocation, short-prefill path, fused kernels v3 | +638 cache slots reclaimed on demand in 3.9 ms; prompts < 1K no longer take the decode path (1K 6.7 → 3.1 s) | 52.2 / 108.3 / 131.6 · 4.93 / 7.18 / 13.99 |
| 11 | Isolation fixes, gate cost split, prefetch scaling, small-prefill threshold 256 | | 54.3 / 108.9 / 134.7 · 5.00 / 7.30 / 13.96 |
| 12 | New MTP verify path, early routing fix, short-layer DMA cap 32, gate v3 | Two-row verify 62.9 → 32.0 ms at 80K, so drafting pays at long context (decode after 80K +12.7 %) | **53.1 / 110.6 / 141.7 · 4.79 / 7.27 / 13.84** |
| 13 | Cold start: parallel O_DIRECT bulk load + prefault overlapped with dense load | Disk reads 1.0 → 7.3 GB/s; 484 → 88–90 s | |
| 14 | Layer-boundary yield during long prefills; coalesce simultaneous arrivals (25 ms) | Short request behind an 85K prefill 14.8 → 1.1–1.4 s; running decode stall 13.4 → 2.3 s | |
| 15 | Pre-pinned host pool for sleep | First sleep 6.8 → 0.6 s, byte-identical restore | |
| 16 | One prefix-reuse chunk instead of two (`HIVE_PREFIX_EXTRA_CHUNKS=0`) | Follow-up turns in long conversations −34 % time to first token, reproduced twice | 80K / 1000-token turn 3.00 → 1.98–2.01 s |
| 17 | Adaptive boundary snapshots (`HIVE_PREFIX_ADAPTIVE`), streamed tool calls, `/flush_cache` | A conversation whose tail is replaced every turn (fresh context before the newest message) found no saved state and prefilled everything again; now the server asks for one boundary chunk only on such turns. Tool-call arguments stream while generated instead of arriving in one piece at the end | 102K-token history with a replaced tail: turns 3+ prefill 17.1 → 0.30 s (102,477 of 102,504 tokens reused), the flagged turn +0.2 s, answers follow the newest context; append-only conversations unchanged (0.19–0.22 s). Tool call: first argument bytes at 2.3 s instead of 6.4 s |
| 18 | Yield admission by rows to prefill | A request waiting behind another prompt's long prefill is admitted at a layer yield when the rows it will actually prefill (ids minus its reusable prefix) fit, not when its whole prompt does — a follow-up turn of a 78K-token conversation has ~540 such rows | Follow-up turn sent 1 s after a 38.6K prompt (4 decoders running): 6.5 → 2.4–3.3 s; the 38.6K prompt's first token 5.6 → 6.7–7.2 s (it serves that turn inside its yields); decoders and the other turns unchanged (4 streams 17.0 tok/s, turns 2.3–3.3 s) |
| 19 | Post-prefill cache warm handed to the decode steps (`HIVE_WARM_DEFER`) | After every prefill the engine refilled the VRAM cache with the top-score experts in one synchronous upload (738–1,065 experts, 0.46–0.69 s each) — every running decoder stalled for it and the new request's first token waited behind it. Now the quota is handed to the following decode steps: each step's promotion runs with the warm's cap (N experts) and threshold until the quota is spent, on the side stream next to the step (the promotion pacing of step 3 kept) | Chat shape — 4 decoders + cached follow-up turns + a fresh 3.9K prompt (bench_turns, off ×3 vs 16/step ×2 vs 32/step ×2): decoder 17.0 → 19.0–19.2 → **21.5–21.6 tok/s**; follow-up turns 3.1–3.3 / 2.3 / 2.3 → 1.9–2.1 / 1.7–1.9 / 1.7 s; fresh 3.9K prompt 2.2 → 1.3–1.5 s; the decoders' longest gap 1.43–1.45 → 1.05–1.07 s (what remains is that turn's own prefill). Answers, long-prompt TTFT (6.7–7.0 s) and errors (0) unchanged. At 32/step the 2048-expert quota completes in 64–85 steps; at 16/step the next prefill resets it first; 64/step changes nothing more (21.5 tok/s, 32–69 steps) |
| 20 | Layer-boundary yields inside small prefills (`HIVE_LAYER_YIELD_SMALL`) | With step 19 in place, the longest stall the running decoders see is a follow-up turn's own short prefill (a few hundred rows, ~1 s, below the prefill threshold, so the step-14 yield never fired inside it); the yield now also fires at layer boundaries of those forwards | Same scenario as step 19, 32/step ×2 vs 32/step + yields ×2: the decoders' longest gap 1.05–1.07 → 0.59–0.60 s (p95 unchanged, 0.05–0.06 s); cost: decoder 21.5–21.6 → 21.2 tok/s, follow-up turns +0.05–0.1 s. Without step 19 it showed no gain — the warm stall dominated |
| 21 | Cache-aware routing (`HIVE_CACHE_PRIOR=0.1`) and expert deferral (`HIVE_DECODE_DEFER`) | Decode only. Each router's selection scores get λ × the layer's running score range for experts already in VRAM, with the top 2 by the original scores always kept and the routing weights left as the model's (idea: Skliar et al., TMLR 2025) — fewer misses reach the CPU. CPU misses ranked 3rd or lower in their row are no longer waited for: they run while the next layer starts and are added to the hidden streams one layer later (idea: KTransformers, SOSP 2025); ~1.5 rows per layer took that path. Both first proved on GLM-5.3 ([below](#glm-53-flash), [glm.md](glm.md#lossy-changes-measured-with-quality-on-top-of-each-other)) | Real-chat benchmark: c1 53.8 → 76.9, c4 81.1 → 124.0, c8 100.2 → 173.0 tok/s; decode after 17–54K prompts +66–74 %; prefill unchanged; quality 172 → 175 / 179 (McNemar p = 0.375; long-context items byte-identical). The first build hung on the first decode: the next layer's job list was rebuilt while the deferred batch still read it — fixed by adding the batch before the list is rebuilt |
| 22 | Decode kernels: expert activations a stage ahead, multi-row head (both bit-identical) | The fused expert kernel (FUSED3) loaded its activation operand only one 4-block step before use, so in launches with few work items (one stream, MTP verify rows) every step waited on an L2 round trip: one expert 53.0 → 37.8 µs, five experts 114.6 → 94.2 µs (`tests/bench_moe_lowm.cu`, L2 flushed); `test_decode_moe` M=1 811 → 912 GB/s. The head kernel read each vocabulary row once per input row (`HIVE_HEAD_ROWS`): 4 rows 1,169 → 911 µs, 8 rows 2,170 → 950 µs (`tests/test_head_rows.cu`) — it serves the verify rows and the MTP draft block | Real-chat benchmark, 7 base runs vs 6, interleaved: c1 83.9 → 86.0 tok/s (+2.5 %), c2 92.4 → 92.9, c4 123.6 → 124.2, c8 171.4 → 173.7; prefill unchanged |
| 23 | Mid-size prompts admitted at layer yields by remaining work (`HIVE_LAYER_YIELD_MID=16384`) | Service log (10-01..10-05): 114 requests of ≥ 4K rows waited > 2 s behind another prefill of ≥ 4K rows (e.g. three 22K-row requests 5.3 s behind a 20K one) — a layer yield admitted only ≤ 1,023-row requests. Now a request of up to 16,384 rows is admitted when it is at most half of the paused forward's remaining rows (its rows × (1 − progress) + its later chunks), at most that forward's rows per forward; its own forward yields once more for decode steps only (`ly::Hooks::max_depth` 2) | 85K prompt + 12K request 3 s later + a decoder (`tools/layer_yield_check.py`, two runs each): request 14.6 → 4.4 s to the first token, the long prompt 13.9 → 18.2 s, mean of the two 14.3 → 11.3 s; decoder's longest stall 1.83 → 2.05 s (5.13 s without the second level) |

### Rejected (measured, not adopted)

| Idea | Result |
| --- | --- |
| Prefetching predicted CPU experts during attention (`HIVE_DECODE_PREGATE`) | c4 −24 %, c8 −29 %. Precision 0.20 / recall 0.69 in real decode: wrong prefetches steal the same DRAM bandwidth the CPU experts need. A score threshold reduced the loss (−6 to −13 %) but "off" stayed fastest |
| Next-layer expert prediction for pre-copies (the measurement switch `HIVE_DECODE_PREDICT_EVAL` stays) | The prediction is good — layer l+1's router on layer l's FFN input names 72.5 % of the real top-6 and its first non-resident candidate is routed 63.1 % of the time (`HIVE_DECODE_PREDICT_EVAL`) — but copying it ahead loses: waiting for it each layer c1 −27 %, c4 −21 %; computing it in the front graph and copying only on an idle link c1 −10 % (the link was busy for nearly every candidate); retrying after the front, 18K copies of which 25 % arrived in time, c1 −15 %. A record takes ~0.67 ms on PCIe 4.0 and a decode layer ~0.45 ms, and demand DMA and promotions already fill the link. The same idea gave GLM-5.3 +2 % (its layers are CPU-bound and longer) |
| Skipping low-weight CPU misses (`HIVE_DECODE_SKIP_MISS=0.1`, on top of step 21) | Lossy (rank ≥ 3 misses below 10 % of the row's weight are dropped; ~0.08 rows per layer). Real-chat c1 +8 %, c2–c32 within ±1 %, decode after 17–54K prompts 4–12 % lower; quality 176 / 179 (p = 1 against step 21). No gain worth a lossy change — the same verdict as on GLM (`HIVE_GLM_SKIP_MISS`, +0 / +2 %) |
| Batched speculative decoding across sequences (`HIVE_MTP_BATCH`) | Lossless (byte-identical replicated-state test) but c2 +0 %, c4 up to −8 % on this machine; re-measured on the current verify path: c2 37–38 vs 37–41, c4/c8 unchanged — still no gain |
| Higher DMA share during decode (`HIVE_DMA_FRAC` 0.9) | Decode −28 to −41 % |
| Promoting misses into the cache during prefill | c4 −14 % (kept off) |
| GEMV v2 / router / PDL variants (decode) | Bit-identical but 0.5–0.97× (register preloads lowered occupancy) |
| Blocking per-layer synchronisation | −7 W but c4 −6 % |
| Whole decode step as one CUDA graph | Engine thread CPU ~0 and −11 to −17 W, but c1 −9 % |
| Adaptive prefill/decode work split models | Prefill +13–37 % slower / c4 −3 %, c8 −5 % |
| Sinkhorn fusion | Kernel ×5.5 but hidden behind another stream; fused version c4 −5 % |
| Request scheduling policies (window / SJF / fair) | Short request TTFT during a 30K prefill 3.6 → 6.3 s |
| Larger VRAM reservation (1.8 GB) | Image requests ran out of VRAM |
| Capped post-prefill warm while others decode (`HIVE_WARM_BUSY_CAP=1`) | Decoders' longest gap 1.44 → 1.04 s, but the cache went stale: 17.0 → 10.3 tok/s per decoder (the warm is the general refill after a prefill, not only the new request's experts) — replaced by step 19 |
| Tuning sweeps | CPU threads 8/12/24, MTP draft depth 2/3, 4 host tiles, promote 4/16, larger cache: no gain or loss |
| Lower promotion threshold of the `seq` cache policy (`HIVE_CACHE_POLICY=seq:min=0.0125` / `0.008`, on top of step 22) | Replay of two service traces predicted 4–13 % fewer decode miss jobs, but promotions per step went from 2–4 to 6–7 and the copies compete with demand DMA: real-chat c1 86.4 → 84.2 / 82.9 tok/s, c8 172.8 → 174.6 / 174.0 (two runs each) — kept at 0.025 |
| Commit paced promotions one step head later (opt-in `HIVE_PROMO_COMMIT_LAG=1`, prototype) | Host timestamps showed the verify head waiting 0.44 ms per cycle for promotion copies issued during the MTP draft (up to ~6 ms of copies against a 3.6 ms draft). Delaying the deterministic commit by one head: real-chat c1 85.1 → 85.9, c2 92.7 → 92.7, c4 124.9 → 123.4, c8 172.8 → 173.5 tok/s, decode after 54K tokens 91.1 → 86.6 (two interleaved runs each; c4 and long-context decode lower in both runs) — not adopted, code not kept |
| Larger per-step promotion budget after step 19 (`--promote 16/32`) | Cache replay of a real trace predicted +0.9 / +1.5 points decode hit rate, but on the service (interleaved ×2 each) the four-stream chat scenario gave 21.7–21.9 (8) vs 20.5–21.4 (16) vs 20.9–21.3 (32) tok/s and c1/c4/c8 stayed within noise — the extra promotion copies cost what the hits save; kept at 8 |

### The Engram tables on SSD

`HIVE_ENGRAM_SSD=1` keeps the ~195 GB Engram value tables on the NVMe and reads only the rows a token needs
(row addresses depend only on token ids, so they are prefetched ahead of use). Same machine, same build:

| | RAM mode | SSD mode |
| --- | --- | --- |
| Cold start | 88–90 s | 58.7–58.8 s (−34 %) |
| Daemon resident memory | 478 GB | 295 GB (−183 GB) |
| Decode c1 / c4 / c8 | baseline | −2.6 % / −1.3 % / −5.6 % |
| Prefill 15K / 80K | baseline | +6 % / +2.4 % |
| Output | | byte-identical Engram lookups (digest of every lookup matched) |

The disk sustained 1.12 M random-read IOPS at queue depth 128; a token needs 48 rows.

### Open problem

Measured on the steps 1–20 build (step 21 then took part of this away): at one stream the GPU waits 2.9 ms of a 19 ms step for CPU experts, and the CPU side is bound by DRAM reads,
not compute. Reading earlier does not help (see the rejected prefetch); the next gains have to come from reading
**less** — which step 21's cache-aware routing started: a higher VRAM cache hit rate (replay of real expert traces of the step-20 build: 85.9 % with the cache policy of
that build, 92.5 % with perfect knowledge of the future — before step 21's cache-aware routing), a better CPU/DMA split per layer, and fewer bytes per CPU expert.

On the GPU side the resident-expert kernel itself has headroom: `bench_gemv` (GPU idle) reads one MXFP4 expert
matrix at about 1.0 TB/s for M=1 even though the 5.9 MB matrix sits in L2, while the grouped six-expert path reaches 1.8–2.4 TB/s
and the VRAM peak is 1.79 TB/s — the single-expert GEMV is latency-bound, not bandwidth-bound.

## GLM-5.3-Flash

GLM-5.3-Flash came second and reuses the DeepSeek machinery that is not model-specific (the expert cache and its promotion,
the CPU expert pool, the daemon's scheduler, sessions, prefix sharing, the speculative-decoding gate, sleep and wake). Its own
parts are new: NVFP4 expert records, the KDA linear-attention and DSA layers, the NextN draft layer and the vision encoder. The
full record, with every A/B, is in [glm.md](glm.md#performance); this section is the summary.

### Result

Real-chat benchmark (`tools/bench_chat.py`, shipped `config/glm.env`; [benchmarks.md](benchmarks.md#real-chat-benchmark-headline)):
decode 70.3 / 85.2 / 97.9 / 129.1 / 128.3 / 129.6 tok/s with 1 / 2 / 4 / 8 / 16 / 32 streams (two runs; before steps 17–21
65.2 / 66.6 / 89.3 / 88.8 / 91.6 / 93.3), time to first token 7.3 / 11.8 / 13.9 / 26.4 / 53.7 / 66.3 s for 17K / 42K / 54K / 100K /
200K / 250K-token prompts, quality 158 / 163 (steps 17–21 do not change outputs).

### Where the time goes

- The routed experts are 12,096 records of 13.5 MiB (159.5 GiB in pinned RAM); about 4,690 of them (38.8 %) fit in the VRAM
  cache next to everything else.
- A missed expert costs 0.507 ms over PCIe 4.0, so in decode misses are computed by the CPU from RAM; the CPU phase reads at
  77–91 GB/s, against 118 GB/s for the same kernel in a node-local stream benchmark and a measured 128 GB/s read ceiling.
- Before cache-aware routing about half of a single-stream step was that CPU phase (a 3-row verify: 26.2 ms of 47.8 ms in the
  routed experts, 23.2 ms of it CPU); the rest is GPU work outside the experts — 34 KDA layers (~10 ms), DSA, the
  hyper-connection mixing, the shared expert, router and head ([glm.md](glm.md#where-a-c1-step-goes-hived-hive_glm_prof2-real-chat)).
- Prefill is bound by PCIe: every expert a prompt uses and the cache does not hold has to cross it.

### Steps, in order

The first steps were measured on the engine alone (`glm_check`: greedy decoding, 600 tokens, one request), the later ones through
the daemon with interleaved A/B runs of 12 chat requests per configuration; the two scales are not comparable with each other or
with the headline benchmark.

| # | Step | Measured |
| --- | --- | --- |
| 1 | First end-to-end run (CPU shared with another job) | engine only 14.6 tok/s |
| 2 | Tensor-core MoE decode kernel (1.45 TB/s) on a quiet machine | 21.6 |
| 3 | Spin-waiting CPU expert pool | 24.3 |
| 4 | NUMA half records (node-pinned workers read local memory only) | 25.8 |
| 5 | 24 CPU threads | 27.7 |
| 6 | Fused decode kernels, no host → GPU copies on the critical path | 33.5 |
| 7 | Next-layer expert prefetch, one expert per layer (`HIVE_GLM_PREFETCH=1`; 0 / 1 / 2 → 33.5 / 34.1 / 30.7) | 34.1 |
| 8 | Speculative decoding with the NextN layer, k = 3 (77–83 % of drafts accepted) | 50–53 |
| 9 | Prefill: 16K blocks, copies overlapped with compute, CPU share for experts used by ≤ 24 rows, layer-major over up to 64K rows in borrowed cache slots | 15K / 42K prompt 22.7 / 61.4 → 6.9 / 11.0 s |
| 10 | 32 CPU threads instead of 24 | daemon, c1 45.6 → 47.0 tok/s; CPU phase 79 → 86 GB/s |
| 11 | CPU-expert scratch reused instead of allocated per layer | CPU phase per 3-row verify 21–22 → 19–20 ms |
| 12 | KDA verify rows in one launch (`HIVE_GLM_KDA_ROWS`, bit-identical) | 3-row verify 44.0 / 44.5 → 43.1 / 43.4 ms |
| 13 | Cache-aware routing (`HIVE_CACHE_PRIOR=0.1`) | daemon, c1 47.8 → 61.3 tok/s, cache hits 86.7 → 96.3 %; quality 158 vs 160 / 158 for two runs without it |
| 14 | Expert deferral (`HIVE_GLM_DEFER`) | daemon, c1 60.6 → 63.6 tok/s; quality 157 / 157 |
| 15 | Prefix sharing across conversations (boundary snapshots including the KDA state) | same 8.6K system prompt in a new conversation: 5.8 → 0.52 s to the first token |
| 16 | 262,144-token context with KV memory that follows use (`GrowBuf`, CUDA VMM) | session pool 4 × 1,370 → 4 × 458 MiB, expert cache 4,420 → 4,693 slots; prompts up to 250K; quality 157 → 158 / 163 |
| 17 | Layer yields inside a prefill (`HIVE_LAYER_YIELD`, as on DeepSeek; nothing parked — inner forwards use rows the paused prefill does not hold) | 60K prompt + 64-row requests + a decoder: first short requests 14.0 / 12.5 → 0.65 / 0.59 s, decoder's longest stall 14.4 → 1.6 s, the long prompt 14.5 → 16.5 s |
| 18 | 8 sequences per step (`HIVE_MAX_BATCH=8`) and batched MTP verify (`HIVE_MTP_BATCH`) | real-chat, 4 base runs vs 2: c8 93.2 → 125.7 tok/s, c8 first token 23 → 0.6–0.9 s; c2 66.6 → 79.5; c1 / c4 inside the base spread |
| 19 | CPU-expert chunks sized to the workers (`HIVE_GLM_CPU_ADAPT`, bit-identical) | gate/up workers busy 58 → 80 %, CPU phase 70 → 81 GB/s; real-chat unchanged |
| 20 | Logits into a pinned buffer, hived's verify row vector reused | no pageable copy into fresh memory per verify step; real-chat unchanged |
| 21 | Promotion victims sorted once per step (same choices) — `after_step` scanned every slot for each used expert and held the GPU idle 1.88 ms per verify step (host timestamps, CUPTI timeline) | real-chat, interleaved ×2: c1 67.3 → 70.4, c2 78.6 → 86.8, c4 88.1 → 96.6, c8 126.0 → 129.9 tok/s |
| 22 | Early routing on the fast decode path (`HIVE_GLM_EARLY_ROUTE`) — the DeepSeek post-and-gate (`er_route_post`, `er::Gate`) after the router; the host classifies and launches experts while the shared expert runs | outputs identical; real-chat, interleaved ×2: c4 95.3 → 99.4 tok/s, c1 / c2 / c8 +0.4 / −1.7 / −0.1 % |

Steps 13 and 14 were then carried over to DeepSeek as its step 21.

### Rejected (measured, not adopted)

| Idea | Result |
| --- | --- |
| Cache-aware routing λ 0.2 | 63.5 tok/s (hits 97.9 %), but a long-context item flipped (157) — kept at 0.1 |
| Skipping low-weight CPU misses (`HIVE_GLM_SKIP_MISS` 0.05 / 0.1) | +0 / +2 % (noise), lossy — kept off |
| CPU workers spinning 1,000 / 300 µs instead of 3,000 after a batch (`HIVE_GLM_SPIN_US`) | c1 69.0 / 67.7 vs 68.3 tok/s — no effect |
| MTP depth 4 / 5 | 46.7 / 45.8 vs 48.1 tok/s at 3 |
| More than one expert copied ahead per layer | 2 → 30.7 tok/s; for verify steps 44.5 / 42.0 / 36.3 vs 53.1 tok/s — the copies compete for DRAM bandwidth |
| W4A8 CPU kernel | 49.4 → 44.8 GB/s on one node, lower accuracy |
| Layer-major buffers reserved permanently | 3.6 GiB less cache, c1 −6 % |
| Pinning CPU workers to physical cores | 45.2 / 45.3 vs 46.3 / 45.1 tok/s unpinned |
| Equal row ranges per CPU worker (`HIVE_GLM_CPU_BALANCE`) | 18.65 / 19.98 vs 19.44 / 19.73 ms per 3-row verify |

### Open problem

With cache-aware routing the CPU share of a decode step is under a quarter; what remains is mostly GPU work outside the
experts, led by the 34 KDA layers whose projections are BF16 (5.2 ms of a 34 ms 4-row verify, at the HBM limit for those
weights), and the hyper-connection mixing. A CUPTI timeline of one stream still shows the GPU idle about a quarter of the time:
waits for CPU experts and for deferred experts (routing is posted early since step 22). Prefill stays bound by PCIe 4.0.
