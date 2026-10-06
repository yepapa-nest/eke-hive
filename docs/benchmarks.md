# Benchmarks

The headline numbers in the README, for both models, come from one benchmark: `tools/bench_chat.py`, real chat prompts sent
to the API. [performance.md](performance.md) records how both models got there step by step, with the benchmarks used at the time
(for DeepSeek's steps 1–20 the development benchmark `tools/bench_tune.py`); those use different methods and should not be
compared row by row with the tables below.

## Real-chat benchmark (headline)

`tools/bench_chat.py`, reference machine, a server that had already been serving (the expert cache filled by earlier
requests — the harness itself sends no warm-up), no other client:

- twelve different chat prompts (5 Korean, 3 English, 2 code, 2 creative), answers up to 600 tokens, temperature 0,
  `reasoning_effort: none`; decode counts only the tokens after each request's first one, so prompt processing is not mixed in;
- one stream: the twelve prompts one after another; N streams (2–32): each walks the list from a different start — total =
  decode tokens over the wall time of the decode phase, per stream = median of the requests' own decode speed, time to first
  token = median (max) over the requests, queueing included;
- one stream also reports the speed per prompt kind; long prompts (~T tokens of documentation text) are sent one at a time.

Commands (the server started with the shipped configuration):

```bash
python3 tools/bench_chat.py --name deepseek                                          # 1–32 streams, 17K / 42K / 54K prompts
python3 tools/bench_chat.py --name glm --long 17000,42000,54000,100000,200000,250000   # GLM, up to 250K
```

Decode, tok/s in total (per stream), and time to first token, median (max):

| Streams | DeepSeek-V4.1-Flash | | GLM-5.3-Flash | |
| --- | --- | --- | --- | --- |
| 1 | **85.3** (83.8) | 0.23 s (0.50) | **70.8** (71.2) | 0.51 s (0.96) |
| 2 | **92.9** (47.1) | 0.26 s (0.49) | **87.8** (47.2) | 0.54 s (0.92) |
| 4 | **124.6** (33.9) | 0.58 s (0.84) | **103.9** (28.6) | 0.53 s (1.73) |
| 8 | **175.0** (23.8) | 1.42 s (1.72) | **130.6** (17.5) | 0.63 s (3.40) |
| 16 | **178.1** (23.0) | 9.82 s (29.1) | **130.3** (17.4) | 12.7 s (39.2) |
| 32 | **181.5** (23.9) | 34.4 s (79.4) | **130.9** (16.9) | 45.1 s (107.8) |

Mean of two runs each (DeepSeek: the build of 2026-10-05; GLM: 2026-10-07, with the 64K-token draft head — measured with a ShareGPT-based draft list; the shipped list is built from OpenAssistant
oasst2 + NSMC and covers ~1 point fewer chat tokens, not yet re-measured). One stream by prompt kind
(tok/s): DeepSeek Korean 87.2 · English 82.7 · code 106.2 · creative 62.4; GLM 75.5 · 67.9 · 73.9 · 54.4. Both models decode up to 8 requests at a time (`HIVE_MAX_BATCH`); beyond that requests
wait, which is the time to first token at 16–32 streams.

Long prompts — time to first token · prefill tok/s · decode tok/s after it:

| Prompt | DeepSeek-V4.1-Flash | GLM-5.3-Flash |
| --- | --- | --- |
| ~17K | 5.05 s · 3,530 · 63.4 | 7.37 s · 2,307 · 54.9 |
| ~42K | 7.53 s · 5,918 · 79.1 | 11.71 s · 3,639 · 49.6 |
| ~54K | 9.69 s · 5,816 · 75.8 | 13.96 s · 3,925 · 45.1 |
| ~100K | — | 26.43 s · 3,833 · 45.1 |
| ~200K | — | 53.72 s · 3,787 · 43.4 |
| ~250K | — | 66.29 s · 3,837 · 40.0 |

Builds and settings: DeepSeek with `config/hive.env`, GLM with `config/glm.env` (8 seats, batched MTP verify, layer yields, context
262,144); two runs each, no other traffic. The GLM 100K–250K rows are from the previous build (a single request's prefill — nothing
in that path changed). Single-stream runs vary most on the code prompts: the previous headline (DeepSeek 76.9 tok/s at one
stream, one run) was measured on the same benchmark; re-measured interleaved with today's build on the same day, the step-22/23
changes gave 83.9 → 86.0 tok/s at one stream ([performance.md](performance.md)) — the rest of the difference to 76.9 is run-to-run spread.
Without the two routing changes of step 21 DeepSeek measured 53.8 / 68.4 / 81.1 / 100.2 / 98.4 / 105.1 tok/s (1–32 streams) on
the same benchmark. GLM before today's steps 17–21 ([performance.md](performance.md#glm-53-flash)): 65.2 / 66.6 / 89.3 / 88.8 / 91.6 / 93.3 tok/s.

Quality on the bundled suite (`tools/quality_eval.py`, [validation.md](validation.md#quality-suite)), same builds:

| Suite | DeepSeek-V4.1-Flash | GLM-5.3-Flash (thinking `low`) |
| --- | --- | --- |
| Long context (needle retrieval) | 32 / 35 (5 lengths, 4K–100K) | 21 / 21 (3 lengths: 64K, 128K, 250K) |
| Korean / English QA | 79 / 80 | 75 / 80 |
| Code | 20 / 20 | 20 / 20 |
| Tool calls | 22 / 22 | 22 / 22 |
| Operations | 22 / 22 | 20 / 20 (the 2 image items skipped: the GLM checkpoint ships no example image) |
| **Total** | **175 / 179** | **158 / 163** |

DeepSeek scored 172 / 179 without the two routing changes (the long-context items gave identical outputs).

## Sleep and wake

`HIVE_SLEEP_LEVEL=3 scripts/hive-sleep.sh sleep` / `wake` with a 120K-token conversation open (shipped configurations,
`HIVE_SLEEP_VMM=1`):

| | DeepSeek-V4.1-Flash | GLM-5.3-Flash |
| --- | --- | --- |
| Sleep (level 3) | 0.68 s | 3.8 s |
| GPU memory in use while asleep | ~1 GB | ~1 GB |
| Wake | 3.2 s | 3.0 s |
| First token of that conversation's next turn after the wake (whole history reused) | 0.6 s | 2.6 s |

Cold start to ready: DeepSeek 88–90 s (RAM mode), GLM 95–97 s. More sleep levels and conditions for GLM:
[glm.md](glm.md#sleep-and-wake).
