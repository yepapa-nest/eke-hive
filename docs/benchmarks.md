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

Decode, tok/s in total (per stream), and time to first token, median (max) — GPU at its full 600 W, and with the board capped
at 300 W (`nvidia-smi -pl 300`):

| Streams | DeepSeek-V4.1-Flash | | DeepSeek, 300 W cap | GLM-5.3-Flash | | GLM, 300 W cap |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **80.7** (82.9) | 0.26 s (0.38) | 82.9 · 0.22 s | **68.1** (68.8) | 0.52 s (0.70) | 69.4 · 0.50 s |
| 2 | **106.4** (56.2) | 0.25 s (0.42) | 99.9 · 0.26 s | **86.5** (46.2) | 0.54 s (0.88) | 85.6 · 0.53 s |
| 4 | **125.0** (31.9) | 0.25 s (0.82) | 115.8 · 0.27 s | **103.5** (28.9) | 0.54 s (1.85) | 102.5 · 0.54 s |
| 8 | **172.8** (23.5) | 0.99 s (1.82) | 157.9 · 0.93 s | **129.0** (17.4) | 0.63 s (3.33) | 129.3 · 0.62 s |
| 16 | **171.6** (22.6) | 9.51 s (29.3) | 160.8 · 11.1 s | **127.4** (17.3) | 12.7 s (39.7) | 129.8 · 12.8 s |
| 32 | **176.9** (23.3) | 34.4 s (80.4) | 163.1 · 39.3 s | **128.9** (17.0) | 40.4 s (108.3) | 130.5 · 43.9 s |

One run per column, 2026-10-09, same builds for both caps (DeepSeek with steps 29–32 of [performance.md](performance.md); the
DeepSeek 300 W 2- and 4-stream values are the mean of two batched-speculation runs), CPU boost off (3.5 GHz). One stream by prompt
kind (tok/s, 600 W): DeepSeek Korean 87.1 · English 82.0 · code 78.2 · creative 62.0; GLM 72.0 · 67.3 · 67.1 · 56.8 (the code prompts
vary most between runs). Both models decode up to 8 requests at a time (`HIVE_MAX_BATCH`); beyond that requests wait, which is
the time to first token at 16–32 streams. The board cap costs DeepSeek 6–9 % at 4–32 streams and makes its long prompts take 34–47 % longer; GLM's
decode does not move and its long prompts take 13–38 % longer.

Long prompts — time to first token · prefill tok/s · decode tok/s after it (the prompts are built from `docs/*.md`, so their size
follows the docs: now 17.7K / 44K / 57K tokens for DeepSeek and 16.9K / 42K / 55K for GLM; each sent once after a cache flush):

| Prompt | DeepSeek-V4.1-Flash | DeepSeek, 300 W cap | GLM-5.3-Flash | GLM, 300 W cap |
| --- | --- | --- | --- | --- |
| ~17K | 4.90 s · 3,608 · 71.9 | 6.56 s · 2,696 · 71.2 | 7.21 s · 2,346 · 55.6 | 8.17 s · 2,071 · 55.4 |
| ~42K | 7.49 s · 5,875 · 82.3 | 11.00 s · 3,998 · 80.6 | 11.31 s · 3,730 · 47.0 | 15.25 s · 2,765 · 48.8 |
| ~54K | 9.59 s · 5,946 · 74.1 | 13.66 s · 4,174 · 72.1 | 13.74 s · 4,000 · 42.2 | 18.90 s · 2,907 · 40.6 |
| ~100K | — | — | 26.43 s · 3,833 · 45.1 | |
| ~200K | — | — | 53.72 s · 3,787 · 43.4 | |
| ~250K | — | — | 66.29 s · 3,837 · 40.0 | |

Builds and settings: DeepSeek with `config/hive.env`, GLM with `config/glm.env` (8 seats, batched MTP verify, layer yields, context
262,144), no other request on the engine during a run — a request held at a lower priority still holds one of the 8 seats, so a
benchmark run waits until the engine is idle. The GLM 100K–250K rows are from an earlier build at 600 W (a single request's prefill —
nothing in that path changed). Earlier headlines on the same benchmark (two runs each): DeepSeek of 2026-10-05 85.3 / 92.9 / 124.6 /
175.0 / 178.1 / 181.5 tok/s, GLM of 2026-10-07 70.8 / 87.8 / 103.9 / 130.6 / 130.3 / 130.9 tok/s — one-stream runs of the same build
spread by a few percent, so the one-stream difference is within run-to-run variation, while DeepSeek's 2-stream gain is batched
speculation (step 31). Without the two routing changes of step 21 DeepSeek measured 53.8 / 68.4 / 81.1 / 100.2 / 98.4 / 105.1 tok/s
(1–32 streams); GLM before its steps 17–21 ([performance.md](performance.md#glm-53-flash)): 65.2 / 66.6 / 89.3 / 88.8 / 91.6 / 93.3 tok/s.

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
