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
| 1 | **76.9** (75.6) | 0.23 s (0.26) | 65.2 (65.5) | 0.50 s (0.68) |
| 2 | **92.6** (46.9) | 0.26 s (0.50) | 66.6 (33.9) | 0.50 s (1.00) |
| 4 | **124.0** (33.6) | 0.37 s (0.84) | 89.3 (24.6) | 0.52 s (1.58) |
| 8 | **173.0** (23.2) | 1.18 s (1.73) | 88.8 (22.8) | 24.0 s (28.4) |
| 16 | **172.7** (22.1) | 9.55 s (30.0) | 91.6 (23.6) | 32.6 s (70.4) |
| 32 | **174.3** (22.8) | 34.7 s (82.1) | 93.3 (23.6) | 77.6 s (157.2) |

One stream by prompt kind (tok/s): DeepSeek Korean 82.5 · English 71.6 · code 82.9 · creative 61.1; GLM 69.6 · 64.2 · 62.1 · 55.7.
DeepSeek decodes up to 8 requests at a time, GLM 4 (`HIVE_MAX_BATCH`); beyond that requests wait, which is the time to first
token at 8–32 streams.

Long prompts — time to first token · prefill tok/s · decode tok/s after it:

| Prompt | DeepSeek-V4.1-Flash | GLM-5.3-Flash |
| --- | --- | --- |
| ~17K | 4.95 s · 3,576 · 64.1 | 7.21 s · 2,400 · 53.6 |
| ~42K | 7.48 s · 5,804 · 72.6 | 11.55 s · 3,697 · 53.2 |
| ~54K | 9.63 s · 5,788 · 77.4 | 13.74 s · 3,966 · 48.2 |
| ~100K | — | 26.43 s · 3,833 · 45.1 |
| ~200K | — | 53.72 s · 3,787 · 43.4 |
| ~250K | — | 66.29 s · 3,837 · 40.0 |

Builds and settings: DeepSeek with `config/hive.env` (cache-aware routing λ 0.1 and expert deferral on); GLM with
`config/glm.env` (context 262,144, KV memory that follows use) — GLM measured in one run (the one-stream step re-run at the
end because another request overlapped its first request; no other traffic otherwise).
One-stream GLM runs of the same build read 65.2–68.4 tok/s; the code prompts vary most between runs.
Without the two routing changes DeepSeek measured 53.8 / 68.4 / 81.1 / 100.2 / 98.4 / 105.1 tok/s (1–32 streams) and
36.9 / 43.8 / 44.5 tok/s decode after the 17K / 42K / 54K prompts on the same benchmark, with the same prefill.

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
