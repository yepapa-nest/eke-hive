# Qwen3-ASR validation — 2026-10-18

Engine: `libhive_qasr.so` + `server/speech/asr_server.py` (docs/qasr.md). Checkpoint: Qwen3-ASR 1.7B, bf16. GPU: RTX PRO 6000 Blackwell
(96 GB, sm_120a), shared with other services. Test set: 30 synthetic Korean clips (3 voices × 10 sentences, 1–24 s, 16 kHz mono) with a
reference transcript. Baseline: the SGLang server this engine replaces (lmsysorg/sglang nightly, `qwen3_asr`, bf16, context 2048,
CUDA graphs).

## 1. Correctness chain

| Step | Result |
|---|---|
| Front end (`qasr_frontend.py`) vs the reference feature extractor | max abs diff ≤ 2.2e-5 (4 synthetic signals + 12 clips), same frame counts |
| fp32 oracle (`oracle/qwen3_asr_oracle.py`) vs the reference implementation (fp32, CPU) | prompt ids identical; **greedy ids identical on 30/30 clips**; first-step logits cosine 1.0000000, max abs diff ≤ 2.9e-5 (4 clips) |
| engine (bf16) vs oracle (fp32) | **greedy ids identical on 30/30 clips**; audio-encoder output cosine ≥ 0.999984 (rel. L2 ≤ 5.7e-3); first-step logits cosine ≥ 0.999982, top-1 identical 30/30 |
| engine kernels (`test_qasr_kernels`) | all pass (GEMV / norms / RoPE + KV store / prefill, encoder and decode attention / argmax / conv1 + im2col) |
| batching, cancel, sampling (`test_qasr_engine.py`) | two concurrent requests = each alone; a cancel flag stops a request; seeded sampling reproducible |
| HTTP vs the SGLang server (`test_qasr_server.py --ref`) | 3 clips × (forced Korean, auto language): **identical** `text`, `output_ids`, prompt / completion tokens and `finish_reason`; auto language returns `language Korean<asr_text>…` exactly as SGLang does |
| error paths | malformed JSON, missing / extra audio, bad base64, non-WAV → HTTP 400; client disconnect cancels the request (log: "cancelled by client disconnect after 332 tokens"); server healthy afterwards |

## 2. Accuracy (CER, character-weighted, whitespace and punctuation removed)

| | CER |
|---|---|
| reference implementation (fp32) = oracle = **hive** | **0.0054** |
| SGLang (bf16), same harness and requests | 0.0067 |

The texts differ on 3 of 30 clips. Two differences are spacing or punctuation only. In the third, SGLang's bf16 path picks *yuryeong* ("ghost") where
the fp32 reference and hive both say *yoryeong* ("knack"), which is the correct word.

## 3. Latency (request → full text, whole clip, client side)

Both servers were measured **interleaved clip by clip** (`tools/test_qasr_bench.py`, 3 rounds; concurrency 2 = two clips sent at once),
so they saw the same GPU conditions. The large LLM daemon was stopped, but a full-model test of another 12B model and the live voice services
were running on the same card.

| | hive p50 / p90 / max (ms) | SGLang p50 / p90 / max (ms) |
|---|---|---|
| forced Korean, concurrency 1 (n = 90) | **71.5 / 168.0 / 359.3** | 151.8 / 295.7 / 963.1 |
| forced Korean, concurrency 2 (n = 60) | **77.8 / 198.5 / 375.8** | 180.0 / 322.0 / 489.1 |
| auto language, concurrency 1 (n = 30) | **102.4 / 182.1 / 335.3** | 179.6 / 307.2 / 450.7 |

| p50 by clip length (concurrency 1) | 0–3 s | 3–6 s | 6–30 s |
|---|---|---|---|
| hive | 31.2 | 68.4 | 164.3 |
| SGLang | 91.5 | 147.3 | 279.6 |

The earlier quiet-GPU figures for SGLang (p50 116 / p90 196 / max 297 ms) were measured through the client application's turn pipeline with
the large LLM daemon off. That is a different harness, so they are context, not a like-for-like comparison.

Engine timing on a quiet GPU (`test_qasr_engine.py`):

| Clip | Encoder | Prefill | Decode | Engine total |
|---|---|---|---|---|
| 4 tokens | 2.0 ms | 3.7 ms | 10.5 ms | 16 ms |
| 55 tokens (19 s audio) | 3.0 ms | 6.5 ms | 159.6 ms (≈ 2.9 ms/token) | 169 ms |

The worst cases (max column) are the 55-token clips while the neighbouring test was busy: decode is memory-bandwidth bound and time-sliced
with the other processes.

Decode GEMVs (one decoder layer's four matrices, 100.7 MB, streamed from DRAM, CUDA graph):

| | m = 1 | m = 2 |
|---|---|---|
| hive | 1,512 GB/s | 1,446 GB/s |
| cuBLAS, same products | 1,413 GB/s | 1,326 GB/s |

## 4. VRAM

| | MiB |
|---|---|
| **hive process, nvidia-smi per process, flat over the whole benchmark** | **5,040** |
| start-up plan (context 555 measured + weights 3,888 + KV 448 for 2 × 2048 positions + work 104 + cuBLAS 32 + graphs / modules 96) | 5,124 |
| SGLang server, same time | 6,028 |

Everything is allocated at start and nothing is allocated per request. `HIVE_VRAM_LIMIT_MB` caps the plan, and the start is refused when
the plan does not fit; with too little free VRAM the start was refused with "need 4568 MiB after the context, 3619 MiB free".
