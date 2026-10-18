# Qwen3-TTS validation — 2026-10-18

Engine: `libhive_qtts.so` + `server/speech/tts_server.py` ([qtts.md](qtts.md)). Checkpoint: Qwen3-TTS 12Hz 1.7B CustomVoice (bf16 talker,
fp32 codec). GPU: RTX PRO 6000 Blackwell (96 GB, sm_120a), shared with other services. Baseline: the server this engine replaces
(faster-qwen3-tts 0.5.4 on qwen-tts / transformers, bf16, CUDA graphs, 4-frame chunks, 25 frames of codec left context, same API).
Speaker `sohee`, language Korean unless noted.

## 1. Correctness chain

| Step | Result |
|---|---|
| fp32 oracle (`oracle/qwen3_tts_oracle.py`) vs the reference implementation (`qwen_tts`, fp32, CPU, greedy talker and code predictor) | prompt token ids identical; **every frame identical** on 4 prompts: 25/25 frames (short sentence), 76/76 (long sentence + instruct), 59/59 (language Auto; the reference stopped one frame earlier at the 60-frame cap), 52/52 (speaker dylan, language Auto → Beijing-dialect id 2074); waveform from the same codes: max abs diff ≤ 2.2e-6, SNR 117.96 / 118.68 / 118.94 dB |
| codec, chunked stream vs whole-sequence decode (oracle, fp32) | 25 frames: max abs diff 1.6e-6; **220 frames (the 72-frame window crossed several times): max abs diff 2.4e-6** (527 of 422,400 PCM16 samples differ by one LSB) |
| engine kernels (`test_qtts_kernels`) | 42 checks pass (GEMV modes / prologues / M 1–4, q/k norm + RoPE, decode attention split and unsplit at 1 / 38 / 701 keys, prefill attention, sampling: penalty + suppression + EOS rules, top-k membership and frequencies (max deviation 0.0036 over 20,000 draws), im2col + SnakeBeta, ConvNeXt front, codec window attention with a wrapped ring, codec output, RVQ) |
| engine codec (bf16) vs oracle whole-sequence decode (fp32), 220 frames | **SNR 43.9 dB** over the whole signal (46.0–50.2 dB per chunk sampled), PCM16 compared |
| engine (bf16) vs oracle (fp32), teacher forcing on the 76-frame instruct prompt (1,217 logit rows) | **talker first-codebook top-1 identical in 77/77 rows**, max diff 0.19 % of the logit range; all rows: top-1 98.5 % (the differing rows are code-predictor near-ties, e.g. margin 0.012), max diff 1.6 %, mean 0.47 %, min cosine 0.99998 |
| engine greedy, free running vs oracle greedy | frame 0 identical; frame 1 differs only in codebook 15 (the 0.012 near-tie above); after that the sequences diverge, as any bf16 vs fp32 autoregression does — teacher forcing (above) is the comparison that isolates the arithmetic |
| CUDA graphs vs eager launches (`test_qtts_engine graphcheck`, seeded sampling, 75 frames) | codes and PCM **bit-identical** |
| bug found and fixed during validation | the per-request sampling settings were kernel arguments, so the frame graph captured at start-up kept the defaults (a greedy test request sampled at 0.9). They now live in the device control block and are read at run time. The server's default settings were never affected; all results above are from the fixed build. |

## 2. Quality: ASR round trip

The same 11 Korean sentences go through each server's `/ws/speak`. Each result is resampled to 16 kHz and transcribed by the Qwen3-ASR 1.7B
server (greedy, language fixed to Korean). CER is character-weighted, with whitespace and punctuation removed. Sampling is random, so each
run is a different take.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| **hive** | 0.035 (9/11 exact) | 0.018 (10/11) | 0.000 (11/11) |
| replaced server | 0.013 (10/11) | 0.009 (9/11) | |
| **hive, language Auto** | 0.013 (10/11) | | |

hive's differences are numerals spoken as words that the ASR writes as digits ("twenty-three degrees" → "23 degrees", "one
liter" → "1 liter", "ten grams" → "10 g", in Korean); the replaced server shows the same pattern. One hive take dropped one particle
(a topic marker). One baseline take inserted a filler and an extra word. The content is the same; none of the takes was truncated or garbled.

## 3. Latency and throughput

`python -m tools.bench_speech tts --voice Sohee --runs 3` (5 sentences × 3, client side, first binary frame after sending the request).
Both servers were measured back to back under the same GPU conditions: the large LLM daemon was stopped, the live Gemma / ASR services
and a full-model test of another model were running. The replaced server's numbers equal its earlier measurement beside the 68 GB daemon
(163 ms / 0.33).

| | first chunk p50 | first chunk max | real-time factor (median) |
|---|---|---|---|
| **hive** | **25.3 ms** | **25.9 ms** | **0.069** |
| replaced server | 162.7 ms | 168.1 ms | 0.330 |

Engine-side breakdown (`test_qtts_engine generate`): prompt (tokens → first code) 3.4–5.7 ms, then 5.3–5.4 ms per frame, codec included
(the codec alone takes 0.32 ms per frame), so 4 frames + codec ≈ 21 ms. Under heavier GPU contention (another full-model test running
at the same time) the step grew to 10.2–10.7 ms per frame: first chunk 48–50 ms, RTF 0.13.

**Barge-in:** closing the socket after the first chunk of a ~30 s utterance stops the generation (it is checked before every frame).
A new request sent right afterwards got its first chunk in 25 ms, so nothing was left running.

## 4. VRAM

| | MiB |
|---|---|
| CUDA context (measured; per-thread stack limit lowered to 256 B, saving ~210 MiB) | 343 |
| weights | 3,297 |
| KV caches (2048 talker positions) + work buffers | 308 |
| cuBLAS workspace + handle, graphs | ~20 |
| **process, nvidia-smi peak under load (2 Hz trace while serving)** | **3,968** (flat) |
| replaced server | 5,296 |

`HIVE_VRAM_LIMIT_MB=3900`: the engine lowered max_seq to 768 positions and measured 3,831 MiB, then served normally.
`HIVE_VRAM_LIMIT_MB=3000`: creation refused with "context 343 + weights 3297 + buffers 167 + fixed 72 MiB".
