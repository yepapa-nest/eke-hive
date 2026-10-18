# Qwen3-ASR in Eke Hive

> **Status (2026-10-18):** validated end to end on an RTX PRO 6000 Blackwell — greedy output identical to the fp32 reference on the 30-clip
> test set, the same HTTP answers as the SGLang server it replaces, about half its latency, 4,828 MiB of VRAM (numbers:
> [qasr-validation-20261018.md](qasr-validation-20261018.md)).

Eke Hive serves **Qwen3-ASR 1.7B** (config.json `model_type` `qwen3_asr`, bf16 safetensors) as a speech-recognition service: an audio
encoder feeding a Qwen3 1.7B decoder. Unlike the chat families it does not run inside `hived`. A small engine library, `libhive_qasr.so`,
sits behind a Python HTTP server (`server/speech/asr_server.py`) that speaks SGLang's native `/generate` API, so a client written for an
SGLang Qwen3-ASR server works unchanged.

## Model

| Part | Shape |
|---|---|
| Front end | 128-bin log-mel, 25 ms Hann window, 10 ms hop (100 frames/s), Slaney mel scale 0–8 kHz, log10, floor at max − 8, (x + 4) / 4 |
| Audio encoder, convolutions | per 100-frame chunk: 3 × Conv2d(3×3, stride 2, pad 1) + GELU, 1 → 480 → 480 → 480 channels, 128 mel → 16, 100 frames → 13 tokens; a short tail chunk is zero-padded to 100 frames when there is more than one chunk; `conv_out` 7680 → 1024 |
| Audio encoder, transformer | sinusoidal positions restarting in every chunk; 24 pre-LayerNorm layers, width 1024, 16 heads × 64, GELU MLP 4096; attention **block-diagonal** over windows of 104 tokens (8 s); `ln_post` → `proj1` + GELU → `proj2` 1024 → 2048 |
| Decoder | Qwen3: 28 layers, width 2048, 16 query / 8 KV heads × 128, q / k RMSNorm per head, SwiGLU 6144, RMSNorm eps 1e-6 |
| Positions | "interleaved mRoPE" (sections 24/20/20), but every token, audio or text, has the same position on all three axes, so it is plain rotate-half RoPE (θ 1e6) over positions 0, 1, 2, … |
| Head | `lm_head` is stored separately but is byte-identical to `embed_tokens`; the engine keeps one copy |

The prompt holds one `<|audio_start|><|audio_pad|><|audio_end|>` group. The single `<|audio_pad|>` stands for the audio and is expanded to
the encoder's token count: 13 per second of audio, plus ⌈⌈⌈r/2⌉/2⌉/2⌉ for a tail of r frames. The encoder's output rows replace the
embeddings at those positions.

The math contract is the docstring of `oracle/qwen3_asr_oracle.py` (this project's own fp32 PyTorch model, written from the published model
description and checked against the reference implementation).

## Engine (`engine/src/qasr/`, C API `engine/include/hive/qasr/qasr.h`)

- **Numerics:** bf16 weights. GEMM inputs are bf16; GEMM outputs and both residual streams are fp32. Norms, softmax and RoPE are fp32;
  the KV cache is bf16.
- **Encoder:** `conv1` is a direct kernel (one input channel). `conv2`, `conv3` and `conv_out` are im2col + cuBLAS in groups of 4 chunks;
  the im2col column order is reordered at load so every copy is a contiguous 16-byte vector. The transformer runs in passes of up to
  8 windows (64 s). The windows are independent in every layer, so a pass is exactly the reference computation for its tokens. Windowed
  attention is our own tiled kernel.
- **Prefill:** cuBLAS GEMMs in chunks of up to 512 prompt tokens; causal attention reads the sequence's KV cache.
- **Decode:** one step serves every active sequence (1–2) through a **CUDA graph** per batch size. The weights are streamed once for the
  whole batch by our own GEMVs: one warp per row, 16-byte non-allocating loads, with fused residual add and fused SwiGLU. Token, position
  and slot are read from a small device array written before each launch. Greedy decoding uses an on-device argmax (first index on ties);
  temperature / top-p / top-k sampling runs on the host from the copied logits, with a per-request seed.
- **GEMM shapes:** cuBLAS loads its kernels lazily, and a first-seen M stalled one request by ~80 ms. So GEMM row counts are rounded up to
  buckets (16, 32, 64, then multiples of 64, capped at the buffer size), and the start-up warm-up runs every bucket.
- **Scheduling:** one worker thread owns the stream. `qasr_transcribe` may be called from many threads; up to `max_seqs` requests decode
  together, and further ones wait in FIFO order. A new request's encoder + prefill runs between two decode steps. A cancel flag is checked
  before every step.
- **Memory:** everything is allocated in `qasr_create` (weights, KV for `max_seqs` × `max_ctx`, one union work arena for the encoder and
  decoder phases, cuBLAS workspace, graphs), followed by a warm-up. No device allocation happens per request. The start-up plan (context
  measured as NVML free before minus CUDA free after creation) is checked against `HIVE_VRAM_LIMIT_MB` and against the free VRAM; if it
  does not fit, `qasr_create` fails with the reason. The per-thread stack limit is lowered to 256 B right after the context is created (the
  qasr kernels use no stack frame; the default 1 KiB reservation on every resident thread cost 212 MiB — measured 2026-10-18).

| VRAM (2 slots × 2048 positions) | MiB |
|---|---|
| CUDA context (measured, stack limit 256 B) | 343 |
| weights (one embedding copy) | 3,888 |
| KV cache | 448 |
| work buffers | 104 |
| cuBLAS workspace | 32 |
| **process total, measured under load** | **4,828** (5,040 with the default stack limit) |

## HTTP API (`server/speech/asr_server.py`)

- `POST /generate`:
  - Request: `{"text": <prompt>, "audio_data": "data:audio/wav;base64,<WAV>", "sampling_params": {...}}`, or `input_ids` in place of
    `text`; `audio_data` may also be a one-element list.
  - Sampling: temperature (default 1.0; 0 = greedy), top_p, top_k, max_new_tokens (default 128), stop_token_ids, ignore_eos, seed. The
    defaults are SGLang's.
  - Response: `{"text", "output_ids", "meta_info": {prompt_tokens, completion_tokens, finish_reason, e2e_latency, timings_ms}}`. As in
    SGLang, `output_ids` ends with the matched stop token, `completion_tokens` counts it, and `finish_reason` is
    `{"type": "stop", "matched": id}` or `{"type": "length", ...}`.
  - Language: with a forced language (prompt ending `language Korean<asr_text>`) the text is the transcript. Without one, the model
    writes `language <Name><asr_text>` itself and the text carries it. `<asr_text>` is an ordinary added token, not a special one, so it
    survives decoding exactly as in SGLang; clients split on it.
  - Audio: RIFF/WAVE (8/16/24/32-bit PCM or float, any channel count). Input that is not 16 kHz is resampled linearly. Decoding and
    log-mel run on the CPU (≈ 7 ms for 24 s of audio).
  - Errors (malformed JSON, audio without a placeholder or a placeholder without audio, bad base64, not a WAV, more than one audio, prompt
    longer than the context) return HTTP 400 with `{"error": {...}}`. A client that disconnects has its request cancelled.
- `GET /health`, `/health_generate` → `{"status": "ok"}`; `GET /v1/models` → `{"data": [{"id": "asr", ...}]}`; `GET /get_model_info`.

## Build and run

```bash
# build (inside the image, like every engine target)
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/hive -w /hive/engine eke-hive:latest \
  bash -c "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 16 --target hive_qasr test_qasr_kernels"

# start: container hive_qasr, port 8444 (config/qasr.env)
HIVE_CKPT=/path/to/Qwen3-ASR-1.7B scripts/hive-start.sh            # hands the qwen3_asr checkpoint to scripts/qasr-start.sh
HIVE_MODEL=asr HIVE_VRAM_GB=8 scripts/hive-start.sh                 # profile: checkpoint looked up under HIVE_MODELS_DIR, 8 GiB cap
scripts/qasr-start.sh --port 18444 --wait                           # direct, another port, wait for /health
HIVE_VRAM_LIMIT_MB=5300 scripts/qasr-start.sh                       # cap: refuses to start if the plan does not fit
scripts/hive-stop.sh                                                # with the same settings (HIVE_CKPT / HIVE_MODEL) — or docker rm -f hive_qasr
```

Settings in `config/qasr.env`: `HIVE_CONTAINER` (hive_qasr), `HIVE_PORT` (8444), `HIVE_BIND`, `HIVE_GPU`, `HIVE_QASR_MAX_SEQS` (2),
`HIVE_MAX_CTX` (2048), `HIVE_QASR_SERVED_NAME` (asr). The server log goes to `$HIVE_STATE_DIR/logs/asr_server.log`.

## Tests

| Test | Needs | Checks |
|---|---|---|
| `tools/test_qasr_frontend_cpu.py` (in `tools/test_cpu.sh`) | CPU | audio token count, WAV decoding, log-mel vs the reference extractor (≤ 2e-4) |
| `engine/build/test_qasr_kernels` | GPU, ~0.7 GB | every kernel vs a CPU reference; decode GEMV bandwidth (CUDA graph) next to cuBLAS |
| `oracle/qwen3_asr_oracle.py --ref <dir>` | CPU, 16 threads | oracle vs `tools/test_qasr_ref_dump.py` output (the reference implementation): prompt ids, greedy ids, first-step logits |
| `tools/test_qasr_engine.py --oracle <dump>` | GPU, ~5.1 GB | engine vs oracle per clip: encoder output, first-step logits, greedy ids; CER; concurrency, cancel, sampling |
| `tools/test_qasr_server.py --url … [--ref …]` | running server | HTTP surface, forced and auto language, error paths, disconnect cancel; identical answers to a reference server |
| `tools/test_qasr_bench.py --url … --url …` | running servers | CER and request-to-text latency, several servers interleaved clip by clip, concurrency 1 or 2 |

`HIVE_QASR_DEBUG_LAYERS=N` (validation only) loads just the first N decoder layers; it pairs with `qwen3_asr_oracle.py --layers N` to check
the whole pipeline on a GPU without room for the full model.

## Limits

- One audio input per request; no streaming responses (the reference client does not stream either: partial results come from re-sending
  the audio so far).
- Context 2048 positions per request by default: up to about 2 minutes of audio (13 tokens/s) with 256 generated tokens; longer audio is
  refused with HTTP 400 (raise `HIVE_MAX_CTX`, which costs 224 KiB of KV per position for 2 slots).
- The GPU is time-sliced with other processes on the card: under heavy load from a neighbour, a long request's decode stretches, as it does
  for any server (see the validation note).
- The start-up "measured" VRAM is NVML free before the context minus CUDA free after the warm-up; it is wrong if another process allocates
  or frees memory during those seconds. For the real figure, read the per-process number from `nvidia-smi`.
