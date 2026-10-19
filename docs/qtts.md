# Qwen3-TTS in Eke Hive

> **Status (2026-10-18):** validated end to end on an RTX PRO 6000 Blackwell. The first audio chunk arrives in 25 ms (the server it replaces:
> 163 ms) at real-time factor 0.069 (0.33). Quality matches in an ASR round trip, and VRAM is 3,968 MiB (5.3 GB before). Numbers:
> [qtts-validation-20261018.md](qtts-validation-20261018.md).

Eke Hive serves **Qwen3-TTS 12Hz 1.7B CustomVoice** (config.json `model_type` `qwen3_tts`, bf16 talker + code predictor, fp32 codec decoder
in `speech_tokenizer/`) as a streaming text-to-speech service. Like the ASR service, it does not run inside `hived`. A small engine
library, `libhive_qtts.so`, sits behind a Python server (`server/speech/tts_server.py`). The server speaks a WebSocket API that streams raw
PCM while the audio is being generated.

## API

| Endpoint | |
|---|---|
| `WS /ws/speak` | One request per connection. The client sends one text frame: `{"type": "speak", "text": "...", "voice": "sohee", "instruct": "", "language": "Korean"}`. `voice` is one of the checkpoint's speakers, case-insensitive; the default is `sohee`. `instruct` is an optional style instruction. `language` is a checkpoint language or `Auto`; the default is `Korean`. Optional `temperature` (0.05–2.0), `top_k` (1–2048) and `seed` (≥ 0) override the server's sampling for this request; `temperature` / `top_k` set both stages, then optional `sub_temperature` / `sub_top_k` set the code predictor (codebooks 1–15) alone. A lower first-codebook temperature gives an evener delivery, but keep the code predictor near 0.9: 0.4 on both stages made comma lists lose their place (2026-10-19, below). Any other field is ignored. The server answers with binary frames of raw PCM16 little-endian mono 24 kHz, without a header: 7680 samples (4 codec frames, 320 ms) each, the last one shorter. It ends with the text frame `{"type": "done", "audio_ms": N, "first_ms": N, "total_ms": N}` (plus `"stopped": "silence"` and `"dropped_ms": N` when the silence rule below acted). The times are measured from the request. An error sends `{"type": "error", "message": "..."}` and then `done`. Empty text sends only `{"type": "done", "audio_ms": 0}`. **Closing the connection stops the generation before its next frame:** the socket is read while generating. |
| `GET /health` | `{"ok": true, "speakers": [...], "tts_impl": "hive-qtts", "vram_process_mb": ..., "max_seq": ...}`. `ok` stays false until the engine is loaded, its CUDA graphs are captured and one warm-up sentence has been synthesized. |

One generation runs at a time. Further connections wait in arrival order, and a client that leaves while waiting is skipped. The
WebSocket layer is a small RFC 6455 implementation on asyncio streams, with no extensions and no subprotocols, so the server needs only the
standard library, `ctypes` and `transformers` (for the checkpoint's Qwen2 tokenizer).

**Language `Auto`** works as in the reference: the codec prompt then carries no language id (`codec_nothink, think_bos, think_eos`). Under
`Chinese` or `Auto`, the dialect speakers (`eric`, `dylan`) use their dialect's id. Dialect names are not accepted as languages.

## Model

| Part | Shape |
|---|---|
| Talker | Qwen3 decoder: 28 layers, width 2048, 16 query / 8 KV heads × 128, q / k RMSNorm per head, SwiGLU 6144, RMSNorm eps 1e-6. The "interleaved mRoPE" (24/20/20) has equal positions on its three axes, so it reduces to rotate-half RoPE (θ 1e6). Inputs are embeddings: text rows are `text_projection(text_embedding[id])` (Linear + bias, SiLU, Linear + bias), codec rows are `codec_embedding[id]` (3072 ids). `codec_head` 2048 → 3072 gives the first codebook. |
| Prompt (non-streaming text mode) | [instruct rows] + 3 role rows + (tts_pad × n, tts_bos) + (think, think_bos, language, think_eos, speaker, codec_pad) + (text rows, tts_eos) + codec_pad + (tts_pad + codec_bos). The text is the content of `<\|im_start\|>assistant\n{text}<\|im_end\|>\n<\|im_start\|>assistant\n`. |
| Code predictor | 5 Qwen3 layers (width 1024, 16 / 8 heads × 128, SwiGLU 3072) on `small_to_mtp(·)` (2048 → 1024 + bias) of [talker hidden, codec_embedding[c0]], then one row per further codebook (15 embedding tables of 2048 × 2048); `lm_head[j]` gives codebook j + 1. |
| Next talker input | codec_embedding[c0] + Σ predictor_embedding_i[c_{i+1}] + tts_pad. |
| Sampling | First codebook: repetition penalty 1.05 over the distinct first codes so far (logit > 0 ? / p : × p); ids 2048–3071 except EOS 2150 suppressed; EOS also suppressed for the first 2 frames; temperature 0.9; top-k 50 (ties at the k-th value kept). Other codebooks: temperature 0.9, top-k 50. |
| Codec decoder | RVQ (1 + 15 codebooks of 256, 1×1 projections to 512) → causal conv k3 → Linear → 8 transformer layers (16 heads × 64, RoPE θ 1e4, **causal sliding window of 72 frames**, layer scales, RMSNorm eps 1e-5) → Linear → 2 × (causal transposed conv k2 s2 + ConvNeXt) → causal conv k7 → 4 blocks (SnakeBeta, causal transposed conv k = 2s for s = 8, 5, 4, 3, residual units with dilations 1, 3, 9) → SnakeBeta → causal conv k7 → clamp(−1, 1). 1920 samples per 80 ms frame. |

The math contract is the docstring of `oracle/qwen3_tts_oracle.py`. That file is this project's own fp32 PyTorch model, written from the
model's math; its `reference` sub-command runs the upstream `qwen_tts` package (when installed) only to cross-check it.

## Engine (`engine/src/qtts/`, C API `engine/include/hive/qtts/qtts_capi.h`)

- **Numerics:** the talker, code predictor and codec weights are bf16. Norms, biases, layer scales, SnakeBeta parameters and the RVQ
  codebooks are fp32. Activations and residual streams are fp32; GEMM inputs are bf16 with fp32 accumulation. The KV cache is bf16
  (talker, predictor) or fp32 (codec window).
- **Prompt:** the text embedding table (151936 × 2048, 594 MiB) **stays in host memory** (the mmapped checkpoint). Its rows are gathered
  on the host, then projected and assembled on the GPU. The talker prompt runs through cuBLAS GEMMs with fused RoPE / KV-store and causal
  attention kernels.
- **Frame step = one CUDA graph:** the 16 code-predictor rows (input projection, 5 layers, head, sampling, each row), the frame embedding, one
  talker decode step and the sampling of the next frame's first code. About 700 kernels; position, frame count, RNG counter and the
  per-request sampling settings live in a device `Ctl` block, so the graph is captured once at start-up. The decode GEMVs are this
  project's own (two weight rows per warp, 16-byte non-allocating loads, staged input with an optional fused RMSNorm, fused bias /
  residual / layer-scale / SiLU·up epilogues). Decode attention splits the keys over 16 blocks and merges the partials. Sampling runs in
  one block on the GPU: exact top-k threshold by radix select, then Gumbel-max with a counter-based RNG (seed, call, id).
- **Codec chunk = a second CUDA graph:** 4 frames per chunk. Every causal conv keeps its last (k − 1)·dilation input rows, every
  transposed conv its last input row, and the transformer keeps a ring of 128 positions holding the 72-frame window. A stream of chunks
  therefore equals a whole-sequence decode, unlike the replaced server, which re-decoded 25 frames of left context per chunk. Convs are
  im2col (with SnakeBeta fused into the gather) + cuBLAS; the transformer uses the same GEMV kernels with 4 rows.
- **Streaming:** `qtts_next_chunk` runs up to 4 frame graphs, checking the cancel flag before each one, then the codec graph, and returns
  7680 samples (fewer at the end). The server starts the next chunk while it sends the current one.
- **Memory:** everything is allocated in `qtts_create`: weights, KV cache for `max_seq` talker positions, work buffers, cuBLAS workspace and
  graphs. No device memory is allocated per request. The start-up plan uses the context measured as NVML free before minus CUDA free after
  creation. If `HIVE_VRAM_LIMIT_MB` is set and the plan exceeds it, `max_seq` is lowered in steps of 256. If even the minimum does not fit,
  creation fails with the shortfall. The per-thread stack limit is lowered to 256 B after context creation: the default 1 KiB reservation
  cost ~210 MiB; the kernels use ≤ 32 B, and the driver raises the limit by itself if a launch needs more.

| VRAM (max_seq 2048) | MiB |
|---|---|
| CUDA context (measured) | 343 |
| weights (talker layers 2,688 · code predictor 334 · codec 235 · codec embedding / head / text projection 40; the text embedding stays on the host) | 3,297 |
| KV caches + work buffers | 308 |
| cuBLAS workspace + handle, graphs, allocator rounding | ~20 |
| **process total (nvidia-smi, peak under load)** | **3,968** |

## Build and run

```sh
# library + GPU tests (inside the build image)
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build --target hive_qtts test_qtts_kernels test_qtts_engine
# service: container hive_qtts, port 8443 (config/qtts.env); dev instance on another port / container
HIVE_MODEL=tts scripts/hive-start.sh --wait                       # or HIVE_CKPT=<checkpoint dir> scripts/hive-start.sh
HIVE_MODEL=tts HIVE_PORT=18443 HIVE_CONTAINER=hive_qtts_dev scripts/qtts-start.sh --wait
HIVE_MODEL=tts scripts/hive-stop.sh
# directly
python3 server/speech/tts_server.py --model <checkpoint dir> --port 8443 [--lib engine/build/libhive_qtts.so] [--vram-limit-mb N] [--max-seq N]
```

Settings (`config/qtts.env`, overridable by the environment): `HIVE_CONTAINER` (hive_qtts), `HIVE_PORT` (8443), `HIVE_BIND`
(127.0.0.1), `HIVE_QTTS_MAX_SEQ` (2048 talker positions ≈ 2 min 40 s of audio per request; 112 KiB of KV each), `HIVE_VRAM_LIMIT_MB`,
`HIVE_GPU`. **Length cap:** a request may produce at most 20 + 8 frames per text token (`--max-frames-base`, `--max-frames-per-token`;
0 = off; logged as `[tts] length cap` when hit). The model, the reference implementation included, sometimes misses EOS on very short inputs
and keeps babbling: the reference did so on 5 of 11 seeds for "あ。" and 2 of 30 for the one-syllable Korean "a." (ah). Normal speech stays at most 4.2 frames per
token, so the cap leaves at least 2× headroom and ends a runaway 2-token take after 2.9 s. Profiles: `HIVE_MODEL=tts` with `HIVE_VRAM_GB` 8–96 (`config/profiles/tts*.env`). Server-only flags: `--seed N` (fixed
sampling seed, tests), `--temperature`, `--top-k`, `--rep-penalty`.

**Silence:** when the model loses its place or misses EOS it can keep emitting silent frames up to the length cap. After speech, silent
chunks (RMS ≤ `--silence-rms`, 328 in int16 units) are held; a pause sends at most `--silence-keep-ms` (960) of them, and
`--silence-stop-ms` (3200; 0 = off) of silence — or that long before any speech — ends the take (`[tts] silence stop` in the log).
Natural pauses measured ≤ 0.9 s in sentences and ≤ 1.0 s in calm comma lists.

**2026-10-19 sampling measurement** (a voice client, the CustomVoice checkpoint with two added Korean voices A and B, a calm instruct;
each take transcribed by Qwen3-ASR and the list items counted; comma lists of the Korean numbers one … ten and eleven … twenty and the
seven weekdays × 6 takes = 162 items per voice):

| first codebook / code predictor | voice A items | voice B items | silence stops | pitch spread of one sentence over 4 takes (semitones, A / B) |
|---|---|---|---|---|
| 0.4 / 0.4 (the client's setting until then) | 111 / 162 | 129 / 162 | 7 / 7 | 1.11 / 1.68 |
| 0.7 / 0.9 | 160 / 162 | 162 / 162 | 0 / 1 | 1.12 / 1.63 |
| 0.9, top-k 20 / 0.9 | 162 / 162 | 161 / 162 | 0 / 0 | 1.74 / 1.51 |
| 0.7 / 0.7 (a client sending `sub_temperature` to an older server) | 162 / 162 | — | 0 | 1.30 |

At 0.4 / 0.4 the failing takes speak the first items and then only silence until the cap ("eleven, twelve." and 19 s of silence).
Without the instruct and at 0.9 / 0.9 the lists come out whole too (107 / 108 in a 4-take run).

## Tests

| Test | What |
|---|---|
| `engine/tests/test_qtts_kernels.cu` (GPU, < 200 MiB) | GEMV in every mode / prologue / row count, q/k norm + RoPE + KV store, split and unsplit decode attention, prefill attention, sampling (penalty, suppression, EOS rules, top-k membership and frequencies), causal im2col with SnakeBeta, ConvNeXt front, codec window attention (wrapped ring), codec output layer, RVQ — against CPU references |
| `engine/tests/test_qtts_engine.cu` (GPU, ~4 GB) | `forced` (teacher-forced logits for `oracle compare`), `decode` (codec only, ~0.5 GB), `generate`, `graphcheck` (graph == eager, bit for bit), `bench` |
| `tools/test_qtts_server_cpu.py` (CPU, in `tools/test_cpu.sh`) | the `/ws/speak` protocol, barge-in, queueing, `/health`, voice / language resolution — fake engine |
| `oracle/qwen3_tts_oracle.py` | `generate` (greedy / seeded / forced), `decode` (whole sequence vs chunked stream), `reference` (upstream package), `compare` |

## Limits and open points

- One generation at a time per process (the replaced server had the same limit). Batching several voices would need per-request state
  in the graphs.
- Only the CustomVoice checkpoint's text mode (`non_streaming_mode`, what the replaced server used) is implemented. Voice cloning (the
  Base checkpoint with reference audio) and VoiceDesign are not.
- Sampling is statistically the reference's, but not the same random stream; equal seeds reproduce the same audio within this engine.
- The first-chunk latency includes one 4-frame chunk (as before). Chunks of 2 frames would cut it further at a small throughput cost; this
  is not measured yet.
