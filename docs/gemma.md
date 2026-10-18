# Gemma 4 12B in Eke Hive

> **Status (2026-10-18):** validated end to end on the GPU against a reference server running the same checkpoint (SGLang, fp8 KV,
> MTP 3) with that server's own client harnesses, both measured back to back on the same card. Greedy decoding agrees with the reference
> server's argmax on 35 of 37 teacher-forced steps (the two misses are near-ties of 0.17 and 0.38 logits; the fp32 oracle picks the same
> token as this engine at one of them).

| Measure (real chat harness, 30 requests, streaming, tools on) | Eke Hive | reference server |
|---|---|---|
| first token p50 / p95 | **42 / 75 ms** | 69 / 100 ms |
| first sentence p50 / p95 | **96 / 165 ms** | 179 / 276 ms |
| decode (median per request) | **303 tok/s** | 181 tok/s |
| delegation accuracy · first tool call · Korean arguments | 1.00 · 20/20 · 10/10 | 1.00 · 20/20 · 10/10 |
| one-token turn judgment (70 / 32 held-out / 30 barge-in) | 94.3 % · **93.8 %** · 100 %, 60 ms | 94.3 % · 87.5 % · 100 %, 64 ms |

One difference: `tool_choice: "required"` is not enforced (the request is served as `auto`).

Eke Hive's fifth model family is **Gemma 4 12B Unified** (config.json `model_type` `gemma4_unified`), served on its **text path** from the
quantization-aware-trained checkpoint `gemma-4-12B-it-qat-w4a16-ct` (compressed-tensors int4, ~9.6 GB), with the Frozen-KV MTP assistant
`gemma-4-12B-it-qat-q4_0-unquantized-assistant` (bf16, 0.84 GB) as the drafter. It is a small real-time chat model: short prompts, a few
conversations at once, first-token latency matters more than long-context throughput. The vision and audio towers in the checkpoint are not
loaded.

## Model and checkpoint

| Part | Shape | Format |
|---|---|---|
| 40 sliding-window layers | 16 query / 8 KV heads × 256, window 1024, RoPE θ 1e4 (all dims) | q / k / v / o **int4 group 32** |
| 8 full-attention layers (5, 11, …, 47) | 16 query heads × 512 / **1 KV head**, `attention_k_eq_v` (no v_proj), proportional RoPE θ 1e6 on 128 of 512 dims | q / k / o **int4 group 32** |
| MLP, every layer | 3840 → 15360 → 3840, GELU-tanh gate | **int4 group 32** |
| norms | input, post-attention, pre-FF, post-FF (RMSNorm `x · w`), q / k norms per head, v norm without scale, a per-layer scalar | BF16 |
| embedding = lm_head (tied, the same bytes in the file) | 262144 × 3840, embedding scaled by bf16(√3840) | BF16 |
| logits | final RMSNorm, then `30 · tanh(l / 30)` | — |
| MTP assistant | `pre_projection` 7680 → 1024, 4 KV-shared layers (3 sliding + 1 full, hidden 1024, 16 heads), final norm, `post_projection` 1024 → 3840, own tied head 262144 × 1024 | BF16 |

int4 group 32 (compressed-tensors `pack-quantized`, symmetric): `weight_packed` int32 [N, K / 8] holds 8 values per word, low nibble first,
value = nibble − 8; `weight_scale` bf16 [N, K / 32]; W[n, k] = (nibble − 8) · scale[n, k / 32]. Attention uses no 1/√d scaling (the q / k
norms take its place).

The math contract is the docstring of `oracle/gemma4_oracle.py` (this project's own PyTorch code, written from the math).

## Engine

- **Decode / verify GEMV** (`engine/src/gemma/gemma_w4.cu`): the weight is the mma A operand (16 output rows), the activations the B operand
  (8 or 16 token rows). Per 32-input group the integer products accumulate in a fresh fp32 tile that is scaled by the group's bf16 scale in
  fp32 — the scale is never rounded into the weight. Weights are repacked at load into the fragment order (each lane reads exactly its
  fragments, 16 contiguous bytes per two groups), so a warp reads its tile's bytes once. A row's result never depends on how many rows share
  the call (the K split is a function of the matrix shape only).
  A block owns one 16-row tile; its warps split K evenly and their partials are summed in shared memory in warp order (a fixed order that
  depends on K only). A warp issues the loads of up to 8 group pairs before the mma work. q / k / v of a layer run as one launch.
- **Programmatic dependent launch** (`src/gemma/gemma_pdl.cuh`): every kernel of a forward starts the next one early; a GEMV bulk-prefetches
  its weights into L2 before waiting for its input, so the weight stream does not stop at kernel boundaries (measured over a chain of 8
  layers: 87 µs per layer, 1.45 TB/s, against 93.5 µs without it; the card's plain cold read peaks at 1.6 TB/s). `HIVE_GEMMA_NO_PDL=1`
  launches plainly (outputs are identical).
- **Norms**: rows are read as 16-byte vectors held in registers; each residual add is fused with the norm that reads its result (the
  pre-feedforward norm, or the next layer's input norm).
- **Prefill**: up to `HIVE_GEMMA_GEMV_ROWS` rows (128) the decode GEMV runs in passes of 16 rows; longer forwards dequantize each matrix
  into one bf16 scratch and multiply by cuBLAS.
- **KV cache** (`engine/include/hive/gemma/gemma_kernels.h`): fp8 e4m3 pages of 64 tokens, token-major (a page's filled part is a contiguous
  prefix, so a session image copies only what grew). Page tables hold page *pointers*: when the VRAM pool is exhausted and no idle session can
  be evicted, new pages come from mapped pinned host memory (`HIVE_GEMMA_HOST_KV_MB`) — slower, never an out-of-memory. Sliding-window layers
  release pages behind `window + HIVE_GEMMA_SWA_KEEP` positions.
- **MTP** (Frozen-KV assistant): the seed is the target's sampled next token (embedded with the target embedding × √3840) together with the
  final-normed hidden of the last processed row; every chain step runs at the rope position of the last written KV slot, reads the target's
  last sliding and last full layer KV (keys 0 … that slot), and feeds back `post_projection(norm(z))`.
- **Drafter sampling**: argmax and Σ exp(l − max) over the 262,144-entry vocabulary in two fixed-order passes (64 column blocks, then one
  warp per row).
- **MTP gate**: `config/gemma.env` turns on the measured gate (`HIVE_MTP_GATE2/3`, `docs/configuration.md`) and batch speculation; the
  default gate's fixed table belongs to DeepSeek and never let a 6.5 ms Gemma step draft.
- **No layer yield**: the engine has one set of forward buffers, so a decode inside a prefill's layer boundary would overwrite the
  prefill's residual; prompts are prefilled in chunks between decode steps instead.
- **VRAM limit**: the engine sizes its KV pool from `HIVE_VRAM_LIMIT_MB` (limit − weights − work buffers − context and cuBLAS estimate −
  `HIVE_CACHE_RESERVE_MB`), and the daemon keeps the reserve for the process lifetime (like DeepSeek, see `docs/configuration.md`).

## API

The usual `/v1/chat/completions` (streaming, tools, `chat_template_kwargs.enable_thinking`, `top_k`). The prompt is the checkpoint's chat
template (or `HIVE_GEMMA_CHAT_TEMPLATE`) with one added branch (`HIVE_GEMMA_TOOL_THOUGHT`, default on): after a tool response with thinking
off, the generation prompt closes an empty thought channel, as it does after a user turn. Tool calls are parsed from the model's own syntax
(`<|tool_call>call:NAME{key:value,…}<tool_call|>`, strings delimited by `<|"|>`), thinking from the `<|channel>thought … <channel|>` channel.
Sampling fields a request leaves out default to the model's `generation_config.json` (temperature 1.0, top_p 0.95, top_k 64).

## Configuration

`config/gemma.env` (`HIVE_MODEL=gemma`; `config/profiles/gemma*.env`):

| Variable | Default | Meaning |
|---|---|---|
| `HIVE_GEMMA_ASSISTANT` | unset | directory of the MTP assistant checkpoint (inside the container); unset = no drafter |
| `HIVE_GEMMA_MTP_K` | 3 | drafts per step (0 = off) |
| `HIVE_GEMMA_CHUNK` | 1024 | prefill rows per forward |
| `HIVE_GEMMA_SWA_KEEP` | 2048 | sliding-layer positions kept behind the window |
| `HIVE_GEMMA_KV_MB` | from the limit | KV page pool size (overrides the limit-based size) |
| `HIVE_GEMMA_HOST_KV_MB` | 8192 | mapped host memory for KV pages when the pool is full |
| `HIVE_GEMMA_CHAT_TEMPLATE` | the checkpoint's | chat template file |
| `HIVE_GEMMA_TOOL_THOUGHT` | 1 | the tool-response branch above |
| `HIVE_GEMMA_CANDS` | ≤ 256 | sampler candidates per row |
| `HIVE_GEMMA_GEMV_ROWS` | 128 | forwards of up to this many rows use the int4 GEMV in passes of 16 rows; longer ones dequantize for cuBLAS |
| `HIVE_GEMMA_NO_PDL` | 0 | 1 = launch the forward's kernels without programmatic dependent launch (A/B) |
