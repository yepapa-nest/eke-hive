# Gemma 4 12B in Eke Hive

> **Status (2026-10-18; measurements updated 2026-10-19):** validated end to end on the GPU against a reference server running the same checkpoint (SGLang, fp8 KV,
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

**2026-10-19 (KV pool split, short-prefill attention, cluster GEMM):** the same chat harness on the served model beside the 72 GB
DeepSeek: first token p50 / p95 45 / 63 ms, first sentence 90 / 131 ms (from 93 / 150), delegation 1.00; one-token turn judgment
94.3 % · 87.5 % · 100 %, 60 ms with a maximum of 67 (from 69 ms, maximum 411); tool battery unchanged. The client's system prompt was
revised the same morning to delegate less; with it the decode median is 301 tok/s and the Korean-argument case delegates 4 of 10
times; with the earlier prompt, on this build, 334 tok/s and 10 of 10 — the engine change leaves both alone. A conversation turn's
tail after 3,000 tokens (`tests/gemma_tail.cpp`, medians of 20): 70 rows 28.9 → 19.4 ms, 128 rows 36.6 → 29.4 ms. Two conversations
taking turns on the served model (client-side first token, streaming): p50 213 → 20 ms, maximum 850 → 22 ms.

**Images (2026-10-18):** image parts are served through the checkpoint's encoder-free vision embedder (below). Measured on the GPU
against the fp32 oracle (`oracle/gemma4_oracle.py --images`, whose embedder matches the Hugging Face module to a relative 2e-7) on three
images through the server's own preprocessing: teacher-forced argmax agrees at 58 of 59 steps (the miss is a near-tie of 0.16 logits).
The server's preprocessing gives the same token ids, soft-token counts and positions as the checkpoint's Hugging Face processor; pixels
differ by at most 1.5 levels of 255 (bicubic resampler). Text requests are unchanged: on a text prompt every logit (except the two
suppressed markers) and 64 decoded tokens with their drafts are bit-identical to the build without vision. Under the voice set's
13,000 MiB limit the embedder costs 102 MiB of the KV pool (1,838 → 1,736 MiB); the process stays at 12,992 MiB while serving images.
Image requests on the served model (stream, 10 different pictures of 300 × 200 … 1920 × 1080, ~290 prompt tokens, preprocessing
included): first token p50 90 ms (max 169, the first request), decode 276 tok/s; the next text turn of the same conversation 71 ms; two
images in one request (559 tokens) 128 ms. The chat harness above with the embedder loaded: first token 45 / 61 ms, first sentence
93 / 150 ms, 320 tok/s, tool battery unchanged.

Eke Hive's fifth model family is **Gemma 4 12B Unified** (config.json `model_type` `gemma4_unified`), served for **text and images** from the
quantization-aware-trained checkpoint `gemma-4-12B-it-qat-w4a16-ct` (compressed-tensors int4, ~9.6 GB), with the Frozen-KV MTP assistant
`gemma-4-12B-it-qat-q4_0-unquantized-assistant` (bf16, 0.84 GB) as the drafter. It is a small real-time chat model: short prompts, a few
conversations at once, first-token latency matters more than long-context throughput. The checkpoint has no vision tower: its encoder-free
vision embedder (~95 MiB) is loaded; the audio projection is not.

## Model and checkpoint

| Part | Shape | Format |
|---|---|---|
| 40 sliding-window layers | 16 query / 8 KV heads × 256, window 1024, RoPE θ 1e4 (all dims) | q / k / v / o **int4 group 32** |
| 8 full-attention layers (5, 11, …, 47) | 16 query heads × 512 / **1 KV head**, `attention_k_eq_v` (no v_proj), proportional RoPE θ 1e6 on 128 of 512 dims | q / k / o **int4 group 32** |
| MLP, every layer | 3840 → 15360 → 3840, GELU-tanh gate | **int4 group 32** |
| norms | input, post-attention, pre-FF, post-FF (RMSNorm `x · w`), q / k norms per head, v norm without scale, a per-layer scalar | BF16 |
| embedding = lm_head (tied, the same bytes in the file) | 262144 × 3840, embedding scaled by bf16(√3840) | BF16 |
| logits | final RMSNorm, then `30 · tanh(l / 30)` | — |
| vision embedder | merged 48 × 48 patch (6912 values) → LayerNorm → Dense 6912 → 3840 (+ bias) → LayerNorm → + position rows (x, axis 0) and (y, axis 1) of a [1120, 2, 3840] table → LayerNorm → RMSNorm (no scale) → Linear 3840 → 3840 | BF16 |
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
- **Prefill GEMM** (`w4_gemm_multi` / `w4_gemm_gu`, 17 … `HIVE_GEMMA_GEMV_ROWS` rows, default 192): the GEMV's math bit for bit — the
  same K split over the block's warps, the same per-group integer tile and fp32 scaling, the same warp-order sum — so a row's result does
  not depend on the row count and equals the decode GEMV's. A warp owns 64 output rows × 16 or 32 tokens and stages the tokens'
  activations of its K range in shared memory (cp.async, ldmatrix); the GEMV it replaces re-read every token's activations for each
  16-row tile, and that read bounded it above ~16 rows. Per layer (q + k + v, o, gate + up, down; cold L2): 70 rows 788 → 305 µs,
  128 rows 1,382 → 477 µs. A conversation turn's tail of 70 rows after 3,000 tokens: 42.7 → 28.9 ms; 128 rows: 70.4 → 36.6 ms (`tests/gemma_tail.cpp`).
  Served (60-turn voice conversation, server-side first token while it is the only conversation): p50 64 → 31 ms.
  That kernel was bound by L2 → SM bytes (each 16-token block re-read the weights, each 64-row block the activations: ~1.2 GB a layer at
  70 rows; this card's L2 gives ~4–5.6 TB/s). The default now moves the K split from the warps of a block to the 8 blocks of a
  thread-block cluster: every block's warps share one K range, so one staged activation tile feeds 128 rows and a block covers up to 80
  tokens (the weights are read once); the ranks' fp32 partials are summed through distributed shared memory in rank order, which keeps
  the GEMV's bits. Per sliding layer (cold L2, 2026-10-19, medians beside the served models): 17 rows 209 → 125 µs, 40 rows 258 → 174,
  70 rows 385 → 250, 128 rows 520 → 387, 192 rows 711 → 553 (full layer 70 rows 414 → 264, 128 rows 551 → 412). The rank sum is now the
  largest remaining cost (~70–95 µs of a 70-row layer: distributed shared memory moves ~2.4 B a clock per SM on this card).
  Longer forwards dequantize each matrix into one bf16 scratch and multiply by cuBLAS (faster from ~256 rows).
- **KV cache** (`engine/include/hive/gemma/gemma_kernels.h`): fp8 e4m3 pages of 64 tokens, token-major (a page's filled part is a contiguous
  prefix, so a session image copies only what grew). Page tables hold page *pointers*: when the VRAM pool is exhausted and no idle session can
  be evicted, new pages come from mapped pinned host memory (`HIVE_GEMMA_HOST_KV_MB`) — slower, never an out-of-memory. Sliding-window layers
  release pages behind `window + HIVE_GEMMA_SWA_KEEP` positions. The sliding pool holds the windows of `HIVE_GEMMA_SWA_SEQS` sequences
  (a sliding page is 20× a full one: 40 sliding layers against 8 full), as far as the full pool keeps `HIVE_GEMMA_FULL_MIN_TOKENS`
  positions; the full pool takes the rest. Sizing both for one max-length context instead left room for about one voice conversation in
  the sliding pool, so two conversations taking turns evicted each other (first token 260–500 ms against 30–70). The keep margin covers
  how far a next turn rewinds: over 2,339 served follow-ups the rewind was 44 tokens at the median, 215 at p95 and 600 at p99.5.
  A session moved to a host image by the KV reclaim is resumed and rewound the same way when it returns.
- **Host memory for session images** (`src/gemma/gemma_runtime.cpp`, `ImageArena`): a save whose new bytes are 8 MiB or more takes a
  block from a background-populated arena (pages faulted in off the request path) when the block is at most twice the bytes; otherwise
  the save allocates its own bytes. Block sizes follow the latest save in 32 MiB steps (256 MiB … 1 GiB), and ready blocks more than
  twice that size are released. The session budget (`--host-session-mb`) counts the image bytes, so this keeps the process RSS close to
  the budget: before, any save took any ready block and the block size only grew with the largest image seen, and a few hours of voice
  conversations left VmRSS at 59.9 GB against 16.96 GB of counted images. Same workload on fresh daemons (one 33K-token prompt, then
  twelve conversations × 8 turns with 1.5 s between turns, 2026-10-19): RSS beyond the counted images 5.96 GB → 0.60 GB (the ready
  blocks). `HIVE_GEMMA_TIMING=1` logs each arena decision (`[gemma-timing] image arena: …`).
- **MTP** (Frozen-KV assistant): the seed is the target's sampled next token (embedded with the target embedding × √3840) together with the
  final-normed hidden of the last processed row; every chain step runs at the rope position of the last written KV slot, reads the target's
  last sliding and last full layer KV (keys 0 … that slot), and feeds back `post_projection(norm(z))`.
- **Drafter sampling**: argmax and Σ exp(l − max) over the 262,144-entry vocabulary in two fixed-order passes (64 column blocks, then one
  warp per row).
- **MTP gate**: `config/gemma.env` turns on the measured gate (`HIVE_MTP_GATE2/3`, `docs/configuration.md`) and batch speculation; the
  default gate's fixed table belongs to DeepSeek and never let a 6.5 ms Gemma step draft.
- **No layer yield**: the engine has one set of forward buffers, so a decode inside a prefill's layer boundary would overwrite the
  prefill's residual; prompts are prefilled in chunks between decode steps instead.
- **Images** (`GemmaImage`, `prefill(…, images)`): an image's soft tokens are prefill rows whose embedding is the vision embedder's output
  (no embedding scale) instead of the placeholder token's. The embedder runs per image on the rows of its chunk (two cuBLAS GEMMs, three
  LayerNorms and an RMSNorm). The checkpoint's `use_bidirectional_attention: "vision"` means an image's soft tokens see each other both
  ways: every attention row carries the last key it may see (`AttnSeq::qlim` — the span's last position for an image row, its own
  position otherwise), on the prefill and the decode / verify kernel alike; a prefill chunk never ends inside an image span. The daemon's
  image rules (session signatures of the patch bytes, no prefix reuse across a different picture) are the shared ones of `hived`.
- **Suppressed tokens**: the end-of-image / end-of-audio markers (generation_config `suppress_tokens`) get logit −inf in every next-token
  row, as the reference generation does. They are input-only; on a plain text prompt the end-of-audio marker still scored 16.3.
- **VRAM limit**: the engine sizes its KV pool from `HIVE_VRAM_LIMIT_MB` (limit − weights − work buffers − context and cuBLAS estimate −
  `HIVE_CACHE_RESERVE_MB`), and the daemon keeps the reserve for the process lifetime (like DeepSeek, see `docs/configuration.md`).

## API

The usual `/v1/chat/completions` (streaming, tools, `chat_template_kwargs.enable_thinking`, `top_k`, image parts — `image_url` data: or
http(s) URLs, Anthropic-style `image` sources, also inside tool results).
Images are preprocessed by the server (`server/families/gemma.py`, from `processor_config.json`): aspect ratio kept, the largest height and
width that are multiples of 48 with at most 280 · 48² pixels (bicubic), cut into 48 × 48 blocks in raster order (values / 255, no
normalization); the prompt's `<|image|>` becomes `<|image>` + one `<|image|>` per block (at most 280) + `<image|>`. Audio and video
parts are refused. The prompt is the checkpoint's chat
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
| `HIVE_GEMMA_SWA_KEEP` | 768 | sliding-layer positions kept behind the window (how far a next turn can rewind the live state) |
| `HIVE_GEMMA_SWA_SEQS` | 4 | sequences whose sliding window the KV pool holds |
| `HIVE_GEMMA_FULL_MIN_TOKENS` | 65536 | full-layer positions the pool keeps at least (capped at the context length) |
| `HIVE_GEMMA_KV_MB` | from the limit | KV page pool size (overrides the limit-based size) |
| `HIVE_GEMMA_HOST_KV_MB` | 8192 | mapped host memory for KV pages when the pool is full |
| `HIVE_GEMMA_CHAT_TEMPLATE` | the checkpoint's | chat template file |
| `HIVE_GEMMA_TOOL_THOUGHT` | 1 | the tool-response branch above |
| `HIVE_GEMMA_CANDS` | ≤ 256 | sampler candidates per row |
| `HIVE_GEMMA_GEMV_ROWS` | 192 | forwards of up to this many rows use the int4 kernels (the GEMV up to 16 rows, the prefill GEMM above); longer ones dequantize for cuBLAS |
| `HIVE_GEMMA_GEMM_V` | unset | force one prefill GEMM variant (0 … 8, `src/gemma/gemma_w4.cu`); unset = 6 (the cluster kernel; 0 … 5 the former kernel, 7 / 8 the single-block form). Every variant gives the same bits |
| `HIVE_GEMMA_VISION` | 1 | 0 = do not load the vision embedder (image requests are then refused); it takes ~102 MiB under a VRAM limit |
| `HIVE_GEMMA_ATTN_SPLIT_Q` | 48 | sliding-layer sequences of up to this many query rows use the key-split attention (16 = decode / verify only) |
| `HIVE_GEMMA_ATTN_SMALL_M` | 128 | sequences of up to this many query rows get 16-row attention tiles (0 = the wide tiles) — a 70-row turn tail: sliding 49.6 → 24.2 µs, full 156.5 → 95.5 µs per layer |
| `HIVE_GEMMA_NO_PDL` | 0 | 1 = launch the forward's kernels without programmatic dependent launch (A/B) |
