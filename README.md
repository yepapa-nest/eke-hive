# Eke Hive

**A from-scratch inference engine for two big mixture-of-experts models — DeepSeek-V4.1-Flash and GLM-5.3-Flash — on one
96 GB GPU, with the experts that do not fit offloaded to CPU RAM.**

> *eke out* — to make a small supply go a long way. In beekeeping, an *eke* is the extra ring added to a hive
> when it runs out of room. Eke Hive does both: it adds RAM and SSD tiers under a single GPU and squeezes the
> most out of each.

This is not a general-purpose engine. It is built for exactly two checkpoints and one kind of machine, and it was
pieced together from what we had: one RTX PRO 6000 Blackwell (96 GB), an older 32-core Threadripper PRO 3975WX (Zen 2),
1 TB of 8-channel **DDR4** and **PCIe 4.0**. No datacenter, no NVLink, no second GPU. Every number below comes from that
box — on DDR5 and PCIe 5.0 the CPU-side and transfer-side work would get faster (see the hardware note).

How the work is split:

- dense weights, attention state and a cache of the hottest experts live in **VRAM** (the cache fills whatever is left of the 96 GB);
- all routed experts (and DeepSeek's Engram tables) live in pinned **RAM**, split across NUMA nodes;
- experts that miss the VRAM cache are computed by the **CPU** in place, or copied over PCIe to the GPU;
- optionally, DeepSeek's ~195 GB Engram value tables stay on **NVMe** and are read row by row.

DeepSeek-V4.1-Flash is the main model; GLM-5.3-Flash is the second option — slower here, but a good model to have
on the same box. Both measured with the same real-chat benchmark (`tools/bench_chat.py`: twelve different
chat prompts — 5 Korean, 3 English, 2 code, 2 creative — answers up to 600 tokens, thinking off, decode speed after
the first token; [docs/benchmarks.md](docs/benchmarks.md#real-chat-benchmark-headline)):

| | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [GLM-5.3-Flash](docs/glm.md) |
| --- | --- | --- |
| Checkpoint used | original (MXFP4 experts, FP8 dense, Engram) | NVIDIA NVFP4 + Z.ai's FP8 attention / shared experts |
| Inputs | text and images | text, images and video |
| Thinking off | yes (`reasoning_effort: none`) | no — the model always thinks; "off" maps to its lowest effort |
| Context | 262,144 | 262,144 |
| Decode, 1 stream | **85.3 tok/s** (Korean 87.2 · English 82.7 · code 106.2 · creative 62.4) | 70.8 tok/s (75.5 · 67.9 · 73.9 · 54.4) |
| Decode, 2 / 4 streams (total) | **92.9 / 124.6 tok/s** | 87.8 / 103.9 tok/s |
| Decode, 8 / 16 / 32 streams (total) | **175.0 / 178.1 / 181.5 tok/s** (8 at a time) | 130.6 / 130.3 / 130.9 tok/s (8 at a time) |
| Time to first token, 1 stream | **0.23 s** | 0.51 s |
| Time to first token, 17K / 42K / 54K prompt | **5.05 / 7.53 / 9.69 s** | 7.4 / 11.7 / 14.0 s |
| Time to first token, 100K / 200K / 250K prompt | — | 26.4 / 53.7 / 66.3 s |
| Quality suite (bundled) | 175 / 179 | 158 / 163 (thinking at its lowest level) |
| Sleep / wake (level 3, a 120K conversation open, GPU memory back to ~1 GB) | 0.7 s / 3.2 s | 3.8 s / 3.0 s |
| Cold start to ready | 88 s | 95–97 s |
| Host RAM in use | ~480 GB (RAM mode) · ~300 GB (Engram on SSD) | ~385 GB (358 GiB) |

### Why DeepSeek runs faster than GLM here

Both models get the same treatment, but on this machine DeepSeek comes out ahead, for measured reasons:

- **Maturity.** The DeepSeek path went through about twenty measured optimisation steps (fused decode kernels, early routing,
  CUDA graphs, a CPU/PCIe split for cache misses, tuned batching — [docs/performance.md](docs/performance.md)). The GLM path is new.
- **Where the time goes.** Only about 39 % of GLM's experts fit in VRAM. Before cache-aware routing about half of a GLM decode
  step was the CPU computing the missed experts (at 77–91 GB/s, bound by this DDR4); with it the CPU share is
  under a quarter, and what remains is mostly GPU work outside the experts — 34 linear-attention (KDA) layers with BF16
  projections, and the hyper-connection mixing ([docs/glm.md](docs/glm.md)).
- **PCIe is already busy.** GLM's cache promotions and next-layer prefetches already send expert records (13.5 MiB each)
  over PCIe 4.0 on every decode step, so its misses are not also streamed to the GPU the way DeepSeek's are.
- **The same two routing changes paid off more on DeepSeek.** Cache-aware routing and expert deferral
  ([docs/configuration.md](docs/configuration.md)) took DeepSeek from 53.8 to 76.9 tok/s for one stream and from 100.2 to
  173.0 for eight (the same benchmark, before the later decode-kernel step).
- **Speculative decoding helps one or two streams.** GLM's draft layer gives 2.5 tokens per step for a single user, and with two
  users both drafts are verified in one step (`HIVE_MTP_BATCH`); with more users at once each step is plain decoding bound by the CPU.
- **Thinking cannot be switched off** on GLM-5.3-Flash, so a "no thinking" request still produces some reasoning tokens.

If you can only run one, run DeepSeek. GLM is there because it is a good model and the engine can serve it.

### Sharing the GPU: keep the model in RAM, hand the card over

The weights never leave host RAM, so the GPU can be lent out and taken back in seconds. `HIVE_SLEEP_LEVEL=3 scripts/hive-sleep.sh sleep`
drains running requests, moves the dense weights, sessions and work buffers to RAM and frees the expert cache — about 1 GB
of VRAM stays in use (the script's default, level 1, frees only the expert cache). `wake` brings it back and refills the expert cache in ~3 s; open conversations
continue where they were (measured with a 120K-token conversation open: the next turn after the wake answered correctly
with its first token after 0.6 s on DeepSeek and 2.6 s on GLM, the whole history reused). We use this every day to alternate between one of the two big models and a **media stack plus a
smaller model (for example a 27B model next to image and video generation)** on the same card: put Eke Hive to sleep,
run the media job, wake it up — no reload from disk (a cold start takes 60–100 s). Details: [docs/install.md](docs/install.md#sleep-and-wake-sharing-the-gpu).

> **Hardware note.** Everything here was measured on 8-channel DDR4 (Zen 2) with PCIe 4.0 — what we had, not what we would
> pick. Experts that miss the VRAM cache are read from host RAM and prefill streams experts over PCIe, so a DDR5 / PCIe 5.0
> platform should do better — an estimate, not a measurement: roughly +30–45 % multi-stream decode, ~2× faster long-prompt
> prefill, and about +10 % single-stream decode for DeepSeek (bound mostly by per-layer GPU work). GLM still spends about a
> quarter of each decode step reading experts from RAM, so it should gain more from faster memory than DeepSeek does.

How each of these numbers was reached — including the ideas that were measured and dropped — is in
[docs/performance.md](docs/performance.md); the benchmark itself is described in [docs/benchmarks.md](docs/benchmarks.md).

## Open weights used

Eke Hive runs the models' published weights as they are — nothing is re-trained, distilled or re-quantized here. All three
releases below are under the MIT license; download them yourself (they are not part of this repository).

| Model | Weights loaded | What is taken from it |
| --- | --- | --- |
| DeepSeek-V4.1-Flash | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (DeepSeek, MIT) | everything: MXFP4 routed experts, FP8 dense weights, Engram tables, the MTP layer, the vision encoder, tokenizer and chat encoding |
| GLM-5.3-Flash | [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (NVIDIA ModelOpt NVFP4 quantization of Z.ai's model, MIT) | the NVFP4 routed experts and the rest of the model, the BF16 vision encoder, tokenizer and chat template |
| GLM-5.3-Flash | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (Z.ai, MIT) | only Z.ai's own FP8 tensors — DSA attention projections, shared experts and the MTP layer's experts — fetched by `scripts/glm-fetch-fp8.py`, so these parts run at the precision Z.ai published |

## Requirements (short version)

- NVIDIA Blackwell GPU with `sm_120a` (RTX PRO 6000 Blackwell, RTX 50 series); 96 GB was used for development.
- x86-64 CPU with AVX2/FMA/F16C, ideally with many memory channels (CPU experts are memory-bandwidth bound).
- RAM: DeepSeek ~480 GB resident in RAM mode (512 GB installed is comfortable), ~300 GB in SSD mode (384 GB installed);
  GLM ~385 GB (358 GiB) resident after start, growing with open conversations; plus headroom.
- Fast local NVMe for the checkpoint: ~510 GB (DeepSeek), ~200 GB (GLM: ~191 GB NVFP4 checkpoint + Z.ai FP8 tensors).
- Linux, Docker and the NVIDIA Container Toolkit.

Details and what is measured vs. estimated: [docs/requirements.md](docs/requirements.md).

## Quick start

```bash
git clone https://github.com/yepapa-nest/eke-hive.git && cd eke-hive
# 1. Get the checkpoint (~510 GB)
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /models/DeepSeek-V4.1-Flash
export HIVE_CKPT=/models/DeepSeek-V4.1-Flash
# 2. Build the image and the engine (no GPU needed for this step)
docker build -t eke-hive:latest -f docker/Dockerfile .
scripts/build.sh
# 3. One-time preparation (exports the Engram hash constants)
scripts/prepare.sh
# 4. Start (daemon + API server on :8430); ready when the log prints "[hived] ready in"
scripts/hive-start.sh
tail -f run/logs/hived.log
# 5. Use it
curl -s localhost:8430/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hive","messages":[{"role":"user","content":"Hello!"}]}'
```

For GLM-5.3-Flash (same image and build — `scripts/build.sh` builds both `hived` and `hived_glm`; no `prepare.sh`):

```bash
huggingface-cli download nvidia/GLM-5.3-Flash-NVFP4 --local-dir /models/GLM-5.3-Flash-NVFP4
python3 scripts/glm-fetch-fp8.py /models/GLM-5.3-Flash-zai-fp8 --mtp-experts   # Z.ai FP8 tensors (HIVE_GLM_FP8_DIR)
HIVE_CKPT=/models/GLM-5.3-Flash-NVFP4 scripts/hive-start.sh
```

The launcher sees a GLM checkpoint, starts `hived_glm` and uses `config/glm.env` (`HIVE_CONFIG=config/glm.env` also works);
see [docs/glm.md](docs/glm.md#quick-start).

Full instructions, smaller-RAM (SSD) mode and running as a service: [docs/install.md](docs/install.md).

## Reasoning levels and budget

DeepSeek-V4.1-Flash thinks before it answers. Two request fields control it:

- `reasoning_effort` (or `reasoning.effort`): `none` turns thinking off; `minimal` 25,
  `low` 50, `medium`/`high` 75 (the default, the model's documented setting), `xhigh`/`max` 100, or any number 1–100.
  The number is the model's own effort scale, not a token count.
- `max_thinking_tokens` (alias `thinking_token_budget`): a hard cap. When the reasoning reaches N tokens the daemon
  inserts a short exit phrase (`HIVE_THINK_EXIT_TEXT`) and closes the thinking block, and the model writes its
  answer from what it has. Uncapped by default; a cap trades answer quality on hard questions for a bounded response
  time, so pick it per use (chat vs. batch jobs).
- Both are per request; nothing is restarted. Thinking text is returned as `reasoning_content` (streamed as it is
  generated), separate from the visible answer.

Details: [docs/configuration.md](docs/configuration.md#http-request-fields). GLM-5.3-Flash always thinks and has three levels
(low, high, max — unset means high): [docs/glm.md](docs/glm.md#thinking).

## Documentation

| | |
| --- | --- |
| [docs/install.md](docs/install.md) | Install, build, start, stop, sleep/wake, SSD mode, monitoring |
| [docs/build.md](docs/build.md) | Building from source, tests and the CPU-only test suite |
| [docs/configuration.md](docs/configuration.md) | Every option and environment switch, with the measurement behind its default |
| [docs/litellm.md](docs/litellm.md) | Running Eke Hive behind LiteLLM (or another OpenAI-compatible gateway): config, thinking fields, timeouts, sleep |
| [docs/architecture.md](docs/architecture.md) | How the engine is put together |
| [docs/performance.md](docs/performance.md) | The optimisation history of both models, step by step, including what was measured and dropped |
| [docs/benchmarks.md](docs/benchmarks.md) | The real-chat benchmark behind the headline numbers, both models |
| [docs/validation.md](docs/validation.md) | How correctness is checked (oracle, kernel bit-exact tests, quality suite) |
| [docs/limitations.md](docs/limitations.md) | Known limitations |
| [docs/glm.md](docs/glm.md) | The GLM-5.3-Flash family: setup, correctness, performance and its switches |

## Status

Eke Hive is a dedicated engine for two models (DeepSeek-V4.1-Flash, GLM-5.3-Flash) on one machine type that we own, put
together from the parts we had. It is shared as-is: it works for us every day, and the documentation says exactly what was
and was not measured. Other models and other GPUs are not planned; faster memory and PCIe would help most.

## License

MIT — see [LICENSE](LICENSE). Third-party material (DeepSeek reference code, GLM-5.3-Flash, nlohmann/json, the llama.cpp
nvfp4 dot-product technique) and its licenses are listed in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Model weights are not part
of this repository and are distributed by DeepSeek, Z.ai and NVIDIA under their own licenses.
