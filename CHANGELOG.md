# Changelog

## Unreleased

Real-chat benchmark (`tools/bench_chat.py`, two runs each; [docs/benchmarks.md](docs/benchmarks.md)):

| | DeepSeek-V4.1-Flash | GLM-5.3-Flash |
| --- | --- | --- |
| Decode, 1 / 2 / 4 / 8 / 16 / 32 streams (total) | 85.3 / 92.9 / 124.6 / 175.0 / 178.1 / 181.5 tok/s | 70.3 / 85.2 / 97.9 / 129.1 / 128.3 / 129.6 tok/s |
| Time to first token, 17K / 42K / 54K prompt | 5.05 / 7.53 / 9.69 s | 7.3 / 11.8 / 13.9 s |

DeepSeek ([docs/performance.md](docs/performance.md), steps 22–23):
- Decode kernels: the fused expert kernel loads its activations a stage ahead, and the head reads each vocabulary row once for all
  rows of a step (`HIVE_HEAD_ROWS`) — both bit-identical; interleaved with the previous build c1 83.9 → 86.0 tok/s, c8 171.4 → 173.7.
- Layer yields also admit a request of up to 16K rows when it is at most half of what the paused prefill still has to do
  (`HIVE_LAYER_YIELD_MID`); its own prefill yields once more for decode steps. A 12K request behind an 85K prefill: 14.6 → 4.4 s to
  the first token.
- Rejected after measurement: a lower promotion threshold for the `seq` cache policy (c1 −2.5 to −4 %), and committing paced
  promotions one step head later (c4 and decode after a 54K prompt lower in both runs).
- `[decode-host]` gives the step-head promotion commit wait (`promo wait … ms/step`).

GLM ([docs/glm.md](docs/glm.md), steps 17–21):
- Layer yields inside a prefill (short requests served, decoders keep stepping): short request 14.0 → 0.65 s, decoder stall 14.4 → 1.6 s.
- 8 sequences per decode step and batched MTP verify by default: c2 66.6 → 79.5, c8 93.2 → 125.7 tok/s.
- Promotion victims sorted once per step instead of a scan of every slot per used expert (the host held the GPU idle 1.88 ms per
  verify step): c1 67.3 → 70.4, c4 88.1 → 96.6 tok/s.
- CPU-expert chunks sized to the worker count (`HIVE_GLM_CPU_ADAPT`, CPU phase 70 → 81 GB/s); logits into a pinned buffer.
- Early routing on the fast decode path (`HIVE_GLM_EARLY_ROUTE`, the DeepSeek post-and-gate code): output identical; c4 95.3 → 99.4,
  c2 86.7 → 85.2 tok/s.
- Monitoring lines in the DeepSeek formats (`HIVE_TRACE_CACHE`, `HIVE_PROFILE`): `[cache]`, `[profile]`, `[decode-host]` (GPU idle
  by cause, deferral wait) and `[early-route]`, read by `tools/hive_monitor.py`.

Tools: `tests/bench_moe_lowm.cu` (low-row expert launch timing), `tests/test_head_rows.cu` (head kernel bit comparison).

## v0.1.0 (first public release)

Eke Hive runs two mixture-of-experts models on one 96 GB Blackwell GPU, with the routed experts in host RAM:

- **DeepSeek-V4.1-Flash** (original MXFP4/FP8 checkpoint, text and images; optionally the Engram tables on NVMe) — the main model;
- **GLM-5.3-Flash** (NVIDIA NVFP4 checkpoint plus Z.ai's FP8 attention and shared-expert tensors; text, images and video).

One daemon serves one of the two at a time; switching restarts it with the other configuration (`config/hive.env`,
`config/glm.env`). HTTP API (OpenAI chat and Anthropic messages, streaming, tool calls, images, reasoning effort and budget),
sleep/wake, prompt-state reuse across turns, 262,144-token context on both.

Measured on the reference machine with the real-chat benchmark (`tools/bench_chat.py`; [docs/benchmarks.md](docs/benchmarks.md)):

| | DeepSeek-V4.1-Flash | GLM-5.3-Flash |
| --- | --- | --- |
| Decode, 1 / 4 / 8 streams (total) | 76.9 / 124.0 / 173.0 tok/s | 65.2 / 89.3 / 88.8 tok/s |
| Time to first token, 17K / 42K / 54K prompt | 4.9 / 7.5 / 9.6 s | 7.2 / 11.6 / 13.7 s |
| Quality suite (bundled) | 175 / 179 | 158 / 163 |
| Sleep / wake (level 3, a 120K conversation open) | 0.7 s / 3.2 s | 3.8 s / 3.0 s |
| Cold start to ready | 88 s | 95 s |

What is in this release (the DeepSeek optimisation steps are numbered in [docs/performance.md](docs/performance.md); the GLM
path is described in [docs/glm.md](docs/glm.md)):

- Expert placement: VRAM cache of the hottest experts with score-based promotion, CPU experts computed in place,
  PCIe streaming for prefill, elastic cache slots lent to long prefills.
- Routing that knows the cache: cache-aware routing (`HIVE_CACHE_PRIOR`) and expert deferral on both models.
- Decode path: fused kernels, early routing, speculative decoding with each checkpoint's own draft layer (lossless),
  batched decode (8 streams on DeepSeek, 4 on GLM).
- Prefill path: chunked and batched prefill, prompt checkpoints, prefix sharing with boundary snapshots, layer
  yields so that short requests and running decoders are not blocked by a long prefill.
- GLM: KV memory and expert-cache slots that grow with use up to the 262K context, and the checkpoint's vision encoder on the GPU
  for images and video.
- Operations: `/flush_cache`, `/release_memory_occupation` and `/resume_memory_occupation` (sleep/wake), optional API
  key and request-size limit, a log monitor and a daily report, the benchmark drivers, a CPU-only test suite that runs
  without the GPU, and a LiteLLM guide ([docs/litellm.md](docs/litellm.md)).

Known limits are listed in [docs/limitations.md](docs/limitations.md); other models and other GPU families are not supported.
