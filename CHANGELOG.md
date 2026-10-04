# Changelog

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
