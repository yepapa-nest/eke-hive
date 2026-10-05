# Known limitations

- **Two models, one at a time.** Eke Hive supports exactly two checkpoints, and one daemon serves one of them:
  - **DeepSeek-V4.1-Flash** (original checkpoint, daemon `hived`): compressed KV attention with a sparse indexer, hyper-connections, Engram
    tables, MTP draft head, routed experts in MXFP4;
  - **GLM-5.3-Flash** (NVIDIA NVFP4 checkpoint plus Z.ai's FP8 attention and shared-expert tensors, daemon `hived_glm`): linear attention (KDA)
    and DSA layers, hyper-connections, NextN draft layer, routed experts in NVFP4, a vision encoder.

  The engine is written for these architectures; other models are not supported. Switching between the two restarts the
  daemon with the other configuration (`config/hive.env`, `config/glm.env`).
- **One GPU family.** Kernels are compiled for `sm_120a` (Blackwell workstation/consumer). One GPU per daemon.
- **Measured on one machine.** All numbers come from a single reference machine. Single-NUMA systems, other
  memory configurations and smaller GPUs run the same code but were not measured. The smallest VRAM that still
  works has not been determined.
- **Memory.** DeepSeek keeps ~478 GB resident in RAM mode and ~295 GB in SSD mode (Engram tables on NVMe). GLM keeps
  ~385 GB (358 GiB) resident after start, growing with open conversations; it has no SSD mode (no Engram tables). Neither
  model has a mode that keeps the routed experts on disk.
- **Determinism.** With temperature 0, regenerating the same answer can differ in wording: whether a cache-miss
  expert is computed by the CPU or the GPU depends on timing, and the two paths round in a different order. Both
  are exact implementations of the same function; results are not bit-reproducible run to run. Cache-aware routing
  (`HIVE_CACHE_PRIOR`, on in both shipped configurations) also makes the chosen experts depend on what is in the cache.
- **Long prefill.** A very long prompt is processed in large chunks. On DeepSeek, short requests that arrive meanwhile are
  served at layer boundaries (first token in ~1–1.5 s during an 85K prefill), and a request of up to 16K rows joins at a yield
  when it is much smaller than what the running prefill still has to do (`HIVE_LAYER_YIELD_MID`); larger prompts are still
  processed one after another. On GLM short requests are served at layer boundaries too (`HIVE_LAYER_YIELD`); requests above
  1,023 rows wait until the running prefill has finished (a 250K-token prompt takes about 66 s).
- **Concurrency.** Both models decode up to 8 requests at a time (`HIVE_MAX_BATCH`); more wait in the daemon's queue.
- **GLM thinking.** GLM-5.3-Flash cannot switch thinking off; "off" maps to its lowest level (`low`).
- **GLM without the FP8 tensors.** Without `HIVE_GLM_FP8_DIR` (fetched by `scripts/glm-fetch-fp8.py`), GLM runs its attention
  projections and shared experts from the BF16 copies of the NVFP4 checkpoint and has no speculative decoding.
- **Reasoning cap.** `max_thinking_tokens` truncates reasoning with a transition phrase. The models were not trained
  for truncated reasoning, so answer quality under a tight cap is not guaranteed.
- **Images and video.** DeepSeek takes images through the model's own preprocessing, not video. GLM-5.3-Flash takes images
  and video (sampled at 2 fps; each pair of frames becomes one span); video needs PyAV in the server image (`requirements.txt`).
