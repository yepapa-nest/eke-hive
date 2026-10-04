# Hardware and software requirements

Eke Hive was developed and measured on a single machine (see [performance.md](performance.md#reference-machine)).
Everything below marked **measured** comes from that machine. Anything marked **estimate** has not been tested
and is our best reading of where the limits are — please report what you see on other hardware.

## Supported models

- **DeepSeek-V4.1-Flash**, original checkpoint (`deepseek-ai/DeepSeek-V4.1-Flash`: MXFP4 experts, FP8 dense weights,
  FP8 Engram tables).
- **GLM-5.3-Flash**, NVIDIA's NVFP4 checkpoint (`nvidia/GLM-5.3-Flash-NVFP4`) plus the FP8 attention and shared-expert tensors
  of Z.ai's original release, fetched by `scripts/glm-fetch-fp8.py` ([glm.md](glm.md#quick-start)).

Other models are not supported.

## Summary

| | DeepSeek, RAM mode (default) | DeepSeek, SSD mode (`HIVE_ENGRAM_SSD=1`) | GLM-5.3-Flash |
| --- | --- | --- | --- |
| Daemon resident memory | 478 GB **measured** | 295 GB **measured** | ~385 GB (358 GiB) after start, growing with open conversations **measured** |
| System RAM to plan for | 512 GB or more (**estimate**: resident set + OS + page cache headroom) | 384 GB or more (**estimate**) | 448 GB or more (**estimate**) |
| GPU | NVIDIA Blackwell, compute capability 12.0 (`sm_120a`) | same | same |
| VRAM | 96 GB used on the reference machine (cache sized to fill it) | same | same |
| Disk space for the checkpoint | ~510 GB | ~510 GB | ~191 GB NVFP4 + ~2.1 GiB FP8 tensors (~9 GiB with `--mtp-experts`) |
| Disk speed | only affects cold start (7.3 GB/s → 88–90 s) | Engram rows are read during inference: an NVMe with high random-read IOPS is required (reference: 1.12 M IOPS at QD128) | cold start 95–97 s (not bound by the disk) |
| CPU | x86-64 with AVX2, FMA and F16C | same | same |

## GPU

- The CUDA kernels are compiled for `sm_120a` only (block-scaled MMA instructions exist only in the `a`
  variant). That covers Blackwell workstation and consumer cards (RTX PRO 6000 Blackwell, RTX 50 series).
  Hopper, Ada, Ampere and data-center Blackwell (`sm_100`) are **not** supported by the current build.
- VRAM holds the dense weights, attention state, KV for active sessions and a cache of hot experts. With
  `HIVE_CACHE_FIT=1` (both shipped configurations) the expert cache takes the VRAM left after everything else minus
  `HIVE_CACHE_RESERVE_MB`; without it the cache is sized by `--vram-cache-mb` and the rest of VRAM must stay free for the
  other buffers. A smaller card
  means a smaller expert cache and more CPU work per token. The minimum VRAM that still runs has **not been
  measured yet**.
- Driver: one that supports CUDA 13.0.

## CPU and memory

- Experts that miss the VRAM cache are computed by the CPU straight from RAM. That work is bound by **memory
  bandwidth**, not by core count: on the reference machine the CPU side reads ~97 GB/s on DeepSeek and 77–91 GB/s
  on GLM, most of the ~122 GB/s its two DDR4 memory nodes measured. **Estimate**: decode speed with many cache misses scales roughly with memory
  bandwidth; a platform with half the bandwidth will see a larger slowdown at high concurrency than at one
  stream.
- The engine places half of every expert on each NUMA node and runs one worker pool per node. It was measured
  with two nodes only; single-node systems run the same code path but were **not measured**.
- Experts (and DeepSeek's Engram tables) live in pinned (page-locked) memory: the container or service needs an unlimited
  `memlock` limit (`--ulimit memlock=-1:-1`).
- AVX-512 is not used.

## Storage

- DeepSeek's RAM mode reads the whole checkpoint once at start (parallel O_DIRECT). Start time is roughly
  `checkpoint size / disk throughput` plus ~20 s. GLM starts in 95–97 s from its 191 GB checkpoint; its load is not bound
  by the disk.
- DeepSeek's SSD mode keeps the Engram value tables on disk and reads ~48 rows per token, with an in-RAM row cache
  (2 GiB by default). Put the checkpoint on a fast local NVMe; network or spinning storage will not work well.

## Software

- Linux, Docker with the NVIDIA Container Toolkit (the provided image is the supported build environment).
- Building without Docker: CUDA Toolkit 13.0, GCC 13, CMake ≥ 3.24, Ninja, libnuma, Python ≥ 3.10 with the
  packages in [`requirements.txt`](../requirements.txt) and PyTorch (CPU build is enough).
- Versions in the image the published numbers were measured with: CUDA 13.0.3, PyTorch 2.14.1 (CPU),
  transformers 5.18.0, tokenizers 0.23.2, safetensors 0.8.0, NumPy 2.5.3, Pillow 12.3.0, PyAV 19.0.1 (GLM video),
  FastAPI 0.142.2, Uvicorn 0.54.0. `requirements.txt` does not pin
  them; pin to these if a newer release misbehaves.
