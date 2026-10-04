# Building and testing

## With Docker (supported)

```bash
docker build -t eke-hive:latest -f docker/Dockerfile .
scripts/build.sh            # -> engine/build (Release, all cores)
scripts/build.sh build-dev  # any other directory name works too
```

The image is `nvidia/cuda:13.0.3-devel-ubuntu24.04` (CUDA 13.0, GCC 13.3) plus CMake, Ninja, libnuma and the
Python packages. Building needs no GPU. The published numbers were measured with binaries from this compiler.

## Without Docker

Requirements: CUDA Toolkit 13.0, GCC 13, CMake ≥ 3.24, Ninja, libnuma development files.

```bash
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
```

- `CMAKE_CUDA_ARCHITECTURES` is fixed to `120a` (the block-scaled MMA instructions used by the kernels exist only
  in the `a` variant).
- Host code is compiled with `-mavx2 -mfma -mf16c`.

Outputs in `engine/build`: `hived` (the DeepSeek-V4.1-Flash daemon), `hived_glm` (the GLM-5.3-Flash daemon — the same source
built with the GLM family types), `hive` (DeepSeek CLI: single-run inference, layer dumps, validation modes), `glm_check`
(GLM forward check against a transformers reference), `st_inspect` (safetensors inspector), and the GPU test and benchmark
programs (`test_*`, `test_glm_*`, `bench_*`).

## Tests

### CPU-only suite (no GPU, no model)

```bash
tools/test_cpu.sh
```

Runs the cache, scheduler, daemon, server, sampler, snapshot, MTP, sleep, loading, Engram-SSD, GLM family adapter (thinking
levels, tool calls, image and video preprocessing) and log-monitor tests against a
fake CUDA runtime (`tools/cpu_fake`), including mutation tests that check the tests detect deliberately broken
code. It takes a while (the Engram-SSD test alone is 15–30 minutes). CI runs the same script with `HIVE_TEST_QUICK=1`,
which shrinks the Engram-SSD gather matrix (threads 1/3, fewer cache sizes and steps) so that it fits the hosted
runner; the full matrix is what maintainers run before a release.

### GPU tests (need the GPU to yourself)

```bash
scripts/hive-stop.sh
scripts/hive-run.sh "./build/test_kernels && ./build/test_expert_cpu"                 # DeepSeek kernels
scripts/hive-run.sh "./build/test_glm_nvfp4 && ./build/test_glm_fp8b && ./build/test_glm_decode && ./build/test_glm_moe"  # GLM kernels
```

`test_glm_kda`, `test_glm_dsa` and `glm_check` compare against reference dumps produced with transformers on the CPU; the
scripts that generate those dumps are not included in this repository.

Every optimised kernel has a test that compares it with the previous kernel, bit for bit or within a documented
tolerance. Model-level validation against the reference "oracle" and the quality suite are described in
[validation.md](validation.md).
