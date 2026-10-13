CUDA stand-in headers for the CPU-only daemons (`hived_qwen_cpu`, …): device memory is host memory, streams run in the caller.
They are copies of `tools/cpu_fake/include/` without `numa.h` (the CPU daemons use the real libnuma) — keep them in step.
