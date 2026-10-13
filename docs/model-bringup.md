# Bringing a new model version up to par

Eke Hive supports four model families (Qwen3.8-27B, Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, GLM-5.3-Flash). When a family ships a
new version, this is the procedure that took each of them to "no worse than the reference engine" in October 2026. Per-family facts,
measurements and the full adopted / not-adopted lists are in [q27.md](q27.md), [qwen.md](qwen.md), [glm.md](glm.md) and
[performance.md](performance.md); this page is the order of work and the traps that cost the most time.

## 0. Targets and rules

- Reference engine per family: Flash-Next = SGLang, 27B = NInfer, DeepSeek / GLM = the previous Eke Hive release. Same weights where
  possible (the 27B comparison used the uncensored checkpoint NInfer ships, because that is what it serves).
- Whole card first (96 GB, vision on, context 262,144, 600 W), every axis: c1 / c2 / c4 / c8 / c16 TPOT and TTFT, prefill 8K / 32K / 64K
  TTFT, decode TPOT after each long prompt.
- Quality is a regression gate, not a score: the non-thinking suites (ruler, qa en/ko, code, tool, session) against the official
  checkpoint's default. A change passes when every category is within its binomial standard error (at least one item), the total is
  not lower, and the oracle tokens are identical. Reduced-precision paths (int8 dense, quantized prefill, bf16 GDN storage) are
  adopted only through this gate; a path that fails it is kept opt-in and recorded as not adopted.
- Batch invariance: the kernel that produces a token's logits must not change with the number of rows in the step, so speculative
  decoding stays equal to greedy. Never switch kernels by M on the output path. The drafter is outside the output path and may differ.
- Only MIT code may be adapted (with attribution in THIRD_PARTY_NOTICES). Apache-licensed engines (SGLang, NInfer) are read for ideas only.

## 1. Port and make it correct

1. Read `config.json` and the safetensors headers; map every tensor to an existing kernel family (NVFP4 / FP8 GEMM, GDN or KDA linear
   attention, sparse attention, MTP or an external drafter, vision). Reuse before writing new code.
2. Correctness gates, in this order, on every kernel change:
   - oracle: logits of a fixed prompt against the reference implementation (top-1 and top-20 overlap);
   - spec == greedy at S1, S4 and S8 (and at the largest verify row count);
   - teacher-forced comparison (the same token history for every variant). Free-running continuations are misleading: one near-tie
     flip at the first token makes every later row "wrong" (the 27B p6k case, 2026-10-11);
   - for deep recurrent parts, per-layer local error with the reference's per-layer inputs injected. End-state comparisons after 48
     layers measure the model's amplification of tiny differences, not the kernel (even an fp32 rewrite failed such a metric).
3. Then the quality gate above.

## 2. Measure against the reference fairly

- Same prompts on both sides (`--seed 0`, the same dataset, one seed for every axis), cache flushed between axes.
- Check that the reference honours the request. NInfer ignores `ignore_eos`, still reports 256 completion tokens and so understates
  TPOT by up to several times on early-stopped requests; correct it by retokenising the output text (true token counts) and
  compare on requests both engines completed at full length.
- Report per-stream TPOT and throughput separately; an engine that caps concurrency (NInfer: 8) queues the rest, so its c16 TPOT is
  not comparable — compare c16 on TTFT and throughput.
- Profile before changing anything, and profile the real serving path, not a test harness: the 27B harness showed 30 % GPU idle that
  the daemon does not have (it copied all logits to the host). With nsys use `--cuda-graph-trace=graph`; node-level tracing inflates
  graph launches about 6x.
- Split the gap per phase against the reference with the same tool on both sides (nsys can be mounted into the reference container).
  Only then pick the lever.

## 3. Levers that paid off (and the ones that did not)

Paid off, roughly in order of size:
- speculative decoding fitted to the model: the checkpoint MTP head for Flash-Next / DeepSeek / GLM; for 27B an external drafter
  (DFlash2, MIT) beat MTP. Pick by measurement per concurrency;
- tensor-core decode attention and GEMV variants that stay flat in rows up to the verify size;
- removing host-side stalls on the request path: freeing multi-GB session images on the engine thread (glibc munmap, about 400 ms
  per request once the host image budget overflowed), per-chunk KV cudaMalloc/cudaFree during prefill, a stream sync at every layer
  boundary when nothing waits;
- linear-attention prefill in WY / UT chunk form with bf16 storage and fp32 accumulation (27B GDN 179.5 → 83.3 ms at 8K), adopted
  through the quality gate;
- greedy requests skipping the rows × vocab logits copy.

Measured and not adopted (do not retry without a new idea): W4A4 activations on the MLP (fails the quality gate, almost no serving
gain), split-N GEMV (slow at one row), GEMM-tile verify kernel (faster at 32 rows, 9 % slower at one row, and kernel switching by M is
not allowed), 64-row verify, staged-activation GEMV, fixed draft length.

## 4. Finish

1. Final perf-core on the frozen build, official and (where relevant) the reference's weights; a per-axis verdict table.
2. Pressure test: 16 concurrent conversations growing toward the context limit plus returning conversations, so the VRAM pool spills to
   the host tier and the host budget evicts. Pass = every request completes with correct output, no OOM, daemon healthy, RSS bounded.
3. Freeze the source (engine, config including profiles, scripts) into the sweep tree, build once, record the source id, and run the
   VRAM-size sweep (96 / 72 / 48 / 32 / 24 / 16 / 12 / 8 GB and CPU only) with the same recipe for every cell.
4. Update the family page, performance.md and the README summary table.
