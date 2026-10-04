# Validation

How Eke Hive checks that it computes what its two models compute, and how to run each check. Most of this page is the
DeepSeek-V4.1-Flash path, which has the oracle and the largest test set; GLM-5.3-Flash is checked against the transformers
implementation ([below](#glm-53-flash) and [glm.md](glm.md#correctness)). Build
instructions are in [build.md](build.md); the engine structure the tests refer to is in
[architecture.md](architecture.md).

## Strategy

For DeepSeek, the numerical contract is the reference implementation shipped with the checkpoint (`inference/model.py`,
`inference/kernel.py`). Correctness is checked at five levels:

| Level | Question | Tool | Needs |
| --- | --- | --- | --- |
| Kernels | Does each optimised kernel compute what the kernel it replaces computes? | `engine/tests/test_*.cu`, `test_expert_cpu` | GPU |
| Layers | Does each layer of the engine match the reference model? | oracle + `tools/compare_golden.py` (`validate_layers.sh`, `validate_inject.sh`, `validate_tail.sh`) | GPU + checkpoint |
| Paths | Do alternative paths (graphs, fused kernels, tiles, MTP, vision) agree with the plain path? | `tools/test_graph.sh`, `test_fuse.sh`, `test_tile.sh`, `test_spec.sh`, `compare_vision.py`, `hive --verify-test` | GPU + checkpoint |
| Service | Does the running server answer correctly? | `tools/quality_eval.py` + `quality_compare.py`, `tools/gap_eval.py` | GPU + checkpoint |
| Host logic | Are the scheduler, caches, snapshots, loaders, server and option parsing correct? | `tools/test_cpu.sh` (fake CUDA runtime) | CPU only |

### Why the engine is not bit-identical end to end

Four effects make exact end-to-end equality the wrong criterion, so each check uses the strongest criterion
that is actually achievable:

1. **Accumulation order.** The engine's tensor-core GEMMs accumulate in a different order than the reference's
   fp32 matmuls. This produces small per-layer errors.
2. **Chaotic amplification.** Stacking layers without correction, routing starts to differ from about layer 4–5
   (relative error ~0.18) — the same happens to the oracle itself at a different precision. Small bf16/fp4
   noise flips near-tied expert choices, and the flip propagates. The decisive layer check therefore *injects*
   the golden input into each layer and measures only that layer's own error.
3. **Timing-dependent CPU/GPU split.** Whether a cache-miss expert is computed by the CPU or copied to the GPU
   depends on timing; the two paths round in a different order. The same input decoded twice by the engine
   differs at the level of relative error 0.039. Requiring bit equality from a new path would be a standard
   that sequential decoding itself does not meet.
4. **Decode routing changes adopted on quality.** Cache-aware routing (`HIVE_CACHE_PRIOR`) and expert deferral
   (`HIVE_DECODE_DEFER`, `HIVE_GLM_DEFER`), on in the shipped configurations, deliberately change which experts a decode
   step uses or when their output is added. They are validated by the quality suite, not by layer equality; the layer
   and path scripts below switch them off themselves (`HIVE_CACHE_PRIOR=0 HIVE_DECODE_DEFER=0`).

Within those limits: kernels are compared bit for bit where they compute the same expressions, layers are
compared by relative error, cosine, routing set agreement and index set agreement, and the service is compared
by task pass rates with a paired significance test.

## Running GPU checks

All GPU checks run inside the project image through `scripts/hive-run.sh`, which starts a throw-away container
with the GPU, the repository at `/hive`, the checkpoint at `/model` (read-only, `HIVE_CKPT_MOUNT`) and the state
directory at `/out`, and runs the given command in `/hive/engine`:

```bash
scripts/hive-stop.sh     # the server uses the whole GPU — stop it first
scripts/hive-run.sh "./build/test_kernels && ./build/test_expert_cpu"
```

The validation scripts write goldens to `/out/golden/` and dumps to `/out/dump/` (i.e. under
`$HIVE_STATE_DIR` on the host) and expect the Engram constants from `scripts/prepare.sh` in `/out/engram`.
Inside the container the checkpoint is mounted at `$HIVE_CKPT_MOUNT` (default `/model`); the validation scripts
point `HIVE_CKPT` there themselves, and they switch off the decode routing changes of the shipped configuration (effect 4). When running the oracle or `hive` by hand, pass `--ckpt /model` (or set
`HIVE_CKPT=/model`). The scripts use `engine/$HIVE_BUILD` (default `build`).

## Kernel tests

`engine/tests/test_*.cu` are built with the engine (`scripts/build.sh`). Each optimised kernel is tested against
the kernel it replaces, and new arithmetic is also tested against a host double-precision reference:

- **Bit-identical** means the new kernel uses the same expressions and the same reduction order as the old
  one, so the outputs must match exactly (`max|Δ| = 0`). Fused kernels, the grouped/F2/F3 expert kernels,
  the v3 decode kernels, the grouped prefill GEMM, the flash-attention variants, the HC fusions and the
  MTP verify kernels are held to this.
- **Within tolerance** means the computation is the same function but the accumulation order differs (for
  example a tensor-core GEMM against cuBLAS fp32, or a fused HC mix against the cuBLAS path), or the reference
  is computed in double. These tests report relative error, cosine and max |Δ| against the double reference
  and check documented bounds (for example `test_idx_f32`: relative error < 1e-4 against double).

| Area | Tests |
| --- | --- |
| Basic kernels (quantisation, norms, GEMV, attention, router, HC, RoPE, ...) against a double host reference; also the fused-vs-unfused decode chain | `test_kernels` |
| Block-scaled tensor-core GEMM and expert kernels | `test_mx`, `test_mx2`, `test_mx3`, `test_mxg`, `test_grouped_prefill`, `test_woa` |
| Prefill attention and indexer | `test_attn_flash`, `test_idx_tc`, `test_idx_f32`, `test_idx_cand`, `test_idx_cand_tc` |
| Hyper-connections | `test_hc_mix`, `test_hc_fuse`, `test_hc_decode` |
| Decode kernels | `test_decode_moe`, `test_decode_attn`, `test_decode_attn2`, `test_decode_attn3`, `test_decode_gemv`, `test_decode_longctx` |
| Decode scheduling on the GPU | `test_decode_step_graph`, `test_decode_ubatch`, `test_early_route`, `test_verify_decode` |
| Device memory / sleep level 3 | `test_devmem` (release and restore at the same addresses, captured graphs replayed after restore) |
| Bandwidth probes | `test_dma` (pinned RAM → VRAM per NUMA node), `bench_gemv` |
| CPU expert kernel | `test_expert_cpu` (AVX2 kernel against the reference, bf16 bit-identical) |

In the last full run of the DeepSeek kernel tests every one passed, bit-identical or within its tolerance, except one: the
alternative long-context top-k kernel (`HIVE_DECODE_TOPK2`) differed in four fp32 cases at k = 2048. That switch
is off and was not adopted.

## Oracle and layer comparisons

### The golden model

`oracle/dsv41_oracle.py` is the reference implementation ported to plain PyTorch, running on the CPU (see
[architecture.md](architecture.md#the-oracle)). It streams weights layer by layer and dequantises only the
routed experts, so it needs RAM and time but no GPU:

```bash
python3 /hive/oracle/dsv41_oracle.py --ckpt /model --prompt "The capital of France is" --bos \
  --max-layer 5 --continue-ids 13,1052,5 --out /out/golden/l5 --threads 32
```

Outputs per layer: the residual stream (`h_L*`), the routed expert ids and weights, the sparse indexer
selections, Engram outputs, and the final logits and greedy tokens. `--continue-ids` processes extra tokens one by
one after the prefill (the decode path), `--decode N` runs greedy decode, `--image FILE` adds an image,
`--spec` produces a golden for the MTP draft block, `--ids`/`--ids-file` take token ids directly.

### Comparing engine dumps

`hive --dump DIR` writes the same quantities from the engine; `tools/compare_golden.py --golden G --dump D`
compares them layer by layer and exits non-zero on failure. Default criteria:

| Check | Default |
| --- | --- |
| Hidden state per layer: relative max error | ≤ 0.05 (`--max-rel`) |
| Hidden state per layer: cosine | ≥ 0.999 (`--min-cos`) |
| Routing: expert set agreement per token | 100 % (`--min-routing`); routing weights checked when the sets agree |
| Sparse indexer: selected set agreement | 100 % (`--min-index`) |
| Missing dump, shape mismatch, NaN/Inf | always a failure |

`--report-only` prints without failing (diagnosis only). `--tail N` compares only the last N rows from layer 20
on (decoder-tail mode); `--chunk` handles split prefills.

### The three layer scripts

```bash
scripts/hive-run.sh "bash /hive/tools/validate_layers.sh 39"   # golden for layers 0..39 + accumulated comparison
scripts/hive-run.sh "bash /hive/tools/validate_inject.sh 39"   # per-layer injected comparison (decisive)
scripts/hive-run.sh "bash /hive/tools/validate_tail.sh 2688"   # decoder-tail mode on a ~3,000-token prompt
```

- `validate_layers.sh N [cache_mb]` creates the golden for layers 0..N (a 14-token prompt plus three decode
  steps) if it does not exist, runs the engine with `--dump` and compares. Errors accumulate across layers here.
- `validate_inject.sh N` reuses that golden, exports it as raw inputs and runs the engine with `--inject`: before
  each layer the hidden state is overwritten with the golden output of the previous layer, so each layer's own
  error is measured in isolation. This is the pass/fail criterion for the model.
- `validate_tail.sh [tail]` builds a full-depth golden for a ~3,000-token prompt (slow) and compares the engine
  with decoder tail off and on.

Recorded results:

- Layer 0: text prefill and decode cosine 1.000000, routing 100 %; image span cosine 0.9989 (the intrinsic bf16
  noise of the vision encoder, the same size as a bf16 CPU reference's deviation from fp32); split prefill
  identical to single-chunk prefill.
- Injected, layers 0–39: prefill relative error ≤ 5e-3 and decode ≤ 1.5e-2, routing 100 %, indexer sets 100 %;
  final logits cosine 0.99998. CUDA-graph path identical to eager execution (`max|Δ| = 0`).
Recorded with the configuration of steps 1–20 of [performance.md](performance.md) (before the decode routing changes):

- Full build against the golden, that configuration vs. all optimisation switches off: final logits cosine
  0.977 vs 0.878, KL 8.0e-4 vs 1.7e-3, same argmax.
- Full model, all-off baseline vs. that configuration, 32 greedy tokens from random-token prompts: 16K
  tokens KL 6.3e-5 with identical tokens; 81K KL 9.4e-3 (baseline against itself: 8.0e-3); a 906-token prompt KL
  0.030 against a noise level of ~0.003 (the short-prefill path's summation order), same argmax.

## Path comparisons

| Script / mode | Compares |
| --- | --- |
| `tools/test_graph.sh [layer]` | CUDA graphs on vs eager (`HIVE_NO_GRAPH=1`): routing, final state, logits; plus profiles |
| `tools/test_fuse.sh [layer]` | Fused decode kernels on vs off (`HIVE_FUSE=1/0`), with graphs and eager |
| `tools/test_tile.sh` | ~40K-token prefill as sequential chunks vs one tiled pass vs tiled without CPU experts: end-of-prefill logits and 32 greedy tokens; then graph-address safety (tiled prefill between decode steps vs eager) |
| `tools/test_spec.sh [tag] [steps]` | MTP: oracle `--spec` golden vs engine draft block (`tools/compare_spec.py`: MTP inputs, per-stage routing and outputs, logits, draft ids, confidences), then N speculative steps |
| `hive --verify-test` | MTP verify: independence from future tokens, rollback at every keep position, continuation equals sequential decoding |
| `hive --verify2-test [N]` | The batched verify paths (`HIVE_MTP_VERIFY2`, `HIVE_MTP_BATCH`) against the reference path, plus timing |
| `tools/test_batch_l0.sh` | Two-sequence batched decode against per-sequence goldens; image path |
| `tools/compare_vision.py` | Vision encoder stage by stage against the oracle's bf16 and fp32 references (`HIVE_VISION_DUMP`) |
| `tools/test_daemon.py` | A running daemon (works with `--fake-head` on a partial checkpoint): concurrent requests batch together, prefix reuse, cancellation |
| `tools/validation_matrix.py` | Prints the boundary/verify regression plan (chunk sizes around 15/16, 127–129, 1023/1024, 2688, 16384, tiles); `--execute` runs it on the GPU |

Recorded MTP results: future-token independence 5/5, draft tokens equal to the oracle 6/6.

## Quality suite

`tools/quality_eval.py` is a self-contained, automatically graded evaluation run against the HTTP server — no
internet and no external datasets; all items are generated from a fixed seed or embedded in the file. Requests
use temperature 0, a fixed seed and thinking off, except two streaming checks of the default thinking path.

| Suite | Items | Content |
| --- | --- | --- |
| `ruler` | 35 | Long-context retrieval at 4K, 16K, 32K, 64K and 100K tokens × 7 kinds: single needle at depth 10/50/90 % (Korean and English), multi-key, multi-value, variable tracking |
| `qa` | 80 | Korean and English, 40 each: arithmetic, units, dates, strings, logic, facts |
| `code` | 20 | Python functions, executed against tests in a sandbox |
| `tool` | 22 | Tool calls and no-call cases, follow-ups, non-streaming, messages-endpoint format |
| `ops` | 22 | 8 concurrent streams, a 6-turn conversation with recall, short request after a 100K one, both endpoints streaming and non-streaming, images |

```bash
python3 tools/quality_eval.py --base http://127.0.0.1:8430 --out base.json --label baseline \
  --image $HIVE_CKPT/inference/examples/images/corn.jpeg   # the default when HIVE_CKPT is set (DeepSeek)
# change one setting, restart, then:
python3 tools/quality_eval.py --base http://127.0.0.1:8430 --out new.json --label candidate --image ...
python3 tools/quality_compare.py base.json new.json
```

Options: `--suite ruler,qa,...`, `--ruler-lengths`, `--corpus FILE` (filler text; generated prose by default),
`--only REGEX`, `--effort`/`--think-budget` to evaluate with reasoning on. `quality_compare.py` prints per-suite
pass rates, the items that flipped in each direction, the exact two-sided McNemar p-value, and the number of
items whose output differs — it sets no thresholds.

GLM-5.3-Flash runs the same suite with `--effort low` (it cannot switch thinking off) and `--ruler-lengths
65536,131072,250000`; its checkpoint ships no example image, so the two image items of `ops` are skipped unless `--image` is
given — 163 items in all.

Recorded (DeepSeek): 174/179 with every optimisation off, 177/179 after step 12 of [performance.md](performance.md) (against
the all-off baseline: long context 31/35 → 33/35, QA 79/80 → 80/80, code, tools and operations unchanged at 20/20, 22/22, 22/22,
no pass → fail flips), 175/179 after step 20 (McNemar p = 1.0; the flips are items that also flip between two runs of the same
build). Step 21 was measured against a re-run of the step-20 build, which gave 172/179 (run-to-run spread); with step 21:
175/179 — long context 32/35, QA 79/80, code 20/20, tools 22/22, operations 22/22.

Recorded (GLM-5.3-Flash, shipped configuration): 158/163 — long context 21/21, QA 75/80, code 20/20, tools 22/22,
operations 20/20.

## GLM-5.3-Flash

There is no oracle for GLM; the reference is the transformers 5.18 implementation (`Glm5NextTextModel`) run on the CPU with
the same weights:

| Check | Tool | Recorded |
| --- | --- | --- |
| Whole model, layer by layer, three chat prompts | `engine/build/glm_check` | next-token top-1 3/3 on the BF16 and the FP8 path; top-20 overlap 18–20/20; lowest hidden-state cosine over 45 layers 0.99996 |
| KDA (linear attention) kernels | `test_glm_kda` | relative error 8.5e-5; chunked prefill and batched decode bit-identical to one-shot |
| DSA (sparse attention) kernels | `test_glm_dsa` | selected token sets identical to transformers at 64–6,000 tokens |
| NVFP4 / FP8 / decode / MoE kernels | `test_glm_nvfp4`, `test_glm_fp8b`, `test_glm_decode`, `test_glm_moe`, `test_glm_moe_fp8` | against CPU or previous-kernel references (NVFP4 and FP8 with negative controls) |
| Speculative decoding | `glm_check` with `GLM_CHECK_MTP` | tokens with MTP k = 3 identical to plain greedy decoding (32 and 120 tokens, prefetch off) |
| Service | `tools/quality_eval.py` | 158/163 (above) |

`glm_check`, `test_glm_kda` and `test_glm_dsa` read reference dumps produced with transformers; the scripts that generate them
are not included in this repository. Details: [glm.md](glm.md#correctness).

## Gap evaluation

`tools/gap_eval.py` covers what the quality suite does not:

| Suite | Checks |
| --- | --- |
| `boundary` | Prompts of exactly 15/16, 127–129, 1023–1025, 2687–2689, 16383–16385, 32767/32768 and 98303–98305 tokens (chunk, tail and tile boundaries); asks for a fact from the start of the prompt |
| `session` | Continuing, editing an earlier turn, changing the system prompt, regenerating, swapping images, disconnecting mid-stream, stop strings (both endpoints), EOS |
| `load` | 16 concurrent streams with answer checks, chunk-gap percentiles, TTFT distribution, cancellations under load |
| `sampling` | 8 prompts × 64 seeds at temperature > 0; compare two runs (e.g. MTP on/off) with `tools/sampling_compare.py` (chi-square on first words, permutation test on length, Fisher test on accuracy) |

```bash
python3 tools/gap_eval.py --base http://127.0.0.1:8430 --out gap.json --suite boundary,session,load,sampling
```

Recorded on an earlier DeepSeek build (with 17 boundary prompts at the time): boundaries 17/17, sessions 12/13 (one temperature-0 regeneration differed in wording — the
timing-dependent CPU/GPU split above), 16 concurrent streams 15/16, cancellation in 0.11 s, sampling
distributions with MTP on and off indistinguishable.

## CPU-only suite

```bash
tools/test_cpu.sh          # no GPU, no checkpoint, no services
tools/test_tsan_cpu.sh     # the same host tests under ThreadSanitizer, plus negative controls
```

The host-side code — caches, CPU expert pool, scheduler, snapshots, loaders, Engram SSD reader, server — is
tested by compiling the **production sources** (including the real `hived.cpp`, `expert_store.cpp`,
`expert_cpu.cpp`) against a fake CUDA runtime in `tools/cpu_fake/`: stub headers for the CUDA runtime, cuBLAS,
bf16/fp8 types and libnuma, a fake runtime with asynchronous fake streams and events, and a fake checkpoint
writer. `tools/cpu_fake/harness.py` builds and caches these test programs.

What it covers:

- Cache transitions (pending promotions, eviction, restore, event errors, staging reuse), cache replay, elastic
  and fit-to-VRAM slot allocation.
- The CPU expert pool with real worker threads: results compared bit for bit with the single-call kernel;
  ownership and stealing; the decode handshake and early-routing gates on a fake GPU thread.
- The real daemon: batching, prefix reuse, cancellation by request id, partial ingress, slow clients, host
  exceptions, graceful stop, batched prefill, layer yield, sleep/wake state machine, the MTP scheduler against
  oracle tokens (including wrong drafts) and the gate cost tables.
- Snapshot fences and the pinned buffer pool; parallel loading (bulk load equals per-layer load, failure
  fallback); Engram SSD lookups equal to RAM lookups byte for byte across modes, cache sizes and thread counts.
- The API server: streaming on both endpoints, stop strings, reasoning cap and exit phrase, sessions, EOF handling.
- The GLM family adapter (`tools/test_glm_family_cpu.py`): thinking levels and their mapping, `HIVE_GLM_NOTHINK`, tool-call
  parsing, image and video preprocessing.
- Option plumbing (`tools/test_options_cpu.py`): every documented engine option is parsed so that unset, empty
  and `0` select the default path, values are used as given, and the launchers forward `HIVE_*` by value.
- Python syntax of all tools, shell syntax of all scripts, and `git diff --check`.

**Negative controls.** A test that cannot fail proves nothing, so most tests include mutants: the harness
rebuilds the production code with a deliberate defect (a missing fence wait, a rollback that keeps one row too
many, relaxed memory ordering in a handshake, a removed ownership check, wrong row addresses, ...) and asserts
that the test detects it. The TSAN script additionally requires that injected races are reported and runs 20
`hived.cpp` mutants under UBSan that must each fail.

TSAN note: on some kernels ThreadSanitizer aborts at start with "unexpected memory mapping" because of ASLR;
the TSAN script therefore runs only the test processes under `setarch x86_64 -R`. Sanitizers are not disabled.

The suite takes a while (the Engram SSD test alone is about 30 minutes). It does **not** execute any CUDA
kernel and says nothing about GPU numerics or speed — it ends by printing exactly that.

## What CI can run

A CI runner without a GPU can run `tools/test_cpu.sh` and `tools/test_tsan_cpu.sh` (it needs GCC with AVX2
code generation and a CPU that executes AVX2, Python with the packages in `requirements.txt`, and a git
checkout) and can compile the whole engine with `scripts/build.sh`. Everything else — kernel tests, oracle
comparisons, path comparisons, the quality and gap suites — needs the GPU and a checkpoint (~510 GB for DeepSeek, ~200 GB for GLM), and is run
by hand on the target machine before a change is adopted.

## Harness variables

Variables read by the test tools (not by the engine):

| Variable | Used by | Meaning |
| --- | --- | --- |
| `HIVE_TEST_SANITIZER` | CPU test builds | `thread` or `undefined` — build the test programs with that sanitizer |
| `HIVE_TEST_BUILD_DIR` | CPU test builds | Shared object cache for the fake-CUDA builds (a temporary directory by default) |
| `HIVE_DAEMON_NEGATIVE` | `test_daemon_cpu.py` | `fence` = the fence-wait negative control · `mutants` = run the `hived.cpp` mutants |
| `HIVE_DAEMON_MUTANTS` | `test_daemon_cpu.py` | Name prefix filter for the mutants |
| `HIVE_DAEMON_CONFIGS`, `HIVE_DAEMON_EXTRA` | `test_daemon_cpu.py` | Subset of daemon configurations / extra scenarios to run |
| `HIVE_MTP_NEGATIVE` | `test_mtp_cpu.py` | `1` = also run the MTP scheduler mutants |
| `HIVE_SLEEP_NEGATIVE` | `test_sleep_cpu.py` | `1` = also run the sleep state-machine mutants |
| `HIVE_SLEEP_TESTS` | `test_sleep_cpu.py` | Subset to run (e.g. `store,daemon`) |
| `HIVE_SNAPSHOT_QUICK` | `test_snapshot_cpu.py` | Shorter snapshot matrix |

Others (`HIVE_POOL_INJECT_RACE`, `HIVE_POOL_STRICT_TDONE`, `HIVE_TEST_EXPECT_DIRECT`, ...) are set by the tests
themselves for their child processes.
