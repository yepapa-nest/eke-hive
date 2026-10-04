# Install and run

Check [requirements.md](requirements.md) first: a Blackwell (`sm_120a`) GPU, an AVX2 CPU, and for DeepSeek-V4.1-Flash
~512 GB of RAM (or ~384 GB in SSD mode) and ~510 GB of fast NVMe; for GLM-5.3-Flash ~448 GB of RAM and ~200 GB of NVMe.
The steps are the same for both models apart from the checkpoint (step 2) and the preparation (step 5).

## 1. Host setup

- NVIDIA driver with CUDA 13.0 support, Docker, and the NVIDIA Container Toolkit
  (`docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi` must work).
- On the host: `nvidia-smi`, `pgrep` (procps), `curl`, `python3` (the scripts use them) and the Hugging Face CLI for the
  download (`huggingface-cli download` or the newer `hf download`).
- The container runs with `--network host`, `--ulimit memlock=-1:-1`, `--cap-add SYS_NICE` (thread priorities of the CPU expert
  pool) and `--security-opt seccomp=unconfined` (the VMM calls of sleep level 3); a hardened Docker host must allow these.
- Nothing else may hold the GPU while Eke Hive runs: it uses almost all of the VRAM.
- The daemon pins hundreds of GB of RAM. The scripts run the container with `--ulimit memlock=-1:-1`; if you run
  the binaries yourself, raise `memlock` for that user.

## 2. Checkpoint

DeepSeek-V4.1-Flash:

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /models/DeepSeek-V4.1-Flash
```

The directory must contain `config.json`, the `model-*.safetensors` shards, `tokenizer.json`, and the
`encoding/` and `inference/` folders shipped with the model (the API server uses the model's own chat encoding
and image preprocessing from there). Put it on a local NVMe: start-up reads the whole checkpoint, and SSD mode
reads Engram rows from it while serving.

GLM-5.3-Flash: NVIDIA's NVFP4 checkpoint plus the FP8 tensors of Z.ai's original release (attention projections and shared
experts; `--mtp-experts` adds the draft layer's experts), fetched with HTTP range requests:

```bash
huggingface-cli download nvidia/GLM-5.3-Flash-NVFP4 --local-dir /models/GLM-5.3-Flash-NVFP4
python3 scripts/glm-fetch-fp8.py /models/GLM-5.3-Flash-zai-fp8 --mtp-experts   # ~9 GiB; HIVE_GLM_FP8_DIR points here
```

## 3. Configure

All scripts read a configuration file — [`config/hive.env`](../config/hive.env) for DeepSeek, or
[`config/glm.env`](../config/glm.env) when `HIVE_CKPT` is a GLM checkpoint (set `HIVE_CONFIG` to choose one explicitly).
These are the configurations the published numbers were measured with. Do not edit it; put your own values in a separate file and point `HIVE_LOCAL_ENV` at it (or export them):

```bash
cat > ~/hive.local.env <<'EOF'
HIVE_CKPT=/models/DeepSeek-V4.1-Flash
HIVE_STATE_DIR=/data/hive-state
EOF
export HIVE_LOCAL_ENV=~/hive.local.env
```

Precedence: environment > `HIVE_LOCAL_ENV` > the configuration file. A `HIVE_CKPT` in the environment or in `HIVE_LOCAL_ENV`
therefore decides which model starts (and, unless `HIVE_CONFIG` is set, which configuration file is read). The values you are
most likely to change (shipped value for DeepSeek / GLM):

| Key | Shipped | Meaning |
| --- | --- | --- |
| `HIVE_CKPT` | (required) | Checkpoint directory on the host |
| `HIVE_STATE_DIR` | `<repo>/run` | Logs, the daemon socket and the exported Engram constants |
| `HIVE_PORT` | 8430 | HTTP port (host network) |
| `HIVE_GPU` | 0 | GPU index |
| `HIVE_CACHE_RESERVE_MB` | 2400 (default) / 3000 | VRAM left free beside the expert cache, which takes the rest (`HIVE_CACHE_FIT=1`) |
| `HIVE_MAX_CTX` | 262144 / 262144 | Maximum context length |
| `HIVE_CPU_THREADS` | 16 / 32 | CPU expert worker threads (split across NUMA nodes) |
| `HIVE_MAX_BATCH` | 8 / 4 | Concurrent sequences in a decode batch |
| `HIVE_GLM_FP8_DIR` | — / `/models/GLM-5.3-Flash-zai-fp8` | GLM only: the directory written by `scripts/glm-fetch-fp8.py` |

Everything else is in [configuration.md](configuration.md).

## 4. Build

```bash
docker build -t eke-hive:latest -f docker/Dockerfile .   # CUDA 13.0.3 devel + Python packages
scripts/build.sh                                         # engine/build/{hived,hived_glm,hive,...}
```

No GPU is needed to build. See [build.md](build.md) for building without Docker and for the tests.

## 5. Prepare (once per checkpoint, DeepSeek only)

```bash
scripts/prepare.sh      # writes $HIVE_STATE_DIR/engram (Engram hash constants exported from the checkpoint)
```

GLM-5.3-Flash has no Engram tables and needs no preparation.

## 6. Start, use, stop

```bash
scripts/hive-start.sh                       # daemon + API server in container "hived"
grep '^\[hived\] ready in' "$HIVE_STATE_DIR/logs/hived.log"   # ~90 s (DeepSeek) / 95–97 s (GLM) on the reference machine
curl -s localhost:8430/health
scripts/hive-stop.sh
```

The API server answers `/v1/models` before the daemon has finished loading; wait for the `ready in` line (or
`/health` reporting `ready`).

Endpoints:

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | Chat completions (streaming, tools, images, video on GLM, `reasoning_effort`) |
| `POST /v1/messages` | Messages (streaming, tools, images, thinking) |
| `GET /v1/models` | Model list |
| `GET /health`, `GET /live` | Daemon state (`loading`, `ready`, `sleeping`, ...) and liveness |
| `POST /release_memory_occupation`, `POST /resume_memory_occupation` | Sleep / wake (below) |
| `POST /flush_cache` | Drop cached prompt state (benchmarks); 400 while a request runs |

Reasoning (DeepSeek): thinking is on by default at effort 75 (the model's documented default, "high"). `reasoning_effort`
accepts `none` (no thinking), `minimal` (25), `low` (50), `medium`/`high` (75), `xhigh`/`max` (100) or a number
1–100; `chat_template_kwargs.thinking=false` also turns thinking off. An optional `max_thinking_tokens` (alias
`thinking_token_budget`) caps the reasoning length: when the cap is reached the server ends the reasoning with a
short transition phrase and the model writes its answer. The model was not trained with truncated reasoning, so
the cap is off unless a request asks for it.

Reasoning (GLM-5.3-Flash): the model always thinks, at `low`, `high` (the default when nothing is asked) or `max`. `none`,
`minimal` and thinking off map to `low` (or to an empty thinking block with `HIVE_GLM_NOTHINK=empty`), `medium` to `high`,
`xhigh` to `max`; a number below 50 is `low`, below 90 `high`, otherwise `max`. `max_thinking_tokens` works the same way.

## Exposure and authentication

The server has no accounts. Whoever can reach the port can run inference and call the control endpoints
(`/flush_cache`, sleep/wake). The entrypoint therefore binds `127.0.0.1` unless `HIVE_BIND` says otherwise (the
container uses the host network, so `HIVE_BIND=0.0.0.0` exposes the port to every interface of the host). To expose it:

- set `HIVE_API_KEY=<secret>` — every endpoint except `/health` then requires `Authorization: Bearer <secret>` (or
  `x-api-key: <secret>`); or put a reverse proxy that authenticates in front and keep the loopback bind;
- `HIVE_MAX_BODY_MB` (default 64) rejects larger request bodies; `HIVE_IMAGE_FETCH=0` limits image (and GLM video) inputs
  to `data:` URLs (by default `http(s)` URLs are fetched by the server).

The request log (`$HIVE_STATE_DIR/logs/requests-server.jsonl`, off when `HIVE_REQUEST_LOG` is set to an empty value) records
timings, token counts, the client's user-agent, headers matching `HIVE_CLIENT_TAG_PREFIXES`, and the request body's `user`
field and `metadata`; credential-shaped header names are never written, and prompt text is not logged.

## SSD mode (less RAM, DeepSeek only)

```bash
echo HIVE_ENGRAM_SSD=1 >> ~/hive.local.env
echo HIVE_RAM_WAIT_GB=360 >> ~/hive.local.env
```

The ~195 GB Engram value tables then stay in the checkpoint on disk; the daemon reads the rows each token needs
through a 2 GiB row cache. Measured: resident memory 478 → 295 GB, start-up 90 → 59 s, throughput −1 to −6 %,
byte-identical Engram lookups (outputs still vary run to run, see [limitations.md](limitations.md)). `HIVE_RAM_WAIT_GB` is
how much free RAM the start script waits for before starting; set it below the installed RAM (the script clamps it to
MemTotal − 8 GiB).

## Sleep and wake (sharing the GPU)

`scripts/hive-sleep.sh sleep` gives the expert cache's VRAM back without losing the loaded model or open
conversations; `scripts/hive-sleep.sh wake` takes it again (a few seconds). Requests that are running finish
before the daemon sleeps; requests that arrive while it sleeps wait in its queue. `HIVE_SLEEP_LEVEL=3` also moves
dense weights to host memory (VRAM use while asleep about 1 GB — 1.2 GB measured on DeepSeek with a 120K-token
conversation moved to RAM); it needs a daemon started with `HIVE_SLEEP_VMM=1` (set in both shipped configurations) or
`HIVE_START_ASLEEP=1` — otherwise the daemon sleeps at level 2.
Hooks (`HIVE_HOOK_BEFORE_SLEEP`, `HIVE_HOOK_AFTER_WAKE`, ...) let a front proxy stop and resume routing traffic.

## Monitoring

With the monitoring switches on (see the end of `config/hive.env`), `tools/hive_monitor.py` follows
`hived.log` and writes per-request and per-minute records, and `tools/hive_daily_report.py` turns them into a daily
report (latency, cache hit rate, speculative decoding, reasoning length, errors).

The monitor follows `hived.log` by inode and logical offset. Do not rotate that file with copy-truncate tools while it runs; if
you rotate it, cut the front with `fallocate --collapse-range` and record `{inode, mode, logical_start, cut, archive}`
in `hived.log.rotations.jsonl` while holding a `flock` on `hived.log.lock`, as described at the top of `tools/hive_monitor.py`, or simply let the file grow (it is append-only text).

## Running as a service

`scripts/hive-start.sh` starts a detached container and returns; a systemd unit can call it as `ExecStart`
(`Type=oneshot`, `RemainAfterExit=yes`) and `scripts/hive-stop.sh` as `ExecStop`.
