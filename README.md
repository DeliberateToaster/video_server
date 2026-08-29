# WanFlash

A REST API for Wan 2.2 video generation. One model resident in VRAM, one
generation at a time, asynchronous job model.

Technical specification (in Finnish): [docs/spec.md](docs/spec.md).

## Installation

The quickest route is the bootstrap script: it installs uv if missing, installs
dependencies, creates `.env`, checks the hardware and runs the test suite.
Python 3.12 comes with uv, so no separate Python install is needed.

```powershell
# Windows. -ExecutionPolicy Bypass is needed if the machine blocks
# unsigned scripts.
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

```bash
# Linux / macOS
bash scripts/bootstrap.sh
```

Options (both scripts): `-Weights` / `--weights` also downloads the model
weights (~32 GB), `-Minimal` / `--minimal` installs only the API dependencies
without torch, `-SkipTests` / `--skip-tests` skips the test run.

The script is idempotent: it is safe to re-run and will not overwrite an
existing `.env`.

### Manual

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed automatically.

```powershell
# API + development tools, no GPU dependencies (~50 MB).
# This is enough to run the mock backend and the whole test suite.
uv sync --group dev

# GPU dependencies (torch + diffusers, ~3 GB). Only needed for real
# inference, not for working on the API.
uv sync --extra gpu --group dev

# Only for the low tier (4-bit quantisation, 12-20 GB VRAM):
# uv sync --extra gpu --extra quantized --group dev

# Check whether the machine is ready for real inference (GPU, CUDA, ffmpeg, tier)
uv run python scripts/check_env.py
```

The dependency split is deliberate: the entire API, queue and test suite run
without a GPU and without torch. Only `backends/wan22.py` needs the `gpu` extra.

## Model weights

The server does **not** download weights by itself. Missing weights produce a
clear error naming the command to run - silently pulling tens of gigabytes on
the first API call would be a poor default.

```powershell
# Default model TI2V-5B. Note: 31.9 GB of disk space (the repo ships
# fp32 weights). Runtime VRAM usage is about 23 GB - disk size and VRAM
# are two different things.
uv run python scripts/download_model.py wan2.2-ti2v-5b

# Already downloaded?
uv run python scripts/download_model.py --check
```

Available profiles: `wan2.2-ti2v-5b` (default), `wan2.2-t2v-a14b`,
`wan2.2-i2v-a14b`. A14B is a MoE model whose weights are ~54 GB in bf16 - it
does not fit a 24 GB card even with CPU offload unless the machine has ~60 GB
of system RAM.

## Hardware and tiers

By default `VIDEO_SERVER_BACKEND=auto`: the server detects VRAM and system
memory, derives a tier, and uses it to pick both the model and the loading
mode. The choice and its reasoning are logged at startup.

| Tier | VRAM | Model and loading mode |
|---|---|---|
| `low` | 12-20 GB | TI2V-5B 4-bit quantised (needs the `quantized` extra, untested) |
| `mid` | 20-30 GB | TI2V-5B bf16 |
| `high` | 30 GB+ | A14B bf16 |
| `a14b-offload` | 24 GB + ~60 GB RAM | A14B with CPU offload, slow |

An explicit `VIDEO_SERVER_BACKEND` always overrides the automatic choice. If no
model fits the detected tier (no CUDA, or a card that is too small), the server
fails with instructions rather than silently falling back to the mock backend
and serving fabricated video.

An RTX 3090 (24 GB) lands in the mid tier: TI2V-5B bf16, ~23 GB VRAM in use.

## Running

```powershell
uv run uvicorn video_server.main:app --reload

# Development without a GPU and without weights:
$env:VIDEO_SERVER_BACKEND = "mock"; uv run uvicorn video_server.main:app --reload
```

The server opens its port immediately, but generation endpoints return `503`
until the model has loaded. Check with `GET /api/v1/health`.

Configuration: copy [.env.example](.env.example) to `.env`. Every setting also
works as an environment variable or via `config.yaml`.

## Example

```powershell
# Start a generation
curl -X POST http://127.0.0.1:8000/api/v1/txt2vid `
  -H "Content-Type: application/json" `
  -d '{\"prompt\":\"a cat walking on a beach\",\"num_frames\":81}'

# Poll the status (job_id from the previous response)
curl http://127.0.0.1:8000/api/v1/jobs/<job_id>

# The finished video is in the video_url field of the response
```

No guessing is needed about what the server accepts for a given model:
`GET /api/v1/models` reports the active model's allowed resolutions, frame
rule and native fps.

## Optional extensions

Both are off by default and do not change the server's behaviour unless you
explicitly enable them.

**Preview frames.** `VIDEO_SERVER_PREVIEW_EVERY_N_STEPS=5` decodes one frame
every fifth step and adds a `preview_url` field to the job response. Costs one
extra VAE call per image. If decoding fails (for example when memory runs out),
generation continues normally and a warning is logged.

**API key.** `VIDEO_SERVER_API_KEY=...` requires an `X-API-Key` header on
`/api/v1` and `/outputs`. Finished videos are protected too, since they are the
actual sensitive content. `/api/v1/health` stays open so monitoring works
without the key.

```powershell
curl http://127.0.0.1:8000/api/v1/models -H "X-API-Key: your-secret-key"
```

## Tests

```powershell
uv run pytest          # full suite, needs no GPU and no weights
uv run ruff check .
```

## Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `WeightsMissingError` on startup | Weights have not been downloaded | `uv run python scripts/download_model.py` |
| `CAS Client Error: ... error decoding response body` | HuggingFace Xet transfer dropped mid-download | The script retries automatically and resumes where it stopped. If the retries run out, run the command again - completed files are kept |
| `ModuleNotFoundError: No module named torch` | GPU dependencies are not installed | `uv sync --extra gpu --group dev` |
| `503` on a generation request | The model is still loading (takes minutes) | Wait; `GET /api/v1/health` reports the state |
| `400` mentioning `num_frames` and `n * 4 + 1` | The frame count is invalid for the model's VAE | Use one of the nearest valid values the error suggests |
| `torch.OutOfMemoryError` / CUDA OOM | The model does not fit in VRAM | `VIDEO_SERVER_CPU_OFFLOAD=true`, a smaller resolution, or fewer frames |
| `ValueError: guidance_scale_2 is only supported when ... boundary_ratio is not None` | Dual guidance passed to a single-expert model | Should not happen: the profile's `has_second_expert` prevents it. If it does, the profile is wrong |
| Server responds `429` | The queue is full | Wait, or raise `VIDEO_SERVER_MAX_QUEUE_SIZE` |

## Status

Phases 0-4 complete: environment, the API on the mock backend, the Wan backend,
tier automation, plus optional preview frames and API key. Verified with a real
run (RTX 3090, 1280x704). Every phase of the specification has been worked
through.

## Known limitations

- No ROCm (AMD) support. CUDA only.
- Authentication is off by default. If the server is exposed to a network, set
  `VIDEO_SERVER_API_KEY`.
- No concurrent runs: one model and one generation at a time.
- The `phase` field in progress distinguishes the denoising and decoding
  stages. Decoding duration cannot be estimated, so `eta_seconds` is `null`
  during it.
- Job state lives in memory; finished videos persist on disk with sidecar
  metadata, so a restart does not lose completed results.

## A note on language

User-facing documentation, configuration comments and script output are in
English. Code comments and the technical specification remain in Finnish,
reflecting how the project was designed and reasoned about.
