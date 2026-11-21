# Contributing

## Setup

From the repository checkout, use the locked `uv` environment so the Torch and
torchaudio overrides in [`pyproject.toml`](pyproject.toml) apply:

```bash
uv sync --locked --python 3.12 --group dev
```

CI tests Python 3.12 and 3.13 on macOS and Ubuntu. Install `ffmpeg` and
`ffprobe` for `setup`, `run`, and tests that exercise the real audio tools.

## Tests and Linting

```bash
uv lock --check
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

CI also builds wheel and source artifacts, checks `doctor` against an empty
model directory, and has a 75% coverage floor. Building an artifact is not a
standalone install or inference test. To see coverage locally:

```bash
uv run --no-sync pytest -q --cov=transcribe_meeting --cov-report=term-missing
```

## Real model checks

CI does not download weights, run model inference, or call providers. For changes
to installation, audio decoding, or checkpoints, use a separate data root with
locally provisioned models and run a short, non-sensitive recording with
`--clean raw,light`. Inspect the transcript, speaker segments, and `result.json`,
then check `resume`. Record the platform, Python version, and route when reporting
results. See [benchmarks](docs/benchmarks.md) for the comparison scripts.

## Style

- Code comments are in English, and only about "why," never "what" (the
  code already shows "what").
- User-facing CLI messages and documentation are in English.
- `ruff` — `line-length = 100`, `target-version = "py312"` (see
  `pyproject.toml`).
- Do not commit real recordings, job output, model caches, credentials, or
  private plans. `.gitignore` covers the default `output/`, `models/`, `.env`,
  and `.hermes/` paths, not every custom path.
