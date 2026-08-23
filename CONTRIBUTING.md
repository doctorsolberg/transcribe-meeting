# Contributing

## Setup

```bash
uv sync --python 3.12 --group dev
```

Python 3.12 and 3.13 are both supported (`requires-python = ">=3.12,<3.14"`
in `pyproject.toml`; CI runs both). `ffmpeg`/`ffprobe` are needed first and
foremost by `setup` and `run` (`check_ffmpeg`), but not only by them: two
tests in `tests/test_audio_preprocess.py` call the real binary directly
(with no stub), instead of reproducing its behavior through monkeypatch.
Both are marked with `requires_ffmpeg` and get skipped if `ffmpeg` isn't
found on `PATH` — the rest of the pytest suite doesn't depend on the real
binary.

## Tests and Linting

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

(equivalent to `uv run pytest -q` / `uv run ruff check .`, if you'd rather
not activate `.venv` by hand).

Both suites being green/clean on every commit is the only thing CI
(`.github/workflows/ci.yml`) checks automatically, on `ubuntu-latest` and
`macos-latest`.

## Test Coverage

`pytest-cov` is in the dev dependency group, but `--cov` isn't on by
default (it would slow down a regular run); to see coverage, run it
separately:

```bash
.venv/bin/python -m pytest -q --cov=transcribe_meeting --cov-report=term-missing
```

CI runs the same report as a separate step with a `--cov-fail-under=75`
threshold: this isn't a target, it's a floor — dropping below it fails the
build, not just prints a smaller number in the log. Actual coverage right
now is 80%.

Coverage below 100% is expected, and concentrated where the code, by its
nature, needs things CI doesn't have: `provisioning.py` (15%) downloads
models and runs smoke tests against them, `audio.py` (73%) calls the real
`ffmpeg`/`ffprobe`, `diarization.py` (68%) loads the local
`pyannote.audio`, and the uncovered part of `cli.py` (65%) is the branches
of `setup` and `run` that reach the models and the network. See "What
Tests and CI Don't Do" below.

## Measurement Tools

`scripts/model_agreement.py` and `scripts/compare_routes.py` aren't part of
the package and aren't covered by tests: they're tools for re-checking the
numbers in the README's "Measured Facts" section on real jobs. They only
read a finished job's `canonical.json` and `manifest.json`, write nothing,
and never touch the network.

## What Tests and CI Don't Do

No test downloads a model or touches the network — the suite runs on
fixtures, generated WAV files, and stubbed ASR/diarization backends (see,
for example, `tests/test_end_to_end_pipeline.py` — the only test that runs
the whole pipeline). The suite's one external dependency is the real
`ffmpeg` used by the two tests marked `requires_ffmpeg` (see "Setup"
above); without it they're skipped, not failed. Real runs against the
actual models (GigaAM, VAD, pyannote, WeSpeaker) and real calls to the
network backends (Groq, Deepgram, OpenAI/readable) need `setup` and API
keys, and **never run in CI at all** — you can only check them locally,
with `transcribe-meeting doctor` (no network, checks what's already
downloaded) and then `transcribe-meeting run` on real audio.

## Style

- Code comments are in English, and only about "why," never "what" (the
  code already shows "what").
- User-facing CLI messages and documentation are in English.
- `ruff` — `line-length = 100`, `target-version = "py312"` (see
  `pyproject.toml`).
- Never commit anything into `output/`, `models/`, `.env` — they're
  already excluded in `.gitignore`.
