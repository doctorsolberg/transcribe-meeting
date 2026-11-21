# Changelog

Format — [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning — [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `readable` supports OpenAI-compatible Chat Completions, JSON object mode, and a configurable batch size. The default remains Responses with strict JSON schema.
- `setup` shows download progress and accepts a trusted HTTPS mirror for GigaAM checkpoints.
- `TRANSCRIBE_HOME` can be set in `.env` to keep models and results outside the checkout without passing `--data-root` on every run.
- `setup --with-vad` provisions optional overlap detection for network-only ASR. A complete local model cache can be verified without Hub access or an HF token.

### Changed

- `setup` defaults to `foxnose`, matching `run`. Use `setup --diarization-provider all` for both local diarizers.
- Moved the historical measurements to [benchmark notes](docs/benchmarks.md). The default cleanup modes remain `raw,light,readable`.
- CI installs with `uv sync --locked` and checks that `doctor` reports an unprovisioned checkout as not ready. Package builds do not validate standalone installation or inference.

### Fixed

- Local model stages read working WAV files without requiring a torchcodec FFmpeg decoder.
- `setup` explains Hugging Face access and model download failures, handles new data directories, and records model verification only after smoke tests pass.
- `resume` checks the inputs for individual checkpoints instead of reusing cached outputs based on file existence alone.
- `resume` verifies the published canonical transcript and repairs derived files without re-reading the original audio or rerunning recognition. Start a new job to reprocess audio.
- A missing readable key no longer drops the requested mode; a later `resume` can retry it. Failed batches can leave a first readable version partial, but do not replace a complete accepted version.
- `resume` restores readable exports added by `clean`, and later LLM passes leave text corrected by `refine` untouched.
- Readable requires HTTPS for remote endpoints and does not follow redirects. Literal loopback addresses can be used over HTTP and without a key.
- The offline comparison scripts report undefined metrics on empty input, show ASR variant coverage, and use exact speaker matching without enumerating permutations.

## [1.0.0] — 2026-08-23

First public release. Everything below is relative to internal version
0.1.0.

### Fixed

- **`readable` no longer loses an entire batch of utterances because of
  one.** The protected-value check (numbers, links, email addresses,
  glossary terms) ran before the line-by-line processing started and
  raised an exception for the whole 150-utterance batch; the values were
  taken from the union of all ASR models, so a number heard by only one of
  them had to appear verbatim in the fused text — even though the prompt's
  own rules explicitly allow preferring another model's variant. On a real
  run with three models, this took down `readable` on every batch. Now
  protected values are checked against the canonical utterance, and a
  violation (like an empty response, or a rewrite exceeding 40%) rolls
  back only **one** utterance to the local text. Only a mismatch in the
  `utterance_id` set fails the whole batch — the one case where retrying
  the request actually makes sense.
- **`status` and `doctor` printed invalid JSON.** Output went through
  `rich`, which wraps long lines to the terminal width; a line wrapped in
  the middle of a value doesn't parse. `status`, `doctor`, and `auth status`
  write JSON to plain `stdout`; `glossary list` writes YAML.
- **Hints on `stderr` broke mid-command** and couldn't be copy-pasted —
  `soft_wrap` was added.
- **`clean`, `review`, `resume`, `rename` on a nonexistent job folder**
  produced a traceback instead of a clear error.
- **`run` no longer shows a traceback on a pipeline failure**: the full
  text is written to `<job>/error.log`, the user gets a short message and
  the `resume` command. Usage errors are still distinguishable by exit
  code (2 versus 1).
- **Temporary ASR WAV chunks weren't deleted** with
  `--no-keep-working-audio`, despite the documentation — 184 MB for a
  35-minute meeting with three models. The JSON checkpoints stay: `resume`
  needs them.
- **Two runs in the same second with the same `--meeting-name`** failed
  when creating the job folder; a numeric suffix is now appended.
- **A shared temporary diagnostics file** interfered with concurrent runs
  and was left behind as litter on an early failure.
- **The Deepgram request ran without a timeout** — a hung connection would
  freeze the run forever.
- **A `.pyc` file was sitting in the repository.**

### Changed

- **Context at VAD chunk edges.** On a real recording, 42% of chunks ended
  exactly at the 22 s target — mid-speech, cutting a word in half. Now
  every chunk is decoded with a 0.5 s margin on each side, and words are
  split between neighboring chunks by word center, so nothing duplicates.
  The chunk's hard ceiling dropped from 24 to 23.5 s, so the margin fits
  under GigaAM's 25 s limit.
  **Being honest about the effect:** on a two-hour meeting, this change
  turned out to be neutral on the only metric available — agreement
  between the two models went 89.33% → 89.42%, which is within noise. The
  metric is also blind to this particular defect: both models got cut at
  the same seams, so their mistakes at those seams matched. The change was
  kept because it removes a real word-corruption mechanism and doesn't
  hurt the result, not because it's measurably better. The project has no
  reference transcript that could prove it either way.
- **Deepgram now returns formatted text.** Words come from
  `punctuated_word` (casing, punctuation, digits, from `smart_format`)
  instead of the raw `word` field — the whole reason this route is
  recommended. The unused `utterances=true` was dropped from the request.
  The `diarize` parameter is deliberately NOT sent: `diarize_model` turns
  on diarization by itself, and the API rejects the combination
  (`diarize_model cannot be used together with diarize or
  diarize_version`) — verified with a live request and pinned down by a
  test.
- **`light` no longer mangles the text.** A period is no longer appended
  to every utterance (an utterance is a fragment, and merging utterances
  from the same speaker used to plant a period mid-phrase); a capital
  letter and a single closing period are now applied to the whole block.
  This doesn't restore punctuation inside the phrase — `readable` still
  owns readability.
- **Groq: an HTTP 429 no longer takes down the recognition stage.** The
  request retries up to three times with backoff, preferring the
  response's own `Retry-After` over its own estimate. Errors unrelated to
  the rate limit (like a bad key) still fail immediately.
- **The `--pause-between-chunks` option was removed, along with all its
  logic.** It applied to local GigaAM chunks, where it added about 3.8
  hours of pure `sleep` on a two-hour meeting under `--profile
  background` (measured on identical input: 13.2 min versus 3.5 min after
  removal), and never applied on the network path at all. There's nothing
  left for it to protect: the documented free-tier limits are held by
  `GroqFreeTierLimiter` — precisely, and only when a breach is actually
  about to happen — and an HTTP 429 now retries with backoff instead. The
  `pause` field was removed from the resource profiles, and the
  `pause_between_chunks` key from the job configuration.
- **The VAD plan is now computed once per recording, not per model.** The
  cache key included the ASR model name, even though the plan doesn't
  depend on it — on a real job, the two models' plans match byte for
  byte. A run with two local models used to run segmentation over every
  part twice, for nothing.
- **Architecture.** `provisioning.py` (model downloads, smoke tests,
  `models.lock.json`) was split out of `cli.py`; `job.py` (job artifacts,
  the `canonical.json` schema, stages) and `review.py` (export and
  intake of edits) were split out of `core.py`. `cli.py` shrank from 1444
  to 1199 lines, `core.py` from 1295 to 1039; `cli.py`'s coverage rose
  from 54% to 65%.
- **Style and linting.** `ruff format` was run; the `E, W, F, I, UP, B,
  SIM, RUF` rule sets are now enabled — the declared 100-character line
  width is now actually enforced. Only `RUF001/002/003` (which flag
  Cyrillic as Latin-script homoglyphs — a fundamental property of a
  Russian-language project) and `E501` for `readable.py` (wrapping a line
  inside the verbatim prompt would mean changing what's actually sent to
  the model) are suppressed.
- **CI.** Added `ruff format --check`, package build (`uv build`), and a
  coverage floor of `--cov-fail-under=75`; switched to `uv sync --frozen`
  instead of silently re-resolving dependencies.
- The project version is now read from the package metadata in a single
  place — previously it lived in three, and they drifted (`0.1.0` in two
  places and `0.2.0` in the job manifest).

### Added

- A global `--version` flag.
- A free-space check before starting a run: the needed volume is now
  estimated from the recording's duration and the number of local models,
  instead of `setup`'s fixed 7 GB. Previously, running out of space was
  discovered mid-run, as an unhelpful ffmpeg error.
- A warning for an uncalibrated `--asr-models` order: the fusion rules are
  tied to slots "a" and "b," and swapping the models silently flipped the
  prompt's hints in the wrong direction.
- `scripts/model_agreement.py` and `scripts/compare_routes.py` — tools to
  compare stored jobs offline; see [benchmark notes](docs/benchmarks.md).
- The `TRANSCRIBE_READABLE_REASONING` environment variable
  (`none|low|medium|high`, default `medium`). The `reasoning` value used to
  be sent unconditionally and broke OpenAI-compatible endpoints that don't
  support it — even though switching endpoints via `OPENAI_BASE_URL` is
  documented as a supported use case.
- Tests: 320 versus 284, coverage 80% versus 76%. Two key gaps were
  closed — `_fuse_batch` (the sole network contract behind `readable`'s
  quality) and `run_readable`. Added a contract test that checks the
  README's option tables against the CLI's real parameters in both
  directions — this project's documentation had already drifted from the
  code once without it, and would drift again.
- Audio preprocessing tests that call real FFmpeg skip when it is absent.
- `numpy` is now declared as an explicit dependency — the code imports it
  directly, but it had been riding along transitively.

### Verified with Real Runs

A 113.7 min meeting (two files, three participants) was run through three
routes:

- 2 local models + Groq + `foxnose` + `readable` — 754 utterances,
  `readable` completed on all six batches, zero rollbacks to local text.
  This is the exact configuration where `readable` used to fail
  completely;
- `deepgram-nova-3` + `deepgram` diarization — one request for both
  stages, text came back with punctuation and casing, 3 participants
  requested, 4 found — a warning, not an error;
- `v3_rnnt` + `pyannote` — `regular` ≠ `exclusive` (1175 versus 1039
  segments).

The routes were compared pairwise: speaker labeling agrees at 92–95%,
duration is identical, speaking time per participant differs by 2–3
minutes out of 15–57.
