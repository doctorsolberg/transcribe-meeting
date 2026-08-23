# transcribe-meeting

A local, resumable meeting transcriber. On Apple Silicon it uses MPS; on
other platforms (including a Linux server) it uses CPU — CUDA is not used
and not needed. By default the audio stays on the local machine: ASR runs by
default through **two** GigaAM models — `multilingual_large_ctc` and
`v3_rnnt` (see "Multiple ASR Models and Fusion" below), VAD uses
`pyannote/segmentation-3.0`, and diarization on CPU defaults to `foxnose`
(for details and the alternative `pyannote` route, see "Diarization: Two
Routes").

## Language support

This project is built for **Russian speech**. The default route is Russian-only:
GigaAM accepts `--language ru` or `auto` and nothing else, and both text-cleanup
stages are tuned for Russian — the filler list `light` strips is Russian, and the
fusion prompt `readable` sends to the LLM is written in Russian, with Russian
examples of the exact mistakes these models make.

Other languages work through the network routes. `--asr-models
groq-whisper-large-v3` takes any language Whisper supports, and
`--asr-models deepgram-nova-3` covers 45+ languages, with `--language auto`
switching nova-3 into code-switching mode across ten of them. Diarization is
language-independent on every route.

What to expect outside Russian: transcription is as good as the provider you
picked, `raw` is fine, but `light` will leave foreign fillers in place and
`readable` will be working against a prompt that argues about Russian morphology.
Use `--clean raw` there, or adapt `READABLE_FUSION_INSTRUCTIONS` in
`transcribe_meeting/readable.py` to your language.

## Table of Contents

- [Language support](#language-support)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Output Formats and Artifacts](#output-formats-and-artifacts)
- [Quality Checks](#quality-checks)
- [Operations](#operations)
- [All Parameters](#all-parameters)
- [Measured Facts](#measured-facts)
- [Known Limitations](#known-limitations)
- [License](#license)

## Requirements

- **Python 3.12 or 3.13** — `pyproject.toml` sets `requires-python =
  ">=3.12,<3.14"`, and CI runs the full test suite on both. Model inference
  (GigaAM, VAD, diarization) was verified by the author on 3.12; on 3.13 only
  the tests and the linter are confirmed.
- **FFmpeg and FFprobe** on `PATH` — a hard requirement: `setup` and the
  audio preparation before `run` both call `check_ffmpeg()`, which stops the
  run if the binaries aren't found.
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `apt install ffmpeg`
- **`uv`** — for installing dependencies and running the tool (`uv sync`,
  `uv run`).
- **At least 7 GB** of free space for the models (GigaAM, VAD, the chosen
  diarization route, the working WAV) — `setup` checks this before
  downloading.
- **No CUDA needed**: Apple Silicon uses MPS, other platforms use CPU.

## Installation

```bash
uv sync --python 3.12 --group dev
```

On a server (without the `dev` group, which is needed only for tests and
linting):

```bash
uv sync --python 3.12
```

`uv.lock` is multi-platform — it carries `manylinux_2_28_x86_64` and
`manylinux_2_28_aarch64` wheels, so `uv sync` on a typical Linux server
resolves without building from source. Data directories are set by the
`TRANSCRIBE_HOME` / `TRANSCRIBE_OUTPUT_DIR` / `TRANSCRIBE_MODELS_DIR`
environment variables, or by the equivalent global options `--data-root` /
`--output-dir` / `--models-dir` (more detail in "Data Directories and
Running Under an Agent" under "Operations").

To be honest: CI (`.github/workflows/ci.yml`) checks dependency installation,
ruff, and the full pytest suite on `ubuntu-latest` and `macos-latest` — this
is the project's first real run on Linux, ever. What CI does not check:
actually running the models (GigaAM, VAD, pyannote, WeSpeaker) or the network
backends (Groq, Deepgram, OpenAI) — `setup` never runs in CI, and no test
downloads a model or touches the network. A real-audio run on Linux has never
happened; the claim that the models work there rests on `uv.lock`'s wheel
list (`manylinux_2_28_*`) and on the absence of anything platform-specific in
the code (`--asr-device auto` picks CPU on its own wherever MPS isn't
available) — not on an actual run.

`setup` downloads GigaAM into `models/gigaam/`, VAD into
`models/huggingface/`, and, depending on `--diarization-provider`, WeSpeaker
into `models/wespeaker/` and/or Community-1 into
`models/speaker-diarization-community-1/` (both by default). It checks each
checkpoint's MD5 and stores SHA-256 hashes, revisions, and device
smoke-test results in `models.lock.json`. Once `setup` succeeds you can
delete `HF_TOKEN`: a normal local run never touches the network. The token is
never written to the manifest or the logs.

`HF_TOKEN` is only needed by the `setup` command — for exactly what and why
it is needed on every diarization route, see "Diarization: Two Routes"
below. You can store it three ways: as an environment variable, via `auth set
huggingface` (a no-echo prompt), or as an `HF_TOKEN=…` line in the project's
`.env` — the file is created with `600` permissions, is listed in
`.gitignore`, and is read only through a name allowlist. An empty
`HF_TOKEN=` counts as a placeholder and does not define the variable. After
`setup` succeeds you can clear the value again.

Models you've already downloaded can be moved to another machine without
rerunning `setup`: copy the whole `models/` directory, including the
`models.lock.json` inside it — its paths are recorded relative to `models/`
(see `store_model_path`/`resolve_model_path` in `paths.py`), so the
directory can move anywhere as-is, as long as `--models-dir`/
`TRANSCRIBE_MODELS_DIR` point at it. The alternative is to run `setup`
directly on the target machine with `HF_TOKEN` in the environment.

## Quick Start

```bash
uv run transcribe-meeting doctor
uv run transcribe-meeting auth set huggingface   # or put HF_TOKEN in .env
uv run transcribe-meeting setup --diarization-provider foxnose
uv run transcribe-meeting run src/1.m4a src/2.m4a \
  --meeting-name example --speakers 3 --asr-device auto
```

## How the Pipeline Works

### Diagnostics and Loudness

Before the run, each part is checked with `ffprobe`: format, codec, channels,
sample rate, duration, track selection. This is instant.

Measuring loudness (LUFS, true peak, LRA) needs a full FFmpeg pass over the
file — about a minute per hour of recording — so it's **off by default**. It
turns on with `--measure-loudness` and lands in `diagnostics.json`. It
doesn't affect recognition: the pipeline has no loudness normalization, and
`--asr-preprocess` (80 Hz highpass + pre-emphasis) and `--asr-compression`
(4:1) apply later and don't depend on this measurement. It's a reference for
a human: see that a recording is too quiet or clipping, and decide whether to
turn preprocessing on.

### Local ASR

Local ASR is pinned to GigaAM commit
`7447938d791c4f3e643386ee22c33777004293a5`. The model registry is
`multilingual_large_ctc`, `v3_rnnt`, `v3_ctc`, `v2_rnnt`, `v2_ctc` (the
`v3_e2e_*` e2e models are deliberately unsupported: they need a separate
tokenizer file that `setup` doesn't download). These aren't e2e models:
`raw` doesn't promise casing or full punctuation. `light` does only
mechanical local cleanup; `readable` stays a separate text-only stage and
never receives audio.

Before ASR, each source part is decoded to mono PCM 16 kHz, and the local
VAD builds deterministic chunks: a 22 s target ceiling, a 23.5 s hard
ceiling. Each chunk is additionally decoded with a 0.5 s margin on each
side, so a word sitting on a boundary is heard in full; words from that
margin go to whichever neighboring chunk owns the word's center, so nothing
duplicates. The margin plus the hard ceiling stay under GigaAM's 25 s limit.
Each chunk has an atomic JSON checkpoint under `asr/<model>/chunks/<part>/`;
its words get timestamps only from GigaAM and are then shifted into the
source-local clock. Silent speech produces a valid, empty ASR artifact.
Diarization runs once, on the whole concatenated track, after GigaAM is
released.

`--asr-device auto|mps|cpu` controls GigaAM. `auto` picks MPS if PyTorch
exposes it, CPU otherwise; an explicit `mps` fails if it isn't available.
The result and its fingerprint record the requested/effective device, the
model, its checksum, the GigaAM commit, the VAD revision, and the
parameters, so incompatible checkpoints never mix. `--language` for local
only accepts `ru` or `auto`; the glossary and the context prompt apply to
the deterministic text processing/readable stage, but are never sent to the
GigaAM API.

### Groq (explicit opt-in)

The only network ASR model is `groq-whisper-large-v3`; it turns on only when
you name it explicitly in `--asr-models` (or through the deprecated
`--asr-provider groq`, see "All Parameters"), and it sends the prepared WAV
to the Groq API. Diarization still stays local.

```bash
export GROQ_API_KEY='…'
uv run transcribe-meeting run src/1.m4a --meeting-name external \
  --speakers 3 --asr-models groq-whisper-large-v3 --chunk-minutes 10
```

`--chunk-minutes` is for Groq only. Local uses VAD chunks, so passing this
flag explicitly for local is an error. For Groq, free-tier limits,
timestamped root-level words, and resume are all preserved.

Limits are held automatically, in two layers. `GroqFreeTierLimiter` waits
exactly as long as needed to stay under the documented 20 requests per
minute and 7200 seconds of audio per hour, and only when a breach would
actually happen. If the account still gets an HTTP 429, the request retries
up to three times with backoff — preferring the value in the `Retry-After`
header over its own estimate. A single 429 in the middle of a two-hour
meeting used to bring down the entire recognition stage; the chunks already
done survived as checkpoints, but the run had to be restarted by hand.

### Deepgram (explicit opt-in): transcription and diarization in one request

`--asr-models deepgram-nova-3` and `--diarization-provider deepgram` turn on
an external route through the Deepgram API. **Audio leaves the machine** —
this is the only route, alongside Groq, where audio goes out instead of
text.

Why it's worth having: it's fast, and it understands languages GigaAM
doesn't. The `nova-3` model covers more than 45 languages, and its
code-switching mode (`--language auto`, sent to the API as `multi`) covers
10, including Russian. It's the right choice for multilingual recordings, or
when you need a result in a minute instead of half an hour. Words in the
response already carry casing, punctuation, and digits — the pipeline reads
them from the `punctuated_word` field (produced by `smart_format`), not from
the raw `word` field.

```bash
uv run transcribe-meeting run meeting.m4a --meeting-name Example --speakers 3 \
  --asr-models deepgram-nova-3 --diarization-provider deepgram
```

**One request serves both stages.** Deepgram returns words already tagged
with speaker labels, so the request goes out once, for the whole meeting's
concatenated track, and the raw response is saved to
`asr/deepgram-nova-3/meeting.json`, where it doubles as a checkpoint: the
diarization stage reads it back from disk and doesn't pay for it twice.
That's also why the request has to carry the concatenated track
specifically: if requests went out part by part, `speaker 0` in the first
part and `speaker 0` in the second would end up being different people.

Deepgram diarization can be paired with local recognition
(`--diarization-provider deepgram` without `deepgram-nova-3` in the model
list) — then the diarization stage makes the request on its own.

**You cannot set the number of participants** — the API has no such
parameter. `--speakers N` on this route becomes a hint: a mismatch produces
a warning, not an error, unlike the local routes.

Pricing at the time of testing was about $0.0048 per minute plus $0.0020 for
diarization — roughly $0.77 for a two-hour meeting. The `DEEPGRAM_API_KEY`
is set in `.env` or via `auth set deepgram`. The API caps requests at 2 GB;
the pipeline checks the size before sending.

### Multiple ASR Models and Fusion

`--asr-models NAME[,NAME[,NAME]]` sets which models to run (1–3); the
default is `multilingual_large_ctc,v3_rnnt` — **both**. Models load and
unload strictly one at a time (~5 GB RSS each; this machine can't hold two
at once), so ASR takes roughly twice as long: on a 35-minute recording,
about 161 s for one model plus about 242 s for the second (more detail
under "Speed" in "Measured Facts").

**The first model in the list is canonical**: only its words define
utterance boundaries, the set and order of `utterance_id`, timecodes, and
speaker attribution. Words from the other models are projected onto the
already-built canonical utterances by word center (`(start + end) / 2`), so
each word from the second model lands in exactly one utterance with no
duplication at the edges. `raw` and `light` are rendered separately per
model — `transcript.raw.<model>.md`, `transcript.light.<model>.md` —
line-comparable files that show where the models disagree.

Fusion into a single readable text happens only at the `readable` stage: the
LLM receives both utterances and picks between them word by word, following
a rule that encodes a measured error asymmetry (180 discrepancies on a real
recording, manually reviewed: `multilingual_large_ctc` is stronger on
anglicisms, terminology, and abbreviations; `v3_rnnt` is stronger on Russian
morphology and word boundaries). The result is always a single
`transcript.readable.md` file, with no model name in it.

**Model order in `--asr-models` is part of the calibration, not just
ordering.** The fusion rule states outright which model is more accurate on
anglicisms and which on Russian endings, and it's tied to slots: the first
model is "a," the second is "b." Swapping them won't break the run, but the
prompt will start telling the model the opposite of what was measured. The
default order (`multilingual_large_ctc,v3_rnnt`) is the one the asymmetry
was measured on; `readable` prints a warning for any other order.

A single model (`--asr-models v3_rnnt`) is the old single-pass behavior: no
fusion, no `alt_*` fields, no second model's files. Groq has no fusion
available — only one model is allowed there.

`setup --asr-models NAME[,NAME]` downloads the chosen checkpoints into
`models/gigaam/` (the same default `run` uses) and checks each one's MD5
before the smoke test.

Fusion is more than picking between variants. The model is explicitly
allowed to:

- **fix mangled words** when the conversation makes the intent unambiguous:
  `ранскрибаторе` → `транскрибаторе`, `самаризирую` → `саммаризирую`;
- **spell known English terms in Latin script**: `дипграм` → `Deepgram`,
  `гигаам` → `GigaAM`, `стейчгейт` → `Stage-Gate`, `ллм` → `LLM`, `сиэрэмка`
  → `CRM`.

It's strictly forbidden — and repeated in the prompt as its own rule — to
change numbers, dates, sums, names, links, and email addresses; to swap a
word for a different one in meaning; to paraphrase a sentence in the
model's own words; or to pull words from the meeting description.

This is a deliberate trade-off: rule precision against readability. There's
no 100% guarantee here — it rests on how strict the prompt is. That's why
**every changed word gets recorded** in `manifest.readable.concerns` as
"before → after," and the `repaired_utterances` and `fallback_utterances`
counters show the scale of the edits. If an utterance is rewritten by more
than 40%, that's no longer an edit, it's a rewrite: that utterance falls
back to the local `light` text. For how to check exactly what changed, see
"Change Report" under "Quality Checks."

Utterances go out in batches of 150: a two-hour meeting in one request risks
hitting the output limit and cutting off midway, which then no longer
matches by `utterance_id` and voids the whole pass. A batch that fails
validation twice leaves its own utterances with the local text; the other
batches aren't affected. If not a single batch passes, `readable` counts as
failed — not as "done" using nothing but local text.

### Diarization: Two Routes

`--diarization-provider foxnose|pyannote` picks how speakers get labeled.
The default is `foxnose`.

| | `foxnose` (default) | `pyannote` |
|---|---|---|
| Model | `diarize` (FoxNoseTech, PyPI) + WeSpeaker ResNet34-LM ONNX | `pyannote/speaker-diarization-community-1` |
| License / access | Apache-2.0, no need to accept Community-1's terms | gated, must accept terms on Hugging Face |
| Overlap within diarization itself (`regular` vs `exclusive`) | **doesn't model it** — they're identical | models it |
| Speaker-count estimation | degrades at 8+ participants — pass `--speakers N` explicitly | stable on typical meetings |
| Speed on a two-hour meeting (M1 Pro, CPU) | about 5 min | **about 95 min** — roughly twenty times longer |

`HF_TOKEN` is needed at the first `setup` either way:
`pyannote/segmentation-3.0` is used by local ASR's VAD regardless of which
diarization route you pick. That's the only reason `setup` without
`HF_TOKEN` fails even for the `foxnose` route, which by itself needs no
gated terms.

The `[overlapping speech]` mark in `light`/`readable` doesn't depend on the
chosen route: it comes from a separate `detect_overlap_regions` pass over
the same local `pyannote/segmentation-3.0` (the same model local ASR's VAD
already needs), not from `foxnose` or Community-1. A failure in that pass
doesn't bring down diarization: `diarization.json` gets `"overlap": []` and
an `"overlap_error"` field with the error text, and the command's output
reports it.

The speed difference is the main practical argument for defaulting to
`foxnose`: on a 113.7 min meeting, `pyannote` diarization took 95 minutes
against about 5 for `foxnose` (measured on M1 Pro, CPU; in both cases, on
top of that, the separate overlap-detection pass takes about 5 minutes).
Reach for `pyannote` when you specifically need its overlap-free
`exclusive` track or a more stable automatic speaker-count estimate on 8+
participants, and budget the time for it.

```bash
# Download only the model for the default route (leave Community-1 alone)
uv run transcribe-meeting setup --diarization-provider foxnose

# Run a job on pyannote if you need a more stable automatic speaker-count
# estimate at 8+ participants ([overlapping speech] is marked on any route)
uv run transcribe-meeting run src/1.m4a --meeting-name example \
  --speakers 3 --diarization-provider pyannote
```

The WeSpeaker model (~26 MB) downloads once into
`models/wespeaker/en/model.onnx` and is recorded in `models.lock.json`;
later runs are fully offline. `setup --diarization-provider all` (the
default for `setup`) downloads both diarization models and merges the
records into `models.lock.json`, without overwriting whatever another route
already downloaded.

## Output Formats and Artifacts

### Text Formats

`raw` is a line-by-line snapshot of the canonical JSON, one line per
utterance; it's what corrections get checked against. `light` and
`readable` **merge consecutive utterances from the same speaker into one
block**, so long speech with pauses doesn't fragment into dozens of lines; a
block's timecode runs from the start of its first utterance to the end of
its last. The merge exists only in the rendering — `canonical.json` keeps
every `utterance_id`, its timecodes, and its speaker ID, and a block never
crosses a source-file boundary. A `light` block gets a capital letter and
one closing period (the `[overlapping speech]` mark, if present, doesn't
get capitalized) — this doesn't restore punctuation inside the phrase; it's
purely block-edge formatting.

### readable by Default

`--clean` defaults to `raw,light,readable`, so `run` prepares the polished
LLM version right away. Audio never reaches readable — only text does. It's
the pipeline's second network stage (alongside the optional Groq ASR — see
"Groq (explicit opt-in)" above, where audio does go out) and the only one
enabled by default.

Degrade instead of fail: no `OPENAI_API_KEY` — `readable` is dropped before
work starts, with a warning; the model is unreachable or its response fails
validation — `raw` and `light` stay untouched, the reason is written to
`manifest.readable`, and the run still finishes successfully. Fill in
readable later:

```bash
uv run transcribe-meeting clean output/<job-id> --mode readable
```

The endpoint switches to any OpenAI-compatible one via `OPENAI_BASE_URL`
(`auth set-openai-base-url`); for endpoints without reasoning support, set
`TRANSCRIBE_READABLE_REASONING=none` (the default is `medium`; the other
values are `low` and `high`) — otherwise the request gets rejected.

`raw` and `light` are a separate file per ASR model:
`transcript.raw.<model>.md`, `transcript.light.<model>.md` (see "Multiple
ASR Models and Fusion"); `readable` is always a single fused
`transcript.readable.md` file. There's no combined `transcript.md` — the
filename always tells you exactly what's inside. If a mode fails to build,
its file simply isn't created, and the reason is printed in the command's
output.

`--context-prompt` is the meeting's topic and purpose as text, up to 2,000
characters. This is deliberately **only** a hint for readable cleanup (it's
passed to the model along with the utterance text and never checked as
fact): it doesn't affect ASR or diarization, because recognition shouldn't
guess words based on the expected topic. Groq is the exception, for
technical reasons rather than by design: its API supports a transcription
prompt hint, so `context_prompt` is additionally sent there in the ASR
request too; GigaAM (local) has no such parameter at all.

### Artifacts and Atomicity

Every stage and every JSON artifact is written atomically. `canonical.json`,
`speakers.yaml`, regular/exclusive diarization, RTTM, the three timelines,
and stable speaker IDs all keep their existing contract. Source audio files
are never modified or deleted.

## Quality Checks

For exactly what the model is allowed to change in readable (and the
rollback rule when an edit exceeds 40% of an utterance), see "Multiple ASR
Models and Fusion" above.

### Change Report

After every `readable` run, `review/readable_changes.csv` gets written —
one row per changed utterance, with the columns `utterance_id`, `time`,
`speaker`, `change_kinds`, `changed_words`, `needs_attention`, `before`,
`after`.

The file is built **by a script, with no model call**: edit distance for
`repair`, Latin characters for `transliteration`, set difference for
`loss`, and everything else falls into `substitution`. Auditing an LLM's
work with another LLM would just repeat its mistakes, so this is plain
arithmetic over two texts, nothing more. Regenerating it is free and works
offline on any old job.

Rows marked `substitution` or `loss` get `needs_attention: yes` and are
sorted first — they're worth a human look; `transliteration` and `repair`
are confirmed by computation and need no attention. The report is scoped to
utterances, not words, on purpose: a transliteration is physically a
deletion plus an insertion, and a word-level report would show `виспер` →
`Whisper` as a standalone "loss." Matched losses and gains cancel each
other out.

On the 35-minute recording: 297 utterances, 83 changed words across 61
utterances, of which 17 are flagged as needing attention, and 1 rollback.

### review and refine: Manual Editing

`review job_dir` dumps **every** utterance in the job to
`review/transcript_edit.csv` — with no confidence-based filtering: none of
the connected ASR providers (local GigaAM, Groq Whisper) return word
probabilities, so there's no confidence threshold to build one on, and
filtering to a subset would always produce an empty file. `review/index.md`
next to it explains the contents.

The CSV has two independent fields for edits:

- `replacement` — the corrected utterance text, if ASR got it wrong;
- `speaker_replacement` — the correct speaker name for this specific
  utterance, if diarization attributed it to the wrong person. It changes
  only this utterance, not the speaker as a whole (that's what `rename` is
  for, see "Operations"); if the given name matches one of the names in
  `speakers.yaml`, the `speaker_id` changes along with it.

```bash
uv run transcribe-meeting review output/<job-id>
# edit review/transcript_edit.csv by hand
uv run transcribe-meeting refine output/<job-id> output/<job-id>/review/transcript_edit.csv
```

`refine` applies the filled-in fields, re-renders every format, and updates
`result.json`/`summary.md`. It's the only way to fix an utterance's text or
a misattributed speaker — it never runs ASR again on its own.

## Operations

### Resuming, Status, and Re-rendering

```bash
uv run transcribe-meeting resume output/<job-id>
uv run transcribe-meeting status output/<job-id>
uv run transcribe-meeting clean output/<job-id> --mode raw,light
```

### Data Directories and Running Under an Agent

**Check readiness before you run.** `doctor` accepts the same
`--asr-models` and `--diarization-provider` as `run`, and prints JSON with a
`route_ready` key. When `route_ready: false`, the command exits with code 1
and lists the exact commands that fix it (which `setup` to run, which key
to set). This is the only check you need to run before a job: `run` itself
also rejects an unready route, but as an argument error at that point.

**How participants get named.** After a run, participants are named
`SPEAKER_00`, `SPEAKER_01`, and so on. In `result.json`, each one has a
`preview` — its first substantive phrase with a timecode — so you can
identify the person without listening to the recording. Names are applied
with `rename JOB_DIR --set SPEAKER_00=Name ...`, which re-renders every
file **without re-running recognition** (seconds) and updates
`result.json` and `summary.md`. If the roster is known up front, pass the
names straight to `run` via `--speaker-names`.


By default, `output/`, `models/`, `glossary.yaml`, and `.env` live in the
current working directory. They're overridden by the `TRANSCRIBE_HOME`
environment variable (the root for everything else, unless overridden
individually), `TRANSCRIBE_OUTPUT_DIR`, `TRANSCRIBE_MODELS_DIR`,
`TRANSCRIBE_ENV_FILE` — or by the equivalent global options `--data-root`,
`--output-dir`, `--models-dir` (available on any command, placed before the
command name). `models.lock.json` lives in `models/` and travels with that
directory.

When `run` finishes (and again after `clean`/`rename`/`refine`), the job's
root gets `result.json` (a machine-readable summary: status, duration,
speakers with previews of their first phrases, transcript files, warnings —
all paths inside are relative to the job directory) and `summary.md` (a
short, human-readable recap that doesn't duplicate the transcript itself) —
enough for an agent to understand how the run went without parsing the
manifest. Speaker names without an interactive terminal are set with `run
--speaker-names SPEAKER_00=Alice,SPEAKER_01=Bob` or `rename
output/<job-id> --set SPEAKER_00=Alice`.

## All Parameters

The full option reference for every command. Default values come straight
from `--help`; the detailed explanations are in the sections above — here
you get a one-line purpose and a cross-reference. Typer's standard flags
(`--help`, `--install-completion`, `--show-completion`) aren't listed.

### Global (before the command name)

| Option | Default | Purpose |
|---|---|---|
| `--data-root` | current working directory | Project data root (`output/`, `models/`, `glossary.yaml`, `.env`); overrides `TRANSCRIBE_HOME`. See "Data Directories and Running Under an Agent." |
| `--output-dir` | `<data-root>/output` | Job directory; overrides `TRANSCRIBE_OUTPUT_DIR`. |
| `--models-dir` | `<data-root>/models` | Models directory; overrides `TRANSCRIBE_MODELS_DIR`. |

### `run`

Argument: `parts` (one or more meeting files, required).

| Option | Default | Purpose |
|---|---|---|
| `--meeting-name` | required | Meeting name — used in the `output/` folder's name. |
| `--language` | `ru` | Speech language (ISO-639-1) or `auto` for autodetection. |
| `--audio-stream` | none (first stream) | FFmpeg audio stream index, if the file has multiple tracks. |
| `--speakers` | `auto` | Exact speaker count, or `auto` to detect within the `--min-speakers`/`--max-speakers` bounds. |
| `--min-speakers` | `1` | Lower bound when `--speakers auto`. |
| `--max-speakers` | `8` | Upper bound when `--speakers auto`; see the `foxnose` warning in "Diarization: Two Routes." |
| `--clean` | `raw,light,readable` | Which outputs to build, comma-separated. See "readable by Default." |
| `--profile` | `balanced` | Resource profile: `background` (2 threads, one FFmpeg process), `balanced`, or `max` (10 threads). Affects decoding threads and parallelism, not pauses. |
| `--asr-provider` | not set | Deprecated local/groq switch kept for compatibility; usually unnecessary — the provider is read from the model in `--asr-models`. |
| `--asr-device` | `auto` | GigaAM device: `auto`, `mps`, or `cpu`. Groq ignores it. See "Local ASR." |
| `--asr-models` | `multilingual_large_ctc,v3_rnnt` | 1–3 comma-separated ASR models. See "Multiple ASR Models and Fusion." |
| `--diarization-provider` | `foxnose` | Diarization route: `foxnose`, `pyannote`, or `deepgram`. See "Diarization: Two Routes" and "Deepgram (explicit opt-in)." |
| `--cpu-threads` | from the profile | Number of CPU threads for diarization. |
| `--ffmpeg-threads` | from the profile | Threads per FFmpeg process; further capped by available CPU. |
| `--chunk-minutes` | none (Groq: no more than 10) | ASR chunk length for Groq; an error for local (local uses VAD chunks up to 23.5 s). |
| `--interactive-names` / `--no-interactive-names` | on only in a TTY | Ask for speaker names interactively in the terminal. |
| `--speaker-names` | not set | Speaker names without interactivity, format `SPEAKER_00=Alice,SPEAKER_01=Bob` — same as `rename --set`. |
| `--resume` / `--no-resume` | `resume` | Rarely needed for `run` — `run` always creates a new job; use the `resume` command to continue one. |
| `--keep-working-audio` / `--no-keep-working-audio` | `no-keep-working-audio` | Keep `work/` and the WAV chunks `asr/*/chunks/**/*.wav` after a successful run instead of deleting them. The JSON checkpoints stay either way — `resume` needs them. |
| `--asr-preprocess` | `none` | Preprocessing for ASR and diarization: `none` or `hpf80-preemphasis`. See "Audio Preprocessing: Measured, Doesn't Help" under "Measured Facts." |
| `--asr-compression` | `none` | Compression for ASR and diarization: `none` or `4:1`. See "Audio Preprocessing: Measured, Doesn't Help" under "Measured Facts." |
| `--context-prompt` | `""` (up to 2,000 characters) | Meeting topic and purpose — a hint for readable. See "readable by Default." |
| `--llm-model` | `gpt-5.6-luna` | Model ID for `clean --mode readable`. |
| `--measure-loudness` / `--no-measure-loudness` | `no-measure-loudness` | Measure LUFS/true peak; lengthens the run, doesn't affect recognition. See "Diagnostics and Loudness." |

### `setup`

| Option | Default | Purpose |
|---|---|---|
| `--diarization-provider` | `all` | What to download for diarization: `pyannote`, `foxnose`, or `all` (both). |
| `--asr-models` | `multilingual_large_ctc,v3_rnnt` | Which ASR models to prepare (1–3), comma-separated; local ones download into `models/gigaam`. |

### `clean`

Argument: `job_dir` (folder of a finished job, required).

| Option | Default | Purpose |
|---|---|---|
| `--mode` | required | What to re-render: `raw`, `light`, `readable`, or several comma-separated. |
| `--llm-model` | not set (read from the manifest) | Override the LLM only for this `readable` run. |

### `rename`

Argument: `job_dir` (folder of a finished job, required).

| Option | Default | Purpose |
|---|---|---|
| `--set` | not set | `SPEAKER_00=Name`; can be repeated or comma-separated in one flag. The only way to rename without a TTY. |

### `review`

Argument: `job_dir` (job folder, required). No options besides the standard
ones — dumps every utterance to `review/transcript_edit.csv`. See "review
and refine: Manual Editing" under "Quality Checks."

### `refine`

Arguments: `job_dir` (job folder, required), `review_csv` (the edited CSV
from `review`, required). No options besides the standard ones.

### `status`

Argument: `job_dir` (job folder, required). No options besides the standard
ones. Shows the manifest and the completed atomic stages.

### `resume`

Argument: `job_dir` (folder of an unfinished job, required).

| Option | Default | Purpose |
|---|---|---|
| `--cpu-threads` | from the manifest | Temporary CPU thread count for this resume; the manifest is not changed. |
| `--ffmpeg-threads` | from the manifest | Temporary thread count per FFmpeg process; the manifest is not changed. |

### `glossary`

- `glossary list` — no options besides the standard ones.
- `glossary add <canonical>` — argument `canonical` (the term's canonical
  spelling, required).

  | Option | Default | Purpose |
  |---|---|---|
  | `--variant` | not set | Alternate spelling; the flag can be repeated. |
  | `--note` | not set | Optional note about the term. |
- `glossary import <path>` — argument `path` (a YAML file with a `terms`
  key, required); no options besides the standard ones.

### `auth`

- `auth set <provider>` — argument `provider` (`openai`, `groq`,
  `deepgram`, or `huggingface`, required). Stores the API key or `HF_TOKEN`
  in the project's `.env` (no-echo prompt).
- `auth set-openai-base-url` — stores the OpenAI-compatible API's URL in
  the project's `.env`; no options besides the standard ones.
- `auth status` — shows which keys are present, without revealing their
  values; no options besides the standard ones.

### `doctor`

| Option | Default | Purpose |
|---|---|---|
| `--asr-models` | `multilingual_large_ctc,v3_rnnt` | Which ASR models to check (same format as `run`/`setup`). |
| `--diarization-provider` | `foxnose` | Which diarization route to check (same as `run`). |

Checks dependencies, architecture, and the readiness of the chosen route,
without touching the network.

## Measured Facts

The numbers in this section can be re-checked on your own jobs — the tools
live in the repository, not just with the author:

```bash
uv run python scripts/model_agreement.py output/<job-id>     # model agreement within a job
uv run python scripts/compare_routes.py output/<job-a> output/<job-b>   # routes against each other
```


### Deepgram vs. Local GigaAM

Measured on the same two-hour recording (113.8 min, three participants):

| | GigaAM ×2 + foxnose | Deepgram |
|---|---|---|
| Run time | about 25 min | **about 6 min** (of which ASR is 40 s) |
| API requests | 0 | 1 |
| Speech rate | 157 words/min | 145 words/min |
| Speakers found | 3 (count given) | 3 (**count not given**) |

**Speaker labeling matched at 93.5%**, second by second across 6495 seconds,
under the best label matching. Two systems with completely different
architectures — WeSpeaker with embeddings versus a proprietary diarizer —
independently arrived at the same split; speaking-time distribution
differed by 2–4 minutes out of 15–57.

Text quality differs in kind, not in level. GigaAM is more accurate on
local terms and meaning: `ццт` versus the garbled `в с т`, "in different
planes" versus the nonsensical "in different bones." Deepgram gives you
punctuation, casing, digits instead of spelled-out numbers, and per-word
confidence right away — the only connected backend that returns it.


### Do the Routes Agree With Each Other

The same meeting (113.7 min, two files) was run through three routes and
compared pairwise: text agreement word by word, speaker-labeling agreement
second by second under the best label matching.

| Route pair | Text | Speaker labeling |
|---|---|---|
| 3 local+Groq / `foxnose` ↔ `v3_rnnt` / `pyannote` | 90.73% | 94.83% |
| 3 local+Groq / `foxnose` ↔ Deepgram / Deepgram | 78.58% | 92.13% |
| Deepgram / Deepgram ↔ `v3_rnnt` / `pyannote` | 81.48% | 92.00% |

Duration is identical across all three; speech rate is 156, 145, and 155
words/min (lower for Deepgram, as in the measurement above). Time
distribution across the three participants matches within 2–3 minutes out
of 15–57. In other words, the routes disagree on details of the text, not
on who spoke when.

The lower text agreement with Deepgram isn't a sign of error: it returns
punctuation, casing, and digits instead of spelled-out numbers, so a
word-by-word comparison against a local model is understated by
construction.

### Audio Preprocessing: Measured, Doesn't Help

`--asr-preprocess hpf80-preemphasis` (an 80 Hz highpass filter plus
pre-emphasis) and `--asr-compression 4:1` (dynamic-range compression) are
**off by default, and there's no reason to turn them on without one**.

Measured on a 54-minute dictaphone recording with three participants, four
runs, metric: agreement between two independent ASR models (there's no
reference text, and GigaAM doesn't return word probabilities, so confidence
isn't available as a metric):

| preprocess / compression | model agreement | words | utterances |
|---|---|---|---|
| **none / none** | **92.62%** | 8837 | 304 |
| none / 4:1 | 92.24% | 8852 | 305 |
| hpf80-preemphasis / none | 92.17% | 8800 | 314 |
| hpf80-preemphasis / 4:1 | 92.01% | 8800 | 309 |

The baseline wins, and the trend is monotonic: any processing lowers
agreement. The pipeline is deterministic — two identical runs produce
byte-identical text — so the 0.61 pp difference is reproducible, not
noise. But it comes from **one** recording, so this isn't proof that
preprocessing is harmful in general — it's a fact about this kind of
material.

The explanation is simple: both treatments were designed for problem
audio — a distant mic, hum, wide loudness swings. On a clean recording
there's nothing to clean up, and the filters change the spectrum and
dynamics regardless, pulling the signal further from what the model was
trained on. Preprocessing applies to the working PCM before everything
else, so it also touches diarization: the utterance count drifts 304→314,
and boundaries shift.

The options stay: they're worth using on genuinely bad audio. Measure the
same way on your own material before turning them on.

### Context at Chunk Edges: The Mechanism Exists, the Gain Is Unmeasured

On the two-hour recording, 42% of VAD chunks ended exactly at the 22 s
target — that is, mid-speech. A 0.5 s context margin on each side
eliminates that: the model hears the word at the boundary in full, and the
extra words from the margin go to whichever neighboring chunk owns the
word's center.

Measured on the same meeting: agreement between the two models was 89.33%
(without the margin) against 89.42% (with it) — a difference within noise.
What's more, this metric is blind to this particular defect: both models
get cut at the same seams, so they make the same mistakes there. A direct
comparison of the canonical text before and after gives 93.01% agreement,
and changes cluster near the old seams only 1.16 times as densely — the
rest is explained by the ceiling drop from 24 to 23.5 s, which shifted
every boundary.

Conclusion: the change removes a real word-corruption mechanism and
doesn't make anything worse, but there's no way to prove a gain — that
needs a reference transcript, which the project doesn't have. If one ever
shows up, this is the first thing worth re-measuring.

### Speed

Measured on a 35-minute Russian recording, M1 Pro, MPS: ASR
`multilingual_large_ctc` takes 161 s, `v3_rnnt` takes 242 s, diarization
takes 73 s. That's **roughly a third of the recording's length for two
models**, about nine times faster than real time for the whole run.
Loading each model takes a few seconds and is included in these numbers.

Measured on a 113.7 min meeting (two files, three participants, M1 Pro,
MPS), from the stage markers in `.stages/`:

| Route | ASR | Diarization (incl. overlap detection) | Total through `align` |
|---|---|---|---|
| 3 models (`multilingual_large_ctc`, `v3_rnnt`, Groq) + `foxnose` | 17.6 min | 9.6 min | **27.3 min** |
| `deepgram-nova-3` + `deepgram` diarization | 0.9 min | 5.2 min | **6.2 min** |
| `v3_rnnt` + `pyannote` | 12.3 min | 95.2 min | **107.6 min** |

The separate overlap-detection pass (`segmentation-3.0` over the whole
track) takes about 5 minutes and is included in the diarization column for
all three routes — on the Deepgram route it accounts for almost all of
that column, because diarization itself arrives there in the same request
as recognition.

### What's Been Verified End-to-End

Not everything covered by unit tests has been run start to finish.
Verified on real audio:

| Route / mode | Status |
|---|---|
| 1–3 ASR models, `foxnose` diarization | run many times, including a two-hour meeting split into two parts |
| `pyannote` diarization | run verified; `regular` ≠ `exclusive`, overlaps get cut out — the key difference from `foxnose` |
| Groq only (`--asr-models groq-whisper-large-v3`) | run verified, word timestamps present, artifacts in `asr/groq-whisper-large-v3/` |
| `background` and `max` profiles | runs verified, thread counts match the profile |
| `resume` after an interruption and on a finished job | run verified, checkpoints get reused |
| Running on Linux | install, import, and the full pytest suite are checked in CI on `ubuntu-latest`; **actually running the models on Linux has not been verified** |
| Mixed route: 2 local models + Groq, `foxnose`, `readable` | run verified on a 113.7 min meeting: 754 utterances, `readable` completed on all six batches, zero rollbacks to local text |
| Deepgram: ASR and diarization in one request | run verified; speaker count can't be set — 3 requested, 4 found, a warning rather than an error |
| `pyannote`: `regular` ≠ `exclusive` | run verified: 1175 segments versus 1039, 4.8 min of overlap cut out |

The pipeline is deterministic: two identical runs produce byte-identical
text. That means comparing setting variants only takes one run per variant.

## Known Limitations

- **Deepgram: request body capped at ~2 GiB**, not by audio duration —
  measured on a real 218 MB / 113.8-minute request, which took 79 s, well
  inside the limit (see `DEEPGRAM_MAX_UPLOAD_BYTES` in `asr.py`). A
  two-hour meeting as 16 kHz mono WAV fits inside this limit with plenty of
  room to spare; chunking for Deepgram isn't implemented, because this
  problem never comes up in practice.
- **Deepgram: a documented 10-minute request timeout** (504 Gateway
  Timeout) — it's a processing-time limit, not an audio-duration one.
  Measured: 113.8 minutes processed in 79 seconds, roughly a fifteen-fold
  margin. Deepgram's async mode (`callback`) doesn't work around this
  limit: it needs a publicly reachable URL to receive the result, and the
  local CLI doesn't have one.
- **You cannot set the number of participants with Deepgram** — neither
  for ASR (`--asr-models deepgram-nova-3`) nor for
  `--diarization-provider deepgram`. An explicit `--speakers` given
  alongside it doesn't fail the run: a mismatch becomes a warning in
  `result.json` rather than an error (unlike `pyannote`/`foxnose`, where
  it is one).
- **The Groq free tier caps a single ASR chunk at 10 minutes**
  (`--chunk-minutes` never exceeds this automatically) and a daily audio
  limit of 28800 s (8 h) — see "Groq (explicit opt-in)" above.
- **There's no `--review-uncertain` in the CLI**: this option would only
  be useful where ASR returns a word probability, and of the connected
  backends only Deepgram does that (see `probability` in
  `words_from_asr`) — GigaAM and Groq always return `None`. Unreliable
  low-confidence flagging isn't possible for most routes, and a flag that
  only worked on one backend out of three wasn't worth building.
- **Audio preprocessing has been measured and doesn't help** on the
  material it was tested on (see "Audio Preprocessing: Measured, Doesn't
  Help" above) — the options stay, for genuinely bad audio, but are off by
  default.
- **`gigaam` is a git dependency pinned to a specific commit** of a
  specific repository (`salute-developers/GigaAM`, see `pyproject.toml`),
  not a PyPI package. If that repository moves, gets deleted, or rewrites
  its history, `uv sync` on a clean machine will break until the
  dependency is repointed at a new address — `uv.lock` pins the revision,
  but that doesn't save you if the repository itself disappears.

## License

This repository's code is **MIT**, full text in [`LICENSE`](LICENSE).
Direct dependencies are under MIT (typer, rich, pydantic, PyYAML, openai,
`gigaam`) and Apache-2.0 (`diarize`); both are compatible.

The code license doesn't cover the models: none of them ship with this
repository — the user downloads all of them with the `setup` command on
first run — and each carries its own license:

| Model | License | Terms |
|---|---|---|
| GigaAM (`salute-developers/GigaAM`) — the `gigaam` package itself | MIT | no restrictions |
| VAD `pyannote/segmentation-3.0` | see the model card on Hugging Face | gated: must accept terms on Hugging Face before `setup` |
| `pyannote/speaker-diarization-community-1` (`--diarization-provider pyannote` route) | see the model card on Hugging Face | gated: must accept terms on Hugging Face before `setup --diarization-provider pyannote` |
| WeSpeaker `voxceleb_resnet34_LM` (`--diarization-provider foxnose` route, default) | CC-BY-4.0 (via the VoxCeleb dataset it was trained on) | no terms to accept up front; attribution required when distributing results trained on this data |
| `diarize` (the package wrapping WeSpeaker) | Apache-2.0 | no restrictions |

Groq (`groq-whisper-large-v3`) and Deepgram (`deepgram-nova-3` and
`--diarization-provider deepgram`) aren't downloadable models but external
APIs; using them is governed by the respective services' terms, not by a
model file's license.

The license choice is the project owner's decision; swapping it for
another compatible license is a one-file edit (`LICENSE`) plus one line
(`license = "..."`) in `pyproject.toml`.
