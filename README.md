# transcribe-meeting

Transcribe Russian meetings with timestamps and speaker labels. By default, GigaAM runs locally (MPS when available, otherwise CPU) and FoxNose labels speakers. A job can resume from checkpoints after an interruption.

**Privacy:** the default `raw,light,readable` outputs include an LLM pass. When enabled and configured, `readable` sends canonical and variant text, utterance and speaker IDs, and any meeting context to the configured endpoint, not audio. Use `--clean raw,light` with the default ASR and diarization route to keep the recording and transcript local after setup. Groq and Deepgram are opt-in: both send audio, and Groq can also send a prompt built from the glossary and meeting context, even with `--clean raw`.

## Install

Install from a [repository checkout](https://github.com/doctorsolberg/transcribe-meeting) using `uv`, Git, FFmpeg and FFprobe on `PATH`. This is the supported install path: the checkout's `uv` configuration overrides conflicting Torch/torchaudio constraints, and GigaAM is pinned to a Git revision. A wheel build does not verify a standalone `pip` install.

Use native Apple Silicon Python on macOS 14 or newer with the current lockfile; Intel macOS is not supported by this lock. CI checks the checkout on macOS and Ubuntu with Python 3.12 and 3.13, but does not download weights or verify inference. Start with Python 3.12 for real runs.

Budget at least 7 GiB of free space on the model volume for initial setup. `doctor` reports this budget but does not require it for an installed route. This does not include `.venv`, the `uv` cache, or job output. The Linux lock can download several additional gigabytes of Torch/CUDA wheels even when local ASR runs on CPU. A run also needs space on the output volume for decoded and working audio; the estimate depends on the recording and route.

```bash
# macOS: brew install ffmpeg
# Debian/Ubuntu: sudo apt install ffmpeg
uv sync --locked --python 3.12 --no-dev
```

1. Sign in to Hugging Face and accept the terms or request access for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). Create a Read token on the same account; if you use a fine-grained token, grant it read access to the required gated repositories. A token does not grant access your account has not already been given.
2. Export `HF_TOKEN` or store it in the project's `.env` with `auth set huggingface` (hidden prompt). Skip the prompt if the token is already configured. Hub CLI login alone is not a substitute for supplying `HF_TOKEN` to this app.
3. Install the default models and check the route:

```bash
uv run --no-sync transcribe-meeting auth set huggingface  # only if HF_TOKEN is not configured
uv run --no-sync transcribe-meeting setup
uv run --no-sync transcribe-meeting doctor
```

`setup` defaults to FoxNose, like `run`. For pyannote diarization, also accept the terms of [Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), grant the token access if needed, and use `setup --diarization-provider pyannote` (or `all` for both local diarizers). `HF_TOKEN` is only required for missing gated downloads, not to verify a complete local setup or run local inference. Network-only ASR does not require segmentation-3.0; add `setup --with-vad` if you want local overlap detection on those routes.

If the GigaAM CDN is unavailable, point `TRANSCRIBE_GIGAAM_BASE_URL` at a trusted HTTPS mirror, or place the original `multilingual_large_ctc.ckpt` and `v3_rnnt.ckpt` in `<models-dir>/gigaam/` before running `setup`. The CLI checks both against the upstream MD5 values and records SHA-256 for provenance. Do not disable TLS verification.

To move an existing model installation, copy the entire `models/` directory, including `models.lock.json` and the Hugging Face cache's `refs`, `snapshots`, and `blobs`. Preserve symlinks (`cp -a models /path/to/new-models` with a new destination) and point `--models-dir` at the copy. If a legacy lock is in the old data root, copy it into the new models directory; absolute entries in a legacy lock still point to their original locations and need accessible targets or locally reprovisioned models. Run `doctor` against the destination. It checks prerequisites, local file presence, and required ASR/diarization key presence, not inference or API access.

## Transcribe

```bash
uv run --no-sync transcribe-meeting run meeting.m4a --meeting-name example
```

To process sequential parts of one meeting, pass the files in recording order. Without `--audio-stream`, each part must have exactly one audio stream (a stereo stream still counts as one). For multiple tracks, run `ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 INPUT` and pass the chosen stream index, not its ordinal among audio tracks. The same index must be an audio stream in every part.

Use `--speakers 3` if the count is known on a local diarization route; otherwise the number is estimated. The local ASR accepts `--language ru` or `auto` (still Russian). For one ASR pass instead of two, choose `--asr-models v3_rnnt`. When using multiple models, the first defines the utterance IDs, timecodes and speakers. Keep the default order to use the calibrated fusion instructions.

| Route | Where audio goes | When to use it |
|---|---|---|
| GigaAM + `foxnose` | Stays local | Default Russian route. |
| GigaAM + `pyannote` | Stays local | You need Community-1's exclusive speaker timeline. It is slower; see the [benchmarks](docs/benchmarks.md). |
| `--asr-models groq-whisper-large-v3` | Groq | Whisper languages; requires `GROQ_API_KEY`. Diarization stays local unless you also select Deepgram. |
| `--asr-models deepgram-nova-3 --diarization-provider deepgram` | Deepgram | ASR and speakers in one request; requires `DEEPGRAM_API_KEY`. Exact speaker count is not supported. |

`light` and `readable` are tuned for Russian. For other languages, start with `--clean raw`. Groq alone uses `--chunk-minutes`; Deepgram submits the meeting as a single request, subject to its upload limit.

## Readable output

Set the key with `auth set openai` or `OPENAI_API_KEY`. For another OpenAI-compatible endpoint, set `OPENAI_BASE_URL` or use `auth set-openai-base-url`. Remote endpoints require HTTPS, and readable requests do not follow redirects. HTTP is allowed for literal loopback addresses such as `http://127.0.0.1:8000/v1`, which can be used without a key. Verify the endpoint before sharing a key or transcript. Choose a model available on that endpoint with `--llm-model` if needed: an OpenAI-compatible API does not necessarily have the default model ID or support the default Responses/JSON schema options.

If the endpoint needs a key and none is set, the default run still writes `raw` and `light`; `readable` remains requested and records a separate failure. Once configured, `resume` can retry the requested cleanup. To explicitly add or repeat cleanup without re-running ASR, use:

```bash
uv run --no-sync transcribe-meeting clean output/JOB --mode readable
```

Set these overrides in the environment or the project's `.env`:

| Setting | When to change it |
|---|---|
| `TRANSCRIBE_READABLE_API` | `responses` (default) or `chat` for servers that expose Chat Completions only. |
| `TRANSCRIBE_READABLE_RESPONSE_FORMAT` | `json_schema` (default) or `json_object` if the server cannot enforce a schema. The CLI validates both locally. |
| `TRANSCRIBE_READABLE_BATCH_SIZE` | Reduce the default of 150 utterances for a smaller context window. |
| `TRANSCRIBE_READABLE_REASONING` | `none` omits the field. `low`, `medium`, `high` request a level only if the server supports it. Default: `medium` on Responses, omitted on Chat Completions. |

`readable` combines ASR variants, restores punctuation and can change words. Check names, numbers, and important quotes against the recording. If a batch fails, a first readable export can be `partial`: it includes local fallback text. A partial pass never replaces an already accepted complete version. Check `readable.status` in `result.json` and inspect `review/readable_changes.csv` when present. Use `clean --mode readable` to retry a partial pass; `resume` does not repeat it automatically. A silent job sends no text to the LLM.

## Files and corrections

Jobs are stored in `output/` by default:

| File | Contents |
|---|---|
| `transcript.raw.<model>.md` | One line per utterance from an ASR model. |
| `transcript.light.<model>.md` | Local cleanup, with adjacent lines from the same speaker grouped. It does not restore punctuation within a phrase. |
| `transcript.readable.md` | One combined LLM-edited transcript, when available. |
| `result.json`, `summary.md` | Status, transcript paths, detected speakers, warnings and independent `readable` status. |
| `canonical.json`, `manifest.json`, `.stages/` | Utterances, job configuration (including input paths and context) and resume state. Keep them if you plan to resume or edit. |

Treat the whole job directory as sensitive. Besides the Markdown exports, it can contain alternate ASR text, speaker previews and names, review edits, provider responses, and working audio.

```bash
uv run --no-sync transcribe-meeting status output/JOB
uv run --no-sync transcribe-meeting resume output/JOB
uv run --no-sync transcribe-meeting rename output/JOB --set SPEAKER_00=Alice --set SPEAKER_01=Bob
uv run --no-sync transcribe-meeting review output/JOB
# Edit replacement and/or speaker_replacement in the CSV, then apply it:
uv run --no-sync transcribe-meeting refine output/JOB output/JOB/review/transcript_edit.csv
```

`rename` and `refine` do not re-run recognition. Use the speaker previews in `result.json` to help identify speakers. Keep the original audio at its recorded path to finish an interrupted job if processing still needs it. Working WAVs are removed after a successful run by default. To process the audio again after publishing a job, start a new `run`.

Text corrected with `refine` is kept verbatim in subsequent `readable` passes.

## Data directories

`output/`, `models/`, `glossary.yaml` and `.env` are relative to the current directory unless `TRANSCRIBE_HOME` or `--data-root` is set. `TRANSCRIBE_OUTPUT_DIR`, `TRANSCRIBE_MODELS_DIR` and `TRANSCRIBE_ENV_FILE` override individual paths. Put global flags before the command. Use the same data root for `setup`, `run`, and `resume`.

`auth` writes an owner-only `.env`; the CLI reads only supported settings from it. Environment values take precedence. Configure a custom `OPENAI_BASE_URL` and its key together in the environment or in the same trusted `.env`; the CLI rejects an `.env` endpoint for another origin when only the key comes from the environment. If copying [`.env.example`](.env.example), restrict your copy to owner access (`chmod 600 .env`) before adding secrets.

When switching providers with `auth set-openai-base-url`, remove the old `OPENAI_API_KEY` from the project's `.env` first, then save the new endpoint and its key.

## CLI reference

Use `transcribe-meeting COMMAND --help` for current defaults and usage.

### Global (before the command name)

| Option | Default | Purpose |
|---|---|---|
| `--version` | off | Print the installed version and exit. |
| `--data-root` | current working directory | Project data root; overrides `TRANSCRIBE_HOME`. |
| `--output-dir` | `<data-root>/output` | Job directory; overrides `TRANSCRIBE_OUTPUT_DIR`. |
| `--models-dir` | `<data-root>/models` | Models directory; overrides `TRANSCRIBE_MODELS_DIR`. |

### `run`

Argument: `parts` (one or more meeting files, required).

| Option | Default | Purpose |
|---|---|---|
| `--meeting-name` | required | Meeting name — used in the `output/` folder's name. |
| `--language` | `ru` | Speech language (ISO-639-1) or `auto` for autodetection. |
| `--audio-stream` | unset (each file must have one audio stream) | FFprobe stream `index`, shared by every part. |
| `--speakers` | `auto` | Positive exact count for local diarization, or `auto` to estimate. With Deepgram, a supplied count does not constrain diarization; a mismatch is a warning. |
| `--min-speakers` | `1` | Lower bound for local diarization with `--speakers auto`; must be at least 1. |
| `--max-speakers` | `8` | Upper bound for local diarization with `--speakers auto`; must be at least `--min-speakers`. FoxNose can undercount large groups. |
| `--clean` | `raw,light,readable` | Which outputs to build, comma-separated. |
| `--profile` | `balanced` | Resource profile: `background` (2 threads, one FFmpeg process), `balanced`, or `max` (10 threads). Affects decoding threads and parallelism, not pauses. |
| `--asr-provider` | not set | Deprecated local/groq switch kept for compatibility; usually unnecessary — the provider is read from the model in `--asr-models`. |
| `--asr-device` | `auto` | GigaAM only: `auto` uses MPS when available, otherwise CPU. Explicit `mps` fails when unavailable; `cpu` forces CPU. |
| `--asr-models` | `multilingual_large_ctc,v3_rnnt` | 1–3 comma-separated ASR models. |
| `--diarization-provider` | `foxnose` | Diarization route: `foxnose`, `pyannote`, or `deepgram`. |
| `--cpu-threads` | from the profile | Number of CPU threads for diarization. |
| `--ffmpeg-threads` | from the profile | Threads per FFmpeg process; further capped by available CPU. |
| `--chunk-minutes` | 10 for Groq; unset otherwise | Groq chunk length in minutes (capped at 10); rejected if no Groq model is selected. Local ASR uses VAD chunks up to 23.5 s. |
| `--interactive-names` / `--no-interactive-names` | on only in a TTY | Ask for speaker names interactively in the terminal. |
| `--speaker-names` | not set | Speaker names without interactivity, format `SPEAKER_00=Alice,SPEAKER_01=Bob` — same as `rename --set`. |
| `--resume` / `--no-resume` | `resume` | Rarely needed for `run` — `run` always creates a new job; use the `resume` command to continue one. |
| `--keep-working-audio` / `--no-keep-working-audio` | `no-keep-working-audio` | Keep `work/` and the WAV chunks `asr/*/chunks/**/*.wav` after a successful run instead of deleting them. The JSON checkpoints stay either way — `resume` needs them. |
| `--asr-preprocess` | `none` | Preprocessing for ASR and diarization: `none` or `hpf80-preemphasis`. |
| `--asr-compression` | `none` | Compression for ASR and diarization: `none` or `4:1`. |
| `--context-prompt` | `""` (up to 2,000 characters) | Meeting topic and purpose; kept in the job configuration and used as a hint for `readable` and Groq, when enabled. |
| `--llm-model` | `gpt-5.6-luna` | LLM model ID for this job's `readable` pass. |
| `--measure-loudness` / `--no-measure-loudness` | `no-measure-loudness` | Measure LUFS/true peak; lengthens the run, doesn't affect recognition. |

### `setup`

| Option | Default | Purpose |
|---|---|---|
| `--diarization-provider` | `foxnose` | Download models for `foxnose`, `pyannote`, or `all` (both). |
| `--asr-models` | `multilingual_large_ctc,v3_rnnt` | Which ASR models to prepare (1–3), comma-separated; local ones download into `models/gigaam`. |
| `--with-vad` | off | Also provision segmentation-3.0 for local overlap detection with network-only ASR. Local ASR already provisions it. |

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
ones — dumps every utterance to `review/transcript_edit.csv`.

### `refine`

Arguments: `job_dir` (job folder, required), `review_csv` (the edited CSV
from `review`, required). No options besides the standard ones.

### `status`

Argument: `job_dir` (job folder, required). No options besides the standard
ones. Prints the job ID, completed stage names, and readable status as JSON.

### `resume`

Argument: `job_dir` (existing job folder, required).

| Option | Default | Purpose |
|---|---|---|
| `--cpu-threads` | from the manifest | Temporary CPU thread count for this resume; the manifest is not changed. |
| `--ffmpeg-threads` | from the manifest | Temporary thread count per FFmpeg process; the manifest is not changed. |

For a published job, `resume` verifies the accepted `canonical.json` against its recorded digest (when present) and repairs derived exports without re-reading the original audio. Restore a missing or changed canonical transcript from backup; do not edit it directly. `resume` can also retry requested readable cleanup if needed. Use `status` to check the separate readable status and `clean --mode readable` to explicitly repeat that pass. A new `run` is required to reprocess the recording.

### `glossary`

- `glossary list` — prints the current glossary as YAML; no options besides the standard ones.
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

Reports Python compatibility, architecture, FFmpeg/FFprobe availability,
free space on the model volume, and route readiness without touching the
network. It does not load the models or test provider credentials.

## More information

- [Historical benchmarks](docs/benchmarks.md): runtime and agreement between models on sample recordings.
- [Contributing](CONTRIBUTING.md) and [changelog](CHANGELOG.md).

The code is [MIT licensed](LICENSE). Model weights are not bundled and have their own terms; see the upstream [GigaAM](https://github.com/salute-developers/GigaAM), [WeSpeaker](https://github.com/wenet-e2e/wespeaker), and Hugging Face model pages. External APIs have separate service terms.
