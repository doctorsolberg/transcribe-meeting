from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
import traceback
from getpass import getpass
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer
from rich import print
from rich.table import Table

from . import __version__, paths, provisioning
from .asr import (
    ASR_API_KEY_ENV,
    ASR_DEVICES,
    ASR_MODELS,
    ASR_PROVIDERS,
    DEEPGRAM_ASR_MODEL_NAME,
    DEFAULT_ASR_MODELS,
    GROQ_ASR_MODEL_NAME,
    GROQ_FREE_AUDIO_SECONDS_PER_DAY,
    GROQ_FREE_MAX_CHUNK_SECONDS,
    VAD_REPOSITORY,
    degrade_deepgram_without_key,
    degrade_groq_without_key,
    effective_asr_device,
    validate_asr_models,
)
from .audio import (
    ASR_COMPRESSION_MODES,
    ASR_PREPROCESS_MODES,
    AudioError,
    check_ffmpeg,
    diagnostics,
)
from .core import (
    CONSOLE,
    PROFILES,
    create_job,
    effective_threads,
    estimate_working_bytes,
    glossary,
    prepare_review,
    route_readiness,
    run_job,
    run_readable,
    try_readable,
    write_glossary,
)
from .core import refine as refine_review
from .core import rename as rename_speakers
from .diarization import DEEPGRAM_REPO, DIARIZATION_PROVIDERS, FOXNOSE_REPO, PYANNOTE_REPO
from .result import write_job_result
from .secrets import PROJECT_ENV_NAMES, load_project_secrets, save_project_secret
from .util import atomic_json, read_json

#: Interpreters the project is tested against: both resolve the lock and pass the
#: full suite. Model inference itself has only been exercised on 3.12 (see README).
SUPPORTED_PYTHON = ((3, 12), (3, 13))

app = typer.Typer(
    no_args_is_help=True,
    help="Meeting transcription: local by default, Groq only with an explicit option.",
)
glossary_app = typer.Typer(no_args_is_help=True)
app.add_typer(glossary_app, name="glossary")
auth_app = typer.Typer(no_args_is_help=True)
app.add_typer(auth_app, name="auth")


def _print_version(value: bool) -> None:
    # is_eager on the option means this fires before any other parameter (including
    # a missing subcommand) is even parsed, which is what makes `--version` alone
    # work despite `no_args_is_help=True` on the app.
    if value:
        typer.echo(f"transcribe-meeting {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
    data_root: Annotated[
        Path | None,
        typer.Option(
            "--data-root",
            help="Project data root: output/, models/, glossary.yaml, and .env live here "
            "by default. Defaults to the current working directory; overrides the "
            f"{paths.ENV_DATA_ROOT} environment variable.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help=f"Job directory; overrides the {paths.ENV_OUTPUT_DIR} environment variable.",
        ),
    ] = None,
    models_dir: Annotated[
        Path | None,
        typer.Option(
            "--models-dir",
            help=f"Models directory; overrides the {paths.ENV_MODELS_DIR} environment variable.",
        ),
    ] = None,
) -> None:
    # Set the env vars first, so every path helper (data_root/output_dir/models_dir/
    # secrets_path in paths.py) picks them up the moment it is next called — this is
    # the one place a CLI flag ever needs to become an env var.
    if data_root is not None:
        os.environ[paths.ENV_DATA_ROOT] = str(data_root)
    if output_dir is not None:
        os.environ[paths.ENV_OUTPUT_DIR] = str(output_dir)
    if models_dir is not None:
        os.environ[paths.ENV_MODELS_DIR] = str(models_dir)
    load_project_secrets(paths.secrets_path())


def _parse_clean(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(item not in {"raw", "light", "readable"} for item in values):
        raise typer.BadParameter("Use one or more modes: raw,light,readable")
    return values


def _parse_asr_models(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    try:
        validate_asr_models(names)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return names


def _asr_models_for_provider(
    asr_provider: str, models: list[str], *, provider_explicit: bool, models_explicit: bool
) -> list[str]:
    """Reconcile the legacy --asr-provider switch with --asr-models.

    Provider is normally read per model from ASR_MODELS; this function only exists so a
    script that still passes --asr-provider explicitly gets a clear BadParameter
    instead of a silently wrong run when it disagrees with an explicit --asr-models.
    When --asr-provider is not explicit (the common path, including the new
    mixed-provider usage), --asr-models alone decides and may freely mix providers.
    """
    if not provider_explicit:
        return models
    if asr_provider == "groq":
        if not models_explicit:
            return [GROQ_ASR_MODEL_NAME]
        if models != [GROQ_ASR_MODEL_NAME]:
            raise typer.BadParameter(
                f"--asr-provider groq implies the single model {GROQ_ASR_MODEL_NAME}; "
                f"either omit --asr-models or set --asr-models {GROQ_ASR_MODEL_NAME}."
            )
        return models
    # asr_provider == "local"
    if models_explicit:
        mismatched = [name for name in models if ASR_MODELS[name]["provider"] != "local"]
        if mismatched:
            raise typer.BadParameter(
                "--asr-provider local is not compatible with network models in --asr-models "
                f"(found: {', '.join(mismatched)}); either remove them from --asr-models "
                "or omit --asr-provider."
            )
    return models


def _validate_asr_options(
    *,
    asr_provider: str | None,
    asr_device: str,
    asr_models: str,
    diarization_provider: str,
    language: str,
    chunk_minutes: int | None,
    cpu_threads: int | None,
    ffmpeg_threads: int | None,
    asr_preprocess: str,
    asr_compression: str,
    context_prompt: str,
) -> tuple[list[str], list[str], bool, bool]:
    """Validate every `run` option about ASR/diarization provider and resource limits.

    Returns (parsed_asr_models, dropped_network_models, has_local_asr_model,
    has_groq_asr_model) — everything `run` needs afterwards to build its config.
    """
    provider_explicit = asr_provider is not None
    resolved_provider_flag = asr_provider or "local"
    if resolved_provider_flag not in ASR_PROVIDERS:
        raise typer.BadParameter("--asr-provider: local or groq")
    if asr_device not in ASR_DEVICES:
        raise typer.BadParameter("--asr-device: auto, mps, or cpu")
    if diarization_provider not in DIARIZATION_PROVIDERS:
        raise typer.BadParameter("--diarization-provider: pyannote, foxnose, or deepgram")
    if diarization_provider == "deepgram" and not os.environ.get("DEEPGRAM_API_KEY"):
        # Unlike a dropped ASR model, silently falling back to another diarization
        # route would change what the result actually means, so
        # this is a hard failure, not a degrade-with-a-warning.
        raise typer.BadParameter(
            "--diarization-provider deepgram requires DEEPGRAM_API_KEY. Run: "
            "transcribe-meeting auth set deepgram (or set the key in the environment; "
            "never pass it in arguments or chat) — or use "
            "--diarization-provider foxnose/pyannote."
        )
    models_explicit = asr_models != ",".join(DEFAULT_ASR_MODELS)
    parsed_asr_models = _asr_models_for_provider(
        resolved_provider_flag,
        _parse_asr_models(asr_models),
        provider_explicit=provider_explicit,
        models_explicit=models_explicit,
    )
    # Degrade like readable does: a missing key drops the network model with a
    # warning instead of failing a run that has perfectly good local models left.
    # Checked before create_job so the manifest reflects what will actually run.
    # Same treatment for both network ASR providers — each dropped independently so
    # one missing key never masks the other.
    parsed_asr_models, dropped_groq_models = degrade_groq_without_key(
        parsed_asr_models, has_key=bool(os.environ.get("GROQ_API_KEY"))
    )
    parsed_asr_models, dropped_deepgram_models = degrade_deepgram_without_key(
        parsed_asr_models, has_key=bool(os.environ.get("DEEPGRAM_API_KEY"))
    )
    dropped_network_models = dropped_groq_models + dropped_deepgram_models
    for dropped in dropped_network_models:
        provider = ASR_MODELS[dropped]["provider"]
        env_name = ASR_API_KEY_ENV[provider]
        print(
            f"[yellow]ASR: {dropped} skipped[/] — {env_name} is missing. Run "
            f"`transcribe-meeting auth set {provider}` to enable it. "
            "Continuing with the remaining models."
        )
    if not parsed_asr_models:
        commands = ", ".join(
            sorted(
                f"transcribe-meeting auth set {ASR_MODELS[name]['provider']}"
                for name in dropped_network_models
            )
        )
        raise typer.BadParameter(
            "No ASR models are left: the only one was "
            f"{', '.join(dropped_network_models)}, and the required key is not set. Run: "
            f"{commands} — or use only local models in --asr-models."
        )
    has_local_asr_model = any(ASR_MODELS[name]["provider"] == "local" for name in parsed_asr_models)
    has_groq_asr_model = any(ASR_MODELS[name]["provider"] == "groq" for name in parsed_asr_models)
    if has_local_asr_model and language not in {"ru", "auto"}:
        raise typer.BadParameter("GigaAM in this configuration only accepts --language ru or auto.")
    if not has_groq_asr_model and chunk_minutes is not None:
        raise typer.BadParameter(
            "--chunk-minutes only applies when a groq model is among --asr-models; "
            "local uses VAD chunks up to 23.5 s."
        )
    if has_local_asr_model and asr_device == "mps":
        # Explicit request must fail in preflight instead of silently switching to CPU.
        try:
            effective_asr_device("mps")
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
    if ffmpeg_threads is not None and ffmpeg_threads < 1:
        raise typer.BadParameter("--ffmpeg-threads must be at least 1")
    if cpu_threads is not None and cpu_threads < 1:
        raise typer.BadParameter("--cpu-threads must be at least 1")
    if chunk_minutes is not None and chunk_minutes < 0:
        raise typer.BadParameter("--chunk-minutes cannot be negative")
    if asr_preprocess not in ASR_PREPROCESS_MODES:
        raise typer.BadParameter("--asr-preprocess: none or hpf80-preemphasis")
    if asr_compression not in ASR_COMPRESSION_MODES:
        raise typer.BadParameter("--asr-compression: none or 4:1")
    if len(context_prompt) > 2000:
        raise typer.BadParameter("--context-prompt must not exceed 2,000 characters")
    return parsed_asr_models, dropped_network_models, has_local_asr_model, has_groq_asr_model


def _check_route_ready(asr_models: list[str], diarization_provider: str) -> None:
    """Reject a run whose route is not ready, before create_job.

    Without this, a missing local checkpoint or diarization model surfaces only
    deep inside run_job — after audio has already been decoded and a job directory
    created — with a generic "run setup" that may not even download the
    missing piece (setup's own --asr-models/--diarization-provider default only
    covers part of the registry). This raises the same BadParameter deepgram
    diarization already got, but for every route, naming the exact command.
    """
    status = route_readiness(asr_models, diarization_provider)
    if not status["route_ready"]:
        raise typer.BadParameter(
            "Route not ready; the run was cancelled before creating the job or "
            "decoding audio:\n" + "\n".join(status["problems"])
        )


def _validate_clean_options(clean: str) -> list[str]:
    modes = _parse_clean(clean)
    if "readable" in modes and not os.environ.get("OPENAI_API_KEY"):
        # A local transcript must never depend on an external model being configured.
        print(
            "[yellow]readable skipped:[/] OPENAI_API_KEY is missing. "
            "raw and light will be created; readable can be added later with the clean command."
        )
        modes = [item for item in modes if item != "readable"] or ["raw", "light"]
    return modes


def _validate_speakers(speakers: str, min_speakers: int, max_speakers: int) -> int | None:
    speaker_count = _speakers(speakers)
    if min_speakers < 1 or max_speakers < min_speakers:
        raise typer.BadParameter("Invalid participant count bounds.")
    return speaker_count


def _speakers(value: str) -> int | None:
    if value == "auto":
        return None
    try:
        number = int(value)
    except ValueError as exc:
        raise typer.BadParameter("--speakers: auto or an integer") from exc
    if number < 1:
        raise typer.BadParameter("The participant count must be at least 1")
    return number


def _parse_speaker_names(value: str) -> dict[str, str]:
    """Parse "SPEAKER_00=Alice,SPEAKER_01=Bob": shared by
    `run --speaker-names` and `rename --set`, one `--set` flag at a time."""
    pairs = [item.strip() for item in value.split(",") if item.strip()]
    replacements: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"Expected format SPEAKER_00=Name, got: {pair!r}")
        identifier, _, name = pair.partition("=")
        identifier, name = identifier.strip(), name.strip()
        if not identifier or not name:
            raise typer.BadParameter(f"Empty speaker_id or name in {pair!r}")
        replacements[identifier] = name
    return replacements


def _require_job_dir(job_dir: Path) -> None:
    """Reject a job folder that does not exist with a clear message.

    Without this, `status`/`refine` fall through to `read_json`/`open` and raise a
    raw FileNotFoundError — a stack trace instead of the same kind of "here is what
    to fix" message every other CLI-layer validation in this module gives.
    """
    if not job_dir.is_dir():
        raise typer.BadParameter(f"Job folder not found: {job_dir}")


def _existing_speaker_names(job_dir: Path) -> dict[str, str]:
    path = job_dir / "speakers.yaml"
    if not path.exists():
        return {}
    import yaml

    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("speakers", {})


def _resolve_speaker_name_overrides(speaker_names: str, found: dict[str, str]) -> dict[str, str]:
    """Validate `run --speaker-names` against speakers.yaml's actual set (plan
    section 6): more names than found speakers, or a name for an unknown
    speaker_id, is a usage error, not something to run() silently in half.
    """
    if not speaker_names.strip():
        return {}
    replacements = _parse_speaker_names(speaker_names)
    if len(replacements) > len(found):
        raise typer.BadParameter(
            f"--speaker-names: {len(replacements)} name(s) given, but only "
            f"{len(found)} speaker(s) were found."
        )
    unknown = [key for key in replacements if key not in found]
    if unknown:
        raise typer.BadParameter(
            f"--speaker-names: unknown speaker_id: {', '.join(unknown)}; "
            f"found: {', '.join(sorted(found))}."
        )
    return replacements


def _resolved_run_config(
    *,
    language: str,
    audio_stream: int | None,
    speakers: int | None,
    min_speakers: int,
    max_speakers: int,
    clean: list[str],
    profile_name: str,
    cpu_threads: int | None,
    ffmpeg_threads: int | None,
    chunk_minutes: int | None,
    interactive_names: bool,
    resume: bool,
    keep_working_audio: bool,
    asr_preprocess: str,
    asr_compression: str,
    context_prompt: str,
    llm_model: str,
    asr_device: str = "auto",
    diarization_provider: str = "foxnose",
    asr_models: list[str] | None = None,
) -> dict[str, object]:
    """Resolve profile defaults now, so manifest and startup output are reproducible.

    asr_provider/asr_backend are computed from the resolved asr_models list (each
    model's provider comes from the ASR_MODELS registry) rather than read from a
    caller-supplied flag: requires that nothing branch on a single
    job-wide provider switch, since a job may mix local and groq models.
    """
    profile = PROFILES[profile_name]
    requested_cpu = cpu_threads if cpu_threads is not None else profile["threads"]
    requested_ffmpeg = ffmpeg_threads if ffmpeg_threads is not None else profile["ffmpeg_threads"]
    resolved_chunk = chunk_minutes if chunk_minutes is not None else profile["chunk_minutes"]
    resolved_asr_models = list(asr_models) if asr_models is not None else list(DEFAULT_ASR_MODELS)
    providers = {ASR_MODELS[name]["provider"] for name in resolved_asr_models}
    computed_provider = next(iter(providers)) if len(providers) == 1 else "mixed"
    if "groq" in providers:
        resolved_chunk = min(
            resolved_chunk or GROQ_FREE_MAX_CHUNK_SECONDS // 60,
            GROQ_FREE_MAX_CHUNK_SECONDS // 60,
        )
    else:
        resolved_chunk = None
    return {
        "language": language,
        "audio_stream": audio_stream,
        "speakers": speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "clean": clean,
        "profile": profile_name,
        "asr_provider": computed_provider,
        "asr_backend": {"local": "gigaam", "groq": "groq", "deepgram": "deepgram"}.get(
            computed_provider, "mixed"
        ),
        "asr_models": resolved_asr_models,
        "asr_device": asr_device,
        "diarization_provider": diarization_provider,
        "diarization_model": {
            "pyannote": PYANNOTE_REPO,
            "foxnose": FOXNOSE_REPO,
            "deepgram": DEEPGRAM_REPO,
        }[diarization_provider],
        "cpu_threads": effective_threads(int(requested_cpu)),
        "ffmpeg_threads": effective_threads(
            int(requested_ffmpeg), reserve_for_parallel_jobs=profile["decode_workers"]
        ),
        "chunk_minutes": resolved_chunk,
        "interactive_names": interactive_names,
        "resume": resume,
        "keep_working_audio": keep_working_audio,
        "asr_preprocess": asr_preprocess,
        "asr_compression": asr_compression,
        "context_prompt": context_prompt.strip(),
        "llm_model": llm_model,
    }


def _display_value(value: object) -> str:
    if value is None:
        return "not set"
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value) or "not set"


def _asr_models_display(models: list[str]) -> str:
    if len(models) == 1:
        return f"{models[0]} (canonical)"
    return ", ".join(f"{name} (canonical)" if i == 0 else name for i, name in enumerate(models))


def _print_run_configuration(
    parts: list[Path], meeting_name: str, config: dict[str, object]
) -> None:
    """Print the complete resolved launch configuration before any audio work begins."""
    table = Table(title="Run parameters", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="bold")
    table.add_column("Value used")
    values = [
        ("Source files", ", ".join(str(path) for path in parts)),
        ("Meeting name", meeting_name),
        ("Language", config["language"]),
        ("FFmpeg audio stream", config["audio_stream"]),
        ("Speakers", config["speakers"] if config["speakers"] is not None else "auto"),
        ("Speaker bounds", f"{config['min_speakers']}–{config['max_speakers']}"),
        ("Output formats", config["clean"]),
        ("Profile", config["profile"]),
        ("ASR provider", config["asr_provider"]),
        ("ASR backend", config["asr_backend"]),
        ("ASR models", _asr_models_display(config["asr_models"])),
        *(
            [("Warning", "Audio will be sent to Groq")]
            if any(ASR_MODELS[name]["provider"] == "groq" for name in config["asr_models"])
            else []
        ),
        *(
            [("Warning", "Audio will be sent to Deepgram")]
            if any(ASR_MODELS[name]["provider"] == "deepgram" for name in config["asr_models"])
            or config["diarization_provider"] == "deepgram"
            else []
        ),
        ("ASR device (requested)", config["asr_device"]),
        (
            "Diarization (provider/model)",
            f"{config['diarization_provider']} / {config['diarization_model']}",
        ),
        ("CPU threads", config["cpu_threads"]),
        ("FFmpeg threads per process", config["ffmpeg_threads"]),
        ("ASR chunk length, min (Groq only)", config["chunk_minutes"]),
        ("Interactive names", config["interactive_names"]),
        ("Checkpoint reuse", config["resume"]),
        ("Keep working audio", config["keep_working_audio"]),
        ("Preprocessing", config["asr_preprocess"]),
        ("Compression", config["asr_compression"]),
        ("Context prompt (readable only)", config["context_prompt"]),
        ("readable model", config["llm_model"]),
    ]
    for name, value in values:
        table.add_row(name, _display_value(value))
    print(table)


@auth_app.command("set")
def set_api_key(
    provider: Annotated[str, typer.Argument(help="openai, groq, deepgram, or huggingface.")],
) -> None:
    """Save an OpenAI/Groq/Deepgram API key or HF_TOKEN to the project's protected .env."""
    names = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
        "huggingface": "HF_TOKEN",
    }
    if provider not in names:
        raise typer.BadParameter("Specify a provider: openai, groq, deepgram, or huggingface")
    name = names[provider]
    value = getpass(f"{name}: ").strip()
    try:
        save_project_secret(name, value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    os.environ[name] = value
    print(f"{provider}: value saved to the project's .env with 600 permissions.")


@auth_app.command("set-openai-base-url")
def set_openai_base_url() -> None:
    """Save an OpenAI-compatible API URL to the project's .env."""
    value = input("OPENAI_BASE_URL: ").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise typer.BadParameter("Enter a full URL, e.g. https://provider.example/v1")
    try:
        save_project_secret("OPENAI_BASE_URL", value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    os.environ["OPENAI_BASE_URL"] = value
    print("OPENAI_BASE_URL saved to the project's .env with 600 permissions.")


@auth_app.command("status")
def auth_status() -> None:
    """Show which keys are present without revealing their values."""
    # typer.echo + json.dumps, not rich's print(dict): the latter prints a Python
    # repr, not JSON, which an agent parsing stdout cannot rely on.
    typer.echo(
        json.dumps(
            {
                name: "configured" if os.environ.get(name) else "not configured"
                for name in sorted(PROJECT_ENV_NAMES)
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _free_space_bytes(path: Path) -> int:
    """Free space where `path` lives, tolerating a directory that does not exist yet.

    doctor runs before setup on a fresh checkout, and TRANSCRIBE_HOME may point at a
    directory the user has not created. shutil.disk_usage raises there, which turned a
    readiness check into an unhandled traceback; the nearest existing ancestor sits on
    the same filesystem, so its free space is the number we actually want.
    """
    for candidate in (path, *path.parents):
        if candidate.exists():
            return shutil.disk_usage(candidate).free
    return 0


def _finish_readable(job: Path, llm_model: str | None = None) -> None:
    """Run the readable pass if the job asked for it and does not have it yet.

    `run` and `resume` share this. The LLM pass lives outside run_job and has no
    `.stages/` marker, so a job interrupted before it used to stay without a
    readable transcript forever: resume finished the local stages, printed "Done"
    and left the manifest saying nothing was ever attempted. What the job asked
    for is in the manifest, which is why the requested modes are read from there
    rather than passed in — resume has no CLI flags to read them from.

    A pass already recorded as complete is never repeated: resuming a finished job
    must not spend the tokens a second time. A previously failed one is retried,
    since the cause is usually a missing key or an unreachable endpoint.
    """
    manifest = read_json(job / "manifest.json")
    if "readable" not in (manifest["config"].get("clean") or []):
        return
    if (manifest.get("readable") or {}).get("status") == "complete":
        return
    if not try_readable(job, llm_model):
        print(
            f"[yellow]readable was not created.[/] Retry separately: "
            f"`transcribe-meeting clean {job} --mode readable`."
        )


@app.command()
def doctor(
    asr_models: str = typer.Option(
        ",".join(DEFAULT_ASR_MODELS),
        "--asr-models",
        help="Check readiness of exactly these ASR models (same format as run/setup).",
    ),
    diarization_provider: str = typer.Option(
        "foxnose",
        "--diarization-provider",
        help="Check readiness of exactly this diarization route (same as run).",
    ),
) -> None:
    """Check dependencies, architecture, and the readiness of a specific route (ASR
    models + diarization — the same route as `run` with no options by default), without
    network access."""
    if diarization_provider not in DIARIZATION_PROVIDERS:
        raise typer.BadParameter("--diarization-provider: pyannote, foxnose, or deepgram")
    parsed_asr_models = _parse_asr_models(asr_models)
    route = route_readiness(parsed_asr_models, diarization_provider)
    checks = {
        "python_supported": sys.version_info[:2] in SUPPORTED_PYTHON,
        # Informational only: the server this runs on has no GPU at
        # all, so architecture must never gate setup or doctor's exit code.
        "apple_silicon": platform.machine() == "arm64",
        "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
        "free_space_ge_7gb": _free_space_bytes(paths.data_root()) >= 7 * 1024**3,
        # Route-specific: which local models are actually on disk
        # and which keys are set, never printing a value — only key presence.
        "asr_models": route["local_asr_models"],
        "asr_keys": route["asr_keys"],
        "vad_ready": route["vad_ready"],
        "diarization_provider": diarization_provider,
        "diarization_ready": route["diarization_ready"],
        "route_ready": route["route_ready"],
    }
    # stdout carries only the JSON so an agent can parse it unconditionally; the
    # remediation lines go to stderr, which is also where every other status line
    # goes. typer.echo, not rich's print: rich wraps long lines at 80 columns and a
    # wrapped line is invalid JSON (reproduced on a long readable.error in `status`).
    typer.echo(json.dumps(checks, ensure_ascii=False, indent=2))
    base_ready = all(checks[key] for key in ("python_supported", "ffmpeg", "free_space_ge_7gb"))
    if not base_ready or not route["route_ready"]:
        for problem in route["problems"]:
            CONSOLE.print(f"[yellow]{problem}[/]")
        raise typer.Exit(1)


@app.command()
def setup(
    diarization_provider: str = typer.Option(
        "all",
        "--diarization-provider",
        help="What to download for diarization: pyannote, foxnose, or all (both by "
        "default). deepgram downloads nothing (it's an API) — no need to specify it here.",
    ),
    asr_models: str = typer.Option(
        ",".join(DEFAULT_ASR_MODELS),
        "--asr-models",
        help=(
            "Which ASR models to prepare, comma-separated (1-3 of: " + ", ".join(ASR_MODELS) + "). "
            f"Local ones download to models/gigaam; {GROQ_ASR_MODEL_NAME} and "
            f"{DEEPGRAM_ASR_MODEL_NAME} have nothing to download — only a record of them "
            "is written to models.lock.json."
        ),
    ),
) -> None:
    """Download models into local storage. Requires HF_TOKEN: VAD segmentation-3.0
    is needed by local ASR. Community-1's terms are needed only for the pyannote route."""
    if diarization_provider not in ("pyannote", "foxnose", "all"):
        raise typer.BadParameter("--diarization-provider: pyannote, foxnose, or all")
    requested_asr_models = _parse_asr_models(asr_models)
    want_pyannote = diarization_provider in ("pyannote", "all")
    want_foxnose = diarization_provider in ("foxnose", "all")
    check_ffmpeg()
    # Architecture is informational only: this must run unmodified
    # on a Linux/CPU-only server, so nothing here may gate on platform.machine().
    if sys.version_info[:2] not in SUPPORTED_PYTHON:
        wanted = " or ".join(f"{a}.{b}" for a, b in SUPPORTED_PYTHON)
        raise typer.BadParameter(f"Python {wanted} is required: `uv sync --python 3.12`.")
    if shutil.disk_usage(paths.data_root()).free < 7 * 1024**3:
        raise typer.BadParameter(
            "At least 7 GB of free space is needed for GigaAM, VAD, the chosen "
            "diarization model, and the working WAV."
        )
    token = os.environ.get("HF_TOKEN")
    if not token:
        if want_pyannote:
            raise typer.BadParameter(
                "First accept the terms of both models: "
                "https://huggingface.co/pyannote/segmentation-3.0 and "
                "https://huggingface.co/pyannote/speaker-diarization-community-1, "
                "then set HF_TOKEN in the environment. Never pass the token in arguments or chat. "
                "To avoid accepting Community-1's terms, use "
                "`setup --diarization-provider foxnose` — this is the default diarization route."
            )
        raise typer.BadParameter(
            "First accept the terms of the model "
            "https://huggingface.co/pyannote/segmentation-3.0 "
            "(needed by local ASR's VAD), then set HF_TOKEN in the environment. "
            "Never pass the token in arguments or chat."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise typer.BadParameter("huggingface_hub is missing; run uv sync.") from exc
    paths.models_dir().mkdir(parents=True, exist_ok=True)
    requested_local_models = [
        name for name in requested_asr_models if ASR_MODELS[name]["provider"] == "local"
    ]
    requested_network_models = [
        name
        for name in requested_asr_models
        if ASR_MODELS[name]["provider"] in ("groq", "deepgram")
    ]
    # provisioning has no typer dependency of its own, so its failures surface as
    # RuntimeError; setup is the one place that turns them back into a BadParameter.
    try:
        gigaam_dir, downloaded_gigaam_models = provisioning.download_gigaam_models(
            requested_local_models
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # Never print, save, or embed token outside these authenticated download calls.
    hf_cache = paths.models_dir() / "huggingface"
    os.environ["HF_HOME"] = str(hf_cache)
    vad = snapshot_download(VAD_REPOSITORY, cache_dir=hf_cache, token=token)

    # Read from lock_path() (old or new location,) but always write
    # below to the new canonical models_dir()/models.lock.json — this is how a
    # project naturally migrates onto the new layout the first time setup runs again.
    new_models = provisioning.asr_lock_entries(
        gigaam_dir, downloaded_gigaam_models, vad, requested_network_models
    )
    diarization_entries, pyannote_path, wespeaker_path = provisioning.download_diarization_models(
        want_pyannote=want_pyannote,
        want_foxnose=want_foxnose,
        token=token,
        snapshot_download=snapshot_download,
    )
    new_models.update(diarization_entries)
    lock_target = provisioning.write_lock(new_models)

    # A generated local WAV proves loading and offline execution without uploading user audio.
    try:
        new_smoke = provisioning.run_smoke_tests(
            requested_local_models=requested_local_models,
            requested_asr_models=requested_asr_models,
            gigaam_dir=gigaam_dir,
            vad_path=vad,
            want_pyannote=want_pyannote,
            want_foxnose=want_foxnose,
            pyannote_path=pyannote_path,
            wespeaker_path=wespeaker_path,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    lock = read_json(lock_target)
    lock["smoke"] = {**lock.get("smoke", {}), **new_smoke}
    atomic_json(lock_target, lock)
    print(
        "Models downloaded and verified locally. HF_TOKEN can be removed from the "
        "environment; a normal run works offline."
    )


@app.command()
def run(
    parts: Annotated[list[Path], typer.Argument(help="One or more sequential meeting files.")],
    meeting_name: str = typer.Option(
        ..., help="Meeting name: used in the new output/ folder's name."
    ),
    language: str = typer.Option(
        "ru", help="Speech language in ISO-639-1, e.g. ru or en; auto for autodetection."
    ),
    audio_stream: int | None = typer.Option(
        None, help="FFmpeg audio stream index, if the source file has multiple tracks."
    ),
    speakers: str = typer.Option(
        "auto", help="Exact speaker count, or auto to detect within the min/max bounds."
    ),
    min_speakers: int = typer.Option(1, help="Lower speaker bound for --speakers auto."),
    max_speakers: int = typer.Option(8, help="Upper speaker bound for --speakers auto."),
    clean: str = typer.Option(
        "raw,light,readable",
        help="Comma-separated outputs: raw, light, and/or readable. readable requires "
        "OPENAI_API_KEY and network access; without it, raw and light remain.",
    ),
    profile: str = typer.Option("balanced", help="Resource profile: background, balanced, or max."),
    asr_provider: str | None = typer.Option(
        None,
        "--asr-provider",
        help=(
            "Legacy switch, kept for compatibility: local or groq. "
            "Each model's provider is now read individually from --asr-models, "
            "so this option usually does not need to be given — it only checks that "
            "an explicit --asr-models (if also given) does not contradict it."
        ),
    ),
    asr_device: str = typer.Option(
        "auto", help="GigaAM device: auto, mps, or cpu. Groq ignores this value."
    ),
    asr_models: str = typer.Option(
        ",".join(DEFAULT_ASR_MODELS),
        "--asr-models",
        help=(
            "ASR models, comma-separated, 1-3 of: " + ", ".join(ASR_MODELS) + ". "
            "The first is canonical: it sets utterance_id, utterances, timecodes, and "
            "the speaker attribution. The rest are only compared against the first during "
            "readable; the order of the first two is part of the fusion rules' "
            "calibration, so do not change it without a reason. By default "
            f"both are local and offline; {GROQ_ASR_MODEL_NAME} and {DEEPGRAM_ASR_MODEL_NAME} "
            "are network models (they send audio out) and are enabled only by explicitly "
            f"naming them here. {DEEPGRAM_ASR_MODEL_NAME} is the only model with "
            "scope=meeting: one request for the whole concatenated meeting instead of "
            "per part, so its speaker labels are consistent across parts; its speaker "
            "count cannot be set. Example: "
            "--asr-models multilingual_large_ctc,v3_rnnt," + GROQ_ASR_MODEL_NAME + "."
        ),
    ),
    diarization_provider: str = typer.Option(
        "foxnose",
        "--diarization-provider",
        help=(
            "Diarization: foxnose (FoxNoseTech/diarize, Apache-2.0, no token, "
            "the default), pyannote (Community-1, gated HF, marks overlap), or "
            "deepgram (Deepgram API, requires DEEPGRAM_API_KEY, sends audio out, "
            "the speaker count cannot be set — a mismatch with --speakers becomes "
            "a warning, not an error)."
        ),
    ),
    cpu_threads: int | None = typer.Option(
        None, help="Number of CPU threads for diarization; the profile sets the default."
    ),
    ffmpeg_threads: int | None = typer.Option(
        None,
        help="Threads per FFmpeg process; the profile sets the default and it is capped "
        "by the CPU.",
    ),
    chunk_minutes: int | None = typer.Option(
        None,
        help="ASR chunk length in minutes; automatically capped at 10 for Groq.",
    ),
    interactive_names: bool = typer.Option(
        None,
        "--interactive-names/--no-interactive-names",
        help="Ask for speaker names in the terminal; enabled by default only in a TTY.",
    ),
    speaker_names: str = typer.Option(
        "",
        "--speaker-names",
        help="Speaker names without interactivity, e.g. SPEAKER_00=Alice,SPEAKER_01=Bob "
        "(same format as `rename --set`). Applied after diarization; if there are more "
        "names than speakers found, or an unfamiliar speaker_id is given — an error.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Rarely needed for run: run always creates a new job; "
        "continue via the resume command.",
    ),
    keep_working_audio: bool = typer.Option(
        False, help="Do not delete temporary WAV files and ASR chunks after successful processing."
    ),
    asr_preprocess: str = typer.Option(
        "none",
        help="Preprocessing for ASR and diarization: none or hpf80-preemphasis.",
    ),
    asr_compression: str = typer.Option(
        "none", help="Compression for ASR and diarization: none or 4:1."
    ),
    context_prompt: str = typer.Option(
        "",
        help="Meeting topic and purpose (up to 2,000 characters); a hint for readable "
        "(the LLM text cleanup), not a source of facts. Has no effect on local GigaAM or "
        f"diarization; if {GROQ_ASR_MODEL_NAME} is among --asr-models, it is additionally "
        "passed as a prompt hint to the Whisper API transcription.",
    ),
    llm_model: str = typer.Option(
        "gpt-5.6-luna",
        help="Model ID for clean --mode readable; e.g. gpt-5.6-luna or gpt-5.6-terra.",
    ),
    measure_loudness: bool = typer.Option(
        False,
        "--measure-loudness/--no-measure-loudness",
        help="Measure LUFS/true peak for each part. Requires a full pass over the file "
        "and noticeably lengthens the run; it does not affect recognition quality.",
    ),
) -> None:
    """Create a job: ASR locally and/or via Groq/Deepgram (per model); diarization
    is local, except for --diarization-provider deepgram; readable is separate."""
    if not parts:
        raise typer.BadParameter("At least one part is required.")
    if profile not in PROFILES:
        raise typer.BadParameter("--profile: background, balanced, or max")
    parsed_asr_models, dropped_asr_models, _has_local_asr_model, has_groq_asr_model = (
        _validate_asr_options(
            asr_provider=asr_provider,
            asr_device=asr_device,
            asr_models=asr_models,
            diarization_provider=diarization_provider,
            language=language,
            chunk_minutes=chunk_minutes,
            cpu_threads=cpu_threads,
            ffmpeg_threads=ffmpeg_threads,
            asr_preprocess=asr_preprocess,
            asr_compression=asr_compression,
            context_prompt=context_prompt,
        )
    )
    # Before any file diagnostics, job directory, or decoded audio:
    # dropped network models never reach here (already removed above), so this only
    # ever catches a missing local checkpoint or a missing diarization model.
    _check_route_ready(parsed_asr_models, diarization_provider)
    modes = _validate_clean_options(clean)
    speaker_count = _validate_speakers(speakers, min_speakers, max_speakers)
    missing = [str(part) for part in parts if not part.is_file()]
    if missing:
        raise typer.BadParameter(f"Missing files: {', '.join(missing)}")
    config = _resolved_run_config(
        language=language,
        audio_stream=audio_stream,
        speakers=speaker_count,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        clean=modes,
        profile_name=profile,
        asr_device=asr_device,
        diarization_provider=diarization_provider,
        asr_models=parsed_asr_models,
        cpu_threads=cpu_threads,
        ffmpeg_threads=ffmpeg_threads,
        chunk_minutes=chunk_minutes,
        interactive_names=sys.stdin.isatty() if interactive_names is None else interactive_names,
        resume=resume,
        keep_working_audio=keep_working_audio,
        asr_preprocess=asr_preprocess,
        asr_compression=asr_compression,
        context_prompt=context_prompt,
        llm_model=llm_model,
    )
    # Persisted only for result.json's warnings — degrade already
    # decided what actually runs; this is purely a record of what got dropped.
    config["asr_dropped_models"] = dropped_asr_models
    _print_run_configuration(parts, meeting_name, config)
    paths.output_dir().mkdir(parents=True, exist_ok=True)
    # PID+timestamp, not a fixed name: two concurrent `run` invocations (or a crash
    # before create_job on a previous one) must not share or trip over one file.
    preflight_path = paths.output_dir() / f".preflight-{os.getpid()}-{time.time_ns()}.json"
    try:
        try:
            print(
                "[cyan]→[/cyan] Diagnosing source audio files"
                + (" and measuring loudness" if measure_loudness else "")
            )
            probes = diagnostics(parts, preflight_path, audio_stream, measure_loudness)
        except AudioError as exc:
            raise typer.BadParameter(str(exc)) from exc
        total_duration = sum(float(probe["duration"]) for probe in probes)
        local_model_count = sum(
            1 for name in parsed_asr_models if ASR_MODELS[name]["provider"] == "local"
        )
        required = estimate_working_bytes(total_duration, local_model_count)
        available = _free_space_bytes(paths.output_dir())
        if available < required:
            # Running out mid-job leaves a half-decoded job and a confusing ffmpeg
            # error; `setup`'s 7 GB check says nothing about how long this meeting is.
            raise typer.BadParameter(
                f"Not enough space: about {required / 1024**3:.1f} GB is needed for working "
                f"files ({len(probes)} part(s), total duration "
                f"{total_duration / 60:.0f} min, local models: {local_model_count}), "
                f"{available / 1024**3:.1f} GB free. Free up space or remove "
                "some models from --asr-models."
            )
        if has_groq_asr_model and total_duration > GROQ_FREE_AUDIO_SECONDS_PER_DAY:
            raise typer.BadParameter(
                "Groq free allows 28,800 seconds (8 hours) of audio per day. "
                "Split the run across several days or use local ASR."
            )
        job = create_job(parts, meeting_name, config, probes)
        # Preflight diagnostics becomes an immutable job artifact.
        shutil.move(str(preflight_path), job / "diagnostics.json")
    finally:
        # No-op once the move above succeeded; cleans up after any earlier exit.
        preflight_path.unlink(missing_ok=True)
    try:
        run_job(job, resume=resume)
        _finish_readable(job, llm_model)
        replacements = _resolve_speaker_name_overrides(speaker_names, _existing_speaker_names(job))
        if replacements:
            rename_speakers(job, replacements)
    except KeyboardInterrupt:
        print(f"Safely interrupted. Continue with: transcribe-meeting resume {job}")
        raise typer.Exit(130) from None
    except (typer.Exit, typer.Abort, typer.BadParameter):
        # typer's own control-flow exceptions derive from Exception (Exit from
        # RuntimeError, BadParameter from ClickException), so the broad handler below
        # would otherwise turn a plain usage error — a bad --speaker-names, say — into
        # a "pipeline crashed" message with an error.log and the wrong exit code.
        raise
    except Exception as exc:
        # A pipeline failure must leave a usable, resumable job, not a stack trace on
        # the user's terminal: the traceback is preserved on disk instead.
        (job / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[red]The run stopped with an error:[/] {exc}")
        print(f"Details: {job / 'error.log'}; continue with: transcribe-meeting resume {job}")
        raise typer.Exit(1) from exc
    write_job_result(job)
    print(f"Done: {job}")


@app.command()
def status(job_dir: Annotated[Path, typer.Argument(help="Job folder from output/. ")]) -> None:
    """Show the manifest and completed atomic stages."""
    _require_job_dir(job_dir)
    manifest = read_json(job_dir / "manifest.json")
    markers = (
        sorted(path.stem for path in (job_dir / ".stages").glob("*.json"))
        if (job_dir / ".stages").exists()
        else []
    )
    # typer.echo, not rich's print: see doctor's comment on the same choice.
    typer.echo(
        json.dumps(
            {
                "job": manifest["job_id"],
                "stages": markers,
                "readable": manifest.get("readable", {"status": "not requested"}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def resume(
    job_dir: Annotated[Path, typer.Argument(help="Existing folder of an unfinished job.")],
    cpu_threads: int | None = typer.Option(
        None, help="Temporary CPU thread count for this resume; the manifest is not changed."
    ),
    ffmpeg_threads: int | None = typer.Option(
        None, help="Temporary thread count per FFmpeg process; the manifest is not changed."
    ),
) -> None:
    _require_job_dir(job_dir)
    if cpu_threads is not None and cpu_threads < 1:
        raise typer.BadParameter("--cpu-threads must be at least 1")
    if ffmpeg_threads is not None and ffmpeg_threads < 1:
        raise typer.BadParameter("--ffmpeg-threads must be at least 1")
    run_job(
        job_dir,
        resume=True,
        cpu_threads_override=cpu_threads,
        ffmpeg_threads_override=ffmpeg_threads,
    )
    # The readable pass the job was configured for, if the interrupt landed before it.
    _finish_readable(job_dir)
    # A job interrupted mid-run (Ctrl-C in `run`) never reached run's own
    # write_job_result call; resume is what finishes it, so it must write result.json
    # itself too — otherwise a resumed-to-completion job would have none at all.
    write_job_result(job_dir)
    print(f"Done: {job_dir}")


@app.command()
def rename(
    job_dir: Annotated[Path, typer.Argument(help="Folder of a finished job from output/. ")],
    set_names: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="SPEAKER_00=Name; can be repeated or comma-separated within one "
            "flag. Without a TTY (e.g. from an agent), this is the only way to rename.",
        ),
    ] = None,
) -> None:
    """Rename participants and re-render without ASR/diarization."""
    _require_job_dir(job_dir)
    names = _existing_speaker_names(job_dir)
    if set_names:
        replacements: dict[str, str] = {}
        for value in set_names:
            replacements.update(_parse_speaker_names(value))
        unknown = [key for key in replacements if key not in names]
        if unknown:
            raise typer.BadParameter(
                f"Unknown speaker_id: {', '.join(unknown)}; found: {', '.join(sorted(names))}."
            )
    elif sys.stdin.isatty():
        replacements = {}
        for identifier, old in names.items():
            value = input(f"Name for {identifier} [{old}]: ").strip()
            if value:
                replacements[identifier] = value
    else:
        raise typer.BadParameter(
            "Without an interactive terminal, provide --set SPEAKER_00=Name[,SPEAKER_01=Name2,...]."
        )
    rename_speakers(job_dir, replacements)


@app.command()
def clean(
    job_dir: Annotated[Path, typer.Argument(help="Folder of a finished job from output/. ")],
    mode: str = typer.Option(
        ..., help="What to re-render: raw, light, readable, or several comma-separated."
    ),
    llm_model: str | None = typer.Option(None, help="Override the LLM only for this readable run."),
) -> None:
    """Re-render raw/light, or run the explicitly requested external readable cleanup."""
    _require_job_dir(job_dir)
    modes = _parse_clean(mode)
    if "readable" in modes:
        try:
            run_readable(job_dir, llm_model)
        except RuntimeError as exc:
            print(f"[yellow]readable pending:[/] {exc}")
            raise typer.Exit(2) from exc
    from .core import render_all

    render_all(job_dir, [item for item in modes if item != "readable"])
    write_job_result(job_dir)


@app.command()
def review(job_dir: Annotated[Path, typer.Argument(help="Job folder from output/. ")]) -> None:
    """Export every utterance to review/transcript_edit.csv for manual text and
    speaker edits (see `refine`)."""
    _require_job_dir(job_dir)
    typer.echo(prepare_review(job_dir))


@app.command()
def refine(
    job_dir: Annotated[Path, typer.Argument(help="Job folder from output/. ")],
    review_csv: Annotated[
        Path, typer.Argument(help="The edited CSV produced by the review command.")
    ],
) -> None:
    _require_job_dir(job_dir)
    if not review_csv.is_file():
        raise typer.BadParameter(f"CSV not found: {review_csv}")
    refine_review(job_dir, review_csv)


@glossary_app.command("list")
def glossary_list() -> None:
    typer.echo(yaml_dump({"terms": glossary()}))


@glossary_app.command("add")
def glossary_add(
    canonical: Annotated[str, typer.Argument(help="Canonical spelling of the term.")],
    variant: Annotated[
        list[str] | None, typer.Option(help="Alternate spelling; the flag can be repeated.")
    ] = None,
    note: Annotated[str, typer.Option(help="Optional note about the term.")] = "",
) -> None:
    variant = variant or []
    terms = glossary()
    terms.append({"canonical": canonical, "variants": variant, "note": note})
    write_glossary(terms)


@glossary_app.command("import")
def glossary_import(
    path: Annotated[Path, typer.Argument(help="YAML file with a terms key.")],
) -> None:
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded.get("terms"), list):
        raise typer.BadParameter("Expected YAML with a terms key.")
    write_glossary(loaded["terms"])


def yaml_dump(value: object) -> str:
    import yaml

    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    app()
