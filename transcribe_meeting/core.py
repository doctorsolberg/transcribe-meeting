from __future__ import annotations

import datetime as dt
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from . import __version__
from .alignment import assign_alt_words, assign_words, make_utterances, mark_overlap
from .asr import (
    ASR_API_KEY_ENV,
    ASR_MODELS,
    ASR_OUTPUT_SCHEMA_VERSION,
    DEEPGRAM_ASR_MODEL_NAME,
    GIGAAM_COMMIT,
    GigaAMBackend,
    GroqFreeTierLimiter,
    deepgram_words,
    deepgram_words_for_source,
    local_vad_parameters,
    transcribe_deepgram,
    transcribe_part,
    validate_word_timestamps,
    words_from_asr,
)
from .audio import concat_pcm, decode_pcm
from .cleaning import glossary_replace, light_clean
from .diarization import (
    DEEPGRAM_REPO,
    DIARIZATION_PROVIDERS,
    FOXNOSE_REPO,
    PYANNOTE_REPO,
    detect_overlap_regions,
    diarize,
)
from .job import (
    _LOCK_MIGRATION_WARNED,  # noqa: F401 - re-exported, tests clear() it via core._LOCK_MIGRATION_WARNED
    CANONICAL_SCHEMA_VERSION,  # noqa: F401 - re-exported, tests use core.CANONICAL_SCHEMA_VERSION
    _model_path,
    _warned_legacy_canonical,  # noqa: F401 - re-exported, tests discard() it via core._warned_legacy_canonical
    complete_stage,
    manifest_asr_models,
    model_lock,
    stage_valid,
)
from .job import load_utterances as _load_utterances
from .job import save_canonical as _save_canonical
from .model import UNKNOWN_SPEAKER, Utterance, Word
from .paths import data_root, glossary_path, models_dir, output_dir
from .readable import (
    CONCERN_FALLBACK,
    CONCERN_REPAIR,
    clean_readable,
    uncalibrated_fusion_order,
)
from .render import render_markdown
from .result import write_job_result
from .review import (
    CHANGE_COLUMNS,  # noqa: F401 - re-exported, historical core.CHANGE_COLUMNS
    prepare_review,  # noqa: F401 - re-exported, cli.py uses core.prepare_review
    refine,  # noqa: F401 - re-exported as core.refine (cli.py: `refine as refine_review`)
    write_readable_changes,
)
from .timeline import build_sources, mapping
from .util import atomic_json, atomic_text, fmt_time, read_json, sha256

PROFILES = {
    "background": {
        "threads": 2,
        "chunk_minutes": 10,
        "ffmpeg_threads": 1,
        "decode_workers": 1,
    },
    "balanced": {
        "threads": 4,
        "chunk_minutes": 30,
        "ffmpeg_threads": 2,
        "decode_workers": 2,
    },
    "max": {
        "threads": 10,
        "chunk_minutes": 0,
        "ffmpeg_threads": 4,
        "decode_workers": 2,
    },
}

# soft_wrap=True: a shell-command suggestion (e.g. from route_readiness) must stay
# copy-pasteable as one line, not get hard-wrapped mid-flag at the terminal width.
CONSOLE = Console(stderr=True, soft_wrap=True)


def report_step(message: str) -> None:
    CONSOLE.print(f"[cyan]→[/cyan] {message}")


#: Bytes one second of working audio occupies: mono 16 kHz signed 16-bit PCM, the
#: format every stage downstream of decode_pcm assumes.
WORKING_AUDIO_BYTES_PER_SECOND = 16_000 * 2


def estimate_working_bytes(total_duration: float, local_model_count: int) -> int:
    """Peak scratch space a run needs before `run_job` deletes work/ and the chunk WAVs.

    Three things coexist at the peak: the decoded per-part PCM, the concatenated
    technical track, and one set of sliced chunk WAVs per local model (they are only
    swept once the whole job finishes). Network-only models add nothing on disk.
    A quarter is added for the JSON artifacts and ordinary filesystem slack.
    """
    tracks = 2 + max(0, local_model_count)
    return int(total_duration * WORKING_AUDIO_BYTES_PER_SECOND * tracks * 1.25)


def effective_threads(requested: int, *, reserve_for_parallel_jobs: int = 1) -> int:
    """Avoid CPU oversubscription; more runnable threads are not faster on this Mac."""
    available = os.cpu_count() or requested
    return max(1, min(requested, max(1, available // reserve_for_parallel_jobs)))


def make_asr_prompt(terms: list[dict[str, Any]], context_prompt: str = "") -> str:
    """Build local Whisper guidance without treating context as spoken evidence."""
    sections: list[str] = []
    if context_prompt.strip():
        sections.append(
            f"Контекст встречи (не добавляй фактов из него в текст): {context_prompt.strip()}"
        )
    canonical_terms = [term["canonical"] for term in terms]
    if canonical_terms:
        sections.append("Канонические термины: " + ", ".join(canonical_terms))
    return "\n".join(sections)


def asr_stage_fingerprint(
    base_fingerprint: dict[str, Any],
    glossary_terms: list[dict[str, Any]],
    effective_device: str | None,
    asr_models: list[str],
) -> dict[str, Any]:
    """Build the asr stage fingerprint; changing the model list must invalidate resume."""
    return {
        **base_fingerprint,
        "stage": "asr",
        "glossary": glossary_terms,
        "output_schema_version": ASR_OUTPUT_SCHEMA_VERSION,
        "effective_asr_device": effective_device,
        "asr_models": list(asr_models),
    }


def route_readiness(asr_models: list[str], diarization_provider: str) -> dict[str, Any]:
    """Check whether every local checkpoint and API key a route needs is already on
    disk or in the environment — no network call, no model load.

    Shared by `run`'s preflight and `doctor`, so
    both agree on the same definition of "ready": `run` turns a not-ready route into
    a BadParameter before create_job even runs, and `doctor` reports it and exits 1.
    Local checkpoint paths are resolved exactly like the stages that actually load
    them (_stage_asr/_stage_diarization above, GigaAMBackend._load), so this can
    never say "ready" about a route that then fails deep inside a stage.
    """
    lock_models = model_lock().get("models", {})
    gigaam_dir = _model_path(lock_models.get("gigaam", {}), "gigaam")
    local_asr_models: dict[str, bool] = {}
    asr_keys: dict[str, bool] = {}
    problems: list[str] = []
    for name in asr_models:
        info = ASR_MODELS.get(name)
        if info is None:
            continue
        if info["provider"] == "local":
            ready = (gigaam_dir / info["checkpoint"]).is_file()
            local_asr_models[name] = ready
            if not ready:
                problems.append(
                    f"Model {name} has not been downloaded. Run: "
                    f"transcribe-meeting setup --asr-models {name}"
                )
        else:
            env_name = ASR_API_KEY_ENV[info["provider"]]
            ready = bool(os.environ.get(env_name))
            asr_keys[env_name] = ready
            if not ready:
                problems.append(
                    f"ASR model {name} requires {env_name}. Run: "
                    f"transcribe-meeting auth set {info['provider']}"
                )

    vad_ready = True
    # VAD (segmentation-3.0) is not selected by --asr-models, but every local model
    # needs it (detect_speech_regions), and plain `setup` always fetches it
    # regardless of --asr-models — unlike the GigaAM checkpoint above, one command
    # fixes this for any local model, so it is not tied to a particular model name.
    if any(ASR_MODELS.get(name, {}).get("provider") == "local" for name in asr_models):
        vad_ready = _model_path(lock_models.get("vad", {}), "huggingface").exists()
        if not vad_ready:
            problems.append(
                "Local VAD model (segmentation-3.0) not found. Accept the terms on "
                "Hugging Face, then run: transcribe-meeting setup"
            )

    diarization_ready = True
    diarization_key: dict[str, bool] | None = None
    if diarization_provider == "pyannote":
        diarization_ready = _model_path(
            lock_models.get("pyannote", {}), "speaker-diarization-community-1"
        ).exists()
        if not diarization_ready:
            problems.append(
                "Diarization pyannote (Community-1) has not been downloaded. Run: "
                "transcribe-meeting setup --diarization-provider pyannote"
            )
    elif diarization_provider == "foxnose":
        diarization_ready = _model_path(
            lock_models.get("wespeaker", {}), str(Path("wespeaker") / "en" / "model.onnx")
        ).is_file()
        if not diarization_ready:
            problems.append(
                "Diarization foxnose (WeSpeaker) has not been downloaded. Run: "
                "transcribe-meeting setup --diarization-provider foxnose"
            )
    elif diarization_provider == "deepgram":
        ready = bool(os.environ.get("DEEPGRAM_API_KEY"))
        diarization_key = {"DEEPGRAM_API_KEY": ready}
        diarization_ready = ready
        if not ready:
            problems.append(
                "Diarization deepgram requires DEEPGRAM_API_KEY. Run: "
                "transcribe-meeting auth set deepgram"
            )

    return {
        "local_asr_models": local_asr_models,
        "asr_keys": asr_keys,
        "vad_ready": vad_ready,
        "diarization_ready": diarization_ready,
        "diarization_key": diarization_key,
        "route_ready": not problems,
        "problems": problems,
    }


def glossary() -> list[dict[str, Any]]:
    path = glossary_path()
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("terms", [])


def write_glossary(terms: list[dict[str, Any]]) -> None:
    atomic_text(
        glossary_path(),
        yaml.safe_dump({"terms": terms}, allow_unicode=True, sort_keys=False),
    )


def create_job(
    parts: list[Path], meeting_name: str, config: dict[str, Any], probes: list[dict[str, Any]]
) -> Path:
    sources = build_sources(
        [
            (str(part.resolve()), sha256(part), probe["duration"])
            for part, probe in zip(parts, probes, strict=True)
        ]
    )
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    slug = (
        "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in meeting_name).strip("-")
        or "meeting"
    )
    # Two runs in the same second with the same --meeting-name would otherwise collide
    # on stamp+slug; mkdir(exist_ok=False) still does the actual race-safe check, this
    # loop only supplies the next name to try after it raises.
    job = output_dir() / f"{stamp}-{slug}"
    suffix = 1
    while True:
        try:
            job.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            job = output_dir() / f"{stamp}-{slug}-{suffix}"
    manifest = {
        "schema_version": 2,
        "job_id": job.name,
        "meeting_name": meeting_name,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "sources": mapping(sources),
        "config": config,
        "code_version": __version__,
        "models": model_lock(),
        "glossary": glossary(),
        "stages": {},
    }
    atomic_json(job / "manifest.json", manifest)
    return job


def speaker_previews(
    aligned_parts: list[tuple[dict[str, Any], list[Word]]], max_characters: int = 180
) -> dict[str, dict[str, Any]]:
    """Return the best early contiguous phrase for each diarization speaker."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for source, words in aligned_parts:
        run: list[Word] = []
        for word in [*words, None]:
            if word is not None and (not run or word.speaker_id == run[0].speaker_id):
                run.append(word)
                continue
            if run and run[0].speaker_id != UNKNOWN_SPEAKER:
                text = "".join(item.text for item in run).strip()
                if text:
                    candidates.setdefault(run[0].speaker_id, []).append(
                        {
                            "meeting_start": source["offset"] + run[0].start,
                            "source_file": source["path"],
                            "text": text,
                        }
                    )
            run = [] if word is None else [word]
    previews = {}
    for speaker_id, runs in candidates.items():
        # Prefer a useful phrase over a one-word acknowledgement, retaining chronology on ties.
        selected = next((run for run in runs if len(run["text"]) >= 40), runs[0])
        previews[speaker_id] = {**selected, "text": selected["text"][:max_characters].rstrip()}
    return previews


def _names(
    job: Path,
    speaker_ids: list[str],
    interactive: bool,
    previews: dict[str, dict[str, Any]],
) -> dict[str, str]:
    path = job / "speakers.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("speakers", {})
    names = {identifier: identifier for identifier in speaker_ids}
    if interactive and sys.stdin.isatty():
        print(f"Found {len(names)} participants.")
        for number, identifier in enumerate(names, 1):
            preview = previews.get(identifier)
            print(f"\nSpeaker {number} ({identifier})")
            if preview:
                print(
                    f"  First fragment: [{fmt_time(preview['meeting_start'])}] "
                    f"{Path(preview['source_file']).name}: {preview['text']}"
                )
            else:
                print("  No confidently attributed words; the default name can be kept.")
            entered = input(f"Name for Speaker {number} [{identifier}]: ").strip()
            if entered:
                names[identifier] = entered
    atomic_text(path, yaml.safe_dump({"speakers": names}, allow_unicode=True, sort_keys=True))
    return names


def _working_audio(
    job: Path, manifest: dict[str, Any], ffmpeg_threads: int, decode_workers: int
) -> list[Path]:
    destination = job / "work" / "pcm"
    result: list[Path] = []
    preprocessing: list[dict[str, Any]] = []
    diagnostics = read_json(job / "diagnostics.json")["parts"]

    def normalized_path(value: str) -> str:
        path = Path(value)
        return str(path.resolve() if path.is_absolute() else (data_root() / path).resolve())

    streams_by_path = {normalized_path(item["path"]): item["stream_index"] for item in diagnostics}
    pending: list[tuple[int, dict[str, Any], Path, int]] = []
    reports: dict[int, dict[str, Any]] = {}
    for index, source in enumerate(manifest["sources"], 1):
        target = destination / f"{index:03d}.wav"
        if not target.exists():
            source_path = Path(normalized_path(source["path"]))
            try:
                decoded_stream = streams_by_path[str(source_path)]
            except KeyError as exc:
                raise RuntimeError(
                    f"Diagnostics has no audio stream for {source_path}. "
                    "Create the job again with the run command."
                ) from exc
            pending.append((index, source, target, decoded_stream))
        else:
            reports[index] = {
                "mode": manifest["config"]["asr_preprocess"],
                "compression": manifest["config"]["asr_compression"],
                "reused": True,
            }
    if pending:
        report_step(
            f"[1/5] Preparing {len(pending)} part(s) with "
            f"{min(decode_workers, len(pending))} FFmpeg thread(s)"
        )

        def decode_one(item: tuple[int, dict[str, Any], Path, int]) -> tuple[int, dict[str, Any]]:
            index, source, target, stream = item
            return index, decode_pcm(
                Path(source["path"]),
                target,
                stream,
                manifest["config"]["asr_preprocess"],
                manifest["config"]["asr_compression"],
                ffmpeg_threads,
            )

        with ThreadPoolExecutor(max_workers=min(decode_workers, len(pending))) as executor:
            futures = {executor.submit(decode_one, item): item[0] for item in pending}
            for future in as_completed(futures):
                index, report = future.result()
                reports[index] = report
                report_step(f"Prepared part {index}/{len(manifest['sources'])}")
    for index, source in enumerate(manifest["sources"], 1):
        preprocessing.append({"source_file": source["path"], **reports[index]})
        result.append(destination / f"{index:03d}.wav")
    diagnostics_path = job / "diagnostics.json"
    diagnostics_data = read_json(diagnostics_path)
    diagnostics_data["asr_and_diarization_preprocessing"] = preprocessing
    atomic_json(diagnostics_path, diagnostics_data)
    return result


def detect_and_record_overlap(
    audio: Path, diar_file: Path, vad_path: Path, cpu_threads: int
) -> None:
    """Run overlap detection and merge its result into an already-written diarization.json.

    Never raises: the overlap mark is a supplement to diarization, not a
    pipeline prerequisite, so any failure here (missing model, bad audio, whatever)
    is recorded as `overlap_error` and the diarization stage still completes with an
    empty `overlap` list, instead of losing the diarization that already succeeded.
    """
    try:
        overlap_regions = detect_overlap_regions(audio, vad_path, cpu_threads)
        overlap_error = None
    except Exception as exc:  # overlap detection must never fail the run
        overlap_regions = []
        overlap_error = str(exc)
        report_step(f"Overlap detection skipped: {exc}")
    diar_payload = read_json(diar_file)
    diar_payload["overlap"] = [[start, end] for start, end in overlap_regions]
    diar_payload["overlap_error"] = overlap_error
    atomic_json(diar_file, diar_payload)


def _run_deepgram_asr(
    job: Path, manifest: dict[str, Any], model_name: str, model_index: int, model_count: int
) -> None:
    """scope="meeting" ASR: one request on work/meeting.wav,
    split into the usual per-part `asr/<model>/NNN.json` files afterwards.

    Never calls transcribe_deepgram (and so never touches the network) when every
    part already has its file: a resumed job whose parts all exist must not re-read
    or re-request meeting.json just to reproduce output it already wrote.
    """
    sources = manifest["sources"]
    missing = [
        (index, source)
        for index, source in enumerate(sources, 1)
        if not (job / "asr" / model_name / f"{index:03d}.json").exists()
    ]
    if not missing:
        return
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError(f"ASR model {model_name} needs DEEPGRAM_API_KEY.")
    report_step(
        f"[2/5] Transcribing the whole meeting in Deepgram ({model_name}, "
        f"model {model_index}/{model_count})"
    )
    checkpoint = job / "asr" / model_name / "meeting.json"
    data = transcribe_deepgram(
        job / "work" / "meeting.wav", api_key, manifest["config"]["language"], checkpoint
    )
    all_words = deepgram_words(data)
    for index, source in missing:
        words = deepgram_words_for_source(
            all_words, source["diarization_offset"], source["duration"]
        )
        # Deepgram estimates each word's span on its own, so spans overlap; the
        # bounds checks still apply. See validate_word_timestamps.
        validate_word_timestamps(words, source["duration"], allow_overlap=True)
        atomic_json(
            job / "asr" / model_name / f"{index:03d}.json",
            {
                "text": "",
                "segments": [],
                "words": words,
                "run_parameters": {
                    "provider": "deepgram",
                    "model": ASR_MODELS[model_name]["api_model"],
                    "word_timestamps": True,
                    "language": manifest["config"]["language"],
                    "output_schema_version": ASR_OUTPUT_SCHEMA_VERSION,
                },
            },
        )
        report_step(f"[2/5] Distributing Deepgram words to part {index}/{len(sources)}")


def _stage_audio(
    job: Path,
    manifest: dict[str, Any],
    fingerprint: dict[str, Any],
    *,
    resume: bool,
    ffmpeg_threads: int,
    decode_workers: int,
) -> None:
    if (
        not resume
        or not stage_valid(job, "audio", fingerprint)
        or not (job / "work" / "meeting.wav").exists()
    ):
        parts = _working_audio(job, manifest, ffmpeg_threads, decode_workers)
        report_step("Assembling a single technical track for diarization")
        concat_pcm(parts, job / "work" / "meeting.wav", ffmpeg_threads)
        complete_stage(job, "audio", fingerprint)
    else:
        report_step("[1/5] Prepared audio is reused from checkpoint")


def _stage_asr(
    job: Path,
    manifest: dict[str, Any],
    fingerprint: dict[str, Any],
    *,
    resume: bool,
    ffmpeg_threads: int,
) -> tuple[list[str], Path]:
    """Run ASR for every configured model. Returns (asr_models, vad_path) — vad_path
    is needed again by the diarization stage's overlap detection."""
    config = manifest["config"]
    asr_models: list[str] = list(config["asr_models"])
    unknown_asr_models = [name for name in asr_models if name not in ASR_MODELS]
    if unknown_asr_models:
        raise RuntimeError(f"Unknown ASR models in manifest: {', '.join(unknown_asr_models)}.")
    # Provider is per-model (ASR_MODELS[name]["provider"]), never a single job-wide
    # switch: a job may mix local GigaAM models with the Groq model.
    # config["asr_provider"] is the pre-computed local/groq/mixed label for display
    # only; nothing below branches on it.
    requested_device = str(config["asr_device"])
    effective_device: str | None = None
    if any(ASR_MODELS[name]["provider"] == "local" for name in asr_models):
        from .asr import effective_asr_device

        effective_device = effective_asr_device(requested_device)
    asr_fp = asr_stage_fingerprint(fingerprint, manifest["glossary"], effective_device, asr_models)
    local_models = model_lock()["models"]
    gigaam_info = local_models.get("gigaam", {})
    vad_info = local_models.get("vad", {})
    gigaam_path = _model_path(gigaam_info, "gigaam")
    vad_path = _model_path(vad_info, "huggingface")
    if not resume or not stage_valid(job, "asr", asr_fp):
        # local's _local_part never reads this argument (GigaAM has no prompt in its
        # API); only groq's request actually uses it, so building it once up front for
        # a run with no groq model at all would just be thrown away.
        prompt = make_asr_prompt(manifest["glossary"], manifest["config"]["context_prompt"])
        groq_model_name = next(
            (name for name in asr_models if ASR_MODELS[name]["provider"] == "groq"), None
        )
        deepgram_model_name = next(
            (name for name in asr_models if ASR_MODELS[name]["provider"] == "deepgram"), None
        )
        groq_limiter = GroqFreeTierLimiter(status=report_step) if groq_model_name else None
        manifest["asr"] = {
            "provider": config["asr_provider"],
            "backend": config["asr_backend"],
            "models": asr_models,
            "canonical_model": asr_models[0],
        }
        if effective_device is not None:
            manifest["asr"]["requested_device"] = requested_device
            manifest["asr"]["effective_device"] = effective_device
        if groq_model_name is not None:
            manifest["asr"]["api_model"] = ASR_MODELS[groq_model_name]["api_model"]
        if deepgram_model_name is not None:
            manifest["asr"]["api_model"] = ASR_MODELS[deepgram_model_name]["api_model"]
        atomic_json(job / "manifest.json", manifest)
        # Local models load and close strictly one at a time: each GigaAM instance is
        # 3.1 GB RSS at peak (measured) and two of them do not fit here. The
        # groq and deepgram models need no local backend at all.
        for model_index, model_name in enumerate(asr_models, 1):
            model_info = ASR_MODELS[model_name]
            model_provider = model_info["provider"]
            if model_info.get("scope") == "meeting":
                # deepgram-nova-3: one request on the whole concatenated track, then
                # sliced into the usual per-part files, instead of
                # the per-source loop below that "part"-scope models use.
                _run_deepgram_asr(job, manifest, model_name, model_index, len(asr_models))
                continue
            local_backend = None
            if model_provider == "local":
                assert effective_device is not None
                local_backend = GigaAMBackend(gigaam_path, effective_device, model_name)
            try:
                for index, source in enumerate(manifest["sources"], 1):
                    destination = job / "asr" / model_name / f"{index:03d}.json"
                    if not destination.exists():
                        report_step(
                            f"[2/5] Transcribing part {index}/{len(manifest['sources'])} "
                            f"({model_name}, model {model_index}/{len(asr_models)})"
                        )
                        local_fp = None
                        if model_provider == "local":
                            local_fp = {
                                "source_sha256": source["sha256"],
                                "gigaam_model": model_name,
                                "gigaam_model_hash": gigaam_info.get("models", {})
                                .get(model_name, {})
                                .get("sha256"),
                                "gigaam_commit": gigaam_info.get("commit", GIGAAM_COMMIT),
                                "vad_revision": vad_info.get("revision"),
                                "vad_parameters": local_vad_parameters(),
                                "device": getattr(local_backend, "device", None),
                            }
                        transcribe_part(
                            job / "work" / "pcm" / f"{index:03d}.wav",
                            destination,
                            gigaam_path,
                            config["language"],
                            prompt,
                            provider=model_provider,
                            chunk_minutes=config["chunk_minutes"] or 0,
                            ffmpeg_threads=ffmpeg_threads,
                            groq_limiter=groq_limiter,
                            local_backend=local_backend,
                            vad_model_path=vad_path,
                            local_fingerprint=local_fp,
                            model_name=model_name,
                        )
                    else:
                        report_step(
                            f"[2/5] ASR for part {index} ({model_name}) is reused from checkpoint"
                        )
            finally:
                if local_backend is not None:
                    local_backend.close()
        complete_stage(job, "asr", asr_fp)
    else:
        report_step("[2/5] ASR is reused from checkpoint")
    return asr_models, vad_path


def _stage_diarization(
    job: Path,
    manifest: dict[str, Any],
    fingerprint: dict[str, Any],
    vad_path: Path,
    *,
    resume: bool,
    cpu_threads: int,
) -> dict[str, Any]:
    """Run diarization plus overlap detection. Returns diar_fp, needed to derive the
    align stage's fingerprint."""
    config = manifest["config"]
    diarization_provider = config["diarization_provider"]
    diarization_models = model_lock()["models"]
    pyannote_path = _model_path(
        diarization_models.get("pyannote", {}), "speaker-diarization-community-1"
    )
    wespeaker_path = _model_path(
        diarization_models.get("wespeaker", {}), str(Path("wespeaker") / "en" / "model.onnx")
    )
    diarize_package_version = (
        diarization_models.get("diarize", {}).get("version")
        if diarization_provider == "foxnose"
        else None
    )
    diar_fp = {
        **fingerprint,
        "stage": "diarization",
        "diarization_provider": diarization_provider,
        "diarize_package_version": diarize_package_version,
        # Bump this when detect_overlap_regions's logic changes: a checkpoint
        # completed before overlap detection existed must not be reused as-is, or
        # diarization.json would keep silently missing the "overlap" key forever.
        "overlap_detection_version": 1,
    }
    diar_file = job / "diarization" / "diarization.json"
    if (
        diarization_provider == "foxnose"
        and config["speakers"] is None
        and config["max_speakers"] > 7
    ):
        # foxnose's speaker-count auto-estimation is known to degrade above ~7 participants;
        # surface this every run (not only when the stage actually executes) so the user can
        # decide whether to pin --speakers N before waiting through diarization.
        report_step(
            "Warning: --diarization-provider foxnose estimates the speaker count less "
            "reliably with 8+ participants; specifying --speakers N explicitly is recommended."
        )
    if not resume or not stage_valid(job, "diarization", diar_fp):
        where = "CPU" if diarization_provider != "deepgram" else "Deepgram API"
        report_step(f"[3/5] Diarizing the whole meeting ({diarization_provider}, {where})")
        diarization_model = {
            "pyannote": PYANNOTE_REPO,
            "foxnose": FOXNOSE_REPO,
            "deepgram": DEEPGRAM_REPO,
        }[diarization_provider]
        manifest["diarization"] = {
            "provider": diarization_provider,
            "model": diarization_model,
            "cpu_threads": cpu_threads,
        }
        atomic_json(job / "manifest.json", manifest)
        diarize(
            job / "work" / "meeting.wav",
            diar_file,
            job / "diarization" / "diarization.rttm",
            pyannote_path,
            config["speakers"],
            config["min_speakers"],
            config["max_speakers"],
            cpu_threads,
            provider=diarization_provider,
            wespeaker_model=wespeaker_path,
            deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY"),
            deepgram_language=config["language"],
            deepgram_checkpoint=job / "asr" / DEEPGRAM_ASR_MODEL_NAME / "meeting.json",
        )
        # Overlap detection runs once on the technical track regardless of which
        # diarization provider ran above: it uses segmentation-3.0 (the VAD model
        # local ASR already requires), never foxnose's or pyannote's own output.
        report_step("[3/5] Detecting overlapping speech (segmentation-3.0)")
        detect_and_record_overlap(job / "work" / "meeting.wav", diar_file, vad_path, cpu_threads)
        complete_stage(job, "diarization", diar_fp)
    else:
        report_step("[3/5] Diarization is reused from checkpoint")
    return diar_fp


def _reconcile_speaker_count(
    diarization_provider: str, ids: list[str], expected: int | None, diar_data: dict[str, Any]
) -> None:
    """Deepgram's API has no speaker-count parameter, so a mismatch
    against an explicit --speakers there is expected behaviour, not a broken run — it
    becomes a warning recorded on `diar_data` (mutated in place, surfaced later via
    canonical.json into result.json's warnings) instead of failing the whole job.
    Every other provider keeps the hard failure: disagreeing with an explicit
    --speakers is still a sign something actually went wrong.
    """
    if not isinstance(expected, int) or len(ids) == expected:
        return
    message = (
        f"{diarization_provider} returned {len(ids)} speaker ID(s) instead of the "
        f"requested {expected}."
    )
    if diarization_provider != "deepgram":
        raise RuntimeError(message)
    report_step(f"Warning: {message} Deepgram has no speaker-count parameter.")
    diar_data["speaker_count_mismatch"] = message


def _stage_align(
    job: Path,
    manifest: dict[str, Any],
    asr_models: list[str],
    diar_data: dict[str, Any],
    align_fp: dict[str, Any],
    diarization_provider: str,
    *,
    resume: bool,
) -> None:
    config = manifest["config"]
    if not resume or not stage_valid(job, "align", align_fp):
        report_step("[4/5] Attributing words to speakers and building utterances")
        ids = sorted({item["speaker"] for item in diar_data["exclusive"]})
        _reconcile_speaker_count(diarization_provider, ids, config["speakers"], diar_data)
        canonical_model = asr_models[0]
        aligned_parts: list[tuple[dict[str, Any], list[Word]]] = []
        for index, source in enumerate(manifest["sources"], 1):
            words = [
                Word(**word)
                for word in words_from_asr(
                    read_json(job / "asr" / canonical_model / f"{index:03d}.json")
                )
            ]
            # Convert exclusive annotation from technical clock to local source clock.
            base = source["diarization_offset"]
            local = [
                (x["start"] - base, x["end"] - base, x["speaker"])
                for x in diar_data["exclusive"]
                if x["end"] > base and x["start"] < base + source["duration"]
            ]
            assign_words(words, local)
            aligned_parts.append((source, words))
        previews = speaker_previews(aligned_parts)
        names = _names(job, ids, config["interactive_names"], previews)
        # The set and order of utterance_id come only from the first
        # (canonical) model. The loop below never reads a second model's words.
        utterances: list[Utterance] = []
        for index, (source, words) in enumerate(aligned_parts, 1):
            utterances.extend(
                make_utterances(
                    words,
                    source_file=source["path"],
                    source_offset=source["offset"],
                    speaker_names=names,
                    prefix=f"p{index:03d}",
                )
            )
        diarization_offsets = {
            source["path"]: source["diarization_offset"] for source in manifest["sources"]
        }
        overlap_regions = [(item[0], item[1]) for item in diar_data.get("overlap", [])]
        mark_overlap(utterances, overlap_regions, diarization_offsets)
        # Same rule again: every non-canonical model only ever contributes text into
        # the canonical utterances built above; it never adds, removes, or resizes one.
        for variant_model in asr_models[1:]:
            report_step(f"[4/5] Projecting {variant_model}'s words onto the canonical utterances")
            for index, source in enumerate(manifest["sources"], 1):
                variant_words = [
                    Word(**word)
                    for word in words_from_asr(
                        read_json(job / "asr" / variant_model / f"{index:03d}.json")
                    )
                ]
                source_utterances = [
                    utterance for utterance in utterances if utterance.source_file == source["path"]
                ]
                assigned = assign_alt_words(variant_words, source_utterances)
                for utterance in source_utterances:
                    text = assigned.get(utterance.utterance_id)
                    if text:
                        utterance.variants[variant_model] = text
        terms = manifest["glossary"]
        for item in utterances:
            item.raw_text = glossary_replace(item.raw_text, terms)
            item.light_text = light_clean(item.raw_text)
            for variant_model, text in item.variants.items():
                cleaned = glossary_replace(text, terms)
                item.variants[variant_model] = cleaned
                item.variant_light[variant_model] = light_clean(cleaned)
        _save_canonical(job, manifest, utterances, diar_data, previews)
        complete_stage(job, "align", align_fp)
    else:
        report_step("[4/5] Attribution and utterances are reused from checkpoint")


def run_job(
    job: Path,
    *,
    resume: bool = True,
    cpu_threads_override: int | None = None,
    ffmpeg_threads_override: int | None = None,
) -> None:
    manifest = read_json(job / "manifest.json")
    config = manifest["config"]
    profile = PROFILES[config["profile"]]
    threads = cpu_threads_override or config["cpu_threads"]
    requested_ffmpeg_threads = ffmpeg_threads_override or config["ffmpeg_threads"]
    cpu_threads = effective_threads(threads)
    ffmpeg_threads = effective_threads(
        requested_ffmpeg_threads, reserve_for_parallel_jobs=profile["decode_workers"]
    )
    if cpu_threads != threads:
        report_step(f"CPU threads limited to {cpu_threads}: the number of Mac cores available")
    if ffmpeg_threads != requested_ffmpeg_threads:
        report_step(
            f"FFmpeg uses {ffmpeg_threads} thread(s) per process to avoid overloading the CPU"
        )
    report_step(
        f"Resources: CPU={cpu_threads}, FFmpeg={ffmpeg_threads} × "
        f"{profile['decode_workers']} task(s)"
    )
    os.environ.update(
        {
            "OMP_NUM_THREADS": str(cpu_threads),
            "VECLIB_MAXIMUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "NUMEXPR_NUM_THREADS": str(cpu_threads),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYANNOTE_METRICS_ENABLED": "0",
            "HF_HOME": str(models_dir() / "huggingface"),
        }
    )
    fingerprint = {
        "sources": [(x["sha256"], x["duration"]) for x in manifest["sources"]],
        "config": config,
        "models": model_lock(),
    }
    _stage_audio(
        job,
        manifest,
        fingerprint,
        resume=resume,
        ffmpeg_threads=ffmpeg_threads,
        decode_workers=profile["decode_workers"],
    )
    diarization_provider = config["diarization_provider"]
    if diarization_provider not in DIARIZATION_PROVIDERS:
        raise RuntimeError(f"Unknown diarization provider in manifest: {diarization_provider}")
    asr_models, vad_path = _stage_asr(
        job, manifest, fingerprint, resume=resume, ffmpeg_threads=ffmpeg_threads
    )
    diar_fp = _stage_diarization(
        job, manifest, fingerprint, vad_path, resume=resume, cpu_threads=cpu_threads
    )
    diar_data = read_json(job / "diarization" / "diarization.json")
    align_fp = {**diar_fp, "word_timestamp_schema": ASR_OUTPUT_SCHEMA_VERSION}
    _stage_align(
        job, manifest, asr_models, diar_data, align_fp, diarization_provider, resume=resume
    )
    report_step("[5/5] Rendering Markdown and JSON artifacts")
    render_all(job)
    if not config["keep_working_audio"]:
        shutil.rmtree(job / "work", ignore_errors=True)
        # The per-chunk JSON checkpoints stay: they are small and are what resume
        # reads. Only the sliced WAVs go — 184 MB on a 35-minute meeting with three
        # models.
        for chunk_wav in (job / "asr").glob("*/chunks/*/*.wav"):
            chunk_wav.unlink(missing_ok=True)


def _renderable(mode: str, utterances: list[Utterance]) -> bool:
    return mode != "readable" or all(item.readable_text for item in utterances)


def render_all(job: Path, requested: list[str] | None = None) -> None:
    """Write one file per requested mode; a mode with no text simply gets no file.

    raw and light each get one file per ASR model (transcript.<mode>.<model>.md), so
    every model's output stays line-by-line comparable with the canonical one;
    readable is always a single merged file with no model suffix. With one model
    there is no variant/model-b(/c) file.
    """
    manifest = read_json(job / "manifest.json")
    utterances = _load_utterances(job)
    asr_models = manifest_asr_models(manifest)
    canonical_model, variant_models = asr_models[0], asr_models[1:]
    for mode in requested or manifest["config"]["clean"]:
        if not _renderable(mode, utterances):
            continue
        if mode == "readable":
            render_markdown(
                job, manifest, utterances, mode, text_attr="readable_text", fallback=True
            )
            continue
        render_markdown(
            job,
            manifest,
            utterances,
            mode,
            text_attr=f"{mode}_text",
            fallback=True,
            suffix=canonical_model,
        )
        for variant_model in variant_models:
            render_markdown(
                job,
                manifest,
                utterances,
                mode,
                fallback=False,
                suffix=variant_model,
                variant_model=variant_model,
            )


def run_readable(job: Path, model: str | None = None) -> None:
    manifest = read_json(job / "manifest.json")
    utterances = _load_utterances(job)
    order_warning = uncalibrated_fusion_order(manifest_asr_models(manifest))
    if order_warning:
        report_step(order_warning)
    protected = {term["canonical"] for term in manifest["glossary"]}
    result = clean_readable(
        utterances,
        model or manifest["config"]["llm_model"],
        protected,
        manifest["config"]["context_prompt"],
        status=report_step,
        variant_models=manifest_asr_models(manifest)[1:],
    )
    for item in utterances:
        item.readable_text = result.texts[item.utterance_id]
    _save_canonical(job, manifest, utterances, read_json(job / "canonical.json")["diarization"])
    render_all(job, ["readable"])
    _, totals = write_readable_changes(job)
    # Concerns carry the words rejected as absent from both variants, and the
    # utterances that fell back to local text because of them. Without persisting
    # them a degraded readable would be indistinguishable from a clean one.
    # changed_utterances/needs_attention feed result.json's readable block (plan
    # section 4) without result.py having to re-parse review/readable_changes.csv.
    manifest["readable"] = {
        "model": model or manifest["config"]["llm_model"],
        "usage": result.usage,
        "status": "complete",
        "concerns": result.concerns,
        "fallback_utterances": sum(
            1 for note in result.concerns if note.get("note", "").startswith(CONCERN_FALLBACK)
        ),
        "repaired_utterances": sum(
            1 for note in result.concerns if note.get("note", "").startswith(CONCERN_REPAIR)
        ),
        "changed_utterances": totals.get("utterances", 0),
        "needs_attention": totals.get("needs_attention", 0),
    }
    atomic_json(job / "manifest.json", manifest)
    if totals.get("utterances"):
        kinds = ", ".join(
            f"{kind} {totals[kind]}"
            for kind in ("transliteration", "repair", "substitution", "loss")
            if totals.get(kind)
        )
        changed_words = sum(totals[k] for k in totals if k not in ("utterances", "needs_attention"))
        report_step(
            f"readable: changed {changed_words} words across {totals['utterances']} utterances "
            f"({kinds}); {totals['needs_attention']} need attention — review/readable_changes.csv"
        )


def try_readable(job: Path, model: str | None = None) -> bool:
    """Run the optional LLM pass without ever failing an already finished transcript.

    raw and light are rendered before this point. A missing key, an unreachable
    endpoint or a response rejected by local validation must leave them untouched,
    so the failure is recorded in the manifest and reported, not raised.
    """
    try:
        run_readable(job, model)
    # Deliberately broad: the OpenAI SDK raises its own hierarchy for auth, transport
    # and rate limits, and none of it may destroy a finished local transcript.
    except Exception as exc:
        manifest = read_json(job / "manifest.json")
        manifest["readable"] = {
            "model": model or manifest["config"]["llm_model"],
            "status": "failed",
            "error": str(exc),
        }
        atomic_json(job / "manifest.json", manifest)
        report_step(f"readable skipped: {exc}")
        report_step("raw and light are ready locally and unchanged.")
        return False
    return True


def rename(job: Path, replacements: dict[str, str]) -> None:
    path = job / "speakers.yaml"
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("speakers", {})
    names.update(replacements)
    atomic_text(path, yaml.safe_dump({"speakers": names}, allow_unicode=True, sort_keys=True))
    utterances = _load_utterances(job)
    for item in utterances:
        item.speaker_name = names.get(item.speaker_id, item.speaker_id)
    canonical = read_json(job / "canonical.json")
    _save_canonical(job, read_json(job / "manifest.json"), utterances, canonical["diarization"])
    render_all(job)
    # rename must refresh result.json/summary.md itself, since both
    # the `rename` CLI command and `run --speaker-names` call this function directly.
    write_job_result(job)
