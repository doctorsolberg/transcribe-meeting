from __future__ import annotations

import datetime as dt
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
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
    deepgram_request_parameters,
    deepgram_response_fingerprint,
    deepgram_words,
    deepgram_words_for_source,
    groq_request_fingerprint,
    local_vad_parameters,
    read_deepgram_response,
    transcribe_deepgram,
    transcribe_part,
    validate_word_timestamps,
    words_from_asr,
)
from .audio import (
    COMPRESSION_4_TO_1,
    PRE_EMPHASIS_COEFFICIENT,
    AudioError,
    concat_pcm,
    decode_pcm,
    probe,
    wav_duration,
)
from .cleaning import glossary_replace, light_clean
from .diarization import (
    DEEPGRAM_REPO,
    DIARIZATION_PROVIDERS,
    FOXNOSE_REPO,
    PYANNOTE_REPO,
    detect_overlap_regions,
    diarize,
    missing_community1_files,
)
from .job import (
    _LOCK_MIGRATION_WARNED,  # noqa: F401 - re-exported, tests clear() it via core._LOCK_MIGRATION_WARNED
    CANONICAL_SCHEMA_VERSION,  # noqa: F401 - re-exported, tests use core.CANONICAL_SCHEMA_VERSION
    _model_path,
    _warned_legacy_canonical,  # noqa: F401 - re-exported, tests discard() it via core._warned_legacy_canonical
    complete_stage,
    has_published_transcript,
    invalidate_stage,
    manifest_asr_models,
    mark_published,
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
    ReadableResult,
    clean_readable,
    uncalibrated_fusion_order,
)
from .readable_publish import job_transaction, publish_readable, publish_transcript_edit
from .render import render_markdown
from .result import build_job_result, render_summary, write_job_result
from .review import (
    CHANGE_COLUMNS,  # noqa: F401 - re-exported, historical core.CHANGE_COLUMNS
    prepare_review,  # noqa: F401 - re-exported, cli.py uses core.prepare_review
    refine,  # noqa: F401 - re-exported as core.refine (cli.py: `refine as refine_review`)
    write_readable_changes,
)
from .timeline import GAP_SECONDS, build_sources, mapping
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


def _vad_model_path(info: dict[str, Any]) -> Path:
    if info.get("path"):
        return _model_path(info, "huggingface")
    repo = models_dir() / "huggingface" / "models--pyannote--segmentation-3.0"
    ref = repo / "refs" / "main"
    if ref.is_file():
        try:
            revision = ref.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise RuntimeError("Cannot read the VAD cache reference. Rerun setup.") from None
        if revision and Path(revision).name == revision and revision not in {".", ".."}:
            return repo / "snapshots" / revision
    return repo / "unresolved"


def route_readiness(asr_models: list[str], diarization_provider: str) -> dict[str, Any]:
    """Check whether every local checkpoint and API key a route needs is already on
    disk or in the environment — no network call, no model load.

    Shared by `run`'s preflight and `doctor`, so
    both agree on the same definition of "ready": `run` turns a not-ready route into
    a BadParameter before create_job even runs, and `doctor` reports it and exits 1.
    Local checkpoint paths are resolved exactly like the stages that actually load
    them. Presence checks do not prove the models can load or run.
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
        vad_model = _vad_model_path(lock_models.get("vad", {}))
        vad_ready = vad_model.is_file() or (vad_model / "pytorch_model.bin").is_file()
        if not vad_ready:
            problems.append(
                "Local VAD model (segmentation-3.0) not found. Accept the terms on "
                "Hugging Face, then run: transcribe-meeting setup"
            )

    diarization_ready = True
    diarization_key: dict[str, bool] | None = None
    if diarization_provider == "pyannote":
        model_dir = _model_path(lock_models.get("pyannote", {}), "speaker-diarization-community-1")
        diarization_ready = not missing_community1_files(model_dir)
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


def _source_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else data_root() / path).resolve()


def _pcm_fingerprints(job: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        diagnostics_path = job / "diagnostics.json"
        diagnostics = read_json(diagnostics_path)
        config = manifest["config"]
        requested_stream = config.get("audio_stream")
        preprocess, compression = config["asr_preprocess"], config["asr_compression"]
        fingerprints = []
        migrated = False
        for source, diagnostic in zip(manifest["sources"], diagnostics["parts"], strict=True):
            path = _source_path(source["path"])
            diagnostic_path = Path(diagnostic["path"])
            stream = diagnostic["stream_index"]
            if (
                type(stream) is not int
                or stream < 0
                or (requested_stream is not None and stream != requested_stream)
            ):
                raise ValueError("Invalid selected stream in diagnostics.json.")
            if _source_path(str(diagnostic_path)) != path:
                if diagnostic_path.is_absolute():
                    raise ValueError(
                        "The source order in diagnostics.json does not match the manifest."
                    )
                # Compare the known suffix after any parent traversal. A relative
                # symlink alias can still be verified if it resolves in this process.
                suffix = diagnostic_path.parts
                for part_index, component in enumerate(diagnostic_path.parts):
                    if component == "..":
                        suffix = diagnostic_path.parts[part_index + 1 :]
                if diagnostic_path.resolve() != path and (
                    not suffix or path.parts[-len(suffix) :] != suffix
                ):
                    raise ValueError("The legacy diagnostics path does not match its source row.")
                # Older probes persisted paths relative to the invocation directory,
                # which was not recorded. Trust row order only after verifying the
                # original file digest and independently confirming its selected stream.
                if sha256(path) != source["sha256"]:
                    raise ValueError(f"The source audio has changed: {path}")
                verified = probe(path, requested_stream)
                if any(
                    key in diagnostic and diagnostic[key] != verified[key]
                    for key in ("stream_index", "duration", "codec", "channels", "sample_rate")
                ):
                    raise ValueError(f"Cannot verify the legacy diagnostics stream for {path}.")
                diagnostic["path"] = str(path)
                migrated = True
            fingerprints.append(
                {
                    "pcm_version": 1,
                    "source_sha256": source["sha256"],
                    "stream": stream,
                    "preprocess": preprocess,
                    "pre_emphasis": PRE_EMPHASIS_COEFFICIENT if preprocess != "none" else None,
                    "compression": compression,
                    "compression_parameters": COMPRESSION_4_TO_1 if compression == "4:1" else None,
                }
            )
        if migrated:
            atomic_json(diagnostics_path, diagnostics)
        return fingerprints
    except (AudioError, OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot verify the source audio or its selected stream from the job metadata. "
            "Restore the manifest and diagnostics.json or create a new job with run."
        ) from exc


def _working_audio(
    job: Path,
    manifest: dict[str, Any],
    ffmpeg_threads: int,
    decode_workers: int,
    *,
    fingerprints: list[dict[str, Any]] | None = None,
    resume: bool = True,
) -> list[Path]:
    fingerprints = _pcm_fingerprints(job, manifest) if fingerprints is None else fingerprints
    destination = job / "work" / "pcm"
    result: list[Path] = []
    pending: list[tuple[int, Path, Path, dict[str, Any]]] = []
    reports: dict[int, dict[str, Any]] = {}
    for index, (source, fp) in enumerate(zip(manifest["sources"], fingerprints, strict=True), 1):
        target = destination / f"{index:03d}.wav"
        result.append(target)
        if not resume or not stage_valid(job, f"audio/{index:03d}", fp, target):
            pending.append((index, _source_path(source["path"]), target, fp))
        else:
            reports[index] = {
                "mode": fp["preprocess"],
                "compression": fp["compression"],
                "reused": True,
            }
    if pending:
        report_step(
            f"[1/5] Preparing {len(pending)} part(s) with "
            f"{min(decode_workers, len(pending))} FFmpeg thread(s)"
        )

        def decode_one(item: tuple[int, Path, Path, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            index, source, target, fp = item
            try:
                actual_hash = sha256(source)
            except OSError as exc:
                raise RuntimeError(f"Cannot verify source audio {source}: {exc}") from exc
            if actual_hash != fp["source_sha256"]:
                raise RuntimeError(
                    f"The source audio has changed: {source}. Restore the original or create "
                    "a new job with run."
                )
            marker = f"audio/{index:03d}"
            invalidate_stage(job, marker)
            report = decode_pcm(
                source, target, fp["stream"], fp["preprocess"], fp["compression"], ffmpeg_threads
            )
            complete_stage(job, marker, fp, target)
            return index, report

        with ThreadPoolExecutor(max_workers=min(decode_workers, len(pending))) as executor:
            futures = {executor.submit(decode_one, item): item[0] for item in pending}
            for future in as_completed(futures):
                index, report = future.result()
                reports[index] = report
                report_step(f"Prepared part {index}/{len(manifest['sources'])}")
    diagnostics_path = job / "diagnostics.json"
    diagnostics_data = read_json(diagnostics_path)
    diagnostics_data["asr_and_diarization_preprocessing"] = [
        {"source_file": source["path"], **reports[index]}
        for index, source in enumerate(manifest["sources"], 1)
    ]
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
    job: Path,
    manifest: dict[str, Any],
    model_name: str,
    model_index: int,
    model_count: int,
    *,
    data: dict[str, Any] | None = None,
    part_indices: set[int] | None = None,
) -> None:
    """scope="meeting" ASR: one request on work/meeting.wav,
    split into the usual per-part `asr/<model>/NNN.json` files afterwards.

    The pipeline supplies the verified response and the indices that need rebuilding.
    """
    sources = manifest["sources"]
    missing = [
        (index, source)
        for index, source in enumerate(sources, 1)
        if (
            index in part_indices
            if part_indices is not None
            else not (job / "asr" / model_name / f"{index:03d}.json").exists()
        )
    ]
    if not missing:
        return
    if data is None:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError(f"ASR model {model_name} needs DEEPGRAM_API_KEY.")
        report_step(
            f"[2/5] Transcribing the whole meeting in Deepgram ({model_name}, "
            f"model {model_index}/{model_count})"
        )
        data = transcribe_deepgram(
            job / "work" / "meeting.wav",
            api_key,
            manifest["config"]["language"],
            job / "asr" / model_name / "meeting.json",
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


_DEEPGRAM_REQUEST_STAGE = "asr/deepgram-request"


def _deepgram_fingerprint(meeting_fp: dict[str, Any], language: str) -> dict[str, Any]:
    return {"meeting": meeting_fp, "parameters": deepgram_request_parameters(language)}


def _deepgram_checkpoint(
    job: Path, config: dict[str, Any], meeting_fp: dict[str, Any]
) -> dict[str, Any]:
    """ASR and diarization share one verified raw response, not just a path."""
    model = DEEPGRAM_ASR_MODEL_NAME
    target = job / "asr" / model / "meeting.json"
    language = config["language"]
    fp = _deepgram_fingerprint(meeting_fp, language)
    request_fp = deepgram_response_fingerprint(meeting_fp["pcm_sha256"], language)
    if (data := read_deepgram_response(target, request_fp)) is not None:
        if not stage_valid(job, _DEEPGRAM_REQUEST_STAGE, fp, target):
            complete_stage(job, _DEEPGRAM_REQUEST_STAGE, fp, target)
        return data
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("A missing or stale Deepgram checkpoint needs DEEPGRAM_API_KEY.")
    invalidate_stage(job, _DEEPGRAM_REQUEST_STAGE)
    # A new response can contain different words and speaker IDs even when the
    # request is identical. Rebuild any existing derivatives from that response.
    for marker in (job / ".stages" / "asr" / model).glob("*.json"):
        marker.unlink()
    invalidate_stage(job, "asr")
    # The diarization marker may belong to an earlier invocation that used Deepgram,
    # even if this invocation selected another diarizer.
    previous_diarization = job / "diarization" / "diarization.json"
    if previous_diarization.is_file():
        try:
            if read_json(previous_diarization).get("provider") == "deepgram":
                invalidate_stage(job, "diarization")
        except (AttributeError, ValueError):
            invalidate_stage(job, "diarization")
    if target.is_file():
        report_step("Deepgram meeting checkpoint could not be verified; requesting it again.")
    target.unlink(missing_ok=True)
    report_step("Transcribing the whole meeting in Deepgram")
    data = transcribe_deepgram(job / "work" / "meeting.wav", api_key, language, target)
    deepgram_words(data)
    complete_stage(job, _DEEPGRAM_REQUEST_STAGE, fp, target)
    return data


def _stage_audio(
    job: Path,
    manifest: dict[str, Any],
    *,
    resume: bool,
    ffmpeg_threads: int,
    decode_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepare_fps = _pcm_fingerprints(job, manifest)
    parts = [job / "work" / "pcm" / f"{index:03d}.wav" for index in range(1, len(prepare_fps) + 1)]
    if not resume or not all(
        stage_valid(job, f"audio/{index:03d}", fp, part)
        for index, (fp, part) in enumerate(zip(prepare_fps, parts, strict=True), 1)
    ):
        parts = _working_audio(
            job, manifest, ffmpeg_threads, decode_workers, fingerprints=prepare_fps, resume=resume
        )
    # The recipes above identify how to prepare audio; decisions derived from
    # it must also identify the bytes actually decoded on this installation.
    pcm_fps = [
        {**fp, "pcm_sha256": _audio_stage_digest(job, f"audio/{index:03d}", part)}
        for index, (fp, part) in enumerate(zip(prepare_fps, parts, strict=True), 1)
    ]
    concat_fp = {"concat_version": 1, "pcm": pcm_fps, "gap_seconds": GAP_SECONDS}
    meeting = job / "work" / "meeting.wav"
    if not resume or not stage_valid(job, "audio", concat_fp, meeting):
        invalidate_stage(job, "audio")
        report_step("Assembling a single technical track for diarization")
        concat_pcm(parts, meeting, ffmpeg_threads)
        complete_stage(job, "audio", concat_fp, meeting)
    else:
        report_step("[1/5] Prepared audio is reused from checkpoint")
    return pcm_fps, {**concat_fp, "pcm_sha256": _audio_stage_digest(job, "audio", meeting)}


def _audio_stage_digest(job: Path, stage: str, product: Path) -> str:
    """Only use this value after checking or producing the matching checkpoint."""
    marker = read_json(job / ".stages" / f"{stage}.json")
    return marker["product_hashes"][str(product.relative_to(job))]


def _refresh_decoded_timeline(job: Path, manifest: dict[str, Any]) -> None:
    """Set both clocks from decoded part lengths, before ASR and attribution."""
    decoded = mapping(
        build_sources(
            [
                (
                    source["path"],
                    source["sha256"],
                    wav_duration(job / "work" / "pcm" / f"{index:03d}.wav"),
                )
                for index, source in enumerate(manifest["sources"], 1)
            ]
        )
    )
    if decoded != manifest["sources"]:
        manifest["sources"] = decoded
        atomic_json(job / "manifest.json", manifest)


def _stage_asr(
    job: Path,
    manifest: dict[str, Any],
    pcm_fps: list[dict[str, Any]],
    meeting_fp: dict[str, Any],
    models: dict[str, Any],
    *,
    resume: bool,
    ffmpeg_threads: int,
) -> tuple[list[str], Path, dict[str, Any]]:
    """Reuse each model/part independently; the aggregate marker is for status."""
    config = manifest["config"]
    asr_models = manifest_asr_models(manifest)
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
    gigaam_info = models.get("gigaam", {})
    vad_info = models.get("vad", {})
    gigaam_path = _model_path(gigaam_info, "gigaam")
    vad_path = _vad_model_path(vad_info)
    vad_fp = _model_fingerprint(vad_info, vad_path)
    prompt = make_asr_prompt(manifest["glossary"], config["context_prompt"])
    chunk_minutes = config["chunk_minutes"] or 0
    groq_fp = groq_request_fingerprint(config["language"], prompt, chunk_minutes)
    groq_limiter = GroqFreeTierLimiter(status=report_step)
    asr_metadata = {
        "provider": config["asr_provider"],
        "backend": config["asr_backend"],
        "models": asr_models,
        "canonical_model": asr_models[0],
    }
    if effective_device is not None:
        asr_metadata.update(requested_device=requested_device, effective_device=effective_device)
    for model_name in asr_models:
        if "api_model" in ASR_MODELS[model_name]:
            asr_metadata["api_model"] = ASR_MODELS[model_name]["api_model"]
    if manifest.get("asr") != asr_metadata:
        manifest["asr"] = asr_metadata
        atomic_json(job / "manifest.json", manifest)
    parts: dict[str, list[dict[str, Any]]] = {}
    stage_changed = not resume
    for model_index, model_name in enumerate(asr_models, 1):
        model_info = ASR_MODELS[model_name]
        provider = model_info["provider"]
        if provider == "local":
            model_lock_entry = gigaam_info.get("models", {}).get(model_name, {})
            settings = {
                "gigaam_model": model_name,
                "gigaam_model_hash": model_lock_entry.get("sha256")
                or model_lock_entry.get("md5")
                or model_info["md5"],
                "gigaam_commit": gigaam_info.get("commit", GIGAAM_COMMIT),
                "vad_model": vad_fp,
                "vad_parameters": local_vad_parameters(),
                "device": effective_device,
            }
        elif provider == "groq":
            settings = groq_fp
        else:
            settings = {"deepgram": _deepgram_fingerprint(meeting_fp, config["language"])}
        model_fps = []
        pending = []
        for index, (source, pcm_fp) in enumerate(zip(manifest["sources"], pcm_fps, strict=True), 1):
            fp = {"part_version": ASR_OUTPUT_SCHEMA_VERSION, "pcm": pcm_fp, **settings}
            if model_info.get("scope") == "meeting":
                fp["clip"] = [source["diarization_offset"], source["duration"]]
                fp["deepgram_slice_version"] = 2
            model_fps.append(fp)
            path = job / "asr" / model_name / f"{index:03d}.json"
            marker = f"asr/{model_name}/{index:03d}"
            if not resume or not stage_valid(job, marker, fp, path):
                pending.append((index, path, marker, fp))
        parts[model_name] = model_fps
        if not pending:
            report_step(f"[2/5] ASR for {model_name} is reused from checkpoints")
            continue
        stage_changed = True
        invalidate_stage(job, "asr")
        if model_info.get("scope") == "meeting":
            data = _deepgram_checkpoint(job, config, meeting_fp)
            # Refresh the pending set: a newly obtained raw response invalidates
            # every previous slice, not just a missing one.
            pending = []
            for index, fp in enumerate(model_fps, 1):
                path = job / "asr" / model_name / f"{index:03d}.json"
                marker = f"asr/{model_name}/{index:03d}"
                if not resume or not stage_valid(job, marker, fp, path):
                    invalidate_stage(job, marker)
                    pending.append((index, path, marker, fp))
            _run_deepgram_asr(
                job,
                manifest,
                model_name,
                model_index,
                len(asr_models),
                data=data,
                part_indices={index for index, *_ in pending},
            )
            for _, path, marker, fp in pending:
                complete_stage(job, marker, fp, path)
            continue
        # Keep at most one GigaAM model in memory (up to 3.1 GB each).
        local_backend = None
        if provider == "local":
            assert effective_device is not None
            local_backend = GigaAMBackend(gigaam_path, effective_device, model_name)
        try:
            for index, destination, marker, fp in pending:
                report_step(
                    f"[2/5] Transcribing part {index}/{len(pcm_fps)} "
                    f"({model_name}, model {model_index}/{len(asr_models)})"
                )
                invalidate_stage(job, marker)
                transcribe_part(
                    job / "work" / "pcm" / f"{index:03d}.wav",
                    destination,
                    gigaam_path,
                    config["language"],
                    prompt,
                    provider=provider,
                    chunk_minutes=chunk_minutes,
                    ffmpeg_threads=ffmpeg_threads,
                    groq_limiter=groq_limiter,
                    local_backend=local_backend,
                    vad_model_path=vad_path,
                    local_fingerprint=fp if provider == "local" else None,
                    model_name=model_name,
                    pcm_fingerprint=pcm_fps[index - 1],
                    resume=resume,
                )
                complete_stage(job, marker, fp, destination)
        finally:
            if local_backend is not None:
                local_backend.close()
    asr_fp = {"asr_models": asr_models, "parts": parts}
    if stage_changed or not stage_valid(job, "asr", asr_fp):
        complete_stage(job, "asr", asr_fp)
    return asr_models, vad_path, asr_fp


def _model_fingerprint(info: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "sha256": info.get("sha256"),
        "revision": info.get("revision"),
        "snapshot": path.name if path.parent.name == "snapshots" else None,
        "path": str(path) if not (info.get("sha256") or info.get("revision")) else None,
    }


def _stage_diarization(
    job: Path,
    manifest: dict[str, Any],
    meeting_fp: dict[str, Any],
    vad_path: Path,
    models: dict[str, Any],
    *,
    resume: bool,
    cpu_threads: int,
) -> dict[str, Any]:
    """Run diarization plus overlap detection. Returns diar_fp, needed to derive the
    align stage's fingerprint."""
    config = manifest["config"]
    diarization_provider = config["diarization_provider"]
    pyannote_path = _model_path(models.get("pyannote", {}), "speaker-diarization-community-1")
    wespeaker_path = _model_path(models.get("wespeaker", {}), "wespeaker/en/model.onnx")
    diar_fp = {
        "diarization_version": 1,
        "meeting": meeting_fp,
        "provider": diarization_provider,
        "overlap_model": _model_fingerprint(models.get("vad", {}), vad_path),
    }
    if diarization_provider == "foxnose":
        diar_fp["wespeaker"] = _model_fingerprint(models.get("wespeaker", {}), wespeaker_path)
        diar_fp["diarize_version"] = models.get("diarize", {}).get("version")
    elif diarization_provider == "pyannote":
        diar_fp["pyannote"] = _model_fingerprint(models.get("pyannote", {}), pyannote_path)
    else:
        diar_fp["deepgram"] = _deepgram_fingerprint(meeting_fp, config["language"])
    if diarization_provider != "deepgram":
        diar_fp["speaker_parameters"] = [
            config["speakers"],
            config["min_speakers"],
            config["max_speakers"],
        ]
    diar_file = job / "diarization" / "diarization.json"
    rttm = job / "diarization" / "diarization.rttm"
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
    if not resume or not stage_valid(job, "diarization", diar_fp, diar_file, rttm):
        invalidate_stage(job, "diarization")
        deepgram_response = None
        if diarization_provider == "deepgram":
            deepgram_response = _deepgram_checkpoint(job, config, meeting_fp)
        where = "CPU" if diarization_provider != "deepgram" else "Deepgram response"
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
            rttm,
            pyannote_path,
            config["speakers"],
            config["min_speakers"],
            config["max_speakers"],
            cpu_threads,
            provider=diarization_provider,
            wespeaker_model=wespeaker_path,
            deepgram_response=deepgram_response,
        )
        # Overlap detection runs once on the technical track regardless of which
        # diarization provider ran above: it uses segmentation-3.0 (the VAD model
        # local ASR already requires), never foxnose's or pyannote's own output.
        report_step("[3/5] Detecting overlapping speech (segmentation-3.0)")
        detect_and_record_overlap(job / "work" / "meeting.wav", diar_file, vad_path, cpu_threads)
        complete_stage(job, "diarization", diar_fp, diar_file, rttm)
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


def _carry_readable(job: Path, manifest: dict[str, Any], utterances: list[Utterance]) -> bool:
    if not (manifest.get("readable") or (job / "transcript.readable.md").is_file()):
        return False

    def inputs(items: list[Utterance]) -> list[tuple[Any, ...]]:
        return [
            (item.utterance_id, item.speaker_id, item.raw_text, item.variants) for item in items
        ]

    try:
        previous = _load_utterances(job)
    except (FileNotFoundError, ValueError, TypeError, KeyError):
        previous = None
    if previous is not None and inputs(previous) == inputs(utterances):
        for old, new in zip(previous, utterances, strict=True):
            new.readable_text = old.readable_text
        if (manifest.get("readable") or {}).get("status") != "complete" or (
            utterances and all(item.readable_text for item in utterances)
        ):
            return False
    return True


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
    if not resume or not stage_valid(job, "align", align_fp, job / "canonical.json"):
        invalidate_stage(job, "align")
        report_step("[4/5] Attributing words to speakers and building utterances")
        ids = sorted({item["speaker"] for item in diar_data["exclusive"]})
        _reconcile_speaker_count(diarization_provider, ids, config["speakers"], diar_data)
        canonical_model = asr_models[0]
        use_native_speakers = (
            canonical_model == DEEPGRAM_ASR_MODEL_NAME and diarization_provider == "deepgram"
        )
        aligned_parts: list[tuple[dict[str, Any], list[Word]]] = []
        for index, source in enumerate(manifest["sources"], 1):
            words = [
                Word(**word)
                for word in words_from_asr(
                    read_json(job / "asr" / canonical_model / f"{index:03d}.json"),
                    native_speakers=use_native_speakers,
                )
            ]
            # Convert exclusive annotation from technical clock to local source clock.
            base = source["diarization_offset"]
            local = [
                (x["start"] - base, x["end"] - base, x["speaker"])
                for x in diar_data["exclusive"]
                if x["end"] > base and x["start"] < base + source["duration"]
            ]
            assign_words(
                [word for word in words if word.speaker_id == UNKNOWN_SPEAKER]
                if use_native_speakers
                else words,
                local,
            )
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
        outdated_readable = _carry_readable(job, manifest, utterances)
        _save_canonical(job, manifest, utterances, diar_data, previews)
        if outdated_readable:
            # Keep the last readable exports if writing their replacement fails.
            if manifest.pop("readable", None) is not None:
                atomic_json(job / "manifest.json", manifest)
            (job / "transcript.readable.md").unlink(missing_ok=True)
            (job / "review" / "readable_changes.csv").unlink(missing_ok=True)
        complete_stage(job, "align", align_fp)
    else:
        if not (job / "speakers.yaml").is_file():
            saved = {item.speaker_id: item.speaker_name for item in _load_utterances(job)}
            ids = {item["speaker"] for item in diar_data["exclusive"]}
            names = {uid: saved.get(uid, uid) for uid in sorted(ids)}
            atomic_text(
                job / "speakers.yaml",
                yaml.safe_dump({"speakers": names}, allow_unicode=True, sort_keys=True),
            )
        report_step("[4/5] Attribution and utterances are reused from checkpoint")


def run_job(
    job: Path,
    *,
    resume: bool = True,
    cpu_threads_override: int | None = None,
    ffmpeg_threads_override: int | None = None,
    preserve_published: bool = False,
) -> None:
    with job_transaction(job):
        if resume and preserve_published and has_published_transcript(job):
            report_step("Rendering the published transcript without reprocessing audio.")
            render_all(job)
            mark_published(job)
            return
        _run_pipeline(job, resume, cpu_threads_override, ffmpeg_threads_override)


def _run_pipeline(
    job: Path,
    resume: bool,
    cpu_threads_override: int | None,
    ffmpeg_threads_override: int | None,
) -> None:
    manifest = read_json(job / "manifest.json")
    config = manifest["config"]
    # A finished job's published canonical transcript is not an inference cache.
    # Old checkpoints cannot prove provenance; keep an accepted document (including
    # manual edits) readable rather than rebuilding over it or paying for ASR again.
    if resume and (job / "result.json").is_file() and (job / "canonical.json").is_file():
        try:
            previous_alignment = read_json(job / ".stages" / "align.json").get("fingerprint")
        except (AttributeError, FileNotFoundError, ValueError):
            previous_alignment = None
        if isinstance(previous_alignment, dict) and "word_timestamp_schema" in previous_alignment:
            report_step(
                "Rendering the existing transcript from this completed legacy job; its "
                "checkpoints are not reusable. To reprocess the audio, create a new job with run."
            )
            render_all(job)
            return
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
    models = model_lock()["models"]
    pcm_fps, meeting_fp = _stage_audio(
        job,
        manifest,
        resume=resume,
        ffmpeg_threads=ffmpeg_threads,
        decode_workers=profile["decode_workers"],
    )
    _refresh_decoded_timeline(job, manifest)
    diarization_provider = config["diarization_provider"]
    if diarization_provider not in DIARIZATION_PROVIDERS:
        raise RuntimeError(f"Unknown diarization provider in manifest: {diarization_provider}")
    if not resume:
        invalidate_stage(job, _DEEPGRAM_REQUEST_STAGE)
    asr_models, vad_path, asr_fp = _stage_asr(
        job, manifest, pcm_fps, meeting_fp, models, resume=resume, ffmpeg_threads=ffmpeg_threads
    )
    diar_fp = _stage_diarization(
        job, manifest, meeting_fp, vad_path, models, resume=resume, cpu_threads=cpu_threads
    )
    if not stage_valid(job, "asr", asr_fp):
        # Diarization had to fetch a new shared Deepgram response. Its speaker
        # IDs must not be paired with slices from the former response.
        asr_models, vad_path, asr_fp = _stage_asr(
            job, manifest, pcm_fps, meeting_fp, models, resume=True, ffmpeg_threads=ffmpeg_threads
        )
    diar_data = read_json(job / "diarization" / "diarization.json")
    align_fp = {
        "align_version": 1,
        "asr": asr_fp,
        "diarization": diar_fp,
        "timeline": manifest["sources"],
        "glossary": manifest["glossary"],
        "expected_speakers": config["speakers"],
        "asr_products": [
            sha256(job / "asr" / model / f"{index:03d}.json")
            for model in asr_models
            for index in range(1, len(pcm_fps) + 1)
        ],
        "diarization_product": sha256(job / "diarization" / "diarization.json"),
    }
    _stage_align(
        job, manifest, asr_models, diar_data, align_fp, diarization_provider, resume=resume
    )
    report_step("[5/5] Rendering Markdown and JSON artifacts")
    render_all(job)
    mark_published(job)
    if not config["keep_working_audio"]:
        shutil.rmtree(job / "work", ignore_errors=True)
        # The per-chunk JSON checkpoints stay: they are small and are what resume
        # reads. Only the sliced WAVs go — 184 MB on a 35-minute meeting with three
        # models.
        for chunk_wav in (job / "asr").glob("*/chunks/*/*.wav"):
            chunk_wav.unlink(missing_ok=True)


def _renderable(mode: str, utterances: list[Utterance]) -> bool:
    return mode != "readable" or (
        bool(utterances) and all(item.readable_text for item in utterances)
    )


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
    modes = list(manifest["config"]["clean"] if requested is None else requested)
    if requested is None and "readable" in modes:
        modes = list(dict.fromkeys(["raw", "light", *modes]))
    for mode in modes:
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


def restore_readable_exports(job: Path) -> bool:
    """Restore missing exports from the accepted canonical transcript without an LLM call."""
    with job_transaction(job):
        if not _renderable("readable", _load_utterances(job)):
            return False
        transcript = job / "transcript.readable.md"
        audit = job / "review" / "readable_changes.csv"
        if not transcript.is_file() or not transcript.stat().st_size:
            render_all(job, ["readable"])
        if not audit.is_file() or not audit.stat().st_size:
            write_readable_changes(job)
    return True


def run_readable(job: Path, model: str | None = None) -> None:
    with job_transaction(job):
        manifest = read_json(job / "manifest.json")
        utterances = _load_utterances(job)
        canonical = read_json(job / "canonical.json")
        if not utterances:
            manifest["readable"] = {"status": "no speech"}
            atomic_json(job / "manifest.json", manifest)
            report_step("readable skipped: no speech detected.")
            return
        digests = (sha256(job / "canonical.json"), sha256(job / "manifest.json"))
    order_warning = uncalibrated_fusion_order(manifest_asr_models(manifest))
    if order_warning:
        report_step(order_warning)
    protected = {term["canonical"] for term in manifest.get("glossary", [])}
    manual = set(manifest.get("manual_text_edits", []))
    candidates = [item for item in utterances if item.utterance_id not in manual]
    result = (
        clean_readable(
            candidates,
            model or manifest["config"]["llm_model"],
            protected,
            manifest["config"].get("context_prompt", ""),
            status=report_step,
            variant_models=manifest_asr_models(manifest)[1:],
        )
        if candidates
        else ReadableResult(texts={}, concerns=[], usage={}, provider="manual")
    )
    previous = manifest.get("readable") or {}
    if (
        result.pending_utterance_ids
        and previous.get("status") == "complete"
        and _renderable("readable", utterances)
    ):
        reason = result.endpoint_failure or "One or more readable batches did not pass validation"
        raise RuntimeError(f"{reason}; the complete readable version was kept.")
    for item in utterances:
        item.readable_text = (
            item.raw_text if item.utterance_id in manual else result.texts[item.utterance_id]
        )
    with TemporaryDirectory(prefix=".readable-stage-", dir=job) as temporary:
        staged = Path(temporary)
        _save_canonical(
            staged,
            manifest,
            utterances,
            canonical["diarization"],
            canonical.get("speaker_previews", {}),
        )
        render_markdown(staged, manifest, utterances, "readable")
        _, totals = write_readable_changes(staged)
        _publish_readable_result(job, staged, manifest, model, result, totals, digests)


def _publish_readable_result(
    job: Path,
    staged: Path,
    manifest: dict[str, Any],
    model: str | None,
    result: ReadableResult,
    totals: dict[str, int],
    digests: tuple[str, str],
) -> None:
    fallback_count = result.fallback_utterances
    if fallback_count is None:
        fallback_count = sum(
            1 for note in result.concerns if note.get("note", "").startswith(CONCERN_FALLBACK)
        )
    # Concerns carry the words rejected as absent from both variants, and the
    # utterances that fell back to local text because of them. Without persisting
    # them a degraded readable would be indistinguishable from a clean one.
    # changed_utterances/needs_attention feed result.json's readable block (plan
    # section 4) without result.py having to re-parse review/readable_changes.csv.
    manifest["readable"] = {
        "model": model or manifest["config"]["llm_model"],
        "usage": result.usage,
        "status": "partial" if result.pending_utterance_ids else "complete",
        "concerns": result.concerns,
        "fallback_utterances": fallback_count,
        "accepted_utterances": result.accepted_utterances,
        "provider": result.provider,
        "api": result.api,
        "response_format": result.response_format,
        "repaired_utterances": result.repaired_utterances,
        "changed_utterances": totals.get("utterances", 0),
        "needs_attention": totals.get("needs_attention", 0),
    }
    if result.repaired_utterances is None:
        manifest["readable"]["repaired_utterances"] = sum(
            1 for note in result.concerns if note.get("note", "").startswith(CONCERN_REPAIR)
        )
    if result.endpoint_failure:
        manifest["readable"]["endpoint_failure"] = result.endpoint_failure
    if result.pending_utterance_ids:
        manifest["readable"]["pending_utterance_ids"] = result.pending_utterance_ids
    summary = build_job_result(job, manifest=manifest, staged=staged)
    atomic_json(staged / "result.json", summary)
    atomic_text(staged / "summary.md", render_summary(summary))
    mark_published(staged)
    publish_readable(job, staged, manifest, canonical_digest=digests[0], manifest_digest=digests[1])
    if result.pending_utterance_ids:
        report_step("Partial readable published with local fallback; rerun clean to retry.")
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
        error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        with job_transaction(job):
            manifest = read_json(job / "manifest.json")
            failure = {
                "model": model or manifest["config"]["llm_model"],
                "status": "failed",
                "error": error,
            }
            previous = manifest.get("readable") or {}
            if previous.get("status") in {"complete", "partial"}:
                try:
                    accepted = _renderable("readable", _load_utterances(job))
                except (OSError, ValueError, TypeError, KeyError):
                    accepted = False
            else:
                accepted = False
            manifest["readable"] = {**previous, "last_attempt": failure} if accepted else failure
            atomic_json(job / "manifest.json", manifest)
        report_step(f"readable failed: {error}")
        report_step("Existing transcripts were kept.")
        return False
    return True


def rename(job: Path, replacements: dict[str, str]) -> None:
    with job_transaction(job):
        _rename(job, replacements)


def _rename(job: Path, replacements: dict[str, str]) -> None:
    path = job / "speakers.yaml"
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("speakers", {})
    names.update(replacements)
    utterances = _load_utterances(job)
    for item in utterances:
        if item.speaker_id in replacements:
            item.speaker_name = replacements[item.speaker_id]
    save_transcript_edit(job, read_json(job / "manifest.json"), utterances, names)


def save_transcript_edit(
    job: Path,
    manifest: dict[str, Any],
    utterances: list[Utterance],
    names: dict[str, str] | None = None,
) -> None:
    """Save corrected text or speaker names while holding the job lock."""
    canonical = read_json(job / "canonical.json")
    with TemporaryDirectory(prefix=".transcript-edit-", dir=job) as temporary:
        staged = Path(temporary)
        _save_canonical(
            staged,
            manifest,
            utterances,
            canonical["diarization"],
            canonical.get("speaker_previews", {}),
        )
        mark_published(staged)
        if names is not None:
            atomic_text(
                staged / "speakers.yaml",
                yaml.safe_dump({"speakers": names}, allow_unicode=True, sort_keys=True),
            )
        if _renderable("readable", utterances):
            render_markdown(staged, manifest, utterances, "readable")
            _, totals = write_readable_changes(staged)
            if state := manifest.get("readable"):
                state.update(
                    changed_utterances=totals["utterances"],
                    needs_attention=totals["needs_attention"],
                )
        publish_transcript_edit(job, staged, manifest)
    # These derived files can be rebuilt by resume without touching accepted edits.
    render_all(job)
    write_job_result(job)
