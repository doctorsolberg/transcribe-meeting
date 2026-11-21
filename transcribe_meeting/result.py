"""Build result.json and summary.md: a job's machine- and human-readable outcome.

Kept separate from core.py so an agent driving the CLI only needs
what a finished job already wrote to disk — manifest.json, canonical.json,
speakers.yaml and the rendered transcripts — never the pipeline itself. Called at
the end of `run`, and again from `clean` and `rename` (core.rename) whenever either
one changes something these files summarize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .asr import ASR_API_KEY_ENV, ASR_MODELS
from .model import UNKNOWN_SPEAKER
from .util import atomic_json, atomic_text, fmt_time, read_json

_RAW_DESCRIPTION = "raw ASR text, one line per utterance, unsmoothed"
_LIGHT_DESCRIPTION = (
    "fillers and exact repeats removed; one speaker's turns merged into a block "
    "starting with a capital letter and ending with a period — punctuation inside the "
    "phrase is not restored"
)
_READABLE_DESCRIPTION = (
    "LLM cleanup for reading; changes text only where it looks like an ASR error"
)
_REVIEW_DESCRIPTION = (
    "what readable changed, one line per utterance; rows needing attention come first"
)


def _short_error(text: str, limit: int = 160) -> str:
    """Collapse a provider error to one readable line."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def build_job_result(
    job: Path, *, manifest: dict[str, Any] | None = None, staged: Path | None = None
) -> dict[str, Any]:
    manifest = read_json(job / "manifest.json") if manifest is None else manifest
    exports = job if staged is None else staged
    canonical_path = exports / "canonical.json"
    canonical = read_json(canonical_path) if canonical_path.is_file() else {}
    utterances: list[dict[str, Any]] = canonical.get("utterances", [])
    diarization_data: dict[str, Any] = canonical.get("diarization") or {}
    previews: dict[str, dict[str, Any]] = canonical.get("speaker_previews") or {}
    speakers_path = job / "speakers.yaml"
    names: dict[str, str] = (
        (yaml.safe_load(speakers_path.read_text(encoding="utf-8")) or {}).get("speakers", {})
        if speakers_path.exists()
        else {}
    )

    warnings: list[str] = []
    sources = manifest.get("sources", [])
    duration_seconds = round(sum(float(item.get("duration", 0.0)) for item in sources), 1)
    source_names = [Path(item["path"]).name for item in sources]

    speech_seconds: dict[str, float] = {}
    for item in utterances:
        speaker_id = item["speaker_id"]
        speech_seconds[speaker_id] = speech_seconds.get(speaker_id, 0.0) + (
            item["meeting_end"] - item["meeting_start"]
        )
    # UNKNOWN_SPEAKER is the bucket for words diarization could not attribute, not a
    # participant: listing it invites the agent to "name" someone who does not exist.
    unattributed = round(speech_seconds.pop(UNKNOWN_SPEAKER, 0.0), 1)
    if unattributed:
        count = sum(1 for item in utterances if item["speaker_id"] == UNKNOWN_SPEAKER)
        warnings.append(
            f"{count} utterance(s) with no speaker ({fmt_time(unattributed)}): diarization "
            "could not attribute them. Fix via review/transcript_edit.csv and refine."
        )
    speaker_ids = sorted((set(speech_seconds) | set(names)) - {UNKNOWN_SPEAKER})
    speakers_out: list[dict[str, Any]] = []
    for speaker_id in speaker_ids:
        name = names.get(speaker_id, speaker_id)
        if name == speaker_id:
            warnings.append(f"Speaker {speaker_id} has no name — name them with rename --set.")
        entry: dict[str, Any] = {
            "id": speaker_id,
            "name": name,
            "speech_seconds": round(speech_seconds.get(speaker_id, 0.0), 1),
        }
        preview = previews.get(speaker_id)
        if preview:
            entry["preview"] = {
                "meeting_start": round(float(preview["meeting_start"]), 1),
                "text": preview["text"],
            }
        speakers_out.append(entry)

    config = manifest.get("config", {})
    asr_models: list[str] = list(config.get("asr_models", []))
    asr_manifest = manifest.get("asr", {})
    asr_out = {
        "models": asr_models,
        "canonical_model": asr_models[0] if asr_models else None,
        "device": asr_manifest.get("effective_device"),
    }
    for dropped in config.get("asr_dropped_models") or []:
        # Provider-specific env var name, not a hardcoded GROQ_API_KEY that would be
        # wrong for a dropped deepgram model.
        provider = ASR_MODELS.get(dropped, {}).get("provider")
        env_name = ASR_API_KEY_ENV.get(provider, "the required API key")
        warnings.append(f"ASR model {dropped} was skipped at run time: {env_name} was missing.")

    overlap_regions = diarization_data.get("overlap") or []
    diarization_out = {
        "provider": diarization_data.get("provider", config.get("diarization_provider")),
        "num_speakers": diarization_data.get("num_speakers", len(speaker_ids)),
        "overlap_regions": len(overlap_regions),
    }
    if diarization_data.get("overlap_error"):
        warnings.append(f"Overlap detection was not performed: {diarization_data['overlap_error']}")
    if diarization_data.get("speaker_count_mismatch"):
        # deepgram has no speaker-count parameter, so this is a
        # warning here (not the RuntimeError every other provider raises in
        # core._stage_align), and belongs in result.json for the same reason the
        # other degraded conditions above do.
        warnings.append(str(diarization_data["speaker_count_mismatch"]))

    canonical_model = asr_models[0] if asr_models else None
    variant_models = asr_models[1:]
    raw_files: dict[str, str] = {}
    light_files: dict[str, str] = {}
    for model_name in ([canonical_model] if canonical_model else []) + variant_models:
        raw_file = job / f"transcript.raw.{model_name}.md"
        if raw_file.exists():
            raw_files[model_name] = raw_file.name
        light_file = job / f"transcript.light.{model_name}.md"
        if light_file.exists():
            light_files[model_name] = light_file.name
    readable_path = exports / "transcript.readable.md"
    readable_file = readable_path.name if readable_path.is_file() else None

    readable_manifest: dict[str, Any] = manifest.get("readable") or {"status": "not requested"}
    review_csv = exports / "review" / "readable_changes.csv"
    readable_out = {
        "status": readable_manifest.get("status"),
        "model": readable_manifest.get("model"),
        "changed_utterances": readable_manifest.get("changed_utterances"),
        "needs_attention": readable_manifest.get("needs_attention"),
        "report": "review/readable_changes.csv" if review_csv.exists() else None,
    }
    for field in (
        "accepted_utterances",
        "fallback_utterances",
        "provider",
        "api",
        "response_format",
        "repaired_utterances",
        "endpoint_failure",
        "pending_utterance_ids",
    ):
        if readable_manifest.get(field) is not None:
            readable_out[field] = readable_manifest[field]
    if readable_out["status"] == "failed":
        # summary.md is pasted into chat as-is, so a raw SDK error dict would land in
        # front of a human. The full text stays in manifest.readable.error.
        warnings.append(f"readable skipped: {_short_error(readable_manifest.get('error', ''))}")
    elif readable_out["status"] == "no speech":
        warnings.append("No speech detected; readable was not run.")
    elif readable_out.get("fallback_utterances"):
        warnings.append(
            f"readable used the local text for {readable_out['fallback_utterances']} utterance(s)."
        )
    elif "readable" in (config.get("clean") or []) and readable_file is None:
        warnings.append("readable was requested, but transcript.readable.md was not created.")
    if readable_out["status"] == "partial":
        warnings.append("Readable did not finish every batch. The export includes local text.")
    last_attempt = readable_manifest.get("last_attempt")
    if isinstance(last_attempt, dict):
        readable_out["last_attempt"] = last_attempt
        if last_attempt.get("status") == "failed":
            warnings.append("The latest readable attempt failed; the previous version was kept.")

    return {
        "job_id": manifest.get("job_id", job.name),
        "job_dir": str(job.resolve()),
        "meeting_name": manifest.get("meeting_name", job.name),
        # Always "complete": the local pipeline (ASR, diarization, alignment,
        # raw/light render) already finished by the time this is written. A failed
        # readable pass is a partial degradation, not a run failure — its own
        # status lives in readable.status and is echoed into warnings above (plan
        # section 4's "complete"/"readable_failed" pair in the schema sketch is
        # this field plus readable.status read together, not two states of one enum).
        "status": "complete",
        "duration_seconds": duration_seconds,
        "sources": source_names,
        "speakers": speakers_out,
        "asr": asr_out,
        "diarization": diarization_out,
        "transcripts": {"raw": raw_files, "light": light_files, "readable": readable_file},
        "readable": readable_out,
        "utterances": len(utterances),
        "warnings": warnings,
    }


def render_summary(result: dict[str, Any]) -> str:
    """One short Markdown page an agent can paste into chat without an LLM pass."""
    lines = [f"# {result['meeting_name']}", ""]
    lines.append(f"- Job: `{result['job_id']}`")
    lines.append(f"- Duration: {fmt_time(result['duration_seconds'])}")
    lines.append(f"- Sources: {', '.join(result['sources']) or '—'}")
    lines.append(f"- Utterances: {result['utterances']}")
    device = result["asr"]["device"] or "cpu"
    lines.append(f"- ASR: {', '.join(result['asr']['models']) or '—'} (device: {device})")
    diarization = result["diarization"]
    lines.append(
        f"- Diarization: {diarization['provider']}, speakers: {diarization['num_speakers']}, "
        f"overlapping speech regions: {diarization['overlap_regions']}"
    )
    lines.append("")
    lines.append("## Participants")
    if result["speakers"]:
        for speaker in result["speakers"]:
            lines.append(
                f"- **{speaker['name']}** ({speaker['id']}) — "
                f"{fmt_time(speaker['speech_seconds'])} of speech"
            )
            preview = speaker.get("preview")
            if preview:
                lines.append(
                    f'  First phrase [{fmt_time(preview["meeting_start"])}]: "{preview["text"]}"'
                )
    else:
        lines.append("- no data")
    lines.append("")
    lines.append("## Files")
    for model_name, filename in result["transcripts"]["raw"].items():
        lines.append(f"- `{filename}` — {_RAW_DESCRIPTION} (model {model_name})")
    for model_name, filename in result["transcripts"]["light"].items():
        lines.append(f"- `{filename}` — {_LIGHT_DESCRIPTION} (model {model_name})")
    if result["transcripts"]["readable"]:
        lines.append(f"- `{result['transcripts']['readable']}` — {_READABLE_DESCRIPTION}")
    if result["readable"]["report"]:
        lines.append(f"- `{result['readable']['report']}` — {_REVIEW_DESCRIPTION}")
    lines.append("")
    lines.append("## Warnings")
    if result["warnings"]:
        lines.extend(f"- {warning}" for warning in result["warnings"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_job_result(job: Path) -> dict[str, Any]:
    result = build_job_result(job)
    atomic_json(job / "result.json", result)
    atomic_text(job / "summary.md", render_summary(result))
    return result
