"""Job-directory bookkeeping: canonical.json's schema, models.lock.json lookups, and
the per-stage resume markers under `.stages/`.

This used to live inside core.py, mixed in with the pipeline stages that use it. It
is genuinely its own domain — a job on disk and what's already downloaded — separate
from the stages that run against a job, so it gets its own module. core.py imports
everything here (with load_utterances/save_canonical kept as their historical
_load_utterances/_save_canonical names) so that existing call sites and tests that
reach through `core.` need no changes.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .model import Utterance
from .paths import lock_path, models_dir, resolve_model_path
from .util import atomic_json, read_json

# Paths already warned about in this process, so a job that calls model_lock()
# several times (create_job, then again per stage in run_job) does not repeat the
# same migration notice on every call.
_LOCK_MIGRATION_WARNED: set[Path] = set()


def model_lock() -> dict[str, Any]:
    from .core import report_step  # deferred: core imports this module at load time

    path = lock_path()
    canonical = models_dir() / "models.lock.json"
    if path != canonical and path.exists() and path not in _LOCK_MIGRATION_WARNED:
        _LOCK_MIGRATION_WARNED.add(path)
        report_step(
            f"models.lock.json found in a legacy location ({path}); move it to "
            f"{canonical}: the lock describes the models directory's contents and must "
            "travel with it."
        )
    return read_json(path) if path.exists() else {"models": {}}


def _model_path(info: dict[str, Any], default_relative: str) -> Path:
    """Resolve one entry's model path: recorded value if present, else models_dir()
    default. Recorded values may be relative (to models_dir()) or absolute — see
    paths.resolve_model_path."""
    value = info.get("path")
    return resolve_model_path(value) if value else models_dir() / default_relative


def _as_stored(fingerprint: dict[str, Any]) -> Any:
    """Normalize a fingerprint into the shape it takes after a JSON round trip.

    A stage marker is written with atomic_json and read back with read_json, and JSON
    has no tuples: `[("sha", 240.0)]` is stored as `[["sha", 240.0]]` and returns as a
    list of lists, which never compares equal to the list of tuples the caller builds.
    Without this, every resume rebuilt audio, diarization and align from scratch.
    """
    return json.loads(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True))


def stage_valid(job: Path, stage: str, fingerprint: dict[str, Any]) -> bool:
    marker = job / ".stages" / f"{stage}.json"
    if not marker.exists():
        return False
    return read_json(marker).get("fingerprint") == _as_stored(fingerprint)


def complete_stage(job: Path, stage: str, fingerprint: dict[str, Any]) -> None:
    atomic_json(
        job / ".stages" / f"{stage}.json",
        {"fingerprint": fingerprint, "completed_at": dt.datetime.now(dt.UTC).isoformat()},
    )


#: canonical.json's own format version — bumped only when the utterance shape
#: changes in a way _load_utterances cannot read blindly. Distinct
#: from manifest["schema_version"], which tracks manifest.json, a different file.
#: Every canonical.json ever written before this constant existed hardcoded 1
#: regardless of shape, which is exactly why it stopped meaning anything: the
#: alt_raw_text -> variants change shipped without bumping it, and _load_utterances
#: raised a raw TypeError instead of a readable error. See the version==1 branch
#: below for why that does not lock the eight real jobs already on disk out.
CANONICAL_SCHEMA_VERSION = 2

#: Job directories already warned about a legacy (schema_version 1) canonical.json
#: in this process, so the warning prints once per job instead of on every stage
#: that happens to reload it.
_warned_legacy_canonical: set[Path] = set()


def load_utterances(job: Path) -> list[Utterance]:
    from .core import report_step  # deferred: core imports this module at load time

    canonical_path = job / "canonical.json"
    data = read_json(canonical_path)
    version = data.get("schema_version")
    if version == 1:
        # Predates CANONICAL_SCHEMA_VERSION: every canonical.json ever written before
        # this check existed claims 1 no matter its actual shape. Only readable here
        # if today's Utterance dataclass can still parse it (checked below by the
        # same unknown-field guard as any other version) — otherwise it falls through
        # to that guard's clear error instead of continuing silently.
        if job not in _warned_legacy_canonical:
            _warned_legacy_canonical.add(job)
            report_step(
                f"Warning: {job.name} — canonical.json is in the legacy format "
                "(schema_version 1, no field validation). The job still "
                "opens; rename, refine, or clean --mode readable "
                f"will resave canonical.json and upgrade it to version "
                f"{CANONICAL_SCHEMA_VERSION}."
            )
    elif version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(
            f"{canonical_path}: schema_version={version!r}, but this code can only read "
            f"{CANONICAL_SCHEMA_VERSION} (version 1 as a legacy format, with a warning). "
            "This job looks like it was made by a different version of transcribe-meeting: "
            "upgrade the tool to the version that produced this run, or, if this is "
            "known to be an older run, use that version of the project."
        )
    known_fields = {item.name for item in fields(Utterance)}
    utterances = []
    for item in data["utterances"]:
        unknown = set(item) - known_fields
        if unknown:
            raise ValueError(
                f"{canonical_path}: utterance {item.get('utterance_id', '?')!r} has "
                f"unknown fields {sorted(unknown)}. This job looks like it was made by a "
                "different version of transcribe-meeting: upgrade the tool to the version "
                "that produced this run, or use that version of the project."
            )
        utterances.append(Utterance(**item))
    return utterances


def save_canonical(
    job: Path,
    manifest: dict[str, Any],
    utterances: list[Utterance],
    diarization_data: dict[str, Any],
    speaker_previews_data: dict[str, Any] | None = None,
) -> None:
    if speaker_previews_data is None:
        # rename/refine/run_readable all resave canonical.json without recomputing
        # previews (they never touch alignment); preserve whatever run_job's align
        # stage last wrote instead of silently dropping it.
        existing = read_json(job / "canonical.json") if (job / "canonical.json").exists() else {}
        speaker_previews_data = existing.get("speaker_previews", {})
    atomic_json(
        job / "canonical.json",
        {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "timeline": manifest["sources"],
            "utterances": [item.to_dict() for item in utterances],
            "diarization": diarization_data,
            "speaker_previews": speaker_previews_data,
        },
    )


def manifest_asr_models(manifest: dict[str, Any]) -> list[str]:
    """Read config.asr_models, reporting an outdated manifest instead of a KeyError.

    Jobs created before ASR became a list carry the singular `asr_model`. Canonical
    JSON already fails such mismatches with a readable message; a manifest raising a
    bare KeyError from deep inside rendering was the one place still doing otherwise.
    """
    config = manifest.get("config", {})
    if "asr_models" in config:
        return list(config["asr_models"])
    legacy = config.get("asr_model")
    raise ValueError(
        "This job was created in a legacy format: manifest.json has no config.asr_models"
        + (f" (only the legacy asr_model={legacy!r} is present)" if legacy else "")
        + ". Such jobs are no longer supported — create the job again with the "
        "run command."
    )
