"""The manual-review round trip: export a job's utterances to CSV, let a human edit
speaker names or wrong ASR text, then fold the edits back into canonical.json. Also
the deterministic audit of what the optional `readable` LLM pass changed.

This used to live inside core.py, mixed in with the pipeline stages that produce the
utterances it edits. core.py imports prepare_review/refine/write_readable_changes/
CHANGE_COLUMNS back under their historical names so existing call sites and tests
that reach through `core.` need no changes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from .cleaning import light_clean
from .job import load_utterances, save_canonical
from .readable import RISKY_CHANGE_KINDS, classify_changes
from .result import write_job_result
from .util import atomic_text, fmt_time, read_json

CHANGE_COLUMNS = [
    "utterance_id",
    "time",
    "speaker",
    "change_kinds",
    "changed_words",
    "needs_attention",
    "before",
    "after",
]


def write_readable_changes(job: Path) -> tuple[Path, dict[str, int]]:
    """Write the deterministic audit of what the LLM changed in readable.

    Reported per utterance, not per word: a transliteration is physically a deletion
    plus an insertion, so a word-level report splits 'виспер' -> 'Whisper' into a
    phantom loss and an unrelated gain. Two texts side by side need no such pairing.
    """
    utterances = load_utterances(job)
    rows = []
    totals: dict[str, int] = {}
    for utterance in utterances:
        kinds = classify_changes(utterance)
        for kind in kinds:
            totals[kind] = totals.get(kind, 0) + 1
        if not kinds:
            continue
        risky = bool(RISKY_CHANGE_KINDS.intersection(kinds))
        rows.append(
            {
                "utterance_id": utterance.utterance_id,
                "time": fmt_time(utterance.meeting_start),
                "speaker": utterance.speaker_name,
                "change_kinds": ",".join(sorted(set(kinds))),
                "changed_words": len(kinds),
                "needs_attention": "yes" if risky else "",
                "before": utterance.light_text or utterance.raw_text,
                "after": utterance.readable_text or "",
            }
        )
    # Risky rows first so the file opens on what actually needs a human.
    rows.sort(key=lambda row: (row["needs_attention"] != "yes", row["utterance_id"]))
    target = job / "review" / "readable_changes.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else CHANGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    totals["utterances"] = len(rows)
    totals["needs_attention"] = sum(1 for row in rows if row["needs_attention"] == "yes")
    return target, totals


def prepare_review(job: Path) -> Path:
    """Export every utterance for manual review — not a confidence-based subset:

    no connected ASR backend (local GigaAM, Groq Whisper) returns word probabilities,
    so `mean_word_probability` is always None and any threshold on it would silently
    yield an empty file. `review`/`refine` are the only way to fix a wrong ASR word
    or a misattributed speaker, so the export must cover the whole transcript instead.
    """
    rows = [
        {
            "utterance_id": utterance.utterance_id,
            "timestamp": utterance.meeting_start,
            "speaker": utterance.speaker_name,
            "text": utterance.raw_text,
            "context": "",
            "suggestion": "",
            "replacement": "",
            "speaker_replacement": "",
        }
        for utterance in load_utterances(job)
    ]
    target = job / "review" / "transcript_edit.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utterance_id",
                "timestamp",
                "speaker",
                "text",
                "context",
                "suggestion",
                "replacement",
                "speaker_replacement",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(
        job / "review" / "index.md",
        f"# Transcript editing\n\n"
        f"Utterances in this file: {len(rows)} (every utterance in the job, not just "
        "the doubtful ones — no ASR provider returns word probabilities, so selecting "
        "by confidence is not possible).\n\n"
        "Two independent fields for edits:\n"
        "- `replacement` — the corrected text of the utterance, if ASR got it wrong.\n"
        "- `speaker_replacement` — the correct speaker name for this specific "
        "utterance, if diarization attributed it to the wrong person. This changes only "
        "this utterance, not the speaker as a whole (that's what `rename` is for); if "
        "the given name matches one of the names in `speakers.yaml`, the "
        "`speaker_id` changes too.\n\n"
        "Fill in the fields you need, then run `refine`.\n",
    )
    return target


def refine(job: Path, review_csv: Path) -> None:
    # Deferred: core imports write_readable_changes/prepare_review/refine from this
    # module, so a module-level `from .core import render_all` would be circular.
    from .core import render_all

    text_replacements: dict[str, str] = {}
    speaker_replacements: dict[str, str] = {}
    with review_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("replacement", "").strip():
                text_replacements[row["utterance_id"]] = row["replacement"].strip()
            if row.get("speaker_replacement", "").strip():
                speaker_replacements[row["utterance_id"]] = row["speaker_replacement"].strip()
    speakers_path = job / "speakers.yaml"
    known_names = (
        (yaml.safe_load(speakers_path.read_text(encoding="utf-8")) or {}).get("speakers", {})
        if speakers_path.exists()
        else {}
    )
    # A replacement that matches an existing speakers.yaml name is a known
    # participant, so the speaker_id can follow along; an unrecognized name only
    # relabels this one utterance's display text, since there is no global speaker
    # to attach an unknown name to.
    name_to_id = {name: identifier for identifier, name in known_names.items()}
    utterances = load_utterances(job)
    for item in utterances:
        if item.utterance_id in text_replacements:
            item.raw_text = text_replacements[item.utterance_id]
            item.light_text = light_clean(item.raw_text)
        if item.utterance_id in speaker_replacements:
            new_name = speaker_replacements[item.utterance_id]
            item.speaker_name = new_name
            if new_name in name_to_id:
                item.speaker_id = name_to_id[new_name]
    canonical = read_json(job / "canonical.json")
    save_canonical(job, read_json(job / "manifest.json"), utterances, canonical["diarization"])
    render_all(job)
    # refine rewrites utterance text and speakers, so the job summary must follow:
    # a stale result.json would report counts that no longer match the transcripts.
    write_job_result(job)
