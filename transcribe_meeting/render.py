from __future__ import annotations

import re
from pathlib import Path

from .model import Utterance
from .util import atomic_text, fmt_time

# raw stays one line per utterance so it remains checkable against canonical.json
# and review/transcript_edit.csv, where a reviewer locates rows by utterance_id.
GROUPED_MODES = ("light", "readable")
# The marker belongs in the modes meant for reading, not in raw, which is a
# line-by-line ASR snapshot; GROUPED_MODES already draws exactly that boundary.
OVERLAP_MARK = "[overlapping speech]"

_LIGHT_BLOCK_START = re.compile(r"^(\[[^\]]*\]\s*)?(.)")


def finish_light_block(text: str) -> str:
    """Capitalize a light block and close it with a single final period.

    light_clean works one utterance at a time and an utterance is a fragment, so
    sentence-final punctuation can only be decided here, where the whole block is known.
    The optional overlap marker is skipped, not capitalized.
    """
    if not text:
        return text
    text = _LIGHT_BLOCK_START.sub(
        lambda match: (match.group(1) or "") + match.group(2).upper(), text, count=1
    )
    if text[-1] not in ".!?…":
        text += "."
    return text


def _mode_text(
    utterance: Utterance, attr: str, *, fallback: bool, mode: str, variant_model: str | None = None
) -> str | None:
    """Read one Utterance field for rendering.

    variant_model, when given, names one of the non-canonical ASR models: its text
    comes from the utterance's variants/variant_light dicts instead of attr, and
    fallback is ignored (an absent entry must render as "nothing to say here", never
    silently borrow the canonical model's text, or the per-model files would stop
    being line-by-line comparable). fallback=True (used for the canonical/model-a
    files and for readable) reproduces the historical fallback to raw_text.
    """
    if variant_model is not None:
        store = utterance.variant_light if mode == "light" else utterance.variants
        text = store.get(variant_model)
    else:
        text = getattr(utterance, attr, None)
        if not text:
            text = utterance.raw_text if fallback else None
    if not text:
        return None
    if utterance.overlap and mode in GROUPED_MODES:
        return f"{OVERLAP_MARK} {text}"
    return text


def blocks(utterances: list[Utterance], mode: str) -> list[list[Utterance]]:
    """Group consecutive turns of one speaker so a long speech reads as a single block.

    Presentation only: canonical.json still carries every utterance_id with its own
    timecodes and speaker id. Grouping never crosses a source
    file boundary, because a merged block could not name a single source.
    """
    if mode not in GROUPED_MODES:
        return [[utterance] for utterance in utterances]
    groups: list[list[Utterance]] = []
    for utterance in utterances:
        previous = groups[-1][-1] if groups else None
        if (
            previous is not None
            and previous.speaker_id == utterance.speaker_id
            and previous.source_file == utterance.source_file
        ):
            groups[-1].append(utterance)
        else:
            groups.append([utterance])
    return groups


def render_markdown(
    job: Path,
    manifest: dict,
    utterances: list[Utterance],
    mode: str,
    *,
    text_attr: str | None = None,
    fallback: bool = True,
    suffix: str = "",
    variant_model: str | None = None,
) -> None:
    """Render one Markdown file.

    `mode` controls the grouping layout (raw stays one line per utterance; light and
    readable merge consecutive same-speaker turns) and, when text_attr is not given,
    also names the Utterance field (`f"{mode}_text"`). `suffix`, when given, names the
    ASR model this file belongs to and is appended to the filename, so raw/light exist
    once per model (`transcript.raw.<model>.md`) while readable stays a single merged
    file (`transcript.readable.md`, no suffix — there is only one merged variant).
    `variant_model`, when given, renders one non-canonical model's text from the
    utterance's variants/variant_light dicts instead of text_attr; see _mode_text.
    """
    text_attr = text_attr or f"{mode}_text"
    title = manifest["meeting_name"]
    sources = " → ".join(Path(item["path"]).name for item in manifest["sources"])
    label = f"{mode} ({suffix})" if suffix else mode
    lines = [f"# Meeting: {title}", "", f"> Sources: {sources}", f"> Text mode: {label}", ""]
    prior = None
    for block in blocks(utterances, mode):
        head, tail = block[0], block[-1]
        text = " ".join(
            part
            for part in (
                _mode_text(
                    item, text_attr, fallback=fallback, mode=mode, variant_model=variant_model
                )
                for item in block
            )
            if part
        )
        if mode == "light":
            text = finish_light_block(text)
        if not text:
            # This repl(ica) group has nothing for this variant; skip the line, not the file.
            continue
        if prior is not None and head.source_file != prior:
            lines.extend([f"--- Recording continues: {Path(head.source_file).name} ---", ""])
        lines.append(
            f"[{fmt_time(head.meeting_start)}–{fmt_time(tail.meeting_end)}] "
            f"{head.speaker_name}: {text}"
        )
        prior = head.source_file
    body = "\n".join(lines) + "\n"
    filename = f"transcript.{mode}.{suffix}.md" if suffix else f"transcript.{mode}.md"
    atomic_text(job / filename, body)
