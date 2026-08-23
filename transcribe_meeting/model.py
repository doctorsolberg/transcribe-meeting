from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Bucket for words diarization could not attribute; not a participant.
UNKNOWN_SPEAKER = "Speaker ?"


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    duration: float
    offset: float
    diarization_offset: float


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None
    speaker_id: str = UNKNOWN_SPEAKER


@dataclass
class Utterance:
    utterance_id: str
    source_file: str
    source_start: float
    source_end: float
    meeting_start: float
    meeting_end: float
    speaker_id: str
    speaker_name: str
    raw_text: str
    confidence: dict[str, Any]
    overlap: bool = False
    uncertain_spans: list[dict[str, Any]] = field(default_factory=list)
    light_text: str | None = None
    readable_text: str | None = None
    # Non-canonical ASR models' text for the same utterance boundaries, assigned by
    # word center (see alignment.assign_alt_words). Keyed by model name from
    # config["asr_models"] (excluding the canonical, first, model). A model is absent
    # from the dict when it has no words assigned to this utterance.
    variants: dict[str, str] = field(default_factory=dict)
    variant_light: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
