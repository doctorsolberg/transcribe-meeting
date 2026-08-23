from __future__ import annotations

from dataclasses import asdict

from .model import Source

GAP_SECONDS = 1.0


def build_sources(items: list[tuple[str, str, float]]) -> list[Source]:
    """Build the three clocks; the synthetic gap only exists in diarization time."""
    result: list[Source] = []
    meeting_offset = 0.0
    diarization_offset = 0.0
    for index, (path, digest, duration) in enumerate(items):
        result.append(Source(path, digest, duration, meeting_offset, diarization_offset))
        meeting_offset += duration
        diarization_offset += duration
        if index < len(items) - 1:
            diarization_offset += GAP_SECONDS
    return result


def mapping(sources: list[Source]) -> list[dict]:
    return [asdict(source) for source in sources]
