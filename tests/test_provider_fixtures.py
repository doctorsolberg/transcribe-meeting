"""Real-response fixtures for the two remaining providers, alongside deepgram_response.json.

Each fixture is a trimmed slice of a genuine job output, kept for the one property that
caused a real defect: GigaAM aligns to acoustic frames and must never overlap, while Groq
(Whisper) estimates each word's span independently and routinely does overlap. A guard
test on each fixture fails loudly if the slice ever stops exhibiting that property, so it
cannot silently be "cleaned up" into a fixture that no longer earns its place.
"""

import json
from pathlib import Path

import pytest

from transcribe_meeting.asr import validate_word_timestamps, words_from_asr

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _duration(words: list[dict]) -> float:
    return max(float(word["end"]) for word in words)


# --- GigaAM -----------------------------------------------------------------


def test_gigaam_fixture_still_has_strictly_monotonic_non_overlapping_words():
    """Guard: this fixture is kept for strict monotonicity; without it, it is just
    another word list and should be replaced rather than trusted."""
    words = _load("gigaam_words.json")["words"]

    assert len(words) >= 30
    assert all(words[i + 1]["start"] >= words[i]["end"] for i in range(len(words) - 1)), (
        "gigaam_words.json no longer demonstrates non-overlapping spans"
    )
    assert all(word.get("probability") is None for word in words)


def test_gigaam_words_survive_words_from_asr_and_strict_validation():
    data = _load("gigaam_words.json")
    words = data["words"]

    parsed = words_from_asr(data)
    assert len(parsed) == len(words)

    # GigaAM is validated strictly (allow_overlap=False) in production; see
    # gigaam_result_to_asr in asr.py.
    validate_word_timestamps(words, _duration(words))


# --- Groq (Whisper) -----------------------------------------------------------


def test_groq_fixture_still_has_an_overlapping_pair_and_empty_probabilities():
    """Guard: this fixture is kept for an overlapping span; without one it would no
    longer exercise allow_overlap at all."""
    words = _load("groq_words.json")["words"]

    assert len(words) >= 30
    assert any(words[i + 1]["start"] < words[i]["end"] for i in range(len(words) - 1)), (
        "groq_words.json no longer contains an overlapping span"
    )
    assert all(word.get("probability") is None for word in words)


def test_groq_words_survive_words_from_asr_and_relaxed_validation():
    data = _load("groq_words.json")
    words = data["words"]

    parsed = words_from_asr(data)
    assert len(parsed) == len(words)

    # Whisper (via Groq) estimates each word's span independently, so overlap is
    # expected and must be tolerated; see the transcribe_part groq branch in asr.py.
    validate_word_timestamps(words, _duration(words), allow_overlap=True)


def test_groq_fixture_fails_strict_validation_without_allow_overlap():
    """The overlap the fixture is kept for is real: strict mode must reject it, or
    this fixture would not actually be demonstrating anything about Groq."""
    words = _load("groq_words.json")["words"]

    with pytest.raises(ValueError):
        validate_word_timestamps(words, _duration(words))
