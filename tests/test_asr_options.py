import math
from types import SimpleNamespace

import pytest

from transcribe_meeting.asr import (
    effective_asr_device,
    gigaam_result_to_asr,
    validate_word_timestamps,
)


def test_gigaam_result_is_normalized_to_root_level_words():
    result = SimpleNamespace(
        text="пример текста",
        words=[
            SimpleNamespace(text="пример", start=0.1, end=0.4),
            SimpleNamespace(text="текста", start=0.5, end=0.9),
        ],
    )

    normalized = gigaam_result_to_asr(result, 1.0)

    assert normalized["words"] == [
        {"word": "пример", "start": 0.1, "end": 0.4, "probability": None},
        {"word": "текста", "start": 0.5, "end": 0.9, "probability": None},
    ]


@pytest.mark.parametrize(
    "words",
    [
        [{"word": "", "start": 0, "end": 1}],
        [{"word": "x", "start": 1, "end": 1}],
        [{"word": "x", "start": 0, "end": math.inf}],
        [{"word": "x", "start": 0.5, "end": 0.8}, {"word": "y", "start": 0.1, "end": 0.4}],
    ],
)
def test_invalid_gigaam_timestamps_are_rejected(words):
    with pytest.raises(ValueError):
        validate_word_timestamps(words, 1.0)


def test_auto_device_falls_back_but_explicit_mps_does_not():
    assert effective_asr_device("auto", mps_available=False) == "cpu"
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        effective_asr_device("mps", mps_available=False)
