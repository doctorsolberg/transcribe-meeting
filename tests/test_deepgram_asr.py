import os
import wave

import pytest

from transcribe_meeting import asr
from transcribe_meeting.util import sha256


def _wav(path, *, seconds: int = 1, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * rate * seconds)


def test_deepgram_language_param_maps_auto_to_multi():
    """nova-3's code-switching mode, not a generic detect-one-language."""
    assert asr._deepgram_language_param("auto") == "multi"
    assert asr._deepgram_language_param("ru") == "ru"
    assert asr._deepgram_language_param("en") == "en"


def test_deepgram_request_builds_expected_url_headers_and_body(monkeypatch):
    captured = {}

    class FakeResponse:
        def read(self):
            return b'{"results": {}}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["data"] = request.data
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(asr.urllib.request, "urlopen", fake_urlopen)

    result = asr._deepgram_request(b"RIFF...", "secret-key", "auto")

    assert result == {"results": {}}
    assert captured["method"] == "POST"
    assert captured["data"] == b"RIFF..."
    assert captured["headers"]["Authorization"] == "Token secret-key"
    assert captured["headers"]["Content-type"] == "audio/wav"
    assert captured["timeout"] == asr.DEEPGRAM_REQUEST_TIMEOUT_SECONDS
    assert "model=nova-3" in captured["url"]
    assert "language=multi" in captured["url"]
    assert "diarize_model=latest" in captured["url"]
    # Diarization comes from diarize_model alone. Sending "diarize" alongside it is a
    # 400 from the live API ("diarize_model cannot be used together with diarize or
    # diarize_version"), so this asserts the absence, not just the presence.
    assert "diarize=" not in captured["url"]
    assert "utterances=true" not in captured["url"]
    assert captured["url"].startswith(asr.DEEPGRAM_URL)


def test_deepgram_request_wraps_http_error_with_status_and_body(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", None, __import__("io").BytesIO(b"bad key")
        )

    monkeypatch.setattr(asr.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        asr._deepgram_request(b"data", "bad-key", "ru")


def test_transcribe_deepgram_writes_checkpoint_and_returns_data(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.wav"
    _wav(audio)
    target = tmp_path / "asr" / "deepgram-nova-3" / "meeting.json"
    seen = {}

    def fake_request(audio_bytes, api_key, language):
        seen["api_key"] = api_key
        seen["language"] = language
        return {"results": {"channels": [{"alternatives": [{"words": []}]}]}}

    monkeypatch.setattr(asr, "_deepgram_request", fake_request)

    result = asr.transcribe_deepgram(audio, "key-1", "ru", target)

    assert result == {"results": {"channels": [{"alternatives": [{"words": []}]}]}}
    assert target.exists()
    fingerprint = asr.deepgram_response_fingerprint(sha256(audio), "ru")
    assert asr.read_deepgram_response(target, fingerprint) == result
    assert seen == {"api_key": "key-1", "language": "ru"}


def test_transcribe_deepgram_second_call_reads_checkpoint_without_a_request(tmp_path, monkeypatch):
    """The raw response is the checkpoint — a second call must never
    pay for (or even attempt) a second Deepgram request."""
    audio = tmp_path / "meeting.wav"
    _wav(audio)
    target = tmp_path / "asr" / "deepgram-nova-3" / "meeting.json"
    response = {"results": {"channels": [{"alternatives": [{"words": []}]}]}}
    monkeypatch.setattr(asr, "_deepgram_request", lambda *_: response)
    assert asr.transcribe_deepgram(audio, "key-1", "ru", target) == response

    def fail_if_called(*args, **kwargs):
        raise AssertionError("transcribe_deepgram must not re-request when checkpoint exists")

    monkeypatch.setattr(asr, "_deepgram_request", fail_if_called)

    result = asr.transcribe_deepgram(audio, "key-1", "ru", target)

    assert result == response


def test_transcribe_deepgram_rejects_oversized_track_before_any_request(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.wav"
    # A sparse file is enough to exercise the size check without writing 2 GiB of
    # real data to disk; os.truncate creates holes on APFS/most filesystems.
    audio.write_bytes(b"\0")
    os.truncate(audio, asr.DEEPGRAM_MAX_UPLOAD_BYTES + 1)
    target = tmp_path / "asr" / "deepgram-nova-3" / "meeting.json"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("must reject before making any request")

    monkeypatch.setattr(asr, "_deepgram_request", fail_if_called)

    with pytest.raises(RuntimeError, match="2 GB"):
        asr.transcribe_deepgram(audio, "key-1", "ru", target)

    assert not target.exists()


def test_deepgram_words_extracts_the_documented_response_path():
    words = [{"word": "привет", "start": 0.0, "end": 0.4}]
    data = {"results": {"channels": [{"alternatives": [{"words": words}]}]}}

    assert asr.deepgram_words(data) is words


def test_deepgram_words_raises_a_clear_error_on_an_unexpected_shape():
    with pytest.raises(RuntimeError, match="Deepgram"):
        asr.deepgram_words({"unexpected": True})


def test_deepgram_words_for_source_slices_shifts_and_preserves_order():
    """Order comes from the response, never re-sorted by timestamp;
    only words whose start falls in [base, base+duration) belong to this source."""
    all_words = [
        {"word": "a", "start": 0.0, "end": 0.5, "confidence": 0.9},
        {"word": "b", "start": 0.6, "end": 1.0, "confidence": 0.8},
        # Gap word (the synthetic 1s silence build_sources adds between parts).
        {"word": "c", "start": 1.5, "end": 1.9, "confidence": 0.7},
        {"word": "d", "start": 2.0, "end": 2.4, "confidence": 0.95},
    ]

    first_part = asr.deepgram_words_for_source(all_words, base=0.0, duration=1.0)
    second_part = asr.deepgram_words_for_source(all_words, base=2.0, duration=1.0)

    assert [w["word"] for w in first_part] == ["a", "b"]
    assert first_part[0] == {"word": "a", "start": 0.0, "end": 0.5, "probability": 0.9}
    assert first_part[1] == {"word": "b", "start": 0.6, "end": 1.0, "probability": 0.8}
    assert [w["word"] for w in second_part] == ["d"]
    assert second_part[0]["word"] == "d"
    assert second_part[0]["start"] == pytest.approx(0.0)
    assert second_part[0]["end"] == pytest.approx(0.4)
    assert second_part[0]["probability"] == 0.95


def test_deepgram_words_for_source_clamps_end_at_the_source_boundary():
    """Timing jitter across the exact boundary must never produce end > duration,
    which would fail validate_word_timestamps downstream."""
    all_words = [{"word": "x", "start": 0.9, "end": 1.2, "confidence": 0.5}]

    sliced = asr.deepgram_words_for_source(all_words, base=0.0, duration=1.0)

    assert sliced == [{"word": "x", "start": 0.9, "end": 1.0, "probability": 0.5}]


def test_deepgram_words_for_source_prefers_the_punctuated_word():
    """smart_format's case, punctuation and digitized numbers live only in
    punctuated_word — README recommends this route specifically for that."""
    all_words = [{"word": "но", "punctuated_word": "Но,", "start": 0.0, "end": 0.4}]

    sliced = asr.deepgram_words_for_source(all_words, base=0.0, duration=1.0)

    assert sliced[0]["word"] == "Но,"


def test_deepgram_words_for_source_falls_back_to_word_without_punctuated_word():
    all_words = [{"word": "но", "start": 0.0, "end": 0.4}]

    sliced = asr.deepgram_words_for_source(all_words, base=0.0, duration=1.0)

    assert sliced[0]["word"] == "но"


def test_overlapping_spans_are_accepted_but_bounds_are_not_relaxed():
    """Deepgram overlapped 290 of 16460 spans on a real recording; that is normal."""
    from transcribe_meeting.asr import validate_word_timestamps

    overlapping = [
        {"word": "будет", "start": 12.32, "end": 13.92},
        {"word": "идеальный", "start": 12.85, "end": 13.41},
    ]
    validate_word_timestamps(overlapping, 20.0, allow_overlap=True)

    with pytest.raises(ValueError, match="outside the chunk bounds"):
        validate_word_timestamps(
            [{"word": "поздно", "start": 19.0, "end": 25.0}], 20.0, allow_overlap=True
        )


def test_frame_aligned_backends_stay_strict():
    """GigaAM aligns to frames: an overlap there means something actually broke."""
    from transcribe_meeting.asr import validate_word_timestamps

    with pytest.raises(ValueError, match="out-of-order"):
        validate_word_timestamps(
            [
                {"word": "будет", "start": 12.32, "end": 13.92},
                {"word": "идеальный", "start": 12.85, "end": 13.41},
            ],
            20.0,
        )


def test_zero_length_word_is_widened_not_dropped():
    """One real word in 16460 came back with start == end and confidence 0.99."""
    from transcribe_meeting.asr import deepgram_words_for_source

    words = [{"word": "агентом", "start": 1484.8873, "end": 1484.8873, "confidence": 0.99}]

    result = deepgram_words_for_source(words, 0.0, 3000.0)

    assert len(result) == 1
    assert result[0]["word"] == "агентом"
    assert result[0]["start"] < result[0]["end"]


def test_a_word_is_never_widened_past_the_source_end():
    from transcribe_meeting.asr import deepgram_words_for_source

    words = [{"word": "край", "start": 99.999, "end": 99.999, "confidence": 0.9}]

    result = deepgram_words_for_source(words, 0.0, 100.0)

    assert result[0]["end"] == 100.0
    assert result[0]["start"] < result[0]["end"]


def _real_response():
    """A trimmed slice of a genuine two-hour Deepgram response.

    Kept because three separate defects reached production on properties no invented
    fixture had: overlapping word spans, a zero-length word, and non-strict ordering.
    The slice deliberately centres on the two spans that broke the pipeline.
    """
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "deepgram_response.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_response_survives_the_whole_word_path():
    """End to end over real data: parse, split into a source, validate."""
    from transcribe_meeting.asr import (
        deepgram_words,
        deepgram_words_for_source,
        validate_word_timestamps,
    )

    words = deepgram_words(_real_response())
    assert words

    sliced = deepgram_words_for_source(words, 0.0, 7000.0)
    validate_word_timestamps(sliced, 7000.0, allow_overlap=True)

    assert len(sliced) == len(words)
    assert all(w["start"] < w["end"] for w in sliced)
    # Order is preserved; text comes from punctuated_word where the fixture has one
    # (see test_deepgram_words_for_source_prefers_the_punctuated_word).
    assert [w["word"] for w in sliced] == [w.get("punctuated_word") or w["word"] for w in words]


def test_real_response_still_carries_the_edge_cases_it_was_kept_for():
    """A guard on the fixture itself: if it stops covering these, it stops earning its place."""
    words = _real_response()["results"]["channels"][0]["alternatives"][0]["words"]

    assert any(w["start"] == w["end"] for w in words)
    assert any(words[i + 1]["start"] + 0.05 < words[i]["end"] for i in range(len(words) - 1))


def test_real_response_diarization_segments_are_speaker_labelled():
    from transcribe_meeting.asr import deepgram_words
    from transcribe_meeting.diarization import _deepgram_segments_from_words

    rows = _deepgram_segments_from_words(deepgram_words(_real_response()))

    assert rows
    assert all(row["speaker"].startswith("SPEAKER_") for row in rows)
    assert all(row["start"] <= row["end"] for row in rows)
