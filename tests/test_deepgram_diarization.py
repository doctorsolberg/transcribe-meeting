import pytest

from transcribe_meeting import diarization


def test_deepgram_segments_glue_same_speaker_words_within_the_gap():
    words = [
        {"word": "a", "start": 0.0, "end": 0.4, "speaker": 0},
        {"word": "b", "start": 0.5, "end": 0.9, "speaker": 0},  # 0.1s gap, glued
        {"word": "c", "start": 1.6, "end": 2.0, "speaker": 0},  # 0.7s gap, still glued
    ]

    segments = diarization._deepgram_segments_from_words(words)

    assert segments == [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]


def test_deepgram_segments_break_on_speaker_change():
    words = [
        {"word": "a", "start": 0.0, "end": 0.4, "speaker": 0},
        {"word": "b", "start": 0.5, "end": 0.9, "speaker": 1},
    ]

    segments = diarization._deepgram_segments_from_words(words)

    assert segments == [
        {"start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 0.9, "speaker": "SPEAKER_01"},
    ]


def test_deepgram_segments_break_on_a_gap_longer_than_the_threshold():
    words = [
        {"word": "a", "start": 0.0, "end": 0.4, "speaker": 0},
        # 0.81s gap: strictly longer than DEEPGRAM_SEGMENT_GAP (0.8s) must split.
        {"word": "b", "start": 1.21, "end": 1.5, "speaker": 0},
    ]

    segments = diarization._deepgram_segments_from_words(words)

    assert segments == [
        {"start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
        {"start": 1.21, "end": 1.5, "speaker": "SPEAKER_00"},
    ]


def test_deepgram_segments_skip_words_with_no_speaker():
    """A speaker-less word is dropped outright rather than starting or ending a
    segment — Deepgram's response should always carry a speaker per word, so this is
    a defensive skip. The gap check that follows only sees the last word that *did*
    carry a speaker, so a nearby same-speaker word on either side still glues through.
    """
    words = [
        {"word": "a", "start": 0.0, "end": 0.4, "speaker": 0},
        {"word": "b", "start": 0.5, "end": 0.9},
        {"word": "c", "start": 1.0, "end": 1.3, "speaker": 0},
    ]

    segments = diarization._deepgram_segments_from_words(words)

    assert segments == [{"start": 0.0, "end": 1.3, "speaker": "SPEAKER_00"}]


def test_deepgram_segments_label_is_zero_padded_for_double_digit_speakers():
    words = [{"word": "a", "start": 0.0, "end": 0.4, "speaker": 11}]

    assert diarization._deepgram_segments_from_words(words) == [
        {"start": 0.0, "end": 0.4, "speaker": "SPEAKER_11"}
    ]


def test_diarize_deepgram_builds_payload_reuses_the_shared_response_and_writes_rttm(
    tmp_path, monkeypatch
):
    calls = []

    def fake_transcribe_deepgram(audio, api_key, language, target):
        calls.append((str(audio), api_key, language, str(target)))
        return {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "words": [
                                    {"word": "a", "start": 0.0, "end": 0.4, "speaker": 0},
                                    {"word": "b", "start": 0.5, "end": 0.9, "speaker": 1},
                                ]
                            }
                        ]
                    }
                ]
            }
        }

    monkeypatch.setattr(diarization, "transcribe_deepgram", fake_transcribe_deepgram)
    output = tmp_path / "diarization.json"
    rttm = tmp_path / "diarization.rttm"
    checkpoint = tmp_path / "asr" / "deepgram-nova-3" / "meeting.json"

    payload = diarization._diarize_deepgram(
        tmp_path / "meeting.wav", output, rttm, "key-1", "ru", checkpoint
    )

    assert payload["provider"] == "deepgram"
    assert payload["repository"] == diarization.DEEPGRAM_REPO
    assert payload["cpu_threads"] == 0
    assert payload["supports_overlap"] is False
    assert payload["regular"] == payload["exclusive"]
    assert payload["num_speakers"] == 2
    assert [row["speaker"] for row in payload["regular"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert output.exists()
    assert rttm.read_text(encoding="utf-8") == (
        "SPEAKER meeting 1 0.000 0.400 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER meeting 1 0.500 0.400 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
    )
    assert calls == [(str(tmp_path / "meeting.wav"), "key-1", "ru", str(checkpoint))]


def test_diarize_dispatch_deepgram_requires_an_api_key(tmp_path):
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        diarization.diarize(
            tmp_path / "a.wav",
            tmp_path / "o.json",
            tmp_path / "o.rttm",
            tmp_path / "m",
            None,
            1,
            8,
            1,
            provider="deepgram",
            deepgram_api_key=None,
            deepgram_checkpoint=tmp_path / "meeting.json",
        )


def test_diarize_dispatch_deepgram_requires_a_checkpoint_path(tmp_path):
    with pytest.raises(RuntimeError, match="checkpoint"):
        diarization.diarize(
            tmp_path / "a.wav",
            tmp_path / "o.json",
            tmp_path / "o.rttm",
            tmp_path / "m",
            None,
            1,
            8,
            1,
            provider="deepgram",
            deepgram_api_key="key-1",
            deepgram_checkpoint=None,
        )


def test_diarize_dispatch_deepgram_reaches_the_real_helper(tmp_path, monkeypatch):
    """End-to-end through the public diarize() dispatcher, not just the private helper."""
    called = {}

    def fake_diarize_deepgram(audio, output, rttm, api_key, language, checkpoint):
        called["args"] = (audio, output, rttm, api_key, language, checkpoint)
        diarization.atomic_json(output, {"provider": "deepgram"})
        return {"provider": "deepgram"}

    monkeypatch.setattr(diarization, "_diarize_deepgram", fake_diarize_deepgram)

    diarization.diarize(
        tmp_path / "a.wav",
        tmp_path / "o.json",
        tmp_path / "o.rttm",
        tmp_path / "m",
        None,
        1,
        8,
        1,
        provider="deepgram",
        deepgram_api_key="key-1",
        deepgram_language="multi",
        deepgram_checkpoint=tmp_path / "meeting.json",
    )

    assert called["args"] == (
        tmp_path / "a.wav",
        tmp_path / "o.json",
        tmp_path / "o.rttm",
        "key-1",
        "multi",
        tmp_path / "meeting.json",
    )
