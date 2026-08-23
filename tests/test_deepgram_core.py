import pytest

from transcribe_meeting import core
from transcribe_meeting.asr import ASR_MODELS, DEEPGRAM_ASR_MODEL_NAME
from transcribe_meeting.util import atomic_json, read_json


def test_deepgram_registry_entry_has_meeting_scope():
    """deepgram-nova-3 is the one scope="meeting" model; every other
    registry entry is scope="part"."""
    assert ASR_MODELS[DEEPGRAM_ASR_MODEL_NAME]["scope"] == "meeting"
    assert ASR_MODELS[DEEPGRAM_ASR_MODEL_NAME]["provider"] == "deepgram"
    part_scoped = [name for name, info in ASR_MODELS.items() if name != DEEPGRAM_ASR_MODEL_NAME]
    assert all(ASR_MODELS[name]["scope"] == "part" for name in part_scoped)


def _manifest(sources):
    return {"sources": sources, "config": {"language": "ru"}}


def test_run_deepgram_asr_requests_once_and_splits_words_per_source(tmp_path, monkeypatch):
    manifest = _manifest(
        [
            {"path": "/x/1.wav", "duration": 1.0, "diarization_offset": 0.0},
            {"path": "/x/2.wav", "duration": 1.0, "diarization_offset": 2.0},
        ]
    )
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
                                    {"word": "a", "start": 0.1, "end": 0.4, "confidence": 0.9},
                                    {"word": "b", "start": 2.1, "end": 2.4, "confidence": 0.8},
                                ]
                            }
                        ]
                    }
                ]
            }
        }

    monkeypatch.setenv("DEEPGRAM_API_KEY", "key-1")
    monkeypatch.setattr(core, "transcribe_deepgram", fake_transcribe_deepgram)

    core._run_deepgram_asr(tmp_path, manifest, DEEPGRAM_ASR_MODEL_NAME, 1, 1)

    assert len(calls) == 1  # exactly one Deepgram request for both parts
    part_one = read_json(tmp_path / "asr" / DEEPGRAM_ASR_MODEL_NAME / "001.json")
    part_two = read_json(tmp_path / "asr" / DEEPGRAM_ASR_MODEL_NAME / "002.json")
    assert [w["word"] for w in part_one["words"]] == ["a"]
    assert [w["word"] for w in part_two["words"]] == ["b"]
    assert part_two["words"][0]["start"] == pytest.approx(0.1)
    assert part_one["run_parameters"]["provider"] == "deepgram"


def test_run_deepgram_asr_skips_the_request_when_every_part_already_exists(tmp_path, monkeypatch):
    manifest = _manifest([{"path": "/x/1.wav", "duration": 1.0, "diarization_offset": 0.0}])
    destination = tmp_path / "asr" / DEEPGRAM_ASR_MODEL_NAME / "001.json"
    atomic_json(destination, {"text": "", "segments": [], "words": []})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not call transcribe_deepgram when every part exists")

    monkeypatch.setattr(core, "transcribe_deepgram", fail_if_called)

    core._run_deepgram_asr(tmp_path, manifest, DEEPGRAM_ASR_MODEL_NAME, 1, 1)  # must not raise


def test_run_deepgram_asr_requires_the_api_key(tmp_path, monkeypatch):
    manifest = _manifest([{"path": "/x/1.wav", "duration": 1.0, "diarization_offset": 0.0}])
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        core._run_deepgram_asr(tmp_path, manifest, DEEPGRAM_ASR_MODEL_NAME, 1, 1)


def test_reconcile_speaker_count_is_a_warning_for_deepgram_not_an_error():
    diar_data = {"exclusive": []}

    core._reconcile_speaker_count(
        "deepgram", ["SPEAKER_00", "SPEAKER_01"], 3, diar_data
    )  # no raise

    assert "speaker_count_mismatch" in diar_data
    assert "3" in diar_data["speaker_count_mismatch"]
    assert "2" in diar_data["speaker_count_mismatch"]


def test_reconcile_speaker_count_still_raises_for_foxnose():
    with pytest.raises(RuntimeError, match="foxnose"):
        core._reconcile_speaker_count("foxnose", ["SPEAKER_00"], 3, {})


def test_reconcile_speaker_count_still_raises_for_pyannote():
    with pytest.raises(RuntimeError, match="pyannote"):
        core._reconcile_speaker_count("pyannote", ["SPEAKER_00"], 2, {})


def test_reconcile_speaker_count_is_a_noop_when_counts_match_or_speakers_is_auto():
    diar_data = {}
    core._reconcile_speaker_count("deepgram", ["SPEAKER_00", "SPEAKER_01"], 2, diar_data)
    core._reconcile_speaker_count("deepgram", ["SPEAKER_00"], None, diar_data)

    assert diar_data == {}


def test_deepgram_diarization_without_a_key_is_refused_not_downgraded(monkeypatch):
    """Silently falling back to another diarizer would change what the result means."""
    import typer

    from transcribe_meeting.cli import _validate_asr_options

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(typer.BadParameter, match="DEEPGRAM_API_KEY"):
        _validate_asr_options(
            asr_provider=None,
            asr_device="cpu",
            asr_models="multilingual_large_ctc",
            diarization_provider="deepgram",
            language="ru",
            chunk_minutes=None,
            cpu_threads=None,
            ffmpeg_threads=None,
            asr_preprocess="none",
            asr_compression="none",
            context_prompt="",
        )
