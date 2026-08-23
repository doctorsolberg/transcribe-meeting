from transcribe_meeting import core
from transcribe_meeting.util import atomic_json, read_json


def test_detect_and_record_overlap_persists_regions_on_success(tmp_path, monkeypatch):
    diar_file = tmp_path / "diarization.json"
    atomic_json(diar_file, {"provider": "foxnose", "regular": [], "exclusive": []})
    monkeypatch.setattr(core, "detect_overlap_regions", lambda audio, model, threads: [(1.0, 2.0)])

    core.detect_and_record_overlap(tmp_path / "meeting.wav", diar_file, tmp_path / "model", 1)

    payload = read_json(diar_file)
    assert payload["overlap"] == [[1.0, 2.0]]
    assert payload["overlap_error"] is None
    # The diarization result written before overlap detection ran must survive untouched.
    assert payload["provider"] == "foxnose"


def test_detect_and_record_overlap_never_raises_and_records_the_error(tmp_path, monkeypatch):
    diar_file = tmp_path / "diarization.json"
    atomic_json(diar_file, {"provider": "pyannote", "regular": [], "exclusive": []})

    def boom(audio, model, threads):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(core, "detect_overlap_regions", boom)

    # Must not raise: overlap detection is a supplement, not a pipeline prerequisite.
    core.detect_and_record_overlap(tmp_path / "meeting.wav", diar_file, tmp_path / "model", 1)

    payload = read_json(diar_file)
    assert payload["overlap"] == []
    assert "model unavailable" in payload["overlap_error"]
