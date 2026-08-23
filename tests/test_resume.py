from transcribe_meeting.core import asr_stage_fingerprint, complete_stage, stage_valid


def test_stage_cache_is_invalidated_by_changed_fingerprint(tmp_path):
    fingerprint = {"sources": ["hash-a"], "model": "revision-a", "config": {"language": "ru"}}
    complete_stage(tmp_path, "asr", fingerprint)
    assert stage_valid(tmp_path, "asr", fingerprint)
    assert not stage_valid(tmp_path, "asr", {**fingerprint, "model": "revision-b"})


def test_asr_stage_fingerprint_changes_when_model_list_changes():
    """Changing --asr-models must invalidate the asr stage's resume checkpoint."""
    base = {"sources": ["hash-a"], "config": {"language": "ru"}, "models": {}}
    one_model = asr_stage_fingerprint(base, [], "cpu", ["v3_rnnt"])
    two_models = asr_stage_fingerprint(base, [], "cpu", ["multilingual_large_ctc", "v3_rnnt"])
    reordered = asr_stage_fingerprint(base, [], "cpu", ["v3_rnnt", "multilingual_large_ctc"])

    assert one_model != two_models
    assert two_models != reordered  # order matters: the first model is canonical
    assert one_model["asr_models"] == ["v3_rnnt"]


def test_asr_stage_cache_is_invalidated_when_the_model_set_changes(tmp_path):
    base = {"sources": ["hash-a"], "config": {"language": "ru"}, "models": {}}
    fp_one = asr_stage_fingerprint(base, [], "cpu", ["v3_rnnt"])
    fp_two = asr_stage_fingerprint(base, [], "cpu", ["multilingual_large_ctc", "v3_rnnt"])
    complete_stage(tmp_path, "asr", fp_one)

    assert stage_valid(tmp_path, "asr", fp_one)
    assert not stage_valid(tmp_path, "asr", fp_two)


def test_diarization_stage_cache_is_invalidated_by_changed_provider(tmp_path):
    fingerprint = {
        "sources": ["hash-a"],
        "config": {"language": "ru"},
        "stage": "diarization",
        "diarization_provider": "pyannote",
        "diarize_package_version": None,
    }
    complete_stage(tmp_path, "diarization", fingerprint)
    assert stage_valid(tmp_path, "diarization", fingerprint)
    assert not stage_valid(
        tmp_path, "diarization", {**fingerprint, "diarization_provider": "foxnose"}
    )


def test_tuple_fingerprint_survives_the_json_round_trip(tmp_path):
    """The real fingerprint holds (sha256, duration) tuples, which JSON stores as lists.

    Comparing the stored list against a freshly built tuple always failed, so every
    resume silently re-ran audio, diarization and align. Earlier tests missed it by
    using plain strings, which round-trip losslessly.
    """
    fingerprint = {
        "sources": [("hash-a", 240.01), ("hash-b", 300.5)],
        "config": {"language": "ru"},
    }
    complete_stage(tmp_path, "audio", fingerprint)

    assert stage_valid(tmp_path, "audio", fingerprint)


def test_a_changed_tuple_fingerprint_is_still_rejected(tmp_path):
    """Normalizing for comparison must not make every fingerprint match."""
    fingerprint = {"sources": [("hash-a", 240.01)], "config": {"language": "ru"}}
    complete_stage(tmp_path, "audio", fingerprint)

    assert not stage_valid(
        tmp_path, "audio", {"sources": [("hash-a", 999.0)], "config": {"language": "ru"}}
    )
