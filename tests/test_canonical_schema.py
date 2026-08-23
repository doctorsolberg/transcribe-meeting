import pytest

from transcribe_meeting import core
from transcribe_meeting.util import atomic_json


def _utterance(uid="u1", **extra):
    base = {
        "utterance_id": uid,
        "source_file": "a.m4a",
        "source_start": 0.0,
        "source_end": 1.0,
        "meeting_start": 0.0,
        "meeting_end": 1.0,
        "speaker_id": "S1",
        "speaker_name": "Игорь",
        "raw_text": "текст",
        "confidence": {},
    }
    base.update(extra)
    return base


def _write_canonical(job, *, schema_version, utterances):
    atomic_json(
        job / "canonical.json",
        {
            "schema_version": schema_version,
            "timeline": [],
            "utterances": utterances,
            "diarization": {},
            "speaker_previews": {},
        },
    )


def test_current_schema_version_reads_without_warning(tmp_path, monkeypatch):
    _write_canonical(
        tmp_path, schema_version=core.CANONICAL_SCHEMA_VERSION, utterances=[_utterance()]
    )
    messages: list[str] = []
    monkeypatch.setattr(core, "report_step", messages.append)

    utterances = core._load_utterances(tmp_path)

    assert len(utterances) == 1
    assert messages == []


def test_legacy_version_1_reads_with_a_one_time_warning(tmp_path, monkeypatch):
    """Jobs written before this check existed are all version 1
    with a shape today's Utterance can read, and must keep opening — with a single
    warning, not a wall of them on every reload."""
    _write_canonical(tmp_path, schema_version=1, utterances=[_utterance()])
    core._warned_legacy_canonical.discard(tmp_path)
    messages: list[str] = []
    monkeypatch.setattr(core, "report_step", messages.append)

    first = core._load_utterances(tmp_path)
    second = core._load_utterances(tmp_path)

    assert len(first) == 1
    assert len(second) == 1
    assert len(messages) == 1
    assert "legacy format" in messages[0]


def test_a_future_schema_version_raises_a_clear_error_not_a_typeerror(tmp_path):
    _write_canonical(tmp_path, schema_version=3, utterances=[_utterance()])

    with pytest.raises(ValueError, match="schema_version"):
        core._load_utterances(tmp_path)


def test_an_unknown_utterance_field_raises_a_clear_error_not_a_typeerror(tmp_path):
    """The historical failure this guards against: renaming alt_raw_text to variants
    once raised a raw TypeError('unexpected keyword argument') instead of a message."""
    _write_canonical(
        tmp_path,
        schema_version=core.CANONICAL_SCHEMA_VERSION,
        utterances=[_utterance(alt_raw_text="старое поле")],
    )

    with pytest.raises(ValueError, match="unknown fields"):
        core._load_utterances(tmp_path)


def test_save_canonical_writes_the_current_schema_version(tmp_path):
    from transcribe_meeting.model import Utterance

    manifest = {"sources": []}
    core._save_canonical(tmp_path, manifest, [Utterance(**_utterance())], {})

    saved = core.read_json(tmp_path / "canonical.json")
    assert saved["schema_version"] == core.CANONICAL_SCHEMA_VERSION


def test_outdated_manifest_reports_itself_instead_of_raising_keyerror():
    """The oldest job carries the singular `asr_model`; rendering must say so plainly."""
    import pytest

    from transcribe_meeting.core import manifest_asr_models

    with pytest.raises(ValueError, match="legacy format"):
        manifest_asr_models({"config": {"asr_model": "multilingual_large_ctc"}})


def test_current_manifest_returns_its_model_list():
    from transcribe_meeting.core import manifest_asr_models

    assert manifest_asr_models({"config": {"asr_models": ["a", "b"]}}) == ["a", "b"]
