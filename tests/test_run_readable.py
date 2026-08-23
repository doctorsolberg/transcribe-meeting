from transcribe_meeting.core import run_readable
from transcribe_meeting.model import Utterance
from transcribe_meeting.readable import CONCERN_FALLBACK, CONCERN_REPAIR, ReadableResult
from transcribe_meeting.util import atomic_json, read_json


def _utterance(uid, raw, light=None):
    return Utterance(
        utterance_id=uid,
        source_file="a.m4a",
        source_start=0.0,
        source_end=1.0,
        meeting_start=0.0,
        meeting_end=1.0,
        speaker_id="SPEAKER_00",
        speaker_name="SPEAKER_00",
        raw_text=raw,
        confidence={},
        light_text=light if light is not None else raw + ".",
    )


def _job(tmp_path):
    atomic_json(
        tmp_path / "manifest.json",
        {
            "meeting_name": "test",
            "sources": [{"path": "/x/a.m4a"}],
            "glossary": [],
            "config": {
                "clean": ["raw", "light", "readable"],
                "llm_model": "gpt-test",
                "context_prompt": "",
                "asr_models": ["v3_rnnt"],
            },
        },
    )
    repaired = _utterance("u1", "усвол материал")
    fallback = _utterance("u2", "встреча пройдёт 15 числа")
    atomic_json(
        tmp_path / "canonical.json",
        {
            "schema_version": 2,
            "timeline": [{"path": "/x/a.m4a"}],
            "utterances": [repaired.to_dict(), fallback.to_dict()],
            "diarization": {},
            "speaker_previews": {},
        },
    )
    return tmp_path


def _stub_clean_readable(
    utterances, model, protected, context_prompt, status=None, variant_models=()
):
    """Stands in for the real OpenAI call: run_readable only has to wire its
    ReadableResult into canonical.json, the manifest and the review CSV."""
    return ReadableResult(
        texts={"u1": "Усвоил материал.", "u2": "встреча пройдёт 15 числа"},
        concerns=[
            {"utterance_id": "u1", "note": f"{CONCERN_REPAIR}: усвол → усвоил"},
            {
                "utterance_id": "u2",
                "note": f"{CONCERN_FALLBACK}: потеряно защищённое значение '15'",
            },
        ],
        usage={"input_tokens": 42, "output_tokens": 7},
    )


def test_run_readable_rewrites_canonical_with_fused_text(tmp_path, monkeypatch):
    job = _job(tmp_path)
    monkeypatch.setattr("transcribe_meeting.core.clean_readable", _stub_clean_readable)

    run_readable(job)

    canonical = read_json(job / "canonical.json")
    texts = {item["utterance_id"]: item["readable_text"] for item in canonical["utterances"]}
    assert texts == {"u1": "Усвоил материал.", "u2": "встреча пройдёт 15 числа"}


def test_run_readable_records_manifest_block(tmp_path, monkeypatch):
    job = _job(tmp_path)
    monkeypatch.setattr("transcribe_meeting.core.clean_readable", _stub_clean_readable)

    run_readable(job)

    readable = read_json(job / "manifest.json")["readable"]
    assert readable["status"] == "complete"
    assert readable["model"] == "gpt-test"
    assert readable["usage"] == {"input_tokens": 42, "output_tokens": 7}
    assert len(readable["concerns"]) == 2
    # Counted by the CONCERN_FALLBACK/CONCERN_REPAIR prefixes, not a substring guess
    # at the (translatable) Russian sentence that follows them.
    assert readable["fallback_utterances"] == 1
    assert readable["repaired_utterances"] == 1
    assert readable["changed_utterances"] == 1  # only u1 differs from its light text
    assert readable["needs_attention"] == 0  # "repair" is not a risky change kind


def test_run_readable_writes_the_review_csv(tmp_path, monkeypatch):
    job = _job(tmp_path)
    monkeypatch.setattr("transcribe_meeting.core.clean_readable", _stub_clean_readable)

    run_readable(job)

    review_csv = job / "review" / "readable_changes.csv"
    assert review_csv.exists()
    assert "u1" in review_csv.read_text(encoding="utf-8")
