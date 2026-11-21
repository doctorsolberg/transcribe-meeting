import pytest

from transcribe_meeting import readable_publish
from transcribe_meeting.core import refine, run_readable, try_readable
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
        accepted_utterances=1,
        fallback_utterances=1,
        provider="custom",
        api="chat",
        response_format="json_object",
    )


def test_run_readable_rewrites_canonical_with_fused_text(tmp_path, monkeypatch):
    job = _job(tmp_path)
    monkeypatch.setattr("transcribe_meeting.core.clean_readable", _stub_clean_readable)

    run_readable(job)

    canonical = read_json(job / "canonical.json")
    texts = {item["utterance_id"]: item["readable_text"] for item in canonical["utterances"]}
    assert texts == {"u1": "Усвоил материал.", "u2": "встреча пройдёт 15 числа"}


@pytest.mark.parametrize("previous_readable", [False, True])
def test_readable_keeps_text_corrected_with_refine(tmp_path, monkeypatch, previous_readable):
    job = _job(tmp_path)
    if previous_readable:
        monkeypatch.setattr("transcribe_meeting.core.clean_readable", _stub_clean_readable)
        run_readable(job)
    corrections = job / "corrections.csv"
    corrections.write_text("utterance_id,replacement\nu1,Ручная правка\n", encoding="utf-8")
    refine(job, corrections)

    def clean_remaining(utterances, *_args, **_kwargs):
        assert [item.utterance_id for item in utterances] == ["u2"]
        return ReadableResult(texts={"u2": "Встреча пройдёт 15 числа."}, concerns=[], usage={})

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", clean_remaining)
    run_readable(job)
    run_readable(job)

    canonical = read_json(job / "canonical.json")
    assert canonical["utterances"][0]["readable_text"] == "Ручная правка"
    assert canonical["utterances"][1]["readable_text"] == "Встреча пройдёт 15 числа."


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
    assert readable["accepted_utterances"] == 1
    assert readable["provider"] == "custom"
    assert readable["api"] == "chat"
    assert readable["response_format"] == "json_object"
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
    assert read_json(job / "result.json")["readable"]["model"] == "gpt-test"
    assert (job / "summary.md").is_file()


@pytest.mark.parametrize("failure", [None, "readable request failed with HTTP 429"])
def test_partial_readable_cannot_replace_complete_version(tmp_path, monkeypatch, failure):
    job = _job(tmp_path)

    def partial(*args, **kwargs):
        result = _stub_clean_readable(*args, **kwargs)
        result.endpoint_failure = failure
        result.pending_utterance_ids = ["u2"]
        return result

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", partial)
    run_readable(job)
    assert read_json(job / "result.json")["readable"]["status"] == "partial"

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", _stub_clean_readable)
    run_readable(job, "complete-version")
    before = (job / "transcript.readable.md").read_bytes()

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", partial)
    assert not try_readable(job, "partial-version")
    assert (job / "transcript.readable.md").read_bytes() == before
    state = read_json(job / "manifest.json")["readable"]
    assert state["model"] == "complete-version"
    assert state["last_attempt"]["status"] == "failed"


def test_readable_rejects_stale_source_after_a_manual_edit(tmp_path, monkeypatch):
    job = _job(tmp_path)

    def edit_during_request(*args, **kwargs):
        canonical = read_json(job / "canonical.json")
        canonical["utterances"][0]["raw_text"] = "Ручное исправление"
        atomic_json(job / "canonical.json", canonical)
        return _stub_clean_readable(*args, **kwargs)

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", edit_during_request)

    assert try_readable(job) is False
    assert read_json(job / "canonical.json")["utterances"][0]["raw_text"] == "Ручное исправление"
    assert not (job / "transcript.readable.md").exists()


def test_failed_readable_audit_keeps_previous_generation(tmp_path, monkeypatch):
    job = _job(tmp_path)
    monkeypatch.setattr("transcribe_meeting.core.clean_readable", _stub_clean_readable)
    run_readable(job)
    paths = [
        job / name
        for name in ("canonical.json", "transcript.readable.md", "review/readable_changes.csv")
    ]
    before = [path.read_bytes() for path in paths]

    def fail_audit(_job):
        raise OSError("audit unavailable")

    def different_result(*args, **kwargs):
        result = _stub_clean_readable(*args, **kwargs)
        result.texts["u1"] = "Усвоил новый материал."
        return result

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", different_result)
    monkeypatch.setattr("transcribe_meeting.core.write_readable_changes", fail_audit)
    assert try_readable(job, "second-attempt") is False
    assert [path.read_bytes() for path in paths] == before
    assert read_json(job / "manifest.json")["readable"]["model"] == "gpt-test"


def test_interrupted_refine_cannot_leave_partial_edits(tmp_path, monkeypatch):
    job = _job(tmp_path)
    canonical = job / "canonical.json"
    before = canonical.read_bytes()
    csv = job / "edit.csv"
    csv.write_text("utterance_id,replacement\nu1,Ручная правка\n", encoding="utf-8")
    replace = readable_publish.os.replace

    def interrupt(src, dst):
        replace(src, dst)
        if dst == canonical:
            raise KeyboardInterrupt

    with monkeypatch.context() as killed:
        killed.setattr(readable_publish.os, "replace", interrupt)
        killed.setattr(readable_publish, "_recover", lambda _job: None)
        with pytest.raises(KeyboardInterrupt):
            refine(job, csv)

    readable_publish.recover_readable_publication(job)
    assert canonical.read_bytes() == before
    assert not read_json(job / "manifest.json").get("manual_text_edits")


def test_interrupted_readable_publish_recovers_on_next_open(tmp_path, monkeypatch):
    job = _job(tmp_path)
    monkeypatch.setattr("transcribe_meeting.core.clean_readable", _stub_clean_readable)
    run_readable(job)
    previous_manifest = (job / "manifest.json").read_bytes()
    original = (job / "canonical.json").read_bytes()
    write_json = readable_publish.atomic_json

    def interrupt(path, value):
        if path == job / "manifest.json":
            raise KeyboardInterrupt
        write_json(path, value)

    monkeypatch.setattr(readable_publish, "atomic_json", interrupt)
    # Skip finally once to reproduce a killed process, not a handled Ctrl-C.
    with monkeypatch.context() as interrupted:
        interrupted.setattr(readable_publish, "_recover", lambda _job: None)
        with pytest.raises(KeyboardInterrupt):
            run_readable(job, "second-attempt")

    assert (job / ".readable-publish/journal.json").is_file()
    assert (job / ".readable-publish").stat().st_mode & 0o077 == 0
    readable_publish.recover_readable_publication(job)
    assert (job / "canonical.json").read_bytes() == original
    assert (job / "canonical.json").stat().st_mode & 0o077 == 0
    assert (job / "manifest.json").read_bytes() == previous_manifest
    assert not (job / ".readable-publish").exists()
