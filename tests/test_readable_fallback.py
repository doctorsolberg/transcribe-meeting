from transcribe_meeting.core import render_all, try_readable
from transcribe_meeting.model import Utterance
from transcribe_meeting.util import atomic_json, read_json


def _utterance(index: int, readable: str | None = None) -> Utterance:
    return Utterance(
        utterance_id=f"p001-u{index:05d}",
        source_file="a.m4a",
        source_start=0.0,
        source_end=1.0,
        meeting_start=0.0,
        meeting_end=1.0,
        speaker_id="SPEAKER_00",
        speaker_name="SPEAKER_00",
        raw_text="текст",
        confidence={},
        light_text="текст.",
        readable_text=readable,
    )


def _rendered_job(tmp_path, utterances, asr_models=("v3_rnnt",)):
    atomic_json(
        tmp_path / "manifest.json",
        {
            "meeting_name": "test",
            "sources": [{"path": "/x/a.m4a"}],
            "config": {
                "clean": ["raw", "light", "readable"],
                "llm_model": "m",
                "asr_models": list(asr_models),
            },
        },
    )
    atomic_json(
        tmp_path / "canonical.json",
        {"schema_version": 2, "utterances": [item.to_dict() for item in utterances]},
    )
    render_all(tmp_path)
    return tmp_path


def test_failed_readable_leaves_no_readable_file(tmp_path):
    """Nothing is written for a mode with no text, so a missing file means 'not made'."""
    job = _rendered_job(tmp_path, [_utterance(1)])

    assert (job / "transcript.raw.v3_rnnt.md").exists()
    assert (job / "transcript.light.v3_rnnt.md").exists()
    assert not (job / "transcript.readable.md").exists()


def test_successful_readable_produces_its_own_file(tmp_path):
    job = _rendered_job(tmp_path, [_utterance(1, "Текст.")])

    assert (job / "transcript.readable.md").exists()


def test_render_never_writes_an_unlabelled_transcript(tmp_path):
    job = _rendered_job(tmp_path, [_utterance(1, "Текст.")])

    assert not (job / "transcript.md").exists()


def _job(tmp_path):
    atomic_json(
        tmp_path / "manifest.json",
        {"config": {"llm_model": "gpt-5.6-terra", "clean": ["raw", "light"]}},
    )
    return tmp_path


def test_try_readable_survives_a_failing_model_and_records_it(tmp_path, monkeypatch):
    """A missing key or a dead endpoint must not fail a finished local transcript."""
    job = _job(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("readable expects OPENAI_API_KEY in the environment")

    monkeypatch.setattr("transcribe_meeting.core.run_readable", boom)

    assert try_readable(job) is False
    recorded = read_json(job / "manifest.json")["readable"]
    assert recorded["status"] == "failed"
    assert "OPENAI_API_KEY" in recorded["error"]
    assert recorded["model"] == "gpt-5.6-terra"


def test_try_readable_swallows_non_runtime_errors_too(tmp_path, monkeypatch):
    """The OpenAI SDK raises its own hierarchy; none of it may escape."""
    job = _job(tmp_path)

    class VendorError(Exception):
        pass

    def boom(*_args, **_kwargs):
        raise VendorError("connection reset")

    monkeypatch.setattr("transcribe_meeting.core.run_readable", boom)

    assert try_readable(job) is False


def test_try_readable_reports_success(tmp_path, monkeypatch):
    job = _job(tmp_path)
    monkeypatch.setattr("transcribe_meeting.core.run_readable", lambda *a, **k: None)

    assert try_readable(job) is True
    assert "readable" not in read_json(job / "manifest.json")
