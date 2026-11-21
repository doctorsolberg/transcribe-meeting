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


def test_silent_job_neither_renders_nor_requests_readable(tmp_path, monkeypatch):
    job = _rendered_job(tmp_path, [])

    def no_request(*args, **kwargs):
        raise AssertionError("silent job must not call a provider")

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", no_request)

    assert not (job / "transcript.readable.md").exists()
    assert try_readable(job) is True
    assert read_json(job / "manifest.json")["readable"]["status"] == "no speech"


def test_readable_only_still_renders_local_outputs(tmp_path):
    job = _rendered_job(tmp_path, [_utterance(1)])
    manifest = read_json(job / "manifest.json")
    manifest["config"]["clean"] = ["readable"]
    atomic_json(job / "manifest.json", manifest)
    (job / "transcript.light.v3_rnnt.md").unlink()
    (job / "transcript.raw.v3_rnnt.md").unlink()

    render_all(job)

    assert (job / "transcript.light.v3_rnnt.md").is_file()
    assert (job / "transcript.raw.v3_rnnt.md").is_file()


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


def test_failed_retry_preserves_previously_accepted_readable(tmp_path, monkeypatch):
    job = _rendered_job(tmp_path, [_utterance(1, "Текст.")])
    path = job / "transcript.readable.md"
    original_text = path.read_bytes()
    manifest = read_json(job / "manifest.json")
    manifest["readable"] = {"status": "complete", "model": "accepted-model"}
    atomic_json(job / "manifest.json", manifest)

    def bad_response(*args, **kwargs):
        raise RuntimeError("endpoint unavailable")

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", bad_response)
    assert try_readable(job, "retry-model") is False
    state = read_json(job / "manifest.json")["readable"]
    assert state["status"] == "complete"
    assert state["model"] == "accepted-model"
    assert state["last_attempt"]["status"] == "failed"
    assert state["last_attempt"]["model"] == "retry-model"
    assert path.read_bytes() == original_text


def test_failed_retry_does_not_replace_local_files(tmp_path, monkeypatch):
    job = _rendered_job(tmp_path, [_utterance(1)])
    path = job / "transcript.light.v3_rnnt.md"
    original_text = path.read_bytes()

    def bad_response(*args, **kwargs):
        raise RuntimeError("endpoint unavailable")

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", bad_response)
    assert try_readable(job) is False
    assert path.read_bytes() == original_text
