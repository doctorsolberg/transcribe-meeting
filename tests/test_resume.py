import pytest
from typer.testing import CliRunner

from transcribe_meeting.cli import app
from transcribe_meeting.core import asr_stage_fingerprint, complete_stage, stage_valid
from transcribe_meeting.model import Utterance
from transcribe_meeting.readable import ReadableResult
from transcribe_meeting.util import atomic_json, read_json


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


# --- resume finishes the readable pass the job was configured for -------------------
#
# The LLM pass lives outside run_job and has no `.stages/` marker, so nothing in the
# fingerprint machinery above can tell resume whether it still has to run. Interrupt
# a job before it and resume used to finish the local stages, print "Done" and leave
# the job without transcript.readable.md forever — the only trace was a warning
# buried in result.json. These tests drive the real `resume` command with run_job
# stubbed out (the local stages are already checkpointed by the time this matters)
# and only the OpenAI call replaced.


def _readable_job(tmp_path, modes=("raw", "light", "readable"), readable_block=None):
    """A job whose local stages are done: canonical.json exists, readable does not."""
    job = tmp_path / "job"
    job.mkdir()
    manifest = {
        "job_id": "job",
        "meeting_name": "test",
        "sources": [{"path": "/x/a.m4a", "duration": 60.0}],
        "glossary": [],
        "config": {
            "clean": list(modes),
            "llm_model": "gpt-test",
            "context_prompt": "",
            "asr_models": ["v3_rnnt"],
        },
    }
    if readable_block is not None:
        manifest["readable"] = readable_block
    atomic_json(job / "manifest.json", manifest)
    utterance = Utterance(
        utterance_id="u1",
        source_file="a.m4a",
        source_start=0.0,
        source_end=1.0,
        meeting_start=0.0,
        meeting_end=1.0,
        speaker_id="SPEAKER_00",
        speaker_name="SPEAKER_00",
        raw_text="усвол материал",
        confidence={},
        light_text="усвол материал.",
        readable_text="Усвоил материал."
        if (readable_block or {}).get("status") == "complete"
        else None,
    )
    atomic_json(
        job / "canonical.json",
        {
            "schema_version": 2,
            "timeline": [{"path": "/x/a.m4a"}],
            "utterances": [utterance.to_dict()],
            "diarization": {},
            "speaker_previews": {},
        },
    )
    return job


def _stub_readable_pass(monkeypatch, *, fail=False):
    """Replace the OpenAI call; return the list its invocations are recorded in."""
    calls = []

    def fake_clean_readable(utterances, model, protected, context_prompt, **kwargs):
        calls.append(model)
        if fail:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return ReadableResult(
            texts={item.utterance_id: "Усвоил материал." for item in utterances},
            concerns=[],
            usage={"input_tokens": 10, "output_tokens": 3},
        )

    monkeypatch.setattr("transcribe_meeting.core.clean_readable", fake_clean_readable)
    return calls


@pytest.mark.parametrize("explicit", [False, True])
def test_clean_honors_dotenv_readable_model_unless_passed_explicitly(
    tmp_path, monkeypatch, explicit
):
    job = _readable_job(tmp_path)
    calls = _stub_readable_pass(monkeypatch)
    settings = tmp_path / "settings.env"
    settings.write_text("TRANSCRIBE_READABLE_MODEL=gpt-6.2-sol\n", encoding="utf-8")
    monkeypatch.setenv("TRANSCRIBE_ENV_FILE", str(settings))
    flags = ["--llm-model", "override-model"] if explicit else []

    result = CliRunner().invoke(app, ["clean", str(job), "--mode", "readable", *flags])

    assert result.exit_code == 0, result.output
    assert calls == ["override-model" if explicit else "gpt-6.2-sol"]
    assert read_json(job / "manifest.json")["config"]["llm_model"] == "gpt-test"


@pytest.fixture
def stub_run_job(monkeypatch):
    """resume's local stages are checkpointed already; only the readable pass matters."""
    monkeypatch.setattr("transcribe_meeting.cli.run_job", lambda *args, **kwargs: None)


def test_resume_runs_the_readable_pass_when_the_job_was_interrupted_before_it(
    tmp_path, monkeypatch, stub_run_job
):
    job = _readable_job(tmp_path)
    calls = _stub_readable_pass(monkeypatch)
    monkeypatch.setenv("TRANSCRIBE_READABLE_MODEL", "new-default")

    result = CliRunner().invoke(app, ["resume", str(job)])

    assert result.exit_code == 0, result.output
    assert calls == ["gpt-test"]  # the model the job was configured with
    assert (job / "transcript.readable.md").exists()
    assert read_json(job / "manifest.json")["readable"]["status"] == "complete"


def test_resume_records_the_readable_result_in_result_json(tmp_path, monkeypatch, stub_run_job):
    """resume writes result.json after the readable pass, not before.

    Ordering is the whole point: run it afterwards and result.json still carries
    "readable was requested, but transcript.readable.md was not created".
    """
    job = _readable_job(tmp_path)
    _stub_readable_pass(monkeypatch)

    CliRunner().invoke(app, ["resume", str(job)])

    result = read_json(job / "result.json")
    assert result["transcripts"]["readable"] == "transcript.readable.md"
    assert result["readable"]["status"] == "complete"
    assert not [item for item in result["warnings"] if "readable was requested" in item]


def test_resume_does_not_repeat_a_readable_pass_that_already_completed(
    tmp_path, monkeypatch, stub_run_job
):
    """Resuming a finished job must not spend the tokens a second time."""
    job = _readable_job(tmp_path, readable_block={"model": "gpt-test", "status": "complete"})
    calls = _stub_readable_pass(monkeypatch)

    result = CliRunner().invoke(app, ["resume", str(job)])

    assert result.exit_code == 0, result.output
    assert calls == []
    assert "Усвоил материал." in (job / "transcript.readable.md").read_text(encoding="utf-8")
    assert (job / "review" / "readable_changes.csv").is_file()


def test_resume_restores_readable_added_with_clean(tmp_path, monkeypatch):
    job = _readable_job(tmp_path, modes=("raw", "light"))
    calls = _stub_readable_pass(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(app, ["clean", str(job), "--mode", "readable"])
    assert result.exit_code == 0, result.output
    for name in ("transcript.readable.md", "review/readable_changes.csv"):
        (job / name).unlink()

    result = runner.invoke(app, ["resume", str(job)])

    assert result.exit_code == 0, result.output
    assert calls == ["gpt-test"]
    assert (job / "transcript.readable.md").is_file()
    assert (job / "review/readable_changes.csv").is_file()


def test_resume_keeps_published_text_without_original_recording(tmp_path):
    job = _readable_job(tmp_path, modes=("raw", "light"))
    complete_stage(job, "align", {"word_timestamp_schema": 1})
    before = (job / "canonical.json").read_bytes()

    outcome = CliRunner().invoke(app, ["resume", str(job)])

    assert outcome.exit_code == 0, outcome.output
    assert (job / "canonical.json").read_bytes() == before
    assert read_json(job / "result.json")["utterances"] == 1


def test_resume_does_not_reprocess_legacy_job_missing_canonical(tmp_path):
    job = _readable_job(tmp_path, modes=("raw", "light"))
    (job / "canonical.json").unlink()
    atomic_json(job / "result.json", {"status": "complete", "utterances": 1})

    outcome = CliRunner().invoke(app, ["resume", str(job)])

    assert outcome.exit_code == 1
    assert "Restore it from backup" in outcome.stderr


def test_resume_rejects_modified_published_canonical_without_losing_exports(tmp_path):
    from transcribe_meeting.job import mark_published

    job = _readable_job(tmp_path, modes=("raw", "light"))
    mark_published(job)
    output = job / "transcript.raw.v3_rnnt.md"
    output.write_text("Previous accepted transcript", encoding="utf-8")
    canonical = read_json(job / "canonical.json")
    canonical["utterances"] = []
    atomic_json(job / "canonical.json", canonical)

    outcome = CliRunner().invoke(app, ["resume", str(job)])

    assert outcome.exit_code == 1
    assert "Restore it from backup" in outcome.stderr
    assert output.read_text(encoding="utf-8") == "Previous accepted transcript"


def test_clean_does_not_mark_a_manifest_only_job_complete(tmp_path):
    job = _readable_job(tmp_path)
    (job / "canonical.json").unlink()

    outcome = CliRunner().invoke(app, ["clean", str(job), "--mode", "readable"])

    assert outcome.exit_code == 2
    assert not (job / "result.json").exists()


def test_resume_keeps_manual_text_with_damaged_alignment_marker(tmp_path):
    from transcribe_meeting.job import mark_published

    job = _readable_job(tmp_path, modes=("raw", "light"))
    canonical = read_json(job / "canonical.json")
    canonical["utterances"][0]["raw_text"] = "Ручное исправление"
    atomic_json(job / "canonical.json", canonical)
    mark_published(job)
    (job / ".stages" / "align.json").write_text("{", encoding="utf-8")

    outcome = CliRunner().invoke(app, ["resume", str(job)])

    assert outcome.exit_code == 0, outcome.output
    assert read_json(job / "canonical.json") == canonical
    assert "Ручное исправление" in (job / "transcript.raw.v3_rnnt.md").read_text(encoding="utf-8")


def test_resume_retries_a_readable_pass_that_previously_failed(tmp_path, monkeypatch, stub_run_job):
    """A failed pass is usually a missing key or an unreachable endpoint: retry it."""
    job = _readable_job(
        tmp_path, readable_block={"model": "gpt-test", "status": "failed", "error": "no key"}
    )
    calls = _stub_readable_pass(monkeypatch)

    CliRunner().invoke(app, ["resume", str(job)])

    assert calls == ["gpt-test"]
    assert read_json(job / "manifest.json")["readable"]["status"] == "complete"


def test_resume_leaves_a_job_that_never_asked_for_readable_alone(
    tmp_path, monkeypatch, stub_run_job
):
    job = _readable_job(tmp_path, modes=("raw", "light"))
    calls = _stub_readable_pass(monkeypatch)

    result = CliRunner().invoke(app, ["resume", str(job)])

    assert result.exit_code == 0, result.output
    assert calls == []
    assert not (job / "transcript.readable.md").exists()


def test_resume_survives_a_failing_readable_pass_and_says_how_to_retry(
    tmp_path, monkeypatch, stub_run_job
):
    """The local transcript is finished; an unreachable LLM must not fail the job."""
    # The retry hint embeds tmp_path, so rich's hard wrap would otherwise land at a
    # machine-dependent point and could split "--mode readable" across two lines.
    monkeypatch.setenv("COLUMNS", "400")
    job = _readable_job(tmp_path)
    _stub_readable_pass(monkeypatch, fail=True)

    result = CliRunner().invoke(app, ["resume", str(job)])

    assert result.exit_code == 0, result.output
    assert f"transcribe-meeting clean {job} --mode readable" in result.output
    assert read_json(job / "manifest.json")["readable"]["status"] == "failed"
    assert (job / "result.json").exists()
