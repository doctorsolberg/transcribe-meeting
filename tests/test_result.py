import json

from transcribe_meeting.result import build_job_result, render_summary, write_job_result
from transcribe_meeting.util import atomic_json, atomic_text, read_json


def _utterance(uid, speaker_id, start, end):
    return {
        "utterance_id": uid,
        "source_file": "/x/src/1.m4a",
        "source_start": start,
        "source_end": end,
        "meeting_start": start,
        "meeting_end": end,
        "speaker_id": speaker_id,
        "speaker_name": speaker_id,
        "raw_text": "текст",
        "confidence": {},
        "overlap": False,
        "uncertain_spans": [],
        "light_text": "текст.",
        "readable_text": None,
        "variants": {},
        "variant_light": {},
    }


def _job(tmp_path, *, readable=None, asr_dropped=None, overlap=None, overlap_error=None):
    manifest = {
        "job_id": "20260101-000000-test",
        "meeting_name": "Тестовая встреча",
        "sources": [
            {
                "path": "/x/src/1.m4a",
                "duration": 15.0,
                "sha256": "abc",
                "offset": 0.0,
                "diarization_offset": 0.0,
            }
        ],
        "config": {
            "clean": ["raw", "light", "readable"],
            "asr_models": ["model_a", "model_b"],
            "asr_dropped_models": asr_dropped or [],
            "diarization_provider": "foxnose",
            "llm_model": "gpt-5.6-terra",
        },
        "asr": {"effective_device": "cpu"},
    }
    if readable is not None:
        manifest["readable"] = readable
    atomic_json(tmp_path / "manifest.json", manifest)
    canonical = {
        "utterances": [
            _utterance("p001-u00001", "SPEAKER_00", 0.0, 6.0),
            _utterance("p001-u00002", "SPEAKER_00", 6.0, 10.0),
            _utterance("p001-u00003", "SPEAKER_01", 10.0, 15.0),
        ],
        "diarization": {
            "provider": "foxnose",
            "num_speakers": 2,
            "overlap": overlap or [],
            "overlap_error": overlap_error,
        },
        "speaker_previews": {
            "SPEAKER_00": {
                "meeting_start": 0.4,
                "source_file": "/x/src/1.m4a",
                "text": "Всем привет, начнём",
            }
        },
    }
    atomic_json(tmp_path / "canonical.json", canonical)
    atomic_text(
        tmp_path / "speakers.yaml",
        "speakers:\n  SPEAKER_00: Игорь\n  SPEAKER_01: SPEAKER_01\n",
    )
    for model in ("model_a", "model_b"):
        (tmp_path / f"transcript.raw.{model}.md").write_text("raw", encoding="utf-8")
        (tmp_path / f"transcript.light.{model}.md").write_text("light", encoding="utf-8")
    return tmp_path


def test_build_job_result_basic_fields(tmp_path):
    job = _job(tmp_path)

    result = build_job_result(job)

    assert result["job_id"] == "20260101-000000-test"
    assert result["meeting_name"] == "Тестовая встреча"
    assert result["status"] == "complete"
    assert result["duration_seconds"] == 15.0
    assert result["sources"] == ["1.m4a"]
    assert result["utterances"] == 3


def test_build_job_result_speaker_speech_seconds_and_preview(tmp_path):
    job = _job(tmp_path)

    result = build_job_result(job)
    by_id = {item["id"]: item for item in result["speakers"]}

    assert by_id["SPEAKER_00"]["name"] == "Игорь"
    assert by_id["SPEAKER_00"]["speech_seconds"] == 10.0
    assert by_id["SPEAKER_00"]["preview"]["text"] == "Всем привет, начнём"
    assert by_id["SPEAKER_00"]["preview"]["meeting_start"] == 0.4
    assert by_id["SPEAKER_01"]["speech_seconds"] == 5.0
    assert "preview" not in by_id["SPEAKER_01"]


def test_build_job_result_warns_about_a_speaker_left_with_its_default_id_as_name(tmp_path):
    job = _job(tmp_path)

    result = build_job_result(job)

    assert any("SPEAKER_01" in warning for warning in result["warnings"])


def test_build_job_result_asr_and_diarization_blocks(tmp_path):
    job = _job(tmp_path)

    result = build_job_result(job)

    assert result["asr"]["models"] == ["model_a", "model_b"]
    assert result["asr"]["canonical_model"] == "model_a"
    assert result["asr"]["device"] == "cpu"
    assert result["diarization"]["provider"] == "foxnose"
    assert result["diarization"]["num_speakers"] == 2
    assert result["diarization"]["overlap_regions"] == 0


def test_build_job_result_transcripts_reflect_files_actually_on_disk(tmp_path):
    job = _job(tmp_path)

    result = build_job_result(job)

    assert result["transcripts"]["raw"] == {
        "model_a": "transcript.raw.model_a.md",
        "model_b": "transcript.raw.model_b.md",
    }
    assert result["transcripts"]["light"] == {
        "model_a": "transcript.light.model_a.md",
        "model_b": "transcript.light.model_b.md",
    }
    # readable was requested (clean includes it) but no file and no manifest entry exist.
    assert result["transcripts"]["readable"] is None
    assert any("readable" in warning for warning in result["warnings"])


def test_build_job_result_reports_a_completed_readable_pass(tmp_path):
    job = _job(
        tmp_path,
        readable={
            "status": "complete",
            "model": "gpt-5.6-terra",
            "changed_utterances": 3,
            "needs_attention": 1,
        },
    )
    (job / "transcript.readable.md").write_text("readable", encoding="utf-8")
    (job / "review").mkdir()
    (job / "review" / "readable_changes.csv").write_text("utterance_id\n", encoding="utf-8")

    result = build_job_result(job)

    assert result["transcripts"]["readable"] == "transcript.readable.md"
    assert result["readable"] == {
        "status": "complete",
        "model": "gpt-5.6-terra",
        "changed_utterances": 3,
        "needs_attention": 1,
        "report": "review/readable_changes.csv",
    }
    assert not any("readable" in warning for warning in result["warnings"])


def test_build_job_result_collects_warnings_for_degraded_conditions(tmp_path):
    job = _job(
        tmp_path,
        readable={"status": "failed", "model": "gpt-5.6-terra", "error": "no key"},
        asr_dropped=["groq-whisper-large-v3"],
        overlap=[[1.0, 2.0], [5.0, 6.0]],
        overlap_error="model unavailable",
    )

    result = build_job_result(job)

    assert result["readable"]["status"] == "failed"
    assert result["diarization"]["overlap_regions"] == 2
    joined = " | ".join(result["warnings"])
    assert "readable skipped" in joined
    assert "groq-whisper-large-v3" in joined
    assert "model unavailable" in joined


def test_write_job_result_writes_relative_paths_and_a_summary(tmp_path):
    job = _job(tmp_path)

    result = write_job_result(job)

    on_disk = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert on_disk == result
    for filename in {**result["transcripts"]["raw"], **result["transcripts"]["light"]}.values():
        assert "/" not in filename  # relative to the job dir, not an absolute path

    summary = (job / "summary.md").read_text(encoding="utf-8")
    assert "Тестовая встреча" in summary
    assert "Игорь" in summary
    # The summary must not duplicate the transcript body itself.
    assert "текст" not in summary.split("## Files")[0]


def test_render_summary_lists_participants_and_warnings_without_transcript_body(tmp_path):
    job = _job(tmp_path)
    result = build_job_result(job)

    summary = render_summary(result)

    assert "## Participants" in summary
    assert "## Warnings" in summary
    assert "SPEAKER_01" in summary


def test_unattributed_bucket_is_not_listed_as_a_participant(tmp_path):
    """UNKNOWN_SPEAKER is where unattributed words go; naming it makes no sense."""
    from transcribe_meeting.model import UNKNOWN_SPEAKER

    job = _job(tmp_path)
    canonical = read_json(job / "canonical.json")
    for item in canonical["utterances"]:
        item["speaker_id"] = UNKNOWN_SPEAKER
        item["speaker_name"] = UNKNOWN_SPEAKER
    atomic_json(job / "canonical.json", canonical)

    result = build_job_result(job)

    assert all(s["id"] != UNKNOWN_SPEAKER for s in result["speakers"])
    assert any("no speaker" in w for w in result["warnings"])
    # A real unnamed speaker may still be flagged; the bucket itself must not be.
    assert not any(UNKNOWN_SPEAKER in w and "name them" in w for w in result["warnings"])


def test_provider_error_is_collapsed_for_the_human_summary(tmp_path):
    """summary.md goes straight into chat; a raw SDK error dict must not."""
    job = _job(
        tmp_path,
        readable={"status": "failed", "model": "m", "error": "Error code: 429 - " + "x" * 500},
    )

    result = build_job_result(job)
    warning = next(w for w in result["warnings"] if "readable skipped" in w)

    assert len(warning) < 200
    assert warning.endswith("…")
