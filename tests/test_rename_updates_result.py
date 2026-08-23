import json

from transcribe_meeting.core import _load_utterances, rename
from transcribe_meeting.model import Utterance
from transcribe_meeting.util import atomic_json, atomic_text


def _job(tmp_path):
    utterances = [
        Utterance(
            utterance_id="p001-u00001",
            source_file="/x/a.m4a",
            source_start=0.0,
            source_end=3.0,
            meeting_start=0.0,
            meeting_end=3.0,
            speaker_id="SPEAKER_00",
            speaker_name="SPEAKER_00",
            raw_text="текст один",
            confidence={},
            light_text="текст один.",
        ),
        Utterance(
            utterance_id="p001-u00002",
            source_file="/x/a.m4a",
            source_start=3.0,
            source_end=5.0,
            meeting_start=3.0,
            meeting_end=5.0,
            speaker_id="SPEAKER_01",
            speaker_name="SPEAKER_01",
            raw_text="текст два",
            confidence={},
            light_text="текст два.",
        ),
    ]
    atomic_json(
        tmp_path / "manifest.json",
        {
            "job_id": tmp_path.name,
            "meeting_name": "test rename",
            "sources": [{"path": "/x/a.m4a", "duration": 5.0}],
            "config": {"clean": ["raw", "light"], "asr_models": ["model_a"]},
        },
    )
    atomic_json(
        tmp_path / "canonical.json",
        {
            "schema_version": 2,
            "utterances": [item.to_dict() for item in utterances],
            "diarization": {},
        },
    )
    atomic_text(
        tmp_path / "speakers.yaml",
        "speakers:\n  SPEAKER_00: SPEAKER_00\n  SPEAKER_01: SPEAKER_01\n",
    )
    (tmp_path / "transcript.raw.model_a.md").write_text("исходно", encoding="utf-8")
    (tmp_path / "transcript.light.model_a.md").write_text("исходно", encoding="utf-8")
    return tmp_path


def test_rename_updates_speaker_names_and_rerenders_all_transcripts(tmp_path):
    job = _job(tmp_path)

    rename(job, {"SPEAKER_00": "Игорь"})

    item = {u.utterance_id: u for u in _load_utterances(job)}["p001-u00001"]
    assert item.speaker_name == "Игорь"
    raw = (job / "transcript.raw.model_a.md").read_text(encoding="utf-8")
    light = (job / "transcript.light.model_a.md").read_text(encoding="utf-8")
    assert "Игорь" in raw
    assert "Игорь" in light


def test_rename_writes_result_json_and_summary_md(tmp_path):
    job = _job(tmp_path)

    rename(job, {"SPEAKER_00": "Игорь"})

    assert (job / "result.json").exists()
    assert (job / "summary.md").exists()
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in result["speakers"]}
    assert by_id["SPEAKER_00"]["name"] == "Игорь"
    assert "Игорь" in (job / "summary.md").read_text(encoding="utf-8")


def test_rename_called_again_refreshes_the_already_written_result(tmp_path):
    job = _job(tmp_path)
    rename(job, {"SPEAKER_00": "Игорь"})
    first = json.loads((job / "result.json").read_text(encoding="utf-8"))

    rename(job, {"SPEAKER_00": "Игорь Z"})

    second = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert first["speakers"] != second["speakers"]
    by_id = {item["id"]: item for item in second["speakers"]}
    assert by_id["SPEAKER_00"]["name"] == "Игорь Z"
    assert "Игорь Z" in (job / "summary.md").read_text(encoding="utf-8")


def test_refine_also_refreshes_the_job_result(tmp_path, monkeypatch):
    """refine rewrites text and speakers, so a stale result.json would misreport."""
    import csv

    from transcribe_meeting.core import refine

    job = _job(tmp_path)
    (job / "result.json").unlink(missing_ok=True)
    review = job / "review" / "transcript_edit.csv"
    review.parent.mkdir(parents=True, exist_ok=True)
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utterance_id",
                "timestamp",
                "speaker",
                "text",
                "context",
                "suggestion",
                "replacement",
                "speaker_replacement",
            ],
        )
        writer.writeheader()
        writer.writerow({"utterance_id": "p001-u00001", "replacement": "исправленный текст"})

    refine(job, review)

    assert (job / "result.json").exists()
