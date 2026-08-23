import csv

import yaml

from transcribe_meeting.core import _load_utterances, prepare_review, refine
from transcribe_meeting.model import Utterance
from transcribe_meeting.util import atomic_json, atomic_text


def _job(tmp_path):
    utterances = [
        Utterance(
            utterance_id="p001-u00001",
            source_file="/x/a.m4a",
            source_start=0.0,
            source_end=1.0,
            meeting_start=0.0,
            meeting_end=1.0,
            speaker_id="SPEAKER_00",
            speaker_name="Иван",
            raw_text="текст один",
            confidence={"mean_word_probability": 0.3},
        ),
        Utterance(
            utterance_id="p001-u00002",
            source_file="/x/a.m4a",
            source_start=1.0,
            source_end=2.0,
            meeting_start=1.0,
            meeting_end=2.0,
            speaker_id="SPEAKER_01",
            speaker_name="Пётр",
            raw_text="текст два",
            confidence={"mean_word_probability": 0.9},
        ),
    ]
    atomic_json(
        tmp_path / "manifest.json",
        {
            "meeting_name": "test",
            "sources": [{"path": "/x/a.m4a"}],
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
        yaml.safe_dump(
            {"speakers": {"SPEAKER_00": "Иван", "SPEAKER_01": "Пётр"}}, allow_unicode=True
        ),
    )
    return tmp_path


def _write_review(job, rows):
    path = job / "review.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["utterance_id", "replacement", "speaker_replacement"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_prepare_review_includes_speaker_and_replacement_columns(tmp_path):
    job = _job(tmp_path)

    target = prepare_review(job)

    with target.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert {"speaker", "speaker_replacement"}.issubset(set(reader.fieldnames))
        assert "confidence" not in reader.fieldnames
        rows = list(reader)
    # No confidence-based filtering: every utterance is exported.
    assert len(rows) == 2
    assert rows[0]["speaker"] == "Иван"
    assert rows[0]["speaker_replacement"] == ""


def test_refine_known_speaker_replacement_changes_name_and_id(tmp_path):
    job = _job(tmp_path)
    review_csv = _write_review(
        job, [{"utterance_id": "p001-u00001", "replacement": "", "speaker_replacement": "Пётр"}]
    )

    refine(job, review_csv)

    item = {u.utterance_id: u for u in _load_utterances(job)}["p001-u00001"]
    assert item.speaker_name == "Пётр"
    assert item.speaker_id == "SPEAKER_01"


def test_refine_unknown_speaker_name_renames_without_changing_id(tmp_path):
    job = _job(tmp_path)
    review_csv = _write_review(
        job, [{"utterance_id": "p001-u00001", "replacement": "", "speaker_replacement": "Гость"}]
    )

    refine(job, review_csv)

    item = {u.utterance_id: u for u in _load_utterances(job)}["p001-u00001"]
    assert item.speaker_name == "Гость"
    assert item.speaker_id == "SPEAKER_00"


def test_refine_empty_speaker_replacement_leaves_speaker_untouched(tmp_path):
    job = _job(tmp_path)
    review_csv = _write_review(
        job,
        [{"utterance_id": "p001-u00001", "replacement": "новый текст", "speaker_replacement": ""}],
    )

    refine(job, review_csv)

    item = {u.utterance_id: u for u in _load_utterances(job)}["p001-u00001"]
    assert item.raw_text == "новый текст"
    assert item.speaker_name == "Иван"
    assert item.speaker_id == "SPEAKER_00"


def test_refine_text_and_speaker_apply_together_in_one_command(tmp_path):
    job = _job(tmp_path)
    review_csv = _write_review(
        job,
        [
            {
                "utterance_id": "p001-u00001",
                "replacement": "починенный текст",
                "speaker_replacement": "Пётр",
            }
        ],
    )

    refine(job, review_csv)

    item = {u.utterance_id: u for u in _load_utterances(job)}["p001-u00001"]
    assert item.raw_text == "починенный текст"
    assert item.speaker_name == "Пётр"
    assert item.speaker_id == "SPEAKER_01"
