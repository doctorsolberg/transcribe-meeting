import pytest
import typer
from typer.testing import CliRunner

from transcribe_meeting.cli import (
    _existing_speaker_names,
    _parse_speaker_names,
    _resolve_speaker_name_overrides,
    app,
)
from transcribe_meeting.util import atomic_json, atomic_text

runner = CliRunner()


def test_parse_speaker_names_reads_comma_separated_pairs():
    assert _parse_speaker_names("SPEAKER_00=Игорь,SPEAKER_01=Руслан") == {
        "SPEAKER_00": "Игорь",
        "SPEAKER_01": "Руслан",
    }


def test_parse_speaker_names_rejects_a_pair_without_equals():
    with pytest.raises(typer.BadParameter):
        _parse_speaker_names("SPEAKER_00")


def test_parse_speaker_names_rejects_an_empty_name():
    with pytest.raises(typer.BadParameter):
        _parse_speaker_names("SPEAKER_00=")


def test_resolve_speaker_name_overrides_is_a_noop_for_an_empty_string():
    assert _resolve_speaker_name_overrides("", {"SPEAKER_00": "SPEAKER_00"}) == {}


def test_resolve_speaker_name_overrides_applies_when_the_count_matches():
    found = {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"}

    result = _resolve_speaker_name_overrides("SPEAKER_00=Игорь", found)

    assert result == {"SPEAKER_00": "Игорь"}


def test_resolve_speaker_name_overrides_rejects_more_names_than_found_speakers():
    found = {"SPEAKER_00": "SPEAKER_00"}

    with pytest.raises(typer.BadParameter, match=r"2 name\(s\) given"):
        _resolve_speaker_name_overrides("SPEAKER_00=Игорь,SPEAKER_01=Руслан", found)


def test_resolve_speaker_name_overrides_rejects_an_unknown_speaker_id():
    found = {"SPEAKER_00": "SPEAKER_00"}

    with pytest.raises(typer.BadParameter, match="unknown speaker_id"):
        _resolve_speaker_name_overrides("SPEAKER_99=Игорь", found)


def _job(tmp_path):
    atomic_json(
        tmp_path / "manifest.json",
        {
            "job_id": tmp_path.name,
            "meeting_name": "test",
            "sources": [{"path": "/x/a.m4a", "duration": 1.0}],
            "config": {"clean": ["raw"], "asr_models": ["model_a"]},
        },
    )
    atomic_json(
        tmp_path / "canonical.json",
        {
            "schema_version": 2,
            "utterances": [
                {
                    "utterance_id": "p001-u00001",
                    "source_file": "/x/a.m4a",
                    "source_start": 0.0,
                    "source_end": 1.0,
                    "meeting_start": 0.0,
                    "meeting_end": 1.0,
                    "speaker_id": "SPEAKER_00",
                    "speaker_name": "SPEAKER_00",
                    "raw_text": "текст",
                    "confidence": {},
                    "overlap": False,
                    "uncertain_spans": [],
                    "light_text": "текст.",
                    "readable_text": None,
                    "variants": {},
                    "variant_light": {},
                }
            ],
            "diarization": {},
        },
    )
    atomic_text(tmp_path / "speakers.yaml", "speakers:\n  SPEAKER_00: SPEAKER_00\n")
    (tmp_path / "transcript.raw.model_a.md").write_text("x", encoding="utf-8")
    return tmp_path


def test_existing_speaker_names_reads_speakers_yaml(tmp_path):
    job = _job(tmp_path)

    assert _existing_speaker_names(job) == {"SPEAKER_00": "SPEAKER_00"}


def test_cli_rename_with_set_works_without_a_tty(tmp_path, monkeypatch):
    # An isolated TRANSCRIBE_HOME keeps the app callback's load_project_secrets()
    # from reading this developer's real project .env into the test process.
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    job = _job(tmp_path / "job")

    result = runner.invoke(app, ["rename", str(job), "--set", "SPEAKER_00=Игорь"])

    assert result.exit_code == 0, result.stdout
    assert (job / "result.json").exists()


def test_cli_rename_without_set_and_without_a_tty_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    job = _job(tmp_path / "job")

    result = runner.invoke(app, ["rename", str(job)])

    assert result.exit_code != 0
