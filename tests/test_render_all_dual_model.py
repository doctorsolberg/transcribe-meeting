from transcribe_meeting.core import render_all
from transcribe_meeting.model import Utterance
from transcribe_meeting.util import atomic_json


def _utterance(index: int, raw: str, variants: dict[str, str] | None = None) -> Utterance:
    variants = variants or {}
    return Utterance(
        utterance_id=f"p001-u{index:05d}",
        source_file="/x/a.m4a",
        source_start=float(index),
        source_end=float(index) + 1.0,
        meeting_start=float(index),
        meeting_end=float(index) + 1.0,
        speaker_id="SPEAKER_00",
        speaker_name="SPEAKER_00",
        raw_text=raw,
        confidence={},
        light_text=raw,
        variants=dict(variants),
        variant_light=dict(variants),
    )


def _job(tmp_path, utterances, asr_models):
    atomic_json(
        tmp_path / "manifest.json",
        {
            "meeting_name": "test",
            "sources": [{"path": "/x/a.m4a"}],
            "config": {"clean": ["raw", "light"], "asr_models": asr_models},
        },
    )
    atomic_json(
        tmp_path / "canonical.json",
        {"schema_version": 2, "utterances": [u.to_dict() for u in utterances]},
    )
    return tmp_path


def test_two_models_produce_four_named_files(tmp_path):
    job = _job(
        tmp_path,
        [_utterance(1, "модель а", {"model_b": "модель б"})],
        ["model_a", "model_b"],
    )

    render_all(job)

    assert (job / "transcript.raw.model_a.md").exists()
    assert (job / "transcript.light.model_a.md").exists()
    assert (job / "transcript.raw.model_b.md").exists()
    assert (job / "transcript.light.model_b.md").exists()
    assert not (job / "transcript.md").exists()
    assert not (job / "transcript.raw.md").exists()


def test_one_model_produces_only_its_own_two_files(tmp_path):
    job = _job(tmp_path, [_utterance(1, "текст")], ["model_a"])

    render_all(job)

    assert (job / "transcript.raw.model_a.md").exists()
    assert (job / "transcript.light.model_a.md").exists()
    assert not (job / "transcript.raw.model_b.md").exists()
    assert not (job / "transcript.md").exists()


def test_alt_file_never_falls_back_to_canonical_text(tmp_path):
    """The whole point is line-by-line comparability: a missing variant entry must
    render as an absent line, never silently borrow model A's text."""
    job = _job(
        tmp_path,
        [_utterance(1, "у модели а есть слово")],
        ["model_a", "model_b"],
    )

    render_all(job)

    body = (job / "transcript.raw.model_b.md").read_text(encoding="utf-8")
    assert "слово" not in body
    body_a = (job / "transcript.raw.model_a.md").read_text(encoding="utf-8")
    assert "слово" in body_a


def test_three_models_produce_six_named_files(tmp_path):
    job = _job(
        tmp_path,
        [_utterance(1, "модель а", {"model_b": "модель б", "model_c": "модель в"})],
        ["model_a", "model_b", "model_c"],
    )

    render_all(job)

    for model in ("model_a", "model_b", "model_c"):
        assert (job / f"transcript.raw.{model}.md").exists()
        assert (job / f"transcript.light.{model}.md").exists()
    body_c = (job / "transcript.raw.model_c.md").read_text(encoding="utf-8")
    assert "модель в" in body_c


def test_third_model_missing_a_word_omits_only_its_own_line(tmp_path):
    """model_c has no words for this utterance at all; model_b still renders."""
    job = _job(
        tmp_path,
        [_utterance(1, "у канонической есть слово", {"model_b": "у второй есть слово"})],
        ["model_a", "model_b", "model_c"],
    )

    render_all(job)

    assert "слово" not in (job / "transcript.raw.model_c.md").read_text(encoding="utf-8")
    assert "слово" in (job / "transcript.raw.model_b.md").read_text(encoding="utf-8")
