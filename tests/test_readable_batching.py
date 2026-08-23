import pytest

from transcribe_meeting import readable as R
from transcribe_meeting.model import Utterance


def _u(index: int) -> Utterance:
    return Utterance(
        utterance_id=f"u{index:04d}",
        source_file="a.m4a",
        source_start=0.0,
        source_end=1.0,
        meeting_start=0.0,
        meeting_end=1.0,
        speaker_id="S1",
        speaker_name="S1",
        raw_text=f"реплика номер {index}",
        confidence={},
        light_text=f"реплика номер {index}.",
    )


def _install(monkeypatch, fuse):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(R, "OpenAI", lambda **_: object(), raising=False)
    monkeypatch.setattr("openai.OpenAI", lambda **_: object())
    monkeypatch.setattr(R, "_fuse_batch", fuse)


def test_a_long_meeting_is_split_into_batches(monkeypatch):
    """One call for a two-hour meeting risks truncation, which fails the id check."""
    sizes = []

    def fuse(_client, _model, _instructions, batch, _protected, _variant_models=()):
        sizes.append(len(batch))
        return ({i.utterance_id: "ок." for i in batch}, [], {"input_tokens": 10})

    _install(monkeypatch, fuse)
    items = [_u(i) for i in range(R.READABLE_BATCH_SIZE * 2 + 7)]

    result = R.clean_readable(items, "m", set())

    assert sizes == [R.READABLE_BATCH_SIZE, R.READABLE_BATCH_SIZE, 7]
    assert len(result.texts) == len(items)
    assert result.usage["input_tokens"] == 30


def test_one_failing_batch_keeps_the_others(monkeypatch):
    def fuse(_client, _model, _instructions, batch, _protected, _variant_models=()):
        if batch[0].utterance_id == "u0000":
            raise ValueError("truncated response")
        return ({i.utterance_id: "ок." for i in batch}, [], {})

    _install(monkeypatch, fuse)
    items = [_u(i) for i in range(R.READABLE_BATCH_SIZE + 5)]

    result = R.clean_readable(items, "m", set())

    assert result.texts["u0000"] == "реплика номер 0."
    assert result.texts[f"u{R.READABLE_BATCH_SIZE:04d}"] == "ок."
    assert any("batch 1 failed validation" in c["note"] for c in result.concerns)


def test_every_batch_failing_is_reported_as_a_failure(monkeypatch):
    """A readable made entirely of local text must not pass as complete."""

    def fuse(*_args):
        raise ValueError("truncated response")

    _install(monkeypatch, fuse)

    with pytest.raises(RuntimeError, match="failed validation in every batch"):
        R.clean_readable([_u(0)], "m", set())


def test_batch_progress_is_reported(monkeypatch):
    seen = []
    _install(
        monkeypatch,
        lambda _c, _m, _i, batch, _p, _v=(): ({i.utterance_id: "ок." for i in batch}, [], {}),
    )

    R.clean_readable([_u(i) for i in range(3)], "m", set(), status=seen.append)

    assert seen == ["readable: batch 1/1 (3 utterances)"]
