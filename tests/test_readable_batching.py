from contextlib import nullcontext

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
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.test/v1")
    for setting in ("API", "RESPONSE_FORMAT", "BATCH_SIZE", "REASONING"):
        monkeypatch.delenv(f"TRANSCRIBE_READABLE_{setting}", raising=False)
    monkeypatch.setattr("openai.OpenAI", lambda **_: nullcontext(object()))
    monkeypatch.setattr(R, "_fuse_batch", fuse)


def test_a_long_meeting_is_split_into_batches(monkeypatch):
    """One call for a two-hour meeting risks truncation, which fails the id check."""
    sizes = []

    def fuse(
        _client, _model, _instructions, batch, _protected, _variant_models=(), *, usage, settings
    ):
        sizes.append(len(batch))
        usage["input_tokens"] = usage.get("input_tokens", 0) + 10
        return R._BatchResult({i.utterance_id: "ок." for i in batch}, [], 0)

    _install(monkeypatch, fuse)
    items = [_u(i) for i in range(R.READABLE_BATCH_SIZE * 2 + 7)]

    result = R.clean_readable(items, "m", set())

    assert sizes == [R.READABLE_BATCH_SIZE, R.READABLE_BATCH_SIZE, 7]
    assert len(result.texts) == len(items)
    assert result.usage["input_tokens"] == 30
    assert result.endpoint_failure is None
    assert result.pending_utterance_ids is None


def test_one_failing_batch_keeps_the_others(monkeypatch):
    def fuse(_client, _model, _instructions, batch, _protected, _variant_models=(), **_settings):
        if batch[0].utterance_id == "u0000":
            raise R.UnusableReply("truncated response")
        return R._BatchResult({i.utterance_id: "ок." for i in batch}, [], 0)

    _install(monkeypatch, fuse)
    items = [_u(i) for i in range(R.READABLE_BATCH_SIZE + 5)]

    result = R.clean_readable(items, "m", set())

    assert result.texts["u0000"] == "реплика номер 0."
    assert result.texts[f"u{R.READABLE_BATCH_SIZE:04d}"] == "ок."
    assert result.pending_utterance_ids == [
        item.utterance_id for item in items[: R.READABLE_BATCH_SIZE]
    ]
    assert any("batch 1 failed validation" in c["note"] for c in result.concerns)


def test_every_batch_failing_is_reported_as_a_failure(monkeypatch):
    """A readable made entirely of local text must not pass as complete."""

    def fuse(*_args, **_settings):
        raise R.UnusableReply("truncated response")

    _install(monkeypatch, fuse)

    with pytest.raises(RuntimeError, match="failed validation in every batch"):
        R.clean_readable([_u(0)], "m", set())


def test_batch_progress_is_reported(monkeypatch):
    seen = []
    _install(
        monkeypatch,
        lambda _c, _m, _i, batch, _p, _v=(), **_settings: R._BatchResult(
            {i.utterance_id: "ок." for i in batch}, [], 0
        ),
    )

    R.clean_readable([_u(i) for i in range(3)], "m", set(), status=seen.append)

    assert seen == ["readable: batch 1/1 (3 utterances)"]
