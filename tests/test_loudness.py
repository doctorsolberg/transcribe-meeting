from pathlib import Path

from transcribe_meeting import audio


def _fake_probe(monkeypatch, calls: list[str]):
    monkeypatch.setattr(audio, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(
        audio,
        "probe",
        lambda path, stream=None: {"path": str(path), "duration": 12.0, "stream_index": 0},
    )
    monkeypatch.setattr(
        audio, "measure_loudness", lambda path: calls.append(str(path)) or {"input_i": -29.4}
    )


def test_loudness_is_skipped_by_default(tmp_path, monkeypatch):
    """The measurement costs a full decode pass, so it must not run unasked."""
    calls: list[str] = []
    _fake_probe(monkeypatch, calls)

    values = audio.diagnostics([Path("a.m4a")], tmp_path / "d.json", None)

    assert calls == []
    assert "loudness" not in values[0]


def test_loudness_runs_only_when_requested(tmp_path, monkeypatch):
    calls: list[str] = []
    _fake_probe(monkeypatch, calls)

    values = audio.diagnostics([Path("a.m4a")], tmp_path / "d.json", None, True)

    assert calls == ["a.m4a"]
    assert values[0]["loudness"] == {"input_i": -29.4}


def test_measure_loudness_asks_ffmpeg_at_info_level(monkeypatch):
    """loudnorm prints its JSON summary at info; a quieter level yields nothing."""
    seen: dict[str, list[str]] = {}

    class Result:
        stderr = '{"input_i": "-29.4"}'

    def fake_run(args: list[str]):
        seen["args"] = args
        return Result()

    monkeypatch.setattr(audio, "run_command", fake_run)

    assert audio.measure_loudness(Path("a.m4a")) == {"input_i": "-29.4"}
    assert seen["args"][seen["args"].index("-v") + 1] == "info"


def test_unparseable_ffmpeg_output_is_reported_not_dropped(monkeypatch):
    """A silent missing key was the old bug; an explicit warning replaces it."""

    class Result:
        stderr = "no json here"

    monkeypatch.setattr(audio, "run_command", lambda args: Result())

    assert audio.measure_loudness(Path("a.m4a")) == {"warning": "measurement unavailable"}
