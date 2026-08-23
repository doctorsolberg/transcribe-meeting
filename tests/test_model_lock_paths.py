import json

from transcribe_meeting import core


def _write_lock(path, models):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": models}), encoding="utf-8")


def test_model_lock_reads_the_new_canonical_location(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    _write_lock(tmp_path / "models" / "models.lock.json", {"gigaam": {"path": "gigaam"}})

    assert core.model_lock()["models"]["gigaam"]["path"] == "gigaam"


def test_model_lock_returns_empty_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))

    assert core.model_lock() == {"models": {}}


def test_model_lock_falls_back_to_legacy_location_and_warns_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    _write_lock(tmp_path / "models.lock.json", {"gigaam": {"path": "/abs/gigaam"}})
    core._LOCK_MIGRATION_WARNED.clear()
    messages: list[str] = []
    monkeypatch.setattr(core, "report_step", messages.append)

    result = core.model_lock()
    assert result["models"]["gigaam"]["path"] == "/abs/gigaam"
    assert len(messages) == 1
    assert "models.lock.json" in messages[0]

    # A second call for the same legacy path must not warn again.
    core.model_lock()
    assert len(messages) == 1


def test_model_path_resolves_a_relative_recorded_value_against_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))

    resolved = core._model_path({"path": "gigaam"}, "unused-default")

    assert resolved == tmp_path / "models" / "gigaam"


def test_model_path_keeps_an_absolute_recorded_value_as_is(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    absolute = tmp_path / "external-cache" / "vad"

    assert core._model_path({"path": str(absolute)}, "unused-default") == absolute


def test_model_path_falls_back_to_models_dir_default_when_no_path_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))

    assert core._model_path({}, "gigaam") == tmp_path / "models" / "gigaam"
