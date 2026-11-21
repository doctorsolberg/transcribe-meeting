import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transcribe_meeting import paths
from transcribe_meeting.cli import app

runner = CliRunner()


def test_data_root_defaults_to_cwd_and_output_models_glossary_follow_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(paths.ENV_SECRETS_FILE, raising=False)
    assert paths.data_root() == tmp_path
    assert paths.output_dir() == tmp_path / "output"
    assert paths.models_dir() == tmp_path / "models"
    assert paths.glossary_path() == tmp_path / "glossary.yaml"
    assert paths.secrets_path() == tmp_path / ".env"


def test_env_vars_override_defaults_read_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path / "data"))
    assert paths.data_root() == tmp_path / "data"
    assert paths.output_dir() == tmp_path / "data" / "output"
    monkeypatch.setenv(paths.ENV_OUTPUT_DIR, str(tmp_path / "elsewhere"))
    assert paths.output_dir() == tmp_path / "elsewhere"
    monkeypatch.setenv(paths.ENV_MODELS_DIR, str(tmp_path / "models-elsewhere"))
    assert paths.models_dir() == tmp_path / "models-elsewhere"
    monkeypatch.setenv(paths.ENV_SECRETS_FILE, str(tmp_path / "custom.env"))
    assert paths.secrets_path() == tmp_path / "custom.env"


def test_output_and_models_dir_stay_relative_to_data_root_when_only_it_is_set(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))
    assert paths.output_dir() == tmp_path / "output"
    assert paths.models_dir() == tmp_path / "models"


def test_resolve_model_path_keeps_absolute_values_as_is(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))
    outside = tmp_path / "outside-cache" / "vad"
    assert paths.resolve_model_path(str(outside)) == outside


def test_resolve_model_path_resolves_relative_values_against_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))
    assert paths.resolve_model_path("gigaam") == tmp_path / "models" / "gigaam"
    assert paths.resolve_model_path("a/b") == tmp_path / "models" / "a" / "b"


def test_store_model_path_stores_paths_inside_models_dir_as_relative(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))
    inside = paths.models_dir() / "gigaam"
    assert paths.store_model_path(inside) == "gigaam"


def test_store_model_path_stores_paths_outside_models_dir_as_absolute(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))
    outside = tmp_path / "somewhere-else" / "vad"
    assert paths.store_model_path(outside) == str(outside.resolve())


def test_lock_path_prefers_the_new_canonical_location(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "models.lock.json").write_text("{}", encoding="utf-8")
    (tmp_path / "models.lock.json").write_text("{}", encoding="utf-8")

    assert paths.lock_path() == tmp_path / "models" / "models.lock.json"


def test_lock_path_falls_back_to_the_legacy_project_root_location(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))
    (tmp_path / "models.lock.json").write_text("{}", encoding="utf-8")

    assert paths.lock_path() == tmp_path / "models.lock.json"


def test_lock_path_defaults_to_the_new_location_when_neither_exists(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))

    assert paths.lock_path() == tmp_path / "models" / "models.lock.json"


def test_lock_path_returns_a_path_object(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path))
    assert isinstance(paths.lock_path(), Path)


def test_cli_data_root_option_overrides_where_doctor_looks_for_models(tmp_path, monkeypatch):
    # The app callback sets these as real process env vars, which
    # CliRunner's own isolation does not undo. `setenv` (not `delenv(raising=False)`)
    # is required here: monkeypatch only registers a teardown action for a key it
    # already holds a prior value for, so delenv on an absent key is a silent no-op
    # and would leave whatever the CLI callback sets leaking into later tests —
    # exactly what made this test order-dependent before this fix.
    monkeypatch.setenv(paths.ENV_DATA_ROOT, "placeholder-overwritten-by-cli-flag")
    monkeypatch.delenv(paths.ENV_OUTPUT_DIR, raising=False)
    monkeypatch.delenv(paths.ENV_MODELS_DIR, raising=False)

    result = runner.invoke(app, ["--data-root", str(tmp_path), "doctor"])

    # doctor checks the default route's actual readiness rather
    # than a fixed informational list, so an empty data root (no models downloaded
    # anywhere under it) correctly fails route_ready and exits 1 — this is the same
    # "which data root doctor looked under" signal the old gigaam_local/vad_local
    # informational booleans gave, just through the new keys.
    checks, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert checks["asr_models"] == {"multilingual_large_ctc": False, "v3_rnnt": False}
    assert checks["vad_ready"] is False
    assert checks["route_ready"] is False
    assert result.exit_code == 1


def test_cli_models_dir_option_is_independent_of_data_root(tmp_path, monkeypatch):
    # See the comment in the previous test: setenv (not delenv) is what actually
    # guarantees these get cleaned up after the CLI callback overwrites them.
    monkeypatch.setenv(paths.ENV_DATA_ROOT, "placeholder-overwritten-by-cli-flag")
    monkeypatch.delenv(paths.ENV_OUTPUT_DIR, raising=False)
    monkeypatch.setenv(paths.ENV_MODELS_DIR, "placeholder-overwritten-by-cli-flag")
    models_dir = tmp_path / "models-elsewhere"
    (models_dir / "gigaam").mkdir(parents=True)
    (models_dir / "gigaam" / "v3_rnnt.ckpt").write_bytes(b"stub checkpoint")
    (models_dir / "models.lock.json").write_text(
        json.dumps({"models": {"gigaam": {"path": "gigaam"}}}), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "--data-root",
            str(tmp_path),
            "--models-dir",
            str(models_dir),
            "doctor",
            "--asr-models",
            "v3_rnnt",
        ],
    )

    checks, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert checks["asr_models"] == {"v3_rnnt": True}


@pytest.mark.parametrize("custom_file", [False, True])
@pytest.mark.parametrize("shell_root", [None, ""])
def test_cli_uses_dotenv_data_root(tmp_path, monkeypatch, custom_file, shell_root):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(paths.ENV_SECRETS_FILE, raising=False)
    if shell_root is None:
        monkeypatch.delenv(paths.ENV_DATA_ROOT, raising=False)
    else:
        monkeypatch.setenv(paths.ENV_DATA_ROOT, shell_root)
    data = tmp_path / "meeting data"
    settings = tmp_path / ("custom.env" if custom_file else ".env")
    settings.write_text(f'TRANSCRIBE_HOME="{data}"\n', encoding="utf-8")
    if custom_file:
        monkeypatch.setenv(paths.ENV_SECRETS_FILE, str(settings))
    checkpoints = data / "models" / "gigaam"
    checkpoints.mkdir(parents=True)
    (checkpoints / "v3_rnnt.ckpt").write_bytes(b"stub checkpoint")

    result = runner.invoke(app, ["doctor", "--asr-models", "v3_rnnt"])

    assert json.loads(result.stdout)["asr_models"] == {"v3_rnnt": True}
    assert paths.output_dir() == data / "output"


@pytest.mark.parametrize("use_flag", [False, True])
def test_cli_flag_and_shell_override_dotenv_data_root(tmp_path, monkeypatch, use_flag):
    settings = tmp_path / "custom.env"
    settings.write_text(f"TRANSCRIBE_HOME={tmp_path / 'from-dotenv'}\n", encoding="utf-8")
    monkeypatch.setenv(paths.ENV_SECRETS_FILE, str(settings))
    shell = tmp_path / "from-shell"
    flag = tmp_path / "from-flag"
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(shell))
    flags = ["--data-root", str(flag)] if use_flag else []

    result = runner.invoke(app, [*flags, "glossary", "list"])

    assert result.exit_code == 0, result.output
    assert paths.data_root() == (flag if use_flag else shell)


def test_auth_writes_to_the_dotenv_that_selected_the_data_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(paths.ENV_SECRETS_FILE, raising=False)
    monkeypatch.setenv(paths.ENV_DATA_ROOT, "")
    monkeypatch.delenv(paths.ENV_DATA_ROOT)
    monkeypatch.setenv("HF_TOKEN", "")
    data = tmp_path / "data"
    settings = tmp_path / ".env"
    settings.write_text(f"TRANSCRIBE_HOME={data}\n", encoding="utf-8")
    monkeypatch.setattr("transcribe_meeting.cli.getpass", lambda _: "hf_example")

    result = runner.invoke(app, ["auth", "set", "huggingface"])

    assert result.exit_code == 0, result.output
    assert paths.data_root() == data
    assert settings.read_text(encoding="utf-8") == f"TRANSCRIBE_HOME={data}\nHF_TOKEN=hf_example\n"
    assert not (data / ".env").exists()
