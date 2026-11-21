import os
import stat

import pytest

from transcribe_meeting.readable import readable_readiness
from transcribe_meeting.secrets import load_project_secrets, save_project_secret


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_project_secrets_skip_non_regular_files(tmp_path, kind):
    path = tmp_path / "settings.env"
    if kind == "directory":
        path.mkdir(mode=0o755)
    else:
        os.mkfifo(path, mode=0o644)
    previous_mode = path.stat().st_mode
    environment = {}

    load_project_secrets(path, environment)

    assert environment == {}
    assert path.stat().st_mode == previous_mode


def test_saving_another_readable_origin_cannot_move_existing_project_key(tmp_path):
    path = tmp_path / "settings.env"
    save_project_secret("OPENAI_API_KEY", "test-key", path)
    original = path.read_bytes()

    with pytest.raises(ValueError, match="OPENAI_API_KEY") as error:
        save_project_secret("OPENAI_BASE_URL", "https://other.test/v1", path)

    assert "test-key" not in str(error.value)
    assert path.read_bytes() == original
    save_project_secret("OPENAI_BASE_URL", "https://api.openai.com:443/another-path", path)


def test_project_endpoint_cannot_redirect_shell_key(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_BASE_URL=https://private.test/where\nOPENAI_API_KEY=project-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_TRANSCRIBE_OPENAI_ENV_CONFLICT", raising=False)
    load_project_secrets(path)

    state = readable_readiness()
    assert not state.ready
    assert state.host is None
    assert "OPENAI_BASE_URL" in state.diagnostic
    assert "OPENAI_API_KEY" in state.diagnostic
    assert ".env" in state.diagnostic
    assert "shell-key" not in repr(state) and "private.test" not in repr(state)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://intended.test/v1")
    assert readable_readiness().ready


def test_shell_endpoint_cannot_repurpose_a_key_for_another_project_origin(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_BASE_URL=https://provider.test/v1\nOPENAI_API_KEY=project-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.test/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("_TRANSCRIBE_OPENAI_ENV_CONFLICT", raising=False)
    load_project_secrets(path)
    state = readable_readiness()
    assert not state.ready
    assert state.host == "other.test"
    assert state.diagnostic is not None and "OPENAI_API_KEY from .env" in state.diagnostic
    assert "project-key" not in repr(state)


def test_blank_shell_values_do_not_override_a_coherent_project_pair(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_BASE_URL=https://provider.test/v1\nOPENAI_API_KEY=project-key\n",
        encoding="utf-8",
    )
    environment = {"OPENAI_BASE_URL": " ", "OPENAI_API_KEY": ""}
    load_project_secrets(path, environment)
    assert environment == {
        "OPENAI_BASE_URL": "https://provider.test/v1",
        "OPENAI_API_KEY": "project-key",
    }


def test_explicit_shell_loopback_does_not_load_project_key(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=project-key\n", encoding="utf-8")
    environment = {"OPENAI_BASE_URL": "http://127.0.0.1:8000/v1"}
    load_project_secrets(path, environment)
    assert "OPENAI_API_KEY" not in environment


def test_same_explicit_origin_can_pair_with_project_key(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_BASE_URL=https://provider.test/v1\nOPENAI_API_KEY=project-key\n",
        encoding="utf-8",
    )
    environment = {"OPENAI_BASE_URL": "https://PROVIDER.test:443/another-route"}
    load_project_secrets(path, environment)
    assert environment["OPENAI_API_KEY"] == "project-key"


def test_project_secrets_are_owner_only_and_replace_existing_value(tmp_path):
    path = tmp_path / ".env"

    save_project_secret("GROQ_API_KEY", "first", path)
    save_project_secret("GROQ_API_KEY", "second", path)

    assert path.read_text(encoding="utf-8") == "GROQ_API_KEY=second\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_loader_only_uses_supported_names_and_keeps_shell_values(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=project-openai\nGROQ_API_KEY='project-groq'\n"
        "OPENAI_BASE_URL=https://provider.test/v1\nOTHER=value\n",
        encoding="utf-8",
    )
    path.chmod(0o644)
    environment = {"OPENAI_API_KEY": "shell-openai", "OPENAI_BASE_URL": "https://shell.test/v1"}

    load_project_secrets(path, environment)

    assert environment == {
        "OPENAI_API_KEY": "shell-openai",
        "GROQ_API_KEY": "project-groq",
        "OPENAI_BASE_URL": "https://shell.test/v1",
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_loader_accepts_only_explicit_readable_and_mirror_settings(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "TRANSCRIBE_READABLE_API=chat\n"
        "TRANSCRIBE_READABLE_RESPONSE_FORMAT=json_object\n"
        "TRANSCRIBE_READABLE_BATCH_SIZE=30\n"
        "TRANSCRIBE_READABLE_REASONING=none\n"
        "TRANSCRIBE_GIGAAM_BASE_URL=https://mirror.example/\n"
        "TRANSCRIBE_OTHER=not-allowed\n",
        encoding="utf-8",
    )
    environment = {"TRANSCRIBE_READABLE_BATCH_SIZE": "50"}
    load_project_secrets(path, environment)
    assert environment == {
        "TRANSCRIBE_READABLE_API": "chat",
        "TRANSCRIBE_READABLE_RESPONSE_FORMAT": "json_object",
        "TRANSCRIBE_READABLE_BATCH_SIZE": "50",
        "TRANSCRIBE_READABLE_REASONING": "none",
        "TRANSCRIBE_GIGAAM_BASE_URL": "https://mirror.example/",
    }


def test_save_rejects_newline_to_prevent_env_injection(tmp_path):
    try:
        save_project_secret("OPENAI_API_KEY", "valid\nINJECTED=value", tmp_path / ".env")
    except ValueError as exc:
        assert "line break" in str(exc)
    else:
        raise AssertionError("expected a validation error")


def test_hf_token_round_trips_through_project_env(tmp_path):
    """setup reads HF_TOKEN from the environment, so .env must actually feed it there."""
    path = tmp_path / ".env"
    save_project_secret("HF_TOKEN", "hf_example", path=path)
    environ: dict[str, str] = {}
    load_project_secrets(path, environ)

    assert environ["HF_TOKEN"] == "hf_example"
    assert path.stat().st_mode & 0o077 == 0


def test_empty_hf_token_placeholder_does_not_define_the_variable(tmp_path):
    """A blank `HF_TOKEN=` line is a placeholder, not a configured empty token."""
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=\n", encoding="utf-8")
    environ: dict[str, str] = {}
    load_project_secrets(path, environ)

    assert "HF_TOKEN" not in environ


def test_deepgram_api_key_round_trips_through_project_env(tmp_path):
    """The loader must actually pick up DEEPGRAM_API_KEY from .env,
    or the key never reaches the deepgram-nova-3/--diarization-provider deepgram code."""
    path = tmp_path / ".env"
    save_project_secret("DEEPGRAM_API_KEY", "dg_example", path=path)
    environ: dict[str, str] = {}
    load_project_secrets(path, environ)

    assert environ["DEEPGRAM_API_KEY"] == "dg_example"
    assert path.stat().st_mode & 0o077 == 0


def test_deepgram_key_round_trips_through_project_env(tmp_path):
    """The key sat in .env unread once already; the whitelist must include it."""
    path = tmp_path / ".env"
    save_project_secret("DEEPGRAM_API_KEY", "dg_example", path=path)
    environ: dict[str, str] = {}
    load_project_secrets(path, environ)

    assert environ["DEEPGRAM_API_KEY"] == "dg_example"
