import stat

from transcribe_meeting.secrets import load_project_secrets, save_project_secret


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
    environment = {"OPENAI_API_KEY": "shell-openai"}

    load_project_secrets(path, environment)

    assert environment == {
        "OPENAI_API_KEY": "shell-openai",
        "GROQ_API_KEY": "project-groq",
        "OPENAI_BASE_URL": "https://provider.test/v1",
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


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
