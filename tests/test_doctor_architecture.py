import json
import platform

from typer.testing import CliRunner

from transcribe_meeting.cli import app

runner = CliRunner()


def test_doctor_does_not_fail_on_a_non_apple_silicon_machine(monkeypatch):
    """Architecture is informational only — doctor must run
    unmodified on a Linux/CPU-only server, so it must never gate on arm64."""
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    result = runner.invoke(app, ["doctor"])
    checks = json.loads(result.stdout)

    assert checks["apple_silicon"] is False
    # Architecture must never appear among the reasons doctor reports as blocking, and
    # the exit code is asserted against the checks doctor itself gates on rather than
    # against 0: on a machine without models (CI, a fresh checkout) route_ready is
    # legitimately false, and hard-coding 0 would make this test pass only here.
    gating = ("python_supported", "ffmpeg", "free_space_ge_7gb", "route_ready")
    assert result.exit_code == (0 if all(checks[key] for key in gating) else 1)


def test_doctor_survives_a_data_root_that_does_not_exist_yet(monkeypatch, tmp_path):
    """doctor runs before setup; a missing TRANSCRIBE_HOME must not raise a traceback.

    The env vars are pinned here because the global --data-root option writes them into
    os.environ, and CliRunner does not roll that back for the tests that follow.
    """
    import json

    from typer.testing import CliRunner

    from transcribe_meeting.cli import app

    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(tmp_path / "models"))

    result = CliRunner().invoke(
        app, ["--data-root", "/tmp/definitely-absent-transcribe-home", "doctor"]
    )

    assert "Traceback" not in result.output
    assert json.loads(result.stdout)["free_space_ge_7gb"] is not None
