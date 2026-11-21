"""Command-level tests: invoke the actual `transcribe-meeting` commands through
CliRunner and check what a user (or an agent driving the CLI) would actually see —
exit code, stdout/stderr split, and message wording — rather than the internal
helper functions cli.py happens to be built from.

Every test is pinned to an isolated TRANSCRIBE_HOME/TRANSCRIBE_OUTPUT_DIR/
TRANSCRIBE_MODELS_DIR under tmp_path and strips every provider API key, so the
result never depends on what is actually downloaded or configured on the machine
running the suite (see `_isolated_environment` below). None of these scenarios ever
reach a network call or a model load: they are all supposed to fail in CLI-level
validation before create_job runs, so nothing here mocks network/model code.
"""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transcribe_meeting.cli import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: Unicode box-drawing block: the frame rich puts around its panels and tables.
_BOX = re.compile(r"[\u2500-\u257f]")


def plain(output: str) -> str:
    """Strip rich's presentation so an assertion checks the message, not the layout.

    Rich colours its error boxes and hard-wraps them to the terminal width, and it
    makes both choices differently on a CI runner than on a developer's machine —
    which is exactly how four of these assertions passed locally and failed in CI.
    Escape codes land inside styled option names; a wrap splits a substring and
    leaves the box border sitting in the middle of it ("setup │ │ --asr-models").
    """
    without_box = _BOX.sub("", _ANSI.sub("", output))
    return re.sub(r"\s+", " ", without_box)


SECRET_ENV_NAMES = ("OPENAI_API_KEY", "GROQ_API_KEY", "DEEPGRAM_API_KEY", "HF_TOKEN")

ALL_COMMANDS = (
    "run",
    "setup",
    "doctor",
    "status",
    "resume",
    "clean",
    "rename",
    "review",
    "refine",
    "glossary",
    "auth",
)


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch):
    """Pin every path env var and strip every secret for every test in this file.

    Two failure modes have already leaked between tests in this project without
    this: (1) the --data-root/--output-dir/--models-dir globals write straight into
    os.environ (see cli.main) and CliRunner never rolls that back — `setenv` (not
    `delenv`) is required here because monkeypatch only restores a key it has
    itself touched at least once, and a bare `delenv` on an already-absent key
    registers nothing to undo; (2) a real .env or real downloaded models on the
    developer's machine could make a "no key"/"no model" scenario here pass locally
    and fail in CI.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("TRANSCRIBE_HOME", str(home))
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_DIR", str(home / "output"))
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(home / "models"))
    for name in SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return home


def _ready_models_dir(base: Path) -> Path:
    """Stub just enough of models_dir for route_readiness() to report the default
    route (multilingual_large_ctc + v3_rnnt ASR, foxnose diarization) as ready,
    without any models.lock.json: route_readiness falls back to models_dir()/gigaam,
    models_dir()/huggingface and models_dir()/wespeaker/en/model.onnx when the lock
    has no entry, and only checks file presence — never content — so an empty stub
    is enough to get past preflight into the option being tested.
    """
    models = base / "ready-models"
    gigaam = models / "gigaam"
    gigaam.mkdir(parents=True)
    (gigaam / "multilingual_large_ctc.ckpt").write_bytes(b"stub")
    (gigaam / "v3_rnnt.ckpt").write_bytes(b"stub")
    vad = models / "huggingface" / "models--pyannote--segmentation-3.0"
    snapshot = vad / "snapshots" / "test-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "pytorch_model.bin").write_bytes(b"stub")
    (vad / "refs").mkdir()
    (vad / "refs" / "main").write_text("test-revision", encoding="utf-8")
    wespeaker = models / "wespeaker" / "en"
    wespeaker.mkdir(parents=True)
    (wespeaker / "model.onnx").write_bytes(b"stub")
    return models


# --- 1. --help for every command exits 0 ------------------------------------------


@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_help_exits_zero(command):
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0, result.output


# --- 2. doctor without any models ---------------------------------------------------


def test_doctor_without_models_reports_not_ready_with_advice_only_on_stderr():
    result = runner.invoke(app, ["doctor"])

    # stdout must be JSON and nothing but JSON, so a caller can parse it
    # unconditionally; remediation advice belongs on stderr instead (see
    # cli.doctor's own comment on the print/CONSOLE.print split).
    checks = json.loads(result.stdout)
    assert checks["route_ready"] is False
    assert result.exit_code == 1
    assert "Traceback" not in plain(result.output)
    assert "setup" in result.stderr


def test_setup_missing_data_root_reaches_huggingface_hint(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr("transcribe_meeting.cli.check_ffmpeg", lambda: None)
    monkeypatch.setattr("transcribe_meeting.cli._free_space_bytes", lambda _: 8 * 1024**3)
    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 2, result.output
    assert "segmentation-3.0" in plain(result.stderr)
    assert "speaker-diarization-community-1" not in plain(result.stderr)


def test_setup_does_not_publish_a_lock_if_smoke_fails(monkeypatch, tmp_path):
    from transcribe_meeting import provisioning

    models = tmp_path / "model-cache"
    models.mkdir()
    lock = models / "models.lock.json"
    original = b'{"models": {"old": {}}}'
    lock.write_bytes(original)
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(models))
    monkeypatch.setenv("HF_TOKEN", "test-placeholder")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr("transcribe_meeting.cli.check_ffmpeg", lambda: None)
    monkeypatch.setattr("transcribe_meeting.cli._free_space_bytes", lambda _: 8 * 1024**3)
    monkeypatch.setattr(provisioning, "check_huggingface_access", lambda **kw: None)
    vad = models / "vad"
    vad.mkdir()
    (vad / "pytorch_model.bin").write_bytes(b"fixture")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: str(vad))
    monkeypatch.setattr(provisioning, "download_gigaam_models", lambda *a, **kw: (models, {}))
    monkeypatch.setattr(provisioning, "download_diarization_models", lambda **kw: ({}, None, None))

    def fail_smoke(**kwargs):
        raise RuntimeError("simulated offline model failure")

    monkeypatch.setattr(provisioning, "run_smoke_tests", fail_smoke)
    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 2, result.output
    assert "simulated offline model failure" in plain(result.stderr)
    assert lock.read_bytes() == original


def test_doctor_rejects_empty_huggingface_cache(tmp_path):
    models = _ready_models_dir(tmp_path)
    cache = models / "huggingface" / "models--pyannote--segmentation-3.0"
    (cache / "snapshots" / "test-revision" / "pytorch_model.bin").unlink()
    result = runner.invoke(app, ["--models-dir", str(models), "doctor"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["vad_ready"] is False


# --- 3. preflight rejections: each exits 2 with a message naming the fix -----------


def test_run_rejects_an_undownloaded_local_asr_model():
    result = runner.invoke(app, ["run", "missing.wav", "--meeting-name", "t"])

    assert result.exit_code == 2
    assert "setup --asr-models" in plain(result.output)


def test_run_rejects_pyannote_diarization_without_community_1():
    result = runner.invoke(
        app, ["run", "missing.wav", "--meeting-name", "t", "--diarization-provider", "pyannote"]
    )

    assert result.exit_code == 2
    assert "setup --diarization-provider pyannote" in plain(result.output)


def test_run_degrades_a_missing_deepgram_asr_key_with_a_warning_not_a_refusal():
    """An ASR model without its key is dropped with a warning (like readable's own
    missing-key handling) — contrast with the diarization-deepgram case below, which
    is a hard refusal instead. deepgram-nova-3 is the only model requested, so it
    ends up as the "no models left" BadParameter — but the warning proves it was
    degraded first, not rejected outright the way diarization-deepgram is."""
    result = runner.invoke(
        app, ["run", "missing.wav", "--meeting-name", "t", "--asr-models", "deepgram-nova-3"]
    )

    assert "ASR: deepgram-nova-3 skipped" in plain(result.output)
    assert "DEEPGRAM_API_KEY is missing" in plain(result.output)
    # It was dropped (leaving zero models), not refused outright: the eventual
    # error is the "no models left" message, not the diarization-style hard block.
    assert "No ASR models are left" in plain(result.output)
    assert result.exit_code == 2


def test_run_refuses_deepgram_diarization_without_a_key():
    result = runner.invoke(
        app, ["run", "missing.wav", "--meeting-name", "t", "--diarization-provider", "deepgram"]
    )

    assert result.exit_code == 2
    assert "auth set deepgram" in plain(result.output)
    assert "DEEPGRAM_API_KEY" in plain(result.output)


def test_run_rejects_more_than_three_asr_models():
    result = runner.invoke(
        app,
        [
            "run",
            "missing.wav",
            "--meeting-name",
            "t",
            "--asr-models",
            "multilingual_large_ctc,v3_rnnt,v3_ctc,v2_rnnt",
        ],
    )

    assert result.exit_code == 2
    assert "No more than three" in plain(result.output)


def test_run_rejects_an_unknown_asr_model_name():
    result = runner.invoke(
        app, ["run", "missing.wav", "--meeting-name", "t", "--asr-models", "bogus-model"]
    )

    assert result.exit_code == 2
    assert "Unknown ASR models" in plain(result.output)


def test_run_rejects_zero_speakers(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(_ready_models_dir(tmp_path)))

    result = runner.invoke(app, ["run", "missing.wav", "--meeting-name", "t", "--speakers", "0"])

    assert result.exit_code == 2
    assert "must be at least 1" in plain(result.output)


def test_run_rejects_a_nonexistent_input_file_once_the_route_is_ready(tmp_path, monkeypatch):
    """With a ready route, the file-existence check (not route readiness) is what
    must reject this — see also the job-directory test below."""
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(_ready_models_dir(tmp_path)))

    result = runner.invoke(app, ["run", "definitely-missing.wav", "--meeting-name", "t"])

    assert result.exit_code == 2
    assert "Missing files" in plain(result.output)
    assert "definitely-missing.wav" in plain(result.output)


# --- 4. run with valid args but missing audio creates no job directory ------------


def test_run_with_missing_audio_creates_no_job_directory(tmp_path, monkeypatch):
    ready_models = _ready_models_dir(tmp_path)
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(ready_models))
    output_dir = tmp_path / "home" / "output"

    result = runner.invoke(app, ["run", "still-missing.wav", "--meeting-name", "t"])

    assert result.exit_code == 2
    assert not output_dir.exists() or not any(output_dir.iterdir())


# --- 5. rename/refine on a nonexistent job: a clear error, not a traceback --------


def test_rename_on_a_nonexistent_job_is_a_clear_error_not_a_traceback(tmp_path):
    missing_job = tmp_path / "home" / "output" / "does-not-exist"

    result = runner.invoke(app, ["rename", str(missing_job), "--set", "SPEAKER_00=Igor"])

    assert "Traceback" not in plain(result.output)
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_refine_on_a_nonexistent_job_is_a_clear_error_not_a_traceback(tmp_path):
    missing_job = tmp_path / "home" / "output" / "does-not-exist"
    review_csv = tmp_path / "review.csv"
    review_csv.write_text("utterance_id,replacement,speaker_replacement\n", encoding="utf-8")

    result = runner.invoke(app, ["refine", str(missing_job), str(review_csv)])

    assert "Traceback" not in plain(result.output)
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    # Not the path: rich breaks a long unbroken token mid-word to fit its box, so no
    # amount of normalising makes a filesystem path a reliable substring. The wording
    # of the message is what this test is actually about.
    assert "Job folder not found" in plain(result.output)
    assert "not found" in plain(result.output)


def test_refine_on_an_existing_job_with_a_missing_csv_is_a_bad_parameter_not_a_traceback(
    tmp_path,
):
    job_dir = tmp_path / "home" / "output" / "real-job"
    job_dir.mkdir(parents=True)
    missing_csv = tmp_path / "no-such-review.csv"

    result = runner.invoke(app, ["refine", str(job_dir), str(missing_csv)])

    assert "Traceback" not in plain(result.output)
    assert "FileNotFoundError" not in plain(result.output)
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)


def _assert_clear_error_not_a_traceback(result):
    assert "Traceback" not in plain(result.output)
    assert "FileNotFoundError" not in plain(result.output)
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "not found" in plain(result.output)


def test_clean_on_a_nonexistent_job_is_a_clear_error_not_a_traceback(tmp_path):
    missing_job = tmp_path / "home" / "output" / "does-not-exist"

    result = runner.invoke(app, ["clean", str(missing_job), "--mode", "raw"])

    _assert_clear_error_not_a_traceback(result)


def test_review_on_a_nonexistent_job_is_a_clear_error_not_a_traceback(tmp_path):
    missing_job = tmp_path / "home" / "output" / "does-not-exist"

    result = runner.invoke(app, ["review", str(missing_job)])

    _assert_clear_error_not_a_traceback(result)


def test_resume_on_a_nonexistent_job_is_a_clear_error_not_a_traceback(tmp_path):
    missing_job = tmp_path / "home" / "output" / "does-not-exist"

    result = runner.invoke(app, ["resume", str(missing_job)])

    _assert_clear_error_not_a_traceback(result)


# --- 6. status on an old-format manifest (no asr_models key) ----------------------


def test_status_on_an_old_format_manifest_gives_a_readable_report_not_a_keyerror(tmp_path):
    job_dir = tmp_path / "home" / "output" / "old-job"
    job_dir.mkdir(parents=True)
    manifest = {
        "job_id": "old-job",
        # Deliberately missing "asr_models" (and everything else newer manifests
        # carry): status must not assume any key beyond job_id/readable exists.
        "config": {"language": "ru"},
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = runner.invoke(app, ["status", str(job_dir)])

    assert "Traceback" not in plain(result.output)
    assert "KeyError" not in plain(result.output)
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["job"] == "old-job"
    assert payload["readable"] == {"status": "not requested"}


def test_status_stdout_is_valid_json_even_with_a_long_readable_error(tmp_path):
    """Reproduces the actual incident: rich's print wraps stdout at 80 columns, which
    turns a long readable.error into invalid JSON. status must use typer.echo instead."""
    job_dir = tmp_path / "home" / "output" / "long-error-job"
    job_dir.mkdir(parents=True)
    long_error = "readable did not pass validation: " + "x" * 250
    manifest = {
        "job_id": "long-error-job",
        "config": {"language": "ru"},
        "readable": {"status": "failed", "error": long_error},
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = runner.invoke(app, ["status", str(job_dir)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["readable"]["error"] == long_error


def test_doctor_stdout_is_valid_json():
    result = runner.invoke(app, ["doctor"])

    payload = json.loads(result.stdout)
    assert "route_ready" in payload


def test_a_usage_error_after_the_run_stays_a_usage_error(tmp_path, monkeypatch):
    """`run`'s broad failure handler must not swallow typer's own control flow.

    typer.BadParameter derives from Exception, so without an explicit re-raise a bad
    --speaker-names became "The run stopped with an error" plus an error.log — a crash
    report for what is only a usage mistake.
    """
    from transcribe_meeting import cli

    job_holder: dict[str, Path] = {}

    def fake_create_job(_parts, name, config, _probes):
        job = tmp_path / "job"
        (job / ".stages").mkdir(parents=True)
        job_holder["job"] = job
        (tmp_path / "home" / "output").mkdir(parents=True, exist_ok=True)
        # The real create_job writes the manifest before run returns the job, and
        # everything run does afterwards (run_job, the readable pass) reads it back.
        (job / "manifest.json").write_text(
            json.dumps({"job_id": job.name, "meeting_name": name, "config": config}),
            encoding="utf-8",
        )
        return job

    monkeypatch.setattr(
        cli,
        "diagnostics",
        lambda parts, target, *_a, **_k: (
            target.write_text("{}", encoding="utf-8"),
            [{"duration": 1.0}],
        )[1],
    )
    monkeypatch.setattr(cli, "create_job", fake_create_job)
    monkeypatch.setattr(cli, "run_job", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_check_route_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.shutil, "move", lambda *_a, **_k: None)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    result = runner.invoke(
        app,
        [
            "run",
            str(audio),
            "--meeting-name",
            "test",
            "--clean",
            "raw",
            "--speaker-names",
            "SPEAKER_42=Nobody",
        ],
    )

    # Exit code 2 is click's usage error; 1 would mean the broad handler caught it.
    assert result.exit_code == 2, result.output
    assert "--speaker-names" in plain(result.output)
    assert not (job_holder["job"] / "error.log").exists()


def test_run_refuses_when_the_working_files_would_not_fit(tmp_path, monkeypatch):
    """A run that runs out of disk halfway leaves a half-decoded job and an opaque
    ffmpeg error; setup's fixed 7 GB check says nothing about this meeting's length."""
    from transcribe_meeting import cli

    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    monkeypatch.setattr(cli, "_check_route_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "diagnostics", lambda parts, target, *_a, **_k: [{"duration": 7200.0}])
    monkeypatch.setattr(cli, "_free_space_bytes", lambda _path: 100 * 1024**2)

    result = runner.invoke(app, ["run", str(audio), "--meeting-name", "test", "--clean", "raw"])

    assert result.exit_code == 2, result.output
    assert "Not enough space" in plain(result.output)
