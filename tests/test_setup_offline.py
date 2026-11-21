"""CLI setup against a synthetic model cache; no real weights, API keys, or network."""

import hashlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from transcribe_meeting import asr, cli, provisioning


def _unexpected_network(*_args, **_kwargs):
    pytest.fail("setup tried to reach the network")


@pytest.fixture
def setup_cache(tmp_path, monkeypatch):
    models = tmp_path / "models"
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(models))
    monkeypatch.setenv("TRANSCRIBE_ENV_FILE", str(tmp_path / "no-secrets"))
    for name in ("HF_TOKEN", "GROQ_API_KEY", "DEEPGRAM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "_free_space_bytes", lambda _path: 8 * 1024**3)
    monkeypatch.setattr(socket.socket, "connect", _unexpected_network)
    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", _unexpected_network)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _unexpected_network)
    return models


def _snapshot(models, repository, files):
    repo = models / "huggingface" / ("models--" + repository.replace("/", "--"))
    revision = "a" * 40
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    snapshot = repo / "snapshots" / revision
    for index, filename in enumerate(files):
        blob = repo / "blobs" / str(index)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"local model fixture")
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(blob)
    return snapshot


def test_setup_reuses_complete_hf_cache_without_a_token(setup_cache, monkeypatch):
    models = setup_cache
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    checkpoint = models / "gigaam" / asr.ASR_MODELS["v3_rnnt"]["checkpoint"]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setitem(asr.ASR_MODELS["v3_rnnt"], "md5", hashlib.md5(b"fixture").hexdigest())
    vad = _snapshot(models, provisioning.VAD_REPOSITORY, ("pytorch_model.bin",))
    community = _snapshot(models, provisioning.PYANNOTE_REPO, provisioning.COMMUNITY1_FILES)
    monkeypatch.setattr(provisioning, "run_smoke_tests", lambda **_: {"tested": True})

    result = CliRunner().invoke(
        cli.app, ["setup", "--asr-models", "v3_rnnt", "--diarization-provider", "pyannote"]
    )

    assert result.exit_code == 0, result.output
    lock = json.loads((models / "models.lock.json").read_text(encoding="utf-8"))
    assert lock["models"]["vad"]["path"] == vad.relative_to(models).as_posix()
    assert lock["models"]["pyannote"]["path"] == community.relative_to(models).as_posix()
    assert lock["smoke"] == {"tested": True}


def test_setup_requests_access_only_for_missing_gated_model(setup_cache, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    _snapshot(setup_cache, provisioning.VAD_REPOSITORY, ("pytorch_model.bin",))
    snapshot = _snapshot(setup_cache, provisioning.PYANNOTE_REPO, provisioning.COMMUNITY1_FILES)
    (snapshot / "embedding/pytorch_model.bin").unlink()

    result = CliRunner().invoke(
        cli.app, ["setup", "--asr-models", "v3_rnnt", "--diarization-provider", "pyannote"]
    )

    assert result.exit_code == 2, result.output
    assert "HF_TOKEN" in result.stderr
    assert provisioning.PYANNOTE_REPO in result.stderr
    assert provisioning.VAD_REPOSITORY not in result.stderr
    assert not (setup_cache / "models.lock.json").exists()


@pytest.mark.parametrize("with_vad", [False, True])
def test_network_asr_can_opt_in_to_vad(setup_cache, monkeypatch, with_vad):
    models = setup_cache
    if with_vad:
        _snapshot(models, provisioning.VAD_REPOSITORY, ("pytorch_model.bin",))
    model = models / "wespeaker" / provisioning.WESPEAKER_LANG / "model.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx fixture")
    monkeypatch.setattr(provisioning, "_wespeaker_sha256", lambda _: "verified")
    monkeypatch.setattr(provisioning, "run_smoke_tests", lambda **_: {})

    args = ["setup", "--asr-models", asr.DEEPGRAM_ASR_MODEL_NAME]
    if with_vad:
        args.append("--with-vad")
    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    lock = json.loads((models / "models.lock.json").read_text(encoding="utf-8"))["models"]
    assert lock[asr.DEEPGRAM_ASR_MODEL_NAME]["provider"] == "deepgram"
    assert ("vad" in lock) is with_vad
    assert "gigaam" not in lock


def test_setup_reuses_community_from_legacy_lock(setup_cache, tmp_path, monkeypatch):
    external = tmp_path / "copied-community"
    for filename in provisioning.COMMUNITY1_FILES:
        cp = external / filename
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(b"model fixture")
    # An HF checkout can store the config as a symlink to a shared blob. Its
    # siblings are resolved relative to the config's lexical location.
    blob = tmp_path / "shared-config-blob"
    (external / "config.yaml").rename(blob)
    (external / "config.yaml").symlink_to(blob)
    legacy_lock = {"models": {"pyannote": {"path": str(external / "config.yaml")}}}
    (tmp_path / "models.lock.json").write_text(json.dumps(legacy_lock), encoding="utf-8")
    monkeypatch.setattr(provisioning, "run_smoke_tests", lambda **_: {})

    result = CliRunner().invoke(
        cli.app,
        [
            "setup",
            "--asr-models",
            asr.DEEPGRAM_ASR_MODEL_NAME,
            "--diarization-provider",
            "pyannote",
        ],
    )

    assert result.exit_code == 0, result.output
    lock = json.loads((setup_cache / "models.lock.json").read_text(encoding="utf-8"))["models"]
    assert lock["pyannote"]["path"] == str(external / "config.yaml")


def test_setup_reuses_recorded_gigaam_and_wespeaker_locations(setup_cache, tmp_path, monkeypatch):
    models = setup_cache
    models.mkdir()
    gigaam = tmp_path / "gigaam-archive"
    gigaam.mkdir()
    checkpoint = gigaam / asr.ASR_MODELS["v3_rnnt"]["checkpoint"]
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setitem(asr.ASR_MODELS["v3_rnnt"], "md5", hashlib.md5(b"fixture").hexdigest())
    wespeaker = tmp_path / "foxnose.onnx"
    wespeaker.write_bytes(b"fixture")
    lock = models / "models.lock.json"
    lock.write_text(
        json.dumps(
            {"models": {"gigaam": {"path": str(gigaam)}, "wespeaker": {"path": str(wespeaker)}}}
        ),
        encoding="utf-8",
    )
    _snapshot(models, provisioning.VAD_REPOSITORY, ("pytorch_model.bin",))
    monkeypatch.setattr(provisioning, "_wespeaker_sha256", lambda _: "verified")
    monkeypatch.setattr(provisioning, "run_smoke_tests", lambda **_: {})
    monkeypatch.setattr(cli, "_free_space_bytes", lambda _path: 0)

    result = CliRunner().invoke(cli.app, ["setup", "--asr-models", "v3_rnnt"])

    assert result.exit_code == 0, result.output
    records = json.loads(lock.read_text(encoding="utf-8"))["models"]
    assert records["gigaam"]["path"] == str(gigaam)
    assert records["wespeaker"]["path"] == str(wespeaker)


def test_cached_setup_rerun_requires_no_writes_to_model_cache(setup_cache, monkeypatch):
    models = setup_cache
    _snapshot(models, provisioning.PYANNOTE_REPO, provisioning.COMMUNITY1_FILES)
    pipeline = SimpleNamespace(to=lambda _: None)
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=lambda _: pipeline)),
    )
    args = [
        "setup",
        "--diarization-provider",
        "pyannote",
        "--asr-models",
        asr.DEEPGRAM_ASR_MODEL_NAME,
    ]
    runner = CliRunner()
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    lock = models / "models.lock.json"
    previous = lock.stat()
    make_temp = provisioning.tempfile.TemporaryDirectory

    def read_only_directory(*args, **kwargs):
        directory = kwargs.get("dir")
        if directory is not None and Path(directory).is_relative_to(models):
            raise PermissionError("read-only model cache")
        return make_temp(*args, **kwargs)

    def no_cache_write(*_args):
        pytest.fail("unchanged cache lock must not be rewritten")

    monkeypatch.setattr(provisioning.tempfile, "TemporaryDirectory", read_only_directory)
    monkeypatch.setattr(provisioning, "atomic_json", no_cache_write)
    monkeypatch.setattr(cli, "_free_space_bytes", lambda _path: 0)
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert lock.stat().st_mtime_ns == previous.st_mtime_ns
