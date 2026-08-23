"""Model provisioning for `setup`: downloads, models.lock.json bookkeeping, and the
offline smoke tests that prove a download actually works.

This used to live inside cli.py, which meant none of it could be unit-tested without
going through Typer's CliRunner and a real network. It has no typer dependency of its
own — errors that should abort `setup` with a user-facing message are raised as
RuntimeError, and cli.setup converts those into typer.BadParameter.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
import wave
from pathlib import Path

from . import paths
from .asr import ASR_MODELS, GIGAAM_COMMIT, VAD_REPOSITORY, effective_asr_device
from .diarization import PYANNOTE_REPO, WESPEAKER_LANG, WESPEAKER_MODEL_URL
from .util import atomic_json, read_json


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_revision(path: str) -> str | None:
    cache = Path(path) / ".cache" / "huggingface" / "download"
    refs = list(cache.rglob("*.metadata")) if cache.exists() else []
    return refs[0].read_text().splitlines()[1] if refs else None


def download_gigaam_models(
    requested_local_models: list[str],
) -> tuple[Path, dict[str, dict[str, str]]]:
    """Download requested local GigaAM checkpoints (models_dir() must already exist).

    Returns (gigaam_dir, records for models newly downloaded or verified this run) —
    the caller merges those records with whatever models.lock.json already has.
    """
    gigaam_dir = paths.models_dir() / "gigaam"
    gigaam_dir.mkdir(exist_ok=True)
    downloaded_gigaam_models: dict[str, dict[str, str]] = {}
    for model_name in requested_local_models:
        info = ASR_MODELS[model_name]
        checkpoint = gigaam_dir / info["checkpoint"]
        if not checkpoint.exists():
            from urllib.request import urlretrieve

            # Same atomicity discipline as the WeSpeaker download below: a torn-off
            # download must never leave a truncated file at the final path, or the
            # `checkpoint.exists()` check above would wrongly skip re-download next run.
            fd, tmp_name = tempfile.mkstemp(dir=gigaam_dir, suffix=".ckpt.part")
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                urlretrieve(
                    f"https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/{info['checkpoint']}",
                    tmp_path,
                )
                os.replace(tmp_path, checkpoint)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        model_md5 = file_digest(checkpoint, "md5")
        if model_md5 != info["md5"]:
            raise RuntimeError(
                f"MD5 of GigaAM checkpoint {model_name} did not match upstream; setup stopped."
            )
        downloaded_gigaam_models[model_name] = {
            "checkpoint": paths.store_model_path(checkpoint),
            "md5": model_md5,
            "sha256": file_digest(checkpoint, "sha256"),
        }
    return gigaam_dir, downloaded_gigaam_models


def download_diarization_models(
    *, want_pyannote: bool, want_foxnose: bool, token: str, snapshot_download
) -> tuple[dict[str, object], str | None, Path | None]:
    """Download the requested diarization backend(s).

    Returns (new models.lock.json entries for pyannote/wespeaker/diarize, the
    pyannote snapshot path, the wespeaker onnx path) — the two paths feed the smoke
    tests that run afterwards.
    """
    new_entries: dict[str, object] = {}
    pyannote_path: str | None = None
    if want_pyannote:
        pyannote_path = snapshot_download(
            PYANNOTE_REPO,
            local_dir=paths.models_dir() / "speaker-diarization-community-1",
            token=token,
        )
        new_entries["pyannote"] = {
            "repository": PYANNOTE_REPO,
            "path": paths.store_model_path(Path(pyannote_path)),
            "revision": snapshot_revision(pyannote_path),
        }
    wespeaker_path: Path | None = None
    if want_foxnose:
        wespeaker_dir = paths.models_dir() / "wespeaker" / WESPEAKER_LANG
        wespeaker_dir.mkdir(parents=True, exist_ok=True)
        wespeaker_path = wespeaker_dir / "model.onnx"
        if not wespeaker_path.exists():
            from urllib.request import urlretrieve

            # Download to a same-directory temp file and os.replace into place — the same
            # discipline as atomic_json/atomic_text in util.py. Upstream publishes no checksum
            # for this file, so atomicity is the only defence against a torn-off download: a
            # truncated file at the final path would both poison models.lock.json's sha256 and
            # block re-download on the next setup, since `wespeaker_path.exists()` would already
            # be true.
            fd, tmp_name = tempfile.mkstemp(dir=wespeaker_dir, suffix=".onnx.part")
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                urlretrieve(WESPEAKER_MODEL_URL, tmp_path)
                os.replace(tmp_path, wespeaker_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        new_entries["wespeaker"] = {
            "repository": "wespeaker/voxceleb_resnet34_LM",
            "url": WESPEAKER_MODEL_URL,
            "path": paths.store_model_path(wespeaker_path),
            "sha256": file_digest(wespeaker_path, "sha256"),
            "lang": WESPEAKER_LANG,
        }
        new_entries["diarize"] = {
            "package": "diarize",
            "version": importlib.metadata.version("diarize"),
        }
    return new_entries, pyannote_path, wespeaker_path


def asr_lock_entries(
    gigaam_dir: Path,
    downloaded_gigaam_models: dict[str, dict[str, str]],
    vad: str,
    requested_network_models: list[str],
) -> dict[str, object]:
    """Build the gigaam/vad/groq models.lock.json entries, merging gigaam's own
    models map with whatever this project already recorded: setting
    up a different --asr-models set must not forget a previously downloaded model.
    """
    existing_lock_for_merge = (
        read_json(paths.lock_path()) if paths.lock_path().exists() else {"models": {}}
    )
    existing_gigaam_models = (
        existing_lock_for_merge.get("models", {}).get("gigaam", {}).get("models", {})
    )
    new_models: dict[str, object] = {
        "gigaam": {
            "repository": "https://github.com/salute-developers/GigaAM",
            "commit": GIGAAM_COMMIT,
            "path": paths.store_model_path(gigaam_dir),
            "models": {**existing_gigaam_models, **downloaded_gigaam_models},
        },
        "vad": {
            "repository": VAD_REPOSITORY,
            "path": paths.store_model_path(Path(vad)),
            "revision": snapshot_revision(vad),
            "parameters": {"min_duration_on": 0.0, "min_duration_off": 0.0},
        },
    }
    # Nothing to download for a groq or deepgram model (both run over the network at
    # request time), but the lock still records provenance for them, exactly like
    # every local model above — so a finished job's manifest can point back to what
    # produced it.
    for model_name in requested_network_models:
        new_models[model_name] = {
            "provider": ASR_MODELS[model_name]["provider"],
            "api_model": ASR_MODELS[model_name]["api_model"],
        }
    return new_models


def write_lock(new_models: dict[str, object]) -> Path:
    """Merge into whatever models.lock.json already has (old or new location, plan
    section 2), but always write to the new canonical location: a provider-scoped
    setup run must not erase records written by a previous run for the other
    provider, and every setup run naturally migrates the file onto the new layout.
    """
    lock_target = paths.models_dir() / "models.lock.json"
    existing_lock = read_json(paths.lock_path()) if paths.lock_path().exists() else {"models": {}}
    atomic_json(
        lock_target, {**existing_lock, "models": {**existing_lock.get("models", {}), **new_models}}
    )
    return lock_target


def run_smoke_tests(
    *,
    requested_local_models: list[str],
    requested_asr_models: list[str],
    gigaam_dir: Path,
    vad_path: str,
    want_pyannote: bool,
    want_foxnose: bool,
    pyannote_path: str | None,
    wespeaker_path: Path | None,
) -> dict[str, object]:
    """Prove every downloaded model loads and runs fully offline, without uploading
    user audio. Never returns partial results: any failure raises RuntimeError."""
    with tempfile.TemporaryDirectory(dir=paths.models_dir(), prefix=".setup-smoke-") as temporary:
        smoke_wav = Path(temporary) / "silence.wav"
        with wave.open(str(smoke_wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 16_000)
        from .asr import GigaAMBackend, detect_speech_regions

        device = effective_asr_device("auto")
        # One model loaded and closed at a time, same discipline as run_job: this
        # machine does not fit two GigaAM instances at once (3.1 GB RSS measured for
        # the largest checkpoint, on 16 GB). The groq model has
        # nothing to smoke-test locally — there is no backend to load offline.
        for model_name in requested_local_models:
            backend = GigaAMBackend(gigaam_dir, device, model_name)
            try:
                backend.transcribe(smoke_wav)
            finally:
                backend.close()
        detect_speech_regions(smoke_wav, Path(vad_path))
        new_smoke: dict[str, object] = {
            "asr": "ok",
            "asr_models": list(requested_asr_models),
            "vad": "ok",
            "device": device,
        }
        if want_pyannote:
            try:
                import torch
                from pyannote.audio import Pipeline

                Pipeline.from_pretrained(str(pyannote_path)).to(torch.device("cpu"))
            except Exception as exc:
                raise RuntimeError(f"Community-1 smoke test failed: {exc}") from exc
            new_smoke["diarization_load"] = "ok"
        if want_foxnose:
            import wespeakerruntime

            from . import diarization as diarization_module
            from .diarization import _diarize_foxnose

            try:
                # Silence makes diarize() return right after VAD, before extract_embeddings
                # ever runs (see diarize/src/diarize/__init__.py) — so _diarize_foxnose alone
                # never opens the WeSpeaker ONNX session and would smoke-test-pass even with a
                # missing/corrupt model file. Load the session directly through the shim first
                # to actually prove the model path and onnxruntime work offline.
                with diarization_module._local_wespeaker(wespeaker_path, 1):
                    wespeakerruntime.Speaker(lang=WESPEAKER_LANG)
                # Silence legitimately yields zero segments; only a raised exception fails setup.
                _diarize_foxnose(
                    smoke_wav,
                    Path(temporary) / "diarization.json",
                    Path(temporary) / "diarization.rttm",
                    None,
                    1,
                    8,
                    1,
                    wespeaker_path,
                )
            except Exception as exc:
                raise RuntimeError(f"foxnose smoke test failed: {exc}") from exc
            new_smoke["diarization_foxnose"] = "ok"
    return new_smoke
