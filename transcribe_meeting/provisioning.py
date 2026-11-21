"""Model provisioning for `setup`: downloads, models.lock.json bookkeeping, and the
offline smoke tests that prove a download actually works.

This used to live inside cli.py, which meant none of it could be unit-tested without
going through Typer's CliRunner and a real network. It has no typer dependency of its
own — errors that should abort `setup` with a user-facing message are raised as
RuntimeError, and cli.setup converts those into typer.BadParameter.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.metadata
import logging
import math
import os
import re
import ssl
import tempfile
import time
import wave
from collections.abc import Callable
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path
from threading import RLock
from urllib import error, parse, request

from . import paths
from .asr import ASR_MODELS, GIGAAM_COMMIT, VAD_REPOSITORY, effective_asr_device
from .diarization import COMMUNITY1_FILES, PYANNOTE_REPO, WESPEAKER_LANG, WESPEAKER_MODEL_URL
from .util import atomic_json, read_json

GIGAAM_BASE_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/"
GIGAAM_MIRROR_ENV = "TRANSCRIBE_GIGAAM_BASE_URL"
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_PROGRESS_SECONDS = 2.0
Progress = Callable[[str], None]

_HF_METADATA_TIMEOUT_SECONDS = 10
_HF_LOG_LOCK = RLock()
_HF_DOWNLOAD_LOGGERS = (
    "huggingface_hub._snapshot_download",
    "huggingface_hub.file_download",
    "huggingface_hub.utils._http",
    "urllib3.connectionpool",
)
_HF_COMMIT = re.compile(r"[0-9a-f]{40}")
_VAD_FILES = ("pytorch_model.bin",)


class ProvisioningError(RuntimeError):
    """A setup error safe to show to the user without raw provider details."""


def _validated_https_url(url: str) -> parse.SplitResult:
    try:
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url):
            raise ValueError
        parsed = parse.urlsplit(url)
        _ = parsed.port  # Detect invalid and out-of-range ports.
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "\\" in url
        ):
            raise ValueError
        return parsed
    except ValueError:
        raise ProvisioningError("an HTTPS URL without embedded credentials is required") from None


class _HTTPSOnlyRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Check before urllib follows the redirect, not only after receiving its response.
        _validated_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _gigaam_url(filename: str) -> str:
    base = os.environ.get(GIGAAM_MIRROR_ENV, GIGAAM_BASE_URL)
    try:
        _validated_https_url(base)
        if "?" in base or "#" in base:
            raise ProvisioningError("query strings and fragments are not supported")
    except ProvisioningError:
        raise ProvisioningError(
            f"{GIGAAM_MIRROR_ENV} must be an HTTPS base URL without credentials, a query, "
            "or a fragment; correct it or unset it."
        ) from None
    return f"{base.rstrip('/')}/{filename}"


def file_digest(path: Path, algorithm: str) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def _content_length(headers) -> int | None:
    lengths = headers.get_all("Content-Length")
    if not lengths:
        return None
    try:
        if len(lengths) == 1 and lengths[0].strip().isascii() and lengths[0].strip().isdigit():
            return int(lengths[0])
    except ValueError:
        pass
    raise ProvisioningError("invalid Content-Length header")


def _download(
    url: str,
    destination: Path,
    *,
    label: str,
    recovery: str,
    validate: Callable[[Path], str],
    progress: Progress | None,
) -> str:
    """Return the verified file's SHA-256 only after publishing it atomically."""
    temporary: Path | None = None
    try:
        _validated_https_url(url)
        # Make verification explicit rather than relying on urllib's process-global opener.
        opener = request.build_opener(
            request.HTTPSHandler(context=ssl.create_default_context()),
            _HTTPSOnlyRedirectHandler(),
        )
        if progress is not None:
            progress(f"Downloading {label} …")
        req = request.Request(url, headers={"Accept-Encoding": "identity"})
        with opener.open(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            _validated_https_url(response.geturl())
            if response.status != 200:
                raise ProvisioningError(f"unexpected HTTP status {response.status}")
            length = _content_length(response.headers)
            transfer_encoding = response.headers.get("Transfer-Encoding", "").strip().lower()
            if transfer_encoding and (transfer_encoding != "chunked" or length is not None):
                raise ProvisioningError("invalid transfer framing")
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding != "identity":
                raise ProvisioningError("unexpected content encoding")
            mime = response.headers.get_content_type()
            if response.headers.get("Content-Type") and (
                mime.startswith("text/") or mime in {"application/json", "application/xml"}
            ):
                raise ProvisioningError("server returned a text response instead of a model")
            received = 0
            last_update = time.monotonic()
            total = f" / {length}" if length is not None else ""
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, suffix=".part", delete=False
            ) as handle:
                temporary = Path(handle.name)
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    received += len(chunk)
                    if length is not None and received > length:
                        raise ProvisioningError("response exceeds its declared length")
                    handle.write(chunk)
                    if progress is not None:
                        now = time.monotonic()
                        if now - last_update >= DOWNLOAD_PROGRESS_SECONDS and received != length:
                            progress(f"Downloading {label}: {received}{total} bytes")
                            last_update = now
            if not received:
                raise ProvisioningError("empty response")
            if length is not None and received != length:
                raise ProvisioningError("response ended before its declared length")
        sha256 = validate(temporary)
        if progress is not None:
            progress(f"Downloaded {label}: {received}{total} bytes; validated")
        os.replace(temporary, destination)
        return sha256
    except (ProvisioningError, OSError, http.client.HTTPException, ValueError) as exc:
        underlying = exc.reason if isinstance(exc, error.URLError) else exc
        if isinstance(exc, ProvisioningError):
            reason = str(exc)
        elif isinstance(exc, error.HTTPError):
            # HTTPError is also a response object; close it without reading the body.
            with suppress(OSError):
                exc.close()
            reason = f"HTTP {exc.code}"
        elif isinstance(underlying, ssl.SSLCertVerificationError):
            reason = "TLS certificate verification failed"
        elif isinstance(underlying, ssl.SSLError):
            reason = "TLS connection failed"
        elif isinstance(underlying, TimeoutError):
            reason = "connection or transfer timed out"
        elif isinstance(exc, http.client.IncompleteRead):
            reason = "response ended before its declared length"
        else:
            reason = "network or local file write failed"
        raise ProvisioningError(
            f"Could not download {label} to {destination}: {reason}. {recovery}"
        ) from None
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _prepare_model_dir(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ProvisioningError(
            f"Cannot create model directory {directory}; check its path and permissions."
        ) from None


@contextmanager
def _quiet_hf_requests():
    # HF can log raw server errors when falling back to a local folder, and HTTP
    # debug logging can expose signed URLs. Quiet these loggers only for this call.
    with _HF_LOG_LOCK:
        loggers = [logging.getLogger(name) for name in _HF_DOWNLOAD_LOGGERS]
        previous = [logger.disabled for logger in loggers]
        try:
            for logger in loggers:
                logger.disabled = True
            yield
        finally:
            for logger, disabled in zip(loggers, previous, strict=True):
                logger.disabled = disabled


def _hf_action[T](repository: str, action: str, *, token: str, call: Callable[[], T]) -> T:
    from huggingface_hub.utils import GatedRepoError

    url = f"https://huggingface.co/{repository}"
    if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}:
        raise ProvisioningError(
            f"Model files for {repository} are not cached. Restore a complete local model "
            "or unset HF_HUB_OFFLINE to download it."
        )
    if not token:
        raise ProvisioningError(
            f"Model files for {repository} are not cached. Accept or request access at {url}, "
            "then set HF_TOKEN in the environment."
        )
    try:
        with _quiet_hf_requests():
            return call()
    except Exception as exc:
        # LocalEntryNotFoundError may wrap the HTTP error from the initial request.
        seen: set[int] = set()
        current: BaseException | None = exc
        gated = False
        status = None
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            gated |= isinstance(current, GatedRepoError)
            if status is None:
                status = getattr(getattr(current, "response", None), "status_code", None)
            current = current.__cause__ or current.__context__

        if gated:
            hint = (
                "Access is gated: sign in and accept or request access on the model page, "
                "then use an HF_TOKEN from that account with read access to the gated repository."
            )
        elif status == 401:
            hint = "HTTP 401: HF_TOKEN may be invalid or expired; check or replace it."
        elif status == 403:
            hint = (
                "HTTP 403: check that your account can access the repository and that any "
                "fine-grained HF_TOKEN permits reads of this gated model."
            )
        else:
            hint = "Check your network, Hugging Face availability, and model directory, then retry."
        failure = ProvisioningError(f"Could not {action} {repository} ({url}). {hint}")
    raise failure from None


def check_huggingface_access(*, token: str, want_pyannote: bool, want_vad: bool = True) -> None:
    """Check gated access only for selected models not already installed."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    for selected, repo, required in (
        (want_vad, VAD_REPOSITORY, _VAD_FILES),
        (want_pyannote, PYANNOTE_REPO, COMMUNITY1_FILES),
    ):
        if selected and _cached_hf_snapshot(repo, required) is None:
            _hf_action(
                repo,
                "check access to",
                token=token,
                call=partial(
                    get_hf_file_metadata,
                    hf_hub_url(repo, required[0]),
                    token=token,
                    timeout=_HF_METADATA_TIMEOUT_SECONDS,
                ),
            )


def _complete_hf_snapshot(root: Path, required_files: tuple[str, ...]) -> bool:
    try:
        if root.is_file():
            # Older locks may reference a model file or a pipeline config directly.
            if len(required_files) > 1 and root.suffix not in {".yaml", ".yml"}:
                return False
            files = (root, *(root.parent / name for name in required_files[1:]))
        else:
            files = tuple(root / name for name in required_files)
        return all(file.is_file() and file.stat().st_size for file in files)
    except OSError:
        return False


def _existing_lock(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        lock = read_json(path)
        models = lock.get("models", {})
        if not isinstance(models, dict) or any(
            not isinstance(entry, dict) for entry in models.values()
        ):
            raise ValueError
        return lock
    except (AttributeError, OSError, ValueError):
        raise ProvisioningError(f"Cannot read model metadata in {path}.") from None


def _recorded_model_path(key: str) -> Path | None:
    value = _existing_lock(paths.lock_path()).get("models", {}).get(key, {}).get("path")
    return paths.resolve_model_path(value) if isinstance(value, str) and value else None


def _cached_hf_snapshot(repository: str, required_files: tuple[str, ...]) -> Path | None:
    """Read the recorded location, legacy Community-1 folder, or local main ref."""
    model_dir = paths.models_dir()
    key = "vad" if repository == VAD_REPOSITORY else "pyannote"
    candidate = _recorded_model_path(key)
    if candidate is not None and _complete_hf_snapshot(candidate, required_files):
        return candidate
    if repository == PYANNOTE_REPO:
        legacy = model_dir / "speaker-diarization-community-1"
        if _complete_hf_snapshot(legacy, required_files):
            return legacy

    repo = model_dir / "huggingface" / ("models--" + repository.replace("/", "--"))
    try:
        revision = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    if _HF_COMMIT.fullmatch(revision):
        root = repo / "snapshots" / revision
        if _complete_hf_snapshot(root, required_files):
            return root
    return None


def _download_hf_snapshot(
    repository: str,
    required_files: tuple[str, ...],
    *,
    token: str,
    snapshot_download,
) -> str:
    cached = _cached_hf_snapshot(repository, required_files)
    if cached is not None:
        return str(cached)
    destination = paths.models_dir() / "huggingface"
    snapshot = _hf_action(
        repository,
        "download",
        token=token,
        call=partial(
            snapshot_download,
            repository,
            token=token,
            cache_dir=destination,
        ),
    )
    try:
        root = Path(snapshot)
        if _complete_hf_snapshot(root, required_files):
            return str(root)
    except (OSError, TypeError, ValueError):
        pass
    raise ProvisioningError(
        f"Hugging Face snapshot for {repository} is incomplete under {destination}. "
        f"Retry setup with access to https://huggingface.co/{repository}."
    )


def download_vad_model(*, token: str, snapshot_download) -> str:
    """Download the gated VAD snapshot into the portable setup cache."""
    return _download_hf_snapshot(
        VAD_REPOSITORY,
        _VAD_FILES,
        token=token,
        snapshot_download=snapshot_download,
    )


def _local_file_revision(root: Path, filename: str) -> str | None:
    metadata = root / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    try:
        with metadata.open(encoding="utf-8") as handle:
            commit = handle.readline(128).strip()
            etag = handle.readline(256).strip()
            timestamp = float(handle.readline(128).strip())
        if (
            _HF_COMMIT.fullmatch(commit)
            and etag
            and math.isfinite(timestamp)
            and (root / filename).stat().st_mtime - 1 <= timestamp
        ):
            return commit
    except (OSError, ValueError):
        pass
    return None


def snapshot_revision(path: str | Path) -> str | None:
    """Get the HF snapshot commit, not a file's ETag or a random metadata entry."""
    root = Path(path)
    if root.is_file():
        root = root.parent
    if root.parent.name == "snapshots" and _HF_COMMIT.fullmatch(root.name):
        return root.name
    revisions = {_local_file_revision(root, name) for name in COMMUNITY1_FILES}
    if len(revisions) == 1:
        return revisions.pop()
    return None


def _wespeaker_sha256(path: Path) -> str:
    if not path.is_file():
        raise ProvisioningError("not a readable ONNX file")
    try:
        from onnxruntime import InferenceSession, SessionOptions
    except ImportError:
        raise ProvisioningError("ONNX Runtime is missing; run uv sync") from None
    try:
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        session = InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        (features,) = session.get_inputs()
        (embeddings,) = session.get_outputs()
        if (
            features.name != "feats"
            or features.type != "tensor(float)"
            or len(features.shape) != 3
            or features.shape[-1] != 80
            or embeddings.name != "embs"
            or embeddings.type != "tensor(float)"
            or len(embeddings.shape) != 2
            or embeddings.shape[-1] != 256
        ):
            raise ValueError
    except Exception:
        raise ProvisioningError("not a loadable WeSpeaker ONNX model") from None
    # The official URL has no pinned digest; this hash records provenance only.
    return file_digest(path, "sha256")


def _gigaam_dir() -> Path:
    recorded = _recorded_model_path("gigaam")
    return recorded if recorded is not None and recorded.is_dir() else paths.models_dir() / "gigaam"


def _wespeaker_path() -> Path:
    recorded = _recorded_model_path("wespeaker")
    if recorded is not None and recorded.is_file():
        return recorded
    return paths.models_dir() / "wespeaker" / WESPEAKER_LANG / "model.onnx"


def download_gigaam_models(
    models: list[str], *, progress: Progress | None = None
) -> tuple[Path, dict[str, dict[str, str]]]:
    """Download or verify requested local GigaAM checkpoints.

    Returns (gigaam_dir, records for models newly downloaded or verified this run) —
    the caller merges those records with whatever models.lock.json already has.
    """
    if not models:
        return paths.models_dir() / "gigaam", {}
    gigaam_dir = _gigaam_dir()
    downloaded_gigaam_models: dict[str, dict[str, str]] = {}
    for model_name in models:
        info = ASR_MODELS[model_name]
        checkpoint = gigaam_dir / info["checkpoint"]

        def checkpoint_sha256(path: Path, expected_md5: str = info["md5"]) -> str:
            if file_digest(path, "md5") != expected_md5:
                raise ProvisioningError("MD5 does not match the upstream checkpoint")
            return file_digest(path, "sha256")

        if checkpoint.exists() or checkpoint.is_symlink():
            try:
                if not checkpoint.is_file():
                    raise ProvisioningError("not a readable regular file")
                sha256 = checkpoint_sha256(checkpoint)
            except (OSError, ProvisioningError) as exc:
                reason = str(exc) if isinstance(exc, ProvisioningError) else "could not read file"
                raise ProvisioningError(
                    f"Existing GigaAM checkpoint {checkpoint}: {reason}. "
                    "Replace it manually with a verified copy, or move it aside and rerun setup."
                ) from None
        else:
            _prepare_model_dir(gigaam_dir)
            sha256 = _download(
                _gigaam_url(checkpoint.name),
                checkpoint,
                label=f"GigaAM {model_name}",
                recovery=(
                    f"Retry, set {GIGAAM_MIRROR_ENV} to a trusted HTTPS mirror, "
                    "or install a verified copy manually at the path above."
                ),
                validate=checkpoint_sha256,
                progress=progress,
            )
        downloaded_gigaam_models[model_name] = {
            "checkpoint": paths.store_model_path(checkpoint),
            "md5": info["md5"],
            "sha256": sha256,
        }
    return gigaam_dir, downloaded_gigaam_models


def setup_needs_downloads(
    *, local_models: list[str], want_vad: bool, want_pyannote: bool, want_foxnose: bool
) -> bool:
    """Check presence only; verification is performed before publishing the lock."""
    if local_models:
        gigaam = _gigaam_dir()
        if any(not (gigaam / ASR_MODELS[name]["checkpoint"]).exists() for name in local_models):
            return True
    return (
        (want_foxnose and not _wespeaker_path().exists())
        or (want_vad and _cached_hf_snapshot(VAD_REPOSITORY, _VAD_FILES) is None)
        or (want_pyannote and _cached_hf_snapshot(PYANNOTE_REPO, COMMUNITY1_FILES) is None)
    )


def download_diarization_models(
    *,
    want_pyannote: bool,
    want_foxnose: bool,
    token: str,
    snapshot_download,
    progress: Progress | None = None,
) -> tuple[dict[str, object], str | None, Path | None]:
    """Download the requested diarization backend(s).

    Returns (new models.lock.json entries for pyannote/wespeaker/diarize, the
    pyannote snapshot path, the wespeaker onnx path) — the two paths feed the smoke
    tests that run afterwards. Complete installed models are reused.
    """
    new_entries: dict[str, object] = {}
    pyannote_path: str | None = None
    if want_pyannote:
        if progress is not None:
            progress(f"Preparing Hugging Face model {PYANNOTE_REPO} …")
        pyannote_path = _download_hf_snapshot(
            PYANNOTE_REPO,
            COMMUNITY1_FILES,
            token=token,
            snapshot_download=snapshot_download,
        )
        pipeline_path = Path(pyannote_path)
        # Normalize the directory, not the config symlink: relative resources
        # in the config must remain siblings of the config, not of the blob.
        stored_path = (
            str(Path(paths.store_model_path(pipeline_path.parent)) / pipeline_path.name)
            if pipeline_path.is_file()
            else paths.store_model_path(pipeline_path)
        )
        new_entries["pyannote"] = {
            "repository": PYANNOTE_REPO,
            "path": stored_path,
            "revision": snapshot_revision(pyannote_path),
        }
    wespeaker_path: Path | None = None
    if want_foxnose:
        try:
            diarize_version = importlib.metadata.version("diarize")
        except importlib.metadata.PackageNotFoundError:
            raise ProvisioningError("FoxNose requires diarize; run `uv sync --locked`.") from None
        wespeaker_path = _wespeaker_path()
        if wespeaker_path.exists() or wespeaker_path.is_symlink():
            try:
                sha256 = _wespeaker_sha256(wespeaker_path)
            except (OSError, ProvisioningError) as exc:
                reason = str(exc) if isinstance(exc, ProvisioningError) else "could not read file"
                raise ProvisioningError(
                    f"Existing WeSpeaker checkpoint {wespeaker_path}: {reason}. "
                    "If the file is invalid, move it aside and rerun setup."
                ) from None
        else:
            _prepare_model_dir(wespeaker_path.parent)
            sha256 = _download(
                WESPEAKER_MODEL_URL,
                wespeaker_path,
                label="WeSpeaker",
                recovery="Check access to the official WeSpeaker server and retry setup.",
                validate=_wespeaker_sha256,
                progress=progress,
            )
        new_entries["wespeaker"] = {
            "repository": "wespeaker/voxceleb_resnet34_LM",
            "url": WESPEAKER_MODEL_URL,
            "path": paths.store_model_path(wespeaker_path),
            "sha256": sha256,
            "lang": WESPEAKER_LANG,
        }
        new_entries["diarize"] = {
            "package": "diarize",
            "version": diarize_version,
        }
    return new_entries, pyannote_path, wespeaker_path


def asr_lock_entries(
    gigaam_dir: Path,
    downloaded_gigaam_models: dict[str, dict[str, str]],
    vad: str | None,
    requested_network_models: list[str],
) -> dict[str, object]:
    """Build the gigaam/vad/groq models.lock.json entries, merging gigaam's own
    models map with whatever this project already recorded: setting
    up a different --asr-models set must not forget a previously downloaded model.
    """
    new_models: dict[str, object] = {}
    if downloaded_gigaam_models:
        existing_lock_for_merge = _existing_lock(paths.lock_path())
        existing_gigaam_models = (
            existing_lock_for_merge.get("models", {}).get("gigaam", {}).get("models", {})
        )
        new_models["gigaam"] = {
            "repository": "https://github.com/salute-developers/GigaAM",
            "commit": GIGAAM_COMMIT,
            "path": paths.store_model_path(gigaam_dir),
            "models": {**existing_gigaam_models, **downloaded_gigaam_models},
        }
    if vad is not None:
        new_models["vad"] = {
            "repository": VAD_REPOSITORY,
            "path": paths.store_model_path(Path(vad)),
            "revision": snapshot_revision(vad),
            "parameters": {"min_duration_on": 0.0, "min_duration_off": 0.0},
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


def write_lock(new_models: dict[str, object], *, smoke: dict[str, object] | None = None) -> Path:
    """Merge into whatever models.lock.json already has (old or new location, plan
    section 2), but always write to the new canonical location: a provider-scoped
    setup run must not erase records written by a previous run for the other
    provider. Call after smoke succeeds to publish both sets of results together.
    """
    lock_target = paths.models_dir() / paths.LOCK_FILENAME
    source = lock_target
    try:
        source = paths.lock_path()
        existing = _existing_lock(source)
        updated = {**existing, "models": {**existing.get("models", {}), **new_models}}
        if smoke is not None:
            updated["smoke"] = {**existing.get("smoke", {}), **smoke}
        if source != lock_target or updated != existing:
            atomic_json(lock_target, updated)
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, OSError):
            hint = "Check disk space and model directory permissions."
        else:
            hint = f"Check the lock format at {source}."
        raise ProvisioningError(f"Could not update {lock_target}. {hint}") from None
    return lock_target


@contextmanager
def _model_smoke(label: str, path: str | Path | None):
    try:
        yield
    except Exception:
        raise ProvisioningError(
            f"{label} smoke test failed for {path}. Check the model files and runtime dependencies "
            "(`uv sync --locked`), then rerun setup."
        ) from None


@contextmanager
def _offline_smoke():
    from huggingface_hub import constants
    from huggingface_hub.utils import reset_sessions

    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYANNOTE_METRICS_ENABLED": "0"}
    # huggingface_hub may already have been imported and initialized by the download phase.
    with _HF_LOG_LOCK:
        previous = {key: os.environ.get(key) for key in env}
        offline = constants.HF_HUB_OFFLINE
        try:
            os.environ.update(env)
            constants.HF_HUB_OFFLINE = True
            reset_sessions()
            yield
        finally:
            constants.HF_HUB_OFFLINE = offline
            reset_sessions()
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def run_smoke_tests(
    *,
    requested_local_models: list[str],
    gigaam_dir: Path,
    vad_path: str | None,
    want_pyannote: bool,
    want_foxnose: bool,
    pyannote_path: str | None,
    wespeaker_path: Path | None,
) -> dict[str, object]:
    """Prove every downloaded model loads and runs fully offline, without uploading
    user audio. Never returns partial results: any failure raises RuntimeError."""
    with (
        _offline_smoke(),
        tempfile.TemporaryDirectory(prefix="transcribe-meeting-setup-") as temporary,
    ):
        smoke_wav = Path(temporary) / "silence.wav"
        with wave.open(str(smoke_wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 16_000)
        from .asr import GigaAMBackend, detect_speech_regions

        new_smoke: dict[str, object] = {}
        if requested_local_models:
            with _model_smoke("ASR device selection", gigaam_dir):
                device = effective_asr_device("auto")
            new_smoke.update(asr="ok", asr_models=list(requested_local_models), device=device)
            # Keep peak memory down by loading and closing one model at a time.
            for model_name in requested_local_models:
                checkpoint = gigaam_dir / ASR_MODELS[model_name]["checkpoint"]
                with _model_smoke(f"GigaAM {model_name}", checkpoint):
                    backend = GigaAMBackend(gigaam_dir, device, model_name)
                    try:
                        backend.transcribe(smoke_wav)
                    finally:
                        backend.close()
        if vad_path is not None:
            with _model_smoke(f"VAD {VAD_REPOSITORY}", vad_path):
                detect_speech_regions(smoke_wav, Path(vad_path))
            new_smoke["vad"] = "ok"
        if want_pyannote:
            with _model_smoke("Community-1", pyannote_path):
                import torch
                from pyannote.audio import Pipeline

                Pipeline.from_pretrained(str(pyannote_path)).to(torch.device("cpu"))
            new_smoke["diarization_load"] = "ok"
        if want_foxnose:
            with _model_smoke("FoxNose/WeSpeaker", wespeaker_path):
                if wespeaker_path is None:
                    raise ValueError("missing WeSpeaker model path")
                import wespeakerruntime

                from .diarization import _diarize_foxnose, _local_wespeaker

                # Silence makes diarize() return right after VAD, before extract_embeddings
                # ever runs (see diarize/src/diarize/__init__.py) — so _diarize_foxnose alone
                # never opens the WeSpeaker ONNX session and would smoke-test-pass even with a
                # missing/corrupt model file. Load the session directly through the shim first
                # to actually prove the model path and onnxruntime work offline.
                with _local_wespeaker(wespeaker_path, 1):
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
            new_smoke["diarization_foxnose"] = "ok"
    return new_smoke
