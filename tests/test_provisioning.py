from __future__ import annotations

import builtins
import hashlib
import io
import json
import logging
import os
import socket
import ssl
import time
from email.message import Message
from pathlib import Path
from urllib import error, request

import pytest
from huggingface_hub import constants as hf_constants
from huggingface_hub.utils import GatedRepoError, LocalEntryNotFoundError
from requests import HTTPError
from requests import Response as RequestsResponse
from typer.testing import CliRunner

from transcribe_meeting import asr, cli, provisioning

MODEL = "v3_rnnt"
CHECKPOINT = "v3_rnnt.ckpt"
PAYLOAD = b"checkpoint bytes\0" * 10
MIRROR = "https://mirror.example/checkpoints/"
# A minimal, offline ONNX graph with the interface of WeSpeaker's exported model.
ONNX = bytes.fromhex(
    "08083a82010a1e0a0573686170651204656d6273220f436f6e7374616e744f665368617065121674"
    "696e795f7765737065616b65725f666978747572652a10080210073a03018002420573686170655a"
    "1d0a05666561747312140a120801120e0a031201420a031201540a02085062170a04656d6273120f"
    "0a0d080112090a0208010a0308800242040a00100d"
)
TOKEN = "hf_token_must_not_be_printed"
REVISION = "a" * 40
COMMUNITY_FILES = (
    "config.yaml",
    "segmentation/pytorch_model.bin",
    "embedding/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)


class Response(io.BytesIO):
    def __init__(self, body, *, url=MIRROR + CHECKPOINT, status=200, length=None, mime=None):
        super().__init__(body)
        self.status = status
        self.url = url
        self.headers = Message()
        if length is not None:
            self.headers["Content-Length"] = str(length)
        if mime is not None:
            self.headers["Content-Type"] = mime

    def geturl(self):
        return self.url


@pytest.fixture(autouse=True)
def _isolate_hf_mode(monkeypatch):
    # Pipeline tests set HF_HUB_OFFLINE process-wide; neither those tests nor
    # the developer's shell should change these stubbed download scenarios.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    models = tmp_path / "separate" / "models"
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path / "new-data-root"))
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(models))
    monkeypatch.delenv("TRANSCRIBE_GIGAAM_BASE_URL", raising=False)
    return models


@pytest.fixture
def checkpoint(models_dir, monkeypatch):
    monkeypatch.setitem(provisioning.ASR_MODELS[MODEL], "md5", hashlib.md5(PAYLOAD).hexdigest())
    return models_dir / "gigaam" / CHECKPOINT


@pytest.fixture
def serve(monkeypatch):
    def install(response):
        calls = []

        class FakeOpener:
            def open(self, req, *, timeout):
                calls.append((req.full_url, timeout))
                if isinstance(response, Exception):
                    raise response
                return response

        monkeypatch.setattr(request, "build_opener", lambda *_handlers: FakeOpener())
        return calls

    # Guard the old implementation too: RED must never reach the external URL.
    monkeypatch.setattr(request, "urlretrieve", lambda *_: pytest.fail("unexpected urlretrieve"))
    return install


def test_gigaam_publishes_verified_download_from_explicit_mirror(checkpoint, monkeypatch, serve):
    monkeypatch.setenv("TRANSCRIBE_GIGAAM_BASE_URL", MIRROR)
    calls = serve(Response(PAYLOAD, length=len(PAYLOAD)))
    events = []

    def progress(message):
        assert not checkpoint.exists()
        events.append(message)

    directory, records = provisioning.download_gigaam_models([MODEL], progress=progress)

    assert directory == checkpoint.parent
    assert checkpoint.read_bytes() == PAYLOAD
    assert list(directory.iterdir()) == [checkpoint]
    assert len(calls) == 1
    assert calls[0][0] == MIRROR + CHECKPOINT
    assert 0 < calls[0][1] <= 60
    assert records == {
        MODEL: {
            "checkpoint": f"gigaam/{CHECKPOINT}",
            "md5": hashlib.md5(PAYLOAD).hexdigest(),
            "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        }
    }
    assert any(f"{len(PAYLOAD)}" in event and "bytes" in event for event in events)


def test_gigaam_accepts_valid_preinstalled_file_without_network(checkpoint, serve):
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(PAYLOAD)
    calls = serve(AssertionError("network not needed"))

    _, records = provisioning.download_gigaam_models([MODEL])

    assert records[MODEL]["md5"] == hashlib.md5(PAYLOAD).hexdigest()
    assert calls == []


def test_gigaam_keeps_corrupt_preinstalled_file_with_manual_path_hint(checkpoint, serve):
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"old file")
    calls = serve(AssertionError("network not needed"))

    with pytest.raises(RuntimeError, match="MD5") as exc:
        provisioning.download_gigaam_models([MODEL])

    assert str(checkpoint) in str(exc.value)
    assert "replace" in str(exc.value).lower()
    assert checkpoint.read_bytes() == b"old file"
    assert calls == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (Response(b"tampered"), "MD5"),
        (Response(PAYLOAD, length=len(PAYLOAD) + 1), "length"),
        (Response(b""), "empty"),
        (Response(PAYLOAD, status=206), "206"),
        (Response(PAYLOAD, url="http://mirror.example/file"), "HTTPS"),
        (TimeoutError("hidden-response-body"), "timed out"),
        (
            error.URLError(ssl.SSLCertVerificationError("hidden-response-body")),
            "certificate",
        ),
    ],
)
def test_gigaam_fails_safely_before_publishing(checkpoint, serve, response, message):
    serve(response)

    with pytest.raises(RuntimeError, match=message) as exc:
        provisioning.download_gigaam_models([MODEL])

    assert str(checkpoint) in str(exc.value)
    assert "TRANSCRIBE_GIGAAM_BASE_URL" in str(exc.value)
    assert "hidden-response-body" not in str(exc.value)
    assert not checkpoint.exists()
    assert not list(checkpoint.parent.glob("*.part"))


@pytest.mark.parametrize(
    "mirror",
    [
        "http://mirror.example/checkpoints",
        "https://hidden:secret@mirror.example",
        "https://mirror.example/models?key=secret",
        "https://mirror.example/#secret",
        "https://mirror.example:bad/checkpoints",
        "https:///checkpoints",
    ],
)
def test_gigaam_rejects_unsafe_mirror_without_leaking_value(checkpoint, monkeypatch, serve, mirror):
    monkeypatch.setenv("TRANSCRIBE_GIGAAM_BASE_URL", mirror)
    calls = serve(AssertionError("mirror must not be queried"))

    with pytest.raises(RuntimeError, match=r"TRANSCRIBE_GIGAAM_BASE_URL.*HTTPS") as exc:
        provisioning.download_gigaam_models([MODEL])

    assert mirror not in str(exc.value)
    assert calls == []
    assert not checkpoint.exists()


def test_gigaam_limits_byte_progress_during_fast_transfer(checkpoint, monkeypatch, serve):
    payload = b"test bytes" * (1 << 20)
    monkeypatch.setitem(provisioning.ASR_MODELS[MODEL], "md5", hashlib.md5(payload).hexdigest())
    serve(Response(payload))
    monkeypatch.setattr(time, "monotonic", lambda: 1.0)
    events = []

    provisioning.download_gigaam_models([MODEL], progress=events.append)

    assert 1 <= len(events) <= 2
    assert f"{len(payload)} bytes" in events[-1]


def test_download_verifies_tls_and_blocks_http_before_following_redirect(
    checkpoint, monkeypatch, serve
):
    calls = serve(Response(PAYLOAD))
    build_opener = request.build_opener

    def inspect_opener(*handlers):
        tls = next(h for h in handlers if isinstance(h, request.HTTPSHandler))
        assert tls._context.check_hostname
        assert tls._context.verify_mode == ssl.CERT_REQUIRED
        redirect = next(h for h in handlers if isinstance(h, request.HTTPRedirectHandler))
        req = request.Request(MIRROR)
        with pytest.raises(RuntimeError, match="HTTPS"):
            redirect.redirect_request(req, None, 302, "", {}, "http://target.example/signed")
        assert redirect.redirect_request(req, None, 302, "", {}, MIRROR).full_url == MIRROR
        return build_opener(*handlers)

    monkeypatch.setattr(request, "build_opener", inspect_opener)

    provisioning.download_gigaam_models([MODEL])

    assert calls[0][0] == provisioning.GIGAAM_BASE_URL + CHECKPOINT


def _no_snapshot(*_args, **_kwargs):
    pytest.fail("unexpected Hugging Face request")


def _download_wespeaker(progress=None):
    return provisioning.download_diarization_models(
        want_pyannote=False,
        want_foxnose=True,
        token="",
        snapshot_download=_no_snapshot,
        progress=progress,
    )


def test_wespeaker_uses_official_source_and_loadable_model(models_dir, monkeypatch, serve):
    monkeypatch.setenv("TRANSCRIBE_GIGAAM_BASE_URL", MIRROR)
    calls = serve(Response(ONNX, url=provisioning.WESPEAKER_MODEL_URL, length=len(ONNX)))
    events = []

    entries, pyannote, checkpoint = _download_wespeaker(events.append)

    assert pyannote is None
    assert calls[0][0] == provisioning.WESPEAKER_MODEL_URL
    assert checkpoint == models_dir / "wespeaker" / provisioning.WESPEAKER_LANG / "model.onnx"
    assert checkpoint.read_bytes() == ONNX
    assert entries["wespeaker"]["sha256"] == hashlib.sha256(ONNX).hexdigest()
    assert any(f"{len(ONNX)}" in event and "bytes" in event for event in events)


@pytest.mark.parametrize(
    ("body", "mime", "reason"),
    [
        (b"<html>secret-body</html>", "text/html", "text response"),
        (b"<html>secret-body</html>", "application/octet-stream", "ONNX"),
        (ONNX[:-12], None, "ONNX"),
        (ONNX.replace(b"embs", b"oops"), None, "ONNX"),
    ],
)
def test_wespeaker_rejects_unusable_response(models_dir, serve, body, mime, reason):
    serve(Response(body, mime=mime))

    with pytest.raises(RuntimeError, match=reason) as exc:
        _download_wespeaker()

    assert "secret-body" not in str(exc.value)
    assert not list((models_dir / "wespeaker").rglob("*.*"))


def test_wespeaker_leaves_preexisting_file_in_place_if_invalid(models_dir, serve):
    model = models_dir / "wespeaker" / provisioning.WESPEAKER_LANG / "model.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"not a model")
    calls = serve(AssertionError("no network needed"))

    with pytest.raises(RuntimeError, match="ONNX") as exc:
        _download_wespeaker()

    assert str(model) in str(exc.value)
    assert model.read_bytes() == b"not a model"
    assert calls == []


def test_foxnose_reports_missing_runtime_before_downloading_model(models_dir, monkeypatch, serve):
    calls = serve(AssertionError("no network needed"))

    def missing(_name):
        raise provisioning.importlib.metadata.PackageNotFoundError("diarize")

    monkeypatch.setattr(provisioning.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match=r"diarize.*uv sync"):
        _download_wespeaker()

    assert calls == []


def _download_hf(repository, snapshot_download):
    if repository == provisioning.VAD_REPOSITORY:
        return provisioning.download_vad_model(token=TOKEN, snapshot_download=snapshot_download)
    return provisioning.download_diarization_models(
        want_pyannote=True, want_foxnose=False, token=TOKEN, snapshot_download=snapshot_download
    )[1]


def _hf_http_error(status, kind=HTTPError):
    response = RequestsResponse()
    response.status_code = status
    response.url = f"https://huggingface.co/hidden?token={TOKEN}"
    return kind(f"private body: {TOKEN}", response=response)


def _cause_wrapped_forbidden():
    outer = LocalEntryNotFoundError("private cache lookup body")
    outer.__cause__ = _hf_http_error(403)
    return outer


@pytest.mark.parametrize("repository", [provisioning.VAD_REPOSITORY, provisioning.PYANNOTE_REPO])
@pytest.mark.parametrize(
    ("failure", "hint"),
    [
        (_hf_http_error(403, GatedRepoError), "accept"),
        (_hf_http_error(401), "invalid or expired"),
        (_hf_http_error(403), "fine-grained"),
        (_cause_wrapped_forbidden(), "fine-grained"),
        (TimeoutError(f"private body: {TOKEN}"), "network"),
    ],
    ids=["gated", "unauthorized", "forbidden", "wrapped-forbidden", "network"],
)
def test_huggingface_reports_sanitized_diagnostics(models_dir, caplog, repository, failure, hint):
    def snapshot(repo_id, *, token, **_kwargs):
        assert repo_id == repository
        assert token == TOKEN
        logger = logging.getLogger("huggingface_hub._snapshot_download")
        logger.warning("provider says: %s", failure)
        raise failure

    with pytest.raises(RuntimeError, match=hint) as exc:
        _download_hf(repository, snapshot)

    assert f"https://huggingface.co/{repository}" in str(exc.value)
    assert TOKEN not in str(exc.value) + caplog.text
    assert "private" not in str(exc.value) + caplog.text
    assert exc.value.__suppress_context__


@pytest.mark.parametrize("repository", [provisioning.VAD_REPOSITORY, provisioning.PYANNOTE_REPO])
def test_huggingface_rejects_empty_returned_snapshot(models_dir, repository):
    def snapshot(_repository, *, cache_dir=None, local_dir=None, token):
        assert token == TOKEN
        if local_dir is None:
            assert cache_dir is not None
            local_dir = cache_dir / "models--pyannote--segmentation-3.0" / "snapshots" / REVISION
        root = local_dir
        root.mkdir(parents=True)
        return str(root)

    with pytest.raises(RuntimeError, match="incomplete") as exc:
        _download_hf(repository, snapshot)

    assert str(models_dir) in str(exc.value)


def test_vad_download_keeps_cache_layout_and_records_commit(models_dir):
    def snapshot(repo_id, *, cache_dir, token):
        assert repo_id == provisioning.VAD_REPOSITORY
        assert token == TOKEN
        assert cache_dir == models_dir / "huggingface"
        path = cache_dir / "models--pyannote--segmentation-3.0" / "snapshots" / REVISION
        path.mkdir(parents=True)
        (path / "pytorch_model.bin").write_bytes(b"valid fixture contents")
        return str(path)

    vad = provisioning.download_vad_model(token=TOKEN, snapshot_download=snapshot)
    entries = provisioning.asr_lock_entries(models_dir / "gigaam", {}, vad, [])

    assert entries["vad"]["revision"] == REVISION
    assert entries["vad"]["path"].endswith(f"/snapshots/{REVISION}")


def _metadata(path, name, commit=REVISION):
    metadata = path / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"{commit}\n{'b' * 40}\n{time.time()}\n", encoding="utf-8")
    return metadata


def test_community_revision_reads_config_commit_not_etag_or_unrelated_metadata(models_dir):
    previous = models_dir / "speaker-diarization-community-1" / "config.yaml"
    previous.parent.mkdir(parents=True)
    previous.write_text("installed model", encoding="utf-8")

    def snapshot(repo_id, *, cache_dir, token):
        assert repo_id == provisioning.PYANNOTE_REPO
        assert token == TOKEN
        assert cache_dir == models_dir / "huggingface"
        local_dir = (
            cache_dir / "models--pyannote--speaker-diarization-community-1" / "snapshots" / REVISION
        )
        _metadata(local_dir, ".gitattributes", commit="c" * 40)
        for name in COMMUNITY_FILES:
            file = local_dir / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"model fixture")
            _metadata(local_dir, name)
        return str(local_dir)

    entries, path, _ = provisioning.download_diarization_models(
        want_pyannote=True, want_foxnose=False, token=TOKEN, snapshot_download=snapshot
    )

    entry = entries["pyannote"]
    assert isinstance(entry, dict) and path is not None
    assert entry["revision"] == REVISION
    assert (models_dir / entry["path"]).resolve() == Path(path).resolve()
    assert previous.read_text(encoding="utf-8") == "installed model"


@pytest.mark.parametrize("corrupt", [b"", b"only one line", b"\xff\xfe"])
def test_snapshot_revision_ignores_malformed_metadata(tmp_path, corrupt):
    _metadata(tmp_path, "config.yaml").write_bytes(corrupt)

    assert provisioning.snapshot_revision(str(tmp_path)) is None


def test_snapshot_revision_does_not_claim_commit_for_mixed_runtime_files(tmp_path):
    _metadata(tmp_path, "config.yaml")
    _metadata(tmp_path, "segmentation/pytorch_model.bin", commit="c" * 40)

    assert provisioning.snapshot_revision(str(tmp_path)) is None


def test_hf_preflight_only_checks_selected_repos_with_bounded_metadata_calls(monkeypatch):
    calls = []

    def get_metadata(url, *, token, timeout):
        assert token == TOKEN
        assert 0 < timeout <= 60
        calls.append(url)

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", get_metadata)
    provisioning.check_huggingface_access(token=TOKEN, want_pyannote=True, want_vad=False)

    assert len(calls) == 1
    assert f"/{provisioning.PYANNOTE_REPO}/" in calls[0]


def test_hf_preflight_stops_on_access_error_before_checking_next_repo(monkeypatch):
    calls = []

    def get_metadata(url, **_kwargs):
        calls.append(url)
        raise _hf_http_error(403)

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", get_metadata)
    with pytest.raises(RuntimeError, match="fine-grained") as exc:
        provisioning.check_huggingface_access(token=TOKEN, want_pyannote=True)

    assert len(calls) == 1
    assert f"https://huggingface.co/{provisioning.VAD_REPOSITORY}" in str(exc.value)
    assert TOKEN not in str(exc.value)


def test_write_lock_merges_smoke_into_first_atomic_write(models_dir, monkeypatch):
    lock = models_dir / "models.lock.json"
    lock.parent.mkdir(parents=True)
    old = {"models": {"pyannote": {"path": "old"}}, "smoke": {"diarization_load": "ok"}}
    lock.write_text(json.dumps(old), encoding="utf-8")
    writes = []
    atomic_json = provisioning.atomic_json

    def capture(target, value):
        writes.append((target, value))
        atomic_json(target, value)

    monkeypatch.setattr(provisioning, "atomic_json", capture)
    assert provisioning.write_lock({"vad": {"path": "new"}}, smoke={"vad": "ok"}) == lock

    expected = {
        "models": {"pyannote": {"path": "old"}, "vad": {"path": "new"}},
        "smoke": {"diarization_load": "ok", "vad": "ok"},
    }
    assert writes == [(lock, expected)]
    assert json.loads(lock.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(("stage", "label"), [("gigaam", MODEL), ("vad", "segmentation-3.0")])
def test_model_load_failure_is_actionable_and_does_not_touch_lock(
    models_dir, monkeypatch, stage, label
):
    lock = models_dir / "models.lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b'{"models": {}}')
    vad_path = models_dir / "huggingface" / "vad"

    def broken_model(*_args):
        raise OSError(f"model library internal details {TOKEN}")

    monkeypatch.setattr(provisioning, "effective_asr_device", lambda _device: "cpu")
    monkeypatch.setattr(asr, "GigaAMBackend", broken_model)
    monkeypatch.setattr(asr, "detect_speech_regions", broken_model)
    with pytest.raises(RuntimeError, match=label) as exc:
        provisioning.run_smoke_tests(
            requested_local_models=[MODEL] if stage == "gigaam" else [],
            gigaam_dir=models_dir / "gigaam",
            vad_path=str(vad_path),
            want_pyannote=False,
            want_foxnose=False,
            pyannote_path=None,
            wespeaker_path=None,
        )

    assert TOKEN not in str(exc.value)
    assert "uv sync" in str(exc.value)
    assert str(models_dir) in str(exc.value)
    assert exc.value.__suppress_context__
    assert lock.read_bytes() == b'{"models": {}}'


def test_smoke_disables_pyannote_telemetry_before_community_import(models_dir, monkeypatch):
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)
    actual_import = builtins.__import__
    imported = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def to(self, _device):
            pass

    def checked_import(name, *args, **kwargs):
        if name == "pyannote.audio":
            imported.append((os.environ["PYANNOTE_METRICS_ENABLED"], hf_constants.HF_HUB_OFFLINE))
            return type("PyannoteAudio", (), {"Pipeline": Pipeline})
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    result = provisioning.run_smoke_tests(
        requested_local_models=[],
        gigaam_dir=models_dir / "gigaam",
        vad_path=None,
        want_pyannote=True,
        want_foxnose=False,
        pyannote_path=str(models_dir / "community"),
        wespeaker_path=None,
    )

    assert imported == [("0", True)]
    assert result["diarization_load"] == "ok"
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert not hf_constants.HF_HUB_OFFLINE


def test_network_asr_setup_smokes_real_foxnose_runtime_offline(models_dir, monkeypatch, serve):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSCRIBE_ENV_FILE", str(models_dir / "nonexistent.env"))
    monkeypatch.setattr(socket.socket, "connect", _no_snapshot)
    # Real ONNX Runtime and bundled Silero VAD; no downloaded model and no API key.
    model = models_dir / "wespeaker" / provisioning.WESPEAKER_LANG / "model.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(ONNX)
    # Keep prior local ASR smoke provenance separate from this network-ASR setup.
    lock_path = models_dir / "models.lock.json"
    previous_smoke = {"asr_models": [MODEL], "asr": "ok", "device": "cpu"}
    lock_path.write_text(json.dumps({"models": {}, "smoke": previous_smoke}), encoding="utf-8")
    calls = serve(AssertionError("cached setup must not download anything"))
    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", _no_snapshot)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_snapshot)

    monkeypatch.setattr(cli, "check_ffmpeg", lambda: None)
    result = CliRunner().invoke(cli.app, ["setup", "--asr-models", asr.DEEPGRAM_ASR_MODEL_NAME])

    assert result.exit_code == 0, result.output
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["smoke"] == {**previous_smoke, "diarization_foxnose": "ok"}
    assert lock["models"]["wespeaker"]["sha256"] == hashlib.sha256(ONNX).hexdigest()
    assert not {"vad", "gigaam"}.intersection(lock["models"])
    assert calls == []
