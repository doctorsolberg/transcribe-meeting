"""Publish transcript updates without losing the last accepted artifacts."""

from __future__ import annotations

import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .util import atomic_json, read_json, sha256

_PENDING = ".readable-publish"
_PRODUCTS = (
    "canonical.json",
    "transcript.readable.md",
    "review/readable_changes.csv",
    "result.json",
    "summary.md",
    ".stages/published.json",
)
_EDIT_PRODUCTS = (*_PRODUCTS, "speakers.yaml")


def recover_readable_publication(job: Path) -> None:
    if (job / _PENDING).is_dir():
        with job_transaction(job):
            pass


@contextmanager
def job_transaction(job: Path):
    with (job / ".readable-lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        _recover(job)
        yield


def _recover(job: Path) -> None:
    pending = job / _PENDING
    if not pending.is_dir():
        return
    journal = read_json(pending / "journal.json")
    manifest = read_json(job / "manifest.json")
    generation = manifest.get(
        "transcript_generation", (manifest.get("readable") or {}).get("generation")
    )
    if generation != journal["generation"]:
        # Copy, not move, so an interruption during rollback is safe to retry.
        products = journal.get("products", _PRODUCTS[:3])
        for name in _EDIT_PRODUCTS:
            if name not in products:
                continue
            target = job / name
            backup = pending / "previous" / name
            if name in journal["previous"]:
                temporary = pending / "restore"
                shutil.copyfile(backup, temporary)
                temporary.chmod(0o600)
                os.replace(temporary, target)
            else:
                target.unlink(missing_ok=True)
    # Removing the journal is the last step; don't leave a half-deleted journal.
    with TemporaryDirectory(prefix=".readable-cleanup-", dir=job) as temporary:
        os.replace(pending, Path(temporary) / _PENDING)


def publish_readable(
    job: Path, staged: Path, manifest: dict, *, canonical_digest: str, manifest_digest: str
) -> None:
    """Swap in already-rendered files, using the manifest as the commit point."""
    with job_transaction(job):
        current = (sha256(job / "canonical.json"), sha256(job / "manifest.json"))
        if current != (canonical_digest, manifest_digest):
            raise RuntimeError("The job changed during readable cleanup. Retry with clean.")
        _publish(job, staged, manifest, _PRODUCTS)


def publish_transcript_edit(job: Path, staged: Path, manifest: dict) -> None:
    """Publish already-staged edited files while the caller holds the job lock."""
    products = tuple(name for name in _EDIT_PRODUCTS if (staged / name).is_file())
    _publish(job, staged, manifest, products)


def _publish(job: Path, staged: Path, manifest: dict, products: tuple[str, ...]) -> None:
    generation = uuid4().hex
    with TemporaryDirectory(prefix=".readable-journal-", dir=job) as temporary:
        journal = Path(temporary) / _PENDING
        journal.mkdir(mode=0o700)
        previous = []
        for name in products:
            path = job / name
            if path.exists():
                backup = journal / "previous" / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, backup)
                backup.chmod(0o600)
                previous.append(name)
        atomic_json(
            journal / "journal.json",
            {"generation": generation, "previous": previous, "products": products},
        )
        pending = job / _PENDING
        os.rename(journal, pending)
        try:
            for name in products:
                target = job / name
                target.parent.mkdir(parents=True, exist_ok=True)
                (staged / name).chmod(0o600)
                os.replace(staged / name, target)
            manifest["transcript_generation"] = generation
            atomic_json(job / "manifest.json", manifest)
        finally:
            _recover(job)
