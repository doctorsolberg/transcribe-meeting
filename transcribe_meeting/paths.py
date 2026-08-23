"""Filesystem locations for project data: models, job output, secrets, glossary.

Every function here reads the environment at call time rather than caching a
module-level constant. The CLI's global --data-root/--output-dir/--models-dir
options (see cli.py's app callback) work by setting these env vars just before a
command runs, and tests override the same env vars per-case; neither would be
possible if a path were computed once at import time. Defaults are relative to
the current working directory, not to the installed package, so a checkout on a
Linux server can keep code and data on different volumes.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_ROOT = "TRANSCRIBE_HOME"
ENV_OUTPUT_DIR = "TRANSCRIBE_OUTPUT_DIR"
ENV_MODELS_DIR = "TRANSCRIBE_MODELS_DIR"
ENV_SECRETS_FILE = "TRANSCRIBE_ENV_FILE"

LOCK_FILENAME = "models.lock.json"


def data_root() -> Path:
    """TRANSCRIBE_HOME, or the current working directory."""
    value = os.environ.get(ENV_DATA_ROOT)
    return Path(value) if value else Path.cwd()


def output_dir() -> Path:
    value = os.environ.get(ENV_OUTPUT_DIR)
    return Path(value) if value else data_root() / "output"


def models_dir() -> Path:
    value = os.environ.get(ENV_MODELS_DIR)
    return Path(value) if value else data_root() / "models"


def glossary_path() -> Path:
    return data_root() / "glossary.yaml"


def secrets_path() -> Path:
    value = os.environ.get(ENV_SECRETS_FILE)
    return Path(value) if value else data_root() / ".env"


def lock_path() -> Path:
    """models.lock.json's canonical location, with a read-only fallback.

    The lock travels with the models directory it describes, so
    models_dir()/models.lock.json is canonical. A file left at the old project-root
    location (data_root()/models.lock.json, pre-stage-3 layout) is still read from
    there when the new location is empty, but this function never copies or deletes
    anything — callers that care about the fallback (core.model_lock) are
    responsible for telling the user to move it.
    """
    current = models_dir() / LOCK_FILENAME
    if current.exists():
        return current
    legacy = data_root() / LOCK_FILENAME
    if legacy.exists():
        return legacy
    return current


def resolve_model_path(value: str) -> Path:
    """Turn one models.lock.json path value back into a real path.

    A relative value is a path recorded relative to models_dir();
    an absolute value is kept as-is, since it may point outside models_dir()
    entirely (an external HF cache, for instance).
    """
    path = Path(value)
    return path if path.is_absolute() else models_dir() / path


def store_model_path(path: Path) -> str:
    """Serialize a model path for models.lock.json, relative to models_dir() when
    possible so the lock file — and the directory it travels in — stay portable
    between machines. A path outside models_dir() is stored
    absolute, since there is nothing meaningful to make it relative to.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(models_dir().resolve()).as_posix()
    except ValueError:
        return str(resolved)
