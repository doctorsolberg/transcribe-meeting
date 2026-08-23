from __future__ import annotations

import os
import re
import tempfile
from collections.abc import MutableMapping
from contextlib import suppress
from pathlib import Path

from .paths import secrets_path

SECRET_NAMES = {"OPENAI_API_KEY", "GROQ_API_KEY", "DEEPGRAM_API_KEY", "HF_TOKEN"}
PROJECT_ENV_NAMES = SECRET_NAMES | {"OPENAI_BASE_URL"}
_ASSIGNMENT = re.compile(r"\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*)\Z")


def _assignment(line: str) -> tuple[str, str] | None:
    match = _ASSIGNMENT.match(line)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _value(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    return raw


def load_project_secrets(
    path: Path | None = None, environ: MutableMapping[str, str] | None = None
) -> None:
    """Load supported project settings without evaluating arbitrary .env content."""
    if path is None:
        path = secrets_path()
    if not path.exists():
        return
    # The file may contain API credentials. Fix an accidentally permissive mode
    # before reading it; this is intentionally not a general-purpose dotenv loader.
    if path.stat().st_mode & 0o077:
        path.chmod(0o600)
    target = os.environ if environ is None else environ
    for line in path.read_text(encoding="utf-8").splitlines():
        item = _assignment(line)
        if item is None:
            continue
        name, raw = item
        if name in PROJECT_ENV_NAMES and name not in target:
            value = _value(raw)
            if value:
                target[name] = value


def save_project_secret(name: str, value: str, path: Path | None = None) -> None:
    """Atomically save one supported project setting with owner-only permissions."""
    if path is None:
        path = secrets_path()
    if name not in PROJECT_ENV_NAMES:
        raise ValueError(
            "Only OPENAI_API_KEY, GROQ_API_KEY, DEEPGRAM_API_KEY, "
            "HF_TOKEN, or OPENAI_BASE_URL can be saved."
        )
    if not value or "\n" in value or "\r" in value:
        raise ValueError("The key must not be empty and must not contain a line break.")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    replacement = f"{name}={value}"
    updated: list[str] = []
    replaced = False
    for line in lines:
        item = _assignment(line)
        if item and item[0] == name:
            if not replaced:
                updated.append(replacement)
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        updated.append(replacement)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".env.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(updated) + "\n")
        os.replace(temporary, path)
        path.chmod(0o600)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
