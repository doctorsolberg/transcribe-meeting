from __future__ import annotations

import os
import re
import stat
import tempfile
from collections.abc import MutableMapping
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

from .paths import ENV_DATA_ROOT, secrets_path

SECRET_NAMES = {"OPENAI_API_KEY", "GROQ_API_KEY", "DEEPGRAM_API_KEY", "HF_TOKEN"}
PROJECT_ENV_NAMES = SECRET_NAMES | {"OPENAI_BASE_URL"}
OPENAI_ENV_CONFLICT = "_TRANSCRIBE_OPENAI_ENV_CONFLICT"
_OPENAI_NAMES = {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
_OPTION_NAMES = {
    ENV_DATA_ROOT,
    "TRANSCRIBE_READABLE_API",
    "TRANSCRIBE_READABLE_RESPONSE_FORMAT",
    "TRANSCRIBE_READABLE_BATCH_SIZE",
    "TRANSCRIBE_READABLE_REASONING",
    "TRANSCRIBE_GIGAAM_BASE_URL",
}
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
    target = os.environ if environ is None else environ
    target.pop(OPENAI_ENV_CONFLICT, None)
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        return
    # The file may contain API credentials. Fix an accidentally permissive mode
    # before reading it; this is intentionally not a general-purpose dotenv loader.
    if mode & 0o077:
        path.chmod(0o600)
    project: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = _assignment(line)
        if item is None:
            continue
        name, raw = item
        if (name in PROJECT_ENV_NAMES or name in _OPTION_NAMES) and (value := _value(raw).strip()):
            project.setdefault(name, value)
    for name, value in project.items():
        if name == ENV_DATA_ROOT and not target.get(name):
            target[name] = value
        elif name not in _OPENAI_NAMES:
            target.setdefault(name, value)
    _load_openai_pair(target, project)


def _same_readable_origin(left: str, right: str) -> bool:
    from .readable import ReadableConfigError, validate_readable_base_url

    def origin(value: str) -> tuple[str, str | None, int]:
        url = urlsplit(validate_readable_base_url(value))
        return url.scheme, url.hostname, url.port or (443 if url.scheme == "https" else 80)

    try:
        return origin(left) == origin(right)
    except ReadableConfigError:
        return False


def _load_openai_pair(target: MutableMapping[str, str], project: dict[str, str]) -> None:
    from .readable import DEFAULT_READABLE_BASE_URL

    shell_key = target.get("OPENAI_API_KEY", "").strip()
    shell_url = target.get("OPENAI_BASE_URL", "").strip()
    project_url = project.get("OPENAI_BASE_URL", "")
    project_key = project.get("OPENAI_API_KEY", "")
    if not shell_url and project_url:
        if shell_key and not _same_readable_origin(DEFAULT_READABLE_BASE_URL, project_url):
            # Never send to a different origin or silently fall back to OpenAI.
            # The marker carries no URL or secret and is not loadable from .env.
            target[OPENAI_ENV_CONFLICT] = "endpoint"
        else:
            target["OPENAI_BASE_URL"] = project_url
    if not shell_key and project_key:
        if not shell_url or _same_readable_origin(
            shell_url, project_url or DEFAULT_READABLE_BASE_URL
        ):
            target["OPENAI_API_KEY"] = project_key
        else:
            target[OPENAI_ENV_CONFLICT] = "key"


def _check_project_endpoint_change(value: str, lines: list[str]) -> None:
    from .readable import DEFAULT_READABLE_BASE_URL, validate_readable_base_url

    validate_readable_base_url(value)
    existing: dict[str, str] = {}
    for line in lines:
        item = _assignment(line)
        if item and item[0] in _OPENAI_NAMES and (current := _value(item[1]).strip()):
            existing.setdefault(item[0], current)
    if existing.get("OPENAI_API_KEY") and not _same_readable_origin(
        existing.get("OPENAI_BASE_URL", DEFAULT_READABLE_BASE_URL), value
    ):
        raise ValueError(
            "Remove OPENAI_API_KEY from the project's .env before changing its API origin. "
            "Then configure the key for the new endpoint."
        )


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
    if path.exists() and not path.is_file():
        raise ValueError("The project settings path must be a regular file.")
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if name == "OPENAI_BASE_URL":
        _check_project_endpoint_change(value, lines)
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
