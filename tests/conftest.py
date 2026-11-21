"""Shared test isolation.

The suite must not depend on the developer's shell. TRANSCRIBE_HOME and friends change
where every path function looks, so a machine that exports them for real work would see
spurious failures — and a test that silently reads the developer's own models directory
proves nothing. Clearing them here makes the default identical to a clean checkout;
tests that need specific locations still set them with monkeypatch as usual.
"""

from __future__ import annotations

import pytest

_LOCATION_ENV = (
    "TRANSCRIBE_HOME",
    "TRANSCRIBE_OUTPUT_DIR",
    "TRANSCRIBE_MODELS_DIR",
    "TRANSCRIBE_ENV_FILE",
)


@pytest.fixture(autouse=True)
def _isolate_location_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for name in _LOCATION_ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    # Test CLI calls from the checkout must never read the developer's real .env.
    monkeypatch.setenv("TRANSCRIBE_ENV_FILE", str(tmp_path / "no-project.env"))
