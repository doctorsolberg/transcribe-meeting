"""Contract test: README's "All Parameters" tables must list exactly the options
the CLI actually has, in both directions. Without this, a renamed or added
typer option (or a stale README row) only shows up when someone happens to
read both side by side — this project is doc-heavy enough that it has already
drifted once (see the plan this test was added from)."""

from __future__ import annotations

import re
from pathlib import Path

import typer.main

from transcribe_meeting.cli import app

README_PATH = Path(__file__).resolve().parent.parent / "README.md"

# Added by typer/click itself, never declared as a typer.Option, so they cannot
# appear in a command's params but must not be reported as "undocumented" either
# if a table ever mentions them.
EXCLUDED_OPTIONS = {"--help", "--install-completion", "--show-completion"}

# The tables checked here. `review`, `refine`, `status`, `glossary` and `auth` are
# prose ("no options beyond the standard ones") or bulleted, not the
# `| --opt | ... | ... |` table shape this parser expects.
TARGET_COMMANDS = ["run", "setup", "clean", "rename", "resume", "doctor"]

_HEADING = re.compile(r"^### `([a-z-]+)`\s*$")


def _command_sections(readme_text: str) -> dict[str, str]:
    """Split the README into per-command chunks, keyed by its '### `name`' heading."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in readme_text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = heading.group(1)
            sections[current] = []
            continue
        if line.startswith("## "):
            current = None
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(body) for name, body in sections.items()}


def _documented_options(section_text: str) -> set[str]:
    """Pull `--flag` names out of the option column only.

    The description column freely cross-references other commands' flags
    (e.g. run's --speaker-names row mentions rename's --set), so scanning the
    whole line would falsely count those as documented for this command.
    """
    names: set[str] = set()
    for line in section_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 2:
            continue
        names.update(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", cells[1]))
    return names


def _cli_options(command_name: str) -> set[str]:
    command = typer.main.get_command(app).commands[command_name]
    names: set[str] = set()
    for param in command.params:
        if isinstance(param, typer.core.TyperArgument):
            continue
        names.update(param.opts)
        names.update(param.secondary_opts)
    return names - EXCLUDED_OPTIONS


def test_every_command_has_a_documented_table():
    sections = _command_sections(README_PATH.read_text(encoding="utf-8"))
    missing = [name for name in TARGET_COMMANDS if name not in sections]
    assert not missing, f"README has no '### `{missing}`' heading under All Parameters"


def test_documented_options_match_the_cli_exactly():
    sections = _command_sections(README_PATH.read_text(encoding="utf-8"))
    for name in TARGET_COMMANDS:
        documented = _documented_options(sections[name])
        actual = _cli_options(name)
        undocumented = actual - documented
        stale = documented - actual
        assert not undocumented, (
            f"`{name}`: options exist in the CLI but not in the README: {sorted(undocumented)}"
        )
        assert not stale, (
            f"`{name}`: options documented in the README but absent from the CLI: {sorted(stale)}"
        )
