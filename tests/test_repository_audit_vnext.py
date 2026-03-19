from __future__ import annotations

from pathlib import Path
import subprocess


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_TRACKED_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    "claude.md",
)
_FORBIDDEN_TRACKED_PREFIXES = (
    "docs/",
    "tests/test_live_",
    "tests/test_surface_",
    "tests/integration/",
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_repository_audit_has_ascii_only_tracked_content() -> None:
    offenders: list[str] = []
    for relative_path in _tracked_files():
        path = _REPO_ROOT / relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.isascii():
            offenders.append(relative_path)
    assert offenders == []


def test_repository_audit_has_no_dev_only_or_forbidden_tracked_paths() -> None:
    tracked = _tracked_files()
    offenders = [
        path
        for path in tracked
        if path in _FORBIDDEN_TRACKED_PATHS or any(path.startswith(prefix) for prefix in _FORBIDDEN_TRACKED_PREFIXES)
    ]
    assert offenders == []
