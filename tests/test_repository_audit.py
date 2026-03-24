from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib
from typing import cast

import yaml  # type: ignore[import-untyped]


_REPO_ROOT = Path(__file__).resolve().parents[1]
def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _tracked_file_set() -> set[str]:
    return set(_tracked_files())


def _read_repository_text(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _repository_policy() -> dict[str, object]:
    pyproject = tomllib.loads(_read_repository_text("pyproject.toml"))
    tool = _mapping(pyproject["tool"])
    aurora = _mapping(tool["aurora"])
    return _mapping(aurora["repository_policy"])


def _read_yaml_object(relative_path: str) -> dict[str, object]:
    with (_REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _has_repository_prefix(prefix: str) -> bool:
    tracked = _tracked_file_set()
    return any(path.startswith(prefix) for path in tracked)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _string_list(value: object) -> list[str]:
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return cast(list[str], value)


def _find_named_step(steps: object, name: str) -> dict[str, object]:
    for step in _list_of_mappings(steps):
        if step.get("name") == name:
            return step
    raise AssertionError(f"missing step: {name}")


def _find_dependabot_update(updates: object, ecosystem: str) -> dict[str, object]:
    for update in _list_of_mappings(updates):
        if update.get("package-ecosystem") == ecosystem:
            return update
    raise AssertionError(f"missing dependabot ecosystem: {ecosystem}")


def _list_of_mappings(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    items = [_mapping(item) for item in value]
    return items


def _makefile_targets(makefile: str) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in makefile.splitlines():
        if raw_line.startswith("\t") and current is not None:
            targets[current].append(raw_line.strip())
            continue
        current = None
        if ":" not in raw_line or raw_line.startswith(".") or raw_line.startswith("#"):
            continue
        target, _, _ = raw_line.partition(":")
        if not target or " " in target:
            continue
        current = target
        targets[current] = []
    return targets


def test_repository_audit_has_ascii_only_tracked_content() -> None:
    non_ascii_allowed_paths = _string_list(_repository_policy()["non_ascii_allowed_paths"])
    offenders: list[str] = []
    for relative_path in _tracked_files():
        if relative_path in non_ascii_allowed_paths:
            continue
        path = _REPO_ROOT / relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.isascii():
            offenders.append(relative_path)
    assert offenders == []


def test_repository_audit_has_no_dev_only_or_forbidden_tracked_paths() -> None:
    policy = _repository_policy()
    forbidden_tracked_paths = _string_list(policy["forbidden_tracked_paths"])
    forbidden_tracked_prefixes = _string_list(policy["forbidden_tracked_prefixes"])
    tracked = _tracked_files()
    offenders = [
        path
        for path in tracked
        if path in forbidden_tracked_paths or any(path.startswith(prefix) for prefix in forbidden_tracked_prefixes)
    ]
    assert offenders == []


def test_repository_audit_has_required_engineering_assets() -> None:
    policy = _repository_policy()
    required_tracked_paths = _string_list(policy["required_tracked_paths"])
    required_tracked_prefixes = _string_list(policy["required_tracked_prefixes"])
    tracked = _tracked_file_set()
    missing_paths = [path for path in required_tracked_paths if path not in tracked]
    assert missing_paths == []

    missing_prefixes = [
        prefix for prefix in required_tracked_prefixes if not _has_repository_prefix(prefix)
    ]
    assert missing_prefixes == []


def test_repository_audit_python_policy_is_consistent_across_configs() -> None:
    policy = _repository_policy()
    pyproject = tomllib.loads(_read_repository_text("pyproject.toml"))
    readme = _read_repository_text("README.md")
    contributing = _read_repository_text("CONTRIBUTING.md")
    workflow = _read_yaml_object(".github/workflows/release-validation.yml")
    dependabot = _read_yaml_object(".github/dependabot.yml")
    makefile = _read_repository_text("Makefile")
    make_targets = _makefile_targets(makefile)

    requires_python = cast(str, policy["requires_python"])
    mypy_python_version = cast(str, policy["mypy_python_version"])
    blocking_python_versions = _string_list(policy["blocking_python_versions"])
    future_smoke_python = cast(str, policy["future_smoke_python"])
    user_install_command = cast(str, policy["user_install_command"])
    developer_sync_command = cast(str, policy["developer_sync_command"])
    validation_command = cast(str, policy["validation_command"])
    pre_commit_command = cast(str, policy["pre_commit_command"])
    dependabot_ecosystems = _string_list(policy["dependabot_ecosystems"])
    dependabot_groups = _string_list(policy["dependabot_groups"])
    developer_make_targets = _string_list(policy["developer_make_targets"])

    assert pyproject["project"]["requires-python"] == requires_python
    assert pyproject["tool"]["mypy"]["python_version"] == mypy_python_version
    assert "pre-commit>=4.5.1" in pyproject["project"]["optional-dependencies"]["dev"]
    assert "pyyaml>=6.0.3" in pyproject["project"]["optional-dependencies"]["dev"]

    assert "Aurora requires Python 3.13 or newer." in readme
    assert user_install_command in readme
    assert developer_sync_command not in readme
    assert validation_command not in readme
    assert pre_commit_command not in readme

    assert "Python 3.13 or newer" in contributing
    assert "make sync" in contributing
    assert developer_sync_command in contributing
    assert validation_command in contributing
    assert pre_commit_command in contributing

    jobs = _mapping(workflow["jobs"])
    validation = _mapping(jobs["validation"])
    validation_strategy = _mapping(validation["strategy"])
    validation_matrix = _mapping(validation_strategy["matrix"])
    assert _string_list(validation_matrix["python-version"]) == blocking_python_versions
    validation_steps = _list_of_mappings(validation["steps"])
    assert cast(str, _find_named_step(validation_steps, "Sync dependencies")["run"]) == developer_sync_command
    assert cast(str, _find_named_step(validation_steps, "Run repository checks")["run"]) == validation_command
    assert cast(str, _find_named_step(validation_steps, "Run pre-commit")["run"]) == pre_commit_command

    future_smoke = _mapping(jobs["future-smoke"])
    assert future_smoke["continue-on-error"] is True
    future_setup_python = _find_named_step(future_smoke["steps"], "Set up Python")
    future_setup_inputs = _mapping(future_setup_python["with"])
    assert cast(str, future_setup_inputs["python-version"]) == future_smoke_python
    assert future_setup_inputs["allow-prereleases"] is bool(policy["future_smoke_allow_prereleases"])
    future_steps = _list_of_mappings(future_smoke["steps"])
    assert cast(str, _find_named_step(future_steps, "Sync dependencies")["run"]) == developer_sync_command
    assert cast(str, _find_named_step(future_steps, "Import package")["run"]) == 'uv run python -c "import aurora"'
    assert cast(str, _find_named_step(future_steps, "Run tests")["run"]) == "uv run pytest -q"

    updates = _list_of_mappings(dependabot["updates"])
    actual_ecosystems = [cast(str, update["package-ecosystem"]) for update in updates]
    assert actual_ecosystems == dependabot_ecosystems
    uv_update = _find_dependabot_update(updates, "uv")
    groups = _mapping(uv_update["groups"])
    assert list(groups) == dependabot_groups

    assert make_targets["sync"] == [developer_sync_command]
    for target in developer_make_targets:
        assert target in make_targets
    assert "check: compile lint typecheck test" in makefile


def test_repository_audit_gitignore_covers_local_state_without_blocking_docs() -> None:
    required_gitignore_patterns = _string_list(_repository_policy()["required_gitignore_patterns"])
    gitignore_lines = {
        line.strip()
        for line in _read_repository_text(".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "docs/" not in gitignore_lines
    for pattern in required_gitignore_patterns:
        assert pattern in gitignore_lines
