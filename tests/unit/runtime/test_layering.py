from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_DIR = ROOT / "aurora" / "runtime"
INTERFACES_DIR = ROOT / "aurora" / "interfaces"


def test_runtime_does_not_depend_on_benchmarks():
    offenders = []
    for path in RUNTIME_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        if "aurora.benchmarks" in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_runtime_and_interfaces_do_not_depend_on_lab_or_legacy_core_paths():
    offenders = []
    forbidden = (
        "aurora.lab",
        "aurora.core",
        "aurora.core.soul_memory",
        "aurora.core.retrieval",
        "aurora.core.graph",
        "aurora.core.models",
    )
    for base in (RUNTIME_DIR, INTERFACES_DIR):
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text()
            if any(token in text for token in forbidden):
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_top_level_package_stays_thin():
    import aurora

    assert aurora.__version__
    assert not hasattr(aurora, "AuroraSoul")
    assert not hasattr(aurora, "AuroraRuntime")
