from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_DIR = ROOT / 'aurora' / 'runtime'


def test_runtime_does_not_depend_on_benchmarks():
    offenders = []
    for path in RUNTIME_DIR.rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        text = path.read_text()
        if 'aurora.benchmarks' in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
