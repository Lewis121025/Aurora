#!/usr/bin/env python3
"""
代码质量检查脚本
==================

运行所有质量检查：类型检查、代码风格、测试覆盖率。

用法：
    python scripts/quality/check.py [--fix] [--strict]

选项：
    --fix: 自动修复可修复的问题
    --strict: 使用严格模式（用于 CI）
"""

import subprocess
import sys
from pathlib import Path


def run_command(
    cmd: list[str],
    description: str,
    project_root: Path,
    allow_failure: bool = False,
) -> bool:
    """运行命令并报告结果"""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, cwd=project_root, capture_output=False)

    if result.returncode != 0:
        print(f"❌ {description} failed")
        if not allow_failure:
            return False
    else:
        print(f"✓ {description} passed")

    return True


def main():
    fix = "--fix" in sys.argv
    strict = "--strict" in sys.argv

    project_root = Path(__file__).resolve().parents[2]

    checks = []

    # 1. 基础语法检查
    compile_cmd = [sys.executable, "-m", "compileall", "-q", "aurora"]
    checks.append((compile_cmd, "Python bytecode compilation", False))

    # 2. 代码风格检查
    ruff_cmd = [sys.executable, "-m", "ruff", "check", "aurora", "tests"]
    if fix:
        ruff_cmd.append("--fix")
    checks.append((ruff_cmd, "Ruff linting", False))

    # 3. 类型检查
    mypy_cmd = [sys.executable, "-m", "mypy", "aurora"]
    if strict:
        mypy_cmd.extend(["--strict", "--warn-unreachable"])
    checks.append((mypy_cmd, "Type checking (mypy)", False))

    # 4. 测试覆盖率
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--cov=aurora",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-fail-under=70" if strict else "--cov-fail-under=60",
    ]
    checks.append((pytest_cmd, "Test coverage", False))

    # 5. 复杂度检查
    radon_cmd = [sys.executable, "-m", "radon", "cc", "aurora", "-a", "-nb"]
    checks.append((radon_cmd, "Cyclomatic complexity", True))

    # 运行所有检查
    failed = []
    for cmd, description, allow_failure in checks:
        if not run_command(cmd, description, project_root, allow_failure):
            failed.append(description)

    # 总结
    print(f"\n{'=' * 60}")
    print("Quality Check Summary")
    print(f"{'=' * 60}")

    if failed:
        print(f"❌ {len(failed)} check(s) failed:")
        for check in failed:
            print(f"  - {check}")
        sys.exit(1)
    else:
        print("✓ All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
