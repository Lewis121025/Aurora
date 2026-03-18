from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from aurora.runtime.engine import AuroraKernel

CompilerOutput = str | Callable[[list[dict[str, str]]], str]
KernelFactory = Callable[..., AuroraKernel]


class QueueLLM:
    """Explicit step-driven LLM stub.

    Tests must provide one step per `complete()` call, avoiding any guesses about
    whether a call is cognition or compilation.
    """

    def __init__(self, *steps: CompilerOutput) -> None:
        self._steps = deque(steps)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self._steps:
            raise AssertionError("QueueLLM received an unexpected extra call")
        step = self._steps.popleft()
        return step(messages) if callable(step) else step

    def assert_exhausted(self) -> None:
        if self._steps:
            raise AssertionError(f"QueueLLM has {len(self._steps)} unconsumed step(s)")

def _first_recalled_content(system_text: str) -> str:
    for line in system_text.splitlines():
        if line.startswith("- ("):
            return line.split(") ", 1)[1].rsplit(" [", 1)[0]
    return "没有命中"


def _first_loop_summary(system_text: str) -> str:
    for line in system_text.splitlines():
        if line.startswith("- "):
            return line.split(": ", 1)[1].rsplit(" (urgency=", 1)[0]
    return "未命名事项"


@pytest.fixture
def kernel_factory(tmp_path: Path) -> Iterator[KernelFactory]:
    data_dir = tmp_path / "aurora-data"
    kernels: list[AuroraKernel] = []
    tracked: list[QueueLLM] = []

    def _create(*, llm: object | None = None) -> AuroraKernel:
        if llm is None:
            raise AssertionError("kernel_factory requires an explicit llm")
        if isinstance(llm, QueueLLM):
            tracked.append(llm)
        kernel = AuroraKernel.create(data_dir=str(data_dir), llm=llm)  # type: ignore[arg-type]
        kernels.append(kernel)
        return kernel

    def _track(llm: QueueLLM) -> QueueLLM:
        """Register an extra QueueLLM for teardown exhaustion checks.

        Some tests swap `kernel.llm` mid-test to drive follow-up turns; those
        stubs should still be required to consume all queued steps.
        """

        tracked.append(llm)
        return llm

    # Allow tests to register additional QueueLLM instances without global state.
    _create.track = _track  # type: ignore[attr-defined]

    yield _create

    close_errors: list[BaseException] = []
    for kernel in kernels:
        try:
            kernel.close()
        except Exception as exc:  # pragma: no cover
            close_errors.append(exc)

    for llm in tracked:
        llm.assert_exhausted()

    if close_errors:
        rendered = "\n".join(f"- {type(exc).__name__}: {exc}" for exc in close_errors)
        raise AssertionError(f"kernel.close() raised {len(close_errors)} error(s):\n{rendered}")
