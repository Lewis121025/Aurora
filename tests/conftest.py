from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from aurora.runtime.engine import AuroraKernel

CompilerOutput = str | Callable[[list[dict[str, str]]], str]
KernelFactory = Callable[..., AuroraKernel]


class ScriptedLLM:
    """Deterministic LLM that separates response generation from memory compilation."""

    def __init__(self, *, compiler_outputs: tuple[CompilerOutput, ...] = ()) -> None:
        self._compiler_outputs = deque(compiler_outputs)

    def complete(self, messages: list[dict[str, str]]) -> str:
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        user_text = messages[-1]["content"] if messages else ""
        if "You are Aurora's memory compiler." in system_text:
            if not self._compiler_outputs:
                return json.dumps({"ops": []}, ensure_ascii=False)
            output = self._compiler_outputs.popleft()
            return output(messages) if callable(output) else output
        return self._response(system_text, user_text)

    def _response(self, system_text: str, user_text: str) -> str:
        if "[ARCHIVE_RECALL]" in system_text:
            return f"根据记忆，{_first_recalled_content(system_text)}"
        if "不要安抚式表达，直接一点。" in system_text:
            return f"直接回答：{user_text}"
        if (
            "[OPEN_LOOPS]" in system_text
            and _first_loop_summary(system_text) != "未命名事项"
            and any(token in user_text for token in ("继续", "接着", "承诺"))
        ):
            return f"继续处理：{_first_loop_summary(system_text)}"
        return f"收到：{user_text}"


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

    def _create(*, llm: ScriptedLLM | None = None) -> AuroraKernel:
        kernel = AuroraKernel.create(data_dir=str(data_dir), llm=llm or ScriptedLLM())
        kernels.append(kernel)
        return kernel

    yield _create

    for kernel in kernels:
        try:
            kernel.close()
        except Exception:
            pass
