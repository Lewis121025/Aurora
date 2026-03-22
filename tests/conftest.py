from __future__ import annotations

from collections import deque
from collections.abc import Generator
from pathlib import Path
from typing import Protocol

import pytest

from aurora.core.config import FieldConfig
from aurora.llm.provider import LLMProvider
from aurora.runtime import AuroraField, AuroraSystem, AuroraSystemConfig


class QueueLLM:
    """Deterministic step-driven LLM stub."""

    def __init__(self, *steps: str, repeat_last: bool = False) -> None:
        self._steps = deque(steps)
        self._repeat_last = repeat_last
        self._last: str | None = None
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        del max_tokens, temperature
        self.calls.append(messages)
        if self._steps:
            self._last = self._steps.popleft()
            return self._last
        if self._repeat_last and self._last is not None:
            return self._last
        raise AssertionError("QueueLLM received an unexpected extra call")


class SystemFactory(Protocol):
    def __call__(self, *, llm: LLMProvider | None = None) -> AuroraSystem: ...


class FieldFactory(Protocol):
    def __call__(self) -> AuroraField: ...


@pytest.fixture
def system_factory(tmp_path: Path) -> Generator[SystemFactory, None, None]:
    created: list[AuroraSystem] = []

    def factory(*, llm: LLMProvider | None = None) -> AuroraSystem:
        db_path = tmp_path / f"aurora-{len(created)}.sqlite"
        system = AuroraSystem(
            AuroraSystemConfig(
                db_path=str(db_path),
            ),
            llm=llm,
        )
        created.append(system)
        return system

    yield factory

    for system in created:
        system.close()


@pytest.fixture
def field_factory(tmp_path: Path) -> Generator[FieldFactory, None, None]:
    created: list[AuroraField] = []

    def factory() -> AuroraField:
        index = len(created)
        config = FieldConfig(
            data_dir=str(tmp_path / f"aurora-field-{index}"),
            db_path=str(tmp_path / f"aurora-field-{index}" / "aurora.sqlite"),
            blob_dir=str(tmp_path / f"aurora-field-{index}" / "blobs"),
        )
        field = AuroraField(config)
        created.append(field)
        return field

    yield factory
