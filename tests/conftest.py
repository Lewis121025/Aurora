from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Protocol

import pytest

if TYPE_CHECKING:
    from aurora.llm.provider import LLMProvider
    from aurora.runtime.engine import AuroraKernel

CompilerOutput = str | Callable[[list[dict[str, str]]], str]
StructuredResponder = Callable[[list[dict[str, str]]], str]


class QueueLLM:
    """Explicit step-driven LLM stub."""

    def __init__(
        self,
        *steps: CompilerOutput,
        repeat_last: bool = False,
        structured: StructuredResponder | None = None,
    ) -> None:
        self._steps = deque(steps)
        self._repeat_last = repeat_last
        self._last_step: CompilerOutput | None = None
        self._structured = structured
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        if self._structured is not None and "[AURORA_" in system_text:
            return self._structured(messages)
        if not self._steps:
            if self._repeat_last and self._last_step is not None:
                step = self._last_step
                return step(messages) if callable(step) else step
            raise AssertionError("QueueLLM received an unexpected extra call")
        step = self._steps.popleft()
        self._last_step = step
        return step(messages) if callable(step) else step

    def assert_exhausted(self) -> None:
        if self._repeat_last:
            return
        if self._steps:
            raise AssertionError(f"QueueLLM has {len(self._steps)} unconsumed step(s)")


_PLAN_PATTERN = re.compile(r"先(.+?)(?:,|，)?再(.+)")
_LOCATION_PATTERN = re.compile(r"我(?:现在)?住在([\u4e00-\u9fffA-Za-z0-9]{1,16})")
_WORK_PATTERN = re.compile(r"我在([\u4e00-\u9fffA-Za-z0-9]{1,16})工作")
_LIKE_PATTERN = re.compile(r"(?:我|也)喜欢([^，。！？；]+)")


def scripted_memory_llm(messages: list[dict[str, str]]) -> str:
    system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
    payload = json.loads(messages[-1]["content"])
    if "[AURORA_USER_FIELD_COMPILER]" in system_text:
        return json.dumps(_compile_user_field(payload), ensure_ascii=False)
    if "[AURORA_TURN_FIELD_COMPILER]" in system_text:
        return json.dumps(_compile_turn_field(payload), ensure_ascii=False)
    raise AssertionError("structured compiler received an unknown prompt")


def _compile_user_field(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise AssertionError("compiler payload must be an object")
    text = str(payload.get("user_text", ""))
    current_field = payload.get("current_field")
    memory = current_field if isinstance(current_field, list) else []
    is_forget = any(marker in text for marker in ("请忘掉", "忘掉", "别记", "不要记", "忘了"))
    atoms: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []

    for match in _LOCATION_PATTERN.finditer(text):
        atoms.append(_field_atom("memory", f"我现在住在{match.group(1).strip()}", 0.94, 0.90))
    for match in _WORK_PATTERN.finditer(text):
        atoms.append(_field_atom("memory", f"我在{match.group(1).strip()}工作", 0.92, 0.84))
    for match in _LIKE_PATTERN.finditer(text):
        value = match.group(1).strip()
        if value and not is_forget:
            atoms.append(_field_atom("memory", f"我喜欢{value}", 0.90, 0.82))

    plan_match = _PLAN_PATTERN.search(text.replace("，", ","))
    if plan_match is not None:
        first_step = plan_match.group(1).strip(" 。！？")
        second_step = plan_match.group(2).strip(" 。！？")
        atoms.append(_field_atom("memory", f"先{first_step}，再{second_step}", 0.86, 0.88))

    if "想" in text and "不想" in text:
        atoms.append(_field_atom("memory", text.strip(" 。！？"), 0.78, 0.76))
    elif "想" in text:
        atoms.append(_field_atom("memory", text.strip(" 。！？"), 0.74, 0.72))

    emotional_trace = _emotion_trace(text)
    if emotional_trace is not None:
        atoms.append(emotional_trace)

    if any(marker in text for marker in ("更正", "其实", "改口")):
        existing_location_ids = _match_atoms(memory, lambda item: _item_kind(item) == "memory" and "住在" in _item_text(item))
        if existing_location_ids:
            for atom_index, atom in enumerate(atoms):
                if atom.get("kind") == "memory" and "住在" in str(atom.get("text", "")):
                    edges.extend(
                        _edge(f"new:{atom_index}", target_atom_id, -0.92, 0.94)
                        for target_atom_id in existing_location_ids
                    )

    if is_forget:
        atoms.append(_field_atom("inhibition", text.strip(), 0.96, 0.84))
        inhibition_index = len(atoms) - 1
        target_ids = _match_atoms(
            memory,
            lambda item: _item_kind(item) != "evidence" and bool(_item_text(item)) and _item_text(item) in text,
        )
        edges.extend(_edge(f"new:{inhibition_index}", atom_id, -0.95, 0.97) for atom_id in target_ids)

    return {"atoms": atoms, "edges": edges}


def _compile_turn_field(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise AssertionError("compiler payload must be an object")
    user_text = str(payload.get("user_text", "")).strip()
    assistant_text = str(payload.get("assistant_text", "")).strip()
    current_field = payload.get("current_field")
    memory = current_field if isinstance(current_field, list) else []
    atoms: list[dict[str, object]] = [
        _field_atom(
            "episode",
            " | ".join(
                fragment
                for fragment in (
                    f"user: {user_text}" if user_text else "",
                    f"aurora: {assistant_text}" if assistant_text else "",
                )
                if fragment
            ),
            0.98,
            0.70,
        )
    ]
    if assistant_text.startswith(("我会", "我来", "我可以")):
        action = assistant_text.removeprefix("我会").removeprefix("我来").removeprefix("我可以").strip(" ，。！？")
        if action:
            atoms.append(_field_atom("memory", f"Aurora承诺：{action}", 0.88, 0.86))
    if any(token in user_text for token in ("杭州", "生活", "适应", "朋友", "运动")):
        atoms.append(_field_atom("memory", user_text[:48], 0.78, 0.80))

    edges: list[dict[str, object]] = []
    if len(atoms) > 1:
        episode_targets = _match_atoms(memory, lambda item: _item_kind(item) == "episode")
        for atom_index in range(1, len(atoms)):
            edges.extend(_edge(f"new:{atom_index}", target_atom_id, 0.35, 0.70) for target_atom_id in episode_targets[-2:])

    return {"atoms": atoms, "edges": edges}


def _field_atom(kind: str, text: str, confidence: float, salience: float) -> dict[str, object]:
    return {
        "kind": kind,
        "text": text,
        "confidence": confidence,
        "salience": salience,
        "referents": ["subject"],
    }


def _emotion_trace(text: str) -> dict[str, object] | None:
    labels: list[str] = []
    for label, _, _ in (
        ("心情很好", 0.7, 0.76),
        ("开心", 0.7, 0.88),
        ("高兴", 0.7, 0.84),
        ("放松", 0.6, 0.66),
        ("轻松", 0.6, 0.64),
        ("难过", -0.7, 0.82),
        ("焦虑", -0.7, 0.86),
        ("紧张", -0.6, 0.78),
        ("沮丧", -0.7, 0.80),
        ("生气", -0.9, 0.90),
    ):
        if label not in text:
            continue
        labels.append(label)
    if not labels:
        return None
    return {
        "kind": "memory",
        "text": " / ".join(labels),
        "confidence": 0.82,
        "salience": 0.78,
        "referents": ["subject"],
    }


def _edge(source: str, target: str, influence: float, confidence: float) -> dict[str, object]:
    return {
        "source": source,
        "target": target,
        "influence": influence,
        "confidence": confidence,
    }


def _match_atoms(memory: list[object], predicate: Callable[[dict[str, object]], bool]) -> list[str]:
    matches: list[str] = []
    for item in memory:
        if not isinstance(item, dict):
            continue
        if predicate(item):
            matches.append(str(item.get("atom_id", "")))
    return [item for item in matches if item]


def _item_kind(item: dict[str, object]) -> str:
    return str(item.get("kind", "")).strip()


def _item_text(item: dict[str, object]) -> str:
    content = item.get("content")
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    return ""


class KernelFactory(Protocol):
    def __call__(self, *, llm: LLMProvider | None = None) -> AuroraKernel: ...

    def track(self, llm: QueueLLM) -> QueueLLM: ...


class _KernelFactory:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._kernels: list[AuroraKernel] = []
        self._tracked: list[QueueLLM] = []

    def __call__(self, *, llm: LLMProvider | None = None) -> AuroraKernel:
        if llm is None:
            raise AssertionError("kernel_factory requires an explicit llm")
        if isinstance(llm, QueueLLM):
            self._tracked.append(llm)
        from aurora.runtime.engine import AuroraKernel

        kernel = AuroraKernel.create(data_dir=str(self._data_dir), llm=llm)
        self._kernels.append(kernel)
        return kernel

    def track(self, llm: QueueLLM) -> QueueLLM:
        self._tracked.append(llm)
        return llm

    def close_all(self) -> list[BaseException]:
        close_errors: list[BaseException] = []
        for kernel in self._kernels:
            try:
                kernel.close()
            except Exception as exc:  # pragma: no cover
                close_errors.append(exc)
        return close_errors

    def assert_exhausted(self) -> None:
        for llm in self._tracked:
            llm.assert_exhausted()


@pytest.fixture
def kernel_factory(tmp_path: Path) -> Iterator[KernelFactory]:
    data_dir = tmp_path / "aurora-data"
    factory = _KernelFactory(data_dir)

    yield factory

    close_errors = factory.close_all()
    factory.assert_exhausted()

    if close_errors:
        rendered = "\n".join(f"- {type(exc).__name__}: {exc}" for exc in close_errors)
        raise AssertionError(f"kernel.close() raised {len(close_errors)} error(s):\n{rendered}")
