from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Protocol, cast

import pytest

if TYPE_CHECKING:
    from aurora.llm.provider import LLMProvider
    from aurora.runtime.engine import AuroraKernel

CompilerOutput = str | Callable[[list[dict[str, str]]], str]
StructuredResponder = Callable[[list[dict[str, str]]], str]


class QueueLLM:
    """Explicit step-driven LLM stub.

    Tests can queue fixed outputs or set `repeat_last=True` for open-ended call
    patterns without touching a real model.
    """

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
    if "[AURORA_USER_MEMORY_COMPILER]" in system_text:
        return json.dumps(_compile_user_payload(payload), ensure_ascii=False)
    if "[AURORA_TURN_MEMORY_COMPILER]" in system_text:
        return json.dumps(_compile_turn_payload(payload), ensure_ascii=False)
    raise AssertionError("structured compiler received an unknown prompt")


def _compile_user_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise AssertionError("compiler payload must be an object")
    text = str(payload.get("user_text", ""))
    current_memory = payload.get("current_memory")
    memory = current_memory if isinstance(current_memory, list) else []
    semantic: list[dict[str, object]] = []
    procedural: list[dict[str, object]] = []
    inhibitions: list[dict[str, object]] = []

    for match in _LOCATION_PATTERN.finditer(text):
        value = match.group(1).strip()
        semantic.append(
            {
                "subject": "user",
                "attribute": "location.current",
                "scope": "self",
                "value": value,
                "text": f"我现在住在{value}",
            }
        )
    for match in _WORK_PATTERN.finditer(text):
        value = match.group(1).strip()
        semantic.append(
            {
                "subject": "user",
                "attribute": "work.location",
                "scope": "self",
                "value": value,
                "text": f"我在{value}工作",
            }
        )
    for match in _LIKE_PATTERN.finditer(text):
        value = match.group(1).strip()
        if not value:
            continue
        semantic.append(
            {
                "subject": "user",
                "attribute": "preference.like",
                "scope": "self",
                "value": value,
                "text": f"我喜欢{value}",
            }
        )

    plan_match = _PLAN_PATTERN.search(text.replace("，", ","))
    if plan_match is not None:
        first_step = plan_match.group(1).strip(" 。！？")
        second_step = plan_match.group(2).strip(" 。！？")
        procedural.append(
            {
                "rule": f"先{first_step}，再{second_step}",
                "trigger": "plan",
                "steps": [first_step, second_step],
                "text": f"先{first_step}，再{second_step}",
                "owner": None,
            }
        )

    beliefs = [f"我住在{match.group(1)}" for match in _LOCATION_PATTERN.finditer(text)]
    goals = [text.split("想", 1)[-1].strip("，。！？ ") for _ in [1] if "想" in text and text.split("想", 1)[-1].strip("，。！？ ")]
    conflicts = [clause.strip() for clause in _clauses(text) if "不想" in clause]
    intentions: list[str] = []
    commitments: list[str] = []
    if plan_match is not None:
        intentions.append(plan_match.group(1).strip(" 。！？"))
        commitments.append(plan_match.group(2).strip(" 。！？"))

    cognitive: dict[str, object] | None = None
    if any((beliefs, goals, conflicts, intentions, commitments)):
        cognitive = {
            "beliefs": beliefs,
            "goals": goals,
            "conflicts": conflicts,
            "intentions": intentions,
            "commitments": commitments,
        }

    affective = _affective_payload(text)

    if any(marker in text for marker in ("请忘掉", "忘掉", "别记", "不要记", "忘了")):
        target_text = text
        target_ids = [
            str(item.get("atom_id"))
            for item in memory
            if isinstance(item, dict)
            and _memory_matches_target(item, target_text)
        ]
        if target_ids:
            inhibitions.append(
                {
                    "summary": text.strip(),
                    "target_summary": text.split("忘", 1)[-1].strip(" ：:。！？"),
                    "target_atom_ids": target_ids,
                }
            )

    return {
        "semantic": semantic,
        "procedural": procedural,
        "cognitive": cognitive,
        "affective": affective,
        "inhibitions": inhibitions,
    }


def _compile_turn_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise AssertionError("compiler payload must be an object")
    user_text = str(payload.get("user_text", "")).strip()
    assistant_text = str(payload.get("assistant_text", "")).strip()
    episode = {
        "title": (user_text or assistant_text)[:24] or "interaction",
        "summary": " | ".join(
            fragment
            for fragment in (
                f"user: {user_text}" if user_text else "",
                f"aurora: {assistant_text}" if assistant_text else "",
            )
            if fragment
        )
        or "interaction",
        "actors": ["user", "aurora", *_actors(user_text)],
        "setting": _setting(user_text),
        "emotion_markers": _episode_markers(user_text),
        "text": " | ".join(
            fragment
            for fragment in (
                f"user: {user_text}" if user_text else "",
                f"aurora: {assistant_text}" if assistant_text else "",
            )
            if fragment
        )
        or "interaction",
    }
    procedural: list[dict[str, object]] = []
    if assistant_text.startswith(("我会", "我来", "我可以")):
        action = assistant_text.removeprefix("我会").removeprefix("我来").removeprefix("我可以").strip(" ，。！？")
        if action:
            procedural.append(
                {
                    "rule": f"Aurora承诺：{action}",
                    "trigger": "assistant_commitment",
                    "steps": [action],
                    "text": f"Aurora承诺：{action}",
                    "owner": "aurora",
                }
            )
    narrative = _narrative_payload(user_text)
    return {
        "episode": episode,
        "procedural": procedural,
        "narrative": [] if narrative is None else [narrative],
    }


def _affective_payload(text: str) -> dict[str, object] | None:
    feelings: list[str] = []
    valence = 0.0
    intensity = 0.0
    for label, current_valence, current_intensity in (
        ("心情很好", 0.7, 0.78),
        ("开心", 0.7, 0.88),
        ("高兴", 0.7, 0.84),
        ("放松", 0.7, 0.66),
        ("轻松", 0.7, 0.64),
        ("难过", -0.7, 0.82),
        ("焦虑", -0.7, 0.86),
        ("紧张", -0.7, 0.76),
        ("沮丧", -0.7, 0.80),
        ("生气", -0.7, 0.90),
    ):
        if label not in text:
            continue
        feelings.append(label)
        valence += current_valence
        intensity += current_intensity
    if not feelings:
        return None
    valence /= len(feelings)
    intensity /= len(feelings)
    mood = "mixed" if valence == 0.0 else "positive" if valence > 0.0 else "negative"
    return {
        "mood": mood,
        "valence": valence,
        "intensity": intensity,
        "feelings": feelings,
        "text": f"{mood}: {'/'.join(feelings)}",
    }


def _episode_markers(text: str) -> list[dict[str, object]]:
    affective = _affective_payload(text)
    if affective is None:
        return []
    intensity = cast(float, affective["intensity"])
    valence = cast(float, affective["valence"])
    feelings = cast(list[str], affective["feelings"])
    return [
        {
            "label": feeling,
            "intensity": intensity,
            "valence": valence,
        }
        for feeling in feelings
    ]


def _setting(text: str) -> str:
    for pattern in (
        re.compile(r"(?:住在|搬到|在)([\u4e00-\u9fff]{1,12}?)(?=工作|生活|住|见|待|，|。|！|？|；|$)"),
        re.compile(r"(杭州|上海|北京|深圳|广州|苏州|成都|南京|武汉|西安)"),
    ):
        match = pattern.search(text)
        if match is not None:
            return match.group(1)
    return "未指明"


def _actors(text: str) -> list[str]:
    actors: list[str] = []
    if "老朋友" in text:
        actors.append("老朋友")
    elif "朋友" in text:
        actors.append("朋友")
    if "团队" in text:
        actors.append("团队")
    return actors


def _narrative_payload(text: str) -> dict[str, object] | None:
    if ("杭州" in text or any(token in text for token in ("朋友", "运动", "周末"))) and any(
        token in text for token in ("适应", "重新", "开始", "生活", "朋友", "运动", "工作")
    ):
        return {
            "theme": "在杭州重建生活",
            "storyline": text[:40],
            "status": "active",
            "unresolved_threads": ["工作节奏"] if "工作节奏" in text else [],
            "role_changes": ["new resident"] if "搬到杭州" in text else [],
            "text": f"在杭州重建生活 - {text[:40]}",
        }
    if "搬到" in text:
        return {
            "theme": "迁居后的调整",
            "storyline": text[:40],
            "status": "active",
            "unresolved_threads": [],
            "role_changes": ["new resident"],
            "text": f"迁居后的调整 - {text[:40]}",
        }
    return None


def _memory_matches_target(item: object, text: str) -> bool:
    if not isinstance(item, dict):
        return False
    content = item.get("content")
    if not isinstance(content, dict):
        return False
    content_text = str(content.get("text", ""))
    if content_text and content_text in text:
        return True
    value = str(content.get("value", ""))
    return bool(value and value in text)


def _clauses(text: str) -> tuple[str, ...]:
    return tuple(fragment.strip() for fragment in re.split(r"[，。！？；;\n]+", text) if fragment.strip())


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
