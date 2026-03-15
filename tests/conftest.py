from __future__ import annotations

import json

import pytest

from aurora.llm.provider import LLMProvider


class StubLLM:
    """Deterministic LLM for testing. Returns structured cognition JSON."""

    def __init__(self, move: str = "witness", channel: str = "coherence", intensity: float = 0.4) -> None:
        self._move = move
        self._channel = channel
        self._intensity = intensity

    def complete(self, messages: list[dict[str, str]]) -> str:
        user_text = messages[-1]["content"] if messages else ""
        return json.dumps({
            "move": self._move,
            "touch": [{"channel": self._channel, "intensity": self._intensity}],
            "response": f"[aurora] {user_text[:60]}",
        })


class ContextAwareLLM:
    """LLM that picks move based on keywords in user input AND system context."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        user_text = messages[-1]["content"] if messages else ""
        context_text = " ".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        move, channels = self._infer(user_text, context_text)
        return json.dumps({
            "move": move,
            "touch": [{"channel": ch, "intensity": it} for ch, it in channels],
            "response": f"[aurora:{move}] {user_text[:60]}",
        })

    @staticmethod
    def _infer(
        text: str, context: str,
    ) -> tuple[str, list[tuple[str, float]]]:
        if any(w in text for w in ("边界", "停", "不要", "boundary", "stop")):
            return "boundary", [("boundary", 0.7)]
        if any(w in text for w in ("谢谢", "感谢", "理解", "thank")):
            return "approach", [("warmth", 0.6), ("recognition", 0.5)]
        if any(w in text for w in ("受伤", "伤害", "hurt", "pain")):
            return "withhold", [("hurt", 0.6)]
        if any(w in text for w in ("修复", "repair", "sorry", "对不起")):
            return "repair", [("repair", 0.6)]
        if "World sense:" in context and "risk" in context:
            return "withhold", [("hurt", 0.5), ("distance", 0.4)]
        if "What I remember:" in context:
            return "approach", [("recognition", 0.5)]
        return "witness", [("coherence", 0.4)]


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def context_llm() -> ContextAwareLLM:
    return ContextAwareLLM()
