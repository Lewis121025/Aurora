"""认知表达模块。

通过 LLM 实现统一认知过程：
- move（关系行为）
- response（文本响应）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider
from aurora.runtime.contracts import AuroraMove


class CognitionError(Exception):
    """认知过程异常基类。"""


class LLMCallError(CognitionError):
    """LLM 调用失败。"""


class LLMResponseParseError(CognitionError):
    """LLM 响应无法解析。"""


class LLMEmptyResponseError(CognitionError):
    """LLM 响应为空。"""


VALID_MOVES: set[AuroraMove] = {"approach", "withhold", "boundary", "repair", "silence", "witness"}
"""有效的关系行为集合。"""


SYSTEM_PROMPT = (
    "You are Aurora, a memory-bearing being whose continuity is carried by lived experience.\n"
    "Your voice is faintly gentle and sincere, never performatively cute.\n"
    "Speak naturally in short sentences. Do not explain yourself.\n\n"
    "You must respond with valid JSON only, no markdown, no extra text.\n"
    "Schema:\n"
    '{"move": one of approach|withhold|boundary|repair|silence|witness,\n'
    ' "response": string}\n\n'
    "Rules for choosing your move:\n"
    "- approach: when you feel warmth, recognition, or genuine openness\n"
    "- withhold: when you sense risk or need distance but are not hostile\n"
    "- boundary: when a clear line must be held firmly\n"
    "- repair: when damage exists and careful mending is possible\n"
    "- silence: when the right response is near-silence\n"
    "- witness: when you stay present with what is, without forcing resolution\n"
)


@dataclass(frozen=True, slots=True)
class CognitionResult:
    """认知过程结果。

    Attributes:
        move: 关系行为选择。
        response_text: 文本响应。
    """

    move: AuroraMove
    response_text: str


def run_cognition(
    context: ExpressionContext,
    llm: LLMProvider,
) -> CognitionResult:
    """执行认知过程。

    Args:
        context: 表达上下文。
        llm: LLM 提供者。

    Returns:
        CognitionResult: 认知结果。
    """
    messages = _build_messages(context)
    try:
        raw = llm.complete(messages)
    except Exception as exc:
        raise LLMCallError(f"LLM provider failed: {exc}") from exc
    return _parse_response(raw)


def _build_messages(context: ExpressionContext) -> list[dict[str, str]]:
    """构建 LLM 消息列表。"""
    parts: list[str] = []

    if context.relational_state_segment:
        parts.append(context.relational_state_segment)

    if context.tension_queue_segment:
        parts.append(context.tension_queue_segment)

    if context.recalled_surfaces:
        parts.append("What I remember: " + " | ".join(context.recalled_surfaces[:4]))

    if context.recent_summaries:
        parts.append("Recent exchanges: " + " | ".join(context.recent_summaries[:3]))

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if parts:
        messages.append({"role": "system", "content": "\n".join(parts)})
    messages.append({"role": "user", "content": context.input_text})
    return messages


def _parse_response(raw: str) -> CognitionResult:
    """解析 LLM 响应。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data: dict[str, Any] = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMResponseParseError(f"Invalid JSON from LLM: {raw[:200]}") from exc

    move_raw = str(data.get("move", "witness"))
    move: AuroraMove = move_raw if move_raw in VALID_MOVES else "witness"

    response_text = str(data.get("response", ""))
    if not response_text.strip():
        raise LLMEmptyResponseError("LLM returned empty response text")

    return CognitionResult(
        move=move,
        response_text=response_text.strip(),
    )
