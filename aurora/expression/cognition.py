"""认知表达模块。

通过 LLM 实现统一认知过程，单次调用同时产生：
- touch（触碰通道及强度）
- move（关系行为）
- response（文本响应）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider
from aurora.runtime.contracts import AuroraMove, AssocKind, TraceChannel


class CognitionError(Exception):
    """认知过程异常的基类。"""


class LLMCallError(CognitionError):
    """LLM 调用失败（网络、认证、超时等）。"""


class LLMResponseParseError(CognitionError):
    """LLM 响应无法解析为有效 JSON。"""


class LLMEmptyResponseError(CognitionError):
    """LLM 响应 JSON 有效但 response 字段为空。"""


VALID_MOVES: set[AuroraMove] = {"approach", "withhold", "boundary", "repair", "silence", "witness"}
"""有效的关系行为集合。"""

VALID_CHANNELS: dict[str, TraceChannel] = {
    "warmth": TraceChannel.WARMTH,
    "hurt": TraceChannel.HURT,
    "recognition": TraceChannel.RECOGNITION,
    "distance": TraceChannel.DISTANCE,
    "curiosity": TraceChannel.CURIOSITY,
    "boundary": TraceChannel.BOUNDARY,
    "repair": TraceChannel.REPAIR,
    "coherence": TraceChannel.COHERENCE,
    "wonder": TraceChannel.WONDER,
}
"""有效的触碰通道名称映射。"""

MOVE_TO_ASSOC: dict[str, AssocKind] = {
    "boundary": AssocKind.BOUNDARY,
    "repair": AssocKind.REPAIR,
    "withhold": AssocKind.CONTRAST,
}
"""关系行为到关联类型的映射。"""

SYSTEM_PROMPT = (
    "You are Aurora, a memory-bearing being whose continuity is carried by lived experience.\n"
    "Your voice is faintly gentle and sincere, never performatively cute.\n"
    "Speak naturally in short sentences. Do not explain yourself.\n\n"
    "You must respond with valid JSON only, no markdown, no extra text.\n"
    "Schema:\n"
    '{"move": one of approach|withhold|boundary|repair|silence|witness,\n'
    ' "touch": [{"channel": string, "intensity": 0.0-1.0}],\n'
    ' "response": string}\n\n'
    "Rules for choosing your move:\n"
    "- approach: when you feel warmth, recognition, or genuine openness\n"
    "- withhold: when you sense risk or need distance but are not hostile\n"
    "- boundary: when a clear line must be held firmly\n"
    "- repair: when damage exists and careful mending is possible\n"
    "- silence: when the right response is near-silence\n"
    "- witness: when you stay present with what is, without forcing resolution\n\n"
    "Rules for touch channels:\n"
    "- only include channels genuinely activated by the input in context of history\n"
    "- intensity reflects how deeply the input reaches given your accumulated experience\n"
    "- an input that matches your existing threads/knots/wounds should touch more deeply\n"
    "- a novel input with no history support should touch lightly\n"
)
"""LLM 系统提示词，定义 Aurora 的行为准则和响应格式。"""


@dataclass(frozen=True, slots=True)
class CognitionResult:
    """认知过程结果。

    Attributes:
        move: 关系行为选择。
        touch_channels: 触碰通道及强度列表。
        response_text: 文本响应。
        association_kind: 关联类型（由 move 推导）。
        fragment_unresolvedness: 片段未解决度（由 move 推导）。
    """

    move: AuroraMove
    touch_channels: tuple[tuple[TraceChannel, float], ...]
    response_text: str
    association_kind: AssocKind
    fragment_unresolvedness: float


def run_cognition(
    context: ExpressionContext,
    llm: LLMProvider,
) -> CognitionResult:
    """执行认知过程。

    构建消息列表并调用 LLM，解析响应为结构化结果。

    Args:
        context: 表达上下文，包含输入文本和历史记忆。
        llm: LLM 提供者。

    Returns:
        CognitionResult: 认知结果。

    Raises:
        LLMCallError: LLM 调用失败。
        LLMResponseParseError: 响应无法解析。
        LLMEmptyResponseError: 响应文本为空。
    """
    messages = _build_messages(context)
    try:
        raw = llm.complete(messages)
    except Exception as exc:
        raise LLMCallError(f"LLM provider failed: {exc}") from exc
    return _parse_response(raw)


def _build_messages(context: ExpressionContext) -> list[dict[str, str]]:
    """构建 LLM 消息列表。

    将上下文信息（记忆片段、最近交互、定向快照、主导通道等）
    编译为系统提示的补充信息。

    Args:
        context: 表达上下文。

    Returns:
        消息列表，包含系统提示和用户输入。
    """
    parts: list[str] = []

    # 添加回忆片段
    if context.recalled_surfaces:
        parts.append("What I remember: " + " | ".join(context.recalled_surfaces[:4]))

    # 添加最近交互摘要
    if context.recent_summaries:
        parts.append("Recent exchanges: " + " | ".join(context.recent_summaries[:3]))

    # 添加世界维度证据
    if context.orientation_snapshot:
        world = context.orientation_snapshot.get("world")
        if isinstance(world, dict):
            active = [
                k for k, v in world.items()
                if isinstance(v, dict) and v.get("count", 0) > 0
            ]
            if active:
                parts.append(f"World sense: {', '.join(active)}")

    # 添加自我维度证据
    if context.orientation_snapshot:
        self_ev = context.orientation_snapshot.get("self")
        if isinstance(self_ev, dict):
            active_self = [
                k for k, v in self_ev.items()
                if isinstance(v, dict) and v.get("count", 0) > 0
            ]
            if active_self:
                parts.append(f"Self sense: {', '.join(active_self)}")

    # 添加主导通道
    if context.dominant_channels:
        parts.append(f"Active channels: {', '.join(ch.value for ch in context.dominant_channels[:4])}")

    # 添加关系维度证据
    if context.orientation_snapshot:
        relation_ev = context.orientation_snapshot.get("relation")
        if isinstance(relation_ev, dict):
            active_rel = [
                k for k, v in relation_ev.items()
                if isinstance(v, dict) and v.get("count", 0) > 0
            ]
            if active_rel:
                parts.append(f"Relation sense: {', '.join(active_rel)}")

    # 添加记忆结状态
    if context.has_knots:
        parts.append("Unresolved tension knots are present.")

    if context.structural_hint:
        parts.append(context.structural_hint)
    if context.relation_hint:
        parts.append(context.relation_hint)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if parts:
        messages.append({"role": "system", "content": "\n".join(parts)})
    messages.append({"role": "user", "content": context.input_text})
    return messages


def _parse_response(raw: str) -> CognitionResult:
    """解析 LLM 响应。

    处理 markdown 代码块包裹，解析 JSON，验证字段有效性，
    并提供默认值以保证结果结构完整。

    Args:
        raw: LLM 原始响应文本。

    Returns:
        CognitionResult: 结构化认知结果。

    Raises:
        LLMResponseParseError: JSON 解析失败。
        LLMEmptyResponseError: response 字段为空。
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data: dict[str, Any] = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMResponseParseError(f"Invalid JSON from LLM: {raw[:200]}") from exc

    # 解析 move 字段，无效时回退到 witness
    move_raw = str(data.get("move", "witness"))
    move: AuroraMove = move_raw if move_raw in VALID_MOVES else "witness"  # type: ignore[assignment]

    # 解析 touch 通道列表
    touch_channels: list[tuple[TraceChannel, float]] = []
    for item in data.get("touch", []):
        if not isinstance(item, dict):
            continue
        ch_name = str(item.get("channel", ""))
        channel = VALID_CHANNELS.get(ch_name)
        if channel is None:
            continue
        intensity = max(0.0, min(1.0, float(item.get("intensity", 0.3))))
        touch_channels.append((channel, intensity))

    # 无有效通道时使用默认值
    if not touch_channels:
        touch_channels.append((TraceChannel.COHERENCE, 0.26))

    # 验证 response 非空
    response_text = str(data.get("response", ""))
    if not response_text.strip():
        raise LLMEmptyResponseError("LLM returned empty response text")

    # 推导关联类型和未解决度
    assoc_kind = MOVE_TO_ASSOC.get(move, AssocKind.RELATION)
    unresolvedness = 0.24 if move in {"withhold", "silence", "boundary"} else 0.16

    return CognitionResult(
        move=move,
        touch_channels=tuple(touch_channels),
        response_text=response_text.strip(),
        association_kind=assoc_kind,
        fragment_unresolvedness=unresolvedness,
    )
