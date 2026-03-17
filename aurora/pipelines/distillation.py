"""蒸馏管道模块。

实现认知蒸馏（Cognitive Distillation）：
- 输入：会话内所有对话轮次
- 输出：Profile.json Patch + Fact 列表 + Tension 条目
- 触发：阈值触发或会话结束

核心逻辑：通过 LLM 分析对话，提取用户偏好、事实冲突、未解决张力。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aurora.llm.provider import LLMProvider
from aurora.memory.ledger import AtomicFact
from aurora.relation.state import RelationalState
from aurora.relation.tension import TensionQueue


@dataclass(frozen=True, slots=True)
class DistillationPatch:
    """蒸馏补丁。

    Attributes:
        intimacy_delta: 亲密度变化。
        new_vibe: 新氛围（可选）。
        new_rules: 新规则列表。
        facts: 原子事实列表。
        tensions: 新张力条目列表。
    """

    intimacy_delta: int | None = None
    new_vibe: str | None = None
    new_rules: list[str] | None = None
    facts: list[str] | None = None
    tensions: list[dict[str, Any]] | None = None


DISTILLATION_SYSTEM_PROMPT = """You are Aurora's cognitive distillation system. Analyze a conversation and extract meaningful updates.

Output a JSON patch:
{
  "intimacy_delta": number|null,        // -1 to +1
  "new_vibe": string|null,             // one word if vibe shifted
  "new_rules": string[]|null,          // interaction rules/preferences discovered
  "facts": string[]|null,               // atomic facts about the user
  "tensions": [
    {
      "topic": string,
      "urgency": number,                // 0.0-1.0
      "prompt": string
    }
  ]|null
}

Rules:
- Only include fields that actually changed.
- intimacy_delta: small (-1, 0, or +1).
- facts: atomic, specific, memorable. E.g. "用户喜欢深色主题" not "用户喜欢编程".
- If a new fact CONTRADICTS a known fact listed below, output a tension with high urgency instead of silently overwriting. Aurora should express doubt or seek clarification, not blindly accept.
- new_rules: genuine preferences. E.g. "用户讨厌被安抚式回应".
- tensions: genuine conflicts, interruptions, unfinished business, or fact contradictions.
"""


def distill_session(
    conversation: list[tuple[str, str]],
    relation_id: str,
    current_state: RelationalState,
    existing_facts: list[AtomicFact],
    now_ts: float,
    llm: LLMProvider,
) -> DistillationPatch:
    """蒸馏会话。

    通过 LLM 分析会话，提取规则/事实/张力。
    已知事实传入以供 LLM 检测矛盾（认知摩擦）。

    Args:
        conversation: 对话列表 [(user_text, aurora_response), ...]。
        relation_id: 关系 ID。
        current_state: 当前关系状态。
        existing_facts: 已知原子事实（来自 ObjectiveLedger）。
        now_ts: 当前时间戳。
        llm: LLM 提供者。

    Returns:
        DistillationPatch: 蒸馏补丁。
    """
    if not conversation:
        return DistillationPatch()

    conversation_text = _format_conversation(conversation)
    profile_text = _format_profile(current_state)
    facts_text = _format_existing_facts(existing_facts)

    messages = [
        {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
        {"role": "system", "content": f"Current Profile:\n{profile_text}"},
    ]
    if facts_text:
        messages.append({"role": "system", "content": f"Known facts:\n{facts_text}"})
    messages.append({"role": "user", "content": f"Conversation to analyze:\n{conversation_text}"})

    try:
        raw = llm.complete(messages)
        return _parse_patch(raw)
    except Exception:
        return DistillationPatch()


def _format_existing_facts(facts: list[AtomicFact]) -> str:
    """格式化已知事实供 LLM 参考（认知摩擦检测）。"""
    if not facts:
        return ""
    lines = [f"- {f.content}" for f in facts[-20:]]
    return "\n".join(lines)


def _format_conversation(conversation: list[tuple[str, str]]) -> str:
    """格式化对话为文本。"""
    lines = []
    for i, (user, aurora) in enumerate(conversation[-20:], 1):
        lines.append(f"Turn {i}:")
        lines.append(f"  User: {user}")
        lines.append(f"  Aurora: {aurora}")
        lines.append("")
    return "\n".join(lines)


def _format_profile(state: RelationalState) -> str:
    """格式化当前 Profile。"""
    lines = [
        f"intimacy_level: {state.intimacy_level}",
        f"current_vibe: {state.current_vibe}",
    ]
    if state.interaction_rules:
        lines.append("interaction_rules:")
        for rule in state.interaction_rules:
            lines.append(f"  - {rule}")
    return "\n".join(lines)


def _parse_patch(raw: str) -> DistillationPatch:
    """解析 LLM 响应为蒸馏补丁。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return DistillationPatch()

    return DistillationPatch(
        intimacy_delta=_clamp_int(data.get("intimacy_delta")),
        new_vibe=data.get("new_vibe"),
        new_rules=data.get("new_rules"),
        facts=data.get("facts"),
        tensions=data.get("tensions"),
    )


def _clamp_int(value: Any) -> int | None:
    """将值 clamp 到 [-1, 0, 1]。"""
    if value is None:
        return None
    try:
        v = int(value)
        return max(-1, min(1, v))
    except (TypeError, ValueError):
        return None


def apply_distillation(
    patch: DistillationPatch,
    relational_state: RelationalState,
    tension_queue: TensionQueue,
    now_ts: float,
) -> None:
    """应用蒸馏补丁。

    Args:
        patch: 蒸馏补丁。
        relational_state: 关系状态。
        tension_queue: 张力队列。
        now_ts: 当前时间戳。
    """
    if patch.intimacy_delta is not None:
        relational_state.apply_patch(intimacy_delta=patch.intimacy_delta)
    if patch.new_vibe is not None:
        relational_state.apply_patch(vibe=patch.new_vibe)
    if patch.new_rules:
        relational_state.apply_patch(new_rules=patch.new_rules)

    if patch.tensions:
        for t in patch.tensions:
            tension_queue.push(
                topic=t.get("topic", ""),
                urgency=t.get("urgency", 0.5),
                halflife_hours=t.get("halflife_hours", 24.0),
                prompt=t.get("prompt", ""),
                now_ts=now_ts,
            )
