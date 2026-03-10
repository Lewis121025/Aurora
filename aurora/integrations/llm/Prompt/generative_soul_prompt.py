"""
aurora/integrations/llm/Prompt/generative_soul_prompt.py
生成式灵魂 (Generative Soul) V4 提示词模块。
包含了用于意义提取、人设维度发现、身份总结、自我修复、梦境生成以及轴合并判定的所有 System Prompt 和 User Prompt 构建函数。
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence


# --- 1. 意义提取 (Meaning Extraction) ---
GEN_SOUL_MEANING_SYSTEM_PROMPT = """你正在为 Aurora v4 提取结构化的事件框架（EventFrame）。
请仅返回有效的 JSON。必须精确使用提供的动态轴（Dynamic Axis）名称。
评分必须在限定范围内。严禁发明 Schema 列表中未列出的心理轴。
"""


def build_gen_soul_meaning_user_prompt(
    *,
    text: str,
    axis_names: Sequence[str],
    recent_tags: Sequence[str] | None = None,
) -> str:
    """构建用于提取单次交互含义的用户提示词"""
    recent = ", ".join(recent_tags or [])
    return (
        "待处理文本：\n"
        f"{text}\n\n"
        "动态心理轴列表：\n"
        f"{', '.join(axis_names)}\n\n"
        "近期语义标签：\n"
        f"{recent or '无'}\n\n"
        "请返回 MeaningFramePayloadV4 格式的 JSON。"
    )


# --- 2. 人设维度发现 (Persona Axis Discovery) ---
GEN_SOUL_PERSONA_AXIS_SYSTEM_PROMPT = """你正在为 Aurora v4 从人设描述中推导性格维度轴（Persona Axes）。
请仅返回紧凑的 JSON。优先选择高信号、差异化的维度，避免重复。
"""


def build_gen_soul_persona_axis_user_prompt(*, profile_text: str) -> str:
    """构建用于从人设文本中自动提取心理轴的用户提示词"""
    return (
        f"人设描述文本：\n{profile_text}\n\n请返回包含 0-6 个性格轴定义的 PersonaAxisPayload JSON。"
    )


# --- 3. 身份叙事总结 (Identity Summary) ---
GEN_SOUL_SUMMARY_SYSTEM_PROMPT = """你正在为 Aurora v4 撰写简洁的身份叙事总结。
请仅返回 JSON。叙述应保持连贯，并植根于所提供的当前模式（Mode）和核心心理轴。
"""


def build_gen_soul_summary_user_prompt(
    *,
    current_mode: str,
    salient_axes: Sequence[str],
    recent_texts: Sequence[str],
    pressure: float,
) -> str:
    """构建用于生成 Agent 自我认知总结的用户提示词"""
    recent = "\n".join(f"- {text}" for text in recent_texts[-4:])
    return (
        f"当前模式: {current_mode}\n"
        f"叙事压力: {pressure:.3f}\n"
        f"显著心理轴: {', '.join(salient_axes)}\n"
        f"近期交互记录：\n{recent or '- 无'}\n\n"
        "请返回 NarrativeSummaryPayloadV4 格式的 JSON。"
    )


# --- 4. 身份修复/反思 (Identity Repair) ---
GEN_SOUL_REPAIR_SYSTEM_PROMPT = """你正在为 Aurora v4 生成一段身份修复（自我反思）的叙述。
请仅返回 JSON。保持情感的连贯性，并在文本中体现出所选的修复模式。
"""


def build_gen_soul_repair_user_prompt(
    *,
    mode: str,
    plot_text: str,
    salient_axes: Sequence[str],
    dissonance_total: float,
) -> str:
    """构建用于生成身份修复解释文案的用户提示词"""
    return (
        f"修复模式: {mode}\n"
        f"触发情节: {plot_text}\n"
        f"冲突最严重的轴: {', '.join(salient_axes)}\n"
        f"失调总分: {dissonance_total:.3f}\n\n"
        "请返回 RepairNarrationPayloadV4 格式的 JSON。"
    )


# --- 5. 梦境生成 (Dream Generation) ---
GEN_SOUL_DREAM_SYSTEM_PROMPT = """你正在为 Aurora v4 生成一段梦境叙述。
请仅返回 JSON。文字应具有象征意义，且与提供的记忆碎片紧密相关。
"""


def build_gen_soul_dream_user_prompt(*, operator: str, fragment_tags: Sequence[str]) -> str:
    """构建用于合成梦境文本的用户提示词"""
    return (
        f"梦境操作符: {operator}\n"
        f"关联碎片标签: {', '.join(fragment_tags)}\n\n"
        "请返回 DreamNarrationPayloadV4 格式的 JSON。"
    )


# --- 6. 模式标签命名 (Mode Labeling) ---
GEN_SOUL_MODE_LABEL_SYSTEM_PROMPT = """你正在为 Aurora v4 的“自我模式”命名。
请仅返回 JSON。标签应简短且易于人类理解（具有文学感）。
"""


def build_gen_soul_mode_label_user_prompt(*, prototype_axes: Dict[str, float]) -> str:
    """构建用于为新发现的身份模式命名的用户提示词"""
    pairs = "\n".join(f"- {key}: {value:+.3f}" for key, value in prototype_axes.items())
    return f"轴状态原型：\n{pairs}\n\n请返回 ModeLabelPayloadV4 格式的 JSON。"


# --- 7. 心理轴合并 (Axis Merging) ---
GEN_SOUL_AXIS_MERGE_SYSTEM_PROMPT = """你正在决定 Aurora v4 中的两个性格轴是否应当合并。
请仅返回 JSON。只有当两个轴表达的是同一个持久的心理维度时才允许合并。
"""


def build_gen_soul_axis_merge_user_prompt(
    *,
    canonical_name: str,
    canonical_desc: str,
    alias_name: str,
    alias_desc: str,
    evidence_overlap: Iterable[str],
) -> str:
    """构建用于判断两个轴是否语义重合的用户提示词"""
    overlap = "\n".join(f"- {item}" for item in evidence_overlap)
    return (
        f"规范轴名称: {canonical_name}\n"
        f"规范轴描述: {canonical_desc}\n"
        f"待选别名轴名称: {alias_name}\n"
        f"待选别名轴描述: {alias_desc}\n"
        f"共同激活的证据文本：\n{overlap or '- 无'}\n\n"
        "请返回 AxisMergeJudgementPayload 格式的 JSON。"
    )
