"""Relation field helper。"""

from __future__ import annotations

from aurora.runtime.contracts import RelationField, clamp


def apply_relation_patch(
    field: RelationField,
    *,
    trust_delta: float = 0.0,
    distance_delta: float = 0.0,
    warmth_delta: float = 0.0,
    tension_delta: float = 0.0,
    repair_debt_delta: float = 0.0,
    compiled_at: float | None = None,
) -> None:
    """对关系场应用确定性增量更新。"""
    field.trust = clamp(field.trust + trust_delta)
    field.distance = clamp(field.distance + distance_delta)
    field.warmth = clamp(field.warmth + warmth_delta)
    field.tension = clamp(field.tension + tension_delta)
    field.repair_debt = clamp(field.repair_debt + repair_debt_delta)
    if compiled_at is not None:
        field.last_compiled_at = compiled_at


def add_rule(field: RelationField, rule: str) -> None:
    """追加交互规则。"""
    normalized = rule.strip()
    if normalized and normalized not in field.interaction_rules:
        field.interaction_rules.append(normalized)


def add_lexicon_items(field: RelationField, items: list[str]) -> None:
    """追加共享词汇。"""
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in field.shared_lexicon:
            field.shared_lexicon.append(normalized)


def to_prompt_segment(field: RelationField) -> str:
    """将关系场投影为运行时 prompt 片段。"""
    rules = ", ".join(field.interaction_rules) if field.interaction_rules else "none"
    lexicon = ", ".join(field.shared_lexicon) if field.shared_lexicon else "none"
    return (
        "[RELATION_FIELD]\n"
        f"trust={field.trust:.2f}\n"
        f"distance={field.distance:.2f}\n"
        f"warmth={field.warmth:.2f}\n"
        f"tension={field.tension:.2f}\n"
        f"repair_debt={field.repair_debt:.2f}\n"
        f"interaction_rules={rules}\n"
        f"shared_lexicon={lexicon}"
    )
