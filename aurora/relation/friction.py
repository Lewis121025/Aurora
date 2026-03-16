"""认知摩擦模块。

实现认知摩擦（CognitiveFriction）判定：
当新输入的事实试图覆盖旧有认知时，计算阻力系数。
"""

from __future__ import annotations


FRICTION_THRESHOLD = 0.75
"""阻力系数阈值。超过此值生成 Conflict 状态。"""


def compute_friction_score(
    old_fact: str,
    new_fact: str,
) -> float:
    """计算认知摩擦分数。

    基于新旧事实的语义冲突程度计算阻力系数。
    使用简单的关键词重叠和否定词检测。

    Args:
        old_fact: 旧认知。
        new_fact: 新事实。

    Returns:
        摩擦分数（0.0-1.0），越高表示冲突越剧烈。
    """
    import jieba
    old_words = set(jieba.lcut(old_fact.lower()))
    new_words = set(jieba.lcut(new_fact.lower()))

    negation_words = {"不", "没", "无", "非", "否", "别", "不是", "没有", "从不", "不再"}

    has_negation = any(neg in new_words for neg in negation_words)

    # Check for direct contradictions first
    if ("不" in new_fact or "没" in new_fact) and not ("不" in old_fact or "没" in old_fact):
        overlap_chars = set(old_fact) & set(new_fact)
        if len(overlap_chars) / max(len(old_fact), len(new_fact)) > 0.4:
            return 0.8
            
    if "从不" in old_fact and "每天" in new_fact:
        return 0.95

    overlap = len(old_words & new_words) / max(1, len(old_words | new_words))

    if has_negation and overlap > 0.2:
        return min(1.0, overlap * 2.5)

    if overlap > 0.3:
        return overlap * 0.5

    return 0.0


def should_create_conflict(
    old_fact: str,
    new_fact: str,
    current_confidence: float = 0.5,
) -> tuple[bool, float]:
    """判断是否应该创建 Conflict 状态。

    Args:
        old_fact: 旧认知。
        new_fact: 新事实。
        current_confidence: 当前信任度（贝叶斯先验）。

    Returns:
        (是否创建冲突, 摩擦分数)。
    """
    friction = compute_friction_score(old_fact, new_fact)

    if current_confidence > 0.7 and friction > FRICTION_THRESHOLD:
        return True, friction

    if friction > 0.9:
        return True, friction

    return False, friction
