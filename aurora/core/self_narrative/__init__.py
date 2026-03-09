"""
AURORA Self-Narrative Module
============================

公共入口保留在本模块，数据模型和跟踪器拆分到独立文件。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from aurora.core.components.metric import LowRankMetric
from aurora.core.models.plot import Plot
from aurora.core.models.theme import Theme
from aurora.core.self_narrative.models import (
    CapabilityBelief,
    IdentityChange,
    RelationshipBelief,
    SelfNarrative,
)
from aurora.core.self_narrative.tracking import IdentityEvolutionTracker, IdentityTracker

__all__ = [
    "CapabilityBelief",
    "IdentityChange",
    "IdentityEvolutionTracker",
    "IdentityTracker",
    "RelationshipBelief",
    "SelfNarrative",
    "SelfNarrativeEngine",
]


class SelfNarrativeEngine:
    """维护并演化自我叙事的引擎。"""

    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)
        self.narrative = SelfNarrative()
        self.momentum = 0.9

    def update_from_themes(self, themes: List[Theme]) -> bool:
        update_handlers: Dict[str, Callable[[Theme], bool]] = {
            "capability": self._update_capability_from_theme,
            "limitation": self._update_limitation_from_theme,
            "pattern": self._update_pattern_from_theme,
            "preference": self._update_preference_from_theme,
        }

        significant_change = False
        for theme in themes:
            if theme.confidence() < 0.5:
                continue
            handler = update_handlers.get(theme.theme_type)
            if handler is None:
                continue
            significant_change = handler(theme) or significant_change

        self.narrative.supporting_theme_ids = [
            theme.id for theme in themes if theme.confidence() > 0.6
        ]

        if significant_change:
            self._regenerate_narratives()
            self.narrative.log_evolution("theme_update", f"Updated from {len(themes)} themes")

        return significant_change

    def update_from_interaction(
        self,
        plot: Plot,
        success: bool,
        entity_id: Optional[str] = None,
    ) -> None:
        if entity_id:
            self.narrative.get_relationship(entity_id).update_interaction(positive=success)

        action_text = (getattr(plot, "action", "") or getattr(plot, "text", "")).lower()
        for keyword, capability_name in (
            (("代码", "编程", "code"), "编程"),
            (("分析", "理解", "explain"), "分析解释"),
            (("计划", "规划", "plan"), "规划"),
        ):
            if any(token in action_text for token in keyword):
                self.narrative.get_capability(capability_name).update(success)

    def add_tension(self, tension: str) -> None:
        if tension not in self.narrative.unresolved_tensions:
            self.narrative.unresolved_tensions.append(tension)
            self.narrative.log_evolution("tension_added", tension)

    def resolve_tension(self, tension: str) -> None:
        if tension in self.narrative.unresolved_tensions:
            self.narrative.unresolved_tensions.remove(tension)
            self.narrative.log_evolution("tension_resolved", tension)

    def check_coherence(self) -> float:
        issues = 0.0
        total_checks = 0

        for capability_name, capability in self.narrative.capabilities.items():
            total_checks += 1
            if capability.capability_probability() <= 0.7:
                continue
            if any(
                "limitation" in theme_id.lower() and capability_name.lower() in theme_id.lower()
                for theme_id in self.narrative.supporting_theme_ids
            ):
                issues += 1.0

        for relationship in self.narrative.relationships.values():
            total_checks += 1
            if relationship.interaction_count <= 5:
                continue
            positive_ratio = relationship.positive_interactions / relationship.interaction_count
            if abs(positive_ratio - relationship.trust()) > 0.3:
                issues += 0.5

        if len(self.narrative.unresolved_tensions) > 5:
            issues += 0.5

        self.narrative.coherence_score = 1.0 - (issues / max(total_checks, 1))
        return self.narrative.coherence_score

    def _update_capability_from_theme(self, theme: Theme) -> bool:
        capability = self.narrative.get_capability(theme.name)
        confidence = theme.confidence()
        previous_probability = capability.capability_probability()

        if confidence > previous_probability:
            capability.a += (confidence - previous_probability) * (1 - self.momentum)
        else:
            capability.b += (previous_probability - confidence) * (1 - self.momentum)

        return abs(confidence - previous_probability) > 0.1

    def _update_limitation_from_theme(self, theme: Theme) -> bool:
        capability = self.narrative.get_capability(theme.name)
        previous_probability = capability.capability_probability()
        confidence = theme.confidence()
        capability.b += confidence * (1 - self.momentum)
        return confidence > 0.5 and previous_probability > 0.5

    def _update_pattern_from_theme(self, theme: Theme) -> bool:
        del theme
        return False

    def _update_preference_from_theme(self, theme: Theme) -> bool:
        del theme
        return False

    def _regenerate_narratives(self) -> None:
        self.narrative.capability_narrative = self._build_capability_narrative()
        self.narrative.identity_narrative = self._build_identity_narrative()

    def _build_capability_narrative(self) -> str:
        if not self.narrative.capabilities:
            return ""

        strong_capabilities = [
            capability.name
            for capability in self.narrative.capabilities.values()
            if capability.capability_probability() > 0.7
        ]
        weak_capabilities = [
            capability.name
            for capability in self.narrative.capabilities.values()
            if capability.capability_probability() < 0.3
        ]

        parts: List[str] = []
        if strong_capabilities:
            parts.append("我擅长" + "、".join(strong_capabilities[:3]))
        if weak_capabilities:
            parts.append("我在" + "、".join(weak_capabilities[:2]) + "方面还在学习")
        return "。".join(parts)

    def _build_identity_narrative(self) -> str:
        if not (self.narrative.capabilities or self.narrative.relationships):
            return ""

        parts: List[str] = []
        total_interactions = sum(
            relationship.interaction_count
            for relationship in self.narrative.relationships.values()
        )
        if total_interactions > 10:
            parts.append(f"通过{total_interactions}次互动，我不断学习和成长")
        if self.narrative.core_values:
            parts.append(f"我坚持{', '.join(self.narrative.core_values[:3])}的原则")
        return "。".join(parts)
