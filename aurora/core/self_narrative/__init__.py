"""
AURORA Self-Narrative Module
============================

公共入口保留在本模块，数据模型和跟踪器拆分到独立文件。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from aurora.core.components.metric import LowRankMetric
from aurora.core.models.plot import Plot
from aurora.core.models.theme import Theme
from aurora.core.personality import PersonalityProfile
from aurora.core.self_narrative.models import (
    CapabilityBelief,
    DarkMatterEntry,
    IdentityChange,
    RelationshipBelief,
    SelfNarrative,
    SubconsciousState,
    TraitBelief,
)
from aurora.core.self_narrative.tracking import IdentityEvolutionTracker, IdentityTracker
from aurora.utils.math_utils import cosine_sim

__all__ = [
    "CapabilityBelief",
    "DarkMatterEntry",
    "IdentityChange",
    "IdentityEvolutionTracker",
    "IdentityTracker",
    "RelationshipBelief",
    "SelfNarrative",
    "SelfNarrativeEngine",
    "SubconsciousState",
    "TraitBelief",
]


class SelfNarrativeEngine:
    """维护并演化自我叙事的引擎。"""

    def __init__(
        self,
        metric: LowRankMetric,
        *,
        profile: PersonalityProfile,
        narrative: Optional[SelfNarrative] = None,
        embedder: Optional[object] = None,
        seed: int = 0,
    ):
        self.metric = metric
        self.profile = profile
        self._embedder = embedder
        self.rng = np.random.default_rng(seed)
        self.narrative = narrative or SelfNarrative.from_profile(profile)
        self.momentum = 0.9
        self._anchor_embeddings: Dict[str, np.ndarray] = {}

        if self._embedder is not None:
            self._build_anchor_embeddings()

    def bootstrap_seed_plot_ids(self, plot_ids: Sequence[str]) -> None:
        self.narrative.seed_plot_ids = list(plot_ids)

    def observe_trait(self, trait_name: str, positive: bool, magnitude: float = 1.0) -> None:
        trait = self.narrative.get_trait(trait_name)
        if trait is None:
            return
        trait.observe(positive=positive, magnitude=magnitude)
        self.narrative.log_evolution(
            "trait_observation",
            f"{trait_name}:{'positive' if positive else 'negative'}:{magnitude:.2f}",
        )

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

        if any(token in action_text for token in ("why", "how", "结构", "因果", "机制")):
            self.observe_trait("curiosity", positive=True, magnitude=0.2)
        self.observe_trait("rigor", positive=success, magnitude=0.1)
        if plot.relational is not None:
            self.observe_trait(
                "warmth",
                positive=plot.relational.relationship_quality_delta >= 0,
                magnitude=0.1,
            )

    def refresh_subconscious_summary(self, subconscious_state: SubconsciousState) -> None:
        self.narrative.subconscious_summary = subconscious_state.summary()

    def intuition_keywords_for_vectors(self, vectors: Sequence[np.ndarray], max_items: int = 2) -> List[str]:
        if not vectors:
            return []
        if self._embedder is None:
            return [anchor.keywords[0] for anchor in self.profile.intuition_anchors[:max_items]]

        self._build_anchor_embeddings()

        selected: List[str] = []
        seen: set[str] = set()
        for vector in vectors:
            best_anchor_id: Optional[str] = None
            best_score = -1.0
            for anchor in self.profile.intuition_anchors:
                anchor_vector = self._anchor_embeddings.get(anchor.anchor_id)
                if anchor_vector is None:
                    continue
                score = float(cosine_sim(vector, anchor_vector))
                if score > best_score:
                    best_score = score
                    best_anchor_id = anchor.anchor_id
            if best_anchor_id is None:
                continue
            anchor = next(
                item for item in self.profile.intuition_anchors if item.anchor_id == best_anchor_id
            )
            keyword = anchor.keywords[0] if anchor.keywords else anchor.text
            if keyword in seen:
                continue
            seen.add(keyword)
            selected.append(keyword)
            if len(selected) >= max_items:
                break
        return selected

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

    def add_tension(self, tension: str) -> None:
        if tension not in self.narrative.unresolved_tensions:
            self.narrative.unresolved_tensions.append(tension)
            self.narrative.log_evolution("tension_added", tension)

    def resolve_tension(self, tension: str) -> None:
        if tension in self.narrative.unresolved_tensions:
            self.narrative.unresolved_tensions.remove(tension)
            self.narrative.log_evolution("tension_resolved", tension)

    def _build_anchor_embeddings(self) -> None:
        if self._embedder is None:
            return
        if self._anchor_embeddings:
            return
        for anchor in self.profile.intuition_anchors:
            self._anchor_embeddings[anchor.anchor_id] = self._embedder.embed(anchor.text)

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
        parts: List[str] = []
        if self.profile.identity_narrative:
            parts.append(self.profile.identity_narrative)

        if self.narrative.trait_beliefs:
            trait_bits = [
                trait.to_summary()
                for trait in sorted(
                    self.narrative.trait_beliefs.values(),
                    key=lambda item: item.probability(),
                    reverse=True,
                )[:3]
            ]
            if trait_bits:
                parts.append("我的人格底色偏向" + "、".join(trait_bits))

        total_interactions = sum(
            relationship.interaction_count
            for relationship in self.narrative.relationships.values()
        )
        if total_interactions > 0:
            parts.append(f"经过{total_interactions}次互动，我的关系经验正在缓慢塑形。")

        return "。".join(parts)
