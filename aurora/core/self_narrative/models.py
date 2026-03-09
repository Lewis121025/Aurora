"""
AURORA 自我叙事模型
========================

自我叙事相关的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts


@dataclass
class CapabilityBelief:
    """关于能力的概率性信念。"""

    name: str
    description: str
    a: float = 1.0
    b: float = 1.0
    positive_contexts: List[str] = field(default_factory=list)
    negative_contexts: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = field(default_factory=now_ts)

    def capability_probability(self) -> float:
        return self.a / (self.a + self.b)

    def sample(self, rng: np.random.Generator) -> bool:
        return rng.beta(self.a, self.b) > 0.5

    def update(self, success: bool, context: Optional[str] = None) -> None:
        if success:
            self.a += 1.0
            self.success_count += 1
            if context and context not in self.positive_contexts:
                self.positive_contexts.append(context)
        else:
            self.b += 1.0
            self.failure_count += 1
            if context and context not in self.negative_contexts:
                self.negative_contexts.append(context)
        self.last_updated = now_ts()

    def to_narrative(self) -> str:
        probability = self.capability_probability()
        if probability > 0.8:
            prefix = "我擅长"
        elif probability > 0.6:
            prefix = "我通常能够"
        elif probability > 0.4:
            prefix = "我有时能够"
        else:
            prefix = "我在...方面有困难"

        narrative = f"{prefix}{self.description}"
        if self.positive_contexts:
            narrative += f"，特别是在{', '.join(self.positive_contexts[:3])}的情况下"
        if self.negative_contexts and probability < 0.7:
            narrative += f"，但在{', '.join(self.negative_contexts[:2])}时可能遇到困难"
        return narrative


@dataclass
class RelationshipBelief:
    """关于关系的概率性信念。"""

    entity_id: str
    entity_type: str
    trust_a: float = 1.0
    trust_b: float = 1.0
    familiarity_a: float = 1.0
    familiarity_b: float = 1.0
    collaboration_a: float = 1.0
    collaboration_b: float = 1.0
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    preferences: Dict[str, float] = field(default_factory=dict)
    last_interaction: float = field(default_factory=now_ts)

    def trust(self) -> float:
        return self.trust_a / (self.trust_a + self.trust_b)

    def familiarity(self) -> float:
        return self.familiarity_a / (self.familiarity_a + self.familiarity_b)

    def collaboration_quality(self) -> float:
        return self.collaboration_a / (self.collaboration_a + self.collaboration_b)

    def update_interaction(self, positive: bool, collaboration_success: bool = True) -> None:
        self.interaction_count += 1
        self.familiarity_a += 0.5

        if positive:
            self.positive_interactions += 1
            self.trust_a += 1.0
        else:
            self.negative_interactions += 1
            self.trust_b += 0.5

        if collaboration_success:
            self.collaboration_a += 1.0
        else:
            self.collaboration_b += 1.0

        self.last_interaction = now_ts()

    def update_preference(self, preference_key: str, value: float) -> None:
        if preference_key in self.preferences:
            self.preferences[preference_key] = 0.7 * self.preferences[preference_key] + 0.3 * value
        else:
            self.preferences[preference_key] = value

    def to_narrative(self) -> str:
        parts: List[str] = []

        trust = self.trust()
        if trust > 0.8:
            parts.append("我们建立了很好的信任关系")
        elif trust > 0.6:
            parts.append("我们的关系比较稳定")
        elif trust > 0.4:
            parts.append("我们还在相互了解")
        else:
            parts.append("我们的互动有时会遇到挑战")

        if self.familiarity() > 0.7:
            parts.append("经过多次互动，我已经比较了解对方")

        if self.preferences:
            top_preferences = sorted(
                self.preferences.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:2]
            preference_text = [
                f"偏好{key}" if value > 0 else f"不太喜欢{key}"
                for key, value in top_preferences
            ]
            parts.append("我注意到对方" + "，".join(preference_text))

        return "。".join(parts)


@dataclass
class SelfNarrative:
    """演化中的自我叙事。"""

    identity_statement: str = "我是一个通过记忆和叙事来学习的AI助手"
    core_values: List[str] = field(default_factory=lambda: ["准确性", "帮助性", "持续学习"])
    identity_narrative: str = ""
    capability_narrative: str = ""
    capabilities: Dict[str, CapabilityBelief] = field(default_factory=dict)
    relationships: Dict[str, RelationshipBelief] = field(default_factory=dict)
    supporting_theme_ids: List[str] = field(default_factory=list)
    coherence_score: float = 1.0
    unresolved_tensions: List[str] = field(default_factory=list)
    evolution_log: List[Tuple[float, str, str]] = field(default_factory=list)
    last_updated: float = field(default_factory=now_ts)

    def to_full_narrative(self) -> str:
        sections = [f"## 我是谁\n{self.identity_statement}"]

        if self.identity_narrative:
            sections.append(self.identity_narrative)
        if self.core_values:
            sections.append(f"## 核心价值\n我重视: {', '.join(self.core_values)}")
        if self.capabilities:
            sections.append(
                "## 能力认知\n" + "。".join(capability.to_narrative() for capability in self.capabilities.values())
            )
        if self.relationships:
            sections.append(
                "## 关系\n" + "\n".join(
                    relationship.to_narrative()
                    for relationship in list(self.relationships.values())[:3]
                )
            )
        if self.unresolved_tensions:
            sections.append(
                "## 待解决的张力\n" + "\n".join(f"- {tension}" for tension in self.unresolved_tensions[:3])
            )

        return "\n\n".join(sections)

    def get_capability(self, name: str) -> CapabilityBelief:
        if name not in self.capabilities:
            self.capabilities[name] = CapabilityBelief(name=name, description=name)
        return self.capabilities[name]

    def get_relationship(self, entity_id: str, entity_type: str = "user") -> RelationshipBelief:
        if entity_id not in self.relationships:
            self.relationships[entity_id] = RelationshipBelief(
                entity_id=entity_id,
                entity_type=entity_type,
            )
        return self.relationships[entity_id]

    def log_evolution(self, change_type: str, description: str) -> None:
        self.evolution_log.append((now_ts(), change_type, description))
        self.last_updated = now_ts()
        if len(self.evolution_log) > 100:
            self.evolution_log = self.evolution_log[-100:]


@dataclass
class IdentityChange:
    """记录身份变化。"""

    ts: float
    dimension: str
    from_identity: str
    to_identity: str
    triggered_by_relationships: List[str]
    triggered_by_events: List[str]
    magnitude: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "dimension": self.dimension,
            "from_identity": self.from_identity,
            "to_identity": self.to_identity,
            "triggered_by_relationships": self.triggered_by_relationships,
            "triggered_by_events": self.triggered_by_events,
            "magnitude": self.magnitude,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityChange":
        return cls(
            ts=data["ts"],
            dimension=data["dimension"],
            from_identity=data["from_identity"],
            to_identity=data["to_identity"],
            triggered_by_relationships=data.get("triggered_by_relationships", []),
            triggered_by_events=data.get("triggered_by_events", []),
            magnitude=data.get("magnitude", 0.5),
        )
