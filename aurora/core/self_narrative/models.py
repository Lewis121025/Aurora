"""
AURORA 自我叙事模型
========================

自我叙事、人格先验与潜意识状态的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.core.personality import PersonalityProfile
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

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "a": self.a,
            "b": self.b,
            "positive_contexts": self.positive_contexts,
            "negative_contexts": self.negative_contexts,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "CapabilityBelief":
        return cls(
            name=data["name"],
            description=data["description"],
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 1.0)),
            positive_contexts=list(data.get("positive_contexts", [])),
            negative_contexts=list(data.get("negative_contexts", [])),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            last_updated=float(data.get("last_updated", now_ts())),
        )


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

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "trust_a": self.trust_a,
            "trust_b": self.trust_b,
            "familiarity_a": self.familiarity_a,
            "familiarity_b": self.familiarity_b,
            "collaboration_a": self.collaboration_a,
            "collaboration_b": self.collaboration_b,
            "interaction_count": self.interaction_count,
            "positive_interactions": self.positive_interactions,
            "negative_interactions": self.negative_interactions,
            "preferences": self.preferences,
            "last_interaction": self.last_interaction,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "RelationshipBelief":
        return cls(
            entity_id=data["entity_id"],
            entity_type=data.get("entity_type", "user"),
            trust_a=float(data.get("trust_a", 1.0)),
            trust_b=float(data.get("trust_b", 1.0)),
            familiarity_a=float(data.get("familiarity_a", 1.0)),
            familiarity_b=float(data.get("familiarity_b", 1.0)),
            collaboration_a=float(data.get("collaboration_a", 1.0)),
            collaboration_b=float(data.get("collaboration_b", 1.0)),
            interaction_count=int(data.get("interaction_count", 0)),
            positive_interactions=int(data.get("positive_interactions", 0)),
            negative_interactions=int(data.get("negative_interactions", 0)),
            preferences={str(k): float(v) for k, v in (data.get("preferences", {}) or {}).items()},
            last_interaction=float(data.get("last_interaction", now_ts())),
        )


@dataclass
class TraitBelief:
    """人格特质的贝叶斯先验与缓慢更新。"""

    name: str
    description: str
    a: float
    b: float
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    last_updated: float = field(default_factory=now_ts)

    def probability(self) -> float:
        return self.a / (self.a + self.b)

    def observe(self, positive: bool, magnitude: float = 1.0) -> None:
        if positive:
            self.a += magnitude
            self.evidence_for += magnitude
        else:
            self.b += magnitude
            self.evidence_against += magnitude
        self.last_updated = now_ts()

    def to_summary(self) -> str:
        return f"{self.description}（{self.probability():.2f}）"

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "a": self.a,
            "b": self.b,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "TraitBelief":
        return cls(
            name=data["name"],
            description=data["description"],
            a=float(data["a"]),
            b=float(data["b"]),
            evidence_for=float(data.get("evidence_for", 0.0)),
            evidence_against=float(data.get("evidence_against", 0.0)),
            last_updated=float(data.get("last_updated", now_ts())),
        )

    @classmethod
    def from_profile_trait(cls, name: str, description: str, alpha: float, beta: float) -> "TraitBelief":
        return cls(
            name=name,
            description=description,
            a=alpha,
            b=beta,
        )


@dataclass
class DarkMatterEntry:
    id: str
    ts: float
    embedding: np.ndarray
    knowledge_type: str = "unknown"
    affect_hint: str = ""
    relational_hint: str = ""
    source_plot_id: Optional[str] = None

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "embedding": self.embedding.tolist(),
            "knowledge_type": self.knowledge_type,
            "affect_hint": self.affect_hint,
            "relational_hint": self.relational_hint,
            "source_plot_id": self.source_plot_id,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "DarkMatterEntry":
        return cls(
            id=str(data["id"]),
            ts=float(data["ts"]),
            embedding=np.asarray(data["embedding"], dtype=np.float32),
            knowledge_type=str(data.get("knowledge_type", "unknown")),
            affect_hint=str(data.get("affect_hint", "")),
            relational_hint=str(data.get("relational_hint", "")),
            source_plot_id=data.get("source_plot_id"),
        )


@dataclass
class SubconsciousState:
    dark_matter_pool: List[DarkMatterEntry] = field(default_factory=list)
    repressed_plot_ids: List[str] = field(default_factory=list)
    last_intuition: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=now_ts)

    def add_dark_matter(self, entry: DarkMatterEntry, max_entries: int) -> None:
        self.dark_matter_pool.append(entry)
        if len(self.dark_matter_pool) > max_entries:
            self.dark_matter_pool = self.dark_matter_pool[-max_entries:]
        self.last_updated = now_ts()

    def mark_repressed(self, plot_id: str) -> None:
        if plot_id not in self.repressed_plot_ids:
            self.repressed_plot_ids.append(plot_id)
        self.last_updated = now_ts()

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dark_matter_pool": [entry.to_state_dict() for entry in self.dark_matter_pool],
            "repressed_plot_ids": self.repressed_plot_ids,
            "last_intuition": self.last_intuition,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "SubconsciousState":
        return cls(
            dark_matter_pool=[DarkMatterEntry.from_state_dict(item) for item in data.get("dark_matter_pool", [])],
            repressed_plot_ids=list(data.get("repressed_plot_ids", [])),
            last_intuition=list(data.get("last_intuition", [])),
            last_updated=float(data.get("last_updated", now_ts())),
        )

    def summary(self) -> List[str]:
        fragments: List[str] = []
        if self.dark_matter_pool:
            fragments.append(f"潜意识残渣 {len(self.dark_matter_pool)} 条")
        if self.repressed_plot_ids:
            fragments.append(f"压抑记忆 {len(self.repressed_plot_ids)} 条")
        if self.last_intuition:
            fragments.append("最近直觉：" + "，".join(self.last_intuition))
        return fragments


@dataclass
class SelfNarrative:
    """演化中的自我叙事。"""

    profile_id: str = "aurora-v2-native"
    identity_statement: str = "我是一个通过记忆和叙事来学习的AI助手"
    core_values: List[str] = field(default_factory=lambda: ["准确性", "帮助性", "持续学习"])
    identity_narrative: str = ""
    seed_narrative: str = ""
    capability_narrative: str = ""
    capabilities: Dict[str, CapabilityBelief] = field(default_factory=dict)
    relationships: Dict[str, RelationshipBelief] = field(default_factory=dict)
    trait_beliefs: Dict[str, TraitBelief] = field(default_factory=dict)
    seed_plot_ids: List[str] = field(default_factory=list)
    subconscious_summary: List[str] = field(default_factory=list)
    supporting_theme_ids: List[str] = field(default_factory=list)
    coherence_score: float = 1.0
    unresolved_tensions: List[str] = field(default_factory=list)
    evolution_log: List[Tuple[float, str, str]] = field(default_factory=list)
    last_updated: float = field(default_factory=now_ts)

    @classmethod
    def from_profile(cls, profile: PersonalityProfile) -> "SelfNarrative":
        narrative = cls(
            profile_id=profile.profile_id,
            identity_statement=profile.identity_statement,
            identity_narrative=profile.identity_narrative,
            seed_narrative=profile.seed_narrative,
            core_values=list(profile.core_values),
        )
        for trait in profile.trait_priors:
            narrative.trait_beliefs[trait.name] = TraitBelief.from_profile_trait(
                name=trait.name,
                description=trait.description,
                alpha=trait.alpha,
                beta=trait.beta,
            )
        return narrative

    def to_full_narrative(self) -> str:
        sections = [f"## 我是谁\n{self.identity_statement}"]

        if self.identity_narrative:
            sections.append(self.identity_narrative)
        if self.seed_narrative:
            sections.append(f"## 原生叙事\n{self.seed_narrative}")
        if self.core_values:
            sections.append(f"## 核心价值\n我重视: {', '.join(self.core_values)}")
        if self.trait_beliefs:
            sections.append(
                "## 性格先验\n" + "。".join(trait.to_summary() for trait in self.trait_beliefs.values())
            )
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
        if self.subconscious_summary:
            sections.append("## 潜意识回声\n" + "\n".join(f"- {item}" for item in self.subconscious_summary[:3]))
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

    def get_trait(self, name: str) -> Optional[TraitBelief]:
        return self.trait_beliefs.get(name)

    def log_evolution(self, change_type: str, description: str) -> None:
        self.evolution_log.append((now_ts(), change_type, description))
        self.last_updated = now_ts()
        if len(self.evolution_log) > 100:
            self.evolution_log = self.evolution_log[-100:]

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "identity_statement": self.identity_statement,
            "identity_narrative": self.identity_narrative,
            "seed_narrative": self.seed_narrative,
            "capability_narrative": self.capability_narrative,
            "core_values": self.core_values,
            "capabilities": {key: value.to_state_dict() for key, value in self.capabilities.items()},
            "relationships": {key: value.to_state_dict() for key, value in self.relationships.items()},
            "trait_beliefs": {key: value.to_state_dict() for key, value in self.trait_beliefs.items()},
            "seed_plot_ids": self.seed_plot_ids,
            "subconscious_summary": self.subconscious_summary,
            "supporting_theme_ids": self.supporting_theme_ids,
            "coherence_score": self.coherence_score,
            "unresolved_tensions": self.unresolved_tensions,
            "evolution_log": self.evolution_log,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "SelfNarrative":
        return cls(
            profile_id=data.get("profile_id", "aurora-v2-native"),
            identity_statement=data.get("identity_statement", "我是一个通过记忆和叙事来学习的AI助手"),
            identity_narrative=data.get("identity_narrative", ""),
            seed_narrative=data.get("seed_narrative", ""),
            capability_narrative=data.get("capability_narrative", ""),
            core_values=list(data.get("core_values", [])),
            capabilities={
                key: CapabilityBelief.from_state_dict(value)
                for key, value in (data.get("capabilities", {}) or {}).items()
            },
            relationships={
                key: RelationshipBelief.from_state_dict(value)
                for key, value in (data.get("relationships", {}) or {}).items()
            },
            trait_beliefs={
                key: TraitBelief.from_state_dict(value)
                for key, value in (data.get("trait_beliefs", {}) or {}).items()
            },
            seed_plot_ids=list(data.get("seed_plot_ids", [])),
            subconscious_summary=list(data.get("subconscious_summary", [])),
            supporting_theme_ids=list(data.get("supporting_theme_ids", [])),
            coherence_score=float(data.get("coherence_score", 1.0)),
            unresolved_tensions=list(data.get("unresolved_tensions", [])),
            evolution_log=[tuple(item) for item in data.get("evolution_log", [])],
            last_updated=float(data.get("last_updated", now_ts())),
        )


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
