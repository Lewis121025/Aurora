"""
AURORA Self-Narrative Module
============================

Emergent self-narrative from themes and experiences.
The agent's sense of "who I am" emerges from accumulated stories.

Key components:
- SelfNarrative: Dynamic self-model
- IdentityTracker: Track identity evolution
- CapabilityModel: Probabilistic capability beliefs
- RelationshipModel: Per-entity relationship narratives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import math
import numpy as np

from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.components.metric import LowRankMetric
from aurora.utils.math_utils import l2_normalize, cosine_sim, sigmoid
from aurora.utils.time_utils import now_ts


# -----------------------------------------------------------------------------
# Core Data Structures
# -----------------------------------------------------------------------------

@dataclass
class CapabilityBelief:
    """Belief about a capability (learned, not hard-coded)"""
    name: str
    description: str
    
    # Beta posterior: P(capable) = a / (a + b)
    a: float = 1.0
    b: float = 1.0
    
    # Context conditions (learned)
    positive_contexts: List[str] = field(default_factory=list)  # Contexts where capable
    negative_contexts: List[str] = field(default_factory=list)  # Contexts where not capable
    
    # Evidence
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = field(default_factory=now_ts)
    
    def capability_probability(self) -> float:
        """P(capable)"""
        return self.a / (self.a + self.b)
    
    def sample(self, rng: np.random.Generator) -> bool:
        """Thompson sampling for capability"""
        return rng.beta(self.a, self.b) > 0.5
    
    def update(self, success: bool, context: Optional[str] = None) -> None:
        """Update belief from outcome"""
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
        """Convert to natural language"""
        prob = self.capability_probability()
        
        if prob > 0.8:
            confidence = "我擅长"
        elif prob > 0.6:
            confidence = "我通常能够"
        elif prob > 0.4:
            confidence = "我有时能够"
        else:
            confidence = "我在...方面有困难"
        
        narrative = f"{confidence}{self.description}"
        
        if self.positive_contexts:
            narrative += f"，特别是在{', '.join(self.positive_contexts[:3])}的情况下"
        
        if self.negative_contexts and prob < 0.7:
            narrative += f"，但在{', '.join(self.negative_contexts[:2])}时可能遇到困难"
        
        return narrative


@dataclass
class RelationshipBelief:
    """Belief about relationship with an entity"""
    entity_id: str
    entity_type: str  # "user", "system", "concept"
    
    # Relationship dimensions (all Beta posteriors)
    trust_a: float = 1.0
    trust_b: float = 1.0
    
    familiarity_a: float = 1.0
    familiarity_b: float = 1.0
    
    collaboration_a: float = 1.0
    collaboration_b: float = 1.0
    
    # Interaction history summary
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    
    # Learned preferences
    preferences: Dict[str, float] = field(default_factory=dict)
    
    last_interaction: float = field(default_factory=now_ts)
    
    def trust(self) -> float:
        return self.trust_a / (self.trust_a + self.trust_b)
    
    def familiarity(self) -> float:
        return self.familiarity_a / (self.familiarity_a + self.familiarity_b)
    
    def collaboration_quality(self) -> float:
        return self.collaboration_a / (self.collaboration_a + self.collaboration_b)
    
    def update_interaction(self, positive: bool, collaboration_success: bool = True) -> None:
        """Update from interaction outcome"""
        self.interaction_count += 1
        self.familiarity_a += 0.5  # Any interaction increases familiarity
        
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
        """Update learned preference"""
        if preference_key in self.preferences:
            # Exponential moving average
            self.preferences[preference_key] = 0.7 * self.preferences[preference_key] + 0.3 * value
        else:
            self.preferences[preference_key] = value
    
    def to_narrative(self) -> str:
        """Convert to natural language"""
        parts = []
        
        # Trust
        trust = self.trust()
        if trust > 0.8:
            parts.append("我们建立了很好的信任关系")
        elif trust > 0.6:
            parts.append("我们的关系比较稳定")
        elif trust > 0.4:
            parts.append("我们还在相互了解")
        else:
            parts.append("我们的互动有时会遇到挑战")
        
        # Familiarity
        familiarity = self.familiarity()
        if familiarity > 0.7:
            parts.append("经过多次互动，我已经比较了解对方")
        
        # Preferences
        if self.preferences:
            top_prefs = sorted(self.preferences.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
            pref_strs = [f"偏好{k}" if v > 0 else f"不太喜欢{k}" for k, v in top_prefs]
            parts.append("我注意到对方" + "，".join(pref_strs))
        
        return "。".join(parts)


@dataclass
class SelfNarrative:
    """
    The agent's evolving self-narrative.
    
    This is not a fixed prompt - it emerges from experiences and evolves over time.
    """
    # Core identity (relatively stable)
    identity_statement: str = "我是一个通过记忆和叙事来学习的AI助手"
    core_values: List[str] = field(default_factory=lambda: [
        "准确性",
        "帮助性",
        "持续学习"
    ])
    
    # Dynamic narratives (evolve with experience)
    identity_narrative: str = ""
    capability_narrative: str = ""
    
    # Structured beliefs
    capabilities: Dict[str, CapabilityBelief] = field(default_factory=dict)
    relationships: Dict[str, RelationshipBelief] = field(default_factory=dict)
    
    # Supporting evidence
    supporting_theme_ids: List[str] = field(default_factory=list)
    
    # Coherence tracking
    coherence_score: float = 1.0
    unresolved_tensions: List[str] = field(default_factory=list)
    
    # Evolution history
    evolution_log: List[Tuple[float, str, str]] = field(default_factory=list)
    last_updated: float = field(default_factory=now_ts)
    
    def to_full_narrative(self) -> str:
        """Generate complete self-narrative"""
        sections = []
        
        # Identity
        sections.append(f"## 我是谁\n{self.identity_statement}")
        
        if self.identity_narrative:
            sections.append(self.identity_narrative)
        
        # Core values
        if self.core_values:
            sections.append(f"## 核心价值\n我重视: {', '.join(self.core_values)}")
        
        # Capabilities
        if self.capabilities:
            cap_narratives = [c.to_narrative() for c in self.capabilities.values()]
            sections.append(f"## 能力认知\n" + "。".join(cap_narratives))
        
        # Key relationships
        if self.relationships:
            rel_narratives = [r.to_narrative() for r in list(self.relationships.values())[:3]]
            sections.append(f"## 关系\n" + "\n".join(rel_narratives))
        
        # Tensions
        if self.unresolved_tensions:
            sections.append(f"## 待解决的张力\n" + "\n".join(f"- {t}" for t in self.unresolved_tensions[:3]))
        
        return "\n\n".join(sections)
    
    def get_capability(self, name: str) -> CapabilityBelief:
        """Get or create capability belief"""
        if name not in self.capabilities:
            self.capabilities[name] = CapabilityBelief(name=name, description=name)
        return self.capabilities[name]
    
    def get_relationship(self, entity_id: str, entity_type: str = "user") -> RelationshipBelief:
        """Get or create relationship belief"""
        if entity_id not in self.relationships:
            self.relationships[entity_id] = RelationshipBelief(
                entity_id=entity_id,
                entity_type=entity_type
            )
        return self.relationships[entity_id]
    
    def log_evolution(self, change_type: str, description: str) -> None:
        """Log a change to self-narrative"""
        self.evolution_log.append((now_ts(), change_type, description))
        self.last_updated = now_ts()
        
        # Keep log bounded
        if len(self.evolution_log) > 100:
            self.evolution_log = self.evolution_log[-100:]


# -----------------------------------------------------------------------------
# Self-Narrative Engine
# -----------------------------------------------------------------------------

class SelfNarrativeEngine:
    """
    Engine for maintaining and evolving self-narrative.
    
    Updates are:
    - Evidence-driven (from themes and stories)
    - Probabilistic (no hard thresholds)
    - Stable (resists rapid changes)
    """
    
    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)
        self.narrative = SelfNarrative()
        
        # Update momentum (controls rate of change)
        self.momentum = 0.9  # High momentum = slow to change
    
    def update_from_themes(self, themes: List[Theme]) -> bool:
        """
        Update self-narrative based on themes.
        Returns True if significant update occurred.
        """
        significant_change = False
        
        for theme in themes:
            if theme.confidence() < 0.5:
                continue  # Skip low-confidence themes
            
            # Check theme type and update accordingly
            theme_type = theme.theme_type
            
            if theme_type == "capability":
                change = self._update_capability_from_theme(theme)
                significant_change = significant_change or change
            
            elif theme_type == "limitation":
                change = self._update_limitation_from_theme(theme)
                significant_change = significant_change or change
            
            elif theme_type == "pattern":
                change = self._update_pattern_from_theme(theme)
                significant_change = significant_change or change
            
            elif theme_type == "preference":
                change = self._update_preference_from_theme(theme)
                significant_change = significant_change or change
        
        # Update supporting themes
        self.narrative.supporting_theme_ids = [
            t.id for t in themes if t.confidence() > 0.6
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
        """Update self-narrative from interaction outcome"""
        
        # Update relationship if entity specified
        if entity_id:
            rel = self.narrative.get_relationship(entity_id)
            rel.update_interaction(positive=success)
        
        # Try to infer capability from action
        action = getattr(plot, 'action', '') or getattr(plot, 'text', '')
        
        # Simple capability inference (could be enhanced with LLM)
        if "代码" in action or "编程" in action or "code" in action.lower():
            cap = self.narrative.get_capability("编程")
            cap.update(success)
        
        if "分析" in action or "理解" in action or "explain" in action.lower():
            cap = self.narrative.get_capability("分析解释")
            cap.update(success)
        
        if "计划" in action or "规划" in action or "plan" in action.lower():
            cap = self.narrative.get_capability("规划")
            cap.update(success)
    
    def add_tension(self, tension: str) -> None:
        """Add an unresolved tension"""
        if tension not in self.narrative.unresolved_tensions:
            self.narrative.unresolved_tensions.append(tension)
            self.narrative.log_evolution("tension_added", tension)
    
    def resolve_tension(self, tension: str) -> None:
        """Resolve a tension"""
        if tension in self.narrative.unresolved_tensions:
            self.narrative.unresolved_tensions.remove(tension)
            self.narrative.log_evolution("tension_resolved", tension)
    
    def check_coherence(self) -> float:
        """Check self-narrative coherence"""
        issues = 0
        total_checks = 0
        
        # Check capability contradictions
        for name, cap in self.narrative.capabilities.items():
            total_checks += 1
            
            # Check if capability conflicts with limitations
            if cap.capability_probability() > 0.7:
                # High capability - check for contradicting limitation themes
                for theme_id in self.narrative.supporting_theme_ids:
                    if "limitation" in theme_id.lower() and name.lower() in theme_id.lower():
                        issues += 1
        
        # Check relationship consistency
        for entity_id, rel in self.narrative.relationships.items():
            total_checks += 1
            
            # Trust should correlate with positive interaction ratio
            if rel.interaction_count > 5:
                positive_ratio = rel.positive_interactions / rel.interaction_count
                trust = rel.trust()
                
                if abs(positive_ratio - trust) > 0.3:
                    issues += 0.5  # Minor inconsistency
        
        # Check for too many unresolved tensions
        if len(self.narrative.unresolved_tensions) > 5:
            issues += 0.5
        
        self.narrative.coherence_score = 1.0 - (issues / max(total_checks, 1))
        return self.narrative.coherence_score
    
    def _update_capability_from_theme(self, theme: Theme) -> bool:
        """Extract capability from theme"""
        # Simple extraction (could be enhanced with LLM)
        name = theme.name
        description = theme.description
        
        cap = self.narrative.get_capability(name)
        
        # Update based on theme confidence
        confidence = theme.confidence()
        
        # Blend with momentum
        old_prob = cap.capability_probability()
        new_evidence = confidence
        
        if new_evidence > old_prob:
            cap.a += (new_evidence - old_prob) * (1 - self.momentum)
        else:
            cap.b += (old_prob - new_evidence) * (1 - self.momentum)
        
        return abs(confidence - old_prob) > 0.1
    
    def _update_limitation_from_theme(self, theme: Theme) -> bool:
        """Extract limitation from theme"""
        name = theme.name
        
        # Limitations reduce capability belief
        cap = self.narrative.get_capability(name)
        
        confidence = theme.confidence()
        old_prob = cap.capability_probability()
        
        # Limitation evidence reduces capability
        cap.b += confidence * (1 - self.momentum)
        
        return confidence > 0.5 and old_prob > 0.5
    
    def _update_pattern_from_theme(self, theme: Theme) -> bool:
        """Update from behavioral pattern theme"""
        # Patterns might indicate preferences or tendencies
        return False  # Placeholder
    
    def _update_preference_from_theme(self, theme: Theme) -> bool:
        """Update from preference theme"""
        return False  # Placeholder
    
    def _regenerate_narratives(self) -> None:
        """Regenerate natural language narratives"""
        
        # Capability narrative
        if self.narrative.capabilities:
            cap_parts = []
            
            strong_caps = [c for c in self.narrative.capabilities.values() if c.capability_probability() > 0.7]
            weak_caps = [c for c in self.narrative.capabilities.values() if c.capability_probability() < 0.3]
            
            if strong_caps:
                cap_parts.append("我擅长" + "、".join(c.name for c in strong_caps[:3]))
            
            if weak_caps:
                cap_parts.append("我在" + "、".join(c.name for c in weak_caps[:2]) + "方面还在学习")
            
            self.narrative.capability_narrative = "。".join(cap_parts)
        
        # Identity narrative
        if self.narrative.capabilities or self.narrative.relationships:
            parts = []
            
            # Summarize experience
            total_interactions = sum(
                r.interaction_count for r in self.narrative.relationships.values()
            )
            
            if total_interactions > 10:
                parts.append(f"通过{total_interactions}次互动，我不断学习和成长")
            
            if self.narrative.core_values:
                parts.append(f"我坚持{', '.join(self.narrative.core_values[:3])}的原则")
            
            self.narrative.identity_narrative = "。".join(parts)


# -----------------------------------------------------------------------------
# Identity Tracker (Evolution Analysis)
# -----------------------------------------------------------------------------

class IdentityTracker:
    """
    Track how self-narrative evolves over time.
    
    Useful for:
    - Detecting identity drift
    - Understanding growth patterns
    - Maintaining stability
    """
    
    def __init__(self):
        self.snapshots: List[Tuple[float, dict]] = []
        self.change_events: List[Tuple[float, str, float]] = []  # (ts, event, magnitude)
    
    def snapshot(self, narrative: SelfNarrative) -> None:
        """Take a snapshot of current identity"""
        state = {
            "identity_statement": narrative.identity_statement,
            "coherence_score": narrative.coherence_score,
            "capability_count": len(narrative.capabilities),
            "relationship_count": len(narrative.relationships),
            "tension_count": len(narrative.unresolved_tensions),
            "capability_probs": {
                name: cap.capability_probability()
                for name, cap in narrative.capabilities.items()
            },
        }
        
        self.snapshots.append((now_ts(), state))
        
        # Keep bounded
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]
    
    def detect_drift(self, narrative: SelfNarrative, window_hours: float = 24) -> float:
        """
        Detect identity drift in recent time window.
        Returns drift magnitude (0 = stable, 1 = major drift).
        """
        if len(self.snapshots) < 2:
            return 0.0
        
        current_ts = now_ts()
        window_start = current_ts - window_hours * 3600
        
        recent_snapshots = [
            (ts, state) for ts, state in self.snapshots
            if ts > window_start
        ]
        
        if len(recent_snapshots) < 2:
            return 0.0
        
        first_state = recent_snapshots[0][1]
        last_state = recent_snapshots[-1][1]
        
        # Compare capability probabilities
        drift = 0.0
        
        common_caps = set(first_state.get("capability_probs", {}).keys()) & \
                     set(last_state.get("capability_probs", {}).keys())
        
        for cap in common_caps:
            diff = abs(
                first_state["capability_probs"][cap] -
                last_state["capability_probs"][cap]
            )
            drift += diff
        
        if common_caps:
            drift /= len(common_caps)
        
        # Add coherence change
        coherence_diff = abs(
            first_state.get("coherence_score", 1.0) -
            last_state.get("coherence_score", 1.0)
        )
        drift = 0.7 * drift + 0.3 * coherence_diff
        
        return min(drift, 1.0)
    
    def log_change(self, event: str, magnitude: float) -> None:
        """Log an identity change event"""
        self.change_events.append((now_ts(), event, magnitude))
        
        if len(self.change_events) > 1000:
            self.change_events = self.change_events[-1000:]
    
    def get_stability_score(self, window_hours: float = 168) -> float:
        """
        Get identity stability score over time window.
        Higher = more stable.
        """
        current_ts = now_ts()
        window_start = current_ts - window_hours * 3600
        
        recent_changes = [
            (ts, event, mag) for ts, event, mag in self.change_events
            if ts > window_start
        ]
        
        if not recent_changes:
            return 1.0
        
        total_magnitude = sum(mag for _, _, mag in recent_changes)
        avg_magnitude = total_magnitude / len(recent_changes)
        
        # Stability decreases with change frequency and magnitude
        change_rate = len(recent_changes) / (window_hours / 24)  # changes per day
        
        stability = 1.0 / (1.0 + change_rate * avg_magnitude)
        return stability


# -----------------------------------------------------------------------------
# Identity Evolution Tracker - "How I became who I am"
# -----------------------------------------------------------------------------

@dataclass
class IdentityChange:
    """
    A recorded change in identity.
    
    Philosophy: Identity is not static. Understanding how we became
    who we are is as important as knowing who we are.
    """
    ts: float
    dimension: str                          # Which identity dimension changed
    from_identity: str                      # What I was before
    to_identity: str                        # What I became
    triggered_by_relationships: List[str]   # Which relationships contributed
    triggered_by_events: List[str]          # Brief summaries of triggering events
    magnitude: float = 0.5                  # How significant was this change
    
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
    def from_dict(cls, d: Dict[str, Any]) -> "IdentityChange":
        return cls(
            ts=d["ts"],
            dimension=d["dimension"],
            from_identity=d["from_identity"],
            to_identity=d["to_identity"],
            triggered_by_relationships=d.get("triggered_by_relationships", []),
            triggered_by_events=d.get("triggered_by_events", []),
            magnitude=d.get("magnitude", 0.5),
        )


class IdentityEvolutionTracker:
    """
    Track "how I became who I am".
    
    Key philosophy: Memory is not about the past, it's about becoming.
    This tracker enables the agent to understand its own evolution
    and to generate meaningful narratives about growth.
    """
    
    def __init__(self):
        self.evolution_log: List[IdentityChange] = []
        
        # Track identity dimensions over time
        self.dimension_history: Dict[str, List[Tuple[float, float]]] = {}  # dimension -> [(ts, strength)]
        
        # Significant milestones
        self.milestones: List[Dict[str, Any]] = []
    
    def record_identity_change(
        self,
        dimension: str,
        old_value: str,
        new_value: str,
        trigger_relationships: List[str],
        trigger_events: List[str],
        magnitude: float = 0.5,
    ) -> None:
        """
        Record a change in identity.
        
        This creates a record of "how I became who I am".
        """
        change = IdentityChange(
            ts=now_ts(),
            dimension=dimension,
            from_identity=old_value,
            to_identity=new_value,
            triggered_by_relationships=trigger_relationships,
            triggered_by_events=trigger_events,
            magnitude=magnitude,
        )
        
        self.evolution_log.append(change)
        
        # Keep bounded
        if len(self.evolution_log) > 500:
            self.evolution_log = self.evolution_log[-500:]
        
        # Check for milestone
        if magnitude > 0.7:
            self.milestones.append({
                "ts": now_ts(),
                "dimension": dimension,
                "description": f"从'{old_value}'成为'{new_value}'",
                "significance": "major" if magnitude > 0.85 else "notable",
            })
    
    def update_dimension_strength(self, dimension: str, strength: float) -> None:
        """Track the strength of an identity dimension over time."""
        if dimension not in self.dimension_history:
            self.dimension_history[dimension] = []
        
        self.dimension_history[dimension].append((now_ts(), strength))
        
        # Keep bounded
        if len(self.dimension_history[dimension]) > 100:
            self.dimension_history[dimension] = self.dimension_history[dimension][-100:]
    
    def generate_becoming_narrative(self) -> str:
        """
        Generate a narrative about "how I became who I am".
        
        This is the key output: a meaningful story of growth,
        not just a list of changes.
        """
        if not self.evolution_log:
            return "我还在探索自己是谁。每一次互动都在帮助我理解自己。"
        
        # Find significant changes
        significant_changes = self._find_significant_changes()
        
        if not significant_changes:
            return "我的身份在稳定地发展中。"
        
        narratives = []
        
        for change in significant_changes[:5]:  # Top 5 most significant
            # Generate narrative for this change
            relationships_str = "、".join(change.triggered_by_relationships[:2]) if change.triggered_by_relationships else "各种互动"
            
            narrative = f"通过与{relationships_str}的交流，"
            
            if change.from_identity and change.to_identity:
                narrative += f"我从「{change.from_identity}」逐渐成为「{change.to_identity}」。"
            else:
                narrative += f"我在「{change.dimension}」方面有了新的认识。"
            
            narratives.append(narrative)
        
        return "\n".join(narratives)
    
    def _find_significant_changes(self) -> List[IdentityChange]:
        """Find the most significant identity changes."""
        if not self.evolution_log:
            return []
        
        # Sort by magnitude and recency
        sorted_changes = sorted(
            self.evolution_log,
            key=lambda c: c.magnitude * (1.0 + 0.5 / (1.0 + (now_ts() - c.ts) / 86400)),
            reverse=True
        )
        
        return sorted_changes[:10]
    
    def support_reflection(self) -> Dict[str, Any]:
        """
        Support self-reflection.
        
        Returns insights about identity evolution that can help
        the agent understand itself better.
        """
        result = {
            "我为什么成为现在这样": self._explain_current_identity(),
            "我的主要成长轨迹": self._identify_growth_trajectories(),
            "哪些关系塑造了我": self._identify_formative_relationships(),
            "我还想成为什么": self._identify_growth_directions(),
        }
        
        return result
    
    def _explain_current_identity(self) -> str:
        """Explain how I became who I am now."""
        if not self.evolution_log:
            return "我的身份还在形成中。"
        
        # Group changes by dimension
        by_dimension: Dict[str, List[IdentityChange]] = {}
        for change in self.evolution_log:
            if change.dimension not in by_dimension:
                by_dimension[change.dimension] = []
            by_dimension[change.dimension].append(change)
        
        explanations = []
        for dim, changes in by_dimension.items():
            if changes:
                latest = changes[-1]
                explanations.append(
                    f"在{dim}方面，我经历了{len(changes)}次变化，"
                    f"现在是「{latest.to_identity}」。"
                )
        
        return "".join(explanations) if explanations else "我的身份在稳定中。"
    
    def _identify_growth_trajectories(self) -> List[str]:
        """Identify main growth trajectories."""
        trajectories = []
        
        for dimension, history in self.dimension_history.items():
            if len(history) >= 3:
                # Check if generally increasing
                first_third = history[:len(history)//3]
                last_third = history[-len(history)//3:]
                
                avg_first = sum(s for _, s in first_third) / len(first_third) if first_third else 0
                avg_last = sum(s for _, s in last_third) / len(last_third) if last_third else 0
                
                if avg_last > avg_first + 0.1:
                    trajectories.append(f"{dimension}：成长中 ↑")
                elif avg_last < avg_first - 0.1:
                    trajectories.append(f"{dimension}：转变中 ↓")
                else:
                    trajectories.append(f"{dimension}：稳定 →")
        
        return trajectories
    
    def _identify_formative_relationships(self) -> List[str]:
        """Identify relationships that shaped identity."""
        relationship_impact: Dict[str, float] = {}
        
        for change in self.evolution_log:
            for rel in change.triggered_by_relationships:
                if rel not in relationship_impact:
                    relationship_impact[rel] = 0.0
                relationship_impact[rel] += change.magnitude
        
        # Sort by impact
        sorted_rels = sorted(
            relationship_impact.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [f"{rel}（影响度：{impact:.1f}）" for rel, impact in sorted_rels[:5]]
    
    def _identify_growth_directions(self) -> List[str]:
        """Identify potential growth directions."""
        directions = []
        
        # Find dimensions that are weak but have shown recent activity
        for dimension, history in self.dimension_history.items():
            if history:
                recent_strength = history[-1][1]
                if recent_strength < 0.5:
                    # Check if there's been recent activity
                    recent_activity = len([h for h in history if now_ts() - h[0] < 7 * 86400])
                    if recent_activity > 0:
                        directions.append(f"继续发展「{dimension}」")
        
        if not directions:
            directions.append("继续在现有方向上深化")
        
        return directions
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of identity evolution."""
        return {
            "total_changes": len(self.evolution_log),
            "dimensions_tracked": list(self.dimension_history.keys()),
            "milestones_count": len(self.milestones),
            "recent_changes": len([c for c in self.evolution_log if now_ts() - c.ts < 7 * 86400]),
            "becoming_narrative": self.generate_becoming_narrative(),
        }
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to state dict."""
        return {
            "evolution_log": [c.to_dict() for c in self.evolution_log],
            "dimension_history": {
                dim: [(ts, s) for ts, s in hist]
                for dim, hist in self.dimension_history.items()
            },
            "milestones": self.milestones,
        }
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "IdentityEvolutionTracker":
        """Restore from state dict."""
        obj = cls()
        obj.evolution_log = [
            IdentityChange.from_dict(c) for c in d.get("evolution_log", [])
        ]
        obj.dimension_history = {
            dim: [(ts, s) for ts, s in hist]
            for dim, hist in d.get("dimension_history", {}).items()
        }
        obj.milestones = d.get("milestones", [])
        return obj
