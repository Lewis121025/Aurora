"""
AURORA 自我叙事跟踪器
=========================

身份漂移、稳定性和“我如何成为现在的我”的演化跟踪。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from aurora.core.self_narrative.models import IdentityChange, SelfNarrative
from aurora.utils.time_utils import now_ts


class IdentityTracker:
    """追踪自我叙事在时间上的变化。"""

    def __init__(self):
        self.snapshots: List[Tuple[float, dict]] = []
        self.change_events: List[Tuple[float, str, float]] = []

    def snapshot(self, narrative: SelfNarrative) -> None:
        state = {
            "identity_statement": narrative.identity_statement,
            "coherence_score": narrative.coherence_score,
            "capability_count": len(narrative.capabilities),
            "relationship_count": len(narrative.relationships),
            "tension_count": len(narrative.unresolved_tensions),
            "capability_probs": {
                name: capability.capability_probability()
                for name, capability in narrative.capabilities.items()
            },
        }
        self.snapshots.append((now_ts(), state))
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]

    def detect_drift(self, narrative: SelfNarrative, window_hours: float = 24) -> float:
        del narrative

        if len(self.snapshots) < 2:
            return 0.0

        window_start = now_ts() - window_hours * 3600
        recent_snapshots = [
            (ts, state)
            for ts, state in self.snapshots
            if ts > window_start
        ]
        if len(recent_snapshots) < 2:
            return 0.0

        first_state = recent_snapshots[0][1]
        last_state = recent_snapshots[-1][1]

        common_capabilities = (
            set(first_state.get("capability_probs", {}))
            & set(last_state.get("capability_probs", {}))
        )
        drift = 0.0
        for capability in common_capabilities:
            drift += abs(
                first_state["capability_probs"][capability]
                - last_state["capability_probs"][capability]
            )
        if common_capabilities:
            drift /= len(common_capabilities)

        coherence_diff = abs(
            first_state.get("coherence_score", 1.0)
            - last_state.get("coherence_score", 1.0)
        )
        return min(0.7 * drift + 0.3 * coherence_diff, 1.0)

    def log_change(self, event: str, magnitude: float) -> None:
        self.change_events.append((now_ts(), event, magnitude))
        if len(self.change_events) > 1000:
            self.change_events = self.change_events[-1000:]

    def get_stability_score(self, window_hours: float = 168) -> float:
        window_start = now_ts() - window_hours * 3600
        recent_changes = [
            change for change in self.change_events
            if change[0] > window_start
        ]
        if not recent_changes:
            return 1.0

        total_magnitude = sum(magnitude for _, _, magnitude in recent_changes)
        average_magnitude = total_magnitude / len(recent_changes)
        change_rate = len(recent_changes) / (window_hours / 24)
        return 1.0 / (1.0 + change_rate * average_magnitude)


class IdentityEvolutionTracker:
    """追踪“我如何成为现在的我”。"""

    def __init__(self):
        self.evolution_log: List[IdentityChange] = []
        self.dimension_history: Dict[str, List[Tuple[float, float]]] = {}
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
        if len(self.evolution_log) > 500:
            self.evolution_log = self.evolution_log[-500:]

        if magnitude > 0.7:
            self.milestones.append(
                {
                    "ts": now_ts(),
                    "dimension": dimension,
                    "description": f"从'{old_value}'成为'{new_value}'",
                    "significance": "major" if magnitude > 0.85 else "notable",
                }
            )

    def update_dimension_strength(self, dimension: str, strength: float) -> None:
        self.dimension_history.setdefault(dimension, []).append((now_ts(), strength))
        if len(self.dimension_history[dimension]) > 100:
            self.dimension_history[dimension] = self.dimension_history[dimension][-100:]

    def generate_becoming_narrative(self) -> str:
        if not self.evolution_log:
            return "我还在探索自己是谁。每一次互动都在帮助我理解自己。"

        significant_changes = self._find_significant_changes()
        if not significant_changes:
            return "我的身份在稳定地发展中。"

        narratives = []
        for change in significant_changes[:5]:
            relationships = "、".join(change.triggered_by_relationships[:2]) if change.triggered_by_relationships else "各种互动"
            if change.from_identity and change.to_identity:
                narratives.append(
                    f"通过与{relationships}的交流，我从「{change.from_identity}」逐渐成为「{change.to_identity}」。"
                )
            else:
                narratives.append(f"通过与{relationships}的交流，我在「{change.dimension}」方面有了新的认识。")

        return "\n".join(narratives)

    def _find_significant_changes(self) -> List[IdentityChange]:
        return sorted(
            self.evolution_log,
            key=lambda change: change.magnitude * (1.0 + 0.5 / (1.0 + (now_ts() - change.ts) / 86400)),
            reverse=True,
        )[:10]

    def support_reflection(self) -> Dict[str, Any]:
        return {
            "我为什么成为现在这样": self._explain_current_identity(),
            "我的主要成长轨迹": self._identify_growth_trajectories(),
            "哪些关系塑造了我": self._identify_formative_relationships(),
            "我还想成为什么": self._identify_growth_directions(),
        }

    def _explain_current_identity(self) -> str:
        if not self.evolution_log:
            return "我的身份还在形成中。"

        changes_by_dimension: Dict[str, List[IdentityChange]] = {}
        for change in self.evolution_log:
            changes_by_dimension.setdefault(change.dimension, []).append(change)

        explanations = []
        for dimension, changes in changes_by_dimension.items():
            latest = changes[-1]
            explanations.append(
                f"在{dimension}方面，我经历了{len(changes)}次变化，现在是「{latest.to_identity}」。"
            )
        return "".join(explanations) if explanations else "我的身份在稳定中。"

    def _identify_growth_trajectories(self) -> List[str]:
        trajectories: List[str] = []
        for dimension, history in self.dimension_history.items():
            if len(history) < 3:
                continue

            first_window = history[: len(history) // 3]
            last_window = history[-len(history) // 3 :]
            avg_first = sum(strength for _, strength in first_window) / len(first_window) if first_window else 0
            avg_last = sum(strength for _, strength in last_window) / len(last_window) if last_window else 0

            if avg_last > avg_first + 0.1:
                trajectories.append(f"{dimension}：成长中 ↑")
            elif avg_last < avg_first - 0.1:
                trajectories.append(f"{dimension}：转变中 ↓")
            else:
                trajectories.append(f"{dimension}：稳定 →")

        return trajectories

    def _identify_formative_relationships(self) -> List[str]:
        relationship_impact: Dict[str, float] = {}
        for change in self.evolution_log:
            for relationship in change.triggered_by_relationships:
                relationship_impact[relationship] = relationship_impact.get(relationship, 0.0) + change.magnitude

        return [
            f"{relationship}（影响度：{impact:.1f}）"
            for relationship, impact in sorted(
                relationship_impact.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ]

    def _identify_growth_directions(self) -> List[str]:
        directions: List[str] = []
        for dimension, history in self.dimension_history.items():
            if not history:
                continue
            recent_strength = history[-1][1]
            recent_activity = sum(1 for ts, _ in history if now_ts() - ts < 7 * 86400)
            if recent_strength < 0.5 and recent_activity > 0:
                directions.append(f"继续发展「{dimension}」")

        return directions or ["继续在现有方向上深化"]

    def get_evolution_summary(self) -> Dict[str, Any]:
        return {
            "total_changes": len(self.evolution_log),
            "dimensions_tracked": list(self.dimension_history.keys()),
            "milestones_count": len(self.milestones),
            "recent_changes": len([change for change in self.evolution_log if now_ts() - change.ts < 7 * 86400]),
            "becoming_narrative": self.generate_becoming_narrative(),
        }

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "evolution_log": [change.to_dict() for change in self.evolution_log],
            "dimension_history": {
                dimension: [(ts, strength) for ts, strength in history]
                for dimension, history in self.dimension_history.items()
            },
            "milestones": self.milestones,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "IdentityEvolutionTracker":
        tracker = cls()
        tracker.evolution_log = [
            IdentityChange.from_dict(change)
            for change in data.get("evolution_log", [])
        ]
        tracker.dimension_history = {
            dimension: [(ts, strength) for ts, strength in history]
            for dimension, history in data.get("dimension_history", {}).items()
        }
        tracker.milestones = data.get("milestones", [])
        return tracker
