"""Coherence orchestration for lab analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from aurora.lab.causal.models import CausalEdgeBelief
from aurora.lab.coherence.components import (
    BeliefNetwork,
    CoherenceScorer,
    ConflictResolver,
    ContradictionDetector,
)
from aurora.lab.coherence.tension import Tension, TensionManager, TensionType
from aurora.lab.coherence.models import (
    CoherenceReport,
    Conflict,
    ConflictType,
    Resolution,
)
from aurora.lab.graph.memory_graph import MemoryGraph
from aurora.lab.knowledge.classifier import KnowledgeClassifier
from aurora.lab.primitives.metric import LowRankMetric
from aurora.lab.knowledge.models import ConflictAnalysis, KnowledgeType
from aurora.lab.models.plot import Plot
from aurora.lab.models.story import StoryArc
from aurora.lab.models.theme import Theme
from aurora.lab.support.types import MemoryElement
from aurora.utils.embedding_utils import get_embedding_from_object


class CoherenceGuardian:
    """Main entrypoint for coherence analysis and targeted conflict handling."""

    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.detector = ContradictionDetector(metric, seed)
        self.scorer = CoherenceScorer(metric, self.detector, seed)
        self.resolver = ConflictResolver(metric, seed)
        self.belief_network = BeliefNetwork()
        self.tension_manager = TensionManager(seed=seed)
        self.knowledge_classifier = KnowledgeClassifier(seed=seed)

    def full_check(
        self,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> CoherenceReport:
        return self.scorer.compute_coherence(graph, plots, stories, themes, causal_beliefs)

    def full_check_with_tension_analysis(
        self,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> Tuple[CoherenceReport, Dict[str, Any]]:
        report = self.scorer.compute_coherence(graph, plots, stories, themes, causal_beliefs)
        return report, self._build_tension_analysis(report, plots, stories, themes)

    def _build_tension_analysis(
        self,
        report: CoherenceReport,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> Dict[str, Any]:
        buckets = {
            "conflicts_to_resolve": [],
            "conflicts_to_preserve": [],
            "conflicts_to_accept": [],
            "conflicts_to_defer": [],
        }

        for conflict in report.conflicts:
            tension = self._conflict_to_tension(conflict, plots, stories, themes)
            if tension is None:
                continue

            tension_type = self.tension_manager.classify_tension(tension)
            if tension_type in {TensionType.ACTION_BLOCKING, TensionType.IDENTITY_THREATENING}:
                buckets["conflicts_to_resolve"].append(
                    {"conflict": conflict, "tension": tension, "reason": f"必须解决：{tension_type.value}"}
                )
            elif tension_type == TensionType.ADAPTIVE:
                buckets["conflicts_to_preserve"].append(
                    {"conflict": conflict, "tension": tension, "reason": "保留：提供灵活性的适应性矛盾"}
                )
            elif tension_type == TensionType.DEVELOPMENTAL:
                buckets["conflicts_to_accept"].append(
                    {"conflict": conflict, "tension": tension, "reason": "接受：成长的标志"}
                )
            else:
                buckets["conflicts_to_defer"].append(
                    {"conflict": conflict, "tension": tension, "reason": "需要更多信息"}
                )

        buckets["summary"] = {
            "total": len(report.conflicts),
            "to_resolve": len(buckets["conflicts_to_resolve"]),
            "to_preserve": len(buckets["conflicts_to_preserve"]),
            "to_accept": len(buckets["conflicts_to_accept"]),
            "to_defer": len(buckets["conflicts_to_defer"]),
        }
        return buckets

    def _conflict_to_tension(
        self,
        conflict: Conflict,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> Optional[Tension]:
        element_a = self._get_element(conflict.node_a, plots, stories, themes)
        element_b = self._get_element(conflict.node_b, plots, stories, themes)
        if element_a is None or element_b is None:
            return None

        tension = self.tension_manager.detect_tension(
            {"id": conflict.node_a, "type": conflict.type.value, "text": conflict.description},
            {"id": conflict.node_b, "type": conflict.type.value, "text": ""},
            self._get_embedding(element_a),
            self._get_embedding(element_b),
        )
        if tension is not None:
            return tension

        fallback = Tension(
            id=str(uuid4()),
            element_a_id=conflict.node_a,
            element_a_type=conflict.type.value,
            element_b_id=conflict.node_b,
            element_b_type=conflict.type.value,
            description=conflict.description,
            severity=conflict.severity,
        )
        self.tension_manager.tensions[fallback.id] = fallback
        return fallback

    def _get_element(
        self,
        node_id: str,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> Optional[MemoryElement]:
        if node_id in plots:
            return plots[node_id]
        if node_id in stories:
            return stories[node_id]
        if node_id in themes:
            return themes[node_id]
        return None

    def _get_embedding(self, element: MemoryElement) -> Optional[np.ndarray]:
        return get_embedding_from_object(element)

    def auto_resolve(
        self,
        report: CoherenceReport,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        max_resolutions: int = 3,
    ) -> int:
        resolved = 0
        conflicts = sorted(
            report.conflicts,
            key=lambda conflict: conflict.severity * conflict.confidence,
            reverse=True,
        )

        for conflict in conflicts[:max_resolutions]:
            if not conflict.resolutions:
                conflict.resolutions = self._generate_resolutions(conflict)
            if self.resolver.resolve(conflict, graph, plots, stories, themes):
                resolved += 1

        return resolved

    def smart_resolve(
        self,
        report: CoherenceReport,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        max_resolutions: int = 3,
    ) -> Dict[str, Any]:
        del report

        _, tension_analysis = self.full_check_with_tension_analysis(graph, plots, stories, themes)
        actions_taken = {"resolved": [], "preserved": [], "accepted": []}

        for item in tension_analysis["conflicts_to_resolve"][:max_resolutions]:
            conflict = item["conflict"]
            tension = item["tension"]
            if not conflict.resolutions:
                conflict.resolutions = self._generate_resolutions(conflict)
            if self.resolver.resolve(conflict, graph, plots, stories, themes):
                resolution = self.tension_manager.handle_tension(tension)
                actions_taken["resolved"].append(
                    {
                        "conflict_id": f"{conflict.node_a}-{conflict.node_b}",
                        "description": conflict.description,
                        "action": resolution.action,
                        "rationale": resolution.rationale,
                    }
                )

        for bucket_name, output_name in (
            ("conflicts_to_preserve", "preserved"),
            ("conflicts_to_accept", "accepted"),
        ):
            for item in tension_analysis[bucket_name]:
                resolution = self.tension_manager.handle_tension(item["tension"])
                actions_taken[output_name].append(
                    {
                        "conflict_description": item["conflict"].description,
                        "reason": item["reason"],
                        "action": resolution.action,
                    }
                )

        return actions_taken

    def _generate_resolutions(self, conflict: Conflict) -> List[Resolution]:
        resolutions: List[Resolution] = []

        if conflict.type == ConflictType.FACTUAL:
            resolutions.append(
                Resolution(
                    strategy="weaken",
                    target_node=conflict.node_b,
                    action_description=f"Reduce confidence in {conflict.node_b}",
                    expected_coherence_gain=conflict.severity * 0.5,
                    cost=0.2,
                )
            )
            resolutions.append(
                Resolution(
                    strategy="condition",
                    target_node=conflict.node_a,
                    action_description=f"Add context condition to {conflict.node_a}",
                    expected_coherence_gain=conflict.severity * 0.7,
                    cost=0.1,
                    condition="May depend on context",
                )
            )
        elif conflict.type == ConflictType.THEMATIC:
            resolutions.append(
                Resolution(
                    strategy="merge",
                    target_node=conflict.node_a,
                    action_description="Merge conflicting themes",
                    expected_coherence_gain=conflict.severity * 0.6,
                    cost=0.3,
                )
            )
        elif conflict.type == ConflictType.CAUSAL:
            resolutions.append(
                Resolution(
                    strategy="remove",
                    target_node=conflict.node_b,
                    action_description="Remove edge to break cycle",
                    expected_coherence_gain=conflict.severity * 0.8,
                    cost=0.15,
                )
            )

        return resolutions

    def update_belief_network(
        self,
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> Dict[str, float]:
        self.belief_network = BeliefNetwork()

        for theme in themes.values():
            self.belief_network.add_belief(
                theme.id,
                prior=theme.confidence(),
                evidence_strength=len(theme.story_ids) * 0.1,
            )

        if causal_beliefs:
            for (source, target), belief in causal_beliefs.items():
                if source not in themes or target not in themes:
                    continue
                dependency_type = "supports" if belief.effective_causal_weight() > 0.5 else "contradicts"
                self.belief_network.add_dependency(
                    source,
                    target,
                    dependency_type=dependency_type,
                    strength=belief.effective_causal_weight(),
                )

        return self.belief_network.propagate_beliefs()

    def get_tension_summary(self) -> Dict[str, Any]:
        return self.tension_manager.get_tension_summary()

    def analyze_knowledge_conflict(
        self,
        plot_a: Plot,
        plot_b: Plot,
    ) -> ConflictAnalysis:
        type_a = self._get_plot_knowledge_type(plot_a)
        type_b = self._get_plot_knowledge_type(plot_b)

        if abs(plot_a.ts - plot_b.ts) < 3600:
            time_relation = "concurrent"
        else:
            time_relation = "sequential"

        return self.knowledge_classifier.resolve_conflict(
            type_a=type_a,
            type_b=type_b,
            time_relation=time_relation,
            text_a=plot_a.text,
            text_b=plot_b.text,
            embedding_a=plot_a.embedding,
            embedding_b=plot_b.embedding,
        )

    def _get_plot_knowledge_type(self, plot: Plot) -> KnowledgeType:
        if plot.knowledge_type is not None:
            type_map = {
                "factual_state": KnowledgeType.FACTUAL_STATE,
                "factual_static": KnowledgeType.FACTUAL_STATIC,
                "identity_trait": KnowledgeType.IDENTITY_TRAIT,
                "identity_value": KnowledgeType.IDENTITY_VALUE,
                "preference": KnowledgeType.PREFERENCE,
                "behavior": KnowledgeType.BEHAVIOR_PATTERN,
                "unknown": KnowledgeType.UNKNOWN,
            }
            return type_map.get(plot.knowledge_type, KnowledgeType.UNKNOWN)

        result = self.knowledge_classifier.classify(plot.text, embedding=plot.embedding)
        return result.knowledge_type

    def check_complementary_traits(
        self,
        plot_a: Plot,
        plot_b: Plot,
    ) -> bool:
        return self.knowledge_classifier.are_complementary_traits(
            text_a=plot_a.text,
            text_b=plot_b.text,
            embedding_a=plot_a.embedding,
            embedding_b=plot_b.embedding,
        )

    def get_conflict_resolution_recommendation(
        self,
        conflict: Conflict,
        plots: Dict[str, Plot],
    ) -> Dict[str, Any]:
        plot_a = plots.get(conflict.node_a)
        plot_b = plots.get(conflict.node_b)
        if plot_a is None or plot_b is None:
            return {
                "strategy": "unknown",
                "rationale": "Could not find one or both plots",
                "actions": ["Manual review required"],
                "confidence": 0.0,
            }

        analysis = self.analyze_knowledge_conflict(plot_a, plot_b)
        return {
            "strategy": analysis.resolution.value,
            "rationale": analysis.rationale,
            "actions": analysis.recommended_actions,
            "confidence": analysis.confidence,
            "is_complementary": analysis.is_complementary,
            "requires_human_review": analysis.requires_human_review,
            "knowledge_type_a": analysis.knowledge_type_a.value,
            "knowledge_type_b": analysis.knowledge_type_b.value,
        }
