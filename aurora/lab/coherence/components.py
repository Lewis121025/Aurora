"""
AURORA 连贯性分析组件
=========================

包含信念传播、矛盾检测、连贯性评分和基础冲突解决。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from aurora.lab.causal.models import CausalEdgeBelief
from aurora.lab.coherence.models import (
    BeliefState,
    CoherenceReport,
    Conflict,
    ConflictType,
    Resolution,
)
from aurora.lab.coherence.config import (
    ANTI_CORRELATION_THRESHOLD,
    BELIEF_PROPAGATION_ITERATIONS,
    COHERENCE_WEIGHTS,
    HIGH_SIMILARITY_THRESHOLD,
    MAX_COHERENCE_PAIRS,
    OPPOSITION_SCORE_THRESHOLD,
    UNFINISHED_STORY_HOURS,
)
from aurora.lab.graph.memory_graph import MemoryGraph
from aurora.lab.primitives.metric import LowRankMetric
from aurora.lab.models.plot import Plot
from aurora.lab.models.story import StoryArc
from aurora.lab.models.theme import Theme
from aurora.lab.support.types import MemoryElement
from aurora.utils.embedding_utils import get_embedding_from_object
from aurora.utils.math_utils import cosine_sim, l2_normalize, sigmoid
from aurora.utils.time_utils import now_ts


class BeliefNetwork:
    """用于连贯性推理的概率性信念网络。"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.beliefs: Dict[str, BeliefState] = {}

    def add_belief(
        self,
        node_id: str,
        prior: float,
        evidence_strength: float = 1.0,
    ) -> None:
        self.graph.add_node(node_id)
        self.beliefs[node_id] = BeliefState(
            prior=prior,
            evidence_strength=evidence_strength,
        )

    def add_dependency(
        self,
        from_id: str,
        to_id: str,
        dependency_type: str,
        strength: float,
    ) -> None:
        self.graph.add_edge(
            from_id,
            to_id,
            type=dependency_type,
            strength=strength,
        )

    def propagate_beliefs(
        self,
        iterations: int = BELIEF_PROPAGATION_ITERATIONS,
    ) -> Dict[str, float]:
        probabilities = {
            node_id: state.prior
            for node_id, state in self.beliefs.items()
        }

        for _ in range(iterations):
            new_probabilities: Dict[str, float] = {}

            for node_id in self.graph.nodes():
                state = self.beliefs.get(node_id)
                if state is None:
                    continue

                support = 0.0
                contradiction = 0.0
                for predecessor in self.graph.predecessors(node_id):
                    edge = self.graph.edges[predecessor, node_id]
                    predecessor_prob = probabilities.get(predecessor, 0.5)
                    if edge["type"] == "supports":
                        support += edge["strength"] * predecessor_prob
                    elif edge["type"] == "contradicts":
                        contradiction += edge["strength"] * predecessor_prob

                influence = support - contradiction
                updated = sigmoid(
                    math.log(state.prior / (1 - state.prior + 1e-9) + 1e-9)
                    + state.evidence_strength * influence
                )
                new_probabilities[node_id] = updated

            probabilities = new_probabilities

        return probabilities


class ContradictionDetector:
    """检测内存元素之间的概率性矛盾。"""

    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)
        self.opposition_patterns: List[Tuple[np.ndarray, np.ndarray]] = []

    def detect_contradiction(
        self,
        a: MemoryElement,
        b: MemoryElement,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        del context

        log_odds_contradiction = 0.0
        explanations: List[str] = []

        emb_a = self._get_embedding(a)
        emb_b = self._get_embedding(b)
        if emb_a is None or emb_b is None:
            return 0.0, "Cannot compare: missing embeddings"

        opposition_score = self._semantic_opposition_score(emb_a, emb_b)
        if opposition_score > OPPOSITION_SCORE_THRESHOLD:
            log_odds_contradiction += opposition_score * 2
            explanations.append(f"Semantic opposition ({opposition_score:.2f})")

        polarity_conflict = self._check_polarity_conflict(a, b)
        if polarity_conflict > 0:
            log_odds_contradiction += polarity_conflict
            explanations.append("Polarity conflict detected")

        temporal_conflict = self._check_temporal_conflict(a, b)
        if temporal_conflict > 0:
            log_odds_contradiction += temporal_conflict
            explanations.append("Temporal inconsistency")

        actor_conflict = self._check_actor_conflict(a, b)
        if actor_conflict > 0:
            log_odds_contradiction += actor_conflict
            explanations.append("Actor state conflict")

        probability = sigmoid(log_odds_contradiction)
        explanation = "; ".join(explanations) if explanations else "No contradiction detected"
        return probability, explanation

    def _get_embedding(self, obj: MemoryElement) -> Optional[np.ndarray]:
        return get_embedding_from_object(obj)

    def _semantic_opposition_score(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
    ) -> float:
        similarity = cosine_sim(emb_a, emb_b)
        if similarity > HIGH_SIMILARITY_THRESHOLD:
            return 0.0
        if similarity < ANTI_CORRELATION_THRESHOLD:
            return abs(similarity)

        for positive_pattern, negative_pattern in self.opposition_patterns:
            projection_a = np.dot(emb_a, positive_pattern)
            projection_b = np.dot(emb_b, negative_pattern)
            if projection_a > 0.5 and projection_b > 0.5:
                return 0.7

        if 0.2 < similarity < 0.5:
            return 0.3 * (0.5 - similarity)
        return 0.0

    def _check_polarity_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        polarity_a = getattr(a, "polarity", None) or getattr(a, "emotion_valence", 0)
        polarity_b = getattr(b, "polarity", None) or getattr(b, "emotion_valence", 0)

        if isinstance(polarity_a, str):
            polarity_a = 1.0 if polarity_a == "positive" else -1.0
        if isinstance(polarity_b, str):
            polarity_b = 1.0 if polarity_b == "positive" else -1.0

        subject_a = getattr(a, "subject", "") or ""
        subject_b = getattr(b, "subject", "") or ""
        predicate_a = getattr(a, "predicate", "") or ""
        predicate_b = getattr(b, "predicate", "") or ""

        if (
            subject_a
            and subject_a.lower() == subject_b.lower()
            and predicate_a.lower() == predicate_b.lower()
            and polarity_a * polarity_b < 0
        ):
            return 1.5
        return 0.0

    def _check_temporal_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        ts_a = getattr(a, "ts", None) or getattr(a, "timestamp", None)
        ts_b = getattr(b, "ts", None) or getattr(b, "timestamp", None)
        if ts_a is None or ts_b is None:
            return 0.0
        return 0.0

    def _check_actor_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        actors_a = set(getattr(a, "actors", []) or [])
        actors_b = set(getattr(b, "actors", []) or [])
        shared_actors = actors_a & actors_b
        if not shared_actors:
            return 0.0

        emb_a = self._get_embedding(a)
        emb_b = self._get_embedding(b)
        if emb_a is None or emb_b is None:
            return 0.0

        similarity = cosine_sim(emb_a, emb_b)
        if similarity < 0.3:
            return 0.5 * len(shared_actors) * (0.3 - similarity)
        return 0.0

    def learn_opposition_pattern(
        self,
        positive_examples: List[np.ndarray],
        negative_examples: List[np.ndarray],
    ) -> None:
        if not positive_examples or not negative_examples:
            return

        positive_pattern = l2_normalize(np.mean(positive_examples, axis=0))
        negative_pattern = l2_normalize(np.mean(negative_examples, axis=0))
        self.opposition_patterns.append((positive_pattern, negative_pattern))


class CoherenceScorer:
    """计算内存系统的整体连贯性分数。"""

    def __init__(
        self,
        metric: LowRankMetric,
        detector: ContradictionDetector,
        seed: int = 0,
    ):
        self.metric = metric
        self.detector = detector
        self.rng = np.random.default_rng(seed)

    def compute_coherence(
        self,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> CoherenceReport:
        del graph

        conflicts: List[Conflict] = []
        factual_conflicts, factual_score = self._check_factual_coherence(plots)
        conflicts.extend(factual_conflicts)

        temporal_conflicts, temporal_score = self._check_temporal_coherence(plots, stories)
        conflicts.extend(temporal_conflicts)

        if causal_beliefs:
            causal_conflicts, causal_score = self._check_causal_coherence(causal_beliefs)
            conflicts.extend(causal_conflicts)
        else:
            causal_score = 1.0

        thematic_conflicts, thematic_score = self._check_thematic_coherence(themes)
        conflicts.extend(thematic_conflicts)

        unfinished_stories = [
            story.id
            for story in stories.values()
            if story.status == "developing"
            and (now_ts() - story.updated_ts) > UNFINISHED_STORY_HOURS * 3600
        ]
        orphan_plots = [
            plot.id
            for plot in plots.values()
            if plot.story_id is None and plot.status == "active"
        ]

        weights = [
            COHERENCE_WEIGHTS["factual"],
            COHERENCE_WEIGHTS["temporal"],
            COHERENCE_WEIGHTS["causal"],
            COHERENCE_WEIGHTS["thematic"],
        ]
        scores = [factual_score, temporal_score, causal_score, thematic_score]
        log_score = sum(weight * math.log(score + 1e-9) for weight, score in zip(weights, scores))
        overall_score = math.exp(log_score)

        return CoherenceReport(
            overall_score=overall_score,
            conflicts=conflicts,
            unfinished_stories=unfinished_stories,
            orphan_plots=orphan_plots,
            factual_coherence=factual_score,
            temporal_coherence=temporal_score,
            causal_coherence=causal_score,
            thematic_coherence=thematic_score,
            recommended_actions=self._generate_recommendations(conflicts),
        )

    def _check_factual_coherence(
        self,
        plots: Dict[str, Plot],
    ) -> Tuple[List[Conflict], float]:
        conflicts: List[Conflict] = []
        total_pairs = 0
        contradiction_sum = 0.0
        plot_list = list(plots.values())
        max_pairs = min(MAX_COHERENCE_PAIRS, len(plot_list) * (len(plot_list) - 1) // 2)

        for idx, first_plot in enumerate(plot_list):
            for second_plot in plot_list[idx + 1 :]:
                if total_pairs >= max_pairs:
                    break

                total_pairs += 1
                probability, explanation = self.detector.detect_contradiction(first_plot, second_plot)
                contradiction_sum += probability
                if probability > 0.6:
                    conflicts.append(
                        Conflict(
                            type=ConflictType.FACTUAL,
                            node_a=first_plot.id,
                            node_b=second_plot.id,
                            severity=probability,
                            confidence=0.7,
                            description=explanation,
                        )
                    )

        avg_contradiction = contradiction_sum / max(total_pairs, 1)
        return conflicts, 1.0 - avg_contradiction

    def _check_temporal_coherence(
        self,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
    ) -> Tuple[List[Conflict], float]:
        conflicts: List[Conflict] = []
        inconsistency_count = 0
        total_checks = 0

        for story in stories.values():
            if len(story.plot_ids) < 2:
                continue

            story_plots = [plots[plot_id] for plot_id in story.plot_ids if plot_id in plots]
            story_plots.sort(key=lambda plot: plot.ts)

            for first_plot, second_plot in zip(story_plots, story_plots[1:]):
                total_checks += 1
                if second_plot.ts - first_plot.ts >= 0:
                    continue

                inconsistency_count += 1
                conflicts.append(
                    Conflict(
                        type=ConflictType.TEMPORAL,
                        node_a=first_plot.id,
                        node_b=second_plot.id,
                        severity=1.0,
                        confidence=1.0,
                        description="Temporal ordering violation",
                    )
                )

        return conflicts, 1.0 - (inconsistency_count / max(total_checks, 1))

    def _check_causal_coherence(
        self,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
    ) -> Tuple[List[Conflict], float]:
        conflicts: List[Conflict] = []
        dag = nx.DiGraph()
        for (source, target), belief in causal_beliefs.items():
            if belief.direction_belief() > 0.5:
                dag.add_edge(source, target, weight=belief.effective_causal_weight())

        cycles = list(nx.simple_cycles(dag))
        for cycle in cycles[:10]:
            conflicts.append(
                Conflict(
                    type=ConflictType.CAUSAL,
                    node_a=cycle[0],
                    node_b=cycle[-1],
                    severity=0.8,
                    confidence=1.0,
                    description=f"Causal cycle detected: {' → '.join(cycle[:5])}...",
                    evidence=cycle,
                )
            )

        cycle_penalty = len(cycles) * 0.1
        return conflicts, max(0.0, 1.0 - cycle_penalty)

    def _check_thematic_coherence(
        self,
        themes: Dict[str, Theme],
    ) -> Tuple[List[Conflict], float]:
        conflicts: List[Conflict] = []
        conflict_count = 0
        total_pairs = 0
        theme_list = list(themes.values())

        for idx, first_theme in enumerate(theme_list):
            for second_theme in theme_list[idx + 1 :]:
                total_pairs += 1
                probability, explanation = self.detector.detect_contradiction(first_theme, second_theme)
                if probability <= 0.5:
                    continue

                conflict_count += 1
                shared_stories = set(first_theme.story_ids) & set(second_theme.story_ids)
                severity = probability * 1.2 if shared_stories else probability * 0.8
                conflicts.append(
                    Conflict(
                        type=ConflictType.THEMATIC,
                        node_a=first_theme.id,
                        node_b=second_theme.id,
                        severity=min(severity, 1.0),
                        confidence=0.6,
                        description=f"Themes may conflict: {explanation}",
                        evidence=list(shared_stories),
                    )
                )

        return conflicts, 1.0 - (conflict_count / max(total_pairs, 1))

    def _generate_recommendations(
        self,
        conflicts: List[Conflict],
    ) -> List[Resolution]:
        recommendations: List[Resolution] = []

        for conflict in sorted(conflicts, key=lambda item: item.severity, reverse=True)[:5]:
            if conflict.type == ConflictType.FACTUAL:
                recommendations.append(
                    Resolution(
                        strategy="condition",
                        target_node=conflict.node_a,
                        action_description=(
                            f"Add condition to {conflict.node_a} to resolve conflict with {conflict.node_b}"
                        ),
                        expected_coherence_gain=conflict.severity * 0.7,
                        cost=0.1,
                        condition="Different contexts may apply",
                    )
                )
            elif conflict.type == ConflictType.CAUSAL:
                recommendations.append(
                    Resolution(
                        strategy="remove",
                        target_node=conflict.node_a,
                        action_description=(
                            f"Remove weakest causal link in cycle involving {conflict.node_a}"
                        ),
                        expected_coherence_gain=conflict.severity * 0.8,
                        cost=0.2,
                    )
                )
            elif conflict.type == ConflictType.THEMATIC:
                recommendations.append(
                    Resolution(
                        strategy="merge",
                        target_node=conflict.node_a,
                        action_description=(
                            f"Merge {conflict.node_a} and {conflict.node_b} into conditional theme"
                        ),
                        expected_coherence_gain=conflict.severity * 0.6,
                        cost=0.3,
                    )
                )

        return recommendations


class ConflictResolver:
    """自动解决连贯性冲突。"""

    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)

    def resolve(
        self,
        conflict: Conflict,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> bool:
        del graph

        if not conflict.resolutions:
            return False

        best_resolution = min(
            conflict.resolutions,
            key=lambda resolution: resolution.cost - resolution.expected_coherence_gain,
        )
        if best_resolution.expected_coherence_gain < 0.1:
            return False

        return self._apply_resolution(best_resolution, plots, stories, themes)

    def _apply_resolution(
        self,
        resolution: Resolution,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> bool:
        del stories

        if resolution.strategy == "weaken" and resolution.target_node in themes:
            themes[resolution.target_node].b += 1.0
            return True

        if resolution.strategy == "condition" and resolution.target_node in themes:
            themes[resolution.target_node].description += f" (Condition: {resolution.condition})"
            return True

        if resolution.strategy == "remove" and resolution.target_node in plots:
            plots[resolution.target_node].status = "archived"
            return True

        return False
