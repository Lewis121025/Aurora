"""
AURORA Coherence Guardian Module
================================

Maintains narrative and factual coherence without hard-coded rules.
All consistency judgments are probabilistic.

Key components:
- ContradictionDetector: Probabilistic contradiction detection
- CoherenceScorer: Compute global coherence score
- ConflictResolver: Automated conflict resolution strategies
- BeliefNetwork: Probabilistic belief propagation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import numpy as np
import networkx as nx

from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.types import MemoryElement
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.causal import CausalEdgeBelief, CausalMemoryGraph
from aurora.algorithms.tension import TensionManager, Tension, TensionType, TensionResolution
from aurora.algorithms.constants import (
    OPPOSITION_SCORE_THRESHOLD,
    HIGH_SIMILARITY_THRESHOLD,
    ANTI_CORRELATION_THRESHOLD,
    UNFINISHED_STORY_HOURS,
    MAX_COHERENCE_PAIRS,
    BELIEF_PROPAGATION_ITERATIONS,
    COHERENCE_WEIGHTS,
)
from aurora.utils.math_utils import l2_normalize, cosine_sim, sigmoid, softmax
from aurora.utils.time_utils import now_ts
from aurora.utils.embedding_utils import get_embedding_from_object


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

class ConflictType(Enum):
    """Types of coherence conflicts"""
    FACTUAL = "factual"           # Contradicting facts
    TEMPORAL = "temporal"         # Timeline inconsistency
    CAUSAL = "causal"             # Causal cycle or paradox
    THEMATIC = "thematic"         # Conflicting themes
    SELF_NARRATIVE = "self"       # Self-narrative inconsistency


@dataclass
class Conflict:
    """Detected coherence conflict"""
    type: ConflictType
    node_a: str
    node_b: str
    
    # Probabilistic severity (not a hard threshold)
    severity: float  # 0-1, higher = more severe
    confidence: float  # 0-1, how confident in this detection
    
    description: str
    evidence: List[str] = field(default_factory=list)
    
    # Potential resolutions
    resolutions: List['Resolution'] = field(default_factory=list)


@dataclass
class Resolution:
    """Proposed resolution for a conflict"""
    strategy: str  # "weaken", "condition", "merge", "remove"
    target_node: str
    action_description: str
    
    # Expected outcomes
    expected_coherence_gain: float
    cost: float  # Information loss or complexity
    
    # Conditions under which this resolution applies
    condition: Optional[str] = None


@dataclass
class CoherenceReport:
    """Full coherence analysis report"""
    overall_score: float  # 0-1, higher = more coherent
    
    conflicts: List[Conflict]
    unfinished_stories: List[str]
    orphan_plots: List[str]
    
    # Per-type scores
    factual_coherence: float
    temporal_coherence: float
    causal_coherence: float
    thematic_coherence: float
    
    recommended_actions: List[Resolution]


# -----------------------------------------------------------------------------
# Belief Network (Probabilistic Coherence)
# -----------------------------------------------------------------------------

class BeliefNetwork:
    """
    Probabilistic belief network for coherence reasoning.
    
    Each node has a belief state (probability distribution).
    Conflicts arise when connected beliefs are incompatible.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.beliefs: Dict[str, BeliefState] = {}
    
    def add_belief(
        self,
        node_id: str,
        prior: float,
        evidence_strength: float = 1.0,
    ) -> None:
        """Add a belief node"""
        self.graph.add_node(node_id)
        self.beliefs[node_id] = BeliefState(
            prior=prior,
            evidence_strength=evidence_strength,
        )
    
    def add_dependency(
        self,
        from_id: str,
        to_id: str,
        dependency_type: str,  # "supports", "contradicts"
        strength: float,
    ) -> None:
        """Add dependency between beliefs"""
        self.graph.add_edge(
            from_id, to_id,
            type=dependency_type,
            strength=strength,
        )
    
    def propagate_beliefs(self, iterations: int = BELIEF_PROPAGATION_ITERATIONS) -> Dict[str, float]:
        """
        Propagate beliefs through network.
        Returns final belief probabilities.
        """
        probabilities = {
            node_id: state.prior
            for node_id, state in self.beliefs.items()
        }
        
        for _ in range(iterations):
            new_probs = {}
            
            for node_id in self.graph.nodes():
                if node_id not in self.beliefs:
                    continue
                
                base_prob = self.beliefs[node_id].prior
                evidence = self.beliefs[node_id].evidence_strength
                
                # Aggregate influence from neighbors
                support = 0.0
                contradiction = 0.0
                
                for pred in self.graph.predecessors(node_id):
                    edge = self.graph.edges[pred, node_id]
                    pred_prob = probabilities.get(pred, 0.5)
                    
                    if edge['type'] == 'supports':
                        support += edge['strength'] * pred_prob
                    elif edge['type'] == 'contradicts':
                        contradiction += edge['strength'] * pred_prob
                
                # Update probability
                influence = support - contradiction
                updated = sigmoid(
                    math.log(base_prob / (1 - base_prob + 1e-9) + 1e-9) +
                    evidence * influence
                )
                
                new_probs[node_id] = updated
            
            probabilities = new_probs
        
        return probabilities


@dataclass
class BeliefState:
    """State of a belief in the network"""
    prior: float
    evidence_strength: float
    last_updated: float = field(default_factory=now_ts)


# -----------------------------------------------------------------------------
# Contradiction Detector
# -----------------------------------------------------------------------------

class ContradictionDetector:
    """
    Detect contradictions between memory elements.
    
    Uses multiple signals combined probabilistically:
    - Semantic opposition
    - Temporal impossibility
    - Causal inconsistency
    - Claim conflicts
    """
    
    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)
        
        # Opposition patterns (learned from data, not hard-coded)
        self.opposition_patterns: List[Tuple[np.ndarray, np.ndarray]] = []
    
    def detect_contradiction(
        self,
        a: MemoryElement,
        b: MemoryElement,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        """
        Detect if a and b contradict each other.
        
        Returns:
            (probability, explanation)
        """
        log_odds_contradiction = 0.0
        explanations = []
        
        # Get embeddings
        emb_a = self._get_embedding(a)
        emb_b = self._get_embedding(b)
        
        if emb_a is None or emb_b is None:
            return 0.0, "Cannot compare: missing embeddings"
        
        # 1. Semantic opposition detection
        opposition_score = self._semantic_opposition_score(emb_a, emb_b)
        if opposition_score > OPPOSITION_SCORE_THRESHOLD:
            log_odds_contradiction += opposition_score * 2
            explanations.append(f"Semantic opposition ({opposition_score:.2f})")
        
        # 2. Polarity inversion (for claims/themes)
        polarity_conflict = self._check_polarity_conflict(a, b)
        if polarity_conflict > 0:
            log_odds_contradiction += polarity_conflict
            explanations.append("Polarity conflict detected")
        
        # 3. Temporal impossibility
        temporal_conflict = self._check_temporal_conflict(a, b)
        if temporal_conflict > 0:
            log_odds_contradiction += temporal_conflict
            explanations.append("Temporal inconsistency")
        
        # 4. Actor contradiction (same actor, incompatible states)
        actor_conflict = self._check_actor_conflict(a, b)
        if actor_conflict > 0:
            log_odds_contradiction += actor_conflict
            explanations.append("Actor state conflict")
        
        # Convert to probability
        p_contradiction = sigmoid(log_odds_contradiction)
        explanation = "; ".join(explanations) if explanations else "No contradiction detected"
        
        return p_contradiction, explanation
    
    def _get_embedding(self, obj: MemoryElement) -> Optional[np.ndarray]:
        """Extract embedding from various object types."""
        return get_embedding_from_object(obj)
    
    def _semantic_opposition_score(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
    ) -> float:
        """
        Detect semantic opposition.
        
        Opposition is not just low similarity - it's active contradiction.
        We look for vectors that point in "opposite" directions in certain subspaces.
        """
        # Basic similarity
        sim = cosine_sim(emb_a, emb_b)
        
        # If highly similar, not contradicting
        if sim > HIGH_SIMILARITY_THRESHOLD:
            return 0.0
        
        # If anti-correlated (pointing opposite), stronger contradiction signal
        if sim < ANTI_CORRELATION_THRESHOLD:
            return abs(sim)
        
        # Check learned opposition patterns
        for pos_pattern, neg_pattern in self.opposition_patterns:
            proj_a = np.dot(emb_a, pos_pattern)
            proj_b = np.dot(emb_b, neg_pattern)
            
            if proj_a > 0.5 and proj_b > 0.5:
                return 0.7  # Matches opposition pattern
        
        # Moderate similarity but different clusters
        if 0.2 < sim < 0.5:
            return 0.3 * (0.5 - sim)
        
        return 0.0
    
    def _check_polarity_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        """Check for polarity conflicts in claims/themes"""
        # Extract polarity if available
        polarity_a = getattr(a, 'polarity', None) or getattr(a, 'emotion_valence', 0)
        polarity_b = getattr(b, 'polarity', None) or getattr(b, 'emotion_valence', 0)
        
        if isinstance(polarity_a, str):
            polarity_a = 1.0 if polarity_a == 'positive' else -1.0
        if isinstance(polarity_b, str):
            polarity_b = 1.0 if polarity_b == 'positive' else -1.0
        
        # Check subject/predicate match with opposite polarity
        subject_a = getattr(a, 'subject', '') or ''
        subject_b = getattr(b, 'subject', '') or ''
        predicate_a = getattr(a, 'predicate', '') or ''
        predicate_b = getattr(b, 'predicate', '') or ''
        
        if subject_a and subject_b:
            # Same subject, same predicate, opposite polarity
            if subject_a.lower() == subject_b.lower():
                if predicate_a.lower() == predicate_b.lower():
                    if polarity_a * polarity_b < 0:
                        return 1.5
        
        return 0.0
    
    def _check_temporal_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        """Check for temporal impossibilities"""
        ts_a = getattr(a, 'ts', None) or getattr(a, 'timestamp', None)
        ts_b = getattr(b, 'ts', None) or getattr(b, 'timestamp', None)
        
        if ts_a is None or ts_b is None:
            return 0.0
        
        # Check for claims about the same time period with conflicts
        # This would require more sophisticated temporal reasoning
        # For now, return 0
        return 0.0
    
    def _check_actor_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        """Check if same actor has incompatible states"""
        actors_a = set(getattr(a, 'actors', []) or [])
        actors_b = set(getattr(b, 'actors', []) or [])
        
        shared = actors_a & actors_b
        if not shared:
            return 0.0
        
        # If same actors and embeddings are very different, possible conflict
        emb_a = self._get_embedding(a)
        emb_b = self._get_embedding(b)
        
        if emb_a is not None and emb_b is not None:
            sim = cosine_sim(emb_a, emb_b)
            if sim < 0.3:
                return 0.5 * len(shared) * (0.3 - sim)
        
        return 0.0
    
    def learn_opposition_pattern(
        self,
        positive_examples: List[np.ndarray],
        negative_examples: List[np.ndarray],
    ) -> None:
        """Learn an opposition pattern from examples"""
        if not positive_examples or not negative_examples:
            return
        
        pos_mean = np.mean(positive_examples, axis=0)
        neg_mean = np.mean(negative_examples, axis=0)
        
        pos_pattern = l2_normalize(pos_mean)
        neg_pattern = l2_normalize(neg_mean)
        
        self.opposition_patterns.append((pos_pattern, neg_pattern))


# -----------------------------------------------------------------------------
# Coherence Scorer
# -----------------------------------------------------------------------------

class CoherenceScorer:
    """
    Compute overall coherence score for the memory system.
    
    Coherence is measured across multiple dimensions:
    - Factual consistency
    - Temporal consistency
    - Causal consistency
    - Thematic consistency
    """
    
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
        """Compute full coherence report"""
        conflicts = []
        
        # 1. Check factual coherence (plot-level contradictions)
        factual_conflicts, factual_score = self._check_factual_coherence(plots)
        conflicts.extend(factual_conflicts)
        
        # 2. Check temporal coherence
        temporal_conflicts, temporal_score = self._check_temporal_coherence(plots, stories)
        conflicts.extend(temporal_conflicts)
        
        # 3. Check causal coherence
        if causal_beliefs:
            causal_conflicts, causal_score = self._check_causal_coherence(causal_beliefs)
            conflicts.extend(causal_conflicts)
        else:
            causal_score = 1.0
        
        # 4. Check thematic coherence
        thematic_conflicts, thematic_score = self._check_thematic_coherence(themes)
        conflicts.extend(thematic_conflicts)
        
        # 5. Find unfinished stories
        unfinished = [
            s.id for s in stories.values()
            if s.status == 'developing' and
            (now_ts() - s.updated_ts) > UNFINISHED_STORY_HOURS * 3600
        ]
        
        # 6. Find orphan plots
        orphans = [
            p.id for p in plots.values()
            if p.story_id is None and p.status == 'active'
        ]
        
        # Compute overall score (weighted geometric mean)
        weights = [
            COHERENCE_WEIGHTS["factual"],
            COHERENCE_WEIGHTS["temporal"],
            COHERENCE_WEIGHTS["causal"],
            COHERENCE_WEIGHTS["thematic"],
        ]
        scores = [factual_score, temporal_score, causal_score, thematic_score]
        
        log_score = sum(w * math.log(s + 1e-9) for w, s in zip(weights, scores))
        overall_score = math.exp(log_score)
        
        # Generate recommended actions
        recommendations = self._generate_recommendations(conflicts)
        
        return CoherenceReport(
            overall_score=overall_score,
            conflicts=conflicts,
            unfinished_stories=unfinished,
            orphan_plots=orphans,
            factual_coherence=factual_score,
            temporal_coherence=temporal_score,
            causal_coherence=causal_score,
            thematic_coherence=thematic_score,
            recommended_actions=recommendations,
        )
    
    def _check_factual_coherence(
        self,
        plots: Dict[str, Plot],
    ) -> Tuple[List[Conflict], float]:
        """Check for factual contradictions between plots"""
        conflicts = []
        total_pairs = 0
        contradiction_sum = 0.0
        
        plot_list = list(plots.values())
        
        # Sample pairs for efficiency
        max_pairs = min(MAX_COHERENCE_PAIRS, len(plot_list) * (len(plot_list) - 1) // 2)
        
        for i, p1 in enumerate(plot_list):
            for p2 in plot_list[i+1:]:
                if total_pairs >= max_pairs:
                    break
                
                total_pairs += 1
                prob, explanation = self.detector.detect_contradiction(p1, p2)
                contradiction_sum += prob
                
                if prob > 0.6:  # Soft threshold for reporting
                    conflicts.append(Conflict(
                        type=ConflictType.FACTUAL,
                        node_a=p1.id,
                        node_b=p2.id,
                        severity=prob,
                        confidence=0.7,
                        description=explanation,
                    ))
        
        # Score = 1 - average contradiction probability
        avg_contradiction = contradiction_sum / max(total_pairs, 1)
        score = 1.0 - avg_contradiction
        
        return conflicts, score
    
    def _check_temporal_coherence(
        self,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
    ) -> Tuple[List[Conflict], float]:
        """Check temporal consistency within stories"""
        conflicts = []
        inconsistency_count = 0
        total_checks = 0
        
        for story in stories.values():
            if len(story.plot_ids) < 2:
                continue
            
            # Get plots in story
            story_plots = [plots[pid] for pid in story.plot_ids if pid in plots]
            story_plots.sort(key=lambda p: p.ts)
            
            # Check if sequence makes sense
            for i in range(len(story_plots) - 1):
                total_checks += 1
                
                p1, p2 = story_plots[i], story_plots[i+1]
                
                # Check for unreasonable time gaps
                gap = p2.ts - p1.ts
                if gap < 0:
                    # Time travel! Definite inconsistency
                    inconsistency_count += 1
                    conflicts.append(Conflict(
                        type=ConflictType.TEMPORAL,
                        node_a=p1.id,
                        node_b=p2.id,
                        severity=1.0,
                        confidence=1.0,
                        description="Temporal ordering violation",
                    ))
        
        score = 1.0 - (inconsistency_count / max(total_checks, 1))
        return conflicts, score
    
    def _check_causal_coherence(
        self,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
    ) -> Tuple[List[Conflict], float]:
        """Check causal graph for cycles and paradoxes"""
        conflicts = []
        
        # Build causal DAG
        dag = nx.DiGraph()
        for (src, tgt), belief in causal_beliefs.items():
            if belief.direction_belief() > 0.5:
                dag.add_edge(src, tgt, weight=belief.effective_causal_weight())
        
        # Check for cycles
        cycles = list(nx.simple_cycles(dag))
        
        for cycle in cycles[:10]:  # Limit reported cycles
            conflicts.append(Conflict(
                type=ConflictType.CAUSAL,
                node_a=cycle[0],
                node_b=cycle[-1],
                severity=0.8,
                confidence=1.0,
                description=f"Causal cycle detected: {' → '.join(cycle[:5])}...",
                evidence=cycle,
            ))
        
        # Score based on cycle count
        cycle_penalty = len(cycles) * 0.1
        score = max(0.0, 1.0 - cycle_penalty)
        
        return conflicts, score
    
    def _check_thematic_coherence(
        self,
        themes: Dict[str, Theme],
    ) -> Tuple[List[Conflict], float]:
        """Check for conflicting themes"""
        conflicts = []
        conflict_count = 0
        total_pairs = 0
        
        theme_list = list(themes.values())
        
        for i, t1 in enumerate(theme_list):
            for t2 in theme_list[i+1:]:
                total_pairs += 1
                
                prob, explanation = self.detector.detect_contradiction(t1, t2)
                
                if prob > 0.5:
                    conflict_count += 1
                    
                    # Check if they share supporting stories
                    shared_stories = set(t1.story_ids) & set(t2.story_ids)
                    
                    if shared_stories:
                        severity = prob * 1.2  # Higher severity if shared evidence
                    else:
                        severity = prob * 0.8
                    
                    conflicts.append(Conflict(
                        type=ConflictType.THEMATIC,
                        node_a=t1.id,
                        node_b=t2.id,
                        severity=min(severity, 1.0),
                        confidence=0.6,
                        description=f"Themes may conflict: {explanation}",
                        evidence=list(shared_stories),
                    ))
        
        score = 1.0 - (conflict_count / max(total_pairs, 1))
        return conflicts, score
    
    def _generate_recommendations(
        self,
        conflicts: List[Conflict],
    ) -> List[Resolution]:
        """Generate resolution recommendations for conflicts"""
        recommendations = []
        
        # Sort by severity
        sorted_conflicts = sorted(conflicts, key=lambda c: c.severity, reverse=True)
        
        for conflict in sorted_conflicts[:5]:  # Top 5 recommendations
            if conflict.type == ConflictType.FACTUAL:
                # Recommend conditioning or weakening
                recommendations.append(Resolution(
                    strategy="condition",
                    target_node=conflict.node_a,
                    action_description=f"Add condition to {conflict.node_a} to resolve conflict with {conflict.node_b}",
                    expected_coherence_gain=conflict.severity * 0.7,
                    cost=0.1,
                    condition="Different contexts may apply",
                ))
            
            elif conflict.type == ConflictType.CAUSAL:
                # Recommend removing weakest edge in cycle
                recommendations.append(Resolution(
                    strategy="remove",
                    target_node=conflict.node_a,
                    action_description=f"Remove weakest causal link in cycle involving {conflict.node_a}",
                    expected_coherence_gain=conflict.severity * 0.8,
                    cost=0.2,
                ))
            
            elif conflict.type == ConflictType.THEMATIC:
                # Recommend merging into conditional theme
                recommendations.append(Resolution(
                    strategy="merge",
                    target_node=conflict.node_a,
                    action_description=f"Merge {conflict.node_a} and {conflict.node_b} into conditional theme",
                    expected_coherence_gain=conflict.severity * 0.6,
                    cost=0.3,
                ))
        
        return recommendations


# -----------------------------------------------------------------------------
# Conflict Resolver
# -----------------------------------------------------------------------------

class ConflictResolver:
    """
    Automatically resolve coherence conflicts.
    
    Strategies:
    - Weaken: Reduce confidence in conflicting element
    - Condition: Add conditions to make both true
    - Merge: Combine into more general element
    - Remove: Archive low-confidence element
    """
    
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
        """
        Attempt to resolve a conflict.
        Returns True if resolution was applied.
        """
        if not conflict.resolutions:
            return False
        
        # Pick best resolution (lowest cost that meets gain threshold)
        best = min(
            conflict.resolutions,
            key=lambda r: r.cost - r.expected_coherence_gain
        )
        
        if best.expected_coherence_gain < 0.1:
            return False  # Not worth it
        
        return self._apply_resolution(best, plots, stories, themes)
    
    def _apply_resolution(
        self,
        resolution: Resolution,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> bool:
        """Apply a specific resolution strategy"""
        
        if resolution.strategy == "weaken":
            # Reduce confidence/evidence
            if resolution.target_node in themes:
                theme = themes[resolution.target_node]
                theme.b += 1.0  # Increase negative evidence
                return True
        
        elif resolution.strategy == "condition":
            # Add condition annotation
            if resolution.target_node in themes:
                theme = themes[resolution.target_node]
                theme.description += f" (Condition: {resolution.condition})"
                return True
        
        elif resolution.strategy == "remove":
            # Mark for archival (soft delete)
            if resolution.target_node in plots:
                plots[resolution.target_node].status = "archived"
                return True
        
        return False


# -----------------------------------------------------------------------------
# Coherence Guardian (Main Interface)
# -----------------------------------------------------------------------------

class CoherenceGuardian:
    """
    Main interface for coherence maintenance with FUNCTIONAL CONTRADICTION MANAGEMENT.
    
    Key philosophy change:
    - Not all contradictions need resolution
    - Some contradictions provide flexibility (adaptive)
    - Some contradictions indicate growth (developmental)
    - Only action-blocking or identity-threatening contradictions must be resolved
    
    Integrates TensionManager for intelligent contradiction handling.
    """
    
    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.detector = ContradictionDetector(metric, seed)
        self.scorer = CoherenceScorer(metric, self.detector, seed)
        self.resolver = ConflictResolver(metric, seed)
        self.belief_network = BeliefNetwork()
        
        # TensionManager for functional contradiction management
        self.tension_manager = TensionManager(seed=seed)
    
    def full_check(
        self,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> CoherenceReport:
        """Run full coherence check"""
        return self.scorer.compute_coherence(
            graph, plots, stories, themes, causal_beliefs
        )
    
    def full_check_with_tension_analysis(
        self,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> Tuple[CoherenceReport, Dict[str, Any]]:
        """
        Run full coherence check with tension analysis.
        
        Returns:
            (CoherenceReport, tension_analysis)
            
        The tension_analysis includes:
        - conflicts_to_resolve: Conflicts that must be resolved
        - conflicts_to_preserve: Conflicts that provide flexibility
        - conflicts_to_accept: Conflicts that indicate growth
        """
        report = self.scorer.compute_coherence(
            graph, plots, stories, themes, causal_beliefs
        )
        
        # Analyze each conflict through the tension lens
        conflicts_to_resolve = []
        conflicts_to_preserve = []
        conflicts_to_accept = []
        conflicts_to_defer = []
        
        for conflict in report.conflicts:
            # Convert Conflict to Tension for analysis
            tension = self._conflict_to_tension(conflict, plots, stories, themes)
            if tension is None:
                continue
            
            # Classify and decide what to do
            tension_type = self.tension_manager.classify_tension(tension)
            
            if tension_type in [TensionType.ACTION_BLOCKING, TensionType.IDENTITY_THREATENING]:
                conflicts_to_resolve.append({
                    "conflict": conflict,
                    "tension": tension,
                    "reason": f"必须解决：{tension_type.value}"
                })
            elif tension_type == TensionType.ADAPTIVE:
                conflicts_to_preserve.append({
                    "conflict": conflict,
                    "tension": tension,
                    "reason": "保留：提供灵活性的适应性矛盾"
                })
            elif tension_type == TensionType.DEVELOPMENTAL:
                conflicts_to_accept.append({
                    "conflict": conflict,
                    "tension": tension,
                    "reason": "接受：成长的标志"
                })
            else:
                conflicts_to_defer.append({
                    "conflict": conflict,
                    "tension": tension,
                    "reason": "需要更多信息"
                })
        
        tension_analysis = {
            "conflicts_to_resolve": conflicts_to_resolve,
            "conflicts_to_preserve": conflicts_to_preserve,
            "conflicts_to_accept": conflicts_to_accept,
            "conflicts_to_defer": conflicts_to_defer,
            "summary": {
                "total": len(report.conflicts),
                "to_resolve": len(conflicts_to_resolve),
                "to_preserve": len(conflicts_to_preserve),
                "to_accept": len(conflicts_to_accept),
                "to_defer": len(conflicts_to_defer),
            }
        }
        
        return report, tension_analysis
    
    def _conflict_to_tension(
        self,
        conflict: Conflict,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> Optional[Tension]:
        """Convert a Conflict to a Tension for analysis."""
        # Get the elements involved
        element_a = self._get_element(conflict.node_a, plots, stories, themes)
        element_b = self._get_element(conflict.node_b, plots, stories, themes)
        
        if element_a is None or element_b is None:
            return None
        
        # Get embeddings
        emb_a = self._get_embedding(element_a)
        emb_b = self._get_embedding(element_b)
        
        # Use TensionManager to detect and create tension
        tension = self.tension_manager.detect_tension(
            {"id": conflict.node_a, "type": conflict.type.value, "text": conflict.description},
            {"id": conflict.node_b, "type": conflict.type.value, "text": ""},
            emb_a, emb_b
        )
        
        if tension is None:
            # Create tension from conflict
            import uuid
            tension = Tension(
                id=str(uuid.uuid4()),
                element_a_id=conflict.node_a,
                element_a_type=conflict.type.value,
                element_b_id=conflict.node_b,
                element_b_type=conflict.type.value,
                description=conflict.description,
                severity=conflict.severity,
            )
            self.tension_manager.tensions[tension.id] = tension
        
        return tension
    
    def _get_element(
        self,
        node_id: str,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> Optional[MemoryElement]:
        """Get element by ID from any collection."""
        if node_id in plots:
            return plots[node_id]
        if node_id in stories:
            return stories[node_id]
        if node_id in themes:
            return themes[node_id]
        return None
    
    def _get_embedding(self, element: MemoryElement) -> Optional[np.ndarray]:
        """Get embedding from element."""
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
        """
        Automatically resolve top conflicts.
        Returns number of successful resolutions.
        """
        resolved = 0
        
        # Sort conflicts by severity
        sorted_conflicts = sorted(
            report.conflicts,
            key=lambda c: c.severity * c.confidence,
            reverse=True
        )
        
        for conflict in sorted_conflicts[:max_resolutions]:
            # Generate resolutions if not present
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
        """
        Smart conflict resolution using TensionManager.
        
        Only resolves conflicts that NEED resolution.
        Preserves adaptive and developmental tensions.
        
        Returns summary of actions taken.
        """
        _, tension_analysis = self.full_check_with_tension_analysis(
            graph, plots, stories, themes
        )
        
        actions_taken = {
            "resolved": [],
            "preserved": [],
            "accepted": [],
        }
        
        # Only resolve action-blocking or identity-threatening
        for item in tension_analysis["conflicts_to_resolve"][:max_resolutions]:
            conflict = item["conflict"]
            tension = item["tension"]
            
            # Generate and apply resolution
            if not conflict.resolutions:
                conflict.resolutions = self._generate_resolutions(conflict)
            
            if self.resolver.resolve(conflict, graph, plots, stories, themes):
                resolution = self.tension_manager.handle_tension(tension)
                actions_taken["resolved"].append({
                    "conflict_id": f"{conflict.node_a}-{conflict.node_b}",
                    "description": conflict.description,
                    "action": resolution.action,
                    "rationale": resolution.rationale,
                })
        
        # Mark adaptive tensions as preserved
        for item in tension_analysis["conflicts_to_preserve"]:
            tension = item["tension"]
            resolution = self.tension_manager.handle_tension(tension)
            actions_taken["preserved"].append({
                "conflict_description": item["conflict"].description,
                "reason": item["reason"],
                "action": resolution.action,
            })
        
        # Mark developmental tensions as accepted
        for item in tension_analysis["conflicts_to_accept"]:
            tension = item["tension"]
            resolution = self.tension_manager.handle_tension(tension)
            actions_taken["accepted"].append({
                "conflict_description": item["conflict"].description,
                "reason": item["reason"],
                "action": resolution.action,
            })
        
        return actions_taken
    
    def _generate_resolutions(self, conflict: Conflict) -> List[Resolution]:
        """Generate resolution options for a conflict"""
        resolutions = []
        
        if conflict.type == ConflictType.FACTUAL:
            resolutions.append(Resolution(
                strategy="weaken",
                target_node=conflict.node_b,  # Weaken the newer one
                action_description=f"Reduce confidence in {conflict.node_b}",
                expected_coherence_gain=conflict.severity * 0.5,
                cost=0.2,
            ))
            resolutions.append(Resolution(
                strategy="condition",
                target_node=conflict.node_a,
                action_description=f"Add context condition to {conflict.node_a}",
                expected_coherence_gain=conflict.severity * 0.7,
                cost=0.1,
                condition="May depend on context",
            ))
        
        elif conflict.type == ConflictType.THEMATIC:
            resolutions.append(Resolution(
                strategy="merge",
                target_node=conflict.node_a,
                action_description=f"Merge conflicting themes",
                expected_coherence_gain=conflict.severity * 0.6,
                cost=0.3,
            ))
        
        elif conflict.type == ConflictType.CAUSAL:
            resolutions.append(Resolution(
                strategy="remove",
                target_node=conflict.node_b,
                action_description=f"Remove edge to break cycle",
                expected_coherence_gain=conflict.severity * 0.8,
                cost=0.15,
            ))
        
        return resolutions
    
    def update_belief_network(
        self,
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> Dict[str, float]:
        """
        Update belief network with current themes and return propagated beliefs.
        """
        self.belief_network = BeliefNetwork()
        
        # Add theme beliefs
        for theme in themes.values():
            self.belief_network.add_belief(
                theme.id,
                prior=theme.confidence(),
                evidence_strength=len(theme.story_ids) * 0.1,
            )
        
        # Add causal dependencies
        if causal_beliefs:
            for (src, tgt), belief in causal_beliefs.items():
                if src in themes and tgt in themes:
                    dep_type = "supports" if belief.effective_causal_weight() > 0.5 else "contradicts"
                    self.belief_network.add_dependency(
                        src, tgt,
                        dependency_type=dep_type,
                        strength=belief.effective_causal_weight(),
                    )
        
        return self.belief_network.propagate_beliefs()
    
    def get_tension_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked tensions."""
        return self.tension_manager.get_tension_summary()
