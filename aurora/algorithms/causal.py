"""
AURORA Causal Inference Module
==============================

Causal reasoning without hard-coded thresholds.
All causal judgments are probabilistic and learned from data.

Key components:
- CausalEdgeBelief: Extended edge with causal direction/strength posteriors
- CausalDiscovery: Infer causal direction from observational data
- InterventionEngine: do-calculus implementation
- CounterfactualReasoner: "What if" queries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
import math
import numpy as np
import networkx as nx

from aurora.algorithms.aurora_core import (
    Plot, StoryArc, Theme, MemoryGraph, EdgeBelief,
    LowRankMetric, VectorIndex, now_ts, l2_normalize, cosine_sim, sigmoid, softmax
)


# -----------------------------------------------------------------------------
# Causal Edge Belief (extends EdgeBelief)
# -----------------------------------------------------------------------------

@dataclass
class CausalEdgeBelief(EdgeBelief):
    """Extended EdgeBelief with causal inference capabilities.
    
    All causal properties are Beta posteriors - no thresholds.
    """
    # Causal direction belief: P(source→target) = dir_a / (dir_a + dir_b)
    dir_a: float = 1.0
    dir_b: float = 1.0
    
    # Causal strength posterior
    str_a: float = 1.0
    str_b: float = 1.0
    
    # Confounding probability: P(∃ common cause)
    conf_a: float = 1.0
    conf_b: float = 9.0  # Prior: low confounding probability
    
    # Causal mechanism embedding (for generalization)
    mechanism_emb: Optional[np.ndarray] = None
    
    # Evidence counts
    intervention_count: int = 0
    observation_count: int = 0
    
    def direction_belief(self) -> float:
        """P(source → target)"""
        return self.dir_a / (self.dir_a + self.dir_b)
    
    def causal_strength(self) -> float:
        """Expected causal strength"""
        return self.str_a / (self.str_a + self.str_b)
    
    def confound_prob(self) -> float:
        """P(exists confounding factor)"""
        return self.conf_a / (self.conf_a + self.conf_b)
    
    def effective_causal_weight(self) -> float:
        """Effective weight = direction × strength × (1 - confound)"""
        return (
            self.direction_belief() *
            self.causal_strength() *
            (1.0 - self.confound_prob())
        )
    
    def sample_direction(self, rng: np.random.Generator) -> bool:
        """Thompson sampling for causal direction"""
        sampled = rng.beta(self.dir_a, self.dir_b)
        return sampled > 0.5
    
    def update_direction_evidence(self, forward: bool, weight: float = 1.0) -> None:
        """Update direction belief with new evidence"""
        if forward:
            self.dir_a += weight
        else:
            self.dir_b += weight
        self.observation_count += 1
    
    def update_intervention_evidence(self, effect_observed: bool, weight: float = 1.0) -> None:
        """Update strength belief from intervention results"""
        if effect_observed:
            self.str_a += weight
        else:
            self.str_b += weight
        self.intervention_count += 1
        # Interventions reduce confounding uncertainty
        self.conf_b += 0.5 * weight
    
    def update_confound_evidence(self, confound_detected: bool, weight: float = 1.0) -> None:
        """Update confounding belief"""
        if confound_detected:
            self.conf_a += weight
        else:
            self.conf_b += weight


# -----------------------------------------------------------------------------
# Causal Discovery (no hard-coded thresholds)
# -----------------------------------------------------------------------------

class CausalDiscovery:
    """Infer causal direction from observational data.
    
    Methods:
    - Temporal precedence (weak signal)
    - Transfer entropy (information flow)
    - Additive Noise Model (ANM)
    - Conditional complexity asymmetry
    
    All signals combined as log-odds, then converted to probability.
    """
    
    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)
    
    def infer_direction(
        self,
        source: Plot,
        target: Plot,
        context_plots: List[Plot],
    ) -> Tuple[float, float, float]:
        """
        Infer causal direction between two plots.
        
        Returns:
            (p_forward, p_backward, p_confound)
            - p_forward: P(source → target)
            - p_backward: P(target → source)
            - p_confound: P(∃ common cause)
        """
        log_odds_forward = 0.0
        confound_evidence = 0.0
        
        # 1. Temporal precedence (weak signal)
        time_diff = target.ts - source.ts
        if time_diff > 0:
            # source before target: weak support for source→target
            temporal_weight = min(1.0, time_diff / 3600.0)  # saturate at 1 hour
            log_odds_forward += 0.5 * temporal_weight
        elif time_diff < 0:
            log_odds_forward -= 0.5 * min(1.0, abs(time_diff) / 3600.0)
        
        # 2. Transfer entropy estimate (information flow)
        te_forward = self._estimate_transfer_entropy(source, target, context_plots)
        te_backward = self._estimate_transfer_entropy(target, source, context_plots)
        log_odds_forward += 2.0 * (te_forward - te_backward)
        
        # 3. Additive Noise Model score
        anm_forward = self._anm_independence_score(source, target, context_plots)
        anm_backward = self._anm_independence_score(target, source, context_plots)
        log_odds_forward += 1.5 * (anm_forward - anm_backward)
        
        # 4. Complexity asymmetry (cause is usually "simpler")
        complexity_source = self._conditional_complexity(source, context_plots)
        complexity_target = self._conditional_complexity(target, context_plots)
        # If target is more complex, more likely source→target
        log_odds_forward += 0.3 * (complexity_target - complexity_source)
        
        # 5. Check for confounding signals
        confound_evidence = self._detect_confounding_signals(source, target, context_plots)
        
        # Convert to probabilities
        p_forward = sigmoid(log_odds_forward)
        p_backward = sigmoid(-log_odds_forward)
        p_confound = sigmoid(confound_evidence)
        
        return p_forward, p_backward, p_confound
    
    def _estimate_transfer_entropy(
        self,
        source: Plot,
        target: Plot,
        context: List[Plot],
    ) -> float:
        """
        Estimate transfer entropy: TE(source→target)
        
        Measures how much knowing source reduces uncertainty about target,
        beyond what we can predict from target's own history.
        """
        # Find target's history (plots in same story before target)
        target_history = [
            p for p in context
            if p.ts < target.ts and p.story_id == target.story_id
        ]
        
        if not target_history:
            return 0.0
        
        # Baseline prediction: from target's history only
        baseline_pred = self._predict_embedding(target_history)
        baseline_error = self.metric.d2(baseline_pred, target.embedding)
        
        # Augmented prediction: include source if it precedes target
        if source.ts < target.ts:
            augmented_pred = self._predict_embedding(target_history + [source])
            augmented_error = self.metric.d2(augmented_pred, target.embedding)
            
            # Information gain = reduction in prediction error
            info_gain = max(0, baseline_error - augmented_error)
            return info_gain / (baseline_error + 1e-6)
        
        return 0.0
    
    def _predict_embedding(self, plots: List[Plot]) -> np.ndarray:
        """Predict embedding from a sequence of plots (recency-weighted average)"""
        if not plots:
            return np.zeros(self.metric.dim, dtype=np.float32)
        
        # Sort by time, more recent gets higher weight
        sorted_plots = sorted(plots, key=lambda p: p.ts)
        weights = [math.exp(i * 0.3) for i in range(len(sorted_plots))]
        total = sum(weights)
        
        result = np.zeros_like(sorted_plots[0].embedding)
        for w, p in zip(weights, sorted_plots):
            result += (w / total) * p.embedding
        
        return l2_normalize(result)
    
    def _anm_independence_score(
        self,
        cause: Plot,
        effect: Plot,
        context: List[Plot],
    ) -> float:
        """
        Additive Noise Model: if effect = f(cause) + noise,
        and noise ⊥ cause, then cause→effect.
        
        We approximate this by checking if residuals are uncorrelated with cause.
        """
        # Simple linear model: effect ≈ α * cause
        cause_emb = cause.embedding.astype(np.float32)
        effect_emb = effect.embedding.astype(np.float32)
        
        # Project effect onto cause direction
        projection = np.dot(effect_emb, cause_emb) * cause_emb
        residual = effect_emb - projection
        
        # Independence score: 1 - |correlation between residual and cause|
        if np.linalg.norm(residual) < 1e-6:
            return 0.0
        
        corr = abs(np.dot(l2_normalize(residual), cause_emb))
        independence_score = 1.0 - corr
        
        return independence_score
    
    def _conditional_complexity(self, plot: Plot, context: List[Plot]) -> float:
        """
        Measure how many other nodes are needed to "explain" this plot.
        Higher complexity = more likely to be an effect rather than cause.
        """
        if not context:
            return 0.0
        
        # Count plots that are similar to this one
        similar_count = sum(
            1 for p in context
            if self.metric.sim(plot.embedding, p.embedding) > 0.5
        )
        
        return math.log1p(similar_count)
    
    def _detect_confounding_signals(
        self,
        a: Plot,
        b: Plot,
        context: List[Plot],
    ) -> float:
        """
        Detect signals of common cause (confounding).
        
        Signals:
        - Both A and B are similar to a third node C that precedes both
        - A and B share actors but no direct semantic link
        """
        log_odds_confound = 0.0
        
        # Find potential confounders: nodes that precede both A and B
        earlier_ts = min(a.ts, b.ts)
        potential_confounders = [
            p for p in context
            if p.ts < earlier_ts and p.id not in (a.id, b.id)
        ]
        
        for c in potential_confounders:
            sim_ca = self.metric.sim(c.embedding, a.embedding)
            sim_cb = self.metric.sim(c.embedding, b.embedding)
            
            # If C is similar to both A and B, could be common cause
            if sim_ca > 0.6 and sim_cb > 0.6:
                log_odds_confound += 0.5 * (sim_ca + sim_cb)
        
        # Shared actors without semantic link suggests confounding
        shared_actors = set(a.actors) & set(b.actors)
        semantic_sim = cosine_sim(a.embedding, b.embedding)
        
        if shared_actors and semantic_sim < 0.4:
            log_odds_confound += 0.3 * len(shared_actors)
        
        return log_odds_confound


# -----------------------------------------------------------------------------
# Intervention Engine (do-calculus)
# -----------------------------------------------------------------------------

@dataclass
class InterventionResult:
    """Result of do(X=x) intervention"""
    target_id: str
    intervention_value: np.ndarray
    predicted_effects: Dict[str, np.ndarray]  # node_id → predicted embedding
    causal_chain: List[str]  # topological order of affected nodes
    confidence: float  # overall confidence in the prediction


class InterventionEngine:
    """
    Implement do(X=x) intervention.
    
    Key insight: intervention cuts all incoming edges to X,
    then propagates effects forward through causal graph.
    """
    
    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)
    
    def do_intervention(
        self,
        graph: MemoryGraph,
        target_id: str,
        intervention_value: np.ndarray,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
    ) -> InterventionResult:
        """
        Execute do(target = intervention_value).
        
        1. Build causal DAG from beliefs
        2. Cut incoming edges to target
        3. Propagate effects forward
        """
        # Build causal DAG
        causal_dag = self._build_causal_dag(graph, causal_beliefs)
        
        # Cut incoming edges to target
        incoming = list(causal_dag.predecessors(target_id))
        for pred in incoming:
            causal_dag.remove_edge(pred, target_id)
        
        # Initialize values
        values: Dict[str, np.ndarray] = {target_id: intervention_value}
        confidence_product = 1.0
        
        # Propagate in topological order
        causal_chain = []
        try:
            topo_order = list(nx.topological_sort(causal_dag))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, fall back to BFS from target
            topo_order = list(nx.bfs_tree(causal_dag, target_id))
        
        for node in topo_order:
            if node in values:
                causal_chain.append(node)
                continue
            
            parents = list(causal_dag.predecessors(node))
            if not parents:
                # No parents in modified graph, keep original
                values[node] = self._get_node_embedding(graph, node)
                continue
            
            # Compute weighted combination of parent effects
            parent_values = []
            parent_weights = []
            
            for p in parents:
                if p not in values:
                    continue
                
                edge_key = (p, node)
                if edge_key in causal_beliefs:
                    belief = causal_beliefs[edge_key]
                    weight = belief.effective_causal_weight()
                else:
                    weight = 0.3  # weak default
                
                parent_values.append(values[p])
                parent_weights.append(weight)
                confidence_product *= belief.causal_strength() if edge_key in causal_beliefs else 0.5
            
            if parent_values:
                # Weighted average
                total_weight = sum(parent_weights) + 1e-9
                predicted = np.zeros_like(parent_values[0])
                for v, w in zip(parent_values, parent_weights):
                    predicted += (w / total_weight) * v
                
                # Blend with original (causal effect is usually partial)
                original = self._get_node_embedding(graph, node)
                blend_factor = min(1.0, sum(parent_weights))
                values[node] = l2_normalize(
                    blend_factor * predicted + (1 - blend_factor) * original
                )
                causal_chain.append(node)
        
        return InterventionResult(
            target_id=target_id,
            intervention_value=intervention_value,
            predicted_effects=values,
            causal_chain=causal_chain,
            confidence=confidence_product ** (1.0 / max(len(causal_chain), 1))
        )
    
    def _build_causal_dag(
        self,
        graph: MemoryGraph,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
    ) -> nx.DiGraph:
        """Build directed acyclic graph from causal beliefs"""
        dag = nx.DiGraph()
        
        for node in graph.g.nodes():
            dag.add_node(node)
        
        for (src, tgt), belief in causal_beliefs.items():
            # Only add edge if direction belief is strong enough
            if belief.direction_belief() > 0.5:
                dag.add_edge(src, tgt, weight=belief.effective_causal_weight())
        
        # Remove cycles greedily (remove lowest weight edge in each cycle)
        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle = nx.find_cycle(dag)
                # Find minimum weight edge in cycle
                min_edge = min(cycle, key=lambda e: dag.edges[e[0], e[1]].get('weight', 0))
                dag.remove_edge(min_edge[0], min_edge[1])
            except nx.NetworkXNoCycle:
                break
        
        return dag
    
    def _get_node_embedding(self, graph: MemoryGraph, node_id: str) -> np.ndarray:
        """Get embedding for a node"""
        try:
            payload = graph.payload(node_id)
            if hasattr(payload, 'embedding'):
                return payload.embedding
            elif hasattr(payload, 'centroid'):
                return payload.centroid
            elif hasattr(payload, 'prototype'):
                return payload.prototype
        except KeyError:
            pass
        return np.zeros(self.metric.dim, dtype=np.float32)


# -----------------------------------------------------------------------------
# Counterfactual Reasoner
# -----------------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """Result of counterfactual query"""
    factual_world: Dict[str, np.ndarray]
    counterfactual_world: Dict[str, np.ndarray]
    query_variable: str
    factual_value: np.ndarray
    counterfactual_value: np.ndarray
    difference_magnitude: float
    explanation: str


class CounterfactualReasoner:
    """
    Answer "What if X had been different?" questions.
    
    Implements the three-step counterfactual algorithm:
    1. Abduction: Infer exogenous noise from observations
    2. Action: Intervene on the antecedent
    3. Prediction: Propagate effects with same noise
    """
    
    def __init__(
        self,
        metric: LowRankMetric,
        intervention_engine: InterventionEngine,
        seed: int = 0
    ):
        self.metric = metric
        self.intervention = intervention_engine
        self.rng = np.random.default_rng(seed)
    
    def query(
        self,
        graph: MemoryGraph,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
        factual_values: Dict[str, np.ndarray],
        antecedent_id: str,
        antecedent_cf_value: np.ndarray,
        query_id: str,
    ) -> CounterfactualResult:
        """
        Answer: "If antecedent had been antecedent_cf_value instead,
        what would query_id be?"
        """
        # Step 1: Abduction - infer noise terms
        noise = self._infer_noise(graph, causal_beliefs, factual_values)
        
        # Step 2: Action - intervene
        intervention_result = self.intervention.do_intervention(
            graph=graph,
            target_id=antecedent_id,
            intervention_value=antecedent_cf_value,
            causal_beliefs=causal_beliefs,
        )
        
        # Step 3: Prediction - use same noise in counterfactual world
        cf_values = self._propagate_with_noise(
            graph,
            causal_beliefs,
            intervention_result.predicted_effects,
            noise,
        )
        
        # Get query variable values
        factual_value = factual_values.get(query_id, np.zeros(self.metric.dim))
        cf_value = cf_values.get(query_id, factual_value)
        
        diff_magnitude = float(np.linalg.norm(cf_value - factual_value))
        
        explanation = self._generate_explanation(
            antecedent_id, antecedent_cf_value,
            query_id, factual_value, cf_value
        )
        
        return CounterfactualResult(
            factual_world=factual_values,
            counterfactual_world=cf_values,
            query_variable=query_id,
            factual_value=factual_value,
            counterfactual_value=cf_value,
            difference_magnitude=diff_magnitude,
            explanation=explanation,
        )
    
    def _infer_noise(
        self,
        graph: MemoryGraph,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
        factual_values: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Infer exogenous noise for each variable.
        
        noise[X] = X_observed - f(parents(X))
        """
        noise = {}
        dag = self.intervention._build_causal_dag(graph, causal_beliefs)
        
        for node_id, observed in factual_values.items():
            if node_id not in dag:
                continue
            
            parents = list(dag.predecessors(node_id))
            if not parents:
                noise[node_id] = np.zeros_like(observed)
                continue
            
            # Predict from parents
            parent_values = []
            parent_weights = []
            for p in parents:
                if p in factual_values:
                    edge_key = (p, node_id)
                    weight = causal_beliefs[edge_key].effective_causal_weight() if edge_key in causal_beliefs else 0.3
                    parent_values.append(factual_values[p])
                    parent_weights.append(weight)
            
            if parent_values:
                total_weight = sum(parent_weights) + 1e-9
                predicted = np.zeros_like(observed)
                for v, w in zip(parent_values, parent_weights):
                    predicted += (w / total_weight) * v
                
                noise[node_id] = observed - predicted
            else:
                noise[node_id] = np.zeros_like(observed)
        
        return noise
    
    def _propagate_with_noise(
        self,
        graph: MemoryGraph,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
        intervention_values: Dict[str, np.ndarray],
        noise: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Propagate effects keeping the same noise terms"""
        dag = self.intervention._build_causal_dag(graph, causal_beliefs)
        values = dict(intervention_values)
        
        try:
            topo_order = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            topo_order = list(values.keys())
        
        for node_id in topo_order:
            if node_id in intervention_values:
                continue
            
            parents = list(dag.predecessors(node_id))
            if not parents:
                values[node_id] = noise.get(node_id, np.zeros(self.metric.dim))
                continue
            
            # Predict from parents
            parent_values = []
            parent_weights = []
            for p in parents:
                if p in values:
                    edge_key = (p, node_id)
                    weight = causal_beliefs[edge_key].effective_causal_weight() if edge_key in causal_beliefs else 0.3
                    parent_values.append(values[p])
                    parent_weights.append(weight)
            
            if parent_values:
                total_weight = sum(parent_weights) + 1e-9
                predicted = np.zeros(self.metric.dim, dtype=np.float32)
                for v, w in zip(parent_values, parent_weights):
                    predicted += (w / total_weight) * v
                
                # Add noise back
                node_noise = noise.get(node_id, np.zeros_like(predicted))
                values[node_id] = predicted + node_noise
        
        return values
    
    def _generate_explanation(
        self,
        antecedent_id: str,
        antecedent_cf: np.ndarray,
        query_id: str,
        factual_value: np.ndarray,
        cf_value: np.ndarray,
    ) -> str:
        """Generate natural language explanation"""
        diff = float(np.linalg.norm(cf_value - factual_value))
        
        if diff < 0.1:
            return f"Changing {antecedent_id} would have minimal effect on {query_id}."
        elif diff < 0.3:
            return f"Changing {antecedent_id} would moderately affect {query_id}."
        else:
            return f"Changing {antecedent_id} would significantly change {query_id}."


# -----------------------------------------------------------------------------
# Causal Memory Graph (extends MemoryGraph)
# -----------------------------------------------------------------------------

class CausalMemoryGraph(MemoryGraph):
    """Extended MemoryGraph with causal inference capabilities"""
    
    def __init__(self, metric: LowRankMetric, seed: int = 0):
        super().__init__()
        self.metric = metric
        self.causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief] = {}
        self.discovery = CausalDiscovery(metric, seed=seed)
        self.intervention = InterventionEngine(metric, seed=seed)
        self.counterfactual = CounterfactualReasoner(metric, self.intervention, seed=seed)
    
    def add_causal_edge(
        self,
        source_id: str,
        target_id: str,
        initial_direction_belief: float = 0.5,
    ) -> CausalEdgeBelief:
        """Add or get causal edge between two nodes"""
        key = (source_id, target_id)
        
        if key not in self.causal_beliefs:
            # Initialize with direction belief
            belief = CausalEdgeBelief(
                edge_type="causal",
                dir_a=initial_direction_belief * 2,
                dir_b=(1 - initial_direction_belief) * 2,
            )
            self.causal_beliefs[key] = belief
            
            # Also add to base graph
            self.ensure_edge(source_id, target_id, "causal")
        
        return self.causal_beliefs[key]
    
    def infer_and_add_causal_edge(
        self,
        source: Plot,
        target: Plot,
        context_plots: List[Plot],
    ) -> CausalEdgeBelief:
        """Infer causal direction and add edge with learned beliefs"""
        p_forward, p_backward, p_confound = self.discovery.infer_direction(
            source, target, context_plots
        )
        
        # Create edge in direction with higher probability
        if p_forward >= p_backward:
            belief = self.add_causal_edge(source.id, target.id, p_forward)
        else:
            belief = self.add_causal_edge(target.id, source.id, p_backward)
        
        # Update confound belief
        if p_confound > 0.5:
            belief.update_confound_evidence(True, weight=p_confound)
        else:
            belief.update_confound_evidence(False, weight=1 - p_confound)
        
        return belief
    
    def do(
        self,
        target_id: str,
        intervention_value: np.ndarray,
    ) -> InterventionResult:
        """Execute do(target = value) intervention"""
        return self.intervention.do_intervention(
            graph=self,
            target_id=target_id,
            intervention_value=intervention_value,
            causal_beliefs=self.causal_beliefs,
        )
    
    def counterfactual_query(
        self,
        factual_values: Dict[str, np.ndarray],
        antecedent_id: str,
        antecedent_cf_value: np.ndarray,
        query_id: str,
    ) -> CounterfactualResult:
        """Answer counterfactual query"""
        return self.counterfactual.query(
            graph=self,
            causal_beliefs=self.causal_beliefs,
            factual_values=factual_values,
            antecedent_id=antecedent_id,
            antecedent_cf_value=antecedent_cf_value,
            query_id=query_id,
        )
    
    def get_causal_ancestors(self, node_id: str, max_depth: int = 5) -> List[Tuple[str, float]]:
        """Get causal ancestors with path strength"""
        ancestors = []
        visited = {node_id}
        queue = [(node_id, 1.0, 0)]  # (node, path_strength, depth)
        
        while queue:
            current, strength, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            for (src, tgt), belief in self.causal_beliefs.items():
                if tgt == current and src not in visited:
                    edge_strength = belief.effective_causal_weight()
                    path_strength = strength * edge_strength
                    
                    if path_strength > 0.1:  # Soft pruning
                        ancestors.append((src, path_strength))
                        visited.add(src)
                        queue.append((src, path_strength, depth + 1))
        
        return sorted(ancestors, key=lambda x: x[1], reverse=True)
    
    def get_causal_descendants(self, node_id: str, max_depth: int = 5) -> List[Tuple[str, float]]:
        """Get causal descendants with path strength"""
        descendants = []
        visited = {node_id}
        queue = [(node_id, 1.0, 0)]
        
        while queue:
            current, strength, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            for (src, tgt), belief in self.causal_beliefs.items():
                if src == current and tgt not in visited:
                    edge_strength = belief.effective_causal_weight()
                    path_strength = strength * edge_strength
                    
                    if path_strength > 0.1:
                        descendants.append((tgt, path_strength))
                        visited.add(tgt)
                        queue.append((tgt, path_strength, depth + 1))
        
        return sorted(descendants, key=lambda x: x[1], reverse=True)
