"""
AURORA 因果干预与反事实
===========================

实现 do-干预和反事实推理。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from aurora.core.causal.models import CausalEdgeBelief, CounterfactualResult, InterventionResult
from aurora.core.components.metric import LowRankMetric
from aurora.core.graph.memory_graph import MemoryGraph
from aurora.utils.math_utils import l2_normalize


class InterventionEngine:
    """执行 `do(X=x)` 干预。"""

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
        causal_dag = self._build_causal_dag(graph, causal_beliefs)

        for predecessor in list(causal_dag.predecessors(target_id)):
            causal_dag.remove_edge(predecessor, target_id)

        values: Dict[str, np.ndarray] = {target_id: intervention_value}
        confidence_product = 1.0
        causal_chain: List[str] = []

        try:
            traversal_order = list(nx.topological_sort(causal_dag))
        except nx.NetworkXUnfeasible:
            traversal_order = list(nx.bfs_tree(causal_dag, target_id))

        for node_id in traversal_order:
            if node_id in values:
                causal_chain.append(node_id)
                continue

            parents = list(causal_dag.predecessors(node_id))
            if not parents:
                values[node_id] = self._get_node_embedding(graph, node_id)
                continue

            parent_values, parent_weights, confidence_delta = self._collect_parent_effects(
                parents,
                node_id,
                values,
                causal_beliefs,
            )
            confidence_product *= confidence_delta
            if not parent_values:
                continue

            predicted = self._weighted_average(parent_values, parent_weights)
            original = self._get_node_embedding(graph, node_id)
            blend_factor = min(1.0, sum(parent_weights))
            values[node_id] = l2_normalize(blend_factor * predicted + (1 - blend_factor) * original)
            causal_chain.append(node_id)

        return InterventionResult(
            target_id=target_id,
            intervention_value=intervention_value,
            predicted_effects=values,
            causal_chain=causal_chain,
            confidence=confidence_product ** (1.0 / max(len(causal_chain), 1)),
        )

    def _build_causal_dag(
        self,
        graph: MemoryGraph,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
    ) -> nx.DiGraph:
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.g.nodes())

        for (source, target), belief in causal_beliefs.items():
            if belief.direction_belief() > 0.5:
                dag.add_edge(source, target, weight=belief.effective_causal_weight())

        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle = nx.find_cycle(dag)
                weakest_edge = min(cycle, key=lambda edge: dag.edges[edge[0], edge[1]].get("weight", 0))
                dag.remove_edge(weakest_edge[0], weakest_edge[1])
            except nx.NetworkXNoCycle:
                break

        return dag

    def _collect_parent_effects(
        self,
        parents: List[str],
        node_id: str,
        values: Dict[str, np.ndarray],
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
    ) -> Tuple[List[np.ndarray], List[float], float]:
        parent_values: List[np.ndarray] = []
        parent_weights: List[float] = []
        confidence_product = 1.0

        for parent_id in parents:
            if parent_id not in values:
                continue

            edge_key = (parent_id, node_id)
            belief = causal_beliefs.get(edge_key)
            weight = belief.effective_causal_weight() if belief else 0.3
            confidence_product *= belief.causal_strength() if belief else 0.5
            parent_values.append(values[parent_id])
            parent_weights.append(weight)

        return parent_values, parent_weights, confidence_product

    def _weighted_average(self, vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
        total_weight = sum(weights) + 1e-9
        prediction = np.zeros_like(vectors[0])
        for vector, weight in zip(vectors, weights):
            prediction += (weight / total_weight) * vector
        return prediction

    def _get_node_embedding(self, graph: MemoryGraph, node_id: str) -> np.ndarray:
        try:
            payload = graph.payload(node_id)
            if hasattr(payload, "embedding"):
                return payload.embedding
            if hasattr(payload, "centroid"):
                return payload.centroid
            if hasattr(payload, "prototype"):
                return payload.prototype
        except KeyError:
            pass
        return np.zeros(self.metric.dim, dtype=np.float32)


class CounterfactualReasoner:
    """回答“如果 X 不同，会怎样？”的问题。"""

    def __init__(
        self,
        metric: LowRankMetric,
        intervention_engine: InterventionEngine,
        seed: int = 0,
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
        noise = self._infer_noise(graph, causal_beliefs, factual_values)
        intervention_result = self.intervention.do_intervention(
            graph=graph,
            target_id=antecedent_id,
            intervention_value=antecedent_cf_value,
            causal_beliefs=causal_beliefs,
        )
        counterfactual_values = self._propagate_with_noise(
            graph,
            causal_beliefs,
            intervention_result.predicted_effects,
            noise,
        )

        factual_value = factual_values.get(query_id, np.zeros(self.metric.dim))
        counterfactual_value = counterfactual_values.get(query_id, factual_value)
        difference = float(np.linalg.norm(counterfactual_value - factual_value))

        return CounterfactualResult(
            factual_world=factual_values,
            counterfactual_world=counterfactual_values,
            query_variable=query_id,
            factual_value=factual_value,
            counterfactual_value=counterfactual_value,
            difference_magnitude=difference,
            explanation=self._generate_explanation(
                antecedent_id,
                query_id,
                factual_value,
                counterfactual_value,
            ),
        )

    def _infer_noise(
        self,
        graph: MemoryGraph,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
        factual_values: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        noise: Dict[str, np.ndarray] = {}
        dag = self.intervention._build_causal_dag(graph, causal_beliefs)

        for node_id, observed in factual_values.items():
            if node_id not in dag:
                continue

            parents = list(dag.predecessors(node_id))
            if not parents:
                noise[node_id] = np.zeros_like(observed)
                continue

            parent_values, parent_weights, _ = self.intervention._collect_parent_effects(
                parents,
                node_id,
                factual_values,
                causal_beliefs,
            )
            if parent_values:
                predicted = self.intervention._weighted_average(parent_values, parent_weights)
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
        dag = self.intervention._build_causal_dag(graph, causal_beliefs)
        values = dict(intervention_values)

        try:
            traversal_order = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            traversal_order = list(values.keys())

        for node_id in traversal_order:
            if node_id in intervention_values:
                continue

            parents = list(dag.predecessors(node_id))
            if not parents:
                values[node_id] = noise.get(node_id, np.zeros(self.metric.dim))
                continue

            parent_values, parent_weights, _ = self.intervention._collect_parent_effects(
                parents,
                node_id,
                values,
                causal_beliefs,
            )
            if not parent_values:
                continue

            predicted = self.intervention._weighted_average(parent_values, parent_weights)
            values[node_id] = predicted + noise.get(node_id, np.zeros_like(predicted))

        return values

    def _generate_explanation(
        self,
        antecedent_id: str,
        query_id: str,
        factual_value: np.ndarray,
        counterfactual_value: np.ndarray,
    ) -> str:
        difference = float(np.linalg.norm(counterfactual_value - factual_value))
        if difference < 0.1:
            return f"Changing {antecedent_id} would have minimal effect on {query_id}."
        if difference < 0.3:
            return f"Changing {antecedent_id} would moderately affect {query_id}."
        return f"Changing {antecedent_id} would significantly change {query_id}."
