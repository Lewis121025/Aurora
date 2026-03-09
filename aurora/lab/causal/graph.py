"""
AURORA 因果图
================

将因果发现、干预和反事实能力组合到 MemoryGraph 上。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from aurora.lab.causal.discovery import CausalDiscovery
from aurora.lab.causal.intervention import CounterfactualReasoner, InterventionEngine
from aurora.lab.causal.models import CausalEdgeBelief, CounterfactualResult, InterventionResult
from aurora.lab.graph.memory_graph import MemoryGraph
from aurora.lab.primitives.metric import LowRankMetric
from aurora.lab.models.plot import Plot


class CausalMemoryGraph(MemoryGraph):
    """带因果推理能力的扩展 MemoryGraph。"""

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
        key = (source_id, target_id)
        if key not in self.causal_beliefs:
            self.causal_beliefs[key] = CausalEdgeBelief(
                edge_type="causal",
                dir_a=initial_direction_belief * 2,
                dir_b=(1 - initial_direction_belief) * 2,
            )
            self.ensure_edge(source_id, target_id, "causal")
        return self.causal_beliefs[key]

    def infer_and_add_causal_edge(
        self,
        source: Plot,
        target: Plot,
        context_plots: List[Plot],
    ) -> CausalEdgeBelief:
        p_forward, p_backward, p_confound = self.discovery.infer_direction(
            source,
            target,
            context_plots,
        )
        if p_forward >= p_backward:
            belief = self.add_causal_edge(source.id, target.id, p_forward)
        else:
            belief = self.add_causal_edge(target.id, source.id, p_backward)

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
        return self.counterfactual.query(
            graph=self,
            causal_beliefs=self.causal_beliefs,
            factual_values=factual_values,
            antecedent_id=antecedent_id,
            antecedent_cf_value=antecedent_cf_value,
            query_id=query_id,
        )

    def get_causal_ancestors(self, node_id: str, max_depth: int = 5) -> List[Tuple[str, float]]:
        return self._walk_causal_graph(node_id=node_id, direction="backward", max_depth=max_depth)

    def get_causal_descendants(self, node_id: str, max_depth: int = 5) -> List[Tuple[str, float]]:
        return self._walk_causal_graph(node_id=node_id, direction="forward", max_depth=max_depth)

    def _walk_causal_graph(
        self,
        node_id: str,
        direction: str,
        max_depth: int,
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        visited = {node_id}
        queue = [(node_id, 1.0, 0)]

        while queue:
            current, strength, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for (source, target), belief in self.causal_beliefs.items():
                if direction == "backward" and target == current and source not in visited:
                    next_node = source
                elif direction == "forward" and source == current and target not in visited:
                    next_node = target
                else:
                    continue

                path_strength = strength * belief.effective_causal_weight()
                if path_strength <= 0.1:
                    continue

                results.append((next_node, path_strength))
                visited.add(next_node)
                queue.append((next_node, path_strength, depth + 1))

        return sorted(results, key=lambda item: item[1], reverse=True)
