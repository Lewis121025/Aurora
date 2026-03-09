"""
AURORA 因果模型
====================

因果推理相关的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from aurora.core.graph.edge_belief import EdgeBelief


@dataclass
class CausalEdgeBelief(EdgeBelief):
    """扩展的 EdgeBelief，增加因果方向、强度和混淆后验。"""

    dir_a: float = 1.0
    dir_b: float = 1.0
    str_a: float = 1.0
    str_b: float = 1.0
    conf_a: float = 1.0
    conf_b: float = 9.0
    mechanism_emb: Optional[np.ndarray] = None
    intervention_count: int = 0
    observation_count: int = 0

    def direction_belief(self) -> float:
        return self.dir_a / (self.dir_a + self.dir_b)

    def causal_strength(self) -> float:
        return self.str_a / (self.str_a + self.str_b)

    def confound_prob(self) -> float:
        return self.conf_a / (self.conf_a + self.conf_b)

    def effective_causal_weight(self) -> float:
        return self.direction_belief() * self.causal_strength() * (1.0 - self.confound_prob())

    def sample_direction(self, rng: np.random.Generator) -> bool:
        return rng.beta(self.dir_a, self.dir_b) > 0.5

    def update_direction_evidence(self, forward: bool, weight: float = 1.0) -> None:
        if forward:
            self.dir_a += weight
        else:
            self.dir_b += weight
        self.observation_count += 1

    def update_intervention_evidence(self, effect_observed: bool, weight: float = 1.0) -> None:
        if effect_observed:
            self.str_a += weight
        else:
            self.str_b += weight
        self.intervention_count += 1
        self.conf_b += 0.5 * weight

    def update_confound_evidence(self, confound_detected: bool, weight: float = 1.0) -> None:
        if confound_detected:
            self.conf_a += weight
        else:
            self.conf_b += weight


@dataclass
class InterventionResult:
    """`do(X=x)` 干预结果。"""

    target_id: str
    intervention_value: np.ndarray
    predicted_effects: Dict[str, np.ndarray]
    causal_chain: List[str]
    confidence: float


@dataclass
class CounterfactualResult:
    """反事实查询结果。"""

    factual_world: Dict[str, np.ndarray]
    counterfactual_world: Dict[str, np.ndarray]
    query_variable: str
    factual_value: np.ndarray
    counterfactual_value: np.ndarray
    difference_magnitude: float
    explanation: str
