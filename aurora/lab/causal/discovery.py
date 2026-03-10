"""
AURORA 因果发现
====================

从观察数据中推断因果方向与混淆概率。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from aurora.lab.primitives.metric import LowRankMetric
from aurora.lab.models.plot import Plot
from aurora.utils.math_utils import cosine_sim, l2_normalize, sigmoid


class CausalDiscovery:
    """从观察数据推断因果方向。"""

    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)

    def infer_direction(
        self,
        source: Plot,
        target: Plot,
        context_plots: List[Plot],
    ) -> Tuple[float, float, float]:
        log_odds_forward = 0.0

        time_diff = target.ts - source.ts
        if time_diff > 0:
            log_odds_forward += 0.5 * min(1.0, time_diff / 3600.0)
        elif time_diff < 0:
            log_odds_forward -= 0.5 * min(1.0, abs(time_diff) / 3600.0)

        te_forward = self._estimate_transfer_entropy(source, target, context_plots)
        te_backward = self._estimate_transfer_entropy(target, source, context_plots)
        log_odds_forward += 2.0 * (te_forward - te_backward)

        anm_forward = self._anm_independence_score(source, target, context_plots)
        anm_backward = self._anm_independence_score(target, source, context_plots)
        log_odds_forward += 1.5 * (anm_forward - anm_backward)

        complexity_source = self._conditional_complexity(source, context_plots)
        complexity_target = self._conditional_complexity(target, context_plots)
        log_odds_forward += 0.3 * (complexity_target - complexity_source)

        confound_evidence = self._detect_confounding_signals(source, target, context_plots)
        return (
            sigmoid(log_odds_forward),
            sigmoid(-log_odds_forward),
            sigmoid(confound_evidence),
        )

    def _estimate_transfer_entropy(
        self,
        source: Plot,
        target: Plot,
        context: List[Plot],
    ) -> float:
        target_history = [
            plot for plot in context if plot.ts < target.ts and plot.story_id == target.story_id
        ]
        if not target_history or source.ts >= target.ts:
            return 0.0

        baseline_pred = self._predict_embedding(target_history)
        baseline_error = self.metric.d2(baseline_pred, target.embedding)

        augmented_pred = self._predict_embedding(target_history + [source])
        augmented_error = self.metric.d2(augmented_pred, target.embedding)
        info_gain = max(0.0, baseline_error - augmented_error)
        return info_gain / (baseline_error + 1e-6)

    def _predict_embedding(self, plots: List[Plot]) -> np.ndarray:
        if not plots:
            return np.zeros(self.metric.dim, dtype=np.float32)

        sorted_plots = sorted(plots, key=lambda plot: plot.ts)
        weights = [math.exp(index * 0.3) for index in range(len(sorted_plots))]
        total_weight = sum(weights)

        prediction = np.zeros_like(sorted_plots[0].embedding)
        for weight, plot in zip(weights, sorted_plots):
            prediction += (weight / total_weight) * plot.embedding
        return l2_normalize(prediction)

    def _anm_independence_score(
        self,
        cause: Plot,
        effect: Plot,
        context: List[Plot],
    ) -> float:
        del context

        cause_emb = cause.embedding.astype(np.float32)
        effect_emb = effect.embedding.astype(np.float32)
        projection = np.dot(effect_emb, cause_emb) * cause_emb
        residual = effect_emb - projection
        if np.linalg.norm(residual) < 1e-6:
            return 0.0

        correlation = abs(np.dot(l2_normalize(residual), cause_emb))
        return 1.0 - correlation

    def _conditional_complexity(self, plot: Plot, context: List[Plot]) -> float:
        if not context:
            return 0.0
        similar_count = sum(
            1 for candidate in context if self.metric.sim(plot.embedding, candidate.embedding) > 0.5
        )
        return math.log1p(similar_count)

    def _detect_confounding_signals(
        self,
        first: Plot,
        second: Plot,
        context: List[Plot],
    ) -> float:
        log_odds_confound = 0.0
        earlier_ts = min(first.ts, second.ts)
        potential_confounders = [
            plot
            for plot in context
            if plot.ts < earlier_ts and plot.id not in (first.id, second.id)
        ]

        for confounder in potential_confounders:
            similarity_first = self.metric.sim(confounder.embedding, first.embedding)
            similarity_second = self.metric.sim(confounder.embedding, second.embedding)
            if similarity_first > 0.6 and similarity_second > 0.6:
                log_odds_confound += 0.5 * (similarity_first + similarity_second)

        shared_actors = set(first.actors) & set(second.actors)
        semantic_similarity = cosine_sim(first.embedding, second.embedding)
        if shared_actors and semantic_similarity < 0.4:
            log_odds_confound += 0.3 * len(shared_actors)

        return log_odds_confound
