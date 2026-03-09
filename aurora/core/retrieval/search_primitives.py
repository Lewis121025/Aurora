"""
AURORA 检索原语
====================

图扩散、语义搜索、关键词增强和因果后处理。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from aurora.core.components.metric import LowRankMetric
from aurora.core.config.query_types import (
    FACTUAL_SEMANTIC_WEIGHT,
    FACT_KEY_BOOST_MAX,
    KEYWORD_MATCH_BOOST,
    KEYWORD_MATCH_MIN_RATIO,
    USER_ROLE_PRIORITY_BOOST,
)
from aurora.core.fact_extractor import FactExtractor
from aurora.core.graph.edge_belief import EdgeBelief
from aurora.core.retrieval.query_analysis import QueryType
from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.utils.math_utils import l2_normalize, softmax


USER_MARKERS = ("user:", "用户:", "user：", "用户：")


class SearchPrimitivesMixin:
    """检索核心算法和评分逻辑。"""

    metric: LowRankMetric

    def _payload_text(self, node_id: str) -> str:
        payload = self.graph.payload(node_id)
        if payload is None:
            return ""
        return getattr(payload, "text", "") or getattr(payload, "name", "")

    def _payload_vector(self, node_id: str) -> Optional[np.ndarray]:
        payload = self.graph.payload(node_id)
        if payload is None:
            return None
        return getattr(
            payload,
            "embedding",
            getattr(payload, "centroid", getattr(payload, "prototype", None)),
        )

    def _payload_mass(self, node_id: str) -> float:
        payload = self.graph.payload(node_id)
        if payload is None or not hasattr(payload, "mass"):
            return 0.0
        return float(payload.mass())

    def _keyword_search(
        self,
        keywords: List[str],
        kinds: Sequence[str],
        max_results: int = 100,
        exhaustive: bool = False,
    ) -> List[Tuple[str, float, str]]:
        if not keywords:
            return []

        keywords_lower = [keyword.lower() for keyword in keywords]
        results: List[Tuple[str, float, str]] = []

        for node_id in self.graph.g.nodes():
            kind = self.graph.kind(node_id)
            if kind not in kinds:
                continue

            text_lower = self._payload_text(node_id).lower()
            if not text_lower:
                continue

            match_count = 0.0
            for keyword in keywords_lower:
                if keyword in text_lower:
                    match_count += 1.0
                elif len(keyword) >= 4 and keyword[:4] in text_lower:
                    match_count += 0.5

            if match_count < 1:
                continue

            text_words = len(text_lower.split())
            density_bonus = min(0.3, match_count / max(text_words, 1) * 10)
            score = match_count / len(keywords_lower) + density_bonus
            results.append((node_id, score, kind))

        results.sort(key=lambda item: item[1], reverse=True)
        return results if exhaustive else results[:max_results]

    def _compute_keyword_boost(self, plot_id: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0

        text_lower = self._payload_text(plot_id).lower()
        if not text_lower:
            return 0.0

        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches == 0:
            return 0.0

        match_ratio = matches / len(keywords)
        if match_ratio < KEYWORD_MATCH_MIN_RATIO:
            return 0.0
        return KEYWORD_MATCH_BOOST * min(1.0, match_ratio * 1.5)

    def _compute_user_role_boost(self, plot_id: str) -> float:
        text_lower = self._payload_text(plot_id).lower()
        return USER_ROLE_PRIORITY_BOOST if any(marker in text_lower for marker in USER_MARKERS) else 0.0

    def _compute_fact_key_boost(
        self,
        plot_id: str,
        query_text: str,
        query_emb: np.ndarray,
        embed_func: Optional[EmbeddingProvider] = None,
    ) -> float:
        del query_emb, embed_func

        try:
            payload = self.graph.payload(plot_id)
            if payload is None or not hasattr(payload, "fact_keys") or not payload.fact_keys:
                return 0.0

            query_facts = self.fact_extractor.extract(query_text)
            if not query_facts:
                return 0.0

            matches = 0.0
            for query_fact in query_facts:
                query_fact_text = query_fact.fact_text.lower()
                for plot_fact_key in payload.fact_keys:
                    plot_fact_key_lower = plot_fact_key.lower()
                    if query_fact_text == plot_fact_key_lower:
                        matches += 1.0
                        break
                    if (
                        query_fact_text in plot_fact_key_lower
                        or plot_fact_key_lower in query_fact_text
                    ):
                        matches += 0.5
                        break
                    if (
                        query_fact.fact_type in plot_fact_key_lower
                        or plot_fact_key_lower.startswith(query_fact.fact_type + ":")
                    ) and query_fact.entities and any(
                        entity.lower() in plot_fact_key_lower
                        for entity in query_fact.entities
                    ):
                        matches += 0.3
                        break

            if matches == 0:
                return 0.0

            match_score = min(1.0, matches / max(1, len(query_facts)))
            return FACT_KEY_BOOST_MAX * match_score
        except Exception:
            return 0.0

    def _mean_shift(
        self,
        x0: np.ndarray,
        candidates: List[Tuple[str, np.ndarray, float]],
        steps: int = 3,
    ) -> List[np.ndarray]:
        if not candidates:
            return [x0]

        x = x0.copy()
        path = [x.copy()]

        for _ in range(steps):
            distances = [self.metric.d2(x, vector) for _, vector, _ in candidates]
            sigma2 = float(np.median(distances)) + 1e-6
            logits = [
                -(distance / (2.0 * sigma2)) + mass
                for distance, (_, _, mass) in zip(distances, candidates)
            ]
            weights = softmax(logits)
            new_x = np.zeros_like(x)
            for weight, (_, vector, _) in zip(weights, candidates):
                new_x += weight * vector
            x = l2_normalize(new_x)
            path.append(x.copy())

        return path

    def _pagerank(
        self,
        personalization: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 50,
    ) -> Dict[str, float]:
        graph = self.graph.g
        personalization = {
            node_id: value for node_id, value in personalization.items() if node_id in graph
        }
        if not personalization:
            return {}

        cached = self.graph.get_cached_pagerank(personalization, damping, max_iter)
        if cached is not None:
            return cached

        weighted_graph = nx.DiGraph()
        weighted_graph.add_nodes_from(graph.nodes())
        for source, target, data in graph.edges(data=True):
            belief: EdgeBelief = data["belief"]
            weighted_graph.add_edge(source, target, w=max(1e-6, belief.mean()))

        try:
            result = nx.pagerank(
                weighted_graph,
                alpha=damping,
                personalization=personalization,
                weight="w",
                max_iter=max_iter,
            )
        except nx.PowerIterationFailedConvergence:
            node_count = len(weighted_graph.nodes())
            result = (
                {node_id: 1.0 / node_count for node_id in weighted_graph.nodes()}
                if node_count > 0
                else {}
            )

        self.graph.set_cached_pagerank(personalization, damping, max_iter, result)
        return result

    def _direct_semantic_search(
        self,
        query_emb: np.ndarray,
        kinds: Sequence[str],
        k: int,
        damping: float = 0.80,
        max_iter: int = 40,
        semantic_weight: float = 0.7,
        query_type: Optional[QueryType] = None,
    ) -> List[Tuple[str, float, str]]:
        if query_type == QueryType.FACTUAL:
            semantic_weight = FACTUAL_SEMANTIC_WEIGHT

        personalization: Dict[str, float] = {}
        semantic_scores: Dict[str, float] = {}

        for kind in kinds:
            for node_id, similarity in self.vindex.search(query_emb, k=k * 2, kind=kind):
                if node_id not in self.graph.g:
                    continue
                if similarity > personalization.get(node_id, 0.0):
                    personalization[node_id] = similarity
                    semantic_scores[node_id] = similarity

        if not personalization:
            return []

        pagerank_scores = self._pagerank(personalization, damping=damping, max_iter=max_iter)
        pr_values = list(pagerank_scores.values())
        pr_max = max(pr_values) if pr_values else 1.0
        pr_min = min(pr_values) if pr_values else 0.0
        pr_range = pr_max - pr_min if pr_max > pr_min else 1.0
        pagerank_weight = 1.0 - semantic_weight

        ranked: List[Tuple[str, float, str]] = []
        for node_id, pr_score in pagerank_scores.items():
            kind = self.graph.kind(node_id)
            if kind not in kinds:
                continue

            semantic_score = semantic_scores.get(node_id, 0.0)
            normalized_pr = (pr_score - pr_min) / pr_range if pr_range > 0 else 0.5
            blended_score = semantic_weight * semantic_score + pagerank_weight * normalized_pr
            blended_score += 1e-4 * self._payload_mass(node_id)
            ranked.append((node_id, blended_score, kind))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]

    def _postprocess_causal(
        self,
        ranked: List[Tuple[str, float, str]],
        query_emb: np.ndarray,
        k: int,
    ) -> List[Tuple[str, float, str]]:
        if not ranked:
            return ranked

        graph = self.graph.g
        causal_expanded: Dict[str, Tuple[float, str]] = {
            node_id: (score, kind) for node_id, score, kind in ranked
        }

        for node_id, score, _kind in ranked[: min(10, len(ranked))]:
            if node_id not in graph:
                continue

            for neighbor in graph.neighbors(node_id):
                edge_data = graph.get_edge_data(node_id, neighbor)
                if edge_data is None or edge_data.get("etype") != "causal":
                    continue

                belief = edge_data.get("belief")
                edge_weight = belief.mean() if belief else 0.5
                neighbor_vector = self._payload_vector(neighbor)
                if neighbor_vector is None:
                    continue

                similarity = float(
                    np.dot(query_emb, neighbor_vector)
                    / (np.linalg.norm(query_emb) * np.linalg.norm(neighbor_vector) + 1e-8)
                )
                causal_score = 0.5 * score * edge_weight + 0.5 * similarity
                neighbor_kind = self.graph.kind(neighbor)

                if neighbor not in causal_expanded or causal_expanded[neighbor][0] < causal_score:
                    causal_expanded[neighbor] = (causal_score, neighbor_kind)

        result = [
            (node_id, score, kind)
            for node_id, (score, kind) in causal_expanded.items()
        ]
        result.sort(key=lambda item: item[1], reverse=True)
        return result[:k]
