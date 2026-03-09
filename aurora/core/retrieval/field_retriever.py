"""
AURORA 字段检索
=======================

主类负责编排查询计划、分支检索和结果合并。
底层算法细节拆分到更小的检索子模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aurora.core.components.metric import LowRankMetric
from aurora.core.config.query_types import (
    FACTUAL_ATTRACTOR_WEIGHT,
    FACTUAL_PLOT_PRIORITY_BOOST,
    MULTI_HOP_EXTRA_PAGERANK_ITER,
    SINGLE_SESSION_USER_K_MULTIPLIER,
)
from aurora.core.fact_extractor import FactExtractor
from aurora.core.graph.memory_graph import MemoryGraph
from aurora.core.graph.vector_index import VectorIndex
from aurora.core.models.trace import RetrievalTrace
from aurora.core.retrieval.query_analysis import QueryAnalysisMixin, QueryType, TimeAnchor
from aurora.core.retrieval.search_primitives import SearchPrimitivesMixin
from aurora.core.retrieval.temporal_ranking import TemporalRankingMixin
from aurora.core.retrieval.time_filter import TimeRange, TimeRangeExtractor
from aurora.integrations.embeddings.base import EmbeddingProvider


@dataclass(frozen=True)
class QueryPlan:
    query_type: QueryType
    effective_k: int
    effective_max_iter: int
    effective_reseed_k: int
    effective_attractor_weight: float
    direct_weight: float
    query_keywords: List[str]
    time_range: Optional[TimeRange]
    is_aggregation: bool


class FieldRetriever(
    QueryAnalysisMixin,
    TemporalRankingMixin,
    SearchPrimitivesMixin,
):
    """具有吸引子追踪和图扩散的两阶段检索。"""

    def __init__(self, metric: LowRankMetric, vindex: VectorIndex, graph: MemoryGraph):
        self.metric = metric
        self.vindex = vindex
        self.graph = graph
        self.time_extractor = TimeRangeExtractor()
        self.fact_extractor = FactExtractor()

    def _build_query_plan(
        self,
        query_text: str,
        kinds: Sequence[str],
        k: int,
        max_iter: int,
        reseed_k: int,
        attractor_weight: float,
        query_type: Optional[QueryType],
    ) -> QueryPlan:
        detected_type = query_type if query_type is not None else self._classify_query(query_text)

        effective_k = k
        effective_max_iter = max_iter
        effective_reseed_k = reseed_k

        if detected_type == QueryType.MULTI_HOP:
            effective_k = int(k * 1.5)
            effective_max_iter = max_iter + MULTI_HOP_EXTRA_PAGERANK_ITER
            effective_reseed_k = int(reseed_k * 1.2)
        elif detected_type == QueryType.USER_FACT:
            effective_k = int(k * SINGLE_SESSION_USER_K_MULTIPLIER)
            effective_reseed_k = int(reseed_k * 1.5)

        effective_attractor_weight = attractor_weight
        if detected_type == QueryType.FACTUAL:
            effective_attractor_weight = FACTUAL_ATTRACTOR_WEIGHT
        elif detected_type == QueryType.USER_FACT:
            effective_attractor_weight = FACTUAL_ATTRACTOR_WEIGHT * 0.8

        return QueryPlan(
            query_type=detected_type,
            effective_k=effective_k,
            effective_max_iter=effective_max_iter,
            effective_reseed_k=effective_reseed_k,
            effective_attractor_weight=effective_attractor_weight,
            direct_weight=1.0 - effective_attractor_weight,
            query_keywords=self._extract_query_keywords(query_text)
            if detected_type == QueryType.USER_FACT
            else [],
            time_range=self._extract_time_range(query_text, kinds, detected_type),
            is_aggregation=self._is_aggregation_query(query_text),
        )

    def _extract_time_range(
        self,
        query_text: str,
        kinds: Sequence[str],
        query_type: QueryType,
    ) -> Optional[TimeRange]:
        if query_type != QueryType.TEMPORAL:
            return None
        return self.time_extractor.extract(query_text, self._build_events_timeline(kinds))

    def _build_events_timeline(self, kinds: Sequence[str]) -> List[Tuple[str, float]]:
        timeline: List[Tuple[str, float]] = []
        for node_id in self.graph.g.nodes():
            if self.graph.kind(node_id) not in kinds:
                continue
            ts = self._get_timestamp(node_id)
            if ts <= 0:
                continue
            timeline.append((self._payload_text(node_id), ts))
        return timeline

    def _iter_index_hits(
        self,
        vector: np.ndarray,
        kinds: Sequence[str],
        limit: int,
        time_range: Optional[TimeRange],
    ):
        for kind in kinds:
            for node_id, similarity in self.vindex.search(vector, k=limit, kind=kind):
                if node_id not in self.graph.g:
                    continue
                if not self._matches_time_range(node_id, time_range):
                    continue
                yield node_id, similarity

    def _run_direct_branch(
        self,
        query_text: str,
        query_emb: np.ndarray,
        embed: EmbeddingProvider,
        kinds: Sequence[str],
        damping: float,
        plan: QueryPlan,
    ) -> List[Tuple[str, float, str]]:
        direct_ranked = self._direct_semantic_search(
            query_emb=query_emb,
            kinds=kinds,
            k=plan.effective_k,
            damping=damping,
            max_iter=plan.effective_max_iter,
            query_type=plan.query_type,
        )

        if plan.time_range is not None and plan.time_range.relation not in {"any", "span"}:
            direct_ranked = self._apply_time_filter(direct_ranked, plan.time_range)

        enhanced: List[Tuple[str, float, str]] = []
        for node_id, score, kind in direct_ranked:
            enhanced_score = score
            if kind == "plot":
                enhanced_score += self._compute_fact_key_boost(
                    plot_id=node_id,
                    query_text=query_text,
                    query_emb=query_emb,
                    embed_func=embed,
                )
                if plan.query_type == QueryType.USER_FACT and plan.query_keywords:
                    enhanced_score += self._compute_keyword_boost(node_id, plan.query_keywords)
                    enhanced_score += self._compute_user_role_boost(node_id)
            enhanced.append((node_id, enhanced_score, kind))

        enhanced.sort(key=lambda item: item[1], reverse=True)
        return enhanced

    def _run_attractor_branch(
        self,
        query_emb: np.ndarray,
        kinds: Sequence[str],
        damping: float,
        initial_k: int,
        mean_shift_steps: int,
        plan: QueryPlan,
    ) -> Tuple[List[np.ndarray], List[Tuple[str, float, str]]]:
        candidates: List[Tuple[str, np.ndarray, float]] = []
        for node_id, _similarity in self._iter_index_hits(
            query_emb,
            kinds,
            initial_k,
            plan.time_range,
        ):
            vector = self._payload_vector(node_id)
            if vector is None:
                continue
            candidates.append((node_id, vector, self._payload_mass(node_id)))

        path = self._mean_shift(query_emb, candidates, steps=mean_shift_steps)
        attractor = path[-1]

        personalization: Dict[str, float] = {}
        for node_id, similarity in self._iter_index_hits(
            attractor,
            kinds,
            plan.effective_reseed_k,
            plan.time_range,
        ):
            personalization[node_id] = max(personalization.get(node_id, 0.0), similarity)

        pagerank_scores = self._pagerank(
            personalization,
            damping=damping,
            max_iter=plan.effective_max_iter,
        )

        ranked: List[Tuple[str, float, str]] = []
        for node_id, score in pagerank_scores.items():
            kind = self.graph.kind(node_id)
            if kind not in kinds:
                continue
            ranked.append((node_id, float(score) + 1e-3 * self._payload_mass(node_id), kind))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return path, ranked[: plan.effective_k]

    def _run_keyword_branch(
        self,
        query_text: str,
        kinds: Sequence[str],
        plan: QueryPlan,
    ) -> List[Tuple[str, float, str]]:
        if not plan.is_aggregation:
            return []
        entities = self._extract_aggregation_entities(query_text)
        if not entities:
            return []
        return self._keyword_search(keywords=entities, kinds=kinds, max_results=100)

    def _merge_branch_results(
        self,
        direct_ranked: List[Tuple[str, float, str]],
        attractor_ranked: List[Tuple[str, float, str]],
        keyword_ranked: List[Tuple[str, float, str]],
        plan: QueryPlan,
    ) -> List[Tuple[str, float, str]]:
        merged_scores: Dict[str, Tuple[float, str]] = {}

        for node_id, score, kind in direct_ranked:
            merged_scores[node_id] = (plan.direct_weight * score, kind)

        for node_id, score, kind in attractor_ranked:
            existing_score, existing_kind = merged_scores.get(node_id, (0.0, kind))
            merged_scores[node_id] = (
                existing_score + plan.effective_attractor_weight * score,
                existing_kind,
            )

        if plan.is_aggregation:
            keyword_weight = 0.6
            for node_id, score, kind in keyword_ranked:
                existing_score, existing_kind = merged_scores.get(node_id, (0.0, kind))
                merged_scores[node_id] = (
                    existing_score + keyword_weight * score,
                    existing_kind,
                )

        if plan.query_type == QueryType.FACTUAL:
            for node_id, (score, kind) in list(merged_scores.items()):
                if kind == "plot":
                    merged_scores[node_id] = (score + FACTUAL_PLOT_PRIORITY_BOOST, kind)

        ranked = [
            (node_id, score, kind)
            for node_id, (score, kind) in merged_scores.items()
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _finalize_ranked(
        self,
        ranked: List[Tuple[str, float, str]],
        query_text: str,
        query_emb: np.ndarray,
        requested_k: int,
        plan: QueryPlan,
    ) -> List[Tuple[str, float, str]]:
        if plan.query_type == QueryType.TEMPORAL:
            ranked = self._temporal_aware_rerank(ranked, query_text, plan.effective_k)
        elif plan.query_type == QueryType.CAUSAL:
            ranked = self._postprocess_causal(ranked, query_emb, plan.effective_k)
        else:
            ranked = ranked[: plan.effective_k]
        return ranked[:requested_k]

    def retrieve_hybrid(
        self,
        query_text: str,
        embed: EmbeddingProvider,
        kinds: Sequence[str],
        k: int = 5,
        attractor_weight: float = 0.5,
        initial_k: int = 60,
        mean_shift_steps: int = 3,
        reseed_k: int = 50,
        damping: float = 0.80,
        max_iter: int = 40,
        query_type: Optional[QueryType] = None,
    ) -> RetrievalTrace:
        plan = self._build_query_plan(
            query_text=query_text,
            kinds=kinds,
            k=k,
            max_iter=max_iter,
            reseed_k=reseed_k,
            attractor_weight=attractor_weight,
            query_type=query_type,
        )
        query_emb = embed.embed(query_text)

        direct_ranked = self._run_direct_branch(
            query_text=query_text,
            query_emb=query_emb,
            embed=embed,
            kinds=kinds,
            damping=damping,
            plan=plan,
        )
        attractor_path, attractor_ranked = self._run_attractor_branch(
            query_emb=query_emb,
            kinds=kinds,
            damping=damping,
            initial_k=initial_k,
            mean_shift_steps=mean_shift_steps,
            plan=plan,
        )
        keyword_ranked = self._run_keyword_branch(query_text, kinds, plan)

        ranked = self._merge_branch_results(
            direct_ranked=direct_ranked,
            attractor_ranked=attractor_ranked,
            keyword_ranked=keyword_ranked,
            plan=plan,
        )
        ranked = self._finalize_ranked(
            ranked=ranked,
            query_text=query_text,
            query_emb=query_emb,
            requested_k=k,
            plan=plan,
        )

        trace = RetrievalTrace(
            query=query_text,
            query_emb=query_emb,
            attractor_path=attractor_path,
            ranked=ranked,
        )
        trace.query_type = plan.query_type
        return trace

    def retrieve(
        self,
        query_text: str,
        embed: EmbeddingProvider,
        kinds: Sequence[str],
        k: int = 5,
        initial_k: int = 60,
        mean_shift_steps: int = 3,
        reseed_k: int = 50,
        damping: float = 0.80,
        max_iter: int = 40,
        query_type: Optional[QueryType] = None,
        attractor_weight: float = 0.5,
    ) -> RetrievalTrace:
        return self.retrieve_hybrid(
            query_text=query_text,
            embed=embed,
            kinds=kinds,
            k=k,
            attractor_weight=attractor_weight,
            initial_k=initial_k,
            mean_shift_steps=mean_shift_steps,
            reseed_k=reseed_k,
            damping=damping,
            max_iter=max_iter,
            query_type=query_type,
        )
