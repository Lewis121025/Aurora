"""
AURORA 检索模块
================

封装查询、时间线组织、关系优先检索与反馈学习。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from aurora.core.config.query_types import (
    AGGREGATION_K_MULTIPLIER,
    BENCHMARK_AGGREGATION_K,
    BENCHMARK_DEFAULT_K,
    BENCHMARK_MULTI_SESSION_K,
    MULTI_HOP_K_MULTIPLIER,
    QUESTION_TYPE_HINT_MAPPINGS,
    SINGLE_SESSION_USER_K_MULTIPLIER,
)
from aurora.core.config.retrieval import (
    MAX_RECENT_PLOTS_FOR_RETRIEVAL,
    RECENT_PLOTS_FOR_FEEDBACK,
    RELATIONSHIP_BONUS_SCORE,
    STORY_SIMILARITY_BONUS,
)
from aurora.core.models.story import StoryArc
from aurora.core.models.trace import KnowledgeTimeline, RetrievalTrace, TimelineGroup
from aurora.core.retrieval.query_analysis import QueryType
from aurora.exceptions import ValidationError
from aurora.utils.math_utils import cosine_sim
from aurora.utils.time_utils import now_ts

logger = logging.getLogger(__name__)


class RetrievalMixin:
    """提供查询、时间线组织与反馈学习能力。"""

    def query(
        self,
        text: str,
        k: int = 5,
        asker_id: Optional[str] = None,
        query_type: Optional[QueryType] = None,
        query_type_hint: Optional[str] = None,
    ) -> RetrievalTrace:
        """使用关系优先和类型感知检索查询内存系统。"""
        if not text or not text.strip():
            raise ValidationError("query text cannot be empty")

        detected_type = query_type
        if detected_type is None and query_type_hint:
            hint_lower = query_type_hint.lower().replace(" ", "-")
            mapped_type = QUESTION_TYPE_HINT_MAPPINGS.get(hint_lower)
            if mapped_type:
                try:
                    detected_type = QueryType[mapped_type]
                except KeyError:
                    pass

        if detected_type is None:
            detected_type = self.retriever._classify_query(text)

        is_aggregation = self.retriever._is_aggregation_query(text)

        effective_k = k
        if self.benchmark_mode:
            if is_aggregation:
                effective_k = max(k, BENCHMARK_AGGREGATION_K)
                logger.debug(f"基准模式 + 聚合查询：使用 k={effective_k}")
            elif detected_type == QueryType.MULTI_HOP:
                effective_k = max(k, BENCHMARK_MULTI_SESSION_K)
                logger.debug(f"基准模式 + 多跳：使用 k={effective_k}")
            elif detected_type == QueryType.USER_FACT:
                effective_k = max(k, int(BENCHMARK_DEFAULT_K * SINGLE_SESSION_USER_K_MULTIPLIER))
                logger.debug(f"基准模式 + 用户事实：使用 k={effective_k}")
            else:
                effective_k = max(k, BENCHMARK_DEFAULT_K)
                logger.debug(f"基准模式：使用 k={effective_k}")
        elif is_aggregation:
            effective_k = int(k * AGGREGATION_K_MULTIPLIER)
            logger.debug(f"检测到聚合查询，将 k 从 {k} 调整为 {effective_k}")
        elif detected_type == QueryType.MULTI_HOP:
            effective_k = int(k * MULTI_HOP_K_MULTIPLIER)
            logger.debug(f"检测到多跳查询，将 k 从 {k} 调整为 {effective_k}")
        elif detected_type == QueryType.USER_FACT:
            effective_k = int(k * SINGLE_SESSION_USER_K_MULTIPLIER)
            logger.debug(f"检测到用户事实查询，将 k 从 {k} 调整为 {effective_k}")

        activated_identity = None
        relationship_story = None

        if asker_id:
            story_id = self._relationship_story_index.get(asker_id)
            if story_id:
                relationship_story = self.stories.get(story_id)
                if relationship_story:
                    activated_identity = relationship_story.my_identity_in_this_relationship

        if relationship_story and activated_identity:
            trace = self._retrieve_with_relationship_priority(
                text, relationship_story, k=effective_k, query_type=detected_type
            )
        else:
            trace = self.retriever.retrieve(
                query_text=text,
                embed=self.embedder,
                kinds=self.cfg.retrieval_kinds,
                k=effective_k,
                query_type=detected_type,
            )

        if len(trace.ranked) > k:
            trace.ranked = trace.ranked[:k]

        trace.timeline_group = self._group_into_timelines(trace.ranked)
        trace.include_historical = True
        trace.ranked = self._filter_active_results(trace.ranked)
        trace.asker_id = asker_id
        trace.activated_identity = activated_identity
        trace.query_type = detected_type

        retrieved_scores = [score for _, score, _ in trace.ranked]
        retrieved_texts = []
        for nid, _, kind in trace.ranked:
            content_text = ""
            if kind == "plot":
                plot = self.plots.get(nid)
                if plot:
                    content_text = plot.text
            elif kind == "story":
                story = self.stories.get(nid)
                if story:
                    if hasattr(story, "to_narrative_summary"):
                        content_text = story.to_narrative_summary()
                    elif hasattr(story, "to_relationship_narrative"):
                        content_text = story.to_relationship_narrative()
                    else:
                        content_text = f"包含 {len(story.plot_ids)} 个 plot 的 story"
            elif kind == "theme":
                theme = self.themes.get(nid)
                if theme:
                    content_text = theme.description or theme.name or f"包含 {len(theme.story_ids)} 个 story 的 theme"
            retrieved_texts.append(content_text)

        if self.benchmark_mode:
            abstention_result = None
        else:
            abstention_result = self.abstention_detector.detect(
                query=text,
                retrieved_scores=retrieved_scores,
                retrieved_texts=retrieved_texts,
            )
        trace.abstention = abstention_result

        self._update_access_counts(trace)
        return trace

    def _filter_active_results(
        self, ranked: List[Tuple[str, float, str]]
    ) -> List[Tuple[str, float, str]]:
        """过滤检索结果以仅包含活跃 plot。"""
        filtered: List[Tuple[str, float, str]] = []

        for nid, score, kind in ranked:
            if kind == "plot":
                plot = self.plots.get(nid)
                if plot is None:
                    continue
                if plot.status != "active":
                    logger.debug(
                        f"过滤掉非活跃 plot {nid[:8]}... "
                        f"(status={plot.status}，superseded_by={plot.superseded_by_id})"
                    )
                    continue
            filtered.append((nid, score, kind))

        return filtered

    def _get_update_chain(self, plot_id: str) -> List[str]:
        """获取 plot 的完整更新链。"""
        if plot_id not in self.plots:
            return [plot_id]

        chain: List[str] = []
        visited: set = set()

        current_id: Optional[str] = plot_id
        backward_chain: List[str] = []

        while current_id and current_id not in visited:
            visited.add(current_id)
            backward_chain.insert(0, current_id)

            current_plot = self.plots.get(current_id)
            if current_plot and current_plot.supersedes_id:
                current_id = current_plot.supersedes_id
            else:
                break

        chain.extend(backward_chain)

        current_id = plot_id
        visited.clear()
        visited.add(plot_id)

        while current_id:
            current_plot = self.plots.get(current_id)
            if not current_plot or not current_plot.superseded_by_id:
                break

            next_id = current_plot.superseded_by_id
            if next_id in visited:
                break

            visited.add(next_id)
            chain.append(next_id)
            current_id = next_id

        return chain

    def _group_into_timelines(
        self, ranked: List[Tuple[str, float, str]]
    ) -> TimelineGroup:
        """将检索结果分组为知识时间线。"""
        timelines: List[KnowledgeTimeline] = []
        standalone: List[Tuple[str, float, str]] = []
        processed_plots: set = set()

        plot_results: Dict[str, Tuple[float, str]] = {}
        other_results: List[Tuple[str, float, str]] = []

        for nid, score, kind in ranked:
            if kind == "plot":
                plot_results[nid] = (score, kind)
            else:
                other_results.append((nid, score, kind))

        for plot_id, (score, kind) in plot_results.items():
            if plot_id in processed_plots:
                continue

            chain = self._get_update_chain(plot_id)

            if len(chain) == 1:
                standalone.append((plot_id, score, kind))
                processed_plots.add(plot_id)
            else:
                for pid in chain:
                    processed_plots.add(pid)

                current_id: Optional[str] = None
                for pid in reversed(chain):
                    plot = self.plots.get(pid)
                    if plot and plot.status == "active":
                        current_id = pid
                        break

                best_score = score
                for pid in chain:
                    if pid in plot_results:
                        pid_score, _ = plot_results[pid]
                        best_score = max(best_score, pid_score)

                topic_sig = ""
                if chain:
                    first_plot = self.plots.get(chain[0])
                    if first_plot:
                        topic_sig = first_plot.text[:50]

                timeline = KnowledgeTimeline(
                    chain=chain,
                    current_id=current_id,
                    topic_signature=topic_sig,
                    match_score=best_score,
                )
                timelines.append(timeline)

        timelines.sort(key=lambda t: t.match_score, reverse=True)
        standalone.extend(other_results)
        return TimelineGroup(timelines=timelines, standalone_results=standalone)

    def query_with_timeline(
        self,
        text: str,
        k: int = 5,
        asker_id: Optional[str] = None,
        query_type: Optional[QueryType] = None,
    ) -> RetrievalTrace:
        """使用完整时间线背景进行查询以进行时间推理。"""
        trace = self.query(
            text=text,
            k=k * 2,
            asker_id=asker_id,
            query_type=query_type,
        )

        if trace.timeline_group:
            all_ranked: List[Tuple[str, float, str]] = []

            for timeline in trace.timeline_group.timelines:
                for plot_id in timeline.chain:
                    plot = self.plots.get(plot_id)
                    if plot:
                        score = timeline.match_score
                        all_ranked.append((plot_id, score, "plot"))

            all_ranked.extend(trace.timeline_group.standalone_results)
            all_ranked.sort(key=lambda x: x[1], reverse=True)
            trace.ranked = all_ranked[:k]

        return trace

    def get_knowledge_evolution(self, topic_query: str, k: int = 5) -> List[KnowledgeTimeline]:
        """获取特定知识主题的演化时间线。"""
        trace = self.query_with_timeline(topic_query, k=k * 2)

        if trace.timeline_group:
            evolved = [t for t in trace.timeline_group.timelines if t.has_evolution()]
            return evolved[:k]

        return []

    def _retrieve_with_relationship_priority(
        self,
        text: str,
        relationship_story: StoryArc,
        k: int,
        query_type: Optional[QueryType] = None,
    ) -> RetrievalTrace:
        """使用优先级检索关系的历史。"""
        query_emb = self.embedder.embed(text)
        relationship_results = self._get_relationship_results(query_emb, relationship_story)

        semantic_trace = self.retriever.retrieve(
            query_text=text,
            embed=self.embedder,
            kinds=self.cfg.retrieval_kinds,
            k=k,
            query_type=query_type,
        )

        ranked = self._merge_retrieval_results(relationship_results, semantic_trace.ranked, k)

        trace = RetrievalTrace(
            query=text,
            query_emb=query_emb,
            attractor_path=semantic_trace.attractor_path,
            ranked=ranked,
        )
        trace.query_type = query_type
        return trace

    def _get_relationship_results(
        self, query_emb: np.ndarray, relationship_story: StoryArc
    ) -> List[Tuple[str, float, str]]:
        """从关系的历史获取检索结果。"""
        results: List[Tuple[str, float, str]] = []

        for plot_id in relationship_story.plot_ids[-MAX_RECENT_PLOTS_FOR_RETRIEVAL:]:
            plot = self.plots.get(plot_id)
            if plot is None or plot.status != "active":
                continue

            sem_sim = self.metric.sim(query_emb, plot.embedding)
            score = sem_sim + RELATIONSHIP_BONUS_SCORE
            results.append((plot_id, score, "plot"))

        if relationship_story.centroid is not None:
            story_sim = self.metric.sim(query_emb, relationship_story.centroid)
            results.append((relationship_story.id, story_sim + STORY_SIMILARITY_BONUS, "story"))

        return results

    def _merge_retrieval_results(
        self,
        relationship_results: List[Tuple[str, float, str]],
        semantic_results: List[Tuple[str, float, str]],
        k: int,
        diversity_lambda: float = 0.3,
    ) -> List[Tuple[str, float, str]]:
        """使用 MMR 多样性合并关系和语义检索结果。"""
        all_results: Dict[str, Tuple[float, str]] = {}

        for nid, score, kind in relationship_results:
            all_results[nid] = (score, kind)

        for nid, score, kind in semantic_results:
            if nid not in all_results:
                all_results[nid] = (score, kind)
            else:
                existing_score, _existing_kind = all_results[nid]
                if score > existing_score:
                    all_results[nid] = (score, kind)

        if not all_results:
            return []

        candidates: List[Tuple[str, float, str, Optional[np.ndarray]]] = []
        for nid, (score, kind) in all_results.items():
            emb = self._get_embedding_for_node(nid)
            candidates.append((nid, score, kind, emb))

        selected: List[Tuple[str, float, str]] = []
        selected_embeddings: List[np.ndarray] = []
        remaining = list(candidates)

        while len(selected) < k and remaining:
            best_idx = -1
            best_mmr = float("-inf")

            for idx, (_nid, score, _kind, emb) in enumerate(remaining):
                relevance = score

                max_sim_to_selected = 0.0
                if selected_embeddings and emb is not None:
                    for sel_emb in selected_embeddings:
                        if sel_emb is not None:
                            sim = cosine_sim(emb, sel_emb)
                            max_sim_to_selected = max(max_sim_to_selected, sim)

                mmr_score = diversity_lambda * relevance - (1.0 - diversity_lambda) * max_sim_to_selected

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                nid, score, kind, emb = remaining.pop(best_idx)
                selected.append((nid, score, kind))
                if emb is not None:
                    selected_embeddings.append(emb)

        return selected

    def _get_embedding_for_node(self, nid: str) -> Optional[np.ndarray]:
        """获取节点（plot、story 或 theme）的嵌入向量。"""
        if nid in self.plots:
            return self.plots[nid].embedding
        if nid in self.stories:
            return self.stories[nid].centroid
        if nid in self.themes:
            return self.themes[nid].prototype
        return None

    def _update_access_counts(self, trace: RetrievalTrace) -> None:
        """更新检索项的访问计数。"""
        for nid, _, kind in trace.ranked:
            if kind == "plot":
                plot = self.graph.payload(nid)
                plot.access_count += 1
                plot.last_access_ts = now_ts()
            elif kind == "story":
                story = self.graph.payload(nid)
                story.reference_count += 1

    def feedback_retrieval(self, query_text: str, chosen_id: str, success: bool) -> None:
        """为检索结果提供反馈以启用在线学习。"""
        query_emb = self.embedder.embed(query_text)
        graph = self.graph.g

        self._update_edge_beliefs(query_emb, chosen_id, success, graph)
        self._update_metric_triplet(query_emb, chosen_id, success, graph)
        self._update_encode_gate(success)

        if chosen_id in self.themes:
            self.themes[chosen_id].update_evidence(success)

    def _update_edge_beliefs(
        self, query_emb: np.ndarray, chosen_id: str, success: bool, graph: nx.DiGraph
    ) -> None:
        """更新最短路径上的边信念。"""
        seeds = [i for i, _ in self.vindex.search(query_emb, k=10)]
        if chosen_id in graph:
            for seed in seeds:
                if seed not in graph:
                    continue
                try:
                    path = nx.shortest_path(graph, source=seed, target=chosen_id)
                except nx.NetworkXNoPath:
                    continue
                for u, v in zip(path[:-1], path[1:]):
                    self.graph.edge_belief(u, v).update(success)

    def _update_metric_triplet(
        self, query_emb: np.ndarray, chosen_id: str, success: bool, graph: nx.DiGraph
    ) -> None:
        """使用三元组损失更新度量。"""
        if chosen_id not in graph:
            return

        chosen = self.graph.payload(chosen_id)
        pos_emb = getattr(chosen, "embedding", getattr(chosen, "centroid", getattr(chosen, "prototype", None)))

        if pos_emb is None:
            return

        cands = [i for i, _ in self.vindex.search(query_emb, k=30) if i != chosen_id and i in graph]
        if not cands:
            return

        neg_id = self.rng.choice(cands)
        neg = self.graph.payload(neg_id)
        neg_emb = getattr(neg, "embedding", getattr(neg, "centroid", getattr(neg, "prototype", None)))

        if neg_emb is not None:
            self.metric.update_triplet(anchor=query_emb, positive=pos_emb, negative=neg_emb)

    def _update_encode_gate(self, success: bool) -> None:
        """使用奖励信号更新编码门。"""
        reward = 1.0 if success else -1.0
        recent = list(self._recent_encoded_plot_ids)[-RECENT_PLOTS_FOR_FEEDBACK:]

        for pid in recent:
            plot = self.plots.get(pid)
            if plot is not None:
                x = self._compute_voi_features(plot)
                self.gate.update(x, reward)
