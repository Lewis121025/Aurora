"""
aurora/soul/engine.py
核心引擎模块：负责 graph-first 记忆摄入、图视图物化、身份演化以及
dream/repair 图算子的协调。它是 Aurora V4 系统的枢纽，将检索、图谱、提取与响应上下文连接起来。
"""

from __future__ import annotations

import copy
import json
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np

from aurora.integrations.embeddings.base import ContentEmbeddingProvider, TextEmbeddingProvider
from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding
from aurora.soul.graph_ops import GraphDreamOperator, GraphRepairOperator
from aurora.soul.graph_views import GraphViewBuilder, GraphViewStats
from aurora.soul.query import BaseQueryAnalyzer, MissingQueryAnalyzer
from aurora.soul.extractors import (
    CombinatorialNarrativeProvider,
    HeuristicMeaningProvider,
    MeaningProvider,
    NarrativeProvider,
)
from aurora.soul.facts import FactExtractor
from aurora.soul.models import (
    AxisSpec,
    DissonanceReport,
    EventFrame,
    IdentityMode,
    IdentitySnapshot,
    IdentityState,
    Message,
    NarrativeSummary,
    Plot,
    PsychologicalSchema,
    RetrievalTrace,
    Summary,
    StoryArc,
    TextPart,
    Theme,
    clamp,
    heuristic_persona_axes,
    l2_normalize,
    message_actors,
    messages_to_text,
    mean_abs,
    schema_from_profile,
)
from aurora.soul.retrieval import (
    FieldRetriever,
    LowRankMetric,
    MemoryGraph,
    OnlineKDE,
    VectorIndex,
)
from aurora.utils.id_utils import det_id
from aurora.utils.math_utils import cosine_sim
from aurora.utils.time_utils import now_ts

# 系统版本标识
SOUL_MEMORY_STATE_VERSION = "aurora-soul-memory-v6"


def moving_average(old: float, new: float, rate: float) -> float:
    """平滑移动平均，用于数值状态的渐进式更新"""
    return (1.0 - rate) * float(old) + rate * float(new)


def stable_uuid(prefix: str) -> str:
    """生成带前缀的唯一标识符"""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class SchemaConsolidator:
    """
    Schema 整合器：负责管理性格轴的演变，发现相似轴并进行合并。
    防止人设维度无限增长导致的性能下降和语义发散。
    """

    def __init__(self, every_events: int = 50, budget: int = 24):
        self.every_events = every_events
        self.budget = budget
        self.events_since_last = 0
        self.axis_texts: Dict[str, List[str]] = {}

    def observe(self, plot: Plot) -> None:
        """观察新情节，记录哪些文本激活了哪些轴，用于后续相似度判断"""
        for axis_name, value in plot.frame.axis_evidence.items():
            if abs(value) < 0.24:
                continue
            self.axis_texts.setdefault(axis_name, []).append(plot.semantic_text[:140])
            self.axis_texts[axis_name] = self.axis_texts[axis_name][-12:]
        self.events_since_last += 1

    def should_run(self, added_axis: bool, persona_count: int) -> bool:
        """判断是否需要运行合并检查"""
        return (
            added_axis or self.events_since_last >= self.every_events or persona_count > self.budget
        )

    def maybe_consolidate(
        self,
        *,
        schema: PsychologicalSchema,
        axis_embedder: TextEmbeddingProvider,
        narrator: NarrativeProvider,
    ) -> List[Tuple[str, str]]:
        """执行合并逻辑：对比轴的向量方向和激活文本，合并重合度高的轴"""
        merges: List[Tuple[str, str]] = []
        if len(schema.persona_axes) < 2:
            self.events_since_last = 0
            return merges

        while True:
            best_pair: Optional[Tuple[str, str]] = None
            best_score = -1.0
            names = list(schema.persona_axes.keys())
            for idx, left_name in enumerate(names):
                left = schema.persona_axes[left_name]
                if left.direction is None or left.positive_anchor is None:
                    continue
                for right_name in names[idx + 1 :]:
                    right = schema.persona_axes[right_name]
                    if right.direction is None or right.positive_anchor is None:
                        continue
                    # 综合评分：方向相似度 + 锚点相似度 + 激活文本重合度
                    direction_sim = cosine_sim(left.direction, right.direction)
                    anchor_sim = cosine_sim(left.positive_anchor, right.positive_anchor)
                    overlap_set = set(self.axis_texts.get(left.name, [])) & set(
                        self.axis_texts.get(right.name, [])
                    )
                    score = (
                        0.55 * direction_sim + 0.35 * anchor_sim + 0.10 * min(len(overlap_set), 3)
                    )
                    if score > best_score:
                        best_score = score
                        best_pair = (left_name, right_name)

            if best_pair is None:
                break

            left_name, right_name = best_pair
            left = schema.persona_axes[left_name]
            right = schema.persona_axes[right_name]
            overlap = list(
                set(self.axis_texts.get(left.name, [])) & set(self.axis_texts.get(right.name, []))
            )
            # 最终合并决策由 narrator (LLM) 做出
            should_merge, reason = narrator.judge_axis_merge(left, right, overlap)
            # 如果超出预算且相似度极高，强制合并
            forced_by_budget = len(schema.persona_axes) > self.budget and best_score >= 0.82
            if not should_merge and not forced_by_budget:
                break

            # 执行合并，保留支持度高的作为规范名
            canonical, alias = (
                (left, right) if left.support_count >= right.support_count else (right, left)
            )
            schema.merge_persona_axes(
                canonical.name, alias.name, note=reason or f"similarity={best_score:.3f}"
            )
            schema.persona_axes[canonical.name].compile(axis_embedder)
            merges.append((canonical.name, alias.name))
            if len(schema.persona_axes) <= self.budget and best_score < 0.90:
                break

        self.events_since_last = 0
        return merges

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "every_events": self.every_events,
            "budget": self.budget,
            "events_since_last": self.events_since_last,
            "axis_texts": {key: list(value) for key, value in self.axis_texts.items()},
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "SchemaConsolidator":
        """反序列化"""
        obj = cls(
            every_events=int(data.get("every_events", 50)),
            budget=int(data.get("budget", 24)),
        )
        obj.events_since_last = int(data.get("events_since_last", 0))
        obj.axis_texts = {
            str(key): [str(item) for item in value]
            for key, value in data.get("axis_texts", {}).items()
        }
        return obj


@dataclass(frozen=True)
class SoulConfig:
    """灵魂引擎配置类：控制维度、阈值和蓄水池容量"""

    dim: int = 384  # 向量维度
    metric_rank: int = 64  # 低秩度量学习的秩
    architecture_mode: Literal["graph_first"] = "graph_first"
    max_plots: int = 5000  # 最大情节存储量
    kde_reservoir: int = 4096  # KDE 采样蓄水池大小
    retrieval_kinds: Tuple[str, ...] = ("summary", "theme", "story", "plot")  # 检索类型
    max_recent_semantic_texts: int = 12  # 最近语义投影历史保留数
    profile_text: str = ""  # 初始人设文本
    persona_axes_json: Optional[str] = None  # 预设人设轴 JSON
    axis_merge_every_events: int = 50  # 轴合并检查频率
    persona_axis_budget: int = 24  # 人设轴数量预算
    graph_temporal_neighbors: int = 2
    graph_semantic_neighbors: int = 3
    graph_contradiction_neighbors: int = 10
    graph_similarity_threshold: float = 0.2
    graph_contradiction_threshold: float = 0.16
    community_refresh_every_plots: int = 50
    dream_walk_steps: int = 6
    dream_walk_samples: int = 24
    dream_persist_threshold: float = 0.18

    def __post_init__(self) -> None:
        if self.architecture_mode != "graph_first":
            raise ValueError(
                "Aurora no longer supports alternate architecture modes; "
                "'graph_first' is the only supported mode."
            )

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "dim": self.dim,
            "metric_rank": self.metric_rank,
            "architecture_mode": self.architecture_mode,
            "max_plots": self.max_plots,
            "kde_reservoir": self.kde_reservoir,
            "retrieval_kinds": list(self.retrieval_kinds),
            "max_recent_semantic_texts": self.max_recent_semantic_texts,
            "profile_text": self.profile_text,
            "persona_axes_json": self.persona_axes_json,
            "axis_merge_every_events": self.axis_merge_every_events,
            "persona_axis_budget": self.persona_axis_budget,
            "graph_temporal_neighbors": self.graph_temporal_neighbors,
            "graph_semantic_neighbors": self.graph_semantic_neighbors,
            "graph_contradiction_neighbors": self.graph_contradiction_neighbors,
            "graph_similarity_threshold": self.graph_similarity_threshold,
            "graph_contradiction_threshold": self.graph_contradiction_threshold,
            "community_refresh_every_plots": self.community_refresh_every_plots,
            "dream_walk_steps": self.dream_walk_steps,
            "dream_walk_samples": self.dream_walk_samples,
            "dream_persist_threshold": self.dream_persist_threshold,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "SoulConfig":
        """反序列化"""
        architecture_mode = str(data.get("architecture_mode", "graph_first"))
        if architecture_mode != "graph_first":
            raise ValueError(
                f"Unsupported snapshot architecture_mode: {architecture_mode!r}. "
                "Supported mode is 'graph_first'."
            )
        return cls(
            dim=int(data.get("dim", 384)),
            metric_rank=int(data.get("metric_rank", 64)),
            architecture_mode="graph_first",
            max_plots=int(data.get("max_plots", 5000)),
            kde_reservoir=int(data.get("kde_reservoir", 4096)),
            retrieval_kinds=tuple(
                data.get("retrieval_kinds", ("summary", "theme", "story", "plot"))
            ),
            max_recent_semantic_texts=int(data.get("max_recent_semantic_texts", 12)),
            profile_text=str(data.get("profile_text", "")),
            persona_axes_json=data.get("persona_axes_json"),
            axis_merge_every_events=int(data.get("axis_merge_every_events", 50)),
            persona_axis_budget=int(data.get("persona_axis_budget", 24)),
            graph_temporal_neighbors=int(data.get("graph_temporal_neighbors", 2)),
            graph_semantic_neighbors=int(data.get("graph_semantic_neighbors", 3)),
            graph_contradiction_neighbors=int(data.get("graph_contradiction_neighbors", 10)),
            graph_similarity_threshold=float(data.get("graph_similarity_threshold", 0.2)),
            graph_contradiction_threshold=float(data.get("graph_contradiction_threshold", 0.16)),
            community_refresh_every_plots=int(data.get("community_refresh_every_plots", 50)),
            dream_walk_steps=int(data.get("dream_walk_steps", 6)),
            dream_walk_samples=int(data.get("dream_walk_samples", 24)),
            dream_persist_threshold=float(data.get("dream_persist_threshold", 0.18)),
        )


@dataclass
class GraphProjectionState:
    graph: MemoryGraph
    vindex: VectorIndex
    retriever: FieldRetriever
    plots: Dict[str, Plot]
    summaries: Dict[str, Summary]
    stories: Dict[str, StoryArc]
    themes: Dict[str, Theme]
    anchor_nodes: Dict[str, Dict[str, Any]]
    core_anchor_ids: List[str]
    view_stats: GraphViewStats


class AuroraSoul:
    """
    数字灵魂引擎主类：
    负责记忆的生命周期管理（Ingest -> Store -> Query -> Evolve）。
    驱动身份动力学模拟（Dissonance -> Repair -> Mode Transition）。
    """

    def __init__(
        self,
        cfg: SoulConfig = SoulConfig(),
        *,
        seed: int = 0,
        event_embedder: Optional[ContentEmbeddingProvider] = None,
        axis_embedder: Optional[TextEmbeddingProvider] = None,
        meaning_provider: Optional[MeaningProvider] = None,
        narrator: Optional[NarrativeProvider] = None,
        query_analyzer: Optional[BaseQueryAnalyzer] = None,
        bootstrap: bool = True,
    ) -> None:
        self.cfg = cfg
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # 基础设施初始化
        self.event_embedder = event_embedder or LocalSemanticEmbedding(dim=cfg.dim, seed=seed)
        self.axis_embedder = axis_embedder or self.event_embedder
        self.meaning_provider = meaning_provider or HeuristicMeaningProvider()
        self.narrator = narrator or CombinatorialNarrativeProvider()
        self.fact_extractor = FactExtractor()

        # 加载初始人设轴并初始化 Schema
        if bootstrap:
            persona_axes = self._load_persona_axes()
            self.schema = schema_from_profile(
                axis_embedder=self.axis_embedder,
                profile_text=cfg.profile_text,
                persona_axes=persona_axes,
            )
        else:
            self.schema = PsychologicalSchema(profile_text=cfg.profile_text)

        # 核心算法模块初始化
        self.kde = OnlineKDE(dim=cfg.dim, reservoir=cfg.kde_reservoir, seed=seed)
        self.metric = LowRankMetric(dim=cfg.dim, rank=cfg.metric_rank, seed=seed)
        self.graph = MemoryGraph()
        self.vindex = VectorIndex(dim=cfg.dim)
        self.retriever = FieldRetriever(
            metric=self.metric,
            vindex=self.vindex,
            graph=self.graph,
            query_analyzer=query_analyzer or MissingQueryAnalyzer(),
        )
        self.consolidator = SchemaConsolidator(
            every_events=cfg.axis_merge_every_events,
            budget=cfg.persona_axis_budget,
        )

        # 状态数据初始化
        self.plots: Dict[str, Plot] = {}
        self.summaries: Dict[str, Summary] = {}
        self.stories: Dict[str, StoryArc] = {}
        self.themes: Dict[str, Theme] = {}
        self.modes: Dict[str, IdentityMode] = {}
        self.recent_semantic_texts: List[str] = []
        self.step = 0

        # 性格统计量初始化
        # 身份状态初始化
        if bootstrap:
            self.identity = self._make_initial_identity()
            self._init_origin_mode()
        else:
            self.identity = IdentityState(
                self_vector=np.zeros(self.cfg.dim, dtype=np.float32),
                axis_state={},
                intuition_axes={},
                current_mode_label="origin",
            )
        self.view_builder = GraphViewBuilder(seed=seed)
        self.dream_operator = GraphDreamOperator(seed=seed)
        self.repair_operator = GraphRepairOperator()
        self.anchor_nodes: Dict[str, Dict[str, Any]] = {}
        self.core_anchor_ids: List[str] = []
        self.view_stats = GraphViewStats()
        self.graph_metrics: Dict[str, Any] = {}
        self._bootstrap_core_anchors()
        self._refresh_graph_metrics()

    def _load_persona_axes(self) -> List[Dict[str, Any]]:
        """加载或提取人设轴"""
        if self.cfg.persona_axes_json:
            try:
                payload = json.loads(self.cfg.persona_axes_json)
                if isinstance(payload, list):
                    return payload
            except Exception:
                pass
        # 尝试通过 provider 智能提取
        bootstrapper = getattr(self.meaning_provider, "bootstrap_persona_axes", None)
        if callable(bootstrapper):
            extracted = cast(List[Dict[str, Any]], bootstrapper(self.cfg.profile_text))
        else:
            extracted = self.meaning_provider.extract_persona_axes(self.cfg.profile_text)
        if extracted:
            return extracted
        # 兜底：启发式提取
        return heuristic_persona_axes(self.cfg.profile_text)

    def _make_initial_identity(self) -> IdentityState:
        """初始化身份状态，设定初始轴坐标和自我向量"""
        axis_state = {name: 0.0 for name in self.schema.ordered_axis_names()}
        # 赋予人设轴一个微小的初始正向偏移
        for name in self.schema.persona_axes.keys():
            axis_state[name] = 0.18
        axis_state["coherence"] = 0.15
        axis_state["regulation"] = 0.10
        # 根据轴得分合成自我向量
        self_vector = self._compose_self_vector(axis_state)
        state = IdentityState(
            self_vector=self_vector,
            axis_state=axis_state,
            intuition_axes={name: 0.0 for name in axis_state},
            current_mode_label="origin",
        )
        state.narrative_log.append("Initialized v6 multimodal soul.")
        return state

    def _compose_self_vector(self, axis_state: Dict[str, float]) -> np.ndarray:
        """核心计算：将各个低维心理轴的得分按方向合成高维空间的“自我向量”"""
        accumulator = np.zeros(self.cfg.dim, dtype=np.float32)
        for axis_name, value in axis_state.items():
            axis = self.schema.all_axes().get(axis_name)
            if axis is None or axis.direction is None:
                continue
            # 向量合成：方向 * 强度
            accumulator += float(value) * axis.direction
        # 如果没有显著偏移，使用 Profile 文本的 Embedding 作为基准
        if np.linalg.norm(accumulator) < 1e-6:
            accumulator += self.axis_embedder.embed_text(self.cfg.profile_text or "origin self")
        return l2_normalize(accumulator)

    def _axis_names(self) -> List[str]:
        """维护当前 Schema 中的轴名称缓存，并初始化缺失的状态位"""
        names = self.schema.ordered_axis_names()
        for name in names:
            self.identity.axis_state.setdefault(name, 0.0)
            self.identity.intuition_axes.setdefault(name, 0.0)
        return names

    def _init_origin_mode(self) -> None:
        """初始化原始身份模式"""
        mode_id = stable_uuid("mode")
        axis_proto = {key: float(value) for key, value in self.identity.axis_state.items()}
        bootstrap_label = getattr(self.narrator, "bootstrap_mode_label", None)
        if callable(bootstrap_label):
            label = bootstrap_label(axis_proto, self.schema, support=1)
        else:
            label = self.narrator.label_mode(axis_proto, self.schema, support=1)
        mode = IdentityMode(
            id=mode_id,
            label=label,
            prototype=self.identity.self_vector.copy(),
            axis_prototype=axis_proto,
        )
        self.modes[mode_id] = mode
        self.identity.current_mode_id = mode_id
        self.identity.current_mode_label = label
        self.identity.narrative_log.append(f"Initialized mode: {label}")

    def _plot_vector_for_index(self, plot: Plot) -> np.ndarray:
        """计算情节在向量索引中的实际位置：混合原始语义向量和心理学含义向量"""
        implication = np.zeros(self.cfg.dim, dtype=np.float32)
        for axis_name, value in plot.frame.axis_evidence.items():
            axis = self.schema.all_axes().get(axis_name)
            if axis is None or axis.direction is None:
                continue
            implication += float(value) * axis.direction
        if np.linalg.norm(implication) < 1e-6:
            return plot.embedding
        # 权重混合：72% 原始语义 + 28% 心理内涵
        return l2_normalize(0.72 * plot.embedding + 0.28 * implication)

    def _story_vector_for_index(self, story: StoryArc) -> np.ndarray:
        if story.centroid is None:
            return np.zeros(self.cfg.dim, dtype=np.float32)
        return story.centroid

    def _theme_vector_for_index(self, theme: Theme) -> np.ndarray:
        if theme.prototype is None:
            return np.zeros(self.cfg.dim, dtype=np.float32)
        return theme.prototype

    def _bootstrap_core_anchors(self) -> None:
        """在图中钉住少量核心自我锚点，供 graph-first 模式使用。"""
        if self.core_anchor_ids:
            for anchor_id, payload in self.anchor_nodes.items():
                self.graph.add_node(anchor_id, "anchor", payload)
            return
        label = self.cfg.profile_text.strip() or "origin self"
        embedding = (
            self.identity.self_vector.copy()
            if np.linalg.norm(self.identity.self_vector) > 1e-6
            else self.axis_embedder.embed_text(label)
        )
        anchor_id = "anchor_core_self"
        payload = {
            "id": anchor_id,
            "label": label,
            "embedding": l2_normalize(embedding.astype(np.float32)),
            "pinned": True,
        }
        self.anchor_nodes[anchor_id] = payload
        self.core_anchor_ids = [anchor_id]
        self.graph.add_node(anchor_id, "anchor", payload)

    def _anchor_centroid(self) -> Optional[np.ndarray]:
        vecs = [
            cast(np.ndarray, payload.get("embedding"))
            for anchor_id, payload in self.anchor_nodes.items()
            if anchor_id in self.core_anchor_ids and isinstance(payload.get("embedding"), np.ndarray)
        ]
        if not vecs:
            return None
        return l2_normalize(np.mean(np.asarray(vecs, dtype=np.float32), axis=0))

    def _refresh_identity_from_anchors(self) -> None:
        """在自我向量过弱时，回退到核心 anchor centroid。"""
        centroid = self._anchor_centroid()
        if centroid is None:
            return
        if np.linalg.norm(self.identity.self_vector) < 1e-6:
            self.identity.self_vector = centroid.astype(np.float32)

    def _refresh_graph_metrics(self) -> None:
        previous = self.graph_metrics if isinstance(self.graph_metrics, dict) else {}
        authoritative_fresh = (
            self.view_stats.graph_edge_version >= 0
            and self.view_stats.graph_edge_version == self.graph.edge_version
        )
        metrics: Dict[str, Any] = {
            "authoritative": {
                "plot_count": len(self.plots),
                "summary_count": len(self.summaries),
                "story_count": len(self.stories),
                "theme_count": len(self.themes),
                "core_anchor_count": len(self.core_anchor_ids),
                "graph_edge_version": self.graph.edge_version,
                "view_refreshed_step": int(self.view_stats.refreshed_step),
                "view_fresh": bool(authoritative_fresh),
            },
            "last_plot": dict(previous.get("last_plot", {})),
            "query": dict(previous.get("query", {})),
            "evolve": dict(previous.get("evolve", {})),
        }
        self.graph_metrics = metrics

    def _refresh_projection_views(self, projection: GraphProjectionState, *, force: bool = False) -> None:
        if not projection.plots:
            projection.stories = {}
            projection.themes = {}
            projection.view_stats = GraphViewStats(
                graph_edge_version=projection.graph.edge_version,
                refreshed_step=int(self.step),
                plot_count=0,
                story_count=0,
                theme_count=0,
            )
            return
        stale = (
            projection.view_stats.graph_edge_version != projection.graph.edge_version
            or projection.view_stats.plot_count != len(projection.plots)
        )
        if not force and not stale:
            return
        stories, themes, stats = self.view_builder.build(
            graph=projection.graph,
            vindex=projection.vindex,
            plots=projection.plots,
            previous_stories=projection.stories,
            previous_themes=projection.themes,
            step=self.step,
        )
        projection.stories = stories
        projection.themes = themes
        projection.view_stats = stats

    def _refresh_materialized_views(self, *, force: bool = False) -> None:
        """按需物化 story/theme 视图。"""
        projection = GraphProjectionState(
            graph=self.graph,
            vindex=self.vindex,
            retriever=self.retriever,
            plots=self.plots,
            summaries=self.summaries,
            stories=self.stories,
            themes=self.themes,
            anchor_nodes=self.anchor_nodes,
            core_anchor_ids=self.core_anchor_ids,
            view_stats=self.view_stats,
        )
        self._refresh_projection_views(projection, force=force)
        self.stories = projection.stories
        self.themes = projection.themes
        self.view_stats = projection.view_stats
        self._refresh_graph_metrics()

    def _plot_semantic_hits(
        self,
        embedding: np.ndarray,
        *,
        exclude_id: Optional[str] = None,
        limit: Optional[int] = None,
        vindex: Optional[VectorIndex] = None,
        plots: Optional[Dict[str, Plot]] = None,
    ) -> List[Tuple[str, float]]:
        search_k = max((limit or self.cfg.graph_contradiction_neighbors) + 4, 8)
        index = self.vindex if vindex is None else vindex
        plot_map = self.plots if plots is None else plots
        hits = index.search(embedding, k=search_k, kind="plot")
        ranked: List[Tuple[str, float]] = []
        for node_id, score in hits:
            if exclude_id is not None and node_id == exclude_id:
                continue
            if node_id not in plot_map:
                continue
            ranked.append((node_id, float(score)))
        if limit is not None:
            return ranked[:limit]
        return ranked

    def _graph_contradiction_score(self, left: Plot, right: Plot) -> float:
        axis_names = set(left.frame.axis_evidence.keys()) | set(right.frame.axis_evidence.keys())
        axis_conflicts: List[float] = []
        for axis_name in axis_names:
            left_value = float(left.frame.axis_evidence.get(axis_name, 0.0))
            right_value = float(right.frame.axis_evidence.get(axis_name, 0.0))
            axis_conflicts.append(max(0.0, -(left_value * right_value)))
        axis_conflict = float(np.mean(axis_conflicts)) if axis_conflicts else 0.0
        semantic_conflict = max(0.0, -cosine_sim(left.embedding, right.embedding))
        affect_conflict = max(0.0, -(left.frame.valence * right.frame.valence))
        return float(0.55 * axis_conflict + 0.30 * semantic_conflict + 0.15 * affect_conflict)

    def _connect_plot_to_anchors(
        self,
        plot: Plot,
        *,
        graph: Optional[MemoryGraph] = None,
        anchor_nodes: Optional[Dict[str, Dict[str, Any]]] = None,
        core_anchor_ids: Optional[Sequence[str]] = None,
    ) -> None:
        target_graph = self.graph if graph is None else graph
        anchors = self.anchor_nodes if anchor_nodes is None else anchor_nodes
        anchor_ids = self.core_anchor_ids if core_anchor_ids is None else core_anchor_ids
        for anchor_id in anchor_ids:
            payload = anchors.get(anchor_id)
            if payload is None:
                continue
            anchor_emb = cast(np.ndarray, payload.get("embedding"))
            sim = cosine_sim(anchor_emb, plot.embedding)
            if sim >= self.cfg.graph_similarity_threshold * 0.5:
                target_graph.ensure_edge(
                    anchor_id,
                    plot.id,
                    "anchors",
                    sign=1,
                    weight=max(0.05, sim),
                    confidence=plot.confidence,
                    provenance="anchor_similarity",
                )
                target_graph.ensure_edge(
                    plot.id,
                    anchor_id,
                    "anchored_by",
                    sign=1,
                    weight=max(0.05, sim),
                    confidence=plot.confidence,
                    provenance="anchor_similarity",
                )
                continue
            if sim <= -self.cfg.graph_contradiction_threshold:
                weight = max(0.05, abs(sim))
                target_graph.ensure_edge(
                    anchor_id,
                    plot.id,
                    "contradicts_self",
                    sign=-1,
                    weight=weight,
                    confidence=plot.confidence,
                    provenance="anchor_similarity",
                )
                target_graph.ensure_edge(
                    plot.id,
                    anchor_id,
                    "contradicts_self",
                    sign=-1,
                    weight=weight,
                    confidence=plot.confidence,
                    provenance="anchor_similarity",
                )

    def _store_plot_graph_first(
        self,
        plot: Plot,
    ) -> None:
        """graph-first: 只写 plot 节点和局部显式边。"""
        graph = self.graph
        vindex = self.vindex
        plots = self.plots
        anchor_nodes = self.anchor_nodes
        core_anchor_ids = self.core_anchor_ids
        plots[plot.id] = plot
        graph.add_node(plot.id, "plot", plot)
        vindex.add(plot.id, self._plot_vector_for_index(plot), kind="plot")

        previous_plots = sorted(
            (item for item in plots.values() if item.id != plot.id),
            key=lambda item: item.ts,
            reverse=True,
        )[: self.cfg.graph_temporal_neighbors]
        for prev in previous_plots:
            graph.ensure_edge(
                prev.id,
                plot.id,
                "precedes",
                weight=0.8,
                confidence=min(prev.confidence, plot.confidence),
                provenance="temporal",
            )
            graph.ensure_edge(
                plot.id,
                prev.id,
                "follows",
                weight=0.8,
                confidence=min(prev.confidence, plot.confidence),
                provenance="temporal",
            )

        semantic_hits = self._plot_semantic_hits(
            self._plot_vector_for_index(plot),
            exclude_id=plot.id,
            limit=self.cfg.graph_contradiction_neighbors,
            vindex=vindex,
            plots=plots,
        )
        for neighbor_id, score in semantic_hits[: self.cfg.graph_semantic_neighbors]:
            if score < self.cfg.graph_similarity_threshold:
                continue
            weight = max(0.05, score)
            graph.ensure_edge(
                plot.id,
                neighbor_id,
                "associates",
                weight=weight,
                confidence=plot.confidence,
                provenance="semantic_neighbor",
            )
            graph.ensure_edge(
                neighbor_id,
                plot.id,
                "associates",
                weight=weight,
                confidence=plots[neighbor_id].confidence,
                provenance="semantic_neighbor",
            )

        for neighbor_id, _score in semantic_hits[: self.cfg.graph_contradiction_neighbors]:
            neighbor = plots.get(neighbor_id)
            if neighbor is None:
                continue
            contradiction_score = self._graph_contradiction_score(plot, neighbor)
            if contradiction_score < self.cfg.graph_contradiction_threshold:
                continue
            graph.ensure_edge(
                plot.id,
                neighbor_id,
                "contradicts",
                sign=-1,
                weight=contradiction_score,
                confidence=plot.confidence,
                provenance="bounded_scan",
            )
            graph.ensure_edge(
                neighbor_id,
                plot.id,
                "contradicts",
                sign=-1,
                weight=contradiction_score,
                confidence=neighbor.confidence,
                provenance="bounded_scan",
            )

        self._connect_plot_to_anchors(
            plot,
            graph=graph,
            anchor_nodes=anchor_nodes,
            core_anchor_ids=core_anchor_ids,
        )

    def _store_summary(self, summary: Summary) -> None:
        self.summaries[summary.id] = summary
        self.graph.add_node(summary.id, "summary", summary)
        if summary.embedding.size == self.cfg.dim:
            self.vindex.add(summary.id, l2_normalize(summary.embedding), kind="summary")
        self._refresh_graph_metrics()

    def remove_plot_from_hot_state(self, plot_id: str) -> None:
        plot = self.plots.pop(plot_id, None)
        if plot is None:
            return
        self.vindex.remove(plot_id)
        self.graph.remove_node(plot_id)
        for story in self.stories.values():
            if plot_id in story.plot_ids:
                story.plot_ids = [item for item in story.plot_ids if item != plot_id]
        self._refresh_graph_metrics()

    def _redundancy(self, emb: np.ndarray) -> float:
        """计算输入向量相对于现有情节的冗余度（最高余弦相似度）"""
        hits = self.vindex.search(emb, k=8, kind="plot")
        return max((score for _, score in hits), default=0.0)

    def _goal_relevance(self, emb: np.ndarray, context_emb: Optional[np.ndarray]) -> float:
        """计算输入相对于当前目标上下文的相关性"""
        if context_emb is None:
            return 0.0
        return float(np.dot(l2_normalize(emb), l2_normalize(context_emb)))

    def _pred_error(self, emb: np.ndarray) -> float:
        """计算预测误差（相对于所有已知故事弧重心的最小距离）"""
        best_story: Optional[StoryArc] = None
        best_sim = -1.0
        for story in self.stories.values():
            if story.centroid is None:
                continue
            sim = self.metric.sim(emb, story.centroid)
            if sim > best_sim:
                best_story = story
                best_sim = sim
        if best_story is None:
            return 1.0
        return 1.0 - best_sim

    def _compute_dissonance(self, frame: EventFrame, embedding: np.ndarray) -> DissonanceReport:
        """
        核心认知计算：计算新输入与当前身份状态的失调程度 (Dissonance)。
        """
        axis_conflicts: Dict[str, float] = {}
        axis_alignments: Dict[str, float] = {}
        for axis_name in self._axis_names():
            current = self.identity.axis_state.get(axis_name, 0.0)
            evidence = frame.axis_evidence.get(axis_name, 0.0)
            # 符号相反且得分大则代表冲突
            axis_conflicts[axis_name] = max(0.0, -current * evidence)
            axis_alignments[axis_name] = max(0.0, current * evidence)

        # 高维空间冲突
        semantic_conflict = (
            max(0.0, -float(np.dot(self.identity.self_vector, embedding))) * frame.self_relevance
        )
        # 情感负荷：基于唤醒度、负效价以及威胁/羞耻等信号
        affective_load = (
            0.30 * frame.arousal
            + 0.20 * max(0.0, -frame.valence)
            + 0.18 * frame.threat
            + 0.12 * frame.shame
            + 0.10 * frame.control
            + 0.10 * frame.abandonment
        )
        # 叙事不一致性
        narrative_incongruity = mean_abs(axis_conflicts.values()) * (
            0.55 + 0.45 * frame.self_relevance
        )
        # 综合总分计算
        total = (
            0.36 * mean_abs(axis_conflicts.values())
            + 0.20 * semantic_conflict
            + 0.28 * affective_load
            + 0.16 * narrative_incongruity
        )
        return DissonanceReport(
            axis_conflicts=axis_conflicts,
            axis_alignments=axis_alignments,
            semantic_conflict=semantic_conflict,
            affective_load=affective_load,
            narrative_incongruity=narrative_incongruity,
            total=float(total),
        )

    def _baseline_assimilation(self, frame: EventFrame) -> None:
        """
        基础同化：即便没有显著重构，日常交互也会导致性格轴的微小漂移。
        模拟了心理学中的渐进式性格改变。
        """
        changed = False
        for axis_name in self._axis_names():
            evidence = frame.axis_evidence.get(axis_name, 0.0)
            if abs(evidence) < 0.06:
                continue
            axis = self.schema.all_axes().get(axis_name)
            # 稳态轴同化速度较快，人设轴较慢（更稳定）
            rate = 0.015 if (axis is not None and axis.level == "persona") else 0.025
            rate *= 0.55 + 0.45 * frame.self_relevance
            self.identity.axis_state[axis_name] = moving_average(
                self.identity.axis_state.get(axis_name, 0.0),
                evidence,
                rate,
            )
            changed = True

        # 情感反馈通道的固定调节逻辑
        self.identity.axis_state["regulation"] = clamp(
            self.identity.axis_state.get("regulation", 0.0)
            + 0.03 * frame.care
            - 0.04 * frame.arousal
            - 0.03 * frame.threat
        )
        self.identity.axis_state["vigilance"] = clamp(
            self.identity.axis_state.get("vigilance", 0.0) + 0.03 * frame.threat - 0.02 * frame.care
        )
        self.identity.axis_state["coherence"] = clamp(
            self.identity.axis_state.get("coherence", 0.0) - 0.02 * frame.shame + 0.01 * frame.care
        )
        changed = True
        if changed:
            # 状态更新后重算自我向量
            self.identity.self_vector = self._compose_self_vector(self.identity.axis_state)

    def _maybe_expand_schema(self, plot: Plot) -> bool:
        """Schema 演化：如果情节中出现了反复提及且未被解释的标签，则自动创建新的性格轴"""
        if plot.source != "wake" or plot.tension < 0.75:
            return False
        existing = set(self.schema.all_axes().keys())
        for tag in plot.frame.tags:
            # 过滤内部标签
            if ":" in tag or tag.startswith("echo:") or tag in existing:
                continue
            # 创建新轴规格
            axis = AxisSpec(
                name=tag[:24],
                positive_pole=tag,
                negative_pole=f"not_{tag}",
                description=f"Discovered from repeated unresolved motif: {tag}",
                level="persona",
            )
            if axis.name in self.schema.persona_axes:
                continue
            self.schema.add_persona_axis(axis, self.axis_embedder)
            # 初始化新轴状态
            self.identity.axis_state.setdefault(axis.name, 0.0)
            self.identity.intuition_axes.setdefault(axis.name, 0.0)
            self.identity.narrative_log.append(
                f"New axis emerged: {axis.name} ({axis.positive_pole} ↔ {axis.negative_pole})"
            )
            return True
        return False

    def _pressure_manage(self) -> None:
        """内存压力管理：基于记忆质量（Mass）删除那些最不重要/最古老的记忆"""
        if len(self.plots) <= self.cfg.max_plots:
            return
        # 挑选 Mass 最小的牺牲品
        victims = sorted(self.plots.values(), key=lambda plot: plot.mass())[
            : max(1, len(self.plots) - self.cfg.max_plots)
        ]
        for victim in victims:
            victim.status = "archived"
            self.vindex.remove(victim.id)

    def _graph_first_repairs(self, limit: int = 1) -> List[Plot]:
        """扫描 anchor-contradiction 组件并生成 resolution plots。"""
        targets = self.repair_operator.find_targets(
            graph=self.graph,
            plots=self.plots,
            anchor_ids=self.core_anchor_ids,
            limit=limit,
        )
        repairs: List[Plot] = []
        for target in targets:
            target_plots = [self.plots[plot_id] for plot_id in target.plot_ids if plot_id in self.plots]
            if not target_plots:
                continue
            salient_axes = sorted(
                (
                    (
                        axis_name,
                        float(
                            np.mean(
                                [
                                    abs(plot.frame.axis_evidence.get(axis_name, 0.0))
                                    for plot in target_plots
                                ]
                            )
                        ),
                    )
                    for axis_name in self._axis_names()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            repair_text = self.narrator.compose_repair(
                "integrate",
                self.identity,
                self.identity,
                target.score,
                [axis_name for axis_name, score in salient_axes[:3] if score > 0.0],
                " / ".join(plot.semantic_text[:96] for plot in target_plots[:2]),
                self.schema,
            )
            repair_plot = self.ingest(
                (Message(role="self", parts=(TextPart(text=repair_text),), actor="self"),),
                source="repair",
                confidence=0.72,
                evidence_weight=0.35,
            )
            self.identity.repair_count += 1
            self.identity.narrative_log.append(repair_text)
            for plot_id in target.plot_ids:
                if plot_id not in self.plots:
                    continue
                self.graph.ensure_edge(
                    repair_plot.id,
                    plot_id,
                    "resolves",
                    weight=max(0.05, target.score),
                    confidence=repair_plot.confidence,
                    provenance="repair_operator",
                )
                self.graph.ensure_edge(
                    plot_id,
                    repair_plot.id,
                    "resolved_by",
                    weight=max(0.05, target.score),
                    confidence=self.plots[plot_id].confidence,
                    provenance="repair_operator",
                )
            repairs.append(repair_plot)
        return repairs

    def _graph_first_dreams(self, n: int = 2) -> List[Plot]:
        """基于随机游走的 graph dream operator。"""
        candidates = self.dream_operator.propose(
            graph=self.graph,
            plots=self.plots,
            samples=self.cfg.dream_walk_samples,
            steps=self.cfg.dream_walk_steps,
        )
        dreams: List[Plot] = []
        fallback = candidates[: max(1, n)] if candidates else []
        chosen = [
            candidate for candidate in candidates if candidate.score >= self.cfg.dream_persist_threshold
        ]
        if not chosen:
            chosen = fallback
        for candidate in chosen[:n]:
            operator = "integration" if candidate.resonance >= 0.35 else "counterfactual"
            dream_text = self.narrator.compose_dream(
                operator,
                candidate.tags,
                self.identity,
                self.schema,
            )
            dream_plot = self.ingest(
                (Message(role="self", parts=(TextPart(text=dream_text),), actor="self"),),
                source="dream",
                confidence=0.34,
                evidence_weight=0.20,
            )
            self.identity.dream_count += 1
            for plot_id in candidate.plot_ids:
                if plot_id not in self.plots:
                    continue
                self.graph.ensure_edge(
                    dream_plot.id,
                    plot_id,
                    "dreams_about",
                    weight=max(0.05, candidate.score),
                    confidence=dream_plot.confidence,
                    provenance="dream_operator",
                )
            dreams.append(dream_plot)
        return dreams

    def _ingest_graph_first(self, plot: Plot, frame: EventFrame, dissonance: DissonanceReport) -> Plot:
        """graph-first: 统一写入 plot 图节点，所有高层结构都降为派生视图或后台算子。"""
        self._store_plot_graph_first(plot)

        energy_scale = {"wake": 1.0, "dream": 0.35, "repair": 0.20, "mode": 0.15}[plot.source]
        self.identity.active_energy += energy_scale * (
            0.38 * dissonance.total
            + 0.26 * frame.arousal
            + 0.16 * max(0.0, -frame.valence)
            + 0.10 * frame.threat
        )
        self.identity.contradiction_ema = moving_average(
            self.identity.contradiction_ema, dissonance.total, 0.24
        )

        if plot.source == "wake":
            self._baseline_assimilation(frame)
            added_axis = self._maybe_expand_schema(plot)
            self.consolidator.observe(plot)
            if self.consolidator.should_run(added_axis, len(self.schema.persona_axes)):
                merges = self.consolidator.maybe_consolidate(
                    schema=self.schema,
                    axis_embedder=self.axis_embedder,
                    narrator=self.narrator,
                )
                for canonical, alias in merges:
                    self.identity.narrative_log.append(f"Axis merged: {alias} -> {canonical}")
                self._axis_names()
            self._refresh_identity_from_anchors()

        self.recent_semantic_texts.append(plot.semantic_text)
        self.recent_semantic_texts = self.recent_semantic_texts[
            -self.cfg.max_recent_semantic_texts :
        ]
        self._pressure_manage()
        self.graph_metrics["last_plot"] = {
            "id": plot.id,
            "source": plot.source,
            "tension": round(float(plot.tension), 4),
            "contradiction": round(float(plot.contradiction), 4),
        }
        self._refresh_graph_metrics()
        return plot

    def ingest(
        self,
        messages: Sequence[Message],
        *,
        event_id: Optional[str] = None,
        context_messages: Optional[Sequence[Message]] = None,
        source: Literal["wake", "dream", "repair", "mode"] = "wake",
        confidence: float = 1.0,
        evidence_weight: float = 1.0,
        ts: Optional[float] = None,
        plot_id: Optional[str] = None,
    ) -> Plot:
        """
        摄入流程入口：将文本输入处理为记忆情节并驱动身份演变。
        """
        self.step += 1
        event_ts = ts or now_ts()
        embedding = self.event_embedder.embed_content(messages)
        meaning = self.meaning_provider.extract(
            messages,
            embedding,
            self.schema,
            recent_tags=self.recent_semantic_texts,
        )
        frame = meaning.frame
        plot = Plot(
            id=plot_id or str(uuid.uuid4()),
            event_id=event_id,
            ts=event_ts,
            messages=tuple(messages),
            semantic_text=meaning.semantic_text,
            actors=message_actors(messages),
            embedding=embedding,
            frame=frame,
            source=source,
            confidence=confidence,
            evidence_weight=evidence_weight,
            fact_keys=[fact.fact_text for fact in self.fact_extractor.extract(meaning.semantic_text)],
        )

        # 动力学指标计算
        context_emb = (
            self.event_embedder.embed_content(context_messages) if context_messages else None
        )
        plot.surprise = float(self.kde.surprise(embedding))
        plot.pred_error = float(self._pred_error(embedding))
        plot.redundancy = float(self._redundancy(embedding))
        plot.goal_relevance = float(self._goal_relevance(embedding, context_emb))

        # 失调与张力计算
        dissonance = self._compute_dissonance(frame, embedding)
        plot.contradiction = float(dissonance.total)
        plot.tension = float(
            (
                0.30 * plot.surprise
                + 0.20 * plot.pred_error
                + 0.20 * plot.contradiction
                + 0.15 * frame.arousal
                + 0.10 * frame.self_relevance
                + 0.05 * max(0.0, 1.0 - plot.redundancy)
            )
            * evidence_weight
        )

        return self._ingest_graph_first(plot, frame, dissonance)

    def refresh_materialized_views(self) -> None:
        self._refresh_materialized_views(force=True)

    def project_interaction_event(
        self,
        *,
        event_id: str,
        messages: Sequence[Message],
        ts: float,
    ) -> Plot:
        plot_id = det_id("plot", event_id)
        existing = self.plots.get(plot_id)
        if existing is not None:
            return existing
        return self.ingest(
            messages,
            event_id=event_id,
            source="wake",
            ts=ts,
            plot_id=plot_id,
        )

    def apply_projected_event(self, *, event_type: str, event_id: str, payload: Dict[str, Any]) -> Optional[str]:
        if event_type == "interaction":
            plot = self.project_interaction_event(
                event_id=event_id,
                messages=tuple(
                    Message.from_state_dict(item)
                    for item in cast(Sequence[Dict[str, Any]], payload.get("messages", ()))
                ),
                ts=float(payload.get("ts", now_ts())),
            )
            return plot.id
        if event_type in {"dream", "repair"}:
            return self._apply_generated_plot_event(event_type=event_type, event_id=event_id, payload=payload)
        if event_type == "compaction":
            summary = self._apply_compaction_event(event_id=event_id, payload=payload)
            return summary.id
        raise ValueError(f"Unsupported projected event type: {event_type}")

    def _apply_generated_plot_event(
        self,
        *,
        event_type: str,
        event_id: str,
        payload: Dict[str, Any],
    ) -> str:
        source = cast(Literal["dream", "repair"], event_type)
        plot_id = det_id("plot", event_id)
        existing = self.plots.get(plot_id)
        if existing is not None:
            return existing.id
        confidence = 0.34 if source == "dream" else 0.72
        evidence_weight = 0.20 if source == "dream" else 0.35
        plot = self.ingest(
            (Message(role="self", parts=(TextPart(text=str(payload.get("text", ""))),), actor="self"),),
            event_id=event_id,
            source=source,
            confidence=confidence,
            evidence_weight=evidence_weight,
            ts=float(payload.get("ts", now_ts())),
            plot_id=plot_id,
        )
        source_plot_ids = [str(item) for item in payload.get("source_plot_ids", [])]
        link_type = "dreams_about" if source == "dream" else "resolves"
        reverse_type = "dreamed_by" if source == "dream" else "resolved_by"
        link_weight = max(0.05, float(payload.get("score", 0.1)))
        for source_plot_id in source_plot_ids:
            if source_plot_id not in self.plots:
                continue
            self.graph.ensure_edge(
                plot.id,
                source_plot_id,
                link_type,
                weight=link_weight,
                confidence=plot.confidence,
                provenance=f"{source}_event",
            )
            self.graph.ensure_edge(
                source_plot_id,
                plot.id,
                reverse_type,
                weight=link_weight,
                confidence=self.plots[source_plot_id].confidence,
                provenance=f"{source}_event",
            )
        if source == "dream":
            self.identity.dream_count += 1
        else:
            self.identity.repair_count += 1
            self.identity.narrative_log.append(str(payload.get("text", "")))
        self._refresh_graph_metrics()
        return plot.id

    def _apply_compaction_event(self, *, event_id: str, payload: Dict[str, Any]) -> Summary:
        summary_id = det_id("summary", event_id)
        existing = self.summaries.get(summary_id)
        if existing is not None:
            return existing
        text = str(payload.get("text", ""))
        embedding = self.event_embedder.embed_content(
            (Message(role="system", parts=(TextPart(text=text),)),)
        )
        summary = Summary(
            id=summary_id,
            event_id=event_id,
            created_ts=float(payload.get("created_ts", now_ts())),
            updated_ts=float(payload.get("updated_ts", payload.get("created_ts", now_ts()))),
            text=text,
            embedding=embedding,
            source_plot_ids=[str(item) for item in payload.get("source_plot_ids", [])],
            source_event_ids=[str(item) for item in payload.get("source_event_ids", [])],
            tags=[str(item) for item in payload.get("tags", [])],
            fact_keys=[str(item) for item in payload.get("fact_keys", [])],
            start_ts=float(payload["start_ts"]) if payload.get("start_ts") is not None else None,
            end_ts=float(payload["end_ts"]) if payload.get("end_ts") is not None else None,
        )
        self._store_summary(summary)
        for story_id in [str(item) for item in payload.get("story_ids", [])]:
            if story_id in self.stories:
                self.graph.ensure_edge(
                    summary.id,
                    story_id,
                    "summarizes_story",
                    weight=0.5,
                    confidence=0.9,
                    provenance="compaction",
                )
        for theme_id in [str(item) for item in payload.get("theme_ids", [])]:
            if theme_id in self.themes:
                self.graph.ensure_edge(
                    summary.id,
                    theme_id,
                    "summarizes_theme",
                    weight=0.5,
                    confidence=0.9,
                    provenance="compaction",
                )
        for plot_id in summary.source_plot_ids:
            self.remove_plot_from_hot_state(plot_id)
        self.refresh_materialized_views()
        return summary

    def plan_repair_events(self, *, limit: int = 1) -> List[Dict[str, Any]]:
        targets = self.repair_operator.find_targets(
            graph=self.graph,
            plots=self.plots,
            anchor_ids=self.core_anchor_ids,
            limit=limit,
        )
        planned: List[Dict[str, Any]] = []
        for target in targets:
            target_plots = [self.plots[plot_id] for plot_id in target.plot_ids if plot_id in self.plots]
            if not target_plots:
                continue
            salient_axes = sorted(
                (
                    (
                        axis_name,
                        float(
                            np.mean(
                                [
                                    abs(plot.frame.axis_evidence.get(axis_name, 0.0))
                                    for plot in target_plots
                                ]
                            )
                        ),
                    )
                    for axis_name in self._axis_names()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            text = self.narrator.compose_repair(
                "integrate",
                self.identity,
                self.identity,
                target.score,
                [axis_name for axis_name, score in salient_axes[:3] if score > 0.0],
                " / ".join(plot.semantic_text[:96] for plot in target_plots[:2]),
                self.schema,
            )
            planned.append(
                {
                    "text": text,
                    "score": float(target.score),
                    "source_plot_ids": [plot.id for plot in target_plots],
                    "source_event_ids": [plot.event_id for plot in target_plots if plot.event_id],
                    "ts": now_ts(),
                }
            )
        return planned

    def plan_dream_events(self, *, n: int = 2) -> List[Dict[str, Any]]:
        candidates = self.dream_operator.propose(
            graph=self.graph,
            plots=self.plots,
            samples=self.cfg.dream_walk_samples,
            steps=self.cfg.dream_walk_steps,
        )
        fallback = candidates[: max(1, n)] if candidates else []
        chosen = [
            candidate for candidate in candidates if candidate.score >= self.cfg.dream_persist_threshold
        ]
        if not chosen:
            chosen = fallback
        planned: List[Dict[str, Any]] = []
        for candidate in chosen[:n]:
            operator = "integration" if candidate.resonance >= 0.35 else "counterfactual"
            text = self.narrator.compose_dream(
                operator,
                candidate.tags,
                self.identity,
                self.schema,
            )
            member_plots = [self.plots[plot_id] for plot_id in candidate.plot_ids if plot_id in self.plots]
            planned.append(
                {
                    "text": text,
                    "score": float(candidate.score),
                    "source_plot_ids": [plot.id for plot in member_plots],
                    "source_event_ids": [plot.event_id for plot in member_plots if plot.event_id],
                    "tags": list(candidate.tags),
                    "ts": now_ts(),
                }
            )
        return planned

    def plan_compaction_events(
        self,
        *,
        cold_age_s: float,
        mass_threshold: float,
        group_min_size: int,
    ) -> List[Dict[str, Any]]:
        self.refresh_materialized_views()
        now = now_ts()
        candidates = [
            plot
            for plot in self.plots.values()
            if plot.event_id
            and now - plot.ts >= cold_age_s
            and plot.mass() < mass_threshold
            and plot.tension < 0.40
            and plot.contradiction < 0.35
            and plot.id not in self.core_anchor_ids
            and all(
                self.graph.kind(neighbor_id) != "anchor"
                for neighbor_id in self.graph.g.predecessors(plot.id)
                if neighbor_id in self.graph.g
            )
        ]
        if len(candidates) < group_min_size:
            return []
        grouped: Dict[str, List[Plot]] = {}
        for plot in candidates:
            group_key = plot.story_id or plot.theme_id or f"time:{int(plot.ts // max(cold_age_s, 1.0))}"
            grouped.setdefault(group_key, []).append(plot)
        planned: List[Dict[str, Any]] = []
        for group in grouped.values():
            if len(group) < group_min_size:
                continue
            group.sort(key=lambda item: item.ts)
            tags: List[str] = []
            fact_keys: List[str] = []
            for plot in group:
                for tag in plot.frame.tags:
                    if tag not in tags:
                        tags.append(tag)
                for fact_key in plot.fact_keys:
                    if fact_key not in fact_keys:
                        fact_keys.append(fact_key)
            story_ids = sorted({plot.story_id for plot in group if plot.story_id})
            theme_ids = sorted({plot.theme_id for plot in group if plot.theme_id})
            snippets = " / ".join(plot.semantic_text[:64] for plot in group[:3])
            label = "、".join(tags[:4]) if tags else "旧记忆片段"
            text = f"[Summary] {label}。{snippets}"
            planned.append(
                {
                    "text": text,
                    "source_plot_ids": [plot.id for plot in group],
                    "source_event_ids": [plot.event_id for plot in group if plot.event_id],
                    "tags": tags[:12],
                    "fact_keys": fact_keys[:16],
                    "story_ids": story_ids,
                    "theme_ids": theme_ids,
                    "start_ts": float(group[0].ts),
                    "end_ts": float(group[-1].ts),
                    "created_ts": now,
                    "updated_ts": now,
                }
            )
        return planned

    def dream(self, n: int = 2) -> List[Plot]:
        """执行 graph-first 梦境演练循环。"""
        return self._graph_first_dreams(n=n)

    def evolve(self, dreams: int = 2) -> List[Plot]:
        """公开的演化接口"""
        evolved = self._graph_first_repairs(limit=1)
        evolved.extend(self._graph_first_dreams(n=dreams))
        self.graph_metrics["evolve"] = {
            "dreams": sum(1 for item in evolved if item.source == "dream"),
            "repairs": sum(1 for item in evolved if item.source == "repair"),
            "total": len(evolved),
        }
        self._refresh_graph_metrics()
        return evolved

    def query(self, messages: Sequence[Message], k: int = 8) -> RetrievalTrace:
        """记忆检索接口：场论驱动的复杂检索"""
        self._refresh_materialized_views(force=True)
        query_started = time.perf_counter()
        semantic_text = self.meaning_provider.project(messages)
        query_embedding = self.event_embedder.embed_content(messages)
        trace = self.retriever.retrieve(
            query_text=semantic_text,
            query_embedding=query_embedding,
            state=self.identity,
            kinds=self.cfg.retrieval_kinds,
            k=k,
        )
        query_duration_ms = max(0.0, (time.perf_counter() - query_started) * 1000.0)
        # 记录访问历史用于 Mass 计算
        for node_id, _, kind in trace.ranked:
            if kind == "plot" and node_id in self.plots:
                self.plots[node_id].access_count += 1
                self.plots[node_id].last_access_ts = now_ts()
            elif kind == "story" and node_id in self.stories:
                self.stories[node_id].reference_count += 1
        plot_hits = [node_id for node_id, _score, kind in trace.ranked if kind == "plot"]
        self.graph_metrics["query"] = {
            "latency_ms": round(float(query_duration_ms), 3),
            "plot_hits": plot_hits[:k],
            "hit_count": len(trace.ranked),
        }
        self._refresh_graph_metrics()
        return trace

    def feedback_retrieval(self, query_text: str, chosen_id: str, success: bool) -> None:
        """检索反馈学习：利用 Triplet Loss 在线微调度量矩阵 (Metric Matrix)"""
        if chosen_id not in self.graph.g:
            return
        query_vec = self.event_embedder.embed_content(
            (Message(role="user", parts=(TextPart(text=query_text),)),)
        )
        chosen_vec = self.retriever._payload_vec(self.graph.payload(chosen_id))
        if chosen_vec is None:
            return
        # 寻找负样本
        coarse = self.vindex.search(query_vec, k=6)
        negative_id = next((node_id for node_id, _ in coarse if node_id != chosen_id), None)
        if negative_id is not None:
            negative_vec = self.retriever._payload_vec(self.graph.payload(negative_id))
            if negative_vec is not None:
                # 训练：让正确项靠近 query，错误项远离
                if success:
                    self.metric.update_triplet(query_vec, chosen_vec, negative_vec)
                else:
                    self.metric.update_triplet(query_vec, negative_vec, chosen_vec)

        # 更新图和主题的证据置信度
        if self.graph.kind(chosen_id) == "theme" and chosen_id in self.themes:
            self.themes[chosen_id].update_evidence(success)
        for neighbor in list(self.graph.g.predecessors(chosen_id)) + list(
            self.graph.g.successors(chosen_id)
        ):
            if self.graph.g.has_edge(neighbor, chosen_id):
                self.graph.edge_belief(neighbor, chosen_id).update(success)
            if self.graph.g.has_edge(chosen_id, neighbor):
                self.graph.edge_belief(chosen_id, neighbor).update(success)
        self.graph._pagerank_cache.clear()

    def intuition_keywords(self, limit: int = 2) -> List[str]:
        """获取当前最显著的直觉（梦中倾向）"""
        items = sorted(
            self.identity.intuition_axes.items(), key=lambda item: abs(item[1]), reverse=True
        )
        phrases: List[str] = []
        for axis_name, value in items[:limit]:
            axis = self.schema.all_axes().get(axis_name)
            if axis is None or abs(value) < 0.05:
                continue
            pole = axis.positive_pole if value >= 0 else axis.negative_pole
            phrases.append(f"{pole}({value:+.2f})")
        return phrases

    def snapshot_identity(self) -> IdentitySnapshot:
        """生成身份快照用于 UI 展示"""
        self._refresh_materialized_views()
        ordered = self._axis_names()
        return IdentitySnapshot(
            current_mode=self.identity.current_mode_label,
            axis_state={
                name: round(float(self.identity.axis_state.get(name, 0.0)), 4) for name in ordered
            },
            intuition_axes={
                name: round(float(self.identity.intuition_axes.get(name, 0.0)), 4)
                for name in ordered
            },
            persona_axes={
                name: {
                    "positive_pole": axis.positive_pole,
                    "negative_pole": axis.negative_pole,
                    "description": axis.description,
                    "support_count": axis.support_count,
                    "aliases": list(axis.aliases),
                }
                for name, axis in self.schema.persona_axes.items()
            },
            axis_aliases=dict(self.schema.axis_aliases),
            modes={
                mode_id: {
                    "label": mode.label,
                    "support": mode.support,
                    "barrier": round(mode.barrier, 4),
                    "hysteresis": round(mode.hysteresis, 4),
                }
                for mode_id, mode in self.modes.items()
            },
            active_energy=round(self.identity.active_energy, 4),
            repressed_energy=round(self.identity.repressed_energy, 4),
            contradiction_ema=round(self.identity.contradiction_ema, 4),
            plasticity=round(self.identity.plasticity(), 4),
            rigidity=round(self.identity.rigidity(), 4),
            repair_count=self.identity.repair_count,
            dream_count=self.identity.dream_count,
            mode_change_count=self.identity.mode_change_count,
            narrative_tail=self.identity.narrative_log[-8:],
        )

    def narrative_summary(self) -> NarrativeSummary:
        """生成自我意识总结文本"""
        self._refresh_materialized_views()
        return self.narrator.compose_summary(self.identity, self.schema, self.recent_semantic_texts)

    def to_state_dict(self) -> Dict[str, Any]:
        """全量状态序列化，用于持久化存储"""
        return {
            "schema_version": SOUL_MEMORY_STATE_VERSION,
            "cfg": self.cfg.to_state_dict(),
            "kde": self.kde.to_state_dict(),
            "metric": self.metric.to_state_dict(),
            "graph": self.graph.to_state_dict(),
            "vindex": self.vindex.to_state_dict(),
            "plots": {plot_id: plot.to_state_dict() for plot_id, plot in self.plots.items()},
            "summaries": {
                summary_id: summary.to_state_dict()
                for summary_id, summary in self.summaries.items()
            },
            "stories": {
                story_id: story.to_state_dict() for story_id, story in self.stories.items()
            },
            "themes": {theme_id: theme.to_state_dict() for theme_id, theme in self.themes.items()},
            "schema": self.schema.to_state_dict(),
            "consolidator": self.consolidator.to_state_dict(),
            "identity": self.identity.to_state_dict(),
            "modes": {mode_id: mode.to_state_dict() for mode_id, mode in self.modes.items()},
            "recent_semantic_texts": list(self.recent_semantic_texts),
            "step": int(self.step),
            "anchors": {
                anchor_id: {
                    "id": payload.get("id", anchor_id),
                    "label": payload.get("label", ""),
                    "embedding": cast(np.ndarray, payload["embedding"]).astype(np.float32).tolist(),
                    "pinned": bool(payload.get("pinned", True)),
                }
                for anchor_id, payload in self.anchor_nodes.items()
            },
            "core_anchor_ids": list(self.core_anchor_ids),
            "view_stats": self.view_stats.to_state_dict(),
            "graph_metrics": copy.deepcopy(self.graph_metrics),
        }

    @classmethod
    def from_state_dict(
        cls,
        data: Dict[str, Any],
        *,
        event_embedder: Optional[ContentEmbeddingProvider] = None,
        axis_embedder: Optional[TextEmbeddingProvider] = None,
        meaning_provider: Optional[MeaningProvider] = None,
        narrator: Optional[NarrativeProvider] = None,
        query_analyzer: Optional[BaseQueryAnalyzer] = None,
    ) -> "AuroraSoul":
        """从状态数据恢复灵魂引擎"""
        if data.get("schema_version") != SOUL_MEMORY_STATE_VERSION:
            raise ValueError("Snapshot schema mismatch: expected Aurora Soul v6")
        cfg = SoulConfig.from_state_dict(data["cfg"])
        obj = cls(
            cfg=cfg,
            seed=0,
            event_embedder=event_embedder,
            axis_embedder=axis_embedder,
            meaning_provider=meaning_provider,
            narrator=narrator,
            query_analyzer=query_analyzer,
            bootstrap=False,
        )
        # 恢复各个组件状态
        obj.kde = OnlineKDE.from_state_dict(data["kde"])
        obj.metric = LowRankMetric.from_state_dict(data["metric"])
        obj.vindex = VectorIndex.from_state_dict(data["vindex"])
        obj.plots = {
            plot_id: Plot.from_state_dict(item) for plot_id, item in data.get("plots", {}).items()
        }
        obj.summaries = {
            summary_id: Summary.from_state_dict(item)
            for summary_id, item in data.get("summaries", {}).items()
        }
        obj.stories = {
            story_id: StoryArc.from_state_dict(item)
            for story_id, item in data.get("stories", {}).items()
        }
        obj.themes = {
            theme_id: Theme.from_state_dict(item)
            for theme_id, item in data.get("themes", {}).items()
        }
        obj.graph = MemoryGraph()
        # 重建图节点
        for plot_id, plot in obj.plots.items():
            obj.graph.add_node(plot_id, "plot", plot)
        for summary_id, summary in obj.summaries.items():
            obj.graph.add_node(summary_id, "summary", summary)
        for story_id, story in obj.stories.items():
            obj.graph.add_node(story_id, "story", story)
        for theme_id, theme in obj.themes.items():
            obj.graph.add_node(theme_id, "theme", theme)
        obj.anchor_nodes = {}
        obj.core_anchor_ids = [str(item) for item in data.get("core_anchor_ids", [])]
        for anchor_id, payload in data.get("anchors", {}).items():
            anchor_payload = {
                "id": str(payload.get("id", anchor_id)),
                "label": str(payload.get("label", "")),
                "embedding": np.asarray(payload.get("embedding", []), dtype=np.float32),
                "pinned": bool(payload.get("pinned", True)),
            }
            obj.anchor_nodes[str(anchor_id)] = anchor_payload
            obj.graph.add_node(str(anchor_id), "anchor", anchor_payload)
        if not obj.core_anchor_ids:
            obj._bootstrap_core_anchors()
        obj.graph.restore_edges(data["graph"])
        obj.retriever = FieldRetriever(
            metric=obj.metric,
            vindex=obj.vindex,
            graph=obj.graph,
            query_analyzer=query_analyzer or MissingQueryAnalyzer(),
        )
        obj.schema = PsychologicalSchema.from_state_dict(data["schema"])
        for axis in obj.schema.all_axes().values():
            if (
                axis.direction is None
                or axis.positive_anchor is None
                or axis.negative_anchor is None
            ):
                axis.compile(obj.axis_embedder)
        obj.consolidator = SchemaConsolidator.from_state_dict(data.get("consolidator", {}))
        obj.identity = IdentityState.from_state_dict(data["identity"])
        obj.modes = {
            mode_id: IdentityMode.from_state_dict(item)
            for mode_id, item in data.get("modes", {}).items()
        }
        obj.recent_semantic_texts = [
            str(item) for item in data.get("recent_semantic_texts", [])
        ]
        obj.step = int(data.get("step", 0))
        obj.view_stats = GraphViewStats.from_state_dict(data.get("view_stats", {}))
        obj.graph_metrics = copy.deepcopy(data.get("graph_metrics", {}))
        obj._refresh_materialized_views(force=True)
        obj._refresh_graph_metrics()
        return obj
