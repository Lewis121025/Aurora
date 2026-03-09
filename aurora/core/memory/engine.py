"""
AURORA 内存核心
==================

主入口点：AuroraMemory 类。

设计：零硬编码阈值。所有决策通过贝叶斯/随机策略进行。

架构：
- 核心类从不同关注点的专用 mixin 继承
- relationship.py：关系识别和身份评估
- temporal.py：时间索引和基于时间的访问
- ingestion.py：摄入、批量导入、冲突处理与图编织
- retrieval.py：查询、时间线组织与反馈学习
- pressure.py：面向增长的压力管理
- evolution.py：演化、反思和意义重构
- serialization.py：状态序列化/反序列化
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from aurora.core.abstention import AbstentionDetector
from aurora.core.coherence import CoherenceGuardian
from aurora.core.components.assignment import CRPAssigner, StoryModel, ThemeModel
from aurora.core.components.bandit import ThompsonBernoulliGate
from aurora.core.components.density import OnlineKDE
from aurora.core.components.metric import LowRankMetric
from aurora.core.config.numeric import EPSILON_PRIOR
from aurora.core.config.retrieval import RECENT_ENCODED_PLOTS_WINDOW
from aurora.core.entity_tracker import EntityTracker
from aurora.core.fact_extractor import FactExtractor
from aurora.core.graph.memory_graph import MemoryGraph
from aurora.core.graph.vector_index import VectorIndex
from aurora.core.knowledge import KnowledgeClassifier
from aurora.core.memory.evolution import EvolutionMixin
from aurora.core.memory.ingestion import IngestionMixin
from aurora.core.memory.pressure import PressureMixin
from aurora.core.memory.relationship import RelationshipMixin
from aurora.core.memory.retrieval import RetrievalMixin
from aurora.core.memory.serialization import SerializationMixin
from aurora.core.memory.temporal import TemporalMemoryMixin
from aurora.core.models.config import MemoryConfig
from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.core.personality import PersonalityProfile, load_personality_profile
from aurora.core.retrieval.field_retriever import FieldRetriever
from aurora.core.self_narrative import SelfNarrativeEngine, SubconsciousState
from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding
from aurora.exceptions import MemoryNotFoundError
from aurora.utils.id_utils import det_id
from aurora.utils.math_utils import cosine_sim, sigmoid
from aurora.utils.time_utils import now_ts

logger = logging.getLogger(__name__)


class AuroraMemory(
    RelationshipMixin,
    TemporalMemoryMixin,
    IngestionMixin,
    RetrievalMixin,
    PressureMixin,
    EvolutionMixin,
    SerializationMixin,
):
    """AURORA 内存：从第一原理产生的叙事性内存。"""

    def __init__(
        self,
        cfg: Optional[MemoryConfig] = None,
        seed: int = 0,
        embedder=None,
        benchmark_mode: bool = False,
        bootstrap_profile: bool = False,
    ):
        """初始化 AURORA 内存系统。"""
        cfg = cfg or MemoryConfig()
        self.cfg = cfg
        self._seed = seed
        self.benchmark_mode = benchmark_mode or cfg.benchmark_mode
        self.rng = np.random.default_rng(seed)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = LocalSemanticEmbedding(dim=cfg.dim, seed=seed)

        self._warn_if_hash_embedding()
        self.kde = OnlineKDE(dim=cfg.dim, reservoir=cfg.kde_reservoir, seed=seed)
        self.subconscious_kde = OnlineKDE(
            dim=cfg.dim,
            reservoir=cfg.subconscious_reservoir,
            seed=seed + 7,
        )
        self.metric = LowRankMetric(dim=cfg.dim, rank=cfg.metric_rank, seed=seed)
        self.gate = ThompsonBernoulliGate(feature_dim=cfg.gate_feature_dim, seed=seed)

        self.graph = MemoryGraph()
        self.vindex = self._create_vector_index(cfg)

        self.plots: Dict[str, Plot] = {}
        self.stories: Dict[str, StoryArc] = {}
        self.themes: Dict[str, Theme] = {}

        self.crp_story = CRPAssigner(alpha=cfg.story_alpha, seed=seed)
        self.story_model = StoryModel(metric=self.metric)
        self.crp_theme = CRPAssigner(alpha=cfg.theme_alpha, seed=seed + 1)
        self.theme_model = ThemeModel(metric=self.metric)

        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)
        self._recent_encoded_plot_ids: Deque[str] = deque(maxlen=RECENT_ENCODED_PLOTS_WINDOW)

        self._relationship_story_index: Dict[str, str] = {}
        self._identity_dimensions: Dict[str, float] = {}

        self._temporal_index: Dict[int, list[str]] = {}
        self._temporal_index_min_bucket: int = 0
        self._temporal_index_max_bucket: int = 0

        self.knowledge_classifier = KnowledgeClassifier(seed=seed)
        self.coherence_guardian = CoherenceGuardian(metric=self.metric, seed=seed)
        self.abstention_detector = AbstentionDetector()
        self.entity_tracker = EntityTracker(seed=seed)
        self.fact_extractor = FactExtractor()
        self.personality_profile: PersonalityProfile = load_personality_profile(cfg.personality_profile_id)
        self.subconscious_state = SubconsciousState()
        self.self_narrative_engine = SelfNarrativeEngine(
            self.metric,
            profile=self.personality_profile,
            embedder=self.embedder,
            seed=seed,
        )
        self._personality_bootstrapped = False

        if bootstrap_profile:
            self.bootstrap_personality()

    def _warn_if_hash_embedding(self) -> None:
        """如果使用 HashEmbedding，则发出警告。"""
        if isinstance(self.embedder, HashEmbedding):
            warning_msg = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠️  CRITICAL WARNING ⚠️                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  You are using HashEmbedding, which produces RANDOM vectors!                 ║
║  Memory retrieval will be essentially RANDOM and INEFFECTIVE.                ║
║                                                                              ║
║  HashEmbedding is for TESTING ONLY. In production, configure a real         ║
║  embedding provider:                                                         ║
║                                                                              ║
║  Option 1: 阿里云百炼 (Bailian)                                              ║
║    export AURORA_BAILIAN_EMBEDDING_API_KEY="your-api-key"                    ║
║    export AURORA_EMBEDDING_PROVIDER="bailian"                                ║
║                                                                              ║
║  Option 2: 火山方舟 (Volcengine Ark)                                         ║
║    export AURORA_ARK_API_KEY="your-api-key"                                  ║
║    export AURORA_EMBEDDING_PROVIDER="ark"                                    ║
║                                                                              ║
║  For benchmarks, this will result in near-random accuracy scores.            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            logger.warning(warning_msg)

    def is_using_hash_embedding(self) -> bool:
        """检查内存系统是否使用 HashEmbedding。"""
        return isinstance(self.embedder, HashEmbedding)

    def _create_vector_index(self, cfg: MemoryConfig) -> VectorIndex:
        """创建本地精确向量索引。"""
        return VectorIndex(dim=cfg.dim)

    def evolve(self) -> None:
        """执行离线演化步骤以进行内存整合。"""
        logger.info(
            f"Starting evolution: plots={len(self.plots)}, "
            f"stories={len(self.stories)}, themes={len(self.themes)}"
        )

        self._reflect_on_relationships()
        self._check_reframe_opportunities()
        self._detect_story_boundaries()
        self._update_story_statuses()
        self._process_theme_emergence()
        self._analyze_identity_tensions()
        self._cleanup_graph_structure()
        self._pressure_manage()

        logger.info(
            f"Evolution complete: plots={len(self.plots)}, "
            f"stories={len(self.stories)}, themes={len(self.themes)}"
        )

    def _update_story_statuses(self) -> None:
        """根据活动概率更新 story 状态。"""
        for story in self.stories.values():
            if story.status != "developing":
                continue

            p_active = story.activity_probability()
            if self.rng.random() < p_active:
                continue

            if len(story.tension_curve) >= 3:
                slope = story.tension_curve[-1] - story.tension_curve[0]
                p_resolve = sigmoid(-slope)
            else:
                p_resolve = 0.5

            story.status = "resolved" if self.rng.random() < p_resolve else "abandoned"

    def _process_theme_emergence(self) -> None:
        """从 resolved stories 中处理主题涌现。"""
        for sid, story in list(self.stories.items()):
            if story.status != "resolved" or story.centroid is None:
                continue

            logps: Dict[str, float] = {}
            for tid, theme in self.themes.items():
                prior = math.log(len(theme.story_ids) + EPSILON_PRIOR)
                logps[tid] = prior + self.theme_model.loglik(story, theme)

            chosen_id, _ = self.crp_theme.sample(logps)

            if chosen_id is None:
                theme = Theme(id=det_id("theme", sid), created_ts=now_ts(), updated_ts=now_ts())
                theme.prototype = story.centroid.copy()

                if story.is_relationship_story() and story.my_identity_in_this_relationship:
                    theme.identity_dimension = f"作为{story.my_identity_in_this_relationship}的我"
                    theme.theme_type = "identity"
                    if story.relationship_with:
                        theme.supporting_relationships.append(story.relationship_with)

                self.themes[theme.id] = theme
                self.graph.add_node(theme.id, "theme", theme)
                self.vindex.add(theme.id, theme.prototype, kind="theme")
                chosen_id = theme.id

            theme = self.themes[chosen_id]
            theme.story_ids.append(sid)
            theme.updated_ts = now_ts()

            if story.is_relationship_story() and story.relationship_with:
                theme.add_supporting_relationship(story.relationship_with)

            theme.prototype = self._update_centroid_online(
                theme.prototype, story.centroid, len(theme.story_ids)
            )

            self._create_bidirectional_edge(sid, theme.id, "thematizes", "exemplified_by")

    def get_story(self, story_id: str) -> StoryArc:
        """按 ID 获取 story。"""
        if story_id not in self.stories:
            raise MemoryNotFoundError("story", story_id)
        return self.stories[story_id]

    def get_plot(self, plot_id: str) -> Plot:
        """按 ID 获取 plot。"""
        if plot_id not in self.plots:
            raise MemoryNotFoundError("plot", plot_id)
        return self.plots[plot_id]

    def get_theme(self, theme_id: str) -> Theme:
        """按 ID 获取 theme。"""
        if theme_id not in self.themes:
            raise MemoryNotFoundError("theme", theme_id)
        return self.themes[theme_id]

    def get_relationship_story(self, entity_id: str) -> Optional[StoryArc]:
        """获取特定关系实体的 story。"""
        story_id = self._relationship_story_index.get(entity_id)
        return self.stories.get(story_id) if story_id else None

    def get_my_identity_with(self, entity_id: str) -> Optional[str]:
        """获取我在特定关系中的身份。"""
        story = self.get_relationship_story(entity_id)
        return story.my_identity_in_this_relationship if story else None

    def get_all_relationships(self) -> Dict[str, StoryArc]:
        """获取所有关系 story。"""
        return {
            entity: self.stories[story_id]
            for entity, story_id in self._relationship_story_index.items()
            if story_id in self.stories
        }

    def get_identity_summary(self) -> Dict[str, Any]:
        """获取代理在所有关系中的身份摘要。"""
        relationships = self.get_all_relationships()

        summary = {
            "identity_dimensions": dict(self._identity_dimensions),
            "relationship_identities": {
                entity: story.my_identity_in_this_relationship
                for entity, story in relationships.items()
            },
            "relationship_count": len(relationships),
            "total_interactions": sum(len(story.plot_ids) for story in relationships.values()),
        }

        if self._identity_dimensions:
            dominant = max(self._identity_dimensions, key=self._identity_dimensions.get)
            summary["dominant_dimension"] = dominant

        return summary

    def bootstrap_personality(self) -> None:
        """在空内存上注入原生人格种子。"""
        if self._personality_bootstrapped or self.plots:
            return

        seed_plot_ids: List[str] = []
        for seed_spec in self.personality_profile.seed_plots:
            emb = self.embedder.embed(seed_spec.text)
            plot = Plot(
                id=det_id("plot", f"seed:{seed_spec.seed_id}"),
                ts=now_ts(),
                text=seed_spec.text,
                actors=("self",),
                embedding=emb,
                knowledge_type="identity_value",
                knowledge_confidence=1.0,
                source="seed",
                exposure="explicit",
                fact_keys=[seed_spec.text],
            )
            self.plots[plot.id] = plot
            self.graph.add_node(plot.id, "plot", plot)
            self.vindex.add(plot.id, plot.embedding, kind="plot")
            seed_plot_ids.append(plot.id)

        self.self_narrative_engine.bootstrap_seed_plot_ids(seed_plot_ids)
        self.self_narrative_engine.refresh_subconscious_summary(self.subconscious_state)
        self._personality_bootstrapped = True

    def is_identity_query(self, query_type: Any) -> bool:
        return getattr(query_type, "name", "") == "IDENTITY"

    def allow_plot_for_query(self, plot: Plot, query_type: Any) -> bool:
        if plot.exposure != "explicit":
            return False
        if plot.source == "seed" and not self.is_identity_query(query_type):
            return False
        if plot.status != "active" and not self.is_identity_query(query_type):
            return False
        return True

    def _dark_matter_entry_from_plot(self, *, plot: Plot, entry_id: str, affect_hint: str = ""):
        from aurora.core.self_narrative.models import DarkMatterEntry

        relational_hint = ""
        if plot.relational is not None:
            relational_hint = plot.relational.what_this_says_about_us
        return DarkMatterEntry(
            id=entry_id,
            ts=plot.ts,
            embedding=plot.embedding.copy(),
            knowledge_type=plot.knowledge_type or "unknown",
            affect_hint=affect_hint,
            relational_hint=relational_hint,
            source_plot_id=plot.id,
        )

    def note_shadow_plot(self, plot: Plot) -> None:
        affect_hint = ""
        if plot.relational is not None and plot.relational.relationship_quality_delta < 0:
            affect_hint = "guarded"
        elif plot.relational is not None and plot.relational.relationship_quality_delta > 0:
            affect_hint = "warm"
        elif plot.knowledge_type in {"identity_trait", "identity_value"}:
            affect_hint = "identity"

        entry_id = det_id("shadow", plot.id)
        self.subconscious_state.add_dark_matter(
            entry=self._dark_matter_entry_from_plot(plot=plot, entry_id=entry_id, affect_hint=affect_hint),
            max_entries=self.cfg.subconscious_reservoir,
        )
        self.subconscious_kde.add(plot.embedding)
        self.self_narrative_engine.refresh_subconscious_summary(self.subconscious_state)

    def mark_plot_repressed(self, plot_id: str) -> None:
        plot = self.plots.get(plot_id)
        if plot is None:
            return
        plot.exposure = "repressed"
        self.subconscious_state.mark_repressed(plot_id)
        self.self_narrative_engine.refresh_subconscious_summary(self.subconscious_state)

    def generate_system_intuition(self, trace: Any, max_items: int = 2) -> List[str]:
        candidate_vectors: List[np.ndarray] = []
        for plot_id in getattr(trace, "intuition_source_ids", []) or []:
            plot = self.plots.get(plot_id)
            if plot is not None:
                candidate_vectors.append(plot.embedding)

        if not candidate_vectors and self.subconscious_state.dark_matter_pool:
            scored_entries = sorted(
                self.subconscious_state.dark_matter_pool,
                key=lambda entry: cosine_sim(trace.query_emb, entry.embedding),
                reverse=True,
            )
            candidate_vectors.extend(entry.embedding for entry in scored_entries[:max_items])

        if not candidate_vectors and self.personality_profile.intuition_anchors:
            self.subconscious_state.last_intuition = [
                self.personality_profile.intuition_anchors[0].keywords[0]
            ]
            self.self_narrative_engine.refresh_subconscious_summary(self.subconscious_state)
            return list(self.subconscious_state.last_intuition)

        keywords = self.self_narrative_engine.intuition_keywords_for_vectors(
            candidate_vectors,
            max_items=max_items,
        )
        self.subconscious_state.last_intuition = keywords
        self.self_narrative_engine.refresh_subconscious_summary(self.subconscious_state)
        return keywords


if __name__ == "__main__":
    mem = AuroraMemory(cfg=MemoryConfig(dim=96, metric_rank=32, max_plots=2000), seed=42)

    mem.ingest("用户：我想做一个记忆系统。助理：好的，我们从第一性原理开始。", context_text="memory algorithm")
    mem.ingest("用户：不要硬编码阈值。助理：可以用贝叶斯决策和随机策略。", context_text="memory algorithm")
    mem.ingest("用户：检索要能讲故事。助理：可以用故事弧 + 主题涌现。", context_text="narrative memory")
    mem.ingest("用户：给我一个可运行的实现。助理：我会给你一份python参考实现。", context_text="implementation")

    trace = mem.query("如何避免硬编码阈值并实现叙事检索？", k=5)
    print("Top results:", trace.ranked)

    if trace.ranked:
        chosen_id = trace.ranked[0][0]
        mem.feedback_retrieval("如何避免硬编码阈值并实现叙事检索？", chosen_id=chosen_id, success=True)

    mem.evolve()
    print("stories:", len(mem.stories), "themes:", len(mem.themes), "plots:", len(mem.plots))
