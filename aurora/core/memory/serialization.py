"""
AURORA V2 序列化模块
===========================

状态序列化与反序列化功能。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

from aurora.core.components.assignment import CRPAssigner, StoryModel, ThemeModel
from aurora.core.components.bandit import ThompsonBernoulliGate
from aurora.core.components.density import OnlineKDE
from aurora.core.components.metric import LowRankMetric
from aurora.core.entity_tracker import EntityTracker
from aurora.core.graph.memory_graph import MemoryGraph
from aurora.core.graph.vector_index import VectorIndex
from aurora.core.models.config import MemoryConfig
from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.core.retrieval.field_retriever import FieldRetriever
from aurora.core.self_narrative import SelfNarrative, SubconsciousState
from aurora.core.config.retrieval import RECENT_ENCODED_PLOTS_WINDOW


STATE_SCHEMA_VERSION = "aurora-memory-v2"


class SerializationMixin:
    """提供状态序列化和反序列化功能的 Mixin 类。"""

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "version": STATE_SCHEMA_VERSION,
            "cfg": self.cfg.to_state_dict(),
            "seed": self._seed,
            "benchmark_mode": getattr(self, "benchmark_mode", False),
            "kde": self.kde.to_state_dict(),
            "subconscious_kde": self.subconscious_kde.to_state_dict(),
            "metric": self.metric.to_state_dict(),
            "gate": self.gate.to_state_dict(),
            "crp_story": self.crp_story.to_state_dict(),
            "crp_theme": self.crp_theme.to_state_dict(),
            "plots": {pid: p.to_state_dict() for pid, p in self.plots.items()},
            "stories": {sid: s.to_state_dict() for sid, s in self.stories.items()},
            "themes": {tid: t.to_state_dict() for tid, t in self.themes.items()},
            "graph": self.graph.to_state_dict(),
            "vindex": self.vindex.to_state_dict(),
            "recent_encoded_plot_ids": list(self._recent_encoded_plot_ids),
            "relationship_story_index": self._relationship_story_index,
            "identity_dimensions": self._identity_dimensions,
            "temporal_index": {str(k): v for k, v in self._temporal_index.items()},
            "temporal_index_min_bucket": self._temporal_index_min_bucket,
            "temporal_index_max_bucket": self._temporal_index_max_bucket,
            "entity_tracker": self.entity_tracker.to_state_dict(),
            "personality_profile_id": self.personality_profile.profile_id,
            "self_narrative": self.self_narrative_engine.narrative.to_state_dict(),
            "subconscious_state": self.subconscious_state.to_state_dict(),
            "personality_bootstrapped": self._personality_bootstrapped,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any], *, embedder: Any = None) -> "SerializationMixin":
        version = d.get("version")
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Aurora memory state version: {version!r}. "
                "Aurora V2 requires a fresh data directory."
            )

        cfg = MemoryConfig.from_state_dict(d["cfg"])
        benchmark_mode = d.get("benchmark_mode", False)
        obj = cls(
            cfg=cfg,
            seed=d.get("seed", 0),
            embedder=embedder,
            benchmark_mode=benchmark_mode,
            bootstrap_profile=False,
        )

        obj._restore_learnable_components(d)
        obj._restore_memory_data(d)
        obj._rebuild_indices_and_models(d)
        obj._restore_bookkeeping(d)
        obj.self_narrative_engine.narrative = SelfNarrative.from_state_dict(d.get("self_narrative", {}))
        obj.subconscious_state = SubconsciousState.from_state_dict(d.get("subconscious_state", {}))
        obj.self_narrative_engine.refresh_subconscious_summary(obj.subconscious_state)
        obj._personality_bootstrapped = bool(d.get("personality_bootstrapped", True))

        return obj

    def _restore_learnable_components(self, d: Dict[str, Any]) -> None:
        self.kde = OnlineKDE.from_state_dict(d["kde"])
        self.subconscious_kde = OnlineKDE.from_state_dict(d["subconscious_kde"])
        self.metric = LowRankMetric.from_state_dict(d["metric"])
        self.gate = ThompsonBernoulliGate.from_state_dict(d["gate"])
        self.crp_story = CRPAssigner.from_state_dict(d["crp_story"])
        self.crp_theme = CRPAssigner.from_state_dict(d["crp_theme"])

    def _restore_memory_data(self, d: Dict[str, Any]) -> None:
        self.plots = {pid: Plot.from_state_dict(pd) for pid, pd in d.get("plots", {}).items()}
        self.stories = {sid: StoryArc.from_state_dict(sd) for sid, sd in d.get("stories", {}).items()}
        self.themes = {tid: Theme.from_state_dict(td) for tid, td in d.get("themes", {}).items()}

    def _rebuild_indices_and_models(self, d: Dict[str, Any]) -> None:
        payloads: Dict[str, Any] = {}
        payloads.update(self.plots)
        payloads.update(self.stories)
        payloads.update(self.themes)

        self.graph = MemoryGraph.from_state_dict(d["graph"], payloads=payloads)
        self.vindex = VectorIndex.from_state_dict(d["vindex"])
        self.story_model = StoryModel(metric=self.metric)
        self.theme_model = ThemeModel(metric=self.metric)
        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)

    def _restore_bookkeeping(self, d: Dict[str, Any]) -> None:
        self._recent_encoded_plot_ids = deque(
            d.get("recent_encoded_plot_ids", []),
            maxlen=RECENT_ENCODED_PLOTS_WINDOW,
        )
        self._relationship_story_index = d.get("relationship_story_index", {})
        self._identity_dimensions = d.get("identity_dimensions", {})

        temporal_index_raw = d.get("temporal_index", {})
        self._temporal_index = {int(k): v for k, v in temporal_index_raw.items()}
        self._temporal_index_min_bucket = d.get("temporal_index_min_bucket", 0)
        self._temporal_index_max_bucket = d.get("temporal_index_max_bucket", 0)

        entity_tracker_dict = d.get("entity_tracker")
        self.entity_tracker = EntityTracker.from_state_dict(entity_tracker_dict) if entity_tracker_dict else EntityTracker(seed=d.get("seed", 0))
