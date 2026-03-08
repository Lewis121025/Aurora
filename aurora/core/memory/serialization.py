"""
AURORA 序列化模块 (Serialization Module)
===========================

状态序列化与反序列化功能。

核心职责：
- 将 AuroraMemory 的状态序列化为兼容 JSON 的字典
- 将状态字典反序列化回 AuroraMemory 实例
- 处理版本迁移以实现向后兼容性

设计原则：
- 人类可读的状态审查
- 跨版本兼容性
- 状态的部分恢复
- 状态差异对比和调试
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

from aurora.core.components.assignment import CRPAssigner, StoryModel, ThemeModel
from aurora.core.components.bandit import ThompsonBernoulliGate
from aurora.core.components.density import OnlineKDE
from aurora.core.components.metric import LowRankMetric
from aurora.core.graph.memory_graph import MemoryGraph
from aurora.core.graph.vector_index import VectorIndex
from aurora.core.models.config import MemoryConfig
from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.core.retrieval.field_retriever import FieldRetriever
from aurora.core.entity_tracker import EntityTracker
from aurora.core.config.retrieval import RECENT_ENCODED_PLOTS_WINDOW

class SerializationMixin:
    """提供状态序列化和反序列化功能的 Mixin 类。"""

    # -------------------------------------------------------------------------
    # 状态序列化 (State Serialization)
    # -------------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """
        将整个 AuroraMemory 状态序列化为兼容 JSON 的字典。
        
        这使用结构化的 JSON 替代了基于 pickle 的序列化机制，
        实现了：
        - 人类可读的状态审查
        - 跨版本兼容性
        - 状态的部分恢复
        - 状态差异对比和调试
        
        返回:
            代表完整状态的兼容 JSON 的字典
        """
        return {
            "version": 2,  # 状态格式版本号，用于向后兼容
            "cfg": self.cfg.to_state_dict(),
            "seed": self._seed,
            "benchmark_mode": getattr(self, 'benchmark_mode', False),
            
            # 可学习组件 (Learnable components)
            "kde": self.kde.to_state_dict(),
            "metric": self.metric.to_state_dict(),
            "gate": self.gate.to_state_dict(),
            
            # 非参数化分配 (Nonparametric assignment)
            "crp_story": self.crp_story.to_state_dict(),
            "crp_theme": self.crp_theme.to_state_dict(),
            
            # 记忆数据 (Memory data)
            "plots": {pid: p.to_state_dict() for pid, p in self.plots.items()},
            "stories": {sid: s.to_state_dict() for sid, s in self.stories.items()},
            "themes": {tid: t.to_state_dict() for tid, t in self.themes.items()},
            
            # 图结构 (内部 payload 会引用 plots/stories/themes)
            "graph": self.graph.to_state_dict(),
            
            # 向量索引 (Vector index) (生产环境中已弃用，请使用 VectorStore)
            "vindex": self.vindex.to_state_dict(),
            
            # 簿记信息 (将 deque 转换为 list 以兼容 JSON)
            "recent_encoded_plot_ids": list(self._recent_encoded_plot_ids),
            
            # 以人际关系为中心的附加信息 (Relationship-centric additions)
            "relationship_story_index": self._relationship_story_index,
            "identity_dimensions": self._identity_dimensions,
            
            # 时间索引 (时间作为一等公民)
            # 将整型的主键转换为字符串以兼容 JSON
            "temporal_index": {str(k): v for k, v in self._temporal_index.items()},
            "temporal_index_min_bucket": self._temporal_index_min_bucket,
            "temporal_index_max_bucket": self._temporal_index_max_bucket,
            
            # 实体属性追踪器 (第3阶段)
            "entity_tracker": getattr(self, 'entity_tracker', None).to_state_dict() if hasattr(self, 'entity_tracker') and self.entity_tracker else None,
        }

    # -------------------------------------------------------------------------
    # 状态反序列化 (State Deserialization)
    # -------------------------------------------------------------------------

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "SerializationMixin":
        """从状态字典中重建 AuroraMemory 实例。"""
        cfg = MemoryConfig.from_state_dict(d["cfg"])
        benchmark_mode = d.get("benchmark_mode", False)
        obj = cls(cfg=cfg, seed=d.get("seed", 0), benchmark_mode=benchmark_mode)
        
        obj._restore_learnable_components(d)
        obj._restore_memory_data(d)
        obj._rebuild_indices_and_models(d)
        obj._restore_bookkeeping(d)
        
        return obj

    def _restore_learnable_components(self, d: Dict[str, Any]) -> None:
        """从状态字典中恢复可学习组件。"""
        self.kde = OnlineKDE.from_state_dict(d["kde"])
        self.metric = LowRankMetric.from_state_dict(d["metric"])
        self.gate = ThompsonBernoulliGate.from_state_dict(d["gate"])
        self.crp_story = CRPAssigner.from_state_dict(d["crp_story"])
        self.crp_theme = CRPAssigner.from_state_dict(d["crp_theme"])

    def _restore_memory_data(self, d: Dict[str, Any]) -> None:
        """从状态字典中恢复情节、故事和主题。"""
        self.plots = {pid: Plot.from_state_dict(pd) for pid, pd in d.get("plots", {}).items()}
        self.stories = {sid: StoryArc.from_state_dict(sd) for sid, sd in d.get("stories", {}).items()}
        self.themes = {tid: Theme.from_state_dict(td) for tid, td in d.get("themes", {}).items()}

    def _rebuild_indices_and_models(self, d: Dict[str, Any]) -> None:
        """重建图结构、向量索引和模型。"""
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
        """恢复由于簿记产生的数据结构。"""
        self._recent_encoded_plot_ids = deque(
            d.get("recent_encoded_plot_ids", []),
            maxlen=RECENT_ENCODED_PLOTS_WINDOW
        )
        
        self._relationship_story_index = d.get("relationship_story_index", {})
        self._identity_dimensions = d.get("identity_dimensions", {})
        
        # 如果关系索引不存在，则从故事中重新构建
        if not self._relationship_story_index:
            for sid, story in self.stories.items():
                if story.relationship_with:
                    self._relationship_story_index[story.relationship_with] = sid
        
        # 恢复时间索引 (时间作为一等公民)
        # 将字符串的键重新转换为整型
        temporal_index_raw = d.get("temporal_index", {})
        self._temporal_index: Dict[int, List[str]] = {int(k): v for k, v in temporal_index_raw.items()}
        self._temporal_index_min_bucket = d.get("temporal_index_min_bucket", 0)
        self._temporal_index_max_bucket = d.get("temporal_index_max_bucket", 0)
        
        # 如果时间索引不存在或为空，从情节记录里重新构建时间索引
        if not self._temporal_index and self.plots:
            for pid, plot in self.plots.items():
                day_bucket = int(plot.ts // 86400)  # 每天 86400 秒
                if day_bucket not in self._temporal_index:
                    self._temporal_index[day_bucket] = []
                self._temporal_index[day_bucket].append(pid)
                # 更新最大/最小的存储桶(buckets)
                if not self._temporal_index_min_bucket or day_bucket < self._temporal_index_min_bucket:
                    self._temporal_index_min_bucket = day_bucket
                if not self._temporal_index_max_bucket or day_bucket > self._temporal_index_max_bucket:
                    self._temporal_index_max_bucket = day_bucket
        
        # 恢复实体跟踪器 (第3阶段)
        entity_tracker_dict = d.get("entity_tracker")
        if entity_tracker_dict:
            self.entity_tracker = EntityTracker.from_state_dict(entity_tracker_dict)
        elif hasattr(self, 'entity_tracker'):
            # 如果状态字典里不存在，但是 entity_tracker 属性存在，则保留（向后兼容性）
            pass
        else:
            # 如果不存在，创建新的实体跟踪器
            seed = d.get("seed", 0)
            self.entity_tracker = EntityTracker(seed=seed)
            # 尽可能从情节里重构
            if self.plots:
                for pid, plot in self.plots.items():
                    self.entity_tracker.update(plot.text, pid, plot.ts)
