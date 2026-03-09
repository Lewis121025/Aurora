"""
叙事重构模块
===============================

核心叙事重构引擎和相关数据类。

职责：
1. 定义NarrativeElement和NarrativeTrace数据类
2. 实现用于故事重构的NarratorEngine
3. 协调视角选择、上下文恢复和叙事生成

设计原则：
- 零硬编码阈值：所有决策使用贝叶斯/随机策略
- 确定性可重现性：所有随机操作支持种子
- 完整的类型注解
- 可序列化状态
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.lab.primitives.metric import LowRankMetric
from aurora.lab.graph.memory_graph import MemoryGraph
from aurora.lab.graph.vector_index import VectorIndex
from aurora.lab.models.plot import Plot
from aurora.lab.models.story import StoryArc
from aurora.lab.models.theme import Theme
from aurora.lab.narrator.context import ContextRecovery, TurningPointDetector
from aurora.lab.narrator.perspective import (
    NarrativePerspective,
    NarrativeRole,
    PerspectiveOrganizer,
    PerspectiveSelector,
)
from aurora.utils.time_utils import now_ts


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NarrativeElement:
    """叙事重构中的单个元素。

    代表具有其在叙事结构中角色的内存片段。
    """
    plot_id: str
    role: NarrativeRole
    content: str
    timestamp: float

    # 叙事意义（计算得出，非硬编码）
    significance: float = 0.5

    # 因果连接
    causes: List[str] = field(default_factory=list)      # 导致此事件的情节ID
    effects: List[str] = field(default_factory=list)     # 由此事件导致的情节ID

    # 叙事注解
    annotation: str = ""                                  # 叙事评论
    tension_level: float = 0.0                           # 此点的张力
    
    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为JSON兼容字典。"""
        return {
            "plot_id": self.plot_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "significance": self.significance,
            "causes": self.causes,
            "effects": self.effects,
            "annotation": self.annotation,
            "tension_level": self.tension_level,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "NarrativeElement":
        """从状态字典重构。"""
        return cls(
            plot_id=d["plot_id"],
            role=NarrativeRole(d["role"]),
            content=d["content"],
            timestamp=d["timestamp"],
            significance=d.get("significance", 0.5),
            causes=d.get("causes", []),
            effects=d.get("effects", []),
            annotation=d.get("annotation", ""),
            tension_level=d.get("tension_level", 0.0),
        )


@dataclass
class NarrativeTrace:
    """叙事重构过程的追踪。

    包含完整的叙事及关于重构过程的元数据。
    """
    query: str
    perspective: NarrativePerspective
    elements: List[NarrativeElement]

    # 重构元数据
    created_ts: float = field(default_factory=now_ts)
    reconstruction_confidence: float = 0.5

    # 视角选择概率
    perspective_probs: Dict[str, float] = field(default_factory=dict)

    # 识别的转折点
    turning_point_ids: List[str] = field(default_factory=list)

    # 因果链摘要
    causal_chain_depth: int = 0

    # 生成的叙事文本
    narrative_text: str = ""
    
    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为JSON兼容字典。"""
        return {
            "query": self.query,
            "perspective": self.perspective.value,
            "elements": [e.to_state_dict() for e in self.elements],
            "created_ts": self.created_ts,
            "reconstruction_confidence": self.reconstruction_confidence,
            "perspective_probs": self.perspective_probs,
            "turning_point_ids": self.turning_point_ids,
            "causal_chain_depth": self.causal_chain_depth,
            "narrative_text": self.narrative_text,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "NarrativeTrace":
        """从状态字典重构。"""
        return cls(
            query=d["query"],
            perspective=NarrativePerspective(d["perspective"]),
            elements=[NarrativeElement.from_state_dict(e) for e in d["elements"]],
            created_ts=d.get("created_ts", now_ts()),
            reconstruction_confidence=d.get("reconstruction_confidence", 0.5),
            perspective_probs=d.get("perspective_probs", {}),
            turning_point_ids=d.get("turning_point_ids", []),
            causal_chain_depth=d.get("causal_chain_depth", 0),
            narrative_text=d.get("narrative_text", ""),
        )


# =============================================================================
# Narrator Engine
# =============================================================================

class NarratorEngine:
    """用于叙事重构和故事讲述的引擎。

    通过以下方式将碎片化的记忆转变为连贯的叙事：
    1. 上下文自适应视角选择
    2. 因果链重构
    3. 转折点识别
    4. 叙事结构生成

    所有决策都是概率性的 - 没有硬编码的阈值。

    属性：
        metric: 用于相似度计算的学习度量
        seed: 用于可重现性的随机种子
        rng: 随机数生成器

        perspective_beliefs: 视角有效性的Beta后验信念
        reconstruction_cache: 最近重构的缓存
    """
    
    def __init__(
        self,
        metric: LowRankMetric,
        vindex: Optional[VectorIndex] = None,
        graph: Optional[MemoryGraph] = None,
        seed: int = 0,
    ):
        """初始化叙述者引擎。

        参数：
            metric: 用于相似度计算的学习低秩度量
            vindex: 用于检索的可选向量索引
            graph: 用于因果追踪的可选内存图
            seed: 用于可重现性的随机种子
        """
        self.metric = metric
        self.vindex = vindex
        self.graph = graph
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Initialize sub-components
        self._perspective_selector = PerspectiveSelector(
            metric=metric,
            rng=self.rng,
        )
        self._organizer = PerspectiveOrganizer(
            metric=metric,
            rng=self.rng,
        )
        self._context_recovery = ContextRecovery(
            metric=metric,
            rng=self.rng,
            graph=graph,
        )
        self._turning_point_detector = TurningPointDetector(rng=self.rng)
        
        # Reconstruction cache (LRU-style, limited size)
        self._reconstruction_cache: Dict[str, NarrativeTrace] = {}
        self._cache_max_size = 100
    
    @property
    def perspective_beliefs(self) -> Dict[NarrativePerspective, Tuple[float, float]]:
        """Get perspective beliefs from the selector."""
        return self._perspective_selector.perspective_beliefs
    
    @perspective_beliefs.setter
    def perspective_beliefs(
        self, value: Dict[NarrativePerspective, Tuple[float, float]]
    ) -> None:
        """Set perspective beliefs on the selector."""
        self._perspective_selector.perspective_beliefs = value
    
    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------
    
    def reconstruct_story(
        self,
        query: str,
        plots: List[Plot],
        perspective: Optional[NarrativePerspective] = None,
        stories: Optional[Dict[str, StoryArc]] = None,
        themes: Optional[Dict[str, Theme]] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> NarrativeTrace:
        """从内存片段重构叙事。

        根据选定的（或自动选定的）视角将给定的情节组织成连贯的叙事结构。

        参数：
            query: 驱动重构的查询或上下文
            plots: 要包含在叙事中的Plot对象列表
            perspective: 可选的特定视角（如果为None则自动选择）
            stories: 用于更丰富叙事的可选故事上下文
            themes: 用于抽象的可选主题上下文
            query_embedding: 可选的预计算查询嵌入

        返回：
            包含重构叙事的NarrativeTrace
        """
        if not plots:
            return NarrativeTrace(
                query=query,
                perspective=perspective or NarrativePerspective.CHRONOLOGICAL,
                elements=[],
                narrative_text="没有找到相关的记忆片段。",
            )

        # 如果未指定，自动选择视角
        perspective_probs = {}
        if perspective is None:
            perspective, perspective_probs = self.select_perspective(
                query=query,
                plots=plots,
                stories=stories,
                themes=themes,
                query_embedding=query_embedding,
            )

        # 根据视角组织情节
        elements = self._organize_by_perspective(
            plots=plots,
            perspective=perspective,
            query=query,
            query_embedding=query_embedding,
        )

        # 识别转折点
        turning_points = self._turning_point_detector.detect_from_elements(
            [e.to_state_dict() for e in elements]
        )
        turning_point_ids = [tp["plot_id"] for tp in turning_points]

        # 分配叙事角色
        elements = self._assign_narrative_roles(elements, turning_point_ids)

        # 生成叙事文本
        narrative_text = self._generate_narrative_text(
            elements=elements,
            perspective=perspective,
            query=query,
            stories=stories,
            themes=themes,
        )

        # 计算重构置信度
        confidence = self._compute_reconstruction_confidence(elements, perspective)

        trace = NarrativeTrace(
            query=query,
            perspective=perspective,
            elements=elements,
            reconstruction_confidence=confidence,
            perspective_probs=perspective_probs,
            turning_point_ids=turning_point_ids,
            narrative_text=narrative_text,
        )

        # 缓存重构
        cache_key = self._compute_cache_key(query, [p.id for p in plots])
        self._cache_reconstruction(cache_key, trace)

        return trace
    
    def select_perspective(
        self,
        query: str,
        plots: Optional[List[Plot]] = None,
        stories: Optional[Dict[str, StoryArc]] = None,
        themes: Optional[Dict[str, Theme]] = None,
        context: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[NarrativePerspective, Dict[str, float]]:
        """选择最合适的叙事视角。

        使用多个信号概率组合：
        - 查询特征
        - 内存结构（时间跨度、对比、主题）
        - 历史有效性（从反馈中学习）

        参数：
            query: 查询文本
            plots: 要考虑的可选情节列表
            stories: 用于上下文的可选故事
            themes: 用于抽象信号的可选主题
            context: 可选的附加上下文
            query_embedding: 可选的预计算查询嵌入

        返回：
            (选定的视角, 概率字典) 元组
        """
        return self._perspective_selector.select_perspective(
            query=query,
            plots=plots,
            stories=stories,
            themes=themes,
            context=context,
            query_embedding=query_embedding,
        )
    
    def recover_context(
        self,
        plot: Plot,
        plots_dict: Dict[str, Plot],
        depth: int = 3,
    ) -> List[Plot]:
        """恢复情节的因果上下文。

        通过因果链和时间连接追踪回溯
        以查找导致此情节的上下文。

        参数：
            plot: 要恢复上下文的情节
            plots_dict: 所有可用情节的字典
            depth: 要追踪的因果链的最大深度

        返回：
            形成因果上下文的情节列表（按相关性排序）
        """
        return self._context_recovery.recover_context(
            plot=plot,
            plots_dict=plots_dict,
            depth=depth,
        )
    
    def identify_turning_points(
        self,
        plots: List[Plot],
        stories: Optional[Dict[str, StoryArc]] = None,
    ) -> List[Plot]:
        """识别情节序列中的转折点。

        转折点是显著变化的时刻：
        - 高张力后跟随解决
        - 关系动态的转变
        - 身份相关事件

        使用概率检测，而不是固定阈值。

        参数：
            plots: 要分析的情节列表
            stories: 可选的故事上下文

        返回：
            被识别为转折点的情节列表
        """
        return self._context_recovery.identify_turning_points(
            plots=plots,
            stories=stories,
        )
    
    # -------------------------------------------------------------------------
    # Perspective-specific organization
    # -------------------------------------------------------------------------
    
    def _organize_by_perspective(
        self,
        plots: List[Plot],
        perspective: NarrativePerspective,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[NarrativeElement]:
        """根据选定的视角组织情节。"""

        # 使用组织器获取元素字典
        if perspective == NarrativePerspective.CHRONOLOGICAL:
            element_dicts = self._organizer.organize_chronological(
                plots, self._compute_significance
            )
        elif perspective == NarrativePerspective.RETROSPECTIVE:
            element_dicts = self._organizer.organize_retrospective(
                plots, query, query_embedding, self._compute_significance
            )
        elif perspective == NarrativePerspective.CONTRASTIVE:
            element_dicts = self._organizer.organize_contrastive(
                plots, self._compute_significance
            )
        elif perspective == NarrativePerspective.FOCUSED:
            element_dicts = self._organizer.organize_focused(
                plots, query, query_embedding, self._compute_significance
            )
        elif perspective == NarrativePerspective.ABSTRACTED:
            element_dicts = self._organizer.organize_abstracted(
                plots, self._compute_significance
            )
        else:
            element_dicts = self._organizer.organize_chronological(
                plots, self._compute_significance
            )

        # 将字典转换为NarrativeElement对象
        elements = []
        for d in element_dicts:
            element = NarrativeElement(
                plot_id=d["plot_id"],
                role=d["role"],
                content=d["content"],
                timestamp=d["timestamp"],
                significance=d["significance"],
                tension_level=d["tension_level"],
                annotation=d.get("annotation", ""),
            )
            elements.append(element)

        return elements
    
    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    
    def _assign_narrative_roles(
        self,
        elements: List[NarrativeElement],
        turning_point_ids: List[str],
    ) -> List[NarrativeElement]:
        """基于结构为元素分配叙事角色。"""
        if not elements:
            return elements

        n = len(elements)
        turning_ids = set(turning_point_ids)

        for i, element in enumerate(elements):
            # 基于位置的角色分配
            position_ratio = i / max(n - 1, 1)

            if element.plot_id in turning_ids:
                element.role = NarrativeRole.CLIMAX
            elif position_ratio < 0.15:
                element.role = NarrativeRole.EXPOSITION
            elif position_ratio < 0.5:
                element.role = NarrativeRole.RISING_ACTION
            elif position_ratio < 0.85:
                element.role = NarrativeRole.FALLING_ACTION
            else:
                element.role = NarrativeRole.RESOLUTION

        return elements
    
    def _generate_narrative_text(
        self,
        elements: List[NarrativeElement],
        perspective: NarrativePerspective,
        query: str,
        stories: Optional[Dict[str, StoryArc]] = None,
        themes: Optional[Dict[str, Theme]] = None,
    ) -> str:
        """从元素生成自然语言叙事。"""
        if not elements:
            return "没有可用的记忆片段来构建叙事。"

        # 基于视角构建叙事
        parts = []

        # 基于视角的开场
        perspective_openings = {
            NarrativePerspective.CHRONOLOGICAL: "按时间顺序回顾：",
            NarrativePerspective.RETROSPECTIVE: "回望这段经历：",
            NarrativePerspective.CONTRASTIVE: "对比分析：",
            NarrativePerspective.FOCUSED: f"关于「{query[:20]}...」的记忆：",
            NarrativePerspective.ABSTRACTED: "从这些经历中可以看到：",
        }
        parts.append(perspective_openings.get(perspective, "记忆重构："))

        # 添加元素
        for element in elements:
            role_markers = {
                NarrativeRole.EXPOSITION: "【背景】",
                NarrativeRole.RISING_ACTION: "",
                NarrativeRole.CLIMAX: "【转折】",
                NarrativeRole.FALLING_ACTION: "",
                NarrativeRole.RESOLUTION: "【结果】",
            }
            marker = role_markers.get(element.role, "")

            content = element.content
            if element.annotation:
                content = f"({element.annotation}) {content}"

            if marker:
                parts.append(f"\n{marker} {content}")
            else:
                parts.append(f"\n• {content}")

        # 如果是抽象视角，添加主题摘要
        if perspective == NarrativePerspective.ABSTRACTED and themes:
            parts.append("\n\n【主题总结】")
            for theme in list(themes.values())[:3]:
                if theme.identity_dimension:
                    parts.append(f"\n• {theme.identity_dimension}")
                elif theme.name:
                    parts.append(f"\n• {theme.name}")

        return "".join(parts)
    
    def _compute_significance(self, plot: Plot) -> float:
        """计算情节的叙事意义。"""
        base = 0.5

        # 张力贡献
        base += 0.2 * plot.tension

        # 身份影响贡献
        if plot.has_identity_impact():
            base += 0.2

        # 访问计数（频繁访问 = 更重要）
        base += 0.1 * min(1.0, plot.access_count / 10.0)

        return min(1.0, base)

    def _compute_reconstruction_confidence(
        self,
        elements: List[NarrativeElement],
        perspective: NarrativePerspective,
    ) -> float:
        """计算叙事重构的置信度。"""
        if not elements:
            return 0.0

        # 基于元素数量的基础置信度
        count_factor = min(1.0, len(elements) / 5.0)

        # 意义分布（更高的意义 = 更有信心）
        significances = [e.significance for e in elements]
        sig_factor = float(np.mean(significances)) if significances else 0.5

        # 视角信念因子
        a, b = self.perspective_beliefs.get(perspective, (1.0, 1.0))
        belief_factor = a / (a + b)

        return 0.4 * count_factor + 0.3 * sig_factor + 0.3 * belief_factor

    def _compute_cache_key(self, query: str, plot_ids: List[str]) -> str:
        """计算重构的缓存键。"""
        plot_hash = hash(tuple(sorted(plot_ids)))
        return f"{hash(query)}_{plot_hash}"

    def _cache_reconstruction(self, key: str, trace: NarrativeTrace) -> None:
        """使用LRU驱逐缓存重构。"""
        if len(self._reconstruction_cache) >= self._cache_max_size:
            # 删除最旧的条目
            oldest_key = next(iter(self._reconstruction_cache))
            del self._reconstruction_cache[oldest_key]

        self._reconstruction_cache[key] = trace
    
    # -------------------------------------------------------------------------
    # Feedback and Learning
    # -------------------------------------------------------------------------
    
    def feedback_narrative(
        self,
        trace: NarrativeTrace,
        success: bool,
    ) -> None:
        """基于叙事反馈更新信念。

        参数：
            trace: 被使用的叙事追踪
            success: 叙事是否有帮助/成功
        """
        self._perspective_selector.feedback(trace.perspective, success)
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_state_dict(self) -> Dict[str, Any]:
        """将引擎状态序列化为JSON兼容字典。"""
        return {
            "seed": self._seed,
            "perspective_beliefs": {
                p.value: list(ab) for p, ab in self.perspective_beliefs.items()
            },
            "cache_max_size": self._cache_max_size,
        }
    
    @classmethod
    def from_state_dict(
        cls,
        d: Dict[str, Any],
        metric: LowRankMetric,
        vindex: Optional[VectorIndex] = None,
        graph: Optional[MemoryGraph] = None,
    ) -> "NarratorEngine":
        """从状态字典重构引擎。"""
        engine = cls(
            metric=metric,
            vindex=vindex,
            graph=graph,
            seed=d.get("seed", 0),
        )

        # 恢复信念
        if "perspective_beliefs" in d:
            for p_str, ab in d["perspective_beliefs"].items():
                p = NarrativePerspective(p_str)
                engine.perspective_beliefs[p] = tuple(ab)

        engine._cache_max_size = d.get("cache_max_size", 100)

        return engine


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    from aurora.lab.primitives.metric import LowRankMetric
    
    # Create a simple metric
    metric = LowRankMetric(dim=96, rank=32, seed=42)
    
    # Create narrator engine
    narrator = NarratorEngine(metric=metric, seed=42)
    
    print("NarratorEngine created successfully.")
    print(f"Perspectives: {[p.value for p in NarrativePerspective]}")
    
    # Test perspective selection (without plots)
    perspective, probs = narrator.select_perspective(
        query="如何处理记忆系统中的矛盾？",
        context="memory algorithm design",
    )
    print(f"\nSelected perspective: {perspective.value}")
    print(f"Probabilities: {probs}")
    
    # Test serialization
    state = narrator.to_state_dict()
    print(f"\nState dict keys: {list(state.keys())}")
    
    # Test reconstruction
    restored = NarratorEngine.from_state_dict(state, metric=metric)
    print(f"Restored engine seed: {restored._seed}")
