"""
AURORA 追踪和快照模型
=================================

用于检索追踪和演变快照的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from aurora.core.retrieval.query_analysis import QueryType
    from aurora.core.abstention import AbstentionResult


@dataclass
class RetrievalTrace:
    """
    检索操作的追踪，包含关系中心和时间线扩展。

    捕捉查询、中间状态和排序结果
    用于分析和学习。

    属性：
        query: 原始查询文本
        query_emb: 查询嵌入向量
        attractor_path: 均值漂移期间的嵌入路径
        ranked: 排序结果的 (id, score, kind) 元组列表

    关系中心扩展：
        asker_id: 提出查询的实体 ID
        activated_identity: 为此关系激活的身份

    查询类型感知：
        query_type: 检测到的或指定的查询类型，用于自适应检索

    时间线扩展（第一原理）：
        timeline_group: 显示知识演变的组织时间线
        include_historical: 是否包含历史（已过时）结果
    """

    query: str
    query_emb: np.ndarray
    attractor_path: List[np.ndarray]
    ranked: List[Tuple[str, float, str]]  # (id, score, kind)

    # 关系中心扩展
    asker_id: Optional[str] = None
    activated_identity: Optional[str] = None

    # 查询类型感知（使用 Any 避免运行时循环导入）
    query_type: Optional[Any] = None  # QueryType 枚举

    # 时间线扩展 - 第一原理：保留时间演变
    timeline_group: Optional[TimelineGroup] = None
    include_historical: bool = True  # 默认包含完整历史

    # 弃权检测
    abstention: Optional[Any] = None  # AbstentionResult（使用 Any 避免循环导入）


@dataclass
class EvolutionSnapshot:
    """
    内存状态的只读快照，用于后台演变。

    捕捉计算演变所需的一切，而不修改状态。
    这使得并发演变计算成为可能。

    属性：
        story_ids: 所有故事 ID 的列表
        story_statuses: 故事 ID 到状态的映射
        story_centroids: 故事 ID 到质心嵌入的映射
        story_tension_curves: 故事 ID 到张力历史的映射
        story_updated_ts: 故事 ID 到最后更新时间戳的映射
        story_gap_means: 故事 ID 到平均间隔的映射

        theme_ids: 所有主题 ID 的列表
        theme_story_counts: 主题 ID 到故事数的映射
        theme_prototypes: 主题 ID 到原型嵌入的映射

        crp_theme_alpha: 主题的 CRP 浓度参数
        rng_state: 随机数生成器状态，用于可重现性
    """

    # 故事数据
    story_ids: List[str]
    story_statuses: Dict[str, str]
    story_centroids: Dict[str, Optional[np.ndarray]]
    story_tension_curves: Dict[str, List[float]]
    story_updated_ts: Dict[str, float]
    story_gap_means: Dict[str, float]

    # 主题数据
    theme_ids: List[str]
    theme_story_counts: Dict[str, int]
    theme_prototypes: Dict[str, Optional[np.ndarray]]

    # CRP 参数
    crp_theme_alpha: float

    # 用于可重现性的 RNG 状态
    rng_state: Dict[str, Any]


@dataclass
class QueryHit:
    """
    单个检索命中。

    代表内存查询的一个结果，包含
    内存单元 ID、类型、相关性分数和文本片段。

    属性：
        id: 内存单元的唯一标识符
        kind: 内存类型（"plot"、"story"、"theme"）
        score: 相关性分数（更高更好）
        snippet: 内容的文本片段或摘要
        metadata: 关于命中的可选附加元数据
    """
    id: str
    kind: str  # "plot"、"story"、"theme"
    score: float
    snippet: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeTimeline:
    """
    单个主题/实体的知识演变时间线。

    第一原理：
    - 在叙事心理学中，过去的事实不是"删除"而是用过去时重新定位
    - "我住在北京" → "我**曾经**住在北京"（仍然是真的，只是时间上的）
    - 已过时 ≠ 无效；已过时 = 过去的真实

    此结构捕捉知识的完整演变，允许
    语义理解层（LLM）基于完整背景做出决策
    而不是让检索层任意过滤结果。

    属性：
        chain: 从最旧到最新的情节 ID 列表（按时间顺序）
        current_id: 当前活跃情节的 ID（如果全部已过时则为 None）
        topic_signature: 用于分组相似更新的语义签名
        match_score: 查询的最佳语义匹配分数

    示例时间线：
        chain: ["plot-001", "plot-002", "plot-003"]
        current_id: "plot-003"

        代表：
        - plot-001: "我住在北京" (2024-01) → 历史
        - plot-002: "我搬到了上海" (2024-06) → 历史
        - plot-003: "我搬到了深圳" (2024-12) → 当前
    """
    chain: List[str]                    # [最旧的情节_id, ..., 最新的情节_id]
    current_id: Optional[str]           # 活跃情节的 ID（最新的非已过时）
    topic_signature: str                # 主题的语义签名
    match_score: float = 0.0            # 查询的最佳匹配分数

    def __len__(self) -> int:
        """返回时间线中的情节数。"""
        return len(self.chain)

    def is_single_version(self) -> bool:
        """检查此时间线是否只有一个版本（无更新）。"""
        return len(self.chain) == 1

    def has_evolution(self) -> bool:
        """检查此时间线是否显示知识演变。"""
        return len(self.chain) > 1

    def get_historical_ids(self) -> List[str]:
        """获取历史（已过时）情节的 ID。"""
        if self.current_id is None:
            return self.chain  # 全部是历史
        return [pid for pid in self.chain if pid != self.current_id]


@dataclass
class TimelineGroup:
    """
    检索操作中相关知识时间线的分组。

    第一原理：
    - 检索应返回结构化的时间信息
    - 让 LLM 看到带有时间标记的完整背景
    - 不在检索层过滤；让语义理解层决定

    此结构将检索结果组织成有意义的时间线，
    使得时间推理查询（"我曾经住在哪里？"）和
    知识更新查询（"我现在住在哪里？"）都能正确工作。

    属性：
        timelines: KnowledgeTimeline 对象列表
        standalone_results: 不属于任何更新链的结果
        total_results: 所有时间线中的唯一情节总数
    """
    timelines: List[KnowledgeTimeline] = field(default_factory=list)
    standalone_results: List[Tuple[str, float, str]] = field(default_factory=list)

    @property
    def total_results(self) -> int:
        """唯一结果的总数。"""
        timeline_count = sum(len(t.chain) for t in self.timelines)
        return timeline_count + len(self.standalone_results)

    def get_all_plot_ids(self) -> List[str]:
        """从时间线和独立结果获取所有情节 ID。"""
        ids: List[str] = []
        for timeline in self.timelines:
            ids.extend(timeline.chain)
        for nid, _, kind in self.standalone_results:
            if kind == "plot":
                ids.append(nid)
        return ids

    def get_current_state_ids(self) -> List[str]:
        """获取代表当前状态的 ID（用于知识更新查询）。"""
        ids: List[str] = []
        for timeline in self.timelines:
            if timeline.current_id:
                ids.append(timeline.current_id)
        for nid, _, kind in self.standalone_results:
            ids.append(nid)
        return ids


@dataclass
class EvolutionPatch:
    """
    从演变计算的变化，要原子性地应用。

    代表一个 diff，可在演变计算完成后
    应用到内存状态。

    属性：
        status_changes: 故事 ID 到新状态的映射
        theme_assignments: (story_id, theme_id) 分配列表
        new_themes: 要创建的新主题列表 (theme_id, prototype)
    """

    # 故事状态变化：story_id -> 新状态
    status_changes: Dict[str, str] = field(default_factory=dict)

    # 主题分配：[(story_id, theme_id)]
    theme_assignments: List[Tuple[str, str]] = field(default_factory=list)

    # 要创建的新主题：[(theme_id, prototype)]
    new_themes: List[Tuple[str, np.ndarray]] = field(default_factory=list)
