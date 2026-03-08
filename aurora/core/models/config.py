"""
AURORA 配置模型
============================

AURORA 记忆系统的配置数据类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AlgorithmConfig:
    """
    从硬编码值提取的算法超参数。

    这些参数控制可学习组件的行为，
    可以针对不同的用例进行调整。

    基准优化说明：
    - AR: 需要精确检索，较少平滑
    - TTL: 需要快速学习，更高的门精度
    - LRU: 需要良好的聚合，适度的 KDE 带宽
    - CR: 需要冲突检测，敏感的度量学习
    """

    # --- OnlineKDE 参数 ---
    # 较低的 k_sigma = 较窄的带宽 = 更尖锐的惊奇峰值
    # 通过使新颖信息更易区分来帮助 AR
    kde_k_sigma: int = 20  # 从 25 降低以获得更尖锐的惊奇检测

    # --- LowRankMetric 参数 ---
    metric_init_noise: float = 0.01  # 初始化噪声规模
    metric_window_size: int = 8000   # 从 10000 降低以加快适应
    metric_decay_factor: float = 0.6  # 从 0.5 增加以减缓遗忘

    # --- ThompsonBernoulliGate 参数 ---
    # 更高的初始精度 = 更少的探索 = 更快的收敛
    # 通过更快地学习规则来帮助 TTL
    gate_init_precision: float = 5e-2  # 从 1e-2 增加以加快学习
    gate_forgetting_factor: float = 0.98  # 从 0.99 降低以增加可塑性
    gate_rms_decay: float = 0.95  # 从 0.99 降低以加快适应

    # --- StoryModel 参数 ---
    # 较低的 beta = 较少的平滑 = 更尖锐的演员可能性
    # 通过区分与不同演员的对话来帮助 AR
    story_dirichlet_beta: float = 0.8  # 从 1.0 降低

    # --- FieldRetriever 参数 ---
    # 为基准优化：平衡精度（AR）与覆盖范围（LRU）
    retrieval_damping: float = 0.80  # 从 0.85 降低以获得更好的直接匹配
    retrieval_mean_shift_steps: int = 6  # 从 8 降低以减少平滑
    retrieval_initial_seed_k: int = 60  # 从 50 增加以获得更好的召回率
    retrieval_reseed_k: int = 50  # 从 60 降低以提高精度
    retrieval_pagerank_max_iter: int = 40  # 从 50 降低，收敛更快

    # --- 滑动窗口 ---
    # 更大的窗口通过考虑更多上下文来帮助 LRU
    recent_plots_window: int = 250  # 从 200 增加

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "kde_k_sigma": self.kde_k_sigma,
            "metric_init_noise": self.metric_init_noise,
            "metric_window_size": self.metric_window_size,
            "metric_decay_factor": self.metric_decay_factor,
            "gate_init_precision": self.gate_init_precision,
            "gate_forgetting_factor": self.gate_forgetting_factor,
            "gate_rms_decay": self.gate_rms_decay,
            "story_dirichlet_beta": self.story_dirichlet_beta,
            "retrieval_damping": self.retrieval_damping,
            "retrieval_mean_shift_steps": self.retrieval_mean_shift_steps,
            "retrieval_initial_seed_k": self.retrieval_initial_seed_k,
            "retrieval_reseed_k": self.retrieval_reseed_k,
            "retrieval_pagerank_max_iter": self.retrieval_pagerank_max_iter,
            "recent_plots_window": self.recent_plots_window,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "AlgorithmConfig":
        """从状态字典重构。"""
        return cls(
            kde_k_sigma=d.get("kde_k_sigma", 25),
            metric_init_noise=d.get("metric_init_noise", 0.01),
            metric_window_size=d.get("metric_window_size", 10000),
            metric_decay_factor=d.get("metric_decay_factor", 0.5),
            gate_init_precision=d.get("gate_init_precision", 1e-2),
            gate_forgetting_factor=d.get("gate_forgetting_factor", 0.99),
            gate_rms_decay=d.get("gate_rms_decay", 0.99),
            story_dirichlet_beta=d.get("story_dirichlet_beta", 1.0),
            retrieval_damping=d.get("retrieval_damping", 0.85),
            retrieval_mean_shift_steps=d.get("retrieval_mean_shift_steps", 8),
            retrieval_initial_seed_k=d.get("retrieval_initial_seed_k", 50),
            retrieval_reseed_k=d.get("retrieval_reseed_k", 60),
            retrieval_pagerank_max_iter=d.get("retrieval_pagerank_max_iter", 50),
            recent_plots_window=d.get("recent_plots_window", 200),
        )


@dataclass
class MemoryConfig:
    """
    核心记忆配置。

    这些参数控制记忆系统的结构和容量。
    """

    dim: int = 384  # 嵌入维度
    metric_rank: int = 64  # 低秩度量维度

    # 有界记忆压力（资源约束是第一性原理现实）
    max_plots: int = 5000  # 要保留的最大 Plot 数量
    kde_reservoir: int = 4096  # KDE 水库采样大小

    # CRP 浓度先验
    # 基准优化：
    # - 较低的 story_alpha = 较少的故事 = 更好的 LRU 聚合
    # - 较高的 theme_alpha = 更多的主题 = 更好的身份维度捕获
    story_alpha: float = 0.7  # 从 1.0 降低：为 LRU 优先选择更少、更丰富的故事
    theme_alpha: float = 0.8  # 从 0.5 增加：捕获更多身份模式

    # 编码门特征维度
    gate_feature_dim: int = 6  # 编码门的特征数量

    # 检索偏好
    retrieval_kinds: Tuple[str, ...] = ("theme", "story", "plot")

    # 向量索引："auto"（如果可用则使用 FAISS）、"faiss"、"brute"
    vector_backend: str = "auto"

    # FAISS HNSW 参数（仅在 vector_backend 为 "faiss" 或 "auto" 时使用）
    faiss_m: int = 32  # 每层 HNSW 连接数（8-64，更高 = 更好的召回率）
    faiss_ef_construction: int = 64  # 构造时搜索深度
    faiss_ef_search: int = 32  # 查询时搜索深度（增加以获得更好的召回率）

    # 基准模式：强制存储所有 Plot，绕过 VOI 门控
    # 对于 LongMemEval 等基准至关重要，其中每个回合可能包含关键信息
    benchmark_mode: bool = False

    # 算法配置（嵌套，为了向后兼容是可选的）
    algo: Optional[AlgorithmConfig] = field(default=None)

    def __post_init__(self):
        if self.algo is None:
            object.__setattr__(self, 'algo', AlgorithmConfig())

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "dim": self.dim,
            "metric_rank": self.metric_rank,
            "max_plots": self.max_plots,
            "kde_reservoir": self.kde_reservoir,
            "story_alpha": self.story_alpha,
            "theme_alpha": self.theme_alpha,
            "gate_feature_dim": self.gate_feature_dim,
            "retrieval_kinds": list(self.retrieval_kinds),
            "vector_backend": self.vector_backend,
            "faiss_m": self.faiss_m,
            "faiss_ef_construction": self.faiss_ef_construction,
            "faiss_ef_search": self.faiss_ef_search,
            "benchmark_mode": self.benchmark_mode,
            "algo": self.algo.to_state_dict() if self.algo else None,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "MemoryConfig":
        """从状态字典重构。"""
        algo_dict = d.get("algo")
        algo = AlgorithmConfig.from_state_dict(algo_dict) if algo_dict else AlgorithmConfig()
        return cls(
            dim=d.get("dim", 384),
            metric_rank=d.get("metric_rank", 64),
            max_plots=d.get("max_plots", 5000),
            kde_reservoir=d.get("kde_reservoir", 4096),
            story_alpha=d.get("story_alpha", 1.0),
            theme_alpha=d.get("theme_alpha", 0.5),
            gate_feature_dim=d.get("gate_feature_dim", 6),
            retrieval_kinds=tuple(d.get("retrieval_kinds", ["theme", "story", "plot"])),
            vector_backend=d.get("vector_backend", "auto"),
            faiss_m=d.get("faiss_m", 32),
            faiss_ef_construction=d.get("faiss_ef_construction", 64),
            faiss_ef_search=d.get("faiss_ef_search", 32),
            benchmark_mode=d.get("benchmark_mode", False),
            algo=algo,
        )
