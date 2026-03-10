"""
AURORA 基准测试接口
===========================

用于基准测试 AURORA 内存系统的统一评估接口。

本模块提供:
    - BenchmarkCapability: 评估维度的枚举
    - EvaluationMethod: 评估方法的枚举
    - BenchmarkInstance: 单个评估实例的数据类
    - BenchmarkResult: 评估结果的数据类
    - EvaluationMetrics: 跨实例的聚合指标
    - BenchmarkAdapter: 基准适配器的抽象基类
    - AURORABenchmarkRunner: 执行基准测试的主运行器

设计哲学:
    - 适配器将外部基准格式转换为 AURORA 的接口
    - 结果与丰富的元数据一起收集以供分析
    - 指标单独计算，允许灵活的评估
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts

logger = logging.getLogger(__name__)


# =============================================================================
# 模拟模型检测
# =============================================================================


def _is_mock_embedder(embedder: Any) -> bool:
    """检查嵌入器是否为模拟/哈希嵌入（不用于生产基准测试）。

    Args:
        embedder: 嵌入提供者实例

    Returns:
        如果嵌入器是 HashEmbedding 或类似的模拟，返回 True
    """
    if embedder is None:
        return True

    # 检查类名
    class_name = type(embedder).__name__
    mock_names = {"HashEmbedding", "MockEmbedding", "FakeEmbedding", "DummyEmbedding"}
    if class_name in mock_names:
        return True

    # 检查模块路径
    module = type(embedder).__module__
    if "hash" in module.lower() or "mock" in module.lower():
        return True

    return False


def _is_mock_llm(llm_provider: Any) -> bool:
    """检查 LLM 提供者是否为模拟/测试替身（不用于生产基准测试）。

    Args:
        llm_provider: LLM 提供者实例

    Returns:
        如果提供者是测试替身或类似实现，返回 True
    """
    if llm_provider is None:
        return True

    # 检查类名
    class_name = type(llm_provider).__name__
    mock_names = {"MockLLM", "FakeLLM", "DummyLLM", "MockProvider"}
    if class_name in mock_names:
        return True

    # 检查模块路径
    module = type(llm_provider).__module__
    if "mock" in module.lower():
        return True

    return False


def verify_benchmark_ready(
    memory: Any,
    llm: Any = None,
    verbose: bool = True,
) -> Tuple[bool, List[str]]:
    """验证配置是否准备好进行有意义的基准评估。

    检查:
    1. 嵌入模型不是模拟 (HashEmbedding)
    2. LLM 提供者不是测试替身或伪实现

    使用模拟模型将导致显著较低的基准分数
    这不反映实际系统性能。

    Args:
        memory: 要检查的 AURORA 内存实例
        llm: 可选的 LLM 提供者检查
        verbose: 如果为 True，打印警告到控制台

    Returns:
        (is_ready, warnings) 的元组:
            - is_ready: 如果配置了真实模型，返回 True
            - warnings: 警告消息列表

    Example:
        from aurora.benchmarks.interface import verify_benchmark_ready
        from aurora.soul.engine import AuroraSoul, SoulConfig

        memory = AuroraSoul(cfg=SoulConfig())
        is_ready, warnings = verify_benchmark_ready(memory)

        if not is_ready:
            print("⚠️ 配置问题:")
            for w in warnings:
                print(f"  - {w}")
    """
    warnings: List[str] = []
    is_ready = True

    # Check embedder
    embedder = getattr(memory, "embedder", None)
    if _is_mock_embedder(embedder):
        is_ready = False
        embedder_name = type(embedder).__name__ if embedder else "None"
        warnings.append(
            f"Embedding model is '{embedder_name}' (non-production). "
            "Use a real embedding model (e.g., Bailian, Ark) for accurate benchmark results."
        )

    # Check LLM provider
    if llm is not None and _is_mock_llm(llm):
        is_ready = False
        llm_name = type(llm).__name__
        warnings.append(
            f"LLM provider is '{llm_name}' (non-production). "
            "Use a real LLM provider for accurate benchmark results."
        )

    # Print warnings if verbose
    if verbose and warnings:
        print("\n" + "=" * 70)
        print("⚠️  BENCHMARK CONFIGURATION WARNING")
        print("=" * 70)
        for w in warnings:
            print(f"  • {w}")
        print("-" * 70)
        print("  Non-production models will result in significantly lower benchmark scores")
        print("  that do NOT reflect actual system performance!")
        print("=" * 70 + "\n")

    return is_ready, warnings


# -----------------------------------------------------------------------------
# 能力维度
# -----------------------------------------------------------------------------


class BenchmarkCapability(Enum):
    """
    内存系统的评估能力维度。

    这些能力映射到内存性能的特定方面:

    ACCURATE_RETRIEVAL:
        从扩展交互历史中提取精确信息的能力。
        测试: 事实回忆、实体提取、细节准确性。
        AURORA: query() + FieldRetriever

    TEST_TIME_LEARNING:
        在对话期间应用新规则而无需参数更新的能力。
        测试: 规则应用、约束遵循、偏好适应。
        AURORA: ingest() + evolve()

    LONG_RANGE_UNDERSTANDING:
        在扩展叙述中形成连贯摘要的能力。
        测试: 摘要、主题提取、叙事弧检测。
        AURORA: Story aggregation + Theme emergence

    CONFLICT_RESOLUTION:
        适当处理矛盾信息更新的能力。
        测试: 事实更新、偏好变化、时间推理。
        AURORA: TensionManager + CoherenceGuardian
    """

    ACCURATE_RETRIEVAL = "accurate_retrieval"
    TEST_TIME_LEARNING = "test_time_learning"
    LONG_RANGE_UNDERSTANDING = "long_range_understanding"
    CONFLICT_RESOLUTION = "conflict_resolution"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> "BenchmarkCapability":
        """
        从字符串值创建能力。

        Args:
            s: 字符串值 (例如 "accurate_retrieval")

        Returns:
            BenchmarkCapability 枚举成员

        Raises:
            ValueError: 如果字符串与任何能力不匹配
        """
        for cap in cls:
            if cap.value == s:
                return cap
        raise ValueError(f"Unknown capability: {s}")


class EvaluationMethod(Enum):
    """用于评估预测与真实值的方法。

    EXACT_MATCH: 精确字符串匹配（不区分大小写）
    CONTAINS: 检查预测是否包含真实值
    FUZZY: 使用编辑距离的模糊字符串匹配
    LLM_JUDGE: LLM-as-judge 评估
    ROUGE: 基于 ROUGE 的评估（用于摘要）
    SEMANTIC: 基于语义相似度的评估
    """

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    FUZZY = "fuzzy"
    LLM_JUDGE = "llm_judge"
    ROUGE = "rouge"
    SEMANTIC = "semantic"


@dataclass
class EvaluationConfig:
    """基准评估的配置。

    Attributes:
        use_llm_judge: 是否使用 LLM-as-Judge 进行评估
        judge_model: 用于判断的模型（如果 use_llm_judge=True）
        judge_temperature: 判断模型的温度
        max_retries: 失败时的最大重试次数
        timeout_s: 每个实例的超时时间（秒）
        save_traces: 是否保存检索跟踪
        verbose: 是否打印进度
    """

    use_llm_judge: bool = True
    judge_model: str = "gpt-4"
    judge_temperature: float = 0.0

    max_retries: int = 3
    timeout_s: float = 60.0

    save_traces: bool = True
    verbose: bool = True

    # 批处理设置
    batch_size: int = 1
    parallel_workers: int = 1

    # 过滤
    capabilities_filter: Optional[List["BenchmarkCapability"]] = None
    max_instances: Optional[int] = None


@dataclass
class EvaluationMetrics:
    """跨多个实例的聚合评估指标。

    Attributes:
        total_instances: 评估的总实例数
        correct_instances: 正确预测的数量
        accuracy: 总体准确率 [0, 1]

        avg_score: 跨实例的平均分数
        avg_latency_ms: 平均延迟（毫秒）
        avg_tokens: 每个查询使用的平均令牌数

        metrics_by_type: 按任务/推理类型分解的指标
        p50_latency_ms: 50 百分位数延迟
        p99_latency_ms: 99 百分位数延迟
    """

    total_instances: int = 0
    correct_instances: int = 0
    accuracy: float = 0.0

    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0

    metrics_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "total_instances": self.total_instances,
            "correct_instances": self.correct_instances,
            "accuracy": self.accuracy,
            "avg_score": self.avg_score,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_tokens": self.avg_tokens,
            "metrics_by_type": self.metrics_by_type,
            "p50_latency_ms": self.p50_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }


# -----------------------------------------------------------------------------
# 数据类
# -----------------------------------------------------------------------------


@dataclass
class BenchmarkInstance:
    """
    来自基准数据集的单个评估实例。

    代表一个测试用例，包含上下文、查询和预期答案。
    支持简单字符串上下文（MemoryAgentBench）和
    对话历史格式（LOCOMO）。

    Attributes:
        id: 此实例的唯一标识符
        capability: 正在测试的能力维度（对于基于 task_type 的可选）
        context: 对话历史或上下文作为字符串（用于向后兼容）
        query: 要评估的查询
        expected_answer: 预期答案（对于开放式任务可为 None）
        metadata: 其他基准特定的元数据

        # LOCOMO 风格基准的扩展字段
        task_type: 任务类型（例如 "qa"、"summarization"）
        conversation_history: 结构化对话历史
        ground_truth: 预期答案（expected_answer 的别名）
        reasoning_type: QA 所需的推理类型
        session_id: 会话标识符
        turn_number: 对话中的轮次号
        created_ts: 创建时间戳

    Example (MemoryAgentBench 风格):
        instance = BenchmarkInstance(
            id="mab_001",
            capability=BenchmarkCapability.ACCURATE_RETRIEVAL,
            context="User: I live in San Francisco...",
            query="Where does the user live?",
            expected_answer="San Francisco",
        )

    Example (LOCOMO 风格):
        instance = BenchmarkInstance(
            id="loc_001",
            task_type="qa",
            conversation_history=[{"speaker": "user", "text": "I live in SF"}],
            query="Where does the user live?",
            ground_truth="San Francisco",
            reasoning_type="single_hop",
        )
    """

    id: str
    query: str

    # 用于与 MemoryAgentBench 风格的向后兼容
    capability: Optional[BenchmarkCapability] = None
    context: str = ""
    expected_answer: Optional[str] = None

    # LOCOMO 风格基准的扩展字段
    task_type: str = "qa"
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    ground_truth: str = ""
    reasoning_type: Optional[str] = None
    session_id: Optional[str] = None
    turn_number: Optional[int] = None
    created_ts: float = field(default_factory=now_ts)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """同步 ground_truth 和 expected_answer。"""
        if self.ground_truth and not self.expected_answer:
            self.expected_answer = self.ground_truth
        elif self.expected_answer and not self.ground_truth:
            self.ground_truth = self.expected_answer

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        result = {
            "id": self.id,
            "query": self.query,
            "context": self.context,
            "expected_answer": self.expected_answer,
            "task_type": self.task_type,
            "conversation_history": self.conversation_history,
            "ground_truth": self.ground_truth,
            "reasoning_type": self.reasoning_type,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "created_ts": self.created_ts,
            "metadata": self.metadata,
        }
        if self.capability is not None:
            result["capability"] = self.capability.value
        return result

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "BenchmarkInstance":
        """从状态字典重建。"""
        capability = None
        if "capability" in d and d["capability"]:
            try:
                capability = BenchmarkCapability.from_string(d["capability"])
            except ValueError:
                pass

        return cls(
            id=d["id"],
            query=d.get("query", ""),
            capability=capability,
            context=d.get("context", ""),
            expected_answer=d.get("expected_answer"),
            task_type=d.get("task_type", "qa"),
            conversation_history=d.get("conversation_history", []),
            ground_truth=d.get("ground_truth", ""),
            reasoning_type=d.get("reasoning_type"),
            session_id=d.get("session_id"),
            turn_number=d.get("turn_number"),
            created_ts=d.get("created_ts", now_ts()),
            metadata=d.get("metadata", {}),
        )


@dataclass
class BenchmarkResult:
    """
    单个基准实例的评估结果。

    包含预测答案、预期答案、计算的分数
    和性能指标。支持 MemoryAgentBench 和 LOCOMO 格式。

    Attributes:
        instance_id: 评估实例的 ID
        capability: 测试的能力维度（可选）
        predicted: 模型的预测答案（别名: prediction）
        expected: 预期答案（如果可用）
        score: [0.0, 1.0] 范围内的评估分数
        latency_ms: 响应延迟（毫秒）

        # LOCOMO 风格基准的扩展字段
        task_type: 评估的任务类型
        prediction: 模型的预测（predicted 的别名）
        ground_truth: 预期答案（expected 的别名）
        is_correct: 二进制正确性标志
        tokens_used: 消耗的令牌数
        retrieval_count: 内存检索次数
        reasoning_trace: 推理步骤的跟踪
        error_message: 如果评估失败，则为错误消息
        evaluated_ts: 评估时间戳

        metadata: 其他结果元数据

    分数解释:
        1.0: 完美匹配 / 完全正确
        0.5-0.99: 部分正确
        0.0: 不正确 / 无匹配
    """

    instance_id: str
    score: float
    latency_ms: float = 0.0

    # 用于与 MemoryAgentBench 风格的向后兼容
    capability: Optional[BenchmarkCapability] = None
    predicted: str = ""
    expected: Optional[str] = None

    # LOCOMO 风格基准的扩展字段
    task_type: str = "qa"
    prediction: str = ""
    ground_truth: str = ""
    is_correct: bool = False
    tokens_used: int = 0
    retrieval_count: int = 0
    reasoning_trace: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    evaluated_ts: float = field(default_factory=now_ts)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """同步别名字段。"""
        if self.prediction and not self.predicted:
            self.predicted = self.prediction
        elif self.predicted and not self.prediction:
            self.prediction = self.predicted

        if self.ground_truth and not self.expected:
            self.expected = self.ground_truth
        elif self.expected and not self.ground_truth:
            self.ground_truth = self.expected or ""

        # 如果未显式设置，设置 is_correct
        if not self.is_correct and self.score >= 0.5:
            self.is_correct = True

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        result = {
            "instance_id": self.instance_id,
            "predicted": self.predicted,
            "expected": self.expected,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "task_type": self.task_type,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
            "tokens_used": self.tokens_used,
            "retrieval_count": self.retrieval_count,
            "reasoning_trace": self.reasoning_trace,
            "error_message": self.error_message,
            "evaluated_ts": self.evaluated_ts,
            "metadata": self.metadata,
        }
        if self.capability is not None:
            result["capability"] = self.capability.value
        return result

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        """从状态字典重建。"""
        capability = None
        if "capability" in d and d["capability"]:
            try:
                capability = BenchmarkCapability.from_string(d["capability"])
            except ValueError:
                pass

        return cls(
            instance_id=d["instance_id"],
            score=d["score"],
            latency_ms=d.get("latency_ms", 0.0),
            capability=capability,
            predicted=d.get("predicted", ""),
            expected=d.get("expected"),
            task_type=d.get("task_type", "qa"),
            prediction=d.get("prediction", ""),
            ground_truth=d.get("ground_truth", ""),
            is_correct=d.get("is_correct", False),
            tokens_used=d.get("tokens_used", 0),
            retrieval_count=d.get("retrieval_count", 0),
            reasoning_trace=d.get("reasoning_trace", []),
            error_message=d.get("error_message"),
            evaluated_ts=d.get("evaluated_ts", now_ts()),
            metadata=d.get("metadata", {}),
        )

    def check_correct(self, threshold: float = 0.5) -> bool:
        """检查结果是否被认为是正确的。"""
        return self.score >= threshold


# -----------------------------------------------------------------------------
# 内存接口协议
# -----------------------------------------------------------------------------


class MemoryProtocol(Protocol):
    """基准期望的内存接口协议。"""

    def ingest(self, text: str, **kwargs: Any) -> Any:
        """将文本摄入内存。"""
        ...

    def query(self, text: str, **kwargs: Any) -> Any:
        """查询内存以获取相关信息。"""
        ...

    def evolve(self) -> None:
        """触发内存演化/整合。"""
        ...

    def clear(self) -> None:
        """清除所有内存状态。"""
        ...


# -----------------------------------------------------------------------------
# 抽象适配器
# -----------------------------------------------------------------------------


class BenchmarkAdapter(ABC):
    """
    基准适配器的抽象基类。

    每个基准（例如 MemoryAgentBench、LOCOMO）应该有自己的适配器
    将基准的格式转换为 AURORA 的接口。

    子类必须实现:
        - name: 返回基准名称的属性
        - load_dataset: 从数据集路径加载实例
        - evaluate: 针对内存评估单个实例
        - aggregate_results: 计算聚合指标

    Example Implementation:
        class MemoryAgentBenchAdapter(BenchmarkAdapter):
            @property
            def name(self) -> str:
                return "MemoryAgentBench"

            def load_dataset(self, path: str) -> List[BenchmarkInstance]:
                # 从 HuggingFace 或本地路径加载
                ...
    """

    def __init__(self, llm_provider: Any = None, seed: int = 0) -> None:
        """初始化适配器。

        Args:
            llm_provider: 用于 LLM-as-judge 评估的可选 LLM 提供者
            seed: 用于可重复性的随机种子
        """
        self.llm = llm_provider
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """返回基准名称。"""

    @abstractmethod
    def load_dataset(self, path: str) -> List[BenchmarkInstance]:
        """
        从数据集加载评估实例。

        Args:
            path: 数据集的路径（本地路径或 HuggingFace 标识符）

        Returns:
            BenchmarkInstance 对象列表

        Raises:
            FileNotFoundError: 如果数据集路径不存在
            ValueError: 如果数据集格式无效
        """

    @abstractmethod
    def evaluate(
        self,
        instance: BenchmarkInstance,
        memory: MemoryProtocol,
    ) -> BenchmarkResult:
        """
        针对内存系统评估单个实例。

        此方法应该:
        1. 准备内存状态（如果需要，摄入上下文）
        2. 使用实例的查询查询内存
        3. 将结果与预期答案进行比较
        4. 返回带有分数和元数据的 BenchmarkResult

        Args:
            instance: 要评估的基准实例
            memory: AURORA 内存系统实例

        Returns:
            BenchmarkResult，包含预测答案、分数和延迟
        """

    def aggregate_results(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, float]:
        """
        将评估结果聚合为摘要指标。

        提供计算基本统计的默认实现。
        子类可以覆盖以获得基准特定的指标。

        Args:
            results: 单个评估结果列表

        Returns:
            将指标名称映射到值的字典，例如:
            {
                "accuracy": 0.85,
                "accuracy_ar": 0.90,  # 精确检索
                "accuracy_ttl": 0.80, # 测试时学习
                "mean_latency_ms": 45.2,
                "p95_latency_ms": 120.5,
            }
        """
        if not results:
            return {"accuracy": 0.0, "mean_latency_ms": 0.0}

        # 基本计数
        total = len(results)
        correct = sum(1 for r in results if r.is_correct or r.score >= 0.5)

        # 计算平均值
        scores = [r.score for r in results]
        latencies = [r.latency_ms for r in results]

        metrics: Dict[str, float] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "total_instances": float(total),
            "correct_instances": float(correct),
        }

        # 按 task_type 分组以获得分解
        from collections import defaultdict

        by_type: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        for r in results:
            by_type[r.task_type].append(r)

        for task_type, type_results in by_type.items():
            if type_results:
                type_total = len(type_results)
                type_correct = sum(1 for r in type_results if r.is_correct or r.score >= 0.5)
                type_scores = [r.score for r in type_results]

                metrics[f"{task_type}_total"] = float(type_total)
                metrics[f"{task_type}_correct"] = float(type_correct)
                metrics[f"{task_type}_accuracy"] = (
                    type_correct / type_total if type_total > 0 else 0.0
                )
                metrics[f"{task_type}_avg_score"] = (
                    float(np.mean(type_scores)) if type_scores else 0.0
                )

        return metrics

    def prepare_memory(
        self,
        memory: MemoryProtocol,
        context: str,
        clear_first: bool = True,
    ) -> None:
        """
        为评估准备内存状态。

        默认实现清除内存并摄入上下文。
        子类可以覆盖以获得基准特定的准备。

        Args:
            memory: 内存系统实例
            context: 要摄入的上下文
            clear_first: 摄入前是否清除内存
        """
        if clear_first:
            memory.clear()

        # 将上下文分成轮次并摄入每一个
        turns = self._split_context(context)
        for turn in turns:
            if turn.strip():
                memory.ingest(turn)

        # 允许内存整合
        memory.evolve()

    def _split_context(self, context: str) -> List[str]:
        """
        将上下文分成单个轮次以供摄入。

        默认实现按双换行符分割。
        子类可以覆盖以获得基准特定的解析。

        Args:
            context: 完整上下文字符串

        Returns:
            单个轮次列表
        """
        # 简单按双换行符分割；子类可能会覆盖
        return [turn.strip() for turn in context.split("\n\n") if turn.strip()]


# -----------------------------------------------------------------------------
# 基准运行器
# -----------------------------------------------------------------------------


class AURORABenchmarkRunner:
    """
    执行基准测试的主运行器。

    协调基准适配器、管理执行和收集结果。

    Attributes:
        memory: AURORA 内存系统实例
        adapters: 将基准名称映射到适配器的字典

    Example:
        from aurora.soul.engine import AuroraSoul, SoulConfig
        from aurora.benchmarks.interface import AURORABenchmarkRunner
        from aurora.benchmarks.adapters.memoryagentbench import MemoryAgentBenchAdapter

        memory = AuroraSoul(cfg=SoulConfig())
        adapters = {"mab": MemoryAgentBenchAdapter()}

        runner = AURORABenchmarkRunner(memory, adapters)
        results = runner.run_benchmark("mab", "/path/to/dataset")

        print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
    """

    def __init__(
        self,
        memory: MemoryProtocol,
        adapters: Optional[Dict[str, BenchmarkAdapter]] = None,
    ):
        """
        初始化基准运行器。

        Args:
            memory: AURORA 内存系统实例
            adapters: 将基准名称映射到适配器实例的字典
        """
        self.memory = memory
        self.adapters: Dict[str, BenchmarkAdapter] = adapters or {}

    def register_adapter(self, name: str, adapter: BenchmarkAdapter) -> None:
        """
        注册新的基准适配器。

        Args:
            name: 注册适配器的名称
            adapter: 适配器实例
        """
        self.adapters[name] = adapter
        logger.info(f"Registered adapter '{name}' ({adapter.name})")

    def run_benchmark(
        self,
        benchmark_name: str,
        dataset_path: str,
        subset: Optional[Sequence[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        运行特定的基准测试。

        Args:
            benchmark_name: 已注册基准适配器的名称
            dataset_path: 基准数据集的路径
            subset: 要评估的实例 ID 的可选列表（用于调试）
            progress_callback: 用于进度报告的可选回调(current, total)

        Returns:
            包含以下内容的字典:
                - "benchmark": 基准名称
                - "dataset_path": 数据集路径
                - "num_instances": 评估的实例数
                - "results": BenchmarkResult 字典列表
                - "metrics": 聚合指标字典

        Raises:
            KeyError: 如果 benchmark_name 未注册
            FileNotFoundError: 如果 dataset_path 不存在
        """
        if benchmark_name not in self.adapters:
            raise KeyError(
                f"Unknown benchmark: {benchmark_name}. Available: {list(self.adapters.keys())}"
            )

        adapter = self.adapters[benchmark_name]
        logger.info(f"Running benchmark '{adapter.name}' from {dataset_path}")

        # 加载数据集
        instances = adapter.load_dataset(dataset_path)
        logger.info(f"Loaded {len(instances)} instances")

        # 如果指定，过滤到子集
        if subset:
            subset_set = set(subset)
            instances = [i for i in instances if i.id in subset_set]
            logger.info(f"Filtered to {len(instances)} instances in subset")

        # 评估每个实例
        results: List[BenchmarkResult] = []
        for idx, instance in enumerate(instances):
            try:
                result = adapter.evaluate(instance, self.memory)
                results.append(result)

                if progress_callback:
                    progress_callback(idx + 1, len(instances))

            except Exception as e:
                logger.error(f"Error evaluating instance {instance.id}: {e}")
                # 记录失败的评估
                results.append(
                    BenchmarkResult(
                        instance_id=instance.id,
                        capability=instance.capability,
                        predicted="",
                        expected=instance.expected_answer,
                        score=0.0,
                        latency_ms=0.0,
                        metadata={"error": str(e)},
                    )
                )

        # 聚合结果
        metrics = adapter.aggregate_results(results)

        return {
            "benchmark": adapter.name,
            "dataset_path": dataset_path,
            "num_instances": len(instances),
            "results": [r.to_state_dict() for r in results],
            "metrics": metrics,
        }

    def run_all(
        self,
        datasets: Dict[str, str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        运行所有已注册的基准测试。

        Args:
            datasets: 将基准名称映射到数据集路径的字典
            progress_callback: 可选回调(benchmark, current, total)

        Returns:
            将基准名称映射到其结果的字典

        Example:
            datasets = {
                "mab": "/path/to/memoryagentbench",
                "locomo": "/path/to/locomo",
            }
            all_results = runner.run_all(datasets)
        """
        all_results: Dict[str, Dict[str, Any]] = {}

        for benchmark_name, dataset_path in datasets.items():
            if benchmark_name not in self.adapters:
                logger.warning(f"Skipping unknown benchmark: {benchmark_name}")
                continue

            def _progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback(benchmark_name, current, total)

            try:
                result = self.run_benchmark(
                    benchmark_name,
                    dataset_path,
                    progress_callback=_progress,
                )
                all_results[benchmark_name] = result
            except Exception as e:
                logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
                all_results[benchmark_name] = {"error": str(e)}

        return all_results

    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        format: str = "markdown",
    ) -> str:
        """
        从基准结果生成摘要报告。

        Args:
            results: 来自 run_all() 或多个 run_benchmark() 调用的结果
            format: 输出格式（"markdown" 或 "text"）

        Returns:
            格式化的报告字符串
        """
        lines = []

        if format == "markdown":
            lines.append("# AURORA Benchmark Report\n")
            lines.append("## Summary\n")
            lines.append("| Benchmark | Instances | Accuracy | Mean Latency (ms) |")
            lines.append("|-----------|-----------|----------|-------------------|")

            for name, data in results.items():
                if "error" in data:
                    lines.append(f"| {name} | ERROR | - | - |")
                    continue

                metrics = data.get("metrics", {})
                accuracy = metrics.get("accuracy", 0.0)
                latency = metrics.get("mean_latency_ms", 0.0)
                num = data.get("num_instances", 0)
                lines.append(f"| {name} | {num} | {accuracy:.2%} | {latency:.1f} |")

            # 详细的能力分解
            lines.append("\n## Capability Breakdown\n")
            for name, data in results.items():
                if "error" in data:
                    continue

                metrics = data.get("metrics", {})
                lines.append(f"### {name}\n")

                for cap in BenchmarkCapability:
                    key = f"accuracy_{cap.value[:3]}"  # 例如 accuracy_acc
                    if key in metrics:
                        lines.append(f"- **{cap.value}**: {metrics[key]:.2%}")

                lines.append("")
        else:
            # 纯文本格式
            lines.append("AURORA Benchmark Report")
            lines.append("=" * 50)

            for name, data in results.items():
                lines.append(f"\n{name}:")
                if "error" in data:
                    lines.append(f"  ERROR: {data['error']}")
                    continue

                metrics = data.get("metrics", {})
                lines.append(f"  Instances: {data.get('num_instances', 0)}")
                lines.append(f"  Accuracy: {metrics.get('accuracy', 0.0):.2%}")
                lines.append(f"  Latency: {metrics.get('mean_latency_ms', 0.0):.1f}ms")

        return "\n".join(lines)


# =============================================================================
# 评估辅助函数
# =============================================================================


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """精确字符串匹配评估（不区分大小写）。"""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def contains_score(prediction: str, ground_truth: str) -> float:
    """检查预测是否包含真实值。"""
    return 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0


def fuzzy_match_score(prediction: str, ground_truth: str, threshold: float = 0.8) -> float:
    """使用编辑距离比率的模糊字符串匹配。"""
    from difflib import SequenceMatcher

    pred_lower = prediction.strip().lower()
    truth_lower = ground_truth.strip().lower()

    ratio = SequenceMatcher(None, pred_lower, truth_lower).ratio()
    return ratio if ratio >= threshold else 0.0


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """计算令牌级 F1 分数。"""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)
