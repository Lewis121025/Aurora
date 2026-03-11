"""
aurora/benchmarks/adapters/locomo.py 的测试
=============================================

测试 LOCOMO 基准适配器：
- 适配器初始化
- 会话解析
- 从会话准备内存
- QA 评估（单跳、多跳、时间）
- 总结评估
- 结果聚合
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from aurora.benchmarks.interface import (
    BenchmarkInstance,
    BenchmarkResult,
    EvaluationMetrics,
    EvaluationMethod,
)
from aurora.benchmarks.adapters.locomo import (
    LOCOMOAdapter,
    LOCOMOReasoningType,
    LOCOMOTaskType,
    LOCOMOTurn,
    create_locomo_adapter,
)
from aurora.utils.time_utils import now_ts


# =============================================================================
# Mock Data
# =============================================================================

MOCK_LOCOMO_SESSION = {
    "session_id": "test_session_001",
    "turns": [
        {"speaker": "user", "text": "Hello, I'm Bob."},
        {"speaker": "assistant", "text": "Nice to meet you, Bob!"},
    ],
}

MOCK_LOCOMO_SESSION_FULL = {
    "session_id": "session_002",
    "turns": [
        {"id": "turn_1", "speaker": "user", "text": "Hello, my name is Alice."},
        {"id": "turn_2", "speaker": "assistant", "text": "Nice to meet you, Alice!"},
        {"id": "turn_3", "speaker": "user", "text": "I live in San Francisco."},
        {"id": "turn_4", "speaker": "assistant", "text": "That's a beautiful city!"},
        {"id": "turn_5", "speaker": "user", "text": "I work as a software engineer."},
        {"id": "turn_6", "speaker": "assistant", "text": "That's a great profession!"},
    ],
    "personas": {"user": "A software engineer from San Francisco named Alice"},
    "events": [
        {"event": "introduction", "turn_id": "turn_1"},
        {"event": "location_mention", "turn_id": "turn_3"},
    ],
}

MOCK_LOCOMO_QUESTIONS = [
    {
        "id": "q_001",
        "session_id": "session_002",
        "question": "What is the user's name?",
        "answer": "Alice",
        "reasoning_type": "single_hop",
        "evidence_turns": ["turn_1"],
    },
    {
        "id": "q_002",
        "session_id": "session_002",
        "question": "Where does Alice work and live?",
        "answer": "Alice is a software engineer who lives in San Francisco",
        "reasoning_type": "multi_hop",
        "evidence_turns": ["turn_3", "turn_5"],
    },
]


# =============================================================================
# Mock Memory Class
# =============================================================================


class MockMemory:
    """用于测试的模拟 AURORA 内存。"""

    def __init__(self):
        self.plots: dict[str, Any] = {}
        self.stories: dict[str, Any] = {}
        self.themes: dict[str, Any] = {}
        self._ingested = []
        self._queries = []
        self._plot_counter = 0
        self.graph = MockGraph(self.plots)
        self.metric = None
        self.vindex = None

    def ingest(self, interaction_text: str, actors=None, event_id=None, **kwargs):
        self._ingested.append(interaction_text)

        # Create a mock plot
        plot_id = event_id or f"plot_{self._plot_counter}"
        self._plot_counter += 1

        mock_plot = MagicMock()
        mock_plot.id = plot_id
        mock_plot.text = interaction_text
        mock_plot.ts = now_ts()
        self.plots[plot_id] = mock_plot

    def query(self, text: str, k: int = 10, **kwargs):
        self._queries.append(text)

        # Return mock trace
        trace = MagicMock()

        # Return ranked results based on stored plots
        ranked = []
        for plot_id, plot in list(self.plots.items())[:k]:
            ranked.append((plot_id, 0.9, "plot"))

        trace.ranked = ranked
        return trace

    def evolve(self):
        pass

    def clear(self):
        self.plots = {}
        self._ingested = []
        self._queries = []


class MockGraph:
    """用于测试的模拟图。"""

    def __init__(self, plots):
        self._plots = plots

    def payload(self, node_id):
        return self._plots.get(node_id)


# =============================================================================
# Test Adapter Initialization
# =============================================================================


class TestAdapterInitialization:
    """LOCOMOAdapter 初始化的测试。"""

    def test_basic_initialization(self):
        """测试基本适配器初始化。"""
        adapter = LOCOMOAdapter()

        assert adapter.name == "LOCOMO"
        assert adapter.llm is None
        assert adapter.evaluation_method == EvaluationMethod.LLM_JUDGE

    def test_initialization_with_llm(self):
        """测试使用 LLM 提供者的初始化。"""
        mock_llm = MagicMock()
        adapter = LOCOMOAdapter(llm_provider=mock_llm)

        assert adapter.llm is mock_llm

    def test_initialization_with_seed(self):
        """测试使用 seed 进行可重复初始化。"""
        adapter1 = LOCOMOAdapter(seed=42)
        adapter2 = LOCOMOAdapter(seed=42)

        # Same seed should produce same random values
        val1 = adapter1.rng.random()
        val2 = adapter2.rng.random()
        assert val1 == val2

    def test_initialization_with_evaluation_method(self):
        """测试使用自定义评估方法的初始化。"""
        adapter = LOCOMOAdapter(evaluation_method=EvaluationMethod.FUZZY)

        assert adapter.evaluation_method == EvaluationMethod.FUZZY

    def test_initialization_with_thresholds(self):
        """测试使用自定义阈值的初始化。"""
        adapter = LOCOMOAdapter(
            f1_threshold=0.6,
            fuzzy_threshold=0.8,
        )

        assert adapter.f1_threshold == 0.6
        assert adapter.fuzzy_threshold == 0.8

    def test_factory_function(self):
        """测试 create_locomo_adapter 工厂函数。"""
        adapter = create_locomo_adapter(seed=42)

        assert isinstance(adapter, LOCOMOAdapter)
        assert adapter.evaluation_method == EvaluationMethod.FUZZY  # No LLM provided

    def test_factory_with_llm(self):
        """测试使用 LLM 的工厂函数。"""
        mock_llm = MagicMock()
        adapter = create_locomo_adapter(llm_provider=mock_llm, use_llm_judge=True)

        assert adapter.evaluation_method == EvaluationMethod.LLM_JUDGE


# =============================================================================
# Test Session Parsing
# =============================================================================


class TestParseSession:
    """会话解析功能的测试。"""

    def test_parse_basic_session(self):
        """测试解析基本会话数据。"""
        adapter = LOCOMOAdapter()

        session = adapter._parse_session(MOCK_LOCOMO_SESSION)

        assert session.session_id == "test_session_001"
        assert len(session.turns) == 2
        assert session.turns[0].speaker == "user"
        assert session.turns[0].text == "Hello, I'm Bob."

    def test_parse_session_with_metadata(self):
        """测试解析包含角色和事件的会话。"""
        adapter = LOCOMOAdapter()

        session = adapter._parse_session(MOCK_LOCOMO_SESSION_FULL)

        assert session.session_id == "session_002"
        assert len(session.turns) == 6
        assert "user" in session.personas
        assert len(session.events) == 2

    def test_parse_session_string_turns(self):
        """测试解析字符串形式的会话轮次。"""
        adapter = LOCOMOAdapter()

        session_data = {
            "session_id": "test_simple",
            "turns": ["Hello", "Hi there"],
        }

        session = adapter._parse_session(session_data)

        assert len(session.turns) == 2
        assert session.turns[0].text == "Hello"

    def test_locomo_turn_to_dict(self):
        """测试 LOCOMOTurn.to_dict() 方法。"""
        turn = LOCOMOTurn(
            turn_id="turn_001",
            speaker="user",
            text="Hello!",
            timestamp=now_ts(),
            metadata={"extra": "data"},
        )

        d = turn.to_dict()

        assert d["id"] == "turn_001"
        assert d["speaker"] == "user"
        assert d["text"] == "Hello!"
        assert d["extra"] == "data"


# =============================================================================
# Test Memory Preparation from Sessions
# =============================================================================


class TestPrepareMemoryFromSessions:
    """从 LOCOMO 会话准备内存的测试。"""

    def test_prepare_memory_basic(self):
        """测试从对话历史进行基本内存准备。"""
        adapter = LOCOMOAdapter()
        memory = MockMemory()

        conversation_history = [
            {"speaker": "user", "text": "Hello, I'm Bob."},
            {"speaker": "assistant", "text": "Nice to meet you, Bob!"},
        ]

        adapter.prepare_memory(conversation_history, memory)

        # Memory should have ingested turns
        assert len(memory._ingested) == 2

    def test_prepare_memory_formats_user_turns(self):
        """测试用户轮次格式化正确。"""
        adapter = LOCOMOAdapter()
        memory = MockMemory()

        conversation_history = [
            {"speaker": "user", "text": "Hello"},
        ]

        adapter.prepare_memory(conversation_history, memory)

        # Should format as "用户：Hello"
        assert "用户" in memory._ingested[0] or "Hello" in memory._ingested[0]

    def test_prepare_memory_multiple_sessions(self):
        """测试从多个会话准备内存。"""
        adapter = LOCOMOAdapter()
        memory = MockMemory()

        sessions = [
            {
                "session_id": "s1",
                "turns": [{"speaker": "user", "text": "Session 1"}],
            },
            {
                "session_id": "s2",
                "turns": [{"speaker": "user", "text": "Session 2"}],
            },
        ]

        adapter._prepare_memory_from_sessions(sessions, memory)

        assert len(memory._ingested) == 2


# =============================================================================
# Test QA Single-Hop Evaluation
# =============================================================================


class TestEvaluateQASingleHop:
    """单跳 QA 评估的测试。"""

    def test_evaluate_single_hop_basic(self):
        """测试基本单跳 QA 评估。"""
        adapter = LOCOMOAdapter(seed=42, evaluation_method=EvaluationMethod.FUZZY)
        memory = MockMemory()

        # Pre-populate memory
        memory.ingest("用户：Hello, my name is Alice.")

        instance = BenchmarkInstance(
            id="q_001",
            task_type="qa",
            conversation_history=[
                {"speaker": "user", "text": "Hello, my name is Alice."},
            ],
            query="What is the user's name?",
            ground_truth="Alice",
            reasoning_type="single_hop",
        )

        result = adapter.evaluate(instance, memory)

        assert result.instance_id == "q_001"
        assert "qa_single_hop" in result.task_type
        assert isinstance(result.score, float)
        assert result.latency_ms >= 0

    def test_evaluate_single_hop_records_trace(self):
        """测试评估记录推理轨迹。"""
        adapter = LOCOMOAdapter(seed=42, evaluation_method=EvaluationMethod.FUZZY)
        memory = MockMemory()

        instance = BenchmarkInstance(
            id="q_002",
            task_type="qa",
            conversation_history=[],
            query="Test question",
            ground_truth="Test answer",
            reasoning_type="single_hop",
        )

        result = adapter.evaluate(instance, memory)

        assert len(result.reasoning_trace) > 0


# =============================================================================
# Test QA Multi-Hop Evaluation
# =============================================================================


class TestEvaluateQAMultiHop:
    """多跳 QA 评估的测试。"""

    def test_evaluate_multi_hop_basic(self):
        """测试基本多跳 QA 评估。"""
        adapter = LOCOMOAdapter(seed=42, evaluation_method=EvaluationMethod.FUZZY)
        memory = MockMemory()

        # Pre-populate memory with related facts
        memory.ingest("用户：I live in San Francisco.")
        memory.ingest("用户：I work as an engineer.")

        instance = BenchmarkInstance(
            id="q_003",
            task_type="qa",
            conversation_history=[
                {"speaker": "user", "text": "I live in San Francisco."},
                {"speaker": "user", "text": "I work as an engineer."},
            ],
            query="Where does the engineer live?",
            ground_truth="San Francisco",
            reasoning_type="multi_hop",
        )

        result = adapter.evaluate(instance, memory)

        assert result.instance_id == "q_003"
        assert "multi_hop" in result.task_type


# =============================================================================
# Test QA Temporal Evaluation
# =============================================================================


class TestEvaluateQATemporal:
    """时间 QA 评估的测试。"""

    def test_evaluate_temporal_basic(self):
        """测试基本时间 QA 评估。"""
        adapter = LOCOMOAdapter(seed=42, evaluation_method=EvaluationMethod.FUZZY)
        memory = MockMemory()

        instance = BenchmarkInstance(
            id="q_004",
            task_type="qa",
            conversation_history=[
                {"speaker": "user", "text": "I went to Paris in June."},
                {"speaker": "user", "text": "Then I visited Rome in July."},
            ],
            query="Which city was visited first?",
            ground_truth="Paris",
            reasoning_type="temporal",
        )

        result = adapter.evaluate(instance, memory)

        assert result.instance_id == "q_004"
        assert "temporal" in result.task_type


# =============================================================================
# Test Summarization Evaluation
# =============================================================================


class TestEvaluateSummarization:
    """事件总结评估的测试。"""

    def test_evaluate_summarization_basic(self):
        """测试基本总结评估。"""
        adapter = LOCOMOAdapter(
            seed=42,
            evaluation_method=EvaluationMethod.FUZZY,
            use_story_summary=False,
        )
        memory = MockMemory()

        # Pre-populate memory
        memory.ingest("用户：I went to Paris.")
        memory.ingest("用户：Then I visited Rome.")

        instance = BenchmarkInstance(
            id="sum_001",
            task_type="summarization",
            conversation_history=[
                {"speaker": "user", "text": "I went to Paris."},
                {"speaker": "user", "text": "Then I visited Rome."},
            ],
            query="Summarize the trip",
            ground_truth="The user visited Paris and Rome.",
        )

        result = adapter.evaluate(instance, memory)

        assert result.instance_id == "sum_001"
        assert result.task_type == "summarization"

    def test_evaluate_summarization_returns_metrics(self):
        """测试总结返回评估指标。"""
        adapter = LOCOMOAdapter(
            seed=42,
            evaluation_method=EvaluationMethod.FUZZY,
            use_story_summary=False,
        )
        memory = MockMemory()

        instance = BenchmarkInstance(
            id="sum_002",
            task_type="summarization",
            conversation_history=[],
            query="Summarize events",
            ground_truth="Event summary",
        )

        result = adapter.evaluate(instance, memory)

        # Should have score and metadata
        assert isinstance(result.score, float)
        assert isinstance(result.metadata, dict)


# =============================================================================
# Test Results Aggregation
# =============================================================================


class TestAggregateResults:
    """结果聚合的测试。"""

    def test_aggregate_empty_results(self):
        """测试空结果的聚合。"""
        adapter = LOCOMOAdapter()

        metrics = adapter.aggregate_results([])

        assert metrics["accuracy"] == 0.0
        assert metrics["mean_latency_ms"] == 0.0

    def test_aggregate_single_result(self):
        """测试单个结果的聚合。"""
        adapter = LOCOMOAdapter()

        results = [
            BenchmarkResult(
                instance_id="test_001",
                task_type="qa_single_hop",
                score=1.0,
                is_correct=True,
                latency_ms=50.0,
                metadata={"reasoning_type": "single_hop"},
            )
        ]

        metrics = adapter.aggregate_results(results)

        assert metrics["total_instances"] == 1
        assert metrics["correct_instances"] == 1
        assert metrics["accuracy"] == 1.0

    def test_aggregate_multiple_results(self):
        """测试多个结果的聚合。"""
        adapter = LOCOMOAdapter()

        results = [
            BenchmarkResult(
                instance_id="t1",
                task_type="qa_single_hop",
                score=1.0,
                is_correct=True,
                latency_ms=40.0,
                metadata={"reasoning_type": "single_hop"},
            ),
            BenchmarkResult(
                instance_id="t2",
                task_type="qa_multi_hop",
                score=0.5,
                is_correct=True,
                latency_ms=50.0,
                metadata={"reasoning_type": "multi_hop"},
            ),
            BenchmarkResult(
                instance_id="t3",
                task_type="qa_temporal",
                score=0.3,
                is_correct=False,
                latency_ms=60.0,
                metadata={"reasoning_type": "temporal"},
            ),
        ]

        metrics = adapter.aggregate_results(results)

        assert metrics["total_instances"] == 3
        assert metrics["correct_instances"] == 2
        assert 0.6 < metrics["accuracy"] < 0.7

    def test_aggregate_by_reasoning_type(self):
        """测试按推理类型的聚合。"""
        adapter = LOCOMOAdapter()

        results = [
            BenchmarkResult(
                instance_id="sh_1",
                task_type="qa_single_hop",
                score=1.0,
                is_correct=True,
                metadata={"reasoning_type": "single_hop"},
            ),
            BenchmarkResult(
                instance_id="sh_2",
                task_type="qa_single_hop",
                score=0.8,
                is_correct=True,
                metadata={"reasoning_type": "single_hop"},
            ),
            BenchmarkResult(
                instance_id="mh_1",
                task_type="qa_multi_hop",
                score=0.3,
                is_correct=False,
                metadata={"reasoning_type": "multi_hop"},
            ),
        ]

        metrics = adapter.aggregate_results(results)

        # Should have single_hop and multi_hop specific metrics
        assert "qa_single_hop_accuracy" in metrics
        assert "qa_multi_hop_accuracy" in metrics
        assert metrics["qa_single_hop_accuracy"] == 1.0  # 2/2 correct
        assert metrics["qa_multi_hop_accuracy"] == 0.0  # 0/1 correct

    def test_aggregate_includes_latency_percentiles(self):
        """测试聚合包含延迟百分位数。"""
        adapter = LOCOMOAdapter()

        results = [
            BenchmarkResult(
                instance_id=f"t{i}",
                task_type="qa_single_hop",
                score=0.5,
                is_correct=True,
                latency_ms=float(i * 10),
            )
            for i in range(1, 11)
        ]

        metrics = adapter.aggregate_results(results)

        assert "p50_latency_ms" in metrics
        assert "p95_latency_ms" in metrics
        assert "p99_latency_ms" in metrics

    def test_get_evaluation_metrics_object(self):
        """测试获取 EvaluationMetrics 对象。"""
        adapter = LOCOMOAdapter()

        results = [
            BenchmarkResult(
                instance_id="t1",
                task_type="qa_single_hop",
                score=0.8,
                is_correct=True,
                latency_ms=50.0,
            )
        ]

        # First aggregate
        adapter.aggregate_results(results)

        # Then get metrics object
        metrics_obj = adapter.get_evaluation_metrics(results)

        assert isinstance(metrics_obj, EvaluationMetrics)
        assert metrics_obj.total_instances == 1
        assert metrics_obj.accuracy == 1.0


# =============================================================================
# Test Reasoning Type Enum
# =============================================================================


class TestLOCOMOReasoningType:
    """LOCOMOReasoningType 枚举的测试。"""

    def test_reasoning_type_values(self):
        """测试推理类型值。"""
        assert LOCOMOReasoningType.SINGLE_HOP.value == "single_hop"
        assert LOCOMOReasoningType.MULTI_HOP.value == "multi_hop"
        assert LOCOMOReasoningType.TEMPORAL.value == "temporal"
        assert LOCOMOReasoningType.COMMONSENSE.value == "commonsense"
        assert LOCOMOReasoningType.WORLD_KNOWLEDGE.value == "world_knowledge"

    def test_from_string_variations(self):
        """Test from_string handles variations."""
        assert LOCOMOReasoningType.from_string("single_hop") == LOCOMOReasoningType.SINGLE_HOP
        assert LOCOMOReasoningType.from_string("single-hop") == LOCOMOReasoningType.SINGLE_HOP
        assert LOCOMOReasoningType.from_string("singlehop") == LOCOMOReasoningType.SINGLE_HOP

        assert LOCOMOReasoningType.from_string("multi_hop") == LOCOMOReasoningType.MULTI_HOP
        assert LOCOMOReasoningType.from_string("multi-hop") == LOCOMOReasoningType.MULTI_HOP

    def test_from_string_unknown_defaults(self):
        """测试 from_string 对未知值返回默认值。"""
        assert LOCOMOReasoningType.from_string("unknown") == LOCOMOReasoningType.SINGLE_HOP


# =============================================================================
# Test Task Type Enum
# =============================================================================


class TestLOCOMOTaskType:
    """LOCOMOTaskType 枚举的测试。"""

    def test_task_type_values(self):
        """测试任务类型值。"""
        assert LOCOMOTaskType.QUESTION_ANSWERING.value == "qa"
        assert LOCOMOTaskType.EVENT_SUMMARIZATION.value == "summarization"
        assert LOCOMOTaskType.DIALOGUE_GENERATION.value == "dialogue"

    def test_from_string_variations(self):
        """测试 from_string 处理变体。"""
        assert LOCOMOTaskType.from_string("qa") == LOCOMOTaskType.QUESTION_ANSWERING
        assert LOCOMOTaskType.from_string("question_answering") == LOCOMOTaskType.QUESTION_ANSWERING

        assert LOCOMOTaskType.from_string("summarization") == LOCOMOTaskType.EVENT_SUMMARIZATION
        assert LOCOMOTaskType.from_string("summary") == LOCOMOTaskType.EVENT_SUMMARIZATION


# =============================================================================
# Test Dataset Loading
# =============================================================================


class TestDatasetLoading:
    """数据集加载功能的测试。"""

    def test_load_from_json_file(self, tmp_path):
        """测试从 JSON 文件加载。"""
        adapter = LOCOMOAdapter()

        # Create test data
        test_data = {
            "sessions": [MOCK_LOCOMO_SESSION_FULL],
            "questions": MOCK_LOCOMO_QUESTIONS,
        }

        data_file = tmp_path / "locomo_test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        instances = adapter.load_dataset(str(data_file))

        assert len(instances) == 2
        assert instances[0].id == "q_001"

    def test_load_from_directory(self, tmp_path):
        """测试从包含分割文件的目录加载。"""
        adapter = LOCOMOAdapter()

        # Create sessions file
        sessions_data = [MOCK_LOCOMO_SESSION_FULL]
        with open(tmp_path / "sessions.json", "w") as f:
            json.dump(sessions_data, f)

        # Create questions file
        with open(tmp_path / "questions.json", "w") as f:
            json.dump(MOCK_LOCOMO_QUESTIONS, f)

        instances = adapter.load_dataset(str(tmp_path))

        assert len(instances) == 2

    def test_load_with_subset_filter(self, tmp_path):
        """测试使用推理类型过滤加载。"""
        adapter = LOCOMOAdapter()

        test_data = {
            "sessions": [MOCK_LOCOMO_SESSION_FULL],
            "questions": MOCK_LOCOMO_QUESTIONS,
        }

        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        # Filter to single_hop only
        instances = adapter.load_dataset(str(data_file), subset="single_hop")

        assert len(instances) == 1
        assert instances[0].reasoning_type == "single_hop"

    def test_load_with_limit(self, tmp_path):
        """测试使用实例限制加载。"""
        adapter = LOCOMOAdapter()

        test_data = {
            "sessions": [MOCK_LOCOMO_SESSION_FULL],
            "questions": MOCK_LOCOMO_QUESTIONS,
        }

        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        instances = adapter.load_dataset(str(data_file), limit=1)

        assert len(instances) == 1

    def test_load_includes_summarization_tasks(self, tmp_path):
        """测试加载包含总结任务。"""
        adapter = LOCOMOAdapter()

        test_data = {
            "sessions": [MOCK_LOCOMO_SESSION_FULL],
            "questions": [],
            "summaries": [
                {
                    "id": "sum_001",
                    "session_id": "session_002",
                    "instruction": "Summarize the conversation",
                    "summary": "User introduced themselves as Alice from San Francisco.",
                }
            ],
        }

        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        instances = adapter.load_dataset(str(data_file))

        assert len(instances) == 1
        assert instances[0].task_type == "summarization"


# =============================================================================
# Test Evaluation Statistics
# =============================================================================


class TestEvaluationStats:
    """评估统计跟踪的测试。"""

    def test_get_evaluation_stats(self):
        """测试获取评估统计。"""
        adapter = LOCOMOAdapter(evaluation_method=EvaluationMethod.FUZZY)

        stats = adapter.get_evaluation_stats()

        assert "total_evals" in stats
        assert "llm_evals" in stats
        assert "fallback_evals" in stats

    def test_stats_increment_after_evaluation(self):
        """测试评估后统计增加。"""
        adapter = LOCOMOAdapter(
            seed=42,
            evaluation_method=EvaluationMethod.FUZZY,
        )
        memory = MockMemory()

        instance = BenchmarkInstance(
            id="test_001",
            task_type="qa",
            conversation_history=[],
            query="Test",
            ground_truth="Answer",
            reasoning_type="single_hop",
        )

        initial_stats = adapter.get_evaluation_stats()
        initial_total = initial_stats["total_evals"]

        adapter.evaluate(instance, memory)

        final_stats = adapter.get_evaluation_stats()
        assert final_stats["total_evals"] > initial_total


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """评估中错误处理的测试。"""

    def test_evaluate_handles_memory_error(self):
        """测试评估优雅处理内存错误。"""
        adapter = LOCOMOAdapter(seed=42, evaluation_method=EvaluationMethod.FUZZY)

        # Create a memory that raises errors
        class FailingMemory:
            def query(self, *args, **kwargs):
                raise RuntimeError("Memory error")

        instance = BenchmarkInstance(
            id="test_001",
            task_type="qa",
            conversation_history=[],
            query="Test",
            ground_truth="Answer",
            reasoning_type="single_hop",
        )

        result = adapter.evaluate(instance, FailingMemory())

        # Should return error result, not raise
        assert result.instance_id == "test_001"
        assert result.score == 0.0
        assert result.error_message is not None

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件抛出错误。"""
        adapter = LOCOMOAdapter()

        with pytest.raises(FileNotFoundError):
            adapter.load_dataset("/nonexistent/path/to/data.json")
