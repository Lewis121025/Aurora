"""
aurora/benchmark/interface.py 的测试
=======================================

测试核心基准接口组件:
- BenchmarkCapability 枚举
- BenchmarkInstance 数据类
- BenchmarkResult 数据类
- AURORABenchmarkRunner
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from aurora.benchmark.interface import (
    BenchmarkCapability,
    BenchmarkInstance,
    BenchmarkResult,
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationMethod,
    BenchmarkAdapter,
    AURORABenchmarkRunner,
    MemoryProtocol,
    exact_match_score,
    contains_score,
    fuzzy_match_score,
    compute_f1_score,
)
from aurora.utils.time_utils import now_ts


# =============================================================================
# 模拟数据
# =============================================================================

MOCK_MAB_INSTANCE = {
    "id": "test_ar_001",
    "capability": "accurate_retrieval",
    "context": "User: What is my name?\nAssistant: Your name is Alice.",
    "query": "What is my name?",
    "expected_answer": "Alice"
}

MOCK_LOCOMO_SESSION = {
    "session_id": "test_session_001",
    "turns": [
        {"speaker": "user", "text": "Hello, I'm Bob."},
        {"speaker": "assistant", "text": "Nice to meet you, Bob!"}
    ]
}


# =============================================================================
# 测试 BenchmarkCapability 枚举
# =============================================================================

class TestBenchmarkCapabilityEnum:
    """BenchmarkCapability 枚举的测试。"""

    def test_capability_values(self):
        """测试所有能力值是否正确。"""
        assert BenchmarkCapability.ACCURATE_RETRIEVAL.value == "accurate_retrieval"
        assert BenchmarkCapability.TEST_TIME_LEARNING.value == "test_time_learning"
        assert BenchmarkCapability.LONG_RANGE_UNDERSTANDING.value == "long_range_understanding"
        assert BenchmarkCapability.CONFLICT_RESOLUTION.value == "conflict_resolution"
    
    def test_capability_str(self):
        """测试能力字符串表示。"""
        assert str(BenchmarkCapability.ACCURATE_RETRIEVAL) == "accurate_retrieval"
        assert str(BenchmarkCapability.CONFLICT_RESOLUTION) == "conflict_resolution"

    def test_capability_from_string(self):
        """测试从字符串创建能力。"""
        cap = BenchmarkCapability.from_string("accurate_retrieval")
        assert cap == BenchmarkCapability.ACCURATE_RETRIEVAL

        cap = BenchmarkCapability.from_string("test_time_learning")
        assert cap == BenchmarkCapability.TEST_TIME_LEARNING

    def test_capability_from_string_invalid(self):
        """测试 from_string 使用无效值时抛出 ValueError。"""
        with pytest.raises(ValueError, match="Unknown capability"):
            BenchmarkCapability.from_string("invalid_capability")

    def test_all_capabilities_iterable(self):
        """测试所有能力是否可迭代。"""
        capabilities = list(BenchmarkCapability)
        assert len(capabilities) == 4
        assert BenchmarkCapability.ACCURATE_RETRIEVAL in capabilities


# =============================================================================
# 测试 BenchmarkInstance 创建
# =============================================================================

class TestBenchmarkInstanceCreation:
    """BenchmarkInstance 数据类的测试。"""

    def test_basic_instance_creation(self):
        """测试创建基本基准实例。"""
        instance = BenchmarkInstance(
            id="test_001",
            query="What is my name?",
            expected_answer="Alice",
        )
        
        assert instance.id == "test_001"
        assert instance.query == "What is my name?"
        assert instance.expected_answer == "Alice"
        assert instance.task_type == "qa"  # default
        assert isinstance(instance.created_ts, float)
    
    def test_instance_with_capability(self):
        """测试创建具有能力的实例。"""
        instance = BenchmarkInstance(
            id="test_002",
            query="Summarize the conversation",
            capability=BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
            context="User: Hello\nAssistant: Hi there!",
            expected_answer="A greeting exchange",
        )

        assert instance.capability == BenchmarkCapability.LONG_RANGE_UNDERSTANDING
        assert instance.context == "User: Hello\nAssistant: Hi there!"

    def test_instance_with_conversation_history(self):
        """测试创建具有对话历史的实例。"""
        turns = [
            {"speaker": "user", "text": "Hello"},
            {"speaker": "assistant", "text": "Hi there!"},
        ]

        instance = BenchmarkInstance(
            id="test_003",
            query="What was the greeting?",
            conversation_history=turns,
            ground_truth="Hi there!",
        )

        assert len(instance.conversation_history) == 2
        assert instance.ground_truth == "Hi there!"
        # ground_truth 应该同步到 expected_answer
        assert instance.expected_answer == "Hi there!"

    def test_instance_ground_truth_sync(self):
        """测试 ground_truth 和 expected_answer 是否同步。"""
        # 测试 expected_answer -> ground_truth
        instance1 = BenchmarkInstance(
            id="test_004",
            query="Test",
            expected_answer="Answer",
        )
        assert instance1.ground_truth == "Answer"

        # 测试 ground_truth -> expected_answer
        instance2 = BenchmarkInstance(
            id="test_005",
            query="Test",
            ground_truth="Answer2",
        )
        assert instance2.expected_answer == "Answer2"

    def test_instance_to_state_dict(self):
        """测试序列化到状态字典。"""
        instance = BenchmarkInstance(
            id="test_006",
            query="What is X?",
            capability=BenchmarkCapability.ACCURATE_RETRIEVAL,
            expected_answer="X is Y",
            task_type="qa",
            reasoning_type="single_hop",
        )
        
        state = instance.to_state_dict()
        
        assert state["id"] == "test_006"
        assert state["query"] == "What is X?"
        assert state["capability"] == "accurate_retrieval"
        assert state["expected_answer"] == "X is Y"
        assert state["task_type"] == "qa"
    
    def test_instance_from_state_dict(self):
        """测试从状态字典重建。"""
        state = {
            "id": "test_007",
            "query": "Test query",
            "capability": "conflict_resolution",
            "expected_answer": "Test answer",
            "context": "Test context",
            "conversation_history": [{"speaker": "user", "text": "Hello"}],
            "task_type": "qa",
        }

        instance = BenchmarkInstance.from_state_dict(state)

        assert instance.id == "test_007"
        assert instance.capability == BenchmarkCapability.CONFLICT_RESOLUTION
        assert instance.expected_answer == "Test answer"
        assert len(instance.conversation_history) == 1


# =============================================================================
# 测试 BenchmarkResult 创建
# =============================================================================

class TestBenchmarkResultCreation:
    """BenchmarkResult 数据类的测试。"""

    def test_basic_result_creation(self):
        """测试创建基本基准结果。"""
        result = BenchmarkResult(
            instance_id="test_001",
            score=0.85,
            predicted="Alice",
            expected="Alice",
        )

        assert result.instance_id == "test_001"
        assert result.score == 0.85
        assert result.predicted == "Alice"
        assert result.expected == "Alice"
        assert result.is_correct is True  # score >= 0.5

    def test_result_with_latency(self):
        """测试具有延迟指标的结果。"""
        result = BenchmarkResult(
            instance_id="test_002",
            score=1.0,
            latency_ms=45.5,
            tokens_used=150,
        )

        assert result.latency_ms == 45.5
        assert result.tokens_used == 150

    def test_result_with_capability(self):
        """测试具有能力字段的结果。"""
        result = BenchmarkResult(
            instance_id="test_003",
            score=0.9,
            capability=BenchmarkCapability.TEST_TIME_LEARNING,
            task_type="test_time_learning",
        )

        assert result.capability == BenchmarkCapability.TEST_TIME_LEARNING
        assert result.task_type == "test_time_learning"

    def test_result_prediction_sync(self):
        """测试 prediction 和 predicted 是否同步。"""
        result1 = BenchmarkResult(
            instance_id="test_004",
            score=1.0,
            predicted="Answer",
        )
        assert result1.prediction == "Answer"

        result2 = BenchmarkResult(
            instance_id="test_005",
            score=1.0,
            prediction="Answer2",
        )
        assert result2.predicted == "Answer2"

    def test_result_ground_truth_sync(self):
        """测试 ground_truth 和 expected 是否同步。"""
        result = BenchmarkResult(
            instance_id="test_006",
            score=1.0,
            ground_truth="Truth",
        )
        assert result.expected == "Truth"

    def test_result_check_correct_default_threshold(self):
        """测试使用默认阈值的 check_correct。"""
        result_correct = BenchmarkResult(
            instance_id="test_007",
            score=0.6,
        )
        assert result_correct.check_correct() is True

        result_incorrect = BenchmarkResult(
            instance_id="test_008",
            score=0.4,
        )
        assert result_incorrect.check_correct() is False

    def test_result_check_correct_custom_threshold(self):
        """测试使用自定义阈值的 check_correct。"""
        result = BenchmarkResult(
            instance_id="test_009",
            score=0.7,
        )
        assert result.check_correct(threshold=0.8) is False
        assert result.check_correct(threshold=0.5) is True

    def test_result_to_state_dict(self):
        """测试序列化到状态字典。"""
        result = BenchmarkResult(
            instance_id="test_010",
            score=0.95,
            capability=BenchmarkCapability.ACCURATE_RETRIEVAL,
            predicted="Answer",
            expected="Answer",
            latency_ms=30.0,
            tokens_used=100,
            retrieval_count=5,
            reasoning_trace=["step1", "step2"],
        )

        state = result.to_state_dict()

        assert state["instance_id"] == "test_010"
        assert state["score"] == 0.95
        assert state["capability"] == "accurate_retrieval"
        assert state["latency_ms"] == 30.0
        assert state["tokens_used"] == 100
        assert len(state["reasoning_trace"]) == 2

    def test_result_from_state_dict(self):
        """测试从状态字典重建。"""
        state = {
            "instance_id": "test_011",
            "score": 0.8,
            "capability": "conflict_resolution",
            "predicted": "New value",
            "expected": "New value",
            "latency_ms": 50.0,
            "is_correct": True,
        }

        result = BenchmarkResult.from_state_dict(state)

        assert result.instance_id == "test_011"
        assert result.score == 0.8
        assert result.capability == BenchmarkCapability.CONFLICT_RESOLUTION


# =============================================================================
# 测试 AURORABenchmarkRunner
# =============================================================================

class MockMemory:
    """用于测试的模拟内存。"""

    def __init__(self):
        self.ingested = []
        self.queries = []

    def ingest(self, text: str, **kwargs):
        self.ingested.append(text)

    def query(self, text: str, **kwargs):
        self.queries.append(text)
        return MagicMock(ranked=[])

    def evolve(self):
        pass

    def clear(self):
        self.ingested = []
        self.queries = []


class MockAdapter(BenchmarkAdapter):
    """用于测试的模拟适配器。"""

    def __init__(self, instances=None, results=None):
        super().__init__()
        self._instances = instances or []
        self._results = results or []
        self._result_idx = 0

    @property
    def name(self) -> str:
        return "MockBenchmark"

    def load_dataset(self, path: str) -> list:
        return self._instances

    def evaluate(self, instance, memory) -> BenchmarkResult:
        if self._results:
            result = self._results[self._result_idx % len(self._results)]
            self._result_idx += 1
            return result
        return BenchmarkResult(
            instance_id=instance.id,
            score=0.5,
            predicted="mock",
            expected=instance.expected_answer,
        )

    def aggregate_results(self, results: list) -> dict:
        if not results:
            return {"accuracy": 0.0}
        correct = sum(1 for r in results if r.score >= 0.5)
        return {
            "accuracy": correct / len(results),
            "mean_latency_ms": sum(r.latency_ms for r in results) / len(results),
        }


class TestAURORABenchmarkRunner:
    """AURORABenchmarkRunner 的测试。"""

    def test_runner_initialization(self):
        """测试运行器初始化。"""
        memory = MockMemory()
        runner = AURORABenchmarkRunner(memory=memory)

        assert runner.memory is memory
        assert len(runner.adapters) == 0

    def test_runner_with_adapters(self):
        """测试使用适配器初始化运行器。"""
        memory = MockMemory()
        adapter = MockAdapter()

        runner = AURORABenchmarkRunner(
            memory=memory,
            adapters={"mock": adapter},
        )

        assert "mock" in runner.adapters
        assert runner.adapters["mock"].name == "MockBenchmark"

    def test_register_adapter(self):
        """测试注册新适配器。"""
        memory = MockMemory()
        runner = AURORABenchmarkRunner(memory=memory)
        adapter = MockAdapter()

        runner.register_adapter("test", adapter)

        assert "test" in runner.adapters

    def test_run_benchmark_unknown_adapter(self):
        """测试使用未知适配器运行基准测试会抛出错误。"""
        memory = MockMemory()
        runner = AURORABenchmarkRunner(memory=memory)

        with pytest.raises(KeyError, match="Unknown benchmark"):
            runner.run_benchmark("nonexistent", "/path/to/data")

    def test_run_benchmark_basic(self):
        """测试基本基准测试运行。"""
        memory = MockMemory()

        instances = [
            BenchmarkInstance(id="inst_1", query="Q1", expected_answer="A1"),
            BenchmarkInstance(id="inst_2", query="Q2", expected_answer="A2"),
        ]

        results = [
            BenchmarkResult(instance_id="inst_1", score=1.0, latency_ms=10.0),
            BenchmarkResult(instance_id="inst_2", score=0.5, latency_ms=20.0),
        ]

        adapter = MockAdapter(instances=instances, results=results)
        runner = AURORABenchmarkRunner(memory=memory, adapters={"test": adapter})

        output = runner.run_benchmark("test", "/fake/path")

        assert output["benchmark"] == "MockBenchmark"
        assert output["num_instances"] == 2
        assert len(output["results"]) == 2
        assert output["metrics"]["accuracy"] == 1.0  # 两个都 >= 0.5

    def test_run_benchmark_with_subset(self):
        """测试使用子集过滤器运行基准测试。"""
        memory = MockMemory()

        instances = [
            BenchmarkInstance(id="inst_1", query="Q1", expected_answer="A1"),
            BenchmarkInstance(id="inst_2", query="Q2", expected_answer="A2"),
            BenchmarkInstance(id="inst_3", query="Q3", expected_answer="A3"),
        ]

        adapter = MockAdapter(instances=instances)
        runner = AURORABenchmarkRunner(memory=memory, adapters={"test": adapter})

        output = runner.run_benchmark("test", "/fake/path", subset=["inst_1", "inst_3"])

        assert output["num_instances"] == 2

    def test_run_benchmark_with_progress_callback(self):
        """测试使用进度回调的基准测试运行。"""
        memory = MockMemory()

        instances = [
            BenchmarkInstance(id=f"inst_{i}", query=f"Q{i}", expected_answer=f"A{i}")
            for i in range(3)
        ]

        adapter = MockAdapter(instances=instances)
        runner = AURORABenchmarkRunner(memory=memory, adapters={"test": adapter})

        progress_calls = []
        def callback(current, total):
            progress_calls.append((current, total))

        runner.run_benchmark("test", "/fake/path", progress_callback=callback)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    def test_run_benchmark_handles_evaluation_error(self):
        """测试评估错误被捕获并记录。"""
        memory = MockMemory()

        class FailingAdapter(MockAdapter):
            def evaluate(self, instance, memory):
                if instance.id == "inst_2":
                    raise ValueError("Test error")
                return super().evaluate(instance, memory)

        instances = [
            BenchmarkInstance(id="inst_1", query="Q1", expected_answer="A1"),
            BenchmarkInstance(id="inst_2", query="Q2", expected_answer="A2"),
        ]

        adapter = FailingAdapter(instances=instances)
        runner = AURORABenchmarkRunner(memory=memory, adapters={"test": adapter})

        output = runner.run_benchmark("test", "/fake/path")

        assert output["num_instances"] == 2
        # 第二个结果应该记录错误
        result_2 = next(r for r in output["results"] if r["instance_id"] == "inst_2")
        assert result_2["score"] == 0.0
        assert "error" in result_2["metadata"]

    def test_run_all_benchmarks(self):
        """测试运行所有已注册的基准测试。"""
        memory = MockMemory()

        instances1 = [BenchmarkInstance(id="a1", query="Q1", expected_answer="A1")]
        instances2 = [BenchmarkInstance(id="b1", query="Q2", expected_answer="A2")]

        runner = AURORABenchmarkRunner(
            memory=memory,
            adapters={
                "bench1": MockAdapter(instances=instances1),
                "bench2": MockAdapter(instances=instances2),
            }
        )

        datasets = {
            "bench1": "/path/1",
            "bench2": "/path/2",
        }

        all_results = runner.run_all(datasets)

        assert "bench1" in all_results
        assert "bench2" in all_results
        assert all_results["bench1"]["num_instances"] == 1
        assert all_results["bench2"]["num_instances"] == 1

    def test_generate_report_markdown(self):
        """测试生成 markdown 报告。"""
        memory = MockMemory()
        runner = AURORABenchmarkRunner(memory=memory)

        results = {
            "test_bench": {
                "benchmark": "TestBenchmark",
                "num_instances": 10,
                "metrics": {
                    "accuracy": 0.85,
                    "mean_latency_ms": 45.2,
                }
            }
        }

        report = runner.generate_report(results, format="markdown")

        assert "# AURORA Benchmark Report" in report
        assert "test_bench" in report
        assert "85.00%" in report

    def test_generate_report_text(self):
        """测试生成文本报告。"""
        memory = MockMemory()
        runner = AURORABenchmarkRunner(memory=memory)

        results = {
            "test_bench": {
                "benchmark": "TestBenchmark",
                "num_instances": 10,
                "metrics": {
                    "accuracy": 0.90,
                    "mean_latency_ms": 30.0,
                }
            }
        }

        report = runner.generate_report(results, format="text")

        assert "AURORA Benchmark Report" in report
        assert "90.00%" in report


# =============================================================================
# 测试评估辅助函数
# =============================================================================

class TestEvaluationHelpers:
    """评估辅助函数的测试。"""

    def test_exact_match_score_identical(self):
        """测试精确匹配相同字符串。"""
        assert exact_match_score("Alice", "Alice") == 1.0

    def test_exact_match_score_case_insensitive(self):
        """测试精确匹配不区分大小写。"""
        assert exact_match_score("ALICE", "alice") == 1.0

    def test_exact_match_score_with_whitespace(self):
        """测试精确匹配处理空格。"""
        assert exact_match_score("  Alice  ", "Alice") == 1.0

    def test_exact_match_score_different(self):
        """测试精确匹配不同字符串。"""
        assert exact_match_score("Alice", "Bob") == 0.0

    def test_contains_score_match(self):
        """测试包含分数当预测包含真实值时。"""
        assert contains_score("The answer is Alice", "Alice") == 1.0

    def test_contains_score_no_match(self):
        """测试包含分数当不匹配时。"""
        assert contains_score("The answer is Bob", "Alice") == 0.0

    def test_fuzzy_match_score_exact(self):
        """测试模糊匹配精确匹配。"""
        score = fuzzy_match_score("Alice", "Alice")
        assert score == 1.0

    def test_fuzzy_match_score_similar(self):
        """测试模糊匹配相似字符串。"""
        score = fuzzy_match_score("Alice", "Alise")  # 拼写错误
        assert score > 0.5

    def test_fuzzy_match_score_different(self):
        """测试模糊匹配非常不同的字符串。"""
        score = fuzzy_match_score("Alice", "XYZ123")
        assert score == 0.0  # 低于默认阈值

    def test_compute_f1_score_exact(self):
        """测试 F1 分数精确匹配。"""
        f1 = compute_f1_score("hello world", "hello world")
        assert f1 == 1.0

    def test_compute_f1_score_partial(self):
        """测试 F1 分数部分重叠。"""
        f1 = compute_f1_score("hello world", "hello there")
        assert 0.0 < f1 < 1.0

    def test_compute_f1_score_no_overlap(self):
        """测试 F1 分数无重叠。"""
        f1 = compute_f1_score("apple orange", "banana grape")
        assert f1 == 0.0

    def test_compute_f1_score_empty(self):
        """测试 F1 分数空字符串。"""
        assert compute_f1_score("", "hello") == 0.0
        assert compute_f1_score("hello", "") == 0.0


# =============================================================================
# 测试 EvaluationConfig
# =============================================================================

class TestEvaluationConfig:
    """EvaluationConfig 数据类的测试。"""

    def test_default_config(self):
        """测试默认配置值。"""
        config = EvaluationConfig()

        assert config.use_llm_judge is True
        assert config.judge_model == "gpt-4"
        assert config.judge_temperature == 0.0
        assert config.max_retries == 3
        assert config.timeout_s == 60.0
        assert config.batch_size == 1

    def test_custom_config(self):
        """测试自定义配置。"""
        config = EvaluationConfig(
            use_llm_judge=False,
            max_instances=100,
            capabilities_filter=[BenchmarkCapability.ACCURATE_RETRIEVAL],
        )

        assert config.use_llm_judge is False
        assert config.max_instances == 100
        assert len(config.capabilities_filter) == 1


# =============================================================================
# 测试 EvaluationMetrics
# =============================================================================

class TestEvaluationMetrics:
    """EvaluationMetrics 数据类的测试。"""

    def test_default_metrics(self):
        """测试默认指标值。"""
        metrics = EvaluationMetrics()

        assert metrics.total_instances == 0
        assert metrics.correct_instances == 0
        assert metrics.accuracy == 0.0
        assert metrics.avg_score == 0.0

    def test_metrics_with_values(self):
        """测试具有实际值的指标。"""
        metrics = EvaluationMetrics(
            total_instances=100,
            correct_instances=85,
            accuracy=0.85,
            avg_score=0.87,
            avg_latency_ms=45.2,
            p50_latency_ms=40.0,
            p99_latency_ms=120.0,
        )

        assert metrics.total_instances == 100
        assert metrics.accuracy == 0.85
        assert metrics.p99_latency_ms == 120.0

    def test_metrics_to_state_dict(self):
        """测试序列化到状态字典。"""
        metrics = EvaluationMetrics(
            total_instances=50,
            accuracy=0.80,
            metrics_by_type={"ar": {"accuracy": 0.90}},
        )

        state = metrics.to_state_dict()

        assert state["total_instances"] == 50
        assert state["accuracy"] == 0.80
        assert "ar" in state["metrics_by_type"]
