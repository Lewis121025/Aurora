"""
Tests for aurora/benchmark/interface.py
=======================================

Tests the core benchmark interface components:
- BenchmarkCapability enum
- BenchmarkInstance dataclass
- BenchmarkResult dataclass
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
# Mock Data
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
# Test BenchmarkCapability Enum
# =============================================================================

class TestBenchmarkCapabilityEnum:
    """Tests for BenchmarkCapability enum."""
    
    def test_capability_values(self):
        """Test that all capability values are correct."""
        assert BenchmarkCapability.ACCURATE_RETRIEVAL.value == "accurate_retrieval"
        assert BenchmarkCapability.TEST_TIME_LEARNING.value == "test_time_learning"
        assert BenchmarkCapability.LONG_RANGE_UNDERSTANDING.value == "long_range_understanding"
        assert BenchmarkCapability.CONFLICT_RESOLUTION.value == "conflict_resolution"
    
    def test_capability_str(self):
        """Test capability string representation."""
        assert str(BenchmarkCapability.ACCURATE_RETRIEVAL) == "accurate_retrieval"
        assert str(BenchmarkCapability.CONFLICT_RESOLUTION) == "conflict_resolution"
    
    def test_capability_from_string(self):
        """Test creating capability from string."""
        cap = BenchmarkCapability.from_string("accurate_retrieval")
        assert cap == BenchmarkCapability.ACCURATE_RETRIEVAL
        
        cap = BenchmarkCapability.from_string("test_time_learning")
        assert cap == BenchmarkCapability.TEST_TIME_LEARNING
    
    def test_capability_from_string_invalid(self):
        """Test from_string with invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Unknown capability"):
            BenchmarkCapability.from_string("invalid_capability")
    
    def test_all_capabilities_iterable(self):
        """Test that all capabilities are iterable."""
        capabilities = list(BenchmarkCapability)
        assert len(capabilities) == 4
        assert BenchmarkCapability.ACCURATE_RETRIEVAL in capabilities


# =============================================================================
# Test BenchmarkInstance Creation
# =============================================================================

class TestBenchmarkInstanceCreation:
    """Tests for BenchmarkInstance dataclass."""
    
    def test_basic_instance_creation(self):
        """Test creating a basic benchmark instance."""
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
        """Test creating instance with capability."""
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
        """Test creating instance with conversation history."""
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
        # ground_truth should sync to expected_answer
        assert instance.expected_answer == "Hi there!"
    
    def test_instance_ground_truth_sync(self):
        """Test that ground_truth and expected_answer are synchronized."""
        # Test expected_answer -> ground_truth
        instance1 = BenchmarkInstance(
            id="test_004",
            query="Test",
            expected_answer="Answer",
        )
        assert instance1.ground_truth == "Answer"
        
        # Test ground_truth -> expected_answer
        instance2 = BenchmarkInstance(
            id="test_005",
            query="Test",
            ground_truth="Answer2",
        )
        assert instance2.expected_answer == "Answer2"
    
    def test_instance_to_state_dict(self):
        """Test serialization to state dict."""
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
        """Test reconstruction from state dict."""
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
# Test BenchmarkResult Creation
# =============================================================================

class TestBenchmarkResultCreation:
    """Tests for BenchmarkResult dataclass."""
    
    def test_basic_result_creation(self):
        """Test creating a basic benchmark result."""
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
        """Test result with latency metrics."""
        result = BenchmarkResult(
            instance_id="test_002",
            score=1.0,
            latency_ms=45.5,
            tokens_used=150,
        )
        
        assert result.latency_ms == 45.5
        assert result.tokens_used == 150
    
    def test_result_with_capability(self):
        """Test result with capability field."""
        result = BenchmarkResult(
            instance_id="test_003",
            score=0.9,
            capability=BenchmarkCapability.TEST_TIME_LEARNING,
            task_type="test_time_learning",
        )
        
        assert result.capability == BenchmarkCapability.TEST_TIME_LEARNING
        assert result.task_type == "test_time_learning"
    
    def test_result_prediction_sync(self):
        """Test that prediction and predicted are synchronized."""
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
        """Test that ground_truth and expected are synchronized."""
        result = BenchmarkResult(
            instance_id="test_006",
            score=1.0,
            ground_truth="Truth",
        )
        assert result.expected == "Truth"
    
    def test_result_check_correct_default_threshold(self):
        """Test check_correct with default threshold."""
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
        """Test check_correct with custom threshold."""
        result = BenchmarkResult(
            instance_id="test_009",
            score=0.7,
        )
        assert result.check_correct(threshold=0.8) is False
        assert result.check_correct(threshold=0.5) is True
    
    def test_result_to_state_dict(self):
        """Test serialization to state dict."""
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
        """Test reconstruction from state dict."""
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
# Test AURORABenchmarkRunner
# =============================================================================

class MockMemory:
    """Mock memory for testing."""
    
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
    """Mock adapter for testing."""
    
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
    """Tests for AURORABenchmarkRunner."""
    
    def test_runner_initialization(self):
        """Test runner initialization."""
        memory = MockMemory()
        runner = AURORABenchmarkRunner(memory=memory)
        
        assert runner.memory is memory
        assert len(runner.adapters) == 0
    
    def test_runner_with_adapters(self):
        """Test runner initialization with adapters."""
        memory = MockMemory()
        adapter = MockAdapter()
        
        runner = AURORABenchmarkRunner(
            memory=memory,
            adapters={"mock": adapter},
        )
        
        assert "mock" in runner.adapters
        assert runner.adapters["mock"].name == "MockBenchmark"
    
    def test_register_adapter(self):
        """Test registering a new adapter."""
        memory = MockMemory()
        runner = AURORABenchmarkRunner(memory=memory)
        adapter = MockAdapter()
        
        runner.register_adapter("test", adapter)
        
        assert "test" in runner.adapters
    
    def test_run_benchmark_unknown_adapter(self):
        """Test running benchmark with unknown adapter raises error."""
        memory = MockMemory()
        runner = AURORABenchmarkRunner(memory=memory)
        
        with pytest.raises(KeyError, match="Unknown benchmark"):
            runner.run_benchmark("nonexistent", "/path/to/data")
    
    def test_run_benchmark_basic(self):
        """Test basic benchmark run."""
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
        assert output["metrics"]["accuracy"] == 1.0  # both >= 0.5
    
    def test_run_benchmark_with_subset(self):
        """Test running benchmark with subset filter."""
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
        """Test benchmark run with progress callback."""
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
        """Test that evaluation errors are caught and recorded."""
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
        # Second result should have error recorded
        result_2 = next(r for r in output["results"] if r["instance_id"] == "inst_2")
        assert result_2["score"] == 0.0
        assert "error" in result_2["metadata"]
    
    def test_run_all_benchmarks(self):
        """Test running all registered benchmarks."""
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
        """Test generating markdown report."""
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
        """Test generating text report."""
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
# Test Evaluation Helper Functions
# =============================================================================

class TestEvaluationHelpers:
    """Tests for evaluation helper functions."""
    
    def test_exact_match_score_identical(self):
        """Test exact match with identical strings."""
        assert exact_match_score("Alice", "Alice") == 1.0
    
    def test_exact_match_score_case_insensitive(self):
        """Test exact match is case insensitive."""
        assert exact_match_score("ALICE", "alice") == 1.0
    
    def test_exact_match_score_with_whitespace(self):
        """Test exact match handles whitespace."""
        assert exact_match_score("  Alice  ", "Alice") == 1.0
    
    def test_exact_match_score_different(self):
        """Test exact match with different strings."""
        assert exact_match_score("Alice", "Bob") == 0.0
    
    def test_contains_score_match(self):
        """Test contains score when prediction contains ground truth."""
        assert contains_score("The answer is Alice", "Alice") == 1.0
    
    def test_contains_score_no_match(self):
        """Test contains score when no match."""
        assert contains_score("The answer is Bob", "Alice") == 0.0
    
    def test_fuzzy_match_score_exact(self):
        """Test fuzzy match with exact match."""
        score = fuzzy_match_score("Alice", "Alice")
        assert score == 1.0
    
    def test_fuzzy_match_score_similar(self):
        """Test fuzzy match with similar strings."""
        score = fuzzy_match_score("Alice", "Alise")  # typo
        assert score > 0.5
    
    def test_fuzzy_match_score_different(self):
        """Test fuzzy match with very different strings."""
        score = fuzzy_match_score("Alice", "XYZ123")
        assert score == 0.0  # Below default threshold
    
    def test_compute_f1_score_exact(self):
        """Test F1 score with exact match."""
        f1 = compute_f1_score("hello world", "hello world")
        assert f1 == 1.0
    
    def test_compute_f1_score_partial(self):
        """Test F1 score with partial overlap."""
        f1 = compute_f1_score("hello world", "hello there")
        assert 0.0 < f1 < 1.0
    
    def test_compute_f1_score_no_overlap(self):
        """Test F1 score with no overlap."""
        f1 = compute_f1_score("apple orange", "banana grape")
        assert f1 == 0.0
    
    def test_compute_f1_score_empty(self):
        """Test F1 score with empty strings."""
        assert compute_f1_score("", "hello") == 0.0
        assert compute_f1_score("hello", "") == 0.0


# =============================================================================
# Test EvaluationConfig
# =============================================================================

class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        
        assert config.use_llm_judge is True
        assert config.judge_model == "gpt-4"
        assert config.judge_temperature == 0.0
        assert config.max_retries == 3
        assert config.timeout_s == 60.0
        assert config.batch_size == 1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EvaluationConfig(
            use_llm_judge=False,
            max_instances=100,
            capabilities_filter=[BenchmarkCapability.ACCURATE_RETRIEVAL],
        )
        
        assert config.use_llm_judge is False
        assert config.max_instances == 100
        assert len(config.capabilities_filter) == 1


# =============================================================================
# Test EvaluationMetrics
# =============================================================================

class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""
    
    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = EvaluationMetrics()
        
        assert metrics.total_instances == 0
        assert metrics.correct_instances == 0
        assert metrics.accuracy == 0.0
        assert metrics.avg_score == 0.0
    
    def test_metrics_with_values(self):
        """Test metrics with actual values."""
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
        """Test serialization to state dict."""
        metrics = EvaluationMetrics(
            total_instances=50,
            accuracy=0.80,
            metrics_by_type={"ar": {"accuracy": 0.90}},
        )
        
        state = metrics.to_state_dict()
        
        assert state["total_instances"] == 50
        assert state["accuracy"] == 0.80
        assert "ar" in state["metrics_by_type"]
