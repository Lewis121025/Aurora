"""
Tests for aurora/benchmark/adapters/locomo.py
=============================================

Tests the LOCOMO benchmark adapter:
- Adapter initialization
- Session parsing
- Memory preparation from sessions
- QA evaluation (single-hop, multi-hop, temporal)
- Summarization evaluation
- Results aggregation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aurora.benchmark.interface import (
    BenchmarkInstance,
    BenchmarkResult,
    EvaluationMetrics,
    EvaluationMethod,
)
from aurora.benchmark.adapters.locomo import (
    LOCOMOAdapter,
    LOCOMOReasoningType,
    LOCOMOTaskType,
    LOCOMOTurn,
    LOCOMOSession,
    LOCOMOQuestion,
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
        {"speaker": "assistant", "text": "Nice to meet you, Bob!"}
    ]
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
    "personas": {
        "user": "A software engineer from San Francisco named Alice"
    },
    "events": [
        {"event": "introduction", "turn_id": "turn_1"},
        {"event": "location_mention", "turn_id": "turn_3"},
    ]
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
    """Mock AURORA memory for testing."""
    
    def __init__(self):
        self.plots = {}
        self.stories = {}
        self.themes = {}
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
    """Mock graph for testing."""
    
    def __init__(self, plots):
        self._plots = plots
    
    def payload(self, node_id):
        return self._plots.get(node_id)


# =============================================================================
# Test Adapter Initialization
# =============================================================================

class TestAdapterInitialization:
    """Tests for LOCOMOAdapter initialization."""
    
    def test_basic_initialization(self):
        """Test basic adapter initialization."""
        adapter = LOCOMOAdapter()
        
        assert adapter.name == "LOCOMO"
        assert adapter.llm is None
        assert adapter.evaluation_method == EvaluationMethod.LLM_JUDGE
    
    def test_initialization_with_llm(self):
        """Test initialization with LLM provider."""
        mock_llm = MagicMock()
        adapter = LOCOMOAdapter(llm_provider=mock_llm)
        
        assert adapter.llm is mock_llm
    
    def test_initialization_with_seed(self):
        """Test initialization with seed for reproducibility."""
        adapter1 = LOCOMOAdapter(seed=42)
        adapter2 = LOCOMOAdapter(seed=42)
        
        # Same seed should produce same random values
        val1 = adapter1.rng.random()
        val2 = adapter2.rng.random()
        assert val1 == val2
    
    def test_initialization_with_evaluation_method(self):
        """Test initialization with custom evaluation method."""
        adapter = LOCOMOAdapter(evaluation_method=EvaluationMethod.FUZZY)
        
        assert adapter.evaluation_method == EvaluationMethod.FUZZY
    
    def test_initialization_with_thresholds(self):
        """Test initialization with custom thresholds."""
        adapter = LOCOMOAdapter(
            f1_threshold=0.6,
            fuzzy_threshold=0.8,
        )
        
        assert adapter.f1_threshold == 0.6
        assert adapter.fuzzy_threshold == 0.8
    
    def test_factory_function(self):
        """Test create_locomo_adapter factory function."""
        adapter = create_locomo_adapter(seed=42)
        
        assert isinstance(adapter, LOCOMOAdapter)
        assert adapter.evaluation_method == EvaluationMethod.FUZZY  # No LLM provided
    
    def test_factory_with_llm(self):
        """Test factory function with LLM."""
        mock_llm = MagicMock()
        adapter = create_locomo_adapter(llm_provider=mock_llm, use_llm_judge=True)
        
        assert adapter.evaluation_method == EvaluationMethod.LLM_JUDGE


# =============================================================================
# Test Session Parsing
# =============================================================================

class TestParseSession:
    """Tests for session parsing functionality."""
    
    def test_parse_basic_session(self):
        """Test parsing basic session data."""
        adapter = LOCOMOAdapter()
        
        session = adapter._parse_session(MOCK_LOCOMO_SESSION)
        
        assert session.session_id == "test_session_001"
        assert len(session.turns) == 2
        assert session.turns[0].speaker == "user"
        assert session.turns[0].text == "Hello, I'm Bob."
    
    def test_parse_session_with_metadata(self):
        """Test parsing session with personas and events."""
        adapter = LOCOMOAdapter()
        
        session = adapter._parse_session(MOCK_LOCOMO_SESSION_FULL)
        
        assert session.session_id == "session_002"
        assert len(session.turns) == 6
        assert "user" in session.personas
        assert len(session.events) == 2
    
    def test_parse_session_string_turns(self):
        """Test parsing session with string turns."""
        adapter = LOCOMOAdapter()
        
        session_data = {
            "session_id": "test_simple",
            "turns": ["Hello", "Hi there"],
        }
        
        session = adapter._parse_session(session_data)
        
        assert len(session.turns) == 2
        assert session.turns[0].text == "Hello"
    
    def test_locomo_turn_to_dict(self):
        """Test LOCOMOTurn.to_dict() method."""
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
    """Tests for memory preparation from LOCOMO sessions."""
    
    def test_prepare_memory_basic(self):
        """Test basic memory preparation from conversation history."""
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
        """Test that user turns are formatted correctly."""
        adapter = LOCOMOAdapter()
        memory = MockMemory()
        
        conversation_history = [
            {"speaker": "user", "text": "Hello"},
        ]
        
        adapter.prepare_memory(conversation_history, memory)
        
        # Should format as "用户：Hello"
        assert "用户" in memory._ingested[0] or "Hello" in memory._ingested[0]
    
    def test_prepare_memory_multiple_sessions(self):
        """Test preparing memory from multiple sessions."""
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
    """Tests for single-hop QA evaluation."""
    
    def test_evaluate_single_hop_basic(self):
        """Test basic single-hop QA evaluation."""
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
        """Test that evaluation records reasoning trace."""
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
    """Tests for multi-hop QA evaluation."""
    
    def test_evaluate_multi_hop_basic(self):
        """Test basic multi-hop QA evaluation."""
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
    """Tests for temporal QA evaluation."""
    
    def test_evaluate_temporal_basic(self):
        """Test basic temporal QA evaluation."""
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
    """Tests for event summarization evaluation."""
    
    def test_evaluate_summarization_basic(self):
        """Test basic summarization evaluation."""
        adapter = LOCOMOAdapter(
            seed=42, 
            evaluation_method=EvaluationMethod.FUZZY,
            use_narrator_for_summary=False,  # Use simple summary for testing
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
        """Test that summarization returns evaluation metrics."""
        adapter = LOCOMOAdapter(
            seed=42,
            evaluation_method=EvaluationMethod.FUZZY,
            use_narrator_for_summary=False,
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
    """Tests for results aggregation."""
    
    def test_aggregate_empty_results(self):
        """Test aggregation with empty results."""
        adapter = LOCOMOAdapter()
        
        metrics = adapter.aggregate_results([])
        
        assert metrics["accuracy"] == 0.0
        assert metrics["mean_latency_ms"] == 0.0
    
    def test_aggregate_single_result(self):
        """Test aggregation with single result."""
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
        """Test aggregation with multiple results."""
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
        """Test aggregation includes reasoning type breakdown."""
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
        """Test that aggregation includes latency percentiles."""
        adapter = LOCOMOAdapter()
        
        results = [
            BenchmarkResult(instance_id=f"t{i}", task_type="qa_single_hop",
                           score=0.5, is_correct=True, latency_ms=float(i * 10))
            for i in range(1, 11)
        ]
        
        metrics = adapter.aggregate_results(results)
        
        assert "p50_latency_ms" in metrics
        assert "p95_latency_ms" in metrics
        assert "p99_latency_ms" in metrics
    
    def test_get_evaluation_metrics_object(self):
        """Test getting EvaluationMetrics object."""
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
    """Tests for LOCOMOReasoningType enum."""
    
    def test_reasoning_type_values(self):
        """Test reasoning type values."""
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
        """Test from_string returns single_hop for unknown."""
        assert LOCOMOReasoningType.from_string("unknown") == LOCOMOReasoningType.SINGLE_HOP


# =============================================================================
# Test Task Type Enum
# =============================================================================

class TestLOCOMOTaskType:
    """Tests for LOCOMOTaskType enum."""
    
    def test_task_type_values(self):
        """Test task type values."""
        assert LOCOMOTaskType.QUESTION_ANSWERING.value == "qa"
        assert LOCOMOTaskType.EVENT_SUMMARIZATION.value == "summarization"
        assert LOCOMOTaskType.DIALOGUE_GENERATION.value == "dialogue"
    
    def test_from_string_variations(self):
        """Test from_string handles variations."""
        assert LOCOMOTaskType.from_string("qa") == LOCOMOTaskType.QUESTION_ANSWERING
        assert LOCOMOTaskType.from_string("question_answering") == LOCOMOTaskType.QUESTION_ANSWERING
        
        assert LOCOMOTaskType.from_string("summarization") == LOCOMOTaskType.EVENT_SUMMARIZATION
        assert LOCOMOTaskType.from_string("summary") == LOCOMOTaskType.EVENT_SUMMARIZATION


# =============================================================================
# Test Dataset Loading
# =============================================================================

class TestDatasetLoading:
    """Tests for dataset loading functionality."""
    
    def test_load_from_json_file(self, tmp_path):
        """Test loading from a JSON file."""
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
        """Test loading from a directory with split files."""
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
        """Test loading with reasoning type filter."""
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
        """Test loading with instance limit."""
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
        """Test that loading includes summarization tasks."""
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
    """Tests for evaluation statistics tracking."""
    
    def test_get_evaluation_stats(self):
        """Test getting evaluation statistics."""
        adapter = LOCOMOAdapter(evaluation_method=EvaluationMethod.FUZZY)
        
        stats = adapter.get_evaluation_stats()
        
        assert "total_evals" in stats
        assert "llm_evals" in stats
        assert "fallback_evals" in stats
    
    def test_stats_increment_after_evaluation(self):
        """Test that stats increment after evaluation."""
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
    """Tests for error handling in evaluation."""
    
    def test_evaluate_handles_memory_error(self):
        """Test that evaluation handles memory errors gracefully."""
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
        """Test loading non-existent file raises error."""
        adapter = LOCOMOAdapter()
        
        with pytest.raises(FileNotFoundError):
            adapter.load_dataset("/nonexistent/path/to/data.json")
