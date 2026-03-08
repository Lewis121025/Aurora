"""
Tests for aurora/benchmarks/adapters/memoryagentbench.py
=======================================================

Tests the MemoryAgentBench adapter:
- Adapter initialization
- Conversation parsing
- Memory preparation
- Capability-specific evaluation (AR, TTL, LRU, CR)
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

from aurora.benchmarks.interface import (
    BenchmarkCapability,
    BenchmarkInstance,
    BenchmarkResult,
    EvaluationConfig,
    EvaluationMetrics,
)
from aurora.benchmarks.adapters.memoryagentbench import (
    MemoryAgentBenchAdapter,
    parse_conversation_turns,
    extract_conflicting_facts,
    normalize_answer,
    exact_match_score,
    capability_to_task_type,
    task_type_to_capability,
    TASK_TYPE_AR,
    TASK_TYPE_TTL,
    TASK_TYPE_LRU,
    TASK_TYPE_CR,
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

MOCK_CONVERSATION = """User: Hello, my name is Alice.
Assistant: Nice to meet you, Alice!
User: I live in San Francisco.
Assistant: San Francisco is a beautiful city."""

MOCK_CR_CONTEXT = """User: I live in New York.
Assistant: Got it, you live in New York.
User: Actually, I moved to San Francisco last month.
Assistant: I've updated your location to San Francisco."""


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
    
    def ingest(self, interaction_text: str, event_id: str = None, **kwargs):
        self._ingested.append(interaction_text)
        
        # Create a mock plot
        plot_id = event_id or f"plot_{self._plot_counter}"
        self._plot_counter += 1
        
        mock_plot = MagicMock()
        mock_plot.id = plot_id
        mock_plot.text = interaction_text
        mock_plot.ts = now_ts()
        self.plots[plot_id] = mock_plot
    
    def query(self, text: str, k: int = 8, **kwargs):
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
    """Tests for MemoryAgentBenchAdapter initialization."""
    
    def test_basic_initialization(self):
        """Test basic adapter initialization."""
        adapter = MemoryAgentBenchAdapter()
        
        assert adapter.name == "MemoryAgentBench"
        assert adapter.llm is None
        assert adapter.embedder is None
    
    def test_initialization_with_llm(self):
        """Test initialization with LLM provider."""
        mock_llm = MagicMock()
        adapter = MemoryAgentBenchAdapter(llm_provider=mock_llm)
        
        assert adapter.llm is mock_llm
    
    def test_initialization_with_seed(self):
        """Test initialization with seed for reproducibility."""
        adapter1 = MemoryAgentBenchAdapter(seed=42)
        adapter2 = MemoryAgentBenchAdapter(seed=42)
        
        # Same seed should produce same random values
        val1 = adapter1.rng.random()
        val2 = adapter2.rng.random()
        assert val1 == val2
    
    def test_initialization_with_embedder(self):
        """Test initialization with custom embedder."""
        def mock_embedder(text: str) -> np.ndarray:
            return np.zeros(64)
        
        adapter = MemoryAgentBenchAdapter(embedder=mock_embedder)
        assert adapter.embedder is mock_embedder
    
    def test_capabilities_property(self):
        """Test that adapter reports all supported capabilities."""
        adapter = MemoryAgentBenchAdapter()
        caps = adapter.capabilities
        
        assert BenchmarkCapability.ACCURATE_RETRIEVAL in caps
        assert BenchmarkCapability.TEST_TIME_LEARNING in caps
        assert BenchmarkCapability.LONG_RANGE_UNDERSTANDING in caps
        assert BenchmarkCapability.CONFLICT_RESOLUTION in caps


# =============================================================================
# Test Conversation Parsing
# =============================================================================

class TestParseConversation:
    """Tests for parse_conversation_turns helper function."""
    
    def test_parse_basic_conversation(self):
        """Test parsing basic User/Assistant format."""
        context = "User: Hello\nAssistant: Hi there!"
        turns = parse_conversation_turns(context)
        
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Hello"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Hi there!"
    
    def test_parse_human_ai_format(self):
        """Test parsing Human/AI format (normalized to User/Assistant)."""
        context = "Human: Hello\nAI: Hi there!"
        turns = parse_conversation_turns(context)
        
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
    
    def test_parse_multi_turn_conversation(self):
        """Test parsing multi-turn conversation."""
        turns = parse_conversation_turns(MOCK_CONVERSATION)
        
        assert len(turns) == 4
        assert turns[0]["content"] == "Hello, my name is Alice."
        assert turns[3]["role"] == "assistant"
    
    def test_parse_empty_context(self):
        """Test parsing empty or whitespace context."""
        turns = parse_conversation_turns("")
        assert len(turns) == 0 or (len(turns) == 1 and turns[0]["content"] == "")
    
    def test_parse_unstructured_context(self):
        """Test parsing context without role markers."""
        context = "Some random text without markers"
        turns = parse_conversation_turns(context)
        
        # Should treat entire text as single user message
        assert len(turns) == 1
        assert turns[0]["content"] == context
    
    def test_parse_with_system_marker(self):
        """Test parsing with System: marker."""
        context = "System: You are a helpful assistant.\nUser: Hello\nAssistant: Hi!"
        turns = parse_conversation_turns(context)
        
        assert len(turns) == 3
        assert turns[0]["role"] == "system"
    
    def test_turns_have_text_field(self):
        """Test that turns have both content and text fields."""
        context = "User: Test message"
        turns = parse_conversation_turns(context)
        
        assert "content" in turns[0]
        assert "text" in turns[0]
        assert turns[0]["content"] == turns[0]["text"]


# =============================================================================
# Test Memory Preparation
# =============================================================================

class TestPrepareMemory:
    """Tests for memory preparation functionality."""
    
    def test_prepare_memory_basic(self):
        """Test basic memory preparation from instance."""
        adapter = MemoryAgentBenchAdapter()
        memory = MockMemory()
        
        instance = BenchmarkInstance(
            id="test_001",
            query="What is my name?",
            context=MOCK_CONVERSATION,
            conversation_history=parse_conversation_turns(MOCK_CONVERSATION),
            expected_answer="Alice",
        )
        
        # Call the internal preparation method
        adapter._prepare_memory_for_instance(instance, memory)
        
        # Memory should have ingested turns
        assert len(memory._ingested) > 0
    
    def test_prepare_memory_triggers_evolve(self):
        """Test that memory preparation triggers evolution."""
        adapter = MemoryAgentBenchAdapter()
        memory = MagicMock()
        memory.ingest = MagicMock()
        memory.evolve = MagicMock()
        
        instance = BenchmarkInstance(
            id="test_002",
            query="Test",
            conversation_history=[
                {"role": "user", "text": "Hello"},
                {"role": "assistant", "text": "Hi"},
            ],
            expected_answer="Hi",
        )
        
        adapter._prepare_memory_for_instance(instance, memory)
        
        memory.evolve.assert_called()


# =============================================================================
# Test Accurate Retrieval (AR) Evaluation
# =============================================================================

class TestEvaluateAR:
    """Tests for Accurate Retrieval capability evaluation."""
    
    def test_evaluate_ar_basic(self):
        """Test basic AR evaluation."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        instance = BenchmarkInstance(
            id="ar_001",
            query="What is my name?",
            capability=BenchmarkCapability.ACCURATE_RETRIEVAL,
            context="User: My name is Alice.\nAssistant: Nice to meet you, Alice!",
            conversation_history=[
                {"role": "user", "text": "My name is Alice."},
                {"role": "assistant", "text": "Nice to meet you, Alice!"},
            ],
            expected_answer="Alice",
            task_type=TASK_TYPE_AR,
            reasoning_type="accurate_retrieval",
        )
        
        result = adapter.evaluate(instance, memory)
        
        assert result.instance_id == "ar_001"
        assert result.capability == BenchmarkCapability.ACCURATE_RETRIEVAL
        assert isinstance(result.score, float)
        assert result.latency_ms >= 0
    
    def test_evaluate_ar_with_matching_answer(self):
        """Test AR evaluation when answer matches."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        # Pre-populate memory with the answer
        memory.ingest("User: My name is Alice. Assistant: Hello Alice!")
        
        instance = BenchmarkInstance(
            id="ar_002",
            query="What is my name?",
            capability=BenchmarkCapability.ACCURATE_RETRIEVAL,
            conversation_history=[
                {"role": "user", "text": "My name is Alice."},
            ],
            expected_answer="Alice",
            task_type=TASK_TYPE_AR,
            reasoning_type="accurate_retrieval",
        )
        
        result = adapter.evaluate(instance, memory)
        
        assert result.instance_id == "ar_002"
        # Score depends on retrieval results


# =============================================================================
# Test Test-Time Learning (TTL) Evaluation
# =============================================================================

class TestEvaluateTTL:
    """Tests for Test-Time Learning capability evaluation."""
    
    def test_evaluate_ttl_basic(self):
        """Test basic TTL evaluation."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        instance = BenchmarkInstance(
            id="ttl_001",
            query="Apply the new rule to this input",
            capability=BenchmarkCapability.TEST_TIME_LEARNING,
            context="User: New rule - always add 'please' at the end.\nAssistant: Got it.",
            conversation_history=[
                {"role": "user", "text": "New rule - always add 'please' at the end."},
                {"role": "assistant", "text": "Got it."},
            ],
            expected_answer="Result please",
            task_type=TASK_TYPE_TTL,
            reasoning_type="test_time_learning",
        )
        
        result = adapter.evaluate(instance, memory)
        
        assert result.instance_id == "ttl_001"
        assert result.capability == BenchmarkCapability.TEST_TIME_LEARNING


# =============================================================================
# Test Long-Range Understanding (LRU) Evaluation
# =============================================================================

class TestEvaluateLRU:
    """Tests for Long-Range Understanding capability evaluation."""
    
    def test_evaluate_lru_basic(self):
        """Test basic LRU evaluation."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        # Create a longer conversation history
        conversation = [
            {"role": "user", "text": "Let me tell you about my trip."},
            {"role": "assistant", "text": "I'd love to hear about it!"},
            {"role": "user", "text": "First I went to Paris."},
            {"role": "assistant", "text": "Paris is beautiful!"},
            {"role": "user", "text": "Then I visited Rome."},
            {"role": "assistant", "text": "Rome has amazing history!"},
        ]
        
        instance = BenchmarkInstance(
            id="lru_001",
            query="Summarize the trip",
            capability=BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
            conversation_history=conversation,
            expected_answer="The user visited Paris and Rome",
            task_type=TASK_TYPE_LRU,
            reasoning_type="long_range_understanding",
        )
        
        result = adapter.evaluate(instance, memory)
        
        assert result.instance_id == "lru_001"
        assert result.capability == BenchmarkCapability.LONG_RANGE_UNDERSTANDING


# =============================================================================
# Test Conflict Resolution (CR) Evaluation
# =============================================================================

class TestEvaluateCR:
    """Tests for Conflict Resolution capability evaluation."""
    
    def test_evaluate_cr_basic(self):
        """Test basic CR evaluation."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        instance = BenchmarkInstance(
            id="cr_001",
            query="Where do I live now?",
            capability=BenchmarkCapability.CONFLICT_RESOLUTION,
            context=MOCK_CR_CONTEXT,
            conversation_history=parse_conversation_turns(MOCK_CR_CONTEXT),
            expected_answer="San Francisco",
            task_type=TASK_TYPE_CR,
            reasoning_type="conflict_resolution",
            metadata={
                "old_fact": "New York",
                "new_fact": "San Francisco",
            },
        )
        
        result = adapter.evaluate(instance, memory)
        
        assert result.instance_id == "cr_001"
        assert result.capability == BenchmarkCapability.CONFLICT_RESOLUTION
    
    def test_extract_conflicting_facts(self):
        """Test extraction of conflicting facts from context."""
        metadata = {
            "old_fact": "New York",
            "new_fact": "San Francisco",
        }
        
        facts = extract_conflicting_facts(MOCK_CR_CONTEXT, metadata)
        
        assert len(facts) >= 2
        assert any("New York" in f for f in facts)
        assert any("San Francisco" in f for f in facts)


# =============================================================================
# Test Results Aggregation
# =============================================================================

class TestAggregateResults:
    """Tests for results aggregation."""
    
    def test_aggregate_via_run_benchmark(self, tmp_path):
        """Test aggregation through run_benchmark_with_config."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        test_data = [
            {
                "id": "ar_1",
                "capability": "accurate_retrieval",
                "context": "User: My name is Alice.",
                "question": "What is my name?",
                "answer": "Alice",
            },
        ]
        
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)
        
        config = EvaluationConfig(use_llm_judge=False, verbose=False)
        
        results, metrics = adapter.run_benchmark_with_config(
            memory=memory,
            source=str(data_file),
            config=config,
        )
        
        assert len(results) == 1
        # run_benchmark_with_config now returns EvaluationMetrics via get_evaluation_metrics()
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.total_instances == 1
    
    def test_evaluate_directly_returns_result(self):
        """Test that evaluate() works correctly without aggregation."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        instance = BenchmarkInstance(
            id="test_001",
            query="What is my name?",
            capability=BenchmarkCapability.ACCURATE_RETRIEVAL,
            context="User: My name is Alice.",
            conversation_history=[
                {"role": "user", "text": "My name is Alice."},
            ],
            expected_answer="Alice",
            task_type=TASK_TYPE_AR,
            reasoning_type="accurate_retrieval",
        )
        
        result = adapter.evaluate(instance, memory)
        
        # Verify result structure
        assert result.instance_id == "test_001"
        assert isinstance(result.score, float)
        assert result.latency_ms >= 0
        assert result.capability == BenchmarkCapability.ACCURATE_RETRIEVAL
    
    def test_multiple_evaluations_collect_results(self):
        """Test evaluating multiple instances and collecting results."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        instances = [
            BenchmarkInstance(
                id=f"test_{i}",
                query=f"Question {i}",
                expected_answer=f"Answer {i}",
                task_type=TASK_TYPE_AR,
                reasoning_type="accurate_retrieval",
            )
            for i in range(3)
        ]
        
        results = []
        for instance in instances:
            memory.clear()
            result = adapter.evaluate(instance, memory)
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert all(r.latency_ms >= 0 for r in results)
    
    def test_results_have_scores(self):
        """Test that evaluation results have valid scores."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        # Add content to memory
        memory.ingest("User: My name is Alice. Assistant: Nice to meet you!")
        
        instance = BenchmarkInstance(
            id="test_score",
            query="What is my name?",
            expected_answer="Alice",
            task_type=TASK_TYPE_AR,
            reasoning_type="accurate_retrieval",
        )
        
        result = adapter.evaluate(instance, memory)
        
        # Score should be between 0 and 1
        assert 0.0 <= result.score <= 1.0


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_normalize_answer_lowercase(self):
        """Test answer normalization lowercases."""
        assert normalize_answer("ALICE") == "alice"
    
    def test_normalize_answer_strips_whitespace(self):
        """Test answer normalization strips whitespace."""
        assert normalize_answer("  Alice  ") == "alice"
    
    def test_normalize_answer_removes_prefixes(self):
        """Test answer normalization removes common prefixes."""
        assert normalize_answer("The answer is Alice") == "alice"
        assert normalize_answer("Answer: Alice") == "alice"
    
    def test_exact_match_score_identical(self):
        """Test exact match with identical strings."""
        is_correct, score = exact_match_score("alice", "alice")
        assert is_correct is True
        assert score == 1.0
    
    def test_exact_match_score_substring(self):
        """Test exact match with substring match."""
        is_correct, score = exact_match_score("alice is here", "alice")
        assert is_correct is True
        assert score > 0.8
    
    def test_exact_match_score_no_match(self):
        """Test exact match with no match."""
        is_correct, score = exact_match_score("xyz", "alice")
        assert is_correct is False
        assert score < 0.5
    
    def test_capability_to_task_type(self):
        """Test capability to task type conversion."""
        assert capability_to_task_type(BenchmarkCapability.ACCURATE_RETRIEVAL) == TASK_TYPE_AR
        assert capability_to_task_type(BenchmarkCapability.TEST_TIME_LEARNING) == TASK_TYPE_TTL
        assert capability_to_task_type(BenchmarkCapability.LONG_RANGE_UNDERSTANDING) == TASK_TYPE_LRU
        assert capability_to_task_type(BenchmarkCapability.CONFLICT_RESOLUTION) == TASK_TYPE_CR
    
    def test_task_type_to_capability(self):
        """Test task type to capability conversion."""
        assert task_type_to_capability("accurate_retrieval") == BenchmarkCapability.ACCURATE_RETRIEVAL
        assert task_type_to_capability("ar") == BenchmarkCapability.ACCURATE_RETRIEVAL
        assert task_type_to_capability("ttl") == BenchmarkCapability.TEST_TIME_LEARNING


# =============================================================================
# Test Dataset Loading
# =============================================================================

class TestDatasetLoading:
    """Tests for dataset loading functionality."""
    
    def test_load_from_local_json(self, tmp_path):
        """Test loading dataset from local JSON file."""
        adapter = MemoryAgentBenchAdapter()
        
        # Create test data file
        test_data = [
            {
                "id": "test_1",
                "capability": "accurate_retrieval",
                "context": "User: My name is Alice.",
                "question": "What is my name?",
                "answer": "Alice",
            }
        ]
        
        data_file = tmp_path / "test_dataset.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)
        
        instances = adapter.load_dataset(str(data_file))
        
        assert len(instances) == 1
        assert instances[0].id == "test_1"
        assert instances[0].query == "What is my name?"
    
    def test_load_from_local_directory(self, tmp_path):
        """Test loading dataset from local directory."""
        adapter = MemoryAgentBenchAdapter()
        
        # Create test data file in directory
        test_data = [
            {
                "id": "test_1",
                "question": "Q1",
                "answer": "A1",
            }
        ]
        
        with open(tmp_path / "test.json", "w") as f:
            json.dump(test_data, f)
        
        instances = adapter.load_dataset(str(tmp_path))
        
        assert len(instances) >= 1
    
    def test_load_with_limit(self, tmp_path):
        """Test loading with instance limit."""
        adapter = MemoryAgentBenchAdapter()
        
        # Create multiple test instances
        test_data = [
            {"id": f"test_{i}", "question": f"Q{i}", "answer": f"A{i}"}
            for i in range(10)
        ]
        
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)
        
        instances = adapter.load_dataset(str(data_file), limit=3)
        
        assert len(instances) == 3
    
    def test_load_caches_results(self, tmp_path):
        """Test that loading caches results."""
        adapter = MemoryAgentBenchAdapter()
        
        test_data = [{"id": "test_1", "question": "Q1", "answer": "A1"}]
        
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)
        
        # Load twice
        instances1 = adapter.load_dataset(str(data_file))
        instances2 = adapter.load_dataset(str(data_file))
        
        # Should return cached results
        assert instances1 == instances2


# =============================================================================
# Test Full Benchmark Run
# =============================================================================

class TestRunBenchmarkWithConfig:
    """Tests for run_benchmark_with_config method."""
    
    def test_run_with_default_config(self, tmp_path):
        """Test running benchmark with default config."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        test_data = [
            {
                "id": "test_1",
                "capability": "accurate_retrieval",
                "context": "User: Hello\nAssistant: Hi there!",
                "question": "What was said?",
                "answer": "Hello",
            }
        ]
        
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)
        
        config = EvaluationConfig(
            use_llm_judge=False,
            verbose=False,
            max_instances=1,
        )
        
        results, metrics = adapter.run_benchmark_with_config(
            memory=memory,
            source=str(data_file),
            config=config,
        )
        
        assert len(results) == 1
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.total_instances == 1
    
    def test_run_with_capability_filter(self, tmp_path):
        """Test running benchmark with capability filter."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        test_data = [
            {"id": "ar_1", "capability": "ar", "context": "User: Test", "question": "Q1", "answer": "A1"},
            {"id": "ttl_1", "capability": "ttl", "context": "User: Test", "question": "Q2", "answer": "A2"},
        ]
        
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)
        
        config = EvaluationConfig(
            use_llm_judge=False,
            verbose=False,
            capabilities_filter=[BenchmarkCapability.ACCURATE_RETRIEVAL],
        )
        
        results, metrics = adapter.run_benchmark_with_config(
            memory=memory,
            source=str(data_file),
            config=config,
        )
        
        assert len(results) == 1
        assert isinstance(metrics, EvaluationMetrics)
    
    def test_load_and_evaluate_instances_separately(self, tmp_path):
        """Test loading and evaluating instances without aggregation."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        memory = MockMemory()
        
        test_data = [
            {"id": "test_1", "question": "Q1", "answer": "A1", "context": "User: Hello"},
            {"id": "test_2", "question": "Q2", "answer": "A2", "context": "User: World"},
        ]
        
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)
        
        # Load instances
        instances = adapter.load_dataset(str(data_file))
        assert len(instances) == 2
        
        # Evaluate each instance
        results = []
        for instance in instances:
            memory.clear()
            result = adapter.evaluate(instance, memory)
            results.append(result)
        
        # Verify results
        assert len(results) == 2
        assert all(r.latency_ms >= 0 for r in results)
        assert results[0].instance_id == "test_1"
        assert results[1].instance_id == "test_2"
    
    def test_evaluation_config_filtering(self, tmp_path):
        """Test that capability filter works during loading."""
        adapter = MemoryAgentBenchAdapter(seed=42)
        
        test_data = [
            {"id": "ar_1", "capability": "ar", "question": "Q1", "answer": "A1"},
            {"id": "ttl_1", "capability": "ttl", "question": "Q2", "answer": "A2"},
            {"id": "cr_1", "capability": "cr", "question": "Q3", "answer": "A3"},
        ]
        
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)
        
        # Load all instances
        all_instances = adapter.load_dataset(str(data_file))
        assert len(all_instances) == 3
        
        # Filter to AR only
        ar_instances = [
            i for i in all_instances 
            if i.capability == BenchmarkCapability.ACCURATE_RETRIEVAL
        ]
        assert len(ar_instances) == 1
        assert ar_instances[0].id == "ar_1"
