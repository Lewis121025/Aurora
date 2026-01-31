"""
AURORA Benchmark Interface
===========================

Unified evaluation interface for benchmarking the AURORA memory system.

This module provides:
    - BenchmarkCapability: Enum of evaluation dimensions
    - EvaluationMethod: Enum of evaluation methods
    - BenchmarkInstance: Data class for single evaluation instances
    - BenchmarkResult: Data class for evaluation results
    - EvaluationMetrics: Aggregated metrics across instances
    - BenchmarkAdapter: Abstract base class for benchmark adapters
    - AURORABenchmarkRunner: Main runner for executing benchmarks

Design Philosophy:
    - Adapters transform external benchmark formats to AURORA's interface
    - Results are collected with rich metadata for analysis
    - Metrics are computed separately, allowing flexible evaluation
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aurora.algorithms.aurora_core import AuroraMemory

from aurora.utils.time_utils import now_ts

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Model Detection
# =============================================================================

def _is_mock_embedder(embedder) -> bool:
    """Check if embedder is a mock/hash embedding (not for production benchmarks).
    
    Args:
        embedder: Embedding provider instance
        
    Returns:
        True if embedder is HashEmbedding or similar mock
    """
    if embedder is None:
        return True
    
    # Check class name
    class_name = type(embedder).__name__
    mock_names = {"HashEmbedding", "MockEmbedding", "FakeEmbedding", "DummyEmbedding"}
    if class_name in mock_names:
        return True
    
    # Check module path
    module = type(embedder).__module__
    if "hash" in module.lower() or "mock" in module.lower():
        return True
    
    return False


def _is_mock_llm(llm_provider) -> bool:
    """Check if LLM provider is a mock (not for production benchmarks).
    
    Args:
        llm_provider: LLM provider instance
        
    Returns:
        True if provider is MockLLM or similar
    """
    if llm_provider is None:
        return True
    
    # Check class name
    class_name = type(llm_provider).__name__
    mock_names = {"MockLLM", "FakeLLM", "DummyLLM", "MockProvider"}
    if class_name in mock_names:
        return True
    
    # Check module path
    module = type(llm_provider).__module__
    if "mock" in module.lower():
        return True
    
    return False


def verify_benchmark_ready(
    memory,
    llm=None,
    verbose: bool = True,
) -> Tuple[bool, List[str]]:
    """Verify if the configuration is ready for meaningful benchmark evaluation.
    
    Checks:
    1. Embedding model is not a mock (HashEmbedding)
    2. LLM provider is not a mock (MockLLM)
    
    Using mock models will result in significantly lower benchmark scores
    that do not reflect actual system performance.
    
    Args:
        memory: AuroraMemory instance to check
        llm: Optional LLM provider to check
        verbose: If True, print warnings to console
        
    Returns:
        Tuple of (is_ready, warnings):
            - is_ready: True if real models are configured
            - warnings: List of warning messages
            
    Example:
        from aurora.benchmark.interface import verify_benchmark_ready
        from aurora.algorithms.aurora_core import AuroraMemory
        
        memory = AuroraMemory(seed=42)
        is_ready, warnings = verify_benchmark_ready(memory)
        
        if not is_ready:
            print("⚠️ Configuration issues:")
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
            f"Embedding model is '{embedder_name}' (mock). "
            "Use a real embedding model (e.g., Bailian, Ark) for accurate benchmark results."
        )
    
    # Check LLM provider
    if llm is not None and _is_mock_llm(llm):
        is_ready = False
        llm_name = type(llm).__name__
        warnings.append(
            f"LLM provider is '{llm_name}' (mock). "
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
        print("  Mock models will result in significantly lower benchmark scores")
        print("  that do NOT reflect actual system performance!")
        print("=" * 70 + "\n")
    
    return is_ready, warnings


def _print_mock_warning(
    component: str,
    component_name: str,
    context: str = "benchmark evaluation",
) -> None:
    """Print a prominent warning about mock model usage.
    
    Args:
        component: Type of component ("embedding" or "llm")
        component_name: Name of the mock class
        context: Context where the warning is shown
    """
    logger.warning(
        f"⚠️ MOCK {component.upper()} DETECTED: Using '{component_name}' for {context}. "
        f"This will result in significantly lower scores that do NOT reflect actual performance!"
    )
    
    # Also print to console for visibility
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: Mock {component.title()} Model Detected")
    print(f"{'='*70}")
    print(f"  Component: {component_name}")
    print(f"  Context: {context}")
    print(f"")
    print(f"  Mock models produce pseudo-random or pattern-based outputs")
    print(f"  that will result in severely degraded benchmark scores.")
    print(f"")
    print(f"  For accurate results, please configure:")
    if component == "embedding":
        print(f"    - Bailian: Set BAILIAN_API_KEY and EMBEDDING_PROVIDER=bailian")
        print(f"    - Ark: Set ARK_API_KEY and EMBEDDING_PROVIDER=ark")
    else:
        print(f"    - Ark: Set ARK_API_KEY and LLM_PROVIDER=ark")
    print(f"{'='*70}\n")


# -----------------------------------------------------------------------------
# Capability Dimensions
# -----------------------------------------------------------------------------

class BenchmarkCapability(Enum):
    """
    Evaluation capability dimensions for memory systems.
    
    These capabilities map to specific aspects of memory performance:
    
    ACCURATE_RETRIEVAL:
        Ability to extract precise information from extended interaction history.
        Tests: factual recall, entity extraction, detail accuracy.
        AURORA: query() + FieldRetriever
    
    TEST_TIME_LEARNING:
        Ability to apply new rules during conversation without parameter updates.
        Tests: rule application, constraint following, preference adaptation.
        AURORA: ingest() + evolve()
    
    LONG_RANGE_UNDERSTANDING:
        Ability to form coherent summaries across extended narratives.
        Tests: summarization, theme extraction, narrative arc detection.
        AURORA: Story aggregation + Theme emergence
    
    CONFLICT_RESOLUTION:
        Ability to handle contradictory information updates appropriately.
        Tests: fact updates, preference changes, temporal reasoning.
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
        Create capability from string value.
        
        Args:
            s: String value (e.g., "accurate_retrieval")
        
        Returns:
            BenchmarkCapability enum member
        
        Raises:
            ValueError: If string doesn't match any capability
        """
        for cap in cls:
            if cap.value == s:
                return cap
        raise ValueError(f"Unknown capability: {s}")


class EvaluationMethod(Enum):
    """Methods for evaluating predictions against ground truth.
    
    EXACT_MATCH: Exact string match (case-insensitive)
    CONTAINS: Check if prediction contains ground truth
    FUZZY: Fuzzy string matching using edit distance
    LLM_JUDGE: LLM-as-judge evaluation
    ROUGE: ROUGE-based evaluation (for summaries)
    SEMANTIC: Semantic similarity-based evaluation
    """
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    FUZZY = "fuzzy"
    LLM_JUDGE = "llm_judge"
    ROUGE = "rouge"
    SEMANTIC = "semantic"


@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation.
    
    Attributes:
        use_llm_judge: Whether to use LLM-as-Judge for evaluation
        judge_model: Model to use for judging (if use_llm_judge=True)
        judge_temperature: Temperature for judge model
        max_retries: Maximum retries on failure
        timeout_s: Timeout per instance in seconds
        save_traces: Whether to save retrieval traces
        verbose: Whether to print progress
    """
    use_llm_judge: bool = True
    judge_model: str = "gpt-4"
    judge_temperature: float = 0.0
    
    max_retries: int = 3
    timeout_s: float = 60.0
    
    save_traces: bool = True
    verbose: bool = True
    
    # Batch settings
    batch_size: int = 1
    parallel_workers: int = 1
    
    # Filtering
    capabilities_filter: Optional[List["BenchmarkCapability"]] = None
    max_instances: Optional[int] = None


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics across multiple instances.
    
    Attributes:
        total_instances: Total number of instances evaluated
        correct_instances: Number of correct predictions
        accuracy: Overall accuracy [0, 1]
        
        avg_score: Average score across instances
        avg_latency_ms: Average latency in milliseconds
        avg_tokens: Average tokens used per query
        
        metrics_by_type: Metrics broken down by task/reasoning type
        p50_latency_ms: 50th percentile latency
        p99_latency_ms: 99th percentile latency
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
        """Serialize to JSON-compatible dict."""
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
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkInstance:
    """
    A single evaluation instance from a benchmark dataset.
    
    Represents one test case with context, query, and expected answer.
    Supports both simple string context (MemoryAgentBench) and 
    conversation history format (LOCOMO).
    
    Attributes:
        id: Unique identifier for this instance
        capability: The capability dimension being tested (optional for task_type based)
        context: Conversation history or context as string (for backward compat)
        query: The query to evaluate
        expected_answer: Expected answer (None for open-ended tasks)
        metadata: Additional benchmark-specific metadata
        
        # Extended fields for LOCOMO-style benchmarks
        task_type: Type of task (e.g., "qa", "summarization")
        conversation_history: Structured conversation history
        ground_truth: Expected answer (alias for expected_answer)
        reasoning_type: Type of reasoning required for QA
        session_id: Session identifier
        turn_number: Turn number in conversation
        created_ts: Creation timestamp
    
    Example (MemoryAgentBench style):
        instance = BenchmarkInstance(
            id="mab_001",
            capability=BenchmarkCapability.ACCURATE_RETRIEVAL,
            context="User: I live in San Francisco...",
            query="Where does the user live?",
            expected_answer="San Francisco",
        )
    
    Example (LOCOMO style):
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
    
    # For backward compatibility with MemoryAgentBench style
    capability: Optional[BenchmarkCapability] = None
    context: str = ""
    expected_answer: Optional[str] = None
    
    # Extended fields for LOCOMO-style benchmarks
    task_type: str = "qa"
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    ground_truth: str = ""
    reasoning_type: Optional[str] = None
    session_id: Optional[str] = None
    turn_number: Optional[int] = None
    created_ts: float = field(default_factory=now_ts)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Synchronize ground_truth and expected_answer."""
        if self.ground_truth and not self.expected_answer:
            self.expected_answer = self.ground_truth
        elif self.expected_answer and not self.ground_truth:
            self.ground_truth = self.expected_answer
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
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
        """Reconstruct from state dict."""
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
    Evaluation result for a single benchmark instance.
    
    Contains the predicted answer, expected answer, computed score,
    and performance metrics. Supports both MemoryAgentBench and LOCOMO formats.
    
    Attributes:
        instance_id: ID of the evaluated instance
        capability: The capability dimension that was tested (optional)
        predicted: Model's predicted answer (alias: prediction)
        expected: Expected answer (if available)
        score: Evaluation score in [0.0, 1.0]
        latency_ms: Response latency in milliseconds
        
        # Extended fields for LOCOMO-style benchmarks
        task_type: Type of task evaluated
        prediction: Model's prediction (alias for predicted)
        ground_truth: Expected answer (alias for expected)
        is_correct: Binary correctness flag
        tokens_used: Number of tokens consumed
        retrieval_count: Number of memory retrievals
        reasoning_trace: Trace of reasoning steps
        error_message: Error message if evaluation failed
        evaluated_ts: Evaluation timestamp
        
        metadata: Additional result metadata
    
    Score Interpretation:
        1.0: Perfect match / Fully correct
        0.5-0.99: Partially correct
        0.0: Incorrect / No match
    """
    
    instance_id: str
    score: float
    latency_ms: float = 0.0
    
    # For backward compatibility with MemoryAgentBench style
    capability: Optional[BenchmarkCapability] = None
    predicted: str = ""
    expected: Optional[str] = None
    
    # Extended fields for LOCOMO-style benchmarks
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
    
    def __post_init__(self):
        """Synchronize aliased fields."""
        if self.prediction and not self.predicted:
            self.predicted = self.prediction
        elif self.predicted and not self.prediction:
            self.prediction = self.predicted
        
        if self.ground_truth and not self.expected:
            self.expected = self.ground_truth
        elif self.expected and not self.ground_truth:
            self.ground_truth = self.expected or ""
        
        # Set is_correct if not explicitly set
        if not self.is_correct and self.score >= 0.5:
            self.is_correct = True
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
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
        """Reconstruct from state dict."""
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
        """Check if the result is considered correct."""
        return self.score >= threshold


# -----------------------------------------------------------------------------
# Protocol for Memory Interface
# -----------------------------------------------------------------------------

class MemoryProtocol(Protocol):
    """Protocol defining the memory interface expected by benchmarks."""
    
    def ingest(self, text: str, **kwargs: Any) -> Any:
        """Ingest text into memory."""
        ...
    
    def query(self, text: str, **kwargs: Any) -> Any:
        """Query memory for relevant information."""
        ...
    
    def evolve(self) -> None:
        """Trigger memory evolution/consolidation."""
        ...
    
    def clear(self) -> None:
        """Clear all memory state."""
        ...


# -----------------------------------------------------------------------------
# Abstract Adapter
# -----------------------------------------------------------------------------

class BenchmarkAdapter(ABC):
    """
    Abstract base class for benchmark adapters.
    
    Each benchmark (e.g., MemoryAgentBench, LOCOMO) should have its own adapter
    that transforms the benchmark's format to AURORA's interface.
    
    Subclasses must implement:
        - name: Property returning the benchmark name
        - load_dataset: Load instances from dataset path
        - evaluate: Evaluate a single instance against memory
        - aggregate_results: Compute aggregate metrics
    
    Example Implementation:
        class MemoryAgentBenchAdapter(BenchmarkAdapter):
            @property
            def name(self) -> str:
                return "MemoryAgentBench"
            
            def load_dataset(self, path: str) -> List[BenchmarkInstance]:
                # Load from HuggingFace or local path
                ...
    """
    
    def __init__(self, llm_provider=None, seed: int = 0):
        """Initialize the adapter.
        
        Args:
            llm_provider: Optional LLM provider for LLM-as-judge evaluation
            seed: Random seed for reproducibility
        """
        self.llm = llm_provider
        self._seed = seed
        self.rng = np.random.default_rng(seed)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the benchmark name."""
    
    @abstractmethod
    def load_dataset(self, path: str) -> List[BenchmarkInstance]:
        """
        Load evaluation instances from a dataset.
        
        Args:
            path: Path to the dataset (local path or HuggingFace identifier)
        
        Returns:
            List of BenchmarkInstance objects
        
        Raises:
            FileNotFoundError: If dataset path doesn't exist
            ValueError: If dataset format is invalid
        """
    
    @abstractmethod
    def evaluate(
        self,
        instance: BenchmarkInstance,
        memory: MemoryProtocol,
    ) -> BenchmarkResult:
        """
        Evaluate a single instance against the memory system.
        
        This method should:
        1. Prepare the memory state (ingest context if needed)
        2. Query the memory with the instance's query
        3. Compare the result with the expected answer
        4. Return a BenchmarkResult with score and metadata
        
        Args:
            instance: The benchmark instance to evaluate
            memory: The AURORA memory system instance
        
        Returns:
            BenchmarkResult with predicted answer, score, and latency
        """
    
    def aggregate_results(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, float]:
        """
        Aggregate evaluation results into summary metrics.
        
        Provides a default implementation that computes basic statistics.
        Subclasses can override for benchmark-specific metrics.
        
        Args:
            results: List of individual evaluation results
        
        Returns:
            Dict mapping metric names to values, e.g.:
            {
                "accuracy": 0.85,
                "accuracy_ar": 0.90,  # Accurate retrieval
                "accuracy_ttl": 0.80, # Test-time learning
                "mean_latency_ms": 45.2,
                "p95_latency_ms": 120.5,
            }
        """
        if not results:
            return {"accuracy": 0.0, "mean_latency_ms": 0.0}
        
        # Basic counts
        total = len(results)
        correct = sum(1 for r in results if r.is_correct or r.score >= 0.5)
        
        # Compute averages
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
        
        # Group by task_type for breakdown
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
                metrics[f"{task_type}_accuracy"] = type_correct / type_total if type_total > 0 else 0.0
                metrics[f"{task_type}_avg_score"] = float(np.mean(type_scores)) if type_scores else 0.0
        
        return metrics
    
    def prepare_memory(
        self,
        memory: MemoryProtocol,
        context: str,
        clear_first: bool = True,
    ) -> None:
        """
        Prepare memory state for evaluation.
        
        Default implementation clears memory and ingests context.
        Subclasses can override for benchmark-specific preparation.
        
        Args:
            memory: The memory system instance
            context: Context to ingest
            clear_first: Whether to clear memory before ingestion
        """
        if clear_first:
            memory.clear()
        
        # Split context into turns and ingest each
        turns = self._split_context(context)
        for turn in turns:
            if turn.strip():
                memory.ingest(turn)
        
        # Allow memory to consolidate
        memory.evolve()
    
    def _split_context(self, context: str) -> List[str]:
        """
        Split context into individual turns for ingestion.
        
        Default implementation splits by double newlines.
        Subclasses can override for benchmark-specific parsing.
        
        Args:
            context: Full context string
        
        Returns:
            List of individual turns
        """
        # Simple split by double newlines; subclasses may override
        return [turn.strip() for turn in context.split("\n\n") if turn.strip()]


# -----------------------------------------------------------------------------
# Benchmark Runner
# -----------------------------------------------------------------------------

class AURORABenchmarkRunner:
    """
    Main runner for executing benchmarks against AURORA memory.
    
    Coordinates benchmark adapters, manages execution, and collects results.
    
    Attributes:
        memory: The AURORA memory system instance
        adapters: Dict mapping benchmark names to adapters
    
    Example:
        from aurora import AuroraMemory
        from aurora.benchmark import AURORABenchmarkRunner
        from aurora.benchmark.adapters import MemoryAgentBenchAdapter
        
        memory = AuroraMemory(seed=42)
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
        Initialize the benchmark runner.
        
        Args:
            memory: AURORA memory system instance
            adapters: Dict mapping benchmark names to adapter instances
        """
        self.memory = memory
        self.adapters: Dict[str, BenchmarkAdapter] = adapters or {}
    
    def register_adapter(self, name: str, adapter: BenchmarkAdapter) -> None:
        """
        Register a new benchmark adapter.
        
        Args:
            name: Name to register the adapter under
            adapter: The adapter instance
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
        Run a specific benchmark.
        
        Args:
            benchmark_name: Name of the registered benchmark adapter
            dataset_path: Path to the benchmark dataset
            subset: Optional list of instance IDs to evaluate (for debugging)
            progress_callback: Optional callback(current, total) for progress reporting
        
        Returns:
            Dict containing:
                - "benchmark": Benchmark name
                - "dataset_path": Path to dataset
                - "num_instances": Number of instances evaluated
                - "results": List of BenchmarkResult dicts
                - "metrics": Aggregated metrics dict
        
        Raises:
            KeyError: If benchmark_name is not registered
            FileNotFoundError: If dataset_path doesn't exist
        """
        if benchmark_name not in self.adapters:
            raise KeyError(
                f"Unknown benchmark: {benchmark_name}. "
                f"Available: {list(self.adapters.keys())}"
            )
        
        adapter = self.adapters[benchmark_name]
        logger.info(f"Running benchmark '{adapter.name}' from {dataset_path}")
        
        # Load dataset
        instances = adapter.load_dataset(dataset_path)
        logger.info(f"Loaded {len(instances)} instances")
        
        # Filter to subset if specified
        if subset:
            subset_set = set(subset)
            instances = [i for i in instances if i.id in subset_set]
            logger.info(f"Filtered to {len(instances)} instances in subset")
        
        # Evaluate each instance
        results: List[BenchmarkResult] = []
        for idx, instance in enumerate(instances):
            try:
                result = adapter.evaluate(instance, self.memory)
                results.append(result)
                
                if progress_callback:
                    progress_callback(idx + 1, len(instances))
                    
            except Exception as e:
                logger.error(f"Error evaluating instance {instance.id}: {e}")
                # Record failed evaluation
                results.append(BenchmarkResult(
                    instance_id=instance.id,
                    capability=instance.capability,
                    predicted="",
                    expected=instance.expected_answer,
                    score=0.0,
                    latency_ms=0.0,
                    metadata={"error": str(e)},
                ))
        
        # Aggregate results
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
        Run all registered benchmarks.
        
        Args:
            datasets: Dict mapping benchmark names to dataset paths
            progress_callback: Optional callback(benchmark, current, total)
        
        Returns:
            Dict mapping benchmark names to their results
        
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
        Generate a summary report from benchmark results.
        
        Args:
            results: Results from run_all() or multiple run_benchmark() calls
            format: Output format ("markdown" or "text")
        
        Returns:
            Formatted report string
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
            
            # Detailed capability breakdown
            lines.append("\n## Capability Breakdown\n")
            for name, data in results.items():
                if "error" in data:
                    continue
                    
                metrics = data.get("metrics", {})
                lines.append(f"### {name}\n")
                
                for cap in BenchmarkCapability:
                    key = f"accuracy_{cap.value[:3]}"  # e.g., accuracy_acc
                    if key in metrics:
                        lines.append(f"- **{cap.value}**: {metrics[key]:.2%}")
                
                lines.append("")
        else:
            # Plain text format
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
# Evaluation Helpers
# =============================================================================

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Exact string match evaluation (case-insensitive)."""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def contains_score(prediction: str, ground_truth: str) -> float:
    """Check if prediction contains ground truth."""
    return 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0


def fuzzy_match_score(prediction: str, ground_truth: str, threshold: float = 0.8) -> float:
    """Fuzzy string matching using edit distance ratio."""
    from difflib import SequenceMatcher
    
    pred_lower = prediction.strip().lower()
    truth_lower = ground_truth.strip().lower()
    
    ratio = SequenceMatcher(None, pred_lower, truth_lower).ratio()
    return ratio if ratio >= threshold else 0.0


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
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
