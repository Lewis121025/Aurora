"""
MemoryAgentBench Adapter
========================

Adapter for MemoryAgentBench (2025.07) - Academic benchmark for LLM Agent memory systems.

Dataset: HuggingFace `ai-hyz/MemoryAgentBench`
- 146 evaluation instances total:
  - 22 Accurate Retrieval (AR)
  - 6 Test-Time Learning (TTL)
  - 110 Long-Range Understanding (LRU)
  - 8 Conflict Resolution (CR)

AURORA Capability Mapping:
| Benchmark Capability | AURORA Implementation |
|---------------------|----------------------|
| Accurate Retrieval  | query() + FieldRetriever |
| Test-Time Learning  | ingest() + evolve() |
| Long-Range Understanding | Story aggregation + Theme emergence + NarratorEngine |
| Conflict Resolution | TensionManager + CoherenceGuardian |

Usage:
    from aurora.benchmark.adapters import MemoryAgentBenchAdapter
    from aurora.algorithms.aurora_core import AuroraMemory
    
    adapter = MemoryAgentBenchAdapter(llm_provider=llm)
    memory = AuroraMemory()
    
    results, metrics = adapter.run_benchmark(
        dataset_path="ai-hyz/MemoryAgentBench",
        memory=memory,
    )
    
    print(metrics.accuracy)
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from aurora.benchmark.interface import (
    BenchmarkAdapter,
    BenchmarkCapability,
    BenchmarkInstance,
    BenchmarkResult,
    EvaluationConfig,
    EvaluationMetrics,
    MemoryProtocol,
)
from aurora.utils.time_utils import now_ts

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Capability type mapping from dataset
CAPABILITY_MAPPING = {
    "accurate_retrieval": BenchmarkCapability.ACCURATE_RETRIEVAL,
    "ar": BenchmarkCapability.ACCURATE_RETRIEVAL,
    "test_time_learning": BenchmarkCapability.TEST_TIME_LEARNING,
    "ttl": BenchmarkCapability.TEST_TIME_LEARNING,
    "long_range_understanding": BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
    "lru": BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
    "conflict_resolution": BenchmarkCapability.CONFLICT_RESOLUTION,
    "cr": BenchmarkCapability.CONFLICT_RESOLUTION,
}

# Task type strings for compatibility with base interface
TASK_TYPE_AR = "accurate_retrieval"
TASK_TYPE_TTL = "test_time_learning"
TASK_TYPE_LRU = "long_range_understanding"
TASK_TYPE_CR = "conflict_resolution"

# Conversation turn markers
USER_MARKER = "User:"
ASSISTANT_MARKER = "Assistant:"
SYSTEM_MARKER = "System:"

# Evaluation prompts
LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for memory system benchmarks.
Your task is to determine if the predicted answer correctly answers the question based on the expected answer.

Consider:
1. Semantic equivalence - different wording but same meaning is acceptable
2. Partial credit - for partially correct answers
3. Factual accuracy - verify key facts match

Respond with a JSON object:
{
    "is_correct": true/false,
    "score": 0.0-1.0,
    "reasoning": "explanation"
}
"""

LLM_JUDGE_USER_TEMPLATE = """Question: {question}

Expected Answer: {expected_answer}

Predicted Answer: {predicted_answer}

Please evaluate if the predicted answer is correct."""


# =============================================================================
# Helper Functions
# =============================================================================

def parse_conversation_turns(context: str) -> List[Dict[str, Any]]:
    """Parse conversation context into structured turns.
    
    Handles various formats:
    - "User: message\\nAssistant: response"
    - "Human: message\\nAI: response"
    - Multi-line messages
    
    Args:
        context: Raw conversation text
        
    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dicts
    """
    turns: List[Dict[str, Any]] = []
    
    # Normalize markers
    normalized = context.replace("Human:", "User:")
    normalized = normalized.replace("AI:", "Assistant:")
    normalized = normalized.replace("Bot:", "Assistant:")
    
    # Split by role markers
    pattern = r"(User:|Assistant:|System:)"
    parts = re.split(pattern, normalized)
    
    current_role = None
    current_content: List[str] = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part in ("User:", "Assistant:", "System:"):
            # Save previous turn
            if current_role and current_content:
                role = "user" if current_role == "User:" else (
                    "assistant" if current_role == "Assistant:" else "system"
                )
                turns.append({
                    "role": role,
                    "content": " ".join(current_content).strip(),
                    "speaker": role,
                    "text": " ".join(current_content).strip(),
                })
            
            current_role = part
            current_content = []
        else:
            current_content.append(part)
    
    # Don't forget the last turn
    if current_role and current_content:
        role = "user" if current_role == "User:" else (
            "assistant" if current_role == "Assistant:" else "system"
        )
        turns.append({
            "role": role,
            "content": " ".join(current_content).strip(),
            "speaker": role,
            "text": " ".join(current_content).strip(),
        })
    
    # If no structured format found, treat entire context as user message
    if not turns:
        turns.append({
            "role": "user",
            "content": context.strip(),
            "speaker": "user",
            "text": context.strip(),
        })
    
    return turns


def extract_conflicting_facts(context: str, metadata: Dict[str, Any]) -> List[str]:
    """Extract conflicting facts from context for CR capability.
    
    Args:
        context: Conversation context
        metadata: Instance metadata that may contain conflict info
        
    Returns:
        List of conflicting fact strings
    """
    facts: List[str] = []
    
    # Check metadata first
    if "conflicts" in metadata:
        facts.extend(metadata["conflicts"])
    
    if "old_fact" in metadata and "new_fact" in metadata:
        facts.append(f"Old: {metadata['old_fact']}")
        facts.append(f"New: {metadata['new_fact']}")
    
    # Pattern-based extraction
    conflict_patterns = [
        r"(?:originally|initially|before)[:\s]+(.+?)(?:,|\.|\n|but)",
        r"(?:now|currently|updated|changed to)[:\s]+(.+?)(?:,|\.|\n)",
        r"(?:correction|update|change)[:\s]+(.+?)(?:,|\.|\n)",
    ]
    
    for pattern in conflict_patterns:
        matches = re.findall(pattern, context, re.IGNORECASE)
        facts.extend(matches)
    
    return facts


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized string
    """
    # Lowercase
    normalized = answer.lower().strip()
    
    # Remove common prefixes
    prefixes = [
        "the answer is",
        "answer:",
        "response:",
        "i think",
        "based on the context,",
    ]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
    
    # Remove punctuation at edges
    normalized = normalized.strip(".,!?;:")
    
    return normalized


def exact_match_score(predicted: str, expected: str) -> Tuple[bool, float]:
    """Compute exact match score.
    
    Args:
        predicted: Predicted answer
        expected: Expected answer
        
    Returns:
        Tuple of (is_correct, score)
    """
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    # Exact match
    if pred_norm == exp_norm:
        return True, 1.0
    
    # Substring match (predicted contains expected)
    if exp_norm in pred_norm:
        return True, 0.9
    
    # Substring match (expected contains predicted)
    if pred_norm in exp_norm:
        return True, 0.8
    
    # Word overlap
    pred_words = set(pred_norm.split())
    exp_words = set(exp_norm.split())
    
    if pred_words and exp_words:
        overlap = len(pred_words & exp_words)
        max_len = max(len(pred_words), len(exp_words))
        overlap_score = overlap / max_len
        
        if overlap_score > 0.7:
            return True, overlap_score
        elif overlap_score > 0.3:
            return False, overlap_score
    
    return False, 0.0


def capability_to_task_type(capability: BenchmarkCapability) -> str:
    """Convert BenchmarkCapability to task_type string."""
    mapping = {
        BenchmarkCapability.ACCURATE_RETRIEVAL: TASK_TYPE_AR,
        BenchmarkCapability.TEST_TIME_LEARNING: TASK_TYPE_TTL,
        BenchmarkCapability.LONG_RANGE_UNDERSTANDING: TASK_TYPE_LRU,
        BenchmarkCapability.CONFLICT_RESOLUTION: TASK_TYPE_CR,
    }
    return mapping.get(capability, TASK_TYPE_AR)


def task_type_to_capability(task_type: str) -> BenchmarkCapability:
    """Convert task_type string to BenchmarkCapability."""
    return CAPABILITY_MAPPING.get(task_type.lower(), BenchmarkCapability.ACCURATE_RETRIEVAL)


# =============================================================================
# MemoryAgentBench Adapter
# =============================================================================

class MemoryAgentBenchAdapter(BenchmarkAdapter):
    """MemoryAgentBench (2025.07) adapter for AURORA evaluation.
    
    Evaluates four core memory capabilities:
    1. Accurate Retrieval (AR) - Extracting precise information
    2. Test-Time Learning (TTL) - Applying new rules without retraining
    3. Long-Range Understanding (LRU) - Coherent summarization
    4. Conflict Resolution (CR) - Handling contradictory updates
    
    Attributes:
        llm: LLM provider for evaluation (optional, for LLM-as-Judge)
        embedder: Embedding function (optional, uses memory's embedder if not provided)
    """
    
    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
        seed: int = 0,
    ):
        """Initialize the adapter.
        
        Args:
            llm_provider: LLM provider implementing complete_json() for judging
            embedder: Embedding function for similarity computation
            seed: Random seed for reproducibility
        """
        super().__init__(llm_provider=llm_provider, seed=seed)
        self.embedder = embedder
        self._instances_cache: Dict[str, List[BenchmarkInstance]] = {}
        self._config: Optional[EvaluationConfig] = None
    
    @property
    def name(self) -> str:
        """Benchmark name."""
        return "MemoryAgentBench"
    
    @property
    def capabilities(self) -> List[BenchmarkCapability]:
        """Supported capabilities."""
        return [
            BenchmarkCapability.ACCURATE_RETRIEVAL,
            BenchmarkCapability.TEST_TIME_LEARNING,
            BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
            BenchmarkCapability.CONFLICT_RESOLUTION,
        ]
    
    # -------------------------------------------------------------------------
    # Dataset Loading
    # -------------------------------------------------------------------------
    
    def load_dataset(
        self,
        path: str,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BenchmarkInstance]:
        """Load MemoryAgentBench dataset.
        
        Supports:
        - HuggingFace: "ai-hyz/MemoryAgentBench"
        - Local JSON: "/path/to/dataset.json"
        - Local directory: "/path/to/dataset/" (with split files)
        
        Args:
            path: Dataset source (HuggingFace ID or local path)
            subset: Dataset split (default: "test")
            limit: Maximum number of instances to load
                
        Returns:
            List of BenchmarkInstance objects
        """
        split = subset or "test"
        cache_key = f"{path}:{split}"
        
        # Check cache
        if cache_key in self._instances_cache:
            instances = self._instances_cache[cache_key]
            if limit:
                return instances[:limit]
            return instances
        
        instances: List[BenchmarkInstance] = []
        
        # Try HuggingFace first
        if not path.startswith("/") and not path.startswith("./"):
            instances = self._load_from_huggingface(path, split)
        
        # Fall back to local file
        if not instances:
            instances = self._load_from_local(path, split)
        
        # Cache results
        if instances:
            self._instances_cache[cache_key] = instances
        
        logger.info(f"Loaded {len(instances)} instances from {path} ({split})")
        
        if limit:
            return instances[:limit]
        return instances
    
    def _load_from_huggingface(
        self,
        dataset_id: str,
        split: str,
    ) -> List[BenchmarkInstance]:
        """Load dataset from HuggingFace Hub.
        
        Args:
            dataset_id: HuggingFace dataset identifier
            split: Dataset split
            
        Returns:
            List of BenchmarkInstance objects
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )
            return []
        
        try:
            dataset = load_dataset(dataset_id, split=split)
            instances = []
            
            for idx, item in enumerate(dataset):
                instance = self._parse_hf_item(item, idx)
                if instance:
                    instances.append(instance)
            
            return instances
            
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}")
            return []
    
    def _load_from_local(
        self,
        path: str,
        split: str,
    ) -> List[BenchmarkInstance]:
        """Load dataset from local file or directory.
        
        Args:
            path: Local path to JSON file or directory
            split: Dataset split (used for directory structure)
            
        Returns:
            List of BenchmarkInstance objects
        """
        path_obj = Path(path)
        instances: List[BenchmarkInstance] = []
        
        if path_obj.is_file():
            # Single JSON file
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            items = data if isinstance(data, list) else data.get("data", [])
            for idx, item in enumerate(items):
                instance = self._parse_local_item(item, idx)
                if instance:
                    instances.append(instance)
                    
        elif path_obj.is_dir():
            # Directory with split files
            split_file = path_obj / f"{split}.json"
            if split_file.exists():
                return self._load_from_local(str(split_file), split)
            
            # Try loading all JSON files
            for json_file in path_obj.glob("*.json"):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                items = data if isinstance(data, list) else data.get("data", [])
                for idx, item in enumerate(items):
                    instance = self._parse_local_item(
                        item, len(instances) + idx
                    )
                    if instance:
                        instances.append(instance)
        
        return instances
    
    def _parse_hf_item(
        self,
        item: Dict[str, Any],
        idx: int,
    ) -> Optional[BenchmarkInstance]:
        """Parse HuggingFace dataset item into BenchmarkInstance.
        
        Expected fields (flexible):
        - capability/type: Capability type string
        - context/conversation/history: Conversation context
        - question/query: The question to answer
        - answer/expected_answer/ground_truth: Expected answer
        
        Args:
            item: Raw dataset item
            idx: Item index for ID generation
            
        Returns:
            BenchmarkInstance or None if parsing fails
        """
        try:
            # Extract capability
            cap_str = item.get("capability") or item.get("type") or "ar"
            capability = CAPABILITY_MAPPING.get(
                cap_str.lower(),
                BenchmarkCapability.ACCURATE_RETRIEVAL,
            )
            task_type = capability_to_task_type(capability)
            
            # Extract context
            context = (
                item.get("context") or
                item.get("conversation") or
                item.get("history") or
                ""
            )
            
            # Extract question
            question = (
                item.get("question") or
                item.get("query") or
                item.get("input") or
                ""
            )
            
            # Extract expected answer
            expected = (
                item.get("answer") or
                item.get("expected_answer") or
                item.get("ground_truth") or
                item.get("output") or
                ""
            )
            
            if not question or not expected:
                return None
            
            # Parse turns from context
            turns = parse_conversation_turns(context) if context else []
            
            # Extract conflicting facts for CR
            metadata = item.get("metadata", {})
            conflicting_facts = extract_conflicting_facts(context, metadata)
            
            # Add capability info to metadata
            metadata["capability"] = capability.value
            metadata["conflicting_facts"] = conflicting_facts
            metadata["raw_item"] = item
            
            return BenchmarkInstance(
                id=item.get("id", f"mab_{idx}"),
                query=question,
                capability=capability,
                context=context,
                expected_answer=expected,
                task_type=task_type,
                conversation_history=turns,
                ground_truth=expected,
                metadata=metadata,
                reasoning_type=capability.value,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse item {idx}: {e}")
            return None
    
    def _parse_local_item(
        self,
        item: Dict[str, Any],
        idx: int,
    ) -> Optional[BenchmarkInstance]:
        """Parse local JSON item (same logic as HuggingFace).
        
        Args:
            item: Raw JSON item
            idx: Item index
            
        Returns:
            BenchmarkInstance or None
        """
        return self._parse_hf_item(item, idx)
    
    # -------------------------------------------------------------------------
    # Memory Preparation
    # -------------------------------------------------------------------------
    
    def _prepare_memory_for_instance(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> None:
        """Prepare memory state by ingesting conversation history.
        
        Args:
            instance: Benchmark instance with conversation history
            memory: AURORA memory instance (AuroraMemory or AuroraTenant)
        """
        turns = instance.conversation_history
        
        # Check memory type and call appropriate method
        has_ingest_interaction = hasattr(memory, "ingest_interaction")
        has_ingest = hasattr(memory, "ingest")
        
        for i, turn in enumerate(turns):
            event_id = f"bench_{uuid.uuid4().hex[:8]}_{i}"
            content = turn.get("content") or turn.get("text", "")
            role = turn.get("role") or turn.get("speaker", "user")
            
            if has_ingest_interaction:
                # AuroraTenant style
                if role == "user":
                    # Look for next assistant response
                    assistant_response = ""
                    if i + 1 < len(turns):
                        next_turn = turns[i + 1]
                        next_role = next_turn.get("role") or next_turn.get("speaker", "")
                        if next_role == "assistant":
                            assistant_response = next_turn.get("content") or next_turn.get("text", "")
                    
                    memory.ingest_interaction(
                        event_id=event_id,
                        session_id="benchmark_session",
                        user_message=content,
                        agent_message=assistant_response,
                    )
                    
            elif has_ingest:
                # AuroraMemory style
                if role == "user":
                    assistant_response = ""
                    if i + 1 < len(turns):
                        next_turn = turns[i + 1]
                        next_role = next_turn.get("role") or next_turn.get("speaker", "")
                        if next_role == "assistant":
                            assistant_response = next_turn.get("content") or next_turn.get("text", "")
                    
                    interaction_text = f"User: {content}"
                    if assistant_response:
                        interaction_text += f"\nAssistant: {assistant_response}"
                    
                    memory.ingest(
                        interaction_text=interaction_text,
                        event_id=event_id,
                    )
        
        # Trigger evolution if available
        if hasattr(memory, "evolve"):
            memory.evolve()
    
    # -------------------------------------------------------------------------
    # Capability-Specific Evaluation
    # -------------------------------------------------------------------------
    
    def _evaluate_ar(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> BenchmarkResult:
        """Evaluate Accurate Retrieval capability.
        
        Tests the system's ability to extract precise information from
        extended interaction history.
        
        AURORA Implementation:
        - Uses query() with FieldRetriever
        - Evaluates retrieval precision
        
        Args:
            instance: Benchmark instance
            memory: AURORA memory instance
            
        Returns:
            BenchmarkResult
        """
        start_time = time.time()
        
        # Query memory
        query_result = self._query_memory(memory, instance.query)
        predicted_answer = self._extract_answer_from_retrieval(
            query_result,
            instance.query,
            memory,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Evaluate answer
        is_correct, score = self._evaluate_answer(
            predicted_answer,
            instance.ground_truth,
        )
        
        return BenchmarkResult(
            instance_id=instance.id,
            capability=instance.capability,
            predicted=predicted_answer,
            expected=instance.expected_answer,
            score=score,
            latency_ms=latency_ms,
            task_type=instance.task_type,
            prediction=predicted_answer,
            ground_truth=instance.ground_truth,
            is_correct=is_correct,
            retrieval_count=len(query_result.get("ranked", [])),
            metadata={"query_result": query_result} if self._config and self._config.save_traces else {},
        )
    
    def _evaluate_ttl(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> BenchmarkResult:
        """Evaluate Test-Time Learning capability.
        
        Tests the system's ability to apply newly learned rules without
        parameter updates.
        
        AURORA Implementation:
        - ingest() learns rules from context
        - query() applies rules to answer questions
        - Evolution consolidates learned rules
        
        Args:
            instance: Benchmark instance
            memory: AURORA memory instance
            
        Returns:
            BenchmarkResult
        """
        start_time = time.time()
        
        # The context should contain rules to learn
        # These were already ingested in _prepare_memory
        # Now query to apply the rules
        
        query_result = self._query_memory(memory, instance.query)
        
        # For TTL, we may need to generate an answer using the rules
        predicted_answer = self._generate_ttl_answer(
            query_result,
            instance.query,
            memory,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Evaluate
        is_correct, score = self._evaluate_answer(
            predicted_answer,
            instance.ground_truth,
        )
        
        return BenchmarkResult(
            instance_id=instance.id,
            capability=instance.capability,
            predicted=predicted_answer,
            expected=instance.expected_answer,
            score=score,
            latency_ms=latency_ms,
            task_type=instance.task_type,
            prediction=predicted_answer,
            ground_truth=instance.ground_truth,
            is_correct=is_correct,
            retrieval_count=len(query_result.get("ranked", [])),
        )
    
    def _evaluate_lru(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> BenchmarkResult:
        """Evaluate Long-Range Understanding capability.
        
        Tests the system's ability to form coherent summaries across
        extended narratives.
        
        AURORA Implementation:
        - Story aggregation organizes related plots
        - Theme emergence captures patterns
        - NarratorEngine reconstructs coherent narratives
        
        Args:
            instance: Benchmark instance
            memory: AURORA memory instance
            
        Returns:
            BenchmarkResult
        """
        start_time = time.time()
        
        # Query for relevant content
        query_result = self._query_memory(memory, instance.query, k=15)
        
        # Generate summary using narrative reconstruction
        predicted_answer = self._generate_lru_summary(
            query_result,
            instance.query,
            memory,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # LRU typically needs LLM-as-Judge for proper evaluation
        # since summaries can be semantically equivalent but worded differently
        is_correct, score = self._evaluate_answer(
            predicted_answer,
            instance.ground_truth,
        )
        
        return BenchmarkResult(
            instance_id=instance.id,
            capability=instance.capability,
            predicted=predicted_answer,
            expected=instance.expected_answer,
            score=score,
            latency_ms=latency_ms,
            task_type=instance.task_type,
            prediction=predicted_answer,
            ground_truth=instance.ground_truth,
            is_correct=is_correct,
            retrieval_count=len(query_result.get("ranked", [])),
        )
    
    def _evaluate_cr(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> BenchmarkResult:
        """Evaluate Conflict Resolution capability.
        
        Tests the system's ability to handle contradictory information
        and produce the most current/correct answer.
        
        AURORA Implementation:
        - TensionManager detects and classifies conflicts
        - CoherenceGuardian maintains consistency
        - Smart resolution decides what to preserve vs resolve
        
        Args:
            instance: Benchmark instance
            memory: AURORA memory instance
            
        Returns:
            BenchmarkResult
        """
        start_time = time.time()
        
        # Get conflicting facts from metadata
        conflicting_facts = instance.metadata.get("conflicting_facts", [])
        
        # Query with awareness of conflicts
        query_result = self._query_memory(memory, instance.query)
        
        # Generate answer with conflict resolution
        predicted_answer = self._generate_cr_answer(
            query_result,
            instance.query,
            conflicting_facts,
            memory,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Evaluate
        is_correct, score = self._evaluate_answer(
            predicted_answer,
            instance.ground_truth,
        )
        
        return BenchmarkResult(
            instance_id=instance.id,
            capability=instance.capability,
            predicted=predicted_answer,
            expected=instance.expected_answer,
            score=score,
            latency_ms=latency_ms,
            task_type=instance.task_type,
            prediction=predicted_answer,
            ground_truth=instance.ground_truth,
            is_correct=is_correct,
            retrieval_count=len(query_result.get("ranked", [])),
        )
    
    # -------------------------------------------------------------------------
    # Main Evaluation Entry Point
    # -------------------------------------------------------------------------
    
    def evaluate(
        self,
        instance: BenchmarkInstance,
        memory: Any,
        **kwargs,
    ) -> BenchmarkResult:
        """Evaluate a single benchmark instance.
        
        Routes to capability-specific evaluation methods.
        
        Args:
            instance: The benchmark instance to evaluate
            memory: AURORA memory instance
            **kwargs: Additional options (e.g., config)
            
        Returns:
            BenchmarkResult with evaluation metrics
        """
        # Store config if provided
        self._config = kwargs.get("config")
        
        # Prepare memory with conversation history
        self._prepare_memory_for_instance(instance, memory)
        
        # Get capability from task_type or reasoning_type
        capability = task_type_to_capability(
            instance.reasoning_type or instance.task_type
        )
        
        # Route to capability-specific evaluation
        if capability == BenchmarkCapability.ACCURATE_RETRIEVAL:
            return self._evaluate_ar(instance, memory)
        elif capability == BenchmarkCapability.TEST_TIME_LEARNING:
            return self._evaluate_ttl(instance, memory)
        elif capability == BenchmarkCapability.LONG_RANGE_UNDERSTANDING:
            return self._evaluate_lru(instance, memory)
        elif capability == BenchmarkCapability.CONFLICT_RESOLUTION:
            return self._evaluate_cr(instance, memory)
        else:
            # Default to AR for unknown capabilities
            return self._evaluate_ar(instance, memory)
    
    # -------------------------------------------------------------------------
    # Results Aggregation (Override for capability-specific metrics)
    # -------------------------------------------------------------------------
    
    def aggregate_results(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, float]:
        """Aggregate evaluation results into summary metrics.
        
        Overrides base method to add capability-specific metrics.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dict mapping metric names to values
        """
        # Use base implementation for basic metrics
        metrics = super().aggregate_results(results)
        
        # Add capability-specific metrics
        for cap in [BenchmarkCapability.ACCURATE_RETRIEVAL, BenchmarkCapability.TEST_TIME_LEARNING,
                    BenchmarkCapability.LONG_RANGE_UNDERSTANDING, BenchmarkCapability.CONFLICT_RESOLUTION]:
            task_type = capability_to_task_type(cap)
            cap_results = [r for r in results if r.task_type == task_type or 
                          (r.capability and r.capability == cap)]
            
            if cap_results:
                cap_total = len(cap_results)
                cap_correct = sum(1 for r in cap_results if r.is_correct or r.score >= 0.5)
                cap_scores = [r.score for r in cap_results]
                
                # Use short capability names for keys
                cap_short = cap.value[:3]  # e.g., "acc", "tes", "lon", "con"
                metrics[f"accuracy_{cap_short}"] = cap_correct / cap_total if cap_total > 0 else 0.0
                metrics[f"count_{cap_short}"] = float(cap_total)
                metrics[f"avg_score_{cap_short}"] = float(np.mean(cap_scores)) if cap_scores else 0.0
        
        # Store for get_evaluation_metrics()
        self._last_metrics = metrics
        self._last_results = results
        
        return metrics
    
    def get_evaluation_metrics(self, results: Optional[List[BenchmarkResult]] = None) -> EvaluationMetrics:
        """Get EvaluationMetrics object from results.
        
        Args:
            results: List of results (uses last evaluated if None)
            
        Returns:
            EvaluationMetrics object with detailed breakdown
        """
        if results is None:
            results = getattr(self, '_last_results', [])
        
        if not results:
            return EvaluationMetrics()
        
        # Aggregate if not already done
        metrics_dict = self.aggregate_results(results)
        
        # Build metrics_by_type
        metrics_by_type: Dict[str, Dict[str, float]] = {}
        
        for cap in [BenchmarkCapability.ACCURATE_RETRIEVAL, BenchmarkCapability.TEST_TIME_LEARNING,
                    BenchmarkCapability.LONG_RANGE_UNDERSTANDING, BenchmarkCapability.CONFLICT_RESOLUTION]:
            task_type = capability_to_task_type(cap)
            cap_short = cap.value[:3]
            
            if f"accuracy_{cap_short}" in metrics_dict:
                metrics_by_type[task_type] = {
                    "accuracy": metrics_dict.get(f"accuracy_{cap_short}", 0.0),
                    "count": metrics_dict.get(f"count_{cap_short}", 0.0),
                    "avg_score": metrics_dict.get(f"avg_score_{cap_short}", 0.0),
                }
        
        return EvaluationMetrics(
            total_instances=int(metrics_dict.get("total_instances", 0)),
            correct_instances=int(metrics_dict.get("correct_instances", 0)),
            accuracy=metrics_dict.get("accuracy", 0.0),
            avg_score=metrics_dict.get("avg_score", 0.0),
            avg_latency_ms=metrics_dict.get("mean_latency_ms", 0.0),
            metrics_by_type=metrics_by_type,
            p50_latency_ms=metrics_dict.get("p50_latency_ms", 0.0),
            p99_latency_ms=metrics_dict.get("p99_latency_ms", 0.0),
        )
    
    # -------------------------------------------------------------------------
    # Convenience Method for Config-Based Benchmark Run
    # -------------------------------------------------------------------------
    
    def run_benchmark_with_config(
        self,
        memory: Any,
        source: str,
        config: Optional[EvaluationConfig] = None,
        split: str = "test",
    ) -> Tuple[List[BenchmarkResult], EvaluationMetrics]:
        """Run full benchmark evaluation with configuration.
        
        Args:
            memory: AURORA memory instance
            source: Dataset source
            config: Evaluation configuration
            split: Dataset split
            
        Returns:
            Tuple of (individual results, aggregate metrics)
        """
        config = config or EvaluationConfig()
        self._config = config
        
        # Load dataset
        instances = self.load_dataset(source, subset=split)
        
        # Apply filters
        if config.capabilities_filter:
            filter_types = [capability_to_task_type(c) for c in config.capabilities_filter]
            instances = [
                i for i in instances
                if i.task_type in filter_types or 
                   i.reasoning_type in [c.value for c in config.capabilities_filter]
            ]
        
        if config.max_instances:
            instances = instances[:config.max_instances]
        
        # Evaluate instances
        results: List[BenchmarkResult] = []
        
        for i, instance in enumerate(instances):
            if config.verbose:
                capability = task_type_to_capability(instance.reasoning_type or instance.task_type)
                print(f"[{i+1}/{len(instances)}] Evaluating {instance.id} ({capability.value})")
            
            try:
                result = self.evaluate(instance, memory, config=config)
                results.append(result)
                
                if config.verbose:
                    status = "✓" if result.is_correct else "✗"
                    print(f"  {status} Score: {result.score:.2f}, Latency: {result.latency_ms:.0f}ms")
                    
            except Exception as e:
                error_result = BenchmarkResult(
                    instance_id=instance.id,
                    task_type=instance.task_type,
                    prediction="",
                    ground_truth=instance.ground_truth,
                    score=0.0,
                    is_correct=False,
                    error_message=str(e),
                )
                results.append(error_result)
                
                if config.verbose:
                    print(f"  ✗ Error: {e}")
        
        # Aggregate results and get EvaluationMetrics
        metrics = self.get_evaluation_metrics(results)
        
        return results, metrics
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _query_memory(
        self,
        memory: Any,
        query_text: str,
        k: int = 8,
    ) -> Dict[str, Any]:
        """Query AURORA memory and return results.
        
        Handles both AuroraMemory and AuroraTenant interfaces.
        
        Args:
            memory: AURORA memory instance
            query_text: Query string
            k: Number of results to retrieve
            
        Returns:
            Dictionary with query results
        """
        result = {}
        
        if hasattr(memory, "query"):
            trace = memory.query(text=query_text, k=k)
            
            # Extract results based on trace type
            if hasattr(trace, "ranked"):
                result["ranked"] = trace.ranked
            if hasattr(trace, "attractor_path"):
                result["attractor_path"] = trace.attractor_path
            if hasattr(trace, "query_emb"):
                result["query_emb"] = trace.query_emb
                
            result["raw_trace"] = trace
        
        return result
    
    def _extract_answer_from_retrieval(
        self,
        query_result: Dict[str, Any],
        question: str,
        memory: Any,
    ) -> str:
        """Extract answer from retrieval results.
        
        For AR, we extract the most relevant content from retrieved plots.
        
        Args:
            query_result: Query results from memory
            question: Original question
            memory: AURORA memory instance
            
        Returns:
            Extracted answer string
        """
        ranked = query_result.get("ranked", [])
        
        if not ranked:
            return "No relevant information found."
        
        # Get top results
        answers: List[str] = []
        
        for item in ranked[:3]:
            # Handle different result formats
            if isinstance(item, tuple):
                node_id, score, kind = item
            else:
                node_id = item.get("id", "")
                score = item.get("score", 0.0)
                kind = item.get("kind", "plot")
            
            # Get content from memory
            content = self._get_node_content(memory, node_id, kind)
            if content:
                answers.append(content)
        
        if not answers:
            return "No relevant information found."
        
        # For simple AR, return the most relevant content
        # In production, this would be refined with LLM
        return answers[0]
    
    def _generate_ttl_answer(
        self,
        query_result: Dict[str, Any],
        question: str,
        memory: Any,
    ) -> str:
        """Generate answer for TTL by applying learned rules.
        
        Args:
            query_result: Query results
            question: Question to answer
            memory: AURORA memory
            
        Returns:
            Generated answer
        """
        # Similar to AR but focused on rule application
        return self._extract_answer_from_retrieval(query_result, question, memory)
    
    def _generate_lru_summary(
        self,
        query_result: Dict[str, Any],
        question: str,
        memory: Any,
    ) -> str:
        """Generate summary for LRU using narrative reconstruction.
        
        Uses NarratorEngine if available.
        
        Args:
            query_result: Query results
            question: Question/topic for summary
            memory: AURORA memory
            
        Returns:
            Generated summary
        """
        ranked = query_result.get("ranked", [])
        
        if not ranked:
            return "Insufficient information for summary."
        
        # Collect all relevant content
        contents: List[str] = []
        for item in ranked:
            if isinstance(item, tuple):
                node_id, score, kind = item
            else:
                node_id = item.get("id", "")
                kind = item.get("kind", "plot")
            
            content = self._get_node_content(memory, node_id, kind)
            if content:
                contents.append(content)
        
        if not contents:
            return "Insufficient information for summary."
        
        # Try to use NarratorEngine for story reconstruction
        if hasattr(memory, "stories") and hasattr(memory, "plots"):
            # Get plots for narrative reconstruction
            plots = []
            for item in ranked:
                node_id = item[0] if isinstance(item, tuple) else item.get("id", "")
                if node_id in memory.plots:
                    plots.append(memory.plots[node_id])
            
            if plots:
                try:
                    from aurora.algorithms.narrator import NarratorEngine
                    
                    narrator = NarratorEngine(metric=memory.metric, seed=42)
                    trace = narrator.reconstruct_story(
                        query=question,
                        plots=plots,
                        stories=memory.stories,
                        themes=getattr(memory, "themes", {}),
                    )
                    
                    if trace.narrative_text:
                        return trace.narrative_text
                        
                except Exception as e:
                    logger.debug(f"NarratorEngine failed: {e}")
        
        # Fallback: concatenate contents
        return "\n".join(contents[:5])
    
    def _generate_cr_answer(
        self,
        query_result: Dict[str, Any],
        question: str,
        conflicting_facts: List[str],
        memory: Any,
    ) -> str:
        """Generate answer for CR with conflict resolution.
        
        Uses TensionManager and CoherenceGuardian if available.
        
        Args:
            query_result: Query results
            question: Question to answer
            conflicting_facts: Known conflicting facts
            memory: AURORA memory
            
        Returns:
            Answer with conflicts resolved
        """
        ranked = query_result.get("ranked", [])
        
        if not ranked:
            return "No relevant information found."
        
        # Get all candidate answers
        candidates: List[Tuple[str, float, float]] = []  # (content, score, recency)
        
        for item in ranked:
            if isinstance(item, tuple):
                node_id, score, kind = item
            else:
                node_id = item.get("id", "")
                score = item.get("score", 0.0)
                kind = item.get("kind", "plot")
            
            content = self._get_node_content(memory, node_id, kind)
            
            # Get recency (timestamp)
            recency = 0.0
            if kind == "plot" and hasattr(memory, "plots"):
                plot = memory.plots.get(node_id)
                if plot:
                    recency = plot.ts
            
            if content:
                candidates.append((content, score, recency))
        
        if not candidates:
            return "No relevant information found."
        
        # For CR, prefer most recent information
        # Sort by recency (most recent first), then by score
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Return most recent/relevant answer
        return candidates[0][0]
    
    def _get_node_content(
        self,
        memory: Any,
        node_id: str,
        kind: str,
    ) -> str:
        """Get content string from a memory node.
        
        Args:
            memory: AURORA memory instance
            node_id: Node ID
            kind: Node type ("plot", "story", "theme")
            
        Returns:
            Content string
        """
        if kind == "plot" and hasattr(memory, "plots"):
            plot = memory.plots.get(node_id)
            if plot:
                return plot.text
                
        elif kind == "story" and hasattr(memory, "stories"):
            story = memory.stories.get(node_id)
            if story:
                # Get story summary
                if hasattr(story, "to_narrative_summary"):
                    return story.to_narrative_summary()
                elif hasattr(story, "to_relationship_narrative"):
                    return story.to_relationship_narrative()
                else:
                    return f"Story with {len(story.plot_ids)} events"
                    
        elif kind == "theme" and hasattr(memory, "themes"):
            theme = memory.themes.get(node_id)
            if theme:
                if hasattr(theme, "identity_dimension") and theme.identity_dimension:
                    return theme.identity_dimension
                elif hasattr(theme, "name") and theme.name:
                    return theme.name
                    
        return ""
    
    def _evaluate_answer(
        self,
        predicted: str,
        expected: str,
    ) -> Tuple[bool, float]:
        """Evaluate predicted answer against expected.
        
        Uses LLM-as-Judge if configured and available.
        
        Args:
            predicted: Predicted answer
            expected: Expected answer
            
        Returns:
            Tuple of (is_correct, score)
        """
        # Try LLM-as-Judge first
        if self._config and self._config.use_llm_judge and self.llm is not None:
            try:
                return self._llm_judge_evaluate(predicted, expected)
            except Exception as e:
                logger.warning(f"LLM judge failed: {e}, falling back to exact match")
        
        # Fall back to exact match
        return exact_match_score(predicted, expected)
    
    def _llm_judge_evaluate(
        self,
        predicted: str,
        expected: str,
    ) -> Tuple[bool, float]:
        """Evaluate using LLM-as-Judge.
        
        Args:
            predicted: Predicted answer
            expected: Expected answer
            
        Returns:
            Tuple of (is_correct, score)
        """
        from pydantic import BaseModel
        
        class JudgeResult(BaseModel):
            is_correct: bool
            score: float
            reasoning: str
        
        user_prompt = LLM_JUDGE_USER_TEMPLATE.format(
            question="Does the predicted answer correctly match the expected answer?",
            expected_answer=expected,
            predicted_answer=predicted,
        )
        
        result = self.llm.complete_json(
            system=LLM_JUDGE_SYSTEM_PROMPT,
            user=user_prompt,
            schema=JudgeResult,
            temperature=0.0,
        )
        
        return result.is_correct, result.score


# =============================================================================
# Utility Functions
# =============================================================================

def create_adapter(
    llm_provider: Optional[Any] = None,
    **kwargs,
) -> MemoryAgentBenchAdapter:
    """Factory function to create MemoryAgentBench adapter.
    
    Args:
        llm_provider: LLM provider for evaluation
        **kwargs: Additional adapter options
        
    Returns:
        Configured MemoryAgentBenchAdapter instance
    """
    return MemoryAgentBenchAdapter(
        llm_provider=llm_provider,
        **kwargs,
    )


# =============================================================================
# CLI Support
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run MemoryAgentBench evaluation"
    )
    parser.add_argument(
        "--source",
        default="ai-hyz/MemoryAgentBench",
        help="Dataset source (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum instances to evaluate",
    )
    parser.add_argument(
        "--capability",
        choices=["ar", "ttl", "lru", "cr"],
        default=None,
        help="Filter by capability",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    
    args = parser.parse_args()
    
    # Create adapter
    adapter = MemoryAgentBenchAdapter()
    
    # Load dataset
    instances = adapter.load_dataset(args.source, subset=args.split)
    
    print(f"Loaded {len(instances)} instances")
    
    # Show capability distribution
    from collections import Counter
    cap_dist = Counter(i.reasoning_type for i in instances)
    print(f"Capability distribution: {dict(cap_dist)}")
    
    # Example: Print first instance
    if instances:
        print(f"\nFirst instance:")
        print(f"  ID: {instances[0].id}")
        print(f"  Task Type: {instances[0].task_type}")
        print(f"  Capability: {instances[0].reasoning_type}")
        print(f"  Query: {instances[0].query[:100]}...")
        print(f"  Expected: {instances[0].ground_truth[:100]}...")
