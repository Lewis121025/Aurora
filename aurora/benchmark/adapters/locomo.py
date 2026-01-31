"""
LOCOMO Benchmark Adapter
========================

Adapter for LOCOMO (ACL 2024) - Long-term Conversation Memory benchmark.

Dataset: GitHub `snap-research/locomo`
- 32+ session conversations
- ~300 turns per conversation, ~9000 tokens
- Based on persona and temporal event graph

Evaluation tasks:
1. Question Answering (QA)
   - Single-hop reasoning
   - Multi-hop reasoning
   - Temporal reasoning
   - Commonsense reasoning
   - World knowledge reasoning

2. Event Summarization
   - Generate coherent event summaries
   - AURORA implementation: Story narrative + NarratorEngine

3. Multimodal Dialogue (optional)
   - Image sharing contexts
   - Initial version can skip

Design principles:
- Zero hard-coded thresholds: Configurable evaluation parameters
- Deterministic reproducibility: Seeded random operations
- Complete type annotations
- LLM-as-judge evaluation support

Usage:
    from aurora.benchmark.adapters.locomo import LOCOMOAdapter, LOCOMOTaskType
    from aurora.algorithms.aurora_core import AuroraMemory
    
    adapter = LOCOMOAdapter(llm_provider=my_llm)
    instances = adapter.load_dataset("path/to/locomo")
    
    memory = AuroraMemory(seed=42)
    results = []
    
    for instance in instances:
        # Prepare memory with conversation history
        adapter.prepare_memory(instance.conversation_history, memory)
        
        # Evaluate
        result = adapter.evaluate(instance, memory)
        results.append(result)
    
    # Aggregate results by reasoning type
    metrics = adapter.aggregate_results(results)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from aurora.benchmark.interface import (
    BenchmarkAdapter,
    BenchmarkInstance,
    BenchmarkResult,
    EvaluationMetrics,
    EvaluationMethod,
    compute_f1_score,
    contains_score,
    exact_match_score,
    fuzzy_match_score,
)
from aurora.utils.time_utils import now_ts

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class LOCOMOReasoningType(Enum):
    """Types of reasoning required for LOCOMO QA tasks.
    
    Each type tests different memory capabilities:
    - SINGLE_HOP: Direct fact retrieval
    - MULTI_HOP: Combining multiple facts
    - TEMPORAL: Time-based reasoning
    - COMMONSENSE: Common knowledge inference
    - WORLD_KNOWLEDGE: External world knowledge
    """
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    COMMONSENSE = "commonsense"
    WORLD_KNOWLEDGE = "world_knowledge"
    
    @classmethod
    def from_string(cls, s: str) -> "LOCOMOReasoningType":
        """Parse reasoning type from string."""
        mapping = {
            "single-hop": cls.SINGLE_HOP,
            "single_hop": cls.SINGLE_HOP,
            "singlehop": cls.SINGLE_HOP,
            "multi-hop": cls.MULTI_HOP,
            "multi_hop": cls.MULTI_HOP,
            "multihop": cls.MULTI_HOP,
            "temporal": cls.TEMPORAL,
            "time": cls.TEMPORAL,
            "commonsense": cls.COMMONSENSE,
            "common_sense": cls.COMMONSENSE,
            "world": cls.WORLD_KNOWLEDGE,
            "world_knowledge": cls.WORLD_KNOWLEDGE,
            "knowledge": cls.WORLD_KNOWLEDGE,
        }
        return mapping.get(s.lower().replace(" ", "_"), cls.SINGLE_HOP)


class LOCOMOTaskType(Enum):
    """Types of tasks in LOCOMO benchmark.
    
    QUESTION_ANSWERING: Answer questions about conversation history
    EVENT_SUMMARIZATION: Generate coherent event summaries
    DIALOGUE_GENERATION: Generate appropriate dialogue responses (optional)
    """
    QUESTION_ANSWERING = "qa"
    EVENT_SUMMARIZATION = "summarization"
    DIALOGUE_GENERATION = "dialogue"
    
    @classmethod
    def from_string(cls, s: str) -> "LOCOMOTaskType":
        """Parse task type from string."""
        mapping = {
            "qa": cls.QUESTION_ANSWERING,
            "question_answering": cls.QUESTION_ANSWERING,
            "question-answering": cls.QUESTION_ANSWERING,
            "summarization": cls.EVENT_SUMMARIZATION,
            "event_summarization": cls.EVENT_SUMMARIZATION,
            "summary": cls.EVENT_SUMMARIZATION,
            "dialogue": cls.DIALOGUE_GENERATION,
            "dialog": cls.DIALOGUE_GENERATION,
            "generation": cls.DIALOGUE_GENERATION,
        }
        return mapping.get(s.lower().replace(" ", "_"), cls.QUESTION_ANSWERING)


# =============================================================================
# Pydantic Models for LLM Evaluation
# =============================================================================

if PYDANTIC_AVAILABLE:
    class QAEvaluationResult(BaseModel):
        """LLM-as-judge evaluation result for QA tasks."""
        is_correct: bool = Field(description="Whether the answer is correct")
        score: float = Field(ge=0.0, le=1.0, description="Confidence score [0, 1]")
        explanation: str = Field(description="Brief explanation of the evaluation")
        
    class SummarizationEvaluationResult(BaseModel):
        """LLM-as-judge evaluation result for summarization tasks."""
        coherence_score: float = Field(ge=0.0, le=1.0, description="Coherence of summary")
        coverage_score: float = Field(ge=0.0, le=1.0, description="Coverage of key events")
        accuracy_score: float = Field(ge=0.0, le=1.0, description="Factual accuracy")
        overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
        missing_events: List[str] = Field(default_factory=list, description="Key events not covered")
        explanation: str = Field(description="Brief explanation")
else:
    QAEvaluationResult = None  # type: ignore
    SummarizationEvaluationResult = None  # type: ignore


# =============================================================================
# LOCOMO Data Structures
# =============================================================================

@dataclass
class LOCOMOTurn:
    """A single turn in a LOCOMO conversation.
    
    Attributes:
        turn_id: Unique turn identifier
        speaker: Speaker identifier (e.g., "user1", "user2")
        text: Turn text content
        timestamp: Optional timestamp
        metadata: Additional turn metadata
    """
    turn_id: str
    speaker: str
    text: str
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for memory ingestion."""
        return {
            "id": self.turn_id,
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
            **self.metadata,
        }


@dataclass
class LOCOMOSession:
    """A LOCOMO conversation session.
    
    Attributes:
        session_id: Unique session identifier
        turns: List of conversation turns
        personas: Persona information for participants
        events: Temporal event graph
        metadata: Session metadata
    """
    session_id: str
    turns: List[LOCOMOTurn]
    personas: Dict[str, str] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LOCOMOQuestion:
    """A LOCOMO evaluation question.
    
    Attributes:
        question_id: Unique question identifier
        session_id: Associated session
        question: Question text
        answer: Ground truth answer
        reasoning_type: Type of reasoning required
        evidence_turns: Turn IDs containing evidence
    """
    question_id: str
    session_id: str
    question: str
    answer: str
    reasoning_type: LOCOMOReasoningType
    evidence_turns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# LOCOMO Adapter
# =============================================================================

class LOCOMOAdapter(BenchmarkAdapter):
    """Adapter for LOCOMO benchmark evaluation.
    
    LOCOMO (Long-term Conversation Memory) is an ACL 2024 benchmark
    designed to evaluate long-term dialogue memory systems.
    
    Key features:
    - Multi-session dialogues with ~300 turns, ~9000 tokens
    - Five reasoning types for QA evaluation
    - Event summarization tasks
    - Persona and temporal event graphs
    
    AURORA capability mapping:
    - Accurate retrieval: query() + FieldRetriever
    - Multi-hop reasoning: Graph traversal + attractor tracing
    - Temporal reasoning: Story timeline + temporal edges
    - Summarization: Story narrative + NarratorEngine
    
    Attributes:
        llm: Optional LLM provider for LLM-as-judge evaluation
        evaluation_method: Method for answer evaluation
        use_narrator_for_summary: Whether to use NarratorEngine for summaries
        
    Usage:
        adapter = LOCOMOAdapter(llm_provider=my_llm)
        instances = adapter.load_dataset("path/to/locomo")
        
        for instance in instances:
            result = adapter.evaluate(instance, memory)
    """
    
    # Dataset URLs
    GITHUB_REPO = "snap-research/locomo"
    DEFAULT_DATA_DIR = "data"
    
    @property
    def name(self) -> str:
        """Return the benchmark name."""
        return "LOCOMO"
    
    # Evaluation prompts
    QA_EVALUATION_PROMPT = """You are evaluating a question-answering task for a conversation memory system.

Conversation context has been ingested into memory. Based on the retrieved information, 
the system generated an answer to a question.

Question: {question}
Ground Truth Answer: {ground_truth}
System Answer: {prediction}

Evaluate whether the system's answer is correct. Consider:
1. Does the answer contain the key information from the ground truth?
2. Is the answer factually consistent with the ground truth?
3. Minor phrasing differences are acceptable if the meaning is preserved.

Respond with your evaluation."""
    
    SUMMARIZATION_EVALUATION_PROMPT = """You are evaluating an event summarization task for a conversation memory system.

The system generated a summary of events from a conversation.

Ground Truth Summary: {ground_truth}
System Summary: {prediction}

Evaluate the summary on:
1. Coherence: Is the summary well-organized and readable?
2. Coverage: Does it cover the key events from the ground truth?
3. Accuracy: Are the facts correct?

List any key events from the ground truth that are missing in the system summary.

Respond with your evaluation."""
    
    def __init__(
        self,
        llm_provider=None,
        seed: int = 0,
        evaluation_method: EvaluationMethod = EvaluationMethod.LLM_JUDGE,
        use_narrator_for_summary: bool = True,
        f1_threshold: float = 0.5,
        fuzzy_threshold: float = 0.7,
    ):
        """Initialize the LOCOMO adapter.
        
        Args:
            llm_provider: Optional LLM provider for LLM-as-judge evaluation
            seed: Random seed for reproducibility
            evaluation_method: Method for evaluating predictions
            use_narrator_for_summary: Whether to use NarratorEngine for summaries
            f1_threshold: Threshold for F1-based correctness
            fuzzy_threshold: Threshold for fuzzy matching
        """
        super().__init__(llm_provider=llm_provider, seed=seed)
        
        self.evaluation_method = evaluation_method
        self.use_narrator_for_summary = use_narrator_for_summary
        self.f1_threshold = f1_threshold
        self.fuzzy_threshold = fuzzy_threshold
        
        # Cache for loaded sessions
        self._session_cache: Dict[str, LOCOMOSession] = {}
        
        # Statistics tracking
        self._eval_stats: Dict[str, int] = {
            "total_evals": 0,
            "llm_evals": 0,
            "fallback_evals": 0,
        }
    
    # -------------------------------------------------------------------------
    # Dataset Loading
    # -------------------------------------------------------------------------
    
    def load_dataset(
        self,
        path: str,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BenchmarkInstance]:
        """Load LOCOMO dataset from local path or remote source.
        
        Supports multiple data formats:
        - JSON files with sessions and questions
        - Directory containing session files
        - GitHub repository URL (will download)
        
        Args:
            path: Path to dataset (local or GitHub URL)
            subset: Optional subset ("train", "test", "val", or reasoning type)
            limit: Optional limit on number of instances
            
        Returns:
            List of BenchmarkInstance objects
        """
        logger.info(f"Loading LOCOMO dataset from {path}")
        
        # Handle different path types
        if path.startswith("github:") or path.startswith("https://github.com"):
            return self._load_from_github(path, subset, limit)
        
        path_obj = Path(path)
        
        if path_obj.is_file():
            return self._load_from_file(path_obj, subset, limit)
        elif path_obj.is_dir():
            return self._load_from_directory(path_obj, subset, limit)
        else:
            raise FileNotFoundError(f"Dataset path not found: {path}")
    
    def _load_from_file(
        self,
        path: Path,
        subset: Optional[str],
        limit: Optional[int],
    ) -> List[BenchmarkInstance]:
        """Load dataset from a single JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return self._parse_dataset(data, subset, limit)
    
    def _load_from_directory(
        self,
        path: Path,
        subset: Optional[str],
        limit: Optional[int],
    ) -> List[BenchmarkInstance]:
        """Load dataset from a directory of files."""
        instances: List[BenchmarkInstance] = []
        
        # Look for standard LOCOMO file structure
        sessions_file = path / "sessions.json"
        questions_file = path / "questions.json"
        
        if sessions_file.exists() and questions_file.exists():
            # Standard LOCOMO structure
            with open(sessions_file, "r", encoding="utf-8") as f:
                sessions_data = json.load(f)
            with open(questions_file, "r", encoding="utf-8") as f:
                questions_data = json.load(f)
            
            data = {
                "sessions": sessions_data,
                "questions": questions_data,
            }
            return self._parse_dataset(data, subset, limit)
        
        # Try loading individual JSON files
        json_files = list(path.glob("*.json"))
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            instances.extend(self._parse_dataset(data, subset, None))
            
            if limit and len(instances) >= limit:
                break
        
        if limit:
            instances = instances[:limit]
        
        return instances
    
    def _load_from_github(
        self,
        url: str,
        subset: Optional[str],
        limit: Optional[int],
    ) -> List[BenchmarkInstance]:
        """Load dataset from GitHub repository.
        
        Note: Requires network access. Falls back to local cache if available.
        """
        try:
            import urllib.request
            import tempfile
            import zipfile
            
            # Extract repo info
            if url.startswith("github:"):
                repo = url[7:]
            else:
                # Parse GitHub URL
                parts = url.replace("https://github.com/", "").split("/")
                repo = "/".join(parts[:2])
            
            # Download repository as zip
            zip_url = f"https://github.com/{repo}/archive/refs/heads/main.zip"
            
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "repo.zip"
                
                logger.info(f"Downloading LOCOMO dataset from {zip_url}")
                urllib.request.urlretrieve(zip_url, zip_path)
                
                # Extract
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Find data directory
                extracted_dirs = list(Path(tmpdir).glob("*/"))
                if extracted_dirs:
                    data_dir = extracted_dirs[0] / self.DEFAULT_DATA_DIR
                    if data_dir.exists():
                        return self._load_from_directory(data_dir, subset, limit)
                
                raise FileNotFoundError("Could not find data directory in repository")
                
        except Exception as e:
            logger.warning(f"Failed to load from GitHub: {e}")
            raise
    
    def _parse_dataset(
        self,
        data: Dict[str, Any],
        subset: Optional[str],
        limit: Optional[int],
    ) -> List[BenchmarkInstance]:
        """Parse dataset JSON into BenchmarkInstance objects."""
        instances: List[BenchmarkInstance] = []
        
        # Parse sessions
        sessions = data.get("sessions", data.get("dialogues", []))
        if isinstance(sessions, dict):
            sessions = list(sessions.values())
        
        for session_data in sessions:
            session = self._parse_session(session_data)
            self._session_cache[session.session_id] = session
        
        # Parse questions
        questions = data.get("questions", data.get("qa_pairs", []))
        if isinstance(questions, dict):
            questions = list(questions.values())
        
        for q_data in questions:
            question = self._parse_question(q_data)
            
            # Filter by subset (reasoning type)
            if subset:
                try:
                    subset_type = LOCOMOReasoningType.from_string(subset)
                    if question.reasoning_type != subset_type:
                        continue
                except (ValueError, KeyError):
                    pass
            
            # Get session for this question
            session = self._session_cache.get(question.session_id)
            if session is None:
                logger.warning(f"Session not found for question {question.question_id}")
                continue
            
            # Create benchmark instance
            instance = BenchmarkInstance(
                id=question.question_id,
                task_type=LOCOMOTaskType.QUESTION_ANSWERING.value,
                conversation_history=[turn.to_dict() for turn in session.turns],
                query=question.question,
                ground_truth=question.answer,
                reasoning_type=question.reasoning_type.value,
                session_id=question.session_id,
                metadata={
                    "evidence_turns": question.evidence_turns,
                    "personas": session.personas,
                    "events": session.events,
                    **question.metadata,
                },
            )
            instances.append(instance)
            
            if limit and len(instances) >= limit:
                break
        
        # Parse summarization tasks if present
        summaries = data.get("summaries", data.get("event_summaries", []))
        if isinstance(summaries, dict):
            summaries = list(summaries.values())
        
        for s_data in summaries:
            session_id = s_data.get("session_id", s_data.get("dialogue_id"))
            session = self._session_cache.get(session_id)
            
            if session is None:
                continue
            
            instance = BenchmarkInstance(
                id=s_data.get("id", f"summary_{session_id}"),
                task_type=LOCOMOTaskType.EVENT_SUMMARIZATION.value,
                conversation_history=[turn.to_dict() for turn in session.turns],
                query=s_data.get("instruction", "Summarize the key events in this conversation."),
                ground_truth=s_data.get("summary", s_data.get("ground_truth", "")),
                session_id=session_id,
                metadata=s_data.get("metadata", {}),
            )
            instances.append(instance)
            
            if limit and len(instances) >= limit:
                break
        
        logger.info(f"Loaded {len(instances)} benchmark instances")
        return instances
    
    def _parse_session(self, data: Dict[str, Any]) -> LOCOMOSession:
        """Parse session data into LOCOMOSession object."""
        session_id = data.get("session_id", data.get("dialogue_id", str(self.rng.integers(10000))))
        
        # Parse turns
        turns_data = data.get("turns", data.get("utterances", data.get("dialogue", [])))
        turns: List[LOCOMOTurn] = []
        
        for i, turn_data in enumerate(turns_data):
            if isinstance(turn_data, str):
                # Simple string format
                turn = LOCOMOTurn(
                    turn_id=f"{session_id}_turn_{i}",
                    speaker="user" if i % 2 == 0 else "agent",
                    text=turn_data,
                )
            elif isinstance(turn_data, dict):
                turn = LOCOMOTurn(
                    turn_id=turn_data.get("id", turn_data.get("turn_id", f"{session_id}_turn_{i}")),
                    speaker=turn_data.get("speaker", turn_data.get("role", "user")),
                    text=turn_data.get("text", turn_data.get("content", turn_data.get("utterance", ""))),
                    timestamp=turn_data.get("timestamp"),
                    metadata={k: v for k, v in turn_data.items() 
                             if k not in ("id", "turn_id", "speaker", "role", "text", "content", "utterance", "timestamp")},
                )
            else:
                continue
            
            turns.append(turn)
        
        return LOCOMOSession(
            session_id=session_id,
            turns=turns,
            personas=data.get("personas", data.get("persona", {})),
            events=data.get("events", data.get("temporal_events", [])),
            metadata={k: v for k, v in data.items() 
                     if k not in ("session_id", "dialogue_id", "turns", "utterances", "dialogue", 
                                 "personas", "persona", "events", "temporal_events")},
        )
    
    def _parse_question(self, data: Dict[str, Any]) -> LOCOMOQuestion:
        """Parse question data into LOCOMOQuestion object."""
        reasoning_str = data.get("reasoning_type", data.get("type", data.get("category", "single_hop")))
        
        return LOCOMOQuestion(
            question_id=data.get("id", data.get("question_id", str(self.rng.integers(10000)))),
            session_id=data.get("session_id", data.get("dialogue_id", "")),
            question=data.get("question", data.get("query", "")),
            answer=data.get("answer", data.get("ground_truth", data.get("response", ""))),
            reasoning_type=LOCOMOReasoningType.from_string(reasoning_str),
            evidence_turns=data.get("evidence_turns", data.get("supporting_turns", [])),
            metadata={k: v for k, v in data.items()
                     if k not in ("id", "question_id", "session_id", "dialogue_id", "question", 
                                 "query", "answer", "ground_truth", "response", "reasoning_type",
                                 "type", "category", "evidence_turns", "supporting_turns")},
        )
    
    def _parse_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse LOCOMO conversation format into list of turns.
        
        Handles various LOCOMO conversation formats.
        
        Args:
            conversation: Raw conversation data
            
        Returns:
            List of turn dictionaries
        """
        if isinstance(conversation, list):
            return conversation
        
        # Handle nested format
        turns = conversation.get("turns", conversation.get("utterances", conversation.get("dialogue", [])))
        
        parsed_turns: List[Dict[str, Any]] = []
        for turn in turns:
            if isinstance(turn, str):
                parsed_turns.append({"text": turn})
            elif isinstance(turn, dict):
                parsed_turns.append({
                    "id": turn.get("id", turn.get("turn_id")),
                    "speaker": turn.get("speaker", turn.get("role", "user")),
                    "text": turn.get("text", turn.get("content", turn.get("utterance", ""))),
                    "timestamp": turn.get("timestamp"),
                })
        
        return parsed_turns
    
    # -------------------------------------------------------------------------
    # Memory Preparation
    # -------------------------------------------------------------------------
    
    def _prepare_memory_from_sessions(
        self,
        sessions: List[Dict[str, Any]],
        memory,
    ) -> None:
        """Prepare memory from multiple conversation sessions.
        
        For LOCOMO's multi-session format, this ingests turns
        with appropriate temporal and speaker context.
        
        Args:
            sessions: List of session data
            memory: AURORA memory instance
        """
        for session in sessions:
            session_id = session.get("session_id", session.get("dialogue_id"))
            turns = self._parse_conversation(session)
            
            for turn in turns:
                text = turn.get("text", turn.get("content", str(turn)))
                speaker = turn.get("speaker", turn.get("role", "user"))
                
                # Format for AURORA ingestion
                if speaker in ("user", "user1", "user2"):
                    formatted_text = f"用户：{text}"
                    actors = ("user", "agent")
                else:
                    formatted_text = f"助理：{text}"
                    actors = ("agent", "user")
                
                memory.ingest(
                    interaction_text=formatted_text,
                    actors=actors,
                    event_id=turn.get("id"),
                    context_text=f"session:{session_id}",
                )
    
    def prepare_memory(
        self,
        conversation_history: List[Dict[str, Any]],
        memory,
    ) -> None:
        """Prepare memory by ingesting conversation history.
        
        Overrides base implementation for LOCOMO-specific format.
        
        Args:
            conversation_history: List of conversation turns
            memory: AURORA memory instance
        """
        for turn in conversation_history:
            text = turn.get("text", turn.get("content", str(turn)))
            speaker = turn.get("speaker", turn.get("role", "user"))
            
            # Format for AURORA ingestion
            if speaker in ("user", "user1", "user2"):
                formatted_text = f"用户：{text}"
                actors = ("user", "agent")
            else:
                formatted_text = f"助理：{text}"
                actors = ("agent", "user")
            
            memory.ingest(
                interaction_text=formatted_text,
                actors=actors,
                event_id=turn.get("id"),
            )
    
    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    
    def evaluate(
        self,
        instance: BenchmarkInstance,
        memory,
        **kwargs,
    ) -> BenchmarkResult:
        """Evaluate a single benchmark instance.
        
        Routes to task-specific evaluation based on instance type.
        
        Args:
            instance: The benchmark instance to evaluate
            memory: AURORA memory instance
            **kwargs: Additional evaluation parameters
            
        Returns:
            BenchmarkResult with evaluation outcome
        """
        task_type = LOCOMOTaskType.from_string(instance.task_type)
        
        if task_type == LOCOMOTaskType.QUESTION_ANSWERING:
            reasoning_type = (
                LOCOMOReasoningType.from_string(instance.reasoning_type)
                if instance.reasoning_type
                else LOCOMOReasoningType.SINGLE_HOP
            )
            return self._evaluate_qa(instance, memory, reasoning_type)
        
        elif task_type == LOCOMOTaskType.EVENT_SUMMARIZATION:
            return self._evaluate_summarization(instance, memory)
        
        elif task_type == LOCOMOTaskType.DIALOGUE_GENERATION:
            return self._evaluate_dialogue(instance, memory)
        
        else:
            # Default to QA evaluation
            return self._evaluate_qa(instance, memory, LOCOMOReasoningType.SINGLE_HOP)
    
    def _evaluate_qa(
        self,
        instance: BenchmarkInstance,
        memory,
        reasoning_type: LOCOMOReasoningType,
    ) -> BenchmarkResult:
        """Evaluate question answering task.
        
        Uses AURORA's query system to retrieve relevant memories
        and generates an answer based on retrieved context.
        
        Args:
            instance: QA benchmark instance
            memory: AURORA memory instance
            reasoning_type: Type of reasoning required
            
        Returns:
            BenchmarkResult with QA evaluation
        """
        start_time = time.perf_counter()
        reasoning_trace: List[str] = []
        
        try:
            # Query memory for relevant context
            trace = memory.query(instance.query, k=10)
            reasoning_trace.append(f"Retrieved {len(trace.ranked)} memories")
            
            # Gather retrieved content
            retrieved_texts: List[str] = []
            for nid, score, kind in trace.ranked[:5]:
                try:
                    payload = memory.graph.payload(nid)
                    if hasattr(payload, "text"):
                        retrieved_texts.append(payload.text)
                    elif hasattr(payload, "to_narrative_summary"):
                        retrieved_texts.append(payload.to_narrative_summary())
                except Exception as e:
                    logger.debug(f"Failed to retrieve payload for node {nid}: {e}")
            
            reasoning_trace.append(f"Gathered {len(retrieved_texts)} text snippets")
            
            # Generate prediction
            if self.llm is not None:
                prediction = self._generate_qa_answer(
                    question=instance.query,
                    context=retrieved_texts,
                    reasoning_type=reasoning_type,
                )
            else:
                # Fallback: use most relevant retrieved text
                prediction = retrieved_texts[0] if retrieved_texts else ""
            
            reasoning_trace.append(f"Generated prediction: {prediction[:100]}...")
            
            # Evaluate
            score, is_correct = self._evaluate_qa_answer(
                prediction=prediction,
                ground_truth=instance.ground_truth,
                question=instance.query,
            )
            
            reasoning_trace.append(f"Score: {score:.2f}, Correct: {is_correct}")
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return BenchmarkResult(
                instance_id=instance.id,
                task_type=f"qa_{reasoning_type.value}",
                prediction=prediction,
                ground_truth=instance.ground_truth,
                score=score,
                is_correct=is_correct,
                latency_ms=latency_ms,
                retrieval_count=len(trace.ranked),
                reasoning_trace=reasoning_trace,
                metadata={
                    "reasoning_type": reasoning_type.value,
                    "retrieved_count": len(retrieved_texts),
                },
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"QA evaluation error: {e}")
            
            return BenchmarkResult(
                instance_id=instance.id,
                task_type=f"qa_{reasoning_type.value}",
                prediction="",
                ground_truth=instance.ground_truth,
                score=0.0,
                is_correct=False,
                latency_ms=latency_ms,
                error_message=str(e),
                reasoning_trace=reasoning_trace,
            )
    
    def _generate_qa_answer(
        self,
        question: str,
        context: List[str],
        reasoning_type: LOCOMOReasoningType,
    ) -> str:
        """Generate QA answer using LLM.
        
        Args:
            question: The question to answer
            context: Retrieved context from memory
            reasoning_type: Type of reasoning required
            
        Returns:
            Generated answer string
        """
        if self.llm is None:
            return context[0] if context else ""
        
        # Build prompt based on reasoning type
        reasoning_hints = {
            LOCOMOReasoningType.SINGLE_HOP: "Answer directly from the context.",
            LOCOMOReasoningType.MULTI_HOP: "Combine information from multiple parts of the context.",
            LOCOMOReasoningType.TEMPORAL: "Pay attention to time references and event ordering.",
            LOCOMOReasoningType.COMMONSENSE: "Use common sense to interpret the context.",
            LOCOMOReasoningType.WORLD_KNOWLEDGE: "You may need to apply general world knowledge.",
        }
        
        hint = reasoning_hints.get(reasoning_type, "")
        context_text = "\n\n".join(context) if context else "No relevant context found."
        
        system_prompt = f"""You are a precise question-answering assistant.
Answer the question based on the provided conversation context.
{hint}
Be concise and answer directly."""
        
        user_prompt = f"""Context from memory:
{context_text}

Question: {question}

Answer:"""
        
        try:
            # Use LLM to generate answer
            # For now, return a simple completion
            # In production, this would use the LLM provider
            from pydantic import BaseModel, Field
            
            class QAAnswer(BaseModel):
                answer: str = Field(description="The answer to the question")
            
            result = self.llm.complete_json(
                system=system_prompt,
                user=user_prompt,
                schema=QAAnswer,
                temperature=0.1,
            )
            return result.answer
            
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to context")
            return context[0] if context else ""
    
    def _evaluate_qa_answer(
        self,
        prediction: str,
        ground_truth: str,
        question: str,
    ) -> Tuple[float, bool]:
        """Evaluate QA answer against ground truth.
        
        Args:
            prediction: Model's answer
            ground_truth: Expected answer
            question: Original question
            
        Returns:
            Tuple of (score, is_correct)
        """
        self._eval_stats["total_evals"] += 1
        
        # Try LLM-as-judge if available and configured
        if (self.evaluation_method == EvaluationMethod.LLM_JUDGE 
            and self.llm is not None 
            and QAEvaluationResult is not None):
            try:
                result = self.llm.complete_json(
                    system="You are evaluating question-answering accuracy.",
                    user=self.QA_EVALUATION_PROMPT.format(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction,
                    ),
                    schema=QAEvaluationResult,
                    temperature=0.1,
                )
                self._eval_stats["llm_evals"] += 1
                return result.score, result.is_correct
                
            except Exception as e:
                logger.warning(f"LLM evaluation failed: {e}, falling back")
        
        # Fallback evaluation methods
        self._eval_stats["fallback_evals"] += 1
        
        # Exact match
        if exact_match_score(prediction, ground_truth) > 0.5:
            return 1.0, True
        
        # Contains match
        if contains_score(prediction, ground_truth) > 0.5:
            return 0.9, True
        
        # F1 score
        f1 = compute_f1_score(prediction, ground_truth)
        if f1 >= self.f1_threshold:
            return f1, True
        
        # Fuzzy match
        fuzzy = fuzzy_match_score(prediction, ground_truth, threshold=self.fuzzy_threshold)
        if fuzzy > 0.5:
            return fuzzy, True
        
        return max(f1, fuzzy * 0.5), False
    
    def _evaluate_summarization(
        self,
        instance: BenchmarkInstance,
        memory,
    ) -> BenchmarkResult:
        """Evaluate event summarization task.
        
        Uses AURORA's NarratorEngine to generate story summaries
        and compares against ground truth.
        
        Args:
            instance: Summarization benchmark instance
            memory: AURORA memory instance
            
        Returns:
            BenchmarkResult with summarization evaluation
        """
        start_time = time.perf_counter()
        reasoning_trace: List[str] = []
        
        try:
            # Generate summary using AURORA
            if self.use_narrator_for_summary:
                prediction = self._generate_narrative_summary(memory, instance)
            else:
                prediction = self._generate_simple_summary(memory, instance)
            
            reasoning_trace.append(f"Generated summary of {len(prediction)} characters")
            
            # Evaluate
            score, metrics = self._evaluate_summary(
                prediction=prediction,
                ground_truth=instance.ground_truth,
            )
            
            is_correct = score >= 0.5
            reasoning_trace.append(f"Score: {score:.2f}")
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return BenchmarkResult(
                instance_id=instance.id,
                task_type=LOCOMOTaskType.EVENT_SUMMARIZATION.value,
                prediction=prediction,
                ground_truth=instance.ground_truth,
                score=score,
                is_correct=is_correct,
                latency_ms=latency_ms,
                reasoning_trace=reasoning_trace,
                metadata=metrics,
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Summarization evaluation error: {e}")
            
            return BenchmarkResult(
                instance_id=instance.id,
                task_type=LOCOMOTaskType.EVENT_SUMMARIZATION.value,
                prediction="",
                ground_truth=instance.ground_truth,
                score=0.0,
                is_correct=False,
                latency_ms=latency_ms,
                error_message=str(e),
                reasoning_trace=reasoning_trace,
            )
    
    def _generate_narrative_summary(
        self,
        memory,
        instance: BenchmarkInstance,
    ) -> str:
        """Generate summary using AURORA's NarratorEngine.
        
        Args:
            memory: AURORA memory instance
            instance: Benchmark instance
            
        Returns:
            Generated summary text
        """
        try:
            from aurora.algorithms.narrator import NarratorEngine
            
            # Get all plots
            plots = list(memory.plots.values())
            
            if not plots:
                return "No events to summarize."
            
            # Create narrator engine with safe attribute access
            vindex = getattr(memory, "vindex", None)
            graph = getattr(memory, "graph", None)
            
            narrator = NarratorEngine(
                metric=memory.metric,
                vindex=vindex,
                graph=graph,
                seed=self._seed,
            )
            
            # Reconstruct story
            trace = narrator.reconstruct_story(
                query=instance.query or "Summarize the conversation events",
                plots=plots,
                stories=memory.stories,
                themes=memory.themes,
            )
            
            return trace.narrative_text
            
        except ImportError:
            return self._generate_simple_summary(memory, instance)
    
    def _generate_simple_summary(
        self,
        memory,
        instance: BenchmarkInstance,
    ) -> str:
        """Generate simple summary without NarratorEngine.
        
        Args:
            memory: AURORA memory instance
            instance: Benchmark instance
            
        Returns:
            Generated summary text
        """
        # Get story summaries
        summaries: List[str] = []
        
        for story in memory.stories.values():
            if story.is_relationship_story():
                summaries.append(story.to_relationship_narrative())
            else:
                summaries.append(story.to_narrative_summary())
        
        if not summaries:
            # Fall back to recent plots
            recent_plots = sorted(
                memory.plots.values(),
                key=lambda p: p.ts,
                reverse=True,
            )[:10]
            
            summaries = [p.text[:200] for p in recent_plots]
        
        return "\n\n".join(summaries) if summaries else "No events to summarize."
    
    def _evaluate_summary(
        self,
        prediction: str,
        ground_truth: str,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate summary against ground truth.
        
        Args:
            prediction: Generated summary
            ground_truth: Expected summary
            
        Returns:
            Tuple of (overall_score, metrics_dict)
        """
        self._eval_stats["total_evals"] += 1
        
        # Try LLM-as-judge if available
        if (self.evaluation_method == EvaluationMethod.LLM_JUDGE 
            and self.llm is not None 
            and SummarizationEvaluationResult is not None):
            try:
                result = self.llm.complete_json(
                    system="You are evaluating event summarization quality.",
                    user=self.SUMMARIZATION_EVALUATION_PROMPT.format(
                        ground_truth=ground_truth,
                        prediction=prediction,
                    ),
                    schema=SummarizationEvaluationResult,
                    temperature=0.1,
                )
                self._eval_stats["llm_evals"] += 1
                
                metrics = {
                    "coherence": result.coherence_score,
                    "coverage": result.coverage_score,
                    "accuracy": result.accuracy_score,
                }
                return result.overall_score, metrics
                
            except Exception as e:
                logger.warning(f"LLM evaluation failed: {e}, falling back")
        
        # Fallback: F1-based evaluation
        self._eval_stats["fallback_evals"] += 1
        
        f1 = compute_f1_score(prediction, ground_truth)
        fuzzy = fuzzy_match_score(prediction, ground_truth, threshold=0.3)
        
        # Estimate coverage by keyword overlap
        ground_keywords = set(ground_truth.lower().split())
        pred_keywords = set(prediction.lower().split())
        coverage = len(ground_keywords & pred_keywords) / max(len(ground_keywords), 1)
        
        overall = 0.4 * f1 + 0.3 * fuzzy + 0.3 * coverage
        
        metrics = {
            "f1": f1,
            "fuzzy": fuzzy,
            "coverage": coverage,
        }
        
        return overall, metrics
    
    def _evaluate_dialogue(
        self,
        instance: BenchmarkInstance,
        memory,
    ) -> BenchmarkResult:
        """Evaluate dialogue generation task (optional).
        
        Args:
            instance: Dialogue benchmark instance
            memory: AURORA memory instance
            
        Returns:
            BenchmarkResult with dialogue evaluation
        """
        # Dialogue generation is optional for initial version
        # Implement basic response relevance evaluation
        
        start_time = time.perf_counter()
        
        try:
            # Query memory for context
            trace = memory.query(instance.query, k=5)
            
            # Generate simple response
            context_texts = []
            for nid, _, _ in trace.ranked[:3]:
                try:
                    payload = memory.graph.payload(nid)
                    if hasattr(payload, "text"):
                        context_texts.append(payload.text)
                except Exception as e:
                    logger.debug(f"Failed to retrieve payload for node {nid}: {e}")
            
            prediction = context_texts[0] if context_texts else "I don't have enough context to respond."
            
            # Basic evaluation
            f1 = compute_f1_score(prediction, instance.ground_truth)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return BenchmarkResult(
                instance_id=instance.id,
                task_type=LOCOMOTaskType.DIALOGUE_GENERATION.value,
                prediction=prediction,
                ground_truth=instance.ground_truth,
                score=f1,
                is_correct=f1 >= 0.3,
                latency_ms=latency_ms,
                retrieval_count=len(trace.ranked),
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return BenchmarkResult(
                instance_id=instance.id,
                task_type=LOCOMOTaskType.DIALOGUE_GENERATION.value,
                prediction="",
                ground_truth=instance.ground_truth,
                score=0.0,
                is_correct=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )
    
    # -------------------------------------------------------------------------
    # Result Aggregation
    # -------------------------------------------------------------------------
    
    def aggregate_results(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, float]:
        """Aggregate results with LOCOMO-specific breakdown.
        
        Provides metrics broken down by:
        - Task type (QA, summarization, dialogue)
        - Reasoning type (single-hop, multi-hop, temporal, etc.)
        
        Also returns an EvaluationMetrics object via get_evaluation_metrics().
        
        Args:
            results: List of BenchmarkResult objects
            
        Returns:
            Dict mapping metric names to values
        """
        if not results:
            return {"accuracy": 0.0, "mean_latency_ms": 0.0}
        
        # Basic counts
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        
        # Averages
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
        
        # Add reasoning type breakdown for QA tasks
        qa_results = [r for r in results if r.task_type.startswith("qa_")]
        
        for reasoning_type in LOCOMOReasoningType:
            type_results = [
                r for r in qa_results 
                if r.metadata.get("reasoning_type") == reasoning_type.value
            ]
            
            if type_results:
                type_total = len(type_results)
                type_correct = sum(1 for r in type_results if r.is_correct)
                type_scores = [r.score for r in type_results]
                
                prefix = f"qa_{reasoning_type.value}"
                metrics[f"{prefix}_total"] = float(type_total)
                metrics[f"{prefix}_correct"] = float(type_correct)
                metrics[f"{prefix}_accuracy"] = type_correct / type_total if type_total > 0 else 0.0
                metrics[f"{prefix}_avg_score"] = float(np.mean(type_scores)) if type_scores else 0.0
        
        # Add summarization metrics if present
        summary_results = [r for r in results if r.task_type == LOCOMOTaskType.EVENT_SUMMARIZATION.value]
        if summary_results:
            sum_total = len(summary_results)
            sum_correct = sum(1 for r in summary_results if r.is_correct)
            sum_scores = [r.score for r in summary_results]
            
            metrics["summarization_total"] = float(sum_total)
            metrics["summarization_correct"] = float(sum_correct)
            metrics["summarization_accuracy"] = sum_correct / sum_total if sum_total > 0 else 0.0
            metrics["summarization_avg_score"] = float(np.mean(sum_scores)) if sum_scores else 0.0
        
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
        
        for reasoning_type in LOCOMOReasoningType:
            prefix = f"qa_{reasoning_type.value}"
            if f"{prefix}_total" in metrics_dict:
                metrics_by_type[prefix] = {
                    "total": metrics_dict[f"{prefix}_total"],
                    "correct": metrics_dict[f"{prefix}_correct"],
                    "accuracy": metrics_dict[f"{prefix}_accuracy"],
                    "avg_score": metrics_dict[f"{prefix}_avg_score"],
                }
        
        if "summarization_total" in metrics_dict:
            metrics_by_type["summarization"] = {
                "total": metrics_dict["summarization_total"],
                "correct": metrics_dict["summarization_correct"],
                "accuracy": metrics_dict["summarization_accuracy"],
                "avg_score": metrics_dict["summarization_avg_score"],
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
    
    def get_evaluation_stats(self) -> Dict[str, int]:
        """Get statistics about evaluation methods used.
        
        Returns:
            Dictionary with evaluation statistics
        """
        return dict(self._eval_stats)


# =============================================================================
# Factory Functions
# =============================================================================

def create_locomo_adapter(
    llm_provider=None,
    seed: int = 0,
    use_llm_judge: bool = True,
) -> LOCOMOAdapter:
    """Factory function to create LOCOMO adapter.
    
    Args:
        llm_provider: Optional LLM provider
        seed: Random seed
        use_llm_judge: Whether to use LLM-as-judge evaluation
        
    Returns:
        Configured LOCOMOAdapter instance
    """
    evaluation_method = (
        EvaluationMethod.LLM_JUDGE if use_llm_judge and llm_provider
        else EvaluationMethod.FUZZY
    )
    
    return LOCOMOAdapter(
        llm_provider=llm_provider,
        seed=seed,
        evaluation_method=evaluation_method,
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from aurora.algorithms.aurora_core import AuroraMemory
    from aurora.algorithms.models.config import MemoryConfig
    
    # Create adapter (without LLM for demo)
    adapter = LOCOMOAdapter(seed=42)
    
    # Create sample instance for testing
    sample_instance = BenchmarkInstance(
        id="test_1",
        task_type="qa",
        conversation_history=[
            {"speaker": "user", "text": "我昨天去了北京。"},
            {"speaker": "agent", "text": "北京是中国的首都，有很多历史景点。"},
            {"speaker": "user", "text": "是的，我参观了故宫和长城。"},
            {"speaker": "agent", "text": "故宫是明清两代的皇宫，长城是世界奇迹之一。"},
        ],
        query="用户去了哪个城市？",
        ground_truth="北京",
        reasoning_type="single_hop",
    )
    
    # Create memory and prepare
    config = MemoryConfig(dim=64, max_plots=100)
    memory = AuroraMemory(cfg=config, seed=42)
    
    adapter.prepare_memory(sample_instance.conversation_history, memory)
    
    # Evaluate
    result = adapter.evaluate(sample_instance, memory)
    
    print(f"Instance: {result.instance_id}")
    print(f"Task Type: {result.task_type}")
    print(f"Prediction: {result.prediction}")
    print(f"Ground Truth: {result.ground_truth}")
    print(f"Score: {result.score:.2f}")
    print(f"Correct: {result.is_correct}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Reasoning Trace: {result.reasoning_trace}")
