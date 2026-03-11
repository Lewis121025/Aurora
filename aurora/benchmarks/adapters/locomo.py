"""
LOCOMO 基准适配器
========================

LOCOMO (ACL 2024) - 长期对话记忆基准的适配器。

数据集: GitHub `snap-research/locomo`
- 32+ 个会话对话
- 每个对话约 300 轮，约 9000 个 token
- 基于角色和时间事件图

评估任务:
1. 问答 (QA)
   - 单跳推理
   - 多跳推理
   - 时间推理
   - 常识推理
   - 世界知识推理

2. 事件总结
   - 生成连贯的事件摘要
   - AURORA 实现: Story/Theme 物化视图 + recent plot synthesis

3. 多模态对话 (可选)
   - 图像共享上下文
   - 初始版本可跳过

设计原则:
- 零硬编码阈值: 可配置的评估参数
- 确定性可重现性: 有种子的随机操作
- 完整的类型注解
- LLM-as-judge 评估支持

使用示例:
    from aurora.benchmarks.adapters.locomo import LOCOMOAdapter, LOCOMOTaskType
    from aurora.soul.engine import AuroraSoul, SoulConfig

    adapter = LOCOMOAdapter(llm_provider=my_llm)
    instances = adapter.load_dataset("path/to/locomo")

    memory = AuroraSoul(cfg=SoulConfig())
    results = []

    for instance in instances:
        # 使用对话历史准备内存
        adapter.prepare_memory(instance.conversation_history, memory)

        # 评估
        result = adapter.evaluate(instance, memory)
        results.append(result)

    # 按推理类型聚合结果
    metrics = adapter.aggregate_results(results)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from aurora.benchmarks.interface import (
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
from aurora.integrations.llm.Prompt.locomo_prompt import (
    LOCOMO_QA_EVALUATION_SYSTEM_PROMPT,
    LOCOMO_QA_EVALUATION_USER_PROMPT,
    LOCOMO_SUMMARIZATION_EVALUATION_SYSTEM_PROMPT,
    LOCOMO_SUMMARIZATION_EVALUATION_USER_PROMPT,
)
from aurora.soul.models import Message, TextPart, messages_to_text
from aurora.integrations.llm.Prompt.qa_prompt import build_qa_prompt

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

logger = logging.getLogger(__name__)


def _text_messages(text: str, *, role: str = "user") -> List[Message]:
    return [Message(role=role, parts=(TextPart(text=str(text)),))]


def _payload_text(payload: Any) -> str:
    semantic_text = getattr(payload, "semantic_text", None)
    if semantic_text:
        return str(semantic_text)
    messages = getattr(payload, "messages", None)
    if messages:
        return messages_to_text(messages)
    if hasattr(payload, "to_narrative_summary"):
        return str(payload.to_narrative_summary())
    return ""


# =============================================================================
# Enums
# =============================================================================


class LOCOMOReasoningType(Enum):
    """LOCOMO QA 任务所需的推理类型。

    每种类型测试不同的内存能力:
    - SINGLE_HOP: 直接事实检索
    - MULTI_HOP: 组合多个事实
    - TEMPORAL: 基于时间的推理
    - COMMONSENSE: 常识推理
    - WORLD_KNOWLEDGE: 外部世界知识
    """

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    COMMONSENSE = "commonsense"
    WORLD_KNOWLEDGE = "world_knowledge"

    @classmethod
    def from_string(cls, s: str) -> "LOCOMOReasoningType":
        """从字符串解析推理类型。"""
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
    """LOCOMO 基准中的任务类型。

    QUESTION_ANSWERING: 回答关于对话历史的问题
    EVENT_SUMMARIZATION: 生成连贯的事件摘要
    DIALOGUE_GENERATION: 生成适当的对话响应 (可选)
    """

    QUESTION_ANSWERING = "qa"
    EVENT_SUMMARIZATION = "summarization"
    DIALOGUE_GENERATION = "dialogue"

    @classmethod
    def from_string(cls, s: str) -> "LOCOMOTaskType":
        """从字符串解析任务类型。"""
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
        """QA 任务的 LLM-as-judge 评估结果。"""

        is_correct: bool = Field(description="Whether the answer is correct")
        score: float = Field(ge=0.0, le=1.0, description="Confidence score [0, 1]")
        explanation: str = Field(description="Brief explanation of the evaluation")

    class SummarizationEvaluationResult(BaseModel):
        """总结任务的 LLM-as-judge 评估结果。"""

        coherence_score: float = Field(ge=0.0, le=1.0, description="Coherence of summary")
        coverage_score: float = Field(ge=0.0, le=1.0, description="Coverage of key events")
        accuracy_score: float = Field(ge=0.0, le=1.0, description="Factual accuracy")
        overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
        missing_events: List[str] = Field(
            default_factory=list, description="Key events not covered"
        )
        explanation: str = Field(description="Brief explanation")
else:
    QAEvaluationResult = None  # type: ignore
    SummarizationEvaluationResult = None  # type: ignore


# =============================================================================
# LOCOMO Data Structures
# =============================================================================


@dataclass
class LOCOMOTurn:
    """LOCOMO 对话中的单个轮次。

    属性:
        turn_id: 唯一的轮次标识符
        speaker: 说话者标识符 (例如, "user1", "user2")
        text: 轮次文本内容
        timestamp: 可选的时间戳
        metadata: 额外的轮次元数据
    """

    turn_id: str
    speaker: str
    text: str
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以供内存摄入。"""
        return {
            "id": self.turn_id,
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
            **self.metadata,
        }


@dataclass
class LOCOMOSession:
    """LOCOMO 对话会话。

    属性:
        session_id: 唯一的会话标识符
        turns: 对话轮次列表
        personas: 参与者的角色信息
        events: 时间事件图
        metadata: 会话元数据
    """

    session_id: str
    turns: List[LOCOMOTurn]
    personas: Dict[str, str] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LOCOMOQuestion:
    """LOCOMO 评估问题。

    属性:
        question_id: 唯一的问题标识符
        session_id: 关联的会话
        question: 问题文本
        answer: 真实答案
        reasoning_type: 所需的推理类型
        evidence_turns: 包含证据的轮次 ID
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
    """LOCOMO 基准评估适配器。

    LOCOMO (长期对话记忆) 是一个 ACL 2024 基准，
    旨在评估长期对话记忆系统。

    主要特性:
    - 多会话对话，约 300 轮，约 9000 个 token
    - 五种推理类型用于 QA 评估
    - 事件总结任务
    - 角色和时间事件图

    AURORA 能力映射:
    - 准确检索: query() + FieldRetriever
    - 多跳推理: 图遍历 + 吸引子追踪
    - 时间推理: 故事时间线 + 时间边
    - 总结: Story/Theme 物化视图 + recent plot synthesis

    属性:
        llm: 用于 LLM-as-judge 评估的可选 LLM 提供者
        evaluation_method: 答案评估方法
        use_story_summary: 是否优先使用 story/theme 视图进行总结

    使用示例:
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
        """返回基准名称。"""
        return "LOCOMO"

    def __init__(
        self,
        llm_provider: Any = None,
        seed: int = 0,
        evaluation_method: EvaluationMethod = EvaluationMethod.LLM_JUDGE,
        use_story_summary: bool = True,
        f1_threshold: float = 0.5,
        fuzzy_threshold: float = 0.7,
    ) -> None:
        """初始化 LOCOMO 适配器。

        参数:
            llm_provider: 用于 LLM-as-judge 评估的可选 LLM 提供者
            seed: 用于可重现性的随机种子
            evaluation_method: 用于评估预测的方法
            use_story_summary: 是否优先使用 story/theme 视图进行总结
            f1_threshold: F1 基础正确性的阈值
            fuzzy_threshold: 模糊匹配的阈值
        """
        super().__init__(llm_provider=llm_provider, seed=seed)

        self.evaluation_method = evaluation_method
        self.use_story_summary = use_story_summary
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
        """从本地路径或远程源加载 LOCOMO 数据集。

        支持多种数据格式:
        - 包含会话和问题的 JSON 文件
        - 包含会话文件的目录
        - GitHub 仓库 URL (将下载)

        参数:
            path: 数据集路径 (本地或 GitHub URL)
            subset: 可选子集 ("train", "test", "val", 或推理类型)
            limit: 可选的实例数量限制

        返回:
            BenchmarkInstance 对象列表
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
        """从单个 JSON 文件加载数据集。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._parse_dataset(data, subset, limit)

    def _load_from_directory(
        self,
        path: Path,
        subset: Optional[str],
        limit: Optional[int],
    ) -> List[BenchmarkInstance]:
        """从文件目录加载数据集。"""
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
        """从 GitHub 仓库加载数据集。

        注意: 需要网络访问。如果可用，将回退到本地缓存。
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
        """将数据集 JSON 解析为 BenchmarkInstance 对象。"""
        instances: List[BenchmarkInstance] = []

        # Parse sessions
        sessions = data.get("sessions", data.get("dialogues", []))
        if isinstance(sessions, dict):
            sessions = list(sessions.values())

        for session_data in sessions:
            parsed_session = self._parse_session(session_data)
            self._session_cache[parsed_session.session_id] = parsed_session

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
            question_session = self._session_cache.get(question.session_id or "")
            if question_session is None:
                logger.warning(f"Session not found for question {question.question_id}")
                continue

            # Create benchmark instance
            instance = BenchmarkInstance(
                id=question.question_id,
                task_type=LOCOMOTaskType.QUESTION_ANSWERING.value,
                conversation_history=[turn.to_dict() for turn in question_session.turns],
                query=question.question,
                ground_truth=question.answer,
                reasoning_type=question.reasoning_type.value,
                session_id=question.session_id,
                metadata={
                    "evidence_turns": question.evidence_turns,
                    "personas": question_session.personas,
                    "events": question_session.events,
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
            summary_session = self._session_cache.get(str(session_id))

            if summary_session is None:
                continue

            instance = BenchmarkInstance(
                id=s_data.get("id", f"summary_{session_id}"),
                task_type=LOCOMOTaskType.EVENT_SUMMARIZATION.value,
                conversation_history=[turn.to_dict() for turn in summary_session.turns],
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
        """将会话数据解析为 LOCOMOSession 对象。"""
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
                    turn_id=str(
                        turn_data.get("id", turn_data.get("turn_id", f"{session_id}_turn_{i}"))
                    ),
                    speaker=str(turn_data.get("speaker", turn_data.get("role", "user"))),
                    text=str(
                        turn_data.get(
                            "text", turn_data.get("content", turn_data.get("utterance", ""))
                        )
                    ),
                    timestamp=turn_data.get("timestamp"),
                    metadata={
                        k: v
                        for k, v in turn_data.items()
                        if k
                        not in (
                            "id",
                            "turn_id",
                            "speaker",
                            "role",
                            "text",
                            "content",
                            "utterance",
                            "timestamp",
                        )
                    },
                )
            else:
                continue

            turns.append(turn)

        return LOCOMOSession(
            session_id=session_id,
            turns=turns,
            personas=data.get("personas", data.get("persona", {})),
            events=data.get("events", data.get("temporal_events", [])),
            metadata={
                k: v
                for k, v in data.items()
                if k
                not in (
                    "session_id",
                    "dialogue_id",
                    "turns",
                    "utterances",
                    "dialogue",
                    "personas",
                    "persona",
                    "events",
                    "temporal_events",
                )
            },
        )

    def _parse_question(self, data: Dict[str, Any]) -> LOCOMOQuestion:
        """将问题数据解析为 LOCOMOQuestion 对象。"""
        reasoning_str = data.get(
            "reasoning_type", data.get("type", data.get("category", "single_hop"))
        )

        return LOCOMOQuestion(
            question_id=data.get("id", data.get("question_id", str(self.rng.integers(10000)))),
            session_id=data.get("session_id", data.get("dialogue_id", "")),
            question=data.get("question", data.get("query", "")),
            answer=data.get("answer", data.get("ground_truth", data.get("response", ""))),
            reasoning_type=LOCOMOReasoningType.from_string(reasoning_str),
            evidence_turns=data.get("evidence_turns", data.get("supporting_turns", [])),
            metadata={
                k: v
                for k, v in data.items()
                if k
                not in (
                    "id",
                    "question_id",
                    "session_id",
                    "dialogue_id",
                    "question",
                    "query",
                    "answer",
                    "ground_truth",
                    "response",
                    "reasoning_type",
                    "type",
                    "category",
                    "evidence_turns",
                    "supporting_turns",
                )
            },
        )

    def _parse_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将 LOCOMO 对话格式解析为轮次列表。

        处理各种 LOCOMO 对话格式。

        参数:
            conversation: 原始对话数据

        返回:
            轮次字典列表
        """
        if isinstance(conversation, list):
            return conversation

        # Handle nested format
        turns = conversation.get(
            "turns", conversation.get("utterances", conversation.get("dialogue", []))
        )

        parsed_turns: List[Dict[str, Any]] = []
        for turn in turns:
            if isinstance(turn, str):
                parsed_turns.append({"text": turn})
            elif isinstance(turn, dict):
                parsed_turns.append(
                    {
                        "id": turn.get("id", turn.get("turn_id")),
                        "speaker": turn.get("speaker", turn.get("role", "user")),
                        "text": turn.get("text", turn.get("content", turn.get("utterance", ""))),
                        "timestamp": turn.get("timestamp"),
                    }
                )

        return parsed_turns

    # -------------------------------------------------------------------------
    # Memory Preparation
    # -------------------------------------------------------------------------

    def _prepare_memory_from_sessions(
        self,
        sessions: List[Dict[str, Any]],
        memory: Any,
    ) -> None:
        """从多个对话会话准备内存。

        对于 LOCOMO 的多会话格式，这会摄入具有适当时间和说话者上下文的轮次。

        参数:
            sessions: 会话数据列表
            memory: AURORA 内存实例
        """
        for session in sessions:
            session_id = session.get("session_id", session.get("dialogue_id"))
            turns = self._parse_conversation(session)

            for turn in turns:
                text = turn.get("text", turn.get("content", str(turn)))
                speaker = turn.get("speaker", turn.get("role", "user"))

                # Format for AURORA ingestion
                if speaker in ("user", "user1", "user2"):
                    messages = _text_messages(f"用户：{text}")
                else:
                    messages = _text_messages(f"助理：{text}", role="assistant")

                memory.ingest(
                    messages=messages,
                    event_id=turn.get("id"),
                    context_messages=_text_messages(f"session:{session_id}", role="system"),
                )

    def prepare_memory(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """通过摄入对话历史来准备内存。

        覆盖基类实现以适应 LOCOMO 特定格式。

        参数:
            conversation_history: 对话轮次列表
            memory: AURORA 内存实例
        """
        if len(args) >= 2 and isinstance(args[0], list):
            conversation_history = cast(List[Dict[str, Any]], args[0])
            memory = args[1]
        elif len(args) >= 2:
            memory = args[0]
            conversation_history = cast(List[Dict[str, Any]], args[1])
        else:
            conversation_history = cast(
                List[Dict[str, Any]], kwargs.get("conversation_history", [])
            )
            memory = kwargs.get("memory")

        if memory is None:
            raise ValueError("memory is required")

        for turn in conversation_history:
            text = turn.get("text", turn.get("content", str(turn)))
            speaker = turn.get("speaker", turn.get("role", "user"))

            # Format for AURORA ingestion
            if speaker in ("user", "user1", "user2"):
                messages = _text_messages(f"用户：{text}")
            else:
                messages = _text_messages(f"助理：{text}", role="assistant")

            memory.ingest(
                messages=messages,
                event_id=turn.get("id"),
            )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        instance: BenchmarkInstance,
        memory: Any,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """评估单个基准实例。

        根据实例类型路由到特定任务的评估。

        参数:
            instance: 要评估的基准实例
            memory: AURORA 内存实例
            **kwargs: 额外的评估参数

        返回:
            包含评估结果的 BenchmarkResult
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
        memory: Any,
        reasoning_type: LOCOMOReasoningType,
    ) -> BenchmarkResult:
        """评估问答任务。

        使用 AURORA 的查询系统检索相关记忆
        并基于检索的上下文生成答案。

        参数:
            instance: QA 基准实例
            memory: AURORA 内存实例
            reasoning_type: 所需的推理类型

        返回:
            包含 QA 评估的 BenchmarkResult
        """
        start_time = time.perf_counter()
        reasoning_trace: List[str] = []

        try:
            # Query memory for relevant context
            trace = memory.query(_text_messages(instance.query), k=10)
            reasoning_trace.append(f"Retrieved {len(trace.ranked)} memories")

            # Gather retrieved content
            retrieved_texts: List[str] = []
            for nid, score, kind in trace.ranked[:5]:
                try:
                    payload = memory.graph.payload(nid)
                    text = _payload_text(payload)
                    if text:
                        retrieved_texts.append(text)
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
        """使用 LLM 生成 QA 答案。

        参数:
            question: 要回答的问题
            context: 从内存检索的上下文
            reasoning_type: 所需的推理类型

        返回:
            生成的答案字符串
        """
        if self.llm is None:
            return context[0] if context else ""

        # Map LOCOMO reasoning type to question type hint
        reasoning_to_qtype = {
            LOCOMOReasoningType.SINGLE_HOP: None,  # Use default
            LOCOMOReasoningType.MULTI_HOP: "multi-session",  # Multi-hop often requires aggregation
            LOCOMOReasoningType.TEMPORAL: "temporal-reasoning",
            LOCOMOReasoningType.COMMONSENSE: None,  # Use default
            LOCOMOReasoningType.WORLD_KNOWLEDGE: None,  # Use default
        }

        qtype_hint = reasoning_to_qtype.get(reasoning_type)
        context_text = "\n\n".join(context) if context else "No relevant context found."

        # Use type-specific prompt template
        prompt = build_qa_prompt(
            question=question,
            context=context_text,
            question_type_hint=qtype_hint,
            max_context_length=5000,
        )

        judge_system_prompt = """You are a precise question-answering assistant.
Answer the question based on the provided conversation context.
Be concise and answer directly."""

        try:
            # Use LLM to generate answer with type-specific prompt
            from pydantic import BaseModel, Field

            class QAAnswer(BaseModel):
                answer: str = Field(description="The answer to the question")

            result = self.llm.complete_json(
                messages=[
                    Message(role="system", parts=(TextPart(text=judge_system_prompt),)),
                    Message(role="user", parts=(TextPart(text=prompt),)),
                ],
                schema=QAAnswer,
                temperature=0.1,
            )
            return str(result.answer)

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to context")
            return context[0] if context else ""

    def _evaluate_qa_answer(
        self,
        prediction: str,
        ground_truth: str,
        question: str,
    ) -> Tuple[float, bool]:
        """根据真实答案评估 QA 答案。

        参数:
            prediction: 模型的答案
            ground_truth: 预期答案
            question: 原始问题

        返回:
            (分数, 是否正确) 的元组
        """
        self._eval_stats["total_evals"] += 1

        # Try LLM-as-judge if available and configured
        if (
            self.evaluation_method == EvaluationMethod.LLM_JUDGE
            and self.llm is not None
            and QAEvaluationResult is not None
        ):
            try:
                result = self.llm.complete_json(
                    messages=[
                        Message(
                            role="system",
                            parts=(TextPart(text=LOCOMO_QA_EVALUATION_SYSTEM_PROMPT),),
                        ),
                        Message(
                            role="user",
                            parts=(
                                TextPart(
                                    text=LOCOMO_QA_EVALUATION_USER_PROMPT.format(
                                        question=question,
                                        ground_truth=ground_truth,
                                        prediction=prediction,
                                    )
                                ),
                            ),
                        ),
                    ],
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
        memory: Any,
    ) -> BenchmarkResult:
        """评估事件总结任务。

        使用 AURORA 当前 story/theme 视图生成故事摘要
        并与真实答案进行比较。

        参数:
            instance: 总结基准实例
            memory: AURORA 内存实例

        返回:
            包含总结评估的 BenchmarkResult
        """
        start_time = time.perf_counter()
        reasoning_trace: List[str] = []

        try:
            # Generate summary using AURORA
            if self.use_story_summary:
                prediction = self._generate_story_summary(memory, instance)
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

    def _generate_story_summary(
        self,
        memory: Any,
        instance: BenchmarkInstance,
    ) -> str:
        """使用 AURORA 当前 story/theme 视图生成摘要。

        参数:
            memory: AURORA 内存实例
            instance: 基准实例

        返回:
            生成的摘要文本
        """
        parts: List[str] = []

        for theme in getattr(memory, "themes", {}).values():
            label = (
                getattr(theme, "label", "")
                or getattr(theme, "name", "")
                or getattr(theme, "description", "")
            )
            if label:
                parts.append(str(label))

        for story in getattr(memory, "stories", {}).values():
            plot_ids = list(getattr(story, "plot_ids", []))
            plot_texts = [
                _payload_text(memory.plots[plot_id])
                for plot_id in plot_ids
                if hasattr(memory, "plots") and plot_id in memory.plots
            ]
            if plot_texts:
                parts.append(" ".join(plot_texts[:3]))

        if parts:
            return "\n".join(parts[:5])

        return self._generate_simple_summary(memory, instance)

    def _generate_simple_summary(
        self,
        memory: Any,
        instance: BenchmarkInstance,
    ) -> str:
        """使用 recent plots 生成简单摘要。

        参数:
            memory: AURORA 内存实例
            instance: 基准实例

        返回:
            生成的摘要文本
        """
        summaries: List[str] = []

        if not summaries:
            recent_plots = sorted(
                memory.plots.values(),
                key=lambda p: p.ts,
                reverse=True,
            )[:10]

            summaries = [_payload_text(p)[:200] for p in recent_plots if _payload_text(p)]

        return "\n\n".join(summaries) if summaries else "No events to summarize."

    def _evaluate_summary(
        self,
        prediction: str,
        ground_truth: str,
    ) -> Tuple[float, Dict[str, float]]:
        """根据真实答案评估摘要。

        参数:
            prediction: 生成的摘要
            ground_truth: 预期摘要

        返回:
            (总体分数, 指标字典) 的元组
        """
        self._eval_stats["total_evals"] += 1

        # Try LLM-as-judge if available
        if (
            self.evaluation_method == EvaluationMethod.LLM_JUDGE
            and self.llm is not None
            and SummarizationEvaluationResult is not None
        ):
            try:
                result = self.llm.complete_json(
                    messages=[
                        Message(
                            role="system",
                            parts=(TextPart(text=LOCOMO_SUMMARIZATION_EVALUATION_SYSTEM_PROMPT),),
                        ),
                        Message(
                            role="user",
                            parts=(
                                TextPart(
                                    text=LOCOMO_SUMMARIZATION_EVALUATION_USER_PROMPT.format(
                                        ground_truth=ground_truth,
                                        prediction=prediction,
                                    )
                                ),
                            ),
                        ),
                    ],
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
        memory: Any,
    ) -> BenchmarkResult:
        """评估对话生成任务 (可选)。

        参数:
            instance: 对话基准实例
            memory: AURORA 内存实例

        返回:
            包含对话评估的 BenchmarkResult
        """
        # Dialogue generation is optional for initial version
        # Implement basic response relevance evaluation

        start_time = time.perf_counter()

        try:
            # Query memory for context
            trace = memory.query(_text_messages(instance.query), k=5)

            # Generate simple response
            context_texts = []
            for nid, _, _ in trace.ranked[:3]:
                try:
                    payload = memory.graph.payload(nid)
                    text = _payload_text(payload)
                    if text:
                        context_texts.append(text)
                except Exception as e:
                    logger.debug(f"Failed to retrieve payload for node {nid}: {e}")

            prediction = (
                context_texts[0] if context_texts else "I don't have enough context to respond."
            )

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
        """使用 LOCOMO 特定的分解聚合结果。

        提供按以下方式分解的指标:
        - 任务类型 (QA, 总结, 对话)
        - 推理类型 (单跳, 多跳, 时间等)

        也通过 get_evaluation_metrics() 返回 EvaluationMetrics 对象。

        参数:
            results: BenchmarkResult 对象列表

        返回:
            将指标名称映射到值的字典
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
                r for r in qa_results if r.metadata.get("reasoning_type") == reasoning_type.value
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
        summary_results = [
            r for r in results if r.task_type == LOCOMOTaskType.EVENT_SUMMARIZATION.value
        ]
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

    def get_evaluation_metrics(
        self, results: Optional[List[BenchmarkResult]] = None
    ) -> EvaluationMetrics:
        """从结果获取 EvaluationMetrics 对象。

        参数:
            results: 结果列表 (如果为 None 则使用最后评估的)

        返回:
            具有详细分解的 EvaluationMetrics 对象
        """
        if results is None:
            results = getattr(self, "_last_results", [])

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
        """获取关于所使用评估方法的统计信息。

        返回:
            包含评估统计信息的字典
        """
        return dict(self._eval_stats)


# =============================================================================
# Factory Functions
# =============================================================================


def create_locomo_adapter(
    llm_provider: Any = None,
    seed: int = 0,
    use_llm_judge: bool = True,
) -> LOCOMOAdapter:
    """创建 LOCOMO 适配器的工厂函数。

    参数:
        llm_provider: 可选的 LLM 提供者
        seed: 随机种子
        use_llm_judge: 是否使用 LLM-as-judge 评估

    返回:
        配置好的 LOCOMOAdapter 实例
    """
    evaluation_method = (
        EvaluationMethod.LLM_JUDGE if use_llm_judge and llm_provider else EvaluationMethod.FUZZY
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
    from aurora.soul.engine import AuroraSoul, SoulConfig

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
    config = SoulConfig(dim=64, max_plots=100)
    memory = AuroraSoul(cfg=config, seed=42)

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
