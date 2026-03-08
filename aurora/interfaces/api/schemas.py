"""
Aurora API 模式 (版本化)
========================

为API契约提供版本化的Pydantic模型。

版本历史:
- v1: 初始稳定API (当前)

用法:
    from aurora.interfaces.api.schemas import IngestRequestV1, QueryResponseV1

    request = IngestRequestV1(
        event_id="evt_123",
        user_id="user_456",
        ...
    )

模式版本控制允许:
- 版本内向后兼容的更改
- 多个版本在迁移期间共存
- 前端/后端之间的清晰API契约
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class MemoryKind(str, Enum):
    """记忆单元的类型。"""
    PLOT = "plot"
    STORY = "story"
    THEME = "theme"


class PlotStatus(str, Enum):
    """情节生命周期状态。"""
    ACTIVE = "active"
    ABSORBED = "absorbed"
    ARCHIVED = "archived"


class StoryStatus(str, Enum):
    """故事生命周期状态。"""
    DEVELOPING = "developing"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ThemeType(str, Enum):
    """主题的类型。"""
    PATTERN = "pattern"
    LESSON = "lesson"
    PREFERENCE = "preference"
    CAUSALITY = "causality"
    CAPABILITY = "capability"
    LIMITATION = "limitation"


# =============================================================================
# V1 Request Models
# =============================================================================

class IngestRequestV1(BaseModel):
    """V1 摄入请求 - 在记忆中存储新的交互。"""
    
    event_id: Optional[str] = Field(
        default=None,
        description="唯一的事件标识符。如果未提供则自动生成。",
        examples=["evt_abc123"],
    )
    user_id: str = Field(
        ...,
        description="用户标识符，用于多租户隔离。",
        examples=["user_456"],
    )
    session_id: str = Field(
        ...,
        description="会话标识符，用于分组相关交互。",
        examples=["sess_789"],
    )
    user_message: str = Field(
        ...,
        description="用户的消息或输入。",
        min_length=1,
        examples=["How do I implement a memory system?"],
    )
    agent_message: str = Field(
        ...,
        description="代理的响应或采取的行动。",
        min_length=1,
        examples=["You can use a narrative memory architecture..."],
    )
    actors: Optional[List[str]] = Field(
        default=None,
        description="交互中的参与者列表。",
        examples=[["user", "agent"]],
    )
    context: Optional[str] = Field(
        default=None,
        description="关于交互的附加上下文。",
        examples=["programming discussion"],
    )
    ts: Optional[float] = Field(
        default=None,
        description="交互的Unix时间戳。如果未提供则自动设置。",
    )
    
    class Config:
        json_schema_extra = {
            "version": "v1",
            "example": {
                "event_id": "evt_abc123",
                "user_id": "user_456",
                "session_id": "sess_789",
                "user_message": "How do I implement a memory system?",
                "agent_message": "You can use a narrative memory architecture...",
                "actors": ["user", "agent"],
                "context": "programming discussion",
            }
        }


class QueryRequestV1(BaseModel):
    """V1 查询请求 - 按语义相似性搜索记忆。"""
    
    user_id: str = Field(
        ...,
        description="用户标识符，用于多租户隔离。",
    )
    text: str = Field(
        ...,
        description="自然语言搜索查询。",
        min_length=1,
        examples=["How do I avoid hard-coded thresholds?"],
    )
    k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="要返回的结果数量。",
    )
    kinds: Optional[List[MemoryKind]] = Field(
        default=None,
        description="按记忆类型过滤。None表示所有类型。",
        examples=[["plot", "story"]],
    )
    include_metadata: bool = Field(
        default=False,
        description="在响应中包含详细元数据。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class FeedbackRequestV1(BaseModel):
    """V1 反馈请求 - 提供关于检索质量的反馈。"""
    
    user_id: str = Field(
        ...,
        description="用户标识符。",
    )
    query_text: str = Field(
        ...,
        description="原始搜索查询。",
    )
    chosen_id: str = Field(
        ...,
        description="被选择/有用的记忆的ID。",
    )
    success: bool = Field(
        ...,
        description="检索是否成功/有帮助。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class EvolveRequestV1(BaseModel):
    """V1 演化请求 - 触发记忆演化。"""
    
    user_id: str = Field(
        ...,
        description="User identifier.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class CausalChainRequestV1(BaseModel):
    """V1 因果链请求 - 获取因果关系。"""
    
    user_id: str = Field(
        ...,
        description="用户标识符。",
    )
    node_id: str = Field(
        ...,
        description="要追踪因果链的节点ID。",
    )
    direction: Literal["ancestors", "descendants"] = Field(
        default="ancestors",
        description="追踪的方向。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


# =============================================================================
# V1 Response Models
# =============================================================================

class IngestResponseV1(BaseModel):
    """V1 摄入响应 - 存储交互的结果。"""

    event_id: str = Field(
        ...,
        description="事件标识符 (输入或生成)。",
    )
    plot_id: str = Field(
        ...,
        description="创建的情节的ID。",
    )
    story_id: Optional[str] = Field(
        default=None,
        description="分配的故事的ID (如果已编码)。",
    )
    encoded: bool = Field(
        ...,
        description="情节是否被存储 (门决策)。",
    )
    tension: float = Field(
        ...,
        description="叙事张力分数。",
    )
    surprise: float = Field(
        ...,
        description="惊喜分数 (信息新颖性)。",
    )
    pred_error: float = Field(
        ...,
        description="与故事模型的预测误差。",
    )
    redundancy: float = Field(
        ...,
        description="与现有情节的冗余度。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class QueryHitV1(BaseModel):
    """V1 查询命中 - 单个搜索结果。"""

    id: str = Field(
        ...,
        description="记忆单元ID。",
    )
    kind: MemoryKind = Field(
        ...,
        description="记忆单元的类型。",
    )
    score: float = Field(
        ...,
        description="相关性分数。",
    )
    snippet: str = Field(
        ...,
        description="文本片段或摘要。",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="如果请求则为附加元数据。",
    )


class QueryResponseV1(BaseModel):
    """V1 查询响应 - 搜索结果。"""

    query: str = Field(
        ...,
        description="原始搜索查询。",
    )
    attractor_path_len: int = Field(
        ...,
        description="吸引子收敛路径的长度。",
    )
    hits: List[QueryHitV1] = Field(
        ...,
        description="匹配的记忆列表。",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="查询延迟 (毫秒)。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class CoherenceResponseV1(BaseModel):
    """V1 一致性响应 - 记忆一致性检查结果。"""

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="总体一致性分数 (0-1)。",
    )
    conflict_count: int = Field(
        ...,
        ge=0,
        description="检测到的冲突数量。",
    )
    unfinished_story_count: int = Field(
        ...,
        ge=0,
        description="未解决的故事数量。",
    )
    recommendations: List[str] = Field(
        ...,
        description="改进一致性的推荐行动。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class SelfNarrativeResponseV1(BaseModel):
    """V1 自我叙事响应 - 代理的自我模型。"""

    identity_statement: str = Field(
        ...,
        description="简短的身份陈述。",
    )
    identity_narrative: str = Field(
        ...,
        description="详细的身份叙事。",
    )
    capability_narrative: str = Field(
        ...,
        description="能力描述。",
    )
    core_values: List[str] = Field(
        ...,
        description="核心价值观列表。",
    )
    coherence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="叙事一致性分数。",
    )
    capabilities: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="具有概率的能力信念。",
    )
    relationships: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="关系信念。",
    )
    unresolved_tensions: List[str] = Field(
        ...,
        description="当前未解决的张力。",
    )
    full_narrative: str = Field(
        ...,
        description="完整的叙事文本。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class CausalChainItemV1(BaseModel):
    """V1 因果链项 - 因果链中的一个节点。"""

    node_id: str
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="因果关系强度。",
    )


class CausalChainResponseV1(BaseModel):
    """V1 因果链响应 - 因果关系链。"""

    chain: List[CausalChainItemV1] = Field(
        ...,
        description="因果相关节点的列表。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class MemoryStatsResponseV1(BaseModel):
    """V1 记忆统计响应 - 记忆系统统计。"""

    plot_count: int = Field(
        ...,
        ge=0,
        description="存储的情节数量。",
    )
    story_count: int = Field(
        ...,
        ge=0,
        description="故事数量。",
    )
    theme_count: int = Field(
        ...,
        ge=0,
        description="主题数量。",
    )
    coherence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="总体一致性分数。",
    )
    self_narrative_coherence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="自我叙事一致性。",
    )
    gate_pass_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="记忆门通过率。",
    )
    cluster_entropy: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="故事集群熵。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class ErrorResponseV1(BaseModel):
    """V1 错误响应 - 标准错误格式。"""

    error: str = Field(
        ...,
        description="错误类型或代码。",
    )
    message: str = Field(
        ...,
        description="人类可读的错误消息。",
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="附加错误详情。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class HealthResponseV1(BaseModel):
    """V1 健康响应 - 服务健康检查。"""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="服务状态。",
    )
    version: str = Field(
        ...,
        description="服务版本。",
    )
    timestamp: float = Field(
        ...,
        description="检查时间戳。",
    )
    components: Optional[Dict[str, str]] = Field(
        default=None,
        description="组件级别的状态。",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


# =============================================================================
# 类型别名便利
# =============================================================================

# 当前版本别名 (发布新版本时更新这些)
IngestRequest = IngestRequestV1
IngestResponse = IngestResponseV1
QueryRequest = QueryRequestV1
QueryResponse = QueryResponseV1
QueryHit = QueryHitV1
FeedbackRequest = FeedbackRequestV1
EvolveRequest = EvolveRequestV1
CoherenceResponse = CoherenceResponseV1
SelfNarrativeResponse = SelfNarrativeResponseV1
CausalChainRequest = CausalChainRequestV1
CausalChainResponse = CausalChainResponseV1
MemoryStatsResponse = MemoryStatsResponseV1
ErrorResponse = ErrorResponseV1
HealthResponse = HealthResponseV1


# =============================================================================
# 模式版本信息
# =============================================================================

SCHEMA_VERSION = "v1"
SCHEMA_VERSIONS = ["v1"]

def get_schema_version() -> str:
    """获取当前模式版本。"""
    return SCHEMA_VERSION

def list_schema_versions() -> List[str]:
    """列出所有可用的模式版本。"""
    return SCHEMA_VERSIONS.copy()
