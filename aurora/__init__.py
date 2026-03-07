"""
AURORA 记忆系统
====================

为 AI Agent 设计的叙事优先记忆系统。

哲学理念：
    - 记忆不是关于过去，而是关于身份
    - 自我是必要的虚构
    - 关系定义了我们是谁
    - 某些矛盾应该被保留

快速开始：
    from aurora import AuroraMemory, MemoryConfig

    mem = AuroraMemory(cfg=MemoryConfig(), seed=42)

    # 摄入交互
    plot = mem.ingest("User asked about memory systems")

    # 使用关系上下文查询
    trace = mem.query("How does memory work?", asker_id="user_123")

    # 演化（整合、反思、成长）
    mem.evolve()

模块：
    algorithms - 核心记忆算法（AuroraMemory, TensionManager）
    services   - 生产服务（IngestionService, QueryService）
    storage    - 存储后端（StateStore, VectorStore）
    embeddings - 嵌入提供者（HashEmbedding, BailianEmbedding）
    llm        - LLM 提供者（MockLLM, ArkLLM）
    api        - REST API（FastAPI）
    mcp        - MCP 服务器（Model Context Protocol）
    privacy    - PII 脱敏
"""

__version__ = "0.1.0"

# 核心算法 - AURORA 的心脏
from aurora.algorithms import AuroraMemory, TensionManager

# 数据模型
from aurora.algorithms.models import (
    Plot,
    StoryArc,
    Theme,
    MemoryConfig,
    RetrievalTrace,
)

# 高级集成
from aurora.hub import AuroraHub
from aurora.config import AuroraSettings

# 异常
from aurora.exceptions import (
    AuroraError,
    MemoryNotFoundError,
    ConfigurationError,
    SerializationError,
    EmbeddingError,
    LLMError,
    StorageError,
    ValidationError,
)

__all__ = [
    # 版本
    "__version__",
    # 核心
    "AuroraMemory",
    "TensionManager",
    # 模型
    "Plot",
    "StoryArc",
    "Theme",
    "MemoryConfig",
    "RetrievalTrace",
    # Hub
    "AuroraHub",
    "AuroraSettings",
    # 异常
    "AuroraError",
    "MemoryNotFoundError",
    "ConfigurationError",
    "SerializationError",
    "EmbeddingError",
    "LLMError",
    "StorageError",
    "ValidationError",
]
