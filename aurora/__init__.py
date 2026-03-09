"""Aurora public API.

维护者默认阅读顺序：
1. ``aurora`` 顶层导出
2. ``aurora.runtime.bootstrap`` 运行时装配
3. ``aurora.runtime.runtime`` 单用户运行时
4. ``aurora.core.soul_memory`` 核心记忆引擎
"""

from aurora.version import __version__

# 核心算法 - AURORA 的心脏
from aurora.core import AuroraSoulMemory, TensionManager
from aurora.core.soul_memory import Plot, RetrievalTrace, SoulMemoryConfig, StoryArc, Theme

# 高级集成
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings

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
    "AuroraSoulMemory",
    "TensionManager",
    # 模型
    "Plot",
    "StoryArc",
    "Theme",
    "SoulMemoryConfig",
    "RetrievalTrace",
    "AuroraRuntime",
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
