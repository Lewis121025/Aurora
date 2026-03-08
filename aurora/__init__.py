"""Aurora public API.

维护者默认阅读顺序：
1. ``aurora`` 顶层导出
2. ``aurora.runtime.bootstrap`` 运行时装配
3. ``aurora.runtime.runtime`` 单用户运行时
4. ``aurora.core.memory`` 核心记忆引擎
"""

from aurora.version import __version__

# 核心算法 - AURORA 的心脏
from aurora.core import AuroraMemory, TensionManager

# 数据模型
from aurora.core.models import (
    Plot,
    StoryArc,
    Theme,
    MemoryConfig,
    RetrievalTrace,
)

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
    "AuroraMemory",
    "TensionManager",
    # 模型
    "Plot",
    "StoryArc",
    "Theme",
    "MemoryConfig",
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
