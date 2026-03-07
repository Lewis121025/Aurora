"""AURORA 自定义异常。

该模块为 AURORA 内存系统定义了一个自定义异常的层次结构。
所有异常都继承自 AuroraError，以便在需要时进行广泛的捕获。
"""


class AuroraError(Exception):
    """所有 AURORA 错误的基础异常。"""

    pass


class MemoryNotFoundError(AuroraError):
    """内存元素（Plot、Story、Theme）未找到。

    Args:
        kind: 内存元素的类型（'plot'、'story'、'theme'）
        element_id: 未找到的 ID
    """

    def __init__(self, kind: str, element_id: str):
        self.kind = kind
        self.element_id = element_id
        super().__init__(f"{kind} not found: {element_id}")


class ConfigurationError(AuroraError):
    """无效的配置。"""

    pass


class SerializationError(AuroraError):
    """序列化或反序列化失败。"""

    pass


class EmbeddingError(AuroraError):
    """Embedding 生成失败。"""

    pass


class LLMError(AuroraError):
    """LLM 调用失败。"""

    pass


class StorageError(AuroraError):
    """存储操作失败。"""

    pass


class ValidationError(AuroraError):
    """输入验证失败。"""

    pass
