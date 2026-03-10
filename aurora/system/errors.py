"""AURORA 自定义异常。"""


class AuroraError(Exception):
    """所有 AURORA 错误的基础异常。"""

    pass


class ConfigurationError(AuroraError):
    """无效的配置。"""

    pass
