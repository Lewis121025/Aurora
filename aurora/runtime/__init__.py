"""
aurora.runtime
运行时包：封装了系统的生命周期管理、配置注入、持久化存储以及与 LLM 的交互流程。
它是系统从“算法库”走向“可用服务”的关键层级。
"""

from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings

__all__ = ["AuroraRuntime", "AuroraSettings"]
