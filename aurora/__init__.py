"""Aurora package.

直接从具体模块导入实现；顶层仅保留版本信息。
"""

from aurora.system.version import __version__

__all__ = ["__version__"]
