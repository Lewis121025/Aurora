"""运行时投影模块。

定义状态投影函数。
"""

from __future__ import annotations


def project_state() -> dict[str, str]:
    """投影状态摘要。"""
    return {"status": "ok"}
