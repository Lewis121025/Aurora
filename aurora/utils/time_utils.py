"""
AURORA 时间工具
=====================

时间相关的实用函数。
支持用于确定性测试的模拟。
"""

from __future__ import annotations

import time
from typing import Callable, Optional

# 用于测试的可选模拟函数
_mock_time: Optional[Callable[[], float]] = None


def now_ts() -> float:
    """
    获取当前时间戳（自纪元以来的秒数）。

    返回：
        当前时间为浮点数（秒.微秒）

    注意：
        可以使用 set_mock_time() 为测试进行模拟
    """
    if _mock_time is not None:
        return _mock_time()
    return time.time()


def set_mock_time(mock_fn: Optional[Callable[[], float]]) -> None:
    """
    为测试设置模拟时间函数。

    参数：
        mock_fn: 返回模拟时间戳的函数，或 None 以禁用模拟

    示例：
        >>> counter = [0.0]
        >>> set_mock_time(lambda: (counter[0] := counter[0] + 1.0))
        >>> now_ts()
        1.0
        >>> now_ts()
        2.0
        >>> set_mock_time(None)  # 恢复真实时间
    """
    global _mock_time
    _mock_time = mock_fn
