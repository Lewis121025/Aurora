"""Embedding 实用函数。"""

from typing import Any, Optional

import numpy as np


def get_embedding_from_object(obj: Any) -> Optional[np.ndarray]:
    """从各种对象类型中提取 embedding。

    尝试从 'embedding'、'centroid' 或 'prototype' 属性获取 embedding。
    处理列表到 ndarray 的转换。

    Args:
        obj: 可能具有 embedding 属性的对象。

    Returns:
        作为 numpy 数组的 embedding，如果未找到则返回 None。
    """
    for attr in ("embedding", "centroid", "prototype"):
        val = getattr(obj, attr, None)
        if val is not None:
            if isinstance(val, list):
                return np.array(val, dtype=np.float32)
            return val
    return None
