"""
AURORA 模型基类
=========================

所有 AURORA 数据模型的基类和混入类。

提供：
- SerializableMixin: 所有模型的统一序列化接口
- TimestampedMixin: 自动时间戳管理
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Type, TypeVar

import numpy as np


T = TypeVar("T")


class SerializableMixin(ABC):
    """
    为数据类提供统一序列化接口的混入类。

    所有 AURORA 模型应该一致地实现这些方法：
    - to_state_dict(): 转换为 JSON 兼容的字典
    - from_state_dict(): 从状态字典重构

    这确保了所有模型类型的序列化一致性，
    实现了可靠的持久化和状态转移。
    """
    
    @abstractmethod
    def to_state_dict(self) -> Dict[str, Any]:
        """
        将模型序列化为 JSON 兼容的字典。

        实现说明：
        - np.ndarray 应通过 .tolist() 转换为列表
        - 嵌套的 SerializableMixin 对象应调用其 to_state_dict()
        - 为 None 的可选字段应包含为 None

        返回：
            表示模型状态的 JSON 兼容字典
        """
        pass

    @classmethod
    @abstractmethod
    def from_state_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        从状态字典重构模型实例。

        实现说明：
        - 列表应在适当时转换为 np.ndarray
        - 嵌套对象应通过其 from_state_dict() 重构
        - 缺失的键应使用合理的默认值

        参数：
            d: 来自 to_state_dict() 的状态字典

        返回：
            重构的模型实例
        """
        pass


def serialize_value(value: Any) -> Any:
    """
    将单个值序列化为 JSON 兼容格式的辅助函数。

    处理：
    - np.ndarray -> 列表
    - SerializableMixin -> 字典
    - 基本类型 -> 原样返回
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, 'to_state_dict'):
        return value.to_state_dict()
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    else:
        return value


def auto_serialize(obj: Any) -> Dict[str, Any]:
    """
    自动将数据类序列化为字典。

    使用反射来序列化所有字段。
    通过实现自定义 to_state_dict() 来覆盖特定字段。

    参数：
        obj: 数据类实例

    返回：
        JSON 兼容的字典
    """
    if not is_dataclass(obj):
        raise TypeError(f"auto_serialize requires a dataclass, got {type(obj)}")

    result = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        result[f.name] = serialize_value(value)
    return result
