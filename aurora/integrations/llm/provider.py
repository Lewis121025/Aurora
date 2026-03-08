from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """LLM 提供者接口。

    生产环境要求：
      - 实现 JSON schema 约束输出（工具调用 / response_format JSON）
      - 强制执行最大令牌数 / 重试 / 退避 / 断路器

    提供两个主要方法：
      - complete()：用于文本生成
      - complete_json()：用于结构化 JSON 输出和 schema 验证
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        """为给定的提示生成文本补全。

        参数：
            prompt：要补全的用户提示
            system：可选的系统消息
            temperature：采样温度 (0.0-1.0)
            max_tokens：要生成的最大令牌数
            timeout_s：请求超时时间（秒）
            stop：可选的停止序列
            metadata：可选的请求元数据
            max_retries：可选的重试次数覆盖值。如果为 None，则
                使用提供者默认配置。

        返回：
            生成的文本字符串
        """
        raise NotImplementedError

    @abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float = 0.2,
        timeout_s: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> T:
        """生成符合 schema 的结构化 JSON 输出。

        参数：
            system：系统消息
            user：用户消息
            schema：用于响应验证的 Pydantic 模型类
            temperature：采样温度 (0.0-1.0)
            timeout_s：请求超时时间（秒）
            metadata：可选的请求元数据
            max_retries：可选的重试次数覆盖值。如果为 None，则
                使用提供者默认配置。

        返回：
            解析并验证的响应，作为 schema 实例
        """
        raise NotImplementedError
