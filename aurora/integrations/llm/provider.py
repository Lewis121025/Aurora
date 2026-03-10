"""
aurora/integrations/llm/provider.py
大语言模型（LLM）提供者基类模块。
定义了抽象基类 LLMProvider，规定了所有具体模型集成（如 Ark, Bailian）必须实现的通用接口。
支持文本补全、流式输出以及严格遵循 Pydantic Schema 的结构化 JSON 生成。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Type, TypeVar

from pydantic import BaseModel

# 定义泛型 T，约束为 Pydantic 的 BaseModel 子类，用于 JSON Schema 校验
T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """
    LLM 提供者抽象接口。

    设计原则：
      - 必须支持 JSON Schema 约束输出，确保 Agent 逻辑的健壮性。
      - 必须处理网络异常、重试、超时和断路逻辑。
      - 解耦具体的大模型厂商 API，为系统提供统一的调用界面。
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
        """
        生成非流式的文本补全响应。

        参数：
            prompt：用户输入的提示词正文。
            system：可选的系统提示词（System Role），用于设定模型行为。
            temperature：采样温度，控制生成结果的随机性 (0.0 为确定性，1.0 为强随机)。
            max_tokens：本次生成允许消耗的最大 Token 数量。
            timeout_s：HTTP 请求超时时间（秒）。
            stop：可选的停止序列列表，命中其中之一则停止生成。
            metadata：可选的请求元数据字典，用于日志追踪或特定厂商参数传递。
            max_retries：可选的重试次数覆盖值。若为 None，则使用提供者的全局默认配置。

        返回：
            生成后的纯文本字符串。
        """
        raise NotImplementedError

    def stream_complete(
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
    ) -> Iterator[str]:
        """
        流式生成文本补全响应。

        默认实现回退到 `complete()` 方法，将整个文本作为单个 Chunk 返回。
        建议具体提供者重写此方法，以实现真正的 Token 级别流式传输（SSE）。
        """
        text = self.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            stop=stop,
            metadata=metadata,
            max_retries=max_retries,
        )
        if text:
            yield text

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
        """
        生成结构化 JSON 响应并执行 Schema 校验。

        参数：
            system：系统提示词，通常包含对 JSON 格式的严格要求。
            user：用户提示词。
            schema：Pydantic 模型类（Type[BaseModel]），用于对 LLM 返回的内容进行解析和验证。
            temperature：采样温度。
            timeout_s：HTTP 请求超时时间（秒）。
            metadata：可选的请求元数据。
            max_retries：可选的重试次数。

        返回：
            解析并校验通过后的 Pydantic 模型实例（T）。
            若 LLM 返回内容不符合 Schema，实现类应抛出解析异常。
        """
        raise NotImplementedError
