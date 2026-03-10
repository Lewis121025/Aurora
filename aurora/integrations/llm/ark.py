"""
aurora/integrations/llm/ark.py
字节跳动火山方舟 (Volcengine Ark) 大语言模型提供者模块。
本模块实现了生产级的 LLM 接入逻辑，支持结构化 JSON 输出、自动重试、令牌统计以及错误处理。
它基于 OpenAI 兼容协议，专门针对豆包 (Doubao) 系列模型进行了优化。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .provider import LLMProvider

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class ArkLLM(LLMProvider):
    """
    火山方舟 LLM 提供者，支持结构化输出。

    主要功能：
    - JSON Schema 约束输出（利用 OpenAI 的 response_format）。
    - 带有指数退避策略的自动重试机制。
    - 全局 Token 消耗统计。
    - 健壮的 JSON 解析逻辑，处理 LLM 返回中的 Markdown 代码块或嵌套结构。
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "doubao-1-5-pro-32k-250115",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        初始化火山方舟提供者。
        
        参数：
            api_key：火山引擎 API Key。
            model：要调用的模型端点 ID。
            base_url：API 基础地址。
            max_retries：最大重试次数。
            timeout：请求超时时间（秒）。
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._total_tokens_used = 0
        
    def _get_client(self):
        """
        延迟初始化 OpenAI 客户端。
        Ark API 完全兼容 OpenAI 协议，因此可以直接使用 openai 官方 SDK。
        """
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0, # 禁用 SDK 默认重试，使用我们自定义的退避逻辑
                )
            except ImportError:
                raise ImportError(
                    "请安装 openai 包: pip install openai"
                )
        return self._client

    def _request_options(self, *, structured: bool) -> Dict[str, Any]:
        """返回特定于提供者的额外请求参数。可在子类（如 BailianLLM）中重写。"""
        return {}

    def _json_max_tokens(self, schema: Type[T]) -> int:
        """为结构化 JSON 输出提供保守的 Token 上限。"""
        return 512
    
    def _build_json_schema(self, schema: Type[T]) -> Dict[str, Any]:
        """将 Pydantic 模型转换为 OpenAI 规范的 JSON Schema 响应格式。"""
        json_schema = schema.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": json_schema,
                "strict": True, # 开启严格模式
            }
        }
    
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
        生成文本补全。
        执行带有指数退避（Exponential Backoff）的同步阻塞调用。
        """
        client = self._get_client()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        last_error = None
        attempt_limit = self.max_retries if max_retries is None else max(1, int(max_retries))
        for attempt in range(attempt_limit):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout_s,
                }
                kwargs.update(self._request_options(structured=False))
                if stop:
                    kwargs["stop"] = stop
                
                response = client.chat.completions.create(**kwargs)

                # 统计 Token 使用量
                if hasattr(response, 'usage') and response.usage:
                    self._total_tokens_used += response.usage.total_tokens
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                logger.info(f"第 {attempt + 1}/{attempt_limit} 次尝试失败: {e}")
                if attempt < attempt_limit - 1:
                    # 指数退避策略
                    sleep_time = (2 ** attempt) * 0.5
                    time.sleep(sleep_time)

        raise RuntimeError(f"所有 {attempt_limit} 次重试均失败。最后一次错误为: {last_error}")

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
    ):
        """流式文本生成。通过迭代器逐块返回生成的字符串。"""
        client = self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        attempt_limit = self.max_retries if max_retries is None else max(1, int(max_retries))
        for attempt in range(attempt_limit):
            emitted = False # 记录是否已经开始输出，若已输出则不重试
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout_s,
                    "stream": True,
                }
                kwargs.update(self._request_options(structured=False))
                if stop:
                    kwargs["stop"] = stop

                stream = client.chat.completions.create(**kwargs)
                for chunk in stream:
                    piece = ""
                    if chunk.choices:
                        delta = getattr(chunk.choices[0], "delta", None)
                        content = getattr(delta, "content", None)
                        if isinstance(content, str):
                            piece = content
                        elif isinstance(content, list): # 处理某些特殊格式的 chunk
                            parts = []
                            for item in content:
                                text = getattr(item, "text", None)
                                if text:
                                    parts.append(str(text))
                            piece = "".join(parts)
                    if piece:
                        emitted = True
                        yield piece
                return
            except Exception as e:
                last_error = e
                logger.info(f"流式交互第 {attempt + 1}/{attempt_limit} 次尝试失败: {e}")
                if emitted:
                    # 如果已经输出了部分内容，通常不建议在中间重试，因为会造成输出截断或重复
                    break
                if attempt < attempt_limit - 1:
                    sleep_time = (2 ** attempt) * 0.5
                    time.sleep(sleep_time)

        raise RuntimeError(f"流式交互所有 {attempt_limit} 次重试均失败。最后一次错误为: {last_error}")

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
        生成符合指定 Pydantic Schema 的结构化 JSON 数据。
        核心逻辑：
        1. 注入强制 JSON 约束的系统提示词。
        2. 启用 response_format JSON Schema 模式。
        3. 处理可能包含 Markdown 代码块的 LLM 原始响应。
        4. 执行严格的 Schema 校验。
        """
        client = self._get_client()
        messages = [
            {"role": "system", "content": f"{system}\n\n只返回符合给定 Schema 的 JSON 数据。"},
            {"role": "user", "content": user},
        ]
        
        last_error = None
        attempt_limit = self.max_retries if max_retries is None else max(1, int(max_retries))
        for attempt in range(attempt_limit):
            try:
                response = client.chat.completions.create(
                    **{
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": self._json_max_tokens(schema),
                        "timeout": timeout_s,
                        "response_format": self._build_json_schema(schema),
                        **self._request_options(structured=True),
                    }
                )
                
                content = response.choices[0].message.content

                # Token 统计
                if hasattr(response, 'usage') and response.usage:
                    self._total_tokens_used += response.usage.total_tokens

                # 解析响应中的 JSON 内容（可能带有冗余文本或 Markdown 标记）
                data = self._parse_json_response(content, schema)
                
                # 执行 Pydantic 校验
                return schema.model_validate(data)
                
            except Exception as e:
                last_error = e
                logger.info(f"JSON 提取第 {attempt + 1}/{attempt_limit} 次尝试失败: {e}")
                if attempt < attempt_limit - 1:
                    sleep_time = (2 ** attempt) * 0.5
                    time.sleep(sleep_time)

        raise RuntimeError(f"结构化 JSON 生成所有 {attempt_limit} 次重试均失败。最后一次错误为: {last_error}")
    
    def _parse_json_response(self, content: str, schema: Type[T]) -> Dict[str, Any]:
        """
        从 LLM 的响应字符串中提取 JSON 内容的多重兜底策略。
        """
        import re

        # 策略 1：尝试全量直接解析
        try:
            data = json.loads(content)
            # 检查数据是否被错误地包装在以 Schema 类名命名的顶层键中
            schema_name = schema.__name__
            if isinstance(data, dict):
                if schema_name in data:
                    return data[schema_name]
                # 检查忽略大小写的匹配
                for key in list(data.keys()):
                    if key.lower() == schema_name.lower():
                        return data[key]
                    # 处理 LLM 偶尔返回单键字典（键名不匹配但内容正确）的情况
                    if len(data) == 1 and isinstance(data[key], dict):
                        inner = data[key]
                        required = schema.model_json_schema().get("required", [])
                        if any(r in inner for r in required):
                            return inner
                return data
        except json.JSONDecodeError:
            pass

        # 策略 2：正则提取 Markdown 代码块中的内容
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._unwrap_if_needed(data, schema)
            except json.JSONDecodeError:
                pass

        # 策略 3：模糊匹配第一个 '{' 到最后一个 '}'
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._unwrap_if_needed(data, schema)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从模型响应中提取有效的 JSON: {content[:300]}")
    
    def _unwrap_if_needed(self, data: Dict[str, Any], schema: Type[T]) -> Dict[str, Any]:
        """如果数据被 Schema 类名嵌套包装，则将其解包。"""
        if not isinstance(data, dict):
            return data
        schema_name = schema.__name__
        if schema_name in data:
            return data[schema_name]
        for key in list(data.keys()):
            if key.lower() == schema_name.lower():
                return data[key]
        return data
    
    def complete_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout_s: float = 30.0,
    ) -> str:
        """
        快速文本补全便捷方法。适用于自由形式的叙事、摘要生成。
        """
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
            **self._request_options(structured=False),
        )
        
        if hasattr(response, 'usage') and response.usage:
            self._total_tokens_used += response.usage.total_tokens

        return response.choices[0].message.content

    @property
    def total_tokens_used(self) -> int:
        """获取系统启动至今累计使用的 Token 总量。"""
        return self._total_tokens_used

    def reset_token_counter(self) -> None:
        """重置 Token 计数器。"""
        self._total_tokens_used = 0
