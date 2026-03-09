"""
火山方舟 (Volcengine Ark) LLM 提供者
======================================

AURORA 内存系统的生产级 LLM 提供者。
支持结构化 JSON 输出、重试和错误处理。
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
    """火山方舟 LLM 提供者，支持结构化输出。

    功能：
    - JSON schema 约束输出
    - 指数退避自动重试
    - 令牌使用跟踪
    - 错误处理和回退
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "doubao-1-5-pro-32k-250115",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._total_tokens_used = 0
        
    def _get_client(self):
        """OpenAI 客户端的延迟初始化（Ark 使用 OpenAI 兼容 API）。"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0,
                )
            except ImportError:
                raise ImportError(
                    "Please install openai package: pip install openai"
                )
        return self._client

    def _request_options(self, *, structured: bool) -> Dict[str, Any]:
        """返回提供者特定的请求选项。"""
        return {}

    def _json_max_tokens(self, schema: Type[T]) -> int:
        """为结构化输出提供保守的令牌上限。"""
        return 512
    
    def _build_json_schema(self, schema: Type[T]) -> Dict[str, Any]:
        """从 Pydantic 模型构建 JSON schema 用于结构化输出。"""
        json_schema = schema.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": json_schema,
                "strict": True,
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
        """为给定的提示生成文本补全。

        参数：
            prompt：要补全的用户提示
            system：可选的系统消息
            temperature：采样温度 (0.0-1.0)
            max_tokens：要生成的最大令牌数
            timeout_s：请求超时时间（秒）
            stop：可选的停止序列
            metadata：可选的请求元数据

        返回：
            生成的文本字符串
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

                # 跟踪令牌使用情况
                if hasattr(response, 'usage') and response.usage:
                    self._total_tokens_used += response.usage.total_tokens
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{attempt_limit} failed: {e}")
                if attempt < attempt_limit - 1:
                    sleep_time = (2 ** attempt) * 0.5
                    time.sleep(sleep_time)

        raise RuntimeError(f"All {attempt_limit} attempts failed. Last error: {last_error}")

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
        """使用结构化 JSON 输出完成。

        参数：
            system：系统提示
            user：用户提示
            schema：用于响应验证的 Pydantic 模型
            temperature：采样温度
            timeout_s：请求超时时间（秒）
            metadata：可选的日志记录元数据

        返回：
            验证的 Pydantic 模型实例
        """
        client = self._get_client()
        messages = [
            {"role": "system", "content": f"{system}\n\nReturn only JSON that matches the provided schema."},
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

                # 跟踪令牌使用情况
                if hasattr(response, 'usage') and response.usage:
                    self._total_tokens_used += response.usage.total_tokens

                # 使用多种策略解析 JSON
                data = self._parse_json_response(content, schema)
                
                return schema.model_validate(data)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{attempt_limit} failed: {e}")
                if attempt < attempt_limit - 1:
                    # 指数退避
                    sleep_time = (2 ** attempt) * 0.5
                    time.sleep(sleep_time)

        raise RuntimeError(f"All {attempt_limit} attempts failed. Last error: {last_error}")
    
    def _parse_json_response(self, content: str, schema: Type[T]) -> Dict[str, Any]:
        """使用多种回退策略从 LLM 响应解析 JSON。"""
        import re

        # 策略 1：直接解析
        try:
            data = json.loads(content)
            # 检查数据是否包装在 schema 名称中
            schema_name = schema.__name__
            if isinstance(data, dict):
                if schema_name in data:
                    return data[schema_name]
                # 检查常见的包装模式
                for key in list(data.keys()):
                    if key.lower() == schema_name.lower():
                        return data[key]
                    # 如果只有一个键且是字典，解包它
                    if len(data) == 1 and isinstance(data[key], dict):
                        inner = data[key]
                        # 检查 inner 是否有必填字段
                        required = schema.model_json_schema().get("required", [])
                        if any(r in inner for r in required):
                            return inner
                return data
        except json.JSONDecodeError:
            pass

        # 策略 2：从 markdown 代码块提取
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._unwrap_if_needed(data, schema)
            except json.JSONDecodeError:
                pass

        # 策略 3：在内容中查找 JSON 对象
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._unwrap_if_needed(data, schema)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from response: {content[:300]}")
    
    def _unwrap_if_needed(self, data: Dict[str, Any], schema: Type[T]) -> Dict[str, Any]:
        """如果包装在 schema 名称中，则解包嵌套数据。"""
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
        """使用纯文本输出完成。

        适用于摘要和叙述等自由形式的生成。
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
        """所有请求中使用的总令牌数。"""
        return self._total_tokens_used

    def reset_token_counter(self) -> None:
        """重置令牌计数器。"""
        self._total_tokens_used = 0


class ArkLLMWithFallback(LLMProvider):
    """Ark LLM 与 MockLLM 回退，用于优雅降级。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "doubao-1-5-pro-32k-250115",
        **kwargs,
    ):
        self._primary: Optional[ArkLLM] = None
        self._fallback = None

        if api_key:
            try:
                self._primary = ArkLLM(api_key=api_key, model=model, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize ArkLLM: {e}")

        # 延迟导入回退
        from .mock import MockLLM
        self._fallback = MockLLM()
    
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
        """使用回退支持生成文本补全。"""
        if self._primary:
            try:
                return self._primary.complete(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                    stop=stop,
                    metadata=metadata,
                    max_retries=max_retries,
                )
            except Exception as e:
                logger.warning(f"Primary LLM complete() failed, using fallback: {e}")

        return self._fallback.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            stop=stop,
            metadata=metadata,
            max_retries=max_retries,
        )

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
        if self._primary:
            try:
                return self._primary.complete_json(
                    system=system,
                    user=user,
                    schema=schema,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    metadata=metadata,
                    max_retries=max_retries,
                )
            except Exception as e:
                logger.warning(f"Primary LLM failed, using fallback: {e}")
        
        return self._fallback.complete_json(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            timeout_s=timeout_s,
            metadata=metadata,
            max_retries=max_retries,
        )
