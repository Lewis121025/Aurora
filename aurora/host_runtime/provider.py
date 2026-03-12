from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from aurora.core_math.contracts import CollapseRequest, CollapseResult
from aurora.host_runtime.errors import CollapseProviderError

_SYSTEM_PROMPT = (
    "You are Aurora's surface voice. "
    "You only see released traces, released virtual traces, and budgets. "
    "Do not invent hidden state or expose internals. "
    "Respect silence decisions, boundary budgets, and verbosity budgets."
)


class CollapseProvider(Protocol):
    def collapse(self, request: CollapseRequest) -> CollapseResult:
        ...

    def is_healthy(self) -> bool:
        ...

    def ensure_ready(self) -> None:
        ...

    def provider_identity(self) -> str:
        ...


@dataclass
class OpenAICompatibleCollapseProvider:
    base_url: str
    model: str
    api_key: str | None
    provider_name: str = "openai-compatible"
    timeout_s: float = 30.0
    _healthy: bool = True

    def ensure_ready(self) -> None:
        if not self.base_url.strip():
            raise CollapseProviderError("Provider base URL is required.")
        if not self.model.strip():
            raise CollapseProviderError("Provider model is required.")
        if not self.api_key:
            raise CollapseProviderError(
                "No usable provider API key found. Set AURORA_PROVIDER_API_KEY "
                "or AURORA_BAILIAN_LLM_API_KEY."
            )

    def collapse(self, request: CollapseRequest) -> CollapseResult:
        if not request.emit_reply:
            return CollapseResult(output_text=None, provider_name=self.provider_name)
        self.ensure_ready()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "language": request.language,
                            "boundary_budget": round(request.boundary_budget, 4),
                            "verbosity_budget": round(request.verbosity_budget, 4),
                            "released_traces": [item.text for item in request.released_traces],
                            "released_virtual_traces": [
                                item.text for item in request.released_virtual_traces
                            ],
                            "user_text": request.user_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            raw = self._read_json(req, timeout_s=self.timeout_s)
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            if self._is_timeout(exc):
                retry_timeout = max(self.timeout_s * 6.0, 60.0)
                try:
                    raw = self._read_json(req, timeout_s=retry_timeout)
                except (TimeoutError, socket.timeout, urllib.error.URLError) as retry_exc:
                    self._healthy = False
                    raise CollapseProviderError(str(retry_exc)) from retry_exc
            else:
                self._healthy = False
                raise CollapseProviderError(str(exc)) from exc
        except OSError as exc:
            self._healthy = False
            raise CollapseProviderError(str(exc)) from exc
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            self._healthy = False
            raise CollapseProviderError("Malformed response from collapse provider.") from exc
        self._healthy = True
        return CollapseResult(output_text=str(content).strip(), provider_name=self.provider_name)

    def is_healthy(self) -> bool:
        return self._healthy

    def provider_identity(self) -> str:
        return self.provider_name

    def _read_json(self, req: urllib.request.Request, *, timeout_s: float) -> dict[str, object]:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _is_timeout(self, exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return True
        if isinstance(exc, urllib.error.URLError):
            return isinstance(exc.reason, (TimeoutError, socket.timeout))
        return False
