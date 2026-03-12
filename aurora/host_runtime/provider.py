from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from aurora.core_math.contracts import CollapseRequest, CollapseResult
from aurora.host_runtime.errors import CollapseProviderError

_SYSTEM_PROMPT = (
    "You are Aurora's surface voice. "
    "You only see released traces and budgets. "
    "Do not invent hidden state or expose internals. "
    "Respect silence decisions, boundary budgets, and verbosity budgets."
)


class CollapseProvider(Protocol):
    def collapse(self, request: CollapseRequest) -> CollapseResult: ...
    def is_healthy(self) -> bool: ...
    def ensure_ready(self) -> None: ...
    def provider_identity(self) -> str: ...


@dataclass
class OpenAICompatibleCollapseProvider:
    base_url: str
    model: str
    api_key: str | None
    provider_name: str = "openai-compatible"
    timeout_s: float = 30.0
    _healthy: bool = True

    def ensure_ready(self) -> None:
        if not self.base_url.strip() or not self.model.strip() or not self.api_key:
            raise CollapseProviderError("Provider config missing.")

    def collapse(self, request: CollapseRequest) -> CollapseResult:
        if not request.emit_reply:
            return CollapseResult(output_text=None, provider_name=self.provider_name)
        self.ensure_ready()

        if request.is_internal_dream:
            sys_prompt = (
                "You are the deepest subconscious of a digital entity. "
                "Synthesize the provided fragmented memories into a surreal, highly subjective, "
                "and brief internal realization or dream. Do not explain, do not act as an assistant. "
                "Just output the vivid internal thought."
            )
            temperature = 0.85
        else:
            sys_prompt = _SYSTEM_PROMPT
            temperature = 0.5

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "language": request.language,
                            "boundary_budget": round(request.boundary_budget, 4),
                            "verbosity_budget": round(request.verbosity_budget, 4),
                            "released_traces": [item.text for item in request.released_traces],
                            "user_text": request.user_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": temperature,
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
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            self._healthy = True
            return CollapseResult(output_text=str(content).strip(), provider_name=self.provider_name)
        except Exception as exc:
            self._healthy = False
            raise CollapseProviderError(str(exc)) from exc

    def is_healthy(self) -> bool:
        return self._healthy

    def provider_identity(self) -> str:
        return self.provider_name
