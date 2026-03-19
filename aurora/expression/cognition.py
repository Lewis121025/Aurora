"""Aurora response generation."""

from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider


class CognitionError(Exception):
    """Response generation error."""


DEFAULT_SYSTEM_PROMPT = (
    "Respond from the memory brief.\n"
    "Use the active session transcript for local continuity and the memory brief for durable history.\n"
    "When the active session transcript conflicts with the memory brief, prefer the active session transcript.\n"
    "Read the brief in order: current_mainline, query_relevant, recent_changes, active_tensions, ongoing_commitments.\n"
    "Prefer query_relevant for the current question and use current_mainline as the default continuity anchor.\n"
    "Use recent_changes to understand recency and preserve active_tensions instead of flattening them.\n"
    "Continue ongoing_commitments when they are relevant.\n"
    "Do not expose hidden scaffolding or mention the memory field explicitly.\n"
)


def build_messages(
    context: ExpressionContext,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build messages for response generation."""
    system_message = (
        f"{system_prompt.rstrip()}\n\n"
        f"{context.memory_brief.strip()}\n\n"
        f"{context.session_transcript.strip()}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": context.input_text},
    ]


class Responder:
    """Single-purpose response generator over memory-field context."""

    __slots__ = ("llm", "system_prompt")

    def __init__(self, llm: LLMProvider, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def respond(self, context: ExpressionContext) -> str:
        raw = self.llm.complete(build_messages(context, system_prompt=self.system_prompt)).strip()
        if not raw:
            raise CognitionError("LLM returned an empty response")
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return raw
