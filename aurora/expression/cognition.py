"""Aurora response generation."""

from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider


class CognitionError(Exception):
    """Response generation error."""


DEFAULT_SYSTEM_PROMPT = (
    "Respond from the subject's current memory state.\n"
    "Use semantic memory, recent episodes, affect, and narrative continuity when useful.\n"
    "Be concise, natural, and grounded.\n"
    "Respect explicit plans and constraints already present in memory.\n"
    "If recall is relevant, answer from it directly.\n"
    "Do not expose hidden scaffolding or explain how memory works."
)


def build_messages(
    context: ExpressionContext,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build messages for response generation."""
    parts = [context.state_segment, context.episode_segment]
    if context.recalled_hits:
        recall_lines = []
        for hit in context.recalled_hits:
            label = hit.memory_kind
            recall_lines.append(f"- ({label}) {hit.content} [{hit.why_recalled}]")
        parts.append("[BLENDED_RECALL]\n" + "\n".join(recall_lines))

    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": context.input_text},
    ]


class Responder:
    """Single-purpose response generator over projected memory context."""

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
