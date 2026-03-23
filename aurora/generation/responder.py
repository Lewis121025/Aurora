"""Aurora response generation over structured workspaces."""

from __future__ import annotations

from aurora.generation.context import GenerationContext
from aurora.llm.provider import LLMProvider


class ResponderError(Exception):
    """Response generation error."""


DEFAULT_SYSTEM_PROMPT = (
    "Respond using the active Aurora workspace.\n"
    "Use ongoing_context and salient_episodes for grounded continuity.\n"
    "Treat active_hypotheses as unresolved alternatives, not settled facts.\n"
    "Use relevant_procedures only when they help the current turn.\n"
    "Preserve provenance when exact support matters.\n"
    "Do not mention hidden memory machinery.\n"
)


def build_messages(
    context: GenerationContext,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    system_message = f"{system_prompt.rstrip()}\n\n{context.rendered_workspace.strip()}"
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": context.input_text},
    ]


class Responder:
    """Single-purpose response generator over structured workspace context."""

    __slots__ = ("llm", "system_prompt")

    def __init__(self, llm: LLMProvider, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def respond(self, context: GenerationContext) -> str:
        raw = self.llm.complete(build_messages(context, system_prompt=self.system_prompt)).strip()
        if not raw:
            raise ResponderError("LLM returned an empty response")
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return raw
