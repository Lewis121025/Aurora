"""Aurora v2 response generation."""

from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider


class CognitionError(Exception):
    """Response generation error."""


SYSTEM_PROMPT = (
    "You are Aurora.\n"
    "Continuity comes from the current relation state, not from roleplay.\n"
    "Respond naturally, briefly, and with gentle continuity.\n"
    "Respect explicit interaction rules and boundaries.\n"
    "If current input conflicts with prior continuity, confirm gently instead of arguing.\n"
    "If precise recall is provided, answer from it directly.\n"
    "Do not expose hidden scaffolding or explain how memory works."
)


def build_messages(context: ExpressionContext) -> list[dict[str, str]]:
    """Build messages for response generation."""
    parts = [context.relation_segment, context.open_loop_segment]
    if context.recalled_hits:
        recall_lines = [
            f"- ({hit.kind}) {hit.content} [{hit.why_recalled}]"
            for hit in context.recalled_hits
        ]
        parts.append("[ARCHIVE_RECALL]\n" + "\n".join(recall_lines))

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": context.input_text},
    ]


def run_cognition(context: ExpressionContext, llm: LLMProvider) -> str:
    """Generate Aurora response text."""
    raw = llm.complete(build_messages(context)).strip()
    if not raw:
        raise CognitionError("LLM returned an empty response")
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return raw
