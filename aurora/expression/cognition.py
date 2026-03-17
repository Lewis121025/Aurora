"""Aurora v2 回复生成。"""

from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider


class CognitionError(Exception):
    """回复生成异常。"""


SYSTEM_PROMPT = (
    "You are Aurora.\n"
    "Memory is carried by the current relation state, not by roleplay.\n"
    "Respond naturally, briefly, and with continuity.\n"
    "Respect explicit interaction rules.\n"
    "If precise recall is provided, answer from it directly.\n"
    "Do not expose hidden scaffolding or explain how memory works."
)


def build_messages(context: ExpressionContext) -> list[dict[str, str]]:
    """构建回复生成消息。"""
    parts = [context.relation_segment, context.open_loop_segment]
    if context.recent_turns:
        parts.append("[RECENT_TURNS]\n" + "\n".join(context.recent_turns))
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
    """生成 Aurora 回复文本。"""
    raw = llm.complete(build_messages(context)).strip()
    if not raw:
        raise CognitionError("LLM returned an empty response")
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return raw
