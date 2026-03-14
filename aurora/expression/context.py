from __future__ import annotations

from aurora.memory.selection import select_expression_fragments
from aurora.memory.store import MemoryStore
from aurora.relation.store import RelationStore
from aurora.runtime.models import ExpressionContext


def build_expression_context(
    turn_text: str,
    touch_modes: tuple[str, ...],
    memory_store: MemoryStore,
    relation_store: RelationStore,
) -> ExpressionContext:
    relation_tone = relation_store.current_tone()
    relation_strength = relation_store.tone_strength(relation_tone)
    return ExpressionContext(
        input_text=turn_text,
        relation_tone=relation_tone,
        relation_strength=relation_strength,
        memory_snippets=select_expression_fragments(
            store=memory_store,
            relation_tone=relation_tone,
            relation_strength=relation_strength,
            limit=3,
        ),
        touch_modes=touch_modes,
    )
