from __future__ import annotations

from aurora.memory.recall import rank_fragment_ids
from aurora.memory.store import MemoryStore
from aurora.runtime.models import Tone


def select_expression_fragments(
    store: MemoryStore,
    relation_tone: Tone,
    relation_strength: float,
    limit: int = 3,
) -> tuple[str, ...]:
    fragment_ids = rank_fragment_ids(
        store=store,
        relation_tone=relation_tone,
        relation_strength=relation_strength,
        limit=limit,
    )
    texts: list[str] = []
    for fragment_id in fragment_ids:
        fragment = store.fragment_by_id(fragment_id)
        if fragment is not None and fragment.text:
            texts.append(fragment.text)
    return tuple(texts)
