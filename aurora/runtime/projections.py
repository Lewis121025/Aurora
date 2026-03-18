"""Aurora v3 runtime projection helpers."""

from __future__ import annotations

from aurora.relation.state import to_prompt_segment as relation_prompt
from aurora.relation.tension import top_open_loops, to_prompt_segment as loop_prompt
from aurora.runtime.contracts import OpenLoop, RecallHit, RelationField


def build_memory_projection(
    field: RelationField,
    loops: tuple[OpenLoop, ...],
    recall_hits: tuple[RecallHit, ...],
    now_ts: float,
) -> tuple[str, str, tuple[RecallHit, ...]]:
    """Build the hot-path context projection."""
    active_loops = top_open_loops(loops, now_ts)
    return (
        relation_prompt(field),
        loop_prompt(active_loops, now_ts),
        recall_hits[:5],
    )
