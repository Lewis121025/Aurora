"""Relation field prompt helpers."""

from __future__ import annotations

from aurora.runtime.contracts import RelationField


def to_prompt_segment(field: RelationField) -> str:
    """Project the derived relation field into the runtime prompt."""
    rules = ", ".join(field.interaction_rules) if field.interaction_rules else "none"
    lexicon = ", ".join(field.shared_lexicon) if field.shared_lexicon else "none"
    return (
        "[RELATION_FIELD]\n"
        f"trust={field.trust:.2f}\n"
        f"distance={field.distance:.2f}\n"
        f"warmth={field.warmth:.2f}\n"
        f"tension={field.tension:.2f}\n"
        f"repair_debt={field.repair_debt:.2f}\n"
        f"interaction_rules={rules}\n"
        f"shared_lexicon={lexicon}"
    )
