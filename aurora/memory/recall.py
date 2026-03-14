from __future__ import annotations

from aurora.being.profile import load_dynamics_profile
from aurora.memory.store import MemoryStore
from aurora.runtime.models import Tone


def _normalized_weights(relation_strength: float, narrative_pressure: float) -> dict[str, float]:
    profile = load_dynamics_profile().recall
    relation = profile.relation_base + profile.relation_scale * relation_strength
    narrative = profile.narrative_base + profile.narrative_scale * narrative_pressure
    recency = profile.recency_base - profile.recency_relation_scale * relation_strength
    resonance = profile.resonance_base + profile.resonance_relation_scale * (
        1.0 - relation_strength
    )
    association = profile.association_base + profile.association_narrative_scale * (
        1.0 - narrative_pressure
    )

    raw = {
        "relation": max(relation, 0.01),
        "resonance": max(resonance, 0.01),
        "association": max(association, 0.01),
        "narrative": max(narrative, 0.01),
        "recency": max(recency, 0.01),
    }
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def rank_fragment_ids(
    store: MemoryStore,
    relation_tone: Tone,
    relation_strength: float,
    limit: int = 3,
) -> tuple[str, ...]:
    fragments = store.fragments
    if not fragments:
        return ()

    total = len(fragments)
    weights = _normalized_weights(
        relation_strength=relation_strength,
        narrative_pressure=store.narrative_pressure(),
    )
    scored: list[tuple[str, float]] = []
    for index, fragment in enumerate(fragments):
        relation_score = store.relation_alignment_for_turn(fragment.turn_id, relation_tone)
        resonance_score = 0.5 * store.trace_intensity_for_turn(
            fragment.turn_id
        ) + 0.5 * store.fragment_salience.get(
            fragment.fragment_id,
            0.0,
        )
        association_score = store.association_strength_for_fragment(fragment.fragment_id)
        narrative_score = store.narrative_weight_for_fragment(fragment.fragment_id)
        recency_score = (index + 1) / total

        score = (
            weights["relation"] * relation_score
            + weights["resonance"] * resonance_score
            + weights["association"] * association_score
            + weights["narrative"] * narrative_score
            + weights["recency"] * recency_score
        )
        scored.append((fragment.fragment_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return tuple(fragment_id for fragment_id, _ in scored[:limit])
