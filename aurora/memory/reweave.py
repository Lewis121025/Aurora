from __future__ import annotations

from aurora.being.profile import load_dynamics_profile
from aurora.memory.store import MemoryStore
from aurora.runtime.models import AssociationDelta, Tone


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _step_toward(current: float, target: float, step: float) -> float:
    if target > current:
        return min(current + step, target)
    return max(current - step, target)


def run_sleep_reweave(
    store: MemoryStore,
    relation_tone: Tone,
    relation_strength: float,
) -> float:
    profile = load_dynamics_profile().sleep
    if not store.fragments:
        store.sleep_cycles += 1
        store.last_reweave_delta = 0.0
        return 0.0

    step = profile.step
    tone_bias = profile.tone_bias[relation_tone]
    total_delta = 0.0

    for fragment in store.fragments:
        fragment_id = fragment.fragment_id
        old_salience = store.fragment_salience.get(fragment_id, 0.4)
        old_narrative = store.fragment_narrative_weight.get(fragment_id, old_salience)
        resonance = store.trace_intensity_for_turn(fragment.turn_id)
        association_score = store.association_strength_for_fragment(fragment_id)
        target_salience = _clamp(
            profile.salience_mix["old"] * old_salience
            + profile.salience_mix["resonance"] * resonance
            + profile.salience_mix["association"] * association_score
            + tone_bias * relation_strength,
            0.0,
            1.0,
        )
        new_salience = _step_toward(old_salience, target_salience, step)
        store.fragment_salience[fragment_id] = new_salience
        total_delta += abs(new_salience - old_salience)

        target_narrative = _clamp(
            profile.narrative_mix["old"] * old_narrative
            + profile.narrative_mix["salience"] * new_salience
            + profile.narrative_mix["association"] * association_score
            + profile.narrative_mix["resonance"] * resonance
            + tone_bias * relation_strength,
            0.0,
            1.0,
        )
        new_narrative = _step_toward(old_narrative, target_narrative, step)
        store.fragment_narrative_weight[fragment_id] = new_narrative
        total_delta += abs(new_narrative - old_narrative)

    for index, link in enumerate(store.associations):
        left_salience = store.fragment_salience.get(link.source_fragment_id, 0.4)
        right_salience = store.fragment_salience.get(link.target_fragment_id, 0.4)
        target_weight = _clamp(
            0.9 * link.weight + 0.1 * ((left_salience + right_salience) / 2.0),
            0.0,
            1.0,
        )
        new_weight = _step_toward(link.weight, target_weight, step)
        total_delta += abs(new_weight - link.weight)
        store.associations[index] = AssociationDelta(
            association_id=link.association_id,
            source_fragment_id=link.source_fragment_id,
            target_fragment_id=link.target_fragment_id,
            weight=new_weight,
            created_at=link.created_at,
        )

    store.sleep_cycles += 1
    normalized_delta = total_delta / max(2 * len(store.fragments) + len(store.associations), 1)
    store.last_reweave_delta = normalized_delta
    return normalized_delta
