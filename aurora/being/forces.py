from __future__ import annotations

import re

from aurora.being.profile import load_dynamics_profile
from aurora.being.touch import lexical_mode_scores
from aurora.memory.store import MemoryStore
from aurora.relation.store import RelationStore


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _tokens(text: str) -> set[str]:
    return {item for item in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower()) if item}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _memory_resonance(text: str, memory_store: MemoryStore) -> float:
    profile = load_dynamics_profile().touch_force.memory
    source = _tokens(text)
    if not source or not memory_store.fragments:
        return 0.0

    candidates = memory_store.fragments[-profile.window :]
    weighted_overlap = 0.0
    total_weight = 0.0
    for fragment in candidates:
        target = _tokens(fragment.text)
        similarity = _overlap(source, target)
        narrative = memory_store.narrative_weight_for_fragment(fragment.fragment_id)
        weight = profile.base_weight + profile.narrative_weight * narrative
        weighted_overlap += similarity * weight
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return _clamp(weighted_overlap / total_weight)


def compute_touch_forces(
    text: str,
    memory_store: MemoryStore,
    relation_store: RelationStore,
) -> dict[str, float]:
    profile = load_dynamics_profile().touch_force
    lexical = lexical_mode_scores(text)
    resonance = _memory_resonance(text, memory_store)
    novelty = 1.0 - resonance
    relation_tone = relation_store.current_tone()
    relation_strength = relation_store.tone_strength(relation_tone)

    relation_gate = {
        "warmth": relation_strength if relation_tone == "warm" else 0.0,
        "hurt": relation_strength if relation_tone in {"cold", "boundary"} else 0.0,
        "insight": relation_strength * 0.5,
        "boundary": relation_strength if relation_tone == "boundary" else 0.0,
        "curiosity": relation_strength * 0.3,
    }

    forces: dict[str, float] = {}
    for mode, mix in profile.mix.items():
        score = (
            mix.lexical * lexical.get(mode, 0.0)
            + mix.resonance * resonance
            + mix.relation * relation_gate.get(mode, 0.0)
            + mix.novelty * novelty
        )
        forces[mode] = _clamp(score)

    if "curiosity" in forces and "hurt" in forces:
        forces["curiosity"] = _clamp(
            forces["curiosity"] * (1.0 - profile.curiosity_hurt_penalty * forces["hurt"])
        )

    return forces
