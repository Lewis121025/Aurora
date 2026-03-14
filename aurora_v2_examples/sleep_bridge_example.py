from __future__ import annotations

from dataclasses import replace

from memory_reweave_v2_example import MemoryStoreV2, run_sleep_reweave_v2
from models_v2_example import ExistentialSnapshot, Phase
from relation_store_v2_example import RelationStoreV2


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def run_sleep_v2(
    snapshot: ExistentialSnapshot,
    memory_store: MemoryStoreV2,
    relation_store: RelationStoreV2,
    relation_id: str,
    now_ts: float,
) -> ExistentialSnapshot:
    """
    这是一个“桥接版” sleep：
    仍然兼容你当前 self/world/open 三元组，
    但不再只依赖单一 delta，而是使用 richer reweave result。
    """
    relation_tone = relation_store.current_tone(relation_id)
    relation_strength = relation_store.tone_strength(relation_id, relation_tone)

    reweave = run_sleep_reweave_v2(
        store=memory_store,
        relation_id=relation_id,
        relation_tone=relation_tone,
        relation_strength=relation_strength,
        now_ts=now_ts,
    )

    world_target = {
        "warm": 0.25,
        "neutral": 0.08,
        "cold": -0.20,
        "boundary": -0.35,
    }[relation_tone]
    open_target = {
        "warm": 0.78,
        "neutral": 0.62,
        "cold": 0.38,
        "boundary": 0.22,
    }[relation_tone]

    world_view = _clamp(
        snapshot.world_view
        + 0.18 * (world_target - snapshot.world_view)
        + 0.16 * reweave.coherence_shift
        - 0.12 * reweave.tension_shift
        + 0.10 * reweave.relation_bias,
        -1.0,
        1.0,
    )
    openness = _clamp(
        snapshot.openness
        + 0.16 * (open_target - snapshot.openness)
        + 0.12 * reweave.coherence_shift
        - 0.15 * reweave.tension_shift,
        0.0,
        1.0,
    )
    self_target = 0.10 + 0.22 * reweave.coherence_shift - 0.18 * reweave.tension_shift
    self_view = _clamp(
        snapshot.self_view + 0.20 * (self_target - snapshot.self_view),
        -1.0,
        1.0,
    )
    coherence_pressure = _clamp(
        0.70 * snapshot.coherence_pressure + 0.30 * (1.0 - reweave.coherence_shift),
        0.0,
        1.0,
    )
    sleep_pressure = _clamp(
        0.40 * snapshot.sleep_pressure + 0.15 * max(reweave.tension_shift - reweave.coherence_shift, 0.0),
        0.0,
        1.0,
    )

    return replace(
        snapshot,
        phase=Phase.SLEEP,
        self_view=self_view,
        world_view=world_view,
        openness=openness,
        coherence_pressure=coherence_pressure,
        sleep_pressure=sleep_pressure,
        updated_at=now_ts,
    )
