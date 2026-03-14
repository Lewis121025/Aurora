from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from aurora.runtime.models import Tone


@dataclass(frozen=True, slots=True)
class ModeDelta:
    self_delta: float
    world_delta: float
    open_delta: float


@dataclass(frozen=True, slots=True)
class DozeProfile:
    base_drift: float
    memory_scale: float
    narrative_scale: float
    self_target: float
    self_scale: float
    world_equilibrium: dict[Tone, float]
    open_equilibrium: dict[Tone, float]


@dataclass(frozen=True, slots=True)
class SleepProfile:
    step: float
    tone_bias: dict[Tone, float]
    salience_mix: dict[str, float]
    narrative_mix: dict[str, float]
    world_targets: dict[Tone, float]
    open_targets: dict[Tone, float]
    self_target: float
    world_scale: float
    open_scale: float
    self_scale: float


@dataclass(frozen=True, slots=True)
class RecallProfile:
    relation_base: float
    relation_scale: float
    narrative_base: float
    narrative_scale: float
    recency_base: float
    recency_relation_scale: float
    resonance_base: float
    resonance_relation_scale: float
    association_base: float
    association_narrative_scale: float


@dataclass(frozen=True, slots=True)
class DynamicsProfile:
    mode_deltas: dict[str, ModeDelta]
    relation_bias: dict[Tone, ModeDelta]
    doze: DozeProfile
    sleep: SleepProfile
    recall: RecallProfile


def _tone_map(source: dict[str, float]) -> dict[Tone, float]:
    return {
        "warm": float(source["warm"]),
        "neutral": float(source["neutral"]),
        "cold": float(source["cold"]),
        "boundary": float(source["boundary"]),
    }


@lru_cache(maxsize=1)
def load_dynamics_profile() -> DynamicsProfile:
    raw = json.loads(Path(__file__).with_name("dynamics_profile.json").read_text(encoding="utf-8"))

    mode_deltas = {
        mode: ModeDelta(
            self_delta=float(values["self"]),
            world_delta=float(values["world"]),
            open_delta=float(values["open"]),
        )
        for mode, values in raw["awake"]["mode_deltas"].items()
    }
    relation_bias = {
        tone: ModeDelta(
            self_delta=0.0,
            world_delta=float(values["world"]),
            open_delta=float(values["open"]),
        )
        for tone, values in raw["awake"]["relation_bias"].items()
    }

    doze_raw = raw["doze"]
    doze = DozeProfile(
        base_drift=float(doze_raw["base_drift"]),
        memory_scale=float(doze_raw["memory_scale"]),
        narrative_scale=float(doze_raw["narrative_scale"]),
        self_target=float(doze_raw["self_target"]),
        self_scale=float(doze_raw["self_scale"]),
        world_equilibrium=_tone_map(doze_raw["world_equilibrium"]),
        open_equilibrium=_tone_map(doze_raw["open_equilibrium"]),
    )

    sleep_raw = raw["sleep"]
    sleep = SleepProfile(
        step=float(sleep_raw["step"]),
        tone_bias=_tone_map(sleep_raw["tone_bias"]),
        salience_mix={k: float(v) for k, v in sleep_raw["salience_mix"].items()},
        narrative_mix={k: float(v) for k, v in sleep_raw["narrative_mix"].items()},
        world_targets=_tone_map(sleep_raw["snapshot_targets"]["world"]),
        open_targets=_tone_map(sleep_raw["snapshot_targets"]["open"]),
        self_target=float(sleep_raw["snapshot_targets"]["self"]),
        world_scale=float(sleep_raw["snapshot_scales"]["world"]),
        open_scale=float(sleep_raw["snapshot_scales"]["open"]),
        self_scale=float(sleep_raw["snapshot_scales"]["self"]),
    )

    recall_raw = raw["recall"]
    recall = RecallProfile(
        relation_base=float(recall_raw["relation_base"]),
        relation_scale=float(recall_raw["relation_scale"]),
        narrative_base=float(recall_raw["narrative_base"]),
        narrative_scale=float(recall_raw["narrative_scale"]),
        recency_base=float(recall_raw["recency_base"]),
        recency_relation_scale=float(recall_raw["recency_relation_scale"]),
        resonance_base=float(recall_raw["resonance_base"]),
        resonance_relation_scale=float(recall_raw["resonance_relation_scale"]),
        association_base=float(recall_raw["association_base"]),
        association_narrative_scale=float(recall_raw["association_narrative_scale"]),
    )

    return DynamicsProfile(
        mode_deltas=mode_deltas,
        relation_bias=relation_bias,
        doze=doze,
        sleep=sleep,
        recall=recall,
    )
