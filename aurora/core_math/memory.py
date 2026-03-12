from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import numpy as np

from aurora.core_math.encoding import HashingEncoder
from aurora.core_math.state import (
    AnchorPacket,
    MemoryFiber,
    SchemaNode,
    SealedState,
    TracePacket,
    cosine,
    isoformat_utc,
    normalize,
)


@dataclass
class SampledFiber:
    fiber_id: str
    energy: float
    released_trace: str
    virtual_trace: str | None


class MemoryField:
    def __init__(self, encoder: HashingEncoder, trace_limit: int = 6):
        self.encoder = encoder
        self.trace_limit = trace_limit

    def fiber_energy(
        self,
        fiber: MemoryFiber,
        cue_embedding: np.ndarray,
        latent_embedding: np.ndarray,
        recent_ids: list[str],
    ) -> float:
        cue_term = -cosine(cue_embedding, fiber.anchor.embedding)
        latent_term = -cosine(latent_embedding, fiber.centroid)
        inertia = -0.18 if fiber.fiber_id in recent_ids else 0.0
        anchor_penalty = 0.08 * max(0, len(fiber.anchor.raw_text) - 180) / 180.0
        basin_bonus = -0.15 * fiber.basin_depth
        return cue_term + latent_term + inertia + anchor_penalty + basin_bonus

    def _proposal_ids(self, state: SealedState, current_id: str | None) -> list[str]:
        if not state.memory:
            return []
        if current_id is None or current_id not in state.memory:
            return list(state.memory.keys())
        current = state.memory[current_id]
        if current.neighbors:
            return list(current.neighbors.keys()) + [current_id]
        return list(state.memory.keys())

    def sample(
        self,
        state: SealedState,
        cue_embedding: np.ndarray,
        latent_embedding: np.ndarray,
        rng: np.random.Generator,
        count: int = 3,
        steps: int = 12,
    ) -> list[SampledFiber]:
        if not state.memory:
            return []
        sampled: list[SampledFiber] = []
        chosen_ids: set[str] = set()
        current_id: str | None = None
        current_energy = float("inf")
        temperature_floor = 0.05

        for _ in range(count):
            for _ in range(steps):
                proposals = self._proposal_ids(state, current_id)
                candidate_id = proposals[int(rng.integers(0, len(proposals)))]
                candidate = state.memory[candidate_id]
                candidate_energy = self.fiber_energy(
                    candidate,
                    cue_embedding=cue_embedding,
                    latent_embedding=latent_embedding,
                    recent_ids=state.recent_fiber_ids,
                )
                if current_id is None:
                    accept = True
                else:
                    temp = max(temperature_floor, candidate.temperature)
                    delta = candidate_energy - current_energy
                    accept = bool(delta <= 0.0 or rng.random() < np.exp(-delta / temp))
                if accept:
                    current_id = candidate_id
                    current_energy = candidate_energy

            if current_id is None:
                continue
            if current_id in chosen_ids:
                current_id = None
                current_energy = float("inf")
                continue
            chosen_ids.add(current_id)
            fiber = state.memory[current_id]
            trace_text = fiber.traces[-1].narrative if fiber.traces else fiber.anchor.raw_text
            virtual = None if fiber.schema is None else self.virtual_trace_text(fiber)
            sampled.append(
                SampledFiber(
                    fiber_id=current_id,
                    energy=current_energy,
                    released_trace=trace_text[:220],
                    virtual_trace=virtual,
                )
            )
            current_id = None
            current_energy = float("inf")
        return sampled

    def reinforce(
        self,
        state: SealedState,
        fiber_ids: list[str],
        error: float,
    ) -> float:
        if not fiber_ids:
            return 0.0
        basin_pressure = 0.0
        for fiber_id in fiber_ids:
            fiber = state.memory[fiber_id]
            fiber.access_count += 1
            fiber.basin_depth = float(np.clip(fiber.basin_depth + 0.18 * error, 0.5, 8.0))
            fiber.temperature = float(np.clip(fiber.temperature - 0.08 * error, 0.15, 1.5))
            basin_pressure += fiber.basin_depth
        return basin_pressure / len(fiber_ids)

    def add_dialogue_trace(
        self,
        state: SealedState,
        when: datetime,
        user_text: str,
        released_text: str,
        cue_embedding: np.ndarray,
    ) -> str:
        anchor_id = uuid4().hex
        timestamp = isoformat_utc(when)
        anchor = AnchorPacket(
            anchor_id=anchor_id,
            timestamp=timestamp,
            speaker="user",
            raw_text=user_text,
            embedding=cue_embedding.copy(),
        )
        narrative = released_text[:220] if released_text else user_text[:220]
        trace = TracePacket(
            trace_id=uuid4().hex,
            timestamp=timestamp,
            narrative=narrative,
            embedding=self.encoder.encode(narrative),
            source="dialogue",
        )
        schema = SchemaNode(summary=narrative[:140], prototype=trace.embedding.copy())
        state.memory[anchor_id] = MemoryFiber(anchor=anchor, traces=[trace], schema=schema)
        self._remember_recent(state, anchor_id)
        self._link_recent(state, anchor_id)
        return anchor_id

    def reconsolidate(
        self,
        state: SealedState,
        fiber_ids: list[str],
        when: datetime,
        user_text: str,
    ) -> None:
        timestamp = isoformat_utc(when)
        for fiber_id in fiber_ids:
            fiber = state.memory.get(fiber_id)
            if fiber is None:
                continue
            narrative = self.rewrite_narrative(fiber, user_text)
            trace = TracePacket(
                trace_id=uuid4().hex,
                timestamp=timestamp,
                narrative=narrative,
                embedding=self.encoder.encode(narrative),
                source="reconsolidation",
            )
            fiber.traces.append(trace)
            fiber.traces = fiber.traces[-self.trace_limit :]
            prototype = normalize(
                np.mean(np.stack([item.embedding for item in fiber.traces], axis=0), axis=0)
            )
            fiber.schema = SchemaNode(
                summary=narrative[:140],
                prototype=prototype,
                revision_count=0 if fiber.schema is None else fiber.schema.revision_count + 1,
            )

    def add_internal_trace(self, state: SealedState, when: datetime, narrative: str) -> None:
        timestamp = isoformat_utc(when)
        anchor_id = uuid4().hex
        emb = self.encoder.encode(narrative)
        anchor = AnchorPacket(
            anchor_id=anchor_id,
            timestamp=timestamp,
            speaker="self",
            raw_text=narrative,
            embedding=emb.copy(),
        )
        trace = TracePacket(
            trace_id=uuid4().hex,
            timestamp=timestamp,
            narrative=narrative,
            embedding=emb.copy(),
            source="internal",
        )
        state.memory[anchor_id] = MemoryFiber(
            anchor=anchor,
            traces=[trace],
            schema=SchemaNode(summary=narrative[:140], prototype=emb.copy()),
        )
        self._remember_recent(state, anchor_id)
        self._link_recent(state, anchor_id)

    def virtual_trace_text(self, fiber: MemoryFiber) -> str:
        if fiber.schema is None:
            return fiber.anchor.raw_text[:140]
        return f"schema echo: {fiber.schema.summary}"

    def rewrite_narrative(self, fiber: MemoryFiber, user_text: str) -> str:
        anchor = fiber.anchor.raw_text[:80].replace("\n", " ")
        return f"current cue reframed [{user_text[:80]}] through anchor [{anchor}]"

    def released_traces(self, state: SealedState, sampled: list[SampledFiber]) -> list[str]:
        return [item.released_trace for item in sampled]

    def released_virtual_traces(self, sampled: list[SampledFiber]) -> list[str]:
        return [item.virtual_trace for item in sampled if item.virtual_trace]

    def _remember_recent(self, state: SealedState, fiber_id: str) -> None:
        state.recent_fiber_ids.append(fiber_id)
        state.recent_fiber_ids = state.recent_fiber_ids[-8:]

    def _link_recent(self, state: SealedState, fiber_id: str) -> None:
        for recent_id in state.recent_fiber_ids[:-1]:
            if recent_id == fiber_id or recent_id not in state.memory:
                continue
            state.memory[fiber_id].neighbors[recent_id] = (
                state.memory[fiber_id].neighbors.get(recent_id, 0.0) + 1.0
            )
            state.memory[recent_id].neighbors[fiber_id] = (
                state.memory[recent_id].neighbors.get(fiber_id, 0.0) + 1.0
            )
