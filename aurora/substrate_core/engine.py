from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4
import numpy as np

from aurora.core_math.dynamics import prediction_error, update_metric
from aurora.core_math.encoding import Encoder, HashingEncoder, SemanticEncoder
from aurora.core_math.state import (
    LatentState,
    MetricState,
    SEALED_STATE_VERSION,
    SealedState,
    SealedStateHeader,
    Spark,
    cosine,
    isoformat_utc,
    normalize,
    parse_utc,
    seal_state,
    unseal_state,
    utc_now,
)


@dataclass(frozen=True)
class AuroraMemoryConfig:
    latent_dim: int = 512
    metric_rank: int = 8
    rng_seed: int = 7
    capacity: int = 4096


class AuroraMemoryCore:
    def __init__(
        self, config: AuroraMemoryConfig | None = None, encoder: Encoder | None = None
    ) -> None:
        self.config = config or AuroraMemoryConfig()
        if encoder is None:
            if self.config.latent_dim == 512:
                self.encoder = SemanticEncoder()
            else:
                self.encoder = HashingEncoder(dim=self.config.latent_dim, seed=self.config.rng_seed)
        else:
            self.encoder = encoder

    def boot(self) -> bytes:
        rng = np.random.default_rng(self.config.rng_seed)

        core_vec = normalize(rng.normal(scale=0.05, size=self.config.latent_dim))
        surface_vec = core_vec.copy()

        state = SealedState(
            header=SealedStateHeader(
                SEALED_STATE_VERSION, isoformat_utc(utc_now()), isoformat_utc(utc_now())
            ),
            latent=LatentState(
                core_vector=core_vec, surface_vector=surface_vec, user_model=core_vec.copy()
            ),
            metric=MetricState.isotropic(self.config.latent_dim, self.config.metric_rank),
            sparks={},
            arrival=None,
            rng_state=dict(rng.bit_generator.state),
            last_event_time=isoformat_utc(utc_now()),
            next_wake_at=None,
        )
        return seal_state(state)

    def _thermodynamics(self, state: SealedState, dt_hours: float) -> dict[str, float]:
        high_decay = float(np.exp(-0.02 * dt_hours))
        activations = {}

        valid_sparks = [
            s for s in state.sparks.values() if s.type in ("episodic", "fossil") and s.text
        ]
        if not valid_sparks:
            return activations

        for spark in valid_sparks:
            if spark.energy > 0.1:
                spark.energy *= high_decay
            else:
                spark.energy = max(1e-5, spark.energy - (0.0005 * dt_hours))

            if spark.type == "episodic" and spark.energy < 0.05:
                if spark.prev_id or spark.next_id:
                    prev_s = state.sparks.get(spark.prev_id) if spark.prev_id else None
                    next_s = state.sparks.get(spark.next_id) if spark.next_id else None
                    if prev_s:
                        prev_s.next_id = spark.next_id
                    if next_s:
                        next_s.prev_id = spark.prev_id
                    spark.prev_id = None
                    spark.next_id = None

        warped_latent = state.metric.matrix() @ state.latent.surface_vector
        vectors = np.stack([spark.vector for spark in valid_sparks])
        resonances = vectors @ warped_latent

        for i, spark in enumerate(valid_sparks):
            if resonances[i] > 0.05:
                spark.energy += 0.2 * float(resonances[i])
            activations[spark.spark_id] = spark.energy + max(0.0, float(resonances[i])) * 2.0

        return activations

    def _extract_episodic_subgraph(
        self, state: SealedState, activations: dict[str, float]
    ) -> list[Spark]:
        if not activations:
            return []

        sorted_ids = sorted(activations.keys(), key=lambda k: activations[k], reverse=True)
        anchors = sorted_ids[:2]

        extracted = set()
        for anchor_id in anchors:
            curr_id = anchor_id
            for _ in range(2):
                if curr_id not in state.sparks:
                    break
                extracted.add(curr_id)
                curr_id = state.sparks[curr_id].prev_id
                if not curr_id:
                    break

            curr_id = state.sparks[anchor_id].next_id
            for _ in range(2):
                if not curr_id or curr_id not in state.sparks:
                    break
                extracted.add(curr_id)
                curr_id = state.sparks[curr_id].next_id

        nodes = [state.sparks[nid] for nid in extracted]
        nodes.sort(key=lambda s: parse_utc(s.timestamp))
        return nodes

    def _add_memory(
        self,
        state: SealedState,
        timestamp: str,
        text: str,
        vector: np.ndarray,
        error: float,
        source: str,
    ) -> Spark:
        if len(state.sparks) >= self.config.capacity:
            coldest_spark = min(
                (s for s in state.sparks.values() if s.type != "concept"), key=lambda s: s.energy
            )
            prev_s = state.sparks.get(coldest_spark.prev_id) if coldest_spark.prev_id else None
            next_s = state.sparks.get(coldest_spark.next_id) if coldest_spark.next_id else None

            if prev_s:
                prev_s.next_id = coldest_spark.next_id
            if next_s:
                next_s.prev_id = coldest_spark.prev_id

            del state.sparks[coldest_spark.spark_id]

        latest_spark = None
        if state.sparks:
            temporal_sparks = [
                s
                for s in state.sparks.values()
                if s.type in ("episodic", "fossil")
                and (s.prev_id or s.next_id or len(state.sparks) == 1)
            ]
            if temporal_sparks:
                latest_spark = max(temporal_sparks, key=lambda s: parse_utc(s.timestamp))

        new_spark = Spark(
            spark_id=uuid4().hex,
            type="episodic",
            timestamp=timestamp,
            text=text,
            vector=vector,
            energy=1.0 + error * 5.0,
            source=source,
            prev_id=latest_spark.spark_id if latest_spark else None,
            next_id=None,
        )

        if latest_spark:
            latest_spark.next_id = new_spark.spark_id

        concept_sparks = [s for s in state.sparks.values() if s.type == "concept"]
        for concept in concept_sparks:
            sim = cosine(concept.vector, new_spark.vector)
            if sim > 0.85:
                new_spark.resonant_links.append(concept.spark_id)
                concept.resonant_links.append(new_spark.spark_id)

        state.sparks[new_spark.spark_id] = new_spark
        return new_spark

    def add_memory(
        self, sealed_state: bytes, text: str, timestamp: str | None = None
    ) -> tuple[bytes, str]:
        state, rng = self._load(sealed_state)
        ts = timestamp or isoformat_utc(utc_now())

        cue = self.encoder.encode(text)
        error = prediction_error(cue, state.latent.surface_vector)

        dt = 0.0
        if state.last_event_time:
            dt = max(
                0.0, (parse_utc(ts) - parse_utc(state.last_event_time)).total_seconds() / 3600.0
            )

        self._thermodynamics(state, dt)

        state.metric = update_metric(state.metric, cue, error)
        state.latent.surface_vector = normalize(0.8 * state.latent.surface_vector + 0.2 * cue)

        new_spark = self._add_memory(state, ts, text, cue, error, "user")

        state.last_event_time = state.header.updated_at = ts

        return self._save(state, rng), new_spark.spark_id

    def resonance_search(self, sealed_state: bytes, query: str, limit: int = 10) -> list[dict]:
        state, _ = self._load(sealed_state)

        if not state.sparks:
            return []

        cue = self.encoder.encode(query)

        valid_sparks = [
            s for s in state.sparks.values() if s.type in ("episodic", "fossil") and s.text
        ]
        if not valid_sparks:
            return []

        vectors = np.stack([spark.vector for spark in valid_sparks])
        similarities = vectors @ cue

        scored = [
            (spark, float(similarities[i]), similarities[i] * spark.energy)
            for i, spark in enumerate(valid_sparks)
        ]
        scored.sort(key=lambda x: x[2], reverse=True)

        results = []
        for spark, sim, resonance in scored[:limit]:
            results.append(
                {
                    "memory_id": spark.spark_id,
                    "text": spark.text,
                    "source": spark.source,
                    "timestamp": spark.timestamp,
                    "energy": spark.energy,
                    "resonance": resonance,
                    "similarity": sim,
                }
            )

        return results

    def get_all_memories(self, sealed_state: bytes) -> list[dict]:
        state, _ = self._load(sealed_state)

        valid_sparks = [s for s in state.sparks.values() if s.text]
        valid_sparks.sort(key=lambda s: parse_utc(s.timestamp), reverse=True)

        return [
            {
                "memory_id": s.spark_id,
                "text": s.text,
                "source": s.source,
                "timestamp": s.timestamp,
                "energy": s.energy,
                "type": s.type,
            }
            for s in valid_sparks
        ]

    def update_memory(self, sealed_state: bytes, memory_id: str, new_text: str) -> bytes:
        state, rng = self._load(sealed_state)

        if memory_id not in state.sparks:
            raise ValueError(f"Memory {memory_id} not found")

        spark = state.sparks[memory_id]
        spark.text = new_text
        spark.vector = self.encoder.encode(new_text)

        state.header.updated_at = isoformat_utc(utc_now())

        return self._save(state, rng)

    def delete_memory(self, sealed_state: bytes, memory_id: str) -> bytes:
        state, rng = self._load(sealed_state)

        if memory_id not in state.sparks:
            raise ValueError(f"Memory {memory_id} not found")

        spark = state.sparks[memory_id]

        if spark.prev_id:
            prev_s = state.sparks.get(spark.prev_id)
            if prev_s:
                prev_s.next_id = spark.next_id
        if spark.next_id:
            next_s = state.sparks.get(spark.next_id)
            if next_s:
                next_s.prev_id = spark.prev_id

        del state.sparks[memory_id]

        state.header.updated_at = isoformat_utc(utc_now())

        return self._save(state, rng)

    def delete_all_memories(self, sealed_state: bytes) -> bytes:
        state, rng = self._load(sealed_state)

        state.sparks.clear()
        state.header.updated_at = isoformat_utc(utc_now())

        return self._save(state, rng)

    def _load(self, blob: bytes) -> tuple[SealedState, np.random.Generator]:
        s = unseal_state(blob)
        r = np.random.default_rng()
        r.bit_generator.state = s.rng_state
        return s, r

    def _save(self, state: SealedState, rng: np.random.Generator) -> bytes:
        state.rng_state = dict(rng.bit_generator.state)
        return seal_state(state)

    def health(self, sealed_state: bytes) -> dict:
        state, _ = self._load(sealed_state)
        alive_count = sum(1 for spark in state.sparks.values() if spark.text)

        total_energy = sum(s.energy for s in state.sparks.values() if s.text)
        avg_energy = total_energy / alive_count if alive_count > 0 else 0.0

        return {
            "version": SEALED_STATE_VERSION,
            "alive": True,
            "anchor_count": alive_count,
            "avg_energy": avg_energy,
            "graph_density": len(state.sparks) / self.config.capacity
            if self.config.capacity > 0
            else 0.0,
        }
