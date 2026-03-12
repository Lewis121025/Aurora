from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

SEALED_STATE_VERSION = "aurora-seed-v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class LatentState:
    vector: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {"vector": self.vector.astype(float).tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentState":
        return cls(vector=np.asarray(data["vector"], dtype=np.float64))


@dataclass
class MetricState:
    dim: int
    basis: np.ndarray
    lambdas: np.ndarray

    @classmethod
    def isotropic(cls, dim: int, rank: int) -> "MetricState":
        basis = np.zeros((dim, rank), dtype=np.float64)
        for idx in range(min(dim, rank)):
            basis[idx, idx] = 1.0
        return cls(dim=dim, basis=basis, lambdas=np.zeros(rank, dtype=np.float64))

    def matrix(self) -> np.ndarray:
        if self.basis.size == 0 or self.lambdas.size == 0:
            return np.eye(self.dim, dtype=np.float64)
        diag = np.diag(self.lambdas)
        return np.eye(self.dim, dtype=np.float64) + self.basis @ diag @ self.basis.T

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "basis": self.basis.astype(float).tolist(),
            "lambdas": self.lambdas.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricState":
        return cls(
            dim=int(data["dim"]),
            basis=np.asarray(data["basis"], dtype=np.float64),
            lambdas=np.asarray(data["lambdas"], dtype=np.float64),
        )


@dataclass
class AnchorPacket:
    anchor_id: str
    timestamp: str
    speaker: str
    raw_text: str
    embedding: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "raw_text": self.raw_text,
            "embedding": self.embedding.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnchorPacket":
        return cls(
            anchor_id=str(data["anchor_id"]),
            timestamp=str(data["timestamp"]),
            speaker=str(data["speaker"]),
            raw_text=str(data["raw_text"]),
            embedding=np.asarray(data["embedding"], dtype=np.float64),
        )


@dataclass
class TracePacket:
    trace_id: str
    timestamp: str
    narrative: str
    embedding: np.ndarray
    source: str
    is_virtual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "narrative": self.narrative,
            "embedding": self.embedding.astype(float).tolist(),
            "source": self.source,
            "is_virtual": self.is_virtual,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TracePacket":
        return cls(
            trace_id=str(data["trace_id"]),
            timestamp=str(data["timestamp"]),
            narrative=str(data["narrative"]),
            embedding=np.asarray(data["embedding"], dtype=np.float64),
            source=str(data["source"]),
            is_virtual=bool(data.get("is_virtual", False)),
        )


@dataclass
class SchemaNode:
    summary: str
    prototype: np.ndarray
    revision_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "prototype": self.prototype.astype(float).tolist(),
            "revision_count": self.revision_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaNode":
        return cls(
            summary=str(data["summary"]),
            prototype=np.asarray(data["prototype"], dtype=np.float64),
            revision_count=int(data.get("revision_count", 0)),
        )


@dataclass
class MemoryFiber:
    anchor: AnchorPacket
    traces: list[TracePacket] = field(default_factory=list)
    schema: SchemaNode | None = None
    neighbors: dict[str, float] = field(default_factory=dict)
    basin_depth: float = 1.0
    temperature: float = 1.0
    access_count: int = 0

    @property
    def fiber_id(self) -> str:
        return self.anchor.anchor_id

    @property
    def centroid(self) -> np.ndarray:
        vectors = [self.anchor.embedding]
        vectors.extend(trace.embedding for trace in self.traces if not trace.is_virtual)
        if self.schema is not None:
            vectors.append(self.schema.prototype)
        return normalize(np.mean(np.stack(vectors, axis=0), axis=0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "schema": None if self.schema is None else self.schema.to_dict(),
            "neighbors": {key: float(value) for key, value in self.neighbors.items()},
            "basin_depth": float(self.basin_depth),
            "temperature": float(self.temperature),
            "access_count": int(self.access_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryFiber":
        return cls(
            anchor=AnchorPacket.from_dict(data["anchor"]),
            traces=[TracePacket.from_dict(item) for item in data.get("traces", [])],
            schema=None if data.get("schema") is None else SchemaNode.from_dict(data["schema"]),
            neighbors={str(key): float(value) for key, value in data.get("neighbors", {}).items()},
            basin_depth=float(data.get("basin_depth", 1.0)),
            temperature=float(data.get("temperature", 1.0)),
            access_count=int(data.get("access_count", 0)),
        )


@dataclass
class ArrivalState:
    last_event_time: str
    no_contact_hours: float
    internal_drive: float
    decay_per_hour: float
    base_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_event_time": self.last_event_time,
            "no_contact_hours": float(self.no_contact_hours),
            "internal_drive": float(self.internal_drive),
            "decay_per_hour": float(self.decay_per_hour),
            "base_rate": float(self.base_rate),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArrivalState":
        return cls(
            last_event_time=str(data["last_event_time"]),
            no_contact_hours=float(data["no_contact_hours"]),
            internal_drive=float(data["internal_drive"]),
            decay_per_hour=float(data["decay_per_hour"]),
            base_rate=float(data["base_rate"]),
        )


@dataclass
class SealedStateHeader:
    version: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SealedStateHeader":
        return cls(
            version=str(data["version"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )


@dataclass
class SealedState:
    header: SealedStateHeader
    latent: LatentState
    metric: MetricState
    memory: dict[str, MemoryFiber]
    recent_fiber_ids: list[str]
    arrival: ArrivalState
    rng_state: dict[str, Any]
    last_event_time: str
    next_wake_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "latent": self.latent.to_dict(),
            "metric": self.metric.to_dict(),
            "memory": {key: fiber.to_dict() for key, fiber in self.memory.items()},
            "recent_fiber_ids": list(self.recent_fiber_ids),
            "arrival": self.arrival.to_dict(),
            "rng_state": self.rng_state,
            "last_event_time": self.last_event_time,
            "next_wake_at": self.next_wake_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SealedState":
        return cls(
            header=SealedStateHeader.from_dict(data["header"]),
            latent=LatentState.from_dict(data["latent"]),
            metric=MetricState.from_dict(data["metric"]),
            memory={key: MemoryFiber.from_dict(value) for key, value in data["memory"].items()},
            recent_fiber_ids=[str(item) for item in data.get("recent_fiber_ids", [])],
            arrival=ArrivalState.from_dict(data["arrival"]),
            rng_state=dict(data["rng_state"]),
            last_event_time=str(data["last_event_time"]),
            next_wake_at=None if data.get("next_wake_at") is None else str(data["next_wake_at"]),
        )
