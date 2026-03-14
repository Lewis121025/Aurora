from __future__ import annotations
import gzip
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import numpy as np

SEALED_STATE_VERSION = "aurora-seed-v6-holistic"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return dt.astimezone(timezone.utc).isoformat()


def parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return np.zeros_like(vec) if norm < eps else vec / norm


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom < eps else float(np.dot(a, b) / denom)


@dataclass
class LatentState:
    core_vector: np.ndarray  # The unshakeable self
    surface_vector: np.ndarray  # The volatile mood
    user_model: np.ndarray  # Theory of Mind: the perceived user

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_vector": self.core_vector.tolist(),
            "surface_vector": self.surface_vector.tolist(),
            "user_model": self.user_model.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentState":
        # Backward compatibility: if old v5 dict only has "vector"
        if "vector" in data and "core_vector" not in data:
            vec = np.asarray(data["vector"], dtype=np.float64)
            return cls(core_vector=vec.copy(), surface_vector=vec.copy(), user_model=vec.copy())
        return cls(
            np.asarray(data["core_vector"], dtype=np.float64),
            np.asarray(data["surface_vector"], dtype=np.float64),
            np.asarray(data["user_model"], dtype=np.float64),
        )


@dataclass
class MetricState:
    dim: int
    basis: np.ndarray
    lambdas: np.ndarray

    @classmethod
    def isotropic(cls, dim: int, rank: int) -> "MetricState":
        basis = np.zeros((dim, rank), dtype=np.float64)
        for i in range(min(dim, rank)):
            basis[i, i] = 1.0
        return cls(dim, basis, np.zeros(rank, dtype=np.float64))

    def matrix(self) -> np.ndarray:
        if self.basis.size == 0:
            return np.eye(self.dim)
        return np.eye(self.dim) + self.basis @ np.diag(self.lambdas) @ self.basis.T

    def to_dict(self) -> dict:
        return {"dim": self.dim, "basis": self.basis.tolist(), "lambdas": self.lambdas.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "MetricState":
        return cls(data["dim"], np.asarray(data["basis"]), np.asarray(data["lambdas"]))


@dataclass
class Spark:
    """Holistic node representing Episodic, Fossil, or Concept memories, potentially multimodal."""

    spark_id: str
    type: str  # 'episodic' | 'fossil' | 'concept' | 'void'
    timestamp: str
    text: str
    vector: np.ndarray
    energy: float
    source: str
    prev_id: str | None = None
    next_id: str | None = None
    resonant_links: list[str] = field(default_factory=list)
    # Multimodal & Affective extensions
    vad_vector: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )  # Valence, Arousal, Dominance
    media_refs: list[str] = field(default_factory=list)
    sensory_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "spark_id": self.spark_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "text": self.text,
            "vector": self.vector.tolist(),
            "energy": float(self.energy),
            "source": self.source,
            "prev_id": self.prev_id,
            "next_id": self.next_id,
            "resonant_links": self.resonant_links,
            "vad_vector": self.vad_vector,
            "media_refs": self.media_refs,
            "sensory_context": self.sensory_context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Spark":
        return cls(
            spark_id=data.get("spark_id", uuid.uuid4().hex),
            type=data.get("type", "episodic" if data.get("text") else "void"),
            timestamp=data["timestamp"],
            text=data["text"],
            vector=np.asarray(data["vector"], dtype=np.float64),
            energy=float(data["energy"]),
            source=data["source"],
            prev_id=data.get("prev_id"),
            next_id=data.get("next_id"),
            resonant_links=list(data.get("resonant_links", [])),
            vad_vector=list(data.get("vad_vector", [0.0, 0.0, 0.0])),
            media_refs=list(data.get("media_refs", [])),
            sensory_context=dict(data.get("sensory_context", {})),
        )


@dataclass
class ArrivalState:
    last_event_time: str
    no_contact_hours: float
    internal_drive: float
    decay_per_hour: float
    base_rate: float
    # Theory of Mind / Bond additions
    mutual_respect: float = 0.0
    cognitive_tension: float = 0.0

    def to_dict(self) -> dict:
        return {
            "last_event_time": self.last_event_time,
            "no_contact_hours": self.no_contact_hours,
            "internal_drive": self.internal_drive,
            "decay_per_hour": self.decay_per_hour,
            "base_rate": self.base_rate,
            "mutual_respect": self.mutual_respect,
            "cognitive_tension": self.cognitive_tension,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArrivalState":
        return cls(
            last_event_time=data["last_event_time"],
            no_contact_hours=data["no_contact_hours"],
            internal_drive=data["internal_drive"],
            decay_per_hour=data["decay_per_hour"],
            base_rate=data["base_rate"],
            mutual_respect=data.get("mutual_respect", 0.0),
            cognitive_tension=data.get("cognitive_tension", 0.0),
        )


@dataclass
class SealedStateHeader:
    version: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SealedStateHeader":
        return cls(data["version"], data["created_at"], data["updated_at"])


@dataclass
class SealedState:
    header: SealedStateHeader
    latent: LatentState
    metric: MetricState
    sparks: dict[str, Spark]  # 节点拓扑图字典
    arrival: ArrivalState | None
    rng_state: dict
    last_event_time: str
    next_wake_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "latent": self.latent.to_dict(),
            "metric": self.metric.to_dict(),
            "sparks": {k: v.to_dict() for k, v in self.sparks.items()},
            "arrival": self.arrival.to_dict() if self.arrival else None,
            "rng_state": self.rng_state,
            "last_event_time": self.last_event_time,
            "next_wake_at": self.next_wake_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SealedState":
        sparks_data = data["sparks"]

        # Backward compatibility for v4 (thermodynamic array)
        header = SealedStateHeader.from_dict(data["header"])
        if isinstance(sparks_data, list) and header.version == "aurora-seed-v4-thermodynamic":
            sparks_dict = {}
            old_sparks = [Spark.from_dict(s) for s in sparks_data]

            # Filter non-void sparks and sort by timestamp
            valid_sparks = [s for s in old_sparks if s.type != "void"]
            valid_sparks.sort(key=lambda s: parse_utc(s.timestamp))

            for i, s in enumerate(valid_sparks):
                if i > 0:
                    s.prev_id = valid_sparks[i - 1].spark_id
                if i < len(valid_sparks) - 1:
                    s.next_id = valid_sparks[i + 1].spark_id
                sparks_dict[s.spark_id] = s

            # Keep void sparks without links to fill capacity if needed,
            # but in v5, capacity management might change to dynamic graph pruning.
            # We will retain void sparks just as disconnected nodes.
            for s in old_sparks:
                if s.type == "void":
                    sparks_dict[s.spark_id] = s

            header.version = SEALED_STATE_VERSION
        else:
            sparks_dict = {k: Spark.from_dict(v) for k, v in sparks_data.items()}

        arrival_data = data.get("arrival")
        arrival = ArrivalState.from_dict(arrival_data) if arrival_data else None

        return cls(
            header=header,
            latent=LatentState.from_dict(data["latent"]),
            metric=MetricState.from_dict(data["metric"]),
            sparks=sparks_dict,
            arrival=arrival,
            rng_state=dict(data["rng_state"]),
            last_event_time=data["last_event_time"],
            next_wake_at=data.get("next_wake_at"),
        )


# ── 封印 / 解封 ──────────────────────────────────────────────────────────────

_MAGIC = b"AUR1"


def seal_state(state: SealedState) -> bytes:
    payload = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(payload).hexdigest().encode("ascii")
    return _MAGIC + checksum + gzip.compress(payload)


def unseal_state(blob: bytes) -> SealedState:
    if not blob.startswith(_MAGIC):
        raise ValueError("Invalid sealed blob prefix.")
    checksum = blob[len(_MAGIC) : len(_MAGIC) + 64]
    payload = gzip.decompress(blob[len(_MAGIC) + 64 :])
    if hashlib.sha256(payload).hexdigest().encode("ascii") != checksum:
        raise ValueError("Sealed blob checksum mismatch.")
    return SealedState.from_dict(json.loads(payload.decode("utf-8")))
