from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import numpy as np

SEALED_STATE_VERSION = "aurora-seed-v4-thermodynamic"


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
    vector: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {"vector": self.vector.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentState":
        return cls(np.asarray(data["vector"], dtype=np.float64))


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
    """万物皆星火。物理位置固定，只有 energy 决定生死。"""
    timestamp: str
    text: str
    vector: np.ndarray
    energy: float
    source: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "text": self.text,
            "vector": self.vector.tolist(),
            "energy": float(self.energy),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Spark":
        return cls(
            data["timestamp"], data["text"],
            np.asarray(data["vector"], dtype=np.float64),
            float(data["energy"]), data["source"],
        )


@dataclass
class ArrivalState:
    last_event_time: str
    no_contact_hours: float
    internal_drive: float
    decay_per_hour: float
    base_rate: float

    def to_dict(self) -> dict:
        return {
            "last_event_time": self.last_event_time,
            "no_contact_hours": self.no_contact_hours,
            "internal_drive": self.internal_drive,
            "decay_per_hour": self.decay_per_hour,
            "base_rate": self.base_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArrivalState":
        return cls(
            data["last_event_time"], data["no_contact_hours"], data["internal_drive"],
            data["decay_per_hour"], data["base_rate"],
        )


@dataclass
class SealedStateHeader:
    version: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {"version": self.version, "created_at": self.created_at, "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, data: dict) -> "SealedStateHeader":
        return cls(data["version"], data["created_at"], data["updated_at"])


@dataclass
class SealedState:
    header: SealedStateHeader
    latent: LatentState
    metric: MetricState
    sparks: list[Spark]      # 脑容量物理锁定：1024 个粒子槽，永不扩容
    arrival: ArrivalState
    rng_state: dict
    last_event_time: str
    next_wake_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "latent": self.latent.to_dict(),
            "metric": self.metric.to_dict(),
            "sparks": [s.to_dict() for s in self.sparks],
            "arrival": self.arrival.to_dict(),
            "rng_state": self.rng_state,
            "last_event_time": self.last_event_time,
            "next_wake_at": self.next_wake_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SealedState":
        return cls(
            SealedStateHeader.from_dict(data["header"]),
            LatentState.from_dict(data["latent"]),
            MetricState.from_dict(data["metric"]),
            [Spark.from_dict(s) for s in data["sparks"]],
            ArrivalState.from_dict(data["arrival"]),
            dict(data["rng_state"]),
            data["last_event_time"],
            data.get("next_wake_at"),
        )


# ── 封印 / 解封 ──────────────────────────────────────────────────────────────
import gzip
import hashlib
import json

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
