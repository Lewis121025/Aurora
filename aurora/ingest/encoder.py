"""Generic non-rule-based encoder implementations."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

import numpy as np

from aurora.core.config import FieldConfig
from aurora.core.types import Anchor
from aurora.ingest.packetizer import PacketRecord
from aurora.store.blob_store import BlobStore


@runtime_checkable
class TraceEncoder(Protocol):
    def encode_query(self, cue: str | dict[str, object]) -> np.ndarray: ...

    def encode_packet(self, packet: PacketRecord) -> np.ndarray: ...

    def to_anchor(self, packet: PacketRecord) -> Anchor: ...


class HashProjectionEncoder:
    """Project raw packets into dense latent vectors without semantic rules."""

    def __init__(
        self,
        config: FieldConfig | None = None,
        blob_store: BlobStore | None = None,
        *,
        latent_dim: int | None = None,
    ) -> None:
        self.config = config or FieldConfig()
        self.latent_dim = int(latent_dim or self.config.latent_dim)
        self.blob_store = blob_store or BlobStore(self.config.blob_dir)

    def to_anchor(self, packet: PacketRecord) -> Anchor:
        vector = self.encode_packet(packet)
        source_quality = 0.95 if packet.source in {"user", "assistant"} else 0.80
        return Anchor(
            anchor_id=f"anc_{packet.packet_id.split('_', 1)[-1]}",
            packet_id=packet.packet_id,
            session_id=packet.session_id,
            turn_id=packet.turn_id,
            source=packet.source,
            z=vector.astype(np.float64, copy=False),
            z_hv=None,
            ts=packet.ts,
            residual_ref=packet.payload_ref,
            source_quality=source_quality,
            meta=dict(packet.meta),
        )

    def encode_packet(self, packet: PacketRecord) -> np.ndarray:
        payload = self.blob_store.read_text(packet.payload_ref)
        return self.encode_text(payload)

    def encode_query(self, cue: str | dict[str, object]) -> np.ndarray:
        if isinstance(cue, dict):
            text = str(cue.get("payload") or cue.get("text") or cue)
        else:
            text = cue
        return self.encode_text(text)

    def encode_text(self, text: str) -> np.ndarray:
        values = np.zeros(self.latent_dim, dtype=np.float32)
        byte_values = text.encode("utf-8")
        if not byte_values:
            return values
        for index in range(len(byte_values)):
            window = byte_values[index : index + 4]
            digest = hashlib.blake2b(window, digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "big") % values.size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[slot] += sign * (1.0 + (digest[5] / 255.0))
        norm = math.sqrt(float(np.dot(values, values)))
        if norm <= 1e-9:
            return values
        return (values / norm).astype(np.float32, copy=False)


HashingEncoder = HashProjectionEncoder


__all__ = ["HashProjectionEncoder", "HashingEncoder", "TraceEncoder"]
