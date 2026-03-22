"""Mechanical packetization for raw events."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from aurora.core.config import FieldConfig
from aurora.core.types import PayloadType, SourceType
from aurora.store import BlobStore


@dataclass(frozen=True, slots=True)
class PacketRecord:
    packet_id: str
    ts: float
    session_id: str
    turn_id: str
    source: SourceType
    payload_type: PayloadType
    payload_ref: str
    token_count: int
    meta: dict[str, Any] = field(default_factory=dict)


class Packetizer:
    """Split raw events only by mechanical boundaries, never semantics."""

    def __init__(
        self,
        config: FieldConfig | None = None,
        blob_store: BlobStore | None = None,
        *,
        max_chars: int | None = None,
    ) -> None:
        self.config = config or FieldConfig()
        self.max_chars = int(max_chars or self.config.packet_max_chars)
        self.blob_store = blob_store or BlobStore(self.config.blob_dir)

    def split(self, raw_event: Mapping[str, Any] | str) -> list[PacketRecord]:
        if isinstance(raw_event, str):
            raw_event = {"text": raw_event}
        ts = float(raw_event.get("ts") or time.time())
        session_id = str(raw_event.get("session_id") or "")
        turn_id = str(raw_event.get("turn_id") or f"turn-{uuid.uuid4().hex[:10]}")
        source = self._source(raw_event.get("source"))
        payload_type = self._payload_type(raw_event.get("payload_type"))
        payload = str(raw_event.get("payload") or raw_event.get("text") or "")
        meta = dict(raw_event.get("meta") or raw_event.get("metadata") or {})
        if not payload.strip():
            raise ValueError("raw_event payload must not be empty")
        packets: list[PacketRecord] = []
        chunk_size = max(64, self.max_chars)
        for index in range(0, len(payload), chunk_size):
            chunk = payload[index : index + chunk_size]
            packet_id = f"pkt_{uuid.uuid4().hex[:12]}"
            payload_ref = self.blob_store.put_text(packet_id, chunk)
            packets.append(
                PacketRecord(
                    packet_id=packet_id,
                    ts=ts,
                    session_id=session_id,
                    turn_id=turn_id,
                    source=source,
                    payload_type=payload_type,
                    payload_ref=payload_ref,
                    token_count=max(1, len(chunk.encode("utf-8")) // 4),
                    meta={**meta, "chunk_index": len(packets)},
                )
            )
        return packets

    @staticmethod
    def _source(value: Any) -> SourceType:
        normalized = str(value or "user").strip().lower()
        if normalized in {"user", "assistant", "tool", "env"}:
            return normalized  # type: ignore[return-value]
        raise ValueError("source must be one of user, assistant, tool, or env")

    @staticmethod
    def _payload_type(value: Any) -> PayloadType:
        normalized = str(value or "text").strip().lower()
        if normalized in {"text", "tool_call", "tool_result", "state_delta", "reward", "observation"}:
            return normalized  # type: ignore[return-value]
        raise ValueError("payload_type must be a supported Aurora payload type")


__all__ = ["Packetizer"]
