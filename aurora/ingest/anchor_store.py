"""In-memory anchor and packet stores."""

from __future__ import annotations

from aurora.core.types import Anchor
from aurora.ingest.packetizer import PacketRecord


class AnchorStore:
    """Keep packet records and anchors addressable for the field runtime."""

    def __init__(self) -> None:
        self.packets: dict[str, PacketRecord] = {}
        self.anchors: dict[str, Anchor] = {}

    def add_packet(self, packet: PacketRecord) -> None:
        self.packets[packet.packet_id] = packet

    def add_anchor(self, anchor: Anchor) -> None:
        self.anchors[anchor.anchor_id] = anchor

    def packet(self, packet_id: str) -> PacketRecord | None:
        return self.packets.get(packet_id)

    def anchor(self, anchor_id: str) -> Anchor | None:
        return self.anchors.get(anchor_id)
