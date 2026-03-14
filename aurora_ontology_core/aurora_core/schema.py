from __future__ import annotations

from typing import Literal

Speaker = Literal["user", "aurora", "system"]
Phase = Literal["awake", "doze", "sleep"]
TraceChannel = Literal[
    "warmth",
    "hurt",
    "recognition",
    "distance",
    "curiosity",
    "boundary",
    "repair",
    "wonder",
]
AssociationKind = Literal[
    "resonance",
    "contrast",
    "repair",
    "boundary",
    "chapter",
    "temporal",
    "causal_guess",
]
OtherMoveKind = Literal["share", "ask", "pressure", "repair", "rupture", "care", "withdraw"]
AuroraMoveKind = Literal["witness", "approach", "boundary", "repair", "withdraw", "silence"]
ChapterStatus = Literal["forming", "stable", "dormant", "fractured"]
