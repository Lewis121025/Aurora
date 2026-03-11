"""Aurora soul package exports."""

from __future__ import annotations

from typing import Any

__all__ = ["AuroraSoul", "SoulConfig", "Plot", "StoryArc", "Theme", "IdentityState"]


def __getattr__(name: str) -> Any:
    if name in {"AuroraSoul", "SoulConfig"}:
        from aurora.soul.engine import AuroraSoul, SoulConfig

        return {"AuroraSoul": AuroraSoul, "SoulConfig": SoulConfig}[name]
    if name in {"Plot", "StoryArc", "Theme", "IdentityState"}:
        from aurora.soul.models import IdentityState, Plot, StoryArc, Theme

        return {
            "Plot": Plot,
            "StoryArc": StoryArc,
            "Theme": Theme,
            "IdentityState": IdentityState,
        }[name]
    raise AttributeError(name)
