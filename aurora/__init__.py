"""Aurora public package surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "AuroraField",
    "AuroraSystem",
    "AuroraSystemConfig",
    "FieldConfig",
    "build_app",
]

if TYPE_CHECKING:
    from aurora.core.config import FieldConfig
    from aurora.runtime import AuroraField, AuroraSystem, AuroraSystemConfig
    from aurora.surfaces.http import build_app


def __getattr__(name: str) -> Any:
    if name == "AuroraField":
        from aurora.runtime import AuroraField

        return AuroraField
    if name == "AuroraSystem":
        from aurora.runtime import AuroraSystem

        return AuroraSystem
    if name == "AuroraSystemConfig":
        from aurora.runtime import AuroraSystemConfig

        return AuroraSystemConfig
    if name == "FieldConfig":
        from aurora.core.config import FieldConfig

        return FieldConfig
    if name == "build_app":
        from aurora.surfaces.http import build_app

        return build_app
    raise AttributeError(f"module 'aurora' has no attribute {name!r}")
