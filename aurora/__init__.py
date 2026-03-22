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
    if name in {"AuroraSystem", "AuroraSystemConfig", "build_app"}:
        from aurora.runtime import AuroraSystem, AuroraSystemConfig
        from aurora.surfaces.http import build_app

        return {
            "AuroraSystem": AuroraSystem,
            "AuroraSystemConfig": AuroraSystemConfig,
            "build_app": build_app,
        }[name]
    if name == "FieldConfig":
        from aurora.core.config import FieldConfig

        return FieldConfig
    raise AttributeError(f"module 'aurora' has no attribute {name!r}")
