from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def _templates() -> dict[str, dict[str, str]]:
    with (
        resources.files("aurora.expression")
        .joinpath("templates.json")
        .open("r", encoding="utf-8") as handle
    ):
        data = json.load(handle)
    return {
        str(section): {str(key): str(value) for key, value in values.items()}
        for section, values in data.items()
    }


def expression_text(section: str, variant: str = "default") -> str:
    section_templates = _templates()[section]
    return section_templates.get(variant, section_templates["default"])
