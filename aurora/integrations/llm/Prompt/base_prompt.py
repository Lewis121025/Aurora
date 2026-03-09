from __future__ import annotations

from aurora.integrations.llm.schemas import SCHEMA_VERSION


def instruction(model_name: str) -> str:
    return (
        "You MUST output ONLY valid JSON. No markdown. No extra keys. "
        f"Schema version: {SCHEMA_VERSION}. Output must conform to {model_name}."
    )


def render(template: str, **kwargs: object) -> str:
    return template.format(**kwargs)
