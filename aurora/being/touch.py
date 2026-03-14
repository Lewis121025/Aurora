from __future__ import annotations

import json
from pathlib import Path


_LEXICON_PATH = Path(__file__).with_name("touch_lexicon.json")
_LEXICON = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _lexicon_score(text: str, mode: str) -> float:
    entries: dict[str, float] = _LEXICON.get(mode, {})
    return sum(weight for token, weight in entries.items() if token in text)


def infer_touch_modes(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    modes: list[str] = [
        mode
        for mode in ("warmth", "hurt", "insight", "boundary", "curiosity")
        if _lexicon_score(lowered, mode) > 0.0
    ]

    if _contains_any(lowered, ("?", "？")) and "curiosity" not in modes:
        modes.append("curiosity")

    if not modes:
        return ("ambient",)
    return tuple(dict.fromkeys(modes))


def touch_intensity(modes: tuple[str, ...]) -> float:
    if modes == ("ambient",):
        return 0.20
    base = 0.30
    weights = {
        "warmth": 0.18,
        "hurt": 0.22,
        "insight": 0.16,
        "boundary": 0.20,
        "curiosity": 0.10,
        "ambient": 0.0,
    }
    score = base + sum(weights.get(mode, 0.08) for mode in modes)
    return min(score, 1.0)
