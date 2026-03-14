from __future__ import annotations

import json
from pathlib import Path

from aurora.being.profile import load_dynamics_profile


_LEXICON_PATH = Path(__file__).with_name("touch_lexicon.json")
_LEXICON = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
_MODES = ("warmth", "hurt", "insight", "boundary", "curiosity")


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _lexicon_score(text: str, mode: str) -> float:
    entries: dict[str, float] = _LEXICON.get(mode, {})
    return sum(weight for token, weight in entries.items() if token in text)


def lexical_mode_scores(text: str) -> dict[str, float]:
    lowered = text.lower()
    scores = {mode: _lexicon_score(lowered, mode) for mode in _MODES}
    if _contains_any(lowered, ("?", "？")):
        scores["curiosity"] += 0.25

    max_score = max(scores.values(), default=0.0)
    if max_score <= 0.0:
        return {mode: 0.0 for mode in _MODES}
    return {mode: min(score / max_score, 1.0) for mode, score in scores.items()}


def infer_touch_modes_from_forces(forces: dict[str, float]) -> tuple[str, ...]:
    profile = load_dynamics_profile().touch_force
    ordered = sorted(forces.items(), key=lambda item: item[1], reverse=True)
    modes = [mode for mode, score in ordered if mode in _MODES and score >= profile.mode_threshold]
    if not modes:
        return ("ambient",)
    return tuple(modes)
