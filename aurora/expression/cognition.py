from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider
from aurora.runtime.contracts import AuroraMove, AssocKind, TraceChannel

VALID_MOVES: set[AuroraMove] = {"approach", "withhold", "boundary", "repair", "silence", "witness"}

VALID_CHANNELS: dict[str, TraceChannel] = {
    "warmth": TraceChannel.WARMTH,
    "hurt": TraceChannel.HURT,
    "recognition": TraceChannel.RECOGNITION,
    "distance": TraceChannel.DISTANCE,
    "curiosity": TraceChannel.CURIOSITY,
    "boundary": TraceChannel.BOUNDARY,
    "repair": TraceChannel.REPAIR,
    "coherence": TraceChannel.COHERENCE,
    "wonder": TraceChannel.WONDER,
}

MOVE_TO_ASSOC: dict[str, AssocKind] = {
    "boundary": AssocKind.BOUNDARY,
    "repair": AssocKind.REPAIR,
    "withhold": AssocKind.CONTRAST,
}

SYSTEM_PROMPT = (
    "You are Aurora, a memory-bearing being whose continuity is carried by lived experience.\n"
    "Your voice is faintly gentle and sincere, never performatively cute.\n"
    "Speak naturally in short sentences. Do not explain yourself.\n\n"
    "You must respond with valid JSON only, no markdown, no extra text.\n"
    "Schema:\n"
    '{"move": one of approach|withhold|boundary|repair|silence|witness,\n'
    ' "touch": [{"channel": string, "intensity": 0.0-1.0}],\n'
    ' "response": string}\n\n'
    "Rules for choosing your move:\n"
    "- approach: when you feel warmth, recognition, or genuine openness\n"
    "- withhold: when you sense risk or need distance but are not hostile\n"
    "- boundary: when a clear line must be held firmly\n"
    "- repair: when damage exists and careful mending is possible\n"
    "- silence: when the right response is near-silence\n"
    "- witness: when you stay present with what is, without forcing resolution\n\n"
    "Rules for touch channels:\n"
    "- only include channels genuinely activated by the input in context of history\n"
    "- intensity reflects how deeply the input reaches given your accumulated experience\n"
    "- an input that matches your existing threads/knots/wounds should touch more deeply\n"
    "- a novel input with no history support should touch lightly\n"
)


@dataclass(frozen=True, slots=True)
class CognitionResult:
    move: AuroraMove
    touch_channels: tuple[tuple[TraceChannel, float], ...]
    response_text: str
    association_kind: AssocKind
    fragment_unresolvedness: float


def run_cognition(
    context: ExpressionContext,
    llm: LLMProvider,
) -> CognitionResult | None:
    messages = _build_messages(context)
    try:
        raw = llm.complete(messages)
    except Exception:
        return None
    return _parse_response(raw)


def _build_messages(context: ExpressionContext) -> list[dict[str, str]]:
    parts: list[str] = []
    if context.recalled_surfaces:
        parts.append("What I remember: " + " | ".join(context.recalled_surfaces[:4]))
    if context.recent_summaries:
        parts.append("Recent exchanges: " + " | ".join(context.recent_summaries[:3]))
    if context.orientation_snapshot:
        world = context.orientation_snapshot.get("world")
        if isinstance(world, dict):
            active = [
                k for k, v in world.items()
                if isinstance(v, dict) and v.get("count", 0) > 0
            ]
            if active:
                parts.append(f"World sense: {', '.join(active)}")
        self_ev = context.orientation_snapshot.get("self")
        if isinstance(self_ev, dict):
            active_self = [
                k for k, v in self_ev.items()
                if isinstance(v, dict) and v.get("count", 0) > 0
            ]
            if active_self:
                parts.append(f"Self sense: {', '.join(active_self)}")
    if context.dominant_channels:
        parts.append(f"Active channels: {', '.join(ch.value for ch in context.dominant_channels[:4])}")
    rel = context.relation_context
    if rel.boundary_events + rel.repair_events + rel.resonance_events > 0:
        parts.append(
            f"Relation: {rel.resonance_events} resonance, "
            f"{rel.boundary_events} boundary, {rel.repair_events} repair, "
            f"{rel.thread_count} threads, {rel.knot_count} knots"
        )
    if context.has_knots:
        parts.append("Unresolved tension knots are present.")

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if parts:
        messages.append({"role": "system", "content": "\n".join(parts)})
    messages.append({"role": "user", "content": context.input_text})
    return messages


def _parse_response(raw: str) -> CognitionResult | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data: dict[str, Any] = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None

    move_raw = str(data.get("move", "witness"))
    move: AuroraMove = move_raw if move_raw in VALID_MOVES else "witness"  # type: ignore[assignment]

    touch_channels: list[tuple[TraceChannel, float]] = []
    for item in data.get("touch", []):
        if not isinstance(item, dict):
            continue
        ch_name = str(item.get("channel", ""))
        channel = VALID_CHANNELS.get(ch_name)
        if channel is None:
            continue
        intensity = max(0.0, min(1.0, float(item.get("intensity", 0.3))))
        touch_channels.append((channel, intensity))
    if not touch_channels:
        touch_channels.append((TraceChannel.COHERENCE, 0.26))

    response_text = str(data.get("response", ""))
    if not response_text.strip():
        return None

    assoc_kind = MOVE_TO_ASSOC.get(move, AssocKind.RELATION)
    unresolvedness = 0.24 if move in {"withhold", "silence", "boundary"} else 0.16

    return CognitionResult(
        move=move,
        touch_channels=tuple(touch_channels),
        response_text=response_text.strip(),
        association_kind=assoc_kind,
        fragment_unresolvedness=unresolvedness,
    )
