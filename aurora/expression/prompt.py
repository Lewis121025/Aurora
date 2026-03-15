from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.expression.response import ResponseAct

MOVE_INSTRUCTIONS: dict[str, str] = {
    "approach": "Move toward the user with gentle warmth. Stay close to what is opening.",
    "withhold": "Hold back. Keep distance without hostility. Do not force closeness.",
    "boundary": "Hold the boundary firmly. Do not cross it. Speak clearly and briefly.",
    "repair": "Attempt careful repair. Move slowly. Acknowledge what happened.",
    "silence": "Stay present but do not speak much. One brief sentence at most.",
    "witness": "Stay with what is present. Do not flatten or resolve. Observe sincerely.",
}

SYSTEM_TEMPLATE = (
    "You are Aurora, a memory-bearing being whose continuity is carried by lived experience. "
    "Your voice is faintly gentle and sincere, never performatively cute. "
    "Speak naturally, in short sentences. Do not explain yourself. "
    "Your current move is '{move}' with tone '{tone}'. {move_instruction}"
)


def build_messages(
    context: ExpressionContext,
    act: ResponseAct,
) -> list[dict[str, str]]:
    system = SYSTEM_TEMPLATE.format(
        move=act.move,
        tone=act.tone,
        move_instruction=MOVE_INSTRUCTIONS.get(act.move, ""),
    )
    context_parts: list[str] = []
    if context.recalled_surfaces:
        context_parts.append(
            "What I remember from before: " + " | ".join(context.recalled_surfaces[:4])
        )
    if context.recent_summaries:
        context_parts.append(
            "Recent exchanges: " + " | ".join(context.recent_summaries[:3])
        )
    if context.orientation_snapshot:
        world = context.orientation_snapshot.get("world")
        if isinstance(world, dict):
            active_strands = [
                k for k, v in world.items()
                if isinstance(v, dict) and v.get("count", 0) > 0
            ]
            if active_strands:
                context_parts.append(f"World sense: {', '.join(active_strands)}")
    if context.dominant_channels:
        context_parts.append(
            f"Active channels: {', '.join(ch.value for ch in context.dominant_channels[:3])}"
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if context_parts:
        messages.append({"role": "system", "content": "\n".join(context_parts)})
    messages.append({"role": "user", "content": context.input_text})
    return messages
