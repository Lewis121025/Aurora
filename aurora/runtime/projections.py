"""Aurora vNext runtime projection helpers."""

from __future__ import annotations

from aurora.runtime.contracts import RecallHit, SubjectMemoryState


def build_memory_projection(
    state: SubjectMemoryState,
    recall_hits: tuple[RecallHit, ...],
) -> tuple[str, str, tuple[RecallHit, ...]]:
    """Build the hot-path context projection."""
    return (
        _state_segment(state),
        _episode_segment(state),
        recall_hits[:5],
    )


def _state_segment(state: SubjectMemoryState) -> str:
    lines = ["[SUBJECT_MEMORY_STATE]"]
    if state.semantic_self_model:
        lines.append("Self model:")
        lines.extend(f"- {item.text}" for item in state.semantic_self_model[-6:])
    if state.semantic_world_model:
        lines.append("World model:")
        lines.extend(f"- {item.text}" for item in state.semantic_world_model[-4:])
    if state.procedural_memory:
        lines.append("Procedural memory:")
        lines.extend(f"- {item.text}" for item in state.procedural_memory[-4:])
    cognition = state.active_cognition
    if any((cognition.beliefs, cognition.goals, cognition.conflicts, cognition.intentions, cognition.commitments)):
        lines.append("Active cognition:")
        if cognition.beliefs:
            lines.append(f"- beliefs: {'; '.join(cognition.beliefs)}")
        if cognition.goals:
            lines.append(f"- goals: {'; '.join(cognition.goals)}")
        if cognition.conflicts:
            lines.append(f"- conflicts: {'; '.join(cognition.conflicts)}")
        if cognition.intentions:
            lines.append(f"- intentions: {'; '.join(cognition.intentions)}")
        if cognition.commitments:
            lines.append(f"- commitments: {'; '.join(cognition.commitments)}")
    affect = state.affective_state
    if affect.active_feelings or affect.mood != "neutral":
        lines.append("Affective state:")
        lines.append(f"- mood: {affect.mood}")
        if affect.active_feelings:
            lines.append(f"- feelings: {'; '.join(affect.active_feelings)}")
    if state.narrative_state.arcs:
        lines.append("Narrative arcs:")
        for arc in state.narrative_state.arcs[-3:]:
            lines.append(f"- {arc.theme}: {arc.storyline}")
    return "\n".join(lines)


def _episode_segment(state: SubjectMemoryState) -> str:
    lines = ["[RECENT_EPISODES]"]
    if not state.recent_episodes:
        lines.append("- none")
        return "\n".join(lines)
    for episode in state.recent_episodes[-4:]:
        lines.append(f"- {episode.text}")
    return "\n".join(lines)
