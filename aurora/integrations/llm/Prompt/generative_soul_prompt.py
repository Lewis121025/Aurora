from __future__ import annotations

from typing import Dict, Iterable, Sequence


GEN_SOUL_MEANING_SYSTEM_PROMPT = """You are extracting a structured EventFrame for Aurora v4.
Return valid JSON only. Use the provided dynamic axis names exactly.
Scores must be bounded. Avoid inventing axes not listed in the schema.
"""


def build_gen_soul_meaning_user_prompt(
    *,
    text: str,
    axis_names: Sequence[str],
    recent_tags: Sequence[str] | None = None,
) -> str:
    recent = ", ".join(recent_tags or [])
    return (
        "Text:\n"
        f"{text}\n\n"
        "Dynamic axis names:\n"
        f"{', '.join(axis_names)}\n\n"
        "Recent tags:\n"
        f"{recent or 'none'}\n\n"
        "Return MeaningFramePayloadV4 JSON."
    )


GEN_SOUL_PERSONA_AXIS_SYSTEM_PROMPT = """You are deriving persona axes from a profile for Aurora v4.
Return compact JSON only. Prefer high-signal axes and avoid duplicates.
"""


def build_gen_soul_persona_axis_user_prompt(*, profile_text: str) -> str:
    return (
        "Profile text:\n"
        f"{profile_text}\n\n"
        "Return PersonaAxisPayload JSON with 0-6 persona axes."
    )


GEN_SOUL_SUMMARY_SYSTEM_PROMPT = """You are writing a concise narrative summary for Aurora v4.
Return JSON only. Keep the wording coherent and grounded in the supplied mode and axes.
"""


def build_gen_soul_summary_user_prompt(
    *,
    current_mode: str,
    salient_axes: Sequence[str],
    recent_texts: Sequence[str],
    pressure: float,
) -> str:
    recent = "\n".join(f"- {text}" for text in recent_texts[-4:])
    return (
        f"Current mode: {current_mode}\n"
        f"Pressure: {pressure:.3f}\n"
        f"Salient axes: {', '.join(salient_axes)}\n"
        f"Recent texts:\n{recent or '- none'}\n\n"
        "Return NarrativeSummaryPayloadV4 JSON."
    )


GEN_SOUL_REPAIR_SYSTEM_PROMPT = """You are generating a repair narration for Aurora v4.
Return JSON only. Preserve emotional continuity and mention the repair mode.
"""


def build_gen_soul_repair_user_prompt(
    *,
    mode: str,
    plot_text: str,
    salient_axes: Sequence[str],
    dissonance_total: float,
) -> str:
    return (
        f"Repair mode: {mode}\n"
        f"Trigger plot: {plot_text}\n"
        f"Salient axes: {', '.join(salient_axes)}\n"
        f"Dissonance: {dissonance_total:.3f}\n\n"
        "Return RepairNarrationPayloadV4 JSON."
    )


GEN_SOUL_DREAM_SYSTEM_PROMPT = """You are generating a dream narration for Aurora v4.
Return JSON only. Keep it symbolic but related to the supplied fragments.
"""


def build_gen_soul_dream_user_prompt(*, operator: str, fragment_tags: Sequence[str]) -> str:
    return (
        f"Dream operator: {operator}\n"
        f"Fragment tags: {', '.join(fragment_tags)}\n\n"
        "Return DreamNarrationPayloadV4 JSON."
    )


GEN_SOUL_MODE_LABEL_SYSTEM_PROMPT = """You are labeling a self mode for Aurora v4.
Return JSON only. The label should be short and human readable.
"""


def build_gen_soul_mode_label_user_prompt(*, prototype_axes: Dict[str, float]) -> str:
    pairs = "\n".join(f"- {key}: {value:+.3f}" for key, value in prototype_axes.items())
    return f"Axis prototype:\n{pairs}\n\nReturn ModeLabelPayloadV4 JSON."


GEN_SOUL_AXIS_MERGE_SYSTEM_PROMPT = """You are deciding whether two persona axes in Aurora v4 should merge.
Return JSON only. Merge only when the two axes express the same enduring dimension.
"""


def build_gen_soul_axis_merge_user_prompt(
    *,
    canonical_name: str,
    canonical_desc: str,
    alias_name: str,
    alias_desc: str,
    evidence_overlap: Iterable[str],
) -> str:
    overlap = "\n".join(f"- {item}" for item in evidence_overlap)
    return (
        f"Canonical axis: {canonical_name}\n"
        f"Canonical description: {canonical_desc}\n"
        f"Candidate alias: {alias_name}\n"
        f"Candidate description: {alias_desc}\n"
        f"Shared evidence:\n{overlap or '- none'}\n\n"
        "Return AxisMergeJudgementPayload JSON."
    )
