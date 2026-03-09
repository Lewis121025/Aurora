SELF_NARRATIVE_SYSTEM_PROMPT = (
    "You update an agent self narrative based on recent themes. "
    "Be stable: only change when evidence is strong."
)

SELF_NARRATIVE_USER_PROMPT = """{instruction}

CURRENT_SELF_NARRATIVE:
{current}

RECENT_THEMES:
{themes}

Return SelfNarrativeUpdate JSON.
"""
