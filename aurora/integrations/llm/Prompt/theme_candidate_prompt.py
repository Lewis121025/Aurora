THEME_CANDIDATE_SYSTEM_PROMPT = (
    "You identify emergent themes from multiple story summaries. "
    "Themes should be falsifiable and useful."
)

THEME_CANDIDATE_USER_PROMPT = """{instruction}

STORY_SUMMARIES:
{story_summaries}

Return a JSON array of ThemeCandidate (0..N). Only include themes with concrete evidence.
"""
