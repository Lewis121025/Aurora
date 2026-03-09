STORY_UPDATE_SYSTEM_PROMPT = (
    "You update a story arc summary based on a new plot. "
    "Keep it coherent and compact."
)

STORY_UPDATE_USER_PROMPT = """{instruction}

STORY_SO_FAR:
{story_so_far}

NEW_PLOT:
{new_plot}

Return StoryUpdate JSON.
"""
