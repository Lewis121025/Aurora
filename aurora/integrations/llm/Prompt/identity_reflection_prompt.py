IDENTITY_REFLECTION_SYSTEM_PROMPT = (
    "You help the agent reflect on its identity based on experiences. "
    "Be honest about capabilities and limitations. Identify growth areas."
)

IDENTITY_REFLECTION_USER_PROMPT = """{instruction}

RECENT THEMES:
{themes}

CAPABILITY BELIEFS:
{capabilities}

RELATIONSHIP SUMMARY:
{relationships}

Generate a self-reflection. Return IdentityReflection JSON.
"""
