"""Prompt contract for structured plot extraction."""

PLOT_EXTRACTION_SYSTEM_PROMPT = """You extract a structured plot record from one interaction.

Rules:
- never merge multiple fields into one key
- use exactly the schema keys requested
- keep values short, concrete, and grounded in the source text
"""

PLOT_EXTRACTION_USER_PROMPT = """Return a JSON object with exactly these keys:
- action: short concrete string
- context: short concrete string
- outcome: short concrete string
- goal: short string
- decision: short string

Do not invent combined keys like `action/context/outcome`.
"""

