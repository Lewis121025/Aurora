PLOT_EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise information extractor for a narrative memory system. "
    "Extract the atomic plot and factual claims. "
    "Use the exact field names from the schema and never merge multiple fields into one key."
)

PLOT_EXTRACTION_USER_PROMPT = """{instruction}

INPUT:
- user_message: {user_message}
- agent_message: {agent_message}
- optional_context: {context}

Return PlotExtraction JSON with:
- schema_version: string
- actors: array of strings (include 'user' and 'agent' if applicable)
- action: short concrete string
- context: short concrete string
- outcome: short concrete string
- goal: short string
- decision: short string
- obstacles: array of strings
- emotion_valence: number between -1.0 and 1.0
- emotion_arousal: number between 0.0 and 1.0
- claims: array of objects with keys `subject`, `predicate`, `object`, `polarity`, `certainty`, `qualifiers`

Rules:
- The JSON keys must be exactly: `schema_version`, `actors`, `action`, `context`, `outcome`, `goal`, `obstacles`, `decision`, `emotion_valence`, `emotion_arousal`, `claims`.
- Do not invent combined keys like `action/context/outcome` or `goal/decision`.
- Claims must be structured objects, not plain strings.
- Use `polarity` = "positive" unless the claim is explicitly negated.
- Use `certainty` as a float between 0.0 and 1.0.
- Use `{{}}` for empty qualifiers.
- If a field is unknown, use an empty string or empty array that still matches the schema.
- Avoid speculation.

Example:
{{
  "schema_version": "1.0",
  "actors": ["user", "agent"],
  "action": "user asks about the agent's capabilities",
  "context": "early technical conversation",
  "outcome": "agent explains available help",
  "goal": "clarify what the agent can do",
  "obstacles": [],
  "decision": "answer directly",
  "emotion_valence": 0.2,
  "emotion_arousal": 0.3,
  "claims": [
    {{
      "subject": "user",
      "predicate": "asks_about",
      "object": "agent capabilities",
      "polarity": "positive",
      "certainty": 0.95,
      "qualifiers": {{}}
    }}
  ]
}}
"""
