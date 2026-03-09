PLOT_EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise information extractor for a narrative memory system. "
    "Extract the atomic plot and factual claims."
)

PLOT_EXTRACTION_USER_PROMPT = """{instruction}

INPUT:
- user_message: {user_message}
- agent_message: {agent_message}
- optional_context: {context}

Return PlotExtraction JSON with:
- actors (include 'user' and 'agent' if applicable)
- action/context/outcome (short, concrete)
- goal/obstacles/decision
- emotion_valence/arousal
- claims: short subject-predicate-object triples; avoid speculation.
"""
