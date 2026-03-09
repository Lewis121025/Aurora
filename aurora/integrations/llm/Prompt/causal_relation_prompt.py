CAUSAL_RELATION_SYSTEM_PROMPT = (
    "You identify causal relationships between events. "
    "Focus on actual causation, not just correlation or temporal sequence. "
    "Be conservative: only claim causation when there is clear evidence."
)

CAUSAL_RELATION_USER_PROMPT = """{instruction}

EVENT A:
{event_a}

EVENT B:
{event_b}

CONTEXT:
{context}

Analyze if there is a causal relationship between A and B.
Return CausalRelation JSON.
"""
