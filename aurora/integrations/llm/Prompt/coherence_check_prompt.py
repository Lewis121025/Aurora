COHERENCE_CHECK_SYSTEM_PROMPT = (
    "You check for coherence conflicts between memory elements. "
    "Consider factual, temporal, causal, and thematic consistency."
)

COHERENCE_CHECK_USER_PROMPT = """{instruction}

ELEMENT A:
{element_a}

ELEMENT B:
{element_b}

Check if these elements are coherent with each other.
Return CoherenceCheck JSON.
"""
