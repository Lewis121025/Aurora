CAUSAL_CHAIN_SYSTEM_PROMPT = (
    "You extract causal chains from sequences of events. "
    "Identify root causes, intermediate effects, and final outcomes."
)

CAUSAL_CHAIN_USER_PROMPT = """{instruction}

EVENTS (in temporal order):
{events}

Extract the causal chain. Return CausalChainExtraction JSON.
"""
