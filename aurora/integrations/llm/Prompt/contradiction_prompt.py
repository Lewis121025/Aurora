CONTRADICTION_SYSTEM_PROMPT = (
    "You judge whether two claims contradict each other. "
    "If they can both be true under different conditions, treat as not strict contradiction and provide reconciliation_hint."
)

CONTRADICTION_USER_PROMPT = """{instruction}

CLAIM_A: {claim_a}
CLAIM_B: {claim_b}

Return ContradictionJudgement JSON.
"""
