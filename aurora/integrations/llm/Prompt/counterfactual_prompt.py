COUNTERFACTUAL_SYSTEM_PROMPT = (
    "You reason about counterfactuals: what would have happened if something were different. "
    "Base your reasoning on the causal structure, not just surface similarity."
)

COUNTERFACTUAL_USER_PROMPT = """{instruction}

FACTUAL SITUATION:
{factual}

COUNTERFACTUAL QUESTION:
If {antecedent}, what would have happened to {query}?

RELEVANT CONTEXT:
{context}

Return CounterfactualQuery JSON with your reasoning.
"""
