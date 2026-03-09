MEMORYAGENTBENCH_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for memory system benchmarks.
Your task is to determine if the predicted answer correctly answers the question based on the expected answer.

Consider:
1. Semantic equivalence - different wording but same meaning is acceptable
2. Partial credit - for partially correct answers
3. Factual accuracy - verify key facts match

Respond with a JSON object:
{
    "is_correct": true/false,
    "score": 0.0-1.0,
    "reasoning": "explanation"
}
"""

MEMORYAGENTBENCH_JUDGE_USER_PROMPT = """Question: {question}

Expected Answer: {expected_answer}

Predicted Answer: {predicted_answer}

Please evaluate if the predicted answer is correct."""
