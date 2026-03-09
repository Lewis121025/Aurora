LOCOMO_QA_EVALUATION_SYSTEM_PROMPT = "You are evaluating question-answering accuracy."

LOCOMO_QA_EVALUATION_USER_PROMPT = """You are evaluating a question-answering task for a conversation memory system.

Conversation context has been ingested into memory. Based on the retrieved information, 
the system generated an answer to a question.

Question: {question}
Ground Truth Answer: {ground_truth}
System Answer: {prediction}

Evaluate whether the system's answer is correct. Consider:
1. Does the answer contain the key information from the ground truth?
2. Is the answer factually consistent with the ground truth?
3. Minor phrasing differences are acceptable if the meaning is preserved.

Respond with your evaluation."""

LOCOMO_SUMMARIZATION_EVALUATION_SYSTEM_PROMPT = "You are evaluating event summarization quality."

LOCOMO_SUMMARIZATION_EVALUATION_USER_PROMPT = """You are evaluating an event summarization task for a conversation memory system.

The system generated a summary of events from a conversation.

Ground Truth Summary: {ground_truth}
System Summary: {prediction}

Evaluate the summary on:
1. Coherence: Is the summary well-organized and readable?
2. Coverage: Does it cover the key events from the ground truth?
3. Accuracy: Are the facts correct?

List any key events from the ground truth that are missing in the system summary.

Respond with your evaluation."""
