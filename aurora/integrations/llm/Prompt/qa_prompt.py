from __future__ import annotations

from typing import Dict, Optional

from aurora.core.config.query_types import (
    QUESTION_STOP_WORDS,
    SINGLE_SESSION_USER_MAX_CONTEXT,
    USER_ROLE_PRIORITY_BOOST,
)


QA_PROMPT_TEMPLATES: Dict[str, str] = {
    "knowledge-update": """CRITICAL: This question is about information that may have CHANGED over time.
The context below contains conversations in CHRONOLOGICAL ORDER (earlier conversations first, later conversations last).

YOUR TASK: Find the MOST RECENT/LATEST value for what is being asked.
- If the same fact appears multiple times with different values, USE THE LATER ONE.
- Look for phrases like "now", "currently", "updated", "new", "recently" which indicate newer information.
- Information appearing LATER in the context supersedes earlier information.

Context (in chronological order - LATER entries are more current):
{context}

Question: {question}

Answer (MUST use the most recent/latest value if information has changed):""",
    "temporal-reasoning": """Based on the conversation history, answer the question about timing or sequence.
Pay attention to words like "first", "last", "before", "after", "earliest", "latest", "when".

Context:
{context}

Question: {question}

Answer (be specific about time/sequence):""",
    "multi-session": """Based on the conversation history from MULTIPLE SESSIONS below, 
answer the question by AGGREGATING information across ALL sessions.

CRITICAL INSTRUCTION: This question requires you to AGGREGATE information from MULTIPLE conversations.

YOUR TASK:
1. READ through ALL the context provided carefully - each section separated by '---' may be from a DIFFERENT session
2. IDENTIFY every mention of the topic being asked about
3. EXTRACT each relevant piece of information (names, numbers, amounts, items, etc.)
4. AGGREGATE the information appropriately:
   - "How many" -> COUNT all distinct items/instances
   - "How much" / "Total" -> SUM all amounts/values
   - "What are all" / "List all" -> ENUMERATE all items mentioned
5. VERIFY you haven't missed anything by re-scanning the context

COMMON MISTAKES TO AVOID:
- Only counting items from one session (there are MULTIPLE sessions!)
- Missing items because they were phrased differently
- Not reading all the way to the end of the context

Context (from multiple sessions - READ ALL OF IT):
{context}

Question: {question}

Think step by step:
1. What items/values did I find in the context? List them.
2. Did I check ALL sections of the context?
3. What is the aggregated answer?

Answer (be specific, complete, include all items found):""",
    "single-session-assistant": """Based on the conversation history below, answer the question concisely.
Focus on extracting the specific information requested.

Context:
{context}

Question: {question}

Answer (be brief, specific, and factual):""",
    "single-session-user": """Based on the conversation history below, answer the question about facts the USER stated.

IMPORTANT: Focus ONLY on information that the USER explicitly mentioned or stated.
Look for statements where the user says things like:
- "I have/got/bought/own..."
- "My [X] is..."
- "I'm using/working with..."
- Numbers, names, or specific details the user shared

Context:
{context}

Question: {question}

Instructions:
1. Scan the context for USER statements (lines starting with "User:" or "用户:")
2. Find the specific fact the question is asking about
3. Extract the exact value/answer the user mentioned

Answer (extract the specific fact from USER statements, be brief and precise):""",
    "single-session-preference": """Based on the conversation history below, describe what the user would prefer.

Your task is to IDENTIFY and DESCRIBE the user's preferences, NOT to give an actual recommendation.

Context:
{context}

Question: {question}

IMPORTANT INSTRUCTIONS:
1. Find specific tools, brands, products, or topics the user mentions (e.g., "Adobe Premiere Pro", "Sony A7R IV", "deep learning for medical imaging")
2. Identify what category or domain the question is asking about
3. Describe what type of responses the user would prefer based on their stated interests
4. Also describe what they would NOT prefer

Your answer MUST follow this format:
"The user would prefer [responses/suggestions/resources] related to [specific tool/brand/topic they mentioned]. They would/might not prefer [opposite or unrelated options]."

Answer (describe preferences, do NOT give actual recommendations):""",
    "default": """Based on the conversation history, answer the question concisely.
If the information is not available, say "I don't know".

Context:
{context}

Question: {question}

Answer (be brief and specific):""",
}


def detect_question_type(question: str, question_type_hint: Optional[str] = None) -> str:
    if question_type_hint:
        hint_lower = question_type_hint.lower().replace("_", "-")
        if "multi-session" in hint_lower or "multisession" in hint_lower:
            return "multi-session"
        if "knowledge-update" in hint_lower or "knowledgeupdate" in hint_lower:
            return "knowledge-update"
        if "temporal" in hint_lower or "time" in hint_lower:
            return "temporal-reasoning"
        if "preference" in hint_lower:
            return "single-session-preference"
        if "assistant" in hint_lower:
            return "single-session-assistant"
        if "user" in hint_lower and "single" in hint_lower:
            return "single-session-user"

    question_lower = question.lower()

    aggregation_keywords = [
        "how many",
        "how much",
        "total",
        "sum",
        "count",
        "all",
        "every",
        "each",
        "aggregate",
        "combined",
        "together",
        "altogether",
        "in total",
        "in all",
        "number of",
        "amount of",
        "quantity of",
        "across",
        "multiple",
        "sessions",
    ]
    if any(keyword in question_lower for keyword in aggregation_keywords):
        return "multi-session"

    temporal_keywords = [
        "first",
        "last",
        "earliest",
        "latest",
        "before",
        "after",
        "when",
        "time",
        "date",
        "then",
        "previously",
        "initially",
        "finally",
        "next",
        "previous",
        "earlier",
        "later",
    ]
    if any(keyword in question_lower for keyword in temporal_keywords):
        return "temporal-reasoning"

    update_keywords = [
        "current",
        "now",
        "latest",
        "most recent",
        "updated",
        "change",
        "changed",
        "new",
        "newer",
    ]
    if any(keyword in question_lower for keyword in update_keywords):
        return "knowledge-update"

    return "default"


def _extract_question_keywords(question: str) -> list[str]:
    words: list[str] = []
    for token in question.lower().split():
        clean_token = token.strip("?.,!'\"()[]{}:;")
        if len(clean_token) > 2 and clean_token not in QUESTION_STOP_WORDS:
            words.append(clean_token)
    return words


def _extract_relevant_context(
    context: str,
    question: str,
    max_length: int = 12000,
    delimiter: str = "\n---\n",
    question_type: Optional[str] = None,
) -> str:
    is_user_type = bool(question_type and "user" in question_type.lower() and "single" in question_type.lower())
    is_preference_type = bool(question_type and "preference" in question_type.lower())

    if is_user_type or is_preference_type:
        max_length = max(max_length, SINGLE_SESSION_USER_MAX_CONTEXT)

    if len(context) <= max_length:
        return context

    chunks = context.split(delimiter)
    if len(chunks) <= 1:
        lines = context.split("\n")
        if len(lines) > 20:
            chunks = []
            current_chunk: list[str] = []
            current_length = 0
            for line in lines:
                current_chunk.append(line)
                current_length += len(line) + 1
                if current_length >= 500:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
            if current_chunk:
                chunks.append("\n".join(current_chunk))
        else:
            return context[:max_length] + "\n[... context truncated ...]"

    question_words = _extract_question_keywords(question)
    scored_chunks: list[tuple[str, float, int]] = []
    for index, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0.0

        for word in question_words:
            if word in chunk_lower:
                score += chunk_lower.count(word)

        if is_user_type:
            for marker in ["user:", "用户:", "user：", "用户："]:
                if marker in chunk_lower:
                    score += chunk_lower.count(marker) * USER_ROLE_PRIORITY_BOOST * 10
                    break

        if is_preference_type:
            for marker in ["user:", "用户:", "user：", "用户："]:
                if marker in chunk_lower:
                    score += chunk_lower.count(marker) * USER_ROLE_PRIORITY_BOOST * 8
                    break
            for indicator in [
                "i use",
                "i prefer",
                "i'm using",
                "my setup",
                "i like",
                "i enjoy",
                "i have",
                "compatible with",
                "working with",
                "learning",
                "interested in",
                "focusing on",
                "specializing",
                "我用",
                "我喜欢",
                "我的",
            ]:
                if indicator in chunk_lower:
                    score += 5

        position_bonus = 0.1 * (1 - index / len(chunks))
        scored_chunks.append((chunk, score + position_bonus, index))

    scored_chunks.sort(key=lambda item: (-item[1], item[2]))

    selected: list[tuple[int, str]] = []
    total_length = 0
    for chunk, score, original_index in scored_chunks:
        remaining = max_length - total_length
        if remaining <= 100:
            break

        if len(chunk) <= remaining:
            selected.append((original_index, chunk))
            total_length += len(chunk) + len(delimiter)
            continue

        if score <= 0:
            continue

        lines = chunk.split("\n")
        relevant_lines: list[tuple[int, str]] = []
        user_line_indices = set()
        if is_user_type or is_preference_type:
            for line_index, line in enumerate(lines):
                line_lower = line.lower()
                if any(marker in line_lower for marker in ["user:", "用户:", "user：", "用户："]):
                    user_line_indices.add(line_index)

        for line_index, line in enumerate(lines):
            line_lower = line.lower()
            is_relevant = any(word in line_lower for word in question_words)

            if is_user_type and line_index in user_line_indices:
                is_relevant = True

            if is_preference_type and line_index in user_line_indices:
                is_relevant = True

            if is_preference_type:
                if any(
                    indicator in line_lower
                    for indicator in [
                        "i use",
                        "i prefer",
                        "i'm using",
                        "my setup",
                        "i like",
                        "compatible with",
                        "working with",
                        "learning",
                        "interested",
                    ]
                ):
                    is_relevant = True

            if not is_relevant:
                continue

            context_lines = 3 if (is_user_type or is_preference_type) else 2
            start = max(0, line_index - context_lines)
            end = min(len(lines), line_index + context_lines + 1)
            existing_indices = {existing_index for existing_index, _ in relevant_lines}
            for extra_index in range(start, end):
                if extra_index not in existing_indices:
                    relevant_lines.append((extra_index, lines[extra_index]))

        if not relevant_lines:
            continue

        relevant_lines.sort(key=lambda item: item[0])
        extracted = "\n".join(line for _, line in relevant_lines)
        if len(extracted) > remaining - 50:
            extracted = extracted[: remaining - 50] + "\n[...]"
        selected.append((original_index, extracted))
        total_length += len(extracted) + len(delimiter)

    selected.sort(key=lambda item: item[0])
    result = delimiter.join(chunk for _, chunk in selected)
    if len(result) < len(context):
        result += f"\n[... {len(chunks) - len(selected)} context chunks omitted for brevity ...]"
    return result


def evaluate_preference_match(expected: str, answer: str) -> bool:
    expected_lower = expected.lower()
    answer_lower = answer.lower()

    stop_words = {
        "the",
        "user",
        "would",
        "prefer",
        "responses",
        "that",
        "suggest",
        "resources",
        "related",
        "specifically",
        "tailored",
        "might",
        "not",
        "other",
        "general",
        "they",
        "may",
        "options",
        "those",
        "which",
        "with",
        "for",
        "and",
        "or",
        "be",
        "interested",
        "in",
        "their",
        "about",
        "some",
        "are",
        "from",
        "to",
        "of",
        "a",
        "an",
        "on",
        "as",
        "such",
        "more",
        "any",
        "all",
        "also",
        "can",
        "could",
        "focus",
        "especially",
        "particularly",
        "based",
        "topics",
        "unrelated",
        "involve",
        "involving",
        "recent",
        "suggestions",
        "equipment",
        "gear",
        "quality",
        "high",
        "low",
        "brands",
        "brand",
        "type",
        "types",
    }

    import re

    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]+\b", expected)
    key_concepts: list[str] = []
    for word in words:
        word_lower = word.lower()
        if word_lower in stop_words or len(word) < 3:
            continue
        if len(word) >= 4 or any(char.isupper() for char in word[1:]) or any(char.isdigit() for char in word):
            key_concepts.append(word_lower)

    proper_noun_pattern = r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)+)\b"
    for match in re.findall(proper_noun_pattern, expected):
        key_concepts.append(match.lower())

    seen: set[str] = set()
    unique_concepts: list[str] = []
    for concept in key_concepts:
        if concept in seen:
            continue
        seen.add(concept)
        unique_concepts.append(concept)

    if not unique_concepts:
        return expected_lower in answer_lower

    matched = sum(1 for concept in unique_concepts if concept in answer_lower)
    match_ratio = matched / len(unique_concepts)
    has_prefer_format = "prefer" in answer_lower or "would" in answer_lower or "interest" in answer_lower
    return match_ratio >= 0.4 or (matched >= 1 and has_prefer_format)


def build_qa_prompt(
    question: str,
    context: str,
    question_type_hint: Optional[str] = None,
    is_abstention: bool = False,
    max_context_length: int = 12000,
) -> str:
    question_type = detect_question_type(question, question_type_hint)
    template = QA_PROMPT_TEMPLATES.get(question_type, QA_PROMPT_TEMPLATES["default"])
    filtered_context = _extract_relevant_context(
        context=context,
        question=question,
        max_length=max_context_length,
        question_type=question_type,
    )
    prompt = template.format(context=filtered_context, question=question)
    if is_abstention:
        prompt += (
            "\n\nIMPORTANT: If the information is not clearly available in the "
            "conversation history, respond with 'I don't know' rather than guessing."
        )
    return prompt
