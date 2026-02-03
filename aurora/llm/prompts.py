from __future__ import annotations

from typing import Any, Dict, Optional
from .schemas import SCHEMA_VERSION


def _json_instruction(model_name: str) -> str:
    return (
        "You MUST output ONLY valid JSON. No markdown. No extra keys. "
        f"Schema version: {SCHEMA_VERSION}. Output must conform to {model_name}."
    )


PLOT_EXTRACTION_SYSTEM = (
    "You are a precise information extractor for a narrative memory system. "
    "Extract the atomic plot and factual claims."
)

PLOT_EXTRACTION_USER = """{instruction}

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

STORY_UPDATE_SYSTEM = (
    "You update a story arc summary based on a new plot. "
    "Keep it coherent and compact."
)

STORY_UPDATE_USER = """{instruction}

STORY_SO_FAR:
{story_so_far}

NEW_PLOT:
{new_plot}

Return StoryUpdate JSON.
"""

THEME_CANDIDATE_SYSTEM = (
    "You identify emergent themes from multiple story summaries. "
    "Themes should be falsifiable and useful."
)

THEME_CANDIDATE_USER = """{instruction}

STORY_SUMMARIES:
{story_summaries}

Return a JSON array of ThemeCandidate (0..N). Only include themes with concrete evidence.
"""

SELF_NARRATIVE_SYSTEM = (
    "You update an agent self narrative based on recent themes. "
    "Be stable: only change when evidence is strong."
)

SELF_NARRATIVE_USER = """{instruction}

CURRENT_SELF_NARRATIVE:
{current}

RECENT_THEMES:
{themes}

Return SelfNarrativeUpdate JSON.
"""

CONTRADICTION_SYSTEM = (
    "You judge whether two claims contradict each other. "
    "If they can both be true under different conditions, treat as not strict contradiction and provide reconciliation_hint."
)

CONTRADICTION_USER = """{instruction}

CLAIM_A: {claim_a}
CLAIM_B: {claim_b}

Return ContradictionJudgement JSON.
"""


# -----------------------------------------------------------------------------
# Causal Inference Prompts
# -----------------------------------------------------------------------------

CAUSAL_RELATION_SYSTEM = (
    "You identify causal relationships between events. "
    "Focus on actual causation, not just correlation or temporal sequence. "
    "Be conservative: only claim causation when there is clear evidence."
)

CAUSAL_RELATION_USER = """{instruction}

EVENT A:
{event_a}

EVENT B:
{event_b}

CONTEXT:
{context}

Analyze if there is a causal relationship between A and B.
Return CausalRelation JSON.
"""

CAUSAL_CHAIN_SYSTEM = (
    "You extract causal chains from sequences of events. "
    "Identify root causes, intermediate effects, and final outcomes."
)

CAUSAL_CHAIN_USER = """{instruction}

EVENTS (in temporal order):
{events}

Extract the causal chain. Return CausalChainExtraction JSON.
"""

COUNTERFACTUAL_SYSTEM = (
    "You reason about counterfactuals: what would have happened if something were different. "
    "Base your reasoning on the causal structure, not just surface similarity."
)

COUNTERFACTUAL_USER = """{instruction}

FACTUAL SITUATION:
{factual}

COUNTERFACTUAL QUESTION:
If {antecedent}, what would have happened to {query}?

RELEVANT CONTEXT:
{context}

Return CounterfactualQuery JSON with your reasoning.
"""

# -----------------------------------------------------------------------------
# Self-Narrative Prompts
# -----------------------------------------------------------------------------

CAPABILITY_ASSESSMENT_SYSTEM = (
    "You assess agent capabilities based on interaction outcomes. "
    "Be balanced: note both successes and areas for improvement."
)

CAPABILITY_ASSESSMENT_USER = """{instruction}

INTERACTION:
User: {user_message}
Agent: {agent_message}
Outcome: {outcome}

Assess what capabilities were demonstrated or what limitations were revealed.
Return CapabilityAssessment JSON.
"""

RELATIONSHIP_ASSESSMENT_SYSTEM = (
    "You assess the quality of agent-user relationships from interactions."
)

RELATIONSHIP_ASSESSMENT_USER = """{instruction}

ENTITY: {entity_id}
INTERACTION HISTORY SUMMARY: {history_summary}
LATEST INTERACTION:
User: {user_message}
Agent: {agent_message}

Assess the relationship quality. Return RelationshipAssessment JSON.
"""

IDENTITY_REFLECTION_SYSTEM = (
    "You help the agent reflect on its identity based on experiences. "
    "Be honest about capabilities and limitations. Identify growth areas."
)

IDENTITY_REFLECTION_USER = """{instruction}

RECENT THEMES:
{themes}

CAPABILITY BELIEFS:
{capabilities}

RELATIONSHIP SUMMARY:
{relationships}

Generate a self-reflection. Return IdentityReflection JSON.
"""

# -----------------------------------------------------------------------------
# Coherence Prompts
# -----------------------------------------------------------------------------

COHERENCE_CHECK_SYSTEM = (
    "You check for coherence conflicts between memory elements. "
    "Consider factual, temporal, causal, and thematic consistency."
)

COHERENCE_CHECK_USER = """{instruction}

ELEMENT A:
{element_a}

ELEMENT B:
{element_b}

Check if these elements are coherent with each other.
Return CoherenceCheck JSON.
"""


def render(template: str, **kwargs: Any) -> str:
    return template.format(**kwargs)


def instruction(model_name: str) -> str:
    return _json_instruction(model_name)


# -----------------------------------------------------------------------------
# Question-Answering Prompts (Type-Specific)
# -----------------------------------------------------------------------------

# Prompt templates for different question types
QA_PROMPT_TEMPLATES: Dict[str, str] = {
    'knowledge-update': """CRITICAL: This question is about information that may have CHANGED over time.
The context below contains conversations in CHRONOLOGICAL ORDER (earlier conversations first, later conversations last).

YOUR TASK: Find the MOST RECENT/LATEST value for what is being asked.
- If the same fact appears multiple times with different values, USE THE LATER ONE.
- Look for phrases like "now", "currently", "updated", "new", "recently" which indicate newer information.
- Information appearing LATER in the context supersedes earlier information.

Context (in chronological order - LATER entries are more current):
{context}

Question: {question}

Answer (MUST use the most recent/latest value if information has changed):""",

    'temporal-reasoning': """Based on the conversation history, answer the question about timing or sequence.
Pay attention to words like "first", "last", "before", "after", "earliest", "latest", "when".

Context:
{context}

Question: {question}

Answer (be specific about time/sequence):""",

    'multi-session': """Based on the conversation history from MULTIPLE SESSIONS below, 
answer the question by AGGREGATING information across ALL sessions.

CRITICAL INSTRUCTION: This question requires you to AGGREGATE information from MULTIPLE conversations.

YOUR TASK:
1. READ through ALL the context provided carefully - each section separated by '---' may be from a DIFFERENT session
2. IDENTIFY every mention of the topic being asked about
3. EXTRACT each relevant piece of information (names, numbers, amounts, items, etc.)
4. AGGREGATE the information appropriately:
   - "How many" → COUNT all distinct items/instances
   - "How much" / "Total" → SUM all amounts/values
   - "What are all" / "List all" → ENUMERATE all items mentioned
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

    'single-session-assistant': """Based on the conversation history below, answer the question concisely.
Focus on extracting the specific information requested.

Context:
{context}

Question: {question}

Answer (be brief, specific, and factual):""",

    'single-session-user': """Based on the conversation history below, answer the question about facts the USER stated.

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

    'single-session-preference': """Based on the conversation history below, describe what the user would prefer.

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

    'default': """Based on the conversation history, answer the question concisely.
If the information is not available, say "I don't know".

Context:
{context}

Question: {question}

Answer (be brief and specific):"""
}


def detect_question_type(question: str, question_type_hint: Optional[str] = None) -> str:
    """Detect question type from question text or use provided hint.
    
    Args:
        question: The question text
        question_type_hint: Optional explicit question type (e.g., from dataset)
        
    Returns:
        Question type key for prompt template selection
    """
    # If explicit type is provided, use it (with normalization)
    if question_type_hint:
        hint_lower = question_type_hint.lower().replace('_', '-')
        # Map common variations
        if 'multi-session' in hint_lower or 'multisession' in hint_lower:
            return 'multi-session'
        if 'knowledge-update' in hint_lower or 'knowledgeupdate' in hint_lower:
            return 'knowledge-update'
        if 'temporal' in hint_lower or 'time' in hint_lower:
            return 'temporal-reasoning'
        if 'preference' in hint_lower:
            return 'single-session-preference'
        if 'assistant' in hint_lower:
            return 'single-session-assistant'
        if 'user' in hint_lower and 'single' in hint_lower:
            return 'single-session-user'
    
    # Auto-detect from question text
    question_lower = question.lower()
    
    # Check for multi-session/aggregation indicators
    # These keywords indicate questions that need information from multiple sessions
    aggregation_keywords = [
        'how many', 'how much', 'total', 'sum', 'count', 'all', 'every', 'each',
        'aggregate', 'combined', 'together', 'altogether', 'in total', 'in all',
        'number of', 'amount of', 'quantity of', 'across', 'multiple', 'sessions'
    ]
    if any(kw in question_lower for kw in aggregation_keywords):
        return 'multi-session'
    
    # Check for temporal reasoning indicators
    temporal_keywords = [
        'first', 'last', 'earliest', 'latest', 'before', 'after',
        'when', 'time', 'date', 'then', 'previously', 'initially',
        'finally', 'next', 'previous', 'earlier', 'later'
    ]
    if any(kw in question_lower for kw in temporal_keywords):
        return 'temporal-reasoning'
    
    # Check for knowledge update indicators
    update_keywords = [
        'current', 'now', 'latest', 'most recent', 'updated',
        'change', 'changed', 'new', 'newer'
    ]
    if any(kw in question_lower for kw in update_keywords):
        return 'knowledge-update'
    
    # Default to generic prompt
    return 'default'


def _extract_question_keywords(question: str) -> list:
    """Extract meaningful keywords from a question for matching.
    
    Args:
        question: The question text
        
    Returns:
        List of keyword strings (lowercase)
    """
    from aurora.algorithms.constants import QUESTION_STOP_WORDS
    
    question_lower = question.lower()
    words = []
    for w in question_lower.split():
        # Clean punctuation
        clean_w = w.strip('?.,!\'\"()[]{}:;')
        if len(clean_w) > 2 and clean_w not in QUESTION_STOP_WORDS:
            words.append(clean_w)
    
    return words


def _extract_relevant_context(
    context: str,
    question: str,
    max_length: int = 12000,
    delimiter: str = '\n---\n',
    question_type: Optional[str] = None
) -> str:
    """Extract most relevant parts of context based on question keywords.
    
    This uses a simple but effective strategy:
    1. Split context into chunks (by delimiter)
    2. Score each chunk by keyword overlap with question
    3. Include highest-scoring chunks up to max_length
    
    Enhanced for single-session-user questions:
    - Prioritizes chunks containing user statements
    - Extracts more context around matching keywords
    - Uses a higher max_length
    
    Args:
        context: Full context string
        question: Question to answer
        max_length: Maximum output length
        delimiter: Chunk delimiter in context
        question_type: Optional question type hint for optimized extraction
        
    Returns:
        Filtered context with most relevant chunks
    """
    from aurora.algorithms.constants import (
        SINGLE_SESSION_USER_MAX_CONTEXT,
        USER_ROLE_PRIORITY_BOOST,
    )
    
    # For single-session-user, use longer context and special handling
    is_user_type = question_type and 'user' in question_type.lower() and 'single' in question_type.lower()
    
    # For single-session-preference, also use special handling
    is_preference_type = question_type and 'preference' in question_type.lower()
    
    if is_user_type or is_preference_type:
        max_length = max(max_length, SINGLE_SESSION_USER_MAX_CONTEXT)
    
    if len(context) <= max_length:
        return context
    
    # Split into chunks
    chunks = context.split(delimiter)
    if len(chunks) <= 1:
        # No delimiter found, try line-based splitting
        lines = context.split('\n')
        if len(lines) > 20:
            # Group lines into ~500 char chunks
            chunks = []
            current_chunk = []
            current_len = 0
            for line in lines:
                current_chunk.append(line)
                current_len += len(line) + 1
                if current_len >= 500:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_len = 0
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
        else:
            # Not enough lines to chunk, just truncate
            return context[:max_length] + "\n[... context truncated ...]"
    
    # Extract keywords from question
    question_words = _extract_question_keywords(question)
    
    # Score each chunk
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0.0
        
        # Base score from keyword matches
        for word in question_words:
            if word in chunk_lower:
                score += chunk_lower.count(word)
        
        # For single-session-user: boost user statements
        if is_user_type:
            # Check if chunk contains user statement markers
            user_markers = ['user:', '用户:', 'user：', '用户：']
            for marker in user_markers:
                if marker in chunk_lower:
                    # Count user statements
                    user_count = chunk_lower.count(marker)
                    score += user_count * USER_ROLE_PRIORITY_BOOST * 10
                    break
        
        # For single-session-preference: boost chunks with user preferences
        if is_preference_type:
            # Check for user statements
            user_markers = ['user:', '用户:', 'user：', '用户：']
            for marker in user_markers:
                if marker in chunk_lower:
                    user_count = chunk_lower.count(marker)
                    score += user_count * USER_ROLE_PRIORITY_BOOST * 8
                    break
            
            # Boost chunks mentioning preferences, tools, or brands
            preference_indicators = [
                'i use', 'i prefer', "i'm using", 'my setup', 'i like', 
                'i enjoy', 'i have', 'compatible with', 'working with',
                'learning', 'interested in', 'focusing on', 'specializing',
                '我用', '我喜欢', '我的'
            ]
            for indicator in preference_indicators:
                if indicator in chunk_lower:
                    score += 5  # Boost preference-related content
        
        # Give slight preference to earlier chunks (recency bias)
        # But not too much, since important info could be anywhere
        position_bonus = 0.1 * (1 - i / len(chunks))
        scored_chunks.append((chunk, score + position_bonus, i))
    
    # Sort by score (highest first), break ties by original position
    scored_chunks.sort(key=lambda x: (-x[1], x[2]))
    
    # Build result by adding chunks until max_length
    # If a chunk is too large, truncate it but still include the most relevant parts
    selected = []
    total_len = 0
    for chunk, score, orig_idx in scored_chunks:
        remaining = max_length - total_len
        if remaining <= 100:
            break
            
        if len(chunk) <= remaining:
            # Chunk fits entirely
            selected.append((orig_idx, chunk))
            total_len += len(chunk) + len(delimiter)
        elif score > 0:
            # Chunk too large but has relevant content - extract relevant parts
            # Find lines containing question words and include surrounding context
            lines = chunk.split('\n')
            relevant_lines = []
            
            # For single-session-user and preference, also mark user statement lines
            user_line_indices = set()
            if is_user_type or is_preference_type:
                for j, line in enumerate(lines):
                    line_lower = line.lower()
                    if any(m in line_lower for m in ['user:', '用户:', 'user：', '用户：']):
                        user_line_indices.add(j)
            
            for j, line in enumerate(lines):
                line_lower = line.lower()
                is_relevant = any(word in line_lower for word in question_words)
                
                # For user-type questions, also include all user statement lines
                if is_user_type and j in user_line_indices:
                    is_relevant = True
                
                # For preference questions, include user statements and preference indicators
                if is_preference_type and j in user_line_indices:
                    is_relevant = True
                    
                # Also check for preference indicators in preference questions
                if is_preference_type:
                    pref_indicators = ['i use', 'i prefer', "i'm using", 'my setup', 'i like', 
                                       'compatible with', 'working with', 'learning', 'interested']
                    if any(ind in line_lower for ind in pref_indicators):
                        is_relevant = True
                
                if is_relevant:
                    # Include this line and some context around it
                    # For user-type and preference, include more context (3 lines vs 2)
                    context_lines = 3 if (is_user_type or is_preference_type) else 2
                    start = max(0, j - context_lines)
                    end = min(len(lines), j + context_lines + 1)
                    for k in range(start, end):
                        if k not in [idx for idx, _ in relevant_lines]:
                            relevant_lines.append((k, lines[k]))
            
            if relevant_lines:
                # Sort by line index to maintain order
                relevant_lines.sort(key=lambda x: x[0])
                extracted = '\n'.join(line for _, line in relevant_lines)
                
                # Truncate if still too long
                if len(extracted) > remaining - 50:
                    extracted = extracted[:remaining - 50] + "\n[...]"
                
                selected.append((orig_idx, extracted))
                total_len += len(extracted) + len(delimiter)
    
    # Sort selected by original position to maintain coherence
    selected.sort(key=lambda x: x[0])
    
    # Join selected chunks
    result = delimiter.join(chunk for _, chunk in selected)
    
    if len(result) < len(context):
        result += f"\n[... {len(chunks) - len(selected)} context chunks omitted for brevity ...]"
    
    return result


def evaluate_preference_match(expected: str, answer: str) -> bool:
    """Evaluate if answer matches preference expectation using key concept extraction.
    
    For preference questions, we need to match KEY CONCEPTS (brands, tools, topics)
    rather than all words. Common words like "prefer", "user", "would" should be ignored.
    
    Args:
        expected: Expected answer (e.g., "The user would prefer resources for Adobe Premiere Pro...")
        answer: LLM's answer
        
    Returns:
        True if key concepts match
    """
    expected_lower = expected.lower()
    answer_lower = answer.lower()
    
    # Common words to ignore (they appear in all preference answers)
    stop_words = {
        'the', 'user', 'would', 'prefer', 'responses', 'that', 'suggest', 'resources',
        'related', 'specifically', 'tailored', 'might', 'not', 'other', 'general',
        'they', 'may', 'options', 'those', 'which', 'with', 'for', 'and', 'or',
        'be', 'interested', 'in', 'their', 'about', 'some', 'are', 'from', 'to',
        'of', 'a', 'an', 'on', 'as', 'such', 'more', 'any', 'all', 'also',
        'can', 'could', 'focus', 'especially', 'particularly', 'based', 'topics',
        'unrelated', 'involve', 'involving', 'recent', 'suggestions', 'equipment',
        'gear', 'quality', 'high', 'low', 'brands', 'brand', 'type', 'types'
    }
    
    # Extract key concepts: longer words that are likely proper nouns or technical terms
    # Also look for multi-word phrases (capitalized sequences in original)
    import re
    
    # Find potential proper nouns and technical terms (3+ chars, not stop words)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]+\b', expected)
    key_concepts = []
    
    for word in words:
        word_lower = word.lower()
        # Skip stop words and very short words
        if word_lower in stop_words or len(word) < 3:
            continue
        # Technical terms and proper nouns are usually longer or have numbers
        if len(word) >= 4 or any(c.isupper() for c in word[1:]) or any(c.isdigit() for c in word):
            key_concepts.append(word_lower)
    
    # Also extract multi-word proper nouns (e.g., "Adobe Premiere Pro", "Sony A7R IV")
    # Look for sequences of capitalized words in original
    proper_noun_pattern = r'\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)+)\b'
    multi_word_matches = re.findall(proper_noun_pattern, expected)
    for match in multi_word_matches:
        key_concepts.append(match.lower())
    
    # Deduplicate while preserving order
    seen = set()
    unique_concepts = []
    for c in key_concepts:
        if c not in seen:
            seen.add(c)
            unique_concepts.append(c)
    
    if not unique_concepts:
        # Fall back to regular keyword matching
        return expected_lower in answer_lower
    
    # Check how many key concepts are in the answer
    matched = 0
    for concept in unique_concepts:
        if concept in answer_lower:
            matched += 1
    
    # Require at least 40% of key concepts to match (more lenient threshold)
    # For preference questions, matching the main tool/brand is enough
    match_ratio = matched / len(unique_concepts) if unique_concepts else 0
    
    # Also check if answer follows the expected format
    has_prefer_format = 'prefer' in answer_lower or 'would' in answer_lower or 'interest' in answer_lower
    
    # Match if either:
    # 1. 40%+ key concepts match, or
    # 2. At least one key concept matches AND answer uses preference language
    return match_ratio >= 0.4 or (matched >= 1 and has_prefer_format)


def build_qa_prompt(
    question: str,
    context: str,
    question_type_hint: Optional[str] = None,
    is_abstention: bool = False,
    max_context_length: int = 12000
) -> str:
    """Build a question-answering prompt based on question type.
    
    This function uses intelligent context filtering to include the most
    relevant parts of context based on question keywords, rather than
    simple truncation which may discard critical information.
    
    Args:
        question: The question to answer
        context: Retrieved context from memory
        question_type_hint: Optional explicit question type
        is_abstention: Whether this is an abstention question
        max_context_length: Maximum length of context to include (default 12000)
        
    Returns:
        Formatted prompt string
    """
    # Detect question type
    qtype = detect_question_type(question, question_type_hint)
    
    # Get template
    template = QA_PROMPT_TEMPLATES.get(qtype, QA_PROMPT_TEMPLATES['default'])
    
    # Intelligently extract relevant context (not simple truncation)
    # Pass question_type for type-specific extraction optimization
    filtered_context = _extract_relevant_context(
        context, question, max_context_length, question_type=qtype
    )
    
    # Build prompt
    prompt = template.format(
        context=filtered_context,
        question=question
    )
    
    # Add abstention instruction if needed
    if is_abstention:
        prompt += "\n\nIMPORTANT: If the information is not clearly available in the conversation history, respond with 'I don't know' rather than guessing."
    
    return prompt
