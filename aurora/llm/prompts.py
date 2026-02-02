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
    'knowledge-update': """Based on the conversation history, answer with the LATEST/MOST RECENT information.
If information has been updated, use the newer value.

Context:
{context}

Question: {question}

Answer (use the most recent information):""",

    'temporal-reasoning': """Based on the conversation history, answer the question about timing or sequence.
Pay attention to words like "first", "last", "before", "after", "earliest", "latest", "when".

Context:
{context}

Question: {question}

Answer (be specific about time/sequence):""",

    'multi-session': """Based on the conversation history from MULTIPLE SESSIONS below, 
answer the question by AGGREGATING information across sessions.

CRITICAL: This question requires information from MULTIPLE CONVERSATIONS/SESSIONS.
You MUST:
1. Search through ALL sessions in the context
2. Extract EVERY relevant piece of information from each session
3. AGGREGATE the information:
   - For "how many" questions: COUNT all instances across all sessions
   - For "total" questions: SUM all values across all sessions  
   - For "all" questions: LIST all items mentioned across all sessions
4. Provide a COMPLETE answer that includes information from all relevant sessions

IMPORTANT: Do NOT miss information from any session. Count carefully and comprehensively.

Context (from multiple sessions):
{context}

Question: {question}

Answer (aggregate across ALL sessions, be brief, specific, and complete):""",

    'single-session-assistant': """Based on the conversation history below, answer the question concisely.
Focus on extracting the specific information requested.

Context:
{context}

Question: {question}

Answer (be brief, specific, and factual):""",

    'single-session-user': """Based on the conversation history below, answer the question about the user.
Extract the specific factual information requested from the user's statements.

Context:
{context}

Question: {question}

Answer (be brief, specific, and factual based on user statements):""",

    'single-session-preference': """Based on the conversation history below, infer the user's preferences.
IMPORTANT: You need to understand what the user likes, prefers, or is interested in based on their statements.
Then provide a response that aligns with those preferences.

Context:
{context}

Question: {question}

Think step by step:
1. What topics/tools/preferences does the user mention in the context?
2. What would be the most relevant answer based on their interests?

Answer (provide a response tailored to the user's preferences shown in context):""",

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


def _extract_relevant_context(
    context: str,
    question: str,
    max_length: int = 12000,
    delimiter: str = '\n---\n'
) -> str:
    """Extract most relevant parts of context based on question keywords.
    
    This uses a simple but effective strategy:
    1. Split context into chunks (by delimiter)
    2. Score each chunk by keyword overlap with question
    3. Include highest-scoring chunks up to max_length
    
    Args:
        context: Full context string
        question: Question to answer
        max_length: Maximum output length
        delimiter: Chunk delimiter in context
        
    Returns:
        Filtered context with most relevant chunks
    """
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
    
    # Extract keywords from question (simple tokenization)
    question_lower = question.lower()
    # Remove common words
    stop_words = {'what', 'where', 'when', 'how', 'why', 'who', 'which', 
                  'is', 'are', 'was', 'were', 'did', 'do', 'does',
                  'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with',
                  'my', 'your', 'i', 'you', 'me', 'we', 'they', 'it',
                  'have', 'has', 'had', 'be', 'been', 'being',
                  'can', 'could', 'would', 'should', 'will', 'shall'}
    question_words = [w.strip('?.,!') for w in question_lower.split() 
                      if len(w.strip('?.,!')) > 2 and w.strip('?.,!') not in stop_words]
    
    # Score each chunk
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0
        for word in question_words:
            if word in chunk_lower:
                score += chunk_lower.count(word)
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
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(word in line_lower for word in question_words):
                    # Include this line and some context around it
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    for j in range(start, end):
                        if j not in [idx for idx, _ in relevant_lines]:
                            relevant_lines.append((j, lines[j]))
            
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
    filtered_context = _extract_relevant_context(
        context, question, max_context_length
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
