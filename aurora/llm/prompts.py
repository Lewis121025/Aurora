from __future__ import annotations

from typing import Any, Dict
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
