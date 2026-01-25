from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


SCHEMA_VERSION = "1.0"


class Claim(BaseModel):
    subject: str
    predicate: str
    object: str
    polarity: Literal["positive", "negative"] = "positive"
    certainty: float = Field(ge=0.0, le=1.0, default=0.7)
    # optional qualifiers: time, condition, location
    qualifiers: Dict[str, str] = Field(default_factory=dict)


class PlotExtraction(BaseModel):
    schema_version: str = SCHEMA_VERSION
    actors: List[str] = Field(default_factory=list)
    action: str
    context: str = ""
    outcome: str = ""
    # narrative signals
    goal: str = ""
    obstacles: List[str] = Field(default_factory=list)
    decision: str = ""
    emotion_valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    emotion_arousal: float = Field(ge=0.0, le=1.0, default=0.2)
    # knowledge layer
    claims: List[Claim] = Field(default_factory=list)


class StoryUpdate(BaseModel):
    schema_version: str = SCHEMA_VERSION
    title: str
    protagonist: str = ""
    central_conflict: str = ""
    stage: Literal["setup", "rising", "climax", "falling", "resolution"] = "rising"
    turning_points: List[str] = Field(default_factory=list)
    resolution: Optional[str] = None
    moral: Optional[str] = None
    summary: str = ""


class ThemeCandidate(BaseModel):
    schema_version: str = SCHEMA_VERSION
    name: str
    description: str
    theme_type: Literal["pattern", "lesson", "preference", "causality", "capability", "limitation"]
    falsification_conditions: List[str] = Field(default_factory=list)
    scope: str = "general"  # e.g. "user:123", "agent", "global"
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)


class SelfNarrativeUpdate(BaseModel):
    schema_version: str = SCHEMA_VERSION
    identity_statement: str
    identity_narrative: str
    capability_narrative: str
    relationship_narratives: Dict[str, str] = Field(default_factory=dict)
    core_beliefs: List[str] = Field(default_factory=list)
    unresolved_tensions: List[str] = Field(default_factory=list)


class ContradictionJudgement(BaseModel):
    schema_version: str = SCHEMA_VERSION
    is_contradiction: bool
    explanation: str = ""
    # e.g. "contextual" meaning both can be true under different conditions
    reconciliation_hint: str = ""


# -----------------------------------------------------------------------------
# Causal Inference Schemas
# -----------------------------------------------------------------------------

class CausalRelation(BaseModel):
    """Extracted causal relation between events"""
    schema_version: str = SCHEMA_VERSION
    cause_id: str
    effect_id: str
    
    # Causal direction confidence (0-1)
    direction_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Causal strength (0-1)
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Type of causal relation
    relation_type: Literal["direct", "indirect", "enabling", "preventing"] = "direct"
    
    # Evidence for this relation
    evidence: str = ""
    
    # Conditions under which this relation holds
    conditions: List[str] = Field(default_factory=list)
    
    # Potential confounders
    confounders: List[str] = Field(default_factory=list)


class CausalChainExtraction(BaseModel):
    """Extract causal chain from a sequence of events"""
    schema_version: str = SCHEMA_VERSION
    
    # List of causal relations in the chain
    relations: List[CausalRelation] = Field(default_factory=list)
    
    # Root causes identified
    root_causes: List[str] = Field(default_factory=list)
    
    # Final effects identified
    final_effects: List[str] = Field(default_factory=list)
    
    # Overall chain confidence
    chain_confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class CounterfactualQuery(BaseModel):
    """Counterfactual reasoning query"""
    schema_version: str = SCHEMA_VERSION
    
    # The factual situation
    factual_description: str
    
    # The hypothetical change
    counterfactual_antecedent: str
    
    # What we want to know
    query: str
    
    # Answer
    counterfactual_consequent: str = ""
    
    # Confidence in the answer
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Reasoning chain
    reasoning: str = ""


# -----------------------------------------------------------------------------
# Self-Narrative Schemas
# -----------------------------------------------------------------------------

class CapabilityAssessment(BaseModel):
    """Assessment of a capability from interaction"""
    schema_version: str = SCHEMA_VERSION
    
    capability_name: str
    description: str = ""
    
    # Evidence of capability (positive)
    demonstrated: bool = False
    demonstration_evidence: str = ""
    
    # Evidence of limitation (negative)
    limitation_found: bool = False
    limitation_evidence: str = ""
    
    # Context where this applies
    applicable_contexts: List[str] = Field(default_factory=list)
    
    # Confidence in assessment
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class RelationshipAssessment(BaseModel):
    """Assessment of relationship quality from interaction"""
    schema_version: str = SCHEMA_VERSION
    
    entity_id: str
    entity_type: Literal["user", "system", "concept"] = "user"
    
    # Interaction quality
    interaction_positive: bool = True
    
    # Trust signals
    trust_signal: float = Field(ge=-1.0, le=1.0, default=0.0)
    
    # Learned preferences
    preferences_observed: Dict[str, float] = Field(default_factory=dict)
    
    # Notes
    notes: str = ""


class IdentityReflection(BaseModel):
    """Self-reflection on identity"""
    schema_version: str = SCHEMA_VERSION
    
    # Current identity summary
    identity_summary: str
    
    # Key capabilities
    strong_capabilities: List[str] = Field(default_factory=list)
    developing_capabilities: List[str] = Field(default_factory=list)
    
    # Values demonstrated
    values_demonstrated: List[str] = Field(default_factory=list)
    
    # Growth areas
    growth_areas: List[str] = Field(default_factory=list)
    
    # Tensions or conflicts
    tensions: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Coherence Schemas
# -----------------------------------------------------------------------------

class CoherenceCheck(BaseModel):
    """Result of coherence check between two elements"""
    schema_version: str = SCHEMA_VERSION
    
    element_a_id: str
    element_b_id: str
    
    # Type of potential conflict
    conflict_type: Optional[Literal["factual", "temporal", "causal", "thematic"]] = None
    
    # Is there a conflict?
    has_conflict: bool = False
    conflict_severity: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Explanation
    explanation: str = ""
    
    # Suggested resolution
    resolution_suggestion: str = ""
    
    # Can both be true under different conditions?
    contextually_compatible: bool = True
    compatibility_conditions: str = ""
