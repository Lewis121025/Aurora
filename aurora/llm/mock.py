from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .provider import LLMProvider

T = TypeVar("T", bound=BaseModel)


class MockLLM(LLMProvider):
    """Deterministic mock used for local runs/tests.

    It produces simplistic outputs that satisfy schemas, without any model calls.
    Replace with a real provider in production.
    """

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate simple mock text completion.
        
        For benchmarks, this provides basic pattern-based extraction
        rather than random text.
        """
        # Basic answer extraction patterns for benchmark evaluation
        prompt_lower = prompt.lower()
        
        # Look for question patterns and try to extract relevant answer
        if "what city" in prompt_lower or "where" in prompt_lower:
            # Extract location mentions
            locations = re.findall(
                r"(?:in|at|to|from|visit(?:ed)?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                prompt
            )
            if locations:
                return locations[-1]  # Return most recent location
        
        if "who" in prompt_lower:
            # Extract person names (simple capitalized words)
            names = re.findall(r"\b([A-Z][a-z]+)\b(?:\s+[a-z]+\s+|\s+said|\s+told)", prompt)
            if names:
                return names[-1]
        
        if "when" in prompt_lower:
            # Extract time references
            times = re.findall(
                r"(?:on|at|in)\s+(\w+\s+\d+|\d+[:\d]*\s*(?:am|pm)?|\w+day)",
                prompt, re.IGNORECASE
            )
            if times:
                return times[-1]
        
        # For summaries or general questions, return first meaningful sentence
        sentences = re.split(r'[.!?]', prompt)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and not sent.startswith(("Question", "Answer", "Context")):
                return sent[:200]
        
        return "Unable to determine answer from context."

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float = 0.2,
        timeout_s: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> T:
        name = schema.__name__
        if name == "PlotExtraction":
            # naive extraction: pull first sentence as action
            m_user = re.search(r"user_message: (.*)", user)
            umsg = m_user.group(1).strip() if m_user else ""
            action = umsg[:120] or "interaction"
            data = {
                "actors": ["user", "agent"],
                "action": action,
                "context": "",
                "outcome": "",
                "goal": "",
                "obstacles": [],
                "decision": "",
                "emotion_valence": 0.0,
                "emotion_arousal": 0.2,
                "claims": [],
            }
            return schema.model_validate(data)

        if name == "StoryUpdate":
            data = {
                "title": "Untitled Story",
                "protagonist": "agent",
                "central_conflict": "",
                "stage": "rising",
                "turning_points": [],
                "resolution": None,
                "moral": None,
                "summary": "",
            }
            return schema.model_validate(data)

        if name == "SelfNarrativeUpdate":
            data = {
                "identity_statement": "I am a helpful assistant.",
                "identity_narrative": "I support the user with reliable help.",
                "capability_narrative": "I can reason, write, and plan; I may be uncertain without data.",
                "relationship_narratives": {},
                "core_beliefs": [],
                "unresolved_tensions": [],
            }
            return schema.model_validate(data)

        if name == "ContradictionJudgement":
            data = {
                "is_contradiction": False,
                "explanation": "",
                "reconciliation_hint": "",
            }
            return schema.model_validate(data)

        # Causal schemas
        if name == "CausalRelation":
            data = {
                "cause_id": "",
                "effect_id": "",
                "direction_confidence": 0.5,
                "strength": 0.5,
                "relation_type": "direct",
                "evidence": "",
                "conditions": [],
                "confounders": [],
            }
            return schema.model_validate(data)

        if name == "CausalChainExtraction":
            data = {
                "relations": [],
                "root_causes": [],
                "final_effects": [],
                "chain_confidence": 0.5,
            }
            return schema.model_validate(data)

        if name == "CounterfactualQuery":
            data = {
                "factual_description": "",
                "counterfactual_antecedent": "",
                "query": "",
                "counterfactual_consequent": "结果可能相似",
                "confidence": 0.5,
                "reasoning": "基于有限信息的推测",
            }
            return schema.model_validate(data)

        # Self-narrative schemas
        if name == "CapabilityAssessment":
            data = {
                "capability_name": "general",
                "description": "一般能力",
                "demonstrated": True,
                "demonstration_evidence": "",
                "limitation_found": False,
                "limitation_evidence": "",
                "applicable_contexts": [],
                "confidence": 0.6,
            }
            return schema.model_validate(data)

        if name == "RelationshipAssessment":
            data = {
                "entity_id": "user",
                "entity_type": "user",
                "interaction_positive": True,
                "trust_signal": 0.1,
                "preferences_observed": {},
                "notes": "",
            }
            return schema.model_validate(data)

        if name == "IdentityReflection":
            data = {
                "identity_summary": "我是一个持续学习的AI助手",
                "strong_capabilities": ["对话", "信息处理"],
                "developing_capabilities": [],
                "values_demonstrated": ["帮助性"],
                "growth_areas": [],
                "tensions": [],
            }
            return schema.model_validate(data)

        # Coherence schemas
        if name == "CoherenceCheck":
            data = {
                "element_a_id": "",
                "element_b_id": "",
                "conflict_type": None,
                "has_conflict": False,
                "conflict_severity": 0.0,
                "explanation": "",
                "resolution_suggestion": "",
                "contextually_compatible": True,
                "compatibility_conditions": "",
            }
            return schema.model_validate(data)

        # default
        return schema.model_validate({})
