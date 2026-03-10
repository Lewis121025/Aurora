"""
aurora.integrations.llm
大语言模型 (LLM) 集成包。
本包提供了统一的 LLM 访问接口，并实现了对接火山方舟 (Ark)、阿里云百炼 (Bailian) 等国内主流厂商的提供者。
"""

from .provider import LLMProvider
from .schemas import (
    AxisMergeJudgementPayload,
    CapabilityAssessment,
    CausalChainExtraction,
    CausalRelation,
    Claim,
    CoherenceCheck,
    ContradictionJudgement,
    CounterfactualQuery,
    DreamNarrationPayloadV4,
    IdentityReflection,
    MeaningFramePayloadV4,
    ModeLabelPayloadV4,
    NarrativeSummaryPayloadV4,
    PersonaAxisPayload,
    PersonaAxisSpec,
    RelationshipAssessment,
    RepairNarrationPayloadV4,
    SCHEMA_VERSION,
    StoryUpdate,
    ThemeCandidate,
)

__all__ = [
    "AxisMergeJudgementPayload",
    "CapabilityAssessment",
    "CausalChainExtraction",
    "CausalRelation",
    "Claim",
    "CoherenceCheck",
    "ContradictionJudgement",
    "CounterfactualQuery",
    "DreamNarrationPayloadV4",
    "IdentityReflection",
    "LLMProvider",
    "MeaningFramePayloadV4",
    "ModeLabelPayloadV4",
    "NarrativeSummaryPayloadV4",
    "PersonaAxisPayload",
    "PersonaAxisSpec",
    "RelationshipAssessment",
    "RepairNarrationPayloadV4",
    "SCHEMA_VERSION",
    "StoryUpdate",
    "ThemeCandidate",
]
