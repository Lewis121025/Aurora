from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .provider import LLMProvider

T = TypeVar("T", bound=BaseModel)


class MockLLM(LLMProvider):
    """用于本地运行/测试的确定性模拟。

    生成满足 schema 的简单输出，无需任何模型调用。
    在生产环境中替换为真实提供者。
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
        max_retries: Optional[int] = None,
    ) -> str:
        """生成简单的模拟文本补全。

        对于基准测试，这提供基本的基于模式的提取
        而不是随机文本。
        """
        if "Current User Message:" in prompt:
            match = re.search(r"Current User Message:\n(.*?)(?:\n\n|$)", prompt, re.DOTALL)
            user_message = match.group(1).strip() if match else ""
            if user_message:
                return f"我听到了你的问题：{user_message}。如果你愿意，我们可以继续把它一起展开。"

        # 用于基准评估的基本答案提取模式
        prompt_lower = prompt.lower()

        # 查找问题模式并尝试提取相关答案
        if "what city" in prompt_lower or "where" in prompt_lower:
            # 提取位置提及
            locations = re.findall(
                r"(?:in|at|to|from|visit(?:ed)?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                prompt
            )
            if locations:
                return locations[-1]  # 返回最近的位置

        if "who" in prompt_lower:
            # 提取人名（简单的大写单词）
            names = re.findall(r"\b([A-Z][a-z]+)\b(?:\s+[a-z]+\s+|\s+said|\s+told)", prompt)
            if names:
                return names[-1]

        if "when" in prompt_lower:
            # 提取时间参考
            times = re.findall(
                r"(?:on|at|in)\s+(\w+\s+\d+|\d+[:\d]*\s*(?:am|pm)?|\w+day)",
                prompt, re.IGNORECASE
            )
            if times:
                return times[-1]

        # 对于摘要或一般问题，返回第一个有意义的句子
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
        max_retries: Optional[int] = None,
    ) -> T:
        name = schema.__name__
        if name == "MeaningFramePayload":
            data = {
                "trait_evidence": {
                    "attachment": 0.0,
                    "autonomy": 0.0,
                    "trust": 0.0,
                    "vigilance": 0.0,
                    "openness": 0.0,
                    "defensiveness": 0.0,
                    "assertiveness": 0.0,
                    "coherence": 0.0,
                },
                "belief_evidence": {
                    "closeness_safe": 0.0,
                    "others_reliable": 0.0,
                    "boundaries_allowed": 0.0,
                    "independence_safe": 0.0,
                    "vulnerability_safe": 0.0,
                },
                "valence": 0.0,
                "arousal": 0.2,
                "tags": ["neutral"],
                "threat": 0.0,
                "care": 0.0,
                "control": 0.0,
                "abandonment": 0.0,
                "agency": 0.0,
                "shame": 0.0,
            }
            return schema.model_validate(data)

        if name == "PersonaAxisPayload":
            data = {
                "axes": [
                    {
                        "name": "curious",
                        "positive_pole": "curious",
                        "negative_pole": "closed",
                        "description": "Interested in exploring new ideas",
                        "positive_examples": ["curious", "explore"],
                        "negative_examples": ["closed"],
                        "weight": 1.0,
                    }
                ]
            }
            return schema.model_validate(data)

        if name == "MeaningFramePayloadV4":
            data = {
                "axis_evidence": {},
                "valence": 0.0,
                "arousal": 0.2,
                "care": 0.0,
                "threat": 0.0,
                "control": 0.0,
                "abandonment": 0.0,
                "agency_signal": 0.0,
                "shame": 0.0,
                "novelty": 0.2,
                "self_relevance": 0.5,
                "tags": ["neutral"],
            }
            return schema.model_validate(data)

        if name == "NarrativeSummaryPayloadV4":
            data = {
                "text": "Aurora is holding a provisional mode while integrating recent signals.",
                "current_mode": "origin",
                "salient_axes": ["coherence", "regulation"],
            }
            return schema.model_validate(data)

        if name == "RepairNarrationPayloadV4":
            data = {
                "text": "She reorders the shock into a shape she can keep living with.",
                "mode": "integrate",
            }
            return schema.model_validate(data)

        if name == "DreamNarrationPayloadV4":
            data = {
                "text": "In the dream, unfinished fragments keep circling until they start to explain one another.",
                "operator": "blend",
            }
            return schema.model_validate(data)

        if name == "ModeLabelPayloadV4":
            data = {"label": "origin"}
            return schema.model_validate(data)

        if name == "AxisMergeJudgementPayload":
            data = {
                "should_merge": False,
                "canonical_name": "",
                "alias_name": "",
                "rationale": "",
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

        if name == "ContradictionJudgement":
            data = {
                "is_contradiction": False,
                "explanation": "",
                "reconciliation_hint": "",
            }
            return schema.model_validate(data)

        # 因果 schema
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

        # 自我叙述 schema
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

        # 一致性 schema
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

        # 默认
        return schema.model_validate({})
