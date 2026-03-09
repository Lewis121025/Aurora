from __future__ import annotations

import json
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
        if name == "PlotExtraction":
            # 朴素提取：将第一句作为动作
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

        if name == "MemoryBriefCompilation":
            def extract_json_block(label: str) -> Dict[str, Any]:
                pattern = rf"{label}:\n(.*?)(?:\n[A-Z_ ]+:\n|\Z)"
                match = re.search(pattern, user, re.DOTALL)
                if not match:
                    return {}
                try:
                    return json.loads(match.group(1).strip())
                except Exception:
                    return {}

            candidate_memory = extract_json_block("CANDIDATE_MEMORY")
            data = {
                "known_facts": candidate_memory.get("known_facts", []),
                "preferences": candidate_memory.get("preferences", []),
                "relationship_state": candidate_memory.get("relationship_state", []),
                "active_narratives": candidate_memory.get("active_narratives", []),
                "temporal_context": candidate_memory.get("temporal_context", []),
                "cautions": candidate_memory.get("cautions", []),
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
