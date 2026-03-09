from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.Prompt.meaning_extraction_prompt import (
    MEANING_EXTRACTION_SYSTEM_PROMPT,
    build_meaning_extraction_user_prompt,
)
from aurora.soul.models import BELIEF_ORDER, EventFrame, TRAIT_ORDER, clamp


class MeaningFramePayload(BaseModel):
    trait_evidence: Dict[str, float] = Field(default_factory=dict)
    belief_evidence: Dict[str, float] = Field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.0
    tags: List[str] = Field(default_factory=list)
    threat: float = 0.0
    care: float = 0.0
    control: float = 0.0
    abandonment: float = 0.0
    agency: float = 0.0
    shame: float = 0.0


class MeaningExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> EventFrame:
        raise NotImplementedError


class HeuristicMeaningExtractor(MeaningExtractor):
    def __init__(self) -> None:
        self.groups: List[Dict[str, Any]] = [
            {
                "name": "care",
                "keywords": [
                    "care", "comfort", "support", "listen", "hug", "gentle", "understand",
                    "陪", "安慰", "理解", "照顾", "温柔", "抱", "在乎", "倾听", "支持",
                ],
                "traits": {
                    "trust": 0.45, "vigilance": -0.30, "defensiveness": -0.25,
                    "coherence": 0.22, "openness": 0.15, "attachment": 0.10,
                },
                "beliefs": {
                    "closeness_safe": 0.35, "others_reliable": 0.35, "vulnerability_safe": 0.18,
                },
                "valence": 0.55,
                "arousal": 0.20,
                "care": 0.85,
                "tags": ("care", "warmth"),
            },
            {
                "name": "rejection",
                "keywords": [
                    "ignore", "ignored", "cold", "reject", "rejected", "leave", "left",
                    "不理", "冷淡", "拒绝", "离开", "丢下", "抛弃", "疏远",
                ],
                "traits": {
                    "trust": -0.45, "vigilance": 0.45, "defensiveness": 0.35,
                    "coherence": -0.22, "attachment": 0.22,
                },
                "beliefs": {
                    "closeness_safe": -0.32, "others_reliable": -0.38,
                },
                "valence": -0.58,
                "arousal": 0.42,
                "threat": 0.55,
                "abandonment": 0.78,
                "tags": ("rejection", "loss"),
            },
            {
                "name": "betrayal",
                "keywords": [
                    "betray", "betrayed", "lied", "lie", "hurt", "hurtful", "deceive",
                    "背叛", "骗", "欺骗", "伤害", "利用", "出卖",
                ],
                "traits": {
                    "trust": -0.62, "vigilance": 0.55, "defensiveness": 0.40,
                    "coherence": -0.18, "assertiveness": 0.10,
                },
                "beliefs": {
                    "others_reliable": -0.55, "vulnerability_safe": -0.22,
                },
                "valence": -0.72,
                "arousal": 0.56,
                "threat": 0.80,
                "tags": ("betrayal", "hurt"),
            },
            {
                "name": "control",
                "keywords": [
                    "control", "order", "command", "criticize", "criticized", "shame",
                    "控制", "命令", "指责", "批评", "羞辱", "操纵",
                ],
                "traits": {
                    "autonomy": 0.30, "assertiveness": 0.35, "trust": -0.28,
                    "vigilance": 0.35, "defensiveness": 0.30, "coherence": -0.18,
                },
                "beliefs": {
                    "boundaries_allowed": 0.35, "independence_safe": 0.15, "others_reliable": -0.25,
                },
                "valence": -0.45,
                "arousal": 0.48,
                "control": 0.82,
                "shame": 0.55,
                "tags": ("control", "pressure"),
            },
            {
                "name": "boundary",
                "keywords": [
                    "boundary", "boundaries", "refuse", "no", "decide for myself",
                    "边界", "拒绝", "不再", "自己决定", "自己来", "说不", "停止迎合",
                ],
                "traits": {
                    "autonomy": 0.55, "assertiveness": 0.62, "defensiveness": 0.10,
                    "coherence": 0.10,
                },
                "beliefs": {
                    "boundaries_allowed": 0.62, "independence_safe": 0.30,
                },
                "valence": 0.18,
                "arousal": 0.36,
                "agency": 0.78,
                "tags": ("boundary", "agency"),
            },
            {
                "name": "independence",
                "keywords": [
                    "independent", "alone", "on my own", "self-reliant",
                    "独立", "靠自己", "一个人也可以", "不依赖", "成熟",
                ],
                "traits": {
                    "autonomy": 0.62, "assertiveness": 0.28, "attachment": -0.18,
                    "coherence": 0.18,
                },
                "beliefs": {
                    "independence_safe": 0.60, "boundaries_allowed": 0.22,
                },
                "valence": 0.24,
                "arousal": 0.26,
                "agency": 0.65,
                "tags": ("independence", "growth"),
            },
            {
                "name": "vulnerability",
                "keywords": [
                    "cry", "crying", "afraid", "scared", "fragile", "sad", "tears",
                    "哭", "害怕", "脆弱", "难过", "眼泪", "委屈",
                ],
                "traits": {
                    "attachment": 0.20, "openness": 0.28, "coherence": -0.08,
                    "assertiveness": -0.10,
                },
                "beliefs": {
                    "vulnerability_safe": 0.10,
                },
                "valence": -0.32,
                "arousal": 0.42,
                "tags": ("vulnerability", "pain"),
            },
            {
                "name": "repair",
                "keywords": [
                    "sorry", "apologize", "repair", "reconcile", "forgive",
                    "对不起", "道歉", "和解", "修复", "原谅",
                ],
                "traits": {
                    "trust": 0.25, "vigilance": -0.12, "defensiveness": -0.10,
                    "coherence": 0.16, "openness": 0.12,
                },
                "beliefs": {
                    "others_reliable": 0.16, "closeness_safe": 0.14,
                },
                "valence": 0.28,
                "arousal": 0.18,
                "care": 0.25,
                "tags": ("repair", "reconciliation"),
            },
            {
                "name": "mastery",
                "keywords": [
                    "solve", "build", "create", "ship", "win", "capable", "confident",
                    "做到", "解决", "构建", "创造", "完成", "有能力", "自信",
                ],
                "traits": {
                    "autonomy": 0.22, "assertiveness": 0.18, "coherence": 0.22,
                    "openness": 0.08,
                },
                "beliefs": {
                    "independence_safe": 0.18,
                },
                "valence": 0.30,
                "arousal": 0.22,
                "agency": 0.40,
                "tags": ("mastery", "competence"),
            },
        ]

    def extract(self, text: str) -> EventFrame:
        text_norm = f" {text.lower()} "
        trait_evidence: Dict[str, float] = {key: 0.0 for key in TRAIT_ORDER}
        belief_evidence: Dict[str, float] = {key: 0.0 for key in BELIEF_ORDER}
        tags: List[str] = []
        valence = 0.0
        arousal = 0.0
        threat = 0.0
        care = 0.0
        control = 0.0
        abandonment = 0.0
        agency = 0.0
        shame = 0.0

        for group in self.groups:
            hits = sum(1 for kw in group["keywords"] if kw.lower() in text_norm)
            if hits <= 0:
                continue
            scale = 1.0 + 0.12 * (hits - 1)
            for key, value in group.get("traits", {}).items():
                trait_evidence[key] = trait_evidence.get(key, 0.0) + float(value) * scale
            for key, value in group.get("beliefs", {}).items():
                belief_evidence[key] = belief_evidence.get(key, 0.0) + float(value) * scale
            valence += float(group.get("valence", 0.0)) * scale
            arousal += float(group.get("arousal", 0.0)) * scale
            threat += float(group.get("threat", 0.0)) * scale
            care += float(group.get("care", 0.0)) * scale
            control += float(group.get("control", 0.0)) * scale
            abandonment += float(group.get("abandonment", 0.0)) * scale
            agency += float(group.get("agency", 0.0)) * scale
            shame += float(group.get("shame", 0.0)) * scale
            tags.extend(group.get("tags", []))

        arousal += min(text.count("!") * 0.08 + text.count("！") * 0.08, 0.30)
        arousal += min(text.count("?") * 0.03 + text.count("？") * 0.03, 0.12)
        if any(word in text_norm for word in ["always", "never", "一定", "永远", "彻底", "完全"]):
            arousal += 0.08
        if not tags:
            tags.append("neutral")

        for key in list(trait_evidence.keys()):
            trait_evidence[key] = max(-1.0, min(1.0, trait_evidence[key]))
        for key in list(belief_evidence.keys()):
            belief_evidence[key] = max(-1.0, min(1.0, belief_evidence[key]))

        return EventFrame(
            trait_evidence=trait_evidence,
            belief_evidence=belief_evidence,
            valence=max(-1.0, min(1.0, valence)),
            arousal=clamp(arousal, 0.0, 1.0),
            tags=tuple(sorted(set(tags))),
            threat=clamp(threat),
            care=clamp(care),
            control=clamp(control),
            abandonment=clamp(abandonment),
            agency=clamp(agency),
            shame=clamp(shame),
        )


class LLMMeaningExtractor(MeaningExtractor):
    def __init__(
        self,
        llm: LLMProvider,
        *,
        schema: Type[MeaningFramePayload] = MeaningFramePayload,
        fallback: Optional[MeaningExtractor] = None,
        timeout_s: float = 12.0,
        max_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._schema = schema
        self._fallback = fallback or HeuristicMeaningExtractor()
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def extract(self, text: str) -> EventFrame:
        try:
            payload = self._llm.complete_json(
                system=MEANING_EXTRACTION_SYSTEM_PROMPT,
                user=build_meaning_extraction_user_prompt(text=text),
                schema=self._schema,
                temperature=0.1,
                timeout_s=self._timeout_s,
                max_retries=self._max_retries,
                metadata={"operation": "meaning_extraction"},
            )
        except Exception:
            return self._fallback.extract(text)

        return EventFrame(
            trait_evidence={k: float(v) for k, v in payload.trait_evidence.items()},
            belief_evidence={k: float(v) for k, v in payload.belief_evidence.items()},
            valence=clamp(float(payload.valence), -1.0, 1.0),
            arousal=clamp(float(payload.arousal), 0.0, 1.0),
            tags=tuple(sorted(set(payload.tags))),
            threat=clamp(float(payload.threat)),
            care=clamp(float(payload.care)),
            control=clamp(float(payload.control)),
            abandonment=clamp(float(payload.abandonment)),
            agency=clamp(float(payload.agency)),
            shame=clamp(float(payload.shame)),
        )
